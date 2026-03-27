import os
import math
import pickle
import argparse
from typing import List, Tuple, Dict

import numpy as np
import torch
from tqdm import tqdm
from transformers import EsmTokenizer, EsmForMaskedLM



def load_id_seq_list(path: str) -> List[Tuple[str, str]]:
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            parts = line.strip().split(None, 1)
            if len(parts) != 2:
                raise ValueError(f"[Line {ln}] Expected 'ID SEQUENCE', got: {line}")
            pairs.append((parts[0], parts[1]))
    return pairs


@torch.no_grad()
def encode_proteins(
    pairs: List[Tuple[str, str]],
    model_name: str,
    device: str,
    batch_size: int,
    max_len: int,
    fp16: bool
) -> Dict[str, list]:
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name).to(device).eval()

    result = {}
    for i in tqdm(range(0, len(pairs), batch_size), desc="Encoding"):
        batch = pairs[i:i+batch_size]
        ids, seqs = zip(*batch)

        enc = tokenizer.batch_encode_plus(
            list(seqs),
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        hidden = out.hidden_states[-1]  # [B, L, D]
        pooled = (hidden * attn_mask.unsqueeze(-1)).sum(1) / attn_mask.sum(1, keepdim=True).clamp(min=1e-6)

        hidden = hidden.cpu().half() if fp16 else hidden.cpu()
        pooled = pooled.cpu().float()
        attn_mask = attn_mask.cpu().bool() # 保留mask

        for idx, pid in enumerate(ids):
            result[pid] = [
                pooled[idx].numpy(),  # vector
                hidden[idx].numpy(),  # matrix
                attn_mask[idx].numpy()  # 保留mask
            ]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Human",
                        choices=["DrugBank", "BioSNAP", "Human"])
    parser.add_argument("--protein_txt", type=str, default=None)
    parser.add_argument("--output_pkl", type=str, default=None)
    parser.add_argument("--prot_encoder_path", type=str, default="westlake-repl/SaProt_650M_AF2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--fp16_matrix", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    base = f"./../../Datasets/{args.dataset}/feature"
    txt = args.protein_txt or os.path.join(base, "protein_list_sa.txt")
    out_pkl = args.output_pkl or os.path.join(base, "sa_SaProt1280.pkl")
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pairs = load_id_seq_list(txt)
    data = encode_proteins(pairs, args.prot_encoder_path, device, args.batch_size, args.max_length, args.fp16_matrix)

    with open(out_pkl, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # k0 = next(iter(data))
    # v0, m0 = data[k0]
    # print(f"Saved to {out_pkl}")
    # print(f"Example -> ID={k0}, vector={v0.shape}, matrix={m0.shape}, dtype={m0.dtype}")
    k0 = next(iter(data))
    v0, m0, mk0 = data[k0]
    print(f"Saved to {out_pkl}")
    print(f"Example -> ID={k0}, vector={v0.shape}, matrix={m0.shape}, mask={mk0.shape}, dtype={m0.dtype}, mask_true={mk0.sum()}")


if __name__ == "__main__":
    main()
