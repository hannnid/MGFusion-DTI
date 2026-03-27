import os
import sys
import math
import pickle
import argparse
from typing import List, Tuple, Dict

import numpy as np
import torch
from tqdm import tqdm
from transformers import EsmTokenizer, EsmForMaskedLM  # SaProt/ESM family


# 这个版本较SaProt.py 去除了两个特殊的token[CLS][SEP]（因为在pooling时用到的是masked_mean_pool()，它不会依赖 [CLS]）
# 去掉之后发现效果并不好，又使用了。这时又尝试了不同的pooling方式，只有mean是最好 遂 用SaPort.py

def load_id_seq_list(txt_path: str) -> List[Tuple[str, str]]:
    """读取每行 `ID<空格>氨基酸序列` 的 txt。"""
    pairs = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)  # 仅分割一次
            if len(parts) != 2:
                raise ValueError(f"[{txt_path}:{ln}] 需为 'ID<空格>序列'：{line}")
            _id, seq = parts[0], parts[1].strip()
            if not _id or not seq:
                raise ValueError(f"[{txt_path}:{ln}] ID 或 序列为空：{line}")
            pairs.append((_id, seq))
    return pairs

def masked_mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """mask-mean 池化得到序列整体向量。"""
    # last_hidden_state: [B, L, D]; attention_mask: [B, L] (bool/0-1)
    mask = attention_mask.float().unsqueeze(-1)  # [B, L, 1]
    summed = (last_hidden_state * mask).sum(dim=1)   # [B, D]
    denom = mask.sum(dim=1).clamp(min=1e-6)         # [B, 1]
    return summed / denom                           # [B, D]

@torch.no_grad()
def encode_proteins_to_dict(
    id_seq_pairs: List[Tuple[str, str]],
    prot_encoder_path: str = "westlake-repl/SaProt_650M_AF2",
    device: str = None,
    batch_size: int = 16,
    max_length: int = 512,
    fp16_matrix: bool = False,
    pooling_type: str = "mean",
) -> Dict[str, list]:
    """
    返回：{ID: [vector(np.float32[D]), matrix(np.float{16/32}[L,D]), mask(np.bool_[L])]}
    其中 mask=True 表示有效 token，False 表示 padding。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = EsmTokenizer.from_pretrained(prot_encoder_path)
    model = EsmForMaskedLM.from_pretrained(prot_encoder_path)
    model.eval().to(device)

    out: Dict[str, list] = {}
    n = len(id_seq_pairs)
    steps = math.ceil(n / batch_size)

    for step in tqdm(range(steps), desc="Encoding proteins (SaProt)"):
        chunk = id_seq_pairs[step * batch_size : (step + 1) * batch_size]
        ids = [x[0] for x in chunk]
        seqs = [x[1] for x in chunk]

        enc = tokenizer.batch_encode_plus(
            seqs,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,  # False 去除两个特殊的token
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)           # [B, L]
        attention_mask = enc["attention_mask"].to(device) # [B, L]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1]           # [B, L, D]

        # 屏蔽无效 padding 区域（注意力 mask 为 False 的位置）
        # last_hidden = last_hidden * attention_mask.unsqueeze(-1)

        if pooling_type == "mean":
            pooled = masked_mean_pool(last_hidden, attention_mask)  # [B, D]
        elif pooling_type == "max":
            masked_hidden = last_hidden.masked_fill(~attention_mask.unsqueeze(-1).bool(), float("-inf"))
            pooled = masked_hidden.max(dim=1).values  # [B, D]
        elif pooling_type == "cls":
            pooled = last_hidden[:, 0, :]  # [B, D]
        else:
            raise ValueError(f"Unsupported pooling_type: {pooling_type}")

        # 回到 CPU / numpy
        last_hidden = last_hidden.detach().cpu()
        pooled = pooled.detach().cpu().to(torch.float32)  # vector 保持 float32
        attn = attention_mask.detach().cpu().to(torch.bool)

        # matrix 可选半精度降体积
        if fp16_matrix:
            last_hidden = last_hidden.half()

        for i, _id in enumerate(ids):
            vec_np = pooled[i].numpy()                    # (D,)
            mat_np = last_hidden[i].numpy()               # (L, D)  L=max_length
            mask_np = attn[i].numpy()                     # (L,)    True=有效
            out[_id] = [vec_np, mat_np, mask_np]

    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Human",
                        choices=["DrugBank", "BioSNAP", "Human"])
    parser.add_argument("--protein_txt", type=str, default=None,
                        help="protein_list.txt 路径（每行：ID 序列）。若不提供，将按 dataset 推断默认路径。")
    parser.add_argument("--output_pkl", type=str, default=None,
                        help="输出 pkl 路径。若不提供，将按 dataset 推断默认路径。")
    parser.add_argument("--prot_encoder_path", type=str, default="westlake-repl/SaProt_650M_AF2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--fp16_matrix", action="store_true", help="将 matrix 以 float16 存盘以节省空间")
    parser.add_argument("--device", type=str, default=None, help="cuda 或 cpu；默认自动选择")
    parser.add_argument("--pooling_type", type=str, default="mean", choices=["mean", "max", "cls"],
                        help="池化方式：mean（默认）、max、cls")
    args = parser.parse_args()

    # 默认路径推断
    base = f"./../../Datasets/{args.dataset}/feature"
    protein_txt = args.protein_txt or os.path.join(base, "protein_list_sa.txt")
    output_pkl = args.output_pkl or os.path.join(base, "proteins_embed_sa1280.pkl")
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)

    # 加载列表并编码
    pairs = load_id_seq_list(protein_txt)
    data = encode_proteins_to_dict(
        id_seq_pairs=pairs,
        prot_encoder_path=args.prot_encoder_path,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        fp16_matrix=args.fp16_matrix,
        pooling_type=args.pooling_type,
    )

    # 存盘
    with open(output_pkl, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 打印一个样例形状
    k0 = next(iter(data.keys()))
    v0, m0, mk0 = data[k0]
    print(f"Saved: {output_pkl}")
    print(f"Example -> ID={k0}, vector={v0.shape}, matrix={m0.shape}, mask={mk0.shape}, "
          f"dtype(matrix)={m0.dtype}, mask_true_count={mk0.sum()}")

if __name__ == "__main__":
    main()
