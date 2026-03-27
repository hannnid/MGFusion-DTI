# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import numpy as np
import pickle
from tqdm import tqdm

from rdkit import Chem
import torch
from transformers import AutoTokenizer, AutoModel


def masked_mean(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    对 token 维度做带 mask 的平均池化。
    last_hidden: (B, L, H)
    attention_mask: (B, L)
    返回: (B, H)
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)  # (B, L, 1)
    summed = (last_hidden * mask).sum(dim=1)                  # (B, H)
    counts = mask.sum(dim=1).clamp(min=1e-6)                  # (B, 1)
    return summed / counts


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default="BioSNAP",
                       choices=['BindingDB_AIBind', 'BioSNAP', 'BindingDB'],
                       help='the scenario of experiment setting')

    # 选择 HuggingFace 模型 & 池化方式 & batch 大小
    parse.add_argument(
        '--hf_model',
        type=str,
        default="HUBioDataLab/SELFormer",
        choices=[
            # SELFIES
            "HUBioDataLab/SELFormer"  # SELFormer  768
            "ibm-research/materials.selfies-ted"  # SELFIES-TED 1024
        ],
        help='HuggingFace model name or local path for SMILES LM'
    )
    parse.add_argument('--pooling', type=str, default="mean", choices=['mean', 'cls'], help='pool token embeddings into molecule-level vector')
    parse.add_argument('--batch_size', type=int, default=32, help='batch size for embedding')
    parse.add_argument('--max_length', type=int, default=512, help='max token length with truncation')
    parse.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='device: cuda or cpu')

    # 输出文件名（可与原 Mol2Vec 名称区分，避免下游混淆）
    parse.add_argument('--desc_out', type=str, default='selfies_SELFormer768_Mol.pkl',
                       help='output pickle for per-molecule vectors')
    parse.add_argument('--seq_out', type=str, default='selfies_SELFormer768_Atom.pkl',
                       help='output pickle for per-token matrices')

    opt = parse.parse_args()
    Dataset = opt.dataset

    # 路径 & 读取 SMILES 列表
    path = "./../../Datasets/{}/feature/".format(Dataset)
    drug_file = os.path.join(path, "drug_list_selfies.txt")
    assert os.path.exists(drug_file), f"Not found: {drug_file}"

    Drugs = {}
    with open(drug_file, "r") as f:
        for line in f:
            line = line.strip().split()
            if len(line) < 2:
                continue
            Drugs[line[0]] = line[1]

    # 加载 HuggingFace 模型
    tokenizer = AutoTokenizer.from_pretrained(opt.hf_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(opt.hf_model, trust_remote_code=True).to(opt.device)
    model.eval()

    # 结果容器
    drug_descriptor = {}  # 分子级向量: (hidden_dim,)
    drug_matrix = {}      # token级矩阵: (seq_len, hidden_dim)

    # 预处理可用列表（直接使用 SELFIES，无规整化）
    valid_items = [(drug_id, selfies_seq) for drug_id, selfies_seq in Drugs.items()]

    # 批处理编码
    bs = opt.batch_size
    for i in tqdm(range(0, len(valid_items), bs), total=(len(valid_items) + bs - 1) // bs):
        batch = valid_items[i:i+bs]
        drug_ids = [d for d, _ in batch]
        smiles_list = [s for _, s in batch]

        with torch.no_grad():
            enc = tokenizer(
                smiles_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=opt.max_length
            )
            # 删除不兼容的 token_type_ids
            if "token_type_ids" in enc:
                del enc["token_type_ids"]

            enc = {k: v.to(opt.device) for k, v in enc.items()}
            outputs = model(**enc)
            # 通常取最后一层隐状态
            last_hidden = outputs.last_hidden_state  # (B, L, H)

            # 分子级向量
            if opt.pooling == "cls" and tokenizer.cls_token_id is not None:
                desc = last_hidden[:, 0, :]  # (B, H)
            else:
                # 默认 mean pooling（带 attention mask）
                desc = masked_mean(last_hidden, enc["attention_mask"])  # (B, H)

            # 逐条保存到 dict
            for b_idx, drug_id in enumerate(drug_ids):
                # 去掉 padding 的 token（根据 attention_mask）
                attn = enc["attention_mask"][b_idx]  # (L,)
                L = int(attn.sum().item())
                per_token = last_hidden[b_idx, :L, :].detach().cpu().numpy()  # (seq_len, H)
                per_mol = desc[b_idx].detach().cpu().numpy()                  # (H,)

                drug_matrix[drug_id] = per_token
                drug_descriptor[drug_id] = per_mol

    # 保存
    desc_out_path = os.path.join(path, opt.desc_out)
    seq_out_path = os.path.join(path, opt.seq_out)

    with open(desc_out_path, 'wb') as f:
        pickle.dump(drug_descriptor, f)

    with open(seq_out_path, 'wb') as f:
        pickle.dump(drug_matrix, f)

    # 加载并打印信息（与原脚本风格一致）
    with open(desc_out_path, 'rb') as f:
        drug_descriptor_loaded = pickle.load(f)

    print("\n=== {} ===".format(opt.desc_out))
    print("类型:", type(drug_descriptor_loaded))
    print("总长度（药物数量）:", len(drug_descriptor_loaded))
    for k, v in list(drug_descriptor_loaded.items())[:1]:
        print("示例药物ID:", k)
        print("向量类型:", type(v))
        print("向量维度:", v.shape)

    with open(seq_out_path, 'rb') as f:
        drug_matrix_loaded = pickle.load(f)

    print("\n=== {} ===".format(opt.seq_out))
    print("类型:", type(drug_matrix_loaded))
    print("总长度（药物数量）:", len(drug_matrix_loaded))
    for k, v in list(drug_matrix_loaded.items())[:1]:
        print("示例药物ID:", k)
        print("token矩阵类型:", type(v))
        print("token矩阵 shape (seq_len, hidden_dim):", v.shape)


if __name__ == "__main__":
    main()
