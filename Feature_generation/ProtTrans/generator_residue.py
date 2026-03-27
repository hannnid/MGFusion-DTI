import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pickle
from tqdm import tqdm
from bio_embeddings.embed import ProtTransBertBFDEmbedder
import argparse


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default="BioSNAP",
                       choices=['BindingDB_AIBind', 'BioSNAP', 'BindingDB'],
                       help='the dataset name')
    opt = parse.parse_args()
    Dataset = opt.dataset

    # 存放蛋白质的embedding
    protein_embed_dict = {}

    # 加载 ProtTransBERT-BFD 模型
    embedder = ProtTransBertBFDEmbedder()

    # 路径
    path = f"./../../Datasets/{Dataset}/feature/"
    with open(path + "protein_list.txt") as file:
        lines = file.readlines()
        lines = tqdm(lines, total=len(lines))

        for line in lines:
            pid, aas = line.strip().split()  # pid: 蛋白质编号, aas: 氨基酸序列
            # 逐残基编码, 得到 shape = (L,1024)
            matrix = np.array(embedder.embed(aas))
            # 只保存逐残基矩阵
            protein_embed_dict[pid] = matrix

    # 保存 pkl
    save_file = path + 'aas_ProtTransBertBFD_residue1024.pkl'
    with open(save_file, 'wb') as f:
        pickle.dump(protein_embed_dict, f)

    # -----------------加载并打印信息---------------------- #
    with open(save_file, 'rb') as f:
        loaded = pickle.load(f)

    print("\n=== aas_ProtTransBertBFD_residue1024.pkl ===")
    print("类型:", type(loaded))
    print("总蛋白数量:", len(loaded))

    # 打印第一个蛋白的信息
    for k, v in list(loaded.items())[:1]:
        print("蛋白ID:", k)
        print("残基矩阵 shape:", v.shape)  # (L, 1024)