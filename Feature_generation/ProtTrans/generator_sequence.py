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
                       choices=['DrugBank', 'BioSNAP', 'BindingDB'],
                       help='the scenario of experiment setting')
    opt = parse.parse_args()
    Dataset = opt.dataset
    protein_embed_dict = {}
    embedder = ProtTransBertBFDEmbedder()
    path = f"./../../Datasets/{Dataset}/feature/"
    with open(path + "protein_list.txt") as file:
        lines = file.readlines()
        lines = tqdm(lines, total=len(lines))
        for line in lines:
            pid, aas = line.strip().split()  # pid蛋白质编号 aas氨基酸序列
            matrix = np.array(embedder.embed(aas))        # (L, 1024)
            vector = np.array(embedder.reduce_per_protein(matrix))  # (1024,)
            protein_embed_dict[pid] = vector  # 只保存整体向量

    save_file = path + 'aas_ProtTransBertBFD_sequence1024.pkl'
    with open(save_file, 'wb') as f:
        pickle.dump(protein_embed_dict, f)

    # -----------------加载并打印保存结果---------------------- #
    with open(save_file, 'rb') as f:
        loaded = pickle.load(f)

    print("\n=== aas_ProtTransBertBFD_sequence1024.pkl ===")
    print("类型:", type(loaded))
    print("总蛋白数量:", len(loaded))

    for k, v in list(loaded.items())[:1]:
        print("蛋白ID:", k)
        print("整体向量 shape:", v.shape)  # 应为 (1024,)