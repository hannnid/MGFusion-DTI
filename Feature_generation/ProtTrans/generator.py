import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pickle
from tqdm import tqdm
from bio_embeddings.embed import SeqVecEmbedder, ProtTransBertBFDEmbedder
import argparse


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default="Human",
                       choices=['DrugBank', 'BioSNAP', 'BindingDB', 'Human'],
                       help='the scenario of experiment setting')
    opt = parse.parse_args()
    Dataset = opt.dataset
    protein_embed_dict = {}
    embedder = ProtTransBertBFDEmbedder()
    path = "./../../Datasets/{}/feature/".format(Dataset)
    with open(path + "./protein_list.txt") as file:
        lines = file.readlines()
        lines = tqdm(lines, total=len(lines))
        for line in lines:
            pid, aas = line.strip().split()  # pid蛋白质编号 aas氨基酸序列
            matrix = np.array(embedder.embed(aas))  # 对整个序列编码 得到shape=(L,1024)的矩阵
            vector = np.array(embedder.reduce_per_protein(matrix))  # 将矩阵压缩为shape=(1024,)的向量
            protein_embed_dict[pid] = [vector, matrix]  # 存为列表结构，包含两部分：整体向量(1024维)，每个残基向量矩阵(L,1024)
    with open(path+'aas_ProtTransBertBFD1024.pkl', 'wb') as f:
        pickle.dump(protein_embed_dict, f)

    # -----------------加载并打印 aas_ProtTransBertBFD1024.pkl 信息---------------------- #
    '''
    aas_ProtTransBertBFD1024.pkl最终的数据结构：
    字典结构：{ID:[vector,matrix]}  shape: vector:(1024) matrix:(L,1024)
    vector：此条氨基酸序列的整体向量，长度为1024
    matrix：氨基酸序列中每个残基也被编码成1024维的向量，共L*1024
    '''
    with open(path + 'aas_ProtTransBertBFD1024.pkl', 'rb') as f:
        loaded = pickle.load(f)

    print("\n=== aas_ProtTransBertBFD1024.pkl ===")
    print("类型:", type(loaded))
    print("总蛋白数量:", len(loaded))

    # 打印第一个蛋白的结构
    for k, v in list(loaded.items())[:1]:
        print("蛋白ID:", k)
        print("整体向量 shape:", v[0].shape)  # 应为 (1024,)
        print("残基矩阵 shape:", v[1].shape)  # 应为 (L, 1024)
