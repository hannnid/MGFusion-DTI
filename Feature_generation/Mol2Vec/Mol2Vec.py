import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import sys, os
from rdkit import Chem
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec, Atom2Substructure
from gensim.models import word2vec
import copy,pickle
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset', type=str, default="Human",
                       choices=['DrugBank', 'BioSNAP', 'BindingDB','Human'],
                       help='the scenario of experiment setting')
    opt = parse.parse_args()
    Dataset = opt.dataset

    path = "./../../Datasets/{}/feature/".format(Dataset)
    Drugs ={}
    # 读取药物SMILES信息---->文件格式为：DrugID SMILES；解析为字典：Drugs={drug_id:smiles}
    with open(path+"drug_list.txt","r") as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split()
            Drugs[line[0]] = line[1]

    # 加载预训练好的MolVec模型（300维）; key是所有出现在词表中的子结构token，UNK表示unknown structure 其向量为unseen_vec
    model = word2vec.Word2Vec.load('./model_300dim.pkl')
    keys = set(model.wv.vocab.keys())
    unseen='UNK'
    unseen_vec = model.wv.word_vec(unseen)
    '''
    # 特征提取
    drug_descriptor，类型:字典，数量:对应药物数量；每个元素的维度：300
    存储了每个药物SMILES序列对应的embedding
    drug_matrix，类型:字典，数量:对应药物数量；每个元素的维度：(num_atom,300)
    (num_atom,300)中的num_atom代表了当前药物存在的片段/原子序列，即drug_matrix中包含了更细致的局部信息
    '''
    drug_descriptor= {}  # 分子级特征vector（1个向量）
    drug_matrix= {}  # 原子级特征matrix（多个原子子结构构成的矩阵）
    max_num = 0
    for drug_id in tqdm(Drugs.keys(),total=len(Drugs)):
        try:
            mol = Chem.MolFromSmiles(Drugs[drug_id])  # RDKit解析
            if mol != None:
                sentence = MolSentence(mol2alt_sentence(mol, 1))
                matrix = Atom2Substructure(mol, 1, model, keys, unseen_vec)
                if matrix.shape[0]>1:
                    vector = sum([model.wv.word_vec(y) for y in sentence
                                  if y in set(sentence) & keys])
                else:
                    vector = matrix[0]
                    print("{} is too small\n".format(drug_id))  # 子结构太少
                if type(vector)==int:
                    print("{} has no feature\n".format(drug_id))
                drug_descriptor[drug_id] = vector
                drug_matrix[drug_id] = matrix
            else:
                print("RDKit don't read {}".format(drug_id))
        except Exception as e:
            print(drug_id, e)
    with open(path+'smiles_Mol2Vec300.pkl', 'wb') as f:
        pickle.dump(drug_descriptor, f)
    with open(path+'smiles_Atom2Vec300.pkl', 'wb') as f:
        pickle.dump(drug_matrix, f)

    '''
        smiles_Mol2Vec300.pkl的数据形式：
        字典:[ID,vector] shape of vector:300 
        smiles_Atom2Vec300.pkl的数据形式：
        字典:[ID,matrix] shape of matrix:(num_atom,300)
    '''


# -----------------加载并打印 compound_Atom2Vec300.pkl/'compound_Atom2Vec300.pkl 信息---------------------- #
    # 加载并打印 smiles_Mol2Vec300.pkl 信息
    with open(path + 'smiles_Mol2Vec300.pkl', 'rb') as f:
        drug_descriptor_loaded = pickle.load(f)

    print("\n=== smiles_Mol2Vec300.pkl ===")
    print("类型:", type(drug_descriptor_loaded))
    print("总长度（药物数量）:", len(drug_descriptor_loaded))
    for k, v in list(drug_descriptor_loaded.items())[:1]:  # 打印第一个样本结构
        print("示例药物ID:", k)
        print("向量类型:", type(v))
        print("向量维度:", v.shape)

    # 加载并打印 smiles_Atom2Vec300.pkl 信息
    with open(path + 'smiles_Atom2Vec300.pkl', 'rb') as f:
        drug_matrix_loaded = pickle.load(f)

    print("\n=== smiles_Atom2Vec300.pkl ===")
    print("类型:", type(drug_matrix_loaded))
    print("总长度（药物数量）:", len(drug_matrix_loaded))
    for k, v in list(drug_matrix_loaded.items())[:1]:  # 打印第一个样本结构
        print("示例药物ID:", k)
        print("原子矩阵类型:", type(v))
        print("原子矩阵 shape (原子数, 维度):", v.shape)

