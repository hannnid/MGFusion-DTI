import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
import os,pickle
class CustomDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, item):
        return self.pairs[item]

    def __len__(self):
        return len(self.pairs)

# 多返回了口袋信息
class collater_embeding:
    """
    并行版 collate：
    - 完全保留 ColdstartCPI 原有 6 个张量（不改内涵、不改语义）。
    - 额外返回 p_g_tensor_sa, p_m_tensor_sa, p_masks_sa 三个张量，来自你新 pkl（[vector, matrix, mask]）。
    - 新 pkl 的 mask 是 attention 语义(True=有效)，这里在 collate 时转换为 ColdstartCPI 语义(True=padding)。
    """

    def __init__(self, drug_f, drug_m, protein_m_sa, protein_m_pocket,   # 新增蛋白质口袋矩阵 {prot_id: matrix(L,1024)}
                 d_max=64, p_max=512, p_max_sa=512):
        # 原有三个数据源（不动）指导

        self.drug_f = drug_f    # {drug_id: vector(300,)}
        self.drug_m = drug_m    # {drug_id: matrix(num_atom,300)}
        # self.protein_m = protein_m  # {prot_id: [vector(1024,), matrix(L,1024)]} 变长

        # 新蛋白侧（SaProt）
        self.protein_m_sa = protein_m_sa    # {prot_id: [vector(D_sa,), matrix(L_src,D_sa), mask(L_src,)]}
        self.protein_m_pocket = protein_m_pocket  # 新增蛋白质口袋矩阵 {prot_id: matrix(L,1024)}
        self.d_max = d_max
        self.p_max = p_max
        self.p_max_sa = p_max_sa

    def _pack_pad_old(self, mat_np, max_len):
        """
        并生成 mask（False=padding, True=有效）。
        """
        L, D = mat_np.shape
        out = np.zeros((max_len, D), dtype=np.float32)
        mask = np.ones((max_len,), dtype=np.bool_)  # True=有效, False=padding
        if L <= max_len:
            out[:L] = mat_np
            mask[L:] = False
        else:
            out[:] = mat_np[:max_len]
            # 截断后没有 padding（全有效，mask 全 True）
        return out, mask

    def _pack_pad_from_attn(self, mat_np, attn_mask_np, max_len):
        """
        新逻辑：输入矩阵 + attention_mask(True=有效) -> [max_len,D] + mask (True=有效, False=padding)
        使用有效长度 eff_len = sum(attn_mask)
        拷贝 mat_np[:eff_len] 到前面，其余补 0；mask = [True]*eff_len + [False]*(max_len-eff_len)
        """
        assert mat_np.ndim == 2 and attn_mask_np.ndim == 1
        L_src, D = mat_np.shape
        eff_len = int(attn_mask_np.sum())  # 有效 token 数
        eff_len = max(0, min(eff_len, max_len, L_src))

        out = np.zeros((max_len, D), dtype=np.float32)
        pad_mask = np.zeros((max_len,), dtype=np.bool_)  # True=有效, False=padding
        if eff_len > 0:
            out[:eff_len] = mat_np[:eff_len]
            pad_mask[:eff_len] = True
        # pad_mask[eff_len:] remains False (padding)
        return out, pad_mask

    def __call__(self, batch_data):
        """
        batch_data: List of samples, where each sample is (drug_id, prot_id, label)
        返回：
          [d_g_tensor, d_m_tensor, p_g_tensor, p_m_tensor, d_masks, p_masks,
           p_g_tensor_sa, p_m_tensor_sa, p_masks_sa, p_m_tensor_pocket, p_masks_pocket], labels_tensor
        """
        batch_size = len(batch_data)

        # 药物 蛋白质sa
        d_g_tensor, d_m_tensor, d_masks = [], [], []
        p_g_tensor_sa, p_m_tensor_sa, p_masks_sa = [], [], []
        # 蛋白质口袋输出
        p_m_tensor_pocket, p_masks_pocket = [], []
        labels_tensor = torch.zeros(batch_size, dtype=torch.long)

        for i, sample in enumerate(batch_data):
            d_id, p_id, label = sample

            # drug 全局向量  matrix & mask（按 d_max
            d_g_tensor.append(self.drug_f[d_id])  # shape (300,)
            drug_mat_np = self.drug_m[d_id]       # (num_atom, 300) 变长
            drug_mat_pad, d_mask_pad = self._pack_pad_old(drug_mat_np, self.d_max)
            d_m_tensor.append(drug_mat_pad)
            d_masks.append(d_mask_pad)

            # 蛋白质SaProt
            if self.protein_m_sa is not None and p_id in self.protein_m_sa:
                vec_sa, mat_sa, attn_sa = self.protein_m_sa[p_id]  # [vector, matrix, mask(attn True=有效)]
                # vector 原样
                p_g_tensor_sa.append(vec_sa.astype(np.float32))

                # matrix + 将 attention_mask 转为 ColdstartCPI 语义 padding_mask
                # 兼容 mat_sa 可能已是定长/变长：统一通过 eff_len 裁剪
                mat_pad_sa, pad_mask_sa = self._pack_pad_from_attn(mat_sa, attn_sa.astype(bool), self.p_max_sa)
                p_m_tensor_sa.append(mat_pad_sa)
                p_masks_sa.append(pad_mask_sa)
            else:
                raise KeyError(f"protein_m_sa 缺少 {p_id} 的条目，请保证新 pkl 覆盖到该 ID")

            # 蛋白质sa口袋特征
            if self.protein_m_pocket is not None and p_id in self.protein_m_pocket:
                prot_mat_pocket = self.protein_m_pocket[p_id]  # (L_pocket,1024)
                prot_mat_pocket_pad, p_mask_pocket_pad = self._pack_pad_old(prot_mat_pocket, self.p_max)
                p_m_tensor_pocket.append(prot_mat_pocket_pad)
                p_masks_pocket.append(p_mask_pocket_pad)
            else:
                # fill zeros if missing
                p_m_tensor_pocket.append(np.zeros((self.p_max, 1280), dtype=np.float32))
                p_masks_pocket.append(np.ones((self.p_max,), dtype=np.bool_))

            labels_tensor[i] = int(float(label))

        # ====== 拼成张量 ======
        d_g_tensor = torch.from_numpy(np.asarray(d_g_tensor)).float()      # [B, 300]
        d_m_tensor = torch.from_numpy(np.asarray(d_m_tensor)).float()      # [B, d_max, 300]
        d_masks    = torch.from_numpy(np.asarray(d_masks))                  # [B, d_max]  (bool)

        p_g_tensor_sa = torch.from_numpy(np.asarray(p_g_tensor_sa)).float()        # [B, D_sa]
        p_m_tensor_sa = torch.from_numpy(np.asarray(p_m_tensor_sa)).float()        # [B, p_max_sa, D_sa]
        p_masks_sa = torch.from_numpy(np.asarray(p_masks_sa))                    # [B, p_max_sa] (bool)

        # sa蛋白质口袋张量
        p_m_tensor_pocket = torch.from_numpy(np.asarray(p_m_tensor_pocket)).float()
        p_masks_pocket = torch.from_numpy(np.asarray(p_masks_pocket))

        return [d_g_tensor, d_m_tensor, p_g_tensor_sa, p_m_tensor_sa, p_m_tensor_pocket,
                d_masks, p_masks_sa, p_masks_pocket], labels_tensor

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def split_train_valid(data_df, fold, val_ratio=0.1):
    cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)
    train_index, val_index = next(iter(cv_split.split(X=range(len(data_df)), y=data_df['label'])))

    train_df = data_df.iloc[train_index]
    val_df = data_df.iloc[val_index]

    return train_df, val_df

def load_scenario_dataset(DATASET,scenarios,fold, batch_size):
    drug_without_feature = []
    with open("./../../Datasets/{}/drug_without_feature.txt".format(DATASET)) as file:
        lines = file.readlines()
        for line in lines:
            drug_without_feature.append(line.split()[0])
    protein_without_feature = []
    with open("./../../Datasets/{}/protein_without_feature.txt".format(DATASET)) as file:
        lines = file.readlines()
        for line in lines:
            protein_without_feature.append(line.split()[0])
    columns = ['head', 'tail', 'label']

    print("load data")
    train_data_list = pd.read_csv("./../../Datasets/{}/{}/train_set{}.csv".format(DATASET, scenarios, fold))[columns].values
    val_data_list = pd.read_csv("./../../Datasets/{}/{}/valid_set{}.csv".format(DATASET, scenarios, fold))[columns].values
    test_data_list = pd.read_csv("./../../Datasets/{}/{}/test_set{}.csv".format(DATASET, scenarios, fold))[columns].values

    # 对每个子集过滤掉那些没有没有特征的drug和protein
    train_data_list = [pair for pair in train_data_list if
                       pair[0] not in drug_without_feature and pair[1] not in protein_without_feature]
    val_data_list = [pair for pair in val_data_list if
                     pair[0] not in drug_without_feature and pair[1] not in protein_without_feature]
    test_data_list = [pair for pair in test_data_list if
                      pair[0] not in drug_without_feature and pair[1] not in protein_without_feature]
    print("data process done")
    train_set = CustomDataSet(train_data_list)
    val_set = CustomDataSet(val_data_list)
    test_set = CustomDataSet(test_data_list)

    print("load feature")
    try:
        drug_features = load_pickle("./../../Datasets/{}/feature/smiles_Mol2Vec300.pkl".format(DATASET))  # 药物vector
        drug_pretrain = load_pickle("./../../Datasets/{}/feature/smiles_Atom2Vec300.pkl".format(DATASET))  # 药物matrix
        protein_pretrain_sa = load_pickle("./../../Datasets/{}/feature/sa_SaProt1280.pkl".format(DATASET))  #  蛋白质Saport序列matrix+vector
        protein_pretrain_pocket = load_pickle("./../../Datasets/{}/feature/sa_SaProt1280_pockets.pkl".format(DATASET)) # 蛋白质SA序列 pocket matrix
        # protein_pretrain_pocket = load_pickle("./../../Datasets/{}/feature/aas_ProtTransBertBFD_pocket1280_sa.pkl".format(DATASET)) # 蛋白质SA序列 pocket matrix

        # drug_features = load_pickle("./../../Datasets/{}/feature/compound_Mol2Vec300.pkl".format(DATASET))  # 药物vector
        # drug_pretrain = load_pickle("./../../Datasets/{}/feature/compound_Atom2Vec300.pkl".format(DATASET))  # 药物matrix
        # protein_pretrain_sa = load_pickle("./../../Datasets/{}/feature/proteins_embed_sa1280.pkl".format(DATASET))  # 蛋白质Saport序列matrix+vector
        # protein_pretrain_pocket = load_pickle("./../../Datasets/{}/feature/proteins_embed_sa1280_pocket.pkl".format(DATASET))  # 蛋白质SA序列 pocket matrix

    except:
        print("Pre-training features for compounds and proteins are not found in the {}/feature folder, \n\
        please check the file naming or run Mol2Vec.py and generator.py first.".format(DATASET))
        raise
    collate_fn = collater_embeding(drug_features, drug_pretrain, protein_pretrain_sa,protein_pretrain_pocket)
    # collate_fn = collater_embeding(drug_features, drug_pretrain, protein_pretrain)

    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,collate_fn=collate_fn,pin_memory=True)
    # val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0,pin_memory=True,collate_fn=collate_fn)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,pin_memory=True,collate_fn=collate_fn)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
        num_workers=6,  # 好的开始
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True,  # 保持worker进程，避免反复创建
        prefetch_factor=2,  # 每个worker预取2个batch
        drop_last=True  # 根据你的需求调整
    )

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
        num_workers=3,  # 验证集可以用少一些
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True
    )

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
        num_workers=3,  # 测试集也可以用少一些
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True
    )

    print("Number of samples in the train set: ", len(train_set))
    print("Number of samples in the validation set: ", len(val_set))
    print("Number of samples in the test set: ", len(test_set))

    return train_loader, val_loader, test_loader

def load_Miss_dataset(DATASET,miss_rate, batch_size, fold=0):
    columns = ['head', 'tail', 'label']
    full_df = pd.read_csv("./../../Datasets/{}/full_pair.csv".format(DATASET))[columns]
    train_df, valid_test_df = split_train_valid(full_df, fold=fold,val_ratio=miss_rate/100)
    test_df, val_df = split_train_valid(valid_test_df, fold=fold, val_ratio=0.1)
    train_set = CustomDataSet(train_df.values)
    val_set = CustomDataSet(val_df.values)
    test_set = CustomDataSet(test_df.values)
    try:
        drug_features = load_pickle("./../../Datasets/{}/feature/compound_Mol2Vec300.pkl".format(DATASET))
        drug_pretrain = load_pickle("./../../Datasets/{}/feature/compound_Atom2Vec300.pkl".format(DATASET))
        protein_pretrain = load_pickle("./../../Datasets/{}/feature/aas_ProtTransBertBFD1024.pkl".format(DATASET))
    except:
        print("Pre-training features for compounds and proteins are not found in the {}/feature folder, \n\
        please check the file naming or run Mol2Vec.py and generator.py first.".format(DATASET))
        raise
    collate_fn = collater_embeding(drug_features, drug_pretrain, protein_pretrain)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                    collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0,collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,collate_fn=collate_fn)
    print("Number of samples in the train set: ", len(train_set))
    print("Number of samples in the validation set: ", len(val_set))
    print("Number of samples in the test set: ", len(test_set))

    return train_loader, val_loader, test_loader

"""load BindingDB of AIBind datasets"""
def load_BindingDB_AIBind_dataset(DATASET,scenarios, batch_size, fold=0):
    drug_without_feature = []
    with open("./../../Datasets/{}/drug_without_feature.txt".format(DATASET)) as file:
        lines = file.readlines()
        for line in lines:
            drug_without_feature.append(line.split()[0])
    protein_without_feature = []
    with open("./../../Datasets/{}/protein_without_feature.txt".format(DATASET)) as file:
        lines = file.readlines()
        for line in lines:
            protein_without_feature.append(line.split()[0])
    columns = ['head', 'tail', 'label']
    print("load data")
    train_data_list = pd.read_csv("./../../Datasets/{}/{}/train_set{}.csv".format(DATASET, scenarios, fold))[columns].values
    val_data_list = pd.read_csv("./../../Datasets/{}/{}/val_set{}.csv".format(DATASET, scenarios, fold))[columns].values
    test_data_list = pd.read_csv("./../../Datasets/{}/{}/test_set{}.csv".format(DATASET, scenarios, fold))[columns].values
    # 对每个子集过滤掉那些没有没有特征的drug和protein
    train_data_list = [pair for pair in train_data_list if pair[0] not in drug_without_feature and pair[1] not in protein_without_feature]
    val_data_list = [pair for pair in val_data_list if pair[0] not in drug_without_feature and pair[1] not in protein_without_feature]
    test_data_list = [pair for pair in test_data_list if pair[0] not in drug_without_feature and pair[1] not in protein_without_feature]
    print("data process done")
    train_set = CustomDataSet(train_data_list)
    val_set = CustomDataSet(val_data_list)
    test_set = CustomDataSet(test_data_list)
    print("load feature")
    try:
        # drug_features = load_pickle("./../../Datasets/{}/feature/compound_Mol2Vec300.pkl".format(DATASET))
        # drug_pretrain = load_pickle("./../../Datasets/{}/feature/compound_Atom2Vec300.pkl".format(DATASET))
        drug_features = load_pickle("./../../Datasets/{}/feature/compound_SMILESLM_desc.pkl".format(DATASET))
        drug_pretrain = load_pickle("./../../Datasets/{}/feature/compound_SMILESLM_tokens.pkl".format(DATASET))
        protein_pretrain = load_pickle("./../../Datasets/{}/feature/aas_ProtTransBertBFD1024.pkl".format(DATASET))
    except:
        print("Pre-training features for compounds and proteins are not found in the {}/feature folder, \n\
        please check the file naming or run Mol2Vec.py and generator.py first.".format(DATASET))
        raise
    collate_fn = collater_embeding(drug_features, drug_pretrain, protein_pretrain)
    print("feature load done")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                    collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0,collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,collate_fn=collate_fn)

    print("Number of samples in the train set: ", len(train_set))
    print("Number of samples in the validation set: ", len(val_set))
    print("Number of samples in the test set: ", len(test_set))

    return train_loader, val_loader, test_loader

def load_BindingDB_AIBind_Miss_dataset(DATASET,miss_rate, batch_size, fold=0):
    drug_without_feature = []
    with open("./../../Datasets/{}/drug_without_feature.txt".format(DATASET)) as file:
        lines = file.readlines()
        for line in lines:
            drug_without_feature.append(line.split()[0])
    protein_without_feature = []
    with open("./../../Datasets/{}/protein_without_feature.txt".format(DATASET)) as file:
        lines = file.readlines()
        for line in lines:
            protein_without_feature.append(line.split()[0])
    columns = ['head', 'tail', 'label']
    full_df = pd.read_csv("./../../Datasets/{}/full_pair.csv".format(DATASET))[columns]
    train_df, valid_test_df = split_train_valid(full_df, fold=fold,val_ratio=miss_rate/100)
    test_df, val_df = split_train_valid(valid_test_df, fold=fold, val_ratio=0.1)
    train_data_list = train_df.values
    val_data_list = val_df.values
    test_data_list = test_df.values
    train_data_list = [pair for pair in train_data_list if
                       pair[0] not in drug_without_feature and pair[1] not in protein_without_feature]
    val_data_list = [pair for pair in val_data_list if
                     pair[0] not in drug_without_feature and pair[1] not in protein_without_feature]
    test_data_list = [pair for pair in test_data_list if
                      pair[0] not in drug_without_feature and pair[1] not in protein_without_feature]
    train_set = CustomDataSet(train_data_list)
    val_set = CustomDataSet(val_data_list)
    test_set = CustomDataSet(test_data_list)
    try:
        drug_features = load_pickle("./../../Datasets/{}/feature/compound_Mol2Vec300.pkl".format(DATASET))
        drug_pretrain = load_pickle("./../../Datasets/{}/feature/compound_Atom2Vec300.pkl".format(DATASET))
        protein_pretrain = load_pickle("./../../Datasets/{}/feature/aas_ProtTransBertBFD1024.pkl".format(DATASET))
    except:
        print("Pre-training features for compounds and proteins are not found in the {}/feature folder, \n\
        please check the file naming or run Mol2Vec.py and generator.py first.".format(DATASET))
        raise
    collate_fn = collater_embeding(drug_features, drug_pretrain, protein_pretrain)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0,
                                    collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0,collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0,collate_fn=collate_fn)
    print("Number of samples in the train set: ", len(train_set))
    print("Number of samples in the validation set: ", len(val_set))
    print("Number of samples in the test set: ", len(test_set))

    return train_loader, val_loader, test_loader


if __name__ == "__main__":

    from prefetch_generator import BackgroundGenerator
    from tqdm import tqdm
    DATASET = "BindingDB_AIBind"
    scenarios = "warm_start"
    dataset_load,_,_ = load_BindingDB_AIBind_dataset(DATASET,scenarios, 32, fold=0)
    data_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_load)),
        total=len(dataset_load))
    for i_batch, i_data in data_pbar:
        '''data preparation '''
        input_tensors, labels_tensor = i_data
        d_g_tensor, d_m_tensor, p_g_tensor, p_m_tensor, \
        d_masks, p_masks = input_tensors
        print(d_m_tensor.shape)
