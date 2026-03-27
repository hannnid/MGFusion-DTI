# visualize_attention.py
import torch
from torch.utils.data import DataLoader
import argparse
import os
from model_visualize_attention import ColdstartCPI  # 你的模型路径
from dataset import load_scenario_dataset
import numpy as np
import pickle

def get_target_input(drug_id, prot_id, test_dataset):
    for i in range(len(test_dataset)):
        if test_dataset[i][0] == drug_id and test_dataset[i][1] == prot_id:
            return [test_dataset[i]]
    raise ValueError(f"未找到指定配对: {drug_id}, {prot_id}")

def save_attention(alpha_pd, alpha_dp, save_dir, drug_id, prot_id):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(alpha_pd.cpu(), f"{save_dir}/attn_pd_{drug_id}_{prot_id}.pt")
    torch.save(alpha_dp.cpu(), f"{save_dir}/attn_dp_{drug_id}_{prot_id}.pt")
    print(f"已保存注意力至: {save_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='BioSNAP')
    parser.add_argument('--scenario', type=str, default='warm_start')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--drug', type=str, default='DB08403')
    parser.add_argument('--protein', type=str, default='P22894')
    parser.add_argument('--checkpoint', type=str,default="./Results/BioSNAP/warm_start/valid_best_checkpoint0.pth")
    parser.add_argument('--save_dir', type=str, default='./attention_results')
    args = parser.parse_args()

    # ============ 加载数据 ===============
    print(f"加载 {args.dataset} 数据集的第 {args.fold} 折...")
    train_loader, val_loader, test_loader = load_scenario_dataset(args.dataset, args.scenario, args.fold, batch_size=256)

    test_dataset = test_loader.dataset
    target_data = get_target_input(args.drug, args.protein, test_dataset)

    # 重新构建 batch_size=1 的 dataloader
    from dataset import collater_embeding
    drug_f, drug_m, protein_sa, protein_pocket = test_loader.collate_fn.drug_f, test_loader.collate_fn.drug_m, \
                                                 test_loader.collate_fn.protein_m_sa, test_loader.collate_fn.protein_m_pocket

    collate_fn = collater_embeding(drug_f, drug_m, protein_sa, protein_pocket)
    target_loader = DataLoader(dataset=target_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # ============ 加载模型 ===============
    model = ColdstartCPI(unify_num=512, head_num=4)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval().cuda()

    # ============ 前向推理并保存注意力 ===============
    for input_batch, label in target_loader:
        input_batch = [x.cuda() for x in input_batch]
        with torch.no_grad():
            _, alpha_pd, alpha_dp, *_ = model(input_batch, return_attn=True)
            save_attention(alpha_pd[0], alpha_dp[0], args.save_dir, args.drug, args.protein)

if __name__ == "__main__":
    main()