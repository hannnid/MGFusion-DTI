# extract_tsne_features.py

import torch
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from model_MutiAttention import ColdstartCPI
from dataset import load_scenario_dataset
import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 默认值
'''
目前是在blind_start场景下,fold3下效果尚可
'''
# 参数设置
DATASET = "BioSNAP"
SCENARIO = "protein_cold_start"  # 双盲冷启动
FOLD = 0  # 选择你要可视化的fold编号  默认场景都是fold3，但是protein cold效果不好
BATCH_SIZE = 256
SAVE_PATH = f"./Results/{DATASET}/{SCENARIO}/"

if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# 加载模型
model = ColdstartCPI(unify_num=512, head_num=4).cuda()
checkpoint_path = SAVE_PATH + f"valid_best_checkpoint{FOLD}.pth"
model.load_state_dict(torch.load(checkpoint_path))
model.eval()

# 加载测试集（确保你加载的是双盲分割的test set）
_, _, test_dataset_load = load_scenario_dataset(DATASET, SCENARIO, FOLD, batch_size=BATCH_SIZE)

# 初始化特征列表
raw_list, global_list, inter_list, fusion_list, label_list, logits_list = [], [], [], [], [], [] # 添加了分类器之后的特征
# raw_list, global_list, inter_list, fusion_list, label_list = [], [], [], [], []

with torch.no_grad():
    for data in tqdm(BackgroundGenerator(test_dataset_load), total=len(test_dataset_load)):
        input_tensors, labels = data
        input_tensors = [x.cuda() for x in input_tensors]
        labels = labels.cuda()

        # raw, g, i, f = model.get_all_features(input_tensors)
        raw, g, i, f, logits = model.get_all_features_with_logits(input_tensors) # 添加了分类器之后的特征
        raw_list.append(raw.cpu())
        global_list.append(g.cpu())
        inter_list.append(i.cpu())
        fusion_list.append(f.cpu())
        label_list.append(labels.cpu())
        logits_list.append(logits.cpu()) # 添加了分类器之后的特征

# 保存特征
torch.save(torch.cat(raw_list), SAVE_PATH + f"tsne_raw_feat_fold{FOLD}.pt")
torch.save(torch.cat(global_list), SAVE_PATH + f"tsne_global_feat_fold{FOLD}.pt")
torch.save(torch.cat(inter_list), SAVE_PATH + f"tsne_inter_feat_fold{FOLD}.pt")
torch.save(torch.cat(fusion_list), SAVE_PATH + f"tsne_fusion_feat_fold{FOLD}.pt")
torch.save(torch.cat(label_list), SAVE_PATH + f"tsne_labels_fold{FOLD}.pt")
torch.save(torch.cat(logits_list), SAVE_PATH + f"tsne_logits_feat_fold{FOLD}.pt") # 添加了分类器之后的特征

print("✔️ 特征提取完毕，已保存为 .pt 文件，可用于 t-SNE 可视化。")