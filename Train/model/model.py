# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class CAN_Layer(nn.Module):
    def __init__(self, hidden_dim, num_heads, agg_mode, use_mask):
        super(CAN_Layer, self).__init__()
        self.agg_mode = agg_mode
        # self.group_size = args.group_size  # Control Fusion Scale
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads
        self.use_mask = use_mask

        self.query_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_p = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.query_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_d = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_d = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def alpha_logits(self, logits, mask_row, mask_col, inf=1e6):
        N, L1, L2, H = logits.shape
        if not self.use_mask:
            alpha = torch.softmax(logits, dim=2)
            return alpha
        else:
            mask_row = mask_row.view(N, L1, 1).repeat(1, 1, H)
            mask_col = mask_col.view(N, L2, 1).repeat(1, 1, H)


            mask_pair = torch.einsum('blh, bkh->blkh', mask_row, mask_col)

            logits = torch.where(mask_pair, logits,
                                 logits - inf)  # 如果mask_pair=True则保持原logits，否则logits设为负无穷。这样在做softmax时，无效未知的注意力权重会变成0
            alpha = torch.softmax(logits, dim=2)
            mask_row = mask_row.view(N, L1, 1, H).repeat(1, 1, L2, 1)
            alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
            return alpha

    def apply_heads(self, x, n_heads, n_ch):
        s = list(x.size())[:-1] + [n_heads, n_ch]  # 通常D=n_heads*n_ch (64,512,768)-->(64,512,8,96)
        return x.view(*s)  # 对x进行reshape 从(B，L，D)-->(B,L,n_head,n_ch)

    def forward(self, protein, drug, mask_prot, mask_drug):
        # True：有效token(需要保留)  False：填充token(需要被mask)
        # protein_grouped:(64,512,768)--->query_prot(64,512,8,96) 经过query_p(线性投影)形状不变，但数值被投影到了新的语义空间，再通过apply_heads被切分为多头
        query_prot = self.apply_heads(self.query_p(protein), self.num_heads, self.head_size)
        key_prot = self.apply_heads(self.key_p(protein), self.num_heads, self.head_size)
        value_prot = self.apply_heads(self.value_p(protein), self.num_heads, self.head_size)

        query_drug = self.apply_heads(self.query_d(drug), self.num_heads, self.head_size)
        key_drug = self.apply_heads(self.key_d(drug), self.num_heads, self.head_size)
        value_drug = self.apply_heads(self.value_d(drug), self.num_heads, self.head_size)

        # Compute attention scores
        logits_pp = torch.einsum('blhd, bkhd->blkh', query_prot, key_prot)
        logits_pd = torch.einsum('blhd, bkhd->blkh', query_prot, key_drug)
        logits_dp = torch.einsum('blhd, bkhd->blkh', query_drug, key_prot)
        logits_dd = torch.einsum('blhd, bkhd->blkh', query_drug, key_drug)
        # print("logits_pp:", logits_pp.shape)

        # 以下计算得到注意力权重 形状都是(64,512,512,8)
        alpha_pp = self.alpha_logits(logits_pp, mask_prot, mask_prot)
        alpha_pd = self.alpha_logits(logits_pd, mask_prot, mask_drug)
        alpha_dp = self.alpha_logits(logits_dp, mask_drug, mask_prot)
        alpha_dd = self.alpha_logits(logits_dd, mask_drug, mask_drug)

        # 得到的prot_embedding形状(64,512,768)
        prot_embedding = (torch.einsum('blkh, bkhd->blhd', alpha_pp, value_prot).flatten(-2) +
                          torch.einsum('blkh, bkhd->blhd', alpha_pd, value_drug).flatten(-2)) / 2
        drug_embedding = (torch.einsum('blkh, bkhd->blhd', alpha_dp, value_prot).flatten(-2) +
                          torch.einsum('blkh, bkhd->blhd', alpha_dd, value_drug).flatten(-2)) / 2

        if self.agg_mode == "cls":
            prot_embed = prot_embedding[:, 0]  # query : [batch_size, hidden]
            drug_embed = drug_embedding[:, 0]  # query : [batch_size, hidden]
        elif self.agg_mode == "mean_all_tok":
            prot_embed = prot_embedding.mean(1)  # query : [batch_size, hidden]
            drug_embed = drug_embedding.mean(1)  # query : [batch_size, hidden]
        elif self.agg_mode == "mean":
            prot_embed = (prot_embedding * mask_prot.unsqueeze(-1)).sum(1) / mask_prot.sum(
                -1).unsqueeze(-1)
            drug_embed = (drug_embedding * mask_drug.unsqueeze(-1)).sum(1) / mask_drug.sum(
                -1).unsqueeze(-1)
        else:
            raise NotImplementedError()



        query_embed = torch.cat([prot_embed, drug_embed], dim=1)

        # print("query_embed:", query_embed.shape)
        return query_embed

#
# 用门控融合代替transformer
class BiGatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate_cp = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.gate_pc = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.out_ln = nn.LayerNorm(dim)

    def forward(self, c, p):  # (B, D), (B, D)
        g_cp = self.gate_cp(torch.cat([c, p], dim=-1))
        g_pc = self.gate_pc(torch.cat([p, c], dim=-1))
        c_new = self.out_ln(g_cp * p + (1 - g_cp) * c)
        p_new = self.out_ln(g_pc * c + (1 - g_pc) * p)
        # return c_new, p_new
        global_embed = torch.cat([c_new, p_new], dim=1)
        return global_embed
class MlPdecoder_CAN(nn.Module):

    def __init__(self, input_dim, binary=2, dropout=0.1):  # unify_num=512
        super(MlPdecoder_CAN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.output = nn.Linear(256, binary)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.bn1(F.relu(self.fc1(x))))
        x = self.dropout(self.bn2(F.relu(self.fc2(x))))
        x = self.dropout(self.bn3(F.relu(self.fc3(x))))
        return self.output(x)
class ColdstartCPI(nn.Module):
    def __init__(self,unify_num,head_num, dataset = "BindingDB_AIBind"):  # unify_num=512,head_num=4
        super(ColdstartCPI, self).__init__()
        self.c_g_unit = nn.Sequential(
            nn.Linear(300, 300),
            nn.PReLU(),
            nn.Linear(300, unify_num),
            nn.PReLU()
        )
        self.c_m_unit = nn.Sequential(
            nn.Linear(300, 300),
            nn.PReLU(),
            nn.Linear(300, unify_num),
            nn.PReLU()
        )

        self.p_g_unit = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.PReLU(),
            nn.Linear(1024, unify_num),
            nn.PReLU()
        )
        self.p_m_unit = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.PReLU(),
            nn.Linear(1024, unify_num),
            nn.PReLU()
        )
        self.p_g_unit_sa = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.PReLU(),
            nn.Linear(1280, unify_num),
            nn.PReLU()
        )
        self.p_m_unit_sa = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.PReLU(),
            nn.Linear(1280, unify_num),
            nn.PReLU()
        )
        # self.Interacting_Layer = nn.TransformerEncoderLayer(unify_num, head_num,batch_first=True)
        self.global_fusion = BiGatedFusion(unify_num)
        self.can_layer = CAN_Layer(hidden_dim=unify_num, num_heads=8, agg_mode="cls", use_mask=True)  # baseline unify_num=512;cross-attention unify_num=768
        # self.mlp_classifier = MlPdecoder_CAN(input_dim=1024)  # 使用源代码维度时 unify_num=512， P_max=1000，d_max=100
        self.mlp_classifier = MlPdecoder_CAN(input_dim=2048)  # 使用源代码维度时 unify_num=512， P_max=1000，d_max=100

    def forward(self, input_tensors):
        '''
        # input_batch
            c_g_f,(B, 300)----------->(B, unify_num)
            c_m,(B, d_max, 300)------>(B, d_max,unify_num)
            p_g_f,(B, 1024)----------->(B, unify_num)
            p_m,(B, p_max, 1024)------>(B, d_max,unify_num)
            d_masks,(B, d_max)
            p_masks,(B, p_max)

            p_g_f_sa,(B, 1280)------->(B, unify_num)
            p_m_sa,(B, 512, 1280)---->(B, 512, unify_num)
            p_masks_sa,(B, 512)

        # d_max=100  p_max=1000
        '''
        c_g_f, c_m, p_g_f_sa, p_m_sa, p_m_pocket_sa, c_mask, p_mask_sa, p_mask_pocket_sa = input_tensors

        # 特征变换
        c_g_f = self.c_g_unit(c_g_f)
        c_m = self.c_m_unit(c_m)
        p_g_f_sa = self.p_g_unit_sa(p_g_f_sa)  # p_g_f_sa(256,512)
        p_m_sa = self.p_m_unit_sa(p_m_sa)  # p_m_sa(256,512,512)
        p_m_pocket_sa = self.p_m_unit_sa(p_m_pocket_sa)


# -------------------------------------药物vector，matrix；蛋白质vector，matrix 口袋matrix----------------------

        # global特征做Bi-gate
        global_embed = self.global_fusion(c_g_f, p_g_f_sa)  # 不加attention指导 AUC：7795 PRC：7926

        # token特征做cross-attention
        joint_embed = self.can_layer(p_m_pocket_sa, c_m, p_mask_pocket_sa, c_mask)  # 口袋token joint_embed(B,unify_num*2) p_m_sa(B,512,512), c_m(B,300,512)
        # joint_embed = self.can_layer(p_m_sa, c_m, p_mask_sa, c_mask)  # 全部token

        # 代替CAN token特征直接平均cat
        # prot_embed = p_m_pocket_sa.mean(1)  # Average over tokens (B, unify_num)
        # drug_embed = c_m.mean(1)
        # joint_embed = torch.cat([prot_embed, drug_embed], dim=1)

        joint_embed = torch.cat([joint_embed, global_embed], dim=1)  # 拼接全局信息joint_embed(B,unify_num*4)

        # 没有global fusion 直接拼接  备用：不使用全局
        # joint_embed = torch.cat([joint_embed, c_g_f, p_g_f_sa], dim=1)  # 拼接全局信息joint_embed(B,unify_num*4)

        predict = self.mlp_classifier(joint_embed)
        return predict


    def get_all_features_with_logits(self, input_tensors):
        """
        返回：
        raw: 原始拼接前的特征
        g: global 特征
        i: cross-attention 特征
        f: fusion 特征（分类器输入）
        logits: 分类器输出（未过sigmoid）
        """
        c_g_f, c_m, p_g_f_sa, p_m_sa, p_m_pocket_sa, c_mask, p_mask_sa, p_mask_pocket_sa = input_tensors

        # 特征变换
        c_g_f = self.c_g_unit(c_g_f)
        c_m = self.c_m_unit(c_m)
        p_g_f_sa = self.p_g_unit_sa(p_g_f_sa)
        p_m_sa = self.p_m_unit_sa(p_m_sa)
        p_m_pocket_sa = self.p_m_unit_sa(p_m_pocket_sa)

        # Bi-gated fusion
        g = self.global_fusion(c_g_f, p_g_f_sa)

        # Cross-attention
        i = self.can_layer(p_m_pocket_sa, c_m, p_mask_pocket_sa, c_mask)

        # Fusion 特征
        f = torch.cat([i, g], dim=1)

        # 分类器输出（未sigmoid）
        logits = self.mlp_classifier(f)

        # raw 为四种输入的 concat（也可自定义）
        raw = torch.cat([c_g_f, p_g_f_sa], dim=1)

        return raw, g, i, f, logits
if __name__ == "__main__":
    c_g_f = torch.ones([2,300]).cuda()
    c_m = torch.ones([2,16,300]).cuda()
    p_g_f = torch.ones([2,1024]).cuda()
    p_m = torch.ones([2,21,1024]).cuda()
    c_mask = torch.zeros([2,16]).cuda()
    p_mask = torch.zeros([2,21]).cuda()

    model = ColdstartCPI(100,4).cuda()
    output = model([c_g_f,c_m,p_g_f,p_m, c_mask, p_mask])
    print(output.shape)
