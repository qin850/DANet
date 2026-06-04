from matplotlib.cbook import flatten
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import os
from torch.nn.functional import dropout
from models.modeling import utils
from models.modeling.backbone_vit.vit_model import vit_base_patch16_224 as create_model

from os.path import join
import pickle
import numpy as np
from torch.autograd import Variable
import torch.nn.init as init
from timm.models.layers import DropPath
Norm =nn.LayerNorm


def trunc_normal_(tensor, mean=0, std=.01):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

class SimpleReasoning(nn.Module):
    def __init__(self, np,ng):
        super(SimpleReasoning, self).__init__()
        self.hidden_dim = np//ng
        self.fc1 = nn.Linear(np, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, np)
        self.avgpool = nn.AdaptiveMaxPool1d(1)
        self.act = nn.GELU()

    def forward(self, x):#x(32,85,768)
        x_1 = self.fc1(self.avgpool(x).flatten(1)) #x_1(32,7)
        x_1 = self.act(x_1)#x_1(32,7)
        x_1 = F.sigmoid(self.fc2(x_1)).unsqueeze(-1)#x_1(32,7)->x_1(32,85,1)
        x_1 = x_1*x + x#(32,85,768)
        return x_1


class Tokenmix(nn.Module):
    def __init__(self, np):
        super(Tokenmix, self).__init__()
        dim =196
        hidden_dim = 512
        dropout = 0.
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(np)
        self.net = nn.Sequential(
            nn.Linear(dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,dim),
            nn.Dropout(dropout))
    def forward(self, x):
        redisual = x
        x = self.norm(x)
        x = rearrange(x, "b p c -> b c p")
        x = self.net(x)
        x = rearrange(x, "b c p-> b p c")
        out = redisual + x
        return out

class Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim):
        super().__init__()
        num_heads = 1
        qkv_bias = False
        qk_scale = None
        attn_drop = 0.
        proj_drop = 0.
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(hidden_features) or in_features
        self.norm = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        fan_in1, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
        bound1 = 1 / math.sqrt(fan_in1)
        nn.init.uniform_(self.fc1.bias, -bound1, bound1)
        fan_in2, _ = nn.init._calculate_fan_in_and_fan_out(self.fc2.weight)
        bound2 = 1 / math.sqrt(fan_in2)
        nn.init.uniform_(self.fc2.bias, -bound2, bound2)


class DEM(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        num_patches = 196
        self.grid_size = int(num_patches ** 0.5)
        # 针对振幅的空间注意力
        self.amp_spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        # 针对振幅的通道注意力 (类似 SE-Net)
        self.amp_channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim // 16, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim // 16, embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, L, D) -> (B, D, H, W)
        B, L, D = x.shape
        H = W = self.grid_size
        identity = x
        x_2d = x.transpose(1, 2).reshape(B, D, H, W)
        # FFT
        spectrum = torch.fft.fft2(x_2d, norm="forward")
        amp = torch.abs(spectrum)
        phase = torch.angle(spectrum)
        # 1. 空间维度过滤 (Spatial Gate)
        avg_out = torch.mean(amp, dim=1, keepdim=True)
        max_out, _ = torch.max(amp, dim=1, keepdim=True)
        s_mask = self.amp_spatial_att(torch.cat([avg_out, max_out], dim=1))
        amp = amp * s_mask
        # 2. 通道维度过滤 (Channel Gate)
        c_mask = self.amp_channel_att(amp)
        amp = amp * c_mask
        # 逆 FFT 还原
        refined_spec = torch.polar(amp, phase)
        output = torch.fft.ifft2(refined_spec, norm="forward").real
        output = output.reshape(B, D, L).transpose(1, 2)
        return identity + output

class MAttention(nn.Module):
    def __init__(self, dim, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(MAttention, self).__init__()
        use_temp = False
        #self.norm_q, self.norm_k, self.norm_v = Norm(dim), Norm(dim), Norm(dim)
        self.norm_semantic = nn.LayerNorm(dim)
        self.norm_visual1 = nn.LayerNorm(dim)
        self.norm_visual2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.scale = (dim ** (-0.5))
        self.act = nn.ReLU6()
        # 可学习温度（可选），能让模型自动调整缩放
        self.use_temp = use_temp
        self.temp = nn.Parameter(torch.tensor(1.0)) if use_temp else 1.0

        # dropout for attention and projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.proj = nn.Linear(dim, dim)

    def get_qkv(self, q, k, v):
        #q, k, v = self.norm_q(q), self.norm_k(k), self.norm_v(v)
        q = self.norm_semantic(q)
        k = self.norm_visual1(k)
        v = self.norm_visual2(v)
        q, k, v = self.to_q(q), self.to_k(k), self.to_v(v)
        return q, k, v

    def forward(self, q=None, k=None, v=None):
        """
        q: (B, Q, C), k: (B, K, C), v: (B, K, C)
        returns attn (B, Q, K) and out (B, Q, C)
        """
        q, k, v = self.get_qkv(q, k, v)  # keep dtype as-is

        # scaled dot-product attention (standard)
        attn = torch.einsum("b q c, b k c -> b q k", q, k)  # (B, Q, K)
        attn = self.act(attn)
        # apply (learnable) temperature and scale
        if self.use_temp:
            attn = attn * self.scale * self.temp
        else:
            attn = attn * self.scale

        # softmax over key dimension -> attention weights
        attn_mask = F.softmax(attn, dim=-1)
        #attn_mask = self.attn_drop(attn_mask)

        # weighted sum of values
        out = torch.einsum("b q k, b k c -> b q c", attn_mask, v)
        out = self.proj(out)
        out = self.proj_drop(out)
        return attn_mask, out


class SVBI(nn.Module):
    def __init__(self, dim, ffn_exp=4, drop_path=0.1, num_heads=1, num_parts=0,num_g=6):
        super(SVBI, self).__init__()
        self.dec_attn = AnyAttention1(dim, True)
        self.ffn1 = Mlp(dim, hidden_features=dim * ffn_exp, act_layer=nn.GELU, norm_layer=Norm)
        self.drop_path = nn.Identity()
        self.reason = Attention(dim)
        self.enc_attn = MAttention(dim, True)
        self.group_compact = SimpleReasoning(num_parts,num_g)
        self.maxpool1d = nn.AdaptiveMaxPool1d(1)
        self.enc_ffn = Mlp(dim, hidden_features=dim, act_layer=nn.GELU)

    def forward(self, x, parts=None):
        x = rearrange(x, "b c p -> b p c")
        attn_0,attn_out = self.enc_attn(q=parts, k=x, v=x)
        attn_0=self.maxpool1d(attn_0).flatten(1)
        parts1 = parts + attn_out
        parts2 = self.group_compact(parts1)
        if self.enc_ffn is not None:
            parts_out = parts2 + self.enc_ffn(parts2) + parts1
        parts_d = parts+parts_out
        attn_1,attn_out = self.enc_attn(q=parts_d, k=x, v=x)
        attn_1=self.maxpool1d(attn_1).flatten(1)
        parts1_d = parts_d + attn_out
        parts_comp = self.group_compact(parts1_d)
        if self.enc_ffn is not None:
            parts_in = parts_comp + self.enc_ffn(parts_comp)+ parts1_d
        attn_mask,feats = self.dec_attn(q=x, k=parts_in, v=parts_in)
        feats = x + feats
        feats = self.reason(feats)
        feats = feats + self.ffn1(feats)
        feats = rearrange(feats, "b p c -> b c p")
        return feats,attn_0,attn_1


class DANet(nn.Module):
    def __init__(self, basenet, c,
                 attritube_num, cls_num, ucls_num, group_num, w2v,
                 scale=20.0, device=None):

        super(DANet, self).__init__()
        self.attritube_num = attritube_num#168
        self.group_num=group_num
        self.feat_channel = c#768
        self.batch =10
        self.cls_num= cls_num#102
        self.ucls_num = ucls_num#12
        self.scls_num = cls_num - ucls_num#90

        self.w2v_att = torch.from_numpy(w2v).float().to(device)  # (312,300)

        self.W = nn.Parameter(trunc_normal_(torch.empty(self.w2v_att.shape[1], self.feat_channel)),#(300,768)
                              requires_grad=True)#
        self.V = nn.Parameter(trunc_normal_(torch.empty(self.feat_channel, self.attritube_num)),#(768,168)
                              requires_grad=True)#


        #print(f"调试信息: w2v_att.shape = {self.w2v_att.shape}")
        #print(f"调试信息: attribute_num = {self.attribute_num}")
        #print(f"调试信息: w2v_att 类型: {type(self.w2v_att)}")
        assert self.w2v_att.shape[0] == self.attritube_num#
        if scale<=0:
            self.scale = nn.Parameter(torch.ones(1) * 20.0)
        else:
            self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)#

        self.backbone_patch = nn.Sequential(*list(basenet.children()))[0]
        self.backbone_drop= nn.Sequential(*list(basenet.children()))[1]
        self.backbone_0 = nn.Sequential(*list(basenet.children()))[2][:-1]#VIT结构 -1
        self.backbone_1 = nn.Sequential(*list(basenet.children()))[2][-1]#VIT最后的Block

        self.drop_path = 0.4

        self.cls_token = basenet.cls_token#(1,1,768)
        self.pos_embed = basenet.pos_embed#(1,196+1,768)

        self.cat = nn.Linear(self.attritube_num*self.feat_channel, attritube_num)#Linear(in_features=239616, out_features=312, bias=True)
        self.avgpool1d = nn.AdaptiveAvgPool1d(1)#AdaptiveAvgPool1d(output_size=1)
        self.CLS_loss = nn.CrossEntropyLoss()#
        self.Reg_loss = nn.MSELoss()

        self.blocks = SVBI(self.feat_channel,
                            num_heads=1,
                            num_parts=168,
                            num_g=self.group_num,
                            ffn_exp=4,
                            drop_path=0.4)

        self.log_softmax_func = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.maxpool1d = nn.AdaptiveMaxPool1d(1)
        # 原型学习和映射
        self.prototype = nn.Linear(self.feat_channel, self.attritube_num, bias=False)
        self.matrix = nn.Linear(self.feat_channel, self.attritube_num, bias=False)
        self.Spectrum = DEM(self.feat_channel)

    def compute_score(self, gs_feat, seen_att, att_all):
        gs_feat = gs_feat.view(self.batch, -1)
        gs_feat_norm = torch.norm(gs_feat, p=2, dim=1).unsqueeze(1).expand_as(gs_feat)
        gs_feat_normalized = gs_feat.div(gs_feat_norm + 1e-5)
        temp_norm = torch.norm(att_all, p=2, dim=1).unsqueeze(1).expand_as(att_all)
        seen_att_normalized = att_all.div(temp_norm + 1e-5)
        score_o = torch.einsum('bd,nd->bn', gs_feat_normalized, seen_att_normalized)
        d, _ = seen_att.shape#d=90
        #print(f"训练集样本数: {d}")
        score_o = score_o*self.scale#(32,102)
        if d == self.cls_num:
            score = score_o
        if d == self.scls_num:
            score = score_o[:, :d]
            uu = self.ucls_num #12
            if self.training:
                mean1 = score_o[:, :d].mean(1)
                std1 = score_o[:, :d].std(1)
                mean2 = score_o[:, -uu:].mean(1)
                std2 = score_o[:, -uu:].std(1)
                mean_score = F.relu6(mean1 - mean2)
                std_score = F.relu6(std1 - std2)
                mean_loss = mean_score.mean(0) + std_score.mean(0)
                #return score, mean_loss
                return score_o, mean_loss
        if d == self.ucls_num:
            score = score_o[:, -d:]
        return score, _


    def compute_loss_Self_Calibrate(self, S_pp,unseenclass):
        # S_pp = in_package['S_pp']
        Prob_all = F.softmax(S_pp, dim=-1)
        Prob_unseen = Prob_all[:, unseenclass]
        assert Prob_unseen.size(1) == len(unseenclass)
        mass_unseen = torch.sum(Prob_unseen, dim=1)
        loss_pmp = -torch.log(torch.mean(mass_unseen))
        return loss_pmp

    def compute_aug_cross_entropy(self, S_pp, Labels, trian_class_counts):
        Prob = self.log_softmax_func(S_pp)
        labels = torch.nn.functional.one_hot(Labels, num_classes=102).float()
        if trian_class_counts != None:
            batch_class_count = torch.matmul(labels[:, :90], trian_class_counts)
            class_weights = (1 - 0.99) / (1 - 0.99 ** batch_class_count)
        loss = -torch.einsum('bk,bk->b', Prob, labels) * (1 + class_weights)
        loss = torch.mean(loss)
        return loss

    def compute_softmax_2(self, S_pp, Labels, att_all):  # (B,att_num),(B,)
        labels = torch.nn.functional.one_hot(Labels, num_classes=102).float()
        tgt = torch.matmul(labels, att_all)
        pred_1 = self.softmax(S_pp[:, :8])
        pred_2 = self.softmax(S_pp[:, 8:168])

        labels_1 = torch.argmax(tgt[:, :8], dim=1)
        labels_2 = torch.argmax(tgt[:, 8:168], dim=1)


        loss_ce_1 = self.CLS_loss(pred_1, labels_1)

        loss_ce_2 = self.CLS_loss(pred_2, labels_2)


        loss_reg = 0.2 * loss_ce_1 + 0.8 * loss_ce_2
        return loss_reg


    def compute_softmax(self, S_pp, Labels, att_all):  # (B,att_num),(B,)
        labels = torch.nn.functional.one_hot(Labels, num_classes=102).float()
        #print(f"labels形状: {labels.shape}")
        #print(f"att_all形状: {att_all.shape}")
        tgt = torch.matmul(labels, att_all)
        pred_1 = self.softmax(S_pp[:, :8])
        pred_2 = self.softmax(S_pp[:, 8:168])

        labels_1 = torch.argmax(tgt[:, :8], dim=1)
        labels_2 = torch.argmax(tgt[:, 8:168], dim=1)
        loss_ce_1 = self.CLS_loss(pred_1, labels_1)
        loss_ce_2 = self.CLS_loss(pred_2, labels_2)

        loss_reg = 0.1 * loss_ce_1 + 1.2 * loss_ce_2
        return loss_reg

    def con_loss(self, features, labels):#(B,att_num),(B,)
        # features = in_package['embed']
        # labels = torch.argmax(Labels, dim=1)
        B, _ = features.shape
        features = F.normalize(features)
        cos_matrix = features.mm(features.t()) 
        pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float() 
        neg_label_matrix = 1 - pos_label_matrix  
        pos_cos_matrix = 1 - cos_matrix
        neg_cos_matrix = cos_matrix - 0.4
        neg_cos_matrix[neg_cos_matrix < 0] = 0
        loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
        loss /= (B * B)
        return loss

    def forward(self, x, att=None, label=None, seen_att=None, att_all=None, seenclass=None, unseenclass=None,
                trian_class_counts=None):
        self.batch = x.shape[0]
        parts = torch.einsum('lw,wv->lv', self.w2v_att, self.W) 
        parts = parts.expand(self.batch, -1, -1)  
        patches = self.backbone_patch(x)  # (32,3,224,224)->(32,196,768)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # (32,1,768)
        patches = torch.cat((cls_token, patches), dim=1)  
        feats_0 = self.backbone_drop(patches + self.pos_embed)
        feats_0 = self.backbone_0(feats_0)  # (32,197,768)
        #feats_0 = feats_0[:, 1:, :]  

        # 获取用于后续的特征
        #feats_in = feats_no_prompts[:, 1:, :] 
        feats_in = feats_0[:, 1:, :] 
        feats_in = self.Spectrum(feats_in)

        feats_out, _, _ = self.blocks(feats_in.transpose(1, 2), parts=parts)
        patches_1 = torch.cat((cls_token, feats_out.transpose(1, 2)), dim=1)  # (32,768,196)->(32,197,768)
        feats_1 = self.backbone_1(patches_1 + self.pos_embed)  # (32,197,768)
        feats_1 = feats_1[:, 1:, :]  # (32,196,768)->(32,197,768)
       
        feats_1, attn_0, _ = self.blocks(feats_1.transpose(1, 2), parts=parts)
        feats_1_ = feats_1  # (32,768,196)
      
        out_1 = self.avgpool1d(feats_1_.view(self.batch, self.feat_channel, -1)).view(self.batch, -1)
        out = torch.einsum('bc,cd->bd', out_1, self.V)  

       
        f_o = out  
      
        score, b = self.compute_score(out, seen_att, att_all)

        if not self.training:
            return score

        Lsa = torch.tensor(0)
        Lhp = self.compute_softmax(f_o, label, att_all)
        Lce = self.compute_aug_cross_entropy(score, label, trian_class_counts)
        scale = self.scale.item()
        b = torch.tensor(0).to(x.device)
        loss_dict = {
            'CE_loss': Lce,
            'HP_loss': Lhp,
            'SA_loss': Lsa,
            'scale': scale,
            'bias_loss': b
        }

        return loss_dict


def build_DANet(cfg):
    dataset_name = cfg.DATASETS.NAME
    info = utils.get_attributes_info(dataset_name)
    attritube_num = 170
    cls_num = 102
    ucls_num = 12
    group_num = 1
    c, w, h = 768, 14, 14
    scale = cfg.MODEL.SCALE
    vit_model = create_model(num_classes=-1)
    vit_model_path = "vit_base_patch16_224.pth"
    weights_dict = torch.load(vit_model_path)
    del_keys = ['head.weight', 'head.bias'] if vit_model.has_logits \
        else ['head.weight', 'head.bias']
    for k in del_keys:
        del weights_dict[k]
    vit_model.load_state_dict(weights_dict, strict=False)
    w2v_file = dataset_name+"_attribute-Q2.pkl"
    w2v_path = join(cfg.MODEL.ATTENTION.W2V_PATH, w2v_file)

    with open(w2v_path, 'rb') as f:
        #w2v = pickle.load(f)
        w2v = pickle.load(f, encoding='latin1')

    device = torch.device(cfg.MODEL.DEVICE)


    return DANet(basenet=vit_model,
                  c=c,scale=scale,
                  attritube_num=attritube_num,
                  group_num=group_num, w2v=w2v,
                  cls_num=cls_num, ucls_num=ucls_num,
                  device=device)
