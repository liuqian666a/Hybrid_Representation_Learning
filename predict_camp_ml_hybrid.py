#!/usr/bin/env python3
import os
import sys
import glob
import math
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

from peptide_features import extract_features as extract_traditional_features

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAMP_DIR = os.path.dirname(SCRIPT_DIR)  # CAMP_pytorch/

# ==================== TMC4 蛋白质序列（与训练时一致） ====================
TMC4_SEQUENCE = "MEENPTLESEAWGSSRGWLAPREARGAPCSSPGPSLSSVLNELPSAATLRYRDPGVLPWGALEEEEEDGGRSRKAFTEVTQTELQDPHPSRELPWPMQARRAHRQRNASRDQVVYGSGTKTDRWARLLRRSKEKTKEGLRSLQPWAWTLKRIGGQFGAGTESYFSLLRFLLLLNVLASVLMACMTLLPTWLGGAPPGPPGPDISSPCGSYNPHSQGLVTFATQLFNLLSGEGYLEWSPLFYGFYPPRPRLAVTYLCWAFAVGLICLLLILHRSVSGLKQTLLAESEALTSYSHRVFSAWDFGLCGDVHVRLRQRIILYELKVELEETVVRRQAAVRTLGQQARVWLVRVLLNLLVVALLGAAFYGVYWATGCTVELQEMPLVQELPLLKLGVNYLPSIFIAGVNFVLPPVFKLIAPLEGYTRSRQIVFILLRTVFLRLASLVVLLFSLWNQITCGGDSEAEDCKTCGYNYKQLPCWETVLGQEMYKLLLFDLLTVLAVALLIQFPRKLLCGLCPGALGRLAGTQEFQVPDEVLGLIYAQTVVWVGSFFCPLLPLLNTVKFLLLFYLKKLTLFSTCSPAARTFRASAANFFFPLVLLLGLAISSVPLLYSIFLIPPSKLCGPFRGQSSIWAQIPESISSLPETTQNFLFFLGTQAFAVPLLLISSILMAYTVALANSYGRLISELKRQRQTEAQNKVFLARRAVALTSTKPAL"
TMC4_SS = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCHHHHHHHCCCHHHHHHHCCCCCCCCCCCCCCCCCCCCHHHHHHHHHHHHCCCCCCCCCCCCHHHHHHHHHHHHHHHHHHHHCCCHHHHHHHHHHHHHHHHHHHHHHCCCCHHHHHHHHHHHCHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCCCCCCCCCCCCHHHHHHHHHHCCHHHHHCCCCHHHCCCCHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHCCHHHHHHHHHHHCCCCCCCCHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHCCHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHCHHHHHCHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHCCCHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCCCHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHCCCHHHHHHCCCECCHHHHHHHHHHHHHHHHHHCCCCCCHHHHHHHHHHHHHHHHHHHHHHCCECCCCCCCHHHHHHHHHHHHHHHHHHHHHHHHHHHHCCCCCCCCCCCCCCCCCCCHHHHHHHHCCHHHHHHHHHHHCHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHCCCC"


# ==================== CAMP 模型组件 ====================

class GlobalMaxPool1d(nn.Module):
    def forward(self, x):
        output, _ = torch.max(x, 1)
        return output


class ConvNN(nn.Module):
    def __init__(self, in_dim, c_dim, kernel_size):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_dim, c_dim, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv1d(c_dim, c_dim * 2, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv1d(c_dim * 2, c_dim * 3, kernel_size, padding="same"),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.convs(x)


class Self_Attention(nn.Module):
    def __init__(self, input_dim, dim_k, dim_v):
        super().__init__()
        self.q = nn.Linear(input_dim, dim_k)
        self.k = nn.Linear(input_dim, dim_k)
        self.v = nn.Linear(input_dim, dim_v)
        self._norm_fact = 1.0 / math.sqrt(dim_k)

    def forward(self, x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        atten = nn.Softmax(dim=-1)(torch.bmm(Q, K.permute(0, 2, 1))) * self._norm_fact
        return torch.bmm(atten, V)


class CAMP(nn.Module):
    """CAMP 模型 — 共享嵌入层版本"""

    def __init__(self):
        super().__init__()
        self.embed_seq = nn.Embedding(66, 128)
        self.embed_ss = nn.Embedding(76, 128)
        self.embed_two = nn.Embedding(8, 128)
        self.pep_convs = ConvNN(512, 64, 7)
        self.prot_convs = ConvNN(512, 64, 8)
        self.pep_fc = nn.Linear(3, 128)
        self.prot_fc = nn.Linear(23, 128)
        self.global_max_pooling = GlobalMaxPool1d()
        self.dnns = nn.Sequential(
            nn.Linear(640, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 512),
        )
        self.att = Self_Attention(128, 128, 128)
        self.output = nn.Linear(512, 1)

    def forward(self, *args, **kwargs):
        return self.extract_features(*args, **kwargs)

    def extract_features(self, x_pep, x_prot, x_pep_ss, x_prot_ss,
                         x_pep_2, x_prot_2, x_pep_dense, x_prot_dense):
        pep_seq_emb = self.embed_seq(x_pep.long())
        prot_seq_emb = self.embed_seq(x_prot.long())
        pep_ss_emb = self.embed_ss(x_pep_ss.long())
        prot_ss_emb = self.embed_ss(x_prot_ss.long())
        pep_2_emb = self.embed_two(x_pep_2.long())
        prot_2_emb = self.embed_two(x_prot_2.long())
        pep_dense = self.pep_fc(x_pep_dense)
        prot_dense = self.prot_fc(x_prot_dense)
        enc_pep = torch.cat([pep_seq_emb, pep_ss_emb, pep_2_emb, pep_dense], dim=-1)
        enc_prot = torch.cat([prot_seq_emb, prot_ss_emb, prot_2_emb, prot_dense], dim=-1)
        enc_pep = self.pep_convs(enc_pep.permute(0, 2, 1)).permute(0, 2, 1)
        enc_prot = self.prot_convs(enc_prot.permute(0, 2, 1)).permute(0, 2, 1)
        pep_cnn = self.global_max_pooling(enc_pep)
        prot_cnn = self.global_max_pooling(enc_prot)
        pep_att = self.global_max_pooling(self.att(self.embed_seq(x_pep.long())))
        prot_att = self.global_max_pooling(self.att(self.embed_seq(x_prot.long())))
        return pep_cnn, prot_cnn, pep_att, prot_att


# ==================== 实时特征生成 ====================

AA_SET = {k: v for v, k in enumerate("ACBEDGFIHKMLONQPSRUTWVYXZ", 1)}
SS_SET = {"H": 1, "C": 2, "E": 3}
PHYSICO_SET = {
    'A': 1, 'C': 3, 'B': 7, 'E': 5, 'D': 5, 'G': 2, 'F': 1,
    'I': 1, 'H': 6, 'K': 6, 'M': 1, 'L': 1, 'O': 7, 'N': 4,
    'Q': 4, 'P': 1, 'S': 4, 'R': 6, 'U': 7, 'T': 4, 'W': 2,
    'V': 1, 'Y': 4, 'X': 7, 'Z': 7
}
SEQ_SS_DICT = {}
_idx = 1
for _aa in AA_SET:
    for _ss in SS_SET:
        SEQ_SS_DICT[f"{_aa}{_ss}"] = _idx
        _idx += 1

AA_PROPS = {
    'A': [1.28, 0.05, 1.00, 0.31, 6.11, 0.42, 0.23],
    'R': [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
    'N': [1.60, 0.13, 2.95, -0.60, 6.52, 0.21, 0.22],
    'D': [1.60, 0.11, 2.78, -0.77, 2.98, 0.25, 0.20],
    'C': [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
    'Q': [1.56, 0.18, 3.97, -0.22, 5.65, 0.36, 0.25],
    'E': [1.56, 0.15, 3.78, -0.64, 3.08, 0.44, 0.19],
    'G': [0.00, 0.00, 0.00, 0.00, 6.06, 0.13, 0.15],
    'H': [2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.30],
    'I': [4.19, 0.19, 4.00, 1.80, 6.04, 0.30, 0.45],
    'L': [2.59, 0.19, 4.00, 1.70, 6.04, 0.39, 0.31],
    'K': [1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
    'M': [2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32],
    'F': [2.94, 0.29, 5.89, 1.79, 5.67, 0.30, 0.38],
    'P': [2.67, 0.00, 2.72, 0.72, 6.80, 0.13, 0.34],
    'S': [1.31, 0.06, 1.60, -0.04, 5.70, 0.20, 0.28],
    'T': [3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36],
    'W': [3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42],
    'Y': [2.94, 0.30, 6.47, 0.96, 5.63, 0.25, 0.41],
    'V': [3.67, 0.14, 3.00, 1.22, 6.02, 0.27, 0.49],
    'X': [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
}

PAD_PEP = 50
PAD_PROT = 800


def _label_sequence(seq, pad_len):
    arr = np.zeros(pad_len, dtype=np.float64)
    for i, aa in enumerate(seq[:pad_len]):
        arr[i] = AA_SET.get(aa, 24)
    return arr


def _label_seq_ss(ss_str, pad_len, seq_str):
    arr = np.zeros(pad_len, dtype=np.float64)
    ss_clean = ss_str.replace(',', '')
    for i, (ss, aa) in enumerate(zip(ss_clean[:pad_len], seq_str[:pad_len])):
        arr[i] = SEQ_SS_DICT.get(f"{aa}{ss}", 0)
    return arr


def _label_physicochemical(seq, pad_len):
    arr = np.zeros(pad_len, dtype=np.float64)
    for i, aa in enumerate(seq[:pad_len]):
        arr[i] = PHYSICO_SET.get(aa, 7)
    return arr


def _get_dense_feature(sequence, pad_len, is_protein=False):
    feature_dim = 23 if is_protein else 3
    arr = np.zeros((pad_len, feature_dim), dtype=np.float64)
    for i, aa in enumerate(sequence[:pad_len]):
        props = AA_PROPS.get(aa, AA_PROPS['X'])
        if is_protein:
            arr[i, :7] = props
        else:
            arr[i, 0] = props[3]  # hydrophobicity
            arr[i, 1] = props[4]  # isoelectric
            arr[i, 2] = props[2]  # volume
    return arr


def generate_features_for_peptide(peptide_seq):
    """为任意新肽序列实时生成 CAMP 所需的全部 8 种特征（无需预计算字典）"""
    protein = TMC4_SEQUENCE
    prot_ss = ",".join(list(TMC4_SS))
    pep_ss = ",".join(["C"] * len(peptide_seq))

    return {
        "X_pep":       _label_sequence(peptide_seq, PAD_PEP),
        "X_prot":      _label_sequence(protein, PAD_PROT),
        "X_pep_ss":    _label_seq_ss(pep_ss, PAD_PEP, peptide_seq),
        "X_prot_ss":   _label_seq_ss(prot_ss, PAD_PROT, protein),
        "X_pep_2":     _label_physicochemical(peptide_seq, PAD_PEP),
        "X_prot_2":    _label_physicochemical(protein, PAD_PROT),
        "X_pep_dense": _get_dense_feature(peptide_seq, PAD_PEP, is_protein=False),
        "X_prot_dense": _get_dense_feature(protein, PAD_PROT, is_protein=True),
    }


# ==================== 特征字典加载 ==========================

def load_feature_dicts():
    """加载 8 个特征字典（可选，缺失时返回 None）"""
    task, name = "cls", "peptide"
    keys_and_paths = {
        "prot_seq":   f"preprocess_v2_salty/{task}_{name}_protein_feature_dict",
        "pep_seq":    f"preprocess_v2_salty/{task}_{name}_peptide_feature_dict",
        "prot_ss":    f"preprocess_v2_salty/{task}_{name}_protein_ss_feature_dict",
        "pep_ss":     f"preprocess_v2_salty/{task}_{name}_compound_ss_feature_dict",
        "prot_dense": f"preprocess_v2_salty/{task}_{name}_protein_dense_feature_dict",
        "pep_dense":  f"preprocess_v2_salty/{task}_{name}_compound_dense_feature_dict",
        "prot_2":     f"preprocess_v2_salty/{task}_{name}_protein_2_feature_dict",
        "pep_2":      f"preprocess_v2_salty_/{task}_{name}_compound_2_feature_dict",
    }
    dicts = {}
    for key, rel_path in keys_and_paths.items():
        full = os.path.join(SCRIPT_DIR, rel_path)
        if not os.path.exists(full):
            print(f"  ⚠️ 特征字典缺失，将使用实时生成模式")
            return None
        with open(full, "rb") as f:
            dicts[key] = pickle.load(f, encoding="latin1")
    print(f"  ✓ 已加载（字典中有 {len(dicts.get('pep_seq', {}))} 条已知序列）")
    return dicts


# ==================== 序列 → 特征字典条目 ====================

def prepare_sequence_features(peptide_seq, fd):
    """
    将一条肽序列转换为 CAMP 所需的 8 种特征。
    优先查特征字典，不在字典中则实时生成。
    """
    # 尝试从字典查表
    protein = TMC4_SEQUENCE
    prot_ss = ",".join(list(TMC4_SS))
    pep_ss = ",".join(["C"] * len(peptide_seq))

    if fd and peptide_seq in fd.get("pep_seq", {}):
        arrays = {}
        arrays["X_prot"] = fd["prot_seq"][protein]
        arrays["X_prot_ss"] = fd["prot_ss"][prot_ss]
        arrays["X_prot_2"] = fd["prot_2"][protein]
        arrays["X_prot_dense"] = fd["prot_dense"][protein]
        arrays["X_pep"] = fd["pep_seq"][peptide_seq]
        arrays["X_pep_ss"] = fd["pep_ss"].get(pep_ss, _label_seq_ss(pep_ss, PAD_PEP, peptide_seq))
        arrays["X_pep_2"] = fd["pep_2"].get(peptide_seq, _label_physicochemical(peptide_seq, PAD_PEP))
        arrays["X_pep_dense"] = fd["pep_dense"].get(peptide_seq, _get_dense_feature(peptide_seq, PAD_PEP))
        return arrays

    # 不在字典中 → 实时生成全部特征
    return generate_features_for_peptide(peptide_seq)


# ==================== CAMP 特征提取 ====================

def extract_camp_features_batch(models, batch_arrays, device):
    """用多个 CAMP 模型提取特征并取平均"""
    all_feats = []
    for model in models:
        model.eval()
        with torch.no_grad():
            tensors = {
                k: torch.from_numpy(v).float().to(device)
                for k, v in batch_arrays.items()
            }
            pep_cnn, prot_cnn, pep_att, prot_att = model.extract_features(
                tensors["X_pep"], tensors["X_prot"],
                tensors["X_pep_ss"], tensors["X_prot_ss"],
                tensors["X_pep_2"], tensors["X_prot_2"],
                tensors["X_pep_dense"], tensors["X_prot_dense"],
            )
            all_feats.append({
                "pep_cnn": pep_cnn.cpu().numpy(),
                "prot_cnn": prot_cnn.cpu().numpy(),
                "pep_att": pep_att.cpu().numpy(),
                "prot_att": prot_att.cpu().numpy(),
            })

    # 平均
    avg = {}
    for key in all_feats[0]:
        avg[key] = np.mean([f[key] for f in all_feats], axis=0)
    return avg


def build_interaction_features(feat):
    """与训练脚本一致的交互特征构造"""
    pc, rc = feat["pep_cnn"], feat["prot_cnn"]
    pa, ra = feat["pep_att"], feat["prot_att"]

    cnn_concat = np.concatenate([pc, rc], axis=1)
    cnn_product = pc * rc
    cnn_diff = np.abs(pc - rc)
    pn = np.linalg.norm(pc, axis=1, keepdims=True) + 1e-8
    rn = np.linalg.norm(rc, axis=1, keepdims=True) + 1e-8
    cnn_cosine = np.sum(pc * rc, axis=1, keepdims=True) / (pn * rn)

    att_concat = np.concatenate([pa, ra], axis=1)
    att_product = pa * ra

    stats = np.concatenate([
        pc.mean(axis=1, keepdims=True),
        pc.std(axis=1, keepdims=True),
        rc.mean(axis=1, keepdims=True),
        rc.std(axis=1, keepdims=True),
    ], axis=1)

    return np.concatenate([
        cnn_concat, cnn_product, cnn_diff, cnn_cosine,
        att_concat, att_product, stats,
    ], axis=1)


# ==================== 加载 CAMP 预训练模型 ====================

def load_camp_models(device, num_models=5):
    """加载 CAMP 预训练模型"""
    models = []
    for i in range(num_models):
        path = os.path.join(CAMP_DIR, f"model_full_ckpts_{i}.pkl")
        if not os.path.exists(path):
            continue
        model = CAMP()
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=True)
        for p in model.parameters():
            p.requires_grad = False
        models.append(model.to(device).eval())
    return models


# ==================== 主函数 ====================

def main():
    print("=" * 60)
    print("咸味肽预测 — CAMP CNN + ML 混合模型")
    print("=" * 60)

    # --- 解析参数 ---
    input_file = None
    model_path = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            model_path = args[i + 1]
            i += 2
        else:
            input_file = args[i]
            i += 1

    # --- 找输入文件 ---
    if input_file is None:
        for candidate in [ "测序结果.xlsx","测序结果_new.xlsx",
                          "salty_peptides_test.xlsx", "peptides.xlsx"]:
            if os.path.exists(os.path.join(SCRIPT_DIR, candidate)):
                input_file = os.path.join(SCRIPT_DIR, candidate)
                break
    if input_file and not os.path.isabs(input_file):
        input_file = os.path.join(SCRIPT_DIR, input_file)

    if input_file is None or not os.path.exists(input_file):
        print(f"❌ 找不到输入文件")
        print("   用法: python predict_camp_ml_hybrid.py <输入文件.xlsx>")
        sys.exit(1)
    print(f"📂 输入文件: {os.path.basename(input_file)}")

    # --- 找模型文件 ---
    if model_path is None:
        results_dir = os.path.join(SCRIPT_DIR, "camp_ml_results")
        candidates = sorted(glob.glob(os.path.join(results_dir, "camp_ml_hybrid_*.pkl")))
        if candidates:
            model_path = candidates[-1]  # 最新的
    if model_path and not os.path.isabs(model_path):
        model_path = os.path.join(SCRIPT_DIR, model_path)

    if model_path is None or not os.path.exists(model_path):
        print(f"❌ 找不到模型文件")
        print("   请先运行 train_camp_ml_hybrid.py 训练模型")
        sys.exit(1)
    print(f"📂 模型文件: {os.path.basename(model_path)}")

    # --- 加载 ML 模型 ---
    with open(model_path, "rb") as f:
        ckpt = pickle.load(f)
    ml_pipeline = ckpt["pipeline"]
    top_di = ckpt.get("top_dipeptides")
    top_tri = ckpt.get("top_tripeptides")
    top_cksaap = ckpt.get("top_cksaap")
    selected_indices = ckpt.get("selected_indices")
    threshold_mcc = ckpt.get("threshold_mcc", 0.5)
    threshold_f1 = ckpt.get("threshold_f1", 0.5)
    if selected_indices is not None:
        print(f"✅ ML Pipeline 已加载 (特征选择: {len(selected_indices)} 维)")
    else:
        print(f"✅ ML Pipeline 已加载 (无特征选择)")
    print(f"  阈值: MCC={threshold_mcc:.3f}, F1={threshold_f1:.3f}")

    # --- 设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 加载特征字典 ---
    print("\n🔄 加载特征字典...")
    fd = load_feature_dicts()

    # --- 加载 CAMP 模型 ---
    print("🔄 加载 CAMP 预训练模型...")
    camp_models = load_camp_models(device)
    print(f"  ✓ 使用 {len(camp_models)} 个 CAMP 模型")

    # --- 读取输入 ---
    print(f"\n📄 读取输入文件...")
    if input_file.endswith(".xlsx"):
        df = pd.read_excel(input_file)
    elif input_file.endswith(".csv"):
        df = pd.read_csv(input_file)
    else:
        df = pd.read_csv(input_file, sep="\t")

    # 找序列列
    if "sequence" not in df.columns:
        for col in df.columns:
            if "seq" in col.lower() or "peptide" in col.lower():
                df["sequence"] = df[col]
                break
    if "sequence" not in df.columns:
        print(f"❌ 无法找到序列列，当前列: {list(df.columns)}")
        sys.exit(1)

    peptide_list = df["sequence"].tolist()
    print(f"   包含 {len(peptide_list)} 条序列")

    # --- 提取特征 & 预测 ---
    print("🔄 提取 CAMP CNN 特征并预测...")
    valid_peptides = []
    all_X = []
    n_from_dict = 0
    n_generated = 0
    n_skipped = 0

    for pep in peptide_list:
        if not isinstance(pep, str) or not pep.strip():
            n_skipped += 1
            continue

        pep = pep.strip().upper()

        # 统计来源
        if fd and pep in fd.get("pep_seq", {}):
            n_from_dict += 1
        else:
            n_generated += 1

        feat_arrays = prepare_sequence_features(pep, fd)

        # CNN 交互特征
        batch = {k: v[np.newaxis, ...] for k, v in feat_arrays.items()}
        raw_feat = extract_camp_features_batch(camp_models, batch, device)
        cnn_feat = build_interaction_features(raw_feat)[0]

        # 传统序列特征
        trad_feat = extract_traditional_features(pep, top_di, top_tri, top_cksaap)

        # 拼接
        all_X.append(np.concatenate([cnn_feat, trad_feat]))
        valid_peptides.append(pep)

    if not valid_peptides:
        print("❌ 没有有效序列可预测")
        sys.exit(1)

    print(f"  ✓ 字典查表: {n_from_dict} 条, 实时生成: {n_generated} 条, 跳过: {n_skipped} 条")

    X = np.array(all_X)

    # 应用特征选择（与训练时一致）
    if selected_indices is not None:
        X = X[:, selected_indices]
        print(f"  ✓ 特征选择: {all_X[0].shape[0]} → {X.shape[1]} 维")

    probs = ml_pipeline.predict_proba(X)[:, 1]
    preds_default = ml_pipeline.predict(X)
    preds_mcc = (probs >= threshold_mcc).astype(int)
    preds_f1 = (probs >= threshold_f1).astype(int)

    # --- 结果 ---
    results = pd.DataFrame({
        "sequence": valid_peptides,
        "Predicted_Score": probs,
        "Is_Salty_default": ["YES" if p == 1 else "NO" for p in preds_default],
        "Is_Salty_MCC": ["YES" if p == 1 else "NO" for p in preds_mcc],
        "Is_Salty_F1": ["YES" if p == 1 else "NO" for p in preds_f1],
    })
    results = results.sort_values("Predicted_Score", ascending=False)

    # 保存
    results_dir = os.path.join(SCRIPT_DIR, "prediction_results")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(results_dir, f"camp_ml_hybrid_results_{ts}.csv")
    results.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print(f"预测完成! 结果保存至: {output_file}")
    print(f"{'='*60}")

    n_default = int(preds_default.sum())
    n_mcc = int(preds_mcc.sum())
    n_f1 = int(preds_f1.sum())
    print(f"\n📊 统计:")
    print(f"  默认阈值(0.5):   预测为咸味 {n_default} / {len(probs)}")
    print(f"  MCC阈值({threshold_mcc:.3f}): 预测为咸味 {n_mcc} / {len(probs)}")
    print(f"  F1阈值({threshold_f1:.3f}):  预测为咸味 {n_f1} / {len(probs)}")

    print(f"\n🏆 预测结果 (前20条, 使用MCC最优阈值 {threshold_mcc:.3f}):")
    for i, (_, row) in enumerate(results.iterrows()):
        if i >= 20:
            print(f"   ... 共 {len(results)} 条，完整结果见 CSV 文件")
            break
        status = "✅" if row["Is_Salty_MCC"] == "YES" else "❌"
        print(f"   {status} {row['sequence']:20} | 分数: {row['Predicted_Score']:.4f} | {row['Is_Salty_MCC']}")


if __name__ == "__main__":
    main()
