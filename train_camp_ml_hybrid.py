#!/usr/bin/env python3
"""
用法:
  cd salty_peptides_workflow
  python train_camp_ml_hybrid.py
"""

import os
import sys
import math
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

from imblearn.over_sampling import SMOTE, BorderlineSMOTE
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

from peptide_features import extract_features as extract_traditional_features
from peptide_features import select_top_ngrams, select_top_cksaap

# ==================== 1. CAMP 模型组件 ====================
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
    """CAMP 模型"""

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
            nn.Linear(640, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
        )
        self.att = Self_Attention(128, 128, 128)
        self.output = nn.Linear(512, 1)

    def forward(self, *args, **kwargs):
        return self.extract_features(*args, **kwargs)

    def extract_features(self, x_pep, x_prot, x_pep_ss, x_prot_ss,
                         x_pep_2, x_prot_2, x_pep_dense, x_prot_dense):
        """
        返回:
            pep_cnn  (batch, 192)  — 肽 CNN 池化特征
            prot_cnn (batch, 192)  — 蛋白质 CNN 池化特征
            pep_att  (batch, 128)  — 肽 Attention 池化特征
            prot_att (batch, 128)  — 蛋白质 Attention 池化特征
        """
        # Embedding
        pep_seq_emb = self.embed_seq(x_pep.long())
        prot_seq_emb = self.embed_seq(x_prot.long())
        pep_ss_emb = self.embed_ss(x_pep_ss.long())
        prot_ss_emb = self.embed_ss(x_prot_ss.long())
        pep_2_emb = self.embed_two(x_pep_2.long())
        prot_2_emb = self.embed_two(x_prot_2.long())
        pep_dense = self.pep_fc(x_pep_dense)
        prot_dense = self.prot_fc(x_prot_dense)

        # Concat → CNN
        enc_pep = torch.cat([pep_seq_emb, pep_ss_emb, pep_2_emb, pep_dense], dim=-1)
        enc_prot = torch.cat([prot_seq_emb, prot_ss_emb, prot_2_emb, prot_dense], dim=-1)

        enc_pep = self.pep_convs(enc_pep.permute(0, 2, 1)).permute(0, 2, 1)
        enc_prot = self.prot_convs(enc_prot.permute(0, 2, 1)).permute(0, 2, 1)

        pep_cnn = self.global_max_pooling(enc_pep)      # (batch, 192)
        prot_cnn = self.global_max_pooling(enc_prot)     # (batch, 192)

        # Attention 分支
        pep_att = self.global_max_pooling(
            self.att(self.embed_seq(x_pep.long()))
        )  # (batch, 128)
        prot_att = self.global_max_pooling(
            self.att(self.embed_seq(x_prot.long()))
        )  # (batch, 128)

        return pep_cnn, prot_cnn, pep_att, prot_att


# ==================== 2. 数据加载 ====================

def load_feature_dicts(base_dir):
    """加载 8 个特征字典（pickle 格式）"""
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
        full = os.path.join(base_dir, rel_path)
        if not os.path.exists(full):
            print(f"  ❌ 缺失: {full}")
            sys.exit(1)
        with open(full, "rb") as f:
            dicts[key] = pickle.load(f, encoding="latin1")
        print(f"  ✓ {key}: {len(dicts[key])} 条")
    return dicts


def load_tsv_data(tsv_path, fd):
    """
    读取 salty_peptides_*.tsv 并从特征字典中查找特征。
    TSV 列: protein_sequence  peptide_sequence  label  peptide_ss  protein_ss
    """
    arrays = {k: [] for k in [
        "X_pep", "X_prot", "X_pep_ss", "X_prot_ss",
        "X_pep_2", "X_prot_2", "X_pep_dense", "X_prot_dense",
    ]}
    Y = []
    peptide_seqs = []  # 保存肽序列用于传统特征提取
    skipped = 0
    with open(tsv_path, encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            protein, peptide, label, pep_ss, prot_ss = parts[:5]
            try:
                arrays["X_pep"].append(fd["pep_seq"][peptide])
                arrays["X_prot"].append(fd["prot_seq"][protein])
                arrays["X_pep_ss"].append(fd["pep_ss"][pep_ss])
                arrays["X_prot_ss"].append(fd["prot_ss"][prot_ss])
                arrays["X_pep_2"].append(fd["pep_2"][peptide])
                arrays["X_prot_2"].append(fd["prot_2"][protein])
                arrays["X_pep_dense"].append(fd["pep_dense"][peptide])
                arrays["X_prot_dense"].append(fd["prot_dense"][protein])
                Y.append(int(label))
                peptide_seqs.append(peptide)
            except KeyError:
                skipped += 1
    if skipped:
        print(f"  ⚠️ 跳过 {skipped} 条（特征字典缺失）")
    return {k: np.array(v) for k, v in arrays.items()}, np.array(Y), peptide_seqs


# ==================== 3. CNN 特征提取 ====================

def extract_camp_features(model, arrays, device, batch_size=64):
    """批量提取 CAMP CNN + Attention 特征"""
    model.eval()
    n = len(arrays["X_pep"])
    results = {"pep_cnn": [], "prot_cnn": [], "pep_att": [], "prot_att": []}

    with torch.no_grad():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            tensors = {
                k: torch.from_numpy(arrays[k][i:j]).float().to(device)
                for k in arrays
            }
            pep_cnn, prot_cnn, pep_att, prot_att = model.extract_features(
                tensors["X_pep"], tensors["X_prot"],
                tensors["X_pep_ss"], tensors["X_prot_ss"],
                tensors["X_pep_2"], tensors["X_prot_2"],
                tensors["X_pep_dense"], tensors["X_prot_dense"],
            )
            results["pep_cnn"].append(pep_cnn.cpu().numpy())
            results["prot_cnn"].append(prot_cnn.cpu().numpy())
            results["pep_att"].append(pep_att.cpu().numpy())
            results["prot_att"].append(prot_att.cpu().numpy())

    return {k: np.concatenate(v) for k, v in results.items()}


# ==================== 4. 交互特征构造 ====================

def build_interaction_features(feat):
    """
    从 CNN/Attention 特征构造丰富的交互特征。

    特征组成:
      - CNN concat          384 维  (192+192)
      - CNN 逐元素乘积      192 维  (肽⊙蛋白)
      - CNN 逐元素差异       192 维  |肽-蛋白|
      - CNN 余弦相似度         1 维
      - Attention concat    256 维  (128+128)
      - Attention 乘积      128 维
      - 统计量                4 维  (pep_mean, pep_std, prot_mean, prot_std)
    总计:                  1157 维
    """
    pc, rc, pa, ra = feat["pep_cnn"], feat["prot_cnn"], feat["pep_att"], feat["prot_att"]

    # CNN 交互
    cnn_concat = np.concatenate([pc, rc], axis=1)           # 384
    cnn_product = pc * rc                                   # 192
    cnn_diff = np.abs(pc - rc)                              # 192
    pn = np.linalg.norm(pc, axis=1, keepdims=True) + 1e-8
    rn = np.linalg.norm(rc, axis=1, keepdims=True) + 1e-8
    cnn_cosine = np.sum(pc * rc, axis=1, keepdims=True) / (pn * rn)  # 1

    # Attention 交互
    att_concat = np.concatenate([pa, ra], axis=1)           # 256
    att_product = pa * ra                                   # 128

    # 统计量
    stats = np.concatenate([
        pc.mean(axis=1, keepdims=True),
        pc.std(axis=1, keepdims=True),
        rc.mean(axis=1, keepdims=True),
        rc.std(axis=1, keepdims=True),
    ], axis=1)  # 4

    X = np.concatenate([
        cnn_concat, cnn_product, cnn_diff, cnn_cosine,
        att_concat, att_product, stats,
    ], axis=1)
    return X


# ==================== 5. ML 模型 & 评估 ====================

def _base_models():
    """返回 (short_name, estimator) 列表"""
    models = [
        ("RF", RandomForestClassifier(
            n_estimators=500, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1)),
        ("ET", ExtraTreesClassifier(
            n_estimators=500, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1)),
        ("GB", GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            min_samples_leaf=4, subsample=0.8, random_state=42)),
        ("SVM", SVC(
            kernel="rbf", C=10.0, gamma="scale",
            class_weight="balanced", probability=True, random_state=42)),
        ("KNN", KNeighborsClassifier(
            n_neighbors=7, weights="distance", metric="minkowski", p=2)),
        ("MLP", MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu",
            max_iter=500, early_stopping=True, validation_fraction=0.15,
            alpha=0.01, random_state=42)),
        ("XGB", xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, eval_metric="logloss", verbosity=0)),
        ("LGB", lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.03,
            min_child_samples=4, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1)),
    ]
    return models


def build_all_models():
    """
    构建模型组合:
      单模型: RF, ET, GB, SVM, KNN, MLP, XGB, LGB
      Soft Voting: 树模型投票, 全模型投票
      Stacking: 树+SVM+MLP stacking
    """
    base = _base_models()

    all_models = []
    # 单模型
    for n, m in base:
        all_models.append((n, clone(m)))

    # Soft Voting — 树模型 (RF+ET+GB)
    tree_estimators = [(n, clone(m)) for n, m in base if n in ("RF", "ET", "GB")]
    if len(tree_estimators) >= 2:
        all_models.append(("Vote-Tree", VotingClassifier(
            estimators=tree_estimators, voting="soft", n_jobs=-1)))

    # Soft Voting — 全部模型
    all_estimators = [(n, clone(m)) for n, m in base]
    if len(all_estimators) >= 3:
        all_models.append(("Vote-All", VotingClassifier(
            estimators=all_estimators, voting="soft", n_jobs=-1)))

    # Stacking — 树+SVM+MLP → LR meta
    stack_estimators = [(n, clone(m)) for n, m in base if n in ("RF", "ET", "GB", "SVM", "MLP")]
    if len(stack_estimators) >= 3:
        all_models.append(("Stack-5", StackingClassifier(
            estimators=stack_estimators,
            final_estimator=LogisticRegression(C=1.0, max_iter=1000,
                                               class_weight="balanced", random_state=42),
            cv=5, n_jobs=-1, passthrough=False,
        )))

    # Stacking — 全部模型 → LR meta
    if len(all_estimators) >= 5:
        all_models.append(("Stack-All", StackingClassifier(
            estimators=all_estimators,
            final_estimator=LogisticRegression(C=1.0, max_iter=1000,
                                               class_weight="balanced", random_state=42),
            cv=5, n_jobs=-1, passthrough=False,
        )))

    return all_models


def compute_metrics(y_true, y_pred, y_prob):
    """计算论文所需的 6 个指标"""
    return {
        "SN":    recall_score(y_true, y_pred) * 100,
        "PRE":   precision_score(y_true, y_pred) * 100,
        "ACC":   accuracy_score(y_true, y_pred) * 100,
        "MCC":   matthews_corrcoef(y_true, y_pred),
        "F1":    f1_score(y_true, y_pred),
        "AUROC": roc_auc_score(y_true, y_prob),
    }


def _run_cv(X, Y, n_splits=5):
    """5-fold CV，返回 [(model_name, avg_metrics), ...]"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_models = build_all_models()
    results = []

    for name, mdl in all_models:
        fold_metrics = []
        for tr_idx, va_idx in skf.split(X, Y):
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clone(mdl))])
            pipe.fit(X[tr_idx], Y[tr_idx])
            y_pred = pipe.predict(X[va_idx])
            y_prob = pipe.predict_proba(X[va_idx])[:, 1]
            fold_metrics.append(compute_metrics(Y[va_idx], y_pred, y_prob))

        avg = {k: np.mean([m[k] for m in fold_metrics]) for k in fold_metrics[0]}
        results.append((name, avg))
    return results


def _run_independent_test(X_train, Y_train, X_test, Y_test):
    """独立测试集评估，返回 [(model_name, metrics), ...]"""
    all_models = build_all_models()
    results = []
    best_pipe = None
    best_auroc = -1

    for name, mdl in all_models:
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clone(mdl))])
        pipe.fit(X_train, Y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        m = compute_metrics(Y_test, y_pred, y_prob)
        results.append((name, m))

        if m["AUROC"] > best_auroc:
            best_auroc = m["AUROC"]
            best_pipe = pipe

    return results, best_pipe


def print_results_table(cv_results, test_results=None):
    """以论文风格打印结果对比表"""
    header = f"{'Cross-Validation':<20} {'Factors':<24} {'SN':>7} {'PRE':>7} {'ACC':>7} {'MCC':>8} {'F1':>8} {'AUROC':>8}"
    sep = "-" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    # 5-fold CV
    for idx, (name, m) in enumerate(cv_results):
        label = "5-fold CV" if idx == 0 else ""
        print(f"{label:<20} {name:<24} {m['SN']:>7.2f} {m['PRE']:>7.2f} {m['ACC']:>7.2f} {m['MCC']:>8.4f} {m['F1']:>8.4f} {m['AUROC']:>8.4f}")

    if test_results:
        print(sep)
        for idx, (name, m) in enumerate(test_results):
            label = "Independent test" if idx == 0 else ""
            print(f"{label:<20} {name:<24} {m['SN']:>7.2f} {m['PRE']:>7.2f} {m['ACC']:>7.2f} {m['MCC']:>8.4f} {m['F1']:>8.4f} {m['AUROC']:>8.4f}")

    print(sep)


# ==================== 6. 加载预训练 CAMP 模型 ====================

def load_pretrained_camp(model_path, device):
    """加载预训练 CAMP 权重"""
    model = CAMP()
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=True)
    print(f"  ✓ 加载预训练权重: {os.path.basename(model_path)}")
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device).eval()


# ==================== 7. 多模型集成特征提取（可选） ====================

def extract_ensemble_features(model_dir, arrays, device, num_models=5):
    """
    用 5 个预训练模型分别提取特征，取平均 → 更稳定的表示。
    """
    all_feats = []
    loaded = 0
    for i in range(num_models):
        path = os.path.join(model_dir, f"model_full_ckpts_{i}.pkl")
        if not os.path.exists(path):
            continue
        model = load_pretrained_camp(path, device)
        feat = extract_camp_features(model, arrays, device)
        all_feats.append(feat)
        loaded += 1

    if loaded == 0:
        return None

    # 平均各模型的特征
    avg_feat = {}
    for key in all_feats[0]:
        avg_feat[key] = np.mean([f[key] for f in all_feats], axis=0)

    print(f"  ✓ 使用 {loaded} 个模型的平均特征")
    return avg_feat


# ==================== 8. 特征选择 ====================

def select_features(X_train, Y_train, X_test=None, max_features=300):
    """
    三步特征筛选:
      1. 方差过滤 (去掉方差接近0的无用特征)
      2. 相关性去冗余 (相关系数>0.95的只保留一个)
      3. 互信息排序 (选top-K最有区分力的特征)

    返回: X_train_sel, X_test_sel (or None), selected_indices
    """
    n_orig = X_train.shape[1]
    keep_mask = np.ones(n_orig, dtype=bool)

    # Step 1: 方差过滤
    vt = VarianceThreshold(threshold=1e-6)
    vt.fit(X_train)
    var_mask = vt.get_support()
    keep_mask &= var_mask
    n_after_var = var_mask.sum()
    print(f"    方差过滤: {n_orig} → {n_after_var} 维 (去掉 {n_orig - n_after_var} 个常量/准常量特征)")

    # Step 2: 相关性去冗余
    X_var = X_train[:, keep_mask]
    # 标准化后计算相关性
    from sklearn.preprocessing import StandardScaler as _SS
    X_scaled = _SS().fit_transform(X_var)
    corr = np.abs(np.corrcoef(X_scaled.T))
    np.fill_diagonal(corr, 0)

    # 贪心去除：对于高相关特征对，移除与标签互信息更低的那个
    mi_scores = mutual_info_classif(X_var, Y_train, random_state=42)
    var_indices = np.where(keep_mask)[0]
    to_remove = set()
    n_var = X_var.shape[1]
    for i in range(n_var):
        if i in to_remove:
            continue
        for j in range(i + 1, n_var):
            if j in to_remove:
                continue
            if corr[i, j] > 0.95:
                # 移除互信息更低的
                if mi_scores[i] < mi_scores[j]:
                    to_remove.add(i)
                    break
                else:
                    to_remove.add(j)

    corr_keep = [idx for idx in range(n_var) if idx not in to_remove]
    n_after_corr = len(corr_keep)
    print(f"    相关性去冗余: {n_var} → {n_after_corr} 维 (去掉 {n_var - n_after_corr} 个高度相关特征)")

    # 更新保留索引
    selected_indices = var_indices[corr_keep]

    # Step 3: 互信息 top-K
    X_corr = X_train[:, selected_indices]
    k = min(max_features, X_corr.shape[1])
    if X_corr.shape[1] > k:
        mi = mutual_info_classif(X_corr, Y_train, random_state=42)
        top_k_idx = np.argsort(mi)[-k:]
        top_k_idx = np.sort(top_k_idx)
        selected_indices = selected_indices[top_k_idx]
        print(f"    互信息 top-K: {n_after_corr} → {k} 维")
    else:
        print(f"    互信息: 保留全部 {X_corr.shape[1]} 维 (< max_features={max_features})")

    X_train_sel = X_train[:, selected_indices]
    X_test_sel = X_test[:, selected_indices] if X_test is not None else None

    print(f"    ✓ 最终特征: {n_orig} → {len(selected_indices)} 维")
    return X_train_sel, X_test_sel, selected_indices


# ==================== 9. 主函数 ====================

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- 路径配置 ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    CAMP_DIR = os.path.dirname(SCRIPT_DIR)  # CAMP_pytorch/
    TRAIN_TSV = os.path.join(SCRIPT_DIR, "salty_peptides_training.tsv")
    TEST_TSV = os.path.join(SCRIPT_DIR, "salty_peptides_test.tsv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # [1/8] 加载特征字典
    print("\n[1/8] 加载特征字典...")
    if not os.path.exists(TRAIN_TSV):
        print(f"❌ 找不到训练数据: {TRAIN_TSV}")
        print("   请先运行: python prepare_salty_peptides.py peptides.xlsx")
        sys.exit(1)
    fd = load_feature_dicts(SCRIPT_DIR)

    # [2/8] 加载数据
    print(f"\n[2/8] 加载数据...")
    train_arrays, Y_train, train_pep_seqs = load_tsv_data(TRAIN_TSV, fd)
    n_pos = int(Y_train.sum())
    print(f"  训练集: {len(Y_train)} 条 (阳性={n_pos}, 阴性={len(Y_train)-n_pos})")

    has_test = os.path.exists(TEST_TSV)
    test_pep_seqs = []
    if has_test:
        test_arrays, Y_test, test_pep_seqs = load_tsv_data(TEST_TSV, fd)
        n_pos_t = int(Y_test.sum())
        print(f"  测试集: {len(Y_test)} 条 (阳性={n_pos_t}, 阴性={len(Y_test)-n_pos_t})")

    # [3/8] 加载 CAMP 预训练模型 & 提取特征
    print(f"\n[3/8] 加载 CAMP 预训练模型并提取特征...")

    train_feat = extract_ensemble_features(CAMP_DIR, train_arrays, device)
    if has_test:
        test_feat = extract_ensemble_features(CAMP_DIR, test_arrays, device)

    # [4/8] 构造混合特征（CNN交互特征 + 传统序列特征）
    print(f"\n[4/8] 构造混合特征...")
    X_cnn_train = build_interaction_features(train_feat)
    print(f"  CNN 交互特征: {X_cnn_train.shape[1]} 维")

    # 传统序列特征
    print("  提取传统序列特征...")
    top_di = select_top_ngrams(train_pep_seqs, n=2, top_k=20)
    top_tri = select_top_ngrams(train_pep_seqs, n=3, top_k=10)
    top_cksaap = select_top_cksaap(train_pep_seqs, k_list=(1, 2, 3), top_k=15)
    X_trad_train = np.array([
        extract_traditional_features(s, top_di, top_tri, top_cksaap)
        for s in train_pep_seqs
    ])
    print(f"  传统序列特征: {X_trad_train.shape[1]} 维")

    X_train_raw = np.concatenate([X_cnn_train, X_trad_train], axis=1)
    print(f"  混合特征总计: {X_train_raw.shape[1]} 维")

    X_test_raw = None
    if has_test:
        X_cnn_test = build_interaction_features(test_feat)
        X_trad_test = np.array([
            extract_traditional_features(s, top_di, top_tri, top_cksaap)
            for s in test_pep_seqs
        ])
        X_test_raw = np.concatenate([X_cnn_test, X_trad_test], axis=1)
        print(f"  测试集混合特征: {X_test_raw.shape}")

    # [5/8] 特征选择
    print(f"\n[5/8] 特征选择 (方差过滤 → 相关性去冗余 → 互信息 top-K)...")
    X_train, X_test_sel, selected_indices = select_features(
        X_train_raw, Y_train, X_test_raw, max_features=300)
    if has_test and X_test_sel is not None:
        X_test = X_test_sel

    # [6/9] 数据增强 (SMOTE + Gaussian Noise)
    print(f"\n[6/9] 数据增强...")
    n_before = len(Y_train)

    smote = BorderlineSMOTE(random_state=42, k_neighbors=min(5, min(
        np.sum(Y_train == 0), np.sum(Y_train == 1)) - 1))
    X_smote, Y_smote = smote.fit_resample(X_train, Y_train)
    print(f"  SMOTE: {n_before} → {len(Y_smote)} 条 (类别平衡)")

    # 6b. Gaussian Noise 增强 — 对每个样本生成多个噪声副本
    rng = np.random.RandomState(42)
    noise_copies = 3  # 每个样本生成 3 个噪声副本
    feature_std = np.std(X_smote, axis=0) + 1e-8
    noise_scale = 0.05  # 5% 标准差的噪声
    X_noise_list = [X_smote]
    Y_noise_list = [Y_smote]
    for _ in range(noise_copies):
        noise = rng.normal(0, noise_scale, X_smote.shape) * feature_std[np.newaxis, :]
        X_noise_list.append(X_smote + noise)
        Y_noise_list.append(Y_smote.copy())
    X_train_for_model = np.vstack(X_noise_list)
    Y_train_for_model = np.concatenate(Y_noise_list)
    n_pos_final = int(Y_train_for_model.sum())
    print(f"  Gaussian Noise ×{noise_copies}: {len(Y_smote)} → {len(Y_train_for_model)} 条")
    print(f"  最终: 阳性={n_pos_final}, 阴性={len(Y_train_for_model)-n_pos_final}")

    # [7/9] 评估
    print(f"\n[7/9] 评估模型...")
    print(f"  CV 使用原始数据 ({len(Y_train)} 条)")
    print(f"  独立测试用增强数据训练 ({len(Y_train_for_model)} 条)")
    cv_results = _run_cv(X_train, Y_train)

    test_results = None
    best_pipe = None
    if has_test:
        test_results, best_pipe = _run_independent_test(
            X_train_for_model, Y_train_for_model, X_test, Y_test)

    print_results_table(cv_results, test_results)

    # [8/9] OOF 阈值优化
    print(f"\n[8/9] OOF 阈值优化 (寻找最佳分类阈值)...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    if best_pipe is not None:
        oof_probs = np.zeros(len(Y_train))
        for tr_idx, va_idx in skf.split(X_train, Y_train):
            pipe_clone = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", clone(best_pipe.named_steps["clf"]))
            ])
            pipe_clone.fit(X_train[tr_idx], Y_train[tr_idx])
            oof_probs[va_idx] = pipe_clone.predict_proba(X_train[va_idx])[:, 1]

        # 搜索最佳 MCC 阈值
        best_t, best_mcc = 0.5, -1.0
        for t in np.linspace(0.2, 0.8, 601):
            y_pred_t = (oof_probs >= t).astype(int)
            mcc_t = matthews_corrcoef(Y_train, y_pred_t)
            if mcc_t > best_mcc:
                best_mcc = mcc_t
                best_t = float(t)

        # 搜索最佳 F1 阈值
        best_t_f1, best_f1 = 0.5, -1.0
        for t in np.linspace(0.2, 0.8, 601):
            y_pred_t = (oof_probs >= t).astype(int)
            f1_t = f1_score(Y_train, y_pred_t, zero_division=0)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_t_f1 = float(t)

        print(f"  最佳 MCC 阈值: {best_t:.3f} (OOF MCC={best_mcc:.4f})")
        print(f"  最佳 F1 阈值:  {best_t_f1:.3f} (OOF F1={best_f1:.4f})")

        # 用最佳阈值在测试集上重新评估
        if has_test:
            y_prob_test = best_pipe.predict_proba(X_test)[:, 1]
            for thr_name, thr in [("MCC", best_t), ("F1", best_t_f1)]:
                y_pred_opt = (y_prob_test >= thr).astype(int)
                m = compute_metrics(Y_test, y_pred_opt, y_prob_test)
                print(f"  {thr_name}阈值({thr:.3f}) 测试集: ACC={m['ACC']:.2f}% MCC={m['MCC']:.4f} F1={m['F1']:.4f} AUROC={m['AUROC']:.4f}")
    else:
        best_t = 0.5
        best_t_f1 = 0.5

    # [9/9] 保存
    out_dir = os.path.join(SCRIPT_DIR, "camp_ml_results")
    os.makedirs(out_dir, exist_ok=True)

    if best_pipe is not None:
        model_save = os.path.join(out_dir, f"camp_ml_hybrid_{ts}.pkl")
        with open(model_save, "wb") as f:
            pickle.dump({
                "pipeline": best_pipe,
                "timestamp": ts,
                "top_dipeptides": top_di,
                "top_tripeptides": top_tri,
                "top_cksaap": top_cksaap,
                "selected_indices": selected_indices,
                "threshold_mcc": best_t,
                "threshold_f1": best_t_f1,
            }, f)
        print(f"\n✓ 最佳模型已保存: {model_save}")
        print(f"  特征选择: {X_train_raw.shape[1]} → {len(selected_indices)} 维")
        print(f"  数据增强: {n_before} → {len(Y_train_for_model)} 条")
        print(f"  最佳阈值: MCC={best_t:.3f}, F1={best_t_f1:.3f}")

    print("完成！")


if __name__ == "__main__":
    main()
