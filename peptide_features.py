"""
肽序列特征提取 - 共享模块
供 train_salty_ml.py 和 predict_salty_ml.py 共同使用
"""

import numpy as np
from collections import Counter


AA_LIST = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

# 氨基酸理化性质 (7维)
AA_PROPS = {
    'A': [1.8, 0.0, 89.1, 6.0, 0.0, 0.0, 0.0],
    'C': [2.5, 0.0, 121.2, 5.1, 0.0, 1.0, 0.0],
    'D': [-3.5, -1.0, 133.1, 2.8, 1.0, 0.0, 0.0],
    'E': [-3.5, -1.0, 147.1, 3.2, 1.0, 0.0, 0.0],
    'F': [2.8, 0.0, 165.2, 5.5, 0.0, 0.0, 1.0],
    'G': [-0.4, 0.0, 75.1, 6.0, 0.0, 0.0, 0.0],
    'H': [-3.2, 1.0, 155.2, 7.6, 0.0, 0.0, 1.0],
    'I': [4.5, 0.0, 131.2, 6.0, 0.0, 0.0, 0.0],
    'K': [-3.9, 1.0, 146.2, 9.7, 0.0, 0.0, 0.0],
    'L': [3.8, 0.0, 131.2, 6.0, 0.0, 0.0, 0.0],
    'M': [1.9, 0.0, 149.2, 5.7, 0.0, 1.0, 0.0],
    'N': [-3.5, 0.0, 132.1, 5.4, 0.0, 0.0, 0.0],
    'P': [-1.6, 0.0, 115.1, 6.3, 0.0, 0.0, 0.0],
    'Q': [-3.5, 0.0, 146.2, 5.7, 0.0, 0.0, 0.0],
    'R': [-4.5, 1.0, 174.2, 10.8, 0.0, 0.0, 0.0],
    'S': [-0.8, 0.0, 105.1, 5.7, 0.0, 0.0, 0.0],
    'T': [-0.7, 0.0, 119.1, 5.6, 0.0, 0.0, 0.0],
    'V': [4.2, 0.0, 117.1, 6.0, 0.0, 0.0, 0.0],
    'W': [-0.9, 0.0, 204.2, 5.9, 0.0, 0.0, 1.0],
    'Y': [-1.3, 0.0, 181.2, 5.7, 0.0, 0.0, 1.0],
    'X': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

# 同义氨基酸组（理化性质相似的氨基酸可以互换）
SIMILAR_AA = {
    'A': ['G', 'V'],
    'V': ['A', 'I', 'L'],
    'I': ['V', 'L', 'M'],
    'L': ['I', 'V', 'M'],
    'M': ['L', 'I'],
    'F': ['Y', 'W'],
    'Y': ['F', 'W'],
    'W': ['F', 'Y'],
    'S': ['T'],
    'T': ['S'],
    'N': ['Q'],
    'Q': ['N'],
    'D': ['E'],
    'E': ['D'],
    'K': ['R'],
    'R': ['K'],
    'H': ['K', 'R'],
    'C': [],
    'G': ['A'],
    'P': [],
}


def select_top_ngrams(sequences, n=2, top_k=20):
    """
    从训练序列中统计最常见的 n-gram（数据驱动选择）
    
    参数:
        sequences: 训练集肽序列列表
        n: n-gram 的 n（2=二肽, 3=三肽）
        top_k: 选择最频繁的 k 个
    返回:
        排序后的 top_k 个 n-gram 列表
    """
    counter = Counter()
    for seq in sequences:
        seq = seq.upper().strip()
        for i in range(len(seq) - n + 1):
            ngram = seq[i:i+n]
            if all(aa in AA_LIST for aa in ngram):
                counter[ngram] += 1
    return [ng for ng, _ in counter.most_common(top_k)]


def select_top_cksaap(sequences, k_list=(1, 2, 3), top_k=15):
    """
    从训练序列中统计最常见的 k-spaced 氨基酸对（CKSAAP）
    
    参数:
        sequences: 训练集肽序列列表
        k_list: 间隔 k 值列表（k=0 即二肽，已单独处理，这里默认 1,2,3）
        top_k: 每个 k 值选择最频繁的 top_k 个
    返回:
        dict: {k: [pair1, pair2, ...], ...}
    """
    result = {}
    for k in k_list:
        counter = Counter()
        for seq in sequences:
            seq = seq.upper().strip()
            for i in range(len(seq) - k - 1):
                aa1 = seq[i]
                aa2 = seq[i + k + 1]
                if aa1 in AA_LIST and aa2 in AA_LIST:
                    counter[f"{aa1}{aa2}"] += 1
        result[k] = [pair for pair, _ in counter.most_common(top_k)]
    return result


def extract_features(sequence, top_dipeptides=None, top_tripeptides=None, top_cksaap=None):
    """
    从肽序列提取特征向量
    
    参数:
        sequence: 氨基酸序列字符串
        top_dipeptides: 二肽列表（数据驱动）。为 None 时使用默认列表。
        top_tripeptides: 三肽列表（数据驱动）。为 None 时使用默认列表。
    """
    sequence = sequence.upper().strip()
    features = []
    
    # 默认 n-gram / CKSAAP 列表（向后兼容）
    if top_dipeptides is None:
        top_dipeptides = ['LL', 'AA', 'VV', 'GG', 'SS', 'PP', 'LA', 'AL', 'GL', 'LG',
                          'VA', 'AV', 'VL', 'LV', 'GA', 'AG', 'SL', 'LS', 'PL', 'LP']
    if top_tripeptides is None:
        top_tripeptides = ['AAA', 'GGG', 'LLL', 'VVV', 'SSS',
                           'ALA', 'GAL', 'LAL', 'VAL', 'SAL']
    
    # 1. 序列长度
    length = len(sequence)
    features.append(length)
    
    # 2. 氨基酸组成 (20维)
    aa_count = {aa: 0 for aa in AA_LIST}
    for aa in sequence:
        if aa in aa_count:
            aa_count[aa] += 1
    aa_freq = [aa_count[aa] / max(length, 1) for aa in AA_LIST]
    features.extend(aa_freq)
    
    # 3. 理化性质统计 (7x4=28维: 均值、标准差、最小、最大)
    props_matrix = []
    for aa in sequence:
        if aa in AA_PROPS:
            props_matrix.append(AA_PROPS[aa])
    
    if props_matrix:
        props_arr = np.array(props_matrix)
        for i in range(7):
            col = props_arr[:, i]
            features.extend([np.mean(col), np.std(col), np.min(col), np.max(col)])
    else:
        features.extend([0.0] * 28)
    
    # 4. 二肽组成（数据驱动 top_k）
    dipeptide_count = {}
    for i in range(len(sequence) - 1):
        dp = sequence[i:i+2]
        if dp[0] in AA_LIST and dp[1] in AA_LIST:
            dipeptide_count[dp] = dipeptide_count.get(dp, 0) + 1
    
    total_dp = max(sum(dipeptide_count.values()), 1)
    for dp in top_dipeptides:
        features.append(dipeptide_count.get(dp, 0) / total_dp)
    
    # 5. 电荷特征
    pos_count = sum(1 for aa in sequence if aa in 'KRH')
    neg_count = sum(1 for aa in sequence if aa in 'DE')
    features.append(pos_count / max(length, 1))
    features.append(neg_count / max(length, 1))
    features.append((pos_count - neg_count) / max(length, 1))
    
    # 6. 疏水性特征
    hydrophobic_count = sum(1 for aa in sequence if aa in 'AILMFVPW')
    hydrophilic_count = sum(1 for aa in sequence if aa in 'RNDQEHKST')
    features.append(hydrophobic_count / max(length, 1))
    features.append(hydrophilic_count / max(length, 1))
    
    # 7. 芳香族比例
    aromatic_count = sum(1 for aa in sequence if aa in 'FWY')
    features.append(aromatic_count / max(length, 1))
    
    # 8. 末端氨基酸理化性质 (14维) + 电荷标记 (4维) = 18维
    n_terminal = sequence[0] if len(sequence) > 0 else 'X'
    c_terminal = sequence[-1] if len(sequence) > 0 else 'X'
    
    n_props = AA_PROPS.get(n_terminal, AA_PROPS['X'])
    features.extend(n_props)
    
    c_props = AA_PROPS.get(c_terminal, AA_PROPS['X'])
    features.extend(c_props)
    
    features.append(1.0 if n_terminal in 'KRH' else 0.0)
    features.append(1.0 if n_terminal in 'DE' else 0.0)
    features.append(1.0 if c_terminal in 'KRH' else 0.0)
    features.append(1.0 if c_terminal in 'DE' else 0.0)
    
    # 9. 三肽组成（数据驱动 top_k）
    tripeptide_count = {}
    for i in range(len(sequence) - 2):
        tp = sequence[i:i+3]
        if all(aa in AA_LIST for aa in tp):
            tripeptide_count[tp] = tripeptide_count.get(tp, 0) + 1
    
    total_tp = max(sum(tripeptide_count.values()), 1)
    for tp in top_tripeptides:
        features.append(tripeptide_count.get(tp, 0) / total_tp)
    
    # 10. 序列复杂性 (Shannon熵)
    aa_counts = [sequence.count(aa) for aa in AA_LIST]
    aa_probs = [c / len(sequence) for c in aa_counts if c > 0]
    entropy = -sum(p * np.log2(p) for p in aa_probs) if aa_probs else 0.0
    features.append(entropy)
    
    # 11. CKSAAP — k-spaced 氨基酸对组成 (捕获远距离残基关联)
    if top_cksaap is None:
        top_cksaap = {}  # 向后兼容：无 CKSAAP 时不增加特征
    for k in sorted(top_cksaap.keys()):
        pairs = top_cksaap[k]
        pair_count = {}
        for i in range(len(sequence) - k - 1):
            aa1 = sequence[i]
            aa2 = sequence[i + k + 1]
            if aa1 in AA_LIST and aa2 in AA_LIST:
                p = f"{aa1}{aa2}"
                pair_count[p] = pair_count.get(p, 0) + 1
        total = max(sum(pair_count.values()), 1)
        for p in pairs:
            features.append(pair_count.get(p, 0) / total)
    
    # 12. PseAAC — 伪氨基酸组成 (捕获序列顺序信息)
    #     使用 疏水性(prop0) 和 等电点(prop3) 两个理化性质，lambda=5
    pseaac_lambda = min(5, length - 1) if length > 1 else 0
    for lag in range(1, pseaac_lambda + 1):
        corr_hydro = 0.0
        corr_pi = 0.0
        count = 0
        for i in range(length - lag):
            aa_i = sequence[i]
            aa_j = sequence[i + lag]
            pi = AA_PROPS.get(aa_i, AA_PROPS['X'])
            pj = AA_PROPS.get(aa_j, AA_PROPS['X'])
            corr_hydro += (pi[0] - pj[0]) ** 2  # 疏水性差异
            corr_pi += (pi[3] - pj[3]) ** 2      # 等电点差异
            count += 1
        if count > 0:
            features.append(corr_hydro / count)
            features.append(corr_pi / count)
        else:
            features.extend([0.0, 0.0])
    # 如果序列太短不足 5 个 lag，用 0 补齐
    for _ in range(pseaac_lambda + 1, 6):
        features.extend([0.0, 0.0])

    # ============================================================
    # 13. CTD — Composition, Transition, Distribution (63维)
    # ============================================================
    # 7类氨基酸分组 (按理化性质)
    CTD_GROUPS = {
        1: set('AVG'),       # 小脂肪族
        2: set('ILMF'),      # 大脂肪族/芳香
        3: set('YW'),        # 大芳香
        4: set('KRH'),       # 正电荷
        5: set('DE'),        # 负电荷
        6: set('STNQ'),      # 极性不带电
        7: set('CP'),        # 特殊
    }

    def _aa_to_group(aa):
        for g, aas in CTD_GROUPS.items():
            if aa in aas:
                return g
        return 0

    group_seq = [_aa_to_group(aa) for aa in sequence]

    # C: Composition (7维) — 每组占比
    for g in range(1, 8):
        features.append(sum(1 for x in group_seq if x == g) / max(length, 1))

    # T: Transition (21维) — 相邻残基属于不同组的转移频率
    transitions = {}
    for i in range(len(group_seq) - 1):
        g1, g2 = group_seq[i], group_seq[i + 1]
        if g1 > g2:
            g1, g2 = g2, g1
        key = (g1, g2)
        transitions[key] = transitions.get(key, 0) + 1
    n_trans = max(len(group_seq) - 1, 1)
    for g1 in range(1, 8):
        for g2 in range(g1, 8):
            features.append(transitions.get((g1, g2), 0) / n_trans)

    # D: Distribution (35维) — 每组首次/25%/50%/75%/末次出现的位置
    for g in range(1, 8):
        positions = [i for i, x in enumerate(group_seq) if x == g]
        if positions:
            n_pos = len(positions)
            features.append(positions[0] / max(length - 1, 1))                      # 首次
            features.append(positions[max(0, n_pos // 4 - 1)] / max(length - 1, 1))  # 25%
            features.append(positions[max(0, n_pos // 2 - 1)] / max(length - 1, 1))  # 50%
            features.append(positions[max(0, 3 * n_pos // 4 - 1)] / max(length - 1, 1))  # 75%
            features.append(positions[-1] / max(length - 1, 1))                      # 末次
        else:
            features.extend([0.0] * 5)

    # ============================================================
    # 14. 位置加权特征 (30维)
    # ============================================================
    # 前3个和后3个残基的理化性质 (各3×7=21维) + 位置电荷标记 (9维)
    for pos_idx in range(3):
        if pos_idx < length:
            aa = sequence[pos_idx]
            features.extend(AA_PROPS.get(aa, AA_PROPS['X']))
        else:
            features.extend([0.0] * 7)

    for pos_idx in range(3):
        idx = length - 3 + pos_idx
        if 0 <= idx < length:
            aa = sequence[idx]
            features.extend(AA_PROPS.get(aa, AA_PROPS['X']))
        else:
            features.extend([0.0] * 7)

    # 前3个/后3个的电荷特征
    for pos_idx in range(3):
        if pos_idx < length:
            aa = sequence[pos_idx]
            features.append(1.0 if aa in 'KRH' else (-1.0 if aa in 'DE' else 0.0))
        else:
            features.append(0.0)

    for pos_idx in range(3):
        idx = length - 3 + pos_idx
        if 0 <= idx < length:
            aa = sequence[idx]
            features.append(1.0 if aa in 'KRH' else (-1.0 if aa in 'DE' else 0.0))
        else:
            features.append(0.0)

    # ============================================================
    # 15. 全局理化属性 (5维): MW, pI估算, GRAVY, 不稳定指数, 脂肪族指数
    # ============================================================
    AA_MW = {
        'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2,
        'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2,
        'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.2, 'R': 174.2,
        'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2,
    }
    # 分子量
    mw = sum(AA_MW.get(aa, 0.0) for aa in sequence) - (length - 1) * 18.02
    features.append(mw / 1000.0)  # 归一化到千道尔顿

    # GRAVY (Grand Average of Hydropathy)
    hydro_vals = [AA_PROPS.get(aa, AA_PROPS['X'])[0] for aa in sequence]
    gravy = np.mean(hydro_vals) if hydro_vals else 0.0
    features.append(gravy)

    # pI 粗略估算: 基于带电残基计数
    n_K = sequence.count('K')
    n_R = sequence.count('R')
    n_H = sequence.count('H')
    n_D = sequence.count('D')
    n_E = sequence.count('E')
    # 简化pI: 正电荷越多pI越高
    pi_approx = 7.0 + 0.5 * (n_K + n_R + 0.5 * n_H - n_D - n_E) / max(length, 1)
    features.append(pi_approx)

    # 脂肪族指数: Ikai (1980)
    x_A = sequence.count('A') / max(length, 1)
    x_V = sequence.count('V') / max(length, 1)
    x_I = sequence.count('I') / max(length, 1)
    x_L = sequence.count('L') / max(length, 1)
    aliphatic_idx = (x_A + 2.9 * x_V + 3.9 * (x_I + x_L)) * 100
    features.append(aliphatic_idx / 100.0)  # 归一化

    # 不稳定指数 (Guruprasad et al.): 简化版 — 用二肽不稳定权重
    # 使用序列中带电-带电和极性-非极性交替的频率作为近似
    instab_pairs = 0
    for i in range(len(sequence) - 1):
        a1, a2 = sequence[i], sequence[i + 1]
        # 不稳定二肽模式: 带电+疏水, 疏水+带电
        if (a1 in 'DEKRH' and a2 in 'AILMFVPW') or (a1 in 'AILMFVPW' and a2 in 'DEKRH'):
            instab_pairs += 1
    features.append(instab_pairs / max(length - 1, 1))

    # ============================================================
    # 16. 咸味感知相关特征 (20维)
    # ============================================================
    # 咸味肽研究表明：带电残基(尤其 D, E, K, R)、序列长度、
    # 疏水-亲水平衡、特定二肽模式与咸味感知密切相关

    # 16.1 带电残基详细比例 (5维)
    features.append(n_D / max(length, 1))   # Asp 比例
    features.append(n_E / max(length, 1))   # Glu 比例
    features.append(n_K / max(length, 1))   # Lys 比例
    features.append(n_R / max(length, 1))   # Arg 比例
    total_charged = n_D + n_E + n_K + n_R + n_H
    features.append(total_charged / max(length, 1))  # 总带电比

    # 16.2 咸味相关二肽模式 (8维)
    # 文献中与咸味相关的二肽: DK, EK, KD, KE, DE, ED, DD, EE
    salty_dipeptides = ['DK', 'EK', 'KD', 'KE', 'DE', 'ED', 'DD', 'EE']
    for dp in salty_dipeptides:
        count = 0
        for i in range(len(sequence) - 1):
            if sequence[i:i+2] == dp:
                count += 1
        features.append(count / max(length - 1, 1))

    # 16.3 电荷密度和分布 (4维)
    # 正电荷密度（每个残基的平均正电荷）
    features.append((n_K + n_R + 0.5 * n_H) / max(length, 1))
    # 负电荷密度
    features.append((n_D + n_E) / max(length, 1))
    # 电荷交替性: 正-负-正 或 负-正-负 模式的频率
    charge_alternations = 0
    for i in range(len(sequence) - 2):
        charges = []
        for j in range(3):
            aa = sequence[i + j]
            if aa in 'KRH':
                charges.append(1)
            elif aa in 'DE':
                charges.append(-1)
            else:
                charges.append(0)
        if charges[0] * charges[2] > 0 and charges[1] * charges[0] < 0:
            charge_alternations += 1
    features.append(charge_alternations / max(length - 2, 1))
    # 短肽指示 (咸味肽通常较短 2-10个残基)
    features.append(1.0 / (1.0 + np.exp(length - 8)))  # sigmoid衰减

    # 16.4 疏水-亲水平衡特征 (3维)
    # 疏水矩 (amphiphilicity): 疏水和亲水残基的交替程度
    hydro_moment = 0.0
    for i in range(len(sequence) - 1):
        h1 = AA_PROPS.get(sequence[i], AA_PROPS['X'])[0]
        h2 = AA_PROPS.get(sequence[i + 1], AA_PROPS['X'])[0]
        hydro_moment += abs(h1 - h2)
    features.append(hydro_moment / max(length - 1, 1))

    # 疏水性方差 (均匀vs两极分化)
    if len(hydro_vals) > 1:
        features.append(np.var(hydro_vals))
    else:
        features.append(0.0)

    # 亲水性 (Hopp-Woods scale 简化版: 取反)
    features.append(-gravy if gravy != 0 else 0.0)

    return np.array(features, dtype=np.float32)


def build_feature_matrix(sequences, top_dipeptides=None, top_tripeptides=None, top_cksaap=None):
    """构建特征矩阵"""
    print("正在提取特征...")
    X = []
    for seq in sequences:
        feat = extract_features(seq, top_dipeptides, top_tripeptides, top_cksaap)
        X.append(feat)
    X = np.array(X)
    print(f"✓ 特征维度: {X.shape}")
    return X
