# SetRank — 候选集交互的 Listwise 排序

> 论文：SetRank: A Setwise Ranking Approach with Self-Attention for Large-Scale Recommendation  
> 核心贡献：用 Self-Attention 对整个候选集做 listwise 编码，让每个 item 的打分能"看到"其他 item，从而捕获候选集内部的竞争/互补关系

---

## 1. 背景动机：Pointwise 精排的根本缺陷

### 1.1 传统精排的工作方式

主流工业精排模型（DIN、DIEN、DCN 等）都是 **pointwise** 的：

```
score_i = f(user, item_i, context)
```

对每个候选 item 独立打分，然后按分数排序取 Top-K。

### 1.2 为什么 pointwise 不够

| 问题 | 描述 |
|------|------|
| **候选集竞争忽略** | 两个高度相似的 item，pointwise 会给它们打接近的分，但实际展示一个就够了，另一个是浪费展示位 |
| **互补关系丢失** | 用户看了一条"穿搭教程"，下一条如果配"同款购买链接"效果更好，pointwise 无法感知这种组合增益 |
| **位置偏差** | 用户注意力随位置递减，但 pointwise 打分与位置无关（或仅用位置特征近似） |
| **多样性盲区** | 独立打分天然倾向于给热门/通用内容高分，导致列表同质化 |

### 1.3 解决思路

既然问题是"item 之间没有交互"，那直接引入 **set-level** 的注意力机制：让每个 item 的表示能 attend 到同一请求的所有其他候选 item，从而感知竞争与互补。

---

## 2. 方法详解：Self-Attention 驱动的 Setwise 排序

### 2.1 整体架构

```
输入：候选集 S = {item_1, item_2, ..., item_N}

Step 1: Base Model 编码
   h_i = MLP(concat(user_emb, item_i_emb, context_emb))   # 每个 item 的 base 表示

Step 2: Setwise Self-Attention
   H = [h_1, h_2, ..., h_N]   # N × d 矩阵
   H' = MultiHeadAttention(H, H, H)   # N × d，每个 item 融合了全局信息

Step 3: 打分
   score_i = MLP(h'_i)   # 融合候选集信息后的打分

Step 4: 排序
   按 score 降序取 Top-K
```

### 2.2 Self-Attention 的细节

```python
# 标准 Transformer Self-Attention
Q = H @ W_Q    # (N, d_k)
K = H @ W_K    # (N, d_k)
V = H @ W_V    # (N, d_v)

Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

关键：这里 Q = K = V = H（来自同一候选集），所以是 **self-attention**。

每个 item 的更新后表示 `h'_i` 是所有其他 item 的加权和：

```
h'_i = Σ_j softmax(q_i · k_j / sqrt(d_k)) · v_j
```

权重 `α_ij` 反映了 item_j 对 item_i 打分的"影响程度"。

### 2.3 与位置编码的结合

SetRank 在 self-attention 中加入 **位置编码** 来捕获位置偏差：

```
h_i = base_emb_i + PE(pos_i)
```

这样模型能学到"第 1 位和第 5 位应该有不同策略"。

### 2.4 训练目标

使用 **listwise loss**，直接优化 NDCG：

```
L = -Σ_i (2^{rel_i} - 1) / Z * log(σ(score_{rank(i)}))

其中 Z 是归一化常数，rel_i 是 item_i 的真实相关性
```

实际工程中常用 pairwise 或 pointwise surrogate loss 近似：

```
L_pairwise = Σ_{i∈正, j∈负} log(σ(score_i - score_j))
```

---

## 3. 与 PRM（阿里重排 Transformer）的区别和联系

### 3.1 PRM 回顾

阿里 2020 年提出的 Personalized Re-ranking Model：
- 同样用 Transformer 做 listwise 重排
- 用 user embedding 作为 Position-wise Attention 的 query
- 主要定位在精排之后的 **重排阶段**

### 3.2 核心差异

| 维度 | SetRank | PRM |
|------|---------|-----|
| **阶段定位** | 可替代精排或用于精排后 | 定位重排阶段 |
| **Attention 机制** | Item-to-Item Self-Attention | Position-wise Attention（user query） |
| **信息流向** | 双向：item_i 和 item_j 互相影响 | 单向：user → item_i（个性化位置分配） |
| **建模目标** | 候选集内部竞争/互补 | 个性化列表组合 |
| **候选集大小** | 需要控制（通常 ≤ 200） | 可以处理更大（重排阶段候选少） |

### 3.3 联系

两者本质上都在做一件事：**打破 pointwise 的独立性假设**。

- PRM 侧重"把对的 item 放到对的位置"（位置分配问题）
- SetRank 侧重"让 item 之间的交互影响打分"（竞争/互补问题）

在工业系统中，两者可以串联使用：精排阶段用 SetRank → 重排阶段用 PRM。

---

## 4. 工业落地挑战

### 4.1 延迟问题

Self-Attention 的复杂度是 O(N²d)，对精排阶段的候选集（通常 200-1000）来说：

```
N = 500, d = 128
Self-Attention 计算量 ≈ 500² × 128 = 32M FLOPs
```

**解决方案：**
- **截断候选集**：粗排后只取 Top-200 进精排，控制 N
- **线性 Attention**：用 Linformer/Performer 近似，O(Nd) 复杂度
- **分块计算**：将候选集分块，块内 attention + 块间稀疏 attention
- **量化蒸馏**：将 SetRank 的知识蒸馏到轻量模型

### 4.2 候选集大小限制

Self-Attention 对 N 敏感，N 太大时：
- GPU 显存爆炸（N² 的注意力矩阵）
- 长尾 item 的 attention 被稀释

**工程经验：**
- 最佳效果区间：N ∈ [50, 200]
- N > 500 后收益递减，延迟成本急剧上升
- 推荐做法：粗排阶段严格截断，保证精排候选集质量 > 数量

### 4.3 部署实践

```
典型 pipeline：
召回(10000+) → 粗排(500) → SetRank精排(200) → PRM重排(50) → 展示(20)
```

SetRank 通常放在粗排之后，候选集控制在 200 以内。

---

## 5. 实验结果

### 5.1 离线实验

| 模型 | NDCG@10 | MAP | Latency/item |
|------|---------|-----|-------------|
| DIN (pointwise) | 0.423 | 0.381 | 0.3ms |
| LambdaRank | 0.438 | 0.396 | 0.4ms |
| SetRank | 0.467 | 0.425 | 1.2ms |
| SetRank + PRM | 0.478 | 0.436 | 1.8ms |

SetRank 相对 pointwise baseline 提升 **NDCG +10%**，说明候选集交互确实有价值。

### 5.2 线上 A/B

- CTR +3.2%
- 多样性指标 +8.5%（Entropy of categories）
- 人均浏览深度 +2.1%

---

## 6. 懂哥点评 🐟

### 6.1 核心洞察

SetRank 抓住了一个被长期忽视的问题：**item 不是孤立存在的，它们在同一个列表中互相竞争用户的注意力**。

这就像招聘：你单独看每个候选人都觉得不错，但放在一起比较时，相似背景的人就会互相挤占名额。

### 6.2 局限

1. **O(N²) 是硬伤**：精排阶段候选集大，Self-Attention 的成本不低
2. **只建模了浅层交互**：一层 self-attention 只能捕获 pairwise 级别的依赖，更深的组合关系需要 stack 多层
3. **训练目标与 NDCG 的 gap**：用 surrogate loss 近似 NDCG 优化，理论上不够 tight

### 6.3 趋势判断

SetRank 的思路在 2024-2025 年被进一步发展：
- **交互方式升级**：从 self-attention 到 cross-attention、hypergraph
- **阶段融合**：精排和重排不再割裂，一个模型同时处理竞争和位置
- **与生成式方法融合**：HiGR 等工作直接把排序做成序列生成

SetRank 的历史定位：**第一次在工业推荐系统中严肃地用 Transformer 建模候选集交互**，为后续所有 listwise 方法铺了路。

---

## 参考链接

1. SetRank 原论文：[SetRank: A Setwise Ranking Approach with Self-Attention](https://arxiv.org/abs/2106.05531)
2. PRM（阿里重排 Transformer）：[Personalized Re-ranking for Recommendation](https://arxiv.org/abs/1904.06813)
3. Attention Is All You Need：[arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
4. LambdaRank：[Learning to Rank with NDCG](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/lambdarank.pdf)
5. 工业实践参考：[Deep Learning for Recommendation Systems - Listwise Approaches](https://arxiv.org/abs/2209.03933)
