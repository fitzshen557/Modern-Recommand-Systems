# DIN 深度解读：目标注意力机制

> **论文**：Deep Interest Network for Click-Through Rate Prediction  
> **来源**：阿里巴巴  
> **发表**：KDD 2018  
> **论文链接**：https://arxiv.org/abs/1706.06978

---

## 一、精排的核心问题

精排需要对每个候选物品打一个 CTR 分数。在 DIN 之前，精排模型（Wide&Deep、DeepFM）处理用户历史行为的方式是：

**把所有历史行为的 embedding 直接平均池化**，作为用户兴趣向量。

这有一个根本性问题：**用户的 2000 条历史行为，对一个"男士跑步鞋"的 CTR 预测，真正有用的只有其中关于"运动鞋"的 30 条**。剩下 1970 条（书、3C、食品等）对这个候选物品不相关，但平均后都混进去了，稀释了有效信号。

**DIN 的解法：用待预测的候选物品（Target Item）来激活历史行为中相关的部分。**

---

## 二、核心机制：Target-aware Attention

### 2.1 标准的 Attention 机制回顾

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

标准 Attention 里 Query、Key、Value 三者独立。

### 2.2 DIN 的设计

DIN 让 **Target Item 充当 Query**，用户历史行为充当 Key 和 Value：

$$e_u = \sum_{i=1}^n a(e_i, e_t) \cdot e_i$$

其中：
- $e_i$：第 $i$ 条历史行为的 embedding
- $e_t$：Target Item 的 embedding  
- $a(\cdot, \cdot)$：注意力权重函数（一个 MLP）

**注意力权重计算**：

$$a(e_i, e_t) = \text{sigmoid}\left(\text{MLP}([e_i, e_t, e_i \odot e_t, e_i - e_t])\right)$$

输入是 $[e_i, e_t]$ 以及它们的**逐元素乘积**和**差**——这些交叉特征帮助模型判断两者的相关性。

注意 DIN 没有用标准的 softmax 归一化（所有权重加和为 1），而是用**独立的 sigmoid**。原因：不同 Target Item 的"有效行为数量"不同，硬性归一化到1会损失这个信号。

### 2.3 完整模型结构

```
历史行为序列 [e_1, ..., e_n]
         ↓ Target-aware Attention（e_t作为Query）
用户兴趣向量 e_u（加权求和）
         ↓
Concat[e_u, e_t, e_用户画像, e_上下文]
         ↓
MLP（全连接层）
         ↓
CTR预测分数
```

---

## 三、训练技巧

DIN 论文中还提出了两个重要的训练技巧，常常被忽视：

### 3.1 Dice 激活函数

传统 PReLU 有一个固定的阈值（0），Dice 让阈值根据数据自适应决定：

$$f(x) = p(x) \cdot x + (1 - p(x)) \cdot \alpha x$$

$$p(x) = \text{sigmoid}\left(\frac{x - \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}}\right)$$

本质是一个软门控：当 $x$ 在均值附近时，$p(x) \approx 0.5$，是 ReLU 和 PReLU 的插值；偏离均值越多，越接近 ReLU。

### 3.2 自适应正则化（Adaptive Regularization）

推荐系统中特征出现频率差异巨大（热门物品 vs 长尾物品），DIN 针对低频特征施加更强的 L2 正则：

$$L_2(\mathbf{W}) = \sum_j \sum_{i: x_i^{(j)} \neq 0} \frac{\alpha_j}{\text{freq}(x_i^{(j)})} \|w_j\|^2$$

低频特征的 Embedding 更容易过拟合（样本少），加强正则防止过拟合到少数样本。

---

## 四、DIN 的效果

在阿里妈妈展示广告系统上：
- AUC 提升约 **+0.01**（精排里 AUC 0.01 是非常显著的提升，通常对应线上 CTR 1% 以上的提升）
- 尤其在**长序列用户**上提升更明显（历史行为越多，注意力机制越有用）

---

## 五、DIN 后续演进

| 问题 | 限制 | 改进 |
|------|------|------|
| 忽略兴趣演化的时序 | DIN 把历史当无序集合 | **DIEN**（2019）：加入 GRU 捕获时序变化 |
| 序列太长，计算量大 | n 条行为 × n 次注意力 = O(n) | **SIM**（2020）：先检索 Top-K 相关行为，再做精细注意力 |
| 行为序列太长 | 工业中历史可达数千 | **TWIN**（2023）：分解注意力，支持万级序列 |

---

## 六、懂哥点评

DIN 是精排领域里程碑式的工作，它的价值在于提出了一个极其自然的问题：**为什么对所有候选物品，用一样的用户向量打分？**

"用候选物品激活相关历史行为"这个思路简洁而深刻，本质上是把**精排建模成一个条件 CTR 预测**（条件于 Target Item），而不是无条件的用户偏好预测。

工业落地时要注意：
1. **序列长度截断**：DIN 在 n 很大时 O(n) 的计算量会成为瓶颈，一般截断到最近 200 条
2. **Item Embedding 的对齐**：Target Item 和 History Item 需要共享同一个 Embedding Table，否则注意力计算的语义不一致
3. **负样本构造**：DIN 的训练样本和精排样本构造有学问，曝光未点击作为负样本，但要过滤掉"跳出率高的位置"的负样本（位置偏差）

---

## 七、配套阅读

- [DIEN 解读](./02-DIEN.md)：在 DIN 基础上加入兴趣演化
- [SIM 解读](./03-SIM.md)：长序列 DIN，支持万级历史行为
- [TWIN 解读](./04-TWIN.md)：快手的长序列精排实践
