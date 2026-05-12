# Mamba4Rec：状态空间模型用于序列推荐

> **论文**：Mamba4Rec: Towards Efficient Sequential Recommendation with Selective State Space Models  
> **来源**：多机构合作  
> **发表**：2024（arXiv）  
> **arXiv**：https://arxiv.org/abs/2403.03900

---

## 一、动机：Transformer 的长序列瓶颈

序列推荐中，Transformer（SASRec/BERT4Rec）的核心问题：
- 自注意力复杂度 $O(n^2)$：序列长度翻倍，计算量翻四倍
- 内存占用 $O(n^2)$：长序列下显存爆炸
- 推断时全序列重计算：无法增量推断

**Mamba** 是 2023 年提出的状态空间模型（SSM），在 NLP 长序列任务上以 $O(n)$ 复杂度匹敌 Transformer，Mamba4Rec 把它引入推荐序列建模。

---

## 二、Mamba 核心机制

### 2.1 连续时间状态空间模型（SSM）

$$
h'(t) = Ah(t) + Bx(t)
$$
$$
y(t) = Ch(t)
$$

离散化后：
$$
h_t = \bar{A}h_{t-1} + \bar{B}x_t, \quad y_t = Ch_t
$$

其中 $\bar{A} = e^{\Delta A}$，$\bar{B} = (\Delta A)^{-1}(e^{\Delta A} - I)\Delta B$，$\Delta$ 是时间步长。

### 2.2 选择性机制（S4 → Mamba 的关键升级）

标准 SSM 中 $A, B, C$ 是固定参数，Mamba 让它们**依赖输入**：
$$
B_t = f_B(x_t), \quad C_t = f_C(x_t), \quad \Delta_t = f_\Delta(x_t)
$$

这使模型能够**选择性地记忆或遗忘**历史信息，类似于 LSTM 的门控机制，但并行效率更高。

### 2.3 并行计算（Hardware-Aware Algorithm）

虽然 SSM 在形式上是递推的（$h_t$ 依赖 $h_{t-1}$），Mamba 设计了专门的**并行前缀和（Parallel Prefix Scan）**算法，训练时实现完全并行化。

---

## 三、Mamba4Rec 架构

```
用户行为序列: [i_1, i_2, ..., i_n]
       ↓ Embedding
[e_1, e_2, ..., e_n]
       ↓ Mamba Block × L
[h_1, h_2, ..., h_n]
       ↓ 取最后一个 h_n
     预测下一个物品
```

**Mamba Block**：
$$
x' = \text{MambaSSM}(\text{LayerNorm}(x)) + x
$$
$$
\text{output} = \text{FFN}(\text{LayerNorm}(x')) + x'
$$

---

## 四、实验结果

在 Amazon（Sports、Beauty、Toys）和 Yelp 数据集上，与 SASRec 对比：

| 指标 | SASRec | Mamba4Rec | 提升 |
|------|--------|-----------|------|
| Recall@10 | baseline | +2~5% | ✅ |
| NDCG@10 | baseline | +3~6% | ✅ |
| 推断延迟（n=512） | 100% | **35%** | ✅✅ |
| 显存占用（n=512） | 100% | **28%** | ✅✅ |

---

## 五、懂哥点评

Mamba4Rec 在效率上的优势是真实的，在效果上的提升幅度比较有限（部分数据集持平甚至略差）。

核心原因：**推荐序列和语言序列的特征不同**：
- 语言序列：局部依赖强，Mamba 的选择性遗忘很适合
- 推荐序列：用户兴趣常常是**非连续的跳跃**（今天买了运动鞋，突然买了一本小说），SSM 的顺序递推对非局部依赖处理不如 Transformer 的全局注意力

**适合场景**：超长序列（>1000条行为），此时 Mamba 的效率优势压倒一切；短序列（<200条）建议 SASRec/HSTU。

---

*参考*：[arXiv 2403.03900](https://arxiv.org/abs/2403.03900)
