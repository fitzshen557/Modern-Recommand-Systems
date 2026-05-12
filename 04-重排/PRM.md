# PRM：个性化重排模型

> **论文**：Personalized Re-ranking for Recommendation  
> **来源**：阿里巴巴  
> **发表**：RecSys 2019  
> **arXiv**：https://arxiv.org/abs/1904.06813

---

## 一、核心思想

精排给每个物品独立打分，**忽略了物品之间的相互影响**。PRM 的核心贡献：用 **Transformer** 对精排候选列表建模，让每个物品的最终得分感知列表中其他物品的存在。

---

## 二、模型架构

### 2.1 输入构造

对精排输出的候选列表 $\{i_1, i_2, \ldots, i_n\}$，每个物品的输入表示：

$$
x_j = \text{concat}(e_{i_j}, e_u, e_{pv_j})
$$

- $e_{i_j}$：物品 embedding
- $e_u$：用户 embedding（个性化信号）
- $e_{pv_j}$：精排分数的 embedding（编码位置/得分信息）

### 2.2 Transformer 编码

用多层 Transformer Encoder 对列表建模：

$$
H = \text{TransformerEncoder}([x_1, x_2, \ldots, x_n])
$$

每个位置 $j$ 的输出 $h_j$ 融合了该物品与整个列表的上下文信息。

### 2.3 输出与训练

$$
\hat{y}_j = \text{sigmoid}(W \cdot h_j)
$$

训练：listwise softmax loss（同一请求内所有物品作为一个列表优化）：

$$
\mathcal{L} = -\sum_{j} y_j \log\left(\frac{\exp(\hat{y}_j)}{\sum_k \exp(\hat{y}_k)}\right)
$$

---

## 三、Attention 的含义

Transformer 中的 Self-Attention 学习到了：
- **互补性**：两件相似商品同时出现，其中一件得分下降（多样性效果）
- **搭配性**：手机和手机壳同时出现，可能互相增强（套装效果）
- **位置效应**：不同列表位置的"注意力"不同

这些都是精排 pointwise 打分无法捕捉的信息。

---

## 四、与后续重排方法的关系

```
PRM (2019) → SetRank (2020) → PIER (2023) → OPERA (2024)
[单向Attn]   [集合感知Attn]   [增量在线学习]  [精排重排联合]
```

PRM 奠定了"用 Transformer 建模列表"的范式，后续工作在此基础上：
- SetRank：处理候选集（非有序列表）感知
- PIER：引入在线学习，动态更新
- OPERA：打破精排/重排边界，联合优化

---

## 五、懂哥点评

PRM 是重排领域的奠基之作，思路简洁但有效：**把列表感知变成一个 Transformer seq2seq 问题**。

工业落地时的关键工程问题：
- 列表长度：一般取精排 top-50，Transformer $O(n^2)$ 在50个物品上完全可接受
- 候选列表顺序：输入是否有序会影响效果，建议打乱输入顺序做数据增强
- 与精排的结合：可以把精排分作为特征输入，让 PRM 做"增量修正"而非完全重排

---

*参考*：[arXiv 1904.06813](https://arxiv.org/abs/1904.06813)
