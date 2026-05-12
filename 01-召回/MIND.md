# MIND：多兴趣网络

> **论文**：Multi-Interest Network with Dynamic Routing for Recommendation at Tmall  
> **来源**：阿里巴巴  
> **发表**：CIKM 2019  
> **arXiv**：https://arxiv.org/abs/1904.08030

---

## 一、核心问题

传统双塔召回用**一个向量**表示用户，无法捕获用户的**多样兴趣**。  
比如某用户既喜欢科幻小说又喜欢运动装备，单向量会被平均化，两个方向都召回不准。

**MIND 的解法**：为每个用户学习**多个兴趣向量**，每个向量对应一种兴趣，召回时用多个向量分别检索再合并。

---

## 二、方法详解

### 2.1 胶囊网络（Dynamic Routing）

MIND 借鉴胶囊网络（Capsule Network）的动态路由机制，把用户行为序列聚类为 $K$ 个兴趣向量：

**路由过程**（EM-like 迭代）：

第 $i$ 个行为到第 $k$ 个兴趣胶囊的路由权重：
$$
w_{ik} = \frac{\exp(b_{ik})}{\sum_{j} \exp(b_{ij})}
$$

兴趣胶囊聚合：
$$
z_k = \text{squash}\left(\sum_i w_{ik} \cdot h_i\right), \quad \text{squash}(x) = \frac{\|x\|^2}{1+\|x\|^2} \cdot \frac{x}{\|x\|}
$$

路由权重更新：
$$
b_{ik} \leftarrow b_{ik} + h_i \cdot z_k
$$

迭代 3 次后收敛，得到 $K$ 个兴趣向量 $\{z_1, \ldots, z_K\}$。

### 2.2 目标注意力（Target Attention）

训练时，对给定的 target item $e_t$，用注意力机制选择最相关的兴趣：

$$
v_u = \text{Attention}(e_t, \{z_1, \ldots, z_K\}) = \text{softmax}(e_t^T \cdot [z_1, \ldots, z_K]) \cdot [z_1; \ldots; z_K]^T
$$

### 2.3 训练与推断

- **训练**：$\text{score}(u, i) = v_u^T \cdot e_i$，优化 sampled softmax loss
- **推断**：用 $K$ 个兴趣向量分别在向量库中做 ANN，合并结果后去重

---

## 三、与 YoutubeDNN 对比

| 维度 | YoutubeDNN | MIND |
|------|-----------|------|
| 用户表示 | 单向量 | K 个兴趣向量 |
| 多样兴趣 | ❌ 被平均 | ✅ 显式建模 |
| 推断复杂度 | 1次 ANN | K次 ANN（K=3~7） |
| 冷启动 | 序列越长越好 | 短序列也可 |

---

## 四、懂哥点评

MIND 的核心贡献是**把用户表示从向量升级为向量集合**，开启了多兴趣召回的时代。  

动态路由机制虽然有效，但计算开销较大。后续 **ComiRec** 用自注意力替代动态路由，速度更快。  

工业界落地时，$K$ 的选择很关键：太小多样性不够，太大 ANN 开销翻倍。实践中一般 $K = 3 \sim 5$。

---

*参考*：[arXiv 1904.08030](https://arxiv.org/abs/1904.08030)
