# DSSM 深度解读

> **论文**：Learning Deep Structured Semantic Models for Web Search using Clickthrough Data  
> **来源**：Microsoft Research  
> **发表**：CIKM 2013  
> **论文链接**：https://dl.acm.org/doi/10.1145/2505515.2505665

---

## 一、它解决的是什么问题

2013 年之前，搜索/推荐的匹配靠的是关键词或 BM25，问题很明显：**词面不同但语义相同的 query 和 document 匹配不上**（比如 "buy shoes" vs "purchase footwear"）。

DSSM 的出发点：**用神经网络把 query 和 document 都映射到同一个低维语义空间，用向量余弦相似度做匹配**，彻底摆脱词面依赖。

这个思路在今天看来是双塔召回的鼻祖，当年是颠覆性的。

---

## 二、模型结构

### 2.1 输入表示：Word Hashing

原始文本不能直接用词袋（词表太大），DSSM 用了一个聪明的 trick —— **Letter-trigram（字符三元组哈希）**：

把单词拆成字符级 n-gram，比如 `"good"` 拆成 `#go`, `goo`, `ood`, `od#`，再做哈希映射到固定维度（约 30k 维）的稀疏向量。

好处：
- 词表大小从几十万压缩到 3 万以内
- 对拼写错误、变形词有天然鲁棒性
- 避免了 OOV（out-of-vocabulary）问题

### 2.2 双塔结构

```
Query 文本                    Document 文本
    ↓                              ↓
 Word Hashing                 Word Hashing
 (30k → 30k 稀疏)             (30k → 30k 稀疏)
    ↓                              ↓
  FC 300                        FC 300
  FC 300                        FC 300
  FC 128  → query 向量          FC 128  → doc 向量
    ↓             ↘ ↙             ↓
              余弦相似度
              cosine(q, d)
```

两个塔结构**完全独立**，不共享参数（除非特别指定）。

### 2.3 训练目标

给定一个 query $Q$，正样本 document $D^+$（用户点击的），负样本 $\{D_1^-, \ldots, D_n^-\}$（随机采样或展示未点击），用 softmax 训练：

$$
P(D^+ | Q) = \frac{\exp(\gamma \cdot \cos(Q, D^+))}{\exp(\gamma \cdot \cos(Q, D^+)) + \sum_{j=1}^{n} \exp(\gamma \cdot \cos(Q, D_j^-))}
$$

$$
\mathcal{L} = -\log P(D^+ | Q)
$$

其中 $\gamma$ 是平滑因子（超参，一般取 10）。

---

## 三、推断阶段

训练完成后：
- **离线**：对所有 document 跑一次前向，得到所有 doc 向量，建 ANN 索引（当时还没 FAISS，用近似 KNN）
- **在线**：query 实时推断得到 query 向量，ANN 检索 Top-K 相似 doc

这就是今天工业界"双塔 + 向量检索"的最早形态。

---

## 四、DSSM 的局限

| 局限 | 具体表现 | 后续工作解法 |
|------|----------|-------------|
| 无交叉特征 | 两塔完全独立，用户和物品没有任何交互 | 精排阶段用 DIN/DCN 做深度交叉 |
| 词袋假设 | Word Hashing 丢失了词序信息 | CNN-DSSM、LSTM-DSSM 加入序列结构 |
| 无个性化 | Query 塔没有用户画像输入 | YoutubeDNN 把用户历史加入 query 塔 |
| 静态 doc 表示 | doc 向量离线计算，不感知 query | 双塔架构的固有限制（精排才能解决）|

---

## 五、工业影响

DSSM 的影响极为深远，今天工业推荐召回的主流范式——**双塔模型 + ANN 向量检索**——就是从 DSSM 一脉相承下来的。

直系后代：
- **CNN-DSSM**（2014）：用 CNN 替代 MLP，保留词序信息
- **YoutubeDNN**（2016）：用户侧加入历史行为序列，物品侧直接做 softmax 分类
- **MIND**（2019）：把用户向量扩展成多个兴趣向量
- **EBR**（Facebook 2020）：工程化落地，FAISS + 在线服务全流程

---

## 六、懂哥点评

DSSM 在 2013 年做到的事情其实只有一件：**把文本匹配变成向量内积**。但这一步的意义在于，它证明了神经网络可以在没有精确词匹配的情况下做语义检索。

从今天的视角看，DSSM 的表达能力很弱（MLP + 词袋），但它的**架构思想**——两个独立的编码器映射到同一空间，用内积打分——至今仍是所有双塔模型的核心设计。

如果你在面试被问"说一个你了解最深的召回模型"，DSSM 是最好的切入点，因为从它能延伸出整条双塔召回的演进脉络。

---

## 七、配套阅读

- [YoutubeDNN 解读](./02-YoutubeDNN.md)：DSSM 在推荐系统的工业级实现
- [EBR 解读](./05-EBR.md)：向量召回的工程落地全貌
- [MIND 解读](./03-MIND.md)：把单向量升级为多兴趣向量集合
