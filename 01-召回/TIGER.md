# TIGER：生成式语义 ID 召回

> **论文**：Recommender Systems with Generative Retrieval  
> **来源**：Google Research  
> **发表**：NeurIPS 2023  
> **arXiv**：https://arxiv.org/abs/2305.05065

---

## 一、核心问题

传统向量召回（双塔 + ANN 检索）有两个根本性限制：
1. **物品 ID 缺乏语义**：ID embedding 纯靠协同过滤学习，无法泛化到冷启动物品
2. **检索与模型分离**：ANN 是后处理步骤，模型训练时不感知检索过程

TIGER 的思路：**让 Transformer 直接生成物品的语义 ID**，把召回变成一个 seq2seq 任务。

---

## 二、方法详解

### 2.1 语义 ID 构建（核心创新）

TIGER 用 **RQ-VAE（Residual Quantized Variational Autoencoder）** 把物品的语义 embedding（来自 Sentence-T5）量化为一个 **短 token 序列**：

$$
\text{Item} \xrightarrow{\text{Sentence-T5}} e_i \in \mathbb{R}^d \xrightarrow{\text{RQ-VAE}} (c_1, c_2, \ldots, c_L)
$$

其中每个 $c_l \in \{1, \ldots, K\}$ 是第 $l$ 层 codebook 的 code。

**RQ-VAE 的分层量化过程**：
$$
r_0 = e_i, \quad c_l = \arg\min_{k} \|r_{l-1} - e_k^{(l)}\|_2, \quad r_l = r_{l-1} - e_{c_l}^{(l)}
$$

- $L = 3$ 层，每层 codebook 大小 $K = 256$，可表示 $256^3 \approx$ 1600 万个物品
- 语义相似的物品会有**相同的前缀 token**（树状结构）

### 2.2 生成式推荐

用 Encoder-Decoder Transformer（T5 骨干）：
- **输入**：用户历史行为序列（每个物品用其语义 ID token 序列表示）
- **输出**：下一个物品的语义 ID token 序列（自回归生成）

损失函数是标准的交叉熵：
$$
\mathcal{L} = -\sum_{t=1}^{L} \log P(c_t^* \mid c_1^*, \ldots, c_{t-1}^*, \text{context})
$$

### 2.3 Beam Search 解码

生成时用 Beam Search，每步在 codebook 中选 top-K token，最终生成多条候选物品 ID 序列，**天然就是 TopK 召回**，无需单独建索引和 ANN。

---

## 三、与传统双塔对比

| 维度 | 双塔 + ANN | TIGER |
|------|-----------|-------|
| 物品表示 | ID embedding（无语义） | 语义量化 token（有语义） |
| 检索方式 | 近邻搜索（与训练解耦） | Beam Search（端到端） |
| 冷启动 | ❌ 新物品无 embedding | ✅ 用文本语义ID |
| 候选集变化 | 需重建索引 | 语义ID不变 |
| 模型容量 | 受 embedding 表大小限制 | 受 vocab 大小限制 |

---

## 四、实验结果

在 Amazon 数据集（Sports、Beauty、Toys）上：

- 相比 SASRec：Recall@10 提升 **+15% ~ +25%**
- 相比 S3-Rec：NDCG@10 提升 **+20% ~ +30%**
- 冷启动场景优势更明显（语义 ID 无需历史交互）

---

## 五、局限与后续工作

1. **语义 ID 学习与推荐目标解耦**：RQ-VAE 用语义相似性训练，不直接优化推荐指标
2. **索引更新**：新物品加入需要 RQ-VAE 推理，虽无需重建 ANN 但有延迟
3. **后续工作**：
   - **LETTER**（2024）：层次化 token 对齐，让语义ID更适配推荐任务
   - **LC-Rec**（2024）：引入协同过滤信号监督语义ID学习

---

## 六、懂哥点评

TIGER 的根本创新在于把**检索变成生成**，解决了双塔模型中检索与训练解耦的痼疾。  
语义 ID 的构建思路（RQ-VAE 分层量化）也影响了后续一大批工作。  

但实际落地时要注意：Beam Search 的计算开销不小，对百毫秒级的召回延迟有挑战。工业界更多把它用在**候选扩充**而非完全替换向量召回。

---

*参考*：[arXiv 2305.05065](https://arxiv.org/abs/2305.05065) | [Google Blog](https://ai.googleblog.com/)
