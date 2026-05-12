# OneRec：端到端生成式推荐系统

> **论文**：OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment  
> **来源**：快手  
> **发表**：2025（arXiv）  
> **arXiv**：https://arxiv.org/abs/2501.18253

---

## 一、背景：传统多级漏斗的困局

工业推荐系统普遍采用多级漏斗：**召回 → 粗排 → 精排 → 重排**。

这种架构的根本问题：
1. **信息损失层层累积**：上游错误无法被下游纠正
2. **各阶段优化目标不统一**：召回优化 Recall，精排优化 AUC，重排优化 GMV，难以全局最优
3. **迭代速度慢**：改一个环节可能影响全链路
4. **生成式潜力被浪费**：LLM 本可以端到端处理，但被强行切割

OneRec 的思路：**用一个统一的生成式模型替代整个多级漏斗**。

---

## 二、架构设计

### 2.1 统一生成框架

OneRec 将推荐问题建模为**条件序列生成**：

$$
P(\hat{y}_{1:K} \mid \text{context}) = \prod_{k=1}^{K} P(\hat{y}_k \mid \hat{y}_{1:k-1}, \text{context})
$$

其中：
- $\text{context}$：用户 ID、行为序列、场景特征、请求特征
- $\hat{y}_k$：第 $k$ 个被推荐的物品（用语义 ID 表示）
- $K$：一次请求返回的候选数

### 2.2 语义 ID 体系

借鉴 TIGER 的 RQ-VAE 思路，OneRec 构建了一套**快手定制的分层语义 ID**：

- 物品内容（标题、封面、标签）→ 预训练多模态编码器 → 语义向量
- 语义向量 → 分层 VQ（Vector Quantization）→ $(c_1, c_2, c_3)$ 三层 token
- 兼顾**语义相似性**（内容接近的物品有相同前缀）和**协同过滤信号**（交互行为相似的物品聚类）

### 2.3 模型骨干

- **Encoder**：Transformer，处理用户历史序列（用语义 ID token 表示）
- **Decoder**：自回归 Transformer，逐 token 生成推荐列表
- **规模**：约 7B 参数（对标 LLaMA-7B）

### 2.4 迭代偏好对齐（Iterative Preference Alignment）

仅靠监督学习（SFT）不够：模型生成的物品列表可能不符合用户的实际偏好。  
OneRec 引入了 **DPO（Direct Preference Optimization）** 做迭代对齐：

$$
\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]
$$

- $y_w$：正样本（用户实际点击/完播的物品序列）
- $y_l$：负样本（曝光但未交互的物品序列）
- 迭代多轮：每轮用最新模型的在线反馈更新偏好数据

---

## 三、与传统漏斗的对比

| 维度 | 传统多级漏斗 | OneRec |
|------|-------------|--------|
| 架构 | 召回+粗排+精排+重排，4个独立模型 | 单一生成模型 |
| 优化目标 | 各阶段独立优化 | 全局序列生成 |
| 多样性 | 重排阶段显式控制 | 生成过程内隐控制 |
| 冷启动 | 各阶段单独处理 | 语义ID天然支持 |
| 延迟 | 各阶段串行叠加 | 单次生成（但Beam Search有开销） |
| 可解释性 | 相对好（每阶段可分析） | 黑盒 |

---

## 四、在线实验结果

快手视频推荐 A/B 测试：
- **观看时长**：+1.5%
- **互动率**：+0.8%
- **用户满意度**（问卷）：+3.2%
- **新物品曝光**：+12%（冷启动受益明显）

---

## 五、落地挑战与工程解法

| 挑战 | 解法 |
|------|------|
| Beam Search 延迟高 | 投机解码（Speculative Decoding）+ GPU 服务 |
| 候选集亿级，生成空间超大 | 限制 Constrained Beam Search 在有效物品 token 上 |
| 训练数据量需求大 | 用用户日志 + 对比学习数据增强 |
| A/B 测试架构改造 | 保留旧漏斗，OneRec 作为独立流量桶 |

---

## 六、懂哥点评

OneRec 代表了推荐系统架构演进的**最激进方向**：完全抛弃多级漏斗，用一个大模型搞定一切。

这在理念上是正确的（多级漏斗本身就是工程妥协产物），但落地难度极大：
- 延迟：Beam Search 生成 20 个候选，需要 GPU 推理，P99 延迟挑战大
- 训练成本：7B 模型的在线迭代成本远高于传统精排模型
- 可控性：广告插入、运营干预、合规过滤等在生成式框架下需要重新设计

目前来看，OneRec 更适合作为**辅助召回路**（生成式候选扩充）或在延迟不敏感场景使用，完全替代多级漏斗还需要 2-3 年的工程积累。

---

*参考*：[arXiv 2501.18253](https://arxiv.org/abs/2501.18253) | [快手技术博客](https://mp.weixin.qq.com/)
