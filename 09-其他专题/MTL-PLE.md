# PLE：渐进式分层提取，解决跷跷板效应

> **论文**：Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations  
> **来源**：腾讯  
> **发表**：RecSys 2020  
> **论文链接**：https://dl.acm.org/doi/10.1145/3383313.3412236

---

## 一、多任务学习的跷跷板效应

在推荐系统中，我们常常需要同时优化多个指标：
- VV（完播率）和 Like（点赞率）
- CTR（点击率）和 CVR（转化率）

直接共享参数（Hard Sharing）或 MMoE 都存在"跷跷板效应"：  
**优化一个任务，另一个任务反而下降。**

根本原因：**任务之间的梯度冲突** + **强制共享导致负迁移**。

---

## 二、PLE 架构

### 2.1 回顾：MMoE 的局限

MMoE（Google 2018）的 Expert 对所有任务共享：

$$
y_k = \text{Tower}_k\left(\sum_i g_k^i \cdot E_i(x)\right)
$$

问题：Expert 被多个任务拉扯，难以专注，仍有负迁移。

### 2.2 PLE 的核心设计：分离 Shared Expert 和 Task-Specific Expert

PLE 为每个任务设置**私有 Expert**，同时保留**共享 Expert**：

```
输入 x
  ├── Shared Experts: E_s^1, E_s^2, ...   (所有任务共用)
  ├── Task-1 Experts: E_1^1, E_1^2, ...  (任务1私有)
  └── Task-2 Experts: E_2^1, E_2^2, ...  (任务2私有)
```

任务 $k$ 的 Gating 网络只从**自己的 Expert + 共享 Expert**中选择：

$$
g_k(x) = \text{Softmax}(W_k^g x)
$$

$$
v_k = g_k(x) \cdot [E_s^1(x), E_s^2(x), \ldots, E_k^1(x), E_k^2(x), \ldots]
$$

任务1不会从任务2的 Expert 中提取信息，从根本上**阻断负迁移路径**。

### 2.3 Progressive（渐进式）分层

PLE 是多层结构（类似 Transformer 的堆叠），每一层的输出作为下一层的输入：

```
Layer 1: [Shared Expert 层 + 各任务 Gate] → 输出
Layer 2: [接收Layer1输出, 继续提取] → 输出
...
Tower: [最终各任务的 Tower 输出预测]
```

层数越多，共享信息越被提炼，任务相关信息越被分离。

---

## 三、与 MMoE 对比

| 维度 | Hard Sharing | MMoE | PLE |
|------|-------------|------|-----|
| Expert 类型 | 全共享 | 全共享 | 私有 + 共享 |
| 负迁移防护 | ❌ | 弱 | ✅ |
| 参数量 | 最少 | 中 | 多（有私有Expert） |
| 跷跷板效应 | 严重 | 存在 | 显著缓解 |

---

## 四、腾讯线上效果

在腾讯视频推荐系统上：
- VV（完播）和 Like（点赞）**同时正向**
- 相比 MMoE：两个任务都提升约 1~2%
- **消除了跷跷板**：调整一个任务的权重，另一个任务不再剧烈波动

---

## 五、懂哥点评

PLE 是工业界多任务学习的**最常用基线**，没有之一。

为什么私有 Expert 有效？直觉是：任务特定的"信号"不应该被强迫分享给相反方向的任务，允许任务保留自己的专有参数是多任务收益的关键。

**实践经验**：
- 私有 Expert 数量：2~4 个（太少效果打折，太多参数爆炸）
- 共享 Expert 数量：2~4 个（一般和私有同规模）
- 层数：2 层足够，3 层以上收益递减
- 任务越相关（如 CTR 和 CTCVR），PLE 优势越不明显；任务越差异大，PLE 优势越大

---

*参考*：[ACM DL](https://dl.acm.org/doi/10.1145/3383313.3412236)
