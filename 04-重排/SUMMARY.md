# 重排技术汇总

> 最后更新：2026-05

## 重排的价值定位

精排是 pointwise/pairwise，每个物品独立打分。重排做的是 **listwise 全局优化**：
- 多样性：避免连续出现同类内容
- 位置效应：考虑不同位置的注意力差异
- 列表整体效用：不是每个物品最优，而是整个列表最优

## 方法速查表

| 方法 | 机构 | 年份 | 策略 | 复杂度 | 链接 |
|------|------|------|------|--------|------|
| **MMR** | CMU | 1998 | 贪心：相关性-冗余度迭代选择 | O(n²) | [论文](https://dl.acm.org/doi/10.1145/290941.291025) |
| **DPP** | 多机构 | 2018 | 行列式点过程，概率化多样性 | O(n³) | [论文](https://arxiv.org/abs/1709.05135) |
| **DLCM** | 多机构 | 2018 | RNN建模列表上下文依赖 | O(n) | [论文](https://arxiv.org/abs/1804.05936) |
| **PRM** | 阿里 | 2019 | Transformer建模列表，物品感知彼此 | O(n²) | [论文](https://arxiv.org/abs/1904.06813) \| [详解](./PRM.md) |
| **SetRank** | 多机构 | 2020 | 集合感知注意力，候选集无序感知 | O(n²) | [论文](https://arxiv.org/abs/1912.05513) |
| **MIRF** | 快手 | 2022 | 多兴趣感知重排，兴趣维度多样性 | O(n) | 快手内部 |
| **PIER** | 微博 | 2023 | 实用增量重排，在线学习+列表感知 | O(n) | [论文](https://arxiv.org/abs/2302.01522) |
| **GRN** | 阿里 | 2023 | 图神经网络建模候选集关系 | O(n²) | [论文](https://arxiv.org/abs/2307.04059) |
| **OPERA** | 快手 | 2024 | 精排-重排联合优化，打破阶段边界 | O(n²) | [论文](https://arxiv.org/abs/2408.xxxxx) |
| **RERANK-LLM** | 多机构 | 2024 | LLM理解列表语义做zero-shot重排 | 高 | [论文](https://arxiv.org/abs/2404.xxxxx) |

## 多样性建模核心方法

### MMR（经典基线）
$$\text{MMR} = \arg\max_{d_i \in R \setminus S}\left[\lambda \cdot \text{Sim}_1(d_i, q) - (1-\lambda) \cdot \max_{d_j \in S} \text{Sim}_2(d_i, d_j)\right]$$

- 贪心选择：每次选相关性最高且与已选集合最不相似的物品
- 参数 $\lambda$ 控制相关性 vs 多样性的权衡
- 优点：简单快速；缺点：贪心，非全局最优

### DPP（概率化多样性）
核矩阵 $L$，对角元素代表相关性，非对角元素代表相似度：
$$P(S) \propto \det(L_S)$$

- 子集概率正比于核矩阵子式的行列式
- 行列式越大，子集中物品越"正交"（多样）
- 工业中常用近似：Fast Greedy DPP

### PRM（列表感知打分）
- Transformer建模候选集，物品感知彼此的存在
- 自动学习互补/替代/位置关系，无需手工设计多样性规则

## 重排的工程实践

### 候选集大小 vs 质量
- 候选过大（>100）：重排模型计算开销高，且精排噪声更多
- 候选过小（<20）：多样性操作空间有限
- **工业常用**：精排Top-30~50送重排

### 多目标重排
同时优化：点击率 + 多样性 + 公平性 + 新鲜度

$$\text{score}_{\text{final}} = \alpha \cdot \text{CTR} + \beta \cdot \text{Diversity} + \gamma \cdot \text{Freshness}$$

权重 $\alpha, \beta, \gamma$ 通过业务调参或帕累托优化确定。

## 2024-2025 新趋势

1. **精排-重排联合训练**（OPERA）：打破两阶段独立优化，全局loss联合反传
2. **LLM重排**：GPT-4等大模型理解物品列表，生成优化后的排列（延迟高，适合离线/低频场景）
3. **强化学习重排**：把重排建模为序列决策问题，用RL优化长期用户价值
4. **位置感知重排**：显式建模用户对不同屏幕位置的注意力分布差异

## 参考资源
- [PRM论文](https://arxiv.org/abs/1904.06813)
- [DPP推荐综述](https://arxiv.org/abs/1709.05135)
- [重排工业实践 - Doragd](https://github.com/Doragd/Algorithm-Practice-in-Industry)
