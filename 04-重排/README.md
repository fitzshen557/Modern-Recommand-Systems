# 重排（Re-Ranking）

> 重排在精排之后，对 Top-K 候选列表做**全局优化**。
> 目标：不仅让每个物品的得分高，还要让**整个列表**的用户体验最佳。

---

## 重排 vs 精排的核心区别

```
精排：pointwise/pairwise，每个物品独立打分
重排：listwise，考虑物品之间的相互关系
      - 多样性：不要连续推5个相似内容
      - 公平性：不同类别/创作者有合理曝光
      - 位置效应：不同位置的用户注意力不同
      - 整体 GMV/时长最大化（非单个物品最优）
```

---

## 方法列表

| 文件 | 方法 | 核心策略 | 年份 |
|------|------|---------|------|
| [MMR.md](./MMR.md) | Maximal Marginal Relevance | 贪心多样性 | 1998 |
| [DPP.md](./DPP.md) | Determinantal Point Process | 概率多样性 | 2018 |
| [PRM.md](./PRM.md) | Personalized Re-ranking Model | Transformer 列表建模 | 2019 |
| [GRN.md](./GRN.md) | Global Reranking Network | 图神经网络 | 2023 |
| [OPERA.md](./OPERA.md) | Ordered-aware Reranking | 精排-重排联合优化 | 2024 |
| [RERANK-LLM.md](./RERANK-LLM.md) | LLM-based Reranking | LLM zero-shot排序 | 2024 |

---

## 2024-2025 重排新趋势

1. **精排-重排联合训练**：OPERA 等工作开始探索打破精排/重排的边界
2. **LLM 重排**：利用 LLM 的列表理解能力，zero-shot 做重排（但延迟高）
3. **多目标重排**：同时优化 CTR + 多样性 + 公平性，帕累托前沿优化
