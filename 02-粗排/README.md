# 粗排（Pre-Ranking）

> 粗排从召回的千级候选中进一步筛选到百级，是推荐系统中**效率与精度矛盾最尖锐**的环节。
> 核心约束：延迟必须控制在 **精排的 1/10 以内**（通常 <5ms）。

---

## 粗排的核心挑战

```
召回  ─→  粗排  ─→  精排
1000      100       50
候选                        
                    
粗排的约束：
✓ 候选数量：1000个，比精排多10x
✓ 延迟预算：<5ms（精排可以>50ms）
✗ 不能用复杂特征交叉（无时间）
✗ 不能用长序列建模（无算力）
```

---

## 粗排技术演进

```
[向量内积] → [轻量MLP] → [知识蒸馏] → [一致性优化] → [多任务粗排]
简单双塔    SE block    精排→粗排    COPR           GNOLR
COLD        LightRank   FSCD         ResFlow        
```

---

## 方法列表

| 文件 | 方法 | 年份 | 核心思路 |
|------|------|------|---------|
| [COLD.md](./COLD.md) | Computation-efficient Online Learning to Rank | 2020 | SE block 特征选择 |
| [FSCD.md](./FSCD.md) | Feature Selection + Cross-layer Distillation | 2021 | 特征选择+蒸馏 |
| [COPR.md](./COPR.md) | Consistency-Optimized Pre-Ranking | 2023 | 一致性感知排序 |
| [ResFlow.md](./ResFlow.md) | Residual Flow Multi-task Pre-ranking | 2024 | 多任务残差流 |
| [GNOLR.md](./GNOLR.md) | Gradient-Normalized OLR | 2025 | 梯度正交多任务 |

---

## 粗排精排一致性：核心 KPI

衡量粗排质量最重要的指标是**粗精排一致性**，即粗排 Top-K 与精排 Top-K 的重叠率：

$$
\text{Consistency@K} = \frac{|\text{粗排 Top-K} \cap \text{精排 Top-K}|}{K}
$$

一致性越高，精排的计算资源越不被浪费。提升一致性的主要方法：
1. **知识蒸馏**：让粗排模拟精排的打分分布
2. **直接优化一致性**：COPR 直接把一致性作为优化目标
3. **联合训练**：粗精排共享 embedding，减少表征 gap
