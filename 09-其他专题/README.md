# 其他专题

> 推荐系统中跨越多个阶段的横向技术专题。

---

## 专题列表

### 9.1 多任务学习（Multi-Task Learning）

解决推荐系统中多目标（CTR/CVR/时长/多样性）联合优化的技术。

| 文件 | 方法 | 年份 |
|------|------|------|
| [MTL-MMoE.md](./MTL-MMoE.md) | Multi-gate Mixture of Experts | 2018 |
| [MTL-PLE.md](./MTL-PLE.md) | Progressive Layered Extraction | 2020 |
| [MTL-AITM.md](./MTL-AITM.md) | Adaptive Information Transfer Multi-task | 2021 |

---

### 9.2 去偏与因果推断（Debiasing & Causal Inference）

推荐系统中普遍存在曝光偏差、位置偏差、选择偏差，需要用因果方法校正。

| 文件 | 方法 | 年份 |
|------|------|------|
| [Debiasing-IPS.md](./Debiasing-IPS.md) | Inverse Propensity Score | 经典 |
| [Debiasing-DICE.md](./Debiasing-DICE.md) | Disentangling Interest and Conformity | 2021 |
| [Debiasing-DCCL.md](./Debiasing-DCCL.md) | Dual Causal Contrastive Learning | 2023 |

---

### 9.3 长序列建模（Long Sequence Modeling）

支持万级用户行为序列的高效建模技术。

| 文件 | 方法 | 年份 |
|------|------|------|
| [LongSeq-Mamba4Rec.md](./LongSeq-Mamba4Rec.md) | Mamba for Sequential Rec | 2024 |
| [LongSeq-SLAB.md](./LongSeq-SLAB.md) | Sparse Linear Attention for Rec | 2024 |

---

### 9.4 对比学习（Contrastive Learning in RecSys）

自监督对比学习被广泛用于解决推荐系统的数据稀疏问题。

- **SimGCL**（2022）：图对比学习，随机噪声增强
- **SGL**（2021）：自监督图学习，节点/边 dropout 增强
- **CL4Rec**（2022）：序列推荐中的对比学习增强
- **DirectAU**（2022）：直接优化 Alignment + Uniformity，替代对比损失

---

## 2024-2025 横向趋势总结

| 趋势 | 具体表现 |
|------|---------|
| **大模型全面渗透** | 召回/精排/冷启动都有 LLM 方案 |
| **生成式推荐崛起** | 从判别式打分转向自回归生成候选 |
| **多任务冲突缓解** | 梯度正交、帕累托优化等方法成熟 |
| **长序列常态化** | 万级序列建模从研究走向工业实践 |
| **多模态统一** | 视觉+文本+行为统一 embedding 空间 |
