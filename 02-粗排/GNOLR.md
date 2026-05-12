# GNOLR：无冲突多任务粗排新范式

> **论文**：GNOLR: Gradient-Normalized Online Learning to Rank for No-Conflict Multi-Task Pre-Ranking  
> **来源**：电商平台（KDD 2025）  
> **发表**：KDD 2025  

---

## 一、背景：多任务粗排的"跷跷板"困境

粗排一般需要同时优化多个目标：CTR、CVR、GMV、时长等。  
多任务学习的天敌是**任务冲突（Task Conflict）**：  
当不同任务的梯度方向相反时，联合优化会导致某些任务的性能下降（即"跷跷板效应"）。

**KDD 2024 的 ResFlow**（前序工作）用残差流来缓解冲突，但没有从根本上解决。  
GNOLR 提出了一种**从梯度角度消除冲突**的新方案。

---

## 二、方法核心

### 2.1 梯度冲突的定义

对任务 $i$ 和任务 $j$，若其梯度点积为负，则存在冲突：
$$
\text{conflict}(i, j) = \mathbf{g}_i \cdot \mathbf{g}_j < 0
$$

### 2.2 梯度正交化（Gradient Orthogonalization）

GNOLR 在冲突任务之间做梯度投影，让各任务的梯度**正交化**，消除相互干扰：

$$
\tilde{\mathbf{g}}_i = \mathbf{g}_i - \frac{\mathbf{g}_i \cdot \mathbf{g}_j}{\|\mathbf{g}_j\|^2} \mathbf{g}_j, \quad \text{if } \mathbf{g}_i \cdot \mathbf{g}_j < 0
$$

修正后的梯度 $\tilde{\mathbf{g}}_i$ 与 $\mathbf{g}_j$ 正交，对任务 $j$ 没有负面影响，同时保留了对任务 $i$ 的优化方向。

### 2.3 梯度归一化（Gradient Normalization）

为防止不同任务梯度量级差异过大导致某任务主导优化，GNOLR 还做了梯度归一化：

$$
\hat{\mathbf{g}}_i = \frac{\tilde{\mathbf{g}}_i}{\|\tilde{\mathbf{g}}_i\|}
$$

最终更新：
$$
\theta \leftarrow \theta - \eta \sum_i w_i \hat{\mathbf{g}}_i
$$

### 2.4 Online Learning to Rank（OLR）框架

GNOLR 在 OLR（在线排序学习）框架下运行：
- 每次曝光产生新的反馈信号
- 实时更新模型，快速适应分布变化
- 结合梯度正交化，实现无冲突的实时多任务学习

---

## 三、对比实验

相比 ResFlow（KDD 2024）：

| 任务 | ResFlow | GNOLR | 提升 |
|------|---------|-------|------|
| CTR | baseline | +2.1% | ✅ |
| CVR | baseline | +3.5% | ✅ |
| GMV | baseline | +1.8% | ✅ |
| 跷跷板 | 仍存在 | 消除 | ✅✅ |

---

## 四、懂哥点评

GNOLR 从**梯度角度**解决多任务冲突，比 ResFlow 的架构级解法更加优雅和通用。

梯度正交化的思路来自 PCGrad（NeurIPS 2020），GNOLR 的贡献在于把它和在线学习（OLR）框架结合，并做了工业级验证。

一个实际问题：梯度正交化在每次 backward 时都需要计算任务间的梯度内积和投影，在任务数量多时（如10+个任务）计算开销不小，需要根据任务数量评估是否值得。

---

*参考*：KDD 2025 Proceedings
