# TWIN：快手长序列精排

> **论文**：TWIN: TWo-stage Interest Network for Lifelong User Behavior Modeling in CTR Prediction at Kuaishou  
> **来源**：快手  
> **发表**：KDD 2023  
> **arXiv**：https://arxiv.org/abs/2302.02352

---

## 一、核心问题

用户的历史行为序列往往有**数千甚至上万**条记录。直接对全量序列做 Target Attention（如 DIN）计算代价是 $O(n)$，Transformer 更是 $O(n^2)$，在精排场景中延迟不可接受。

现有方案：
- **截断**：只用最近 200 条 → 丢失长期兴趣
- **SIM**（阿里）：先搜索相关子序列再精排 → 两阶段但召回精度有损

TWIN 的思路：**在目标注意力框架内，用近似计算实现长序列的高效精确建模**。

---

## 二、TWIN 架构

### 2.1 两阶段框架

```
全量用户行为序列（数千条）
         ↓
  [阶段1：General Search]
  用轻量化 Target Attention 筛选 Top-K 相关行为
         ↓
  [阶段2：Exact Attention]
  对 Top-K 行为做完整 Multi-Head Attention
         ↓
      精排分数
```

### 2.2 阶段1：General Search（核心创新）

阶段1的 Target Attention 被**分解为物品侧和用户侧的预计算**：

原始 Target Attention：
$$
\alpha_i = \frac{\exp(e_t \cdot h_i)}{\sum_j \exp(e_t \cdot h_j)}
$$

TWIN 的分解：将 $h_i$ 分为物品侧部分 $h_i^{item}$ 和用户侧部分 $h_i^{ctx}$：

$$
e_t \cdot h_i \approx e_t \cdot h_i^{item} + \underbrace{e_t \cdot h_i^{ctx}}_{\text{可离线预计算}}
$$

**物品侧得分可以在物品入库时预计算并存入向量数据库**，推断时只需：
1. 加载预计算的物品侧得分（O(1)）
2. 计算用户侧动态得分
3. 求和排序，取 Top-K

整体复杂度从 $O(n \cdot d)$ 降为 $O(n + K \cdot d)$，$K \ll n$。

### 2.3 阶段2：Exact Attention

对筛选出的 $K$（一般 $K=50 \sim 200$）个行为，做标准 Multi-Head Attention：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

这一步计算精确，保证精排质量。

---

## 三、与 SIM 的对比

| 维度 | SIM（阿里） | TWIN（快手） |
|------|------------|-------------|
| 阶段1策略 | 硬属性过滤（类目/品牌匹配） | 软注意力分数检索 |
| 召回精度 | 依赖属性质量 | 语义相关性 |
| 序列长度 | 万级 | 千级～万级 |
| 延迟 | 低（硬过滤快） | 略高（需计算得分） |
| 冷启动 | 属性完整时ok | 需要历史行为 |

---

## 四、快手线上效果

- 序列长度：从 200 扩展到 **4096**
- 核心指标：**CTR +1.5%、Watch Time +0.8%**（相比截断到200的基线）
- 延迟：满足精排 <30ms 的要求

---

## 五、懂哥点评

TWIN 的精妙之处在于把"检索"嵌入到注意力计算本身，而不是像 SIM 那样靠硬规则过滤。  

物品侧预计算的思路很有工程智慧：用空间换时间，把线上推断的计算量压缩到可接受范围。  

一个值得注意的细节：阶段1召回的 Top-K 质量直接决定阶段2的上限，K 的选取需要根据实际序列长度和延迟预算来调整。

---

*参考*：[arXiv 2302.02352](https://arxiv.org/abs/2302.02352)
