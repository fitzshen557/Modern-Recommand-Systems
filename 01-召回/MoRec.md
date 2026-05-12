# MoRec：用预训练模型 Embedding 做召回

> **论文**：Where to Go Next for Recommender Systems? ID- vs. Modality-based Recommender Models Revisited  
> **来源**：NUS（新加坡国立大学）  
> **发表**：SIGIR 2023  
> **arXiv**：https://arxiv.org/abs/2303.13835

---

## 一、核心问题

推荐系统长期依赖**协同过滤 ID embedding**（每个物品学一个独立向量）。  
随着 BERT、CLIP 等预训练模型崛起，一个自然的问题出现了：

> **用预训练模型的语义 embedding 替换 ID embedding，效果如何？**

MoRec 做了系统性的实证研究，结论令人惊讶。

---

## 二、方法框架

### 2.1 ID-based 模型（基线）

传统方式：
$$
e_i^{\text{ID}} = \text{Embedding\_Table}[i] \in \mathbb{R}^d
$$

这个向量从随机初始化开始，完全靠交互数据学习。

### 2.2 Modality-based 模型（MoRec）

MoRec 的思路：

$$
e_i^{\text{MoRec}} = \text{MLP}\left(f_\theta(x_i)\right)
$$

其中 $f_\theta$ 是预训练编码器（文本用 BERT/SentenceT5，图像用 CLIP），$x_i$ 是物品的内容（标题/图片等），MLP 做维度适配。

**关键选择**：预训练模型是否 fine-tune？MoRec 的实验结论是：
- 冻结 encoder：效果一般
- 轻量 fine-tune（只调最后几层）：效果接近甚至超越 ID-based

### 2.3 对比实验框架

论文在 **SASRec**（序列推荐）骨干上替换 embedding 来源，覆盖：
- 纯 ID embedding（基线）
- 纯 Modality embedding（MoRec）
- ID + Modality 融合

---

## 三、核心实验结论

在 Amazon 5-core 数据集（8 个类目）上：

| 场景 | 结论 |
|------|------|
| **高密度数据** | ID-based 仍略优（交互信息丰富，协同过滤优势明显） |
| **低密度/冷启动** | MoRec 显著优于 ID-based（语义迁移能力强） |
| **新物品** | MoRec 天然支持，ID-based 需重新训练 |
| **跨域迁移** | MoRec 泛化能力远强于 ID-based |

**结论**：预训练模型 embedding 在**冷启动和稀疏数据**场景下已经可以超越 ID embedding，未来两种方式的融合是趋势。

---

## 四、对工业界的启示

1. **双塔召回**：物品侧用预训练文本/图像 embedding，可以大幅降低冷启动物品的冷启时间（不需要积累交互数据）
2. **统一表征**：ID embedding + 内容 embedding 的 concatenation 是工业界常见做法
3. **Embedding 更新策略**：新物品上线时，内容 embedding 即刻可用；随交互数据增多，逐渐加入 ID 信号

---

## 五、懂哥点评

MoRec 的贡献不是提出新模型，而是**重新界定了推荐系统的 foundation**。  

它打破了"ID embedding 是唯一正解"的迷信，为后续大量 LLM4Rec 工作提供了理论基础。  
实际上，后续的 TIGER、LC-Rec、E4SRec 等工作都可以看作 MoRec 思路的深化。

---

*参考*：[arXiv 2303.13835](https://arxiv.org/abs/2303.13835) | [GitHub](https://github.com/westlake-repl/MoRec)
