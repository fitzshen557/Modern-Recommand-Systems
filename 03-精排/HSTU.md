# HSTU：Meta 的新一代推荐 Backbone

> **论文**：Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations  
> **来源**：Meta AI  
> **发表**：ICML 2024  
> **arXiv**：https://arxiv.org/abs/2402.17152

---

## 一、背景与动机

Meta 的推荐系统在 2024 年做了一次重大架构升级：**用 HSTU 替换传统 MLP-based 精排模型**。

传统推荐精排的问题：
- 特征交叉依赖手工设计的 FM/DCN 结构
- Transformer 直接用于推荐序列计算复杂度 $O(n^2)$，难以处理长序列
- 用户行为序列与内容特征分开处理，信息割裂

HSTU 的核心思路：**把推荐问题建模为序列转换（Transduction）问题**，用类 Transformer 结构统一处理行为序列。

---

## 二、HSTU 架构详解

### 2.1 输入表示

用户的每一次"行动"（action）被表示为一个 token：
$$
x_t = [e_{\text{item}}, e_{\text{action\_type}}, e_{\text{timestamp}}, \ldots]
$$

不同类型的行为（点击/购买/收藏/跳过）被统一表示为 token，形成一个**混合行为序列**。

### 2.2 层次化注意力机制

HSTU 的核心是**层次化（Hierarchical）**设计：

**局部层**（Local Block）：在固定窗口 $w$ 内做注意力，复杂度 $O(nw)$：
$$
\text{LocalAttn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} \odot M_{\text{local}}\right) V
$$

**全局层**（Global Block）：对局部块的代表向量做跨块注意力：
$$
\text{GlobalAttn}(\tilde{Q}, \tilde{K}, \tilde{V}) = \text{softmax}\left(\frac{\tilde{Q}\tilde{K}^T}{\sqrt{d}}\right) \tilde{V}
$$

总体复杂度从 $O(n^2)$ 降为 $O(n \cdot w + (n/w)^2)$，支持 **万级** 用户行为序列。

### 2.3 序列转换（Transduction）目标

HSTU 不仅做排序，还做**生成式预测**：
$$
P(\text{next action} \mid \text{history}) = \text{Softmax}(\text{HSTU}(\mathbf{x}_{1:t}) \cdot E^T)
$$

其中 $E$ 是物品 embedding 矩阵。训练时多任务联合：
- CTR 预测（分类）
- 下一个行为预测（生成）
- 时间预测（回归）

### 2.4 万亿参数扩展

论文展示了 HSTU 可以扩展到 **万亿（Trillion）参数**：
- Embedding 表：~1T 参数（物品数量巨大）
- Transformer 层：~10B 参数
- 通过模型并行 + 稀疏激活实现

---

## 三、工业实验结果

在 Meta 内部 Feed 推荐上：
- 相比上一代基于 MLP 的系统：**Engagement +12%**
- 长序列处理效率：**10x 以上提升**
- 上线后的 A/B 测试：**多个核心指标显著正向**

---

## 四、关键设计选择

| 设计点 | 选择 | 理由 |
|--------|------|------|
| 位置编码 | 时间戳 embedding（非绝对位置） | 行为序列时间间隔不均匀 |
| 注意力掩码 | 因果掩码（causal） | 防止未来信息泄露 |
| 激活函数 | SwiGLU | 比 ReLU/GELU 效果好 |
| 归一化 | RMSNorm | 比 LayerNorm 更稳定 |
| 序列长度 | 支持 $n > 10000$ | 长序列是核心优势 |

---

## 五、懂哥点评

HSTU 是 2024 年工业推荐系统最重要的架构革新之一。它把推荐精排从"特征工程 + MLP"时代推进到了"序列 Transformer"时代。

几个值得注意的点：
1. **行为统一表示**：把点击/购买/跳过统一为 token，是工程哲学上的转变
2. **万亿参数**：embedding 表的规模决定了系统上限，这是 Meta 的基础设施优势
3. **生成与判别联合**：这是和 GPT 时代 NLP 一样的思路，推荐也在朝这个方向走

对国内厂商的参考价值：局部 + 全局层次注意力的思路值得借鉴，具体超参（窗口大小、块数）需根据序列长度分布调整。

---

*参考*：[arXiv 2402.17152](https://arxiv.org/abs/2402.17152) | [Meta AI Blog](https://ai.meta.com/)
