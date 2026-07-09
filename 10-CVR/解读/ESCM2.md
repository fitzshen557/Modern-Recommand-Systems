# ESCM²: Entire Space Counterfactual Multi-Task Model

> **论文**: Entire Space Counterfactual Multi-Task Model for Post-Click Conversion Rate Estimation  
> **作者**: Bin Hu, Zhiqiang Zhang, Chuan Yu, Yanru Qu, Wei Zhang, Zhirong Liu, Wenwu Ou (阿里巴巴)  
> **发表**: SIGIR 2022  

---

## 1. 背景与动机

### ESMM解决了SSB，但引入了新问题

ESMM通过CTCVR = CTR × CVR的分解，在全空间训练，解决了样本选择偏差（SSB）。但仔细看这个等式：

```
ESMM的隐含假设：
p(y_cv | x) = p(y_ct | x) × p(y_cv | x, y_ct = 1)

关键问题：右边的 p(y_cv | x, y_ct = 1) 仍然是在"点击"条件下的CVR
- 这本质上还是在clicked space上建模
- 只不过通过CTCVR/CTR的除法"绕"了一圈
```

### 残留的偏差：选择偏差 + 不确定性偏差

ESCM²指出ESMM存在两个残余偏差：

**偏差1：选择偏差仍然存在**

```
ESMM的CVR Tower虽然在全空间训练（通过CTCVR），但：
- 在推理时，CVR = CTCVR / CTR
- 当CTR预测不准时（尤其是低CTR样本），CVR的估计会不稳定
- 这种不稳定性等价于一种"软性"的选择偏差
```

**偏差2：不确定性（Uncertainty）**

```
CTCVR是一个极小的值（比如CTR=0.05, CVR=0.1, CTCVR=0.005）
- 两个小概率相乘，信号极其微弱
- CTCVR的正样本极其稀疏（转化本身就很稀少）
- 模型很难从这么稀疏的信号中学到好的CVR表示
```

### 因果视角的分析

ESCM²用因果推断的框架重新审视CVR问题：

```
因果图：
S → C → Y
(S=选择/点击, Y=转化)

ESMM的做法相当于：
P(Y|S) = P(C|S) × P(Y|C,S)

但因果推断告诉我们，真正需要的是：
P(Y|do(C=1), S) — 在干预"强制点击"后的转化概率

这和 P(Y|C=1, S) 的区别在于：
- P(Y|C=1, S) 包含了"自然选择点击"的confounding
- P(Y|do(C=1), S) 才是我们想要的CVR
```

---

## 2. 方法详解

### 核心思路

ESCM²同时使用两种因果去偏策略来解决ESMM的残余偏差：

1. **IPS（Inverse Propensity Scoring）**：逆倾向得分加权
2. **DR（Double Robust）**：双重鲁棒估计

两者结合ESMM的CTCVR框架，形成三种互补的去偏方法。

### 方法1：IPS-based ESCM²

```
目标：对CVR模型的loss进行倾向得分加权，消除选择偏差

原始CVR loss（只在clicked样本上）：
L_CVR = (1/|D_click|) × Σ_{x∈D_click} loss(f_CVR(x), y)

IPS加权后：
L_IPS = (1/|D_all|) × Σ_{x∈D_all} I(click=1) / e(x) × loss(f_CVR(x), y)

其中：
- e(x) = p(click|x)，倾向得分（由CTR模型给出）
- I(click=1) 是指示函数，只有点击样本贡献梯度
- 分母 |D_all| 是全空间样本数（不是|D_click|）

直觉理解：
- 一个被点击的样本，如果它的点击概率只有1%，那它"代表"了100个类似的未点击样本
- 所以给它的loss乘以权重 1/0.01 = 100
- 这样就让clicked样本的梯度"代表"了全空间的梯度
```

**IPS的问题**：

```
当e(x)很小时，1/e(x)会爆炸
- 极端case：某个样本CTR=0.001，权重=1000
- 这个样本的loss会主导整个梯度
- 训练极度不稳定

解决方案：截断（clipping）
e_clip(x) = max(e(x), ε)
L_IPS_clipped = (1/|D_all|) × Σ I(click=1) / e_clip(x) × loss(f_CVR(x), y)

但截断会引入新的偏差，是个trade-off
```

### 方法2：DR-based ESCM²（核心贡献）

```
Double Robust的核心思想：
结合IPS和imputation（填补），两者任一正确就能得到无偏估计

DR estimator：
L_DR = (1/|D_all|) × Σ_{x∈D_all} [
    loss(f_CVR(x), y_hat)                                              # imputation项
    + I(click=1) / e(x) × [loss(f_CVR(x), y) - loss(f_CVR(x), y_hat)]  # 残差修正项
]

其中：
- y_hat = f_impute(x) 是imputation模型的预测
- 第一项：对所有样本（包括未点击），用imputation模型"填补"label
- 第二项：对点击样本，用真实label修正imputation的误差

两个极端情况：
1. 如果imputation完美（y_hat = y）：第二项=0，只留imputation，零方差
2. 如果CTR模型完美（e(x)准确）：第二项正确修正，无偏

只要imputation或CTR任一准确，DR就是无偏的 → "Double Robust"
```

**DR vs IPS的对比**：

```
IPS:
- 无偏（理论上）
- 高方差（小CTR样本权重爆炸）
- 不需要imputation模型

DR:
- 无偏（double robustness）
- 低方差（imputation兜底）
- 需要额外的imputation模型

工程实践中，DR几乎总是优于IPS
```

### 方法3：间接法（ESMM原始方法）

```
保留ESMM的 CTCVR = CTR × CVR 分解
CVR = CTCVR / CTR

作为DR的直接估计的补充，提供另一条CVR信号
```

### 完整架构

```
                    Input Features
                         |
              +----------+----------+
              |          |          |
          Embedding   Embedding  Embedding
              |          |          |
          CTR Tower  CVR Tower  CTCVR Tower
              |          |          |
          e(x)      f_CVR(x)   f_CTCVR(x)
              |          |          |
         p(click)   p(conv|click)  p(conv)
              |          |          |
              +----+-----+-----+---+
                   |           |
               DR Loss     CTCVR/CTR
                   |           |
              f_CVR_DR    f_CVR_indirect
                   |           |
                   +-----+-----+
                         |
                   最终CVR输出
              (加权融合 或 选择其一)
```

### Imputation模型设计

```
Imputation模型的训练：
- 输入：全空间特征
- 输出：y_hat(x)，预测"如果这个样本被点击，转化概率是多少"
- 训练数据：只能在clicked样本上训练（因为只有clicked样本有真实的转化label）

这里有个微妙之处：
- imputation模型本身也在clicked space上训练
- 但它只需要"大致准确"就行，不需要完美
- DR的double robustness保证了即使imputation有偏差，只要CTR准确，整体仍无偏
```

### 训练Loss汇总

```python
# 总Loss = L_DR + L_CTCVR + L_IMPUTE

# 1. DR Loss（CVR直接估计，全空间）
L_DR = mean_over_all_samples([
    loss_BCE(f_CVR(x), y_hat(x)) +                          # imputation
    I(click=1) / e_clip(x) * (loss_BCE(f_CVR(x), y) -       # 残差修正
                              loss_BCE(f_CVR(x), y_hat(x)))
])

# 2. CTCVR Loss（间接估计，全空间）
L_CTCVR = mean_over_all_samples([
    loss_BCE(f_CTCVR(x), y_conv)
])

# 3. Imputation Loss（只在clicked样本上）
L_IMPUTE = mean_over_clicked_samples([
    loss_BCE(f_impute(x), y_conv)
])

# 4. CTR Loss（全空间）
L_CTR = mean_over_all_samples([
    loss_BCE(f_CTR(x), y_click)
])

total_loss = L_DR + L_CTCVR + L_CTR + L_IMPUTE
```

### 推理时的CVR输出

```
推理时可以用两种方式得到CVR：

方式1：直接法（DR）
cvr_direct = f_CVR(x)

方式2：间接法（ESMM）
cvr_indirect = f_CTCVR(x) / f_CTR(x)

方式3：融合
cvr_final = α × cvr_direct + (1-α) × cvr_indirect

实践中，方式3通常最好，因为两种方法互补
```

---

## 3. 实验结果

### 离线实验

在阿里巴巴的生产广告数据集上：

| 模型 | AUC | GAUC |
|------|-----|------|
| ESMM | 0.6616 | 0.6318 |
| ESCM²-IPS | 0.6652 | 0.6357 |
| ESCM²-DR | **0.6724** | **0.6438** |
| ESCM²-DR vs ESMM | +1.63% | +1.90% |

### 在线AB实验

在阿里巴巴真实广告系统中：

| 指标 | ESMM | ESCM²-DR | 提升 |
|------|------|----------|------|
| RPM | baseline | +2.54% | 显著提升 |
| CVR预估偏差 | - | 减小30%+ | 更准确 |
| 广告主ROI | baseline | +1.8% | 正向 |

### 关键消融实验

| 设置 | AUC | 说明 |
|------|-----|------|
| 只用IPS | 0.6652 | 方差大，效果有限 |
| 只用DR | 0.6724 | 稳定优于IPS |
| DR + 间接融合 | 0.6741 | 融合进一步提升 |
| 不用imputation | 0.6689 | imputation对DR贡献大 |
| 不用截断 | 训练崩溃 | 截断是必须的 |

---

## 4. 懂哥点评

### 工业价值：⭐⭐⭐⭐

ESCM²是一个理论深度和工程实践结合得很好的工作：

1. **因果视角清晰**：用因果推断框架统一理解了CVR建模的各种偏差
2. **DR方法实用**：Double Robust在实际中确实比纯IPS稳定得多
3. **兼容ESMM**：可以在ESMM基础上无缝升级，迁移成本低

### 局限性

1. **工程复杂度**：需要维护imputation模型、CTR模型、CVR模型、CTCVR模型，至少4个组件
2. **超参数敏感**：clip阈值ε、融合权重α等需要仔细调参
3. **Imputation模型的偏差**：imputation本身在clicked space训练，如果偏差很大，DR的效果会打折扣
4. **训练不稳定**：IPS部分的梯度方差大，即使有clip，在训练初期也可能不稳定
5. **理论假设**：Double Robust的"任一正确即无偏"要求CTR或imputation至少一个是一致的（consistent），如果两个都不准，效果可能不如ESMM

### 适用场景

- ✅ 已有ESMM在线上运行，想进一步提升精度
- ✅ 工程团队有能力维护复杂的多模型系统
- ✅ 对CVR精度要求极高的场景（高客单价、大预算广告主）
- ❌ 快速迭代的早期项目（ESMM更简单有效）
- ❌ 工程资源有限，难以维护imputation模型

### ESMM → ESCM² 升级路径

```
Step 1: 先跑ESMM，确认baseline
Step 2: 加CTR Tower（ESMM本来就有）
Step 3: 训练imputation模型（在clicked样本上训练CVR模型）
Step 4: 加DR loss，逐步替代纯CTCVR分解
Step 5: 融合直接估计和间接估计
Step 6: AB验证，调参
```

---

## 5. 参考链接

- 论文PDF: https://arxiv.org/abs/2201.05547
- 阿里技术博客: https://mp.weixin.qq.com/s/ESCM2_blog
- 因果推断在推荐中的应用综述: https://arxiv.org/abs/2110.05054
- Double Robust估计理论: https://arxiv.org/abs/1103.4541
