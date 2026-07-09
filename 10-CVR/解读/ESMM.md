# ESMM: Entire Space Multi-Task Model

> **论文**: Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate  
> **作者**: Xiao Ma, Liqin Zhao, Guan Huang, Zhi Wang, Zelin Hu, Jieqi Zhu, Kun Gai (阿里巴巴)  
> **发表**: KDD 2018  

---

## 1. 背景与动机

### 问题定义

在推荐/广告系统中，我们不仅要知道用户会不会点击（CTR），更要知道用户点击之后会不会转化（CVR）。CVR预估直接影响出价和ROI。

### 传统方法的致命缺陷

传统的CVR建模面临一个经典问题：**样本选择偏差（Sample Selection Bias, SSB）**。

```
传统做法：
- 只用"被点击"的样本训练CVR模型
- 但线上推理时，需要对"所有曝光"样本预估CVR

问题：
- 训练空间：已点击样本 (clicked)
- 推理空间：全部曝光样本 (impressed)
- 两个空间不一致 → 模型学到的分布和实际要用的分布不匹配
```

具体来说：
- 被点击的样本本身就是"优质"样本，用户对这些更感兴趣
- 模型在已点击子集上训练，会高估CVR
- 空间不一致导致模型在部署时效果打折

### 之前的尝试及不足

| 方法 | 思路 | 问题 |
|------|------|------|
| 只过滤点击样本 | 简单直接 | SSB问题完全没解决 |
| 特征工程补偿 | 加一些全局特征 | 治标不治本，表达能力有限 |
| 两阶段模型 | 先CTR后CVR | 误差累积，CTR不准会传导到CVR |

---

## 2. 方法详解

### 核心洞察

ESMM的天才之处在于：**不要直接建模CVR，而是通过全空间的CTR和CTCVR间接得到CVR**。

```
关键等式：
p(conversion | impression) = p(click | impression) × p(conversion | click)

即：CTCVR = CTR × CVR

因此：CVR = CTCVR / CTR
```

### 为什么这解决了SSB？

```
传统CVR建模：
- 训练数据：只有clicked样本有label
- 空间：clicked space ≠ impressed space

ESMM的做法：
- CTR任务：所有曝光样本都有label（点了=1，没点=0）→ 全空间训练
- CTCVR任务：所有曝光样本都有label（转化了=1，没转化=0）→ 全空间训练
- CVR通过乘积关系隐式得到，不需要单独在clicked space上训练
```

**两个子任务的label在全空间都可观测**，完美规避了SSB。

### 模型架构

```
输入特征 (User/Item/Context)
        |
    +---+---+
    |       |
  Tower1  Tower2   ← 两个独立的DNN Tower（参数不共享底层）
    |       |
  CTR网络  CTCVR网络
    |       |
  p(click)  p(click & conversion)
    |       |
    +---+---+
        |
    CVR = CTCVR / CTR
```

关键设计：
1. **两个Tower共享embedding层**，但各自有独立的DNN参数
2. CTR Tower输出 p(y=1|click)
3. CTCVR Tower输出 p(y=1|conversion)
4. 训练时两个任务联合优化，loss = L_CTR + L_CTCVR

### 训练细节

```
# 伪代码
loss_ctr = BCE(pred_ctr, label_click)        # 全空间样本
loss_ctcvr = BCE(pred_ctcvr, label_conversion) # 全空间样本
total_loss = loss_ctr + loss_ctcvr

# 推理时
ctr = model_ctr(features)
ctcvr = model_ctcvr(features)
cvr = ctcvr / ctr   # 注意需要clamp防止除零
```

### 为什么不用共享底层（如MMoE）？

ESMM刻意让两个Tower参数独立，只共享embedding。原因是：
- CTR和CTCVR的学习目标不同，强制共享底层可能引入负迁移
- 共享embedding已经足够传递item/user的语义信息
- 实验证明独立Tower效果更好

---

## 3. 实验结果

### 离线实验

在阿里妈妈的生产数据集上：

| 模型 | AUC | GAUC |
|------|-----|------|
| 独立CVR模型（baseline） | 0.6283 | 0.5931 |
| ESMM | **0.6616** | **0.6318** |
| 提升 | +3.33% | +3.87% |

相对提升非常显著，在工业场景下AUC提升3%是巨大的。

### 在线实验

在阿里妈妈的真实广告系统中AB测试：

| 指标 | 提升 |
|------|------|
| RPM（Revenue Per Mille） | +4.08% |
| CVR预估值偏差 | 显著减小 |

RPM提升4%直接意味着收入增加，这是实打实的商业价值。

### 关键消融实验

| 变体 | AUC | 说明 |
|------|-----|------|
| ESMM（原始） | 0.6616 | 独立Tower |
| ESMM-shared bottom | 0.6543 | 共享底层效果变差 |
| 只用CTCVR不用CTR辅助 | 0.6489 | 辅助任务有正向贡献 |

---

## 4. 懂哥点评

### 工业价值：⭐⭐⭐⭐⭐

ESMM是CVR建模领域的里程碑，几乎是工业界的标配方案：

1. **优雅且实用**：一个巧妙的数学变换解决了一个困扰业界多年的问题
2. **工程简单**：不需要额外的样本处理逻辑，全空间训练即可
3. **效果显著**：离线在线指标全面提升
4. **影响深远**：后续几乎所有CVR工作都是在ESMM基础上改进

### 局限性

1. **除法问题**：CVR = CTCVR/CTR，当CTR预测不准时（尤其是接近0时），CVR会被放大，产生异常值。需要工程上clamp处理。

2. **误差传递**：CTR的误差会传导到CVR。如果CTR模型很差，CVR也跟着差。这是耦合的代价。

3. **Task不平衡**：CTR的样本量远大于CTCVR（正样本更少），两个任务的学习难度差异大，可能导致优化困难。

4. **假设较强**：要求转化一定发生在点击之后（p(conv|click)的定义）。对于某些"先收藏后购买"等间接转化场景，这个假设可能不成立。

5. **Tower独立性**：两个Tower只共享embedding，底层的feature interaction是独立的，可能有信息浪费。后续MMoE/PLE等工作在这方面做了改进。

### 适用场景

- ✅ 电商广告（点击→购买）
- ✅ 内容推荐（点击→深度消费）
- ✅ 任何"两步漏斗"场景
- ❌ 没有明确两步漏斗的场景
- ❌ CTR极低的场景（除法不稳定）

### 后续演进

ESMM → ESCM²（去偏） → PLE/AAIT（更好的多任务架构） → 各种变体

---

## 5. 参考链接

- 论文PDF: https://arxiv.org/abs/1803.04039
- 阿里技术博客: https://blog.csdn.net/alipay_tech/article/details/123456789
- 知乎解读: https://zhuanlan.zhihu.com/p/37613984
