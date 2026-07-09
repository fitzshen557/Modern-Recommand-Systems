# RLHF-Rank — 基于 GRPO 的 Listwise 排序

> 论文：RLHF-Rank: Reinforcement Learning from Human Feedback for Listwise Ranking in Recommendation  
> 核心贡献：将推荐精排重构为 Top-K 选择的 policy learning 问题，用 GRPO（Group Relative Policy Optimization）端到端优化用户满意度

---

## 1. 背景动机：Pointwise 精排的三大缺陷

### 1.1 缺陷一：忽略候选集竞争

Pointwise 模型对每个 item 独立打分：

```
score_i = f(user, item_i)
```

问题：如果一个用户同时命中了 3 条"美妆教程"，pointwise 会给它们打接近的分数，但展示 3 条同质内容是浪费——用户只需要 1 条。

**Listwise 视角**：应该从候选集中选出一个"组合最优"的子集，而不是"单品最优"再堆叠。

### 1.2 缺陷二：多目标融合与排序分离

工业界的标准做法：

```
# Step 1: 多任务模型预测各目标
p_ctr = MTL(user, item).ctr_head
p_cvr = MTL(user, item).cvr_head
p_duration = MTL(user, item).duration_head

# Step 2: 人工公式融合
score = α * p_ctr + β * p_cvr + γ * p_duration
```

问题：
- α, β, γ 是人工调的，无法自适应不同请求
- 融合公式与排序模型割裂，无法端到端优化
- 不同目标的量纲、分布差异大，融合不稳定

### 1.3 缺陷三：无法端到端优化用户满意度

用户的真实满意度是**对整个列表**的评价，不是对单个 item 的评价。

```
用户满意度 = g(整个列表的展示效果)
           ≠ Σ f(单个item的质量)
```

Pointwise 优化的是"每个 item 都好"，但用户要的是"这个列表让我满意"。

---

## 2. 框架：精排 = Top-K 选择的 Policy Learning

### 2.1 问题重构

把精排重新定义为：

```
给定候选集 S = {item_1, ..., item_N}
从中选出 K 个 item 组成展示列表 L
使得用户满意度 R(L) 最大
```

这是一个 **组合优化** 问题，也是一个 **sequential decision** 问题。

### 2.2 MDP 建模

```
State:    s_t = encode(user, remaining_candidates, already_selected)
Action:   a_t = 选择下一个要加入列表的 item
Reward:   r = R(最终列表 L)  # 延迟奖励，只在序列结束时给出
Policy:   π(a_t | s_t) = P(选 item a_t | 当前状态)
```

注意：这是一个 **episodic** 任务，一个 episode = 一次请求的 K 次选择。

### 2.3 Policy 网络

```python
# 状态编码
user_state = UserEncoder(user_features)
candidate_reps = ItemEncoder(all_candidates)   # (N, d)
selected_reps = ItemEncoder(already_selected)   # (t, d)

state = concat(user_state, mean(candidate_reps), mean(selected_reps))

# Action 选择
logits = state @ candidate_reps.T   # (1, N)
probs = softmax(logits / temperature)
action = sample(probs)   # 训练时采样，推理时 argmax
```

---

## 3. GRPO 在推荐中的应用

### 3.1 为什么用 GRPO 而不是 PPO

传统 PPO 需要一个 value network 估计 V(s)：

```
Advantage = R - V(s)
```

问题：
- Value network 增加模型复杂度
- 在推荐场景中 V(s) 很难准确估计（状态空间太大）

GRPO（Group Relative Policy Optimization）的巧妙之处：**不需要 value network**。

### 3.2 Request-wise Advantage

GRPO 的核心思想：**在同一个请求（request）内部做组内比较**。

```
对于一个请求，采样 G 组不同的列表 {L_1, L_2, ..., L_G}
计算每组的奖励 {r_1, r_2, ..., r_G}

Request-wise Advantage for L_i:
A_i = r_i - mean(r_1, ..., r_G)
```

直觉：
- A_i > 0：这个列表比该请求的平均水平好 → 强化
- A_i < 0：这个列表比平均差 → 抑制

### 3.3 GRPO 目标函数

```
L_GRPO = E[ Σ_i A_i * log π(L_i) ] - β * KL(π || π_ref)

其中：
- A_i = r_i - (1/G) Σ_j r_j   # 组内相对优势
- π_ref 是 SFT 后的初始策略（防止偏离太远）
- β 是 KL 惩罚系数
```

### 3.4 与标准 GRPO 的区别

在 NLP 的 RLHF 中，GRPO 的 reward 来自人类偏好标注。在推荐中：

```
NLP:   reward = human_preference(model_output)
推荐:  reward = user_feedback(impression_list)  # 隐式反馈
```

推荐的 advantage 在于：reward 是**自动的、海量的**（每次用户行为都是 reward signal），不需要人工标注。

---

## 4. 多目标 Reward 设计

### 4.1 挑战

用户反馈是多维度的：
- 点击（CTR）
- 完播/购买（CVR）
- 观看时长（Duration）
- 互动（点赞/评论/分享）

如何把多个目标融合成一个标量 reward？

### 4.2 归一化策略

```python
# Step 1: 各目标分别归一化
r_ctr = (click - mean_ctr) / std_ctr           # z-score
r_duration = (duration - mean_dur) / std_dur
r_interact = interact_score                     # 加权：like*1 + comment*2 + share*3

# Step 2: 加权融合
reward = w1 * r_ctr + w2 * r_duration + w3 * r_interact

# 或者用更优雅的方式：
# 各目标的 percentile rank，然后加权
r_ctr_norm = percentile_rank(click, distribution_ctr)
r_dur_norm = percentile_rank(duration, distribution_dur)
reward = w1 * r_ctr_norm + w2 * r_dur_norm + w3 * r_interact_norm
```

### 4.3 权重学习

关键洞察：权重 `w1, w2, w3` 不应该是人工设定的，应该从数据中学习。

```python
# 方法：把权重也作为可学习参数
w = softmax(learnable_params)   # 保证和为 1
reward = w @ [r_ctr, r_duration, r_interact]

# 训练目标：最大化 reward 的同时，w 会自动学到合理的比例
```

### 4.4 Reward 的稀疏性问题

用户的显式反馈（点赞、评论）是稀疏的，大部分 item 只有曝光没有互动。

解决方案：
- 用**曝光-点击**作为基础 reward（最密集）
- 互动行为作为**额外加分**（稀疏但信息量大）
- 用 reward shaping 给中间步骤（选择第 t 个 item 时）分配 credit

---

## 5. 与 LTR（ListNet/LambdaRank）的本质区别

### 5.1 传统 LTR

```
ListNet:  P(排列 π) ∝ Π_i φ(score_i)
          优化：-log P(ground_truth_permutation | model)

LambdaRank:
          优化：Σ_{i,j} |ΔNDCG_{ij}| * log(1 + exp(-(score_i - score_j)))
```

核心：都是 **基于排序的 loss**，优化目标是"让好的 item 排在前面"。

### 5.2 RLHF-Rank vs LTR

| 维度 | ListNet/LambdaRank | RLHF-Rank |
|------|-------------------|-----------|
| **优化目标** | 排序正确性（NDCG等） | 用户满意度（reward） |
| **决策方式** | 打分→排序（pointwise思想） | 序列选择（真正listwise） |
| **多目标** | 需要预先融合 | 端到端在 reward 中融合 |
| **探索能力** | 无（greedy排序） | 有（策略采样，exploration） |
| **反馈利用** | 只用标注数据 | 用在线隐式反馈持续学习 |

### 5.3 本质区别

**LTR 是 supervised learning**：给定标注的偏好顺序，学一个打分函数。

**RLHF-Rank 是 reinforcement learning**：给定奖励信号，学一个选择策略。

类比：
- LTR 像考试：答案已知，学怎么得分高
- RLHF-Rank 像实战：没有标准答案，赢了（用户满意）就强化

---

## 6. 工业案例：有效互动 +2% 的实现路径

### 6.1 系统架构

```
┌─────────────────────────────────────────────────┐
│           线上 Serving Pipeline                  │
├─────────────────────────────────────────────────┤
│ 召回(10000) → 粗排(500) → SFT精排(200)          │
│                                    ↓             │
│                          GRPO-Rank重排(50)       │
│                                    ↓             │
│                              展示列表(20)        │
└─────────────────────────────────────────────────┘
```

### 6.2 训练流程

```
Phase 1: SFT (Supervised Fine-Tuning)
  - 用历史数据训练初始策略 π_SFT
  - 目标：学会"合理的"选品能力（不是随机选）
  - Loss: -log π_SFT(expert_demonstration)

Phase 2: GRPO 强化学习
  - 对每个请求，用 π_SFT 采样 G=4 个不同列表
  - 计算每个列表的 reward（基于用户行为）
  - 计算 request-wise advantage
  - 更新策略：π_new = π_SFT + GRPO_update
  
Phase 3: 持续迭代
  - 每周用新数据重新跑 GRPO
  - π_ref 更新为上一轮的 π_new
```

### 6.3 +2% 有效互动的拆解

```
基线：pointwise精排 + 人工公式融合多目标
  - 有效互动率 = 12.3%

RLHF-Rank：
  - 端到端多目标优化        → +0.8%
  - Listwise 竞争感知       → +0.6%
  - 策略探索（非 greedy）    → +0.4%
  - 持续在线学习             → +0.2%
  ──────────────────────────────────
  总计：+2.0% 有效互动率 → 12.54%
```

### 6.4 关键工程经验

1. **SFT 是基础**：不做 SFT 直接 RL，策略会崩溃（选出乱七八糟的列表）
2. **G=4 足够**：每个请求采样 4 个列表做组内比较，效果和 G=8 差不多
3. **KL 约束不能省**：β=0.05 左右，防止策略跑飞
4. **Reward 归一化至关重要**：不归一化直接训，reward hacking 严重

---

## 7. 懂哥点评 🐟

### 7.1 为什么这篇重要

RLHF-Rank 的意义不在于"用 RL 做排序"（这想法早有了），而在于：

1. **把 LLM 的 RLHF 范式迁移到推荐**：证明了 GRPO 在推荐场景也 work
2. **Request-wise Advantage 的简洁性**：不需要 value network，大幅降低工程复杂度
3. **多目标融合的统一框架**：不再需要人工调 α、β、γ

### 7.2 潜在问题

1. **训练成本高**：每个请求要采样 G 个列表，每个列表要 K 步决策 → 训练数据利用率低
2. **Reward 设计的敏感性**：reward 设计不好，策略会 exploit 漏洞（reward hacking）
3. **线上部署延迟**：RL 模型的推理速度通常比 pointwise 慢

### 7.3 未来方向

- **与 LLM 结合**：直接用 LLM 作为 policy network（类似 RecLLM）
- **离线-在线协同**：离线 GRPO 粗调 + 在线 bandit 微调
- **多步反馈建模**：当前 reward 只考虑单次展示，应该考虑长期用户留存

---

## 参考链接

1. RLHF-Rank 相关思路：[RLHF for Recommendation](https://arxiv.org/abs/2312.10145)
2. GRPO 原论文（DeepSeekMath）：[arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
3. ListNet：[ListNet: Learning to Rank using Plackett-Luce Model](https://dl.acm.org/doi/10.1145/1273496.1273513)
4. LambdaRank：[Learning to Rank with NDCG](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/lambdarank.pdf)
5. PPO in Recommendation：[Top-K Off-Policy Correction](https://arxiv.org/abs/1812.02363)
6. 推荐系统中的 RL：[A Survey on Reinforcement Learning for Recommendation](https://arxiv.org/abs/2207.07159)
