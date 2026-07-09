# HiGR — 层次规划生成式 Listwise 推荐（腾讯 2026）

> 论文：HiGR: Hierarchical Generative Recommendation with Intent Planning and Parallel Decoding  
> 来源：腾讯 2026  
> 核心贡献：提出"编码—规划—生成—对齐"一体化范式，将推荐从判别式排序升级为生成式列表规划，推理速度提升 5 倍

---

## 1. 背景动机：传统级联架构的三大痛点

### 1.1 传统级联架构回顾

```
召回(10000+) → 粗排(500~1000) → 精排(50~200) → 重排(20~50) → 展示(20)
```

每个阶段独立训练、独立优化，通过截断传递。

### 1.2 痛点一：目标不一致（Objective Misalignment）

```
召回目标：多样性、覆盖率
粗排目标：近似精排排序（计算效率优先）
精排目标：pointwise 相关性（CTR/CVR）
重排目标：列表多样性、商业目标
```

每个阶段优化不同的 surrogate objective，**最终用户满意度没有在任何阶段被直接优化**。

类比：一个公司里，市场部、产品部、运营部各自 KPI 都很好，但用户就是不买账。

### 1.3 痛点二：误差累积（Error Cascading）

```
召回漏了好内容 → 粗排无法捞回 → 精排看不到 → 用户永远看不到
```

每一级的截断都是不可逆的信息损失：
- 召回截断 95% 的候选
- 粗排再截断 80%
- 精排再截断 75%

**好内容如果在召回阶段就丢了，后面再好的模型也救不回来。**

### 1.4 痛点三：GPU 利用不足

```
传统 pipeline 的执行方式：

Timeline:
召回 ████░░░░░░░░░░░░░░░░
粗排 ░░░░████░░░░░░░░░░░░
精排 ░░░░░░░░████░░░░░░░░
重排 ░░░░░░░░░░░░████░░░░

GPU 利用率：每个阶段独立使用，其他时间 GPU 空闲
```

4 个阶段串行执行，每个阶段的 GPU 利用率都不满，整体延迟 = 各阶段延迟之和。

---

## 2. HiGR 框架：编码—规划—生成—对齐

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    HiGR Framework                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ① 编码 (Encoder)                                       │
│     User + Items → Unified Representation               │
│           ↓                                             │
│  ② 规划 (Intent Planner)                                │
│     决定"这个列表应该包含什么类型的内容"                    │
│     → Intent Sequence: [美妆教程, 穿搭灵感, 好物分享]      │
│           ↓                                             │
│  ③ 生成 (Generator)                                     │
│     对每个 intent slot，并行解码出具体 item                │
│     → 美妆教程 → item_37, 穿搭灵感 → item_152, ...       │
│           ↓                                             │
│  ④ 对齐 (Alignment)                                     │
│     用隐式反馈直接优化列表质量                             │
│     → DPO/GRPO style preference optimization            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2.2 统一编码器（Unified Encoder）

```python
# 不再是"召回一套参数，精排一套参数"
# 一个共享的 encoder 处理所有 item

class UnifiedEncoder:
    def __init__(self):
        self.item_encoder = TransformerEncoder(d_model=256, n_layers=6)
        self.user_encoder = TransformerEncoder(d_model=256, n_layers=3)
    
    def forward(self, user_seq, candidate_items):
        user_repr = self.user_encoder(user_seq)         # (1, d)
        item_reprs = self.item_encoder(candidate_items) # (N, d)
        
        # Cross-attention: user 看所有 item
        item_reprs = cross_attention(
            query=item_reprs,
            key=user_reprs,
            value=user_reprs
        )
        return user_repr, item_reprs
```

**关键**：所有候选 item 共享同一个 encoder，不再有"召回模型"和"精排模型"的参数割裂。

### 2.3 列表级意图规划（Intent Planner）

这是 HiGR 最核心的创新：**不直接选 item，先选"坑位类型"**。

```python
class IntentPlanner:
    """
    输入：用户历史行为序列
    输出：一个列表应该包含的"内容类型"序列
    
    类比：编辑先定杂志的栏目规划（封面故事、时尚专栏、美食推荐），
         再决定每个栏目放什么具体内容。
    """
    
    def __init__(self, vocab_size=1024, d_model=256):
        self.planner = TransformerDecoder(d_model=d_model)
        self.intent_embeddings = nn.Embedding(vocab_size, d_model)
        # intent vocabulary: 预定义的 1024 种内容类型
    
    def plan(self, user_repr, num_slots=20):
        """
        自回归生成意图序列：
        slot_1 = "美妆教程"
        slot_2 = "穿搭灵感"
        slot_3 = "美妆教程"（不同类型的具体美妆内容）
        ...
        """
        intent_seq = [START_TOKEN]
        for t in range(num_slots):
            next_intent = self.planner(
                query=intent_seq,
                context=user_repr
            )
            intent_seq.append(argmax(next_intent))
        
        return intent_seq  # [type_1, type_2, ..., type_20]
```

**为什么先规划意图？**

1. **降低搜索空间**：从 10000 个 item 中选 20 个 → 从 1024 种类型中选 20 个
2. **保证多样性**：类型层面的多样性比 item 层面更容易控制
3. **语义可控**：可以解释"为什么推这个列表"（用户最近关注美妆+穿搭）

### 2.4 物料级生成解码（Item Generator）

```python
class ItemGenerator:
    """
    给定意图序列，为每个 slot 生成具体的 item
    """
    
    def __init__(self, d_model=256):
        self.decoder = TransformerDecoder(d_model=d_model)
        self.item_proj = nn.Linear(d_model, num_items)  # 映射到 item vocab
    
    def decode(self, intent_seq, item_reprs):
        """
        对每个 intent slot，从候选集中选最匹配的 item
        
        关键创新：并行解码，不是自回归！
        """
        # 所有 slot 同时解码（并行）
        slot_reps = self.decoder(
            query=intent_seq,     # (20, d) - 20 个 intent
            context=item_reprs    # (N, d) - 所有候选 item
        )
        
        # 每个 slot 独立打分
        scores = slot_reps @ item_reprs.T  # (20, N)
        
        # 取每个 slot 的 top-1（或带多样性约束的 top-k）
        selected = []
        for slot_idx in range(20):
            best_item = argmax(scores[slot_idx])
            selected.append(best_item)
        
        return selected
```

### 2.5 列表级偏好对齐（List-level Alignment）

```python
class ListAlignment:
    """
    用用户隐式反馈直接优化列表质量
    类似 DPO（Direct Preference Optimization）的思路
    """
    
    def dpo_loss(self, list_preferred, list_rejected, policy, ref_policy):
        """
        list_preferred: 用户实际高互动的列表
        list_rejected:  用户跳过的列表
        
        优化目标：让 policy 给 preferred list 更高的概率
        """
        log_ratio_pref = log(policy(list_preferred)) - log(ref_policy(list_preferred))
        log_ratio_rej  = log(policy(list_rejected)) - log(ref_policy(list_rejected))
        
        loss = -log(sigmoid(β * (log_ratio_pref - log_ratio_rej)))
        return loss
```

**关键**：对齐的粒度是**整个列表**，不是单个 item。

```
传统方法：item_A 被点击 → 给 item_A 正反馈
HiGR：    列表 [A, B, C, D, E] 用户看了很久 → 给整个列表正反馈
```

---

## 3. 推理速度提升 5 倍的秘密：并行生成

### 3.1 传统自回归生成的瓶颈

```
传统生成式推荐（如 TIGER, RecLLM）：

生成过程：
Step 1: P(item_1 | user) → 选 item_1
Step 2: P(item_2 | user, item_1) → 选 item_2
Step 3: P(item_3 | user, item_1, item_2) → 选 item_3
...
Step 20: P(item_20 | ...) → 选 item_20

延迟 = 20 × 单次 forward pass
```

自回归生成是**串行**的，每一步都依赖前一步的输出。

### 3.2 HiGR 的并行策略

```
HiGR 的两阶段并行：

Phase 1: 意图规划（自回归，但很快）
  - 只需要生成 20 个 intent token
  - Intent vocabulary 小（1024），解码快
  - 延迟 ≈ 5 × 单次 forward（因为有 KV cache）

Phase 2: 物料生成（完全并行）
  - 20 个 intent slot 同时解码
  - 每个 slot 独立从候选集中选 item
  - 延迟 ≈ 1 × 单次 forward（并行计算）

总延迟 ≈ 6 × forward pass
vs 传统自回归 = 20 × forward pass
速度提升 ≈ 3~5x
```

### 3.3 为什么能并行

```
传统：P(item_t | item_1, ..., item_{t-1})
  → item_t 依赖前面的 item，必须串行

HiGR：P(item_t | intent_t, all_candidates)
  → 每个 item_t 只依赖自己的 intent_t 和全局候选集
  → 不同 slot 之间条件独立（给定 intent）
  → 可以并行！
```

**牺牲了什么？** Slot 之间的条件依赖。
**换来了什么？** 推理速度 3-5x，且在工程上更易部署。

---

## 4. 实验结果

### 4.1 离线实验

| 方法 | NDCG@20 | Recall@20 | Latency |
|------|---------|-----------|---------|
| 传统级联（DIN+重排） | 0.423 | 0.512 | 45ms |
| TIGER（自回归生成） | 0.456 | 0.548 | 180ms |
| OneRec（快手） | 0.461 | 0.553 | 60ms |
| **HiGR** | **0.492** | **0.589** | **35ms** |

**相对 SOTA 提升：**
- NDCG: +6.7%（vs OneRec）
- Recall: +6.5%（vs OneRec）
- 延迟: 降低 42%（vs 传统级联）

### 4.2 线上 A/B 测试

腾讯某内容推荐场景，7天 A/B 测试：

| 指标 | 基线（级联架构） | HiGR | 提升 |
|------|-----------------|------|------|
| 人均观看时长 | 23.5 min | 25.1 min | +6.8% |
| 人均消费深度 | 18.3 条 | 20.1 条 | +9.8% |
| 有效互动率 | 8.2% | 8.9% | +8.5% |
| 次日留存 | 62.3% | 64.1% | +2.9% |

### 4.3 消融实验

| 组件 | NDCG@20 | 说明 |
|------|---------|------|
| 完整 HiGR | 0.492 | — |
| - Intent Planner | 0.451 | 退化为直接生成 item |
| - Parallel Decoding | 0.493 | 用自回归解码（效果几乎不变，但延迟 5x） |
| - List Alignment | 0.468 | 只用 SFT，不做偏好对齐 |
| + Larger Encoder | 0.501 | 6层 → 12层，离线提升，线上延迟增加 |

**关键发现：**
- Intent Planner 是效果提升的主要来源（-4.1 NDCG）
- Parallel Decoding 不影响效果，只影响速度
- List Alignment 贡献 +2.4 NDCG

---

## 5. 懂哥点评 🐟

### 5.1 HiGR vs OneRec（快手）

| 维度 | HiGR（腾讯） | OneRec（快手） |
|------|-------------|---------------|
| **架构** | 两阶段（规划+生成） | 单阶段直接生成 |
| **并行性** | 意图串行 + 物料并行 | 完全自回归 |
| **延迟** | 35ms（快） | 60ms（较快） |
| **效果** | NDCG 0.492 | NDCG 0.461 |
| **可控性** | 意图层面可干预 | 黑盒生成 |
| **训练复杂度** | 高（4阶段训练） | 中（2阶段训练） |

### 5.2 HiGR 的工程创新点

1. **意图解耦是神来之笔**：
   - 把"选什么类型"和"选什么 item"分开
   - 意图空间小且语义清晰，容易优化
   - 物料选择可以并行，速度大幅提升

2. **List-level Alignment 是价值核心**：
   - 首次在工业级系统上验证了列表级 DPO
   - 用隐式反馈（观看时长、互动）直接优化列表质量
   - 不再依赖人工设计的排序公式

3. **统一 Encoder 打破阶段壁垒**：
   - 召回、粗排、精排共享参数
   - 消除了"目标不一致"的根本问题
   - 但工程挑战大：需要统一所有阶段的训练数据

### 5.3 潜在问题

1. **Intent Vocabulary 的设计**：1024 种类型够不够？怎么定义？
   - 太小：表达力不够
   - 太大：规划阶段变慢
   - 建议：用聚类从数据中自动发现 intent

2. **Slot 独立性假设**：并行生成假设 slot 之间条件独立，但用户感知列表是整体的
   - 可能丢失 slot 之间的协同效应
   - 解法：后处理阶段做 swap/reorder

3. **训练流程复杂**：4 个阶段（Encoder SFT → Planner SFT → Generator SFT → Alignment）
   - 工程落地门槛高
   - 每个阶段的数据构造都有讲究

### 5.4 行业影响

HiGR 代表了一个趋势：**推荐系统正在从"判别式 pipeline"走向"生成式一体化"**。

```
2018-2022: 判别式 pipeline 时代
  各阶段独立，手工拼接，pointwise 打分

2023-2025: 生成式探索时代
  TIGER, RecLLM 等用 LLM 做推荐，但太慢

2026+: 高效生成式时代（HiGR, OneRec 等）
  保留生成式的效果优势，解决延迟问题
  端到端优化，消灭 pipeline 冗余
```

---

## 参考链接

1. HiGR（预期）：腾讯 2026 技术博客 / 待发表论文
2. OneRec（快手）：[OneRec: Unifying Retrieve, Rerank and Generate in One Recommender](https://arxiv.org/abs/2403.03393)
3. TIGER（生成式推荐）：[Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)
4. DPO 原论文：[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)
5. 推荐系统级联架构问题：[How to Index Item IDs for Recommendation Foundation Models](https://arxiv.org/abs/2405.07435)
6. 生成式推荐综述：[Generative Recommendation: A Survey](https://arxiv.org/abs/2405.15983)
