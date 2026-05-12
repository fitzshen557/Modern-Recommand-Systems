# 大模型 LLM × 推荐系统

> LLM 进入推荐系统是 2023-2025 年最重要的技术浪潮之一。
> 核心范式有三：**特征增强器 / 排序器 / 端到端生成器**

---

## LLM4Rec 三大范式

```
┌─────────────────────────────────────────────────────────┐
│                  LLM 在推荐系统中的角色                   │
├─────────────────┬───────────────────┬───────────────────┤
│  特征增强器      │    排序器          │   端到端生成器     │
│ (Augmentor)     │  (Ranker)         │  (Generator)      │
│                 │                   │                   │
│ LLM 生成侧信息  │ LLM 直接打分/排序  │ LLM 生成物品序列  │
│ 反哺传统模型    │ (zero/few-shot)   │ 替代整个漏斗      │
│                 │                   │                   │
│ LLMRec, RLMRec  │ TALLRec,          │ P5, BIGRec,       │
│ TagCF, KAR      │ InstructRec       │ OneRec, ETEGRec   │
└─────────────────┴───────────────────┴───────────────────┘
```

---

## 方法列表

| 文件 | 方法 | 范式 | 年份 |
|------|------|------|------|
| [P5.md](./P5.md) | Pretrain, Personalized Prompt, Predict | 生成器 | 2022 |
| [LLMRec.md](./LLMRec.md) | LLM + Graph Augmentation | 增强器 | 2024 |
| [RLMRec.md](./RLMRec.md) | Representation Alignment | 增强器 | 2024 |
| [BIGRec.md](./BIGRec.md) | Grounding Language Model to Rec | 生成器 | 2023 |
| [TALLRec.md](./TALLRec.md) | Tuning-based Alignment LLM | 排序器 | 2023 |
| [InstructRec.md](./InstructRec.md) | Instruction Tuning for Rec | 排序器 | 2023 |
| [AgentCF.md](./AgentCF.md) | LLM Agent Collaborative Filtering | Agent | 2024 |
| [ETEGRec.md](./ETEGRec.md) | End-to-End Generative Rec | 生成器 | 2024 |
| [TagCF.md](./TagCF.md) | Tag-based Collaborative Filtering | 增强器 | 2025 |
| [OneRec.md](./OneRec.md) | 端到端生成式推荐（快手） | 生成器 | 2025 |

---

## 核心挑战

1. **语义 gap**：LLM 理解自然语言，协同过滤靠 ID，如何对齐是核心问题
2. **效率**：LLM 推理慢，推荐需要 <100ms，工程上必须蒸馏/量化
3. **幻觉**：LLM 可能生成不存在的物品 ID，需要 constrained generation
4. **个性化**：LLM 通用，个性化能力需要用户行为数据 fine-tune
