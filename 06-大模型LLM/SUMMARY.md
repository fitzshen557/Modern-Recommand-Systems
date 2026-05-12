# 大模型 LLM × 推荐系统 汇总

> 最后更新：2026-05

## 三大范式总览

```
LLM4Rec 三大范式：

[增强器 Augmentor]          [排序器 Ranker]           [生成器 Generator]
LLM离线生成知识/特征          LLM在线打分/排序            LLM直接生成推荐列表
反哺传统CF模型               (zero/few-shot)           替代多级漏斗

代表：                       代表：                     代表：
LLMRec, RLMRec              TALLRec, InstructRec        P5, BIGRec
TagCF, KAR                  RankGPT, LLMRank            OneRec, ETEGRec

延迟：低（离线预计算）          延迟：高（LLM推断）           延迟：极高（自回归生成）
落地难度：★★☆               落地难度：★★★★              落地难度：★★★★★
```

## 方法速查表

### 增强器类（工业落地最成熟）

| 方法 | 机构 | 年份 | 增强方式 | 链接 |
|------|------|------|----------|------|
| **LLMRec** | WSDM 2024 | 2024 | LLM生成用户/物品画像补充图结构 | [论文](https://arxiv.org/abs/2311.00423) \| [详解](./LLMRec.md) |
| **RLMRec** | WWW 2024 | 2024 | LLM表征与CF表征对齐，联合训练 | [论文](https://arxiv.org/abs/2310.15950) |
| **KAR** | 阿里 | 2023 | LLM生成推理知识+直觉知识，离线预计算 | [论文](https://arxiv.org/abs/2306.10933) |
| **TagCF** | 快手 NeurIPS 2025 | 2025 | LLM生成用户角色标签增强协同过滤 | [论文](https://arxiv.org/abs/2411.xxxxx) \| [详解](./TagCF.md) |
| **RecInterpreter** | 浙大 | 2024 | LLM解释推荐行为并反哺模型 | [论文](https://arxiv.org/abs/2401.xxxxx) |

### 排序器类（研究热点，工业有限落地）

| 方法 | 机构 | 年份 | 排序方式 | 链接 |
|------|------|------|----------|------|
| **TALLRec** | 多机构 | 2023 | 指令调优LLaMA做二分类推荐 | [论文](https://arxiv.org/abs/2305.00447) |
| **InstructRec** | 微软 | 2023 | 自然语言指令驱动个性化排序 | [论文](https://arxiv.org/abs/2305.11532) |
| **RankGPT** | 多机构 | 2023 | GPT直接排列文档列表 | [论文](https://arxiv.org/abs/2304.09542) |
| **E4SRec** | 多机构 | 2024 | 高效LLM4Rec，压缩token序列 | [论文](https://arxiv.org/abs/2312.02443) |

### 生成器类（前沿探索）

| 方法 | 机构 | 年份 | 生成方式 | 链接 |
|------|------|------|----------|------|
| **P5** | Purdue | 2022 | 统一5类任务为文本生成（T5骨干） | [论文](https://arxiv.org/abs/2203.13366) |
| **BIGRec** | 多机构 | 2023 | 物品ID grounding到LLM词表 | [论文](https://arxiv.org/abs/2307.02046) |
| **TIGER** | Google | 2023 | RQ-VAE语义ID+seq2seq召回 | [论文](https://arxiv.org/abs/2305.05065) |
| **ETEGRec** | 多机构 | 2024 | 端到端生成式推荐，联合召回+排序 | [论文](https://arxiv.org/abs/2402.xxxxx) |
| **OneRec** | 快手 | 2025 | 生成式系统替代多级漏斗+DPO对齐 | [论文](https://arxiv.org/abs/2501.18253) \| [详解](./OneRec.md) |

### Agent类（探索阶段）

| 方法 | 机构 | 年份 | Agent方式 | 链接 |
|------|------|------|----------|------|
| **AgentCF** | 多机构 | 2024 | LLM Agent模拟用户行为做CF | [论文](https://arxiv.org/abs/2310.09233) |
| **RecAgent** | 人大 | 2024 | 多Agent协作推荐系统 | [论文](https://arxiv.org/abs/2306.02552) |

## LLM4Rec 核心技术问题

### 1. 语义 gap（最核心问题）
LLM擅长自然语言，CF擅长行为ID，两者表征空间不同。
- **解法A（特征融合）**：LLMRec/RLMRec，在特征层对齐
- **解法B（统一ID空间）**：TIGER/BIGRec，构建语义物品ID
- **解法C（LLM微调）**：TALLRec，直接在行为数据上微调LLM

### 2. 幻觉（Hallucination）
LLM可能生成不存在的物品名称或ID。
- 解法：Constrained Generation，限制解码只在合法物品token上

### 3. 效率（最大工业落地瓶颈）
LLM推理P99延迟通常>500ms，推荐系统要求<100ms。
- **增强器路线**：LLM离线预生成，绕过在线推断
- **蒸馏路线**：LLM→轻量学生模型，学生模型在线推断
- **投机解码**：OneRec等使用Speculative Decoding加速

### 4. 个性化能力
通用LLM缺乏用户个性化，需要在行为数据上fine-tune。
- LoRA/QLoRA：参数高效微调，成本可接受
- 检索增强（RAG）：把用户历史行为作为上下文输入LLM

## 工业落地成熟度评估

| 范式 | 成熟度 | 代表落地 |
|------|--------|---------|
| 增强器（离线特征增强） | ★★★★☆ | 阿里KAR、快手TagCF |
| 排序器（LLM在线打分） | ★★☆☆☆ | 少量低频场景 |
| 生成器（替代漏斗） | ★☆☆☆☆ | 快手OneRec（试验中）|
| Agent | ★☆☆☆☆ | 仍在研究阶段 |

## 2024-2025 关键进展

1. **TagCF（NeurIPS 2025）**：快手证明LLM标签增强在超大规模工业系统上有效
2. **OneRec（2025）**：快手生成式推荐系统首个完整工业落地报告
3. **Foundation Model for RecSys（Netflix 2025）**：Netflix用统一基础模型替代多个专用推荐模型

## 参考资源
- [LLM4Rec Awesome Papers](https://github.com/WLiK/LLM4Rec-Awesome-Papers)
- [eugeneyan.com - LLM+RecSys专题](https://eugeneyan.com/writing/recsys-llm/)
- [2025生成式推荐论文综述](https://zhuanlan.zhihu.com/p/1983610196241704717)
