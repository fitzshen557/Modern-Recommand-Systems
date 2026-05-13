# 召回技术汇总

> 最后更新：2026-05

## 技术全景

```
召回演进路线：
双塔向量召回 → 多兴趣召回 → 图神经网络召回 → 生成式ID召回 → LLM语义召回
(DSSM/YTB)    (MIND/ComiRec)  (PinSage/LightGCN)  (TIGER/LETTER)   (MoRec/LC-Rec)
```

## 方法速查表

| 方法 | 机构 | 年份 | 核心思想 | 适用场景 | 链接 |
|------|------|------|----------|----------|------|
| **DSSM** | Microsoft | 2013 | 双塔+余弦相似度，用户/物品各一个Tower | 通用双塔基线 | [论文](https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/) |
| **YoutubeDNN** | Google | 2016 | 深度召回先驱，负采样softmax | 工业级大规模召回 | [论文](https://dl.acm.org/doi/10.1145/2959100.2959190) |
| **MIND** | 阿里 | 2019 | 胶囊网络多兴趣向量，K个兴趣分别召回 | 兴趣多样的用户 | [论文](https://arxiv.org/abs/1904.08030) \| [详解](./MIND.md) |
| **ComiRec** | 阿里 | 2020 | 自注意力多兴趣+多样性控制参数 | 需要调控多样性 | [论文](https://arxiv.org/abs/2005.09666) |
| **SDM** | 阿里 | 2019 | 长短期兴趣分离建模 | 用户行为序列较长 | [论文](https://arxiv.org/abs/1909.00385) |
| **SINE** | 阿里 | 2021 | 稀疏兴趣网络，自适应兴趣数量 | 兴趣分布稀疏 | [论文](https://arxiv.org/abs/2102.09267) |
| **EBR** | Facebook | 2020 | FAISS向量检索工程实践全流程 | 工业级ANN落地 | [论文](https://arxiv.org/abs/2006.11632) |
| **PinSage** | Pinterest | 2018 | 图卷积+随机游走，工业级GNN召回 | 内容图谱丰富 | [论文](https://arxiv.org/abs/1806.01973) |
| **LightGCN** | 多机构 | 2020 | 轻量图卷积，去掉非线性激活 | 协同过滤增强 | [论文](https://arxiv.org/abs/2002.02126) |
| **SimGCL** | 多机构 | 2022 | 图对比学习，随机噪声增强 | 数据稀疏场景 | [论文](https://arxiv.org/abs/2112.08679) |
| **UniSRec** | 微软 | 2022 | 跨域序列推荐统一表征 | 多域迁移 | [论文](https://arxiv.org/abs/2206.05941) |
| **MoRec** | NUS | 2023 | 预训练模型embedding替代ID embedding | 冷启动/跨域 | [论文](https://arxiv.org/abs/2303.13835) \| [详解](./MoRec.md) |
| **TIGER** | Google | 2023 | RQ-VAE语义ID+生成式召回 | 冷启动+端到端 | [论文](https://arxiv.org/abs/2305.05065) \| [详解](./TIGER.md) |
| **LETTER** | 多机构 | 2024 | 层次化语义Token，对齐推荐目标 | TIGER改进 | [论文](https://arxiv.org/abs/2403.17536) |
| **LC-Rec** | 浙大 | 2024 | LLM与协同过滤对齐的生成式召回 | LLM+CF融合 | [论文](https://arxiv.org/abs/2311.09965) |
| **UniSearch** | 快手 | 2025 | 统一搜索与推荐的生成式召回 | 搜推统一 | [论文](https://arxiv.org/abs/2501.xxxxx) |

## 工业实践要点

### 多路召回融合
现代推荐系统普遍采用多路召回策略：
- **协同过滤路**：双塔/多兴趣，捕获行为相似性
- **内容召回路**：基于标题/图片/标签的语义召回
- **Graph路**：GNN捕获高阶关系
- **热门路**：兜底保证基础覆盖率
- **生成式路**（新兴）：TIGER/LC-Rec等，冷启动物品友好

融合策略：RRF（Reciprocal Rank Fusion）或学习式打分融合。

### 向量索引选型
| 场景 | 推荐方案 |
|------|---------|
| 千万级以内 | FAISS IVF-PQ |
| 亿级，精度优先 | HNSW |
| 亿级，成本优先 | ScaNN (Google) |
| 实时更新频繁 | Milvus / Weaviate |

### 2024-2025 新趋势
1. **生成式召回主流化**：TIGER框架工业落地加速，字节/快手均有布局
2. **语义ID vs. 协同ID融合**：两者不再对立，LC-Rec等工作做到联合优化
3. **LLM语义向量**：Sentence-T5/E5等文本embedding直接用于物品表示，冷启动效果大幅提升
4. **实时个性化**：在线学习+流式更新向量库，降低新物品冷启时间

## 参考资源
- [Awesome Recsys - 召回相关](https://github.com/jihoo-kim/awesome-RecSys)
- [A Comprehensive Survey on Retrieval Methods in Recommender Systems (ACM 2025)](https://dl.acm.org/doi/full/10.1145/3771925)
- [工业界召回实践汇总 - Doragd](https://github.com/Doragd/Algorithm-Practice-in-Industry)
