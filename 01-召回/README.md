# 召回（Retrieval / Matching）

> 召回是推荐系统的第一关，从亿级候选集中高效筛选出千级候选。
> 核心挑战：**效率 vs. 精度** 的 trade-off，以及**多路召回的融合**。

## 召回的演进脉络

```
[经典双塔] → [多兴趣] → [图召回] → [生成式ID召回] → [LLM语义召回]
  DSSM/YTB    MIND/ComiRec   PinSage     TIGER/LETTER     MoRec/LC-Rec
```

## 文件列表

| 文件 | 方法 | 年份 |
|------|------|------|
| [DSSM.md](./DSSM.md) | Deep Structured Semantic Model | 经典 |
| [YoutubeDNN.md](./YoutubeDNN.md) | Youtube Deep Neural Network | 2016 |
| [MIND.md](./MIND.md) | Multi-Interest Network with Dynamic Routing | 2019 |
| [ComiRec.md](./ComiRec.md) | Controllable Multi-Interest Framework | 2020 |
| [EBR.md](./EBR.md) | Embedding-Based Retrieval (Facebook) | 2020 |
| [GNN-Retrieval.md](./GNN-Retrieval.md) | GNN-Based Retrieval Methods | 2020-2024 |
| [MoRec.md](./MoRec.md) | Model-based Pre-training for Rec | 2023 |
| [TIGER.md](./TIGER.md) | Transformer Index for GEnerative Recommendation | 2023 |
| [LETTER.md](./LETTER.md) | Hierarchical Semantic Tokenization | 2024 |
| [LC-Rec.md](./LC-Rec.md) | LLM-Collaborative Generative Rec | 2024 |
| [UniSearch.md](./UniSearch.md) | Unified Search & Recommendation | 2025 |
