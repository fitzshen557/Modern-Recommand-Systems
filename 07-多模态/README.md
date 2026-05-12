# 多模态推荐（Multimodal RecSys）

> 利用图像、文本、视频、音频等多模态信息提升推荐质量。
> 核心挑战：**模态对齐** + **跨模态融合** + **模态噪声过滤**

---

## 多模态推荐的发展脉络

```
[早期] 单模态辅助   →   [中期] 多模态融合   →   [近期] 基础模型+推荐
 VBPR（图像）          MMGCN（图+文+音）        CLIP4Rec
 仅用视觉特征          GNN跨模态聚合            SiBraR（冷启动+多模态）
                       LATTICE/BM3              UniMMRec
```

---

## 方法列表

| 文件 | 方法 | 模态 | 年份 |
|------|------|------|------|
| [BM3.md](./BM3.md) | Bootstrap Multimodal Model | 图+文 | 2022 |
| [MGCN.md](./MGCN.md) | Modal-aware Graph Convolution | 图+文+音 | 2023 |
| [SiBraR.md](./SiBraR.md) | Single-Branch embedding for Rec | 多模态 | 2024 |
| [CLIP4Rec.md](./CLIP4Rec.md) | CLIP for Recommendation | 图+文 | 2024 |
| [UniMMRec.md](./UniMMRec.md) | Unified Multimodal Multi-scenario Rec | 多模态+多场景 | 2024 |
| [IISAN.md](./IISAN.md) | Item-side Semantic Aware Network | 视频多模态 | 2024 |

---

## 多模态推荐的三大痛点

1. **模态噪声**：图片和标题可能不一致（用户误传/商家虚假信息），直接融合引入噪声
2. **模态缺失**：某些物品只有图片，某些只有文本，模型需要鲁棒处理
3. **效率**：CLIP/BERT 等大模型推理慢，需要离线预提取特征

---

## 2024-2025 新方向

- **视频推荐多模态**：短视频推荐中视频帧特征+音频特征+文案的联合建模
- **CLIP 统一视觉语言**：用 CLIP 做图文对齐，大幅降低模态 gap
- **多模态 + 冷启动**：SiBraR 等工作打通多模态和冷启动，新物品上线即有高质量表示
