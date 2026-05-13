# 多模态推荐技术汇总

> 最后更新：2026-05

## 多模态推荐的价值

- **破解ID冷启动**：新物品无交互数据，但有图片/文本/视频，内容特征即刻可用
- **增强长尾物品表示**：长尾物品交互稀疏，多模态补充语义信息
- **跨域迁移**：模态特征在不同领域间可迁移，协同ID不行

## 模态分类

| 模态 | 典型应用场景 | 提取方式 |
|------|------------|---------|
| 图像 | 时装、美食、家居、商品 | ResNet/ViT/CLIP |
| 文本 | 新闻、书籍、评论、商品描述 | BERT/Sentence-T5/LLM |
| 视频帧 | 短视频、直播截帧 | 视频Transformer/TimeSFormer |
| 音频 | 音乐、播客、视频BGM | CNN-based/Wav2Vec |
| 视频+音频+字幕 | 短视频综合 | 多流融合 |

## 方法速查表

### 早期经典

| 方法 | 机构 | 年份 | 核心思想 | 链接 |
|------|------|------|----------|------|
| **VBPR** | UCSD | 2016 | CNN图像特征+MF，视觉推荐先驱 | [论文](https://arxiv.org/abs/1510.01784) |
| **MMGCN** | 多机构 | 2019 | 多模态图卷积，各模态构建独立交互图 | [论文](https://arxiv.org/abs/1904.12575) |
| **LATTICE** | 多机构 | 2021 | 学习多模态物品关系图+GCN协同过滤 | [论文](https://arxiv.org/abs/2104.11031) |

### 对比学习时代（2022-2023）

| 方法 | 机构 | 年份 | 核心思想 | 链接 |
|------|------|------|----------|------|
| **BM3** | 多机构 | 2022 | 多模态对比学习，Dropout增强去噪 | [论文](https://arxiv.org/abs/2207.05969) |
| **SLMRec** | 多机构 | 2023 | 轻量自监督多模态推荐 | [论文](https://arxiv.org/abs/2304.13325) |
| **MGCN** | 多机构 | 2023 | 模态感知图卷积，显式建模模态差异 | [论文](https://arxiv.org/abs/2308.03588) |
| **VQ-Rec** | 多机构 | 2023 | 向量量化压缩跨模态表征 | [论文](https://arxiv.org/abs/2210.12316) |

### 基础模型融合时代（2024-2025）

| 方法 | 机构 | 年份 | 核心思想 | 链接 |
|------|------|------|----------|------|
| **SiBraR** | RecSys 2024 | 2024 | 单分支embedding网络，冷启+多模态统一 | [论文](https://dl.acm.org/doi/10.1145/3640457) |
| **CLIP4Rec** | 多机构 | 2024 | CLIP对齐图文表征直接用于推荐 | [论文](https://arxiv.org/abs/2312.xxxxx) |
| **UniMMRec** | 多机构 | 2024 | 统一多模态多场景推荐框架 | [论文](https://arxiv.org/abs/2405.xxxxx) |
| **IISAN** | SIGIR 2024 | 2024 | 物品侧语义感知网络，短视频多模态精排 | [论文](https://dl.acm.org/doi/10.1145/3626772) |

## 多模态融合策略

### 早期融合（Early Fusion）
在特征层直接拼接各模态：
$$e_{\text{item}} = \text{MLP}([e_{\text{img}}, e_{\text{text}}, e_{\text{audio}}])$$
- 优点：简单
- 缺点：各模态特征尺度和分布不同，噪声相互污染

### 晚期融合（Late Fusion）
各模态独立打分，最后加权融合：
$$\text{score} = \alpha_{\text{img}} \cdot s_{\text{img}} + \alpha_{\text{text}} \cdot s_{\text{text}} + \alpha_{\text{CF}} \cdot s_{\text{CF}}$$
- 优点：各模态独立优化，互不干扰
- 缺点：无法捕获模态间的交互

### 注意力融合（Attention-based Fusion）
用注意力机制自适应加权各模态：
$$e_{\text{item}} = \text{Attention}(q_{\text{user}}, [e_{\text{img}}, e_{\text{text}}, e_{\text{CF}}])$$
- 优点：用户偏好自适应决定模态权重（对图片感兴趣的用户vs对文字敏感的用户）
- 目前工业主流方向

### 对比学习对齐（Contrastive Alignment）
- 跨模态对比：让图片embedding和文本embedding对齐（CLIP思路）
- 模态-协同过滤对齐：让内容embedding和行为ID embedding对齐（RLMRec思路）

## 多模态推荐的工程实践

1. **特征预提取**：大模型（CLIP/BERT）推理代价高，离线预计算，存向量库
2. **特征维度**：一般投影到 128~256 维，与CF embedding保持一致
3. **更新策略**：物品更新内容（如修改封面），需要重新提取特征
4. **冷启动衔接**：新物品上线时直接用多模态embedding，交互积累后再融合CF信号

## 2024-2025 新趋势

1. **CLIP统治视觉语言对齐**：CLIP4Rec证明CLIP特征直接用于推荐效果远超CNN特征
2. **短视频多模态增强精排**：IISAN等工作在精排阶段引入视频帧+音频+字幕联合建模
3. **多模态冷启统一框架**：SiBraR打通多模态和冷启动，单一框架解决两个问题
4. **生成式多模态**：用视频生成模型为内容生成多视角特征，扩充物品表示

## 参考资源
- [Multimodal RecSys Survey](https://arxiv.org/abs/2302.03883)
- [RecSys 2024 Cold Start Session（含SiBraR）](https://recsys.acm.org/recsys24/session-7/)
- [多模态推荐实践 - Doragd](https://github.com/Doragd/Algorithm-Practice-in-Industry)
