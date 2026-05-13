# 精排技术汇总

> 最后更新：2026-05

## 技术演进路线

```
LR/GBDT → Wide&Deep → DNN特征交叉 → 注意力机制 → Transformer序列 → 生成式精排
(2012-)    (2016)      (DeepFM/DCN)   (DIN/DIEN)    (BST/SIM/TWIN)   (GenRank/HSTU)
```

## 方法速查表

### 特征交叉类

| 方法 | 机构 | 年份 | 核心思想 | 链接 |
|------|------|------|----------|------|
| **Wide&Deep** | Google | 2016 | 记忆（Wide）+ 泛化（Deep），工业基线 | [论文](https://arxiv.org/abs/1606.07792) |
| **DeepFM** | 华为 | 2017 | FM替代Wide部分，自动二阶特征交叉 | [论文](https://arxiv.org/abs/1703.04247) |
| **xDeepFM** | 微软 | 2018 | CIN网络，显式高阶向量级交叉 | [论文](https://arxiv.org/abs/1803.05170) |
| **DCN-V2** | Google | 2021 | Cross Network V2，低秩分解提效 | [论文](https://arxiv.org/abs/2008.13535) |
| **FINAL** | 阿里 | 2023 | 因式分解特征交叉，极低延迟 | [论文](https://arxiv.org/abs/2304.00902) |
| **GDCN** | 多机构 | 2023 | 门控深度交叉，自适应特征选择 | [论文](https://arxiv.org/abs/2208.10091) |
| **DHEN** | Meta | 2022 | 异构交叉集成，Meta主力精排模型 | [论文](https://arxiv.org/abs/2203.11014) |

### 用户兴趣建模类

| 方法 | 机构 | 年份 | 序列长度 | 核心思想 | 链接 |
|------|------|------|----------|----------|------|
| **DIN** | 阿里 | 2018 | ~50 | Target Attention，差异化兴趣激活 | [论文](https://arxiv.org/abs/1706.06978) |
| **DIEN** | 阿里 | 2019 | ~50 | GRU兴趣演化，捕获兴趣动态变化 | [论文](https://arxiv.org/abs/1809.03672) |
| **BST** | 阿里 | 2019 | ~20 | Transformer建模行为序列 | [论文](https://arxiv.org/abs/1905.06874) |
| **SIM** | 阿里 | 2020 | 万级 | 两阶段检索：硬属性过滤+精细注意力 | [论文](https://arxiv.org/abs/2006.05639) |
| **ETA** | 阿里 | 2021 | 万级 | Hash LSH端到端长序列检索 | [论文](https://arxiv.org/abs/2108.04468) |
| **TWIN** | 快手 | 2023 | 千~万级 | 双子塔：物品侧预计算+精确注意力 | [论文](https://arxiv.org/abs/2302.02352) \| [详解](./TWIN.md) |
| **FEARec** | 中科大 | 2023 | ~200 | 频域FFT全局兴趣+时域局部兴趣融合 | [论文](https://arxiv.org/abs/2301.09780) |
| **HSTU** | Meta | 2024 | 万级 | 层次化序列Transducer，O(n)复杂度 | [论文](https://arxiv.org/abs/2402.17152) \| [详解](./HSTU.md) |

### 生成式精排类（2024-2025新方向）

| 方法 | 机构 | 年份 | 核心思想 | 链接 |
|------|------|------|----------|------|
| **GenRank** | 多机构 | 2025 | 物品-动作重组机制+时空偏置，生成式精排 | arXiv 2025 |
| **GRACE** | 微信 | 2024 | 生成式推荐与协同过滤对齐精排框架 | [论文](https://arxiv.org/abs/2408.xxxxx) |

## 长序列建模对比

| 方法 | 支持序列长度 | 复杂度 | 精度 | 适合场景 |
|------|------------|--------|------|---------|
| DIN截断 | ~200 | O(n) | 低 | 快速baseline |
| SIM | 万级 | O(k)，k<<n | 中 | 阿里系电商 |
| TWIN | 千~万级 | O(n+k·d) | 高 | 短视频/内容推荐 |
| HSTU | 万级 | O(n·w) | 极高 | 大规模工业系统 |
| Mamba4Rec | 万级 | O(n) | 高 | 超长序列效率优先 |

## 精排模型设计原则

1. **特征交叉**：DCN-V2/GDCN 作为特征交叉骨干，比纯MLP提升2~5%
2. **用户兴趣**：序列长度决定选型：<500用TWIN局部注意力，>1000考虑HSTU
3. **多任务**：PLE作为多任务基线，GNOLR/MetaBalance解决梯度冲突
4. **训练技巧**：
   - 负采样：热度降权（In-batch Negatives + Hard Negatives）
   - 特征归一化：Batch Norm对连续特征，Layer Norm对序列
   - 学习率：Embedding层和MLP层用不同学习率（Embedding用更小LR）

## 2024-2025 新趋势

1. **HSTU引领序列建模新范式**：Meta的万亿参数实践证明序列Transformer在工业落地可行
2. **生成式精排萌芽**：GenRank等工作开始探索自回归生成替代判别式打分
3. **精排-重排联合优化**：OPERA等工作打破两阶段边界
4. **LLM辅助精排**：用LLM生成的语义特征增强精排模型，不做在线推断

## 参考资源
- [DIN论文精读](https://arxiv.org/abs/1706.06978)
- [HSTU论文](https://arxiv.org/abs/2402.17152)
- [精排工业实践 - Doragd](https://github.com/Doragd/Algorithm-Practice-in-Industry)
