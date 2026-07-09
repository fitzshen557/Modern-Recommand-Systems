# 直播间进房转化率（CVR）预估

> 作者：懂哥 🐟 | 推荐算法方向 · 直播业务

---

## 一、背景与动机

### 1.1 什么是进房转化率？

在直播推荐场景中，用户从信息流（Feed）看到一个直播间的推荐卡片，到点击进入直播间，这个转化过程的成功率即为**进房转化率（Click-Through Rate to Room, CTR→Room CVR）**。它衡量的是"推荐系统把一个直播间展示给用户后，用户是否真的愿意点进去看"。

### 1.2 为什么直播进房CVR和传统电商CVR不一样？

传统的电商CVR预估（如淘宝、京东的商品详情页转化）面临的核心挑战是**样本稀疏**和**反馈延迟**——用户点击商品后可能过几小时甚至几天才下单。但直播场景的CVR预估有其独特的困难：

| 挑战 | 电商CVR | 直播进房CVR |
|------|---------|-------------|
| 时效性 | 小时~天级 | 秒~分钟级 |
| 直播状态 | 商品属性稳定 | 主播状态/内容实时变化 |
| 库存概念 | 有库存限制 | 无库存，但在线人数是实时信号 |
| 价格因素 | 明确价格 | 打赏成本不确定 |
| 内容新鲜度 | 商品可重复 | 直播内容不可逆 |

### 1.3 为什么传统CTR/CVR模型在直播场景不够用？

1. **静态特征主导**：传统模型以商品属性、用户画像为主，但直播间的状态（在线人数、互动率、主播情绪）在分钟级别剧烈变化。
2. **样本偏差严重**：直播推荐中，用户看到的直播间大多不会点进去（负样本极多），且热门直播间天然获得更多曝光，导致选择偏差。
3. **多目标耦合**：进房只是一个中间目标——平台真正关心的是用户进房后是否停留（留存）、是否互动（评论/点赞/打赏）。单纯优化进房率可能导致"标题党"直播间获得优势。
4. **冷启动问题突出**：新开播的直播间几乎没有历史数据，新用户对直播推荐也缺乏行为积累。

---

## 二、方法详解

### 2.1 直播进房CVR的整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    直播推荐系统架构                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────┐  │
│  │ 召回层   │ →  │  粗排层       │ →  │  精排层        │  │
│  │ (Match)  │    │ (Pre-Rank)   │    │ (Fine-Rank)   │  │
│  └──────────┘    └──────────────┘    └───────────────┘  │
│                                          │              │
│                                          ▼              │
│                                   ┌───────────────┐     │
│                                   │  进房CVR模型   │     │
│                                   │  (核心组件)    │     │
│                                   └───────────────┘     │
│                                          │              │
│                                          ▼              │
│                                   ┌───────────────┐     │
│                                   │  多目标融合    │     │
│                                   │  (进房+留存)   │     │
│                                   └───────────────┘     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

进房CVR模型通常位于精排层的核心位置，它的预估结果会和多个子目标的预估结果融合，形成最终的排序分数。

### 2.2 实时特征体系

直播进房CVR的核心差异在于**实时特征的引入**。以下是关键的实时特征分类：

```python
# 实时特征体系设计
realtime_features = {
    # === 直播间维度实时特征 ===
    "room_realtime": {
        "online_user_count": "当前在线人数（1min/5min/15min多时间窗）",
        "online_user_trend": "在线人数变化趋势（增长/下降/平稳）",
        "total_like_count": "累计点赞数",
        "like_rate_per_min": "每分钟点赞增量",
        "comment_rate_per_min": "每分钟评论增量",
        "gift_income_per_min": "每分钟打赏收入",
        "viewer_duration_avg": "当前观众平均观看时长",
        "viewer_enter_rate": "当前观众进入速率（人/分钟）",
        "viewer_leave_rate": "当前观众离开速率（人/分钟）",
        "hot_score": "综合热度分",
    },
    
    # === 主播维度实时特征 ===
    "anchor_realtime": {
        "current_stream_duration": "当前开播时长",
        "session_total_gift": "本场累计打赏",
        "session_avg_viewer": "本场平均在线人数",
        "interaction_intensity": "互动强度（评论+点赞频率）",
        "content_tag_realtime": "实时内容标签（由多模态模型给出）",
    },
    
    # === 用户-直播间交叉实时特征 ===
    "user_room_realtime": {
        "user_last_similar_room_duration": "用户最近一次看同类直播的时长",
        "user_today_enter_count": "用户今日已进房次数",
        "user_room_overlap_audience": "用户画像与当前观众画像的相似度",
        "user_recent_interaction_type": "用户最近的互动类型偏好",
    },
}
```

### 2.3 模型结构：多时间尺度特征融合

直播实时特征具有多时间尺度特性——有些特征反映的是"此刻"的状态（秒级），有些是"近期"趋势（分钟级），有些是"本场"累计（小时级）。如何有效融合这些不同时间尺度的特征是建模的关键。

```python
# 多时间尺度特征融合架构（概念代码）
class MultiScaleFeatureFusion:
    """
    多时间尺度特征融合模块
    
    输入：
    - user_static: 用户静态画像（离线计算）
    - room_static: 直播间静态属性（开播前可计算）
    - room_realtime_sec: 秒级实时特征（如当前在线人数）
    - room_realtime_min: 分钟级实时特征（如近5分钟评论率）
    - room_realtime_hour: 小时级累计特征（如本场总打赏）
    
    融合策略：
    1. 门控机制（Gating）：不同尺度的特征通过门控权重自适应融合
    2. 注意力机制（Attention）：用cross-attention让长期特征query短期特征
    3. 残差连接：短期变化以残差形式叠加到长期baseline上
    """
    
    def forward(self, user_static, room_static, feat_sec, feat_min, feat_hour):
        # Step 1: 分层编码
        h_sec = self.sec_encoder(feat_sec)      # 秒级：当前状态快照
        h_min = self.min_encoder(feat_min)      # 分钟级：近期趋势
        h_hour = self.hour_encoder(feat_hour)   # 小时级：本场累计
        h_user = self.user_encoder(user_static) # 用户画像
        h_room = self.room_encoder(room_static) # 房间属性
        
        # Step 2: 门控融合——让模型学习何时信任实时信号
        # 关键思想：当实时信号可信时（如开播较久），给实时特征更高权重
        # 当实时信号不可信时（如刚开播），更依赖静态特征
        gate = sigmoid(W_gate @ concat(h_sec, h_min, h_hour, h_room))
        h_dynamic = gate * h_sec + (1-gate) * h_room
        
        # Step 3: 用户-直播间交叉
        h_cross = cross_attention(query=h_user, key=h_dynamic, value=h_dynamic)
        
        # Step 4: 输出进房概率
        prob = sigmoid(MLP(concat(h_cross, h_user, h_dynamic)))
        return prob
```

### 2.4 多目标建模：进房 + 留存联合优化

单纯的进房CVR预估可能导致"诱导点击"——比如封面很诱人但内容空洞的直播间获得高进房率但用户秒退。因此需要联合建模进房和留存。

```python
# 多目标建模方案
class MultiTaskEnterStayModel:
    """
    多目标联合建模：进房 + 有效观看
    
    核心思路：
    - 进房CVR: P(enter | impression)
    - 有效观看: P(stay > threshold | enter)
    - 最终排序分: P(enter) * P(stay > threshold | enter)
    
    这样做的效果：
    即使用户很容易点进去（高进房率），但如果进去后留不住（低留存率），
    最终排序分也会很低。这就抑制了"标题党"直播间。
    """
    
    def forward(self, features):
        # 共享底层表达
        h_shared = self.shared_bottom(features)
        
        # Tower 1: 进房预估
        p_enter = sigmoid(self.tower_enter(h_shared))
        
        # Tower 2: 有效观看预估（条件概率）
        # 条件：给定用户进房的前提下，观看时长超过阈值（如30秒）的概率
        p_stay = sigmoid(self.tower_stay(h_shared))
        
        # Tower 3: 互动预估（可选）
        p_interact = sigmoid(self.tower_interact(h_shared))
        
        # 融合公式（类似ESMM的思路）
        # 最终分 = 进房概率 * (有效观看概率 + alpha * 互动概率)
        final_score = p_enter * (p_stay + self.alpha * p_interact)
        
        return final_score, p_enter, p_stay, p_interact
    
    def loss(self, labels):
        # 多任务loss：各任务独立计算binary cross entropy
        loss_enter = BCE(p_enter, labels.enter)
        loss_stay = BCE(p_stay, labels.stay_gt_threshold)
        loss_interact = BCE(p_interact, labels.interacted)
        
        # 加权求和，权重可调
        total_loss = loss_enter + self.w1 * loss_stay + self.w2 * loss_interact
        return total_loss
```

### 2.5 冷启动方案

新开播的直播间缺乏实时特征积累，需要特殊处理：

```python
# 冷启动处理策略
class ColdStartHandler:
    """
    新直播间冷启动方案
    
    策略1: 主播历史迁移
    - 用主播过去N场直播的数据构建prior
    - 开播初期用prior作为实时特征的替代
    - 随着本场数据积累，逐步切换到实时特征
    
    策略2: 同类直播间参考
    - 找到相似类型（内容标签、时段、主播等级）的直播间
    - 用它们在同一开播时段的统计特征作为参考
    
    策略3: 探索流量池
    - 为新直播间分配专门的探索流量
    - 用小流量快速收集反馈
    - 用Bandit算法（如UCB/Thompson Sampling）平衡探索与利用
    """
    
    def get_room_score(self, room, elapsed_minutes):
        if elapsed_minutes < 2:
            # 刚开播：主要依赖主播历史画像 + 内容理解
            score = self.prior_from_history(room.anchor_id)
            # 混入内容理解信号（封面/标题/开播标签）
            content_signal = self.content_model(room.cover, room.title)
            score = 0.6 * score + 0.4 * content_signal
            
        elif elapsed_minutes < 10:
            # 开播初期：实时特征开始积累但不够稳定
            realtime = self.get_realtime_features(room.id)
            prior = self.prior_from_history(room.anchor_id)
            # 用贝叶斯平滑：先验 + 似然 → 后验
            confidence = min(elapsed_minutes / 10.0, 1.0)
            score = (1 - confidence) * prior + confidence * realtime
            
        else:
            # 正常阶段：完全依赖实时特征
            score = self.get_realtime_score(room.id)
        
        return score
```

### 2.6 样本选择偏差校正

直播进房预估存在严重的样本选择偏差——只有被推荐曝光的直播间才有"是否进房"的标签，但曝光本身是由上一版模型决定的。

```python
# 偏差校正方法
class BiasCorrection:
    """
    常见的选择偏差校正方法：
    
    1. Inverse Propensity Scoring (IPS)
       - 对每个样本按其被曝光的概率加权
       - 曝光概率低的样本权重更高
       - 问题：方差可能很大，需要截断
    
    2. 两阶段训练
       - 第一阶段：训练一个曝光预估模型
       - 第二阶段：用预估的曝光概率做样本加权
    
    3. 数据增强
       - 随机给部分直播间额外曝光（类似A/B测试）
       - 用这些随机曝光的样本作为无偏数据
    
    工业界最常用的是方法2+3的组合
    """
    pass
```

---

## 三、实验结果

### 3.1 公开数据集 & 工业界报告的关键结论

由于直播进房CVR预估大多是各公司的内部工作，以下总结来自公开论文和工业分享的共识结论：

| 方法 | 进房CVR AUC | 有效观看率提升 | 关键改进点 |
|------|-------------|---------------|-----------|
| 纯静态特征 DNN | 0.72~0.75 | baseline | 仅用用户画像+主播属性 |
| + 实时特征 | 0.78~0.81 | +15~25% | 在线人数/互动率/变化趋势 |
| + 多时间尺度融合 | 0.80~0.83 | +20~30% | 秒/分/小时特征分层融合 |
| + 多目标联合 | 0.81~0.84 | +25~35% | 进房+留存+互动联合优化 |
| + 冷启动方案 | — | 新直播间曝光效率+40% | 历史迁移+内容理解+探索 |

### 3.2 关键数据洞察

- **实时特征的重要性**：引入实时特征后，进房CVR预估的AUC提升通常在5~8个点，这远高于传统电商场景中实时特征的收益（通常2~3个点）。原因在于直播内容的时效性极强。
- **在线人数的双刃剑效应**：在线人数是预测力最强的单特征之一（F1重要性通常排第一），但过度依赖会导致"马太效应"——大主播越来越强，小主播越来越弱。需要配合探索机制使用。
- **多目标优化的ROI**：联合建模进房+留存后，虽然进房率可能下降3~5%，但用户平均观看时长提升20%以上，平台整体DAU和打赏收入都有显著增长。

---

## 四、懂哥点评

### 4.1 工业价值 ⭐⭐⭐⭐⭐

直播进房CVR预估是直播推荐系统中**ROI最高**的模型优化方向之一。原因：

1. **直接影响核心指标**：进房率直接决定直播间的流量获取能力，是DAU和营收的基础。
2. **实时特征是差异化壁垒**：实时特征工程的能力（特征计算的速度、稳定性、覆盖度）是团队核心竞争力的体现。
3. **多目标框架可复用**：进房+留存的多目标框架可以扩展到更多子目标（打赏、关注、分享）。

### 4.2 局限性

1. **实时特征系统的复杂度高**：秒级特征更新需要完整的流计算基础设施（Flink/Kafka），运维成本不低。
2. **冷启动仍有瓶颈**：新主播的前几场直播，模型预估准确度仍然较低，需要产品侧的配合（如新主播流量扶持策略）。
3. **多目标权衡困难**：进房率和留存率有时是矛盾的，权重的设定往往需要大量实验和业务判断。

### 4.3 适用场景

- **最适用**：大型直播平台（如抖音、快手、B站）的直播推荐Feed流
- **部分适用**：中小平台的直播入口页（如果实时特征系统不够完善，可以先用准实时特征）
- **不太适用**：纯音频直播（缺乏视觉内容理解信号）、录播回放（没有实时性）

### 4.4 未来方向

1. **大模型赋能**：用LLM做多模态内容理解（封面+标题+直播片段），提供更强的冷启动信号。
2. **因果推断**：更系统地进行去偏，区分"用户因为这个直播间好而进房"和"因为推荐位好而进房"。
3. **序列决策**：将进房建模为RL问题——用户进一个直播间的决策受之前看了哪些直播间的影响。

---

## 五、参考链接

1. **ESMM（Entire Space Multi-Task Model）** - 多任务建模的经典框架
   - Ma et al., "Entire Space Multi-Task Model", SIGIR 2018
   
2. **直播推荐系统综述**
   - "A Survey on Live Streaming Recommendation", 2023

3. **实时特征工程**
   - "Real-time Attention based Look-alike Model for Recommendation System", KDD 2019

4. **多目标优化在推荐中的应用**
   - "Modeling Task Relationships in Multi-Task Learning", KDD 2018 (MMoE)
   - "Progressive Layered Extraction (PLE)", KDD 2020

5. **冷启动推荐**
   - "Meta-Rating: Towards Fast and Accurate Cold-Start Recommendation via Meta-Learning", 2022

6. **工业实践分享**
   - 快手直播推荐系统技术分享
   - 字节跳动直播推荐架构演进
