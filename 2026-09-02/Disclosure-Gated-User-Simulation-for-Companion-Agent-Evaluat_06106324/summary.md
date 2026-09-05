---
title: "Disclosure-Gated-User-Simulation-for-Companion-Agent-Evaluat"
source: https://arxiv.org/pdf/2609.00982v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 22:33:06"
field: "多轮对话评估与模拟用户"
keywords: ["user simulation", "companion agent evaluation", "disclosure gating", "LLM benchmark", "emotional companionship", "simulation fidelity"]
innovations: ["提出可执行的五阶门控状态机规范，实现信息释放对陪伴agent行为的条件依赖", "双分支训练语料设计，合成分支提供门控轨迹、真实分支提供语言保真", "建立排序保持与分数稳定双验收标准，揭示仅检查排名会遗漏分数尺度漂移"]
benchmarks: ["CompanionBench"]
---

# 论文速读：Disclosure-Gated-User-Simulation-for-Companion-Agent-Evaluat

## 一句话总结
本文针对LLM模拟用户"过度合作"问题，提出了一套**披露门控机制**（disclosure gate），使信息释放成为陪伴agent行为的函数；训练语料由真实对话分支（提供语言纹理）与合成分支（提供门控轨迹）构成，最终在CompanionBench基准上发布了122B规模的模拟用户（M1b/M3b），并给出排序保持与分数稳定两个评估环境验收标准。

## 研究问题与动机
- **过度合作失败模式**：现有LLM模拟用户会主动提前透露个人信息，导致被测系统仅靠"问更多问题"就能得高分，而无法区分"问得多"与"让用户愿意说"两种能力（Zhou et al. 2026; Chopra et al. 2026）。
- **已有机制不可复现**：尽管存在多种缓解方案（Yang et al. 2025; Wu et al. 2026a; Han et al. 2026; Sabour et al. 2026），但均缺乏足够精确的规范说明，第三方无法重建相同的门控机制。
- **缺乏可消融性验证**：现有工作多从prompt侧移除组件，极少将已训练进权重的门控作为可消融变量，并在噪声带内测量排名位移。
- **情感陪伴领域的空白**：上述证据主要来自任务导向或通用助手评估，情感陪伴领域的模拟用户评估缺乏专门设计。

## 核心贡献（创新点）
1. **可执行门控规范**：提出五阶门控状态机（opening < asked_or_natural < felt_heard < felt_safe < earned_deep_trust）及八类行为转移函数，填补了原基准仅用四百词描述的空白。
2. **双分支训练语料设计**：合成分支提供门控轨迹（反事实对比），真实分支提供语言分布保真度，二者结合实现门控行为与语言自然性的兼顾。
3. **门控内化与可剥离性验证**：证明门控已被训练进权重，推理时剥离门控标注仍能保持相近行为；而prompt调用的前沿模型剥离后行为急剧下降。
4. **双验收标准框架**：提出排序保持（rank correlation ≥ 0.95）与分数稳定（per-system rubric total无显著偏移）两个环境验收标准，揭示仅检查排名会掩盖分数尺度漂移的风险。
5. **多层审计证据链**：从内在层（门控审计、人类判别任务）到下游层（CompanionBench排名位移），形成互为补充的验证体系。

## 方法详解
### 披露清单与门控图例
- 每个persona携带一份**披露清单**（disclosure_inventory），每条包含content、depth（surface/mid/core）、gate、guard、may_never_surface五个字段。
- **门控图例**（gate_legend）由清单编译而来，仅包含该example实际用到的门控与保护机制定义，全局机制留给训练内化。

### 五阶门控与状态转移
- 状态为当前已获得的最高门控索引 level ∈ {0, 1, 2, 3, 4}，对应五阶门控。
- 五阶合并为三层可观察深度：level < 2 → surface；2 ≤ level < 4 → mid；level = 4 → core。
- 八类行为（ai_move）触发状态转移，关键规则包括：
  - 不可跳级（earned_deep_trust需先满足felt_safe）
  - neutral_noop（平淡回应）将level降至min(level, 1)
  - tripped_anti_goal与judged_or_lectured使level减1并设rupture标志
- fh_streak记录连续准确反射的轮数，用于审核felt_safe获得条件（≥2轮连续）。

### 非对称撤退与七种保护机制
- 触发撤退后：状态回退一级 + 当前暴露项加guard + 轻度退缩。
- 不对称性体现于：从felt_safe到core仅需一次earning动作，但一次评判行为即可回退至felt_heard（差两级）。
- 七种guard形式：minimized、rationalization、intellectualization、deflection、humor、withdrawal、null。

### 提示组装与可见性契约
- 训练时与推理时prompt必须**字节级一致**，仅director signal与anti_goal block位置互换。
- 陪伴agent不可见披露清单，仅获知age/gender/recap三个字段。
- 审计judge可见披露清单及深度标签，premature判定为合规性检查而非独立重建。

### 训练语料构建
- **真实分支**：约31,000条标注对话（中文生产日志），提供语言纹理与真实反应模式。
- **合成分支**：68个单元格×4种依恋类型×6种行为脚本（含三段弧：good→bad、bad→good、good→bad→repair），保证同一persona在门控打开与破裂两侧均有轨迹。
- 中英双语语料按60:40合成/真实比例，中文约714,000 trainable turns，英文约454,000 turns。

## 实验与结果
### 数据集与评估基线
- 使用**CompanionBench**（Liu et al. 2026）英语版12个SUTs、100个persona、7个评估环境（含参考环境、新seed重运行噪声带、5个候选模拟器）。
- 主judge为deepseek-v4-pro，次judge为claude-opus-4-8（交叉验证）。

### 核心结果
| 指标 | 参考模拟器 | M1b (122B) | gpt-5.1 (prompted) |
|---|---|---|---|
| Mid-layer disclosure contrast | +0.680 | +0.700 | +0.600 |
| Core-reach rate (trips) | 3% | — | 8% |
| Rank correlation ρ | 1.000 | 0.993 | 0.972 |
| 最大排名位移 | 0 | 0 | 2 |
| 评分偏移 | 0 | +0.032 | +0.521 |
| Gate readings偏差 | — | 13% | 46% |

- **唯一通过双标准**：仅M1b同时满足order-preserving (ρ=0.993) 与scale-stable (per-system scores无显著变化)。
- **gpt-5.1陷阱**：排名 barely 移动但所有SUT分数整体上升0.521分，仅查排名会遗漏此偏差。
- **合成分支关键性**：纯真实分支训练(M4) contrast仅+0.080，纯合成分支(M5)为+0.610，混合(M3)为+0.520。
- **门控内化**：推理时剥离门控信息(A2)，contrast变化仅−0.022（CI含0）；而gpt-5.3剥离后下降0.240（p<0.001）。
- **人类判别**：37.8%判别率显著低于50%，但68.3%偏好M3b比gpt-5.1更像真人（p<0.0001）。

## 相关工作脉络
1. **Zhou et al. (2026)**：大规模User-Sim Index研究显示模拟用户与真人差距（76.0 vs 92.7），指出"easy mode"问题，但未提供情感陪伴领域解决方案。
2. **Chopra et al. (2026)**：提出生成现实用户persona的方法，属行为分布层干预，未触及机制层门控设计。
3. **Sabour et al. (2026) / PatientAct**：信任机制双向非对称变化且每轮重计算，本文的门控在训练后固定于权重，re-locking触发机制不同。
4. **Chen et al. (2026) / Adaptive Virtual Patient**：基于近2000小时治疗转录，声明"分数不衰减"，但无行为触发的撤退机制，本文强调asymmetric retreat。
5. **Naous et al. (2026) / Flipping the Dialogue**：从weights侧传递机制的先例，本文在此基础上进行可消融性审计。
6. **Balog & Zhai (2024)**：user simulation作为评估仪器的传统，本文延续state-based information gating路线但引入情感状态与 rupture-repair 动态。

## 局限性与未来方向
- **工具分辨率限制**：judge可见深度标签，premature判定为合规性检查；turn-by-turn行为标注Kappa仅0.427，绝对值跨批次不可比。
- **门控为规范而非描述模型**：未体现去抑制效应（disinhibition effect）；real branch的disposition-divergence标记从未带入训练样本，down-weighting机制失效。
- **证据覆盖不均衡**：下游证据仅英语，无M3b排序保持的直接证据；人类锚定仅中文5名内部评测员。
- **训练止步于SFT**：无偏好优化阶段，导致单条消息长度坍缩（53.5% ≤8字符，vs 真人27.1%）。
- **浅层门控规则缺陷**：在asked_or_natural及以下，平淡回应无代价而触发anti-goal仍降一级，与设计意图相反。

## 研究启发与可借鉴点
- **双验收标准框架**：ranking preservation与score scale stability分离评估，揭示仅看排名会遗漏系统性偏差，适用于任何模拟器替换场景。
- **双分支语料架构**：合成数据负责机制轨迹覆盖，真实数据负责分布保真，60:40比例可在门控强度与语言自然性间取得平衡。
- **门控可内化性证明**：剥离门控标注后行为仍在，说明机制可训练进权重而非依赖运行时提示，降低部署复杂度。
- **领域泛化模板**：附录G给出的四组件模板（有序状态梯、转移谓词、非对称撤退、示例级门控图例）可迁移至面试、医疗咨询、谈判等场景。
- **审计-生成一致性设计**：director信号仅合成时使用，训练/推理prompt字节级一致，确保行为可比性。

## 关键术语表
- **Disclosure Gate（披露门控）**：控制信息释放的行为条件机制，五阶门控对应三层次深度。
- **Mid-layer Disclosure Contrast（中层披露对比）**：primary endpoint，衡量good与trips条件下披露深度差异。
- **Asymmetric Retreat（非对称撤退）**：破裂后状态回退一级+加guard，重建成本高于获取成本。
- **Gate Legend（门控图例）**：由披露清单编译的示例级定义表，仅含实际用到的gate/guard语义。
- **Order-preserving（排序保持）**：环境替换后SUT排名相关系数≥0.95的验收标准。
- **Scale-stable（分数稳定）**：环境替换后各SUT绝对评分无显著变化的验收标准。
- **Synthetic Branch（合成分支）**：提供门控轨迹的反事实对话数据，占训练语料60%。
- **Director Signal（导演信号）**：合成时私密状态提示，训练后移除，不作为机制组成部分。

## 可复现要素
- **数据集**：CompanionBench英语版；中文真实对话为去标识化授权生产日志（内部可见，不公开）；合成轨迹904 seeds + 2,433 trajectories已开源。
- **代码/权重**：M1b (EN, 122B) 与 M3b (ZH, 122B) 已发布；完整规范、审计工具链、judge prompt、人类研究材料见 github.com/liuyaox/CompanionBench。
- **关键超参**：训练2 epoch (ZH) / 3 epoch (EN)；temperature 0.7 rollout；发布默认T=1.0 (ZH) / T=0.7 (EN)。
