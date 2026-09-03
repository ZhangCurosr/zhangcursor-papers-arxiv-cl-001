---
title: "Confident-at-the-moment-of-action-belief-miscalibration-in-L"
source: https://arxiv.org/pdf/2608.24691v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 05:25:38"
field: "LLM可信评估与校准"
keywords: ["置信度校准", "隐藏信息", "LLM评估", "行动-信念gap", "Regent Chess", "agentic系统"]
innovations: ["构建可同时 elicite 信念声明与独立行动并对照可恢复真值的游戏化评估设置", "揭示高置信度吃子命中率仅1.6%的行动时刻严重失准现象", "证明传统评估指标与信念质量完全脱钩的 within-model 对照证据"]
benchmarks: ["Kaggle Game Arena Chess", "ConfidenceBench"]
---

# 论文速读：Confident-at-the-moment-of-action-belief-miscalibration-in-L

## 一句话总结
论文在一种名为 Regent Chess 的国际象棋变体中，测试了 LLM 在隐藏信息场景下"行动时置信度校准"问题，发现高置信度（≥0.5）声明的吃子行动仅 1.6%（62 次中 1 次）正确；且传统评估指标（合法性、成本、延迟等）与信念质量完全脱钩，表现最优的配置反而信念质量最差。

## 研究问题与动机
1. **Agentic 系统的假设漏洞**：部署中行动决策日益依赖模型自身声明的置信度（如置信度低于阈值才升级给人），但该假设"行动时刻的置信度跟踪正确性"尚未在隐藏信息场景中实证检验。
2. **现有评估无法分离信念与行动**：QA 基准将信念和答案合并为同一输出通道；棋类基准仅有行动（落子）却无隐藏状态的信念声明，无法评估"模型说一套做另一套"的 gap。
3. **缺失的关键评估条件**：需要一个同时满足三个条件的设置——(a) 可独立声明对不可观测事实的信念、(b) 与行动在同一时刻分别 elicited、(c) 隐藏状态的真值可在游戏结束后精确恢复。
4. **已有证据提示问题普遍**：ConfidenceBench [10] 发现顶级模型中置信度最准的不等于校准最好；RLHF 训练可能系统性推高模型的口头置信度而与实际质量无关 [9]。

## 核心贡献（创新点）
1. **首次在行动时刻测量信念校准**：构建 Regent Chess 设置，在每个合法棋步同步 elicite 移动选择与对手隐藏王位概率分布，并在吃子时刻对照可恢复的真值评分，填补了"信念—行动一致性"在动态交互中的评估空白。
2. **揭示传统指标与信念质量的完全脱钩**：在同一模型内比较发现，禁用推理模式后的配置在所有常规轴（100% 合法走法、0% 非法尝试、截断率最低、成本降 28×、速度升 25×）均最优，但信念质量测试中反而最差。
3. **证明"赢家"可在信念完全错误时获胜**：构建了模型胜出的真实案例——模型对第三格赋予零概率，却因普通战术原因捕获该格并获胜，说明纯结果评估无法捕捉此类失败。
4. **提出隐藏状态内源性设计**：与既有 benchmark（ dealt hand / assigned role 等外生隐藏状态）不同，Crown Shift 机制让玩家自己创建并可反复修改隐藏事实，使"行动—信念 gap"具备可表达的决策基础。
5. **建立 oracle-vs-public 可解性边界**：证明对手策略的可读性是关键——针对固定启发式对手，一旦知道已发生 shift，shift target 可从公开位置特征预测（83.1% top-3 准确率），但 shift timing 近乎随机，导致任何预测器（含完整内部状态 oracle）均无法可靠超越均匀先验。

## 方法详解
**游戏设置（Regent Chess）**：
- 王位（Royal Status）可秘密转移：双方从第 4 回合起可执行 Crown Shift（将王位秘密转移到任意己方棋子），每次使用后需间隔 15 回合方可再次使用；转移不消耗回合，必须额外再走一步。
- 胜利条件：捕获对手当前持有王位的棋子（Regent），而非固定 king。
- 内部记录每步 Crown Shift，游戏结束后可回放恢复真值，确保信念可对照 ground truth 评分。

**信念 elicitation**：
- 每回合模型输入棋盘状态 + 己方历史 + 己方颜色，**不可见对手**；输出结构化响应：一个合法走法 + 对手所有存活棋子中谁是 Regent 的概率分布（top-k 列表，含残余质量）。
- 提示中明确指定双方颜色及对应方格约束。

**评分方法**：
- **capture-only scoring**：在模型实际吃子的棋步，将声明概率与该格是否含真实 Regent 配对。
- **every-ply scoring**：对所有合法棋步（board-ply ≥ 8，排除规则决定的 pre-shift 窗口）的每个候选格进行评分。
- **基线**：population-matched uniform prior（每步按对手存活棋子数重新计算 1/n），使用 Brier score + paired bootstrap 95% CI。
- 预注册四个假设及 falsifier，结果逐条对照核查。

**Seats 配置**：
- 5 个 seat-configurations 跨越两实验室（Google 四席，OpenAI 一席）；仅 S1（gemini-3.1-flash-lite）接受完整校准电池（capture-only + every-ply + 多局信念质量）。
- 表 1 报告各 seat 的 chess-text leaderboard 分数、temperature、max_tokens 及校准样本量。

## 实验与结果
**核心发现（S1，两批独立实验）**：
- 高置信度（≥0.5）吃子正确率：**原批次 0/22，复验批次 1/40，合并 1/62（1.6%）**
- Brier score：S1 = 0.1445 vs. uniform prior = 0.0203，比率 CI [3.8×, 15.9×]
- **99.3%**（复验 98.7%）的总 Brier gap 集中于高置信度事件，其余低置信度事件几乎不贡献失准。
- 可靠性图显示 S1 在整个置信度区间命中率接近零。

**跨五席排序（§4.3）**：
| Seat | 模型 | 高置信吃子命中率 |
|------|------|------------------|
| S1 | gemini-3.1-flash-lite | 1.6% |
| S7 | gpt-5-mini | 14.3% |
| S6-base | gemini-3-flash (4k) | 29.2% |
| S6B | gemini-3-flash (16k) | 40.0% |
| S5 | gemini-3.1-pro-preview | 50.0% |

- 点估计单调排序，跨两实验室；但仅 S1 vs S5 达到统计显著，其余相邻对不可区分。
- 关键反例：S6 与 S6B 为同模型同 leaderboard 分（1258.32），仅 max_tokens 不同（4096 vs 16384），命中率差 10.8 个百分点，与跨模型 635.7 分差距相当。

**第二支柱（§4.4）**：DeepSeek V4 Flash 关闭 extended reasoning 后，所有常规指标最优（100% 合法、0% 截断、28× 更便宜、25× 更快），但 well-formed belief rate 仅 3.8%，为五席最差。

**深度衰减（§4.5）**：S1 有效回应率从开局 85.6% 降至中期 47.8% 再降至后期 41.0%（n=39 欠功）；非法走法率从 1.2% → 4.5% → 20.0%（17 倍增长）。对照实验（关闭 Crown Shift 标准国际象棋）同样下降 20.1 个百分点，说明衰减非隐藏状态追踪独有，但 Regent Chess 额外有 17.7 个百分点的 hidden-state-specific residual。

**Coverage gap（§4.6）**：S1 吃子目标格被列入候选的概率仅 26.4%（24/91），与 permutation null（29.2%）无显著差异；**73.6% 吃子目标从未出现在声明候选中**。

**静态测试高估（§4.7）**：S1 孤立位置测试 89.3% well-formed vs 真实游戏 60.9%（28.4 点差距）；S2 差距更大（53.6% vs 12.4%，41.2 点）。

## 相关工作脉络
1. **Kaggle Game Arena（[1][2][3][4]）**：以棋类/Game 评估 LLM 竞赛能力，使用 Elo/BT 评分，指标均为 outcome 型（胜负/筹码/投票），不涉及隐藏状态信念 elicitation——本文设置在其评测层之上叠加信念质量评估。
2. **ConfidenceBench [10]**：15 模型 verbalized confidence 校准 benchmark，发现 S1 为最差校准模型；本文与其独立印证但聚焦"行动时刻"而非通用 QA 校准。
3. **Agent-BRACE [11]**：将 agent 拆分为 belief-state model + policy model，但以 task reward 优化而非 action 一致性校准；本文反向测量未被设计的 gap。
4. **BayesBench [12]**：追踪多轮证据积累下的信念更新，对比精确贝叶斯后验；本文对比对象为可恢复真值且在 action-selected locus 实时评分。
5. **Action-belief gap 工作 [13]**：发现 LLM 在不同 settings 下行动与声明置信度矛盾；本文差异在于测量"是否校准到频率 p"而非"是否 coherent"，且 belief elicited 在同一交互流内。
6. **WOLF [14] / MafiaScope [15]**：social deduction 游戏中的欺骗与信念探测，但隐藏状态为 setup 时分配的外生角色，不可玩家自我创建/修改；本文 Crown Shift 机制反转此设定。

## 局限性与未来方向
1. **单一 seat 完整电池**：仅 S1 有 capture-only、every-ply、coverage、depth-degradation 全部测量；其余四席仅有高置信命中率，广度受限。
2. **固定对手**：所有结果针对单一确定性启发式对手（Tier-1），面对 position-responsive 或更强对手时是否成立未验证。
3. **无 transfer validation**：未建立此 instrument 分数对真实部署/下游任务表现的预测力。
4. **采样不可复现**：S1 provider 拒绝固定随机种子，游戏不可 byte-for-byte 重现，仅可 re-run under same configuration。
5. **截断率未控**：各 seat 截断率差异大（近零至 80%+），影响 seat 间公平比较。
6. **token budget 引入评估偏差**：固定 max_tokens 惩罚需要更长 deliberation 的配置，可能系统性低估强模型上限。
7. **未来方向**：（a）测试 position-responsive 对手；（b）多 seat 完整电池复现；（c）探索 elicitation architecture（joint vs split）对测量噪声的影响。

## 研究启发与可借鉴点
1. **信念—行动分离评估范式**：可迁移至任何需要"声明置信度 + 独立行动"的 agentic 场景（工具调用、金融决策、医疗建议），构建"声明—执行—真值恢复"三元评估 pipeline。
2. **传统指标与目标能力脱钩检测**：本文证明了禁用推理模式配置在常规轴全优但目标能力最差，提示任何 benchmark 设计需警惕"表面最优 = 实际失败"的 selection bias。
3. **oracle-vs-public 可解性边界测试**：面对"模型是否失败"的指控，先用 oracle 对比 public predictor 建立 computable ceiling，避免将不可解任务归因为模型缺陷——此方法可复用至其他隐含信息 benchmark。
4. **静态 screening 对动态性能的夸大效应**：孤立位置测试（89.3% well-formed）与真实游戏（60.9%）差距 28.4 点，提示前端部署检查不能替代交互内评估。
5. **预注册 + 版本控制分析代码**：四项 pre-registered expectations 逐条对照 falsifier 核查， headline figures 由 version-controlled code 重算，可降低 post-hoc cherry-picking 风险，值得作为 robust evaluation 实践推广。

## 关键术语表
**Regent Chess**：论文设计的国际象棋变体，王位可秘密转移，胜利条件为捕获对手当前 Regent 而非固定 king。
**Crown Shift**：隐藏信息机制，玩家可秘密将王位移至任意己方棋子，不消耗回合，每 15 回合可再次使用。
**Capture-time calibration**：在模型实际执行吃子行动的棋步，对照该格是否为真实 Regent 评估声明置信度的校准程度。
**Well-formed rate**：模型响应成功解析为合法走法 + 有效概率分布的比例，衡量 elicitation 格式遵循度。
**Population-matched uniform prior**：每步按对手存活棋子数动态计算的均匀分布（1/n），作为无信息基线。
**Action-belief gap**：模型声明的信念与实际行动之间不一致的模式，本文将其特化为"高置信吃子命中率极低"。
**Oracle-vs-public tractability**：通过比较拥有完整内部状态的 oracle 与仅可见公开信息的 predictor，判断任务本身是否可解。
**Belief–action correspondence**：信念声明与后续行动在动作选择位点的一致性关系，是本文核心测量对象。

## 可复现要素
- **数据集**：游戏记录来自 gemini-3.1-flash-lite、gemini-3.1-pro-preview、gemini-3-flash、gpt-5-mini 多 seat，对固定启发式对手的真实对局日志。
- **代码开源**：论文声明分析代码为 version-controlled，游戏引擎有 test suite 覆盖各规则条款；但论文未明确提供 GitHub 链接或代码仓库 URL。
- **权重开源**：未提及；使用 API 调用的商业模型（Google/OpenAI）。
- **关键超参**：temperature=0.7（S7 受 API 强制为 1.0）；max_tokens=4096（S6B 为 16384）；top-k 信念声明格式；pre-shift 窗口（board-ply<8）排除。
- **随机种子**：S1 provider 拒绝固定种子，复验采用相同配置 re-run 设计而非 seed 复现。
