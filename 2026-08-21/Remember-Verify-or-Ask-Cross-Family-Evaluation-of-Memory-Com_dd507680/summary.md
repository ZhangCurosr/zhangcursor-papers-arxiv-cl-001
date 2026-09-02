---
title: "Remember-Verify-or-Ask-Cross-Family-Evaluation-of-Memory-Com"
source: https://arxiv.org/pdf/2608.19564v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:45:51"
field: "LLM Agent 记忆与交互决策"
keywords: ["LLM agents", "long-term memory", "memory commitment", "clarification benchmark", "tool-call evaluation", "cross-family evaluation", "over-memory", "under-asking"]
innovations: ["提出 MCB 基准，首次联合评估持久化/局部使用/验证/澄清四种记忆承诺动作", "设计 MCB-Act 工具调用变体，量化标签-工具行为分歧", "跨 Claude 与 Qwen 家族的配对 Holm 校正检验，分离 prompt 敏感性与模型特质"]
benchmarks: ["MCB", "MCB-Act", "LongMemEval", "LoCoMo", "PerMemBench", "Memory-R1", "CLAMBER", "τ-bench"]
---

# 论文速读：Remember, Verify, or Ask? Cross-Family Evaluation of Memory Commitment in LLM Agents

## 一句话总结
论文提出了 **Memory-Clarification Boundary（MCB）基准**，系统评估 LLM Agent 在面对交互-derived 信息时，应将其持久化存储、仅限本次使用、验证现实状态还是向用户澄清——这是 Agent 记忆承诺能力的核心决策边界。跨 Claude 和 Qwen 两家模型、两种评测模式（标签选择 vs 工具调用）的配对实验表明：**所有模型在"询问用户"上显著不足，且口头决策与工具调用行为存在巨大偏离**。

## 研究问题与动机
1. **持久记忆的静默破坏风险**：LLM Agent 的长期记忆可以个性化，但一条错误写入的持久化信息会无声地扭曲后续所有行为——问题的本质不是"记忆好不好"，而是"该不该记住"。
2. **"验证"与"澄清"被混淆**：世界是可变事实的来源，用户是意图和范围的权威；现有评测无法区分模型是在查世界还是在问用户。
3. **标签 ≠ 行为**：静态 action label 的选择未必能预测 Agent 实际发出的工具调用，但现有 benchmark 几乎都只测标签。
4. **单一模型的发现无法泛化**：不同模型家族对提示干预的响应模式不同，需要跨家族对照才能区分"模型特质"和"可干预行为"。

## 核心贡献（创新点）
1. **发布 MCB 基准**：140 个主场景（70 开发 + 70  Held-out）+ 70 个对比验证项，含非作者独立标注（κ = 0.962）和反 shortcut 陷阱——与已有工作相比，首次将"持久化/局部使用/验证/澄清"四选一并提供审计链。
2. **跨家族配对实验设计**：在同批 Held-out 项上同时跑 Claude（Haiku 4.5、Sonnet 4.6）和 Qwen3.5-9B，共享项允许精确配对检验，而非依赖重叠置信区间——比 PerMemBench、LongMemEval 等单模型评测更具因果识别力。
3. **引入 MCB-Act 工具调用变体**：去掉标签词汇表，强制模型输出 `memory_write` / `use_now` / `check_source` / `ask_user` 四类结构化调用——揭示了"标签-工具分歧"这一此前未被系统测量的维度。
4. **行为特异性指标体系**：除了总准确率，首次将 **Over-memory（错误持久化率）** 和 **Clarification/Verification Recall** 作为独立指标并做 Holm 校正配对检验——证明政策提示可以在总准确率不显著变化时大幅降低安全相关错误。
5. **完整可复现产物**：数据、盲标注、提示词、Runner、item-level 预测、模型元数据、测试脚本全部公开——数值结论可由确定性代码 + 存储预测完全复现。

## 方法详解
**任务定义**：每条样本包含（1）采集上下文 acquire context、（2）候选更新 candidate update、（3）复用上下文 reuse context。模型需从中选出一个金标 action。

**四个金标动作及判定规则**：
- **Persist**：明确持久的偏好、授权策略或稳定事实；
- **Ephemeral**：限定于单个工件/会话/时间窗口的信息；
- **Verify**：世界状态可变或单一噪声信号，必须重新检查；
- **Clarify**：指代/持久性/范围/冲突未决，或二手偏好——**用户是唯一权威来源**。

**偏置原则（weaker-action tie-breaker）**：当 persist 与一个较弱 action 平局时，规则强制选择较弱的那个。编码了不对称代价：不必要的提问可见且可恢复，而错误的持久化会沉默地残留。

**三种提示干预**：
- **Bare**：仅定义四个 action；
- **Policy**：追加五条承诺规则（含偏置原则）；
- **Few-shot**：每种 action 加一个开发集示例（共 4 个）。

**MCB-Act 工具调用模式**：去掉标签词汇，强制模型输出一个结构化 JSON 调用，参数为字符串，由确定性最小内容/相关性规则校验。评分时将工具映射回 action 类，保留 payload 供审计。仅使用 bare 条件。

**统计方法**：2000 次重采样 percentile bootstrap 95% CI； headline 差异用配对 McNemar 检验（精确双侧）；行为率用配对 item-level 检验；Holm 校正控制族错误率。

## 实验与结果
**数据集**：70 Held-out 主测试项（非作者双盲标注，κ = 0.962）+ 70 对比验证项（35 证据翻转对，含 7 个 cue-conflicting 陷阱）。金标分布：38 persist、40 ephemeral、33 verify、29 clarify。

**模型与设置**：Claude Haiku 4.5、Claude Sonnet 4.6（API 逐 item 调用）、Qwen3.5-9B（Ollama Q4 K_M，temperature=0，seed=13，thinking 禁用，label 模式 96 token / act 模式 128 token）。每模型每条件 70 次独立调用。

**关键结果（Held-out, n=70）**：

| 模型 | 条件 | 准确率 | Macro-F1 | OM↓ | Clar↑ | Ver↑ |
|---|---|---|---|---|---|---|
| Claude Haiku 4.5 | bare | 0.629 | 0.615 | 0.029 | 0.500 | 0.944 |
| Claude Haiku 4.5 | policy | **0.857** | 0.846 | 0.014 | 0.750 | 0.944 |
| Claude Sonnet 4.6 | bare | 0.814 | 0.790 | 0.057 | 0.500 | 0.889 |
| Claude Sonnet 4.6 | policy | 0.843 | 0.831 | 0.014 | 0.667 | **1.000** |
| Qwen3.5-9B | bare | 0.557 | 0.450 | **0.243** | **0.000** | 0.667 |
| Qwen3.5-9B | policy | 0.629 | 0.542 | 0.100 | 0.083 | 0.944 |
| Qwen3.5-9B | few-shot | **0.771** | 0.726 | 0.129 | 0.333 | 0.833 |

**最强结果**：Haiku policy = 0.857 准确率，p_H = 0.002；Qwen few-shot Δ = +0.214，p_H = 0.002。

**核心发现**：
1. **跨家族"少问"（under-asking）不对称**：两家族均更善于验证世界状态，但对用户澄清显著不足。Bare Qwen：verify 12/18，clarify **0/12**——澄清用例被映射到 persist（7）、verify（4）、ephemeral（1）。
2. **政策提示减错误持久化但不提总准确率**：Qwen policy OM 从 0.243 降至 0.100（p_H = 0.038），但总准确率仅 +0.071（p_H = 0.539，不显著）。
3. **标签-工具严重偏离**：Haiku 57%、Sonnet 57%、Qwen **仅 23%** 一致。Qwen act 准确率 0.343，较 bare 下降 0.214（p_H = 0.047）；Sonnet 下降 0.286（p_H < 0.001）。瓶颈在工具选择，不在参数格式。
4. **Contrast 验证扩展**：Qwen 在 140 项上，bare/policy/few-shot 准确率 = 0.614/0.757/0.843；澄清召回仍是弱项（0.074/0.407/0.519）。

## 相关工作脉络
1. **LongMemEval / LoCoMo / MemoryBank / MemBench**：以检索/复用为核心评分目标，不测存储门控、不问用户、不查世界——MCB 在此基础上补全后三个维度。
2. **PerMemBench [9]**：学习二值 session-level 存储门，但将"世界不确定"与"意图不确定"混为一谈；MCB 将它们拆成 verify vs clarify 两个独立 class。
3. **Memory-R1 [10]**：学习 ADD/UPDATE/DELETE/NOOP 操作，但缺少"查世界"和"问用户"两类外部接口——MCB 的 MCB-Act 补足这一缺口。
4. **CLAMBER [12]**：关注歧义识别与澄清，但未引入"世界作为真值源"的对照；MCB 的 verify 项弥补这一点。
5. **τ-bench / τ²-bench [13, 14]**：揭示静态答案与工具化 Agent 行为可能偏离；MCB-Act 首次在"记忆承诺边界"这一具体场景复现并量化该现象。
6. **ask-when-uncertain 系列 [15, 16]**：用不确定性或 EV 决定是否提问——MCB 用 source-of-truth 分类法（世界源 vs 用户源）提供了一套更细粒度的评估框架。

## 局限性与未来方向
1. **样本量小**：70 项置信区间宽，每 category 仅 10 项，不足以做稳健的类别级排名。
2. **合成英文场景**：无真实对话、无中文、无个人记录；自然语言分布偏差未知。
3. **MCB-Act 只记录不调用**：没有模拟用户回答、没有真实世界查询执行、没有下游任务分数——"标签-工具差距"已证实，但端到端效用增益未测。
4. **开发集演示保留作者标签**：few-shot 示例可能含有轻微 author bias。
5. **Qwen 仅一个量化 checkpoint**：Q4 K_M 量化 + 禁用 thinking + Ollama 栈本身即属于被评测系统的一部分，不可泛化到原始模型。
6. **政策提示对 Sonnet 无效**：同一政策在不同 family 内效果不一致，说明 prompt 敏感性是模型特异的。

## 研究启发与可借鉴点
1. **配对检验 + 行为率单独分析**：headline accuracy 掩盖了安全相关行为的改善——论文证明必须在 item-level 上报告 OM、Clar recall 等 class-sensitive 指标，这对评估 Agent 安全行为极具参考价值。
2. **"较弱 action 优先"偏置原则**：将不对称代价（错误持久化 > 多余提问）形式化为优先级规则，可直接迁移到 Agent 记忆管理的 RL reward shaping 中。
3. **标签-工具分歧度量框架**：MCB-Act 的设计（去掉标签词汇表，强制工具调用，保留 payload 审计）是一个通用模板，可用于评估任何 Agent 规划模块的"说-做一致性"。
4. **反 shortcut 陷阱项**：8 个 lexical traps（如"always"/"today"出现在反直觉语境）可借鉴到任何意图/记忆类 benchmark 以防止模型走捷径。
5. **跨家族对齐干预**：相同提示条件 × 相同 Held-out 项 × 相同统计检验——这一设计范式可作为后续 "prompt 敏感性" 研究的标准协议。

## 关键术语表
**Memory-Clarification Boundary**：Agent 判断一条交互-derived 信息应持久化/局部使用/验证/澄清的决策边界。
**MCB（Memory-Clarification Boundary benchmark）**：本文提出的 140 项主场景 + 70 项对比验证的评测基准。
**Over-memory (OM)**：所有样本中被错误预测为 persist 的比例——衡量"记多了"的安全风险。
**Under-asking**：模型未能选择 clarify 而改选其他 action 的系统性倾向，本文指出现有主流模型均有此缺陷。
**MCB-Act**：MCB 的工具调用变体，要求输出 `memory_write`/`use_now`/`check_source`/`ask_user` 而非 action 标签。
**Weaker-action tie-breaker**：当 persist 与较弱 action 平局时强制选较弱者——编码"不必要提问的代价 < 错误持久化的代价"。
**Paired exact McNemar test with Holm correction**：论文核心统计检验，对同批 item 的前后条件差异做配对检验并控制族错误率。
**Source-of-truth 区分**：区分"世界是事实的真值源（→ verify）"与"用户是意图的真值源（→ clarify）"——MCB 的核心分类学贡献。

## 可复现要素
- **数据集**：MCB，140 主场景 + 70 对比验证项，**公开**（artifact 含全部 frozen 标注）。
- **代码/权重**：提示词、Runner、测试脚本、item-level 预测全部开源；模型权重为 HuggingFace 公开checkpoint（Qwen3.5-9B）及 Claude API。
- **关键超参**：temperature = 0，seed = 13，thinking 禁用，label 模式 output limit = 96 tokens，act 模式 = 128 tokens，Qwen 使用 Q4 K_M 量化（Ollama 0.24.0）。
- **配对重抽样**：2000 次 bootstrap，Holm 校正。
- **模型元数据**：每次 API 调用服务 identifier 均记录在案。
