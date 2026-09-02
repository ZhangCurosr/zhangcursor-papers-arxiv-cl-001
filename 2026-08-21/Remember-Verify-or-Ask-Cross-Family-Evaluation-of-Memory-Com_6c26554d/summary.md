---
title: "Remember-Verify-or-Ask-Cross-Family-Evaluation-of-Memory-Com"
source: https://arxiv.org/pdf/2608.19564v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 19:45:38"
field: "LLM Agent 长期记忆与工具使用"
keywords: ["LLM agents", "long-term memory", "clarification", "tool-use evaluation", "benchmark", "memory commitment"]
innovations: ["提出 MCB 基准，首次区分持久化/局部使用/验证/澄清四类记忆承诺决策并独立非作者标注", "设计 MCB-Act 工具调用变体，揭示 label↔tool-call 行为偏差跨模型差异显著", "建立跨家族配对评测协议，证明 policy prompt 可在总准确率不变时显著降低 over-memory 安全错误"]
benchmarks: ["MCB (Memory-Clarification Boundary)", "MCB-Act (tool-call variant)", "LongMemEval", "CLAMBER", "MemBench", "PerMemBench"]
---

# 论文速读：Remember, Verify, or Ask — Cross-Family Evaluation of Memory Commitment

## 一句话总结
论文提出了 **MCB（Memory-Clarification Boundary）基准**，系统评测 LLM Agent 在面临候选更新时是持久化存储、仅当前上下文使用、验证外部事实，还是向用户澄清的四类决策能力；跨 Claude 与 Qwen 两大家族实验揭示模型普遍存在"少问"（under-asking）偏差，且陈述性选择与结构化工具调用行为之间存在显著不一致。

## 研究问题与动机
- **持久化记忆的静默风险**：LLM Agent 保留交互历史可支持个性化，但一次性的临时请求若被错误持久化，会无声地扭曲后续行为。
- **现有基准侧重"回忆"而非"承诺"**：已有评测（如 LongMemEval、LoCoMo）聚焦检索与复用能力，缺少对"信息应在何时、以何种方式被持久化"这一承诺决策的直接评测。
- **澄清与验证的本质区分**：用户是意图与范围的信息权威来源（clarify），世界是可变事实的信息权威来源（verify），两者不应被混同；但模型常将模糊意图误映射为向外部世界核查。
- **标签选择≠工具调用行为**：Agent 在生成 action label 时的口头决策，未必能正确迁移到实际的 structured tool-call 选择，需分开评测。

## 核心贡献（创新点）
1. **首个面向记忆承诺边界的可审计基准（MCB）**：非作者独立标注 70 个 held-out 测试项（Cohen's κ = 0.962），含 8 个反捷径词法陷阱，并区分持久化/局部使用/验证/澄清四类原子决策。
2. **跨家族（Claude + Qwen）一致干预实验**：两套不同架构模型在相同的三条 prompt 条件下（bare / policy / few-shot）进行配对比较，证明"少问"偏差跨厂商泛化但具体错误策略存在模型依赖差异。
3. **MCB-Act 工具调用选择变体**：模型必须生成含参数的一类结构化工具调用（`memory_write` / `use_now` / `check_source` / `ask_user`），而非自由文本标签，首次揭示 label-mode 与 act-mode 行为偏差（Qwen 仅 23% 一致）。
4. **安全相关行为的分离指标设计**：除总准确率外，单独报告 over-memory（错误持久化率）和 clarification recall，证明 policy prompt 可在总准确率不显著提升时大幅降低安全相关错误（Qwen OM 从 0.243 降至 0.100）。

## 方法详解
- **任务定义**：每个 item 含 acquire 上下文（候选更新）、later reuse 上下文（后续重用场景），要求模型输出四类动作之一：
  - **persist**：明确持久的偏好、授权策略或稳定事实；
  - **ephemeral**：限定于单一 artifact、会话或时间窗口的信息；
  - **verify**：易变的世间状态（如餐厅营业时间）或单次噪声信号；
  - **clarify**：未解决的指代、范围冲突、耐久性歧义等，仅用户有最终权威。
- **评分规则**：当 persist 与更弱动作并列时，规则倾向于较弱承诺（不对称代价：不必要的提问易被发现和纠正，错误持久化则可能静默累积）。
- **三种 prompt 条件**：
  - **bare**：仅定义四类动作；
  - **policy**：增加五条承诺规则（含上述 tie-breaker）；
  - **few-shot**：每种动作附一个 development-set 示例。
- **MCB-Act 变体**：移除标签词汇表，模型必须输出一个带字符串参数的结构化调用对象（`memory_write(x)` / `use_now(x)` / `check_source(x)` / `ask_user(x)`），由确定性内容规则验证参数合法性。
- **统计检验**：所有系统共享同一 70 项 held-out 集， headline 差异采用配对 bootstrap 95% CI + Holm 校正精确 McNemar 检验（$p_H$）；行为率（如 OM）在 item 级事件上应用相同配对检验。
- **数据拆分**：140 项主集按类别内有序 ID 确定性地分为 70/70；另设 70 项 contrast 集合（含 35 对证据翻转对），独立盲标 $\kappa=1.000$。

## 实验与结果
- **数据集**：MCB 主集 140 项（70 dev + 70 held-out），contrast 集 70 项；action 分布：38 persist / 40 ephemeral / 33 verify / 29 clarify；每类 20 项（含 8 个词法陷阱）。
- **评测模型**：Claude Haiku 4.5、Claude Sonnet 4.6（API 调用）、Qwen3.5-9B（Ollama 本地，Q4_K_M 量化，temp=0，seed=13，禁用 thinking）。
- **主要结果（Table III，70 held-out 项）**：
  - **Claude Haiku bare**：准确率 0.629，OM 0.029，Clar. 0.500，Ver. 0.944。
  - **Claude Haiku policy**：准确率 0.857（$p_H=0.002$），Clar. 0.750，Ver. 0.944。
  - **Qwen bare**：准确率 0.557，OM 0.243（极高），**Clar. 0.000**，Ver. 0.667。
  - **Qwen policy**：准确率 0.629（ns），OM **0.100**（$p_H=0.038$），Clar. 0.083，Ver. 0.944。
  - **Qwen few-shot**：准确率 **0.771**（$\Delta=+0.214$, $p_H=0.002$），Clar. 0.333，Ver. 0.833。
  - **MCB-Act 工具调用一致性**：Claude 两款 label↔act 一致率 0.571；Qwen 仅 0.229，准确率从 0.557 骤降至 0.343（$p_H=0.047$），其中 `use_now` 调用占比达 54/70。
- **对比强基线**：Always-Persist 准确率仅 0.257、OM 0.743；Category oracle 0.800（暴露场景类型携带信号）。
- **关键结论**：few-shot 提升总准确率，policy 降低过记忆错误——**总准确率掩盖了安全行为维度的改善**；澄清召回是所有模型最弱能力（bare Qwen 为 0）。

## 相关工作脉络
1. **LongMemEval / LoCoMo / MemBench**：评测长期对话记忆与检索复用，但不评分存储门控、澄清或验证行为——MCB 补充了承诺决策层面的直接评测。
2. **PerMemBench**：学习二元 session-level 存储门，但混淆"世界"与"用户"两类不确定性来源——MCB 明确分离二者。
3. **Memory-R1**：学习 ADD/UPDATE/DELETE/NOOP 操作，聚焦记忆管理动作而非承诺边界的四类决策（含 verify/clarify 区分）。
4. **Mem2ActBench**：评测检索记忆是否 grounded 到工具参数，与 MCB 互补——后者关注候选更新到达时的承诺决策，而非检索后利用。
5. **CLAMBER**：关注识别和澄清模糊信息需求，但未区分"向用户澄清"与"向世界验证"两种权威来源——MCB 建立了 source-of-truth 区分。
6. **τ-bench / τ²-bench**：展示静态答案与工具中介行为可能 diverge——MCB-Act 将这一洞察应用于记忆边界，要求结构化工具调用而非标签输出。

## 局限性与未来方向
- **测试集规模有限**：70 项主集置信区间较宽，每类别仅 10 项不支持稳健的类别级排名。
- **场景为合成且英文**：未覆盖真实多语言对话或个人记录，外部有效性存疑。
- **Qwen 仅一个量化 checkpoint**：Q4_K_M 量化、Ollama 服务栈、禁用 thinking 均为评测系统一部分，推广需谨慎。
- **Act 模式仅评测 bare 条件**：policy/few-shot 条件下的工具调用选择留作未来工作。
- **MCB-Act 不执行下游效果**：记录的调用不实际写入记忆、不查询外部源、无模拟用户回复，未量化端到端效用改善。
- **标签由作者制定**：虽经非作者盲审，但场景和规则本身为作者编写；contrast 集规则与模板高度对齐，perfect $\kappa$ 可能反映人工对齐而非自然泛化。

## 研究启发与可借鉴点
1. **行为特异性指标设计**：总准确率无法捕捉安全相关干预效果（Qwen policy 总准确率 ns 但 OM 显著下降），后续工作应报告 class-sensitive 错误率（如 OM、under-asking）而非仅 headline accuracy。
2. **Label↔Tool 一致性评测框架**：MCB-Act 的设计揭示了"会说"和"会做"之间的 gap，可迁移至其他 agent 评测场景（如 planner→executor 对齐、intent→action 映射）。
3. **配对检验 + Holm 校正的轻量统计协议**：所有系统在相同 held-out 集上运行，允许精确配对测试，避免了独立置信区间重叠的误导；适合小样本 agent 评测。
4. **反捷径词法陷阱（lexical traps）**：在场景中嵌入"always""today"等词但在错误上下文中，防止模型依赖表面启发式——可在记忆/规划相关基准中复用作鲁棒性检测。
5. **source-of-truth 分离评估**：区分"向用户问"与"向世界查"两类澄清，为 agent 多源信息获取决策提供新的评测维度。

## 关键术语表
- **Memory-Clarification Boundary（记忆-澄清边界）**：Agent 在接收到候选更新时，决定持久化、局部使用、验证或澄清四种操作的决策分界线。
- **Over-memory（过度记忆，OM）**：所有 item 中错误预测为 persist 的比例，衡量错误持久化的安全相关风险。
- **Under-asking（提问不足）**：模型在应触发 clarify 的场景中选择其他动作（如 persist 或 verify），而非向用户澄清的偏差。
- **MCB-Act**：MCB 的工具调用选择变体，要求模型输出带参数的结构化调用而非自由标签，用于检验 label↔action 一致性。
- **Contrast set（对比集）**：70 项独立于主测试集的鲁棒性检验集合，含 35 对证据翻转对，用于验证评测抗短路径能力。
- **Holm-adjusted exact McNemar $p_H$**：对多重比较进行 Holm 校正的精确配对 McNemar 检验，控制 familywise error rate。
- **Ephemeral**：四类承诺动作之一，指仅适用于单一 artifact/会话/时间窗口的信息，不写入持久记忆。
- **Verification vs. Clarification**：Verify 查询外部世界以确认可变事实，Clarify 向用户询问以解决意图/范围歧义——两者权威来源不同。

## 可复现要素
- **数据集**：MCB 主集 140 项 + contrast 集 70 项，论文声明包含在附加材料中。
- **代码/权重**：完整 artifact 包含数据、盲标注、prompts、runners、item-level 预测、模型元数据、检验脚本；Ollama 0.24.0 + Qwen3.5-9B（Q4_K_M 量化，digest: 6488c96fa5fa…eda893ea7）。
- **关键超参**：temperature=0，seed=13，禁用 thinking；label mode 输出限制 96 tokens，act mode 128 tokens；API 独立 per-item 调用。
- **统计实现**：2,000-resample percentile bootstrap 95% CI，配对精确 McNemar 检验，Holm 校正。
