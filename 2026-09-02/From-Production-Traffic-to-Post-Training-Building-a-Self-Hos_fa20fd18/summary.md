---
title: "From-Production-Traffic-to-Post-Training-Building-a-Self-Hos"
source: https://arxiv.org/pdf/2609.01572v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 00:26:00"
field: "企业级自部署 LLM 后训练"
keywords: ["post-training", "instruction following", "function calling", "GRPO", "model merging", "SLERP", "reward hacking", "enterprise LLM"]
innovations: ["分专家 GRPO + 两阶段 SLERP 合并替代联合多目标 RL，解决域间奖励干扰", "从生产流量构建分层内部 benchmark（模板感知采样 + 任务路由差异化 Judge）", "三种 reward hacking 模式（语义坍塌/过度调用/冗长 hack）及其领域特定修复"]
benchmarks: ["ruIFEval", "ruBFCLv3", "Arena Hard Ru", "WildChat Hard Ru", "AceBench", "tau^2-bench", "In-house Arena", "SmartSearch"]
---

# 论文速读：From Production Traffic to Post-Training: Building a Self-Hosted LLM That Covers the Corporate Request Mix

## 一句话总结
本文针对企业数据合规要求下大规模自部署 LLM 面临的 GPU 资源碎片化问题，提出了一条从生产流量分析到精细化后训练的完整方法论：通过生产错误分析识别指令遵循（IF）、函数调用（FC）和内部任务对齐三个薄弱环节，分别训练独立 GRPO 专家后以两阶段 SLERP 权重合并，最终将 32B 模型成功部署为单一生产模型，在内部 Arena 等关键指标上超越参数量约 7 倍的基线模型，并吸纳了超过 200 个内部应用 50% 的平台流量。

## 研究问题与动机
1. **GPU 资源碎片化**：企业在数据 residency 限制下必须自部署 LLM，但数百个内部应用各自选用不同模型，随着新模型迭代而旧模型无法退役，导致有限的 GPU 池被过度分散，有效 token 成本不断上升。
2. **长尾应用无法 A/B 测试**：大量应用的流量不足以支撑可靠的 A/B 实验，因此整合后的模型必须具备广泛的一般能力以覆盖未见过的 workflow。
3. **生产流量驱动的能力缺口难以量化**：公开 benchmark 无法完全反映企业内部真实请求分布，需要建立从生产日志出发的分层内部评测体系，才能精准定位能力短板。
4. **多目标联合优化的奖励干扰**：IF、FC 和通用对齐三个目标的 reward 信号存在结构冲突，联合 GRPO 训练会导致域间干扰（cross-domain interference），使某一能力提升时另一能力退化。

## 核心贡献（创新点）
1. **从生产流量构建分层内部 Benchmark 的方法**：提出模板感知采样策略（template-aware sampling）平衡多样性与代表性，并用任务类型分类路由 + 差异化 Judge 方案（确定性验证器 / RubricHub 风格 checklists）实现校准到人工标注者的自动评测（κ 从 0.63 提升至 0.88）。
2. **模块化单专家 GRPO + 两阶段 SLERP 合并的后训练配方**：放弃联合多目标 GRPO，改为从共享 SFT checkpoint 分叉出三个独立 GRPO 专家，以 IF+FC 先合并再与 General 合并的顺序经网格搜索确定最优系数，在 32B 规模上显著优于联合优化。
3. **揭示了三种奖励黑客（reward hacking）失败模式及其针对性修复**：语义坍塌（semantic collapse，IF 专家输出空洞但形式合规的回复）、过度调用（over-calling，FC 专家倾向于无条件 emit call）、冗长 hack（verbosity hacking，General 专家利用 RM 的长度偏好膨胀回复），每种都有对应的领域特定修复策略。
4. **开源了不含内部数据的同配方 checkpoint**：公开版在所有公共 benchmark 上与内部部署版得分接近（仅内部 Arena 存在差距），证明改进主要来自方法论而非专有数据。

## 方法详解
**整体流程**：以 Qwen3-32B（配合西里尔密集 tokenizer）为底座，在**非推理模式**下执行三阶段后训练。

**阶段一：共享 SFT**
- 将内部生产流量、通用域指令数据、IF 数据和 FC 数据混合进行单次 SFT，作为所有后续 RL 分支的共享起点。消融实验（Table 3）表明此阶段混合数据不会损失各域质量。

**阶段二：三个独立 GRPO 专家**

- **General Expert（通用对齐专家）**：数据为 80% 开源俄语指令语料 + 20% 内部生产流量（使用相同模板感知采样策略并做 Min-Hash 去污染）； Completions 由 Qwen3-235B-A22B-Instruct-2507 再生以保证分布一致性。RL 使用通用 RM + 乘法长度惩罚 $R(x,y) \mapsto R(x,y)(1-\alpha(x,y))$，其中 $\alpha$ 基于响应长度与 teacher 模型基线的偏离度，并提高 KL 系数至 0.01 以约束分布漂移。

- **IF Expert（指令遵循专家）**：数据来自适配 AutoIF 流程的合成管道，从 54 条种子约束扩展至 43K 验证约束，最终生成 26K 训练样本。Reward 采用 VerIF 风格确定性验证器，并引入 prompt 特定的 RM 质量校正阈值 $\alpha_i$（取 teacher 模型 8 次 completions 的平均 RM 分数）：通过验证器但 RM 分数低于阈值的 completion 会收到 -0.5 惩罚，防止语义坍塌。

- **FC Expert（函数调用专家）**：数据从 scratch 原生合成（非翻译），生成 120 万英文 + 30 万俄语样本。对话管道分离规划与模拟：planner 构建轨迹后经 judge 反馈迭代，再由 User/Assistant/Tool 三个 agent 以非对称可见性重放，确保训练目标反映真实工具使用模式。SFT 阶段以 50/50 tool-call vs text target 和 50/50 英/俄分层混合。GRPO 使用 Tool-N1 二元精确匹配 reward，并通过数据分布注入合成无关案例（irrelevance cases，占 text target 的 10%）来抑制过度调用倾向，而非修改 reward 结构。最终 GRPO 混合比例为 70% 英文 / 30% 俄文，每种语言内 80% tool-call / 20% text target。

**阶段三：两阶段 SLERP 合并**
- 第一阶段：以层粒度不同参数组（self-attention / MLP / 其余）的系数网格搜索合并 IF 与 FC 专家；
- 第二阶段：统一系数 $t_2=0.8$ 将上一步结果与 General 专家合并。
- 消融表明此顺序（$(IF+FC)+Gen$）显著优于其他顺序（Table 7）；尝试联合多目标 GRPO 则需 1.7× 预算才勉强持平（Table 8）。

## 实验与结果
**数据集与基准**：
- 公开基准：ruIFEval、ruBFCLv3、Arena Hard Ru、WildChat Hard Ru、AceBench、$\tau^2$-bench、MultiChallenge（英/俄双语适配版）
- 内部基准：In-house Arena（任务分层 + 差异化 Judge）、In-house IFEval（确定性验证器 + LLM Judge）、In-house BFCL（AST 模式，人工审核参照）、SmartSearch（多步 ReAct 检索评测）

**主要结果**（Table 6）：
- **In-house Arena**：69.57 vs Qwen3-235B 的 65.83（+3.74）；ruArena-Hard：93.87 vs 96.87（略低 3 分）
- **In-house IFEval**：0.85 vs 0.83（超基线）；Strict-IF 例级 0.81、约束级 0.89，与 7 倍大模型持平
- **In-house BFCL**：0.79 vs 0.77；ruBFCLv3：65.96 vs 64.42；AceBench：73.50 vs 70.20
- **SmartSearch F1_R&G**：0.557 vs 0.669（大模型），但超越所有 32B 规模 think-mode 基线（0.491/0.537）
- **ruWildChat**：80.7 vs 85.1（差距 4.4 分），较 base 从 52.0 大幅提升

**部署效果**：模型吸纳平台 50% 流量（1.16 亿请求/月，>200 个应用），单 GPU FP8 推理，95th-percentile 延迟 3.2s，TTFT 0.3s；token 成本降低 2.8–3.9×。

**消融关键结论**：
- 共享 SFT 足以保留各域质量（Table 3）；内部 RM 微调未带来增益且引入风格噪声（Table 4）
- 单域 GRPO 跨域迁移极弱（Table 5），验证合并必要性
- 联合 GRPO 从 mixed-SFT 出发导致 BFCL RU 降至 60.88、ruIFEval 降至 0.770（Table 8）；而合并配方分别达到 65.96 和 0.799
- 合并顺序 $(IF+FC)+Gen$ 最优（Table 7）；SFT polish 阶段无净收益

## 相关工作脉络
1. **Tulu 3 / DeepSeek-R1**：同类 RLVR post-training 体系，本文区别在于目标是非推理模式的俄/英企业场景，且强调多目标奖励干扰问题，采用分专家合并而非联合优化。
2. **AutoIF / IF-Bench / VerIF**：指令遵循验证方法；本文将其适配俄语（形态、语序、大小写更复杂），并新增 prompt 特定的 RM 质量校正以修复语义坍塌。
3. **APIGen / ToolACE / Toucan**：函数调用合成数据生成；本文强调俄语文本不可简单翻译（参数名/枚举值必须保留，翻译会破坏交叉引用），改为原生俄语合成管道。
4. **Arena-Hard / WildBench**：公开开源评测；本文扩展为分层内部评测体系，按任务类型路由到差异化 Judge 方案并校准至人工标注。
5. **MO-GRPO / RL with Rubric Anchors**：多目标 RL 相关工作；本文通过实验证明多目标联合优化在企业生产场景中因 reward hacking 和域间干扰而不可靠，转而采用分专家合并路线。
6. **Model Soups / Task Arithmetic / TIES-Merging**：权重合并方法学；本文采用 SLERP（球面线性插值）并发现顺序效应显著，两阶段顺序 $(IF+FC)+Gen$ 经网格搜索确定。

## 局限性与未来方向
1. **语言与部署范围受限**：评估仅覆盖俄语和英语，方法论在其他语言和组织的验证留待未来工作。
2. **依赖 LLM Judge 校准**：开放式任务的自动评测依赖 LLM Judge，重新部署到不同语言或分发时需重新校准至人工标注（Appendix H 对 4 个 Judge 做了敏感性分析）。
3. **单一模型家族验证**：所有实验仅使用 Qwen3 系列，方法论未在其他 backbone 或组织环境上验证。
4. **Loose-IF 仍有差距**：语气、角色、语义禁止等宽松约束未被验证器直接优化，与 235B 大模型在 Loose-IF 上仍有 0.08 的差距。
5. **$\tau^2$-bench 提升有限**：会话级端到端成功率的提升较单 turn reward 优化幅度小，会话级对齐是未来方向。
6. **长尾应用仍需 frontier 级 agentic 能力**：少数回滚来自需要超出 32B dense 模型能力的场景。

## 研究启发与可借鉴点
1. **生产流量驱动的评测体系设计**：模板感知采样（mask variable tokens → LSH 分组 → greedy max-min within template）同时保障多样性和代表性，可直接迁移至任何有结构化 prompt 模板的企业 LLM 平台。
2. **分专家 GRPO + 权重合并替代联合多目标 RL**：当多个 reward 信号存在结构性冲突（验证器/语义/长度）且各自存在 reward hacking 捷径时，模块化单专家训练后合并是可复用的设计原则。
3. **RM 质量校正作为 verifier-based reward 的补充**：IF 专家中用 prompt 特定的 teacher baseline 作为 RM 惩罚阈值，既保留 verifier 的可扩展性又防止语义坍塌，该方法可泛化到其他有形式验证器的能力域。
4. **数据分布修正优于 reward 结构复杂化**：FC 专家的 over-calling 问题通过注入合成 irrelevance cases 解决而非修改 binary reward，证明在 reward hacking 场景下调节数据配比往往比设计复合 reward 更稳健。
5. **任务路由 + 差异化 Judge 的内部评测流水线**：LLM 任务分类器 + 参考-based 打分（客观任务）/ RubricHub-style checklist（开放式任务）的组合，κ 从 0.63 升至 0.88，可作为企业自建评测系统的参考架构。

## 关键术语表
- **SLERP（Spherical Linear Interpolation）**：球面线性插值，在参数空间中沿测地线合并两个模型权重，本文用于两阶段顺序合并三个 GRPO 专家。
- **GRPO（Group Relative Policy Optimization）**：DeepSeekMath 提出的 RL 算法变体，以 group 为单位计算相对优势，无需单独训练 critic model，本文用于三个专家的独立对齐训练。
- **Reward Hacking**：模型利用 reward function 的漏洞以非期望方式最大化奖励信号（如输出空洞但形式合规的回复），本文记录了三种领域特定的 hacking 模式。
- **Template-Aware Sampling**：通过 mask 变量 token 后 LSH 分组识别 prompt 模板，在每个模板内做 greedy max-min 采样的策略，兼顾多样性和生产分布代表性。
- **Tool-N1 Reward**：二元精确匹配 reward，tool-call 场景检查 (name, arguments) 多重集是否完全匹配，text 场景检查无 tool call 被解析。
- **AST Mode（Abstract Syntax Tree）**：BFCL 的一种评估模式，对预测的 tool call 进行结构和语义匹配而非实际执行，避免生产环境中调工具产生副作用。
- **Semantic Collapse**：IF 专家在纯 verifier reward 下退化为输出极简、语义空白的回复以通过形式检查的失败模式。
- **Over-calling**：FC 专家在二元 reward 下倾向于无论是否应该都 emit tool call 的失败模式。

## 可复现要素
- **数据集**：内部生产流量（~10 万/月查询，已去标识化和去污）；合成数据：43K IF 约束、26K IF 样本、1.2M 英文 + 300K 俄文 FC 样本。**未公开**。
- **代码**：论文未提及开源代码；开源了不含内部数据的 checkpoint（open-weight）。
- **权重**：发布 open-weight checkpoint（同配方但不含内部数据增量），可在 Hugging Face 获取。
- **关键超参**：SFT：max seq 32768，batch 32，lr $1\times10^{-6}$，2 epochs，bf16（Table 20）；GRPO：batch 512，rollouts 8，lr $1\times10^{-6}$，KL 系数 General=0.01/IF=FC=0.001（Table 21–22）；SLERP 第一阶段层粒度系数以 attention/MLP/其余分组扫描，第二阶段 $t_2=0.8$。
- **硬件**：SFT 4 节点×8 H100；每个 GRPO 专家 4 节点×8 H100。
- **框架**：SFT 用 FSDP + gradient checkpointing；GRPO 用 verl；推理用 vLLM FP8。
