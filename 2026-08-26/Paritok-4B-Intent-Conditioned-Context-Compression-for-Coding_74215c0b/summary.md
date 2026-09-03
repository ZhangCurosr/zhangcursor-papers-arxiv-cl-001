---
title: "Paritok-4B-Intent-Conditioned-Context-Compression-for-Coding"
source: https://arxiv.org/pdf/2608.24188v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 18:35:55"
field: "代码智能体上下文压缩"
keywords: ["prompt compression", "coding agents", "extractive summarization", "SWE-bench", "LoRA fine-tuning", "context compression", "intent-conditioned", "LLM distillation"]
innovations: ["意图条件化抽取式段级压缩，96%+ 输出 Token 来自输入", "四级别重要性标签与结构标记词汇表的双层压缩框架", "真实 Agent 轨迹五阶段蒸馏管道，40K 验证样本"]
benchmarks: ["SWE-bench Lite (300 instances)", "OOD segment holdout (n=100)", "Extractiveness audit (212,506 tokens)"]
---

# 论文速读：Paritok-4B: Intent-Conditioned Context Compression for Coding Agents

## 一句话总结
论文提出 Paritok-4B，一个专为编码代理（coding agent）设计的 4B 参数 LoRA 上下文压缩器，通过在真实 Agent 轨迹上进行教师蒸馏训练，实现抽取式、意图条件化的段级压缩，在 SWE-bench Lite 上将上下文压缩至原始的 25.7%（约 GPT 压缩器的 2 倍），同时保留 86.5%–89.3% 的单次解决质量。

## 研究问题与动机
- 编码代理在多轮交互中不断将文件读取、工具输出等累积上下文重发给前沿 LLM，这些输入 Token 构成主要成本。
- 通用提示压缩模型（如 LLMLingua-2）面向散文文本训练，会改写标识符或丢失代理编辑所需的确切字符串，导致下游 edit 失败。
- 代理上下文的语义价值取决于当前任务意图，静态压缩无法适应"什么重要"的语境变化。
- 代理上下文高度异构（cat -n 文件读取、pytest traceback、ls 输出、chain-of-thought 等），单一均匀压缩比无法适配所有类型。

## 核心贡献（创新点）
1. **抽取式压缩承诺**：模型选择保留 spans 而非改写，96.0% 的输出标识符/路径/数字直接来自输入，保证精确匹配编辑安全；与改写式摘要模型的本质区别在于保留原文字节完整性。
2. **意图条件化设计**：压缩器以代理当前任务 q 为条件输入，决定哪些行存活；与任务无关压缩器不同，意图在此作为第一特征而非辅助信号。
3. **段级分解架构**：网关将请求分割为带类型标签的段，独立并行压缩，每个段可单独恢复；与整体文档压缩的本质区别在于每个压缩单元可增量更新、并行处理。
4. **真实轨迹五阶段数据管道**：从 67,074 条 OpenHands 轨迹经过滤、标注、蒸馏得到 40,606 个验证样本；与合成数据相比，保留了 cat -n 行号框架等真实代理输入分布。

## 方法详解
- **分段与网关机制**：将 agent 请求分解为 file_read、bash_command、log_output、tool_result、assistant_thinking 等段类型，分配 L0–L3 重要性级别（L0=保护，L1=近期操作，L2=中期历史，L3=过期上下文），目标压缩比分别为 ≤0.50、≤0.35、≤0.25、≤0.20。
- **结构标记词汇表**：使用固定语法的闭合标记替代删除内容（如 `[file: basename.py] [body: N lines]`、`[lines L1-L2: fnA / fnB]`、`[N tests collected]` 等），不在列表内的标记视为格式违规。
- **训练数据来源**：以 gpt-4.1-mini 为教师，通过 OpenAI Batch API（T=0）对 45K 段进行蒸馏，经有效性验证后保留 40,606 个样本（file_read 占 9,830，其他占 30,776）。
- **骨干模型选择**：对比 Qwen2.5-Coder-3B、Qwen3-4B、Qwen2.5-Coder-7B 三种候选，最终选用 Qwen3-4B-Instruct，因代码预训练优势在此任务中未显现——任务由意图驱动的保留决策主导，而非代码生成能力。
- **LoRA 配置**：r=32，α=64，dropout=0，作用于 q/k/v/o/gate/up/down_proj 七个投影，bf16 精度（非 QLoRA），有效 batch=32，max_seq_len=16384。
- **Drop 监督失败**：对 drop 样本损失加权 20 倍导致过度 drop（准确率 42–47%，低于 always-keep 基线 59%），最终 shipped 模型使用 weight=1。
- **Checkpoint 选择**：基于 OOD holdout 滚动评估而非训练 loss 早停；step 2,000 被选为发布版本，因其通过全部四项真实负载测试，且无 level 反序异常。

## 实验与结果
- **基准**：SWE-bench Lite（300 实例），单次调用 claude-sonnet-4-5，temperature=0，无工具调用。
- **压缩率**：Paritok-4B 达 25.7%（原始源输入）/ 27.8%（线号输入，分布内），对比 gpt-4.1-mini 压缩器 50.2%、gpt-5 压缩器 61.9%。
- **质量保留**：86.5%（原始源）/ 89.3%（线号），与 gpt-4.1-mini 的 85.6% 相当，低于 gpt-5 的 93.6%（但 gpt-5 压缩极少）。
- **配对检验**：300 实例中 30 例仅无压缩时解决、17 例仅压缩时解决，McNemar p=0.079，压缩至约 1/4 规模不显著降低解决率。
- **抽取性审计**：在 212,506 个输出 Token 上，96.2% 的标识符/路径/数字来自输入；100% 输出格式合法（well-formed [SEG]）。
- **成本分析**：gpt-5 作为压缩器净成本为负（$9.30 vs $3.00 基线）；Paritok-4B 自托管阈值 < $2.23/M tokens（107 tok/s @ $0.75/h GPU）。

## 相关工作脉络
- **LLMLingua / Selective Context**：基于 perplexity 信号丢弃低信息 Token，Paritok-4B 继承蒸馏到小模型的架构，但以 agent 段为单位、意图条件化、训练于代理轨迹而非文档。
- **LLMLingua-2**：将 GPT-4 蒸馏为 Token 分类压缩器，Paritok-4B 与之的关键差异：输入单位是带类型标签的段（非散文段落）、目标由意图和级别驱动（非任务无关）、三分之一决策为整段丢弃（Token 分类器无法表达）。
- **SWE-bench / OpenHands / SWE-Gym / SWE-rebench**：为训练提供真实 Agent 轨迹分布，使压缩器可学习 cat -n 行号框架等代理特有输入模式，而非直接处理原始源码。
- **Prompt Compression 社区**：Paritok-4B 的定位在于证明"意图可用时不应隐藏"——coding agent 场景中当前任务可零成本获得，应作为压缩决策的核心特征而非辅助信号。

## 局限性与未来方向
- Drop 召回率仅 0.24，低于 always-keep 基线 0.59，是当前最大性能缺口。
- 四级别 L0–L3 设计在蒸馏结果中坍缩为两档（保护/近期 vs. 过期），学生模型从未接收数值级 token 预算。
- 标识符保留是倾向性（96–98% 来自输入）而非保证，约 60% 的标识符类 Token 被有意丢弃，需配合 read_original 回退机制。
- 训练分布以 Python/SWE-bench 为主，其他语言未评测；架构语言无关但启发式规则针对 Python 调优。
- 端到端评测为单次调用、无工具循环，未覆盖多轮 Agent 成本与精确匹配编辑场景。
- 仅监督蒸馏自单一教师，未探索下游信号的直接优化（如 DPO/RL）；v2 方向包括显式数值预算传递与偏好学习。

## 研究启发与可借鉴点
- **部署约束即设计约束**：将 24GB GPU 自托管作为第一性原理，反向推导模型尺寸上限（4B），而非先追求学术指标再考虑成本——对 Agent tooling 类研究具有参考价值。
- **蒸馏教师 prompt hill-climbing**：用人工迭代而非自动搜索优化教师 prompt（v5→v15），针对 over-/under-folding 收集 preference pairs，可作为小型蒸馏项目的实用策略。
- **审计驱动的设计承诺**：将"抽取式"作为可测量指标（96.0% 行级拷贝、96.2% Token 拷贝）而非口头声明，并在 OOD 数据上重新验证；这种透明度为领域内建立评估标准提供了范式。
- **失败经验即贡献**：公开 drop 监督加权 20 倍的失败及原因分析，避免后人重复相同错误——对于可复现工程研究具有示范价值。
- **分布内评测的重要性**：论文额外报告 cat -n 线号输入下的结果（27.8% / 89.3%），揭示训练分布与保守评测分布的差异，提醒读者关注"模型实际部署形态"对应的性能。

## 关键术语表
- **Extractive compression**：抽取式压缩——模型从原文中选择保留 spans 而非改写生成，保证输出与输入间的字节级对应关系。
- **Intent-conditioned**：意图条件化——压缩决策以代理当前任务查询 q 为条件，使保留内容相对任务动态变化而非静态评分。
- **Segment-level compression**：段级压缩——将请求分解为 typed segments（file_read/tool_result/bash_command 等），独立压缩每个段，支持并行与增量更新。
- **Structural marker**：结构标记——用于替代被删除内容的闭合词汇表占位符（如 `[body: N lines]`），携带输入中的计数/名称信息，不在列表内即为格式违规。
- **L0–L3 importance levels**：四级重要性标签——系统（L0）、近期操作（L1）、中期历史（L2）、过期上下文（L3），每级对应不同目标压缩比。
- **Must-keep spans**：必保 spans——由启发式规则提取的路径、标识符、错误类、行号等必须完整保留的关键实体集合。
- **Edit recovery / read_original**：编辑恢复——网关暴露的精确回退接口，允许代理按需从原始上下文读取任意段的未压缩字节。
- **McNemar test**：McNemar 检验——用于配对二分类数据的统计检验，此处用于判断压缩 vs 无压缩在 300 实例上的解决率差异是否显著。

## 可复现要素
- **数据集**：OpenHands 轨迹来源于 SWE-rebench 与 SWE-Gym（~67K 条）；SWE-bench Lite 300 实例仅用作 held-out 评估集，不进入训练。数据管道脚本已开源。
- **代码/权重**：全部开源（Apache 2.0），GitHub: https://github.com/Paritok-official/paritok-4b-v1；模型权重在 Hugging Face Hub。
- **关键超参**：LoRA r=32, α=64, dropout=0；lr=1e-5 linear decay, warmup=10%；effective batch=32；max_seq_len=16384；bf16；seed=42；2 epochs / 2,538 steps；shipped checkpoint at step 2,000。
- **教师模型**：gpt-4.1-mini-2025-04-14，T=0，通过 OpenAI Batch API 蒸馏。
- **评估脚本**：eval/extractiveness.py（Table 2）、eval/design_claims_audit.py（Table 4）、eval_model/（SWE-bench Lite harness）均已开源。
