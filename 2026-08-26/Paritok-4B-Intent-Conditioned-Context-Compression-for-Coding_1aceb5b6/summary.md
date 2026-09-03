---
title: "Paritok-4B-Intent-Conditioned-Context-Compression-for-Coding"
source: https://arxiv.org/pdf/2608.24188v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 14:53:03"
field: "代码 Agent 上下文压缩"
keywords: ["context compression", "coding agent", "extractive compression", "intent-conditioned", "SWE-bench", "LoRA fine-tuning", "prompt compression"]
innovations: ["提取式+意图条件化的 coding agent 专用压缩器，96.2% 输出标识符来自输入复制", "五阶段可复现蒸馏管线，将67K真实agent轨迹压缩为40K高质量segment样本", "per-segment分解架构支持并行压缩与单段精确恢复，压缩至25.7%大小且不显著降低解决率"]
benchmarks: ["SWE-bench Lite", "OOD segment holdout", "intrinsic format/drop/identR evaluation"]
---

# 论文速读：Paritok-4B-Intent-Conditioned-Context-Compression-for-Coding

## 一句话总结
Paritok-4B 是一个 4B 参数 LoRA 压缩器，专门针对 coding agent 的工作轨迹训练，通过**提取式（extractive）**选择和**意图条件化（intent-conditioned）**两条核心设计，将 agent 上下文压缩至原始大小的约 1/4，同时在 SWE-bench Lite 上保留 86.5%~89.3% 的单次解决质量，显著优于同等规模下的 GPT API 压缩器。

## 研究问题与动机
- **Coding agent 的 token 账单主要由上下文累积主导**：每个 turn 都会重新发送文件读取、命令输出、历史消息等到前沿 LLM，这些输入而非模型输出占据了大部分 token 成本。
- **通用 prompt 压缩模型不适配代码场景**：现有压缩器（如 LLMLingua-2）训练于散文语料，会改写标识符、丢失 agent 精确匹配的编辑 span，导致下游编辑失败。
- **代码段的价值取决于 agent 当前意图**：一个函数对即将修改它的 agent 至关重要，但对其他任务无关紧要——压缩策略需要感知实时意图而非静态打分。
- **Agent 上下文高度异构**：`cat -n` 文件读取、pytest traceback、`ls` 输出、chain-of-thought 等段落类型差异巨大，单一均匀压缩比无法适用所有类型。

## 核心贡献（创新点）
1. **提取式压缩承诺**：模型选择保留 span 而非重写，96.2% 的输出标识符/路径/数字来自输入复制；与已有工作本质区别在于通过审计（而非断言）验证"提取性"，并设计封闭结构化标记词表替代删除内容。
2. **意图条件化压缩**：压缩器接收 agent 当前任务作为输入，训练目标是在保留 segment 内基于意图相关性选择行（保留行比删除行意图相关度高 +0.067）；与通用 task-agnostic 压缩器相比，充分利用了 agent 当前意图这一免费且高度 informative 的特征。
3. **五阶段可复现数据蒸馏管线**：将 67,074 条真实 OpenHands 轨迹蒸馏为 40,606 个教师验证样本；创新在于保留 `cat -n` 行号格式的训练分布，而非使用原始源码。
4. **per-segment 分解架构**：gateway 将请求拆分为带类型和重要性级别的 segment，独立并行压缩，每个 segment 可单独恢复原始字节；与端到端全量压缩形成对比。
5. **公开的自我审计与评估工具链**：发布 extractiveness audit 脚本、设计审计脚本，以及在 OOD holdout 上的系统化内蕴评估与端到端 SWE-bench Lite 评测。

## 方法详解
- **任务形式化**：将编码 agent 的请求序列 $s_1, \ldots, s_n$ 拆解为 typed segments（kind + level），gateway 对每个 segment 调用一次压缩模型，输入为 agent 当前任务 query $q$ 和带类型/级别标签的 segment，输出为该 segment 的压缩形式。
- **四种 segment 类型**：`file_read`、`bash_command`、`log_output`、`tool_result`、`file_operation`、`directory_listing`、`assistant_thinking`、`meta_action`，另有受保护的 system/user messages。
- **四级重要性标签（L0–L3）**：L0 受保护（system prompt、用户任务、最近工具输出）、L1 近期读取/动作、L2 中期历史、L3 过期上下文；目标是实现逐级更高压缩率，但实际训练中 L0/L1 坍缩为"受保护/近期"、L2/L3 坍缩为"过期"两个有效波段。
- **提取性保证机制**：保留的代码行、标识符、文件路径、行号、import、错误类、shell 命令原文均逐字复制；删除内容用封闭词表的结构化标记替换（如 `[body: N lines]`、`[lines L1-L2: fnA / fnB - note]` 等共 10 种）。
- **意图条件化作用位置**：不在 segment 级筛选（由 kind/level 规则主导），而在 segment 内部行级选择——保留行在意图命名实体覆盖上显著高于删除行（paired difference +0.067，95% CI [+0.056, +0.078]）。
- **数据管线五阶段**：(1) 从 SWE-rebench/SWE-Gym 下载 67,074 条 OpenHands 轨迹；(2) 在 assistant 决策点分段，保留 `cat -n` 格式；(3) 过滤真实动作样本且含 ≥1000 token 的 compressible segment；(4) 无 LLM 自动标注级别与 must-keep span；(5) 轨迹级 80K train/4K val 池；(6) 从池中提取 45,000 个 segment 经 gpt-4.1-mini 教师蒸馏 + 严格校验（通过率 90.2%），最终 40,606 个有效样本。
- **训练配置**：基于 Qwen3-4B-Instruct-2507，LoRA (r=32, α=64, dropout=0) 作用于 7 个投影层，bf16 精度，8-bit AdamW，学习率 $1 \times 10^{-5}$ 线性衰减+10% warmup，effective batch=32，max sequence=16,384，2 epoch 计划；最终选择 step 2,000 checkpoint（1.58 epoch）。
- **Drop 监督失败教训**：对 drop 样本 loss 加权 20 倍导致过度 drop（accuracy 42–47% 低于 always-keep 基线 59%）， shipped model 以 weight=1 保守训练，接受 keep-biased 策略。
- **Checkpoint 选择策略**：不采用 loss early-stop，而是对最后 5 个 checkpoint 在 OOD holdout 上 sweep，并结合真实流量 smoke test，step 2,000 通过全部四项行为测试且无 level inversion 异常。

## 实验与结果
- **内蕴评估（OOD segment holdout, n=100）**：
  - Format reliability：100% 输出为合法单 [SEG] 结构（95% CI [0.963, 1.000]）。
  - Drop accuracy：0.49（低于 always-keep 基线 0.59），drop recall 仅 0.24——主要压缩空间来源。
  - Must-keep identifier retention（IdentR）：0.383，在 student/teacher 均选择保留的 39 个 segment 上，student 0.385 vs teacher 0.287（paired difference +0.098，p=0.08，方向有利但不显著）。
  - Token copy rate（SWE-bench Lite 212,506 tokens）：96.2% 已存在于输入。

- **端到端评估（SWE-bench Lite, 300 instances）**：
  - **压缩率**：Paritok-4B 原始源码 25.7%，line-numbered 分布内 27.8%；gpt-4.1-mini 压缩器 50.2%，gpt-5 压缩器 61.9%。
  - **质量保留**：原始源码 86.5%（与 gpt-4.1-mini 的 85.6% 无显著差异），line-numbered 分布内 89.3%；gpt-5 为 93.6%。
  - **配对显著性**：line-numbered 模式下，30 个仅无压缩解决、17 个仅压缩解决，McNemar p=0.079——在 300 样本量下压缩至约 1/4 不显著降低解决率。
  - **应用失败**：压缩 arm 有 16 个 patch 无法 apply（baseline 5 个），压缩引入的真实失败模式。

- **成本分析**：
  - gpt-5 作为压缩器净亏损 +$6.30/M tokens；gpt-4.1-mini 仅节省 -$0.29/M。
  - Paritok-4B 自托管无 per-token 费用，上游仅需支付压缩后 0.257× 费用；当自托管成本 <$2.23/M tokens 时优于直传，<$1.94/M 时优于最佳 API 压缩器。

## 相关工作脉络
1. **LLMLingua / Selective Context**：基于 perplexity 信号丢弃低信息 token；Paritok-4B 继承 LLMLingua-2 的"蒸馏到小模型"结构，但单位是带类型的 agent segment 而非散文段落，且目标由显式任务 query 和重要性级别条件化。
2. **LLMLingua-2**：将 GPT-4 蒸馏为 token 分类压缩器；作者说明不做 head-to-head 对比的原因是输入协议不同（LLMLingua-2 无 query、均匀压缩，Paritok-4B 按 segment 类型+意图+级别调用），可比的是端到端 agent 评测。
3. **SWE-bench / SWE-Gym / SWE-rebench**：建立 GitHub issue 解决 benchmark；Paritok-4B 利用这些轨迹源做监督训练，区分于直接使用原始源码的压缩器（agent 以 `cat -n` 读取文件，训练分布必须匹配）。
4. **OpenHands**：开源 agent 框架，提供大规模真实 agent 轨迹；Paritok-4B 的训练数据来自此生态，使 agent-specific 压缩器的监督训练成为可能。
5. **Prompt compression 通用方法**：Paritok-4B 定位在"coding agent 专用压缩"子领域，强调 exact-string-match 安全与意图条件化，区别于 prose 压缩器追求的信息密度优化。

## 局限性与未来方向
- **Drop recall 不足**：drop accuracy 0.49 低于 always-keep 基线，约 3/4 教师丢弃的 segment 未被学生识别；loss 加权方案失败，需 RL/偏好优化 refinement。
- **Level 控制失效**：L0–L3 四级设计在蒸馏目标中坍缩为两波段，student 未获得数值型 budget 输入，无法实现细粒度级别压缩调控。
- **标识符保留是趋势非保证**：约 60% 标识符 token 被有意丢弃，需生产部署时做 presence check 并在失败时 fallback 原始 segment。
- **行重排（line reflow）问题**：多行签名偶尔被折为一行，影响 exact-match edit，需训练集层面约束。
- **Python 偏向**：训练分布以 Python 仓库为主，其他语言未评测。
- **仅监督学习**：v1 继承单一教师策略，未探索双目标（更压缩 vs 保留答案相关内容）的 preference/DPO 优化。
- **单次调用 harness 局限**：仅测量压缩下的理解能力，未覆盖多轮 agent 成本与 exact-match 编辑的实际表现。

## 研究启发与可借鉴点
1. **可复现的蒸馏管线设计**：五阶段漏斗 + 严格验证（长度比约束、must-keep 覆盖率、hallucination 检测）值得借鉴——任何依赖教师蒸馏的场景都可复用此质量保障模式。
2. **"审计而非断言"的工程文化**：对"提取性"、"意图条件化"、"level 设计有效性"三项声称均给出量化审计结果（含负面发现），这种透明报告方式可作为方法论示范。
3. **分布对齐的训练数据构建**：保留 `cat -n` 行号格式的 file_read segment，而非使用原始源码——提示我们在 agent 场景中，训练数据的**呈现格式**与内容同等重要。
4. **Checkpoint 选择超越 loss 曲线**：采用 OOD sweep + 行为测试（level inversion 检测）而非 early stopping，揭示了训练指标与部署指标的分离问题，值得在 SFT 场景中推广。
5. **失败案例公开的价值**：drop 监督加权失败、level 设计未生效等"dead ends"的完整披露，为后续研究者提供了明确避坑指南和 v2 方向。

## 关键术语表
**Extractive compression**：模型从输入中直接选择并复制 span，而非生成重写内容；对代码/工具输出而言确保标识符、路径、行号逐字节保留。
**Intent-conditioned**：压缩决策以 agent 当前任务 query 为条件，保留的 segment 行在意图实体覆盖上显著高于删除行。
**Per-segment decomposition**：将 agent 请求按类型拆分为独立 segment，分别压缩，支持并行、增量更新和单段精确恢复。
**Must-keep span**：每个 segment 中被强制逐字保留的 token 集合（路径、标识符、错误类、行号、代码关键字），由无 LLM 的标注器自动提取。
**Compression rate (CR)**：输出 token 数 ÷ 输入 token 数，macro-average 跨实例平均；越小表示压缩越激进。
**IdentR（Identifier Retention）**：原始 segment 中 must-keep-ish token（路径、结构化标识符、行号、错误类）在压缩输出中的存活比例。
**Drop accuracy / recall**：模型预测 segment 是否保留与教师一致的比例（accuracy）及教师确实丢弃的 segment 中被正确丢弃的比例（recall）。
**Line reflow**：模型将多行代码签名折行为单行的现象，虽可读但不兼容 exact-match edit，需网关层 re-anchoring 缓解。

## 可复现要素
- **数据集**：OpenHands 轨迹来自 SWE-rebench 和 SWE-Gym（67,074 条），SWE-bench Lite 300 instances 作为 held-out 评测集；pipeline 脚本公开。
- **代码/权重**：Apache 2.0 开源，模型权重在 Hugging Face Hub；仓库 https://github.com/Paritok-official/paritok-4b-v1 包含五阶段数据管线、SFT 配置、checkpoint sweep 结果、audit 脚本（`eval/extractiveness.py`、`eval/design_claims_audit.py`）、端到端 SWE-bench Lite harness（`eval_model/`）。
- **关键超参**：LoRA r=32, α=64, dropout=0；学习率 $1 \times 10^{-5}$，linear decay，warmup 0.1；effective batch=32；max_seq_len=16,384；2 epoch；bf16；seed=42。
- **硬件**：1× H100 80GB（Unsloth，无 DeepSpeed）；发布 adapter 264MB，可自托管于单张 24GB GPU。
