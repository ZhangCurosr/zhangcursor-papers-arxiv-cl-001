---
title: "ClueWeaver-Reward-Guided-Dual-Agent-Evidence-Reasoning-for-C"
source: https://arxiv.org/pdf/2608.25531v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 21:31:06"
field: "长上下文推理与证据选择"
keywords: ["long-context question answering", "evidence selection", "dual-agent reasoning", "reward-guided RL", "compact LLMs", "literary narrative"]
innovations: ["将证据选择与解释解耦为 Finder/Interpreter 双代理流水线并显式保留段落 ID 引用", "对两代理分别采用基于规则的奖励引导 GRPO 训练以对齐最终任务表现"]
benchmarks: ["DetectiveQA", "∞Bench", "LongBench v2", "NoCha"]
---

# 论文速读：ClueWeaver-Reward-Guided-Dual-Agent-Evidence-Reasoning-for-C

## 一句话总结
论文提出 ClueWeaver，一个面向紧凑型本地模型的证据感知双代理框架，通过 Finder 选择含关键线索的段落并保留段落 ID，再由 Interpreter 生成带段落引用的解释和答案，辅以自校准机制；两者均通过奖励引导的 GRPO 强化学习优化，显著提升本地模型在文学长叙事问答任务上的性能。

## 研究问题与动机
- 人文社科研究常需对小说、剧本、档案等长叙事材料进行细读，但受限于成本，研究者多依赖闭源长上下文模型，缺乏可本地部署的经济方案。
- 紧凑型本地模型直接接收完整长上下文时，面临截断证据、遗漏稀疏线索、无法追溯来源等问题；仅靠增大上下文窗口并不能保证可靠理解。
- 现有 RAG 及多步推理方法主要针对开放域检索或多跳 QA，未直接解决"源文本已给定、答案线索稀疏分布、证据选择需显式可审查"的封闭长叙事场景。
- 全文提示让证据选择隐式化，难以定位失败源于线索遗漏还是推理错误，缺乏对证据选择与解释过程的分离控制。

## 核心贡献（创新点）
- 提出 ClueWeaver 双代理流水线，将长叙事问答解耦为显式证据选择、自校准解释和证据 grounding 三步，适用于紧凑型本地模型。
- 用奖励引导的 GRPO 分别优化 Finder 与 Interpreter，Finder 奖励强调高召回的证据保留与段落 ID 引用准确性，Interpreter 奖励强调答案正确性、证据 grounding 与简洁解释。
- 在 DetectiveQA、∞Bench、LongBench v2、NoCha 四个长上下文叙事基准上验证，ClueWeaver 以 Qwen3-4B 为底座取得 59.0% 的整体准确率，超越最强本地基线（IRCoT，52.6%）6.4 个百分点，并接近 API 级大模型性能。

## 方法详解
- 整体流程：输入长叙事 X（含 m 个段落）和问题 q，先用检索感知分割构建候选段落序列 S，Finder 对每个片段预测是否保留（YES/NO）、引用哪些段落 ID、给出简短理由；被保留的片段按原始段落顺序打包成证据包 E（受证据预算 B 约束），交给 Interpreter 生成最终答案和带段落引用的解释；对高风险问题（二元判断、含否定/因果/例外等措辞），Interpreter 会运行内部自校准步骤再次校验。
- 两个代理均采用 XML 结构化输出：`<reason>` 包含段落引用的解释，`<answer>` 包含最终答案或 YES/NO 判断。
- Finder 训练目标（公式 5）：$R_F = \lambda_{fmt}R_{fmt} + \lambda_{dec}R_{dec} + \lambda_{cite}R_{cite} + \lambda_{comp}R_{comp} + \lambda_{neg}R_{neg}$，其中奖励偏好保留含答案线索的片段、忠实引用段落 ID、紧凑且不含未支持的段落号；未命中的正样本仅获得格式奖励。
- Interpreter 训练目标（公式 7）：$R_I = \lambda_{fmt}R_{fmt} + \lambda_{ans}R_{ans} + \lambda_{cite}R_{cite} + \lambda_{ground}R_{ground} - \lambda_{hall}R_{hall}$，奖励答案正确性、段落 ID 与证据集 H 的重合、简洁 grounded 解释，并对不支持的推测/无效段落引用施加惩罚。
- 优化器：两代理均使用 GRPO（Group Relative Policy Optimization），从旧策略采样 K=8 个输出，计算组内归一化优势（公式 1-3），以 clip PPO 形式更新策略参数，KL 正则系数 β=0.04，禁用模型内部思维链。
- 检索辅助分段：使用 BGE-M3 密集检索与 BM25  lexical 匹配找出与问题相关的锚段落，并在其附近扩展局部窗口构成候选段，保留段落 ID 供后续溯源。

## 实验与结果
- 数据集：DetectiveQA（侦探情节稀疏线索）、∞Bench、LongBench v2（长上下文推理）、NoCha（小说级主张验证）。
- 基线：本地端到端读者（Qwen3-4B/8B、Ministral-3-14B、GPT-OSS-20B、Qwen3-30B-A3B、Gemma-4-31B-it）、API 大模型（Claude Haiku 4.5、GPT-5 nano）、代理管道基线（ReAct、IRCoT、Self-Ask、Chain-of-Agents、RAG-DDR，均使用 BGE-M3）。
- 主要结果：ClueWeaver 在四个基准上总体准确率为 59.0%，在各数据集上均领先所有本地方法；相对最强本地基线 IRCoT（52.6%）提升 +6.4 点；相对同底座 Qwen3-4B 端到端读者（44.5%）提升 +14.5 点；仍弱于最佳 API 模型（约 63.9%）4.9 点，但在 LongBench v2 上反超 +11.5 点。
- 消融结论：关闭 Interpreter<sub>self-cal</sub> 导致准确率下降 4.8 点；移除 Finder 直接传入全部检索段落下降 5.8 点；关闭 Finder RL 最严重，下降 6.8 点至 49.0%，说明未经训练的 Finder 反而丢弃有用证据；关闭 Interpreter RL 下降 1.0 点。
- 效率与成本：在单 GPU 上使用 Qwen3-4B，ClueWeaver 单题耗时 8.6–9.8 秒（主要开销在 Finder），对比直接读取的 2.8 秒；仅需 24GB 显存（RTX 4090/5090），远低于 30B 级本地模型的 80GB。
- 可追溯性审计：在 310 个测试问题上，275 条输出含段落引用，其中 685/690（99.3%）引用的段落 ID 有效，98.2% 的输出仅含合法引用。

## 相关工作脉络
- NarrativeQA、QuALITY、DetectiveQA、NoCha 等长叙事 QA 基准强调情节、人物、时间顺序与隐含因果推理，本文在相同任务设定上进一步要求显式证据选择与段落溯源，弥补既往评测只关注最终答案的不足。
- Long-Bench、∞Bench、LongBench v2 证明单纯扩展上下文窗口不能保证长程理解鲁棒性，本文通过检索引导的分段与证据打包缓解该问题。
- RAG、Self-RAG、RAG-DDR 等开放域检索增强方法侧重于跨文档检索，而本文聚焦封闭文档内稀疏线索的保留与引用。
- IRCoT、Self-Ask、Chain-of-Agents、ReAct 等多步推理/代理方法针对通用多跳 QA，本文将其适配到封闭长叙事场景，并通过独立训练 Finder 强化证据选择环节。
- DeepSeekMath 等证明 GRPO 可提升推理能力，本文首次将 reward-guided RL 同时作用于证据选择与解释两个环节，强调段落级引用忠实度。
- 与端到端全文提示相比，本文通过双代理解耦使证据选择成为可观察、可训练、可校准的显式步骤，提升失败可定位性。

## 局限性与未来方向
- 当前训练数据以侦探小说域为主，LongBench v2 等其它领域覆盖不足，可能影响跨域泛化。
- 面对依赖远距离多跳线索或表面重叠较弱的段落时，性能仍受限，检索对远距离线索的发现能力有待增强。
- Finder 在高召回与避免全 YES 退化之间的权衡依赖奖励权重设计，可能存在对某些难例过度保留的问题。
- 自校准仅在同一 Interpreter 内二次检查，未引入外部验证或多轮反思机制。
- 未来方向包括更强远的线索检索、更鲁棒的多跳证据整合、以及在更多样的人文社科长文本上的验证。

## 研究启发与可借鉴点
- 将证据选择与答案生成解耦为独立代理，并通过显式段落 ID 引用实现可追溯推理，该方法论可迁移到其他需要证据支撑的封闭文档 QA 任务。
- 用规则型奖励（format、decision、citation F1、compactness、neg quality）而非学习型 reward model 进行 GRPO 训练，降低了对额外模型标注的依赖，适合资源受限场景。
- 检索仅用于引导分段边界而非替代阅读，兼顾了候选覆盖与原始叙事顺序保留，这种"检索辅助分段 + 模型精读"的范式值得借鉴。
- 自校准步骤不引入新代理，而是复用 Interpreter 对同一证据包进行二次校验，既提高高风险判断的可靠性，又控制推理成本。
- 训练数据中 hard negative 占比约 50%、Interpreter 训练中 27.5% 为基座模型易错题，说明难度聚焦与负样本平衡对收敛关键。

## 关键术语表
- **ClueWeaver**：一种面向紧凑型本地模型的双代理证据感知流水线，用于长叙事问答与主张验证。
- **Finder**：第一个代理，负责在分段层面判断是否保留含答案线索的段落，并引用具体段落 ID。
- **Interpreter**：第二个代理，基于 Finder 提供的证据包生成带段落引用的解释和最终答案。
- **Interpreter<sub>self-cal</sub>**：Interpreter 的内部自校准模式，在高风险问题中对已有答案进行二次校验。
- **GRPO（Group Relative Policy Optimization）**：一种基于组内相对优势估计的强化学习算法，本文用于两代理的奖励引导训练。
- **BGE-M3**：多语言、多功能、多粒度文本嵌入模型，用于密集检索与段落锚点选择。
- **DetectiveQA / ∞Bench / LongBench v2 / NoCha**：四个用于评估长上下文叙事理解与主张验证的基准数据集。
- **段落 ID 引用**：在推理过程中显式标注所依据的原始段落编号，用于提升答案的可追溯性与可审查性。

## 可复现要素
- 数据集：DetectiveQA、∞Bench、LongBench v2、NoCha，论文声明在各自基准的训练集上训练、在统一测试集上评估（具体公开状态论文未详述，通常上述基准可公开获取）。
- 代码/权重：论文未明确声明开源仓库与模型权重发布地址，仅提到附录提供 prompt 模板、实现细节与更多案例。
- 关键超参：base model 为 Qwen3-4B-Instruct；GRPO group size K=8；KL 系数 β=0.04；lr(Finder)=5e-7、lr(Interpreter)=3e-7；batch size 8，梯度累积 Finder=8/Interpreter=2；temperature=1.0、top-p=0.9、top-k=20；max_input_len=9216、max_output_len=256；训练样本 Finder=12K（150 step）、Interpreter=1.0K（100 step）；硬件为单节点 8×A100。
- 实现细节：检索使用 BGE-M3，推理时使用 vLLM 加速 rollout；证据打包参数因数据集不同（如 DetectiveQA 取 N_E=10, P_r=4, P_w=6, B_c=15000）。
