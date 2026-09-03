---
title: "Evaluating-Language-Models-on-Cross-Language-Code-Functional"
source: https://arxiv.org/pdf/2608.23961v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 10:42:22"
field: "代码理解与大模型评测"
keywords: ["代码功能等价性", "跨语言代码分析", "LLM评测", "PolyHuman基准", "代码语义理解", "实证软件工程"]
innovations: ["构建PolyHuman人类手写多语言代码基准，首次系统性评测跨语言功能等价判断能力", "提出抽象层次失败分类体系（知识失败/抽象推理失败/实现级错误），揭示LLM语义推理瓶颈", "发现GPT-o4-mini跨运行不稳定性（18%）及difficulty×equivalence非对称脆弱性"]
benchmarks: ["PolyHuman", "EquiBench", "SeqCoBench"]
---

# 论文速读：Evaluating-Language-Models-on-Cross-Language-Code-Functional

## 一句话总结
本文构建了来自竞赛编程平台的人类手写代码数据集 **PolyHuman**（覆盖 CPP、Java、Python），首次系统性评测了 9 个 LLM 在同语言与跨语言场景下的**代码功能等价性判断能力**，并通过对 GPT-o4-mini 的系统性失败案例分析，揭示了当前模型在语义推理层面仍存在显著的局限性。

## 研究问题与动机
1. **现有基准高估 LLM 语义理解能力**：EquiBench、SeqCoBench 等主流评测多依赖合成变换（如变量重命名、结构化改写）生成代码对，等价性可通过表层相似性信号（词法/结构相似度）推断，无法检验真正的语义推理能力。
2. **跨语言等价性评估几乎空白**：现有工作多聚焦同语言场景；跨语言翻译后代码在语法、结构和惯用法上差异显著，浅层相似信号完全失效，但尚无系统性评测验证 LLM 在此场景下的真实水平。
3. **人类手写代码带来更大挑战**：独立编写的人类实现具有更高的结构多样性，与合成代码相比，功能等价性判断难度显著提升，而现有研究缺乏对此场景的量化分析。
4. **模型预测偏差未被充分揭示**：已有研究多依赖聚合准确率，难以发现模型在 Pass vs Pass 与 Pass vs Fail 判断上的系统性非对称偏差，需要更细粒度的分析框架。

## 核心贡献（创新点）
1. **构建 PolyHuman 基准数据集**：从 CodeContests 提取人类手写的 CPP/Java/Python 竞赛代码，构建包含同语言与跨语言的 15 类子任务（各 500 对），覆盖等价与非等价配对；与 EquiBench、SeqCoBench 相比，代码来源真实、结构多样性更高。
2. **首次系统性跨语言功能等价评测**：在统一提示与度量下对 9 个 LLM（含开源与闭源）进行同语言/跨语言等价判断对比，揭示了模型在跨语言场景下比预想更差的性能。
3. **提出抽象层次失败分类体系**：基于 81 例系统性分歧的 CoT 逐案例人工分析，构建了三层次错误分类码本——知识失败（Knowledge Failure）、抽象层次推理失败（Abstraction-Level Reasoning Failure，含过早结论/语义级误解/实现级细节错误），以及 17 例Ground-Truth 标注错误。
4. **揭示模型决策偏差与稳定性问题**：通过决策差（Gap = AccNonEq − AccEq）量化各模型的语言/标签偏向，发现 GPT-o4-mini 存在跨运行不稳定性（18% 不稳定率），说明部分"正确"结果反映的是不稳定能力而非稳健能力。

## 方法详解
- **数据构建（PolyHuman）**：以 CodeContests 为基础，筛选三种语言（CPP、Java、Python）的正确（通过全部测试用例）与错误解；每个问题每种语言随机采样 2 个正确解 + 1 个错误解，组合为 Pass vs Pass（等价）和 Pass vs Fail（非等价）配对，共 5,035 个问题，形成 15 个子任务（每子任务 500 对）。
- **实验设置**：基准数据集 EquiBench（OJ-A：人类独立编写；OJ-V：变量重命名变换）和 SeqCoBench（20 类语义保持/破坏变换）；使用统一零样本提示，要求模型输出 Yes/No，GPT-o4-mini 以外模型 temperature=0。
- **决策差（Gap）度量**：$Gap = Acc_{NonEq} - Acc_{Eq}$，正值表示倾向拒绝等价（保守），负值表示倾向接受等价（宽松）。
- **影响因素分析（RQ₃）**：提取问题难度（A–G 级）、代码长度（LOC）、表层相似信号（Lexsim、CodeBLEU、CodeBERT/GraphCodeBERT/UniXcoder/BGE-Code embedding 相似度）；采用 Spearman 相关系数与二元 logistic 回归建模预测驱动因素。
- **错误分析框架（RQ₄）**：对 GPT-o4-mini 迭代评估 900 实例，取 majority voting，达到主题饱和（连续 16 例错误无新类别）停止；对 81 例系统性分歧使用 CoT prompt 分析推理链；在同一组 81 例上对比 Claude-Opus-4.7 和 Gemini-3-Flash。
- **关键回归公式**：logistic 回归模型以 difficulty、sum_LOC、CodeBLEU、language、ground-truth label 及 difficulty×label 交互项为预测变量，Pseudo $R^2 = 0.429$。

## 实验与结果
**内语言准确率（Table 3）**：GPT-o4-mini 在合成基准 SeqCoBench 达 0.97、EquiBench OJ-V 达 0.91，但在人类手写 PolyHuman 上降至 0.80（Python Pass vs Pass），显示合成→真实代码的性能断崖式下降。多个开源模型（Llama、Mistral、CodeLlama）在 Pass vs Fail 上准确率接近 0，呈现极端非等价偏向。

**跨语言准确率（Table 4）**：GPT-o4-mini 跨语言等价对准确率 0.81–0.85，非等价对 0.87–0.93；Mistral-7B-Instruct-v0.3 在等价对达 1.00 但非等价对仅 0.01–0.02，体现极端过接受偏差。

**决策差（Table 5）**：GPT-o4-mini Gap 在 CPP/Java/Python 分别为 0.072/0.075/0.088，呈保守倾向；Qwen2.5-Coder-14B-Instruct 最均衡（CPP Gap=0.013，Java Gap=0.005）；Mistral-7B-Instruct-v0.3 Gap = −0.988（极端过接受）；CodeLlama-13B Gap = 0.363（CPP，强过拒绝）。

**难度影响（Finding 4）**：GPT-o4-mini 对等价对的 Yes 响应率随难度从 83.0% 降至 54.2%，对非等价对的 Yes 率从 5.5% 升至 33.3%，**难度越高越容易误判非等价对为等价**。

**代码长度影响（Finding 5）**：代码长度对非等价对影响显著——输入规模越大，假等价判断越多；对等价对影响较小。

**相似性信号（Finding 6）**：CodeBLEU 与 UniXcoder 在等价对检测中关联最强（ρ=0.87/0.84），但在非等价对中 CodeBERT（ρ=0.76）和 CodeBLEU（ρ=0.70）显著增加假等价率——相似性是一把"双刃剑"。

**回归分析（Table 6）**：CodeBLEU 独立贡献显著（β=0.212，OR=1.236，p<0.001）；Python 比 CPP 更保守（β=−0.450，OR=0.637，p=0.001）；sum_LOC 无独立显著影响（p=0.353）。

**错误分析（Finding 9）**：81 例分歧中 24 例三模型均错；GPT-o4-mini 失败中 17 例被 Claude-Opus-4.7 单独修正、12 例被 Gemini-3-Flash 单独修正；9 例经 CoT 后修正；共发现 17 例 Ground-Truth 标注问题（占 900 实例的 1.89%）。

## 相关工作脉络
1. **EquiBench（Wei et al., NeurIPS 2025）**：当前功能等价基准，但 OJ-V 子集依赖变量重命名合成，OJ-A 仅含 Python 代码，且不包含跨语言场景与代码专属 LLM。本文在统一设置下扩展了 EquiBench 的覆盖范围。
2. **SeqCoBench（Maveli et al., NAACL 2025）**：基于 20 类语义变换的 Python 基准，强调合成数据特征；本文指出此类变换产生的等价信号可被相似性捷径利用，PolyHuman 通过人类独立实现提供了更严格的测试。
3. **跨语言代码克隆检测（Moumoula et al., FSE 2025）**：关注跨语言克隆识别，但任务定义不同于严格的功能等价；本文首次系统评估跨语言功能等价判断能力。
4. **代码翻译错误分析（Pan et al., ICSE 2024）**：研究 LLM 跨语言翻译引入的 bug，但未直接评估等价性判断；本文填补了这一评测缺口。
5. **SE-Jury 集成策略（Zhou et al., ASE 2025）**：提出 LLM 集合判决机制缩小与人工评估的差距；本文发现各模型偏差互补，支持采用多模型集成策略缓解单一模型的系统性偏差。
6. **代码相似性度量（CodeBERT, GraphCodeBERT, UniXcoder, BGE-Code）**：本文量化了这些嵌入模型与 LLM 预测的相关性，发现结构性/语义性相似信号（CodeBLEU、CodeBERT）比纯词法信号（Lexsim）更易导致假等价判断。

## 局限性与未来方向
1. **数据集局限性**：PolyHuman 来源于竞赛编程平台，代码以算法密集型为主，与工业级真实项目代码存在差异，泛化性受限。
2. **模型覆盖有限**：开源模型主要集中在 3B–20B 参数规模；最大规模闭源模型仅有 GPT-o4-mini 完整评测，其他顶级模型仅在 81 例上补充对比。
3. **训练数据污染风险**：CodeContests 为公开数据集，可能被模型训练数据覆盖，实际性能可能被低估（结论偏保守）。
4. **测试套件覆盖不全**：PolyHuman 以在线裁判通过为等价标准，但测试用例无法覆盖所有边界情况，存在 1.89% 的 ground-truth 标注争议。
5. **手动分类主观性**：81 例错误分析依赖人工编码，虽经第二作者独立验证，但未报告 Cohen's Kappa 等互评一致性指标。
6. **未来方向**：开发更多样化的真实世界程序基准；减少对表层相似信号的依赖；提升推理链稳定性；探索模型集成与多智能体互补策略；改进对复杂控制流中执行路径与变量演化的追踪能力。

## 研究启发与可借鉴点
1. **决策差（Gap）度量可直接迁移**：本文提出的 $Gap = Acc_{NonEq} - Acc_{Eq}$ 可复用于其他二元判断任务（如 bug 检测、漏洞识别），量化模型的假设检验偏向，比单一准确率更能揭示系统性偏差。
2. **困难度分层评测范式值得推广**：本文按竞赛难度 A–G 分层分析性能退化曲线，揭示了"更难问题→更多假等价"的非对称脆弱性；这一分层方法可移植到其他代码理解基准评测中。
3. **CoT 分析+多模型交叉验证的失败分析流程**：对系统性分歧用例使用 CoT 提示提取推理链，并在多模型间对比验证，可有效区分"共性能力瓶颈"与"模型特定缺陷"，为后续针对性改进提供依据。
4. **多模型互补集合策略的实践启示**：Claude-Opus-4.7 在抽象推理上强于 GPT-o4-mini，Gemini-3-Flash 在领域知识上更有优势；在代码等价判断等高可靠性要求的场景中，可采用角色分工的多智能体架构。
5. **相似性信号的双刃剑效应警示**：CodeBLEU/CodeBERT 在高相似度下提升等价检测，但也显著增加非等价对的假阳性率；在设计代码理解评测或训练数据时，需控制相似性混淆因素的分布。

## 关键术语表
**功能等价性（Functional Equivalence）**：两个程序在所有合法输入下产生相同输出的性质，是代码重构、迁移、克隆检测等应用的基础判定标准。

**PolyHuman**：本文构建的新基准数据集，来源于 CodeContests 竞赛编程平台的人类手写代码，覆盖 CPP、Java、Python 三语言，包含同语言和跨语言的等价/非等价代码对。

**决策差（Decision Gap）**：$Gap = Acc_{NonEq} - Acc_{Eq}$，衡量模型在非等价对与等价对检测准确率上的差异；正值表示保守（倾向拒绝等价），负值表示宽松（倾向接受等价）。

**Chain-of-Thought（CoT）提示**：要求模型在给出最终判断前逐步推理的方法，本文用于错误分析以揭示模型的推理链和失败根源。

**抽象层次推理失败（Abstraction-Level Reasoning Failure）**：模型具备相关知识但在跨越不同抽象层级（算法→语义→实现）进行推理时出现断裂，是本文识别的主要失败类型之一。

**实现级细节错误（Implementation-Level Detail Error）**：模型把握了高层意图但未能验证具体执行细节（如索引对应、状态更新、控制流路径、变量演化），是最常见的子类错误。

**CodeBLEU**：结合词法、语法和语义相似度的代码质量评估指标，本文用作衡量代码对表层相似性的量化信号之一。

**主题饱和（Thematic Saturation）**：定性分析中的停止准则，当连续若干案例不再产生新的错误类别时认为分类体系已充分覆盖，本文设定为连续 16 例。

## 可复现要素
- **数据集**：PolyHuman 数据集及复现包已公开（Zenodo DOI: 10.5281/zenodo.21800077）；EquiBench 和 SeqCoBench 为已有开源基准。
- **代码/权重**：模型推理代码与完整编码实例在复现包中提供；使用 GPT-o4-mini、Claude-Opus-4.7、Gemini-3-Flash 等 API 模型及开源 Llama/Mistral/Qwen/CodeLlama 系列模型。
- **关键超参**：temperature=0（确定性解码，GPT-o4-mini 除外）；CoT prompt 使用 JSON schema 约束输出格式；评估子集为 PolyHuman 前 500 实例、SeqCoBench 随机抽样 500 实例、EquiBench 全部 800 实例。
