---
title: "Predicting-Program-Exit-Code-with-LLMs-and-Programming-Langu"
source: https://arxiv.org/pdf/2609.00579v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 09:55:47"
field: "代码语义理解与大语言模型评估"
keywords: ["Program Executability Prediction", "Operational Semantics", "K Framework", "Large Language Models", "Code Understanding", "Semantic Reasoning"]
innovations: ["提出PrEx任务，以离散可执行性判断和规则引用评估LLM的语义推理能力", "设计KeywordSwap/KeywordObf语义偏移，区分模型对显式规则的应用与预训练先验依赖", "构建含五种语义感知无效变换的C*程序数据集，覆盖三种程序来源（人类编写/LLM翻译/Fuzzer生成）"]
benchmarks: ["PrEx Dataset", "PLSemanticsBench"]
---

# 论文速读：Predicting-Program-Exit-Code-with-LLMs-and-Programming-Langu

## 一句话总结
论文提出了程序可执行性预测（PrEx）任务，要求LLM在给定程序及其形式化语义（S或K）的情况下，判断程序是否可执行（成功或语义错误），并在出错时指出违反的具体规则，以此评估LLM是遵循显式提供的形式化规则还是依赖预训练先验。

## 研究问题与动机
- LLM在代码生成、翻译等软件工程任务中表现出色，但其在程序语义推理方面的真实能力仍不明确，可能更多依赖模式匹配而非真正的执行模拟。
- 已有基准PLSemanticsBench中的任务（状态预测、规则预测、执行轨迹生成）输出复杂，模型普遍得分极低，无法区分失败是源于多步推理的复杂度还是语义推理的根本缺陷。
- 需要设计一个更基础的基线任务，将语义判断与长轨迹生成解耦，专门考察模型是否能系统性地将给定规则应用于程序。
- 通过引入语义偏移（KeywordSwap和KeywordObf）来强制模型覆盖预训练符号关联，从而检测模型是否真正依赖提供的规则。

## 核心贡献（创新点）
1. **提出PrEx任务**：以程序可执行性预测作为更基础的语义评估任务，要求模型仅输出离散的可执行判断及违反的规则编号，简化了输出空间但仍保留语义推理要求。与PLSemanticsBench相比，PrEx剥离了长轨迹生成的复杂度，聚焦于语义判断能力。
2. **构建含系统性无效变换的C*数据集**：基于PLSemanticsBench的有效程序，通过五种语义感知变换（break/continue越界、除零、模零、变量未声明即使用）生成匹配的无效程序，每个有效程序对应五个无效变体，确保数据集同时覆盖可执行和不可执行情况。
3. **设计两种语义偏移（KeywordSwap/KeywordObf）**：通过交换运算符含义或用全新单token符号替换标准运算符，测试模型能否强制遵循显式规则而非依赖预训练符号关联，为诊断"记忆vs推理"提供了受控实验条件。
4. **跨形式化与多来源的综合评测**：在同一任务上同时评估S（细粒度）和K（粗粒度）两种语义形式化，以及人类编写、LLM翻译、Fuzzer生成三类程序来源，揭示了模型在不同复杂度下的系统性差距。

## 方法详解
- **任务定义**：给定C*程序、EBNF语法和语义规则（S或K），模型需输出`##success##`（程序语义合法）或`##error##`（程序语义非法）并在非法时注明违反的规则编号。
- **语义形式化**：
  - **S（小步操作语义）**：每条规则代表一个原子计算步骤，采用Gentzen风格推理表示法，前提和边条件写在横线上方，结论在下方。例如Rule 1/2处理变量查找（已声明则返回绑定值，否则报错），Rule 18/19处理除法（除数非零则整除，为零则错误）。
  - **K框架（K-semantics）**：基于重写的粗粒度形式化，通过内置重写机制隐式处理中间表达式归约，规则块更大。例如Rule 3将加法表达式的求值封装为单个重写步骤。
- **语义偏移设计**：
  - **KeywordSwap**：交换运算符的语义含义（如`+`变为减法），模型必须覆盖预训练关联，严格依据提供的规则推理。
  - **KeywordObf**：用新型单token符号替换所有标准运算符和关键字（如`★`表示加法），消除预训练中的符号识别线索，每个新符号均为单一token以避免膨胀prompt长度。
- **提示结构**：为每个程序提供C*语法（EBNF）、语义规则（S或K）和程序本身，要求模型根据规则判断可执行性。对非推理模型同时评估直接回答与CoT两种模式。
- **数据集构建**：从491个有效C*程序出发，对每个程序应用五种语义感知变换各生成一个无效程序，共2455个无效程序，总计2946个程序（Table 5显示每类错误类型各占16.7%）。

## 实验与结果
- **数据集**：共2946个C*程序，分为三组：Human-Written（162个，中位数19行）、LLM-Translated（165个，中位数106行）、Fuzzer-Generated（164个，中位数786行）。每种错误类型（break/continue越界、除零、模零、变量未声明）各491个无效程序。
- **评测模型**：Qwen2.5-Coder系列（3B/7B/14B/32B）、DeepSeek-Qwen（14B/32B）、Ministral 3（3B/8B/14B），均测试CoT与非CoT两种模式。
- **最强结果**：DeepSeek-Qwen 32B在Human-Written + K语义 + Standard条件下达到99%准确率；Qwen2.5-Coder 32B-CoT在相同条件下亦达99%。
- **主要下降**：
  - Human-Written → LLM-Translated：最佳模型平均下降2.9–8.7pp；
  - Human-Written → Fuzzer-Generated：最佳模型平均下降13–33pp，最大单配置下降达55pp（Qwen2.5-Coder 14B-CoT在K/KeywordObf条件下从78%跌至23%）。
- **语义偏移影响**：在Human-Written上，KeywordSwap中位数下降19pp，KeywordObf中位数下降32pp；KeywordObf整体更难。DeepSeek-Qwen 32B在K/KeywordObf条件下几乎不受影响（99%→98%，仅降1pp），是少数例外。
- **错误类型差异**：keyword-dependent错误（continue-outside-loop、break-outside-loop）在KeywordObf下退化更严重；arithmetic错误（divide/modulo by zero）相对稳定。
- **失败模式**：Human-Written上主要是wrong-rule混淆（如Rule 73 vs 76）；LLM-Translated和Fuzzer-Generated上false success显著增加，模型倾向于将无效程序判定为合法。

## 相关工作脉络
- **PLSemanticsBench [39]**：同类工作，包含PredState/PredRule/PredTrace三个任务，但输出复杂（长执行轨迹），模型全面低分。本文PrEx是其简化版，剥离轨迹生成，聚焦语义判断，能更精确地定位能力边界。
- **CRUXEval [10] & REval [5]**：以Python程序为主，考察输入输出推理或中间运行时行为，但未在prompt中提供显式形式化语义。本文相比这些工作在评测维度上更聚焦于"遵循给定规则"这一能力。
- **PredEx [20] / CodeFlow [17]**：结合程序分析与LLM预测静态运行时错误或覆盖率，依赖控制流图或执行轨迹学习，而非显式规则应用。本文方法与之互补，更关注形式化语义的理解与遵守。
- **Wang et al. [41]**：证明变量命名影响CodeBERT性能，揭示LLM依赖标识符语义而非逻辑。本文KeywordSwap/KeywordObf设计受此启发，进一步测试模型在符号含义被篡改时的规则遵循能力。
- **NExT [30]**：通过CoT训练让LLM学习从执行轨迹推理，改进了MBPP/HumanEval上的bug修复。本文指出即使有CoT辅助，模型在语义偏移下仍大量依赖先验，提示单纯训练轨迹推理不足以解决问题。

## 局限性与未来方向
- **仅评估单一规则违规**：当前数据集每个无效程序只引入一种错误，实际程序中可能出现多种语义违规交互，未来可扩展至多规则违规场景。
- **C*语言规模有限**：程序最长约2000行（Fuzzer-Generated），与工业级代码库差距较大，语义推理能力在大规模嵌套结构中的表现尚待验证。
- **仅覆盖五种特定错误类型**：break/continue越界、除零、模零、变量未声明，未涵盖类型不匹配、空指针解引用等其他常见语义错误。
- **未探索微调/指令微调效果**：仅通过prompt和CoT评估，未来可通过SFT在PrEx数据上微调，观察模型是否能学会遵循形式化规则。
- **语义偏移仅针对运算符**：未测试控制流关键字（if/while/loop等）的语义偏移效果，KeywordSwap中对这些关键字的处理仍为标准语义。

## 研究启发与可借鉴点
1. **显式规则提供+语义偏移**的设计可作为通用诊断工具，用于评估任何代码理解模型是否真正"理解"而非"记忆"，可迁移至代码翻译、代码补全等任务的评估中。
2. **CoT对语义推理的帮助有限**：在Semantic Shift条件下，CoT并未显著改善模型表现（部分模型甚至下降），提示在涉及规则应用的任务中，单纯推理链不足以克服预训练偏差，可能需要结合约束解码或符号执行引导。
3. **数据集构建方法论可复用**：通过语义感知变换从有效程序批量生成无效程序的设计，可推广至其他语言（如Python、Java）的语义基准构建，保证有效/无效样本的结构对称性。
4. **错误类型的雷达图分析**：Figure 5中按错误类型分面的可视化方式，能清晰揭示模型在特定语义概念上的系统性弱点，可作为后续研究的标准化评估报告模板。
5. **与形式化方法结合的机会**：PrEx任务可被视为LLM与形式化验证的接口——若模型能可靠预测程序可执行性，则可集成到自动bug检测或程序合成流程中，作为轻量级语义检查器。

## 关键术语表
**PrEx（Program Executability Prediction）**：本文提出的任务，要求LLM在给定程序和形式化语义的情况下，判断程序执行是否成功或违反哪条规则。
**S（Small-step Operational Semantics）**：细粒度的操作语义形式化，每条规则代表一个原子计算步骤，采用Gentzen风格推理表示法。
**K（K-Framework Semantics）**：基于重写的粗粒度操作语义形式化，通过内置重写机制处理中间表达式归约，规则块更大。
**KeywordSwap**：语义偏移策略，交换标准运算符的含义（如`+`变为减法），迫使模型覆盖预训练关联并遵循提供的新规则。
**KeywordObf**：语义偏移策略，用全新单token符号替换所有标准运算符和关键字，消除模型对预训练符号的依赖。
**False Success**：模型将语义无效程序误判为有效的预测错误，在复杂程序上为主要失败模式。
**Wrong Rule**：模型正确判断程序无效，但引用了错误的规则编号，常发生在同一错误家族内的相邻规则混淆。
**PLSemanticsBench**：前作提出的语义理解评估框架，包含PredState/PredRule/PredTrace三个任务，本文在其C*程序基础上扩展。

## 可复现要素
- **数据集**：PrEx数据集已公开，地址为https://github.com/EngineeringSoftware/prex。
- **代码**：论文未提及代码仓库，但数据集已开源。
- **权重**：评估模型为开源的Qwen2.5-Coder、DeepSeek-Qwen、Ministral 3系列，可直接从官方获取。
- **关键超参**：论文未详细报告超参数（如temperature、top_p、max tokens），仅说明每个模型在每个配置下运行三次取平均值。
- **基线**：Random（随机猜测）作为下限基线，accuracy约16–18%。
- **程序长度统计**：Table 4提供了三组程序的LOC和Token数量分布（GPT-4o tokenizer）。
