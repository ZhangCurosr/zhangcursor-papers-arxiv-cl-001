---
title: "Predicting-Program-Exit-Code-with-LLMs-and-Programming-Langu"
source: https://arxiv.org/pdf/2609.00579v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 09:55:37"
field: "程序语义理解与形式化方法"
keywords: ["Program Executability Prediction", "Operational Semantics", "K Framework", "Large Language Models", "Semantic Reasoning", "Code Understanding"]
innovations: ["提出PrEx任务：给定形式化语义规则和程序，预测执行结果及违反的规则，剥离轨迹生成复杂度", "设计KeywordSwap和KeywordObf语义偏移机制，系统判别LLM是遵循给定规则还是依赖预训练先验", "构建含有效/无效程序对的C*基准（2946个程序），支持跨来源和语义偏移的系统性评估"]
benchmarks: ["PrEx", "PLSemanticsBench", "CRUXEval", "REval"]
---

# 论文速读：Predicting-Program-Exit-Code-with-LLMs-and-Programming-Langu

## 一句话总结
论文提出**程序可执行性预测（PrEx）**任务，通过在提示中提供 C* 语言的形式化语义规则，评估 LLM 是依据显式给定的规则进行语义推理，还是依赖预训练先验进行猜测。核心发现：即使任务输出被简化为离散判决，当前 LLM 仍主要依赖预训练模式匹配，在语义偏移和程序复杂度提升时性能急剧下降。

## 研究问题与动机
1. **LLM 是否真正理解程序语义？** LLM 在代码生成、翻译等任务上表现优异，但其能力来源尚不明确——是基于模式匹配/记忆，还是具备真正的语义推理能力？
2. **前置基准（PLSemanticsBench）未能定位能力瓶颈。** 已有工作展示模型在预测变量状态、生成执行轨迹等复杂多步任务上全面失败，但不清楚是任务复杂度（多步推理）导致，还是根本性的语义推理能力缺失。
3. **缺乏判别模型"遵循规则 vs 依赖先验"的评估设计。** 现有基准（如 CRUXEval、REval）不提供显式形式化规则，无法判断模型预测是源于给定规则还是预训练记忆。
4. **需要系统性控制语义理解难度的基准。** 通过语义偏移（改变符号含义）和程序复杂度梯度（三种程序来源）来精确刻画模型能力边界。

## 核心贡献（创新点）
1. **提出 PrEx 任务：给定程序与形式化语义，预测可执行性及违反的规则。** 与已有工作（如 PredEx、PLSemanticsBench）相比，输出为离散判决（##success## / ##error## + 规则编号）而非长执行轨迹，但评估条件更加系统化和多维度。
2. **构建首个同时包含有效/无效程序对的 C* 基准（2946 个程序）。** 基于 PLSemanticsBench 的有效程序，通过五种语义感知变换（break/continue 越界、除零、取模零、未声明变量使用）生成等量无效程序，并区分三种来源（Human-Written / LLM-Translated / Fuzzer-Generated）。
3. **设计 KeywordSwap 和 KeywordObf 两种语义偏移机制。** 前者交换已知操作符语义（如 + 变为减法），后者用新的单 token 符号替换标准符号，迫使模型放弃预训练关联，严格依赖提示中提供的规则。
4. **系统评估两个语义形式化体系（S 小步语义 vs K 框架）下的模型表现。** 揭示规则粒度差异不影响模型的先验依赖行为，同时展示 CoT 对非推理模型有帮助但对推理模型效果有限。

## 方法详解
**任务定义：** 给定 C* 程序 P 及其形式化语义规则集 R，模型需输出：（1）是否可执行（##success## / ##error##）；（2）若不可执行，指出违反的规则编号。

**程序语言 C*：** 类 C 语法的命令式小语言，EBNF 定义包括语句（int 声明、赋值、if-else、while、loop、halt、continue、break）、算术表达式（+, -, *, /, %）、布尔表达式、关系运算等。

**两种语义形式化：**
- **S（小步操作语义）：** 细粒度推理规则，每条规则对应一个原子计算步骤。使用 Gentzen 风格推理记号，前提在横线上方，结论在下方。例如变量查找（Rule 1/2）、整数除法（Rule 18/19）。
- **K 框架：** 粗粒度重写规则，利用内置重写处理中间表达式归约。将多个 S 步骤封装为单一重写规则，如 Rule 3 直接表达 I1 + I2 的求值。

**语义偏移设计：**
- **KeywordSwap：** 交换操作符语义映射，如 +↔−、×↔÷、<↔> 等，模型需抑制预训练习惯，严格依从提示中的规则表。
- **KeywordObf：** 用全新单 token 符号替换所有标准操作符和关键字（如 ★ 表示加法、▲ 表示乘法），彻底消除预训练中的符号关联。

**数据集构造：** 基于 491 个有效 C* 程序，每个程序生成 5 个无效变体（每类错误各一个），共 2946 个程序。变换通过 ANTLR 解析器+访问者模式实现，确保每段无效程序仅违反一个规则。

**实验配置：** 模型 × 形式化（K/S）× 语义偏移（Standard/KeywordSwap/KeywordObf）× 提示方式（non-CoT / CoT），每个配置运行 3 次取平均准确率。

## 实验与结果
**数据集统计：** Human-Written（162 程序，中位 19 行）、LLM-Translated（165 程序，中位 106 行）、Fuzzer-Generated（164 程序，中位 786 行），复杂度递增。

**评估模型：** Qwen2.5-Coder（3B/7B/14B/32B，含 CoT）、DeepSeek-Qwen（14B/32B）、Ministral 3（3B/8B/14B，含 CoT），以及随机基线（18%）。

**主要结果（Standard 语义下 K 形式化）：**
- 最强模型 DeepSeek-Qwen 32B 在 Human-Written 上达 99%，Qwen2.5-Coder 32B-CoT 达 99%，Ministral 3 14B-CoT 达 99%。
- 但所有模型在 LLM-Translated 上平均下降 2.9–8.7pp，在 Fuzzer-Generated 上平均下降 13–33pp。
- 最大单配置下降：Qwen2.5-Coder 14B-CoT 在 K+KeywordObf 下从 78% 跌至 23%（−55pp）。

**语义偏移影响（Human-Written，K 形式化）：**
- KeywordSwap 中位下降 19pp，KeywordObf 中位下降 32pp，后者更难。
- DeepSeek-Qwen 32B 在 K+KeywordObf 下仅降 2pp（99→98），是 outlier；其他模型普遍下降显著（如 Ministral 3 14B：90→32，−58pp）。

**错误类型分析（Figure 5 雷达图）：**
- 关键字相关错误（continue-outside-loop、break-outside-loop）在 KeywordObf 下退化最严重。
- divide-by-zero 在 KeywordSwap 下下降明显，反映预训练对除法的强先验。
- Fuzzer-Generated 上所有错误类型精度普遍大幅收缩。

**定性分析（Table 9）：** 即使是 5 行短程序，最强模型也可能在规则编号上出错（如 Rule 73 vs Rule 76），说明模型能识别错误类别但难以精确定位到具体规则。

**失败模式：**
- Human-Written：主要是 false success（误判无效为有效）和 wrong-rule（规则编号混淆）。
- LLM-Translated：false success 极为频繁（模型倾向于将翻译代码视为可执行）。
- Fuzzer-Generated：三类错误（false success、wrong rule、false error）均匀增加，malformed output 也更多。

## 相关工作脉络
1. **PLSemanticsBench（Thimmaiah et al., 2026）：** 同团队提出的前身基准，评估 PredState/PredRule/PredTrace 三项复杂语义推理任务。本文 PrEx 将其简化为更基础的"可执行性判决"，以分离复杂度干扰。
2. **CRUXEval（Gu et al., 2024）：** Python 程序输入-输出配对推理基准。本文区别在于提供显式形式化规则并引入无效程序对和语义偏移，以诊断规则遵循能力。
3. **REval（Chen et al., 2025）：** 评估代码执行的多步推理一致性（覆盖率、状态、路径、输出）。PrEx 是其更简单的变体，但增加了语义偏移控制和形式化规则显式输入。
4. **PredEx（Li et al., 2025）：** 结合程序分析与 LLM 提示预测 Python 执行轨迹。本文不生成轨迹，仅做判决，但通过语义偏移设计更直接地检验规则遵循 vs 先验依赖。
5. **CodeFlow（Le et al., 2025）：** 学习 CFG 预测代码覆盖率和运行时错误定位。属于"辅助工具型"方法，本文聚焦纯 LLM 内在语义理解能力。
6. **符号命名与语义推理研究（Wang et al., 2024；Sultan et al., 2026）：** 表明 LLM 对标识符语义高度依赖。本文 KeywordSwap/KeywordObf 受此启发，进一步系统化了符号扰动对语义推理的影响。

## 局限性与未来方向
1. **语言与错误类型受限：** 仅测试 C* 小语言和 5 种错误，尚未扩展到真实语言（如 C/Python/Java）或更丰富的运行时错误（如空指针、类型不匹配）。
2. **形式化规则的呈现方式固定：** 规则以文本形式嵌入 prompt，实际编译器/验证器中规则可能以结构化形式存储，LLM 如何与非文本格式交互待探索。
3. **未探索微调/蒸馏路径：** 仅测试 zero-shot/few-shot 提示，未研究通过语义推理数据微调能否改善规则遵循能力。
4. **Fuzzer 生成程序的多样性有限：** 深度控制的 fuzzer 虽保证语义合法性，但可能无法覆盖真实代码中的全部结构复杂性。
5. **CoT 对推理模型的增益有限：** 推理模型（如 DeepSeek-Qwen 32B）本身自带 CoT，额外 non-CoT 条件未体现增量价值，未来需探索更有效的推理引导策略。

## 研究启发与可借鉴点
1. **语义偏移设计可迁移至其他代码理解任务。** KeywordSwap/KeywordObf 范式可用于评估 LLM 在代码翻译、bug 检测、代码摘要等任务中是否真正依赖语义还是表层模式。
2. **PrEx 可作为训练数据生成器。** 其无效程序对构造流程（语义感知变换 + 规则标注）可扩展到其他语言，生成高质量的语义推理训练样本。
3. **与静态分析工具结合的创新机会。** 可将 PrEx 判决结果作为编译时检查的补充，或在 LLM 生成的代码修复中引入形式化规则约束，减少 false success 率。
4. **Prompt 工程中形式化规则的呈现方式值得优化。** 当前 S/K 规则以文本嵌入 prompt，可能存在 token 效率低、模型难以精确定位规则的问题，可探索结构化表示（如 JSON/YAML）或规则检索增强。
5. **与可验证 AI 方向的结合：** PrEx 的发现（模型依赖先验而非遵循规则）支持"需要形式化验证辅助 LLM 代码输出"的研究路线，可作为该路线的实证论据。

## 关键术语表
**PrEx（Program Executability Prediction）：** 本文提出的新任务，要求 LLM 在给定程序与形式化语义的前提下，判断程序是否可执行并指出违反的规则。

**S-semantics（Small-step Operational Semantics）：** 小步操作语义，以细粒度推理规则描述程序执行，每条规则对应一个原子计算步骤。

**K-framework（K 语义框架）：** 基于rewriting的粗粒度语义形式化，将多个小步计算封装为单一重写规则，利用内置重写机制处理中间表达式归约。

**KeywordSwap：** 语义偏移技术，交换已知操作符/关键字的语义映射（如 + 变为减法），测试模型能否抑制预训练先验。

**KeywordObf：** 语义偏移技术，用全新的单 token 符号替换所有标准操作符和关键字，彻底消除预训练符号关联。

**False Success：** 错误地将语义无效程序判定为可执行（##success##），是模型依赖预训练模式而非形式化规则的主要失败模式。

**Wrong Rule：** 正确判断程序无效但引用了错误的规则编号，反映模型能识别错误类别但难以精确匹配到具体语义规则。

**C*：** 本文使用的类 C 小型命令式语言，具有完整的 EBNF 语法定义，用于控制实验语言复杂度。

## 可复现要素
- **数据集：** PrEx 数据集已公开，仓库地址 https://github.com/EngineeringSoftware/prex
- **代码/工具：** 转换流程使用 ANTLR 解析器+访问者模式，具体实现见 GitHub 仓库；程序来源（Human-Written 来自 LeetCode/HumanEval/CodeContests/MBPP，LLM-Translated 来自 CodeForces via Qwen2.5-Coder 32B 翻译，Fuzzer-Generated 使用深度控制 grammar-based fuzzer）均可复现
- **评估模型：** Qwen2.5-Coder（3B/7B/14B/32B）、DeepSeek-Qwen（14B/32B）、Ministral 3（3B/8B/14B），均为开源模型
- **关键超参：** 每个配置运行 3 次取平均；CoT 提示仅对非推理模型额外添加"请逐步推理"指令
- **硬件：** 使用 TACC 和 Cisco AMD AI & HPC Cluster
