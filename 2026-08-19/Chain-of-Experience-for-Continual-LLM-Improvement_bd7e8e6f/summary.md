---
title: "Chain-of-Experience-for-Continual-LLM-Improvement"
source: https://arxiv.org/pdf/2608.18027v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 11:29:07"
field: "大语言模型推理与测试时优化"
keywords: ["Chain-of-Experience", "Test-time Learning", "LLM Reasoning", "Feedback-driven Iteration", "Test-time Scaling", "Self-Refinement"]
innovations: ["提出CoE统一框架，将多轮迭代求解形式化为顺序决策过程并系统比较四类反馈信号", "证明自反馈可在降低19% API成本的同时平均提升5.6%性能，打破多轮迭代必高耗能的认知", "发现双反馈通道互补性与基座能力-学习增益正相关（Pearson r=0.50）"]
benchmarks: ["AIME 2025", "OmniMath", "LiveCodeBench (V6)", "LiveBench (Code)", "GPQA Diamond", "EvaLearn"]
---

# 论文速读：Chain-of-Experience-for-Continual-LLM-Improvement

## 一句话总结
本文提出 Chain-of-Experience (CoE) 框架，使大语言模型在推理阶段通过多轮迭代与多样化反馈（自评、执行器、正确性信号）持续积累经验并自我改进，填补了传统 LLM 评估忽视测试时交互学习能力的空白。

## 研究问题与动机
1. **固定状态部署的局限**：现代 LLM 训练完成后状态冻结，每次推理被视为孤立事件，无法像人类一样从成功/失败经验中持续演化。
2. **现有测试时方法的碎片化**：多数投票、Self-Refine、Reflexion 等方法产生的经验多为一次性或浅层自修正，未形成闭环的迭代学习机制。
3. **跨任务经验利用在先进模型上失效**：ICL、DC、ACE 等依赖外部记忆检索的策略在面对 GPT-5、o3 等新一代推理模型时，平均性能反而低于基础零样本设定。
4. **缺乏统一的反馈谱系统计研究**：不同来源与强度的反馈如何驱动测试时持续改进尚未被系统化量化对比。

## 核心贡献（创新点）
1. **提出 CoE 统一测试时迭代学习框架**：将多轮求解形式化为顺序决策过程，首次系统比较无反馈、执行反馈、模型反馈与正确性反馈四类信号对连续改进的影响差异。
2. **证明反馈驱动的效率优势显著**：仅使用模型自反馈即可在六大基准上平均提升 5.6% 且 API 成本降低 19%，并以更低的 token 消耗实现更高的准确率/token，打破了“多轮必然高耗能”的刻板印象。
3. **揭示双反馈通道的互补机制**：模型自评信号与正确性/执行器信号结合可覆盖不同改进维度（如逻辑纠偏与格式对齐），在中等难度任务上产生明确协同增益。
4. **发现基座能力与迭代学习增益的正相关性**：模型初始零样本性能与其通过 CoE 获得的提升幅度呈显著正相关（平均 Pearson r = 0.50），说明测试时经验吸收是一种随模型容量涌现的能力。

## 方法详解
- **序列决策形式化**：将单轮生成 $A \sim P(A \mid Q)$ 扩展为多轮迭代过程 $a_t \sim P(a_t \mid Q, e_0, e_1, \dots, e_{t-1})$，其中第 $i$ 轮经验 $e_i = (a_i, f_i)$ 由模型回答 $a_i$ 与环境反馈 $f_i$ 构成。
- **反馈谱系设计**：
  - **No feedback**：$f_i = \emptyset$，改进完全依赖模型对历史输出的内部反思。
  - **Execution feedback**：针对代码/交互任务，$f_i = \mathcal{E}(Q, a_i)$，由解释器或单元测试返回运行痕迹、错误日志或通过率。
  - **Model feedback**：引入辅助裁判模型 $\mathcal{M}_{\mathrm{fb}}$，$f_i = \mathcal{M}_{\mathrm{fb}}(Q, a_i)$，输出文本批评、偏好评分或结构化评估。
  - **Correctness feedback**：提供二值 Oracle 信号 $f_i = \mathbf{1}\{a_i \text{ is correct}\}$，作为性能上限参考。
- **迭代机制**：每轮将完整历史轨迹 $(a_0, f_0, \dots, a_{t-1}, f_{t-1})$ 拼接至 prompt，模型重新生成回答，最多迭代 20 轮；全程不更新模型参数，纯依赖上下文经验复用。

## 实验与结果
- **数据集**：AIME 2025、OmniMath、LiveCodeBench (V6)、LiveBench (Code)、EvaLearn、GPQA Diamond。
- **基线**：ICL、Dynamic CheatSheet (DC)、Agentic Context Engineering (ACE)、不同推理力度（Reasoning-high/low）、无反馈 CoE。
- **核心结果**：
  - 自反馈平均较 ICL/ACE/DC 提升 7-9%（62.9% → 71.0%），整体平均提升 5.6%，API 成本降低 19%。
  - 正确性/执行反馈平均提升 11.1%；代码任务执行反馈提升 8.6%（66.4% → 75.0%）。
  - 双反馈组合在 AIME 2025 达 76.7%（超正确性单一 70.0% 与模型单一 60.0%），LiveBench (Code) 达 81.2%。
  - DC 与 SimpleMem 等压缩检索方法均低于完整轨迹，说明激进压缩会丢弃关键中间推理。
  - 最强结果：GPT-5 / Claude-4.5 Sonnet 在正确性反馈下 AIME 2025 最高达 89.05%，GPQA Diamond 达 99.52%。
  - 提升归因分析显示 47.7% 的改进由反馈直接驱动，代码任务中 30.0% 来自规格/格式对齐（Specification Recall）。

## 相关工作脉络
1. **Test-time scaling (CoT/ToT/Majority Voting)**：侧重并行搜索与静态验证，缺乏基于反馈的时序迭代闭环；CoE 以单轨迹多轮演进替代并行冗余计算。
2. **Self-Refine / Reflexion / Self-Debug**：聚焦单一任务内的自修正，未系统探索多源反馈谱系与跨域泛化边界；CoE 将其统一纳入反馈驱动的经验累积框架。
3. **ICL / DC / ACE**：依赖跨任务检索与外部记忆压缩；本文证明在具备强推理能力的现代模型上，完整历史轨迹的直接复用优于主动蒸馏与压缩。
4. **RL / 后训练对齐**：依赖离线参数更新实现经验内化；CoE 属于纯训练无关（training-free）的测试时上下文学习范式，可与参数更新阶段互补。
5. **Step-level / Output-level Verifier**：仅用于最终答案筛选；CoE 将验证信号融入生成回路，使模型在迭代中动态调整推理路径。

## 局限性与未来方向
1. **评估范围受限**：主要聚焦数学、代码与知识类基准，未充分验证于长程工具调用、GUI 自动化等需多步环境交互的真实场景。
2. **无参数更新机制**：改进仅依赖上下文经验复用，并非真正的权重学习；未来需探索将 CoE 轨迹回灌至 SFT/RLHF 以实现经验内化。
3. **分布外知识任务适应性弱**：在 BrowseComp-Plus 等需外部检索的知识任务中，自反馈反而导致性能下降，表明纯内部反思难以弥补知识盲区。
4. **反馈依赖性强**：正确性反馈效果显著但现实中常不可得，模型反馈质量受裁判模型基座能力制约，存在“强模型互评更优、弱模型互评失效”的阈值现象。

## 研究启发与可借鉴点
1. **反馈谱系可作为通用插件**：将执行器/正确性信号与模型自评组合，能低成本替代高算力的多路径蒙特卡洛搜索，适合资源受限的 Agent 部署。
2. **“保全轨迹优于激进压缩”**：记忆检索与摘要策略在推理型任务中可能破坏中间思维链，后续工作应优先保留完整迭代历史再考虑选择性聚合。
3. **早期迭代收敛规律**：大部分性能增益在前 20 轮内完成，提示实际系统中可设置动态停止阈值以进一步压低延迟与成本。
4. **基座能力预测学习上限**：Pearson r=0.50 的正相关关系为模型选型提供依据，强基座模型更适合开启 CoE 循环，弱模型可先做针对性 SFT 再启用测试时迭代。

## 关键术语表
**Chain-of-Experience (CoE)**：模型在推理阶段通过多轮迭代与反馈积累经验并持续改进的统一测试时框架。
**Feedback Spectrum**：按信号显式程度划分的反馈类型谱系，涵盖无反馈、执行反馈、模型反馈与正确性反馈。
**Self/Model Feedback**：由主模型自身或辅助裁判模型生成的文本批评、评分或结构化评估信号。
**Execution Feedback**：在代码/交互环境中通过程序运行获取的错误日志、测试用例通过情况等硬信号。
**Correctness Feedback**：二值 Oracle 监督信号（0/1），提供答案对错判定，通常作为性能上限基准。
**Selective Majority Voting (SelMV)**：对前 n 次有效尝试的答案进行多数投票，用于缓解虚假或弱反馈带来的性能波动。
**Test-time Scaling**：不更新模型参数，通过在推理阶段增加计算/迭代轮次来提升最终性能的方法范式。
**Specification Recall**：模型在迭代中重新对齐题目指令或输出格式要求从而触发正确性翻转的改进模式。

## 可复现要素
- **数据集**：AIME 2025、OmniMath、LiveCodeBench (V6)、LiveBench (Code)、EvaLearn、GPQA Diamond（均为公开基准）。
- **代码/权重**：论文未提及开源代码、数据或模型权重。
- **关键超参**：OpenAI 模型使用默认温度 1.0；Gemini/Claude 温度 0.2；最大迭代轮数 20；检索候选数 k ∈ [1, 5, 8, 12, 15, 20]；Claude 低推理力度对应 thinking budget 10,000，高推理力度关闭 thinking mode；实验重复 3 次取均值。
