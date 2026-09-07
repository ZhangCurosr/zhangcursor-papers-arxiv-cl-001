---
title: "When-Models-Edit-Too-Much-On-the-Fidelity-of-Minimal-Code-Ed"
source: https://arxiv.org/pdf/2609.04061v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 23:14:29"
field: "代码大模型评测与后训练"
keywords: ["代码编辑", "最小修复", "编辑保真度", "大语言模型", "强化学习微调", "过度编辑", "BigCodeBench", "Levenshtein 距离"]
innovations: ["以已知最小 AST 逆转构建受控编辑保真评测框架", "提出 excess Levenshtein 与 added cognitive complexity 双指标衡量最小修复偏差", "证明 RL 可在保持泛化编码能力的同时学会可迁移的最小编辑偏好"]
benchmarks: ["BigCodeBench", "LiveCodeBench v6", "Defects4J"]
---

# 论文速读：When-Models-Edit-Too-Much-On-the-Fidelity-of-Minimal-Code-Ed

## 一句话总结
论文首次系统研究代码大模型的**过度编辑（over-editing）**现象：模型在通过测试的前提下，仍对本地 bug 执行不必要的大规模改写。作者构建了一个含已知最小 patch 的评估基准，并提出保留提示词与 RL 后训练两条路径以提升编辑保真度。

## 研究问题与动机
- 现有代码评测以 Pass@1/Pass@k 为主，只能反映功能正确性，无法衡量**最小化与原始意图保留**。
- 工程维护（brownfield）场景需要易审查、低风险且贴近原实现的修复，而非健壮性重写或防御性扩展。
- 当前基准虽包含仓库级/修复类任务，但仍缺乏以“最小修复”为对照的受控度量。
- 推理模型与模型规模增大并未自动带来更保守的局部修复策略。

## 核心贡献（创新点）
- 构建基于 BigCodeBench 的受控 AST 注入评估框架，提供已知最小反向 patch 的 400 个修复任务。与已有基准的区别在于：以构造最小修复为 ground truth，直接度量**超过必要改动**的编辑保真轴。
- 提出并验证两个编辑保真指标：**excess normalized token Levenshtein distance**与**added cognitive complexity**，并通过人工标注与审计证明其与审查友好性、保真性的强一致性。与 CodeBLEU 等相似度指标相比，能避免“长 n-gram 表面相似却实际大量重写”的误判。
- 系统评测多款前沿/开源模型，揭示高 Pass@1 与过大编辑可并存，且过编辑多为颗粒度错位（defensive/generalization/data-flow rewrite 等模式）。与仅报告通过率的工作相比，揭示同一模型在正确性与最小性上的解耦行为。
- 证明简单 preservation instruction 即可显著降低过度编辑并提升 Pass@1；与依靠更大模型或更长 reasoning budget 的做法相比，提示干预具有更稳定的跨模型收益。
- 对比 SFT/rSFT/DPO/RL 的训练路径，发现 SFT 易 overfit 到训练 corruption 模式，而 RL 在 OOD 场景取得更高 Pass@1 并维持更小过度编辑，且不损害 LiveCodeBench 泛化能力。与 PaFT/PRepair 等面向最小编辑的前作相比，本文强调最小编辑可作为可学习且可迁移的**后训练偏好**。

## 方法详解
- **Benchmark 构造**：从 BigCodeBench 采样 400 题，对参考解注入 1–2 个可控 AST 级别 corruption，保留使测试失败的样本。gold repair 即为 injected corruption 的可逆操作，构成已知最小 patch。corruption 类型覆盖 comparison/range/sort/accumulator/arithmetic/guards/indexing/call substitution/copy removal/boolean/numeric/slice/conditional/step 等 14 种家族。
- **度量设计**：
  - **Pass@1**：功能正确性。
  - **Token-level excess normalized Levenshtein distance**：$E_{\mathrm{Lev}}(M)=d(M,C)-d(G,C)$，其中 $d(\cdot,\cdot)$ 为移除注释与格式后的函数体 token 距离并按两者较大 token 数归一化。$E_{\mathrm{Lev}}>0$ 表示比 gold patch 改得更多。
  - **Added cognitive complexity**：以 Python AST visitor 计算 $CC(M)-CC(G)$，反映除 patch 大小之外的人读复杂度增量。
- **提示干预**：在通用请求末尾增加 preservation clause “but keep as much of the original code as possible”，不改变 system prompt/task description/test suite。
- **后训练设置**：以 Qwen3-4B-Instruct-2507 为基础，使用 DeepCoder 数据训练；训练集每样例 1–10 个 corruption，测试集包含 in-domain 与 20 类 OOD corruption。方法包括 SFT、rSFT、DPO 与 GRPO-style RL；RL reward 为 $r(M)=\lambda_{\mathrm{exec}}-\lambda_{\mathrm{edit}}e(M)$，失败/unparsable 给 -0.2；$e(M)$ 为上述 excess 距离，$\lambda_{\mathrm{exec}}=0.1,\lambda_{\mathrm{edit}}=1.0$。
- **评估泛化与能力保留**：以外置 OOD 指标衡量编辑保真迁移，并以 LiveCodeBench v6 的相对变化衡量是否损害更广泛编码能力。

## 实验与结果
- **数据集**：400 个 BigCodeBench Python 任务，共 568 次 corruption 注入；中位函数长度 10 行，gold patch 中位 token edit dist 为 1，91.8% 的 patch 至多 2 token。
- **基线模型**：涵盖 GPT-5.x、Claude Opus/Sonnet 系列、Gemini、DeepSeek V3/R1、Qwen3-Coder/GLM/Kimi/Magistral/Grok 等前沿与非推理模型，并在多个开放权重模型上做提示消融。
- **关键结果（聚合）**：
  - 通用提示下，前沿模型平均 excess Lev. 为 0.195；加入 preservation instruction 后降至 0.131，added CC 下降 26.6%，Pass@1 提升 2.3 点（配对检验显著）。
  - GPT-5.5 High：Pass@1=0.823，但 excess Lev.=0.299，约为最佳非推理模型 Opus 4.7 的四倍以上。
- **推理与规模**：推理开启并非单调降低过度编辑；部分模型在显式保留提示下仍出现复杂度上升。模型放大（Qwen2.5-Coder 0.5B–32B）使 Pass@1 单调提升，但过量编辑并不单调下降，14B→32B 时 excess Lev. 从 0.108 升至 0.127。
- **最易引发过编辑的 bug 类型**：slice bounds（excess Lev.=0.353，Pass@1=0.874）、list indexing（0.244/0.780）、comparison ops（0.210/0.799）、sort order（0.239/0.709）、conditional inv（0.207/0.778）等，呈现“高通过率+高过度编辑”的共存。
- **过编辑模式（n=530）**：defensive generalization 64.2%，data-flow rewrite 63.2%，contract drift 34.7%，feature accretion 23.6%，dependency fallback 3.2%。
- **后训练最优**（OOD）：RL 取得 Pass@1=0.782、excess Lev.=0.050、added CC=0.185，且 LiveCodeBench v6 为 33.2%（相对 base 32.6% 提高 +0.6 点）；SFT 在 OOD 上 Pass@1 跌至 0.458 并损失 LCB 14.9 点。LoRA 秩 64 可恢复大部分 RL 收益。
- **跨语言迁移**：在 Defects4J Java 单方法 bug 上，RL 保持通过率并显著减少 token edits 与 excess Lev./CC；绝对通过率受模型规模限制仍较低。

## 相关工作脉络
- **Correctness-centered benchmarks**（HumanEval/MBPP/SWE-bench 等）关注任务通过或 issue 解决，但未控制“最小修复”ground truth，无法单独度量编辑保真。本文在此基础上引入额外正交维度。
- **CanItEdit / EDIT-Bench / CodeEditorBench / CoreCodeBench / DebugBench** 等扩展了指令编辑/仓库级任务，但仍以通过率为主体；本文的差异在于构造可控的最小逆转 patch 并量化“超过必要修改”的部分。
- **CREF（Yang et al., 2024）**面向辅导场景的 patch precision；本文面向前沿模型的系统性评估，并提供可学习的最小编辑偏好。
- **AdaPatcher（Dai et al., 2025）**结合定位与 preference learning；本文在无 trace 设定下与 DPOP 直接比较，证明 on-policy RL 在 OOD 泛化上优于离线偏好优化。
- **PAFT（Yang et al., 2026）**与 **PRepair（Ke et al., 2026）**同样关注最小/精确编辑；本文的差异在于：受控 benchmark+跨前沿模型测量+RL 可学习性证据+跨语言/跨规模迁移评估。
- **CodeBLEU 等传统相似度度量**更偏 lexical/n-gram 重合；本文选用 token Levenshtein 以避免“改写大量代码但保留长段周边文字”时被误判为更小的偏差。

## 局限性与未来方向
- 基于函数级、可控注入的 Python 任务，较仓库级真实多文件缺陷更简单，推广需进一步验证。
- 训练与评估主要集中在 Qwen 系列，跨家族 generalize 仍待扩大。
- 人工评测规模有限（100 对 pair、3 位 annotator），需更大规模人类审查研究支撑。
- RL reward 中使用 added cognitive complexity 未获最佳效果，未来需探索更精细的结构/审查成本 reward。
- 真实场景的边界：防御性重写有时在 open-ended 编程中合理，如何更好地界定“必要”与“过度”仍需语义层面的补充评估。

## 研究启发与可借鉴点
- **正交评估维度**：将 edit fidelity（excess Lev. + added CC）与 Pass@1 并列，可作为团队在代码编辑/修复任务上的新评测标准。
- **受控最小 patch 构造思路**：从参考实现注入可逆 AST 变化以获取 ground truth 修复，便于在自建数据上复现并扩展到其他语言。
- **RL reward 设计**：execution success + edit minimality 的组合在 OOD 泛化与能力保留上表现最佳，可迁移至“按约束修改已有实现”的任务（如 refactoring、lint fix）。
- **LoRA 微调足以捕获风格化偏好**：最小编辑更接近可学习的偏好而非硬技能，提示与轻量 adapter 的组合是低成本的落地路径。
- **提示工程稳定且显著**：单一 preservation clause 即可在多模型上带来收益，适合工程侧作为默认编辑指令。

## 关键术语表
- **Over-editing**：模型修复通过测试，但对本地 bug 执行超出最小必要改动的行为。
- **Excess normalized Levenshtein distance**：模型 patch 与污染代码的 token 距离减去 gold 最小 patch 距离后的归一化剩余量，衡量“多余改动”。
- **Added cognitive complexity**：修复前后 AST 访客计算的认知复杂度差值，反映引入的结构开销。
- **Preservation instruction**：要求模型在修复时尽可能保留原代码的提示语句。
- **GRPO-style RL**：采用 group-relative 优势估计的组内强化学习优化方式。
- **In-domain / Out-of-domain**：评估 corruption 是否与训练中所用家族一致；OOD 衡量泛化性。
- **DPOP / DPO**：分别指对正样本偏好优化的 DPO 变体与标准直接偏好优化。
- **Data-flow rewrite / Defensive generalization**：两类典型过编辑模式，前者替换数据流，后者添加广泛防御性检查。

## 可复现要素
- **数据集**：BigCodeBench（Apache 2.0）400 题及 DeepCoder（MIT）训练样本；注入方案与 14 类 corruption 家族见 Appendix A。
- **代码/权重**：使用 Qwen3/Qwen2.5-Coder（Apache 2.0）与 Llama-3.1-70B-Instruct；训练使用 LlamaFactory（Apache 2.0）与 PRIME-RL（Apache 2.0）。论文未公开自有代码/权重，但提供了完整实验设置与提示模板。
- **关键超参**：SFT/rSFT/DPO 学习率 $1e-5$、3 epoch；DPO beta=0.1；RL 学习率 $1e-6$、rollout K=16、group-mean baseline；$\lambda_{\mathrm{exec}}=0.1$、$\lambda_{\mathrm{edit}}=1.0$；失败/unparsable reward=-0.2。LoRA rank 扫 1/8/16/32/64。推理采样 temperature=1，thinking budget 10k tokens（如适用）。
