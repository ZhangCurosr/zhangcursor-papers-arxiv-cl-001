---
title: "HSRM-Hidden-State-Reward-Models-for-Test-Time-Verification"
source: https://arxiv.org/pdf/2608.30841v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 11:05:35"
field: "大模型推理验证与测试时扩展"
keywords: ["test-time reasoning", "hidden-state verifier", "best-of-N selection", "reward model", "mathematical reasoning"]
innovations: ["提出 HSRM，以约2M参数直接读取冻结生成器步骤边界隐藏状态进行候选排序，避免文本重编码", "采用 tie-safe Bradley–Terry 成对排序损失，适配多正确候选的 best-of-N 场景", "系统验证隐藏状态验证在跨生成器规模、跨模型族与零样本迁移中的高效性"]
benchmarks: ["GSM8K", "MATH-500", "AIME", "OlympiadBench"]
---

# 论文速读：HSRM-Hidden-State-Reward-Models-for-Test-Time-Verification

## 一句话总结
本文提出 HSRM，一种轻量级隐藏状态奖励模型，直接读取冻结生成器在推理步骤边界处的内部隐藏状态来对候选解答进行评分排序，无需重新编码文本；在四个数学推理基准上，仅用约 2M 参数即匹配或超越 55M 参数的文本能量验证器 EORM，并在 GSM8K 上与 7B 领域 PRM 接近。

## 研究问题与动机
- 现有测试时推理验证器多为文本基，需在生成后对候选方案重新进行完整前向编码，导致验证成为推理成本的主要部分。
- LLM 内部表示中已包含与其自身答案正确性相关的信号，文本-only 验证器仅通过输出词元间接推断，未充分利用生成过程中已产生的隐藏状态。
- 在 best-of-N 选择任务中，若验证器能直接读取生成器的内部表示，则可在不额外调用生成器前向传播的前提下高效排序候选。
- 如何在保持验证性能的同时显著降低参数规模与推理 FLOPs，是当前测试时计算扩展中验证模块的瓶颈之一。

## 核心贡献（创新点）
- 提出 HSRM 架构，直接从冻结生成器的最后层（或上层多层的）推理步骤边界提取隐藏状态，经小型 Transformer 编码器映射为标量候选分数，避免对生成文本的重编码。
- 设计 tie-safe ranking loss（基于 Bradley–Terry 的成对排序损失），仅要求正确候选得分高于错误候选，不对同类正确候选之间强加虚假排序。
- 验证数据完全由目标域自身的 best-of-N 采样轨迹与最终答案标签构建，无需人工过程监督或预建 verifier 语料，可随候选采样轻松迁移到新域。
- 系统评估表明，约 2M 参数的 HSRM 在 16 组生成器-数据集设置中 15 组达到或优于 55M 参数 EORM，且验证成本较 7B PRM 低约五个数量级；并进行层位、输入模态、训练数据量、零样本迁移、thinking-mode 提取策略与跨模型族泛化等多维消融。

## 方法详解
- **步骤边界隐藏状态提取**：对候选解 y 的最后层隐藏序列 {h^ℓ_t}，使用步骤停止分隔符（如 `\n\n`、`\n`、`.` 等）定位步骤边界位置 t_1 < t_2 < ··· < t_S，收集 H^ℓ(y) = [h_{t_1}^ℓ; …; h_{t_S}^ℓ] ∈ R^{S×d_gen}，默认取 ℓ = L。
- **HSRM 编码器结构**：先通过输入线性投影 z_s^{(0)} = W_in h_{t_s}^ℓ + b_in，再经 2 层 Transformer 编码器（默认 d_model = 256，4 头注意力，FFN 宽 4d_model），随后对步骤维度做 mean-pooling 并加 LayerNorm，最后接线性标量头 f_φ(x, y) = w^T z + b。
- **训练目标**：采用 tie-safe Bradley–Terry 对数损失 L_rank = (1/|P||N|) Σ_{i∈P} Σ_{j∈N} log(1 + exp(-(s_i - s_j)))，仅惩罚“正确候选低于错误候选”的情况；同题无正确或无错误候选时跳过；使用缓存的隐藏状态张量进行训练，训练过程中不再调用生成器。
- **推理流程**：给定 x，生成器采样 N 个候选并缓存其步骤边界隐藏状态；HSRM 对 N 个候选批量打分，best-of-N 选取得分最高者 ŷ = argmax_i f_φ(x, y_i)，无需额外文本重编码与前向传递。

## 实验与结果
- **设置**：Qwen3 生成器 1.7B/4B/8B/14B，数学推理基准 GSM8K、MATH-500、AIME、OlympiadBench；每训练题采样 64 候选，评估时 best-of-8；标签使用正则/符号匹配加 LLM judge 两级校正。
- **主结果**：在 16 组设置中 15 组 HSRM 匹配或优于 55M 参数 EORM；GSM8K 上与 7B Qwen2.5-Math-PRM 接近，随生成器规模增大 HSRM 逼近 oracle pass@8 上界；更难的 MATH-500/AIME/OlympiadBench 仍持续优于 EORM。
- **消融要点**：
  - 层位：正确性信号分布于多个上层，最终层表现有竞争力但不唯一最优；Top-4 层拼接的 step 输入取得最佳 AUROC。
  - 输入模态：纯文本 baseline 仅 82.23% Bo8 Acc 与 0.519 AUROC；单步最后层隐藏状态已达 86.11%/0.691；Top-4 层 step 输入达到 GSM8K 86.86%/0.724；文本+隐藏状态混合并未继续提升，表明内部表示信号已足够强。
  - 效率：HSRM 参数约 2.12–3.4M、权重 4–7MB、输入 ≤100 个步骤向量、零额外生成器调用；相比 EORM 55M（~107MB，需文本重编码）与 7B PRM（~15GB，giant-scale 重编码），验证 FLOPs 低约五个数量级。
  - 数据量：GSM8K 上 K 从 25 增至 500，Bo8 准确率 83.4→86.2，AUROC 0.612→0.716。
  - 零样本迁移：在 GSM8K/MATH-500 训练后直接评测 OlympiadBench，HSRM 持续优于 EORM 并贴近 7B PRM。
  - Thinking 模式：仅提取 `
</think>

` 之后的答案段，MATH-500 上 Qwen3-4B 的 within-problem AUROC 从 0.672 提升至 0.884。
  - 跨族泛化：在 Llama-3.2-1B/3B、Llama-3.1-8B 上，HSRM 在 6 组设置均优于匹配 EORM（GSM8K 提升 3.1–8.1pp，MATH-500 提升 3.6–4.5pp）。
- **最强结果**：GSM8K Qwen3-14B 下 HSRM 达 95.8%，逼近 oracle 96.0%，同时以 2–3.4M 参数击败 55M EORM 并与 7B PRM 接近；跨族 Llama-3.1-8B GSM8K 达 88.3%，超出 EORM 85.2% 达 3.1pp。

## 相关工作脉络
- **Text-based verifiers / EORM**：Cobbe 等 outcome reward、Uesato/Lightman/Wang 等过程奖励；EORM 使用 55M Transformer 对完整 CoT 文本做能量打分。HSRM 的差异在于直接读取生成器内部表示，避免二次文本编码。
- **Process Reward Models**：Yang/Zhang/Zheng 等将 PRM 做到大规模 LLM 级别，验证成本高；HSRM 以 2M 参数实现可比性能，适合测试时 compute 受限场景。
- **Correctness signals in internals**：Kadavath、Azaria/Mitchell、Burns、Li 等揭示 LLM 内部含truth/correctness 信号；本文将其从诊断/probing 推进到作为验证器的主要输入并可端到端学习。
- **Hidden-state probes / SWIFT / ReProbe**：Zhang、Liang、Piotrowski 及 Guo 等的 probe 与激活特征方法偏诊断或免训练；HSRM 是端到端小 verifier，且在步骤边界做聚合而非逐 token 聚合，更适合 best-of-N 候选级排序。
- **Scaling test-time compute / Inference scaling laws**：Brown/Snell/Wu 等工作强调以重复采样+验证提升推理；本文把 verifier 成本压至极低，使“以计算换准确”更易扩展。
- **Representation engineering / layer semantics**：Zou、Marks/Tegmark、Liu 等研究表征结构；本文层位与多高层拼接消融支持“正确性分布在上层且跨层互补”的假设。

## 局限性与未来方向
- 聚焦数学推理，尚未系统在代码、科学、开放域等更广泛任务上验证可迁移性。
- 对 thinking-mode 的探索仅限 Qwen3-1.7B/4B 与 MATH-500，缺乏更大规模或多任务对照。
- 训练数据依赖目标域可采样候选与可获取的最终答案标签；对无 ground-truth 答案或高噪声标注的域，标签质量可能制约训练。
- 输入采用固定分隔符切分步骤，对未遵循约定格式或步骤粒度极细/极粗的生成行为可能欠鲁棒。
- 目前为 outcome-level 验证器，未集成过程级错误定位或步骤修正能力。

## 研究启发与可借鉴点
- **隐藏状态作为验证输入**：任何 best-of-N/MCTS/自搜索管道均可复用 frozen generator 的步骤边界隐藏状态，避免昂贵文本重编码，显著扩展可负担的 N 上限。
- **Tie-safe pairwise 排序损失**：在多正确候选场景下比 pointwise BCE 和 ListMLE 更贴合 best-of-N 目标，可迁移至多项选择、多解生成与程序合成等“多正确项”设定。
- **多高层拼接 + 步骤聚合**：证明正确性信号并非仅集中于末层，采用 top-K 层拼接与语义边界（step delimiter）聚合可在不显著增参的前提下提升排序质量。
- **低成本 verifier 的零样本迁移**：在源域训练 verifier 直接用于目标域的实验设计，可作为快速适配新域的基线模板，用于检验信号通用性而非仅过拟合数据。
- **Thinking-mode 提取策略**：仅取 deliberation 结束后的最终答案段隐藏状态优于全 trace，提示在引入反思/自我修正结构的模型中，验证输入应在“稳定输出阶段”而非“探索阶段”采样。

## 关键术语表
- **Best-of-N reasoning**：对同一输入采样多个候选解，由验证器评分后选择最优者输出，以测试时计算换取准确率。
- **Outcome reward / Process reward**：前者仅依据最终答案判对错，后者对中间推理步骤给予逐步反馈以支撑学习或搜索。
- **Tie-safe Bradley–Terry ranking loss**：仅在正负对间施加排序约束，同标签候选之间保持 indifferent，避免虚假排序压力。
- **Step-boundary hidden state**：在链式推理的步骤分隔符前截取的最后一词元隐藏表示，用于把长序列压缩为步骤级序列。
- **Energy-based verifier**：通过对输入序列计算能量/打分函数，以低能量对应高质量候选的无归一化评分器。
- **Within-problem AUROC**：在同一问题内候选对之间的 ROC 曲线下面积，反映验证器对正确/错误候选的相对排序能力。
- **Verifier cost frontier**：以验证 FLOPs 或参数量为代价衡量验证器选择增益的帕累托效率曲线。
- **Thinking-mode / deliberation trace**：模型在给出最终答案前先进行内部推导/反思的生成模式，常用 `
</think>

` 界定反思段与答案段。

## 可复现要素
- **数据集**：GSM8K、MATH-500、AIME、OlympiadBench（公开数据集）；训练/评测题集索引见 Appendix A，训练池与评测集 disjoint。
- **代码/权重**：论文未声明开源仓库与 HSRM 权重下载链接，仅给出训练超参与架构细节。
- **关键超参**：HSRM 默认 2 层 Transformer、d_model=256、4 头、FFN=4d、dropout=0.1；输入取生成器最后层；AdamW、lr=1e-4、problem batch=8、1000 gradient steps、20% 题级验证集；候选采样 N=64（训练）、N=8（评估）；生成器 temperature=0.7、top-p=0.9、fp16。
- **标签流水线**：正则/符号匹配为主，难样本由 LLM judge 复核；约 3.1% 标签被修改，净增 7512 个正确标签。
- **硬件**：L40S 与 H100 GPU；训练缓存最终层 float16 隐藏状态。
