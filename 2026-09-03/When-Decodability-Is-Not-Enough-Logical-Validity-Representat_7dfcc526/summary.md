---
title: "When-Decodability-Is-Not-Enough-Logical-Validity-Representat"
source: https://arxiv.org/pdf/2609.02438v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 00:33:07"
field: "语言模型可解释性"
keywords: ["logical validity", "mechanistic interpretability", "linear probing", "causal intervention", "behavior-representation dissociation", "out-of-distribution generalization"]
innovations: ["构建配对受控逻辑验证数据集以隔离关系属性与表面形式", "提出行为表达-线性可及-因果使用三层分离评估框架", "发现行为错误样本中有效性信息仍可高度线性解码"]
benchmarks: ["自制受控验证集（800样本/400对）"]
---

# 论文速读：When-Decodability-Is-Not-Enough-Logical-Validity-Represent

## 一句话总结
本文在逻辑验证任务上发现：语言模型虽然行为表现接近随机，但其隐藏状态中却几乎完美编码了逻辑有效性信息；该信息即使在模型回答错误时仍可被线性解码，但通过探针导出的方向进行因果干预却仅产生微弱且非特异性的行为影响，表明**表示可及性 ≠ 行为表达 ≠ 因果使用**。

## 研究问题与动机
- 现有大模型推理评测依赖行为正确性，但正确答案可能源于词汇/语义捷径而非真正的逻辑计算，无法揭示内部表征性质。
- 已有工作证明真相（truth）可从激活中恢复，但逻辑有效性（validity）是前提与主张之间的**关系属性**，需要更精细的控制设计来隔离。
- 高探针准确率并不等价于该特征被模型真正"使用"；需要区分线性可解码性、行为表达与因果影响三者。
- 若有效性信息在错误样本中仍可解码，则说明模型"知道"正确答案却无法输出，这一 dissociation 对解释机制 interpretability 具有重要意义。

## 核心贡献（创新点）
1. **构建受控逻辑验证数据集**：400对匹配valid-invalid示例，固定前提/推理族/语义域/模板/难度，仅改变主张，隔离关系属性与表面形式——相比 FOLIO/PrOntoQA 等基准更强调配对控制。
2. **三层分离评估框架**：同时评估行为表达、隐藏状态线性可及性、探针方向因果干预，证明三者独立；现有工作通常只测前两层。
3. **发现行为错误下的代表性保留**：在 Llama-3.2 等模型上，行为错误样本的有效性 AUROC 仍达 0.994–1.000，表明"不会答"不等于"不知道"。
4. **系统性OOD泛化分析**：leave-one-domain-out 与 leave-one-family-out 显示泛化广但非均匀，反复暴露 syllogism 为最弱迁移族。
5. **因果干预对照**：探针方向仅产生 ±0.002–0.004 的margin变化，与 norm-matched 随机方向相当，否定线性方向的强因果控制假设。

## 方法详解
- **数据集构造**：每个示例表示为 $x_i = (P_i, c_i, y_i, f_i, d_i, t_i, \delta_i)$，其中 $P_i$ 为前提集合，$c_i$ 为候选主张，$y_i \in \{0,1\}$ 为有效性标签；400对匹配样本满足 $P^+=P^-$、$f^+=f^-$、$d^+=d^-$、$t^+=t^-$、$\delta^+=\delta^-$，仅 $y^+=1, y^- = 0$。
- **推理族×语义域×难度**：5 个推理族（syllogism、transitivity、set inclusion、causal chain、permission logic）× 5 个域（nonce、spatial、social、biological、legal-policy）× 3 个难度（1步直接、2步链、2步+干扰项）。
- **四层划分**：Random（分布内）、Template held-out、Domain held-out、Family held-out；外加 exhaustive leave-one-out。
- **行为评估**：计算 VALID/INVALID token 的条件对数似然 $s_V(x), s_I(x)$，预测 $\hat{y} = \mathbb{I}[s_V > s_I]$，输出 margin $M(x) = s_V - s_I$。
- **线性探针**：提取每层最终 prompt-token 隐藏状态 $h_{i,\ell} \in \mathbb{R}^m$，拟合 $\ell_2$ 正则逻辑回归；超参 $C \in \{0.1, 1, 10\}$ 通过 pair-aware CV 选择；选层 $\ell^* = \arg\max_\ell \text{AUROC}_\ell^{\text{val}}$；指标为 AUROC 与 matched-pair accuracy。
- **控制实验**：Full-prompt TF-IDF、Claim-only TF-IDF、Premises-only、Metadata-only（推理族/域/难度/模板ID/长度等）；200 次 shuffle-label 置换检验。
- **因果干预**：在选定层 $\ell^*$ 沿归一化探针方向 $v$ 添加扰动 $h' = h + \alpha \sigma_v v$，$\alpha \in \{-4,-2,-1,0,1,2,4\}$；测量 $\Delta M_\alpha(x) = M_\alpha(x) - M_0(x)$；对照使用 5 个 norm-matched 正交随机方向；另做 matched-pair patching：$h_i' = h_i + (v^\top h_j - v^\top h_i)v$。

## 实验与结果
- **模型**：Pythia-1.4B、Pythia-2.8B、SmolLM3-3B、Llama-3.2-3B、Mistral-7B（均为开源 transformer，部分量化）。
- **行为结果**：所有模型准确率接近 0.5（chance），Pythia-1.4B/SmolLM3-3B/Mistral-7B 呈现极端答案标签偏好（全 INVALID 或全 VALID），Llama-3.2-3B 为 0.470；margin AUROC 亦近随机（0.458–0.576）。
- **探针可解码性**：Random-split AUROC = 1.000（所有模型）；第一层即达 0.935–0.995；Template-held-out 0.963–0.999；Domain-held-out 0.771–0.991；Family-held-out 0.938–0.995。
- **OO D泛化**：Leave-one-out 显示 domain 失败多为模型特异（Pythia 在 biological 降至 0.576/0.714，Mistral 在 legal-policy 降至 0.771）；family 层面 **syllogism 普遍最弱**（0.500–0.762）。
- **行为错误下的表示保留**：Matched-pair accuracy 在所有模型/划分下均为 1.000；Llama-3.2 在行为错误子集上 AUROC 达 1.000/0.995/1.000/0.994（random/template/domain/family）。
- **控制结果**：Full-prompt TF-IDF 0.970（random）→ 0.814（domain）；Claim-only TF-IDF 0.965 → 0.682；Shuffled-label null AUROC ≈ 0.497±0.035；所有模型的 observed random-split 均超过全部 200 次置换（$p=1/201\approx0.005$）。
- **因果干预**：$\alpha=+4$ 时 $\Delta M$ 为 −0.0037（Pythia-2.8B）、−0.0022（Llama-3.2）、+0.0023（Mistral-7B），符号不一致；随机方向 Mean |ΔM| 为 0.0058–0.0077，大于或等于探针方向；零预测翻转（仅 Llama-3.2 在 α=−4 翻转 1/160）。

## 相关工作脉络
- **Bertolazzi et al. (ACL 2026)** [1]：证明三段论中有效性与语义合理性均可线性解码并可因果引导；本文扩展至多推理族/多域，并强调"可解码≠可因果控制"，且聚焦错误样本中的保留性。
- **Sahoo et al. (2026)** [12]：指出线性探针可能检测 task format 而非推理模式；本文通过 matched-pair、shuffled-label、metadata 等多重控制回应该质疑。
- **Burns et al. (ICLR 2023)** [2] / **Marks & Tegmark (2023)** [8]：证明 latent truth/plausibility 可线性恢复；本文聚焦 relational validity（前提-主张关系）而非 claim 本身的真值。
- **FOLIO** [3] / **LogicBench** [10] / **Multi-LogiEval** [11] / **PrOntoQA-OOD** [13]：行为基准，评估推理准确性；本文补充其盲区——行为正确/错误背后的内部表示性质。
- **Li et al. (ICLR 2023)** [6] / **Nanda et al. (2023)** [9]：在 Othello-GPT 等合成任务中恢复并干预世界模型变量；本文将此范式迁移至自然语言逻辑验证，并发现干预效果极弱。
- **Zou et al. (2023)** [18]（Representation Engineering）：主张通过顶向下方向工程实现透明性；本文结果对其乐观假设提出谨慎限定。

## 局限性与未来方向
- 数据集为合成/受控数据（800 样本、5 族×5 域），外部效度有限，难以直接推广至自然推理或开放生成任务。
- 仅评估最终 prompt-token 的线性探针；有效性可能以非线性、跨 token 分布或中间计算位点编码。
- 因果干预仅在选定单层施加线性扰动，未探索多层/非线性干预；弱效应不等同于有效性信息因果无关。
- 所有模型为小/中规模开源 transformer（1.4B–7B）且经量化；未在大模型或不同架构上复现。
- 未考察 chain-of-thought / 提示工程等外部干预对"表示-行为 gap"的缓解作用。

## 研究启发与可借鉴点
- **三层分离范式**（行为表达→线性可及→因果使用）可迁移至其他解释性研究（如事实性、因果性、规划能力），避免将高探针准确率直接等同于模型"掌握"某概念。
- **Matched valid–invalid pair 控制设计**有效剥离上下文/词汇 confound；可推广至任何二元关系判断任务（如蕴含、等价、道德判断）。
- **Correctness-conditioned probing**：按模型行为对错分组后分别评估探针，能揭示"隐性知识"；适用于分析 instruction-tuned / RLHF 模型的内部状态。
- **Exhaustive leave-one-out 替代单一 held-out**：发现泛化的异质性（如 syllogism 普遍脆弱），避免 averaged metric 掩盖系统性弱点。
- **Norm-matched random orthogonal control 作为因果干预 baseline**：本文对照设计严谨，可成为后续 activation steering 论文的标配实验。

## 关键术语表
**Logical validity**：前提与主张之间的逻辑关系属性，主张是否由前提必然推出，与主张的事实真值无关。
**Matched valid–invalid pair**：前提、推理族、域、模板、难度完全相同，仅主张真假标签相反的一对样本，用于控制上下文 confound。
**Linear probe**：在固定层冻结激活条件下训练的 $\ell_2$ 正则逻辑回归分类器，用于测试某特征是否线性可分离。
**AUROC**：Receiver Operating Characteristic 曲线下面积，衡量探针区分 valid/invalid 的排序能力，0.5 为随机，1.0 为完美。
**Behavioral margin**：$M(x) = s_V(x) - s_I(x)$，模型分配给 VALID 与 INVALID token 的条件对数似然之差，反映输出倾向强度。
**Causal intervention (activation steering)**：在选定层的隐藏状态上沿特定方向加减标量扰动，测量对输出 margin 的因果效应。
**Correctness-conditioned evaluation**：按模型行为预测是否正确将样本分组，分别评估探针在各子集上的可解码性。
**Leave-one-out generalization**：逐一排除某一域或推理族后重新训练探针，测试跨条件泛化能力。

## 可复现要素
- **数据集**：论文构造的 800 样本受控验证集；论文未声明开源链接（需联系作者或查看 arXiv 附录/代码库）。
- **代码/权重**：五个开源模型（Pythia-1.4B、Pythia-2.8B、SmolLM3-3B、Llama-3.2-3B、Mistral-7B）可通过 HuggingFace 获取；论文未提供统一代码仓库声明。
- **关键超参**：探针正则化系数 $C \in \{0.1, 1, 10\}$；干预强度 $\alpha \in \{-4, -2, -1, 0, 1, 2, 4\}$；层选择基于 pair-aware CV 在验证集上最大化 AUROC；bootstrapping 2000 次 pair-level resamples。
