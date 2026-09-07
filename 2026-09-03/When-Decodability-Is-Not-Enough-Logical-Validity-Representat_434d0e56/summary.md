---
title: "When-Decodability-Is-Not-Enough-Logical-Validity-Representat"
source: https://arxiv.org/pdf/2609.02438v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 00:33:03"
field: "语言模型内在表征与可解释性"
keywords: ["逻辑验证", "表示可解码性", "线性探针", "因果干预", "分布外泛化", "内在表征", "语言模型解释性"]
innovations: ["提出行为表达-表示可及性-因果控制三层分离框架", "在行为接近随机的条件下揭示有效性信息仍高度可解码且 persist on errors", "通过规范匹配随机方向对照证明高探针AUROC不等于因果控制变量"]
benchmarks: ["自制受控逻辑验证数据集（800示例）", "Pythia-1.4B/2.8B", "SmolLM3-3B", "Llama-3.2-3B", "Mistral-7B"]
---

# 论文速读：When-Decodability-Is-Not-Enough-Logical-Validity-Representations,-Behavioral-Dissociation,-and-Causal-Tests-in-Language-Models

## 一句话总结
该论文在受控的逻辑验证任务中发现，尽管五个开源 Transformer 模型的行为表现接近随机，逻辑有效性信息仍能从其隐藏状态中以近完美精度线性解码，且在分布外条件下保持高度可解码性；然而沿探针导出的有效性方向进行的因果干预仅产生微弱且不一致的输出变化，由此得出"表示可及性 ≠ 行为表达 ≠ 因果控制"的核心结论。

## 研究问题与动机
1. **行为准确性与内部表征的割裂**：模型在逻辑验证任务上给出正确/错误答案，仅凭行为输出无法判断其内部是否真正编码了逻辑关系，正确预测可能源于词汇/语义捷径而非真正的逻辑运算。
2. **现有探针研究缺乏因果检验**：此前的内部表示工程研究（如 Bertolazzi et al. [1]、Marks & Tegmark [8]）证明了逻辑有效性或真值可从激活中解码，但大多未系统检验分布外泛化边界，也缺少因果干预证据。
3. **行为错误时内部表征是否消失未知**：当模型给出错误答案时，逻辑有效性相关信息是否完全缺失于隐藏状态，这一问题此前未有受控实验回答。
4. **可解码性不等于可用性**：即使某特征在隐藏状态中线性可分离，该特征是否被模型实际用于决策，缺乏从探针方向到行为输出的因果链路验证。

## 核心贡献（创新点）
1. **构建受控有效-无效匹配数据集**：设计了 800 个样本（400 对）的逻辑验证数据集，每对内前提、推理族、领域、模板和难度完全一致，仅改变主张的有效性标签，从而剥离了语境混淆因素；与已有工作（如 FOLIO、LogicBench）相比，本文强调"关系"而非"内容"的可分离性。
2. **三层证据框架分离可解码性、行为表达与因果控制**：首次在同一任务上同时测量输出行为（准确率、边距 AUROC）、线性探针可解码性（各层 AUROC、泛化）、以及激活干预效应（沿探针方向的边距变化），三者结论相互独立；与 Sahoo et al. [12] 类似地挑战探针性能的过度解读，但本文进一步加入因果干预维度。
3. **揭示"行为错误时表征仍存"现象**：在行为错误样本上（如 Llama-3.2 的 incorrect 子集），有效性信息仍以 AUROC≈1.000 可解码，打破了"错误答案意味着内部表征缺失"的直觉假设。
4. **系统性 OOD 泛化分析揭示异质性边界**：通过留一域/留一家族分析发现，通用性并非均匀分布——三段论（syllogism）是所有模型最弱的 OOD 转移对象，且各模型域级弱点存在显著差异。
5. **阴性因果干预结果限定探针含义**：沿探针有效性方向的干预产生的边距变化（约 ±0.002–0.004）与随机正交方向相当，证明高探针 AUROC 不能直接解释为"模型在用该方向做决策"。

## 方法详解
1. **数据集构造**：共 800 个示例，分为 400 对匹配的有效-无效对；覆盖 5 个推理族（syllogism、transitivity、set inclusion、causal chain、permission logic）、5 个语义域（nonce、spatial、social、biological、legal-policy）和 3 个难度层级（1步直接验证、2步推理链、含无关干扰前提的2步链）。严格模板划分：训练集600、测试集200，保证完整模板不跨集。
2. **行为评估**：对每个模型计算 VALID 和 INVALID 的条件对数似然分数 $s_V(x)$ 和 $s_I(x)$，预测为 $\hat{y} = \mathbb{I}[s_V(x) > s_I(x)]$，输出边距 $M(x) = s_V(x) - s_I(x)$；对答案标签的 tokenization 进行审计以确保公平比较。
3. **分层线性探针**：在每个 Transformer 块的最终 prompt-token 隐藏状态 $h_{i,\ell}$ 上拟合 $\ell_2$ 正则化逻辑回归探针预测有效性标签，正则化系数 $C \in \{0.1, 1, 10\}$；最优层 $\ell^* = \arg\max_\ell \text{AUROC}_\ell^{\text{val}}$，仅在训练分区选择层与超参后，在测试分区单次评估。
4. **OOD 泛化协议**：四种主要划分（random、template-held-out、domain-held-out、family-held-out）加上穷举留一域/留一家族分析；匹配对在整个划分、验证和 bootstrap 过程中始终保持在一起。
5. **配对准确性与正确性条件分析**：配对准确率 $\text{PairAcc} = \frac{1}{P}\sum_p \mathbb{I}[q(i_V) > q(i_I)]$ 衡量模型是否在每对内保持有效>无效的排序；正确性条件 AUROC 在模型同时产出两种金标签答案的子集上评估。
6. **控制实验**：TF-IDF 分类器（full prompt / claim only / premises only）、metadata-only 分类器、200 次随机标签置换；所有探针结果均通过 pair-level bootstrap 计算 95% CI。
7. **因果干预**：将探针权重 $\tilde{v} = w_{\ell^*} \oslash s_{\ell^*}$ 归一化为 $v$，干预尺度 $\sigma_v = \text{SD}_{i\in\text{train}}(h_{i,\ell^*}^\top v)$，修改隐藏状态 $h'_{i,\ell^*} = h_{i,\ell^*} + \alpha\sigma_v v$，其中 $\alpha \in \{-4, -2, -1, 0, 1, 2, 4\}$；与 5 个规范匹配的随机正交方向比较，并额外实施 matched-projection patching。

## 实验与结果
- **模型**：Pythia-1.4B、Pythia-2.8B、SmolLM3-3B、Llama-3.2-3B、Mistral-7B。
- **行为表现**：所有模型准确率接近 0.500（随机水平），Pythia-1.4B 和 SmolLM3-3B/Mistral-7B 出现极端答案标签偏好（全部预测 INVALID 或 VALID），Llama-3.2 为 0.470；margin AUROC 同样近随机（0.458–0.576）。
- **随机划分探针 AUROC**：全部模型达到 ~1.000（Pythia-1.4B: 1.000, Pythia-2.8B: 1.000, SmolLM3-3B: 1.000, Llama-3.2: 1.000, Mistral-7B: 1.000）。
- **OOD 泛化**：Template-held-out AUROC：0.963–0.999；Domain-held-out：0.771（Mistral-7B legal-policy 最弱）– 0.991（Llama-3.2）；Family-held-out：0.938–0.995；**Syllogism 留一效应最强**：所有模型在 syllogism held-out 时 AUROC 降至 0.500–0.762。
- **配对准确率**：所有模型在所有四种主要划分下 PairAcc = 1.000（95% CI [1.000, 1.000]）。
- **正确性条件 AUROC**（Llama-3.2）：correct 子集 0.970–1.000，incorrect 子集 0.994–1.000；Pythia-2.8B domain holdout 下 correct=0.965，incorrect=1.000。
- **控制实验**：Full prompt TF-IDF random=0.970，claim-only=0.965，下降至 template OOD=0.855/0.682；premises-only 和 metadata-only 均为 ~0.500；200 次置换 null AUROC mean=0.496–0.499，p=1/201≈0.005。
- **因果干预**（α=+4）：ΔM 分别为 −0.0037（Pythia-2.8B）、−0.0022（Llama-3.2）、+0.0023（Mistral-7B），无预测翻转；随机方向均值 |ΔM| 0.0058–0.0077，大于或等于探针方向。Matched-projection patching 边距变化仅 0.0011–0.0045。

## 相关工作脉络
1. **Bertolazzi et al. [1]**：研究逻辑有效性 vs. 语义合理性在三段论推理中的线性表示及因果操控；本文在其基础上进一步考察分布外泛化边界和错误行为下的表征存留，并提出更强的阴性因果结果。
2. **Sahoo et al. [12]**：指出线性探针检测的可能是任务格式而非推理模式；本文通过 TF-IDF、metadata、shuffled-label 等多重控制呼应这一观点，但以因果干预实验为补充证据。
3. **Marks & Tegmark [8] / Burns et al. [2]**：证明真值等高层变量可从 LM 激活中线性恢复；本文聚焦"逻辑有效性"这一关系属性（而非绝对真值），并扩展至因果控制层面检验。
4. **Han et al. [3] (FOLIO) / Parmar et al. [10] (LogicBench)**：行为基准衡量推理能力；本文以诊断性数据集补充，强调行为准确率无法反映内部表征质量。
5. **Li et al. [6] / Nanda et al. [9]**：在 Othello-GPT 和自监督序列模型中通过探针+干预揭示内部状态变量；本文沿用类似框架但应用于逻辑验证，且因果干预结果为阴性。
6. **Zou et al. [18] (Representation Engineering)**：代表工程通过激活方向操控模型行为；本文结果表明，简单线性探针方向不足以实现可靠的因果行为控制。

## 局限性与未来方向
1. **数据集为合成受控数据**：800 个样本和 5 个推理族/领域仅覆盖逻辑推理空间的极小部分，结论推广到自然语言长上下文和开放推理任务的能力有限。
2. **仅考察线性探针和最终 prompt-token 层**：有效性信息可能以非线性方式编码、跨 token 位置分布，或在其他计算节点表征，本文方法未能捕捉。
3. **阴性因果干预不排除整体因果相关性**：探针方向的弱干预效应仅说明该单一线性方向不是充分控制变量，不能推断有效性信息对整个模型计算完全因果无关。
4. **模型规模偏小**：所有模型为 ≤7B 参数开源模型且经量化，更大模型或不同架构下的模式未验证。
5. **难度分级未产生行为梯度**：Difficulty 1/2/3 构造复杂度并未对应单调行为劣化，提示任务设计对行为评估的敏感度不足。

## 研究启发与可借鉴点
1. **三层证据框架的普适性**：将"行为表达→表示可及性→因果控制"作为解 interpretability 研究的标准化流程，可作为团队后续研究逻辑/推理能力的基线评估范式。
2. **匹配对设计的价值**：在每对内固定所有构造变量仅改变目标标签，是剥离 confound 的高效策略，可迁移至其他关系型推理研究（如蕴含、等价、矛盾检测）。
3. **阴性因果干预结果的方法学意义**：证明高探针 AUROC 必须配以干预实验才能得出有意义的 interpretability 结论；团队在设计 probe-based steering 实验时需同步设置随机方向对照和充分干预强度扫描。
4. **OOD 异质性分析的诊断价值**：留下 oone-domain-out / leave-one-family-out 的细粒度分析比汇总 AUROC 更能揭示表征的真实结构，建议在未来泛化研究中采用同等细致的划分策略。
5. **行为错误时的表征存留**：为 "representation vs. execution" 争议提供了实证数据，启发团队在评估推理模型时不应仅看准确率，而应同时测量错误样本上的内部表征质量。

## 关键术语表
**Logical Validity**：前提与主张之间的形式关系属性，即主张是否必然由前提推导得出，独立于主张的事实真值。
**Matched Valid–Invalid Pair**：前提、推理结构、领域、模板完全相同的一对示例，仅主张的有效性标签相反，用于消除语境混淆。
**Probe AUROC**：用线性分类器（逻辑回归探针）从模型隐藏状态中预测逻辑有效性标签的 ROC 曲线下面积，衡量线性可分离性。
**Output Margin M(x)**：模型对 VALID 和 INVALID 标签的条件对数似然之差，$M(x)=s_V(x)-s_I(x)$，衡量行为置信度与倾向。
**Leave-one-out Generalization**：逐一排除某个领域或推理族后评估探针性能，揭示表征在不同分布偏移下的异质性泛化能力。
**Correctness-conditioned Evaluation**：按模型行为正确/错误将样本划分为子集，分别评估有效性信息的可解码性。
**Activation Intervention**：沿探针导出的方向对特定层的隐藏状态施加偏移，观察输出边距的变化以检验因果影响。
**Norm-matched Random Direction**：与探针方向范数相同但正交于探针方向的随机扰动向量，作为因果干预实验的阴性对照基线。

## 可复现要素
- **数据集**：论文声明为合成数据集，未提供公开链接；需按附录 A 描述的方法手动复现（800 个示例，400 对匹配有效-无效对）。
- **代码/权重**：模型权重来自 Hugging Face（EleutherAI/pythia-1.4b、EleutherAI/pythia-2.8b、HuggingFaceTB/SmolLM3-3B-Base、meta-llama/Llama-3.2-3B、mistralai/Mistral-7B-v0.3）；代码论文未明确声明开源链接，需自行实现或联系作者获取。
- **关键超参**：探针正则化系数 $C \in \{0.1, 1, 10\}$；干预尺度 $\alpha \in \{-4, -2, -1, 0, 1, 2, 4\}$；bootstrap 采样次数 2000，置换次数 200；pair_id 作为分组变量用于交叉验证和 resampling。
- **硬件/量化**：模型以量化版本评估，具体量化方式论文未详细说明。
