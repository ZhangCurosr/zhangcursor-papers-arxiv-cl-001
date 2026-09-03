---
title: "FARCA-Fact-Aligned-Reliability-Aware-Credit-Assignment-for-R"
source: https://arxiv.org/pdf/2608.24350v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 14:51:27"
field: "大语言模型事实性强化学习"
keywords: ["Reinforcement Learning", "Factuality", "Hallucination", "Credit Assignment", "Token-level Policy Optimization", "Counterfactual Attribution", "RLVR"]
innovations: ["提出事实–token对齐与反事实证据归因联合框架，解决事实监督中的信用定位与可靠性模糊问题", "将NLI验证信号转化为连续符号事实分数并通过可靠性权重软插值重塑token级优势", "在保持通用推理能力的同时显著提升知识密集型任务的事实性"]
benchmarks: ["SimpleQA", "TruthfulQA", "HalluQA", "HaluEval-QA", "AIME2026", "AIME2025", "MATH-500", "GSM8K"]
---

# 论文速读：FARCA-Fact-Aligned-Reliability-Aware-Credit-Assignment-for-R

## 一句话总结
论文提出FARCA框架，解决强化学习（RLVR）中事实监督的“噪声化信用分配”问题。通过原子事实与token级溯源对齐，以及反事实证据归因评估可靠性，将粗糙的事实验证信号转化为精细、可靠的token级训练信号，在提升模型事实性的同时保持通用推理能力。

## 研究问题与动机
- **结果奖励加剧幻觉**：基于verifiable rewards的RL（RLVR）只评估最终答案正确性，在知识密集型任务中会强化“事实错误但答案正确”的推理轨迹，放大幻觉风险。
- **现有事实监督方法的粗糙聚合**：KnowRL、FSPO、FaithRL等方法将事实验证信号聚合到轨迹级或推理步级，存在粒度错配：验证单元（原子事实/步）与策略更新单元（token）不一致，导致同一信号内正确与错误事实共享信用，产生“信用定位模糊”。
- **验证信号可靠性缺失**：现有方法直接使用NLI验证器输出作为监督，未评估验证结果的可信度；验证器易受语言先验或无关片段误导，其误判会污染策略梯度，造成“信用可靠性模糊”。
- **实证观察**：对DeepSeek-R1-Distill-Qwen-7B在2WikiMultiHopQA上的rollout分析显示，混合正确性原子事实普遍存在，信用聚合会为部分事实标注赋予与真实标签相反的符号。

## 核心贡献（创新点）
1. **识别并形式化“噪声化事实信用分配”问题**，将其分解为信用定位模糊与信用可靠性模糊两个相互关联的方面，指出有效事实优化需同时满足精确的定位与可靠的监督信号。
2. **提出FARCA统一框架**，通过事实–token对齐解决定位模糊，通过反事实证据归因解决可靠性模糊，将验证信号转化为局部化、可靠性加权token级训练信号，使事实监督能更忠实、稳健地指导策略优化。
3. **实验验证有效性**：在多个模型（Qwen2.5-3B-Instruct、Llama-3.2-3B-Instruct）和多个事实推理基准上，FARCA显著提升模型事实性，同时保持并增强了数学推理能力，且优于现有事实强化学习基线（KnowRL、FSPO、FaithRL）。

## 方法详解
- **原子事实提取与Token溯源（Token Provenance）**：将推理文本分割为句子，过滤无验证内容的句子；用GPT‑4o将每个可验证句子分解为语义自包含的原子事实集合，同时进行去上下文化改写；要求GPT‑4o返回每个原子事实的来源token跨度及索引集$\mathcal{T}(c_{i,j,k})$，即token溯源。随后应用基于规则的检查与校正（处理重复实体、代词重写、并列结构等），丢弃无法定位来源的原子事实。
- **原子事实验证**：将事实验证建模为NLI任务，给定知识片段$K$和原子事实$c_{i,j,k}$，验证器输出$h_{i,j,k}\in[0,1]$；FARCA将其线性映射为连续符号事实分数$r_{i,j,k}=2h_{i,j,k}-1\in[-1,1]$，保留验证强度信息，避免离散阈值带来的信号损失。
- **反事实证据归因与可靠性估计**：将证据句子$K$与原子事实进行语义相似度计算，选取最相关的$K_{rel}$个证据句子作为证据锚点；移除这些句子后重新验证，得到新分数$\tilde{r}_{i,j,k}$；计算符号分数变化$\Delta_{i,j,k}=|r_{i,j,k}-\tilde{r}_{i,j,k}|$作为验证结果对关键证据的依赖程度。依赖程度越大，可靠性越高。将$\Delta$通过sigmoid映射为连续可靠性权重$w_{i,j,k}=\text{sigmoid}((\Delta_{i,j,k}-\mu)/\tau)$，其中$\mu$为校准集中位数，$\tau$控制映射平滑度。最终可靠性加权事实分数为$\tilde{r}_{i,j,k}^{\text{fact}}=w_{i,j,k}r_{i,j,k}$。
- **奖励设计**：每个rollout由三部分组成：格式奖励$R_i^{\text{format}}$（符合指定格式+1，否则-1）、答案奖励$R_i^{\text{answer}}$（答案正确+1，否则-1）、可靠性加权事实奖励$R_i^{\text{fact}}$（轨迹内所有原子事实的$\tilde{r}_{i,j,k}^{\text{fact}}$均值）。总奖励$R_i=R_i^{\text{format}}+R_i^{\text{answer}}+R_i^{\text{fact}}$。
- **可靠事实引导的优势重塑（Advantage Reshaping）**：在GRPO基础上进行组内归一化得原始优势$A_i$；对每个原子事实构造事实校正优势$A_{i,j,k}^{\text{fact}}=r_{i,j,k}|A_i|$，用$|A_i|$保留相对训练强度，用$r_{i,j,k}$决定更新方向。然后通过可靠性权重进行软插值：$\bar{A}_{i,j,k}=(1-w_{i,j,k})A_i+w_{i,j,k}A_{i,j,k}^{\text{fact}}$，实现全局信用与局部事实修正的权衡。最终对每个token $t$，将其覆盖的所有原子事实的$\bar{A}_{i,j,k}$取平均作为token级优势$\hat{A}_{i,t}$。
- **训练目标**：将$\hat{A}_{i,t}$代入token‑level PPO‑clip目标，结合KL散度正则化，完成参数更新。

## 实验与结果
- **数据集**：训练数据使用HotpotQA和2WikiMultiHopQA的挑战性子集（共8,498个知识密集型问答样本）以及SimpleRL数学推理数据集（8,523题）；评估使用SimpleQA、TruthfulQA、HalluQA、HaluEval‑QA（事实性/幻觉）和AIME2026、AIME2025、MATH‑500、GSM8K（数学推理）。
- **基线**：Zero‑shot prompting、GRPO（仅数学/完整数据）、KnowRL、FSPO、FaithRL。
- **主要结果**：在Qwen2.5‑3B‑Instruct上，FARCA相比最强事实基线FaithRL，在四个幻觉基准上平均提升1.75个百分点（TruthfulQA +2.09，HalluQA +2.67），同时在数学推理基准上取得最优（AIME2026 6.67→10.00，GSM8K 84.46）；在Llama‑3.2‑3B‑Instruct上平均提升进一步扩大至2.21个百分点。
- **对比基线劣势**：KnowRL和FSPO虽引入事实监督，但因粗糙聚合在多数据集上劣于标准GRPO（Full），表明信用定位模糊会削弱优化效果；FaithRL采用步级监督仍视整个推理步为不可分单元，性能亦不及FARCA。
- **消融实验**：移除token溯源（w/o token provenance）使平均幻觉分数从25.36降至24.40；移除可靠性估计（w/o reliability estimation）下降明显；使用离散事实分数（w/o continuous factual score）亦有小幅下降，证明连续分数提供更平滑的监督信号。
- **超参敏感性**：可靠性温度$\tau$在$\{0.10,0.20,0.30\}$区间内平均分数差异仅0.16个百分点，框架对$\tau$设置鲁棒，默认取$\tau=0.20$。
- **进一步分析**：约98.3%的原子事实成功对齐至源token跨度；36.3%的事实$\Delta>\mu$，仅1.1%触发fallback，平均权重0.512，表明可靠性信号稳定非退化；可靠性软插值使约49.6%的span保持原优势方向，39.5%保持方向但重新缩放幅度，10.7%发生反向纠正，有效混合局部事实反馈与全局信用。

## 相关工作脉络
1. **KnowRL**：将推理过程分解为原子事实，聚合为轨迹级事实奖励；但未进行token级定位，且直接使用验证器输出，忽视可靠性评估。
2. **FSPO**：在推理步级进行事实验证，用验证结果调整策略优势；仍以步为单位广播信用，无法区分步内不同事实的正确性。
3. **FaithRL**：在步级验证推理faithfulness并抑制无证据支持的步骤；但同样将推理步视为不可分单元，且未量化验证信号可靠性。
4. **RLVR (DeepSeek‑R1, OpenAI o1等)**：利用verifiable outcome rewards激发复杂推理能力；仅关注最终答案，不验证中间事实，导致幻觉风险。
5. **事实一致性评估方法**（AlignScore、MiniCheck、HHEM等）：多为静态验证器，本文将其集成到RL循环并引入可靠性加权机制，实现动态信用分配。
6. **Token‑level policy optimization**：传统PPO按token计算策略梯度；FARCA在此基础上注入事实信号，实现细粒度、可靠性的监督融合。

## 局限性与未来方向
- **事实提取依赖大语言模型**：原子事实抽取使用GPT‑4o，可能引入成本与延迟，且抽取质量受提示模板影响；可探索更轻量、领域适配的抽取器。
- **证据依赖性假设**：可靠性估计基于“移除关键证据应改变验证结果”的假设，但若验证器依赖非证据因素（如语言先验）较少，则$\Delta$可能较小，导致可靠性权重偏低，低估真实可靠信号。
- **计算开销增加**：反事实证据归因需对每个原子事实额外运行一次验证，且需维护证据检索模块；在长推理轨迹中可能显著增加训练时间。
- **仅针对知识密集型任务**：当前实验主要聚焦于问答类幻觉评估；对于开放式生成、创意写作等场景的幻觉缓解效果尚未验证。
- **超参数需校准**：可靠性权重映射中的$\mu$和$\tau$依赖校准集，不同数据集分布可能需重新调整；可探索自适应估计方法。

## 研究启发与可借鉴点
1. **粒度对齐思想**：将验证粒度（原子事实）与策略更新粒度（token）对齐，解决信用分配模糊问题；该思路可迁移至其他过程监督任务（如代码生成、多步规划），实现更精准的信用溯源。
2. **反事实归因评估可靠性**：通过移除关键证据观测验证结果变化，以证据依赖程度作为可靠性代理；此方法无需额外标注，可推广至其他依赖外部知识的验证场景（如法律、医疗问答）。
3. **软插值信用融合机制**：以可靠性权重在原始优势与事实校正优势间进行凸组合，避免硬翻转带来的梯度震荡；类似平滑混合策略可用于其他噪声监督信号的去噪过程。
4. **连续符号分数替代离散阈值**：将NLI输出线性映射为$[-1,1]$连续分数，保留支持强度信息，提供更平滑的监督信号；可替代硬标签用于任何基于分类器的信用分配任务。
5. **训练‑评估分离设计**：同时使用知识密集型数据优化事实性、数学数据保持通用推理能力，避免过拟合单一领域；该数据配比策略对多目标强化学习具有参考价值。

## 关键术语表
- **FARCA**：Fact‑Aligned Reliability‑Aware Credit Assignment，论文提出的政策优化框架，实现细粒度、可靠性加权的事实信用分配。
- **RLVR**：Reinforcement Learning with Verifiable Rewards，基于可验证结果奖励的强化学习范式，常见于数学与推理任务。
- **Token Provenance**：Token溯源，指原子事实与其在推理文本中源token跨度的可追踪映射关系，决定事实信号的定位范围。
- **Credit Localization Ambiguity**：信用定位模糊，因验证粒度与策略更新粒度不匹配，导致同一信号覆盖正确与错误事实的问题。
- **Credit Reliability Ambiguity**：信用可靠性模糊，因验证器输出本身存在不确定性，未评估其可靠性即直接用于策略更新的问题。
- **Counterfactual Evidence Attribution**：反事实证据归因，通过移除关键证据观察验证分数变化，以此估计验证结果对证据的依赖程度作为可靠性代理。
- **Continuous Signed Factual Score**：连续符号事实分数，将NLI验证输出$h\in[0,1]$线性映射至$r=2h-1\in[-1,1]$，保留支持强度信息。
- **Reliability‑Weighted Advantage Reshaping**：可靠性加权优势重塑，以证据依赖性计算的可靠性权重，在原始优势与事实校正优势间进行软插值。

## 可复现要素
- **数据集**：训练数据（HotpotQA、2WikiMultiHopQA挑战子集，SimpleRL）未公开链接；评估基准（SimpleQA、TruthfulQA、HalluQA、HaluEval‑QA、AIME2026/2025、MATH‑500、GSM8K）均为公开数据集。
- **代码/权重**：论文未提及代码开源情况；模型权重未提供。
- **关键超参数**：学习率$5\times10^{-7}$，rollout数$G=6$，temperature=1.0，PPO mini‑batch size=128（per‑GPU batch=2），KL系数=0.001，最大prompt/response长度=2048；可靠性映射中$\tau=0.20$，$\mu=0.16$（校准集$\Delta$中位数），$K_{rel}=1$。
