---
title: "The-Asymmetric-Harms-of-LLM-Compression"
source: https://arxiv.org/pdf/2608.19670v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 12:37:33"
field: "大语言模型压缩与可信部署"
keywords: ["LLM压缩", "知识保留", "模型置信度", "社会偏见", "量化", "剪枝", "评估基准"]
innovations: ["提出相对保留偏移(RS)指标揭示head知识比例损失大于tail", "发现压缩后模型对丢失知识保持中高置信度（0.4-0.6）", "揭示aggregate bias稳定可掩盖子群极端偏移（如secretary -53.1pp vs整体-2.2pp）"]
benchmarks: ["PopQA", "Head-to-Tail", "WinoBias", "BBQ", "WikiText-2"]
---

# 论文速读：The Asymmetric Harms of LLM Compression

## 一句话总结
本文系统评估了11种LLM压缩方法（4种量化+7种剪枝）在3个开源模型（Llama-3.1-8B-Instruct、Qwen-3-8B、Gemma-2-9B-it）上的行为影响，发现压缩对知识的破坏具有**不对称性**：头部知识（head）相对尾部知识（tail）损失更严重，且压缩后模型常对被遗忘的知识点保持中等以上置信度，整体偏见分数的稳定可能掩盖不同人口统计子群间的相反偏移。

## 研究问题与动机
- **问题1（RQ1）**：压缩是否不成比例地削弱模型对低频事实（tail knowledge）的保留？现有研究多聚焦accuracy单一指标，缺乏跨流行度分组的细粒度分析。
- **问题2（RQ2）**：压缩导致知识丢失后，模型在新错误答案上的置信度如何？是否有"过度自信"现象？
- **问题3（RQ3）**：压缩对刻板偏好的影响是否在不同人口统计子群间呈现异质性？整体bias score的稳定性可能掩盖子群层面的相反偏移。
- **现有方法不足**：aggregate metrics（perplexity、平均accuracy、整体bias）无法捕捉上述非对称行为变化；已有结果碎片化且相互矛盾（如some研究显示pruning放大underrepresented subgroup的error，others报告moderate quantization改善fairness）。

## 核心贡献（创新点）
- **系统化评估框架**：统一protocol评估11种压缩方法×3个模型×2个知识流行度benchmark（PopQA、Head-to-Tail）+ 2个bias benchmark（WinoBias、BBQ），填补了跨方法比较的空白。
- **相对保留偏移指标（Relative Retention Shift, RS）**：提出$(r_g - r) \times 100$量化各分组相对整体保留的偏离程度，揭示head知识的**比例损失更大**这一aggregate accuracy掩盖的现象。
- **置信度-知识丢失联合分析**：定义"lost-knowledge set"并测量其在不同流行度分组上的confidence分布，发现压缩后模型对已错误答案仍保持0.4–0.6的median confidence。
- **bias子群异质性揭示**：展示aggregate bias change（如$\Delta B = -2.2$pp）可掩盖单一occupation subgroup的极端变化（如secretary $\Delta B_g = -53.1$pp）。

## 方法详解
**评估指标体系**：
- **RQ1（知识保留）**：
  - 分组准确率：$\text{Acc}_g = \frac{1}{N_g}\sum_{i \in \mathcal{T}_g} \text{Match}(M(x_i), y_i)$
  - 保留率：$r_g = \text{Acc}_{c,g} / \text{Acc}_{0,g}$
  - 相对保留偏移：$\text{RS}_g = (r_g - r) \times 100$（正=优于整体，负=劣于整体）
- **RQ2（丢失知识的置信度）**：
  - 知识丢失率：$\text{LR}_g = |\mathcal{T}_{\text{loss},g}| / N_g$
  - 丢失知识的置信度：$\text{Conf}(x_i) = p_M(\hat{y}_i|x_i)^{1/T_i}$（length-normalized sequence probability）
  - ECE：$\text{ECE}_g = \sum_{k=1}^K \frac{|B_{k,g}|}{N_g}|\text{Acc}(B_{k,g}) - \text{Conf}(B_{k,g})|$
- **RQ3（偏见变化）**：
  - 整体偏见变化：$\Delta B(M_c) = B(M_c) - B(M_0)$
  - 子群偏见变化：$\Delta B_g(M_c) = B_g(M_c) - B_g(M_0)$

**压缩方法**：
- 量化：GPTQ、AWQ、OmniQuant、AQLM（2/3/4-bit）
- 剪枝：Magnitude、WANDA、SparseGPT（unstructured 30/50/70%；semi-structured 4:8/2:4）、ShortGPT（5/10/15/20/25% layer dropping）

**校准**：对confidence使用isotonic regression校准（20/80 split），基于substring-based correctness indicator训练。

## 实验与结果
**数据集**：
- PopQA：~14K questions，按subject-entity page-view count分head/middle/tail三组
- Head-to-Tail：18,171 QA pairs，按popularity signals分组
- WinoBias：3,160 sentences，性别-职业刻板关联
- BBQ：58,492 examples，9个social dimensions + 2个intersectional categories

**关键发现**：
- **RQ1**：PopQA上105个model-configuration组合中，104个显示tail shift为正、head shift为负（唯一例外：4-bit AWQ on Qwen）。Tail保留率**系统性高于**head，剪枝比量化产生更大偏移（N:M pruning tail shift +22.7~+39.5pp，head shift -3.6~-14.4pp）。ShortGPT在10-15%层数时不对称性最大，25%时因整体collapse而收窄。
- **RQ2**：即使 mild compression（4-bit GPTQ、30% WANDA、5% ShortGPT）已造成22-35%的知识丢失；丢失知识的median confidence达0.46-0.60，且head知识丢失后的置信度常高于tail（Llama在4-bit GPTQ下head/tail confidence比1.20×）。强压缩下confidence和accuracy同时collapse（如2-bit GPTQ下confidence≈0.03），但ECE改善可能仅反映collapse而非可靠性提升。
- **RQ3**：aggregate $\Delta B$可接近0但子群呈相反偏移（如3-bit OmniQuant下male +6pp、female -4pp）；单个occupation subgroup变化可达±50pp（如Llama-SparseGPT 2:4下secretary $\Delta B_g = -53.1$pp，而$\Delta B = -2.2$pp）；BBQ上disability subgroup（如Down's syndrome）在ShortGPT 10%下$\Delta B_g = +37.5$pp。

## 相关工作脉络
- **LLM压缩方法**：GPTQ（Frantar et al., 2023）、AWQ（Lin et al., 2026）、OmniQuant（Shao et al., 2024）、AQLM（Egiazarian et al., 2024）、WANDA（Sun et al., 2024b）、SparseGPT（Frantar & Alistarh, 2023）、ShortGPT（Men et al., 2024）。本文首次在同一protocol下系统比较这11种方法的行为影响。
- **压缩与公平性**：Hooker et al.（2020）、Tran et al.（2022）显示pruning/quantization放大underrepresented subgroup错误；Kamal & Talbert（2024）、Hong et al.（2024）报告moderate quantization改善fairness。本文揭示这些矛盾源于aggregate score的掩盖效应。
- **知识流行度评估**：PopQA（Mallen et al., 2023）、Head-to-Tail（Sun et al., 2024a）首次用于系统分析压缩对不同流行度知识的非对称影响。
- **校准与置信度**：Zhang et al.（2024）指出LLM常对错误答案过度自信；本文首次将confidence-on-lost-knowledge作为压缩评估维度。
- **定位差异**：Chang et al.（2025a）仅评估单方法单bit width的accuracy；本文提供跨方法、跨compression level、跨模型、跨metric的完整画像。

## 局限性与未来方向
- **模型规模**：仅评估8-9B instruction-tuned模型，结论未必generalize到更大模型或其他架构（如MoE）。
- **压缩范式**：未覆盖distillation、routing等压缩方法。
- **RQ3校准缺失**：因WinoBias/BBQ无natural correctness target，stereotypical-preference scores未做isotonic calibration。
- **未来方向**：探索compression-aware debiasing方法；建立fine-grained evaluation protocol作为deployment prerequisite；分析confidence-on-lost-knowledge与downstream trustworthiness的关联。

## 研究启发与可借鉴点
- **评估协议迁移价值**：统一RS指标和lost-knowledge confidence分析可复用于其他压缩参数（如MoE routing、adapter pruning）的行为评估。
- **实验设计借鉴**：20/80 popularity-stratified calibration split确保confidence估计的稳健性；95% CI报告增强subgroup shift的可解释性。
- **可结合本团队方向**：在知识图谱增强LLM或参数高效微调（PEFT）场景中，可引入RS指标评估"头部知识 vs. 长尾知识"的非对称退化；将confidence-on-lost-knowledge作为trustworthy deployment的early warning signal。
- **方法论启发**：aggregate metric的"稳定"可能是虚假安全信号，需配合subgroup-level analysis才能检测compression-induced harm。

## 关键术语表
- **Relative Retention Shift (RS)**：分组保留率与整体保留率的差值（百分点），正值表示该分组相对整体受损更少，负值表示受损更多。
- **Lost-knowledge set**：base model正确但compressed model错误的样本集合，用于分析压缩导致的新错误。
- **Length-normalized confidence**：$\text{Conf}(x_i) = p_M(\hat{y}_i|x_i)^{1/T_i}$，消除答案长度对序列概率的影响。
- **Isotonic regression calibration**：用于将raw confidence映射到校准后概率的非参数回归方法。
- **Collapsed setting**：accuracy接近零且perplexity急剧上升的极端压缩配置（如2-bit GPTQ、70% WANDA）。
- **Stereotypical preference score**：模型给予stereotypical answer vs. reference answer的概率比值。
- **Aggregation masking**：aggregate score稳定但subgroup-level变化相反或极端隐藏的现象。
- **Popularity stratification**：按实体流行度（page views/traffic/ratings）将知识分为head/middle/tail分组。

## 可复现要素
- **数据集**：PopQA（公开）、Head-to-Tail（因license限制需通过官方pipeline生成）、WinoBias（公开）、BBQ（公开）、WikiText-2（公开）、C4（calibration dataset，公开）。
- **代码/权重**：论文未声明开源仓库，但基线方法（GPTQ/AWQ/OmniQuant/AQLM/WANDA/SparseGPT/ShortGPT）均有公开实现；三个base模型（Llama-3.1-8B-Instruct、Qwen-3-8B、Gemma-2-9B-it）均为open-weight。
- **关键超参**：quantization bit widths=2/3/4；pruning sparsities=30/50/70%（unstructured）、4:8/2:4（semi-structured）、5/10/15/20/25%（layer dropping）；calibration samples=128（quantization）/32（pruning）；sequence length=512。
