---
title: "Do-Cantonese-Adapted-Language-Models-Better-Predict-Cantones"
source: https://arxiv.org/pdf/2609.02163v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 16:46:32"
field: "计算 psycholinguistics / 低资源语言变体建模"
keywords: ["粤语眼动预测", "信息论指标", "词汇 surprisal", "熵减", "语言模型适配", "MCFIX", "CatBoost"]
innovations: ["配对族内轻量 vs 大规模粤语适配对比，揭示适配策略而非单纯规模决定预测拟合", "四项互补信息论指标联合评估，发现熵减最优模型不同于词汇 surprisal 最优模型", "共享候选词表构造跨模型可比 POS surprisal"]
benchmarks: ["MCFIX Cantonese"]
---

# 论文速读：Do-Cantonese-Adapted-Language-Models-Better-Predict-Cantones

## 一句话总结
本文利用粤语眼动数据（MCFIX），通过四个信息论指标（词汇 surprisal、词性 surprisal、目标词前熵、熵减）评估轻量与大规模粤语适配对语言模型预测人类阅读行为的增益，发现大规模适配（CantoneseLLM-7B）在词汇 surprisal 与联合指标上显著优于通用基座，但适配收益高度依赖指标类型与阅读任务。

## 研究问题与动机
- **核心问题**：粤语专用训练是否能让语言模型更好预测人类在线阅读行为？现有 NLP 粤语评估结果矛盾，尚缺从心理语言学预期角度的系统回答。
- **现有方法不足**：既有粤语 NLP 研究多聚焦 benchmark 绩效（NLU、推理、文化理解），缺乏对"模型概率预期与读者实时加工难度"对齐的评估；且区域/低资源方言的眼动预测研究明显不足。
- **适配策略差异未解**：轻量适配（词表扩展 + 小规模微调）与大规模持续预训练 + SFT 是否能同等提升预测拟合，尚不清楚。
- **指标选择单一风险**：既往工作多用单一词汇 surprisal，但不同信息论指标刻画了阅读过程中不同的认知阶段（目标前预期、目标词本身、目标后更新），值得联合评估。

## 核心贡献（创新点）
- **配对族内对比设计**：在相同架构下分别比较 CKIP↔JED351（小模型）与 Qwen2.5-7B↔CantoneseLLM-7B（大模型），隔离适配策略差异而非规模/架构差异。
- **四项互补信息论指标联合评估**：同时考察词汇 surprisal、POS surprisal、目标前熵与熵减，覆盖"目标前预期—目标词加工—目标后不确定性更新"三个阶段。
- **五种子稳健性 + 任务分层 + 留一块消融**：5-split 稳健性验证、NR/TSR 阅读任务分层、以及 LLM 指标块与语言学基线块的联合消融，揭示指标贡献的稳健性与任务敏感性。
- **粤语眼动预测的新证据**：证明大规模粤语适配（CantoneseLLM-7B）在词汇 surprisal 和联合四指标模型上系统性优于通用中文基座，而轻量适配（JED351）未显示一致增益，修正"适配即有效"的简单假设。
- **指标异质性发现**：熵减最优模型为 CKIP 而非 CantoneseLLM-7B，揭示"预测目标词概率更优≠建模输入引发的不确定性更新更强"，强调需多指标联合诊断。

## 方法详解
- **数据**：MCFIX 繁体粤语组件（《小王子》粤语译本），30 名母语者，含自然阅读（NR）与任务特定阅读（TSR）两种条件；眼动指标为词级 FFD、SFD、TFD（ms），共 10,011 行用于分析。
- **模型配对**：
  - 小模型对：CKIP GPT-2 Tiny（繁体中文基座）vs JED351（词表 patch + 约 50 MB 粤语维基百科微调 10 epochs）。
  - 大模型对：Qwen2.5-7B（多语言基座）vs CantoneseLLM-7B（大规模粤语持续预训练 + 75,000 对 SFT）。
- **四项信息论指标**（单位 bits）：
  1. **Lexical surprisal**：$S(w_t) = -\log_2 P(w_t | c_t)$，表征目标词的条件预测难度。
  2. **POS surprisal**：$S(z_t) = -\log P(z_t | c_t)$，聚合同 POS 候选词的总概率后取负对数，刻画句法类别预期。
  3. **Entropy before**：$H_{\text{before}} = -\sum_{v \in V} P(v|c_t) \log_2 P(v|c_t)$，目标词出现前的下一个 token 分布熵。
  4. **Entropy reduction**：$ER_t = H_{\text{before}} - H_{\text{after}}$，加入目标词后的下一 token 熵降，表征观察目标后对后续不确定性的削减。
- **POS surprisal 构造**：基于 Cifu 粤语词表统一候选集，用 PyCantonese 标注 POS，跨所有模型共享同一候选集合以保证可比性。
- **回归框架**：CatBoost 梯度提升树（1,000 轮迭代，内部 seed=10）。
  - Baseline（no-LLM）：阅读任务指示、POS 虚拟变量、线性距离到根/头节点、依赖深度、邻域大小、当前/先前词频、当前/先前音节数、相对词位置。
  - 单指标添加模型（每加一项 LLM 指标）、全四指标联合模型、留一预测块消融模型。
- **评估与稳健性**：词级 K-Fold（5 folds, seeds 42–46），以 MAE 为主，另保留 RMSE、$R^2$、Pearson/Spearman 相关；定义 $\Delta \mathrm{MAE} = \mathrm{MAE}_{\text{base}} - \mathrm{MAE}_{\text{aug}}$，并计算基线归一化百分比提升。

## 实验与结果
- **基线数据集**：MCFIX Cantonese 子集，5,007 NR + 5,004 TSR，3 个眼动指标（FFD/SFD/TFD）。
- **最强结果**：联合四指标模型中，CantoneseLLM-7B 表现最优——对 FFD、SFD、TFD 的 mean $\Delta$MAE 分别为 +0.729 ms、+0.267 ms、+1.251 ms；基线归一化后 FFD 最大相对提升达 1.991%。所有改善均稳定通过 5/5 seeds 正方向检验。
- **排序一致性**：词汇 surprisal 和联合四指标在全三个眼动指标上均为 CantoneseLLM-7B > Qwen2.5-7B > CKIP > JED351；POS surprisal 和 Entropy before 同样倾向 CantoneseLLM-7B。
- **指标特异性**：Entropy reduction 的最优模型为 CKIP（FFD +0.183、SFD +0.104、TFD +0.362 ms），显著区别于其他指标的排序。
- **适配策略对比**：JED351 vs CKIP 在联合模型上呈负向对比（无一致性增益）；CantoneseLLM-7B vs Qwen2.5-7B 在全部指标/种子下呈正向对比（FFD +0.132、SFD +0.066、TFD +0.430 ms）。
- **任务差异**：FFD 在所有模型上 TSR > NR 提升；但 SFD/TFD 在 CantoneseLLM-7B 上 TSR 更大，其余模型 NR 更大，CKIP 差异最显著（NR 的 SFD/TFD 联合提升分别为 +0.257 / +0.778 ms，TSR 为 +0.138 / +0.523 ms）。
- **消融结论**：当前音节数和相对词位置始终是最强预测块；LLM 指标块整体居中——CantoneseLLM-7B/Qwen2.5-7B 上词汇 surprisal 条件贡献最大，CKIP 上熵减最大，JED351 上 entropy before 最大；指标间存在部分重叠（entropy before 与 entropy reduction 相关 $\rho=0.55$–$0.67$），并非可互换。
- **整体判断**：LLM 指标提供稳定但幅度有限的增量预测（联合改善均 < 2% 基线 MAE），属于互补而非替代型贡献。

## 相关工作脉络
- **CMCL 系列共享任务**（Hollenstein et al., 2021a,b; 2022）：确立梯度提升 + 信息论指标预测眼动的范式，本文延续该框架并扩展至粤语与多指标联合评估。
- **多语言 surprisal 预测**（Wilcox et al., 2023; Pimentel et al., 2023）：11 语种与预期/熵效应研究为本文四项指标的选择提供理论依据。
- **中文眼动建模**（Li & Pollatsek, 2020; Rayner et al., 2007; Yan et al., 2006; Zang et al., 2018）：奠定中文书写系统与词长/频率效应的眼动理论基础。
- **MCFIX 中英眼动平行语料**（Li et al., 2023）：本文直接使用其粤语子集，并拓展到不同适配规模模型的对比。
- **粤语 NLP 评估**（CantoNLU, Cheng et al. HKCanto-eval, Jiang et al.）：揭示粤语适配的收益高度依赖任务类型，支撑本文选择"预测阅读行为"而非纯 NLU benchmark 作为评估角度。
- **模型规模与 surprisal 拟合**（Oh & Schuler, 2023; Oh et al., 2024; Alves, 2025）：英语/葡语研究发现规模增大可能恶化 surprisal 拟合，本文据此谨慎解释 CantoneseLLM-7B 的优势来自适配策略而非单纯规模。

## 局限性与未来方向
- **未隔离单一适配成分**：JED351 与 CKIP 在词表/嵌入/微调三方面均不同，CantoneseLLM-7B 与 Qwen2.5-7B 差异包含持续预训练与 SFT，无法精确归因到单一训练组件。
- **单一语料与样本**：仅使用《小王子》粤语译本和 30 名参与者，结果可迁移性受限；仅用传统字形。
- **Tokenization 不一致**：上下文与目标词分开 tokenize，与真实连续 tokenize 在 246/5,072 行产生边界差异。
- **跨模型熵不可直接比较**：不同 tokenizer 和续写空间下，熵值的绝对量纲不可跨模型直接等同。
- **未来方向**：扩大语料多样性和样本量；解耦词汇扩展、持续预训练、SFT 的贡献；将指标对比扩展至更多低资源汉语变体（如闽南语、吴语）；探索将多指标联合用于实时阅读辅助系统。

## 研究启发与可借鉴点
- **配对族内对比设计**：保持架构固定、仅改变适配策略，是控制混杂变量的有效范式，可迁移到汉语其他方言或低资源语言的模型评估中。
- **多信息论指标联合诊断**：不同指标（surprisal / entropy / entropy reduction）可能指向模型不同维度的优势，单一指标易产生偏颇结论，建议在阅读预测任务中标准化采用多指标组合。
- **共享候选词表的 POS surprisal 构造**：使用统一词表（如 Cifu）为所有模型构造 POS 候选集，确保跨模型可比性，这一技巧可直接复用于其他多语言/多方言评估。
- **基线归一化百分比提升**：不同眼动指标（FFD/SFD/TFD）的绝对尺度差异较大，用 $\%\Delta$MAE 做跨指标比较能避免 TFD 数值优势主导结论，值得作为标准报告形式。
- **任务分层分析**：同时报告 NR 与 TSR 的表现有助于揭示模型预期在不同阅读目标下的敏感性，可作为阅读眼动研究的标配。

## 关键术语表
- **Lexical surprisal**：目标词在其上下文条件分布下的负对数概率，衡量该词出现的"意外程度"。
- **POS surprisal**：将目标词所属词类的所有候选词概率聚合后计算的负对数概率，反映句法类别层面的预期。
- **Entropy before**：在观察到目标词之前，模型对下一个 token 分布的不确定性（熵），刻画目标前的预期状态。
- **Entropy reduction**：目标词出现前后下一 token 分布熵的差值，衡量该词对后续不确定性的削减力度。
- **MCFIX**：包含普通话与粤语平行眼动语料的阅读数据集，以《小王子》翻译为材料，提供 FFD/SFD/TFD 等指标。
- **FFD / SFD / TFD**：眼动读取指标，分别为首次注视时长、二次注视时长与总注视时长（单位 ms）。
- **CatBoost**：支持类别特征的梯度提升树库，本文用作眼动预测回归器。
- **CantoneseLLM-7B**：基于 Qwen2.5-7B 的大规模粤语持续预训练 + 75K 对指令微调模型。

## 可复现要素
- **数据集**：MCFIX Cantonese 子集（公开语料《小王子》粤语译本眼动数据）。
- **代码**：论文声明接受后开源。
- **模型权重**：CKIP GPT-2 Tiny、JED351、Qwen2.5-7B、CantoneseLLM-7B 均可从公开仓库获取（CantoneseLLM 见 HuggingFace hon9kon9ize）。
- **关键超参**：CatBoost 1,000 轮迭代、内部 seed=10；K-Fold 5 折、shuffled、split seeds 42–46；POS 候选集来自 Cifu 词表，POS 标注用 PyCantonese。
- **报告指标**：mean $\Delta$MAE ± SD（ms）、5/5 seeds 正方向计数、基线归一化百分比 $\%\Delta$MAE。
