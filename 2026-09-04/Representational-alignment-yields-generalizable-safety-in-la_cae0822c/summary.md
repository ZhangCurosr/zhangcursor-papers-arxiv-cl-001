---
title: "Representational-alignment-yields-generalizable-safety-in-la"
source: https://arxiv.org/pdf/2609.04022v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 20:21:43"
field: "大语言模型安全对齐"
keywords: ["LLM安全对齐", "表征相似度优化", "对抗鲁棒性", "道德分类", "原型理论", "DPO", "红队攻击"]
innovations: ["提出ReSO直接对齐LLM潜层表征与人类道德分类结构，无需监督生成响应", "系统量化23个LLM的道德分类保留程度并揭示行为对齐与表征重构的可分离性", "建立表征相似度RSA与对抗攻击成功率ASR之间的剂量效应关系（R²=0.86）"]
benchmarks: ["HarmBench", "OpenRT", "DeceptionBench", "XSTest", "Flames", "Ethics Benchmark", "MoReBench", "MMLU-Pro", "HaluEval"]
---

# 论文速读：Representational-alignment-yields-generalizable-safety-in-la

## 一句话总结
本文提出**表征相似度优化（ReSO）**方法，通过直接对齐LLM潜层表征与人类道德分类结构，使模型在未经监督生成响应的情况下获得更通用的对抗安全性；实验表明，标准行为对齐（DPO）虽提升显式判断准确率，却增加越狱攻击成功率，而ReSO一致降低多种红队攻击成功率。

## 研究问题与动机
- **现有对齐方法仅优化可观测响应**：RLHF、SFT和推理时对齐等方法直接约束模型输出方向，在标准评测上表现良好，但对语言改写后的恶意意图泛化能力差。
- **LLM难以跨形式泛化安全认知**：同样具有恶意意图的请求，经过叙事包裹（如"以祖母睡前故事形式讲述受限信息"）即可绕过安全防线，而人类能轻易识别其共同有害意图。
- **原型理论为解释人类适应性提供框架**：人类概念围绕原型组织，新实例按相对于原型的典型性分级归类，支持跨语境泛化；而LLM表征主要服务于统计压缩与预测，未充分保留graded typicality结构。
- **现有LLM道德分类表征与人类判断弱对应**：对23个开源LLM的分析显示，对立道德类别常重叠、类别内典型性梯度仅被部分保留，且基座、指令微调和安全变体内部表征轨迹高度一致。

## 核心贡献（创新点）
1. **系统量化了23个LLM的人类道德分类保留程度**：通过原型分离度、典型性梯度和线性可解码性三个维度揭示当前LLM内部道德表征的缺陷。
2. **提出ReSO（表征相似度优化）方法**：直接对齐LLM潜层表征相似度与人类道德判断中的分类关系，使用Bradley-Terry排序目标，无需监督生成响应、无需奖励模型或裁判模型。
3. **证明行为对齐与表征重构的可分离性**：同一数据集上，DPO快速学习显式道德判断但几乎不改变RSA，ReSO大幅提升RSA而仅带来适度的判断准确率提升。
4. **首次建立表征相似度与对抗鲁棒性的剂量效应关系**：在Qwen3-8B训练轨迹中，验证RSA解释了HarmBench ASR变化的86%，Late-training阶段RSA回退伴随ASR上升。

## 方法详解
- **人类道德参考构建**：基于Social-Chemistry-101数据集的251,334条原子判断，利用Moral Foundations Theory（MFT）将5对对立基础分解为10个德行/恶行类别（care/harm, fairness/cheating, loyalty/betrayal, authority/subversion, sanctity/degradation）。每条动作表示为稀疏十维向量，活性维度表示道德类别，幅值编码confidence-weighted typicality：$m^+ = \text{ReLU}(v_a) \times w_a$，$m^- = \text{ReLU}(-v_a) \times w_a$。

- **表征提取与各向异性修正**：提取每层残差流隐藏状态，经全局均值中心化 $\hat{z}_a^{(l)} = z_a^{(l)} - \mu^{(l)}$ 消除Transformer各向异性。

- **ReSO损失函数**：包含结构损失与保持损失：$\mathcal{L} = \mathcal{L}_{struct} + \beta \mathcal{L}_{pres}$。
  - **结构损失**：对每个decoder层$l$，对有效三元组$(i,j,k)$（来自同一MFT基础段，满足$s_H(i,j) - s_H(i,k) \geq \delta_h$）计算：$\mathcal{L}_{struct} = \sum_l w_l \frac{1}{|\mathcal{T}|}\sum_{(i,j,k)\in\mathcal{T}} -\log\sigma\left(\frac{\Delta_{ijk}^{(l)} - \delta}{\tau}\right)$，其中$\Delta_{ijk}^{(l)} = s_M^{(l)}(i,j) - s_M^{(l)}(i,k)$，$s_M$为模型余弦相似度，$s_H(a,b) = -\|h_a - h_b\|_2$，$\delta_h=0.2$，$\delta=0.05$，$\tau=0.1$。
  - **保持损失**：对50,000条通用语料计算KL散度正则项$\mathcal{L}_{pres}$，防止能力退化。
  - 输入嵌入与unembedding被冻结，仅训练decoder堆栈。

- **对比方法DPO**：相同251,334条标注，将动作渲染为道德判断问题，偏好/拒绝完成分别表达标注的善/恶或错误判断，$\beta_{DPO}=0.1$。

- **Shuffled control**：相同的ReSO流水线但标签被随机置换，匹配优化压力但破坏动作-判断对应关系。

## 实验与结果
- **数据集**：Social-Chemistry-101，251,334条原子判断（train 201,023 / val 25,170 / test 25,141）。
- **模型**：4个模型——Qwen3-8B/14B/32B（稠密）和gpt-oss-20b（MoE）。
- **评估基准（9个，均OOD）**：HaluEval、MMLU-Pro（能力）；Ethics Benchmark、Flames（中文）、MoReBench（道德）；XSTest（过度拒绝）；HarmBench、DeceptionBench、OpenRT（27种红队攻击）。

**主要结果**：

| 模型 | 条件 | HarmBench ASR↓ | DeceptionBench ASR↓ | OpenRT ASR↓ | Flames ↑ |
|------|------|---------------|-------------------|-------------|----------|
| Qwen3-8B | Base | 26.17% | 53.28% | 24.77% | 63.80 |
| Qwen3-8B | DPO | 37.58% (+11.41) | 65.67% (+12.39) | 29.28% | 61.15 |
| Qwen3-8B | ReSO | **14.72%** (-11.45) | **31.61%** (-21.67) | **19.63%** | **67.24** |
| Qwen3-14B | ReSO | **13.33%** (-9.15) | **30.56%** (-13.22) | **19.88%** | **67.30** |
| Qwen3-32B | ReSO | **13.67%** (-5.33) | **26.39%** (-12.11) | **21.74%** | **64.99** |
| gpt-oss-20b | Base | 3.33% | 18.95% | 19.13% | 72.30 |
| gpt-oss-20b | DPO | 6.00% (+2.67) | 26.67% (+7.72) | 25.30% | 66.43 |
| gpt-oss-20b | ReSO | **1.33%** (-2.00) | **16.24%** (-2.71) | **15.88%** | **73.98** |

- 在OpenRT的27种攻击中，ReSO在Qwen3-8B/14B/32B上分别减少了23/23/19种攻击的ASR，DPO则使22/23/22种攻击恶化。
- **gpt-oss-20b关键发现**：ReSO在已强安全对齐的模型上同时降低了所有三种ASR并提升了XSTest平衡分（从54.38→88.95），而DPO虽提升XSTest至93.03但牺牲了安全性。
- **训练动态**：Qwen3-8B上RSA从0.060升至峰值0.24，ASR从24.2%降至15.1%（Δ=-9.1pp, p=3.5×10⁻⁶），RSA解释了86%的ASR变异（R²=0.855）。Shuffled control仅恢复ReSO效果的约25%。

## 相关工作脉络
1. **DPO（Rafailov et al., 2023）**：本文核心对比基线。DPO直接优化条件输出分布，在相同数据上能快速学习道德判断但几乎不改变表征相似度，证明行为对齐可与表征重构解耦。
2. **Constitutional AI（Bai et al., 2022）与RLHF（Ouyang et al., 2022）**：传统响应级对齐方法。本文延伸了对其局限性的批判——高效的学习输出策略不等于建立了支持泛化的概念组织。
3. **HarmBench（Mazeika et al., 2024）、OpenRT（Wang et al., 2026）、DeceptionBench（Huang et al., 2026）**：本文使用的对抗鲁棒性评测基准，用于系统评估跨27种红队攻击的安全性。
4. **Prototype Theory（Rosch, 2024）**：认知科学的理论基础。本文首次将该理论 operationalized 到LLM对齐中，为adaptive categorization提供了计算支持。
5. **Social-Chemistry-101（Forbes et al., 2020）与Moral Foundations Theory（Graham et al., 2011）**：本文道德参考的来源基础。通过将MFT的10维graded判断映射到表征空间，构建了可优化的对齐目标。
6. **Jailbreak攻击研究（Wei et al., 2023; Andriushchenko & Flammarion, 2025）**：揭示了现有安全对齐的脆弱性。本文从表征层面提供了一条增强鲁棒性的补充路径。

## 局限性与未来方向
- **道德参考的非完备性**：基于MFT的社会化学反应数据库不能穷尽人类道德认知，且不同文化和社区存在差异；但ReSO框架本身不绑定特定理论。
- **仅评估了文本攻击**：OpenRT测试集中于27种文本攻击，未涵盖多模态场景。
- **行为判断提升有限**：ReSO在显式道德判断上的改善不如DPO显著，特别是在gpt-oss-20b上几乎无提升。
- **Late-training剂量效应**：RSA过度优化后出现回退并伴随ASR上升，提示需寻找最优对齐强度，尚未完全理解机制。
- **未探索跨语言泛化机制**：尽管Flames（中文）结果显示ReSO有正向转移效果，但具体机制有待进一步研究。

## 研究启发与可借鉴点
1. **表征对齐作为行为对齐的补充路径**：本文证明了不监督生成响应也能通过优化潜层关系提升安全性，为未来研究提供了脱离response-level优化的可行范式。
2. **RSA作为安全性的可解释代理指标**：训练过程中RSA与ASR的高度线性相关（R²=0.86）表明，表征相似度可作为安全性的低成本监控指标，无需频繁运行完整红队评估。
3. **梯度典型性的量化与对齐**：将MFT判断转化为confidence-weighted稀疏向量并用于三元组排序损失的设计，可作为其他领域（如价值观对齐）的参考模板。
4. **Shuffled-label control的严格性**：用标签置换控制匹配优化压力但破坏语义对应，验证了ReSO效果源于真正的结构对齐而非单纯优化强度，此实验设计值得借鉴。
5. **与团队方向的结合机会**：可将ReSO的思想迁移到知识忠诚、工具调用正确性等非安全领域的表征对齐研究，探索"表征结构优化→泛化鲁棒性"的普适性。

## 关键术语表
- **ReSO（Representational Similarity Optimization）**：直接对齐LLM潜层表征相似度与人类道德判断分类关系的训练方法，使用Bradley-Terry排序目标，不监督生成响应。
- **RSA（Representational Similarity Analysis）**：通过计算模型表征相似度矩阵与人类判断相似度矩阵之间的Spearman相关来量化内部表征与人类认知的对应程度。
- **MFT（Moral Foundations Theory）**：道德基础理论，认为人类道德推理建立在5对对立基础（care-harm、fairness-cheating等）之上。
- **Bradley-Terry目标**：一种配对比较排序模型，用于将模型内部相似度关系拟合到人类判断的有序关系上。
- **各向异性（Anisotropy）**：Transformer激活空间中所有输入共享一个主导方向的现象，本文通过全局均值中心化进行修正。
- **ASR（Attack Success Rate）**：红队攻击成功率的指标，越低表示模型越安全。
- **XSTest**：评估模型在拒绝不安全请求与避免过度拒绝良性请求之间平衡的测试套件。
- **Shuffled Control**：将ReSO的标签随机置换后的控制实验，用于验证优化压力之外的结构对应是否必要。

## 可复现要素
- **数据集**：Social-Chemistry-101（公开数据集）；清洗后251,334条原子判断（train 201,023 / val 25,170 / test 25,141）。
- **代码**：已开源，地址 https://github.com/LingyuLi-Cogs/ReSO。
- **模型**：Qwen3-8B/14B/32B、gpt-oss-20b（均为开源模型）。
- **关键超参**：$\delta_h=0.2$（人类关系阈值），$\delta=0.05$（相似度margin），$\tau=0.1$（温度），$\beta=1$（保持损失权重），峰值学习率分别为$1\times10^{-5}$（8B/14B）、$1\times10^{-6}$（32B）、$5\times10^{-6}$（gpt-oss-20b），1000步训练，每25步验证。
- **硬件**：8× NVIDIA H200 GPU。
- **评估工具**：HarmBench、DeceptionBench、OpenRT、XSTest、Flames、Ethics Benchmark、MoReBench、MMLU-Pro、HaluEval，Judge模型为CompassJudger2-32B-Instruct。
