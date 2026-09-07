---
title: "Representational-alignment-yields-generalizable-safety-in-la"
source: https://arxiv.org/pdf/2609.04022v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 20:21:38"
field: "大模型安全对齐"
keywords: ["LLM安全对齐", "表示对齐", "对抗鲁棒性", "道德分类", "ReSO", "RSA", "原型理论"]
innovations: ["提出ReSO方法，直接对齐LLM潜在表征与人类道德分类结构", "揭示行为对齐与表征对齐的可分离性，证明DPO不重构内部道德分类", "建立RSA与对抗鲁棒性的强相关（R²=0.86），提供安全对齐的新路径"]
benchmarks: ["HarmBench", "DeceptionBench", "OpenRT", "XSTest", "MMLU-Pro", "HaluEval", "Flames", "MoReBench", "Ethics Benchmark"]
---

# 论文速读：Representational-alignment-yields-generalizable-safety-in-language-models

## 一句话总结
本文提出了一种表示对齐方法 ReSO（Representational Similarity Optimization），通过将 LLM 的潜在表示与人类道德分类结构直接对齐，在不监督生成响应的情况下，显著提升了模型在对抗性攻击下的安全泛化能力。

## 研究问题与动机
- 当前LLM对齐方法（如DPO、RLHF）主要优化可观察的响应行为，但模型在面对相同有害意图的不同表述（如叙事型越狱攻击）时仍容易失效。
- 人类基于原型理论的分类系统具有 graded typicality（分级典型性），能够跨语境泛化安全判断；而LLM的表征仅保留统计规律，未能组织类似的道德概念结构。
- 同一模型谱系中（base→instruct→safeguard），响应级对齐并未实质性重构底层道德分类的潜在表征轨迹。
- 需要一种不依赖响应监督、直接从表示层面对齐人类道德分类的方法，以实现更鲁棒的安全泛化。

## 核心贡献（创新点）
- **提出ReSO表示对齐方法**：通过Bradley-Terry排序目标直接对齐潜在表示间的相似度与人类道德判断的分类关系，不监督生成的token。
- **首次系统量化23个LLM的道德表征对齐程度**：发现当前模型在对立道德类别分离（如authority-subversion和care-harm）、类别内典型性梯度保留、以及线性可解码性方面均较弱。
- **揭示行为对齐与表征对齐的可分离性**：DPO快速提升显式道德判断准确率但几乎不改变RSA，ReSO则显著重构内部表征结构。
- **证明表示对齐可转化为对抗鲁棒性**：ReSO在多个模型规模上 consistently 降低HarmBench、DeceptionBench和OpenRT的ASR，且与RSA呈强相关（R²=0.86）。

## 方法详解
- **人类道德参考构建**：从Social-Chemistry-101提取251,334条原子道德判断，基于MFT五对基础（care-harm等）分解为10个善/恶类别，每个行为表示为稀疏十维向量，维度值=道德判断分×一致性权重。
- **表征提取与各向异性校正**：对16,135个行为的残余流隐藏状态做全局均值中心化（per-layer global mean subtraction），纠正Transformer激活的各向异性。
- **ReSO损失函数**：
  - 结构损失L_struct：对每个decoder层l，从有效三元组集合T中采样，用Bradley-Terry目标拟合模型相似度与人类相似度排序：
    $$\mathcal{L}_{struct} = \sum_{l=1}^{L} w_l \frac{1}{|\mathcal{T}|} \sum_{(i,j,k) \in \mathcal{T}} -\log\sigma\left(\frac{\Delta_{ijk}^{(l)} - \delta}{\tau}\right)$$
    其中$\Delta_{ijk}^{(l)} = s_M^{(l)}(i,j) - s_M^{(l)}(i,k)$，δ=0.05，τ=0.1。
  - 保持损失L_pres：在50,000条通用语料上用KL散度防止能力退化。
  - 总损失：$\mathcal{L} = \mathcal{L}_{struct} + \beta \mathcal{L}_{pres}$，β=1。
- **对照组**：DPO使用相同标注数据构建偏好对优化响应分布；Shuffled control对标签做置换后应用ReSO流程。

## 实验与结果
- **评估基准**：9个OOD基准，包括HaluEval、MMLU-Pro（能力）、Ethics Benchmark、Flames、MoReBench（道德价值）、XSTest（过度拒绝）、HarmBench、DeceptionBench、OpenRT（27种攻击策略的ASR）。
- **模型**：Qwen3-8B/14B/32B、gpt-oss-20b（MoE，基线安全已很强）。
- **最强结果**：
  - Qwen3-8B：HarmBench ASR从26.17%降至14.72%（↓11.45pp），DeceptionBench从53.28%降至31.61%（↓21.67pp），OpenRT均值从24.77%降至19.63%。
  - gpt-oss-20b：HarmBench ASR从3.33%降至1.33%，同时XSTest平衡分从54.38提升至88.95（DPO虽提升XSTest至93.03但HarmBench升至6.00%）。
  - Qwen3-8B训练轨迹：RSA解释86%的ASR方差（R²=0.855），RSA与ASR呈剂量效应关系。
- **基线对比**：DPO在所有模型上均提升各类攻击ASR（HarmBench 8B:+11.41pp，14B:+11.41pp，32B:+2.33pp），表明行为对齐反而降低对抗鲁棒性。

## 相关工作脉络
- **DPO（Rafailov et al., 2023）**：直接偏好优化，优化输出分布；本文与之形成对比，证明响应级对齐不重构内部表征。
- **Social-Chemistry-101（Forbes et al., 2020）**：提供大规模众包道德判断数据，本文将其转化为分级稀疏向量作为表征对齐的参考标准。
- **Moral Foundations Theory（Graham et al., 2011）**：五对道德基础理论，本文据此构建十维道德分类框架。
- **Representational Similarity Analysis（RSA，Kriegeskorte et al., 2008）**：用于量化模型内部表征与人类认知的对应关系，本文将其作为训练目标而非仅评估工具。
- **Prototype Theory（Rosch, 1978）**：人类概念以原型为中心、按典型性梯度分类；本文将其操作化为计算框架并验证其对安全泛化的功能作用。
- **HarmBench/DeceptionBench/OpenRT**：对抗评估基准，本文用其证明表示对齐的泛化安全性。

## 局限性与未来方向
- 人类道德参考基于MFT框架，无法穷尽所有文化/社区的道德认知差异；但ReSO框架本身不绑定MFT，可扩展至更细粒度分类或替代理论。
- 仅在10个道德类别上操作，未覆盖更丰富的价值维度。
- 实验集中在文本生成模型，未验证多模态场景。
- 训练步数固定为1000步，后期RSA回退现象暗示可能存在过拟合风险，需探索最优训练剂量。
- 未来可将此表示对齐原则推广至更广泛的价值对齐任务，或与推理时对齐方法结合。

## 研究启发与可借鉴点
- **表征对齐可作为独立对齐信号**：不依赖响应标注，直接优化内部表征结构，为安全对齐提供新范式。
- **Bradley-Terry排序目标应用于表示学习**：通过三元组排序约束模型相似度与人类判断的一致性，方法简洁且可微。
- **各向异性校正+全局均值中心化**：作为表征分析的标准预处理步骤，在本文中被用于构建可靠的相似度度量。
- **RSA作为训练监控与 checkpoint 选择依据**：建立了RSA与对抗鲁棒性的定量关联（R²=0.86），可作为安全对齐的代理指标。
- **保留损失（preservation loss）防止能力退化**：在通用语料上用KL散度约束分布偏移，平衡对齐与能力保持。

## 关键术语表
- **ReSO（Representational Similarity Optimization）**：一种直接对齐LLM潜在表示与人类道德分类结构的训练方法，不监督生成响应。
- **RSA（Representational Similarity Analysis）**：通过比较模型内部表征相似度矩阵与人类判断相似度矩阵的对应程度，量化表征对齐水平。
- **Bradley-Terry模型**：用于有序比较的统计模型，本文用于将模型相似度排序拟合到人类道德判断的三元组关系。
- **MFT（Moral Foundations Theory）**：道德基础理论，将道德推理分为care-harm、fairness-cheating等五对对立维度。
- **Graded Typicality（分级典型性）**：原型理论中概念成员身份的连续程度，如"杀人"比"扇耳光"更典型地属于"harm"类别。
- **ASR（Attack Success Rate）**：攻击成功率，衡量越狱攻击绕过模型安全防护的比例。
- **Anisotropy Correction（各向异性校正）**：通过减去每层全局均值激活，纠正Transformer表示空间中由共享方向主导的偏差。
- **Social-Chemistry-101**：基于Reddit社区的大规模众包道德判断数据集，本文从中提取251,334条原子判断构建道德参考向量。

## 可复现要素
- 数据集：Social-Chemistry-101（公开），本文处理后得到201,023训练/25,170验证/25,141测试样本。
- 代码：开源，https://github.com/LingyuLi-Cogs/ReSO
- 权重：使用开源模型Qwen3-8B/14B/32B和gpt-oss-20b。
- 关键超参：β=1（preservation weight），δ_h=0.2（人类关系阈值），δ=0.05（相似度margin），τ=0.1（温度），学习率1e-5/1e-6/5e-6（按模型），1000步训练。
- 硬件：8×NVIDIA H200 GPUs。
