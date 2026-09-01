---
title: "Training-Chemical-Plausibility-Aware-Large-Language-Models-f"
source: https://arxiv.org/pdf/2608.18940v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 21:55:29"
field: "计算化学与逆合成规划"
keywords: ["单步逆合成", "大型语言模型", "化学可归化性", "Top-K提示", "强化学习微调", "CREED数据集"]
innovations: ["提出Top-K Prompting训练范式捕捉逆合成一对多特性", "构建4560万反应的超大规模验证数据集CREED-CCV-2+USPTO-XL", "设计ChemCensor奖励与新颖性奖励结合的强化学习微调方案"]
benchmarks: ["URSA-expert-2026", "USPTO-50K-test-mini"]
---

# 论文速读：Training-Chemical-Plausibility-Aware-Large-Language-Models-f

## 一句话总结
本文提出将Top-K提示模式与ChemCensor可归化性奖励相结合的训练范式，在超大规模数据集CREED-CCV-2+USPTO-XL（约4560万验证反应）上训练C3LM，使其在单步逆合成任务上超越传统模型与通用LLM，达到当前最优性能。

## 研究问题与动机
- **单步逆合成的多路径本质未被现有评估充分捕捉**：传统SSRS任务常以单答案（Top-1）评估，但逆合成本质上是一对多问题，一个目标分子可通过多种独立断开方式合成。
- **现有LLM在逆合成任务上性能不足**：先前研究（Zagribelnyy et al., 2026b）表明，LLM作为SSRS模型在ChemCensor可归化性指标上劣于传统SSRS模型。
- **多样性与可归化性难以兼顾**：如何在保证反应化学合理性的同时提升生成反应的多样性，是当前LLM驱动逆合成的关键挑战。
- **训练数据规模限制模型上限**：现有数据集（如USPTO）通常每个产物仅含单一反应，限制了模型探索多样化反应空间的能力。

## 核心贡献（创新点）
1. **提出Top-K Prompting作为SSRS任务的新标准**：通过显式要求模型生成K个不同答案，更好捕捉逆合成的一对多特性，显著提升多样性指标。
2. **构建CREED-CCV-2+USPTO-XL超大规模数据集**：整合专家编码模板生成的验证反应与基于虚拟合成引擎扩展的USPTO数据，达4560万唯一反应，约是之前数据集的5-7倍。
3. **设计ChemCensor感知训练范式**：结合监督微调与强化学习，以ChemCensor分数和反应新颖性作为奖励信号，直接优化生成反应的可归化性。
4. **揭示LLM与传统模型探索互补的化学空间**：分析反应唯一性发现，最优LLM与传统SSRS模型生成的可归化反应交集有限，为集成学习提供依据。

## 方法详解
- **Top-K Prompting模式**：在原有15个自然语言模板基础上添加"Give me 15 different answers"后缀，使模型每次生成最多15个不同的反应物组合。训练与推理均在此模式下进行。
- **数据集构建**：CREED-CCV-2通过虚拟合成引擎（VSE）对ChEMBL化合物枚举反应并结合CREED-CCV历史数据；USPTO-XL通过VSE为USPTO中的产物扩展多个验证反应物；两者合并去重后达4560万反应。
- **监督微调（SFT）**：以LFM2 2.6B为基座，采用Top-K模式训练，每个CoT示例前缀包含输入分子的canonical SMILES和BRICS片段列表。
- **强化学习微调（RFT）**：采用单奖励Group Relative Policy Optimization（GRPO），group size=8，采样温度=1，KL正则化权重=0.1，训练1000步，学习率10⁻⁶。
- **奖励函数设计**：6个分量加权求和——思考格式有效性（0.1）、有效SMILES生成（0.5）、准确生成K个答案（0.1）、反应唯一性（0.2）、ChemCensor可归化性得分（1.0）、新颖性奖励（1.0，鼓励生成训练集外但ChemCensor为正的反应）。

## 实验与结果
- **基准数据集**：URSA-expert-2026（100个专家验证的新颖目标分子，OOD测试）和USPTO-50K-test-mini（497个目标分子）。
- **评估指标**：Max CC（每目标最高ChemCensor分数均值）、Av. PT-Top-K CC（Top-K平均可归化性分数）。
- **最强模型C3LM-LFM2-RFT-CC-NR**：在URSA-expert-2026上，Max=2.16，@10=1.37；在USPTO-50K-test-mini上，Max=4.28，@10=1.85。
- **提升幅度**：相比传统最强模型LocalRetro（Max=2.11，@10=1.22），C3LM-RFT-CC-NR在URSA上Max提升+0.05，@10提升+0.15；在USPTO-50K上Max提升-0.56，@10提升-0.06。
- **LLM vs 传统模型对比**：所有GP LLM集成（Max=2.19）略优于所有传统模型集成（Max=2.18），但个体最佳仍逊于传统最强。C3LM单模型已超过多数传统模型。
- **Top-K vs Top-1效果**：Top-K模式使C3LM的@10指标从0.38提升至0.98（+1.58倍），@3从1.06提升至1.68。

## 相关工作脉络
- **URSA基准框架**：Zagribelnyy et al. (2026b)提出的逆合成评估框架，使用ChemCensor作为可归化性代理指标而非简单Top-K准确率。
- **传统SSRS模型**：LocalRetro、MHNreact、RetroKNN等基于模板或图神经网络的方法，在精确匹配评估下表现优异。
- **化学LLM**：如NaCHO、BioT5等预训练语言模型在化学任务上的应用。
- **CREED数据集**：作者前期工作构建的包含约2270万反应的验证数据集，为本研究的数据扩展奠定基础。
- **GRPO强化学习**：Shao et al. (2024)提出的Group Relative Policy Optimization，本文首次将其应用于化学逆合成LLM微调。
- **本文定位**：首次在LLM训练中系统整合Top-K多样性与ChemCensor可归化性双重优化，填补了可归化性感知LLM训练的空白。

## 局限性与未来方向
- **评估-优化循环性**：模型直接以ChemCensor为奖励训练，与ChemCensor基准评估存在部分循环性。
- **ChemCensor局限性**：未考虑实际实验室合成参数（反应条件、溶剂、纯化方法）。
- **多样性度量单一**：仅通过SMILES精确匹配评估唯一性，缺乏对反应类型/机理层面的化学对比。
- **模板偏差**：训练数据依赖模板引擎，可能偏向特定化学模式，缺乏"新化学"变换。
- **参考数据集有限**：ChemCensor验证仅基于专利数据，未涵盖全部已知反应空间。

## 研究启发与可借鉴点
- **Top-K训练范式可迁移至其他化学生成任务**：对于多解性质的化学预测问题（如反应条件预测、逆合成多步规划），Top-K模式可作为提升多样性的通用策略。
- **奖励函数设计思路**：将可归化性评分与新颖性奖励结合，兼顾质量与探索，可用于其他化学语言模型微调。
- **LLM与传统模型集成潜力**：研究发现两者探索互补反应空间，建议在实际CASP系统中采用LLM+传统模型的集成方案。
- **超大规模化学数据集构建方法**：虚拟合成引擎+化学验证框架的流水线可复用于其他化学领域的数据扩展。
- **评估基准设计启示**：从单答案评估转向多样性感知的Top-K评估，对化学AI基准建设具有示范意义。

## 关键术语表
**ChemCensor**：评估反应或合成路线化学可归化性的量化指标，通过匹配反应中心与功能团上下文计算置信度等级。
**Top-K Prompting**：显式要求LLM生成K个不同答案的提示范式，用于捕捉逆合成的多路径特性。
**C3LM**：Chemistry Constraint-Consistent Language Model，本文训练的可归化性感知化学语言模型系列。
**CREED-CCV-2+USPTO-XL**：约4560万验证反应的超大规模训练数据集，由CREED-CCV-2与扩展USPTO合并去重得到。
**GRPO**：Group Relative Policy Optimization，一种基于组的强化学习策略优化算法。
**URSA-expert-2026**：100个专家验证的新颖目标分子组成的OOD逆合成基准。
**单步逆合成（SSRS）**：预测单个化学反应步骤中目标分子转化为前体反应物的任务。
**化学可归化性**：反应符合有机合成基本原理（如化学选择性、区域选择性、立体选择性）的程度。

## 可复现要素
- **数据集**：CREED-CCV-2+USPTO-XL未公开（论文未提及开源链接）；USPTO-full为公开数据。
- **代码**：ChemCensor源码已开源：https://github.com/insilicomedicine/ChemCensor。
- **模型权重**：C3LM系列模型权重未明确开源（附录A提及可在内部获取）。
- **基准结果**：动态更新结果页面：https://dddbench.insilico.com/。
- **关键超参**：LFM2 2.6B基座；SFT训练50000步；RFT训练1000步，学习率10⁻⁶，KL权重0.1，GRPO group size=8，温度=1。
- **Token化处理**：扩展词表包含SMILES专用token，输入时50%概率token化SMILES，输出时始终token化；训练时采用非规范随机遍历增强。
