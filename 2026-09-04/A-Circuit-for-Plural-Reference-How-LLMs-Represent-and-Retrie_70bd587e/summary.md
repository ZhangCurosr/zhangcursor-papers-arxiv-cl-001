---
title: "A-Circuit-for-Plural-Reference-How-LLMs-Represent-and-Retrie"
source: https://arxiv.org/pdf/2609.03687v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 00:34:43"
field: "机制可解释性与语言理解"
keywords: ["mechanistic interpretability", "plural reference", "coreference resolution", "attention circuit", "activation patching"]
innovations: ["发现由代词解释头、复数形成头、代词选择头构成的复数指称电路", "验证LLMs复数偏好与人类本体相似性和连接词约束对齐", "定位代词选择头并证明其注意力差异与单/复数概率差异强相关"]
benchmarks: ["D_pl复数期望数据集", "D_sg单数期望数据集", "Qwen3-1.7B", "GPT2-medium"]
---

# 论文速读：A-Circuit-for-Plural-Reference-How-LLMs-Represent-and-Retrie

## 一句话总结
本文通过机制可解释性（activation patching、path patching、注意力模式分析）揭示了大语言模型在预测复指代词时识别和检索单数/复数实体的内部电路，发现由三层注意力头（代词选择头、复数形成头、代词解释头）构成的小而密集的信息流网络。

## 研究问题与动机
- **核心问题**：现有共指解析研究多聚焦单数实体追踪，对复数实体的表示与检索机制缺乏深入了解；LLMs如何处理"John and Mary came to the shop. They bought some milk"这类复数指称尚不清楚。
- **理论缺口**：心理语言学表明复数处理比单数更复杂，需同时追踪多个实体并进行分组，但NLP领域对LLMs如何识别"哪些实体构成复数实体"仍无机制级证据。
- **方法动机**：行为评估难以揭示内部状态；采用机制可解释性技术（如Meng et al. 2022的事实检索电路、Wang et al. 2022的间接宾语识别电路）可逆向工程LLMs的推理路径。
- **应用动机**：理解LLMs内部机制对提升AI安全性、可控性、与人类对齐至关重要（Bereska & Gavves, 2024; Ferrando et al., 2024）。

## 核心贡献（创新点）
1. **发现复数指称电路**：首次提取出由三层注意力头构成的信息流电路（代词解释头→复数形成头→代词选择头），明确各组件在复数预测中的因果角色。
2. **验证LLMs与人类的复数偏好对齐**：证明LLMs在复数构建上遵循与人类相同的约束——本体相似性（proper name + proper name > proper name + relation > proper name + object）和连接词效应（and比with更倾向触发复数）。
3. **定位代词选择头（Pronoun Selection Heads）**：发现Layer 23的H12和H13直接决定单/复数代词预测，且其对e₂与e₁的注意力差异与P_plural vs P_singular的概率差异呈强正相关（r=0.78, p<0.001）。
4. **揭示复数形成头的选择性注意机制**：L17H9在合格复数结构（Mary and her sister）时广泛覆盖整个连接短语，而在不合格结构（Mary and her bike）或with连接时仅关注e₁，显示其对复数实体候选的过滤功能。
5. **跨模型泛化验证**：电路在Qwen3-0.6B、Qwen3-1.7B和GPT2-medium中保持一致，证明其并非特定架构的偶然现象。

## 方法详解
**数据集构建**：
- D_pl（复数期望集）：300条前缀，如"When John saw Mary and her sister, he waved at ___"，预期输出复数代词them。
- D_sg（单数期望集）：300条前缀，将e₂替换为无生命实体，如"When John saw Mary and her bike, he waved at ___"，预期输出单数代词her。
- 变量操控：性别（相同/不同）、本体相似度（N+N > N+R > N+O）、连接词（and vs with）。

**统计建模**：
- 使用R lme4包拟合线性混合效应回归（REML），动词为随机效应，因变量为概率差P_pl - P_sg，验证本体相似性和连接词的显著效应（p<0.001）。

**干预技术**：
1. **Activation Patching**：测量各组件对P_pl的间接影响——用s_sg的激活替换s_pl中特定位置的激活，观察P_pl变化。
2. **Path Patching**：在替换目标组件激活的同时恢复上游所有组件的原始激活，隔离直接效应。
   - 步骤：①原始运行记录P_pl,ori；②s_sg前向传播并缓存所有层激活X^l；③用s_sg的C位置激活替换s_pl对应位置，恢复上游激活后重新前向传播，计算P_pl,int。
   - 指标M = (P_pl,int - P_pl,ori) / P_pl,ori × 100，负值表示抑制复数信号。
3. **干预位置**：仅在最后token和e₂/c位置产生显著效应。

**QK/OV分解**：
- 对代词选择头的Q、K、V向量分别干预：Q在最后一token位置，K/V在e₂位置。
- 发现Q向量不影响预测，而K/V向量干预导致P_pl强下降。
- 进一步追溯K/V头来源：通过path patching定位Layer 15/17/19/21的K/V头直接影响代词选择头的Value向量，其中L17H9效应最强。

**注意力模式分析**：
- 可视化各头在D_pl、D_sg、D_pl,w上的平均注意力权重分布。
- 计算α_diff = α_{i,e₂} - α_{i,e₁}与P_diff = P_pl - P_sg的Pearson相关（Appendix C）。

## 实验与结果
**模型**：Qwen3-0.6B（28层，16头，d_model=1024）、Qwen3-1.7B（主实验，28层，16头，d_model=2048）、GPT2-medium（24层，16头，d_model=1024）。

**关键结果**：
| 组件 | 层级 | 核心发现 |
|------|------|----------|
| Pronoun Selection Heads | L23H12, L23H13 | Layer 23是唯一直接影响输出的MHSA模块；L23H12/13在D_pl中对e₁和e₂分配高注意力，e₂ > e₁；在D_sg中仅关注e₁ |
| Plurality Formation Head | L17H9 | D_pl,a：关注整个复数结构（c和e₂注意力最高）；D_sg,a：注意力止于possessive pronoun；D_pl,w：几乎不关注e₂ |
| Pronoun Interpretation Head | L13H6 | 编码共指信息：代词attend到其先行词（her→Mary, he→John）；添加them后attend整个复数结构 |

**定量证据**：
- L23H12的α_diff与P_diff相关系数r=0.78（p<0.001），L23H13的r=0.77（p<0.001）。
- K/V头干预：L17H9对代词选择头的Value向量影响最大（Figure 5）。
- Pronoun Interpretation Head影响约25%的概率差。

**行为验证**：
- 回归模型复制心理语言学经典发现：本体相似度越低，复数偏好越低；with连接比and连接的复数偏好显著降低（β=-0.669, p<0.001）。

## 相关工作脉络
1. **Wang et al. (2022) 间接宾语识别电路**：本文发现的代词选择头类似其"Name Mover Head"，但目标从单一实体追踪扩展到复数实体识别。
2. **Meng et al. (2022) 事实知识检索电路**：方法学直接沿用activation patching和path patching，应用于新的共指解析任务。
3. **Dai et al. (2024, 2026) 实体追踪表征**：本文将其扩展至复数实体，揭示LLMs不仅追踪单实体还构建"复数实体"抽象。
4. **Anh et al. (2025) LLM复数指称歧义检测**：前序行为研究发现LLMs对复数歧义敏感且响应不稳定，本文从机制层面解释其内部表征缺失的原因。
5. **Clark et al. (2019); Tenney et al. (2019) BERT注意力分析**：本文确认L13H6的共指编码功能与其发现的句法依赖追踪头一致，但强调其在自回归模型中的因果作用。
6. **心理语言学文献**：Koh & Clifton (2002)的本体相似性、Moxey et al. (2004, 2012)的连接词效应、Clifton & Ferreira (2016)的split-antecedent、Cokal et al. (2023)的mereological reference，本文在LLMs中验证了这些约束。

## 局限性与未来方向
- **复数结构单一**：仅研究最简单的并列名词短语（两个元素、and连接），未涵盖split-antecedent（如"John arrived. Mary left. They..."）或 mereological reference（"engine and boxcar → it"）。
- **电路非完备**：仅聚焦注意力头，MLP模块功能未深入（虽确认晚期层MLP干预影响P_pl，但内部机制未知）；电路可能非共指独有，与其他依赖解析机制存在重叠。
- **模型范围有限**：仅在Qwen3和GPT2家族验证，未测试Llama、PaLM等主流架构。
- **未见代码/数据开源声明**。

## 研究启发与可借鉴点
1. **双数据集对比设计**：D_pl与D_sg保持结构、长度、token数一致，仅改变e₂的本体属性或连接词，确保patching结果的效度，此设计可迁移至其他语言理解任务。
2. **QK/OV分解策略**：分别干预Query、Key、Value向量并定位其来源头，为电路逆向提供系统化方法。
3. **注意力-概率相关性验证**：通过计算α_diff与P_diff的Pearson相关（Appendix C）建立机制与行为的定量桥梁，增强发现的说服力。
4. **跨层信息流追踪**：从最终决策头（L23）回溯至中间处理头（L17）再至早期表征头（L13），揭示"共指编码→复数筛选→代词选择"的层级流水线，可作为其他任务电路挖掘的模板。
5. **结合心理语言学先验**：用统计回归验证模型行为符合人类认知约束，增强机制发现的生态学效度。

## 关键术语表
**Mechanistic Interpretability（机制可解释性）**：一种试图将LLMs的计算逆向工程为人类可解释组件的方法论，通过因果干预定位特定模块对预测的贡献。

**Activation Patching（激活修补）**：将输入序列中特定位置的激活替换为另一输入的对应激活，测量目标组件对输出的间接效应。

**Path Patching（路径修补）**：在替换目标组件激活的同时恢复所有上游组件的原始激活，以隔离该组件的直接因果效应。

**Pronoun Selection Heads（代词选择头）**：位于模型晚期层（Layer 23）的注意力头，直接决定单/复数代词预测，通过关注候选先行词实现选择。

**Plurality Formation Head（复数形成头）**：位于中间层（Layer 17）的注意力头，识别构成复数实体的实体集合，仅当实体满足本体相似性和连接词约束时才广泛覆盖整个结构。

**Pronoun Interpretation Head（代词解释头）**：位于早期层（Layer 13）的注意力头，作为"共指地图"编码所有提及实体的共指关系，代词在此处attend到其先行词。

**Ontological Similarity（本体相似性）**：实体所属范畴的相似程度，本文分三级：proper name + proper name（最高）> proper name + relation > proper name + object（最低）。

**Residual Stream（残差流）**：Transformer中各组件输出的累加通道，信息在此流动并跨越层传递。

## 可复现要素
- **数据集**：合成数据集D_pl和D_sg各300条前缀，模板见Table 3/4；**论文未声明公开**。
- **代码**：论文未提供开源仓库链接，**未声明开源**。
- **模型**：Qwen3-0.6B、Qwen3-1.7B、GPT2-medium，可使用transformers库加载。
- **关键超参**：干预位置仅限last token和e₂/c位置；修复动词为随机效应；使用lme4包REML拟合。
