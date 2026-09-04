---
title: "CRITICL-Inference-Time-Weak-to-Strong-Generalization-from-Sm"
source: https://arxiv.org/pdf/2608.27455v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 12:28:13"
field: "大语言模型推理增强"
keywords: ["推理时增强", "弱到强泛化", "失败模式", "上下文学习", "大语言模型推理"]
innovations: ["提出 CritBank 离线失败知识库，将检索目标从语义相似转向失败模式对齐", "提出 CritICL-dynamic/static 两种推理时弱到强指导范式，仅需 1-2 次生成即可超越多倍生成的 test-time scaling 方法"]
benchmarks: ["GSM8K", "MATH", "AMC23", "AIME24", "AIME25", "GPQA"]
---

# 论文速读：CRITICL-Inference-Time-Weak-to-Strong-Generalization-from-Sm

## 一句话总结
CritICL 是一种推理时弱到强泛化框架，利用同一家族中小规模模型的系统性失败模式作为检索信号，以 critique-based in-context learning 的形式一次性注入大模型，从而在极少额外推理开销下显著提升数学与科学推理性能。

## 研究问题与动机
1. **现有推理时扩展方法成本高**：Self-consistency、Self-reflection、LLM-as-a-Judge 等方法依赖多轮生成或额外大模型评判，token 消耗大幅增加。
2. **现有弱到强推理方法未充分利用失败模式**：已有 W2S 推理方法（如 W2S-AlignTree）仍为每个输入引入在线额外推理开销，且未将"系统性失败模式具有跨规模可迁移性"这一观察转化为检索信号。
3. **同家族模型存在高度一致的失败分布**：论文实验（Section 4.1 / Table 4）表明，Qwen 和 Llama 同家族内，小模型与大模型的失败模式相对排序与分布高度吻合（Spearman 相关 0.88–0.91，JS 距离低至 0.041），提示失败模式具有跨规模结构性。
4. **核心科学问题**：能否在不增加多次生成或昂贵外部验证的前提下，将小模型的系统性失败模式高效迁移到大模型推理中？

## 核心贡献（创新点）
1. **提出 CritBank 离线失败知识库**：首次将弱模型生成的错误回答、结构化失败模式标签与 LLM 生成的自然语言批判一起组织成可复用数据集；与仅用正确 exemplar 的 ICL 检索相比，引入失败模式维度，使检索目标从"语义相似"转向"失败相关"。
2. **提出 CritICL-dynamic 与 CritICL-static 两种推理时检索范式**：前者对每个输入自适应预测失败模式再检索，后者基于模型家族层面的全局失败概况提供稳定指导；两者的检索均建立在失败模式匹配评分之上，区别于常规 embedding 相似度检索。
3. **实证揭示跨规模失败模式一致性**：通过 Spearman、Kendall's τ、Top-10 重叠和 JS 距离四个定量指标系统验证同家族模型共享失败分布；发现多个弱模型聚合比单一弱模型更贴近强模型分布，为 CritICL-static 的设计提供直接依据。
4. **证明 CritICL 的低开销高收益**：CritICL-static 仅需 1 次生成（CritICL-dynamic 需 2 次），在 Qwen2.5-72B 上 MATH 达 84.0%、整体平均达 59.2%，超越 Consistency@5 等需要 5 次生成的方法，且总 token 消耗显著更低（3768 vs. 4814+）。

## 方法详解
- **CritBank 构建**：对训练集 Q 中的每个问题 q 和小模型 m ∈ M（同家族 1.5B/3B/7B 等），以 CoT 提示生成 5 条回答 R(q,m)，按正确性函数 φ 划分正确/错误子集；对每条错误回答，用 gpt-4o-mini 生成至多 5 个失败模式标签（从预设词汇表选），并进行聚类去重；再为每对 (q,r) 生成自然语言 critique。最终 CritBank = {(q, r, l, C(q,r))}，包含问题、错误回答、失败模式集合与批判文本。
- **Failure Mode-Based Sample Selection**：给定目标失败模式集合 S，计算每个候选 (q,r) 的匹配分 score(q,r;S) = Σ_{l ∈ L(q,r)∩S} w(l)；按得分降序排列后贪心选取前 K 个示例，并优先纳入尚未被覆盖的新失败模式以减少冗余。
- **CritICL-dynamic**：对输入 q'，先让目标模型预测至多 5 个可能的失败模式 S_inst(q')，再调用上述检索过程从 CritBank 中取 K 个相关 critique 示例，拼入 prompt 后单次生成最终答案。
- **CritICL-static**：基于同家族弱模型的失败模式频率分布 P_M(l) 提取 Top-T 全局主导失败模式 S_prof(M)，对所有查询共用同一组 critique 示例，每次仅需 1 次生成。

## 实验与结果
- **数据集**：GSM8K（训练 7.4k，测试 1.3k）、MATH（训练 7.5k，测试 5k）、AMC23、AIME24、AIME25；跨域实验还包含 GPQA（Chemistry/Biology/Physics/Quantum）。CritBank 由 15k 训练样本构建。
- **模型**：Qwen 家族（1.5B/3B/7B → 32B/72B）与 Llama 家族（1B/3B/8B → 70B）；greedy decoding（temperature=0）。
- **主要结果（Qwen2.5-72B-Instruct，Table 1b）**：
  - CritICL-static 整体平均 59.2%，优于 Consistency@5 的 59.0%；MATH 达到 84.0%（超 Consistency@5 的 83.2%），GSM8K 达 95.4%。
  - CritICL-dynamic 整体平均 58.7%，同样与最强 test-time scaling 持平或更优。
- **主要结果（LLaMA-3.1-70B-Instruct，Table 12）**：CritICL-static 整体平均 53.1%，超越 Consistency@5 的 51.3%；OOD（AIME24/AIME25）提升尤为明显。
- **推理成本（MATH 上，Table 2/13/14/15）**：CritICL-static 总 token 3768（Qwen-32B）、3928（Qwen-72B）、2888（Llama-70B），远低于 Consistency@7（5440/5689/4135）与 Self-Reflection（7533/7891/5449）。
- **消融（Table 3）**：Failure mode-based 检索在 4 个数学基准上全面优于随机/固定/语义相似度检索；在 AMC23/AIME25 上准确率提升 4–6 个百分点。
- **跨域/跨家族（Table 10/11）**：同家族迁移最优；跨家族仍可超越标准 ICL，但增益小于同家族；跨域时域内 CritBank 最好，混合 CritBank 次之，纯跨域较弱但仍有提升。

## 相关工作脉络
1. **推理时扩展 / test-time scaling**（Wei et al., 2022; Wang et al., 2022; Madaan et al., 2023; Shinn et al., 2023; Zheng et al., 2023b; Huang et al., 2024）：这些方法通过重复采样、自我反思或外部 judge 提升性能，但 token 开销成倍增长；CritICL 以单次生成替代多次生成，效率显著占优。
2. **弱到强泛化（训练时）**（Burns et al., 2024; Lang et al., 2024; Charikar et al., 2024）：研究弱监督如何提升强模型训练表现；CritICL 聚焦于推理时即插即用的 guidance，不修改模型参数。
3. **W2S-AlignTree**（Ding et al., 2026）：同为推理时 W2S 方法，但需 MCTS + 多步生成；CritICL 仅需 1–2 次生成，且在同等设定下达到更高或相当的精度。
4. **错误感知批评与自我纠正**（Didolkar et al., 2024; Tyen et al., 2024; Zhang et al., 2024; Tang et al., 2025; Gou et al., 2024）：关注模型是否能识别/修正自身错误；CritICL 与之互补——通过离线构建的失败库提供外部 critique 信号，而非在线训练更强的 critic。
5. **基于检索的 ICL**（Wang et al., 2024; Shao et al., 2025; He et al., 2025a/b）：强调示例选择的重要性；CritICL 将检索目标从语义/表面相似切换为失败模式对齐，是对这一路线的新拓展。
6. **失败模式聚类与分类**（Didolkar et al., 2024）：本文沿用其聚类思路并对标签体系进行系统化扩展（附录 H 提供 60+ 类失败模式），使 CritBank 更具可解释性与可复用性。

## 局限性与未来方向
1. **跨家族迁移有限**：Qwen↔Llama 跨家族 transfer 虽优于标准 ICL，但性能明显低于同家族迁移（Table 4 JS 距离 0.132/0.146 vs. 0.041/0.047），根因与架构/训练策略差异有关。
2. **跨域迁移存在性能缺口**：跨域 CritBank 效果不如域内版本（Table 11），提示部分失败模式（如公式误用、概念混淆）具有较强领域特异性。
3. **CritBank 构建成本偏高**：需要弱模型生成 + 前沿 LLM 标注 + 聚类，虽为离线一次性成本，但在资源受限场景下仍具门槛。
4. **标签粒度需权衡**：粒度太粗丢失信息、太细则检索池过稀疏（Table 6 显示细粒度最佳，极细粒度略有回落）。
5. **动态变体仍需额外一次生成**：CritICL-dynamic 需 2 次生成（失败模式预测 + 最终答案），相比 CritICL-static 的 1 次仍多一步推理。
6. **评测领域仍偏数学与基础科学**：尚未在对话、代码生成、长文本推理等更复杂场景系统验证。

## 研究启发与可借鉴点
1. **检索目标可从"语义相似"转向"失败相关"**：在 ICL / RAG 系统中引入结构化错误标签作为检索维度，能够更精准地命中目标模型的真实弱点，值得在团队知识检索方向尝试。
2. **弱到强的跨规模知识迁移可作为低成本增强范式**：利用小规模模型的低成本错误样本构建通用知识库，再为大规模模型提供靶向指导，兼具经济性与实用性。
3. **CritBank 的离线构建 + 在线检索架构具备良好的工程复用价值**：一旦构建完成即可被大量下游查询分摊成本；可与团队现有在线服务架构对接，形成即插即用模块。
4. **失败模式标签体系（附录 H）可直接迁移至错误分析、红队测试与模型评估**：60+ 类失败模式为系统性误差诊断提供了现成的分类框架。
5. **与团队现有方向的潜在结合点**：可探索将 CritICL 的失败模式检索与团队在少样本推理、工具使用（tool-augmented reasoning）或主动学习上的工作结合，构造"失败感知型主动示例选择器"。

## 关键术语表
**弱到强泛化（Weak-to-Strong Generalization, W2SG）**：利用小规模模型或弱监督信号来提升大规模模型性能的现象与方法体系。
**CritBank**：由 CritICL 构建的离线结构化数据集，每条记录包含问题、错误回答、失败模式标签与批判性自然语言反馈。
**失败模式（Failure Mode）**：模型在推理过程中反复出现的结构化错误类型（如 incorrect_formula_application、logical_step_skipping）。
**Inference-time Scaling**：在推理阶段通过增加计算（多次采样、自我反思、外部验证等）提升模型表现的技术路线。
**CritICL-dynamic**：针对每个输入预测其最可能触发的失败模式，并据此从 CritBank 检索相关 critique 示例的推理时增强方法。
**CritICL-static**：基于模型家族层面的全局失败模式分布选取主导失败模式，对所有查询复用同一组 critique 示例的推理时增强方法。
**Failure Mode-Based Sample Selection**：按目标失败模式与候选样本失败模式的交集打分，并以贪心策略选取覆盖最多新失败模式的 K 个示例的检索过程。
**Test-time Scaling**：与 Inference-time Scaling 同义，指在测试/推理阶段额外分配计算以提高输出质量的策略。

## 可复现要素
- **数据集**：GSM8K、MATH、AMC23、AIME24、AIME25、GPQA（均为公开数据集）；CritBank 由作者自建，构建代码可见于 GitHub。
- **代码开源**：是，https://github.com/umwyf/CRITICL（论文已给出链接）。
- **权重**：使用开源 Qwen2.5 与 Llama 系列模型；失败模式与 critique 生成使用 gpt-4o-mini（论文声明）。
- **关键超参**：每问题生成 5 条回答；失败模式标签上限 5；检索示例数 K=5（消融实验默认）；CritICL-static 全局 Top-T 失败模式未明确给出具体 T 值（论文未提及具体数值，需查阅附录）。
- **解码设置**：所有实验采用 greedy decoding（temperature=0）。
