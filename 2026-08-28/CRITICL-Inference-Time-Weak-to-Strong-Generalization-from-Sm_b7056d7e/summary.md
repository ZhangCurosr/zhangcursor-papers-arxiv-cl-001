---
title: "CRITICL-Inference-Time-Weak-to-Strong-Generalization-from-Sm"
source: https://arxiv.org/pdf/2608.27455v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 12:28:32"
field: "推理时模型优化与弱到强泛化"
keywords: ["推理时弱到强泛化", "失败模式", "In-context Learning", "推理效率", "测试时缩放"]
innovations: ["提出 CritBank 离线构建失败模式知识库实现推理时弱到强知识迁移", "设计基于失败模式对齐的示例检索策略替代语义相似度检索", "实证揭示同族模型跨规模失败模式高度一致（Spearman 0.88-0.91）"]
benchmarks: ["GSM8K", "MATH", "AMC23", "AIME24", "AIME25", "GPQA"]
---

# 论文速读：CRITICL-Inference-Time-Weak-to-Strong-Generalization-from-Sm

## 一句话总结
本文提出 CritICL，一种高效的推理时弱到强泛化（W2SG）框架，通过离线构建 CritBank（包含弱模型失败模式标签和自然语言批评的结构化数据集），将小模型的系统性推理错误知识转移给同族大模型，以单次或少量生成实现推理能力提升，显著优于标准 ICL 并达到与测试时缩放方法相当的性能，同时大幅降低推理开销。

## 研究问题与动机
- **推理时缩放的计算成本过高**：Consistency、Self-Reflection、LLM-as-Judge 等方法依赖多次采样或外部验证，导致显著的重复推理开销，难以在延迟敏感场景应用。
- **现有推理时弱到强方法的局限**：W2S-AlignTree 等在线监督方法仍需为每个输入调用弱模型产生实时指导，未能充分挖掘弱模型失败模式的离线可复用价值。
- **弱模型失败模式具有跨规模一致性**：论文发现同族模型（如 Qwen 2.5B→72B、Llama 3B→70B）的错误分布在相对排序和频率上高度一致，暗示失败模式是可迁移的结构化信号而非随机噪声。
- **如何在推理时高效利用结构化失败知识**：核心研究问题是能否将弱模型的失败模式转化为离线可复用的批评资源，以极低的额外推理成本（仅 1-2 次生成）提升强模型推理表现。

## 核心贡献（创新点）
- **CritBank 构建框架**：首次提出将弱模型离线生成的错误响应与失败模式标签、自然语言批评结合，构建可重用的结构化失败知识库，与 Burn et al. (2024) 的训练时弱到强范式形成互补。
- **两种推理时失败模式检索策略**：CritICL-dynamic（输入自适应，预测每个问题的最可能失败模式）与 CritICL-static（模型族全局画像，提供稳定指导），填补了测试时失败模式检索的方法空白。
- **失败模式驱动的示例选择机制**：区别于传统 ICL 基于语义相似度或随机选择示例，CritICL 按失败模式匹配度排序检索示例，直击模型推理漏洞而非表面问题相似性。
- **实证揭示跨规模失败模式可迁移性**：定量证明同族模型间失败模式分布的 Spearman 相关系数达 0.88-0.91、JS 散度低至 0.04-0.05，为弱到强泛化提供新的理论支撑。

## 方法详解
**CritBank 构建（离线阶段）**：
- 对训练集问题 q，用 CoT 提示让弱模型 m 生成 5 个回答 R(q,m)。
- 分离正确/错误回答，对错误回答用 gpt-4o-mini 生成最多 5 个失败模式标签（从预定义 8 类中选取：incorrect_formula_application、problem_misinterpretation、logical_step_skipping、arithmetic_sign_error、insufficient_constraint_understanding、overcounting_in_combinatorics、geometric_relationship_misinterpretation、algebraic_manipulation_miscalculation），并通过聚类去重。
- 同时生成自然语言批评（critique），描述错误原因及修正建议。
- 最终 CritBank = {(q, r, l, C(q,r))}，支持按失败模式反向检索。

**CritICL-dynamic（推理阶段）**：
1. 对目标问题 q'，提示目标模型预测最多 5 个最可能的失败模式 S_inst(q')。
2. 基于失败模式匹配分数 score(q,r;S) = Σ w(l) 从 CritBank 检索 Top-K 批评示例。
3. 将检索到的失败模式对齐示例拼接到提示中，驱动目标模型单次生成答案。

**CritICL-static（推理阶段）**：
1. 聚合同族所有弱模型的失败模式频率，构建全局画像 P_M(l)。
2. 取 Top-T 高频失败模式作为检索目标。
3. 与 dynamic 相同方式检索示例并拼接，目标模型单次生成答案。

**关键设计要点**：检索时兼顾覆盖度与多样性——优先选择能引入新失败模式的样本，避免冗余；fine-grained 的失败模式分类（默认 8 类）在特异性与检索覆盖率间取得最佳平衡。

## 实验与结果
**数据集**：GSM8K（7.4k 训/1.3k 测）、MATH（7.5k 训/5k 测）；OOD 测试：AMC23、AIME24、AIME25；跨域：GPQA（化学/生物/物理/量子）。

**模型设置**：Qwen 族（弱：1.5B/3B/7B → 强：32B/72B）、Llama 族（弱：1B/3B/8B → 强：70B）。

**主要结果（Qwen2.5-72B，Table 1b）**：
- CritICL-static 整体平均 59.2%，超越 Consistency@5（59.0%）、Self-Reflection（58.2%）等最强基线，MATH 上达 84.0%（+0.8 over Consistency@5）。
- CritICL-dynamic 整体平均 58.7%，GSM8K 达 95.1%（+0.1 over Consistency@5）。
- CritICL-static 在 Llama-3.1-70B 上整体平均达 53.1%，超越 Consistency@5（51.3%）。

**成本对比（Table 2，MATH+Qwen32B）**：
- CritICL-static 仅需 1 次生成，总 token 3768（输入 3472 + 输出 296），远低于 Consistency@7（5440）、Self-Reflection（7533）。
- CritICL-dynamic 2 次生成，总 token 3897，仍低于所有测试时缩放方法。

**Ablation（Table 3）**：失败模式选择 vs 随机/固定/语义检索，在 AMC23/AIME25 上差距达 4-6 points；失败模式对齐是性能增益的核心来源（Table 7 消融验证）。

**跨族/跨域**：跨族迁移仍可提升但幅度小于同族；GPQA 科学推理同样有效（Table 16）。

## 相关工作脉络
- **Burns et al. (2024)**：开创性证明弱监督可提升强模型训练效果，但属训练时范式；本文转向推理时、不修改参数。
- **Ding et al. (2026) W2S-AlignTree**：在线 MCTS 搜索用弱模型监督信号，仍需多次弱模型推理；本文离线构建 CritBank，推理仅 1 次目标模型生成。
- **Madaan et al. (2023) Self-Reflection / Shinn et al. (2023) Reflexion**：通过多轮自我批判迭代优化输出，成本高；本文一次生成即得改进效果。
- **Didolkar et al. (2024)**：首次系统分析 LLM 数学推理失败模式分类；本文将其扩展为跨规模可迁移的资源并用于推理增强。
- **Wang et al. (2024) ICIL-retriever / Shao et al. (2025) ReasonIR**：语义相似度检索示例提升 ICL；本文证明失败模式对齐比语义对齐更有效（Table 3）。
- **Tyen et al. (2024)**：发现模型在给定错误位置时更擅长纠正；本文与之互补，离线构建错误位置库供推理时调用。

## 局限性与未来方向
- **同族依赖**：跨族迁移效果显著下降（Spearman 从 0.91 降至 0.46），限制了跨架构的通用性。
- **失败模式分类粒度敏感**：粗粒度损失特异性、细粒度导致检索稀疏，最优粒度需针对任务调优。
- **离线构建成本**：虽可复用，但 CritBank 构建需弱模型生成 + 大模型标注，对资源受限场景仍有门槛。
- **领域扩展有限**：当前主要在数学和 GPQA 验证，对开放域对话、代码生成等场景的有效性待验证。
- **未来方向**：探索跨族失败模式对齐、自动化粒度搜索、扩展至代码/开放问答领域、动态更新 CritBank 以适应模型迭代。

## 研究启发与可借鉴点
- **失败模式作为可复用推理资源**：将模型错误从"噪声"重新定义为"结构化知识"的思路，可迁移至其他领域（如代码生成、数学证明）的推理增强研究。
- **离线-在线计算迁移范式**：将重复的推理时开销转化为一次性离线成本（CritBank 构建后可复用于无数查询），为低延迟应用场景提供新思路。
- **失败模式对齐替代语义对齐**：示例选择目标从"问题表面相似"转向"推理漏洞匹配"，这一检索目标转移可推广至任何 ICL-based 应用。
- **弱模型失败画像的聚合效应**：多弱模型聚合比单弱模型更接近强模型失败分布，提示团队在多模型失败模式融合设计上可进一步优化。
- **可结合团队方向的创新机会**：可将 CritICL 的失败模式检索机制与团队的领域适配工作（如科学推理、代码补全）结合，构建领域专用 CritBank 以提升下游任务推理质量。

## 关键术语表
**CritBank**：离线构建的结构化数据集，包含问题、弱模型错误回答、失败模式标签及自然语言批评，供推理时检索使用。
**Inference-Time Weak-to-Strong Generalization (W2SG)**：利用弱模型知识在不修改强模型参数的推理阶段提升强模型表现的新型范式。
**Failure Mode**：模型系统性推理错误的分类标签，如公式误用、逻辑步骤跳跃、算术符号错误等。
**CritICL-dynamic**：输入自适应版本，目标模型先预测最可能失败模式再检索对应批评示例。
**CritICL-static**：模型族全局版本，基于弱模型聚合失败模式画像进行稳定检索。
**Test-Time Scaling**：通过多次采样/迭代/外部验证提升推理性能但增加计算开销的方法族。
**Pass@1 Accuracy**：单次生成答案的正确率，评估目标指标。
**Jensen-Shannon (JS) Distance**：衡量两个概率分布相似度的度量，越低表示失败模式分布越一致。

## 可复现要素
- **数据集**：GSM8K、MATH、AMC23、AIME24/25、GPQA（均为公开数据集）；CritBank 构建数据来自训练集（GSM8K 7.4k + MATH 7.5k = 15k 问题）。
- **代码/权重**：代码已开源 https://github.com/umwyf/CRITICL；未提及额外权重发布。
- **关键超参**：弱模型生成回答数=5；失败模式标签上限=5；检索示例数 K=5（dynamic）或按全局画像 Top-T（static）；目标模型 greedy decoding（temperature=0）；失败模式分类粒度=8 类（fine-grained）。
- **Prompt 模板**：见 Appendix G.1，含失败模式标注、批评生成、失败模式预测、最终答案生成四套模板。
