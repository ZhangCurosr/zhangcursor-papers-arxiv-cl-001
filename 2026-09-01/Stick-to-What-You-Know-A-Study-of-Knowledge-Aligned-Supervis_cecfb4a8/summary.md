---
title: "Stick-to-What-You-Know-A-Study-of-Knowledge-Aligned-Supervis"
source: https://arxiv.org/pdf/2608.30987v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 20:51:16"
field: "大语言模型事实性与幻觉缓解"
keywords: ["knowledge-aligned SFT", "hallucination mitigation", "factual consistency", "supervised fine-tuning", "claim decomposition", "recusal behavior"]
innovations: ["提出统一的知识对齐SFT框架，将现有方法纳入同一数据构造流程", "提出Recall Rewrite方法，通过多样化探测问题一致性检验基座模型参数化知识", "揭示知识对齐比例与事实性的单调因果关系"]
benchmarks: ["WildHalu", "Biography", "UnknownBench", "OLMES"]
---

# 论文速读：Stick-to-What-You-Know-A-Study-of-Knowledge-Aligned-Supervised-Fine-Tuning

## 一句话总结
本文研究通过知识对齐监督微调（Knowledge-Aligned SFT）减少大语言模型幻觉的问题，提出两种新方法 Evidence Rewrite 和 Recall Rewrite，实验表明该方法能在基本不损害通用能力的前提下显著降低事实性幻觉。

## 研究问题与动机
1. **SFT 诱发幻觉的根源**：监督微调将预训练基座模型训练为指令跟随模型时，训练数据中的事实性目标可能包含基座模型未掌握的知识，迫使模型在不确定性下生成看似合理但无事实依据的声称，从而产生幻觉。
2. **现有缓解方法的不足**：FLAME 等生成式方法直接用基座模型生成替换训练目标，但基座模型自身也可能产生幻觉；UNIT_cut 等基于置信度的方法依赖 token 级概率信号，对表述和位置敏感，且低估了隐含知识（如元知识）的作用。
3. **SFT 并非注入新知识的有效途径**：可靠学习新的事实关联需要大量证据支持，而 SFT 阶段数据量相对较小；检索增强或继续预训练更适合知识注入。
4. **行为信号混淆**：SFT 同时教授模型"如何回答"和"回答什么"，当训练目标超出模型知识边界时，模型被奖励在不确定下猜测而非拒绝回答。

## 核心贡献（创新点）
1. **提出知识对齐 SFT 的统一框架**：将现有方法（FLAME、UNIT_cut）和新方法纳入同一数据构造流程，明确定义"已知/未知"声明的分类标准，与已有工作相比提供了系统化的理论分析和统一实验对比。
2. **提出 Evidence Rewrite 方法**：在 FLAME 的生成式对齐基础上增加外部证据验证步骤，通过claim分解、证据检索和验证过滤，仅保留有外部证据支持的声明，区别于 FLAME 盲目接受所有自生成内容的做法。
3. **提出 Recall Rewrite 方法**：无需外部证据，通过生成多样化探测问题并让基座模型重复回答，检验声明是否能被一致回忆，比 UNIT_cut 的 token 级置信度估计更直接地探测参数化知识。
4. **揭示知识对齐程度的因果效应**：通过控制已知声明比例（%Known）的消融实验，在固定训练集规模和拒绝率的情况下，证明提升知识对齐比例能单调改善事实性指标。

## 方法详解
**统一数据构造流程**（Algorithm 1）：对每条训练数据 (P, R)，首先判断是否为知识-seeking prompt，若是则进行 claim 分解，分类每个声明为 known/unknown，保留 known 声明并通过 rewriter 生成对齐目标 R* 或拒绝响应。

**Evidence Rewrite**（Figure 2）：
- 对知识-seeking prompt，从基座模型采样长回答 $\hat{R}$
- 使用 VeriScore 进行 claim 分解，从 Wikipedia 检索 top-5 证据片段
- 使用 FActScore 验证器判断每个 claim 为 SUPPORTED 或 NOT_SUPPORTED
- 引入 brainstorming 步骤鼓励基座模型生成更详细的回答以缓解过滤后的信息损失
- 使用 rewriter 模型根据支持性 claim 重写响应，信息不足时返回拒绝模板

**Recall Rewrite**（Figure 3，核心方法）：
- 将 gold response R 的 claim 分为 knowledge-dependent（需参数化知识）和 non-knowledge-dependent（上下文/主观/通用推理）
- 对每个 knowledge-dependent claim $c_n$，用 teacher 模型生成 J=5 个独立探测问题 $\{q_{n,j}\}$
- 从基座模型采样 K=2 个答案 $\{y_{n,j,k}\}$
- 对每个问答对进行 entainment 判断：ENTAILS / CONTRADICTS / UNRELATED
- 声明被分类为 known 的判定条件（Equation 1）：
  - 至少 $j_e=2$ 个问题的 entailing 答案数 $\ge k_e=1$
  - 至多 $j_c=2$ 个问题的 contradicting 答案数 $\ge k_c=1$
- 使用 rewriter 移除 unknown claim 相关内容，保留原始结构和风格

## 实验与结果
**实验设置**：
- 基座模型：Qwen3-4B-Base、OLMo3-7B
- 训练数据：OASST1 英文子集（3,468 条首轮对话）
- 评估基准：WildHalu（500个真实实体，约半数无Wikipedia页面）、Biography（500个Wikipedia人物）
- 拒绝行为评估：UnknownBench（FalseQA、NEC、RefuNQ）
- 通用能力评估：HumanEval+、GSM8K、IFEval、TruthfulQA

**主要结果**（Table 1，Qwen3 4B）：
- WildHalu：Recall Rewrite 的 %Supp. 达 84.2%（标准 SFT 76.6%，提升 7.6pp），FActScore 84.1（标准 SFT 74.4%，提升 9.7pp）
- Bios：Recall Rewrite 的 %Supp. 达 56.2%（标准 SFT 36.0%，提升 20.2pp），FActScore 76.4（标准 SFT 34.1%，提升 42.3pp）
- Evidence Rewrite 同样优于标准 SFT 和 FLAME
- FLAME 与标准 SFT 无显著差异，说明 naive 生成式对齐不可靠
- UNIT_cut 优于生成式方法但弱于 Recall Rewrite

**拒绝行为**（Table 4，UnknownBench）：
- Recall Rewrite 在所有子任务上获得最高 F1-Score
- FalseQA：F1=68.7；NEC：F1=68.8；RefuNQ：F1=69.9
- 以更低 precision 换取更高 recall，体现更保守的响应策略

**通用能力**（Table 5）：
- Recall Rewrite 的 OLMES 平均分 68.9，与标准 SFT 的 69.8 相差仅 0.9pp
- WildHalu FActScore 提升 10pp、Bios 提升 42pp 的情况下，通用能力基本保持

**OLMo3 7B 对比**（Table 2）：
- Recall Rewrite 仅用 SFT 即在 WildHalu 上超过 OLMo3 Instruct 的 SFT、DPO、RLVR 各阶段 checkpoint
- Bios 上略低于 RLVR checkpoint

**%Known 消融**（Table 3）：
- 100% known 条件下 FActScore 最高（WildHalu 86.1，Bios 69.9）
- 证实知识对齐比例与事实性呈单调正相关

## 相关工作脉络
1. **FLAME（Lin et al., 2024）**：用基座模型生成替换 gold response，假设自生成内容受限于参数化知识；本文指出该方法忽略基座模型自身幻觉，且可能丢失原 response 中 model 已知但未被生成的正确 claim。
2. **UNIT_cut（Wu et al., 2025）**：基于 claim-conditioned probability (CCP) 过滤高不确定性声明；本文指出 token 级置信度对表述敏感，且无法捕捉隐含的元知识。
3. **Gekhman et al. (2024)**：在 closed-book QA 中证明 fine-tuning 未知知识会增加幻觉；本文在 open-domain instruction data 中复现并扩展该发现。
4. **Calderon et al. (2026)**（同期工作）：提出一致的 recall 判据判断事实是否被模型掌握，与本文 Recall Rewrite 理念相似，但本文提供完整的数据构造 pipeline 和 SFT 实验验证。
5. **Kaplan et al. (2026)**：将 SFT 幻觉归因于 factual forgetting，通过优化级正则化缓解；本文从数据侧干预而非优化侧，方法正交。
6. **FActScore（Min et al., 2023）/ VeriScore（Song et al., 2024）**：factuality 评估框架；本文沿用并改进 claim 分解策略以适应开放域指令数据。

## 局限性与未来方向
1. **知识边界近似的不完美**：无法直接观测 $\kappa(M_{base})$，所有方法仅提供近似；二值化已知/未知分类忽略了模型的渐变置信度和部分知识。
2. **评估基准的实体中心局限**：WildHalu 和 Biography 均围绕单一命名实体展开，未测试非实体组织的 factual content。
3. **自动 factuality 测量的误差**：claim 分解、证据检索和验证各阶段可能出错，尤其对 underspecified 或需要领域专家知识的 claim。
4. **与后续 post-training 阶段的交互未充分研究**：未测试 Recall Rewrite 与 DPO/RLVR 等事实导向后训练目标的叠加效应。
5. **可扩展性限制**：Recall Rewrite 流程复杂（分解、探测、采样、验证、重写），目前更适合作为高精度诊断干预而非全量数据准备方案。
6. **teacher 模型偏差**：依赖强 teacher 模型（gpt-4o-mini、gpt-5-mini）引入自身偏差，需用 open-weight teacher 或人工审计分离偏差来源。

## 研究启发与可借鉴点
1. **数据侧干预幻觉的思路**：与其在优化层面正则化，不如从训练数据质量入手，将知识对齐作为 SFT 数据构造的可控变量，这一范式可迁移至其他对齐场景。
2. **Recall Rewrite 的 probing 机制**：通过多样化重表述问题探测模型知识掌握程度，比 token 级置信度更鲁棒；该思路可应用于知识图谱补全、模型能力评估等领域。
3. **覆盖率-事实性权衡的显式建模**：本文通过 %Known 和 filter threshold 提供训练时的可控旋钮，为后续工作优化 trade-off 提供量化框架。
4. **拒绝行为的训练期塑造**：证明 SFT 阶段即可影响模型何时拒绝回答，无需依赖 inference-time 阈值调节，可与 selective abstention 方法结合。
5. **claim decomposition 的改进实践**：使用 VeriScore 而非 FActScore 分解 open-domain 数据，避免 unverifiable claims（如代词指代不明、meta-commentary），这一经验可直接复用于其他 factuality 研究。

## 关键术语表
**Knowledge-aligned SFT**：将监督微调的训练目标限制在基座模型参数化知识范围内的方法，通过过滤或重写训练数据中的事实性声称减少幻觉。
**Parametric knowledge $\kappa(M_{base})$**：基座模型在预训练阶段内化并robust存储的事实性知识集合， inherently incomplete。
**Consistently recalled**：Recall Rewrite 中声明被分类为 known 的判据，要求至少 $j_e$ 个独立探测问题获得 entailing 答案且至多 $j_c$ 个问题获得 contradicting 答案。
**Knowledge-dependent claim**：编码可验证事实、程序或结构信息的声明，需要基座模型的参数化知识；与之相对的 non-knowledge-dependent claim 仅依赖上下文或通用推理。
**FActScore**：per-example  supported 声明比例的平均值，拒绝响应被视为 fully supported，用于量化长文本生成的事实精度。
**WildHalu**：包含500个真实世界实体查询的 benchmark，约半数实体无 Wikipedia 页面，用于评估开放域长文本生成的事实性。
**Recall Rewrite**：本文提出的核心方法，通过生成探测问题并检验基座模型能否一致回忆 gold response 中的每个声明，实现无需外部证据的知识对齐。
**Evidence Rewrite**：本文提出的方法，对基座模型生成内容进行 claim 分解和外部证据验证，仅保留有证据支持的声明用于 SFT。

## 可复现要素
- **训练数据集**：OASST1 英文子集（3,468 条首轮对话），论文已开源 Recall Rewrite 训练数据及所有中间 pipeline 输出（claim 分解、探测问题、基座模型答案、entailment 标签）
- **基座模型**：Qwen3-4B-Base、OLMo3-7B（均为开源模型）
- **代码/权重**：论文未明确提供代码仓库链接，但声明 release 训练数据；TRL library 用于 SFT 实现
- **关键超参**：Epoch=5，Batch size=32，Learning rate=1e-5，Cosine warmup ratio=0.1，Weight decay=0.1，Context length=1024
- **Recall Rewrite 默认阈值**：$j_e/k_e/j_c/k_c = 2/1/2/1$，J=5 个探测问题，K=2 个答案采样，temperature=0.5
- **Teacher 模型**：Evidence Rewrite 使用 gpt-4o-mini，Recall Rewrite 使用 gpt-5-mini
- **硬件**：1× NVIDIA A100 (80GB)
