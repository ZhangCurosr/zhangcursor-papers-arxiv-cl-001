---
title: "From-Final-Artifacts-to-Trajectories-Retrospective-Process-S"
source: https://arxiv.org/pdf/2608.30461v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 16:24:54"
field: "证据锚定的长文本生成与Agent训练"
keywords: ["process supervision", "retrieval-augmented generation", "evidence-grounded generation", "self-improving agents", "trajectory reconstruction", "artifact-to-trajectory", "long-form generation"]
innovations: ["将专家最终产物视为压缩过程，通过回溯重构生成可验证的轨迹监督信号", "提出多维度Rubric-guided验证机制（rubric/quality/grounding/consistency）筛选自生成轨迹", "无需更强教师模型，仅用公开expert artifacts构建25K验证轨迹实现自改进SFT"]
benchmarks: ["ALCE", "ScholarQA", "QASPER", "LAiW", "LegalBench", "AstaBench (LitQA2, SQA)", "GSM8K", "MMLU-Pro", "IFEval", "HumanEval"]
---

# 论文速读：From-Final-Artifacts-to-Trajectories-Retrospective-Process-S

## 一句话总结
本文提出 RETROGEN，一种基于专家最终产物（artifacts）回溯重构轨迹的自改进过程监督框架，通过将高质量文献综述、分析报告、法律判决书等静态产物视为"压缩过程"，自动重构可验证的搜索轨迹并用于模型训练，无需更强教师模型即可显著提升证据锚定的长文本生成能力。

## 研究问题与动机
- **开放域证据生成缺乏可验证轨迹**：数学、编程等可验证领域可通过正确答案反馈，但文献综述、金融分析、法律判决等开放任务缺乏单一 ground truth，难以自动化验证中间推理路径。
- **现有轨迹获取方式不可扩展**：依赖人类专家标注成本过高；从更强教师模型蒸馏需要反复调用昂贵闭源模型，且前向轨迹生成易导致"漫游式搜索"（wandering search behaviors）。
- **高质量终产物资源丰富**：已发表的学术综述、合规的法律判决、机构分析师报告等专家产物在预训练语料中大量存在，可作为回溯重构的可靠锚点。
- **终产物蕴含潜在过程信息**：一篇结构严谨的 related work section 隐式编码了作者如何检索、筛选、比较和综合先前研究的完整流程，是一种"有损但信息丰富"的过程压缩。

## 核心贡献（创新点）
- **将专家产物重新定义为可验证过程目标**：提出将 expert-curated artifacts 视为压缩轨迹，使静态输出成为可回溯重构的监督信号，与已有工作仅用 final answer 作为监督目标的本质区别在于引入了"过程可逆性"假设。
- **提出 RETROGEN 自改进框架**：通过 Artifact-Anchored Initialization → Trajectory Reconstruction → Evidence-Constrained Verification 三步自动循环生成轨迹，无需外部教师模型或人工标注，与 WebGLM/WebCPM 等需更强系统蒸馏的监督形成对比。
- **多维度 Rubric-guided 验证机制**：设计 artifact fidelity、evidence faithfulness、procedural plausibility 三项 desiderata 对应的四个评分信号（rubric、quality、grounding、consistency），加权阈值筛选替代单一过滤，消融实验证明多维度组合效果最优（Table 2）。
- **在三大证据锚定领域验证有效性**：科学写作（相关文献综述）、金融分析（SEC 10-K 报告）、法律判决（中国一审裁判文书），在 ALCE/ScholarQA/QASPER/LAiW/LegalBench 等基准上取得平均 +4.8 分提升（Table 1）。

## 方法详解
### 3.1 Artifact-Anchored Initialization
- 从专家产物 $y^*$ 推断隐式任务规格 $\hat{x} = g_x(y^*)$ 与细粒度 rubric $R = \{(r_k, c_k)\}_{k=1}^K$，将产物特征映射为可验证标准（如法律文档中的"引用具体侵权法先例"）。
- 提取产物中的显式线索（citation、statutory reference、key entities）作为种子，在领域语料中进行 artifact-conditioned retrieval，恢复证据集 $\hat{\mathcal{E}}$。
- 完全 self-bootstrapped，无需人工过程标注。

### 3.2 Trajectory Reconstruction
- 给定 $\hat{x}$ 和 $\hat{\mathcal{E}}$，Agent 使用抽象动作词表 $\mathcal{A} = \{\text{Search, Open, Extract, Compare, Outline, Draft}\}$ 制定高层计划 $\hat{\pi} = (s_1, s_2, ..., s_T)$。
- 轨迹表示为 $\hat{\tau} = (\hat{o}_0, \hat{a}_1, \hat{o}_1, ..., \hat{n}_t, \tilde{y})$，其中观察 $\hat{o}_t$ 通过实际工具调用填充，保证实证忠实性而非叙事 plausible。
- 引入五种多样性扰动（query formulation、plan realization、outline granularity、reflection frequency、surface formatting）作为 procedural regularization。

### 3.3 Evidence-Constrained Verification
四个正交评分维度：
1. **rubric score** $s_{\text{rub}}$：$\tilde{y}$ 是否满足 self-induced rubric $R$ 中的各项标准。
2. **quality score** $s_{\text{qual}}$：宏观领域严谨性（连贯性、事实准确性、法律正确性等）。
3. **grounding score** $s_{\text{grd}}$：中间 notes $\hat{n}_t$ 与最终 claims 是否严格由 $\hat{\mathcal{E}}$ 支撑。
4. **consistency score** $s_{\text{con}}$：工作流内部逻辑一致性（compare 步骤引用的文档是否正确、outline 是否映射到最终结构等）。

综合评分：
$$\text{Score}(\hat{\tau}) = \lambda_r s_{\text{rub}} + \lambda_q s_{\text{qual}} + \lambda_g s_{\text{grd}} + \lambda_c s_{\text{con}}$$

阈值因域而定（科学 0.52 / 金融 0.48 / 法律 0.63），保留约 25K 条 verified trajectories。

### 3.4 Retrospective Process Supervision
- 训练样本形式：$(\hat{x}_i, \hat{\mathcal{E}}_i, \hat{\tau}_i^{\text{pre}}, \tilde{y}_i)$，序列化为 $z_i$。
- 使用合成产物 $\tilde{y}_i$ 而非原始 $y_i^*$ 作为 target，确保因果一致性。
- 损失函数为标准化 autoregressive LM loss：
$$\mathcal{L}(\theta) = \sum_i \sum_t \log p_\theta(z_{i,t} \mid z_{i,<t})$$
- SFT corpus 构成：80% verified agentic trajectories（科学:金融:法律=40:20:20）+ 20% 通用数据（instruction/multi-turn/math/coding）。

## 实验与结果
### 数据集与领域
- **科学写作**：arXiv 论文的 related-work section 作为 $y^*$，条件为 abstract + citation list。
- **金融分析**：SEC EDGAR 10-K 中 Item 7（MD&A）作为 $y^*$，环境为其余 filing 内容。
- **法律判决**：中国一审民事/行政判决书作为 $y^*$，需通过工具检索 statutes 和 prior cases。

### 评估基准
- **证据生成**：ALCE、ScholarQA、QASPER
- **领域推理**：LAiW、LegalBench
- **通用能力保留**：GSM8K、MMLU-Pro、IFEval、HumanEval

### 主要结果（Table 1，以 Qwen3-8B 为例）
| 方法 | ALCE | ScholarQA | QASPER | LAiW | LegalBench | Avg |
|---|---|---|---|---|---|---|
| Initial | 83.33 | 66.36 | 53.25 | 34.14 | 74.93 | 62.40 |
| ForwardGen | 85.18 | 65.42 | 51.84 | 33.52 | 74.95 | — |
| Artifact-Only | 87.86 | 67.18 | 52.52 | 34.69 | 74.93 | — |
| WebGLM | 84.52 | 65.84 | 53.36 | 36.61 | 76.12 | — |
| WebCPM | 46.73 | 60.27 | 49.89 | 23.23 | 74.92 | — |
| **RetroGen (Ours)** | **88.03** | **67.71** | **56.15** | **37.65** | **75.37** | **64.98** |

- **相对于 ForwardGen**：平均证据生成得分提升 **+4.8 分**（ALCE +4.3，ScholarQA +6.7，QASPER +3.5）。
- **相对于 Artifact-Only**：域推理提升 **+2.7 分**（LAiW +3.4，LegalBench +2.0），证明 trajectory 级监督有价值。
- **相对于 Public-Baseline**：多步证据聚合任务提升显著。

### 消融实验（Table 2，Qwen3-8B）
- 均匀加权四维度组合得分最高（Avg 64.98）。
- 单维度滤波均优于 Initial，但次优；next-N replace 次之，说明质量分级重要。

### AstaBench 动态 Agent 评估（Figure 2-3）
- **LitQA2**：Accuracy 0.20 → 0.40（翻倍），Precision 0.33 → 0.67。
- **SQA**：Global Avg 0.47 → 0.59；Citation Precision 0.47 → 0.67，Citation Recall 0.33 → 0.57。
- **动作转移矩阵**：初始模型 Search→Search 概率 0.54（冗余探索），RetroGen 降至 0.39；Search→Read 0.15 → 0.23；Read→End 0.20 → 0.55，表明学会"有目标的研究策略"。

### 通用能力保留（Figure 4）
- GSM8K、MMLU-Pro、IFEval、HumanEval 无系统性退化，混合 20% 通用 SFT 数据可避免 catastrophic forgetting。

## 相关工作脉络
- **RAG / 证据生成**：Lewis et al. (2020) 提出原始 RAG，本文与之对比的区别在于：RAG 通常监督 retrieve-then-write 的输入-输出映射，而 RETROGEN 显式重构多步 tool-use 轨迹。
- **WebGLM / WebCPM**：Liu et al. (2023)、Qin et al. (2023) 从更强系统蒸馏 citation-grounded 数据，但组织为 retrieve-then-write 格式；RETROGEN 用 expert artifacts 作为 target anchor，自生成轨迹。
- **Process Supervision**：Lightman et al. (2024) 在数学/代码领域展示中间步骤价值；RETROGEN 将过程监督扩展到开放证据生成任务。
- **Reasoning Bootstrapping**：Zelikman et al. (2022) STAR、Shao et al. (2024) DeepSeekMath；本文与之区别在于任务类型（非 verifiable 开放域）且无需外部 verifier。
- **Self-improvement**：Shinn et al. (2023) Reflexion 通过 verbal RL 迭代；RETROGEN 通过 artifact 回溯 + 自动验证实现无教师自改进。

## 局限性与未来方向
- **依赖高质量 expert artifacts 的可获得性**：在产物高度 underspecified、风格多样或弱证据锚定的领域，回溯重构可靠性下降。
- **实验局限于 retrieval-oriented 工具**：尚未评估 code execution、database query、multimodal perception 等更复杂交互环境下的表现。
- **阈值需手动校准**：当前 per-domain 阈值从分数分布中经验确定（Appendix B Table 3），缺乏自动校准机制。
- **未讨论对更强 backbone 的扩展性**：仅验证了 7B-8B 尺度 open-source LLM，未探索 scaling law。

## 研究启发与可借鉴点
- **"压缩过程"范式可迁移至其他静态产出**：任何包含结构化证据链的文档（如技术白皮书、专利说明书、临床指南）均可作为 retrospective trace 的候选源，值得在本团队方向中验证。
- **Rubric 自诱导机制（rubric extraction prompt）设计精巧**：从产物反向推断可验证标准的做法比固定 metric 更适应领域异质性，可参考其 prompt 模板（Appendix C）。
- **多样性扰动作为 procedural regularization**：随机化 plan flow、thought template、format 五种轴可有效避免 SFT 数据坍塌为单一模板风格，值得在 agent SFT pipeline 中复用。
- **AstaBench 动态评估 + action transition 分析**：超越静态 QA 度量，用状态转移概率刻画 agent 行为结构变化，为本团队评测研究提供新视角。
- **混合 SFT 配方（80% 任务数据 + 20% 通用数据）**：平衡领域专精与通用能力保留的经验配置，可作为后续工作的基线 recipe。

## 关键术语表
- **Retrospective Process Supervision**：从已知最终产物反推可能经历的工具使用轨迹，并以此作为训练信号的过程监督范式。
- **Artifact-Anchored Initialization**：以专家产物为核心锚点，同步推断任务规格、提取 rubric、恢复证据集的自举初始化步骤。
- **Rubric-guided Verification**：将产物中隐含的质量标准提取为结构化 checkable criteria，用于对候选轨迹进行多维度量化评分的验证协议。
- **Evidence Faithfulness**：轨迹中所有中间 notes 与最终 claims 必须严格被恢复证据集 $\hat{\mathcal{E}}$ 支撑，不允许 model-hallucinated observations。
- **Procedural Plausibility**：要求重建的工作流呈现逻辑连贯的领域标准步骤（search → inspect → compare → synthesize），而非跳跃式推理。
- **ForwardGen**：对照基线，指仅从推断任务规格 $\hat{x}$ 出发前向生成轨迹、不利用 expert artifact 的对比设置。
- **Next-N Replace**：消融变体，用次高分轨迹替换被筛除轨迹，测试数据量 vs 质量的权衡。
- **SFTA (Score-Filtered Trajectory Augmentation)**：隐含概念，指用验证通过的合成轨迹对模型进行 SFT 的训练目标。

## 可复现要素
- **数据集**：科学写作（arXiv related-work sections）、金融分析（SEC EDGAR 10-K MD&A，Loukas et al. 2021）、法律判决（中国公开一审裁判文书）；具体来源见 Appendix A.1，部分为公开数据，部分需自行收集。
- **代码**：论文未明确声明开源仓库，未提及 GitHub/URL。
- **权重**：使用开源 backbone（Qwen3-8B、Qwen2.5-7B、Mistral-7B-v0.3、Olmo-3-1025-7B）进行 SFT，未声明发布 fine-tuned weights。
- **关键超参**：
  - 最大 tool 步骤：30
  - 滑动上下文窗口：8,000 chars
  - 训练 token 预算：50M（agentic 80% + 通用 20%）
  - 保留轨迹数：约 25K
  - 验证权重：$\lambda_r = \lambda_q = \lambda_g = \lambda_c = 0.25$
  - 域阈值：科学 0.52 / 金融 0.48 / 法律 0.63
  - 优化器：AdamW，lr=1e-5，cosine schedule，warmup=5%，weight decay=0.01，gradient clip=1.0
  - 序列长度上限：32,768 tokens
  - batch size：~512K tokens/step
  - 训练轮次：1 epoch
