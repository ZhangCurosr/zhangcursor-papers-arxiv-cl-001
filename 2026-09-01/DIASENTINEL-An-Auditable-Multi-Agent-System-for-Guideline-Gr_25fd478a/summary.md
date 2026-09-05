---
title: "DIASENTINEL-An-Auditable-Multi-Agent-System-for-Guideline-Gr"
source: https://arxiv.org/pdf/2608.31128v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 06:35:15"
---

# 论文速读：DIASENTINEL-An-Auditable-Multi-Agent-System-for-Guideline-Gr

## 一句话总结
本文提出了 DIASENTINEL，一个完全本地部署的多智能体临床决策支持系统，通过结合校准后的 LLM 风险预测、确定性临床信号提取、基于 RRF 的 ADA 指南检索以及混合验证层，实现了可审计、防幻觉且符合临床指南的一年期 T2DM 风险筛查与结构化报告生成。

## 研究问题与动机
- **LLM 临床落地的三大隐患：** 预测概率未校准（无法反映真实发病率）、易 hallucinate 化验值或无依据建议、RAG 检索易产生 citation drift（引用错位）。
- **现有研究重算法轻部署：** 多数工作仅追求 AUROC 等指标提升，缺乏端到端可追溯、可审计的临床工作流设计，难以直接接入医院信息化环境。
- **指南检索存在重排序偏差：** 纯交叉编码器在临床文本上倾向给长段落高分，而有效答案常为短句推荐或表格，导致检索质量不升反降。
- **缺乏人机协同的容错机制：** 现有 LLM 医疗系统多为黑盒直出，缺少在报告生成后、临床使用前进行自动审计与人工覆写的闭环设计。

## 核心贡献（创新点）
- **提出“最小 LLM 介入”的可审计多智能体架构：** 将诊断流程拆解为独立子任务，仅 Risk Function、Synthesizer 和 entailment check 使用 LLM，其余全为确定性逻辑，从根本上压缩幻觉产生空间。
- **引入 RRF 融合的两阶段指南检索管道：** 结合 dense retrieval 与 bge-reranker 重排，通过 Reciprocal Rank Fusion 抵消单一 reranker 的长度偏好偏差，提升临床指南片段的命中精度。
- **设计混合验证层（Hybrid Verification）：** 集成四项确定性规则检查与 LLM 蕴含检查，在报告展示前自动审计事实一致性、数值篡改与引用漂移，并生成只追加的 JSONL 审计日志。
- **实现临床导向的概率校准与分层阈值策略：** 对 LoRA 微调的 Qwen2.5-14B 进行 Platt scaling，使预测概率与约 6% 的真实一年发病率对齐；基于 Youden’s J 与最低敏感度约束设定高危/中危分层阈值，契合筛查场景需求。

## 方法详解
- **系统编排与模式划分：** 基于 LangGraph 构建双模式管线：批量风险筛查（Risk Function 输出校准概率与高/中/低分层）与患者级详细报告生成（按需触发 Evidence Agent、Explanation Agent、Trend Agent、Synthesizer 与 Verification Agent）。
- **校准风险预测：** 使用 LoRA 微调 Qwen2.5-14B，输入结构化 EHR 特征，经 softmax 得原始概率后应用 Platt scaling（a=1.0049, b=-1.5693）。校准后概率作为统一风险分缓存复用，避免多次推理导致的不一致。
- **确定性临床信号提取（Explanation Agent）：** 定义 6 条硬阈值规则覆盖 HbA1c、空腹血糖、BMI、血压、LDL 及代谢综合征脂质模式，触发后输出结构化信号（变量、观测值、严重程度、 rationale），并按临床优先级排序。
- **纵向趋势追踪（Trend Agent）：** 对比 365 天窗口内首尾数值，结合变量特定噪声容忍带判定 `stable/rising/improving/insufficient_data`，输出结构化事实与 `summary_nl`；Synthesizer 被强制要求直接复用该摘要，阻断数值重算幻觉。
- **指南检索与引用绑定：** ADA Standards of Care (2026) 经 Docling 切分为 382 个 ≤512 tokens 的 chunk，用 bgem3 向量化存入 Chroma。推理时 dense 召回 → bge-reranker-v2-m3 重排 → RRF 融合得 `evidence_chunks`。Synthesizer 将每条推荐显式绑定 source/section/page 元数据，无法匹配检索结果的陈述仅输出无引用的通用建议。
- **混合验证与只追加审计：** 四项确定性检查分别核对报告与（i）风险分数/分层、（ii）EHR 数值、（iii）纵向趋势、（iv）指南引用的一致性；附加 LLM entailment check 验证语义支持度。所有结果以 pass/flag/skipped 写入只追加 JSONL 日志，医生可通过 ACCEPT/OVERRIDE 最终裁决，系统不自动改写或拦截报告。

## 实验与结果
- **数据集与划分：** 单医院去标识化真实 EHR，严格 train/test split。测试集 N=2,491，校准/评估集 N=4,982；一年期 T2DM 新发发病率约 6%（低患病率）。
- **风险预测性能：** DIASENTINEL 在测试集上 AUROC = 0.737 (95% CI: 0.694–0.773)，Brier score = 0.054，AUPRC = 0.146。Platt scaling 后平均预测风险从
