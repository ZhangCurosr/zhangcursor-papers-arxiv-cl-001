---
title: "Unmatched-Does-Not-Mean-False-Incomplete-Reference-Sets-Can"
source: https://arxiv.org/pdf/2608.25654v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 01:52:10"
field: "语言模型校准与评估"
keywords: ["Theory of Mind", "calibration", "label-source bias", "Brier risk", "open-ended evaluation", "prediction-powered inference", "TriSource-Restore"]
innovations: ["内容固定的标签来源效应识别与 Brier 排序反转证明", "TriSource-Restore 三源恢复方法", "select-repair-escalate 审计闭环协议"]
benchmarks: ["DoubtfulToM-Bench", "OpenToM", "NQ-open", "CuratedTREC"]
---

# 论文速读：Unmatched-Does-Not-Mean-False-Incomplete-Reference-Sets-Can

## 一句话总结
本文揭示了在开放型 Theory-of-Mind（ToM）追踪器评估中，**"未匹配即视为错误"的有限参考协议会系统性压低正例流行率，从而反转严格 proper 得分（Brier）的模型校准排序**；并提出了 TriSource-Restore 方法，仅需约 50 次概率采样的人工标注即可恢复正确排序并修复置信度校准。

---

## 研究问题与动机
- 开放型 ToM 追踪器随着证据流式到达持续生成细粒度命题（beliefs），其输出空间**不是预先固定的**，因此有限参考集合必然遗漏部分有效信念。
- 现有评估管道（有限参考 + 语义匹配器）将**未匹配输出一律标记为 false**，制造出有偏的代理标签，从而在相同固定输出下产生错误的校准排序。
- 这种"标签来源效应"不是模型能力变化，而是**评估协议本身的结构偏差**，在公开发布的 NQ-open DPR–BERT 校准流水线与独立审计的 OpenToM 叙事中均被实证发现。
- 需要一种**内容固定**（contents fixed）的设计来隔离标签来源效应，并提供可审计、可修复的闭环协议。

---

## 核心贡献（创新点）
1. **内容固定的标签来源效应识别**：在固定 259 条已裁定信念和 6 个场景下，证明有限参考标签将 Brier 风险排序反转（Δ = −0.227 → +0.152），区别于以往仅更换评估指标的做法。
2. **诊断真实流水线中的校准反向**：在已发布的 301 题 NQ-open DPR–BERT 流水线中，精确匹配与人工正确性两种标签导致 Scale/Average 两个校准规则的 ICE 方向**均显著反转**（区间不含零）。
3. **TriSource-Restore 三源恢复方法**：将完整帧参考标签 Z、冻结自动判断 Q 和概率采样人工真相 Y 三者结合，50 次尝试性标注即可在所有三个审计单元中以 ≥0.996 概率恢复正确排序，区间宽度最大缩减 37%。

---

## 方法详解
- **Brier 风险与标签源分解**：定义真实标签 Y（盲审人工逐条裁定）与参考衍生标签 Z（有限参考 + 匹配器），通过恒等式
  $$[R_Z(C_A) - R_Z(C_B)] - [R_Y(C_A) - R_Y(C_B)] = 2\mathbb{E}[(C_A - C_B)(Y - Z)]$$
  将排序反转归因于 $\mathbb{E}[(C_A - C_B)\mathbf{1}\{Y=1,Z=0\}]$（遗漏真值项 $T_{\text{omit}}$）与 $\mathbb{E}[(C_A - C_B)\mathbf{1}\{Y=0,Z=1\}]$（误报项 $T_{\text{fp}}$）之差。
- **实验控制**：DoubtfulToM-Bench 包含 6 个手工编写场景共 371 条任务命题；采用 Qwen2.5-7B 追踪器 + GPT-4o-mini 语义匹配器；EG（DoubtfulToM-EG）为冻结源先验探针，将置信度替换为常数 $(r_{\text{claim}}, r_{\text{inference}}, r_{\text{direct}}) = (0.20, 0.35, 0.55)$，仅改变 C 而不改变命题集合。
- **TriSource-Restore 估计**：以预测驱动推断（Prediction-Powered Inference）为框架，结合三源标签：
  $$\widehat{\Delta}_\beta = \Delta_Z + \bar{X}^\top \beta + \frac{1}{W}\sum_{i\in S}\frac{w_i}{\rho_i}[d_i(Y_i) - d_i(Z_i) - X_i^\top\beta]$$
  其中 $Q$ 为冻结置信度盲自动概率，$\rho_i$ 为包含概率，$\beta$ 受约束学习以提升效率但不改变目标无偏性；随后以单调 Platt 映射 $p_\theta(c) = \sigma(\alpha + \gamma \log\text{it}(c))$ 修复校准置信度。
- **部署门控（deployment gate）**：比较选中规则与基于标签一致的基线流行率常数，不优于则报告排序但标记规则低于部署阈值。

---

## 实验与结果
- **数据集**：DoubtfulToM-Bench（6 场景，10–30 事件/场景）、OpenToM（独立编写）、NQ-open（301 题子集）、CuratedTREC（444 题）。
- **主要结果**：
  - 固定 259 条信念：参考标签下 $\Delta_{\text{EG-N}} = -0.227$，人工裁定下 $\Delta_{\text{EG-N}} = +0.152$，**6/6 场景均反转**；流行率从 0.783 降至 0.295。
  - NQ-open 301 题：Exact match vs Human correctness 导致 Scale 和 Average 两个规则的 ICE 符号**双双反转**（Table 1 区间均不含零）。
  - OpenToM：90–96% 未匹配信念为字面真；Brier 差再次反转。
  - 遗漏真值主导失真：$T_{\text{omit}}$ 占总变异的约 57–65%。
- **TriSource-Restore**（Table 5）：50 次尝试性标注获得 46.1/44.0/43.6 可用标签；覆盖率 0.959/0.996/0.961；区间宽度缩减高达 37%；参考-only Platt 校准在所有单元恶化人工真 Brier。
- **最强结果**：保守 OpenToM 上 TriSource-Brier 0.0867，接近全训练 Y oracle 的 0.0866；局部 Local 单位宽度缩减 36.8%。

---

## 相关工作脉络
- **ToM 基准**（Le et al. 2019; Kim et al. 2023; Xu et al. 2024 等）：假设封闭问题集，侧重已完成叙事中的假信念评分；本文聚焦流式证据下开放输出空间的评估协议偏差。
- **开放域 QA 校准**（Si et al. 2022; Kamalloo et al. 2023）：使用有限精确匹配标签；本文保持其预测和置信度固定，仅替换正确性来源，证明同一流水线产生反向结论。
- **信息检索池化偏差**（Zobel 1998; Buckley & Voorhees 2004; Sakai & Kando 2008）：bpref/condensed-list 不把未 judgments 项标记为 nonrelevant；本文为该 warning 提供数值化度量并展示其如何翻转校准排序。
- **事实性评估**（Min et al. 2023 FActScore; Wei et al. 2024）：针对生成长文本的事实验证；本文研究对象是追踪器维持的细粒度命题，评估的是校准而非逐字复现。
- **预测驱动推断**（Angelopoulos et al. 2023; Fisch et al. 2024）：以模型预测为辅助、小样本人工标注为锚；本文 Eq. (3) 直接嵌入该传统，但提供完整的 select–repair–escalate 闭环协议与部署门控。

---

## 局限性与未来方向
- 人工审核依赖外部志愿者盲审，成本较高，目前仅在数百量级信念上可行；大规模部署需进一步降低标注负担。
- 六个压力场景为手工编写而非随机部署样本，推广至开放部署环境需更多实证。
- 未分别分离 matcher 假阴性与参考缺失的贡献，整体归因于"参考衍生标签管道"。
- 未来方向：扩展 TriSource-Restore 至更大规模流式 ToM 系统；探索自动化候选审核以减少人工预算；在多任务/多场景交叉验证部署门控阈值。

---

## 研究启发与可借鉴点
- **内容固定识别策略**：通过冻结输出、仅替换标签来源来隔离评估协议效应，比单纯比较不同指标更具因果说服力；可迁移到任何存在"未标注即否定"偏置的评测场景。
- ** prevalence collapse 诊断框架**：以 $\pi = P(Y=1 | Z=0)$ 为核心诊断量，将失真归因于流行率坍塌；这一思路可推广至 POSITIVE-UNLABELED / label-shift 等设置。
- **TriSource-Restore 的三源设计**：参考标签 Z + 自动判定 Q + 人工采样 Y 的结合方式，为低成本高精度校准修复提供了通用模板；可直接借鉴到 RAG 系统或长文生成校准评测。
- **select–repair–escalate 协议**：分阶段决策（先采样估计、再选择规则、再修复置信度、最后部署门控）为审计类工作流提供了可复用的工程化范式。
- **与团队方向结合机会**：若团队涉及开放域 QA 事实校验、RAG 响应置信度评估或 LLM 事实性评测，可将本论文的标签源分解框架与 TriSource-Restore 接入现有评测管线，作为"参考不完整"场景的标准审计步骤。

---

## 关键术语表
- **Unmatched ≠ False**：开放输出空间中未与有限参考匹配的命题不一定为假，仍可能为真。
- **Literal Truth**：基于完整上下文逐条裁定命题的字面真实性，而非匹配参考或评估下游实用性。
- **Label-Source Decomposition**：将 Brier 风险差分解为遗漏真值项 $T_{\text{omit}}$ 与误报项 $T_{\text{fp}}$ 的贡献，用于量化标签来源偏置。
- **TriSource-Restore**：融合完整帧参考标签 Z、冻结自动判断 Q 和概率采样人工真相 Y 的三源恢复方法。
- **Instance-Level Calibration Error (ICE)**：衡量每个样本置信度与标签误差的平均绝对偏差 $n^{-1}\sum|L_i - c_i|$。
- **Proper Score / Strictly Proper**：Brier 为严格 proper 得分，期望风险在真实概率处唯一最小，避免 binning 引入的人为偏置。
- **Prevalence Collapse**：将未匹配输出标为 false 导致正例加权流行率从 0.783 骤降至 0.295。
- **Deployment Gate**：比较选中规则与基于流行率的基线常数，不优于则阻止部署但保留排序结论。

---

## 可复现要素
- **数据集**：DoubtfulToM-Bench（作者编写）、OpenToM（独立编写）、NQ-open 301 题子集（来自 Si et al. 2022 / Kamalloo et al. 2023 发布）、CuratedTREC（Kamalloo et al.）。论文未声明完整数据集开源链接。
- **代码/权重**：论文未明确提及开源代码仓库或模型权重链接；TriSource-Restore 算法伪代码见 Algorithm 1，实现细节在 Supplement A.11。
- **关键超参**：EG 冻结先验 $(r_{\text{claim}}, r_{\text{inference}}, r_{\text{direct}}) = (0.20, 0.35, 0.55)$；Platt 映射约束 $\gamma \geq 0$；$\beta$ 满足 $\beta_k \geq 0, \sum_k \beta_k \leq 1$，仅在源域残差 MSE 改善 ≥5% 时启用非零 $\beta$。
- **标注预算**：50 次尝试性人工标注（产生约 43–46 条可用标签）。
