---
title: "Multimodal-Rapport-Estimation-in-Real-World-HRI"
source: https://arxiv.org/pdf/2608.18401v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:44:09"
field: "真实场景人机交互评估"
keywords: ["real-world HRI", "rapport estimation", "zero-shot LLM", "multimodal late fusion", "third-party annotation", "CCR-8", "Concordance Correlation Coefficient"]
innovations: ["首个真实零售 HRI 多模态 CCR-8 第三方标注数据集", "零样本 LLM 与预训练嵌入的互补融合框架", "基于 CCC 损失的第三方可泛化 rapport 回归训练目标"]
benchmarks: ["CCR-8 third-party scoring", "MAE/PCC/CCC regression metrics", "30-fold session-level split"]
---

# 论文速读：Multimodal Rapport Estimation in Real-World HRI

## 一句话总结
本研究在真实零售场景（日本药店）中收集多模态人机交互数据，首次利用零样本大语言模型（LLM）与预训练音视频特征模型进行第三方评定亲密关系（Rapport）评分的自动估计；最佳方案将 Gemini 2.5 Flash 文本输出与 HuBERT（音频）及 V-JEPA（视觉）嵌入进行晚期融合，CCC 达 0.656，显著优于单一模型或纯监督多模态融合基线。

## 研究问题与动机
- **核心问题**：真实世界非受控人机交互（HRI）环境中，如何可靠地自动估计第三方评定的互动亲密关系（Rapport）质量。
- **现有方法不足**：
  1. 多数 HRI 互动质量评估研究基于受控实验室设置，用户参与/退出时间固定、对话结构受控，难以直接迁移至自由出入、多 participant 并发的真实场景。
  2. 已有自动评估方法多依赖文本或单一模态特征，对短时、碎片化、群体性真实交互的鲁棒性不足。
  3. 第三方标注在真实 HRI 中缺乏大规模、公开且带多模态对齐标注的数据集支撑。
  4. 零样本 LLM 在多模态 HRI 实时评估中的潜力尚未被系统验证，尤其在与传统音视频嵌入模型对比与互补机制上缺乏实证。

## 核心贡献（创新点）
1. **首个面向真实零售 HRI 的多模态 Rapport 标注数据集**：在自然药店场景中收集 62 场会话、97 名参与者的对齐音视频文本数据，并采用 CCR-8 量表由第三方标注，填补真实场景标注数据空白。
2. **确立多模态基线并揭示 LLM 与嵌入模型的互补性**：系统对比零样本 LLM（GPT‑5.4、Claude Sonnet 4.6、Gemini 2.5 Flash）与预训练文本/音频/视觉嵌入（ST5、HuBERT、V‑JEPA），证明仅靠文本的零样本 LLM 即可达到强性能，且与音频/视觉嵌入融合可进一步提升。
3. **提出基于 CCC 损失的主干训练目标并构建晚期融合框架**：采用 Concordance Correlation Coefficient 作为训练损失，兼顾相关性与均值/方差一致性；设计未加权预测级融合策略，避免额外参数调优。
4. **开展跨交互时长与群体规模的实证分析**：发现监督嵌入模型在短交互和多 participant 场景下性能波动较大，而零样本 LLM 表现更稳定，揭示了真实 HRI 评估需考虑上下文变异性的关键洞察。

## 方法详解
- **任务定义**：对每个参与者实例，给定单模态特征序列 $\mathbf{X}^{(m)} = (\mathbf{x}_1^{(m)}, \ldots, \mathbf{x}_T^{(m)})$，学习映射 $f_\theta: \mathbf{X}^{(m)} \mapsto \hat{y}$ 预测标量 rapport 分数 $y \in \mathbb{R}$。
- **训练损失**：采用 CCC（Concordance Correlation Coefficient）作为优化目标：
  $$\rho_c = \frac{2 \sigma_{\hat{y} y}}{\sigma_{\hat{y}}^2 + \sigma_y^2 + (\bar{\hat{y}} - \bar{y})^2}, \quad \mathcal{L} = 1 - \rho_c$$
  该损失同时惩罚线性关联不足与均值/方差不一致。
- **特征提取**：
  - **文本**：Whisper‑large 转录+说话人识别，Sentence‑T5‑large（ST5）提取 768 维 utterance 级嵌入，L2 归一化。
  - **音频**：HuBERT‑large‑ll60k 提取 1024 维 utterance 级嵌入，L2 归一化。
  - **视觉**：V‑JEPA 2.1 ViT‑Gigantic 以约 8 fps 采样、64 帧 clip 提取 1664 维特征，配合目标用户 bounding box 的空间加权池化。
- **嵌入式模型架构**：每模态共享两阶段结构——先经 additive attention pooling 将变长序列聚合为交互级 embedding（公式 5‑7），再经两层 MLP（含 Dropout 0.2）预测 z‑score 尺度输出，逆标准化后得到原始 rapport 分。
- **零样本 LLM**：GPT‑5.4、Claude Sonnet 4.6、Gemini 2.5 Flash 以日语提示扮演第三方 annotator，按 CCR‑8 八条目 5 分制打分并输出 JSON；支持 T、T+A、T+A+V 三种输入条件。
- **晚期融合**：采用未加权平均（unweighted averaging）合并多模型预测分数；实验验证加权平均与未加权差异不显著。
- **评估协议**：30‑fold session‑level 划分，无数据泄漏；指标报告 MAE、PCC、CCC；随机基线从训练集分布有放回采样生成预测。

## 实验与结果
- **数据集**：日本药店 Wizard‑of‑Oz 远程操控机器人（Sota，Vstone）32 小时记录，过滤后得 62 场会话、97 名参与者（平均时长 54.23 秒，人均 7.06 次话语）。标注者为 3 名日语母语者，ICC(2,3)=0.85（个体）、0.86（群体），Cronbach's α 达 0.93–0.96。
- **关键结果**（表 2、表 3）：
  - **最佳单模型**：Gemini 2.5 Flash (T+A+V) CCC=0.618；Gemini 2.5 Flash (T) PCC=0.665。
  - **最佳融合**：Gemini (T) + HuBERT + V‑JEPA (T+A+V) 达到 MAE=0.471、PCC=0.717、CCC=0.656，较参考基线（Gemini T+A+V CCC=0.618）提升约 0.038 CCC。
  - **嵌入模型单独表现**：HuBERT (A) 在单模态监督模型中最优（CCC=0.460）；ST5 (T) 性能最弱（CCC=0.281）。
  - **互补性分析**：HuBERT+V‑JEPA (A+V) 相对 Gemini (T) 的部分相关 $r=0.359$，增量 $R^2=0.072$，证实音频/视觉提供独立信息。
- **时长分组分析**：以中位数 40 秒划分短/长交互；Gemini (T) CCC 差异仅 0.012，而 ST5 (T) 从 0.168 升至 0.411，显示监督文本模型在短时数据上退化严重。
- **群体规模分析**：Gemini (T) 在三人群组 CCC 最高（0.721），而 V‑JEPA (V) CCC 从 0.503 骤降至 0.043，表明群体干扰对视觉嵌入影响显著。

## 相关工作脉络
- **CCR 量表**（Lin et al., 2025/2026）：提出 Connection‑Coordination Rapport 双因子结构及 8 条目精简版，本文采用其作为第三方标注标准，区别于以往自评量表。
- **Speech‑to‑Joy**（Santana et al., 2025）：实验室环境下的 HRI 愉悦度预测，文本嵌入在该设置中接近音频信息量；本文在真实短交互场景中发现文本嵌入弱于 LLM，凸显场景差异。
- **多模态满意度/情感识别**（Wei et al., 2021/2022/2025）：在受控对话系统中利用多任务学习联合预测 exchange‑level 情感与 dialogue‑level 印象；本文聚焦真实 HRI 中的第三方 rapport，强调非受控环境下的鲁棒性。
- **Rapport 预测的音视频线索**（Cerekovic et al., 2017）：在实验室环境中从视听社会线索预测 rapport；本文扩展至自然零售场景并引入 LLM 作为强 baseline。
- **零样本 LLM 用于 HRI 评估**（Pereira et al., 2024）：探索 LLM 检测 enjoyment；本文系统比较三种主流 LLM 在 rapport 估计上的表现，并验证其与预训练嵌入的融合价值。
- **真实场景 HRI 数据集**（Ben‑Youssef et al., 2017; Nielsen et al., 2023）：UE‑HRI、YouTube 公众视频等；本文提供带精确 multimodal 对齐与第三方 CCR 标注的新公共数据集资源。

## 局限性与未来方向
- **场景与文化单一**：数据仅来自日本单一药店，参与者主要为日语母语者；跨语言、跨文化、跨机器人形态的泛化性未知。
- **WoZ 操控因素**：机器人言语/手势/注视由人工操作员控制，评分可能部分反映操作员社交能力；向完全自主 HRI 推广需谨慎。
- **样本量有限**：97 个个体样本对机器学习评估偏小；时长/群体规模子分析样本更少，结论具探索性。
- **标签为第三方而非自评**：估计的是旁观者判定的 rapport，非参与者内在体验；未对日文版 CCR‑8 进行完整心理测量学验证。
- **公平性比较受限**：LLM 与嵌入模型输入格式与推理框架不一致，结论为实用基线对比而非本质优劣证明；提示词、模型版本、API 设定均可能影响结果。
- **未来方向**：收集更大规模跨场景数据；开发端到端自动预处理（说话人跟踪/识别）；探究 LLM 内部推断线索；构建对时长与群体规模鲁棒的自适应融合架构。

## 研究启发与可借鉴点
- **零样本 LLM 可作为强大 baseline**：在资源受限或标注稀缺的真实 HRI 场景中，优先评估 T 模式零样本 LLM，再考虑是否引入昂贵音视频流。
- **晚期预测级融合比特征级融合更稳健**：未加权平均 LLM 与嵌入输出即可显著提升，避免复杂联合训练带来的过拟合风险。
- **CCC 损失更适合人类评分回归任务**：同时约束相关性与校准性，优于单纯 MSE 或 PCC，值得在多模态主观质量评估中推广。
- **互补性分析框架可复用**：利用 partial correlation 与 $\Delta R^2$ 量化模块贡献，为多模型选择提供量化依据。
- **场景变量（时长、群体规模）应纳入设计**：真实 HRI 系统必须报告这些维度上的性能分解，以便识别脆弱点并指导数据收集策略。

## 关键术语表
- **Rapport（亲密关系/融洽度）**：交互伙伴间动态形成的关系质量，包含相互关注、积极情绪与协调三个维度。
- **CCR‑8（Connection‑Coordination Rapport Scale 8‑item）**：面向 HRI 的第三方评估精简量表，含 Connection 与 Coordination 两个四条目因子。
- **CCC（Concordance Correlation Coefficient）**：综合衡量预测与目标间线性相关与均值/方差一致性的指标，取值 [−1,1]。
- **Wizard of Oz（WoZ）**：研究者远程操控机器人行为而用户不知情的实验范式，用于快速原型交互数据收集。
- **Additive Attention Pooling**：通过可学习线性打分函数计算 attention weight，将变长特征序列聚合为固定维度交互表示。
- **Late Fusion（晚期融合）**：在模型预测层而非特征层进行合并，通常以平均或加权方式集成多源输出。
- **Utility–Redundancy Map**：以单模型效用（与 ground truth 相关）与冗余度（与参考模型相关）为轴的定位图，用于识别互补模块。
- **Partial Correlation**：控制其他变量影响后两变量间的净相关，用于量化新增预测器的独立解释力。

## 可复现要素
- **数据集**：论文未公开数据集链接，但描述收集流程与过滤标准；需联系作者获取。
- **代码/权重**：论文未明确开源代码仓库；使用了 ST5、HuBERT、V‑JEPA 等公开预训练模型权重。
- **关键超参**：视频采样 8 fps、clip 长度 64 帧；ST5 768 维、HuBERT 1024 维、V‑JEPA 1664 维；Dropout 0.2；30‑fold session‑level split；未加权平均融合。
- **标注细节**：3 名 annotator，CCR‑8 日文版，Likert 1–5 分，个体与群体双级评分，以个体均分为主实验目标。
- **LLM 调用**：GPT‑5.4（reasoning effort none）、Claude Sonnet 4.6（无 extended thinking）、Gemini 2.5 Flash（API‑default thinking），2026 年 4 月单次运行。
- **评估指标**：MAE、PCC、CCC；CCC 作为训练损失与主要汇报指标。
