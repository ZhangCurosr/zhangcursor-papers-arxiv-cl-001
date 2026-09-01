---
title: "Multimodal-Rapport-Estimation-in-Real-World-HRI"
source: https://arxiv.org/pdf/2608.18401v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 19:43:53"
field: "真实场景人机交互与多模态情感/关系计算"
keywords: ["human-robot interaction", "rapport estimation", "multimodal late fusion", "zero-shot LLM", "real-world HRI", "third-party annotation", "consent correlation coefficient"]
innovations: ["构建真实零售场景第三方 CCR-8 标注集并建立多模态基线", "证明零样本文本 LLM 与视听嵌入的强互补性并实现预测级融合最优", "针对交互时长与群体大小给出真实 HRI 融洽度估计的性能分层分析"]
benchmarks: ["CCR-8 third-party rapport score", "30-fold session-level split with CCC/PCC/MAE evaluation"]
---

# 论文速读：Multimodal-Rapport-Estimation-in-Real-World-HRI

## 一句话总结
本研究在日本药妆店真实场景中收集 Wizard-of-Oz 人机对话多模态数据，构建第三方 CCR-8 融洽度标注集；对比零样本 LLM 与多模态预训练表征模型的自动融洽度估计性能，发现单模态文本零样本 LLM（Gemini 2.5 Flash）已具备强预测力，将其与 HuBERT/V-JEPA 嵌入预测做晚期融合可获得最优结果（CCC=0.656，PCC=0.717，MAE=0.471）。

## 研究问题与动机
1. 真实世界 HRI（如零售场景）中用户可随时加入/退出、交互时长不可控、常出现多方参与，而现有互动质量/融洽度评估方法主要在受控实验室条件下开发，泛化性存疑。
2. 以第三方视角自动估计“融洽度（rapport）”可作为改进对话策略与实现机器人自适应行为的基础，但目前真实场景下缺乏标注数据与可信基线。
3. 多模态特征在短时长、多方交互的真实场景下是否仍能提供互补信息，以及零样本 LLM 是否足以胜任该任务，尚不明确。
4. 需要一套可在真实服务部署中复用的融洽度自动估计流程与评估基准。

## 核心贡献（创新点）
1. **首个（之一）真实零售场景下的 HRI 融洽度标注数据集**：与已有 lab-based 工作不同，本文基于 62 场自然发生会话（97 个个体样本）并附第三方 CCR-8 标注，填补场景空白。
2. **建立多模态早期/晚期基线并系统对比零样本 LLM 与预训练特征**：首次在同一真实 HRI 任务中公平比较 Gemini/GPT/Claude 与 ST5/HuBERT/V-JEPA，并提供可复用的预测级融合范式。
3. **揭示文本 LLM 与视听嵌入的互补性**：证明 Gemini 2.5 Flash（仅文本）与 HuBERT/V-JEPA 在预测空间上低冗余，融合后各项指标显著提升，优于任一单支模型。
4. **给出真实 HRI 情境下的细分分析**：针对交互时长（≤40s vs >40s）与参与人数（1/2/3人）的 CCC 变化趋势分析，为后续模型鲁棒性设计提供证据。

## 方法详解
1. **任务定义**：对每位参与者构建实例 $(\mathbf{X}^{(m)}, y)$，目标为回归标量融洽分 $\hat{y}$；训练目标采用一致性相关系数 CCC 损失 $\mathcal{L}=1-\rho_c$，兼顾线性关联与均值/方差对齐。
2. **预处理**：基于 DETR 检测与 DEIMv2-Wholebody34 获取参与者边界框；Whisper-large 转录并切分发音段；人工校正说话人 ID 以对齐音频/视频/文本。
3. **单模态特征提取**：文本采用 Sentence-T5-large（ST5，768维，utterance级）；音频采用 HuBERT-large-ll60k（1024维，utterance级）；视觉采用 V-JEPA 2.1 ViT-Gigantic（1664维，64帧 clip，约 8fps）；均做 L2 归一化，视觉阶段结合边界框做加权空间池化。
4. **单支预测头**：各模态共用“加法注意力池化 + 两层 MLP（Dropout 0.2）”结构，将变长序列聚合为交互嵌入并回归 $\hat{y}$；输出先做 z-score 再反变换到原始 1-5 量表评估。
5. **零样本 LLM**：GPT-5.4、Claude Sonnet 4.6、Gemini 2.5 Flash 以日语 prompt 扮演第三方标注员，输入逐轮带说话人前缀的转录文本（T）、音频片段（T+A）及视频+边界框（T+A+V），返回结构化 JSON 后计算 8 项 CCR-8 均值。
6. **晚期融合**：采用等权预测级平均（unweighted average），无需额外训练参数；互补性分析以 Gemini(T) 为锚，用偏相关与 $\Delta R^2$ 量化其他分支增量。

## 实验与结果
- **数据集**：日本药妆店 Wizard-of-Oz 场景；经筛选 62 场会话、101 参与者，有效个体样本 $n=97$；平均会话时长 $54.23\pm42.42$ 秒；单/多人会话分别为 28/34 场。
- **标注质量**：三人平均 ICC(2,3)=0.85（个体）/0.86（群体），Cronbach's α≈0.93–0.96，可靠。
- **主要结果（Table 2/3）**：
  - 零样本 LLM 中，Gemini 2.5 Flash (T+A+V) 获最高 CCC=0.618；仅文本 Gemini 2.5 Flash (T) PCC=0.665 最高。
  - 预训练单模态中，HuBERT (A) 全指标领先；ST5 (T) 表现最弱（PCC=0.327，CCC=0.281）。
  - 最优融合：Gemini (T) + HuBERT + V-JEPA（T+A+V）达 MAE=0.471、PCC=0.717、CCC=0.656，优于所有单支与全监督融合（全监督 T+A+V 为 CCC=0.444）。
- **关键结论**：LLM 文本推理在真实短会话/多方场景更具鲁棒性；视听嵌入提供互补信号，但与 LLM 直接多模态处理相比仍能带来增益。

## 相关工作脉络
1. **Lin et al. CCR/CCR-8**：提出面向 HRI 的第三方融洽度量表；本文使用其 8 项短版量表进行标注与评测，是目标变量的基础。
2. **Speech-to-Joy (Santana et al.)**：lab-based HRI 中端到端/嵌入融合预测愉悦度；本文与其在数据分布、标签来源（第三方 vs 自陈）、交互长度与任务目标（rapport vs enjoyment）上存在关键差异，结果不可直接对比。
3. **Cerekovic et al.**：利用视听社会线索与人格预测 rapport；但其处于受控 lab 设定，本文聚焦无约束真实部署并对比 LLM 基线。
4. **Wei et al. (satisfaction/impression)**：多任务学习链接情感与印象；本文与之不同在于使用第三方 CCR 标尺并在真实零售环境取证。
5. **Gratch 等 (虚拟 agent rapport)**：奠定 rapport 构念与测量思路；本文将其扩展到机器人真实部署并强调多方/短时动态。
6. **Kanda et al. / Ben-Youssef et al. UE-HRI**：真实场景中自发互动的背景研究；支撑本文对自然进出、多方参与等场景特征的认识。

## 局限性与未来方向
1. 单一场景与文化（日本药妆店、日语用户），跨文化/跨语言/跨机器人形态的外推性待验证。
2. Wizard-of-Oz 操作者能力会影响互动质量；结论主要适用于远程操控社交机器人场景，向全自动机器人泛化需谨慎。
3. 样本量较小（$n=97$），按交互时长与群体大小细分后子集更小，结论偏探索性。
4. 仅覆盖有可用音频/转录的交互，极短或未参与的 encounter 未被充分捕捉。
5. 标签为第三方判断而非被试自陈，评估的是“可观测行为层面的融洽推断”而非内在体验。
6. 零样本 LLM 与监督嵌入的输入格式与推理框架不完全对齐，比较更接近实用基线而非理论最优对比；未来需更严格等条件实验并解释 LLM 使用的线索类型。

## 研究启发与可借鉴点
1. **预测级晚期融合的低成本高收益路径**：将零样本 LLM 作为稳定主干，再用轻量视听嵌入提供互补信息，避免端到端重训即可获得可观增益，适合资源受限的真实部署。
2. **真实 HRI 评估必须拆分交互时长与群体大小维度**：本文对两类关键现实因子的 CCC 分解揭示了监督嵌入的脆弱性，建议后续工作在其上增加场景分层评测。
3. **利用 CCC 而非单纯 MSE/PCC 作为训练目标**：CCC 同时惩罚均值与方差漂移，更贴合人与标注器之间的“一致判定”需求，可推广至其它主观连续打分任务。
4. **以“效用-冗余”二维图指导特征选择**：通过偏相关与 $\Delta R^2$ 定位与主干低冗余、高增量的模态分支，为融合架构设计提供可操作性准则。
5. **从单方标注到可解释提示工程**：零样本 LLM 表现强势但黑箱；后续可结合 prompt 可解释与关键片段检索，提升方法可审计性以适配隐私合规要求。

## 关键术语表
- **Rapport（融洽度）**：互动双方关系质量的动态构念，本文以第三方观测为依据进行评估。
- **CCR-8**：Connection-Coordination Rapport Scale 的 8 项精简版，面向 HRI 第三方视频评估。
- **CCC（一致性相关系数）**：联合衡量相关性与均值/方差一致性的指标，$\rho_c=1$ 表示完美一致。
- **Wizard of Oz（WoZ）**：由人类隐形操控、对外表现为自主机器人的实验/部署方式。
- **Late fusion（晚期融合）**：在多模型各自输出预测后，再进行加权/等权平均的融合策略。
- **Additive attention pooling**：通过可学习标量打分与 softmax 对变长特征序列进行加权的池化方式。
- **Utility–Redundancy map**：以与真值的关联为纵轴、与主干预测的关联为横轴，展示特征增量价值的可视化方法。

## 可复现要素
- **数据集**：本文构建的真实 HRI 数据集（日本药妆店 62 场、97 个个体样本），论文未声明公开。
- **代码/权重**：论文未声明开源代码；使用模型权重为预训练公开模型（ST5、HuBERT-large-ll60k、V-JEPA 2.1 ViT-Gigantic、Whisper-large、DEIMv2-Wholebody34）。
- **关键超参**：视觉采样频率约 8 fps、clip 长度 64 帧；MLP Dropout 率 0.2；融合策略为等权平均；训练折数 30-fold；早停与模型选择基于验证集。
- **评估指标**：MAE、PCC、CCC（主指标）；分维度统计使用 CCC 的 Connection/Coordination 子量。
