---
title: "The-Visual-Insensitivity-Gap-Diagnosing-When-Vision-Language"
source: https://arxiv.org/pdf/2609.00868v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 13:23:20"
field: "多模态大模型可解释性与评估"
keywords: ["Vision-Language Models", "Hallucination", "Visual Insensitivity", "Selective Generation", "Encoder-LLM Disconnect", "Interpretability"]
innovations: ["提出VSI指标量化样本级视觉不敏感性，揭示40%-97%样本上模型忽略视觉证据", "证明视觉不敏感性是sample-intrinsic属性（跨模型Spearman ρ=+0.40）而非model特性", "揭示encoder-LLM disconnect机制：线性probe准确率0.72-0.79 vs argmax变化率2%-11%", "在多选题推理（MMStar math/science）上达到AUROC 0.85-0.87的诊断能力"]
benchmarks: ["POPE", "MMVP", "HallusionBench", "MMStar"]
---

# 论文速读：The Visual Insensitivity Gap: Diagnosing When Vision-Language Models Fail to Use Visual Evidence

## 一句话总结
论文提出"视觉不敏感性差距"（Visual Insensitivity Gap）概念，通过样本级的视觉敏感性指数（VSI）量化VLMs对视觉证据的忽略行为——在40%-97%的感知基准样本上，模糊问题相关视觉区域几乎不改变模型的下一个token分布；该现象是样本固有属性（跨模型Spearman ρ=+0.40），且在Qwen2.5-VL-32B的多选题推理任务上达到AUROC=0.85-0.87的诊断能力。

## 研究问题与动机
1. **聚合准确率的隐性假设失效**：现有VLM评估依赖benchmark上的聚合准确率（如POPE、MMVP、HallusionBench、MMStar），隐含假设模型"使用了视觉输入"，但论文证明该假设在40%-97%的样本上不成立。
2. **自信错误 vs. 谨慎错误的本质差异**：在临床分诊、文档问答、具身智能等决策关键场景中，"忽略视觉证据却给出高置信度错误答案"与"谨慎回答但参与视觉推理"是两类不同性质的失败，聚合指标无法区分。
3. **现有基线的盲点**：softmax max-probability（内在预测置信度）和verbalised confidence（语言模型自我评估）均基于输出分布计算，无法区分"自信但使用了视觉证据"与"自信但忽略了视觉证据"。
4. **输入级干预的因果诊断需求**：注意力图等方法提供描述性映射，但不建立"编码的信息是否被用于输出"的因果链；需要移除视觉证据并观察softmax变化来验证。

## 核心贡献（创新点）
1. **形式化Visual Insensitivity Gap并提出VSI指标**：用KL散度量度问题相关视觉区域模糊前后next-token分布的变化，首次将"模型是否使用视觉"量化为样本级标量；区别于之前仅报告聚合准确率的做法，VSI揭示heavy left tail结构。
2. **证明视觉不敏感性是样本固有属性而非模型特性**：跨15对模型对的grand-mean Spearman ρ=+0.40（permutation p<10⁻³），即使共享架构仅止于contrastively pretrained vision tower的跨家族模型对也保持0.20-0.55的相关性。
3. **揭示encoder-LLM disconnect机制**：在低VSI样本上，模型自有vision tower的线性probe准确率0.72-0.79，但argmax token仅2%-11%发生变化，gap达0.65以上，证明信息存在于编码器但路由未传递至输出端。
4. **绘制VSI诊断价值的条件性地图**：在POPE/HallusionBench等事实性基准上VSI是hallucination信号（dangerous quadrant密度是top-VSI的2.1×），在多选题推理（MMStar math/science）上AUROC达0.85-0.87，但在校准良好的场景下softmax max-prob仍占优（10/18 cells胜出）。
5. **提出conditional ensemble框架**：VSI+whole-image VSI的等权z-score集成在8/18 cells中表现最佳，hybrid信号在calibration差或hallucination主导的cell上比单一信号高+0.09至+0.18 AUROC。

## 方法详解
**VSI定义**：对VLM f，给定图像x和问题q，令x_σ为对问题相关视觉区域施加σ=20高斯模糊的扰动图像，则
$$\text{VSI}(x, q; f) = \text{KL}(f(\cdot|x, q) \| f(\cdot|x_\sigma, q))$$
衡量输出分布的整体偏移（不仅捕获argmax翻转，还包括分布形态变化）。

**问题相关区域定位管线**：(1) 轻量级解析器从问题q提取主名词短语；(2) Grounding-DINO（base）以0.3置信度阈值查询开放词汇bounding box；(3) SAM-ViT-base以box为prompt细化为像素mask，并膨胀4像素避免边界伪影；无可靠box时回退至whole-image模糊（fallback率<3% on POPE/MMVP，~19% on HallusionBench因图表题目）。

**Encoder-LLM disconnect实验**：对83个被Qwen3-VL标记为低VSI（<0.05）的样本，在每个模型自有vision tower的最后层特征上训练L2正则逻辑回归线性probe（C=1.0，ℓ2归一化特征，grouped 5-fold CV），标签为perturbed(+1)/unperturbed(-1)；同时记录扰动后argmax token变化率。Gap = probe acc - Δargmax，所有模型均在0.65-0.71。

**诊断评估指标**：AUROC以per-sample error为正类、-VSI为排序分数；PRR@80为80%覆盖率下的prediction-rejection ratio；四象限分析按bottom-VSI quintile内correct/wrong × high/low softmax（以cell中位数划分）划分。

**VSI变体**：Region-blur VSI（局部扰动）和Whole-image VSI（整图模糊），两者Spearman跨18 cells均值为+0.24，捕捉重叠但不同的样本子集。

## 实验与结果
**模型**：六款VLM跨越三架构家族、四种vision tower：LLaVA-1.5-7B、LLaVA-NeXT-Mistral-7B（CLIP-ViT-L/14）、Idefics3-8B（SigLIP-SO400M）、Qwen3-VL-8B、Qwen2.5-VL-7B、Qwen2.5-VL-32B（native ViT）。

**基准**：三个感知基准（POPE、MMVP、HallusionBench）+ 一个多选题推理基准（MMStar）。

**VSI分布**：low-VSI（<0.05）样本占比40%-97%，HallusionBench最高（图表/图示降低局部扰动信息量），POPE最低；whole-image variant insensitive fraction仅~6%，证实局部heavy tail非扰动artifact。

**跨模型一致性**：grand-mean Spearman ρ=+0.40（15对模型对），same-family对均值ρ=0.51 vs cross-family对均值ρ=0.37；permutation test全部p<10⁻³（最弱对Qwen2.5-VL-32B vs Idefics3 on MMVP: ρ=+0.20, p=3.4×10⁻⁴）。

**Encoder-LLM gap**：低VSI样本上own-tower probe acc 0.72-0.79，Δargmax仅2%-11%，gap 0.65-0.71；高VSI control样本上probe acc升至0.86-0.91且Δargmax同步上升，gap闭合。

**诊断最强结果**：MMStar上Qwen2.5-VL-32B的math类别AUROC=0.851（95% CI [0.73, 0.94]），science类别AUROC=0.867（[0.78, 0.95]）；bottom-quintile错误率是top-quintile的4.05倍（science）。

**Selective generation**：18 cells中max-prob赢10个（校准良好场景）、hybrid含VSI成分赢7个（calibration差/hallucination主导场景）、VSI单独赢1个；hyb(region+whole)在Qwen3-VL POPE上AUROC 0.636 vs max-prob 0.544（+0.09, p<0.01），在Qwen2.5-VL-7B MMVP上AUROC 0.676 vs max-prob 0.496（+0.18）。

**鲁棒性**：σ∈{10,20,40}变化下AUROC变动≤0.05，per-sample rank相关ρ=0.76-0.97（均值0.89）；VSI阈值在{0.01, 0.05, 0.10, 0.20}间变化时错误率方向不变，bottom/top quintile误差比变动≤1.3×。

## 相关工作脉络
1. **Hallucination evaluation in VLMs**：POPE（Li et al. 2023）、MMVP（Tong et al. 2024b）、HallusionBench（Guan et al. 2024）、MMStar（Chen et al. 2024）等新基准提出，但均报告聚合准确率；本文将其作为substrate，在样本级分解使heavy-tailed VSI分布可见。
2. **Selective generation与校准**：max-probability（Hendrycks & Gimpel 2017）、verbalised confidence（Tian et al. 2023; Lin et al. 2022）开发于单模态场景，继承多模态盲区——仅基于输出分布计算，无法区分"自信使用视觉"与"自信忽略视觉"；本文用其作为集成组件。
3. **Input-level intervention vs internal probing**：内部方法（linear probing Alain & Bengio 2017、attention-map分析Neo et al. 2025）问"信息是否存在"；输入级方法（counterfactual synthesis Vo et al. 2025、occlusion Zeiler & Fergus 2014）问"输出是否使用"；本文结合两者：probe证明扰动信息存在于编码器，counterfactual证明输出不响应。
4. **VQA language prior**：Agrawal et al. 2018证明模型可不用图像回答；本文将其转化为per-sample、cross-model可迁移的测量。
5. **Diagnostic probing perspectives**：Yuksekgonul et al. 2023证明VLMs像bag-of-words；Tong et al. 2024b将感知失败归因于CLIP盲点；本文诊断定位为sample-intrinsic而非model-intrinsic，属性 residing于样本而非架构。
6. **Attention/activation probing**：Neo et al. 2025、Ben Melech Stan et al. 2024提供descriptive maps；Darcet et al. 2024发现ViT冗余patch token被repurpose为全局计算；Laurencçon et al. 2024b消融显示cross-modal projection大幅影响感知准确率；本文建议下一步做activation patching定位failure的structural locus。

## 局限性与未来方向
1. **跨模型一致性benchmark-conditioned**：POPE均值ρ=0.55，MMVP ρ=0.34，HallusionBench ρ=0.32（图表题目与局部blur管线冲突导致最弱）。
2. **VSI-错误关系可逆转**：Qwen3-VL on MMVP perception subtypes出现inverse correlation（p=0.043 uncorrected），部署阈值需per-cell校准。
3. **Linear probe是下界**：probe准确率仅反映linear-decodability，非线性信息可能更丰富。
4. **Input-level intervention的因果边界**：断开在输入-输出层面是因果性的，但无法 pinpoint language head内的具体结构 locus（如cross-modal projection层的routing failure）。
5. **Verbalised confidence对prompt敏感**：同一模型不同prompt导致AUROC差异达0.06。
6. **仅测量next-token softmax**：未涉及中间层激活、attention pattern或KV-cache content。
7. **未来方向**：(1) activation patching across cross-modal projection定位failure；(2) targeted retraining up-weight insensitive sub-population；(3) 探索VSI与reasoning-time grounding-decay（Raghu & Pandey 2026）的关系。

## 研究启发与可借鉴点
1. **条件性信号选择范式**：VSI不是universal best signal（max-prob在10/18 cells胜出），而是conditional component——这一"诊断价值地图"思路可迁移至其他模态或任务的failure-mode分析，避免过度泛化单一指标。
2. **样本级分解超越聚合指标**：将benchmark性能分解为per-sample分布（heavy left tail而非bimodality），揭示sub-population异质性；该方法论可用于诊断其他LLM/VLM系统性失败（如reasoning breakdown、context window decay）。
3. **_encoder-LLM disconnect测量框架_：线性probe + counterfactual Δargmax的组合，简洁量化"信息存在但未路由"现象；可复用于诊断其他多模态架构（audio-language、graph-language）中的类似gap。
4. **整图vs局部扰动的互补信号**：Region-VSI和whole-image-VSI Spearman仅+0.24，证明捕捉不同失败亚型；等权集成优于单一信号，提示多尺度扰动设计在诊断任务中的价值。
5. **可与本团队方向结合的创新机会**：(a) 将VSI作为selective generation的候选信号纳入团队的uncertainty quantification pipeline；(b) 在团队的多模态RAG系统中监测VSI以识别vision-ignoring hallucination；(c) 结合activation patching定位gap的structural locus，指导cross-modal projection层的finetuning策略。

## 关键术语表
**Visual Insensitivity Gap**：VLMs的vision encoder能编码视觉信息，但language head未将该信息传播至输出的现象，量化为encoder probe准确率与argmax变化率之差（>0.65）。
**Visual Sensitivity Index (VSI)**：样本级指标，用KL散度量度问题相关视觉区域模糊前后next-token分布的变化；值越小表示模型越忽略视觉证据。
**Encoder-LLM Disconnect**：vision tower的线性probe可区分扰动/清洁图像（0.72-0.79准确率），但相同样本上argmax token仅2%-11%变化，gap 0.65-0.71。
**Sample-intrinsic**：VSI的rank order在跨模型间保持相关性（grand-mean ρ=+0.40），表明视觉不敏感性是样本属性而非模型特性。
**Heavy Left Tail**：VSI分布在所有model-benchmark cells上呈现重左尾结构（40%-97%样本VSI<0.05），而非bimodal；Hartigan's dip test不拒绝单峰性。
**Dangerous Quadrant**：bottom-VSI quintile内wrong且high softmax confidence的子集，代表"自信忽略视觉"的hallucination，是VSI诊断价值最大的目标群体。
**Selective Generation / Abstention**：允许模型在低置信度时拒绝回答而非生成错误答案的任务设置，VSI作为abstention signal需与max-prob、verbalised confidence集成。
**PRR@80 (Prediction-Rejection Ratio)**：在80%覆盖率下，信号实现的风险降低与oracle实现的风险降低之比；1为oracle完美，负值表示信号优先拒绝正确预测。

## 可复现要素
- **数据集**：POPE（500 samples）、MMVP（300 samples）、HallusionBench（93-179 samples per cell）、MMStar（28-50 samples per category）；raw sample pool across models identical（per-cell n为answer-parseable subset）。
- **代码/权重**：代码及per-sample CSVs已随论文release；checkpoint identifiers和revision hashes pinned；hybrid信号的paired-bootstrap p-values和95% CIs已发布。
- **关键超参**：Gaussian blur σ=20（default，robust across {10,20,40}）；KL计算基于top-50 next-token distribution；linear probe的C=1.0、5-fold grouped CV；Grounding-DINO base variant置信度阈值0.3；SAM-ViT-base mask膨胀4像素。
- **硬件**：NVIDIA A100-80GB，BF16；总compute约32 A100-hours；单GPU作业（Qwen2.5-VL-32B用双GPU naive model parallelism）。
- **软件版本**：PyTorch + HuggingFace Transformers，scikit-learn 1.8.0（pinning因GroupKFold split assignment跨版本变化）。
- **随机种子**：seed=42固定。
