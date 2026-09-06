---
title: "How-Output-Format-Confounds-Data-Quality-and-Capability-in-I"
source: https://arxiv.org/pdf/2609.02015v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 21:00:25"
field: "大模型微调与数据质量评估"
keywords: ["instruction tuning", "output format", "gradient signature", "data quality", "capability lock-in", "spectral metrics"]
innovations: ["提出梯度签名生成模型并证明谱统计量对接口旋转不变", "发现能力锁定现象：训练接口决定能力的可迁移性", "证明接口残差携带任务特异性内容（跨三架构100%命中率）"]
benchmarks: ["ARC-Challenge", "GSM8K", "RTE", "QNLI", "CoLA", "SST-2", "HellaSwag", "Winogrande"]
---

# 论文速读：How Output Format Confounds Data Quality and Capability in Instruction Tuning

## 一句话总结
本文揭示了指令微调中"输出格式"对数据质量评估和模型能力测量的系统性混淆：谱统计量（如有效秩、核范数）对接口变化完全盲视，而能力往往被锁定在训练接口上，改变单一评估预算即可翻转微调效果的正负号。

## 研究问题与动机
- **数据质量评估的格式依赖**：现有方法使用梯度谱统计量（有效秩、核范数）评估训练数据质量，但这些指标对输出格式变化完全盲视，无法区分语义正确但格式不同的数据。
- **能力测量的接口锁定**：训练在特定格式下获得的能力，在迁移到其他格式时几乎不可见，导致跨接口评估严重低估模型真实能力。
- **评估协议的脆弱性**：单一生成预算差异可导致微调效果的符号翻转（从增益变为损失），当前报告的能力提升可能只是协议产物。
- **理论缺口**：缺乏将输出接口形式化为混淆轴的理论框架，现有工作未系统分析格式如何同时影响数据质量度量与能力测量。

## 核心贡献（创新点）
1. **提出梯度签名的生成模型与谱盲视定理**：将归一化梯度签名建模为共享分量、内容分量、接口偏移和接口-内容交互之和；证明任何仅依赖奇异值谱的泛函对接口旋转不变（Theorem 1）。
2. **证实接口残差携带任务特异性内容**：共识移除后的残差在三个架构上以100%命中率识别样本的目标任务，证明$\Gamma_k(x)$是真结构而非噪声。
3. **绘制跨架构能力锁定图谱**：展示训练于单格式的 skill 在跨格式迁移时几乎不可见（部分任务提升40+分但其他格式接近零），并审计到单一生成预算翻转GSM8K微调效果符号。
4. **预注册负面结果界定标量几何的边界**：移除接口子空间无法因果解锁迁移；预训练梯度几何无法预测哪些任务-接口对会锁定；对齐分数混淆输入表面与语义质量。

## 方法详解
- **梯度签名建模**：对每个指令单元$x$和接口$k$，取适配器梯度方向$u_k(x) = g_k(x)/\|g_k(x)\|$，分解为$u_k(x) \propto g_0 + s(x) + \delta_k + \Gamma_k(x) + \varepsilon_k$，其中$g_0$为池级公共方向，$s(x)$为内容分量，$\delta_k$为接口偏移，$\Gamma_k(x)$为接口-内容交互。
- **共识与残差计算**：跨接口共识方向$m(x) = \frac{1}{K}\sum_k u_k(x)$，归一化得$\hat{g}_{sem}(x)$；接口易感性$S(x) = 1 - \|m(x)\|$；中心化残差$r_k(x) = u_k(x) - m(x)$。
- **方向性度量**：语义对齐$A_{sem}(x,t) = \langle \hat{g}_{sem}(x), \hat{g}_{sem}(t) \rangle$；匹配对齐$A_{match}(x,t) = \frac{1}{K}\sum_k \langle u_k(x), u_k(t) \rangle$；残差对齐$RA(x,t) = \frac{\langle R(x), R(t) \rangle_F}{\|R(x)\|_F \|R(t)\|_F}$。
- **能力锁定指数**：$Lock_T(j) = 1 - \frac{\text{mean}_{k\neq j} \Delta_T(j\to k)}{\max(\Delta_T(j\to j), \epsilon)}$，接近1表示能力被锁定在接口$j$。
- **谱盲视证明**：接口变化近似为$G \mapsto c_k R_k G Q_k^\top$，任何仅依赖奇异值谱的泛函满足$\varphi(c_k R_k G Q_k^\top) = \varphi(c_k G)$，只保留尺度$c_k$而丢弃方向信息。

## 实验与结果
- **数据集与模型**：12个分类与常识推理任务（sst2、yelp、rte、qnli、boolq、copa、piqa、hellaswag、winogrande、arc_challenge、commonsense_qa、ag_news），三个模型家族Qwen3.5-4B、Qwen3.5-9B、Mistral-7B，每个任务四种训练接口（plain answer、raw span、JSON field、task tag）。
- **数据质量度量**：残差对齐在4B上AUC=0.789，匹配对齐0.784；有效秩0.415、核范数0.468（均在盲视带[0.35, 0.65]内）。方向性度量在4B和Mistral上清晰脱出盲视带，在9B上收窄但仍有效。
- **残差特异性**：三个架构上clean unit的own-target argmax命中率均为1.00（chance≈0.083），$p<10^{-4}$，证实$\Gamma_k(x)$携带任务特异性内容。
- **能力锁定**：跨四张转移地图，对角增益≥10分的细胞共70个，其中完全锁定（off-diagonal transfer ratio < 0.2）共26个；qnli/raw对在所有四张地图中均锁定。
- **预算翻转实验**：GSM8K上，base模型在192-token预算下得分19.5（4B）/1.0（9B），768-token下达78.0/24.0；微调后 tuned 模型在两预算下均平坦（19.5→19.5 / 14.0→14.5），导致报告效果从+13分翻转为-9.5分（9B）。

## 相关工作脉络
- **LESS等梯度相似性数据选择**：Xia et al. (2024) 等基于梯度低秩相似度选择数据；本文将其分解为内容/接口分量，指出单接口谱标量无效但方向性选择仍有效（接近oracle）。
- **谱统一质量度量**：Li et al. (2026) 等用有效秩/核范数统一数据质量信号；本文证明这类标量对接口旋转不变且无法区分语义污染。
- **输出格式敏感性**：Sclar et al. (2024)、Do et al. (2025) 在推理端发现格式影响few-shot准确率；本文首次将分析移至训练数据与梯度层面。
- **数据估值与可识别性**：Data Shapley、datamodels等讨论数据价值单标量的局限性；本文形式化接口为混淆轴，补充了"内容仅在多视角下可识别"的理论基础。
- **概念擦除与格式专门化适应**：Ravfogel et al. (2020, 2022)、Wang et al. (2024) 尝试移除特定子空间；本文预注册实验证明移除接口子空间无法解锁迁移。

## 局限性与未来方向
- 实验局限于分类与短形式推理任务、低秩适配器，未扩展到长文生成与全参数微调。
- 构造的污染家族虽含外部锚点（fluent paraphrase、natural dirty pool），但无法覆盖真实数据的所有错误模式。
- 计算限制使研究止于十亿参数以下模型，更大规模可能改变效应幅度但不改变方向。
- 未定位锁定效应的实际存储位置（不在LoRA-B池级接口子空间），未来需探索跨层、跨模块的锁定机制。

## 研究启发与可借鉴点
1. **多接口梯度签名提取范式**：对同一内容渲染多种输出格式、分别提取梯度方向，再计算共识/残差，可作为评估数据选择器稳健性的标准协议。
2. **残差对齐作为质量信号**：$RA(x,t)$比单接口谱统计量更能区分干净/污染数据，且携带任务特异性；可迁移至数据过滤、混合策略设计。
3. **跨接口能力评估协议**：报告微调效果时应同时给出seen-interface gain与held-out-interface gain，避免单一格式的虚假提升。
4. **评估预算审计**：报告benchmark结果时需验证base模型与tuned模型是否在同一预算下均能完整输出，否则效果符号可能翻转。

## 关键术语表
- **梯度签名（gradient signature）**：归一化的适配器梯度方向，编码数据会推动模型参数向哪个方向更新。
- **接口（interface）**：同一语义内容的外在输出表面格式（如纯文本答案、JSON字段、标签化跨度等）。
- **共识方向（consensus direction）**：跨所有接口的梯度方向均值，近似提取与接口无关的内容分量。
- **残差对齐（residual alignment, RA）**：接口变化部分的内积相似度，衡量两个样本在接口交互结构上的匹配程度。
- **能力锁定（capability lock-in）**：在某接口训练获得的技能几乎无法迁移到其他接口的现象。
- **谱盲视（spectral blindness）**：基于奇异值谱的统计量对接口旋转不变，无法区分格式不同但语义相同的样本。
- **语义选择性（semantic selectivity）**：度量区分干净数据与语义污染数据的能力，用AUC衡量。
- **生成预算翻转（generation budget flip）**：改变输出token上限导致微调效果符号从正变负的评估脆弱性。

## 可复现要素
- 数据集：12个公开基准任务（GLUE子集、ARC、HellaSwag、Winogrande、CommonsenseQA等），代码与预处理脚本论文未明确提及开源状态。
- 模型：Qwen3.5-4B、Qwen3.5-9B、Mistral-7B-v0.3，权重公开可下载。
- 关键超参：LoRA rank=8，每选择器24 units×64 examples=1536 training examples，随机种子3个，接口K=4，signature从o_proj和down_proj的lora_B因子提取。
- 代码/权重开源状态：论文未明确声明代码仓库或数据共享链接。
