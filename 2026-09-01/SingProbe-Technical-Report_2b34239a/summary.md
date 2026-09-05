---
title: "SingProbe-Technical-Report"
source: https://arxiv.org/pdf/2608.30703v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 20:51:03"
field: "LLM 安全与对齐"
keywords: ["LLM guardrail", "runtime safety", "hallucination detection", "intrinsic monitoring", "streaming safety", "constrained decoding", "internal representation"]
innovations: ["直接复用基座模型隐藏状态实现轻量内嵌式 token 级流式安全与幻觉检测", "基于 EMA 的动态多任务权重与自适应置信度 token 加权训练策略"]
benchmarks: ["SingStreamBench", "SingStreamBench-Full", "Aegis2", "XGuard", "HarmBench", "ExpGuard", "BeaverTails", "PKU-SafeRLHF", "WildGuard", "XSTest", "FineHarm", "FactCHD", "FaithDial", "FAVA", "RAGTruth", "Shroom", "WikiBio"]
---

# 论文速读：SingProbe-Technical-Report

## 一句话总结
SingProbe 是一个仅约 2M 参数的轻量级内嵌式运行时安全护栏，直接复用 LLM 解码过程中的隐藏状态，在零额外文本编码开销的前提下实现查询意图分类、响应安全性检测与幻觉识别的 token 级流式监控；其衍生版本 SingProbe-Med 进一步展示了利用内部状态信号触发选择性干预解码的潜力。

## 研究问题与动机
1. **独立护栏推理成本高**：现有护栏多为完全独立的外部模型，需从头计算语义表示，造成重复编码、额外参数与部署复杂度；流式护栏仅在 chunk 层面检查输出，延迟较高且无法利用 LLM 中间表示。
2. **安全信号滞后**：传统方法仅在完整响应生成后评估安全性，无法在生成过程中进行细粒度干预；chunk 级别检查进一步延后了安全信号。
3. **护栏容量不足**：现有护栏远小于被监测 LLM，面对长上下文（百万 token）和复杂 agent 任务时存在容量不匹配问题。
4. **缺少对"安全前缀→不安全后文"的精准评估**：既有安全基准多为响应级标签，无法衡量流式护栏是否在 benign prefix 阶段保持静默并在不安全内容出现时迅速反应。

## 核心贡献（创新点）
1. **内嵌式 token 级流式检测**：直接复用基座模型选定层的隐藏状态经轻量预测头输出 query intent / response unsafe / hallucination 三类分数，无需独立编码 pass；与外部独立护栏的本质区别在于"内嵌共享计算"而非"外挂另跑模型"。
2. **统一多任务训练框架与自适应权重机制**：联合训练查询意图、响应安全、幻觉检测三项任务，并使用基于 EMA 的动态权重分配自动聚焦当前更难子任务；区别于静态加权的多任务训练，该方法能随训练进展自适应调整各目标比重。
3. **响应级粗粒度监督下的自适应置信度 token 加权**：针对缺少 fine-grained token 级标注的问题，设计仅在预测充分分化时激活 confidence 权重的聚合器，避免初始化阶段噪声被放大；与 Constitutional Classifier++ 固定 softmax 加权相比，本方法在前置阶段退化为均匀平均以抑制噪声。
4. **SingStreamBench 流式安全评测基准**：构建 6 层分级（safe prefix + safe continuation、complex safe prefix + unsafe response 等）的人工校验细粒度基准，显式评估护栏在 benign 前缀上是否保持静默；弥补 FineHarm 等既有基准中 token 级标签噪声大、安全/不安全位置分布偏差严重的不足。
5. **从被动检测到主动干预的范式扩展（SingProbe-Med）**：将 token 级内部状态信号用于"何时干预"（Admission Eligibility / Global Risk / Future Risk / Error Boundary 四维信号层级）与"如何干预"（sGDS 对比解码），仅在风险相关 suffix 上激活双分支解码，避免始终干预对正常生成的扰动。

## 方法详解
- **输入与输出**：冻结的 LLM M（L 层），选取子集 $\mathcal{S} \subseteq \{1,\dots,L\}$ 的隐藏状态 $\mathbf{h}_t^{(\ell)}$，拼接后送入轻量预测头 $g_\theta$，每个 token 位置输出 $[ \mathbf{s}_t^{\text{query}}, s_t^{\text{unsafe}}, s_t^{\text{hallu}} ] \in \mathbb{R}^{\mu+1+1}$；所有分数在解码过程中因果更新。
- **Query Intent**：8 类（7 类风险 + Safe），BCE 独立分类 + soft mutual-exclusion 鼓励 Safe 与最可能风险类分离。
- **多任务损失**：$\mathcal{L}_{\text{total}} = w_q \mathcal{L}_{\text{query}} + w_s \mathcal{L}_{\text{safety}} + w_h \mathcal{L}_{\text{hallu}}$，其中 $w_\tau^{(t)} = w_\tau^{\text{base}} \cdot \tilde{\mathcal{L}}_\tau^{(t)} / \bar{\mathcal{L}}^{(t)}$，$\alpha=0.9$ 的 EMA 跟踪各任务近期难度。
- **响应安全/幻觉自适应 token 加权**：$\alpha_{b,t}^{(r)} = \text{softmax}_t(\beta_r p_{b,t}^{(r)} / T)$，$\beta_r = \text{clip}(\text{std}(p^{(r)}) / \tau, 0, 1)$；预测分化不足时 $\beta_r \to 0$ 退化为均匀平均，分化充分后加大高风险 token 权重。
- **训练数据**：开源安全数据集（BeaverTails、PKU-SafeRLHF、WildGuard、XSTest、expguard 等）+ 政策驱动合成数据（jailbreak 模型生成有害 prompt、对齐模型生成对照 benign 样本）；三方 LLM judge（Qwen3-235B-A22B、GLM-5.2、Kimi-K2.6）一致认可才保留；去重、类别平衡、边界样本保留。
- **架构默认设置**：从浅/中/深三层均匀采样隐藏状态，经 query-residual MHA block（Norm(Q+O)）+ 线性分类器输出；约 2M 额外参数。
- **系统级集成**：嵌入 SGLang 与 vLLM 推理管线，实现 token 生成与护栏打分在同一流水线中完成。

## 实验与结果
- **查询意图分类**（6 基准均值 F1）：SingProbe-Ling-3.0-flash 达 **0.8674**，仅次于 YuFeng-XGuard-Reason-8B（0.8714）与 GPT-5.1（0.8683）；在 ExpGuardTest 上以 0.8825 领先。Ling-3.0-tiny → flash 提升 +0.0113。
- **响应安全分类**（8 基准均值 F1）：SingProbe-Ling-3.0-flash 达 **0.8728**，超越最强开源 baseline Qwen3Guard Gen-8B-strict（0.8604）1.24 点；在 WildGuard（0.8000）、XGuard Test（0.8588）、ExpGuardTest（0.9247）三个基准上最好。
- **流式安全检测（SingStreamBench + Full + FineHarm）**：SingProbe-Ling-3.0-tiny 平均 **R-AUC=0.9888 / T-AUC=0.9479**，相对最强 Qwen3Guard-Stream 提升 **+2.48% / +4.91%**；在长或复杂 safe prefix 场景下定位能力更强。
- **误报率（5 良性基准）**：SingProbe-Ling-3.0-flash 平均 response FPR **0.03%**，与 Qwen3Guard-Stream-8B-Loose（0.05%）相当。
- **幻觉检测（6 离线基准 Macro AUC）**：Ling-3.0-flash 上 **0.8012**，超 DRIFT（0.8000）；Ling-3.0-tiny 上 0.7765 超 DRIFT（0.7408）3.57 点。
- **在线生成检测**（留一法 pseudo-ground-truth）：Ling-3.0-flash 上 Accuracy=0.9641，F1=0.6452，超 YuFeng-XGuard-Reason-8B（F1=0.5964）**+4.88 点**。
- **在线幻觉检测（7 基准）**：Ling-3.0-tiny 平均 AUC=0.6786（超 DRIFT +0.047）；Ling-3.0-flash 平均 AUC=0.7271（超 DRIFT +0.073），7 个数据集中有 5 个第一。
- **开销**：Decode Probe 模式 ITL 额外开销 **<0.5%**（+0.18%~+0.45%），TTFT 在测量噪声内；Prefill-enabled Probe TTFT +0.86%~+2.25%；并发 1–32 下开销稳定不随并发增长。
- **Abalation**：仅用最后一层 Response AUC=0.9648，三层微增至 0.9677；幻觉检测从 L=1 的 0.7750 提升至 L=3 的 0.7963；自适应置信度 token 加权相对均匀平均 Response AUC 提升 +2.3 点、Token AUC +3.3 点。
- **SingProbe-Med（100B MoE 医疗模型）**：Full 干预 broad mitigation 25.03%、strict repair 12.58%；64-token 窗口保留 97.1% broad mitigation 与 93.2% strict repair；仅 4.69% HealthBench 样本触发干预，其他基准触发率为 0，能力无损；request 级平均延迟 ~39.16s（vs 原生 37.32s vs 常驻双分支 60.93s）。

## 相关工作脉络
1. **Llama Guard 3 / WildGuard / GraniteGuardian / ShieldGemma-9B**：独立外置护栏模型，需额外文本编码与部署；SingProbe 直接复用已有 hidden states 免额外编码。
2. **Qwen3Guard-Stream**：开源流式护栏代表，但仍是 4B/8B 独立模型；SingProbe 参数量 ~2M，且嵌入同一推理管线。
3. **YuFeng-XGuard-Reason-0.6B/8B**：推理型护栏；SingProbe 在 Flash 配置下以极小参数逼近/超越 8B 独立护栏。
4. **DRIFT / HaMI / SAPLMA**：隐藏状态-based 幻觉检测器；SingProbe 在同一框架内联合安全+幻觉检测，且在多数基准上取得更高 AUC。
5. **Constitutional Classifier++**：使用 logits 直接做 softmax 加权的响应检测；SingProbe 的自适应置信度加权在预测分化不足时退化为均匀平均，避免初始化噪声放大。
6. **FineHarm**：提供 token 级有害标注，但标注噪声大且受位置偏差影响；SingStreamBench 以 sentence-level 前缀标注避开 token 语义歧义，显式构造 safe→unsafe 过渡。

## 局限性与未来方向
1. **内部状态依赖基座模型能力**：SingProbe 性能随 Ling-3.0-tiny → flash 提升明显，说明其对基座 hidden state 质量高度依赖；对能力较弱的基座或小模型泛化性待验证。
2. **训练数据覆盖范围**：虽经合成数据补充，但风险类别仍限于 8 类查询意图与 5 类医疗风险，Emerging/低频率风险覆盖有限。
3. **在线检测 pseudo-ground-truth 局限**：在线评估使用留一法多数投票，存在detector 间共偏风险；真实用户场景 label 缺失，需更多 human-in-the-loop 评估。
4. **SingProbe-Med 的误干预代价**：64-token 窗口下 safe regression 仍达 2.53%（activated 轨迹上 9.32%），对高风险医疗场景不可接受；阈值选择依赖开发集，泛化性需验证。
5. **未来方向**：扩展到多模态（图文）联合监控、跨语言风险检测、基于内部状态的主动拒绝/重生成机制、与 agentic workflow 的深度集成。

## 研究启发与可借鉴点
1. **"内嵌复用 hidden states"作为通用范式**：将 safety/hallucination probe 做成基座模型的附加轻量子网络，避免独立编码 pass，可迁移至任何需要运行时监控的 LLM 服务场景（如 RAG 真实性、agent 工具调用安全）。
2. **自适应置信度 token 加权（adaptive $\beta$）**：应对响应级粗标签+token 级 streamed 预测的监督不匹配问题，当预测分化不足时退化为均匀平均的策略具有普适参考价值。
3. **EMA-based 动态多任务权重**：基于各任务近期 loss 的 EMA 比自动调整任务权重，避免手动调参，适用于多目标联合训练的各类下游任务。
4. **SingStreamBench 的分层构造思路**：通过 Tier 0–5 逐步增加 context 复杂度（safe prefix、disclaimer、multi-question）来测试虚假关联，为流式检测器的评测提供了可复用的方法论模板。
5. **单/双分支切换的 on-demand 干预架构**：SingProbe-Med 将"廉价持续监控"与"昂贵选择性干预"解耦，先通过轻量信号决定何时启动 sGDS 双分支，再限定干预窗口 W；该架构可推广到其他高风险领域（金融合规、法律建议）。

## 关键术语表
**SingProbe**：内嵌于 LLM 的轻量级运行时安全与幻觉检测器，直接复用基座模型隐藏状态进行 token 级流式预测。
**SingStreamBench**：专为评估流式安全护栏设计的细粒度基准，以 sentence-level 前缀标注显式测试 safe→unsafe 过渡处的检测精度与延迟。
**sGDS (Support-constrained Guided Decoding Steering)**：目标模型支持下的对比引导解码干预方法，仅在风险相关 suffix 上用 risk-pattern 分支 logit 对候选 token 重排序。
**Adaptive confidence-weighted aggregator**：仅当 token 预测充分分化时激活置信度加权、否则退化为均匀平均的训练聚合策略。
**EMA-based task weighting**：按各任务近期 loss 的指数移动平均比例动态调整多任务损失权重，使训练聚焦于当前较难子任务。
**Intervention Eligibility / Global Risk / Future Risk / Error Boundary**：SingProbe-Med 的四维干预决策信号，分别判断临床命题适用性、轨迹风险、提前 1–32 token 出错概率与错误确认边界。
**R-AUC / T-AUC**：响应级 AUC（取整条响应最大分数）与 token 级 AUC（逐 token 与细粒度标注比较），用于衡量流式检测的判别力。
**Decode Probe / Prefill-enabled Probe**：两种部署模式；前者仅在 autoregressive decoding 阶段运行 probe，后者在 prefill 阶段也运行。

## 可复现要素
- **数据集**：SingStreamBench（210 样本，手动校验）与 SingStreamBench-Full（2,428 样本）由论文构建，数据来源包括 BeaverTails、PKU-SafeRLHF、WildGuard、XSTest、expguard 等公开基准；论文未声明 SingStreamBench 单独公开，但整体代码/模型见 HuggingFace。
- **代码**：https://github.com/inclusionAI/SingProbe
- **模型权重**：https://huggingface.co/collections/inclusionAI/singprobe
- **关键超参**：tap 层数 L=3（默认）；EMA 平滑系数 α=0.9；分母 floor $10^{-9}$；响应加权温度 T、分化敏感度 τ（论文未给出具体数值）；sGDS 干预参数 k=50/100、α=2/3、窗口 W=64（bounded）。
- **服务集成**：SGLang、vLLM。
