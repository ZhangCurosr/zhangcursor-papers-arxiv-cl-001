---
title: "Context-Grounding-Gains-Are-Mediated-by-Pre-existing-Machine"
source: https://arxiv.org/pdf/2609.00925v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 22:31:48"
field: "大语言模型后训练与机制可解释性"
keywords: ["context grounding", "knowledge conflict", "GRPO", "SFT", "DPO", "mechanism reuse", "activation steering", "mechanistic interpretability"]
innovations: ["在统一起点比较 GRPO/SFT/DPO 的遵循增益差异，揭示三者行为解离", "从起始模型事前估算遵循方向并证明训练后保持高对齐（cosine ≥ 0.968）", "因果干预显示起始方向可恢复 35% DPO 增益且抑制 SFT/DPO 训练后增益"]
benchmarks: ["CounterFact", "ConFiQA", "FaithEval", "HotpotQA"]
---

# 论文速读：Context-Grounding-Gains-Are-Mediated-by-Pre-existing-Machine

## 一句话总结
该论文从同一检查点出发比较 GRPO、SFT 和 DPO 三种后训练方法在知识冲突场景下的上下文遵循（context grounding）增益，发现这些增益主要依赖于起始模型中已存在的因果机制，而非训练创造了全新结构。

## 研究问题与动机
- 语言模型在提示证据与参数记忆冲突时往往忽略提示证据，后训练可改善此行为，但**不清楚改进是否依赖新建内部机制还是强化已有机制**。
- 先前研究表明微调可保留并强化已有机制（Prakash et al., 2024；Minder et al., 2025），但缺乏在**同一检查点、同一行为指标**上系统比较不同后训练配方（recipe）的工作。
- GRPO 等 RL 方法理论上可通过奖励信号提升遵循，但实际效果取决于初始策略的上下文答案覆盖率及奖励设计，**现有工作未明确区分"RL 增益来自新机制还是已有机制放大"**。
- 激活导向（activation steering）可作为因果验证工具，但需严格的对照（匹配随机方向、KL/通用能力检查），现有工作往往缺乏此类完整性验证。

## 核心贡献（创新点）
- **首次在同一检查点、同一行为指标上系统性比较 GRPO/SFT/DPO 的遵循增益差异**，揭示三者存在显著的行为解离（dissociation），而不仅是对比单个方法的绝对效果。
- **从起始模型中估算"遵循方向"（grounding direction）并在训练前后验证其稳定性**，证明该方向在 DPO 训练中保持高对齐（cosine ≥ 0.968），支持"机制复用/放大"而非"新机制涌现"的解释。
- **因果干预实验同时证明"起始方向可 lifted 起始模型遵循能力"与"减去该方向可抑制 SFT/DPO 增益"**，建立了事前可识别方向与事后训练增益之间的因果联系。
- **独立发现的因果头集与起始模型头部高度重叠（7–8/8）**，且跨任务消融呈现强非对称性，排除了"头集重叠是任务无关的偶然现象"的替代解释。
- **提出冻结分母评估协议**解决干预条件下比率指标被系统性扭曲的问题，为后续激活干预研究提供可复用的评估规范。

## 方法详解
- **训练臂设计**：以 Qwen2.5-1.5B-Instruct 为起始检查点，训练 9 个臂（5 GRPO 变体 A/A'/B/C/D、3 SFT 变体 E0/E2/E3、1 DPO），关键臂扩展至 Qwen2.5-3B/7B、Llama-3.2-3B、Phi-3.5-mini。GRPO 使用 TRL 实现，200 步、lr 3e-6、KL β=0.02、group size 8；SFT 2 epoch；DPO β=0.5、3 epoch、4500 对偏好样本。
- **评估协议（CounterFact 两阶段）**：Pass 1 在无提示下筛选模型凭记忆答对的已知项（base 模型 n=1089，冻结后所有臂共享）；Pass 2 注入反事实上下文，记录输出是否包含上下文答案（follow-ctx）或记忆答案（follow-mem）。 headline metric = follow-ctx / (follow-ctx + follow-mem)。额外使用 ConFiQA 与 FaithEval 作为分布/格式验证。
- **遵循方向估算（DiM）**：在起始模型第 21/28 层，计算 last-position residual 在 follow-ctx 项与 follow-mem 项上的均值差，得到方向向量 **d̂**。3B+ 模型通过逐层扫描与奇偶划分验证过拟合。
- **因果头定位**：对每个训练臂独立进行 per-head knockout，找出 top-8 因果头，计算与起始模型 top-8 头的重叠率；同时做跨任务对照（recall 任务头集应完全不同）。
- **激活干预**：生成时在最后一位置添加 α·d̂（lift）或减去 α·d̂（remove），与匹配范数的随机方向对照。报告 exclusive follow-ctx rate（分母为干预前冻结的 known set）以避免干预诱导的分母偏移。
- **统计检验**：配对 McNemar 精确检验；种子级 CI 使用 t(0.975, n-1)；等价检验（TOST）以冲突-SFT 增益 ±0.044 为边界。

## 实验与结果
- **主要行为解离（CounterFact，1.5B）**：GRPO 变体增益小且不稳定（A: +0.011, p=0.62；A': +0.036, p=0.007，但 4-seed CI 含零，等价检验 p=0.0001/0.0197）；冲突 SFT（E3）中等增益 +0.062（p=1.2e-7）；DPO 最大增益 +0.087（p=1.0e-13）。GRPO 虽未提升遵循，但 HotpotQA F1 显著改善（+0.120），说明训练本身有效。
- **跨模型/尺度 DPO 近天花板增益（ConFiQA-QA）**：Qwen2.5-1.5B (+0.378)、3B (+0.599)、7B (+0.524)；Llama-3.2-3B (+0.488)；Phi-3.5-mini (+0.356)。全部 paired McNemar p ≤ 2.5e-25。
- **跨数据集复现**：FaithEval 上同样观察到 GRPO 近似零效应、SFT/DPO 正向增益的模式（Table G5）。
- **因果头集复用**：各臂独立发现的 top-8 头与起始模型重叠 7–8/8（hypergeometric p ≤ 7.1e-13 @ 1.5B；≤ 3.5e-18 @ 3B）；matched recall 任务重叠 0/8，跨任务消融显示 conflict heads 移除使冲突任务 logit 差下降 6.76、翻转率 0.843，但对 recall 任务仅降 0.35。
- **方向对齐稳定性**：1.5B 各臂方向与起始方向 cosine 0.915–0.984；3B/7B/Llama 0.942–0.987；DPO 各 seed 0.950–0.974。DPO 训练中 step 160/800 时遵循已达最终水平的 ~90%，方向余弦仍 ≥ 0.968。
- **干预因果验证**：在起始模型加 d̂（α=20）提升遵循 +0.109（p=2.7e-4），恢复 DPO 增益的 40%（通过相同 items 计算）；在 DPO 模型减 d̂（α=50）抑制增益 −0.276（p=2.5e-8）；在冲突 SFT 减 d̂ 抑制 −0.405（p=2.2e-16）。3B/7B/Llama 上抑制效应一致为负。α=20 时 KL=0.076（<0.1 阈值）、MMLU/ARC 变化 ≤1 SE，side-effect 通过；α=30 时 KL=0.170 超标，故最强有效剂量为 α=20，可恢复 35% 的 DPO 增益。
- **GRPO 覆盖率解释被拒绝**：监督 warm start（200 示例 SFT）将 training distribution 上 hit@8 从 0.380 提升至 0.453，遵循提升 +0.023（p=0.012），但在此基础上运行相同 GRPO 配方仅再增 +0.001（p=0.91），说明覆盖率不是 GRPO 无效的充分原因。

## 相关工作脉络
- **Context grounding & post-training（Li et al., 2023; Bi et al., 2025; Ming et al., 2025）**：本文不声称首次发现后训练可改善遵循，而是首次在**统一起点**下比较多种 recipe 的效果差异，并追问"增益来自何处"。
- **RLVR with context reward（Chen et al., 2026; Tamo et al., 2026; Si et al., 2026）**：这些工作使用合成数据或直接针对证据使用的 reward，属于本文 GRPO 结果的**例外情形**（off-policy 或 reward 设计不同），本文明确声明不适用于此类变体。
- **Mechanism reuse under finetuning（Prakash et al., 2024; Lee et al., 2024; Minder et al., 2025）**：先前的 reuse 证据多来自单一任务/方法；本文扩展至 GRPO/SFT/DPO 多 recipe 对比，并用**因果头定位 + 事前方向减法**提供更强因果证据。
- **Activation steering（Rimsky et al., 2024; Zou et al., 2025; Anand et al., 2026）**：本文将 steering 用作**因果验证工具**而非替代训练，并系统化报告 per-input helped/hurt fraction、KL 与通用能力指标，回应 steering 可靠性质疑（Tan et al., 2024; Pres et al., 2024）。
- **Circuit stability across training（Curtis Tigges et al., 2024; Wang et al., 2023）**：本文通过 per-arm 独立 head discovery 与 cross-task ablation 控制"头集重叠可能是任务无关的统计假阳性"的替代解释。
- **Evaluation metric artifacts（Schaeffer et al., 2023; Miller et al., 2024）**：Appendix A 揭示干预条件下比率指标分母被隐式改变的偏差模式，提出冻结分母协议，呼应可复现性关切。

## 局限性与未来方向
- **指标局限**：headline metric 为 lexical containment，与 LLM judge 的 κ 仅 0.507；SFT 与 GRPO 对比部分仅依赖 lexical 指标，绝对速率可能被低估。
- **机制审计范围**：仅覆盖 6 个臂的因果头定位，干预为单 seed；声称的是"因果头集"而非完整 circuit，存在 interpretability illusion 风险（Wang et al., 2023; Mueller et al., 2025）。
- **"pre-existing" 的定义边界**：仅指 instruction-tuned 起始 checkpoint 中存在的机制，并非 raw pretrained 模型；不能排除训练在**未测量轴向上**引入了新计算。
- **GRPO 结论的适用范围**：仅限 on-policy 全参数训练 + 本 reward 家族 + 低上下文答案覆盖率的起始策略；synthetic-coverage 方法（Si et al., 2026）与 frozen-backbone gate-module 方案（Li et al., 2026）不在覆盖范围内。
- **方向单一性**：只测量了一个 behavioral axis（follow-ctx vs. follow-mem），尽管 Appendix F.1 指出存在约 2% 方差重叠的第二轴（context-presence DiM），但未做系统双轴审计。
- **模型规模上限**：扩展至 7B，但更大规模（14B+）下的机制复用假设尚未经过验证； Mohammadi et al. (2025) 在 14B 上报告 GRPO 优于 DPO（不同行为：CoT faithfulness），提示规模/行为边界效应。

## 研究启发与可借鉴点
- **事前方向估算作为机制复用的因果探针**：在训练前从 base 模型估算目标方向（DiM 或 probe-based），训练后通过减法验证增益是否依赖该方向——此范式可迁移至其他后训练行为（如 toxicity reduction、sycophancy）的机制审计。
- **冻结分母评估协议**：在 activation steering 或任何其他干预实验中，应先冻结 known set 再计算比率指标，避免干预诱导分母偏移造成虚假效应；该做法可作为领域最佳实践推广。
- **跨任务不对称消融验证头集特异性**：仅报告头集重叠率不足以排除通用性，需补充跨任务 ablation（如本文 conflict vs. recall）证明重叠是非平凡的特异性复用，而非随机巧合。
- **等价检验（TOST）用于界定"无效"效应**：对 GRPO 等"未见显著改进"的方法，用冲突-SFT 的实际增益作为 equivalence margin 做 TOST，可给出更严谨的上界声明而非简单"p > 0.05"。
- **监督 warm start 作为 RL 探索瓶颈的诊断工具**：通过低成本 SFT 提升 rollout 覆盖率再运行 RL，可分离"探索不足"与"算法结构性缺陷"——本工作证明两者在 GRPO 上均存在，为后续设计混合训练策略提供实验依据。

## 关键术语表
- **Context grounding**：语言模型在提示提供证据（尤其与参数记忆冲突时）遵循提示内容的行为能力。
- **Knowledge conflict**：提示中的反事实/外部证据与模型参数化记忆相矛盾的场景。
- **GRPO（Group Relative Policy Optimization）**：基于组内相对优势的强化学习对齐方法，本文测试了五种 reward 变体。
- **SFT（Supervised Fine-Tuning）**：监督微调，本文区分 no-conflict 控制（E0）、KAFT 混合（E2）与高冲突比例（E3）三种配方。
- **DPO（Direct Preference Optimization）**：直接偏好优化，无需显式 reward model，本文使用 ConFiQA 偏好对训练。
- **Grounding direction（遵循方向）**：从起始模型 last-position residual 的 follow-ctx 与 follow-mem 均值差估算的低维方向，可因果操控遵循行为。
- **Mechanism reuse（机制复用）**：后训练通过放大或招募起始模型中已有的计算组件实现行为改进，而非引入全新机制。
- **Exclusive follow-ctx rate**：在冻结的干预前 known set 上计算的纯粹遵循比例，避免干预诱导分母偏移的评估指标。

## 可复现要素
- **数据集**：CounterFact（Meng et al., 2022）、ConFiQA（Bi et al., 2025）、FaithEval（Ming et al., 2025）、HotpotQA（Yang et al., 2018）——均为公开数据集。
- **代码**：论文声明将在发表时开源（per-item dumps、estimated direction vectors、one-command figure 复现、154 项自动化 verification harness），TRL 与 TRL-compatible 训练配置已在 Appendix C 详细列出。
- **关键超参**：GRPO 200 steps、lr 3e-6、KL β=0.02、group size 8、temperature 1.0；SFT 2 epochs；DPO β=0.5、lr 5e-6、3 epochs、4500 preference pairs；steering 剂量 α=20（side-effect 通过）与 α=30（40% recovery 但 KL=0.170 超标）。
- **模型**：Qwen2.5-1.5B/3B/7B-Instruct、Llama-3.2-3B-Instruct、Phi-3.5-mini-Instruct——均为公开权重。
