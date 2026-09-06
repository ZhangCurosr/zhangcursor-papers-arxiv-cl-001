---
title: "IntroConformal-Conformal-Factuality-Guarantees-for-Large-Vis"
source: https://arxiv.org/pdf/2609.01375v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 09:53:19"
field: "多模态大模型可靠性与不确定性量化"
keywords: ["Conformal Risk Control", "Vision-Language Models", "Factuality Guarantee", "Hallucination Detection", "Introspective Signals", "Hidden State Analysis", "Selective Prediction"]
innovations: ["提出训练无关的 CRC 框架 IntroConformal，从模型自身提取分层语义稳定性和验证概率两种内省 conformity score", "验证概率 S_prob 通过单次前向传播提取 Yes/No token 概率，相比 CoVe 等多步验证方法更高效且判别力更强", "系统实验证明内省信号在多种 LVLM 架构和任务上可同时满足有限样本风险保证与低拒答率"]
benchmarks: ["MSCOCO General Scene Understanding", "Fine-Grained Captioning (CUB+Cars+Dogs)", "Document Understanding (SROIE)"]
---

# 论文速读：IntroConformal-Conformal-Factuality-Guarantees-for-Large-Vis

## 一句话总结
论文提出 IntroConformal，一种无需训练的 Conformal Risk Control (CRC) 框架，通过从 LVLM 自身提取的内省信号（分层语义稳定性 S_sem 和验证概率 S_prob）提供有限样本、分布无关的事实性保证，相比依赖外部验证器或 token 概率的基线方法显著降低拒答率并提升 F1。

## 研究问题与动机
1. **LVLMs 的事实性失控问题**：大型视觉语言模型（如 LLaVA、Qwen2.5-VL）在高价值领域部署时容易产生与输入图像无关的非事实性声明，且用户难以区分正确与错误输出。
2. **现有方法的信号瓶颈**：Conformal Prediction 方法依赖生成时 token log-probability（如 CoVe、T_prob）或外部验证器（如 CLIP），前者在模型"自信但错误"时失效，后者引入额外部署依赖。
3. **内省信号的利用不足**：机械可解释性研究表明非事实性生成伴随分层语义漂移和隐藏状态不稳定，但现有 conformal 框架将 LVLM 视为黑盒，未利用这些信号。
4. **正式统计保证的需求**：仅靠启发式缓解（prompting、解码策略）无法提供形式化、有限样本的事实性风险上界，难以满足安全关键部署需求。

## 核心贡献（创新点）
1. **提出 IntroConformal 训练无关 CRC 框架**：直接利用模型自身的内省信号构建 conformity score，无需外部验证器或辅助监督，与 CONFLVLM 依赖 CLIP 验证器形成本质区别。
2. **设计分层语义稳定性 S_sem**：通过中间层与后期层隐藏状态余弦相似度捕捉跨层语义漂移，首次将该诊断信号集成到 conformal 框架中，区别于仅依赖输出 token 概率的方法。
3. **设计验证概率 S_prob**：通过向同一模型询问二元事实判断并提取 Yes-token 概率，实现单次前向传播的事实性评分，相比 CoVe 的采样验证或 VCD/ICD 的解码扰动策略更高效且判别力更强。
4. **系统实验验证保障与效用平衡**：在 MSCOCO、细粒度描述、文档理解三个任务和五种 LVLM 架构上证明 S_prob 可同时满足 CRC 风险保证、降低拒答率并提升 F1，覆盖范围远超此前单一任务验证的工作。

## 方法详解
- **问题设定**：给定图像 I、提示 X 和模型响应 Y，将 Y 分解为原子可验证声明集合 C = {c_1, ..., c_N}，目标是选择子集 Ĉ ⊆ C，使保留声明中的非事实率 E[ΣL(c,I)/|Ĉ|] ≤ α，其中 L(c,I)=1 表示声明不被图像支持。
- **分层语义稳定性 S_sem**：取 transformer 网络倒数第 8 层（M 集）与最后 4 层（T 集）的隐藏状态，对每个声明 token 计算中期与后期平均隐藏状态的余弦相似度并求均值，公式为 S_sem(c_i) = (1/|c_i|) Σ CosSim(h̄_mid, h̄_late)，高分表示语义稳定（事实性），低分表示语义漂移（非事实性）。
- **验证概率 S_prob**：对每个声明 c_i，使用同一 LVLM 以二元事实判断提示（"Based on the image, is the following statement true? Answer with Yes or No."）进行一次前向传播，提取 Yes-token 概率归一化为 S_prob(c_i) = P(Yes|I,c_i) / [P(Yes|I,c_i) + P(No|I,c_i)]，该过程无需额外解码步骤。
- **CRC 校准流程（Learn-Then-Test）**：在大小为 n=400 的校准集上枚举所有候选阈值 λ ∈ Λ（每个声明的 conformity score 唯一值），对每个 λ 计算经验风险 R̂(λ)，利用 Hoeffding 不等式构建上置信界 UCB_δ(λ) = R̂(λ) + √[log(1/δ)/(2n)]，经 Bonferroni 校正后选择满足 UCB_δ(λ) ≤ α 的最小 λ̂，保证非事实率以概率 ≥1-δ 控制在 α' ≤ 0.170 内。
- **选择规则**：在测试时对每个声明保留满足 S(c) ≥ λ̂ 的声明，全部过滤则响应 abstention 计为零损失。

## 实验与结果
- **数据集**：MSCOCO（500图，400校准/100测试）、细粒度描述（CUB+Stanford Cars+Stanford Dogs，516图，400/116）、文档理解（SROIE 发票，500图，400/100），声明用 GPT-4o-mini 分解、GPT-5.4 标注事实性。
- **模型**：LLaVA-1.5-7B、Phi-3.5-Vision、Llama-3.2-11B-Vision、Qwen2.5-VL-7B-Instruct、Qwen3-VL-8B-Instruct。
- **S_prob 区分力最优**：MSCOCO LLaVA-1.5 上 AUROC=0.819（比 CLIP 的 0.631 高 28.8%），声明差值 +0.2014；细粒度描述 LLaVA-1.5 AUROC=0.765；文档理解 LLaVA-1.5 AUROC=0.728。
- **CRC 控制与效率优势**：MSCOCO LLaVA-1.5 上 S_prob 拒答率从 CONFLVLM 的 57% 降至 25%，F1 从 0.504 提升至 0.581；Llama-3.2-Vision 拒答率仅 13%，F1 达 0.302；所有设置均满足风险保证（empirical risk < α'=0.170）。
- **对比解码/验证方法**：在声明过滤效率（TPR）和响应准确率上，IntroConformal 达 97.4%/91%，优于 Woodpecker（59.1%/41%）、CoVe（37.0%/23%）、VCD（35.5%/20%）、ICD（41.1%/26%）和 CONFLVLM（95.3%/90%）。
- **鲁棒性**：校准标签注入对称噪声 5%-15% 后 test risk 仍低于目标（从 0.054 降至 <0.001），保证以拒答率上升为代价保持有效性。

## 相关工作脉络
1. **CONFLVLM (Li et al., 2025)**：首个针对 LVLM 事实性的 conformal 框架，使用 CLIP-ViT-Large 外部验证器计算 conformity score；本文方法摒弃外部依赖，改用模型内省信号。
2. **CoVe (Dhuliawala et al., 2024)**：通过多次采样生成验证答案来减轻幻觉；本文 S_prob 单次前向传播即得概率，无额外解码开销。
3. **VCD/ICD (Leng et al., 2024; Wang et al., 2024)**：基于视觉/指令对比解码识别幻觉；本文不提供解码策略，而是提供统计保证的过滤机制。
4. **Woodpecker (Yin et al., 2024)**：通过外部 QA 工具链纠正幻觉；本文方法无需额外工具，成本更低。
5. **Conformal Prediction for LLMs (Quach et al., 2024; Cherian et al., 2024)**：将 conformal prediction 应用于 LLM，但主要基于 token log-probability；本文首次将 conformal 框架与机械可解释性的内省信号结合。
6. **Mechanistic Interpretability (Azaria & Mitchell, 2023; Chen et al., 2024; Zhang et al., 2025)**：揭示非事实性生成伴随语义漂移；本文将这些诊断信号形式化并纳入 CRC 框架提供统计保证。

## 局限性与未来方向
1. **S_sem 需要白盒访问**：依赖隐藏状态，仅适用于开放权重模型，不适用仅暴露 logit 的 API。
2. **每个声明需额外前向传播**：成本与 CONFLVLM 的 CLIP 评分相当，大规模部署可能有延迟开销。
3. **保证基于校准标签而非人类真值**：对对称噪声鲁棒，但系统性标注偏差可能导致阈值过于宽松；人类验证仅用单一标注员。
4. **α-to-α' 差距**：目标 α=0.10 时并发保证为 α'=0.170，虽随校准集增大缩小，但仍非精确控制。
5. **抗对抗攻击未验证**：S_prob 读取模型自身 verification logits，对抗性输入可能扭曲 Yes/No 概率，破坏保证。
6. **概率性保证**：安全关键部署仍需人工监督。

## 研究启发与可借鉴点
1. **内省信号与 conformal 结合**：将机械可解释性发现的内部诊断信号（语义漂移、隐藏状态稳定性）形式化为 conformity score 并集成到 CRC 框架，为其他不确定性量化任务（如可靠性检测）提供范式。
2. **S_prob 的单次前向传播设计**：通过向模型自身提问并提取指定 token 概率实现高效自检，可迁移至纯语言模型的事实性校验，避免多步解码开销。
3. **响应级拒答率定义**：将 abstention 定义为"整响应所有声明被过滤"而非单个声明，与 selective prediction 惯例一致，值得在声明级过滤任务中推广。
4. **标签噪声鲁棒性分析**：通过对称噪声注入分析表明 CRC 保证在标签错误下仍以保守滤波维持有效性，为实际场景中校准标签质量不佳提供实用洞察。
5. **多层选择 ablation 设计**：比较"早期-中期"与"中期-晚期"隐藏状态对比策略，证明后者的方向一致性更优，为未来内省信号设计提供调参依据。

## 关键术语表
**Conformal Risk Control (CRC)**：基于 Learn-Then-Test 范式的统计框架，为带连续损失的选择规则提供有限样本风险上界保证。
**Introspective Signals**：从模型自身内部激活（隐藏状态、logits）提取的信号，无需外部模型或监督。
**Layer-wise Semantic Stability (S_sem)**：衡量中间层与后期层隐藏状态余弦相似度的 conformity score，反映声明生成过程中的语义一致性。
**Verification Probability (S_prob)**：通过向模型提问二元事实判断并提取 Yes-token 归一化概率的 conformity score。
**Atomic Claim Decomposition**：将模型生成的自由文本响应分解为独立、可验证的单事实声明的过程。
**Abstention Rate**：整个响应被完全过滤（无声明保留）的比例，在 selective prediction 中计为零损失。
**Hoeffding Upper Confidence Bound (UCB)**：基于 Hoeffding 不等式的经验风险上界估计，用于 CRC 校准。
**Learn-Then-Test (LTT)**：Angelopoulos 等人提出的框架，在假设检验中同时完成模型学习与保证推导。

## 可复现要素
- **数据集**：MSCOCO（公开）、CUB（公开）、Stanford Cars（公开）、Stanford Dogs（公开）、SROIE（公开）；声明分解与标注用 GPT-4o-mini/GPT-5.4。
- **代码**：论文未明确提及代码开源。
- **权重**：需使用 LLaVA-1.5-7B、Phi-3.5-Vision、Llama-3.2-11B-Vision、Qwen2.5-VL-7B-Instruct、Qwen3-VL-8B-Instruct 的开放权重或 API。
- **关键超参**：M 集取倒数 8 层、T 集取最后 4 层；校准集大小 n=400；目标风险 α=0.10；失败概率 δ=0.10；并发保证 α'≤0.170。
