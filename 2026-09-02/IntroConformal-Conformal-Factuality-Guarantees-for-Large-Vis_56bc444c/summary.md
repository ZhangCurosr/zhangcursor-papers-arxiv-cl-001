---
title: "IntroConformal-Conformal-Factuality-Guarantees-for-Large-Vis"
source: https://arxiv.org/pdf/2609.01375v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 09:53:33"
field: "视觉语言模型可信生成"
keywords: ["Conformal Risk Control", "LVLM Factuality", "Introspective Signals", "Hallucination Detection", "Conformal Prediction", "Vision-Language Models"]
innovations: ["提出基于内省信号的无训练 CRC 框架 IntroConformal", "设计验证概率 S_prob 单次前向提取 Yes-token 归一化概率"]
benchmarks: ["MSCOCO", "CUB+Stanford Cars+Stanford Dogs", "SROIE"]
---

# 论文速读：IntroConformal-Conformal-Factuality-Guarantees-for-Large-Vis

## 一句话总结
论文提出 IntroConformal，一种无需训练的 Conformal Risk Control (CRC) 框架，通过提取 LVLM 自身内省信号（层间语义稳定性 $S_{\text{sem}}$ 与验证概率 $S_{\text{prob}}$）计算 conformity scores，在有限样本下提供分布无关的事实性风险保证；$S_{\text{prob}}$ 在多个架构上显著降低拒答率并提升 F1，全面超越依赖 CLIP 外部验证器的 CONFLVLM 基线。

## 研究问题与动机
1. **事实性失控威胁高风险部署**：LVLM 在医疗报告、自动驾驶等场景中易生成未扎根于输入图像的内容，且自信但错误的输出难以被用户识别，限制可信部署。
2. **现有统计保证方法依赖外部或不可靠信号**：生成时 token log-probability 对自信幻觉不可靠；外部验证器（如 CLIP）引入额外依赖且在资源受限场景部署困难。
3. **内部不一致性信号未被利用**：机制可解释性工作表明非事实生成伴随层间语义漂移与 hidden-state 轨迹不稳定，但现有 conformal 方法将 LVLM 视为黑盒，忽略这些信号。
4. **需要形式化且实用的风险保证**：需要在保障有限样本、分布无关的事实性上界的同时，尽量降低响应拒答率、保持输出可用性。

## 核心贡献（创新点）
1. **提出首个完全基于模型内省信号的无训练 CRC 框架**：IntroConformal 不依赖外部验证器或辅助监督，直接从同一 LVLM 提取 conformity scores。
2. **设计两种内省 conformity score**：层间语义稳定性 $S_{\text{sem}}$（衡量 mid/late layer hidden-state 对齐）与验证概率 $S_{\text{prob}}$（单次前向提取 Yes-token 归一化概率），后者比前者更具判别力。
3. **$S_{\text{prob}}$ 在区分度与实用性上全面优于外部验证器基线**：在 MSCOCO、细粒度 captioning、SROIE 三大任务与五个架构上，$S_{\text{prob}}$ 显著降低拒答率（如 LLaVA-1.5 从 57% 降至 25%）、提升 claim F1（0.504→0.581），同时严格满足 CRC 保证。
4. **建立 LTT 框架下的并发风险保证机制**：针对响应级非事实率这一非单调分数，采用 Hoeffding UCB + Bonferroni FWER 修正确保所有候选 threshold 并发有效，并给出 calibrated threshold 选择算法。

## 方法详解
- **问题设定**：将响应 $Y$ 分解为原子 claim 集合 $\mathcal{C}=\{c_1,\ldots,c_N\}$，定义非事实指示 $\mathcal{L}(c,I)\in\{0,1\}$，目标是保留子集 $\hat{\mathcal{C}}$ 使 $\mathbb{E}[\frac{1}{|\hat{\mathcal{C}}|}\sum_{c\in\hat{\mathcal{C}}}\mathcal{L}(c,I)]\le\alpha$。
- **$S_{\text{sem}}$**：对每个 claim token $t_j$，取中间 $K_{\text{mid}}=8$ 层与最后 $K_{\text{late}}=4$ 层 hidden-state 平均表示 $\bar{h}^{\text{mid}},\bar{h}^{\text{late}}$，计算余弦相似度并跨 token 平均：$S_{\text{sem}}(c_i)=\frac{1}{|c_i|}\sum_j \text{CosSim}(\bar{h}_{t_j}^{\text{mid}},\bar{h}_{t_j}^{\text{late}})$；值越高语义越稳定。
- **$S_{\text{prob}}$**：用标准 chat template 向同一模型发 prompt "Based on the image, is the following statement true? Answer with Yes or No. Statement: $c_i$"，提取首个答案位置的 Yes-token 概率并归一化：$S_{\text{prob}}(c_i)=P(\text{Yes})/(P(\text{Yes})+P(\text{No}))$；单次前向，无采样开销。
- **CRC 校准流程（LTT）**：
  - 候选阈值集合 $\Lambda$ 取自校准集所有 claim score 的唯一值。
  - 对每个 $\lambda\in\Lambda$，按嵌套选择规则 $\hat{\mathcal{C}}_\lambda=\{c:S(c)\ge\lambda\}$ 计算响应级非事实率 $r_k(\lambda)$（全拒时计为 0）。
  - 经验风险 $\hat{R}(\lambda)=\frac{1}{n}\sum_k r_k(\lambda)$，用 Hoeffding 不等式构造 UCB：$\text{UCB}_\delta(\lambda)=\hat{R}(\lambda)+\sqrt{\frac{\log(1/\delta)}{2n}}$。
  - 经 Bonferroni 修正得到并发有效水平 $\alpha'\le0.170$（$n=400$、$|\Lambda|\le20000$、$\alpha=\delta=0.10$）。
  - 最终阈值 $\hat{\lambda}=\inf\{\lambda\in\Lambda':\text{UCB}_\delta(\lambda)\le\alpha\}$，最大化保留率。

## 实验与结果
- **数据集**：MSCOCO 通用场景理解（500 图，400/100 分割）、CUB+Cars+Dogs 细粒度 captioning（516 图，400/116）、SROIE 发票文档理解（500 图，400/100）；标签由 GPT-4o/GPT-5.4 生成并经单标注员验证（ICC=0.85、κ=0.71）。
- **架构**：LLaVA-1.5-7B、Phi-3.5-Vision、Llama-3.2-11B-Vision、Qwen2.5-VL-7B、Qwen3-VL-8B。
- **基线**：CONFLVLM（CLIP 外部验证器）、$T_{\text{prob}}$（token 概率）、Woodpecker、CoVe、VCD、ICD。
- **信号质量（Table 1）**：$S_{\text{prob}}$ 在所有任务/架构上 AUROC 最高（0.699–0.819），事实-非事实差值远超 CLIP（+0.019）和 $T_{\text{prob}}$（+0.052）；$S_{\text{sem}}$ 区分度弱但稳定。
- **CRC 控制（Table 2）**：所有方法均满足保证；$S_{\text{prob}}$ 拒答率最低：LLaVA-1.5 MSCOCO 从 57%→25%、F1 0.504→0.581；Llama-3.2 从 51%→13%、F1 0.269→0.302；Qwen2.5-VL 实现 0% 拒答。
- **对比解码/验证方法（Table 3）**：IntroConformal claim filtering efficiency 97.4%、response accuracy 91%，超越 CONFLVLM（95.3%/90%）及其他所有基线。
- **标签噪声鲁棒（Table 4）**：5%–15% 对称噪声下 test risk 持续低于目标（0.054→<0.001），拒答上升但保证不失效，体现结构性保守。

## 相关工作脉络
1. **CONFLVLM (Li et al., 2025)**：首次将 CRC 用于 LVLM 事实性控制，依赖 CLIP 外部验证器；本文定位差异在于完全内部信号、无需外部模型。
2. **Chain-of-Verification / CoVe (Dhuliawala et al., 2024)**：通过多步采样生成验证答案；$S_{\text{prob}}$ 单次前向提取概率，无采样延迟。
3. **机制可解释性研究 (Azaria & Mitchell 2023; Chen et al. 2024; Zhang et al. 2025)**：发现非事实生成伴随层间语义漂移，但仅作诊断；本文首次将其形式化为 conformity score 并提供有限样本保证。
4. **Conformal Prediction for LLMs (Quach et al. 2024; Angelopoulos et al. 2021)**：应用于多项选择或文本生成不确定性量化；本文拓展至 LVLM claim-level factuality，设计适配内省信号的特化框架。
5. **解码扰动方法 (VCD, ICD, Woodpecker)**：基于 next-token 对比解码的启发式幻觉缓解；本文提供形式化风险上界，且无需额外扰动开销。

## 局限性与未来方向
1. $S_{\text{sem}}$ 需白盒访问 hidden states，$S_{\text{prob}}$ 需输出 logits，均不适用于隐藏内部状态的 API。
2. 每个 claim 需额外一次前向传播，与 CONFLVLM 的 CLIP 评分成本相当，批量推理时可优化。
3. 保证相对校准标签定义，系统性标注偏差可能选出宽松阈值；人类验证仅用单个标注员。
4. $\alpha'=0.170$ 是 FWER 修正后的 benchmark 演示点，非部署推荐；模型微调、RLHF、checkpoint 变更需重新校准。
5. $S_{\text{prob}}$ 依赖模型自身 Yes/No logits，对抗性输入可能偏置 logits 从而破坏保证，需研究鲁棒化内省信号。
6. 概率保证（$\ge1-\delta$）意味安全关键部署仍需人工监督。

## 研究启发与可借鉴点
1. **内省信号 → conformal score 的范式迁移**：将机制可解释性发现的内部不一致性（语义漂移、轨迹不稳定）直接转化为 conformity score，为其他可靠性问题（如指令遵循、代码正确性）提供可复用设计模式。
2. **单次前向验证概率设计**：$S_{\text{prob}}$ 避免 CoVe 的多步采样开销，通过提取 Yes-token 概率实现高效二分类判断，可降低延迟、适用于在线系统。
3. **非单调风险的 LTT 处理策略**：针对响应级非事实率这种移除 factual claim 可能提升比率的分段分数，采用 Hoeffding UCB + Bonferroni FWER 修正保证并发有效；该策略可直接复用于其他分数-风险映射非单调的场景。
4. **校准标签噪声的结构性鲁棒性分析**：对称噪声下保证自动收紧而非违反，为 label quality 对 conformal 方法的影响提供系统性评估框架。

## 关键术语表
**Conformal Risk Control (CRC)**：基于 Learn-Then-Test 范式的统计算法，为选择函数提供有限样本、分布无关的风险上界保证。
**Introspective Signals**：从模型自身内部状态（hidden states 或输出 logits）提取的、无需外部监督的事实性相关信号。
**Layer-wise Semantic Stability ($S_{\text{sem}}$)**：通过比较 claim token 在中间层与最后层 hidden-state 表示的余弦相似度，衡量生成过程中的语义稳定性。
**Verification Probability ($S_{\text{prob}}$)**：向同一模型发起二分类事实性 prompt，提取 Yes-token 归一化概率作为 conformity score。
**Atomic Claim**：从 LVLM 响应中分解出的单个、可独立验证的事实陈述。
**Abstention Rate**：校准后所有 claim 均被过滤、响应被整体拒绝的比例，反映保守程度。
**Learn-Then-Test (LTT)**：通过置信上界校准阈值以同时保证预测质量与风险约束的统计学习框架。
**Hoeffding Upper Confidence Bound (UCB)**：基于 Hoeffding 不等式构造的 empirical risk 上界，用于 conformal 阈值校准。

## 可复现要素
- **数据集**：MSCOCO benchmark（来自 CONFLVLM，公开）；CUB+Cars+Dogs 细粒度 captioning 组合（论文构建，未明确公开声明）；SROIE 发票数据集（公开）。
- **代码/权重**：论文未提及代码开源声明；使用开源 LVLM 架构（LLaVA-1.5、Phi-3.5-Vision、Llama-3.2-Vision、Qwen2.5/3-VL）。
- **关键超参**：$K_{\text{mid}}=8$、$K_{\text{late}}=4$；$\alpha=0.10$、$\delta=0.10$；校准集 $n=400$；候选阈值集 $\Lambda$ 取自校准集所有 claim score 唯一值。
- **标签质量**：MSCOCO 使用 GPT-4o 标注，ICC=0.85；细粒度/文档任务使用 GPT-5.4 标注，单标注员验证 κ=0.71。
