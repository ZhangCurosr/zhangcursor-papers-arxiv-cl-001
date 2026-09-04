---
title: "A-Single-Suffix-to-Break-Them-All-Basin-Aware-Jailbreaks-for"
source: https://arxiv.org/pdf/2608.26506v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 06:44:16"
field: "大语言模型安全与越狱攻击"
keywords: ["model merging", "jailbreak attack", "LLM safety", "adversarial robustness", "transferable attack", "min-max optimization"]
innovations: ["提出BAJ方法，通过在合并系数空间进行min–max优化构造跨家族可迁移的对抗后缀", "揭示即使所有组件模型安全对齐，共享预训练主干的合并模型仍暴露可迁移脆弱方向", "建立家族级越狱威胁模型，攻击者无需知晓目标模型确切合并配置即可实施攻击"]
benchmarks: ["AdvBench", "Llama-2-7B/13B", "Llama-3-8B", "DeepSeek-7B", "Qwen-7B", "Gemma-7B"]
---

# 论文速读：A-Single-Suffix-to-Break-Them-All-Basin-Aware-Jailbreaks-for

## 一句话总结
本文揭示了模型合并（model merging）中一个被忽视的安全风险：即使所有组件模型均经过安全对齐，共享同一预训练主干（pretrained backbone）的合并模型仍会暴露可迁移的越狱漏洞。作者提出盆地感知越狱（Basin-Aware Jailbreak, BAJ），通过在合并空间上进行min–max优化，构造一个通用的对抗后缀，可在不同合并配置下稳定突破多个安全对齐模型的防御。

## 研究问题与动机
1. **模型合并的安全风险归因偏差**：现有研究将合并安全风险归因于"不安全组件模型的污染"，假设只要组件模型安全对齐，合并后仍然安全。
2. **预训练主干的共享脆弱性未被重视**：现代基础模型在预训练阶段已保留广泛的有害能力，安全对齐仅起行为抑制作用；参数聚合可能扰动这些抑制机制，暴露与预训练主干相关的脆弱方向。
3. **攻击者威胁模型更贴近实际**：攻击者无需知道目标模型的确切合并系数或组件检查点，只需知道共享的预训练主干即可构造可迁移的越狱提示。
4. **现有防御对家族级越狱迁移性不足**：针对单模型的防御策略难以应对合并模型家族中跨配置共享的脆弱性。

## 核心贡献（创新点）
1. **重新定义模型合并的安全风险来源**：指出风险不仅来自下游组件模型，更根源于共享预训练主干，即使所有组件模型均安全对齐，合并仍会暴露可迁移的脆弱方向。
2. **提出家族级越狱威胁模型**：攻击者只需知道预训练主干，即可针对整个合并模型家族构造通用对抗后缀，无需访问目标合并模型的确切参数。
3. **设计BAJ优化框架**：将越狱生成建模为合并空间上的min–max优化问题，通过交替更新对抗后缀与合并系数，迫使攻击针对最安全对齐的合并配置，从而捕获跨家族共享的脆弱方向。
4. **实证验证脆弱性的普遍性与持久性**：在Llama-2/3、DeepSeek、Qwen、Gemma等多个backbone及六种合并方法、多种部署配置下，BAJ均实现高转移成功率，且对现有防御具有鲁棒性。

## 方法详解
**核心思想**：合并模型位于参数空间中的同一低损失盆地（low-loss basin）内，该结构约束了模型多样性并保留了与预训练主干相关的 correlated vulnerability directions。

**目标函数**（公式7）：
$$\min_{s} \max_{\theta \in \mathcal{B}} \mathcal{L}(\boldsymbol{x} \oplus s, \theta)$$
内层最大化寻找盆地内对当前后缀最抵抗的合并模型，外层最小化优化后缀使其在该最坏情况下仍有效。

**任务向量参数化**（公式8）：
$$\theta(\boldsymbol{\alpha}) = \theta_{\mathrm{pre}} + \sum_{k=1}^{K} \alpha_k \Delta_k, \quad \Delta_k = \theta_k - \theta_{\mathrm{pre}}$$
通过合并系数 $\boldsymbol{\alpha}$ 参数化盆地内所有合并模型，将连续优化转化为系数空间搜索。

**交替优化策略**：
- **后缀更新**：固定 $\boldsymbol{\alpha}$，使用基于变异的进化优化器（类似AutoDAN）对离散文本后缀进行token级替换、插入、删除、重排。
- **系数更新**：固定后缀 $s$，对合并系数进行梯度上升：$\boldsymbol{\alpha} \leftarrow \boldsymbol{\alpha} + \eta \nabla_{\boldsymbol{\alpha}} \mathcal{L}(\boldsymbol{x} \oplus s, \theta(\boldsymbol{\alpha}))$，使优化朝向最安全对齐的合并区域。

**表征引导损失**（公式13）：
$$\mathcal{L}(\boldsymbol{x} \oplus s, \theta(\boldsymbol{\alpha})) = [h_{\theta(\boldsymbol{\alpha})}(\boldsymbol{x} \oplus s) - h_{\theta(\boldsymbol{\alpha})}(\boldsymbol{x})]^T \boldsymbol{u}$$
其中 $\boldsymbol{u}$ 是通过线性SVM在最后一层表征上训练得到的探测方向，用于区分有害/无害行为。最小化该损失推动提示表征远离无害区域。

## 实验与结果
**数据集与模型**：六个预训练主干（Llama-2-7B/13B、Llama-3-8B、DeepSeek-7B、Qwen-7B、Gemma-7B），五个下游任务（Alpaca、Dolly、CodeAlpaca、GSM8K、CodeEvol），AdvBench恶意指令集。

**评估指标**：家族级转移成功率（TSR），定义为在所有 unseen 合并模型上的平均攻击成功率。

**主要结果**：
- BAJ在全部六个backbone上 TSR 达 61.3%–89.1%，显著优于所有基线（Table 1）。最强结果：DeepSeek-7B 上 TSR=89.1%，较次优方法 SCAV（76.8%）提升12.3个百分点。
- 跨合并方法（Linear、Task Arithmetic、TIES、DELLA、DARE-TIES、Model Stock）与部署配置（水印、NF4/INT8/FP16量化）均保持稳定，TSR 58.1%–69.3%（Table 2）。
- 现有防御（Perplexity、ICD、Self-Reminder、Safety-Tuned、Intent-FT、Safety-Aware Merging）仅小幅降低 TSR（Table 3），无法有效缓解。
- 跨主干迁移实验显示明显的 backbone 依赖性（Figure 1），证实脆弱性源于共享预训练主干而非特定下游任务。
- Ablation 表明：随机采样合并系数 vs. 最大化策略，TSR 从 65.8% 骤降至 32.4%（Table 10）；变异优化 vs. 梯度优化，TSR 从 53.2% 提升至 65.8%（Table 9）。

## 相关工作脉络
1. **GCG与AutoDAN**：基于梯度或变异的单模型越狱攻击，仅针对特定checkpoint优化，未考虑合并空间的结构特性；本文BAJ通过min–max优化实现跨家族迁移。
2. **模型合并安全性研究（Hammoud et al., 2024; Zhang et al., 2024）**：聚焦"不安全组件污染"视角，认为风险来自misaligned源模型；本文揭示即使所有组件对齐，风险仍源于预训练主干。
3. **任务算术与合并方法（Ilharco et al., 2023; Yadav et al., 2023; Deep et al., 2024）**：提供参数聚合的技术基础；本文利用其系数参数化探索合并盆地。
4. **表征引导越狱（TUJA、SCAV等）**：利用模型中间层表征指导攻击；本文扩展该思路至合并空间，动态更新探测方向 $\boldsymbol{u}$。
5. **Jailbreak防御（Jain et al., 2023; Zheng et al., 2024; Bianchi et al., 2024）**：针对单模型设计的防御策略；本文证明其对家族级共享脆弱性保护有限。

## 局限性与未来方向
1. **防御策略缺失**：论文主要识别风险而非提供完整缓解方案，现有防御仅能提供有限保护。
2. **实验规模限制**：仅评估五个下游任务，且 surrogate 模型数量上限为5个；更多任务或更大规模 surrogate 集的泛化性未验证。
3. **自适应防御未深入探索**：Appendix K 提及对 BAJ 生成样本进行拒绝微调可大幅降低 TSR（降至15%–21%），但未系统研究此类自适应防御的实用性。
4. **实际部署场景简化**：假设攻击者可自由获取同主干的 surrogates，实际中可能存在访问限制。

## 研究启发与可借鉴点
1. **min–max优化范式可迁移至其他模型融合场景**：任何基于参数插值/加权的模型集成方法（如ensemble、routing）均可借鉴此"最坏情况优化"思路提升攻击/防御的鲁棒性。
2. **表征探测方向动态更新机制**：BAJ在每次系数更新后重新训练线性SVM探测方向，该设计可与 representation-guided 攻击/解释方法结合，用于分析模型安全边界。
3. **合并盆地的结构假设值得验证**：论文假设同主干合并模型位于连通低损失盆地，这一假设可拓展至 LoRA 合并、adapter 集成等场景，作为理论分析基础。
4. **家族级安全评估协议**：当前 LLM 安全评估多针对单个模型，本文提出的 TSR 指标与 threat model 可为合并模型的标准安全评测提供参考框架。
5. **cross-backbone 非迁移性结论**：脆弱性高度依赖预训练主干，提示安全对齐需针对具体 backbone 定制，通用对齐策略可能遗漏特定主干的共享脆弱方向。

## 关键术语表
**Model Merging（模型合并）**：无需额外训练，通过直接聚合多个微调模型的参数来构建单一模型的技术。
**Task Vector（任务向量）**：微调模型参数与预训练模型参数之差 $\Delta_k = \theta_k - \theta_{\mathrm{pre}}$，表示模型在参数空间中偏离基础分布的方向。
**Basin-Aware Jailbreak (BAJ)**：本文提出的攻击方法，通过在合并系数空间上进行min–max优化，构造可跨合并模型家族迁移的对抗后缀。
**Transfer Success Rate (TSR)**：家族级转移成功率，衡量攻击提示在未见过合并模型上的平均成功比例。
**Low-Loss Basin（低损失盆地）**：参数空间中围绕预训练权重的连通区域， Fine-tuned 模型通常位于此区域内，使得参数插值可行。
**Representation-Guided Objective（表征引导目标）**：利用线性探测分类器提取的表征方向，量化 adversarial prompt 与 benign prompt 在隐藏状态空间的偏移程度。
**Safety-Aware Merging**：一种合并防御方法，根据候选模型的安全风险评估调整合并系数，以减少不安全行为的传播。
**Adaptive Defense（自适应防御）**：针对特定攻击方法生成的对抗样本进行微调的防御策略，本文显示其对BAJ有一定效果。

## 可复现要素
- **数据集**：Alpaca（20k）、Dolly（15k）、CodeAlpaca（80k）、GSM8K（8k）、CodeEvol（18k）、AdvBench（恶意指令）；**公开**，可从Hugging Face获取。
- **代码/权重**：论文未明确声明开源，仅提及附录提供详细实现；模型权重为公开预训练模型（Llama-2/3、DeepSeek、Qwen、Gemma）。
- **关键超参**：fine-tuning学习率 $1 \times 10^{-5}$、batch size 16、1 epoch；BAJ外层步数 $T$、后缀优化步数 $M=500$、系数更新步数 $N=10$、学习率 $\eta=0.2$、人口大小 $P$、精英数 $k$；Perplexity防御阈值150。
