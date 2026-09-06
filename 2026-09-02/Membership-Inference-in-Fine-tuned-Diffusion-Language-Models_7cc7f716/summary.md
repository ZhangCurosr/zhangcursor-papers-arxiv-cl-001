---
title: "Membership-Inference-in-Fine-tuned-Diffusion-Language-Models"
source: https://arxiv.org/pdf/2609.00873v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 05:22:12"
field: "生成模型隐私安全"
keywords: ["Membership Inference", "Diffusion Language Models", "Privacy Attacks", "Token Memorization", "PII Extraction"]
innovations: ["首次理论化扩散训练中令牌级记忆不对称性（逆mask率缩放）", "提出Q-SKEW分位数加权偏度指标，统一评估DFT与IFT场景"]
benchmarks: ["ArXiv", "WikiText", "XSUM", "MedQA", "Alpaca", "Tulu-3"]
---

# 论文速读：Membership-Inference-in-Fine-tuned-Diffusion-Language-Models

## 一句话总结
本文首次发现并理论化了扩散语言模型（DLMs）中的"令牌级记忆不对称性"——单步记忆增益与mask比率成反比，并据此提出Q-SKEW指标，在领域微调和指令微调场景中均显著优于现有成员推断基线方法。

## 研究问题与动机
- **核心问题**：扩散语言模型作为新兴生成范式，其微调阶段的隐私风险（特别是成员推断攻击）尚未被系统研究。
- **现有方法不足**：
  - 针对自回归语言模型的MI方法（如Min-K%、Min-K++）依赖左到右分解假设，不适用于DLMs的any-order去噪训练目标；
  - DLMs中每个token以不同概率被mask，破坏了基于下尾token score的方法有效性；
  - 视觉扩散模型的MI方法在离散状态空间下性能显著下降。
- **评估场景局限**：现有DLMs MI研究仅关注领域微调（DFT），未覆盖实际更常见的指令微调（IFT）场景。

## 核心贡献（创新点）
1. **首次理论化令牌级记忆不对称性**：证明扩散训练中单步记忆增益与mask比率严格反比，且成员样本的累积记忆增益分布呈严格右偏（Theorem 3.2）。
2. **提出Q-SKEW指标**：基于逆mask率现象，设计分位数加权偏度指标，从数据分布视角而非单样本记忆层面区分成员与非成员。
3. **统一评估DFT与IFT场景**：覆盖真实世界的两种微调范式，实验证明Q-SKEW在各类设置下平均AUC提升超过10%。
4. **揭示PII提取新路径**：证明偏度指标可补充CE损失信号，提升个人 identifiable information 重构攻击效果（ASR提升4-10个百分点）。

## 方法详解
**理论基础**：
- 定义令牌记忆增益：$\mathcal{G}_{w_i} = \mathcal{L}(w_i; \theta_0) - \mathcal{L}(w_i; \theta_K) = \sum_{k=1}^K \delta_{w_i,k}$
- **假设3.1（逆Mask率缩放）**：当token被选中时，记忆增益幅度是mask率的严格单调递减函数：$\delta_{w_i,k} = m_{w_i,k} \cdot \phi(w_i, \beta_k)$，其中$\phi$单调递减。
- **定理3.2**：成员样本的$\mathcal{G}_{w_i}$分布严格右偏：$Skew(\mathcal{G}_{w_i}) > 0$；非成员样本偏度≈0。

**Q-SKEW计算方法**：
1. **令牌级记忆分数**：对mask子集$\mathcal{M}$，计算$\Delta_{i,\mathcal{M}} = [-\log p_{\theta_{ref}}(w_i|\mathbf{x}_{\mathcal{M}^c})] - [-\log p_{\theta_{tar}}(w_i|\mathbf{x}_{\mathcal{M}^c})]$
2. **循环采样**：执行R轮全覆盖采样，每轮随机排列索引并划分不交batch，确保均匀覆盖：$S_\mathbf{x} = \bigcup_{r=1}^R \bigcup_{j=1}^m \{\Delta_{i,\mathcal{M}_{r,j}} | i \in \mathcal{M}_{r,j}\}$
3. **分位数加权偏度**：$A'(x) = \frac{\frac{1}{|S_x|}\sum_{s \in S_x} W(s) \cdot (s - \tilde{S}_x)}{\sigma(S_x) \cdot \bar{W}}$，其中权重$W(s)$锚定在15th和85th百分位。

## 实验与结果
**数据集与模型**：
- 模型：LLaDA-8B-Base/Instruct、Dream-Base-7B/Instruct
- DFT数据集：ArXiv、WikiText、XSUM
- IFT数据集：MedQA、Alpaca、Tulu-3

**主要结果**（LLaDA系列，Table 1-2）：
- **DFT场景**：Q-SKEW在ArXiv上AUC达83.4%（较SAMA提升13.2%），TPR@1%FPR达50.6%（较SAMA提升33个百分点）
- **IFT场景**：Alpaca上AUC达81.4%（较SAMA提升12.3%），TPR@1%FPR达41.6%（较SAMA提升30.4个百分点）
- **跨Epoch稳定性**：不同微调轮数下性能持续优于基线（Figure 2）
- **Dream模型**：同样取得最佳性能，但整体脆弱性低于LLaDA（Table 3）

**PII提取实验**（Table 6）：
- 结合CE与偏度的Combine-0.1方法：手机号ASR达34%（较纯CE提升4%），邮箱ASR达24%（较纯CE提升4%）

## 相关工作脉络
1. **AR-LM MI方法**：Loss、Min-K%、Min-K++、Calibration——基于左到右分解，无法直接应用于DLMs的any-order去噪。
2. **视觉扩散MI方法**：SecMI、CLiD、PIA——SecMI可适配但依赖连续状态空间假设；PIA和CLiD因公式推导依赖连续域或条件生成设定而无法适配。
3. **SAMA（Chen et al., 2026）**：同期工作，仅评估DFT场景，缺乏DLMs与AR模型的理论学习，本文在DFT和IFT上均超越SAMA。
4. **MIMIR数据集争议**：论文指出MIMIR存在先验偏差（未微调模型AUC达0.64），不适合评估微调阶段MI。

## 局限性与未来方向
- **模型覆盖有限**：仅评估当前主流开源DLMs（LLaDA、Dream系列），未来可能出现的新架构泛化性未知。
- **Reference模型假设**：虽验证了对齐误差下的鲁棒性，但攻击者需获取未微调参考模型（论文附录E论证了合理性）。
- **PII提取深度不足**：偏度 alone 效果弱于CE，仅作为补充信号，未深入探索其在结构化数据提取中的作用机制。

## 研究启发与可借鉴点
1. **逆mask率现象的理论价值**：揭示扩散训练动态的核心特性，为后续研究DLMs过拟合机制提供分析工具。
2. **循环采样策略**：解决Monte Carlo采样的随机方差问题，可迁移至其他需要稳定统计估计的隐私攻击场景。
3. **偏度指标的扩展潜力**：从成员推断延伸至PII提取，证明分布形状特征比绝对值更具区分力。
4. **DFT与IFT统一评估框架**：填补指令微调隐私评估空白，为后续工作提供标准化Benchmark。

## 关键术语表
- **Diffusion Language Model (DLM)**：通过迭代去噪过程生成文本的语言模型，支持并行生成和任意顺序掩码训练。
- **Token-level Memorization Asymmetry**：成员样本的令牌级记忆增益分布呈右偏，非成员样本接近对称。
- **Inverse Mask-Ratio Scaling**：扩散训练中，低mask比率导致更强的单步记忆增益。
- **Q-SKEW**：分位数加权偏度指标，基于记忆增益分布的形状差异进行成员推断。
- **Cyclic Sampling**：全覆盖循环采样策略，确保每个token在不同mask配置下被均匀评估。
- **Domain Fine-tuning (DFT)**：与预训练目标一致的微调，使用固定block size（如64/128/256 tokens）。
- **Instruction Fine-tuning (IFT)**：指令跟随微调，数据长度可变，更贴近实际应用。
- **PII Extraction**：从模型中提取个人可识别信息（如邮箱、手机号）的攻击。

## 可复现要素
- **数据集**：ArXiv、WikiText、XSUM、MedQA、Alpaca、Tulu-3（均为公开数据集）
- **代码/权重**：LLaDA和Dream模型权重从Hugging Face获取（论文未提及自有代码开源）
- **关键超参**：mask ratio ρ=0.35，denoising steps=16，epochs=6，batch size=16，bandwidth h=0.1，cyclic rounds R未明确（从Algorithm 1推断）
