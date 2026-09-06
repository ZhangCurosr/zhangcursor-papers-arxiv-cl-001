---
title: "Post-Training-Science-for-Supervised-Fine-Tuning"
source: https://arxiv.org/pdf/2609.01244v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 13:19:24"
field: "大语言模型微调与后训练"
keywords: ["Supervised Fine-Tuning", "LoRA", "Learning Rate Scaling", "Post-Training Science", "Mixture of Experts", "Hyperparameter Sweep", "Scaling Laws", "Instruction Following"]
innovations: ["Established a transferable learning rate law showing LoRA optimal LR is flat at 10^-3 across 0.6-32B models, approximately 33x higher than FullFT optimum", "Defined LoRA hyperparameter defaults (rank=64, alpha=32) through systematic sweep, showing rank plateau around 64 and alpha=32 being optimal across all tested configurations", "Demonstrated that validation NLL reliably ranks downstream quality within fixed recipes but fails to transfer across model families, while Fisher trace adds no reliable ranking at matched loss"]
benchmarks: ["IFEval", "Customer-task production judges (security, leasing, support, docs)", "Qwen3 and Llama model families across 0.6B-235B parameters"]
---

# 论文速读：Post-Training Science for Supervised Fine-Tuning

## 一句话总结
本文通过对 Qwen3 和 Llama 系列稠密及 MoE 模型在真实客户 SFT 数据集上的受控超参数扫描，建立了一套可迁移的 SFT 超参数选择法则：LoRA 最优学习率平坦地固定在 $10^{-3}$（约为 FullFT 的 33 倍），Rank 64/Alpha 32 是性价比最佳默认配置，且在约 2 个 epoch 后验证损失过拟合但通用指令遵循能力开始衰减。

## 研究问题与动机
*   **SFT 决策缺乏系统性法则**：每次微调都需要从头重新决定学习率、批量大小、LoRA rank/alpha、优化器及数据量，这些决策通常依赖预训练启发式规则或针对特定数据集的临时扫描。
*   **预训练法则在微调中的有效性未知**：现有的学习率与批量大小的缩放法则（如线性缩放规则）主要基于预训练，是否适用于 SFT，以及 LoRA 与 FullFT 的最优超参数如何随模型规模、家族和数据变化，尚未被系统测试。
*   **验证损失作为代理指标的可靠性存疑**：SFT 中通常使用验证 NLL 来选择超参数，但其是否能可靠地预测下游任务质量（由人工评估或专家构建的 judge 判定）以及跨模型家族的泛化性尚不明确。
*   **后训练资源的可扩展性（Scaling Trends）不明**：模型规模、数据量（例子数 vs token 数）以及适配器可训练参数预算对 SFT 收益的影响规律，特别是扩展到 MoE 架构时的表现，缺乏实证研究。

## 核心贡献（创新点）
*   **建立了跨模型家族的 SFT 超参数选择法则**：发现 LoRA 的最优学习率在 0.6B-32B 范围内平坦地维持在 $10^{-3}$，约为 FullFT 最优学习率（约 $3\times10^{-5}$）的 33 倍，且该法则可盲调到 30B/80B/235B 的 MoE 模型。
*   **界定了 LoRA 适配器的最佳几何配置**：证明了 Rank 在达到约 64 后收益趋于平缓，而 $\alpha=32$ 在所有测试细胞中均为最佳缩放比例，确立了 $r=64, \alpha=32$ 为默认配置，同时指出 Rank 32 是节省参数的次优选择。
*   **验证了验证损失在单配方内的代理有效性并揭示了其局限性**：确认在固定模型、数据集和适配器配置下，验证 NLL 能可靠排名下游 judged quality（Spearman $\rho_s$ 达 -0.88），但发现跨家族迁移时因 tokenizer 和拟合文本/行为的差异而失效；同时证明在匹配 loss 下，Fisher trace（损失景观平坦度）不能提供额外的可靠排名信息。
*   **刻画了 SFT 的损失可扩展性定律**：发现随着稠密模型规模的增加，默认配置下的验证损失遵循饱和幂律下降（FullFT 指数比 LoRA 更陡峭），且 MoE 模型的表现大致符合其激活参数与总参数的几何平均值。

## 方法详解
*   **受控超参数扫描框架**：在四个匿名客户 SFT 数据集（security, leasing, support, docs）上，固定种子、数据划分和 assistant-only 标签策略，每次只改变一个杠杆（学习率、批量大小、LoRA rank/alpha、模型大小、数据量、epoch 数或优化器）。实验仅纳入那些验证 NLL 严格低于预训练基线的单元格。
*   **学习率与批量大小扫描**：对 Qwen3 (0.6B-32B) 和 Llama (1B-8B) 进行网格搜索，LoRA 学习率范围 $\{3\times10^{-5}, ..., 3\times10^{-3}\}$，批量大小 $\{8, 16, 32, 64\}$；FullFT 学习率范围 $\{3\times10^{-6}, ..., 10^{-4}\}$，批量大小 $\{16, 64\}$。通过拟合最优学习率为模型隐藏尺寸的幂律来建立选择法则。
*   **LoRA 超参数扫描**：固定学习率 $10^{-3}$ 和最优批量大小，在 Qwen3-4B/8B 上扫描 Rank $\{8, 16, 32, 64, 128\}$ 和 Alpha $\{16, 32, 64\}$，评估其对验证 NLL 和可训练参数比例的影响。
*   **下游性能预测校准**：对于每个完成的基单元格，测量验证 NLL、可训练参数的 Fisher trace（通过单反向传播样本估算 Hessian 迹）以及在客户构建的生产评估（judge）上的表现，分析这些指标与下游任务评分的相关性。
*   **可扩展性趋势拟合**：在固定的超参数默认值下，将验证 NLL 对模型总参数拟合饱和幂律 $L(N) = L_\infty + A N^{-\alpha}$，并将 MoE 模型放置在稠密拟合曲线的不同位置（总参数、激活参数、几何平均值）以评估其稠密等效大小。
*   **优化器对比**：在 Qwen3-8B 上使用 AdamW 和 Muon 优化器进行全参数微调，比较其在验证 NLL、Fisher trace、生产任务 judge 评分和 IFEval 通用指令遵循能力上的表现。
*   **Epoch 扫描**：在固定数据量和超参数下，训练 1, 2, 4, 8 个 epoch，追踪验证 NLL、任务 judge 评分和 IFEval 严格提示准确率，以量化过拟合和通用能力衰减。

## 实验与结果
*   **数据集与基线**：使用四个真实客户 SFT 数据集（security, leasing, support, docs），每个数据集通过迭代 SFT（iSFT）生成以确保标签一致性。基线包括 Qwen3 和 Llama 系列的 9 个稠密模型（0.6B-32B）以及 3 个 Qwen3 MoE 模型（30B-A3B, 80B-A3B, 235B-A22B）。
*   **学习率法则**：LoRA 最优学习率平坦地为 $10^{-3}$，FullFT 最优学习率约为 $3\times10^{-5}$。跨家族迁移误差中位数为 0，90 百分位数约为 $0.5$ log10 单位。在全 16 个数据集-批量组合中，MoE 模型的 13 个单元格的离散最优学习率为 $10^{-3}$。
*   **LoRA vs FullFT**：在匹配的超参数下，FullFT 在所有 72 个比较中均优于或等于 LoRA，但 LoRA 恢复了 FullFT 相对于基线改进的中位数 98%，而仅需训练 3.1%-12.6% 的参数。
*   **LoRA 超参数**：Rank 32 仅比 Rank 64 差最多 0.003 nats，但参数更少；Rank 128 仅在三个四单元格中略优，且参数几乎翻倍。$\alpha=32$ 在所有单元格中表现最佳。
*   **验证损失的预测能力**：在固定配方内，验证 NLL 与 judged quality 呈负相关（Spearman $\rho_s$ 从 -0.38 到 -0.88）。但在匹配 loss 下，Fisher trace 未能提供可靠的额外排名信息。
*   **可扩展性趋势**：验证损失随模型规模呈饱和幂律下降，LoRA 指数为 0.31-0.40，FullFT 指数为 0.40-0.51。所有三个 MoE 模型的表现大致符合其激活参数与总参数的几何平均值。将新鲜例子数从 5k 增加到 15k 可使验证 NLL 降低约 20%（leasing）到 8%（security）。
*   **优化器对比**：Muon 在四个数据集上均达到与 AdamW 相当或略低的验证 NLL（最多低 0.012 nats），学习率约为 AdamW 的 1/3，且达到更平坦的最小值。在生产任务 judge 上两者表现持平，但在 IFEval 上 Muon 平均高出 0.09。
*   **Epoch 效应**：验证 NLL 在约两个 epoch 后达到最小值并开始过拟合上升，但任务 judge 评分在八个 epoch 内保持或略有提升。通用指令遵循能力（IFEval）随 epoch 增加而显著衰减，特别是在指令密集型任务上（如 support 任务从 0.73 降至 0.34）。

## 相关工作脉络
*   **Thinking Machines Lab (2025)**：报告 LoRA 的最优学习率比 FullFT 高一个数量级。本文通过更大规模的扫描确认了这一发现，并进一步指出 LoRA 的最优学习率在整个模型规模范围内是平坦的，且约比 FullFT 高 33 倍。
*   **Zhang et al. (2024)**：拟合了微调损失随模型大小和数据量变化的幂律。本文扩展了这一工作，将缩放趋势延伸到 MoE 架构，并发现 MoE 模型的稠密等效大小更符合其激活参数与总参数的几何平均值，而非直接使用激活参数或总参数。
*   **Liu et al. (2023)**：提出匹配预训练 loss 的模型中，更平坦的极小值具有更好的下游迁移性。本文在微调尺度上发现，在匹配 loss 下，Fisher trace 不能可靠排名同配方内的任务质量，但追踪了优化器变化时的通用能力保留情况。
*   **Kalajdzievski (2023)**：提出 rank stabilization scaling factor ($\alpha/\sqrt{r}$) 以解决大 rank 下的梯度坍缩问题。本文在标准 $\alpha/r$ 缩放下发现 rank 64 后收益饱和，未测试 rank stabilization，留待未来工作。
*   **Jordan et al. (2024) / Muon**：Muon 优化器在预训练中显示出比 AdamW 更高的计算效率。本文将其优势扩展到 SFT 全参数微调场景，发现 Muon 能以更低的学习率达到更平坦的最小值并更好地保留通用指令遵循能力。
*   **Biderman et al. (2024)**：研究发现 LoRA "learns less and forgets less"。本文通过系统扫描量化了 LoRA 对 FullFT 改进的保留比例（98%），并分析了不同 rank 和 alpha 配置下的参数效率权衡。

## 局限性与未来方向
*   **数据与评估的依赖性**：使用的四个数据集均由迭代 SFT 生成以通过客户评估，导致训练数据与下游评估 judge 非独立；评估结果反映的是对指定任务的拟合程度，而非绝对真实的任务性能。
*   **跨模型家族的迁移限制**：验证损失作为指标无法跨模型家族迁移，因为不同家族的 tokenizer 和拟合文本/行为特性存在差异，需要下游检查。
*   **优化器比较的范围限制**：仅在全参数微调中比较了 AdamW 和 Muon，尚未探索 Muon 在 LoRA 适配器上的表现，这可能是未来的重要研究方向。
*   **数据扩展研究中的混淆变量**：在数据量可扩展性研究中，例子数和 token 数在同一数据集内共变，无法完全分离两者的独立影响。
*   **模型规模的边界**：可扩展性趋势和研究主要在 0.6B-32B 稠密模型和 30B-235B MoE 模型上进行；对于更大规模的模型或更长的训练运行，受限于端到端评估的内存需求，尚未完全探索。
*   **推理模型的扩展**：当前研究未涉及 reasoning models 的微调行为，如使用无推理轨迹的数据或来自其他模型的 off-policy 轨迹进行微调时的表现。

## 研究启发与可借鉴点
*   **受控扫描的实验设计范式**：采用"每次只改变一个杠杆"的受控扫描方法，固定种子、数据划分和标签策略，确保实验结果的变化可归因于特定超参数而非数据噪声，这种方法论值得在后训练研究中推广。
*   **迭代 SFT 构建高质量微调数据集**：使用迭代 SFT（iSFT） pipeline 生成训练数据，通过 evaluator 评分和反馈循环 refinement，确保训练对在任务评估上的一致性，这为构建高质量、低噪声的 SFT 数据集提供了实用方法。
*   **验证损失作为内部选择指标的有效性边界**：明确了验证 NLL 在固定配方内可作为下游质量的可靠代理指标（Spearman $\rho_s$ 达 -0.88），但在跨配方或跨家族迁移时需要谨慎，这为研究中的指标选择提供了理论依据和边界条件。
*   **MoE 模型的稠密等效大小估算**：提出使用激活参数与总参数的几何平均值来估算 MoE 模型在稠密缩放趋势中的表现位置，这一方法为理解和预测 MoE 架构的微调行为提供了新的视角。
*   **训练轮数与通用能力衰减的权衡**：发现验证损失在约两个 epoch 后过拟合，而通用指令遵循能力（IFEval）随之衰减，这提示在实际应用中应通过监控通用能力指标而非仅依赖验证损失来决定早停点。

## 关键术语表
**Iterative Supervised Fine-Tuning (iSFT)**：一种数据生成方法，模型生成输出草稿，评估器评分并提供反馈，模型迭代改进直至通过评估，从而构建一致且高质量的对齐数据集。
**Fisher Trace**：通过单反向传播估算的 Hessian 矩阵迹，用于衡量收敛极小值的平坦程度；值越低表示损失景观越平坦。
**Muon Optimizer**：一种结构感知优化器，通过对每个 2D 权重更新执行 Newton-Schulz 迭代进行正交化，实现谱范数下的最速下降。
**Saturating Power Law**：形式为 $L(N) = L_\infty + A N^{-\alpha}$ 的拟合模型，用于描述验证损失随模型规模增加而递减并逐渐饱和的趋势。
**Dense-Equivalent Size**：MoE 模型在稠密模型缩放趋势中的等效大小，本文发现其近似等于激活参数与总参数的几何平均值。
**General Instruction Following (IFEval)**：评估模型遵循通用指令能力的外部基准，用于检测微调过程中可能发生的通用能力衰减。
**Within-Dataset-Standardised Spearman**：在固定模型和数据集单元格内对评估分数和损失进行标准化处理后计算的 Spearman 秩相关系数，用于衡量指标间的单调关系。
**Global Batch Size**：每个优化器步骤使用的总样本数，通过梯度累积实现；它权衡了损失优化与计算成本，本文发现其对最优学习率的影响很小。

## 可复现要素
*   **数据集**：四个匿名客户 SFT 数据集（security, leasing, support, docs），论文未提供公开链接；但描述了通过 iSFT pipeline 生成的机制。
*   **代码**：论文未明确提及代码开源，但提到使用 Megatron-Core 和 ms-swift 后端，建议查看相关仓库以获取实现细节。
*   **权重**：使用了 Qwen3 和 Llama 3.1/3.2 系列开源模型的权重，具体型号包括 Qwen3-0.6B, 1.7B, 4B, 8B, 14B, 32B 以及 Llama-3.2-1B, 3.2-3B, 3.1-8B。
*   **关键超参数**：
    *   LoRA 学习率：$10^{-3}$（平坦最优）
    *   FullFT 学习率：$3\times10^{-5}$
    *   LoRA rank：推荐 64，高效选择 32
    *   LoRA alpha：32
    *   训练 epoch：约 2 个（之后验证损失过拟合）
    *   数据量：5,000 例子为基础，可扩展至 15,000
    *   批量大小：无单一最优值，需权衡损失与计算成本；推荐使用 8-64 范围
