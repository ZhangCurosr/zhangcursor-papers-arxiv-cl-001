---
title: "Latent-Recurrent-Thoughts-Recurrent-Refinement-of-Proposed-L"
source: https://arxiv.org/pdf/2609.01117v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 05:21:10"
---

# 论文速读：Latent-Recurrent-Thoughts-Recurrent-Refinement-of-Proposed-L

## 一句话总结
论文提出 Latent Recurrent Thoughts (LRT)，在严格冻结 8B 参数 LLM 的前提下，通过任务专用编码器生成基础潜变量，再由轻量递归网络（TRM）进行多步迭代残差修正，最终将精炼后的连续向量作为软词注入解码器完成推理。该方法以约 11.2M 可训练参数（占骨干 0.14%）在符号与自然语言推理任务上均大幅超越既有冻结解码器基线，且推理计算开销远低于思维链（CoT）提示。

## 研究问题与动机
- **离散 CoT 的固有缺陷**：现有思维链在 token 空间自回归生成，每一步必须显式输出文本，错误会逐级累积传播；且生成高质量推理轨迹高度依赖已有的轨迹数据用于模仿训练。
- **连续潜空间推理的生成瓶颈**：将推理移至连续表示空间可规避 token 约束，但现有冻结解码器方法（SoftCoT、EBM-CoT）仅依赖单次前向传播的通用助理模型生成潜变量，或在标量能量场上做浅层梯度校准，缺乏深度计算能力。
- **通用 proposer 的非 NL 任务失效**：实验发现，在 Countdown-4 等符号任务上，通用助理生成的潜变量不仅无效，还会显著降低解码器性能（主动误导），说明必须采用任务专用的潜变量生成器。
- **递归计算模块的跨域潜力未被挖掘**：TRM 等微型递归网络已被证明是高效的迭代计算引擎，但此前仅作为单一任务的独立求解器使用，尚未与预训练 LLM 的序列建模能力结合，实现“计算深度与模型规模解耦”。

## 核心贡献（创新点）
1. **任务专用 proposer 替代通用助理**：放弃无指令遵循先验的通用 LM，设计小型双向 Transformer 编码器直接将问题映射为解码器可用的基础潜变量 $L^{(0)}$，从根本上消除非自然语言任务上的性能崩溃。
2. **递归残差精化框架**：将 TRM 重用于潜变量精化，输出有界残差 $\Delta$ 而非重生成完整 $L^\star$，通过快/慢双时态状态与多步迭代实现约束传播，使计算深度独立于骨干模型规模。
3. **截断梯度展开训练策略**：对 45 步递归仅对最后 1 个高层循环反向传播，前序 40 步采用 stop-gradient 预热状态，在几乎不增加显存与时间开销的前提下实现深度迭代。
4. **受控对比与机制验证**：在相同解码器、提示、数据与训练预算下严格对比 SoftCoT/EBM-CoT，并通过线性探针、NLL 分析与流形相似度证明推理能力由提议器与递归精化器共同承担，计算分布在两模块之间。

## 方法详解
- **整体管道**：问题 $x$ 经 proposer $g_\psi$ 生成 $K$ 个基础潜变量 $L^{(0)} \in \mathbb{R}^{K \times d}$；refiner $r_\phi$ 将其降维为外部信号 $u$，输入 TRM 结构经 $S{\cdot}H$ 次循环输出残差 $\Delta$，精化结果为 $L^\star = L^{(0)} + \Delta$；最终拼接 $[I; x; L^\star]$ 送入冻结 LLM $M$ 自回归解码 $\hat{y}$。
- **Task-Dedicated Proposer**：共享解码器 token embedding 表 $E$，经可学习投影 $\mathrm{P}_\downarrow: \mathbb{R}^d \to \mathbb{R}^{d'}$（$d'=256$）降维，拼接 $K=32$ 个可学习 query 向量，经两层双向 Block（Self-Attention + SwiGLU）交互后读出，再经 $\mathrm{P}_\uparrow$ 升维回 $\mathbb{R}^d$。参数量约 4.2M。
- **Recurrent Refiner (TRM-based)**：维护快速 scratch 状态 $z_L$ 与慢速积分状态 $z_H$（$\mathbb{R}^{K \times d'}$），初始化为学习 buffer。每高层循环执行 $T$ 次快速更新 $z_L \leftarrow f(z_L, z_H + u)$（$u$ 每步重新注入以保持问题锚定），再执行一次慢更新 $z_H \leftarrow f(z_H, z_L)$。输出经 $\mathrm{P}_\uparrow'$ 映射为 $\Delta$，加 $\lambda \|\Delta\|^2$（$\lambda=0.01$）正则防漂移。参数量约 7M。
- **训练协议**：两阶段只优化最终答案交叉熵（无轨迹监督）。Stage 1 训练 proposer；Stage 2 冻结 proposer（$L^{(0)}$ 预计算缓存），训练 refiner。推理时采用 train/inference 不对称：训练 $K=32$，测试取前 $K_{\mathrm{infer}}=4$；展开 $S=3, H=3, T=4$（共 45 次过渡，仅最后 15 次可微分）。
- **注入机制**：$L^\star$ 直接作为额外输入 embedding 拼接在 $[I; x]$ 之后，解码器权重 $\theta$ 全程冻结。

## 实验与结果
- **数据集**：符号推理 Countdown-4（算术表达式构造）、Sudoku（数独约束满足）；自然语言推理 HumanEval、MBPP（Python 函数合成）、StrategyQA（是/否多步问答）。
- **基线与设置**：统一冻结 Qwen3-8B 解码器，相同 prompt、训练数据与预算；对比 Zero-Shot CoT、Direct、SoftCoT、EBM-CoT，以及同参数量级（≈11M）的 Prefix-tuning / P-tuning v2 控制。
- **主要结果**：
  - **符号任务**：SoftCoT 与 EBM-CoT 严重崩溃（CD4 仅 5.9% / 8.4%，Sudoku 10.4% / 17.2%），
