---
title: "Controlling-Refusal-Behavior-of-LLMs-via-Stiefel-Constrained"
source: https://arxiv.org/pdf/2608.30986v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 06:34:27"
field: "LLM 安全与可控性"
keywords: ["activation steering", "LLM safety", "Riemannian optimization", "Stiefel manifold", "refusal behavior", "rotation-based intervention", "model interpretability"]
innovations: ["基于 Stiefel-SO(n) 参数化的低秩旋转干预，无需外部拒绝向量且精确保范", "在乘积流形上端到端学习旋转子空间与旋转角度，优于已有加法和旋转基线", "攻击-防御算子精确互逆，共享同一学习子空间实现双向控制"]
benchmarks: ["SALADBench", "Llama-Guard-3-8B", "Qwen3Guard-Gen-8B", "ARC-Challenge", "GSM8K", "MMLU", "WikiText-2"]
---

# 论文速读：Controlling-Refusal-Behavior-of-LLMs-via-Stiefel-Constrained

## 一句话总结
本文提出 **StiefelSteer**，一种基于黎曼优化的旋转式激活干预方法，通过学习 Stiefel 流形上的低秩旋转子空间和旋转矩阵来控制 LLM 的拒绝行为；在攻击和防御两种场景下均优于现有加法式和旋转式基线，同时更好地保留了模型能力。

## 研究问题与动机
1. **现有拒绝向量方法的不足**：现有激活干预方法依赖统计估计的"拒绝方向"（refusal vector，通过有害/无害激活差均值获得），但近年研究表明有害内容与指令拒绝可能编码在独立的方向上，单一向量不足以刻画安全机制。
2. **加法干预的缺陷**：加法式干预（如 RDO）通过放大位移来改变模型行为，过小则无效，过大则导致残差流范数膨胀、损害模型通用能力，且缺乏对余弦相似度的保证。
3. **已有旋转方法的局限性**：现有旋转式方法（Angular Steering、Spherical Steering 等）主要在二维平面内操作，表达能力受限；COAST 虽在黎曼流形上操作但仍依赖统计估计方向。
4. **需要一个自包含的旋转干预框架**：设计一种不依赖外部拒绝向量、在高维子空间内学习旋转、且计算和内存高效的参数化方案。

## 核心贡献（创新点）
1. **提出 StiefelSteer，将旋转干预参数化为 Stiefel 流形上的低秩分解**：用 $B \in \mathrm{St}(d,n)$ 和 $Q \in \mathrm{SO}(n)$ 参数化旋转算子 $M = I_d + B(Q-I_n)B^\top$，仅需 $dn + n^2$ 存储而非完整的 $d^2$。与已有方法本质区别：此前所有旋转方法固定 $n=2$ 且平面由外部估计确定，本文 $n$ 和子空间均由优化学习得到。
2. **引入 Riemannian 优化训练管道**：在 $\mathrm{St}(d,n) \times \mathrm{SO}(n)$ 积流形上对损失进行黎曼梯度下降，使用 QR-based retraction（B）和 Cayley/SVD retraction（Q）保持每次迭代可行。与已有方法本质区别：RDO 等方法采用欧氏梯度+投影，忽略流形几何。
3. **证明算子的精确范数保持、恒等初始化和攻击-防御互逆三大性质**：$\|\tilde{x}\|_2 = \|x\|_2$ 精确成立；$Q=I_n$ 时 $M=I_d$，从原始模型连续增长；防御算子为 $M^\top = M^{-1}$，是精确代数逆而非启发式符号翻转。
4. **系统实验验证在攻击和防御双场景下的优越性**：在三款不同对齐程度的模型（DeepSeek-R1-7B、Falcon3-7B-Base、Qwen2.5-1.5B-EASE）上全面超越基线，同时显著更低地保持能力指标。

## 方法详解
1. **旋转算子设计**：在每层 $\ell$ 学习参数对 $(B_\ell, Q_\ell)$，其中 $B_\ell \in \mathrm{St}(d,n)$（列正交）、$Q_\ell \in \mathrm{SO}(n)$。干预公式为 $\tilde{x} = x + B((Q-I_n)B^\top x)$，即子空间外不变、子空间内旋转。
2. **损失函数**：$\mathcal{L} = \mathrm{CE}(f_{\mathrm{rotate}}(t_{\mathrm{harmful}}), p_{\mathrm{harmful}}) + \lambda \cdot \mathrm{KL}(f_{\mathrm{rotate}}(t_{\mathrm{retain}}), f(t_{\mathrm{retain}}))$，其中 CE 诱导有害输出，KL 正则项保留通用能力；防御时仅需将目标替换为拒绝输出。
3. **Riemannian 优化**：每一步计算 Euclidean 梯度后用正交投影 (Eq.5/6) 映射到切空间，再用 retraction 返回流形。$B$ 用 QR-based retraction（$\mathcal{O}(dn^2)$），$Q$ 用 Cayley retraction（保持 det=+1，$\mathcal{O}(n^3)$）或 SVD/polar retraction。
4. **初始化**：$Q_\ell = I_n$ 保证初始干预为恒等，训练从原始模型连续演化。
5. **层选择**：通过 refusal 信号选层（中间层最有效），但旋转方向本身不依赖 refusal 向量。

## 实验与结果
- **模型**：DeepSeek-R1-7B（推理蒸馏对齐模型）、Falcon3-7B-Base（无安全对齐）、Qwen2.5-1.5B-EASE（强对齐）。
- **数据集**：训练用 Alpaca（无害）和 SALADBench（有害）；评测用 200 条有害 prompt + 50 条无害 prompt。
- **评估基准**：Llama-Guard-3-8B 和 Qwen3Guard-Gen-8B 作为 LLM-as-judge；能力评测用 ARC-Challenge、GSM8K、MMLU、WikiText-2 perplexity。
- **攻击结果最强**：DeepSeek-R1-7B 上 LG 从 0.52→**0.90**（Cayley），LG↑0.38，QG↑0.27；Falcon3-7B-Base 上 0.73→**0.92**；Qwen2.5-1.5B-EASE 上从 0.00→**0.89/0.97**（全部基线在此模型上失败）。
- **防御结果最强**：DeepSeek-R1-7B 上 LG↓至 **0.00/1.00**（Stiefel），Falcon3-7B-Base 上从 0.73→**0.20**（远优于基线最低 0.67）。
- **能力保留**：除 Falcon3-7B 攻击 ARC 下降 4 点外，其余均接近未干预水平；对比 Spherical Steering 防御时 perplexity 飙升至 4603（DeepSeek）和 1008（Qwen）。
- **超参敏感性**：$n=35$ 在 D=3584 中仅占 1%；$L=2$ 即饱和，更多层无增益。

## 相关工作脉络
1. **Arditi et al. (2024)**：提出 refusal vector 的加法干预，本文认为单向量不足以刻画安全机制，转向旋转范式。
2. **Wollschl¨ager et al. (2025) / RDO**：梯度优化 steering vector 的加法方案，本文指出加法带来范数膨胀风险，旋转天然保范。
3. **Vu & Nguyen (2025) / Angular Steering**：二维旋转（PCA 估计平面），本文推广至任意维度 $n$ 并联合学习子空间。
4. **You et al. (2026) / Spherical Steering**：球面线性插值旋转，仍在低维约束下操作；本文高维旋转同时大幅领先且能力损失更小。
5. **Nguyen et al. (2026) / COAST**：在 Riemannian 流形上操作但依赖统计方向估计，本文完全端到端学习无需外部方向。

## 局限性与未来方向
1. **防御评估不够严格**：未在自适应攻击（优化 adversarial suffix、自动 red-teaming）下测试防御鲁棒性；防御后是否 over-refuse 未测量。
2. **模型规模有限**：仅测试 1.5B–7B 三个 checkpoint，未验证更大模型或 MoE 架构。
3. **理论保障缺失**：无收敛性分析、无 $n$ 与可控制行为集的关系界定、攻击-防御互逆仅经验验证而非定理保证。
4. **子空间维度选择依赖实验**：虽发现 $n=35$ 即饱和，但未见理论指导如何选择最优 $n$。

## 研究启发与可借鉴点
1. **Stiefel-SO(n) 参数化范式可迁移**：该低秩旋转参数化不局限于安全控制，可用于任何需要保范干预的激活操控任务（如知识编辑、风格迁移）。
2. **攻击-防御共享同一算子的精确互逆结构**：$M^{-1}=M^\top$ 为双向控制提供了优雅的代数基础，后续工作可探索将此结构扩展至多方向联合控制。
3. **中间层干预而非末层**：实验表明 $L=2$（中间层）即饱和，这颠覆了"干预应在最靠近输出的层"的直觉，值得在其他激活操控任务中验证。
4. **Riemannian 优化 pipeline 的工程实现**：QR-based 和 Cayley retraction 的组合提供了高效且保持几何性质的实现范式，可直接复用。
5. **能力评估应多轴进行**：仅看安全分数无法区分"成功干预"与"模型损坏"，本文同时报告 ARC/GSM8K/MMLU/PPL 的思路值得借鉴。

## 关键术语表
**StiefelSteer**：本文提出的基于 Stiefel 流形参数化旋转的激活干预方法。
**Stiefel 流形 $\mathrm{St}(d,n)$**：满足 $B^\top B = I_n$ 的 $d \times n$ 列正交矩阵集合。
**特殊正交群 $\mathrm{SO}(n)$**：行列式为 1 的 $n \times n$ 正交旋转矩阵群。
**Refusal cone（拒绝锥）**：模型因激活落入此区域而拒绝执行有害指令的表征空间区域。
**Activation steering（激活干预/引导）**：通过在推理时修改内部激活来控制模型行为的轻量级技术。
**Cayley retraction**：将李代数元素映射回 SO(n) 的二阶重traction，保持 det=+1，复杂度 $\mathcal{O}(n^3)$。
**Riemannian optimization**：在微分流形上直接进行梯度优化的方法，保证每次迭代满足几何约束。
**Addition vs. Rotation steering**：加法干预通过偏移激活向量改变行为；旋转干预通过正交变换改变方向但保持范数。

## 可复现要素
- **数据集**：Alpaca（无害训练）、SALADBench（有害训练）、ARC-Challenge、GSM8K、MMLU、WikiText-2（能力评测）；SALADBench 和 Alpaca 为公开数据集。
- **代码**：论文声明"Code is included with the submission"（附提交物中），arXiv 提交附带。
- **权重**：未提及开源干预权重。
- **关键超参**：AdamW，lr=$1\times10^{-5}$，effective batch size=16，epochs=1，λ=1.0，n=35，bfloat16；各模型微调 L（干预层数）。
- **硬件**：单卡 NVIDIA H100 80GB，单训练 run 约 3 GPU-hours。
