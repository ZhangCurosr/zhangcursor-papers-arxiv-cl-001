---
title: "Controlling-Refusal-Behavior-of-LLMs-via-Stiefel-Constrained"
source: https://arxiv.org/pdf/2608.30986v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 06:34:45"
field: "大语言模型安全与可控性"
keywords: ["Activation Steering", "Riemannian Optimization", "LLM Safety", "Stiefel Manifold", "Refusal Control", "Rotation-based Intervention"]
innovations: ["提出 StiefelSteer：基于 Stiefel 流形与 SO(n) 黎曼优化的参数高效旋转干预框架，无需外部拒绝向量", "严格保证精确范数保持、恒等初始化和攻击-防御代数可逆三大几何性质", "首次将旋转子空间维度 n 作为可测量量，揭示拒绝行为由低维子空间主导（n≈35，占 d=3584 不足 1%）"]
benchmarks: ["SALADBench", "Llama-Guard-3-8B", "Qwen3Guard-Gen-8B", "ARC-Challenge", "WikiText-2", "MMLU", "GSM8K"]
---

# 论文速读：Controlling-Refusal-Behavior-of-LLMs-via-Stiefel-Constrained

## 一句话总结
本文提出 StiefelSteer，一种基于黎曼优化学习的参数高效旋转变换激活干预方法，通过 Stiefel 流形和 SO(n) 上可学习子空间实现拒绝行为的精确旋转控制，在攻击与防御两种模式下均显著优于现有的加性（RDO）和旋转类（Angular/Spherical Steering）基线。

## 研究问题与动机
- **拒绝行为的几何控制缺失可靠手段**：LLM 即便经过安全对齐，仍保留有害知识；现有"拒绝锥"理论表明可通过干预激活流来改变拒绝行为，但主流加性干预依赖固定拒绝向量，缺乏几何保证。
- **加性干预存在缩放困境与能力退化风险**：加性偏移过弱无法影响行为，过强则使激活范数过大，破坏模型通用能力；且对安全/不安全输入余弦相似性的无保证性导致干预鲁棒性存疑。
- **已有旋转方法受限于二维子空间**：Angular Steering、Spherical Steering 等均仅在二维平面内旋转，表达力受限，难以捕捉 harmfulness 的高维复杂结构。
- **COAST 虽在黎曼流形上操作但仍依赖统计估计的固定方向**：该方法仍需通过均值差异等统计量估计拒绝方向，未能摆脱"向量表征不足以覆盖安全机制全貌"的限制。

## 核心贡献（创新点）
1. **提出基于 Stiefel 流形与 SO(n) 的旋转干预框架 StiefelSteer**，以 $M = I_d + B(Q-I_n)B^\top$ 为旋转算子，无需任何外部拒绝向量或探针即可学习旋转子空间与旋转角度。
2. **首次将旋转子空间维度 $n$ 作为可学习/可测量量而非设计超参**：实验表明控制拒绝仅需 $n \approx 35$（占 $d=3584$ 不到 1%），且 $L=2$ 层旋转即可饱和效果，揭示拒绝行为的低维结构。
3. **严格保证三大几何性质**：（i）精确范数保持，避免加性干预的范数爆炸问题；（ii）恒等初始化（$Q=I_n$ 时 $M=I_d$），优化从原始模型连续增长；（iii）攻击与防御互为转置逆运算 $M^{-1}=M^\top$，共享同一学习子空间。
4. **在三个异构模型（对齐基座/推理蒸馏/强对齐）上同时实现最优攻击与防御**：DeepSeek-R1-7B 攻击率从 0.52 升至 0.90，Falcon3-7B-Base 防御降至 0.20（基线无法低于 0.67），且在 Qwen2.5-1.5B-EASE（原本拒绝率 0.00）上仍可达 0.89/0.97，而竞争性方法在此模型上完全失效。

## 方法详解
- **旋转算子设计**：在选定层集合 $\mathcal{L}$ 上，每层维护一对参数 $(B_\ell, Q_\ell)$，其中 $B_\ell \in \mathrm{St}(d, n)$（Stiefel 流形，$B^\top B = I_n$）和 $Q_\ell \in \mathrm{SO}(n)$。激活干预公式为：
  $$\tilde{x}_i^\ell(t) = M(B_\ell, Q_\ell)\, x_i^\ell(t), \quad M = I_d + B(Q - I_n)B^\top$$
  等价于正交投影分解 $M = (I_d - P) + BQB^\top$（$P=BB^\top$），子空间外分量不变，子空间内由 $Q$ 旋转。
- **损失函数**：攻击模式下使用有害提示-回答对，防御模式下使用有害提示-拒绝对：
  $$\mathcal{L} = \mathrm{CE}\!\left(f_{\mathrm{rotate}}^{(B,Q)}(t_{\mathrm{harmful}}),\, p_{\mathrm{target}}\right) + \lambda \cdot \mathrm{KL}\!\left(f_{\mathrm{rotate}}^{(B,Q)}(t_{\mathrm{retain}}),\, f(t_{\mathrm{retain}})\right)$$
  第二项为 KL 正则，防止通用能力退化，$\lambda=1.0$。
- **黎曼优化流程**：
  - **切空间投影**：$B$ 的投影 $\Pi_B^{St}(G) = G - \frac{1}{2}B(B^\top G + G^\top B)$；$Q$ 的投影 $\Pi_Q^{SO}(G) = \frac{1}{2}Q(Q^\top G - G^\top Q)$。
  - **Retraction（回缩）**：$B$ 使用 QR-based retraction $\mathrm{Retr}_B(\xi) = \mathrm{qf}(B+\xi)$；$Q$ 提供两种变体：**Cayley** retraction（保定向，$\mathcal{O}(n^3)$）和 **SVD/Polar** retraction（需显式恢复定向）。
  - **更新规则**：AdamW 优化 Euclidean gradient，投影到切空间后经 retraction 回到流形，每步均保持可行性，Proposition 1 全程成立。
- **两个实现变体**：Cayley vs. Stiefel-frame（区别仅在 $Q$ 的 retraction 方式），其余完全相同。

## 实验与结果
- **数据集**：训练使用 Alpaca（无害）+ SALADBench（有害）；评估 200 条有害提示 + 50 条无害提示。
- **评估模型**：Llama-Guard-3-8B 和 Qwen3Guard-Gen-8B 双裁判；能力评估含 ARC-Challenge、ARC-Easy、GSM8K、MMLU、WikiText-2 perplexity。
- **基线对比方法**：Angular Steering、Spherical Steering、RDO（含两者变体），以及无干预基线。
- **关键结果（攻击模式，Unsafe Rate / Mean LG Score）**：
  - DeepSeek-R1-7B：Ours(Cayley) **0.90/3.73**（最佳），对比 RDO 0.84/3.55、Spherical 0.78/3.38、Angular 0.72/3.21
  - Falcon3-7B-Base：Ours(Cayley) **0.92/3.74**（最佳），对比 Spherical 0.87/3.65、RDO 0.76/3.33
  - Qwen2.5-1.5B-EASE：Ours(Cayley) **0.89/3.67**（唯一有效攻击方法），其他所有基线均为 0.00/1.00
- **关键结果（防御模式，LG↓/QG↓）**：
  - DeepSeek-R1-7B：Ours(Stiefel) **0.00/1.00**（理论下限），对比 RDO 0.02/1.01、Spherical 0.20/1.68、Angular 0.47/2.49
  - Falcon3-7B-Base（最难防御，基座模型）：Ours(Cayley) **0.20/1.61**，基线最低仅 0.67/3.02
- **消融结论**：
  - 子空间维度 $n$：$n=2$ 完全无效；$n \geq 20$ 后性能饱和，$n=35$ 达到最优（图 2）
  - 旋转层数 $L$：$L=1$ 不足；$L=2$ 即饱和，更多层无增益（图 3）
  - 旋转方向几何分析：学习到的解与 refuse direction（均值差）几乎正交，且与 RDO 轨迹完全不同（图 4），说明无需显式估计拒绝方向即可有效控制拒绝行为。
- **能力保持**：除 Falcon3-7B 攻击时 ARC 下降 4 个百分点外，其余情况下 ARC 准确率与 perplexity 均与未干预模型接近；而 Spherical Steering 在同等分数下使 perplexity 升至数千（退化输出）。

## 相关工作脉络
- **Arditi et al. [2024]（Refusal Vector 加性干预奠基作）**：本文与之本质区别在于放弃加性偏移，转而使用精确正交旋转，避免范数失控并保证逆操作代数精确。
- **RDO [Wollschlager et al. 2025]（梯度优化加性向量）**：RDO 通过梯度直接优化 steering vector $r$ 后仍做加法干预；本文通过黎曼优化同时学习子空间和旋转矩阵，且不依赖任何统计方向估计，实验表明 RDO 对层数选择 $L$ 不敏感（图 3b），而本文方法敏感度高出一个数量级。
- **Angular Steering [Vu & Nguyen 2025] / Spherical Steering [You et al. 2026]**：均在二维平面内旋转（即本文框架中 $n=2$ 的特例）；本文将其推广至任意 $n$，并证明 $n=2$ 完全无效，需 $n \geq 20$ 才能饱和。
- **COAST [Nguyen et al. 2026]**：在黎曼流形上操作但依赖固定统计拒绝方向；本文方法完全不依赖此类方向，学习到的旋转方向与 refuse direction 正交。
- **Zhao et al. [2025b]**（有害性与拒绝分离编码的实证发现）：为本文方法论提供了动机——单一方向估计不足以捕获安全机制全貌，故本文转向无方向依赖的旋转学习。
- **Marshall et al. [2024] / Piras et al. [2026] / Prakash et al. [2026]**：加性干预的变体（仿射、多流形、多组件因果分析），均属加性范式，本文从根本上采用不同干预机制。

## 局限性与未来方向
- **防御评估缺乏对抗鲁棒性测试**：未测试自适应对抗后缀、自动化 red-teaming 或 response prefilling 等标准防御场景；防御结果仅验证分布内有害提示的有效性，非鲁棒性声明。
- **未测量防御后的过度拒绝（over-refusal）**：无害提示的评估仅验证无 unsafe 响应，但无法排除模型对所有输入（包括无害）均拒绝的情况（附录指出 Qwen2.5-1.5B-EASE 原本就在 50 条无害提示中拒绝了 12 条）。
- **模型规模与架构覆盖有限**：仅测试 1.5B–7B 参数模型，未涉及 MoE 架构或更大规模模型，所需 $n$ 和 $L$ 可能不具跨尺度泛化性。
- **缺乏理论保证**：无收敛性陈述、无 $n$ 与可操控行为集合的关系界、无"转置逆算子必然导致拒绝"的理论证明（仅为实验观察）。
- **未来方向**：需开展对抗鲁棒性评估、探索更大规模的子空间泛化、建立旋转干预的理论界。

## 研究启发与可借鉴点
1. **正交旋转干预可作为加性干预的替代范式**：精确范数保持 + 恒等初始化 + 代数可逆三大性质，为后续研究提供了一套"安全改造"激活干预的理论保证框架，可直接迁移到其他需精确控制激活分布的场景（如知识编辑、风格控制）。
2. **黎曼优化在 LLM 内部表示控制中的适用性**：将 Stiefel 流形与 SO(n) 的切空间投影 + retraction 流程完整落地于 LLM 激活干预，代码组织清晰（附录含完整配置），可作为后续黎曼优化介入方法的标准参考实现。
3. **子空间维度与层数的双轴消融实验设计**：$n$ 与 $L$ 的实验揭示了拒绝行为由少数中层激活、低维子空间主导的规律，此消融策略可复用至其他激活编辑/干预方法的效果归因分析。
4. **攻击-防御共享同一算子的对称性设计**：$M^{-1}=M^\top$ 使得一次训练同时支持两种逆向控制模式，这一设计比简单符号翻转的方法论更优雅，可在多目标干预场景中推广。
5. **几何可视化为方法解释力的增强手段**：图 4 的 PCA 轨迹分析（比较 RDO 直线位移 vs. 本文曲线正交轨迹）为"为何无需 refusal direction"提供了直观证据，此类可视化可纳入后续方法的解释性评估标准流程。

## 关键术语表
- **Stiefel 流形 St(d, n)**：所有 $d \times n$ 列正交矩阵的集合（$B^\top B = I_n$），是本文参数化旋转子空间的基础流形。
- **特殊正交群 SO(n)**：$n \times n$ 行列式为 1 的正交矩阵集合，代表 $n$ 维空间中的纯旋转，本文为旋转算子 $Q$ 的参数空间。
- **拒绝锥 Refusal Cone**：激活空间中触发模型拒绝输出的区域，本文方法通过旋转将激活移出该区域而非加性偏移。
- **Riemannian Gradient**：沿黎曼流形切空间方向的梯度，本文通过将欧氏梯度正交投影到 St(d,n) 和 SO(n) 切空间后迭代更新。
- **Retraction（回缩映射）**：将切空间中的更新向量映射回流形的非线性映射（如 QR 分解、Cayley 变换），保证每步迭代后参数仍在约束流形上。
- **Activation Steering**：在推理时修改模型内部激活以实现行为控制的轻量级干预技术，本文归属旋转类 steering 分支。
- **Attack / Defence Regime**：攻击模式指诱导模型输出有害内容以测量拒绝抑制效果；防御模式指强制模型拒绝有害请求以测量安全强化效果。

## 可复现要素
- **数据集**：Alpaca（[Taori et al., 2023]）与 SALADBench（[Li et al., 2024]）；评估协议参考 Wollschlager et al. [2025] Section 4。
- **模型**：DeepSeek-R1-Distill-Qwen-7B、Falcon3-7B-Base、Qwen2.5-1.5B-Instruct-EASE（均为公开权重）。
- **代码**：论文附录声明"Code is included with the submission"，但未提供明确开源仓库链接；需查阅论文 supplement 获取。
- **关键超参**：$n=35$（子空间维度）、$L=2$（DeepSeek-R1-7B 攻击）或 $L=20$（Falcon3-7B-Base 攻击）；学习率 $1\times10^{-5}$（AdamW）、$\lambda=1.0$、micro-batch=1、effective batch=16、epoch=1、bfloat16 精度。
- **硬件**：单卡 NVIDIA H100 80GB，单次训练约 3 GPU-hours，总计 600 GPU-hours。
