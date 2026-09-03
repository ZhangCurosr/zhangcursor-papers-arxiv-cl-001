---
title: "ST-sup-2-sup-U-Stateful-Test-Time-Unlearning-via-Restricted"
source: https://arxiv.org/pdf/2608.23034v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 00:58:49"
field: "大语言模型机器遗忘与安全对齐"
keywords: ["Machine Unlearning", "Test-Time Intervention", "Activation Editing", "Stateful Control", "Risk Boundary", "Restricted Knowledge Re-entry", "Large Language Models"]
innovations: ["提出轨迹级状态化测试时机器遗忘框架ST²U，通过低维可逆坐标与历史状态传递实现受限知识边界的持续控制", "构建RKG风险几何与密度比风险函数，首次系统量化并缓解受限知识重新进入问题"]
benchmarks: ["RWKU", "WMDP", "MUSE-Books", "MMLU"]
---

# 论文速读：ST²U: Stateful Test-Time Unlearning via Restricted Knowledge Boundary Control

## 一句话总结
本文提出 ST²U（Stateful Test-Time Unlearning via Restricted Knowledge Boundary Control），一种面向冻结大语言模型的**轨迹级状态化测试时机器遗忘框架**，通过将受限制知识的边界控制转化为沿生成轨迹的低维可逆坐标下的风险约束优化，有效缓解了现有测试时方法中"局部修正有效但后续解码重新进入受限知识区域"的**受限知识重新进入（restricted knowledge re-entry）**问题。

## 研究问题与动机
- **核心问题**：现有测试时机器遗忘方法（如 activation editing、decoding-based control）仅在单个解码位置进行孤立点态修正，未建模自回归生成过程中隐藏状态如何从 prompt、cache 和已生成 prefix 中不断重建，导致**局部有效的干预无法保证整个生成轨迹的持续性安全**。
- **观察到的失败模式**：如图1所示，模型可能在初始干预后生成安全的高层解释，但后续解码仍会返回到支持受限知识生成的区域并输出有害细节，作者称之为"受限知识重新进入（restricted knowledge re-entry）"。
- **现有方法的不足**：
  - 参数更新类方法（如 RMU、AS、ALTER）计算成本高且可能损害保留能力；
  - 测试时方法（如 ICUL、SCANS、INNSTEER）缺乏跨 token 的状态耦合机制，仅对当前 query 或激活做独立修正，未显式建模风险在 evolving hidden-state trajectory 上的时间演化。
- **动机**：将测试时遗忘建模为**在线隐藏状态轨迹的状态控制问题**，在保持非目标能力的前提下，实现对生成全程的风险约束。

## 核心贡献（创新点）
1. **构建受限知识风险几何（Restricted Knowledge Geometry, RKG）**：将目标相关激活映射到低维可逆坐标，估计连续风险边界（基于密度比），同时保留正交非目标成分不变——与已有工作通过静态向量或随机目标驱动激活的区别在于，RKG 学习的是任务相关的低维流形结构而非单一方向偏移。
2. **提出状态化边界控制（Stateful Boundary Control, SBC）**：结合法向梯度下降、切向历史状态投影与上下文锚定，实现轨迹级的风险边界穿越控制——与现有 position-wise 编辑的本质区别在于引入了固定维度的历史修正状态 $\mathbf{m}_t$ 来传递跨 token 的时间一致性信息。
3. **定义并量化受限知识重新进入（R-Ent.）指标**：首次系统刻画测试时干预后生成轨迹中风险重新上升的现象——区别于现有工作仅评估单点遗忘准确率，本文强调"遗忘持久性"这一被忽视的评测维度。
4. **在三个基准（RWKU、WMDP、MUSE-Books）和三种模型架构（Llama、Qwen、SpikingBrain）上验证**：ST²U 实现最优的遗忘-保留权衡，同时将 R-Ent. 降至 13.76%–19.84%，显著优于基线方法的 46.50%–59.10%。

## 方法详解
ST²U 由两大组件构成：

### 1. 受限知识几何建模（RKG）
- **配对上下文轨迹构造**：对每个受限序列 $x_i^F \in \mathcal{F}$，构建保留共享高层上下文但省略目标细节的配对参考 $x_i^S$，提供对比监督以隔离目标相关激活变异。
- **低维可逆坐标变换**：
  - 对隐藏状态 $\mathbf{h}$ 进行标准化：$\widetilde{\mathbf{h}} = (\mathbf{h} - \pmb{\mu}) \oslash \pmb{\sigma}$
  - 学习列正交基 $\mathbf{U} \in \mathbb{R}^{d \times k}$（$k=32$），投影到目标相关坐标 $\mathbf{p} = \mathbf{U}^\top \widetilde{\mathbf{h}}$ 及其正交残差 $\mathbf{r} = (\mathbf{I} - \mathbf{U}\mathbf{U}^\top)\widetilde{\mathbf{h}}$
  - 通过可逆映射 $\psi: \mathbb{R}^k \to \mathbb{R}^k$（由 affine-coupling blocks 实现）变换坐标：$\mathbf{z} = \psi(\mathbf{p})$
  - 重构：$\widetilde{\mathbf{h}} = \mathbf{r} + \mathbf{U}\psi^{-1}(\mathbf{z})$，仅编辑 $\mathbf{z}$ 而保持 $\mathbf{r}$ 不变
- **风险几何与上下文锚点**：
  - 在 $\mathbf{z}$ 空间使用 KDE 建模受限/安全分布密度 $d_F, d_S$
  - 风险函数：$s(\mathbf{z}) = \log d_F(\mathbf{z}) - \log d_S(\mathbf{z})$，边界由 $s(\mathbf{z}) = \tau$ 定义
  - 上下文锚点：将受限轨迹分为 $J=16$ 个邻域，计算责任加权参考 $\mathbf{a}(\mathbf{z}) = \sum_j \rho_j^F(\mathbf{z}) \mathbf{a}_j^S$

### 2. 状态化边界控制（SBC）
- **风险触发控制**：若 $s_t \leq \tau$，隐藏状态不变，历史状态衰减 $\mathbf{m}_t = \gamma \mathbf{m}_{t-1}$；若 $s_t > \tau$，应用边界修正。
- **法向-切向联合修正**：
  - 法向分量：$\mathbf{n}_t = \mathbf{g}_t / \|\mathbf{g}_t\|_2$（梯度方向，降低当前风险）
  - 切向投影：$\mathbf{P}_t = \mathbf{I} - \mathbf{n}_t\mathbf{n}_t^\top$
  - 修正方向：$\mathbf{d}_t = \text{Normalize}(-\mathbf{n}_t + \lambda_m \mathbf{P}_t \mathbf{m}_{t-1})$
  - 状态更新：$\mathbf{m}_t = \gamma \mathbf{m}_{t-1} + (1-\gamma) q_t \mathbf{v}_t$，其中门控 $q_t = 1 - \exp(-[s_t-\tau]_+/T_g)$ 确保高风险修正积累更多信息
- **上下文锚定**：将修正后状态 $\mathbf{z}_t^e$ 软锚定至参考 $\mathbf{a}_t$：$\mathbf{z}_t^* = (1-\beta_t)\mathbf{z}_t^e + \beta_t \mathbf{a}_t$，通过有限回溯选择 $\beta_t$，确保锚定点仍低于风险边界且有足够密度支撑。
- **预填充与解码一致性**：Prefill 阶段因果扫描初始化 $\mathbf{m}_n$，Decoding 阶段复用同一状态递推，无需存储完整激活历史。

## 实验与结果
- **数据集**：
  - **RWKU**：真实世界实体知识遗忘（FB/QA 探针）
  - **WMDP**：生物安全（Bio）与网络安全（Cyber）危险知识
  - **MUSE-Books**：版权遗忘（哈利波特 corpus）
  - **MMLU**：保留能力评估
- **模型**：Llama3.1-8B-Instruct、Qwen3-14B、SpikingBrain-7B
- **主要结果**（Llama3.1-8B-Instruct）：
  | 方法 | RWKU FB↓ | QA↓ | WMDP Bio↓ | Cyber↓ | R-Ent.↓ | MMLU↑ |
  |---|---|---|---|---|---|---|
  | Base | 88.31 | 74.92 | 72.74 | 47.35 | 91.14 | 64.38 |
  | ST²U (Ours) | **18.40** | **22.08** | **28.04** | **30.48** | **13.76** | 60.58 |
  | SCANS | 22.80 | 28.47 | 31.70 | 34.00 | 53.10 | 57.20 |
  | INNSTEER | 20.60 | 24.91 | 30.60 | 32.80 | 47.20 | 58.60 |
- **跨模型泛化**：Qwen3-14B（R-Ent. 19.84/16.20）、SpikingBrain-7B（R-Ent. 18.10/15.40）均取得最低重新进入率。
- **MUSE-Books 版权遗忘**：ST²U 取得 BLEU=14.61、ROUGE-L=12.90，MMLU=42.34，Fluency=3.39，接近 AS（BLEU=13.72）但 MMLU 高 4.78 点。
- **参数敏感性**：风险边界分位数 $q_\tau \in [88\%, 98\%]$ 变化影响小；$\alpha_{\max}=12$ 为最佳步长上限。
- **消融实验**：移除历史状态 $\mathbf{m}_t$ → R-Ent. 增至 37.50%；移除风险门控 $q_t$ → WMDP 增加 13.09、R-Ent. 增加 32.99；移除残差保留 → MMLU 下降 7.79。
- **鲁棒性**：白盒攻击下 WMDP 恢复仅 +1.16%（vs. RMU +4.87%、SCANS +15.84%）。
- **部署效率**：离线准备仅 2.8 分钟，在线延迟 2.21×（优于 ICUL 的 3.16×）。

## 相关工作脉络
- **参数更新类遗忘**：RMU（将 forget-set 激活推向固定随机向量）、AS（注意力平滑+自蒸馏）、ALTER（token entropy + 非对称 LoRA）——ST²U 不更新权重，通过冻结模型的激活干预实现遗忘，避免重训成本与 retained utility 退化。
- **测试时遗忘-输入层面**：SPUL（soft prompt）、ICUL（in-context few-shot）——ST²U 作用在 hidden state 层面，提供比 prompt 更细粒度的风险抑制。
- **测试时遗忘-解码/激活层面**：SEGUE（entropy-guided decoding）、SCANS（safety-conscious activation steering）、INNSTEER（invertible latent transformations）——这些方法均为 position-wise 独立修正，缺乏跨 token 状态记忆；ST²U 通过固定维度状态 $\mathbf{m}_t$ 显式建模轨迹连续性。
- **激活编辑方法**：CiPO（counterfactual unlearning）、LUME（multitask evaluations）——ST²U 的独特性在于将编辑空间限制在低维可逆坐标内，同时保留正交残差以维持非目标能力。

## 局限性与未来方向
- **离线准备依赖**：需要配对受限/安全轨迹来学习 RKG，虽仅需 2.8 分钟但仍需额外数据准备；未来可探索更稀疏的触发策略与自适应干预调度以降低开销。
- **白盒攻击假设**：鲁棒性评估仅考虑已知 backbone 和低秩扰动注入，未覆盖黑盒提取攻击场景（论文承认 details 延期）。
- **固定维度局限**：状态维度 $k=32$ 对所有任务固定，可能无法最优适配不同知识结构的复杂度差异。
- **长期上下文扩展**：MUSE-Books 已验证长文本续写有效性，但更长序列（如完整书籍）下的轨迹稳定性有待进一步检验。
- **未来方向**：sparser trigger policies、adaptive intervention schedules、扩展至多模态与大规模生成场景。

## 研究启发与可借鉴点
1. **轨迹级状态控制范式**：将 test-time intervention 建模为带历史状态递推的轨迹控制问题，而非孤立点态修正，可迁移至 safety steering、factuality correction 等需要跨 token 一致性的任务。
2. **低维可逆坐标分解**： affine-coupling blocks 实现的可逆变换结合正交残差保留，为激活编辑提供了"精准靶向+全局保真"的通用解耦框架。
3. **密度比风险函数**：使用 KDE 密度比而非单一分布建模，降低对共 Support 区域的敏感性，可推广至其他风险管控场景（如 fairness、bias）。
4. **R-Ent. 指标的引入**：为 test-time unlearning 开辟了"遗忘持久性"评测维度，提醒社区关注单点指标外的轨迹稳定性。
5. **上下文锚定的语义保护机制**：风险下降后软锚定至安全知识参考点，有效防止语义漂移，可借鉴至 other intervention-based alignment methods。

## 关键术语表
- **Test-time Unlearning（测试时机器遗忘）**：冻结模型参数，仅在推理阶段通过 prompt、解码或激活编辑实现知识抑制，避免重训成本。
- **Restricted Knowledge Re-entry（受限知识重新进入）**：测试时干预后，后续解码步骤中隐藏状态重新进入支持受限知识生成的高风险区域的现象。
- **Restricted Knowledge Geometry（RKG）**：基于配对轨迹学习的低维可逆坐标系统，将目标相关知识分离为可编辑坐标 $\mathbf{z}$ 与正交残差 $\mathbf{r}$。
- **Stateful Boundary Control（SBC）**：沿生成轨迹动态监控风险、应用法向-切向联合修正、并通过固定维度状态 $\mathbf{m}_t$ 传递历史修正信息的控制机制。
- **Density Ratio Risk（密度比风险）**：受限分布与 sanitized 分布的 KDE 对数密度比 $s(\mathbf{z}) = \log d_F(\mathbf{z}) - \log d_S(\mathbf{z})$，用于定义连续风险边界。
- **Contextual Anchoring（上下文锚定）**：将修正后状态软投影至由责任加权安全知识参考点 $\mathbf{a}(\mathbf{z})$，防止语义漂移。
- **Affine-coupling Blocks**：用于实现可逆映射 $\psi$ 的流式网络结构，保证坐标变换的可逆性与数值稳定性。
- **R-Ent.（Re-entry Rate）**：确认发生受限知识重新进入的生成样本比例，衡量遗忘持久性的核心指标。

## 可复现要素
- **数据集**：RWKU、WMDP、MUSE-Books、MMLU（均为公开 benchmark）
- **代码/权重**：论文未明确声明开源仓库，但提供详细附录（Appendix B–D）描述实现细节
- **关键超参**：风险维度 $k=32$、邻域数 $J=16$、affine-coupling blocks 数=6（hidden width=128）、阈值分位数=94%、最大步长 $\alpha_{\max}=12$、步长缩放=1.25
- **硬件**：4× NVIDIA RTX 4090 GPUs
- **模型**：Llama3.1-8B-Instruct、Qwen3-14B、SpikingBrain-7B
