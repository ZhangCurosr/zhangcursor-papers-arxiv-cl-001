---
title: "ST-sup-2-sup-U-Stateful-Test-Time-Unlearning-via-Restricted"
source: https://arxiv.org/pdf/2608.23034v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 00:58:48"
field: "大语言模型安全与可控生成"
keywords: ["Machine Unlearning", "Test-Time Unlearning", "Activation Editing", "Restricted Knowledge Boundary Control", "Stateful Trajectory Control", "LLM Alignment"]
innovations: ["提出受限知识风险几何（RKG），通过低维可逆坐标与KDE密度比建模连续风险边界，实现目标/正交分量解耦", "设计状态化边界控制（SBC），结合法向-切向联合修正与固定维度历史状态记忆，缓解自回归生成中的知识重入", "建立轨迹级遗忘形式化框架，将测试时干预从点对点修正扩展为跨token的持续风险约束"]
benchmarks: ["RWKU", "WMDP", "MUSE-Books", "MMLU"]
---

# 论文速读：ST²U: Stateful Test-Time Unlearning via Restricted Knowledge Boundary Control

## 一句话总结
本文提出 ST²U，一种面向冻结大语言模型的"状态化测试时遗忘"框架，通过将受控的测试时干预建模为**轨迹级边界控制**，有效缓解自回归生成过程中受限知识的**重新进入（re-entry）**问题，在保持非目标能力的前提下实现更持久、更稳定的知识遗忘。

## 研究问题与动机
1. **现有测试时遗忘方法的局部性缺陷**：当前激活编辑（activation-editing）等方法仅在单个解码位置施加点对点修正，忽略了自回归生成中隐藏状态会持续从 prompt、KV cache 和已生成前缀重构的过程，导致局部有效的修正无法持久。
2. **受限知识重新进入（Restricted Knowledge Re-entry）**：即便在某步成功将隐藏状态推离受限知识区域，后续解码步骤仍可能"漂移回"支持受限生成的风险区域，论文将此定义为 $R_{\mathcal{F}}(\mathbf{h}_{t_0}') \leq \tau$ 但 $\exists t_1 > t_0: R_{\mathcal{F}}(\mathbf{h}_{t_1}) > \tau$。
3. **参数更新方法的成本与效用退化**：训练时近似遗忘（如 GA、RMU、ALTER 等）计算开销大且反复更新参数会损害模型整体效用，亟需冻结参数下的在线控制方案。
4. **现有评测指标的不足**：主流评测只关注单点输入/输出的抑制效果，未要求修正在整个生成轨迹中保持一致性，低估了重入风险。

## 核心贡献（创新点）
1. **构建受限知识风险几何（Restricted Knowledge Geometry, RKG）**：通过配对上下文轨迹学习低维可逆坐标，将目标相关激活映射到可区分维度，同时保持正交非目标分量不变，从而提供可解释、可控的风险度量而非粗暴的激活截断。
2. **提出状态化边界控制（Stateful Boundary Control, SBC）**：以正交投影分解梯度方向（法向降风险 + 切向保留历史修正），结合上下文锚定（contextual anchoring）防止语义漂移，并在固定维度状态 $\mathbf{m}_t \in \mathbb{R}^k$（$k \ll d$）中累积历史修正方向，实现跨 token 的时间一致性。
3. **建立轨迹级遗忘形式化框架**：将测试时遗忘明确表述为沿生成轨迹的最小扰动约束优化问题（Eq. 2），纠正了现有方法仅针对孤立状态的局部视角，强调每步修正应服务于整体轨迹的风险抑制。
4. **系统性基准验证**：在 RWKU（实体遗忘）、WMDP（生物安全/网络安全）、MUSE-Books（版权长文续写）三个基准及三类模型架构（Llama、Qwen、SpikingBrain）上验证，在遗忘-保留权衡与重入率方面取得最优或次优结果。

## 方法详解
### 1. Restricted Knowledge Geometry（RKG）
- **配对轨迹构造**：对每个受限序列 $x_i^F \in \mathcal{F}$，离线构建保留共享高层上下文和非目标内容、但省略目标细节的配对参考序列 $x_i^S$，提供对比监督以隔离目标相关激活变异。
- **低维可逆坐标**：设激活统计量 $\pmb{\mu}, \pmb{\sigma}$，标准化 $\widetilde{\mathbf{h}} = (\mathbf{h} - \pmb{\mu}) \oslash \pmb{\sigma}$；从对齐轨迹差中学习列正交基 $\mathbf{U} \in \mathbb{R}^{d \times k}$，投影得到目标相关坐标 $\mathbf{p} = \mathbf{U}^\top \widetilde{\mathbf{h}}$ 与正交残差 $\mathbf{r} = (\mathbf{I} - \mathbf{U}\mathbf{U}^\top)\widetilde{\mathbf{h}}$；再通过可逆映射 $\psi: \mathbb{R}^k \to \mathbb{R}^k$ 变换得 $\mathbf{z} = \psi(\mathbf{p})$。重构时仅修改 $\mathbf{z}$，保留 $\mathbf{r}$ 不变：
$$
\widetilde{\mathbf{h}} = \mathbf{r} + \mathbf{U}\psi^{-1}(\mathbf{z})
$$
- **风险几何与上下文参考**：在 $\mathbf{z}$ 空间使用 KDE 建模受限分布 $d_F$ 与干净分布 $d_S$，定义风险得分 $s(\mathbf{z}) = \log d_F(\mathbf{z}) - \log d_S(\mathbf{z})$，阈值 $\tau$ 对应边界。将受限轨迹分为 $J$ 个邻域，以责任权重 $\rho_j^F(\mathbf{z})$ 加权得到上下文锚点 $\mathbf{a}(\mathbf{z}) = \sum_j \rho_j^F(\mathbf{z}) \mathbf{a}_j^S$，用于防止语义漂移。

### 2. Stateful Boundary Control（SBC）
- **风险触发控制**：每步评估 $s_t = s(\mathbf{z}_t)$，若 $s_t \leq \tau$ 则不修改，历史状态指数衰减 $\mathbf{m}_t = \gamma \mathbf{m}_{t-1}$；若 $s_t > \tau$ 则触发边界修正。
- **法向-切向联合方向**：令 $\mathbf{g}_t = \nabla s(\mathbf{z}_t)$，单位法向量 $\mathbf{n}_t = \mathbf{g}_t / \|\mathbf{g}_t\|$，切向投影矩阵 $\mathbf{P}_t = \mathbf{I} - \mathbf{n}_t\mathbf{n}_t^\top$，修正方向为：
$$
\mathbf{d}_t = \text{Normalize}(-\mathbf{n}_t + \lambda_m \mathbf{P}_t \mathbf{m}_{t-1})
$$
确保 $\mathbf{g}_t^\top \mathbf{d}_t < 0$（历史切向分量不会反转法向风险下降）。
- **迭代边界闭合**：沿 $\mathbf{d}_t$ 做有限步迭代更新 $\mathbf{z}_t$，直至越过边界或达最大迭代次数 $K_{\max}$。
- **上下文锚定与重构**：获得边界修正状态 $\mathbf{z}_t^e$ 后，软锚定至上下文参考 $\mathbf{a}_t$：
$$
\mathbf{z}_t^* = (1 - \beta_t)\mathbf{z}_t^e + \beta_t \mathbf{a}_t, \quad \mathbf{h}_t' = (\mathbf{r}_t + \mathbf{U}\psi^{-1}(\mathbf{z}_t^*)) \odot \pmb{\sigma} + \pmb{\mu}
$$
通过有限回溯选择 $\beta_t \in [0, \beta_{\max}]$，确保锚定后仍低于风险边界且有足够密度支撑。
- **历史状态更新**：压缩每次修正方向至固定维度状态：
$$
\mathbf{v}_t = \frac{\mathbf{z}_t^* - \mathbf{z}_t}{\|\mathbf{z}_t^* - \mathbf{z}_t\| + \epsilon}, \quad q_t = 1 - \exp\left(-\frac{[s_t - \tau]_+}{T_g}\right), \quad \mathbf{m}_t = \gamma \mathbf{m}_{t-1} + (1-\gamma)q_t \mathbf{v}_t
$$
高Violation 的步骤获得更大权重，累积一致方向以抑制后续重入。

## 实验与结果
- **数据集与基线**：RWKU（实体遗忘，FB/QA探针）、WMDP（生物安全/网络安全，25%随机选择=最佳遗忘）、MUSE-Books（《哈利·波特》版权长文续写，BLEU/ROUGE-L）；基线包括 RMU、ICUL、AS、ULD、SCANS、INNSTEER 等。
- **主要遗忘-保留权衡**：ST²U 在 Llama3.1-8B 上 RWKU FB=18.40%、QA=22.08%，MMLU 保留 60.58%；Qwen3-14B 上 FB=16.10%、QA=24.57%；SpikingBrain-7B 上 FB=21.40%、QA=23.76%。在 MUSE-Books 上 BLEU=14.61，接近 AS（13.72）但 MMLU 高出 4.78 点。
- **重入率（R-Ent.）对比**：ST²U 将重入率控制在 **13.76%–19.84%**，显著优于所有测试时基线（46.50%–59.10%），验证了状态化轨迹控制的核心价值。
- **白盒攻击鲁棒性**：在低秩隐藏状态扰动攻击下，ST²U 仅损失 1.16% WMDP 且 MMLU 漂移 -1.92%，远优于 SCANS（+15.84% 恢复 / +1.28% MMLU 变动）与 RMU（+4.87% 恢复 / -17.32% MMLU 损失）。
- **层鲁棒性**：在 Llama3.1-8B 的 16 层（16–31）中，13 层改进、10 层增益 ≥5 WMDP 点，层选择敏感性显著降低。
- **部署效率**：离线准备仅需 2.8 分钟，在线延迟 2.21×，优于多数参数更新方法且显著优于 ICUL（离线0但延迟 3.16×且遗忘较弱）。

## 相关工作脉络
1. **参数更新遗忘**（GA、RMU、AS、ALTER）：通过梯度上升或表征重定向修改权重，ST²U 定位差异在于**冻结参数+在线激活控制**，避免重复更新的效用退化与计算开销。
2. **测试时遗忘-输入/解码级**（SPUL、ALU、SEGUE）：软提示或熵引导解码仅作用于输入或 token 分布层面，ST²U 直接在**隐藏状态轨迹**上施加边界约束，提供更细粒度的语义级控制。
3. **激活编辑方法**（CiPO、LLM Unlearning via Neural Activation Redirection）：现有激活编辑为点对点修正，无跨 token 历史，ST²U 的核心突破是**引入固定维度状态记忆**以维持修正的方向一致性。
4. **SCANS / INNSTEER**：虽同为测试时激活干预，但 SCANS 对安全相关激活过度 steer 导致效用损失，ST²U 通过**正交残差保持+上下文锚定**减轻非目标语义漂移。
5. **In-Context Unlearning**（ICUL）：仅通过提示词诱导遗忘，无需离线几何学习，但目标遗忘较弱且易受上下文干扰；ST²U 需离线训练但提供更强的持久抑制。
6. **知识几何建模**：与 RKG 思想相近的表征学习方法多集中于监督分类，本文首次将其用于**连续风险边界建模**并结合 KDE 密度比实现任务自适应阈值。

## 局限性与未来方向
1. **离线几何学习依赖配对数据**：RKG 需要构造高质量的配对参考轨迹（相同高层上下文、不同目标细节），在复杂或多模态场景下配对数据难以自动获取。
2. **在线延迟仍偏高**：尽管远低于参数更新方法，但 2.21× 延迟（相对 Base）对于实时性要求极高的部署场景仍是瓶颈。
3. **白盒假设局限**：鲁棒性实验仅覆盖了已知防御架构的低秩扰动攻击，未评估黑盒模型提取、上下文注入等更强攻击形式。
4. **固定超参泛化**：风险分位数阈值 $q_\tau$ 在不同 backbone 上表现非单调，论文采用共享阈值而非 per-model 调优，可能在极端场景下牺牲部分性能。
5. **未来方向（作者自述）**：研究更稀疏的触发策略与自适应干预调度，以在保持遗忘效果的同时降低在线开销。

## 研究启发与可借鉴点
1. **轨迹级控制视角可迁移至其他在线干预任务**：如安全对齐、价值观调控、多轮对话中的知识抑制等，任何需要"跨 token 一致性"的激活干预均可借鉴 SBC 的状态化设计。
2. **正交残差保持策略**：将激活分解为目标相关子空间（需编辑）与正交补空间（保持不变），这一思路可推广至多概念共同编辑、多任务 forgetting 等场景，避免"过激修正"导致的效用坍缩。
3. **低维可逆坐标 + KDE 密度比的风险建模**：相比直接使用激活范数或线性边界，可逆坐标提升了流形可分离性，密度比降低了对双分布共同支持区域的敏感度，该几何建模方式可与对比学习、normalizing flow 等方法结合。
4. **状态记忆机制的形式化**：固定维度状态 $\mathbf{m}_t$ 的更新规则（指数衰减 + 门控加权 + 切向投影）提供了一种简洁的"历史修正压缩"范式，可复用于其他需要时间一致性约束的在线控制问题。
5. **层鲁棒性分析为干预位置选择提供启发**：实验表明 ST²U 在 13/16 层均有效，降低了对精确层选择的依赖，后续工作可进一步探索"最佳干预层"的可迁移规律与自动化搜索。

## 关键术语表
**Restricted Knowledge Re-entry（受限知识重入）**：在单步测试时干预成功将隐藏状态推离风险区域后，后续自回归解码因上下文演化仍使状态重新进入受限知识区域的失败模式。
**Restricted Knowledge Geometry（RKG）**：通过配对轨迹学习低维可逆坐标空间，在该空间中用 KDE 密度比刻画受限知识风险边界，同时保留正交方向上的非目标激活。
**Stateful Boundary Control（SBC）**：在推理过程中沿生成轨迹在线监控风险，通过法向下降（直接降风险）与切向历史投影（保持修正方向一致性）的组合方向施加最小扰动修正。
**Contextual Anchoring（上下文锚定）**：以当前激活位置对应的干净参考 $\mathbf{a}(\mathbf{z})$ 为锚点，在边界修正后通过软插值防止语义漂移，确保修正后状态仍在干净分布的支撑区域内。
**Risk Gate（风险门控）**：门控因子 $q_t = 1 - \exp(-[s_t - \tau]_+/T_g)$，根据当前风险违反程度加权历史修正方向的累积，高风险步骤获得更高权重。
**Re-entry Rate（R-Ent.）**：论文提出的新评测指标，衡量"初始风险闭合后在后续解码中再次检测到违规"的比例，用于量化测试时遗忘的持久性。
**Invertible Coordinate Transform**：通过正交投影 + 可逆映射 $\psi$ 将高维激活投影到低维目标空间，实现精确可逆变换，使编辑仅在目标坐标上进行而正交残差保持不变。
**Paired Trajectory Construction**：为每个受限样本构造保留共享高层语境但省略目标细节的配对参考序列，用于离线学习目标激活变异与通用上下文结构的解耦。

## 可复现要素
- **数据集**：RWKU（Jin et al. 2024）、WMDP（Li et al. 2024）、MUSE-Books（Shi et al. 2025）、MMLU（Hendrycks et al. 2020）——均为公开基准。
- **代码/权重**：论文未明确声明代码开源，需关注作者主页或后续发布。
- **关键超参**：风险维度 $k=32$，上下文邻域数 $J=16$，affine-coupling block 数 6、隐藏宽度 128，风险阈值分位数 94%，最大步长 $\alpha_{\max}=12$，步长缩放 1.25。
- **硬件**：NVIDIA RTX 4090 × 4。
- **模型**：Llama3.1-8B-Instruct、Qwen3-14B、SpikingBrain-7B。
