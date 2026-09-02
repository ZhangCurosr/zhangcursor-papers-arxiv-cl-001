---
title: "MileGPO-Milestone-Inference-with-Local-Evidence-for-Graph-Ba"
source: https://arxiv.org/pdf/2608.19803v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 02:07:09"
field: "Agent Reinforcement Learning"
keywords: ["Long-horizon LLM Agent", "Credit Assignment", "Graph-based Policy Optimization", "Milestone Discovery", "Reliability-Calibrated Shaping", "Progress-Contrastive Calibration"]
innovations: ["无外部监督的 rollout-native 三步信用校准算法（MD+RCS+PCC）", "利用同源分支反事实对比与局部进度分打破距离平局", "仅在现有 rollouts 图内部挖掘证据，零额外环境交互"]
benchmarks: ["ALFWorld", "Web-Shop"]
---

# 论文速读：MileGPO: Milestone Inference with Local Evidence for Graph-Based Policy Optimization of Long-Horizon LLM Agents

## 一句话总结
论文针对长视距 LLM Agent 强化学习中“最终奖励难以精细分配给中间步骤”的信用分配难题，提出 **MileGPO**——一种仅依赖同策略 rollouts 与任务本地转移图的三步校准算法（里程碑发现 MD、可靠性校准塑形 RCS、进展对比校准 PCC），无需任何外部过程标注或辅助模型。在 ALFWorld 与 Web‑Shop 两个典型长视距基准上均取得 SOTA，且将 ID–OOD 泛化差距压缩至 1.69 分，显著优于现有图基线。

## 研究问题与动机
- **最终奖励监督导致中间信用模糊**：长视距 Agent 任务（如网页导航、物体操作）只有 episode 级成功/失败信号，传统轨迹级方法（如 REINFORCE/PPO）将所有动作赋予相同优势，无法区分有效中间步骤。
- **同状态对比方法仍不足以刻画可靠进展**：GiGPO 等在同一源状态比较分支优势，但未回答“该状态本身是否代表通向目标的可靠进展”，亦未过滤被偶然成功“污染”的候选节点。
- **GraphGPO 的距离衰减信用存在结构性盲区**：GraphGPO 将 grouped rollouts 合并为任务本地有向图，按目标最短路径距离 $d(v,g)$ 衰减赋回报。实验统计显示：ALFWorld/Web‑Shop 中约 **74%** 的转移来自共享状态，但 **54.4%/72.7%** 的同源状态动作对在 GraphGPO 下获得相同的距离回报，且同一转移可同时出现在成功与失败轨迹中，导致信用歧义。
- **现有过程监督依赖昂贵标注或易受分布偏移影响**：Process Reward Model、TreeRPO 等方法需要人工过程标签或改变采样结构，引入额外训练复杂度且泛化不稳定。

## 核心贡献（创新点）
1. **揭示最终目标距离信用的不可靠性**：通过统计分析证明 GraphGPO 的距离衰减无法区分同源不同结局的分支，且成功访问的状态未必代表可靠进展。
2. **提出 Rollout‑Native 的策略优化算法 MileGPO**：仅利用同一 group rollouts 构成的转移图与 on‑policy 结果 reward，无需外部过程标注、critic、reward model 或任何额外环境交互即可完成中间信用校准。
3. **里程碑‑陷阱双势能的可靠性加权塑形**：MD 发现候选里程碑与陷阱，RCS 以归一化分数 $\overline{S}_0^+$、$\overline{S}^-$ 对正负势能进行距离衰减与符号分离，使高置信度锚点获得更强放大、低置信度被抑制。
4. **进展‑对比校准（PCC）进一步过滤噪声**：通过局部距离减少、成功增益、失败超额三项融合进度分数，并结合同源分支反事实对比（BCC）保留/增强/收缩候选，使信用信号与真实局部进展对齐。
5. **ALFWorld 与 Web‑Shop 均达 SOTA，泛化差距显著缩小**：ALFWorld 整体成功率 **94.60%**（较 GraphGPO 提升 **+3.13** 分），ID–OOD 差距仅 **1.69** 分（GraphGPO 为 3.78 分）；Web‑Shop 成功率 **78.58%**（较 GraphGPO 提升 **+3.78** 分），验证了局部证据对歧义分辨的有效性。

## 方法详解
MileGPO 在 GraphGPO 的图结构上叠加三步嵌套信用设计，最终仅修改 step‑level advantage 估计。

### 1. MD：Milestone Discovery（里程碑发现）
- **候选提取**：对 task $q$ 的 $K$ 条 grouped rollouts，分别统计成功集 $\mathcal{T}_q^+$ 与失败集 $\mathcal{T}_q^-$。
  - **Milestone 候选**：至少被一条成功轨迹访问的非终态节点 $v$，评分要素：
    - 条件成功率优势 $\ell^+(v)=\max(p^+(v)-p_q^+,0)$，其中 $p^+(v)=|\mathcal{T}^+(v)|/|\mathcal{T}(v)|$
    - 成功覆盖度 $m(v)=|\mathcal{T}^+(v)|/\max(|\mathcal{T}_q^+|,1)$
    - 图连通中心度 $C(v)=(\text{in}+\text{out})/\max_{v'\in S^+}(\text{in}+\text{out})$
    - $S_0^+(v)=w_s\ell^+(v)+w_m m(v)+w_c C(v)$
  - **Trap 候选**：仅被失败轨迹访问、且在失败中反复出现或回访的节点 $v$，评分要素：
    - 超额失败率 $\ell^-(v)=\max(p^-(v)-p_q^-,0)$
    - 失败覆盖度 $f^-(v)=|\mathcal{T}^-(v)|/\max(|\mathcal{T}_q^-|,1)$
    - 回访强度 $L(v)=n_{re}(v)/\max(|\mathcal{T}(v)|,1)$
    - $S^-(v)=w_f\ell^-(v)+w_l f^-(v)L(v)$
- **均匀传播**：先以统一权重 $\omega^{d(s,v)}$ 构建初步正/负势能 $\Phi_S^+(s)=\max_{v\in S^+}\omega^{d(s,v)}$，$\Phi_S^-(s)=\max_{v\in S^-}\omega^{d(s,v)}$。

### 2. RCS：Reliability‑Calibrated Shaping（可靠性校准塑形）
- **分数归一化**：将 $S_0^+$、$S^-$ 在 task 内 max‑normalize 得 $\overline{S}_0^+$、$\overline{S}^-$。
- **加权势能**：$\Phi_R^+(s)=\max_{v\in S^+}\overline{S}_0^+(v)\omega^{d(s,v)}$，$\Phi_R^-(s)=\max_{v\in S^-}\overline{S}^-(v)\omega^{d(s,v)}$。
- **单向增量**：transition $(s_t\to s_{t+1})$ 的里程碑/陷阱势能增量为
  $$\delta_t^+=\max(\gamma_\Phi w_+\Phi^+(s_{t+1})-w_+\Phi^+(s_t),0),\quad \delta_t^-=\max(\gamma_\Phi w_-\Phi^-(s_{t+1})-w_-\Phi^-(s_t),0)$$
  仅保留正向变化，体现“靠近可靠里程碑受奖、靠近陷阱受罚”的非对称信用。

### 3. PCC：Progress‑Contrastive Calibration（进展对比校准）
- **BCC（Branch‑Counterfactual Credit）**：对源状态 $u$ 的所有出边分支 $e_i=(u,v)$，计算其成功率与同父其他分支平均成功率的差值 $\widetilde{c}_{bcc}(e_i)=p^+(e_i)-\mu^+(e_i)$，经 task 内绝对值最大 margin 归一化后取入边的最强正 margin 作为 $b(v)$。
- **Local Progress**：对入边 $e=(u,v)$ 计算进度得分
  $$\psi(e)=\alpha_d\Delta d(e)+\alpha_s[p^+(v)-p_q^+]-\alpha_f\ell^-(e)$$
  其中 $\Delta d(e)=d(u,g)-d(v,g)$，$\ell^-(e)$ 为同父分支失败率超额（无兄弟时退化为节点超额失败率）。
- **候选更新**：取成功入边中最大进度 $\widetilde{\psi}(v)$ 并归一化得 $\overline{\psi}(v)$，合并为 $E(v)=w_{bc}b(v)+w_{pg}\overline{\psi}(v)$。通过保留指示 $z(v)$（满足进度>0 或覆盖达标或 BCC 触发）决定：
  $$S_{pcc}^+(v)=\begin{cases}S_0^+(v)[1+w_{pcc}E(v)],& z(v)=1\\ \rho S_0^+(v),& z(v)=0\end{cases}$$
  再次 max‑normalize 得到 $\overline{S}_{pcc}^+$。

### 4. 最终 Step Return 与 Policy Update
- 合并势能：$\Phi^+(s)=\max_{v\in S^+}\overline{S}_{pcc}^+(v)\omega^{d(s,v)}$，$\Phi^-(s)=\Phi_R^-(s)$。
- Step return：$r_t^M=c\gamma_G^{d(s_{t+1},g)}+c\lambda(\delta_t^+-\delta_t^-)$。
- **分离归一化**：对同 task 同源状态下的转移，分别归一化 $\widehat{A}_t^G=\mathrm{Norm}_{q,s_t}(r_t^G)$ 与 $\widehat{A}_t^{mix}=\mathrm{Norm}_{q,s_t}(r_t^M)$，提取残差 $\widehat{A}_t^{res}=\widehat{A}_t^{mix}-\widehat{A}_t^G$，避免微小校正在大 tie 场景下主导尺度。
- Token‑level 优势：
  $$A_t=w_{step}\left[\widehat{A}_t^G+\eta\widehat{A}_t^{res}\right]+w_{episode}\mathrm{Norm}_q(Z_i)$$
  其中 $Z_i$ 为轨迹 token reward 之和，$\eta\in[0,1]$ 控制校正强度。Actor 使用与基线相同的 clip 目标与 KL 正则。

**理论性质**（Appendix B）：
- **Prop 1**：MileGPO 对 GraphGPO 的 step 分量构成凸插值 $B_t=(1-\eta)B_t^G+\eta B_t^M$，$\eta=0$ 还原 GraphGPO。
- **Prop 2**：对 GraphGPO 下同等距 tie 的两条出边，只要 $\eta>0$，经归一化后的 shaping 差异即转化为严格优势序，打破距离平局。
- **Prop 3**：在报告超参下，raw correction 有界于 $[-0.625,2.5]$，保证训练稳定。

## 实验与结果
- **数据集**：
  - **ALFWorld**（文本导向 embodied 导航与物品操作）：3,553 训练任务，含 ID 与 OOD 评测集。
  - **Web‑Shop**（网页商品搜索购买）：official‑small 512 样本评测。
- **基线**：GiGPO（group‑in‑group 同状态对比）、GraphGPO（距离衰减图信用）为最主要 step‑level/graph‑based 对比；亦对比 PPO、RLOO、GRPO 及 ReAct/Reflexion 等 prompt 方法。所有本地复现实验使用统一 Qwen2.5‑1.5B‑Instruct 与相同 rollout/优化配置。
- **主要结果**：
  - **ALFWorld 总体成功率**：MileGPO **94.60%**，相对 GraphGPO（91.47%）**+3.13** 分，相对 GiGPO（90.17%）**+4.43** 分。
  - **Web‑Shop 成功率**：**78.58%**（GraphGPO 74.80%，+3.78 分）；**任务得分** **90.29%**（GraphGPO 88.24%，+2.05 分）。
  - **泛化**：ALFWorld ID‑OOD 差距 **1.69** 分，低于 GiGPO（1.89）与 GraphGPO（3.78），方差亦更小。
  - **Ablation**（Table 3）：逐步移除 PCC 与 RCS 均在两 benchmark 上降级；移除 PCC 使 ALFWorld OOD 下降 3.71 分、ID–OOD 差距扩大 3.32 分，说明 PCC 主要贡献于泛化与部分任务完成；再移除 RCS 导致 WebShop 成功率额外下降 3.32 分，表明 RCS 负责抑制不一致里程碑。
  - **Credit diagnostics**（Figure 4、9）：RCS 能纠正 **26.5%** 的 GraphGPO tie（Uniform MD 仅 17.0%）；PCC 逐步提升带有本地证据的候选比例。

## 相关工作脉络
1. **GiGPO（Feng et al., 2025）**：在同状态比较 sibling branches 的优势，但未对候选里程碑按结果置信度加权，亦未引入局部进展信号，导致同等距 tie 仍难区分。
2. **GraphGPO（Cheng et al., 2026）**：首次将 grouped rollouts 合并为任务本地图并沿最短路径衰减赋回报，但完全依赖距离，无法区分“同距离但成功率迥异”的状态，且成功访问节点可能因距离远而信用微弱。
3. **TreeRPO / TreeRL（Yang et al., 2025; Hou et al., 2025）**：基于树状 prefix 共享做 branch 对比，需显式树形采样结构，依赖搜索/分支机制，引入额外计算。
4. **Process Reward Model（Lightman et al., 2024; Wang et al., 2024）**：通过人工过程标签或独立 PRM 提供 step‑level 监督，精度高但标注成本高且易受分布偏移退化。
5. **StepPO / BiPACE / Progress Advantage（Wang et al., 2026a,b; Oh et al., 2026）**：分别利用 step‑aligned 信号、bisimulation 对比、策略派生进展，但多需额外价值模型或历史条件假设；MileGPO 则完全从现有 rollout 图内部挖掘证据。
6. **GRPO / RLOO / DAPO（Kool et al., 2019; Ahmadian et al., 2024; Yu et al., 2025）**：用于可验证推理的相对奖励比较，适用于单轮生成；本文将其思想扩展至多步 agent 环境交互中的图信用校准。

## 局限性与未来方向
- **仅适用于有明确 episode‑level 成功/失败标记的任务**，对连续评分或细粒度过程奖励环境的泛化需进一步验证。
- **候选发现依赖固定阈值与手工权重**（如 trap 需至少 2 次失败访问、$\theta_m$ 覆盖阈值），在不同环境上需调参（论文 ALFWorld 与 WebShop 使用不同配置）。
- **未考虑多模态观测或更复杂的长期依赖结构**，仅在文本/图标界面基准上评估。
- **潜在遗漏**：MD 基于当前 group 统计挖掘，若某中间步骤仅出现在极少量成功轨迹中可能被漏检；未来可引入更鲁棒的贝叶斯或时序平滑。
- **未来方向**：将 PCC 与 BCC 框架推广至多智能体协作、带工具调用链的长 Horizon 任务，或与 Value‑free 的 offline RL 结合。

## 研究启发与可借鉴点
- **Rollout 图即自监督信用矿藏**：无需额外标注，仅凭同一 task 内 grouped rollouts 的共享状态与分支结构即可挖掘中间信用的有效代理，为资源受限的 agent 训练提供了零开销的校准范式。
- **“可靠性加权 + 局部对比”双重过滤**：RCS 按结果置信度放大/压制候选，PCC 再用同源分支反事实与目标距离减少做二次验证，二者互补且可叠加，避免单一信号被噪声污染。
- **Separate normalization 防止校正主导**：将 GraphGPO 原始回报与 shaping 后回报分别归一化后再求残差，避免了在大量 tie 场景下微小校正掩盖主信号，值得在同类图信用算法中沿用。
- **与团队方向的结合点**：
  - 若团队关注代码生成/执行、机器人规划、自动化运维等长链任务，可直接复用 MileGPO 的 MD+RCS+PCC 模块，接入现有 grouped rollout pipeline。
  - 可探索将 BCC 的分支对比思路迁移至多路检索/多智能体辩论场景，用于筛选高质量推理路径。
  - PCC 的进度得分 $\psi(e)$ 可与过程奖励模型（PRM）的软标签融合，在保留零样本优势的同时进一步提升早期训练稳定性。

## 关键术语表
- **Credit Assignment（信用分配）**：在序列决策中将 episode 级最终奖励分解为每个 step 的归因值，以指导策略梯度更新。
- **Milestone（里程碑）**：成功轨迹中反复访问且条件成功率高于组平均的状态，视为通向目标的可靠中间锚点。
- **Trap（陷阱）**：仅出现在失败轨迹中且高频回访的状态，代表易导致任务失败的局部回路或错误路径。
- **On‑policy Rollout Group（同策略 Rollout 组）**：当前策略对同一任务采样的 $K$ 条交互轨迹，构成后续图构建与统计的基础单元。
- **Grouped Rollout Graph（Grouped 转移图）**：将同一 task 的多条 rollout 按标准化状态节点合并为有向图 $\mathcal{G}_q=(\mathcal{V}_q,\mathcal{E}_q)$，边为观测到的状态‑动作转移。
- **Branch‑Counterfactual Credit（分支反事实信用，BCC）**：通过比较同一源状态下不同出边分支的成功率差异，量化该边的相对优势。
- **Reliability‑Calibrated Shaping（RCS）**：按候选节点的归一化置信分数进行距离衰减的势能塑形，使高可靠里程碑/陷阱获得更强的正/负信用放大。
- **Progress‑Contrastive Calibration（PCC）**：利用距离减少、成功率增益与失败超额三项构造局部进度分，并结合 BCC 对里程碑候选进行保留/增强/收缩的二次校准。

## 可复现要素
- **数据集**：ALFWorld、Web‑Shop 均为公开 benchmark（论文已给出官方 split 协议）。
- **代码/权重**：论文未明确声明代码开源，需关注 arXiv 源码页或作者主页后续更新。
- **关键超参**（汇报值）：
  - 基础：$c=10,\;\gamma_G=0.20,\;\omega=0.20,\;\gamma_\Phi=1.0,\;\lambda=0.25$
  - 势能权重：$w_+=1.0,\;w_-=0.25$；MD 得分权重 $w_s=w_m=w_f=1,\;w_c=w_l=0.25$
  - PCC 权重：$\alpha_d=\alpha_s=\alpha_f=w_{bc}=w_{pg}=w_{pcc}=1$
  - 策略更新：$w_{step}=w_{episode}=1$
  - 环境差异：ALFWorld $\theta_m=1.1,\;\rho=0,\;\eta=0.20,\;\kappa_{bc}=0$；WebShop $\theta_m=0.5,\;\rho=0.5,\;\eta=1.0,\;\kappa_{bc}=1$
- **训练配置**：Qwen2.5‑1.5B‑Instruct，lr=$10^{-6}$，KL 系数 0.01，300 steps，per‑GPU 2×H20，batch/minibatch 等详见附录 Table 4。
