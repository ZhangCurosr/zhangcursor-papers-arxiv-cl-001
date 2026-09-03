---
title: "Safety-Hacking-in-Constrained-Best-of-N-Inference-time-Scali"
source: https://arxiv.org/pdf/2608.22915v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 01:00:00"
field: "LLM推理时安全与对齐"
keywords: ["inference-time scaling", "safety hacking", "best-of-N sampling", "reward overoptimization", "coverage control", "safety alignment"]
innovations: ["形式化安全劫持并推导CBoN有限N上尾边界及渐近必然性定理", "提出基于χ²覆盖控制的cPes方法以获得N无关安全上界"]
benchmarks: ["JailbreakBench", "HarmBench", "AdvBench"]
---

# 论文速读：Safety-Hacking-in-Constrained-Best-of-N-Inference-time-Scaling

## 一句话总结
本文揭示了约束条件下 Best-of-N (CBoN) 推理时扩展的一个系统性风险：**学习性安全过滤器允许不安全输出混入可行集后，奖励模型的最大值选择会进一步放大这种污染**，即"安全劫持（safety hacking）"。作者给出了有限 N 边界和渐近必然性定理，并提出了受覆盖控制约束的悲观采样（cPes）作为缓解手段。

## 研究问题与动机
1. **推理时管线的双重失败机制**：基础模型推理管线通常先用学习的安全模型过滤候选输出，再用奖励模型对通过者排序。安全过滤器存在假阳性（错误接受不安全输出），而奖励模型在后续最大化过程中会进一步放大这些残留误差。
2. **平均安全精度不足以预测安全性**：即使不安全假接受质量和奖励/安全代理的平均误差趋于零，只要不安全类在奖励尾部分布上占优，CBoN 在高 N 下的安全劫持概率仍趋于 1。
3. **现有方法的不足**：先前关于奖励过优化（reward overoptimization）的研究聚焦于奖励代理的不完美；而 BoN 越狱（jailbreaking）关注重复随机攻击的成功率。本文揭示了过滤器污染与奖励放大之间的交互，这是一个此前未被系统分析的新故障源。
4. **推理时扩展的实践紧迫性**：随着 LLM 推理计算不断扩展（如 test-time compute scaling），N 增大是常见操作，安全劫持的渐近必然性意味着这种扩展策略本身可能带来系统性安全风险。

## 核心贡献（创新点）
1. **形式化"安全劫持"概念**：将安全劫持定义为"输出通过了学习的安全约束，但违反了真实安全标准"的选择事件，首次明确区分了过滤器污染与奖励放大两个阶段。
2. **推导出 CBoN 的有限 N 界与渐近必然性定理**：建立了由安全/不安全候选联合奖励上尾决定的精确边界；证明当不安全类的奖励上尾更重时，安全劫持概率随 N 趋于 1，即使平均代理误差任意小（Corollary 4.3）。
3. **提出覆盖控制（coverage control）原则与 cPes 实例化**：在 χ² 散度有界条件下推导出不依赖 N 的安全劫持上界，并以约束悲观采样（constrained pessimistic sampling, cPes）——一种带 clipped 奖励的正则化重加权方案——作为具体实现。
4. **提供双阶段机制的实验验证**：在玩具问题和真实 LLM 上都分解了"仅暴露"（unsafe-only exposure）与"竞争性胜出"（competitive unsafe wins）两个分量，实证确认了奖励尾竞争是驱动安全劫持放大的主因。

## 方法详解
**问题设定**：对固定 prompt $x$，存在未知的真实安全函数 $g^\star$ 和奖励函数 $r^\star$；推理时仅能使用参考策略 $\pi_\mathrm{ref}$、学习安全代理 $\hat{g}$ 和学习奖励代理 $\hat{r}$。定义代理可行集 $\widehat{\mathcal{S}}_+(x,b)=\{y:\hat{g}(x,y)\geq b\}$，其可分解为真实安全交可行集 $A$ 和安全但被误接受的集合 $B$。

**联合奖励上尾**：核心量 $\widehat{\Psi}_\diamond(t;x,b)=\mathbb{P}_{y\sim\pi_\mathrm{ref}}[y\in\diamond,\hat{r}(x,y)>t]$，融合了两类输出的参考概率与奖励得分尾部行为。

**CBoN 有限 N 界（Theorem 4.1）**：定义 $\widehat{L}_N(t)=(1-\widehat{\Psi}_A(t))^N-(1-\widehat{\Psi}_A(t)-\widehat{\Psi}_B(t))^N$ 和 $\widehat{U}_N(s)$ 类似形式，则 $\widehat{L}_N(t)\leq \mathbb{P}[\hat{y}_N^\mathrm{BoN}\in B]\leq\widehat{U}_N(s)$。下界是"所有 A 类样本得分≤t 且至少一个 B 类样本得分>t"的概率；上界是"至少一个 A 类得分>s 且无 B 类得分>s"的补事件。

**渐近必然性（Theorem 4.2）**：若存在 $t_N$ 使 $N\widehat{\Psi}_A(t_N)\to 0$ 且 $N\widehat{\Psi}_B(t_N)\to\infty$，则 $\lim_{N\to\infty}\mathbb{P}[\hat{y}_N^\mathrm{BoN}\in B]=1$。关键条件是两类奖励上尾的相对分离，而非假阳性质量的大小。

**覆盖控制定理（Theorem 5.1）**：定义代理可行参考分布 $\pi_\mathrm{ref}^\sharp$ 及覆盖系数 $C_\pi^\sharp=1+\chi^2(\pi\|\pi_\mathrm{ref}^\sharp)$，则对任意满足绝对连续的推理策略，其从 B 中选中的概率偏差被 $\sqrt{(C_\pi^\sharp-1)\kappa(1-\kappa)}$ 控制，其中 $\kappa=\pi_\mathrm{ref}^\sharp(B)$ 为污染基线。

**cPes（约束悲观采样，Proposition 5.2）**：将奖励截断至 $[\widehat{R}_\min,\widehat{R}_\max]$，求解正则化问题 $\hat{\pi}=\arg\max_\pi\{\mathbb{E}_\pi[\tilde{r}]-\frac{\beta}{2}(C_\pi^\sharp-1)\}$，解为指数族形式 $\hat{\pi}(y|x)\propto\pi_\mathrm{ref}^\sharp(y|x)\cdot\mathrm{ReLU}((\tilde{r}(x,y)-\lambda)/\beta)$，覆盖系数有界：$C_{\hat{\pi}}^\sharp\leq 1+\widehat{R}_\mathrm{span}/\beta$。

**有限 N 分解（Eq.22）**：实验中将 CBoN 安全劫持率分解为 $\Pr[K_B>0,K_A=0|\cdots]$（仅暴露项）和 $\Pr[K_A>0,K_B>0,M_B>M_A|\cdots]$（竞争胜出项），前者衡量仅因无安全候选而被选中的概率，后者衡量不安全候选通过奖励竞争击败安全候选的概率。

## 实验与结果
**玩具实验**：三分类设置（A=安全可行，B=不安全但可行，C=被拒绝），B 类质量仅 0.01 但其奖励噪声标准差 $\sigma_B=1.0$ 远大于 $\sigma_A=0.2$。CBoN 的安全劫持率从 $N=1$ 时的 0.023 激增至 $N=8192$ 时的 0.9996，同时平均真实奖励从约 0.786 降至 0.200；cPes（$\beta=1.0$）将劫持率稳定在 ~0.015–0.024。

**LLM 实验**：
- **数据集**：JailbreakBench（714 测试 + 179 验证）、HarmBench、AdvBench（去重后）。
- **模型**：生成器 Qwen/Qwen2.5-7B-Instruct（temperature=1.0, top-p=0.95, 512 tokens）；安全过滤器 Llama-Guard-3-8B（阈值 $b=0.95$）；奖励模型 PKU-Alignment/beaver-7b-v1.0-reward；安全性评估用 HarmBench-Llama-2-13b-cls。
- **主要结果**：
  - CBoN 安全劫持率从 $N=1$ 的 **9.1%** 升至 $N=256$ 的 **13.4%**。
  - 分解显示：$N=1$ 时几乎全为"仅暴露"项；$N=256$ 时"仅暴露"项降至 1.6%，而"竞争胜出"项升至 **11.7%**，证实放大机制是主导因素。
  - 安全感知奖励（gpt-5-mini 评分）随 N 增大而下降，而代理奖励持续上升，说明 CBoN 优化了错误目标。
  - **奖励代理消融**：将 Beaver 替换为 Skywork-Reward-V2-Llama-3.1-8B 后，$N=256$ 时 CBoN 劫持率从 13.4% **降至 6.8%**，竞争胜出项从 11.7% 降至 5.3%，说明奖励代理的排序行为决定了放大程度。
  - cPes 在全部 N 下均将劫持率控制在接近随机可行选择的污染基线水平。

## 相关工作脉络
1. **Reward overoptimization（Skalse et al., 2022; Gao et al., 2023）**：已有工作指出对不完美的奖励代理过度优化会降低真实奖励；本文的设置在奖励误配基础上叠加了约束域本身的误配（安全过滤器假阳性），并分析了两个误差源的交互放大效应。
2. **Inference-time pessimism / coverage control（Huang et al., 2025）**：该工作研究了 BoN 的覆盖与 scaling 并提供悲观替代方案；本文区分了"污染进入"和"下游放大"两个独立风险，将覆盖控制应用于条件于代理可行集的分布，而非原始参考分布。
3. **Regulated BoN with min-Bayes-risk（Jinnai et al., 2025）**：已有正则化 BoN 研究；本文的 cPes 同样基于正则化框架，但目标是从 χ² 覆盖控制的角度保证 N 无关的安全上界，而非最小化贝叶斯风险。
4. **Learned safety filters（Llama Guard, ShieldGemma）**：安全过滤器作为操作化约束已是行业标配，但本文揭示了它们的假阳性与下游奖励优化的交互效应——这是此前未被量化的风险来源。
5. **BoN jailbreaking（Hughes et al., 2024）**：该工作展示随机搜索如何提高攻击成功率；本文保持 prompt 固定而搜索模型输出，聚焦于"过滤器允许进入但奖励排名放大"的特定失效路径，而非随机尝试的暴力搜索。

## 局限性与未来方向
1. **非自适应固定管线假设**：理论分析针对 i.i.d. 采样-过滤-排序的固定流程，不适用于 beam search、Monte Carlo tree search、自我精炼（self-refinement）或 agent 规划等自适应搜索过程，后者的迭代反馈可能产生更强的放大效应，但本文未建立量化分析。
2. **cPes 不能消除污染**：覆盖控制仅限制放大，无法修复安全过滤器已引入的假阳性；在某些奖励代理配置下（如 Skywork ablation），cPes 的有限 N 表现可能不如 CBoN。
3. **离散/可数输出空间**：理论推导基于有限或可数 Y，连续空间的测度论推广未详细讨论。
4. **实验范围**：LLM 实验仅涉及 7B 级模型和单一参考策略，高自由度或更复杂的多轮场景下的外推尚不明确。

## 研究启发与可借鉴点
1. **污染-放大二阶段分析框架**：将推理时安全的失效机制拆分为"过滤器假阳性污染可行集"和"奖励优化放大污染"两个独立步骤，可作为分析其他"过滤器+排名"系统安全性的通用范式。
2. **有限 N 联合尾部边界（Theorem 4.1）的实验诊断价值**：通过估计 $N\widehat{\Psi}_A(\cdot)$ 和 $N\widehat{\Psi}_B(\cdot)$ 的交叉顺序，可在实际系统中提前检测是否存在 compute-amplified 安全风险，为安全监控提供可操作指标。
3. **覆盖控制（χ² 散度约束）作为推理时安全的正则化工具**：cPes 的指数族重加权形式（ReLU 截断型）可直接移植到其他基于 BoN 的安全对齐管线中，作为对激进最大化行为的稳定化补充。
4. **奖励代理消融实验设计**：通过固定候选池与安全过滤器标签、仅替换奖励模型来隔离"排名排序"对安全劫持的影响，这一实验设计简洁有力地揭示了放大机制的核心驱动力，值得在类似研究中复用。

## 关键术语表
**Safety hacking（安全劫持）**：推理时选择了一个通过学习安全代理检验（$\hat{g}\geq b$）但违反真实安全标准（$g^\star=0$）的输出。
**Constrained Best-of-N（CBoN）**：从 N 个 i.i.d. 候选中筛选出通过安全代理的输出，再从中选取学习奖励最高的一个。
**Proxy-feasible set（代理可行集）**：由学习安全代理 $\hat{g}$ 定义的输出集合 $\{y:\hat{g}(x,y)\geq b\}$，可能包含真实不安全的假阳性。
**Joint reward tail（联合奖励上尾）**：$\widehat{\Psi}_\diamond(t)=\mathbb{P}[y\in\diamond,\hat{r}(x,y)>t]$，描述参考分布下某类输出的奖励得分超过阈值 $t$ 的概率质量。
**Coverage coefficient（覆盖系数）**：$C_\pi^\sharp=1+\chi^2(\pi\|\pi_\mathrm{ref}^\sharp)$，度量推理策略相对于代理可行参考分布的集中程度。
**cPes（constrained pessimistic sampling，约束悲观采样）**：基于 $\chi^2$ 正则化的推理方法，通过截断奖励和指数族重加权控制对代理可行集的过度集中。
**Unsafe-only exposure（仅暴露）**：可行集中不存在安全候选、不得不从不安全候选中选择的情况。
**Competitive unsafe win（竞争性不安全胜出）**：安全与不安全候选均存在时，不安全候选的学习奖励更高从而被选中的情况。

## 可复现要素
- **数据集**：JailbreakBench、HarmBench、AdvBench；论文未声明原始数据公开状态，但 benchmark 均为开源数据集。
- **代码/权重**：生成的参考模型（Qwen2.5-7B-Instruct）、安全过滤器（Llama-Guard-3-8B）、奖励模型（beaver-7b-v1.0-reward、Skywork-Reward-V2）均通过 Hugging Face 开源；论文附录提供了 Algorithm 1（cPes 伪代码）和 A.15/A.3 节实验细节，但未提及项目级代码仓库链接。
- **关键超参**：安全阈值 $b=0.95$；cPes 正则化系数 $\beta=1.0$；奖励截断范围 $[\widehat{R}_\min,\widehat{R}_\max]=[0,4]$；采样参数 temperature=1.0、top-p=0.95、512 tokens；N 取值 $\{1,2,4,8,16,32,64,128,256\}$。
