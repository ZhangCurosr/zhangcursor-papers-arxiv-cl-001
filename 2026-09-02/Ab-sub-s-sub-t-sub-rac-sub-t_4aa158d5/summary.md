---
title: "Ab-sub-s-sub-t-sub-rac-sub-t"
source: https://arxiv.org/pdf/2609.01274v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 20:53:24"
field: "大语言模型推理与后训练行为分析"
keywords: ["RLVR", "Unified Decoding Framework", "SearchLens", "BOPTR", "inference-time search", "behavioral analysis", " SimpleRL-Zoo", "transfer learning"]
innovations: ["提出SearchLens/UDF统一解码框架，将token级采样/束搜索/树搜索/序列重采样表示为共享预算操作空间中的可执行策略，并分离策略与指标", "提出BOPTR-P1低维预算转移规律 N_Base ≈ α N_RL^β，以基准机制（math/nonmath/floor）决定指数β、1D模型偏移吸收跨模型差异，在4基准×7模型上实现3.41 pp水平转移误差", "构建RL信息前沿（Z0→Z1→V3→V1），证明预测后RL行为所需RL信息上限约为1个锚点单元，零RL base@rltrain先验仅差~1 pp"]
benchmarks: ["Math500", "AIME 2024/2025", "GPQA-Diamond", "IFEval", "AMC23", "Minerva", "OlympiadBench", "TinyMMLU"]
---

# 论文速读：From Base Rollouts to RL Reasoning: A Budgeted Search Perspective

## 一句话总结
本文通过构建统一解码框架（SearchLens/UDF），行为性地检验了"RL提升推理能力是否等同于将基座模型的采样效率推向其已可到达的操作点"，发现SimpleRL-Zoo配方下的大部分RL增益对应于一个**基准条件化的预算转移规律（BOPTR）**——即RL默认策略曲线可由Base模型在外部搜索预算下的低维路径近似，误差仅3.41 pp；同时揭示了该规律的跨基准、跨模型可迁移性及失败边界。

## 研究问题与动机
1. **核心问题**：RL with Verifiable Rewards (RLVR) 到底产生了基座模型原本不具备的推理行为，还是仅仅将默认rollout分布向基座模型已能到达但极少采样的轨迹偏移？
2. **现有方法不足**：同一策略对比（same-policy）仅在固定解码策略和预算下比较Base与RL，会系统性低估二者关系——因为它隐藏了Base+UDF中已有的匹配操作点；同时大规模策略网格上的逐预算匹配极易过拟合为事后匹配（post-hoc match），缺乏可迁移的结构。
3. **挑战**：解码与搜索方法异质（token级采样、束搜索、树搜索、序列级重采样控制结构不同）；恢复若缺乏低复杂度路径约束会被高估；跨基准、跨模型、跨训练分布的转移是否可行尚未系统检验。
4. **动机来源**：先前工作（Yue et al., 2025; Zhao et al., 2025; Karan & Du, 2025）提示RL增益可能不反映全新推理能力，但"预算作为显式轴"这一视角的系统化框架仍缺失。

## 核心贡献（创新点）
1. **SearchLens/UDF统一解码框架**：将token级采样、束状扩展、树状探索、序列级重采样表示为共享预算操作空间中的可执行策略，且评估指标（pass@k/SC/BoN/FFS）与生成策略分离，避免了生成-评估耦合偏差。
2. **二维恢复分析**：不再仅问"Base在相同策略和预算下能否达到RL"，而是问"RL默认策略曲线是否嵌入Base模型更宽广的策略-预算景观之中"，并以near-match集合+低复杂度路径来区分结构化恢复与事后单点匹配。
3. **BOPTR-P1（SearchPath）低维转移规律**：提出 $N_{\text{Base}} \approx \alpha N_{\text{RL}}^{\beta}$ 的预算转移规则，$\beta$ 由基准机制（regime）决定、模型差异吸收进1D偏移 $\delta_m$，在Math500/AIME/GPQA/IFEval上得到3个机制分类。
4. **水平与垂直转移的系统检验**：在同一Hard RL训练分组内直接锚点路径转移误差低至1.40–3.73 pp；跨family/小容量模型误差上升但1D偏移可将Llama误差从10.16降至3.68 pp；不加任何RL信息仅用base@rltrain特征仍可达5.08 pp，揭示预测后RL行为所需RL信息上限约为1个锚点单元。
5. **公开代码与分析脚本**（https://github.com/HALIS-sh/Searchlens_boptr），含UDF等价性验证（11种主流解码算法与vLLM/HF/MCMC参考实现一致）、3种子复刻（$3.07 \pm 0.39$ pp）、预算扩展至 $b=64/256$ 等完整附录。

## 方法详解
- **UDF操作点定义**：$u = (\pi, n)$，其中 $\pi \in \Pi_{\text{UDF}}$ 为策略配置（5个组件：local policy / controller / evaluator / transition / scheduler），$n \in \mathcal{B} = \{1,2,4,8,16\}$ 为推理预算。指标仅在生成后应用，不构成策略本身。
- **策略几何**：每个操作点映射到行为签名 $z(u) = [P(u), Q(u), T(u), D(u), C(u)]$（pass@k / SC / first-finish、多样性、成本），距离函数 $d(u_i, u_j) = \lambda_{comp} d_{comp} + \lambda_z \|z_i - z_j\|_1 + \lambda_b |\log n_i - \log n_j|$ 用于路径诊断。
- **Same-policy增益**：$\Delta_{\text{same}}^c(b) = A_{\text{RL}}^c(\pi_0, b) - A_{\text{Base}}^c(\pi_0, b)$ 隔离训练效应与 richer 策略效应；观察到低预算RL增益、大预算Base恢复、GPQA/IFEval指标语义偏移三种模式。
- **Base+UDF包络与near-match集合**：$E_{\text{Base}}^c(b) = \max_{\pi, n \le b} A_{\text{Base}}^c(\pi, n)$ 构成支持区域；near-match集合 $N_\epsilon^c(b) = \{(\pi, n): |A_{\text{Base}}^c(\pi, n) - A_{\text{RL}}^c(\pi_0, b)| \le \epsilon\}$ 捕捉每预算的多匹配性。
- **BOPTR-P1公式**：$\widehat{N}_{m,d}^{\text{ch}}(b) = \text{round}_\mathcal{B}(\alpha_{m,d}^{\text{ch}} b^{\beta_{g(d)}^{\text{ch}}})$，$\widehat{\pi}_{m,d}^{\text{ch}} = \arg\min_\pi \mathcal{L}_{m,d}^{\text{ch}}(\pi)$，$\log \alpha_{m,d} = \mu_{g(d)} + \delta_m$。选择损失 $\mathcal{L}$ 使用 $r = [\text{rank}_P, \text{rank}_{P-Q}, \text{rank}_Q, \text{ctrl}, \log\text{cost}]$，权重 $(1,1,1,0.5,0.2)$。
- **三层机制（regime）分类**：Math500为sublinear（$\beta_\text{math}=0.60$）、GPQA/IFEval为near-linear OOD（$\beta_\text{nonmath}=1.00$）、AIME为floor-bound（$\beta_\text{floor}=0.00$）；后者为描述性标签非能力断言（与两机制反事实统计不可区分）。
- **RL信息前沿（Z0–Z1层次）**：Zero-RL仅用base@rltrain饱和度先验（5.13 pp）；One-anchor引入单一校准单元（4.08 pp）；Per-model V3用7个模型各自Math500对（4.44 pp）；V1 per-cell oracle为结构上界（2.77 pp）。

## 实验与结果
- **数据集**：Math500（500题）、AIME 2024/2025（各30题）、GPQA-Diamond（198道）、IFEval（~500 prompt）；另外4个held-out基准AMC23/Minerva/OlympiadBench/TinyMMLU用于冻结规则测试。
- **模型对**：SimpleRL-Zoo配对的Qwen2.5-0.5B/1.5B/7B/14B、Qwen2.5-Math-7B、Llama3.1-8B、Mistral-7B-v0.1；外加ORZ-7B、OAT-7B、DS-Math-7B三个外部配方。
- **主要结果**：
  - 水平转移（Qwen2.5-7B锚点→同模型其他基准）：BOPTR-P1（H5）3.41 pp，显著优于H2 base same-π（8.10 pp）、H3 anchor per-b copy（6.01 pp）、H4严格β=0.6（6.56 pp）；95% CI [2.32, 5.53]；3种子复刻 $3.07 \pm 0.39$ pp。
  - 垂直转移（同分组内7模型）：V3 (δ_m from Math500) 整体4.44 pp，V5b base-only预测偏移4.19 pp；跨family Llama从10.16降至3.68 pp；1.5B仍高达7.82–10.96 pp。
  - 外部配方与新基准：ORZ-7B 4.87 pp、OAT-7B 4.48 pp、DS-Math-7B 3.28 pp；4个冻结新基准跨7模型均值5.03 pp vs 拟合时4.44 pp；所有4基准均值均≤8 pp预注册阈值。
  - 预算扩展至 $b=64/256$：整体误差升至5.02 pp，主要源于AIME floor分配在 $b>16$ 持续固定为2造成的分配误差（AIME单点12.50 pp），非Base搜索能力天花板（base envelope在b=256已达25.0% vs RL 20.0%，置信区间重叠）。
  - UDF等价性：Tier 1-2（vLLM SamplingParams 7种策略）output match≥0.76、EM agree≥0.92；Tier 3-4（HF typical_p、HF beam、MCMC power_samp、entropy_tree）|ΔEM|≤0.08。
  - 计算开销：锚点单元约461 H100 GPU-hours（AR sweep ~9 H GPU-hours，expanded controllers ~452 H GPU-hours），~0.31B tokens。

## 相关工作脉络
1. **RLVR增益来源争论**（Yue et al., 2025; Zhao et al., 2025）：前者提示Base在大budget pass@k可接近RL，后者主张RL放大预训练分布中已有行为；本文以UDF框架将其形式化为"RL曲线是否可被Base搜索景观中的低维路径覆盖"的可检验命题。
2. **无需训练的推理采样**（Karan & Du, 2025）：证明base model抽样可匹敌甚至超越RL-posttrained模型；本文将其纳入BOPTR信息前沿（Zero-RL prior仅+1 pp代价），并将该观察解释为"采样效率提升而非新能力涌现"的行为学表述。
3. **Inference-time compute scaling laws**（Snell et al., 2024; Wu et al., 2025）：固定测试预算的分配方式可能与模型规模同等重要；本文将其操作化为UDF中$(\pi, n)$二维网格的系统扫描。
4. **延长RL训练扩展推理边界**（Liu et al., 2025, ProRL）：提示reweighting-only解释在特定训练时长下失效；本文明确承认其适用范围限定于 tested recipe 和 durations，并以AIME floor regime和1.5B弱容量cell作为边界案例。
5. **SFT vs RL**（Chu et al., 2025）："SFT memorizes, RL generalizes"；本文聚焦于通用性度量——跨基准/跨family/跨配方的转移误差如何变化。
6. **CoT faithfulness**（Turpin et al., 2023; Lanham et al., 2023）：推理轨迹未必忠实于内部计算；本文因此聚焦输入-输出可观测行为而非参数层机制识别。

## 局限性与未来方向
1. **行为而非参数层面等价性**：BOPTR是operating-point景观上的行为诊断，不声称RL参数等价于任一外部解码策略。
2. **策略池可比性问题**：部分setting仅含AR-only pool，部分含expanded controllers（MCMC/entropy tree/MCTS/thought tree），恢复性陈述应相对于各cell实际评测的UDF策略解读，不同pool间不直接比较。
3. **指标语义跨基准异质**：Math500/AIME为support任务，GPQA为concentration/sharpening任务，IFEval为termination/validity任务；跨基准数字需在此机制结构下解读。
4. **模型特异性参数限制**：1D偏移 $\delta_m$ 仍依赖cohort内其他模型做leave-one-out拟合，无法仅凭目标模型自身预测；single linear ridge at $n=6$ 完全失效（LOO pseudo-$R^2 = -0.38$）。
5. **模型覆盖范围有限**：仅一行Qwen family scale sweep + 一个cross-family + 一个weak-capacity + 一个Math-specialized； regimes 数量(3)由4个基准最小AIME样本量决定，新增基准可能需重新分类。
6. **训练时长效应未检验**：prolonged RL training是否产生base无法到达的行为（Liu et al., 2025）在当前配方和时长下未触及。
7. **未来方向**：v8-H规则族计划以base-feature adapter替代 $\delta_m$ 标量；需更大cohort或rltrain-side特征才能可靠识别。

## 研究启发与可借鉴点
1. **UDF五组件参数化是可复用的解码-搜索统一框架**：local policy/controller/evaluator/transition/scheduler的解耦设计使任意主流解码算法均可表达为操作点，且经11种算法与vLLM/HF/MCMC参考实现的严格等价验证；可移植到其它后训练（如DPO/SFT）行为分析。
2. **"同一策略增益 + Base+UDF包络 + 低复杂度路径" 三段式分析范式值得借鉴**：先隔离训练效应（5.1节），再检验支持区域包含性（5.2节），最后验证路径结构（第6节），层层递进，避免单一角度误导。
3. **RL信息前沿（Z0→Z1→V3→V1）的层次化定量**：以"需要多少RL数据才能预测后RL行为"为坐标重新框架问题，揭示1个锚点即足够（4.08 pp），零RL亦可（5.13 pp），为后续"轻量校准"工程提供量化依据。
4. **跨基准机制分类（sublinear/near-linear/floor）启发新的benchmark regime taxonomy**：不仅适用于当前4基准，亦可用于后续扩展（AMC23/Minerva/Olymp/TinyMMLU已按此先验分类且全部通过≤8 pp准则）；可将此分类框架推广到其它任务族。
5. **与团队方向的结合机会**：若本团队从事推理模型后训练评估，BOPTR可作为fast behavioral diagnostic快速定位新recipe相对于SimpleRL-Zoo的行为偏移幅度；其基座-only预测偏移思路（V5b）可直接嫁接至无RL配对的新模型评估管线。

## 关键术语表
- **RLVR（Reinforcement Learning with Verifiable Rewards）**：使用程序/规则级正确性信号而非人类偏好进行强化学习训练，代表配方如DeepSeekMath、SimpleRL-Zoo。
- **UDF（Unified Decoding Framework / SearchLens）**：将token级采样、束搜索、树搜索、序列重采样统一表示为$(\pi, n)$操作点的预算化框架，指标与策略分离。
- **Operating point $(\pi, n)$**：解码策略配置 $\pi$ 与推理预算 $n$ 的组合，表征一次有限rollout预算分配。
- **BOPTR（Budgeted Operating-Point Transition Rule）**：预算转移规律 $N_{\text{Base}} \approx \alpha N_{\text{RL}}^{\beta}$，$\beta$ 由基准机制决定、模型差异吸收进1D偏移 $\delta_m$。
- **Regime（机制分类）**：按 $\beta$ 拟合划分的基准类型——math（sublinear, 0.60）、nonmath/OOD（near-linear, 1.00）、floor（budget-insensitive, 0.00）。
- **Same-policy gain**：固定解码策略 $\pi_0$ 下RL相对Base的精度差 $\Delta_{\text{same}}(b)$，隔离训练效应与策略丰富度效应。
- **Base+UDF support region**：各预算下Base模型所有UDF操作点构成的性能包络（上限）与低分位带（下限）所围成的区域。
- **RL-information frontier（Z0–V1层次）**：从"零RL信息"到"per-cell oracle"的预测精度阶梯，刻画预测后RL行为所需RL数据的下界。

## 可复现要素
- **数据集**：Math500、AIME 2024/2025、GPQA-Diamond、IFEval、AMC23、Minerva、OlympiadBench、TinyMMLU——均为公开基准，论文未创建新train/dev/test切分。
- **代码/权重**：代码、配置和分析脚本开源（https://github.com/HALIS-sh/Searchlens_boptr）；使用SimpleRL-Zoo开源的配对Base/RL checkpoint（Qwen2.5-0.5B/1.5B/7B/14B等）及外部配方checkpoint（ORZ-7B、OAT-7B、DS-Math-7B）。
- **关键超参**：预算集 $\mathcal{B} = \{1,2,4,8,16\}$；near-match tolerance $\epsilon = 3$ pp；selector权重 $W=(1,1,1,0.5,0.2)$、$\lambda_{\text{cost}}=0.01$；机制参数 $\beta_{\text{math}}=0.60, \beta_{\text{nonmath}}=1.00, \beta_{\text{floor}}=0.00$；饱和度 knee 阈值 $\tau=0.02$；bootstrap重复 $10^4$ 次。
- **硬件**：H100 GPU，vLLM tensor-parallel=2；锚点单元约461 H100 GPU-hours。
- **论文未提及**：具体训练超参（仅使用现有SimpleRL-Zoo checkpoint）；V5b中典型性门控multi-signal predictor的详细架构。
