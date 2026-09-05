---
title: "A-Certificate-Producing-Cascade-for-Equational-Implication-T"
source: https://arxiv.org/pdf/2609.00706v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 20:52:15"
field: "符号定理证明与等式推理"
keywords: ["等式理论", "定理证明", "证书生成", "有序超叠", "Lean 4", "反例搜索", "SAIR 挑战"]
innovations: [" cheapest-first 单文件级联求解器，将结构化/不规则反例与 proof-producing superposition 统一于 judge-accepted 证书契约下", "KBO ordered unit superposition 配合 tautology 删除修复与 anytime deepening，并将推导重放为 Lean 4 正形式核心项", "将 broad tactic 依赖的 Austin 无限载体证书重写为 core-only 引理序列，使编译时间从 330s 降至 3.1s"]
benchmarks: ["six public sets (1,889 rows)", "Stage 1 distribution drill (4×200)", "Marathon normal_100", "hosted playground evaluation_normal", "stage2_stress_test (200)", "research_order5_hard (100)"]
---

# 论文速读：A-Certificate-Producing-Cascade-for-Equational-Implication-T

## 一句话总结
本文提出了一个单次文件（189,504字节 Python）的确定性级联求解器，用于 SAIR 等式理论 Distillation Challenge Stage 2 的"等式蕴含判定+证书生成"任务；该求解器在六个公开数据集（共 1,889 行）上全部产生被 Lean 4 judge 接受的证书，全程零 LLM 调用。

## 研究问题与动机
- **核心问题**：给定两个 magma 等式 $E_1$ 和 $E_2$，判定"每个满足 $E_1$ 的 magma 是否都满足 $E_2$"，并对真假两个判决都返回可被确定性 Lean 4 judge 接受的形式证书（true=Lean 证明，false=反例模型）。
- **现有方法为何不足**：传统分类器可以"有用但不可审计"，而比赛契约要求输出的每一步都必须能在 kernel 边界内被检查；仅靠 label 精度不足，错误搜索启发式可能导致无效证书通过。
- **设计目标转变**：从"输出正确标签"转向"输出结构可重建的证书"，允许搜索阶段是 untrusted 的，但 accepted 证书必须由 judge 的最终类型检查/依赖策略决定。
- **挑战特殊性**：等式理论含无结合律/单位元/消去律的二元运算，既存在大量有限反例，也存在只在无限载体上成立的假蕴含（如 Austin 对），并要求证书兼容 Lean 4 的 allow-list 与核心项重写，而非引入广域 tactic 依赖。

## 核心贡献（创新点）
1. **提出 cheapest-first 单文件级联架构**，将结构化代数反例族、有界不规则有限模型搜索、显式 central-groupoid  witness、无限载体反例与生产 proof ordered unit superposition 统一于同一求解器背后，区别在于以"证书成本最小化"而非"极性先验"排序阶段。
2. **设计满足 judge 依赖策略的证书编码**，包括超出 `finOpTable` 单 digit 范围的较大载体的计算型操作定义与 core-only 无限载体证明，将 broad tactic import 替换为显式核心引理序列，使 Lean 4 编译从约 330 秒降至 3.1 秒。
3. **识别并修复两个关键实现故障**：active 集中保留 tautology 导致 hard3_0314 等真实残差无法证明（修复后 0.6 秒内找到、官方 runner 14.6 秒接受证书），以及操作符星号/菱形双写法的输入规范化问题（修复后 800/800 全接受），这些仅能通过端到端证书运行暴露。
4. **提供 hash-bound 可审计证据链**，将求解器字节、judge revision、结果文件与每个评测行绑定到 SHA-256，防止事后替换或 stale measurement 复用。

## 方法详解
- **整体架构**：单文件 `solver.py`，支持 Solo（每行新进程+固定预算）与 Marathon（多行共享全局预算、追加输出）两种协议；输入先在入口将 `*` 规范化为 `⋄`，保证内部词法唯一。
- **级联五阶段（按预期成本排序）**：
  - **Stage A 浅层反例**：暴力小表、标量线性/仿射、向量线性（小有限域上 2D/3D）、低次多项式、已知 Austin 无限载体模型；True 输入额外耗时约 0.8s。
  - **Stage B 直接真证书**：语法单点折叠与代入实例检查；专用 singleton-forced prover（8s cap）；一般 superposition 引擎配 quick budget `min(6, max(1, T/600))` 秒。
  - **Stage C 有界不规则模型**：载体 4–10，8 秒；传播假设的全称实例与 Latin square 约束，而非枚举完整表。
  - **Stage D 深层假搜索**：重多项式/向量族（60s cap）+ 另一轮有界有限模型（120s cap）；优先放在深证明之前以防困难假行吞噬大部分预算。
  - **Stage E  anytime 证明搜索**：ordered superposition，budget `min(600, max(20, T/6))` 秒；反复提升 term-size 与变量 cap；失败时以小 pool 派生等式作为 exact lemmas + kernel-checked closing tactic。LLM 反馈回路存在于 fallback 中，但已归档接受行未调用。
- **证书与信任边界**：所有搜索产出均为 candidate；每一阶段独立验证后提交 judge，拒绝则继续下一策略；不接受缓存未经验证的布尔判决。Python 侧检查用于提前淘汰畸形候选，judge 决定最终 acceptance。
- **反例分支关键设计**：
  - **系数匹配**：对 $x \diamond y = ax + by \pmod n$ 将项表示为 formal coefficient vector，节点处递归合并 $a\vec{v}_L + b\vec{v}_R$；假设与目标向量不同即分离，避免全赋值枚举。向量线性用矩阵替代标量；多项式在可行时用编译评估器。
  - **不规则有限模型**：Cayley 表作为约束问题，MRV 选择 + watch list 传播；由出现模式推断 Latin 约束（行/列置换）、拒绝重复值、强制裸单点；完成后再做全量重评估。
  - **显式 central-groupoid witness**：针对 E168 家族 12 个残差，嵌入一个 order-9 非自然 central groupoid 固定表；假设经变量重命名匹配且目标被分离即发出。
  - **无限载体 Austin 对**：选固定 $\mathbb{N}$ 上运算及其对偶，附固定引理序列证明假设，再用具体 tuple 否定目标。
  - **证书形状**：≤10 载体用 `finOpTable` + `decideFin!`；更大载体用 `submission.op` 命名空间内计算型 helper（表查找/算术），规避 `finOpTable` 单 digit 限制；judge allow-list 控制依赖，helper 本体不在限制内。
- **证明分支关键设计**：
  - **有序 unit superposition**：KBO 定序、正向/反向 demodulation、discrimination-tree 索引按 linear skeleton 检索、统一器返回三角 substitution 并缓存、重叠前基于静态大小估算拒绝超限。
  - **Tautology 删除修复**：demodulation 产生 $t=t$ 时必须 retire 主动子句并丢弃新等式；否则小实例持续占据 weight heap，阻塞有效推导。
  - **Anytime deepening**：多轮递增 term-size 与变量 cap；饱和则转移剩余时间；后期降低变量 penalty、增大 age pressure，近似另一配置而不需新实现。
  - **Lean 4 正形式重放**：保留 DAG 推理 recipe；重叠步用 `congrArg` lift + `trans` 组合；每个 selected inference 一条 `have`；共享祖先拓扑序发出一次；robust re-emission 仅在 elaboration 歧义时加局部类型信息，不改推导。

## 实验与结果
- **数据集**：六个公开集（`sample_20`、`sample_200`、`hard1`、`hard2`、`hard3`、`normal`）共 1,889 行；四个 Stage 1 分布 drill 各 200 行；Marathon `normal_100` 100 行；hosted playground `evaluation_normal` 200 行；`stage2_stress_test` 200 行；`research_order5_hard` 100 行。
- **主要结果**：
  - 六个公开集：**1,889/1,889 accepted**，LLM 调用 0，每行 1 次 judge 调用（Table 1）。
  - 分布 drill 四个集合：**各 200/200 accepted**，零 LLM（Table 2）。
  - Marathon `normal_100`：**100/100 accepted**，零 tokens。
  - Hosted playground `evaluation_normal`：**200/200 accepted**，mean 4.89s/problem。
  - `stage2_stress_test`：**200/200 accepted**，零 LLM，类别分布与公告一致。
  - `research_order5_hard`（至少一边为 order-5 Austin 律、假方向无可构造有限反例）：**0/100 accepted**，界定能力边界。
  - G3 聚焦测量：254 个含 Vampire 衍生的 hard ETP 对在 30s cap 下全部找到；819 个公开真问题 median 1ms、max 1.2s。
  - Austin 证书迁移后：120 份跨 family 证书在 Lean 4.32 下 119/120 通过；原 broad-import 版本 330s → 重写后 3.1s。
- **最强结果与提升**：v1 遗留 14 个残差（11 真+3 假），v2 全部解决；硬3_0314 由 810s 无解到 0.6s 找到/14.6s 接受；输入双写规范化修复使 800/800 从 0 跃升至全接受。
- **时间统计**：六公开集 wall 合计约 9,828s；hosted 假行 mean 2.69s/max 6.5s、真行 mean 7.08s/max 23.8s。

## 相关工作脉络
- **Equational Theories Project**（Bolan et al., 2025）：提供等式蕴含图与形式化数据；本文继承其定律与问题构造，但契约从"标签预测"升级为"judge 接受的证明/反例证书"。
- **Cazares (2026)**：分析本挑战 Stage 1 的预测设定；Stage 2 输出空间完全不同（可审计证书 vs. 标签）。
- **Vampire / E / Prover9**：成熟的一阶定理 prover，实现 resolution/superposition/completion；本文借用 KBO、demodulation、discrimination-tree、given-clause 等标准思想，但特化至单二元符号 unit 等式，并以正形式 Lean 4 证书为输出契约，不作吞吐量对比。
- **Knuth (1970) Central Groupoids**：为 E168 残差家族提供组合结构背景；本文仅嵌入一个经校验的非自然 order-9 反例，不做一般性分类。
- **Austin (1965)**：无限载体模型的来源；本文沿用其构造并改写为 core-only 正形式证书以降低依赖。
- **Lean 4 + mathlib**：judge 的 kernel 与库环境；本文强调避免 broad tactic import，采用 local named facts 串接以通过依赖检查。

## 局限性与未来方向
- **无完备性定理**：结构化反例族、有界有限模型搜索与 superposition 深度均在时间与载体预算内，不能推广到隐藏集。
- **公开数据集可能引导开发**：14 个 v1 残差与 E168 drill 直接驱动了特定模块；回归覆盖不代表跨分布泛化。
- **Wall-clock 不可移植**：计时融合 Python 搜索、进程开销与 Lean 4 编译，受主机负载与文件系统缓存影响；hosted playground 均值部分缓解但仍非最终评分协议。
- **LLM fallback 仅做负向测量**：单模型/单配置在 6 个 sample_20 残差上 0/6 解决，成本 $0.135；不足以判定 LLM 在本任务的普遍效能。
- **复现边界**： reproducibility 需同时签出本文 artifact 与官方 judge pinned revision；Lean 4.32 兼容语料在外部目录运行，未打包为单命令测试。

## 研究启发与可借鉴点
- **"搜索 untrusted + 证书 trusted"分离**：用独立 polarized candidate + judge 最终验收取代缓存 label，可广泛迁移至需要可审计输出的自动化定理证明/反例搜索任务。
- **Cheapest-first 级联按"证书成本"而非"极性先验"排序**：浅层结构化反例与直接真证书优先，既降平均成本又为两侧保留时间；适合多策略竞争、预算严格的竞赛/托管环境。
- **正形式重放优于 tactic 打包**：将推导 DAG 编译为逐部 `have/rw/exact` 链，既压缩证书体积又避免广域库依赖；对需要通过与严格 allow-list 的 kernel 检查的场景极具参考性。
- **Tautology 删除这种"微观策略"可决定宏观求解**：一处 demodulation 后的 active 集维护缺陷即可使整个推导失败；提示在 saturation 类引擎中 simplification 与 selection 的耦合必须严格验证。
- **Hash-bound claim ledger 提升可审计性**：将求解器字节、judge revision、结果文件与每行绑定 SHA-256，可有效防止 stale measurement 复用，适合竞赛/基准评测的可重复性规范。

## 关键术语表
- **Magma**：仅含一个二元运算、无结合律/单位元/消去律假设的代数结构。
- **Equational implication**：判断"等式 $E_1$ 是否逻辑蕴含 $E_2$"，即所有满足 $E_1$ 的 magma 是否都满足 $E_2$。
- **Certificate-producing solver**：输出不仅含判决还要含可被 kernel/judge 验收的形式对象（证明项或分离反例）。
- **Ordered unit superposition**：在 unit 等式设定下基于排序的超叠推理，配合 demodulation 与 given-clause 调度进行饱和搜索。
- **Knuth–Bendix ordering (KBO)**：用于定向等式侧与选择最大项的项序，是有序超叠的基础。
- **Central groupoid**：满足特定恒等式的二元运算结构；本文嵌入一个 order-9 的非自然实例以分离 E168 族目标。
- **Austin pair**：在已搜索有限载体上为真、但在一般（如 $\mathbb{N}$）载体上为假的等式对。
- **Solo / Marathon 协议**：Solo 每行独立进程与预算；Marathon 多行共享全局预算并以追加 JSONL 输出。

## 可复现要素
- **数据集**：六个公开集、四个 Stage 1 distribution drill、Marathon `normal_100`、hosted playground、`stage2_stress_test` 与 `research_order5_hard`；论文以官方仓库与 PROVENANCE.json 形式提供。
- **代码/权重**：求解器为单文件 Python 源码（189,504 bytes，SHA-256 `f2392533c9f4c03b...`），随仓库冻结；judge 需签出官方 Stage 2 仓库的 pinned revision。
- **关键超参**：各级 budget 公式——Stage B quick superposition `min(6, max(1, T/600))`s，Stage E anytime `min(600, max(20, T/6))`s；有限模型载体区间 4–10；G3 聚焦测量 cap 30s。
- **复现步骤**：`python3 scripts/check_freeze.py` → 签出官方 judge → `bash scripts/setup.sh` + `source .env.judge` → Solo/ Marathon runner 命令（论文第 8.4 节）。
- **未提及**：GPU/硬件需求无特定要求（纯 CPU Python + Lean 4 编译）；外部网络无需。
