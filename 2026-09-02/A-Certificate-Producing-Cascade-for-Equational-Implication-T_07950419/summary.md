---
title: "A-Certificate-Producing-Cascade-for-Equational-Implication-T"
source: https://arxiv.org/pdf/2609.00706v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 20:52:18"
field: "自动化定理证明与形式化验证"
keywords: ["等式理论", "Lean 4 证书", "有序超叠加", "magma 拟群", "形式验证", "定理证明", "SAIR 挑战"]
innovations: ["将结构化反例族、不规则有限模型搜索与证明生成型超叠加集成于单文件最廉价优先级联", "正形式 Lean 4 重放策略将搜索错误隔离在可信基之外", "SHA-256 哈希绑定账本实现 solver/结果/judge 版本的端到端可追溯性"]
benchmarks: ["SAIR EQT2 公开六集（1889行）", "Stage 1 分布 drill（800行）", "Marathon normal_100", "stage2_stress_test（200行）", "research_order5_hard（100行）"]
---

# 论文速读：A-Certificate-Producing-Cascade-for-Equational-Implication-T

## 一句话总结
本文提出了一个 189,504 字节的单文件 Python 求解器，用于 SAIR 数学蒸馏挑战赛第 2 阶段（EQT2），对每一个" magma 等式是否蕴含另一等式"的问题输出 Lean 4 可验证证书（true 为形式化证明，false 为反例），在全部 1,889 行公开评测集上均获得官方 judge 接受。

## 研究问题与动机
- **等式蕴含判定问题**：给定两个 magma（二元运算，无结合律/单位元/消去律假设）等式 $E_1, E_2$，判断每一满足 $E_1$ 的 magma 是否也满足 $E_2$；SAIR 挑战额外要求对 true/false 两种判决都产出 Lean 4 格式的证书。
- **已有分类器缺乏审计性**：纯语言模型或启发式分类器虽可给出正确标签，但无法产生经内核检查的可审计推导，不符合挑战赛契约。
- **证书生产需要结构保留**：证书必须能被确定性 Lean 4 judge 编译和依赖检查，搜索阶段可以激进但不允许产生非法证书被接受——可信基仅为 Lean 4 内核。
- **公开集与隐藏集割裂**：作者刻意声明本结果仅覆盖已发布数据集，不对隐藏评测集做任何推断，体现诚实边界。

## 核心贡献（创新点）
- **集成式最廉价优先级联架构**：将结构化代数反例族（系数匹配/多项式/向量线性）、不规则有限模型搜索、显式无限载体见证和证明生成型有序超叠加引擎整合进单一求解器文件，与以往"分类器+独立证明器"的两段式方案本质不同。
- **满足 judge 依赖策略的证书编码**：针对大载体的 Cayley 表无法使用 `finOpTable`（仅支持单数字符）的问题，设计了 `submission.op` 命名空间辅助定义路线，以及 Austin 无限载体系列的正形式核心引理重写，避免了宽依赖 tactic import。
- **发现并修正两处实质性实现缺陷**：demodulation 后未删除 tautology 导致关键证明被掩盖（hard3_0314，修复后 14.6s 出证书）；未统一星号(*)与菱形(⋄)两种合法运算符号导致 Stage 1 评估集 0/800 接受——只有端到端证书运行能暴露这两类错误。
- **构建基于 SHA-256 的可追溯证据账本**：solver 字节、judge 修订版、每个结果文件和评估行均绑定哈希，防篡改且可复现。

## 方法详解
- **级联结构（Solo 模式）**：Stage A（浅层反例：常数/线性/仿射/向量线性/多项式操作、Austin 无限族）→ Stage B（直接 true 证明：结构简化+八秒 G3 预算）→ Stage C（不规则有限模型，载体 4–10，8s）→ Stage D（深层代数族+120s 有限模型）→ Stage E（ anytime 深度加深超叠加，最长 600s）。每步独立验证证书后再决定继续与否。
- **系数匹配（Section 3.1）**：对 $x \diamond y = ax + by \pmod n$，将项的每个叶节点映射到其系数向量，通过递归公式 $\vec{c}(t_1 \diamond t_2) = a\vec{c}(t_1) + b\vec{c}(t_2)$ 快速比较两侧。向量线性族用矩阵替代标量；多项式族在小载体上枚举完整赋值。
- **不规则有限模型搜索（Section 3.2）**：把 Cayley 表视为约束满足问题，使用 MRV 启发式，传播假设实例和拉丁方阵约束，遇到违反立即回溯。
- **中心群胚显式见证（Section 3.3）**：对 E168 系列（假设是中心群胚律），因自然构造均同时满足假设与目标，无法分离，作者直接内嵌一个 order-9 的非自然中央群胚表作为固定反例族。
- **无限载体见证（Section 3.4）**：对 Austin 对，选取 $\mathbb{N}$ 上固定运算，以正形式核心引理序列证明假设，并用具体元组否定目标。
- **有序超叠加引擎（Section 4）**：KBO 排序，单位子句定向，正向/反向 demodulation，鉴别树索引，记忆化合一，被动集只存储"配方"（源方程、目标方程、方向、重叠位置）而非完整项；任意时间尺寸加深。
- **Tautology 删除 bug（Section 4.2）**：demodulation 将方程改写为 $t=t$ 时，原实现未将其从 active 集移除，致使大量冗余 tautology 占据 given-clause 选择权，掩盖有效推导；修复为"先 retire 再丢弃"。
- **Lean 4 重放（Section 4.5）**：将搜索产生的 DAG 拓扑序重放为正形式 `have/rw/exact` 链，不导入饱和数据结构或 KBO 比较，仅包含最终推导所需的等式，通过 `congrArg`+`symmetry`+`transitivity` 组合。

## 实验与结果
- **数据集**：六个公开集合（sample_20/200, hard1/200/400, normal/1000）合计 1,889 行；四个 Stage 1 分布 drill 集共 800 行；Marathon manifest（normal_100，100 行）；hosted playground（200 行）。另有 stage2_stress_test（200/200 通过）和研究级 research_order5_hard（100 题全部未解）。
- **基线对比**：作者明确不做与其他求解器的性能/覆盖对比；v1 baseline 在 1,889 行中遗留 14 个残差（11 true 无证明，3 false 无反例）。
- **主要结果**：
  - sample_20：20/20 接受，wall 67.53s
  - sample_200：200/200 接受，wall 964.12s
  - hard1：69/69 接受，wall 559.81s
  - hard2：200/200 接受，wall 1250.67s
  - hard3：400/400 接受，wall 2857.38s
  - normal：1000/1000 接受，wall 4128.30s
  - 四个 drill 集：各 200/200 接受，零 LLM 调用
  - Marathon normal_100：100/100 接受，零 tokens
  - Hosted playground：200/200 接受，mean 4.89s/题
  - **最强结果**：全部 1,889 行公开集 + 800 drill + 100 Marathon + 200 playground，共 3,089 行全部接受；stage2_stress_test 200/200 接受
- **LLM 回退测量**：在 6 个 sample_20 残差上用 gpt-oss-120b 解得 0/6，成本约 $0.135
- **G3 独立评测**：254 个 Vampire 难例在 30s/题上限内全部得证（max 2.7s，sum 16s）；819 个公开 true 题 median 1ms、max 1.2s

## 相关工作脉络
- **Equational Theories Project [3]**：提供等式蕴含图和形式化证明，本文从该项目获取等式与问题数据，但增加在线证书接口。
- **Cazares [4]**：分析同一挑战 Stage 1（纯分类任务），本文 Stage 2 改变输出契约为 judge-accepted 证明/反例。
- **Vampire [9] / E [13] / Prover9 [10]**：成熟一阶定理 prover，实现超叠加/完备化，本文借鉴有序推理/KBO/demodulation 思想但专为单二元符号单位子句定制，并以 Lean 4 证书输出为系统差异。
- **Knuth 中心群胚研究 [7]**：本文仅在 E168 残差中嵌入一个 order-9 显式反例，非一般性结构理论。
- **Lean 4 + Mathlib [5][14]**：作为证书语言和最终可信检查器；本文策略是正形式核心证明替代宽 tactic 依赖。
- **Austin [1]**：无限载体反例构造的来源；本文沿用其 N 上运算。

## 局限性与未来方向
- **无完备性保证**：两套分支（反例和证明）均未证明完备；Stage D/E 均受 wall-clock 限制。
- **对隐藏评测集无推断**：公开集 tuning 可能过拟合，作者明确不作 leaderboard 宣称。
- **研究前沿完全未触及**：research_order5_hard（100 题，至少一边为 order-5 Austin 律，false 方向无可达有限反例）全部未解，显示无限深反例搜索仍为开放难题。
- **复现边界受限**：Lean 4 4.32 兼容性语料在外部目录运行，未打包为单命令仓库测试；hosted 交互与未来 leaderboard 需主办方基础设施。
- **资源依赖单进程**：所有缓存和搜索状态与进程绑定，暂未实现跨题 lemma 共享。

## 研究启发与可借鉴点
- **端到端证书检查暴露隐蔽 bug**：tautology 删除和输入符号规范化两类缺陷仅通过完整 Lean 4 提交才能发现，提示任何"输出验证"系统应以最终可信检查器为唯一裁判。
- **正形式重放（positive replay）分离搜索与信任**：搜索错误最多导致证书被拒或延迟，不会导致错误接受；这一设计原则可迁移至其他需形式验证的自动化系统。
- **级联"最廉价优先"调度策略**：先做低成本高命中率阶段，再逐步加深昂贵阶段，适用于具有多种互补启发式的证明/搜索任务。
- **SHA-256 绑定账本机制**：将 solver 字节、judge 版本、结果文件全部哈希绑定，为可复现科研报告提供工程化范式。
- **与 LLM 结合的纪律**：LLM 仅作为确定性级联失败后的最后手段，且不在 Marathon 中启用以保护全局 token 预算；这种分层调用策略值得在大型混合系统中推广。

## 关键术语表
- **Magma（拟群）**：仅具一个二元运算的代数结构，无结合律/单位元/消去律等额外公理。
- **Equational Implication（等式蕴含）**：给定等式 $E_1, E_2$，判断每个满足 $E_1$ 的 magma 是否自动满足 $E_2$。
- **Certificate（证书）**：对 true 判决为 Lean 4 证明项，对 false 判决为分离反例的 magma，经内核检查后接受。
- **Ordered Superposition（有序超叠加）**：基于项排序的等式推理规则，将等式一侧重叠到另一侧的子项上，是完备化与超叠加 prover 的核心推理。
- **KBO（Knuth–Bendix Ordering）**：一种用于项排序的递归良基序，本文据此确定超叠加可 orient 的方向。
- **Demodulation（去模化/重写归约）**：用已有等式作为 rewrite rule 化简其他项的操作，包括正向（化简新项）和反向（化简 active 集）。
- **Anytime Deepening（ anytime 加深）**：逐步扩大项尺寸和变量上界，将上一轮剩余时间转入下一轮，直至 deadline。
- **Marathon / Solo 协议**：Solo 每行独立进程；Marathon 单一进程按全局预算分发时间处理问题清单。

## 可复现要素
- **数据集**：六个公开集合（sample_20, sample_200, hard1, hard2, hard3, normal）、四个 Stage 1 drill 集、Marathon normal_100 manifest、stage2_stress_test、research_order5_hard；均来自 SAIR 官方仓库，论文引用 `https://github.com/SAIRcompetition/equational-theories-lean-stage2`
- **代码/权重**：solver v2.5 为 189,504 字节单文件 Python，SHA-256 `f2392533c9f4c03b...`；提供 `scripts/check_freeze.py`、`scripts/run_marathon.py`；Lean 4 judge 官方仓库
- **关键超参**：Stage B G3 预算 `min(6, max(1, T/600))` 秒；Stage C 载体 4–10，8s；Stage D 多项式/向量族 60s，有限模型 120s；Stage E G3 预算 `min(600, max(20, T/6))` 秒；Marathon 最多 55% 残差预算用于 false 分支
- **硬件环境**：官方 judge revision 2848228ff490...；Lean 4 4.30.0-rc2 / 4.32.0（兼容性语料）；论文未提及具体 CPU/GPU 型号，wall-clock 为单进程累加值
