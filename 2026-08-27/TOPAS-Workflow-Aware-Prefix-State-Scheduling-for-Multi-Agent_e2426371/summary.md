---
title: "TOPAS-Workflow-Aware-Prefix-State-Scheduling-for-Multi-Agent"
source: https://arxiv.org/pdf/2608.25523v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 01:49:55"
field: "LLM serving 系统优化"
keywords: ["multi-agent LLM serving", "prefix caching", "request scheduling", "job completion time", "workflow-aware scheduling", "KV cache management"]
innovations: ["首个将前缀驻留与请求接纳联合优化的工作流感知在线调度器", "以任务最长剩余服务路径期望缩减为核心的JCT导向效用函数", "短视复用价值与任务老化机制联合抑制上游饿死与下游延迟"]
benchmarks: ["Chain-3", "DAG-4", "DAG-10-Wide", "MetaGPT-SOP", "MetaGPT-TL"]
---

# 论文速读：TOPAS-Workflow-Aware-Prefix-State-Scheduling-for-Multi-Agent

## 一句话总结
TOPAS是首个将前缀驻留（prefix residency）作为显式工作流级调度决策的在线调度器，联合优化GPU内存中保留哪些智能体前缀与接纳哪些就绪请求，通过JCT导向的效用函数权衡任务最长剩余服务路径的缩短与近期前缀复用的收益，从而在多智能体LLM服务中降低端到端任务完成时间。

## 研究问题与动机
- **问题核心**：在共享KV缓存预算约束下，调度器需联合决定哪些智能体静态前缀驻留GPU以及哪些就绪请求被执行，以最小化任务级JCT。
- **现有方法不足一**：主流框架（如SGLang）的请求级调度仅间接反映前缀驻留，导致异构前缀共存于缓存，压缩了动态批次容量，并可能频繁触发前缀驱逐与重载。
- **现有方法不足二**：Locality-first策略（如LPM）优先复用已有前缀，但会持续将服务滞留于工作流上游，饿死下游阶段；Progress-first策略（如SRPT）优先推进任务，但频繁切换智能体导致前缀迁移开销放大，有效服务能力下降。
- **根本矛盾**：端到端进度与前缀复用存在内在冲突——合并同智能体请求可提升批处理效率，但会阻塞上游；优先工作流进度则牺牲前缀复用率，引发大量前缀搬运成本。

## 核心贡献（创新点）
1. **形式化在线前缀状态调度问题**：首次将前缀驻留与请求接纳统一建模为共享KV预算下的联合决策问题，揭示异构前缀共存压缩动态批次容量的机制。
2. **TOPAS联合调度框架**：设计面向JCT的效用函数，显式衡量 admitted 请求对任务最长剩余服务路径的期望缩减，同时建模前缀搬运、抢占重做与信用撤销成本，区别于此前将前缀驻留视为请求排序副产品的工作。
3. **短视前缀复用评估**：引入一步工作流前瞻，统计即将就绪的下层阶段并按前缀长度加权前缀复用价值，避免单纯依赖当前前缀长度（LPM）造成的上游饿死。
4. **任务级老化机制**：通过log型老化因子对admission进步加权，防止新到达任务反复抢占老任务，缓解饥饿问题。
5. **层次化状态搜索算法**：针对组合爆炸设计小池全枚举+大池贪心单增/对增/修复的搜索策略，使前缀集生成与请求分配解耦，保证实时可行性。

## 方法详解
- **状态表示与约束**：调度点 $t$ 的状态为 $S_t = (R_t, Q_t)$，其中 $R_t$ 为驻留智能体前缀集合，$Q_t$ 为运行请求集合；候选后状态 $(R', Q')$ 须满足 $\sum_{a \in R'} \ell_a + \sum_{r \in Q'} d_r(t) \leq B_t$，同智能体请求只支付一次静态前缀成本。
- **基础转移效用**：$\text{Score} = J_{\text{admit}} - J_{\text{move}} - J_{\text{redo}} - J_{\text{revoke}} + V_{\text{reuse}}$，各项以秒度量。
- **任务进展度量**：依据历史服务分布 $F_{a(v),v}$ 估计每阶段剩余服务随机变量 $Z_{i,v}(t)$，在逆拓扑序上计算最长剩余路径期望 $\widehat{L}_i(\mathcal{H},t)=\mathbb{E}[\text{LP}(G,\mathbf{Z}_i)]$；单请求条件进展 $\Delta_{\text{task}}(r|\mathcal{H},t)=\widehat{L}_i(\mathcal{H},t)-\widehat{L}_i(\mathcal{H}\cup\{r\},t)$，将顺序更新的累计进展求和得到 $J_{\text{admit}}$。
- **前缀搬运代价**：迁移前缀集合 $R_{\text{in}}, R_{\text{out}}$ 的KV字节经CPU-GPU swap，串行搬运时间为 $\frac{\mu}{B_{\text{swap}}}(\sum \ell_a^{\text{out}}+\sum \ell_a^{\text{in}})$，每条受影响任务的等待次数计为 $|T_{\text{in}}|$，得 $J_{\text{move}}$。
- **抢占重做与撤销**：被抢占请求的自回归重建代价 $J_{\text{redo}}=\sum_{r\in P} T_{\text{decode}} \cdot y_r(t)$； admission 时记录的信用 $\lambda_r$ 在抢占时撤销，防止未完成尝试重复累积进展，$J_{\text{revoke}}=\sum_{r\in P} \lambda_r$。
- **短期复用价值**：统计下一跳即将就绪且仅依赖当前运行前驱的阶段数 $n_a(t)$，加权前缀重新加载时间：$V_{\text{reuse}}=\beta \sum_{a\in R'} n_a(t) \frac{\mu \ell_a}{B_{\text{swap}}}$。
- **任务老化**：$g_i(t)=1+\rho \log(1+(t-\tau_i)/H_{\text{age}})$，admission 项按任务年龄缩放为 $J_{\text{admit}}^{\text{age}}$。
- **层次化搜索**：活跃智能体池 $|\mathcal{A}_t|\leq M$ 时枚举所有子集并调用 GREEDYPACK（贪心打包）；否则从 $R_t$ 出发贪心单增，停滞时尝试所有对增，辅以单增删除与置换修复，搜索复杂度控制在 $O(|\mathcal{A}_t|^2)$ 次前缀集评估。

## 实验与结果
- **实现与平台**：基于 SGLang v0.5.3 在单卡 NVIDIA A100 80GB 上实现，模型 Qwen2.5-32B-Instruct；调度开销实测平均 1.9 ms/决策，占墙钟时间 0.31%。
- **工作负载**：三个合成DAG（Chain-3、DAG-4、DAG-10-Wide）与两个真实 MetaGPT 软件工程工作流（MetaGPT-SOP 五角色静态流水线、MetaGPT-TL 九阶段星形组织含重复 TeamLeader）。
- **对比基线**：FCFS、LPM、Parrot-FCFS、Autellix-LAS、SPF（Chain-3 上等价于 SRPT）。
- **合成DAG结果**：相对各 workload 最强基线，TOPAS mean/p99 JCT 最高降低 39.8%/49.4%（DAG-4）；Chain-3 上降低 27.5%/31.7%，DAG-10-Wide 上降低 27.7%/30.8%。
- **MetaGPT结果**：MetaGPT-SOP 相对最强基线 SPF，mean/p99 JCT 降低 9.8%/4.5%，请求吞吐提升 6.7%；MetaGPT-TL 相对 SPF/Parrot-FCFS，mean/p99 JCT 降低 22.0%/26.6%，随着负载上升优势扩大。
- **消融**：MetaGPT-TL 上完整 TOPAS 相对去除复用与老化的 TOPAS-base 进一步降低 mean/p99 JCT 60.5%/53.6%；相对各单组件变体降低 44.9%–51.0%/44.2%–48.5%。

## 相关工作脉络
- **请求级优化框架**：SGLang 等系统聚焦 batching、prefixed caching 与 paged KV cache，但仅从请求视角排名就绪请求，未将前缀驻留与工作流进展联合。
- **数据流/程序感知调度**：Parrot 将应用语义变量纳入请求排序，Autellix 按程序进展调度，侧重任务级优先而非显式前缀驻留控制。
- **KV缓存跨上下文复用**：KVFlow 利用 Agent Step Graph 引导缓存驱逐与预取，但缓存 Placement 与请求 admission 仍解耦；TOPAS 将其统一到状态转移效用中。
- **分布式/解耦服务**：Splitwise、DistServe 分离 prefill/decode，Mooncake、MemServe 弹性管理跨节点缓存，关注吞吐而非多智能体任务 JCT。
- **多智能体系统优化**：Kairos、Self-resource allocation 等从资源分配或路由角度优化多智能体执行；TOPAS 定位在推理服务端，填补“前缀驻留×工作流进展”联合调度的空白。

## 局限性与未来方向
- **未知未来事件**：ReAct 迭代轮数、tool 延迟与下游到达时间不可预知，当前利用经验分布做期望估计，极端长尾场景下预测偏差未量化。
- **大活跃池的贪心局限**：当同时就绪智能体远超 M 时依赖贪心单增/对增，可能错过全局更优前缀组合；搜索上限受 $K$ 限制，极端多分支 fork 场景未验证。
- **单卡评估**：实验仅在单 A100 上进行，未覆盖 disaggregated 或多机多卡场景下的跨节点 KV 迁移与并行 batch 调度。
- **固定长度合成负载**：合成 DAG 使用固定查询与固定阶段生成时长，未纳入真实用户内容变异带来的动态 prefix length 分布。
- **未来方向**：可探索学习型残差分布建模、结合多机 disaggregated 架构的跨节点前缀路由、以及面向更长 ReAct 链路的深度前瞻。

## 研究启发与可借鉴点
- **进展度量替代时间度量**：用任务最长剩余服务路径期望缩减 $\Delta_{\text{task}}$ 代替原始请求剩余时间，能将 DAG 瓶颈结构显式纳入调度，适用于其他依赖图驱动的服务调度场景。
- **联合状态搜索解耦策略**：先枚举小前缀集再贪心打包请求的“层次化搜索”思路，可在资源耦合的调度问题中推广（如 GPU/TPU 分片与批次联合决策）。
- **成本显式化到效用函数**：将前缀搬运、抢占重做、信用撤销都以时间为单位折算入 Score，使调度器可在不同操作间公平比较，类似思路可用于其他存在迁移/回滚开销的系统。
- **老化因子抑制饥饿**：log 型任务老化仅缩放 admission 进展而不影响已提交进展，平衡新到达响应与老任务完成，可移植到消息队列、数据库查询调度等公平性敏感系统。
- **工作流前瞻复用价值**：一步看底下层即将就绪阶段数 $n_a(t)$ 并与前缀长度加权，为"cache placement + admission"联合优化提供了轻量启发式，可作为通用模板嵌入其他工作流感知调度器。

## 关键术语表
- **Prefix residency（前缀驻留）**：智能体静态系统提示的 KV cache 保留在 GPU 显存中，避免重复 prefill，但占用可用于动态后缀与 decode 的预算。
- **Job Completion Time (JCT)**：任务从到达至所有阶段完成的墙钟时间，本文以 mean/p99 衡量多智能体工作流端到端延迟。
- **Longest Remaining Path (LP)**：在多阶段 DAG 中从当前状态沿逆拓扑序计算的所有未完成阶段服务时间分布的最大期望路径。
- **GREEDYPACK**：在给定前缀集合约束下，贪心选取 age-weighted 条件进展最大的请求填充 KV 预算，直至无法继续或收益为零。
- **Task-level aging**：基于任务到达时间的 log 因子 $g_i(t)$，对 admission 进展加权以抑制新到达任务反复抢占老任务。
- **One-hop lookahead reuse**：统计下一跳即将就绪且仅依赖当前运行前驱的下游阶段数，按前缀加载时间加权计入 $V_{\text{reuse}}$。
- **Hierarchical state search**：小池全枚举 + 大池贪心单增/对增/修复的组合搜索，将前缀集生成与请求分配解耦，控制在线调度延迟。
- **SGLang**：支持结构化 LLM 程序执行与 radix-tree 前缀缓存的开源推理框架，本文在其上实现 TOPAS 调度模块。

## 可复现要素
- **数据集**：三个合成 DAG（Chain-3、DAG-4、DAG-10-Wide）由固定长度查询与阶段生成构成；两个 MetaGPT 工作流使用 MetaGPT SoftwareDev 数据集采样提示。论文未提供独立公开数据集链接。
- **代码**：TOPAS 作为调度模块实现在 SGLang v0.5.3 之上；论文未声明独立开源仓库与权重，需自行在 SGLang 中集成。
- **关键超参**：前缀集枚举上限 $M$、单次 admit/preempt 上限 $K$、复用强度 $\beta$、老化强度 $\rho$、老化归一尺度 $H_{\text{age}}$、KV字节/token $\mu$、swap带宽 $B_{\text{swap}}$；补充材料给出老化更新规则，正文未列具体数值。
- **硬件与环境**：单 NVIDIA A100 80GB，模型 Qwen2.5-32B-Instruct，SGLang v0.5.3；工作负载到达率服从 Poisson 过程，采样轨迹在所有策略间共享。
