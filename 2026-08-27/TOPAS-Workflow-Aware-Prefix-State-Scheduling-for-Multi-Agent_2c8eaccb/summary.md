---
title: "TOPAS-Workflow-Aware-Prefix-State-Scheduling-for-Multi-Agent"
source: https://arxiv.org/pdf/2608.25523v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 01:49:45"
field: "LLM Serving 系统"
keywords: ["LLM Serving", "Multi-Agent Scheduling", "Prefix Caching", "Job Completion Time", "KV Cache Management", "Workflow Scheduling"]
innovations: ["首次将前缀驻留作为显式工作流级调度决策与请求 admission 联合优化", "基于任务最长剩余路径（LP）的 JCT 导向效用函数，显式建模前缀迁移与抢占成本", "层次化状态搜索 + 任务级老化机制，在合成与 MetaGPT 工作流上大幅降低 mean/p99 JCT"]
benchmarks: ["Chain-3", "DAG-4", "DAG-10-Wide", "MetaGPT-SOP", "MetaGPT-TL"]
---

# 论文速读：TOPAS-Workflow-Aware-Prefix-State-Scheduling-for-Multi-Agent

## 一句话总结
论文提出了 TOPAS（Task-Oriented Prefix-Aware Scheduler），一种面向多智能体 LLM 服务的在线调度器，首次在共享 KV 缓存预算下联合优化前缀驻留与请求 admission，通过权衡任务级进展与前缀复用收益来降低端到端作业完成时间（JCT）。

## 研究问题与动机
1. **前缀驻留与批处理能力的根本权衡**：保留长系统提示的 KV 缓存可加速后续调用，但会挤占 GPU 内存，减少可并发的动态请求数量。
2. **现有方法孤立优化导致次优**：局部性优先策略（如 LPM）会阻塞下游任务，进度优先策略（如 SRPT）频繁切换代理引发前缀迁移开销。
3. **异构前缀共存压缩动态批容量**：诊断实验表明，交替服务不同代理的请求会使平均批大小减半，运行时间增加 1.8–1.9×。
4. **缺乏显式前缀状态决策**：已有系统未将前缀驻留作为独立调度维度，仅作为请求排序的间接副产品。

## 核心贡献（创新点）
1. **形式化在线前缀状态调度问题**：首次将前缀驻留与请求 admission 纳入同一联合决策框架，区别于 Parrot/Autellix 等仅关注请求级排序的工作。
2. **提出 JCT 导向的转换效用函数**：以最长剩余服务路径（LP）的预期缩减作为任务进展度量，显式建模前缀迁移、抢占重做与信用撤销成本，而非依赖启发式局部/全局优先级。
3. **设计层次化状态搜索算法**：对小规模活跃代理池枚举所有前缀子集，对大规模场景采用 O(|A_t|²) 的受限贪婪搜索加修复操作，兼顾求解质量与延迟。
4. **引入任务级老化机制防饥饿**：通过 log 型增长因子放大老任务 admission 增益，避免新到达请求持续抢占，区别于 KVFlow 等无老化设计的系统。
5. **在 SGLang 上的完整实现与评估**：覆盖 3 个合成 DAG 与 2 个 MetaGPT 真实工作流，在最优基线上实现最高 39.8%/49.4% 的 mean/p99 JCT 下降。

## 方法详解
**问题形式化**：在调度点 t，GPU 状态 S_t = (R_t, Q_t) 由驻留前缀集合 R_t 和运行请求集合 Q_t 定义。调度动作选择后决策状态 S' = (R', Q')，需满足联合 KV 预算约束：Σ_{a∈R'} ℓ_a + Σ_{r∈Q'} d_r(t) ≤ B_t。

**基础转换效用**：
- **Admission 收益 J_admit**：对每个 admit 请求 r，计算其在任务 DAG 中预期最长剩余路径（LP）的缩减量 Δ_task(r)，沿历史条件残差分布求期望，允许临界路径随阶段推进动态变化。
- **前缀移动成本 J_move**：基于 KV 字节/令牌 μ、swap 带宽 B_swap 和静态前缀长度 ℓ_a，串行传输所有进出前缀的费用，受影响的每条任务仅计费一次。
- **抢占重做成本 J_redo**：被抢占请求需自回归重建已生成 token，代价为 T_decode × y_r(t)。
- **信用撤销 J_revoke**：为防止未完成任务反复累积 admission 价值，preempt 时回收 admission 时刻记录的 λ_r 信用。

**短期前缀复用价值**：统计即将就绪的下游阶段 n_a(t)（仅未完成前驱正在运行），乘以对应前缀的重载时间 μℓ_a/B_swap 并与超参 β 相乘，仅排名激活状态而非预取不活动代理。

**任务级老化**：g_i(t) = 1 + ρ·log(1 + (t − τ_i)/H_age)，仅缩放 admission 项而非全部分数，确保老任务不会因新 arrival 持续被绕过。

**层次化搜索**：GREEDY-PACK 对每个候选前缀集合 R 构造兼容请求分配；小池（|A_t| ≤ M）枚举全部子集，大池从当前 R_t 出发逐步添加最优单代理/对，并执行 REPAIR 修复；cap K 限制单次 admit/preempt 数量。

## 实验与结果
**实验设置**：实现于 SGLang v0.5.3，单卡 NVIDIA A100 80GB，模型 Qwen2.5-32B-Instruct；任务按固定 Poisson 到达，对比基线 FCFS/LPM/Parrot-FCFS/Autellix-LAS/SPF。

**合成 DAG 工作流**（Chain-3/DAG-4/DAG-10-Wide）：
- 相对最优基线，mean JCT 降低 27.5%/39.8%/27.7%，p99 JCT 降低 31.7%/49.4%/30.8%。

**MetaGPT 真实工作流**：
- MetaGPT-SOP（五角色静态流水线）：相对最优基线 SPF，mean JCT 降 9.8%，request throughput 提升 6.7%。
- MetaGPT-TL（九阶段 TeamLeader 循环）：相对最优基线，mean JCT 降 22.0%，p99 JCT 降 26.6%。

**消融实验**（MetaGPT-TL, 0.0175 task/s）：完整 TOPAS 相对 TOPAS-base（无复用/老化）mean/p99 JCT 再降 60.5%/53.6%；相对单组件变体再降 44.9–51.0%/44.2–48.5%。

**调度开销**：MetaGPT-SOP 0.15 task/s 下平均 1.9 ms/决策，占实验总 wall time 0.31%。

## 相关工作脉络
1. **SGLang/Orca/Sarathi-serve**：底层推理引擎，优化 batching、paged KV-cache、chunked prefill，但仅作用于请求级，不感知工作流拓扑。
2. **Parrot/Autellix**：引入应用数据流或程序级进度指导请求排序，但前缀驻留仍由 runtime 决定，与 admission 解耦。
3. **KVFlow**：利用 Agent Step Graph 指导 KV 节点驱逐/预取，跳过缓存加载中的请求；TOPAS 将前缀驻留本身纳入工作流调度决策。
4. **LPM/SRPT 启发式**：分别代表局部性与进度优先两极，诊断表明各自在互补场景下表现劣化，TOPAS 统一二者。
5. **InferCept/Continuum**：跨外部等待保留 KV 缓存，关注缓存生命周期管理，而非联合 admission-驻留决策。
6. **Multi-agent 系统优化**（Kairos/FlowMesh/DroidSpeak/KV-Comm）：聚焦路由、数据访问、跨 LLM 通信，未将 GPU 内存竞争建模为前缀-请求联合调度问题。

## 局限性与未来方向
1. **未来到达与 ReAct 迭代未知**：调度器无法预知未来任务到达、ReAct 轮次与工具延迟，仅基于历史经验分布做预期估计。
2. **搜索空间受限于活跃代理池大小**：枚举/贪婪搜索在高并发场景下可能遗漏全局最优前缀组合。
3. **单 GPU 假设**：实验仅在单卡 A100 上进行，多卡 disaggregated 场景下的前缀迁移成本模型可能需要扩展。
4. **固定长度的合成工作流**：Chain-3/DAG-4 使用固定长度查询与生成，未覆盖超长 prompt 或高度可变输入的极端压力。
5. **老化机制参数依赖调优**：ρ 与 H_age 的选择需随工作负载动态调整，论文未给出自适应方法。

## 研究启发与可借鉴点
1. **前缀驻留显式化为调度决策**：可将此思路迁移至任何具有静态/半静态前缀的服务系统（如 RAG、多轮对话），将缓存管理从被动 eviction 转为主动规划。
2. **最长剩余路径（LP）作为任务进展度量**：在 DAG 工作流中替代传统剩余工作量估计，能更准确地反映瓶颈分支的真实影响，适用于其他 orchestrator-aware 调度器设计。
3. **层次化搜索策略的工程折衷**：小池全枚举 + 大池受限贪婪+修复的模式，可作为组合调度问题的通用搜索模板，平衡质量与延迟。
4. **任务级老化防饥饿机制**：log 型年龄因子仅放大 admission 项而非全分，避免过度惩罚新 arrival，可复用于多租户排队系统。
5. **经验条件残差分布建模 ReAct**：以已完成执行的 empirical distribution 更新 stage 服务时间，而非固定 profile，提升了不确定性下的决策鲁棒性。

## 关键术语表
**TOPAS**：Task-Oriented Prefix-Aware Scheduler，论文提出的在线调度器，联合决策前缀驻留与请求 admission。
**Job Completion Time (JCT)**：任务完成时间与到达时间之差，本文优化的核心指标。
**Longest Remaining Path (LP)**：基于工作流 DAG 和阶段残差服务时间分布计算的未完成阶段最长链，用于度量任务进展。
**Prefix Residency**：某 agent 的静态 KV 缓存驻留在 GPU 中的状态，由调度器显式控制。
**GREEDY-PACK**：TOPAS 的 Request Allocation 子程序，对每个候选前缀集合迭代 admit 收益最大的请求直到预算耗尽。
**Longest Prefix Match (LPM)**：基线策略，优先 service 拥有最长可重用前缀的请求。
**Shortest Remaining Processing Time (SRPT)**：基线策略，优先 service 剩余工作量最短的任务，等价于本实验 Chain-3 上的 SPF。
**Task-Level Aging**：TOPAS 的老化机制，按任务年龄 log 放大其 admission 增益，防止饥饿。

## 可复现要素
- **数据集**：合成 DAG（Chain-3/DAG-4/DAG-10-Wide，固定 prompt）；MetaGPT 工作流来自 MetaGPT SoftwareDev 数据集 [8]。**论文未声明公开**。
- **代码**：基于 SGLang v0.5.3 实现的调度模块；**论文未声明独立代码仓库**。
- **模型**：Qwen2.5-32B-Instruct。
- **硬件**：单卡 NVIDIA A100 80GB。
- **关键超参**：β（复用强度）、ρ（老化强度）、H_age（年龄尺度）、M（枚举阈值）、K（admit/preempt cap）；**论文正文未给出具体数值，需在 supplementary material 中查找**。
