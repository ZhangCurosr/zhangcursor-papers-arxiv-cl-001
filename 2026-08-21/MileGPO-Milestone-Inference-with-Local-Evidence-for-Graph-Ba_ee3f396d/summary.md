---
title: "MileGPO-Milestone-Inference-with-Local-Evidence-for-Graph-Ba"
source: https://arxiv.org/pdf/2608.19803v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-02 02:07:03"
---

# 论文速读：MileGPO: Milestone Inference with Local Evidence for Graph-Based Policy Optimization of Long-Horizon LLM Agents

## 一句话总结
针对长程 LLM Agent 强化学习中最终奖励稀疏、现有图基信用分配方法仅依赖终点最短路径距离导致同状态分支难以区分的问题，论文提出 MileGPO，通过里程碑发现、可靠性校准塑形与局部对比校准三步，仅从同策略轨迹组与滚动图中挖掘过程级信用信号，无需外部标注或辅助模型，在 ALFWorld 和 Web-Shop 上达到 SOTA 并显著缩小 ID–OOD 泛化差距。

## 研究问题与动机
- 长程 Agent RL 通常仅在任务结束时获得环境奖励，中间决策的信用分配困难。
- GraphGPO 将同策略轨迹合并为转移图并按终点最短路径距离衰减分配信用，但该距离仅反映可达性，无法区分距离相同但成功概率差异显著的中间状态。
- 实际 Agent 轨迹中约 74% 的转移来自跨轨迹共享的状态，且多数共享状态包含多个观测动作，但终点距离对 54.4%（ALFWorld）和 72.7%（Web-Shop）的同状态动作对赋以相同优势，造成局部信用平局。
- 成功轨迹访问过的状态未必代表可靠进展，失败轨迹中的重复状态亦构成陷阱，需更细粒度的局部证据对中间信用进行筛选与校准。

## 核心贡献（创新点）
- **揭示图距离信用的不可靠性**：指出终点最短路径信用无法区分同距离分支，且 success-visited states 不等于 reliable progress，为后续校准提供问题锚点。
- **提出纯轨迹原生的 MileGPO 算法**：无需过程标注、Critic、奖励模型或额外环境交互，仅利用当前任务 Rollout 组与聚合转移图完成步骤级信用重塑。
- **设计“发现-加权-验证”嵌套信用机制**：通过 MD 挖掘候选、RCS 按任务内可靠性打分塑形、PCC 结合分支反事实与局部进展二次验证，实现有界修正与距离平局的理论可消解性。

## 方法详解
- **分组轨迹图构建**：对任务 $q$ 采样 $K$ 条轨迹 $\tau_i$，记录最终环境奖励 $R_i^{\mathrm{env}}$ 与成功标志 $y_i$，将所有轨迹规范化状态后合并为有向图 $\mathcal{G}_q=(\mathcal{V}_q,\mathcal{E}_q)$，边 $e=(u,v)$ 表示一次转移，$d(u,v)$ 为有向最短路径距离，$g$ 为目标成功态。
- **MD（里程碑发现）**：成功轨迹访问的非终止态记为里程碑候选，仅失败轨迹出现或重复访问的态记为陷阱候选。里程碑分 $S_0^+(v)=w_s\ell^+(v)+w_m m(v)+w_c C(v)$，其中 $\ell^+$ 为条件成功率超出组平均的部分，$m(v)$ 为成功覆盖率，$C(v)$ 为图中心度归一值；陷阱分 $S^-(v)=w_f\ell^-(v)+w_l f^-(v)L(v)$，综合失败率超额、失败覆盖与平均重访次数。
- **RCS（可靠性校准塑形）**：对候选分做任务内 Max-Normalization 得到 $\overline{S}_0^+$ 与 $\overline{S}^-$，生成距离衰减的正/负势能 $\Phi_R^+(s)=\max_{v\in S^+}\overline{S}_0^+(v)\omega^{d(s,v)}$ 与 $\Phi_R^-(s)=\max_{v\in S^-}\overline{S}^-(v)\omega^{d(s,v)}$。沿转移 $s_t\to s_{t+1}$ 计算单向增量 $\delta_t^+=\max(\gamma_\Phi w_+\Phi^+(s_{t+1})-w_+\Phi^+(s_t),0)$ 与 $\delta_t^-=\max(\gamma_\Phi w_-\Phi^-(s_{t+1})-w_-\Phi^-(s_t),0)$，塑形返回为 $r_t^{\mathrm{M}}=c\gamma_G^{d(s_{t+1},g)}+c\lambda(\delta_t^+-\delta_t^-)$。
- **PCC（局部对比校准）**：BCC 计算入边 $e=(u,v)$ 的成功率与同父状态其他分支平均成功率的差，取归一化最大值 $b(v)$ 作为分支对比证据；Local Progress 综合距离缩减 $\Delta d(e)$、成功率增益 $p^+(v)-p_q^+$ 与失败率抑制 $\ell^-(e)$ 计算 $\psi(e)$，取最大并归一化为 $\bar{\psi}(v)$。更新得分 $E(v)=w_{\mathrm{bc}}b(v)+w_{\mathrm{pg}}\bar{\psi}(v)$，按保留指示 $z(v)$（满足进展证据、覆盖率阈值或分支证据之一）放大或收缩候选，生成 $\overline{S}_{\mathrm{pcc}}^+$ 替代 RCS 的原始分。
- **策略更新**：将 GraphGPO 返回 $r_t^{\mathrm{G}}$ 与 MileGPO 返回 $r_t^{\mathrm{M}}$ 分别按任务-源状态归一化得 $\widehat{A}_t^{\mathrm{G}}$ 与 $\widehat{A}_t^{\mathrm{mix}}$，残差 $\widehat{A}_t^{\mathrm{res}}=\widehat{A}_t^{\mathrm{mix}}-\widehat{A}_t^{\mathrm{G}}$ 经 $\eta$ 缩放后与 episode 级归一化分数组合，输入与基线相同的 PPO 裁剪目标与 KL 正则。

## 实验与结果
- **数据集**：ALFWorld（室内导航与对象操作，3553 训练任务，含 ID 与 OOD 测试集）、Web-Shop（网页购物交互，官方-small 测试集）。
- **基线与设置**：本地复现 GiGPO 与 GraphGPO，对比 GPT-4o、Gemini-2.5-Pro、ReAct、Reflexion、PPO、RLOO、GRPO 等提示/RL 基线；策略模型 Qwen2.5-1.5B-Instruct，每步 16 任务组×8 轨迹，300 优化步，3 次随机种子，ALFWorld 最大 horizon 50，Web-Shop 30。
- **主结果**：ALFWorld 总成功率 94.60%，较 GraphGPO* 提升 3.13 点、较 GiGPO* 提升 4.43 点；Web-Shop 任务得分 90.29%（+2.05）、成功率 78.58%（+3.78）。ALFWorld ID–OOD gap 仅 1.69，低于 GraphGPO 的 3.78 与 GiGPO 的 1.89。
- **消融**：移除 PCC 使 ALFWorld OOD 下降 3.71 点、Gap 扩大 3.32 点；再移除 RCS 使 Web-Shop 成功率再降 3.3
