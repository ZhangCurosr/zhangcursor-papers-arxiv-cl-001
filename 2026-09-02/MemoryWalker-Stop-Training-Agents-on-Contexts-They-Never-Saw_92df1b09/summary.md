---
title: "MemoryWalker-Stop-Training-Agents-on-Contexts-They-Never-Saw"
source: https://arxiv.org/pdf/2609.00865v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-09-06 05:23:20"
---

# 论文速读：MemoryWalker-Stop-Training-Agents-on-Contexts-They-Never-Saw

## 一句话总结
论文针对生产级 Agent Harness 在 Rollout 过程中频繁压缩 Context Window 所导致的训练-推理条件不一致问题，形式化了 **Conditioning Invariant**，并提出 LogitTree、4D Attention Mask 与 SDCC 三种训练修正方案，使 Agent 在 RL 微调时不再“看到”未曾真实遭遇的上下文，最终以极低开销逼近无压缩基线性能。

## 研究问题与动机
- **条件树与单序列的错位**：生产级 Harness（如 Claude Code、Qwen-Agent）在 Rollout 中会按策略驱逐 Context，导致实际交互历史展开为一棵 **Conditioning Tree**，而非传统 RL 假设的单条序列。
- **Naive-Compressed 的时间旅行泄漏**：仅保留最右 root-to-leaf 路径会导致未来发生的 eviction 信息反向流入已生成 token 的 conditioning，破坏因果一致性。
- **Naive-Full 的陈旧上下文泄漏**：按 depth-first 遍历完整物理轨迹会使已驱逐内容继续被后续 token 看见，训练分布与部署分布产生偏差。
- **缺乏条件不变性保障机制**：现有 Agent RL 流程未显式约束 $\forall t,\ c_t^{\text{train}} \equiv c_t^{\text{rollout}}$，导致压缩策略直接套用会引入系统性漂移。

## 核心贡献（创新点）
1. **形式化 Conditioning Invariant 并量化朴素策略的泄漏量级**：首次将训练条件分布与 rollout 真实视图严格对齐，揭示 Naive-Compressed 在 AgentFold 上漂移高达地板的 26 倍。
2. **LogitTree：分支显式分解的梯度精确方案**：将条件树拆为 $K{+}1$ 条 root-to-leaf 路径独立前向，配合 loss mask 限定 token 归属，在 dense softmax 下与完整树注意力的梯度严格等价（Theorem 1）。
3. **4D Attention Mask：单次前向的打包掩码实现**：通过 $(B, \text{num\_heads}, T, T)$ 四维掩码将物理序列打包为单序列，每 token 仅 attend 自身分支，避免多轮 backward 开销。
4. **SDCC：叶节点门控的自适应自蒸馏**：在压缩 walk 上执行单次 student backward，仅在 diverging leaf 处用 stop-gradient teacher 重建驱逐前 prefix 并施加 forward KL，训练成本仅 1.55× base 且 KL 强度随驱逐密度单调自适应。

## 方法详解
- **Conditioning Invariant**：$\forall t,\ c_t^{\text{train}} \equiv c_t^{\text{rollout}} \equiv \text{VIEW}_t(\mathbf{H}_{\leq t}; \mathcal{E}_{\leq t})$，即 $t$ 步训练 conditioning 只能包含截至该步已发生的编辑，严禁引入未来 eviction 或已驱逐内容。
- **LogitTree**：对每轮 rollout 的 $K{+}1$ 条 root-to-leaf 分支分别执行前向，loss mask 将每个 token 的损失/梯度严格限定于其所属分支。wallclock 开销约 $5{-}20\times$，仅适用于 white-box harness（可获取 per-leg live view）。
- **4D Attention Mask**：将整棵物理序列 $\mathbf{H}_T$ 打包为单一 sequence，通过四维自定义 mask 阻断跨分支注意力。实际训练采用 **branch-replicated packing**（每分支复制共享 trunk token，block-diagonal mask），在 Proposition 4 所列五个条件下与 physical-union 视角数学等价，单次 forward 完成。
- **SDCC**：学生端在压缩 walk 上单次 backward；教师端（stop-gradient）在 diverging leaf 处重建 pre-eviction prefix。对齐目标为 **forward KL**（学生需覆盖教师高概率输出，符合 wake–sleep 语义）。Pinsker 界保证若 residual per-junction $\varepsilon_{\text{KL}}$ 足够小，则 total variation behavioral gap $\leq O(\sqrt{\varepsilon_{\text{KL}}})$。KL 权重随 batch 内是否存在 live fold 动态开关，实现稀疏精准修正。
- **关键理论命题**：Theorem 1 证明 LogitTree 与 4D mask 在 dense softmax attention 下梯度等价；Proposition 2 建立 KL 残差与行为分布 TV 距离的收敛界。

## 实验与结果
- **训练数据**：81,638 条轨迹（REDSEARCHER + ASEARCHER 拼接）。
- **测试基准**：NQ、TRIVIAQA、HOTPOTQA、2WIKI、MUSIQUE、BAMBOOGLE、FRAMES（共 7 个，38,270 题/checkpoint）。
- **驱逐编辑器**：MemexRL（model tool/memory_offload）、TC-RAG（model tool/pop）、AgentFold（length fold/3k tokens）。
- **Conditioning Drift（logdif）**：
  - No-compression floor：**0.014**
  - Naive-Compressed @ AgentFold：**0.366**（≈ 26× floor）
  - Naive-Full @ AgentFold：**0.203**
  - 4D mask @ AgentFold：**0.022**（接近 floor）
  - SDCC @ AgentFold（末轮）：**0.071**，训练全程下降 **55%** 单调趋近地板
  - LogitTree（全 harness）：**0.012–0.013**（稳定在 floor）
- **Qwen3-4B / MemexRL 7-bench Agentic-Eval（Table 9）**：
  | 方法 | EM↑ | 漂移↓ | 成本（× base） |
  |---|---|---|---|
  | Search-R1（no-comp.） | 23.0 | 0.0135 | 1.05 |
  | Naive-Compressed | 28.9 | 0.0237 | 1.00 |
  | Naive-Full | 32.1 | 0.0163 | 1.05 |
  | **LogitTree** | **45.9** | 0.0133 | 4.20 |
  | **4D mask** | 33.4 | 0.0140 | 1.35 |
  | **SDCC** | **43.1** | 0.0165 | 1.55 |
- **Black-box Harness 表现**：Claude Code 下 SDCC avg EM **37.5** vs LogitTree **35.9**；OpenCode 下 **36.9** vs **35.0**。
- **SDCC KL 与驱逐密度单调正相关**：MemexRL $0.01\times10^{-3}$、TC-RAG $0.11\times10^{-3}$、AgentFold $2.86\times10^{-3}$，证明修正量精准匹配编辑器改写程度。
- **GRPO 超参**：$G{=}16$、lr $=10^{-6}$、$\beta_{\text{KL}}{=}10^{-3}$。

## 相关工作脉络
- **Search-R1**：无压缩原始基线（EM 23.0），本文将其视为理论下界参考，凸显压缩泄漏带来的性能损失空间。
- **Naive-Compressed / Naive-Full**：工业界常见两套默认处理范式，本文通过 logdif 量化其系统性漂移，证明直接复用会导致训练分布偏移。
- **Claude Code / OpenCode**：作为 black-box reward trace 来源，噪声较大且窗口较短，不被视作可直接对比的 reward-based method，而是验证 SDCC 兼容性的部署场景。
- **MemexRL / AgentFold / TC-RAG**：三种典型驱逐策略构成不同密度压力测试床，本文方法在其上均实现 drift 收敛，证明通用性。
- **定位差异**：现有 Agent RL 工作多假设完整 context 可用或仅做朴素截断；本文首次在机制层保障 conditioning invariant，并提供从精确（LogitTree）到高效（SDCC）的完整技术栈，填补 context-aware RL 训练的理论-工程缺口。

## 局限性与未来方向
- **LogitTree 计算开销较高**（4.20× base），在超长 rollouts 或高分支树场景下面临显存与 wallclock 压力，需依赖更高效的稀疏 attention kernel。
- **SDCC 依赖 white-box teacher prefix 重建**，在完全黑盒 harness 中需依赖近似重构，重建误差可能成为漂移上限。
- **实验规模局限于 Qwen3-4B**，未验证百倍参数量级或更长 horizon 任务下的 scaling 行为。
- **驱逐策略与训练算法的联合优化尚未探索**，当前 SDCC 的 KL 权重为离线经验设定，未来可学习自适应调度。
- **冷启动阶段所有方法漂移均收敛至 0.0115–0.0151**，表明初始阶段 conditioning 偏差较小，主要收益体现在训练中后期累积漂移的抑制。

## 研究启发与可借鉴点
- **Conditioning Invariant 的形式化思路**可迁移至任何存在状态截断/缓存淘汰的 sequential decision-making 系统（如 RAG agent、tool-calling pipeline）。
- **Leaf-gated sparse KL 正则化**提供了一种“仅在分布分歧处施加修正”的高效范式，避免全程蒸馏的计算浪费，适合高稀疏扰动环境。
- **Branch-replicated packing + block-diagonal mask** 实现了 LogitTree 精度与单次前向效率的折衷，可作为通用训练 trick 集成至主流 RLHF 框架。
- **Logdif 动态监控指标**可作为 context-aware RL 的训练诊断信号，早停或 adaptive clipping 均可基于该指标触发。
- **驱逐密度-KL 权重单调对应关系**提示未来可设计 density-conditional hyperparameter scheduler，实现零人工调参的跨 harness 适配。

## 关键术语表
- **Conditioning Invariant**：训练时第 $t$ 步的上下文条件必须严格等于 rollout 时模型实际接收的视图，禁止引入未来编辑或已驱逐内容。
- **Time-Travel Leakage**：朴素压缩训练中，未来发生的 eviction 操作信息
