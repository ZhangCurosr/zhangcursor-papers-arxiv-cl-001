---
title: "MemoryWalker-Stop-Training-Agents-on-Contexts-They-Never-Saw"
source: https://arxiv.org/pdf/2609.00865v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-09-06 05:23:26"
---

# 论文速读：MemoryWalker-Stop-Training-Agents-on-Contexts-They-Never-Saw

## 一句话总结
本文针对长上下文 Agent 训练中因上下文压缩/驱逐导致的“条件漂移（Conditioning Drift）”问题，提出 SDCC（基于前向 KL 正则化的停止梯度补偿训练）与 LogitTree 两种方法，使模型在仅见过压缩后上下文的情况下，仍能通过轻量 KL 对齐逼近完整上下文下的策略梯度，在不增加额外计算成本的前提下有效抑制训练-评估分布偏差。

## 研究问题与动机
- **核心问题**：长上下文 Agent 在 rollout 阶段常触发压缩或驱逐机制（如 TC-RAG、AgentFold、MemexRL），导致训练时模型仅接触压缩后历史 $H^{comp}$，而推理/评估依赖完整历史 $H_T$，产生 Conditioning Drift。
- **现有方法不足**：朴素压缩训练（Naive-Compressed）的 drift 不会随 GRPO 自适应自修正，训练中持续存在 Pitfall A 陷阱；传统反向 KL 序列级 Monte-Carlo 估计器存在方差无界与采样盲区两类病理性失败。
- **工程与理论缺口**：多数偏差校正方法需自定义 attention kernel 或暴露 eviction spans，难以与 vLLM/SGLang 等原生推理栈兼容；缺乏在稳态步骤下统一测量并控制压缩偏置的严格理论框架。
- **目标**：以单次反向传播的成本隐式利用被驱逐 token，使策略梯度偏置有界并以确定速率收敛，同时保持对不同驱逐拓扑与黑盒编辑器的通用兼容。

## 核心贡献（创新点）
1. **提出 SDCC 前向 KL 补偿训练框架**：通过冻结 teacher 并以 teacher 分布采样拟合 student，将梯度转化为普通 cross-entropy score，规避反向 KL 的方差无界问题，以单次 backward 实现漂移校正。与 LogitTree 相比，牺牲 $O(\sqrt{\varepsilon_KL})$ 残差换取计算成本常数化。
2. **建立 Behavioral Pinsker 界与策略梯度偏置收敛理论**：证明前向 KL 残差 $\varepsilon_p$ 可直接控制 teacher-student 分布的 TV 距离，且策略梯度偏置以 $O(T^{-1/4})$ 衰减。与既有 VAE 类理论相比，首次将分布漂移上界显式绑定到长上下文 Agent 的策略优化目标。
3. **揭示 LogitTree 与 SDCC 的双向目标等价性**：在 conditioning invariant (CI) 零集上两方法严格一致，偏离时误差受 $A_{\max}\sqrt{2\varepsilon_p}$ 控制。与单纯经验对比不同，本文给出双向紧 bound 与 $\lambda$ 调节上下界紧度的理论解释。
4. **设计 Rollout-Training Contract 统一训练接口**：提出 $(H^{comp}, H_T, \{(J_k, E_k)\}_{k=1}^K)$ 的记录契约，原生对齐 vLLM/SGLang，无需自定义 kernel 即可兼容白盒（栈式/规则折叠/外置 KV）与黑盒（API 仅见 rendered payload）编辑器。

## 方法详解
- **Rollout-Training Contract**：每条 rollout 统一产出压缩上下文 $H^{comp}$、完整历史 $H_T$、以及 $K$ 个驱逐事件 $(J_k, E_k)$（$J_k$ 为触发位置，$E_k$ 为被驱逐 token 集合），作为教师回放与 drift 诊断的基准。
- **SDCC（核心方法）**：冻结 teacher 参数（stop-gradient），在压缩上下文上以 teacher 分布采样生成训练样本，student 仅执行前向 KL 对应的 cross-entropy 学习。KL 系数 $\lambda$ 线性预热（前 20% 步从 0→0.1）后恒定。单轨迹仅需 1 次 backward，且梯度方差有界。
- **LogitTree**：将完整历史按驱逐边界分段切片，对每段执行 teacher forward，通过 $K$ 个 segment 边界重建教师 prefix。成本为 $K+1$ 次 forward/backward，支持任意 backbone。
- **4D Mask**：在全序列上施加 4D 注意力掩码并配合无孔 position ids，恢复 CI（仅 dense 结构有效），但需自定义 attention kernel。
- **理论保障**：
  - **Prop. 10（Behavioral Pinsker 界）**：$\|p_\theta(\cdot|z_p) - q_\theta(\cdot|x_p)\|_{TV} \leq \sqrt{\varepsilon_p/2}$。
  - **Cor. 1/2**：策略梯度偏置 $\|\nabla L_{task} - \nabla L^*_{task}\| \leq C\sum\sqrt{\varepsilon_p/2} = O(\sum\sqrt{\varepsilon_p})$，且 $\varepsilon_p = O(1/\sqrt{T}) \Rightarrow$ 偏置以 $O(T^{-1/4})$ 衰减。
  - **Thm. 3**：$|L^{SDCC}_p - L^{LT}_p - \lambda\varepsilon_p| \leq A_{\max}\sqrt{2\varepsilon_p}$，CI 零集上严格等价。
  - **Thm. 4（稳态收缩）**：$\min_{t\leq T} E[\Phi(\theta_t)] \leq$ 初值衰减项 $+$ 噪声底，$K(\theta)=\sum\varepsilon_p$ 以 $O(1/\sqrt{T})$ 收敛。

## 实验与结果
- **数据集**：REDSEARCHER + ASEARCHER 聚合，共 81,638 个 composite QA 实例。
- **评测基准（7 个，共 38,270 题）**：NQ、TRIV-IAQA、HOT-POTQA、2Wiki-MultiHopQA、MUSIQUE、BAMBOOGLE、FRAMES。
- **奖励机制**：Exact-Match + 0.2 格式 bonus。
- **主要结果（Table 9，Qwen3-4B / MemexRL）**：

| Method | EM↑ | Drift↓ | Cost (×base) |
|---|---|---|---|
| Search-R1 (no-comp.) | 23.0† | 0.0135 | 1.05 |
| Naive-Compressed | 28.9† | 0.0237 | 1.00 |
| Naive-Full | 32.1† | 0.0163 | 1.05 |
| **LogitTree (ours)** | **45.9†** | **0.0133** | **4.20** |
| **4D mask (ours)** | **33.4†** | **0.0140** | **1.35** |
| **SDCC (ours)** | **43.1†** | **0.0165** | **1.55** |

- **关键结论**：SDCC 以 1.55× 计算成本达到 EM 43.1†，显著优于 Naive-Compressed（+14.2 EM，Drift 下降 30%），性能接近 LogitTree 但成本降低约 3×。Naive-Compressed 的 Pitfall A 在 GRPO 自适应下仍持续存在，而 SDCC 通过 KL 正则化逐步缩小与无压缩基准差距。训练动力学显示 KL_SDCC 非零步仅略超一半，恰好对应含实时信息的驱逐事件。

## 相关工作脉络
1. **Naive-Compressed / Naive-Full**：直接丢弃或被驱逐 token 或全量保留完整序列，分别面临条件漂移或显存/计算不可行的极端，本文揭示前者 drift 无法随 GRPO 自修正。
2. **4D Mask 方法**：通过多维注意力掩码近似完整上下文因果结构，仅对 dense backbone 有效且需自定义 kernel，本文以其作为中等成本基准对比。
3. **LogitTree**：将长序列按驱逐边界分段并重放 teacher logits，支持任意 backbone 但成本随 K 线性增长，本文证明其与 SDCC 在 CI 零集上的理论等价性。
4. **VAE / 反向 KL 估计器**：传统序列级 Monte-Carlo 估计存在方差无界与采样盲区，本文转向前向 KL 并引入 stop-gradient 规避病理性问题。
5. **长上下文 Agent 系统（TC-RAG / AgentFold / MemexRL）**：
