---
title: "A-Token-Level-Analysis-of-Sampled-Token-Reverse-KL-On-Policy"
source: https://arxiv.org/pdf/2608.25643v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-03 18:38:51"
field: "大语言模型蒸馏与后训练"
keywords: ["on-policy distillation", "reverse KL", "K2 estimator", "token-level analysis", "surprise-aware reweighting", "SuRe", "gradient decomposition", "Qwen3"]
innovations: ["推导 K2 估计量 per-token 逆 KL 梯度的 l1 范数精确分解: 2|Δlog p_t|·(1−π_S)", "揭示低学生概率 token 集中大部分梯度范数并与大赛段正相关的经验模式", "提出 SuRe 无额外开销的有界 surpise-aware 重加权方案, 单系数 α 平滑插值到 vanilla OPD"]
benchmarks: ["AIME24", "AIME25", "AMC23", "MATH-500", "CRUX", "IFEval", "MMLU-Pro"]
---

# 论文速读：A-Token-Level-Analysis-of-Sampled-Token-Reverse-KL-On-Policy

## 一句话总结
本文从梯度层面解析了样本令牌级逆 KL（reverse-KL）在线蒸馏（OPD）的更新分配机制，推导出 K2 估计量梯度的精确分解式，并基于该分析提出 Surprise-aware Reweighting（SuRe）轻量干预方法，在 Qwen3 数学蒸馏上提升了多项指标。

## 研究问题与动机
- **核心问题**：OPD 以 teacher 信号对 student 自身轨迹做监督，但"采样损失如何在不同 token 间分配更新量"缺乏理论刻画，已有工作多用熵或概率等诊断量观察，而非直接分析 realized gradient。
- **现有方法不足**：离线蒸馏（KD/SeqKD）存在 train-inference 分布错位；仅依赖 teacher 访问不一定带来提升（文中 KD/SeqKD 在部分基线下降）；已有的 entropy-aware token selection / probability-based filtering 关注"选/过滤 token"，而非"梯度范数如何分配"。
- **动机切入点**：Kimi K3 的 detached sampled-token log-ratio OPD reward 与 K2 在无 clip 时梯度方向一致，理解"哪些 token 接收大梯度以及为何"有助于设计更有效的 per-token 更新策略。
- **分析缺口**：已有工作用 JSD、entropy 描述 token 不确定性与分歧，但这些是非方向性量；作者主张直接用带符号的 teacher–student 残差 Δlog p_t 作为更强的排序器。

## 核心贡献（创新点）
1. **K2 估计量的梯度恒等式**：推导出 per-token K2 逆 KL 对 student logits 的 l1 范数精确分解为 `2|Δlog p_t|·(1−π_S(y_t|c_t))`，揭示了学生侧 softmax 几何项的显式作用。
   - 与已有工作的区别：此前 OPD 分析多停留在 loss 值或 rollout 层面，本文给出闭式梯度身份，可直接指导 per-token 权重设计。
2. **经验性 token 级梯度分配刻画**：在 Qwen3 数学设置下，低 student-probability token 占梯度范数之和的异常份额，且大 teacher–student gap 在同一区域富集（联合经验模式，非单因子决定）。
   - 与已有工作的区别：不同于 WANG 等用 entropy/top-k 解释 RLVR 梯度集中，本文直接以 realized gradient norm 为刻画对象，发现低 π_S 与大赛段正相关。
3. **SuRe 分析与启发式干预**：提出 detach、有界的 per-token 权重 `w_t=1+α(1−π̄_{S,t})`，不引入额外参考模型或前向开销，仅需单系数 α 即可平滑插值到 vanilla OPD（α=0）。
   - 与已有工作的区别：有别于 entropy-aware token selection（Jin 等）或 probability-based failure filtering（Li 等），SuRe 直接放大 OPD 固有的"surprise 集中"而非截断/丢弃。

## 方法详解
- **符号与设定**：teacher π_T 冻结，student π_S=π_θ；第 t 步上下文 c_t=(x,y_<t)，student 对下一 token 的分布 `π_S(v|c_t)=exp(z_v)/Σ_u exp(z_u)`；训练提示来自固定 D_x，学生 rollout y~π_θ(·|x)。
- **K2 估计量**：per-token 逆 KL 采用 K2 估计（Schulman, 2020）：
  - `Δlog p_t = log π_T(y_t|c_t) − log π_S(y_t|c_t)`
  - `L_t^{RKL} = ½(Δlog p_t)²`
  - 整体损失 `L_RKL = E[ (1/N_valid) Σ_t m_t L_t^{RKL} ]`，teacher 输出视为固定，无额外 policy-gradient 项。
- **Softmax 梯度恒等式**：`∇_z log π_S(y_t|c_t) = e_{y_t} − π_S(·|c_t)`。
- **K2 梯度恒等式（Lemma 1）**：
  - `∇_z L_t^{RKL} = −Δlog p_t · (e_{y_t} − π_S(·|c_t))`
  - 取 l1 范数得 `‖∇_z L_t^{RKL}‖_1 = 2|Δlog p_t|·(1−π_S(y_t|c_t))`。
  - 几何含义： sampled-token 坐标的梯度幅度与其余所有 competitor 坐标之和相等，各贡献 `|Δlog p_t|(1−π_S)`，合计 2 倍。
  - 选择 l1 而非 l2 的原因：l1 是 logit 扰动 l∞ 对偶敏感度，且给出干净的 `1−π_S` 乘子；l2 会引入 `√((1−p)^2 + Σ_{v≠y} π_S(v)^2)` 的复杂形式。
- **Checkpoint-shift 诊断**（Sec 3.1）：比较 Base 与最终 OPD checkpoint 在相同 rollout 上的 log-prob 偏移；仅 8.5%/7.1% 的 token 满足 |Δ_OPD-Base|>1，活跃尾部分布因 rollout 源而异，但这只是端点统计，非梯度分配的直接证据。
- **SuRe 目标**：
  - 定义 detached 学生概率 `π̄_{S,t} = sg(π_S(y_t|c_t))`。
  - 权重 `w_t = 1 + α(1−π̄_{S,t})`，其中 α≥0。
  - 优化目标：`L_RKL-rw = (1/|T|) Σ_{t∈T} w_t L_t^{RKL}`，分母不做加权归一化。
  - 等价效果：梯度被放大为 `2|Δlog p_t|·[(1−π_S) + α(1−π_S)²]`，gap 因子不变，低 π_S 位置相对权重更高。
  - 不引入额外参考模型、额外前向、hard threshold 或 learned selector，仅改变 per-token 标量权重。
- **诊断量对比**：用 |Δlog p_t| 排序可覆盖 top 5%/10% token 的 54.1%/74.6% 梯度范数之和；JSD 仅覆盖 28.4%/47.3%，entropy 覆盖 23.7%/42.4%（图 2），说明带符号残差是更强的排序指标。

## 实验与结果
- **数据集**：主实验使用 DeepMath 57K hard split（难度≥6）；数学评测基准 AIME2024/2025、AMC23、MATH-500；OOD 基准 CRUX（代码推理）、IFEval（指令遵循）、MMLU-Pro（通用）。
- **模型**：teacher = Qwen3-8B；student = Qwen3-1.7B-Base 与 Qwen3-4B-Base；32×H20，lr=1e-6，batch=512，2 epochs，seed=42（另做 seed=43 校验）。
- **主要结果（Table 1）**：
  - **1.7B 尺度**：
    - AIME24 pass@8：Vanilla OPD 16.67 → SuRe(α=1) 23.33（+6.7pp）。
    - AMC23 pass@8：Vanilla OPD 67.50 → SuRe 75.00（+7.5pp）；avg@8 提升 3.7pp（39.38→43.12）。
    - MATH-500 pass@4：Vanilla 80.80 → SuRe 80.80（持平）；avg@4 66.55→67.65（+1.1pp）。
  - **4B 尺度**：
    - AIME24 pass@8：30.00→36.67（+6.7pp）；AMC23 pass@8：85.00→90.00（+5.0pp）；MATH-500 pass@4：87.20→88.40。
  - AIME25 在 4B 上两项均略有下降（pass@8：40.00→36.67）。
  - Base/KD/SeqKD 均不如 OPD，体现离线蒸馏在该流水线中的劣势。
- **OOD 结果**：CRUX、IFEval 上 OPD 与 SuRe 大致持平；1.7B MMLU-Pro 略低于 Base，未见清晰 OOD 退化也未见强泛化提升。
- **训练动力学**（Fig 4/7）：SuRe 提升早期 gradient norm 的同时 actor entropy 与 vanilla OPD 接近，表明提升并非单纯来自 scale-up。
- **强度扫描（α）**：α∈{0.2, 0.5, 1.0, 2.0}，α=1.0 在 k≤4 时整体最优；α=2.0 在 pass@2 上回退至接近 vanilla，说明更大放大并非线性更好。
- **对照实验**：
  - High-reweight（反向给高 π_S 上更大的权重）< SuRe，方向重要。
  - Random-reweight < SuRe；exact-shuffled > vanilla 但 < mean-normalized SuRe，表明"非均匀权重"本身也有收益，但 surprise 对齐带来额外增量。
  - Mean-normalized SuRe 在 MATH-500 上达 avg@4 69.20、pass@4 82.20，优于 vanilla。
  - Uplift-only（仅当 Δlog p_t>0 时才放大）也优于 vanilla，但不如均值归一化 SuRe。
  - 第二 seed（43）在 MATH-500 上复现：OPD 66.40 → SuRe 67.50，趋势一致。
- **GRPO 对比**（Appendix C.5）：SuRe 在 avg@k 上全面匹配或超越 GRPO（如 AMC23 avg@8 43.12 vs 41.56）；pass@k 在 AIME25 略落后，作者强调两者监督信号不同，属参考对比。

## 相关工作脉络
- **OPD 基础工作**（Agarwal et al., 2024; Lu & Thinking Machines Lab, 2025; Gu et al., 2024b）：建立 on-policy distillation 与逆 KL 动机；本文在此基础上分析"梯度如何在 token 间分配"，而不改变 rollout 源或 divergence。
- **MiniLLM / GKD**：逆 KL 因 mode-seeking 特性被引入；Yang 等（2026）进一步将 teacher log-ratio 与 KL-constrained RL reward 联系；本文与其互补，聚焦已实现梯度的 token 级分解。
- **Entropy-aware token selection**（Jin 等, 2026）：基于熵做 token 选择；本文指出 entropy/JSD 是非方向性的，而 |Δlog p_t| 是更强的梯度范数预测量。
- **概率阈值过滤**（Li 等, 2026; Ko 等, 2026）：通过概率过滤失败 token 或做 relaxed OPD；本文不做 hard 截断，而是用有界平滑权重持续放大 surprise。
- **RLVR 梯度集中分析**（Wang 等, 2025; Huang 等, 2026）：发现 RLVR 梯度集中于高熵少数 token；本文将此视角移植到 OPD 的 K2 梯度上，给出闭式分解并据此设计 SuRe。
- **DeepSeekMath / Kimi K3 / GLM-5 / MiniMo** 等多阶段 post-training 工程案例：展示 OPD 作为统一后训练 stage 的广泛应用背景。

## 局限性与未来方向
- 仅研究 sampled-token reverse-KL OPD；full-vocabulary distillation 与 JSD 作为优化目标的系统性对比未做。
- 实验集中在数学推理，reasoning trace 长度有限，难以外推到更长推理链或更广领域。
- 模型规模仅测到 1.7B/4B，更大参数 scale 上是否保持增益未知。
- SuRe 的对照未能完全分离"exact surprise 分配"与"非均匀权重本身"的收益（exact-shuffled 仍优于 vanilla）。
- 与 GRPO 等 RL 方法的比较属参考性质，未做等价目标下的严格 ablation。

## 研究启发与可借鉴点
1. **闭式梯度分解指导权重设计**：将反向传播的 realized gradient 分解为"gap × 几何项"的形式，可用于构造低成本 per-token reweight（无需额外模型），这是从理论到干预的直接路径。
2. **带符号 residual 优于熵/JSD 排序**：在"识别需重点学习的 token"任务中，|Δlog p_t| 比 entropy/JSD 更能预测梯度范数集中区，可作为后续 OPD/RLVR 混合方法的特征选择依据。
3. **detach + 有界放大避免训练不稳定**：SuRe 的 w_t 做 stop-gradient 且 w_t≥1，配合 token-mean 分母避免对 loss scale 的隐式重缩放；这种"单调、有界、单系数"设计易于调参且稳定。
4. **非均匀权重 vs 精确对齐的剥离策略**：通过 exact-shuffle / rank-reverse / uplift-only 等 matched control 区分"任意非均匀权重"和"surprise 对齐"的贡献，该 ablation 范式值得移植到其他 reweighting 方法。
5. **与小模型蒸馏流水线结合**：文中 1.7B/4B 学生蒸馏自 8B teacher，在 DeepMath hard 上提升显著；可探索将 SuRe 嵌入更广泛的多阶段 post-training pipeline（SFT→RL→OPD）进行端到端验证。

## 关键术语表
- **On-Policy Distillation (OPD)**：student 在其自身 sampled rollout 上接受 frozen teacher 的 per-token 监督信号进行蒸馏，避免 train-inference distribution shift。
- **K2 Estimator**：Schulman (2020) 提出的逆 KL 单样本无偏梯度估计，`L=½(Δlog p)²`，其 logit 梯度期望等于真实逆 KL 梯度。
- **Reverse KL**：`D_KL(π_S‖π_T)`，具有 mode-seeking 特性，区别于 forward KL 的 mass-covering。
- **Teacher–student gap (Δlog p_t)**：同一 context 下 teacher 与 student 对被采样 token 的对数概率之差，携带方向信息。
- **Surprise-aware Reweighting (SuRe)**：基于 student 采样概率的 detach 有界权重 `w_t=1+α(1−π̄_S)`，放大低概率（surprise）token 的更新量。
- **Checkpoint-shift**：同一 rollout 在不同 checkpoint（Base vs OPD）下的 log-prob 偏移，用于描述端点分布变化，非梯度分配的直接度量。
- **Jensen–Shannon Divergence (JSD)**：两分布的中点对称散度，值域 [0, log 2]，用于非方向性度量 token 分布差异。
- **Pass@k / Avg@k**：在 k 次采样中至少一次正确的比例（pass@k）与正确率的期望（avg@k），用于竞赛数学评估。

## 可复现要素
- **数据集**：DeepMath-103k hard split（57K, 难度≥6）；评测集 AIME24/AIME25/AMC23/MATH-500/CRUX/IFEval/MMLU-Pro；论文未明确声明数据集开源，DeepMath 在引用中给出 arXiv 编号。
- **代码/权重**：论文未提及代码开源声明；基线模型 Qwen3 系列可从 HuggingFace 获取。
- **关键超参**：lr=1e-6，batch=512，epochs=2，seed=42（主）/43（校验），rollout temperature=1.0/top-p=1.0，评估 temperature=0.7/top-p=0.9，max response length=8192，max model length=12288；SuRe 系数 α=1.0（主实验）及扫参 {0.2, 0.5, 2.0}。
