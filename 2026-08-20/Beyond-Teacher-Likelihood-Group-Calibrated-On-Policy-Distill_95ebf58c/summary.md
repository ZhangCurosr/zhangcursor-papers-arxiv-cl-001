---
title: "Beyond-Teacher-Likelihood-Group-Calibrated-On-Policy-Distill"
source: https://arxiv.org/pdf/2608.19181v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-01 16:06:09"
field: "长上下文推理与知识蒸馏"
keywords: ["on-policy distillation", "long-context reasoning", "verifier calibration", "teacher-verifier disagreement", "group-relative normalization", "credit assignment"]
innovations: ["构建group-normalized teacher-verifier signed residual校准分歧", "RACA基于相对OPD优势进行token级信用分配", "无需额外forward pass即可整合response-level验证器反馈与dense token-level指导"]
benchmarks: ["DocMath", "Frames", "MRCR", "CorpusQA", "LBv1QA"]
---

# 论文速读：Beyond-Teacher-Likelihood: Group-Calibrated On-Policy Distillation for Long-Context Reasoning

## 一句话总结
本文提出了GC-OPD（Group-Calibrated On-Policy Distillation），通过在校内组内归一化的验证器奖励与轨迹级OPD得分之间构建符号残差，校准教师与验证器之间的分歧，在保留密集token级指导的同时纠正全局任务偏好不一致的问题；在Qwen3-4B/8B上五个长上下文基准的平均分分别提升1.16分和1.09分。

## 研究问题与动机
- **核心问题**：在长上下文推理任务中，token级别的教师支持（OPD）倾向于偏好局部合理的响应，而这些响应可能遗漏分散在输入中的关键证据或违反全局任务约束，导致教师偏好与任务验证结果不一致。
- **现有方法不足**：
  - 已有verifier-aware OPD方法（SCOPE、MOPD、Uni-OPD等）通过目标路由、teacher条件化、返回校准或token门控引入验证器反馈，但无法同时保留组内相对梯度差异并转化为token依赖校准。
  - 直接添加归一化验证器奖励会强化OPD已表达的相对偏好，而非聚焦分歧本身。
  - 轨迹级OPD得分与验证器奖励的排序随输入长度增加而越来越不一致（在多表提取和高召回检索任务中，分歧率从~35-40%升至60%+）。

## 核心贡献（创新点）
1. **发现teacher-verifier分歧模式**：在两个需要分布证据聚合的长上下文任务中，轨迹级OPD得分与验证器奖励的对齐度随输入长度增加显著下降，pairwise disagreement rate从~40%升至64%。
2. **提出GC-OPD框架**：通过在rollout组内分别归一化验证器奖励和轨迹级OPD得分，构建符号残差ρ=R̃−s̃，聚焦于教师与验证器的分歧而非直接叠加奖励。
3. **设计RACA信用分配**：Relative-advantage-based credit assignment根据token相对OPD优势（保留符号和排序）将残差分配到token级别，而非均匀分配或使用绝对值。
4. **系统验证有效性**：在Qwen3-4B/8B上五个长上下文基准，GC-OPD超越vanilla OPD及其他基线方法（ExOPD、Uni-OPD、FiRe-OPD、PowerOPD）。

## 方法详解
- **Group-Relative Assessments**：对每个rollout组内的验证器奖励R⁽ⁱ⁾和轨迹级OPD得分s⁽ⁱ⁾分别做z-score归一化：R̃⁽ⁱ⁾=z(R⁽ⁱ⁾)，s̃⁽ⁱ⁾=z(s⁽ⁱ⁾)。
- **Teacher-Verifier Disagreement Residual**：构建符号残差ρ⁽ⁱ⁾=R̃⁽ⁱ⁾−s̃⁽ⁱ⁾，当教师与验证器偏好一致时残差为0，分歧越大残差绝对值越大。若组内标准差接近零则设残差为0。
- **RACA（Relative-Advantage-Based Credit Assignment）**：
  - 计算token相对OPD优势：uₜ⁽ⁱ⁾=(Aₜ⁽ⁱ⁾−s⁽ⁱ⁾)/σₐ⁽ⁱ⁾，保留符号和组内排序。
  - 映射为正定有界信用：cₜ⁽ⁱ⁾=1+tanh(uₜ⁽ⁱ⁾/2)∈(0,2)。
  - GC-OPD优势：A'ₜ⁽ⁱ⁾=Aₜ⁽ⁱ⁾+β·cₜ⁽ⁱ⁾·ρ⁽ⁱ⁾，其中β=0.10为残差系数。
- **关键设计**：保留原始OPD信号作为base，仅通过残差修正；无需额外forward pass；使用τ_G=10⁻⁶和τ_T=10⁻⁶作为数值稳定性阈值。

## 实验与结果
- **数据集**：GoLongRL子集（9,527 prompts，≤32K tokens），包含精确长距检索、证据推理等多任务。
- **评估基准**：DocMath、Frames、MRCR、CorpusQA、LBv1QA。
- **主要结果**：
  - Qwen3-4B：Raw 29.08 → OPD 39.31 → GC-OPD **40.47**（+1.16 over OPD）
  - Qwen3-8B：Raw 35.12 → OPD 43.56 → GC-OPD **44.65**（+1.09 over OPD）
  - CorpusQA提升最大（4B: +5.77, 8B: +3.95），集中体现于证据聚合任务。
- **消融结论**：
  - 符号残差优于直接添加归一化奖励（44.65 vs 44.19）和额外OPD项（44.65 vs 43.60）。
  - RACA优于Uniform（44.65 vs 44.28）和Absolute OPD（44.65 vs 43.93）。
- **训练配置**：100 steps，batch=32，每prompt 8 responses，β=0.10，lr=1e-6，8×H800/H100。

## 相关工作脉络
- **OPD基线**：vanilla OPD [1]、ExOPD [31]（teacher-reference外推）、Uni-OPD [9]（outcome margin校准）、FiRe-OPD [16]（trajectory过滤重加权）、PowerOPD [34]（bounded power transformation）。
- **Verifier-aware方法**：SCOPE [36]（dual-path adaptive weighting）、MOPD [32]（peer successes/failures）、SG-OPD [29]（sign-gated）。
- **定位差异**：GC-OPD不修改teacher pass，而是构建group-relative residual聚焦分歧；不依赖step labels或auxiliary rollouts；保留graded within-group差异。
- **Long-context RL**：LongRLVR [5]、LoongRL [27]、QwenLong-L1 [25]（侧重数据/curriculum/reward构造）。

## 局限性与未来方向
- **自述局限**：
  - 分歧诊断仅在固定响应集上测量，未证明训练因果效应。
  - 残差系数β仅在High-Recall任务holdout上选择，可能不适配所有任务。
  - 未处理教师-学生上下文长度不匹配（教师用原生窗口，学生用YaRN扩展）。
- **可推断方向**：
  - 扩展到多步reasoning/chain-of-thought场景。
  - 探索动态β或任务自适应校准策略。
  - 结合process supervision获取更细粒度credit assignment。
  - 研究分歧与模型规模、训练步数的关系。

## 研究启发与可借鉴点
1. **Group-relative校准范式**：在校内组内归一化后再比较不同信号的分歧，避免scale差异干扰；可迁移到其他multi-signal融合场景。
2. **符号残差设计**：用差值而非直接叠加聚焦分歧，避免强化已有偏好；适用于任何teacher-student-verifier三方校准任务。
3. **相对优势信用分配**：用相对OPD优势（保留符号）而非绝对值分配token权重，避免负优势token被过度强化；可借鉴到RLHF/RLAIF的credit assignment。
4. **无额外forward pass**：仅需聚合和elementwise变换，计算开销极小，适合大规模训练集成。
5. **诊断先行**：先量化teacher-verifier分歧随输入长度的变化趋势，再设计干预方法，避免盲目建模。

## 关键术语表
**On-Policy Distillation (OPD)**：在student生成的自一致响应用dense token-level teacher指导进行蒸馏的训练范式。
**Teacher-Verifier Disagreement**：轨迹级OPD得分（教师偏好）与任务验证器奖励（任务结果）之间的排序或数值不一致。
**Group-Relative Calibration**：在rollout组内对验证器奖励和OPD得分分别做z-score归一化后再比较，消除scale差异。
**Signed Residual (ρ)**：组归一化验证器奖励与组归一化OPD得分之差，量化teacher-verifier分歧的方向和幅度。
**RACA (Relative-Advantage-Based Credit Assignment)**：根据token相对OPD优势（保留符号）将轨迹级残差分配到token级别的信用分配机制。
**GoLongRL**：面向长上下文强化学习能力的多任务对齐数据集，包含9个任务族和binary/graded验证器。
**YaRN**：用于扩展LLM上下文窗口的位置插值技术，本文scaling factor=4将context扩展到131K。

## 可复现要素
- **数据集**：GoLongRL [20]，训练子集9,527 prompts（≤32K tokens），评估基准DocMath、Frames、MRCR、CorpusQA、LBv1QA。
- **代码开源**：https://github.com/SolereZhang/GC-OPD
- **模型权重**：Qwen3-4B、Qwen3-8B官方checkpoint；教师Qwen3-30B-A3B-Thinking-2507。
- **关键超参**：β=0.10，τ_G=τ_T=10⁻⁶，a_max=10，rollout temperature=1，top-p=1，lr=1e-6，batch=32，responses=8，steps=100。
- **硬件**：8×80GB H800/H100 GPU。
