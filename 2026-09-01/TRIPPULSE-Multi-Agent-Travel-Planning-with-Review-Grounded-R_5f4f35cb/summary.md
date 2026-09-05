---
title: "TRIPPULSE-Multi-Agent-Travel-Planning-with-Review-Grounded-R"
source: https://arxiv.org/pdf/2608.30924v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-05 18:45:30"
---

# 论文速读：TRIPPULSE-Multi-Agent-Travel-Planning-with-Review-Grounded-R

## 一句话总结
本文提出 TRIPPULSE，一种将旅游行程规划分解为五个领域专用智能体的多智能体框架，并通过将十万级真实用户评论蒸馏为结构化的 Pros/Cons 体验信号，结合全局协调器的预算递进分配与双轨调度后端，在严格满足时空/预算约束的同时生成高度个性化、体验导向的行程计划。

## 研究问题与动机
1. **单体提示的推理瓶颈**：现有 LLM 旅行规划器多采用单体提示（monolithic prompting），需同时处理实体选择、偏好推理、排程与约束满足，随行程复杂度上升极易产生幻觉与时空可行性违反。
2. **结构化属性无法捕捉体验信号**：传统方法仅依赖价格、房型、菜系等结构化字段，忽略了真实旅行决策中由评论揭示的舒适度、安全性、氛围、拥挤度等主观体验因素。
3. **评论数据规模与上下文负载的矛盾**：直接灌入海量非结构化评论会显著撑爆上下文窗口并降低规划可靠性，缺乏高效的评论特征提取与局部化注入机制。
4. **评测维度缺失体验与人本指标**：TripCraft 等基准主要考核约束满足率与路径分数，缺乏衡量个性化契合度、体验质量与风险规避能力的评估体系，难以反映真实用户满意度。

## 核心贡献（创新点）
1. **Review-augmented TripCraft 基准**：将 10万+ 条 Airbnb/TripAdvisor 真实评论蒸馏为结构化的 Pros/Cons 并合并入库，首次将高密度体验信号引入约束感知型旅行规划评测体系。
2. **TRIPPULSE 多智能体规划框架**：提出五领域专用智能体（住宿/交通/餐饮/景点/活动）+ 全局协调器的模块化架构，通过预算递进分配（85%/15%）与局部上下文隔离，解耦语义灵活性与结构可行性。
3. **RGPA（Review-Grounded Persona Alignment）评测指标**：设计 LLM-as-a-Judge 范式，从人设契合度、体验质量、风险规避与整体满意度四维量化评论增益，填补主观体验评测空白。
4. **神经-符号双调度后端对比**：提供 LLM-Based POI Scheduler 与 Deterministic Algorithmic Scheduler 两条路径，实证证明“LLM 语义选择 + 确定性算法排程”的混合架构在长程复杂规划中的可靠性优势。

## 方法详解
- **问题形式化**：查询定义为 $q = (c_s, C_d, W, k, B, C_{local}, \pi)$，目标是在满足预算、时序、本地约束的前提下，最大化 $\pi$ 对应的评论驱动体验质量。输出为时间有序行程 $I = (e_i, t_i^{start}, t_i^{end})_{i=1}^N$。
- **全局协调器（Global Orchestrator）**：采用混合序贯-并行调度。住宿、交通、餐饮智能体因共享全局预算池 $B$ 而顺序执行；景点与活动智能体仅做偏好排序，异步并行。预算分配采用确定性比例：住宿选定后，剩余预算的 85% 划为交通上限，15% 划为餐饮上限，避免递归重试。
- **五大领域智能体**：
  - `Accommodation Agent`：结合硬性本地约束（house rules、room type）与 Pros/Cons 筛选唯一住宿。
  - `Transportation Agent`：在全局视角锁定单一交通策略（flight/taxi/self-driving），确保跨城移动时序一致。
  - `Meals Agent`：基于餐饮预算上限与画像输出各城市餐厅排名列表。
  - `Attraction Agent` & `Events Agent`：并行执行，分别按体验信号（文化氛围/拥挤与安全风险）排序，且每日事件 ≤1 个。
- **双调度后端**：
  - `LLM-Based Scheduler`：先生成固定骨架 $S$，再由 LLM 填充餐饮/景点槽位，最后经 POI Scheduler 分配精确起止时间与换乘缓冲 $\Delta_{transit}$。
  - `Deterministic Algorithmic Scheduler`：将已选实体按排名贪婪填入骨架空闲时间窗，严格执行换乘缓冲、餐饮最小间隔 $\gamma = 240$ 分钟、时段窗口（早餐 8:00–10:30、午餐 12:00–15:51、晚餐 18:30–22:30、景点 9:00–19:00）等规则，数学保证可行性。
- **评论蒸馏流水线**：对每个实体聚合最多 5 条评论，使用 Qwen3-4B-Instruct 零样本贪婪解码（max_new_tokens=256）提取 Pros/Cons JSON；解析层修正格式偏差，失败样本替换为空列表，最终 merge 回数据库。
- **时序指标修正**：原始 $T_a$ 因 Poisson 概率质量函数最大值 <1 导致得分被人为压低（约 0.35），本文引入归一化 $\tilde{T}_a = \frac{1}{n}\sum_i \exp\left(-\frac{(d_i-\mu_d^i)^2}{2\sigma_d^2}\right) \cdot \frac{P(N=n)}{\max_k P(N=k)}$ 使理想行程得分为 1.0。

## 实验与结果
- **数据集与模型**：在 3/5/7 天行程的增强版 TripCraft 上评估；测试 GPT-5、Llama-3.1-70B-Instruct、DeepSeek-R1-Distill-Qwen-14B、Phi-4、Mistral-Nemo、Qwen-2.5-7B-Instruct，全部 zero-shot，无微调。
- **约束满足与时效指标**：确定性格子调度器在所有模型与天数下 Delivery Rate (Del) 接近 100%，Hard Constraint Pass Rate (HCPR) 维持高位（如 GPT-5 3/5/7 天 HCPR 分别为 98.92%、98.
