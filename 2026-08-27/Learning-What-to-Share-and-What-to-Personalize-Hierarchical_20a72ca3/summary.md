---
title: "Learning-What-to-Share-and-What-to-Personalize-Hierarchical"
source: https://arxiv.org/pdf/2608.25329v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 06:42:09"
---

# 论文速读：Learning-What-to-Share-and-What-to-Personalize-Hierarchical

## 一句话总结
提出HiPS框架，将Agent记忆管理策略解耦为全局共享的通用规则与用户自适应的个性化增量规则，通过对比蒸馏、发散门控与跨层规则流动实现策略与策略的动态协同进化，在多项个性化长对话基准上显著提升准确率与域外泛化能力。

## 研究问题与动机
- 现有记忆增强Agent多采用静态、一刀切的记忆管理策略，无法从交互反馈中学习，也无法适应不同用户的动态偏好演变。
- 直接为所有用户独立制定策略会导致冗余计算与冷启动问题，而完全共享策略又会使小众行为信号在群体平均奖励中被淹没。
- 冻结于训练前的显式策略会随策略优化产生偏差，缺乏在线证据驱动的动态校准机制。
- 核心问题：如何在无需预定义边界的情况下，动态发现并协同进化“共享”与“个性化”的规则划分。

## 核心贡献（创新点）
1. 形式化策略个性化问题，论证共享与专属边界必须通过在线证据实证发现。与传统方法预先设定或人工编写固定规则的做法本质不同，本文证明该边界是动态且任务驱动的。
2. 提出HiPS框架，将记忆管理策略解耦为全局通用层与用户自适应层，并设计跨层规则流动机制。区别于MemCoE等单次蒸馏后冻结策略的静态方案，本文实现策略与策略在统一RL循环中的持续协同进化。
3. 设计USD与PDD双模块，前者基于结构化Diff与证据生命周期迭代优化全局规则，后者通过预测增益(PG)与发散阈值门控精准触发个性化生成。与MemAgent等纯参数化RL方法不同，本文显式维护可解释的自然语言规则集，避免隐式策略的不可控性。
4. 构建反自证循环的奖励分离架构，严格隔离任务奖励与遵循奖励的下游用途。区别于直接以综合奖励驱动数据采样的主流做法，本文确保策略蒸馏仅锚定于客观任务结果，阻断自我强化的确认偏误。

## 方法详解
- **策略分层架构**：主动策略定义为 $S_p = S_u \cup \Delta_p$，其中 $S_u$ 为跨用户泛化的基础规则集，$\Delta_p$ 为仅对行为偏离人群适用的自适应规则。记忆更新方程为 $m_t = \pi_\theta(c_t, m_{t-1}; S_p)$。
- **USD（通用策略蒸馏）**：定期采样Top-k/Bottom-k轨迹进行对比分析，LLM元优化器输出结构化Diff $\delta_{\text{USD}} = \{\text{V}, \text{H}, \text{R}\}$，分别代表验证、假设新规则与修订/退役旧规则。每条规则携带证据等级（Tentative→Supported→Established），通过自动升/降级过滤噪声。
- **PDD（用户差异蒸馏）**：计算用户发散度 $D_p = \sum_{r \in S_u} |\mathrm{PG}(r|p) - \mathrm{PG}(r)|$，仅当 $D_p \geq \theta_{\text{div}}$ 时触发。生成的规则被强制约束为描述“管理行为”而非“事实内容”，确保推理时能泛化至未见用户。
- **Cross-Level Rule Flow**：Generalization（$\Delta_p \to S_u$）：当某 $\Delta_p$ 规则在 $\geq \theta_{\text{flow}}$ 比例的用户中达到Supported及以上，则提升为全局规则；Specialization（$S_u \to \Delta_p$）：当 $S_u$ 中某规则被降级时，为受影响用户生成专属替代规则，防止一刀切修订损害小众用户。
- **子模规则选择与奖励设计**：在Token预算 $B$ 下贪心最大化 $\sum_{r \in S} \mathrm{PG}(r) \cdot (1 - \max_{s \neq r} \sigma(r,s))$ 以实现覆盖度与多样性平衡。遵循奖励 $R_{\text{follow}}(\tau,p) = \frac{\sum \mathrm{PG}(r) \cdot h(\tau,r)}{\sum \mathrm{PG}(r)}$，总奖励 $R = R_{\text{ans}} + \lambda R_{\text{follow}}$。为防止自证循环，蒸馏Buffer仅按 $R_{\text{ans}}$ 排序，$R_{\text{follow}}$ 仅用于GRPO Advantage计算。

## 实验与结果
- **数据集与基线**：PersonaMem (32K/128K)、PrefEval (Explicit/Implicit)、PersonaBench (4个噪声水平)、PERMA (C-S/C-M/N-S/N-M)。基线涵盖长上下文直投、RAG、静态记忆库（Mem0/A-Mem/LightMem）及RL记忆Agent（MemAgent/MEM-α/MemSkill），主干模型为Qwen2.5-7B-Instruct。
- **主结果**：HiPS在全部12项评测中均取得最优。PersonaMem 128K达62.01%（较次优MemSkill 55.37%提升6.64pp）；PrefEval Explicit达89.20%；PERMA C-S达66.95%。长上下文下Long Context基线暴跌至20.74%，验证HiPS通过策略蒸馏有效过滤冗余信息。
- **消融发现**：域内（PersonaMem）性能最依赖USD与PG；域外（PERMA）性能最依赖Flow与Gate，移除Flow导致C-S暴跌至45.39（平均降19.4pp）。PDD单独贡献稳健但幅度适中（PersonaMem 128K降6.0pp，PERMA均降4.1pp）。
- **跨模型迁移**：用Qwen2.5-7B蒸馏的策略可直接注入GPT-4o-mini/Gemini 2.5 flash/GPT-5推理，HiPS ($S_u+\Delta_p$) 在GPT-5上达70.16，证明策略捕获的是模型无关的记忆管理原则。
- **扩展性**：128K对话下记忆库仅~1900 tokens，演化时间线性增长（140~1700秒），计算开销可控。

## 相关工作脉络
- **静态记忆库方法**（Mem0, A-Mem, LightMem）：依赖预定义提取/压缩规则或固定向量检索，缺乏从交互反馈中学习的能力；HiPS通过RL与LLM蒸馏实现策略动态演化。
- **RL
