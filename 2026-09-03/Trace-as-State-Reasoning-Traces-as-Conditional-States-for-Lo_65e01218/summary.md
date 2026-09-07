---
title: "Trace-as-State-Reasoning-Traces-as-Conditional-States-for-Lo"
source: https://arxiv.org/pdf/2609.02702v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-07 05:28:21"
field: "长上下文语言模型推理"
keywords: ["长上下文推理", "推理轨迹", "条件状态更新", "因果注意力", "推理缩放", "文本状态代理"]
innovations: ["通过条件状态更新任务的形式化分析揭示因果处理器中条件优先的指数级内存优势", "提出TRACE AS STATE方法，将推理轨迹作为文本状态代理前置以实现跨pass状态反馈"]
benchmarks: ["GraphWalks 256K", "MRCRv2 8-needle", "NUB-1M Season 2"]
---

# 论文速读：Trace-as-State-Reasoning-Traces-as-Conditional-States-for-Long-Context-Transformers

## 一句话总结
本文提出 **TRACE AS STATE**，将大模型推理过程中生成的推理轨迹作为"任务状态"的文本代理，放置在长上下文块之前进行二次因果处理，从而显著提升长上下文推理能力。该方法基于条件状态更新任务的理论分析，证明了条件先于信息序列时因果处理器的内存需求呈指数级降低。

## 研究问题与动机
- **因果处理与信息顺序的结构性错配**：Transformer采用因果注意力，后出现的信息无法影响前面token的表示。然而某些长上下文任务（如图遍历、引用解析）所需的关键状态可能在处理完整上下文后才能被发现。
- **现有方法对推理轨迹的利用不足**：尽管推理轨迹（reasoning traces）携带中间计算信息，但已有工作多将其视为解释或仅用于最终答案生成，未系统性地将其作为可跨pass复用的"文本状态代理"。
- **输入顺序对长上下文推理性能有显著影响**：已有研究表明，支持信息的相对呈现顺序会影响推理结果，但现有长上下文架构改进多聚焦于容量扩展，忽视了顺序优化的潜力。
- **理论上的内存复杂度差异**：对于条件状态更新任务，条件先出现时只需跟踪单个状态路径（O(log|S|) bits），而条件后出现时需保留所有可能响应配置（最坏O(|S|·log|S|) bits），存在指数级差距。

## 核心贡献（创新点）
- **提出条件状态更新任务的形式化分析**，证明了对因果处理器而言，条件先于信息序列相比后于信息序列在最坏情况下可将内存需求从O(|S|·log|S|)降至O(log|S|)，揭示了输入顺序的根本性影响。
- **引入TRACE AS STATE推理缩放方法**，将模型生成的推理轨迹序列化后作为文本状态代理，放置在长上下文之前进行二次因果pass处理，在不改变单pass内因果结构的前提下实现了跨pass的状态反馈。
- **设计严格匹配的对照方法TRACE APPEND**，将相同轨迹置于上下文之后，证明了顺序放置而非轨迹本身才是性能提升的核心因素——27组实验中TRACE AS STATE在26组上优于TRACE APPEND。
- **在三个前沿模型（DeepSeek V4 Pro Preview、GLM-5.2、Qwen 3.7 Max）和三个长上下文数据集（GraphWalks、MRCRv2、NUB-1M）上验证了方法的通用性**，在GraphWalks Parents任务上将GLM-5.2从83.2%提升至100.0%的Exact Match。

## 方法详解
- **因果状态更新处理器的形式化定义**：设信息序列C=(c₁,...,cₙ)，处理器维护有限状态空间S中的任务状态sᵢ，更新规则为sᵢ = U(sᵢ₋₁, cᵢ)，初始状态s₀由条件z决定。若z先于C到达（条件优先顺序[z, C]），处理器只需跟踪当前状态sᵢ；若z后于C到达（[C, z]），处理器在处理C时必须保留对所有可能z的响应映射，最坏需存储|S|^|S|种不同函数。
- **推理轨迹作为文本状态代理**：对同一问题运行n_tr次，每次生成推理轨迹rⱼ和可见答案aⱼ，通过固定序列化器π将轨迹拼接为T=π(r₁,...,rₙ_tr)，保留源顺序并添加固定标签和分隔符（如<trace_start>/<trace_end>）。
- **TRACE AS STATE的处理流程**：首次pass M([x, q])生成推理轨迹和答案；序列化后得到T；第二次pass M([T, x, q])将T置于长上下文x之前，使模型在重新处理x时能利用T中携带的任务状态信息。
- **TRACE APPEND对照设计**：第二pass为M([x, T, q])，保持x和T的原始顺序，T仅在x处理完成后才能影响后续推理和答案生成，无法改变x中已有token的表示。
- **实际实现细节**：每个问题进行5次repeat以收集5条轨迹；超长轨迹截断至前50,000字符；serializer保留固定前缀提示"Below are selected reasoning traces... They may contain mistakes. Use them only as scratchpad hints"；问题q始终置于prompt末尾以确保模型专注。

## 实验与结果
- **数据集**：GraphWalks 256K（图遍历，含BFS和Parents子任务）、MRCRv2 8-needle（256K/512K两档）、NUB-1M Season 2（百万token级别小说阅读理解）。
- **模型**：DeepSeek V4 Pro Preview（CSA+HCA+mHC混合注意力）、GLM-5.2（1M-token MoE + DSA + IC）、Qwen 3.7 Max（GDN+GA）。
- **主要结果**：TRACE AS STATE在27组model×task×metric组合中胜过了26组。
  - **GraphWalks Parents最强结果**：DeepSeek V4 Pro从43.0%（TRACE APPEND）→81.8%（TRACE AS STATE，+38.8pp）；Qwen 3.7 Max从87.2%→99.1% F1；GLM-5.2从83.2%→**100.0%** EM/F1。
  - **GraphWalks BFS**：DeepSeek V4 Pro从36.4%→65.9% F1；GLM-5.2从70.7%→75.0% F1。
  - **MRCRv2 256K**：DeepSeek V4 Pro从78.5%→88.7% EM。
  - **NUB-1M**：DeepSeek V4 Pro从71.0%→73.0% accuracy。
- **消融实验关键发现**：
  - Question First（仅将问题前置）在Parents上有一定提升但远低于TRACE AS STATE。
  - Re2（重复上下文）优于首pass但仍显著低于TRACE AS STATE。
  - Answer Feedback（仅放答案）效果远弱于TRACE AS STATE，说明推理文本比答案本身携带更多有用状态信息。
  - Random Trace（用无关问题的轨迹）显著差于首pass，排除了格式效应的解释。
  - Trace Only（无原始上下文）表现接近TRACE APPEND，说明原始输入在轨迹后仍然重要。
  - TRACE AS STATE超越Oracle@5（从5个首pass答案中选最优），证明二次pass本身产生了真正改进。
  - 轨迹数量ablation：n_tr从1到5逐步提升，TRACE AS STATE始终优于TRACE APPEND。

## 相关工作脉络
- **Ok & Lee (2026)** 发现因果注意力导致多选择提示顺序敏感性，通过重复选项缩小差距；本文在此基础上将"重复/反馈"从选项扩展为完整的推理轨迹，并从理论上证明条件优先的指数级优势。
- **Re2 (Xu et al., 2024)** 通过重复问题实现二次阅读；本文证明仅重复问题或重复上下文的效果均弱于使用推理轨迹作为状态代理并将其前置。
- **The Markovian Thinker (Aghajohari et al., 2026)** 和 **ReContext (Zhao et al., 2026)** 通过文本历史或证据池跨chunk传递状态；本文的核心差异在于不修改模型架构，仅通过输入顺序优化即可实现跨pass反馈。
- **State over Tokens (Levy et al., 2025)** 描述推理前缀为外部化计算状态，但指出模型未必可靠使用自身中间步骤；本文从实证角度验证了推理轨迹确实可作为有用状态代理，但需在正确时序下使用。
- **Racing Thoughts (Lepori et al., 2025)** 将上下文化错误归因于层间竞争；本文从更底层的条件状态更新理论视角分析顺序敏感性，提供了不同于层间分析的互补视角。
- **CoRe (Yu et al., 2025)** 通过重复完整上下文减少支持文档顺序敏感性；本文与CoRe本质区别在于：CoRe重复上下文以缓解顺序问题，本文通过前置状态代理主动引导二次阅读。

## 局限性与未来方向
- **接口依赖**：当前实现需要模型暴露推理轨迹接口；仅提供最终答案的API需设计替代状态接口。
- **推理开销**：额外pass增加延迟和token消耗，且前置状态会减少多轮对话中的KV缓存复用。
- **评估范围有限**：仅测试了三个因果Transformer模型和三个长上下文任务家族，未涉及多轮agent任务。
- **轨迹质量**：推理轨迹可能不完整或有错误，当前方法将其视为"有损代理"，未做轨迹筛选或纠错。
- **未来方向**：轨迹选择与压缩、学习更优的文本状态接口、联合训练模型与反馈接口、扩展至交互式和agent场景。

## 研究启发与可借鉴点
- **条件优先原则可迁移**：在任意需要维持任务状态且状态迟到的因果处理场景中，可考虑将"已知状态"前置而非后置，这一原则不限于NLP，也可应用于程序合成、代码执行等需要维护运行时状态的领域。
- **推理轨迹的文本化复用策略**：将模型中间输出（不仅是CoT，还可是日志、中间变量等）序列化为结构化文本并前置，是一种零架构修改的推理增强技术，可快速集成到现有推理pipeline中。
- **严格的对照设计值得借鉴**：TRACE AS STATE vs TRACE APPEND仅差一个顺序变量，配合Question First、Answer Feedback、Random Trace等多层消融，有效排除了"重复""格式""答案访问"等替代解释，这种实验设计范式适用于其他推理增强方法验证。
- **轨迹数量的缩放效应**：ablation显示随着n_tr增加性能单调提升，说明推理轨迹质量+数量的联合优化是一个有潜力的缩放维度，可探索动态轨迹选择策略。
- **与团队方向的结合机会**：若团队关注长上下文推理效率，可探索将TRACE AS STATE与压缩注意力、KV缓存优化等技术结合，在保持性能增益的同时降低额外pass的计算开销。

## 关键术语表
- **TRACE AS STATE**：将推理轨迹作为文本状态代理并置于长上下文之前的二次推理方法。
- **TRACE APPEND**：将推理轨迹置于长上下文之后的对照方法，用于验证顺序效应。
- **条件状态更新任务（Conditional State Update Task）**：形式化模型，其中处理器需根据条件z初始化状态s₀，再依次处理信息序列C更新状态。
- **因果状态更新处理器（Causal State Update Processor）**：按输入顺序逐次读取信息单元并更新持久工作记忆的处理器。
- **推理轨迹（Reasoning Trace）**：模型在生成最终可见答案之前产生的中间推理文本，包含逐步计算过程。
- **序列化器（Serializer π）**：将多条推理轨迹拼接为固定格式文本的工具，保留源顺序并添加标记。
- **KV缓存（Key-Value Cache）**：Transformer推理中缓存各token的key和value以减少重复计算的结构，本文指出前置轨迹可能降低其复用率。
- **Residual Map**：给定信息序列C后，从每个可能条件z出发所确定的最终状态映射φ_C: S→S。

## 可复现要素
- **数据集**：GraphWalks（HuggingFace公开）、MRCRv2（HuggingFace公开）、NUB-1M（GitHub公开）。
- **代码/权重**：论文未提及开源代码；模型通过官方API调用（DeepSeek、Qwen/Alibaba Cloud、GLM/Bigmodel）。
- **关键超参**：n_tr=5（每条问题收集5条推理轨迹）；轨迹截断长度50,000字符；推理effort设为max/xhigh；5次repeat取均值。
