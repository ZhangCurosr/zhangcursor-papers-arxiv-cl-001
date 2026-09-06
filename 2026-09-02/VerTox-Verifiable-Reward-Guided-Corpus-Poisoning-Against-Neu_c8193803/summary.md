---
title: "VerTox-Verifiable-Reward-Guided-Corpus-Poisoning-Against-Neu"
source: https://arxiv.org/pdf/2609.01325v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 16:43:12"
field: "检索系统安全与对抗鲁棒性"
keywords: ["corpus poisoning", "neural ranking models", "RLVR", "adversarial attacks", "information retrieval security", "RAG robustness"]
innovations: ["首个将语料库中毒建模为 RLVR 问题的框架，通过 GRPO 微调小型 LLM 生成对抗文档", "设计联合排名扭曲与事实破坏的三原子可验证奖励，实现从可见性到信息污染的完整攻击链", "证明子优化 dense retriever 代理训练的生成器可跨密集、稀疏、交叉编码器及商业嵌入模型高效迁移"]
benchmarks: ["MS MARCO", "TREC DL 2019/2020", "BEIR (NQ, FiQA, Touché-2020, TREC-COVID, SciFact, NFCorpus)"]
---

# 论文速读：VerTox-Verifiable-Reward-Guided-Corpus-Poisoning-Against-Neu

## 一句话总结
本文提出了 VerTox，首个将语料库中毒攻击形式化为可验证奖励强化学习（RLVR）问题的框架，通过联合优化排名扭曲与事实破坏的奖励设计，使小型 LLM 生成高度流畅且难以检测的对抗性文档，实现对神经排序模型的近乎完美攻击成功率（ASR ≈ 1.0）及显著下游 RAG 性能退化。

## 研究问题与动机
- **神经排序模型的安全盲区**：当前检索系统广泛采用神经排序模型（如密集检索、交叉编码器），但其高相关性分数并不意味着文档可信，LLM 可大规模生成语义丰富但事实错误的内容，利用模型对表面相关性的偏好实施攻击。
- **现有攻击方法的局限性**：词替换（如 HotFlip）计算成本高且降低流畅性；嵌入扰动（如 EMPRA）跨架构迁移性差；直接提示 LLM 虽流畅但缺乏显式排名优化信号，难以稳定超越顶级文档。
- **实用威胁模型的缺口**：现实攻击者类似普通用户，无法修改已有语料库或内部基础设施，仅能通过查询观察、搜索评估、注入新文档实施攻击，现有工作对这一约束下的攻击能力评估不足。
- **攻击效果的评估片面性**：既有研究过度依赖 ASR 指标，未充分考察对抗文档的实际可见性（Top@1/Top@10）、流畅性、事实破坏程度及其对下游 RAG 应用的连锁影响。

## 核心贡献（创新点）
- **首个 RLVR 框架用于语料库中毒**：将对抗文档生成建模为可验证奖励的强化学习问题，通过 GRPO 微调小型 LLM（0.6B-2B），与直接提示 8B 模型相比获得更高 Top@1 且推理更快。
- **联合优化排名扭曲与事实破坏的三原子奖励设计**：引入归一化密集排名边际奖励（R_rank）、双向 NLI 幻觉检测的事实破坏奖励（R_fact）、以及防查询堆砌的部分 Levenshtein 重复惩罚（R_rep），实现从"仅提升排名"到"同时污染信息"的攻击升级。
- **跨架构黑盒迁移性的实证**：在 SimLM-base（白盒代理）上训练的生成器，可成功攻击 BGE-base、Cohere 商业嵌入、SPLADE（稀疏）、RepLLaMA、RankLLaMA（交叉编码器）等更强且异构的排序模型，证明不同架构共享可被利用的相关性信号。
- **端到端 RAG 影响评估**：首次在 FlashRAG 环境下验证对抗文档对下游生成的实际危害，显示即使仅注入少量中毒文档，问答准确率也可从 0.70 降至约 0.30。

## 方法详解
- **RLVR 优化框架**：策略 π_θ 在输入 (q, d⁺) 条件下生成对抗文档 d̃，通过确定性验证器返回标量奖励 R(q, d⁺, d̃)，目标函数为 J(θ) = E[R]。使用 GRPO 进行策略优化，采样 G 个候选组内计算归一化优势 A_i，采用 clipped surrogate objective 稳定训练，省略 KL 惩罚项以增强探索。
- **排名扭曲奖励 R_rank**：使用本地代理密集检索器 s_φ 计算原始排名边际 Δ_rank = s_φ(q, d̃) − s_φ(q, d⁺)，经剩余分数头空间归一化后映射为连续奖励：R_rank = 2·σ(α·Δ̄_rank) − 1，其中 σ 为 sigmoid 函数、α=5 控制陡峭度，提供从"接近"到"超越"的密集反馈信号。
- **事实破坏奖励 R_fact**：采用 NLI 幻觉检测器 HHEMv2，计算双向事实一致性 h_ψ(d⁺, d̃) 与 h_ψ(d̃, d⁺)，定义 R_fact = ½[(1−h_ψ(d⁺, d̃)) + (1−h_ψ(d̃, d⁺))]，迫使生成文档不仅保留主题相关性，还需实质性篡改查询相关事实，避免仅追加或删除的平凡操作。
- **查询重复惩罚 R_rep**：使用部分 Levenshtein 相似度 s_q(d) 衡量文档与查询的词汇重叠，计算超出良性基准的冗余重叠 Δ_rep = max{0, s_q(d̃) − s_q(d⁺)}，经剩余相似度归一化后通过 sigmoid 映射为 [0,1] 区间惩罚：R_rep = 2·σ(β·Δ̄_rep) − 1，β=5。
- **长度正则化 R_len**：二元惩罚文档词数偏离良性参考的倍数范围 [η_min, η_max]，防止极端长度异常。
- **综合奖励**：R = λ_rank·R_rank + λ_fact·R_fact − λ_rep·R_rep − λ_len·R_len，权重设为 λ_rank=λ_fact=λ_rep=1.0、λ_len=0.3。

## 实验与结果
- **数据集**：训练集为 MS MARCO 的 3000 对 (query, passage)；评测集为 TREC DL 2019/2020（域内）及 BEIR 六个子集 NQ、FiQA、Touché-2020、TREC-COVID、SciFact、NFCorpus（域外）。
- **评估基线**：RandomToken（随机替换 30% token）、HotFlip（梯度词替换）、EmbedPerturb（嵌入空间扰动）、Prompting（直接提示 Qwen₃-8B thinking 模式）。
- **目标排序模型**：白盒 SimLM-base；黑盒 BGE-base、Cohere-embed-english-v3.0（商业）、SPLADE、RepLLaMA、RankLLaMA。
- **Top-1 攻击核心结果**：
  - 域内白盒：VerTox(Gemma₂-2b-it) ASR=1.00、Top@1=0.84，显著优于 Prompting(0.99/0.54)；
  - 域外白盒：Gemma₂-2b-it 平均 Top@1=0.80，较 Prompting 的 0.49 提升 63%；
  - 黑盒跨架构：Gemma₂-2b-it 在 Cohere 上平均 Top@1=0.89，BGE-base 上 0.82，RankLLaMA 上 0.68，全面超越所有基线。
- **Top-10 扩展**：随 k 增加 nDCG@10 持续下降，BGE-base、SPLADE、RepLLaMA 在 Top-10 攻击下退化严重，证明攻击可规模化放大。
- **RAG 下游影响**：原始上下文 Qwen₃-8B 准确率达 0.70，中毒上下文降至约 0.30，跨 k∈{1,3,5,7,10} 均保持低准确率。
- **流畅性验证**：Dale-Chall 可读性得分仅比良性文档高 0.3 分，远低于 HotFlip 的 +7.0 分，表明对抗文档难以被语言信号过滤。
- **消融实验**：密集排名奖励（归一化边际）显著优于二值成功/失败奖励，在所有数据集与模型上平均 Top@1 提升明显。

## 相关工作脉络
- **词替换类攻击**：PRADA、MCARA、TARA、HotFlip 通过梯度或搜索替换 token 操纵排序，但计算成本高（单次优化可达数百秒/查询）且降低文本流畅性，VerTox 以端到端生成替代逐 token 扰动。
- **嵌入扰动类攻击**：Vec2Text、EMPRA 在嵌入空间优化后解码回文本，存在拓扑结构模型依赖、跨架构迁移差、小扰动放大为语法错误等问题，VerTox 完全在文本空间优化规避该缺陷。
- **LLM 生成对抗内容**：AttChain 使用思维链迭代修改，扩展性受限；PoisonedRAG 聚焦答案操纵并重复杂询，未针对神经排序器优化；VerTox 是首个将 LLM 中毒建模为 RLVR 问题并联合优化排名+事实的工作。
- **RAG 安全研究**：Cuconasu 等提出噪声增强检索，Shi 等研究无关上下文干扰，本工作从攻击视角揭示 RAG 在中毒语料下的脆弱性链条。
- **RLVR 在 NLP 中的应用**：Lambert、Guo 等将可验证奖励用于数学/推理任务，本文首次将其引入信息检索安全领域，证明奖励塑造对攻击生成器的有效性。

## 局限性与未来方向
- **威胁模型限制**：仅针对开放检索环境（如网页搜索），防火墙保护的封闭语料库未覆盖。
- **防御机制缺失**：未评估重复过滤、垃圾检测、内容审核、对抗重训练等常见防御手段的有效性。
- **多代理训练未探索**：当前使用单一 dense retriever 代理，多代理或人类事实评估可增强迁移性分析。
- **RAG 评估规模有限**：仅 100 个 FlashRAG 查询、两个回答模型、一个 LLM judge，跨领域与多 judge 扩展必要。
- **未来方向**：可控攻击（针对特定实体/声明）、多轮 agent 化攻击（适应中间查询与动作）、联合相关性与事实一致性的防御机制。

## 研究启发与可借鉴点
- **奖励塑造的细粒度设计**：将"排名逼近"转化为归一化边际的连续 sigmoid 奖励，而非二值成功信号，显著提升 RL 训练稳定性与跨架构泛化，可迁移至其他对抗生成任务。
- **双向 NLI 事实一致性度量**：采用 h_ψ(d⁺, d̃) 与 h_ψ(d̃, d⁺) 的平均不等式，有效识别实质性事实篡改而非平凡编辑，该信号可用于防御侧的事实漂移检测。
- **小规模 LLM 经 RLVR 超越大模型提示**：0.6B-2B 参数模型经 RL 微调后 Top@1 显著优于 8B 直接提示，说明任务特定奖励比模型规模更重要，低成本对抗生成具现实威胁。
- **黑盒迁移的实证价值**：在子优化代理上训练的生成器可攻击更强的异构排序器，提示防御方应以最严苛假设评估安全性。
- **评估指标体系建议**：单一 ASR 易虚高（仅要求超越最低相关文档），Top@1/Top@10、流畅性（PPL、Dale-Chall）、下游应用影响应联合报告。

## 关键术语表
- **Corpus Poisoning（语料库中毒）**：攻击者向检索语料库注入恶意文档，操纵排序结果使目标文档获得更高排名的攻击类型。
- **RLVR（Reinforcement Learning with Verifiable Rewards）**：使用自动可计算的确定性验证器返回奖励信号、无需人类标注的强化学习范式。
- **GRPO（Group Relative Policy Optimization）**：DeepSeekMath 提出的策略优化算法，通过组内相对优势归一化与 clipped surrogate 稳定训练，省略 KL 惩罚项。
- **ASR（Attack Success Rate）**：对抗文档超过最低排名相关文档的比例，衡量攻击覆盖广度。
- **HHEMv2（Hallucination Evaluation Model v2）**：基于 NLI 的事实一致性检测器，用于量化对抗文档对良性证据的事实偏离程度。
- **Dale-Chall Readability Score**：衡量文本可读性的标准指标，分数越高表示越难理解，用于评估对抗文档的隐蔽性。
- **Top@k**：对抗文档出现在检索结果前 k 位的比例，直接反映用户可见性与攻击严重程度。
- **Normalized Ranking Margin**：将对抗文档与良性文档的分数差除以剩余分数空间，实现跨查询可比且 bounded 的奖励信号。

## 可复现要素
- **数据集**：MS MARCO（训练）、TREC DL 2019/2020、BEIR（NQ、FiQA、Touché-2020、TREC-COVID、SciFact、NFCorpus）；论文未明确声明数据集公开状态，MS MARCO 与 TREC DL 为公开基准，BEIR 子集亦为开源。
- **代码**：已开源，GitHub 地址：https://github.com/zhiqihuang/vertox-corpus-poisoning。
- **模型权重**：Generator 使用 Qwen₃-0.6B、Llama₃.₂-1B-Instruct、Gemma₂-2b-it 的 Hugging Face 开源 checkpoint；Performer 使用 vectara/hallucination_evaluation_model（HHEMv2）。
- **关键超参**：LoRA rank=16、scaling=32、dropout=0.05、lr=1e-5、mini-batch=4、gradient accumulation=4、epochs=5；奖励权重 λ_rank=λ_fact=λ_rep=1.0、λ_len=0.3；sigmoid 陡峭度 α=β=5。
