---
title: "VerTox-Verifiable-Reward-Guided-Corpus-Poisoning-Against-Neu"
source: https://arxiv.org/pdf/2609.01325v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 16:43:11"
field: "信息检索安全与对抗鲁棒性"
keywords: ["corpus poisoning", "neural ranking models", "RLVR", "adversarial attacks", "information retrieval", "RAG security", "reward shaping"]
innovations: ["首个将语料库中毒形式化为RLVR问题的框架，联合优化排名扭曲与事实腐败奖励", "三原子信号（排名扭曲/事实腐败/查询重复惩罚）专用奖励设计，实现紧凑LLM对抗生成", "仅用次优dense retriever代理训练的生成器可强迁移攻击cross-encoder/sparse/commercial embedding模型"]
benchmarks: ["MS MARCO", "TREC DL 2019", "TREC DL 2020", "BEIR (NQ, FiQA, Touché-2020, TREC-COVID, SciFact, NFCorpus)", "FlashRAG RAG benchmark"]
---

# 论文速读：VerTox-Verifiable-Reward-Guided-Corpus-Poisoning-Against-Neu

## 一句话总结
论文提出了 VerTox，首个将语料库中毒攻击形式化为可验证奖励引导强化学习（RLVR）的框架，通过联合优化排名扭曲与事实腐败奖励，将紧凑 LLM 微调为主攻神经网络排序模型的内容生成器；在多种白盒/黑盒排序架构及商业 embedding 模型上实现了近完美攻击成功率（ASR），并显著降低了下游 RAG 应用的答案准确率。

## 研究问题与动机
- **核心问题**：神经网络排序模型在面对 LLM 可大规模生成流畅但具有欺骗性的内容时，其对抗鲁棒性未被充分理解，尤其是针对语料库中毒攻击（adversary 向语料库注入精心构造的恶意文档以扭曲排序行为）的脆弱性。
- **现有方法的不足**：
  - Word Substitution 方法（如 HotFlip、PRADA）需昂贵梯度累积，计算开销大，且迭代替换易破坏语法流畅性，容易被 perplexity-based 过滤检测。
  - Embedding Perturbation 方法（如 Vec2Text、EMPRA）在 embedding 空间扰动后解码回文本时易产生低流畅度/不自然输出，且 embedding 拓扑模型特定，跨架构迁移性差。
  - LLM 直接 Prompting 方法（如 AttChain、PoisonedRAG）虽能生成流畅文本，但缺乏来自排序目标的显式反馈，难以稳定超越当前最优文档，且往往只保留原始证据并附加伪造声明，未能实质性地"事实腐败"。

## 核心贡献（创新点）
1. **首个将语料库中毒形式化为 RLVR 问题的框架**：利用 GRPO 优化 + 可验证奖励微紧凑 LLM 作为内容生成器，与之前基于梯度/token 替换或 embedding 扰动的方法本质不同。
2. **三原子信号联合设计的专用奖励函数**：排名扭曲奖励（基于局部 proxy retriever 的归一化边际）、事实腐败奖励（双向 NLI  hallucination detector）、查询重复惩罚（基于 Levenshtein 的部分相似度），首次将排序暴露与语义欺骗性联合优化，区别于仅追求 ASR 或仅依赖 prompt 的方法。
3. **跨架构强迁移的黑盒攻击能力**：仅使用子优 dense retriever（SimLM-base）作为训练代理，即可攻击更强的 dense、learned sparse、cross-encoder reranker 及商业 closed-source embedding（Cohere）模型，揭示了不同神经网络排序架构共享可利用的相关性信号。
4. **证实了对下游 RAG 应用的实质性危害**：注入的对抗文档不仅扭曲排序，其包含的事实腐败还会显著降低下游 RAG 答案准确率（从约 0.70 降至约 0.30），超越了以往仅关注排序指标的攻击工作。

## 方法详解
- **整体框架**：给定目标查询 $q$ 及其当前最优良性文档 $d^+$，内容生成器策略 $\pi_\theta$ 生成对抗文档 $\tilde{d} \sim \pi_\theta(\cdot|q, d^+)$，使用 GRPO 进行 RLVR 优化。
- **GRPO 目标函数**：采样 $G$ 个候选 $\{\tilde{d}_i\}$，计算组内归一化优势 $A_i$，使用 clipped surrogate objective $\min(\rho_i A_i, \bar{\rho}_i A_i)$ 优化策略，省略 KL 惩罚项。
- **排名扭曲奖励（Rank-distortion Reward）**：
  - 使用本地 proxy retriever（SimLM-base）的归一化评分 $s_\phi(q,d) \in [-1,1]$ 作为可访问的替代信号。
  - 归一化排名边际：$\bar{\Delta}_{\text{rank}} = \frac{s_\phi(q,\tilde{d}) - s_\phi(q,d^+)}{1 - s_\phi(q,d^+)}$，使进展跨查询可比。
  - 连续奖励：$R_{\text{rank}} = 2 \cdot \sigma(\alpha \bar{\Delta}_{\text{rank}}) - 1$，$\alpha=5$ 控制陡峭度，提供渐进式稠密反馈。
- **事实腐败奖励（Factual Corruption Reward）**：
  - 使用 NLI-based hallucination detector（HHEMv2）计算双向事实一致性方向得分 $h_\psi(a,b) \in [0,1]$。
  - 双向取平均避免 trivial edit（仅追加或仅删除）：$R_{\text{fact}} = \frac{1}{2}[(1-h_\psi(d^+,\tilde{d})) + (1-h_\psi(\tilde{d},d^+))]$。
- **查询重复惩罚（Query Repetition Penalty）**：
  - 基于部分 Levenshtein 相似度 $s_q(d)$ 衡量文档与查询的lexical overlap。
  - 归一化超额重复：$\bar{\Delta}_{\text{rep}} = \frac{\max\{0, s_q(\tilde{d}) - s_q(d^+)\}}{1 - s_q(d^+) + \epsilon}$，仅惩罚超出良性基线部分的重复。
  - Sigmoid 映射：$R_{\text{rep}} = 2 \cdot \sigma(\beta \bar{\Delta}_{\text{rep}}) - 1$，$\beta=5$。
- **完整奖励**：$R = \lambda_{\text{rank}} R_{\text{rank}} + \lambda_{\text{fact}} R_{\text{fact}} - \lambda_{\text{rep}} R_{\text{rep}} - \lambda_{\text{len}} R_{\text{len}}$，其中 $\lambda_{\text{rank}}=\lambda_{\text{fact}}=\lambda_{\text{rep}}=1.0$，$\lambda_{\text{len}}=0.3$。
- **训练设置**：LoRA（rank=16, scaling=32, dropout=0.05, lr=$1\times10^{-5}$），per-device batch=4，gradient accumulation=4，训练5个epoch，使用三个紧凑 LLM：Qwen3-0.6B、Llama3.2-1B-Instruct、Gemma2-2b-it。

## 实验与结果
- **数据集**：训练集 MS MARCO（3000 query-passage pairs）；测试集 TREC DL 2019/2020（in-domain），BEIR 六个 benchmark（NQ, FiQA, Touché-2020, TREC-COVID, SciFact, NFCorpus）用于 out-of-domain 迁移评估。
- **评估基线**：RandomToken（p=0.3 随机替换）、HotFlip（梯度替换）、EmbedPerturb、Prompting（Qwen3-8B thinking mode）。
- **排序模型（白盒/黑盒）**：SimLM-base（白盒 proxy）、BGE-base、Cohere（商业闭源）、SPLADE、RepLLaMA、RankLLaMA。
- **主要结果（Top-1 Attack, White-box）**：
  - VerTox (Gemma2-2b-it) 在 in-domain 平均 ASR=1.00、Top@1=0.84；out-of-domain 平均 ASR=0.99、Top@1=0.80，全面超越所有基线。
  - Prompting (Qwen3-8B) 仅 ASR=0.99（in-domain）/0.93（out-of-domain），Top@1 分别为 0.54/0.49，弱于最小参数量的 VerTox。
- **黑盒迁移结果**：
  - 对 Cohere 商业模型的 Top@1 平均达 0.84–0.89（不同 generator），显著高于 Prompting 的 0.66。
  - 对 RankLLaMA（cross-encoder）Top@1 平均达 0.59–0.68，表明跨架构迁移有效。
- **Top-10 扩展实验**：随着 k 增大，nDCG@10 持续下降，Top-1 到 Top-2 降幅最显著，表明仅需少量中毒即可系统破坏检索质量。
- **RAG 下游实验**：原始上下文准确率约 0.70，中毒上下文准确率降至约 0.30（对所有 k∈{1,3,5,7,10} 和 Qwen3-8B、GPT-OSS-20B 均如此）。
- **可读性**：VerTox 输出 Dale-Chall 分数仅比良性文档高 0.3 分（其他方法高 0.8–7.0 分），极低可检测性。
- **最强结果**：VerTox (Gemma2-2b-it) 在 BGE-base 黑盒 Top@1 平均达 0.82，Cohere 上达 0.89；ASR 接近饱和（0.99–1.00）。

## 相关工作脉络
1. **Word Substitution 攻击（PRADA、HotFlip、MCARA、TARA）**：基于梯度或关键词替换操控排序，计算成本高且破坏文本流畅性，VerTox 以端到端 LLM 生成替代逐 token 优化，同时保证可读性。
2. **Embedding Perturbation（Vec2Text、EMPRA）**：在 embedding 空间扰动再解码回文本，存在拓扑不兼容和低流畅度问题；VerTox 直接在文本空间生成，通过奖励信号间接引导排序优化。
3. **LLM 生成对抗内容（AttChain、PoisonedRAG）**：AttChain 使用 chain-of-thought 迭代修改，扩展性差；PoisonedRAG 聚焦答案操纵且依赖查询重复；VerTox 首次将 LLM 语料库中毒建模为 RLVR 问题，联合优化排名与事实腐败。
4. **Corpus Poisoning（Zhong et al. 2023、Wu et al. 2023、Su et al. 2025a）**：早期工作集中于近似梯度下降或 token 级扰动；VerTox 转向可微/可验证奖励的 RL 训练范式，实现了更强的跨架构迁移性。
5. **RLVR/GRPO 在 LLM 微调中的应用（DeepSeek-R1、Tulu 3）**：本文首次将这些 RL 技术应用于 IR 安全领域，区别于它们在数学推理或通用对齐中的应用场景。

## 局限性与未来方向
- **威胁模型局限**：仅适用于开放检索环境（如 web search），防火墙后的私有语料库不在范围内；未评估重复过滤、垃圾检测、内容审核、对抗重训练等防御措施。
- **单一 proxy 依赖**：仅使用 SimLM-base 一个代理模型进行训练，多 proxy 训练可能提升更强迁移性；factuality 和 detectability 评估缺少 human study。
- **RAG 评估规模有限**：下游实验仅使用 100 FlashRAG 查询、两个回答模型和 LLM judge，跨更多领域和检索管道的更全面评估仍待开展。
- **随机事实腐败**：当前生成的是随机腐败内容，未来可扩展为针对特定主张/实体的可控攻击，或引入多轮 agentic  formulation 以适配 agent 的中间查询和行动。

## 研究启发与可借鉴点
1. **奖励函数设计范式**：三原子信号（排名扭曲+事实腐败+重复惩罚）的解耦设计思路可迁移至其他对抗生成任务，尤其是需要兼顾"表面相关性"与"实质欺骗性"的场景（如 adversarial dialogue、fake review generation）。
2. **稠密归一化奖励的技术细节**：基于剩余分数空间的归一化（normalized margin）策略可泛化到其他 RLVR 场景，解决跨样本奖励尺度不一致的问题；sigmoid 映射提供有界平滑反馈，值得复用。
3. **黑盒迁移实验设计**：用次优 proxy 训练、在更强黑盒模型上评估的协议，为评估对抗样本的跨架构鲁棒性提供了可比标准，可参考于其他模型的 security evaluation。
4. **紧凑 LLM + LoRA 的高效对抗训练**：仅用 0.6B–2B 参数模型通过 GRPO 即超越 8B 直接 prompt，揭示了小模型 RL 微调在对抗生成中的性价比优势，可结合本团队低资源 LLM 方向进一步探索。
5. **从排序攻击到 RAG 危害的链路验证**：不仅报告 ASR/Top@k，还验证了下游应用层面的实质伤害（RAG 准确率下降），这种端到端评估范式对安全论文的提升有价值。

## 关键术语表
**Corpus Poisoning**：攻击者向检索语料库注入恶意构造文档以扭曲排序结果、误导用户或下游系统的攻击类型。
**RLVR (Reinforcement Learning with Verifiable Rewards)**：使用自动可计算的确定性验证器返回标量奖励而非人工偏好标签的 RL 训练范式，避免了对人类标注的依赖。
**GRPO (Group Relative Policy Optimization)**：DeepSeekMath 提出的策略优化算法，通过组内归一化优势估计替代 advantage baseline，省略 KL 惩罚项以简化训练。
**ASR (Attack Success Rate)**：衡量对抗文档是否超过最低排名的相关文档的比例，是 corpus poisoning 攻击的核心成功率指标。
**NLI-based Hallucination Detector**：基于自然语言推理的事实一致性检测器，用于量化对抗文档相对于良性文档的事实偏离程度。
**Proxy Retriever**：攻击者在黑盒设置下可访问的本地排序代理模型，用于在训练阶段近似计算排名扭曲奖励。
**Dale-Chall Readability Score**：衡量文本可读性的经典指标，分数越低表示文本越易读，用于评估对抗文档的语言自然性。
**Top@k**：衡量对抗文档是否出现在前 k 个检索结果中的指标，比 ASR 更能反映实际排名暴露程度。

## 可复现要素
- **训练数据集**：MS MARCO（3000 query-passage pairs），公开可获取。
- **测试数据集**：TREC DL 2019/2020、BEIR（NQ, FiQA, Touché-2020, TREC-COVID, SciFact, NFCorpus），全部公开。
- **代码开源**：是，公开于 https://github.com/zhiqihuang/vertox-corpus-poisoning。
- **模型权重**：所有 base checkpoint 来自 Hugging Face，公开可下载；verifier 使用 vectara/hallucination_evaluation_model（HHEMv2），公开。
- **关键超参**：LoRA rank=16, scaling=32, dropout=0.05, lr=1e-5, per-device batch=4, gradient accumulation=4, epochs=5, α=β=5, λ_rank=λ_fact=λ_rep=1.0, λ_len=0.3。
