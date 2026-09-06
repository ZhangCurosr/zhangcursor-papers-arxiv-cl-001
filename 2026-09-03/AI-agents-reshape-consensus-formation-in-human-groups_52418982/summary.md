---
title: "AI-agents-reshape-consensus-formation-in-human-groups"
source: https://arxiv.org/pdf/2609.02122v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-06 20:58:32"
field: "人机协同与社会计算"
keywords: ["LLM agents", "consensus formation", "human-AI interaction", "referential communication", "cultural evolution", "mixed populations"]
innovations: ["发现agent比例与共识强度的非单调U型关系，揭示三阶段相变动力学", "构建词汇-概念双层贡献度量框架，提出CLR阈值识别agent影响力渗透深度", "阐明共享语言先验驱动的语义引力源机制与身份抵抗-从众压力竞争框架"]
benchmarks: ["pure-human baseline (0% agent)", "12.5% / 33.3% / 50% / 75% agent proportion conditions", "Qwen2.5-VL-32B vs Qwen2.5-VL-7B capability comparison"]
---

# 论文速读：AI-agents-reshape-consensus-formation-in-human-groups

## 一句话总结
本文通过参照性沟通游戏实验，系统研究了不同比例的LLM agent参与对混合人类-AI群体共识形成过程的影响，发现agent比例与共识强度呈非线性关系，并在低/中/高三种区间呈现截然不同的共识动力学机制与语义内容差异。

## 研究问题与动机
- **核心问题**：当LLM agent以不同比例嵌入人类群体时，是否仅放大人类驱动的既有动态，还是会重塑群体共识的内容与归属？
- **现有方法不足**：
  - 既有研究多关注纯人类群体或纯agent群体的共识，缺乏对混合群体的系统性探索；
  - 真实场景中的共识研究常受固定社会角色、既有关系、任务背景等因素混淆，难以分离agent比例的独立效应；
  - 已有AI影响力研究聚焦于外部任务绩效，而非群体内部涌现规范的形成机制。

## 核心贡献（创新点）
- **发现非单调三阶段共识动力学**：首次系统揭示agent比例与共识强度之间的U型关系，分别命名为H1（低比例促进）、H2（中比例破坏）、H3（高比例恢复但转向agent主导），突破了"agent越多越好"的线性假设。
- **揭示agent影响力从词汇到概念的级联渗透**：提出词汇贡献率（$C_{lex}$）与概念贡献率（$C_{con}$）的双层度量，发现agent先在表层词汇层面占据主导，后在高比例下实现概念结构的全面接管，两者之间存在CLR > 1的质变阈值。
- **阐明agent influence的双重机制**：从agent侧提出"共享语言先验+跨轮稳定表达"形成的语义引力源模型，从人类侧提出"身份抵抗 vs. 从众压力"的竞争框架，二者耦合解释了共识方向逆转的微观机制。
- **构建人机共识内容的五维语义分析框架**：引入具体性（concreteness）、命题密度（propositional idea density）、类比比例（analogical ratio）、整体性（holistic ratio）、事件框架比例（event framing ratio）五个维度，量化揭示agent-led共识趋向抽象、稀疏、几何化、静态化。
- **通过prompt工程实现因果干预验证**：在固定33.3%比例下操控agent的adopt/persistence倾向，证明agent影响力并非"越坚持越好"或"越顺从越好"，而依赖于早期适应与晚期巩固的动态平衡。

## 方法详解
- **实验范式**：基于referential communication paradigm的collaborative description game，24名参与者（人类+LLM agent混合）组成一组，进行40轮随机配对的 pairwise 描述游戏，每轮双方独立描述同一张tangram图形，获得相似度反馈后进入下一轮，新配对。
- **共识强度度量**：使用all-MiniLM-L6-v2对所有表达式进行embedding，计算群体内两两cosine相似度，经个体中心化处理后得到群体共识值$C^{(t)}$，最终轮值即为共识强度。
- **方向性移动**：计算每个参与者向对方群体初始linguistic centroid的cosine距离变化量$\Delta d_i$，判断是human→agent适应还是agent→human适应占主导。
- **双层贡献度量**：
  - 词汇层（$C_{lex}$）：提取稳定词表（最后5轮全现），统计agent/human在各词出现早期的使用频次，经人口比例归一化后加权平均；
  - 概念层（$C_{con}$）：将表达式解析为scene graph（spaCy依赖解析），用soft SPICE度量（objects/attributes/relations加权F1）计算pairwise相似度，HDBSCAN聚类得到语义cluster，同样统计早期扩散贡献。
  - CLR = $C_{con}/C_{lex}$，>1表示agent在概念层影响力超过词汇占比。
- **行为动力学分解**：
  - Adoption：相邻两轮与partner表达相似度的增量；
  - Persistence：相邻两轮自身表达的cosine相似度；
  - 分early（1-20轮）/late（21-40轮）两阶段分析。
- **干预实验**：在33.3%条件下通过system prompt添加behavioral tendency clause，分别设置high-adoption（强调模仿partner）、high-persistence（强调保持自述一致性）、neutral三组。
- **内容分析**：用Qwen2.5-72B-Instruct（temperature=0）作为judge，对每唯一表达打分analogical/holistic/event framing三个连续维度（0-1）。
- **主观感知测量**：post-experiment questionnaire中，让participant判断partner身份并评价adopt willingness，构建OLS回归分解identity resistance与conformity pressure的贡献。

## 实验与结果
- **数据集**：KiloGram Tangrams数据集（20个候选figure），单stimulus设计（所有session使用同一张figure）。
- **被试与条件**：127名中国大学生，英语能力筛选通过后随机分配至8个条件（Study A：0%/12.5%/33.3%/50%/75%五档；Study B：33.3%下high-adoption/high-persistence；Study C：33.3%下强/弱模型对比，Qwen2.5-VL-32B vs Qwen2.5-VL-7B）。
- **主要结果**：
  - 共识强度：pure-human基线0.695 → 12.5%提升8.0%（z=3.282, p=0.001）→ 33.3%下降23.1%（z=-12.33, p<0.001）→ 50%下降14.5%（z=-6.16, p<0.001）→ 75%回升至0.725；
  - 方向性移动反转：12.5%时agent移动量0.822 > human 0.575；≥33.3%时human向agent移动显著大于反向（33.3% U=104, p=0.007；50% U=129, p<0.001；75% U=103, p<0.001）；
  - 词汇贡献：33.3%的agent已贡献60.7%最终词汇；概念贡献在75%时达100%；
  - CLR阈值：12.5%/33.3% <1，50%时1.24，75%时1.27；
  - 内容差异（H1→H3）：concreteness 2.986→2.724（p<0.001），idea density 5.419→5.321（p=0.020），analogical ratio 0.802→0.050（p<0.001），holistic ratio 0.685→0.353（p<0.001），event framing 0.385→0.000（p<0.001）；
  - 干预：high-persistence使共识下降9.6%（z=-3.289, p=0.001），CLR飙升117.8%（0.599→1.304）；high-adoption共识无显著变化，CLR降42.8%；
  - 弱模型（7B）：共识回升至0.725（z=14.14, p<0.001），CLR降22.9%（0.599→0.462）。
- **最强结果**：75% agent条件下共识强度0.725，但CLR=1.27，conceptual贡献达100%，且human满意度显著降低（p=0.004）。

## 相关工作脉络
- **Centola & Baronchelli (2015)**：纯人类群体中自发惯例涌现的经典实验范式，本文沿用referential communication paradigm但扩展至mixed human-AI群体；
- **Ashety et al. (2025, Sci Adv)**：纯LLM agent群体的emergent social convention研究，本文填补了混合群体这一空白，并额外分析内容质量差异；
- **Tessler et al. (2024, Science)**：AI作为designated mediator促进民主审议共识，本文指出influence可自然涌现于普通参与而非仅靠角色设计；
- **Bai et al. (2025, Nat Commun)**：LLM生成信息对人类政策态度的persuasion效应，本文区分了结构性conformity与显性说服的机制差异；
- **Schroeder et al. (2026, Science)**：恶意AI swarm威胁民主的警告，本文提供了其成为可能的定量条件与机制解释；
- **Clark & Wilkes-Gibbs (1986)**：协作参照沟通理论框架，为本文内容分析维度提供理论基础。

## 局限性与未来方向
- **生态效度受限**：使用content-neutral抽象视觉刺激，排除了现实情境中的content sensitivity、权力不对称、情感投入等因素，推广需谨慎；
- **网络拓扑单一**：采用fully mixed随机配对，未考虑真实社会的community结构、opinion leaders、information bottlenecks，未来可探索network design对agent influence的调控作用；
- **模型族单一**：仅测试Qwen系列两种能力级别，虽代表性覆盖主流训练范式（pretrain+SFT+alignment），但定量相位边界可能因模型架构/训练数据/对齐策略而异；
- **样本量有限**：尤其H3条件仅6名人类被试，peer-centroid proximity变化趋势未达统计显著（p=0.170）；
- **未来方向**：跨model family系统性对比、更丰富沟通语境验证、network结构干预实验、long-term norm persistence追踪。

## 研究启发与可借鉴点
- **非单调效应建模思路**：将agent比例作为连续干预变量而非二元分类，揭示系统行为的phase transition特征，该方法可迁移至其他混合智能体系统设计；
- **双层贡献度量框架**：词汇层与概念层的解耦分析，为评估AI对群体语言规范的渗透深度提供可复用的量化指标体系；
- **adopt-persistence动态平衡设计**：通过prompt engineering在agent侧实施行为干预，证明"早期适应+晚期巩固"的时序策略最优，这一原则可应用于agent行为调控设计；
- **身份感知与结构压力的竞争模型**：OLS回归分解identity resistance与peer-centroid proximity的相对效应，为理解human-AI协作中的主观接受度提供了机制框架；
- **弱模型对比实验**：展示capacity退化如何削弱semantic attractor效应，提示agent proportion需与capability协同考量，而非单纯增加数量即可达成目标。

## 关键术语表
- **Collaborative description game**：参照性沟通范式下的协作描述游戏，参与者通过多轮随机配对描述同一视觉刺激，观察共享惯例的自发涌现；
- **Consensus strength**：群体共识强度，以最后轮次所有参与者表达式两两embedding相似度均值衡量；
- **Conceptual-Lexical Ratio (CLR)**：概念-词汇比率，概念层贡献与词汇层贡献之比，>1表示agent对共识语义结构的影响力超过其词汇占比；
- **Semantic attractor**：语义引力源，由agent间共享pretrained prior形成的初始表达聚集，叠加跨轮稳定性后对群体收敛产生定向牵引效应；
- **Adoption-Persistence dynamics**：采用-坚持动力学，文化传递理论下群体规范传播的两大竞争力量——模仿partner vs. 坚持自身惯例；
- **Identity-based resistance**：身份基础抵抗，人类对被判断为AI生成表达的主观排斥倾向，构成adopt decision中的负向预测因子；
- **Peer-centroid proximity**：对等者质心接近度，个体partner表达与群体实时centroid的相似度，表征结构性从众压力；
- **Scene graph parsing**：场景图解析，利用NLP工具将文本描述结构化拆解为objects/attributes/relations三元组，用于细粒度语义比较；

## 可复现要素
- **数据集**：KiloGram Tangrams（公开数据集）；实验stimulus由Qwen2.5-VL-32B-Instruct评估ambiguity index后筛选，single-stimulus设计；代码仓库见GitHub；
- **代码/权重开源**：Python代码已开源于 https://github.com/tsinghua-fib-lab/human-ai-consensus，主模型Qwen2.5-VL-32B-Instruct可通过HuggingFace获取，弱模型Qwen2.5-VL-7B-Instruct同样开源；
- **关键超参**：rounds=40，group size=24，embedding模型all-MiniLM-L6-v2，scene graph解析spaCy，软SPICE权重objects:0.4/attributes:0.3/relations:0.3，HDBSCAN聚类通过Optuna优化（30 trials TPE sampler），LLM-judge用Qwen2.5-72B-Instruct temperature=0，early-diffusion window默认3 rounds（1-3 robust），PageRank damping=0.85，OLS使用HC3稳健标准误。
