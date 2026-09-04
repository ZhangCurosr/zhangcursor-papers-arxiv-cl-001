---
title: "Query-Side-Attacks-on-GNN-Based-KGQA-Tracing-Failures-from-E"
source: https://arxiv.org/pdf/2608.25922v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-09-04 01:48:40"
field: "知识图谱问答鲁棒性与可解释性"
keywords: ["KGQA", "GNN-RAG", "adversarial robustness", "query-side attack", "stage-isolation", "compositional restructuring", "subgraph retrieval"]
innovations: ["提出阶段隔离协议并量化子图拓扑归因（52.08pp/52.22pp）", "定义答案保留型CR/RS扰动，揭示答案存在≠可达的PPR锚定失败", "推理时路径注入（RoG+PathOnly）免微调恢复CR损失至51.43% EM"]
benchmarks: ["ComplexWebQuestions (CWQ)", "WebQSP", "MetaQA (WikiMovies)"]
---

# 论文速读：Query-Side Attacks on GNN-Based KGQA: Tracing Failures from Entity Linking to Answer Generation

## 一句话总结
论文首次对基于GNN的知识图谱问答（KGQA）管道进行**阶段隔离式**对抗评估，提出组合重构（CR）与关系同义词交换（RS）两种保留答案的扰动，证明管道崩溃主要源于子图构建拓扑失配而非GNN推理器本身；在CWQ上CR攻击导致99%以上端到端指标下降均来自子图阶段，而恢复正确子图后GNN指令解码器仍保持近基线准确率（52.76% EM）。

## 研究问题与动机
- **管道级崩溃难以定位**：既有鲁棒性评测仅给出端到端EM/Hit@1，无法区分实体链接、子图检索、GNN推理、答案生成四阶段中哪一阶段最先失效、失效机制为何。
- **Query-side缺失的系统对照**：现有RAG/KGQA鲁棒性研究集中于corpus-side（毒化检索索引、注入对抗段落），要求攻击者写KG的强能力；本文聚焦仅需提交问题的黑盒场景。
- **高覆盖率≠可达性**：实测发现ELQ在CR下子图仍包含74%金答案实体，但EM仅0.68%，暴露“答案存在≠答案可达”的关键盲点，端到端指标无法识别该拓扑错配。
- **PPR固定拓扑的脆弱性假设**：基于PPR的子图构建以原始问题锚定推理链，面对结构改写是否仍稳健尚无定量证据；需要对照不同检索变体（NSM、EPR-KGQA、GMT-KBQA）定位架构边界。

## 核心贡献（创新点）
1. **阶段隔离协议与四指标量化**：通过冻结上游输出、仅替换当前阶段输入的配置分离$\Delta_{EL}$、$\Delta_{SR}$、$\alpha_{pert}$与$\Delta_{AG}$，首次给出逐阶段退化路径的可重复测量；与既有工作相比，区别在于同时支持“子图固定快模式”与“全链路慢模式”，从而将子图归因与解码器归因从同一EM差值中拆出。
2. **答案保留型CR/RS扰动族**：CR实施跳序反转/约束注入/中间实体别名替换，RS替换谓词表面形式，均经SPARQL指称验证保持gold答案不变；既不同于改答的ES诊断扰动，也不同于破坏语义的通用文本攻击，直接对标管道不同脆弱环节。
3. **揭示“答案存在≠可达”**：ELQ+CR在$\alpha_{pert}=74\%$(CWQ)时EM=0.68%，而GraftNet-Oracle以$\alpha_{pert}=63.3\%$达29.82% EM，证明检索拓扑与推理链的对齐比单纯覆盖更重要；这一区分未被既有KGQA文献显式提出。
4. **归因结论指导加固靶点**：阶段隔离量化得$\delta_{SG}=52.08$pp、$\delta_{DEC}=0.14$pp（CWQ CR总下降52.22pp），明确 mitigation 应落在子图构建而非训练更强GNN；与多数把鲁棒性改进放在decoder的做法不同。
5. **推理时路径注入无需微调即可复原**：RoG+PathOnly把预测关系路径拼入提示即可把CR CWQ Hit@1提升至51.43%，且与路径增强微调版本RoG+RA相差<1pp，证明结构先验注入比重训更高效。

## 方法详解
- **管道形式化**：$q \xrightarrow{f_{EL}} E_q \xrightarrow{f_{SR}} \mathcal{G}_q \xrightarrow{f_{GR}} c_q \xrightarrow{f_{AG}} \hat{a}$。GNN-RAG实例中$f_{EL}$=ELQ（BERT-Large双编码器）、$f_{SR}$=PPR以$E_q$为种子、$f_{GR}$=3层GAT（ReaRev）、$f_{AG}$=Llama-2-7B指令生成。
- **阶段退化指标**：
  - 实体链接退化：$\Delta_{EL} = \mathrm{SeedHit}(E_q) - \mathrm{SeedHit}(E_{q'})$，使用any-match SeedHit。
  - 子图漂移：$\Delta_{SR} = 1 - J(\mathcal{G}_q, \mathcal{G}_{q'})$（triplet-level Jaccard）。
  - 答案存在率：$\alpha_{pert} = \frac{1}{N}\sum_i \mathbf{1}[a_i^* \in \mathcal{G}_{q'_i}]$。
  - 端到端下降：$\Delta_{AG} = \mathrm{EM}_{clean} - \mathrm{EM}_{pert}$。
- **三配置隔离协议**：① 干净基线$(q\to a)$；② 完整扰动管线$(q'\to \hat{a})$；③ 干净子图快模式（冻结$\mathcal{G}_q$，仅把question文本换为$q'$送入GNN）。由此得$\delta_{SG}=\mathrm{EM}_{fast}-\mathrm{EM}_{pert}$、$\delta_{DEC}=\mathrm{EM}_{clean}-\mathrm{EM}_{fast}$。
- **扰动生成与验证**：由Llama-3.3-70B-Instruct产生候选$q'$，过三重门控：PPL(q')<50、BERTScore(q,q')>0.85、SPARQL指称守恒（经本地Virtuoso验证）。ES仅做诊断、CR/RS为主攻击，另含S1–S4辅助类型。
- **子图变体对照**：ELQ（flat PPR）、GraftNet-Orig（关系感知cosine PPR）、NSM（BFS+relation-weighted剪枝）、GraftNet（question-embedding加权，GEM oracle seed）、EPR-KGQA（原子邻接模式单次检索，PPR-free）。

## 实验与结果
- **数据集**：CWQ（3,531题，2-4跳）、WebQSP（1,639题，1-2跳），均以Freebase MID为gold；另在MetaQA（WikiMovies）做跨库复现（2-hop 14,872 / 3-hop 14,274）。
- **基线与设置**：主模型GNN-RAG/ReaRev；多子图变体共用同一GNN权重，仅$\mathcal{G}$不同；清洗基线CWQ EM=52.9%、WebQSP EM=74.3%。指标为MID-based Hit@1/EM，bootstrap 95% CI（n=1000）；主要结论经Bonferroni 99.8% CI仍成立。
- **CR端到端结果（关键）**：
  - ELQ：CWQ 0.68%（↓52.2pp）、WebQSP 0.49%（↓73.8pp）。
  - NSM：CWQ 5.81%、WebQSP 1.65%。
  - ELQ+cosine（部署现实）：CWQ 14.98%、WebQSP 22.33%。
  - GraftNet-Oracle（GEM seed）：CWQ 29.82%（↑至74%中最高$\alpha_{pert}=63.3\%$）、WebQSP 36.55%。
  - EPR-KGQA（PPR-free对照）：CWQ 59.22%（↓仅1.2pp）、WebQSP 64.25%（↓3.9pp）。
- **RS端到端结果**：ELQ最鲁棒，CWQ 20.31%、WebQSP 50.95%；因其保持子图结构与ELQ种子一致，EM主要反映GNN指令解码器敏感度。
- **阶段归因（CWQ CR）**：$\delta_{SG}=52.08$pp、$\delta_{DEC}=0.14$pp；用干净子图+CR问题时ReaRev达52.76% EM，证明解码器近乎无损。
- **$\alpha_{pert}$对EM的预测力**：24-cell组件消融网格上Spearman $\rho=0.91$（p<1e-9）；但ELQ CR两格（最高$\alpha_{pert}$却最低EM）为拓扑锚定异常点，剔除后$\rho=0.64$（p=0.015）。
- **MetaQA跨库复现**：CR在3-hop仅↓6.91pp、2-hop仅↓0.11pp；全链路重执行CR变动≤0.6pp，证实故障随**检索算法**（PPR锚定）而非KG模式转移。
- **LLM推理对照**：RoG+PathOnly（无微调）在CR下CWQ Hit@1=51.43%，与RoG+RA（微调路径增强）≈50.5%基本持平；Llama-3.1-8B无微调则低于GNN基线。

## 相关工作脉络
- **GNN-RAG / ReaRev（Mavromatis & Karypis, 2024）**：本文主评测对象；以PPR+GAT+Llama-2生成构成级联失败链，本文用阶段隔离将其拆开量化。
- **GraftNet-Orig（Sun et al., 2018）**：关系感知PPR；本文扩展为question-embedding加权并引入GEM oracle seed，用于对照ELQ的拓扑锚定缺陷。
- **NSM（He et al., 2021）**：BFS+关系权重剪枝的subgraph检索；作为另一PPR类变体，CR下表现弱于GraftNet-Oracle，凸显种子质量主导。
- **EPR-KGQA（Ding et al., 2024）**：原子邻接模式单次检索，PPR-free；CR/RS下均维持近基线，作为“绕过固定PPR即鲁棒”的结构对照。
- **GMT-KBQA（Das et al., 2021）**：直接生成S-expression绕开subgraph；仅≤5pp下降，进一步验证PPR子图阶段是瓶颈。
- **ExplaiGNN（Christmann et al., 2023）**：迭代多轮Wikidata对话QA；CR/RS致P@1下降>70%，表明**迭代式**链式检索同样脆弱，差异在于“turn级上下文污染”而非单一PPR锚定。
- **实体链接系列（ELQ/BLINK/ReFinED/LLM-based linkers）**：本文CR/RS下SeedHit稳定（≤0.3pp下降），说明EL不是这两类攻击瓶颈；ES诊断显示EL本身脆弱（ELQ CWQ从57.4%跌至23.1%）。
- **RAG鲁棒性攻击（PoisonedRAG/Zhong et al., 2023; BADRAG/Xue et al., 2024; Phantom/Chaudhari et al., 2024）**：依赖KG写权限的corpus-side攻击；本文定位为仅需提交问题的query-side黑盒攻击，威胁模型更贴近实际。

## 局限性与未来方向
- **单系统×双基准**：核心发现主要来自GNN-RAG/ReaRev在Freebase-backed CWQ与WebQSP；跨模型与更宽KG（如完整Wikidata复刻）仍需验证。
- **防御性结论受限于GEM oracle**：GraftNet-Oracle的29.82% EM依赖未部署可用的黄金种子；部署现实配置（ELQ+cosine）仅到14.98%，种子质量仍是瓶颈。
- **幸存者偏差**：未通过PPL/BERTScore/SPARQL门控的扰动回退到原问题，报告的攻击强度是保守下界。
- **多比较CI校正说明**：主要95% CI为单次比较未校正，尽管Bonferroni 99.8%区间仍支持五大结论，但小效应（如GEM+Cosine vs GEM+Flat CWQ CR仅0.46pp差距）不显著。
- **可迁移性范围**：扰动模板由Llama-3.3-70B-Instruct在Freebase约束下生成并通过英语表面形式过滤，多元语言/领域KG的表现未测。
- **未来方向（论文提及）**：① 问题条件化检索或关系路径beam搜索；② 拓扑感知检索与路径注入结合；③ 强化ES场景下的实体链接；④ CR×RS级联的阶段引导攻击；⑤ 完整Wikidata复现。

## 研究启发与可借鉴点
- **阶段隔离协议可直接迁移**到其他“检索→推理→生成”级联系统（例如RAG QA、Program-of-Thought、Tool-use agent）的鲁棒性诊断：冻结上游、替换当前输入，量化每一段的$\delta$归因。
- **$\alpha_{pert}$（答案存在率）应纳入子图检索评测**：作为SPCC≈0.91的强预测因子，能提前暴露“高覆盖但低可达”的检索病态，优于只看Jaccard或跳数统计。
- **路径注入作为免微调加固原语**：在推理时拼接预测关系路径即可复原约一半CR损失，适合作为部署期即插即用补强，避免重新训练大参数GNN/LLM。
- **架构选择优先级：先换检索再换推理器**：EPR-KGQA的单次模式、GMT-KBQA的S-expression直接生成，对CR/RS天然免疫，提示团队若追求鲁棒性应优先重构检索阶段而非堆叠更深的GNN。
- **跨库对照实验增强结论可信度**：在MetaQA上重复协议，证实故障由PPR锚定引发而非Freebase schema特有，这种KB-switch设计可推广到其它结构化问答系统的归因验证。

## 关键术语表
- **Stage-isolation protocol**：冻结上游阶段输出、仅替换当前阶段输入的评测配置，用以拆分端到端误差并归因到EL/SR/GR/AG四段之一。
- **Compositional Restructuring (CR)**：保持实体提及与SPARQL指称不变，对问题做跳序反转/约束注入/中间实体别名替换，以扰动subgraph拓扑与多跳推理链。
- **Relation Synonym Swap (RS)**：将谓词短语替换为同义表达、保留实体提及与答案，使得子图结构基本不变，主要用于孤立测试GNN指令解码器对表面词形变化的敏感度。
- **Entity Swap (ES)**：把主题实体替换为同域另一实体，同时改变gold MID与答案；属诊断扰动，用于刻画实体链接的脆性上限。
- **$\alpha_{pert}$（答案存在率）**：扰动后子图中仍包含gold MID的比例；文中被证明是EM的强预测因子，但在ELQ CR格中出现反例（74%存在却0.68% EM）。
- **GraftNet / GraftNet-Orig / GraftNet-V2**：本文扩展的PPR检索变体；-Orig用GloVe关系连续权重，本文GraftNet用question-embedding加权，V2用二值mask替代连续权重。
- **GEM (Golden Entity Map)**：基于黄金SPARQL MIDs的oracle种子，非部署可用；用于给出子图检索的理论上限以分离“种子质量”与“PPR策略”的贡献。
- **RoG+PathOnly / RoG+RA**：RoG推理模型在推理时注入预测关系路径（PathOnly，不微调）与在训练时加入路径增强样本（RA，微调）；前者已能取得与后者相近的恢复效果。

## 可复现要素
- **数据集**：CWQ（Apache-2.0）、WebQSP（MSR License）、Freebase 2015静态RDF dump（CC-BY 2.5）；MetaQA（WikiMovies KB）。
- **代码与扰动集**：扰动数据集与评测基础设施已匿名开源：https://anonymous.4open.science/r/atkgrag-E85C（CC BY 4.0）。
- **模型权重**：GNN-RAG/ReaRev使用作者发布checkpoint；ELQ/BLINK/EPR-KGQA/GMT-KBQA按论文许可使用；RoG checkpoint公开。
- **关键超参**：扰动生成模型Llama-3.3-70B-Instruct；PPL阈值<50、BERTScore>0.85、SPARQL指称守恒验证；GNN为3层GAT；LLM解码器为Llama-2-7B（微调）与Llama-3.1-8B（零样本对比）。
- **硬件**：单节点4×A100 80GB + 96核CPU；总测算约800 A100 GPU小时。
