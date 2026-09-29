# DR.CREDIT: RUBRIC-GROUNDED PROCESS CREDIT ASSIGNMENT FOR DEEP RESEARCH AGENTS

Yingjian Zhu<sup>1,2,3∗</sup> Zhenyi Wang<sup>3†</sup> Jiaxin Guo<sup>3,4</sup> Kun Ding<sup>2†</sup> Ying Wang<sup>2</sup> Shen Huang<sup>3</sup> Xunjie Zhu<sup>3</sup> Pengjun Xie<sup>3</sup> Shiming Xiang<sup>2</sup>

<sup>1</sup>School of Artificial Intelligence,

University of Chinese Academy of Sciences

<sup>2</sup>State Key Laboratory of Multimodal Artificial Intelligence Systems (MAIS), Institute of Automation, Chinese Academy of Sciences

<sup>3</sup>Alibaba Token Hub, Alibaba Group <sup>4</sup>Peking University

## ABSTRACT

Rubric-based tasks are increasingly addressed through reinforcement learning (RL), with rubric scores used as training rewards. However, these rewards typically supervise final answers without distinguishing the contributions of intermediate decisions. Many existing credit assignment methods rely on groundtruth answers to define process rewards, limiting their applicability to open-ended tasks without canonical solutions. To address this limitation, the proposed rubricgrounded credit uses task requirements as a shared reference for final answer evaluation and process supervision. The information returned by tools is assessed for the additional support it provides toward satisfying each rubric relative to that rubric’s history of accepted support. By referencing these histories, credit distinguishes new support from evidence already present in the trajectory while recognizing partial support for each rubric. Dr.Credit uses rubric-grounded credit to supervise intermediate tool turns in an RL framework for deep research agents. The resulting process advantages are combined with GRPO outcome advantages to guide research decisions while retaining supervision of final-report quality. Evaluations on four in-domain and out-of-domain benchmarks show that Dr.Credit outperforms the evaluated open deep research baselines on every primary metric and submetric. Meanwhile, with an 8B-parameter backbone, the trained agent achieves average performance competitive with the evaluated frontier proprietary models. Further analyses suggest more efficient evidence acquisition and higherquality reports under limited research-turn budgets, motivating the extension of rubric-grounded process supervision to a broader range of rubric-based tasks.

## 1 INTRODUCTION

Real-world applications of language models increasingly involve open-ended generation tasks, in cluding novel and screenplay writing (Wu et al., 2025), medical consultation (Arora et al., 2025), and deep research. Response quality in these settings is difficult to assess through exact answer matching, as evaluation requires nuanced judgments across multiple dimensions. Task-specific rubrics make these evaluation requirements explicit by specifying the expected content and qualities of a satisfactory response (Sharma et al., 2026). Rubrics as Rewards (RaR) (Gunjal et al., 2026) extends this evaluation approach to reinforcement learning (RL) by turning task-specific requirements into training rewards. In deep research, DR Tulu (Shao et al., 2026) and DeepRubric (Zhu et al., 2026a) train agents with rubric-based rewards, directly optimizing report quality against these requirements.

Recent work on rubric-based RL has focused on improving rubric quality to provide more reliable supervision for final answers (Shao et al., 2026; Zhu et al., 2026a; Xie et al., 2026a), but leaves credit assignment across intermediate turns unresolved. When these turns share an outcome advantage, a high-scoring answer can reinforce ineffective decisions, while later synthesis errors can cause useful intermediate steps to be penalized (Figure 1(a)). Many existing credit assignment methods use provided ground-truth answers as a reference for assigning intermediate rewards (Wang et al., 2026; Zhu et al., 2026b; Lu et al., 2026) (Figure 1(b)). This dependence limits direct transfer to open-ended tasks where rubrics specify what a satisfactory response should accomplish without prescribing a canonical answer to serve as a reference for turn-level credit assignment.

To address this challenge, the key is to identify a reference for evaluating intermediate turns without relying on a canonical answer. The rubric set used to evaluate the final answer offers a natural reference, since it defines task-specific requirements that remain applicable across different high-quality responses. Our core idea is to derive process credit from the additional support that newly obtained information provides toward satisfying each rubric. We maintain a separate support history for each rubric to assess these contributions and aggregate them into turn-level credit (Figure 1(c)). The credits are normalized into process advantages and combined with GRPO outcome advantages, ex tending rubric-based supervision to intermediate decisions while retaining final-answer evaluation.

![](images/97fd06d95583abd35b8912759658d4bb9d4bfd92b25260e5ec8177a2ee71bbcd.jpg)  
Figure 1: Credit assignment for rubric-based agent training. (a) Intermediate turns share an outcome advantage. (b) Ground-truth answers guide turn-level credit. (c) Task rubrics ground turn-level credit in additional evidence support while outcome supervision is retained. In (c), each rubric’s track darkens as evidence accumulates across turns. Turn colors denote normalized advantages.

This work applies these rubric-grounded credit assignment and optimization mechanisms to deep research, a representative open-ended task with rubric-based evaluation. The training framework is named Dr.Credit, which uses task-specific rubrics to assign process credit to intermediate tooluse turns. Turns that extract webpage content are credited for the additional support they provide relative to the existing evidence history for each rubric. These credits are then attributed to earlier search turns that returned the corresponding URLs. Search turns can also receive supplementary credit through an independent assessment of rubric support in snippets from results whose webpage content is not subsequently extracted. The resulting credits enter the preceding policy optimization scheme, providing turn-level supervision alongside the reward for the final report.

To evaluate the effectiveness of Dr.Credit, we conduct extensive experiments on both in-domain and out-of-domain benchmarks with deep research agents. Results show that Dr.Credit consistently outperforms the evaluated open deep research baselines, suggesting good generalization across benchmarks. Our main contributions are as follows: (1) We propose rubric-grounded credit, a general approach to process reward design for agent RL on open-ended tasks with rubric-based evaluation. It extends task rubrics from final-answer evaluation to turn-level credit assignment by assessing additional support relative to per-rubric histories, without requiring a canonical answer. (2) We apply this approach to deep research and introduce Dr.Credit, a training framework that assigns credit to intermediate research tool turns and integrates the resulting process advantages with GRPO outcome advantages. (3) With an 8B backbone, Dr.Credit achieves average performance competitive with the evaluated frontier proprietary models. Further analyses indicate that Dr.Credit acquires evidence more efficiently and produces higher-quality reports under limited research-turn budgets.

## 2 PRELIMINARIES

## 2.1 PROBLEM FORMULATION

Given an open-ended question q, a policy $\pi _ { \theta }$ generates a rollout $O = ( \tau _ { 0 } , \dots , \tau _ { T } )$ through successive rounds of reasoning and interaction with an external environment. Each tool-interaction turn $\tau _ { t }$ $( t < T )$ follows the ReAct paradigm (Yao et al., 2023), comprising [think], [tool call], and [tool response]. Conditioned on the question and the preceding interaction history, the policy generates the reasoning and tool call, while the environment supplies the observation returned by its execution. The sampled rollout $O _ { i }$ at the bottom of Figure 2 illustrates how each returned observation is appended to the context, where information gathered so far remains available to subsequent reasoning and tool use. At the final turn $\tau _ { T }$ , [think] is followed by [answer], yielding a final answer aˆ whose quality is then evaluated against the requirements specified by the task’s rubric set.

## 2.2 RUBRIC-BASED REINFORCEMENT LEARNING

Rubrics as Rewards (RaR) (Gunjal et al., 2026) turns rubric-based evaluations of final answers into reward signals for training the policy $\pi _ { \theta }$ through reinforcement learning. For a question $q ,$ the taskspecific rubric set $\mathcal { R } _ { q }$ contains K rubrics, with a nonnegative weight $w _ { k }$ assigned to each rubric and $\textstyle \sum _ { k = 1 } ^ { K } w _ { k } > 0$ . An LLM judge evaluates the final answer aˆ against each rubric, assigning a satisfaction score $s _ { k } ( q , \hat { a } ) \in [ 0 , 1 ] $ ; the weighted average of these scores provides the rubric reward:

$$
r _ { \mathrm { r u b r i c } } ( \boldsymbol { q } , \hat { \boldsymbol { a } } ) = \frac { \sum _ { k = 1 } ^ { K } w _ { k } s _ { k } ( \boldsymbol { q } , \hat { \boldsymbol { a } } ) } { \sum _ { k = 1 } ^ { K } w _ { k } } .\tag{1}
$$

Group Relative Policy Optimization (GRPO) (Shao et al., 2024) converts rollout-level outcome rewards into advantages by normalizing rewards across rollouts sampled for the same question. Since each advantage is shared by all policy-generated tokens within a rollout, outcome supervision does not explicitly distinguish the contributions of intermediate turns toward satisfying the task’s rubrics.

## 3 METHODOLOGY

![](images/a527c742eca357a90a4e281d8e616068e6231beede26d3706ee91cefada4ff3a.jpg)  
Figure 2: Overall framework of Dr.Credit, which incorporates rubric-grounded process credit into reinforcement learning for deep research agents. Left: the policy generates a group of rollouts for each query through ReAct interactions with the web environment. Center: an LLM judge assesses the additional support each tool turn provides relative to per-rubric evidence histories within its rollout, then updates the histories with accepted support points. Right: process credits are normalized across research turns in the rollout group and added to GRPO outcome advantages. Bottom: an example rollout $O _ { i }$ . Black dashed boxes mark tool responses masked out of the loss.

Rubrics can support fine-grained process rewards in reinforcement learning, extending their use beyond evaluating final outputs. However, tool turns cannot be judged by rubric satisfaction in the same way as final answers. Their contributions lie in the information they provide toward satisfying each rubric, but assessing it in isolation can repeatedly credit support established earlier. We therefore ground process credit in the evidence accumulated for each rubric, maintaining support histories against which an LLM judge evaluates additional support from newly obtained information.

Following this principle, we introduce Dr.Credit, a rubric-based credit assignment framework for deep research agents. As illustrated in Figure 2, Dr.Credit assigns credit to tool turns based on their contributions toward satisfying the task’s rubrics and combines the resulting process advantages with GRPO outcome advantages for policy optimization. We first formalize the underlying rubricgrounded process credit and the maintenance of support histories (Section 3.1), then describe how the resulting credits enter policy optimization (Section 3.2). Section 3.3 presents Dr.Credit as a concrete instantiation of these mechanisms for a basic deep research agent.

## 3.1 RUBRIC-GROUNDED PROCESS CREDIT

Rubric-Grounded Evidence Support. For a question q with rubric set $\mathcal { R } _ { q }$ , let $e _ { t }$ denote the information in the environment’s response at tool turn t of rollout O. For each rubric, let $S _ { k } ( q )$ be the space of evidence-grounded support statements for the k-th rubric, including facts, premises, relations, and qualifications that help fulfill its requirement without prescribing a canonical answer or an action sequence. While $s _ { k } ( q , \hat { a } )$ measures how well the final answer satisfies the rubric, process assessment identifies the support that $e _ { t }$ establishes toward meeting the same requirement.

Per-Rubric Support Histories. To distinguish new support from previously established information, we maintain a history $H _ { t , k } \subseteq S _ { k } ( q )$ containing the support points accepted for the k-th rubric before turn t. These histories start empty, $H _ { 0 , k } = \emptyset$ , and evolve independently within each rollout. Their collection, $\mathcal { H } _ { t } = ( H _ { t , 1 } , \dots , H _ { t , K } )$ , forms the history bank in Figure 2, preserving prior support as the rubric-specific reference for assessing subsequent information.

History-Aware Credit Assignment. The assessor $\mathcal { I }$ compares the current information $e _ { t }$ with histories $\mathcal { H } _ { t } .$ , evaluating additional support per rubric and identifying the support increments:

$$
\begin{array} { r } { ( \mathbf { g } _ { t } , \Delta \mathcal { H } _ { t } ) = \mathcal { I } ( q , \mathcal { R } _ { q } , e _ { t } , \mathcal { H } _ { t } ) . } \end{array}\tag{2}
$$

Here, $\mathbf { g } _ { t } = ( g _ { t , k } ) _ { k = 1 } ^ { K }$ contains nonnegative per-rubric contributions, and $\Delta \mathcal { H } _ { t } = ( \Delta H _ { t , k } ) _ { k = 1 } ^ { K }$ contains the corresponding support increments, with $\Delta H _ { t , k } \subseteq S _ { k } ( q )$ . Each $g _ { t , k }$ measures the support that $e _ { t }$ adds relative to $H _ { t , k } ,$ , with zero assigned when no additional support is established; the sum of these contributions defines the rubric-grounded credit for the current turn as

$$
c _ { t } ^ { r g } = \sum _ { k = 1 } ^ { K } g _ { t , k } .\tag{3}
$$

Once the contributions have been assessed against the existing histories, the accepted support increments update the reference available to subsequent turns through the per-rubric union

$$
H _ { t + 1 , k } = H _ { t , k } \cup \Delta H _ { t , k } .\tag{4}
$$

Assessment reads the history before the update, so the current evidence is compared with prior support before its accepted points enter the history; an empty increment leaves that history unchanged.

## 3.2 POLICY OPTIMIZATION WITH PROCESS CREDIT

Process credits measure additional rubric support from each turn, while the outcome reward evaluates the final answer. To use both signals in policy optimization, the old policy $\pi _ { \mathrm { o l d } }$ samples $G > 1$ rollouts for each question–rubric pair $\left( q , \mathcal { R } _ { q } \right)$ in the training set D. Let $c _ { i , t }$ be the process credit for non-final turn $t < T _ { i }$ of rollout $O _ { i }$ , where $T _ { i }$ indexes its final turn, and let $r _ { i }$ denote its rollout-level outcome reward. We normalize both signals separately within each rollout group to obtain

$$
A _ { i , t } ^ { r g } = \frac { c _ { i , t } - \mu ^ { r g } } { \sigma ^ { r g } } , \qquad A _ { i } ^ { O } = \frac { r _ { i } - \mu ^ { O } } { \sigma ^ { O } } .\tag{5}
$$

The process statistics $\mu ^ { r g }$ and $\sigma ^ { r g }$ are the mean and population standard deviation over all nonfinal policy-turn credits in the rollout group, including zero-credit turns, with equal weight per turn regardless of the length of its rollout. The outcome statistics $\mu ^ { O }$ and $\sigma ^ { O }$ retain the GRPO mean and sample standard deviation over the G rollout rewards obtained for the same question.

These separately normalized advantages provide the two inputs to the fusion step on the right of Figure 2, where the process term supplements the outcome advantage on each non-final turn:

$$
\widetilde { A } _ { i , t } = \left\{ \begin{array} { l l } { A _ { i } ^ { O } + A _ { i , t } ^ { r g } , } & { 0 \leq t < T _ { i } , } \\ { A _ { i } ^ { O } , } & { t = T _ { i } . } \end{array} \right.\tag{6}
$$

To apply this turn-level signal to token-level updates, let $\mathcal { P } _ { i }$ contain the policy-generated token positions in $O _ { i } ,$ , and let $t ( i , j )$ identify the turn containing token $x _ { i , j }$ . All tokens in that turn share $\widetilde { A } _ { i , t ( i , j ) }$ , while the policy ratio remains $\begin{array} { r } { \rho _ { i , j } ( \theta ) = \frac { \pi _ { \theta } \left( x _ { i , j } | q , x _ { i , < j } \right) } { \pi _ { \mathrm { o l d } } \left( x _ { i , j } | q , x _ { i , < j } \right) } } \end{array}$ , where $x _ { i , < j }$ includes earlier policy tokens and environment observations. Substituting $A _ { i , t ( i , j ) }$ into the GRPO surrogate gives the following objective, averaged over all policy-generated tokens in the sampled rollout group

$$
\begin{array} { r l } & { \mathcal { I } _ { \mathrm { D r } , \mathrm { C r e d i t } } ( \theta ) = \mathbb { E } _ { ( q , \mathcal { R } _ { q } ) \sim \mathcal { D } , \{ O _ { i } \} \sim \pi _ { \mathrm { o l d } } ( \cdot | q ) } \Bigg [ \frac { 1 } { \sum _ { i = 1 } ^ { G } | \mathcal { P } _ { i } | } \displaystyle \sum _ { i = 1 } ^ { G } \sum _ { j \in \mathcal { P } _ { i } } \operatorname* { m i n } \Bigl ( \rho _ { i , j } ( \theta ) \widetilde { A } _ { i , t ( i , j ) } , } \\ & { \qquad \mathrm { c l i p } \bigl ( \rho _ { i , j } ( \theta ) , 1 - \epsilon , 1 + \epsilon \bigr ) \widetilde { A } _ { i , t ( i , j ) } \Bigr ) - \beta \mathbb { D } _ { \mathrm { K L } } ( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } ) \Bigg ] . } \end{array}\tag{7}
$$

Here, ϵ controls probability-ratio clipping, and the KL term denotes token-averaged regularization against reference policy $\pi _ { \mathrm { r e f } } ,$ with coefficient $\beta .$ The dashed masks in Figure 2 exclude environment observations from the loss while retaining them in the policy’s conditioning context.

## 3.3 RUBRIC-GROUNDED CREDIT FOR DEEP RESEARCH

![](images/4bcd644526ecadb7810217037ca7405885c88f723282eb52b22af69479db15b0.jpg)  
Figure 3: Composite credit for Search turns in Dr.Credit. Upper green branch: historyaware Visit credits are attributed to the search through URL matching. Lower yellow branch: an independent judge matches snippets from results without later visits to rubrics, without reading or updating Visit histories. Navigation and snippet credits are added to obtain Search credit.

Dr.Credit instantiates the preceding credit assignment and policy optimization mechanisms for a basic deep research agent equipped with only two common tools. Search discovers candidate sources and returns snippets, while $\nabla \mathtt { i } \mathtt { s i t }$ extracts the content of selected webpages. Visit provides the main body of evidence for the final report and is the primary focus of our credit assessment. Search receives credit for discovering sources that yield additional support and for evidence available directly in its snippets, as illustrated in Figure 3. The resulting Visit and Search credits, denoted $c _ { t } ^ { v }$ and $c _ { t } ^ { s }$ , provide $c _ { i , t }$ for policy optimization in Section 3.2.

For a Visit turn, the page evidence serves as $e _ { t }$ . An LLM judge implements $\mathcal { I }$ by assessing the additional support this evidence provides relative to each rubric’s history $H _ { t , k }$ and assigning a contribution score $g _ { t , k }$ . These per-rubric scores are aggregated into Visit credit $c _ { t } ^ { v }$ following the summation in Equation 3. For positive assessments, the judge extracts support points in the current page to form $\Delta \bar { H } _ { t , k }$ , which updates the corresponding history after scoring according to Equation 4.

For a Search turn, we match normalized result URLs to later visits within the same rollout and aggregate the matched Visit credits into navigation credit $c _ { t } ^ { \mathrm { n a v } }$ . The upper branch of Figure 3 illustrates this attribution for visits to $u _ { a }$ and $u _ { b }$ . Results without later visits enter the complementary snippet branch, where an independent LLM judge identifies rubric support directly in their snip pets. Each accepted rubric match contributes η to snippet credit $c _ { t } ^ { \mathrm { s n p } }$ ; in the illustrated example, the snippet from $u _ { c }$ supports $R _ { 1 }$ and $R _ { 3 } ,$ , yielding two such contributions. The brief, scattered information in snippets motivates coarse rubric-level matching without reading or updating Visit histories. Navigation and snippet credits are combined additively into Search credit $c _ { t } ^ { s }$

## 4 EXPERIMENTS

Table 1: Overall performance on four deep research benchmarks. Dr.Credit outperforms all open deep research baselines on every metric, including all submetrics, and is competitive with proprietary systems. Average is the unweighted mean of the four benchmark-level scores. Bold denotes the best scores across the lower three model groups. <sup>∗</sup> and <sup>†</sup> indicate results taken from DeepRubric (Zhu et al., 2026a) and ResearchRubrics (Sharma et al., 2026), respectively.
<table><tr><td rowspan="2">Methods</td><td colspan="3">DeepRubric (val)</td><td rowspan="2">ResearchQA</td><td colspan="5">DeepResearchBench</td><td rowspan="2">ResearchRubrics Average</td><td rowspan="2"></td></tr><tr><td></td><td>Overall Factual Logical</td><td></td><td>Overall</td><td>Comp.</td><td> Depth Instr. Read.</td><td></td><td></td></tr><tr><td>Frontier Proprietary Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Kimi K3 + Our Tools</td><td>71.5</td><td>55.5</td><td>90.2</td><td>76.7</td><td>48.4</td><td>47.2</td><td>47.5</td><td>50.2</td><td>49.5</td><td>58.0</td><td>63.6</td></tr><tr><td>GPT-5.6-luna + Our Tools</td><td>73.2</td><td>55.6</td><td>93.9</td><td>73.2</td><td>48.8</td><td>47.3</td><td>48.4</td><td>50.4</td><td>49.8</td><td>56.6</td><td>63.0</td></tr><tr><td>DeepSeek-V4-Pro + Our Tools</td><td>68.4</td><td>52.7</td><td>87.0</td><td>78.0</td><td>44.5</td><td>43.5</td><td>43.7</td><td>46.1</td><td>45.6</td><td>50.8</td><td>60.4</td></tr><tr><td>Opus 4.8 + Our Tools</td><td>71.0</td><td>55.2</td><td>89.7</td><td>72.9</td><td>45.5</td><td>43.6</td><td>44.8</td><td>47.7</td><td>47.6</td><td>51.5</td><td>60.2</td></tr><tr><td>GLM-5.3 + Our Tools</td><td>68.9</td><td>54.2</td><td>85.9</td><td>72.1</td><td>45.6</td><td>44.4</td><td>45.5</td><td>46.1</td><td>46.0</td><td>52.5</td><td>59.8</td></tr><tr><td>Perplexity Deep Research</td><td>=</td><td>一</td><td>1</td><td>75.3*</td><td>42.3*</td><td>40.7*</td><td>39.3*</td><td>46.4*</td><td>44.3*</td><td>48.7†</td><td>=</td></tr><tr><td>Gemini Deep Research</td><td>-</td><td>一</td><td>一</td><td>68.5*</td><td>48.8*</td><td>48.5*</td><td>48.5*</td><td>49.2*</td><td>49.4*</td><td>61.5†</td><td>-</td></tr><tr><td>OpenAI Deep Research</td><td>=</td><td>-</td><td>=</td><td>79.2*</td><td>46.9*</td><td>46.8*</td><td>45.2*</td><td>49.2*</td><td>47.1*</td><td>59.7†</td><td>-</td></tr><tr><td>Naive RAG</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-8B + RAG</td><td>31.6</td><td>20.9</td><td>44.2</td><td>46.0</td><td>22.3</td><td>19.0</td><td>14.3</td><td>32.8</td><td>26.6</td><td>26.2</td><td>31.5</td></tr><tr><td>Qwen3.5-9B + RAG</td><td>42.5</td><td>25.4</td><td>63.4</td><td>47.2</td><td>26.1</td><td>22.5</td><td>18.6</td><td>36.1</td><td>31.3</td><td>32.9</td><td>37.2</td></tr><tr><td>Qwen3.6-35B-A3B + RAG</td><td>57.7</td><td>38.1</td><td>81.0</td><td>59.1</td><td>36.7</td><td>33.7</td><td>32.3</td><td>44.3</td><td>39.3</td><td>43.6</td><td>49.3</td></tr><tr><td>Open Deep Research Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ASearcher-Web-7B</td><td>12.3</td><td>7.8</td><td>16.9</td><td>19.4*</td><td>7.8*</td><td>5.1*</td><td>1.7*</td><td>15.2*</td><td>11.8*</td><td>7.6</td><td>11.8</td></tr><tr><td>Search-R1-7B</td><td>10.3</td><td>7.2</td><td>14.3</td><td>27.9*</td><td>9.5*</td><td>5.2*</td><td>2.1*</td><td>18.6*</td><td>16.8*</td><td>5.2</td><td>13.2</td></tr><tr><td>WebExplorer-8B</td><td>45.0</td><td>37.0</td><td>53.3</td><td>64.8*</td><td>36.7*</td><td>33.7*</td><td>28.5*</td><td>45.7*</td><td>42.2*</td><td>29.5</td><td>44.0</td></tr><tr><td>Tongyi DeepResearch-30B-A3B</td><td>57.3</td><td>44.3</td><td>72.4</td><td>66.7*</td><td>40.6*</td><td>39.1*</td><td>34.3*</td><td>46.8*</td><td>45.4*</td><td>37.1</td><td>50.4</td></tr><tr><td>Qwen3-8B-Based Deep Research Agents</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-8B + Our Tools</td><td>40.5</td><td>30.0</td><td>52.8</td><td>48.1</td><td>28.8</td><td>26.6</td><td>23.3</td><td>35.4</td><td>33.0</td><td>33.0</td><td>37.6</td></tr><tr><td>DeepRubric-8B + Our Tools</td><td>62.5</td><td>43.6</td><td>85.6</td><td>73.5</td><td>41.6</td><td>39.7</td><td>39.1</td><td>45.4</td><td>44.3</td><td>43.1</td><td>55.2</td></tr><tr><td>DR Tulu-8B + Our Tools</td><td>60.4</td><td>41.7</td><td>83.3</td><td>72.9</td><td>42.9</td><td>41.2</td><td>42.4</td><td>45.6</td><td>43.2</td><td>45.8</td><td>55.5</td></tr><tr><td>Qwen3-8B-SFT</td><td>59.3</td><td>45.2</td><td>75.3</td><td>69.2</td><td>40.3</td><td>39.3</td><td>34.6</td><td>45.4</td><td>41.8</td><td>40.9</td><td>52.4</td></tr><tr><td>Qwen3-8B-GRPO</td><td>67.4</td><td>52.5</td><td>84.0</td><td>75.4</td><td>43.2</td><td>42.1</td><td>40.9</td><td>46.8</td><td>44.0</td><td>46.8</td><td>58.2</td></tr><tr><td>Dr.Credit (Ours)</td><td>70.6</td><td>54.0</td><td>90.0</td><td>79.8</td><td>46.1</td><td>45.1</td><td>44.7</td><td>48.7</td><td>46.7</td><td>50.3</td><td>61.7</td></tr></table>

## 4.1 EXPERIMENTAL SETUP

Datasets & Metrics. DeepRubric Zhu et al. (2026a) provides our training data, pairing research queries with evidence-grounded rubric sets; we randomly hold out 128 examples for in-domain validation. Out-of-domain evaluation covers three research benchmarks: ResearchQA Yifei et al. (2026) for scholarly question answering, DeepResearchBench Du et al. (2026) for comprehensive report generation, and ResearchRubrics for open-ended tasks with expert-written rubrics. We report weighted rubric satisfaction on the DeepRubric validation set, rubric coverage on ResearchQA, weighted rubric compliance on ResearchRubrics, and report-quality scores on DeepResearchBench, all on a 100-point scale. Data construction and evaluation settings are detailed in Appendix A.

Baselines. Our baselines cover four groups: (1) frontier models equipped with our tools, including Kimi K3 Team et al. (2026), DeepSeek-V4-Pro Xu et al. (2026), and GLM-5.3 Zeng et al. (2026), together with commercial deep research services; (2) Qwen-based naive RAG Yang et al. (2025); (3) open search agents, including ASearcher Gao et al. (2026), Search-R1 Jin et al. (2025), WebExplorer Liu et al. (2025), and Tongyi DeepResearch Team et al. (2025); and (4) open deep research agents built on Qwen3-8B, including DeepRubric-8B Zhu et al. (2026a), DR Tulu-8B Shao et al. (2026), and our SFT and GRPO baselines, with the base Qwen3-8B agent included as a reference. The GRPO baseline uses outcome rewards without process credit and shares the same SFT initialization, training data, outcome reward, training budget, and evaluation setup as Dr.Credit. Baseline evaluation settings and result provenance are detailed in Appendix A.3.

Implementation Details. For credit assignment (Section 3.3), Visit assessments of no additional support, partial new support, and high-value new evidence correspond to $g _ { t , k } = 0 , 0 . 2 ,$ , and $0 . 5 ,$ respectively, while each accepted rubric match in a Search snippet contributes $\eta = 0 . 1 \mathrm { t o } c _ { t } ^ { \mathrm { s n p } }$ . To keep turn-level credits on a shared scale before process-advantage normalization, the sums defining Visit credit $c _ { t } ^ { v } .$ , navigation credit $c _ { t } ^ { \mathrm { n a v } }$ , and Search credit $c _ { t } ^ { s }$ are capped at 1. An offline analysis in Appendix B.3 reports similar normalized process advantages across alternative parameter settings.

With these credit settings, RL starts from a Qwen3-8B checkpoint obtained after two epochs of SFT on 1,505 filtered research trajectories. Dr.Credit is then trained on 7,149 query–rubric pairs for 200 steps, with 32 queries per batch and $G = 8$ rollouts per query. For both process and outcome supervision, Qwen3.6-35B-A3B performs history-aware Visit assessment and independent Search snippet matching, as well as evaluating rubric satisfaction and citation quality in final reports. The corresponding judge prompts are provided in Appendix D, while Appendix A documents data construction, training and interaction settings, and the definitions of each outcome reward component.

## 4.2 OVERALL PERFORMANCE

Table 1 shows that Dr.Credit is competitive with frontier models, surpassing DeepSeek-V4-Pro, Opus 4.8, and GLM-5.3 in average score while approaching Kimi K3 and GPT-5.6-luna. With an 8B backbone, Dr.Credit achieves an average score of 61.7 and outperforms all models in the Naive RAG and Open Deep Research Models groups across every metric and submetric, including larger models such as Qwen3.6-35B-A3B and Tongyi DeepResearch-30B-A3B.

Among deep research agents built on the same Qwen3-8B base model, Dr.Credit achieves the highest scores on all four benchmarks and their submetrics. The results for DeepRubric-8B and DR Tulu-8B in Table 1 are from our evaluations using the same web search and web visit tools as Dr.Credit (see Appendix A.3). Under matched training and evaluation settings, Dr.Credit improves over the GRPO baseline on all four benchmarks, with a 3.5-point average gain from adding process credit. Beyond the in-domain DeepRubric validation set, Dr.Credit achieves clear gains on all three external benchmarks, suggesting that our credit assignment framework improves out-of-domain generalization.

## 4.3 ABLATION STUDY

To understand which design choices contribute to Dr.Credit’s performance, we conduct three groups of ablation experiments. Specifically, we evaluate the process credit design, the benefits of combining outcome and process supervision, and the contribution of per-rubric evidence history.

Table 2: Ablation of process credit designs. Visit denotes history-aware credit for Visit turns; gray shading marks the final Dr.Credit configuration. DRub, RQA, DRB, and RR denote DeepRubric (val), ResearchQA, DeepResearchBench, and ResearchRubrics, respectively. Avg is the arithmetic mean of the four benchmark scores; bold marks the best score per benchmark and the highest Avg.
<table><tr><td>Visit</td><td>Navigation</td><td>Snippets</td><td>Discount</td><td>DRub</td><td>RQA</td><td>DRB</td><td>RR</td><td> $\operatorname { A v g }$ </td></tr><tr><td>√</td><td>x</td><td>x</td><td>x</td><td>68.8</td><td>76.8</td><td>44.7</td><td>48.6</td><td>59.7</td></tr><tr><td>√</td><td>√</td><td>x</td><td>x</td><td>70.8</td><td>78.4</td><td>45.0</td><td>47.2</td><td>60.4</td></tr><tr><td>√</td><td>√</td><td>√</td><td>x</td><td>70.6</td><td>79.8</td><td>46.1</td><td>50.3</td><td>61.7</td></tr><tr><td>√</td><td>√</td><td>x</td><td>√</td><td>68.2</td><td>77.3</td><td>44.5</td><td>45.9</td><td>59.0</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>68.0</td><td>78.0</td><td>45.3</td><td>48.1</td><td>59.9</td></tr></table>

Process Credit Design. Table 2 reports the performance of agents trained with different process credit designs. Relative to Visit-only credit, adding navigation attribution from subsequent Visits improves scores on DeepRubric, ResearchQA, and DRB, while reducing the score on RR. Adding independent snippet scoring for unvisited search results further improves scores on ResearchQA, DRB, and RR, yielding the highest four-benchmark average of 61.7, with only a 0.2-point decrease on DeepRubric. These comparisons support assigning credit to both source discovery and evidence available directly in search snippets. Discounted propagation with $\gamma = 0 . 9 5$ lowers scores on all four benchmarks, both with and without snippet scoring, relative to the corresponding variants without propagation. Together, these results support the final process credit design adopted in Dr.Credit.

Outcome and Process Supervision. As shown in Table 3(a), removing the process advantage $A ^ { r g }$ reduces Dr.Credit to GRPO and lowers the three-benchmark average by 3.5 points. This gap demonstrates the value of rubric-grounded process credit as additional supervision for research turns. However, relying on process supervision alone for these turns also lowers the average by 2.3 points. These results suggest complementary roles for the two signals: process credit guides evidence acquisition at individual turns, while outcome supervision helps align these decisions with the overall quality of final reports.

Table 3: Ablations of Dr.Credit: (a) removing either outcome or process advantage from research-turn supervision; (b) removing evidence history when computing process credit. Each ablation is relative to full Dr.Credit. Values are score changes, with $\Delta \mathrm { { A v g } _ { 3 } }$ averaged across the three benchmarks.
<table><tr><td>Removed</td><td>∆DRub</td><td>∆RQA</td><td>∆DRB</td><td>∆Avg3</td></tr><tr><td>(a) Outcome and Process Supervision</td><td></td><td></td><td></td><td></td></tr><tr><td>Process  $A ^ { r g }$ </td><td>-3.2</td><td>-4.4</td><td>-2.9</td><td>-3.5</td></tr><tr><td>Outcome  $A ^ { O }$ </td><td>-1.2</td><td>-3.6</td><td>-2.1</td><td>-2.3</td></tr><tr><td>(b) Evidence History</td><td></td><td></td><td></td><td></td></tr><tr><td>History</td><td>-3.0</td><td>-3.9</td><td>-2.1</td><td>-3.0</td></tr></table>

Effect of Evidence History. Table 3(b) shows that computing process credit without evidence history lowers scores on all three benchmarks, with an average decrease of 3.0 points. A page can support a rubric without extending the evidence already available, so scoring it in isolation risks rewarding repeated retrieval. The paired scoring analysis in Appendix B.2 compares Visit credits and rubric-level judgments with and without prior support on fixed trajectories. These diagnostics complement the training ablation, suggesting that per-rubric histories help supervise the additional evidence acquired at each turn while accounting for support already available within the trajectory.

## 4.4 IN-DEPTH ANALYSIS

![](images/5fc17130d76ad7a4637bc403e031df8006e983c070aedaba9c940841cd7ae796.jpg)

![](images/d31cab689616b79740fd930c55c0a4bd783ca33a2e9107839705ef20f9f0ecac.jpg)

![](images/5b2b646f5eb552737b76fce02a7c849a7a1c5b8d519e9cbb195250ab154bd3a5.jpg)  
Figure 4: Evidence acquisition and report quality on DeepRubric. (a) Mean cumulative evidence proxy score. (b) Mean proxy score per actual research turn. (c) Report rubric scores under inferencetime turn budgets. Blue shading in (b) and (c) indicates differences between method means.

To investigate what underlies Dr.Credit’s strong performance, we analyze how it acquires evidence along research trajectories and assess report quality under constrained turn budgets. Figure 4 summarizes these analyses on the DeepRubric validation set. Evidence acquisition is tracked using a credit-based proxy that accumulates uncapped credit from retrieved pages and search snippets, excluding navigation credit. For each trajectory prefix, the proxy includes only contributions from content returned up to that point. Scoring and aggregation details are provided in Appendix B.1.

Under this proxy, Dr.Credit accumulates evidence more rapidly in early research turns, reaching a cumulative score of 2.94 within 6 turns compared with GRPO’s 2.83 within 8 (Figure 4(a)). Dr.Credit leads at every measured prefix through ten turns, while GRPO slightly surpasses it at twelve and fourteen turns. Together with the decline in estimated research-turn counts during training (Appendix C.1, Figure 10(b)), these results suggest that Dr.Credit maintains effective evidence acquisition as its trajectories become shorter. Figure 4(b) reports mean per-turn proxy scores by dividing each trajectory’s cumulative score by the number of research turns actually completed within the prefix and then averaging these ratios across trajectories (see Appendix B.1 for detailed calculations). Dr.Credit scores higher at every measured prefix, reaching 0.53 versus $\mathrm { G R P O } ^ { \prime } \mathrm { s } 0 . 3 6$ at eight turns (+47.2%), indicating more efficient evidence acquisition per research turn under this proxy.

Separate evaluations under inference-time turn budgets assess final-report quality (Figure 4(c)), following the report-generation protocol in Appendix B.1. Dr.Credit outperforms GRPO at all seven tested budgets. With an eight-turn budget, it achieves a rubric score of 67.69, exceeding GRPO’s best observed score of 66.41 at twelve turns. Dr.Credit thus exceeds GRPO’s best observed quality with a one-third smaller allowed turn budget. These results suggest that Dr.Credit follows more efficient research trajectories to produce higher-quality reports under constrained turn budgets.

## 4.5 COMPUTATIONAL EFFICIENCY

Although Dr.Credit achieves strong performance, its process-credit scoring may introduce substantial computational overhead during training. Figure 5 therefore compares the learning progress and cumulative training time of Dr.Credit and GRPO over 200 steps, including process-credit scoring.

![](images/18a71f69eaafc6fea5acc6ba4cacde4cacc57967730eb07197acfef2efd810c0.jpg)

![](images/a167b3682bd58dbc0b29f1103975806c87a183c453c8422f8f8f37d720e13ed7.jpg)  
Figure 5: Learning progress and recorded training time. (a) Validation rubric scores on DeepRubric; GRPO without SFT is included as an initialization reference. (b) Cumulative recorded training-step time over 200 steps, including rollout collection, reward scoring, and policy updates. The inset shows GRPO time minus Dr.Credit time over the first 40 steps, in minutes.

Learning progress. As illustrated in Figure 5(a), our rubric-grounded credit assignment enables faster empirical convergence and higher final validation performance than GRPO baseline. Dr.Credit surpasses GRPO’s final validation score by step 120 and finishes at step 200 with a 3.2-point advantage. The supplementary training curves in Appendix C.1 show consistent improvements.

Training time. Figure 5(b) shows an early cumulative-time disadvantage for Dr.Credit. This is expected given the additional workload of process-credit scoring, as reported in Table 7. From step 34 onward, however, Dr.Credit maintains a lower cumulative training time than GRPO, saving 11.18 hours (14.9%) over 200 steps with scoring included. Dr.Credit learns more efficient research behavior that reduces interaction and evaluation work, while asynchronous execution in the training framework allows scoring to overlap with rollout generation. These factors help explain the time savings under the measured setup, with a detailed analysis provided in Appendix C.2.

## 5 RELATED WORK

Rubric-Based Rewards for Deep Research. Instance-specific rubrics and checklists provide evaluation criteria for open-ended tasks, with recent work improving their coverage and discriminability Sharma et al. (2026); Viswanathan et al. (2025); Shen et al. (2026). In deep research, DR Tulu Shao et al. (2026) evolves rubrics using retrieved evidence and on-policy rollouts, while Deep-Rubric Zhu et al. (2026a) and DR-Rubric Mei et al. (2026) ground rubrics in evidence from structured expansion or agentic search. Learning Query-Specific Rubrics Lv et al. (2026) aligns rubric generation with human preferences for research reports, and QUEST Xie et al. (2026a) uses rubric trees to define rewards across research tasks. RubricEM Li et al. (2026) extends rubric supervision to intermediate research stages, assigning a shared advantage to all policy tokens within each stage.

Fine-Grained Credit Assignment. Classical temporal credit assignment uses TD(λ) Sutton (1988) to propagate prediction errors through eligibility traces; generalized advantage estimation Schulman et al. (2016) builds on this principle to balance bias and variance in policy-gradient estimation. RUDDER Arjona-Medina et al. (2019) addresses delayed rewards by redistributing returns according to changes in predicted return. For LLM agents, GiGPO Feng et al. (2025) compares actions from repeated anchor states across trajectories; HCAPO Tan et al. (2026) uses hindsight reasoning to refine step-level value estimates; and HiPER Peng et al. (2026) estimates advantages for both planning and execution. A complementary line measures progress toward a target answer: IGPO Wang et al. (2026) rewards increases in the policy’s probability of generating the correct answer; TIPS Xie et al. (2026b) shapes rewards using a teacher model’s answer potential; and ∆Belief-RL Auzina et al. (2026) rewards changes in the agent’s belief in the target solution.

## 6 CONCLUSION

In this work, we propose rubric-grounded credit, which extends task rubrics from final-answer evaluation to process supervision. Intermediate steps receive credit for the additional support they provide relative to each rubric’s evidence history, without requiring a canonical answer. Applied to deep research, this approach underlies Dr.Credit, a training framework combining process and GRPO outcome advantages to supervise research decisions and final-report quality. Evaluations across four benchmarks show that Dr.Credit outperforms the evaluated open deep research baselines and generalizes well to out-of-domain tasks. Even with only 8B parameters, the trained Qwen3-8B agent achieves average performance competitive with that of the evaluated frontier proprietary models. Beyond these benchmark results, further analyses suggest that the trained agent acquires useful evidence more efficiently and produces higher-quality reports under the same research-turn budgets. These findings support rubric-based supervision of evidence acquisition and motivate extending rubric-grounded credit to other open-ended tasks trained with rubric-based reinforcement learning.

## REFERENCES

Jose A. Arjona-Medina, Michael Gillhofer, Michael Widrich, Thomas Unterthiner, Johannes Brandstetter, and Sepp Hochreiter. RUDDER: Return decomposition for delayed rewards. In Advances in Neural Information Processing Systems (NeurIPS), volume 32. Curran Associates, Inc., 2019.

Rahul K. Arora, Jason Wei, Rebecca Soskin Hicks, Preston Bowman, Joaquin Quiñonero-Candela, Foivos Tsimpourlas, Michael Sharman, Meghan Shah, Andrea Vallone, Alex Beutel, et al. Health-Bench: Evaluating large language models towards improved human health. arXiv preprint arXiv:2505.08775, 2025.

Ilze Amanda Auzina, Joschka Strüber, Sergio Hernández-Gutiérrez, Shashwat Goel, Ameya Prabhu, and Matthias Bethge. Intrinsic credit assignment for long horizon interaction. In International Conference on Machine Learning (ICML), 2026.

Mingxuan Du, Benfeng Xu, Chiwei Zhu, Licheng Zhang, Xiaorui Wang, and Zhendong Mao. Deepresearch bench: A comprehensive benchmark for deep research agents. In International Confer ence on Learning Representations (ICLR), volume 2026, pp. 42414–42448, 2026.

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-group policy optimization for LLM agent training. In Advances in Neural Information Processing Systems (NeurIPS), volume 38, pp. 46375–46408, 2025.

Jiaxuan Gao, Wei Fu, Minyang Xie, Shusheng Xu, Chuyi He, Zhiyu Mei, Banghua Zhu, and Yi Wu. Unlocking long-horizon agentic search with large-scale end-to-end rl. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations (ICLR), volume 2026, pp. 107331–107352, 2026.

Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Yunzhong He, Bing Liu, and Sean Hendryx. Rubrics as rewards: Reinforcement learning beyond verifiable domains. In International Conference on Learning Representations (ICLR), volume 2026, pp. 127924–127945, 2026.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.

Gaotang Li, Bhavana Dalvi Mishra, Zifeng Wang, Jun Yan, Yanfei Chen, Chun-Liang Li, Long T Le, Rujun Han, George Lee, Hanghang Tong, et al. Rubricem: Meta-rl with rubric-guided policy decomposition beyond verifiable rewards. arXiv preprint arXiv:2605.10899, 2026.

Junteng Liu, Yunji Li, Chi Zhang, Jingyang Li, Aili Chen, Ke Ji, Weiyu Cheng, Zijia Wu, Chengyu Du, Qidi Xu, et al. Webexplorer: Explore and evolve for training long-horizon web agents. arXiv preprint arXiv:2509.06501, 2025.

Yijun Lu, Rui Ye, Jiajun Wang, Yuwen Du, Tian Jin, Songhua Liu, and Siheng Chen. Abseeker: Training long-horizon search agents via answer-backtracked credit assignment. arXiv preprint arXiv:2608.05102, 2026.

Changze Lv, Jie Zhou, Wentao Zhao, Jingwen Xu, Shihan Dou, Zisu Huang, Muzhao Tian, Xiaohua Wang, Zhengkang Guo, Yang Liu, et al. Learning query-specific rubrics from human preferences for DeepResearch report generation. In Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2026.

Wangyi Mei, Zhouhong Gu, Zhenhan Bai, Yin Cai, Lefan Zhang, Zhenxin Ding, Bo Chen, Yan Gao, Yi Wu, Yao Hu, et al. Deep research as rubric for reinforcement learning. arXiv preprint arXiv:2606.01091, 2026.

Jiangweizhi Peng, Yuanxin Liu, Ruida Zhou, Charles Fleming, Zhaoran Wang, Alfredo Garcia, and Mingyi Hong. HiPER: Hierarchical plan–execute RL for multi-turn LLM agents. In International Conference on Machine Learning (ICML), 2026.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. Highdimensional continuous control using generalized advantage estimation. In International Conference on Learning Representations (ICLR), 2016.

Rulin Shao, Akari Asai, Shannon Zejiang Shen, Hamish Ivison, Varsha Kishore, Jingming Zhuo, Xinran Zhao, Molly Park, Samuel G Finlayson, David Sontag, et al. DR Tulu: Reinforcement learning with evolving rubrics for deep research. In International Conference on Machine Learning (ICML), 2026.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Manasi Sharma, Chen Bo Calvin Zhang, Chaithanya Bandi, Clinton Wang, Ankit Aich, Huy Nghiem, Tahseen Rabbani, Ye Htet, Brian Jang, Sumana Basu, et al. Researchrubrics: A benchmark of prompts and rubrics for evaluating deep research agents. In International Conference on Learning Representations (ICLR), volume 2026, pp. 90447–90472, 2026.

William F Shen, Xinchi Qiu, Chenxi Whitehouse, Lisa Alazraki, Shashwat Goel, Francesco Barbieri, Timon Willi, Akhil Mathur, and Ilias Leontiadis. Rethinking rubric generation for improving llm judge and reward modeling for open-ended tasks. arXiv preprint arXiv:2602.05125, 2026.

Richard S. Sutton. Learning to predict by the methods of temporal differences. Machine Learning, 3(1):9–44, 1988. doi: 10.1007/BF00115009.

Hui-Ze Tan, Xiao-Wen Yang, Hao Chen, Jie-Jing Shao, Yi Wen, Yuteng Shen, Weihong Luo, Xiku Du, Lan-Zhe Guo, and Yu-Feng Li. Hindsight credit assignment for long-horizon llm agents. arXiv preprint arXiv:2603.08754, 2026.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y Charles, et al. Kimi k3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026.

Tongyi DeepResearch Team, Baixuan Li, Bo Zhang, Dingchu Zhang, Fei Huang, Guangyu Li, Guoxin Chen, Huifeng Yin, Jialong Wu, Jingren Zhou, et al. Tongyi deepresearch technical report. arXiv preprint arXiv:2510.24701, 2025.

Vijay Viswanathan, Yanchao Sun, Xiang Kong, Meng Cao, Graham Neubig, and Sherry Wu. Checklists are better than reward models for aligning language models. In Advances in Neural Information Processing Systems (NeurIPS), volume 38, pp. 114728–114754, 2025.

Guoqing Wang, Sunhao Dai, Guangze Ye, Zeyu Gan, Wei Yao, Yong Deng, Xiaofeng Wu, et al. Information gain-based policy optimization: A simple and effective approach for multi-turn search agents. In International Conference on Learning Representations (ICLR), volume 2026, pp. 104502–104529, 2026.

Yuning Wu, Jiahao Mei, Ming Yan, Chenliang Li, Shaopeng Lai, Yuran Ren, Zijia Wang, Ji Zhang, Mengyue Wu, Qin Jin, et al. WritingBench: A comprehensive benchmark for generative writing. In Advances in Neural Information Processing Systems (NeurIPS), volume 38, 2025.

Jian Xie, Tianhe Lin, Zilu Wang, Yuting Ning, Yuekun Yao, Tianci Xue, Zhehao Zhang, Zhongyang Li, Kai Zhang, Yufan Wu, et al. Quest: Training frontier deep research agents with fully synthetic tasks. arXiv preprint arXiv:2605.24218, 2026a.

Yutao Xie, Nathaniel Thomas, Nick Hansen, Yang Fu, Li Li, and Xiaolong Wang. Tips: Turn-level information-potential reward shaping for search-augmented llms. In International Conference on Learning Representations (ICLR), volume 2026, pp. 156549–156584, 2026b.

Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al. Deepseek-v4: Towards highly efficient milliontoken context intelligence. arXiv preprint arXiv:2606.19348, 2026.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023.

Li S. Yifei, Allen Chang, Chaitanya Malaviya, and Mark Yatskar. ResearchQA: Evaluating scholarly question answering at scale across 75 fields with survey-mined questions and rubrics. Transactions ofthe Associationfor Computational Linguistics (TACL), 14:1365–1389, 2026.

Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, et al. Glm-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, and Zheyan Luo. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings ofthe 62nd annual meeting ofthe association for computational linguistics (ACL), pp. 400–410, 2024.

Minghang Zhu, Chuyang Wei, Junhao Xu, Yilin Cheng, Zhumin Chen, and Jiyan He. DeepRubric: Evidence-tree rubric supervision for efficient reinforcement learning of deep research agents. arXiv preprint arXiv:2606.17029, 2026a.

Qiang Zhu, Jiajun Wu, and Longyi Wang. LOTAPO: Leave-one-turn attribution for self-generated process rewards in multi-turn search reasoning. arXiv preprint arXiv:2607.13501, 2026b.

## A EXPERIMENTAL SETUP

## A.1 DATA AND TRAINING

The data originate from 9,062 DeepRubric question–rubric pairs (Zhu et al., 2026a). Excluding 1,785 questions represented in an intermediate SFT collection leaves 7,277 examples, of which 128 are randomly held out as DeepRubric-Val and 7,149 are used for RL. Exact question-text matching confirms that the final SFT dataset, RL training set, and validation set are pairwise disjoint.

For SFT, GLM-5.2 generates research trajectories from questions without access to their rubric sets, using the prompt in Figure 12. Retained trajectories pass checks on report structure, citations, and tool arguments, achieve a normalized report rubric score of at least 0.65, and contain 4–18 research turns followed by a report. All 1,505 retained trajectories enter SFT without a validation split, with their system instructions replaced by the shared research-agent prompt in Figure 11.

Full-parameter SFT of Qwen3-8B uses LLaMA-Factory (Zheng et al., 2024) for three epochs, with the second-epoch checkpoint initializing both RL methods. The cosine schedule and warmup span the full three-epoch run, so the initialization fixes the SFT duration at two epochs without validationbased checkpoint selection. Subsequent RL uses VeRL for 200 steps, with 32 questions and eight rollouts per question in each sampling batch; Table 4 lists the remaining hyperparameters.

Table 4: SFT and RL hyperparameters. The cosine schedule spans three SFT epochs; both RL methods start from the checkpoint after epoch two. Interaction limits are specified in Appendix A.3.
<table><tr><td>Stage</td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="5">SFT</td><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Learning-rate schedule</td><td>Cosine; 30% warmup</td></tr><tr><td>Per-device batch size / accumulation steps</td><td>1/4</td></tr><tr><td>Training epochs / selected epoch Precision</td><td>3/2 BF16</td></tr><tr><td colspan="2">RL</td><td> $5 \times 1 0 ^ { - 7 }$ </td></tr><tr><td rowspan="8"></td><td>Learning rate Questions per sampling batch</td><td>32</td></tr><tr><td>Rollouts per question (G)</td><td></td></tr><tr><td>Training steps</td><td>8</td></tr><tr><td>Rollout temperature / top-p</td><td>200</td></tr><tr><td></td><td>1.0 / 1.0</td></tr><tr><td>PPO epochs / clipping threshold KL coefficient</td><td>1/0.2</td></tr><tr><td></td><td>0</td></tr><tr><td>Process-advantage weight</td><td>1</td></tr></table>

GRPO and Dr.Credit share the initialization, training data, outcome reward, and training budget, with process supervision added only in Dr.Credit. Both runs use 16 PPU-ZW810E accelerators with 96 GB of memory each; auxiliary models run on separate nodes or through APIs. Qwen3.6-35B-A3B handles webpage extraction, report and citation assessment, Visit scoring, and independent snippet assessment, using the full agent and judge prompt templates in Appendix D.

## A.2 TOOLS AND INTERACTION LIMITS

The shared web environment exposes Search through Serper’s Google Search interface and Visit through Firecrawl. Each Search call accepts at most two queries and returns up to three results per query, including titles, URLs, snippets, and source IDs, within a combined 4,096-token response. A Visit call accepts at most two URLs; Qwen3.6-35B-A3B extracts evidence and a goalconditioned summary within 2,048 tokens per page (Figure 13). Each returned page carries its URL and source ID, while failed retrieval or extraction produces a tool error. The agent makes exactly one tool call in each research turn, which may include multiple queries or URLs within these limits.

## A.3 INFERENCE AND EVALUATION

The policy context limit is 32,768 tokens for SFT, RL, and the main-table evaluations. RL and evaluation allow at most 20 assistant turns, including the final answer, with up to 8,192 generated tokens per turn. Tool observations consume context, and the final report must fit within the remaining context. Each question receives one research attempt and one evaluated report; the separate turnbudget experiment is defined in Appendix B.1.

If the turn or context limit is reached without a valid report, tool use stops and the evaluated model completes the report from the original question and collected trajectory. Evaluation uses temperature 0.6, top-p 0.95, and top-k 20, except that Opus 4.8 uses provider defaults and GPT-5.6-luna omits top-k from API requests. DeepRubric and DR Tulu retain their method-specific prompts with tool calls adapted to the shared services. In Table 1, <sup>∗</sup> and <sup>†</sup> identify results reported by DeepRubric (Zhu et al., 2026a) and ResearchRubrics (Sharma et al., 2026), respectively; unmarked values come from the present evaluations. The Average column reports the unweighted four-benchmark mean only when all scores are available.

## A.4 BENCHMARK METRICS

Evaluation covers 128 DeepRubric-Val questions, the same 776-question ResearchQA subset used by DR Tulu (Shao et al., 2026), 100 DeepResearchBench questions, and 101 ResearchRubrics questions. DeepRubric-Val measures weighted rubric satisfaction, with factual and logical breakdowns in the main table. For ResearchQA, DeepSeek-V4-Flash scores rubric coverage at five levels, mapped to 0, 0.25, 0.5, 0.75, and 1 before averaging rubric scores within each evaluation question.

DeepResearchBench reports are cleaned with GPT-5.4-mini $( { \tt g p t - 5 . 4 - m i n i - 2 0 2 6 - 0 3 - 1 7 } )$ and scored with Gemini-3.1-Pro-Preview (gemini-3.1-pro-preview) under RACE, which reports comprehensiveness, insight, instruction following, readability, and an overall score. ResearchRubrics uses binary judgments from Gemini-2.5-Pro: the weighted sum of satisfied rubrics is divided by the sum of positive weights, retaining penalties from satisfied negative-weight rubrics. Benchmark scores aggregate questions on a 100-point scale without auxiliary training rewards.

## A.5 OUTCOME REWARD

Both methods use the following weighted sum of report quality and auxiliary rewards:

$$
{ \boldsymbol { r } } ^ { O } = 0 . 7 0  { \boldsymbol { r } } _ { \mathrm { r u b r i c } } + 0 . 0 5  { \boldsymbol { r } } _ { \mathrm { f o r m a t } } + 0 . 0 5  { \boldsymbol { r } } _ { \mathrm { s e a r c h } } + 0 . 1 5  { \boldsymbol { r } } _ { \mathrm { c i t e } } + 0 . 0 5  { \boldsymbol { r } } _ { \mathrm { l e n g t h } } .\tag{8}
$$

Rubric judgments range from 0 to 4 and are divided by four before weighted aggregation as in Equation 1. The format component checks the answer block, citation markup, query-bearing calls, and reasoning blocks; the call-count and report-length components increase linearly up to their caps. The call-count reward is shared by both methods and is separate from Dr.Credit’s evidence-based Search credit. If no nonempty final report can be extracted, the outcome reward is zero.

Citation reward is $r _ { \mathrm { c i t e } } = 0 . 4 r _ { \mathrm { I D } } + 0 . 6 r _ { \mathrm { s e m a n t i c } }$ , where r measures the fraction of well-formed source references that resolve to trajectory evidence. For each assessable claim, semantic support is the harmonic mean of joint source support and the fraction of individually supporting sources. Averaging these claim scores assigns zero to structurally unscoreable citations and excludes technical judge failures. Figures 14 and 15 provide the prompts.

## B SUPPLEMENTARY ANALYSES

## B.1 EVIDENCE ACQUISITION AND TURN BUDGETS

Figure 4 uses all 128 DeepRubric-Val questions, with one trajectory per question for each method and setting. Its prefix analysis accumulates uncapped Visit credit assessed against per-rubric history and independent snippet credit, excluding navigation credit. Snippets qualify only when their results remain unvisited in the complete trajectory, so eligibility is determined retrospectively; each prefix nevertheless includes contributions only from content already returned at that point.

For trajectory i with $T _ { i } > 0$ research turns, let $u _ { i , t }$ denote the uncapped evidence contribution and $L _ { i } ( b ) = \operatorname* { m i n } ( b , T _ { i } )$ . The plotted statistics at prefixes $b \in \{ 2 , 4 , 6 , 8 , \bar { 1 0 } , 1 2 , 1 4 \}$ are

$$
C _ { i } ( b ) = \sum _ { t = 1 } ^ { L _ { i } ( b ) } u _ { i , t } , \qquad { \overline { { C } } } ( b ) = \mathrm { m e a n } _ { i } C _ { i } ( b ) , \qquad { \overline { { U } } } ( b ) = \mathrm { m e a n } _ { i } { \frac { C _ { i } ( b ) } { L _ { i } ( b ) } } .\tag{9}
$$

Each ratio is calculated before averaging across questions; after a trajectory ends, both its cumulative credit and denominator remain fixed. The separate budget evaluation regenerates reports at each of

the same seven research-turn budgets, excluding the final-answer turn. At the budget limit, the evaluated model completes a report from the question and collected trajectory using Appendix A.3; each point in Figure 4(c) therefore scores a report generated under that budget.

## B.2 EFFECT OF EVIDENCE HISTORY

Fixed Dr.Credit and GRPO trajectories are rescored with Qwen3.6-35B-A3B, retaining or clearing prior Visit support points to isolate their effect on scoring. Rubrics, evidence, prompts, and filters for unusable or exactly duplicated pages remain fixed. Trajectories have a 16-turn research budget; scoring uses temperature zero without thinking and reuses identical empty-history judgments.

![](images/597df516a96f8aad4c34fe5b9afebd8447cd8b0061025f6dead3a90c854709ce.jpg)  
Figure 6: History-conditioned Visit credit on DeepRubric-Val questions. (a) Mean capped credit per Visit. (b) Uncapped Visit credit per actual research turn, including Search in the denominator but excluding Search credit from the numerator. Bars average per-trajectory statistics; error bars are 95% percentile intervals from 10,000 paired question-bootstrap resamples.

Figure 6 summarizes per-trajectory credit, while Figure 7 shows its distribution across Visits. With history, Dr.Credit retains higher mean credit per Visit and per research turn; both policies receive fewer capped scores and more intermediate scores. Figure 8 shows changes concealed by aggregation and capping through paired rubric-level transitions for nonempty per-rubric histories.

![](images/3a14a86465608c275d8456e168e5455a5fd913da7b17c614e44533ba11f8ea92.jpg)  
Figure 7: Capped Visit credit with and without history, pooling the same 365 Dr.Credit Visits or 668 GRPO Visits in each comparison. Each percentage uses the corresponding method’s total number of Visits as its denominator; intermediate credit lies strictly between zero and one.

The transitions include both decreases and increases in credit, reflecting how prior support changes the assessment of additional evidence. Among pairs assigned level 2 without history, 12.0% fall to zero for Dr.Credit and 20.9% for GRPO. These diagnostics complement the training ablation in Table 3(b).

![](images/0643eff43891391d1b000341cd1c07039fa126309434bde35ca3fe7a1fdbed8a.jpg)

![](images/7c6260c8c3ebfa4959aad7926cd8ad5456eb5795adedfaaa60ace75cb376aac8.jpg)  
Figure 8: Transitions with nonempty histories: 916 Visit–rubric pairs for Dr.Credit and 2,342 for GRPO. Rows give levels without history and columns levels with history; each row sums to 100% up to rounding. Both panels share a color scale; levels 0, 1, and 2 map to credit 0, 0.2, and 0.5.

## B.3 SENSITIVITY TO CREDIT PARAMETERS

The sensitivity analysis examines whether the training signal depends strongly on the credit parameters, providing empirical evidence for the default setting. To isolate their effect, replay varies Visit level scores, the shared cap, and snippet weight while holding trajectories, ordinal judgments, evidence histories, and native GRPO advantages fixed. The sample contains 519 groups of eight rollouts (4,152 rollouts and 32,966 non-final policy turns), with the original groups and zero-credit turns retained for normalization and a shared cap applied to Visit, navigation, and total Search credit.

Table 5 measures changes in the pooled process-advantage distribution using the 1-Wasserstein dis tance (W<sub>1</sub>), alongside RMSE and sign agreement between corresponding turns. After adding the fixed native GRPO advantage, fused sign agreement measures how often the direction of the combined signal is preserved; all turns receive equal weight, with values within $1 0 ^ { - 6 }$ of zero treated as neutral. For settings that require snippet judgments skipped under saturated navigation credit, unobserved match counts range from zero to the rubric count, and normalization of these intervals yields conservative upper bounds on distances and lower bounds on sign agreement across all turns.

Table 5: Sensitivity of normalized credit signals to parameter changes, with settings selected for signal stability from a 53-configuration sweep. The reference uses Visit levels (0, 0.2, 0.5), cap 1, and snippet weight 0.1; each row changes only the indicated parameters across the same 519 rollout groups and 32,966 turns. Inequalities denote conservative bounds from missing snippet judgments, while all other values are exact and sign agreement across corresponding turns is reported in percent.
<table><tr><td rowspan="2">Parameter</td><td rowspan="2">Setting</td><td rowspan="2"> $W _ { 1 \downarrow }$ </td><td rowspan="2">RMSE↓</td><td colspan="2">Sign agreement↑</td></tr><tr><td>Process</td><td>Fused</td></tr><tr><td rowspan="2">Visit levels</td><td>(0, 0.25, 0.5)</td><td>0.044</td><td>0.136</td><td>97.1</td><td>97.5</td></tr><tr><td>(0, 0.2, 1.0)</td><td>0.012</td><td>0.178</td><td>98.5</td><td>98.4</td></tr><tr><td rowspan="2">Shared cap</td><td>0.75</td><td>0.066</td><td>0.179</td><td>96.1</td><td>96.6</td></tr><tr><td>1.25</td><td>≤ 0.077</td><td>≤ 0.165</td><td>≥ 96.9</td><td>≥ 96.9</td></tr><tr><td rowspan="2">Snippet η</td><td>0.025</td><td>0.007</td><td>0.066</td><td>99.2</td><td>99.0</td></tr><tr><td>0.30</td><td>0.026</td><td>0.165</td><td>97.5</td><td>97.7</td></tr><tr><td>Joint change</td><td>Partial = 0.1, cap = 0.5</td><td>0.020</td><td>0.192</td><td>97.9</td><td>97.4</td></tr></table>

Across the displayed settings, the normalized process advantages remain close to the reference, with $W _ { 1 } \leq 0 . 0 7 7$ and $\mathrm { R M S E } \le 0 . 1 9 2$ , while at least 96.6% of fused advantage signs are preserved. Figure 9 shows the corresponding distributions for three exactly replayable settings, including a doubled high-support score and a tripled snippet weight. The limited changes in these signals support the adopted Visit scores, cap, and snippet weight as reasonable defaults for credit assignment.

Normalized process advantage  
![](images/71eff778257cc41a82284da16a5412e9b631d62a7e1805952feae40d071a8e8e.jpg)

![](images/5a10edec22df250d2f43cf25717b6b209d5137d1f6f514fa320b6f3ca64eab33.jpg)

![](images/7f38332f12cd4265a10041596e3fd08f3e85122ea3842fbf976321e8b52c7cd2.jpg)  
Figure 9: Normalized process-advantage distributions for three selected settings: (a) Visit levels (0, 0.2, 1.0); (b) partial-support score 0.1 and shared cap 0.5; (c) snippet weight 0.30, with other parameters unchanged. Curves pool all 32,966 turns after normalization within the 519 rollout groups over the full observed range; all three comparisons require no unrecorded snippet judgments.

## B.4 TEMPORAL DISCOUNTING ABLATION

The discounted variants in Table 2 use Visit and navigation credit, with or without independent snippet scoring. Credits are first normalized over every non-final policy turn in the original rollout group, including zero-credit turns, using Equation 5 and the population standard deviation. The resulting process advantages are then accumulated backward within each rollout:

$$
\begin{array} { r l r } {  { D _ { i , t } ^ { ( \gamma ) } = \sum _ { u = t } ^ { T _ { i } - 1 } \gamma ^ { u - t } A _ { i , u } ^ { r g } = A _ { i , t } ^ { r g } + \gamma D _ { i , t + 1 } ^ { ( \gamma ) } , } } \\ & { } & { D _ { i , T _ { i } } ^ { ( \gamma ) } = 0 , \qquad \gamma = 0 . 9 5 , \qquad 0 \leq t < T _ { i } . } \end{array}\tag{10}
$$

Here, u−t counts research turns, and the final report is excluded. Substituting $D _ { i , t } ^ { ( \gamma ) }$ for $A _ { i , t } ^ { r g }$ in Equation 6 retains unit process weight and outcome-only supervision on the report, without renormalization or additional advantage clipping. With PPO clipping and the policy-token mask unchanged, each matched ablation varies only temporal propagation; $\gamma = 0$ recovers the immediate case.

## C TRAINING BEHAVIOR AND COMPUTATIONAL COST

## C.1 TRAINING DIAGNOSTICS

Figure 10 complements the validation curves in Figure 5 with training rubric scores, research-turn counts, and validation report lengths. Over the final five validation checkpoints, their reports average approximately 4,576 and 4,900 tokens, respectively, following increases in report length for both methods during training.

## C.2 TRAINING TIME AND SCORING WORKLOAD

Table 6 sums the logged timers over the same 200-step window as Figure 5(b), using the same accelerator configuration for both runs. Dr.Credit takes 63.96 hours including scoring, compared with 75.14 hours for GRPO, a reduction of 14.9%. This comparison fixes training steps without measuring time to a matched quality threshold; the listed stages do not exhaust the total.

Process supervision adds the workload in Table 7, where full scoring latency includes outcome and process evaluation plus associated waiting. Completed trajectories enter scoring while others

![](images/6ad4f0c78c8a765f4da64822cf521e2fc29a13af6fb8fa230efdc6fee49128f7.jpg)

![](images/b53adc21f1a0f977aba282c19a7ec390ba601666f7faf160b31808f5bf5f1f71.jpg)

(c) Validation Report Length  
![](images/84cf0e8f69194527af7974ce259007ec481e31aee16c6696108eb1bae6966509.jpg)  
Figure 10: Training over 200 steps. (a) Rubric scores excluding auxiliary rewards, with GRPO without SFT as an initialization reference. (b) Mean research turns estimated from training rollouts. (c) Mean report length in tokens on DeepRubric-Val. In (b) and (c), faint lines show recorded values and solid lines show trailing means over 10 steps or five checkpoints, using available data.

Table 6: Recorded training time over steps 1–200 in hours. Rollout collection includes scoring; the total covers all work within the training-step timer, including stages beyond those listed. Row-wise reductions use GRPO as the reference, under the same accelerator configuration.
<table><tr><td>Stage</td><td>GRPO</td><td>Dr.Credit</td><td>Reduction</td></tr><tr><td>Rollout collection</td><td>49.41</td><td>42.89</td><td>13.2%</td></tr><tr><td>Policy update</td><td>17.12</td><td>13.91</td><td>18.8%</td></tr><tr><td>Total</td><td>75.14</td><td>63.96</td><td>14.9%</td></tr></table>

continue generating, and independent rubric histories are assessed concurrently through external services. Through this overlap, scoring shares the rollout collection interval, so its reported latency is already included in the total recorded training time and must not be added again.

Table 7: Scoring workload over steps 1–200. Latency is averaged over trajectories within each step, then over steps. Task counts exclude deterministic zeros and do not count retries separately.
<table><tr><td>Measurement</td><td>GRPO</td><td>Dr.Credit</td></tr><tr><td>Full scoring latency (s/trajectory)</td><td>2.40</td><td>23.03</td></tr><tr><td>Process Judge tasks per trajectory</td><td>0</td><td>29.09</td></tr></table>

Shorter trajectories also reduce evaluation work: from steps 1–40 to 161–200, Dr.Credit’s recorded interaction-turn count falls from 13.2 to 7.8 per trajectory, and process-evaluation tasks decrease from 10,043.93 to 6,231.20 per step (38.0%). The lower total training time is therefore consistent with reduced interaction work alongside asynchronous scoring and external service capacity.

## D PROMPT TEMPLATES

The following templates retain the wording used by the agent and judges, with blue brace-delimited placeholders filled at runtime and JSON formatted for readability. Typed placeholders are substituted before serialization, and lists may contain multiple entries. Templates cover data generation, tool interaction, and supervision under the experimental settings in Appendix A.

## D.1 RESEARCH AND DATA GENERATION

Figures 11 and 12 give the shared research-agent instructions and the separate SFT generation prompt described in Appendix A.1. Both receive the question without rubrics; the extraction template in Figure 13 takes retrieved webpage content and the goal supplied by the research agent.

![](images/6f093acdb9b7a74bf8e0e69c9509e0f9d4697d62f0bc09cf9882a412d6b5b281.jpg)  
Figure 11: Research instructions shared by SFT, RL, and evaluation: the agent uses Search and Visit, cites successful visits, and places its structured report inside one <answer> block.

![](images/a199235596b5cbe6a434d4f01d6331e033ed08b3a58324f78bf8717c5a0c6c4e.jpg)

Prompt continued   
- The complete Markdown report must be inside exactly one non-empty \`<answer>...</answer>\` block,   
with a clear heading hierarchy using at least two headings across at least two levels.   
- Before finishing, verify that every \`<cite>\` wraps substantive claim words between its opening and   
closing tags, all important factual claims are supported, all citations are valid, and the answer directly   
addresses the question.   
User Prompt:   
{Question}  
Figure 12: Prompt used by GLM-5.2 to generate SFT demonstration trajectories from research questions. It specifies reasoning, tool use, evidence-grounded citations, and report structure. At export, these system instructions are replaced by the research agent prompt in Figure 11.

![](images/d93f41c0c76fdafd9dcb10fabcd78f6238e7d9034962242c2f81b3f170bf4f13.jpg)  
Figure 13: Goal-conditioned webpage extraction for Visit, returning original evidence and a summary from the page content and research goal supplied in a single user message.

## D.2 OUTCOME ASSESSMENT

Figures 14 and 15 define rubric and citation assessment for the reward in Appendix A.5. The word criterion in the report template denotes one rubric; citation verification applies the same instructions to sources individually and jointly when a claim cites multiple pages.

Report Rubric Assessment Prompt.   
System Prompt:   
You will be given a question (in <question></question> tags), an answer (in <response></response>   
tags), and a single criterion (in <criterion></criterion> tags).   
Your job is to judge only how well the answer satisfies that specific criterion.

![](images/89bb142e03e1a875ab5fef66278adc16751e4485d0b1f2097d193949943ba3bd.jpg)  
Figure 14: Final-report assessment against one factual or logical rubric, returning a score from 0 to 4 that is normalized to [0, 1] before aggregation into the weighted rubric reward.

![](images/194ea7eb3adfbc8d8af27a1604644b9e142cd0fd62f7fe36684b5dd8b3d1e207.jpg)

```jsonl
Prompt continued
- Otherwise, 0 (none): No material proposition is directly supported.
4. When multiple sources are supplied, judge their joint coverage. A proposition may be supported by
any one of them.
Consistency rules:
- Do not infer missing facts from topic similarity, plausibility, or outside knowledge.
- Extra unrelated text in a source does not reduce the score.
- Judge factual entailment, not writing quality or source authority.
- If the evidence does not clearly meet the higher score’s definition, use the lower score.
Example input:
{"claim":"A launched in 2024 and raised $10 million.","sources":["A launched in 2024.","A raised $10
million."]}
Example output:
{"score":2}
Example input:
{"claim":"A launched in 2024 and raised $10 million.","sources":["A launched in 2024."]}
Example output:
{"score":1}
Example input:
{"claim":"A launched in 2024.","sources":["The article discusses A’s products."]}
Example output:
{"score":0}
User Prompt:
{
"claim": "{exact_cited_claim}",
"sources": ["{reference_text}"]
```  
Figure 15: Citation verification using only the supplied texts, with the same prompt assessing individual sources and their joint support for the exact claim when multiple pages are cited.

## D.3 PROCESS CREDIT ASSESSMENT

The Visit instructions and input in Figures 16 and 17 implement the history-aware assessment in Section 3.3. Figure 18 supplies independent snippet judgments without Visit histories; navigation credit follows URL matching in Section 3.3 and requires no additional prompt.

History-Aware Visit Credit Prompt: System Instructions.   
System Prompt:   
You are a strict rubric-aware retrieval scorer. Evaluate exactly ONE complete web\_visit turn against   
exactly ONE rubric. Score only the turn’s MARGINAL contribution beyond prior confirmed support for   
this rubric in the same rollout.   
The visit turn includes pre-visit reasoning, the web\_visit call, and returned pages. Every page includes   
raw Evidence and a generated Summary.   
Evidence policy:   
- Evidence is the primary factual record. Summary is required context for noisy or HTML-heavy   
Evidence, but cannot replace missing Evidence.   
- If Summary conflicts with Evidence, follow Evidence.   
- CAPTCHA, access errors, empty pages, navigation text, citation metadata, a title alone, or a statement   
of the intended visit goal provide no support.   
- Evaluate whether the evidence is applicable to the rubric’s entities, concepts, relationships, population,   
setting, and scope. Apparent similarity alone is not sufficient.   
- Score only evidence that advances THIS rubric, not evidence that merely helps the overall question or   
a different rubric.

![](images/a45851e2fd498609ae97e75fb7e83a2481aa98d9e718d3e8355c3031d723fbb5.jpg)  
Figure 16: History-aware assessment of a Visit against one rubric, returning a marginal-support level and new support points grounded in current pages. Numerical credit is computed separately from these judgments; Figure 17 gives the question, rubric, history, and current pages.

History-Aware Visit Credit Prompt: Input Template.   
User Prompt:   
{   
"question": "{question}",   
"rubric": {   
"id": "{rubric\_id}", "type": "{factual\_or\_logical}",   
"description": "{description}", "weight": "{weight:number}",   
"trusted\_evidence": ["{trusted\_evidence}"]   
},   
"prior\_confirmed\_support\_points\_for\_this\_rubric": ["{prior\_support\_point}"],   
"current\_visit\_turn": {   
"visit\_index": "{visit\_index:integer}",   
"research\_turn\_index": "{research\_turn\_index:integer}",   
"reasoning": "{pre\_visit\_reasoning}",   
"tool\_calls": ["{parsed\_tool\_call:object}"],   
"pages": [{   
"page\_id": "{W\_id}", "url": "{url}",   
"evidence": "{evidence}",   
"summary": "{summary}"}]   
},   
"output\_schema": {

```jsonl
Prompt continued
"level": "0 | 1 | 2", "page_ids": ["W1"],
"support_points": ["1-3 concise propositions from current pages"],
"rationale": "brief reason for the marginal level"
}
```  
Figure 17: Visit-judge inputs separate rubric reference evidence from support accumulated within the rollout. Current pages supply the evidence for assessing additional support for each rubric.

![](images/ff98404a5971fcc4a8ac855857449992ff56eddd5f2d01d810f27bf9cf9dc169.jpg)  
Figure 18: Independent assessment of unvisited Search snippets, returning sparse rubric matches and literal supporting quotes for snippet credit without reading or updating Visit histories.