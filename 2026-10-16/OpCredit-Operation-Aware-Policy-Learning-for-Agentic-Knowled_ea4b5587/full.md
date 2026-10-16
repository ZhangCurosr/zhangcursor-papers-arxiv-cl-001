# OpCredit: Operation-Aware Policy Learning for Agentic Knowledge Graph Question Answering

Yucheng Li<sup>1</sup> Mingyuan Li<sup>1</sup> Xiaxia Wang<sup>2</sup> Gong Cheng <sup>1</sup>State Key Laboratory of Novel Software Technology, Nanjing University, Nanjing, China <sup>2</sup>Department of Computer Science, University of Oxford <sup>∗</sup>Corresponding author: gcheng@nju.edu.cn

## Abstract

Agentic knowledge graph question answering (KGQA) relies on sequences of executable operations, but trajectory-level rewards obscure their individual contributions. Successful trajectories may contain ineffective operations, while failed trajectories may contain useful ones. To assess individual operations, we propose OpCredit, an operationaware policy-learning framework that combines downstream outcome credit with immediate transition credit: Branch Outcome Credit (BOC) reuses ordinary rollouts to compare terminal task rewards observed after different operations under matched navigation contexts; Frontier Transition Credit (FTC) measures operation-induced changes in answer-conditioned frontier potential. The two complementary signals jointly supervise the corresponding decision spans during training, while inference uses only the learned policy and standard KG interface. Experiments on ComplexWebQuestions, WebQSP, and GrailQA across three backbones demonstrate consistent improvements in KGQA performance.

## 1 Introduction

Knowledge graph question answering (KGQA) grounds answers in structured facts and offers an explicit reasoning substrate for knowledgeintensive questions. Recent agentic approaches answer complex questions through multi-turn interaction between a large language model (LLM) and a knowledge graph (KG): the policy issues executable operations, observes the returned evidence, and decides how to continue navigating or when to answer (Sun et al., 2024; Chen et al., 2024; Lin et al., 2025). Each operation determines which evidence becomes available to subsequent decisions. The choice of local operations therefore shapes the evidence on which successful reasoning depends.

![](images/793f8b376ab69fb0c9c4e4929d6f124c100a655294481ab72e2ef66407f54570.jpg)  
(b) Operation composition by trajectory outcome  
Figure 1: Trajectory–operation granularity mismatch in KGQA. Progressive operations advance toward the gold answer. (a) Examples in successful and failed rollouts. (b) Operation proportions on CWQ with SFT-initialized Qwen2.5-7B.

Final-answer rewards provide a natural way to improve such agents without annotating every intermediate decision, but they evaluate an entire trajectory at once: a successful rollout may contain offtarget operations, while a failed rollout may contain useful navigation. Group Relative Policy Optimization (GRPO) compares same-question rollouts and applies each trajectory’s advantage to all policygenerated tokens (Shao et al., 2024), so it does not explicitly distinguish operations that advance navigation from those that stall or divert it. Figure 1(a) illustrates this mismatch: the successful rollout includes an off-target soundtrack lookup before recovering to the correct actor, whereas the failed rollout reaches a relevant actor but ends with an incorrect answer; the diagnostic in Figure 1(b) likewise finds many progressive operations in failed rollouts and non-progressive operations in successful rollouts. These cases expose a trajectory–operation granularity mismatch: the outcome is trajectorylevel, while the policy must select operations individually.

Recent agentic reinforcement learning (RL) methods redistribute trajectory feedback across local decisions by grouping repeated states or aggregating sampled rollout structures (Feng et al., 2025; Wang et al., 2026a; Cheng et al., 2026). Prior KGQA work provides intermediate guidance through reference-path supervision, learned process evaluators, or graph-based progress signals (Yan et al., 2026; Wang et al., 2026b; Song et al., 2026b). Specifically for agentic KGQA, a natural approach is to assess local operations through their observed downstream outcomes.

This motivates our Branch Outcome Credit (BOC). BOC organizes same-question rollouts post hoc by navigation context and compares terminal task rewards observed after different operations. These context-relative comparisons assign outcome credit to individual operations under the sampled policy, without constructing an additional rollout tree or training an auxiliary evaluator.

BOC captures the downstream consequence of an operation, but later decisions can obscure the operation’s immediate effect on the KG evidence frontier. To address it, we introduce Frontier Transition Credit (FTC). FTC evaluates the immediate effect of each operation by comparing the KG evidence frontier’s proximity to the answer before and after execution.

Bringing these two views together, we propose OpCredit, an operation-aware policy-learning framework for agentic KGQA. BOC and FTC jointly supervise the corresponding decision spans to update a single KGQA policy. At inference time, the policy uses the standard KG interface without training-time signals or an auxiliary evaluator.

Experiments on ComplexWebQuestions (CWQ), WebQSP, and GrailQA with three LLM backbones show consistent gains in KGQA performance. Controlled comparisons show that BOC and FTC each improve on GRPO, with their combination performing best. A reference-blinded operation-level audit finds closer alignment between assigned credit and assessed usefulness for OpCredit than for GRPO.

Our contributions are twofold:

• We propose OpCredit, an operation-aware policy-learning framework for agentic KGQA that addresses the trajectory–operation granularity mismatch through complementary outcome and transition credit.

• We instantiate this framework with BOC and FTC. BOC compares terminal task rewards observed after different operations under matched navigation contexts, while FTC measures signed changes in answer-conditioned frontier potential.

Our code is available at https://github. com/kaka1215/OpCredit.

## 2 Related Work

## 2.1 LLM-Based KGQA

For training-free KG exploration, ToG uses LLMguided beam search (Sun et al., 2024), while PoG adds adaptive planning and self-correction (Chen et al., 2024). Learning-based methods learn path planning, evidence use, or graph interaction (Luo et al., 2024; Lin et al., 2025; Yan et al., 2026). We focus on operation-aware policy learning for agents that select executable operations during multi-turn KG interaction.

## 2.2 Supervision for KGQA

Task outcomes and interaction feedback provide supervision from answers and graph access. KnowCoder-A1 trains agentic reasoning with a curriculum of trajectory-level outcome rewards (Chen et al., 2026b). KG-R1 combines global answer and retrieval rewards with turn-wise validity feedback (Lin et al., 2025). However, terminal task rewards require further attribution to distinguish individual operations, while interaction validity alone does not establish an operation’s contribution to reasoning.

Reference trajectories and learned evaluators provide more explicit guidance for reasoning. GraphWalker learns KG exploration and error recovery through two-stage supervised fine-tuning on synthetic and expert trajectories before outcomebased RL (Xu et al., 2026). EoG rewards agreement with verified reference paths (Yan et al., 2026). GAPD aligns rollout and gold states for gold-action-conditioned token-level selfdistillation (Sun et al., 2026). DPRM learns dual implicit process reward models (PRMs) from outcome supervision to score KG and chainof-thought (CoT) steps for inference-time selection (Wang et al., 2026b). Such supervision remains constrained by the coverage ofdemonstrations, reference paths, gold actions, or by the generalization capabilities oflearned evaluators.

Graph-based progress uses the task KG to assess intermediate decisions. DTPO uses answerdistance changes and downstream outcomes to supervise KG operations sampled through tree expansion (Song et al., 2026b). Its local comparisons require explicit branch expansion and KG execution, making rollout cost sensitive to branching width.

## 2.3 Credit Assignment in Agentic RL

Agentic RL methods also derive local credit from shared structure in sampled rollouts. Table 1 compares outcome attribution and process signals across representative methods. GiGPO augments episode-level advantages through action grouping at repeated environment states (Feng et al., 2025). RTMC builds rollout trees from compact state–action signatures to estimate step-level advantages (Wang et al., 2026a). GraphGPO combines a trajectory-level outcome advantage with progress measured by distances to successful states in a sampled experience graph (Cheng et al., 2026). These approaches derive local credit from relationships among sampled states and outcomes. Our OpCredit uses such outcome evidence for BOC andfurther evaluates immediate transitions in the task KG through FTC. The distinction lies in aligning rollout-based outcome credit and KG-grounded transition credit with the same operation’s decision span.

## 3 Method

OpCredit supervises individual KG operations through two complementary signals: Branch Outcome Credit (BOC) compares terminal task rewards under matched navigation contexts, while Frontier Transition Credit (FTC) measures immediate changes in answer-conditioned frontier potential. Both signals supervise the same decision span to update a single policy (Figure 2).

## 3.1 Preliminaries

Let $\mathcal { K } = ( \nu , \mathcal { E } )$ be a knowledge graph, q a question, and $\mathcal { A } ^ { \ast } ( q ) \subseteq \mathcal { V }$ the set of gold answer entities mapped to the KG for training. An agent interacts with K through operations such as entityand-direction selection, relation selection, and answer emission. At step t, the policy observes a navigation state $s _ { t }$ and selects an operation $o _ { t } \in$ $\mathcal { O } ( s _ { t } )$ , where $\mathcal { O } ( s _ { t } )$ denotes the available operations. Navigation operations return graph observations that update the state to $s _ { t + 1 }$ , while answer emission terminates the interaction. A trajectory $\tau = ( s _ { 0 } , o _ { 0 } , \ldots , s _ { T - 1 } , o _ { T - 1 } , s _ { T } )$ consists of $T$ operations, including the terminal answer decision.

<table><tr><td>Method</td><td>Outcome attribution</td><td>Process signal</td></tr><tr><td colspan="3">Supervision for KGQA</td></tr><tr><td>KnowCoder-A1</td><td>Trajectory</td><td></td></tr><tr><td>KG-R1</td><td>Trajectory</td><td>Turn validity</td></tr><tr><td>GraphWalker</td><td>Trajectory</td><td>Trajectory SFT</td></tr><tr><td>EoG</td><td>Trajectory</td><td>Reference-path matching</td></tr><tr><td>GAPD</td><td>Trajectory</td><td>Gold-action distillation</td></tr><tr><td>DPRM</td><td>Local (implicit)</td><td>KG/CoT step scores</td></tr><tr><td>DTPO</td><td>Local</td><td>Answer distance</td></tr><tr><td colspan="3">Credit Assignment in Agentic RL</td></tr><tr><td>GiGPO</td><td>Local</td><td>Validity penalties</td></tr><tr><td>RTMC</td><td>Local</td><td>Validity / validation rewards</td></tr><tr><td>GraphGPO</td><td>Trajectory</td><td>Experience-graph distance</td></tr><tr><td>OpCredit</td><td>Local</td><td>Frontier transition (context-relative) (answer-conditioned)</td></tr><tr><td colspan="3">Trajectory/Local: attribution ries/intermediate decisions; -: no corresponding signal. Validity and validation rewards are task-dependent.</td></tr></table>

Table 1: Comparison of outcome attribution and process signals across representative methods.

We initialize the policy through supervised finetuning (SFT) to generate reasoning followed by executable operations, conditioned on the question and interaction history. Appendix C details the interaction protocol, template-based SFT trajectory construction, and serialized examples. At each training iteration, the rollout policy $\pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } }$ samples N trajectories $\{ \tau _ { i } \} _ { i = 1 } ^ { N }$ for the same question through the standard KG interface. Each trajectory $\tau _ { i }$ receives a terminal task reward $R _ { \mathrm { t a s k } } ( \tau _ { i } )$ based on its final-answer outcome; Appendix A.4 specifies the reward and training hyperparameters. GRPO computes a trajectory-level advantage within this group:

$$
A _ { i } ^ { \mathrm { t r a j } } = \frac { R _ { \mathrm { t a s k } } ( \tau _ { i } ) - \mu _ { q } } { \sigma _ { q } + \varepsilon } ,\tag{1}
$$

where $\mu _ { q }$ and $\sigma _ { q }$ are the mean and standard deviation of terminal task rewards within the group, and $\varepsilon > 0$ is a small constant for numerical stability. GRPO assigns this advantage to all policy-

![](images/56fca5e8506b19705f6f0e1e67bc4275301985f61250f6f276d3d21cd8a648e0.jpg)  
Figure 2: Overview of OpCredit. BOC compares terminal task rewards under matched navigation contexts; FTC measures answer-conditioned changes in frontier potential. Both signals supervise the same decision span during training. Inference uses the learned policy and standard KG interface.

generated tokens in $\tau _ { i }$

## 3.2 Branch Outcome Credit (BOC)

BOC assigns context-relative outcome credit to individual operations by comparing terminal task rewards under matched navigation contexts.

Navigation context matching. For samequestion rollouts with $R _ { i } = R _ { \mathrm { t a s k } } ( \tau _ { i } )$ , BOC first groups decisions by the active entity before each operation. Within each group, the pre-decision navigation state s is mapped to

$$
c = \chi ( s ) = \bigl ( E _ { \mathrm { v i s } } ( s ) , b ( s ) , \phi ( s ) \bigr ) ,\tag{2}
$$

where $E _ { \mathrm { v i s } } ( s )$ is the visited entity set, $b ( s )$ the remaining navigation budget, and $\phi ( s )$ the decision phase: relation selection, entity-and-direction selection, or answer emission. Here $b ( s ) = H - h ( s )$ with maximum hop budget H and $h ( s )$ completed hops; decisions within a hop share the remaining budget. Matching these context elements helps to pool outcome evidence and keep rollout comparisons within the same decision phase, without additional environment interaction. Let $\Omega _ { q } ( c , o )$ contain the unique rollout indices observing operation o under context c, and $\begin{array} { r } { \Omega _ { q } ( c ) = \bigcup _ { o } \Omega _ { q } ( c , o ) } \end{array}$ Both support sets are restricted to the same active entity, omitted from the notation. Each rollout contributes once per set, avoiding duplicate outcome evidence.

Context-relative outcome credit. From sampled continuations, we estimate the operation’s observed return and context baseline by

$$
\begin{array} { r l } & { \widehat { Q } _ { \mathrm { o u t } } ( c , o ) = \displaystyle \frac { 1 } { | \Omega _ { q } ( c , o ) | } \sum _ { i \in \Omega _ { q } ( c , o ) } R _ { i } , } \\ & { \qquad \widehat { V } _ { \mathrm { o u t } } ( c ) = \displaystyle \frac { 1 } { | \Omega _ { q } ( c ) | } \sum _ { i \in \Omega _ { q } ( c ) } R _ { i } . } \end{array}\tag{3}
$$

Here $\widehat { Q } _ { \mathrm { o u t } }$ averages returns after operation $^ { O , }$ while $\widehat { V } _ { \mathrm { o u t } }$ averages returns over rollouts encountering a matched context; both include successful and unsuccessful completions. Unlike the question-level mean $\mu _ { q }$ in Eq. (1), $\widehat { V } _ { \mathrm { o u t } } ( c )$ is context-specific. For decision t in rollout $\tau _ { i }$ , let $\Delta _ { i , t } = \widehat { Q } _ { \mathrm { o u t } } ( c _ { i , t } , o _ { i , t } ) -$ $\widehat { V } _ { \mathrm { o u t } } ( c _ { i , t } )$ denote the observed outcome contrast under context $c _ { i , t }$ . BOC uses the normalized contrast when nonzero and otherwise retains the trajectorylevel advantage in Eq. (1):

$$
A _ { \mathrm { B O C } } ^ { i , t } = \left\{ \begin{array} { l l } { \frac { \Delta _ { i , t } } { \sigma _ { q } + \varepsilon } , } & { \Delta _ { i , t } \neq 0 , } \\ { A _ { i } ^ { \mathrm { t r a j } } , } & { \Delta _ { i , t } = 0 } \end{array} \right. .\tag{4}
$$

The contrast’s sign indicates whether the operation’s observed return is above or below its context baseline; both cases use the same question-level reward scale. In Figure 2, $R ^ { \prime } = R ^ { \prime \prime } = 0$ and $R = 1$ yield positive BOC for $O t$ and negative BOC for $o ^ { \prime } , o ^ { \prime \prime } ;$ Appendix D.1 gives the calculation.

## 3.3 Frontier Transition Credit (FTC)

FTC evaluates the effect of each operation on the KG evidence frontier through an answerconditioned before/after comparison, without requiring a reference navigation sequence.

Answer-conditioned frontier potential. For operation $o _ { i , t }$ at decision t in trajectory $\tau _ { i } .$ , let $F _ { i , t } ^ { - }$ and $F _ { i , t } ^ { + }$ denote the operation-local entity sets before and after its transition, respectively. Figure 2 omits the rollout index i. A useful expansion may expose an answer-relevant candidate alongside distractors, so we assess the potential of a nonempty frontier F by its best available opportunity:

$$
\Phi _ { q } ( F ) = \operatorname* { m a x } _ { e \in F } \bigl ( 2 - \delta _ { q } ( e ) \bigr ) .\tag{5}
$$

Here $\delta _ { q } ( e ) = 0$ for a gold answer entity, 1 for an entity outside the gold answer set but in its constructed one-hop neighborhood, and 2 for all remaining entities. These labels describe coarse answer proximity, not exact graph distance or remaining operation counts. Thus $\Phi _ { q }$ measures frontierto-answer proximity. The gold answer set is used only during training and is not added to policy observations; frontier definitions by operation, neighborhood construction, and unavailable-frontier handling appear in Appendix D.2.

Signed transition credit. The potential of a resulting frontier alone does not distinguish progress made by the current operation from proximity already achieved by earlier decisions. We therefore assign the operation the signed change between its before/after frontier potentials:

$$
A _ { \mathrm { F T C } } ^ { i , t } = \Phi _ { q } ( F _ { i , t } ^ { + } ) - \Phi _ { q } ( F _ { i , t } ^ { - } ) .\tag{6}
$$

Positive, zero, and negative values indicate measured progress, no measured change, and regression, respectively. The comparison uses operationlocal frontiers, not the cumulative interaction history, so earlier answer encounters do not automatically make later operations favorable. The potential-difference form is related to potentialbased reward shaping (Ng et al., 1999); here it supplies an answer-conditioned auxiliary objective on decision spans rather than changing the terminal task reward used for BOC. An operation can therefore receive positive FTC even if later decisions fail, while BOC evaluates its sampled continuations through terminal task rewards.

## 3.4 Unified Operation-Aware Policy Learning

BOC and FTC are constructed from rollouts and aligned to the same decision span for a joint policy update.

Decision-span alignment. Let y<sub>i</sub> = $( y _ { i , 1 } , \dots , y _ { i , L _ { i } } )$ be the policy-generated token sequence for rollout i, where $L _ { i }$ is its length and $j$ indexes token positions. Each decision comprises the generated reasoning together with its executable operation; both belong to the same decision span and receive its operation-level supervision. For operation $o _ { i , t } ,$ let $\mathcal { T } _ { i , t }$ contain the token indices of this span. The decision spans, including the terminal answer decision, partition the generated sequence: $\mathcal { T } _ { i } = \{ 1 , \dots , L _ { i } \} = \bigcup _ { t } \mathcal { T } _ { i , t }$ For $j \in \mathcal { I } _ { i , t }$ , the outcome credit is $A _ { \mathrm { B O C } } ^ { i , t }$ from Eq. (4). For $j \in \mathcal { I } _ { i , t }$ , we set $B _ { i , j } = A _ { \mathrm { F T C } } ^ { i , t }$ when the operation has a valid before/after frontier comparison; tokens without an applicable FTC signal have $B _ { i , j } = 0$ . The FTC eligibility rules are given in Appendix D.2.

Joint policy optimization. Let $h _ { i , j }$ denote the full policy context preceding token $y _ { i , j }$ , including the question, interaction history, and previously generated tokens. The policy ratio is

$$
\rho _ { i , j } ( \theta ) = \frac { \pi _ { \theta } ( y _ { i , j } \mid h _ { i , j } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { i , j } \mid h _ { i , j } ) } .\tag{7}
$$

The BOC objective uses a clipped GRPO surrogate (Shao et al., 2024), with dual clipping for negative advantages. Writing the dual-clipped surrogate defined in Appendix D.3 as D, we obtain

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { B O C } } ( \theta ) = - \mathbb { E } _ { 1 \leq i \leq N } \left[ D \Big ( \rho _ { i , j } , A _ { \mathrm { B O C } } ^ { i , t } \Big ) \right] . } \end{array}\tag{8}
$$

Here t denotes the decision whose span contains token j, i.e., $j \in \mathcal { I } _ { i , t }$ . FTC supplies signed auxiliary log-likelihood supervision on the same decision span:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F T C } } ( \theta ) = - \mathbb { E } _ { 1 \leq i \leq N } \left[ B _ { i , j } \log \pi _ { \theta } ( y _ { i , j } \mid h _ { i , j } ) \right] . } \end{array}\tag{9}
$$

Both objectives average over generated tokens within each trajectory and then over trajectories. Positive FTC reinforces the corresponding generated decision, whereas negative FTC suppresses it. FTC supplies transition supervision alongside BOC’s outcome credit, and we jointly optimize

$$
\begin{array} { r } { \mathcal { L } ( \theta ) = \mathcal { L } _ { \mathrm { B O C } } ( \theta ) + \lambda _ { \mathrm { a u x } } \mathcal { L } _ { \mathrm { F T C } } ( \theta ) + \mathcal { L } _ { \mathrm { r e g } } ( \theta ) , } \end{array}\tag{10}
$$

where $\lambda _ { \mathrm { a u x } }$ weights FTC and $\mathcal { L } _ { \mathrm { r e g } }$ contains KL and entropy regularization. Credits are computed from collected rollouts and held fixed during their policy updates; subsequent iterations refresh the samples and their supervision. At inference, only the learned policy and standard KG interface are required; rollout aggregation and answer-conditioned frontier measurements are training-time procedures.

## 4 Experiments

## 4.1 Experimental Setup

Datasets and metrics. We evaluate on the CWQ (Talmor and Berant, 2018) and WebQSP (Yih et al., 2016) test splits and the full official GrailQA (Gu et al., 2021) development split. For our models, we report F1 over normalized answer sets and Hit@1; experimental and dataset details are in Appendices A and B, respectively.

Baselines. Table 2 compares two groups: training-free KGQA (direct prompting, ToG, PoG, and ReKnoS) and training-based KGQA (GNN-RAG, KG-Agent, KG-R1, Graph-RFT, EoG, KnowCoder-A1, KBQA-o1, DPRM, SCPRM, DoG, and GAPD). Named KGQA baseline scores come from published reports; direct prompting uses our evaluation protocol. $O p C r e d i t _ { \mathrm { S F T } }$ denotes the SFT-initialized policy.

Implementation. We experiment with three backbones: Qwen2.5-3B, Qwen2.5-7B, and Llama-3.1-8B. All RL configurations use the corresponding SFT-initialized policy and same-question rollout groups. Appendix A gives the training, rollout, and evaluation settings.

## 4.2 Main Results

Table 2 shows generally stronger results for training-based KGQA. OpCredit improves Hit@1 and F1 over SFT in all nine dataset–backbone settings, including GrailQA with Qwen2.5-3B. This smaller model also exceeds all training-free baselines on Hit@1. Gains persist on WebQSP despite its higher initial scores.

OpCredit exceeds KG-R1 on CWQ/WebQSP and Graph-RFT on all three datasets. It exceeds EoG on WebQSP and on CWQ with Qwen2.5-7B, but trails on GrailQA and on CWQ with Llama-3.1-8B. GAPD also reports higher F1 on WebQSP/GrailQA with Llama-3.1-8B. Note that the comparisons with EoG and GAPD may not be entirely fair, as they both use additional resources for process supervision. EoG uses Gemini-2.5- Flash-generated demonstrations and model-verified reference paths, while GAPD uses gold-actionconditioned self-distillation (Yan et al., 2026; Sun et al., 2026); OpCredit requires only the benchmark training data and task KG, using gold answers to construct RL supervision.

## 4.3 Ablation Study

Table 3 compares four RL configurations on Qwen2.5-7B with the same SFT initialization and training and evaluation protocols. Removing both BOC and FTC gives the GRPO baseline. Adding BOC improves all six metrics over GRPO with the same terminal task rewards. Its gains also persist when added to FTC, indicating that contextrelative outcome comparisons contribute to policy learning alongside frontier-transition supervision. FTC alone yields larger gains than BOC alone in this setting, and adding it to BOC further improves both metrics on every dataset. Immediate frontier changes therefore provide useful supervision beyond observed terminal task rewards. The full method performs best across the three datasets, supporting the complementary contributions of the two signals to operation selection.

## 4.4 Analysis of Navigation Context Matching

BOC compares each operation’s observed return with its comparison group’s outcome baseline. Relation selection and entity-and-direction selection can share the same active entity, visited entity set, and remaining navigation budget. With rollouts and terminal task rewards fixed, we compare the full context against three alternatives: active entity only, visited entity set only, and visited entity set with remaining navigation budget. All comparisons use the same question and active entity. Without decision phase, 46–48% of decision occurrences across checkpoints belong to groups containing both operation types, pooling outcome evidence from both relation and entity-and-direction choices. Matching the decision phase provides a type-consistent baseline: relation choices are evaluated relative to relation choices, and entity-and-direction choices relative to their own type. This aligns BOC’s local outcome comparison with the kind of decision the policy is making. Appendix D.4 gives the matching rules and mixing statistics.

## 4.5 Reference-Blinded Credit Audit

We evaluate whether operation-level credit aligns with operation usefulness through a referenceblinded audit. We implement the credit-assignment rules of GiGPO, RTMC, and GraphGPO for KGQA and replay their signals alongside GRPO and Op-Credit on a common rollout pool generated by the SFT policy. With operations and terminal task rewards fixed, the audit compares credit assignments to the same decisions. Given the question, interaction history preceding the operation, the operation itself, and the graph observations it returns, an LLM judge assigns a usefulness rating without access to gold answers, terminal task rewards, or assigned credit. We report mean credit across four ordered usefulness levels and the area under the receiver operating characteristic curve (AUC), using assigned credit to distinguish operations judged useful from those judged unhelpful. Appendix E provides the complete protocol and results.

<table><tr><td>Method</td><td>Backbone</td><td colspan="2">CWQ</td><td colspan="2">WebQSP</td><td colspan="2">GrailQA</td></tr><tr><td></td><td></td><td>Hit@1</td><td>F1</td><td>Hit@1</td><td>F1</td><td>Hit@1</td><td>F1</td></tr><tr><td colspan="8">Training-Free KGQA</td></tr><tr><td>Direct prompting</td><td>Qwen2.5-3B</td><td>22.0</td><td>17.7</td><td>44.6</td><td>30.3</td><td>13.8</td><td>11.6</td></tr><tr><td>Direct prompting</td><td>Qwen2.5-7B</td><td>25.7</td><td>20.7</td><td>50.9</td><td>33.2</td><td>15.9</td><td>13.6</td></tr><tr><td>Direct prompting</td><td>Llama-3.1-8B</td><td>28.1</td><td>22.4</td><td>55.5</td><td>34.8</td><td>21.3</td><td>17.7</td></tr><tr><td>ToG (Sun et al., 2024)</td><td>GPT-4</td><td>68.5</td><td></td><td>82.6</td><td></td><td>81.4</td><td></td></tr><tr><td>PoG (Chen et al., 2024)</td><td>GPT-4</td><td>75.0</td><td></td><td>87.3</td><td></td><td>84.7</td><td></td></tr><tr><td>ReKnoS (Wang et al., 2025)</td><td>GPT-4o-mini</td><td>66.8</td><td>一</td><td>83.8</td><td></td><td>80.5</td><td></td></tr><tr><td colspan="8">Training-Based KGQA</td></tr><tr><td>GNN-RAG (Mavromatis and Karypis, 2025)</td><td>Llama-2-7B</td><td>66.8</td><td>59.4</td><td>85.7</td><td>71.3</td><td></td><td></td></tr><tr><td>KG-Agent (Jiang et al., 2025)</td><td>Llama-2-7B</td><td>72.2</td><td>69.8</td><td>83.3</td><td>81.0</td><td></td><td>86.1</td></tr><tr><td>KG-Ri (Lin et al., 2025)</td><td>Qwen2.5-3B</td><td>65.3</td><td></td><td>82.8</td><td></td><td></td><td></td></tr><tr><td>OpCreditsFT</td><td>Qwen2.5-3B</td><td>71.95</td><td>66.02</td><td>85.17</td><td>74.96</td><td>81.23</td><td>74.07</td></tr><tr><td>OpCredit</td><td>Qwen2.5-3B</td><td>76.56</td><td>69.31</td><td>88.86</td><td>77.72</td><td>86.35</td><td>79.17</td></tr><tr><td>Graph-RFT (Song et al., 2026a)</td><td>Qwen2.5-7B</td><td>78.4</td><td></td><td>87.3</td><td></td><td>82.7</td><td></td></tr><tr><td>EoG (Yan et al., 2026)</td><td>Qwen2.5-7B</td><td>82.7</td><td>73.8</td><td>90.7</td><td>78.1</td><td>91.7</td><td>88.5</td></tr><tr><td>OpCreditsFT</td><td>Qwen2.5-7B</td><td>72.15</td><td>66.06</td><td>86.28</td><td>76.38</td><td>83.73</td><td>76.77</td></tr><tr><td>OpCredit</td><td>Qwen2.5-7B</td><td>83.47</td><td>74.58</td><td>92.13</td><td>82.97</td><td>90.18</td><td>83.42</td></tr><tr><td>KnowCoder-A1 (Chen et al., 2026b)</td><td>Qwen2.5-Coder-7B</td><td></td><td>68.3</td><td></td><td>77.2</td><td></td><td>80.5</td></tr><tr><td>KBQA-o1 (Luo et al., 2025)</td><td>Llama-3.1-8B</td><td></td><td></td><td></td><td>59.8</td><td></td><td>78.5</td></tr><tr><td>DPRM (Wang et al., 2026b)</td><td>Llama-3.1-8B</td><td>73.0</td><td>59.9</td><td>87.2</td><td>72.3</td><td></td><td></td></tr><tr><td>SCPRM (Chen et al., 2026a)</td><td>Llama-3.1-8B</td><td>73.3</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DoG (Li et al., 2025)</td><td>Llama-3.1-8B</td><td>76.2</td><td></td><td>91.4</td><td></td><td></td><td></td></tr><tr><td>EoG (Yan et al., 2026)</td><td>Llama-3.1-8B</td><td>86.6</td><td>77.9</td><td>92.8</td><td>81.3</td><td>92.1</td><td>90.6</td></tr><tr><td>GAPD (Sun et al., 2026)</td><td>Llama-3.1-8B</td><td></td><td></td><td></td><td>86.1</td><td></td><td>89.5</td></tr><tr><td>OpCreditsFT</td><td>Llama-3.1-8B</td><td>73.32</td><td>66.62</td><td>87.84</td><td>76.54</td><td>86.40</td><td>81.63</td></tr><tr><td>OpCredit</td><td>Llama-3.1-8B</td><td>84.04</td><td>74.03</td><td>93.10</td><td>83.86</td><td>91.25</td><td>84.33</td></tr></table>

Table 2: Overall Hit@1 and F1 on CWQ, WebQSP, and GrailQA. Bold highlights the best Hit@1 and F1 among methods using each OpCredit backbone. – denotes an unreported metric.
<table><tr><td>Configuration</td><td>BOC</td><td>FTC</td><td colspan="2">CWQ</td><td colspan="2">WebQSP</td><td colspan="2">GrailQA</td></tr><tr><td></td><td></td><td></td><td>Hit@1</td><td>F1</td><td>Hit@1</td><td>F1</td><td>Hit@1</td><td>F1</td></tr><tr><td>OpCreditsFT</td><td></td><td></td><td>72.15</td><td>66.06</td><td>86.28</td><td>76.38</td><td>83.73</td><td>76.77</td></tr><tr><td>OpCredit-BOC,—FTC</td><td></td><td></td><td>79.43</td><td>71.57</td><td>89.44</td><td>78.98</td><td>87.17</td><td>79.62</td></tr><tr><td>OpCredit-FTC</td><td>√</td><td></td><td>80.65</td><td>72.15</td><td>90.27</td><td>79.09</td><td>88.03</td><td>81.49</td></tr><tr><td>OpCredit-BOC</td><td></td><td>√</td><td>82.41</td><td>73.70</td><td>91.30</td><td>81.48</td><td>89.58</td><td>82.36</td></tr><tr><td>OpCredit</td><td>V</td><td></td><td>83.47</td><td>74.58</td><td>92.13</td><td>82.97</td><td>90.18</td><td>83.42</td></tr></table>

Table 3: Ablation of OpCredit with Qwen2.5-7B.

Figure 3(a) shows that OpCredit assigns negative mean credit to the least useful operations and increasingly positive credit across levels 2–4. GRPO and RTMC also exhibit increasing mean credit, whereas GiGPO and GraphGPO reverse this ordering between the middle levels. To compare usefulness discrimination beyond these group averages, Figure 3(b) reports AUC based on operationlevel credit rankings. OpCredit achieves the highest AUC among the compared methods (0.718), ahead of GRPO (0.606) and RTMC (0.492). These results indicate OpCredit’s stronger discriminability between useful and unhelpful operations within the same rollout pool.

![](images/e8f83918b445b3b599122fa1d37498ecead2669407d4cc27dd9fc3b8ba4de7a1.jpg)  
Figure 3: Credit alignment on a common SFT rollout pool. (a) Mean replay credit across four ordered usefulness levels (1 is least useful and 4 is most useful). (b) Usefulness-discrimination AUC; the dashed line marks random discrimination.

## 4.6 Error Analysis

Figure 4 compares the composition of GRPO and OpCredit errors on matched diagnostic subsets across three datasets. We distinguish answerselection errors despite answer-relevant evidence, first-hop path deviations (Path-first), later deviations or failures without reference-path alignment (Path-late), and empty predicted answer sets (Empty answer). Answer-selection errors dominate CWQ, whereas first-hop deviations dominate GrailQA. WebQSP exhibits a more mixed error profile, with substantial shares of Path-late and Empty answer cases. The same terminal failure label can thus reflect difficulties in acquiring useful evidence or selecting answers from available evidence.

Compared with GRPO, OpCredit has a smaller share of answer-selection errors on all three datasets. Path-late accounts for a larger share of its remaining errors on CWQ and GrailQA, while Empty answer accounts for a smaller share on GrailQA. These profiles highlight navigation as a prominent source of the remaining errors. They complement the credit audit by locating failures in evidence acquisition and answer selection. A checkpoint-wise behavioral diagnostic on CWQ also finds higher late-training recovery rates for OpCredit than for BOC-only after backward operations (Appendix F). Appendix G provides the protocol, category definitions, and complete statistics.

![](images/bc55bbd4d0fa7302c68ae4f111166190a7973f1a3468ae3d9c5b1ffe8923ddce.jpg)  
Figure 4: Error composition on matched diagnostic subsets. Labels show rounded percentages within each method’s error set.

## 5 Conclusion

We presented OpCredit, an operation-aware policylearning framework for agentic KGQA. BOC compares terminal task rewards from ordinary rollouts under matched navigation contexts, while FTC measures immediate answer-conditioned frontier changes. Both signals supervise the same decision span during training; inference uses only the learned policy and standard KG interface. Results across three benchmarks and three backbones show improved KGQA performance; controlled ablations support complementary contributions from both signals, and the audit shows closer credit alignment with assessed operation usefulness on a shared rollout pool.

## Limitations

Our evaluation is limited to three Freebase-based KGQA benchmarks with a fixed executableoperation interface. Performance on other graph schemas and tool environments remains to be evaluated. We focus on operation-aware policy learning within this interface; comparisons of alternative agent interfaces are left for future work.

Our experiments do not cover multilingual or cross-lingual KGQA. Further evaluation could examine questions expressed in different languages, including settings where the question language differs from that of KG entity and relation labels.

We also evaluate against a fixed KG snapshot. The current experiments do not assess how a trained policy responds to subsequent additions or updates to the graph. Evaluation on evolving KGs would complement the present benchmark results.

## References

Jiujiu Chen, Yazheng Liu, Sihong Xie, and Hui Xiong. 2026a. SCPRM: A schema-aware cumulative process reward model for knowledge graph question answering. CoRR, abs/2605.02819.

Liyi Chen, Panrong Tong, Zhongming Jin, Ying Sun, Jieping Ye, and Hui Xiong. 2024. Plan-on-graph: Self-correcting adaptive planning of large language model on knowledge graphs. In Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.

Zhuo Chen, Fei Wang, Zixuan Li, Zhao Zhang, Weiwei Ding, Chuanguang Yang, Yongjun Xu, and Xiaolong Jin. 2026b. Incentivizing agentic reasoning capability with outcome supervision for knowledge base question answering. In Proceedings ofthe ACM Web Conference 2026, WWW 2026, Dubai, United Arab Emirates, originally scheduledfor April 13-17, 2026, rescheduledfor June 29 - July 3, 2026, pages 4302– 4313. ACM.

Xin Cheng, Shuo He, Lang Feng, Haiyang Xu, Ming Yan, Lei Feng, and Bo An. 2026. Beyond trajectory-level attribution: Graph-based credit assignment for agentic reinforcement learning. CoRR, abs/2605.26684.

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. 2025. Group-in-group policy optimization for LLM agent training. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025.

Yu Gu, Sue Kase, Michelle Vanni, Brian M. Sadler, Percy Liang, Xifeng Yan, and Yu Su. 2021. Beyond I.I.D.: three levels of generalization for question answering on knowledge bases. In WWW ’21: The Web Conference 2021, Virtual Event / Ljubljana, Slovenia, April 19-23, 2021, pages 3477–3488. ACM / IW3C2.

Jinhao Jiang, Kun Zhou, Xin Zhao, Yang Song, Chen Zhu, Hengshu Zhu, and Ji-Rong Wen. 2025. Kgagent: An efficient autonomous agent framework

for complex reasoning over knowledge graph. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 9505–9523. Association for Computational Linguistics.

Kun Li, Tianhua Zhang, Xixin Wu, Hongyin Luo, James R. Glass, and Helen M. Meng. 2025. Decoding on graphs: Faithful and sound reasoning on knowledge graphs through generation of well-formed chains. In Proceedings ofthe 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 24349–24364. Association for Computational Linguistics.

Junhong Lin, Shicheng Liu, Jinyeop Song, Song Wang, Julian Shun, and Yada Zhu. 2025. Efficient and transferable agentic knowledge graph RAG via reinforcement learning. CoRR, abs/2509.26383.

Haoran Luo, Haihong E, Yikai Guo, Qika Lin, Xiaobao Wu, Xinyu Mu, Wenhao Liu, Meina Song, Yifan Zhu, and Anh Tuan Luu. 2025. Kbqa-o1: Agentic knowledge base question answering with monte carlo tree search. In Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, volume 267 of Proceedings of Machine Learning Research. PMLR / OpenReview.net.

Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. 2024. Reasoning on graphs: Faithful and interpretable large language model reasoning. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Costas Mavromatis and George Karypis. 2025. GNN-RAG: graph neural retrieval for efficient large language model reasoning on knowledge graphs. In Findings ofthe Associationfor Computational Linguistics, ACL 2025, Vienna, Austria, July 27 - August 1, 2025, volume ACL 2025 of Findings ofACL, pages 16682–16699. Association for Computational Linguistics.

Andrew Y. Ng, Daishi Harada, and Stuart Russell. 1999. Policy invariance under reward transformations: Theory and application to reward shaping. In Proceedings of the Sixteenth International Conference on Machine Learning (ICML 1999), Bled, Slovenia, June 27 - 30, 1999, pages 278–287. Morgan Kaufmann.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. CoRR, abs/2402.03300.

Yanlin Song, Ben Liu, Víctor Gutiérrez-Basulto, Zhiwei Hu, Qianqian Xie, Min Peng, Sophia Ananiadou, and Jeff Z. Pan. 2026a. Plan then retrieve: Reinforcement learning-guided complex reasoning over knowledge

graphs. In Proceedings ofthe ACM Web Conference 2026, WWW 2026, Dubai, United Arab Emirates, originally scheduledfor April 13-17, 2026, rescheduled for June 29 - July 3, 2026, pages 3666–3676. ACM.

Yurun Song, Xiangqing Shen, Jianfei Yu, and Rui Xia. 2026b. Interactive semantic parsing with reinforcement learning for knowledge graph reasoning. In Findings ofthe Associationfor Computational Linguistics, ACL 2026, San Diego, California, United States, July 2-7, 2026, pages 20464–20484. Association for Computational Linguistics.

Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel M. Ni, Heung-Yeung Shum, and Jian Guo. 2024. Think-on-graph: Deep and responsible reasoning of large language model on knowledge graph. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. Open-Review.net.

Xin Sun, Jianan Xie, Zhongqi Chen, Qiang Liu, Shu Wu, Bowen Song, Weiqiang Wang, Zilei Wang, and Liang Wang. 2026. GAPD: gold-action policy distillation for agentic reinforcement learning in knowledge base question answering. CoRR, abs/2605.29584.

Alon Talmor and Jonathan Berant. 2018. The web as a knowledge-base for answering complex questions. In Proceedings ofthe 2018 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2018, New Orleans, Louisiana, USA, June 1-6, 2018, Volume 1 (Long Papers), pages 641– 651. Association for Computational Linguistics.

Song Wang, Junhong Lin, Xiaojie Guo, Julian Shun, Jundong Li, and Yada Zhu. 2025. Reasoning of large language models over knowledge graphs with superrelations. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

Tao Wang, Suhang Zheng, and Xiaoxiao Xu. 2026a. RTMC: step-level credit assignment via rollout trees. CoRR, abs/2604.11037.

Xinyi Wang, Yiping Song, Zhiliang Tian, Bo Liu, Tingjin Luo, and Minlie Huang. 2026b. DPRM: A dual implicit process reward model in multi-hop question answering. In Fortieth AAAI Conference on Artificial Intelligence, Thirty-Eighth Conference on Innovative Applications of Artificial Intelligence, Sixteenth Symposium on Educational Advances in Artificial Intelligence, AAAI 2026, Singapore, January 20-27, 2026, pages 33683–33691. AAAI Press.

Shuwen Xu, Yao Xu, Jiaxiang Liu, Chenhao Yuan, Wenshuo Peng, Jun Zhao, and Kang Liu. 2026. Graphwalker: Agentic knowledge graph question answering via synthetic trajectory curriculum. CoRR, abs/2603.28533.

Shiqi Yan, Yubo Chen, Ruiqi Zhou, Zhengxi Yao, Shuai Chen, Tianyi Zhang, Shijie Zhang, Wei Qiang Zhang, Yongfeng Huang, Haixin Duan, and Yunqi Zhang. 2026. Explore-on-graph: Incentivizing autonomous exploration of large language models on knowledge graphs with path-refined reward modeling. In The Fourteenth International Conference on Learning Representations, ICLR 2026, Rio de Janeiro, Brazil, April 23-27, 2026. proceedings.iclr.cc.

Wen-tau Yih, Matthew Richardson, Christopher Meek, Ming-Wei Chang, and Jina Suh. 2016. The value of semantic parse labeling for knowledge base question answering. In Proceedings ofthe 54th Annual Meeting ofthe Associationfor Computational Linguistics, ACL 2016, August 7-12, 2016, Berlin, Germany, Volume 2: Short Papers. The Association for Computer Linguistics.

## A Experimental Protocol

## A.1 Hardware and software

All experiments were run on a single node equipped with eight NVIDIA RTX 5880 Ada GPUs (48 GB each). Reinforcement learning (RL) used tensor parallelism of one and four-way data parallelism; evaluation used single-GPU multi-process workers. The total compute for SFT, RL training, ablations, and evaluation is estimated at approximately 2,500 GPU hours. The software environment used Python 3.10, PyTorch 2.11.0, Transformers 5.5.4, veRL 0.7.1, and vLLM 0.21.0. The knowledge graph (KG) was built from the final Freebase RDF dump (2015-08-09) and accessed through a local SPARQL endpoint.

## A.2 Task, data, and metrics

We evaluate CWQ and WebQSP on their test splits and GrailQA on its full official development split. Dataset statistics and preprocessing are described in Appendix B. Hit@1 records whether the predicted answer set contains a gold answer; F1 compares the predicted and gold answer sets after answer normalization. Formal evaluation uses one sampled trajectory per question. The audit and behavioral analyses use CWQ diagnostic subsets with multiple rollouts per question; their sampling settings are listed separately in Table 4.

Reported baseline scores. Table 2 retains the published scores for the named KGQA baselines. For KnowCoder-A1, we report its published F1 values with the Qwen2.5-Coder-7B backbone. Its CWQ EM and WebQSP Random Hits@1 are not entered in the Hit@1 columns; its GrailQA overall score of 80.5 is F1. For GAPD, we report its published entity-level F1 on WebQSP and the full GrailQA development split; its exact-set-match EM is not entered in the Hit@1 columns.

## A.3 Training configurations

We use the instruction-tuned checkpoints Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-7B-Instruct, and meta-llama/Llama-3.1-8B-Instruct, each initialized through SFT. The ablation study in Table 3 holds the SFT initialization, rollout protocol, and optimization settings fixed while enabling BOC, FTC, both, or neither; the configuration with neither signal is GRPO. Following the checkpoint-selection protocol of EoG (Yan et al.,

2026), we evaluate the final RL checkpoint unless training diverges, in which case we use the most recent stable checkpoint according to the training reward curve. For our configurations, Tables 2 and 3 report results from selected runs.

## A.4 Hyperparameters

Table 4 summarizes training, formal evaluation, and sampling for the audit and behavioral diagnostics. RL training batch sizes count questions before sampling six rollouts per question: 64 questions yield 384 trajectories for Qwen2.5-3B, and 16 yield 96 for Qwen2.5-7B and Llama-3.1-8B. The audit and behavioral diagnostic protocols are specified in Appendices E and F.

## B Dataset Statistics

WebQSP provides natural-language questions with Freebase semantic parses; CWQ extends WebQSP with compositional questions; GrailQA covers i.i.d., compositional, and zero-shot generalization. The official releases are available from Microsoft Research for WebQSP,<sup>1</sup> the authors’ project page for CWQ,<sup>2</sup> and the GrailQA project page.<sup>3</sup> GrailQA is released under CC BY-SA 4.0. Dataset files and pretrained model weights are obtained separately from their respective providers and are not included in the code repository.

Table 5 summarizes the source dataset split sizes. The CWQ/WebQSP SFT–RL pool excludes empty gold answers or topic entities and training examples sharing an original source question with any evaluation example. Deduplication leaves 27,237 questions, distinct from the raw split sizes in the table. CWQ records with missing gold answers or topic entities are omitted from scoring. Each example stores the topic entity and gold answer entities; GrailQA is trained separately.

## C Data Format and Interaction Templates

## C.1 Agent interaction protocol

Each decision includes reasoning in a <think> span followed by an executable operation. Entityand-direction selection returns the available relations for the selected entity. Relation selection then returns the traversed path and reached entity set in an <information> block. Within a relationselection operation, unnamed intermediate nodes are expanded by one additional edge in either direction to return named endpoints; this expansion counts as one navigation hop in the interaction budget. The agent continues from these entities or emits its predicted answer set in an <answer> block.

<table><tr><td>Stage</td><td>Parameter</td><td>Setting</td></tr><tr><td>SFT</td><td>Effective batch size</td><td>256</td></tr><tr><td>SFT</td><td>Learning rate</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>SFT</td><td>Epochs</td><td>8 (CWQ/WebQSP); 6 (GrailQA)</td></tr><tr><td>SFT</td><td>Maximum sequence length</td><td>5120</td></tr><tr><td>SFT</td><td>Precision and sharding</td><td>bf16; DeepSpeed ZeRO-2</td></tr><tr><td>RL</td><td>Optimizer learning rate</td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>RL</td><td>Training batch size (questions)</td><td>64 (Qwen2.5-3B); 16 (Qwen2.5-7B/Llama-3.1-8B)</td></tr><tr><td>RL</td><td>PPO minibatch size (configured)</td><td>64 (Qwen2.5-3B); 16 (Qwen2.5-7B/Llama-3.1-8B)</td></tr><tr><td>RL</td><td>Rollouts per question</td><td>6</td></tr><tr><td>RL</td><td>Rollout temperature</td><td>1.0</td></tr><tr><td>RL</td><td>Maximum prompt length</td><td>2048</td></tr><tr><td>RL</td><td>Maximum response length</td><td>4096</td></tr><tr><td>RL</td><td>Maximum navigation hops</td><td>10</td></tr><tr><td>RL</td><td>KL coefficient  $\bar { \boldsymbol \beta }$ </td><td>0.002</td></tr><tr><td>RL</td><td>Clipping</td><td>ratio 0.2 (low/high); dual-clip  $c _ { \mathrm { d c } } = 3 . 0$ </td></tr><tr><td>RL</td><td>Entropy coefficient</td><td>0.001</td></tr><tr><td>RL</td><td>Training steps</td><td>1600; checkpoint every 100 steps</td></tr><tr><td>RL</td><td>Terminal task reward</td><td>0.5 Hit@1 + 0.5 F1</td></tr><tr><td>OpCredit</td><td>FTC auxiliary weight λaux</td><td>0.1</td></tr><tr><td>OpCredit OpCredit</td><td>FTC eligibility</td><td>valid operation-local before/after frontier comparison  $1 0 ^ { - 6 }$ </td></tr><tr><td>OpCredit</td><td>Numerical stability constant ε Outcome normalization</td><td>group standard deviation for GRPO and BOC; FTC is an</td></tr><tr><td></td><td></td><td>independent signed term</td></tr><tr><td>Evaluation Evaluation</td><td>Sampled trajectories per question</td><td>1 0.8</td></tr><tr><td>Evaluation</td><td>Sampling temperature</td><td>10</td></tr><tr><td></td><td>Maximum navigation hops</td><td></td></tr><tr><td>Audit/behavior</td><td>Rollouts per question</td><td>8 1.0</td></tr><tr><td>Audit/behavior</td><td>Sampling temperature</td><td></td></tr><tr><td>Audit/behavior</td><td>Maximum navigation hops</td><td>10</td></tr></table>

Table 4: Training and evaluation settings, including rollout sampling for the CWQ credit audit and behavioral diagnostic.

<table><tr><td>Dataset</td><td>Train</td><td>Eval.</td></tr><tr><td>WebQSP</td><td>3,071</td><td>1,639</td></tr><tr><td>CWQ</td><td>27,639</td><td>3,531</td></tr><tr><td>GrailQA</td><td>44,337</td><td>6,763</td></tr></table>

Table 5: Source dataset split sizes. Eval. denotes the test split for CWQ/WebQSP and the full development split for GrailQA.

SFT trajectories are constructed from gold reasoning paths in the benchmark training data. Templates generate the reasoning text, and each trace is serialized as reasoning, executable operations,

KG observations, and final answers. All three backbones share the same SFT training data. Tables 6 and 7 give the fixed system prompt and the user-message templates. Braced fields are filled with the question or current KG observations. Entity-and-direction selection also uses the first template at the start of a rollout, with the topic entity as the current entity. For incoming relation selection, the first line of the second template is Incoming relations (to {entity\_name}):. Table 8 illustrates serialized SFT examples.

## C.2 Recorded rollout information

Credit computation uses the terminal task reward of each rollout and the navigation context, executed operation, decision span, and available before/after frontiers of its decisions. These records align BOC and FTC with the generated reasoning and executable operation in the same decision span. Appendix D gives the credit calculations and their alignment with the policy objectives.

System prompt.   
You are a KGQA agent that answers questions by navigating a knowledge graph.   
Explore the graph step by step:   
1. Choose an entity and a direction to explore:   
<explore direction="outgoing">entity\_name</explore> — follow outgoing edges from   
entity   
<explore direction="incoming">entity\_name</explore> — follow incoming edges to   
entity   
Reason in <think > tags before each action.   
2. After seeing the available relations, select the most relevant one by outputting   
its name exactly.   
3. After seeing the resulting entities, either:   
- Answer: <answer>["name1", "name2"]</answer> — JSON list of entity names   
exactly as shown   
- Continue: <explore direction="outgoing/incoming">entity\_name</explore>   
Important:   
- Check ALL entities before answering. Do not pick the first plausible one.   
- If multiple entities satisfy the question, list them all.   
- Do NOT fabricate facts. Only use information visible in names.  
Table 6: Fixed system prompt for the KGQA agent.

Entity-and-direction selection.   
Question: {question}   
Current entity: {entity\_name}   
Explore outgoing or incoming relations from this entity.   
Relation selection.   
Outgoing relations (from {entity\_name}):   
{relation\_1}   
{relation\_2}   
Select the most relevant relation name exactly as shown.   
Returned entities.   
<information>   
Question: {question}   
Paths explored:   
{head} --{relation}-->   
Entities:   
{entity\_1}   
{entity\_2}   
</information>   
Based on the entities above, answer the question or continue exploring. If   
answering,   
your answer MUST be one of the Entities listed above, copied exactly as shown.   
Maximum navigation depth.   
<information>   
Question: {question}   
Paths explored:   
{explored\_paths}   
Entities:   
{entities}   
</information>   
You have reached the maximum exploration depth. Provide your final answer in   
<answer>   
tags as a JSON list of entity names copied exactly from the results above.  
Table 7: User-message templates for KG navigation and final answering.

WebQSP SFT trace.   
Question what 5 countries border Ethiopia?   
Assistant <think> I need to inspect outgoing relations for Ethiopia. </think>   
<explore direction="outgoing">Ethiopia</explore>   
Observation Outgoing relations: location.location.adjoin\_s, ...   
Assistant <think> location.location.adjoin\_s is relevant. </think>   
location.location.adjoin\_s   
Observation <information> Ethiopia -> Eritrea, Djibouti, Somalia, Sudan, Kenya   
</information>   
Answer <answer>["Eritrea", "Djibouti", "Somalia", "Sudan", "Kenya"]</answer>   
CWQ SFT trace.   
Question Which movies produced by Universal Studios has Ron Howard directed?   
Assistant <think> I need to inspect outgoing relations for Ron Howard. </think>   
<explore direction="outgoing">Ron Howard</explore>   
Observation Outgoing relations: film.director.film, film.producer.film, ...   
Assistant <think> film.director.film is relevant. </think>   
film.director.film   
Observation <information> Ron Howard -> EDtv, Cinderella Man, A Beautiful Mind, ...   
</information>   
Answer <answer>["EDtv", "Cinderella Man", "A Beautiful Mind"]</answer>   
Table 8: Serialized SFT examples from WebQSP and CWQ. The examples illustrate the operation and observation   
format.

## D Credit Computation and Token Alignment

## D.1 Branch Outcome Credit

For the example in Figure 2, consider three samequestion rollouts τ<sub>1</sub>, τ<sub>2</sub>, and $\tau _ { 3 }$ whose operations $o ^ { \prime } .$ $o _ { t } ,$ , and $o ^ { \prime \prime }$ at decision t share the same active entity and context $c ,$ with terminal task rewards $R ^ { \prime } = 0$ $R = 1$ , and $R ^ { \prime \prime } = 0$ , respectively. Each operation occurs in one rollout, so Eq. (3) gives

$$
\begin{array} { c } { { \displaystyle \widehat { Q } _ { \mathrm { o u t } } ( c , o _ { t } ) = 1 , } } \\ { { \displaystyle \widehat { Q } _ { \mathrm { o u t } } ( c , o ^ { \prime } ) = \widehat { Q } _ { \mathrm { o u t } } ( c , o ^ { \prime \prime } ) = 0 , } } \\ { { \displaystyle \widehat { V } _ { \mathrm { o u t } } ( c ) = \displaystyle \frac { 0 + 1 + 0 } { 3 } = \displaystyle \frac { 1 } { 3 } . } } \end{array}
$$

Substituting these values into Eq. (4) yields

$$
\begin{array} { r } { A _ { \mathrm { B O C } } ^ { 2 , t } = \cfrac { 2 } { 3 \left( \sigma _ { q } + \varepsilon \right) } , } \\ { A _ { \mathrm { B O C } } ^ { 1 , t } = A _ { \mathrm { B O C } } ^ { 3 , t } = - \cfrac { 1 } { 3 \left( \sigma _ { q } + \varepsilon \right) } . } \end{array}
$$

Thus $o _ { t }$ receives positive BOC, while $o ^ { \prime }$ and $o ^ { \prime \prime }$ receive negative BOC.

## D.2 Frontier Transition Credit

Table 9 specifies the operation-local frontiers. An active entity is the entity from which the current navigation hop proceeds; a singleton frontier contains that entity alone. These comparisons use the entities involved in each operation, rather than the cumulative set of retrieved entities.

<table><tr><td>Operation</td><td>Before frontier  $F ^ { - }$ </td><td>After frontier  $F ^ { + }$ </td></tr><tr><td>Entity-and- direction</td><td>{previous active entity}</td><td>{selected entity}</td></tr><tr><td>selection Relation selection {active entity}</td><td></td><td>Returned entity set</td></tr><tr><td>Answer emission {active entity}</td><td></td><td>Predicted answer</td></tr><tr><td></td><td></td><td>entity set</td></tr></table>

Table 9: Operation-local frontiers for FTC. Braces denote a singleton entity set.

To construct the one-hop neighborhood used by $\delta _ { q }$ in Eq. (5), we retrieve bounded incoming and outgoing KG neighbors of each mapped gold answer entity, exclude deprecated or internal predicates, and take the union across gold answer entities. The bounded lookup uses shallow bidirectional neighborhood expansion and set intersection. If no gold answer can be mapped to a KG entity, the question receives no FTC; an entity unresolved within the agent’s visible frontier suppresses only that decision’s FTC. At answer emission, the after frontier contains the predicted answer entities and is evaluated using the same maximum-potential aggregation. Empty or unavailable frontiers also receive no FTC. For nonempty, available frontiers, Eqs. (5) and (6) give the equivalent computation

$$
A _ { \mathrm { F T C } } ^ { i , t } = \operatorname* { m i n } _ { e \in F _ { i , t } ^ { - } } \delta _ { q } ( e ) - \operatorname* { m i n } _ { e \in F _ { i , t } ^ { + } } \delta _ { q } ( e ) .\tag{11}
$$

## D.3 Decision spans and joint objective

We use the decision spans and credit assignments defined in Section 3.4. Each loss term is averaged over generated tokens within a trajectory and then over trajectories, giving each trajectory equal weight. Observation and padding tokens are excluded. KL regularization uses the frozen SFTinitialized policy as its reference; the entropy bonus uses the current policy’s token distribution.

BOC surrogate. For the policy ratio in Eq. (7), define

$$
\begin{array} { c } { \bar { \rho } = \mathrm { c l i p } ( \rho , 1 - \eta , 1 + \eta ) , } \\ { m ( \rho , A ) = \operatorname* { m i n } \{ \rho A , \bar { \rho } A \} . } \end{array}
$$

The dual-clipped surrogate $D$ in ${ \mathcal { L } } _ { \mathrm { B O C } } ( \theta )$ is

$$
\begin{array} { l l } { { D ( \rho , A ) = m ( \rho , A ) , } } & { { A \geq 0 , } } \\ { { { \cal D } ( \rho , A ) = \operatorname* { m a x } \{ m ( \rho , A ) , c _ { \mathrm { d c } } A \} , } } & { { A < 0 . } } \end{array}\tag{12}
$$

Here η is the policy-ratio clipping parameter and $c _ { \mathrm { d c } }$ is the dual-clip bound (set to 3.0 in our runs).

## D.4 Analysis of Navigation Context Matching

We examine the role of decision phase in BOC navigation context matching using CWQ diagnostic rollouts from Qwen2.5-3B with FTC at steps 100, 300, 500, and 600, and Qwen2.5-7B with BOC at steps 100, 300, and 600. Each checkpoint contributes eight rollouts per question on 500 questions. The analysis considers relation selection and entity-and-direction selection. Every matching rule restricts both outcome estimates to the same question and active entity, as in Section 3.2. We hold terminal task rewards fixed and perform no parameter updates. For each matching rule, we recompute the context-relative outcome credit using the same normalization as in training. Within each question–active-entity group, we compare three alternatives with the full context: (i) active entity only, with no further matching; (ii) visited entity set only, matching $E _ { \mathrm { v i s } } ( s )$ ; and (iii) visited entity set with remaining navigation budget, matching $( E _ { \mathrm { v i s } } ( s ) , b ( s ) )$ . The full context in Eq. (2) additionally matches decision phase $\phi ( s )$ . The three alternatives without decision phase produce identical partitions on these rollouts and are grouped in one row in Table 10.

The decision-weighted mixing rate is the fraction of decision occurrences in context groups containing both relation selection and entity-and-direction selection. The group-weighted rate is the fraction of context groups containing both operation types.

<table><tr><td>Context</td><td>Decision- weighted</td><td>Group- weighted</td></tr><tr><td>Without decision phase</td><td>46–48%</td><td>30-31%</td></tr><tr><td>Full</td><td>0%</td><td>0%</td></tr></table>

Table 10: Decision-phase mixing under different navigation context matching rules. Rates are consistent across the evaluated checkpoints.

The decision-weighted rate measures how many decisions share a context-relative outcome baseline with a different operation type. Without decision phase, this affects nearly half of the sampled decision occurrences. Matching the decision phase provides a baseline formed from the same operation type, aligning the local outcome comparison with the kind of decision being made. The visited entity set and remaining navigation budget retain the other matching constraints.

## E Reference-Blinded Credit Audit

## E.1 Audit setup

The audit uses a common rollout pool generated by the Qwen2.5-3B SFT policy on 500 CWQ questions, with eight rollouts per question. It contains 11,298 operation occurrences; 1,200 are selected for usefulness assessment after removing duplicate operation signatures within each question and capping the number assessed per question.

The compared methods are GRPO, GiGPO, RTMC, GraphGPO, and OpCredit. We reimplement the published credit-assignment rules for KGQA and compute their credits on the complete rollout pool before selecting the judged occurrences. The rollout pool, terminal task rewards, and assessed operations are held fixed across methods.

## E.2 Reference-blinded usefulness assessment

We use Qwen3.7-Max as the reference-blinded judge with temperature 0. Given the question, interaction history preceding the operation, the operation itself, and the graph observations it returns, the judge provides a usefulness rating and a onesentence reason. It does not receive the gold answer, terminal task reward, answer-distance label, assigned credit, or later trajectory steps. Table 12 gives the prompt and five-point annotation rubric. For mean-credit visualization, ratings 2 and 3 share one group, yielding the four ordered levels in Figure 3. Table 11 relates the original ratings to the displayed levels and AUC labels.

<table><tr><td>Judge rating</td><td>Displayed level</td><td>AUC label</td></tr><tr><td>1</td><td>1</td><td>Unhelpful</td></tr><tr><td>2</td><td>2</td><td>Unhelpful</td></tr><tr><td>3</td><td>2</td><td>Excluded (neutral)</td></tr><tr><td>4</td><td>3</td><td>Useful</td></tr><tr><td>5</td><td>4</td><td>Useful</td></tr></table>

Table 11: Annotation ratings, displayed usefulness levels, and AUC labels.

Truncated relation names were resolved against the KG, and operations whose visible judge prompts changed were reassessed.

## E.3 Credit metrics and results

For OpCredit, the replay credit on an eligible operation is

$$
A _ { \mathrm { r e p l a y } } = A _ { \mathrm { B O C } } + \lambda _ { \mathrm { a u x } } A _ { \mathrm { F T C } } ,\tag{13}
$$

where $\lambda _ { \mathrm { a u x } } = 0 . 1$ and $A _ { \mathrm { B O C } }$ follows the assignment in Eq. (4), with rollout and decision indices omitted. With policy ratio one and fixed aggregation, this is the operation-level policy-gradient coefficient excluding KL and entropy terms.

For each method, we record the mean replay credit in each usefulness level and the usefulnessdiscrimination AUC. AUC uses assigned credit as the score, with original ratings 4–5 as positive labels and 1–2 as negative labels; rating 3 is excluded. All methods use the same labeled operations. AUC is computed from individual-operation credit rankings before grouping ratings for the mean-credit display. Mean credit describes the direction of supervision within each method, while AUC compares usefulness discrimination without relying on a common credit scale. Table 13 reports the complete results.

On the shared rollout pool, OpCredit, GRPO, and RTMC all show increasing mean credit across usefulness levels. OpCredit achieves the highest AUC, indicating stronger discriminability between useful and unhelpful operations.

## F Behavioral Diagnostic

## F.1 Diagnostic setup and metrics

We compare Qwen2.5-3B OpCredit with BOConly (OpCredit without FTC) on 500 CWQ questions sampled with seed 42, using the diagnostic rollout settings in Table 4. Each configuration is evaluated at SFT and six training checkpoints (100, 200, 300, 400, 500, and 600).

For each operation, the diagnostic records a four-level diagnostic distance label $d _ { \mathrm { d i a g } } \mathrm { : }$ 0 for a gold answer entity, 1 for the constructed one-hop neighborhood of the gold answer entities, 2 for the restricted second-level neighborhood, and 3 for a more distant candidate. The diagnostic distinguishes second-level and more distant candidates within FTC label 2 in Eq. (5).

The wrong-answer near-distance rate is

$$
P ( d _ { \mathrm { a n s w e r } } \leq 1 \mid \mathrm { f i n a l ~ a n s w e r ~ i s ~ w r o n g } ) .
$$

Here $d _ { \mathrm { a n s w e r } }$ is the diagnostic distance label of the predicted answer. The denominator is the set of wrong-answer trajectories, and the metric asks whether an incorrect answer is nevertheless in the answer neighborhood. The recovery rate is a trajectory-level conditional probability:

$$
P \big ( \mathrm { c o r r e c t } \mid \mathrm { t r a j e c t o r y h a s b a c k w a r d o p . } \big ) ,
$$

where a backward operation is a navigation operation with $\Delta d _ { \mathrm { d i a g } } = d _ { \mathrm { d i a g } } ^ { + } - d _ { \mathrm { d i a g } } ^ { - } > 0$ . The denominator contains trajectories with at least one such operation.

Illustrative recovery. For “What all has the actor who played in Knights of the Roundtable: King Arthur been $\mathrm { i n ? } ^ { \flat }$ , the recorded before/after $d _ { \mathrm { d i a g } }$ pairs for four operations are $0  3 , 0  3 , 3  { \bar { 0 } }$ and $3 \to 0$ . Each pair describes an operation-local comparison, so adjacent pairs need not share an endpoint. The trajectory first moves away from the answer region, then changes relation and returns to it before producing the correct answer.

## F.2 Seven-checkpoint results

Figure 5 and Table 14 report the checkpoint-wise results. OpCredit maintains higher recovery rates than BOC-only at the later checkpoints, reaching 26.5% versus 18.3% at step 600. This comparison concerns trajectories that contain a backward operation, locating the difference in recovery following such operations. The wrong-answer near-distance rate shows smaller differences with no consistent direction across checkpoints. The clearest distinction between the configurations is therefore in recovery after backward operations.

<table><tr><td>Message</td><td>Content</td></tr><tr><td>System</td><td>You are an expert evaluator for knowledge-graph navigation steps. Given a question and an agent&#x27;s navigation trajectory on a Freebase-style knowledge graph, rate one operation for how much it advances answering the question. You do not know the correct answer: judge purely from the semantics of the question, the operation description, and the entities involved. Use the following scale: 5 = clearly moves toward the answer; 4 = probably helpful; 3 = plausible exploration, neutral, or uncertain; 2 = probably unhelpful; 1 = clearly off-topic or counterproductive. Respond with only a</td></tr><tr><td>User</td><td>JSON object of the form  $\{ " \Upsilon { \bar { \mathrm { a t i n g } } } " \} \quad < \dot { 1 } - 5 > ,$  &quot;reason&quot;:&quot;&lt;one sentence&gt;&quot;}. Question: {question}. Navigation path so far: {path-before-the-operation}. Operation to evaluate: {operation description}. Rate this single operation from 1 to 5.</td></tr></table>

Graph observations. The observations returned by the evaluated operation are included in {operation description}.

Table 12: Reference-blinded judge prompt.  
![](images/59b128fb1882317ea22133325d7036b99a66f158a75d09296f607243037760f6.jpg)

![](images/622dd17e042558aa46bbc9d1af0f25c68aeecfc8c42ff8e1f6f9b1e9822d6eef.jpg)  
Figure 5: Behavioral diagnostic on CWQ for Qwen2.5-3B. The recovery rate is conditioned on trajectories containing a backward operation $( \Delta d _ { \mathrm { d i a g } } > 0 ) ;$ the near-distance rate is conditioned on wrong final answers. Values are single-run percentages over the fixed diagnostic protocol. OpCredit rises to the highest late-training recovery rate, while the two methods have smaller, less consistent differences in near-distance errors.

<table><tr><td>Method</td><td>Level 1 Level 2</td><td>Level 3</td><td>Level 4 AUC</td></tr><tr><td>GRPO</td><td>-0.134</td><td>-0.019 +0.005 +0.014</td><td>0.606</td></tr><tr><td>GiGPO</td><td>-0.556</td><td>-0.003 -0.056</td><td>+0.087 0.592</td></tr><tr><td>RTMC</td><td>-0.108</td><td>-0.031 -0.016</td><td>+0.003 0.492</td></tr><tr><td>GraphGPO</td><td>-0.471</td><td>+0.000 -0.020</td><td>+0.057 0.595</td></tr><tr><td>OpCredit</td><td>-0.347 +0.016</td><td>6+0.050</td><td>+0.1440.718</td></tr></table>

Table 13: Credit-assignment replay on the common SFT rollout pool. Mean credit is reported for the four ordered usefulness levels; AUC measures usefulness discrimination.

<table><tr><td>Metric SFT 100 200 300 400 500 600</td></tr><tr><td>Near-distance, OpCredit 7.9 8.4 8.8 8.8 9.8 12.2 10.7</td></tr><tr><td>Near-distance, BOC-only 7.9 8.0 9.3 8.9 10.3 9.6 9.8</td></tr><tr><td>Recovery, OpCredit 20.6 23.9 24.6 23.8 26.5 28.1 26.5</td></tr><tr><td>Recovery, BOC-only 20.6 25.4 24.8 23.1 22.9 21.0 18.3</td></tr></table>

Table 14: Behavioral diagnostic on CWQ. Values are percentages; BOC-only denotes OpCredit without FTC.

## G Error Analysis Details

## G.1 Diagnostic protocol

We compare GRPO and OpCredit on 492 matched diagnostic questions per dataset using the agent’s executable KG interface. Errors are summarized at the question level on these diagnostic subsets. Figure 4 reports category shares within each method’s error set.

## G.2 Error categories

The automated diagnostic assigns four labels based on candidate availability, reference-path alignment, and answer emission:

• Answer selection. An incorrect answer receives this label when a gold-answer name occurs in the final frontier or the complete path aligns with a reference path. Completepath alignment serves as a proxy for reaching relevant evidence.

• Path-first. The recorded deviation from a reference path occurs at the first relation hop.

• Path-late. The recorded deviation occurs at the second or a later hop, or the unsuccessful trajectory has no usable reference-path alignment. This category therefore includes unaligned residual cases as well as localized later deviations.

• Empty answer. The predicted answer set is empty.

Path deviations are defined relative to the available reference paths.

## G.3 Complete diagnostic statistics

Table 15 reports error counts and rounded category shares. The percentages may not sum to 100%; Figure 4 preserves these labels and normalizes the bar widths.

<table><tr><td>Method</td><td>Err.</td><td>Sel. First Late Empty</td><td></td><td></td><td></td></tr><tr><td>CWQ</td><td colspan="5"></td></tr><tr><td>GRPO</td><td>105</td><td>67</td><td>23</td><td>10</td><td>0</td></tr><tr><td>OpCredit</td><td>97</td><td>61</td><td>23</td><td>16</td><td>0</td></tr><tr><td>WebQSP</td><td></td><td>27</td><td>22</td><td></td><td></td></tr><tr><td>GRPO</td><td>55</td><td>20</td><td>24</td><td>35</td><td>16</td></tr><tr><td>OpCredit</td><td>49</td><td></td><td></td><td>33</td><td>22</td></tr><tr><td>GrailQA</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GRPO</td><td>107</td><td>31</td><td>47</td><td>14</td><td>8</td></tr><tr><td>OpCredit</td><td>95</td><td>23</td><td>51</td><td>20</td><td>6</td></tr></table>

Table 15: Total error counts (Err.) and within-method category percentages on 492 diagnostic questions per dataset. Sel.: Answer selection; First: Path-first; Late: Path-late; Empty: Empty answer.