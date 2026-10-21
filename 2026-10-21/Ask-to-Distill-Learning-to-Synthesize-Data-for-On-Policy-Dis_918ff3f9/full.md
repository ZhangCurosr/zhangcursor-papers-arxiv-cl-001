# Ask to Distill: Learning to Synthesize Data for On-Policy Distillation

Jiashuo Sun <sup>1</sup> Hao Zhang <sup>2</sup> Xueqiang Xu <sup>1</sup> Heng Wang <sup>1</sup> Yixuan Xie <sup>1</sup> Pengcheng Jiang <sup>1</sup> Zhenrui Yue <sup>1</sup> Jiajun Fan <sup>1</sup> Peiran Li <sup>3</sup> Ge Liu <sup>1</sup> Shaowen Wang <sup>1</sup> Jiawei Han <sup>1</sup>

GitHub Model and Data Collections Eval Data

## Abstract

On-policy distillation (OPD) can succeed with surprisingly few training questions, motivating a view of questions as carriers of distributional knowledge transfer. How can we assess whether a newly generated question is valuable for distillation, and learn to generate better questions— without access to its correct answer? We introduce Ask-to-Distill (A2D), a framework that treats synthetic questions as distribution elicitors and optimizes the teacher supervision they expose along student-generated trajectories. A student-initialized synthesizer is optimized using a distributional surrogate for question value, computed from a frozen student–teacher pair without reference answers, correctness rewards, or per-candidate student updates. The reward combines teacher–student disagreement with teacher probability mass on the student’s candidate support, together with length and diversity controls. The generated questions are filtered and deduplicated into a prompt-only bank for standard OPD. Across mathematical reasoning, code generation, and agentic tasks, our method improves average performance and teacher-gap recovery over unoptimized student-generated prompts and KL-filtered real questions, while matching or exceeding real-bank OPD in the evaluated settings. Signal analyses further show that optimized questions can elicit denser supervision and more total signal from shorter student responses, supporting elicited distributional supervision as an objective for synthetic distillation data. Our code, data and models are publicly available at https://github.com/GasolSun36/As k\_to\_Distill.

![](images/546dfccc47344e4083b4e414896e11eaf3012e0e69ad4b36c6713ab5aa057694.jpg)

![](images/ea77238feed2d00c316aa1b61d2a23b344f403dcdb2a0529d400e05b704bb9df.jpg)  
Figure 1. A2D learns to ask questions where the teacher can help. (a) Schematic. A student-initialized synthesizer is RLtrained against a frozen student–teacher pair to elicit disagreement δ that remains within the student’s top-K support (compatibility ω). Real questions mostly elicit little disagreement or disagreement outside the student’s reach, which KL filtering cannot separate. Insets: typical student and teacher next-token distributions per region. (b) Generated questions yield 20% shorter responses and 28% more total signal than DAPO-17k-Math.

## 1. Introduction

On-policy distillation (OPD) transfers the behavior of a strong teacher to a student on trajectories sampled by the student itself (Agarwal et al., 2024; Gu et al., 2024). By supervising the states the student actually visits, OPD mitigates the distribution mismatch associated with learning only from fixed teacher demonstrations and provides dense token-level guidance without requiring a task-specific correctness verifier. Although the teacher supplies this supervision, the training questions determine which trajectories the student generates and where teacher feedback becomes available. Constructing effective distillation data therefore requires understanding not only what a question asks, but also what supervision it elicits from a particular student– teacher pair.

Recent studies suggest that OPD can be highly promptefficient: small subsets can approach full-data performance, and even a single question can recover much of the gain obtained from a large question bank (Fu et al., 2026; Hou et al., 2026). Moreover, self-generated questions can support effective OPD without external seed examples (Li et al., 2026a). These findings motivate a different perspective on OPD training data, centered on the supervision that questions elicit. In supervised fine-tuning, synthetic examples provide target responses for imitation (Wang et al., 2023; Xu et al., 2024); in self-play reinforcement learning, generated tasks provide opportunities to learn from answer-quality or task-success feedback (Huang et al., 2026; Chen et al., 2025). OPD instead obtains supervision from the teacher’s token distributions on student-generated prefixes. A question therefore serves as a distribution elicitor: it guides the student into states where teacher–student differences become available for learning. From this perspective, training questions act as carriers of distributional knowledge transfer, rather than directly supplying the target knowledge through reference answers. Their distillation value depends on the usable teacher supervision exposed along the induced trajectories for a particular student–teacher pair, consistent with findings that prompt utility is pair-dependent (Wang et al., 2026a). Although OPD supplies teacher supervision without reference answers, optimizing a question synthesizer requires an additional signal: feedback on which questions are worth generating. This raises a central question: without access to the correct answer to a newly generated question, how can we assess its valuefor a target student–teacher pair and use that assessment to learn to generate better questions? We address this question by scoring the distributiona supervision exposed along student-generated trajectories, using a frozen student–teacher pair without performing a student update for each candidate.

Turning elicited supervision into an optimization objective requires distinguishing disagreement from useful correction. Token-selection and reweighting methods already exploit the unequal value of supervision across response positions (Xu et al., 2026; Li et al., 2026b). In particular, support-aware analyses of token teachability show that large divergence can reflect either corrective teacher preferences within the student’s candidate support or a mismatch concentrated outside that support (Wang et al., 2026b). Building on this characterization, we combine teacher–student discrepancy with teacher probability mass on the student’s candidate tokens as a surrogate for local learning value. We adapt this token-level principle to question generation by aggregating the signal along student trajectories. This introduces additional challenges: a synthesizer may increase its reward through longer responses or repeatedly produce a narrow family of questions. The question-level objective must therefore balance elicited signal with trajectory length and question diversity.

We introduce Ask-to-Distill (A2D), a self-contained framework that optimizes synthetic questions for a target student– teacher pair before performing OPD. We initialize the question-generation policy from the student and train it with reinforcement learning, while keeping the scoring student and teacher frozen. Its reward aggregates support-aware token signals using sublinear length normalization, a soft response-budget penalty, and a batch-diversity term. We then curate the generated questions through format filtering and lexical and semantic deduplication, and perform stan dard OPD on the resulting prompt-only dataset. Synthesizer training reuses the target student–teacher pair and requires neither external training questions nor auxiliary models. After training, a frozen Qwen3-Embedding-4B model is used only for semantic deduplication of the generated question bank; it does not participate in synthesizer optimization. Our method operates on the prompt-construction axis, which is complementary to approaches that select supervision positions, reweight trajectories, or modify the downstream distillation procedure.

Our contributions are threefold: (i) we formulate synthetic prompt construction as optimizing the distributional supervision elicited from a target student–teacher pair; (ii) we develop a question-level reward that turns support-aware teacher–student signals into feedback for synthesizer optimization, without reference answers, task-specific correctness rewards, or per-candidate student updates, while controlling response length and question diversity; and (iii) we validate the framework across multiple model scales and mathematical reasoning, code generation, and agentic tasks through downstream comparisons, reward-component ablations, and learning-signal analyses. A2D exceeds real-bank OPD on the average scores of the evaluated student–teacher pairs. Our analyses further show that optimized questions can elicit denser distributional signals from shorter student responses, supporting the value of optimizing elicited supervision beyond simply increasing rollout length.

## 2. Method

## 2.1. Problem formulation: synthetic prompts as distribution elicitors

On-policy distillation (OPD) provides teacher supervision on trajectories sampled by the student. Unlike synthetic examples used for supervised fine-tuning, a question used for OPD need not include a reference response: it supplies the context from which the student generates a trajectory and the teacher provides token-level supervision. Our synthetic data therefore consist of questions only, without groundtruth answers or reference solutions; neither synthesizer reward computation nor downstream OPD requires such annotations. We therefore study synthetic questions as distribution elicitors, whose value depends on the supervision they expose for a particular student–teacher pair.

![](images/69b911b98f52530ad31eb4668fc3a9067a63e657d9b7598771daee4d59a120d0.jpg)  
Figure 2. Overview of A2D. A student-initialized synthesizer is optimized using learning-value signals from a frozen student–teacher pair. The generated questions are filtered and deduplicated into a prompt-only synthetic bank, which is then used for on-policy distillation with a frozen teacher providing supervision on the student’s top-K candidates.

Let $\pi _ { S }$ and $\pi _ { T }$ denote the student and teacher, with policies $\pi _ { S }$ and $\pi _ { T }$ . Given a question $q ,$ the student samples a response $y \sim \pi _ { S } ( \cdot \mid q )$ of length L. At each response position t, both models are evaluated on the student-generated prefix $c _ { t } = ( q , y _ { < t } )$ , producing next-token distributions $\pi _ { S } ( \cdot \mid c _ { t } )$ and $\pi _ { T } ( \cdot \mid c _ { t } )$ . The question thus induces a distribution over the states at which distillation takes place.

Our goal is to construct questions that expose useful teacher supervision along these trajectories. Disagreement alone is insufficient: a question may elicit substantially different predictions while placing little teacher probability mass on the candidates considered by the student’s update. We therefore assess questions using both teacher–student discrepancy and compatibility, while retaining basic format and self-containedness requirements during curation.

We formalize the elicited signal through a trajectory-based surrogate:

$$
\mathcal { V } ( q ; \pi _ { S } , \pi _ { T } ) = \mathbb { E } _ { y \sim \pi _ { S } ( \cdot | q ) } \left[ \frac { 1 } { L ^ { \alpha } } \sum _ { t = 1 } ^ { L } s _ { t } \right] ,\tag{1}
$$

where $s _ { t }$ is a support-aware token score defined in §2.3, and α controls the balance between signal density and trajectory length. This quantity measures the distributional signal exposed by a question; it is a surrogate for learning value rather than a direct estimate of downstream performance improvement.

We parameterize the question distribution with a synthesizer $S _ { \theta }$ , initialized as an independent copy of the initial student. During synthesizer optimization, the scoring student and teacher remain frozen, and only θ is updated. The synthesizer’s reward combines a sampled estimate of Eq. (1) with response-length and question-diversity controls. The resulting questions are curated and used for downstream OPD, which starts from the original student parameters. Question generation and synthesizer training use only the target student–teacher pair and the student-initialized synthesizer, without external training questions, labeled examples, or auxiliary models. A separate frozen embedding model is used only for semantic deduplication after question generation (§2.5).

## 2.2. Overview

The basic pipeline consists of three stages, each performed once. We also consider repeating the pipeline with the updated student in §3.10.

1. Synthetic question optimization (§2.3–§2.4). We train $S _ { \theta }$ with group relative policy optimization (GRPO) (Shao et al., 2024). Each generated question is answered by the frozen student, and the frozen teacher is evaluated on the same student-generated prefixes. Their predictions determine a question-level learning-value reward.

2. Synthetic question curation (§2.5). We sample the trained synthesizer at scale, filter malformed outputs, and remove lexical and semantic duplicates to construct a prompt-only question bank.

3. On-policy distillation (§2.6). We update the original student using teacher supervision on responses sampled from the curated questions, with the same response budget used for question scoring.

The synthesizer and scoring student share their initial parameters but are separate model copies. Only the synthesizer is updated in stage 1; only the student is updated in stage 3. The teacher remains frozen throughout. Algorithm 1 in the appendix summarizes the full pipeline; hyperparameter values for all stages are reported in Appendix A.2.

## 2.3. The learning-value reward

Per-position quantities. Given a question q, we sample a student response $y \sim \pi _ { S } ( \cdot \mid q )$ under a response budget $L _ { \mathrm { m a x } }$ . Let $L \leq L _ { \operatorname* { m a x } }$ be its length and $c _ { t } = ( q , y _ { < t } )$ its prefix at position t. We write $p _ { S } ( v ) \ = \ \pi _ { S } ( v \ | \ c _ { t } )$ and $p _ { T } ( v ) = \pi _ { T } ( v \mid c _ { t } )$ , suppressing the dependence on t for readability. For a candidate size $K .$ , we define the student and teacher candidate sets and their union:

$$
\begin{array} { r l r } & { A _ { t } = \mathrm { T o p } { - } K ( p _ { S } ) , \ } & { B _ { t } = \mathrm { T o p } { - } K ( p _ { T } ) , } \\ & { U _ { t } = A _ { t } \cup B _ { t } , } & { | U _ { t } | \leq 2 K . } \end{array}\tag{2}
$$

For $M \in \{ S , T \}$ , let

$$
\bar { p } _ { M } ^ { U } ( v ) = \frac { p _ { M } ( v ) } { \sum _ { u \in U _ { t } } p _ { M } ( u ) } , \qquad v \in U _ { t } ,\tag{3}
$$

denote the corresponding distribution restricted to and renormalized over $U _ { t }$

Following support-aware characterizations of token teachability (Wang et al., 2026b), we use two quantities to charac-

terize the supervision at each position:

$$
\begin{array} { r l } & { \delta _ { t } = \mathrm { K L } \big ( \hat { p } _ { T } ^ { U } \| \hat { p } _ { S } ^ { U } \big ) = \displaystyle \sum _ { v \in U _ { t } } \hat { p } _ { T } ^ { U } ( v ) \log \frac { \hat { p } _ { T } ^ { U } ( v ) } { \hat { p } _ { S } ^ { U } ( v ) } , } \\ & { \omega _ { t } = \displaystyle \sum _ { v \in A _ { t } } p _ { T } ( v ) . } \end{array}\tag{4}
$$

(5)

The elicited discrepancy $\delta _ { t }$ measures differences between teacher and student preferences over their combined candidate set. Its teacher-to-student KL direction emphasizes teacher-preferred candidates to which the student assigns relatively low probability. The compatibility $\omega _ { t }$ measures the teacher’s probability mass on the student’s top-K candidates, which are the candidates used by our downstream OPD update. Unlike $\delta _ { t } , \omega _ { t }$ uses the original full-vocabulary teacher probabilities without renormalization.

Gating discrepancy by compatibility. Large discrepancy can occur even when the teacher places little probability mass on the student’s candidate set. Candidates outside this set do not receive explicit candidate-level terms in our downstream update (§2.6). We therefore combine discrepancy and compatibility at each position:

$$
s _ { t } = \mathrm { m i n } \Bigg ( \frac { \delta _ { t } } { \delta _ { \mathrm { { r e f } } } + \epsilon } , 1 \Bigg ) \omega _ { t } ,\tag{6}
$$

where ϵ is a small constant for numerical stability and $\delta _ { \mathrm { r e f } }$ is a fixed calibration scale. The resulting score lies in [0, 1] and favors positions with both substantial discrepancy and teacher probability mass within the student’s candidate support. Multiplying the quantities before aggregation requires them to occur at the same position; multiplying their trajectory-level means could instead combine high discrepancy at one position with high compatibility at another. This construction remains a proxy for useful supervision and does not guarantee that every high-scoring position yields a large learning gain.

Calibrating $\delta _ { \mathrm { r e f } } .$ . We set $\delta _ { \mathrm { r e f } }$ to the κ-th percentile of token-level discrepancies measured on $N _ { \mathrm { c a l } }$ questions sampled from the initial student policy. The frozen student answers these questions under the response budget used for downstream OPD, and the teacher is evaluated on the resulting student-generated prefixes. We pool the resulting $\delta _ { t }$ values to compute the percentile and hold it fixed during synthesizer optimization. This calibration uses no external questions and provides a common discrepancy scale across optimization steps.

Length control. We aggregate the token scores using

$$
R _ { \mathrm { r a w } } ( q ) = \frac { 1 } { L ^ { \alpha } } \sum _ { t = 1 } ^ { L } s _ { t } .\tag{7}
$$

Writing $\begin{array} { r } { \bar { s } ( q ) = L ^ { - 1 } \sum _ { t = 1 } ^ { L } s _ { t } } \end{array}$ gives $R _ { \mathrm { r a w } } ( q ) = L ^ { 1 - \alpha } \bar { s } ( q )$ Thus, α controls the balance between average signal and accumulated signal. A per-token mean $( \alpha = 1 )$ gives the same score to trajectories with equal average signal regardless of length, while a sum $( \alpha = 0 )$ grows linearly with length at fixed average signal. We use an intermediate exponent $0 < \alpha < 1$ to reward sustained signal while reducing the advantage conferred by additional tokens.

This aggregation retains an incentive for longer responses. We moderate it with a multiplicative penalty based on response-budget consumption:

$$
P _ { \mathrm { l e n } } ( q ) = 1 - \lambda _ { \mathrm { l e n } } \operatorname* { m i n } \biggl ( \frac { L } { L _ { \mathrm { m a x } } } , 1 \biggr ) .\tag{8}
$$

Here, $\lambda _ { \mathrm { l e n } } \in ( 0 , 1 )$ sets the penalty strength. The penalty decreases with response length and equals $1 - \lambda _ { \mathrm { l e n } }$ at the generation cap. Its multiplicative form makes the relative discount independent of the absolute scale of $R _ { \mathrm { r a w } }$ . Together, these controls attenuate length-driven reward growth while retaining a preference for signal accumulated across response positions; they do not eliminate the incentive for longer trajectories.

Diversity control. Optimizing the trajectory-level signal alone can concentrate generation on repetitive questions. Following the cluster-based repetition penalty in R-Zero (Huang et al., 2026), we measure within-batch redundancy using BLEU-based text similarity, without extracting hidden-state representations or using an embedding model.

Let $\boldsymbol { B } = \{ q _ { 1 } , \dots , q _ { N _ { B } } \}$ denote the full batch of questions scored in one synthesizer-training step. We construct a pairwise distance matrix from one minus the sentence-level BLEU similarity between question texts, following the R-Zero implementation, and apply average-linkage agglomerative clustering with distance threshold τ. If $\ell _ { i }$ is the cluster label of $q _ { i } .$ , its cluster share is

$$
\mathrm { s h a r e } ( q _ { i } ; B ) = \frac { | \{ j : \ell _ { j } = \ell _ { i } \} | } { N _ { B } } .\tag{9}
$$

We then define

$$
P _ { \mathrm { d i v } } ( q _ { i } ; \mathcal { B } ) = \operatorname* { m a x } ( 0 , 1 - \lambda _ { \mathrm { d i v } } \mathrm { s h a r e } ( q _ { i } ; \mathcal { B } ) ) ,\tag{10}
$$

where $\lambda _ { \mathrm { d i v } } > 0$ sets the penalty strength.

Questions in larger text-similarity clusters receive a stronger discount. A singleton cluster receives a factor of $1 \ : -$ $\lambda _ { \mathrm { d i v } } / N _ { B }$ , while for $\lambda _ { \mathrm { d i v } } \geq 1$ a batch assigned entirely to one cluster receives zero reward. The multiplicative form discounts the elicited signal proportionally, avoiding an additive penalty whose relative strength would depend on the signal’s numerical scale. Clustering is performed across the full scoring batch, separately from the within-group advantage normalization used by GRPO.

Final question-level reward. The reward for a question combines its sampled token signals with the length and diversity factors:

$$
R ( q ; \mathcal { B } ) = \frac { P _ { \mathrm { l e n } } ( q ) P _ { \mathrm { d i v } } ( q ; \mathcal { B } ) } { L ^ { \alpha } } \sum _ { t = 1 } ^ { L } s _ { t } .\tag{11}
$$

We make the batch dependence explicit because the diversity factor depends on the other questions scored in the same step. The token-signal term is estimated from one student response per question, providing a single-sample estimate of the expectation in Eq. (1) before applying the additional controls. Appendix B establishes a compatibledisagreement coverage bound for the normalized reward and relates compatibility to the OPD candidate advantages.

## 2.4. Optimizing the synthetic data distribution

We initialize $S _ { \theta }$ from the student to use its existing generative capabilities as the starting point for question construction. Shared initialization supplies a prior for generation; the reward determines which questions are favored for the target student–teacher pair. Throughout this stage, the scoring student and teacher remain frozen.

For each question-generation instruction $x ,$ we sample a group of $g$ candidate questions from $S _ { \theta } ( \cdot \mid x )$ under a question budget $L _ { q }$ . The synthesis and student-solving prompt templates are provided in Appendix C. For each candidate $q _ { i } ,$ the frozen student generates one response, and the teacher is evaluated on the same response prefixes. These predictions provide the token scores and response-length factor. We compute the batch-diversity factor across the full scoring batch and assign each candidate its reward from Eq. (11).

We optimize the synthesizer with GRPO, normalizing rewards within each group of questions sampled from the same instruction. The resulting group-relative advantage is applied to the generated tokens of the corresponding question. Thus, the optimization updates the policy that generates questions; the student responses are used only to score those questions. Operationally, the question-level reward is attached to the final generated question token as a sequencelevel outcome reward. This reward closes the feedback loop for question generation: it evaluates a candidate through the supervision exposed by a frozen student–teacher pair and uses that signal to improve the synthesizer. Candidate scoring requires neither reference answers nor task-specific correctness rewards, and does not involve training a student on each candidate. It therefore provides a distributional surrogate for question value before downstream distillation.

## 2.5. Synthetic data curation

After synthesizer optimization, we sample N raw outputs from the selected checkpoint. We first filter malformed or incomplete outputs. When a valid question is followed by model self-talk, we retain the question by truncating at the earliest meta-commentary marker. We otherwise reject outputs with repeated n-grams, low lexical diversity, unbalanced mathematical delimiters, answer leakage, missing context, or references to absent figures.

Table 1. Results across six mathematical reasoning benchmarks for three student–teacher pairs. Within each block and metric, shading is relative to Student-Generated OPD (white): blue indicates improvement and red indicates degradation. Positive and negative changes are normalised separately across OPD methods and mapped nonlinearly (exponent 1.5) to muted palettes. Bold indicates the best result among OPD methods, including ties. The baselines compare real-bank OPD, untrained student-generated prompts, KL-filtered selection, and two additional synthetic-data methods.
<table><tr><td rowspan=1 colspan=9>Method                        AIME24  AIME25  AIME26  AMC23  BRUMO25  MATH500  Avg. Recovery (%)Qwen3-1.7B (Non-thinking) Teacher: Qwen3-4B-Non-Thinking-RL-Math</td></tr><tr><td rowspan=1 colspan=2>Student                           10.0</td><td rowspan=1 colspan=2>8.3      8.3</td><td rowspan=1 colspan=1>37.5</td><td rowspan=1 colspan=1>17.9</td><td rowspan=1 colspan=2>72.0     25.7      0.0</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>Teacher†                         41.2</td><td rowspan=1 colspan=1>33.1</td><td rowspan=1 colspan=1>34.4</td><td rowspan=1 colspan=1>57.5</td><td rowspan=1 colspan=1>45.8</td><td rowspan=1 colspan=2>78.3     48.4     100.0</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=1>Student-GeneratedOPD</td><td rowspan=3 colspan=1>27.5</td><td rowspan=3 colspan=1>22.7</td><td rowspan=3 colspan=1>21.3</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td></td></tr><tr><td rowspan=2 colspan=1>61.6</td><td rowspan=2 colspan=1>35.0</td><td rowspan=2 colspan=1>78.4</td><td rowspan=2 colspan=1>41.1      67.9</td><td></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>KL-Filtered OPD</td><td rowspan=1 colspan=1>33.5</td><td rowspan=1 colspan=1>26.7</td><td rowspan=1 colspan=1>23.3</td><td rowspan=1 colspan=1>64.4</td><td rowspan=1 colspan=1>36.7</td><td rowspan=1 colspan=1>80.2</td><td rowspan=1 colspan=1>44.1      81.3</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Vanilla OPD (Li et al., 2026c)</td><td rowspan=1 colspan=1>31.5</td><td rowspan=1 colspan=1>23.5</td><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1>71.1</td><td rowspan=1 colspan=1>31.7</td><td rowspan=1 colspan=1>80.4</td><td rowspan=1 colspan=1>43.0      76.4</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Evol-Instruct (Xu et al., 2024)</td><td rowspan=1 colspan=1>30.0</td><td rowspan=1 colspan=1>21.2</td><td rowspan=1 colspan=1>21.0</td><td rowspan=1 colspan=1>63.9</td><td rowspan=1 colspan=1>37.9</td><td rowspan=1 colspan=1>78.2</td><td rowspan=1 colspan=1>42.1      72.0</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>R-Zero (Huang et al., 2026)</td><td rowspan=1 colspan=1>29.6</td><td rowspan=1 colspan=1>21.7</td><td rowspan=1 colspan=1>20.8</td><td rowspan=1 colspan=1>65.0</td><td rowspan=1 colspan=1>37.5</td><td rowspan=1 colspan=1>79.5</td><td rowspan=1 colspan=1>42.3      73.4</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>A2D</td><td rowspan=1 colspan=1>34.8</td><td rowspan=1 colspan=1>26.0</td><td rowspan=1 colspan=1>26.5</td><td rowspan=1 colspan=1>71.9</td><td rowspan=1 colspan=1>35.4</td><td rowspan=1 colspan=1>82.4</td><td rowspan=1 colspan=1>46.2      90.2</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=8>DeepSeek-R1-Distill-Qwen-1.5BTeacher: JustRL-DeepSeek-1.5B</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Student</td><td rowspan=1 colspan=1>31.0</td><td rowspan=1 colspan=1>26.0</td><td rowspan=1 colspan=1>19.4</td><td rowspan=1 colspan=1>65.8</td><td rowspan=1 colspan=1>29.6</td><td rowspan=1 colspan=1>60.9</td><td rowspan=1 colspan=1>38.8      0.0</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Teacher†</td><td rowspan=1 colspan=1>44.8</td><td rowspan=1 colspan=1>32.9</td><td rowspan=1 colspan=1>32.3</td><td rowspan=1 colspan=1>85.5</td><td rowspan=1 colspan=1>34.6</td><td rowspan=1 colspan=1>79.8</td><td rowspan=1 colspan=1>51.7     100.0</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=1>Student-GeneratedOPD</td><td rowspan=2 colspan=1>37.9</td><td rowspan=2 colspan=1>30.4</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1>26.9</td><td rowspan=1 colspan=1>77.8</td><td rowspan=1 colspan=1>34.0</td><td rowspan=1 colspan=1>82.2</td><td rowspan=1 colspan=1>48.2      73.2</td></tr><tr><td rowspan=1 colspan=1>KL-Filtered OPD</td><td rowspan=1 colspan=1>37.5</td><td rowspan=1 colspan=1>29.2</td><td rowspan=1 colspan=1>29.0</td><td rowspan=1 colspan=1>79.1</td><td rowspan=1 colspan=1>35.2</td><td rowspan=1 colspan=1>81.7</td><td rowspan=1 colspan=1>48.6      76.4</td><td></td></tr><tr><td rowspan=1 colspan=1>Vanilla OPD (Li et al., 2026c)</td><td rowspan=1 colspan=1>45.4</td><td rowspan=1 colspan=1>32.7</td><td rowspan=1 colspan=1>29.2</td><td rowspan=1 colspan=1>78.1</td><td rowspan=1 colspan=1>35.8</td><td rowspan=1 colspan=1>81.5</td><td rowspan=1 colspan=1>50.5      90.7</td><td></td></tr><tr><td rowspan=1 colspan=1>Evol-Instruct (Xu et al., 2024)</td><td rowspan=1 colspan=1>41.5</td><td rowspan=1 colspan=1>32.5</td><td rowspan=1 colspan=1>28.1</td><td rowspan=1 colspan=1>79.5</td><td rowspan=1 colspan=1>35.0</td><td rowspan=1 colspan=1>82.3</td><td rowspan=1 colspan=1>49.8      85.6</td><td></td></tr><tr><td rowspan=1 colspan=1>R-Zero (Huang et al., 2026)</td><td rowspan=1 colspan=1>42.1</td><td rowspan=1 colspan=1>32.9</td><td rowspan=1 colspan=1>28.3</td><td rowspan=1 colspan=1>80.0</td><td rowspan=1 colspan=1>35.2</td><td rowspan=1 colspan=1>81.7</td><td rowspan=1 colspan=1>50.0      87.2</td><td></td></tr><tr><td rowspan=1 colspan=1>A2D</td><td rowspan=1 colspan=1>45.0</td><td rowspan=1 colspan=1>35.8</td><td rowspan=1 colspan=1>29.2</td><td rowspan=1 colspan=1>83.6</td><td rowspan=1 colspan=1>34.0</td><td rowspan=1 colspan=1>82.2</td><td rowspan=1 colspan=1>51.6      99.9</td><td></td></tr><tr><td rowspan=1 colspan=2>DeepSeek-R1-Distill-Qwen-7BTeacher</td><td rowspan=1 colspan=4>: Skywork-OR1-Math-7B</td><td rowspan=1 colspan=2></td><td></td></tr><tr><td rowspan=1 colspan=2>Student                           46.5</td><td rowspan=1 colspan=1>36.7</td><td rowspan=1 colspan=1>44.0</td><td rowspan=1 colspan=1>84.1</td><td rowspan=1 colspan=1>49.8</td><td rowspan=1 colspan=1>87.3</td><td rowspan=1 colspan=1>58.1       0.0</td><td></td></tr><tr><td rowspan=1 colspan=2>Teacher†                         62.7</td><td rowspan=1 colspan=1>42.9</td><td rowspan=1 colspan=1>50.8</td><td rowspan=1 colspan=1>89.7</td><td rowspan=1 colspan=1>59.0</td><td rowspan=1 colspan=1>91.0</td><td rowspan=1 colspan=1>66.0     100.0</td><td></td></tr><tr><td rowspan=2 colspan=2>Student-GeneratedOPD                             53.1</td><td rowspan=2 colspan=2>41.0      45.0</td><td rowspan=2 colspan=1>84.4</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td></td></tr><tr><td rowspan=1 colspan=1>53.3</td><td rowspan=1 colspan=1>88.0</td><td rowspan=1 colspan=1>60.8      34.4</td><td></td></tr><tr><td rowspan=1 colspan=1>KL-Filtered OPD</td><td rowspan=1 colspan=1>53.5</td><td rowspan=1 colspan=1>40.2</td><td rowspan=1 colspan=1>49.0</td><td rowspan=1 colspan=1>87.0</td><td rowspan=1 colspan=1>55.0</td><td rowspan=1 colspan=1>89.3</td><td rowspan=1 colspan=1>62.3      53.7</td><td></td></tr><tr><td rowspan=1 colspan=1>Vanilla OPD (Li et al., 2026c)</td><td rowspan=1 colspan=1>54.8</td><td rowspan=1 colspan=1>41.5</td><td rowspan=1 colspan=1>50.0</td><td rowspan=1 colspan=1>88.8</td><td rowspan=1 colspan=1>52.9</td><td rowspan=1 colspan=1>89.5</td><td rowspan=1 colspan=1>62.9      61.0</td><td></td></tr><tr><td rowspan=1 colspan=1>Evol-Instruct (Xu et al., 2024)</td><td rowspan=1 colspan=1>54.6</td><td rowspan=1 colspan=1>41.5</td><td rowspan=1 colspan=1>49.0</td><td rowspan=1 colspan=1>86.8</td><td rowspan=1 colspan=1>54.0</td><td rowspan=1 colspan=1>90.4</td><td rowspan=1 colspan=1>62.7      58.3</td><td></td></tr><tr><td rowspan=1 colspan=1>R-Zero (Huang et al., 2026)</td><td rowspan=1 colspan=1>55.0</td><td rowspan=1 colspan=1>42.1</td><td rowspan=1 colspan=1>49.6</td><td rowspan=1 colspan=1>87.0</td><td rowspan=1 colspan=1>52.5</td><td rowspan=1 colspan=1>89.0</td><td rowspan=1 colspan=1>62.5      55.8</td><td></td></tr><tr><td rowspan=1 colspan=1>A2D</td><td rowspan=1 colspan=2>57.5      44.8</td><td rowspan=1 colspan=1>52.5</td><td rowspan=1 colspan=1>89.2</td><td rowspan=1 colspan=1>53.3</td><td rowspan=1 colspan=1>89.9</td><td rowspan=1 colspan=1>64.5      81.3</td><td></td></tr></table>

<sup>†</sup> Teachers are reference models. Recovery: $( \mathrm { A v g . _ { m e t h o d } - A v g . _ { s t u d e n t } } ) / ( \mathrm { A v g . _ { t e a c h e r } - A v g . _ { s t u d e n t } } )$ × 100%, computed from unrounded averages and rounded to one decimal place.

We then deduplicate the remaining questions in two passes. Token-set Jaccard matching removes lexical near-duplicates whose similarity reaches a threshold $\tau _ { \mathrm { l e x } } .$ . A subsequent greedy semantic pass uses a frozen Qwen3-Embedding-4B model (Zhang et al., 2025): a question is retained only if the cosine similarity of its embedding to that of every previously retained question is below $\tau _ { \mathrm { s e m } } .$ . This embedding model is used only for post-generation deduplication and does not participate in synthesizer training or reward computation.

The retained questions form a prompt-only synthetic bank. Any placeholder answers stored for file-format compatibility are ignored during OPD; all training responses are sampled from the student.

## 2.6. On-policy distillation

We use the curated question bank for candidate-restricted OPD, starting from the original student parameters. For each question, the current rollout policy $\pi _ { \boldsymbol { S } , \mathrm { o l d } }$ samples $n _ { \mathrm { r o l l } }$ responses under the response budget $L _ { \mathrm { m a x } }$ used for question scoring. The teacher remains frozen and is evaluated on the student-generated prefixes.

At each response position t, let $p _ { S , \mathrm { o l d } } ( v ) = \pi _ { S , \mathrm { o l d } } ( v \vert$ $c _ { t } )$ and $p _ { T } ( v ) ~ = ~ \pi _ { T } ( v ) ~ \vert ~ c _ { t } )$ We construct $\begin{array} { r l } { A _ { t } } & { { } = } \end{array}$ $\mathrm { T o p } { \cdot } K ( p _ { S , \mathrm { o l d } } )$ with the same K as in §2.3 and define the student probabilities normalized over this set:

$$
\bar { p } _ { S , \mathrm { o l d } } ^ { A } ( v ) = \frac { p _ { S , \mathrm { o l d } } ( v ) } { \sum _ { u \in A _ { t } } p _ { S , \mathrm { o l d } } ( u ) } , \qquad v \in A _ { t } .\tag{12}
$$

The superscript A distinguishes this normalization from the union-support normalization used to score questions in §2.3. Each candidate receives the fixed advantage

$$
r _ { t } ( v ) = - \bar { p } _ { S , \mathrm { o l d } } ^ { A } ( v ) \left[ \log p _ { S , \mathrm { o l d } } ( v ) - \log p _ { T } ( v ) \right] ,\tag{v ∈ A<sub>t</sub>.}
$$

(13)

The log-probabilities in brackets are full-vocabulary probabilities; only the weighting factor $\bar { p } _ { S , \mathrm { o l d } } ^ { A }$ is normalized over $A _ { t }$ . The candidate sets and advantages are computed from the rollout policy and held fixed during the subsequent update.

Let $p _ { S , \phi } ( v ) = \pi _ { S , \phi } ( v \mid c _ { t } )$ denote the trainable student’s probability and define

$$
\rho _ { t } ( v ) = \frac { p _ { S , \phi } ( v ) } { p _ { S , \mathrm { o l d } } ( v ) } .\tag{14}
$$

We use the candidate advantages directly in a PPO-style clipped objective (Schulman et al., 2017):

$$
\begin{array} { r l } & { \widehat { r } _ { t } ( v ) = \operatorname* { m i n } \{ \rho _ { t } ( v ) r _ { t } ( v ) , } \\ & { \quad \quad \quad \mathrm { c l i p } ( \rho _ { t } ( v ) , 1 - \varepsilon , 1 + \varepsilon ) r _ { t } ( v ) \} , } \\ & { \quad \quad \quad \mathcal { L } ( \phi ) = - \mathbb { E } \left[ \frac { \sum _ { t } m _ { t } \sum _ { v \in A _ { t } } \widehat { r } _ { t } ( v ) } { \sum _ { t } m _ { t } } \right] , } \end{array}\tag{15}
$$

where $m _ { t }$ masks valid response positions and the expectation is over sampled questions and student responses. Gradients flow through $p _ { { S } , \phi } ,$ while the rollout probabilities, candidate sets, and advantages are treated as constants. We use a single inner epoch per rollout batch, without a value function, return accumulation, group normalization of these advantages, or a KL anchor to a reference model.

At the beginning of an update, the trainable student matches the rollout policy and $\rho _ { t } ( v ) = 1 { \ : }$ ; the ratios can change as optimization proceeds. The advantages provide a studentprobability-weighted, reverse-KL-style learning signal restricted to the student’s top-K candidates. We do not identify this clipped, candidate-restricted objective with exact full-vocabulary reverse-KL minimization.

The question-generation reward and the downstream update serve different roles. The forward-KL discrepancy in Eq.(4)

identifies teacher-preferred candidates that the scoring student underweights, while compatibility measures how much teacher probability mass lies within the candidate set used for training. Their combination scores the supervision a question exposes without reproducing the downstream loss or directly estimating its gradient magnitude. We hold the downstream OPD procedure fixed across prompt-source comparisons to assess the effect of the constructed question distribution.

## 3. Experiments

## 3.1. Models

Mathematical reasoning. We evaluate three student– teacher pairs spanning different model sizes and reasoning modes. The first pair uses Qwen3-1.7B (Team, 2025) in nonthinking mode as the student and Qwen3-4B-Non-Thinking-RL-Math (Yang et al., 2026a) as the teacher. The other two pairs use DeepSeek-R1-Distill-Qwen-1.5B and DeepSeek-R1-Distill-Qwen-7B (Guo et al., 2025), paired with JustRL-DeepSeek-1.5B (He et al., 2025a) and Skywork-OR1-Math-7B (He et al., 2025b), respectively. These pairs cover both non-thinking and reasoning models, including settings where the student and teacher have the same parameter count but differ in post-training.

Code generation. We evaluate Qwen3-4B in non-thinking mode with two teachers: Qwen3-4B-Instruct-2507 and Qwen3-30B-A3B-Instruct-2507 (Team, 2025). This comparison examines distillation from different teachers while keeping the initial student fixed. We additionally evaluate Qwen2.5-Coder-3B-Instruct with Qwen2.5-Coder-32B-Instruct (Hui et al., 2024), extending the evaluation to a code-specialized model family.

Agentic tasks and model roles. For agentic tasks, we use Qwen3-4B as the student and Qwen3-4B-Thinking-2507 as the teacher, both configured in non-thinking mode. In our default setup, the question synthesizer is initialized as an independent copy of the corresponding student. During synthesizer optimization, the scoring student and teacher remain frozen; downstream OPD starts from the original student parameters and updates only the student. We separately examine teacher-initialized synthesizers in the synthesizerinitialization ablation. Exact checkpoint identifiers and inference configurations are provided in the appendix.

Auxiliary models. For agentic tasks, we use GPT-4o-mini (OpenAI, 2024) as the user simulator and judge. For semantic deduplication of the generated question bank, we use a frozen Qwen3-Embedding-4B model (Zhang et al., 2025) only after synthesizer training. It is not used to compute training-time diversity rewards or optimize the synthesizer.

Table 2. Results across six code benchmarks for three student–teacher pairs. Within each block and metric, shading is relative to Student-Generated OPD (white): blue indicates improvement and red indicates degradation. Positive and negative changes are normalised separately across OPD methods and mapped nonlinearly (exponent 1.5) to muted palettes. Bold indicates the best result among OPD methods, including ties. The baselines compare real-bank OPD, untrained student-generated prompts, KL-filtered selection, and two additional code-generation methods.
<table><tr><td>Method</td><td>LCB</td><td>MBPP+</td><td>APPS-Intro</td><td>APPS-Interv.</td><td>APPS-Comp.</td><td>CodeContests</td><td>Avg.</td><td>Recovery (%)</td></tr><tr><td colspan="9">Qwen3-4B (Non-thinking) Teacher: Qwen3-4B-Instruct-2507</td></tr><tr><td>Student</td><td>21.6</td><td>68.8</td><td>46.4</td><td>19.1</td><td>8.8</td><td>7.3</td><td>28.7</td><td>0.0</td></tr><tr><td>Teacher†</td><td>33.7</td><td>72.0</td><td>65.6</td><td>41.0</td><td>19.7</td><td>18.8</td><td>41.8</td><td>100.0</td></tr><tr><td>Student-Generated OPD</td><td>30.2</td><td>67.2</td><td>55.8</td><td>27.8</td><td>10.4</td><td>9.7</td><td>33.5</td><td>36.9</td></tr><tr><td>KL-Filtered OPD</td><td>31.1</td><td>67.7</td><td>58.2</td><td>30.0</td><td>11.5</td><td>11.5</td><td>35.0</td><td>48.2</td></tr><tr><td>Vanilla OPD (Li et al., 2026c)</td><td>31.0</td><td>68.5</td><td>57.1</td><td>27.6</td><td>10.7</td><td>12.1</td><td>34.5</td><td>44.4</td></tr><tr><td>Evol-Instruct (Xu et al., 2024)</td><td>30.3</td><td>68.5</td><td>56.8</td><td>27.8</td><td>10.0</td><td>10.3</td><td>34.0</td><td>40.2</td></tr><tr><td>SQLM (Chen et al., 2025)</td><td>31.3</td><td>69.8</td><td>57.7</td><td>28.9</td><td>10.9</td><td>10.9</td><td>34.9</td><td>47.6</td></tr><tr><td>A2D</td><td>35.9</td><td>68.5</td><td>64.0</td><td>36.0</td><td>15.9</td><td>17.0</td><td>39.6</td><td>82.9</td></tr><tr><td colspan="9">Qwen3-4B (Non-thinking) Teacher: Qwen3-30B-A3B-Instruct-2507</td></tr><tr><td>Student</td><td>21.6</td><td>68.8</td><td>46.4</td><td>19.1</td><td>8.8</td><td>7.3</td><td>28.7</td><td>0.0</td></tr><tr><td>Teacher†</td><td>42.3</td><td>75.9</td><td>73.4</td><td>51.2</td><td>30.4</td><td>26.7</td><td>50.0</td><td>100.0</td></tr><tr><td>Student-Generated OPD</td><td>28.0</td><td>66.9</td><td>50.5</td><td>20.0</td><td>5.5</td><td>6.1</td><td>29.5</td><td>3.9</td></tr><tr><td>KL-Filtered OPD</td><td>29.4</td><td>66.9</td><td>53.4</td><td>25.1</td><td>8.4</td><td>9.7</td><td>32.2</td><td>16.3</td></tr><tr><td>Vanilla OPD (Li et al., 2026c)</td><td>30.6</td><td>68.8</td><td>51.3</td><td>21.3</td><td>4.2</td><td>4.8</td><td>30.2</td><td>7.0</td></tr><tr><td>Evol-Instruct (Xu et al., 2024)</td><td>28.8</td><td>67.1</td><td>51.9</td><td>22.5</td><td>5.0</td><td>5.0</td><td>30.1</td><td>6.5</td></tr><tr><td>SQLM (Chen et al., 2025)</td><td>29.9</td><td>67.5</td><td>53.5</td><td>23.7</td><td>5.7</td><td>6.3</td><td>31.1</td><td>11.4</td></tr><tr><td>A2D</td><td>31.7</td><td>67.5</td><td>58.3</td><td>31.6</td><td>11.0</td><td>13.9</td><td>35.7</td><td>32.8</td></tr><tr><td colspan="9">Qwen2.5-Coder-3B-Instruct Teacher: Qwen2.5-Coder-32B-Instruct</td></tr><tr><td>Student Teacher†</td><td>13.7</td><td>64.0</td><td>28.7</td><td>6.6 18.8</td><td>3.0</td><td>2.4 5.5</td><td>19.7</td><td>0.0</td></tr><tr><td>Student-Generated</td><td>27.9</td><td>76.7</td><td>51.8</td><td></td><td>11.1</td><td></td><td>32.0</td><td>100.0</td></tr><tr><td>OPD</td><td>14.5</td><td>64.3</td><td>29.0</td><td>6.3</td><td>2.8</td><td>1.8</td><td>19.8</td><td>0.4</td></tr><tr><td>KL-Filtered OPD</td><td>15.3</td><td>64.8</td><td>30.0</td><td>6.8</td><td>3.4</td><td>2.4</td><td>20.5</td><td>5.9</td></tr><tr><td>Vanilla OPD (Li et al., 2026c)</td><td>16.2</td><td>64.0</td><td>30.7</td><td>6.5</td><td>2.9</td><td>0.6</td><td>20.2</td><td>3.4</td></tr><tr><td>Evol-Instruct (Xu et al., 2024) SQLM (Chen et al., 2025)</td><td>15.5</td><td>64.6</td><td>29.8</td><td>6.9</td><td>3.2</td><td>1.6 1.8</td><td>20.3 20.6</td><td>4.4 6.7</td></tr><tr><td></td><td>15.8 17.0</td><td>64.8 65.1</td><td>30.4</td><td>7.1 7.6</td><td>3.4 3.7</td><td>2.4</td><td>21.2</td><td>12.3</td></tr><tr><td>A2D</td><td></td><td></td><td>31.6</td><td></td><td></td><td></td><td></td><td></td></tr></table>

<sup>†</sup> Teachers are reference models. Recovery: $( \mathrm { A v g . _ { m e t h o d } - A v g . _ { s t u d e n t } } ) / ( \mathrm { A v g . _ { t e a c h e r } - A v g . _ { s t u d e n t } } ) \times 1 0 0 \%$ , computed from unrounded averages and rounded to one decimal place.

These auxiliary models remain fixed throughout our experiments.

## 3.2. Baselines

We compare A2D with the following baselines and reference models: (1) Student and Teacher: the initial student before distillation and its corresponding teacher, which establish the starting performance and the reference for teacher-gap recovery; (2) Vanilla OPD (Li et al., 2026c): on-policy distillation using the original question bank for each domain, DAPO-17k-Math (Yu et al., 2025) for mathematical reasoning, OpenCodeReasoning (Ahmad et al., 2025) for code generation, and the official τ<sup>2</sup>-Bench training set (Barrès et al., 2025) for agentic tasks—where the student generates responses and the teacher provides token-level supervision on student-generated prefixes, serving as the standard realdata reference; (3) Student-Generated OPD: OPD using questions sampled directly from the initial student without synthesizer optimization, evaluating whether explicitly training the question-generation policy provides benefits beyond the student’s existing generation capability; (4) KL-Filtered OPD: OPD using the 20% of real questions with the highest teacher–student KL disagreement, testing whether retaining high-disagreement questions alone provides effective distillation data;

(5) Evol-Instruct (Xu et al., 2024): an instruction-evolution method that iteratively rewrites seed instructions to increase complexity and diversity, which we adapt to construct mathematical and programming questions for OPD; (6) R-Zero (Huang et al., 2026): a self-evolution framework in which a Challenger proposes tasks and a Solver learns to solve them, with Challenger optimization guided by feedback about the Solver’s capabilities, whose task-generation procedure we use to construct mathematical questions for OPD; (7) SQLM (Chen et al., 2025): an asymmetric selfplay framework with a topic-conditioned proposer and a solver, whose coding formulation uses generated unit tests and solver test performance to encourage non-trivial but solvable tasks, providing programming questions for downstream OPD. Evol-Instruct is evaluated in both domains, while R-Zero and SQLM are included in the mathematical and code-generation comparisons, respectively. For these question-bank comparisons, the generated or selected questions are used as prompts for the same downstream OPD procedure. We reproduce Evol-Instruct, R-Zero, and SQLM using their official implementations.

![](images/8f299e96b35a9efb28984650626989f79e0da6ae3e4e353b4de72c0513d577f6.jpg)  
Figure 3. Downstream performance during training for student- and teacher-initialized question synthesizers, with and without pair-specific RL, using Qwen3-4B (non-thinking) as the student and Qwen3-4B-Instruct-2507 as the teacher, evaluated on LCB, APPS-Intro, and APPS-Interview. Dashed lines show the student and teacher reference scores. RL improves both initializations; student initialization with RL is competitive across benchmarks and exceeds the teacher reference on LCB, while remaining below it on APPS-Intro and APPS-Interview.

Table 3. Results across the three $\tau ^ { 2 } .$ -Bench domains. Within each metric, shading is relative to Student-Generated OPD (white): blue indicates improvement and red indicates degradation. Positive and negative changes are normalised separately across OPD methods and mapped nonlinearly (exponent 1.5) to muted palettes. Bold indicates the best result among OPD methods, including ties. The three baselines isolate the effects of synthesizer training and learning-value-based question generation.
<table><tr><td>Method</td><td>Airline</td><td>Retail</td><td>Telecom</td><td>Avg.</td><td>Recovery (%)</td></tr><tr><td>Qwen3-4B (Non-thinking)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Teacher: Qwen3-4B-Thinking-2507 (Non-thinking) Student</td><td>15.0</td><td>17.5</td><td>10.6</td><td>14.4</td><td>0.0</td></tr><tr><td>Teacher†</td><td>47.5</td><td>48.1</td><td>34.4</td><td>43.3</td><td>100.0</td></tr><tr><td>Student-Generated</td><td>42.5</td><td>42.5</td><td>26.9</td><td>37.3</td><td>79.2</td></tr><tr><td>OPD KL-Filtered OPD</td><td>45.0</td><td>48.8</td><td>28.8</td><td>40.9</td><td>91.7</td></tr><tr><td>Vanilla OPD</td><td>45.0</td><td>46.9</td><td>29.4</td><td>40.4</td><td>90.0</td></tr><tr><td>A2D</td><td>46.3</td><td>49.4</td><td>30.0</td><td>41.9</td><td>95.2</td></tr></table>

<sup>†</sup> The teacher is a reference model. Recovery: $\mathrm { \Gamma ( A v g . _ { m e t h o d } ) ^ { \prime } }$ $\mathrm { A v g . _ { s t u d e n t } ) / ( A v g . _ { t e a c h e r } - A v g . _ { s t u d e n t } ) }$ × 100%, computed from unrounded averages and rounded to one decimal place.

![](images/e6fc340eb66220d62149caf59cf0a3a4adf157021098c919facdb64c8464a405.jpg)  
Figure 4. Reward-component ablation on AIME24, AIME25, and AIME26. Numbers above bars indicate accuracy (%). Removing compatibility, the training-time diversity term, or the explicit length penalty reduces accuracy on all three benchmarks.

## 3.3. Datasets and Evaluation Metrics

Mathematical reasoning. We evaluate on six benchmarks: AIME24 (Mathematical Association of America, 2024), AIME25 (Zhang & Math-AI, 2025),

Table 4. Iterative training on code pair A, with student Qwen3-4B (non-thinking) and teacher Qwen3-30B-A3B-Instruct-2507. Avg. is the mean of LiveCodeBench v5\_v6, APPS-Intro and APPS-Interview. Recovery = (Avg − student)/(teacher − student). Dashes indicate entries that are not applicable.
<table><tr><td>Training variant</td><td>Avg.</td><td>Gain over Recovery round 1</td><td>(%)</td></tr><tr><td>Student</td><td>29.0</td><td>-</td><td>0.0</td></tr><tr><td>Teacher</td><td>55.6</td><td>-</td><td>100.0</td></tr><tr><td>A2D: round 1</td><td>40.5</td><td>0.0</td><td>43.3</td></tr><tr><td>A2D: round 2</td><td>40.8</td><td>+0.3</td><td>44.3</td></tr><tr><td>A2D: final round  $( R = 3 )$ </td><td>41.0</td><td>+0.5</td><td>44.8</td></tr></table>

AIME26 (Dekoninck et al., 2026), AMC23 (Mathematical Association of America, 2023), BRUMO25 (Dekoninck et al., 2026), and MATH500 (Hendrycks et al., 2021b; Lightman et al., 2024). Each AIME benchmark and BRUMO25 contains 30 problems, AMC23 contains 40, and MATH500 contains 500. These benchmarks cover competition-level mathematical reasoning across algebra, geometry, number theory, and combinatorics. We evaluate final-answer correctness using rule-based normalization and symbolic equivalence checking.

Code generation. We evaluate on six benchmarks: Live-CodeBench (Jain et al., 2025), MBPP+ (Liu et al., 2023; Austin et al., 2021), the introductory, interview, and competition subsets of APPS (Hendrycks et al., 2021a), and CodeContests (Li et al., 2022). We use the LiveCodeBench v5–v6 subset with 342 problems and MBPP+ v0.2.0 with 378 problems. For APPS, we use all 1,000 introductory and 1,000 competition test problems, together with a fixed sample of 1,000 interview test problems. CodeContests contains 165 test problems. A generated program is considered correct only if it passes every test case used by the evaluator within the time limit. For MBPP+, we apply entry-point name normalization when the required function is absent and the generated program defines exactly one top-level function.

Agentic tasks. We additionally evaluate on the Airline, Retail, and Telecom domains of τ<sup>2</sup>-Bench (Barrès et al., 2025), using the official test splits containing 20, 40, and 40 tasks, respectively. These tasks require an agent to interact with a simulated user and invoke tools to complete domainspecific objectives. We run four trials per task and report pass<sup>1</sup>, the average single-trial task success rate under the benchmark evaluator.

Evaluation metrics. Following Li et al. (2026c), for mathematical reasoning and code generation, we report avg@k, defined as the fraction of correct responses among k generations per problem, averaged over all problems:

$$
\arg @ k = \frac { 1 0 0 } { N k } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { k } \mathbb { I } \Big [ y _ { i } ^ { ( j ) } \mathrm { ~ i s ~ c o r r e c t } \Big ] ,\tag{16}
$$

where N is the number of problems. This metric measures average response accuracy rather than whether at least one of the k responses is correct. We use $k = 1 6$ for AIME24/25/26, AMC23, and BRUMO25; k = 4 for MATH500 and LiveCodeBench; and k = 1 for the remaining code benchmarks. MBPP+ uses greedy decoding. Within each domain, Avg. is the unweighted mean of the benchmark scores. To measure how much of the initial student–teacher performance gap is recovered, we also report

$$
{ \mathrm { R e c o v e r y } } = 1 0 0 \times { \frac { { \mathrm { A v g } } { \cdot } { \mathrm { m e t h o d } } - { \mathrm { A v g } } { \cdot } _ { \mathrm { s t u d e n t } } } { { \mathrm { A v g } } { \cdot } _ { \mathrm { t e a c h e r } } - { \mathrm { A v g } } { \cdot } _ { \mathrm { s t u d e n t } } } } .\tag{17}
$$

A recovery of 100% indicates that the distilled student matches the teacher’s average performance, while values above 100% indicate that it exceeds the teacher. Recovery is computed from unrounded averages. Detailed decoding settings and evaluation procedures are provided in Appendix A.3.

All reported results are means over three independent runs with different random seeds. Evaluation settings, including decoding parameters and answer verification, are described in Appendix A.3.

## 3.4. Main Results

We evaluate A2D across six mathematical reasoning and six code-generation benchmarks, with three student–teacher pairs in each domain. Tables 1 and 2 report the main results. Across all six settings, A2D achieves the highest average performance among the compared distillation methods, demonstrating the effectiveness of optimizing synthetic questions for the target student–teacher pair.

Mathematical reasoning. As shown in Table 1, A2D achieves average scores of 46.2, 51.6, and 64.5 for Qwen3- 1.7B, R1-Distill-1.5B, and R1-Distill-7B, respectively, compared with 43.0, 50.5, and 62.9 for Vanilla OPD. These results show that optimized synthetic questions can outperform the original real question bank across both nonthinking and reasoning students. The corresponding teachergap recovery rates reach 90.2%, 99.9%, and 81.3%; notably, R1-Distill-1.5B nearly matches its teacher’s average performance. Relative to Student-Generated OPD, our method improves the reported average scores by 5.1, 3.4, and 3.7 percentage points, indicating that optimizing the synthesizer provides additional value beyond sampling questions directly from the student.

![](images/d1451598788f75ae582affc0a2594d7a6bedd3a3a82015fb75b1b375f255b034.jpg)  
Figure 5. Average accuracy versus cumulative training tokens for (a) R1-Distill-1.5B on JustRL and (b) R1-Distill-7B on Skywork. Gray squares show Vanilla checkpoints and the dashed line its best-so-far accuracy; blue circles show A2D checkpoints. Shaded regions indicate intervals where A2D is above (gain) or below (loss) Vanilla’s best-so-far reference. Average accuracy is computed over AIME24, AIME25, and AMC23, and therefore differs from the six-benchmark average in Table 1.

A2D also outperforms KL-Filtered OPD, Evol-Instruct, and R-Zero on the average score in every mathematical setting. The comparison with KL-Filtered OPD supports the value of optimizing the prompt distribution using support-aware supervision beyond selecting real questions with large teacher– student disagreement. At the benchmark level, representative gains over Vanilla OPD include an increase from 20.0 to 26.5 on AIME26 for Qwen3-1.7B and from 78.1 to 83.6 on AMC23 for R1-Distill-1.5B. The improvements are not uniform across benchmarks: other methods remain stronger on some BRUMO25 and MATH500 evaluations. Nevertheless, the consistent average gains indicate that our optimized question banks provide effective supervision across the evaluated mathematical tasks.

Code generation. Table 2 shows that A2D also improves code-generation performance across all three student– teacher pairs. Its average scores reach 39.6, 35.7, and 21.2, compared with 34.5, 30.2, and 20.2 for Vanilla OPD, respectively. Our method also exceeds Student-Generated OPD, KL-Filtered OPD, Evol-Instruct, and SQLM in average performance, and achieves the best or tied-best score among the compared distillation methods in 16 of the 18 benchmark–pair combinations. The gains extend across multiple coding benchmarks. For example, with Qwen3- 30B-A3B as the teacher, APPS-Interview improves from 21.3 to 31.6 and CodeContests from 4.8 to 13.9 relative to Vanilla OPD. These results extend the benefits of optimizing elicited supervision beyond mathematical reasoning to code generation.

The magnitude of improvement nevertheless varies substantially across student–teacher pairs. Teacher-gap recovery reaches 82.9% for Qwen3-4B distilled from the 4B teacher, but remains at 32.8% with the stronger 30B-A3B teacher and 12.3% for the Qwen2.5-Coder-3B/32B pair. Moreover, for the same Qwen3-4B student, the stronger teacher yields a lower distilled average score than the 4B teacher under our setup. This comparison indicates that teacher benchmark performance alone does not determine downstream distillation effectiveness. Optimizing the question distribution consistently improves the evaluated outcomes, while substantial headroom remains in some student–teacher settings.

## 3.5. Ablation: Can a stronger model replace synthetic data optimization?

A natural alternative to training a synthesizer for each student–teacher pair is to ask a stronger off-the-shelf model— in our setting, the teacher itself—to generate the questions. To separate the effect of synthesizer initialization from that of pair-specific optimization, we compare student- and teacher-initialized synthesizers, each with and without RL, while holding the downstream distillation setup fixed. Figure 3 reports performance over training for a representative code student–teacher pair; the dashed lines mark the student and teacher reference scores.

Without RL, teacher initialization improves over student initialization on the two APPS subsets, but not on LCB, and neither unoptimized synthesizer matches the student-initialized RL variant across the three benchmarks. Applying RL improves the trajectories for both initializations, indicating that the benefit is not explained by simply prompting a stronger model to write questions. The student-initialized RL variant reaches roughly 35% on LCB, above the teacher reference of 33.7%, and about 62% on APPS-Intro. On APPS-Interview it reaches about 36%; both APPS scores remain below their teacher references (65.6% and 41.0%, respectively). The teacher-initialized RL variant is competitive—and slightly higher at some APPS-Intro checkpoints—but does not consistently dominate the student-initialized variant. Thus, synthesizer strength alone is not a substitute for optimizing questions against the target pair, while the choice of initialization can still affect performance by benchmark. These curves are descriptive: they show checkpoint trends, not <sup>ffi</sup>uncertainty across independent runs.

![](images/b7b943fe45f9ad4cc6f3d8e066a6035259e3ff31893e227a3962a935c7a0517b.jpg)  
Figure 6. Simple prompts can elicit dense distillation signals. Comparison of (a) a real question from DAPO-17k-Math and (b) a synthetic question generated by our method. Darker token backgrounds indicate higher support-aware scores s<sub>t</sub>; the strips summarize the complete rollouts, with boxes marking the displayed excerpts. Although both student answers are incorrect, the apparently simple synthetic question elicits a higher mean s<sub>t</sub> (0.172 versus 0.117; approximately 1.5×) with a shorter rollout (1,241 versus 2,002 tokens), illustrating that surface complexity alone does not characterize the density of elicited supervision.

## 3.6. Ablation: which reward components matter?

We ablate compatibility, the training-time diversity term, and the explicit length penalty, and evaluate the resulting question banks by downstream best accuracy on AIME24, AIME25, and AIME26 (Figure 4). The full reward performs best on all three benchmarks, reaching 45.0%, 35.8%, and 29.2%, respectively. Removing any one of the three components lowers accuracy on every benchmark, with drops ranging from 2.1 to 6.6 percentage points.

The size of each drop depends on the benchmark. On AIME25, removing compatibility, diversity, or the length penalty reduces accuracy by 6.2, 6.0, and 6.6 points, respectively. On AIME24, the largest reduction is 4.2 points without diversity; on AIME26, it is 3.4 points without the length penalty. These results suggest that all three components contribute to the performance of the complete reward, <sup>ffi</sup>while their relative importance varies across benchmarks. The figure reports downstream accuracy rather than synthesizer reward, directly assessing the quality of the prompts for distillation.

## 3.7. Applicability to agentic tasks

To test whether A2D applies beyond mathematical reasoning and code generation, we evaluate it on the three agentic task domains in τ<sup>2</sup>-Bench: Airline, Retail, and Telecom. Table 3 reports results for Qwen3-4B (non-thinking).

Our method reaches 46.3%, 49.4%, and 30.0% in the three domains, respectively, for an average of 41.9%. This exceeds Student-Generated OPD by 3.8, 6.9, and 3.1 percentage points, respectively (4.6 points on average), and raises teacher-gap recovery from 79.2% to 95.2%. It also outperforms Vanilla OPD (40.4% average) and KL-Filtered OPD (40.9%) on the average score. For this student–teacher pair, these results support the applicability of pair-specific question optimization to agentic task domains as well as to reasoning and code generation. The average remains 1.4 percentage points below the teacher reference (43.3%).

## 3.8. Cross-Domain Transfer between Math and Code

Our distribution-elicitation perspective suggests that the benefits of OPD need not be confined to the domain of its training questions. We examine this hypothesis in two directions: code-to-math transfer with Qwen3-4B (non-thinking) distilled from Qwen3-4B-Instruct-2507, and math-to-code transfer with R1-Distill-7B distilled from Skywork-OR1- Math-7B. In each setting, we directly evaluate the student after source-domain OPD, without additional target-domain training, comparing our optimized synthetic questions with the real source-domain questions used by Vanilla OPD. We report gains over the initial student and compute an unweighted average across the target benchmarks.

Code-to-math transfer. We evaluate code-trained students on AIME24, AIME26, AMC23, and BRUMO25. As shown in Figure 7(a), both methods improve performance on all four mathematics benchmarks. A2D raises average accuracy from 31.38% to 38.60%, a gain of 7.23 percentage points, compared with 5.33 points for Vanilla OPD. Our method consistently yields larger improvements across the four benchmarks, with gains of 11.0, 5.2, 8.5, and 4.2 points, respectively. These results show that supervision elicited by code questions can also benefit mathematical reasoning.

Math-to-code transfer. We evaluate mathematics-trained students on LiveCodeBench, APPS-Intro, and APPS-Interview. Figure 7(b) shows that A2D improves performance by 2.7, 4.8, and 8.7 percentage points, respectively, compared with 1.9, 3.8, and 6.7 points for Vanilla OPD. Average accuracy increases from 47.00% to 52.40% with our method, exceeding Vanilla OPD’s 51.13% and approaching the teacher’s 52.70%. The student also surpasses the teacher on APPS-Intro (70.1% versus 68.7%), while remaining below it on the other two benchmarks.

Together, these results demonstrate cross-domain gains in both evaluated directions, with our optimized synthetic questions outperforming real source-domain questions on every reported target benchmark. This finding is consistent with questions serving as carriers of distributional supervision whose benefits extend beyond the training domain. Although these experiments do not isolate the mechanism underlying transfer, they show that optimizing the supervision elicited from a particular student–teacher pair can improve performance beyond source-domain tasks.

## 3.9. Training-token efficiency and checkpoint gains

We compare downstream accuracy against cumulative training tokens for R1-Distill-1.5B on JustRL and R1-Distill-7B on Skywork (Figure 5). The plot shows the Vanilla and A2D checkpoint trajectories, the Vanilla best-so-far reference, and the best checkpoint for each method. A2D reaches a best accuracy of 54.80% on R1-Distill-1.5B, compared with 52.27% for Vanilla, a gain of 2.53 percentage points. On R1-Distill-7B, A2D reaches 63.80% versus 61.70% for Vanilla, a gain of 2.10 points. The best-checkpoint token counts are similar for the 1.5B model, while A2D reaches its best checkpoint with substantially fewer training tokens for the 7B model.

These trajectories indicate that the accuracy gains are achieved without a substantial increase in downstream training-token use, particularly for R1-Distill-7B. They report downstream accuracy and token use only; they do not establish lower end-to-end cost, which also depends on the upstream cost of generating and scoring synthetic questions.

## 3.10. Iterative Training

Our main framework optimizes synthetic questions for a fixed student–teacher pair before performing OPD. We further examine whether repeating this process as the student improves yields additional benefits. We conduct three rounds on the code-generation pair consisting of a Qwen3- 4B student and a Qwen3-30B-A3B teacher. After each round, we update the student used for question optimization, construct a new synthetic question bank, and continue OPD while keeping the teacher fixed. We evaluate on Live-CodeBench, APPS-Intro, and APPS-Interview and report the highest observed average accuracy across the evaluated checkpoints within each round. This comparison provides an optimistic assessment of the benefits of additional rounds, rather than a validation-selected final result.

The first round raises average accuracy from 29.03% to 40.53%, an improvement of 11.50 percentage points. In contrast, the best observed averages in the second and third rounds reach only 40.82% and 40.97%, respectively. Thus, two additional rounds yield a cumulative gain of just 0.44 points over the first round, despite the additional questiongeneration and distillation effort. Even under this bestcheckpoint comparison, iterative training provides only marginal improvements in the evaluated setting.

One possible explanation is that the first round already captures much of the readily accessible supervision for this student–teacher pair. As the student becomes better aligned with the teacher, the synthesizer may find it increasingly difficult to identify questions that expose substantial disagreement while remaining within regions where the teacher provides useful guidance. In terms of our learning-value objective, maintaining both disagreement and compatibility may therefore become harder across rounds. This interpretation remains a hypothesis, as downstream scores alone do not establish how these signals evolve during training. Nevertheless, the observed diminishing returns suggest that repeatedly adapting the question distribution is not sufficient to sustain the large gains achieved in the first round, supporting single-round optimization as a practical choice in this setting.

![](images/4d0e47a471272e6d720ab7e2f87d71cb1e73a0f1f0f98e1a4502078e0d7ff639.jpg)  
Figure 7. Cross-domain transfer between math and code. Gains over the initial student after source-domain OPD, without additiona target-domain training. (a) Code-to-math transfer with Qwen3-4B (non-thinking) distilled from Qwen3-4B-Instruct-2507. (b) Math-tocode transfer with R1-Distill-7B distilled from Skywork-OR1-Math-7B. A2D uses optimized synthetic source-domain questions, while Vanilla OPD uses real source-domain questions. Bars report gains in percentage points; shaded groups show unweighted averages across the displayed benchmarks. Both methods improve all seven evaluated pair–benchmark combinations, and A2D consistently yields larger gains.

![](images/83429c6e5823a99696e7ead3f91205af0d8a2d0b932398abbf4222ce4fab2214.jpg)  
Ours (optimized synthetic questions)

## 3.11. Case study: where does the teacher provide useful signal?

To examine what the learning-value reward captures at the token level, Figure 6 contrasts a real geometry question from DAPO-17k-Math with a short combinatorics question posed by our student-initialized synthesizer, both scored with the same student–teacher pair. Each panel reports the apparent difficulty of the question, the mean teacher– student discrepancy, and the mean support-aware score $s _ { t }$ (§2.3) over one student rollout. Below these statistics, an excerpt of the rollout is shaded by $s _ { t }$ and discrepancy, and two strips summarize both quantities over the full response, with a box marking the excerpt. The student answers both questions incorrectly.

Signal density. Despite its simpler surface form, the synthetic question elicits denser supervision. Its rollout contains 1,241 tokens, compared with 2,002 for the real question, while achieving approximately 1.5× the mean $s _ { t }$ (0.172 versus 0.117) and 1.7× the mean discrepancy (0.513 versus 0.307). Thus, the synthetic question produces a 38% shorter trajectory with a higher average support-aware score. Its cumulative score remains comparable to that of the real question $\textstyle ( \sum _ { t } s _ { t } \ = \ 2 1 4$ versus 234), retaining approximately 91% of the scored signal with only 62% of the response tokens.

Where the signal arises. The excerpts indicate where this difference comes from. In the geometry rollout, the displayed segment consists of routine coordinate algebra: the teacher largely agrees with the student token by token, and most positions receive low scores. In the combinatorics rollout, high-scoring positions cluster at the decisions that structure the solution—treating B and S as a single entity and then counting the resulting items—including the step where the student counts three items although there are four (U, E, the B–S block, and C). In both rollouts, $s _ { t }$ is almost perfectly rank-correlated with the discrepancy (Spearman $\rho = 0 . 9 9 4 \ \mathrm { a n d } \ 0 . 9 9 7 )$ . Thus, the combined score largely preserves the ordering of positions by discrepancy in these examples. This rank correlation alone does not determine the magnitude of the compatibility factor.

This example illustrates that surface complexity alone is an incomplete indicator of a question’s value for OPD: a concise, apparently straightforward problem can elicit dense teacher–student disagreement with substantial supportaware signal. The observed advantage concerns signal density; the cumulative support-aware score is slightly lower for the synthetic question. These observations provide a qualitative illustration of our distribution-elicitation perspective, complementing the downstream performance comparisons.

Aggregate signal analysis. To complement the individual examples, we sample 1,000 questions each from DAPO-

![](images/10679de80827f6d9ec1bed6b1410e6aa27484fc984af687a6be61c70f786587e.jpg)

(b) Signal throughout the response  
![](images/a6dd01759b6361cfd9625b0a5bdbd55f4e60885b0b35ffed548f6350013371ec.jpg)  
Figure 8. Optimized questions elicit more signal with fewer tokens. We compare student rollouts on 1,000 questions each from DAPO-17k-Math and our synthetic bank. (a) Mean response length and cumulative support-aware score $\sum _ { t } s _ { t } ,$ , normalized to the real-question baseline (100%); labels above bars show raw means. Synthetic questions yield 20% shorter responses and 28% more cumulative signal. (b) Mean token-level score $s _ { t }$ across relative response-position bins. Synthetic questions maintain higher scores throughout the response, indicating that the signal-density advantage extends across the trajectory.

17k-Math and our synthetic question bank and analyze the support-aware signal along student rollouts. As shown in Figure 8(a), synthetic questions elicit shorter responses on average (1,020.7 versus 1,280.2 tokens, a 20% reduction), yet yield a higher mean cumulative score $\textstyle \sum _ { t } s _ { t }$ (181.7 versus 141.7, a 28% increase). Thus, across these samples, the higher signal density translates into more total scored signal despite fewer response tokens. Figure 8(b) further shows that the mean $s _ { t }$ remains higher for synthetic questions across all displayed relative-position bins, indicating that the advantage extends throughout the response rather than being confined to a narrow segment. Together with the concrete examples, these aggregate results support our distribution-elicitation perspective: optimized questions can expose denser and more abundant support-aware supervision without relying on longer student trajectories.

## 3.12. Is Elicited Divergence a Learning Signal?

Our reward rests on the premise that questions eliciting larger teacher–student divergence along student trajectories provide more useful supervision. On natural questions, the support-aware score closely tracks its divergence component (Spearman $\rho ( \bar { s } , \bar { \delta } ) = 0 . 9 0 )$ ). We examine whether these signals predict actual learning gains through two experiments on the Qwen3-1.7B ← Qwen3-4B-RL-Math pair.

Per-question learning value. For each of 1,500 held-out DAPO questions, we apply a single OPD update starting from the same base student and measure the resulting decrease in teacher–student KL on a fixed probe set. As shown in Figure 9 (left), mean divergence $\bar { \delta }$ strongly correlates with this learning value $( \rho = 0 . 7 2 7$ , 95% CI [0.695, 0.755]). The association remains positive after controlling for response length, student pass rate, and student entropy (partial $\rho = 0 . 3 8 6$ , 95% CI [0.329, 0.443]). The full score s¯ exhibits similar associations, with raw and partial correlations of 0.750 and 0.399, respectively. Consistently, probe KL decrease rises monotonically across quintiles ranked by s¯, with the highest quintile yielding 2.15× the decrease of the lowest (middle).

Downstream training utility. We further train students on five score-ranked question banks under an identical token budget of 24.4M tokens, using three seeds per bank. Figure 9 (right) shows that higher-scoring banks generally yield better final accuracy across six benchmarks. Average accuracy rises from 41.06% in R1 to 43.49% in R4, while R5 reaches 43.11%, exceeding R1 by 2.05 percentage points. Thus, the overall trend favors higher-scoring banks, although performance is not strictly monotonic. Additional high-versus-low divergence comparisons retain a positive advantage after balancing compatibility (Appendix A.5). Together, these results support elicited divergence as a useful indicator of distillation value, with evidence from both single-update gains and downstream training.

Both experiments use natural DAPO-17k-Math questions, without synthesizer outputs, to evaluate whether the proposed signals predict learning value independently of synthesizer optimization. The one-step analysis measures changes on a fixed, disjoint probe set, while downstream comparisons use a common training-token budget. Appendix A.5 provides the data splits, update and evaluation protocols, controlled designs, and statistical procedures.

## 4. Related Work

## 4.1. On-Policy Distillation and Selective Supervision

On-policy distillation supervises student-generated trajectories using teacher predictions at the same visited states. GKD addresses the distribution mismatch of fixedresponse distillation by training on student-generated sequences (Agarwal et al., 2024), while MiniLLM develops an on-policy optimization approach based on reverse KL divergence (Gu et al., 2024). Subsequent work improves the trajectories and teacher targets used for distillation. Speculative Knowledge Distillation interleaves student and teacher sampling to improve training sequences while accounting for the student’s generation distribution (Xu et al., 2025). Best-of-N Teacher Rollout Selection augments student-context supervision with an auxiliary teacher trajectory selected according to correctness and student alignment (Zhang et al., 2026). PromptKD instead tunes teacherside prompts to produce student-friendly supervision (Kim et al., 2024). These approaches modify how supervision is produced or delivered, whereas our method learns the questions that elicit it.

![](images/fa8ab8a80cdef3dffa96e75b3330c3fe6ffce3c3eacd43191d27c9e6616d638b.jpg)

![](images/e559b3ba61a831f79b56cb67c1346963d7805a7ef852318a78ef3f809e8ad0eb.jpg)

![](images/7027eec5abd2e6ee4e3f41428f4fd7a3e93971b4dfa38c7b33f6fa8c6893afab.jpg)  
Figure 9. Teacher–student divergence is positively associated with learning value. Left: Raw and partial Spearman correlations of mean divergence <sup>¯</sup>δ and mean score s¯ with one-step learning value over 1,500 questions. Partial correlations control for response length, pass rate, and entropy; error bars indicate 95% confidence intervals. Middle: Probe KL decrease increases across question quintiles ranked by s¯, with the highest quintile yielding 2.15× the decrease of the lowest. Right: Final average accuracy over six benchmarks for training on five score buckets, annotated with their mean <sup>¯</sup>δ. Points show means over three seeds; error bars and shading indicate ±1 standard deviation.

Selective OPD further recognizes that response positions provide unequal learning value. TIP characterizes token importance through student entropy and teacher–student divergence (Xu et al., 2026). Entropy-Aware OPD augments reverse KL with forward KL at high-teacher-entropy positions (Jin et al., 2026), and FiRe-OPD combines trajectory filtering with soft token reweighting (Li et al., 2026b). Token Teachability distinguishes locally learnable disagreement from mismatch outside the student’s candidate support, using teacher–student discrepancy and teacher probability mass on the student’s top-K candidates (Wang et al., 2026b). In the related setting of on-policy self-distillation, matching supervision to student capacity has also been studied through joint token weighting and privileged-information adjustment (Yang et al., 2026b). We build on the supportaware characterization of teachability, adapting it into a question-level reward with trajectory aggregation, length control, and batch-diversity control. Our optimization changes the synthetic prompt distribution while holding the downstream OPD procedure fixed.

## 4.2. Synthetic Data Generation and Selection for Distillation

Synthetic-data methods differ in the feedback used to construct training examples. Self-Instruct generates instruction– response pairs for supervised finetuning (Wang et al., 2023), and WizardLM evolves instructions toward greater complexity (Xu et al., 2024). R-Zero and Self-Questioning Language Models train task proposers through solver feedback to support self-improving reasoning (Huang et al., 2026; Chen et al., 2025). Self-Adapting Language Models learns to generate finetuning data and update directives using the adapted model’s downstream performance (Zweiger et al., 2025). Particularly relevant to our setting, Prompting to Distill uses reinforced prompts and teacher–student discrepancy to synthesize data for text-classification distillation (Ma et al., 2022). Our work extends the broader idea of feedback-guided synthesis to prompt-only OPD data, where value is assessed through teacher distributions along student-generated reasoning trajectories.

Recent studies directly examine the role of training questions in OPD. Rethinking OPD identifies teacher–student compatibility as an important factor and proposes teacheraligned prompt selection (Li et al., 2026c). Its follow-up shows that a single query can expose broad state coverage and recover much of the gain from full-data training (Fu et al., 2026). What Matters in OPD studies difficulty-based selection, finding that a small set of hard questions can match a much larger dataset in its evaluated settings and relating their utility to longer reasoning trajectories (Hou et al., 2026). Beyond Prompt Count shows that prompt usefulness depends on the student–teacher pair and target capability, with targeted selection not consistently outperforming random sampling (Wang et al., 2026a). Data-Free OPD further demonstrates effective distillation using self-generated questions without external seed examples (Li et al., 2026a).

A2D learns a synthesizer to actively shape the question distribution for a target student–teacher pair. Its objective evaluates the support-aware supervision elicited by each question, rather than using surface complexity, answer correctness, or raw disagreement alone as the generation criterion. Unlike selection from an existing question bank or direct sampling from an unoptimized model, our pipeline trains the synthesizer using a frozen scoring pair and then constructs a curated prompt-only bank for OPD. Question generation and synthesizer training reuse the target student and teacher without external training questions or auxiliary models. A frozen embedding model is used only to remove semantically redundant questions during post-generation curation.

## 5. Conclusion

We introduced A2D, a framework that treats synthetic questions as distribution elicitors for on-policy distillation. Rather than judging a question by its surface form, A2D trains a student-initialized synthesizer against a frozen student–teacher pair to expose useful teacher supervision along student-generated trajectories. Its learning-value reward combines teacher–student disagreement with teacher probability mass on the student’s candidate support, together with length and diversity controls, and the generated questions are filtered and deduplicated into a prompt-only bank for standard OPD. Synthesizer training reuses the target student and teacher without external training questions or auxiliary models; a frozen embedding model is used only for post-generation semantic deduplication. Across mathematical reasoning, code generation, and agentic tasks, A2D improves average performance and teacher-gap recovery over unoptimized student-generated prompts and KL-filtered real questions, while matching or exceeding real-bank OPD in the evaluated settings. Signal analyses further show that optimized questions can elicit denser supervision from shorter student responses. These results support elicited distributional supervision as an objective for constructing synthetic distillation data.

Our learning-value reward is a surrogate estimated from a single student response per question, and the synthesizer is optimized once for a fixed student–teacher pair. Developing more effective adaptive question-generation strategies as the student improves, and combining learned question generation with token-level selection methods for OPD, are natural directions for future work.

## References

Agarwal, R., Vieillard, N., Zhou, Y., Stanczyk, P., Garea, S. R., Geist, M., and Bachem, O. On-policy distillation of language models: Learning from self-generated mistakes. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openre view.net/forum?id=3zKtaqxLhW.

Ahmad, W. U., Narenthiran, S., Majumdar, S., Ficek, A., Jain, S., Huang, J., Noroozi, V., and Ginsburg, B. Opencodereasoning: Advancing data distillation for competitive coding. CoRR, abs/2504.01943, 2025. doi: 10.48550/ARXIV.2504.01943. URL https:

//doi.org/10.48550/arXiv.2504.01943.

Austin, J., Odena, A., Nye, M. I., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C. J., Terry, M., Le, Q. V., and Sutton, C. Program synthesis with large language models. CoRR, abs/2108.07732, 2021. doi: 10.48550/A RXIV.2108.07732. URL https://doi.org/10.4 8550/arXiv.2108.07732.

Barrès, V., Dong, H., Ray, S., Si, X., and Narasimhan, K. τ<sup>2</sup>-bench: Evaluating conversational agents in a dualcontrol environment. CoRR, abs/2506.07982, 2025. doi: 10.48550/ARXIV.2506.07982. URL https://doi. org/10.48550/arXiv.2506.07982.

Chen, L., Prabhudesai, M., Fragkiadaki, K., Liu, H., and Pathak, D. Self-questioning language models. CoRR, abs/2508.03682, 2025. doi: 10.48550/ARXIV.2508.03 682. URL https://doi.org/10.48550/arXiv .2508.03682.

Dekoninck, J., Jovanovic, N., Gehrunger, T., Rögnvaldsson, K., Petrov, I., Sun, C., and Vechev, M. T. Beyond benchmarks: Matharena as an evaluation platform for mathematics with llms. CoRR, abs/2605.00674, 2026. doi: 10.48550/ARXIV.2605.00674. URL https: //doi.org/10.48550/arXiv.2605.00674.

Fu, Z., He, B., Zuo, Y., Huang, H., Zhang, J., Xiao, R., Qian, C., Luo, Q., Gao, H., Wang, Y., Liu, Z., Ding, N., and Xiao, C. Rethinking on-policy distillation of large language models II: one training example. CoRR, abs/2609.04172, 2026. doi: 10.48550/ARXIV.2609.04 172. URL https://doi.org/10.48550/arXiv .2609.04172.

Gu, Y., Dong, L., Wei, F., and Huang, M. Minillm: Knowledge distillation of large language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net /forum?id=5h0qf7IBZZ.

Guo, D., Yang, D., Zhang, H., Song, J., Wang, P., Zhu, Q., Xu, R., Zhang, R., Ma, S., Bi, X., Zhang, X., Yu, X., Wu, Y., Wu, Z. F., Gou, Z., Shao, Z., Li, Z., Gao, Z., Liu, A., Xue, B., Wang, B., Wu, B., Feng, B., Lu, C., Zhao, C., Deng, C., Ruan, C., Dai, D., Chen, D., Ji, D., Li, E., Lin, F., Dai, F., Luo, F., Hao, G., Chen, G., Li, G., Zhang, H., Xu, H., Ding, H., Gao, H., Qu, H., Li, H., Guo, J., Li, J., Chen, J., Yuan, J., Tu, J., Qiu, J., Li, J., Cai, J. L., Ni, J., Liang, J., Chen, J., Dong, K., Hu, K., You, K., Gao, K., Guan, K., Huang, K., Yu, K., Wang, L., Zhang, L., Zhao, L., Wang, L., Zhang, L., Xu, L., Xia, L., Zhang, M., Zhang, M., Tang, M., Zhou, M., Li, M., Wang, M., Li, M., Tian, N., Huang, P., Zhang, P., Wang, Q., Chen,

Q., Du, Q., Ge, R., Zhang, R., Pan, R., Wang, R., Chen, R. J., Jin, R. L., Chen, R., Lu, S., Zhou, S., Chen, S., Ye, S., Wang, S., Yu, S., Zhou, S., Pan, S., Li, S. S., Zhou, S., Wu, S., Yun, T., Pei, T., Sun, T., Wang, T., Zeng, W., Liu, W., Liang, W., Gao, W., Yu, W., Zhang, W., Xiao, W. L., An, W., Liu, X., Wang, X., Chen, X., Nie, X., Cheng, X., Liu, X., Xie, X., Liu, X., Yang, X., Li, X., Su, X., Lin, X., Li, X. Q., Jin, X., Shen, X., Chen, X., Sun, X., Wang, X., Song, X., Zhou, X., Wang, X., Shan, X., Li, Y. K., Wang, Y. Q., Wei, Y. X., Zhang, Y., Xu, Y., Li, Y., Zhao, Y., Sun, Y., Wang, Y., Yu, Y., Zhang, Y., Shi, Y., Xiong, Y., He, Y., Piao, Y., Wang, Y., Tan, Y., Ma, Y., Liu, Y., Guo, Y., Ou, Y., Wang, Y., Gong, Y., Zou, Y., He, Y., Xiong, Y., Luo, Y., You, Y., Liu, Y., Zhou, Y., Zhu, Y. X., Huang, Y., Li, Y., Zheng, Y., Zhu, Y., Ma, Y., Tang, Y., Zha, Y., Yan, Y., Ren, Z. Z., Ren, Z., Sha, Z., Fu, Z., Xu, Z., Xie, Z., Zhang, Z., Hao, Z., Ma, Z., Yan, Z., Wu, Z., Gu, Z., Zhu, Z., Liu, Z., Li, Z., Xie, Z., Song, Z., Pan, Z., Huang, Z., Xu, Z., Zhang, Z., and Zhang, Z. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nat., 645(8081):633–638, 2025. doi: 10.1038/S41586-025-09422-Z. URL https: //doi.org/10.1038/s41586-025-09422-z.

He, B., Qu, Z., Liu, Z., Chen, Y., Zuo, Y., Qian, C., Zhang, K., Chen, W., Xiao, C., Cui, G., Ding, N., and Liu, Z. Justrl: Scaling a 1.5b LLM with a simple RL recipe. CoRR, abs/2512.16649, 2025a. doi: 10.48550/ARXIV .2512.16649. URL https://doi.org/10.48550 /arXiv.2512.16649.

He, J., Liu, J., Liu, C. Y., Yan, R., Wang, C., Cheng, P., Zhang, X., Zhang, F., Xu, J., Shen, W., Li, S., Zeng, L., Wei, T., Cheng, C., An, B., Liu, Y., and Zhou, Y. Skywork open reasoner 1 technical report. CoRR, abs/2505.22312, 2025b. doi: 10.48550/ARXIV.2505.22312. URL http s://doi.org/10.48550/arXiv.2505.22312.

Hendrycks, D., Basart, S., Kadavath, S., Mazeika, M., Arora, A., Guo, E., Burns, C., Puranik, S., He, H., Song, D., and Steinhardt, J. Measuring coding challenge competence with APPS. In Vanschoren, J. and Yeung, S. (eds.), Proceedings ofthe Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual, 2021a. URL https://datasets-benchmarks-proce edings.neurips.cc/paper/2021/hash/c2 4cd76e1ce41366a4bbe8a49b02a028-Abstr act-round2.html.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the MATH dataset. In Vanschoren, J. and Yeung, S. (eds.), Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and

Benchmarks 2021, December 2021, virtual, 2021b. URL https://datasets-benchmarks-proceedin gs.neurips.cc/paper/2021/hash/be83ab 3ecd0db773eb2dc1b0a17836a1-Abstract-r ound2.html.

Hou, Z., Zhang, J., Cai, X., and You, K. What matters in on-policy distillation? A perspective on data efficiency and data selection. CoRR, abs/2609.05198, 2026. doi: 10.48550/ARXIV.2609.05198. URL https://doi. org/10.48550/arXiv.2609.05198.

Huang, C., Yu, W., Wang, X., Zhang, H., Li, Z., Li, R., Huang, J., Mi, H., and Yu, D. R-zero: Self-evolving reasoning LLM from zero data. In Vondrick, C., Hariharan, B., Raffel, C., Pinto, L., Yang, D., and Faust, A. (eds.), The Fourteenth International Conference on Learning Representations, ICLR 2026, Rio de Janeiro, Brazil, April 23-27, 2026. proceedings.iclr.cc, 2026. URL https: //proceedings.iclr.cc/paper\_files/pa per/2026/hash/d49b9aacebda6105116633 5af6fd3061-Abstract-Conference.html.

Hui, B., Yang, J., Cui, Z., Yang, J., Liu, D., Zhang, L., Liu, T., Zhang, J., Yu, B., Dang, K., Yang, A., Men, R., Huang, F., Ren, X., Ren, X., Zhou, J., and Lin, J. Qwen2.5-coder technical report. CoRR, abs/2409.12186, 2024. doi: 10.48550/ARXIV.2409.12186. URL https: //doi.org/10.48550/arXiv.2409.12186.

Jain, N., Han, K., Gu, A., Li, W., Yan, F., Zhang, T., Wang, S., Solar-Lezama, A., Sen, K., and Stoica, I. Livecodebench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum ?id=chfJJYC3iL.

Jin, W., Min, T., Yang, Y., Kadhe, S. R., Zhou, Y., Wei, D., Baracaldo, N., and Lee, K. Entropy-aware on-policy distillation of language models. CoRR, abs/2603.07079, 2026. doi: 10.48550/ARXIV.2603.07079. URL https: //doi.org/10.48550/arXiv.2603.07079.

Kim, G., Jang, D., and Yang, E. Promptkd: Distilling student-friendly knowledge for generative language models via prompt tuning. In Al-Onaizan, Y., Bansal, M., and Chen, Y. (eds.), Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, Miami, Florida, USA, November 12-16, 2024, volume EMNLP 2024 of Findings of ACL, pp. 6266–6282. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.F INDINGS-EMNLP.364. URL https://doi.org/ 10.18653/v1/2024.findings-emnlp.364.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In Flinn, J., Seltzer, M. I., Druschel, P., Kaufmann, A., and Mace, J. (eds.), Proceedings of the 29th Symposium on Operating Systems Principles, SOSP 2023, Koblenz, Germany, October 23-26, 2023, pp. 611– 626. ACM, 2023. doi: 10.1145/3600006.3613165. URL https://doi.org/10.1145/3600006.3613 165.

Li, G., Zheng, M., Song, M., Sun, J., Liu, Z., Liu, R., Zhong, Q., Guo, H., Fang, J., and Wang, J. Data-free on-policy distillation. CoRR, abs/2609.14193, 2026a. doi: 10.48550/ARXIV.2609.14193. URL https: //doi.org/10.48550/arXiv.2609.14193.

Li, Y., Choi, D. H., Chung, J., Kushman, N., Schrittwieser, J., Leblond, R., Eccles, T., Keeling, J., Gimeno, F., Lago, A. D., Hubert, T., Choy, P., de Masson d’Autume, C., Babuschkin, I., Chen, X., Huang, P., Welbl, J., Gowal, S., Cherepanov, A., Molloy, J., Mankowitz, D. J., Robson, E. S., Kohli, P., de Freitas, N., Kavukcuoglu, K., and Vinyals, O. Competition-level code generation with alphacode. CoRR, abs/2203.07814, 2022. doi: 10.48550/ARXIV.2203.07814. URL https: //doi.org/10.48550/arXiv.2203.07814.

Li, Y., Zheng, L., Yu, Y., Zhou, W., Zhong, X., Hu, X., Jin, J., Yuan, H., and Feng, T. Filter, then reweight: Rethinking optimization granularity in on-policy distillation. CoRR, abs/2606.02684, 2026b. doi: 10.48550/ARXIV.2606.02 684. URL https://doi.org/10.48550/arXiv .2606.02684.

Li, Y., Zuo, Y., He, B., Zhang, J., Xiao, C., Qian, C., Yu, T., Gao, H., Yang, W., Liu, Z., and Ding, N. Rethinking onpolicy distillation of large language models: Phenomenology, mechanism, and recipe. CoRR, abs/2604.13016, 2026c. doi: 10.48550/ARXIV.2604.13016. URL http s://doi.org/10.48550/arXiv.2604.13016.

Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum ?id=v8L0pN6EOi.

Liu, J., Xia, C. S., Wang, Y., and Zhang, L. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. In Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., and Levine, S. (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural

Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper\_files /paper/2023/hash/43e9d647ccd3e4b7b5b aab53f0368686-Abstract-Conference.ht ml.

Ma, X., Wang, X., Fang, G., Shen, Y., and Lu, W. Prompting to distill: Boosting data-free knowledge distillation via reinforced prompt. In Raedt, L. D. (ed.), Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, IJCAI 2022, Vienna, Austria, 23-29 July 2022, pp. 4296–4302. ijcai.org, 2022. doi: 10.24963/IJCAI.2022/596. URL https://doi.org/ 10.24963/ijcai.2022/596.

Mathematical Association of America. American Mathematics Competitions (AMC) 2023, 2023. URL https: //maa.org/student-programs/amc/.

Mathematical Association of America. American Invitational Mathematics Examination (AIME) 2024, 2024. URL https://maa.org/maa-invitationa l-competitions/.

Meurer, A., Smith, C. P., Paprocki, M., Certík, O., Kirpichev, S. B., Rocklin, M., Kumar, A., Ivanov, S., Moore, J. K., Singh, S., Rathnayake, T., Vig, S., Granger, B. E., Muller, R. P., Bonazzi, F., Gupta, H., Vats, S., Johansson, F., Pedregosa, F., Curry, M. J., Terrel, A. R., Roucka, S., Saboo, A., Fernando, I., Kulal, S., Cimrman, R., and Scopatz, A. M. Sympy: symbolic computing in python. PeerJ Comput. Sci., 3:e103, 2017. doi: 10.7717/PEER J-CS.103. URL https://doi.org/10.7717/pe erj-cs.103.

OpenAI. GPT-4o mini: Advancing cost-efficient intelligence, July 2024. URL https://openai.com/i ndex/gpt-4o-mini-advancing-cost-effic ient-intelligence/.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. CoRR, abs/1707.06347, 2017. doi: 10.48550/ARXIV.1 707.06347. URL https://doi.org/10.48550/a rXiv.1707.06347.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Zhang, M., Li, Y. K., Wu, Y., and Guo, D. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. CoRR, abs/2402.03300, 2024. doi: 10.48550/A RXIV.2402.03300. URL https://doi.org/10.4 8550/arXiv.2402.03300.

Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R., Peng, Y., Lin, H., and Wu, C. Hybridflow: A flexible and efficient RLHF framework. In Proceedings of the

Twentieth European Conference on Computer Systems, EuroSys 2025, Rotterdam, The Netherlands, 30 March 2025 - 3 April 2025, pp. 1279–1297. ACM, 2025. doi: 10.1145/3689031.3696075. URL https://doi.or g/10.1145/3689031.3696075.

Team, Q. Qwen3 technical report. CoRR, abs/2505.09388, 2025. doi: 10.48550/ARXIV.2505.09388. URL https: //doi.org/10.48550/arXiv.2505.09388.

Wang, J., Lyu, J., Cai, Y., Wu, S., Wang, P., Liu, J., Cheng, X., Yang, K., Chen, Y., Yang, S., and Guo, L.-Z. Beyond prompt count: How data shapes transfer in on-policy distillation, 2026a. URL https://arxiv.org/ab s/2609.37377.

Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., and Hajishirzi, H. Self-instruct: Aligning language models with self-generated instructions. In Rogers, A., Boyd-Graber, J. L., and Okazaki, N. (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, pp. 13484–13508. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.ACL-LONG.754. URL https://doi.org/10.18653/v1/2023 .acl-long.754.

Wang, Y., Lu, S., Gu, Y., Wang, P., Yang, Y., Yan, Z., Xie, C., Wu, J., and Yang, H. Not all disagreement is learnable: Token teachability in on-policy distillation. CoRR, abs/2605.26844, 2026b. doi: 10.48550/ARXIV .2605.26844. URL https://doi.org/10.48550 /arXiv.2605.26844.

Xu, C., Sun, Q., Zheng, K., Geng, X., Zhao, P., Feng, J., Tao, C., Lin, Q., and Jiang, D. Wizardlm: Empowering large pre-trained language models to follow complex instructions. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https: //openreview.net/forum?id=CfXh93NDgH.

Xu, W., Han, R., Wang, Z., Le, L. T., Madeka, D., Li, L., Wang, W. Y., Agarwal, R., Lee, C., and Pfister, T. Speculative knowledge distillation: Bridging the teacher-student gap through interleaved sampling. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum ?id=EgJhwYR2tB.

Xu, Y., Sang, H., Zhou, Z., He, R., Wang, Z., and Geramifard, A. Tip: Token importance in on-policy distillation, 2026. URL https://arxiv.org/abs/2604.1 4084.

Yang, W., Liu, W., Xie, R., Yang, K., Yang, S., and Lin, Y. Learning beyond teacher: Generalized on-policy distillation with reward extrapolation. CoRR, abs/2602.12125, 2026a. doi: 10.48550/ARXIV.2602.12125. URL http s://doi.org/10.48550/arXiv.2602.12125.

Yang, Y., Hao, Z., Zhang, H., Liu, Y., Lin, X., Ji, W., Wei, F., Yu, J., Lin, Q., Liang, X., and Dong, H. Matching supervision to the student’s learning capacity: A unified framework for on-policy self-distillation, 2026b. URL https://arxiv.org/abs/2608.08176.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, J., Liu, L., Liu, X., Lin, H., Lin, Z., Ma, B., Sheng, G., Tong, Y., Zhang, C., Zhang, M., Zhang, R., Zhang, W., Zhu, H., Zhu, J., Chen, J., Chen, J., Wang, C., Yu, H., Song, Y., Wei, X., Zhou, H., Liu, J., Ma, W., Zhang, Y., Yan, L., Wu, Y., and Wang, M. DAPO: an open-source LLM reinforcement learning system at scale. In Belgrave, D., Zhang, C., Montoya, L. N., Lin, H., Pascanu, R., Koniusz, P., Ghassemi, M., Chen, N., Ruíz, I. V. M., and Loaiza-Bonilla, A. (eds.), Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025, 2025. URL http://papers.nips.cc/pap er\_files/paper/2025/hash/a4277440d50 f1f15d2cb4c14f7e0c0d2-Abstract-Confe rence.html.

Zhang, K., Tian, Y., Zhao, D., Li, Y., Liu, Y., Patel, V. M., and Fu, D. On-policy distillation with best-of-n teacher rollout selection. CoRR, abs/2605.09725, 2026. doi: 10.48550/ARXIV.2605.09725. URL https://doi. org/10.48550/arXiv.2605.09725.

Zhang, Y. and Math-AI, T. American invitational mathematics examination (aime) 2025, 2025.

Zhang, Y., Li, M., Long, D., Zhang, X., Lin, H., Yang, B., Xie, P., Yang, A., Liu, D., Lin, J., Huang, F., and Zhou, J. Qwen3 embedding: Advancing text embedding and reranking through foundation models. CoRR, abs/2506.05176, 2025. doi: 10.48550/ARXIV.2506.05 176. URL https://doi.org/10.48550/arXiv .2506.05176.

Zweiger, A., Pari, J., Guo, H., Akyürek, E., Kim, Y., and Agrawal, P. Self-adapting language models. CoRR, abs/2506.10943, 2025. doi: 10.48550/ARXIV.2506. 10943. URL https://doi.org/10.48550/arX iv.2506.10943.

## A. Experimental Details

## A.1. Algorithm

Algorithm 1 summarizes our method.

## A.2. Implementation Details

Our framework was developed based on verl (Sheng et al., 2025). For reward computation, we use $K = 1 6$ candidate tokens, a question-generation budget of $L _ { q } = 4 0 9 6$ tokens, and a student-response budget of $L _ { \mathrm { m a x } } = 7 1 6 8$ tokens, with the latter matched to the downstream OPD response budget by construction (§2.6). For each student–teacher pair, the discrepancy scale $\delta _ { \mathrm { r e f } }$ is set to the $\kappa = 9 5 \mathrm { { t h } }$ percentile of $\delta _ { t }$ estimated from $N _ { \mathrm { c a l } } = 2 0 0$ questions, and we use $\epsilon = 1 0 ^ { - 8 }$ in Eq. (6). For reward shaping, we set the aggregation exponent $\alpha = 0 . 5$ , the length-penalty coefficient $\lambda _ { \mathrm { l e n } } = 0 . 3 .$ , the diversity-penalty coefficient $\lambda _ { \mathrm { d i v } } = 1 . 0$ , and the clustering threshold $\tau = 0 . 1$ . The synthetic-data synthesizer is optimized with GRPO using 8 prompts and a group size of $g = 8 ,$ , yielding 64 candidate questions per step. We use a learning rate of $1 \times 1 0 ^ { - 6 }$ , entropy and KL coefficients of 0.01 and 0.005, respectively, and a rollout temperature of 1.0, training for 60 steps and saving checkpoints every 5 steps. We generate $N = 2 0 { , } 0 0 0$ raw questions in 10 shards of 2000 samples each and apply Jaccard and cosine deduplication thresholds of $\tau _ { \mathrm { l e x } } = 0 . 7$ and $\tau _ { \mathrm { s e m } } = 0 . 9 5$ , respectively. Downstream OPD uses token-level rewards directly as advantages, with the student’s top-K candidates $( K = 1 6 )$ weighted by $\bar { p } _ { S }$ . We sample $n _ { \mathrm { r o l l } } = 4$ responses per prompt at temperature 1.0 and train with a batch size of 64, a learning rate of $1 \times 1 0 ^ { - 6 }$ , and $\tt f p 3 2$ parameters for 2 epochs, saving checkpoints every 5 steps. For each student–teacher pair and each OPD method, we select a single downstream student checkpoint based on AMC23 performance for mathematical reasoning and LiveCodeBench performance for code generation. All benchmark scores reported for that method are evaluated at this same checkpoint, rather than selected independently for each benchmark. AMC23 and LiveCodeBench therefore serve as checkpoint-selection benchmarks; their reported scores are not held-out estimates.

## A.3. Evaluation Details

Metric. For each benchmark, we sample k responses per problem and report avg@k: the fraction of correct responses for each problem, averaged over problems and scaled to [0, 100]. With $k = 1$ , this reduces to single-sample accuracy. All reported results are means over three independent runs with different random seeds. The reported averages weight each benchmark equally, regardless of its number of problems. Table 5 lists the per-benchmark settings. All evaluations use vLLM (v0.11.0) (Kwon et al., 2023) with bf16 inference on NVIDIA A100 80GB GPUs.

Mathematical reasoning. We use the DAPO prompt format, which asks the model to reason step by step and to give its final answer within \boxed{}. Qwen3 students and teachers are evaluated in non-thinking mode, whereas DeepSeek R1-Distill students and their teachers (JustRL and Skywork-OR1) use their default thinking templates. We extract the last \boxed{} expression, falling back to the last “Answer:” line, and verify it against the reference answer by rule-based string normalization and symbolic equivalence checking with SymPy (Meurer et al., 2017); no model-based judge is used.

Code generation. For LiveCodeBench, we use the v5–v6 release subset of 342 problems published between September 2024 and April 2025, with the original LiveCodeBench prompt. For APPS, we use the full introductory and competition test splits (1,000 problems each) and 1,000 interview problems drawn from the test split with a fixed random seed, keeping at most 50 test cases per problem; prompts follow the LiveCodeBench standard-input template. MBPP+ uses EvalPlus v0.2.0 (378 problems) with its standard prompt template and greedy decoding. Qwen3 models are evaluated in non-thinking mode. We extract the last fenced code block from each response and execute it against all test cases; a response is counted as correct only if it passes every test within the time limit. In the math-to-code transfer experiment (Figure 7), the Qwen3 models use the non-thinking settings in Table 5, whereas DeepSeek-R1-Distill students and their mathematics teachers are evaluated in thinking mode with their default chat templates, using temperature 0.6, top-p 0.95, top-k 20, and a maximum generation length of 32,768 tokens on all three benchmarks.

## A.4. Discussion

Combining with filtering and OPD variants A2D learns a question distribution for a target student–teacher pair, while filtering determines which sampled questions are retained and the downstream OPD procedure determines how teacher supervision updates the student. These operations act at different stages of the pipeline. Consequently, learned question generation can be followed by additional filtering or selection rules, and the resulting synthetic bank can be used by alternative distillation procedures that accept prompt-only datasets. Our format filtering and deduplication already illustrate this separation: they curate the synthesizer’s outputs before those questions are used for OPD.

Table 5. Evaluation settings. k is the number of sampled responses per problem; “Max tokens” is the maximum generation length.
<table><tr><td>Benchmark</td><td>#Problems</td><td>k</td><td>Temperature</td><td>Top-p</td><td>Top-k</td><td>Max tokens</td></tr><tr><td>AIME24 / AIME25 / AIME26</td><td>30 each</td><td>16</td><td>0.7</td><td>0.95</td><td></td><td>31,744</td></tr><tr><td>AMC23</td><td>40</td><td>16</td><td>0.7</td><td>0.95</td><td></td><td>31,744</td></tr><tr><td>BRUMO25</td><td>30</td><td>16</td><td>0.7</td><td>0.95</td><td></td><td>31,744</td></tr><tr><td>MATH500</td><td>500</td><td>4</td><td>0.7</td><td>0.95</td><td>一</td><td>31,744</td></tr><tr><td>LiveCodeBench (v5-v6)</td><td>342</td><td>4</td><td>0.7</td><td>0.8</td><td>20</td><td>8,192</td></tr><tr><td>APPS-Intro</td><td>1,000</td><td>1</td><td>0.2</td><td>0.95</td><td>20</td><td>4,096</td></tr><tr><td>APPS-Interview</td><td>1,000</td><td>1</td><td>0.2</td><td>0.95</td><td>20</td><td>4,096</td></tr><tr><td>APPS-Competition</td><td>1,000</td><td>1</td><td>0.2</td><td>0.95</td><td>20</td><td>4,096</td></tr><tr><td>MBPP+</td><td>378</td><td>1</td><td></td><td>greedy</td><td></td><td>1,024</td></tr><tr><td>CodeContests</td><td>165</td><td>1</td><td>0.2</td><td>0.95</td><td>20</td><td>4096</td></tr></table>

The learning-value reward nevertheless connects prompt optimization to the downstream distillation procedure. Our support-aware score is designed around teacher supervision on the student’s candidate support. A procedure using a different divergence, candidate support, or supervision allocation rule may benefit from a corresponding adaptation of the generation reward. We therefore view A2D as a modular approach to constructing distillation prompts, with the effectiveness of any particular combination depending on how well the generation objective matches the downstream learning signal. Exploring such combinations is a natural direction for future work.

## A.5. Experimental Details and Additional Signal Validation

![](images/653a58f7f87a56eb3b98df275147239ecb4d634d21257a7e76b43e9dd019fc3c.jpg)  
Figure 10. Accuracy gains of high- over low-divergence training groups. Each cell reports the gain in percentage points, its 95% confidence interval, and p-value.

Data and splits. All questions in these analyses are natural questions from DAPO-17k-Math; no synthesizer outputs are used. Among 17,177 eligible questions, we exclude a 128-question probe set and 500 questions reserved for tokenlevel analysis from downstream training, leaving a pool of 16,549 questions. The one-step analysis uses 1,500 update questions disjoint from the probe set: the 500 token-level questions and 1,000 additional questions from the training pool. The one-step and downstream training analyses are conducted independently.

One-step learning value. For each update question, we apply a single plain-SGD OPD step with a learning rate of $3 \times 1 0 ^ { - 5 }$ using four responses sampled from the base student. Each question is evaluated starting from the same base-student parameters. Its learning value is the resulting decrease in full-vocabulary $\mathrm { K L } ( \pi _ { T } \| \pi _ { S } )$ on the fixed probe set, averaged over tokens of one fixed base-student response per probe question. This measurement differs from the OPD update in its questions, prefixes, and KL direction: the

update uses a reverse-KL-style objective on the student’s top-16 candidate tokens, whereas the probe evaluates full-vocabulary teacher-to-student KL on separate questions and fixed prefixes.

Test–retest reliability. We estimate test–retest reliability on 200 of the update questions by repeating the analysis with an independent draw of four student responses per question. For each pair of quantities being correlated, we compute a reliability-corrected correlation by dividing the observed Spearman correlation by the square root of the product of their two test–retest reliabilities. We report this correction as a supplementary analysis of sensitivity to rollout-sampling noise.

Score-ranked training banks. We partition the training pool into five approximately equal-sized buckets by mean support-aware score s¯, yielding approximately 3,310 questions per bucket. A random bank of the same size serves as a reference. Each bank is evaluated using three training seeds.

Factorial designs and divergence matching. For the $2 \times 2$ designs, we select the bottom and top tertiles of mean divergence ${ \bar { \delta } } ,$ corresponding to $\bar { \delta } \leq 0 . 2 9 0$ and ${ \bar { \delta } } \geq 0 . 4 0 4 .$ , respectively. Within each tertile, we split questions at the median of mean compatibility $\bar { \omega } ,$ , producing four cells of approximately 2,758 questions each. Because $\bar { \delta }$ and ω¯ are negatively correlated, this split can leave differences in divergence between the compatibility cells. The $\delta \mathfrak { - }$ matched variant therefore divides each divergence tertile into 20 strata by $\bar { \delta }$ and splits ω¯ at its median within each stratum. This procedure approximately balances mean divergence between the two compatibility cells within each divergence level: 0.528 versus 0.531 in the high-divergence tertile and 0.229 versus 0.230 in the low-divergence tertile. Each resulting cell contains approximately 2,758 questions.

Training budgets and checkpoints. All banks use a common target budget of 24.37M training tokens. Runs reach this budget within 26–36 steps, including 30–31 steps for the five score buckets and the random reference bank, and consume at most 2,304 questions. No question is used more than once within a run. Seeds differ in question order and therefore in the subset consumed before reaching the budget. We evaluate the end checkpoint at the first step reaching the full budget. For the mid checkpoint, we use the first saved checkpoint reaching half the budget, with checkpoints saved every five steps. Consequently, mid checkpoints span 12.4M–18.2M tokens across banks and provide less tightly budget-controlled comparisons than end checkpoints.

Statistical analysis. Confidence intervals for the one-step analysis are obtained by bootstrapping over the 1,500 update questions. For downstream training, benchmark scores are first averaged within each run, and the training run is the unit of inference, with three seeds per bank. Pairwise contrasts use run-level two-sample t-tests, and effects in the $2 \times 2$ designs are evaluated using a two-way ANOVA over runs. The Jonckheere–Terpstra test assesses the ordered trend across the five score buckets using permutations over their 15 training runs. Figure 9 reports means and standard deviations across seeds.

Additional controlled comparisons. To examine whether the training advantage associated with higher divergence persists after accounting for compatibility, we compare high- and low-divergence question banks under both the original design and a design balanced on the compatibility factor ω. Figure 10 reports accuracy differences in percentage points at intermediate and final training checkpoints, together with 95% confidence intervals and p-values.

At the final checkpoint, high-divergence banks outperform low-divergence banks by 2.21 points in the original design (95% $\mathbf { C I } \ [ 1 . 3 7 , 3 . 0 6 ] , p = 3 . 1 \times 1 0 ^ { - 4 } )$ . After balancing ω, the advantage remains positive at 1.59 points (95% CI [0.63, 2.56], $p = 0 . 0 0 5 )$ . The intermediate checkpoint shows the same direction: gains of 2.27 points in the original design (95% CI $[ 0 . 6 2 , 3 . 9 1 ] , p = 0 . 0 1 3 )$ and 2.79 points after balancing $\omega \ ( 9 5 \% \ C I \ [ 1 . 7 5 , 3 . 8 2 ] , p = 2 . 6 \times 1 0 ^ { - 4 } )$ . All four confidence intervals exclude zero, supporting a positive training advantage for high-divergence banks that persists under compatibility balancing.

## B. Theoretical Properties of the Elicited-Signal Objective

We establish two properties of our framework: high normalized reward certifies positions with simultaneous discrepancy and compatibility, and compatibility enters the implemented OPD advantages through an explicit log-ratio term. These results characterize the elicited supervision and its connection to the update, without assuming convergence of synthesizer optimization.

Notation. Fix the scoring student $\pi _ { S }$ and teacher $\pi _ { T }$ . For a question $q ,$ let $y \sim \pi _ { S } ( \cdot \mid q )$ be a nonempty response of length $1 \leq L \leq L _ { \mathrm { m a x } }$ . At position $t ,$ let $p _ { S , t }$ and $p _ { T , t }$ denote the next-token distributions on the student-generated prefix, and define

$$
A _ { t } = \mathrm { T o p } { \cdot } K ( p _ { S , t } ) , \qquad B _ { t } = \mathrm { T o p } { \cdot } K ( p _ { T , t } ) , \qquad U _ { t } = A _ { t } \cup B _ { t } .
$$

For a nonempty token set $C ,$ , write $\begin{array} { r } { p _ { M , t } ( C ) = \sum _ { v \in C } p _ { M , t } ( v ) } \end{array}$ and let $\bar { p } _ { M , t } ^ { C }$ denote the distribution conditioned on $C ,$ extended by zero outside C. We assume strictly positive next-token probabilities. All logarithms are natural.

Following Eqs. (4)–(6), write

$$
\delta _ { t } = \mathrm { K L } \left( \bar { p } _ { T , t } ^ { U _ { t } } \Big \| \bar { p } _ { S , t } ^ { U _ { t } } \right) , \qquad \omega _ { t } = p _ { T , t } ( A _ { t } ) ,
$$

and

$$
s _ { t } = \operatorname* { m i n } \{ \delta _ { t } / \Delta , 1 \} \omega _ { t } , ~ \Delta = \delta _ { \mathrm { r e f } } + \epsilon > 0 .
$$

Thus $s _ { t } \in [ 0 , 1 ]$ . With $\textstyle { \bar { s } } = L ^ { - 1 } \sum _ { \ell = 1 } ^ { L } s _ { t }$ , the reward in Eq. (11) is

$$
R = P _ { \mathrm { l e n } } P _ { \mathrm { d i v } } L ^ { 1 - \alpha } \bar { s } , \qquad 0 < \alpha < 1 ,
$$

where both penalty factors lie in [0, 1]. We suppress the dependence of R on the sampled response and scoring batch, and write $[ x ] _ { + } = \operatorname* { m a x } \{ x , 0 \}$

## B.1. Compatible-disagreement coverage

For a threshold $h \in ( 0 , 1 )$ , define

$$
\mathcal { E } _ { h } = \{ t : \delta _ { t } \geq h \Delta , \omega _ { t } \geq h \} .
$$

These positions simultaneously exhibit teacher–student discrepancy and teacher probability mass within the student’s candidate set.

Proposition B.1 (Compatible-disagreement coverage). For every nonempty response,

$$
\frac { \left| \mathcal { E } _ { h } \right| } { L } \geq \left[ \frac { \bar { s } - h } { 1 - h } \right] _ { + } \geq \left[ \frac { R / L ^ { 1 - \alpha } - h } { 1 - h } \right] _ { + } .
$$

Proof. $\operatorname { I f } t \notin { \mathcal { E } } _ { h }$ , then either $\delta _ { t } < h \Delta$ or $\omega _ { t } < h$ , and hence $s _ { t } \leq h$ . Since $s _ { t } \leq 1$ at every position,

$$
\bar { s } \leq h + ( 1 - h ) \frac { | \mathcal { E } _ { h } | } { L } .
$$

Rearranging, using nonnegativity, and noting that

$$
R / L ^ { 1 - \alpha } = P _ { \mathrm { l e n } } P _ { \mathrm { d i v } } \bar { s } \leq \bar { s }
$$

gives the result.

Thus, a normalized reward above h certifies a nonzero fraction of positions exhibiting both discrepancy and compatibility.   
Normalization by $L ^ { 1 - \alpha }$ is necessary because the unnormalized reward also depends on response length.

Expected coverage. Sample a scoring batch, select a question uniformly, sample its student response, and select a response position uniformly. Then

$$
\operatorname* { P r } ( t \in \mathcal { E } _ { h } ) \geq \left[ \frac { \mathbb { E } [ R / L ^ { 1 - \alpha } ] - h } { 1 - h } \right] _ { + } .
$$

This follows by taking expectations of the pointwise inequality in the proof above and using nonnegativity. The expectation includes question generation, student responses, and the batch-dependent diversity factor.

## B.2. Compatibility and OPD advantages

We next relate compatibility to the candidate advantages used in §2.6. Unlike the coverage certificate computed with the frozen scoring student, the following identity holds for any rollout policy during OPD.

Proposition B.2 (Compatibility in the OPD advantage). Fix a response position and let $p _ { S } = p _ { S , \mathrm { o l d } }$ be the rollout-policy distribution. Write

$$
A = \mathrm { T o p } { \cdot } K ( p _ { S } ) , \qquad a = p _ { S } ( A ) , \qquad \omega = p _ { T } ( A ) .
$$

For the candidate advantages in Eq. (13),

$$
r ( v ) = \bar { p } _ { S } ^ { A } ( v ) \log \frac { p _ { T } ( v ) } { p _ { S } ( v ) } , \qquad v \in A ,
$$

we have

$$
\sum _ { v \in A } r ( v ) = - \mathrm { K L } \big ( \bar { p } _ { S } ^ { A } \big \| \bar { p } _ { T } ^ { A } \big ) + \log \frac { \omega } { a } .
$$

If the position is scored using these same student and teacher distributions, then $s \geq h$ implies lo $\operatorname { g } ( \omega / a ) \geq \log h .$

Proof. For each $v \in A .$

$$
\log \frac { p _ { T } ( v ) } { p _ { S } ( v ) } = \log \frac { \bar { p } _ { T } ^ { A } ( v ) } { \bar { p } _ { S } ^ { A } ( v ) } + \log \frac { \omega } { a } .
$$

Multiplying by $\bar { p } _ { S } ^ { A } ( v )$ and summing over A gives the identity. The final claim follows from ω $\geq s \geq h$ and $a \leq 1$ □

Compatibility therefore contributes a common additive offset to the candidate log-ratios, while the conditional distributions determine their relative values. When $\omega < a ,$ this offset is negative; at high-scoring positions, it is bounded below by log h. The identity characterizes the fixed advantages, rather than the resulting parameter-gradient norm.

Teacher truncation. Compatibility also has a distributional interpretation:

$$
\mathrm { T V } ( p _ { T } , \bar { p } _ { T } ^ { A } ) = 1 - \omega ,
$$

where $\begin{array} { r } { \mathrm { T V } ( P , Q ) = \frac { 1 } { 2 } \sum _ { v } | P ( v ) - Q ( v ) | } \end{array}$ . Indeed, the total absolute difference inside A is $1 - \omega .$ , as is the teacher mass outside A. Consequently, $s _ { t } \geq h$ implies

$$
\mathrm { T V } ( p _ { T , t } , \bar { p } _ { T , t } ^ { A _ { t } } ) \leq 1 - h .
$$

Here $\hat { p } _ { T } ^ { A }$ is an analysis device: the implemented update uses full-vocabulary teacher log-probabilities. Moreover, a fixed top-K set need not contain most of the student’s probability mass; this must be assessed separately through $p _ { S } ( A )$

Scope. These results characterize the scoring signal and its relationship to the fixed OPD advantages. They do not lower-bound parameter gradients or downstream improvement: the scoring discrepancy is computed on $U _ { t } ,$ whereas explicit candidate-level update terms are restricted to $A _ { t }$ , and the scoring and update divergences have different directions. Coverage certificates computed with the frozen scoring student also need not persist as the student evolves during OPD. Compatibility concerns candidate probability mass, rather than question correctness or membership in either model’s training distribution. The effectiveness of the elicited supervision for student learning is evaluated empirically in §3.

Algorithm 1 A2D: Learning to Elicit Teacher Supervision   
Require: Student $\pi _ { S } ,$ teacher $\pi _ { T } ,$ question-generation instruction $x ;$ candidate size $K ,$ question budget $L _ { q } ,$ shared response budget   
$L _ { \mathrm { m a x } } ;$ calibration percentile κ and question count $N _ { \mathrm { c a l } } ;$ synthesizer steps $T _ { \mathrm { s y n } }$ , groups per step B, group size g; raw question count   
N, OPD epochs $\bar { E , }$ responses per question n<sub>roll</sub>.   
Ensure: Synthetic question bank $\bar { \mathcal D }$ and distilled student $\pi _ { S } .$   
1: Initialize synthesizer $S _ { \theta } \gets \mathrm { C o p y } ( \pi _ { S } ) .$   
2: Freeze $\pi _ { S }$ and π<sub>T</sub> during synthesizer optimization.   
3: Calibrate $\delta _ { \mathrm { r e f } }$ as the κ-th percentile of token discrepancies on student rollouts for $N _ { \mathrm { c a l } }$ student-generated questions.   
4: Stage 1: Optimize the synthetic question distribution   
5: for $\bar { k } = 1 , \bar { \ldots } , T _ { \mathrm { s y n } }$ do   
6: Sample $B$ groups of questions $\mathcal { Q } _ { b } = \{ q _ { b , i } \} _ { i = 1 } ^ { g }$ from $S _ { \theta } ( \cdot \mid x )$ under budget $L _ { q } .$   
7: Form the question batch Q by concatenating all groups.   
8: Cluster questions using BLEU-based pairwise distances to compute share $: ( q )$ for each $q \in \mathcal { Q } .$   
9: for all $\grave { q } \in \mathcal { Q }$ do   
10: Sample $y \sim \pi _ { S } ( \cdot \mid q )$ under budget $L _ { \mathrm { m a x } } ;$ let $L = | y | .$   
11: At each position t, evaluate $p _ { S } = \pi _ { S } ( \cdot \mid q , y _ { < t } )$ and $\dot { p } _ { T } = \pi _ { T } ( \cdot \mid q , y _ { < t } ) .$   
12: Set $A _ { t } \doteq \mathrm { T o p K } ( p _ { S } ) , B _ { t } = \mathrm { \bar { T o p K } } ( p \mathrm { \bar { \it T } } ) \quad$ , and ${ \tilde { U _ { t } } } = A _ { t } \cup B _ { t } .$   
13: Normalize p<sub>S</sub> and p<sub>T</sub> over $U _ { t }$ to obtain p¯<sub>S</sub> and ${ \bar { p } } _ { T } .$   
14: Compute $\begin{array} { r } { \delta _ { t } = \mathrm { K L } ( \bar { p } _ { T } \| \bar { p } _ { S } ) , \omega _ { t } = \sum _ { v \in A _ { t } } p _ { T } ( v ) , \mathrm { a n d } s _ { t } = \operatorname* { m i n } ( \delta _ { t } / ( \delta _ { \mathrm { r e f } } + \epsilon ) , 1 ) \omega _ { t } . } \end{array}$   
15: Compute $P _ { \mathrm { l e n } } ( q ) = 1 - \lambda _ { \mathrm { l e n } }$ min $( L / \bar { L _ { \mathrm { m a x } } } , 1 )$ and $P _ { \mathrm { d i v } } ( q ) = \mathrm { m a x } ( 0 , 1 - \lambda _ { \mathrm { d i v } }$ share(q)).   
16: Assign question reward $\begin{array} { r } { R ( q ) = P _ { \mathrm { l e n } } ( q ) P _ { \mathrm { d i v } } ( q ) L ^ { - \alpha } \sum _ { t = 1 } ^ { L } s _ { t } . } \end{array}$   
17: end for   
18: Compute reward mean $\mu _ { b }$ and standard deviation $\sigma _ { b }$ within each group $\mathcal { Q } _ { b } .$   
19: Set $\widehat { A } _ { b , i } = ( R ( q _ { b , i } ) - \mu _ { b } ) / ( \sigma _ { b } + \epsilon ) .$   
20: Update only $S _ { \theta }$ with GRPO, applying $\widehat { A } _ { b , i }$ to the generated tokens of question ${ { q } _ { b , i } }$   
21: Save synthesizer checkpoints at the prescribed interval.   
22: end for   
23: Stage 2: Construct the synthetic question bank   
24: Select a synthesizer checkpoint $S _ { \theta ^ { \star } } .$   
25: Sample $\dot { N }$ questions from $\setminus _ { \theta ^ { \star } } ( \cdot \mid x )$ under budget $L _ { q } .$   
26: Apply format filtering and remove malformed, incomplete, or answer-leaking outputs.   
27: Apply lexical deduplication using token-set Jaccard similarity.   
28: Apply semantic deduplication using the frozen Qwen3-Embedding-4B model.   
29: Store the retained questions as the prompt-only bank $\mathcal { D } .$   
30: Stage 3: Distill the student on the synthetic bank   
31: Unfreeze $\pi _ { S }$ and keep $\pi _ { T }$ frozen.   
32: for $e = 1 , \ldots , E$ do   
33: for all minibatches $B \subset \mathcal { D }$ do   
34: Snapshot the current student as $\pi _ { \mathrm { o l d } } .$   
35: For each $q \in B ,$ sample $n _ { \mathrm { r o l l } }$ responses from $\pi _ { \mathrm { o l d } } ( \cdot \mid q )$ under budget $L _ { \mathrm { m a x } } .$   
36: At each response position t, evaluate $p _ { S } = \pi _ { \mathrm { o l d } } ( \cdot \mid q , y _ { < t } )$ and $p _ { T } = \pi _ { T } ( \cdot \mid q , y _ { < t } )$   
37: Set $A _ { t } = \hat { \mathrm { T o p K } } ( \hat { p } _ { S } )$ and w<sub>t</sub>(v) = p<sub>S</sub>(v)/ P p<sub>S</sub>(u) for $) \in A _ { t } .$   
38: Compute per-candidate advantages $r _ { t } ( v ) = w _ { t } \bar { ( v ) } \mathrm { l o g } ( p _ { T } ( v ) / p _ { S } ( v ) ) , v \in A _ { t } .$   
39: Update $\pi _ { S }$ with the PPO-clipped objective using $r _ { t } ( v )$ directly as fixed advantages, summing over candidates and averaging   
over valid response tokens.   
40: end for   
41: end for   
42: return $\mathcal { D } , \pi _ { \boldsymbol { S } } .$

## C. Prompt Templates

We provide the synthesizer and student-solving templates below. Both mathematical and code question synthesis use fixed instructions without persona conditioning or few-shot example problems. In the templates, {q} denotes the question to be solved. Line wrapping is adjusted for readability.

## SYNTHESIZER / MATH

SYSTEM

FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial problem. The problem could come from any field of mathematics, including but not limited to algebra, geometry, number theory, combinatorics, prealgebra, probability, statistics, and calculus. Aim for a difficulty such that fewer than 30% of advanced high-school students could solve it. Avoid re-using textbook cliches or famous contest problems.

The problem must be self-contained, have a well-defined answer, and require several non-obvious reasoning steps that build on each other, not a single formula lookup.

THEN, without revealing any of your private thoughts, output ONLY the problem statement. Do not include a solution, an answer, or any commentary.

USER

Generate one new, challenging mathematical reasoning problem now.

## SYNTHESIZER / CODE

## SYSTEM

FIRST, in your private scratch-pad, think step-by-step to design a brand-new, non-trivial competitive programming problem on your own subject. Aim for a difficulty such that a strong contestant would need several non-obvious algorithmic insights, not a single library call. Avoid re-using classic textbook tasks or well-known contest problems.

The problem must be self-contained and must state its Input format, its Output format, and the constraints on the input.   
The program that solves it reads from standard input and writes to standard output.

THEN, without revealing any of your private thoughts, output ONLY the problem statement. Do not include a solution, sample code, an answer, or any commentary.

USER

Generate one new, challenging competitive programming problem now.

## Student-solving templates

The following templates are user messages. Mathematical reasoning uses the stage-specific formats below; the code template is shared by reward computation, discrepancy calibration, code OPD banks, and LiveCodeBench evaluation.

MATH / Reward computation and discrepancy calibration

USER

{q}

Please reason step by step, and put your final answer within \boxed{}.

## MATH / Downstream OPD on synthetic questions

USER

Solve the following math problem step by step. The last line of your response should be of the form Answer: \$Answer (without quotes) where \$Answer is the answer to the problem.

{q}

## MATH / Vanilla OPD on real questions

USER

Solve the following math problem step by step. The last line of your response should be of the form Answer: \$Answer (without quotes) where \$Answer is the answer to the problem.

{q}

Remember to put your answer on its own line after "Answer:".

## MATH / Evaluation suffix

Append to the existing evaluation prompt in the real-question format above.

Please reason step by step, and put your final answer within \boxed{}.

## CODE / Student solving

USER

You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.

Question: {q}

Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.

\`\`\`python   
# YOUR CODE HERE