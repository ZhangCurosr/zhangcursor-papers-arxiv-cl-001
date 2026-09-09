# Entropy-Regularized Rank-Masked Policy Optimization for Test-Time Reinforcement Learning in Code Generation

Jiacheng Xu, Feng Chen, Xiuneng Xu, Bo An Nanyang Technological University, Singapore jiacheng005@e.ntu.edu.sg, boan@ntu.edu.sg

## Abstract

Existing methods for test-time reinforcement learning (TTRL) derive rewards from answerlevel self-voting on unlabeled test-time tasks with canonical answers, but this breaks down for code generation because programs cannot be compared by surface form and therefore do not directly provide a usable training signal. To make TTRL applicable to code generation, we propose probe-driven TTRL, which constructs output-free probe inputs from the problem statement, executes candidate programs on these probes, and defines a Probe Consensus Reward (PCR) from the resulting behavioral agreement. PCR provides a behavioral training signal for open-vocabulary programs, but it is not a fully reliable verifier and remains susceptible to reward hacking through spurious consensus. We therefore introduce Entropy-Regularized Rank-Masked Policy Optimization (ERPO), which converts low PCR into conservative negative updates through rank masking and controls policy drift with an entropy ceiling. On coding benchmarks, ERPO substantially improves pass@1 and pass@k in both in-domain adaptation and zero-shot transfer.

## 1 Introduction

As large language models are increasingly trusted with critical tasks, deployment-time response quality has become a central concern (Aghazadeh-Chakherlou et al., 2026). A common way to improve quality at test time is inference-time scaling, which spends additional computation without changing model parameters, typically by sampling multiple solutions and selecting one through voting, verification, or search (Wang et al., 2023; Shi et al., 2022; Li et al., 2025). Such search can be effective, but it is expensive and increases response latency when high-quality answers require many samples. Test-time reinforcement learning (TTRL) offers a complementary direction: it converts testtime signals into rewards and adapts the model on unlabeled test-time tasks, aiming to improve single-pass response quality at deployment time (Zuo et al., 2025; Liao et al., 2026; Zhou et al., 2025). This setting is most natural for tasks such as mathematical reasoning, where sampled solutions can be reduced to directly comparable canonical answers and the majority answer can serve as a pseudo-label (Wang et al., 2023; Zuo et al., 2025).

For code generation, however, model outputs are open-vocabulary programs rather than canonical final answers, so exact-match voting cannot be directly used to construct a reward (Li et al., 2022; Shi et al., 2022). Prior training-free inference-time search methods avoid exact matching by comparing execution behavior rather than program text, using execution traces or functional consensus to compare candidate programs (Shi et al., 2022; Chen et al., 2023; Launer et al., 2026). Turning this behavioral comparison into a TTRL reward is nontrivial because test-time adaptation lacks the inputs needed to evaluate generated programs. The model must therefore construct inputs itself and turn the resulting outputs into a reward signal suitable for reinforcement learning. A further challenge is that existing TTRL objectives often emphasize pass@1, which can improve single-pass performance and reduce the need for costly training-free search, but can also weaken the pass@k frontier when the gain comes from reduced solution diversity (Zhou et al., 2025). This trade-off is especially undesirable in quality-sensitive applications, where pass@k serves as a practical upper bound on what trainingfree inference-time scaling can extract from the model (Yue et al., 2025; Snell et al., 2025).

![](images/4e330965a26834f54a01a0a631e1e953100e1ea549ea79742077dfe4e21d048a.jpg)

![](images/33524b602e7af72018bae4091c6e70f4fc48a836e96f3cddc2a333fcde0ad5ae.jpg)

![](images/22756130eb691a72968669b3a29f86c92cb0ac72dc724ed663b5727f9764632e.jpg)  
Figure 1: Performance overview. Left: Qwen3-8B per-step LCB results. Middle: Qwen3-4B mean transfer performance across three target benchmarks. Right: Qwen3-4B LCB inference-time scaling.

To make TTRL applicable to code generation while improving both single-pass accuracy and the pass@k frontier, we propose probe-driven testtime reinforcement learning. For each coding problem, the model constructs a set of problemspecific, output-free probe inputs from the problem statement. It then executes sampled programs on these inputs and defines a Probe Consensus Reward (PCR) from the induced output distributions. PCR turns open-vocabulary programs into comparable training samples without relying on oracle outputs. However, because PCR is built from generated inputs and consensus over candidate outputs, it is not a fully reliable verifier. High consensus may reflect a correct solution, but it may also arise from a common bug or from probes that miss the decisive corner case (Tambon et al., 2025; He et al., 2025). Directly optimizing PCR can therefore trigger reward hacking and degrade the pass@k frontier while improving pass@1.

Motivated by this observation, we propose Entropy-Regularized Rank-Masked Policy Optimization (ERPO). Our analysis shows that the PCR signal is asymmetric: low-consensus programs are easier to identify as poor candidates than high-consensus programs are to certify as correct. Rather than treating high-PCR programs as positive labels, ERPO uses PCR mainly as a negative signal. Within each candidate group, it masks out the high-ranked half. The remaining lower-ranked samples typically have negative group-normalized advantages, so the update reduces the likelihood of low-consensus programs while avoiding direct reinforcement of potentially spurious majorities. A fixed entropy-ceiling regularizer further controls upward entropy drift and preserves stable sampling behavior for pass@k.

We summarize our contributions as follows:

1. We formulate probe-driven TTRL for code generation, where output-free inputs generated from the problem statement provide a behavioral interface for rewards without oracle tests or oracle outputs.

2. We analyze the noise pattern of probe-based PCR rewards and show why direct optimization can be brittle: behavioral consensus is more reliable as a negative signal than as a correctness certificate.

3. We propose ERPO, a policy optimization algorithm that combines negative rank-masked PCR updates with fixed entropy-ceiling control, improving both pass@1 and pass@k with gains that transfer across coding benchmarks.

## 2 Preliminaries

## 2.1 Test-Time Reinforcement Learning

Reinforcement learning with verifiable rewards (RLVR) (Guo et al., 2025; Yu et al., 2025) optimizes a policy using scalar rewards from an external verifier. Given an input $x$ from a training distribution D and a sampled output $y \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid x )$ the verifier produces $r ( x , y )$ , which is converted into an advantage estimate $\hat { A } ( x , y )$ . Abstracting away clipping, normalization, and regularization, a generic policy-gradient objective is

$$
\mathcal { I } _ { \mathrm { R L V R } } ( \theta ) = { \bf E } \left[ \hat { A } ( x , y ) \log \pi _ { \theta } ( y \mid x ) \right] .
$$

Test-time reinforcement learning (TTRL) applies this idea during inference, where a model adapts to unlabeled test-time tasks rather than a fixed offline training set. Because a trusted verifier is unavailable, TTRL must construct a surrogate reward $\tilde { r } ( x , y )$ . For canonical problems such as math, this reward is often derived from answer-level selfvoting: given G sampled solutions $\{ y _ { i } \} _ { i = : } ^ { G }$ with extracted answers $a _ { i } = \operatorname { A n s } ( y _ { i } )$ , define

$$
a ^ { \star } = \mathrm { m o d e } \{ a _ { i } \} _ { i = 1 } ^ { G } ,
$$

$$
\tilde { r } _ { \mathrm { v o t e } } ( x , y _ { i } ) = \left\{ \begin{array} { l l } { 1 , } & { a _ { i } = a ^ { \star } , } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$

![](images/8a184630a8d1d6ebaa9955a33f5f87f26f79678185329eaa34a85fabaa4e0c5e.jpg)  
Figure 2: Overview of ERPO. Our method has three stages: (1) generating output-free probe inputs for each problem; (2) executing sampled candidates on the shared probes and scoring them by behavioral agreement (PCR); (3) a rank-masked, entropy-regularized update that uses PCR primarily to suppress low-consensus candidates.

This construction assumes that outputs expose comparable final answers. For open-vocabulary outputs such as code, semantically equivalent programs may have unrelated surface forms, so reward construction must compare program behavior rather than answer identity.

## 2.2 RLVR for Code Generation

In code generation, the model receives a naturallanguage problem description P and produces a program $C$ that should satisfy the problem specification. A standard verifier evaluates $C$ against a test suite $\mathcal { T } = \{ ( I _ { j } , O _ { j } ) \} _ { j = 1 } ^ { m }$ , where $I _ { j }$ is a test input and $O _ { j }$ is the corresponding oracle output. When an ideal test suite $\mathcal { T } ^ { \star }$ is available, the code reward can be written as (Yu et al., 2025)

$$
R _ { \mathrm { c o d e } } ( C , \mathcal { T } ^ { \star } ) = \left\{ \begin{array} { l l } { 1 , } & { \forall ( I _ { j } , O _ { j } ) \in \mathcal { T } ^ { \star } , } \\ { ~ \mathrm { E x e c } ( C , I _ { j } ) = O _ { j } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

Here ${ \mathrm { E x e c } } ( C , I _ { j } )$ denotes the output produced by executing program $C$ on input $I _ { j }$ . This formulation makes test quality central to training and evaluation: the reward is reliable only when the test suite is complete and the oracle outputs are trusted.

When TTRL is applied to code generation, however, such a trusted test suite is not available for the target problem. The problem does not provide test inputs and corresponding oracle outputs from which to obtain a reliable reward signal. As a result, applying the standard code reward is impossible.

The central challenge is to build a surrogate reward that can compare candidate programs without private test cases or ground-truth outputs, while providing a useful signal for test-time policy updates.

## 3 Method

Overview. We study test-time reinforcement learning for code generation, where a model adapts to target coding problems without reliable inputoutput tests for full program verification. Beyond missing labels, code generation lacks canonical answers for self-voting: generated programs are open-vocabulary outputs that cannot be compared by surface form for majority voting. The model must therefore construct training signals from program behavior, instead.

Our method addresses this difficulty in three stages. For each problem, we first generate a set of distinct, output-free probe inputs from the problem statement, as described in Section 3.1. Because corresponding oracle outputs are unavailable, these probes expose behavior rather than verify correctness. During training or evaluation, we sample multiple candidate programs, execute them on the shared probes, and compute a Probe Consensus Reward (PCR) from output consensus (Section 3.2). Finally, we introduce Entropy-Regularized Rank-Masked Policy Optimization (ERPO). Instead of directly maximizing this noisy reward, ERPO applies a rank-masked negative update to the lower-

PCR part of each candidate group, while assigning high-PCR responses zero policy-gradient weight. The full objective is given in Section 3.3.

## 3.1 Probe Input Generation

Let ${ \mathcal { D } } = \{ x _ { i } \} _ { i = 1 } ^ { m }$ be the target set of coding problems. For each problem $x ,$ the model observes the natural-language problem statement without access to any test suite for reliable evaluation. Because generated programs are open-vocabulary text outputs and do not support consistency voting by surface form, we compare programs by execution behavior on shared inputs.

Unlike supervised code RL with input-output test suites for verifying code correctness, these shared inputs must be constructed at adaptation time and have no oracle outputs. Before optimization, for each problem x, we ask the model to generate $K _ { x }$ raw test inputs and form the probe set $\mathcal { T } _ { x }$ $\{ t _ { j } \} _ { j = 1 } ^ { K _ { x } }$ . This set is generated once and reused across training steps. Generation proceeds iteratively: at iteration $j ,$ the prompt contains the problem statement and all previously generated inputs $\{ t _ { 1 } , \dotsc , t _ { j - 1 } \}$ . The model produces one new input $t _ { j }$ that is format-correct, constraint-satisfying, and structurally different from the previous ones. The prompt template is provided in Appendix B. Thus $\mathcal { T } _ { x }$ encourages distinct execution regimes and provides label-free common execution points for behavioral comparison, not tests with trusted outputs. The reward below depends only on candidate outputs induced by $\mathcal { T } _ { x }$ , without any reference outputs.

## 3.2 Probe Consensus Reward

Execution-based consensus has been used for code selection and functional voting. Here, we adopt this idea as a test-time training reward over generated, output-free probes. Given a problem $x ,$ we sample a group of G candidate programs $\{ C _ { i } \} _ { i = 1 } ^ { G }$ from the current policy. For each probe input $t _ { j } \in \mathcal { T } _ { x }$ , we execute each candidate $C _ { i }$ and record its result $z _ { i j }$ If execution fails, the candidate receives no credit for that input and is excluded from that input’s majority computation.

For each probe input $t _ { j }$ , we count only valid outputs. The count

$$
n _ { j } ( o ) = \sum _ { i = 1 } ^ { G } { \bf 1 } \{ z _ { i j } = o \}
$$

is the number of candidates that produce output o

on $t _ { j }$ . The tied majority set is

$$
{ \mathcal { M } } _ { j } = \{ o : n _ { j } ( o ) = \operatorname* { m a x } _ { o ^ { \prime } } n _ { j } ( o ^ { \prime } ) \} .
$$

Ties are credited to all tied outputs.

Candidate $C _ { i }$ receives the PCR score

$$
s _ { i } = \frac { 1 } { K _ { x } } \sum _ { j = 1 } ^ { K _ { x } } \mathbf { 1 } \{ z _ { i j } \in \mathcal { M } _ { j } \} ,
$$

which is the fraction of probe inputs for which its output belongs to the majority set. Thus $s _ { i } \in [ 0 , 1 ]$ measures behavioral agreement with the candidate group on the generated probes, without oracle outputs or any claim that the majority output is correct.

PCR is useful but not a correctness verifier. A high score can still correspond to an incorrect program when many candidates share a bug or the probes miss the relevant corner case. Conversely, a correct program can receive a low score if it follows a rare but valid behavior while most sampled programs agree on a wrong one. Our policy optimization algorithm therefore uses PCR conservatively.

## 3.3 Entropy-Regularized Rank-Masked Policy Optimization Algorithm

A direct GRPO update on PCR scores would reinforce high-scoring candidates and penalize lowscoring ones. This is unsafe because high PCR is not a reliable correctness certificate. In practice, direct optimization can initially improve pass@1 by reducing sampling diversity, but continued training can overfit to the reward and degrade the upperbound performance captured by pass@k. ERPO therefore does not treat high-PCR candidates as targets. Instead, it uses PCR through a rank mask and fixed entropy-ceiling regularization.

For a group of G candidates from the same problem, let $s _ { i }$ be the PCR score defined above. We first compute the standard group-normalized GRPO advantage (Shao et al., 2024):

$$
A _ { i } = \frac { s _ { i } - \mu _ { x } } { \sigma _ { x } } , \quad \mu _ { x } = \frac { 1 } { G } \sum _ { i = 1 } ^ { G } s _ { i } ,
$$

where $\sigma _ { x }$ is the within-group standard deviation of the PCR scores. This centers the signal within each problem, so the update depends on relative behavioral support rather than the absolute PCR scale.

ERPO then masks the GRPO advantage by rank. Within each group, candidates are ranked by $A _ { i }$ in descending order. The top $\lceil G / 2 \rceil$ candidates are masked out, and the loss keeps only the signed advantages of the lower-ranked half:

$$
\widetilde A _ { i } = A _ { i } \cdot { \mathbf 1 } \{ \mathrm { r a n k } _ { x } ^ { \ast } ( i ) > \lceil G / 2 \rceil \} .
$$

Here $\mathrm { r a n k } _ { x } ^ { \downarrow } ( i ) ~ = ~ 1$ denotes the candidate with the largest advantage within problem $x .$ This mask turns PCR into a conservative negative signal. High-PCR candidates are excluded from positive policy-gradient updates, while the retained lowerranked candidates usually have $A _ { i } < 0$ and thus reduce the likelihood of low-consensus programs.

The policy component of ERPO is the rankmasked PCR loss $\mathcal { L } _ { \mathrm { P C R } }$ . Ignoring clipping for notation, its gradient direction is proportional to

$$
- \sum _ { i = 1 } ^ { G } \widetilde { A } _ { i } \sum _ { t = 1 } ^ { | C _ { i } | } \nabla _ { \theta } \log \pi _ { \theta } ( C _ { i , t } \mid x , C _ { i , < t } ) .
$$

Here $\mathit { C } _ { i , < t }$ denotes the prefix of candidate program $C _ { i }$ before token t.

Rank-masked updates can still destabilize training when they drive policy entropy upward without bound. We therefore add a fixed entropy-ceiling regularizer. Let $H _ { \theta }$ be the token-level policy entropy averaged over the current actor update, and let $H _ { \mathrm { c e i l } }$ be a fixed entropy ceiling chosen before adaptation. We penalize only violations of this ceiling:

$$
\mathcal { L } _ { H } = \lambda _ { H } \operatorname* { m a x } \left( 0 , H _ { \theta } - H _ { \mathrm { c e i l } } \right) ^ { 2 } .
$$

Because the regularizer is inactive below $H _ { \mathrm { c e i l } }$ and grows quadratically above it, it leaves lowerentropy updates unpenalized and suppresses excessive entropy growth. Putting the two components together, ERPO minimizes the following objective, with the update summarized in Algorithm 1:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { E R P O } } = \mathcal { L } _ { \mathrm { P C R } } + \mathcal { L } _ { H } . } \end{array}
$$

## 4 Experiments

## 4.1 Experimental Setup

We use Qwen3-4B and Qwen3-8B as base models (Yang et al., 2025). The main experiments adapt on LiveCodeBench (LCB) (Jain et al., 2025) and evaluate zero-shot transfer to CodeContests (CC) (Li et al., 2022), CodeForces (CF) (Wang et al., 2025), and TACO (Li et al., 2023). As unlabeled-reward baselines, $\mathrm { G R P O } _ { \mathrm { P C R } }$ and $\mathrm { N S R } _ { \mathrm { P C R } }$ access only problem statements, without provided test cases, and update the model with GRPO or NSR using PCR rewards (Shao et al., 2024; Zhu et al., 2025).

Algorithm 1 Entropy-Regularized Rank-Masked   
Policy Optimization (ERPO)   
1: Input: problem set D, precomputed probe sets   
$\{ \mathcal { T } _ { x } \} _ { x \in \mathcal { D } } ,$ policy $\pi _ { \theta } ,$ group size $G ,$ entropy  
ceiling parameters $( \lambda _ { H } , H _ { \mathrm { c e i l } } )$   
2: for each training step do   
3: Sample problem $x \sim \mathcal { D }$   
4: Retrieve the precomputed probe set $\mathcal { T } _ { x }$   
5: Set $\pi _ { \theta _ { \mathrm { o l d } } }  \pi _ { \theta }$   
6: Sample programs $C _ { i } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid x )$ for   
$i = 1 , \dots , G$   
7: Execute $\{ C _ { i } \} _ { i = 1 } ^ { G }$ on the probe set $\mathcal { T } _ { x }$   
8: Compute PCR scores $s _ { i }$ for $i = 1 , \dots , G$   
9: Compute group advantages $\{ A _ { i } \} _ { i = 1 } ^ { G }$   
10: Apply the rank mask to obtain $\{ \widetilde { A } _ { i } \} _ { i = 1 } ^ { G }$   
11: Compute $\mathcal { L } _ { \mathrm { E R P O } } = \mathcal { L } _ { \mathrm { P C R } } + \mathcal { L } _ { H }$   
12: Update θ by one gradient step on L<sub>ERPO</sub>   
13: end for   
14: Output: updated policy π<sub>θ</sub>

In particular, $\mathrm { N S R } _ { \mathrm { P C R } }$ uses PCR as a negative-only signal by penalizing low-score samples. ERPO follows the same setting, using the objective in Section 3.3. We also include ${ \mathrm { G R P O } } _ { \mathrm { P u b } } .$ a more informative baseline that observes the complete inputs and outputs of public test cases at test time for reward computation. This verifier is reliable but incomplete: failing a public test implies incorrectness, but passing all public tests may still fail hidden tests. Implementation details for all methods are provided in Appendix A.

Evaluation protocol. We report two settings. Indomain adaptation updates only on unlabeled problem statements from the target benchmark: during training, the model never observes any evaluation test input and its oracle output. The complete evaluation suite is used only post hoc to compute pass@k (P@k) (Chen et al., 2021). The transfer experiment evaluates reusability: we adapt once on unlabeled LCB problems, then test the checkpoint on CC, CF, and TACO without target-benchmark updates. This setting distinguishes benchmark-specific optimization from broader capability gains across the code-generation domain. Unless otherwise noted, we use the final checkpoint from the prespecified schedule, without access to hidden-test pass@k.

## 4.2 Main Results

Table 1 shows ERPO’s clear advantage. Indomain on LCB, ERPO improves both pass@1 and pass@16, raising single-pass accuracy and the pass@k frontier, i.e., the coverage of correct solutions under repeated sampling. Its gains also transfer to CC, CF, and TACO, showing crossbenchmark generalization. $\mathbf { G R P O } _ { \mathrm { P C R } }$ , which directly optimizes PCR, performs worst: on LCB it obtains only a small pass@1 gain with a large pass@16 drop, and the pass@1 gain disappears on transfer benchmarks, suggesting overfitting to noisy PCR patterns. $\mathrm { G R P O } _ { \mathrm { P u b } }$ , with ground-truth public tests, is second-best for pass@1, but its pass@16 gains are marginal and its 8B transfer gains barely persist, indicating reward hacking on the adaptation source. NSR improves pass@16 but shows little pass@1 gain because repeated penalties on negative samples sharply increase entropy, so its pass@16 gain mainly comes from diversity. As shown in the appendix, $\mathrm { N S R } _ { \mathrm { P C R } }$ soon suffers entropy explosion and policy collapse. ERPO is therefore the only adaptation method that substantially improves both pass@1 and pass@16, using a small amount of fully unlabeled test-time data to exploit the PCR signal while avoiding its noisy patterns.

Table 1: Main code-generation results. Mean averages the three transfer targets. Values are percentages.
<table><tr><td rowspan="3">Model</td><td rowspan="3">Method</td><td colspan="2">In-Domain Adapt.</td><td colspan="8">Zero-Shot Transfer</td></tr><tr><td colspan="2">LCB</td><td colspan="2">CC</td><td colspan="2">CF</td><td colspan="2">TACO</td><td colspan="2">Mean</td></tr><tr><td>P@1</td><td>P@16</td><td>P@1</td><td>P@16</td><td>P@1</td><td>P@16</td><td>P@1</td><td>P@16</td><td>P@1</td><td>P@16</td></tr><tr><td>Qwen3-4B</td><td>Base</td><td>26.0</td><td>34.4</td><td>25.1</td><td>42.3</td><td>9.8</td><td>21.8</td><td>6.4</td><td>13.1</td><td>13.8</td><td>25.7</td></tr><tr><td></td><td> $\mathrm { G R P O } _ { \mathrm { P u b } }$ </td><td>33.8</td><td>35.4</td><td>26.2</td><td>41.0</td><td>10.7</td><td>24.4</td><td>6.1</td><td>13.6</td><td>14.3</td><td>26.3</td></tr><tr><td></td><td> $\mathrm { G R P O } _ { \mathrm { P C R } }$ </td><td>27.8</td><td>30.9</td><td>26.0</td><td>36.4</td><td>8.5</td><td>20.1</td><td>5.5</td><td>9.8</td><td>13.3</td><td>22.1</td></tr><tr><td></td><td> $\mathrm { N S R } _ { \mathrm { P C R } }$ </td><td>25.7</td><td>39.4</td><td>27.4</td><td>51.0</td><td>10.8</td><td>32.1</td><td>6.6</td><td>16.6</td><td>14.9</td><td>33.2</td></tr><tr><td></td><td>ERPO</td><td>36.7</td><td>46.3</td><td>42.1</td><td>58.2</td><td>20.7</td><td>38.5</td><td>13.3</td><td>21.3</td><td>25.4</td><td>39.3</td></tr><tr><td>Qwen3-8B</td><td>Base</td><td>25.6</td><td>34.6</td><td>27.2</td><td>45.6</td><td>9.3</td><td>22.5</td><td>7.2</td><td>14.6</td><td>14.6</td><td>27.6</td></tr><tr><td></td><td> $\mathrm { G R P O } _ { \mathrm { P u b } }$ </td><td>32.1</td><td>37.1</td><td>28.3</td><td>41.8</td><td>11.5</td><td>27.2</td><td>7.6</td><td>13.2</td><td>15.8</td><td>27.4</td></tr><tr><td></td><td> $\mathrm { G R P O } _ { \mathrm { P C R } }$ </td><td>26.3</td><td>29.1</td><td>27.2</td><td>41.0</td><td>10.1</td><td>22.1</td><td>7.0</td><td>13.1</td><td>14.8</td><td>25.4</td></tr><tr><td></td><td> $\mathrm { N S R } _ { \mathrm { P C R } }$ </td><td>25.9</td><td>39.4</td><td>28.8</td><td>52.3</td><td>10.3</td><td>30.0</td><td>7.1</td><td>17.3</td><td>15.4</td><td>33.2</td></tr><tr><td></td><td>ERPO</td><td>35.1</td><td>50.9</td><td>39.4</td><td>57.7</td><td>17.5</td><td>37.3</td><td>11.8</td><td>21.3</td><td>22.9</td><td>38.8</td></tr></table>

## 4.3 Analysis of the PCR Score

We analyze PCR to clarify when the surrogate reward is reliable and how its noise affects policy optimization. Table 2 compares PCR with hiddentest correctness on LiveCodeBench. Hidden tests are used only for analysis and are unavailable during test-time adaptation. PCR has useful ranking power, with AUC 0.790 for Qwen3-4B and 0.750 for Qwen3-8B, but its calibration is highly asymmetric. Low PCR is a strong negative signal: candidates with $s \leq 0 . 5$ fail hidden tests more than

Table 2: PCR score s calibration on LiveCodeBench. y<sub>hid</sub> denotes hidden-test correctness. Fail rate is the fraction of candidates that do not pass hidden tests.
<table><tr><td>Metric</td><td>Qwen3-4B</td><td>Qwen3-8B</td></tr><tr><td> $\mathsf { A U C } ( \boldsymbol { s } _ { \mathrm { P C R } } , \boldsymbol { y } _ { \mathrm { h i d } } )$ </td><td>0.790</td><td>0.750</td></tr><tr><td>Fail rate among  $s \leq 0 . 5$ </td><td>92.0%</td><td>91.3%</td></tr><tr><td>Fail rate among  $s \leq 0 . 2 5$ </td><td>91.1%</td><td>91.0%</td></tr><tr><td>Fail rate among  $s \geq 0 . 9$ </td><td>50.7%</td><td>54.6%</td></tr><tr><td>Fail rate among  $s = 1 . 0$ </td><td>44.2%</td><td>51.9%</td></tr></table>

91% of the time for both model sizes. High PCR, however, is not a correctness certificate: even candidates with $s = 1 . 0$ are wrong 44.2% of the time for Qwen3-4B and 51.9% for Qwen3-8B.

This asymmetry explains why direct PCR optimization is brittle. A GRPO update can turn relatively high-PCR candidates into positive targets, reinforcing consensual wrong programs when a group contains no hidden-correct solution. ERPO uses PCR mainly as a conservative negative signal, penalizing low-consensus programs while avoiding positive updates on high-consensus false positives.

## 4.4 Inference-Time Scaling

For test-time training, pass@k is crucial (Chen et al., 2021) because training-free inference-time search needs many samples to obtain stronger submissions and is bounded by the underlying pass@k frontier. We evaluate inference-time scaling for checkpoints adapted on each of LiveCodeBench and CodeContests by sampling 256 solutions per problem and reporting pass@k as k varies (Table 3). ERPO substantially improves pass@k on both datasets at every k, showing better single-pass accuracy and a stronger frontier for test-time search.

Table 3: Inference-time pass@k on Qwen3-4B. All values are percentages.
<table><tr><td>Method</td><td>P@1</td><td>P@2</td><td>P@4</td><td>P@8</td><td>P@16</td><td>P@32</td><td>P@64</td><td>P@128</td><td>P@256</td></tr><tr><td colspan="10">LiveCodeBench</td></tr><tr><td>Base</td><td>26.0</td><td>28.8</td><td>31.1</td><td>32.9</td><td>34.4</td><td>35.9</td><td>37.3</td><td>38.6</td><td>40.1</td></tr><tr><td> $\mathrm { G R P O } _ { \mathrm { P u b } }$ </td><td>32.4</td><td>33.6</td><td>34.5</td><td>35.2</td><td>35.6</td><td>35.9</td><td>36.1</td><td>36.3</td><td>36.6</td></tr><tr><td> $\mathrm { G R P O } _ { \mathrm { P C R } }$ </td><td>27.3</td><td>28.0</td><td>28.6</td><td>29.1</td><td>29.6</td><td>30.2</td><td>30.8</td><td>31.4</td><td>31.9</td></tr><tr><td> $\mathrm { N S R } _ { \mathrm { P C R } }$ </td><td>25.0</td><td>29.7</td><td>33.3</td><td>36.0</td><td>38.4</td><td>40.9</td><td>43.3</td><td>45.7</td><td>47.9</td></tr><tr><td>ERPO</td><td>36.2</td><td>39.2</td><td>41.8</td><td>44.4</td><td>46.6</td><td>48.2</td><td>49.6</td><td>50.9</td><td>51.9</td></tr><tr><td colspan="10">CodeContests</td></tr><tr><td>Base</td><td>25.0</td><td>29.3</td><td>33.5</td><td>37.5</td><td>41.2</td><td>44.6</td><td>47.3</td><td>49.2</td><td>50.3</td></tr><tr><td> $\mathrm { G R P O } _ { \mathrm { P u b } }$ </td><td>42.2</td><td>43.3</td><td>44.2</td><td>45.0</td><td>45.6</td><td>46.1</td><td>46.6</td><td>47.1</td><td>47.7</td></tr><tr><td>GRPOPCR</td><td>27.1</td><td>28.6</td><td>30.4</td><td>32.5</td><td>34.7</td><td>36.7</td><td>38.4</td><td>39.8</td><td>40.7</td></tr><tr><td>NSRPCR</td><td>27.3</td><td>34.8</td><td>41.1</td><td>46.3</td><td>50.9</td><td>54.9</td><td>58.3</td><td>61.4</td><td>64.1</td></tr><tr><td>ERPO</td><td>41.5</td><td>46.5</td><td>50.5</td><td>53.8</td><td>56.9</td><td>59.8</td><td>62.4</td><td>64.5</td><td>65.9</td></tr></table>

Table 4: Best-of-n accuracy of training-free search for Qwen3-4B on LiveCodeBench.
<table><tr><td>Method</td><td>CodeT-BoN RM-BoN</td></tr><tr><td>Base, n=1 26.0</td><td>26.0</td></tr><tr><td>Base, n=4</td><td>28.6 29.7</td></tr><tr><td>Base, n=8 28.0</td><td>28.0</td></tr><tr><td>Base, n=16</td><td>28.0 28.0</td></tr><tr><td>Base, n=32</td><td>29.1 27.4</td></tr><tr><td colspan="2">ERPO 36.2 (single sample, no search cost)</td></tr></table>

In contrast, $\mathrm { G R P O _ { P u b } }$ and GRPO<sub>PCR</sub> severely sacrifice pass@k headroom. $\mathrm { N S R } _ { \mathrm { P C R } }$ also improves pass@k, but its gains grow with k and remain limited at pass@1, suggesting that they mainly reflect sampling diversity rather than higher single-pass accuracy.

On LiveCodeBench, ERPO’s pass@1 exceeds the training-free Base model’s pass@32, and on CodeContests exceeds Base pass@16. Existing inference-time search methods must select among candidates using a reward model or generated tests, which adds search cost and remains bounded by pass@k, whereas ERPO adapts the model itself and reduces the deployment search budget. The pass@k values in Table 3 are oracle upper bounds, assuming a perfect selector can pick any correct sample out of k, whereas deployment must submit one solution using only test-time information. Table 4 therefore reports Best-of-n accuracy on the Qwen3-4B base model with two representative selectors: CodeT-BoN (Chen et al., 2023), which samples M=32 LLM-generated (input, expected output) test cases per problem from the base model and ranks n candidate solutions by pass-pattern cluster size, and RM-BoN, which scores the n candidates with Skywork-Reward-Llama-3.1-8B-v0.2 (Liu et al., 2024). Both selectors stagnate below 30% even at $n = 3 2 .$ , whereas ERPO reaches 36.2% with one sample and no inference-time search, surpassing every Best-of-n configuration. This gap shows that current selectors cannot realize much of the oracle pass@k headroom, while test-time adaptation captures it directly.

![](images/7ca73d63dc6acca4c97c0049eb7ac5d28f413dc5dbcf6d53614896ac56bbb906.jpg)  
Figure 3: Entropy ceiling ablation on LiveCodeBench.

## 4.5 Hyperparameter Ablation

We ablate the ERPO entropy ceiling with $H _ { \mathrm { c e i l } } \in$ {0.2, 0.5, 1.0} on Qwen3-8B / LiveCodeBench, holding all other hyperparameters fixed as in Appendix A. Figure 3 shows per-step pass@1 (dashed) and pass@16 (solid), with $\mathrm { G R P O } _ { \mathrm { P u b } }$ as a non-PCR reference. All three values improve pass@1 and pass@16, indicating that the ERPO objective is the main driver of the gains. The ceiling affects training efficiency and final stability: lower values slow training, whereas an overly high value accelerates early pass@k improvement but degrades quickly after its peak. Thus, higher entropy does not necessarily imply a stronger pass@k frontier. We therefore fix $H _ { \mathrm { c e i l } } { = } 0 . 5$ , which best balances efficiency and stability, across all training experiments without model- or dataset-specific tuning. Ablations on the rank-mask percentile and generated probe count are provided in Appendices H and I, respectively.

## 5 Related Work

## 5.1 Unsupervised RLVR and Test-Time Reinforcement Learning

Reinforcement learning with verifiable rewards is most reliable when outputs can be scored by a trusted verifier (Yu et al., 2025). Unsupervised RLVR and test-time reinforcement learning weaken this assumption by deriving rewards from signals available at adaptation time, often through answerlevel agreement or self-consistency (Zuo et al., 2025; Wang et al., 2023; Liao et al., 2026; He et al., 2026). Such rewards are natural for tasks with canonical final answers, but code generation lacks this structure: correct programs can differ substantially in surface form, so reward construction must compare behavior rather than text.

Execution-based comparison offers a natural behavioral interface for code, and functional-voting or label-free code-training methods use candidate outputs on shared inputs to support consensus-based test-time training, supervision from sampled programs and tests, and co-evolution of coders and testers through execution feedback (Launer et al., 2026; Fan et al., 2026; Wang et al., 2025). Our work follows this execution-based direction but uses it for probe-driven TTRL: we construct outputfree probes from the problem statement, treat the induced agreement as a noisy PCR reward, and use ERPO to avoid directly reinforcing high-consensus samples whose correctness is not certified.

## 5.2 Inference-Time Scaling for Code Generation

Inference-time scaling improves code generation with more test-time compute, typically by sampling many candidate programs and selecting or revising them with an auxiliary execution signal (Snell et al., 2025). MBR-Exec selects programs at inference by behavioral agreement on test inputs (Shi et al., 2022). CodeT and TCS generate additional tests and use execution results for solution selection (Chen et al., 2023; Xu et al., 2026). AlphaCode and later systems scale code generation through large-scale sampling, generated tests, functional decomposition, pairwise execution comparisons, or cluster-based selection (Li et al., 2022; Chen et al., 2024; Li et al., 2025; Samadi et al., 2026).

These methods use execution to guide inferencetime generation and selection. Selection alone does not improve the model’s generation capability and is bounded by candidate coverage, measured by pass@k under the corresponding generation procedure (Chen et al., 2021, 2023). Test-time policy optimization can improve this coverage by changing the generation distribution, but may also reinforce incorrect consensus across updates, propagating errors beyond the current problem. PCR therefore uses execution not as a correctness certificate, but as a behavioral reward interface for test-time adaptation. ERPO optimizes under this interface with rank masking and fixed entropy-ceiling control, aiming to improve pass@1 while preserving the pass@k frontier and providing a stronger base model for current training-free inference-time scaling algorithms.

## 6 Conclusion

This work revisits test-time reinforcement learning from the perspective of deployment, where improving response quality by spending more computation on inference-time search is costly and can increase latency. TTRL is attractive because it adapts a model on unlabeled test-time tasks, but code generation exposes two obstacles that standard answerlevel voting does not address. Programs are openvocabulary outputs, so they do not provide directly comparable canonical answers for reward construction. At the same time, objectives that primarily improve pass@1 can weaken the pass@k frontier, limiting what quality-sensitive inference-time search can recover from the adapted model.

To address these issues, we propose probe-driven TTRL for code generation. By constructing outputfree probe inputs and defining a Probe Consensus Reward (PCR), our framework makes TTRL for code possible without oracle outputs or directly comparable responses. We further introduce Entropy-Regularized Rank-Masked Policy Optimization (ERPO), a policy optimization algorithm for this noisy reward, which substantially improves both pass@1 and pass@k. The adapted model can serve as a stronger test-time base policy: it supports lower-cost, higher-quality single-pass responses in latency-sensitive scenarios, while also providing a better performance frontier for search in quality-sensitive scenarios. These properties make probe-driven TTRL practically relevant for model deployment and worthy of broader study.

## Limitations

Our empirical evaluation is limited to competitiveprogramming-style coding benchmarks. The proposed framework, however, is not inherently restricted to this setting. In principle, it can be applied to broader code-generation tasks where candidate code can be executed and assessed through unit tests or other executable checks. We do not evaluate project-level code generation in this work because such settings require substantially more engineering infrastructure for reliable execution, training, and evaluation, along with much larger computational resources. We therefore leave project-level and industrial-scale code-generation scenarios as an important direction for future work.

## References

Robab Aghazadeh-Chakherlou, Qing Guo, Siddartha Khastgir, Peter Popov, Xiaoge Zhang, and Xingyu Zhao. 2026. A hierarchical imprecise probability approach to reliability assessment of large language models. Reliability Engineering & System Safety, 272:112615.

Bei Chen, Fengji Zhang, Anh Nguyen, Daoguang Zan, Zeqi Lin, Jian-Guang Lou, and Weizhu Chen. 2023. CodeT: Code generation with generated tests. In International Conference on Learning Representations (ICLR).

Jingchang Chen, Hongxuan Tang, Zheng Chu, Qianglong Chen, Zekun Wang, Ming Liu, and Bing Qin. 2024. Divide-and-conquer meets consensus: Unleashing the power of functions in code generation. In Advances in Neural Information Processing Systems (NeurIPS), volume 37, pages 67061–67105.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, and 39 others. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Lishui Fan, Mouxiang Chen, Tingwei Zhu, Kui Liu, Xin Xia, Shanping Li, and Zhongxin Liu. 2026. ZeroCoder: Can LLMs improve code generation without ground-truth supervision? arXiv preprint arXiv:2604.07864.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. 2025. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. Nature, 645(8081):633–638.

Bingxiang He, Yuxin Zuo, Zeyuan Liu, Shangziqi Zhao, Zixuan Fu, Junlin Yang, Cheng Qian, Kaiyan Zhang, Yuchen Fan, Ganqu Cui, Xiusi Chen, Youbang Sun, Xingtai Lv, Xuekai Zhu, Li Sheng, Ran Li, Huanang Gao, Yuchen Zhang, Lifan Yuan, and 3 others. 2026. How far can unsupervised RLVR scale LLM training? In International Conference on Learning Representations (ICLR).

Zhongmou He, Yee Man Choi, Kexun Zhang, Jiabao Ji, Junting Zhou, Dejia Xu, Ivan Bercovich, Aidan Zhang, and Lei Li. 2025. HardTests: Synthesizing high-quality test cases for LLM coding. arXiv preprint arXiv:2505.24098.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. 2025. Live-CodeBench: Holistic and contamination free evaluation of large language models for code. In International Conference on Learning Representations (ICLR).

Yuhang Lai, Chengxi Li, Yiming Wang, Tianyi Zhang, Ruiqi Zhong, Luke Zettlemoyer, Wen-Tau Yih, Daniel Fried, Sida Wang, and Tao Yu. 2023. DS-1000: A natural and reliable benchmark for data science code generation. In International Conference on Machine Learning (ICML), pages 18319–18345.

Tim Launer, Jonas Hübotter, Marco Bagatella, Ido Hakimi, and Andreas Krause. 2026. Majority voting for code generation. arXiv preprint arXiv:2604.15618.

Dacheng Li, Shiyi Cao, Chengkun Cao, Xiuyu Li, Shangyin Tan, Kurt Keutzer, Jiarong Xing, Joseph E. Gonzalez, and Ion Stoica. 2025. S\*: Test time scaling for code generation. In Findings of the Associationfor Computational Linguistics: EMNLP 2025, pages 15964–15978, Suzhou, China.

Rongao Li, Jie Fu, Bo-Wen Zhang, Tao Huang, Zhihong Sun, Chen Lyu, Guang Liu, Zhi Jin, and Ge Li. 2023. TACO: Topics in algorithmic COde generation dataset. arXiv preprint arXiv:2312.14852.

Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien de Masson d’Autume, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov, and 7 others. 2022. Competitionlevel code generation with AlphaCode. Science, 378(6624):1092–1097.

Ruotong Liao, Nikolai Röhrich, Xiaohan Wang, Yuhui Zhang, Yasaman Samadzadeh, Volker Tresp, and Serena Yeung-Levy. 2026. Tool verification for test-time reinforcement learning. arXiv preprint arXiv:2603.02203.

Chris Yuhao Liu, Liang Zeng, Jiacai Liu, Rui Yan, Jujie He, Chaojie Wang, Shuicheng Yan, Yang Liu, and Yahui Zhou. 2024. Skywork-Reward: Bag of

tricks for reward modeling in LLMs. arXiv preprint arXiv:2410.18451.

Mehrzad Samadi, Aleksander Ficek, Sean Narenthiran, Siddhartha Jain, Wasi Uddin Ahmad, Somshubra Majumdar, Vahid Noroozi, and Boris Ginsburg. 2026. Scaling test-time compute to achieve IOI gold medal with open-weight models. In Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 33156–33169, San Diego, CA. Association for Computational Linguistics.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2025. HybridFlow: A flexible and efficient RLHF framework. In European Conference on Computer Systems (EuroSys), pages 1279–1297, Rotterdam, Netherlands.

Freda Shi, Daniel Fried, Marjan Ghazvininejad, Luke Zettlemoyer, and Sida I. Wang. 2022. Natural language to code translation with execution. In Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 3533–3546, Abu Dhabi, United Arab Emirates.

Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2025. Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning. In International Conference on Learning Representations (ICLR).

Florian Tambon, Arghavan Moradi-Dakhel, Amin Nikanjam, Foutse Khomh, Michel C. Desmarais, and Giuliano Antoniol. 2025. Bugs in large language models generated code: An empirical study. Empirical Software Engineering, 30(3):65.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models. In International Conference on Learning Representations (ICLR).

Yinjie Wang, Ling Yang, Ye Tian, Ke Shen, and Mengdi Wang. 2025. Co-evolving LLM coder and unit tester via reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 38.

Jiacheng Xu, Wentao Zhang, Zhiyi Lyu, Fuxiang Zhang, Chaojie Wang, Yang Liu, and Bo An. 2026. Twostage reinforcement learning for sound and adversarial test generation in code LLMs. arXiv preprint arXiv:2609.03955.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,

Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Juncai Liu, LingJun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, and 17 others. 2025. DAPO: An open-source LLM reinforcement learning system at scale. In Advances in Neural Information Processing Systems (NeurIPS), volume 38.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. 2025. Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? In Advances in Neural Information Processing Systems (NeurIPS), volume 38, pages 64304–64339.

Yujun Zhou, Zhenwen Liang, Haolin Liu, Wenhao Yu, Kishan Panaganti, Linfeng Song, Dian Yu, Xiangliang Zhang, Haitao Mi, and Dong Yu. 2025. Evolving language models without labels: Majority drives selection, novelty promotes variation. arXiv preprint arXiv:2509.15194.

Xinyu Zhu, Mengzhou Xia, Zhepei Wei, Wei-Lin Chen, Danqi Chen, and Yu Meng. 2025. The surprising effectiveness of negative reinforcement in LLM reasoning. In Advances in Neural Information Processing Systems (NeurIPS), volume 38.

Terry Yue Zhuo, Minh Chien Vu, Jenny Chim, Han Hu, Wenhao Yu, Ratnadira Widyasari, Imam Nur Bani Yusuf, Haolan Zhan, Junda He, Indraneil Paul, Simon Brunner, Chen Gong, James Hoang, Armel Zebaze, Xiaoheng Hong, Wen-Ding Li, Jean Kaddour, Ming Xu, Zhihan Zhang, and 14 others. 2025. Big-CodeBench: Benchmarking code generation with diverse function calls and complex instructions. In International Conference on Learning Representations (ICLR).

Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Yuchen Zhang, Xinwei Long, Ermo Hua, Biqing Qi, Youbang Sun, Zhiyuan Ma, Lifan Yuan, Ning Ding, and Bowen Zhou. 2025. TTRL: Test-time reinforcement learning. In Advances in Neural Information Processing Systems (NeurIPS), volume 38.

## A Experimental Configuration

All our RL experiments adopt the verl (Sheng et al., 2025) framework with vLLM rollouts and an FSDP actor. The two backbones are Qwen3- 4B and Qwen3-8B in non-thinking mode. Unless noted otherwise, every method shares the optimization and sampling settings below. Only the reward source, loss term, and method-specific hyperparameters change between methods.

Optimization. We use AdamW with a constant learning rate of $5 \times 1 0 ^ { - 7 }$ after a 20-step linear warmup. The training batch is 16 prompts; each prompt produces $G = 1 6$ on-policy rollouts, giving a $1 6 { \times } 1 6 = 2 5 6 { \cdot } \mathrm { r o }$ llout step. PPO mini-batches contain 4 prompts and we run a single inner PPO epoch per global step. The maximum number of global steps is 150. Unless otherwise specified, we evaluate the final checkpoint from the prespecified training schedule, or the last pre-collapse checkpoint for methods that undergo catastrophic policy collapse before the end of training.

Sampling. A single sampling profile is shared between RL rollouts and validation rollouts. For Qwen3 models in non-thinking mode, generation uses $\mathsf { t o p \mathrm { { - } p \ l = \ 0 . 8 , \ t o p \mathrm { { - } k \ l = \ 2 0 } } }$ , and mi $\mathsf { n \_ p } = \mathsf { 0 } .$ For sequence lengths, we use max\_response\_length = 8192 in experiments. P@1 and P@16 in the main and transfer tables (Tables 1 and 5) are computed from 16 validation samples per problem.

Method-specific hyperparameters. For PCRbased methods, the probe sets are precomputed once for each problem and reused across training steps. The same probe set is shared by $\mathrm { G R P O } _ { \mathrm { P C R } } .$ $\mathrm { N S R } _ { \mathrm { P C R } }$ , and ERPO within each backbone and benchmark setting. For all experiments, we generate 10 additional probe inputs per problem. PCR is computed over the deduplicated union of these generated probes and the example inputs in the problem statement.

$\mathbf { G R P O } _ { \mathrm { P u b } }$ : We use standard GRPO with public-binary rewards and a KL coefficient of $1 0 ^ { - 3 }$ . The reward is 1 only when the candidate program passes all provided public tests, and 0 otherwise.

$\mathbf { G R P O } _ { \mathrm { P C R } } ;$ We use standard GRPO with pointwise PCR rewards and KL coefficient $1 0 ^ { - 3 }$ . The PCR score is used directly as the scalar GRPO reward.

$\mathbf { N S R } _ { \mathrm { P C R } } ;$ : We use pointwise PCR rewards with KL coefficient $1 0 ^ { - 3 } .$ . For a raw PCR score $s _ { i } ,$ we define the NSR reward $r _ { i } =$ $2 ( s _ { i } - 0 . 5 ) \in [ - 1 , 1 ]$ and keep only samples with $r _ { i } < 0$ (equivalently, $s _ { i } < 0 . 5 )$ for the actor update.

• ERPO: We use pointwise PCR rewards with KL coefficient $1 0 ^ { - 3 }$ . ERPO computes the group-normalized GRPO advantage on the PCR scores and masks the top $\lceil G / 2 \rceil$ candidates in each group. The remaining lower-PCR candidates usually have negative normalized advantages, so the actor update penalizes them rather than reinforcing them, while high-PCR candidates receive zero positive policygradient weight. The fixed entropy-ceiling term uses $\lambda _ { H } = 0 . 5$ and $H _ { \mathrm { c e i l } } = 0 . 5$ with actor entropy computation enabled.

Inference-time scaling. For the inferencescaling experiment, we resample N = 256 rollouts per problem from each method’s checkpoint with the same sampling profile and report pass@k as a function of $k \in \{ 1 , 2 , 4 , 8 , 1 6 , 3 2 , 6 4 , 1 2 8 , 2 5 6 \}$

Compute resources. All RL training runs were executed on eight NVIDIA A100 GPUs with 80GB memory per GPU. A complete RL training run for one algorithm takes approximately 12 hours; the inference-time scaling experiment takes about 3 hours to sample 256 times for every 100 problems.

Datasets. We use LiveCodeBench v6 for all experiments involving LiveCodeBench. For all other benchmarks, we use the complete official releases.

Execution environment. Candidate programs were executed in isolated subprocesses with perexecution timeout and resource limits. Network access was disabled, and compilation errors, runtime errors, and timeouts were treated as execution failures. Failed executions received no credit for the corresponding probe and were excluded from that probe’s majority computation.

Artifact licenses and terms. We use publicly available research artifacts, including the coding benchmarks, model checkpoints, and software frameworks described in Section 4 and Appendix A. We cite the original creators of these artifacts and use them under their respective licenses and terms of use. Our experiments use these artifacts for research evaluation and adaptation only, and we do not redistribute the original datasets or model checkpoints beyond what is permitted by their original licenses.

## B Probe Input Generation Prompt

Section 3.1 constructs the output-free probe set for each problem using an iterative chat prompt. At iteration $j ,$ , the prompt is instantiated with the problem statement and the j − 1 probe inputs already generated for the same problem. The same prompt structure is used for all probe sets reported in our experiments. We include the template here to make the probe-generation procedure reproducible, since probe quality affects the behavioral comparison used by PCR.

## Probe Input Generation Prompt Template

## System message.

You are an expert at designing test inputs for competitive programming problems. You generate test inputs that are strictly format-correct, satisfy all stated constraints, and are structurally distinct from any inputs already produced.

User message.

Generate input #{iter\_idx} of {k\_total} for the given competitive-programming problem.

Inputs provided to the prompt.

• {problem\_content}: the full problem statement.

• {prior\_inputs\_json}: all previously generated probe inputs for this problem.

The final answer must be a raw input string, not a JSON object.

Generation requirements.

• Follow the input format specified by the problem statement, including quoting, newlines, list nesting, and any outer test-case count line.

• If the format uses an outer test-case count line, such as $t , T ,$ , or q on the first line, begin with a valid count and include exactly that number of case blocks.

• Make the new input structurally different from both the problem-statement examples and all prior generated inputs.

## Required output structure.

1. Reasoning: identify an uncovered aspect of the problem, such as size, range, sparsity, pattern, character set, distribution, or adversarial structure.

2. Format and constraint check: state whether an outer test-case count line is used, state the generated count when applicable, and verify all size, value, structural, and format constraints.

3. Distinctness verification: for each prior input, give a concrete comparison of the form “Prior X had $\mathbf { A } ,$ mine has B, and A differs from B on dimension $\mathrm { Y } . \mathrm { ^ { \circ } }$ Use concrete values rather than abstract claims.

4. Final Input: output the single raw input string in canonical format. If an outer test-case count line is used, the raw input must begin with that count line.

Size constraint. The raw input string must not exceed 5000 characters. For problems with very large limits, use representative sizes between 100 and 1000 rather than enumerating inputs at the maximum scale.

## C Transfer Results after CodeContests Adaptation (Qwen3-4B)

To complement the main transfer results (Table 1), which use LiveCodeBench as the adaptation source, we additionally adapt Qwen3-4B on the unlabeled CodeContests problem statements and transfer the resulting checkpoint to the other three code benchmarks, as reported in Table 5. The CodeContests columns report in-domain transductive adaptation, while the remaining columns report zero-shot transfer without target-benchmark adaptation. We include $\mathrm { N S R } _ { \mathrm { P C R } }$ because its checkpoint is also used in Table 3. As in the main experiments, we evaluate the final checkpoint from the prespecified adaptation schedule without hidden-test-based checkpoint selection.

Table 5 shows that the main conclusion is not specific to LiveCodeBench adaptation. ERPO performs best on both in-domain metrics and achieves the best mean transfer pass@1 and pass@16. $\mathrm { G R P O _ { P u b } }$ improves in-domain pass@1 using public test cases, but this gain does not transfer, while direct PCR optimization provides no consistent benefit. NSR<sub>PCR</sub> mainly improves pass@16 without comparable gains in pass@1. This pattern agrees with Table 1: ERPO is the only method that consistently strengthens both single-pass accuracy and the pass@k frontier.

## D Training Trajectory Analysis

Figure 4 shows the training dynamics behind the main results. $\mathrm { G R P O _ { P u b } }$ improves pass@1, but its pass@16 remains close to the Base reference and its entropy declines rapidly, suggesting a narrower solution distribution without a stronger pass@k frontier. Direct PCR optimization does not improve pass@1 and steadily reduces pass@16, consistent with Table 2, where high PCR is not a reliable correctness label. NSR<sub>PCR</sub> shows the opposite failure mode: it initially improves pass@16, but pass@1 declines as entropy grows and validation pass@k eventually collapses. ERPO avoids both patterns, sustaining improvements in pass@1 and pass@16 while keeping entropy stable. Its rank mask avoids positive updates on high-PCR candidates, while the retained low-PCR samples typically provide negative updates. The entropy ceiling further limits upward entropy drift. These dynamics are consistent with the joint gains in Table 1.

Table 5: Transfer results for Qwen3-4B adapted on CodeContests (CC) and evaluated on LiveCodeBench (LCB), CodeForces (CF), and TACO. The CC columns report in-domain transductive adaptation; the remaining columns report zero-shot transfer from the final CC-adapted checkpoint. All values are percentages.
<table><tr><td rowspan="3">Model</td><td rowspan="3">Method</td><td colspan="2">In-Domain Adapt.</td><td colspan="8">Zero-Shot Transfer</td></tr><tr><td colspan="2">CC</td><td colspan="2">LCB</td><td colspan="2">CF</td><td colspan="2">TACO</td><td colspan="2">Mean</td></tr><tr><td>P@1</td><td>P@16</td><td>P@1</td><td>P@16</td><td>P@1</td><td>P@16</td><td>P@1</td><td>P@16</td><td>P@1</td><td>P@16</td></tr><tr><td>Qwen3-4B</td><td>Base</td><td>25.13</td><td>42.26</td><td>26.25</td><td>36.00</td><td>9.81</td><td>21.84</td><td>6.09</td><td>13.28</td><td>14.05</td><td>23.71</td></tr><tr><td></td><td> $\mathrm { G R P O } _ { \mathrm { P u b } }$ </td><td>40.13</td><td>44.12</td><td>25.61</td><td>32.57</td><td>10.71</td><td>22.48</td><td>6.09</td><td>12.94</td><td>14.14</td><td>22.66</td></tr><tr><td></td><td> $\mathrm { G R P O } _ { \mathrm { P C R } }$ </td><td>27.05</td><td>34.45</td><td>25.54</td><td>36.00</td><td>9.89</td><td>23.77</td><td>5.99</td><td>12.71</td><td>13.81</td><td>24.16</td></tr><tr><td></td><td>NSRPCR</td><td>27.42</td><td>50.42</td><td>24.61</td><td>39.43</td><td>10.95</td><td>32.33</td><td>6.73</td><td>17.03</td><td>14.10</td><td>29.60</td></tr><tr><td></td><td>ERPO</td><td>41.62</td><td>57.56</td><td>33.75</td><td>45.14</td><td>15.81</td><td>30.62</td><td>11.78</td><td>20.32</td><td>20.45</td><td>32.03</td></tr></table>

Base GRPOPub GRPOPCR NSRPCR ERPO (ours)

![](images/2ad475ce855d62e15c5e505c3e1aac9a1ad7f8e90243ba5864c562d861641335.jpg)

![](images/8534b01c6db20fbc3933eb099a45859185fc2d9b2c9ac3259d81b875e3fd7775.jpg)

![](images/b71eedbccb0419787114c72ecab3c9889b8808b853b2ddc40386cdf28b906729.jpg)  
Figure 4: Per-step training dynamics on Qwen3-8B / LiveCodeBench. Left: pass@1. Middle: pass@16. Right: actor policy entropy. Dashed gray lines mark the Base reference. The dotted indigo line marks ERPO’s fixed entropy ceiling $H _ { \mathrm { c e i l } } = 0 . 5 . \mathrm { N S R } _ { \mathrm { P C R } }$ is shown through step 50 because its entropy subsequently diverges and validation pass@k collapses.

Table 6: Inference-time pass@k on Qwen3-8B adapted on LiveCodeBench, evaluated on the LiveCodeBench test split. All values are percentages.
<table><tr><td>Method</td><td>P@1</td><td>P@2</td><td>P@4</td><td>P@8</td><td>P@16</td><td>P@32</td><td>P@64</td><td>P@128</td><td>P@256</td></tr><tr><td>Base</td><td>25.46</td><td>28.07</td><td>30.55</td><td>32.56</td><td>34.04</td><td>35.45</td><td>36.99</td><td>38.56</td><td>40.00</td></tr><tr><td> $\mathrm { G R P O } _ { \mathrm { P u b } }$ </td><td>30.47</td><td>31.83</td><td>32.96</td><td>34.09</td><td>35.35</td><td>36.66</td><td>37.92</td><td>39.25</td><td>40.57</td></tr><tr><td>GRPOPCR</td><td>26.13</td><td>27.54</td><td>28.80</td><td>29.94</td><td>31.00</td><td>32.07</td><td>33.20</td><td>34.61</td><td>36.57</td></tr><tr><td>NSRPCR</td><td>25.40</td><td>29.14</td><td>32.28</td><td>35.11</td><td>37.83</td><td>40.44</td><td>42.96</td><td>45.49</td><td>48.00</td></tr><tr><td>ERPO</td><td>34.73</td><td>39.03</td><td>42.69</td><td>45.89</td><td>48.56</td><td>50.84</td><td>52.99</td><td>55.09</td><td>57.14</td></tr></table>

## E Inference-Time Scaling after LiveCodeBench Adaptation (Qwen3-8B)

To assess whether the inference-time scaling behavior in Table 3 extends to a larger model capacity, we report pass@k for Qwen3-8B adapted on Live-CodeBench, using up to N = 256 samples per problem.

Table 6 shows that ERPO leads at every reported sampling budget and that its advantage over the Base model grows with k. Its pass@1 also exceeds the Base model’s pass@16 with a single output. $\mathrm { G R P O _ { P u b } }$ improves the small-k region but saturates near the Base frontier at large $k ,$ while $\mathrm { G R P O } _ { \mathrm { P C R } }$ falls below the Base model beyond pass@1. $\mathrm { N S R } _ { \mathrm { P C R } }$ leaves pass@1 nearly unchanged while improving large-k performance, suggesting that its gains mainly reflect sampling diversity. ERPO improves both single-pass accuracy and the high-k frontier, showing that its rankmasked update and entropy ceiling remain effective at the larger model capacity.

Table 7: Results across three independent runs in the Qwen3-4B setting adapted on LiveCodeBench. Each entry reports the mean ± standard deviation. Values are percentages.
<table><tr><td></td><td colspan="2">LCB</td><td colspan="2">CC</td><td colspan="2">CF</td><td colspan="2">TACO</td></tr><tr><td>Method</td><td>P@1</td><td>P@16</td><td>P@1</td><td>P@16</td><td>P@1</td><td>P@16</td><td>P@1</td><td>P@16</td></tr><tr><td>Base</td><td> $2 6 . 0 4 \pm 0 . 4 2 3 4 . 4 3 \pm 0 . 8 7 2 4 . 9 4 \pm 0 . 1 7 4 1 . 1 4 \pm 0 . 9 7 9 . 6 8 \pm 0 . 1 2 2 . 4 8 \pm 1 . 1 1$ </td><td></td><td></td><td></td><td></td><td></td><td> $6 . 2 7 \pm 0 . 1 1$ </td><td> $1 3 . 2 4 \pm 0 . 2 4$ </td></tr><tr><td> $\mathrm { G R P O } _ { \mathrm { P u b } }$ </td><td> $3 3 . 4 5 \pm 0 . 2 7 \ 3 5 . 6 1 \pm 0 . 4 5 \ 2 6 . 3 9 \pm 0 . 1 5 \ 4 0 . 0 3 \pm 0 . 8 7 \ 1 0 . 7 2 \pm 0 . 0 5 \ 2 3 . 4 8 \pm 0 . 8 1$ </td><td></td><td></td><td></td><td></td><td></td><td> $6 . 0 1 \pm 0 . 1 3$ </td><td> $1 2 . 9 8 \pm 0 . 6 3$ </td></tr><tr><td></td><td> $\mathrm { G R P O _ { P C R } ~ 2 7 . 2 8 \pm 0 . 2 2 ~ 2 9 . 5 4 \pm 0 . 6 2 ~ 2 6 . 0 4 \pm 0 . 1 1 ~ 3 6 . 5 4 \pm 0 . 2 4 ~ 8 . 4 1 \pm 0 . 1 0 ~ 2 0 . 2 0 \pm 0 . 1 2 ~ 2 0 . 2 6 ~ . }$ </td><td></td><td></td><td></td><td></td><td></td><td> $5 . 5 8 \pm 0 . 1 2$ </td><td> $1 0 . 0 3 \pm 0 . 3 6$ </td></tr><tr><td> $\mathrm { N S R } _ { \mathrm { P C R } }$ </td><td> $2 4 . 9 0 \pm 0 . 3 2 \ 3 8 . 3 6 \pm 1 . 2 7 \ 2 7 . 9 7 \pm 0 . 5 7 \ 5 1 . 6 7 \pm 0 . 8 9 \ 1 0 . 6 0 \pm 0 . 1 7 \ 2 9 . 7 6 \pm 0 . 6 1$ </td><td></td><td></td><td></td><td></td><td></td><td> $6 . 8 6 \pm 0 . 1 3$ </td><td> $1 6 . 6 9 \pm 0 . 8 0$ </td></tr><tr><td>ERPO</td><td> $3 6 . 6 4 \pm 0 . 3 3$ </td><td>46.96 ±1.15 42.13 ±0.47 58.44 ± 0.24 20.64 ± 0.29</td><td></td><td></td><td></td><td> $3 8 . 2 6 \pm 0 . 3 3$ </td><td> $1 3 . 4 2 \pm 0 . 0 9$ </td><td> $2 1 . 3 0 \pm 0 . 4 0$ </td></tr></table>

## F Results across Independent Runs

To better demonstrate the robustness of ERPO’s performance gains, we repeated the complete Qwen3-4B setting reported in Table 1 using three independent end-to-end runs for each adaptation method on LCB. Each resulting checkpoint was evaluated in-domain on LCB and zero-shot on CC, CF, and TACO. We report the mean and standard deviation across these runs in Table 7. For the Base model, which does not involve training, the mean and standard deviation are computed across three independent evaluation runs. Across the three runs, ERPO achieves the highest mean on all eight evaluation metrics. Its margins over the strongest baselines are consistently larger than the corresponding standard deviations, indicating that the improvements are robust to the observed run-to-run variation.

## G Results on More Benchmarks

We conducted additional in-domain test-time adaptation experiments on BigCodeBench (BCB; Zhuo et al., 2025) and DS-1000 (Lai et al., 2023). Each method was trained on the unlabeled problems of the corresponding benchmark using the same Qwen3-4B backbone, adaptation budget, and optimization settings as in the main experiments. Table 8 reports the results.

ERPO consistently improves both pass@1 and pass@16 on both benchmarks.

## H Sensitivity to Rank-Mask Percentile

We selected 50% as a simple median split because it avoids an absolute PCR threshold and task-specific tuning without labeled validation data. To assess sensitivity to this choice, we vary the masked percentile on LiveCodeBench while keeping all other experimental settings fixed. Table 9 reports P@1 and P@16 for both backbones.

Table 8: In-domain adaptation results on two additional benchmarks using Qwen3-4B. Parentheses show absolute changes from Base. Values are percentages.
<table><tr><td>Benchmark Method</td><td></td><td>P@1</td><td>P@16</td></tr><tr><td rowspan="3">BCB</td><td>Base</td><td>35.5</td><td>45.4</td></tr><tr><td>GRPOPCR</td><td>36.7 (+1.2)</td><td>44.3 (−1.1)</td></tr><tr><td>NSRPCR</td><td>30.4 (−5.1)</td><td>49.4 (+4.0)</td></tr><tr><td></td><td>ERPO</td><td>39.2 (+3.7)</td><td>57.8 (+12.4)</td></tr><tr><td rowspan="3">DS-1000</td><td>Base</td><td>22.7</td><td>29.2</td></tr><tr><td>GRPOPCR</td><td>23.8 (+1.1)</td><td>28.9 (−0.3)</td></tr><tr><td>NSRPCR</td><td>17.2 (−5.5)</td><td>36.3 (+7.1)</td></tr><tr><td></td><td>ERPO</td><td>27.4 (+4.7)</td><td>44.9 (+15.7)</td></tr></table>

Table 9: Sensitivity to the rank-mask percentile on Live-CodeBench. Values are percentages.
<table><tr><td>Backbone</td><td colspan="4">Metric 1 Base 10% 25% 50% 75%</td></tr><tr><td>Qwen3-4B</td><td>P@1</td><td>26.0 29.8</td><td>36.9 36.7 36.4 46.3</td></tr><tr><td>Qwen3-8B P@1</td><td>P@16 P@16</td><td>34.4 36.6 25.6 30.0 34.6</td><td>46.3 46.3 35.8 35.1 34.9 6 37.7 47.4 50.9 49.7</td></tr></table>

Performance remains relatively stable across masking ratios from 25% to 75%, and all three settings substantially improve over the corresponding Base models. Masking only 10% still improves over the Base models but performs worse than masking 25%–75%, likely because more candidates with unreliable positive advantages remain unmasked.

## I Sensitivity to Generated Probe Count

Let $N _ { \mathrm { g e n } }$ denote the number of additional probe inputs generated per problem. We use $N _ { \mathrm { g e n } } =$ 10 by default to balance PCR reliability and probe-execution cost. To assess sensitivity to this choice, we first evaluate how the number of generated probes affects PCR AUC for predicting whether a candidate passes the hidden tests on Live-CodeBench with Qwen3-4B.

Table 10: PCR AUC for different numbers of generated probes on LiveCodeBench with Qwen3-4B.
<table><tr><td> $N _ { \mathrm { g e n } }$ </td><td>1</td><td>3</td><td>5</td><td>10</td><td>20</td></tr><tr><td>PCR AUC</td><td>0.706</td><td>0.754</td><td>0.774</td><td>0.790</td><td>0.799</td></tr></table>

PCR AUC improves as $N _ { \mathrm { g e n } }$ increases, indicating that additional probes produce a more reliable consensus signal. The improvement diminishes beyond $N _ { \mathrm { g e n } } = 1 0$ , with AUC increasing by only 0.009 when the generated probe count is doubled from 10 to 20. We further evaluate the effect of $N _ { \mathrm { g e n } }$ on downstream performance in Table 11.

Table 11: Sensitivity to the generated probe count on LiveCodeBench. Values are percentages.
<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Metric Base</td><td rowspan="2"></td><td colspan="2"> $N _ { \mathrm { g e n } }$ </td></tr><tr><td>5</td><td>10 20</td></tr><tr><td>Qwen3-4B</td><td>P@1</td><td>26.0</td><td>35.8 36.7</td><td>36.5</td></tr><tr><td rowspan="3">Qwen3-8B</td><td>P@16</td><td>34.4</td><td>45.7 46.3</td><td>46.9</td></tr><tr><td>P@1</td><td>25.6</td><td>34.5 35.1</td><td>35.5</td></tr><tr><td>P@16</td><td>34.6</td><td>48.0 50.9</td><td>50.3</td></tr></table>

Performance remains stable across $N _ { \mathrm { g e n } } \in$ {5, 10, 20}, and increasing $N _ { \mathrm { g e n } }$ from 10 to 20 does not yield consistent improvements despite the additional execution cost. These results support $N _ { \mathrm { g e n } } = 1 0$ as a practical balance between reward reliability, downstream performance, and computational cost.

## J Probe Examples and Hidden-Test Correctness

We provide two representative cases illustrating PCR’s asymmetric reliability.

Case A: Probe disagreement reveals incorrect candidates. Consider a task that counts arrays whose elements satisfy given bounds while preserving the adjacent differences of a reference array. One generated probe is:

original = [1, 3, 2, 4, 3]   
bounds = [[1, 3], [2, 4], [1, 3],   
[3, 5], [2, 4]]

The outputs of the 16 candidates form four clusters:

• 2: 7 candidates, all passing the hidden tests;

• 3: 4 candidates, all failing;

• 0: 3 candidates, all failing;

## • 1: 2 candidates, all failing.

The feasible first-element values are 1 and 2, so the correct answer is 2. Across the complete set of ten generated probes, all seven candidates that pass the hidden tests receive $\mathrm { P C R } = 1 . 0$ , whereas all nine failing candidates receive lower PCR. Thus, PCR perfectly separates passing and failing candidates in this case.

Case B: Unanimous probe agreement misses a boundary error. Consider a task that computes $X = 1 + N + N ^ { 2 } + \cdot \cdot \cdot + N ^ { M }$ and outputs inf when $X \mathrm { ~ > ~ } 1 0 ^ { 9 }$ . All 16 candidates agree on all ten generated probes, including $2 3 0 $ inf and $2 \ 2 8 \to 2 8 9 7 1 5 1$ , and therefore receive PCR = 1.0. However, two candidates fail the hidden input 2 29, where $X \ = \ 2 ^ { 3 0 } - 1 \ = \ 1 0 7 3 7 4 1 8 2 3 \ >$ $1 0 ^ { 9 }$ . They output 1073741823 instead of inf. The probes miss this narrow boundary region, so the incorrect candidates remain indistinguishable from the correct ones.

Together, these cases illustrate why ERPO uses PCR asymmetrically: low consensus can provide useful negative evidence, whereas high consensus is not treated as positive supervision.