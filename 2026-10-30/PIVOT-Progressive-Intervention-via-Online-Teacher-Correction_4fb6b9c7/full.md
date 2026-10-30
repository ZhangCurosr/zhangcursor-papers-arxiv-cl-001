# PIVOT: Progressive Intervention via Online Teacher Correction

Zhihao Yang<sup>1,4,\*</sup> Boyuan Zheng<sup>2,\*</sup> Puhui Tan<sup>2,\*</sup> Yukun Chen<sup>1</sup> Kuan Zhang<sup>2</sup>

Longze Chen<sup>1</sup> Jiaming Li<sup>1</sup> Renke Shan<sup>1</sup> Lu Wang<sup>2,†</sup> Xianpei Han<sup>3,†</sup>

<sup>1</sup>University of Chinese Academy of Sciences <sup>2</sup>Tsinghua University

<sup>3</sup>Chinese Information Processing Laboratory, Institute of Software, Chinese Academy of Sciences

<sup>4</sup>School of Advanced Interdisciplinary Sciences, University of Chinese Academy of Sciences

<sup>\*</sup>Equal contribution <sup>†</sup>Corresponding authors

yangzhihao25@mails.ucas.ac.cn zhengby25@mails.tsinghua.edu.cn

Abstract. Reinforcement learning from verifiable rewards (RLVR) is limited by sparse outcome-level feedback on hard reasoning tasks: when all rollouts in a group fail, rewards provide no learning signal. Stronger teachers can provide additional guidance, but this raises a central question: how can such signals be incorporated into RLVR to support successful rollouts while remaining learnable for the student? We distinguish two complementary mechanisms: trajectory-level intervention, which improves access to reward-bearing trajectories, and token-level supervision, which provides dense targets to guide student learning. Our controlled experiments show that either mechanism alone can remain insuficient, motivating their combination. We therefore introduce PIVOT, which inserts teacher-generated corrections into student rollouts and applies token-level KL supervision to these segments alongside the RLVR objective. Beyond their combination, we systematically analyze where to intervene and how to supervise, yielding a recipe for efective guidance. Across hard logic games and mathematical reasoning benchmarks, PIVOT consistently outperforms existing methods, nearly doubling mean avg@8 on logic games from 19.5% for the best-performing baseline to 35.6%.

## 1 Introduction

Reinforcement learning from verifiable rewards (RLVR) has become a dominant paradigm for eliciting reasoning in large language models (Shao et al., 2024; Lambert et al., 2024; Guo et al., 2025). Despite its success, recent studies suggest that RLVR primarily amplifies reasoning behaviors present in the base model, raising questions about its ability to acquire new capabilities (Yue et al., 2025; Zhao et al., 2025). This limitation is particularly pronounced on hard problems for which the initial policy rarely generates a correct solution (Yan et al., 2026; Chen et al., 2025; Qu et al., 2026). In this regime, sparse outcome-level rewards provide little information for policy improvement: groups of unsuccessful rollouts receive identical zero rewards, leaving the policy without a reward-based learning signal (Wu et al., 2026; Tao et al., 2025). These challenges have motivated the use of stronger teachers to provide guidance beyond verifier feedback. However, introducing teacher signals does not by itself ensure that the student can learn from them. This raises a central question: how can teacher guidance support successful rollouts while remaining learnable for the student?

Existing approaches incorporate teacher signals into RLVR through two broad mechanisms. Trajectorylevel intervention introduces teacher-generated trajectories or segments into training, improving access to reward-bearing trajectories (Guo et al., 2025; Wen et al., 2025; Yan et al., 2025; Qu et al., 2026). However, teacher-generated trajectories may difer substantially from the student’s own rollout distribution, making the teacher’s behavior dificult to learn even when these trajectories are successful. Token-level supervision, in contrast, provides dense supervision along student-generated rollouts using the teacher’s predictive distributions (Xu et al., 2025). Yet when these rollouts deviate from a valid reasoning path, local corrective signals may not support a coherent recovery. These limitations suggest that efective teacher guidance must shape both the trajectories used for training and the supervision derived from them.

![](images/d31aa94e46049792d2305ff43dda3aca3d64b5c551db9e2e84b8bfb0a8085104.jpg)  
Figure 1. Overview of PIVOT. For each query, the policy model samples a group of rollouts, including teacher-assisted trajectories. In a teacher-assisted rollout, the student first generates a prefix, the teacher provides a correction segment, and the student then continues generation. The completed trajectory is optimized by the verifier-based GRPO objective, while the teacher-generated segment provides dense token-level supervision through a teacher-student KL loss.

We therefore introduce Progressive Intervention via Online Teacher Correction (PIVOT), a teacher-assisted RLVR framework that integrates teacher corrections into both rollout generation and policy optimization. PIVOT augments the rollout group with teacher-assisted trajectories, in which the teacher provides a correction based on a student-generated prefix and the student continues from the corrected context. As illustrated in Fig. 1, verifier-based RLVR optimizes the resulting trajectories, while a teacher–student KL objective provides dense token-level supervision on the correction segments. Our controlled experiments show that this joint use of teacher signals yields stronger learning than either intervention or token-level supervision alone. We further examine how intervention placement and the KL objective afect the efectiveness of teacher guidance, informing a concrete recipe for where to intervene, how to supervise, and how much teacher guidance to apply.

We evaluate PIVOT on hard logic games from SynLogic (Liu et al., 2026) and mathematical reasoning benchmarks. On hard logic games, where RL-only training makes limited progress, PIVOT nearly doubles mean avg@8 from 19.5% for the best-performing baseline to 35.6%. On mathematical reasoning tasks, PIVOT consistently outperforms the evaluated baselines across both in-distribution and out-of-distribution benchmarks.

## Our contributions are threefold:

• We identify two complementary mechanisms for teacher guidance in RLVR—trajectory-level intervention and token-level supervision—and show that either mechanism alone can remain insuficient for efective learning on hard reasoning tasks.

• We introduce PIVOT, a teacher-assisted RLVR framework that integrates teacher corrections into both rollout generation and policy optimization, and provide a systematic analysis of how teacher guidance should be incorporated into training.

• We conduct extensive experiments and show that PIVOT nearly doubles mean avg@8 over the bestperforming baseline on hard logic games while consistently outperforming baselines across in-distribution and out-of-distribution mathematical reasoning benchmarks.

## 2 Preliminaries

Problem setup. We consider reinforcement learning with verifiable rewards (RLVR) for a student policy $\pi _ { \theta } ,$ with access to a teacher policy $\pi _ { T }$ during training. Given a prompt q drawn from the prompt distribution D, a response $\tau = ( \tau _ { 1 } , \dots , \tau _ { | \tau | } )$ receives a binary verifier reward $R ( \mathbf { q } , \tau ) \in \{ 0 , 1 \}$ , with 1 indicating a correct response. The student factorizes its response distribution as $\begin{array} { r } { \pi _ { \theta } ( \tau \mid \mathbf { q } ) = \prod _ { t = 1 } ^ { | \tau | } \pi _ { \theta } ( \tau _ { t } \mid h _ { t } ) } \end{array}$ , where $h _ { t } = ( \mathbf { q } , \tau _ { < t } )$ denotes the token context. The objective is to maximize $\mathbb { E } _ { \mathbf { q } \sim \mathcal { D } , \tau \sim \pi _ { \theta } ( \cdot | \mathbf { q } ) } \left[ R ( \mathbf { q } , \tau ) \right]$

RLVR objective. Under Group Relative Policy Optimization (GRPO) (Shao et al., 2024), groups of rollouts are sampled from $\pi _ { \theta _ { \mathrm { o l d } } }$ for each prompt, and their verifier rewards are normalized within each group to obtain advantages $A _ { i } .$ . For a batch of � rollouts, the policy loss is averaged over all tokens:

$$
\mathcal { L } _ { \mathrm { G R P O } } ( \theta ) = - \frac { 1 } { \sum _ { i = 1 } ^ { N } | \tau _ { i } | } \sum _ { i = 1 } ^ { N } \sum _ { t = 1 } ^ { | \tau _ { i } | } \operatorname* { m i n } \Bigl ( r _ { i , t } ( \theta ) A _ { i } , \mathrm { c l i p } ( r _ { i , t } ( \theta ) , 1 - \varepsilon , 1 + \varepsilon ) A _ { i } \Bigr ) ,
$$

Here, $| \tau _ { i } |$ is the rollout length, $\varepsilon > 0$ is the clipping threshold, and $\begin{array} { r } { r _ { i , t } ( \theta ) = \frac { \pi _ { \theta } ( \tau _ { i , t } | h _ { i , t } ) } { \pi _ { \theta _ { 0 \mathrm { l d } } } ( \tau _ { i , t } | h _ { i , t } ) } } \end{array}$ is the importance ratio between the current and rollout policies. The RLVR objective additionally includes a KL penalty to a fixed reference policy $\pi _ { \mathrm { r e f } } .$

$$
\mathcal { L } _ { \mathrm { r e f } } ( \theta ) = \frac { 1 } { \sum _ { i = 1 } ^ { N } \left| \tau _ { i } \right| } \sum _ { i = 1 } ^ { N } \sum _ { t = 1 } ^ { \left| \tau _ { i } \right| } \mathbb { D } _ { \mathrm { K L } } \left( \pi _ { \theta } ( \cdot \mid h _ { i , t } ) \parallel \pi _ { \mathrm { r e f } } ( \cdot \mid h _ { i , t } ) \right) .
$$

The RLVR loss is $\mathcal { L } _ { \mathrm { G R P O } } + \beta _ { \mathrm { r e f } } \mathcal { L } _ { \mathrm { r e f } } ,$ , with regularization weight $\beta _ { \mathrm { r e f } } \geq 0$

Token-level teacher supervision. Beyond outcome-level rewards, some methods use teacher predictions to provide dense token-level supervision (Xu et al., 2025). This is commonly formulated as a forward- or reverse-KL loss between the teacher and student distributions:

$$
\mathcal { L } _ { \mathrm { K L } } ^ { \mathrm { f o r w a r d } } ( \theta ; S ) = \frac { 1 } { | S | } \sum _ { ( i , t ) \in S } \mathbb { D } _ { \mathrm { K L } } \big ( \pi _ { T } ( \cdot \mid h _ { i , t } ) \mid \mid \pi _ { \theta } ( \cdot \mid h _ { i , t } ) \big ) ,
$$

$$
\mathcal { L } _ { \mathrm { K L } } ^ { \mathrm { r e v e r s e } } ( \theta ; S ) = \frac { 1 } { \left| S \right| } \sum _ { ( i , t ) \in S } \mathbb { D } _ { \mathrm { K L } } \big ( \pi _ { \theta } ( \cdot \mid h _ { i , t } ) \mid \big | \pi _ { T } ( \cdot \mid h _ { i , t } ) \big ) ,
$$

where S denotes the token positions (�, �) that receive teacher supervision.

## 3 RLVR through Teacher Intervention and Supervision

Teacher guidance can shape both the trajectories a student explores (i.e., trajectory-level intervention) and the learning signals it receives (i.e., token-level supervision). Existing methods typically emphasize one of these two mechanisms. Below, we examine why either mechanism alone can be insuficient for efective student learning and how combining them motivates the design of PIVOT.

## 3.1 Reward-Bearing Trajectories Are Not Necessarily Learnable

While the student may struggle to discover successful trajectories, trajectory-level intervention by a stronger teacher provides access to reward-bearing reasoning paths.

Supervised fine-tuning (SFT) on these trajectories provides a warm start for subsequent RL (Guo et al., 2025; Wen et al., 2025), while LUFFY incorporates successful teacher trajectories directly into RL training (Yan et al., 2025). POPE instead provides a teacher-generated prefix and lets the student generate the remaining continuation (Qu et al., 2026). These approaches difer in when and how teacher trajectories enter training, but all use teacher-generated trajectories or segments to improve access to successful reasoning paths.

The success of a teacher trajectory, however, does not ensure that its behavior is readily learnable by the student. Full teacher trajectories may difer substantially from the student’s own rollout distribution, making them dificult imitation targets even when they receive positive rewards. Providing only a teacher prefix allows the student to generate its own continuation, but learning to continue from a supplied prefix does not necessarily teach the student to produce the teacher decisions encoded in that prefix during independent generation.

Fig. 2 illustrates this distinction on Survo. POPE yields limited gains even with stronger teacher prefixes. We additionally consider oracle prefixes extracted from correct solutions to test whether prefix errors account for the limited gains. Even with these prefixes, POPE makes little progress, whereas adding explicit teacher supervision on the provided prefixes improves training rewards. This contrast suggests that prefix quality alone does not determine the usefulness of teacher guidance: the training objective must also enable the student to learn from the teacher-provided behavior.

![](images/f8e19cb974ff978b44b0250a6d100b8194bfe48b7ccf50ca8c7bb9452f7c66e3.jpg)  
Figure 2. Prefix supervision vs. prefix-only guidance on Survo. Supervised prefixes improve rewards, while POPE-style prefixes fail to yield comparable gains.

## 3.2 Dense Supervision Does Not Guarantee Trajectory Recovery

Unlike trajectory-level intervention, token-level supervision introduces teacher guidance directly into the learning objective. KDRL (Xu et al., 2025), for example, combines GRPO with reverse-KL distillation along student-generated rollouts. The teacher provides dense feedback at each token, but the student still determines the reasoning path and hence the contexts on which supervision is applied.

This dependence on student-generated contexts can complicate recovery from reasoning errors. As observed in Trajectory-Refined Distillation (TRD) (Jiang et al., 2026), when a student prefix contains errors that require backtracking, the teacher may assign high probability both to continuing the student’s reasoning and to initiating a correction. Under such prefix failure, the resulting bimodal teacher mixture provides conflicting targets for token-level supervision.

Fig. 3 (left) illustrates this issue on a Survo rollout trained with KDRL. After an early error, the student continues along the wrong path and the rollout ends with zero reward. At the supervised positions, teacher predictions repeatedly split between corrective tokens (“Wait”) and tokens that continue the reasoning (“Let”, “So”), pulling the student in two directions at once. Even when the teacher favors a correction, the remainder of the training sequence is still the student’s erroneous continuation. The student thus receives local corrective signals without observing how the proposed recovery unfolds. When successful student rollouts are rare, training provides few examples of how to recover from an initial error and carry the revised reasoning through to a correct solution, limiting the ability of supervision alone to guide the student toward successful completion.

## 3.3 PIVOT: Integrating Trajectory Guidance and Token-Level Supervision

The preceding results suggest that the value of teacher guidance depends on both the reasoning paths it introduces and the supervision applied along those paths. For methods relying solely on trajectory-level intervention, Fig. 2 shows that adding a teacher-matching loss enables the student to learn more efectively from the provided prefixes. For methods that apply token-level supervision solely along student-generated rollouts, Fig. 3 (right) shows how teacher intervention changes the supervision: the teacher generates a recovery path that becomes part of the training sequence and leads to a successful solution, providing largely unimodal targets both when initiating a correction (“Wait”) and when continuing the revised reasoning (“So”).

![](images/9e6bc0a960fc99009103bd6ef481deebd480bdc2bdbf63297f206594c82c3874.jpg)

![](images/10f61296854dfb6a3384a25d187644e4ee22c05b41f6bfc0a4331e8b69548c87.jpg)

![](images/2256720a1e707f9918a97741a2754d4edda8efe54a5e0c734dcb359cbd8bac7c.jpg)  
Figure 3. Dense supervision under KDRL and PIVOT on a Survo rollout; insets show the teacher’s top-2 next-token probabilities at supervised positions. Left: on the student’s failed trajectory, the teacher is split between fixing the error (“Wait”) and continuing along it $( ^ { 6 6 } \mathrm { L e t } ^ { 3 3 } , ^ { 6 } \mathrm { S o } ^ { 3 3 } )$ , giving conflicting targets. Right: the teacher corrects the error itself and continues from there, so its targets are unimodal whether it backtracks (“Wait”) or proceeds (“So”), and the trajectory reaches the reward.

These observations motivate PIVOT’s joint use of teacher-generated segments for trajectory-level intervention and token-level supervision, allowing teacher guidance to shape both exploration and the learning objective. We next describe the rollout construction and training objective.

Teacher-assisted rollout. For each prompt q, we sample a group of rollouts $\mathcal { G } = \mathcal { G } _ { S } \cup \mathcal { G } _ { T }$ , comprising student-only rollouts $\mathcal { G } _ { S }$ and teacher-assisted rollouts $\mathcal { G } _ { T }$ , with sizes $G _ { S }$ and $G _ { T ; }$ , respectively. Student-only rollouts are sampled directly from $\pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid \mathbf { q } )$ . For teacher-assisted rollouts, the student first generates a reasoning prefix, the teacher continues from that prefix for a limited segment, and the student then resumes generation from the resulting context. Writing the resulting rollout as ${ \boldsymbol \tau } = ( u , \nu , w )$ , the student generates � and $w ,$ while the teacher generates �. For an intervention position $\ell _ { u } = | u |$ and teacher segment length $\ell _ { \nu } = | \nu | .$ , this gives the sampling distribution

$$
p _ { \theta _ { \mathrm { o l d } } , T } ^ { ( \ell _ { u } , \ell _ { v } ) } ( \tau \mid \mathbf { q } ) = \pi _ { \theta _ { \mathrm { o l d } } } ( u \mid \mathbf { q } ) \pi _ { T } ( \nu \mid \mathbf { q } , u ) \pi _ { \theta _ { \mathrm { o l d } } } ( w \mid \mathbf { q } , u , \nu ) .
$$

Setting $\ell _ { u } = 0$ places the teacher segment at the beginning of the rollout; otherwise, the teacher conditions its intervention on a student-generated prefix.

Training objective. We combine the RLVR objective with a teacher-supervision loss applied only to teachergenerated tokens. Let $S _ { T }$ denote the set of token positions belonging to the teacher-generated segments � in the training batch. The overall objective is

$$
\begin{array} { r } { \mathcal { L } ( \theta ) = \mathcal { L } _ { \mathrm { G R P O } } ( \theta ) + \beta _ { \mathrm { r e f } } \mathcal { L } _ { \mathrm { r e f } } ( \theta ) + \beta _ { \mathrm { T } } \mathcal { L } _ { \mathrm { K L } } ( \theta ; S _ { T } ) , } \end{array}
$$

where ${ \mathcal { L } } _ { \mathrm { K L } }$ is either the forward- or reverse-KL loss defined in Section 2. Together, the rollout construction and training objective define how PIVOT uses teacher guidance, but leave open where to intervene, how to supervise, and how much guidance to provide, which we study next.

## 4 Towards Efective Teacher Guidance: A Roadmap

We investigate where to intervene (Section 4.2), how to formulate teacher supervision (Section 4.3), and how to configure the remaining guidance settings (Section 4.4) within PIVOT. These analyses yield a practical training recipe, which we evaluate against existing methods in Section 4.5.

## 4.1 Experimental Setup

Tasks. We conduct experiments on hard logic games and mathematical reasoning. For logic games, we train and evaluate on the hard subsets of Survo, Minesweeper, and Kakurasu from SynLogic (Liu et al., 2026), on which the base student almost never succeeds. For mathematical reasoning, we pool problems from Polaris-53K (An et al., 2025) and the math subset of Skywork-OR1-RL-Data (He et al., 2025). We screen this pool with the base student Qwen3-4B, retaining challenging problems with low empirical success rates. After removing problems with invalid answers and removing overlap with out-of-distribution benchmarks, we sample 30k training, 1k validation, and 2k held-out in-distribution test problems (Math-ID) from the filtered pool. We also evaluate on the out-of-distribution benchmarks AIME 2024/2025 (Li et al., 2024) and MATH500 (Hendrycks et al., 2021).

Models and training. We use Qwen3-1.7B as the student for logic games and Qwen3-4B for mathematical reasoning, with Qwen3-4B-Thinking as the teacher in both settings (Yang et al., 2025). All experiments use verl (Sheng et al., 2025) with GRPO (Shao et al., 2024). By default, we sample six student-only and two teacher-assisted rollouts per prompt, all with thinking mode enabled. The maximum response length is 8,192 tokens for logic games and 16,384 tokens for mathematical reasoning; by default, the teacher generates at most $\ell _ { \nu } = 4 , 0 9 6$ tokens per intervention for logic games and 8,192 for math.

Evaluation. We evaluate all models with thinking mode enabled, reporting avg@8/pass@8 for the logic games, Math-ID, and MATH500, and avg@64/pass@64 for AIME 2024/2025. Detailed descriptions of the baselines, datasets, and training hyperparameters are provided in Section A.

## 4.2 Where Should the Teacher Intervene?

To study how intervention timing afects student learning, we compare prefix intervention $( \ell _ { u } = 0 )$ with intervention after a short student attempt $( \ell _ { u } \in \{ 5 1 2 , 1 0 2 4 , 2 0 4 8 \} )$ , keeping the teacher-segment budget fixed. Tab. 1 shows that $\ell _ { u } = 5 1 2$ middle intervention improves avg@8 and pass@8 over prefix intervention across all three games, with the largest gains on Minesweeper and Kakurasu. We include a brief case study to further examine why middle intervention helps in practice.
<table><tr><td rowspan="2">Intervention</td><td colspan="2">Survo</td><td colspan="2">Minesweeper</td><td colspan="2">Kakurasu</td></tr><tr><td>avg@8</td><td>pass@8</td><td>avg@8</td><td>pass@8</td><td>avg@8</td><td>pass@8</td></tr><tr><td>Prefix  $( \ell _ { u } = 0 )$ </td><td>45.90</td><td>87.00</td><td>14.81</td><td>42.50</td><td>0.57</td><td>3.20</td></tr><tr><td>Middle  $( \ell _ { u } = 5 1 2 )$ </td><td> $5 0 . 7 0 \ + 4 . 8 0$ </td><td>90.40+3.40</td><td>19.25 +4.44</td><td>55.50+13.00</td><td> $3 . 5 2 \ + 2 . 9 5$ </td><td>16.60+13.40</td></tr><tr><td>Middle  $( \ell _ { u } = 1 0 2 4 )$ </td><td> $4 9 . 8 8 ~ + 3 . 9 8 ~ $ </td><td>86.40 -0.60</td><td>17.94+3.13</td><td>55.00+12.50</td><td> $3 . 7 7 + 3 . 2 0 $ </td><td>15.20+12.00</td></tr><tr><td>Middle  $( \ell _ { u } = 2 0 4 8 )$ </td><td> ${ \pmb 5 } { \pmb 1 } . { \pmb 5 } { \pmb 5 } + { \pmb 5 } . 6 { \pmb 5 }$ </td><td>87.20+0.20</td><td>12.50 -2.31</td><td> $4 3 . 5 0 \ \overline { { + 1 . 0 0 } }$ </td><td> $3 . 3 2 \ + 2 . 7 5$ </td><td> $1 4 . 0 0 \substack { + 1 0 . 8 0 }$ </td></tr></table>

Table 1. Efect of the intervention position $\ell _ { u }$ on hard logic games. $\ell _ { u } = 0$ is prefix intervention.

![](images/c851ad3e0ea48c8045cbb8c412684575f1ee05f4efcb693c4d103c7ea25e47ac.jpg)

<table><tr><td colspan="7">Why does middle intervention help? We compare two explanations for the effect of intervention timing:</td></tr><tr><td colspan="7">1. Later tokens are more informative. Early tokens may largely restate the problem or organize its conditions. Moving supervision later could thus make better use of the same distillation budget. 2. Student-conditioned guidance is more learnable. Conditioning teacher guidance on the student&#x27;s partial solution may make it easier for the student to learn how to continue or recover. We test these explanations</td></tr><tr><td rowspan="7">by replacing the student-generated prefix with a teacher-generated one, leaving the rest of the rollout unchanged. Since the teacher may provide a stronger starting point, if middle intervention helps by shifting supervision to more informative tokens, this variant should match or surpass student-prefix intervention. Yet this replacement substantially reduces performance (Fig. 4), supporting the second explanation. This result echoes the</td><td colspan="2"></td><td></td><td></td><td colspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td colspan="2"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td><img src="images/4077bd99daee2273fd347cc348f717c971a3d6697be20078e023b9ba0ac907ec.jpg"/></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="8">motivation behind DAgger (Ross et al., 2011) and on-policy distillation (Agarwal et al., Figure 4. Effect of opening source on Minesweeper. 2024), which expose the learner to states from its own rollouts. From these states, the teacher</td></tr></table>

A longer student prefix, however, does not necessarily make this guidance more efective. On Minesweeper, for example, avg@8 falls from 19.25% at $\ell _ { u } = 5 1 2$ to 12.50% at $\ell _ { u } = 2 0 4 8 _ { \mathrm { ~ ~ } }$ , below the 14.81% achieved by prefix intervention. We also observe more truncation with later intervention. A longer student prefix leaves fewer tokens to continue generating after the teacher segment, especially when earlier steps need to be revised. As a result, the rollout may reach the length limit before the student can complete the revised reasoning and produce a final answer.

![](images/c072dadb0e5d787960299a370c258df3fb6bc6a6991f918e3f51c67054e12218.jpg)

![](images/90767dd1bbebb70d57f862803ee6debf84254b0451e34e6a2ce82e37175af3cf.jpg)  
Figure 5. Prefix vs. middle intervention. Guidance is more efective when the teacher can build on the student’s partial solution and correct errors along the way.

Takeaway. Let the student make a short attempt, then guide it from its own reasoning. Intervene early enough to leave room for the student to revise its reasoning and complete the solution.

Base 0.00% 0.05% 0.06% 5.04% 49.17% 60.89% 88.68%

## 4.3 How Should the Teacher Supervise?

Beyond where the teacher intervenes, the form of supervision also matters for student learning. A common approach is to minimize the KL divergence between the student and teacher token distributions, providing dense token-level supervision (Agarwal et al., 2024; Xu et al., 2025). The forward- and reverse-KL objectives introduced in Section 2 both encourage distribution matching, but take expectations under the teacher and student token distributions, respectively. We compare these objectives within PIVOT to examine how the choice of KL direction afects learning from teacher-generated segments across tasks with diferent levels of initial competence.

Fig. 6 shows that reverse KL performs better on all three hard logic games, where the base student almost never succeeds: mean avg@8 rises from 24.5% to 35.6%, and mean pass@8 from 54.2% to 63.8%, relative to forward KL (Tab. 4). On math benchmarks, both objectives improve average accuracy over the base model. On the challenging Math-ID, where the base model achieves only 5.04% accuracy, reverse KL slightly outperforms forward KL by 0.57 percentage points. On the OOD benchmarks AIME 2024/2025 and MATH500, where the student already solves a substantial fraction of problems, forward KL yields modestly larger gains in average accuracy (Tab. 5).

A possible explanation lies in the mode-seeking and mode-covering tendencies of reverse and forward KL (Tajwar et al., 2024; Li et al., 2026). Reverse KL discourages the student from assigning probability to continuations the teacher considers unlikely, favoring concentration on teacher-preferred modes. When the student rarely succeeds, learning to reliably produce even one efective reasoning mode could yield substantial progress. With reliable teacher guidance, this concentration may help the student establish a solution strategy that it previously struggled to pro-

![](images/a3485d144d84dcb0cbc892b46a8b8b08fba5a9b383b3b5114c0f000bdd3fe758.jpg)  
Figure 6. Reverse KL favors low initial accuracy; forward KL has a modest advantage at higher accuracy. Dashed lines mark base accuracy. Panels use diferent scales; labels show gains over the other KL, colored by the better KL.

duce independently. Once successful reasoning is already accessible, learning additional efective strategies may ofer further room for improvement across diferent problems. Forward KL encourages coverage of teacher-supported alternatives, including those the student currently assigns little probability, and may therefore support this broader learning. The results are consistent with this interpretation, though they do not directly measure changes in reasoning-mode coverage.

We therefore favor reverse KL, given its substantial gains on hard tasks and the modest accuracy advantage of forward KL when initial success rates are higher.

![](images/5940c588ca184c63806abc7daefc67348ba33c8fee6228190531bff861dbd613.jpg)  
Figure 7. Calibrating teacher guidance on hard logic games under middle intervention with forward KL. Each column varies one factor from the default setting, which is shaded. Solid lines denote avg@8 and dashed lines denote pass@8. Full numbers are given in Tab. 6 in the appendix.

## 4.4 What Else Matters for Teacher Guidance?

With the intervention position and supervision objective fixed, we investigate which other aspects of teacher guidance afect student learning. We vary teacher capacity, the allocation of a fixed teacher-token budget across assisted rollouts, and the support size of the top-� approximation. Fig. 7 summarizes ablations, which vary one factor at a time while keeping other settings fixed.

Teacher model capacity. We examine whether the student benefits from a stronger teacher by replacing Qwen3-4B-Thinking with Qwen3-30B-A3B-Thinking. The stronger teacher improves avg@8 across all three games: performance more than doubles on Minesweeper, from 19.25% to 40.12%, and increases nearly fivefold on Kakurasu, from 3.52% to 17.27%, while the improvement on Survo is more modest. These results indicate that PIVOT can make efective use of a more capable teacher despite the larger teacher–student capacity gap. The additional capability of the teacher translates into improvements in the student’s own problem-solving performance, even on games where learning with the default teacher remains dificult.

Allocation of a fixed teacher budget. We examine how a fixed teacher-token budget should be distributed across assisted rollouts. Keeping $G _ { T } \times \ell _ { \nu }$ fixed at 8k tokens, we allocate this budget to either two assisted rollouts with a 4k-token teacher budget each, four with 2k each, or eight with 1k each. Both avg@8 and pass@8 decline monotonically across all three games as the budget is spread across more rollouts; on Survo, avg@8 falls from 50.70% to 43.50% and 33.50%. This suggests that longer teacher segments give the student more opportunity to learn from a sustained sequence of connected reasoning or corrective steps, while shorter segments may end before the teacher has fully developed the reasoning needed for the student to continue. Under the same teacher-token budget, fewer, longer interventions therefore provide more efective guidance in our experiments.

Top-� approximation. The top-� approximation controls how much of the teacher’s token distribution is available for supervision. Our results suggest that efective guidance requires a suficiently broad set of teacher-supported alternatives, but does not necessarily benefit from retaining more of the distribution. Moving from � = 10 to � = 30 improves both metrics across all three games, whereas increasing � to 50 ofers no consistent benefit and reduces performance on Kakurasu. This pattern suggests that the most probable teacher tokens alone may provide too narrow a learning signal, while extending supervision to lower-probability alternatives is not uniformly useful. We therefore use � = 30 as a balance between supervision quality and the cost of storing and matching teacher logits.

Takeaway. A stronger teacher provides better guidance, but making efective use of that guidance also requires careful allocation. With a fixed teacher-token budget, fewer, longer interventions are more efective than many short ones. For KL supervision, the teacher distribution should cover enough alternatives for learning; extending to lower-probability tokens ofers no consistent benefit.

## 4.5 Putting the Recipe into Practice

Following the recipe above, we configure PIVOT to intervene after a 512-token student prefix, allowing the teacher to build on the student’s initial reasoning. For each prompt, we sample six student-only and two teacher-assisted rollouts, concentrating the teacher-token budget on fewer, longer interventions. We apply teacher–student KL supervision to the teacher-generated segments with a top-� support of 30. We favor reverse KL for its substantial gains when initial success is rare, while also reporting the forward-KL variant. Tab. 2 compares these configurations with baselines.

<table><tr><td rowspan="2">Method</td><td colspan="3">Hard Logic Games (Qwen3-1.7B)</td><td colspan="6">Math Benchmarks (Qwen3-4B)</td></tr><tr><td>Survo</td><td>Minesweeper</td><td>Kakurasu</td><td>Math-ID</td><td colspan="2">AIME 2024</td><td>AIME 2025</td><td></td><td>MATH500</td></tr><tr><td>Base model</td><td>0.05</td><td>0.06</td><td>0.00</td><td>5.04</td><td>60.89</td><td></td><td>49.17</td><td></td><td>88.68</td></tr><tr><td colspan="8">Baselines</td></tr><tr><td>RL only</td><td>0.05</td><td>1.94+1.88</td><td>0.00</td><td>20.70+15.66</td><td>60.62</td><td>-0.27</td><td>49.74+0.57</td><td></td><td>94.30+5.62</td></tr><tr><td>SFT+RL</td><td>39.77+39.72</td><td>13.44+13.38</td><td>1.38+1.38</td><td>8.82+3.78</td><td></td><td>60.94+0.05</td><td>49.21+0.04</td><td></td><td>89.98+1.30</td></tr><tr><td>LUFFY</td><td>1.30+1.25</td><td>0.81+0.75</td><td>0.00</td><td>16.66+11.62</td><td>60.68</td><td>-0.21</td><td>46.93</td><td>-2.24</td><td>90.60+1.92</td></tr><tr><td>POPE</td><td>0.03 -0.02</td><td>0.00 -0.06</td><td>0.00</td><td>22.09+17.05</td><td>60.00</td><td>-0.89</td><td>46.61</td><td>-2.56</td><td>94.20+5.52</td></tr><tr><td>KDRL</td><td>41.68+41.63</td><td>13.88+13.82</td><td>2.93 +2.93</td><td>17.11+12.07</td><td>49.11</td><td>1-11.78</td><td>43.39</td><td>-5.78</td><td>92.92+4.24</td></tr><tr><td colspan="8">PIVOT</td></tr><tr><td>Forward KL</td><td>50.70+50.65</td><td>19.25+19.19</td><td>3.52+3.52</td><td>22.48+17.44</td><td>63.33 +2.44</td><td></td><td>52.45</td><td>+3.28</td><td></td></tr><tr><td>Reverse KL</td><td>72.90+72.85</td><td>25.44+25.38</td><td>8.38+8.38</td><td>23.05+18.01</td><td>62.24+1.35</td><td></td><td>52.08+2.91</td><td></td><td>95.40 +6.72 95.25+6.57</td></tr></table>

Table 2. Main results on hard logic games (Qwen3-1.7B) and math (Qwen3-4B). Math-ID is in-distribution; other math benchmarks are OOD. The base model uses thinking mode. Metrics are avg@8, or avg@64 for AIME. Colored boxes show absolute changes over the base model.

On the hard logic games, the base student rarely produces a correct solution, and training with outcome rewards alone does little to improve its performance. Adding teacher-generated segments to rollouts, as in LUFFY and POPE, brings only limited improvements, with both methods performing worse than even the simple SFT+RL baseline. KDRL provides token-level supervision along student rollouts and makes greater progress, achieving the strongest baseline performance on all three games. PIVOT goes further by pairing teacher-generated reasoning segments with direct supervision on those segments: with reverse KL, it raises mean avg@8 from KDRL’s 19.5% to 35.6%. A similar advantage appears on the hard math split, where the base model solves only 5.04% of problems. Here, PIVOT reaches 23.05%, exceeding POPE, the strongest baseline, at 22.09%.

On AIME 2024/2025 and MATH500, where the base student already solves a substantial fraction of problems, teacher guidance does not consistently improve performance. LUFFY, POPE, and KDRL improve on MATH500 yet fall below even the base model on both AIME benchmarks, while SFT+RL brings little improvement on either. PIVOT outperforms baselines across all three benchmarks with either KL objective, with forward KL achieving the highest accuracy. PIVOT therefore benefits students both when initial success is rare and when reasoning abilities are already present.

## 5 Conclusion

Efective teacher guidance in RLVR should support successful reasoning while remaining learnable for the student. In this work, we identify two complementary mechanisms for achieving this goal: trajectory-level intervention and token-level supervision. Existing methods typically emphasize one of them, but our analysis shows that either alone can be insuficient for student learning. Building on this insight, we introduce PIVOT, which integrates teacher corrections into student rollouts and applies token-level supervision to those corrections. We examine where to intervene, how to supervise, and how to allocate teacher guidance, yielding a recipe for student learning. Experiments on logic games and mathematical reasoning demonstrate the efectiveness of combining these mechanisms.

Limitations. Our study has several limitations. First, our evaluation focuses on logic games and mathematical reasoning; applicability to code generation, instruction following, and open-ended reasoning remains to be explored. Second, PIVOT uses fixed intervention and supervision settings throughout training, even as the student’s capabilities evolve. How to efectively adapt teacher guidance to the student’s changing needs remains an open question for future work.

## AI use statement

We used generative AI tools to assist with coding and with drafting, revising, and polishing the manuscript. The authors reviewed AI-assisted code and text, verified the scientific claims and citations, and remain responsible for the research design, failure analysis, and interpretation of results. No human-subject data were collected. We take full responsibility for the manuscript, reported results, and accompanying artifacts, including AI-assisted content.

## Reproducibility Statement

We describe dataset construction, student and teacher models, training schedules, optimization settings, teacher-intervention budgets, KL-supervision settings, and evaluation protocols in Appendix A. Full experimental results are provided in Appendix B. After the review process concludes, we will open-source the complete training code and publicly release the PIVOT model checkpoints to support reproduction of our results and further research on teacher-guided reinforcement learning with verifiable rewards.

## References

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations, volume 2024, pp. 21246–21263, 2024.

Chenxin An, Zhihui Xie, Xiaonan Li, Lei Li, Jun Zhang, Shansan Gong, Ming Zhong, Jingjing Xu, Xipeng Qiu, Mingxuan Wang, et al. Polaris: A post-training recipe for scaling reinforcement learning on advanced reasoning models, 2025. URL https://hkunlp. github. io/blog/2025/Polaris, 2, 2025.

Justin Chih-Yao Chen, Becky Xiangyu Peng, Prafulla Kumar Choubey, Kung-Hsiang Huang, Jiaxin Zhang, Mohit Bansal, and Chien-Sheng Wu. Nudging the boundaries of llm reasoning. arXiv preprint arXiv:2509.25666, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, et al. Skywork open reasoner 1 technical report. arXiv preprint arXiv:2505.22312, 2025.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Li Jiang, Haoran Xu, Yichuan Ding, and Amy Zhang. Trajectory-refined distillation. arXiv preprint arXiv:2606.08432, 2026.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

Jia Li, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Huang, Kashif Rasul, Longhui Yu, Albert Q Jiang, Ziju Shen, et al. Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions. Hugging Face repository, 13(9):9, 2024.

Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan-ang Gao, Wenkai Yang, Zhiyuan Liu, and Ning Ding. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. arXiv preprint arXiv:2604.13016, 2026.

Junteng Liu, Yuanxiang Fan, Jiang Zhuo, Han Ding, Yongyi Hu, Chi Zhang, Yiqi Shi, Shitong Weng, Aili Chen, Shiqi Chen, et al. Synlogic: Synthesizing verifiable reasoning data at scale for learning logical reasoning and beyond. Advances in Neural Information Processing Systems, 38:100976–100997, 2026.

Yuxiao Qu, Amrith Setlur, Virginia Smith, Ruslan Salakhutdinov, and Aviral Kumar. Pope: Learning to reason on hard problems via privileged on-policy exploration. arXiv preprint arXiv:2601.18779, 2026.

Stéphane Ross, Geofrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the fourteenth international conference on artificial intelligence and statistics, pp. 627–635. JMLR Workshop and Conference Proceedings, 2011.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and eficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pp. 1279–1297, 2025.

Fahim Tajwar, Anikait Singh, Archit Sharma, Rafael Rafailov, Jef Schneider, Tengyang Xie, Stefano Ermon, Chelsea Finn, and Aviral Kumar. Preference fine-tuning of llms should leverage suboptimal, on-policy data. arXiv preprint arXiv:2404.14367, 2024.

Leitian Tao, Ilia Kulikov, Swarnadeep Saha, Tianlu Wang, Jing Xu, Sharon Li, Jason E Weston, and Ping Yu. Hybrid reinforcement: When reward is sparse, it’s better to be dense. arXiv preprint arXiv:2510.07242, 2025.

Liang Wen, Yunke Cai, Fenrui Xiao, Xin He, Qi An, Zhenyu Duan, Yimin Du, Junchen Liu, Tanglifu Tanglifu, Xiaowei Lv, et al. Light-r1: Curriculum sft, dpo and rl for long cot from scratch and beyond. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 6: Industry Track), pp. 318–327, 2025.

Yangzhen Wu, Shanda Li, Zixin Wen, Xin Zhou, Ameet Talwalkar, Yiming Yang, Wenhao Huang, and Tianle Cai. Learn hard problems during rl with reference guided fine-tuning. arXiv preprint arXiv:2603.01223, 2026.

Hongling Xu, Qi Zhu, Heyuan Deng, Jinpeng Li, Lu Hou, Yasheng Wang, Lifeng Shang, Ruifeng Xu, and Fei Mi. Kdrl: Post training reasoning llms via unified knowledge distillation and reinforcement learning. arXiv preprint arXiv:2506.02208, 2025.

Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. Learning to reason under of-policy guidance. arXiv preprint arXiv:2504.14945, 2025.

Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. Learning to reason under of-policy guidance. Advances in Neural Information Processing Systems, 38:117157–117186, 2026.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.

Rosie Zhao, Alexandru Meterez, Sham Kakade, Cengiz Pehlevan, Samy Jelassi, and Eran Malach. Echo chamber: Rl post-training amplifies behaviors learned in pretraining. arXiv preprint arXiv:2504.07912, 2025.

## A Experimental Setup

We detail the dataset construction, training configuration, evaluation protocol, and baseline implementations.

Dataset construction. For synthetic logic games, we follow SynLogic (Liu et al., 2026) and construct three datasets: Survo, Minesweeper, and Kakurasu. Each game is generated at a hard dificulty setting, controlled by grid size, the number of unknown entries, and constraint density, at which the base student is near zero accuracy. Each game has 2k training problems and a held-out test split of 500 problems. For mathematical reasoning, we pool problems from Polaris-53K (An et al., 2025) and the math subset of Skywork-OR1-RL-Data (He et al., 2025), and screen them with the base student: Qwen3-4B samples 8 responses per problem, and we keep problems solved at most 3 times. After removing problems with invalid gold answers and decontaminating against AIME 2024/2025 and MATH500, we sample 30k problems for training, 1k for validation, and a held-out in-distribution test split of 2k problems, denoted Math-ID; 77.7% of the training problems are never solved by the base student.

Training setup. All experiments are conducted using the verl framework (Sheng et al., 2025). We use GRPO (Shao et al., 2024) as the policy optimization algorithm, with reference KL coeficient $\beta _ { \mathrm { r e f } } = 0 . 0 1$ and clipping coeficient $\varepsilon = 0 . 2$ . Unless otherwise specified, we use a learning rate of $3 \times 1 0 ^ { - 6 } .$ , a rollout batch size of 64 prompts, a mini-batch size of 32, 8 rollouts per prompt, a maximum prompt length of 2,048 tokens, and sampling temperature 0.7 for both student and teacher. Logic-game experiments use Qwen3-1.7B as the student and Qwen3-4B-Thinking as the teacher (Yang et al., 2025); we train for 3 epochs with a maximum response length of 8,192 tokens and compute rewards with the SynLogic verifier. Math experiments use Qwen3-4B as the student and the same teacher; we train for 1 epoch with a maximum response length of $^ { 1 6 , 3 8 4 }$ tokens and compute rewards with Math-Verify. No additional format or length rewards are used.

PIVOT configuration. For each prompt, we sample $G _ { S } + G _ { T } = 8$ rollouts, where $G _ { S } = 6$ are student-only rollouts and $G _ { T } = 2$ are teacher-assisted rollouts. By default, the teacher intervenes after a student prefix of $\ell _ { u } = 5 1 2$ tokens and generates at most $\ell _ { \nu } = 4 , 0 9 6$ tokens for logic games and $\ell _ { \nu } = 8 , 1 9 2$ tokens for math. The student then resumes autoregressive generation from the teacher-augmented context. The teacher-generated segment is supervised with a token-level KL objective toward the teacher distribution; the placement and calibration ablations use forward KL, and we compare both directions in Section 4.3. The KL is computed on the teacher’s top-30 tokens, with the remaining probability mass collected into a single tail bucket. The teacher KL coeficient $\beta _ { \mathrm { T } }$ is cosine-decayed from 0.1 to 0. Ablations on the intervention position $\ell _ { u } ,$ the teacher model, the allocation of the teacher-token budget between $G _ { T }$ and $\ell _ { \nu } ,$ and the distillation top-� are provided in Sections 4.2 and 4.4, with full numbers in Tables 1 and 6.

Evaluation. For synthetic logic games, we evaluate models on the held-out hard splits and report avg@8 and pass@8. For mathematical reasoning, we evaluate in distribution on Math-ID and out of distribution on MATH500 (Hendrycks et al., 2021) and AIME 2024/2025 (Li et al., 2024), reporting avg@8 and pass@8 for Math-ID and MATH500 and avg@64 and pass@64 for AIME 2024/2025. Decoding uses temperature 0.7 and the training response-length limit.

Baseline methods. We compare PIVOT with five representative baselines that address sparse rewards in diferent ways. All baselines share the GRPO setup above and the budget of 8 rollouts per prompt. RL only is the standard on-policy RLVR baseline without a teacher. SFT+RL first fine-tunes the student on 5k verified-correct teacher-generated traces from the same training distribution and then continues RLVR training. LUFFY (Yan et al., 2025) replaces one of the 8 rollouts with a full teacher-generated trajectory and trains on it with LUFFY’s policy-shaping objective $( \gamma = 0 . 1 )$ . POPE (Qu et al., 2026) starts 2 of the 8 rollouts from a teacher-generated prefix whose length matches our default segment length $\ell _ { \nu }$ (4,096 tokens for logic games, 8,192 for math); prefix tokens are excluded from all losses and the student completes the rollout. KDRL (Xu et al., 2025) adds no teacher tokens to the trajectory; it combines GRPO with a reverse-KL distillation term on every student-generated token, computed on the student’s top-30 tokens with a constant coeficient of 0.1.

## B Additional Experimental Results

We complement the main results with a prefix-only comparison, complete benchmark results, and teacher guidance ablations. In the benchmark tables, small signed values give absolute percentage-point changes from the base model; bold values identify the best result in each column.

## B.1 Prefix-Only Guidance

Table 3 compares POPE-style prefix-only guidance with supervised prefix training on Survo. Oracle prefixes are verifier-filtered teacher rollouts. They improve on noisy prefixes, but both prefix-only variants remain below 1% on avg@8 and pass@8. Prefix-Fwd reaches 59.05 avg@8 and 94.20 pass@8, suggesting that teacher prefixes provide a stronger learning signal when they are directly supervised rather than supplied only as context.

<table><tr><td>Method</td><td>avg@8</td><td>pass@8</td></tr><tr><td>POPE w/ noisy prefix</td><td>0.03</td><td>0.20</td></tr><tr><td>POPE w/ oracle prefix</td><td>0.14</td><td>0.91</td></tr><tr><td>Prefix-Fwd</td><td>59.05</td><td>94.20</td></tr></table>

Table 3. Prefix-only guidance and supervised prefix training on Survo. Oracle prefixes use verifier-filtered teacher rollouts.

## B.2 Hard Logic Games

Table 4 reports avg@8 and pass@8 on the held-out hard splits of Survo, Minesweeper, and Kakurasu. Both PIVOT variants outperform every baseline on all six metrics. Reverse KL achieves the highest scores throughout, with the largest absolute gain on Survo: 72.85 percentage points in avg@8 and 97.00 points in pass@8 over the base model. Kakurasu remains the most challenging game, with a best avg@8 of 8.38 despite these gains.

<table><tr><td rowspan="2">Method</td><td colspan="4">Survo</td><td colspan="4">Minesweeper</td><td colspan="4">Kakurasu</td></tr><tr><td>avg@8</td><td colspan="2"></td><td colspan="2">pass@8</td><td colspan="2">avg@8 pass@8</td><td colspan="2">avg@8</td><td colspan="2">pass@8</td></tr><tr><td>Base model</td><td>0.05</td><td></td><td>0.40</td><td></td><td>0.06</td><td></td><td>0.50</td><td></td><td>0.00</td><td></td><td>0.00</td></tr><tr><td>Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RL only</td><td>0.05</td><td></td><td>0.40</td><td></td><td>1.94</td><td>+1.88</td><td>8.00 +7.50</td><td></td><td>0.00</td><td>0.00</td><td></td></tr><tr><td>SFT+RL</td><td>39.77</td><td>+39.72</td><td>86.40+86.00</td><td></td><td>13.44 +13.38</td><td></td><td>41.50 +41.00</td><td>1.38</td><td>+1.38</td><td>7.60</td><td>+7.60</td></tr><tr><td>LUFFY</td><td>1.30</td><td>+1.25</td><td>9.00</td><td>+8.60</td><td>0.81</td><td>+0.75</td><td>5.00 +4.50</td><td>0.00</td><td></td><td>0.00</td><td></td></tr><tr><td>POPE</td><td>0.03</td><td>-0.02</td><td>0.20</td><td>-0.20</td><td>0.00</td><td>-0.06</td><td>0.00 -0.50</td><td></td><td>0.00</td><td>0.00</td><td></td></tr><tr><td>KDRL</td><td>41.68</td><td>+41.63</td><td>81.40+81.00</td><td></td><td>13.88 +13.82</td><td></td><td>42.50 +42.00</td><td>2.93</td><td>+2.93</td><td>12.40</td><td>+12.40</td></tr><tr><td>PIVOT</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Forward KL</td><td>50.70</td><td>+50.65</td><td>90.40</td><td>+90.00</td><td>19.25 +19.19</td><td></td><td>55.50 +55.00</td><td></td><td>3.52 +3.52</td><td>16.60</td><td>+16.60</td></tr><tr><td>Reverse KL</td><td>72.90</td><td>+72.85</td><td>97.40</td><td>+97.00</td><td>25.44 +25.38</td><td></td><td>63.00 +62.50</td><td>8.38</td><td>+8.38</td><td>31.00</td><td>+31.00</td></tr></table>

Table 4. Full results on hard logic games (Qwen3-1.7B; base model in thinking mode). Signed values show changes from the base model. Shading identifies PIVOT; bold indicates the best result in each column.

## B.3 Mathematical Reasoning

Table 5 separates the full math results by evaluation budget: avg@8/pass@8 for Math-ID and MATH500, and avg@64/pass@64 for AIME 2024/2025. Reverse KL performs best on the held-out in-distribution split (Math-ID), while forward KL achieves the highest scores on all three out-of-distribution benchmarks. Both PIVOT variants improve avg@� over the base model on every benchmark, with no decrease in pass@�.

Math-ID and MATH500
<table><tr><td>Method</td><td colspan="4">Math-ID</td><td colspan="4">MATH500</td></tr><tr><td></td><td colspan="2">avg@8</td><td colspan="2">pass@8</td><td colspan="2">avg@8</td><td colspan="2">pass@8</td></tr><tr><td>Base model</td><td>5.04</td><td></td><td>22.20</td><td></td><td>88.68</td><td></td><td>92.20</td><td></td></tr><tr><td>Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RL only</td><td>20.70</td><td>+15.66</td><td>37.00</td><td>+14.80</td><td>94.30</td><td>+5.62</td><td>98.00</td><td>+5.80</td></tr><tr><td>SFT+RL</td><td>8.82</td><td>+3.78</td><td>28.20</td><td>+6.00</td><td>89.98</td><td>+1.30</td><td>94.20</td><td>+2.00</td></tr><tr><td>LUFFY</td><td>16.66</td><td>+11.62</td><td>33.00</td><td>+10.80</td><td>90.60</td><td>+1.92</td><td>97.20</td><td>+5.00</td></tr><tr><td>POPE</td><td>22.09</td><td>+17.05</td><td>38.90</td><td>+16.70</td><td>94.20</td><td>+5.52</td><td>97.60</td><td>+5.40</td></tr><tr><td>KDRL</td><td>17.11</td><td>+12.07</td><td>30.20</td><td>+8.00</td><td>92.92</td><td>+4.24</td><td>96.80</td><td>+4.60</td></tr><tr><td>PIVOT</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Forward KL</td><td>22.48</td><td>+17.44</td><td>41.70</td><td>+19.50</td><td>95.40</td><td>+6.72</td><td>98.80</td><td>+6.60</td></tr><tr><td>Reverse KL</td><td>23.05</td><td>+18.01</td><td>42.75</td><td>+20.55</td><td>95.25</td><td>+6.57</td><td>98.60</td><td>+6.40</td></tr></table>

AIME 2024 and AIME 2025
<table><tr><td rowspan="3">Method</td><td colspan="4">AIME 2024</td><td colspan="4">AIME 2025</td></tr><tr><td colspan="2">avg@64</td><td colspan="2">pass@64</td><td colspan="2">avg@64</td><td colspan="2">pass@64</td></tr><tr><td>60.89</td><td></td><td>83.33</td><td></td><td>49.17</td><td></td><td>76.67</td><td></td></tr><tr><td colspan="2">Baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RL only</td><td></td><td>60.62</td><td>-0.27</td><td>83.33</td><td></td><td>49.74</td><td>+0.57</td><td>76.67</td></tr><tr><td>SFT+RL</td><td></td><td>60.94</td><td>+0.05</td><td>83.33</td><td></td><td>49.21</td><td>+0.04</td><td>76.67</td></tr><tr><td>LUFFY</td><td></td><td>60.68</td><td>-0.21</td><td>83.33</td><td></td><td>46.93</td><td>-2.24</td><td>76.67</td></tr><tr><td>POPE</td><td></td><td>60.00</td><td>-0.89</td><td>83.33</td><td></td><td>46.61</td><td>-2.56</td><td>76.67</td></tr><tr><td>KDRL</td><td></td><td>49.11</td><td>-11.78</td><td>76.67</td><td>-6.66</td><td>43.39</td><td>-5.78</td><td>63.33 -13.34</td></tr><tr><td colspan="2">PIVOT</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Forward KL</td><td></td><td>63.33</td><td>+2.44</td><td>86.67 +3.34</td><td></td><td>52.45 +3.28</td><td>80.00</td><td>+3.33</td></tr><tr><td>Reverse KL</td><td></td><td>62.24</td><td>+1.35</td><td>83.33</td><td></td><td>52.08</td><td>+2.91</td><td>76.67</td></tr></table>

Table 5. Full mathematical reasoning results (Qwen3-4B, thinking mode). Signed values show changes from the base model. The two panels use diferent sampling budgets; bold identifies the best result for each benchmark and metric.

## B.4 Teacher-Guidance Ablations

Table 6 gives the complete results behind Figure 7. Each block changes one factor while retaining middle intervention and forward KL; the default setting is shaded and repeated for reference. The prefix-length ablation appears in Table 1.

A larger teacher improves avg@8 on all three games, although Survo pass@8 decreases slightly. At a fixed teacher-token budget, two longer segments outperform four or eight shorter segments on every metric. For the teacher’s top-� distribution, the default � = 30 leads on four of the six metrics; � = 50 gives the highest Survo avg@8 and Minesweeper pass@8.
<table><tr><td rowspan="2">Setting</td><td colspan="2">Survo</td><td colspan="2">Minesweeper</td><td colspan="2">Kakurasu</td></tr><tr><td>avg@8</td><td>pass@8</td><td>avg@8</td><td>pass@8</td><td>avg@8</td><td>pass@8</td></tr><tr><td>Teacher capacity</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-4B-Thinking *</td><td>50.70</td><td>90.40</td><td>19.25</td><td>55.50</td><td>3.52</td><td>16.60</td></tr><tr><td>Qwen3-30B-A3B-Thinking</td><td>53.75</td><td>88.80</td><td>40.12</td><td>61.40</td><td>17.27</td><td>50.20</td></tr><tr><td>Fixed teacher budget  $( G _ { T } \times \ell _ { \nu } = 8 \mathrm { k } )$ </td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $2 \times 4 \mathrm { k } ^ { * }$ </td><td>50.70</td><td>90.40</td><td>19.25</td><td>55.50</td><td>3.52</td><td>16.60</td></tr><tr><td> $4 \times 2 \mathrm { k }$ </td><td>43.50</td><td>84.40</td><td>17.94</td><td>55.00</td><td>1.80</td><td>7.50</td></tr><tr><td>8×1k</td><td>33.50</td><td>74.20</td><td>15.19</td><td>44.50</td><td>0.93</td><td>4.40</td></tr><tr><td>Teacher top-k</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>k = 10</td><td>35.23</td><td>75.40</td><td>17.38</td><td>50.50</td><td>2.55</td><td>14.00</td></tr><tr><td> $k = 3 0 ^ { \ast }$ </td><td>50.70</td><td>90.40</td><td>19.25</td><td>55.50</td><td>3.52</td><td>16.60</td></tr><tr><td> $k = 5 0$ </td><td>51.27</td><td>89.40</td><td>19.19</td><td>58.00</td><td>1.83</td><td>12.60</td></tr></table>

Table 6. Teacher-guidance ablations with middle intervention and forward KL. Shaded rows repeat the default setting (<sup>∗</sup>). Bold values identify the best result within each block.

## C Case Studies on Intervention Placement

We trace one Survo puzzle through the three intervention positions in Table 1 (trained student, step 60). The comparison asks whether the resulting state can be completed by the student, and whether enough response budget remains to emit the answer.

Shared puzzle Prompt and controlled continuations   
8 2 7 8 X 2 32 Fill the 10 cells marked X with $[ 5 , 8 , 4 , 6 , 2 , 6 , 4 , 8 , 8 , 5 ]$ , each used once, so   
5 5 6 5 7 1 29 that the last number of every row and column is the sum of the others. Answer   
3 6 7 5 9 32 with the filled matrix in a Python code block.   
For each $\ell _ { u } \in \{ 0 , 5 1 2 , 2 0 4 8 \}$ , freeze the rollout from the corresponding train-  
2 6 8 4 31   
ing run after the student prefix and the 4,096-token teacher segment. Con-  
7 3 X 6 35   
X X 5 9 X X 38 tinue that exact context four times each with the teacher and one fixed student (the ℓ = 512 checkpoint, step 60; temperature 0.7). The 8,192-token re-  
29 30 39 35 34 30 197 sponse limit leaves 4,096, 3,584, or 2,048 continuation tokens, respectively.

Excerpts follow generation order. Token ofsets [+k] start at the continuation; other ofsets start at the response. Pink marks wasted or failing reasoning; green marks progress.

<table><tr><td colspan="2">Case 1 / Prefix intervention  $( \ell _ { u } = 0 )$ </td></tr><tr><td colspan="2">TRAJECTORY EXCERPTS OBSERVATIONS</td></tr><tr><td>Teacher segment</td><td>tokens 0-4,096</td></tr><tr><td>1 [0.5k]I think I have a misunderstanding. Let me read the problem again.[1.0k]I think there's confusion in indexing.</td><td>1 Slow start. Re-reads the task for ~1.5k tokens.</td></tr><tr><td>2 [2.5k]Yes, so row 0, col4 must be 5.[3.0k]So row 2, col3 must be 2. 3 [3.4k]Let's use different notation.[...] let A = M[3][3], B = M[3][4],</td><td>2 2/10 cells solved. Most cells remain unresolved at the hand-off.</td></tr><tr><td>so A + B = 11 [.. . ] E + F + G + H = 24[4. 1k] (segment ends)</td><td>3 Notation changes. Eight open cells are renamed  $A , \ldots , H ;$  the student inherits this system.</td></tr><tr><td>Student continuation</td><td>budget 4,096 4 Student: 0/4. Drops E, F and loops over</td></tr><tr><td>4[+0.5k] withG + H = 24[...] 14 &lt; 24.Contradiction.Wait, this can't be.What's wrong?[+2.4k]Contradiction again.[+4.1k] (length limit, no answer)</td><td>contradictions, or solves the grid but re-checks until truncation.</td></tr><tr><td>Teacher continuation from the same hand-off</td><td></td></tr><tr><td>5 [+0k]thus E = 4[...]thus C = 6[...] All good. [+3.7k]``python</td><td>5 Teacher: 4/4. Completes from the same state and with the same remaining budget.</td></tr><tr><td colspan="2">Case 1 (continued) / Middle intervention (lu = 512)</td></tr><tr><td colspan="2">TRAJECTORY EXCERPTS OBSERVATIONS</td></tr><tr><td>Student prefix 1 [0.1k] Row 0: [8, 2, 7, 8, 'X’, 2, 32] [...] Row 6: [29, 30, 39, 35, 34,</td><td>tokens 0–512 1 Student's set-up. Lists the rows and</td></tr><tr><td>30, 197] [0.5k]Let me count.</td><td>states the sum rule, giving the teacher a context built by the student.</td></tr><tr><td>Teacher segment 2 [0.6k] 8+2+7+8+2 = 27, so 27 + X = 32,so X = 5.</td><td>tokens 512–4,608 2 Immediate progress. First cell solved 87</td></tr><tr><td>3 [2.9k]E is (5,0), so (5,0) = 4[3.3k]thus C = 6[4.1k]thus H=8 [4.3k]Perfect.[4.6k] (segment ends)</td><td>tokens after take-over, versus 2.5k from scratch; the teacher uses the student's notation. 3 10/10 cells solved. All cells are fixed and verified at hand-off; 3,584 tokens remain for</td></tr><tr><td>Student continuation</td><td>the response. budget 3,584</td></tr><tr><td>4 [+1.4k]All good.&lt;/think&gt; ``python</td><td>4 Student and teacher: 4/4. The student</td></tr><tr><td></td><td>finishes in 1.4k–1.8k tokens.</td></tr></table>

Figure 8. Prefix versus middle intervention on one Survo puzzle. With prefix intervention, the hand-of state is completed by the teacher (4/4), but not by the student (0/4). The matched middle intervention continues on the next page.  
Figure 8. Prefix versus middle intervention (continued). The student-conditioned middle state is completed by both models. All excerpts and counts refer to the same puzzle and the controlled continuations described on the preceding page.

Comparison at the hand-of
<table><tr><td></td><td>Prefix</td><td>Middle</td></tr><tr><td>Student prefix</td><td>0 tokens</td><td>512 tokens</td></tr><tr><td>Cells solved at hand-off</td><td>2/10</td><td>10/10</td></tr><tr><td>Continuation budget</td><td></td><td>4,096 tokens 3,584 tokens</td></tr><tr><td>Student completions</td><td>0/4</td><td>4/4</td></tr><tr><td>Teacher completions</td><td>4/4</td><td>4/4</td></tr></table>

Observation. Both frozen states are solvable for the teacher under the original response limit. Only the middle state is completed by the student, even though it leaves fewer continuation tokens.

Interpretation. In this example, intervening after the student’s initial set-up lets the teacher make earlier progress and return a fully resolved grid. The prefix intervention instead returns an incomplete state with new notation that the student struggles to use. These four continuations illustrate the distinction between a state the teacher can solve and a state the student can reliably finish; they do not isolate notation or timing as the sole cause.

<table><tr><td colspan="2">Case 2 / Late intervention  $( \ell _ { u } = 2 0 4 8 )$ </td></tr><tr><td colspan="2">TRAJECTORY EXCERPTS OBSERVATIONS</td></tr><tr><td>Student prefix</td><td>tokens 0–2,048</td></tr><tr><td>1 [0.5k]So there are 12 empty cells, but the list says 12?[...] total 1+2+2+2+2=9? [...]= 10 empty cells.[1.8k] Let me write the equations.</td><td>1 0/10 cells solved. The student recounts cells and begins writing equations without solving a cell in 2,048 tokens.</td></tr><tr><td>Teacher segment</td><td>tokens 2,048–6,144</td></tr><tr><td>2[2.3k]thus X=5.[2.4k]So row2, col3 should be 2.[6.1k] Let I = (4,2), J = (4,4) = D. So row4: col0=7, col1=3, col2=I,col3=(segment cap)</td><td>2 6/10 cells solved. The teacher makes partial progress before its segment ends. Only 2,048 continuation tokens remain.</td></tr><tr><td>Student continuation</td><td>budget 2,048</td></tr><tr><td>3 [+0.6k]D=8,so (4,4)=8[...]So(4,2)=6</td><td>3 Grid solved. The last four cells are fixed within 0.7k tokens.</td></tr><tr><td>4[+2.0k]All good.Now, write the matrix in Python(length limit)</td><td>4 Student and teacher: 0/4. Re-checking exhausts the remaining budget before the answer is emitted.</td></tr><tr><td>Teacher continuation</td><td>enlarged budget 8,192</td></tr><tr><td>5 [+2.0k]``python [+2.3k] &lt;/think&gt;</td><td>5 Teacher: 4/4. With more continuation budget, the teacher finishes in 2.4k–3.8k tokens.</td></tr></table>

Figure 9. Late intervention on the same Survo puzzle. With $\ell _ { u } = 2 0 4 8 _ { \mathrm { ; } }$ , the teacher and student solve the remaining cells from the hand-of state but exhaust the original budget before emitting the answer. Extending the teacher’s continuation budget restores completion.

Remaining budget and response completion
<table><tr><td>Continuation from the late state</td><td>Budget</td><td>Completions</td></tr><tr><td>Student</td><td>2,048 tokens</td><td>0/4</td></tr><tr><td>Teacher</td><td>2,048 tokens</td><td>0/4</td></tr><tr><td>Teacher, enlarged budget</td><td>8,192 tokens</td><td>4/4</td></tr></table>

Observation. The student resolves the remaining cells within 0.7k tokens, yet reaches the response limit before producing the requested answer. The teacher also fails under the 2,048-token continuation budget and succeeds in all four continuations when given 8,192 tokens.

Interpretation. The failure in this late-intervention example is consistent with insuficient budget for completing the response after solving the puzzle. A useful hand-of must leave time for verification and answer emission as well as for the remaining reasoning.