![](images/86035b4eb5c18f938e3b3406bd09cb6203dd922225f23fb28c61e219ec58981e.jpg)

# Agent-G<sup>2</sup>: Gaussian Guidance for Agentic Reinforcement Learning

Zixuan Wang<sup>1,2</sup>\* Yanrui Miao<sup>1,3</sup>\* Zhengxi Lu<sup>1</sup> Teng Pan<sup>1,2</sup>

Yiwen Qiu<sup>1</sup> Hongxing Li<sup>1</sup> Peng Qiu<sup>2</sup> Ruiqing Zhang<sup>2</sup> Yongliang Shen<sup>1†</sup>

<sup>1</sup>Zhejiang University <sup>2</sup>Baidu Inc. <sup>3</sup>Shandong University

{wang.zixuan, syl}@zju.edu.cn

§ Code  Project Page Models

## Abstract

Hint-based reinforcement learning addresses reward sparsity in long-horizon agentic tasks by retaining a prefix of an expert trajectory before each rollout, letting the policy explore from a state closer to success. Its effectiveness hinges on the guidance depth: how much of the trajectory to keep. Existing methods treat this depth as a deterministic scalar. Scheduled approaches share one value across samples and ignore per-task heterogeneity; per-sample probing estimates it separately at the cost of extra rollouts. We find that useful guidance occupies a band of depths whose informativeness profile is approximately Gaussian around the band center, rather than concentrating at a single optimal point. We propose Agent-G<sup>2</sup>, a Gaussian guidance framework that draws the depth per task from a Gaussian whose center and spread are estimated online from rollouts already collected for policy optimization, requiring no probe rollouts or learned depth predictor. The center combines a global baseline with per-cluster difficulty, and the spread tracks within-cluster variance. We evaluate Agent-G<sup>2</sup> on ALFWorld and WebShop on Qwen2.5-1.5B / 7B-Instruct. Agent-G<sup>2</sup> outperforms the strongest hint-based, hint-free, and Aux-RL baselines on ALFWorld by 2.3 / 3.9 / 7.4 points at under one-third the rollout cost of per-sample probing.

## 1 Introduction

LLM agents trained with reinforcement learning have made progress on long-horizon decisionmaking tasks (Shen et al., 2023; Zhang et al., 2026a), including web navigation (Yao et al., 2023a; Zhou et al., 2024a; Deng et al., 2023), embodied control (Ahn et al., 2022; Huang et al., 2022; Wang et al., 2025c), and scientific experimentation (Boiko et al., 2023; Bran et al., 2023). A recurring obstacle is reward sparsity: tasks span dozens of sequential decisions, yet only a binary signal is issued at termination, and on-policy exploration from the initial state rarely reaches a successful terminal state. Hint-based RL mitigates this obstacle by retaining a prefix of an expert trajectory before each rollout, so the policy explores from a state closer to success (Xi et al., 2024a; Zhang et al., 2025a; Su et al., 2025), directly addressing advantage collapse in GRPO-style training (Zhang et al., 2025b; Wang et al., 2025b; Li et al., 2025).

![](images/443e66334c5d112aec50d44b3d7a9f047ee438e6d0c320609d0520efedf42e25.jpg)  
Figure 1: Hint-based RL paradigms across training steps $t _ { 0 } , t _ { 1 } , t _ { 2 } .$ . (a) Schedule-based: shared $d _ { t }$ across samples. (b) Probe-based: O(log n) per-sample probes. (c) Agent-G<sup>2</sup> (ours): $d _ { i } \sim \mathcal { N } ( \mu _ { t } , \sigma _ { t } ^ { 2 } )$ per task, estimated online from existing rollouts.

The effectiveness of hint-based RL depends critically on the guidance depth: how much of the expert trajectory to retain. Too little guidance leaves few successful rollouts; too much saturates the reward and eliminates the contrast needed for advantage estimation. Despite this sensitivity, existing methods uniformly treat depth as a deterministic scalar, differing only in how they select it (Figure 1). Schedule-based methods (Xi et al., 2024a; Guo et al., 2025; Huang et al., 2026) derive one depth from the training step or batch-level feedback and share it across all samples, ignoring pertask heterogeneity. Per-sample methods (Su et al., 2025; Zhang et al., 2025b; Li et al., 2025; Zhang et al., 2025a) estimate a separate depth through binary search or enumeration, but pay $O ( \log n )$ extra rollouts per sample or $N \times$ the rollout budget. Moreover, these methods have been developed almost exclusively on mathematical reasoning, where tasks share uniform structure; agentic tasks present a different challenge, as difficulty varies widely within a single batch (e.g., a two-step "Pick" vs. a twenty-step "Pick Two" in ALFWorld) (Tu et al., 2026; Lian et al., 2026b; Li et al., 2026; Chen et al., 2026).

Our diagnostic in Section 2 quantifies both failure modes: shared-depth schedulers place over 60% of assignments outside the informative band (Figure 2(a)), while per-sample probing reduces this mismatch only by spending extra rollouts that remain too noisy under a GRPO-matched budget (Figure 2(b)). Both families share a deeper assumption: that one optimal depth exists per task, and the goal is to pinpoint it. Our analysis in Section 2 reveals that this assumption is mismatched to the problem structure: informative depths form a band around the optimal point, with a training-signal profile that is unimodal, approximately symmetric, and well fit by a Gaussian $( \sigma { = } 0 . 2 2 , R ^ { 2 } { = } 0 . 9 2 ; \mathrm { F i g } .$ ure 3(b)). The right guidance depth is not a point to be found, but a neighborhood to be covered.

This observation motivates $\mathrm { \bf A g e n t { - } G ^ { 2 } }$ , a Gaussian guidance framework for hint-based RL on long-horizon agentic tasks. Agent- $\cdot \mathrm { G ^ { 2 } }$ models guidance depth as a per-task Gaussian whose center and spread are estimated online through a global-local decomposition: a global baseline tracks the policy’s overall progress, while per-cluster statistics adjust the center by task difficulty and widen the spread to match within-cluster variance. For each task, one depth is drawn from its cluster’s Gaussian and converted to a prefix length shared by all rollouts. The same rollouts that update the policy also refresh the Gaussian parameters, achieving per-task depth variation at no extra rollout cost.

We evaluate $\mathrm { \bf A g e n t { - } G ^ { 2 } }$ on ALFWorld and Web-Shop with Qwen2.5-1.5B-Instruct and Qwen2.5- 7B-Instruct. On ALFWorld, $\mathrm { \bf A g e n t { - } G ^ { 2 } }$ reaches 95.3% overall success at 1.5B and 98.4% at 7B, improving over the strongest hint-based RL baseline by 1.5 and 2.3 points and over the strongest hint-free RL baseline by 3.9 points at both scales. On WebShop, $\mathrm { \bf A g e n t { - } G ^ { 2 } }$ obtains a 92.3 reward score at both scales, with 78.9% and 84.4% finalpurchase success. These gains hold across benchmarks, model scales, and horizon groups, without the probe-rollout overhead of per-sample search.

![](images/1064b0d80dd90020efc7cc77c5a068b323a706398b25da1542a05bcf09e4b8cb.jpg)

![](images/28c372d9679cdb93082a69bc5994714fc4e5ec3330e80c643515da633cfc45d8.jpg)  
(a) Long-horizon guidance  
(b) Rollout cost  
Figure 2: Scalar-depth schedulers misallocate guidance. (a) Shared-depth schedulers keep fewer than half of assignments in-range; (b) per-sample probing lowers the mismatch ratio $\rho$ only by spending extra rollouts.

In summary, our contributions are as follows:

• We reveal the neighborhood structure of guidance depth: informative depths form a band with approximately Gaussian training-signal profile, explaining the structural failure modes of both shared-depth schedulers and per-sample probing.

• We propose $\mathrm { \bf A g e n t { - } G ^ { 2 } }$ , a Gaussian guidance framework that estimates the per-task depth distribution online from rollout statistics via a global-local decomposition, requiring no probe rollouts or learned depth predictor.

• On ALFWorld and WebShop with Qwen2.5- 1.5B / 7B-Instruct, we show that $\mathrm { \bf A g e n t { - } G ^ { 2 } }$ consistently outperforms the strongest non-probing hint-based, hint-free, and Aux-RL baselines on both benchmarks, at under one-third the rollout cost of per-sample probing.

## 2 Preliminary Analysis

We diagnose guidance-depth assignment on Qwen2.5-1.5B-Instruct / ALFWorld at step 50, where the policy’s no-hint success rate is near 50% and the spread of per-task informative depths is widest. We sweep a depth grid D and estimate $p _ { i } ( d )$ from 32 rollouts per (task, depth) pair. The in-range set of task i is $\mathcal { R } _ { i } = \{ d \in \mathcal { D } : p _ { i } ( d ) \in [ 0 . 4 , 0 . 6 ] \}$ , where training is most informative (Florensa et al., 2018); an assignment $d _ { i }$ is under-guided if $p _ { i } ( d _ { i } ) < 0 . 4$ and over-guided if $p _ { i } ( d _ { i } ) > 0 . 6$ The mismatch ratio $\rho$ is the fraction of tasks with $d _ { i } \notin \mathcal { R } _ { i }$ . Full details appear in Appendix A.

![](images/ce9c9e4d0fde86da24d784bdabbafbec9b3473a21c8ec88f54e653dbf0643c2a.jpg)  
(a) Injection depth k

![](images/7e46c9682f04679b94aba37ffcf93e8004735a4f135953b4dfc75410894f71d4.jpg)  
(b) Relative depth Δd  
Figure 3: Useful guidance forms a band around $d _ { i } ^ { \star }$ (a) Rollout success rate $p _ { i } ( d )$ vs. depth, with under-/inrange/over-guided regimes shaded; the in-range band spans several depths. (b) Bernoulli-variance informativeness aligned by $\Delta d = d - d _ { i } ^ { \star }$ , with a Gaussian fit $( \sigma { = } 0 . 2 2 , R ^ { 2 } { = } 0 . 9 2 )$

Mismatch in Shared-Depth Scheduling Shareddepth schedulers set one guidance depth from the training step or batch-level feedback and apply it to all tasks in a batch. While avoiding additional probe rollouts, this scalar assignment ignores tasklevel heterogeneity. Figure 2(a) shows that among the shared-depth schedulers we test, the three stepbased schedulers place only 15%–23% of rollout groups inside the in-range band, with most assignments being over-guided. Even the best shared scheduler, Target-acc, leaves 38% of its assignments outside the in-range band. This failure is structural rather than a tuning issue: a single depth cannot match tasks with different in-range bands.

Cost-precision Trade-off of Per-sample Probing Per-sample probing samples rollouts at candidate depths and selects the depth whose estimated success rate $\hat { p } _ { i }$ is closest to 0.5, removing the shareddepth assumption at the cost of additional probe rollouts. Figure 2(b) quantifies this cost-precision trade-off. With $M _ { \mathrm { p r o b e } } { = } 2$ rollouts per candidate depth, Binary Search still leaves 75% of assignments mismatched. Enumeration is more accurate but costlier: $M _ { \mathrm { p r o b e } } { = } 4$ reaches 47% mismatch at $2 \times$ the GRPO rollout budget, while near-zero mismatch requires $M _ { \mathrm { p r o b e } } { = } 3 2$ at 20× the budget. Thus, methods that try to pinpoint a single depth from noisy rollout estimates must trade additional rollout cost for selection error.

Modeling Depth as a Gaussian. The two failure modes above stem from a shared premise: that one optimal depth exists per task. We now show that useful guidance has a richer structure.

Observation 1: Useful guidance occupies a band. Figure 3(a) plots $p _ { i } ( d )$ against $d ,$ with the underguided, in-range, and over-guided regimes shaded. The in-range regime spans multiple neighboring depths rather than a single value, indicating that several nearby depths can keep rollouts in the justlearnable regime. Thus, a scheduler that localizes one exact depth ignores a neighborhood of informative choices.

Observation 2: Informative signal is Gaussian-like. We define $d _ { i } ^ { \star } = \arg \operatorname* { m i n } _ { { d \in { \mathcal { D } } } } | p _ { i } ( d ) - 0 . 5 |$ and align task-depth pairs by $\Delta d = d - d _ { i } ^ { \star }$ . As a proxy for training informativeness we use the Bernoulli variance $p _ { i } ( d ) ( 1 - p _ { i } ( d ) )$ , which peaks at $p _ { i } { = } 0 . 5$ and drops to 0 at deterministic rollouts. The peak at $\Delta d = 0$ follows from the definition of $d _ { i } ^ { \star }$ , but the symmetry and decay along $\Delta d$ do not: they reflect how $p _ { i } ( d )$ varies with depth. The aligned profile (Figure 3(b)) is unimodal and approximately symmetric, with a Gaussian fit yielding $\sigma { = } 0 . 2 2$ and $R ^ { 2 } { = } 0 . 9 2$ . The Gaussian provides the two parameters our sampler needs: a center $\mu$ for the band location and a spread $\sigma$ for its width. Both map onto the mean and variance of batch rollouts, so the sampler in Section 3 runs online without extra probing. We discuss the choice of parametric family in Section A.

## 3 Method: Agent-G<sup>2</sup>

Section 2 showed that scalar-depth schedulers either ignore per-task heterogeneity or require costly per-sample probing, while informative depths form Gaussian-like profiles around task-specific effective depths. $\mathrm { { A g e n t { - } G ^ { 2 } } }$ translates these findings into a practical training framework (Figure 4): an adaptive Gaussian schedule derives per-task distributions from existing rollout statistics (Section 3.2), sampling draws one depth per task (Section 3.3), and the same rollouts update both the policy and the schedule (Section 3.4).

## 3.1 Problem Formulation.

We consider LLM agents trained with reinforcement learning on long-horizon tasks with reward only at termination. Each task i has a naturallanguage instruction $q _ { i }$ and one expert trajectory $\begin{array} { r c l } { \tau _ { i } ^ { \star } } & { = } & { ( a _ { i , 1 } ^ { \star } , \ldots , a _ { i , L _ { i } } ^ { \star } ) } \end{array}$ of length $L _ { i }$ . A hint scheduler selects a guidance ratio $r _ { i } ~ \in ~ [ 0 , 1 ]$ which is converted to a prefix length $n _ { i }$ = min $\left( \lceil r _ { i } L _ { i } \rceil , L _ { i } - 1 \right)$ . The first $n _ { i }$ expert actions are executed in the environment to reach a postprefix state, from which the policy $\pi _ { \theta }$ runs R independent rollouts. Each rollout $j \in \{ 1 , \ldots , R \}$ terminates with a binary reward $y _ { i , j } \in \{ 0 , 1 \}$ , and $\begin{array} { r } { \hat { p } _ { i } = \frac { 1 } { R } \sum _ { j = 1 } ^ { R } y _ { i , j } } \end{array}$ denotes the empirical success rate. We optimize $\pi _ { \theta }$ with GRPO (Shao et al., 2024), which computes a group-normalized advantage from the R terminal rewards within each task.

![](images/44582bc4583bc5d50107be0bafebe89fe4bb12f644404a47030537c90c21827a.jpg)  
Figure 4: $\mathbf { A g e n t { - } G ^ { 2 } }$ pipeline. (1) Tasks are clustered offline by difficulty. (2) A global baseline $\mu _ { \mathrm { g l o b a l } }$ and per-cluster statistics $( A _ { k } , V _ { k } )$ form a per-task Gaussian $\mathcal { N } ( \mu _ { i } , \sigma _ { i } ^ { 2 } )$ . (3) One prefix length is drawn per task; R rollouts run from the post-prefix state. (4) The policy is updated with prefix SFT plus GRPO loss. (5) The same rollouts refresh $\mu _ { \mathrm { g l o b a l } } , A _ { k } , V _ { k }$ for the next batch.

## 3.2 Adaptive Gaussian Schedule

$\mathrm { \bf A g e n t { - } G ^ { 2 } }$ assigns each task i a Gaussian distribution $\mathcal { N } ( \mu _ { i } , \sigma _ { i } ^ { 2 } )$ over the guidance ratio $r _ { i }$ . Its center and spread are computed online from rollouts already collected for policy optimization. The scheduler maintains three types of state: a global guidance baseline $\mu _ { \mathrm { g l o b a l } }$ that tracks the overall guidance level, and for each offline cluster $\mathcal { C } _ { k }$ , EMA estimates $A _ { k }$ and $V _ { k }$ that summarize the level and dispersion of empirical success rates. We use the standard midpoint target $p _ { \mathrm { t a r g e t } } = 0 . 5$ for binary success feedback, following the curriculum principle that training is most informative near the success–failure boundary (Florensa et al., 2018; Zhang et al., 2025a). All schedule states are initialized from weak priors that only determine the first-batch schedule, then continuously refreshed by rollout statistics.

Global baseline. The global baseline $\mu _ { \mathrm { g l o b a l } } \in$ [0, 1] tracks the guidance level required by the current policy. For a training batch B with average success rate acc $\begin{array} { r } { { \bf \ddot { \theta } } \cdot { \bf \sigma } _ { B } = \frac { 1 } { | \boldsymbol { \mathcal { B } } | } \sum _ { i \in B } \hat { p } _ { i } } \end{array}$ , we shift $\mu _ { \mathrm { g l o b a l } }$ by $\Delta$ in the direction that drives acc<sub>B</sub> toward $p _ { \mathrm { t a r g e t } }$

$$
\mu _ { \mathrm { g l o b a l } }  \mathrm { c l i p } \Big ( \mu _ { \mathrm { g l o b a l } } + \mathrm { s i g n } ( p _ { \mathrm { t a r g e t } } - \mathrm { a c c } _ { \mathcal { B } } ) \Delta , 0 , 1 \Big ) .\tag{1}
$$

Low batch success raises the baseline for deeper next-batch prefixes; high success lowers it.

Cluster statistics. The training set is partitioned offline into K clusters $\{ \mathcal { C } _ { k } \} _ { k = 1 } ^ { K }$ by each task’s expert-trajectory length, which serves as a simple proxy for separating tasks of different difficulty levels, and $k ( i )$ denotes the cluster of task i. For each non-empty $B _ { k } = B \cap \mathcal C _ { k }$ , the batch produces

$$
\begin{array} { l } { \displaystyle \bar { p } _ { { \boldsymbol { \mathscr { B } } } _ { k } } = \frac { 1 } { \left| \mathscr { B } _ { k } \right| } \sum _ { i \in \mathscr { B } _ { k } } \hat { p } _ { i } , } \\ { \displaystyle v _ { { \boldsymbol { \mathscr { B } } } _ { k } } = \frac { 1 } { \left| \mathscr { B } _ { k } \right| } \sum _ { i \in \mathscr { B } _ { k } } \left( \hat { p } _ { i } - \bar { p } _ { { \boldsymbol { \mathscr { B } } } _ { k } } \right) ^ { 2 } , } \end{array}\tag{2}
$$

which are folded into $A _ { k }$ and $V _ { k }$ by EMA:

$$
\begin{array} { r } { A _ { k }  ( 1 - \alpha ) A _ { k } + \alpha \bar { p } _ { \mathsf { B } _ { k } } , } \\ { V _ { k }  ( 1 - \alpha ) V _ { k } + \alpha v _ { B _ { k } } . } \end{array}\tag{3}
$$

Clusters absent from the batch are left unchanged. $A _ { k }$ estimates the current success level of cluster k, while $V _ { k }$ measures how unevenly tasks in the cluster respond to the current policy.

Per-task distribution. The center and spread for task i in cluster $k { = } k ( i )$ are

$$
\begin{array} { r l } & { \mu _ { i } = \mathrm { c l i p } ( \mu _ { \mathrm { g l o b a l } } + \lambda ( p _ { \mathrm { t a r g e t } } - A _ { k } ) , 0 , 1 ) , } \\ & { \sigma _ { i } = \operatorname* { m a x } ( \gamma V _ { k } , \sigma _ { \mathrm { m i n } } ) . } \end{array}\tag{4}
$$

A lower cluster success rate $A _ { k }$ shifts $\mu _ { i }$ toward deeper guidance; a higher within-cluster variance $V _ { k }$ widens $\sigma _ { i } .$ , giving the sampler broader coverage of the informative band identified in Section 2.

## 3.3 Per-Task Sampling and Rollout

Given $( \mu _ { i } , \sigma _ { i } )$ from the schedule, we draw one guidance ratio per task:

$$
z _ { i } \sim \mathcal { N } ( \mu _ { i } , \sigma _ { i } ^ { 2 } ) , \qquad r _ { i } = \mathrm { c l i p } ( z _ { i } , 0 , 1 ) .\tag{5}
$$

The sampled ratio determines the prefix length $n _ { i } ,$ executed once to reach the post-prefix state shared by all R rollouts of task i (Section 3.1). Since $( \mu _ { i } , \sigma _ { i } )$ are induced by the statistics of cluster $k ( i )$ tasks in the same cluster share Gaussian parameters, while independent sampling yields task-level depth variation without extra rollouts.

## 3.4 Training

Each batch B closes the loop between policy optimization and schedule adaptation: rollouts generated from sampled prefixes drive both updates.

Policy update. We combine the GRPO loss $\mathcal { L } _ { \mathrm { G R P O } } ( B )$ with a teacher-forced loss on sampled expert prefixes, following common practice in hintbased RL (Zhang et al., 2025b; Huang et al., 2026):

$$
\mathcal { L } _ { \mathrm { a u x } } ( \mathcal { B } ) = - \sum _ { i \in \mathcal { B } } \sum _ { t = 1 } ^ { n _ { i } } \log \pi _ { \theta } ( a _ { i , t } ^ { \star } \mid s _ { i , t } , q _ { i } ) ,\tag{6}
$$

where $s _ { i , t }$ is the state reached after the first $t - 1$ expert actions. The full objective is

$$
\begin{array} { r } { \mathcal { L } ( \mathcal { B } ) = \mathcal { L } _ { \mathrm { G R P O } } ( \mathcal { B } ) + \eta \mathcal { L } _ { \mathrm { a u x } } ( \mathcal { B } ) , } \end{array}\tag{7}
$$

with η controlling the auxiliary weight. The auxiliary term preserves imitation supervision on the sampled prefixes and is ablated in Section 4.4.

Schedule update. The terminal rewards from the same rollouts refresh $\mu _ { \mathrm { g l o b a l } } , A _ { k } , V _ { k }$ for the next batch as specified in Section 3.2. Schedule adaptation therefore reuses rollouts already collected for policy optimization and introduces no probe rollouts or learned depth predictor. Algorithm 1 summarizes the full per-batch loop.

## 4 Experiments

## 4.1 Setup

Benchmarks. We evaluate $\mathrm { \bf A g e n t { - } G ^ { 2 } }$ on two long-horizon agent benchmarks. ALFWorld (Shridhar et al., 2021) is a text-based embodied environment where the agent completes household tasks through multi-step interaction with sparse terminal rewards. We follow the standard six task types and group them by horizon into Short (Pick, Look),

Algorithm 1 Agent- $G ^ { 2 }$ per-batch training loop.   
Require: Policy π<sub>θ</sub>, expert trajectories $\{ \tau _ { i } ^ { \star } \}$ , clusters   
$\{ \mathcal { C } _ { k } \} _ { k = 1 } ^ { K }$   
Require: Hyperparameters $\begin{array} { r l } { \Delta , \alpha , \lambda , \gamma , \sigma _ { \mathrm { m i n } } , \eta , R ; p _ { \mathrm { t a r g e t } } = } \end{array}$   
0.5   
1: Initialize $\mu _ { \mathrm { g l o b a l } }  0 . 8 , A _ { k }  0 , V _ { k }  0 \mid$ ▷ weak priors   
2: for each batch B do   
▷ Per-task schedule, sampling, and prefix execution   
3: for each task $i \in \mathcal { B }$ with $k \overset { = } { = } k ( i )$ do   
4: $\mu _ { i }  \mathrm { c l i p } ( \mu _ { \mathrm { g l o b a l } } + \lambda ( p _ { \mathrm { t a r g e t } } - A _ { k } ) , 0 , 1 )$   
5: $\sigma _ { i } \gets \operatorname* { m a x } ( \gamma V _ { k } , \sigma _ { \operatorname* { m i n } } )$   
6: $z _ { i } \sim \mathcal { N } ( \mu _ { i } , \sigma _ { i } ^ { 2 } ) ; r _ { i } \gets \mathrm { c l i p } ( z _ { i } , 0 , 1 )$   
7: $n _ { i } \gets \operatorname* { m i n } ( \lceil r _ { i } L _ { i } \rceil , L _ { i } - 1 )$   
8: Execute $\tau _ { i , 1 : n _ { i } } ^ { \star }$ to reach state $s _ { i }$   
9: Record $\tau _ { i , 1 : n _ { i } } ^ { \star }$ for L<sub>aux</sub>   
10: for $j = 1 , \dotsc , R$ do   
11: Roll out π from $s _ { i } ;$ record $y _ { i , j }$   
12: end for   
13: end for   
▷ Policy update   
14: Update π<sub>θ</sub>: minimize $\mathcal { L } _ { \mathrm { G R P O } } ( \boldsymbol { B } ) + \eta \mathcal { L } _ { \mathrm { a u x } } ( \boldsymbol { B } )$   
▷ Schedulefeedback   
15: $\begin{array} { r } { \hat { p } _ { i }  \frac { 1 } { R } \sum _ { j } y _ { i , j } } \end{array}$ for $i \in \mathcal { B }$   
16: acc<sub>B</sub> $\textstyle \gets \frac { 1 } { | B | } \sum _ { i \in B } \hat { p } _ { i }$   
17: Refresh A<sub>k</sub>, V<sub>k</sub>, µ<sub>global</sub> (Section 3.2)   
18: end for   
19: return π<sub>θ</sub>

Medium (Clean, Heat, Cool), and Long (Pick2). WebShop (Yao et al., 2023a) is a web navigation environment where the agent searches, filters, and purchases products matching a natural-language specification, with reward issued at final purchase.

We compare Agent- $\mathbf { \delta G ^ { 2 } }$ against five families. (1) Prompting: ReAct (Yao et al., 2023b) and Reflexion (Shinn et al., 2023) on the same base, with GPT-4o (OpenAI et al., 2024) and Gemini-2.5-Pro (Comanici et al., 2025) as closed-source references. (2) Imitation: Full SFT, supervised fine-tuning on entire expert trajectories. (3) RL without hints: GRPO (Shao et al., 2024), GiGPO (Feng et al., 2025), and BEACON (Wang et al., 2026), trained with sparse terminal rewards and no expert prefix. (4) Aux-RL: ETO (Song et al., 2024) and RLVMR (Zhang et al., 2025d), which add auxiliary supervision beyond the terminal reward. (5) Hint-based RL, subdivided into: schedule-based methods (Linear, Cosine, Step decay, Target acc); search-based methods (Binary Search, Enumeration); and end-to-end methods (StepHint (Zhang et al., 2025a), TraPO (Su et al., 2025)). All schedule- and search-based baselines share Agent-$\mathrm { G ^ { 2 } \vec { s } }$ GRPO backbone, isolating prefix-depth allocation as the only variable.

Implementation. We instantiate $\mathrm { \bf A g e n t { - } G ^ { 2 } }$ on Qwen2.5-1.5B / 7B-Instruct (Qwen et al.,

Table 1: Main results on ALFWorld and WebShop (success rate, %). Best and second-best are highlighted.
<table><tr><td rowspan="2">Type Method</td><td rowspan="2"></td><td colspan="7">ALFWorld</td><td rowspan="2">WebShop</td></tr><tr><td>Short</td><td></td><td></td><td>Medium</td><td></td><td>Long</td><td>All Score</td></tr><tr><td></td><td></td><td>Pick</td><td>Look</td><td>Clean</td><td>Heat</td><td>Cool</td><td>Pick2</td><td></td><td></td></tr><tr><td>Closed-Source Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Prompting Prompting</td><td>GPT-4o (OpenAI et al., 2024) Gemini-2.5-Pro (Comanici et al., 2025)</td><td>75.3 92.8</td><td>60.8 63.3</td><td>31.2 62.1</td><td>56.7 69.0</td><td>21.6 26.6</td><td>49.8 58.7</td><td>48.0 31.8 60.3 42.5</td><td>23.7</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>35.9</td></tr><tr><td>Base: Qwen2.5-1.5B-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Prompting</td><td>Direct Prompt (Qwen et al., 2025) ReAct (Yao et al., 2023b)</td><td>5.9</td><td>5.5 20.5</td><td>3.3 15.7</td><td>9.7</td><td>4.2</td><td>0.0 2.0</td><td>4.1 23.1 12.8</td><td>5.2</td></tr><tr><td>Prompting</td><td>Reflexion (Shinn et al., 2023)</td><td>17.4 35.3</td><td>22.2</td><td>21.7</td><td>6.2 13.6</td><td>7.7</td><td>3.7 21.8</td><td>40.1 55.8</td><td>11.3</td></tr><tr><td>Prompting</td><td></td><td></td><td>70.6</td><td></td><td>71.4</td><td>19.4</td><td>28.0</td><td></td><td>21.9</td></tr><tr><td>SFT</td><td>Full SFT</td><td>68.8</td><td>53.7</td><td>61.9 84.5</td><td></td><td>42.9</td><td>56.3</td><td>86.6</td><td>69.5</td></tr><tr><td>RL</td><td>GRPO (Shao et al., 2024) GiGPO (Feng et al., 2025)</td><td>85.3</td><td>76.5</td><td>91.8</td><td>78.2 91.3</td><td>59.7</td><td>53.5 72.8 79.5</td><td>75.8</td><td>56.8</td></tr><tr><td>RL</td><td>BEACON (Wang et al., 2026)</td><td>96.0</td><td>88.2</td><td>86.7</td><td>100</td><td>71.7</td><td>86.1</td><td>83.1</td><td>65.0</td></tr><tr><td>RL</td><td></td><td>100 73.6</td><td>46.3</td><td>66.2</td><td>68.3</td><td>78.9 62.8</td><td>92.9 91.4 55.6 66.4</td><td>86.1</td><td>75.6</td></tr><tr><td>Aux-RL</td><td>ETO (Song et al., 2024)</td><td>95.2</td><td>78.8</td><td>91.2</td><td>90.2</td><td>83.9</td><td>77.6 87.9</td><td></td><td></td></tr><tr><td>Aux-RL</td><td>RLVMR (Zhang et al., 2025d)</td><td>92.5</td><td>83.3</td><td>90.0</td><td>66.6</td><td></td><td>82.6 83.6</td><td></td><td></td></tr><tr><td>Hint-RL Hint-RL</td><td>Linear decay</td><td>97.6</td><td>70.0</td><td>83.3</td><td>58.3</td><td>71.4 85.0</td><td>76.2 83.6</td><td>88.9 83.2</td><td>74.2</td></tr><tr><td>Hint-RL</td><td>Cosine decay Step decay</td><td>85.0</td><td>100</td><td>95.0</td><td>83.3</td><td></td><td>86.9</td><td>89.1</td><td>73.4</td></tr><tr><td>Hint-RL</td><td>Target acc</td><td></td><td>83.3</td><td></td><td></td><td>95.2</td><td>89.8</td><td></td><td>74.2</td></tr><tr><td>Hint-RL</td><td>Enumeration</td><td>100</td><td>81.8</td><td>100 100</td><td>100</td><td>90.5</td><td>82.6 93.8</td><td>86.4</td><td>75.0</td></tr><tr><td>Hint-RL</td><td></td><td>100</td><td></td><td></td><td>68.8</td><td>80.0</td><td>61.1 86.0</td><td>90.1</td><td>78.1</td></tr><tr><td>Hint-RL</td><td>Binary Search (Zhang et al., 2025b)</td><td>83.3</td><td>75.0</td><td>90.9</td><td>78.6</td><td>95.7</td><td>81.5 85.2</td><td>89.7</td><td>77.3</td></tr><tr><td>Hint-RL</td><td>StepHint (Zhang et al., 2025a)</td><td>86.0</td><td>60.0</td><td>84.9</td><td>63.7</td><td>75.0</td><td>65.2 77.3</td><td>87.6</td><td>71.9</td></tr><tr><td>Hint-RL</td><td>TRAPO (Su et al., 2025) Agent-G² (Ours)</td><td>79.2</td><td>50.0</td><td>85.2</td><td>56.3</td><td>54.8</td><td>18.2 59.4</td><td>89.4</td><td>75.8</td></tr><tr><td></td><td></td><td>96.8</td><td>100</td><td>100</td><td>92.9</td><td>84.2</td><td>94.7</td><td>95.3</td><td>92.3 78.9</td></tr><tr><td>Base: Qwen2.5-7B-Instruct</td><td>ReAct (Yao et al., 2023b)</td><td>48.5</td><td>35.4</td><td>34.3</td><td>13.2</td><td>18.2</td><td>17.6</td><td>31.2</td><td></td></tr><tr><td>Prompting SFT</td><td>Full SFT</td><td>75.0</td><td>100</td><td>80.0</td><td>43.8</td><td>59.1</td><td>37.8 64.9</td><td>46.2 88.3</td><td>19.5</td></tr><tr><td>RL</td><td>GRPO (Shao et al., 2024)</td><td>90.8</td><td>66.1</td><td>89.3</td><td>74.7</td><td>72.5</td><td>64.7 77.6</td><td>79.3</td><td>79.7</td></tr><tr><td>RL</td><td>GiGPO (Feng et al., 2025)</td><td>91.8</td><td>88.6</td><td>95.9</td><td>90.2</td><td>86.5</td><td>85.2 90.2</td><td>84.4</td><td>66.1 72.8</td></tr><tr><td>RL</td><td>BEACON (Wang et al., 2026)</td><td>100</td><td>81.8</td><td>96.3</td><td>92.9</td><td>94.7</td><td>90.0 94.5</td><td>87.7</td><td>79.7</td></tr><tr><td>Aux-RL</td><td>ETO (Song et al., 2024)</td><td>88.2</td><td>70.5</td><td>82.3</td><td>83.6</td><td>71.0</td><td>51.2 74.2</td><td></td><td></td></tr><tr><td>Aux-RL</td><td>RLVMR (Žhang et al., 2025d)</td><td>95.3</td><td>88.2</td><td>90.1</td><td>92.4</td><td>89.8</td><td>86.7 91.8</td><td></td><td></td></tr><tr><td>Hint-RL</td><td>Linear decay</td><td>97.5</td><td>92.9</td><td>66.7</td><td>85.2</td><td>87.5</td><td>92.0 85.2</td><td>91.1</td><td>82.8</td></tr><tr><td>Hint-RL</td><td>Cosine decay</td><td>97.1</td><td>100</td><td>78.9</td><td>85.7</td><td>83.3</td><td>90.0 89.4</td><td>90.1</td><td>82.0</td></tr><tr><td>Hint-RL</td><td>Step decay</td><td>95.0</td><td>92.9</td><td>90.5</td><td>91.7</td><td>100</td><td>80.0 91.4</td><td>92.1</td><td>83.6</td></tr><tr><td>Hint-RL</td><td>Target acc</td><td>97.5</td><td>100</td><td>95.2</td><td>83.3</td><td>81.3</td><td>92.0 92.9</td><td>89.3</td><td>79.7</td></tr><tr><td>Hint-RL</td><td>Enumeration</td><td>100</td><td>100</td><td>96.4</td><td>100</td><td>90.5</td><td>86.7 96.1</td><td>96.0</td><td>89.8</td></tr><tr><td>Hint-RL</td><td>Binary Search (Zhang et al., 2025b)</td><td>97.6</td><td>100</td><td>96.3</td><td>73.7</td><td>84.6</td><td>93.8 92.2</td><td>91.2</td><td>83.6</td></tr><tr><td>Hint-RL</td><td>StepHint (Zhang et al., 2025a)</td><td>97.3</td><td>100</td><td>89.4</td><td>94.1</td><td>79.1</td><td>91.7 91.4</td><td>90.9</td><td>82.0</td></tr><tr><td>Hint-RL</td><td>TRAPO (Su et al., 2025)</td><td>89.3</td><td>66.7</td><td>88.5</td><td>90.0</td><td>68.8</td><td>44.8 75.0</td><td>90.6</td><td>77.3</td></tr><tr><td>Hint-RL</td><td> $\mathbf { A g e n t { - } G ^ { 2 } }$  (Ours)</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>91.7 98.4</td><td>92.3</td><td>84.4</td></tr><tr><td></td><td></td><td></td><td></td><td></td></table>

2025). Agent- $\mathbf { \cdot G ^ { 2 } }$ -specific hyperparameters $( \Delta { = } 0 . 1 , ~ \alpha { = } 0 . 2 , ~ \gamma { = } 1 . 0 )$ are fixed across both benchmarks without per-task tuning. We report success rate (%) on the held-out test set averaged over 5 seeds; full configurations appear in Appendix B.

## 4.2 Main Results

Overall performance. On ALFWorld, $\mathrm { { A g e n t { - } G ^ { 2 } } }$ reaches 95.3% (1.5B) and 98.4% (7B). It outperforms RLVMR, the strongest method using auxiliary supervision, by $+ 7 . 4 / + 6 . 6$ , and Enumeration, the strongest probe-based baseline, by $+ 9 . 3 / + 2 . 3 $ without any auxiliary network or probe rollouts. On WebShop, $\mathrm { \bf A g e n t { - } G ^ { 2 } }$ is the strongest non-probing method at both scales (92.3 reward score, 78.9% /

84.4% purchase success), and at 1.5B also outperforms Enumeration by +2.2 reward-score points.

Cross-scale robustness. $\mathrm { \bf A g e n t { - } G ^ { 2 } }$ remains effective as the backbone scales from 1.5B to 7B. On ALFWorld, the margin over the strongest hintbased baseline grows from +1.5 to +2.3, so finer scheduling does not become redundant on stronger backbones. On WebShop, purchase success climbs from 78.9% to 84.4%, consistent with stronger action precision at scale. The 1.5B model also surpasses all 7B non-probing baselines, indicating that schedule design substitutes for backbone scaling rather than only compensating for limited capacity.

![](images/b19b2a58f6505c8d7ecca1b5c565869cf82289d17437873c736c39429334a97b.jpg)  
(a) Beyond Imitation

![](images/139d5926978c4488ef94c2cb7a15075dfddd032b88834aedf3fd031b814b11fa.jpg)  
(b) Schedule at t = 29

Figure 5: Beyond imitation. (a) ALFWorld success on Qwen2.5-1.5B: Agent- $\mathbf { \delta G ^ { 2 } }$ (95%) far exceeds Full SFT (56%) and Sampled-Prefix SFT (27%). (b) Per-cluster (Short / Medium / Long) depth densities at t=29.  
![](images/72e032d26cfd26408aa80bf8f734951b6d62e9a4bc7c564aeb004e69bde4b48d.jpg)  
(a) Val Success Rate

![](images/2f7b65df1e9f68bb9c36180e325ea9754c5079253777916faedc417fe6895874.jpg)  
(b) Schedule $\mathcal { N } ( \mu _ { t } , \sigma _ { t } ^ { 2 } )$  
Figure 6: Training dynamics of $\mathbf { A g e n t - G } ^ { 2 } .$ . (a) Success rate vs. gradient steps for $\mathbf { A g e n t - G ^ { 2 } }$ and the strongest scalar-depth baseline. (b) Evolution of the per-task guidance-depth distribution across training.

## 4.3 Analysis

Beyond imitation. The gain of $\mathrm { { A g e n t { - } G ^ { 2 } } }$ cannot be explained by imitation alone. Full SFT reaches 56.3%, and Sampled-Prefix SFT, which trains only on the prefix supervision pairs collected during Agent- $\mathbf { \delta G ^ { 2 } }$ training, reaches 26.6%; both are far below $\mathrm { A g e n t - G ^ { 2 } } { \mathrm { : } } 9 5 . 3$ % (Figure 5(a)). The 68.4- point gap between Sampled-Prefix SFT and Agent-$\mathrm { G ^ { 2 } }$ isolates the contribution of post-prefix RL: the missing ingredient is not copying expert tokens, but learning from sampled post-prefix states.

Heterogeneous depths in the same batch. $\mathrm { \bf A g e n t { - } G ^ { 2 } }$ gains most when samples in the same batch require different prefix depths. In Figure 6(a), the convergence gap is largest between steps 50 and 150, the window in which the depth distribution in Figure 6(b) is widest. A scalar $d ^ { \mathrm { s c h e d } }$ is brittle here: a single depth is too deep for easy samples and too shallow for hard ones, pushing most rollouts outside the informative band. $\mathrm { \bf A g e n t { - } G ^ { 2 } }$ samples a neighborhood of depths instead, so both ends of the difficulty spectrum receive usable guidance.

Schedule emerges from the rollouts. The shape of the distribution in Figure 6(b) is set by training itself. At t=5 it favors deep prefixes, because the policy has yet to reach learnable states on its own. Around $t { = } 2 5 { - } 5 0$ it widens, as the cluster variance $V _ { k }$ peaks while easy and hard clusters separate. By $t { = } 2 0 0$ it concentrates near zero, because the success rate has cleared the threshold of Equation (1) on every cluster. The mean follows global and cluster success, the spread follows $V _ { k } ,$ and both come from rollouts already collected for policy optimization. Figure 7 in Appendix B samples this pattern across four training phases, with the Long cluster retaining a wider $\sigma _ { k }$ for longer.

Table 2: Per-step training cost on Qwen2.5-1.5B / ALFWorld. Median wall-clock seconds per gradient step. Ratios are relative to Agent- $\mathbf { \delta G ^ { 2 } }$
<table><tr><td>Method</td><td>Type</td><td>Cost/step (s)</td><td>Ratio</td></tr><tr><td> $\mathbf { A g e n t { - } G ^ { 2 } }$ </td><td>Distribution</td><td>88</td><td>1.00×</td></tr><tr><td>Step decay</td><td>Schedule</td><td>57</td><td>0.65×</td></tr><tr><td>Cosine decay</td><td>Schedule</td><td>60</td><td>0.68×</td></tr><tr><td>Linear decay</td><td>Schedule</td><td>61</td><td>0.69×</td></tr><tr><td>Target acc</td><td>Schedule</td><td>80</td><td>0.91×</td></tr><tr><td>Enumeration</td><td>Probe</td><td>425</td><td>4.83×</td></tr><tr><td>Binary Search</td><td>Probe</td><td>285</td><td>3.24×</td></tr></table>

Training efficiency. We assess efficiency by convergence speed and per-step wall-clock cost. Figure 6(a) shows that $\mathrm { \bf A g e n t { - } G ^ { 2 } }$ reaches the final accuracy of scheduled baselines in roughly half the gradient steps. Although $\mathrm { \bf A g e n t { - } G ^ { 2 } }$ incurs a modest per-step overhead over scheduled methods (88s vs. 57–80s), it is substantially cheaper than probing-based methods (285–425s, or 3.24–4.83× higher), which spend extra rollouts to estimate depths. Therefore, by combining faster convergence with much lower cost than probing, Agent-$\mathrm { G ^ { 2 } }$ achieves the best overall training efficiency.

Decomposing the learned schedule. Figure 5(b) decomposes the learned schedule at t=29 by horizon group. These patterns are not manually specified: each cluster only maintains rollout-derived statistics $A _ { k }$ and $V _ { k }$ , and the separation emerges from training feedback.

## 4.4 Ablation Studies

Sampling drives the gain. Replacing the Gaussian draw $d _ { i } \sim \mathcal N ( \mu _ { i } , \sigma _ { i } ^ { 2 } )$ with the deterministic mean $d _ { i } = \mu _ { i }$ removes stochastic depth coverage and drops the overall success rate from 95.3% to 89.8%. This shows that a single-depth estimate cannot cover the informative band. Replacing the Gaussian with a variance-matched uniform distribution reduces performance to 88.3%, confirming that the gain comes from coverage of the informative band, not the exact distributional shape.

Table 3: Ablation of $\mathbf { A g e n t { - } G ^ { 2 } }$ on ALFWorld / Qwen2.5-1.5B-Instruct. $\Delta$ is the change in All relative to the full method. Removing $\mathcal { L } _ { \mathrm { G R P O } }$ is equivalent to Sampled-Prefix SFT.
<table><tr><td>Variant</td><td>Long</td><td>All</td><td> $\pmb { \Delta }$ </td></tr><tr><td> $\mathbf { A g e n t - G ^ { 2 } }$ </td><td>94.7</td><td>95.3</td><td></td></tr><tr><td>Sampling form</td><td></td><td></td><td></td></tr><tr><td>w/o sampling:  $d _ { i , j } { = } \mu _ { i } , \sigma _ { i } { = } 0$   $\mathcal { U } [ \mu _ { i } \pm \sigma _ { i } ]$ </td><td>84.6</td><td>89.8</td><td>↓5.5</td></tr><tr><td>Uniform sampling:</td><td>80.8</td><td>88.3</td><td>↓7.0</td></tr><tr><td>Adaptive moments</td><td></td><td></td><td></td></tr><tr><td>w/o adaptive center:  $\mu _ { i } { = } \mu _ { \mathrm { g l o b a l } }$ </td><td>79.2</td><td>91.4</td><td>↓3.9</td></tr><tr><td>w/o adaptive spread:  $\sigma _ { i } { = } \sigma _ { \operatorname* { m i n } }$ </td><td>78.3</td><td>93.8</td><td>↓1.5</td></tr><tr><td>w/o grouping:  $\scriptstyle \sum _ { k = 1 }$ </td><td>65.4</td><td>89.1</td><td>↓6.2</td></tr><tr><td>Loss components</td><td></td><td></td><td></td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { a u x } }$ </td><td>76.7</td><td>86.7</td><td> $\downarrow ~ 8 . 6$ </td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { G R P O } }$ </td><td>10.0</td><td>26.6</td><td> $\downarrow ~ 6 8 . 7$ </td></tr></table>

Both center and spread matter. Removing the cluster-dependent center correction by setting $\mu _ { i } =$ $\mu _ { \mathrm { g l o b a l } }$ lowers the overall success rate to 91.4%, showing that a global baseline alone cannot capture difficulty variation across clusters. Removing the adaptive spread by setting $\sigma _ { i } = \sigma _ { \operatorname* { m i n } }$ reduces performance to 93.8%, with a pronounced drop on Long tasks $( 9 4 . 7 \%  7 8 . 3 \% )$ , indicating that dynamically adapting $\sigma _ { i }$ is important for maintaining sufficient depth coverage, especially on hard long-horizon tasks. Collapsing all tasks into a single cluster $( K = 1 )$ drops the overall success rate to 89.1%, with the largest degradation on Long tasks, decreasing by 29.3 points. Together, these results show that both cluster-aware centering and adaptive spreading are necessary for handling heterogeneous task difficulty.

Auxiliary loss is supportive. Removing $\mathcal { L } _ { \mathrm { a u x } }$ from Equation (7) lowers the overall success rate by 8.6 points, showing that prefix supervision helps stabilize training. In contrast, removing $\mathcal { L } _ { \mathrm { G R P O } }$ reduces the method to Sampled-Prefix SFT and drops performance to 26.6%, a 68.7-point decrease. This indicates that imitating sampled prefixes alone is insufficient; the main gain comes from GRPO learning on sampled post-prefix states, while $\mathcal { L } _ { \mathrm { a u x } }$ acts as a stabilizer for prefix-token imitation.

## 5 Related Work

Hint-Guidance Strategies. Hint-based RL injects an expert trajectory prefix into rollouts to address reward sparsity, and methods differ primarily in how they choose the prefix depth d. Schedulebased methods set d from the training step or batch success rate and share one value across samples (Xi et al., 2024a; Huang et al., 2026; Guo et al., 2025; Lu et al., 2026b). Per-sample methods make d instance-specific by probing each sample with extra rollouts (Zhang et al., 2025a,b; Wang et al., 2025b; Zhang et al., 2026b; Su et al., 2025; Li et al., 2025). In both lines, d is a deterministic scalar that ignores per-task heterogeneity within a training context, and existing methods are evaluated almost exclusively on math reasoning. $\mathbf { A g e n t - G } ^ { 2 }$ draws a pertask $d _ { i }$ from a cluster-level Gaussian whose mean and variance are estimated from GRPO’s rollout statistics, removing both the shared-scalar assumption and the probing overhead, and we evaluate it on long-horizon agentic benchmarks.

Auxiliary Supervision for Agentic RL. In agentic settings, prior work typically converts expert trajectories into supervision for an auxiliary model, rather than using them to seed rollouts directly. These auxiliary signals take five forms: SFT targets for behavior cloning (Zeng et al., 2023; Xi et al., 2024b; Qi et al., 2025; Bai et al., 2024), preference pairs for trajectory-level DPO (Song et al., 2024; Putta et al., 2024; Lai et al., 2024; Yuan et al., 2025; Lian et al., 2026a), value heads or Q-critics distilled from demonstrations (Xiang et al., 2024; Zhou et al., 2024b; Feng et al., 2024; Gu et al., 2025), step-level process reward models (Choudhury, 2025; Wang et al., 2025a; Xiong et al., 2024; Lu et al., 2026a), and milestone-shaped rewards extracted from successful traces (Wang et al., 2026; Zheng et al., 2026). Each route adds an auxiliary network and inherits its annotation, Monte-Carlo labeling, or reward-model fragility cost (Zhang et al., 2025c; Pan et al., 2026). $\mathbf { A g e n t - G } ^ { 2 }$ uses the same expert trajectories directly as rollout-starting prefixes inside an on-policy GRPO loop, training no auxiliary model and requiring no further offline annotation.

## 6 Conclusion

We proposed $\mathrm { \bf A g e n t { - } G ^ { 2 } }$ , a Gaussian guidance framework for hint-based reinforcement learning on long-horizon agentic tasks. Rather than treating guidance depth as a deterministic scalar, Agent-$\mathrm { G ^ { 2 } }$ draws it per task from a Gaussian whose center and spread are estimated online from rollouts collected for policy optimization, without learned depth predictor or extra probe rollouts. Agent-G² is the strongest on ALFWorld and the strongest non-probing method on WebShop at both 1.5B and 7B scales, with 1.5B Agent-G² already surpassing 7B BEACON, showing that schedule design can substitute for backbone scaling.

## Limitations

Agent-G<sup>2</sup> relies on the availability of one expert trajectory per training task. When such trajectories are unavailable or costly to obtain, the framework cannot be applied directly; extending it to weaker forms of supervision (e.g., suboptimal demonstrations or language hints) is an open direction.

The Gaussian parameterization is motivated by the empirical informativeness profile observed in Section 2, which fits well on our two benchmarks. Tasks with multimodal or heavily skewed depth profiles may benefit from richer distributional families, though our uniform-distribution ablation (Section 4.4) suggests that the exact shape matters less than stochastic coverage of the informative band.

The difficulty clusters are defined offline by expert-trajectory length, which serves as a simple proxy for task horizon and difficulty. Although effective in our experiments, this fixed partitioning does not adapt as the policy improves and taskrelative difficulty changes. Online difficulty estimation could further improve adaptivity.

## Ethics Statement

We adhere to the ACL Code of Ethics and Code of Conduct. Our work uses only publicly available benchmarks and pretrained open-source models. Code and training scripts will be released under the MIT License upon publication. We used Claude for grammatical refinement; all scientific content and conclusions are the authors’ own work.

## Acknowledgements

This work was supported by National Key Research and Development Project (No. 2024YFB3312900), National Natural Science Foundation of China (No. 62506332) and CCF-Baidu Open Fund.

## References

Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Eric Jang,

Rosario Jauregui Ruano, Kyle Jeffrey, and 26 others. 2022. Do as i can, not as i say: Grounding language in robotic affordances. Preprint, arXiv:2204.01691.

Hao Bai, Yifei Zhou, Mert Cemri, Jiayi Pan, Alane Suhr, Sergey Levine, and Aviral Kumar. 2024. Digirl: Training in-the-wild device-control agents with autonomous reinforcement learning. Preprint, arXiv:2406.11896.

Daniil A. Boiko, Robert MacKnight, and Gabe Gomes. 2023. Emergent autonomous scientific research capabilities of large language models. Preprint, arXiv:2304.05332.

Andres M Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D White, and Philippe Schwaller. 2023. Chemcrow: Augmenting large-language models with chemistry tools. Preprint, arXiv:2304.05376.

Tongbo Chen, Zhengxi Lu, Zhan Xu, Guocheng Shao, Shaohan Zhao, Fei Tang, Yong Du, Kaitao Song, Yizhou Liu, Yuchen Yan, Wenqi Zhang, Xu Tan, Weiming Lu, Jun Xiao, Yueting Zhuang, and Yongliang Shen. 2026. Knowu-bench: Towards interactive, proactive, and personalized mobile agent evaluation. Preprint, arXiv:2604.08455.

Sanjiban Choudhury. 2025. Process reward models for llm agents: Practical framework and directions. Preprint, arXiv:2502.10325.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, Luke Marris, Sam Petulla, Colin Gaffney, Asaf Aharoni, Nathan Lintz, Tiago Cardal Pais, Henrik Jacobsson, Idan Szpektor, Nan-Jiang Jiang, and 3416 others. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. Preprint, arXiv:2507.06261.

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. 2023. Mind2web: Towards a generalist agent for the web. Preprint, arXiv:2306.06070.

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. 2025. Group-in-group policy optimization for llm agent training. Preprint, arXiv:2505.10978.

Peiyuan Feng, Yichen He, Guanhua Huang, Yuan Lin, Hanchong Zhang, Yuchen Zhang, and Hang Li. 2024. Agile: A novel reinforcement learning framework of llm agents. Preprint, arXiv:2405.14751.

Carlos Florensa, David Held, Xinyang Geng, and Pieter Abbeel. 2018. Automatic goal generation for reinforcement learning agents. Preprint, arXiv:1705.06366.

Yu Gu, Kai Zhang, Yuting Ning, Boyuan Zheng, Boyu Gou, Tianci Xue, Cheng Chang, Sanjari Srivastava, Yanan Xie, Peng Qi, Huan Sun, and Yu Su. 2025. Is your llm secretly a world model of the internet?

model-based planning for web agents. Preprint, arXiv:2411.06559.

Yongxin Guo, Wenbo Deng, Zhenglin Cheng, and Xiaoying Tang. 2025. G<sup>2</sup>rpo-a: Guided group relative policy optimization with adaptive guidance. Preprint, arXiv:2508.13023.

Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. 2022. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. Preprint, arXiv:2201.07207.

Zeyu Huang, Tianhao Cheng, Zihan Qiu, Zili Wang, Yinghui Xu, Edoardo M. Ponti, and Ivan Titov. 2026. Blending supervised and reinforcement fine-tuning with prefix sampling. Preprint, arXiv:2507.01679.

Hanyu Lai, Xiao Liu, Iat Long Iong, Shuntian Yao, Yuxuan Chen, Pengbo Shen, Hao Yu, Hanchen Zhang, Xiaohan Zhang, Yuxiao Dong, and Jie Tang. 2024. Autowebglm: A large language model-based web navigating agent. Preprint, arXiv:2404.03648.

Dinging Li, Yingxiu Zhao, Xinrui Cheng, Kangheng Lin, Hongbo Peng, Hongxing Li, Zixuan Wang, Yuhong Dai, Haodong Li, Jia Wang, Yukang Shi, Liang Zhao, Jianjian Sun, Zheng Ge, Xiangyu Zhang, Weiming Lu, Jun Xiao, Yueting Zhuang, and Yongliang Shen. 2026. Spatialevo: Self-evolving spatial intelligence via deterministic geometric environments. Preprint, arXiv:2604.14144.

Ziheng Li, Zexu Sun, Jinman Zhao, Erxue Min, Yongcheng Zeng, Hui Wu, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Xu Chen, and Zhi-Hong Deng. 2025. Staying in the sweet spot: Responsive reasoning evolution via capability-adaptive hint scaffolding. Preprint, arXiv:2509.06923.

Niu Lian, Tongbo Chen, Zhehao Yu, Chengzhen Duan, Fazhan Liu, Hui Liu, Pei Fu, Jian Luan, Heng Qu, Shu-Tao Xia, and Jinpeng Wang. 2026a. Ui-mopd: Multi-platform on-policy distillation for unified gui agents. Preprint, arXiv:2607.04425.

Niu Lian, Yuting Wang, Hanshu Yao, Jinpeng Wang, Bin Chen, Yaowei Wang, Min Zhang, and Shu-Tao Xia. 2026b. From verbatim to gist: Distilling pyramidal multimodal memory via semantic information bottleneck for long-horizon video agents. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11601–11617.

Zhengxi Lu, Zhiyuan Yao, Zhuowen Han, Zi-Han Wang, Jinyang Wu, Qi Gu, Xunliang Cai, Weiming Lu, Jun Xiao, Yueting Zhuang, and Yongliang Shen. 2026a. Self-distilled agentic reinforcement learning. Preprint, arXiv:2605.15155.

Zhengxi Lu, Zhiyuan Yao, Jinyang Wu, Chengcheng Han, Qi Gu, Xunliang Cai, Weiming Lu, Jun Xiao, Yueting Zhuang, and Yongliang Shen. 2026b. Skill0: In-context agentic reinforcement learning for skill internalization. Preprint, arXiv:2604.02268.

OpenAI, :, Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, Aleksander M ˛adry, Alex Baker-Whitcomb, Alex Beutel, Alex Borzunov, Alex Carney, Alex Chow, Alex Kirillov, and 401 others. 2024. Gpt-4o system card. Preprint, arXiv:2410.21276.

Teng Pan, Yuchen Yan, Zixuan Wang, Ruiqing Zhang, Guiyang Hou, Wenqi Zhang, Weiming Lu, Jun Xiao, and Yongliang Shen. 2026. CoVerRL: Breaking the consensus trap in label-free reasoning via generatorverifier co-evolution. In Proceedings ofthe 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 29833– 29853, San Diego, California, United States. Association for Computational Linguistics.

Pranav Putta, Edmund Mills, Naman Garg, Sumeet Motwani, Chelsea Finn, Divyansh Garg, and Rafael Rafailov. 2024. Agent q: Advanced reasoning and learning for autonomous ai agents. Preprint, arXiv:2408.07199.

Zehan Qi, Xiao Liu, Iat Long Iong, Hanyu Lai, Xueqiao Sun, Wenyi Zhao, Yu Yang, Xinyue Yang, Jiadai Sun, Shuntian Yao, Tianjie Zhang, Wei Xu, Jie Tang, and Yuxiao Dong. 2025. Webrl: Training llm web agents via self-evolving online curriculum reinforcement learning. Preprint, arXiv:2411.02337.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, and 25 others. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. 2023. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. In Advances in Neural Information Processing Systems, volume 36, pages 38154–38180. Curran Associates, Inc.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2025. Hybridflow: A flexible and efficient rlhf framework. In Proceedings ofthe Twentieth European Conference on Computer Systems, EuroSys ’25, page 1279–1297, New York, NY, USA. Association for Computing Machinery.

Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. Preprint, arXiv:2303.11366.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. 2021. Alfworld: Aligning text and embodied environments for interactive learning. Preprint, arXiv:2010.03768.

Yifan Song, Da Yin, Xiang Yue, Jie Huang, Sujian Li, and Bill Yuchen Lin. 2024. Trial and error: Exploration-based trajectory optimization for LLM agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL).

Mingyu Su, Jian Guan, Yuxian Gu, Minlie Huang, and Hongning Wang. 2025. Trust-region adaptive policy optimization. Preprint, arXiv:2512.17636.

Geng Tu, Dingming Li, Jun Huang, and Ruifeng Xu. 2026. Consensus-driven multi-agent cognitive reasoning for enhancing the emotional intelligence of large language models. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 40, pages 17751–17759.

Hanlin Wang, Chak Tou Leong, Jiashuo Wang, Jian Wang, and Wenjie Li. 2025a. Spa-rl: Reinforcing llm agents via stepwise progress attribution. Preprint, arXiv:2505.20732.

Xinyi Wang, Jinyi Han, Zishang Jiang, Tingyun Li, Jiaqing Liang, Sihang Jiang, Zhaoqian Dai, Shuguang Ma, Fei Yu, and Yanghua Xiao. 2025b. Hint: Helping ineffective rollouts navigate towards effectiveness. Preprint, arXiv:2510.09388.

Zixuan Wang, Dingming Li, Hongxing Li, Shuo Chen, Yuchen Yan, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, and Yueting Zhuang. 2025c. Omniear: Benchmarking agent reasoning in embodied tasks. Preprint, arXiv:2508.05614.

Zixuan Wang, Yuchen Yan, Hongxing Li, Teng Pan, Dingming Li, Ruiqing Zhang, Weiming Lu, Jun Xiao, Yueting Zhuang, and Yongliang Shen. 2026. Milestone-guided policy learning for long-horizon language agents. Preprint, arXiv:2605.06078.

Zhiheng Xi, Wenxiang Chen, Boyang Hong, Senjie Jin, Rui Zheng, Wei He, Yiwen Ding, Shichun Liu, Xin Guo, Junzhe Wang, Honglin Guo, Wei Shen, Xiaoran Fan, Yuhao Zhou, Shihan Dou, Xiao Wang, Xinbo Zhang, Peng Sun, Tao Gui, and 2 others. 2024a. Training large language models for reasoning through reverse curriculum reinforcement learning. Preprint, arXiv:2402.05808.

Zhiheng Xi, Yiwen Ding, Wenxiang Chen, Boyang Hong, Honglin Guo, Junzhe Wang, Dingwen Yang, Chenyang Liao, Xin Guo, Wei He, Songyang Gao, Lu Chen, Rui Zheng, Yicheng Zou, Tao Gui, Qi Zhang, Xipeng Qiu, Xuanjing Huang, Zuxuan Wu, and Yu-Gang Jiang. 2024b. Agentgym: Evolving large language model-based agents across diverse environments. Preprint, arXiv:2406.04151.

Yufei Xiang, Yiqun Shen, Yeqin Zhang, and Cam-Tu Nguyen. 2024. Retrospex: Language agent meets offline reinforcement learning critic. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP).

Weimin Xiong, Yifan Song, Xiutian Zhao, Wenhao Wu, Xun Wang, Ke Wang, Cheng Li, Wei Peng, and Sujian Li. 2024. Watch every step! llm agent learning via iterative step-level process refinement. Preprint, arXiv:2406.11176.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2023a. Webshop: Towards scalable real-world web interaction with grounded language agents. Preprint, arXiv:2207.01206.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023b. React: Synergizing reasoning and acting in language models. Preprint, arXiv:2210.03629.

Siyu Yuan, Zehui Chen, Zhiheng Xi, Junjie Ye, Zhengyin Du, and Jiecao Chen. 2025. Agent-r: Training language model agents to reflect via iterative selftraining. Preprint, arXiv:2501.11425.

Aohan Zeng, Mingdao Liu, Rui Lu, Bowen Wang, Xiao Liu, Yuxiao Dong, and Jie Tang. 2023. Agenttuning: Enabling generalized agent abilities for llms. Preprint, arXiv:2310.12823.

Kaiyi Zhang, Ang Lv, Jinpeng Li, Yongbo Wang, Feng Wang, Haoyuan Hu, and Rui Yan. 2025a. Stephint: Multi-level stepwise hints enhance reinforcement learning to reason. Preprint, arXiv:2507.02841.

Ningyu Zhang, Yunzhi Yao, Jiaxin Qin, Haoming Xu, Yuqi Zhu, Zeping Yu, Mengru Wang, Yuqi Tang, Jia-Chen Gu, Shumin Deng, and Huajun Chen. 2026a. Towards principled knowledge editing methods for large language model reasoning. Nature Machine Intelligence, 8(8):1189–1200.

Xichen Zhang, Sitong Wu, Yinghao Zhu, Haoru Tan, Shaozuo Yu, Ziyi He, and Jiaya Jia. 2026b. Scaf-grpo: Scaffolded group relative policy optimization for enhancing llm reasoning. Preprint, arXiv:2510.19807.

Xuechen Zhang, Zijian Huang, Yingcong Li, Chenshun Ni, Jiasi Chen, and Samet Oymak. 2025b. Bread: Branched rollouts from expert anchors bridge sft & rl for reasoning. Preprint, arXiv:2506.17211.

Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, and Junyang Lin. 2025c. The lessons of developing process reward models in mathematical reasoning. Preprint, arXiv:2501.07301.

Zijing Zhang, Ziyang Chen, Mingxiao Li, Zhaopeng Tu, and Xiaolong Li. 2025d. Rlvmr: Reinforcement learning with verifiable meta-reasoning rewards for robust long-horizon agents. Preprint, arXiv:2507.22844.

Congmin Zheng, Xiaoyun Mo, Xinbei Ma, Qiqiang Lin, Yin Zhao, Jiachen Zhu, Xingyu Lou, Jun Wang, Zhaoxiang Wang, Weiwen Liu, Zhuosheng Zhang, Yong Yu, and Weinan Zhang. 2026. Adaptive milestone reward for gui agents. Preprint, arXiv:2602.11524.

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. 2024a. Webarena: A realistic web environment for building autonomous agents. Preprint, arXiv:2307.13854.

Yifei Zhou, Andrea Zanette, Jiayi Pan, Sergey Levine, and Aviral Kumar. 2024b. Archer: Training language model agents via hierarchical multi-turn rl. Preprint, arXiv:2402.19446.

## A Diagnostic Protocol Details

In-range set and mismatch ratio. Let D denote the guidance-depth grid and T the diagnostic task set. For each $( i , d ) \in \mathcal { T } \times \mathcal { D }$ , we estimate the perdepth success rate by enumeration with $M = 3 2$ independent rollouts:

$$
\hat { p } _ { i } ( d ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } { \bf 1 } [ \mathrm { r o l l o u t } m \mathrm { a t } \mathrm { d e p t h } d \mathrm { s u c c e e d s } ] .\tag{8}
$$

Following Florensa et al. (2018), the in-range set of task i is the subset of depths whose estimated success rate lies in the just-learnable band [0.4, 0.6]:

$$
\mathcal { R } _ { i } = \left\{ d \in \mathcal { D } : \hat { p } _ { i } ( d ) \in [ 0 . 4 , 0 . 6 ] \right\} .\tag{9}
$$

Depths with $\hat { p } _ { i } ( d ) \ < \ 0 . 4$ are under-guided and those with $\hat { p } _ { i } ( d ) > 0 . 6$ are over-guided. Given a scheduler that assigns depth $d _ { i }$ to each task $i ,$ the mismatch ratio is the fraction of tasks placed outside their in-range set:

$$
\rho = \frac { 1 } { | T | } \sum _ { i \in \mathcal { T } } \mathbf { 1 } [ d _ { i } \notin \mathcal { R } _ { i } ] .\tag{10}
$$

$\rho = 0$ corresponds to every task receiving an inrange depth, and $\rho = 1$ corresponds to every assignment being under- or over-guided.

Shared-depth schedulers. A shared scheduler outputs one depth $d _ { t }$ per training step that is applied to every task in the batch, so the per-task assignment is $d _ { i } = d _ { t }$ for all i. The stacked bars in Figure 2(a) report the empirical fraction of assignments in each of the three categories defined above: under-guided $( \hat { p } _ { i } ( d _ { t } ) < 0 . 4 )$ , in-range $( d _ { t } \in \mathcal { R } _ { i } )$ and over-guided $( \hat { p } _ { i } ( d _ { t } ) > 0 . 6 )$

Mismatch across training checkpoints. The diagnostic in Figure 2(a) is computed at step 50. To verify that shared-depth mismatch is not specific to this checkpoint, we repeat the protocol at five additional Qwen2.5-1.5B / ALFWorld checkpoints (Table 4). The Fix-step baseline, which holds $d _ { t }$ constant within 20-step windows, exceeds 48% mismatch at every checkpoint, and the two step-decay schedules (Linear, Step-dec) stay near 50%. Targetacc, the only shared scheduler with batch-level feedback, achieves the lowest mismatch (39.8% on average) but still misses the in-range band on roughly two in five tasks. The structural mismatch of shared-depth scheduling thus persists across the training trajectory and is not a step-50 artifact.

Table 4: Mismatch ratio $\rho$ of shared-depth schedulers across five training checkpoints on Qwen2.5-1.5B / ALFWorld. n is the number of diagnostic tasks at that step.
<table><tr><td>Step</td><td>n</td><td>Fix-step</td><td>Linear</td><td>Step-dec</td><td>Tar-acc</td></tr><tr><td>10</td><td>28</td><td>67.9</td><td>42.9</td><td>42.9</td><td>32.1</td></tr><tr><td>20</td><td>28</td><td>64.3</td><td>39.3</td><td>39.3</td><td>35.7</td></tr><tr><td>40</td><td>28</td><td>75.0</td><td>57.1</td><td>57.1</td><td>50.0</td></tr><tr><td>80</td><td>58</td><td>48.3</td><td>55.2</td><td>55.2</td><td>41.4</td></tr><tr><td>160</td><td>45</td><td>71.1</td><td>55.6</td><td>55.6</td><td>40.0</td></tr><tr><td>Avg</td><td>一</td><td>65.3</td><td>50.0</td><td>50.0</td><td>39.8</td></tr></table>

Probing schedulers. A probing scheduler samples $M _ { \mathrm { p r o b e } }$ rollouts per visited depth on a candidate subset $\mathcal { D } ^ { \prime } \subseteq \mathcal { D }$ , forms the noisy estimate $\tilde { p } _ { i } ^ { ( M _ { \mathrm { p r o b e } } ) } ( d )$ , and outputs

$$
d _ { i } = \arg \operatorname* { m i n } _ { d \in \mathcal { D } ^ { \prime } } \big | \tilde { p } _ { i } ^ { ( M _ { \mathrm { p r o b e } } ) } ( d ) - 0 . 5 \big | .\tag{11}
$$

$\mathcal { D } ^ { \prime }$ is the full grid for Enumeration and the subset visited during the search for Binary Search. The output $d _ { i }$ is judged against the reference in-range set $\mathcal { R } _ { i }$ via the mismatch ratio defined above. Figure 2(b) varies $M _ { \mathrm { p r o b e } } \in \{ 2 , 4 , 8 , 1 6 , 3 2 \}$ . Scheduler implementations follow the hint-based RL baselines in Table 7.

Agent- $\mathbf { G } ^ { 2 }$ . For $\mathrm { { A g e n t { - } G ^ { 2 } } }$ in Figure 2(b), we draw $d _ { i } \sim \mathcal N ( \mu _ { i } , \sigma _ { i } ^ { 2 } )$ using the schedule parameters reached at step 50 (5 independent draws per task) and report the mismatch ratio averaged over draws and over tasks. No probing rollouts are performed: $\mu _ { i }$ and $\sigma _ { i }$ are read from the rollouts already collected for the GRPO update at that step.

Rollout-cost axis. The horizontal axis in Figure 2(b) reports $| { \mathcal { D } } ^ { \prime } | \cdot M _ { \mathrm { p r o b e } } / R .$ where $R = 8$ is the GRPO rollout budget per task. $\mathrm { \bf A g e n t { - } G ^ { 2 } }$ is plotted at 1× because its schedule is refreshed from these R rollouts without any additional probing.

Reference depth $d _ { i } ^ { \star }$ . The per-task reference depth

$$
d _ { i } ^ { \star } = \arg \operatorname* { m i n } _ { d \in \mathcal { D } } | \hat { p } _ { i } ( d ) - 0 . 5 | ,\tag{12}
$$

with ties broken in favor of the smaller depth, is used only in Figure 3(b) to align profiles via $\Delta d =$ $d - d _ { i } ^ { \star }$ . It plays no role in the mismatch computation of Section 2.

Gaussian fit. For Figure 3(b), we bin pairs by relative depth $\Delta d$ with width 0.1 and fit a exp(− $\cdot \Delta d ^ { 2 } / 2 \sigma ^ { 2 } )$ to the binned mean Bernoulli variance by least squares.

Table 5: Agent- $\mathbf { \nabla } \cdot \mathbf { G } ^ { 2 }$ hyperparameters. Shared across ALFWorld and WebShop unless noted. Hint-based RL baselines reuse the Policy optimization, Rollout, and Training schedule blocks.
<table><tr><td>Hyperparameter</td><td>Symbol</td><td>Value</td></tr><tr><td> $A g e n t - G ^ { 2 } - s p e c i f i c$ </td><td></td><td></td></tr><tr><td>Global baseline step</td><td>∆</td><td>0.1</td></tr><tr><td>Cluster EMA rate</td><td>α</td><td>0.2</td></tr><tr><td>Center scale</td><td> $\lambda$ </td><td>1.0</td></tr><tr><td>Variance scale</td><td> $\gamma$ </td><td>1.0</td></tr><tr><td>Variance floor</td><td> $\sigma _ { \mathrm { m i n } }$ </td><td>0.1</td></tr><tr><td>Aux. loss weight</td><td> $\eta$ </td><td>0.5</td></tr><tr><td>Target success rate</td><td> $p _ { \mathrm { t a r g e t } }$ </td><td>0.5</td></tr><tr><td>Initial baseline</td><td>(0)  $\mu _ { \mathrm { g l o b a l } } ^ { \mathrm { ( v ) } }$ </td><td>0.8</td></tr><tr><td>Difficulty clusters</td><td> $K$ </td><td></td></tr><tr><td>Policy optimization</td><td></td><td></td></tr><tr><td>Optimizer</td><td></td><td>AdamW</td></tr><tr><td>Peak learning rate</td><td></td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>LR schedule</td><td></td><td>cosine, 10% warmup</td></tr><tr><td>AdamW  $( \beta _ { 1 } , \beta _ { 2 } )$ </td><td></td><td>(0.9,0.999)</td></tr><tr><td>Weight decay</td><td></td><td>0.01</td></tr><tr><td>GRPO clip ratio</td><td>€</td><td>0.2</td></tr><tr><td>KL penalty</td><td> $\beta _ { \mathrm { K L } }$ </td><td>0.01</td></tr><tr><td>Precision</td><td></td><td>bfloat16</td></tr><tr><td>Rollout</td><td></td><td></td></tr><tr><td>Tasks per step</td><td>|B|</td><td></td></tr><tr><td>Rollouts per task</td><td>R</td><td>16</td></tr><tr><td>Max prompt length</td><td></td><td>7000</td></tr><tr><td>Max response length</td><td></td><td>512</td></tr><tr><td>Temperature (train/eval)</td><td></td><td>1.0/0.4</td></tr><tr><td>Training schedule</td><td></td><td></td></tr><tr><td>Steps (ALFWorld)</td><td></td><td>200</td></tr><tr><td>Steps (WebShop)</td><td></td><td>150</td></tr></table>

Gaussian fit across training. Figure 3(b) reports the fit at step 50. To check that the Gaussian shape is not specific to this checkpoint, we partition the training trajectory into three non-overlapping windows, pool aligned pairs (∆d, $p _ { i } ( d ) ( 1 - p _ { i } ( d ) ) )$ within each window to enlarge the per-window sample size and reduce statistical noise in the fit, and refit the same form (Table 6). The fit quality stays high across all three windows $( R ^ { 2 } \in [ 0 . 8 9 , 0 . 9 4 ] )$ and the fitted width σ drifts only mildly from 0.198 early to 0.231 late, consistent with a gradually widening informative band as the policy improves and with the upward trend of the per-cluster $\sigma _ { k }$ in Figure 6(b). The two-parameter Gaussian thus captures the informativeness profile throughout training, not only at the mid-training checkpoint used in the figure.

Choice of Gaussian for the sampler. Our choice of Gaussian for the sampler in Section 3 rests on two design considerations rather than fit quality alone. First, the Gaussian provides a simple twomoment parameterization whose mean and variance align directly with the two statistics that batch rollouts naturally produce: $\mu _ { i }$ is driven by the per-cluster empirical success rate, and $\sigma _ { i }$ by the per-cluster success-rate variance. Other symmetric families require additional shape parameters (for instance, the degrees of freedom of Student-t or the shape exponent of generalized normal) that cannot be read directly from rollout aggregates. Second, Gaussian sampling on the bounded ratio $r _ { i } \in [ 0 , 1 ]$ admits a closed-form truncated form and is numerically stable; heavy-tailed alternatives such as Cauchy produce unbounded magnitudes that would require ad-hoc clipping. The ablation in Table 3 confirms that on top of these design considerations, replacing the Gaussian with a variance-matched uniform distribution over the same band drops overall success by 7.0 points and Long-horizon success by 13.9 points.

Table 6: Gaussian fit of the aligned informativeness profile across training windows on Qwen2.5-1.5B / ALFWorld. Each window pools $( \Delta d , p _ { i } ( d ) ( 1 - p _ { i } ( d ) ) )$ pairs collected within the listed checkpoint range; n counts pairs entering the least-squares fit.
<table><tr><td>Training Stage</td><td>σ</td><td> $R ^ { 2 }$ </td><td>n</td></tr><tr><td>Early (ckpts 0–50)</td><td>0.198</td><td>0.906</td><td>116</td></tr><tr><td>Mid (ckpts 50–100)</td><td>0.206</td><td>0.888</td><td>146</td></tr><tr><td>Late (ckpts 100–150)</td><td>0.231</td><td>0.936</td><td>109</td></tr></table>

## B Implementation Details

Hyperparameters. Table 5 lists the full Agent-$\mathrm { G ^ { 2 } }$ configuration. On both benchmarks we build on the GiGPO codebase (Feng et al., 2025), adopting its training recipe for shared hyperparameters and inheriting its train/validation/test splits.

Baseline configurations. Table 7 summarizes the trained baselines; prompting baselines run inference-time without further specification. We use the authors’ public implementations with default hyperparameters for GiGPO, BEACON, StepHint, and TraPO.

Expert trajectory source. The expert trajectories used by $\mathcal { L } _ { \mathrm { a u x } }$ are drawn from the same ALFWorld and WebShop expert pool as RLVMR (Zhang et al., 2025d) and ETO (Song et al., 2024), so the underlying expert data is identical across the three methods. The difference lies in usage: $\mathrm { \bf A g e n t { - } G ^ { 2 } }$ treats the prefix tokens only as a token-level imitation target inside the joint objective, without introducing any auxiliary network or additional supervision signal. The gains over RLVMR and ETO in Table 1 therefore reflect algorithmic differences rather than any advantage in expert-data access.

![](images/269a0e8983706c1eb7c8f1b7833ffd001ea554a3fcf042f637c99900676edd5c.jpg)

![](images/04c232b51426c508c525ade412214f0abf6d45b88f1dfd458fb2896c014c8e45.jpg)

![](images/8fe02075e47e8b6cec90858568e57eced3d761a2a72c34ec2e248142c1656642.jpg)

(i) t = 5 — Deep init (ii) t = 30 — Clusters separate (iii) t = 75 — Spreads peak (iv) t = 200 — Converged  
![](images/6ab38782168fbba2be0b86398d89cb072421bf83d705ea365292de982dcb1a13.jpg)

![](images/e6518ab9be0e90d04cdf8e7b08640009b10ed81a4aaa274dc7a1039525027644.jpg)  
Figure 7: Four-phase snapshot of the per-cluster Gaussian schedule on Qwen2.5-1.5B / ALFWorld. Each panel shows the density $\mathcal { N } ( \mu _ { k } , \sigma _ { k } ^ { 2 } )$ for the K=3 horizon clusters (Short, Medium, Long) at one training step. Their batch-weighted mixture yields the aggregate in Figure 6(b).

Table 7: Baseline configurations. Hint-based RL baselines reuse the training blocks of Table 5; only the depthassignment rule differs.
<table><tr><td>Method</td><td>Configuration</td></tr><tr><td>Imitation Full SFT</td><td> ${ \mathrm { L R ~ } } 1 { \times } 1 0 ^ { - 5 }$  , |B|=32, 2 epochs</td></tr><tr><td>RL without hints</td><td></td></tr><tr><td>GRPO</td><td>Table 5 with  $d _ { i } { = } 0$ </td></tr><tr><td>GiGPO</td><td>authors&#x27;default</td></tr><tr><td>BEACON</td><td>authors&#x27; default</td></tr><tr><td>Hint-based RL: schedule</td><td></td></tr><tr><td>Linear</td><td>linear,  $1 . 0  0 . 0$ </td></tr><tr><td>Cosine</td><td>cosine, 1.0 → 0.0</td></tr><tr><td>Step</td><td>step decay 0.1 / 20 steps</td></tr><tr><td>Target-acc</td><td>online to batch-acc 0.8</td></tr><tr><td>Hint-based RL: search</td><td></td></tr><tr><td>Binary Search Enumeration</td><td>O(log |D|) probes per task exhaustive over D</td></tr><tr><td>Hint-based RL: end-to-end</td><td></td></tr><tr><td>StepHint</td><td>authors&#x27;default</td></tr><tr><td></td><td></td></tr><tr><td>TraPO</td><td>authors&#x27; default</td></tr></table>

Hardware and framework. We implement training in the verl framework (Sheng et al., 2025) and run all experiments on a single node with 8 NVIDIA H800 GPUs; a Qwen2.5-1.5B-Instruct training run on ALFWorld (200 gradient steps) takes approximately 10 hours.

Per-cluster training dynamics. Figure 7 samples the per-cluster Gaussian schedule $\mathcal { N } ( \mu _ { k } , \sigma _ { k } ^ { 2 } )$ at four training steps that mark distinct phases of training. At t=5 (saturation), $\mu _ { k }$ stays near 1 on every cluster (0.98, 1.00, 1.00 for Short, Medium, Long) with $\sigma _ { k } \approx 0 . 1 1$ , because the policy cannot yet reach learnable states unaided. At t=30 (separation), the means split by horizon (0.19, 0.35, 0.68) and $\sigma _ { k }$ widens with the cluster variance $V _ { k }$ At t=75 (peak spread), the means contract toward 0 on all clusters while $\sigma _ { k }$ reaches its peak (0.15, 0.21, 0.23); the sampler draws a wide neighborhood around the shallow center to keep harder tasks in the just-learnable band. By t=200 (collapse), $\mu _ { k } = 0$ on every cluster and $\sigma _ { k }$ recedes toward $\sigma _ { \mathrm { m i n } } { = } 0 . 1$ , with only Long retaining a wider spread (0.16), consistent with the slower convergence of long-horizon tasks. Figure 8 plots the full $( \mu _ { k } , \sigma _ { k } )$ trajectories across all 200 training steps for the same three clusters.

Figure 8: Per-cluster schedule trajectories on Qwen2.5-1.5B / ALFWorld. Each row shows one horizon cluster $( k _ { 1 }$ Short, $k _ { 2 }$ Medium, $k _ { 3 }$ Long); columns plot $\mu _ { t }$ and $\sigma _ { t }$ over training steps.

Table 8: Clustering design ablation on Qwen2.5-1.5B / ALFWorld (success rate %). Length-quantile bins tasks by horizon; random samples a partition uniformly. ± values are 5-seed standard deviations. $K { = } 1$ and lengthquantile $K { = } 3$ are reproduced from Table 3.
<table><tr><td>Clustering signal</td><td>K</td><td>All</td></tr><tr><td>None</td><td>1</td><td>89.1</td></tr><tr><td>Length quantile</td><td>3</td><td> $9 5 . 3 { \scriptstyle \pm 2 . 3 }$ </td></tr><tr><td>Length quantile</td><td>5</td><td> $9 6 . 5 { \scriptstyle \pm 1 . 9 }$ </td></tr><tr><td>Length quantile</td><td>10</td><td>86.2</td></tr><tr><td>Random</td><td>3</td><td>89.4</td></tr></table>

Clustering design. Table 8 sweeps the cluster count for length-quantile clustering and adds a random-assignment control on Qwen2.5-1.5B / ALFWorld.

Length-quantile clustering is stable in the $K { = } 3 -$ 5 range: $K { = } 3 \ ( 9 5 . 3 { \pm } 2 . 3 )$ and $K { = } 5 \ ( 9 6 . 5 { \scriptstyle \pm 1 . 9 } )$ overlap within run-to-run noise, so the difference between them is not significant under our seed budget. Pushing to $K { = } 1 0$ thins each cluster to under two tasks per batch under |B|=16 and drops accuracy by 9.1 points relative to $K { = } 3$ , as per-cluster sample size becomes inadequate for stable EMA estimation. Random assignment at $K { = } 3$ comes within 0.5 points of $K { = } 1$ , so the gain from clustering relies on alignment with difficulty, not on partitioning itself.

We pick length as the alignment signal for two reasons. (i) Length correlates with the difficulty of reaching the terminal reward without guidance, so it accelerates the convergence of $( A _ { k } , V _ { k } )$ relative to a difficulty-orthogonal signal. (ii) Length needs no task-type annotation or semantic embedding, so the same rule applies to both benchmarks.

We fix $K { = } 3$ across both ALFWorld and Web-Shop without per-benchmark tuning, because it matches the Short / Medium / Long taxonomy of Section 4.1 and keeps per-cluster sample size sufficient for stable EMA under $| B | { = } 1 6$ . The 6.2-point K=1 vs. K=3 gap in Table 3 bounds what a richer signal could recover; in benchmarks where horizon and difficulty decouple, additional signals beyond length may help.