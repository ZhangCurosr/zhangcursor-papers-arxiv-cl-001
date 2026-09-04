# Headroom-Drift Replay: A Primitive for Principled Replay Control in GRPO

Hyun Bin Park & Du-Seong Chang Department of Artificial Intelligence, Sogang University Seoul, Republic of Korea tjrtkroal@sogang.ac.kr duseong.chang@gmail.com

## Abstract

RL-based post-training for reasoning models is increasingly bottlenecked by repeated fresh rollout generation, particularly in agentic settings where environment interaction dominates wall-clock cost. Replay can reduce this burden by reusing past trajectories, but existing methods typically embed it within larger training pipelines involving exploration, experience restructuring, or mixed-policy optimization. This makes replay’s own contribution difficult to isolate. We ask a focused question: how far can principled replay selection alone go? We introduce Headroom-Drift Replay, a group-level replay control primitive for GRPO that separates reuse into two decisions. Headroom ranks stored groups by remaining learning value, while Drift gates them by compatibility with the current policy. The fresh on-policy stream remains unchanged, and the method adds no auxiliary generation or training machinery. Across mathematical reasoning, multimodal reasoning, and Agentic Search benchmarks, this single intervention outperforms naive replay and matches or exceeds broader replay methods on Avg Mean@32. In Agentic Search, where environment interaction dominates cost, it delivers comparable quality at materially lower wall-clock time.

(a) Mathematical Reasoning: Mean@32  
![](images/87c298e8e6c0c5cbf071bbe944275d4995a364bc6e5d3a3e7d7602cb05e04c6b.jpg)

(b) Agentic Search: Quality-Cost  
![](images/3c9c624ed9c76cfc54c6d82b83b7ec2a83e84a9bf9ee2fc3786b666e27a8ebbb.jpg)  
Figure 1: Summary of two representative views. (a) In mathematical reasoning, Headroom-Drift Replay leads on Avg Mean@32 over all baselines. (b) In agentic search, Headroom-Drift combines the highest Avg Best@32 with competitive wall-clock cost.

## 1 Introduction

Reinforcement learning (RL) has become a central ingredient in post-training language models for reasoning (Ouyang et al., 2022; Bai et al., 2022), but it often incurs substantial cost through repeated fresh rollout generation. This burden is especially high in GRPOstyle training and grows further in agentic settings, where interaction with external environments adds significant wall-clock overhead (Wang et al., 2026a; Liu et al., 2026a;b). Replay offers a natural remedy by reusing previously collected trajectories. Yet naive reuse is often insufficient for stable learning. The question is not whether to replay, but how to control replay itself: which stored experience should re-enter training, and under what conditions.

A stored replay group can fail for two distinct reasons. It may still contain useful learning signal but have become too stale under the current policy. Conversely, it may remain close to the current policy while offering little room for further learning. Reliable replay therefore requires two separate judgments: whether a group still offers learning value and whether it remains compatible with the current policy.

Prior work has shown that replay can improve sample efficiency in reasoning-oriented RL, but existing methods typically couple it with auxiliary mechanisms. This makes replay itself difficult to study as a standalone control problem. We therefore isolate replay-side control within GRPO (Shao et al., 2024; Guo et al., 2025) and ask how far it can go on its own.

Headroom-Drift Replay is a group-level replay control primitive for GRPO built around separate judgments of learning value and current-policy compatibility. Headroom measures the remaining room to shift probability toward good actions and away from bad ones. Policy Drift measures how far the current policy has moved from the policy that generated the group and gates out stale groups. Reuse operates on full groups rather than individual responses, preserving the within-group comparison structure central to GRPO. Together, these two controls form the only addition to standard GRPO; the fresh on-policy stream remains unchanged. This design isolates the effect of replay-side control on training dynamics.

Empirically, Headroom-Drift outperforms naive replay across mathematical reasoning, multimodal reasoning, and Agentic Search, and matches or exceeds methods that combine replay with additional training machinery. The strongest evidence comes from mathematical reasoning under the fullest baseline set. In Agentic Search, where environment interaction dominates wall-clock cost, the benefit takes a different form: Headroom-Drift outperforms on-policy scaling in both quality and per-step efficiency because the selected trajectories remain informative enough to substitute for costly fresh environment interaction. Multimodal reasoning shows that the same control principle extends beyond text-only settings.

## Our contributions are threefold:

(i) We formulate replay in GRPO as a two-part control problem—retaining learning value while ensuring current-policy compatibility—and instantiate it in Headroom-Drift Replay, a group-level replay control primitive that requires no auxiliary training machinery.

(ii) We construct role-aligned baselines that isolate distinct control questions—on-policy budget matching, fresh-data scaling, naive replay volume, strong non-replay alternatives, and broader replay methods with auxiliary machinery—and show across mathematical reasoning, Agentic Search, and multimodal reasoning that this primitive outperforms naive replay and matches or exceeds broader replay methods.

(iii) We analyze how principled replay control behaves across training, covering samebuffer counterfactual comparisons, replay-age compatibility and lifetime exposure concentration, multi-reward ingress dynamics, and the relationship between replay and entropy collapse.

## 2 Related work

Classical reinforcement learning motivates the two control axes in our design. Prioritized Experience Replay showed that stored samples differ substantially in learning value, and that replay is more effective when guided by expected training utility rather than random selection (Schaul et al., 2016). A complementary line of off-policy correction methods addresses policy mismatch, showing that past experience can remain useful when reuse is appropriately weighted or truncated under the current policy (Munos et al., 2016; Espeholt et al., 2018). These works operate on transitions or trajectories in classical RL rather than on groups in LLM post-training, but they establish the same two requirements that drive our design: replay should account for both learning value and current-policy compatibility. Headroom provides a PER-style learning-value priority over full GRPO groups; Headroom-Drift applies this priority only among groups admitted by the Drift-based current-policy compatibility gate.

In reasoning-oriented LLM training, replay has moved from a peripheral heuristic to a more central optimization component. RePO (Li et al., 2025), for example, incorporates replay into the GRPO training loop to improve sample efficiency beyond fully on-policy updates. This shift confirms that replay carries real value in modern reasoning settings where rollout generation is expensive.

Several recent methods strengthen replay by coupling it with broader training mechanisms. EFRame (Wang et al., 2025) combines replay with exploration and filtering to determine which trajectories are retained and revisited. ExGRPO (Zhan et al., 2026) organizes reasoning trajectories by correctness and entropy signals before mixed-policy reuse. BAPO (Wan et al., 2026) adopts a buffer-centric off-policy design that pairs historical hard samples with adaptive batch construction. These methods reinforce the value of replay in modern reasoning RL, but because replay is embedded within larger pipelines, the two control questions central to our work—which stored groups still offer learning value, and which remain compatible with the current policy—are rarely made explicit as separable design choices.

Our work treats these two questions as the primary intervention. The goal is not to replace broader replay-centric designs, but to provide a composable primitive within that larger design space: one that can stand alone and whose effects can be independently evaluated.

## 3 Method

## 3.1 Method overview.

Headroom-Drift Replay is a group-level replay control primitive for GRPO that decomposes reuse into two explicit control axes. Rather than treating reuse as a single inclusion decision, it separates learning-value prioritization from current-policy compatibility control through Headroom-based ranking and Policy-Drift gating. The minimum unit of reuse is the full group rather than individual responses, preserving the within-group comparison structure central to GRPO. The fresh on-policy stream is left untouched: the only modification to standard GRPO is that selected replay groups are added to the mixed batch. Any change in training dynamics therefore traces to replay-side control alone.

## 3.2 GRPO setup and replay state.

We first specify the GRPO state needed to express replay groups within the same update form as fresh on-policy groups. For a prompt Q, the current policy π generates n responses $G _ { 1 } , \ldots , G _ { n } ,$ which together form a group ${ \dot { g } } = ( Q , G _ { 1 } , \dots , { \dot { G } } _ { n } )$ . Let $\scriptstyle { \mathcal { T } } ( g )$ denote the set of token positions in the group, and let $a _ { i , j }$ and $s _ { i , j }$ denote the j-th generated token in response $G _ { i }$ and its conditioning context. The current policy assigns token probability $\pi _ { \theta } ( a _ { i , j } \mid s _ { i , j } )$ GRPO computes rewards at the response level and derives a group-relative advantage $A _ { i }$ for each response $G _ { i } ;$ this same $A _ { i }$ is then applied to all token positions belonging to that response.

![](images/e716bb721b91199937ec470dfec02418d263fa58ec46df8caddd6c0bcb345161.jpg)  
Figure 2: Headroom-Drift Replay in one GRPO step. The current policy generates fresh on-policy groups. Pre-existing buffered groups are ordered by Headroom and filtered by the Policy Drift gate; up to $K _ { \mathrm { r e p } } ^ { \mathrm { ^ { - } } }$ accepted replay groups are merged with the fresh groups to form the mixed batch for the GRPO update. After the update, fresh groups with nonidentical rewards are appended to the FIFO replay buffer.

What distinguishes replay from the fresh on-policy case is that each stored group is paired with the policy state under which it was originally generated. We denote this reference policy by $\pi _ { \mathrm { g e n } ( g ) }$ . For a freshly generated group, $\pi _ { \mathrm { g e n } ( g ) }$ coincides with the usual rollouttime old policy in GRPO. For a replayed group, however, it is the generation-time policy stored with that group in the buffer. Accordingly, the replay buffer maintains immutable per-group reference state: the stored actions, their generation-time log-probabilities under $\pi _ { \mathrm { g e n } ( g ) }$ , and the response-level advantages $A _ { i }$ . This frozen reference state anchors each replayed group to the rollout under which it was originally collected.

Using this reference state, both fresh and replayed groups can be handled within the same GRPO update form. For a group ${ \mathit { g } } ,$ the token-level importance ratio is

$$
r _ { i , j } ( \theta ; g ) = \frac { \pi _ { \theta } ( a _ { i , j } \mid s _ { i , j } ) } { \pi _ { \mathrm { g e n } ( g ) } ( a _ { i , j } \mid s _ { i , j } ) } ,
$$

so the difference between fresh and replayed groups lies only in which reference policy appears in the denominator.

## 3.3 Headroom: learning-value prioritization.

Headroom provides the learning-value signal in our replay-side control primitive. It measures the remaining directional correction room of a stored group under a reference policy. Intuitively, replay is more useful when positively advantaged stored actions still have room for probability increase, or negatively advantaged stored actions still have room for probability decrease. For a policy π and a stored token position $( i , j )$ , define the token-level Headroom contribution by

$$
h _ { i , j } ( \pi ; g ) = \left\{ \begin{array} { l l } { 1 - \pi ( a _ { i , j } \mid s _ { i , j } ) , } & { A _ { i } > 0 , } \\ { \pi ( a _ { i , j } \mid s _ { i , j } ) , } & { A _ { i } < 0 , } \\ { 0 , } & { A _ { i } = 0 , } \end{array} \right.
$$

and aggregate it over the group as

$$
\mathrm { H e a d r o o m } ( g ; \pi ) = \frac { 1 } { | T ( g ) | } \sum _ { ( i , j ) \in { \mathcal T } ( g ) } h _ { i , j } ( \pi ; g ) .
$$

When evaluated under the generation-time policy $\pi _ { \mathrm { g e n } ( g ) . }$ , Headroom defines the valueside replay priority stored for group $g .$

## 3.4 Policy drift: current-policy compatibility control.

Policy Drift provides the current-policy compatibility control in our replay-side control primitive. A stored group may still have high Headroom while already being too far from the current policy for reliable reuse (Munos et al., 2016; Espeholt et al., 2018).

To measure this mismatch for a stored group $g ,$ define the token-level log-probability drift on the stored action by

$$
\Delta _ { i , j } ( g ; \theta ) = \log \pi _ { \theta } ( a _ { i , j } \mid s _ { i , j } ) - \log \pi _ { \mathrm { g e n } ( g ) } ( a _ { i , j } \mid s _ { i , j } ) .
$$

We then aggregate these tokenwise deviations into a group-level Policy Drift score by averaging their squared values over the group:

$$
\mathrm { D r i f t } ( g ; \theta ) = \frac { 1 } { | \mathcal { T } ( g ) | } \sum _ { ( i , j ) \in \mathcal { T } ( g ) } \Delta _ { i , j } ( g ; \theta ) ^ { 2 } .
$$

A group is replayed only if its Policy Drift does not exceed a threshold τ.

This choice is motivated by the role of Policy Drift as a replay-compatibility gate rather than a full distribution-distance objective. GRPO commonly monitors policy deviation through approximate-KL proxies (Guo et al., 2025; Yu et al., 2025) built from sampled log-ratios on generated tokens. For replay gating, however, what matters is mismatch magnitude on the stored token sequence, not a signed average: if some tokenwise shifts are positive while others are negative, they can offset each other in the average even when the overall sequence has moved substantially under the current policy. We therefore use a non-cancelling aggregation of tokenwise drift, and adopt a squared (L2-style) form so that larger local mismatches are penalized more strongly. An L1 variant yields a tighter formal bound but is less sensitive to concentrated per-token mismatch; in practice, the L2 gate achieves higher late-stage validation while filling less of the replay budget, indicating that it wins through better-compatible selection rather than replay volume (Appendix G.1). This squared aggregation also provides a formal link between the two control axes: when a group passes the Drift gate, its current-policy Headroom can deviate from the stored Headroom priority by at most $\sqrt { \tau } ,$ so the gate directly limits how much staleness can distort the stored Headroom priority (proof in Appendix A.3, Proposition 2).

## 3.5 One GRPO step with replay-side control

Phase A: Generate fresh groups and identify replay ingress candidates. At each training step, the current policy generates fresh grouped rollouts, and rewards and group-relative advantages are computed as in standard GRPO. Replay ingress candidates $\textstyle { \mathcal { T } } _ { t }$ are then identified from these fresh groups.

Phase B: Order pre-existing buffered groups by Headroom. The pre-existing buffered groups are retrieved and ordered by Headroom, which defines the value-side priority for replay candidate scanning.

Phase C: Apply the Policy Drift gate. The Headroom-ordered candidates are scanned under the current policy. For each scanned group, the model reevaluates the group via parallel teacher-forced log-probability computation, and admits the group for replay only if its Policy Drift does not exceed τ. Scanning stops once the replay budget $K _ { \mathrm { r e p } }$ is filled or no further admissible groups remain.

Phase D: Form the mixed batch and run the GRPO update. The accepted replay groups are merged with the fresh on-policy groups to form the mixed batch for the GRPO update.

Phase E: Append valid ingress groups and maintain the FIFO buffer. To prevent samestep replay, the previously identified ingress candidates $\mathcal { T } _ { t }$ are appended to the replay buffer only after the GRPO update. The buffer is maintained as a fixed-capacity FIFO queue, so once capacity is exceeded, the oldest groups are evicted regardless of Policy Drift or Headroom.

Operational note. Replay requires no fresh autoregressive generation and no fresh environment interaction. Because reevaluation is teacher-forced on fixed stored token se-

Algorithm 1 HEADROOM-DRIFT REPLAY as a replay-side control augmentation to stan  
dard GRPO. Blue lines mark replay-specific additions; the fresh on-policy rollout path re  
mains unchanged.   
Input: current policy $\pi _ { \theta _ { t } } ,$ prompt batch $\mathcal { X } _ { t } ,$ , replay buffer $\mathbf { B } _ { t } ,$ replay budget $K _ { \mathrm { r e p } } ,$ drift threshold   
1: for each training step t do   
2: Sample prompt batch $\mathcal { X } _ { t }$   
3: Generate grouped fresh rollouts under $\pi _ { \theta _ { t } }$   
4: Compute rewards and group-relative advantages for the fresh groups   
5: + Identify mixed-outcome fresh groups as replay ingress candidates $\mathcal { T } _ { t }$   
6: + Order pre-existing buffered groups by Headroom (descending)   
7: for each candidate group $g$ in Headroom order do   
8: if $| \mathcal { R } _ { t } | \geq K _ { \mathrm { r e p } }$ then   
9: break   
10: end if   
11: + Reevaluate the stored actions of $g$ under $\pi _ { \theta _ { t } }$ ▷ singleforward pass onfixed trajectories   
12: + Compute Policy Drift and refresh Headroom for $g$ from the same log-probability pass   
13: + if Policy $\textstyle \operatorname { D r i f t } ( { \tilde { g } } ) \leq \tau$ then add $g$ to $\mathcal { R } _ { t }$   
14: end for   
15: + Form mixed actor batch $\mathcal { A } _ { t }  \mathcal { G } _ { t } ^ { \mathrm { o n } } \cup \mathcal { R } _ { t }$   
16: Run the standard GRPO update on $\boldsymbol { A } _ { t }$   
17: + Append ingress groups $\dot { \mathcal { T } } _ { t }$ to $\mathbf { B } _ { t }$ after selection ▷ no same-step replay   
18: + If capacity is exceeded, evict the oldest groups from $\mathbf { B } _ { t }$ (FIFO)   
19: end for

quences, it can be parallelized over stored tokens. For efficiency, the current-policy logprobabilities computed for Policy Drift are reused to refresh Headroom in the same pass. Because scanning stops once the replay budget is filled, cached Headroom for unscanned groups can become stale. The two mechanisms therefore act complementarily: Policy Drift gating protects admission from overly stale groups, while same-pass Headroom refresh keeps the cached priority accurate for groups that are actually revisited. Appendix A makes the cached/reference/runtime Headroom distinction precise and formalizes replay selection in a form that directly mirrors the implementation above. Appendix F details the engineering choices required for stable and efficient execution, including frozen reference-state management, replay-prefix batch layout, and sequence-length balancing across GPUs.

## 4 Experiments

## 4.1 Experimental setup

We evaluate Headroom-Drift Replay in three GRPO-style settings with fixed, verifiable rewards: mathematical reasoning, Agentic Search, and multimodal reasoning. For mathematical reasoning, we use AIME24, AMC23, MATH500, Minerva, and OlympiadBench. For Agentic Search, we evaluate on NQ, TriviaQA, PopQA, HotpotQA, 2WikiMulti HopQA, Musique, and Bamboogle. For multimodal reasoning, we report a compact threebenchmark view covering Geometry3K (Geo3K), MathVista, and MathVision. We use Mean@32 as the headline metric for the main cross-domain comparisons. For their crossbenchmark summaries, we report Avg Mean@32. Best@32 and Weighted Mean@32 are reported in Appendix B for completeness. We additionally include a 7B Agentic Search scale check.

Our baselines are chosen to answer distinct control questions, and we refer to them by role rather than by raw budget alone. GRPO on-policy matched keeps the fresh rollout budget aligned with replay and isolates the effect of replay reuse itself. GRPO on-policy larger tests whether replay gains can be explained by simple fresh-data scaling. GRPO + replay matched and, where available, GRPO + replay larger test whether buffered reuse alone is sufficient, or whether principled replay control is necessary. DAPO serves as a strong nonreplay comparator, and where directly comparable we also include broader replay/offpolicy methods such as ExGRPO and BAPO. In mathematical reasoning, we include the full comparison set. In multimodal reasoning, we report GRPO on-policy matched, GRPO onpolicy larger, and DAPO in a simplified three-benchmark view. In Agentic Search, the main role-aligned comparisons are GRPO on-policy matched, GRPO on-policy larger, GRPO + replay matched, and DAPO. We do not include ExGRPO or BAPO in this setting because their published implementations target prompt-conditioned reasoning trajectories rather than externally interleaved tool-calling trajectories, and a like-for-like adaptation would require substantial re-engineering rather than a clean baseline comparison.

![](images/1daea4696557fd065a9945cc853fa5dded41529f84f5f3617be95c0b1f342bf1.jpg)  
Figure 3: Cross-domain summary with Mean@32.

Because replay cost varies across settings—negligible in single-turn mathematical reasoning but substantial in Agentic Search where environment interaction dominates wall-clock time—we interpret Mean@32 together with acquisition cost. Table 1 reports mean per-step wall-clock time as the main cost-side comparator.

## 4.2 Evaluation across domains

## 4.2.1 Mathematical reasoning

Figure 3 reports the mathematical-reasoning held-out results under the fullest comparison set. Headroom-Drift leads on Avg Mean@32 across all baselines. Relative to GRPO on-policy matched, the improvement confirms that principled replay adds value beyond purely on-policy training at matched fresh-rollout budget. Relative to GRPO on-policy larger, Headroom-Drift still leads despite using fewer fresh responses, ruling out simple fresh-data scaling as an explanation. Against GRPO + replay matched and GRPO + replay larger, the margin widens further, indicating that replay quality depends on principled subset selection rather than on replay volume. Among the broader comparators, DAPO, ExGRPO, and BAPO are all surpassed on Avg Mean@32; this is the one domain where the primitive matches or exceeds every baseline under a single headline metric.

Figure 4 provides supporting training-dynamics evidence. The replay trajectory stays above GRPO on-policy matched over a broad mid-to-late window and tracks above GRPO on-policy larger through much of the same range, indicating that the Avg Mean@32 advantage is reflected in the training trajectory rather than confined to a single checkpoint. A training-score-matched analysis in Appendix E further shows that replay delays the onset of entropy collapse: at comparable score levels, the replay run enters the low-entropy regime later and remains there longer than GRPO on-policy larger, suggesting that principled replay contributes to more sustained learning dynamics.

## 4.2.2 Agentic search

Agentic search serves a different role from mathematical reasoning: the main question is not the largest standalone gain in Mean@32, but whether replay-side control remains worthwhile when training is dominated by multi-turn environment interaction cost.

Against naive replay baselines, the picture depends on the metric. On the headline Avg Mean@32, Headroom-Drift leads GRPO + replay matched by only 0.0029 (0.3577 vs. 0.3548), a gap within noise range. But on Avg Best@32, the margin is substantially larger (0.4879 vs. 0.4623, as shown in Figure 1(b)), suggesting that principled selection shifts the upper tail of trajectory quality more than it shifts the average.

![](images/498373ed3746ff4aa3b29fac03dc44a478e33447520847c535fbc2383ce50a14.jpg)  
Figure 4: Training-score trajectories on mathematical reasoning.

As an additional scale check, we conduct a controlled Search-R1 comparison between Headroom-Drift and GRPO + replay matched with Qwen2.5-7B-Instruct. To make 7B training and agentic evaluation feasible within the available compute budget, both methods use an 8-bit optimizer and we evaluate four samples per input; we therefore report Mean@4 and Best@4 rather than their @32 counterparts. Under this matched setup, Headroom-Drift improves Avg Mean@4 from 0.3737 to 0.3955 and Weighted Mean@4 from 0.4158 to 0.4298, leading on six of seven benchmarks. Full per-benchmark Mean@4 and Best@4 results are reported in Appendix B, Table 6.

<table><tr><td>Method Step Time (s)</td></tr><tr><td>GRPO on-policy matched (b128) 146.4</td></tr><tr><td>GRPO + replay matched (b128+r64) 154.4</td></tr><tr><td>Headroom-Drift (b128+r64) 166.3</td></tr><tr><td>GRPO + replay larger (b128+r128) 187.3</td></tr><tr><td>GRPO on-policy larger (b192) 197.2</td></tr><tr><td>DAPO 232.5</td></tr></table>

Table 1: Mean per-step wall-clock time on agentic search (Search-R1).

The more revealing comparison is against on-policy scaling. GRPO on-policy larger uses 1.5× the fresh rollout budget yet achieves lower Avg Mean@32 (0.3212 vs. 0.3577) at higher per-step cost (197.2 s vs. 166.3 s). Headroom-Drift surpasses it on both axes because principled selection substitutes reevaluation of stored trajectories for costly fresh environment interaction (Table 1). This is a Pareto improvement over on-policy scaling.

## 4.2.3 Multimodal reasoning

Multimodal reasoning serves as breadth evidence. The question is whether the same replay-side control principle generalizes beyond text-only settings. In this simplified threebenchmark view, Headroom-Drift leads on Avg Mean@32 over GRPO on-policy matched, GRPO on-policy larger, and DAPO (0.4137 vs. 0.3986, 0.4005, and 0.4056 respectively).

Initial stored Headroom

![](images/e53b7704e7b16f16f953dfda6554abf67028bad8b6faeae21a006003e6e765fa.jpg)

![](images/56570d308c03fc67a130a1f25851f3a72ab0c32d1db1129801f1d1f71b585cf4.jpg)

![](images/534c4226ae67da2371b12af84993af75d9cf433b7af3d9379d31dc1fa973903a.jpg)  
Figure 5: (A) Same-buffer counterfactual against GRPO + replay. (B) Initial stored Headroom and lifetime replay. Bars show mean replay count $N _ { \mathrm { r e p . } }$ , and the line shows $P ( N _ { \mathrm { r e p } } \geq$ 1). (C) Conceptual view of replay source staleness under Headroom-Drift versus $\mathrm { G R P O + }$ replay.

## 4.3 Mechanism analysis of replay selection

We now ask whether the two control axes operate as designed. Figure 5 examines this question through three complementary views: step-level subset quality, lifetime reuse concentration, and temporal diversity of replay sources.

Figure 5(a) compares, at each training step, the replay subset actually selected by Headroom-Drift against the subset that GRPO + replay would select by recency alone, on matched pre-step buffer states and matched subset sizes. Each point plots the compatibility gain (x: positive means our subset has lower Policy Drift) against the learningvalue gain (y: positive means higher current-policy Headroom). The majority of steps fall in the jointly improved quadrant, confirming systematic advantage on both axes. The gain is asymmetric: policy compatibility improvement is consistent, while learning-value improvement weakens in late-stage steps where the admissible pool contracts. This pattern is expected—Policy Drift acts as a hard gate, while Headroom prioritizes within the feasible set that the gate admits.

Figure 5(b) shows that these step-level decisions accumulate into a sharp lifetime reuse pattern. Groups with initial stored Headroom below 0.8 are rarely replayed (mean $N _ { \mathrm { r e p } } <$ 0.5; replay probability < 24%), while groups above 0.8 are replayed frequently (mean $N _ { \mathrm { r e p } } ~ \stackrel { . } { \approx } ~ \dot { 2 } . 0 \dot { ; } > 5 6 \%$ . Headroom ranking preferentially surfaces high-value groups, but lower-Headroom groups are not excluded; they still enter replay when the Policy Drift gate judges them sufficiently compatible.

Figure 5(c) reveals the temporal consequence. GRPO + replay draws exclusively from the immediately preceding step (100% at age 1), since each step’s buffer ingress exceeds the replay target. Headroom-Drift distributes replay across ages 1–7, with 44% coming from sources three or more steps back. This multi-age mixture arises because Headroom ranking surfaces older high-value groups that still pass the Policy Drift gate, producing a temporally diverse corrective signal rather than uniform one-step reuse.

In practice, we calibrate the Drift threshold τ with a short log-spaced sweep instead of a fine-grained full-training search. On Search-R1, three decade-spaced settings yield mean replay acceptance rates of 6.4%, 40.2%, and 94.4%; only the middle setting filters selectively while keeping replay KL stable. Collapsing acceptance under a tight threshold and rising replay KL under a loose one provide early signals for ruling out unsuitable values without running every candidate to completion. Across the evaluated runs, the selected values fall into two task-family settings, and the Search-R1 threshold is reused from 3B to 7B without scale-specific retuning. Appendix G.2 reports the exact thresholds and full analysis.

Appendix D provides further empirical detail on these reuse patterns. Three findings stand out: age alone does not determine replay compatibility, as the Drift gate rejects a nonnegligible fraction of even the youngest groups; scan depth grows over training as the admissible set contracts; and most buffered groups are never replayed, while a small repeatedly reused subset accounts for the majority of total replay exposure. A component ablation supports the role of both axes. On MATH-500, the full Headroom-Drift combination achieves the highest Mean@32, followed by Headroom-only (the PER-style priorityonly analogue) and then Drift-only (Appendix G.3, Table 17).

## 5 Conclusion and future directions

We introduced Headroom-Drift Replay, a replay-side control primitive that decomposes reuse into learning-value prioritization (Headroom) and current-policy compatibility control (Drift). Across mathematical reasoning, agentic search, and multimodal reasoning, this single intervention outperforms naive replay. In mathematical reasoning under the fullest baseline set, it matches or exceeds broader replay methods on Avg Mean@32. In agentic search, it delivers a Pareto improvement over on-policy scaling. Replay, when controlled along the right axes, functions as a capable optimization primitive in its own right.

We studied this primitive in isolated form, but its two explicit control axes are designed to compose with the exploration, filtering, and adaptive batch construction strategies in recent replay-centric methods (Wang et al., 2025; Zhan et al., 2026; Wan et al., 2026). The primitive does not compete with these methods; it provides a separable control layer within them. Preliminary training-score evidence on a CISPO-style objective suggests that the two-axis decomposition transfers beyond GRPO (Appendix H), though held-out evaluation across objective families, including PPO-, GSPO-, and CISPO-style formulations (Zheng et al., 2025; MiniMax et al., 2025; Yuan et al., 2026; Xi et al., 2026), remains open.

## Acknowledgments

This research was conducted as part of the Sovereign AI Foundation Model Project (GPU Track), organized by the Ministry of Science and ICT (MSIT) and supported by the National IT Industry Promotion Agency (NIPA), S. Korea (PJT-26-010017). This research was supported by the MSIT (Ministry of Science, ICT), Korea, under the Top-Tier AI Global HRD invitation program (RS-2025-25461932) supervised by the IITP (Institute for Information & Communications Technology Planning & Evaluation).

## Ethics Statement

This work studies replay selection mechanisms for reinforcement-learning post-training of reasoning models. The experiments use standard benchmark datasets and simulated training environments; no new human-subject data were collected. Potential risks include misuse of improved reasoning capability and increased compute consumption in large-scale training. To mitigate overclaiming, we report results with explicit scope limits and emphasize quality–cost trade-offs under the evaluated settings. During camera-ready preparation, the authors used a large language model to revise prose and assist with drafting. The authors reviewed the resulting text and take responsibility for all content in the paper.

## References

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, et al. Constitutional AI: Harmlessness from AI feedback, 2022. URL https://arxiv.org/abs/2212.08073.

Lasse Espeholt, Hubert Soyer, Rémi Munos, Karen Simonyan, Vlad Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, Shane Legg, and Koray Kavukcuoglu. IMPALA: Scalable distributed deep-RL with importance weighted actorlearner architectures. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 1407–1416, 2018. URL https://proceedings.mlr.press/v80/espeholt18a.html.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning. Nature, 645(8081):633–638, 2025. doi: 10.1038/ s41586-025-09422-z. URL https://www.nature.com/articles/s41586-025-09422-z.

Siheng Li, Zhanhui Zhou, Wai Lam, Chao Yang, and Chaochao Lu. RePO: Replay-enhanced policy optimization, 2025. URL https://arxiv.org/abs/2506.09340.

Ben Lipkin, Aleksei Petrenko, Kevin Chen, Erik Wijmans, Marco Cusumano-Towner, Raja Giryes, and Philipp Krähenbühl. Entropy-preserving reinforcement learning. In International Conference on Learning Representations (ICLR), 2026. doi: 10.48550/arXiv.2603.11682. URL https://iclr.cc/virtual/2026/poster/10010707.

Xiaoqian Liu, Ke Wang, Yuchuan Wu, Fei Huang, Yongbin Li, Jianbin Jiao, and Junge Zhang. Agentic reinforcement learning with implicit step rewards. In International Conference on Learning Representations (ICLR), 2026a. doi: 10.48550/arXiv.2509.19199. URL https://iclr.cc/virtual/2026/poster/10007373.

Zichen Liu, Anya Sims, Keyu Duan, Changyu Chen, Simon Yu, Xiangxin Zhou, Haotian Xu, Shaopan Xiong, Bo Liu, Chenmien Tan, Weixun Wang, Hao Zhu, Weiyan Shi, Diyi Yang, Michael Qizhe Shieh, Yee Whye Teh, Wee Sun Lee, and Min Lin. GEM: A gym for generalist LLMs. In International Conference on Learning Representations (ICLR), 2026b. URL https://openreview.net/forum?id=vsqQ1lG52a.

Yuchun Miao, Sen Zhang, Liang Ding, Yuqi Zhang, Lefei Zhang, and Dacheng Tao. The energy loss phenomenon in RLHF: A new perspective on mitigating reward hacking. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 44076–44105, 2025. URL https://proceedings. mlr.press/v267/miao25c.html.

MiniMax, Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, et al. MiniMax-M1: Scaling test-time compute efficiently with lightning attention, 2025. URL https://arxiv.org/abs/2506.13585.

Rémi Munos, Tom Stepleton, Anna Harutyunyan, and Marc G. Bellemare. Safe and efficient off-policy reinforcement learning. In Advances in Neural Information Processing Systems, volume 29, 2016. URL https://proceedings.neurips.cc/paper/2016/hash/ c3992e9a68c5ae12bd18488bc579b30d-Abstract.html.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pp. 27730–27744, 2022. doi: 10.52202/ 068431-2011. URL https://proceedings.neurips.cc/paper\_files/paper/2022/hash/ b1efde53be364a73914f58805a001731-Abstract-Conference.html.

Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. Prioritized experience replay. In International Conference on Learning Representations (ICLR), 2016. doi: 10.48550/ arXiv.1511.05952. URL https://arxiv.org/abs/1511.05952.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/ 2402.03300.

Xu Wan, Yansheng Wang, Wenqi Huang, and Mingyang Sun. Buffer Matters: Unleashing the power of off-policy reinforcement learning in large language model reasoning. In International Conference on Learning Representations (ICLR), 2026. doi: 10.48550/arXiv.2602. 20722. URL https://iclr.cc/virtual/2026/poster/10009473.

Chen Wang, Lai Wei, Yanzhi Zhang, Chenyang Shao, Zedong Dan, Weiran Huang, Yuzhi Zhang, and Yue Wang. EFRame: Deeper reasoning via exploration-filter-replay reinforcement learning framework, 2025. URL https://arxiv.org/abs/2506.22200.

Guoqing Wang, Sunhao Dai, Guangze Ye, Zeyu Gan, Wei Yao, Yong Deng, Xiaofeng Wu, and Zhenzhe Ying. Information gain-based policy optimization: A simple and effective approach for multi-turn LLM agents. In International Conference on Learning Representations (ICLR), 2026a. doi: 10.48550/arXiv.2510.14967. URL https://iclr.cc/virtual/ 2026/poster/10007215.

Shumin Wang, Yuexiang Xie, Wenhao Zhang, Yuchang Sun, Yanxi Chen, Yaliang Li, and Yanyong Zhang. On the entropy dynamics in reinforcement fine-tuning of large language models, 2026b. URL https://arxiv.org/abs/2602.03392.

Zhiheng Xi, Xin Guo, Yang Nan, Enyu Zhou, Junrui Shen, Wenxiang Chen, Jiaqi Liu, Jixuan Huang, Zhihao Zhang, Honglin Guo, Xun Deng, Zhikai Lei, Miao Zheng, Guoteng Wang, Shuo Zhang, Peng Sun, Rui Zheng, Hang Yan, Tao Gui, Qi Zhang, and Xuanjing Huang. BAPO: Stabilizing off-policy reinforcement learning for LLMs via balanced policy optimization with adaptive clipping. In International Conference on Learning Representations (ICLR), 2026. URL https://openreview.net/forum?id=jIeJJqG7dz.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Juncai Liu, et al. DAPO: An open-source LLM reinforcement learning system at scale. In Advances in Neural Information Processing Systems, volume 38, pp. 113222–113244, 2025. doi: 10.52202/ 085713-3775. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ a4277440d50f1f15d2cb4c14f7e0c0d2-Abstract-Conference.html.

Dun Yuan, Di Wu, and Xue Liu. Escaping policy contraction: Contraction-aware PPO (CaPPO) for stable language model fine-tuning. In International Conference on Learning Representations (ICLR), 2026. URL https://openreview.net/forum?id=vDlkJewkDu.

Runzhe Zhan, Yafu Li, Zhi Wang, Xiaoye Qu, Dongrui Liu, Jing Shao, Derek F. Wong, and Yu Cheng. ExGRPO: Learning to reason from prior successes. In International Conference on Learning Representations (ICLR), 2026. doi: 10.48550/arXiv.2510.02245. URL https: //iclr.cc/virtual/2026/poster/10011339.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization, 2025. URL https://arxiv.org/abs/2507.18071.

## A Formal interpretation, mixed-batch objective, and replay-step formalization

This appendix makes precise the quantities and replay-step construction used by Headroom-Drift Replay. We first fix replay-state conventions, including the distinction between reference, cached, current-policy, and runtime Headroom. We then formalize the mixed-batch objective, define Headroom and Policy Drift, prove a reference-to-current Headroom mismatch bound, introduce lifetime replay exposure quantities, and finally write the replay-augmented GRPO step in set notation.

## A.1 Notation and replay-state conventions

For any buffered group g and step t, we distinguish four Headroom-related quantities. The reference Headroom is the immutable generation-time score

$$
H ^ { \mathrm { r e f } } ( g ) : = \mathrm { H e a d r o o m } ( g ; \pi _ { \mathrm { g e n } ( g ) } ) .\tag{1}
$$

The current-policy Headroom is the conceptual value of the same policy-conditioned quantity evaluated at the current policy,

$$
H _ { t } ^ { \operatorname { c u r } } ( g ) : = { \mathrm { H e a d r o o m } } ( g ; \pi _ { \theta _ { t } } ) .\tag{2}
$$

The cached Headroom $\widehat { H } _ { t } ( g )$ is the priority stored in the buffer at the beginning of step t and used to order replay candidates:

$$
\widehat { H } _ { t } ( g ) : = \mathrm { H e a d r o o m } ( g ; \pi _ { \theta _ { r _ { t } ( g ) } } ) ,\tag{3}
$$

where $r _ { t } ( g )$ is the last step at which the cached priority of $g$ was refreshed. Finally, for a buffered group $g$ that is actually scanned under the current policy at step t, the runtime Headroom is

$$
H _ { t } ^ { \mathrm { r u n } } ( g ) : = \mathrm { H e a d r o o m } ( g ; \pi _ { \theta _ { t } } ) .\tag{4}
$$

Thus, whenever $g$ is scanned at step $t ,$ we have

$$
H _ { t } ^ { \operatorname { r u n } } ( g ) = H _ { t } ^ { \operatorname { c u r } } ( g ) .\tag{5}
$$

At insertion time, the cached priority is initialized at the generation-time value, so a newly buffered group starts with $\widehat { H } _ { t } ( g ) = H ^ { \mathrm { r e f } } ( g )$ ). If a buffered group is scanned at step $t ,$ then its cached priority is refreshed after selection using the runtime value from the same currentpolicy reevaluation pass; otherwise the cached value remains unchanged. Consequently, current-step replay ordering is governed by $\widehat { H } _ { t } ( g )$ , not by a freshly recomputed currentpolicy Headroom over the entire buffer.

## A.2 Mixed-batch objective as constrained replay augmentation

Let $\mathcal { G } _ { t } ^ { \mathrm { o n } }$ denote the fresh on-policy groups generated at step $t ,$ and let $\mathcal { R } _ { t }$ denote the accepted replay groups selected from the pre-existing buffer. The mixed actor batch is

$$
\begin{array} { r } { \mathcal { A } _ { t } = \mathcal { G } _ { t } ^ { \mathrm { o n } } \not  \mathcal { R } _ { t } , } \end{array}\tag{6}
$$

where ⊎ denotes disjoint union.

For any group g, let

$$
\ell _ { \mathrm { G R P O } } \bigl ( g ; \theta , \pi _ { \mathrm { g e n } ( g ) } \bigr )\tag{7}
$$

be the usual group-level GRPO objective evaluated with the generation-time reference policy attached to $g .$ The one-step mixed-batch objective is then

$$
\mathcal { L } _ { t } ^ { \mathrm { m i x } } ( \theta ) = \frac { 1 } { \left| \mathcal { A } _ { t } \right| } \sum _ { g \in \mathcal { A } _ { t } } \ell _ { \mathrm { G R P O } } \big ( g ; \theta , \pi _ { \mathrm { g e n } ( g ) } \big ) .\tag{8}
$$

<table><tr><td>Symbol</td><td>Type</td><td>Meaning</td></tr><tr><td colspan="3">Core GRPO and replay state</td></tr><tr><td></td><td>group</td><td>grouped rollout associated with one prompt</td></tr><tr><td> $\mathbf { \Delta } _ { { \mathcal { T } } ( g ) } ^ { g }$ </td><td>index set</td><td>token positions contained in group 8 response-level group-relative advantage for response</td></tr><tr><td> $A _ { i }$ </td><td>scalar</td><td>i</td></tr><tr><td> $\pi _ { \mathrm { g e n } ( g ) }$ </td><td>policy</td><td>generation-time reference policy that originally produced group g</td></tr><tr><td> $\pi _ { \theta _ { t } }$ </td><td>policy</td><td>current policy at training step t</td></tr><tr><td> $t _ { 0 } ( g )$ </td><td>integer</td><td>step at which group g is freshly generated  $t _ { 0 } ( g )$ </td></tr><tr><td> $\mathsf { a g e } _ { t } ( g )$ </td><td>scalar</td><td>age of buffered group g at step  $t ,$  defined as t –</td></tr><tr><td colspan="3">Headroom and Policy Drift</td></tr><tr><td>Headroom(g; π)</td><td>scalar score</td><td>policy-conditioned remaining directional correction</td></tr><tr><td> $H ^ { \mathrm { r e f } } ( g )$ </td><td>scalar score</td><td>room of group g under policy π reference Headroom Headroom  $( g ; \pi _ { \mathrm { g e n } ( g ) } )$ </td></tr><tr><td> $H _ { t } ^ { \mathrm { c u r } } ( g )$ </td><td>scalar score</td><td>conceptual current-policy Headroom Headroom  $( g ; \pi _ { \theta _ { t } } )$ </td></tr><tr><td> $\widehat { H } _ { t } ( g )$ </td><td>scalar score</td><td>cached Headroom priority stored in the buffer at the beginning of step t and used for buffer ordering</td></tr><tr><td> $H _ { t } ^ { \operatorname { r u n } } ( g )$ </td><td>scalar score</td><td>runtime Headroom actually computed for a group scanned under the current policy at step t</td></tr><tr><td> $r _ { t } ( g )$ </td><td>integer</td><td>last refresh step index of the cached Headroom of  ${ \mathit { g } } ,$  so that  $\widehat { H } _ { t } ( g ) \overset { = } { = }$  Headroom  $( g ; \pi _ { \theta _ { r _ { t } ( g ) } } )$ </td></tr><tr><td> $\Delta _ { i , j } ( g ; \theta _ { t } )$ </td><td>scalar</td><td>token-level stored-action log-probability drift under the current policy</td></tr><tr><td> $\operatorname { D r i f t } ( g ; \theta _ { t } )$ </td><td>scalar score</td><td>group-level Policy Drift, i.e., squared-average tokenwise log-probability mismatch relative to</td></tr><tr><td colspan="3"> $\pi _ { \mathrm { g e n } ( g ) }$  Replay-step notation</td></tr><tr><td> $\mathbf { B } _ { t }$ </td><td>ordered queue</td><td>replay buffer at the beginning of step t</td></tr><tr><td> $\mathcal { G } _ { t } ^ { \mathrm { o n } }$ </td><td>set</td><td>freshly generated on-policy groups at step t</td></tr><tr><td> $\dot { \mathcal { T } _ { t } }$ </td><td>set</td><td>ingress candidate set from  $\mathop { \mathcal { G } _ { t } ^ { \mathrm { o n } } }$  determined by the task-specific ingress rule</td></tr><tr><td> $S _ { t }$ </td><td>ordered set</td><td>Headroom-ordered scan prefix actually reevaluated at step t</td></tr><tr><td> $\mathcal { F } _ { t } ( \tau )$ </td><td>set</td><td>Drift-feasible buffered groups satisfying  $\mathrm { D r i f t } ( g ; \theta _ { t } ) \le \tau$ </td></tr><tr><td> $\mathcal { R } _ { t }$ </td><td>set</td><td>accepted replay groups at step t</td></tr><tr><td> $\boldsymbol { A } _ { t }$ </td><td>batch-level</td><td>mixed actor batch combining fresh on-policy groups</td></tr><tr><td></td><td>collection</td><td>and accepted replay groups</td></tr><tr><td> $K _ { \mathrm { r e p } }$ </td><td>scalar</td><td>replay budget per training step</td></tr><tr><td> $\tau$   $C$ </td><td>scalar</td><td>Policy Drift threshold replay-buffer capacity</td></tr><tr><td colspan="3">scalar Lifetime reuse and exposure</td></tr><tr><td> $R _ { t } ( g )$ </td><td>indicator</td><td>equals 1 if group g is replayed at step t, else 0</td></tr><tr><td> $N _ { T } ^ { \mathrm { r e p } } ( g )$ </td><td>scalar</td><td>cumulative replay count of group g up to step T total training uses of group &amp; up to step T, including</td></tr><tr><td> $N _ { T } ^ { \mathrm { { \bar { u s e } } } } ( g )$ </td><td>scalar</td><td>its initial fresh use</td></tr><tr><td> $W _ { T } ( g )$ </td><td>scalar</td><td>cumulative mixed-objective exposure weight of group g up to step  $T$ </td></tr></table>

Table 2: Notation used in Appendix A. The key state distinction is between reference Headroom $H ^ { \mathrm { r e f } } ( g )$ , cached Headroom $\widehat { H } _ { t } ( g )$ used for current-step ordering, conceptual currentpolicy Headroom $H _ { t } ^ { \mathrm { c u r } } ( g )$ , and runtime Headroom $H _ { t } ^ { \operatorname { r u n } } ( g )$ obtained only for groups actually scanned at step t.

Define the fresh and replay parts separately as

$$
\mathcal { L } _ { t } ^ { \mathrm { o n } } ( \theta ) = \frac { 1 } { | \mathcal { G } _ { t } ^ { \mathrm { o n } } | } \sum _ { g \in \mathcal { G } _ { t } ^ { \mathrm { o n } } } \ell _ { \mathrm { G R P O } } \big ( g ; \theta , \pi _ { \mathrm { g e n } ( g ) } \big ) ,\tag{9}
$$

$$
\mathcal { L } _ { t } ^ { \mathrm { r e p } } ( \theta ) = \frac { 1 } { \vert \mathcal { R } _ { t } \vert } \sum _ { g \in \mathcal { R } _ { t } } \ell _ { \mathrm { G R P O } } \big ( g ; \theta , \pi _ { \mathrm { g e n } ( g ) } \big ) \qquad ( \vert \mathcal { R } _ { t } \vert > 0 ) ,\tag{10}
$$

with

$$
\alpha _ { t } : = \frac { | \mathcal { G } _ { t } ^ { \mathrm { o n } } | } { | \mathcal { A } _ { t } | } .\tag{11}
$$

Proposition 1 (Exact mixed-objective decomposition). $\mathrm { I f } \left| \mathcal { R } _ { t } \right| > 0$ , then

$$
\mathcal { L } _ { t } ^ { \mathrm { m i x } } ( \theta ) = \alpha _ { t } \mathcal { L } _ { t } ^ { \mathrm { o n } } ( \theta ) + \left( 1 - \alpha _ { t } \right) \mathcal { L } _ { t } ^ { \mathrm { r e p } } ( \theta ) .\tag{12}
$$

$\mathrm { I f } \left| \mathcal { R } _ { t } \right| = 0 .$ , then ${ \mathcal { L } } _ { t } ^ { \mathrm { m i x } } ( \theta ) = { \mathcal { L } } _ { t } ^ { \mathrm { o n } } ( \theta )$

Proof. The identity follows immediately by splitting the sum in (8) over the disjoint union (6) and normalizing by $| \mathcal { A } _ { t } |$

This decomposition makes explicit that replay augments rather than replaces the fresh onpolicy stream: the fresh rollout path remains intact, while replay contributes an additional constrained term determined by buffer-side selection.

## A.3 Headroom and policy drift

For a stored group g and any policy $\pi ,$ define the token-level Headroom contribution as

$$
h _ { i , j } ( \pi ; g ) = \left\{ \begin{array} { l l } { 1 - \pi ( a _ { i , j } \mid s _ { i , j } ) , } & { A _ { i } > 0 , } \\ { \pi ( a _ { i , j } \mid s _ { i , j } ) , } & { A _ { i } < 0 , } \\ { 0 , } & { A _ { i } = 0 , } \end{array} \right.\tag{13}
$$

and the group-level Headroom as

$$
\mathrm { H e a d r o o m } ( g ; \pi ) = \frac { 1 } { | T ( g ) | } \sum _ { ( i , j ) \in \mathcal { T } ( g ) } h _ { i , j } ( \pi ; g ) .\tag{14}
$$

Thus Headroom is a policy-conditioned measure of remaining directional correction room on the stored action sequence. The reference Headroom $H ^ { \mathrm { r e f } } ( g )$ is obtained by evaluating (14) at the immutable generation-time policy.

To measure current-policy mismatch on the stored actions, define the token-level logprobability drift

$$
\Delta _ { i , j } ( g ; \theta _ { t } ) : = \log \pi _ { \theta _ { t } } ( a _ { i , j } \mid s _ { i , j } ) - \log \pi _ { \mathrm { g e n } ( g ) } ( a _ { i , j } \mid s _ { i , j } ) ,\tag{15}
$$

and the group-level Policy Drift

$$
\mathrm { D r i f t } ( g ; \theta _ { t } ) : = \frac { 1 } { | \mathcal { T } ( g ) | } \sum _ { ( i , j ) \in \mathcal { T } ( g ) } \Delta _ { i , j } ( g ; \theta _ { t } ) ^ { 2 } .\tag{16}
$$

Replay admissibility is controlled by the hard gate Drift $( g ; \theta _ { t } ) \leq \tau$

Remark (Policy Drift as a sampled log-ratio magnitude signal). The term $\Delta _ { i , j } ( g ; \theta _ { t } )$ is the tokenwise sampled log-ratio on the stored action sequence underlying approximate-KL-style diagnostics in GRPO-like training. For replay gating, what matters is mismatch magnitude on the stored sequence rather than a signed average that can cancel across tokens. This is why Policy Drift uses the non-cancelling squared aggregation in (16).

Proposition 2 (Policy-Drift control of Headroom mismatch). For any stored group g and current policy $\pi _ { \theta _ { t } }$

$$
\left| H _ { t } ^ { \mathrm { c u r } } ( g ) - H ^ { \mathrm { r e f } } ( g ) \right| \leq \sqrt { \mathrm { D r i f t } ( g ; \theta _ { t } ) } .\tag{17}
$$

Proof. By the definitions of $H _ { t } ^ { \mathrm { c u r } } ( g )$ and $H ^ { \mathrm { r e f } } ( g )$ 1

$$
\left| H _ { t } ^ { \operatorname { c u r } } ( g ) - H ^ { \operatorname { r e f } } ( g ) \right| \leq { \frac { 1 } { | T ( g ) | } } \sum _ { ( i , j ) \in { \mathcal { T } } ( g ) } \left| h _ { i , j } ( \pi _ { \theta _ { t } } ; g ) - h _ { i , j } ( \pi _ { \operatorname { g e n } ( g ) } ; g ) \right| .
$$

For any token $( i , j )$ , the case definition in (13) implies

$$
\left| h _ { i , j } ( \pi _ { \theta _ { t } } ; g ) - h _ { i , j } ( \pi _ { \mathrm { g e n } ( g ) } ; g ) \right| \le \left| \pi _ { \theta _ { t } } ( a _ { i , j } \mid s _ { i , j } ) - \pi _ { \mathrm { g e n } ( g ) } ( a _ { i , j } \mid s _ { i , j } ) \right| .
$$

Now write

$$
x = \log \pi _ { \theta _ { t } } ( a _ { i , j } \mid s _ { i , j } ) , \qquad y = \log \pi _ { \mathrm { g e n } ( g ) } ( a _ { i , j } \mid s _ { i , j } ) .
$$

Because $x , y \leq 0$ and the derivative of $u \mapsto e ^ { u }$ is at most 1 on $( - \infty , 0 ]$ , the exponential map is 1-Lipschitz on this domain. Hence,

$$
\left| \pi _ { \theta _ { t } } ( a _ { i , j } \mid s _ { i , j } ) - \pi _ { \operatorname { g e n } ( g ) } ( a _ { i , j } \mid s _ { i , j } ) \right| \le \left| x - y \right| = \left| \Delta _ { i , j } ( g ; \theta _ { t } ) \right| .
$$

Substituting this bound and applying Cauchy–Schwarz gives

$$
\begin{array} { r l } & { \Big | H _ { t } ^ { \mathrm { c u r } } ( g ) - H ^ { \mathrm { r e f } } ( g ) \Big | \leq \displaystyle \frac { 1 } { | T ( g ) | } \sum _ { ( i , j ) \in { \mathcal T } ( g ) } \big | \Delta _ { i , j } ( g ; \theta _ { t } ) \big | } \\ & { \qquad \leq \sqrt { \displaystyle \frac { 1 } { | T ( g ) | } \sum _ { ( i , j ) \in { \mathcal T } ( g ) } \Delta _ { i , j } ( g ; \theta _ { t } ) ^ { 2 } } = \sqrt { \mathrm { D r i f t } ( g ; \theta _ { t } ) } . } \end{array}
$$

This proves (17).

Corollary 2.1 (Ranking certificate). Let g and h be two stored groups. If

$$
H ^ { \mathrm { r e f } } ( g ) - H ^ { \mathrm { r e f } } ( h ) > \sqrt { \mathrm { D r i f t } ( g ; \theta _ { t } ) } + \sqrt { \mathrm { D r i f t } ( h ; \theta _ { t } ) } ,\tag{18}
$$

then

$$
H _ { t } ^ { \operatorname { c u r } } ( g ) > H _ { t } ^ { \operatorname { c u r } } ( h ) .\tag{19}
$$

Thus a sufficiently large margin in reference Headroom guarantees preservation of the current-policy Headroom ordering.

Corollary 2.2 (Drift-gated Headroom distortion bound). If a replayed group g passes the Policy Drift gate at step $t , \mathrm { i . e . , i f } g \in \mathcal { R } _ { t } ,$ , then

$$
\left| H _ { t } ^ { \operatorname { c u r } } ( g ) - H ^ { \operatorname { r e f } } ( g ) \right| \leq { \sqrt { \tau } } .\tag{20}
$$

Equivalently,

$$
H _ { t } ^ { \operatorname { c u r } } ( g ) \geq H ^ { \operatorname { r e f } } ( g ) - { \sqrt { \tau } } .\tag{21}
$$

This gives a conservative lower bound on current-policy Headroom for accepted replay groups.

Remark (reference Headroom versus cached and runtime Headroom). Proposition 2 and Corollaries 2.1–2.2 are stated against the immutable reference quantity $H ^ { \mathrm { r e f } } ( g )$ and the conceptual current-policy quantity $H _ { t } ^ { \mathrm { c u r } } ( g )$ because Policy Drift is defined relative to the generation-time policy. The buffer ordering actually used at step t is determined by the cached priority $\widehat { H } _ { t } ( g )$ , which is initialized at $H ^ { \mathrm { r e f } } ( g )$ and refreshed opportunistically only when a group is scanned. For scanned groups, the runtime value equals the conceptual current-policy value by (5). Since current-step selection does not recompute current-policy Headroom for the entire buffer, Proposition 2 should be read as a principled control of reference-to-current mismatch, not as a direct theorem about a uniformly refreshed buffer cache.

## A.4 Replay multiplicity and lifetime exposure

A buffered group can contribute to training more than once: it is used once as a fresh onpolicy group at generation time, and may later re-enter training multiple times via replay. To make this explicit, define the replay indicator

$$
R _ { t } ( g ) : = \mathbf { 1 } [ g \in \mathcal { R } _ { t } ] ,\tag{22}
$$

which equals 1 if group g is replayed at step t and 0 otherwise.

The cumulative replay count of group g up to step T is

$$
N _ { T } ^ { \mathrm { r e p } } ( g ) : = \sum _ { t = t _ { 0 } ( g ) + 1 } ^ { T } R _ { t } ( g ) ,\tag{23}
$$

where the sum starts at $t _ { 0 } ( g ) + 1$ because same-step replay is disallowed. The corresponding total training-use count is

$$
N _ { T } ^ { \mathrm { u s e } } ( g ) : = 1 + N _ { T } ^ { \mathrm { r e p } } ( g ) ,\tag{24}
$$

where the leading 1 accounts for the initial fresh on-policy use at generation time.

To connect replay reuse to the mixed-batch objective, define the cumulative mixedobjective exposure weight

$$
W _ { T } ( g ) : = \sum _ { t \leq T } { \frac { \mathbf { 1 } [ g \in { \mathcal { A } } _ { t } ] } { | { \mathcal { A } } _ { t } | } } .\tag{25}
$$

This quantity records how much objective mass group g receives over time under the stepwise mixed-batch objective.

These definitions make explicit that replay selection induces a structured reuse process over buffered groups. A group’s lifetime replay count and cumulative exposure are jointly shaped by (i) the ingress rule that determines whether it enters the buffer, (ii) finite FIFO residence, (iii) Headroom-ordered prefix truncation during scanning, and (iv) Policy Drift admissibility under the current policy. In later appendices, these quantities are summarized empirically through replay age, replay count, and scan-prefix statistics.

## A.5 Replay-step formalization

Let $\mathcal { T } _ { t } \subseteq \mathcal { G } _ { t } ^ { \mathrm { o n } }$ denote the ingress candidate set produced by the task-appropriate ingress rule at step t. Replay selection operates only on the pre-existing buffer $\mathbf { B } _ { t } \bar { ; }$ current-step ingress is appended only after replay selection and the GRPO update.

Let

$$
\mathbf { C } _ { t } = \operatorname { S o r t } _ { \downarrow \hat { H } _ { t } } ( \mathbf { B } _ { t } ) = ( g _ { t } ^ { ( 1 ) } , g _ { t } ^ { ( 2 ) } , \dotsc , g _ { t } ^ { ( | \mathbf { B } _ { t } | ) } )\tag{26}
$$

be the buffer ordered by descending cached Headroom at the beginning of step t. Define the Drift-feasible buffered subset

$$
\mathcal { F } _ { t } ( \tau ) : = \{ g \in \mathbf { B } _ { t } \ | \ \mathrm { D r i f t } ( g ; \theta _ { t } ) \leq \tau \} .\tag{27}
$$

The accepted replay set is the top- $\cdot K _ { \mathrm { r e p } }$ subset of $\mathcal { F } _ { t } ( \tau )$ under the ordering induced by $\widehat { H } _ { t }$

$$
\mathcal { R } _ { t } : = \mathrm { T o p K } _ { K _ { \mathrm { r e p } } } ^ { \downarrow \widehat { H } _ { t } } \big ( \mathcal { F } _ { t } ( \tau ) \big ) .\tag{28}
$$

By construction,

$$
\left| \mathscr { R } _ { t } \right| \leq K _ { \mathrm { r e p } } , \qquad \forall g \in \mathscr { R } _ { t } , \mathrm { D r i f t } ( g ; \theta _ { t } ) \leq \tau .\tag{29}
$$

Proposition 3 (Equivalent constrained characterization of replay selection). At each training step t, the accepted replay set is equivalently characterized as a solution of

$$
\mathcal { R } _ { t } \in \arg \operatorname* { m a x } _ { \mathcal { S } \subseteq \mathcal { F } _ { t } ( \tau ) , | \mathcal { S } | \leq K _ { \mathrm { r e p } } } \sum _ { g \in \mathcal { S } } \widehat { H } _ { t } ( g ) .\tag{30}
$$

That is, replay selection chooses the cardinality-constrained subset with maximum total cached Headroom among the Drift-feasible candidates.

Proof. Ordering $\mathcal { F } _ { t } ( \tau )$ by descending $\widehat { H } _ { t }$ and taking the first at most $K _ { \mathrm { r e p } }$ elements is exactly the solution of (30).

For the implementation-aware view, define the cumulative admissible count over the ordered buffer prefix as

$$
c _ { t } ( m ) : = \sum _ { u = 1 } ^ { m } \mathbf { 1 } \bigl [ \mathrm { D r i f t } ( g _ { t } ^ { ( u ) } ; \theta _ { t } ) \leq \tau \bigr ] .\tag{31}
$$

Let the stopping index be

$$
\begin{array} { r } { \boldsymbol { m } _ { t } ^ { \star } : = \left\{ \begin{array} { l l } { \operatorname* { m i n } \{ \boldsymbol { m } : c _ { t } ( \boldsymbol { m } ) = K _ { \mathrm { r e p } } \} , } & { \mathrm { i f ~ s u c h ~ a n ~ } m \mathrm { ~ e x i s t s , } } \\ { | \mathbf { B } _ { t } | , } & { \mathrm { o t h e r w i s e . } } \end{array} \right. } \end{array}\tag{32}
$$

Then the actual scan prefix is

$$
S _ { t } : = \{ g _ { t } ^ { ( 1 ) } , \ldots , g _ { t } ^ { ( m _ { t } ^ { \star } ) } \} ,\tag{33}
$$

and the accepted replay set can equivalently be written as

$$
\mathcal { R } _ { t } = \{ g \in S _ { t } : \mathrm { D r i f t } ( g ; \theta _ { t } ) \leq \tau \} .\tag{34}
$$

This makes explicit the Headroom-ordered prefix truncation induced jointly by cached priority ordering and the replay budget $K _ { \mathrm { r e p } }$

For every scanned group $g \in S _ { t }$ , the same current-policy reevaluation pass used to compute $\operatorname { D r i f t } ( g ; \theta _ { t } )$ also yields the runtime Headroom $\dot { H } _ { t } ^ { \mathrm { r u n } } ( \check { g } ) = H _ { t } ^ { \mathrm { c u r } } ( g )$ . The cached priority is then updated after selection according to

$$
\widehat { H } _ { t } ^ { + } ( g ) : = \left\{ \begin{array} { l l } { H _ { t } ^ { \mathrm { r u n } } ( g ) , } & { g \in { \mathcal S } _ { t } , } \\ { \widehat { H } _ { t } ( g ) , } & { g \in { \mathbf B } _ { t } \setminus { \mathcal S } _ { t } . } \end{array} \right.\tag{35}
$$

Thus runtime Headroom affects future candidate ordering through cache maintenance, but it does not retroactively alter the current-step selection order.

The mixed actor batch used by the GRPO optimizer is then

$$
\mathcal { A } _ { t } = \mathcal { G } _ { t } ^ { \mathrm { o n } } \not  \mathcal { R } _ { t } ,\tag{36}
$$

and the step objective is the mixed-batch objective in (8).

Finally, the updated cache state is appended with the current-step ingress groups and maintained under FIFO capacity C:

$$
\mathbf { B } _ { t + 1 } = \mathrm { F I F O A p p e n d } _ { C } \big ( \mathbf { B } _ { t } ^ { + } , \mathrm { O r d } ( \mathcal { T } _ { t } ) \big ) ,\tag{37}
$$

where ${ \bf B } _ { t } ^ { + }$ denotes the ordered buffer with refreshed cached priorities from (35), and each newly appended ingress group $g \in \mathcal { T } _ { t }$ is initialized with cached Headroom

$$
\widehat H _ { t + 1 } ( g ) = H ^ { \mathrm { r e f } } ( g ) .\tag{38}
$$

Because appending happens only after selection, groups generated at step t cannot be replayed in the same step.

## B Detailed benchmark tables

## B.1 Metric definitions

Let each input instance produce n sampled responses with scalar scores in [0, 1].

• Best@n: for each input, take the maximum score among its n samples, then average over inputs.

• Mean@n: for each input, average the n sample scores, then average over inputs.

<table><tr><td rowspan="2">Metric</td><td rowspan="2">GRPO on-pol. matched (b256)</td><td colspan="3">GRPO on-pol.</td></tr><tr><td>larger (b384)</td><td>DAPO</td><td>ExGRPO</td></tr><tr><td>AIME24↑</td><td>0.3083 / 0.0888</td><td>0.2833 / 0.0833</td><td>0.2750 / 0.0966</td><td>0.3417 / 0.0758</td></tr><tr><td>AMC23↑</td><td>0.8000 / 0.4531</td><td></td><td>0.8500 / 0.46330.8500 / 0.5023</td><td>0.8750 / 0.4469</td></tr><tr><td>MATH500↑</td><td>0.8760 / 0.6927</td><td></td><td>0.8800 / 0.6895 0.9100 / 0.6944</td><td>0.9060 / 0.6701</td></tr><tr><td>Minerva↑</td><td>0.3529 / 0.1994</td><td>0.3566 / 0.1957</td><td>0.3824 / 0.1790</td><td>0.3934 / 0.1545</td></tr><tr><td>OlympiadBench↑</td><td>0.4095 / 0.2437</td><td>0.4095 / 0.2403</td><td>0.4347 / 0.2319</td><td>0.4318 / 0.2220</td></tr><tr><td>Macro Avg↑</td><td>0.5494 / 0.3356</td><td></td><td>0.5559 / 0.33440.5704 / 0.3409</td><td>0.5896 / 0.3139</td></tr><tr><td>Weighted Avg↑</td><td>0.5473 / 0.36960.5486 / 0.36640.5722 / 0.3636</td><td></td><td></td><td>0.5772 / 0.3448</td></tr><tr><td>Fresh Responses↓</td><td>1,843,200</td><td>2,764,800</td><td>2,445,312</td><td>1,785,600</td></tr><tr><td colspan="3">GRPO+r matched</td><td>GRPO+r larger</td><td></td></tr><tr><td>Metric</td><td>BAPO</td><td>(b256+r128)</td><td>(b256+r256)</td><td>Headroom-Drift</td></tr><tr><td>AIME24↑</td><td>0.3083 / 0.0891</td><td>0.1417 / 0.0813</td><td>0.1333 / 0.0792</td><td>0.2750 / 0.0943</td></tr><tr><td>AMC23↑</td><td>0.9000 / 0.5117</td><td>0.7000 / 0.5312</td><td>0.6250 / 0.4813</td><td>0.8750 / 0.5117</td></tr><tr><td>MATH500↑</td><td>0.9060 / 0.7026 0.8200 / 0.71650.8160 / 0.7160</td><td></td><td></td><td>0.8820 / 0.7104</td></tr><tr><td>Minerva↑</td><td>0.3971 / 0.1911</td><td></td><td>0.2721 / 0.19760.2721 / 0.1976</td><td>0.3456 / 0.2022</td></tr><tr><td>OlympiadBench↑</td><td>0.4362 / 0.24000.3323 / 0.24590.3234 / 0.2474</td><td></td><td></td><td>0.4050 / 0.2477</td></tr><tr><td>Macro Avg↑</td><td>0.5895 / 0.3469</td><td>0.4166 / 0.3117</td><td>0.3922 / 0.3053</td><td>0.5565 / 0.3533</td></tr><tr><td>Weighted Avg↑ Fresh</td><td>0.5778 / 0.3712</td><td></td><td>0.4525 / 0.35950.4421 / 0.3595</td><td>0.5455 / 0.3792</td></tr><tr><td>Responses↓</td><td>2,027,520</td><td>1,843,200</td><td>1,843,200</td><td>1,843,200</td></tr></table>

Table 3: Held-out benchmark results on mathematical reasoning. Each cell reports Best@32 / Mean@32. Arrows indicate metric direction (↑ higher is better, ↓ lower is better). Top-1 values are bold, and Top-2 values are underlined. GRPO + replay matched (b256+r128) and GRPO + replay larger (b256+r256) denote recency-only replay using the latest 128/256 buffered groups without Headroom ranking or Policy Drift gating.
<table><tr><td></td><td>GRPO on-pol. matched</td><td>GRPO on-pol. larger</td><td></td><td></td></tr><tr><td>Metric</td><td>(b128)</td><td>(b192)</td><td>DAPO</td><td>Headroom-Drift</td></tr><tr><td>Geo3K↑</td><td>0.5870 / 0.4927</td><td>0.5753 / 0.4864</td><td>0.5827  / 0.4952 0.7720 / 0.5860</td><td>0.6245 / 0.5111</td></tr><tr><td>MathVista↑</td><td>0.7600 / 0.5695 0.3727 / 0.1336</td><td>0.7740 / 0.5747 0.3865 / 0.1404</td><td>0.3852 / 0.1357</td><td>0.7720 / 0.5825 0.4086 / 0.1475</td></tr><tr><td>MathVision↑ Macro Avg↑</td><td>0.5732 / 0.3986</td><td>0.5786 / 0.4005</td><td>0.5800 / 0.4056</td><td>0.6017 / 0.4137</td></tr><tr><td>Weighted Avg↑</td><td>0.4839 / 0.2740</td><td>0.4945 / 0.2788</td><td>0.4941 / 0.2793</td><td>0.5148 / 0.2883</td></tr></table>

Table 4: Multimodal benchmark results (simplified 3-benchmark view). Each cell reports Best@32 / Mean@32. Top-1 values are bold, and Top-2 values are underlined.

• Macro Avg Mean@n: unweighted average of benchmark-level Mean@n values across benchmarks.

• Weighted Mean@n: weighted average of benchmark-level Mean@n values, with benchmark weights proportional to benchmark size.

Unless noted otherwise, the paper’s Avg Mean@n notation is identical to Macro Avg Mean@n.

As shown in Table 1, Headroom-Drift achieves a lower per-step time than GRPO on-policy larger (b192) despite including replay-selection overhead, suggesting that in agentic set-

<table><tr><td></td><td>GRPO on-pol. matched (b128)</td><td>GRPO on-pol. larger (b192)</td><td>DAPO</td></tr><tr><td>Metric</td><td>0.4908 / 0.4026</td><td>0.5020 / 0.4152</td><td>0.4548 / 0.3461</td></tr><tr><td>NQ↑ TriviaQA↑</td><td>0.6569 / 0.5711</td><td></td><td>0.6420 / 0.56260.6533 / 0.5488</td></tr><tr><td>PopQA↑</td><td></td><td></td><td>0.4885 / 0.40550.4907 / 0.41960.4786 / 0.3916</td></tr><tr><td>HotpotQA↑</td><td></td><td></td><td>0.4013 / 0.28950.4050 / 0.3017 0.3906 / 0.2684</td></tr><tr><td>2WikiMultiHopQA↑ 0.3913 / 0.2312 0.3702 / 0.2306 0.3877 / 0.2200</td><td></td><td></td><td></td></tr><tr><td>Musique↑</td><td></td><td></td><td>0.1528 / 0.08900.1546 / 0.09090.1483 / 0.0838</td></tr><tr><td>Bamboogle↑</td><td></td><td></td><td>0.3370 / 0.24800.3477 / 0.22800.3136 / 0.1985</td></tr><tr><td>Macro Avg↑</td><td></td><td></td><td>0.4169 / 0.31960.4160 / 0.32120.4038 / 0.2939</td></tr><tr><td>Weighted Avg↑</td><td></td><td></td><td>0.4733 / 0.36740.4669 / 0.37190.4646 / 0.3486</td></tr><tr><td>Fresh Responses↓</td><td>204,800</td><td>307,200</td><td>309,248</td></tr><tr><td></td><td>GRPO+r matched</td><td>GRPO+r larger</td><td></td></tr><tr><td>Metric NQ↑</td><td>(b128+r64)</td><td>(b128+r128)</td><td>Headroom-Drift</td></tr><tr><td>TriviaQA↑</td><td>0.5319 / 0.4496 0.6785 / 0.5984</td><td>0.5507  / 0.4413 0.6856 / 0.5903</td><td>0.5449 /0.4269 0.6940 / 0.5889</td></tr><tr><td>PopQA↑</td><td>0.5114 / 0.4339</td><td>0.5221 / 0.4239</td><td>0.5301 / 0.4375</td></tr><tr><td>HotpotQA↑</td><td>0.4539 / 0.3321</td><td>0.4771 / 0.3407</td><td>0.4721 / 0.3320</td></tr><tr><td>2WikiMultiHopQA↑</td><td>0.4862 / 0.2906</td><td>0.5033 / 0.2906</td><td>0.5115 / 0.3083</td></tr><tr><td>Musique↑</td><td>0.1986 / 0.1132</td><td>0.2247 / 0.1239</td><td>0.2305 / 0.1241</td></tr><tr><td>Bamboogle↑</td><td>0.3760 / 0.2660</td><td>0.4000 / 0.2560</td><td>0.4320 / 0.2860</td></tr><tr><td>Macro Avg↑</td><td>0.4623 / 0.3548</td><td>0.4805 / 0.3524</td><td>0.4879 / 0.3577</td></tr><tr><td>Weighted Avg↑</td><td>0.5201 / 0.4062</td><td>0.5346 / 0.4028</td><td>0.5399 / 0.4084</td></tr><tr><td>Fresh</td><td></td><td></td><td></td></tr><tr><td>Responses↓</td><td>204,800</td><td>204,800</td><td>204,800</td></tr></table>

Table 5: Agentic Search results. Arrows indicate metric direction (↑ higher is better, ↓ lower is better). Top-1 values are bold, and Top-2 values are underlined.
<table><tr><td>Benchmark</td><td>GRPO+r matched Mean@4</td><td>Headroom-Drift Mean@4</td><td>Best@4</td><td>GRPO+r matched Headroom-Drift Best@4</td></tr><tr><td>NQ</td><td>0.4057</td><td>0.4432</td><td>0.5244</td><td>0.5604</td></tr><tr><td>TriviaQA</td><td>0.6107</td><td>0.6228</td><td>0.7108</td><td>0.7172</td></tr><tr><td>PopQA</td><td>0.4069</td><td>0.4292</td><td>0.5002</td><td>0.5253</td></tr><tr><td>HotpotQA</td><td>0.3596</td><td>0.3748</td><td>0.5018</td><td>0.5114</td></tr><tr><td>2WikiMultiHopQA</td><td>0.3394</td><td>0.3378</td><td>0.5789</td><td>0.5597</td></tr><tr><td>Musique</td><td>0.1436</td><td>0.1587</td><td>0.2491</td><td>0.2627</td></tr><tr><td>Bamboogle</td><td>0.3500</td><td>0.4020</td><td>0.5280</td><td>0.5200</td></tr><tr><td>Macro Avg</td><td>0.3737</td><td>0.3955</td><td>0.5133</td><td>0.5224</td></tr><tr><td>Weighted Avg</td><td>0.4158</td><td>0.4298</td><td>0.5556</td><td>0.5638</td></tr></table>

Table 6: Additional 7B Agentic Search scale check on Search-R1 with Qwen2.5-7B-Instruct. Both methods use identical configurations and an 8-bit optimizer, and the controlled comparison is based on one run per method with four samples per input. The better value in each metric pair is bold.

tings where environment interaction dominates rollout cost, replay can reduce wall-clock training time in addition to improving sample quality.

## C Replay ingress in the multi-reward Geo3K setting

## C.1 Motivation and problem setup

In this section, we analyze a multi-reward refinement of the replay-ingress rule in Headroom-Drift Replay, using Geometry3K as a focused case study. As stated in the main method description, the binary-reward setting uses mixed-outcome-only ingress; the present section clarifies how that same ingress principle should be specialized when multiple reward components jointly determine the final reward. The key issue is not datasetspecific: reward heterogeneity can distort the semantics of "success" and therefore distort replay-ingress decisions. To make the scope of this analysis explicit, we frame the central question as follows.

RQ1. Under a multi-reward setting, which groups should be admitted to the replay buffer in order to preserve a useful training signal for subsequent replay learning?

The original replay ingress rule was designed to admit a group into the replay buffer only when its success count was neither zero nor full. The motivation is that if all samples within a group receive the same reward, the group-relative advantage in GRPO becomes zero, so the group provides no useful signal for policy updates. Accordingly, our goal was to prioritize replaying groups that still preserve within-group reward differences and therefore retain a useful training signal.

In binary-reward settings, the original sample-level success indicator $s _ { i } ^ { + } = \mathbf { 1 } [ r _ { i } > 0 ]$ served as an exact indicator of correctness. Under that setting, excluding groups that were allfailure or all-success under $s _ { i } ^ { + }$ functioned as an appropriate replay-ingress criterion. In Geometry3K, however, the final reward combines answer correctness and formatting quality, so $s _ { i } ^ { + }$ no longer corresponds one-to-one to answer correctness. The problem is that the same $s _ { i } ^ { + }$ -based definition used in single-reward settings was applied unchanged to this multi-reward setting.

<table><tr><td>Condition</td><td>Final reward</td></tr><tr><td>wrong answer + format failure</td><td>0.0</td></tr><tr><td>wrong answer + format success</td><td>0.1</td></tr><tr><td>correct answer + format failure</td><td>0.9</td></tr><tr><td>correct answer + format success</td><td>1.0</td></tr></table>

Table 7: Geometry3K reward structure.

As shown in Table $^ { 7 , }$ the final reward in Geometry3K jointly reflects answer correctness and formatting quality. Accordingly, $s _ { i } ^ { + } = \mathbf { 1 } [ r _ { i } > 0 ]$ no longer corresponds exclusively to correctness: an incorrect response may still receive 0.1 for satisfying the format requirement, whereas a correct response may receive 0.9 even when the format requirement is violated. Therefore, carrying over the $s _ { i } ^ { + }$ -based success definition from single-reward settings changes the replay ingress dynamics: the dynamics in a multi-reward setting are no longer the same as those in a single-reward setting.

## C.2 Engineering observation: replay ingress drying

Applying the original $s _ { i } ^ { + }$ -based success definition to Geometry3K leads to a rapid drying of replay ingress. An analysis of the training logs shows that, as training progresses, the number of groups newly entering the replay buffer decreases sharply. Figure 6 summarizes this pattern. The same logs also indicate that, in later stages of training, an increasing number of groups are filtered out because all responses in the group are judged either all unsuccessful or all successful under the $s _ { i } ^ { + }$ -based definition.

This phenomenon did not simply mean that the model had become correct on almost all problems. Figure $7$ shows that, already after the early stage of training, many groups were classified as all-success groups under the original $s _ { i } ^ { + }$ -based definition. Yet these allsuccess groups were usually not groups in which all responses were actually answercorrect. Rather, many of them contained multiple answer-incorrect samples that nevertheless received positive reward from the format component alone.

![](images/b2cb710b00b9e999b5d43e5628d8d8d49126b8b1b125e7f64ae04c0fa5ebf522.jpg)  
Figure 6: Declining replay-buffer ingress over the course of training in Geometry3K. The batch size is 128.

![](images/f176445ef978fb461c1afed95fd8b4229237ebe575d62c18cfcae6d8ca3816b2.jpg)  
Figure 7: Comparison between the number of exact-8 groups in which all responses satisfy the original success indicator $s _ { i } ^ { + } = \mathbf { 1 } [ r _ { i } > 0 ]$ and the number of such groups that contain at least one answer-positive response $( a _ { i } = 1 \iff r _ { i } \geq 0 . 9 )$ . Under the $s _ { i } ^ { + } .$ -based definition, all such groups are treated uniformly as all-success groups, even though they differ in whether they contain answer-correct responses.

Thus, in Geometry3K, the label all-success under $s _ { i } ^ { + }$ no longer uniquely identifies answercorrect groups; the $s _ { i } ^ { + }$ -based definition collapsed the distinction among format-only, answer-only, and fully correct responses by treating all of them uniformly as success. Accordingly, the direct cause of ingress collapse was not that the model had made almost all samples correct, but that the $s _ { i } ^ { + }$ -based definition failed to separate answer and format signals and counted format-only positives as success. As a result, even groups that still contained a meaningful mix of correct and incorrect responses in terms of $a _ { i }$ could be treated as fully positive groups and excluded from replay ingress.

## C.3 Why a correctness-aligned ingress criterion is necessary

## C.3.1 Format-only reward variance

In Geometry3K, variation in the final reward is not sufficient by itself to determine whether a group is useful for replay. Because formatting contributes to the final reward, two responses can receive different rewards even when both are answer-incorrect. The example below illustrates this failure mode.

Example 1. Format-only reward variance without answer-correctness variance   
Case Final reward Answer correctness Format reward   
A 0.0 incorrect fail   
B 0.1 incorrect success   
Original response A (score = 0.0)   
<think>   
Let's call the missing side length "x". Since the sum of angles in a   
triangle is 180 degrees, we can use the angle sum property to find angle x.   
x + 20° + 135° = 180°   
Simplifying the equation, we get:   
x + 155° = 180°   
Subtracting 155° from both sides, we get:   
x = 25°   
Therefore, angle x is equal to 25°.   
\boxed{25°}   
</think>

Example 1 (continued). Original response B (score = 0.1)   
<think>   
To find the value of x, we can use the sine function in a right triangle.   
The sine function relates the angle of a side (adjacent) to the ratio of   
the opposite side to the hypotenuse. Since angle 20° is adjacent to x,   
we can write:   
sin(20°) = (18) / x   
Now, let's solve for x:   
x = (18) / sin(20°)   
x = (18) / (0.3420)   
x 52.90   
</think>   
\boxed{52.90}   
Case A is format-incorrect because the boxed final answer remains inside the <think>   
block, whereas Case B is format-correct because the boxed final answer appears outside   
<think>.

This example highlights the core mismatch. From the perspective of final reward, the group contains non-trivial variation, and such variation would indeed induce a non-zero grouprelative advantage in GRPO. However, that advantage is not necessarily aligned with answer correctness. In this case, the resulting update would favor a format-correct but still answer-incorrect response over a response that is both format-incorrect and answerincorrect. Therefore, a replay criterion based only on final-reward variation can retain groups whose internal preference structure is driven primarily by formatting rather than by answer quality.

The same issue appears at run level. Because GRPO updates the policy from relative reward differences within a group, format-driven variance can be repeatedly reinforced even when answer quality does not improve. Figure 8 shows that, even under the baseline onpolicy setting, format reward rises quickly whereas answer reward improves more gradually. This suggests that formatting can be learned sufficiently from on-policy data, while replay capacity should be reserved more selectively for groups informative about answer correctness.

![](images/c71347cd6777435bd031dd7ef7c8caa41a122871fbfcda67ba0a7bc1152caacf.jpg)  
Figure 8: Format reward saturates much earlier than answer reward in Geometry3K.

Consistent with this interpretation, Figure 9 shows that the original ingress rule can continue to admit groups composed only of 0.0/0.1 rewards. These groups contain no answercorrect samples, and their within-group reward variance is driven entirely by formatting. As a result, replay-buffer capacity can be consumed by groups that do not contribute directly to answer-level improvement.

These observations imply that the main issue is not the existence of a formatting reward, but the ingress criterion itself: the original rule treats answer-driven and format-driven variation as equally replay-worthy. We therefore redesign ingress around the answer-based indicator $a _ { i } \ \stackrel { \scriptscriptstyle } { = } \ { \bf 1 } [ \dot { r } _ { i } \ \stackrel { \scriptscriptstyle } { \ge } \ 0 . \dot { \bf g } ]$ , so that limited replay capacity is preferentially allocated to groups that contain answer-relevant variation. To make this design choice explicit and reproducible, we now formalize the answer-based ingress rule used in Geometry3K.

## C.4 Formal definition of answer-based ingress

The discussion above motivates the need for an answer-based ingress rule, but the term answer-based must be made precise in order to avoid ambiguity. In particular, our analysis involves both sample-level and group-level notions, as well as both the original $s _ { i } ^ { + }$ -based criterion and the revised a -based criterion. We therefore summarize the notation and terminology first, and then define the answer-based ingress rule formally.

Using the notation above, we define answer-based ingress at both the sample and group levels. Let a group consist of G sampled responses, and let $i \in \{ 1 , \ldots , G \}$ index the responses within the group. For each response, we define an answer-based indicator

$$
a _ { i } \in \{ 0 , 1 \} ,
$$

where $a _ { i } = 1$ if the response is correct under the answer criterion, and $a _ { i } = 0$ otherwise.

![](images/5ced37d80bcf168ccef06effba863e656393a6e658ce36af748b3a69bd2c6a75.jpg)  
Figure 9: The count and fraction of groups that can enter the replay buffer under the original ingress rule while containing only 0.0 and 0.1 rewards. Such groups contain no answercorrect samples at all, and their within-group reward variation is driven entirely by formatting rather than answer correctness. $\check { \mathbf { A } } s$ the figure shows, these groups are not confined to the very early stage of training, but continue to appear as ingress candidates well into the middle and later stages.

<table><tr><td>Level</td><td>Symbol / term</td><td>Meaning</td></tr><tr><td>Sample</td><td> $r _ { i }$ </td><td>final reward of response i</td></tr><tr><td>Sample</td><td> $s _ { i } ^ { + }$ </td><td>original success indicator,  $s _ { i } ^ { + } = \mathbf { 1 } [ r _ { i } > 0 ]$ </td></tr><tr><td>Sample</td><td> $a _ { i }$ </td><td>answer-based indicator,  $a _ { i } \doteq \mathbf 1 [ r _ { i } \ge 0 . 9 ]$ </td></tr><tr><td>Group</td><td> $C ^ { + } ( g )$ </td><td>original success count,  $\begin{array} { r } { C ^ { + } ( g ) = \sum _ { i = 1 } ^ { G } s _ { i _ { - } } ^ { + } } \end{array}$ </td></tr><tr><td>Group</td><td> $S ( g )$ </td><td>answer-based success count,  $\textstyle S ( g ) = \sum _ { i = 1 } ^ { G } a _ { i }$ </td></tr><tr><td>Group</td><td>answer-mixed group</td><td>a group satisfying  $1 \leq S ( g ) \leq { \ddot { G } } - 1$ </td></tr></table>

Table 8: Notation and terminology used in the Geometry3K replay-ingress analysis.

For a group $g = \{ a _ { i } \} _ { i = 1 } ^ { G }$ , we then define the answer-based success count as

$$
S ( g ) = \sum _ { i = 1 } ^ { G } a _ { i } .
$$

Under the answer-based ingress rule, a group is admitted to the replay buffer if and only if

$$
1 \leq S ( g ) \leq G - 1 .
$$

That is, replay ingress excludes both answer-all-failure groups $( S ( g ) = 0 )$ and answer-allsuccess groups $( \breve { S } ( g ) = G ) ,$ , and retains only answer-mixed groups in which correct and incorrect responses coexist.

In the Geometry3K experiments, the group size is fixed to $G = 8 .$ The answer-based indicator is instantiated from the task reward structure as

$$
a _ { i } = \mathbf 1 [ r _ { i } \geq 0 . 9 ] ,
$$

where $r _ { i }$ denotes the final reward of response i. Under the reward structure in Table $^ { 7 , }$ this means that responses with rewards 0.9 and 1.0 are treated as answer-correct, whereas responses with rewards 0.0 and 0.1 are treated as answer-incorrect.

This definition makes the role of answer-based ingress precise. It ensures that replay ingress is determined by whether answer correctness actually varies within the group, rather than by whether the final reward varies for any reason. As a result, groups whose internal variation is driven only by formatting do not satisfy the answer-based criterion, whereas groups containing both correct and incorrect responses remain eligible for replay.

## C.5 Empirical effect of answer-based ingress

## C.5.1 Improved training dynamics

![](images/a0a8ee2983243952e83a3894eaf24beb3b1f5510919dea8ba1a2e00d3d92c4ae.jpg)  
Figure 10: Comparison of training-score trajectories under the original ingress rule and the answer-based ingress rule over the shared early-stage window. The top panel shows the raw trajectories (faint) together with their running averages (bold), while the bottom panel shows the score difference between the two runs over the same window.

Figure 10 shows that over the shared early-stage window, the answer-based ingress rule tends to yield higher training scores than the original ingress rule. The upper panel indicates that the smoothed score trajectory remains above that of the original rule for most

of the comparison range, while the lower panel shows a mostly positive gap rather than a single-point spike. These results suggest that aligning replay ingress with answer correctness improves training dynamics in Geometry3K.

## C.5.2 Ingress remains viable

![](images/fbefb760a2169b32b916d888a83276dd370e437b5f1520a4a6f7f414ad8db7c4.jpg)  
Figure 11: Number of groups newly entering the replay buffer under the answer-based ingress rule during the shared early-stage window. Despite the stricter criterion, replay ingress remains sufficiently active and does not collapse prematurely.

A natural concern with the answer-based ingress rule is that the stricter a -based definition may reduce ingress so aggressively that replay no longer operates reliably. Figure 11 shows that this is not what happens. Even in the early stage of training, the number of groups newly entering the replay buffer remains sufficiently large. In other words, strengthening the ingress criterion around answer correctness does not cause replay-buffer supply to dry out prematurely within this window.

These results indicate that answer-based ingress is well motivated in Geometry3K. It is associated with higher training scores while preserving enough replay ingress for replay to remain operational. More broadly, this supports a general design point: when final rewards mix heterogeneous components, replay ingress should be aligned with the target semantic signal (here, answer correctness), not with a coarse positivity test alone.

Answer to RQ1. In multi-reward settings such as Geometry3K, replay ingress should prioritize groups whose within-group variation reflects the target task signal (answer correctness) rather than auxiliary components (e.g., formatting). This answerbased ingress is associated with better training-score dynamics while maintaining sufficient replay ingress.

## D Replay age and reuse statistics

Appendix A introduced replay age, replay indicators, lifetime replay counts, and cumulative mixed-objective exposure as formal quantities associated with buffer-side replay selection. This section summarizes how those quantities behave empirically in training. Our goal here is descriptive: we make visible how Headroom ranking, Policy Drift gating, and replay-budget truncation shape the actual reuse patterns observed in training. Unless otherwise noted, the statistics below are reported for the mathematical-reasoning run.

## D.1 Replay age and drift compatibility

A natural first question is whether absolute replay age alone is sufficient to characterize replay compatibility. To examine this, we compare the age distributions of replay candidates accepted by the Policy Drift gate and those rejected by it. Figure 12 summarizes the resulting age statistics in three complementary views.

Figure 12 shows a clear age gradient in compatibility: older groups are less likely to pass the Policy Drift gate. Accepted groups have a lower mean absolute replay age than rejectedby-drift groups, and the age-conditioned acceptance rate decreases sharply as replay age increases. This is consistent with the intended role of Policy Drift as a current-policy compatibility filter.

At the same time, the figure also shows that age alone is not sufficient to determine compatibility. The accepted and rejected empirical CDFs overlap substantially in the low-age regime, and rejection remains visible even for the youngest replay ages. Thus, while older groups tend to be less reusable, recent groups are not automatically safe. This is precisely why replay compatibility is controlled by Policy Drift rather than by absolute recency alone.

The thin tail at large replay ages is not produced only by the Drift gate; it is also mechanically constrained by finite FIFO residence in the replay buffer. Accordingly, the main message of Figure 12 is not that old groups disappear solely because of gating, but that conditional on being scanned, older groups are less likely to remain compatible, while age alone still does not fully determine acceptance.

## D.2 Headroom-ordered prefix truncation in practice

The replay rule in Appendix A can be viewed as a Headroom-ordered prefix truncation procedure: buffered groups are ordered by cached Headroom, scanned in that order, and admitted only if they remain Drift-compatible, with selection terminating once the replay budget is filled. Figure 13 shows how this mechanism behaves in practice.

Figure 13 shows two complementary properties of the replay-selection mechanism. First, accepted replay is strongly concentrated near the top of the Headroom-ordered buffer. In Panel (B), a relatively small top prefix already accounts for a large share of all accepted replay, indicating that cached Headroom ordering is not merely decorative but materially shapes which groups are reused.

Second, the scan depth needed to satisfy the replay budget grows over training. As shown in Panel (A), early steps can often fill the replay target after scanning only a shallow prefix, whereas later steps increasingly require deep scans and sometimes fail to fill the target at all. This pattern is consistent with a tightening effective admissible set under the Policy Drift gate: as training progresses, fewer buffered groups remain simultaneously highpriority and sufficiently near-policy.

Taken together, the two panels show that replay selection is not well described as either a fixed top-K rule or a uniform scan over the entire buffer. Rather, it is an adaptive prefixtruncation process in which Headroom ordering determines where replay mass is concentrated, while Policy Drift and the replay budget determine how deep the scan must go to recover enough admissible groups.

![](images/33bf90da370d02679ceb0c756c8e80b5594b151598225f8a739e240f9cc6269d.jpg)

![](images/5d0a260aaa572d629c37fa249b6db5fd32b2cb59198a5467c308df7fa84dab05.jpg)

![](images/6811b4bfdd26ddcea14005f82a5a78db6e47bd37e8f76c62bf7cf2fd2551415e.jpg)  
Figure 12: Replay age and Drift-gate compatibility in mathematical-reasoning. (A) Age histogram normalized separately within the accepted and rejected populations. (B) Empirical CDF of replay age for accepted and rejected candidates. (C) Age-conditioned acceptance rate, with the gray bars indicating the number of scanned candidates in each age bucket. Accepted replay is concentrated on younger groups overall, but the two age distributions overlap substantially, and rejection remains non-negligible even at the smallest replay ages. The raw tail counts at large ages are additionally constrained by finite FIFO turnover, so the age-conditioned acceptance rate in Panel (C) is the most direct view of compatibility.

![](images/a34d48197b69bd18524a6e5325625292f067723139b9b4b76fae106518757a9b.jpg)

![](images/b05851f4fdaebcd8c50cd82b3601964ee00585ddf39640d2af0546581533a020.jpg)  
Figure 13: Headroom-ordered prefix truncation in practice. (A) Actual scan depth per training step, measured as the number of buffered candidates examined before replay selection terminates. Blue circles indicate steps that filled the replay target, and orange diamonds indicate underfilled steps. (B) Cumulative share of accepted replay recovered from the top Headroom-ordered buffer prefix. Accepted replay is strongly concentrated near the top prefix of the Headroom-ordered buffer, but later training increasingly requires deeper scans because the Policy Drift gate becomes more selective.

## D.3 Initial stored headroom and lifetime replay pressure

We next ask whether stored Headroom acts as a genuine replay-selection pressure over training, rather than merely serving as a definitional score attached to buffered groups. To test this, we bucket groups by the decile of their initial stored Headroom at buffer entry, then measure how often those groups are replayed over their lifetime.

Figure 14 shows a clear association between initial stored Headroom and subsequent lifetime replay exposure. Low-Headroom deciles exhibit both small mean replay counts and low replay-at-least-once probabilities, whereas the upper deciles show markedly stronger reuse. Thus, the stored Headroom score is not merely a conceptual ranking variable: it induces a real selection pressure on how often groups re-enter training.

![](images/268548aabe1243496e0cc131e6483cd174d3060056512bcb2b248e4cb44b4f5b.jpg)  
Figure 14: Initial stored Headroom and lifetime replay pressure. Bars show the mean lifetime replay count $N _ { \mathrm { r e p } }$ for groups in each initial stored-Headroom decile, and the line shows the probability that a group is replayed at least once. Higher initial stored Headroom is associated with both more frequent reuse and a higher probability of entering replay at least once, indicating a real Headroom-driven selection pressure rather than a purely definitional score.

Importantly, the relationship is strong but not perfectly deterministic. The uppermost deciles are clearly favored, but they do not form a strictly monotone sequence, and some lower-decile groups are still replayed. This is expected: lifetime reuse is shaped not only by Headroom ranking, but also by Policy Drift admissibility, scan-prefix competition, replaybudget truncation, and finite buffer residence. Higher initial stored Headroom is therefore associated with stronger lifetime replay exposure, though it does not fully determine the reuse process alone.

## D.4 Replay multiplicity and exposure concentration

Finally, we examine the overall distribution of replay reuse across buffered groups. The quantities $N _ { T } ^ { \mathrm { r e p } } ( g )$ and $W _ { T } ( g )$ from Appendix A allow us to distinguish two different views of replay concentration: how many groups fall into each replay-count cohort, and how much total replay exposure mass those cohorts account for.

Figure 15 shows that replay is far from uniformly distributed across buffered groups. Most groups are never replayed at all, and the cohort share decreases rapidly as the lifetime replay count grows. In this sense, Headroom-Drift Replay is highly selective at the cohort level: the majority of buffered groups never re-enter training through replay.

However, the exposure-mass view in Panel (B) tells a more informative story. Although $\mathrm { h i g h } { - } N _ { \mathrm { r e p } }$ cohorts are small in population share, they account for a much larger share of total replay-use events. Thus, replay selection is not merely deciding one-shot inclusion versus exclusion; it is redistributing lifetime training exposure toward a comparatively small subset of repeatedly reused groups.

![](images/5a3a6ef8e77d67995745585d4f76f5fd94030694d9d930576f0a79e7efb8d091.jpg)

![](images/f2c5156508650e7468625e4ccce761b7674d19434f1c65255c476794ffaa61f4.jpg)  
Figure 15: Replay multiplicity and exposure concentration. (A) Cohort share by lifetime replay count ${ \mathrm { \hat { N } } } _ { \mathrm { r e p } } ,$ showing what fraction of groups are replayed $0 , 1 , 2 , \ldots$ times. (B) Exposure-mass share by lifetime replay count, showing what fraction of total replay-use events comes from each $N _ { \mathrm { r e p } }$ cohort. Most groups are never replayed, but a much smaller subset of repeatedly reused groups accounts for a disproportionately large share of total replay exposure.

This distinction between cohort share and exposure mass is important for interpreting replay as a training mechanism. A replay rule may appear sparse if viewed only through the fraction of groups that are ever replayed, yet still exert substantial influence on optimization if the repeatedly reused groups receive a large share of total replay exposure. In this sense, Headroom-ranked, Drift-gated replay induces a structured concentration of lifetime exposure rather than a uniform spreading of replay across all buffered groups.

## E Replay dynamics and the delay of entropy collapse

## E.1 Why entropy collapse matters in GRPO

In GRPO-style reinforcement learning, late-stage failures such as entropy collapse, policy contraction, reward hacking, and exploration failure are widely reported. Terminology differs across studies, but the core concern is consistent: even when reward or score improves, the policy distribution can become overly concentrated, making subsequent training brittle. From this perspective, entropy collapse is not merely a benign sharpening of the policy; it is a practical indicator of broader training failure (Miao et al., 2025; Yuan et al., 2026; Xi et al., 2026; Wang et al., 2026b).

Prior work addresses this issue from several complementary directions, including optimizer and clipping design (Yuan et al., 2026; Xi et al., 2026), explicit regularization against reward hacking and late-stage degradation (Miao et al., 2025), and theoretical analyses of entropy dynamics in reinforcement fine-tuning (Wang et al., 2026b). Taken together, these studies establish entropy collapse as an informative signal of broader training failure. Building on this line of work, this appendix treats replay as an operational factor and asks whether replay changes collapse timing itself, rather than merely shifting the learning curve in absolute training steps. In line with Contribution 3, we keep the analysis observational and focus on how replay shapes training dynamics through gate behavior, mismatch, and entropy/performance evolution.

In our experiments, the replay run appeared to enter entropy collapse later, even after reaching a comparable training-score level. However, this pattern alone is insufficient evidence of more stable learning. A run can also appear to collapse later in absolute-step time simply because it learns more slowly and reaches high-training-score regimes later. We therefore reinterpret replay effects under training-score-matched comparison and ask whether replay merely shifts the trajectory in time or meaningfully delays collapse onset.

RQ2. Under training-score-matched comparison, does replay meaningfully delay entropy collapse?

## E.2 Experimental setting and comparison axes

We compare three training configurations: the standard on-policy baseline (GRPO on-policy matched (b256)); Headroom-Drift Replay (b256+r128), which adds Headroom-Drift Replay to the same on-policy setup; and GRPO on-policy larger (b384), which increases the on-policy batch size without replay. All three runs use the same Qwen2.5-Math-1.5B model, GRPO training framework, and dataset. They differ only in replay usage, actor-batch composition, and the number of GRPO mini-batches per update.

<table><tr><td>Setting</td><td>GRPO on-policy matched (b256)</td><td>Headroom-Drift Replay (b256+r128)</td><td>GRPO on-policy larger (b384)</td></tr><tr><td>Replay</td><td>No</td><td>Yes</td><td>No</td></tr><tr><td>On-policy batch size</td><td>256</td><td>256</td><td>384</td></tr><tr><td>Replay target groups</td><td>0</td><td>128</td><td>0</td></tr><tr><td>GRPO mini-batch size</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Number of GRPO mini-batches per update</td><td>4</td><td>6</td><td>6</td></tr></table>

Table 9: Runs compared in the entropy-collapse analysis.

A GRPO mini-batch here denotes one sequential subdivision of the full actor batch within a single update step. Concretely, Headroom-Drift Replay (b256+r128) and GRPO on-policy larger (b384) both use six GRPO mini-batches per update, whereas GRPO on-policy matched (b256) uses four. We therefore treat GRPO on-policy larger (b384) as the stricter on-policy control, because it better matches Headroom-Drift Replay (b256+r128) in per-update optimization load. This comparison helps separate the structural effect of replay from the effect of simply processing more mini-batches.

<table><tr><td>Run</td><td>Entry delay  $\Delta _ { \mathrm { e n t r y } }$ </td><td>Rebound delay  $\underline { { \Delta _ { \mathrm { r e b o u n d } } } }$ </td></tr><tr><td>GRPO on-policy matched (b256)</td><td>318</td><td>121</td></tr><tr><td>Headroom-Drit Replay</td><td>237</td><td>261</td></tr><tr><td>(b256+r128)</td><td>207</td><td></td></tr><tr><td>GRPO on-policy larger (b384)</td><td></td><td>180</td></tr></table>

Table 10: Entry and rebound delays for the three runs under the representative thresholds defined in the text.

## E.3 Empirical characterization of the observed delay

Although the replay run appears to enter entropy collapse later in absolute training progress, this observation alone does not establish collapse-free, stable learning. A run may look delayed simply because it progresses more slowly: if it reaches comparable trainingscore levels later, its transition into low-entropy regions will also appear later on the raw progress axis, even when the underlying collapse dynamics are unchanged. We therefore move beyond raw-progress comparison and use training-score-matched analysis to distinguish two possibilities: replay may merely shift the trajectory in time, or it may delay collapse onset at matched performance levels. We first state the limitation of raw-progress comparison and then summarize the training-score-matched patterns across the three runs.

## E.3.1 From raw-progress comparison to matched-score analysis

To remove learning-speed confounds, we compare runs at matched training-score levels rather than at absolute progress indices. Let S denote the training score and H denote policy entropy. For a training-score anchor $s _ { 0 } ,$ , define the first matching index as

$$
t _ { \mathrm { f i r s t } } ( S \geq s _ { 0 } ) .
$$

We then define entry into and exit from the low-entropy regime as

$$
t _ { \mathrm { e n t r y } } ( H \leq \theta ) , \qquad t _ { \mathrm { r e b o u n d } } ( H \geq \theta ^ { \prime } ) .
$$

To reduce sensitivity to short-lived fluctuations, t<sub>entry</sub> is detected using 10 consecutive log points, and $t _ { \mathrm { r e b o u n d } }$ using 5 consecutive log points.

Using these events, we define two training-score-conditioned delays:

$$
\Delta _ { \mathrm { e n t r y } } ( \theta \mid s _ { 0 } ) = t _ { \mathrm { e n t r y } } ( H \leq \theta ) - t _ { \mathrm { f i r s t } } ( S \geq s _ { 0 } ) ,
$$

$$
\Delta _ { \mathrm { r e b o u n d } } ( \theta , \theta ^ { \prime } \mid s _ { 0 } ) = t _ { \mathrm { r e b o u n d } } ( H \geq \theta ^ { \prime } ) - t _ { \mathrm { e n t r y } } ( H \leq \theta ) .
$$

$\Delta _ { \mathrm { e n t r y } }$ measures how long a run takes to enter the low-entropy regime after reaching the shared training-score anchor, and $\Delta _ { \mathrm { r e b o u n d } }$ measures how long it remains in that regime before rebound.

## E.3.2 Observed patterns across the three runs

In this analysis, we use the representative score anchor $S \geq 0 . 4 0 _ { \cdot }$ , the low-entropy plateau entry threshold $H _ { L } = 0 . 0 3$ , and the exit threshold $H _ { U } = 0 . 0 4$ . Table 10 summarizes the score-conditioned delays of the three runs under these criteria, and Figure 16 visualizes the corresponding entropy trajectories and marker timings.

Figure 16 shows that Headroom-Drift Replay (b256+r128) reaches the shared score anchor $S  \geq 0 . 4 0$ earlier than GRPO on-policy larger (b384). The replay run therefore does not lag in entering the high-score regime. Yet the same trajectory indicates later entry into the low-entropy regime, suggesting that score improvement and entropy compression are not strictly synchronized.

![](images/f314cc0ddbda386fbd264b16b7062eec9b48d642effd50128ffc7f4e1c514944.jpg)  
Figure 16: Entropy trajectories for the three runs. The gray GRPO on-policy matched (b256) is shown only as a reference. Diamonds denote first arrival at the shared training-score anchor, circles denote entry into the low-entropy regime, squares denote rebound, and the highlighted segments indicate the retained low-entropy window.

The matched-score comparison reinforces this point and extends beyond entry timing. Under $H _ { L } = 0 . 0 3$ GRPO on-policy larger (b384) shows a shorter post-anchor entry delay than Headroom-Drift Replay (b256+r128) (207 vs. 237 in Table 10). Under ${ \cal H } _ { U } = 0 . 0 \dot { 4 } .$ , GRPO onpolicy larger (b384) also rebounds sooner, while Headroom-Drift Replay (b256+r128) sustains the low-entropy regime longer. In this run pair, replay reaches the shared high-score regime earlier, but enters the low-entropy regime later and remains there longer.

Taken together, these observations indicate that the delay cannot be explained only by slower learning in the replay run. At matched score levels, both low-entropy entry and rebound are delayed under replay. The observed pattern is therefore more consistent with a change in post-anchor entropy dynamics than with a simple shift in optimization speed.

## E.4 Interpretation of the observed delay

The previous subsection suggests that the delayed collapse in the replay run is not simply a by-product of slower optimization; entropy dynamics differ even after runs reach comparable score levels. This subsection develops a working interpretation of that difference and then examines why the effect weakens in the late stage.

## E.4.1 Replay as a near-policy corrective mechanism

As shown above in Figure 16 and Table 10, the replay run reaches the shared score anchor earlier, yet enters and exits the low-entropy regime later. This pattern makes a purely slower-learning explanation unlikely and motivates a working interpretation.

Figure 5(c) provides the corresponding conceptual pattern: replay under Headroom-Drift remains concentrated on recent sources, but is not reducible to trivial previous-step reuse.

Table $1 1 ( \mathrm { A } ) ^ { 1 }$ , together with the conceptual view in Figure 5(c), indicates that replay in this run is neither trivial one-step reuse nor indiscriminate long-range reuse. The replay groups actually used for updates span source ages from 1 to 7 steps; 66.00% come from sources at least 2 steps old, and 44.06% from sources at least 3 steps old. At the same time, the used subset is more recent than the replay pool as a whole (mean source staleness: 2.67 vs. 3.32 steps), indicating a clear recent-source preference rather than uniform reuse across the full buffer. Importantly, this recency bias does not collapse to a single age: Table 11(B) shows a mixed-age structure (mean distinct source ages per update: 4.07; updates with at least three ages: 85.71%; updates containing both age = 1 and age ≥ 3: 76.79%).

<table><tr><td colspan="8">(A) Source staleness summary</td></tr><tr><td>Population</td><td>Mean</td><td>Median</td><td>P90</td><td>Max</td><td>% age ≥ 2</td><td>% age ≥ 3</td></tr><tr><td>Replay pool</td><td>3.32</td><td>3</td><td>6</td><td>8</td><td>73.51</td><td>60.22</td></tr><tr><td>Used replay subset</td><td>2.67</td><td>2</td><td>5</td><td>7</td><td>66.00</td><td>44.06</td></tr></table>

(B) Mixing structure of the used replay subset
<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Mean distinct source ages per update</td><td>4.07</td></tr><tr><td>Fraction of updates with at least 3 source ages</td><td>85.71%</td></tr><tr><td>Fraction of updates with both age = 1 and  $a g e \ge 3$ </td><td>76.79%</td></tr></table>

Table 11: True source staleness of replay groups. Panel (A) shows that the replay groups actually used for updates are more concentrated on recent source samples than the replay pool as a whole, while still spanning a non-trivial short horizon. Panel (B) shows that the used replay subset typically mixes multiple source ages within the same update.

These statistics are consistent with a near-policy corrective interpretation. Replay mostly reuses recent samples but still incorporates selected older ones, so each update reflects a short-horizon multi-age mixture rather than only the immediately preceding rollout. Relative to pure on-policy learning, replay therefore introduces trajectories generated at nearby policy states that remain useful under current criteria. Within this process, Headroom prioritizes higher-value groups in the replay pool, while the Policy Drift gate filters out trajectories that have drifted too far from the current policy. Under this interpretation, replay is less a pull toward stale policies than a constrained reintroduction of still-compatible trajectories from nearby policy snapshots, which can moderate one-step acceleration while retaining informative signal.

## E.4.2 Late-stage transition and failure mode

<table><tr><td>Metric</td><td>Early learning</td><td>Score-rise</td><td>Pre-collapse</td><td>Collapse</td></tr><tr><td>Mean source age</td><td>2.77</td><td>3.26</td><td>2.24</td><td>1.26</td></tr><tr><td>% age = 1</td><td>21.15</td><td>22.90</td><td>42.13</td><td>76.47</td></tr><tr><td>% age ≥ 3</td><td>48.08</td><td>59.35</td><td>33.50</td><td>2.94</td></tr><tr><td>Mean distinct ages / update</td><td>4.33</td><td>4.83</td><td>3.80</td><td>1.67</td></tr><tr><td>% updates with ≥ 3 ages % updates with age = 1</td><td>100.00</td><td>100.00</td><td>85.00</td><td>16.67</td></tr><tr><td>and age ≥ 3</td><td>66.67</td><td>87.50</td><td>85.00</td><td>16.67</td></tr></table>

Table 12: Phase-wise contraction of the used replay mixture. The replay subset is most diverse during the score-rise regime and collapses toward a narrow age-1/2-dominated mixture near the failure phase.

Figure 17 shows that the late-stage transition is not well explained by a simple ingress collapse. The number of newly ingressed groups does not drop sharply, even in the late stage. By contrast, the amount of replay actually used in updates falls markedly in the same region, and steps that fail to fill the replay target become frequent near collapse. The transition is therefore better interpreted as a rapid contraction of the subset that remains reusable under current criteria.

This interpretation is reinforced by the phase-wise mixture changes in Table 12. During the score-rise phase, the used replay subset has mean source age 3.26, share $a g e \ge 3$ of

![](images/1edfbb844f11764919c01ee2513b42128222a3a3caf859e39b2a2e6aebf595e1.jpg)  
Figure 17: Replay ingress and actual replay use over training. The gray curve shows the number of newly ingressed groups added to the replay buffer at each step, while the orange curve shows the number of replay groups actually used for the update. Although ingress remains substantial into the late stage, actual replay use drops sharply near collapse, suggesting that the late-stage transition is better understood as a contraction of the usable replay subset rather than as a simple ingress collapse.

59.35%, mean distinct ages per update of 4.83, and 100% of updates with at least three ages, indicating a rich multi-age mixture during rapid improvement. This structure weakens in pre-collapse and then contracts sharply in collapse: mean source age falls to 2.24 and then 1.26, the age = 1 share rises to 42.13% and then 76.47%, the age ≥ 3 share drops to 33.50% and then 2.94%, and mean distinct ages decrease to 3.80 and then 1.67. The late-stage issue is therefore not only reduced replay volume; the corrective mixture itself narrows from a multi-age structure to a highly concentrated age range.

This transition also co-occurs with larger policy movement in the late stage. As shown in Supplementary Figure 18, on-policy GRPO-KL and gradient norm rise together during collapse. At the same time, the main logs show a sharp drop in Drift-gate acceptance rate relative to pre-collapse, along with repeated steps that fail to fill the replay target. This combination is consistent with increasing mismatch between the current policy and replayed trajectories, which shrinks the replay groups admissible under the Policy Drift threshold.

Taken together, the late-stage failure in this run is consistent with a phase in which the nearpolicy corrective mechanism can no longer be sustained in sufficiently rich form. In the early-to-middle stage, replay appears to combine useful trajectories from multiple recent policy steps, coinciding with both score improvement and delayed entropy collapse. In the late stage, replay ingress remains present, but the used replay subset drops sharply and its internal source-age mixture contracts into a narrow age-1/2-dominant range. We therefore interpret the transition as being linked less to replay presence per se and more to the duration over which a usable corrective mixture can be maintained.

![](images/0e971496d0aa680c2a30337d91dac616040d06e52de5cd566141e5f5ed15b90a.jpg)  
Figure 18: Supplementary policy-drift proxies over training. The upper panel shows onpolicy $\mathrm { { G R P O - K L } } ,$ and the lower panel shows gradient norm on a log scale. Both quantities rise sharply in the late stage, consistent with a phase in which the current policy begins to move more abruptly while the Drift-gate-admissible replay subset simultaneously contracts.

## F Implementation details for headroom-drift replay

This appendix provides implementation-level detail for the replay-augmented training procedure described in Algorithm 1. The main text focuses on the conceptual design of replay selection (Headroom ranking, Policy Drift gating, FIFO eviction); this section supplements it with the engineering choices required to run the system stably in practice. Algorithm 2 expands each step of Algorithm 1 to this level, with concrete instantiations from the mathematical reasoning setting $( | \widetilde { \mathbf { R } } _ { t } | = 1 2 8$ replay groups, $| \mathcal { G } _ { t } ^ { \mathrm { o n } } | { = } 2 5 6$ on-policy groups, mini-batch size 64, 8 GPUs).

```latex
Algorithm 2 Two-column implementation view of Algorithm 1. Left: pseudocode. Right:
concrete instantiation in the mathematical reasoning setting.
Left: General Procedure (Pseudocode) Right: Concrete Instantiation (Math
Setting)
A. Rollout, Reward, Advantage A/C. Shared with Alg. 1
Same as Algorithm 1. Rollout, reward, advantage, ingress,
and retrieval are unchanged.
B. Frozen Reference State B. Frozen references
On-policy groups: compute old log-probs via forward Replay groups: 128 groups reuse
pass. stored log $\pi _ { \mathrm { g e n } ( g ) }$ and ${ \breve { A } } _ { i }$ (no recom-
Replay groups: reuse stored log $\pi _ { \mathrm { g e n } ( g ) } .$ putation).
Replay advantages: reuse stored ${ \bar { A } } _ { i } .$
D. Selection with Early Termination D. Selection
Sort buffer by cached Headroom (descending). Stop when $K _ { \mathrm { r e p } } { = } 1 2 8$ is filled (early
Initialize selected $ \varnothing .$ termination).
for each candidate $g$ in sorted order do Use one parallelized forward pass per
if |selected| $\geq K _ { \mathrm { r e p } } { \mathrm { ~ } }$ then break scanned group to compute both Pol-
Compute log $\pi _ { \theta _ { t } }$ for $g .$ icy Drift and refreshed cached Head-
Compute Drift $\left( g ; \theta _ { t } \right)$ using current and stored log-probs. room.
Refresh cached Headroom from the same current-policy
pass.
if Drift $( g ; \theta _ { t } ) \le \tau$ append $g ,$ else reject $g$ (cached Head
room still refreshed).
end for
E. Batch Construction E. Batch construction
$\widetilde { \mathbf { R } } _ { t } \gets \mathrm { s e l e c t e d } \big [ 1 : u \big \lfloor m / u \big \rfloor \big ] .$ u=1 (no truncation).
actor_batch $ \big [ \widetilde { \mathbf { R } } _ { t } : \mathcal { G } _ { t } ^ { \mathrm { o n } } \big ] .$ [128 replay; 256 on-policy] = 384 total
Split into k mini-batches (preserving order). groups.
k=6 mini-batches: MB 1–2 replay, MB
3–6 on-policy.
F. Sequential PPO Update + Eviction F. PPO update
for $i { \dot { = } } 1 , \dots , k$ do Compute numerator log $\pi _ { \theta }$ only, then
Partition MB<sub>i</sub> across GPUs with length balancing. clipped PPO update; ingress append +
Compute log π on stored trajectories. FIFO eviction after update.
Compute $r \bar { = } \pi _ { \theta } / \pi _ { \mathrm { g e n } ( g ) } ,$ clipped PPO objective, and up
date θ.
end for
Append ingress groups and apply FIFO eviction.
```

## F.1 Frozen reference state

When a replay group is admitted to the buffer, its generation-time per-token logprobabilities log $\breve { \pi } _ { \mathrm { g e n } ( g ) } ( a _ { i , j } ~ \mid ~ s _ { i , j } )$ and response-level advantages $A _ { i }$ are stored as immutable state (Algorithm 2, Frozen Reference State). The importance-ratio denominator is therefore always anchored at the generation-time policy $\pi _ { \mathrm { g e n } ( g ) } ;$ only the numerator is recomputed under the current policy $\pi _ { \theta _ { t } }$ at each PPO update. Advantages are likewise reused without recomputation. As described in Section $3 . { \dot { 3 } } ,$ the Policy Drift gate measures how far this frozen reference state has drifted from the current policy and blocks groups that exceed threshold τ.

Replay-buffer storage. In the mathematical-reasoning configuration, the CPU-resident FIFO buffer holds at most 512 query groups (8,192 responses). At the configured maximum sequence lengths, the stored token sequences, generation-time log-probabilities, and GRPO advantages total approximately 448 MiB at full capacity.

## F.2 Selection with early termination and cached-headroom refresh

Replay selection scans buffer candidates in descending cached-Headroom order and accepts groups satisfying $\mathrm { D r i f t } ( g ; \theta _ { t } ) ~ \le ~ \tau$ until the replay budget $K _ { \mathrm { r e p } }$ is filled, at which point the scan terminates immediately (Algorithm 2, Selection with Early Termination). This early termination ensures that selection cost scales with $K _ { \mathrm { r e p } }$ and the number of candidates actually scanned, rather than with the full buffer size. When the budget cannot be filled—for instance, when τ is tight and many candidates are rejected—the accepted count is used as-is.

For each scanned candidate, the current-policy forward pass—which is parallelized over stored trajectories—simultaneously computes Policy Drift and the runtime Headroom needed to refresh cached Headroom (Section 3.3). This cached-Headroom refresh is applied to every scanned group regardless of whether it passes or fails the Policy Drift gate, and requires no additional forward pass. As a result, the selection procedure performs Policy Drift gating and cached-Headroom refresh in a single same-pass reevaluation.

## F.3 Replay-prefix layout

When forming the mixed actor batch $\boldsymbol { A } _ { t } ,$ replay groups are placed before on-policy groups (Algorithm 2, Batch Construction). Because the actor batch is split into mini-batches by sequential slicing, this prefix layout causes replay groups to occupy the earlier mini-batches and on-policy groups to occupy the later ones. In the mathematical reasoning setting, for example, the six mini-batches are partitioned as MB 1–2 (replay) and MB 3–6 (on-policy). Since PPO processes mini-batches sequentially and updates the policy parameters after each one, the relatively more off-policy replay samples are consumed first within each training step.

## F.4 Batch alignment

Micro-batch truncation. The number of accepted replay groups may not be a multiple of the micro-batch processing unit u. To avoid misalignment during gradient accumulation, the accepted sequence is truncated to the largest multiple of u that does not exceed the accepted count (Algorithm 2, Batch Construction). Formally, given the accepted replay sequence $\mathbf { R } _ { t } = \left( g _ { 1 } , \ldots , g _ { m } \right)$ , the groups actually used for training are

$$
\widetilde { \mathbf { R } } _ { t } = \big ( g _ { 1 } , \ldots , g _ { u \lfloor m / u \rfloor } \big ) .
$$

Because cached-Headroom ordering is preserved, only the lowest-priority tail is removed.   
The number of discarded groups is at most u − 1; when u = 1, no truncation occurs.

Sequence-length balanced partitioning. Each mini-batch is distributed across GPUs for data-parallel processing (Algorithm 2, Sequential PPO Update). A naive sequential partition can concentrate long sequences on a single GPU, leaving other GPUs idle while that GPU finishes. This imbalance is amplified in the mixed batch because replay groups and on-policy groups are generated at different training steps and may have different responselength distributions. To mitigate this, each mini-batch is repartitioned so that the total sequence length per GPU is approximately balanced. In practice, this reduces the maximumminus-minimum sequence-length gap across GPUs from approximately 73K tokens to under 5K tokens. This repartitioning does not change which samples are trained on or the objective used; it affects only the mapping of samples to GPUs within each mini-batch, reducing idle time caused by uneven workloads.

## G Ablation studies

## G.1 L1-style versus L2-style policy drift: formal interpretation and empirical comparison

The main text adopts a squared (L2-style) aggregation for Policy Drift and notes that an L1 variant yields a tighter formal bound but underperforms empirically (§3.3). This appendix develops both sides of that claim: we first show that the two gates define qualitatively different compatibility regions, and then compare them empirically on mathematical reasoning.

## G.1.1 Formal interpretation: variance-sensitive gating

The main method defines Policy Drift as

$$
\mathrm { D r i f t } _ { L 2 } ( g ; \theta _ { t } ) : = \frac { 1 } { | \mathcal { T } ( g ) | } \sum _ { ( i , j ) \in \mathcal { T } ( g ) } \Delta _ { i , j } ( g ; \theta _ { t } ) ^ { 2 } .
$$

The corresponding L1-style alternative replaces the squared terms with absolute values:

$$
\mathrm { D r i f t } _ { L 1 } ( g ; \theta _ { t } ) : = \frac { 1 } { | \mathcal { T } ( g ) | } \sum _ { ( i , j ) \in \mathcal { T } ( g ) } \left| \Delta _ { i , j } ( g ; \theta _ { t } ) \right| .
$$

A natural question is whether the squared gate is simply a stricter version of the absolute gate. It is not. The squared gate is better understood as a variance-sensitive compatibility score.

Proposition (mean–variance decomposition). Let $\mu _ { g }$ denote the mean absolute tokenwise drift and $v _ { g }$ the within-group variance of absolute tokenwise drift. Then

$$
\mathrm { D r i f t } _ { L 2 } ( g ; \theta _ { t } ) = \mu _ { g } ^ { 2 } + v _ { g } .
$$

Proof. Immediate from the standard second-moment identity $\mathbb { E } [ X ^ { 2 } ] = ( \mathbb { E } [ X ] ) ^ { 2 } + \operatorname { V a r } ( X )$

Corollary (acceptance-region geometry). An L1 gate with threshold $\tau _ { 1 }$ accepts a group whenever $\mu _ { g } \le \tau _ { 1 }$ . The L2 gate with threshold $\tau _ { 2 }$ accepts whenever $\mu _ { g } ^ { 2 } + v _ { g } \leq \tau _ { 2 }$ . The L2 gate is therefore not uniformly stricter: it may reject a group with small mean drift if that drift is concentrated on a few tokens, while accepting another group with larger mean drift if the tokenwise mismatch is diffuse.

Illustrative example. Consider two four-token groups with identical L1 drift:

$$
( 0 . 0 2 , \ 0 . 0 2 , \ 0 . 0 2 , \ 0 . 0 2 ) \quad { \mathrm { v s . } } \quad ( 0 . 0 8 , \ 0 , \ 0 , \ 0 ) .
$$

Both have Drif $\mathrm { t } _ { L 1 } = 0 . 0 2$ , but $\mathrm { D r i f t } _ { L 2 } = 0 . 0 0 0 4$ for the uniform case and 0.0016 for the spike case—a fourfold difference invisible to the L1 gate.

In our experiments, the L1 threshold is $\tau _ { 1 } = 0 . 0 0 5$ and the L2 threshold is $\tau _ { 2 } = 0 . 0 0 1$ Because the L2 gate operates on a second moment, its threshold is more naturally compared on the root-mean-square scale: $\sqrt { \tau _ { 2 } } \approx 0 . 0 3 2$ . The two gates therefore define different compatibility regions rather than tighter or looser versions of the same region.

## G.1.2 Empirical comparison on mathematical reasoning

We compare L1-style and L2-style Drift gating under otherwise identical training configurations on mathematical reasoning. Figure 19 summarizes step-aligned trajectories over 500 steps. Figures 20–21 characterize the replay groups selected by the L2 gate.

![](images/8ae69227cb7d2bde6eb83369a1ca57dc0c3ec632c96e71fe24ee892f91bf68ef.jpg)

![](images/cb4ef3e26bfa2aa905e07d899a9d4de48a0255c40aa719f0a7944185c77dbba0.jpg)

![](images/11e7747b2e2b5b2e4bb87c9643c373cfc121f74edd1528ddb4d9c3f9a5f73be2.jpg)

![](images/0f2bfda4faeb3b8d0e81cf424e60521a8d19f1ee8f821b090a13189323a7d5f8.jpg)  
Figure 19: Step-aligned comparison of L1-style and L2-style Drift gating. L2 attains better late-stage validation accuracy while maintaining lower replay mismatch, despite matched or lower replay-budget fulfillment.

L2 wins through selection quality, not replay volume. The two gates achieve similar replay-budget fulfillment through the middle phase of training. At step 350, both fill the full budget, yet L2 already yields higher validation accuracy (Mean@4: 0.3636 vs. 0.3565) with an order-of-magnitude lower replay KL $( 8 . 5 \times 1 0 ^ { - 5 }$ vs. $5 . 9 \times 1 0 ^ { - 4 } )$ . The gap widens in the late phase. Averaged over steps 350–500, L2 achieves higher Mean@4 (0.3613 vs. 0.3430) despite lower average budget fulfillment (0.81 vs. 0.88), while maintaining replay KL roughly nine times smaller $( 7 . 0 \times 1 0 ^ { - 5 }$ vs. $6 . 3 \times 1 0 ^ { - 4 } )$ . L2 does not use more replay; it uses better-compatible replay.

Selection dynamics within the L2 gate. Figure 20 shows how the L2-selected replay subset evolves. Early in training (step 50), the selected groups have much lower success rate (0.080 vs. 0.423 in the candidate pool) and higher Headroom (0.863 vs. 0.592), indicating strong concentration on hard, high-value cases. This concentration weakens as training progresses: by step 350, the selected subset approaches the pool boundary (success rate 0.390 vs. 0.474 and Headroom 0.643 vs. 0.555), and by step 500 the gap has largely closed. Once the high-value frontier is depleted, the L2 gate contracts replay usage rather than forcing the budget full with less compatible groups.

Figure 21 visualizes this pattern as snapshots. At steps 50 and 200, selected groups cluster tightly in the low-success, high-Headroom corner. By step 350, coverage broadens as the high-value region thins. By step 500, selection spans a wider range of success rates, reflecting the depleted frontier. The L2 gate adapts its effective selectivity to the available pool rather than maintaining a fixed selection profile.

![](images/8c7fdc565ba950bd0bf5a3d77bb485b1567b3a7083e9ffed9c1afdb92a3803dd.jpg)

![](images/539710a8f68896d13cf952fcbfd2894828cf61a8ef9fa5af116ac00ed9ff714e.jpg)

![](images/78556d8255115af3bf385aaae307403f7bb520812eebab2572bb6144a4499b6c.jpg)

D  
![](images/ceeda5614e1943ab2d0d7e626f3bb6f8fba3c6282af713c311539e69d585e4c8.jpg)  
Figure 20: Aggregate properties of the replay groups selected by L2-style Drift gating. Early training concentrates replay on hard, high-Headroom groups; later training moves toward the remaining frontier, and replay usage contracts as that frontier is depleted.

![](images/eb81a40611aa4f72dd5aab22972cbed315afdc9f29c7379264023c1692035648.jpg)

B  
![](images/234dadb4d1481ed2284b8a536b46e7b34f39f117fda197d34c633abd680dbc3a.jpg)

![](images/f65b8f8c3e2fc81fad207bd238872ea8359eaa0da28f4205bd06423cf591b14d.jpg)

D  
![](images/d3a0929ddeeb244d2166a10c2f832ee8499ed4d69e92157b6faffcaee9d18303.jpg)  
Figure 21: Selection snapshots in the success-rate / Headroom plane at four training stages. Selected groups (orange) concentrate near the upper-left frontier of the candidate cloud (gray), particularly in early and middle training.

## G.2 Practical calibration and sensitivity of the Policy Drift threshold τ

We use τ to place the Policy Drift gate in a viable operating regime, not to locate a narrow performance optimum. Table 13 summarizes the thresholds used across the evaluated runs. The values fall into two task-family settings: the single-turn reasoning runs and the CISPOstyle portability study share one value, while the 3B and 7B Search-R1 runs share another. In particular, the 7B Search-R1 run reuses the 3B threshold without scale-specific retuning.
<table><tr><td>Evaluated setting</td><td>Scope</td><td>T</td><td>Reuse note</td></tr><tr><td>Mathematical reasoning</td><td>Single-turn reasoning</td><td> $1 0 ^ { - 3 }$ </td><td>Shared single-turn setting</td></tr><tr><td>Multimodal reasoning</td><td>Single-turn reasoning</td><td> $1 0 ^ { - 3 }$ </td><td>Shared single-turn setting</td></tr><tr><td>CISPO-style portability</td><td></td><td></td><td>Objective portability 10-3 Replay hyperparameters unchanged</td></tr><tr><td>Search-R1 (3B)</td><td>Agentic Search</td><td>10⁻²</td><td>Agentic setting</td></tr><tr><td>Search-R1 (7B)</td><td></td><td></td><td>Agentic scale check 10-2 Reused without scale-specific retuning</td></tr></table>

Table 13: Drift thresholds used across the evaluated settings. The values reduce to two taskfamily settings, and the 7B Search-R1 run reuses the 3B threshold without scale-specific retuning.

We then use a controlled Search-R1 sweep to examine how a viable regime can be identified without a dense full-training search. Agentic Search is a useful testbed because its multiturn interaction with an external retrieval environment makes replay valuable while also exposing training instability more clearly than the single-turn settings (Wang et al., 2026a; Liu et al., 2026a;b).

RQ. How can a useful operating regime for τ be identified without an exhaustive full-training sweep, and what diagnostic signatures appear when the gate is too tight or too loose?

## G.2.1 Experimental setup

Experiments are conducted on Agentic Search (Qwen2.5-3B-Instruct, 8×H100). The three decade-spaced conditions are designed to probe distinct gate states rather than locate a narrow performance optimum. All other training settings—including batch size, learning rate, replay buffer size, n, and K<sub>rep</sub>—are held fixed.

## G.2.2 Gate behavior and calibration diagnostics

We interpret replay acceptance and replay KL jointly: acceptance measures whether the gate supplies enough replay, while replay KL measures whether the admitted groups remain compatible with the current policy. Table 14 summarizes the three conditions.

<table><tr><td></td><td colspan="4">Gate rejections</td></tr><tr><td>Condition</td><td>T</td><td>Mean acceptance</td><td>per update</td><td>Replay-KL behavior</td></tr><tr><td>TIGHT</td><td>0.001</td><td>6.4%</td><td>450.3</td><td>0.0005 → 0.00003 as replay vanishes</td></tr><tr><td>MODERATE</td><td>0.01</td><td>40.2%</td><td>119.4</td><td>Stable at approximately 0.002</td></tr><tr><td>LOOSE</td><td>0.1</td><td>94.4%</td><td>1.9</td><td>0.0045 → 0.0170</td></tr></table>

Table 14: Diagnostic summary of the Search-R1 threshold sweep. Acceptance rate and gate rejections per update are training averages; replay-KL entries summarize the early-to-final trajectory.

The three settings produce qualitatively different gate states. Under TIGHT, only 6.4% of candidates are accepted, effectively starving replay. Its low replay KL does not indicate better compatibility; it results from admitting very few replay groups. Under LOOSE, 94.4% of candidates pass, leaving the gate nearly inactive, and replay KL rises over training. MOD-ERATE instead admits replay selectively while keeping replay KL stable at approximately 0.002. Neither diagnostic is sufficient alone: a useful threshold must preserve replay supply and control policy mismatch at the same time.

## G.2.3 Accumulation of distribution mismatch

The clearest difference across τ settings appears in replay KL. Replay KL is computed at each training step over the replay samples and measures how far the current policy $\pi _ { \theta }$ has drifted from the policy $\pi _ { \theta _ { \mathrm { o l d } } }$ that generated each replay sample. Specifically, it is the mean of $D _ { \mathrm { K L } } ( \pi _ { \theta } \| \pi _ { \theta _ { \mathrm { o l d } } } )$ across token positions in the replay batch. A higher value indicates that the replay data lies farther from the current policy distribution. This replay KL is a related but distinct diagnostic from the Policy Drift gate quantity itself: the gate uses token-level squared log-probability drift on stored actions (the Policy Drift score), whereas replay KL summarizes mismatch over the accepted replay batch.

Figure 22 shows the replay KL trajectories, and Table 15 reports interval-wise averages.

![](images/2c64b3af2e70b76420f6f4b55f03abf2e2252540d5896b1f2b291d61807837fd.jpg)  
Figure 22: Replay KL over training for the three τ conditions (running average, window = 15). Under MODERATE, replay KL remains stable throughout training. Under LOOSE, it increases monotonically.

<table><tr><td>Interval</td><td>TIGHT</td><td>MODERATE</td><td>LOOSE</td><td>LOOSE /MOD.</td></tr><tr><td>Early interval</td><td>0.0005</td><td>0.0021</td><td>0.0045</td><td>2.1×</td></tr><tr><td>Middle interval</td><td>0.0001</td><td>0.0023</td><td>0.0089</td><td>3.8×</td></tr><tr><td>Final interval</td><td>0.00003</td><td>0.0022</td><td>0.0170</td><td>7.7×</td></tr></table>

Table 15: Mean replay KL by training interval. Under MODERATE, replay KL remains flat at ∼0.002 throughout training, whereas under LOOSE it grows monotonically and reaches its largest gap from MODERATE in the final interval.

Under MODERATE, replay KL remains stable at approximately 0.002 throughout training. Under LOOSE, it increases monotonically and reaches a large multiple of the MODERATE level in the final interval. This reflects the continued admission of replay data that has progressively diverged from the current policy, without filtering by the gate. We report this growth as a distributional diagnostic; the extent to which it translates into gradient-level effects after clipping is not isolated in this analysis.

The replay KL under TIGHT falls below 0.0001 late in training. This does not indicate superior distribution compatibility; rather, it reflects the scarcity of replay samples admitted to the training batch.

## G.2.4 Performance and entropy dynamics

Figure 23 compares the training score and policy entropy trajectories, and Table 16 reports interval-wise averages.

![](images/521ddbca8c29681f0a032523059ceb628f7bca24d3c38f3a1977f68f4f6e1139.jpg)  
Figure 23: Training score (upper) and policy entropy (lower) over training for the three τ conditions (running average, window = 15).

<table><tr><td></td><td colspan="2">TIGHT (τ=0.001)</td><td colspan="2">MODERATE (τ=0.01)</td><td colspan="2">LOOSE (τ=0.1)</td></tr><tr><td>Interval</td><td>Score</td><td>Entropy</td><td>Score</td><td>Entropy</td><td>Score</td><td>Entropy</td></tr><tr><td>Early interval</td><td>0.206</td><td>0.783</td><td>0.272</td><td>0.757</td><td>0.269</td><td>0.756</td></tr><tr><td>Middle interval</td><td>0.365</td><td>0.527</td><td>0.392</td><td>0.447</td><td>0.382</td><td>0.441</td></tr><tr><td>Final interval</td><td>0.432</td><td>0.271</td><td>0.434</td><td>0.300</td><td>0.424</td><td>0.255</td></tr></table>

Table 16: Training score and policy entropy by interval for the three τ conditions.

TIGHT records the lowest score in the middle interval (0.365 vs. 0.392 for MODERATE and 0.382 for LOOSE). This appears to be due to the near-absence of replay, which prevents the run from benefiting from replay-based acceleration. TIGHT approaches MODERATE by the final interval, suggesting that on-policy learning alone can converge gradually, though with slower progress in the early-to-mid phase.

The score gap between MODERATE and LOOSE is approximately 1 percentage point, which is modest in absolute terms but consistent across all intervals in favor of MODERATE.

All three conditions exhibit a common decreasing trend in entropy over the course of training. In the final interval, the ordering of entropy across conditions (MODERATE 0.300 > TIGHT 0.271 > LOOSE 0.255) coincides with the ordering of score (MODERATE 0.434 > TIGHT 0.432 > LOOSE 0.424). MODERATE maintains both the highest score and the highest entropy among the three conditions at this stage. We note that this alignment does not imply a simple monotonic relationship between entropy and score across all training phases: in the middle interval, TIGHT exhibits the highest entropy yet the lowest score, reflecting slow learning due to the near-absence of replay rather than a beneficial preservation of exploration. Nonetheless, the co-occurrence of faster entropy decline and lower score under LOOSE is consistent with prior findings that rapid entropy reduction in LLM reinforcement learning tends to be associated with degraded learning outcomes (Lipkin et al., 2026; Wang et al., 2026b; Yu et al., 2025).

## G.2.5 Summary

Answer to RQ. A short log-spaced sweep is sufficient to identify a useful operating regime for τ. Collapsing acceptance rules out thresholds that starve replay, while rising replay KL rules out thresholds that leave the gate too permissive. The remaining regime admits replay selectively and keeps replay KL stable. Because these signals are available before final performance is known, unsuitable candidates need not be trained to completion. Together with threshold reuse across settings and model scales, these results support treating τ as a coarse calibration choice rather than a finely tuned performance parameter.

The same diagnostics could support an adaptive schedule that loosens the gate when replay is starved and tightens it when replay KL rises.

## G.3 Component ablation on mathematical reasoning

Table 17 reports a focused component ablation on MATH-500 at the evaluation endpoint. Headroom-only is the closest PER-style priority-only analogue in this group-level setting: it ranks buffered groups by Headroom without the current-policy compatibility gate. Under the same training configuration, the full Headroom-Drift combination records the highest endpoint values on both Best@32 and Mean@32, while Headroom-only and Policy-Driftonly are lower.

<table><tr><td>Variant</td><td>Best@32 (↑)</td><td>Mean@32 (↑)</td></tr><tr><td>Policy Drift only</td><td>0.7820</td><td>0.6794</td></tr><tr><td>Headroom only</td><td>0.8100</td><td>0.7140</td></tr><tr><td>Headroom-Drift Replay (full)</td><td>0.8200</td><td>0.7215</td></tr></table>

Table 17: MATH-500 ablation results at the evaluation endpoint. Arrows indicate metric direction (↑ higher is better). Best@32 denotes the average of the highest score among 32 samples per input, and Mean@32 denotes the average score across all 32 samples per input.

![](images/4504f1c55e392ba163ef543c8207a95ceab98ec7aa5258b6fc4fd752387148e1.jpg)  
Figure 24: Training-score curves for the mathematical-reasoning component ablation. In the figure legend, M2 denotes Policy Drift only, ZVP denotes Headroom only, and M2 + ZVP denotes full Headroom-Drift Replay.

Figure 24 provides the corresponding training-score trajectories for the same three variants. All three curves improve over training, and the late-stage ordering is consistent with Table 17: the full variant remains on top, Headroom-only follows closely, and Policy-Driftonly stays lower. The margin between full and Headroom-only is modest, while the gap to Policy-Drift-only is visibly larger in the later phase. These trajectories provide complementary context to the endpoint comparison.

## G.4 Sign-only versus advantage-weighted headroom

The main text defines Headroom as a token-level score that depends only on the sign of the response-level advantage $A _ { i } { \mathrm { : } }$ tokens in positively-advantaged responses contribute 1 − $\pi ( a _ { i , j } \mid s _ { i , j } )$ , tokens in negatively-advantaged responses contribute $\mathbf { \bar { \pi } } _ { \pi ( a _ { i , j } \mid s _ { i , j } ) }$ , and the group score is a flat average over all tokens. A natural question is whether reweighting each response’s contribution by $\left| A _ { i } \right|$ would produce a better ranking.

What changes mathematically. Let $s _ { g }$ denote the group success rate, and let $H _ { g } ^ { + } , H _ { g } ^ { - }$ denote the mean per-token headroom of the positive-advantage and negative-advantage responses respectively. Under equal response lengths, the two variants reduce to

$$
\begin{array} { r } { Z _ { \mathrm { s i g n } } ( g ) \approx s _ { g } H _ { g } ^ { + } + ( 1 - s _ { g } ) H _ { g } ^ { - } , \qquad Z _ { \mathrm { a d v } } ( g ) \approx \frac { 1 } { 2 } H _ { g } ^ { + } + \frac { 1 } { 2 } H _ { g } ^ { - } . } \end{array}
$$

The approximation for $Z _ { \mathrm { a d v } }$ follows because, under group-centered binary rewards, positive responses carry $| A _ { i } | \approx 1 - s _ { g }$ and negative responses carry $| A _ { i } | \approx \dot { s } _ { g } ,$ , so the count imbalance between the two sides is cancelled to first order by the advantage magnitudes.

The key difference is that $Z _ { \mathrm { s i g n } }$ contains the success ratio $s _ { g }$ directly as a mixing weight. In failure-heavy groups $( s _ { g } \check { \ll } 0 . 5 )$ , the negative branch dominates the score, and highconfidence wrong tokens (large π on negatively-advantaged responses) can push the group to the top of the ranking. $Z _ { \mathrm { a d v } }$ removes this structural dependence on $s _ { g } ,$ so groups are ranked more by branch-specific correction room than by failure ratio.

In short, the two variants do not define slightly different weightings of the same ranking;   
they define qualitatively different ranking geometries along the success-rate axis.

Empirical comparison. We conducted a small-scale preliminary experiment comparing the two variants under otherwise identical training configurations. Over the shared evaluation window (steps 1–38), sign-only Headroom achieves a higher average training score (0.3288 vs. 0.3022). Sign-only leads on 32 of 38 steps; advantage-weighted leads on 5. The gap is negligible in the earliest steps (at step 10: 0.2754 vs. 0.2783) but widens once replay becomes active. Over steps 19–38, where both runs use replay, sign-only leads on every step (mean 0.3863 vs. 0.3397). Representative checkpoints:

<table><tr><td>Step</td><td>Sign-only</td><td>Adv-weighted</td><td>∆</td></tr><tr><td>10</td><td>0.2754</td><td>0.2783</td><td>-0.0029</td></tr><tr><td>20</td><td>0.3481</td><td>0.3108</td><td>+0.0373</td></tr><tr><td>35</td><td>0.4177</td><td>0.3572</td><td>+0.0605</td></tr><tr><td>38</td><td>0.4119</td><td>0.3501</td><td>+0.0618</td></tr></table>

The mathematical analysis above suggests a possible explanation. By removing the structural dependence on $s _ { g } ,$ advantage weighting deprioritizes failure-heavy groups that signonly would rank highly. In this setting, those failure-heavy groups appear to carry useful corrective signal; deprioritizing them reduces the effectiveness of replay. We retain signonly Headroom in the main method accordingly.

## H Preliminary portability to a CISPO-style objective

![](images/f9927fc75f31635747e66fbac9db2197fec9607eae093265b20ac1a30b7291f0.jpg)  
Figure 25: Training-score trajectories under a CISPO-style objective. Headroom-Drift denotes full Headroom-Drift Replay (b256+r128). Baseline (b256) is the on-policy matched control; Baseline (b384) is the on-policy larger control. Running average with window 10.

The Conclusion identifies objective-level portability as an open question: whether the same two-axis decomposition transfers beyond GRPO-style clipped objectives. We provide preliminary training-score evidence on a CISPO-style objective using the same mathematical reasoning setup as the main experiments (Qwen2.5-Math-1.5B, identical dataset and evaluation protocol). The only change is the policy-optimization objective; Headroom ranking, Policy Drift gating, buffer mechanics, and all replay hyperparameters remain unchanged.

Figure 25 shows three patterns consistent with the GRPO results in the main text. First, Headroom-Drift Replay outperforms the on-policy matched baseline (b256) across nearly the entire training trajectory, confirming that principled replay adds value beyond the matched fresh-rollout budget under CISPO as well. Second, Headroom-Drift tracks above the on-policy larger baseline (b384) in the late phase and reaches a higher raw peak score (0.522 vs. 0.510 at steps 380 and 343 respectively), despite using fewer fresh responses.

Third, the on-policy larger baseline exhibits a sharp score collapse after step 400, whereas the replay trajectory remains more stable through the same region—a pattern that mirrors the delayed entropy-collapse observation under GRPO (Appendix E).