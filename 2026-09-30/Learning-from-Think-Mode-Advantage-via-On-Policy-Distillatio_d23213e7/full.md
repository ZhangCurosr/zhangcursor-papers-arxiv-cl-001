# Learning from Think-Mode Advantage via On-Policy Distillation

Wanqi Ren<sup>∗1</sup>, Jianxiang Wang<sup>1</sup>, Danxuan Liu<sup>1</sup>,

Linyi Ding<sup>1</sup>, Yuan Zhang<sup>1</sup>

<sup>1</sup>ByteDance, China renwanqi@bytedance.com

## Abstract

Explicit intermediate reasoning gives large language models (LLMs) a stronger problem-solving mode. We study learning from this think-mode advantage via on-policy distillation (OPD). OPD preserves student-generated trajectories and provides dense token-level teacher targets at student-visited prefixes. Privileged reasoning is used during distillation rather than student inference. Uniform ThinkOPD, a natural thinkenabled OPD baseline, conditions a fixed teacher on one shared think trace and uniformly distills every sibling student response. Although its prefixes are on-policy, the trace need not follow a route compatible with every complete response: the same privileged trace can induce diferent teacher–student discrepancies even when responses reach the same outcome. We summarize this interaction with trace–response divergence (TRD) and introduce ThinkOPD, which routes supervision at the response level by combining group-relative reward gain with a TRD-based compatibility proxy. Final response weights are normalized within each rollout group. Across mathematical reasoning and code generation, ThinkOPD outperforms Uniform ThinkOPD in both same-model settings and both cross-model teacher–student pairs, and it exceeds representative rationale and self-distillation baselines in a controlled comparison. Controlled interventions show that outcome benefit and the TRD-based proxy provide complementary routing signals in this setting. Think-enabled OPD provides a controlled setting for studying how teacher advantage becomes transferable along student responses.

## 1 Introduction

Intermediate reasoning helps LLMs solve problems that resist direct generation (Wei et al. 2022; Wang et al. 2023). Recent interfaces expose this capability through explicit think and no-think modes (Yang et al. 2025). We study learning from this think-mode advantage via OPD: a no-think student explores its own trajectories while a think-mode teacher supplies privileged supervision during distillation. We define no-think inference as generation without conditioning on the teacher’s privileged think trace, without constraining response length. The central question is how efectively the stronger think-mode view transfers through the sequence of states that the student visits.

Several transfer routes are possible. Ofline rationale tuning and reasoning internalization learn from completed teacher solutions but anchor supervision to fixed target sequences (Hsieh et al. 2023; Ho, Schmid, and Yun 2023; Li et al. 2023; Yu et al. 2024; Xu et al. 2025a). Reinforcement learning preserves student exploration but typically communicates task success through outcome-level rewards (Yu et al. 2025). Self-distillation ofers a third route, and OPD is particularly suitable here: it samples trajectories from the current student, then queries dense teacher distributions at the states that the student actually visits (Gu et al. 2024; Agarwal et al. 2024; Ye et al. 2026). Our matched baselines instantiate this space diferently: SDFT uses a demonstrationconditioned self-teacher to produce on-policy signals, OPSD uses a fixed privileged self-teacher, SDPO updates supervision with feedback, and BRTS searches over multiple teacher trajectories (Shenfeld et al. 2026; Zhao et al. 2026; Hübotter et al. 2026; Zhang et al. 2026). We study privileged selfdistillation under think-enabled OPD, pairing student exploration with a stronger teacher view.

A direct construction conditions a fixed teacher on its privileged think trace and uniformly distills the current student’s rollouts. We call this controlled baseline Uniform ThinkOPD. Throughout, think-enabled OPD denotes the broader setting. At first glance, the baseline ofers both a stronger teacher view and on-policy state coverage. These properties concern diferent objects: the scored prefix comes from the student, but the privileged trace follows the teacher’s solution route. The same trace can suit one response and conflict with another even when their outcomes are identical.

The natural think-enabled OPD construction reveals a trace–response discrepancy while holding the teacher model fixed. This makes learning from think-mode advantage a controlled setting for studying how outcome benefit and trace– response compatibility interact during transfer. Prior comparisons often vary teacher capacity, family, checkpoint, or data together with compatibility (Mirzadeh et al. 2020; Xu et al. 2025c; Li et al. 2025). Here the think trace instead exposes a concrete solution route whose compatibility can vary across complete student responses. We summarize this interaction with trace–response divergence (TRD), an operational response-level proxy for compatibility.

![](images/48302d8e7963c4711100816c58c6e2b38a0ec8ea9aa158cdc09d9002aeb55293.jpg)  
Figure 1: ThinkOPD response routing under a shared privileged trace. Uniform ThinkOPD weights sibling responses equally; ThinkOPD routes them using outcome benefit and a TRD-based compatibility proxy with group-normalized weights.

The conflict suggests two complementary decisions. Reward gain measures outcome benefit: whether the teacher obtains a better outcome for a response. TRD operationalizes trace–response compatibility through the discrepancy induced by the shared trace along that response. Neither signal alone defines distillation utility. Low discrepancy without benefit can emphasize an already adequate response, while benefit without a compatibility proxy can favor supervision delivered through an unsuitable route. ThinkOPD treats each trace–response pair as a routing unit (Figure 1). Within each prompt group, reward gain ranks outcome benefit, and inverse relative TRD from discounted future KL ranks responses by this compatibility proxy. Their product is normalized within the group to obtain the response-level OPD weight. Because Uniform ThinkOPD and ThinkOPD share the teacher view and OPD interface, their comparison isolates sibling-response routing (Table 1).

Across both same-model settings, ThinkOPD improves five-benchmark Overall avg@16 over Uniform ThinkOPD by 1.7–3.0 percentage points. It also improves pooled math avg@16 by 1.6–1.9 points in both cross-model pairs. On Qwen3-1.7B, it exceeds the highest-scoring external baseline by 2.3 points. The controlled component analysis directly tests the preceding design claim: TRD-only routing falls below uniform weighting, reward-only routing remains below the full method, their combination performs best, and reversing the TRD preference is harmful. Thus, reward gain captures outcome benefit, while lower TRD favors routes that are more compatible under the proposed proxy.

Our contributions are threefold:

• We identify TRD in natural think-enabled OPD, revealing a controlled setting for studying how outcome benefit and trace–response compatibility interact during transfer.

• We define TRD over complete student solution paths and propose ThinkOPD, a group-relative router combining reward gain with a TRD-based compatibility proxy.

• We show consistent same- and cross-model gains and controlled evidence that outcome benefit and trace compatibility are complementary.

<table><tr><td>Method</td><td>On-policy</td><td>Compatibility</td><td>Benefit</td><td>Multi-Trace</td></tr><tr><td>SDFT</td><td>√</td><td>X</td><td>X</td><td>X</td></tr><tr><td>OPSD/SDPO</td><td>√</td><td>×</td><td>×</td><td>X</td></tr><tr><td>BRTS</td><td>√</td><td>√</td><td>√</td><td>√</td></tr><tr><td>U-ThinkOPD</td><td>√</td><td>×</td><td>×</td><td>X</td></tr><tr><td>ThinkOPD</td><td>√</td><td>√</td><td>√</td><td>×</td></tr></table>

Table 1: Supervision interfaces compared by on-policy scoring, compatibility modeling, outcome-aware selection or weighting, and use of multiple teacher traces. Multi-Trace means that a method generates or compares multiple teacher reasoning trajectories per training instance; it does not denote sibling student responses. All methods use privileged teacher context, and U-ThinkOPD denotes Uniform ThinkOPD.

## 2 Related Work

Reasoning transfer and privileged self-distillation. Sequence and rationale distillation use fixed teacher targets, while reinforcement learning uses task feedback (Hinton, Vinyals, and Dean 2015; Kim and Rush 2016; Hsieh et al. 2023; Ho, Schmid, and Yun 2023; Yu et al. 2024; Xu et al. 2025a; Yu et al. 2025). OPD and MiniLLM align teachers to student trajectories. Speculative KD interleaves student proposals and teacher corrections under unreliable feedback (Agarwal et al. 2024; Gu et al. 2024; Xu et al. 2025b). OPCD adds context-conditioned teachers (Ye et al. 2026). Our controls span demonstration-conditioned on-policy SDFT, fixed privileged OPSD, and feedbackconditioned SDPO (Shenfeld et al. 2026; Zhao et al. 2026; Hübotter et al. 2026). ThinkOPD fixes the teacher view and routes one trace across sibling responses, separating response routing from teacher evolution and rationale generation.

Compatibility and selective supervision. Teacher strength can fail under capacity or reasoning mismatch (Mirzadeh et al. 2020; Xu et al. 2025c; Li et al. 2025, 2026). Selective objectives adapt local supervision by entropy, teachability, position, or near-future guidance (Jin et al. 2026; Wang et al. 2026; Liu et al. 2026; Jiang and Ferraro 2026). Concurrent BRTS instead searches teacher trajectories (Zhang et al. 2026). ThinkOPD aggregates discrepancy across each response and routes a shared trace by outcome benefit and compatibility.

## 3 Method

ThinkOPD treats each shared-trace/student-response pair as the unit of supervision. It first estimates response-level trace compatibility and then combines it with outcome benefit to route dense OPD targets across sibling trajectories.

## 3.1 Problem Formulation

We consider a no-think student $\pi _ { S }$ and a fixed think-mode teacher π . During ofline bank construction, the teacher jointly generates each think trace z and a paired response $\widetilde { y } _ { T }$ A task verifier scores the response before the trace–response– reward record is stored. At training time, the current student samples G sibling responses $y ^ { ( 1 ) } , \ldots , y ^ { ( G ) }$ for prompt x. The resulting prompt group $\mathcal G ( x )$ holds x and the selected z fixed while varying the student trajectory. The teacher-side record paired with sibling k supplies $r _ { T } ^ { ( k ) } = r ( \widetilde { y } _ { T } ^ { ( k ) } )$ , and the online student response receives $r _ { S } ^ { ( k ) } = r ( y ^ { ( k ) } )$ . At prefix $y _ { < t } ^ { ( k ) }$ , the two policy views are:

$$
\begin{array} { r } { p _ { S , t } ^ { ( k ) } ( \cdot ) = \pi _ { S } ( \cdot \mid x , y _ { < t } ^ { ( k ) } ) , } \\ { p _ { T , t } ^ { z , ( k ) } ( \cdot ) = \pi _ { T } ( \cdot \mid x , z , y _ { < t } ^ { ( k ) } ) . } \end{array}\tag{1}
$$

The student samples without $z .$ The teacher conditions on z only when scoring the same prefix. We compute sparse forward KL on the teacher’s top-K support, with $q _ { T , t } ^ { ( k ) }$ and $q _ { S , t } ^ { ( k ) }$ denoting the corresponding renormalized views. Because z is fixed within a prompt group, we suppress it in this notation below. We denote by $\mathcal { R } ( \boldsymbol { y } ^ { ( k ) } )$ all valid generated-token positions in the complete student response, excluding prompt and padding tokens. Every sampled response and every valid token within it participate in scoring. Here G determines the number of sibling trajectories in each within-prompt comparison set, and K determines the sparse teacher support used to evaluate token-level disagreement. Both are fixed by the training protocol.

## 3.2 Uniform ThinkOPD

To isolate response routing, Uniform ThinkOPD and ThinkOPD use the same ofline trace–response–reward bank, student rollouts, and response-conditioned teacher scores. The teacher view $q _ { T , t } ^ { ( k ) }$ is queried at every student prefix and is response-specific. The paired teacher response stored in the bank supplies $r _ { T } ^ { ( k ) }$

Let V be the scored response-token triples in an update. The direct baseline, Uniform ThinkOPD, gives every trace– response pair the same OPD weight:

$$
\mathcal { L } _ { \mathrm { v a n i l l a } } = \frac { 1 } { | \mathcal { V } | } \sum _ { ( x , k , t ) \in \mathcal { V } } { \mathrm { K L } } \Bigl ( q _ { T , t } ^ { ( k ) } \| q _ { S , t } ^ { ( k ) } \Bigr ) .\tag{2}
$$

This is prefix-level on-policy supervision: every target is queried at a state visited by the current student.

![](images/942d0e91d1d158d5ef68a55e668f30bbec1b4b8a1d23af0fe526d2785ce77f9e.jpg)  
Figure 2: A controlled trace–response discrepancy case. One closed-form think trace conditions two correct responses with the same answer and reward; the route-aligned response has lower TRD than the alternative recurrence route.

## 3.3 Trace–Response Divergence

The on-policy property of Equation 2 applies to the prefix, not to the privileged trace. Because z follows the teacher’s solution route, reusing it across siblings couples one teacher trajectory with several student trajectories. Uniform ThinkOPD nevertheless treats every coupling as equally useful. TRD measures the discrepancy induced by trace-conditioned supervision as each complete student response unfolds.

Holding teacher capacity, family, and checkpoint fixed makes route-dependent discrepancy directly observable at the trace–response level, providing a controlled compatibility proxy for think-enabled OPD.

Figure 2 illustrates the interaction while holding the problem, teacher trace, answer, and reward fixed. The response following the teacher’s closed-form route has lower TRD than the alternative correct route. Equal outcomes can conceal diferent trace-conditioned discrepancies.

TRD captures a sequential interaction rather than a singletoken discrepancy. A local KL spike can reflect a temporary lexical choice, whereas a route-level departure can alter teacher targets across several subsequent prefixes. Discounted continuation distinguishes these patterns by accumulating persistent disagreement while emphasizing its nearer consequences. The finite horizon bounds this continuation, and response aggregation yields one operational score for the complete trace–response pair.

We construct TRD in three stages: local disagreement, discounted continuation, and response aggregation. For response k, sparse forward KL first records the traceconditioned disagreement at position $t ;$ a bounded future sum then captures whether that discrepancy persists; averaging these local-future values produces one response-level score. Formally:

$$
D _ { t } ^ { ( k ) } = \mathrm { K L } \Big ( q _ { T , t } ^ { ( k ) } \| q _ { S , t } ^ { ( k ) } \Big )\tag{3}
$$

$$
\mathrm { T R D } _ { t } ^ { ( k ) } = \sum _ { \tau = t } ^ { \operatorname * { m i n } \{ t + H , | { \boldsymbol { y } } ^ { ( k ) } | \} } { \gamma ^ { \tau - t } D _ { \tau } ^ { ( k ) } m _ { \tau } ^ { ( k ) } } , \qquad 0 < \gamma < 1\tag{4}
$$

$$
\overline { { \mathrm { T R D } } } ^ { ( k ) } = \frac { 1 } { | \mathcal { R } ( y ^ { ( k ) } ) | } \sum _ { t \in \mathcal { R } ( y ^ { ( k ) } ) } \mathrm { T R D } _ { t } ^ { ( k ) }\tag{5}
$$

Here $m _ { \tau } ^ { ( k ) } \in \{ 0 , 1 \}$ masks invalid generated positions, H bounds the continuation window, and $0 < \gamma < 1$ 1 discounts more distant discrepancies. The set $\mathcal { R } ( y ^ { ( k ) } )$ contains the valid positions used for response aggregation. Low $\overline { { \mathrm { T R D } } } ^ { ( k ) }$ means trace-conditioned targets remain close to the student across the response. High values indicate sustained departure. This aggregation turns token-level KL into a responselevel trace–response compatibility proxy.

## 3.4 Group-Relative Response Routing

Having measured the shared-trace interaction, ThinkOPD routes supervision with two signals. Reward gain compares response-specific teacher and student rewards, and TRD measures how the shared trace interacts with each sibling trajectory. Because both vary across prompts, we compare them only within the same prompt group. Let $\operatorname { S t d } _ { \mathcal { G } }$ denote z-score standardization across siblings in $\mathcal { G } ( x ) \mathrm { : \ }$

$$
\begin{array} { r l } & { \boldsymbol { g } ^ { ( k ) } = \boldsymbol { r } _ { T } ^ { ( k ) } - \boldsymbol { r } _ { S } ^ { ( k ) } , } \\ & { \widehat { \boldsymbol { g } } ^ { ( k ) } = \operatorname { S t d } _ { \mathcal { G } } \left( \boldsymbol { g } ^ { ( k ) } \right) , } \\ & { \widehat { \operatorname { T R D } } ^ { ( k ) } = \operatorname { S t d } _ { \mathcal { G } } \left( \overline { { \operatorname { T R D } } } ^ { ( k ) } \right) . } \end{array}\tag{6}
$$

Because both rewards are response-specific, ${ \widehat { g } } ^ { ( k ) }$ retains teacher-side as well as student-side variation within the group. Larger ${ \widehat { g } } ^ { ( k ) }$ indicates greater outcome benefit, and smaller $\widehat { \mathrm { T R D } } ^ { ( k ) }$ indicates greater compatibility under the operational proxy. Their response score is:

$$
\begin{array} { r l } & { \widetilde { \lambda } ^ { ( k ) } = \sigma \Bigl ( \widehat { g } ^ { ( k ) } \Bigr ) \exp \left[ - \eta \sigma \Bigl ( \widehat { \mathrm { T R D } } ^ { ( k ) } \Bigr ) \right] , } \\ & { \lambda ^ { ( k ) } = \widetilde { \lambda } ^ { ( k ) } / Z _ { x } . } \end{array}\tag{7}
$$

Here $\sigma$ is the sigmoid function. The coeficient $\eta > 0$ controls routing contrast: larger values suppress relatively high-TRD responses more strongly, while smaller values approach benefit-dominated routing. The group-specific normalizer $\begin{array} { r } { Z _ { x } = G ^ { - 1 } \sum _ { j = 1 } ^ { G } \widetilde { \lambda } ^ { ( j ) } } \end{array}$ preserves the response ranking. The first factor increases with relative outcome benefit, and the second decreases smoothly with relative TRD without introducing a hard threshold. Figure 3 visualizes this joint routing pattern across reward-advantage and TRD bins and reports the sample mass in every region of the grid.

The two normalization stages make routing relational within each prompt group. Each response is compared only with siblings generated for the same problem and paired with the same trace; the resulting response weights are then normalized within that group. Consequently, $\lambda ^ { ( k ) }$ reallocates dense OPD supervision among sibling paths according to their relative benefit and TRD.

![](images/2651d0978df9ceda30140315bc4092052ed3620b0d32738f9f5ddb70a78d5172.jpg)

![](images/c69be18c3554936346cab31e2b0262363f2d6c7b719e6c710c423af57b89588f.jpg)  
Figure 3: Sample mass (left) and mean response weight (right) for 768 valid responses in 192 four-response prompt groups from Qwen3-1.7B, binned by reward advantage and TRD. Each coarse region spans two fine bins per axis.

The routing parameters govern complementary decisions: H sets how far the trace–response interaction is observed, while η controls its efect on response weights. We examine both with the within-window discount γ fixed.

Figure 3 connects the motivating phenomenon to the router. Sibling responses occupy low, middle, and high TRD bins even within the same reward-advantage band, so outcome benefit does not determine the trace-conditioned interaction. The weight map assigns its largest coeficients where positive benefit coincides with low TRD; at comparable benefit, weight falls as TRD increases. The two signals therefore organize distinct axes of the response-level decision.

## 3.5 Training Objective and Procedure

Each response coeficient is shared by all valid generated token positions and fixed during the policy update. With sg(·) denoting stop-gradient, the weighted objective is:

$$
\mathcal { L } _ { \mathrm { T h i n k O P D } } = \frac { 1 } { | \mathcal { V } | } \sum _ { ( \boldsymbol { x } , \boldsymbol { k } , t ) \in \mathcal { V } } \mathrm { s g } \Big ( \lambda ^ { ( \boldsymbol { k } ) } \Big ) \mathrm { K L } \Big ( \boldsymbol { q } _ { T , t } ^ { ( \boldsymbol { k } ) } \| \boldsymbol { q } _ { S , t } ^ { ( \boldsymbol { k } ) } \Big ) ,\tag{8}
$$

where $\nu$ is the same set of scored positions as in Equation 2.   
Setting every $\lambda ^ { ( k ) }$ to one recovers Uniform ThinkOPD.

Sharing $\bar { \lambda } ^ { ( k ) }$ across the response keeps the routing decision aligned with the complete trace–response pair from which TRD is computed. The underlying supervision remains token-level and dense, but its aggregate contribution reflects whether that student path ofers both outcome benefit and compatible trace-conditioned guidance. This separates the granularity of supervision from the granularity of routing.

At each step, the student generates sibling responses before routing, and the teacher scores their visited prefixes under the shared trace. The ofline bank supplies fixed context and teacher records, while current paths receive recomputed response-conditioned targets. This preserves the defining OPD interaction summarized in Algorithm 1.

```latex
Algorithm 1 ThinkOPD
Input: Ofline trace–response–reward bank, student $\pi _ { S } ,$ teacher
π , and routing parameters
Output: Updated no-think student policy $\pi _ { S }$
1: for each minibatch X of prompts do
2: for each prompt $\boldsymbol { x } \in \hat { \mathcal { X } }$ do
3: Retrieve the shared trace z and paired teacher re
sponses/rewards
4: Sample sibling responses $y ^ { ( 1 : G ) } \sim \pi _ { S } ( \cdot \mid x )$
5: for each response $\bar { y } ^ { ( k ) }$ do
6: Score $y ^ { ( \bar { k } ) }$ and compute response-token ${ \mathrm { K L } } D _ { t } ^ { ( k ) }$
7: Compute reward gain $g ^ { ( k ) }$ and response TRD TRD (k)
8: end for
9: Standardize reward gain and TRD within $\mathcal G ( x )$
10: Compute and stop-gradient $\lambda ^ { ( k ) }$ with Equation 7
11: end for
12: Update π<sub>S</sub> with weighted full-response forward KL
13: end for
```

## 4 Experiments

We evaluate ThinkOPD across model scales, mathematical reasoning, code generation, and cross-model distillation. The experiments answer four questions:

Q1 Does routing by outcome benefit and a TRD-based compatibility proxy improve think-mode transfer?

Q2 Does it generalize across models, tasks, and model pairs? Q3 Does the TRD proxy outperform alternative compatibility proxies?

Q4 Is response routing more eficient than trajectory search?

## 4.1 Settings

Models. The primary comparison uses Qwen3-1.7B as both student and think-mode teacher. Same-model experiments additionally evaluate Qwen3-0.6B (Yang et al. 2025); crossmodel experiments pair Qwen3-4B-Thinking-2507 with Qwen3-1.7B and Qwen3-1.7B with Qwen3-0.6B.

Training data and evaluation benchmarks. Training and evaluation data are disjoint in both domains. Math training uses DAPO-Math-17K (Yu et al. 2025), with 256 prompts held out for checkpoint selection; evaluation uses AIME24, AIME25, and AMC23 (Hugging Face H4 2025; OpenCompass 2025; Washbourne 2025). Code training uses Skywork-OR1-Coding-14K (He et al. 2025) with a separate 256- problem validation split; evaluation uses Skywork heldout512 and LiveCodeBench v5 (Jain et al. 2025). We report avg@16, the mean correctness over 16 samples per problem, and best@16, the fraction of problems solved at least once. Baselines. Uniform ThinkOPD is the uniform-routing control: it shares the fixed think-mode teacher, ofline bank, student rollouts, training budget, and checkpoint-selection protocol with ThinkOPD, but assigns equal weight to every response. Cross-model experiments use the same control.

The method families motivating the external comparison are summarized in Table 1. On the representative Qwen3- 1.7B setting, SDFT, OPSD, SDPO, and BRTS use matched training data, response budgets, checkpoint selection, and evaluation. Detailed implementations and protocol adaptations appear in the supplementary material.

<table><tr><td rowspan="2">Method</td><td colspan="2">Math</td><td colspan="2">Coding</td><td rowspan="2">Overall</td></tr><tr><td>AIME24 AIME25 AMC23 Skywork LCBv5</td><td></td><td></td><td></td></tr><tr><td>SDFT</td><td>28.4</td><td>25.2</td><td>56.9</td><td>18.8 12.0</td><td>28.2</td></tr><tr><td>OPSD</td><td>28.8</td><td>24.7</td><td>57.7</td><td>19.4 13.0</td><td>28.7</td></tr><tr><td>SDPO</td><td>30.0</td><td>25.1</td><td>58.4</td><td>20.2 13.5</td><td>29.4</td></tr><tr><td>BRTS</td><td>30.1</td><td>26.6</td><td>58.2</td><td>20.4 13.4</td><td>29.7</td></tr><tr><td>U-ThinkOPD</td><td>29.2</td><td>25.0</td><td>57.6</td><td>20.1 13.3</td><td>29.0</td></tr><tr><td>ThinkOPD</td><td>35.2</td><td>25.4</td><td>60.8</td><td>25.2 13.4</td><td>32.0</td></tr></table>

Table 2: Qwen3-1.7B baseline comparison under eval16 (%, avg@16; higher is better). Overall is the unweighted mean over the five benchmarks. Bold and underlined entries indicate first and second place.

Implementation. ThinkOPD and its uniform control use full-parameter BF16 training with verl (Sheng et al. 2025) for 100 steps, batch size 64, and learning rate $1 0 ^ { - 6 }$ on 32 highperformance GPUs. Teacher distributions are recomputed on sampled student prefixes, and checkpoints are selected by held-out validation avg@3. The primary Qwen3-1.7B math comparison averages three independent runs. Other modeltask settings, baselines, interventions, proxies, and sensitivity points are matched validation-selected point estimates. The supplement provides full implementation, uncertainty, and resource details.

Ofline Teacher Bank and Asynchronous OPD. To reduce training latency, we use two complementary measures. First, an immutable ofline Teacher Bank stores the fixed teacher’s think trace, paired response, verifier reward, and provenance for each prompt. The student still samples fresh responses, and the teacher recomputes forward-only distributions at visited prefixes, so scoring remains on-policy. Second, asynchronous OPD overlaps rollout, trace-conditioned teacher scoring, and optimization through a bounded queue of complete, single-version sibling groups; consumed groups lag by at most one update. The bank removes repeated teacher generation, while asynchronous execution reduces pipeline idle time. Further details appear in the supplementary material.

## 4.2 Results and Analysis

ThinkOPD Outperforms Matched Baselines Table 2 tests the central claim under matched supervision. Relative to Uniform ThinkOPD, ThinkOPD raises five-benchmark Overall avg@16 from 29.0 to 32.0. This 3.0-point diference isolates response-level routing in our implementation because the teacher, trace bank, student rollouts, and optimization protocol are unchanged. ThinkOPD also exceeds the strongest external baseline, BRTS at 29.7, by 2.3 points.

The external baselines occupy a narrow Overall range of 28.2–29.7, and Uniform ThinkOPD is already competitive within it. This comparison sharpens the motivating distinction. A privileged think trace supplies potential outcome benefit, but its availability does not determine which sibling response can use that supervision coherently. The matched improvement from Uniform ThinkOPD to ThinkOPD is consistent with routing this shared signal by both benefit and trace–response compatibility.

<table><tr><td rowspan="3">Model</td><td rowspan="3">Method</td><td colspan="6">Math</td><td colspan="4">Coding</td><td rowspan="2" colspan="2">Overall</td></tr><tr><td colspan="2">AIME24</td><td colspan="2">AIME25</td><td colspan="2">AMC23</td><td colspan="2">Skywork held.</td><td colspan="2">LCBv5</td></tr><tr><td>Avg.</td><td>Best</td><td>Avg.</td><td>Best</td><td>Avg.</td><td>Best</td><td>Avg.</td><td>Best</td><td>Avg.</td><td>Best</td><td>Avg.</td><td>Best</td></tr><tr><td rowspan="2">Qwen3 0.6B</td><td>Think off</td><td>2.5</td><td>16.6</td><td>2.5</td><td>6.6</td><td>23.5</td><td>62.6</td><td>10.1</td><td>24.8</td><td>6.1</td><td>15.0</td><td>8.9</td><td>25.1</td></tr><tr><td>Think on</td><td>8.7</td><td>36.6</td><td>17.0</td><td>40.0</td><td>46.3</td><td>78.3</td><td>19.5</td><td>41.7</td><td>12.8</td><td>24.7</td><td>20.9</td><td>44.3</td></tr><tr><td rowspan="2">Qwen3 1.7B</td><td>Think off</td><td>13.5</td><td>46.6</td><td>10.0</td><td>30.0</td><td>40.6</td><td>73.4</td><td>18.1</td><td>33.3</td><td>11.2</td><td>21.8</td><td>18.6</td><td>41.0</td></tr><tr><td>Think on</td><td>43.3</td><td>76.6</td><td>36.0</td><td>70.0</td><td>73.7</td><td>92.7</td><td>40.1</td><td>58.0</td><td>31.1</td><td>46.2</td><td>44.8</td><td>68.7</td></tr><tr><td rowspan="2">Qwen3 0.6B</td><td>U-ThinkOPD</td><td>2.2</td><td>16.6</td><td>3.3</td><td>10.0</td><td>23.7</td><td>61.4</td><td>16.8</td><td>39.4</td><td>10.5</td><td>22.5</td><td>11.3</td><td>29.9</td></tr><tr><td>ThinkOPD</td><td>1.6</td><td>13.3</td><td>4.5</td><td>20.0</td><td>25.9</td><td>68.6</td><td>19.7</td><td>42.5</td><td>13.7</td><td>25.8</td><td>13.0</td><td>34.0</td></tr><tr><td rowspan="2">Qwen3 1.7B</td><td>U-ThinkOPD ThinkOPD</td><td>29.2</td><td>65.5</td><td>25.0</td><td>62.2</td><td>57.6</td><td>88.3</td><td>20.1</td><td>41.4</td><td>13.3</td><td>27.9</td><td>29.0</td><td>57.0</td></tr><tr><td></td><td>35.2</td><td>73.3</td><td>25.4</td><td>61.1</td><td>60.8</td><td>87.5</td><td>25.2</td><td>55.0</td><td>13.4</td><td>41.2</td><td>32.0</td><td>63.6</td></tr></table>

Table 3: Same-model performance under eval16 across mathematical reasoning and code generation (%, higher is better). Overall Avg. and Best are the unweighted means of their corresponding five benchmark columns. Think-of and think-on rows are undistilled base-policy references; within each distilled model group, boldface marks the higher result.

<table><tr><td>Model Method</td><td>AIME24 AIME25 AMC23</td><td></td><td></td><td>Avg.</td><td>∆</td></tr><tr><td>4B OPD</td><td>33.9</td><td>29.7</td><td>66.2</td><td>51.8</td><td>0.0</td></tr><tr><td>↓ U-ThinkOPD</td><td>35.2</td><td>30.6</td><td>65.5</td><td>51.8</td><td>0.0</td></tr><tr><td>1.7B ThinkOPD</td><td>37.5</td><td>34.3</td><td>66.5</td><td>53.7</td><td>+1.9</td></tr><tr><td>1.7B OPD</td><td>2.5</td><td>1.4</td><td>24.5</td><td>15.0</td><td>-6.0</td></tr><tr><td>↓ U-ThinkOPD</td><td>5.2</td><td>6.6</td><td>31.9</td><td>21.0</td><td>0.0</td></tr><tr><td>0.6B ThinkOPD</td><td>5.0</td><td>7.0</td><td>34.5</td><td>22.6</td><td>+1.6</td></tr></table>

Table 4: Cross-model Qwen3 math performance under eval16 (%, avg@16, higher is better); the first column gives teacher and student sizes. Avg. pools all AIME24, AIME25, and AMC23 problems, and ∆ is its gain over Uniform ThinkOPD within each pair.

ThinkOPD achieves the strongest Overall result through clear gains on AIME24, AMC23, and Skywork while remaining competitive on AIME25 and LCBv5. This breadth distinguishes response routing from baselines whose improvements concentrate on individual benchmarks.

Gains Across Models and Tasks Table 3 separates teacher-side opportunity from realized transfer. Enabling think mode creates 12.0–26.2 points of base-policy headroom across the two Qwen3 scales, but this headroom is an opportunity rather than the amount that uniform distillation can recover. Across both same-model scales, ThinkOPD improves Overall by 1.7–3.0 points over the uniform control.

The two cross-model settings make the distinction more visible. For Qwen3-4B→1.7B, no-think OPD and uniform think-mode supervision both score 51.8, while routing reaches 53.7. For Qwen3-1.7B→0.6B, uniform supervision raises the score from 15.0 to 21.0, and routing further improves it to 22.6. Teacher headroom alone thus predicts neither the direction nor the magnitude of transfer. Together with the controlled interventions below, this pattern is consistent with transfer depending on both available benefit and the trace–response interaction. The Qwen3-1.7B result also raises LCBv5 best@16 by 13.3 points, broadening solvedproblem coverage across code tasks.

Benefit and Compatibility Useful supervision should combine a response that ofers outcome benefit with a think trace that yields compatible supervision along that response. Table 5 tests this claim by changing one routing component at a time while fixing the teacher bank, reward-valid responses, normalization, training budget, and checkpoint selection.

The directional interventions separate three explanations. If arbitrary nonuniform weighting were suficient, reversing the TRD preference should remain useful; instead, Reversed TRD reaches 42.5, below uniform routing at 44.8. If compatibility alone determined utility, removing reward gain should retain the improvement, yet this variant also falls below the uniform control. Reward-only routing improves over uniform weighting but remains below full ThinkOPD at 48.0. Within this controlled ablation, the observed gain is best explained by combining correction opportunity with the TRD-based compatibility proxy.

The matched controls then replace only the compatibility distance while retaining the reward branch and group normalization. Length uses the number of valid generated tokens and tests whether routing merely favors shorter responses. Raw-KL averages the undiscounted sparse forward KL across the response, testing whether instantaneous disagreement magnitude already captures the efect of TRD. Top-k agreement averages Teacher–Student support overlap at the scored prefixes and converts it to a distance, testing whether coarse token-set similarity sufices. All three reuse tensors already produced during distillation and require no additional Teacher or Student forward pass; exact definitions appear in the supplementary material.

TRD attains the highest aggregate result at 48.0, exceeding all three matched controls. Discounted continuation therefore captures routing information beyond response length, instantaneous KL, and coarse token-support agreement in this controlled setting. The ranking also clarifies the signals’ division of labor: reward gain identifies responses with improvement headroom, while TRD diferentiates which of them remain coherent with the shared teacher route.

<table><tr><td>Configuration</td><td>AIME24</td><td>AIME25</td><td>AMC23</td><td>Avg.</td><td>∆</td></tr><tr><td>U-ThinkOPD</td><td>29.2</td><td>25.0</td><td>57.6</td><td>44.8</td><td>+0.0</td></tr><tr><td colspan="6">Routing component ablations</td></tr><tr><td>w/o TRD</td><td>31.0</td><td>26.3</td><td>59.4</td><td>46.5</td><td>+1.7</td></tr><tr><td>w/o Reward</td><td>27.5</td><td>25.4</td><td>55.6</td><td>43.4</td><td>-1.4</td></tr><tr><td>w/ Reversed TRD</td><td>26.3</td><td>25.1</td><td>54.8</td><td>42.5</td><td>-2.3</td></tr><tr><td colspan="6">Alternative compatibility proxies</td></tr><tr><td>Raw-KL</td><td>29.8</td><td>26.2</td><td>60.3</td><td>46.8</td><td>+2.0</td></tr><tr><td>Length</td><td>32.1</td><td>25.2</td><td>59.1</td><td>46.3</td><td>+1.5</td></tr><tr><td>Top-k agreement</td><td>31.9</td><td>27.7</td><td>59.7</td><td>47.2</td><td>+2.4</td></tr><tr><td>ThinkOPD (Ours)</td><td>35.2</td><td>25.4</td><td>60.8</td><td>48.0</td><td>+3.2</td></tr></table>

Table 5: Math routing ablations under eval16 (%, avg@16; higher is better). Avg. pools all AIME24, AIME25, and AMC23 problems; ∆ is relative to Uniform ThinkOPD.

![](images/e0d7156d7b150fbbece47ca827edfd3644c658efd0e81fe7bd1c5adc37d55e72.jpg)  
Figure 4: Sensitivity of pooled Math avg@16 to future horizon H and inverse-TRD strength η. Each sweep varies one parameter while fixing the other at its default.

Parameter sensitivity. Figure 4 varies one routing parameter at a time around the default configuration. Performance stays above uniform routing throughout the tested future horizons, showing robustness over a broad range of continuation windows. The routing strength η has an interior optimum at 0.5: weaker values approach benefit-dominated weighting, while stronger values can suppress responses that combine high TRD with substantial outcome benefit. This trend reinforces the component analysis by locating the best operating point at a balance between outcome benefit and compatibility. Across the tested range, the broad tolerance to H and sharper response to η indicate that temporal aggregation is stable, while balancing benefit against compatibility remains the consequential calibration choice.

Response Routing Improves Supervision Eficiency All methods in Figure 5 use the same asynchronous OPD execution, so the comparison separates three sources of workload rather than diferent pipeline schedules. Uniform ThinkOPD adds the privileged think trace to every scored prefix, lengthening the teacher context and increasing forward and attention/KV processing; its cost is 1.28 versus 1.00 for standard OPD even though trace generation is ofline. ThinkOPD reaches only 1.31 because routing reuses rewards and tokenlevel divergences already computed for distillation, while raising Overall avg@16 from 29.0 to 32.0.

![](images/8f89909a36d03254e5886fdfdfddc7696d9af9421c928930aae59173ed3e0dab.jpg)  
Figure 5: Recurring cost under matched asynchronous OPD, normalized to standard OPD (1.00). Shared ofline-bank construction is excluded; BRTS-4 selects among four online teacher trajectories.

BRTS-4 instead generates four candidate teacher trajectories online and selects among them for every group. It cannot amortize one shared ofline trace across sibling responses, increasing recurring cost to 2.10. ThinkOPD reallocates the available trace-conditioned supervision and achieves a 2.3- point higher Overall score with 37.6% less recurring time. Detailed accounting appears in the supplement.

## 5 Conclusion

Think-enabled OPD ofers a stronger teacher view while preserving student-side exploration, yet its on-policy prefixes do not guarantee that one privileged trace provides equally suitable supervision for every sibling response. ThinkOPD makes this interaction explicit: reward gain measures outcome benefit, TRD supplies an operational proxy for trace– response compatibility, and group-normalized routing combines them across sibling responses. Across mathematics and code, this router improves over Uniform ThinkOPD in both same-model and both cross-model settings and leads the matched external baselines in the representative comparison. In the controlled ablation, the observed gain is best explained by combining both routing signals, while reusing existing teacher signals keeps the router practical. ThinkOPD preserves student exploration and allocates the available privileged trace across sampled student paths. More broadly, think-enabled OPD exposes route-dependent compatibility with a fixed teacher, providing a controlled setting for studying supervision transfer across student paths.

## References

Agarwal, R.; Vieillard, N.; Zhou, Y.; Stanczyk, P.; Ramos Garea, S.; Geist, M.; and Bachem, O. 2024. Onpolicy distillation of language models: Learning from selfgenerated mistakes. In Proceedings ofthe 12th International Conference on Learning Representations (ICLR). Vienna, Austria.

Gu, Y.; Dong, L.; Wei, F.; and Huang, M. 2024. MiniLLM: Knowledge distillation of large language models. In Proceedings of the 12th International Conference on Learning Representations (ICLR). Vienna, Austria.

He, J.; Liu, J.; Liu, C. Y.; Yan, R.; Wang, C.; Cheng, P.; Zhang, X.; Zhang, F.; Xu, J.; Shen, W.; Li, S.; Zeng, L.; Wei, T.; Cheng, C.; An, B.; Liu, Y.; and Zhou, Y. 2025. Skywork open reasoner 1 technical report. arXiv:2505.22312.

Hinton, G.; Vinyals, O.; and Dean, J. 2015. Distilling the knowledge in a neural network. arXiv:1503.02531.

Ho, N.; Schmid, L.; and Yun, S.-Y. 2023. Large language models are reasoning teachers. In Proceedings ofthe 61stAnnual Meeting ofthe Associationfor Computational Linguistics (ACL), Volume 1: Long Papers, 14852–14882. Toronto, Canada.

Hsieh, C.-Y.; Li, C.-L.; Yeh, C.-k.; Nakhost, H.; Fujii, Y.; Ratner, A.; Krishna, R.; Lee, C.-Y.; and Pfister, T. 2023. Distilling step-by-step! Outperforming larger language models with less training data and smaller model sizes. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, 8003–8017. Toronto, Canada.

Hübotter, J.; Lübeck, F.; Behric, L.; Baumann, A.; Bagatella, M.; Marta, D.; Hakimi, I.; Shenfeld, I.; Kleine Buening, T.; Guestrin, C.; and Krause, A. 2026. Reinforcement learning via self-distillation. arXiv:2601.20802.

Hugging Face H4. 2025. AIME 2024 dataset. Hugging Face Datasets.

Jain, N.; Han, K.; Gu, A.; Li, W.-D.; Yan, F.; Zhang, T.; Wang, S.; Solar-Lezama, A.; Sen, K.; and Stoica, I. 2025. LiveCodeBench: Holistic and contamination-free evaluation of large language models for code. In Proceedings of the 13th International Conference on Learning Representations (ICLR). Singapore.

Jiang, Y.; and Ferraro, F. 2026. Bridging reasoning trajectories in on-policy distillation via near-future guidance. arXiv:2606.00305.

Jin, W.; Min, T.; Yang, Y.; Wei, D.; Zhou, Y.; Kadhe, S. R.; Baracaldo, N.; and Lee, K. 2026. Entropy-aware on-policy distillation of language models. arXiv:2603.07079.

Kim, Y.; and Rush, A. M. 2016. Sequence-level knowledge distillation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1317–1327. Austin, TX.

Li, L. H.; Hessel, J.; Yu, Y.; Ren, X.; Chang, K.-W.; and Choi, Y. 2023. Symbolic Chain-of-Thought distillation: Small models can also “think” step-by-step. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), Volume 1: Long Papers, 2665– 2679. Toronto, Canada.

Li, Y.; Yue, X.; Xu, Z.; Jiang, F.; Niu, L.; Lin, B. Y.; Ramasubramanian, B.; and Poovendran, R. 2025. Small models struggle to learn from strong reasoners. In Findings of the Association for Computational Linguistics: ACL 2025, 25366–25394. Vienna, Austria.

Li, Y.; Zuo, Y.; He, B.; Zhang, J.; Xiao, C.; Qian, C.; Yu, T.; Gao, H.-a.; Yang, W.; Liu, Z.; and Ding, N. 2026. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. arXiv:2604.13016.

Liu, X.; Wang, X.; Ma, Y.; Zhang, Y.; and Xiao, C. 2026. When are teacher tokens reliable? Position-weighted onpolicy self-distillation for reasoning. arXiv:2605.21606.

Mirzadeh, S. I.; Farajtabar, M.; Li, A.; Levine, N.; Matsukawa, A.; and Ghasemzadeh, H. 2020. Improved knowledge distillation via teacher assistant. In Proceedings of the Thirty-Fourth AAAI Conference on Artificial Intelligence (AAAI), 5191–5198. New York, NY.

OpenCompass. 2025. AIME 2025 dataset. Hugging Face Datasets.

Shenfeld, I.; Damani, M.; Hübotter, J.; and Agrawal, P. 2026. Self-distillation enables continual learning. arXiv:2601.19897.

Sheng, G.; Zhang, C.; Ye, Z.; Wu, X.; Zhang, W.; Zhang, R.; Peng, Y.; Lin, H.; and Wu, C. 2025. HybridFlow: A flexible and eficient RLHF framework. In Proceedings of the Twentieth European Conference on Computer Systems (EuroSys), 1279–1297. Rotterdam, Netherlands.

Wang, X.; Wei, J.; Schuurmans, D.; Le, Q. V.; Chi, E. H.; Narang, S.; Chowdhery, A.; and Zhou, D. 2023. Selfconsistency improves chain of thought reasoning in language models. In Proceedings ofthe 11th International Conference on Learning Representations (ICLR). Kigali, Rwanda.

Wang, Y.; Lu, S.; Gu, Y.; Wang, P.; Yang, Y.; Yan, Z.; Xie, C.; Wu, J.; and Yang, H. 2026. Not all disagreement is learnable: Token teachability in on-policy distillation. arXiv:2605.26844.

Washbourne, R. 2025. AMC 2023 dataset. Hugging Face Datasets.

Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Ichter, B.; Xia, F.; Chi, E. H.; Le, Q. V.; and Zhou, D. 2022. Chain-of-Thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems 35 (NeurIPS), 24824–24837. New Orleans, LA.

Xu, J.; Zhou, M.; Liu, W.; Liu, H.; Han, S.; and Zhang, D. 2025a. TwT: Thinking without tokens by habitual reasoning distillation with multi-teachers’ guidance. In Findings ofthe Association for Computational Linguistics: EMNLP 2025, 16475–16489. Suzhou, China.

Xu, W.; Han, R.; Wang, Z.; Le, L.; Madeka, D.; Li, L.; Wang, W. Y.; Agarwal, R.; Lee, C.-Y.; and Pfister, T. 2025b. Speculative knowledge distillation: Bridging the teacher–student gap through interleaved sampling. In Proceedings of the 13th International Conference on Learning Representations (ICLR). Singapore.

Xu, Z.; Jiang, F.; Niu, L.; Lin, B. Y.; and Poovendran, R. 2025c. Stronger models are not always stronger teachers for instruction tuning. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT), Volume 1: Long Papers, 4392–4405. Albuquerque, NM.

Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Gao, C.; Huang, C.; Lv, C.; Zheng, C.; Liu, D.; Zhou, F.; Huang, F.; Hu, F.; Ge, H.; Wei, H.; Lin, H.; Tang, J.; Yang, J.; Tu, J.; Zhang, J.; Yang, J.; Yang, J.; Zhou, J.; Zhou, J.; Lin, J.; Dang, K.; Bao, K.; Yang, K.; Yu, L.; Deng, L.; Li, M.; Xue, M.; Li, M.; Zhang, P.; Wang, P.; Zhu, Q.; Men, R.; Gao, R.; Liu, S.; Luo, S.; Li, T.; Tang, T.; Yin, W.; Ren, X.; Wang, X.; Zhang, X.; Ren, X.; Fan, Y.; Su, Y.; Zhang, Y.; Zhang, Y.;

B.; Sheng, G.; Tong, Y.; Zhang, C.; Zhang, M.; Zhang, W.;

Wan, Y.; Liu, Y.; Wang, Z.; Cui, Z.; Zhang, Z.; Zhou, Z.; and Qiu, Z. 2025. Qwen3 technical report. arXiv:2505.09388.

Ye, T.; Dong, L.; Wu, X.; Huang, S.; and Wei, F. 2026. On-policy context distillation for language models. arXiv:2602.12275.

Yu, P.; Xu, J.; Weston, J.; and Kulikov, I. 2024. Distilling system 2 into system 1. arXiv:2407.06023.

Y.; Wei, X.; Zhou, H.; Liu, J.; Ma, W.-Y.; Zhang, Y.-Q.; Yan, L.; Qiao, M.; Wu, Y.; and Wang, M. 2025. DAPO: An open-source LLM reinforcement learning system at scale. arXiv:2503.14476.

Zhang, K.; Tian, Y.; Zhao, D.; Li, Y.; Liu, Y.; Patel, V. M.; and Fu, D. 2026. On-policy distillation with best-of-N teacher rollout selection. arXiv:2605.09725.

Zhao, S.; Xie, Z.; Liu, M.; Huang, J.; Pang, G.; Chen, F.; and Grover, A. 2026. Self-distilled reasoner: On-policy selfdistillation for large language models. arXiv:2601.18734.