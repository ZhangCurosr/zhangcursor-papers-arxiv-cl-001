# Selective, Risk-Controlled Emotion Forecasting in Conversation: From a Persistence Barrier to Departure Detection with Conformal Guarantees

Bin Wen<sup>1</sup>, Dai-Qiao Zhang<sup>2</sup> and Tien-Ping Tan<sup>1</sup>

<sup>1</sup>School of Computer Sciences, Universiti Sains Malaysia, 11800 Gelugor, Penang, Malaysia <sup>2</sup>School of Mathematical Sciences, Universiti Sains Malaysia, 11800 Gelugor, Penang, Malaysia

Corresponding author: Tien-Ping Tan (e-mail: tienping@usm.my)

Abstract: Forecasting emotion in conversation is increasingly framed as predicting the next emotional state’s value. We show this targets a hard quantity in a state-observed testbed and propose a learnable, deployable reformulation instead. On valence-arousal-dominance trajectories from a dyadic corpus, repeating a speaker’s own previous state sets a persistence baseline that no frozen-encoder correction meaningfully improves upon and a state-space filter exceeds; the floor requires the historical state to be observed, since an estimated anchor collapses persistence below a direct forecaster. What survives is the timing of a departure from persistence, learnable on three of four corpora and near chance on the highest-base-rate corpus, a negative result we report. We recast the task as selective departure forecasting: a lead-time-weighted metric with a leakage protocol, and commitment policies whose best choice switches with the departure rate. A dialogue-level conforma risk control bounds the expected loss of committed predictions, meeting the target in all twelve configurations tested, though the guarantee is bought through abstention, with coverage and alarm recall falling sharply on the hardest corpus. One tested language model’s confidence fails as a ranking signal where another succeeds, a model-dependent failure not yet attributable to family versus scale. Code and result logs are available from the corresponding author upon request.

Keywords: afective computing, conformal prediction, emotion forecasting, selective prediction

## 1. Introduction

Emotion forecasting in conversation (EFC) asks a model to commit to a prediction about an emotional state that is not yet observable – the condition that separates it from the mature emotion-recognition literature [1], [2], [3], [4]. The emerging consensus formulation inherits recognition’s target: predict the value of the forthcoming state, whether a categorical label [5], [6], [7] or a point in valencearousal-dominance (VAD) space. The appeal is practical: a customer-service agent that anticipates a caller’s escalating frustration can intervene before the call is lost; a mentalhealth support chatbot that anticipates an afective downturn can adjust its response before distress is expressed; a tutoring system that anticipates disengagement can reengage a student pre-emptively. Across all three use cases, the underlying assumption is the same – that the value of the next emotional state is the right quantity to predict, and that a suficiently capable model will eventually predict it well.

This paper reports a systematic, audited attempt to build such value forecasters, and its central finding is that – within the state-observed testbed we study – the field may be aiming at the wrong quantity. On continuous VAD forecasting on IEMOCAP, where a historical emotionalstate value is available as the reference, a trivial rule – repeat what the same speaker was feeling last time they spoke – is not beaten by any frozen-encoder model we train, while a classical state-space (Kalman) baseline over the same anchor performs somewhat better. This result is explicitly conditional on observing that historical state: when the anchor must instead be estimated from raw content, its dominance disappears (Section 4), so we do not claim persistence is a good deployable value forecaster, only that residual value beyond an observed state is hard to predict. What is beatable, and more actionable, is a narrower question – not what the next emotional state will be, but whether it is about to change – learnable on three of the four corpora studied, with EmoryNLP providing a chancelevel negative result. Redirecting the target from value to occurrence yields a task that is both empirically solvable and deployable with formal statistical guarantees, which we develop and validate in this paper. We are explicit throughout about the boundary of the value-forecasting claim: it rests on frozen-encoder architectures with correction heads up to 3.7M parameters, an observed-state reference, and continuous-VAD evaluation on IEMOCAP, and does not rule out that full end-to-end fine-tuning or substantially diferent architectures could close the gap (Section 9).

The Persistence Barrier. On turn-level continuous VAD trajectories from IEMOCAP [8], the strongest zeroparameter baseline is not the conventional “repeat the previous turn” but same-speaker persistence: repeat the target speaker’s own previous state, skipping over the interlocutor’s intervening turn. This baseline reaches concordance correlation coeficient (CCC) 0 625/0 609/0 552 at forecast horizons t+1/t+2/t+3 – far above the turnlevel variant (0 446/0 549/0 426). A classical, causal, fewparameter same-speaker Kalman filter does slightly better still (0 662/0 647/0 603), while an AR(1) linear recalibration of the same anchor does not (0 560/0 539/0 471). No unconstrained direct value forecaster we train approaches either classical baseline: the best one attains CCC 0 140 at t+1. The controlled version of this comparison – an architecture that structurally anchors its output on the same gold same-speaker label persistence uses and only learns a gated correction on top – lands, unsurprisingly, close to the persistence floor by construction (0 629 versus 0 625 CCC at t+1), and a dedicated ablation shows that gated correction contributes nothing measurable beyond what the anchor already provides; this apples-to-apples result, not the raw gap, is what the paper’s learnability claim rests on. The gap for unconstrained models survives a 60× capacity sweep and four feature configurations up to multimodal RoBERTalarge plus WavLM audio; a psychologically structured Ornstein-Uhlenbeck (OU) dynamics model [9], also unconstrained, trails persistence by 6 5–11×. We name this the Persistence Barrier, sharpening the “forecasting floor” phenomenon reported for coarser afect series [10].

The Crack in the Barrier. The same sweeps that flatten value prediction leave exactly one signal intact: whether a departure from persistence is about to occur. Departure occurrence is predictable above base rate in fifteen of fifteen fold-model cells (AUPRC up to 0 305 versus a base rate of 0 237); departure magnitude and direction are not – direction accuracy at horizons t+2/t+3 falls to 0 42–0 44, below chance. The informative target is the break, not the level, and across the three corpora we test its predictability shows a consistent monotone association with the departure base rate, strongest exactly where departures are rare – the regime in which anticipation matters most.

From Forecasting to Selective Commitment. Because departures are rare and hard to predict, a deployable forecaster must also decide when to speak (Fig. 2). We study four commitment policies and find that none dominates: a cost-sensitive myopic policy wins at low departure base rates, a planning-based optimal-stopping policy wins at high rates, and a regime-adaptive meta-policy that estimates the local base rate matches the best single policy on every corpus tested. Reliability cannot be assumed: wrapping any policy in our dialogue-level conformal risk control (CRC) procedure yields a finite-sample guarantee that the expected loss of committed predictions on a test dialogue stays below a user-chosen risk level α, which we verify empirically in all twelve primary configurations, while exposing the coverage cost at which the guarantee is bought.

Contributions. (1) A unified selective departure forecasting formalization across discrete and continuous afect representations, with a lead-time-weighted metric (LT-F1) and a data-leakage test suite available from the corresponding author upon request. (2) The Persistence Barrier: a triply audited negative result, within a frozen-encoder testbed, with a partial annotator-artifact control. (3) A study of commitment policies establishing base-rate-dependent optimality, and a regime-adaptive meta-policy that recovers the per-regime optimum without oracle knowledge of the base rate. (4) A dialogue-level instantiation of conformal risk control for risk-controlled anticipation, together with an empirical analysis of the risk–coverage–recall tradeof it buys. (5) Three design principles distilled from audited negative results, including the double falsification of psychology-inspired dynamical priors as value forecasters and the model-dependent unreliability of LLM verbalized confidence as a ranking signal. All experiments run on a single consumer laptop; code, configuration files, and result logs are available from the corresponding author upon request.

## 2. Related Work

A recent survey [4] maps EFC as a target-absent task distinct from recognition, shift-aware recognition, and flip reasoning, and calls for lead-time-aware evaluation and leakage reporting; we cite it for that framing, not as evidence for our own novelty. Existing forecasters predict next-turn labels or distributions from history [11], [12], [13], [14], [5], [15], [6], [7], [16], including implicitreaction [17] and consequence [18] variants; Hi-EF [19] provides a dedicated benchmark, and calibration has been studied for recognition [20] but not forecasting. None of this published prior work reports a lead-time-weighted metric, an executable leakage protocol, selective commitment, or a dialogue-level risk guarantee – the pieces of evaluation and decision machinery this paper contributes on top of the value-vs-occurrence reformulation. Forecasting conversational derailment introduced unknown-horizon commitment [21] and, recently, an explicit “when to trigger” decision mechanism [22]; we transfer the commitment question to afective targets, replace heuristic rules with cost-sensitive, planning-based, and base-rate-adaptive policies, and add distribution-free risk control. Selective prediction has a long line of foundational work [23], [24], with temperature scaling as the standard calibration baseline [25]. Conformal risk control (CRC) extends conformal prediction to bounding expected losses of thresholded predictors [26], including a corollary for non-monotone losses via sufix-max monotonization; selective variants have recently been unified [27], and further non-monotone extensions are emerging [28]. We do not claim new CRC theory: our contribution is its instantiation on a dialogue-level exchangeable unit with a ratio-type selective-forecasting loss and abstention convention, plus the empirical risk–coverage–recall analysis of what that guarantee costs. Psychology models emotion as an inertial, mean-reverting process with interpersonal coupling [29], [9], [30]; we test these priors computationally and report their double falsification for forecasting, while confirming the internal importance ordering of their components. Latent world models [32] have recently been extended with afect [33], [34], [35]; none of this work addresses target-absent selective forecasting.

![](images/1ee44f34d7696164721d63360f92f780eb14dd69168097c5f280cfe0e6fba610.jpg)  
Figure 1. The Persistence Barrier (IEMOCAP, 5-fold session-held-out, CCC averaged over valence/arousal/dominance). Solid bars: zero/fewparameter baselines; hatched bars: trained value forecasters. Both classical same-speaker baselines dominate every trained model at every horizon; free-form latent dynamics sits at essentially zero CCC.

![](images/5f7c326448cf4762a8eba041b30dc826bb67d68b60f25fa8fae8006c07412945.jpg)  
Figure 2. Method overview: same-speaker persistence anchor → departure event → selective commitment → dialogue-level conformal risk control.

## 3. Selective Departure Forecasting

A conversation is a sequence of utterances $u _ { 1 } , \ldots , u _ { T }$ with speakers $s _ { t }$ and annotations y (a categorical label, or a vector in VAD space). Let $C _ { t } = ( u _ { 1 } , \ldots , u _ { t } )$ . A forecast at time t is any prediction about $y _ { t ^ { \prime } } , \ t ^ { \prime } \ > \ t ,$ computed from $C _ { t }$ alone: no token, audio frame, or annotation from u ′ may enter the input. All feature windows, normalization statistics, and thresholds are fit on training folds only; a test suite asserts this boundary programmatically (56 tests).

Is the future speaker’s identity known? The boundary above excludes future content but is silent on future turntaking structure: our implementation assumes the identity $s _ { t ^ { \prime } }$ of the speaker at the target position is known at commit time t, since it selects which same-speaker anchor π(t<sup>′</sup>) the forecast is measured against. This is essentially free on

DailyDialog (a strictly alternating two-party corpus), a real but modest assumption on the dyadic MELD subset used in our main tables (25 6% of turns there are followed by the same speaker), and most consequential on the full multiparty corpora: an order-2 Markov next-speaker predictor, fit on speaker sequences alone, reaches only 52 7%/56 7% top-1 accuracy on MELD/EmoryNLP test dialogues. We report this as an explicit scope boundary: this paper establishes that given the future speaker’s identity, predicting whether they will depart from their own persistence is learnable; forecasting jointly over an unknown future speaker and their departure is a harder problem we leave open.

Let π(t) index speaker $s _ { t } \mathbf { \vec { s } }$ most recent prior utterance. The same-speaker persistence forecast is $\hat { y } _ { t } ~ = ~ y _ { \pi ( t ) }$ . A departure is the event that reality deviates from it:

$$
D _ { t } = { \bf 1 } [ y _ { t } \neq y _ { \pi ( t ) } ] \mathrm { ~ } ( \mathrm { d i s c r e t e } ) , \quad D _ { t } ^ { ( q ) } = { \bf 1 } [ \| y _ { t } - y _ { \pi ( t ) } \| > \tau _ { q } ] ,\tag{1}
$$

with $\tau _ { q }$ the q-th percentile of training-fold deviations (continuous case). This unifies discrete emotion shift and continuous change, and makes persistence – not an arbitrary model – the reference point a forecaster must beat. A selective forecaster is a pair $( f , \rho ) \colon$ a base model emitting $\hat { P } ( D _ { t ^ { \prime } } \mid C _ { t } )$ at each decision point, and a commitment policy ρ that either commits or waits. Committing at t against target $t ^ { \prime }$ yields lead time $\Delta t = t ^ { \prime } - t$

Lead-Time-Weighted Evaluation. Let TP be committed, correct departure predictions (each with lead $\Delta t _ { i } ) .$ , FP committed false alarms, FN missed true departures, and $| C | = | \mathrm { T P } | + | \mathrm { F N } |$ . With $w ( \Delta t ) = \operatorname* { m i n } ( \Delta t / \tau , 1 )$ and horizon τ = 3,

$$
P = { \frac { | \mathrm { T P } | } { | \mathrm { T P } | + | \mathrm { F P } | } } , \qquad R _ { L T } = { \frac { 1 } { | C | } } \sum _ { i \in \mathrm { T P } } w ( \Delta t _ { i } ) , \qquad \mathrm { L T - F } 1 = { \frac { 2 P R _ { L T } } { P + R _ { L T } } } .\tag{2}
$$

A perfect but latest-possible forecaster (∆t=1) earns $w { = } 1 / 3 ;$

TABLE I. Same-speaker persistence, oracle vs. estimated anchor (IEMOCAP, 5-fold, mean CCC over V/A/D). “Estimated” substitutes a same-turn frozen-encoder probe’s own output for the gold anchor value.
<table><tr><td>Anchor</td><td>t+1</td><td> $t { + } 2$ </td><td> $t { + 3 }$ </td></tr><tr><td>Oracle (gold label)</td><td>0.625</td><td>0.609</td><td>0.552</td></tr><tr><td>Estimated (same-turn probe)</td><td>0.071</td><td>0.069</td><td>0.058</td></tr></table>

earlier correct commitments earn proportionally more, capped at τ; indiscriminate alarming is penalized through P. On MELD, the optimal-stopping policy’s advantage over the calibrated threshold is stable across $\tau \in \{ 2 , 3 , 5 \}$ $( + 3 . 6 / + 6 . 8 / + 6 . 7$ points). On DailyDialog it is not: the calibrated threshold edges out the meta-policy by 0 3 points at τ=2 (statistically indistinguishable), but the meta-policy leads by 1 7–1 8 points at τ=3 5.

## 4. The Persistence Barrier

An input asymmetry. The same-speaker persistence baseline’s prediction is the gold label $y _ { \pi ( t ) }$ . The unconstrained value forecasters it is compared against (GRU multihorizon, OU-coupled dynamics, free-form latent dynamics) receive no such label history as input – only frozen text/audio embeddings and a speaker identity per turn, never the numeric VAD trajectory itself. Part of the headline gap is therefore attributable to persistence having direct access to the exact quantity being predicted, one step removed. Audit 1 below is the controlled version – a persistenceanchored architecture given the gold anchor $y _ { \pi ( t ) }$ directly, with only its additional learned correction tested – and it is this audit, not the raw gap, that the paper’s learnability claim rests on.

What if the anchor itself has to be estimated? Every number above assumes the anchor $y _ { \pi ( t ) }$ is available as ground truth at commit time. We quantify this oracle assumption directly: a minimal same-turn probe (a small frozen-encoder regression head, no dialogue history) is trained to predict a turn’s own gold VAD from its own content, and its output substituted for the gold anchor. The probe itself is weak (mean validation CCC 0 130), and substituting it collapses same-speaker persistence’s CCC from 0 625/0 609/0 552 (oracle anchor) to just 0 071/0 069/0 058 (estimated anchor, Table I) – below every unconstrained trained model in Table II except freeform dynamics. The entire Persistence Barrier result is therefore conditional on oracle access to the historical gold label, and Table II’s framing should be read as stateobserved emotion forecasting, not as a claim about a fully realistic pipeline that must also recover that state from raw signal.

Audit 1 (ablation). An architecture that anchors its prediction on same-speaker persistence and learns only a gated correction matches its own frozen-head control (t+1: 0 6253 frozen vs. 0 6288 trained) to within 0 001–0 02 CCC per fold: the correction head contributes nothing measurable beyond the persistence anchor it is built on.

TABLE II. The Persistence Barrier (IEMOCAP, 5-fold session-heldout, CCC over valence/arousal/dominance).
<table><tr><td>Predictor</td><td>t+1</td><td>t+2</td><td> $t { + 3 }$ </td></tr><tr><td>Same-speaker persistence (0 par.)</td><td>0.625</td><td>0.609</td><td>0.552</td></tr><tr><td>Same-speaker Kalman filter (1 hyperpar.)</td><td>0.662</td><td>0.647</td><td>0.603</td></tr><tr><td>Same-speaker AR(1) (2 par.)</td><td>0.560</td><td>0.539</td><td>0.471</td></tr><tr><td>Turn-level persistence (0 par.)</td><td>0.446</td><td>0.549</td><td>0.426</td></tr><tr><td>Linear extrapolation (0 par.)</td><td>0.186</td><td>0.201</td><td>0.051</td></tr><tr><td>Unconstrained direct value forecasters:</td><td></td><td></td><td></td></tr><tr><td>GRU multi-horizon</td><td>0.140</td><td>0.130</td><td>0.126</td></tr><tr><td>OU + interpersonal coupling</td><td>0.096</td><td>0.087</td><td>0.049</td></tr><tr><td>Free-form latent dynamics</td><td>-0.005</td><td>-0.005</td><td>-0.002</td></tr><tr><td>Persistence-anchored trained model (t+1 only, Audit 1):</td><td></td><td></td><td></td></tr><tr><td>Frozen correction head (zero)</td><td>0.625</td><td></td><td></td></tr><tr><td>Trained correction head (full)</td><td>0.629</td><td></td><td></td></tr></table>

Audit 2 (capacity). Scaling the anchored model from $6 . 2 \times 1 0 ^ { 4 }$ to $3 . 7 \times 1 \bar { 0 } ^ { 6 }$ parameters moves 3-seed CCC by at most 0 0015 per dimension.

Audit 3 (features). Across RoBERTa-base/-large [36], each with and without WavLM [37] audio, the head’s endto-end CCC increment is noise at t+1 and significantly negative at t+2/t+3, and conditional direction accuracy falls below chance (0 42–0 44, Fig. 3).

Ruling out an annotation artifact – partially. IEMO-CAP’s sequential human VAD ratings [8] raise a confound: an annotator’s self-anchoring, not real emotion dynamics, could drive the persistence floor. We test this directly by predicting one evaluator’s rating from a diferent evaluator’s rating of the same speaker’s prior utterance – self-anchoring is physically impossible across annotators. Table III reports this cross-annotator channel against a same-turn agreement ceiling. For valence, cross-annotator persistence retains 85 1% of the agreement ceiling (CCC 0 587 against a ceiling of 0 689 at gap 1, decaying to 73 9% at gap 3): the signal is carried by what independent observers perceive, and self-anchoring is not a plausible explanation for the valence floor. For arousal and dominance the picture is materially weaker (ratio 39 9% and 55 3% at gap 1, falling to 13 1% and 38 5% at gap 3), so we do not claim the artifact hypothesis is ruled out for those dimensions – only substantially weakened for valence, which also carries the largest share of the barrier’s magnitude.

The Crack. Departure occurrence escapes the barrier: at threshold q=75, AUPRC exceeds the base rate (0 237) in fifteen of fifteen fold-model cells, up to 0 305 with audio. Occurrence is learnable; magnitude and direction are not, under any tested configuration. The rest of the paper builds a forecasting system on the learnable side of this boundary.

![](images/d42147643f1acafeb5e7eb02f5284308d8e5b0cab36aacb56a24a4864ef0a7ba.jpg)

![](images/96346061dc730ab7f44366a98a77e6949ca5f59e73af3a024f4c029d302a2d19.jpg)

![](images/4c20798c23a6f66d9b967a20f48f60327827def96f0b07380607fee90193e907.jpg)  
Figure 3. Three audits. (a) Freezing the correction head changes CCC by at most 0 02 on all folds. (b) A 60× capacity sweep moves CCC in the third decimal (3 seeds). (c) Across four feature configurations, the head’s end-to-end CCC increment is negative with a bootstrap CI excluding zero in 23 of 24 cells at t+2/t+3.

TABLE III. Annotator-artifact control (IEMOCAP, same-speaker gap 1, mean over V/A/D unless noted). Cross-annotator retains 68 4% of the ceiling on average, concealing a per-dimension spread of 39 9– 85 1% at gap 1.
<table><tr><td>Channel (gap= 1)</td><td>CCC</td><td>n pairs</td><td>Ratio to ceiling</td></tr><tr><td>Mean-label persistence</td><td>0.627</td><td>9737</td><td></td></tr><tr><td>Within-annotator</td><td>0.620</td><td>20786</td><td></td></tr><tr><td>Cross-annotator</td><td>0.281</td><td>24916</td><td>68.4%</td></tr><tr><td>Agreement ceiling</td><td>0.411</td><td>12456</td><td>100%</td></tr></table>

TABLE IV. Commitment policies, LT-F1 (3-seed mean). a: calibrated threshold; b: cost-sensitive myopic; c: derailment-style delay rule; d: optimal stopping.
<table><tr><td>Dataset</td><td>Model</td><td>a</td><td>b</td><td>C</td><td>d</td></tr><tr><td rowspan="2">DailyDialog</td><td>GRU-MH</td><td>0.337</td><td>0.343</td><td>0.336</td><td>0.304</td></tr><tr><td>PsyDyn</td><td>0.248</td><td>0.265</td><td>0.256</td><td>0.210</td></tr><tr><td rowspan="2">MELD</td><td>GRU-MH</td><td>0.542</td><td>0.552</td><td>0.543</td><td>0.635</td></tr><tr><td>PsyDyn</td><td>0.659</td><td>0.702</td><td>0.642</td><td>0.702</td></tr></table>

## 5. Commitment Policies

We compare four policies (Table IV). Paired dialoguelevel bootstrap (1,000 resamples) confirms the split: the optimal-stopping policy d beats the calibrated threshold on MELD by +9 3 points (GRU-MH) and +4 3 point (PsyDyn), yet loses on DailyDialog by 3 3–3 8 points, both with confidence intervals excluding zero. No policy dominates.

Base rate is the switch. A simpler adaptive policy that thresholds directly on an online estimate of the local departure rate $\rho _ { t }$ recovers the better regime at each extreme: at low base rate (DailyDialog, 16 5%) it beats the fixed threshold by +5 1 points, with a further +5 7 points available to an oracle-ρ upper bound; at high base rate (MELD,

56 7%) the oracle bound collapses to +0 1 points – the local-rate feature saturates, and multi-step planning takes over. Within-dataset stratification complicates the story: inside DailyDialog, optimal stopping loses most heavily in the highest-shift-rate stratum (−13 7 points), suggesting an interaction between base rate and confidence-trajectory quality.

A regime-adaptive meta-policy. We train a small gate $g ( \rho _ { t } ) \in [ 0 , 1 ]$ on an online estimate of the local departure rate, acting as the optimal-stopping policy with probability g and the cost-sensitive policy otherwise. On DailyDialog the pooled meta-policy attains $\mathrm { L T - F 1 } = 0 . 3 1 0 .$ , ahead of the pooled best single policy (0 304) by +0 6 points; on MELD it matches the pooled best single policy within 0 06 points. Zero-shot gate transfer across corpora is unreliable, so per-corpus calibration remains required; an oracle-g bound retains +1 7–+2 8 points of headroom. CRC-wrapping the meta-policy keeps mean risk at or below target in all 12 cells on the main pipeline.

Mechanism evidence for policy-model co-design. Increasing decision points from 3 to 5 systematically narrows the gap between the optimal-stopping and myopic policies in both directions (DailyDialog: $- \bar { 4 } . 9 \  \ - \bar { 2 } . 3$ points; MELD: +11 2 → +6 9 points), i.e., the planning advantage is largest when decisions are coarse. Architectures also exhibit divergent earliness-accuracy profiles: shared-rollout heads self-select easy early commitments and gain with lead time in all 3 seeds, while independent multi-horizon heads gain far less and are sometimes more accurate at the last moment (Fig. 5) – a policy tuned for one confidencetrajectory shape can be actively harmful for the other.

## 6. Dialogue-Level Risk Control

Given any fixed selective forecaster, we seek a distribution-free guarantee on the error rate of committed predictions. Turns within a dialogue are strongly dependent, so we calibrate at the dialogue level. For dialogue d with decision points $i = 1 , \ldots , k _ { d }$ , base decisions committed<sub>i</sub>, confidences $c _ { i } ~ \in ~ \left[ \frac { 1 } { 2 } , 1 \right]$ , and correctness corr , define a confidence veto submit (λ) = committed $\wedge \ ( c _ { i } \ \geq \ \lambda )$ and the ratio loss

![](images/bfd161a0251448d8f1481142686f6792cb7a42328468134140acd329d2595540.jpg)

![](images/e8c6be4d7176c81ef04b4df79cc4fe36616d99978115fb4ce9dc2393b8cf1e52.jpg)  
Figure 4. Base rate is the switch, but not the whole story. Left: a rate-thresholded adaptive policy wins at low departure rates (+5 1 points over the fixed threshold on DailyDialog) and saturates at high rates, where optimal stopping takes over. Right: within DailyDialog, optimal stopping loses most heavily in the highest-shift stratum (−13 7 points, CI excluding zero).

![](images/88a43ea2b7e716a9d393d062b2de3dc73989f9fbfae84018a632391f0dcba72e.jpg)  
Figure 5. Opposite earliness-accuracy profiles across architectures $( 3$ seeds, mean line with seed-range shading, DailyDialog, single seed set).

$$
L ( d , \lambda ) = \frac { \# \{ i : \mathrm { s u b m i t } _ { i } ( \lambda ) , \neg \mathrm { c o r r } _ { i } \} } { \# \{ i : \mathrm { s u b m i t } _ { i } ( \lambda ) \} } \in [ 0 , 1 ] ,\tag{3}
$$

with $L = 0$ if the dialogue fully abstains. Although submission sets are nested in λ, the ratio loss is not monotone: raising λ can first evict a correct low-confidence submission and raise the error ratio. We therefore work with the perdialogue envelope loss $\tilde { L } ( d , \lambda ) = \mathrm { m a x } _ { \lambda ^ { \prime } \geq \lambda } L ( d , \lambda ^ { \prime } )$ , which is non-increasing in λ and dominates L pointwise.

Proposition 1 (Dialogue-level CRC). Assume calibration dialogues $d _ { 1 } , \ldots , d _ { n }$ and a test dialogue $d _ { n + 1 }$ are exchangeable, and let $\begin{array} { r c l } { \tilde { R } ( \lambda ) } & { = } & { \frac { 1 } { n } \sum _ { j \leq n } \tilde { L } ( d _ { j } , \lambda ) } \end{array}$ over a finite grid $\Lambda \subset \ [ \frac { 1 } { 2 } , 1 + \varepsilon ]$ for some $\varepsilon \ > \ 0 .$ Let $\hat { \lambda } ~ = ~ \operatorname* { i n f } \{ \lambda ~ \in ~ \Lambda ~ :$

$\begin{array} { r } { \frac { n } { n + 1 } \tilde { R } ( \lambda ) + \frac { 1 } { n + 1 } \leq \alpha \} } \end{array}$ (and $\hat { \lambda } = \operatorname* { m a x } \Lambda ,$ forcing full abstention, if no such λ exists). Then $\mathbb { E } [ L ( d _ { n + 1 } , \hat { \lambda } ) ] \leq \mathbb { E } [ \tilde { L } ( d _ { n + 1 } , \hat { \lambda } ) ] \leq \alpha .$

We state this as a Proposition, not a new theorem: it is a direct instantiation of the standard conformal-risk-control guarantee for monotone (or monotonized) losses [26] – whose own non-monotone corollary already prescribes exactly the sufix-max monotonization L<sup>˜</sup> we use – applied to our dialogue-level exchangeable unit and ratio loss; the proof follows that argument and is included in Appendix A. What is specific to this paper is not the risk-control mechanism but its target: the dialogue-equal ratio loss, its abstention convention, its use inside a selective departureforecasting pipeline, and the empirical risk–coverage–recall analysis of what the guarantee costs; the general nonmonotone and selective-CRC theory is prior and concurrent work [27], [28]. The commitment policy is fit on validation dialogues only, and every CRC calibration/evaluation split is drawn exclusively from test dialogues that never participated in that fitting, as the Proposition’s exchangeability assumption requires; MELD’s dyadic test split has only 68 usable dialogues under this restriction, so its confidence intervals are correspondingly wide.

Empirical verification and the price of the guarantee. Across 2 datasets × 2 base models $\times ~ \alpha ~ \in ~ \{ 0 . 1 , 0 . 2 , 0 . 3 \}$ (Table V), the mean empirical risk across 20 test-only splits is at or below target in 12 of 12 cells. This is a claim about the mean over repeats, not about every individual repeat: on MELD’s small test split, several individual repeats exceed α even where the mean does not. The guarantee comes at a measurable coverage cost – 16–77% on DailyDialog, only 2–11% on MELD – since the procedure partly controls risk through full abstention. Conditional risk (restricted to dialogues with at least one commitment) is substantially higher than the macro risk wherever abstention is heavy: on MELD, 0 51–0 57 against a macro risk of only 0 08–0 14, roughly a 7× gap. “Mean risk” is what the Proposition’s guarantee is about; “conditional risk” is what a user should expect the next time the system actually says something, and on MELD’s small test split it is not small. In the continuous setting the trade-of is nearly binary (Fig. 6, right): quantilerecalibrating α jumps coverage from under 5% directly to 99 5%, with no tunable middle ground.

TABLE V. Dialogue-level CRC, test-only calibration/evaluation splits (20 repeats per cell, per-dialogue-envelope calibrator). “Frac. $\leq \alpha ^ { \prime \prime }$ is the fraction of repeats individually at or below target; mean risk is at or below α in all 12 cells. “Cond. risk” restricts to dialogues with at least one committed prediction (not what the Proposition bounds)
<table><tr><td>Dataset</td><td>Model</td><td>α</td><td>Mean risk</td><td>95% CI</td><td>Cond. risk</td><td>Frac. ≤ α</td><td>Coverage</td></tr><tr><td rowspan="3">DailyDialog</td><td rowspan="3">GRU-MH</td><td>0.1</td><td>0.086</td><td>[0.061,0.113]</td><td>0.154</td><td>0.85</td><td>0.169</td></tr><tr><td>0.2</td><td>0.166</td><td>[0.127,0.209]</td><td>0.217</td><td>0.95</td><td>0.308</td></tr><tr><td>0.3</td><td>0.247</td><td>[0.212,0.287]</td><td>0.279</td><td>1.00</td><td>0.471</td></tr><tr><td rowspan="3">DailyDialog</td><td rowspan="3">PsyDyn</td><td>0.1</td><td>0.074</td><td>[0.050,0.097]</td><td>0.140</td><td>1.00</td><td>0.160</td></tr><tr><td>0.2</td><td>0.156</td><td>[0.133,0.185]</td><td>0.179</td><td>1.00</td><td>0.388</td></tr><tr><td>0.3</td><td>0.203</td><td>[0.182,0.225]</td><td>0.206</td><td>1.00</td><td>0.773</td></tr><tr><td rowspan="3">MELD</td><td rowspan="3">GRU-MH</td><td>0.1</td><td>0.077</td><td>[0.000,0.184]</td><td>0.570</td><td>0.75</td><td>0.021</td></tr><tr><td>0.2</td><td>0.141</td><td>[0.043,0.206]</td><td>0.511</td><td>0.90</td><td>0.042</td></tr><tr><td>0.3</td><td>0.144</td><td>[0.073,0.206]</td><td>0.516</td><td>1.00</td><td>0.043</td></tr><tr><td rowspan="3">MELD</td><td rowspan="3">PsyDyn</td><td>0.1</td><td>0.060</td><td>[0.000,0.177]</td><td>0.150</td><td>0.90</td><td>0.049</td></tr><tr><td>0.2</td><td>0.165</td><td>[0.059,0.294]</td><td>0.248</td><td>0.50</td><td>0.085</td></tr><tr><td>0.3</td><td>0.252</td><td>[0.147,0.353]</td><td>0.310</td><td>0.80</td><td>0.113</td></tr></table>

![](images/7596547ad0d98545adc2d79c50ce9dc90bc57cad66767e94caac9952f8fa9469.jpg)

![](images/eb63e29dacb4762b22ca810943222336adce939650134f0a88f0f302c8d1bc60.jpg)  
Figure 6. Risk-coverage. Left: discrete tasks admit a tunable trade-of. Right: in the continuous setting the trade-of is nearly binary – quantile recalibrated α jumps coverage from under 5% directly to 99 5%.

Risk on the alarms that matter. Restricting to committed departure alarms, a threshold recalibrated directly against the alarm-only loss achieves mean risk at or below α in 12 of 12 cells as well, so a practitioner who cares specifically about alarm risk gets the same style of guarantee by calibrating for it directly rather than reusing the pooled threshold. The cost is a severe recall penalty: on DailyDialog/GRU-MH, base recall 0 656 becomes deployment recall 0 062/0 162/0 262 at α=0 1/0 2/0 3; on MELD/GRU-MH, base recall 0 530 becomes 0 016/0 036/0 092. A risk-controlled alarm system on these corpora is a high-precision, very-low-recall device: right about the few departures it commits to, but silent on most of them – a trade-of, not risk alone, that a deployer must weigh.

## 7. Experiments

## A. Setup

We evaluate on four corpora (Table VI): DailyDialog [39] (13,118 dialogues; departure rate 17 3%), MELD [40] (1,432 dialogues; 55 4%), IEMOCAP [8] (151 dyadic dialogues, continuous VAD, 5-fold session-held-out), and EmoryNLP [41] (897 scenes; 64 6%); Hi-EF [19] (53 dialogues) is reported separately in supplementary material available from the corresponding author upon request due to its small size. Because only PsyDyn’s interpersonalcoupling mechanism currently supports more than two speakers, Tables IV, V, and VIII evaluate all four model families on MELD’s dyadic subset (643 of 1,432 dialogues, filtered to exactly two distinct speakers) using the same checkpoints and data loader; Table VII additionally reports PsyDyn on the full multi-party MELD via a separately trained mean-pooled coupling variant, marked with a dagger and not a same-data comparison against the other three families. Base models are GRU direct, GRU multi-horizon, freeform latent dynamics, and OU-coupled dynamics (PsyDyn), all built over frozen encoders (RoBERTa-base/-large [36]; WavLM audio [37] where available); 3 seeds throughout. Splits are dialogue-level; leakage is asserted by a test suite.

TABLE VI. Dataset statistics. All splits are dialogue-level, verified by a leakage test suite.
<table><tr><td>Dataset</td><td>Dialogues (tr/val/te)</td><td>Turns</td><td>Shift rate</td></tr><tr><td>DailyDialog</td><td>13118 (10494/1311/1313)</td><td>102979</td><td>0.173</td></tr><tr><td>MELD (mp)</td><td>1432 (1145/143/144)</td><td>13708</td><td>0.554</td></tr><tr><td>Hi-EF</td><td>53 (42/5/6)</td><td>4779</td><td>0.640</td></tr><tr><td>EmoryNLP</td><td>897 (717/89/91)</td><td>12606</td><td>0.646</td></tr><tr><td>IEMOCAP (5-fold CV)</td><td>151</td><td>10039</td><td>N/A (VAD)</td></tr></table>

All experiments ran on one consumer laptop (CPU/MPS).

## B. Occurrence Learnability Tracks the Base Rate Across Corpora

Shift-F1 against a majority-class base rate is misleading on EmoryNLP: an always-positive rule scores shift-F1 0 76 with zero ranking ability, a base-rate artifact. We instead read occurrence learnability with threshold-free AUROC (Table VIII), which shows a monotone association across the three corpora we test: AUROC drops from 0 72 to 0 56 to 0 50 as the base rate climbs from 0 17 to 0 55 to 0 65 (DailyDialog / MELD / EmoryNLP). We describe this as a cross-corpus empirical association, not a law: three data points cannot establish a causal or statistical law, and the three corpora difer simultaneously in label taxonomy, source domain, dialogue length, speaker count, and annotation protocol. Within this scope, occurrence is most learnable in the low-base-rate regime where anticipation matters most, and the one chance-level corpus (EmoryNLP) is consistent with, rather than a counterexample to, the trend.

## C. Architecture Dependence, Abstention, and LLM Confidence

A shallow, non-recurrent baseline (logistic regression / small MLP over a single frozen embedding, no dialogue history, no recurrence) trained on the true same-speaker anchor remains below every trained architecture on DailyDialog (AUROC 0 672/0 643 versus 0 687–0 720), but exceeds every trained architecture on MELD (logistic regression AUROC 0 587, MCC 0 150, versus 0 516–0 563/0 031– 0 091). The trained architectures’ advantage over a simple classifier is itself base-rate- and corpus-dependent – real but modest on DailyDialog, and on MELD, where the departure rate is high and the occurrence signal is weak for everyone, the added complexity does not clearly pay for itself. At 70% coverage, calibrated abstention improves shift-F1 in 27 of 28 model-dataset-capacity cells across a base/512/1024 parameter sweep on DailyDialog/MELD (+1 35 to +5 38 points) and 3 of 4 cells on EmoryNLP; the lone exception, free-form dynamics on EmoryNLP (−6 1 points), is exactly the chance-AUROC corpus. Full extended tables for this sweep are available from the corresponding author upon request.

Two instruction-tuned open-weight LLMs from diferent parameter-count families (qwen3:14b and mistral:7b, served locally, no external API calls) are prompted zeroshot to predict whether the named target speaker’s next turn will depart from their own previous state, together with a self-reported confidence; the 14B model was queried on the full DailyDialog/MELD test sets and the 7B model on a fixed 500-target subsample (Table IX, Fig. 7). Temperature scaling improves expected calibration error (ECE) for both models by 65%/29% (14B) and 22%/33% (7B), but the two diverge sharply on ranking: abstention by the 14B model’s confidence hurts in 6 of 6 cells (−3 7 to −16 2 points); by the 7B model’s, it helps in 6 of 6 (+0 7 to +3 3 points). Because the two models difer simultaneously in parameter count, model family, and evaluation-set size, scale and family are confounded, and we state the finding as model-dependent, not family-dependent: at least one widely used LLM fails this ranking test, while at least one other passes it. A replication under error-framing and a twostep decoupled confidence protocol (supplementary material available from the corresponding author upon request) reproduces the same split between the two models, ruling out a single-protocol artifact as the explanation. Trained forecasters, by contrast, are positive in every cell we test.

(a) Calibration improves for both  
![](images/bc18c3a2a6cbd7443918d6768c67014da38bffe70dd04e6242ca4ab23eaa818f.jpg)

(b) ...ranking is model-dependent  
![](images/6795703641c849f3c437e37c1dc6e9cb73414f40962ea5b8e4ec480cb4805e2b.jpg)  
Figure 7. Calibrated confidence does not imply reliable ranking, for at least one of the two LLMs tested. Top: temperature scaling improves ECE for both. Bottom: abstention by the 14B model’s confidence hurts in 6/6 cells; by the 7B model’s, it helps in 6/6.

Finally, the specific discretized OU-plus-coupling parameterization we implement underperforms an encodermatched GRU baseline both discretely (−4 7 points shift-F1) and continuously (worse change-segment CCC in 15 of 15 fold-horizon cells). We do not read this as falsifying the psychological theory it is inspired by [9], [30] – only this specific implementation as a value forecaster; its internal ablation ordering is stable and consistent with the theory’s own emphasis, since removing interpersonal coupling hurts far more than sharing the inertia parameter across speakers.

TABLE VII. Departure forecasting across corpora (shift-F1). MELD is split into its dyadic subset (all four families) and the full multi-party corpus (PsyDyn only, marked †). EmoryNLP is a 3-seed mean±std; bold marks the best departure forecaster where its shift-F1 exceeds Random@prior on the same pool.
<table><tr><td>Model</td><td>DailyDialog</td><td>MELD (dyadic)</td><td>MELD (multi-party)†</td><td>EmoryNLP (3-seed)</td></tr><tr><td>Random@prior</td><td>0.179</td><td>0.581</td><td>0.581†</td><td> $0 . 6 0 9 \pm 0 . 0 2 5$ </td></tr><tr><td>GRU direct</td><td>0.393</td><td>0.685</td><td>_†</td><td> $\mathbf { 0 . 7 3 7 \pm 0 . 0 3 1 }$ </td></tr><tr><td>GRU multi-horizon</td><td>0.387</td><td>0.697</td><td>_†</td><td> $0 . 6 7 3 \pm 0 . 0 8 8$ </td></tr><tr><td>Free-form dynamics</td><td>0.389</td><td>0.710</td><td>_†</td><td> $0 . 6 7 3 \pm 0 . 1 5 0$ </td></tr><tr><td>OU-coupled (PsyDyn)</td><td>0.371</td><td>0.688</td><td>0.665†</td><td> $0 . 6 6 5 \pm 0 . 1 1 7$ </td></tr></table>

TABLE VIII. Threshold-free occurrence learnability (3 seeds each). AUROC/MCC are 0.5/0 for both baselines by construction; EmoryNLP is indistinguishable from chance across the full model family.
<table><tr><td>Dataset</td><td>Model</td><td>AUROC [range]</td><td>MCC</td></tr><tr><td>DailyDialog</td><td>GRU-direct</td><td>0.720 [0.717,0.726]</td><td>0.247</td></tr><tr><td>DailyDialog</td><td>GRU-MH</td><td>0.716 [0.713,0.721]</td><td>0.239</td></tr><tr><td>DailyDialog</td><td>Free-form</td><td>0.720 [0.719,0.722]</td><td>0.242</td></tr><tr><td>DailyDialog</td><td>PsyDyn</td><td>0.687 [0.683,0.689]</td><td>0.220</td></tr><tr><td>MELD</td><td>GRU-direct</td><td>0.529 [0.497,0.556]</td><td>0.031</td></tr><tr><td>MELD</td><td>GRU-MH</td><td>0.516 [0.492,0.545]</td><td>0.057</td></tr><tr><td>MELD</td><td>Free-form</td><td>0.541 [0.507,0.561]</td><td>0.033</td></tr><tr><td>MELD</td><td>PsyDyn</td><td>0.563 [0.549,0.585]</td><td>0.091</td></tr><tr><td>EmoryNLP</td><td>GRU-direct</td><td>0.495 [0.488,0.506]</td><td>0.028</td></tr><tr><td>EmoryNLP</td><td>GRU-MH</td><td>0.504 [0.485,0.542]</td><td>-0.013</td></tr><tr><td>EmoryNLP</td><td>Free-form</td><td>0.497 [0.490,0.501]</td><td>-0.023</td></tr><tr><td>EmoryNLP</td><td>PsyDyn</td><td>0.463 [0.456,0.475]</td><td>-0.011</td></tr></table>

TABLE IX. Both LLMs tested: full metrics. The 14B model’s ranking deltas are negative in 6 of 6 cells; the 7B model’s are positive in 6 of 6 cells.
<table><tr><td>Family</td><td>Dataset</td><td>Shift-F1</td><td>ECE raw</td><td>ECE cal.</td><td>∆50%</td><td>∆70%</td><td>∆90%</td></tr><tr><td>14B</td><td>DailyDialog</td><td>0.101</td><td>0.084</td><td>0.029</td><td>-8.84</td><td>-9.22</td><td>-5.48</td></tr><tr><td>14B</td><td>MELD</td><td>0.334</td><td>0.368</td><td>0.261</td><td>-16.24</td><td>-5.22</td><td>-3.72</td></tr><tr><td>7B</td><td>DailyDialog</td><td>0.274</td><td>0.704</td><td>0.549</td><td>+3.26</td><td>+3.26</td><td>+0.85</td></tr><tr><td>7B</td><td>MELD</td><td>0.719</td><td>0.330</td><td>0.221</td><td>+1.62</td><td>+1.49</td><td>+0.67</td></tr></table>

## 8. Design Principles

P1: When a reliable historical emotion-state observation is available, report persistence and state-space baselines and measure incremental value beyond them. Same-speaker persistence is the floor and a classical causal Kalman filter over the same observed anchor is the closest thing we find to a ceiling; every unconstrained neural model we train falls well below both. We do not generalize this to “don’t fight persistence” as blanket advice, since the ordering depends on state observability: when the anchor must be estimated rather than observed, estimated persistence drops below even a direct neural forecaster. The robust recommendation is evaluative, not architectural: whenever a trustworthy historical state is available, report persistence and a state-space baseline over it, and evaluate learned models on the value they add beyond those references.

P2: Within the tested state-observed setting, prioritize forecasting departure occurrence over magnitude. Occurrence is learnable while direction and size are not, under ablation, capacity, and multimodal sweeps, and occurrence learnability tracks the departure base rate across the three corpora we test – an empirical association, not an established law, and near chance on EmoryNLP – scaling inversely with it, so anticipation appears strongest in the rare-event regime that makes forecasting worth doing.

P3: Never proxy ranking by calibration. A model can be well-calibrated and still order its own errors worse than chance; selective deployment must test ranking directly via risk-coverage curves and, where stakes demand it, wrap commitment in distribution-free risk control rather than trusting verbalized or softmax confidence alone.

## 9. Discussion

## A. Limitations

Our strongest negative results are bounded by the configurations tested: frozen pretrained encoders with trained heads up to 3 7M parameters, evaluated on four corpora. They are also bounded by the oracle-anchor assumption of Section 4: same-speaker persistence’s dominance collapses once its anchor must be estimated rather than observed, so the Persistence Barrier is a statement about state-observed forecasting, not a fully realistic pipeline. We do not test full end-to-end fine-tuning, larger Transformer-based sequence models, or a directly-CCC-optimized regression head, and cannot rule out that one of these would close some of the gap. Several claims in this paper are stated as counts over many cells (e.g., “15 of 15 fold-model cells”); these are not independent replications, since multiple seeds and folds share the same underlying dialogues and encoder, and should be read as describing consistency across our experimental grid rather than additional independent statistical evidence – confidence intervals, where reported, use dialogueor session-level bootstrap rather than treating seeds as i.i.d. samples. All dialogue-level splits are re-derived at an approximately 80/10/10 ratio with a fixed seed rather than the original DailyDialog/MELD releases’ own splits, so our numbers are not directly comparable to prior work on the original splits; an episode-disjoint re-split of MELD and EmoryNLP (supplementary material available from the corresponding author upon request) moves shift-F1 by at most a few points per cell, though recurring-character-level leakage across episodes is not tested and remains an open question. Our architectures assume the target position’s speaker identity is known at commit time, a real but modest assumption on IEMOCAP and MELD’s dyadic subset and the most consequential on the full multi-party corpora, where the next speaker is a genuine scheduling fact we do not attempt to forecast jointly with the emotional departure. The mean-then-envelope CRC calibrator, reported alongside the Proposition-covered envelope-then-mean calibrator as a secondary comparison, fails to control mean risk in 1 of 12 main-pipeline cells in supplementary material available from the corresponding author upon request, which is itself part of why we do not treat it as the primary claim.

## B. Ethical and Societal Considerations

Anticipating a person’s emotional state before it is expressed raises surveillance concerns under emerging regulation. We scope this precisely: the EU AI Act (Art. 3(39)) defines an “emotion recognition system” as one inferring emotion from biometric data, and Art. 5(1)(f) restricts such inference specifically in workplace/education contexts – it does not categorically cover text-only inference, which is most of our experiments. This is precisely why we frame the deployable output of this work as selective: our risk-controlled abstention procedure makes “decline to predict” a first-class, statistically licensed output, so that a deployed system comes with an auditable, finite-sample bound on dialogue-averaged macro loss (not a bound on the conditional error among submitted predictions) instead of an unbounded average error rate. We recommend that any deployment in emotionally sensitive settings be accompanied by disclosure to the monitored party, human-inthe-loop review of high-stakes committed predictions, and periodic re-calibration of the CRC threshold against a heldout population.

## C. Practical Deployment Implications

Three findings bear directly on system design. First, because occurrence learnability degrades as the departure base rate rises, a deployed system should measure its target population’s local departure rate before committing to a forecasting-based intervention, rather than assuming the benefit transfers. Second, because the best commitment policy is base-rate-dependent and zero-shot policy transfer across domains is unreliable, a policy trained on one deployment’s data should not be assumed to transfer without re-calibration, even within the same architecture family. Third, because LLM verbalized confidence is an unreliable ranking signal for at least one widely used model, and reliability depends on the confidence-elicitation protocol, any system that relies on an LLM’s self-reported confidence to gate action should validate that signal empirically before deployment.

## 10. Conclusion

Within the frozen-encoder, state-observed testbed we audit, residual value forecasting beyond an observed emotional state is barricaded by same-speaker persistence – a barrier that itself dissolves once the state must be estimated rather than observed, which limits this conclusion to the state-observed setting. On the other side of it, the task we find tractable and useful is selective, risk-controlled anticipation of departures. We provide the formalization, the metric, the policies, the guarantee, and the audited evidence, including the negative results and their disclosed scope boundaries, on which future afective forecasting systems can be built and tested against stronger baselines. All code, configurations, and result logs needed to recompute every number and regenerate every figure are available from the corresponding author upon request; the proof of Proposition 1 is included in Appendix A, and implementation details and extended per-cell tables are available from the corresponding author upon request.

## Acknowledgment

The authors used a large language model assistant to help with English-language editing, LaTeX formatting, code review, and the documentation and presentation of supplementary analyses. All experimental designs, executions, interpretations, and conclusions were independently reviewed and verified by the authors, who assume full responsibility for the submitted work.

## Appendix A: Proof of Dialogue-Level Risk Control

Fix the finite grid $\Lambda \ = \ \{ \lambda _ { 1 } \ < \ \cdots \ < \ \lambda _ { m } \}$ and recall $\begin{array} { r } { \tilde { L } ( d , \lambda ) = \operatorname* { m a x } _ { \lambda ^ { \prime } \geq \lambda } \bar { L } ( d , \lambda ^ { \prime } ) . } \end{array}$ , so that for every dialogue d: (i) L<sup>˜</sup>(d ·) is non-increasing on $\Lambda ; ( \operatorname { i i } ) 0 \leq L ( d , \lambda ) \leq { \tilde { L } } ( d , \lambda ) \leq 1 ;$ and (iii) at λ = max $\Lambda > 1$ every dialogue fully abstains, so L(d<sub>,</sub> max $\Lambda ) = { \tilde { L } } ( d .$ max $\Lambda ) = 0$

Consider the augmented (exchangeable) collection $d _ { 1 } , \ldots , d _ { n + 1 }$ and define the augmented empirical risk

$$
\tilde { R } _ { n + 1 } ( \lambda ) = \frac { 1 } { n + 1 } \sum _ { j = 1 } ^ { n + 1 } \tilde { L } ( d _ { j } , \lambda ) ,
$$

non-increasing in λ as an average of non-increasing functions. Define the augmented threshold $\hat { \lambda } _ { n + 1 } = \operatorname* { i n f } \{ \bar { \lambda } \in \Lambda :$ $\tilde { R } _ { n + 1 } ( \lambda ) \leq \alpha \} ;$ by property (iii), $\tilde { R } _ { n + 1 } ( \operatorname* { m a x } \Lambda ) = 0 \leq \alpha .$ , so the feasible set is nonempty and the infimum is attained on the finite grid.

Lemma 1. $\begin{array} { r l r } { \hat { \lambda } } & { { } \ge } & { \hat { \lambda } _ { n + 1 } } \end{array}$ almost surely, where λ<sup>ˆ</sup> is the calibration-only threshold of Proposition 1.

Proof: For any $\lambda \in \Lambda$

$$
\tilde { R } _ { n + 1 } ( \lambda ) = \frac { n } { n + 1 } \tilde { R } ( \lambda ) + \frac { 1 } { n + 1 } \tilde { L } ( d _ { n + 1 } , \lambda ) \ \leq \ \frac { n } { n + 1 } \tilde { R } ( \lambda ) + \frac { 1 } { n + 1 } ,
$$

using $\tilde { L } \ \leq \ 1$ . Hence whenever the calibration condition $\begin{array} { r } { \frac { n } { n + 1 } \tilde { R } ( \lambda ) + \frac { 1 } { n + 1 } \le a } \end{array}$ holds, $\tilde { R } _ { n + 1 } ( \lambda ) \leq$ α also holds: the feasible set defining λ<sup>ˆ</sup> is contained in that defining $\hat { \lambda } _ { n + 1 }$ . Both infima exist on the finite grid (the latter’s feasible set is nonempty by property (iii); if the former’s is empty, λ<sup>ˆ</sup> = max $\Lambda \geq \hat { \lambda } _ { n + 1 }$ trivially). Infima over nested sets are ordered: $\hat { \lambda } \ge \hat { \lambda } _ { n + 1 }$

Proof of Proposition 1: By monotonicity of $\tilde { L } ( d _ { n + 1 } , \cdot )$ and Lemma 1, $\tilde { L } ( d _ { n + 1 } , \hat { \lambda } ) \leq \tilde { L } ( \dot { d } _ { n + 1 } , \hat { \lambda } _ { n + 1 } )$ , so it sufices to bound E $\boldsymbol { \tilde { L } } ( d _ { n + 1 } , \boldsymbol { \hat { \lambda } _ { n + 1 } } ) ]$

The threshold $\hat { \lambda } _ { n + 1 }$ is a symmetric function of the multiset $\{ d _ { 1 } , \ldots , d _ { n + 1 } \} \colon$ it depends on the dialogues only through the augmented risk curve ${ \tilde { R } } _ { n + 1 } ( \cdot )$ , which is permutation invariant. By exchangeability, conditionally on the multiset the identity of the test dialogue is uniform over the $n + 1$ elements; hence

$$
\mathbb { E } \Big [ \tilde { L } ( d _ { n + 1 } , \hat { \lambda } _ { n + 1 } ) \Big ] = \mathbb { E } \Bigg [ \frac { 1 } { n + 1 } \sum _ { j = 1 } ^ { n + 1 } \tilde { L } ( d _ { j } , \hat { \lambda } _ { n + 1 } ) \Bigg ] = \mathbb { E } \big [ \tilde { R } _ { n + 1 } ( \hat { \lambda } _ { n + 1 } ) \big ] \leq \alpha ,
$$

where the final inequality holds deterministically because $\hat { \lambda } _ { n + 1 }$ is, by construction, a feasible point of its defining condition on the finite grid. Combining, $\mathbb { E } [ \tilde { L } ( d _ { n + 1 } , \hat { \lambda } ) ] \leq \alpha ,$ and finally $\mathbb { E } [ L ( d _ { n + 1 } , \hat { \lambda } ) \bar { ] } \leq \mathbb { E } [ \tilde { L } ( d _ { n + 1 } , \hat { \lambda } ) ] \leq \alpha$ by pointwise dominance.

## References

[1] N. Majumder, S. Poria, D. Hazarika, R. Mihalcea, A. Gelbukh, and E. Cambria, “DialogueRNN: An attentive RNN for emotion detection in conversations,” in Proc. AAAI, 2019, pp. 6818–6825.

[2] D. Ghosal, N. Majumder, S. Poria, N. Chhaya, and A. Gelbukh, “DialogueGCN: A graph convolutional neural network for emotion recognition in conversation,” in Proc. EMNLP-IJCNLP, 2019, pp. 154–164.

[3] W. Shen, S. Wu, Y. Yang, and X. Quan, “Directed acyclic graph network for conversational emotion recognition,” in Proc. ACL-IJCNLP, 2021.

[4] B. Wen and T.-P. Tan, “Emotion forecasting in conversation: A survey of tasks, methods, and the anticipatory gap,” unpublished manuscript, 2026.

[5] E. Altarawneh, A. Agrawal, M. Jenkin, and M. Papagelis, “Predicting evoked emotions in conversations,” arXiv:2401.00383, 2023.

[6] H. Shi, Z. Liang, and J. Yu, “Emotional cues extraction and fusion for multi-modal emotion prediction and recognition in conversation,” in Proc. Interspeech, 2024, pp. 4074–4078.

[7] X. Lu, W. Zhao, Y. Zhao, B. Qin, Z. Zhang, and J. Wen, “A topic-enhanced approach for emotion distribution forecasting in conversations,” in Proc. IEEE ICASSP, 2023.

[8] C. Busso, M. Bulut, C.-C. Lee, A. Kazemzadeh, E. Mower, S. Kim, J. N. Chang, S. Lee, and S. S. Narayanan, “IEMOCAP: Interactive emotional dyadic motion capture database,” Language Resources and Evaluation, vol. 42, no. 4, pp. 335–359, 2008.

[9] Z. Oravecz, F. Tuerlinckx, and J. Vandekerckhove, “A hierarchical

latent stochastic diferential equation model for afective dynamics,” Psychological Methods, vol. 16, no. 4, pp. 468–490, 2011.

[10] X. Sun, “CARDIO: Cardiovascular- and physiology-informed afect series forecasting,” 2025.

[11] C. Bothe, S. Magg, C. Weber, and S. Wermter, “Dialogue-based neural learning to estimate the sentiment of a next upcoming utterance,” in Proc. Int. Conf. Artificial Neural Networks, 2017, pp. 477–485.

[12] Z. Wang, X. Zhu, Y. Zhang, S. Li, and G. Zhou, “Sentiment forecasting in dialog,” in Proc. COLING, 2020, pp. 2448–2458.

[13] D. Li, Y. Li, and S. Wang, “Interactive double states emotion cell model for textual dialogue emotion prediction,” Knowledge-Based Systems, vol. 189, p. 105084, 2020.

[14] R. Zhang, Z. Wang, Z. Huang, L. Li, and M. Zheng, “Predicting emotion reactions for human–computer conversation: A variational approach,” IEEE Trans. Human-Machine Systems, vol. 51, no. 4, pp. 279–287, 2021.

[15] X. Shi, X. Li, and T. Toda, “Emotion awareness in multi-utterance turn for improving emotion prediction in multi-speaker conversation,” in Proc. Interspeech, 2023, pp. 765–769.

[16] X. Ju, D. Zhang, S. Zhu, J. Li, S. Li, and G. Zhou, “Real-time emotion pre-recognition in conversations with contrastive multimodal dialogue pre-training,” in Proc. ACM CIKM, 2023.

[17] Y. Koga, S. Kando, and Y. Miyao, “Forecasting implicit emotions elicited in conversations,” in Proc. INLG, 2024, pp. 145–152.

[18] X. Ju, D. Zhang, S. Zhu, J. Li, S. Li, and G. Zhou, “ECFCON: Emotion consequence forecasting in conversation,” in Proc. ACM Multimedia, 2024, pp. 2233–2241.

[19] H. Wang, X. Mai, Z. Tao, J. Lin, X. Tong, I. Pan, S. Yan, Y. Wang, and S. Gao, “Hi-EF: Benchmarking emotion forecasting in humaninteraction,” Proc. AAAI Conf. Artificial Intelligence, vol. 40, no. 3, pp. 2110–2118, 2026.

[20] S. Roohi, R. Skarbez, and H. D. Nguyen, “Reliable uncertainty estimation in emotion recognition in conversation using a conformal prediction framework,” Natural Language Processing, vol. 31, no. 5, pp. 1163–1186, 2025.

[21] J. P. Chang and C. Danescu-Niculescu-Mizil, “Trouble on the horizon: Forecasting the derailment of online conversations as they develop,” in Proc. EMNLP-IJCNLP, 2019, pp. 4743–4754.

[22] L. Kim, V. Nguyen, and C. Danescu-Niculescu-Mizil, “Wait! There’s a way out: A decision mechanism for forecasting conversational derailment,” in Proc. 64th Annu. Meeting Assoc. Computational Linguistics, 2026, pp. 41955–41974.

[23] R. El-Yaniv and Y. Wiener, “On the foundations of noise-free selective classification,” Journal ofMachine Learning Research, vol. 11, pp. 1605–1641, 2010.

[24] Y. Geifman and R. El-Yaniv, “Selective classification for deep neural networks,” in Advances in Neural Information Processing Systems, 2017.

[25] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration

of modern neural networks,” in Proc. Int. Conf. Machine Learning, 2017, pp. 1321–1330.

[26] A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, and T. Schuster, “Conformal risk control,” in Int. Conf. Learning Representations, 2024.

[27] Y. Xu, W. Guo, and Z. Wei, “Selective conformal risk control,” 2025.

[28] A. N. Angelopoulos, “Conformal risk control for non-monotonic losses,” 2026.

[29] P. Kuppens, N. B. Allen, and L. B. Sheeber, “Emotional inertia and psychological maladjustment,” Psychological Science, vol. 21, no. 7, pp. 984–991, 2010.

[30] E. A. Butler, “Temporal interpersonal emotion systems: The ‘TIES’ that form relationships,” Personality and Social Psychology Review, vol. 15, no. 4, pp. 367–393, 2011.

[31] P. Kuppens and P. Verduyn, “Emotion dynamics,” Current Opinion in Psychology, vol. 17, pp. 22–26, 2017.

[32] D. Ha and J. Schmidhuber, “Recurrent world models facilitate policy evolution,” in Advances in Neural Information Processing Systems, 2018.

[33] B. Zhao, F. Ye, Y. Ji, S. Zhao, X. Peng, and Z. Yu, “AfectVerse: Emotional world models for multimodal afective computing,” 2026.

[34] C. Song, Y. Zhang, H. Gao, C. Yang, and P. Zhang, “Large emotional world model,” 2025.

[35] Y. Zhao, X. Wang, D. Wang, Z. Jiang, Q. Gu, T. Chen, N. Xi, J. Qu, Y. Chen, and L. Ji, “Dream to chat: Model-based reinforcement learning on dialogues with user belief modeling,” in Findings of ACL: EMNLP, 2025.

[36] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, “RoBERTa: A robustly optimized BERT pretraining approach,” arXiv:1907.11692, 2019.

[37] S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, Z. Chen, J. Li, N. Kanda, T. Yoshioka, X. Xiao, J. Wu, L. Zhou, S. Ren, Y. Qian, Y. Qian, J. Wu, M. Zeng, X. Yu, and F. Wei, “WavLM: Large-scale self-supervised pre-training for full stack speech processing,” IEEE J. Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1505– 1518, 2022.

[38] M. Xiong, Z. Hu, X. Lu, Y. Li, J. Fu, J. He, and B. Hooi, “Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs,” in Int. Conf. Learning Representations, 2024.

[39] Y. Li, H. Su, X. Shen, W. Li, Z. Cao, and S. Niu, “DailyDialog: A manually labelled multi-turn dialogue dataset,” in Proc. IJCNLP, 2017, pp. 986–995.

[40] S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, and R. Mihalcea, “MELD: A multimodal multi-party dataset for emotion recognition in conversations,” in Proc. ACL, 2019, pp. 527–536.

[41] S. M. Zahiri and J. D. Choi, “Emotion detection on TV show transcripts with sequence-based convolutional neural networks,” in AAAI Workshops, 2018.