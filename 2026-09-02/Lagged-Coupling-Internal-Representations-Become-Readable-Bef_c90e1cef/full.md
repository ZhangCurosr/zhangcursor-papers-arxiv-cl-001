# Lagged Coupling: Internal Representations Become Readable Before They Become Causal

Xining Xun

Tsingjiao Information Science (Beijing) Co., Ltd.

## Abstract

Across the full Pythia suite (160M–12B, eight checkpoints, four task families), a linear probe can read a target variable from the residual stream as early as step 1,000 at every scale—yet steering along that same reading direction remains null-equivalent in 43 of 48 model×checkpoint cells. Internal readability systematically outruns causal efficacy, and the lag does not shrink with scale. We call this structure lagged coupling and decompose it into three dissociable tracks: (i) internal readability, saturated (AUROC ≥ 0.990) from the first checkpoint everywhere; (ii) behavioral readability, the output layer’s own answer information, which develops gradually and progressively later at larger scales (12B reaches 0.909 only at the final checkpoint); (iii) causal efficacy, almost always null-equivalent, occasionally counterproductive early in training, with a single isolated positive pulse (12B, step 8,000, z = +2.49) that our grid cannot resolve. The ordering is dominantly read-before-write (11/11 pattern-1 units, no inversion). Mechanistically, representation headroom along the probe direction grows up to 57× with training and scale while causal write-in stays pinned below 0.11% of headroom—the variable is increasingly written into the representation and increasingly ignored by the readout. These findings were produced under a fully pre-registered decision protocol whose two single-onset hypotheses both resolve INDETERMINATE (scale-axis slope +0.24, 95% CI [−0.60, +0.87]; time-axis vote 3:3; 34.6% of units develop backwards)—a disciplined negative that is explained, rather than contradicted, by the three-track decomposition: single-axis onset regressions conflate clocks that run at different speeds. A pre-registered OLMo-2 replication preserves the direction at attenuated magnitude (“weak” by the frozen rule). Our results caution against inferring steerability from probe accuracy and establish a developmental bottleneck—representation formation reliably outpaces causal readout consolidation— together with a fully audited, hash-chained pipeline as a reusable template for developmental claims about mechanism.

Keywords: mechanistic interpretability, probing, causal intervention, training dynamics, scaling, preregistration

## 1 Introduction

A growing body of work argues that the capabilities of language models are not just measurable in their outputs but locatable in their internals: linear probes recover structured information from hidden states<sup>[13][14]</sup>, causal tracing localizes factual recall to specific sites<sup>[20][21]</sup>, and activationsteering methods move behavior along interpretable directions<sup>[24][25][26]</sup>. These successes suggest an implicit syllogism that is increasingly load-bearing for AI control: if a concept is readable from a representation, then the representation is a useful lever on behavior. Whether this syllogism holds —and when in training, and at what scale, it comes to hold—is an empirical question that has not been subjected to a pre-registered test.

The question matters because the two sides of the syllogism have opposite failure modes for safety. If readability precedes causal efficacy, then monitoring-based approaches (probes as early-warning sensors) are viable but intervention-based approaches (steering as a control surface) may lag dangerously behind—models could become transparent to us long before they become steerable by us. If causal efficacy precedes readability, the opposite asymmetry holds. And if the two are developmentally coupled, the distinction collapses. Distinguishing these accounts requires tracking three quantities over training time and model scale simultaneously: how readable a target variable is internally, how readable it is behaviorally, and how causally effective an intervention along the reading direction actually is.

Doing this credibly is harder than it looks. Developmental claims about mechanism are exactly where the field’s methodological weaknesses bite hardest: checkpoint suites can be silently corrupted (and one widely used one is—Appendix A), probe accuracies saturate trivially, steering nulls are rarely matched for norm and site, and post-hoc threshold tuning can manufacture “emergence” out of noise<sup>[7]</sup>. We therefore built the study around a pre-registered protocol: a frozen grid (6 model sizes × 8 checkpoints × 4 task families), frozen estimands with stratified nulls, a frozen two-axis decision tree (a scale axis and a time axis, each resolving to fork A, fork C, or INDETERMINATE), golden-standard self-checks for every instrument before any criterion data were produced, and an append-only, md5-chained audit ledger covering every configuration, code revision, and engineering event (including a cloud restart at 174/192 test units, from which the run recovered with zero data loss).

What the protocol returns is a stable, cross-scale structure rather than a verdict on either fork. Decomposing “controllability” into the three quantities above reveals a consistent lagged-coupling structure: internal readability is present from the earliest checkpoint at every scale; behavioral readability develops gradually and progressively later at larger scales; causal efficacy remains nullequivalent almost everywhere and is occasionally counterproductive early in training. The ordering of these events is dominantly read-before-write, the lag between them does not shrink with scale, and a mechanistic measurement—representation headroom growing up to 57× while causal writein stays pinned below 0.11% of headroom—explains why. Both pre-registered forks resolve INDETERMINATE, and the three-track decomposition shows why they had to: a single-onset hypothesis cannot describe three clocks running at different speeds. We frame the study throughout as a developmental scaling study, not a law. Our contributions are:

A three-track dissociation with a read-before-write ordering: internal readability, behavioral readability, and causal efficacy develop on different clocks (§4.2), the ordering is dominantly $\mathrm { I } _ { \mathrm { l e a d } } .$ -catch-up with no observed inversion (11/11 pattern-1 units), and scale does not close the lag (§4.3). Steering is inert in 43/48 cells and significantly harmful in four early cells (§4.4).

A mechanistic account of the bottleneck: representation headroom along the probe direction grows 2.0–56.8× with training and scale while causal write-in remains ≤ 0.11% of headroom, and the instrument’s noise floor itself scales monotonically and is reported openly rather than hidden (§4.5).

A reusable, fully audited pre-registration template for developmental claims about mechanism—frozen estimands and decision tree, golden-standard instrument checks, matched nulls, an md5-chained ledger, and a content-hash gate that caught a silently corrupted public checkpoint lineage, later corroborated independently (§3, §7, Appendix A).

## 2 Related Work

Scaling, emergence, and development. Loss and capability scaling with model size, data, and compute are well characterized<sup>[3][4][5]</sup>, and whether capabilities emerge sharply or improve gradually is contested<sup>[6][7]</sup>. Developmental-time analyses of internals are rarer: grokking work shows generalization can lag memorization with identifiable internal progress measures<sup>[8][9]</sup>, and induction heads form in a narrow training window and mediate in-context learning<sup>[10]</sup>. We extend this line from “when does a circuit form” to “when does a circuit become a lever,” across the full scale range of a single suite.

Probing and its limits. Structural and behavioral probes recover linguistic and factual information from hidden states<sup>[13][14]</sup>, but probe accuracy confounds representation with probe capacity and task priors; control tasks, information-theoretic corrections, and amnesic or projection-based variants quantify these confounds<sup>[15][16][17][18][19]</sup>. We treat probe AUROC as one track among three and pair every probe with a matched causal readout, so the probe–causal gap becomes a measured quantity rather than an untested assumption.

Circuit-level mechanism. The circuits program—mathematical frameworks for transformer computation, interpretable features such as induction heads, and full reverse-engineered behaviors like indirect object identification—establishes that behaviorally relevant structure is localizable<sup>[10][11][12]</sup>. Causal mediation analysis and weight-level edits (e.g., ROME) intervene on those structures<sup>[20][21]</sup>, and causal-abstraction frameworks (interchange interventions, IIT) formalize when a neural subsystem implements a causal model<sup>[22][23]</sup>. Our interventions are deliberately weaker—probe-direction output surgery rather than weight edits—because our estimand is developmental: does the same reading direction, frozen, gain causal leverage over training?

Activation steering and representation-level control. Representation fine-tuning, activation addition, representation engineering, and inference-time intervention all steer behavior along internal directions<sup>[24][25][26][27]</sup>, and linear structure in representation spaces supports simple readouts of relations and concepts<sup>[28][29]</sup>. Nearly all of this work is conducted on fully trained models. Our results bear directly on it: the directions these methods exploit are readable long before they are causal, and at most scales and times they are not causal at all. Table 1 makes the gap explicit: the steering literature operates on final checkpoints and assumes the causal track is in place once readability is; the developmental dimension—when each track comes online—is, to our knowledge, unmeasured prior to this study.

Table 1. Where prior work sits relative to the three tracks. “Track assumed” names the property each line of work relies on; “developmental?” asks whether the property is measured across training. Readability results exist developmentally for probes; steering and editing results do not.
<table><tr><td>Line of work</td><td>Representative methods</td><td>Track assumed</td><td>Developmental?</td></tr><tr><td>Probing</td><td>structural / control-task / amnesic Track 1 probes[13][15][18]</td><td>readability)</td><td>(internal partially</td></tr><tr><td>Activation steering</td><td>ActAdd, RepE, ITI, ReFT[24][25][26][27]</td><td>Track 3 (causal no efficacy)</td><td></td></tr><tr><td>Weight editing</td><td>ROME[21]</td><td>Track 3 (via weights)</td><td>no</td></tr><tr><td>Developmental circuits</td><td>induction heads, grokking progress Track 1 + behavior measures[9][10]</td><td></td><td>yes (no steering)</td></tr><tr><td>This work</td><td>frozen probes + matched-null output all three tracks jointly yes (6 sizes surgery</td><td></td><td>× 8 checkpoints)</td></tr></table>

Checkpoint suites as scientific instruments. Pythia<sup>[1]</sup> and OLMo-2<sup>[2]</sup> make developmental studies possible by releasing intermediate checkpoints. Appendix A documents a caveat we now treat as a hard requirement: checkpoint lineages can be silently corrupted in public mirrors, and percheckpoint content hashing is a necessary gate for any study of this kind.

## 3 Pre-registered Setup and Decision Protocol

All estimands, thresholds, decision rules, and degradation branches were frozen before any criterion data were produced; the protocol revisions that occurred during execution (a hotfix to the

aggregation fallback chain, a manifest refresh after a contamination incident, and restart-recovery procedures) were logged in the append-only audit ledger before the affected data were reproduced, and none touched thresholds, nulls, or the decision tree (§7).

## 3.1 Grid and instrumentation

The grid crosses the six publicly released Pythia sizes<sup>[1]</sup> with eight checkpoints spanning training, and four task families designed to separate distinct controllability demands (Table 2). Each of the resulting 192 units is evaluated by the same frozen pipeline: a probe track, an output-level track, an intervention track with matched nulls, and per-unit resolution bookkeeping. Every model×checkpoint weight file passed a mandatory content-hash gate (per-checkpoint hashes pairwise distinct, lineage pinned by revision) before use; this gate is what caught the corrupted 2.8B lineage described in Appendix A. Total compute was approximately 240 instance-hours on 2×A100-SXM4-40GB.

Table 2. The pre-registered grid. Checkpoints are training steps (in thousands: 1, 2, 4, 8, 16, 32, 64, 143). Parameter counts are taken from the model cards<sup>[1]</sup>.
<table><tr><td>Model</td><td>Parameters</td><td>Checkpoints (steps)</td><td>Task families</td></tr><tr><td>Pythia-160M</td><td> $1 . 6 2 \times 1 0 ^ { 8 }$ </td><td>1k, 2k, 4k, 8k, 16k, 32k, 64k, 143k</td><td>T: template contrast</td></tr><tr><td>Pythia-410M</td><td> $4 . 0 5 \times 1 0 ^ { 8 }$ </td><td></td><td>D: distractor robustness P: prior-bias override</td></tr><tr><td>Pythia-1B</td><td> $1 . 0 1 \times 1 0 ^ { 9 }$ </td><td></td><td>N: negative transfer</td></tr><tr><td>Pythia-2.8B</td><td> $2 . 7 8 \times 1 0 ^ { 9 }$ </td><td></td><td></td></tr><tr><td>Pythia-6.9B</td><td> $6 . 8 5 \times 1 0 ^ { 9 }$ </td><td></td><td></td></tr><tr><td>Pythia-12B</td><td> $1 . 1 8 \times 1 0 ^ { 1 0 }$ </td><td></td><td></td></tr></table>

The four families probe different ways a target variable can fail to be behaviorally expressed: T isolates the answer-format contrast itself; D adds competing surface cues; P pits the target against a learned prior (the basis of the bias arm, §4.6); N measures interference from related-but-wrong structure. The bias arm additionally fits a perunit bias surface over intervention strengths and aggregates its scale dependence mechanically (no researcher degrees of freedom; §4.6).

## 3.2 Estimands

Let $z _ { t } ( x )$ be the logit of token at the answer position for prompt , with designated answer token t x $t ^ { * }$ . The answer score is the logit margin

$$
s ( x ) = z _ { t ^ { * } } ( x ) - \operatorname* { m a x } _ { t \neq t ^ { * } } z _ { t } ( x ) .\tag{1}
$$

Track 1 — internal readability. A logistic probe $\begin{array} { r } { \hat { \boldsymbol { y } } ( \boldsymbol { x } ) = \sigma ( \boldsymbol { w } ^ { \top } a _ { \ell } ( \boldsymbol { x } ) + b ) } \end{array}$ is trained per unit on held-apart items at the pre-registered best site $\ell ;$ inducibility is its AUROC on the dev split, withi test-split AUROC $i _ { \mathrm { r a w } } ^ { \mathrm { ( t e s t ) } }$ computed under the frozen pipeline (a discrepancy flag triggers $\mathrm { i f } \mid i _ { \mathrm { r a w } } ^ { \mathrm { ( t e s t ) } } -$ $i ^ { \mathrm { ( d e v ) } } | > 0 . 1 ;$ ; it fired in only 3 of 192 units, all at 12B, all < 0.12).

Track 2 — behavioral readability. Independently of any probe, we score how much answer information the output layer itselfcarries:

$$
\mathrm { A U R O C } _ { \mathrm { d e v } } \ = \ \mathrm { A U R O C } ( y , \ s ( x ) ) ,\tag{2}
$$

the AUROC of the model’s own answer score against the gold label. Tracks 1 and 2 are different quantities by construction—one reads an internal site through a trained probe, the other reads the logit margin directly—and their divergence is a result, not a leak (§4.2).

Track 3 — causal efficacy. The intervention adds the unit-norm probe direction at the probed site, $a _ { \ell } \gets a _ { \ell } + \lambda$ , and the paired effect is^w

$$
e ( \lambda ) \ = \ \mathbb { E } \big [ s ( x ^ { ( \lambda ) } ) - s ( x ^ { ( 0 ) } ) \big ] ,\tag{3}
$$

with expectation over matched item pairs. The null distribution is generated by edits along random directions matched for site and norm, and efficacy is reported as

$$
z _ { e } = \frac { e - \mu _ { \mathrm { n u l l } } } { \sigma _ { \mathrm { n u l l } } } , ~ \mathbf { \hat { c } } = \frac { e ( 1 ) } { C } ,\tag{4}
$$

where is representation headroom—the per-unit RMS magnitude of the residual-streamC component along before the edit—and ^w $\scriptstyle \mathbf { \hat { c } }$ expresses causal write-in as a fraction of that headroom. The instrument’s per-unit noise floor $\varepsilon _ { m }$ (median absolute within-unit null effect) is recorded for every unit; a unit whose $| e |$ is below its own $\varepsilon _ { m }$ is unresolved by construction, and $\varepsilon _ { m }$ enters the audit trail rather than being silently absorbed.

Development index. For each unit with a complete milestone series, $D _ { u }$ is the Spearman correlation between checkpoint order and the unit’s pre-registered milestone progress; under monotone developmental consolidation, most units should have $D _ { u } > 0$ . We report the negative fraction $\# \{ D _ { u } < 0 \} / N$ over the $N = 1 9 1$ units with a defined index.

## Notation and terminology (used throughout).

Readable: a quantity is linearly decodable above chance under the frozen split. Causal: an edit along the reading direction changes the answer score beyond the matched-null band. Headroom : howC strongly the representation expresses the variable along the probe direction. Write-in $\mathbf { \hat { c } } = e ( 1 ) / C \mathbf { : }$ the causal push achieved per unit of headroom. Noise floor $\varepsilon _ { m }$ : the instrument’s per-unit resolution, reported alongside every null. Pattern-1 unit: a unit whose milestones activate in the canonical order analyzed by the frozen ordering analysis. $\mathbf { I _ { l e a d } } .$ -catch-up: internal readability leads; the other tracks follow. Every term is defined once here and used with no other meaning.

## 3.3 The decision tree

Pre-registered decision procedure (frozen before criterion data).

Scale axis. Weighted least squares (weights $1 / \mathrm { s e } ^ { 2 } )$ of unit onset step against $\log _ { 1 0 }$ parameters, main line $\left( \mathrm { n } = 6 \right)$ and sub line (n = 5). Time axis. Per size, WLS of milestone index against log step; fork A if the 95% CI lies entirely below zero, fork C if entirely above, INDETERMINATE otherwise; the axis verdict is the majority vote across sizes. Either axis resolves A, C, or INDETERMINATE. Discipline clauses. No threshold lowering, no estimand switching, no seed completion, no post-hoc re-gridding; if a fork fails, the pre-written INDETERMINATE branch is taken and the paper must not claim a “law.” A replication arm (OLMo-2) was pre-registered with an explicit degradation rule: with fewer than three replication sizes, the replication verdict is capped at “weak.”

## 3.4 Why additive single-site surgery is the right estimand

Our intervention is deliberately the weakest and most transparent member of the steering family: an additive edit along the frozen probe direction at a single site. We motivate this choice on design grounds, not convenience. (i) Lower-bound semantics. This edit is the minimum-energy path that pushes the representation along the very direction a probe reads; if even this direct, signal-aligned push produces no causal effect, then any claim of early controllability must explain why more elaborate machinery succeeds where the straightest path fails. A null here is therefore not “steering is impossible” but a strong stress test of the direction-generalization assumption—that a

direction good for reading is good for writing. (ii) Comparability. The same frozen direction, norm, and site rule applies identically at every size and checkpoint, which is what makes developmental and cross-scale statements well-posed; learned or multi-site interventions inject per-cell degrees of freedom that would confound exactly the comparisons this study exists to make. (iii) Direct correspondence with the probing literature. Probe directions are the currency of readability claims<sup>[13][14][18]</sup>; testing their causal efficacy is the sharpest possible probe–steering juxtaposition. (iv) Runtime control, not weight editing. Our target is activation-level steering at inference time—the only class of mechanism usable as a real-time guardrail on a frozen deployed model. Weight-space edits<sup>[21]</sup> and trained representation interventions<sup>[23][24]</sup> answer a different question (can the model be changed), and our nulls do not speak against them; we return to this boundary in §6. Finally, the silence we report is not an artifact of a timid dose: dose–response curves $e ( \lambda )$ over the frozen strength grid are flat in the null cells (§4.4), and each cell’s null is matched for edit norm, so a larger push along random directions does not do what the reading direction fails to do.

## 4 Results

## 4.1 The pre-registered verdict: INDETERMINATE on both axes

Table 3 reports the frozen decision. On the scale axis, onset does not move systematically with parameters (main line slope +0.24, 95% CI [−0.60, +0.87]; sub line +0.32, [−0.12, +1.18]). On the time axis, per-size slopes (Figure 1) split 3 fork-A (410M, 2.8B—an edge case whose CI upper bound is $- 5 . 8 { \times } 1 0 ^ { - 1 2 } .$ —and 6.9B) against 3 INDETERMINATE (160M, 1B, 12B), so no majority forms. The development index confirms the non-monotonicity directly: 34.55% of units (66/191) develop backwards by the frozen definition. Under the protocol’s discipline clauses, both axes resolve INDETERMINATE and no scaling law is claimed. What remains—and what the rest of this section develops—is the structure of why the monotone-emergence hypotheses fail.

Table 3. The frozen decision tree, evaluated. Slopes are WLS with 95% CIs; the time axis is the majority vote of the six per-size verdicts (Figure 1).
<table><tr><td>Axis</td><td>Line</td><td>Estimate</td><td>95% CI</td><td>n</td><td>Verdict</td></tr><tr><td>Scale</td><td>main</td><td>+0.240</td><td>[-0.600, +0.866]</td><td>6</td><td>INDETERMINATE</td></tr><tr><td>Scale</td><td>sub</td><td>+0.325</td><td>[-0.121, +1.182]</td><td>5</td><td>INDETERMINATE</td></tr><tr><td>Time</td><td>main</td><td>3 A/ 3 IND</td><td></td><td>6</td><td>INDETERMINATE (no majority)</td></tr><tr><td>Development</td><td>neg. D fraction</td><td>0.3455</td><td>66/191</td><td>191</td><td>non-monotone</td></tr></table>

![](images/f9403f680def584183a7609146c45fe15bea027069e983e56d554c041a05338c.jpg)  
Figure 1. Per-size time-axis WLS slopes of milestone index against log step, with 95% CIs. Blue: fork decided A (410M, 2.8B edge, 6.9B); grey: INDETERMINATE (160M, 1B, 12B). The majority-vote rule yields no axis verdict. Note the 2.8B edge case: its CI upper bound clears zero by $5 . 8 { \times } 1 0 ^ { - 1 2 } -$ —the decision is mechanical, not visual.

## 4.2 Three tracks, three clocks

![](images/37b587eff0ee97cbd9dcc0e4fb61fb79e574fde0ce2c4c6cb76317120f364e5c.jpg)  
Figure 2. Lagged coupling at a glance (schematic of the measured structure). Internal readability is saturated from step 1k at every scale; behavioral readability ramps gradually and later at larger scales; causal efficacy sits inside the null band almost everywhere, with significant negative effects early in training and one isolated positive pulse at 12B step 8k. The two lags—internal→behavioral and behavioral→causal—do not shrink with scale. Quantitative versions: Figures 3, 4, 5.

Decomposing the units into the three tracks of §3.2 dissolves the apparent paradox of §4.1. Track 1 (internal readability) saturates immediately and universally: probe inducibility $\mathrm { i } s \geq 0 . 9 9 0$ at every size from step 1,000 onward (dev minimum 0.990; test minimum 0.982; the pre-registered dev↔test discrepancy flag fired in only 3/192 units). A linear reader can recover the target variable from the residual stream essentially as soon as training begins, at every scale. Track 2 (behavioral readability) tells a different story (Figure 3, blue): output-level answer information develops gradually, and its onset is later at larger scales—the 2.8B model climbs from 0.534 to 0.968, 6.9B from 0.569 to 0.896, and 12B from 0.524 to 0.909, crossing 0.9 only at the final checkpoint, while 1B plateaus near 0.71. Track 3 (causal efficacy) is nearly silent: 43 of 48 model×checkpoint cells lie inside the null band $( | z _ { \mathrm { e } } | < 2 )$ , and the median effect at unit strength is pinned near zero (§4.4).

Three developmental tracks dissociate: readable first, behavioural next, causal last  
![](images/58f9320616dd6226a198cbf3abd105c1b3127ab767abbea03c17d76c07aee15d.jpg)

![](images/c6b6c407effa451431b31cb33542bee97dd1e268091705c7a47cc8ce1f8fafed.jpg)  
Figure 3. Three tracks, three clocks (6.9B and 12B). Left axis: internal readability (green; probe inducibility, saturated ≈ 1.0 from step 1k) and behavioral readability (blue; output-level answer-score AUROC). Right axis: causal efficacy $z _ { e }$ (orange) with the $| z | < 2$ null band shaded. The 12B panel marks the isolated pulse at step 8k $( z _ { e } = + 2 . 4 9 )$ . Readability—internal then behavioral—precedes causal efficacy throughout, and the gap widens with scale.

The three tracks order themselves. At no scale does causal efficacy precede readability, and at larger scales behavioral readability itself lags internal readability by tens of thousands of steps. “Emergence of controllability” is therefore not one event with one onset time—which is exactly why a single-axis onset regression returns INDETERMINATE.

## 4.3 The ordering is read-before-write, and scale does not close the lag

The frozen ordering analysis classifies each unit’s developmental pattern by which track moves first. The dominant pattern is $\mathbf { I _ { l e a d } - c a t c h - u p } \colon$ internal readability leads, and the other tracks follow (11 of 11 pattern-1 units; no instance of the inverted, write-before-read pattern was observed). Crucially for the scaling question, the catch-up is not faster at larger scales: the pre-registered regression of catch-up magnitude on log parameters has 95% CI [−0.722, +0.438], crossing zero. Scale amplifies the early reader and the late writer alike; it does not couple them. This is the sense in which the coupling is lagged: the arrow from representation to behavior exists in the ordering statistics, but its latency is scale-invariant within our range.

## 4.4 Interventions are largely inert, occasionally counterproductive

Figure 4 maps $z _ { e }$ over the full 6×8 grid. The headline is silence: 43 of 48 cells lie inside the null band, including every cell at 160M and every late cell at 12B. The more informative half of the remainder is early backfire: four of the five significant cells are negative, and all four occur in the first 2,000 steps—410M step 2k ( ), 2.8B step 1k ( ), 2.8B step 2k ( ), 12B step 2k (−2.65 −2.68 −2.48 ). Before the reading direction is causally inert, it is causally counterproductive: pushing the−2.43 representation toward the answer pushes the output away from it. From mid-training onward, 2.8B (+0.75, +0.72, +0.45 at steps 16k–64k) and 6.9B (+0.80 at 4k; +0.70, +0.51, +0.45 at 32k–143k) sit consistently on the positive side of the null without reaching significance, while 160M and 410M straddle zero throughout (ranges [−0.95, +0.55] and [−2.65, +1.22]).

Dose–response rules out an under-powered edit. The effect curves $e ( \lambda )$ over the frozen strength grid are flat at all tested strengths in every null cell: the silence is not the artifact of a single timid intervention strength, and per-cell nulls are matched for edit norm, so “push harder” does not convert a null direction into an effective one. This closes the most direct alternative explanation of Track 3 and is why we read the grid as evidence about the direction, not the dose.

One isolated positive pulse, reported and bounded. The sole cell where steering significantly helps is 12B at step 8,000 $( z _ { e } = + 2 . 4 9 ;$ ; dev phase +2.04); it does not persist to step $1 6 , 0 0 0 ~ ( z _ { e } =$ $- 0 . 3 1 )$ . We report it because the audit protocol requires reporting every cell, and because it is consistent with a transient re-wiring window—the behavioral track is mid-transition at that point $( \mathrm { A U R O C _ { d e v } }$ 0.632 at 4k, 0.561 at 8k). But it occupies exactly one of 48 cells, our eight-checkpoint grid cannot resolve any such window, and we make no timing claim about it (§6). It should be read as an invitation to denser developmental sampling, not as evidence of usable controllability.

![](images/7768ee4198632d9c589a75eeef2b41f27a041c811932c32dfb55ca2f2ac9e7bf.jpg)  
Figure 4. Boxing rule: cells with $| z | \geq 2$ are boxed; the colormap is clipped at ±2.7. Causal efficacy $z _ { e }$ over the full grid (6 sizes × 8 checkpoints). 43/48 cells lie within the null band; the four significant negative cells are all early (backfire); the only positive significant event is the isolated 12B step-8k pulse (+2.49).

## 4.5 Mechanism: headroom outruns write-in by orders of magnitude

Why does readability fail to convert into causal leverage? The coupling measurements localize the bottleneck. Representation headroom along the probe direction grows steeply with both trainingC time and scale (Figure 5a): over training, expands by factors of 2.0× (160M), 8.1× (410M), 3.0×C (1B), 46.9× (2.8B), 37.7× (6.9B), and 56.8× (12B, from 0.038 to 2.16). Yet the normalized write-in stays pinned: at the final checkpoint, is 0.070% (2.8B), 0.109% (6.9B), and 0.108% (12B), with per-cell∣^c∣ effects never exceeding 0.016 in absolute answer-score units anywhere on the grid. Thee(1) direction is increasingly occupied—the representation expresses the target variable ever more strongly along it—but the output circuit draws on that component ever more weakly in relative terms. Meanwhile the instrument’s own resolution degrades smoothly with scale: the noise floor $\varepsilon _ { m }$ is strictly monotone across sizes, 0.00398 (160M) → 0.00597 (410M) → 0.00619 (1B) → 0.01179 (2.8B) → 0.0286 (6.9B) → 0.0328 (12B) (Figure 5b), a 8.2× range that we report rather than hide, since it bounds what “null-equivalent” means at each scale.

![](images/07d0cc2e96f1f34701dad6c8c71776cab481a2212036f673ff16a337bdb658db.jpg)

![](images/5160ee59dbc2d057b9b599fb89af8fb7c8c3587c78cffae5519706f519377e84.jpg)  
Figure 5. (a) Representation headroom along the probe direction over training (log scale); growthC factors from step 1k to 143k annotate each curve. (b) Instrument noise floor $\varepsilon _ { m }$ against parameter count (log–log), strictly monotone over six sizes; the dashed line marks the largest causal effect observed anywhere on the grid $( | { \bf e } ( 1 ) | ~ = ~ 0 . 0 1 6 )$ , which sits at or below the noise floor for the three largest sizes. Headroom grows up to 57× while causal write-in stays below 0.11% of headroom.

Read together with §4.2, a consistent picture emerges: the probe direction becomes an increasingly accurate description of a component that the downstream computation increasingly ignores. Readability tracks what the representation contains; causality tracks what the readout uses; development moves the two in opposite relative directions.

## 4.6 The prior-bias arm: a mechanical positive result

The protocol’s bias arm produced the study’s one pre-registered positive: the prior-bias effect grows with scale. Aggregating the fitted bias surfaces over all 24 size×family cells mechanically (the claim text is assembled by the pipeline, not written by hand), the WLS slope of bias magnitude on log parameters has 95% CI [+0.00062, +0.291]—excluding zero by a hair—with intercept −0.719. Larger models are harder to steer off their priors. In the lagged-coupling frame this is the mirror image of the main result: the same consolidation that makes behavior progressively self-consistent also makes it progressively resistant to external direction, and it is the prior—not the probe direction— that wins. We flag the lower bound’s proximity to zero honestly: the effect is small per unit of logscale, though unambiguous in sign.

## 4.7 Replication on OLMo-2: direction preserved, magnitude weak

The pre-registered replication arm reran the pipeline on OLMo-2<sup>[2]</sup> at two sizes. Under the frozen degradation rule (fewer than three sizes ⇒ verdict capped at “weak”), the arm replicates the direction of every headline finding—probe saturation from the earliest checkpoint, a readablebefore-causal ordering, null-equivalent steering effects—at attenuated magnitude, and is therefore scored weak. We report this as calibrated evidence, not confirmation: two sizes cannot carry a scale claim, and the protocol says so.

## 5 Discussion

## 5.1 Implications for AI safety: sensors before levers

The central practical claim of activation-level interpretability—that directions found by probes are handles on behavior<sup>[24][25][26][27]</sup>—is, in our grid, true only asymptotically and weakly. Concretely:

monitoring-based safety schemes that use probes as early-warning sensors rest on Track 1, which our data support strongly and unconditionally (saturated from step 1,000 at every scale). Intervention-based runtime control schemes that steer along probe-derived directions rest on Track 3, which our data fail to support for nearly all of training—inert in 43/48 cells and significantly harmful in four early ones. A probe can be an excellent sensor at step 1,000 and a useless lever at step 143,000. Safety cases built on probe accuracy alone are measuring Track 1 and asserting Track 3. Table 4 separates the two bets.

Table 4. Two families of safety-relevant methods bet on different tracks; our grid supports one bet and not the other across training.
<table><tr><td>Safety-relevant scheme</td><td>Examples</td><td>Track it bets on</td><td></td><td>Supported by our grid?</td></tr><tr><td>Monitoring early warning</td><td>/ probes as sensors[13][14]</td><td>Track 1: readability</td><td></td><td>internal Yes—saturated (AUROC ≥ 0.990) from step 1k, all scales</td></tr><tr><td>Runtime activation control</td><td>ActAdd / RepE / ITI / ReFT-style steering[24][25] [26][27]</td><td>Track 3: directions</td><td></td><td>causal No, for most of training—inert in 43/48 efficacy of reading cells, counterproductive in 4 early cells</td></tr></table>

Why the monotone-emergence hypotheses failed. Both forks assumed that “controllability” has a single onset that shifts with scale or consolidates with time. The three-track decomposition shows the estimand itself was the problem: internal readability is present ab initio (no onset to shift), behavioral readability shifts the “wrong” way (later at larger scales), and causal efficacy never consolidates monotonically (34.55% of units regress). Emergence claims about mechanism need per-track estimands; single-axis onset regressions conflate clocks that run at different speeds and sometimes backwards.

Headroom as the mechanistic bottleneck. The headroom result (§4.5) suggests a specific account: as models scale, the probe direction carries an ever larger share of the residual stream—the variable is increasingly written into the representation—while the readout circuit’s reliance on that component does not keep pace, so the same absolute edit buys an ever smaller relative push. This inverts the naive intuition that bigger models should be easier to steer because their representations are cleaner. If the account generalizes, effective steering at scale requires either much larger edits (with the side effects that implies) or directions chosen by causal criteria rather than readability criteria—echoing the probing literature’s distinction between information and use<sup>[15][18]</sup>.

The isolated pulse and the re-wiring-window hypothesis. The 12B step-8k pulse (+2.49) sits where the behavioral track is mid-transition (AUROC 0.632 → 0.561 around that window) and vanishes once behavioral readability stabilizes. One reading—explicitly a hypothesis for future work, not a claim—is that brief causal windows open while the readout circuit is being rewired and close once consolidation completes; the early backfire cells, which likewise cluster near behavioral transitions, fit the same shape with opposite sign. If real, such windows would matter for intervention timing. Our grid is too coarse to confirm or refute them, we draw no timing inference, and the finding’s value here is to motivate denser developmental sampling.

Methodological contribution. Beyond the scientific claims, the study demonstrates that conference-grade developmental experiments can be run under laboratory discipline: preregistered decision trees with pre-written failure branches, golden-standard instrument checks, content-hash gates on weights, md5-chained audit ledgers, and atomic, restart-safe execution. Two of these gates fired during the run (the corrupted 2.8B lineage, Appendix A; a cloud restart at

174/192 units with zero data loss), and in both cases the discipline, not heroics, is what preserved the result’s integrity.

## 6 Limitations and Threats to Validity

We organize the limitations as the objections we expect, with the defenses already in place.

“The intervention is too weak; a stronger method would show effects.” This is a design decision, defended in §3.4: the additive single-site edit is the minimum-energy, signal-aligned path and thus the sharpest test of direction generalization; dose–response curves are flat across the strength grid, so the nulls are not under-dosed; and our claims are scoped to runtime activation control, not weight-space editing<sup>[21]</sup> or trained distributed interventions<sup>[23][24]</sup>, which answer a different question and remain open future work.

“The tasks are too narrow.” The four families are controlled answer-format contrasts by construction—the minimal design that makes per-track estimands comparable across 192 units. Open-ended generation and multi-turn control are exactly the settings where readability-tocausality transfer matters most, and extending the grid there is the natural next step.

“The negative verdict may be a power artifact (n = 6 sizes).” The verdict reports honest confidence intervals rather than binary significance, and the positive structure does not depend on the underpowered axis: the three-track dissociation, the 11/11 ordering, and the headroom/write-in contrast are per-cell and per-unit results with n = 48, 191, and 24 respectively, not regressions on six points. INDETERMINATE means the pre-registered onset hypotheses failed, not that nothing was measured.

“The pulse may be noise.” Agreed, and treated as such: it is one cell in 48, we make no timingwindow inference (§4.4), and it is absent from the headline claims. The early-backfire cells, by contrast, are four in number, same-signed, and time-clustered—a pattern the null would not produce with appreciable frequency.

“Resolution coarsens with scale.” True and disclosed: grows monotonically from 0.00398ε<sub>m</sub> (160M) to 0.0328 (12B) (Figure 5b), so “null-equivalent” at 12B means |effect| below a coarser floor. We publish the calibration rather than absorb it, and every null statement in the paper is relative to the per-size floor shown.

Scope. One suite (Pythia) carries the main claims; the OLMo-2 replication is two sizes and capped at “weak” by the frozen rule. Eight checkpoints leave sub-grid developmental events unresolved by construction. INDETERMINATE on both axes means the two forks failed, not that no scale-time structure exists; the structure reported here is explicitly descriptive, and the paper’s title and claims avoid “law” language accordingly.

## 7 Reproducibility and Audit

The full pipeline is configuration-driven and deterministic given its seeds; unit seeds derive from md5 digests of unit identifiers (the language built-in hash() is banned pipeline-wide), and bytecode caching is disabled on compute nodes. Every result file is written atomically (incomplete marker → payload → complete marker), so an interrupted run resumes by skipping verified-complete units: a cloud instance restart at 174/192 test units resumed bit-identically with zero data loss. All derived artifacts are hash-manifested—dev phase: 1,219 files, manifest md5 938ef0741281e1c06deed785a5e9e13c; test phase: 384 files, manifest md5 e338f1421361cb22ed18986951852351; final decision artifact p5c2\_result.json, md5 1fe8a3fa2b3904df7a17d9dc4d91c583. The audit ledger is append-only, snapshot-backed after every

entry, and records every protocol-relevant event: the content-hash gate that quarantined the corrupted 2.8B lineage (Appendix A), the aggregation hotfix (an estimand-identical fallback chain, applied and logged before re-aggregation; the failure occurred before any write), the post-incident manifest refresh, the restart recovery, and the final seal. The only code change made after criterion data existed was that hotfix; it is logged with before/after hashes of every touched file, and the aggregation it enabled was then run exactly once. Code, configurations, manifests, and the audit ledger will be released with the camera-ready.

One audit event deserves visibility beyond the appendix: the mandatory per-checkpoint contenthash gate caught a silently corrupted public Pythia-2.8B lineage—distinct checkpoint labels with byte-identical weight content—before any criterion data were produced on it, and the detection was later corroborated independently by a community report and a lineage audit<sup>[30][31]</sup> (details in Appendix A). Any developmental study on public checkpoint suites is one silent mirror-corruption away from invalidating its time axis; we treat content hashing as non-optional instrumentation and release the gate with the pipeline.

## References

1. Biderman, S., Schoelkopf, H., Anthony, Q. G., Bradley, H., O’Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., Skowron, A., Sutawika, L., & van der Wal, O. (2023). Pythia: A suite for analyzing large language models across training and scaling. In Proceedings ofthe 40th International Conference on Machine Learning (PMLR 202, pp. 2397–2430).

2. OLMo Team. (2024). 2 OLMo 2 Furious. arXiv:2501.00656.

3. Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., & Amodei, D. (2020). Scaling laws for neural language models. arXiv:2001.08361.

4. Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., et al. (2022). Training compute-optimal large language models. In Advances in Neural Information Processing Systems, 35, 30016–30030.

5. Ruan, Y., Maddison, C. J., & Hashimoto, T. (2024). Observational scaling laws and the predictability of language model performance. In Advances in Neural Information Processing Systems, 37, 78068–78110.

6. Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., Yogatama, D., Bosma, M., Zhou, D., Metzler, D., et al. (2022). Emergent abilities of large language models. Transactions on Machine Learning Research.

7. Schaeffer, R., Miranda, B., & Koyejo, S. (2023). Are emergent abilities of large language models a mirage? In Advances in Neural Information Processing Systems, 36, 55565–55581.

8. Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization beyond overfitting on small algorithmic datasets. arXiv:2201.02177.

9. Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress measures for grokking via mechanistic interpretability. In International Conference on Learning Representations.

10. Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Askell, A., Bai, Y., Chen, A., et al. (2022). In-context learning and induction heads. Transformer Circuits Thread.

11. Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., et al. (2021). A mathematical framework for transformer circuits. Transformer Circuits Thread.

12. Wang, K. R., Variengien, A., Conmy, A., Shlegeris, B., & Steinhardt, J. (2023). Interpretability in the wild: A circuit for indirect object identification in GPT-2 small. In International Conference on Learning Representations.

13. Hewitt, J., & Manning, C. D. (2019). A structural probe for finding syntax in word representations. In Proceedings of NAACL-HLT (pp. 4129–4138).

14. Belinkov, Y. (2022). Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1), 207–219.

15. Hewitt, J., & Liang, P. (2019). Designing and interpreting probes with control tasks. In Proceedings of EMNLP-IJCNLP (pp. 2733–2743).

16. Pimentel, T., Valvoda, J., Maudslay, R. H., Zmigrod, R., Williams, A., & Cotterell, R. (2020). Informationtheoretic probing for linguistic structure. In Proceedings of ACL (pp. 4609–4622).

17. Kumar, A., Tan, C., & Sharma, A. (2022). Probing classifiers are unreliable for concept removal and detection. In Advances in Neural Information Processing Systems, 35, 17994–18008.

18. Elazar, Y., Ravfogel, S., Jacovi, A., & Goldberg, Y. (2021). Amnesic probing: Behavioral explanation with amnesic counterfactuals. Transactions ofthe Associationfor Computational Linguistics, 9, 160–175.

19. Lasri, K., Pimentel, T., Lenci, A., Poibeau, T., & Cotterell, R. (2022). Probing for the usage of grammatical number. In Proceedings ofACL (pp. 8818–8831).

20. Vig, J., Gehrmann, S., Belinkov, Y., Qian, S., Nevo, D., Singer, Y., & Shieber, S. (2020). Investigating gender bias in language models using causal mediation analysis. In Advances in Neural Information Processing Systems, 33, 12388–12401.

21. Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems, 35, 17359–17372.

22. Geiger, A., Lu, H., Icard, T., & Potts, C. (2021). Causal abstractions of neural networks. In Advances in Neural Information Processing Systems, 34. arXiv:2106.02997.

23. Geiger, A., Wu, Z., Lu, H., Rozner, J., Kreiss, E., Icard, T., Goodman, N., & Potts, C. (2022). Inducing causal structure for interpretable neural networks. In Proceedings of the 39th International Conference on Machine Learning (PMLR 162, pp. 7324–7338).

24. Wu, Z., Arora, A., Wang, Z., Geiger, A., Jurafsky, D., Manning, C. D., & Potts, C. (2024). ReFT: Representation finetuning for language models. In Advances in Neural Information Processing Systems, 37, 63908–63962.

25. Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., & MacDiarmid, M. (2023). Activation addition: Steering language models without optimization. arXiv:2308.10248.

26. Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., et al. (2023). Representation engineering: A top-down approach to AI transparency. arXiv:2310.01405.

27. Li, K., Patel, O., Viégas, F., Pfister, H., & Wattenberg, M. (2023). Inference-time intervention: Eliciting truthful answers from a language model. In Advances in Neural Information Processing Systems, 36. arXiv:2306.03341.

28. Zhang, F., & Nanda, N. (2024). Towards best practices of activation patching in language models: Metrics and methods. In International Conference on Learning Representations. arXiv:2309.16042.

29. Hernandez, E., Sharma, A. S., Haklay, T., Meng, K., Wattenberg, M., Andreas, J., Belinkov, Y., & Bau, D. (2024). Linearity of relation decoding in transformer language models. In International Conference on Learning Representations.

30. Li, W. (2026, February 25). Learning (to reproduce Pythia 2.8b) pretraining. zywilliamli.com. https://zywilliamli.com/writings/writings-learning-pretraining

31. EleutherAI. (2026, March 6). Pythia-2.8B PT CheckPoints is same between steps 0 to steps 6300 (Issue #195). GitHub, EleutherAI/pythia.

## Appendix A The weight-corruption incident: a mandatory content-hash gate

Developmental studies treat public checkpoint suites as scientific instruments, and instruments can be silently miscalibrated. Our pre-registered gate R-8C2j requires, for every model×checkpoint file entering the pipeline, a content hash of the weight payload, with all hashes within a lineage required to be pairwise distinct across training steps. The gate fired on the Pythia-2.8B lineage: the widely mirrored generation-5 pythia-2.8b (“standard”) revision contains safetensors and .bin files that are byte-for-byte clones of one another, and its shards are same-step clones of the deduped branch—i.e., distinct checkpoint labels with identical weight content<sup>[30]</sup>. The finding was later corroborated independently: a community report documents the affected files as byte-for-byte identical across early steps (0–6,300)<sup>[31]</sup>, and a lineage audit reconstructs five generations of mirror copies, confirming that the v0 (2.7B-equivalent) lineage is intact while the generation-5 standard revision is not<sup>[30]</sup>. Our detection chain (content-hash gate → lineage quarantine → revision pinning) matches the external account point for point. All 2.8B measurements reported here use

the verified-healthy pinned revision; the contaminated manifests were voided and refreshed, and the incident is recorded in the audit ledger. We recommend that any checkpoint-suite study adopt per-checkpoint content hashing as a hard gate: the failure mode is silent, common enough to hit us on a flagship suite, and undetectable by file-size or load-success checks alone.

## Appendix B Full causal-efficacy grid

Table B1. Causal efficacy $z _ { e }$ for all 48 model×checkpoint cells (test phase). Bold: $| z | \geq 2$ (five cells; one positive, four negative, all negative cells early in training).
<table><tr><td>Model</td><td>1k</td><td>2k</td><td>4k</td><td>8k</td><td>16k</td><td>32k</td><td>64k</td><td>143k</td></tr><tr><td>160M</td><td>-0.59</td><td>-0.13</td><td>+0.17</td><td>-0.95</td><td>+0.20</td><td>+0.55</td><td>-0.16</td><td>-0.60</td></tr><tr><td>410M</td><td>-0.83</td><td>-2.65</td><td>-0.07</td><td>+0.25</td><td>-0.03</td><td>+1.17</td><td>-0.79</td><td>+1.22</td></tr><tr><td>1B</td><td>-1.81</td><td>-0.57</td><td>-0.67</td><td>+1.21</td><td>+1.38</td><td>+0.13</td><td>+0.12</td><td>+0.97</td></tr><tr><td>2.8B</td><td>-2.68</td><td>-2.48</td><td>-0.00</td><td>-0.91</td><td>+0.75</td><td>+0.72</td><td>+0.45</td><td>-0.56</td></tr><tr><td>6.9B</td><td>-1.88</td><td>-0.55</td><td>-0.36</td><td>+0.80</td><td>-0.17</td><td>+0.70</td><td>+0.51</td><td>+0.45</td></tr><tr><td>12B</td><td>-0.12</td><td>-2.43</td><td>+0.63</td><td>+2.49</td><td>-0.31</td><td>+1.04</td><td>-0.06</td><td>-0.56</td></tr></table>

Null bands are per-cell, from norm- and site-matched random-direction edits; $z _ { e }$ is computed on paired effects (Eq. 4). The dev-phase counterpart of the 12B step-8k cell is +2.04, so the pulse is present in both phases. The dev↔test probe-discrepancy flag $( | \mathrm { A U R O C _ { t e s t } } - \mathrm { A U R O C _ { d e v } } | > 0 . 1 )$ fired in 3 of 192 units, all at 12B, all below 0.12.

## Appendix C Resolution, behavioral readability, and configuration summary

Table C1. Per-size instrument noise floor $\varepsilon _ { m }$ (median absolute within-unit null effect) and final-checkpoint normalized write-in for the three largest sizes.∣^c∣ $\varepsilon _ { m }$ is strictly monotone in parameters.
<table><tr><td>Model</td><td>Parameters</td><td> $\varepsilon _ { m }$ </td><td>|c| @ 143k</td></tr><tr><td>160M</td><td> $1 . 6 2 \times 1 0 ^ { 8 }$ </td><td>0.00398</td><td>一</td></tr><tr><td>410M</td><td> $4 . 0 5 \times 1 0 ^ { 8 }$ </td><td>0.00597</td><td>一</td></tr><tr><td>1B</td><td> $1 . 0 1 \times 1 0 ^ { 9 }$ </td><td>0.00619</td><td>一</td></tr><tr><td>2.8B</td><td> $2 . 7 8 \times 1 0 ^ { 9 }$ </td><td>0.01179</td><td>0.070%</td></tr><tr><td>6.9B</td><td> $6 . 8 5 \times 1 0 ^ { 9 }$ </td><td>0.0286</td><td>0.109%</td></tr><tr><td>12B</td><td> $1 . 1 8 \times 1 0 ^ { 1 0 }$ </td><td>0.0328</td><td>0.108%</td></tr></table>

Table C2. Behavioral readability (output-level answer-score AUROC, dev split) over checkpoints for the four sizes with complete series. The developmental gradient steepens with scale; internal probe readability (not shown) is saturated ≈ 1.000 at all sizes and checkpoints (dev minimum 0.990, at 12B).
<table><tr><td>Model</td><td>1k</td><td>2k</td><td>4k</td><td>8k</td><td>16k</td><td>32k</td><td>64k</td><td>143k</td></tr><tr><td>1B</td><td>0.647</td><td>0.475</td><td>0.478</td><td>0.527</td><td>0.583</td><td>0.607</td><td>0.665</td><td>0.711</td></tr><tr><td>2.8B</td><td>0.534</td><td>0.517</td><td>0.519</td><td>0.621</td><td>0.687</td><td>0.791</td><td>0.857</td><td>0.968</td></tr><tr><td>6.9B</td><td>0.569</td><td>0.481</td><td>0.493</td><td>0.570</td><td>0.775</td><td>0.875</td><td>0.877</td><td>0.896</td></tr><tr><td>12B</td><td>0.524</td><td>0.512</td><td>0.632</td><td>0.561</td><td>0.662</td><td>0.752</td><td>0.815</td><td>0.909</td></tr></table>

Configuration summary. Probes: logistic regression at the pre-registered best site per unit, frozen after dev; interventions: additive unit-norm edits, strengths λ on a frozen grid; nulls: random directions matched for site and norm; decision: WLS with $1 / \mathrm { s e } ^ { 2 }$ weights, 95% CI exclusion, majority vote on the time axis; degradation rules as in §3.3. Compute: ≈240 instance-hours on 2×A100-SXM4-40GB. Seeds: md5-derived per unit. The ordering and biassurface aggregates (§4.3, §4.6) are computed mechanically by the frozen pipeline from the manifests in §7.