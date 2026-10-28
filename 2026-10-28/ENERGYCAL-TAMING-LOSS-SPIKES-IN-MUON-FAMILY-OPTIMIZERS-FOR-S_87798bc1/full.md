# ENERGYCAL: TAMING LOSS SPIKES IN MUON-FAMILY OPTIMIZERS FOR SPARSE MOE TRAINING

Masato Fujitake

FA Research, Fast Accounting Co., Ltd.

fujitake@fastaccounting.co.jp

## ABSTRACT

Sparse mixture-of-experts (MoE) language-model training can exhibit large loss spikes with Muon-family optimizers. We introduce EnergyCal, a lightweight rule that attenuates routedexpert updates according to the normalized squared Frobenius norm of their finite-step Newton–Schulz directions. EnergyCal leaves the spectral transform unchanged and adds no persistent optimizer state. We evaluate it on a 4B-parameter MoE model. Against a development-selected, lower-learning-rate MuonH recipe, EnergyCal–MuonH reduces mean thresholded lossspike burden by 99.51% and final geometric-mean perplexity by 4.06%, with a modest throughput cost. With plain Muon at matched nominal learning rate and weight decay, the reduction in mean spike burden averages 95.98% across the two tested exponents, accompanied by small endpoint improvements. These findings support stabilization both with and without Hyperball in the evaluated setting.

Index Terms— Muon, MuonH, mixture-of-experts, energy calibration, training stability

## 1. INTRODUCTION

Sparse mixture-of-experts (MoE) language models route tokens unevenly across experts, creating time-varying optimization conditions [1]. This setting can be challenging for Muon-family optimizers, which use Newton–Schulz (NS) iterations to transform momentum-based updates. Muon has shown promise for language-model pretraining [2–4], while Hyperball introduces explicit control over weight and update norms [5]. Its Muon variant, MuonH, normalizes update directions and projects weights onto a fixed-radius sphere. In our custom 4B-parameter DeepSeekMoE-style model, both Muon and MuonH exhibit intermittent loss spikes, although all full-schedule runs complete. We investigate whether these loss excursions can be mitigated through a simple modification of routed-expert updates.

An expert can retain nonzero momentum even when no tokens are assigned to it in the current step. This motivates examining expert-wise update magnitudes, particularly under MuonH’s fixed-norm normalization. We propose EnergyCal, a lightweight rule that attenuates each routed-expert update according to the normalized squared Frobenius norm of its finite-step NS direction. Figure 1 illustrates the method. The coefficient is bounded between zero and one and is applied continuously during training. EnergyCal leaves the NS transform, momentum, and router objective unchanged, requires no token-utilization statistics, and adds no persistent optimizer state. Its energy statistic is not scale-invariant stable rank: low energy can reflect direction concentration, amplitude, or both.

![](images/cec0360304dcab345acd173ca3be7c0de48db03ff48d28f1f4366ba7585ca0c2.jpg)  
Fig. 1. EnergyCal overview. (1) Sparse routing creates uneven expert traffic. (2) Each routed matrix’s post-NS energy determines $q _ { e } ;$ when $q _ { e } > 0 ,$ , scaling preserves the individual NS direction, not necessarily the displacement after Hyperball projection or writeback. (3) Continuous calibration reduces observed thresholded loss-spike burden. Curves are schematic, not measurements; coefficients use direction energy, not token counts.

We evaluate EnergyCal in two separate full-schedule studies, each using three initialization seeds. First, a frozen EnergyCal–MuonH recipe reduces mean thresholded lossspike burden by 99.51% and final geometric-mean perplexity by 4.06% relative to a development-selected, lower-learningrate MuonH recipe, with 1.62% lower mean throughput. Second, applying EnergyCal to plain Muon at matched nominal learning rate and weight decay reduces mean spike burden by 95.98% on average across the two tested exponents, with small endpoint improvements. Together, these results show that stabilization is observed both with and without Hyperball in the evaluated MoE setting. However, plain Muon retains intermediate-quality and speed advantages in the first study, and different evaluation banks prevent a direct ranking of the calibrated optimizer families.

Our contribution is an empirical evaluation of one-sided, post-NS energy attenuation across MuonH and plain Muon. Unlike utilization- or alignment-guided expert scaling methods [6, 7], EnergyCal relies only on the energy of each logical matrix’s NS direction. The results do not establish general optimizer superiority or identify the cause of individual spikes: single-update interventions fail to rescue two examined events, and the plain-Muon study lacks a routed-only uniform-scaling control to separate energy-dependent attenuation from a simpler reduction in update magnitude. We therefore focus on the measured stability benefits and their practical trade-offs within the tested setting; Section 7 discusses these evidence boundaries.

## 2. RELATED WORK

Muon-family optimization and Hyperball. Muon applies a short Newton–Schulz (NS) iteration to momentum-based matrix updates [2]. Large-scale studies identify weight decay and update scaling as important for Muon pretraining [4]. Our plain-Muon baseline uses a shape factor of one rather than the full Moonlight recipe. Hyperball controls update and weight radii; its Muon variant, MuonH, normalizes trial directions and projects weights onto a fixed-radius sphere [5]. EnergyCal changes routed-expert step magnitudes after NS without changing the spectral transform.

Expert-wise update scaling. Existing methods scale expert updates using other signals. Excitation uses expert utilization [6], while ExpertMuon-Compass uses postorthogonalization gradient alignment across experts and rows [7]. EnergyCal instead computes an independent coefficient $q ~ \in ~ [ 0 , 1 ]$ from each routed matrix’s normalized squared post-NS Frobenius norm, without utilization inputs or cross-expert normalization. For $q > 0$ , scalar scaling preserves the NS direction; when $q \ = \ 0$ , the directional trial vanishes. Neither case guarantees the direction of the stored parameter displacement after Hyperball projection or low-precision writeback. Direction preservation itself is not unique to EnergyCal.

Magnitude and spectral corrections. NorMuon uses neuronlevel normalization after orthogonalization [8], and TrasMuon uses RMS calibration and energy-based clipping [9]. Thus, energy-aware scaling is not new by itself. Pion changes NS iterations [10], while SoftMuon uses smooth spectral relaxation [11]. EnergyCal instead applies a matrix-level scalar after the unchanged NS transform. We do not report native-method head-to-head comparisons with these alternatives.

Training stability in sparse models. Training instability in sparse MoE models has motivated router-centered approaches, including ST-MoE’s router z-loss and methods for reducing routing fluctuations [12, 13]. Spike-aware optimization has also been studied in dense language-model training through momentum resets and gradient clipping [14]. Unlike these approaches, EnergyCal attenuates routed-expert updates using post-NS direction energy without modifying routing objectives or introducing spike-triggered interventions.

Finite-step orthogonalization. Recent analyses characterize the convergence and spectral behavior of Muon with finitestep Newton–Schulz iterations [15, 16]. MuonEq further investigates pre-NS equilibration to improve the geometry of orthogonalized updates [17]. EnergyCal instead leaves the NS transform unchanged and applies a matrix-level attenuation coefficient to its returned direction.

Evaluation and optimizer comparisons. Optimizer rankings depend on tuning and training budget [18]; angular effective learning rates also complicate the interpretation of Hyperball’s nominal learning rate [19]. We therefore report final quality, intermediate quality, loss-spike burden, and throughput separately. Matched nominal learning rate does not imply matched parameter movement, and fixed-token quality is distinct from equal-wall-time quality.

## 3. ENERGY-CALIBRATED UPDATES

## 3.1. Logical matrices and the base update

Let $W \in \mathbb { R } ^ { m \times n }$ denote a logical parameter matrix, $G _ { t }$ its gradient after accumulation, distributed reduction, and common clipping, and $B _ { t }$ its momentum. The evaluated implementation uses Nesterov momentum:

$$
B _ { t } = \beta B _ { t - 1 } + G _ { t } , \qquad A _ { t } = G _ { t } + \beta B _ { t } ,\tag{1}
$$

with $\beta = 0 . 9 5$ . A five-step quintic Newton–Schulz (NS) map produces the update direction $D _ { t } = \mathrm { N S } _ { 5 } ( A _ { t } )$ . NS arithmetic uses BF16, while momentum and norm calculations use FP32. Because the iteration is finite, $D _ { t }$ is an approximate orthogonalized direction, not an exact polar factor. Implementation details and numerical guards are provided in Appendix A.

Gate and up projections share fused storage for efficient forward execution but are optimized as separate logical matrices by both MuonH and plain Muon. Each routed gate, up, and down matrix has $d = \operatorname* { m i n } ( m , n ) = 2 5 6$ . Attention and shared-expert matrices also receive matrix-optimizer updates, whereas embeddings/tied output weights, router weights, and vector parameters use the Adam fallback.

MuonH additionally constrains weight and update norms. For the initial radius $R ~ = ~ \| W _ { 0 } \| _ { \mathrm { F } } ,$ define $\mathcal { P } _ { R } ( X ) \ =$ $R X / \operatorname* { m a x } ( \| X \| _ { \mathrm { F } } , \epsilon _ { H } )$ . The canonical MuonH update forms

$$
\Delta _ { t } ^ { \mathrm { t r i a l } } = - \eta _ { t } R \frac { D _ { t } } { \operatorname* { m a x } ( \| D _ { t } \| _ { \mathrm { F } } , \epsilon _ { H } ) } ,\tag{2}
$$

$$
\widetilde { W } _ { t + 1 } = \mathcal { P } _ { R } ( W _ { t } + \Delta _ { t } ^ { \mathrm { t r i a l } } ) ,\tag{3}
$$

followed by BF16 parameter writeback using the common dithered precision policy. The trial update has a prescribed Frobenius length when its denominator guard is inactive, but the subsequent projection and writeback can change the resulting parameter displacement. We therefore distinguish the trial displacement, FP32 projected chord, BF16 stored chord, and resulting function change. All MuonH-family comparisons use the same projection and writeback policy.

## 3.2. Expert-wise energy calibration

EnergyCal introduces a one-sided attenuation of routed-expert updates based on the energy of their post-NS directions. For each logical matrix, define

$$
\rho _ { t } = \frac { \| D _ { t } \| _ { \mathrm { F } } ^ { 2 } } { d } ,\tag{4}
$$

$$
q _ { t } = \mathrm { c l i p } \left[ \left( \frac { \rho _ { t } } { \rho _ { \mathrm { r e f } } } \right) ^ { p } , q _ { \mathrm { m i n } } , 1 \right] .\tag{5}
$$

Here, $\rho _ { t }$ is the normalized squared Frobenius norm of the finite-step NS direction, $\rho _ { \mathrm { r e f } }$ is a reference energy, and p controls the strength of attenuation. With $0 \leq q _ { \mathrm { m i n } } \leq 1$ and $p > 0$ , the coefficient satisfies $q _ { t } \in [ 0 , 1 ]$ . For the evaluated choice $q _ { \mathrm { m i n } } = 0$ , directions with $\rho _ { t } < \rho _ { \mathrm { r e f } }$ receive $q _ { t } < 1$ while directions at or above the reference receive no additional attenuation from EnergyCal.

For MuonH, EnergyCal replaces $\eta _ { t }$ in Eq. (2) with $q _ { t } \eta _ { t }$ only for routed-expert matrices. All other matrices retain $q _ { t } = 1$ . As illustrated in Fig. 1, positive scaling preserves the orientation of the individual NS direction, but not necessarily that of the projected or stored parameter displacement. Setting $q _ { t } = 0$ nulls the trial update, not necessarily the final stored parameter change. The figure uses expert indices e, whereas the equations use update indices t.

The coefficient is computed independently for each routed logical matrix at every update. It uses neither token-utilization statistics nor comparisons across experts. The NS transform, momentum, router objective, attention and shared-expert updates, and Adam fallback remain unchanged. No additional gradient history or persistent optimizer state is introduced. MuonH reuses its existing direction norm, whereas plain Muon requires an additional norm reduction.

The primary EnergyCal–MuonH recipe uses $p = 1 , \rho _ { \mathrm { r e f } } =$ 0.8, and $q _ { \mathrm { m i n } } = 0$ , with calibration active from the first update and no sequence auxiliary loss. These settings were fixed during development and not reselected on the confirmation seeds. The energy statistic is used as an attenuation signal, not as a calibrated probability of update correctness or a validated signal-to-noise estimate.

## 3.3. Transfer to plain Muon

The same energy coefficient can be applied to plain Muon without Hyperball normalization or projection. The resulting

Algorithm 1 EnergyCal–MuonH on one logical matrix   
Require: $\overline { { W , G , B , R , \eta , \beta , p , \rho _ { \mathrm { r e f } } } } ,$ q<sub>min</sub>   
1: $B  \beta B + G ; A  G + \beta B$   
2: $D \gets \mathrm { N S } _ { 5 } ( A ) ; s \gets \| D \| _ { \mathrm { F } }$ in FP32   
3: $q \gets 1$   
4: if W is a routed-expert matrix then   
5: $q  \mathrm { c l i p } ( ( s ^ { 2 } / ( d \rho _ { \mathrm { r e f } } ) ) ^ { p } , q _ { \mathrm { m i n } } , 1 )$   
6: end if   
7: $V \gets W - \eta R q D / \operatorname* { m a x } ( s , \epsilon _ { H } )$   
8: $\widetilde { W } \gets R V / \operatorname* { m a x } ( \| V \| _ { \mathrm { F } } , \epsilon _ { H } )$   
9: Write $\widetilde { W }$ using the shared precision policy

update is

$$
\widetilde { W } _ { t + 1 } = ( 1 - \eta _ { t } \lambda ) W _ { t } - \eta _ { t } s _ { W } q _ { t } D _ { t } ,\tag{6}
$$

followed by the common writeback policy. Here, λ denotes decoupled weight decay and $s w$ is the matrix-shape factor. All evaluated plain-Muon runs use $s _ { W } = 1$ , unlike the scaleadjusted large-model recipe of Liu et al. [4].

As in MuonH, energy is measured independently for each routed gate, up, and down matrix before any shape-factor scaling. Only routed matrices may have $q _ { t } \neq 1 ;$ weight decay, momentum, non-routed updates, and the Adam fallback remain unchanged. Unlike MuonH, plain Muon does not subsequently normalize the update to a prescribed Frobenius length or project the weights onto a sphere. Consequently, the same coefficient acts on a different base update and does not imply the same parameter displacement. In particular, $q _ { t } = 0$ removes the NS update term but does not disable weight decay.

We evaluate $p \in \{ 0 . 5 , 1 \}$ with $\rho _ { \mathrm { r e f } } = 0 . 8$ and $q _ { \mathrm { m i n } } = 0$ without retuning the nominal learning rate. In this setting, EnergyCal adds energy-dependent attenuation to plain Muon’s existing direction-magnitude dependence, rather than compensating for Hyperball normalization. Whether this improves training stability is therefore evaluated empirically.

## 3.4. Geometric motivation, not a causal theorem

The motivation for post-NS energy calibration is partly geometric. For a nonzero direction D and an inactive denominator guard, the MuonH trial update satisfies

$$
\mathrm { s r } ( D ) = \frac { \left\| D \right\| _ { \mathrm { F } } ^ { 2 } } { \left\| D \right\| _ { 2 } ^ { 2 } } , \qquad \left\| \Delta ^ { \mathrm { t r i a l } } \right\| _ { 2 } = \frac { \eta R q } { \sqrt { \mathrm { s r } ( D ) } } .\tag{7}
$$

Thus, at a fixed trial Frobenius length $\eta R q ,$ , a smaller stable rank corresponds to a larger spectral concentration of the update. However, the operator norm remains bounded by $\lVert \bar { \boldsymbol { \Delta } } ^ { \mathrm { t r i a l } } \rVert _ { 2 } \leq \eta R q$ . This identity concerns the trial update, not the projected or BF16-stored displacement, and does not bound the nonlinear network’s loss change or identify the cause of individual spikes.

Importantly, the energy used by EnergyCal is not stable rank. The two quantities satisfy

$$
\rho ( D ) = { \frac { \operatorname { s r } ( D ) } { d } } \left\| D \right\| _ { 2 } ^ { 2 } .\tag{8}
$$

Unlike stable rank, $\rho ( D )$ is not invariant under rescaling $D \mapsto a D$ . Small energy can therefore arise from concentrated singular values, small direction amplitude, or both. Energy and normalized stable rank become approximately proportional only when the operator norm is independently known to remain approximately constant.

The exponent also changes how attenuation interacts with MuonH normalization. In the unclipped, unguarded regime, $p = 1 / 2$ gives

$$
\frac { q D } { \| D \| _ { \mathrm { F } } } = \frac { D } { \sqrt { d \rho _ { \mathrm { r e f } } } } ,\tag{9}
$$

so the Frobenius normalization cancels algebraically. The trial operator norm still depends on $\| D \| _ { 2 } .$ , however, and this identity does not predict which exponent yields better languagemodel training outcomes. The primary MuonH choice $p = 1$ is empirical.

These relations motivate energy-dependent attenuation but do not establish a causal mechanism for loss spikes. In partic ular, Eq. (7) does not describe the unnormalized plain-Muon update, and a zero MuonH trial is not equivalent to an exact weight freeze because projection and low-precision writeback remain active. Our single-update intervention experiments explicitly distinguish these operations.

## 4. EXPERIMENTAL PROTOCOL

## 4.1. Model, data, and training setup

We train a custom 4B-parameter DeepSeekMoE-style language model from scratch, with 24 layers, hidden width 1,536, 128 routed experts of width 256 and top-2 routing, and one shared expert per layer. Its input and output embeddings are tied. The routed/shared design follows DeepSeekMoE [20], with a separate traffic-driven router-bias controller [21, 22]. Neither mechanism enters EnergyCal’s coefficient.

Each run uses eight H100 80GB GPUs, 2,048-token sequences, and 131,072 tokens per optimizer update. A complete run has 10,000 updates (1.31B tokens), with 500 warmup updates followed by cosine decay to 10% of peak learning rate. All methods use gradient clipping at norm 1, five NS iterations, and the same logical matrix partitions. These experiments do not claim compute-optimal or fully converged pretraining. Appendix A gives the relevant implementation details.

Training uses FineWeb sample-10BT [23], with a deterministic train/validation partition. All final runs share the same data order; the independently varied initialization seeds do not constitute independently sampled corpora. Validation uses a fixed 524,288-token bank evaluated every 500 updates. Banks are fixed within each comparison but differ between development, Cohort H, and Cohort M. They are not certified previously unused external test sets across the complete exploratory history.

Table 1. Frozen Cohort H recipes. These are developmentselected, best-tested settings rather than a coefficient-only factorial. Fallback peak LR is 0.001; weight decay applies only to eligible parameters.
<table><tr><td>Recipe</td><td>Matrix LR Matrix WD</td><td>Fallback WD</td></tr><tr><td>MuonH</td><td>0.005</td><td>0 0</td></tr><tr><td>EnergyCal  $( p = 1 )$ </td><td>0.010</td><td>0 0</td></tr><tr><td>Muon + WD</td><td>0.005</td><td>0.1 0.1</td></tr></table>

## 4.2. Cohort H: development and confirmation

Development screening evaluates the MuonH and plain-Muon learning-rate candidates over two initialization runs, using 3,000 updates of the 10,000-update schedule. Each baseline is selected by mean screening-endpoint validation cross-entropy (CE), not spike frequency. Both selected learning rates are 0.005, at the lower boundaries of their respective bounded search grids. The EnergyCal–MuonH recipe remains fixed at LR 0.01 and $p = 1 ;$ its settings are not reselected using confirmation outcomes.

Confirmation compares these recipes over three new initialization runs for the complete schedule. The primary quality endpoint is final validation CE at fixed token budget. Conditions are paired by initialization and use matched recorded hardware and environments, although host and initialization effects are not fully crossed. Because learning rate differs between MuonH recipes, and the plain-Muon comparator also differs in decay and optimizer formulation, Cohort H estimates recipe-level outcomes rather than the isolated effect of the coefficient.

## 4.3. Cohort M: plain-Muon extension

A separate study compares plain Muon with EnergyCal exponents $p \in \{ 0 . 5 , 1 \}$ over three new initialization runs. All nine runs complete 10,000 updates. Within each initialization, the methods use the same host and evaluation bank. All conditions share matrix LR 0.005, eligible matrix/fallback weight decay 0.1, and fallback LR 0.001. The sole optimizer change is the routed coefficient in Eq. (6); the plain-Muon baseline is rerun under the same implementation as the calibrated variants. No learning-rate retuning or retrospective exponent selection is performed.

Cohort M uses a different evaluation bank and implementation revision from Cohort H. Their absolute CE/PPL values cannot be pooled or used to rank calibrated Muon against MuonH. Appendix C gives additional results.

## 4.4. Stability and statistical reporting

We quantify excursions using a fixed adaptive training-loss detector. For each update $t > 5 0 0$ , let $m _ { t }$ and $a _ { t }$ denote the median and median absolute deviation of the preceding 101 logged losses, excluding the current observation. The detection threshold is

$$
\tau _ { t } = \operatorname* { m a x } \{ m _ { t } + 0 . 5 , 1 . 2 m _ { t } , m _ { t } + 8 \sigma _ { t } \} ,\tag{10}
$$

$$
\sigma _ { t } = \operatorname* { m a x } \{ 1 . 4 8 2 6 a _ { t } , 1 0 ^ { - 6 } \} .\tag{11}
$$

At least 20 preceding observations are required. We flag $L _ { t } > \tau _ { t }$ and merge flags separated by at most 16 steps into an episode. The thresholded loss-spike burden is

$$
B _ { 1 } = \sum _ { t : L _ { t } > \tau _ { t } , \ t > 5 0 0 } ( L _ { t } - m _ { t } ) .\tag{12}
$$

It sums loss above the rolling median at flagged updates, not excess above the threshold or an integral over entire episodes. Its reduction is not a reduction in divergence probability. We also report episodes, maximum post-warmup training loss, and final validation CE, since persistent high loss can escape the adaptive detector.

Repeated-run outcomes are reported as arithmetic mean ± sample standard deviation (SD). Final quality comparisons use $n = 3$ initialization runs per condition; development screening uses $n = 2$ and timing uses three execution repeats. For $x _ { 1 } , \ldots , x _ { n }$

$$
{ \overline { { x } } } = { \frac { 1 } { n } } \sum _ { j = 1 } ^ { n } x _ { j } , \qquad s _ { x } ^ { 2 } = { \frac { \sum _ { j = 1 } ^ { n } ( x _ { j } - { \overline { { x } } } ) ^ { 2 } } { n - 1 } } .\tag{13}
$$

The reported SD is descriptive, not a confidence interval; tokens, experts, and correlated steps are not independent replicates. Paired differences are computed within initialization before aggregation. Tabulated PPL uses per-run exponentiation; percentage reductions in geometric-mean PPL use $1 - \exp ( \overline { { \Delta L } } )$ . Burden percentages use ratios of conditionlevel mean burdens. The 95.98% Cohort M summary is the equally weighted average of the two exponent-specific reductions, not a pooled six-run estimate. With three initializations per condition, no general significance inference is claimed.

## 4.5. Timing and integrity

Throughput is measured separately for Cohort H using three execution repeats per recipe on one eight-GPU node in cyclic method order. Each repeat runs 1,000 updates with evaluation, checkpointing, and heavy diagnostics disabled. Tokens divided by elapsed time over updates 501–1000 gives the run-level throughput; its arithmetic mean and SD are then reported across repeats. Peak allocated memory is maximized across recorded ranks within each run before aggregation. These short benchmarks do not measure equal-wall-time training quality.

All full-schedule runs finish. Recorded configurations, fixed validation banks, and selected state/input checks support within-comparison consistency, but not bitwise-identical trajectories or exhaustive data identity. Infrastructure failures and retries are not counted as optimizer divergence.

## 5. FULL-SCHEDULE RESULTS

## 5.1. Cohort H: recipe-level quality and stability

Cohort H evaluates the frozen EnergyCal–MuonH recipe against development-selected MuonH and plain-Muon baselines over three confirmation seeds. As shown in Table 2, EnergyCal improves both final validation quality and training stability relative to MuonH. Mean final validation cross-entropy (CE) decreases from $3 . 2 7 6 0 \pm 0 . 0 2 7 0 \mathrm { ~ t o ~ } 3 . 2 3 4 5 \pm 0 . 0 0 1 1$ The paired CE difference $\mathrm { i s \ - 0 . 0 4 1 5 \pm 0 . 0 2 6 0 }$ nats/token, corresponding to 4.06% lower geometric-mean perplexity (PPL). All three paired endpoints favor EnergyCal.

The largest improvement is in loss-spike burden. Mean thresholded burden $B _ { 1 }$ decreases from $2 0 4 6 . 0 1 \pm 1 2 8 9 .$ 75 for MuonH to $1 0 . 1 1 \pm 9 . 6 5$ for EnergyCal, a 99.51% reduction in the ratio of condition means. Mean spike episodes decrease from 37.33 ± 4.04 to $3 . 3 3 \pm 2 . 8 9$ . The maximum post-warmup training CE, summarized across runs, also falls from $3 9 . 9 9 \pm 1 2 . 6 4 \mathrm { t o } 5 . 9 2 \pm 1 . 0 7$ , supporting smaller excursions independently of the burden metric. Both endpoint CE and burden improve in all three paired runs. No EnergyCal run triggers the detector between steps 4001 and 10,000, although this does not establish spike-free training.

Plain Muon with weight decay (Muon + WD) provides a stronger comparator. EnergyCal has a paired endpoint CE difference of −0.0065 ± 0.0057 nats/token relative to Muon + WD, equivalent to 0.65% lower geometric-mean PPL. Its mean burden is 85.01% lower, but the burden comparison is not favorable in every seed: plain Muon has lower burden in one of the three runs. These percentages are reductions in conditionlevel mean burdens, not averages of per-run reductions.

All compared recipes complete their training schedules. The results support an improved stability–quality trade-off for the tested EnergyCal–MuonH recipe relative to the development-selected MuonH baseline. Because the recipes differ in learning rate and, for plain Muon, weight decay and optimizer formulation, these comparisons do not isolate the effect of energy calibration or establish a reduction in optimizer failure probability.

## 5.2. Intermediate quality and the late crossover

EnergyCal’s final quality advantage does not extend uniformly across training. Figure 2 shows all fixed-bank validation measurements, including evaluations affected by loss excursions. At step 3000, mean CE is 3.7995 for EnergyCal versus 3.7437 for Muon + WD; at step 5000, the corresponding values are 3.6038 and 3.5479. The mean ordering reverses between the observed 8000- and 8500-step evaluations (supplementary Fig. 4). Thus, EnergyCal’s endpoint advantage emerges late in the schedule rather than persisting throughout training. Appendix B gives the run-level summaries and late-window comparison.

Table 2. Cohort H confirmation: complete 10k schedule, 1.3107B tokens per run. Values are arithmetic mean ± sample SD over $n = 3$ unused initialization seeds. PPL is computed per run before aggregation; episodes and burden $B _ { 1 }$ are per-run quantities. $L _ { \mathrm { m a x } }$ is the maximum post-warmup training loss within each run before aggregation. LRs are selected within bounded development grids.
<table><tr><td>Recipe</td><td>Peak LR</td><td>Final CE</td><td>PPL</td><td>Episodes</td><td> $B _ { 1 }$ </td><td> $L _ { \mathrm { m a x } }$ </td></tr><tr><td>MuonH</td><td>0.005</td><td> $3 . 2 7 6 0 \pm 0 . 0 2 7 0$ </td><td> $2 6 . 4 7 6 \pm 0 . 7 2 1$ </td><td> $3 7 . 3 3 \pm 4 . 0 4$ </td><td> $2 0 4 6 . 0 1 \pm 1 2 8 9 . 7 5$ </td><td> $3 9 . 9 9 \pm 1 2 . 6 4$ </td></tr><tr><td>EnergyCal (p = 1)</td><td>0.010</td><td> $3 . 2 3 4 5 \pm 0 . 0 0 1 1$ </td><td> $2 5 . 3 9 4 \pm 0 . 0 2 7$ </td><td> $3 . 3 3 \pm 2 . 8 9$ </td><td> $1 0 . 1 1 \pm 9 . 6 5$ </td><td> $5 . 9 2 \pm 1 . 0 7$ </td></tr><tr><td> $\mathrm { M u o n + W D }$ </td><td>0.005</td><td> $3 . 2 4 1 0 \pm 0 . 0 0 6 0$ </td><td> $2 5 . 5 6 0 \pm 0 . 1 5 2$ </td><td> $8 . 3 3 \pm 3 . 5 1$ </td><td> $6 7 . 4 1 \pm 5 2 . 7 6$ </td><td> $7 . 6 6 \pm 1 . 8 3$ </td></tr></table>

To summarize intermediate quality, we compute normalized trapezoidal CE over the observed evaluations from steps 500 to 10,000. The per-run summaries yield $3 . 6 2 7 0 \pm 0 . 0 0 1 6$ for EnergyCal, 3.6111±0.0094 for Muon + WD, and 3.9923± 0.3394 for MuonH (mean ± sample SD). By this measure, plain Muon retains better intermediate quality than EnergyCal, despite EnergyCal’s better final endpoint.

This summary depends on evaluation spacing and the timing of individual spikes; it is not an integral of continuously measured validation loss. As a post hoc sensitivity check, averaging the evaluations at steps 9000, 9500, and 10,000 favors EnergyCal over both comparators in every paired run. The predefined final endpoint remains the primary quality measure.

## 5.3. Throughput, memory, and equal-time interpretation

EnergyCal’s stability improvements come with a modest throughput cost. Table 3 reports three execution repeats of each frozen Cohort H recipe. EnergyCal achieves 6 $5 , 9 7 6 . 8 \pm 3 5 3 . 3$ tokens/s, compared with 67,061.4 ± 570.4 for MuonH and 68,458.9±487.7 for Muon + WD. The ratios of mean throughput correspond to reductions of 1.62% and 3.63%, respectively, or increases of 1.64% and 3.76% in estimated fixed-token training time. Peak allocated memory is effectively unchanged relative to MuonH, consistent with the absence of additional persistent optimizer state.

Measured throughput reflects the full training execution, including routing and optimizer-state differences, rather than only the computational cost of the calibration coefficient. Moreover, these short benchmarks do not establish equal-wall-time training quality.

We did not conduct a full wall-clock-matched accuracy experiment. As an illustrative sensitivity check, we assume constant mean throughput and linearly interpolate the measured validation curves. At the estimated time when plain Muon completes 10,000 steps, EnergyCal would reach approximately 9637 steps, with interpolated mean CE 3.2453, compared with plain Muon’s observed 3.2410. Under the same approximation, EnergyCal remains ahead of MuonH. These calculations are not measured time-to-quality results, and the small fixed-token endpoint advantage over plain Muon should not be interpreted as equal-time superiority.

## 5.4. Cohort M: stabilization without Hyperball

Cohort M tests whether EnergyCal also reduces loss excursions when applied to plain Muon, without Hyperball normalization or projection. Unlike Cohort H, this study holds nominal learning rate and weight decay fixed. Table 4 reports the results over three new initialization seeds.

Relative to its plain-Muon baseline, EnergyCal reduces mean thresholded loss-spike burden by 95.98% on average across the two tested exponents. This is the equally weighted average of the two exponent-specific reductions in conditionlevel mean burden, not an average of per-seed percentage reductions (Appendix C.1). Both exponents reduce burden in all three paired runs.

Endpoint improvements are smaller. For $p = 0 . 5$ and $p ~ = ~ 1$ , the paired CE differences relative to plain Muon $\mathrm { a r e - 0 . 0 0 4 2 \pm 0 . 0 0 4 2 }$ and $- 0 . 0 0 3 8 \pm 0 . 0 0 2 8$ nats/token, respectively. These correspond to 0.423% and 0.380% lower geometric-mean PPL. The $p = 0 . 5$ variant improves two of three paired endpoints, while $p = 1$ improves all three. With only three seeds, these small quality differences remain descriptive. The main finding is substantially reduced loss-spike burden without a large penalty in final validation quality.

Neither tested exponent consistently dominates the other. The paired endpoint CE difference between $p = 0 . 5$ and $p = 1$ is only $- 0 . 0 0 0 4 { \scriptstyle \pm 0 . 0 0 1 6 }$ nats/token. We therefore report both settings without selecting an optimal exponent. The observedrange mean validation CE is $3 . 5 7 8 7 \pm 0 . 0 4 2 8$ for plain Muon, $3 . 5 4 9 0 \pm 0 . 0 0 2 7$ for $p = 0 . 5 ,$ , and $3 . 5 5 0 8 \pm 0 . 0 0 0 8$ for $p = 1$ A plain-Muon spike at the 4000-step evaluation contributes substantially to this difference. Supplementary curves retain this observation and show that neither calibrated variant is best at every evaluation.

Coefficient telemetry provides additional context. After averaging the recorded coefficients within each run, the mean coefficients across three initialization runs are $0 . 9 6 7 9 { \scriptstyle \pm 0 . 0 0 2 1 }$ for $p = 0 . 5$ and $0 . 9 5 9 8 \pm 0 . 0 0 1 7$ for $p = 1$ (mean ± sample SD). These summaries describe the applied attenuation, not its causal contribution to stabilization. Zero coefficients also occur, so not every routed direction is retained. However, a routed-only uniform-scaling control has not been evaluated. Consequently, matched nominal learning rate and weight decay do not establish whether the stability gains specifically require energy-dependent coefficient assignment or primarily reflect reduced routed-update magnitude. Appendix C.3 describes the missing control, and Section 7 discusses the resulting limitations.

![](images/73da82d60b73076fcfd74634f8540311f821234cfa5f99e7fb4d93c61630de99.jpg)  
Fig. 2. Fixed-bank validation CE across Cohort H confirmation runs. Lines show means and shaded bands show ±1 sample SD over three initialization seeds. All 20 evaluations are included, including loss excursions. Bands describe run-to-run variation, not confidence intervals. The predefined primary endpoint is step 10,000; development observations are not concatenated with these curves.

Table 3. Cohort H production-style benchmark: arithmetic mean ± sample SD over $n = 3$ execution repeats on one eight-H100 node, not three quality seeds. Throughput is aggregated within each run over steps 501–1000. Optimizer time is summarized first by its within-run median and then across repeats. Peak allocated memory is first maximized over recorded ranks, including initialization, and then summarized across repeats. SD 0.0000 indicates identical recorded peaks, not an absence of hardware variability.
<table><tr><td>Recipe</td><td>Tokens/s</td><td> $\mathrm { O p t i m i z e r } \left( \mathrm { s } \right)$ </td><td> $M _ { \mathrm { a l l o c } } \left( \mathrm { G i B } \right)$ </td></tr><tr><td>MuonH</td><td> $6 7 0 6 1 . 4 \pm 5 7 0 . 4$ </td><td> $0 . 5 4 6 0 \pm 0 . 0 0 1 7$ </td><td> $3 9 . 7 2 2 4 \pm 0 . 0 0 0 0$ </td></tr><tr><td>EnergyCal  $( p = 1 )$ </td><td> $6 5 9 7 6 . 8 \pm 3 5 3 . 3$ </td><td> $0 . 5 4 8 7 \pm 0 . 0 0 1 5$ </td><td> $3 9 . 7 2 2 4 \pm 0 . 0 0 0 0$ </td></tr><tr><td> $\mathrm { M u o n + W D }$ </td><td> $6 8 4 5 8 . 9 \pm 4 8 7 . 7$ </td><td> $0 . 4 8 2 4 \pm 0 . 0 0 1 4$ </td><td> $3 9 . 7 2 4 5 \pm 0 . 0 0 0 0$ </td></tr></table>

results are not extra confirmation seeds.

## 6. DIAGNOSTICS AND LOCAL INTERVENTIONS

## 6.1. Energy, rank, and amplitude ablations

A separate single-initialization audit compares six coefficient conditions at common matrix LR 0.01 over the first 3,000 updates (Table 5). EnergyCal with $p = 1$ has the lowest thresh olded spike burden, whereas a power-iteration-based stablerank coefficient has the lowest endpoint CE. The amplitudeonly and near-zero-gate conditions do not match both outcomes. The amplitude-only run ends during a loss excursion; it is not labeled permanent divergence. These exploratory

The audit independently measures NS direction spectra and actual projected and BF16-stored parameter displacements using CPU float64 SVD. Sampled routed-down matrices include fixed experts and low-positive-energy selections; the sample is biased toward low-energy directions and is not an all-expert census. Crucially, a routed expert can receive zero current tokens yet retain momentum. One such NS direction has $\left. D \right. _ { \mathrm { F } } = 0 . 9 9 2 1 , \left. D \right. _ { 2 } = 0 . 9 7 2 4$ , and $\operatorname { s r } ( D ) = 1 . 0 4 0 9$ illustrating that low stable rank does not imply small operator amplitude. Energy also depends on amplitude, as in Eq. (8).

These measurements motivate studying post-NS energy but do not identify energy as a calibrated reliability estimate, select a universally optimal exponent, or show that Hyperball is the sole source of loss spikes. Appendix D specifies the ablations and interventions.

Table 4. Cohort M: EnergyCal added to plain Muon without Hyperball. All conditions share matrix LR 0.005, eligible matrix and fallback weight decay 0.1, and the same 10k-update token budget. Values are mean ± sample SD over $n = 3$ new initialization seeds; episodes and $B _ { 1 }$ are per-run quantities. The validation bank differs from Cohort H, so absolute CE and PPL values must not be compared directly with Table 2.
<table><tr><td>Recipe</td><td>Final CE</td><td>PPL</td><td>Episodes</td><td> $B _ { 1 }$ </td></tr><tr><td> $\mathrm { M u o n } + \mathrm { W D }$ </td><td> $3 . 1 8 9 7 8 6 \pm 0 . 0 0 2 3 8 1$ </td><td> $2 4 . 2 8 3 \pm 0 . 0 5 8$ </td><td> $1 1 . 6 7 \pm 4 . 0 4$ </td><td> $8 4 . 7 9 \pm 3 3 . 1 3$ </td></tr><tr><td> $\mathrm { M u o n } + \mathrm { E n e r g y C a l } \left( p = 0 . 5 \right)$ </td><td> $3 . 1 8 5 5 4 7 \pm 0 . 0 0 1 8 7 7$ </td><td> $2 4 . 1 8 1 \pm 0 . 0 4 5$ </td><td> $3 . 0 0 \pm 2 . 0 0$ </td><td> $4 . 2 5 \pm 4 . 0 6$ </td></tr><tr><td> $\mathrm { M u o n } + \mathrm { E n e r g y C a l } \left( p = 1 \right)$ </td><td> $3 . 1 8 5 9 7 5 \pm 0 . 0 0 1 1 7 3$ </td><td> $2 4 . 1 9 1 \pm 0 . 0 2 8$ </td><td> $1 . 6 7 \pm 1 . 1 5$ </td><td> $2 . 5 7 \pm 2 . 4 6$ </td></tr></table>

Table 5. Single-initialization diagnostic audit at common matrix LR 0.01, over 3k updates. Alternative coefficients are not matched for effective update magnitude. An endpoint occurring during a spike is retained.
<table><tr><td>Condition</td><td>3k CE</td><td>Ep.</td><td> $B _ { 1 }$ </td></tr><tr><td>MuonH</td><td>3.791280</td><td>4</td><td>430.51</td></tr><tr><td>Energy (p = 0.5)</td><td>3.774685</td><td>4</td><td>89.03</td></tr><tr><td>EnergyCal (p = 1)</td><td>3.769136</td><td>3</td><td>10.59</td></tr><tr><td>Stable-rank</td><td>3.761237</td><td>4</td><td>76.17</td></tr><tr><td>Amplitude only</td><td>9.318367</td><td>9</td><td>364.67</td></tr><tr><td>Near-zero gate</td><td>3.814775</td><td>6</td><td>531.36</td></tr></table>

Table 6. Next-step training CE after single-update interventions in two distinct saved-state events. Pre-intervention CE is 4.1928 (A) and 4.2143 (B). No tested branch eliminates the subsequent loss rise.
<table><tr><td>Single-update intervention</td><td>Event A</td><td>Event B</td></tr><tr><td>Canonical</td><td>6.3165</td><td>6.9069</td></tr><tr><td>Candidate: freeze</td><td>6.3172</td><td>6.9110</td></tr><tr><td>Control: freeze</td><td>6.3160</td><td>6.9085</td></tr><tr><td>All routed: EnergyCal</td><td>6.3024</td><td>6.9873</td></tr></table>

## 6.2. Single-update interventions do not rescue two events

We examine two baseline-scout loss excursions, events A and B. For each, candidate routed-down matrices were selected using low stable rank and traffic-weighted projected operator displacement, with same-bank controls. The heuristic was devised from exploratory captures but fixed before intervention outcomes. Seven branches per event modify one optimizer update and then return to the canonical training procedure (Appendix D). These events were selected for diagnostic analysis and should not be treated as representative samples of loss spikes.

All 14 branches complete. Initial state and recorded input checks agree within each event, although replays are not certified bitwise-identical. Exact weight freezing restores the original parameter bits while retaining momentum and ordinary writeback behavior; it does not remove an expert.

Candidate-only EnergyCal reduces actual BF16-stored operator displacement by 86.3% (A) and 76.3% (B), while exact freezing eliminates that candidate displacement. Nevertheless, next-step CE remains near 6.3 and 6.9, respectively. In the common short windows, candidate freezing reduces burden by 3.09% and 0.76%, compared with 1.61% and 1.40% for control freezing. Applying EnergyCal to all routed matrices for a single update reduces short-window burden by 2.12% in event A but increases it by 3.41% in event B. This shortwindow burden is distinct from the full-schedule $B _ { 1 }$ defined in Eq. (12).

![](images/8f85352a8537f85094dbef256a2498f3749061d1ef1f22ad22e7bcef9ea66930.jpg)  
Fig. 3. Selected diagnostic measurements of direction stable rank and BF16-stored operator displacement. Stored movement is measured directly; the selected observations do not establish a general bound or a causal link to spikes.

A separate fixed-input 512-token probe shows no large reduction in output KL; probe CE decreases slightly in every branch. The probe differs from the next high-loss training batch and omits the subsequent router-bias update. These interventions therefore do not explain continuous calibration’s observed full-schedule benefit through immediate removal of the two examined spikes, but they do not establish that smaller or history-dependent effects are absent.

## 7. DISCUSSION AND LIMITATIONS

Empirical contribution and practical trade-offs. The main finding is a substantial reduction in thresholded loss-spike burden with both MuonH and plain Muon, accompanied by small improvements in final validation quality. In Cohort H, the frozen EnergyCal–MuonH recipe reduces mean burden by 99.51% relative to a development-selected, lower-rate MuonH recipe, with improved endpoint quality and 1.62% lower mean throughput. Because the learning rates differ, this comparison does not isolate the effect of calibration alone. The plain-Muon comparator is faster and achieves better intermediate quality; an illustrative equal-time estimate can also favor plain Muon. In Cohort M, EnergyCal reduces mean burden by 95.98% on average across two tested exponents at matched nominal learn ing rate and weight decay, showing that stabilization is also observed without Hyperball. However, different validation banks preclude a direct quality ranking of the calibrated opti mizer families, and no production-style throughput benchmark is available for Cohort M. These results support empirical stabilization in the evaluated setting, not general optimizer superiority.

The unresolved uniform-scaling alternative. EnergyCal attenuates routed-expert updates, but the observed improvements need not arise specifically from assigning different coefficients according to direction energy. Cohort M lacks a routed-only uniform-scaling or update-budget-matched control, so it cannot distinguish energy-dependent attenuation from the benefit of reducing routed-update magnitudes more generally. A fixed coefficient near 0.96, leaving weight decay and non-routed updates unchanged, would provide an informative comparison, although it would not exactly match the update-norm budget. Appendix C.3 distinguishes these controls. Furthermore, we do not establish relative superiority over other expert-wise scaling methods, including Compass and Excitation, because native-method head-to-head comparisons were not performed.

Geometry and causal interpretation. The geometric analysis provides motivation for attenuation, not a causal explanation of loss spikes. The inverse-stable-rank identity describes the operator norm of MuonH’s trial update, not the final projected or stored parameter displacement, and does not directly apply to plain Muon. Moreover, the energy statistic depends on both spectral concentration and direction amplitude, rather than stable rank alone. In two examined events, single-update interventions substantially reduce the nominated parameter movements but fail to prevent the subsequent loss spikes. Thus, the full-schedule benefits cannot be explained simply as immediate removal of these two events through one-step attenuation. Effects accumulated over training, including changes in expert parameters or routing dynamics, remain possible explanations but are not established here. These negative interventions limit the mechanistic interpretation without negating the observed benefits of continuous calibration.

Scope, measurement, and generalizability. Both cohorts use one custom 4B-parameter, 128-expert, top-2 MoE model and 1.31B training tokens per run, with three initialization seeds and a shared data order. The results therefore do not establish generalization across architectures, scales, data distributions, or training budgets. Cohort H selects baselines within bounded learning-rate grids whose best-tested values lie at the lower boundaries; Cohort M fixes the plain-Muon recipe rather than independently retuning every variant. Reported standard deviations describe run-to-run variation, not confidence intervals. The adaptive spike detector can miss persistent loss elevation, while periodic validation may miss transient quality changes. Selected spectral measurements are not an all-layer census, and the local probes omit the subsequent router-bias-controller update.

## 8. CONCLUSION

EnergyCal applies a simple, one-sided attenuation to routedexpert updates using finite-step Newton–Schulz direction energy. In a 4B-parameter sparse MoE model, it substantially reduces thresholded loss-spike burden with both MuonH and plain Muon, showing that stabilization is also observed without Hyperball in the evaluated setting. The frozen Energy-Cal–MuonH recipe reduces mean burden by 99.51% and final geometric-mean perplexity by 4.06% relative to a developmentselected, lower-rate MuonH recipe, with 1.62% lower mean throughput. In a separate plain-Muon study at matched nominal learning rate and weight decay, reductions in mean burden average 95.98% across the two tested exponents, with small endpoint gains. These results support empirical stabilization in the evaluated MoE setting rather than general optimizer superiority. However, single-update interventions did not rescue two examined spikes, and the absence of a routed-only uniform-scaling control leaves the specific contribution of energy-dependent attenuation unresolved. Further controlled comparisons are needed to distinguish this contribution from reduced update magnitude.

## 9. REFERENCES

[1] Masato Fujitake, “MESH: Memory-Efficient Sinkhorn Optimization for Mixture-of-Experts Training,” arXiv preprint arXiv:2608.04407, 2026.

[2] Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein, “Muon: An Optimizer for Hidden Layers in Neural Networks,” https://kell erjordan.github.io/posts/muon/, 2024, December 8.

[3] Ishaan Shah et al., “Practical Efficiency of Muon for Pretraining,” arXiv preprint arXiv:2505.02222, 2025.

[4] Jingyuan Liu, Jianlin Su, Xingcheng Yao, et al., “Muon is Scalable for LLM Training,” arXiv preprint arXiv:2502.16982, 2025.

[5] Kaiyue Wen, Xingyu Dang, Kaifeng Lyu, Tengyu Ma, and Percy Liang, “Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization,” arXiv preprint arXiv:2606.16899, 2026.

[6] Sagi Shaier, “Excitation: Momentum For Experts,” arXiv preprint arXiv:2602.21798, 2026.

[7] Omatharv Bharat Vaidya, Ashwin Vinod, Pedram Akbarian, Aditya Sai Ellendula, Connor T. Jerzak, and Nhat Ho, “ExpertMuon-Compass: Alignment-Guided Step Sizes for Mixture-of-Experts Training,” arXiv preprint arXiv:2610.04140, 2026.

[8] Zichong Li, Liming Liu, Chen Liang, Weizhu Chen, and Tuo Zhao, “NorMuon: Making Muon More Efficient and Scalable,” in Proceedings ofthe 43rd International Conference on Machine Learning. 2026, vol. 306 of Proceedings of Machine Learning Research, pp. 68514–68533, PMLR.

[9] Peng Cheng, Jiucheng Zang, Qingnan Li, Liheng Ma, Yufei Cui, Yingxue Zhang, Boxing Chen, Ming Jian, and Wen Tong, “TrasMuon: Trust-Region Adaptive Scaling for Orthogonalized Momentum Optimizers,” arXiv preprint arXiv:2602.13498, 2026.

[10] Chongyu Fan, Gaowen Liu, Mingyi Hong, Ramana Rao Kompella, and Sijia Liu, “Rethinking Muon Beyond Pretraining: Spectral Failures and High-Pass Remedies for VLA and RLVR,” arXiv preprint arXiv:2605.19282, 2026.

[11] Dmitrii Feoktistov, Timofey Belinsky, Andrey Veprikov, Amir Zainullin, and Aleksandr Beznosikov, “Softsign: Smooth Sign in Your Optimizer For Better Parameter Heterogeneity Handling,” in Proceedings of the 43rd International Conference on Machine Learning. 2026, vol. 306 of Proceedings of Machine Learning Research, pp. 30720–30746, PMLR.

[12] Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus, “ST-MoE: Designing stable and transferable sparse expert models,” arXiv preprint arXiv:2202.08906, 2022.

[13] Damai Dai, Li Dong, Shuming Ma, Bo Zheng, Zhifang Sui, Baobao Chang, and Furu Wei, “StableMoE: Stable routing strategy for mixture of experts,” in Proceedings of the 60th An nual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2022, pp. 7085–7095, Association for Computational Linguistics.

[14] Tianjin Huang, Ziquan Zhu, Gaojie Jin, Lu Liu, Zhangyang Wang, and Shiwei Liu, “SPAM: Spike-aware adam with momentum reset for stable LLM training,” in International Conference on Learning Representations, 2025.

[15] Gyu Yeol Kim and Min-hwan Oh, “Convergence of Muon with Newton–Schulz,” in International Conference on Learning Representations, 2026.

[16] Gagik Magakyan, Pablo Parrilo, and Asuman Ozdaglar, “Spectral scaling laws of Muon,” arXiv preprint arXiv:2606.04058, 2026.

[17] Da Chang, Qiankun Shi, Lvgang Zhang, Yu Li, Ruijie Zhang, Yao Lu, Yongxiang Liu, and Ganzhao Yuan, “MuonEq: Balancing before orthogonalization with lightweight equilibration,” arXiv preprint arXiv:2603.28254, 2026.

[18] Kaiyue Wen, David Hall, Tengyu Ma, and Percy Liang, “Fantastic Pretraining Optimizers and Where to Find Them,” in International Conference on Learning Representations, 2026.

[19] Yihao Xiao, Jialong Sun, Zitian Gao, Zeming Wei, Chutian Wang, Ran Tao, Jiaye Teng, and Bryan Dai, “Hyperball May Not Be a Free Lunch,” arXiv preprint arXiv:2607.22444, 2026.

[20] Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang, “DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models,” arXiv preprint arXiv:2401.06066, 2024.

[21] Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, and Damai Dai, “Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts,” arXiv preprint arXiv:2408.15664, 2024.

[22] DeepSeek-AI, “DeepSeek-V3 Technical Report,” arXiv preprint arXiv:2412.19437, 2024.

[23] Guilherme Penedo, Hynek Kydl´ıcek, Loubna Ben Allal, An-ˇ ton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro von Werra, and Thomas Wolf, “The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale,” arXiv preprint arXiv:2406.17557, 2024.

## A. IMPLEMENTATION AND EVALUATION DETAILS

This appendix records the numerical and parameter-partition details needed to interpret EnergyCal. The model, training budget, and comparison design are described in Section 4.

## A.1. Finite-step Newton–Schulz

The matrix input to NS is cast to BF16 and, if needed, transposed so that the smaller dimension defines the Gram matrix. For the working matrix $X ,$ normalization and five quintic iterations are

$$
X  X / \operatorname* { m a x } ( \| X \| _ { \mathrm { F } } , 1 0 ^ { - 7 } ) ,\tag{14}
$$

$$
\boldsymbol { Q } = \boldsymbol { X } \boldsymbol { X } ^ { \intercal } ,\tag{15}
$$

$$
X  3 . 4 4 4 5 X + ( - 4 . 7 7 5 0 Q + 2 . 0 3 1 5 Q ^ { 2 } ) X .\tag{16}
$$

The norm is accumulated in FP32, and the output is returned in FP32 with its original orientation. Finite-step BF16 NS does not guarantee exact unit singular values, so post-NS direction energy remains informative. MuonH separately uses a denominator guard $\epsilon _ { H } = 1 0 ^ { - 1 2 }$ for normalization and projection.

## A.2. Logical matrices and writeback

Fused gate/up storage is split into separate logical matrices for optimization; each routed gate, up, or down matrix has $d \ = \ \operatorname* { m i n } ( m , n ) \ = \ 2 5 6$ . EnergyCal acts only on routed matrices. Attention and shared-expert matrices retain their matrix-optimizer updates, and embeddings, router weights, and vectors use an Adam-style fallback.

MuonH uses a fixed initial Frobenius radius and the trial up date and projection of Eqs. (2)–(3). Plain Muon has no sphere projection and uses Eq. (6). Both use a common dithered BF16 writeback policy. Thus $q = 0$ suppresses the NS trial but need not produce an exact bitwise freeze: projection and writeback may change MuonH weights, while plain Muon can still apply weight decay.

## A.3. Geometry and coefficient interpretation

For nonzero D, $\operatorname { s r } ( D ) = \left\| D \right\| _ { \mathrm { F } } ^ { 2 } / \left\| D \right\| _ { 2 } ^ { 2 }$ . With the MuonH denominator guard, the trial operator norm is

$$
\left\| \Delta ^ { \mathrm { t r i a l } } \right\| _ { 2 } = \frac { \eta R q \ \left\| D \right\| _ { 2 } } { \operatorname* { m a x } ( \left\| D \right\| _ { \mathrm { F } } , \epsilon _ { H } ) } .\tag{17}
$$

Equation (7) follows when the guard is inactive. Stable rank is undefined at the zero matrix.

For comparison, the unclipped, unguarded exact-stablerank coefficient $q _ { r } = \sqrt { \mathrm { s r } ( D ) / ( d \rho _ { \mathrm { r e f } } ) }$ gives

$$
\big \| \Delta ^ { \mathrm { t r i a l } } \big \| _ { 2 } = \frac { \eta R } { \sqrt { d \rho _ { \mathrm { r e f } } } } .\tag{18}
$$

The tested stable-rank alternative estimates operator norm rather than computing an exact SVD during training. In contrast, EnergyCal with $\textit { p } = \ 1 / 2$ yields $\left. \Delta ^ { \mathrm { t r i a l } } \right. _ { 2 } ~ =$ $\eta R \left. D \right. _ { 2 } / \sqrt { d \rho _ { \mathrm { r e f } } }$ in the same regime. These identities characterize the trial, not the projected/stored displacement or a bound on the network’s loss change.

## A.4. Validation boundaries

A deterministic partition of FineWeb sample-10BT reserves 0.5% of rows for validation. Evaluation uses fixed banks within each comparison, but development, Cohort H, and Cohort M use different banks. Different banks and implementation revisions prevent direct cross-cohort ranking of absolute CE/PPL. The validation observations are not certified as previously unused external test examples, and content-level near-duplicate contamination was not independently audited.

## B. DEVELOPMENT SELECTION AND CONFIRMATION DETAIL

## B.1. Development screening

The baseline learning-rate candidates are screened over two initialization runs, each observed after 3,000 updates. The selected baseline rate minimizes mean screening-endpoint validation CE. Both baseline choices are at the lower boundaries of their tested grids, so the comparison is best-tested within a bounded search. EnergyCal–MuonH is fixed at matrix LR $0 . 0 1 , p = 1 , \rho _ { \mathrm { r e f } } = 0 . 8 .$ , and $q _ { \mathrm { m i n } } = 0$ before confirmation.

The selected lower-rate baselines have better development endpoints than EnergyCal. The independent full-schedule confirmation therefore tests a previously fixed recipe, not an extrapolation of the shorter screen. Development and confirmation validation banks are not combined.

## B.2. Confirmation effects

Percentage burden reductions use $1 - \overline { { B } } _ { E } / \overline { { B } } _ { C }$ , with $\overline { B }$ denoting the condition-level mean burden. This gives 99.51% versus MuonH and 85.01% versus Muon + WD; neither is the average of run-wise percentage reductions. All confirmation runs contribute, including the zero-burden EnergyCal run and the unfavorable burden comparison against Muon + WD.

## B.3. Intermediate-quality summaries

For evaluation steps $s _ { j } \in \{ 5 0 0 , 1 0 0 0 , \ldots , 1 0 0 0 0 \}$ , normalized trapezoidal CE is computed per run:

$$
\overline { { L } } _ { \mathrm { o b s } } = \frac { 1 } { s _ { K } - s _ { 1 } } \sum _ { j = 1 } ^ { K - 1 } ( s _ { j + 1 } - s _ { j } ) \frac { L _ { s _ { j } } + L _ { s _ { j + 1 } } } { 2 } , \qquad K = 2 0 .\tag{19}
$$

Table 7. Completed development screen: mean ± sample SD over $n = 2$ initialization runs after 3,000 updates. All endpoints, including unfavorable ones, are retained. Episodes and burden are per-run quantities.
<table><tr><td>Recipe</td><td>LR</td><td>Final CE</td><td>Episodes</td><td> $B _ { 1 }$ </td></tr><tr><td>MuonH</td><td>0.005000</td><td> $3 . 7 0 3 7 4 0 \pm 0 . 0 1 3 6 5 8$ </td><td> $8 . 0 0 \pm 0 . 0 0$ </td><td> $3 3 9 . 4 0 \pm 3 7 9 . 0 7$ </td></tr><tr><td>MuonH</td><td>0.007071</td><td> $3 . 7 3 3 1 1 9 \pm 0 . 0 1 3 7 9 5$ </td><td> $7 . 0 0 \pm 1 . 4 1$ </td><td> $5 6 2 . 7 6 \pm 0 . 8 8$ </td></tr><tr><td>MuonH</td><td>0.010000</td><td> $3 . 9 4 9 3 2 4 \pm 0 . 2 2 6 6 8 9$ </td><td> $1 1 . 5 0 \pm 4 . 9 5$ </td><td> $5 9 2 . 8 7 \pm 9 9 . 2 1$ </td></tr><tr><td>EnergyCal (p = 1)</td><td>0.010000</td><td> $3 . 7 6 9 4 2 5 \pm 0 . 0 0 0 6 4 7$ </td><td> $2 . 5 0 \pm 0 . 7 1$ </td><td> $8 . 7 6 \pm 0 . 2 4$ </td></tr><tr><td> $\mathrm { M u o n + W D }$ </td><td>0.005000</td><td> $3 . 7 0 3 0 6 1 \pm 0 . 0 0 4 3 6 6$ </td><td> $1 . 5 0 \pm 0 . 7 1$ </td><td> $1 . 8 4 \pm 1 . 5 3$ </td></tr><tr><td> $\mathrm { M u o n + W D }$ </td><td>0.010000</td><td> $3 . 7 7 2 3 9 4 \pm 0 . 0 1 1 7 0 7$ </td><td> $5 . 5 0 \pm 0 . 7 1$ </td><td> $7 5 . 8 7 \pm 0 . 1 4$ </td></tr><tr><td> $\mathrm { M u o n + W D }$ </td><td>0.020000</td><td> $3 . 9 0 9 8 2 2 \pm 0 . 0 1 7 1 7 7$ </td><td> $5 . 0 0 \pm 1 . 4 1$ </td><td> $6 9 . 5 1 \pm 1 2 . 9 1$ </td></tr></table>

Table 8. Paired EnergyCal-minus-baseline differences over $n = 3$ confirmation initializations (mean ± sample SD). Negative values favor EnergyCal. Pairing precedes aggregation.
<table><tr><td>EnergyCal minus</td><td>CE difference</td><td> $B _ { 1 } ~ \mathrm { d i f f e r e n c e }$ </td><td>Episode difference</td></tr><tr><td>MuonH</td><td> $- 0 . 0 4 1 4 9 1 \pm 0 . 0 2 6 0 4 3$ </td><td> $- 2 0 3 5 . 9 1 \pm 1 2 9 7 . 3 7$ </td><td> $- 3 4 . 0 0 \pm 1 . 7 3$ </td></tr><tr><td> $\mathrm { M u o n } + \mathrm { W D }$ </td><td> $- 0 . 0 0 6 5 1 0 \pm 0 . 0 0 5 7 0 9$ </td><td> $- 5 7 . 3 1 \pm 5 5 . 5 7$ </td><td> $- 5 . 0 0 \pm 6 . 2 4$ </td></tr></table>

![](images/0caa7d3d64c5e530864e95566d06b49ed66e8614953a5e7cef590dae14d8de2c.jpg)  
Fig. 4. Late-schedule validation CE (mean ± sample SD, $n = 3 )$ . EnergyCal’s small endpoint advantage over plain Muon appears late; the full trajectory is shown in Fig. 2.

Mean and sample SD are then computed across runs. A separate post hoc late-window check averages the last three evaluations within each run. Neither summary replaces the predefined final endpoint. The observed-range summary favors plain Muon over EnergyCal, whereas the late-window summary favors EnergyCal.

The adaptive spike detector may miss persistent high-loss periods, while narrow excursions can produce large maxima but modest burden. Final validation CE, maximum training loss, episodes, and thresholded burden therefore remain complementary outcomes. All final runs complete.

## C. PLAIN-MUON TRANSFER

Cohort M tests EnergyCal without Hyperball. Plain Muon and its $p \in \{ 0 . 5 , 1 \}$ EnergyCal variants are rerun over three new initializations for the complete 10,000-update schedule. The same architecture, data order, and token budget are used as in Cohort H, but a separate implementation revision and validation bank preclude direct cross-cohort ranking.

All Cohort M conditions share matrix LR 0.005, eligible matrix/fallback weight decay 0.1, and fallback peak LR 0.001. Only the routed coefficient in Eq. (6) changes; momentum, decay, non-routed updates, and the Adam fallback do not. EnergyCal uses $\rho _ { \mathrm { r e f } } ~ = ~ 0 . 8 , q _ { \mathrm { m i n } } = 0$ , and the two stated exponents. Every run completes.

## C.1. Quality and stability

With $\overline { { B } } _ { p }$ denoting the condition-level three-run mean burden for exponent p and $\overline { { B } } _ { 0 }$ the plain-Muon mean, the reduction is $r _ { p } = 1 0 0 \big ( 1 - \overline { { B } } _ { p } / \overline { { B } } _ { 0 } \big )$ . The two reductions are 94.99% and 96.97%; their equal-weight average is 95.98%. This is not a mean of per-initialization percentages or an estimate using six independent replicates.

Both exponents reduce thresholded burden in all three paired runs. Endpoint CE improves in two of three runs for $p = 0 . 5$ and all three for $p = 1 ;$ neither exponent is uniformly optimal. The paired endpoint CE difference between the calibrated variants is only $- 0 . 0 0 0 4 \pm 0 . 0 0 1 6$ nats/token.

The mean observed-range advantage is partly influenced by a plain-Muon spike at one evaluation. Calibrated variants do not dominate at every point; the late-window check also remains post hoc. Final CE at 10,000 updates is the primary endpoint.

Table 9. Cohort H curve summaries: mean ± sample SD of run-level results. The late-window comparison is post hoc.
<table><tr><td>Recipe</td><td>Observed-range CE</td><td>Late-window CE</td></tr><tr><td>MuonH</td><td> $3 . 9 9 2 2 8 9 \pm 0 . 3 3 9 4 0 0$ </td><td> $3 . 2 7 8 5 0 3 \pm 0 . 0 0 7 9 6 1$ </td></tr><tr><td>EnergyCal (p = 1)</td><td> $3 . 6 2 7 0 0 2 \pm 0 . 0 0 1 5 7 2$ </td><td> $3 . 2 5 1 4 3 5 \pm 0 . 0 0 0 6 3 9$ </td></tr><tr><td> $\mathrm { M u o n } + \mathrm { W D }$ </td><td> $3 . 6 1 1 1 3 7 \pm 0 . 0 0 9 4 3 3$ </td><td> $3 . 2 5 6 0 5 9 \pm 0 . 0 0 5 0 4 5$ </td></tr></table>

![](images/666c96c529fdddda5d54ba4d73daaf9ce314833b204d08c2a6c70981b57a4327.jpg)  
Fig. 5. Cohort M validation CE: mean ± sample SD over three runs on its own fixed evaluation bank. All observations, including the plain-Muon loss excursion at step 4000, are retained. Bands are not confidence intervals.

## C.2. Coefficient measurements

Recorded coefficients are summarized within each run before aggregation across three initializations. Their means are $0 . 9 6 7 9 \pm 0 . 0 0 2 1$ for $p = 0 . 5$ and $0 . 9 5 9 8 \pm 0 . 0 0 1 7$ for $p = 1$ (mean ± sample SD). Some coefficients equal zero. Thus positive scaling preserves a nonzero NS direction, whereas $q = 0$ suppresses that direction without necessarily freezing the stored parameter. These summaries are not activationweighted effective learning rates, and no diagnostic-off production-style throughput benchmark was performed for this cohort.

## C.3. Unperformed routed-only scalar controls

The matched nominal LR/decay comparison measures the effect of adding the evaluated attenuation rule but does not identify whether expert-wise energy-dependent assignment is better than uniform routed-update reduction. A simple untested control would fix $q _ { e } ~ = ~ \alpha$ for all routed matrices $( \mathrm { e . g . , } \alpha = 0 . 9 6 )$ , leaving other updates and decay unchanged.

Equal average coefficients do not imply equal update-norm budgets. At a common state, let $U _ { e , t } = - \eta _ { t } s _ { e } D _ { e , t }$ denote the pre-decay, pre-writeback plain-Muon direction step for routed matrix e. When the denominator is positive, a norm-budgetmatched uniform coefficient would be

$$
\alpha _ { t } = \left( \frac { \sum _ { e \in \mathcal { R } } q _ { e , t } ^ { 2 } \left. U _ { e , t } \right. _ { \mathrm { F } } ^ { 2 } } { \sum _ { e \in \mathcal { R } } \left. U _ { e , t } \right. _ { \mathrm { F } } ^ { 2 } } \right) ^ { 1 / 2 } .\tag{20}
$$

This matches only the summed squared Frobenius norms of the direction steps at that state. It does not match stored parameter changes, decay, or future trajectories. Neither fixed-scalar nor norm-budget-matched controls were evaluated, so a specific causal advantage of assigning coefficients by energy remains unresolved.

## D. SPECTRAL AUDIT AND EVENT PROTOCOL

## D.1. Alternative coefficients and spectral measurements

A separate single-initialization audit compares six update rules at common matrix LR 0.01 over 3,000 updates, with additional scouts used only for diagnostic observations. The stable-rank, amplitude-only, and near-zero-gate controls use

$$
q _ { \mathrm { r a n k } } = \operatorname* { m i n } \left\{ 1 , \sqrt { \frac { \widehat { \mathrm { s r } } ( D ) } { 0 . 8 d } } \right\} ,\tag{21}
$$

$$
q _ { \mathrm { a m p } } = \operatorname* { m i n } \{ 1 , \widehat { \| D \| _ { 2 } } \} ,\tag{22}
$$

$$
q _ { \mathrm { z e r o } } = { \bf 1 } \{ \| D \| _ { \mathrm { F } } > 1 0 ^ { - 7 } \} .\tag{23}
$$

Table 10. Additional Cohort M results: mean ± sample SD over $n = 3$ initialization runs. Observed-range CE integrates the 500–10,000 evaluations; late-window CE is a post hoc check over the final three evaluations.
<table><tr><td>Recipe</td><td>PPL</td><td> $L _ { \mathrm { m a x } }$ </td><td>Observed-range CE</td><td>Late-window CE</td></tr><tr><td> $\mathrm { M u o n } + \mathrm { W D }$ </td><td> $2 4 . 2 8 3 3 \pm 0 . 0 5 7 8$ </td><td> $8 . 5 4 \pm 0 . 8 2$ </td><td> $3 . 5 7 8 7 0 1 \pm 0 . 0 4 2 8 0 9$ </td><td> $3 . 2 0 3 7 5 9 \pm 0 . 0 0 2 1 2 2$ </td></tr><tr><td> $\mathrm { M u o n } + \mathrm { E n e r g y C a l } \left( p = 0 . 5 \right)$ </td><td> $2 4 . 1 8 0 5 \pm 0 . 0 4 5 4$ </td><td> $5 . 6 1 \pm 0 . 8 7$ </td><td> $3 . 5 4 8 9 5 7 \pm 0 . 0 0 2 6 6 9$ </td><td> $3 . 2 0 0 2 9 9 \pm 0 . 0 0 1 4 0 3$ </td></tr><tr><td> $\mathrm { M u o n } + \mathrm { E n e r g y C a l } \left( p = 1 \right)$ </td><td> $2 4 . 1 9 0 9 \pm 0 . 0 2 8 4$ </td><td> $5 . 4 9 \pm 0 . 7 5$ </td><td> $3 . 5 5 0 8 2 1 \pm 0 . 0 0 0 7 8 8$ </td><td> $3 . 2 0 0 7 7 8 \pm 0 . 0 0 1 2 1 7$ </td></tr></table>

Table 11. Cohort M paired effects: mean ± sample SD over three within-initialization differences (EnergyCal minus plain Muon). Negative values favor calibration.
<table><tr><td>Coefficient</td><td> $\Delta \mathrm { C E }$ </td></tr><tr><td> $p = 0 . 5$ </td><td> $- 0 . 0 0 4 2 3 9 \pm 0 . 0 0 4 1 9 4$ </td></tr><tr><td> $p = 1$   $- 0 . 0 0 3 8 1 1 \pm 0 . 0 0 2 8 4 9$ </td><td> $- 8 0 . 5 4 \pm 3 6 . 4 4$  1  $- 8 2 . 2 3 \pm 3 0 . 6 7$  1</td></tr></table>

The spectral quantities used during optimization are estimated by power iteration rather than exact SVD. These alternatives were not independently tuned to equal update budgets.

For diagnostics, NS spectra and projected/BF16-stored displacements are measured independently using CPU float64 SVD on selected routed-down matrices. Selection deliberately includes low-energy directions, so the sampled distribution is not an unbiased census. Norms reconstructed from the trial formula do not independently verify Eq. (7). A zero directional coefficient can still yield stored changes after projection and writeback, motivating exact-weight-freeze interventions.

## D.2. Single-update interventions

Two saved-state spike events (A and B) are examined separately. Candidates are selected from routed-down matrices using low stable rank and traffic-weighted projected displacement, with comparable same-bank control matrices. This heuristic was informed by exploratory captures but fixed before treatment outcomes. Each event uses seven single-update branches: canonical update, candidate EnergyCal, candidate zero trial, candidate exact weight freeze, control freeze, all routed EnergyCal, and all-routed EnergyCal with the candidate restored to canonical. Training then resumes normally.

Exact freezing restores the selected weight bits after the ordinary update and writeback while retaining momentum evolution and RNG consumption. It differs from zero trial, which suppresses only the direction proposal. Branches start from matched saved states and input sequences, but independent recomputation does not guarantee identical numerical trajectories.

For an event’s common short window W and fixed preintervention reference loss $b ,$ the local burden and percent improvement relative to its canonical continuation are

Table 12. All intervention branches in events A and B $( n =$ 1 per branch and event). $r _ { A } , r _ { B }$ are short-window burden improvements, not reductions in the full-schedule $B _ { 1 }$
<table><tr><td>Intervention</td><td>CE A</td><td>CE B  $r _ { A } \ : ( \% )$ </td><td>rB (%)</td></tr><tr><td>Canonical</td><td>6.3165</td><td>6.9069 0.00</td><td>0.00</td></tr><tr><td>Candidate ECal</td><td>6.3163 6.9068</td><td>+0.93</td><td>+0.22</td></tr><tr><td>Candidate zero</td><td>6.31606.9109</td><td>-1.25</td><td>+0.32</td></tr><tr><td>Candidate freeze</td><td>6.3172</td><td>6.9110 +3.09</td><td>+0.76</td></tr><tr><td>Control freeze</td><td>6.3160</td><td>6.9085 +1.61</td><td>+1.40</td></tr><tr><td>All routed ECal</td><td>6.3024</td><td>6.9873 +2.12</td><td>-3.41</td></tr><tr><td>All except cand.</td><td>6.3020</td><td>6.9840 +1.48</td><td>-3.15</td></tr></table>

$$
A _ { \mathrm { w i n } } = \sum _ { s \in \mathcal { W } } \operatorname* { m a x } ( 0 , L _ { s } - b ) ,\tag{24}
$$

$$
r _ { \mathrm { w i n } } = 1 0 0 \left( 1 - \frac { A _ { \mathrm { a r m } } } { A _ { \mathrm { c a n o n i c a l } } } \right) .\tag{25}
$$

Unlike $B _ { 1 } , A _ { \mathrm { w i n } }$ includes all positive deviations above the fixed reference in the short window, without threshold-flag filtering. It is used only for intervention comparisons. For events A and B, the evaluation windows are steps 1068–1100 and 926–957, with fixed reference losses $b = 4 . 1 3 3 4 6 6$ and $b = 4 . 2 4 0 4 2 3 .$ , respectively.

Candidate-only EnergyCal reduces stored operator displacement by 86.3% in event A and 76.3% in event B, while exact freezing eliminates that displacement. The next-step loss rise persists. Candidate freeze reduces short-window burden by 3.09% and 0.76%, versus 1.61% and 1.40% for control freeze; all-routed EnergyCal reduces short-window burden by 2.12% in event A but increases it by 3.41% in event B. These small, inconsistent effects do not establish substantial immediate rescue or a zero causal effect.

## D.3. Fixed-input probes and causal limitations

A fixed 512-token probe compares CE and output KL immediately before and after each treated update. Mean probe CE decreases slightly in every branch, with no large KL reduction from the tested corrections. The probe is different from the next high-loss training batch and omits the subsequent routerbias-controller update. Consequently, the local observations do not identify the combination of history, routing, and inputs responsible for the two events. They also do not negate the full-schedule improvement under continuous EnergyCal.