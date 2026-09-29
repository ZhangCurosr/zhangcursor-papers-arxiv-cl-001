# Commutator Memory: Sparse, Path-Local Reading and Steering in Language Models

John Sweeney Sideplane AI john.sweeney@sideplane.ai

## Abstract

Gradient updates on different data generally do not commute: training a language model on two data sources in opposite orders gives different weights, even with the same data and total exposure. Loss or benchmark deltas show that the models differ, not where. We ask whether this path dependence leaves a parametric training-history memory: a weight component that flips sign when the two sources are swapped, is localized in output space, changes the held-out loss gap between the two orders under targeted interventions, and reveals which trained model came from which order. For one small SGD step of size η on each of sources A and B, the weight difference $\theta _ { A B } - \theta _ { B A }$ is, to leading order, $\eta ^ { 2 } b _ { A B }$ where $b _ { A B } = H _ { B } g _ { A } - H _ { A } g _ { B }$ is the Lie bracket of the two gradient fields at the base model. We define commutator memory by projecting the bracket through the logits into one score per vocabulary token; the scores sum to the bracket’s prediction of the gap. The scores are localized: on three models, the same readout of the measured $\theta _ { A B } - \theta _ { B A } ,$ , or of a bracket from disjoint batches, shares 82–99% of the original top-20 tokens, versus 35–49% for norm-matched random directions. They are causally actionable: in Qwen-3-4B SFT, downweighting the ten tokens with the largest predicted share of the gap closes a median 32% of the measured gap, while frequency-matched tokens with near-zero scores have almost no effect. The weights themselves carry the component: projecting the difference between the two trained models onto $b _ { A B }$ identifies which came from which order in 92% of cases across four LLMs (chance 50%). Controlled tests also cover matched-batch DPO, a frozen-rollout GRPO-style objective, and an AdamW endpoint check. The memory is defined per source pair, not per example, and its projection on $b _ { A B }$ decays with further training.

## 1 Introduction

Modern LLM post-training composes multiple data sources sequentially: domain pre-training, SFT, preference tuning, safety updates. Prior fine-tuning and training-order studies show that data order and task order can change final performance, and that recently learned information can leave a detectable representation-space trace [Dodge et al., 2020, Chen et al., 2024, Krasheninnikov et al., 2026]. In current practice, however, order effects are usually observed only through aggregate benchmark deltas, leaving the practitioner with no view of where the interaction between training stages lands in output space. The natural worry is that the order-dependent residue, what the two endpoints disagree on, is diffuse optimizer noise: spread thinly across capabilities, interpretable only in retrospect, and untargetable. Figure 1 previews our answer: it is not.

This paper shows the residue is not diffuse at the readout level. The order-asymmetric component of source interaction has a concentrated, interpretable, vocabulary-level signature. At the calibrated step sizes studied here (BCH-local: small enough that the second-order expansion of $\theta _ { A B } - \theta _ { B A }$ holds), the few tokens that carry most of the effect can be estimated from the base model, specified source data, and evaluation slice before both orderings are run. We use “memory” operationally, as parametric training-history memory. With training content and total exposure held fixed, composing two updates leaves an order-specific component in the weights; exchanging their order flips its sign. To count as memory, this component should be localized (concentrated on identifiable outputs), causally actionable (intervening on the predicted support changes the order gap), and assignable from paired endpoints (the endpoint weights retain an antisymmetric trace of the order that produced them). Access is pair-conditioned: the two candidate sources must be supplied, and we do not claim item-addressable recall, personalization, or long-horizon agent memory. Table 1 maps the central objects in plain language and marks which are inherited from prior work and which are new here.

![](images/1c6b60ba77b51e00987518c74c263e976d7a6d349c6674385fc67eacfed4e7b8.jpg)  
Figure 1: Commutator memory pipeline. One bracket produces an order forecast, token map, intervention target, and endpoint trace.

Table 1: Plain-language map of the central objects (A, B: the two training data sources; E: the held-out evaluation slice). Inherited from prior work: the endpoint-difference identity [Sweeney, 2026a, Lemma 2.1], the scalar criterion at $\theta _ { 0 }$ [Rukhovich et al., 2025] and at $\theta _ { \mathrm { r e f } }$ [Sweeney, 2026a], the bracket-only step [Sweeney, 2026a, App. E.8], and the fixed-clock AdamW regime, in which Adam’s buffers advance per step regardless of η [Sweeney, 2026b, Thm. 1]. This paper develops the token-level readout, order-gap interventions, paired-endpoint assignment, bracket-only guarantee, and AdamW closure test.
<table><tr><td>Object</td><td>Role</td><td>Origin</td></tr><tr><td> $b _ { A B } = H _ { B } g _ { A } - H _ { A } g _ { B }$ </td><td>effect of swapping A and B:  $\theta _ { A B } - \theta _ { B A } \stackrel { \sim } { = } \eta ^ { 2 } b _ { A B } + O ( \eta ^ { 3 } )$ </td><td>inherited</td></tr><tr><td> $\sigma = \langle g _ { E } ( \theta _ { \mathrm { r e f } } ) , b _ { A B } \rangle$ </td><td>one scalar predicting which order has lower held-out loss</td><td>inherited</td></tr><tr><td>τk (commutator memory)</td><td>token-level decomposition of  $\begin{array} { r } { \sigma ; \sum _ { k } \tau _ { k } = \eta ^ { 2 } \sigma } \end{array}$ </td><td>new</td></tr><tr><td>order-gap closure</td><td>fraction of the AB/BA NLL gap removed by an intervention</td><td>new</td></tr><tr><td> $\Delta s = \langle \theta _ { A B } - \theta _ { B A } , b _ { A B } \rangle$ </td><td>which order produced each of two endpoints</td><td>new</td></tr><tr><td>θBO TAdam</td><td>bracket-only step; beats both orders under Prop. 1 AdamW analogue on the optimizer state (parameters,</td><td>partly inherited partly inherited</td></tr></table>

Setup and token-level decomposition. Let $\theta _ { 0 }$ be the base model and η the step size, and let $g _ { D } : = \nabla \mathcal { L } _ { D } ( \theta _ { 0 } )$ and $H _ { D } : = \bar { \nabla } ^ { 2 } \mathcal { L } _ { D } ( \theta _ { 0 } )$ be the gradient and Hessian of the training loss of source D. For two gradient sources $A , B$ with single-step SGD, the endpoint difference obeys $\theta _ { A B } - \theta _ { B A } =$ $\eta ^ { 2 } b _ { A B } ( \theta _ { 0 } ) \stackrel {  } { + } O ( \eta ^ { 3 } )$ , where $b _ { A B } : = \bar { H _ { B } ^ { - } } g _ { A } - \mathbf { \bar { \Sigma } } \mathbf { \bar { \Sigma } } \mathbf { \bar { \Sigma } }$ [Sweeney, 2026a, Lemma 2.1]; we give a self-contained Euler/BCH derivation in Appendix A. Projecting this bracket onto a target gradient gives a scalar order criterion [Rukhovich et al., 2025]. Evaluated at the Trotter reference $\theta _ { \mathrm { r e f } } : =$ $\theta _ { 0 } - \eta ( g _ { A } + g _ { B } )$ , the scalar $\sigma = \langle g _ { E } ( \theta _ { \mathrm { r e f } } ) , b _ { A B } ( \theta _ { 0 } ) \rangle$ predicts the better pairwise order with 82– 92% overall sign accuracy and 82–100% on the decisions with the largest predicted gaps [Sweeney, 2026a]. The scalar says which order, not where. Let $e ( x , y ; \theta ) : = \mathrm { s o f t m a x } ( z ( x ; \theta ) ) - \mathrm { o n e h o t } ( y )$ be the cross-entropy error in logit space and $\delta z ( x ; v ) : = \dot { J } _ { \theta } \dot { z ( x ) }$ v the logit-space directional derivative along a parameter displacement v. We decompose σ across vocabulary by the chain rule:

$$
\tau _ { k } ( A , B ; E ) : = \mathbb { E } _ { ( x , y ) \sim E } \big [ e _ { k } ( x , y ; \theta _ { \mathrm { r e f } } ) \cdot \delta z _ { k } ( x ; \eta ^ { 2 } b _ { A B } ( \theta _ { 0 } ) ) \big ] , \qquad \sum _ { k } \tau _ { k } = \eta ^ { 2 } \sigma .\tag{1}
$$

The identity is exact for the true logit JVP; the implementation uses central finite differences and reports the corresponding finite-difference token readout (Appendix B). We call the map from a parameter displacement to the scores $\tau _ { k }$ the token readout, and τ itself the commutator memory of source pair (A, B) on slice E: a property of the training path, not a state-invariant property of the trained endpoint. Operationally, it is a per-token attribution of the commutator of the two updates, distinct from activation-level mechanistic interpretability: given two sources and an evaluation slice, the method returns a scalar order forecast, a signed token report, and a set of tokens to downweight.

Contributions. We introduce commutator memory, a token-level readout of the Lie bracket of two training updates. We evaluate it by three tests: localization, where bracket-derived token supports are sparse, pair-specific, and recover empirical order-effect supports in endpoint checks; causal effect, where interventions on predicted harmful tokens reduce held-out ordering gaps while τ - neutral controls do not; and paired-endpoint assignment, where an antisymmetric statistic tells which order produced each of two k=1 SGD endpoints, across four LLMs. Additional DPO, GRPOstyle, and AdamW experiments test matched-path extensions beyond the main SFT setting.

Evidence map. The main SFT experiments use empirically BCH-local SGD with $k = 1 { - } 8$ steps on two-source pairs. We then ask whether the same bracket remains structured under matchedbatch DPO, a frozen-rollout reward surrogate, iterative correction toward the observed reverse-order endpoint, and AdamW (an endpoint check on the optimizer state); matched means that the bracket is computed from the same batches or rollouts that produced $\theta _ { A B }$ and $\theta _ { B A }$ . This sequence of tests keeps the main claims fixed (localized, causally actionable, paired-endpoint-readable commutator memory), while the later regimes test the same object on realized update paths. The closest prior work on training-history memory is the activation recency probe of Krasheninnikov et al. [2026]; our object is different because it is a signed, pair-specific bracket readout computed before either order is run, projected to output tokens, and used for causal order-gap interventions. The closest causal-localization comparison is mechanistic data attribution [Chen et al., 2026]: both find that an apparently diffuse training phenomenon is carried by a sparse causal support, but their support consists of training samples for circuit emergence, while ours consists of output tokens for the antisymmetric order residue.

## 2 Related Work

Training order, curricula, and temporal traces. Curriculum learning studies deliberately chosen presentation orders and frames curricula as continuation methods that can speed convergence or guide non-convex optimization [Bengio et al., 2009]. Fine-tuning work shows that data-order seeds contribute substantially to BERT fine-tuning variance [Dodge et al., 2020], and that sequential intermediate-task order can help or hurt CodeBERT transfer in software-engineering tasks [Chen et al., 2024]. Most directly, Krasheninnikov et al. [2026] show that sequential fine-tuning leaves a linear activation-space recency signal: probes can distinguish early vs. late entity datasets and the model can be trained to report a stage. Their work and ours both ask what training history survives later learning. We study the weight-space noncommutativity of two training updates: the bracket gives a signed AB-vs-BA displacement, an output-token decomposition, causal token interventions, and a paired-endpoint assignment statistic. We also measure how this displacement attenuates under continued training. We use “memory” in this parametric, pair-conditioned sense, not in the sense of agent memory systems with storage and retrieval.

Lie brackets and splitting geometry. The BCH/Magnus/splitting viewpoint is classical in differential equations and geometric numerical integration [Magnus, 1954, Blanes et al., 2009, Hairer et al., 2006]. The ICML paper by Sweeney [2026a] proves the SGD bracket identity and shows that the scalar $\sigma = \langle g _ { E } , b _ { A B } \rangle$ predicts ordering quality across multi-domain suites; we inherit this identity and lift it to token-level localization, intervention, and paired-endpoint readability. Rukhovich et al. [2025] project the same bracket onto a target-loss gradient as a local optimality criterion for multidomain learning. Sweeney [2026b] shows that optimizer state that advances with the step count, such as AdamW moment buffers and de-biasing counters, makes the effect of reordering the same data first order in η rather than second order; our AdamW appendix replays this optimizer state, but only as an endpoint check after training.

Continual learning and editing. Catastrophic interference and continual-learning methods such as EWC, SI, GEM, and PCGrad [McCloskey and Cohen, 1989, Kirkpatrick et al., 2017, Zenke et al., 2017, Lopez-Paz and Ranzato, 2017, Yu et al., 2020] address forgetting or gradient conflict; ROME/MEMIT [Meng et al., 2022, 2023] show how factual content can be written into weights; our complementary question is what order-specific interaction is created when parameter writes are composed. Commutator memory instead resolves the antisymmetric path residue $\theta _ { A B } - \theta _ { B A }$ into output-token support. Task arithmetic [Ilharco et al., 2023] composes whole-model fine-tuning vectors as task differences; the bracket $b _ { A B }$ is an order-asymmetric residue within a single training sequence, and loss responses to small masked bracket edits on Qwen-3-4B are near-additive (4.3% mean relative error; App. P).

Mechanistic localization and data attribution. Mechanistic-interpretability work localizes behavior to circuits, activation features, and editable internal representations [Olah et al., 2020, Elhage et al., 2021, Wang et al., 2023, Conmy et al., 2023, Bricken et al., 2023, Templeton et al., 2024, Belrose et al., 2023]. Influence functions, datamodels, TRAK, and recent LLM influence studies [Koh and Liang, 2017, Ilyas et al., 2022, Park et al., 2023, Grosse et al., 2023] attribute predictions or behaviors to training examples. Mechanistic data attribution [Chen et al., 2026] traces interpretable units to high-influence training samples; our attribution is not sample-level or activation-level, but a token-level decomposition of the commutator of two training updates.

Attribution, fingerprinting, and training protocols. Watermarking [Kirchenbauer et al., 2023, Zhao et al., 2024] embeds a signal during generation; commutator memory arises without one. Membership inference [Shokri et al., 2017, Carlini et al., 2022] recovers set membership, not order. Seed fingerprinting [Tong et al., 2026] recovers the initialization seed from seed-induced prediction biases; our paired-endpoint statistic identifies which of two candidate sources was trained first (k=1 SGD). DPO [Rafailov et al., 2023], PPO [Schulman et al., 2017], and GRPO [Shao et al., 2024] provide the preference/RL objectives used only in the controlled matched-path tests. Adam and AdamW [Kingma and Ba, 2015, Loshchilov and Hutter, 2019] motivate the lifted-state optimizer check rather than the main SGD claims.

## 3 Background: Lie Bracket Analysis of Alternating Training

## 3.1 Problem Setup

Let $\theta \in \mathbb { R } ^ { p }$ denote model parameters and $\mathcal { L } _ { A } , \mathcal { L } _ { B }$ denote loss functions on domains A and B. Consider alternating gradient descent:

$$
\theta _ { 1 } = \theta _ { 0 } - \eta \nabla \mathcal { L } _ { A } ( \theta _ { 0 } )\tag{2}
$$

$$
\theta _ { 2 } = \theta _ { 1 } - \eta \nabla \mathcal { L } _ { B } ( \theta _ { 1 } )\tag{3}
$$

versus the reverse ordering (B then A).

The ordering gap measures the difference:

$$
\Delta ( \theta _ { 0 } ) = \mathcal { L } _ { E } ( \theta _ { A B } ) - \mathcal { L } _ { E } ( \theta _ { B A } )\tag{4}
$$

where $\mathcal { L } _ { E }$ is a held-out evaluation loss and $\theta _ { A B } , \theta _ { B A }$ denote final parameters after sequences AB and BA.

## 3.2 Lie Bracket Decomposition

Smoothness assumption (used throughout). The source and target losses considered below are $C ^ { 3 }$ on the local neighborhood of $\theta _ { 0 }$ and $\theta _ { \mathrm { r e f } }$ with bounded third derivatives, so the $O ( \eta ^ { 3 } )$ endpoint remainders and target-loss Taylor remainders are uniform; this is the regularity inherited from Sweeney [2026a] Lemma 2.1.

For discrete gradient descent with step size $\eta ,$ the parameter difference after two alternating steps satisfies

$$
\theta _ { A B } - \theta _ { B A } = \eta ^ { 2 } b _ { A B } ( \theta _ { 0 } ) + O ( \eta ^ { 3 } ) ,\tag{5}
$$

where the bracket $b _ { A B }$ is defined via Hessian-vector products:

$$
b _ { A B } ( \theta ) : = H _ { B } ( \theta ) g _ { A } ( \theta ) - H _ { A } ( \theta ) g _ { B } ( \theta )\tag{6}
$$

with $g _ { D } = \nabla _ { \theta } \mathcal { L } _ { D }$ and $H _ { D } = \nabla _ { \theta } ^ { 2 } \mathcal { L } _ { D }$ (Lemma 2.1 of Sweeney [2026a]; Euler/BCH derivation in Appendix A). Appendix A also reads $b _ { A B }$ as the curvature of an optimizer connection. We compute $b _ { A B }$ via exact HVPs [Pearlmutter, 1994], cast to fp32 on bf16 models.

## 3.3 Target Score and Trotter Reference

Following Sweeney [2026a] and the standard splitting perspective in geometric numerical integration [Hairer et al., 2006], define the first-order Trotter reference $\theta _ { \mathrm { r e f } } : = \theta _ { 0 } - \eta ( g _ { A } ( \theta _ { 0 } ) + g _ { B } ( \bar { \theta _ { 0 } } ) )$ the shared first-order endpoint of the two schedules. Taylor-expanding around $\theta _ { \mathrm { r e f } }$ and substituting eq. (5) yields the bracket score

$$
\Delta _ { E } ( A , B ) : = \mathcal { L } _ { E } ( \theta _ { A B } ) - \mathcal { L } _ { E } ( \theta _ { B A } ) = \eta ^ { 2 } \langle g _ { E } ( \theta _ { \mathrm { r e f } } ) , b _ { A B } ( \theta _ { 0 } ) \rangle + O ( \eta ^ { 3 } ) .\tag{7}
$$

The ${ \cal O } ( \eta ^ { 4 } )$ loss-linearization remainder for the actual endpoint displacement is dominated by the $O ( \eta ^ { 3 } )$ Euler truncation in $\theta _ { A B } - \theta _ { B A } \approx \eta ^ { 2 } b _ { A B }$ . Proposition 1 (§6, App. A) gives a conditional leading-order comparison: when the symmetric drift on E raises the target loss $( \mu > 0 , \ S 6 )$ and the sign of σ is estimated correctly, the bracket-only step at $\theta _ { \mathrm { r e f } }$ has lower target loss than both pure orders up to the stated remainder. The companion closure identity $\theta _ { A B } - \bar { \eta } ^ { 2 } b _ { A B } = \theta _ { B A } +$ $\mathrm { { \dot { \cal { O } } } } ( \eta ^ { 3 } )$ underlies the matched-batch correction results summarized in Section 5.6 and detailed in Appendix G and Appendix AF.

## 3.4 Token-Level Decomposition

The bracket $b _ { A B } \in \mathbb { R } ^ { p }$ acts on parameters; Eq. (1) maps it to tokens through the logit Jacobian. Token k contributes $\tau _ { k }$ to the leading-order gap; $\tau _ { k } > 0$ means AB disadvantages token k relative to BA at the bracket-prediction level. Individual entries are reported in the model’s native logit coordinates; the aggregate identity is invariant to position-wise logit shifts because $\textstyle \sum _ { k } e _ { k } = 0$ . In experiments we compute the logit JVP by central finite difference with step 1.0 after exact HVPs. All downstream sections characterize this readout: sparsity (§5.1), support interpretation (§5.2), and the controlled tests beyond supervised fine-tuning in Section 5.6.

## 4 Experimental Setup

## 4.1 Models and Domains

Models: Qwen-3-4B [Yang et al., 2025], Qwen-2.5-1.5B [Qwen Team, 2024], Llama-3.1- 8B [Grattafiori et al., 2024], Llama-3.2-1B [Meta AI, 2024]; DPO sparsity also uses Qwen-3-8B, and the GRPO-style reward surrogate uses Qwen-2.5-0.5B-Instruct. SFT domains: code, news, math, legal, biomedical, wikipedia from The Pile [Gao et al., 2020]; 512 tokens/sequence. Sparsity and token-prediction runs use all C(6, 2)=15 pairs × 3 seeds. The SFT intervention rows in Table 3 use three fixed domain pairs × 30 seeds (90 trials/model; a trial is one seeded run of one pair); single-step and iterative endpoint correction (edits of $\theta _ { A B }$ toward $\theta _ { B A } )$ use the fixed six-pair subset stated in their tables. DPO uses pairs of UltraFeedback [Cui et al., 2024] source partitions. The GRPO-style protocol uses frozen matched rollouts and two analytic reward sources (math correctness and brevity/format compliance). Learning rates η are selected by an empirical BCH-locality check (Qwen-3-4B 6.42e-4, Qwen-2.5-1.5B 1.91e-3, Llama-3.1-8B 1.28e-3, Llama-3.2-1B 8.0e-4; Appendix T); for the DPO and GRPO-style correction experiments the local η is recalibrated per protocol (Appendix G and Appendix AF). HVPs are computed in fp32 via upcast; held-out evaluation uses documents disjoint from training.

## 4.2 Closure metric and measurement floor

The closure metric used in the intervention and correction sections is $C : = 1 - | \Delta _ { \mathrm { e d i t e d } } | / | \Delta _ { \mathrm { b a s e l i n e } } |$ where $\Delta _ { \mathrm { b a s e l i n e } } = \mathcal { L } _ { E } ( \theta _ { A B } ) - \mathcal { L } _ { E } ( \theta _ { B A } )$ and $\Delta _ { \mathrm { e d i t e d } }$ is the same quantity after the bracket-based edit. The metric is bounded above by 1 (perfect closure) and unbounded below; its sampling variance diverges as $| \Delta _ { \mathrm { b a s e l i n e } } |  0$ , since the denominator approaches the per-trial NLL noise floor (fp32 $\sim 2 { \times } 1 0 ^ { - 5 }$ ; SFT $\mathsf { b f } 1 6 \sim 1 0 ^ { - 4 } )$ . We adopt three complementary reporting conventions that are inherited by every downstream result.

(i) Above-floor subset. For each protocol, a floor on $| \Delta _ { \mathrm { b a s e l i n e } } |$ is fixed before evaluation: $1 0 ^ { - 4 }$ for the Qwen-2.5-1.5B fp32 SFT intervention (28 of 90 trials retained), $5 \times 1 0 ^ { - 3 }$ for Qwen-3-4B iterative correction (5 of 6 pairs retained: news vs. biomedical has $| \Delta | = 8 . 5 { \times } 1 0 ^ { - 4 } \mathrm { a t } k { = } 1$ , below every other pair), and $2 \times 1 0 ^ { - 5 }$ as the DPO diagnostic denominator floor used in Appendix N. We report all-trial and above-floor statistics in parallel where both are computable.

(ii) Winsorized effect-size summaries. Because closure ratios are heavy-tailed near the measurement floor, we report the median and a winsorized 5/95 mean (App. F) as descriptive effect-size summaries. Inferential claims use paired comparisons or pair-level tests rather than the raw triallevel mean.

(iii) Pair-level summary. The DPO closure summary uses source-pair medians of the bracketminus-random closure difference. The criterion fixed in advance, a per-trial bootstrap 95% CI strictly above zero, holds for 11/12 source pairs (the original six UltraFeedback pairs and six disjoint fresh pairs; App. O); all 12/12 pair medians are positive. The pairs share source domains, so this sign agreement is descriptive, not a significance test. The trial-level count $( 3 1 6 / 3 6 0 = 8 7 . 8 \%$ matched trials with bracket > random) is within-cluster consistency, not an independent-sample p-value. Winsorized 5/95 means and seed-clustered bootstrap CIs are effect-size summaries, not decision criteria.

## 5 Results

The results follow the three tests set out in the introduction. Sections 5.1–5.2 test whether the antisymmetric residue is localized; Section 5.4 tests whether the localized support is causal; Section 5.5 tests whether the endpoint weights retain a readable antisymmetric trace. The other sections are validity checks and controlled matched-path tests: agreement with the prior scalar predictor, persistence across multiple SGD steps, matched-path correction, and the lack of transfer to unmatched GRPO rollouts.

## 5.1 Localized: Sparse, Pair-Specific Token Readout

The finite-difference readout is highly concentrated (Fig. 2), but concentration alone is not the specificity claim: random directions can also produce heavy-tailed logit readouts. Our bracket-specific evidence is support alignment. Bracket-derived supports are pair-specific and, when the measured $\theta _ { A B } - \theta _ { B A }$ is passed through the same readout, recover empirical order-effect supports far above the corresponding random-direction controls (App. R).

Table 2: Concentration and support-specificity checks from the fp32 random-direction control subset (3 pairs × 3 seeds/model). “80% mass”: fraction of the vocabulary holding 80% of the |τ| mass (mean/median). Endpoint (the measured $\theta _ { A B } - \theta _ { B A } )$ and resampled (a bracket from disjoint batches) columns report top-20 overlap with the bracket support; the random column reports mean top-20 overlap for norm-matched random controls.
<table><tr><td>Model</td><td>Precision</td><td>80% mass mean/med.</td><td>Endpoint/resampled overlap</td><td>Random overlap</td></tr><tr><td>Llama-3.2-1B</td><td>fp32</td><td>1.47% / 1.47%</td><td>99% / 97%</td><td>35%</td></tr><tr><td>Qwen-2.5-1.5B</td><td>fp32</td><td>0.99% / 1.16%</td><td>99% /97%</td><td>39-40%</td></tr><tr><td>Qwen-3-4B</td><td>fp32</td><td>0.30% / 0.042%</td><td>82% / 93%</td><td>36-49%</td></tr></table>

Support specificity, not Gini alone. Top-token sets are pair-specific rather than a universal highinfluence vocabulary set: cross-pair Jaccard is 0.004 on Qwen-3-4B (Fig. 6). In the endpoint checks used in Table 2, empirical endpoint displacements and batch-resampled brackets preserve the bracket’s top-token support, while norm-matched random controls share only the tokens that any direction excites. Frequency-normalized $\kappa _ { k } = \tau _ { k } / ( \mathrm { P r } ( y { = } k ) + \varepsilon )$ is reported as a label-frequency diagnostic; support recovery, not κ-Gini, is the specificity test. Appendix S checks how much of $b _ { A B }$ the token readout can see.

![](images/f9658a576136fc8678f30d0b0ca4659e0bc1b071d30cf0ba33914248e229c1ef.jpg)  
Figure 2: Concentration diagnostics for the token-level bracket readout. Lorenz curves show the operational finite-difference τ readout across four reported main runs; Table 2 reports the separate fp32 random-direction control subset used for the support-specificity claim.

## 5.2 Localized: The Sparse Support Is Interpretable

Top-20 $\left| \tau _ { k } \right|$ tokens fall into three token categories, each pair with its own signature. Domainmarker tokens dominate code-vs-news at $\theta _ { \mathrm { r e f } }$ (lexical proper nouns: “Facebook”, “Amazon”, “Washington”; a $\theta _ { 0 }$ -reference variant of the same pair surfaces syntactic markers xml, \$, php, App. Q) and code-vs-math (academic metadata + LaTeX source structure). Digit asymmetry dominates code-vs-legal (9/20 are digits 0–9; token $\ " 0 \ "$ alone has $\tau = + 2 3 . 1 )$ plausibly array indices vs. section numbers. BPE morpheme fragments dominate code-vs-biomedical (“-ervation” from observation/conservation/preservation; “-fluence”; “-ulatory”): polysyllabic domain terms collapse into a few shared subword pieces, concentrating |τ<sub>k</sub>| via many-to-few mapping. Token heatmap and per-pair token examples: Appendix Y, Fig. 6.

Category summary. The three categories identify tokens with the largest error-weighted logit response $\boldsymbol { e } _ { k } \delta z _ { k }$ along the bracket. The prediction and label parts of $\tau _ { k }$ (App. Y) are comparable in aggregate (ratio 0.88); 30% of $| \tau |$ mass comes from prediction-only tokens. Smoothed $\kappa _ { k } = \tau _ { k } / ( \mathrm { P r } ( y { = } k ) + \varepsilon )$ is a label-frequency diagnostic. The operational readout is consistently heavy-tailed across the tested architectures; the bracket-specific claim is the stability and endpoint alignment of the high-mass support. The categories recur across pairs; reference and eval-slice choices can change the surface tokens, but the observed high-mass categories remain domain-linked. Per-model detail: Appendix Y.

Concrete examples. In held-out code-vs-news text, “Ask HN: Should I include my GPA and/or transcripts when applying for jobs?” starts with Ask, the largest-|τ| token of that slice $( \tau = - 3 . 5 1 ) ;$ in code-vs-biomedical text, “the ambulatory health care environment” contains ulatory, a top-10 |τ | subword $( \tau = 0 . 4 3 7 )$ . The readout is aggregated over the slice, not scored per sentence (Fig. 8).

## 5.3 Agreement with the Scalar Bracket Predictor

The scalar bracket-projection predictor $\sigma ~ = ~ \langle g _ { E } , b _ { A B } \rangle$ for pairwise order quality was validated as an order predictor in prior ICML work [Sweeney, 2026a], which reports 82–92% overall sign accuracy and 82–100% accuracy on highest-impact decisions. Our identity gives $\textstyle \sum _ { k } \tau _ { k } = \eta ^ { \bar { 2 } } \sigma$ (Eq. 1); we verify that our pipeline reproduces its sign accuracy on three models. Bracket signprediction accuracy on 18 pair-seed units (6 domain pairs × 3 seeds): Qwen-3-4B 72%, Qwen-2.5- 1.5B 100%, Llama-3.1-8B 100%, vs. first-order baselines (grad cosine, grad norm) at 11–56% and random at 39–50%. Magnus-expansion validation $( R ^ { 2 } = \bar { 0 . 9 9 4 }$ on Qwen-3-4B) and full baseline table: Appendix W, Z.

## 5.4 Causal: Token Interventions Close the SFT Gap

To test whether the predicted support causally affects the ordering gap, we intervene on the ten tokens whose $\tau _ { k }$ contributes most to the measured baseline gap (App. U) during alternating training. On Qwen-3-4B we use mean-preserving loss reweighting $( \alpha = 0 . 1 , \mathrm { i . e . }$ , 90% downweight) on these label tokens; on Qwen-2.5-1.5B fp32 we use the lower-noise variant that scales the learning rate of those tokens’ output-layer (lm-head) rows, on the predefined above-floor subset. Controls are random token sets with $\tau _ { k }$ near zero, drawn to match the harmful tokens’ label-frequency bins (30 trials per domain pair, held-out evaluation data disjoint from training).

Table 3: Intervention results on two models (30 trials/pair, held-out evaluation). Qwen-3-4B reports all 90 trials. $\mathsf { \Pi } ^ { \dagger } \mathbf { Q } \mathrm { w e n } { - 2 . 5 }$ denotes Qwen-2.5-1.5B fp32 on the predefined above-floor subset (28/90 trials with $| \Delta _ { \mathrm { b a s e l i n e } } | > 1 0 ^ { - 4 } )$ ). Harmful: the ten tokens whose $\tau _ { k }$ contributes most to the measured gap; Random: frequency-matched tokens with $\tau _ { k } \approx 0 ;$ entries are closure $C$ in %; Winsor. mean: $5 / 9 5$ winsorized mean, with its bootstrap CI; Gap shrinks: fraction of trials with a smaller ordering gap after the intervention (68/90, 45/90, 27/28, 11/28 by row).
<table><tr><td>Model</td><td>Condition</td><td>Winsor. mean</td><td>Median</td><td>95% CI</td><td>Gap shrinks</td></tr><tr><td>Qwen-3-4B</td><td>Harmful</td><td>+31.2%</td><td>+32.0%</td><td>[+21.4%, +39.7%]</td><td>75.6%</td></tr><tr><td>Qwen-3-4B</td><td>Random</td><td>+1.2%</td><td>+0.2%</td><td>[–5.8%, +8.7%]</td><td>50.0%</td></tr><tr><td>Qwen-2.5†</td><td>Harmful</td><td>+48.6%</td><td>+47.9%</td><td>[+40.5%, +57.1%]</td><td>96.4%</td></tr><tr><td> $\mathrm { Q w e n } { - 2 . 5 ^ { \dagger } }$ </td><td>Random</td><td>-0.006%</td><td>-0.003%</td><td>[–0.03%, +0.02%]</td><td>39.3%</td></tr></table>

On Qwen-3-4B (held-out, all 90 trials), intervening on predicted harmful tokens reduces the ordering gap by a median 32% (the winsorized-mean CI excludes zero), while random controls show no effect. The paired harmful-vs-random comparison yields a CI entirely above zero (harmful-minusrandom gap reduction [0.00096, 0.0022] nats, $p \ < \ 0 . 0 0 1$ , Cohen’s $d = 0 . 5 3 )$ . Both HVP terms matter: selecting tokens with the full bracket $H _ { B } g _ { A } - H _ { A } g _ { B }$ gives $d = 0 . 5 3$ , whereas the $H _ { B } g _ { A }$ term alone gives $d = 0 . 2 5$ . On Qwen-2.5-1.5B (full fp32 with lm-head learning-rate scaling), the predefined above-floor protocol keeps only the 28 of 90 trials with $\vert \Delta _ { \mathrm { b a s e l i n e } } \vert > 1 \bar { 0 } ^ { - 4 }$ (the remaining 62 trials are below the reporting floor); on this above-floor fp32 subset, learning-rate scaling on the harmful tokens yields $a { \bar { + } } 4 7 . 9 \%$ median gap reduction, with the random control near zero. The paper’s comparison is the within-model bracket-vs-random separation; cross-model magnitudes are not directly compared. The 32% figure is the effect of this reweighting edit, not the share of the ordering gap that commutator memory explains: after the edit, the leading-order bracket still predicts 71% of the original gap, and finite-step corrections, which change sign across pairs, do not explain a common remainder (App. V).

## 5.5 Readable: The Endpoint Pair Identifies Which Order Was Trained

A complementary stringent test of “memory” is paired-endpoint assignment: after both orders have been run, can the bracket assign which endpoint came from which order? Given a public base $\theta _ { 0 } .$ candidate training-domain pair $( A , B )$ , and paired alternate-order endpoints $\theta _ { A B } , \theta _ { B A }$ , Theorem 1 and Corollary 1 (App. A) give the $k = 1$ SGD criterion: the antisymmetric statistic

$$
\Delta s : = \langle \theta _ { A B } - \theta _ { B A } , b _ { A B } \rangle\tag{8}
$$

has leading term $\eta ^ { 2 } \| b _ { A B } \| ^ { 2 }$ when the first endpoint is $\theta _ { A B }$ , with the sign reversed if the endpoints are swapped. Thus the sign assigns the endpoints whenever the $O ( \eta ^ { \overline { { 2 } } } )$ term dominates finite-η, sampling, and numerical remainders. The single-endpoint estimator sign s(θ), with $s ( \theta ) : = \langle \theta -$ $\theta _ { \mathrm { r e f } } , \ b _ { A B } \rangle$ , reaches 77.8% only on Qwen-3-4B and falls to chance at 1B, 1.5B, and 8B (both $s ( \theta _ { A B } )$ and $s ( \theta _ { B A } )$ land on the same side of zero). The antisymmetric difference removes the shared firstorder reference offset and the second-order symmetric term $c _ { A B } \ : ( \ S 6 )$ in the expansion, and it stays above chance at all four scales.

Table 4: Paired-endpoint training-order assignment via sign(∆s) across four LLMs (6 pairs $\times ~ 3$ seeds = 18 pair-seed units per model). Replication across precision regimes (bf16 vs. fp32) in Appendix K.
<table><tr><td>Model (precision)</td><td>sign(∆s) correct</td><td>Wilson 95% CI</td></tr><tr><td>Llama-3.2-1B (1B, fp32)</td><td> $1 4 / 1 8 = 7 7 . 8 \%$ </td><td>[55%, 91%]</td></tr><tr><td>Qwen-2.5-1.5B (1.5B, fp32)</td><td> $1 6 / 1 8 = 8 8 . 9 \%$ </td><td>[67%, 97%]</td></tr><tr><td>Qwen-3-4B (4B, bf16)</td><td> $1 8 / 1 8 = 1 0 0 . 0 \%$ </td><td>[82%, 100%]</td></tr><tr><td>Llama-3.1-8B (8B, bf16)</td><td> $1 8 / 1 8 = 1 0 0 . 0 \%$ </td><td>[82%, 100%]</td></tr><tr><td>Combined</td><td> ${ \bf 6 6 } / 7 2 = 9 { \bf 1 . 7 \% }$ </td><td>[83%, 96%]</td></tr></table>

Assignment is binary, so chance is 50%; the combined Wilson interval [83%, 96%] (Table 4) lies above it. In this paired-endpoint setting, antisymmetry supplies the leading-order sign criterion; the empirical results test whether it remains readable under finite-sample and finite-step effects. Scope: $k = 1 \mathrm { S G D }$ ; AdamW transfer is in App. AB.

## 5.6 Controlled Tests Beyond Supervised Fine-Tuning

The bracket’s top-token set persists beyond the single-step derivation: across $k \in \{ 1 , 2 , 4 , 8 \}$ SGD steps per source, both Qwen models retain concentrated τ readouts, and $\geq 9 8 \%$ of the k=8 empirical top-1% tokens (among those occurring as labels) lie in the single-step $t o p \mathrm { - } 1 \% \ \tau$ set (Appendix Z). This asymmetric recall is distinct from the equal-cardinality support-prediction check in Appendix J. The scalar BCH forecast degrades on Qwen-3-4B at the largest calibrated $\eta ,$ but the support remains useful: the same model still gives strong support recovery, output-row prediction (Spearman $\rho$ between $\tau _ { k }$ and the output-row displacement, from 0.83 at k=1 to 0.71 at k=8), and 32% intervention closure.

Parameter-space correction. Edits to the trained endpoint using $- \eta ^ { 2 } b _ { A B }$ recover the reversed endpoint only in matched-path settings: subtracting $\eta ^ { 2 } b _ { A B }$ once from $\theta _ { A B }$ closes amounts that vary by pair, but $\tau _ { k }$ predicts the actual per-token output-row displacement on three models (per-pair Spearman $\rho = 0 . 7 8 – 0 . 9 0 \mathrm { { : } }$ ; App. I), and iterative projection toward the observed $\theta _ { B A }$ closes about 90% of the gap on the 5/6 SFT pairs above floor. The conditional bracket-only update at $\theta _ { \mathrm { r e f } }$ also beats both pure orders when the symmetric drift on E is target-disadvantageous $( \mu > 0 ;$ , §6) and the sign of σ is estimated correctly: on Qwen-2.5-1.5B fp32, 12 of the 13 units with $\hat { \mu } > 0$ satisfy $\mathcal { L } _ { E } ( \bar { \theta _ { \mathrm { B O } } } ) < \mathrm { m i n } ( \mathcal { L } _ { E } ( \theta _ { A B } ) , \mathcal { L } _ { E } ( \bar { \theta } _ { B A } ) )$ ), and the improvement over the better order regresses on the predicted −η<sup>2</sup>µˆ (µˆ the estimated µ) with slope 1.15 (proof in App. A).

Preference and reward-surrogate tests. Matched-batch DPO gives the most direct correction test: subtracting the second-order $\eta ^ { \ddag } b _ { A B }$ update from $\theta _ { A B }$ closes a median 77% of the held-out DPO gap to $\theta _ { B A } ~ ( \mathrm { A p p }$ . D). The flat-bootstrap strict-CI criterion holds for $1 1 / 1 2$ source pairs, all 12/12 pair medians are positive, and trial-level 316/360 is reported only as within-cluster consistency. DPO concentration checks on Qwen-3-8B and Qwen-2.5-1.5B are descriptive (App. L). A frozen matched-rollout reward surrogate on Qwen-2.5-0.5B-Instruct tests whether the closure identity holds under analytic rewards through $T = 1 6$ sequential update blocks. Detailed DPO/GRPO controls, random, wrong-sign $( + \eta ^ { 2 } b _ { A B } )$ and first-order (norm-matched $\eta ( g _ { A } - g _ { B } ) )$ comparisons, and the AdamW endpoint check are in Appendices G, AF, and AB.

Persistence under continued training. In a new-seed Qwen-3-4B study (20 runs over ten source pairs), the AB and BA endpoints each take sixteen more SGD updates on the same third-domain batches. After four, eight, and sixteen updates, 15, 14, and 13 of 20 runs retain the aligned trace, a positive projection of $\theta _ { A B } - \theta _ { B A }$ on the original commutator direction; some runs reverse sign, and over the 18 runs that start aligned the median retained fraction falls from 27% to 14% to 7%. The aligned trace persists briefly and then fades; this does not establish behavioral retrieval or long-term memory (App. AC).

## 6 Discussion

Path dependence. $b _ { A B }$ is a local correction for a realized update path. Matched-batch SFT/DPO and matched-rollout GRPO closures succeed because the bracket comes from the same updates that produced the two endpoints; calibration on unmatched rollouts is the negative control (App. AF). A residue concentrated on few tokens and rows is consistent with aggregate benchmarks missing it. With $\theta _ { B A }$ as target, iterative projection closes about 90% of the gap on the $5 / 6 ~ \mathrm { S F T }$ pairs above floor, removing the antisymmetric residue after training with the observed $\theta _ { B A }$ as target.

Why this matters. The practical target is controlled measurement of sequential post-training in teractions. Instead of discovering source interactions only after expensive runs as scalar regressions, a pipeline could estimate which evaluation slices and output tokens carry the antisymmetric residue, then test reordering, token-local reweighting, or matched-path bracket correction in a calibrated BCH-local window. In the current evidence these are measurements, not deployment tools: the bracket remains tied to realized batches, rollouts, optimizer state, and the calibrated step-size window.

Why bracket-driven updates are more than order prediction. At the drift-matched reference $\theta _ { \mathrm { r e f } } = \theta _ { 0 } - \eta ( g _ { A } + g _ { B } )$ , define $c _ { A B } : = { \textstyle { \frac { 1 } { 2 } } } ( H _ { B } g _ { A } + H _ { A } g _ { B } ) , \mu : = \langle g _ { E } ( \theta _ { \mathrm { r e f } } ) , c _ { A B } \rangle$ , and $\sigma : =$ $\langle g _ { E } ( \theta _ { \mathrm { r e f } } ) , b _ { A B } \rangle$ . For the bracket-only step $\begin{array} { r } { \bar { \theta _ { \mathrm { B O } } } : = \theta _ { \mathrm { r e f } } - \frac { 1 } { 2 } \eta ^ { 2 } \cdot \mathrm { s i g n } ( \hat { \sigma } ) \cdot b _ { A B } } \end{array}$ (σˆ an estimate of σ), Proposition 1 gives

$$
{ \mathcal L } _ { E } ( \theta _ { \mathrm { B O } } ) - \mathrm { m i n } \{ { \mathcal L } _ { E } ( \theta _ { A B } ) , { \mathcal L } _ { E } ( \theta _ { B A } ) \} = - \eta ^ { 2 } \mu + O ( \eta ^ { 3 } ) .\tag{9}
$$

When $\mu > 0$ (the symmetric drift on E is target-disadvantageous) and the sign of $\sigma$ is estimated correctly, this update beats the better pure order by $\eta ^ { 2 } \mu$ to leading order in $\eta \ ( \mathrm { A p p . \ : A } )$ . Thus the correction experiments are not redundant with order prediction, which forgoes the $\eta ^ { 2 } \mu$ term. The companion closure identity $\theta _ { A B } - \eta ^ { 2 } b _ { A B } = \theta _ { B A } + \mathsf { \bar { O } } ( \eta ^ { 3 } )$ motivates the matched-batch DPO edit $\theta _ { A B } - \eta ^ { 2 } b _ { A B } ;$ GRPO-style results are closure checks on matched rollouts (App. AF).

## 7 Limitations

The framework measures the antisymmetric, path-local component of source interaction. The SFT intervention metric is order-gap NLL closure, not absolute loss improvement or discrete capability preservation; at $k { = } 1$ , we make no claim about GSM8K exact match or HumanEval pass@1 (App. AA). The reported SFT protocol uses SGD with k = 1–8 and two sources per path; DPO correction requires fp32 and is matched-batch; iterative recovery requires $\theta _ { B A }$ . GRPO-style evidence comes from a frozen matched-rollout KL surrogate on a 0.5B model with analytic rewards $( \mathsf { A p } \cdot$ pendix AF). AdamW evidence is limited. The unmodified SGD bracket is at most weakly predictive of AdamW order (57.1% at k=20; Sweeney, 2026a, App. D.7), so we use the commutator on the optimizer state instead: we find token concentration of the gradient-space readout on Qwen-3-4B (Gini 0.980) and endpoint closure in one subspace of Qwen-2.5-1.5B fp32 (parameter cosine 0.998, NLL gap closure 79.5% [72.6%, 85.9%]; App. AB). DPO bracket-direct edits and GRPO-style closure checks in the main text use SGD, where the BCH identity is exact at $O ( \eta ^ { 2 } )$ . “Memory” here is parametric and pair-conditioned: the two candidate sources are supplied, the strongest test compares both orderings, and the trace attenuates under continued training (App. AC); we claim neither itemaddressable recall nor broad capability gains. App. AD gives the per-pair cost, exact reuse across pairs, and cheaper curvature approximations.

## 8 Conclusion

Commutator memory turns the noncommutativity of two training updates into a measurable object with four uses of one bracket: a scalar order forecast, a sparse token report, a path-local parameterspace direction, and a paired-endpoint statistic that tells which order produced each $k { = } 1 \operatorname { S G D }$ endpoint, correct 91.7% of the time across four LLMs (Corollary 1).

## Acknowledgments and Disclosure of Funding

No funding or competing interests to declare.

## References

Nora Belrose, Igor Ostrovsky, Lev McKinney, Zach Furman, Logan Smith, Danny Halawi, Stella Biderman, and Jacob Steinhardt. Eliciting latent predictions from transformers with the tuned lens. arXiv preprint arXiv:2303.08112, 2023.

Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In Proceedings of the 26th Annual International Conference on Machine Learning, pages 41–48. ACM, 2009. doi: 10.1145/1553374.1553380. URL https://icml.cc/2009/papers/119. pdf.

Sergio Blanes, Fernando Casas, Jose A Oteo, and Jose Ros. The magnus expansion and some of its applications. Physics Reports, 470(5-6):151–238, 2009.

Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E. Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards monosemanticity: decomposing language models with dictionary learning. Transformer Circuits Thread, 2023.

Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and Florian Tramer. Membership inference attacks from first principles. In IEEE Symposium on Security and Privacy, pages 1897–1914, 2022.

Jianhui Chen, Yuzhang Luo, and Liangming Pan. Mechanistic data attribution: Tracing the training origins of interpretable LLM units. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. URL https://arxiv.org/abs/2601.21996.

Qihong Chen, Jiawei Li, Hyunjae Suh, Lianghao Jiang, Zheng Zhou, Jingze Chen, Jiri Gesi, and Iftekhar Ahmed. Does the order of fine-tuning matter and why? arXiv preprint arXiv:2410.02915, 2024. doi: 10.48550/arXiv.2410.02915. URL https://arxiv.org/abs/2410.02915.

Arthur Conmy, Augustine N Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adrià Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability. Advances in Neural Information Processing Systems, 36:16318–16352, 2023.

Ganqu Cui, Lifan Yuan, Ning Ding, Guanming Yao, Bingxiang He, Wei Zhu, Yuan Ni, Guotong Xie, Ruobing Xie, Yankai Lin, Zhiyuan Liu, and Maosong Sun. UltraFeedback: Boosting language models with scaled AI feedback. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 9722–9744. PMLR, 2024. URL https://proceedings.mlr.press/v235/cui24f.html.

Jesse Dodge, Gabriel Ilharco, Roy Schwartz, Ali Farhadi, Hannaneh Hajishirzi, and Noah A. Smith. Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping. arXiv preprint arXiv:2002.06305, 2020. doi: 10.48550/arXiv.2002.06305. URL https://arxiv.org/abs/2002.06305.

Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. A mathematical framework for transformer circuits. Transformer Circuits Thread, 2021.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. doi: 10.48550/arXiv.2407.21783. URL https://arxiv. org/abs/2407.21783.

Roger Grosse, Juhan Bae, Cem Anil, Nelson Elhage, Alex Tamkin, Amirhossein Tajdini, Benoit Steiner, Dustin Li, Esin Durmus, Ethan Perez, Evan Hubinger, Kamile Lukoši˙ ut¯ e, Karina Nguyen,˙ Nicholas Joseph, Sam McCandlish, Jared Kaplan, and Samuel R. Bowman. Studying large language model generalization with influence functions. arXiv preprint arXiv:2308.03296, 2023. doi: 10.48550/arXiv.2308.03296. URL https://arxiv.org/abs/2308.03296.

Ernst Hairer, Christian Lubich, and Gerhard Wanner. Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations, volume 31 of Springer Series in Computational Mathematics. Springer, 2 edition, 2006.

Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi, and Ali Farhadi. Editing models with task arithmetic. In International Conference on Learning Representations (ICLR), 2023.

Andrew Ilyas, Sung Min Park, Logan Engstrom, Guillaume Leclerc, and Aleksander Madry. Datamodels: Understanding predictions with data and data with predictions. In International Conference on Machine Learning (ICML), 2022.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR), 2015. URL https://arxiv.org/abs/1412. 6980.

John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. A watermark for large language models. In International Conference on Machine Learning, 2023.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings ofthe National Academy ofSciences, 114(13):3521–3526, 2017.

Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions. International Conference on Machine Learning, pages 1885–1894, 2017.

Dmitrii Krasheninnikov, Richard E. Turner, and David Krueger. Fresh in memory: Training-order recency is linearly encoded in language model activations. In International Conference on Learning Representations, 2026. doi: 10.48550/arXiv.2509.14223. URL https://openreview.net/ forum?id=Tn6famjSxN. Poster.

David Lopez-Paz and Marc’Aurelio Ranzato. Gradient episodic memory for continual learning. Advances in Neural Information Processing Systems, 30, 2017.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. doi: 10.48550/arXiv.1711.05101. URL https: //arxiv.org/abs/1711.05101.

Wilhelm Magnus. On the exponential solution of differential equations for a linear operator. Com munications on Pure and Applied Mathematics, 7(4):649–673, 1954.

James Martens and Roger Grosse. Optimizing neural networks with Kronecker-factored approximate curvature. International Conference on Machine Learning, pages 2408–2417, 2015.

Michael McCloskey and Neal J. Cohen. Catastrophic interference in connectionist networks: The sequential learning problem. In Psychology ofLearning and Motivation, volume 24, pages 109– 165. Academic Press, 1989.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in gpt. Advances in Neural Information Processing Systems, 35:17359–17372, 2022.

Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. Mass-editing memory in a transformer. In International Conference on Learning Representations, 2023.

Meta AI. Llama 3.2: Revolutionizing edge AI and vision with open, customizable models. Meta AI Blog, 2024. URL https://ai.meta.com/blog/ llama-3-2-connect-2024-vision-edge-mobile-devices/.

Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter. Zoom in: An introduction to circuits. Distill, 5(3):e00024, 2020.

Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, and Aleksander Madry. TRAK: attributing model behavior at scale. In International Conference on Machine Learning (ICML), 2023.

Barak A Pearlmutter. Fast exact multiplication by the hessian. Neural computation, 6(1):147–160, 1994.

Qwen Team. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024. doi: 10.48550/ arXiv.2412.15115. URL https://arxiv.org/abs/2412.15115.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: your language model is secretly a reward model. In Advances in Neural Information Processing Systems, 2023.

Alexey Rukhovich, Alexander Podolskiy, and Irina Piontkovskaya. Commute your domains: Tra jectory optimality criterion for multi-domain learning. arXiv preprint arXiv:2501.15556, 2025. URL https://arxiv.org/abs/2501.15556.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. doi: 10.48550/arXiv.1707. 06347. URL https://arxiv.org/abs/1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. doi: 10.48550/arXiv.2402.03300. URL https://arxiv.org/abs/2402.03300.

Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In IEEE Symposium on Security and Privacy, pages 3–18, 2017.

John Sweeney. The geometry of sequential learning: Lie-Bracket prediction of transfer order. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research. PMLR, 2026a.

John Sweeney. Optimizer memory makes shuffle order a first-order source of fine-tuning noise. arXiv preprint arXiv:2606.29554, 2026b. URL https://arxiv.org/abs/2606.29554.

Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, Hoagy Cunningham, Nicholas L. Turner, Callum McDougall, Monte MacDiarmid, Alex Tamkin, Esin Durmus, Tristan Hume, Francesco Mosconi, C. Daniel Freeman, Theodore R. Sumers, Edward Rees, Joshua Batson, Adam Jermyn, Shan Carter, Chris Olah, and Tom Henighan. Scaling monosemanticity: extracting interpretable features from Claude 3 Sonnet. Transformer Circuits Thread, 2024. URL https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html.

Yao Tong, Haonan Wang, Siquan Li, Kenji Kawaguchi, and Tianyang Hu. SeedPrints: Fingerprints can even tell which seed your large language model was trained from. In International Conference on Learning Representations (ICLR), 2026. URL https://arxiv.org/abs/2509.26404.

Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in GPT-2 small. In International Conference on Learning Representations, 2023.

An Yang et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. doi: 10.48550/ arXiv.2505.09388. URL https://arxiv.org/abs/2505.09388.

Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn. Gradient surgery for multi-task learning. Advances in Neural Information Processing Systems, 33:5824–5836, 2020.

Friedemann Zenke, Ben Poole, and Surya Ganguli. Continual learning through synaptic intelligence. In International Conference on Machine Learning, pages 3987–3995, 2017.

Xuandong Zhao, Prabhanjan Vijendra Ananth, Lei Li, and Yu-Xiang Wang. Provable robust watermarking for AI-generated text. In International Conference on Learning Representations, 2024.

## Broader impact and release statement

The framework reveals where in vocabulary space the noncommutativity of training updates concentrates and could support more interpretable diagnosis of capability shifts in sequential post-training pipelines. Corollary 1 (App. A) establishes that, given a public base $\theta _ { 0 } ,$ candidate training-domain pair $( A , B )$ , and access to both alternate-order endpoints $\theta _ { A B } , \theta _ { B A }$ , the antisymmetric statistic $\mathbf { \bar { \Delta } } s : = \langle \theta _ { A B } - \theta _ { B A } , b _ { A B } \rangle$ identifies which order produced each $k = 1$ SGD endpoint via sign(∆s) whenever its leading $\eta ^ { 2 } \dot { \| \boldsymbol { b } _ { A B } \| ^ { 2 } }$ term dominates the $O ( \eta ^ { 3 } )$ remainder (eq. 13; the paired difference cancels the shared drift). Across 72 pair-seed units on four models (Qwen-3-4B, Qwen-2.5-1.5B, Llama-3.2-1B, Llama-3.1-8B; two architectures, four scales), combined sign accuracy is $6 6 / 7 2 = 9 1 . 7 \%$ (Wilson 95% CI [83%, 96%]); both 4B and 8B units reach 100% (§5.5, Appendix K). The main paired-endpoint accuracy is for $k = 1 \mathrm { S G D } \mathrm { : }$ ; AdamW evidence is in App. AB, and we do not release endpoint-assignment tooling. We have not evaluated bypassing safety training, private-data recovery, or selective capability editing; the released material is limited to order-gap measurement.

The supplementary archive includes source code, fixed protocol definitions, result files for reported claims, and analysis scripts. Dataset versions, model checkpoints, hyperparameters, seed schedules, learning-rate calibrations, and compute budgets are documented in Appendix AE. Token-level interventions (Appendix U), the DPO correction (Appendix G), and the GRPO-style test (Appendix AF) follow these fixed protocol definitions.

## A Bracket derivation and optimizer connection

Optimizer connection and training anholonomy. It is useful to separate the training schedule from the model state. Let Θ be parameter space and let the exposure base space be $\breve { B } = \mathbb { R } _ { > 0 } ^ { N } ,$ with coordinate $s _ { i }$ measuring cumulative exposure to source i in learning-rate units. The total space is the trivial bundle $B \times \Theta \stackrel { - } { \to } B .$ Each source defines an optimizer vector field on the fiber; for SGD, $f _ { i } ( \theta ) = - \nabla _ { \theta } \mathcal { L } _ { i } ( \theta ) = - g _ { i } ( \theta )$ . We define the horizontal lift of the exposure direction $\partial _ { s _ { i } }$ by $\widetilde { X } _ { i } = \partial _ { s _ { i } } + f _ { i } ( \theta )$ . The resulting optimizer connection has vertical-valued curvature $\Omega _ { i j } : = $ $\mathrm { v e r } [ \widetilde { X } _ { i } , \widetilde { X } _ { j } ]$ . Because the exposure coordinates commute and the losses are stationary in $s ,$ this reduces to the optimizer-vector-field bracket $\Omega _ { i j } = [ f _ { i } , f _ { j } ]$ . For SGD this curvature component is exactly $\Omega _ { A B } = \mathcal { D } f _ { B } f _ { A } - \cal { D } f _ { A } f _ { B } = H _ { B } g _ { A } - \mathcal { H } _ { A } g _ { B } ^ { - }$

The two schedules $A  B$ and $B  A$ are paths in exposure space with the same endpoints (0, 0) and $( \eta , \eta )$ , but their horizontal lifts end at different model parameters. The endpoint difference is the small-rectangle anholonomy of this optimizer connection.

Self-contained derivation of eq. (5). Let $F _ { A } ( \theta ) = \theta + \eta f _ { A } ( \theta )$ and $F _ { B } ( \theta ) = \theta + \eta f _ { B } ( \theta )$ denote one Euler optimizer step, with $f _ { i } = - g _ { i }$ . Taylor expansion gives

$$
F _ { B } ( F _ { A } ( \theta _ { 0 } ) ) = \theta _ { 0 } + \eta f _ { A } + \eta f _ { B } + \eta ^ { 2 } D f _ { B } f _ { A } + O ( \eta ^ { 3 } ) ,\tag{10}
$$

$$
F _ { A } ( F _ { B } ( \theta _ { 0 } ) ) = \theta _ { 0 } + \eta f _ { B } + \eta f _ { A } + \eta ^ { 2 } D f _ { A } f _ { B } + O ( \eta ^ { 3 } ) ,\tag{11}
$$

where all vector fields and derivatives are evaluated at $\theta _ { 0 }$ . Subtracting cancels the first-order drift and leaves

$$
\theta _ { A B } - \theta _ { B A } = \eta ^ { 2 } ( D f _ { B } f _ { A } - D f _ { A } f _ { B } ) + O ( \eta ^ { 3 } ) = \eta ^ { 2 } ( H _ { B } g _ { A } - H _ { A } g _ { B } ) + O ( \eta ^ { 3 } ) .
$$

This is Lemma 2.1 of Sweeney [2026a]. The sign convention differs from continuous-time exponential-flow commutators; for the discrete update $\theta \mapsto \theta - \eta g$ , the leading-order AB-vs-BA defect is $\eta ^ { 2 } b _ { A B }$ with no $1 / 2$ factor.

Lemma 1 (Symmetric/antisymmetric Trotter expansion). Let $\theta _ { \mathrm { r e f } } : = \theta _ { 0 } - \eta ( g _ { A } + g _ { B } ) , c _ { A B } : =$ $\begin{array} { r } { \frac { 1 } { 2 } ( H _ { B } g _ { A } + \dot { H } _ { A } g _ { B } ) } \end{array}$ , and $b _ { A B } : = H _ { B } g _ { A } - H _ { A } g _ { B }$ , all evaluated at $\theta _ { 0 }$ . Then

$$
\theta _ { A B } = \theta _ { \mathrm { r e f } } + \eta ^ { 2 } \bigl ( c _ { A B } + { \textstyle \frac 1 2 } b _ { A B } \bigr ) + O ( \eta ^ { 3 } ) , \quad \theta _ { B A } = \theta _ { \mathrm { r e f } } + \eta ^ { 2 } \bigl ( c _ { A B } - { \textstyle \frac 1 2 } b _ { A B } \bigr ) + O ( \eta ^ { 3 } ) .
$$

The antisymmetric component $\eta ^ { 2 } b _ { A B }$ is the AB-vs-BA defect (Lemma 2.1 of Sweeney [2026a]); the symmetric component $\dot { \eta } ^ { 2 } c _ { A B }$ is shared by both orders.

Proof. From the two expansions above, $\theta _ { A B } = \theta _ { 0 } + \eta ( f _ { A } + f _ { B } ) + \eta ^ { 2 } D f _ { B } f _ { A } + O ( \eta ^ { 3 } )$ with $f _ { i } = - g _ { i }$ so $\begin{array} { r } { \theta _ { A B } - \theta _ { \mathrm { r e f } } = \eta ^ { 2 } H _ { B } \overset { \cdot } { g } _ { A } + O ( \eta ^ { 3 } ) = \eta ^ { 2 } ( c _ { A B } + \frac { 1 } { 2 } \dot { b } _ { A B } ^ { \cdot } ) + O ( \eta ^ { 3 } ) . \mathrm { S y m m e t r i c a l l y } } \end{array}$ for $\theta _ { B A } . \qquad \bigsqcup$

This expansion is used in the proof of Proposition 1 and Theorem 1.

Trotter-reference Taylor expansion $( \mathbf { e q . } ( 7 ) )$ . Let $\delta \theta _ { \mathrm { e x a c t } } : = \theta _ { A B } - \theta _ { B A }$ . Taylor expansion of $\mathcal { L } _ { E }$ around $\theta _ { \mathrm { r e f } }$ gives $\Delta _ { E } ( A , B ) \stackrel { - } { = } \langle g _ { E } ( \theta _ { \mathrm { r e f } } ) , \delta \theta _ { \mathrm { e x a c t } } \rangle + O ( \eta ^ { 4 } )$ . Substituting $\delta \theta _ { \mathrm { { e x a c t } } } = \eta ^ { \hat { 2 } } b _ { A B } + O ( \eta ^ { 3 } )$ yields eq. $( 7 ) ; O ( \eta ^ { 4 } )$ is the loss-linearization error for the actual endpoint displacement, while the bracket-only predictor has the expected $O ( \eta ^ { 3 } )$ Euler truncation.

Proposition: when the bracket-only update beats both pure orders. We give a proof of Equation 9 and the associated proposition stated in Section 6. The result is a direct corollary of Lemma 1. Sweeney [2026a, App. E.8] evaluated this bracket-only step empirically. The proposition below gives its conditional leading-order guarantee; we state it because it clarifies the symmetric/antisymmetric decomposition used by the matched-batch DPO closure (Appendix G) and the GRPO-style closure check (Appendix AF).

Proposition 1 (Bracket-only update can beat both pure orders). Let $\theta _ { \mathrm { r e f } } : = \theta _ { 0 } - \eta ( g _ { A } + g _ { B } )$ be the drift-matchedfirst-order reference. Let

$$
\begin{array} { r } { c _ { A B } : = \frac 1 2 ( H _ { B } g _ { A } + H _ { A } g _ { B } ) , \qquad b _ { A B } : = H _ { B } g _ { A } - H _ { A } g _ { B } } \end{array}
$$

be the symmetric and antisymmetric components ofthe second-order endpoint correction at $\theta _ { 0 } ,$ , with $\mu : = \langle g _ { E } ( \theta _ { \mathrm { r e f } } ) , c _ { A B } \rangle , \sigma : = \langle g _ { E } ( \theta _ { \mathrm { r e f } } ) , b _ { A B } \rangle$ . Define the bracket-only update

$$
\begin{array} { r } { \theta _ { \mathrm { B O } } : = \theta _ { \mathrm { r e f } } - \frac { 1 } { 2 } \eta ^ { 2 } \cdot \mathrm { s i g n } ( \hat { \sigma } ) \cdot b _ { A B } , } \end{array}
$$

where σˆ is an estimate ofthe sign ofσ (Section 5.3 reports σ-correctness on the SGD predictor). Under the smoothness assumptions ofSweeney [2026a] Lemma 2.1 (which gives an $O ( \eta ^ { 3 } )$ parameter remainder for $\theta _ { A B } , \theta _ { B A }$ relative to their second-order Taylor truncations) and a loss-linearization Taylor remainder bounded at ${ \cal O } ( \eta ^ { 4 } )$

$$
\mathcal { L } _ { E } ( \theta _ { \mathrm { B O } } ) - \operatorname* { m i n } \{ \mathcal { L } _ { E } ( \theta _ { A B } ) , \mathcal { L } _ { E } ( \theta _ { B A } ) \} = - \eta ^ { 2 } \mu + O ( \eta ^ { 3 } ) ,\tag{12}
$$

on the support where $\sigma \neq 0 a n d \mathrm { s i g n } ( \hat { \sigma } )$ is correct. In particular, when $\mu > 0$ and η is small enough that the leading $\eta ^ { 2 } \mu$ term dominates, $\theta _ { \mathrm { B O } }$ has strictly lower target loss than the better pure order.

Proof. By Lemma 1, the actual two-step SGD endpoints satisfy

$$
\theta _ { A B } - \theta _ { \mathrm { r e f } } = \eta ^ { 2 } ( c _ { A B } + { \textstyle \frac 1 2 } b _ { A B } ) + r _ { A B } , \qquad \theta _ { B A } - \theta _ { \mathrm { r e f } } = \eta ^ { 2 } ( c _ { A B } - { \textstyle \frac 1 2 } b _ { A B } ) + r _ { B A } ,
$$

with $\| r _ { A B } \| , \| r _ { B A } \| = O ( \eta ^ { 3 } )$ uniformly in the local smoothness neighborhood. The bracket-only construction $\begin{array} { r } { \dot { \theta } _ { \mathrm { B O } } - \theta _ { \mathrm { r e f } } = - \frac { 1 } { 2 } \eta ^ { 2 } \cdot \mathrm { s i g n } ( \hat { \sigma } ) \cdot b _ { A B } } \end{array}$ has no $\eta ^ { 3 }$ parameter remainder: it is defined exactly by an arithmetic combination of $g _ { A } , g _ { B }$ , and their HVPs at $\theta _ { 0 }$ . Let $s : = \mathrm { s i g n } ( \sigma )$ ; on the correct-sign support, sign(ˆσ) = s.

Taylor expansion of $\mathcal { L } _ { E }$ around $\theta _ { \mathrm { r e f } }$ gives, for any displacement $\Delta \theta$ with $\| \Delta \theta \| = O ( \eta ^ { 2 } )$

$$
\begin{array} { r } { \mathcal { L } _ { E } ( \theta _ { \mathrm { r e f } } + \Delta \theta ) = \mathcal { L } _ { E } ( \theta _ { \mathrm { r e f } } ) + \langle g _ { E } ( \theta _ { \mathrm { r e f } } ) , \Delta \theta \rangle + \frac { 1 } { 2 } \langle \Delta \theta , H _ { E } \Delta \theta \rangle + O ( \| \Delta \theta \| ^ { 3 } ) , } \end{array}
$$

where the quadratic term is ${ \cal O } ( \eta ^ { 4 } )$ and the cubic remainder is ${ \cal O } ( \eta ^ { 6 } )$ . Substituting the pure-order displacements and absorbing the linear contribution of $r _ { A B } , r _ { B A }$ into $O ( \eta ^ { 3 } )$ gives

$$
\begin{array} { r } { \mathcal { L } _ { E } ( \theta _ { A B } ) = \mathcal { L } _ { E } ( \theta _ { \mathrm { r e f } } ) + \eta ^ { 2 } ( \mu + \frac { 1 } { 2 } \sigma ) + O ( \eta ^ { 3 } ) , } \end{array}
$$

$$
\begin{array} { r } { \mathcal { L } _ { E } \big ( \theta _ { B A } \big ) = \mathcal { L } _ { E } \big ( \theta _ { \mathrm { r e f } } \big ) + \eta ^ { 2 } ( \mu - \frac { 1 } { 2 } \sigma ) + O ( \eta ^ { 3 } ) . } \end{array}
$$

Therefore

$$
\begin{array} { r } { \operatorname* { m i n } \{ \mathcal { L } _ { E } ( \theta _ { A B } ) , \mathcal { L } _ { E } ( \theta _ { B A } ) \} = \mathcal { L } _ { E } ( \theta _ { \mathrm { r e f } } ) + \eta ^ { 2 } \mu - \frac { 1 } { 2 } \eta ^ { 2 } | \sigma | + O ( \eta ^ { 3 } ) . } \end{array}
$$

This min expansion does not require a separation condition on $| \sigma | ;$ : the map $( x , y ) \mapsto$ min $\{ x , y \}$ is 1- Lipschitz, so the $O ( \eta ^ { 3 } )$ endpoint remainders perturb the leading-order minimum by at most $\dot { O } ( \eta ^ { 3 } )$ A nonzero σ is only needed to define the sign-selected BO direction; in finite-sample experiments, a margin $| \sigma |$ above the estimator noise floor is the practical sign-stability condition. For the bracketonly point,

$$
\begin{array} { r } { \mathcal { L } _ { E } ( \theta _ { \mathrm { B O } } ) = \mathcal { L } _ { E } ( \theta _ { \mathrm { r e f } } ) - \frac { 1 } { 2 } \eta ^ { 2 } s \sigma + O ( \eta ^ { 4 } ) = \mathcal { L } _ { E } ( \theta _ { \mathrm { r e f } } ) - \frac { 1 } { 2 } \eta ^ { 2 } | \sigma | + O ( \eta ^ { 4 } ) , } \end{array}
$$

because $\mathrm { s i g n } ( \hat { \sigma } )$ is correct. Subtracting gives Equation 12. The $O ( \eta ^ { 3 } )$ residual is the Euler endpoint truncation that the bracket-only construction does not encode; it would improve to ${ \cal O } ( \eta ^ { 4 } )$ ) only against the truncated second-order pure endpoints (or for exactly quadratic losses). When $\mu > 0$ and η is small enough that the leading $- \eta ^ { 2 } \mu$ term governs the sign of the difference, $\theta _ { \mathrm { B O } }$ improves on the better pure order by $\eta ^ { 2 } \mu$ at leading order, and therefore improves on both pure orders. The empirical regression slope of 1.15 and 12/13 successful trials on the $\mu > 0$ subset (Section 5.6) at $\eta = 1 . 9 1 \cdot 1 \mathrm { { \bar { 0 } } ^ { - 3 } }$ indicate that this regime is realized in practice. □

Connection to the matched-batch endpoint edit. The DPO closure (Appendix G) and SFT iterative correction (Appendix AA) both apply the operation $\theta _ { A B } \mapsto \theta _ { A B } - \bar { \eta ^ { 2 } } b _ { A B }$ rather than the BO construction at $\theta _ { \mathrm { r e f } }$ . Direct expansion gives $\begin{array} { r } { \theta _ { A B } \dot { \bf \Phi } - \eta ^ { 2 } b _ { A B } = \theta _ { \mathrm { r e f } } + \eta ^ { 2 } ( \dot { c } _ { A B } - \frac 1 2 b _ { A B } ) + O ( \eta ^ { 3 } ) = } \end{array}$ $\theta _ { B A } + { \cal O } ( \eta ^ { 3 } )$ , recovering the reversed-order endpoint at $O ( \eta ^ { 3 } )$ . Proposition 1 is the complementary statement: at the drift-matched reference rather than at the trained $\theta _ { A B }$ , the sign-selected bracketonly step removes the symmetric target-loss drift and beats the better pure order by $\eta ^ { 2 } \mu$ at leading order. Both constructions exploit the same antisymmetric/symmetric decomposition; the matchedbatch DPO edit $\theta _ { A B } - \lambda \eta ^ { 2 } b _ { A B } ^ { - }$ at $\lambda { = } 1$ is the closure-targeting case $( \lambda { = } 1$ recovers $\theta _ { B A } )$ , while for $0 < \lambda < 1$ the edit lies, to leading order, between the two endpoints and keeps the symmetric $c _ { A B }$ term that the BO step omits. The non-monotone λ-sweep peaking at $\lambda { = } 1 \ ( \mathrm { A p p }$ . E) is therefore a signature of the second-order prescribed step rather than “any small move along $b _ { A B }$ helps.”

Theorem 1 (The order of two updates is identifiable from the endpoint). Let $\theta _ { 0 } \in \mathbb { R } ^ { d }$ be a public base model. Let $g _ { A } : = \nabla \mathcal { L } _ { A } ( \theta _ { 0 } )$ , g<sub>B</sub> $: = \nabla \mathcal { L } _ { B } ( \theta _ { 0 } )$ be one-step SGD gradients on candidate training-domain pair $( A , B )$ at $\theta _ { 0 }$ , and let $H _ { A } : = \nabla ^ { 2 } \mathcal { L } _ { A } ( \theta _ { 0 } ) , \bar { H _ { B } } : = \nabla ^ { 2 } \mathcal { L } _ { B } ( \theta _ { 0 } )$ be the corresponding Hessians. Let $\theta _ { \mathrm { q u e r y } } \in \{ \theta _ { A B } , \theta _ { B A } \}$ result from $k = 1$ SGD on either ordering with step size η. Define

$$
\begin{array} { r l } & { b _ { A B } : = H _ { B } g _ { A } - H _ { A } g _ { B } , \qquad c _ { A B } : = \frac 1 2 ( H _ { B } g _ { A } + H _ { A } g _ { B } ) , \qquad \theta _ { \mathrm { r e f } } : = \theta _ { 0 } - \eta ( g _ { A } + g _ { B } ) , } \\ & { \qquad s ( \theta ) : = \langle \theta - \theta _ { \mathrm { r e f } } , b _ { A B } \rangle , \qquad \mathrm { S C R } : = \frac { \frac 1 2 \| b _ { A B } \| ^ { 2 } } { | \langle c _ { A B } , b _ { A B } \rangle | } , } \end{array}
$$

where SCR is the ratio of the antisymmetric term to the symmetric one, with $\mathrm { S C R : = \infty }$ when the denominator is zero and $b _ { A B } \neq 0$ . Under the smoothness conditions ofSection 3 and the inequality $\mathrm { S C R } > 1$ (equivalently $\| b _ { A B } \| ^ { 2 } > 2 | \langle c _ { A B } , b _ { A B } \rangle | )$

$$
\mathrm { s i g n } \big ( s ( \theta _ { A B } ) \big ) = + 1 \qquad a n d \qquad \mathrm { s i g n } \big ( s ( \theta _ { B A } ) \big ) = - 1
$$

to leading order as $\eta  0$ . At finite η, the conclusion requires the displayed $O ( \eta ^ { 2 } )$ margins to dominate the $O ( \eta ^ { 3 } \lVert \dot { b _ { A B } } \rVert )$ remainders and estimator error.

Proof. By the second-order endpoint expansion derived above (eq. (5) and the symmetric companion),

$$
\begin{array} { r } { \theta _ { A B } = \theta _ { \mathrm { r e f } } + \eta ^ { 2 } ( c _ { A B } + \frac { 1 } { 2 } b _ { A B } ) + O ( \eta ^ { 3 } ) , \qquad \theta _ { B A } = \theta _ { \mathrm { r e f } } + \eta ^ { 2 } ( c _ { A B } - \frac { 1 } { 2 } b _ { A B } ) + O ( \eta ^ { 3 } ) . } \end{array}
$$

Substituting into s:

$$
\begin{array} { r l } & { s ( \theta _ { A B } ) = \eta ^ { 2 } \langle c _ { A B } + \frac { 1 } { 2 } b _ { A B } , b _ { A B } \rangle + O ( \eta ^ { 3 } \| b _ { A B } \| ) } \\ & { \qquad = \eta ^ { 2 } \left( \langle c _ { A B } , b _ { A B } \rangle + \frac { 1 } { 2 } \| b _ { A B } \| ^ { 2 } \right) + O ( \eta ^ { 3 } \| b _ { A B } \| ) , } \\ & { s ( \theta _ { B A } ) = \eta ^ { 2 } \left( \langle c _ { A B } , b _ { A B } \rangle - \frac { 1 } { 2 } \| b _ { A B } \| ^ { 2 } \right) + O ( \eta ^ { 3 } \| b _ { A B } \| ) . } \end{array}
$$

For sign $( s ( \theta _ { A B } ) ) = + 1$ at leading order we need $\begin{array} { r } { \langle c _ { A B } , b _ { A B } \rangle + \frac { 1 } { 2 } \| b _ { A B } \| ^ { 2 } > 0 } \end{array}$ . If $\langle c _ { A B } , b _ { A B } \rangle \geq$ 0, this is trivially satisfied since $\frac { 1 } { 2 } \left\| b _ { A B } \right\| ^ { 2 } \geq 0$ . If $\langle c _ { A B } , b _ { A B } \rangle \ < \ 0 .$ , the condition reduces to $\frac { 1 } { 2 } \| b _ { A B } \| ^ { 2 } > \left| \left. c _ { A B } , b _ { A B } \right. \right|$ , which is exactly $\mathrm { S C R } > 1$ . The argument for sign $( s ( \theta _ { B A } ) ) = - 1$ is symmetric. □

Corollary 1 (Paired-endpoint identifiability with drift cancellation). Under the smoothness $h y -$ potheses of Theorem 1 and given an unordered pair of alternate-order endpoints $\{ \theta ^ { ( 1 ) } , \theta ^ { ( 2 ) } \} \stackrel { \cdot } { = }$ $\left\{ \theta _ { A B } , \theta _ { B A } \right\}$ , define the antisymmetric statistic

$$
\Delta s _ { 1 2 } : = \langle \theta ^ { ( 1 ) } - \theta ^ { ( 2 ) } , b _ { A B } \rangle .\tag{13}
$$

$I f \theta ^ { ( 1 ) } = \theta _ { A B }$ and $\theta ^ { ( 2 ) } = \theta _ { B A }$ , then $\Delta s _ { 1 2 } = \eta ^ { 2 } \vert \vert b _ { A B } \vert \vert ^ { 2 } + O ( \eta ^ { 3 } \vert \vert b _ { A B } \vert \vert ) ,$ ; if the presentation is swapped, the sign is reversed. Thus the sign assigns the two endpoints whenever the leading term dominates thefinite-η and estimator remainders, with no SCR condition required.

Proof. By Lemma 1, $\theta _ { A B } - \theta _ { B A } = \eta ^ { 2 } b _ { A B } + O ( \eta ^ { 3 } )$ . Therefore $\langle \theta _ { A B } - \theta _ { B A } , b _ { A B } \rangle = \eta ^ { 2 } \lVert b _ { A B } \rVert ^ { 2 } +$ $O ( \eta ^ { \mathrm { 5 } } \lVert b _ { A B } \rVert )$ . Swapping the two endpoints multiplies the statistic by −1. The leading term is strictly positive when $b _ { A B } \neq 0$ □

When the per-query test fails. The per-query sign condition in Theorem 1 suffers from a finite-η failure mode: when the symmetric drift $\eta ^ { 2 } \bar { \langle } c _ { A B } , \bar { b _ { A B } } \rangle$ approaches $\eta ^ { 2 } \cdot \frac { 1 } { 2 } \lVert b _ { A B } \rVert ^ { 2 }$ in absolute value (i.e. SCR near 1), both $s ( \theta _ { A B } )$ and $s ( \theta _ { B A } )$ can lie on the same side of zero, breaking the per-query sign test. Corollary 1 removes this leading symmetric contribution in the paired subtraction, leaving $\eta ^ { \perp } \| b _ { A B } \| ^ { 2 }$ as the leading signal. Equivalently $\Delta s _ { 1 2 }$ projects the antisymmetric endpoint difference onto $b _ { A B }$ itself. We adopt $\mathrm { s i g n } ( \Delta s _ { 1 2 } )$ as the practical paired-endpoint assignment statistic.

Reference-point invariance of $\Delta s .$ The Trotter reference $\theta _ { \mathrm { r e f } } \ : = \ : \theta _ { 0 } \ : - \ : \eta ( g _ { A } \ : + \ : g _ { B } )$ enters the per-query score $s ( \theta ) = \langle \theta - \theta _ { \mathrm { r e f } } , b _ { A B } \rangle$ , but cancels in the paired subtraction: substituting $\theta _ { 0 }$ for $\theta _ { \mathrm { r e f } }$ in the per-query score subtracts the same $\eta \langle g _ { A } + g _ { B } , b _ { A B } \rangle$ from both $s ( \theta _ { A B } )$ and $s ( \theta _ { B A } )$ . The implementation may equivalently use $\theta _ { 0 }$ or $\theta _ { \mathrm { r e f } }$ for the paired statistic. This contrasts with the scalar σ predictor of Sweeney [2026a], a single inner product $\langle g _ { E } , b _ { A B } \rangle$ with no endpoint subtraction; evaluating its target gradient at the drift-matched $\theta _ { \mathrm { r e f } }$ rather than at $\theta _ { 0 }$ reduces the drift error from $O ( \eta ^ { 3 } )$ to $\mathrm { { \bar { \cal O } } } ( \eta ^ { 4 } )$ (their Remark 2.6).

Empirical regime. At finite $\eta$ the $O ( \eta ^ { 3 } \lVert b _ { A B } \rVert )$ remainder, sampling error, HVP error, and finitedifference error are non-negligible, so the per-query $\mathrm { S C R } > 1$ test is leading-order necessary but not sufficient on the 1B, 1.5B and 8B runs, where $\| \bar { b } _ { A B } \| ^ { 2 }$ is small relative to noise. The paired $\Delta s$ estimator identifies the order across model scales: combined accuracy $6 6 / 7 2 = 9 1 . 7 \%$ (Wilson 95% CI [83%, 96%]) over 72 pair-seed units across four models (Qwen-3-4B, Qwen-2.5-1.5B, Llama-3.2-1B, Llama-3.1-8B) and two architectures, with per-model rates 77.8%–100% (Appendix K). Both 4B and 8B units reach 100%; 1B-class accuracy is 77.8%–88.9%. Per-query SCR statistics: 72%–89% of units satisfy $\mathrm { S C R } > 1$ across runs. Replication across precision regimes (bf16 storage and full fp32) on the 1B and 1.5B models gives identical per-model sign-recovery rates (Table 10, App. K); the main count uses each model once.

Practical relevance and prior work. Theorem 1 requires only the public base model $\theta _ { 0 } ,$ , candidate training-domain pair $( A , B )$ and corresponding gradient/HVP access at $\theta _ { 0 }$ , and the trained query $\theta _ { \mathrm { q u e r y } }$ (the paired statistic of Corollary 1 needs both endpoints); no training-time signal is injected. This differs from adjacent settings: watermarking [Kirchenbauer et al., 2023, Zhao et al., 2024] embeds a signal during generation, membership-inference attacks [Shokri et al., 2017, Carlini et al., 2022] target individual training examples, and influence-function methods [Koh and Liang, 2017, Grosse et al., 2023, Park et al., 2023] estimate per-sample influence on predictions.

## B τ implementation: native-logit convention and finite differences

Native-logit convention. The aggregate $\sum _ { k } \tau _ { k }$ is invariant to position-wise constant shifts of $\delta z$ because $\dot { \sum _ { k } } e _ { k } = 0$ . Individual token attributions are reported in the model’s native logit coordinates. Thus $\tau _ { k }$ depends on the model’s logit parameterization; only the sum is shift-invariant.

Finite-difference $\delta z .$ The analytic identity in Eq. (1) is exact for the true JVP $J _ { \theta } z v$ . In experiments we compute an operational finite-difference readout

$$
\widetilde { \delta } z _ { \epsilon } ( v ) = \frac { z ( \theta _ { \mathrm { r e f } } + \epsilon v ) - z ( \theta _ { \mathrm { r e f } } - \epsilon v ) } { 2 \epsilon } ,
$$

with $\epsilon = 1 . 0$ for SFT and $\epsilon = 0 . 1$ in the corrected DPO identity checks. This is applied after computing $b _ { A B }$ via exact HVPs; only the logit-space JVP is finite-differenced. Consequently $\sum _ { k } \widetilde { \tau } _ { k , \epsilon }$ approximates the analytic JVP $\mathbb { E } [ e ( \theta _ { \mathrm { r e f } } ) ^ { \top } J _ { \theta } z v ]$ with $O ( \epsilon ^ { 2 } )$ logit-truncation error and converges to the analytic aggregate $\eta ^ { 2 } \sigma$ as $\epsilon  0$ for smooth logits. Individual token entries can vary with ϵ because finite differences integrate local curvature along the supplied displacement.

Choice of displacement v. The decomposition over k is exact for the displacement supplied to the linear readout. Using $v = \eta ^ { 2 } b _ { A B }$ gives the leading bracket readout used throughout; using $v = \theta _ { A B } - \theta _ { B A }$ would give an endpoint-displacement readout whose aggregate matches $\Delta _ { E } ( A , B )$ up to the loss-linearization remainder.

## C Precision policy for HVP computation

We distinguish model storage dtype, HVP compute dtype, the dtype in which the finite-difference perturbations $\theta _ { \mathrm { r e f } } \pm \epsilon \imath$ are formed, and loss/readout dtype. Bracket computations (double-backward HVPs for $H _ { B } g _ { A } - H _ { A } g _ { B } )$ use float32. For top-token rank and support claims on bf16-stored models, the perturbed parameters are formed in fp32 during the readout check. The choice of endto-end loss/intervention dtype is determined by whether the measured ordering gap is safely above the numerical resolution of the loss evaluation:

• Qwen-3-4B has an ordering gap of $\sim 7 \times 1 0 ^ { - 3 }$ in the reported intervention setting. We run the intervention with bf16 model storage and use fp32 HVPs and fp32 perturbed parameters for the bracket readouts.

• Qwen-2.5-1.5B has a much smaller fp32 gap of $\sim 1 0 ^ { - 4 }$ . We therefore run the intervention arm in full fp32 and report only the predefined above-floor subset (28/90 trials) in Table 3.

## D Matched-batch DPO correction: per-pair table and figure (Qwen-3-4B fp32)

Table 5: Matched-batch bracket-correction closure on the six original DPO source pairs (Qwen-3-4B fp32, $\beta _ { \mathrm { D P O } } { = } 0 . 1 , \eta { = } 2 \mathrm { e - 3 } ,$ 3 seeds $\times 1 0$ trials each). Closure is heavy-tailed near the $\mathrm { f p } 3 2$ noise floor; per-pair seed-cluster CIs are descriptive diagnostics of within-pair variability with $n _ { \mathrm { c l u s t e r s } } = 3 .$ not formal 95% inference. Columns: mean and median are the winsorized $5 / 9 5$ mean and the median closure of the $\lambda { = } 1$ edit; flat CI and cluster CI are per-trial and seed-clustered bootstrap 95% intervals of the mean; wrong-sign mean and random mean are the mean closures of the $+ \bar { \eta } ^ { 2 } b _ { A B }$ edit and of a norm-matched random edit. Pair abbreviations: fq=false\_qa, flan=flan\_v2\_p3, share=sharegpt, ultra=ultrachat.
<table><tr><td>Pair</td><td>mean</td><td>flat CI</td><td>cluster CI</td><td>median</td><td>wrong-sign mean</td><td>random mean</td></tr><tr><td>fq:flan</td><td>0.675</td><td>[0.561,0.767]</td><td>[0.425,0.772]</td><td>0.796</td><td>-1.083</td><td>+0.001</td></tr><tr><td>fq:share</td><td>0.454</td><td>[0.157,0.703]</td><td>[0.126, 0.536]</td><td>0.739</td><td>-1.104</td><td>+0.001</td></tr><tr><td>fq:ultra</td><td>0.647</td><td>[0.477,0.798]</td><td>[-0.386, 0.669]</td><td>0.778</td><td>-1.874</td><td>-0.001</td></tr><tr><td>flan:share</td><td>0.545</td><td>[0.247,0.776]</td><td>[-2.306, 0.825]</td><td>0.807</td><td>-1.915</td><td>-0.000</td></tr><tr><td>flan:ultra</td><td>0.720</td><td>[0.645,0.788]</td><td>[0.455,0.754]</td><td>0.785</td><td>-1.174</td><td>+0.001</td></tr><tr><td>ultra:share</td><td>0.454</td><td>[0.219,0.665]</td><td>[−0.107, 0.604]</td><td>0.735</td><td>-1.928</td><td>+0.003</td></tr></table>

Combined flat (180 trials): winsor mean = 0.588 [CI 0.507, 0.660], median = 0.770, raw = 0.318. Seed-cluster bootstrap (pairs fixed): winsor mean = 0.583 [CI 0.469, 0.676]. Seed-cluster strict-CI: 3/6.

## E Matched-batch DPO correction: λ-sweep and control-direction signs

The matched-batch correction protocol prescribes $\lambda = 1$ in the bracket subtraction $\theta _ { A B } - \lambda \cdot \eta ^ { 2 } b _ { A B }$ (Lemma 2.1 prediction; not tuned). We additionally report diagnostic $\lambda \in \{ 0 . 5 , 2 . 0 \}$ to test that the closure curve in λ is non-monotone with a peak at $\lambda = 1 { \mathrm { - } } 1 { \mathrm { h } }$ e signature of a second-order predicted step rather than a generic “any small move along $b _ { A B }$ helps” regime.

Across the six original pairs, per-pair median closures track the linearized prediction $C ( \lambda ) = 1 -$ $| 1 - \lambda |$ from $\bar { \theta _ { A B } } \ \bar { - \lambda \eta ^ { 2 } } b _ { A B } \ \bar { = } \ \hat { \theta _ { B A } } + ( 1 - \lambda ) \eta ^ { 2 } b _ { A B } + O ( \eta ^ { 3 } ) \colon \lambda = 0 . 5$ closes about half the gap $( 6 / 6$ pair medians positive, range $[ + 0 . 4 5 , + 0 . 5 5 ] ) ; \lambda = 1$ peaks at 0.78 (median of pair medians); $\lambda = 2$ falls back $\mathrm { t o } - 0 . 0 1$ on average. The symmetric peak at $\lambda { = } 1$ rules out a generic “any move along $b _ { A B }$ helps” alternative.

The control-direction signs in Table 5 are accordingly:

• Random direction (norm-matched): closure ≈ 0 on every pair (within $1 0 ^ { - 3 } )$ . Random Gaussian directions of the same Frobenius norm as $\eta ^ { 2 } b _ { A B }$ produce no systematic gap closure.

R79 matched-batch DPO bracket correction: closure of the AB-vs-BA gap Qwen-3-4B fp32, η = 2 × 10<sup>−3</sup>, 6 R71-pairs × 30 trials each  
![](images/1fcc31aae016cd3cc1191bd1738bffcb74125c2704fa58ec46199669eb289a49.jpg)  
Figure 3: Matched-batch DPO bracket-correction closure, per original pair. Bars are winsorized (5/95) means with seed-clustered bootstrap intervals (3 seed clusters per pair) shown as descriptive diagnostics of within-pair variability, not formal 95% inference. Random direction near zero; wrong-sign edit at or below the Lemma 2.1 prediction of −1 (red dashed line); first-order normmatched control underperforms the bracket.

• Wrong-sign edit $( \theta _ { A B } + \eta ^ { 2 } b _ { A B } ) \colon$ winsorized mean closure −1.21 across pairs, individualpair medians in $\left[ - 1 . 2 4 , - 0 . 8 9 \right]$ . The Lemma 2.1 prediction is exactly −1 (the gap roughly doubles); the pull of the raw means toward −1.5 comes from three pairs with raw means in $[ - 1 . 9 3 , - 1 . 8 7 ]$ , where the trained $\theta _ { A B }$ is far enough from $\theta _ { B A }$ that adding the bracket moves into a higher-curvature region of the DPO-loss surface than $\theta _ { B A }$ itself.

• First-order norm-matched $\displaystyle ( \eta ( g _ { A } - g _ { B } )$ rescaled to $\| \eta ^ { 2 } b _ { A B } \| ) \rangle$ : winsorized mean −0.93, uniformly worse than bracket. First-order gradient differences are not the AB-vs-BA control variable, even when scaled to bracket magnitude.

## F DPO closure-ratio statistical convention

The closure metric of the matched-batch DPO correction $1 - | \Delta _ { \mathrm { e d i t e d } } | / | \Delta _ { \mathrm { b a s e l i n e } } |$ is a ratio that inherits the same heavy-tail behavior as the SFT closure C in Table 3: when a trial’s baseline gap is near the fp32 DPO-loss noise floor of $2 \times 1 0 ^ { - 5 }$ , the denominator is small and individual trials are dominated by noise. To prevent the primary statistic from being dominated by ∼ 10% of trials at the noise floor, the SFT intervention runs already use winsorized 5/95 means with bootstrap CI. For example, the Qwen-3-4B winsorized mean in Table 3 is +31.2% (median +32.0%), while the raw mean over the same 90 trials is +20.4% (the same ratio-tail effect at the SFT scale).

We apply the same effect-size convention to the DPO correction: winsorized 5/95 mean as the primary point estimate, with the median and raw mean reported alongside. Because each DPO pair has only three seed-index clusters, per-pair seed-cluster bootstrap intervals are descriptive diagnostics of within-pair variability rather than formal 95% inference. The source-pair summary is the median paired bracket-minus-random closure difference: all 12/12 pair medians are positive across the original and fresh-pair grids, while the flat-bootstrap strict-CI criterion holds for 11/12 source pairs. The pairs share source domains, so this sign agreement is descriptive, not a significance test. A seed-level check on the 36 pair-seed cells gives 36/36 positive bracket-minus-random medians, but the seed cluster remains nested in pair so this is also descriptive. Flat bootstrap CIs are reported only as a comparison with the SFT convention. None of the qualitative cross-pair rankings (bracket ≫ random ≈ 0; wrong-sign ≈ −1; first-order < bracket) depend on the choice of statistic.

## G DPO extended methodology and per-pair detail

DPO loss derivative and corrected τ. The DPO loss $L = - \log s ( \beta _ { \mathrm { D P O } } m _ { i } )$ , with $s ( x ) : = 1 / ( 1 +$ $e ^ { - x } )$ and margin $m _ { i } = ( \log p _ { c } - \log p _ { c } ^ { \mathrm { r e f } } ) - ( \log p _ { r } - \log p _ { r } ^ { \mathrm { r e f } } )$ ), compresses per-token effects through a scalar; the per-token DPO derivative is

$$
\begin{array} { r l } & { \tau _ { k } = \displaystyle \sum _ { i } w _ { i } \left[ \sum _ { t \in C _ { i } } \left( p _ { i , t } ^ { c } [ k ] - \mathbf { 1 } _ { [ c _ { t } = k ] } \right) \delta z _ { i , t } ^ { c } [ k ] \right. } \\ & { \qquad \quad \left. - \sum _ { t \in R _ { i } } \left( p _ { i , t } ^ { r } [ k ] - \mathbf { 1 } _ { [ r _ { t } = k ] } \right) \delta z _ { i , t } ^ { r } [ k ] \right] , } \\ & { w _ { i } = \displaystyle \beta _ { \mathrm { D P O } } s ( - \beta _ { \mathrm { D P O } } m _ { i } ) . } \end{array}
$$

restricted to completion positions $C _ { i } , R _ { i }$ and weighted by the per-example DPO factor $w _ { i }$ . The token-reweighting DPO-loss attributions use this derivative; Table 11 instead reads DPO-derived brackets through the cross-entropy readout on Pile evaluation text. The algebraic identity $\sum _ { k } \tau _ { k } =$ $d L _ { \mathrm { D P O \_ s u m } } / d \theta$ · d holds at relative error below $1 0 ^ { - 2 }$ (median $2 \times 1 0 ^ { - 4 }$ , max $4 \times 1 0 ^ { - 3 }$ across the 6 original pairs; Appendix H table).

DPO token reweighting: detailed table on Qwen-3-4B fp32. Table 6 lists the per-pair results.

Table 6: DPO token reweighting with the corrected τ on the six original pairs (Qwen-3-4B fp32, $\eta { = } 2 \mathrm { e } { \mathrm { - } } 3 , \alpha { = } 0 . 0 5 .$ , the top- $\cdot 1 0 0 \left| \tau \right|$ tokens whose sign matches a probe measurement of the baseline gap, 3 seeds × 20 trials). Pair abbreviations as in Table 5. Paired differenc $\dot { \mathbf { \eta } } = | \Delta _ { \mathrm { h a r m f u l } } | - | \Delta _ { \mathrm { r a n d o m } } | ;$ flat diagnostic: whether its per-trial bootstrap CI excludes zero; direction: mitigation if the harmful edit shrinks the gap.
<table><tr><td>Pair</td><td>paired difference (mean)</td><td>CI 95%</td><td>flat diagnostic</td><td>direction</td></tr><tr><td>fq:flan</td><td>-1.15e-3</td><td>[-1.93e-3, -4.30e-4]</td><td>positive</td><td>mitigation</td></tr><tr><td>fq:share</td><td>negative</td><td>excludes 0</td><td>positive</td><td>mitigation</td></tr><tr><td>fq:ultra</td><td>negative</td><td>excludes 0</td><td>positive</td><td>mitigation</td></tr><tr><td>flan:share</td><td></td><td>includes 0</td><td>inconclusive</td><td>no effect</td></tr><tr><td>flan:ultra</td><td>negative</td><td>excludes 0</td><td>positive</td><td>mitigation</td></tr><tr><td>ultra:share</td><td>negative</td><td>excludes 0</td><td>positive</td><td>mitigation</td></tr></table>

Summary: 5/6 pairs have a per-trial CI excluding zero; all 6 pass the protocol checks (probe gap above its magnitude threshold, equal total loss-weight change for the harmful and random sets, and $\sum _ { k } \tau _ { k }$ matching the measured DPO loss change).

DPO token reweighting on Qwen-2.5-1.5B fp32. At $\begin{array} { r l r } { \eta } & { { } = } & { 2 . 0 3 \mathrm { { e } - 3 } } \end{array}$ on Qwen-2.5-1.5B fp32, the token-reweighting test has 4/6 flat descriptive positives (false\_qa\_vs\_sharegpt, false\_qa\_vs\_ultrachat, flan\_v2\_p3\_vs\_sharegpt, ultrachat\_vs\_sharegpt). We report this Qwen-2.5-1.5B arm as descriptive.

Calibration discipline: η as a BCH-locality condition. Lemma 2.1 of Sweeney [2026a] is accurate when the $\bar { O } ( \eta ^ { 3 } )$ endpoint remainder is small relative to the leading $\eta ^ { 2 } \lVert b _ { A B } \rVert$ bracket displacement, and when that bracket displacement remains subordinate to the shared first-order update. Heuristics that scale η proportionally across model sizes (e.g., extrapolating an SFT-to-DPO ratio observed on one model to a different model) can violate this locality window and produce incoherent control signs. We use Sweeney [2026a]’s σ-correctness check as the operational diagnostic: η is BCH-local if $\sigma = \langle g _ { E } , b _ { A B } \rangle$ predicts the sign of the held-out AB-vs-BA loss gap with high accuracy and wrong-sign controls behave according to local antisymmetry. For the DPO arms, the same recipe gives $\eta = 2 . 0 3 \mathrm { { e } { - } 3 }$ on Qwen-2.5-1.5B fp32 and 1.05e-3 on Llama-3.2-1B fp32 (on SFT probe units, sign σ matches the measured gap sign in 96/96 and 199/204), larger than the SFT values in Table 17. The calibrated $\eta$ is specific to the model and objective, not a universal cross-model constant.

Token reweighting vs. parameter correction: intervention design, not readout correctness. The two DPO tests share the same $b _ { A B }$ but differ in how they intervene: token reweighting acts indirectly through SGD-time loss reweighting on top-|τ | tokens (token reweight → per-position residual → scalar margin → logistic-saturated derivative); the matched-batch correction acts directly as a one-step parameter update. Direct magnitude comparison between the reweighting test’s paired difference and the correction’s closure ratio is not meaningful; what is comparable is the bracket-vsrandom separation within each protocol. Together they support the limited claim that the per-token $\tau _ { k }$ is a readout of a parameter-level direction $b _ { A B }$ , and that the direct matched-batch update along that direction controls the aggregate AB-vs-BA loss gap.

Cross-pair summary. All 12/12 source-pair medians are positive; 11/12 pass the flatbootstrap strict-CI criterion; the across-pair seed-clustered aggregate is winsorized mean 0.583 [CI 0.469, 0.676] over $n = 6$ original pairs. The wrong-sign cross-pair pull to $[ - 1 . 9 , - 1 . 2 ]$ on three pairs (vs. predicted −1) reflects $\theta _ { A B }$ being far enough from $\theta _ { B A }$ that adding $\eta ^ { 2 } b$ <sub>AB</sub> moves into a higher-curvature region than $\theta _ { B A }$ itself. The qualitative ranking bracket ≫ random $\approx 0 .$ , wrong-sign $\approx - 1$ , first-order < bracket holds on every pair.

## H DPO τ token-sum identity check (original pairs, per pair)

The corrected DPO τ derivation predicts that $\sum _ { k } \tau _ { k }$ should equal the directly-measured $\Delta L _ { \mathrm { D P O } } ^ { \mathrm { F D } }$ on the same DPO batches at the finite-difference (FD) step $\varepsilon = 0 . 1$ in the direction $v = \eta ^ { 2 } b _ { A B }$ The token-reweighting protocol requires this token-sum identity check to pass at a 5% relative-error tolerance; values reported here are well below tolerance (median $2 \times 1 0 ^ { - 4 }$ , max $4 \times 1 0 ^ { - 3 } )$ .

Table 7: DPO τ identity check on the original pairs. Each pair: $\sum _ { k } \tau _ { k }$ computed by finite-difference attribution at $\varepsilon = 0 . 1 , v = \eta ^ { 2 } b _ { A B }$ , with $\eta = 2 \times 1 0 ^ { - 3 }$ (original-pairs protocol); compared to the directly-measured $\Delta L _ { \mathrm { D P O } } ^ { \mathrm { F D } }$ on the same DPO batches.
<table><tr><td>Pair</td><td> $\sum _ { k } \tau _ { k }$ </td><td> $\Delta L _ { \mathrm { D P O } } ^ { \mathrm { F D } }$ </td><td>rel. err.</td></tr><tr><td rowspan="5">false_qa_vs_flan_v2_p3 false_qa_vs_sharegpt false_qa_vs_ultrachat flan_v2_p3_vs_sharegpt flan_v2_p3_vs_ultrachat ultrachat_vs_sharegpt</td><td>-1.247e-1</td><td>-1.247e-1</td><td>5.9e-5</td></tr><tr><td>-9.150e-2</td><td>-9.149e-2</td><td>2.6e-5</td></tr><tr><td>-1.805e-1</td><td>-1.804e-1</td><td>1.4e-4</td></tr><tr><td>-6.669e-2</td><td>-6.670e-2</td><td>2.4e-4</td></tr><tr><td>+1.900e-3 -3.304e-2</td><td>+1.907e-3 -3.307e-2</td><td>3.6e-3 8.9e-4</td></tr></table>

## I Per-pair endpoint-correction closure (Qwen-3-4B, k = 1)

Per-domain-pair closure for the single-step bracket correction of the trained endpoint; aggregate trends are summarized in Section 5.6. The signal of interest is the dependence on $| \Delta _ { \mathrm { b a s e l i n e } } | :$ large-gap pairs (math-vs-legal, news-vs-legal) have positive closure; near-noise-floor pairs (newsvs-biomedical, 0.001; code-vs-biomedical, 0.002) overshoot or invert.

Table 8: Endpoint-correction closure by domain pair at $k = 1 ( { \mathrm { Q w e n } } { \mathrm { - } } 3 { \mathrm { - } } 4 \mathrm { B } , \lambda = 1 . 0$ , averaged over 3 seeds). $\Delta \bar { W } _ { k }$ is output-layer row k of $\theta _ { A B } - \theta _ { B A }$
<table><tr><td>Pair</td><td> $\left| \Delta _ { \mathrm { b a s e l i n e } } \right|$ </td><td>Single-step closure  $C \left( \% \right)$ </td><td>Spearman(τk,  $\| \Delta W _ { k } \| )$ </td></tr><tr><td>math-vs-legal</td><td>0.042</td><td>+80.3%</td><td>0.833</td></tr><tr><td>code-vs-legal</td><td>0.013</td><td>+28.4%</td><td>0.800</td></tr><tr><td>news-vs-legal</td><td>0.037</td><td>+23.9%</td><td>0.830</td></tr><tr><td>code-vs-news</td><td>0.004</td><td>+5.9%</td><td>0.832</td></tr><tr><td>news-vs-biomedical</td><td>0.001</td><td>-32.9%</td><td>0.844</td></tr><tr><td>code-vs-biomedical</td><td>0.002</td><td>-51.1%</td><td>0.821</td></tr></table>

## J Token-level ordering prediction: support recovery and rank-correlation discussion

A complementary test of the bracket attribution is whether $\tau _ { k } .$ , computed from the base model before either order is run, predicts which specific tokens will be most affected by swapping the two updates. For each of 15 domain pairs × 3 seeds (45 experiments per model), we compare bracket-derived $\tau _ { k }$ to the empirical per-token ordering effect $\Delta \hat { \bf N L L } _ { k } = \hat { \bf N L L } _ { k } ( \theta _ { A B } ) - \hat { \bf N L L } _ { k } ^ { \top } ( \theta _ { B A } )$ measured after actual SGD. Because the vocabulary distribution is extremely sparse, the operational statistic is high-mass support recovery rather than full-vocabulary rank correlation.

Table 9: Token-level ordering prediction as support recovery (45 experiments per model, 15 domain pairs × 3 seeds). Top-1% and top-5% are mean overlaps of the predicted and empirical high-effect token sets; equal-size-set baselines are 1% and 5%. Full-vocabulary Spearman is retained as a diagnostic, not the main metric for the heavy-tailed Qwen-3-4B readout.
<table><tr><td>Model</td><td>Dtypes</td><td>Top-1%</td><td></td><td>Top-5%</td><td></td><td>ρ</td><td></td><td>Sign</td></tr><tr><td>Qwen-3-4B</td><td>bf16/fp32 FD</td><td>53.0 med.)%</td><td>(54.5</td><td>51.0 med.)%</td><td>(54.3</td><td>0.033 0.051</td><td> $\pm \_ { 5 2 . 4 \% }$ </td><td></td></tr><tr><td>Qwen-2.5-1.5B</td><td>fp32/fp32 FD</td><td>57.3 med.)%</td><td>(57.1</td><td>46.8 med.)%</td><td>(47.5</td><td>0.521 0.034</td><td> $\pm \quad 7 0 . 6 \pm$ </td><td>1.9%</td></tr></table>

On Qwen-3-4B, full-rank Spearman is uninformative under sharp concentration: the long tail contains many near-zero coordinates whose ordering is dominated by numerical and sampling noise. The bracket nevertheless recovers the empirical high-effect support: predicted top-1% tokens recover 53% of the empirical top-1% set on average (54.5% median) versus a 1% equal-size-set baseline, and the top-5% overlap is 51% on average versus a 5% equal-size-set baseline. On Qwen-2.5- 1.5B fp32, where the empirical endpoint signal has a lower numerical noise floor, full-vocabulary Spearman is also informative $( \rho = 0 . 5 2 )$ and agrees with the support-recovery view. The separate parameter-row test summarized in Section 5.6 gives per-pair $\rho = 0 . 7 8 – 0 . 9 0$ without this vocabularytail rank degradation, since parameter row norms are not heavy-tailed in the same way as per-token $\tau _ { k }$

## K Paired-endpoint training-order readability

The bracket direction $b _ { A B }$ used in this appendix is computed at $\theta _ { 0 }$ on an evaluation split shared across pairs, from a follow-up analysis, distinct from the pair-specific Trotter-reference per-token interpretability readout used in Section 5.2. The two protocols differ in eval-batch composition and reference point $( \theta _ { 0 }$ vs $\theta _ { \mathrm { r e f } } )$ , so the corresponding $\tau$ vectors are not directly comparable token-bytoken.

Given a public base $\theta _ { 0 }$ , a query model $\theta _ { \mathrm { q u e r y } } .$ , and candidate training-domain pairs $( A , B )$ , we ask: can we tell from $\theta _ { \mathrm { q u e r y } } \mathrm { \hat { s } }$ weights whether it was trained $A  B$ or $B {  } A { \because }$ Define the Trotter reference $\theta _ { \mathrm { r e f } } : = \theta _ { 0 } - \eta ( g _ { A } \mathrm { \tiny ~ \dot { + } ~ } \mathrm { \dot { g } } _ { B } )$ and the signed score

$$
s ( \theta _ { \mathrm { q u e r y } } ) : = \langle \theta _ { \mathrm { q u e r y } } - \theta _ { \mathrm { r e f } } , \ b _ { A B } \rangle .\tag{14}
$$

By Theorem $1 \left( \mathrm { A p p . A } \right)$ , sign $( s ( \theta _ { A B } ) ) = + 1$ and sign $( s ( \theta _ { B A } ) ) = - 1$ at leading order whenever the antisymmetric commutator $b _ { A B }$ dominates the symmetric Trotter term $c _ { A B } : = \overset { \triangledown } { \operatorname { 1 } } ( H _ { B } g _ { A } + H _ { A } g _ { B } )$ Concretely, the ratio of the antisymmetric signal to the symmetric term

$$
\begin{array} { r } { \mathrm { S C R } : = \frac { 1 } { 2 } \| b _ { A B } \| _ { 2 } ^ { 2 } / | \langle c _ { A B } , b _ { A B } \rangle | , } \end{array}\tag{15}
$$

so that $\mathrm { S C R } > 1$ corresponds to the theoretical inequality $\| b _ { A B } \| _ { 2 } ^ { 2 } > 2 | \langle c _ { A B } , b _ { A B } \rangle | $

Cross-model results (∆s estimator). Per-pair-seed unit (6 domain pairs $\times 3$ seeds = 18 units per run), the drift-cancelling antisymmetric statistic $\Delta s = \langle \theta _ { A B } - \theta _ { B A } , b _ { A B } \rangle ( \mathbf { e q . } 1 3 )$ gives:

Why the antisymmetric statistic and not per-query sign $. ( s ( \theta ) )$ . The original per-query test $\mathrm { s i g n } ( s ( \theta _ { A B } ) ) = + 1 , \mathrm { s i g n } ( s ( \theta _ { B A } ) ) = - 1$ is the leading-order condition of Theorem 1, but suffers from a finite-η failure: when the symmetric Trotter drift contribution $\eta ^ { 2 } \langle c _ { A B } , b _ { A B } \rangle$ is comparable in magnitude to the antisymmetric $\eta ^ { 2 } \cdot { \textstyle \frac { 1 } { 2 } } \| b _ { A B } \| ^ { 2 } \left( \mathrm { i . e . } \mathrm { S C R } \approx 1 \right)$ , both $s ( \theta _ { A B } )$ and $s ( \theta _ { B A } )$ can lie on the same side of zero. On the 1B-class and 1.5B-class runs and the 8B Llama-3.1 run, $1 7 / 1 8 – 1 8 / 1 8$ pair-seed units have $s ( \theta _ { A B } )$ and $s ( \theta _ { B A } )$ matching in sign, mechanically pinning per-query sign accuracy to $5 0 \% ;$ on Qwen-3-4B run (Qwen-3-4B) only $8 / 1 8$ are sign-aligned, giving per-query accuracy of $7 7 . 8 \%$ . The antisymmetric $\Delta s$ removes the shared drift terms in the paired subtraction and identifies the order at $\geq 7 7 . 8 \%$ on all six runs.

<table><tr><td>Model</td><td>Scale</td><td>sign(∆s) correct</td><td>Wilson 95% CI</td></tr><tr><td>Llama-3.2-1B (bf16 storage)</td><td>1B</td><td> $1 4 / 1 8 = 7 7 . 8 \%$ </td><td>[55%, 91%]</td></tr><tr><td>Llama-3.2-1B (fp32)</td><td>1B</td><td> $1 4 / 1 8 = 7 7 . 8 \%$ </td><td>[55%, 91%]</td></tr><tr><td>Qwen-2.5-1.5B (fp32)</td><td>1.5B</td><td> $1 6 / 1 8 = 8 8 . 9 \%$ </td><td>[67%, 97%]</td></tr><tr><td>Qwen-2.5-1.5B (bf16 storage)</td><td>1.5B</td><td> $1 6 / 1 8 = 8 8 . 9 \%$ </td><td>[67%, 97%]</td></tr><tr><td>Qwen-3-4B (bf16 storage)</td><td>4B</td><td> $1 8 / 1 8 = 1 0 0 . 0 \%$ </td><td>[82%, 100%]</td></tr><tr><td>Llama-3.1-8B (bf16 storage)</td><td>8B</td><td> $1 8 / 1 8 = 1 0 0 . 0 \%$ </td><td>[82%, 100%]</td></tr><tr><td>Combined (both precisions)</td><td>4 models, 2 architectures, 4 scales</td><td> $9 6 / 1 0 8 = 8 8 . 9 \%$ </td><td>[81.6%, 93.5%]</td></tr></table>

Table 10: Paired-endpoint training-order assignment via the $\mathrm { s i g n } ( \Delta s )$ estimator. Six runs across four unique models, with the 1B and 1.5B models replicated under both fp32 and bf16-storage precision regimes (identical sign-recovery in each). Counting each model once gives $6 6 / 7 2 =$ 91.7% (≥ 4B-class 100%, 1B-class $7 7 . { \dot { 8 \% } } - 8 8 . 9 \% ) ;$ ; the dual-precision count $9 6 / 1 0 8 \ : = \ : 8 8 . 9 \%$ shown here is a precision-replication check.

Per-query SCR satisfaction. Of 18 (pair, seed) units per run, the Theorem 1 inequality SCR $> 1$ is satisfied at: 16/18 (Qwen-3-4B), 14/18–15/18 (1B-class runs), 13/18 (Llama-3.1-8B). On the Qwen-3-4B run, $\dot { 7 } / 1 8$ units have $\mathrm { S C R } \ge 2$ , and all of them pass the per-query test.

Pair identification (six-candidate task, Qwen-3-4B run only). Cosine score between $\theta _ { \mathrm { q u e r y } } - \theta _ { \mathrm { r e f } }$ and each candidate $b _ { A B }$ for pair identification among six candidates: $1 1 / 3 6$ per-query joint-correct (30.6% vs. 16.7% chance; 1.83×). The pair-identification subtask is harder than sign recovery and uses an absolute (not antisymmetric) score; it is reported as a directional indicator only.

Comparison to prior work. Watermarking [Kirchenbauer et al., 2023, Zhao et al., 2024] embeds a signal during generation; membership-inference attacks [Shokri et al., 2017, Carlini et al., 2022] target individual training examples; influence-functions / TRAK [Koh and Liang, 2017, Grosse et al., 2023, Park et al., 2023] estimate per-sample influence on predictions. Paired-endpoint readability follows from the antisymmetry of the bracket $b _ { A B }$ on unmodified weights.

## L DPO commutator memory: full protocol and per-pair table

Setup. For a frozen reference model $\pi _ { \mathrm { r e f } }$ and a policy $\pi _ { \theta } .$ , DPO [Rafailov et al., 2023] optimizes the margin $\beta _ { \mathrm { D P O } }$ · [(log $\pi _ { \theta } ( c | p ) - \log \pi _ { \mathrm { r e f } } ( c | p ) ) - ( \log \pi _ { \theta } ( r | p ) - \log \pi _ { \mathrm { r e f } } ( r | p ) ) ]$ ] on (prompt, chosen, rejected) triples. The chosen-vs-rejected split alone is degenerate for the bracket: both terms appear in the same loss with a fixed coefficient, not as two independent gradient sources. We therefore use preference-data sources as the $A , B$ domains: the commutator $b _ { A B } = H _ { B } g _ { A } - H _ { A } g _ { B }$ is computed from DPO gradients on preference pairs drawn from UltraFeedback Cui et al. [2024] grouped by source (e.g., false\_qa vs. flan $\mathtt { - } \mathtt { v } 2 \mathtt { \_ p 3 }$ , ultrachat vs. sharegpt). This preserves the two-independent-sources semantics of the SFT protocol while moving the loss into the preference regime. We project the bracket to vocabulary space via the same finite-difference $\tau _ { k }$ procedure at the Trotter reference, using a Pile-domain eval set (code, news, legal, math) shared with the SFT runs so that the resulting $\tau$ vectors live in a comparable logit subspace. The parameter subspace is lastlayer $\mathsf { o \_ p r o j }$ and $\mathtt { d o w n \_ p r o j }$ (the attention output projection and MLP down-projection; matching the template used for the scalar ordering-quality predictor in Sweeney [2026a]); magnitudes are therefore not directly comparable between SFT and DPO.

Scale pattern. Going $\mathrm { Q w e n } { - 2 . 5 { - } 1 . 5 \mathrm { B } } \to \mathrm { Q w e n } { - } 3 { - } 4 \mathrm { B } \to \mathrm { Q w e n } { - } 3 { - } 8 \mathrm { B }$ , “pct for $8 0 \%$ moves $1 . 1 6 \%  0 . 0 0 4 \%  0 . 0 1 7 \% - \mathrm { s h a r p e n i n g }$ between the 1.5B and 4B checkpoints, then easing from 4B to 8B. The 1.5B-to-4B comparison crosses both a scale boundary and a pre-training-procedure boundary (Qwen-2.5 vs. Qwen-3 families), so we cannot separate scale from pre-training as the driver; these rows show that concentration recurs across the tested 1.5B–8B DPO checkpoints, not that support specificity has been established for DPO.

Table 11: Per-pair DPO commutator memory concentration. “pct for $8 0 \%$ = fraction of vocabulary required to account for 80% of |τ| mass. These descriptive concentration rows do not include the random-direction support nulls used for the SFT specificity claim.
<table><tr><td>Model</td><td>Pair</td><td>Gini</td><td>pct for 80%</td><td>pct for 99%</td></tr><tr><td>Qwen-2.5-1.5B (fp32)</td><td>false_qa vs. flan_v2_p3</td><td>0.960</td><td>1.26%</td><td>30.1%</td></tr><tr><td>Qwen-2.5-1.5B (fp32)</td><td>ultrachat vs. sharegpt</td><td>0.969</td><td>1.05%</td><td>25.1%</td></tr><tr><td>Qwen-3-4B (bf16)</td><td>false_qa vs. flan_v2_p3</td><td>0.999</td><td>0.003%</td><td>0.70%</td></tr><tr><td>Qwen-3-4B (bf16)</td><td>ultrachat vs. sharegpt</td><td>0.998</td><td>0.006%</td><td>0.97%</td></tr><tr><td>Qwen-3-4B (bf16)</td><td>false_qa vs. ultrachat</td><td>1.000</td><td>0.002%</td><td>0.08%</td></tr><tr><td>Qwen-3-8B (bf16)</td><td>false_qa vs. flan_v2_p3</td><td>0.997</td><td>0.019%</td><td>2.90%</td></tr><tr><td>Qwen-3-8B (bf16)</td><td>ultrachat vs. sharegpt</td><td>0.998</td><td>0.016%</td><td>1.55%</td></tr><tr><td>Qwen-3-8B (bf16)</td><td>false_qa vs. ultrachat</td><td>0.997</td><td>0.016%</td><td>2.69%</td></tr></table>

## M DPO token reweighting on fresh pairs

The planned fresh-pair arm of the token-reweighting test had three evaluable pairs under the released UltraFeedback source taxonomy. The available pairs ran with the full token-reweighting protocol and are reported as a partial replication; the main fresh-pair evidence is the six-pair replication of the matched-batch correction in Appendix N.

Table 12: Token-reweighting results on the three evaluable fresh pairs (Qwen-3-4B fp32, same protocol as the original pairs).
<table><tr><td>Pair</td><td>CI excludes 0</td><td>paired difference CI 95%</td><td>direction</td></tr><tr><td>evol_instruct_vs_truthful_qa</td><td>yes</td><td>excludes 0</td><td>mitigation</td></tr><tr><td>evol_instruct_vs_false_qa</td><td>no</td><td>includes 0</td><td>no effect</td></tr><tr><td>truthful_qa_vs_flan_v2_p3</td><td>yes</td><td>excludes 0</td><td>mitigation</td></tr></table>

We treat this three-pair arm as descriptive; the confirmatory fresh-pair evidence is the six-pair matched-batch replication below.

## N Matched-batch DPO correction on fresh pairs

To test dependence on the pair list, we re-run the matched-batch correction protocol unchanged on a six-pair list disjoint from the original six. The fresh list crosses two new sources (evol\_instruct, truthful\_qa) with the four original sources available in the released snapshot. Hyperparameters, seed schedule, and the matched-batch protocol are identical to the original pairs (Qwen-3-4B fp32, β<sub>DPO</sub>=0.1, η = 2e−3, 3 seeds × 10 trials per pair).

Combined with original pairs (161/180 bracket > random), 316/360 = 87.8% matched trials show bracket > random across both pair sets, with all 12/12 pair-level medians of the bracket-minusrandom closure difference positive. Excluding the two trials below the per-pair baseline noise floor leaves 316/358, so the 12/12 result is not driven by near-zero denominator cases. The control hierarchy (random ≈ 0, wrong-sign ≈ −1, first-order widening the gap) reproduces uniformly under the fixed pair list and edit-size rule.

## O Seed-clustered bootstrap diagnostics for the DPO correction

This appendix reports per-pair seed-clustered bootstrap intervals as descriptive diagnostics of within-pair variability. Because each pair has only $n _ { \mathrm { c l u s t e r s } } = 3$ seed-index clusters, the resampling distribution is too coarse to support these intervals as formal 95% inference. The 11/12 count reported in the main text uses flat per-trial bootstrap intervals $( 6 / 6$ original pairs and $5 / 6$ fresh pairs) and serves as a check rather than formal inference; under the seed-clustered bootstrap, 3/6 original-pair intervals exclude zero (Table 14). The source-pair summary is the median bracketminus-random closure difference: all 12/12 pair medians are positive across the original and freshpair grids. The seed-clustered bootstrap (3 clusters per pair, concatenated 10 trials each) is reported here alongside the flat per-trial bootstrap to show that the cross-pair effect ranking does not depend on the choice of within-pair variability estimator.

Table 13: Matched-batch bracket-correction closure on six disjoint fresh UltraFeedback pairs (Qwen-3-4B fp32, 3 seeds × 10 trials each, $b _ { A B }$ recomputed per trial). Pair abbreviations: evol=evol\_instruct, truth=truthful\_qa; other abbreviations as in Table 5. All six pair me dians are positive in [0.71, 0.85].
<table><tr><td>Pair</td><td>bracket mean</td><td>median</td><td>random</td><td>wrong-sign</td><td>first-order</td></tr><tr><td>evol:fq</td><td>+0.436</td><td>+0.775</td><td>+0.001</td><td>-1.085</td><td>-0.573</td></tr><tr><td>evol:flan</td><td>+0.375</td><td>+0.774</td><td>+0.001</td><td>-1.059</td><td>-2.004</td></tr><tr><td>evol:share</td><td>+0.613</td><td>+0.802</td><td>-0.001</td><td>-0.904</td><td>-2.252</td></tr><tr><td>truth:flan</td><td>+0.378</td><td>+0.849</td><td>-0.000</td><td>-0.910</td><td>-0.132</td></tr><tr><td>truth:share</td><td>+0.501</td><td>+0.726</td><td>-0.001</td><td>-1.280</td><td>-0.956</td></tr><tr><td>truth:ultra</td><td>+0.509</td><td>+0.714</td><td>-0.001</td><td>-1.331</td><td>-0.805</td></tr></table>

Pooled (180 fresh trials): bracket mean +0.408, median +0.778; bracket > random in $1 5 5 / 1 8 0$ (86.1%); 6/6 pair medians positive.

Table 14: Original-pairs winsorized closure of the matched-batch DPO correction: flat bootstrap (1000 iter on the 30 trials) vs seed-clustered bootstrap (1000 iter cluster-resampling 3 seeds, then aggregating their 10 trials each). Both columns are descriptive diagnostics of within-pair variability; with $n _ { \mathrm { c l u s t e r s } } = 3$ per pair the seed-cluster percentile is too coarse for formal 95% inference. Bold rows have intervals strictly above zero under the corresponding estimator (descriptive, not a decision criterion).
<table><tr><td>Pair</td><td>flat-bootstrap CI 95%</td><td>seed-cluster CI 95%</td></tr><tr><td>false_qa_vs_flan_v2_p3</td><td>[0.561, 0.767]</td><td>[0.425, 0.772]</td></tr><tr><td>false_qa_vs_sharegpt</td><td>[0.157, 0.703]</td><td>[0.126, 0.536]</td></tr><tr><td>false_qa_vs_ultrachat</td><td>[0.477, 0.798]</td><td>[-0.386, 0.669]</td></tr><tr><td>flan_v2_p3_vs_sharegpt</td><td>[0.247, 0.776]</td><td>[-2.306, 0.825]</td></tr><tr><td>flan_v2_p3_vs_ultrachat</td><td>[0.645, 0.788]</td><td>[0.455, 0.754]</td></tr><tr><td> $\mathtt { u l t r a c h a t \_ v s \_ s h a r e g p t }$ </td><td>[0.219, 0.665]</td><td>[-0.107,0.604]</td></tr><tr><td>Strict-CI positive count</td><td>6/6</td><td>3/6</td></tr></table>

Under the seed-clustered estimator, three of the six original pairs have CI strictly above zero on the winsorized closure: false\_qa\_vs\_flan\_ $\therefore \mathtt { v 2 \_ p 3 } ,$ false\_qa\_vs\_sharegpt, and flan\_v2\_p3\_vs\_ultrachat. The other three pairs (false\_qa\_vs\_ultrachat, flan\_v2\_p3\_vs\_sharegpt, ultrachat\_vs\_sharegpt) have positive cluster-mean closure but CI including zero, reflecting the limited effective sample size at 3 seed-clusters. None of the six pairs flips sign under cluster-bootstrap; the qualitative ranking (bracket $\gg$ random ≈ 0, wrong-sign ≈ −1, first-order norm-matched negative) is preserved on every pair under both estimators. The cluster bootstrap is computed from the per-trial result files in the supplement (App. AE).

## P Triplet additivity of bracket edits

Setup. For each triplet $( A , B , C )$ on Qwen-3-4B $( \eta ~ = ~ 6 . 4 2 \times 1 0 ^ { - 4 } )$ we compute the three pair-brackets $b _ { A B } , ~ b _ { A C } , ~ b _ { B C } ~ \mathrm { a t } ~ \theta _ { 0 }$ on lm\_head, then train $A \ \to \ B \ \to \ C$ with four SGD steps per domain (last-block MLP + lm\_head) to reach $\theta _ { A B C }$ A subset edit $\mathcal { P } \in$ $\{ \emptyset , \{ A \dot { B } \} , \{ A C \} , \{ B C \} , \{ A B , A C \} , \{ A B , B C \} , \{ A B , A C , B C \} \}$ subtracts, for each $p \in \mathcal P .$ the correction $\eta ^ { 2 } b _ { p }$ from the lm\_head rows of the fewest tokens whose row norms account for 80% of that correction’s total row norm. For each $d \in \{ A , B , C \} , \Delta { \mathrm { N L L } } d$ is the change in perdomain evaluation NLL relative to $\theta _ { A B C }$ , with every edit evaluated on the same fixed batches.

Additivity is scored for the two composite edits $\{ A B , A C \}$ and {AB, BC} by comparing the actual $\Delta \mathrm { \bar { N L L } } _ { d }$ under the joint edit with the sum of the two singleton-edit values; the three-pair edit is measured but not scored. The relative additivity error for one (composite, domain) cell is | actual − predicted $| / |$ actual |, where predicted is the singleton sum.

Result. Across 3 triplets $\times ~ 2$ seeds $\times \ : 6$ (composite, domain) cells per triplet-seed (36 cells total), the grand mean relative additivity error is 4.29% (data: the triplet run’s result file in the supplement; raw ∆NLL scale $\sim 1 0 ^ { - 6 } – 1 0 ^ { - 5 } )$

Table 15: Per-triplet relative additivity error of bracket edits on Qwen-3-4B. “Per-seed mean” averages over the 6 (composite, domain) cells for that seed; “Cells $| \mathrm { e r r } | < 5 \% ^ { 3 }$ counts the cells whose relative additivity error is under 5%; “Triplet mean” averages over all 12 cells of the triplet. The grand mean over 36 cells is 4.29%.
<table><tr><td>Triplet  $( A , B , C )$ </td><td>Seed 42 mean</td><td>Seed 43 mean</td><td>Cells |err|  $< 5 \%$ </td><td>Triplet mean</td></tr><tr><td>code, legal, news</td><td>0.95%</td><td>1.11%</td><td>12/12</td><td>1.03%</td></tr><tr><td>code, biomedical, legal</td><td>5.17%</td><td>15.16%</td><td>7/12</td><td>10.17%</td></tr><tr><td>news, biomedical, math</td><td>1.41%</td><td>1.95%</td><td>12/12</td><td>1.68%</td></tr><tr><td>Grand mean (36 cells)</td><td colspan="4">4.29%</td></tr></table>

Interpretation. $3 1 / 3 6$ cells (86%) have relative error $\leq 5 \% ; 3 5 / 3 6$ have $\leq 1 4 \%$ . The grand mean is inflated by one cell of the code, biomedical, legal triplet at seed 43 where the prediction $( - 7 . 9 { \times } 1 0 ^ { - \bar { 8 } } )$ and the measured change $( - 2 . 7 { \times } 1 0 ^ { - 7 } )$ are both well below the triplet’s typical effect $( \sim 1 0 ^ { - 6 } )$ , so a small absolute error becomes a large relative one.

## Q Descriptive SVD of the cross-pair τ-matrix

The τ rows used in this appendix are follow-up $\theta _ { 0 }$ -reference readouts on an evaluation split shared across pairs, not the pair-specific Trotter-reference SFT interpretability readout used in Section 5.2; the two protocols give different surface tokens for the same nominal pair $( \mathrm { e . g . }$ , the follow-up code\_vs\_news top tokens are dominated by ⟨, \$, xml, php rather than by domain-marker proper nouns), so per-token semantic claims from Section 5.2 are not a direct argument about the SVD rows here. We report a descriptive SVD of the per-pair τ-matrix on the 6-pair benchmark grid. Stacking the per-pair τ vectors across six domain pairs (Qwen-3-4B, k = 1, seed 42) and taking the SVD of the resulting $6 \times V$ matrix yields singular values (22.83, 7.59, 3.84, 2.72, 0.94, 0.52) with cumulative energy fractions (0.866, 0.961, 0.986, 0.998, 0.999, 1.000). The top-2 singular directions carry 96.1% of the Frobenius mass, and the top-4 carry 99.8%. Because the rows are not norm-matched, this concentration can reflect unequal row norms as well as correlation across pairs; we report it descriptively.

Scope. The SVD is descriptive over six τ -vectors at one model, seed, and parameter subspace; principal directions are not interpreted semantically. The descriptive concentration is consistent with the 4.3% compositional additivity error (App. P) and the rapid plateau of iterative correction once the dominant modes are absorbed.

## R Null baselines for support specificity

We use null directions to separate generic heavy-tailed behavior of the logit readout from bracketspecific support structure. The fp32 check covers Llama-3.2-1B, Qwen-2.5-1.5B, and Qwen-3-4B over 3 pairs × 3 seeds per model, using the same finite-difference readout pipeline for the bracket and every control.

Controls. (i) Norm-matched random parameter direction, both globally norm-matched and pertensor norm-matched (3 instances each). (ii) First-order endpoint direction $\eta ( g _ { B } - g _ { A } )$ , normmatched to $\eta ^ { 2 } b _ { A B }$ . (iii) Pairing-permuted HVP $H _ { B } g _ { B } - H _ { A } g _ { A }$ , norm-matched. (iv) Empirical endpoint displacement $\theta _ { A B } - \theta _ { B A }$ from actual one-step SGD. (v) Batch-resampled bracket: $b _ { A B }$ recomputed on disjoint A/B batches, norm-matched. All directions read out via the same $\tau _ { k } =$ $\mathbb { E } [ e _ { k } \delta \dot { z } _ { k } ]$ at $\theta _ { \mathrm { r e f } }$ with ϵ = 1.

Concentration calibration. Gini and pct-for-80% mass are descriptive concentration summaries, not the specificity test. Random norm-matched directions can be as concentrated as the bracket on the two smaller fp32 checks: random directions require about 0.6–0.7% of vocabulary for 80% mass, while the bracket requires about 1.0–1.5%. On Qwen-3-4B the pattern reverses: the bracket requires 0.30% of vocabulary on average (0.042% median), compared with 0.66–0.68% for random directions. The stable claim across models is therefore support specificity, not Gini magnitude alone.

Support alignment. Empirical endpoint displacements and batch-resampled brackets preserve the bracket’s high-mass support, while random controls recover only the tokens that any direction excites. Mean top-20 overlap with the bracket is 99%/97% (endpoint/resampled) on Llama-3.2-1B, 99%/97% on Qwen-2.5-1.5B, and 82%/93% on Qwen-3-4B. The corresponding random-direction top-20 overlaps are 35%, 39–40%, and 36–49%. Because random directions share the same readout operator, uniform-vocabulary chance is used only as a scale reference; random-direction overlap is the relevant null.

Frequency correction. The smoothed $\kappa \mathrm { - } \mathrm { G i n i } = \mathrm { G i n i } ( \tau _ { k } / ( \mathrm { P r } ( y { = } k ) + \varepsilon ) )$ is high for both bracket and random directions. We therefore use κ as a label-frequency diagnostic, while endpoint and resampled-bracket support agreement provide the specificity check.

Precision check. Top-20 support is stable across $\epsilon \in \{ 0 . 0 5 , 0 . 1 , 0 . 3 , 1 . 0 \}$ on the precision checks (90–100% overlap, 100% sign stability among high-|τ| tokens; verified at fp32 perturbation materialization on representative pairs).

## S Observable/null decomposition of the training commutator

Setup. Let $\theta _ { 0 }$ denote the pre-trained base-model parameters. One gradient step from $\theta _ { 0 }$ with deterministic mean-gradient loss on domain A then $\dot { B }$ (no minibatch noise) yields $\theta _ { A B } ;$ the reverse order yields $\theta _ { B A }$ . The subspace-restricted training commutator is

$$
b _ { A B } : = H _ { B } g _ { A } - H _ { A } g _ { B } ,\tag{16}
$$

computed on the selected parameter subspace S consisting of the last transformer block’s MLP parameters together with the language-model head, d ≈ $4 . { \bar { 6 4 } } \times 1 0 ^ { 8 }$ entries. The Trotter reference point is $\theta _ { \mathrm { r e f } } : = \theta _ { 0 } - \eta ( g _ { A } + g _ { B } )$ with $\eta = 6 . 4 1 7 \times 1 0 ^ { - 4 }$ fixed by the cube-root step-size calibration.

The aggregated τ operator. Let $B _ { \mathrm { o b s } }$ be a fixed 4-batch subset of the unified the attribution evaluation split $E _ { \mathrm { a t t r } } ,$ the held-out slice used for τ (13 batches total; subset chosen to bound compute at Qwen-3-4B scale). We define the aggregated τ operator

$$
B : = P _ { K } \cdot S \cdot D \cdot J _ { z } ( \theta _ { \mathrm { r e f } } ) : S \longrightarrow \mathbb { R } ^ { K } ,\tag{17}
$$

where $J _ { z } ( \theta _ { \mathrm { r e f } } )$ is the logit Jacobian at the non-padding label positions of $B _ { \mathrm { o b s } } ;$ ; D is the diagonal error weighting $e _ { n , k } =$ softma $\mathfrak { x } ( z ) _ { n , k } - \mathcal { k } [ y _ { n } = k ]$ at $\theta _ { \mathrm { r e f } } ; S$ sums over those positions, $( S \cdot u ) _ { k } =$ $\textstyle \sum _ { n } u _ { n , k }$ for u $\in \mathbb { R } ^ { N \times V }$ ; and $P _ { K }$ restricts to the top-K vocabulary support covering 80% of $| \tau |$ mass, with τ computed by finite-difference attribution on the full 13-batch $E _ { \mathrm { a t t r } }$ at $\theta _ { \mathrm { r e f } }$ . The unrestricted counterpart is $B _ { \mathrm { f u l l } } \dot { \mathbf { \eta } } : = C \cdot E \cdot J _ { z } ( \theta _ { \mathrm { r e f } } ) : S  \mathbb { R } ^ { V }$

Observable/null decomposition. Under the Euclidean inner product on flattened selectedparameter coordinates, every $\delta \theta \in { S }$ admits a unique orthogonal decomposition

$$
\delta \theta = \delta \theta ^ { \mathrm { o b s } } + \delta \theta ^ { \mathrm { n u l l } } , \quad \delta \theta ^ { \mathrm { o b s } } \in \mathrm { r o w } ( B _ { \mathrm { f u l l } } ) , \quad \delta \theta ^ { \mathrm { n u l l } } \in \mathrm { k e r } ( B _ { \mathrm { f u l l } } ) .\tag{18}
$$

The readout-null component produces zero aggregated logit-loss effect by construction: it lies in the kernel of the chosen observation operator. The observablefraction of the training commutator under operator B is

$$
\rho _ { \mathrm { o b s } } ( B ) : = \frac { \| P _ { \mathrm { r o w } ( B ) } \cdot b _ { A B } \| _ { 2 } } { \| b _ { A B } \| _ { 2 } } \in [ 0 , 1 ] .\tag{19}
$$

Computation. We obtain ρ via the least-squares construction $x ^ { \star } = B ^ { \top } u$ with $\left( B B ^ { \top } + \nu I \right) w =$ $\boldsymbol { B } \cdot \boldsymbol { b } _ { A B }$ and Tikhonov damping $\nu = 1 0 ^ { - 6 }$ , solved by conjugate gradient. In the exact zero-damping, fully-converged limit, $\cos ( \bar { x } ^ { \star } , \bar { b } _ { A B } ) = \rho _ { \mathrm { o b s } } ( B )$ . Each CG iteration costs one Jacobian-vector product and one vector-Jacobian product. The CG runs in the model’s native dtype as loaded (bfloat16 for Qwen-3-4B, no explicit fp32 cast), with maximum 200 iterations and target relative residual $1 0 ^ { - 5 } $ ; the achieved residual is reported per pair.

Three-way decomposition (approximate empirical). Comparing $\rho _ { \mathrm { t o p K } } : = \rho _ { \mathrm { o b s } } ( B )$ to $\rho _ { \mathrm { f u l l } } : =$ $\rho _ { \mathrm { o b s } } ( B _ { \mathrm { f u l l } } )$ yields

$$
\underbrace { \rho _ { \mathrm { t o p K } } ^ { 2 } } _ { \mathrm { t o p - K \ " o b s e r v a b l e } } + \underbrace { \rho _ { \mathrm { f u l l } } ^ { 2 } - \rho _ { \mathrm { t o p K } } ^ { 2 } } _ { \mathrm { n o n - t o p - K \ : o b s e r v a b l e } } + \underbrace { 1 - \rho _ { \mathrm { f u l l } } ^ { 2 } } _ { \mathrm { r e a d o u t - n u l l } } = 1 ,\tag{20}
$$

exact only in the zero-damping and full-CG-convergence limit; reported here as an approximate empirical decomposition at the achieved CG residual.

Results. Three pairs on Qwen-3-4B (Table 16): across all three the aggregated top-K τ operator captures the majority of the bracket’s parameter-space energy in the selected subspace. Empirical $\rho _ { \mathrm { t o p K } }$ lies in [0.88, 0.94] (mean 0.918), and $\rho _ { \mathrm { f u l l } }$ lies in [0.92, 0.97] (mean 0.953). The squared-cosine proxies give a top-K observable fraction $0 . 7 7 \mathrm { - } 0 . 8 9$ (mean 0.844), a further non-top-K observable fraction 0.05–0.08, and an unexplained fraction 0.06–0.15 (mean 0.092) at the achieved CG residual per pair. The achieved CG relative residual ranges from $2 . 7 \times 1 0 ^ { - 3 }$ on legal/biomedical $( K = 4 8 5 )$ to $2 . 5 \times 1 0 ^ { - 2 }$ on code/math $( K = 8 8 4 )$ and code/news $( K = 9 2 8 )$ ; the best-converged pair has the smallest unexplained fraction, consistent with the decomposition being sensitive to incomplete CG convergence on larger-K systems.
<table><tr><td>Pair</td><td> $\rho _ { \mathrm { t o p K } }$ </td><td> $\rho _ { \mathrm { f u l l } }$ </td><td>K</td><td>V</td><td> $\mathrm { t o p } { \cdot } \mathrm { K }$ </td><td>non-top</td><td>null</td><td>CG topK</td><td>CG full</td></tr><tr><td>code/news</td><td>0.931</td><td>0.966</td><td>928</td><td>151 936</td><td>0.868</td><td>0.066</td><td>0.067</td><td> $2 . 3 { \cdot } 1 0 ^ { - 2 }$ </td><td> $2 . 1 { \cdot } 1 0 ^ { - 2 }$ </td></tr><tr><td>code/math</td><td>0.880</td><td>0.923</td><td>884</td><td>151 936</td><td>0.775</td><td>0.077</td><td>0.148</td><td> $2 . 5 { \cdot } 1 0 ^ { - 2 }$ </td><td> $2 . 3 { \cdot } 1 0 ^ { - 2 }$ </td></tr><tr><td>legal/bio</td><td>0.942</td><td>0.970</td><td>485</td><td>151 936</td><td>0.888</td><td>0.053</td><td>0.060</td><td> $2 . 7 { \cdot } 1 0 ^ { - 3 }$ </td><td> $6 . 7 { \cdot } 1 0 ^ { - 3 }$ </td></tr><tr><td>mean</td><td>0.918</td><td>0.953</td><td></td><td></td><td>0.844</td><td>0.065</td><td>0.092</td><td></td><td></td></tr></table>

Table 16: Observable/null decomposition of $b _ { A B }$ on Qwen-3-4B, selected subspace S (last-block $\mathbf { M L P + L M }$ head, $d \approx 4 . 6 4 \times 1 0 ^ { 8 } ) , \eta = 6 . 4 1 7 \times 1 0 ^ { - 4 } , k = 1$ , seed 42. $\rho _ { \mathrm { o b s } } ( B ) : = \cos ( x ^ { \star } , b _ { A B } )$ at the achieved CG relative residual shown in the final two columns; Tikhonov damping $\nu = 1 0 ^ { - 6 }$ Decomposition columns report $\rho _ { \mathrm { t o p K } } ^ { 2 } , \rho _ { \mathrm { f u l l } } ^ { 2 } - \rho _ { \mathrm { t o p K } } ^ { 2 } .$ , and $1 - \rho _ { \mathrm { f u l l } } ^ { 2 } ;$ exact only in the zero-damping / full-convergence limit, reported here as approximate empirical fractions.

Interpretation. The decomposition quantifies what sparsity $o f \tau$ means as a statement about the training commutator itself in the selected parameter subspace and under aggregated logit readout. Empirically the top-K τ signature captures the dominant observable component of the bracket: the squared-cosine proxies assign $7 7 - 8 9 \%$ of $\| b _ { A B } \| _ { 2 } ^ { 2 }$ to row(B), a further $5 { - } 8 \% \mathrm { t o } \mathrm { r o w } ( B _ { \mathrm { f u l l } } )$ beyond row(B) (observable, but outside the top-K vocabulary cut), and leave 6–15% unexplained. These proxies come from damped, finite-CG solves in native bf16. In exact arithmetic each proxy lowerbounds its observable energy fraction, but the difference between the two proxies is not a certified subspace fraction. The unexplained part mixes a readout-null component (directions in parameter space that leave the aggregated logit-loss readout unchanged) with damping, solve and rounding error. The high proxy values indicate that the aggregated readout reaches most of the bracket in the selected parameter subspace, so the sparse top-K τ readout is not merely a compressed summary of the bracket (Table 16). Measured in the aggregated-readout subspace rather than in the original parameter coordinates, sparsity of the readout and sparsity of the bracket approximately coincide.

Scope and caveats.

1. The selected parameter subspace is last-block MLP plus LM head, not the full model. Claims about parameter-space reach apply to this subspace only.

2. Operator B is evaluated on a 4-batch subset of $E _ { \mathrm { a t t r } }$ while top-K is defined on the full   
13-batch $E _ { \mathrm { a t t r } }$ . This is a compute-bounded approximation; $\rho$ depends on the eval-batch

sampling, so reported values should be read as batch-subset-specific estimates of the aggregated operator.

3. ρ is an approximate projection coefficient under finite CG and nonzero damping. We report ρ at the achieved CG residual, not the nominal tolerance target.

4. Single training seed (42) per pair; $N = 3$ domain pairs. These estimates are descriptive for the reported grid.

## T Learning Rate Calibration Details

The choice of $\eta$ is critical (Eq. 5): too small η yields undetectable signals; too large η violates second-order truncation accuracy. We use an automated calibration procedure to select η in the “cube-root” regime.

## T.1 Calibration regimes

The magnitude of the leading bracket term scales as $\eta ^ { 2 } ,$ , so η must be large enough to clear numerical noise but small enough to remain BCH-local. We use “cube-root” as an empirical regime label inherited from the scalar-bracket protocol, not as a theorem required by the present paper. Operationally, the accepted window is the range where (i) the bracket sign predictor is stable on held-out evaluation batches, (ii) the bracket displacement remains subordinate to the first-order update, and (iii) wrong-sign controls behave according to the local antisymmetry prediction.

We distinguish three practical regimes. In the small-signal regime, bracket effects are below the loss-evaluation floor. In the BCH-local window, the bracket is measurable while higher-order terms remain controlled. In the saturated regime, loss improvement plateaus or sign controls become incoherent, indicating that the second-order truncation is no longer the right local model.

## T.2 Calibration procedure

The calibration algorithm:

1. Initialize: Start with a conservative $\eta _ { 0 } ( \mathrm { e . g . , 1 0 ^ { - 4 } } )$

2. Probe: For candidate $\eta ,$ run a short training sequence (10 steps) and measure:

• Loss improvement $\Delta \mathcal { L }$

• Parameter movement $\| \Delta \theta \|$

• Gradient norm $\| \nabla \mathcal L \|$

3. Classify regime: Compute scaling exponents via finite differences across multiple $\eta$ values. Classify as linear/cuberoot/strongly-powered based on exponent ranges.

4. Binary search: Adjust η to target the cube-root regime boundary. Typical range: $1 0 ^ { - 4 }$ to $1 0 ^ { - 2 }$ for LLMs.

5. Validation: Verify that Lie bracket magnitude is $1 0 ^ { - 2 }$ to $1 0 ^ { - 1 }$ times first-order gradient norm (detectable but subordinate).

## T.3 Calibrated Values

Table 17: Learning rates calibrated per model.
<table><tr><td>Model</td><td>η</td><td>Regime</td></tr><tr><td>Qwen-3-4B</td><td> $6 . 4 2 \times 1 0 ^ { - 4 }$ </td><td>cube-root</td></tr><tr><td>Qwen-2.5-1.5B</td><td> $1 . 9 1 \times 1 0 ^ { - 3 }$ </td><td>cube-root</td></tr><tr><td>Llama-3.1-8B</td><td> $1 . 2 8 \times 1 0 ^ { - 3 }$ </td><td>cube-root</td></tr><tr><td>Llama-3.2-1B</td><td> $8 . 0 \times 1 0 ^ { - 4 }$ </td><td>cube-root</td></tr><tr><td>GPT-2 (reference)</td><td> $4 . 1 6 \times 1 0 ^ { - 3 }$ </td><td>strongly powered</td></tr></table>

Table 17 reports final calibrated SFT learning rates. All target models fall in the cube-root regime;   
GPT-2 is reference only.

## T.4 Implementation

The calibration implementation is included in the supplementary code; the reported runs use the fixed calibrated values in Table 17.

## U Intervention Types

We tested four intervention methods to validate the causal role of the predicted harmful $\tau _ { k }$ support. Main mitigation runs target the ten harmful tokens defined below; descriptive sparsity analyses continue to rank by $| \tau _ { k } |$

## U.1 Loss Reweighting (Main Method)

During alternating training $\theta _ { A B }$ , we reweight per-position cross-entropy loss for domain A:

$$
\mathcal { L } _ { \mathrm { r e w e i g h t e d } } = \frac { 1 } { N _ { \mathrm { p o s } } } \sum _ { t = 1 } ^ { N _ { \mathrm { p o s } } } w _ { t } \cdot \ell _ { t }\tag{21}
$$

where $w _ { t } ~ = ~ \alpha$ if the label at position t is a harmful token and $w _ { t } ~ = ~ \omega$ otherwise, with meanpreserving normalization. The harmful tokens, used by every method in this appendix, are the ten largest- $| \tau _ { k } |$ | tokens whose $\tau _ { k }$ has the sign of the measured baseline order gap. That sign is positive in $8 9 / 9 0$ Qwen-3-4B trials and $2 5 / 2 8$ above-floor Qwen-2.5-1.5B trials, where the harmful tokens are therefore the top-positive-τ<sub>k</sub> tokens; the helpful tokens are the ten $\mathrm { l a r g e s t - } | \tau _ { k } |$ tokens of the opposite sign. The normalization is

$$
\omega = \frac { 1 - p \alpha } { 1 - p } , \quad p = \mathrm { { f r a c t i o n \ o f \ p o s i t i o n s \ w h o s e \ t a r g e t i s \ a \ h a r m f u l \ t o k e n } . }\tag{22}
$$

This keeps the mean position weight at 1.

Rationale. Loss reweighting suppresses the label-token gradient contribution associated with the selected harmful tokens. By downweighting positions where harmful tokens appear as labels, we target the predicted support of the ordering gap while keeping the mean position weight fixed.

Hyperparameter. We sweep $\alpha \in \{ 0 . 0 1 , 0 . 1 , 0 . 5 , 1 . 0 \}$ where $\alpha < 1$ suppresses harmful tokens (Figure 4). Results use $\alpha = 0 .$ 1 (90% downweight, i.e., harmful tokens contribute 10% of baseline loss weight) unless noted.

## U.2 Learning Rate Scaling

Modify the per-token learning rate by scaling gradients for lm\_head rows corresponding to harmful tokens:

$$
g _ { \mathrm { s c a l e d } } [ k , : ] = { \left\{ \begin{array} { l l } { \alpha \cdot g [ k , : ] } & { { \mathrm { i f ~ } } k \in \mathrm { t o p } { \mathrm { - } } 1 0 \mathrm { h a r m f u l } } \\ { g [ k , : ] } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }\tag{23}
$$

Difference from loss reweighting. Loss reweighting modifies the loss landscape (affects both the hidden-state and the weight pathway). Learning rate scaling modifies only the gradient applied to lm\_head weights (weight pathway). For small-gap models, the more targeted learning-rate scaling can reduce noise by avoiding amplification through the hidden-state pathway.

## U.3 Projection (PCGrad-Inspired)

Project harmful token gradients orthogonal to helpful token gradients:

$$
g _ { \mathrm { p r o j } } [ k , : ] = g [ k , : ] - \frac { g [ k , : ] \cdot g _ { \mathrm { h e l p f u l } } [ k , : ] } { \| g _ { \mathrm { h e l p f u l } } [ k , : ] \| ^ { 2 } } g _ { \mathrm { h e l p f u l } } [ k , : ]\tag{24}
$$

for $k \in \mathrm { t o p } \mathrm { - } 1 0$ harmful tokens, where $g _ { \mathrm { h e l p f u l } }$ is the mean gradient for top-10 helpful tokens (the opposite sign).

![](images/d19fc49eba7dc7628f226495f480bf84d9149660da23bac42c78ba06f26ea61f.jpg)  
Intervention strength α (log scale)  
Figure 4: Dose-response of loss reweighting on Qwen-3-4B (an earlier sweep, run before the held out evaluation of Table 3). Median gap reduction (%) against the weight α applied to harmful (red), helpful (green), and random (gray) token sets; $\alpha < 1$ downweights. Downweighting harmful tokens gives a positive response at every tested $\alpha < 1$ , while random controls remain near zero. Shaded bands show 95% confidence intervals (30 trials each).

Rationale. Inspired by PCGrad [Yu et al., 2020], this prevents harmful token learning from interfering with helpful token learning. In practice, this showed weaker effects than loss reweighting on Qwen-3-4B.

## U.4 Regularization

Add L2 penalty driving harmful token embedding rows toward initialization:

$$
\mathcal { L } _ { \mathrm { r e g } } = \mathcal { L } _ { A } + \lambda \sum _ { k \in \mathrm { h a r m f u l } } \| W [ k , : ] - W _ { 0 } [ k , : ] \| ^ { 2 }\tag{25}
$$

where W is the lm-head weight matrix, $W _ { 0 }$ its value at $\theta _ { 0 } ,$ and $\lambda = 1 0 ^ { - 3 }$

Rationale. Prevents harmful tokens from accumulating large updates. Less effective than direct gradient manipulation (loss reweighting or learning-rate scaling).

## U.5 Comparative Results

Table 18: Intervention type comparison $( \alpha = 0 . 1$ , 30 trials/pair). Sep.: ✓ clear harmful-vs-random separation, ∼ weak, × none. <sup>†</sup>Qwen-3-4B LR scaling, Projection, and Regularizer rows are from earlier runs without held-out evaluation; the Loss reweighting row uses held-out evaluation. The Qwen-2.5-1.5B row is the full-fp32 above-floor subset reported in Table 3.
<table><tr><td>Model</td><td>Precision</td><td>Method</td><td>Harmful (med.)</td><td>Random (med.)</td><td>Sep.</td></tr><tr><td>Qwen-3-4B</td><td>bf16</td><td>Loss reweighting</td><td>+32.0%</td><td>+0.2%</td><td>√</td></tr><tr><td>Qwen-3-4B</td><td>bf16</td><td>LR scaling†</td><td>+31.2%</td><td>+1.7%</td><td>V</td></tr><tr><td>Qwen-3-4B</td><td>bf16</td><td>Projection†</td><td>+14.6%</td><td>-5.1%</td><td>2</td></tr><tr><td>Qwen-3-4B</td><td>bf16</td><td>Regularizer†</td><td>+8.3%</td><td>+2.9%</td><td>X</td></tr><tr><td>Qwen-2.5-1.5B</td><td>fp32</td><td>LR scaling</td><td>+47.9%</td><td>-0.003%</td><td>V</td></tr></table>

## Summary.

• Qwen-3-4B (bf16): Loss reweighting closes a median 32% of the gap on harmful tokens and ∼0% on random tokens under held-out evaluation (disjoint from training data); LR scaling gives a similar effect (+31.2%) in earlier runs without held-out evaluation.

• Qwen-2.5-1.5B (fp32): LR scaling yields +47.9% harmful closure on the 28/90 abovefloor trials, with –0.003% random closure. We treat this as a conditioned fp32 result and do not compare its magnitude directly to Qwen-3-4B.

• Attribution term: Selecting tokens with the full bracket $H _ { B } g _ { A } - H _ { A } g _ { B }$ (Cohen’s $d =$ 0.53) outperforms the $H _ { B } g _ { A }$ term alone $( d = 0 . 2 5 )$ , confirming that both Hessian-vector product terms contribute to the ordering signal.

## V What remains after the token edit

Table 3 evaluates one targeted edit: during the update on source A, positions whose target is one of the ten harmful tokens defined in Appendix U are downweighted by 90%, and other positions are rescaled to keep the mean weight unchanged. The edit changes only the source-A update; it neither subtracts the predicted commutator displacement nor forces the A-then-B and B-then-A endpoints to agree. The 32% median closure is therefore the effect of this edit, not the fraction of the ordering gap explained by commutator memory, and its complement is not a separately identified 68% component.

To test whether higher-order terms carry the remainder, we applied the one-update-per-source Qwen-3-4B intervention protocol across 90 trials (three source pairs, 30 trials each) and compared three quantities after reweighting: (i) the held-out NLL gap predicted by the leading commutator; (ii) the linearized NLL gap obtained from the parameter difference produced by the two actual updates, which includes finite-step path effects; and (iii) the NLL gap measured at the two endpoints. For each source pair we flip signs so that the original gaps are positive, divide the summed leading-order predictions by the summed magnitudes of the original gaps, and average the three pair-level ratios equally.

After the edit, the leading-order commutator still predicts 71% of the pre-intervention measured gap, and it points in the original direction for every source-pair average. The step from (i) to (ii) is where finite-step corrections, including any net higher-order BCH effect, enter: it reduces the remaining effect for code-vs-news and code-vs-math but increases it for code-vs-legal, so these corrections do not explain a common remainder. The step from (ii) to (iii), the loss-surface correction, reduces the gap for all three pairs. Although 71% is numerically close to 68%, the two are different statistics: 68% is the complement of the median absolute closure in Table 3, whereas 71% is a leading-order prediction normalized within each source pair and then averaged across pairs.

## W Magnus expansion validation

Using the BCH/Magnus expansion logic underlying classical exponential integrators [Magnus, 1954, Blanes et al., 2009], we compare, on Qwen-3-4B, the measured held-out gap $\mathcal { L } _ { E } ( \theta _ { A B } ) -$ $\mathcal { L } _ { E } ( \theta _ { B A } )$ after k SGD steps per source with its leading bracket prediction $k ^ { 2 } \eta ^ { 2 } \langle g _ { E } ( \bar { \theta _ { \mathrm { r e f } } } ) , \dot { b _ { A B } } ( \dot { \theta } _ { 0 } ) \rangle$ (Eq. 7). The validation uses 12 points (3 seeds × 4 step counts), with the same A and B batches repeated at every step.

Table 19: Magnus/bracket validation details (Qwen-3-4B). The slope 0.706 in Fig. 5 is the pooled regression slope over $k \in \{ 1 , 2 , 4 , 8 \}$ ; the $k = 1$ ratio is reported separately because the BCH expansion is most accurate there.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Plotted points</td><td> $n = 1 2$ </td></tr><tr><td> $R ^ { 2 }$  of the  $k ^ { 2 }$  fit (mean over seeds)</td><td>0.994</td></tr><tr><td>Pooled predicted-vs-actual  $R ^ { 2 }$ </td><td>0.990</td></tr><tr><td>Best-fit slope in pooled plot</td><td>0.706</td></tr><tr><td>Mean magnitude ratio at  $k = 1$ </td><td> $1 . 0 0 4 \pm 0 . 0 1 5$ </td></tr></table>

The measured gap grows nearly as $k ^ { 2 }$ , the scaling of the leading bracket term. The pooled slope is below one because higher-order BCH terms grow with k: the measured-to-predicted ratio falls from 1.004 at k = 1 to 0.97, 0.90, and 0.74 at $k = 2 , 4$ , and 8.

Magnus Expansion Validation (Qwen-3-4B)  
![](images/d2fd6344b5b6d80f06463a65bd486c2b2695e211abeb6c6b53d55cef32c25b73.jpg)  
Figure 5: Magnus/bracket validation (Qwen-3-4B). Measured held-out gap $\mathcal { L } _ { E } ( \theta _ { A B } ) - \mathcal { L } _ { E } ( \theta _ { B A } )$ against its leading bracket prediction $\bar { k ^ { 2 } \eta ^ { 2 } } \langle g _ { E } ( \theta _ { \mathrm { r e f } } ) , b _ { A B } \rangle$ for 3 seeds and $k \in \{ 1 , 2 , 4 , 8 \} \ \mathrm { S G D }$ steps per source. The $k ^ { 2 }$ fit has mean $R ^ { 2 } = \stackrel { . . } { 0 . 9 9 4 } ;$ the pooled predicted-vs-actual fit has slope 0.706 and $\bar { R ^ { 2 } } = 0 . 9 9 0 ;$ ; signs agree at all 12 points.

## X Per-Trial Results

Per-trial records for the Qwen-3-4B held-out intervention run (90 trials across 3 domain pairs $\times \ 3 0$ seeds) and the Qwen-2.5-1.5B float32 run (90 trials) are included in the supplementary runs/ directory with columns: trial, seed, domain\_A, domain\_B, gap\_baseline, gap\_intervened, gap\_reduction\_pct.

## Y Interpretability extended

Per-pair token characters. Code vs. news: 17/20 top tokens are complete words, predominantly proper nouns and brand names (“Ask”, “Facebook”, “ESPN”, “Amazon”, “Washington”); 87% of $\left| \tau _ { k } \right|$ mass. Code vs. math: 17/20 are complete words – academic metadata (“abstract” $\stackrel { , } { \tau = } - 9 . 5 ,$ , “ti-$\mathrm { t l e } ^ { , ; } \tau = - 4 . 0 .$ “author”, “layout”, “sidebar”) reflecting LaTeX source structure, plus code keywords (“extends”, “pragma”, “include”). Code vs. legal: 9/20 are single digit tokens contributing 92% of mass; token $\ " 0 \ "$ alone has $\tau = + 2 3 . 1$

Model scale. Comparing Qwen-3-4B (4B) to Qwen-2.5-1.5B (1.5B), the same domain pairs show |τ<sub>k</sub>| magnitudes 19–3700× larger in the larger model. The concentration pattern is consistent, while the absolute readout strength is larger in the larger model.

Prediction/label decomposition. Since $e _ { k } = p _ { k } - \nVdash _ { y = k } , \tau _ { k } = \tau _ { k } ^ { p } - \tau _ { k } ^ { y }$ where $\tau _ { k } ^ { p } = \mathbb { E } [ p _ { k }$ $\delta z _ { k } ]$ (predictions) and $\tau _ { k } ^ { \bar { y } } = \mathbb { E } [ \mathcal { H } _ { y = k } \cdot \delta z _ { k } ]$ (label occurrence). Aggregate $\lvert \tilde { \tau } ^ { p } \rvert / \lvert \tau ^ { y } \rvert \stackrel {  } { = } 0 . 8 8 ;$ both $\mathrm { G i n \bar { i } > 0 . 9 7 }$ . 5 of top-20 are pure-prediction tokens with $\tau _ { k } ^ { y } = \bar { 0 }$ , and 30% of mass comes from such unobserved tokens. Cancellation between terms is rare (11% of tokens). Smoothed frequency correction $\kappa _ { k } = \tau _ { k } / ( \mathrm { P r } ( y { = } k ) + \varepsilon )$ is reported as a frequency diagnostic (Spearman 0.98, top-100 overlap 80% within this check).

Token transfer within a model family (Qwen-3-4B vs. Qwen-2.5-1.5B; Llama-3.1-8B vs. Llama-3.2-1B). Top-500 overlap is 28.6× above the equal-size baseline on Qwen pairs (47/500 on code-vs-news; 19/500 on code-vs-math, 11.5×), and 9.7× on Llama pairs (19/500 on code-vsmath). Shared tokens are semantically coherent domain markers (e.g., “Facebook”, “Linux”, “Programming”). Qwen and Llama use different tokenizers; cross-family is quantified only in semantic categories.

Top-10 tokens by |τ<sub>k</sub>| per domain pair (Qwen-3-4B)  
![](images/b213eeb275e075e0f7998671e3ce1767529b861587c2255b3d44035e8e08dc5e.jpg)

Figure 6: Top-10 tokens by $| \tau _ { k } |$ for each domain pair (Qwen-3-4B). Color encodes signed $\tau _ { k }$ (red = order AB disadvantages token; blue = advantages). Tokens are almost entirely disjoint across pairs (Jaccard = 0.004), confirming domain specificity. Code vs. legal is dominated by digit tokens; code vs. biomedical by BPE suffix fragments.  
![](images/4dc7c7cbe1b4b59932f26555b7fd28fbe30181b0e60cae83956d1eacd49dff01.jpg)

![](images/5a668d1db037b4153fb7796b17fb3b2a506254da34820f33d87e96467d32deab.jpg)  
Figure 7: Label frequency vs. $\left| \tau _ { k } \right|$ for all vocabulary tokens (code vs. news). $\mathrm { H i g h - } | \tau _ { k } |$ tokens span the full frequency range; low rank correlation $( \rho \approx 0 . 2 3 )$ argues against a Zipf-only explanation.

Rank vs. magnitude. Overlap statistics measure rank-order agreement within each model’s |τ| distribution; they do not imply absolute-magnitude convergence. Top-20 mean |τ | differs by 4+ orders of magnitude across the four models (≈ 1.31 on Qwen-3-4B, $8 . 8 \times 1 0 ^ { - 3 }$ on Qwen-2.5-

<table><tr><td>Slice (run) Tokens, and the readout of the colored ones</td><td></td></tr><tr><td>code/news Ask</td><td>HN: Should include my GPA and /or transcripts when applying for jobs?</td></tr><tr><td rowspan="2">code/biomedical Comput ers</td><td>Ask: τ = −3.51 (rank 1)</td></tr><tr><td>are a ubiquitous part of the amb ulatory health care environment ulatory: τ = +0.437 (rank 8); of: τ = −0.364 (rank 19)</td></tr><tr><td rowspan="2">code/news (iterative Facebook run) after bracket correction (97.0% of its NLL gap closed)</td><td></td></tr><tr><td>Dating launch blocked in Europe after it fails to show privacy workings Facebook: τ = −1.75 (rank 5); its NLL is 10.074 under AB and 10.896 under BA, and 10.920</td></tr></table>

Figure 8: Tokenwise view of the held-out sentences in Section 5.2, with one box per Qwen-3-4B token. A token is colored if it is among its pair’s 20 largest-|τ| tokens in the slice-level readout (rank in parentheses): blue for $\tau < 0$ (order AB advantages the token) and red for $\tau > 0$ (AB disadvantages it), as in Fig. 6. Uncolored tokens are outside that set. The readout is aggregated over the evaluation slice (evaluation seed 44) rather than scored per sentence. The last row also tracks the Facebook position through the instrumented iterative-correction run, which targets $\theta _ { B A }$ and overshoots it slightly (by 0.025).

1.5B, $1 . 2 \times 1 0 ^ { - 2 }$ on Llama-3.1-8B, $4 . 3 \times 1 0 ^ { - 5 }$ on Llama-3.2-1B); ∼ 30,000× residual after $\eta ^ { 2 }$ normalization.

Why these categories. The three token categories (domain markers, BPE morpheme fragments, digit asymmetry) all identify tokens with large error-weighted logit responses $\boldsymbol { e } _ { k } \delta z _ { k }$ along the bracket. Domain-exclusive tokens carry large responses; morpheme fragments aggregate them via BPE compression; digit tokens reflect differences in how domains use numbers. The category counts and |τ |-mass fractions are computed on the top-50 tokens per pair from the Qwen-3-4B main run; a held-out re-run reproduces the same top-token ranking on the two pairs available in both runs.

## Z Tables: ordering baselines, multi-step persistence, scaling in k

Table 20: Ordering prediction accuracy on 18 pair-seed units (6 domain pairs × 3 seeds per model).
<table><tr><td>Method</td><td>Qwen-3-4B</td><td>Qwen-2.5-1.5B</td><td>Llama-3.1-8B</td></tr><tr><td>Bracket</td><td>72% (13/18)</td><td>100% (18/18)</td><td>100% (18/18)</td></tr><tr><td>Grad Cosine</td><td>11% (2/18)</td><td>33% (6/18)</td><td>22% (4/18)</td></tr><tr><td>Grad Norm</td><td>11% (2/18)</td><td>56% (10/18)</td><td>22% (4/18)</td></tr><tr><td>Random</td><td>44% (8/18)</td><td>50% (9/18)</td><td>39% (7/18)</td></tr></table>

## Ordering-prediction baselines.

Multi-step persistence. This recall statistic is intentionally asymmetric and is not the same as the equal-cardinality support-recovery metric in Table 9. Table 9 compares predicted and empirical top sets of the same size within the active vocabulary for single-step token prediction; Table 21 asks whether the empirical high-effect active tokens across multiple step counts remain covered by one fixed top-1% full-vocabulary τ set. Table 21 draws fresh batches at every step, so its gap growth (3.1× at k=8 on Qwen-3-4B) is not comparable with the repeated-batch $k ^ { 2 }$ check of Appendix W.

τ -vs-∆W scaling in $k .$ The τ -vs-∆W Spearman correlation decays log-linearly across the six Qwen-3-4B domain pairs at $k \in \{ 1 , 4 , 8 \} \colon { \dot { \rho } } ( k ) \approx 0 . 8 3 - 0 . 0 5 5$ · log k (mean decay rate 0.055 ± $\bar { 0 . 0 1 5 , \ r ^ { 2 } \ > \ 0 . 9 5 ) }$ . The bracket’s Lemma 2.1 expansion is a k = 1 object; higher-order terms accumulate at roughly this rate for $k \leq 8 .$

## AA Iterative correction: per-position trajectories and downstream metrics

Iterative correction starts at $\theta _ { A B }$ . Each iteration recomputes the bracket b at the current point θ on fresh A and B batches, sets $\gamma = \langle \theta - \theta _ { B A } , b \rangle / \langle b , \bar { b } \rangle$ , and updates the trained parameters by

Table 21: Multi-step persistence with fresh batches per step (averaged over 12 experiments per model). τ recall@1%V is the fraction of the empirical top-1% set (among tokens occurring as labels) at step count k contained in the single-step bracket top-1% full-vocabulary τ set. Qwen-3- 4B uses bf16 storage with fp32 HVPs; Qwen-2.5-1.5B is full fp32. gap/gap<sub>1</sub>: measured gap relative to $k = 1 ; k ^ { 2 }$ : the leading-order prediction.
<table><tr><td>Model</td><td>Storage/HVP</td><td>k</td><td>Gini</td><td>τ recall@1%V</td><td>gap/gap1</td><td> $k ^ { 2 }$ </td></tr><tr><td rowspan="4">Qwen-3-4B</td><td rowspan="4">bf16/fp32</td><td>1</td><td>0.9998</td><td>100%</td><td>1.0</td><td>1</td></tr><tr><td>2</td><td>0.9998</td><td>100%</td><td>1.5</td><td>4</td></tr><tr><td>4</td><td>0.9998</td><td>100%</td><td>2.0</td><td>16</td></tr><tr><td>8</td><td>0.9997</td><td>100%</td><td>3.1</td><td>64</td></tr><tr><td rowspan="4">Qwen-2.5-1.5B</td><td rowspan="4">fp32</td><td>1</td><td>0.9982</td><td>99%</td><td>1.0</td><td>1</td></tr><tr><td>2</td><td>0.9982</td><td>99%</td><td>4.0</td><td>4</td></tr><tr><td>4</td><td>0.9983</td><td>98%</td><td>17.8</td><td>16</td></tr><tr><td>8</td><td>0.9983</td><td>98%</td><td>57.1</td><td>64</td></tr></table>

$\theta  \theta - \gamma b$ . It stops at 20 iterations, or earlier if the remaining held-out gap falls below 1% of the original or closure changes by less than 0.5 percentage points over three iterations.  
![](images/38dc145154cc2bf5d792ec7cfb0e154b4a3a6dd8ce4c905818184ab79eded31a.jpg)  
Figure 9: Iterative bracket correction convergence (averaged per pair). Five of six pairs reach >85% closure within 20 iterations; news-vs-biomedical $( k { = } 1 \mathrm { g a p } \mathrm { 8 } . \bar { 5 } { \times } \bar { 1 } 0 ^ { - 4 } $ , near the noise floor) diverges. On the other five pairs the iterative projection closes 90%, against 13.5% for the single-step fullbracket edit averaged over all pairs and step counts (6.7×; App. I).

Per-position movement. Instrumented runs track NLL of individual sequence positions throughout the correction. For each experiment we select the 50 positions with the largest $\left| \mathrm { N L L } ( \theta _ { A B } ) \right.$ $\mathrm { N L L } ( \theta _ { B A } ) |$ and measure their NLL at every iteration. Mean per-position convergence is 85% across five domain pairs; top-gap tokens $( \mathrm { e . g . }$ , “Netflix”, “Facebook”, “Google” for code-vs-news) reach 93–97% of their $\theta _ { B A }$ targets. Figure 10 shows representative trajectories.

Downstream likelihood detail. On a representative code-vs-math pair on Qwen-3-4B at $k = 1$ gold-answer teacher-forced NLL closes 14.2% on GSM8K (1319 problems, 1000-iterate bootstrap CI [0.126, 0.157]) and 20.4% on HumanEval (164 problems, CI [0.178, 0.232]). Discrete accuracy (pass@1, exact match) moves by at most 7 percentage points at $\bar { k = 1 }$ ; these changes are not tested for significance.

![](images/fceb0140c97dd4736a417ed3ef33c242b94e39066ab010995040a4b08832e21b.jpg)  
Figure 10: Per-position NLL trajectories during iterative bracket correction (code-vs-news, seed=42, $k = 1 )$ . Each line tracks one sequence position’s NLL; dashed lines show $\theta _ { B A }$ targets.

Selectivity diagnostic. Across the 34 experiments that did not diverge (all but the two newsvs-biomedical seeds with negative closure), domain-A NLL moves to $9 9 . 3 \% \pm \ 3 \%$ of $\theta _ { B A } \ '  $ s value on A and domain-B NLL to $1 0 9 . 9 \% \pm 1 5 \%$ of $\theta _ { B A } \mathrm { ^ { * } s }$ value on $B .$ The selectivity ratio |movement $_ { B } / \mathrm { m o v e m e n t } _ { A } | .$ , where movemen $\dot { } _ { \mathcal { D } }$ is the change in domain-D NLL from $\theta _ { A B }$ to the corrected endpoint, has median 1.04, mean $1 . 1 2 \ : ( \pm 0 . 3 2 )$ .

## AB AdamW augmented-state commutator: endpoint check

The Lie bracket structure of Sweeney [2026a] is proved for SGD; the Adam family’s per-coordinate normalization [Kingma and Ba, 2015] and AdamW’s decoupled weight decay [Loshchilov and Hutter, 2019] do not commute with the gradient-only commutator. We define an augmented-state commutator on the lifted state $\boldsymbol { z } ~ = ~ ( \theta , m , v , s )$ (parameters, first- and second-moment buffers, step counter s):

$$
\tau _ { \mathrm { A d a m } } = u ( g _ { A } ; z _ { 0 } ) + u ( g _ { B } ; z _ { A } ^ { * } ) - u ( g _ { B } ; z _ { 0 } ) - u ( g _ { A } ; z _ { B } ^ { * } ) ,
$$

where $u ( g ; z )$ is the AdamW parameter update produced by gradient g from optimizer state $z ,$ and $z _ { A } ^ { * }$ advances the optimizer buffers and step counter under $g _ { A }$ but freezes θ at $\theta _ { 0 }$ (and symmetrically for $z _ { B } ^ { * } )$ . The associated frozen-surrogate state pair $\hat { \theta } _ { A B } , \hat { \theta } _ { B A }$ satisfies $\hat { \theta } _ { A B } - \hat { \theta } _ { B A } = - \eta \tau _ { \mathrm { A d a m } } + O ( \eta ^ { 2 } )$ the AdamW analogue of Lemma $2 . 1 ^ { \circ } \mathrm { s } \ O ( \eta ^ { 2 } )$ SGD identity. Up to sign, η τ<sub>Adam</sub> is the first-order term of Theorem 1 in Sweeney [2026b] applied to the two orders AB and BA; that term comes from replaying the optimizer state with parameters frozen.

Fixed-clock regime (cited). Sweeney [2026b] characterizes the regime in which this check operates; we summarize and apply, but do not re-prove, two of its results:

• Lifted-state clock theorem (Sweeney, 2026b, Thm. 1). Fixed-β moment buffers and de-biasing counters advance with the step index rather than with the learning-rate-scaled time $\eta k ,$ , so the lifted-state step map is not a regular family $F _ { D } ^ { \eta } = I + \eta \bar { X _ { D } } + O ( \eta ^ { 2 } ) ;$ the buffers keep changing even as the parameter displacement is scaled down. For such fixed-clock state, including full frozen-gradient AdamW replay, reordering the same data (in particular, swapping $\bar { A } B$ for BA) changes the endpoint by η times the difference of the two frozen-parameter optimizer-state replays, plus $\dot { O } ( \eta ^ { 2 } )$ . The effect is therefore $\Theta ( \eta )$

whenever that difference is nonzero. For regular (memoryless) optimizers the first-order term cancels, and the leading effect is the ${ \cal O } ( \overline { { { \eta } } } ^ { 2 } )$ bracket.

• Matched-clock restoration (Sweeney, 2026b, §3 and App. C.6). For first-moment memory with a frozen coordinate preconditioner, matching both the memory and de-biasing clocks to the learning-rate-scaled time ηk $( \beta = e ^ { - a \bar { \eta } } , s _ { 0 } = T _ { c } / \eta .$ , with a and $T _ { c }$ fixed constants of that control) flattens the optimizer’s per-gradient position weights to an ${ \cal { O } } ( \eta )$ spread, restoring the regular $O ( \eta ^ { 2 } )$ scaling. The supporting experiment is a regularized first-moment control, not full AdamW. The matched clock is not itself a regular family, since $\beta$ and $s _ { 0 }$ depend on $\eta ;$ standard fine-tuning recipes use fixed-clock state.

Sweeney [2026b, §5 and App. C.4] measure this split in LoRA fine-tuning: under fixed-clock AdamW, the order effect has fitted η-slopes of 0.910 (Pythia-1B) and 0.915 (Llama-3.2-1B), compared with 1.960–2.005 for SGD and 1.968–1.973 for a matched-clock first-moment control. Adding only a fixed-β momentum buffer moves the slope from 1.988 (buffer-free SGD) to 0.998. The paper’s bracket-direct DPO edit and GRPO-style closure check (Appendices G and AF) are therefore restricted to SGD, where the bracket identity is exact at ${ \dot { O } } ( { \dot { \eta } } ^ { 2 } )$ ; the ${ \cal { O } } ( \eta )$ fixed-clock regime requires the lifted-state object $\tau _ { \mathrm { A d a m } }$

Midpoint scoring (Remark). The standard scalar score $\langle g _ { E } ( \theta _ { 0 } ) , \theta _ { A B } - \theta _ { B A } \rangle$ has $O ( \eta ^ { 2 } )$ error in fixed-β AdamW because the symmetric drift $\begin{array} { r } { \bar { d } : = \frac { 1 } { 2 } ( \theta _ { A B } + \bar { \theta _ { B A } } ) - \theta _ { 0 } } \end{array}$ is $O ( \eta )$ and the order split $\Delta _ { \mathrm { s p l i t } } : = \theta _ { A B } - \theta _ { B A }$ is ${ \cal { O } } ( \eta )$ , so the cross term $\bar { d } ^ { \top } H _ { E } \Delta _ { \mathrm { s p l i t } }$ contributes an $O ( \eta ^ { 2 } )$ residual. Scoring at the midpoint $\theta _ { m } : = { \textstyle \frac { 1 } { 2 } } ( \theta _ { A B } + \theta _ { B A } )$ instead gives

$$
{ \mathcal L } _ { E } ( \theta _ { A B } ) - { \mathcal L } _ { E } ( \theta _ { B A } ) = \langle \nabla { \mathcal L } _ { E } ( \theta _ { m } ) , \theta _ { A B } - \theta _ { B A } \rangle + O ( \eta ^ { 3 } ) ,
$$

because the second-order Taylor term cancels exactly by symmetry around $\theta _ { m }$ . This is the AdamW analogue of the $\theta _ { \mathrm { r e f } }$ drift-cancellation identity that makes Proposition 1 clean in the SGD regime. It is a loss-linearization identity at the observed endpoints; an $\dot { \cal O } ( \eta ^ { 3 } )$ AdamW correction step, which we leave to follow-up work, would also need an endpoint prediction accurate to that order.

Order-prediction transfer (cited). Sweeney [2026a, App. D.7] evaluates the unmodified SGDderived order predictor against AdamW ground truth over 1,020 Llama-3.2-1B trials. Sign accuracy is $4 7 . 7 \% , 5 1 . 3 \% , 5 \bar { 7 } . 1 \% .$ , and $4 6 . 7 \%$ at k=5, 10, 20, and 50 AdamW steps per domain; an exploratory Adam-aware augmented-state score reaches 59.6% at k=10. The SGD bracket is therefore at most weakly informative under AdamW, which is why the checks below use the lifted-state object $\tau _ { \mathrm { A d a m } }$ rather than the SGD commutator.

AdamW gradient-space check (our experiment). On Qwen-3-4B bf16 across six UltraFeedback source-pairs $( \eta = 1 0 ^ { - 4 } , \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ , weight decay $1 0 ^ { - 2 }$ , last-layer down/up/gate + lm\_head, 463M params, 20-step warmup (AdamW steps before the two source updates), 3 seeds each), the augmented-state readout is concentrated in the SGD-comparable gradient-space metric:

• Reweighted Gini (multiply each row by $\sqrt { \hat { v } }$ before computing Gini—this undoes AdamW’s per-coordinate rescaling and recovers the SGD-comparable metric): median 0.980 (bf16; range 0.977–0.982 across pairs); 0.989 on a fp32 control. Compare SGD’s gradient-space 0.97.

• Gradient-difference Gini (Gini of the lm\_head row of $g _ { A } - g _ { B }$ , a first-order proxy for the bracket direction in gradient-space row-norm structure only, not a parameter-space edit; cf. the first-order controls below, where applying $g _ { A } - g _ { B }$ as a parameter step at normmatched magnitude widens the ordering gap): mean 0.974. This is consistent with the row-norm structure that drives the SGD readout.

• Cross-pair top-1000 Jaccard: $\leq 0 . 0 5 1$ across all 15 pair×pair combinations (mean 0.024, min 0.000). Top tokens are pair-specific, matching the cross-pair Jaccard ≈ 0.004 of SFT τ (Section 5.2). The high-mass support is domain-specific, not universal, although the leading singular direction is shared across pairs (next item).

• Stable rank $( \| \cdot \| _ { F } ^ { 2 } / \| \cdot \| _ { 2 } ^ { 2 } )$ of the $\tau _ { \mathbf { A d a m } }$ subspace mean 6.08 across 18 pair-seed units (range 5.26–6.75); on a matched-row-L Gaussian random control the stable rank is 963, a ∼160× compression. The augmented-state commutator is low-rank, consistent with the cross-pair compositionality observed for SGD.

• Shared leading singular direction across pairs: pairwise cosine between leading singular directions has off-diagonal mean 0.996 across 306 entries of the $1 8 \times 1 8$ matrix (min 0.989, max 0.998); same-pair mean 0.997, different-pair mean 0.995. The leading bracket direction is largely shared; pair-specificity emerges in the rank-∼6 subspace beyond the leading direction. This is reported as a descriptive observation in this AdamW check.

• Raw lm\_head Gini (without $\sqrt { \hat { v } }$ reweighting): $0 . 4 9 \mathrm { - 0 . 5 5 - } b y$ AdamW design, not a sparsity loss. AdamW’s update is $\hat { m } / \sqrt { \hat { v } } ;$ ; the $1 / \sqrt { \hat { v } }$ factor intentionally rescales coordinates with high second-moment estimate, spreading the per-coordinate update magnitudes. The gradient-space Gini above removes that rescaling and shows the concentration metric in SGD-comparable coordinates.

• Top tokens (raw): "???", " Neither", " substance", " seasons", " Mostly" on $\mathbf { f a l s e _ { - } q a _ { - } v s _ { - } f l a n _ { - } v 2 _ { - } p 3 - a }$ mix of formatting and content tokens consistent with DPO’s chosen-vs-rejected structure.

Lifted-state endpoint closure (our experiment). On Qwen-2.5-1.5B fp32 with AdamW $( \beta _ { 1 } { = } 0 . 9 , \ \beta _ { 2 } { = } 0 . 9 \bar { 9 } 9$ , weight decay $1 0 ^ { - 2 } , \mathsf { \bar { \eta } } { = } 5 { \times } 1 0 ^ { - 5 }$ , 20-step warmup, last-layer MLP ∼ 41M params: gate\_proj, up\_proj, down\_proj on layer 27), the framework’s lifted-state object τ<sub>Adam</sub> predicts the AdamW endpoint difference in direction and magnitude across 6 source pairs × 3 seeds = 18 experiments. The closure edit $\theta _ { A B }  \theta _ { A B } + \lambda \eta \tau _ { \mathrm { A d a m } }$ (sign convention: $\theta _ { A B } - \theta _ { B A }$ ≈ $- \eta \tau _ { \mathrm { A d a m } } .$ so closure adds):

• Parameter cosine between $\theta _ { B A } - \theta _ { A B }$ and $\eta \tau _ { \mathrm { A d a m } } .$ mean 0.998 (the framework’s predicted closure direction matches the actual endpoint difference direction at near-unit cosine).

• Relative endpoint error at $\lambda { = } 1 \colon r _ { \mathrm { e n d } } : = \| \Delta _ { \mathrm { e n d } } - \eta \tau _ { \mathrm { A d a m } } \| / \| \Delta _ { \mathrm { e n d } } \|$ with $\Delta _ { \mathrm { e n d } } = \theta _ { B A } -$ $\theta _ { A B }$ has mean 0.068, i.e., the predicted edit leaves a residual of about 7% of the endpoint difference (the companion statistic $1 - r _ { \mathrm { e n d } } ^ { 2 }$ has mean 0.995).

• NLL gap closure at λ=1: mean ${ \bf 7 9 . 5 \% }$ , median 83.8%, 95% CI [72.6%, 85.9%].

• Sign-convention controls: random norm-matched edit −0.03% CI [−0.17%, 0.10%] (null); wrong-sign edit $\theta _ { A B } - \eta \tau _ { \mathrm { A d a m } }$ drives closure $\mathrm { t o - 9 6 . 6 \% C I \left[ - 1 0 2 \% , - 9 0 \% \right] ( - 2 \times }$ the gap, the predicted reflection).

• First-order baseline (our experiment): applying $g _ { A } - g _ { B }$ at the warmed-up state $\theta _ { \mathrm { w a r m } } .$ norm-matched to $\| \eta \tau _ { \mathrm { A d a m } } \|$ with parameter-space cos $\left( g _ { A } - g _ { B } , \tau _ { \mathrm { A d a m } } \right) = 0 . 4 3 ( \mathrm { s . c }$ d. 0.02) across all 18 pair-seed units, drives closure to −668% at $\lambda { = } 1$ (median $- 5 2 3 \%$ , CI $[ - 8 3 9 \% , - 5 0 \bar { 3 } \% ] ) ;$ the λ-scan is monotonic $( \lambda \mathrm { = } 0 . 5 \mathrm { : } - 2 5 2 \% , 1 . 5 \mathrm { : } - 1 0 8 8 \% , 2 . 0 \mathrm { : } - 1 5 1 1 \% )$ $\mathrm { ~ i ~ - ~ } \cos ^ { 2 } \approx \mathrm { ~ 0 . 8 \bar { 1 } ~ }$ of the first-order edit’s energy is orthogonal to $\tau _ { \mathrm { A d a m } } .$ , exciting highcurvature eval directions that wrong-sign (which stays along $\tau _ { \mathrm { A d a m } } )$ does not. The same gap-widening pattern under first-order norm-matched edits appears in the DPO (App. E) and GRPO-style (App. AF) controls.

• λ-scan for $\tau _ { \mathrm { A d a m } }$ ramps cleanly: $\lambda \mathrm { = } 0 \to 0 \% , \lambda \mathrm { = } 0 . 5 \to 4 7 \% , \lambda \mathrm { = } 1 \to 8 0 \% , \lambda \mathrm { = } 1 . 5 \to 4 4 \%$ (overshoot), λ=2 → −2% (full overshoot). Optimum at the predicted λ=1 closure step.

This endpoint closure check gives an AdamW analogue when the SGD bracket $b _ { A B }$ is replaced by the augmented-state commutator $\tau _ { \mathrm { A d a m } }$ on the lifted state ${ \boldsymbol { z } } = ( \theta , m , v , s )$ . The control hierarchy (random ≈ 0, wrong-sign $\approx - 1$ , first-order $\ll - 1 , \tau _ { \mathrm { A d a m } }  + 0 . 8 0 )$ mirrors the DPO and GRPOstyle pattern.

AdamW summary. The unmodified SGD bracket is at most weakly predictive of AdamW order (57.1% at k=20; Sweeney, 2026a, App. D.7), consistent with the fixed-clock regime above. Our AdamW experiments add a gradient-space concentration analogue on Qwen-3-4B (Gini 0.980, stable rank ${ \sim } 6 ,$ Jaccard $\leq 0 . 0 5 1 )$ and a lifted-state endpoint closure check on Qwen-2.5-1.5B fp32 (parameter cosine 0.998, relative endpoint error 0.068, NLL gap closure 79.5% [72.6%, 85.9%]). The main DPO edit and GRPO-style closure check use SGD, where the BCH identity is exact at $O ( \eta ^ { 2 } )$ ; an AdamW analogue of this edit with midpoint scoring is left to follow-up work.

Table 22: Continuation study, all 20 runs (Qwen-3-4B; SGD continuation on domain C). $M _ { 0 }$ is the initial trace on the predicted scale, and $R _ { h }$ is the signed fraction retained after h later updates. <sup>◦</sup>: the run does not retain the aligned trace at that horizon $( M _ { h } \le 0 . 0 1$ , including sign reversals). The two news/math runs do not start with an aligned trace $( M _ { 0 } \leq 0 . 0 1 )$ , so $R _ { h }$ is not reported for them. Rows above the middle rule are non-math pairs; rows below include math.
<table><tr><td>Pair (A, B)</td><td> $C$ </td><td>Seed</td><td> $M _ { 0 }$ </td><td> $R _ { 4 }$ </td><td> $R _ { 8 }$ </td><td> $R _ { 1 6 }$ </td></tr><tr><td>code/news</td><td>legal</td><td>8042</td><td>0.663</td><td>+0.286</td><td>+0.175</td><td>+0.134</td></tr><tr><td>code/news</td><td>legal</td><td>9042</td><td>0.530</td><td>+0.291</td><td>+0.219</td><td>+0.113</td></tr><tr><td>code/legal</td><td>news</td><td>8042</td><td>2.513</td><td>-0.433°</td><td>-0.309°</td><td>-0.170°</td></tr><tr><td>code/legal</td><td>news</td><td>9042 8042</td><td>2.337 0.721</td><td>-0.348°</td><td>-0.257°</td><td>-0.157°</td></tr><tr><td>code/biomedical</td><td>news</td><td>9042</td><td>0.577</td><td>+0.583</td><td>+0.446</td><td>+0.292</td></tr><tr><td>code/biomedical</td><td>news</td><td>8042</td><td>1.221</td><td>+0.576</td><td>+0.439</td><td>+0.283</td></tr><tr><td>news/legal</td><td>code</td><td>9042</td><td>1.121</td><td>+0.220</td><td>+0.144</td><td>+0.063</td></tr><tr><td>news/legal news/biomedical</td><td>code</td><td>8042</td><td>0.599</td><td>+0.191 +0.289</td><td>+0.121 +0.248</td><td>+0.055</td></tr><tr><td>news/biomedical</td><td>math</td><td>9042</td><td>0.555</td><td>+0.365</td><td>+0.339</td><td>+0.217</td></tr><tr><td>legal/biomedical</td><td>math</td><td>8042</td><td>2.023</td><td>-0.010°</td><td>-0.015°</td><td>+0.295 -0.015°</td></tr><tr><td>legal/biomedical</td><td>math math</td><td>9042</td><td>2.200</td><td>+0.027</td><td>+0.030</td><td>+0.053</td></tr><tr><td></td><td></td><td>8042</td><td>0.275</td><td>+0.197</td><td></td><td></td></tr><tr><td>code/math code/math</td><td>legal legal</td><td>9042</td><td>0.033</td><td>+0.502</td><td>+0.145 +0.016°</td><td>+0.086</td></tr><tr><td>news/math</td><td>biomedical</td><td>8042</td><td>0.005</td><td></td><td></td><td>-0.170°</td></tr><tr><td>news/math</td><td>biomedical</td><td>9042</td><td>-0.006</td><td></td><td></td><td></td></tr><tr><td>legal/math</td><td>biomedical</td><td>8042</td><td>0.557</td><td>+0.190</td><td>+0.093</td><td></td></tr><tr><td>legal/math</td><td>biomedical</td><td>9042</td><td>0.436</td><td>+0.252</td><td>+0.094</td><td>+0.022</td></tr><tr><td>biomedical/math</td><td>code</td><td>8042</td><td>0.158</td><td>+0.455</td><td></td><td>+0.022°</td></tr><tr><td>biomedical/math</td><td></td><td>9042</td><td></td><td></td><td>+0.428</td><td>+0.287</td></tr><tr><td></td><td>code</td><td></td><td>0.081</td><td>+0.433</td><td>+0.327</td><td>+0.301</td></tr><tr><td></td><td></td><td></td><td>18/20</td><td></td><td></td><td></td></tr><tr><td colspan="3">Runs with an aligned trace Median  $R _ { h }$  over the 18 runs that start aligned</td><td></td><td>15/20 0.269</td><td>14/20</td><td>13/20</td></tr></table>

Implementation. The supplementary code includes src/lie\_fisher/adamw\_commutator.py, with tests for agreement against torch.optim.AdamW on single steps and for the closure-edit sign convention. AdamW correction during training remains follow-up work.

## AC Continuation study: retention and attenuation

Protocol. This study measures how long the order-specific parameter component survives later training. It uses Qwen-3-4B at the step size of the main SFT runs $( \eta = 6 \hat { . } 4 2 \times 1 0 ^ { - 4 } )$ , training the last-block MLP and the language-model head. It covers all ten source pairs over {code, news, legal, biomedical, math} (the six pairs over {code, news, legal, biomedical} and the four pairs that include math), each with two new seeds, for 20 runs. After the two source updates, the endpoints $\theta _ { A B }$ and $\theta _ { B A }$ are both continued with SGD on a third domain $C \notin \{ A , B \}$ (Table 22), using the same sixteen disjoint 50-record blocks in the same order, with gradients recomputed at each branch’s current parameters. With $D _ { h } = \theta _ { A B } ^ { ( h ) } - \theta _ { B A } ^ { ( h ) }$ after h later updates, we track

$$
M _ { h } = \frac { \langle D _ { h } , b _ { A B } \rangle } { \eta ^ { 2 } \vert \vert b _ { A B } \vert \vert ^ { 2 } } , \qquad R _ { h } = \frac { \langle D _ { h } , b _ { A B } \rangle } { \langle D _ { 0 } , b _ { A B } \rangle } .
$$

$M _ { h }$ is the trace on the scale of the predicted initial effect, and $R _ { h }$ is the signed fraction of the initial trace that remains. A run starts with an aligned trace if $M _ { 0 } > 0 . 0 1$ , and it retains the aligned trace at horizon h if also $M _ { h } > 0 . 0 1$ . The criterion is one-sided: a run whose projection changes sign does not retain the aligned trace, even when the reversed component is large. No run is excluded from any count.

Pass rule. The rule was fixed before the runs. A horizon passes only if all of the following hold: (i) at least 15 of the 20 runs retain the aligned trace, including at least 9 of the 12 non-math runs and 6 of the 8 math-paired runs; (ii) at least 8 of the 10 pairs have at least one seed that retains it; (iii) the median $R _ { h }$ over runs that start aligned is positive; and (iv) the continuation trains on $C { : }$ at least 32 of the 40 branches lower their loss on C while moving at least as far as the original source updates (i.e., for each branch $r \in \{ A B , B A \} , \| \theta _ { r } ^ { ( h ) } - \theta _ { r } ^ { ( 0 ) } \|$ is at least the root-mean-square norm of the two source updates), and the median relative drop in C loss is at least 0.1%.

Results. Table 22 lists every run. Eighteen of the 20 runs start with an aligned trace; the two news-vs-math runs do not. The rule is met after four later updates $( 1 5 / 2 0$ runs; 9/12 non-math and 6/8 math-paired) but not after eight $( 1 4 / 2 0 ; 5 / 8$ math-paired) or sixteen (13/20; 4/8 math-paired). The non-math count stays at $9 / \bar { 1 } 2$ throughout, so both failures come from the math-paired runs. Condition (iv) holds at every horizon: all 40 branches lower their loss on C; 35, 40, and 40 of them also meet the displacement condition; and the median relative drop in C loss is 2.2%, 3.1%, and 4.1%. Over the 18 runs that start aligned, the median $R _ { h }$ falls from 0.27 to 0.14 to 0.07 after four, eight, and sixteen later updates. Both code-vs-legal runs reverse sign within four updates; they fail the one-sided criterion but keep a large reversed component $( M _ { 4 } = ^ { ^ { - } } 1 . 0 9 \mathrm { a n d } - 0 . { \bar { 8 } } 1 )$ . The aligned trace therefore persists over a short horizon and then attenuates in the median, with sign reversals in some runs. This does not establish behavioral retrieval or long-term memory.

## AD Cost accounting and cheaper curvature approximations

Exact matrix-free cost. The exact calculation never forms or stores a Hessian. For one source pair it computes two gradients and two Hessian-vector products $( \mathrm { H V P s } ) { \ ; }$ with each HVP costing roughly two forward–backward passes (Appendix AE), the bracket costs about six forward–backward equivalents for the reported parameter block. The scalar score requires one additional held-out gradient, and the token readout adds three forward-only passes. The number of passes per pair is fixed, although the cost of each pass grows with model size, sequence length, sample count, and the number of parameters included. The paper applies this exact calculation to the reported parameter blocks at model sizes through 8B.

Exact reuse across pairs. Suppose N candidate sources share a base model $\theta _ { 0 }$ and a held-out evaluation loss, with $g _ { D } , H _ { D } ,$ , and $g _ { E }$ evaluated at $\theta _ { 0 }$ . By Hessian symmetry, the leading predicted gap for pair (A, B) is $\eta ^ { 2 } [ g _ { A } ^ { \top } ( H _ { B } { \ ' } g _ { E } ) - g _ { B } ^ { \top } ( H _ { A } g _ { E } ) ]$ ]. Each $H _ { D } g _ { E }$ is therefore computed once per source, so ranking all pairs among N sources needs N HVPs instead of $N ( N - 1 )$ , plus inexpensive dot products. This is the identity behind the Lie-Bracket Tournament of Sweeney [2026a, §3.1], and it is exact for the fixed-base leading-order score. On the Qwen-3-4B code-vs-news test, reused and direct scores agree to a maximum relative error of $1 . 6 \times 1 0 ^ { - 5 }$ . If different pairs use different evaluation losses, reuse applies only within groups sharing the same loss. The paper’s primary score instead evaluates the held-out gradient at the pair-specific reference $\theta _ { 0 } - \eta ( g _ { A } + g _ { B } ) \quad$ ; replacing it by the base-model gradient changes the predicted gap at $O ( \eta ^ { 3 } )$ [Sweeney, 2026a, Remark 2.6], the order already neglected by the second-order approximation. The fixed-base formula can therefore screen all pairs cheaply; the pair-specific score, exact matrix-free HVPs, and token readout are then computed pair by pair.

Cheaper curvature approximations. A second route avoids HVPs with a streamed diagonal curvature approximation. For the final vocabulary-prediction layer, with hidden states held fixed, the layer’s own contribution to the Hessian diagonal at weight $( k , j )$ can be accumulated in one pass from $p _ { k } ( 1 - p _ { k } ) h _ { j } ^ { 2 }$ , where $p _ { k }$ is the probability of token k and $h _ { j }$ is hidden-state coordinate j. This discards cross-coordinate interactions. Block-diagonal or low-rank generalized Gauss–Newton approximations [Martens and Grosse, 2015] retain more of those interactions and provide intermediate cost–fidelity trade-offs, but are not automatically HVP-free. The relevant fidelity tests are whether an approximation preserves held-out ordering signs and the largest token effects. All reported pairspecific scores and token readouts use exact matrix-free HVPs.

## AE Reproducibility

The supplementary materials include source code, protocol definitions, and result files for reported claims. Run IDs in the paper are stable identifiers; artifact locations are documented in the supplement and need not correspond one-to-one to local directory names. The src/lie\_fisher/ package contains the bracket, attribution, and intervention machinery; src/lie\_fisher/protocols.py contains the frozen configuration records (Protocol dataclasses) used by the reported intervention protocols.

## AE.1 Software environment

• Reported runs used CUDA/PyTorch/Transformers environments recorded in released run metadata; requirements.txt gives the reproduction baseline. The run envelope is Python 3.11–3.12, PyTorch 2.x, Transformers 4/5 depending on run, NumPy 1.26–2.x, and matplotlib 3.x.

• One CUDA backend per run (no multi-GPU sharding for the reported experiments). Float32 is enforced for bracket/HVP/readout paths that require the fp32 precision policy in Appendix C.

• Seed control: each experiment locks {Python, NumPy, PyTorch CPU, PyTorch CUDA} seeds at the value reported in run\_meta.base\_seed of the corresponding result JSON. Per-trial seeds are derived deterministically from base\_seed via the experiment’s protocol.

## AE.2 Models and dataset revisions

• Qwen-3-4B (Qwen/Qwen3-4B, 4B params); Qwen-3-8B for DPO concentration checks; Qwen-2.5-1.5B (Qwen/Qwen2.5-1.5B, 1.5B params); Qwen-2.5-0.5B-Instruct for the GRPO-style surrogate; Llama-3.1-8B (meta-llama/Llama-3.1-8B, 8B params); Llama-3.2-1B (meta-llama/Llama-3.2-1B, 1B params). All models loaded via transformers.AutoModelForCausalLM with trust\_remote\_code=False.

• SFT training/eval data: The Pile [Gao et al., 2020], six domain partitions: code (Stack-Exchange), news (CC-News), legal (FreeLaw), biomedical (PubMed Central), math (StackExchange Math), wikipedia. Section 4 lists the domain pairs used by each SFT experiment; the continuation study uses the ten pairs over {code, news, legal, biomedical, math} (Appendix AC). Eval offset = 50 documents past the training partition for held-out evaluation (Section 4).

• DPO training/eval data: UltraFeedback [Cui et al., 2024], source partitions: false\_qa, flan\_v2\_p3, sharegpt, ultrachat for the original six-pair grid; evol\_instruct, truthful\_qa for the fresh-pairs replication (App. N). The token-reweighting fresh-pair check uses the three source pairs evaluable under the released snapshot; the matched-batch fresh-pairs grid uses a pair-disjoint source-pair list.

## AE.3 Compute budget and hardware

The main per-experiment cost is HVP computation (each Hessian-vector product is roughly 2× a forward-backward pass; the bracket requires two HVPs). The hardware table is indicative; released result files correspond to the reported protocols.

## AE.4 Fixed protocols and conflict checks

Each intervention reported in the paper has a corresponding frozen configuration record (Protocol dataclass) in src/lie\_fisher/protocols.py that fixes {model, dtype, η, k, batch composition, dose-match policy, identity-check tolerance, top-K filter, control set}. The runner aborts when the protocol’s immutable fields conflict with command-line overrides. Released run\_meta blocks record the active protocols for the final reported runs.

## AE.5 Repository pointers

• Bracket and HVP machinery: src/lie\_fisher/lie.py (HVP via torch.autograd.grad on the gradient inner product, in-module).

• Per-token attribution: src/lie\_fisher/head\_moments.py (SFT) and compute\_token\_attribution\_dpo (corrected DPO τ).

• Winsorized aggregation (5/95 mean + bootstrap CI): src/lie\_fisher/stats.py:robust\_aggregate.

• η calibration (cube-root regime): src/lie\_fisher/eta\_autopilot.py.

• Protocol definitions: src/lie\_fisher/protocols.py.

Table 23: Indicative GPU-hours and hardware per experiment family. Total project compute is driven primarily by the DPO matched-batch runs and the endpoint-correction runs; sparsity/baseline runs are lower-cost. The rows sum to about 32 GPU-hours; the project total is cumulative over all runs. Run identifiers are listed under Repository pointers below.
<table><tr><td>Experiment family</td><td>Hardware</td><td>GPU- hours</td><td>Notes</td></tr><tr><td>Sparsity (4 models)</td><td>RTX 4090 / Pro 6000</td><td>~2</td><td>R8b, R9, R13_sparsity, R14_sparsity, R15_sparsity, R16</td></tr><tr><td rowspan="2">Baselines (4 models) SFT interventions</td><td>RTX 4090 / A40</td><td>~1</td><td>R8b, R9, R13</td></tr><tr><td>RTX Pro 6000 RTX 4090</td><td>~3</td><td>R14a, R14b, R15 (full fp32 for Qwen-2.5)</td></tr><tr><td>Endpoint correction / multi-step</td><td>A40  /  L40  / Pro 6000</td><td>~5</td><td>R26, R27, R28, R19, R20</td></tr><tr><td>Iterative bracket cor- rection</td><td>L40</td><td>~3</td><td>R29, R30 (instrumented per-position NLL)</td></tr><tr><td>DPO sparsity (3 mod- els)</td><td>A40 / Pro 6000</td><td>~2</td><td>R50, R52, R60</td></tr><tr><td>DPO token reweight- ing</td><td>RTX Pro 6000 WS</td><td>~3</td><td>R78: 6 original pairs + three-pair fresh check, 3 seeds × 20 trials</td></tr><tr><td>DPO matched-batch correction</td><td>RTX Pro 6000 WS / H100 NVL</td><td>~10</td><td>R79: 6 original pairs (3 seeds × 10 trials) +</td></tr><tr><td>Frozen-rollout GRPO- style</td><td>A100 80GB</td><td>~2</td><td>6 disjoint fresh pairs (3 seeds × 10) R81/R82 matched-rollout checks</td></tr><tr><td>AdamW endpoint clo- sure</td><td>RTX-class RTX 4090 24GB</td><td>~1</td><td>R86 + R87 first-order baseline</td></tr><tr><td>Project total (cumula- tive)</td><td>mixed academic GPUs</td><td>~48</td><td>approximately USD 50–55 in cumulative spot-rental cost</td></tr></table>

• Run identifiers (used in the compute table above and as directory names in the archive): R8b (Qwen-3-4B main run: sparsity, interpretability, baselines, Magnus check), R9 (Qwen-2.5-1.5B main run), R13/R16 (Llama runs), R8c/R10 (earlier Qwen-3-4B intervention sweeps), R14a/R14b (Qwen-3-4B held-out SFT interventions), R15 (Qwen-2.5-1.5B fp32 SFT interventions), R19/R20 (multistep persistence), R26/R28 (endpoint correction on Qwen-2.5-1.5B/Qwen-3-4B), R27 (triplet additivity), R29/R30 (iterative correction; R30 instrumented per position), R44 (follow-up $\theta _ { 0 }$ -reference readouts), R50/R52/R60 (DPO concentration), R71 (the original six DPO source pairs), R78 (DPO token reweighting with the corrected τ), R79 (matched-batch DPO correction), R80 (AdamW gradientspace check), R81/R82 (GRPO-style single-step and block-sequential checks), R86/R87 (AdamW lifted-state closure and first-order baseline).

## AF Frozen-rollout GRPO-style test: protocol and controls

Protocol. These experiments test whether the BCH-local bracket remains structured in a frozenrollout, KL-regularized reward-surrogate setting inspired by GRPO/PPO. We use Qwen-2.5-0.5B-Instruct, freeze the sampled rollouts shared by the two orders, and define two analytic reward sources: math correctness and brevity/format compliance. The bracket $b _ { A B }$ is computed at $\theta _ { 0 }$ on the realized rollouts, and endpoints $\theta _ { A B } , \theta _ { B A }$ are trained on the matched rollout records. This is a matched-rollout, KL-regularized GRPO-style surrogate with analytic rewards, using the GRPO/PPO objective family as protocol motivation [Shao et al., 2024, Schulman et al., 2017].

Single-step matched-rollout BCH check. Across 20 trials in 5 seed clusters, the endpointidentity check is tight: the cosine between the measured $\theta _ { A B } - \theta _ { B A }$ and $\eta ^ { 2 } b _ { A B }$ (identity cosine) has median 0.999 (mean 0.998, CI [0.996, 0.999]), and the projection coefficient $\zeta ~ = ~ \langle \theta _ { A B } ~ -$ $\theta _ { B A } , \eta ^ { 2 } b _ { A B } \rangle / \| \eta ^ { 2 } b _ { A B } \| ^ { 2 }$ has median 1.002 (mean 1.002, CI [0.995, 1.009]). The finite-difference HVP checks have minimum HVP cosine median 0.999998 and maximum residual median 0.0026; PPO/GRPO clipping is inactive in both orders. All checks pass together in 17/20 trials (0.85).

Table 24: Single-step frozen-rollout GRPO-style controls. Values are medians over 20 trials unless a CI is shown. Unmatched rollouts: the bracket from one rollout set applied to endpoints trained on another. Reported r are closure ratios r := 1 − ∥edited − target $\| / \|$ baseline − target∥ against the target AB-vs-BA displacement on held-out rollouts.
<table><tr><td>Condition</td><td>Parameter closure r</td><td>Log-prob closure r</td><td>Loss closure r</td></tr><tr><td>Bracket edit</td><td>0.952</td><td>0.886</td><td>0.805</td></tr><tr><td>Random norm-matched</td><td>-0.414</td><td>-0.001</td><td>0.010</td></tr><tr><td>Wrong sign</td><td>-0.997</td><td>-0.987</td><td>-1.008</td></tr><tr><td>First-order norm-matched</td><td>-0.410</td><td>-0.197</td><td>-2.384</td></tr><tr><td>Unmatched rollouts (negative control)</td><td>-0.396</td><td>-0.286</td><td>-0.158</td></tr></table>

Block-sequential T-sweep. The block-sequential sweep composes the correction over longer matched-rollout blocks and selects the largest T in the fixed grid {2, 4, 8, 16} satisfying identity cosine $> 0 . 7 , \zeta \in [ 0 . 5 , 1 . 5 ]$ , and parameter-space closure $C _ { \theta } \stackrel { - } { > } 0 . 6 \dot { }$ , without reward-outcome tuning. All evaluable block lengths pass; $T ^ { \star } = 1 6$ is selected by this rule.

Table 25: Block-sequential frozen-rollout T-sweep. $C _ { \theta }$ is the parameter-space closure of the composed correction.
<table><tr><td> $T$ </td><td>identity cosine</td><td>ζ</td><td>relative residual</td><td> $C _ { \theta }$ </td></tr><tr><td>2</td><td>0.997</td><td>0.967</td><td>0.083</td><td>0.917</td></tr><tr><td>4</td><td>0.998</td><td>0.904</td><td>0.122</td><td>0.878</td></tr><tr><td>8</td><td>0.988</td><td>0.818</td><td>0.267</td><td>0.733</td></tr><tr><td>16</td><td>0.991</td><td>0.884</td><td>0.188</td><td>0.812</td></tr></table>

Interpretation. The GRPO-style result is strongest as a closure check: in a third objective regime, the same bracket satisfies a near-exact matched-path identity, wrong-sign edits obey antisymmetry, random controls stay near zero in log-probability and loss, and static cross-rollout controls fail. The caveat is also structural: a bracket computed for one realized path is not expected to transfer to a different path.