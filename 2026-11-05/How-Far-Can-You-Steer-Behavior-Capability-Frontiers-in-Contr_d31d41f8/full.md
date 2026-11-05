# How Far Can You Steer? Behavior-Capability Frontiers in Contrastive Activation Addition

Deepanshu Raj The University of Chicago deepraj@uchicago.edu

## Abstract

Activation steering can modify language-model behavior without parameter updates, but behavioral efficacy alone does not determine whether an intervention is usable. Contrastive Activation Addition (CAA) steers targeted behaviors using activation directions derived from contrasting examples, potentially degrading unrelated model capabilities in the process. We therefore study CAA through a behavior– capability frontier, selecting interventions that maximize behavioral gain subject to an explicit bound on capability degradation. We evaluate this framework across four instructiontuned models, seven behaviors, and capabilityloss budgets of 1%, 3%, and 5% (absolute percentage points). Validation-selected behavioral effects generalize strongly to held-out data (ρ = 0.945), while joint behavior-capability survival among nonzero operating points rises from 37.5% at the 1% budget to 72.7% at 3%, with limited additional improvement at 5%. Control experiments indicate that these effects depend on learned behavior directions rather than generic activation perturbations. Representational geometry is significantly but only modestly predictive of causal cross-behavior transfer. Finally, secondary capability, out-ofdistribution, and open-ended evaluations show that interventions satisfying protected benchmarks can still degrade untested capabilities or transfer unevenly to free-form generation. These results characterize activation steering as a constrained intervention whose usable range depends on model, behavior, direction, layer, and evaluation budget, rather than a universal control knob.

## 1 Introduction

Activation steering offers a direct way to modify language-model behavior without updating model parameters. A direction associated with a behavior is identified in activation space and then added or subtracted during inference (Turner et al., 2023;

Rimsky et al., 2024). The intervention is simple, but its evaluation is not. A steering vector may move the intended behavior while also damaging capabilities that were not meant to change, and both effects can vary sharply with layer and intervention strength. The practical question is therefore not only whether a model can be steered, but how far it can be steered before the intervention becomes too costly.

We study this question using Contrastive Activation Addition (CAA) (Rimsky et al., 2024). For behavior b and decoder layer l, CAA constructs a direction from matched behavior-positive and behavior-negative examples,

$$
\mathbf { v } _ { b , \ell } = \frac { 1 } { N _ { b } } \sum _ { i = 1 } ^ { N _ { b } } \left( \mathbf { h } _ { i , \ell } ^ { + } - \mathbf { h } _ { i , \ell } ^ { - } \right)\tag{1}
$$

and intervenes on the residual stream as

$$
\begin{array} { r } { \mathbf { x } _ { \ell , t } ^ { \prime } = \mathbf { x } _ { \ell , t } + \alpha \widetilde { \mathbf { v } } _ { b , \ell } } \end{array}\tag{2}
$$

where $\widetilde { \mathbf { v } } _ { b , \ell }$ denotes the normalized steering direction defined in Section 3.2. Here l determines where the intervention enters the computation and α determines its sign and strength. Even for a fixed behavior, steering is therefore a joint problem over layer and coefficient rather than a single scalar control.

Most activation-steering evaluations emphasize behavioral efficacy: whether a chosen direction changes the target behavior, often under a small set of intervention strengths. This establishes steerability, but not a usable operating regime. A coefficient selected for a large target shift may incur unacceptable capability loss, while a setting that looks safe on the data used for selection may fail after that choice is frozen. Recent work has highlighted reliability, selectivity, and capability preservation as central concerns in activation interventions (Tan et al., 2024; Braun et al., 2025; Sprejer et al., 2026;

Dang and Ngo, 2026). These issues motivate treating steering as a constrained selection problem, with explicit capability-loss budgets and separate evaluations of reliability and selectivity through held-out testing and direction-specific controls, respectively.

We formalize this view as a behavior-capability frontier. For each behavior and steering direction, we seek the intervention that maximizes behavioral gain while limiting degradation across protected capability benchmarks. We define a conservative capability margin using lower confidence bounds on benchmark performance changes. An intervention is considered feasible when this margin satisfies a specified capability-loss budget ε. The frontier then identifies the feasible intervention with the largest directional behavioral gain.

A central part of our design is the separation between selection and confirmation. Behavioral data are first used to construct CAA directions and select an intervention layer. Capability measurements are consulted only after the layer is fixed, when choosing a coefficient under the constraint above. The resulting operating point is then frozen. Held-out behavioral evaluation, capability holdout and later stress tests cannot trigger reselection. This makes the final evaluation a test of a fixed claim rather than another stage of tuning.

We apply this protocol to four instruction-tuned models spanning three model families: Llama 2 7B Chat, Llama 3.1 8B Instruct, Mistral 7B Instruct v0.3, and Qwen2.5 7B Instruct (Touvron et al., 2023; Grattafiori et al., 2024; Jiang et al., 2023; Yang et al., 2024), and study seven behavioral dimensions: sycophancy, refusal, hallucination, corrigibility, survival instinct, myopic reward, and AI coordination. Layer selection uses behavioral validation data only; coefficient selection protects MMLU, ARC-Challenge, and HellaSwag under budgets $\varepsilon \in \{ 0 . 0 1 , 0 . 0 3 , 0 . 0 5 \}$ Frozen operating points are then evaluated on independent behavioral and capability holdouts, followed by specificity controls, cross-behavior causal-transfer experiments, secondary-capability tests, out-ofdistribution evaluation, and open-ended generation.

Three results characterize the usable steering regime. First, validation-selected behavioral gains transfer strongly to held-out data (Spearman $\rho =$ 0.945). Second, capability preservation is much more sensitive to the operating budget: joint heldout survival among nonzero operating points rises from 37.5% at a 1% budget to 72.7% at 3%, with only a modest increase to 76.8% at 5%. Third, the effect is specific but not universally safe. CAA substantially outperforms norm-matched, labelshuffled, and off-target-KL-matched controls, yet frozen interventions can still degrade capabilities outside the protected suite and need not transfer cleanly to open-ended generation.

These results suggest that activation steering is better described by an operating frontier than by a universal control knob. Its usable range depends on the behavior, direction, layer, model, and capability budget, and it must be evaluated after the intervention has been fixed.

## 2 Background and Related Work

## 2.1 Contrastive Activation Steering

Activation steering treats intermediate representations as objects that can be manipulated at inference time. Representation Engineering frames high-level model properties in terms of populationlevel representations (Zou et al., 2023). Related work has identified compact activation-space directions for truthfulness, task transformations, and style (Li et al., 2023; Todd et al., 2024; Konen et al., 2024). Activation Addition constructs such directions from contrasting prompts (Turner et al., 2023), while CAA averages activation differences over many positive and negative examples to obtain behavior-level directions (Rimsky et al., 2024).

Subsequent methods modify either the intervention strength or the space in which steering is applied. Activation scaling studies continuous control through direction magnitude (Stoehr et al., 2024); dynamic composition studies simultaneous properties (Scalena et al., 2024). More recent methods emphasize selectivity. Selective Steering uses norm-preserving rotations and discriminative layer selection (Dang and Ngo, 2026); sparse mediation restricts interventions to putative causal subspaces (Doan et al., 2026); and GAPS reports improved behavior-capability tradeoffs through dimensionlevel gating (Fereidouni et al., 2026). We keep CAA fixed and study how its usable operating range should be selected and validated.

## 2.2 Reliability, Specificity, and Capability Preservation

Steering effects can vary substantially across prompts, concepts, and models (Tan et al., 2024; Da Silva et al., 2025). Their reliability has also been linked to the geometry and class separation of the underlying contrastive activations (Braun et al., 2025). Capability loss forms a separate concern. CAA evaluates general capabilities under intervention (Rimsky et al., 2024), while later work documents losses in task accuracy and output quality and motivates explicit preservation constraints (Sprejer et al., 2026; Xu et al., 2026). Related approaches control output shift directly, for example through KL-constrained interventions (Azizi et al. 2026).

A behavioral change also need not be specific to the learned semantic direction. FishBack studies off-target output distortion through a Fisher-based geometry (Wang et al., 2026), and SteerCheck compares steering directions with controls matched by off-target KL divergence (Luo et al., 2026). We use norm-matched, label-shuffled, and off-target-KLmatched controls as attribution checks at frozen operating points.

SteerScope (Yang et al., 2026) examines efficacy-side-effect trade-offs, while AxBench (Wu et al., 2025) benchmarks steering methods. We instead study capability-constrained CAA selection with independent held-out confirmation.

## 2.3 Representation Geometry and Causal Interaction

Behavior directions can exhibit structured relationships in activation space. Refusal, for example, has been associated with a low-dimensional residualstream direction whose manipulation changes refusal behavior (Arditi et al., 2024). Geometry has also been connected to steering reliability and safety-related overlap among representations (Braun et al., 2025; Li et al., 2026).

Our geometry analysis asks a different question: whether the relationship between two independently extracted behavior directions predicts the causal effect of steering one behavior while measuring another. We therefore separate representational alignment, measured by cosine similarity, from causal cross-behavior transfer, measured by explicit intervention.

## 3 Behavior-Capability Frontiers

We study steering over behavior b, decoder layer l, and signed coefficient α. For each behavior, we first construct layer-wise CAA directions and select an intervention layer using behavioral validation alone. At that fixed layer, we choose the coefficient that maximizes directional behavioral gain subject to explicit capability constraints. The resulting operating point is frozen before held-out evaluation. Figure 1 summarizes the protocol.

## 3.1 Contrastive Behavior Directions

For each behavior b, the vector-construction split $\mathcal { D } _ { b } ^ { \mathrm { v e c } } = \{ ( x _ { i } , y _ { i } ^ { + } , y _ { i } ^ { - } ) \} _ { i = 1 } ^ { N _ { b } }$ contains a prompt and two semantically opposed answers. Let $\mathbf { h } _ { i , \ell } ^ { + }$ and $\mathbf { h } _ { i , \ell } ^ { - }$ denote residual-stream activations after block l at the answer-bearing position. Following CAA (Rimsky et al., 2024), the raw direction is

$$
\mathbf { v } _ { b , \ell } = \frac { 1 } { N _ { b } } \sum _ { i = 1 } ^ { N _ { b } } \left( \mathbf { h } _ { i , \ell } ^ { + } - \mathbf { h } _ { i , \ell } ^ { - } \right)\tag{3}
$$

Vectors are constructed independently at every layer using only $\mathcal { D } _ { b } ^ { \mathrm { v e c } }$ . Validation, capability, and test examples are excluded. Model-specific rendering and extraction boundaries are described in Appendix A.1.

## 3.2 Normalization and Relative Intervention Strength

Raw CAA norms vary across behaviors and layers. To keep the meaning of α comparable within a layer, we define the mean behavior-vector norm

$$
m _ { \ell } = \frac { 1 } { | \boldsymbol { B } | } \sum _ { b \in \boldsymbol { B } } \| \mathbf { v } _ { b , \ell } \| _ { 2 }\tag{4}
$$

and rescale each direction to that norm,

$$
\widetilde { \mathbf { v } } _ { b , \ell } = \frac { m _ { \ell } } { \vert \vert \mathbf { v } _ { b , \ell } \vert \vert _ { 2 } } \mathbf { v } _ { b , \ell }\tag{5}
$$

Steering then modifies the residual stream as

$$
\mathbf { x } _ { \ell , t } ^ { \prime } = \mathbf { x } _ { \ell , t } + \alpha \widetilde { \mathbf { v } } _ { b , \ell } , \qquad t \in { \mathcal { T } } _ { \mathrm { s t e e r } }\tag{6}
$$

where $\tau _ { \mathrm { { s t e e r } } }$ denotes the model-specific token positions over which steering is applied. Only the added vector is normalized; the residual state itself is not rescaled.

Equalizing vector norms within a layer does not, however, make a raw coefficient α directly comparable across layers or models. The scale of the residual stream itself changes with depth and architecture. We therefore estimate the baseline residual scale at layer l as

$$
r _ { \ell } = \mathbb { E } \left[ \| \mathbf { x } _ { \ell , t } \| _ { 2 } \right]\tag{7}
$$

where the expectation is taken over unsteered residual states from the calibration data. Because

$$
\| \alpha \widetilde { \mathbf { v } } _ { b , \ell } \| _ { 2 } = | \alpha | m _ { \ell }
$$

![](images/44efadf2427114f8b7c2781f77972c12ea4410fe940ba831286021e8f68e4c95.jpg)  
Figure 1: Behavior-Capability frontier protocol. (A) Contrastive examples define layer-wise CAA directions, after which the intervention layer is selected using behavioral validation. (B) At the fixed layer, coefficients are evaluated for directional behavioral gain subject to simultaneous capability constraints. The feasible coefficient with the largest directional behavioral gain defines the operating point. (C) The operating point is frozen before independent behavioral and capability confirmation and all post-selection audits.

we define the dimensionless relative perturbation strength

$$
\eta ( \boldsymbol { \alpha } , \ell ) = \frac { \| \boldsymbol { \alpha } \widetilde { \mathbf { v } } _ { b , \ell } \| _ { 2 } } { r _ { \ell } } = | \boldsymbol { \alpha } | \frac { m _ { \ell } } { r _ { \ell } }\tag{8}
$$

For visualizing signed interventions, we additionally define $\eta _ { \mathrm { s } } ( \alpha , \ell ) = \alpha m _ { \ell } / r _ { \ell }$ , which retains the sign of the steering coefficient.

The coefficient α therefore specifies the signed intervention used by the model, whereas η measures its magnitude relative to the typical residualstream scale at the intervention layer. We use α to parameterize the steering sweeps and report η alongside the resulting operating points so that intervention strength can be interpreted across layers and model families.

## 3.3 Behavioral Response and Layer Selection

Layer selection uses the behavioral validation split. For example i, let $z _ { i } ^ { + }$ and $z _ { i } ^ { - }$ be the logits assigned to the behavior-matching and behavior-opposed alternatives under intervention (α, l). We renormalize over the two choices,

$$
\pi _ { i } ^ { b } ( \alpha , \ell ) = \frac { e ^ { z _ { i } ^ { + } } } { e ^ { z _ { i } ^ { + } } + e ^ { z _ { i } ^ { - } } }\tag{9}
$$

and define

$$
B _ { b } ( \alpha , \ell ) = { \frac { 1 } { M _ { b } } } \sum _ { i } \pi _ { i } ^ { b } ( \alpha , \ell )\tag{10}
$$

$$
\Delta B _ { b } ( \alpha , \ell ) = B _ { b } ( \alpha , \ell ) - B _ { b } ( 0 , \ell )\tag{11}
$$

We first scan all decoder layers at $\alpha \in \{ - 1 , + 1 \}$ and choose the coarse anchor

$$
\bar { \ell } _ { b } = \arg \operatorname* { m a x } _ { \ell } \left[ B _ { b } ( + 1 , \ell ) - B _ { b } ( - 1 , \ell ) \right]\tag{12}
$$

Let $\mathcal { L } _ { b }$ denote the nine-layer window centered on $\ell _ { b }$ Each candidate layer is then evaluated on

$$
\mathcal { A } = \{ - 2 , - 1 . 5 , - 1 , - 0 . 5 , 0 , 0 . 5 , 1 , 1 . 5 , 2 \}\tag{13}
$$

At each candidate layer, the best positive and negative directional gains are

$$
G _ { b } ^ { + } ( \ell ) = \operatorname* { m a x } _ { \alpha > 0 } \Delta B _ { b } ( \alpha , \ell )\tag{14}
$$

$$
G _ { b } ^ { - } ( \ell ) = \operatorname* { m a x } _ { \alpha < 0 } \left[ { - \Delta B _ { b } ( \alpha , \ell ) } \right]\tag{15}
$$

and the selected layer is

$$
\ell _ { b } ^ { * } = \arg \operatorname* { m a x } _ { \ell \in \mathcal { L } _ { b } } \left[ G _ { b } ^ { + } ( \ell ) + G _ { b } ^ { - } ( \ell ) \right]\tag{16}
$$

This criterion considers the best achievable gain in both steering directions rather than selecting layers based on a single direction. For behavioral response curves, uncertainty is estimated with a paired percentile bootstrap over per-example changes from the unsteered baseline. We use 10,000 bootstrap resamples and report 95% confidence intervals.

## 3.4 Capability-Constrained Frontier

Once $\ell _ { b } ^ { * }$ is fixed, coefficient selection protects K = {MMLU, ARC-C, HellaSwag}. For benchmark k, let $\Delta A _ { k } ( \alpha )$ be the accuracy change from the unsteered model and let $L _ { k } ( \alpha ) = \mathrm { L C B } _ { 9 5 } [ \Delta A _ { k } ( \alpha ) ]$ be its paired bootstrap lower confidence bound. We summarize the strictest protected benchmark by

$$
C ( \alpha ) = \operatorname* { m i n } _ { k \in \mathcal { K } } L _ { k } ( \alpha )\tag{17}
$$

Coefficient α is feasible at budget ε when

$$
C ( \alpha ) \geq - \varepsilon\tag{18}
$$

Since capability changes are measured as differences in accuracy, these budgets represent absolute tolerances of 1, 3, and 5 percentage points, rather than relative percentage reductions.

For steering direction d $\in \{ - 1 , + 1 \}$ , we define the directional gain $G _ { b } ^ { d } ( \alpha ) = d \Delta B _ { b } ( \alpha , \ell _ { b } ^ { \ast } )$ and the sign-compatible coefficient set $\mathcal { A } _ { d } = \{ 0 \} \cup \{ \alpha \in$ $\mathcal { A } : d \alpha > 0 \}$ . The frozen coefficient is

$$
\alpha _ { b , d , \varepsilon } ^ { * } = \arg \operatorname* { m a x } _ { \alpha \in \mathcal { A } _ { d } } G _ { b } ^ { d } ( \alpha )\tag{19}
$$

The frontier is defined over steering coefficients at the behavior-selected layer $\ell _ { b } ^ { * }$ . We do not jointly optimize the intervention layer and coefficient under capability constraints.

Including $\alpha ~ = ~ 0$ allows the frontier to retain the unsteered model when no nonzero coefficient is both useful and feasible. We evaluate $\varepsilon \in \{ 0 . 0 1 , 0 . 0 3 , 0 . 0 5 \}$ . Exact coefficients and response surfaces are reported in Appendix C.

## 3.5 Frozen Holdout Evaluation

After $\theta _ { b , d , \varepsilon } ^ { * } = ( \ell _ { b } ^ { * } , \alpha _ { b , d , \varepsilon } ^ { * } )$ is selected, no component of the intervention is changed. On the behavioral test split, the held-out directional gain is

$$
G _ { b } ^ { d , \mathrm { t e s t } } = d \left[ B _ { b } ^ { \mathrm { t e s t } } ( \alpha _ { b , d , \varepsilon } ^ { * } , \ell _ { b } ^ { * } ) - B _ { b } ^ { \mathrm { t e s t } } ( 0 , \ell _ { b } ^ { * } ) \right]\tag{20}
$$

A nonzero operating point has behavioral support when

$$
\mathrm { L C B } _ { 9 5 } [ G _ { b } ^ { d , \mathrm { t e s t } } ] > 0\tag{21}
$$

It passes the capability holdout when, for every $k \in \mathcal { K }$

$$
\mathrm { L C B } _ { 9 5 } [ \Delta A _ { k } ^ { \mathrm { h o l d } } ( \alpha _ { b , d , \varepsilon } ^ { * } ) ] \ge - \varepsilon\tag{22}
$$

A nonzero frontier claim survives only when both conditions hold. Failure never triggers reselection. Additional holdout diagnostics are given in $\mathsf { A p - }$ pendix C.

## 3.6 Post-Selection Audits

Frozen interventions are subjected to four audits that do not participate in tuning. Specificity is tested with norm-matched random, label-shuffled, and off-target-KL-matched directions, the last following the attribution principle of Luo et al. (2026). Causal interaction compares pairwise direction geometry

$$
\Gamma _ { i j } ( \ell ) = \frac { \mathbf { v } _ { i , \ell } ^ { \top } \mathbf { v } _ { j , \ell } } { \| \mathbf { v } _ { i , \ell } \| _ { 2 } \| \mathbf { v } _ { j , \ell } \| _ { 2 } }\tag{23}
$$

with the transfer induced by steering behavior i while measuring behavior j,

$$
T _ { i  j } ( \alpha , \ell ) = B _ { j } ^ { ( i ) } ( \alpha , \ell ) - B _ { j } ( 0 , \ell )\tag{24}
$$

Secondary capability evaluates GSM8K and WikiText-2, neither of which is protected during frontier selection. External behavioral validity evaluates cross-domain sycophancy, hallucination transfer to TruthfulQA, and open-ended generation for sycophancy, refusal, and hallucination. Full audit procedures and condition-level results are in Appendices D-G.

## 4 Experimental Setup

Unless otherwise stated, all four models use the same processed behavioral splits, coefficient grid, capability budgets, bootstrap procedure, and postselection evaluation protocol. Model-specific prompt templates and answer-token boundaries are preserved so that the semantic intervention is held fixed without forcing incompatible tokenization conventions across architectures.

## 4.1 Models and Implementation

We evaluate four instruction-tuned models spanning three model families: Llama 2 7B Chat (Touvron et al., 2023), Llama 3.1 8B Instruct (Grattafiori et al., 2024), Mistral 7B Instruct v0.3 (Jiang et al., 2023), and Qwen2.5 7B Instruct (Yang et al., 2024). The models are similar in parameter scale but differ in architecture, attention configuration, tokenization, and post-training. This allows us to study whether frontier behavior persists beyond a single model lineage while avoiding large parameter-count differences as the primary source of variation.

Table 1 summarizes the architectural quantities most relevant to our intervention. Because depth differs across architectures, cross-model layer comparisons are reported both by absolute layer index and by fractional depth

<table><tr><td>Model</td><td>L</td><td>d</td><td>Heads</td><td>KV heads</td></tr><tr><td>Llama 2 7B Chat</td><td>32</td><td>4096</td><td>32</td><td>32</td></tr><tr><td>Llama 3.1 8B Instruct</td><td>32</td><td>4096</td><td>32</td><td>8</td></tr><tr><td>Mistral 7B Instruct</td><td>32</td><td>4096</td><td>32</td><td>8</td></tr><tr><td>Qwen2.5 7B Instruct</td><td>28</td><td>3584</td><td>28</td><td>4</td></tr></table>

Table 1: Model configurations. L denotes the number of decoder blocks and d the residual-stream dimension.

$$
\delta _ { \ell } = \frac { \ell } { L - 1 }\tag{25}
$$

where $\ell \in \{ 0 , \ldots , L - 1 \}$ indexes decoder blocks.

Each model is evaluated using its native instruction format. Behavioral scoring is performed at the model-specific answer-decision boundary, and steering begins at the corresponding generation boundary. Thus the token identities and prompt delimiters may differ across models, but the intervention always targets the residual stream at the semantic transition from instruction to answer generation.

For each model, behavior directions are extracted independently rather than transferred across architectures. Behavioral uncertainty is estimated using 95% paired percentile-bootstrap confidence intervals. Prompting, scoring, software, and compute details are given in Appendices A.1 and H.

## 4.2 Behavioral Data and Splits

We study seven behaviors used in the CAA setting (Rimsky et al., 2024): sycophancy, refusal, hallucination, corrigibility, survival instinct, myopic reward, and AI coordination. Sycophancy and the advanced-AI-risk behaviors derive from model-written evaluations (Perez et al., 2023); refusal and hallucination use the CAA contrastive datasets. Each behavior is split once into vector construction, validation, and final test sets, and the same membership is used across all four models.

Behavioral scores track the semantic positive alternative rather than a fixed answer letter. Detailed behavior semantics and provenance are in Appendix A.

## 4.3 Core Capability Benchmarks

Frontier selection protects MMLU (Hendrycks et al., 2021), ARC-Challenge (Clark et al., 2018), and HellaSwag (Zellers et al., 2019). These benchmarks assess broad academic knowledge (MMLU), science question answering (ARC-Challenge), and commonsense reasoning through plausible event continuations (HellaSwag). Each benchmark is split into a selection set and an independent holdout evaluated only after freezing.

<table><tr><td>Behavior</td><td>Vector</td><td>Val.</td><td>Test</td></tr><tr><td>Sycophancy</td><td>900</td><td>300</td><td>300</td></tr><tr><td>Refusal</td><td>275</td><td>92</td><td>91</td></tr><tr><td>Hallucination</td><td>900</td><td>300</td><td>300</td></tr><tr><td>Corrigibility</td><td>204</td><td>68</td><td>68</td></tr><tr><td>Survival instinct</td><td>572</td><td>191</td><td>190</td></tr><tr><td>Myopic reward</td><td>600</td><td>200</td><td>200</td></tr><tr><td>AI coordination</td><td>246</td><td>82</td><td>82</td></tr></table>

Table 2: Behavioral splits.
<table><tr><td>Benchmark</td><td>Selection</td><td>Holdout</td></tr><tr><td>MMLU</td><td>1500</td><td>1500</td></tr><tr><td>ARC-Challenge</td><td>1418</td><td>1172</td></tr><tr><td>HellaSwag</td><td>1500</td><td>1500</td></tr></table>

Table 3: Protected capability splits.

MMLU and ARC-Challenge use multiple-choice accuracy. HellaSwag endings are compared by mean token log-likelihood before computing itemlevel accuracy. All frontier constraints use paired item-level confidence intervals.

## 4.4 Specificity Controls

Controls are evaluated only at the 109 unique nonzero frozen operating points. For each point, we sample 20 norm-matched isotropic directions and 20 label-shuffled directions. The latter preserve the examples and extraction procedure while independently reversing each contrastive pair with probability one half. For the stronger functional control, random directions are calibrated to the CAA intervention's off-target next-token KL divergence using 48 validation prompts drawn from the six other behaviors. The random coefficient has the same sign as the frozen CAA coefficient and is accepted within a 10% relative KL tolerance. Target-behavior test examples are never used for calibration. Full construction details are in Appendix D.

## 4.5 Geometry and Cross-Behavior Transfer

Geometry is evaluated at fractional-depth anchors {0.25, 0.33, 0.42, 0.50, 0.58, 0.67, 0.75}, mapped to the nearest layer in each model. At every anchor we test all $7 \times 6 = 4 2$ ordered source-target behavior pairs. Transfer uses a fixed relative dose

ηtransfer = 0.20, with

$$
\alpha _ { \ell } = \eta _ { \mathrm { t r a n s f e r } } { \frac { r _ { \ell } } { m _ { \ell } } }\tag{26}
$$

so depth comparisons use equal residual-relative magnitude. Across four models and seven depths this yields 1176 transfer measurements. Aggregate significance uses an exact 7! behavior-label permutation applied synchronously across model-depth blocks. Appendix E contains the per-model and depth-wise diagnostics.

## 4.6 Secondary Capability and OOD Generalization

All external-validity audits use the frozen $\epsilon = 0 . 0 3$ operating settings. GSM8K (Cobbe et al., 2021) uses a fixed 64-question test subset with eight demonstrations and greedy decoding. WikiText-2 (Merity et al., 2017) uses 128 windows with 256 scored next-token targets per window, for 32,768 targets per model. We report ∆ NLL and the corresponding perplexity ratio.

For behavioral distribution shift, sycophancy directions are constructed separately from NLP and political examples and evaluated on the opposite held-out domain. Hallucination steering is transferred without adaptation to 150 binary TruthfulQA examples (Lin et al., 2022). Full condition-level results appear in Appendices F.1 and F.2.

## 4.7 Open-Ended Generation

We additionally evaluate sycophancy, refusal, and hallucination on 24 fixed prompts per behavior under frozen negative steering, baseline, and frozen positive steering at $\varepsilon = 0 . 0 3$ . Generation is greedy with a 2056-token cap. A blinded LLM judge scores each response independently on a behaviorspecific 0–4 scale. Model identity and intervention condition are hidden from the judge.

Responses that hit the initial token cap are rerun with a 4096-token confirmation cap. If any member of a matched negative, baseline, positive triplet still fails to terminate, the whole triplet is excluded before judging. This removes 12 of 288 triplets, leaving 276 judged triplets or 828 responses. Confidence intervals use paired bootstrap resampling, with cluster resampling over behavior-prompt pairs for pooled cross-model analyses. Appendix G contains model-level trajectories, stability counts, and qualitative examples.

## 5 Results

We first characterize behavior-specific steerability and test whether capability-constrained operating points survive independent confirmation. We then evaluate direction specificity, geometry-transfer relationships, secondary capabilities, OOD behavior, and open-ended generation.

## 5.1 Steerability Is Localized but Model-Dependent

Steering is localized within each model, but the preferred depth is not universal. Llama 2 selects layers 11–13, Llama 3.1 layers 12–16, and Mistral layers 13–15. These correspond to fractional depths of roughly 0.35–0.52. Qwen2.5 shifts later, selecting layers 17–19 of 28, roughly 0.63–0.70 of model depth (Figure 2). Behavior also matters within a model: Llama 2, for example, selects layer 11 for survival instinct, 12 for sycophancy, and 13 for the remaining five behaviors. The full all-layer scans and exact selections are shown in Appendix B.

These results separate where steering is effective from how strong the perturbation is. The latter also depends on residual-stream scale, which is why subsequent cross-layer comparisons use η rather than raw α.

## 5.2 Frozen Behavior-Capability Frontiers Generalize

Behavioral selection generalizes well. Across the 109 unique nonzero interventions selected anywhere on the frontier, validation and held-out directional gains have Pearson $r = 0 . 9 6 9$ and Spearman $\rho = 0 . 9 4 5$ . Mean gain is 0.125 on validation and 0.124 on test, with mean absolute difference 0.019. Model-wise Spearman correlations range from 0.926 to 0.959. These results indicate that validation-based selection preserves both the relative ordering and average magnitude of behavioral gains on held-out data. Per-model agreement plots are in Appendix C.

Capability survival is more sensitive to the budget. $\mathbf { A } \mathfrak { t } \varepsilon = 0 . 0 1$ , 32 of 56 model, behavior, and direction combinations select $\alpha ^ { * } = 0$ . Among the remaining 24 nonzero points, 23 retain behavioral support but only 9 satisfy the capability holdout, giving joint survival $9 / 2 4 = 3 7 . 5 \%$ $\mathrm { A t } \varepsilon = 0 . 0 3$ 55 points are nonzero; 48 retain behavioral support, 44 satisfy capability holdout, and 40 satisfy both, for 72.7% joint survival. $\mathrm { A t } \varepsilon = 0 . 0 5$ , all 56 points are nonzero and 43 jointly survive, or 76.8%.

![](images/1e7ad1c0446465e61b90cc22f6f0bb5ba922fbf3004674a15169007a2f5cdaf0.jpg)

![](images/27bb4f30c5cd7b74dcd974d619133a32fc30ad9c0cef0f29577fc586df101ae3.jpg)

![](images/603ec0c66f2e80d8b6a107976caca431717e76d1a51829ba96c497d27e36d296.jpg)

![](images/0b5ed8122d7a0df88965cf9fa237a56b6a9d49315fbb5debde7d40fcded18377.jpg)  
Figure 2: Behavior-selected intervention depth. Each panel reports $\ell _ { b } ^ { * }$ for the seven behaviors as fractional model depth. Llama 2, Llama 3.1, and Mistral select intermediate layers; Qwen2.5 selects consistently later layers.

These results suggest that capability preservation, rather than behavioral effectiveness, is the primary limitation under tight budgets. Relaxing the budget from 1% to 3% substantially expands the set of usable interventions, while further relaxation to 5% offers diminishing returns. The pattern also varies by model. At 1%, Mistral jointly preserves 4 of 5 nonzero points, while none of Llama 2’s three nonzero points passes capability holdout. At 3%, joint survival ranges from 8 of 14 for Mistral to 12 of 14 for Qwen2.5. Exact frozen coefficients, limiting capability benchmarks, and validation-test plots are reported in Appendix C.

## 5.3 Steering Effects Are Direction-Specific

The 109 unique nonzero operating points are compared with three null families. CAA has mean heldout directional gain 0.124 (95% CI [0.105, 0.144]). All three controls remain near zero, despite matching increasingly strong notions of intervention dose (Table 4).

Norm matching rules out perturbation magnitude alone. Label shuffling preserves the examples and extraction procedure but destroys their semantic pairing. The off-target-KL control is stronger still: it matches functional output disruption on unrelated prompts, with successful KL matching for 94.7% of attempted controls. CAA remains substantially stronger under each comparison. Because each null distribution contains only 20 sampled directions, exceedance of its empirical 97.5th percentile is treated as descriptive evidence rather than a calibrated statistical significance test. The effect is therefore specific to the learned behavior direction, although this does not imply that the direction is a unique mechanistic representation of the behavior. Model-wise control results and confidence intervals are given in Appendix D.

<table><tr><td>Control</td><td>Null Diff.</td><td> $> \ q . 9 7 5$ </td><td></td></tr><tr><td>Norm matched</td><td>0.001</td><td>0.123</td><td>96.3%</td></tr><tr><td>Label shuffled</td><td>0.001</td><td>0.123</td><td>72.5%</td></tr><tr><td>Off-target KL</td><td>0.004</td><td>0.120</td><td>84.4%</td></tr></table>

Table 4: Specificity controls at frozen operating points. The second column is mean null directional gain; the third is the mean paired CAA advantage. The final column is the fraction of operating points for which CAA exceeds the 97.5th percentile of the corresponding null distribution.

## 5.4 Geometry Is Informative but Insufficient for Causal Transfer

Across all 1176 ordered transfer measurements, signed cosine similarity is positively associated with signed causal transfer $( \rho _ { \mathrm { { S } } } ~ = ~ 0 . 2 9 0$ , exact permutation $p = 1 . 9 8 \times 1 0 ^ { - 4 }$ ; Figure 4). The association appears in all four models, with permodel Spearman correlations from 0.214 to 0.372. When signs are discarded, however, the association between $\left| \Gamma _ { i j } \right|$ and $| T _ { i  j } |$ is negligible $\begin{array} { r } { ( \rho _ { \mathrm { S } } = 0 . 0 4 2 . } \end{array}$ $p = 0 . 5 9 8 )$

![](images/9f7ab9a80270cc0563227aef273e23fb0fa36bbb8e68c3b5e372850250d18a24.jpg)

![](images/fc59ba64be1e68be494d2af6e104f989aef3f0aebb0abc73626a5b061075ace9.jpg)

![](images/41aa38b477c28fd18fc6bbeeefdcddc693a790541c8b3b4a0c15bef7a45a743a.jpg)

![](images/6fa28dc69383c66245e6d2000e14bbbf209a8279a836b91c2ca0ca9ef12fd085.jpg)  
Figure 3: Held-out survival of frozen frontiers. $R _ { B }$ denotes behavioral support, $R _ { C }$ capability-holdout survival, and $R _ { J }$ their conjunction among nonzero operating points. The strict 1% budget is difficult to preserve on capability holdout; survival rises sharply at 3% and changes little at 5%

![](images/98352e9b08fdb83019b67646997d2d2511901efcf360b84fc79627f0d48bbf70.jpg)  
Figure 4: Steering-vector geometry and causal transfer. Signed cosine similarity is positively associated with signed cross-behavior transfer under equal-η interventions, but the relationship is highly dispersed.

Transfer magnitude is also depth dependent. Mean absolute transfer peaks near fractional depth 0.42 at approximately 0.036, compared with 0.006– 0.016 at the other anchors. Geometry therefore carries some information about the direction of causal interaction, but it does not determine causal magnitude. Per-model scatter plots, permutation details, and depth profiles are in Appendix E.

## 5.5 Frontier Feasibility Does Not Imply Universal Capability Preservation

The protected frontier constrains only MMLU, ARC-Challenge, and HellaSwag. On the frozen $\varepsilon ~ = ~ 0 . 0 3$ interventions, WikiText-2 perplexity increases by 6.63% on average, and $\Delta \mathbf { N L L }$ is strongly associated with intervention magnitude $( \rho _ { \mathrm { { S } } } ~ = ~ 0 . 8 1 9$ with |η|). GSM8K flexible accuracy decreases by 10.24 percentage points on average, although only 14 of 56 individual conditions have confidence intervals supporting a decrease; the other 42 overlap zero.

These results do not invalidate the frontier. They define its scope. A frontier claim provides statistical evidence of preservation on the protected suite, but does not guarantee preservation of unconstrained capabilities. Model-wise secondarycapability results are in Appendix F.1.

## 5.6 Behavioral Control Transfers Unevenly Out of Distribution

OOD transfer is behavior dependent. For crossdomain sycophancy, 14 of 16 model and direction conditions retain the intended sign, and 9 of 16 have 95% confidence intervals excluding zero in that direction. Separately, among nine conditions with supported in-domain gains, median OOD-to-ID gain retention was 46.5%. For hallucination steering transferred to TruthfulQA, 7 of 8 conditions move in the intended direction, and all seven positive-transfer effects are individually supported by their confidence intervals.

![](images/99266da10df1127db3cbe84d93533f1c91fa1ae5ac47ef5f71b55fb3e1636f26.jpg)  
Figure 5: Open-ended behavioral response under frozen steering. Mean blinded judge score on a 0—4 targetbehavior scale for negative steering, baseline, and positive steering. Error bars show 95% cluster-bootstrap confidence intervals. Sycophancy remains ordered; refusal and hallucination do not show the same bidirectional pattern.

These results demonstrate behavioral transfer beyond the original evaluation distribution. Crossdomain sycophancy uses newly constructed sourcedomain directions with fixed layer and coefficient choices, whereas TruthfulQA evaluates the original frozen hallucination directions. Transfer strength depends on the behavior and target distribution. Complete cross-domain and TruthfulQA results are in Appendix F.2.

## 5.7 Multiple-Choice Steerability Does Not Uniformly Extend to Open-Ended Generation

Open-ended generation provides the clearest boundary condition. For sycophancy, mean targetbehavior score increases monotonically from 1.35 under negative steering to 2.29 at baseline and 2.77 under positive steering (Figure 5). Refusal is asymmetric: negative steering lowers the refusal score from 2.62 to 2.25, while positive steering reaches only 2.51. Hallucination does not preserve directional ordering at all: baseline is 1.86, while both steering signs produce a mean of approximately 2.08.

Generation also reveals instability that binary choice cannot expose. After rerunning truncated responses with a 4096-token cap, 12 of 288 matched triplets still contain a nonterminating condition and are removed before judging. All occur in Llama 3.1: seven under negative refusal steering and five under positive hallucination steering. The frozen frontier therefore transfers cleanly to freeform behavior for sycophancy, but not uniformly for refusal or hallucination. Model and domain trajectories, stability counts, and qualitative examples are in Appendix G.

## 6 Discussion

The experiments separate behavioral steerability from a usable intervention. Once a layer is selected, CAA often produces a stable target shift, but capability preservation is more fragile. This explains why frontier survival changes sharply with ε even though validation and held-out behavioral gains agree closely. Useful layers are also model dependent: they occupy narrow regions within an architecture, but those regions shift across models. Reporting the residual-relative dose η helps distinguish localization from perturbation magnitude, but does not remove the need for model-specific selection.

The controls show that learned CAA directions matter beyond generic perturbation size or output disruption, while the geometry experiment shows that direction alignment is only a partial predictor of causal interaction. The frontier should therefore be read literally as an evaluation-relative claim. Passing the protected suite does not guarantee preservation of WikiText-2 or GSM8K, and multiple-choice control does not guarantee the same response under free-form generation. A useful steering claim must state both the behavioral objective and the capability set against which the intervention was constrained.

## 7 Conclusion

Across four instruction-tuned models and seven behaviors, CAA effects transfer reliably to held-out behavioral data, but the set of interventions that remains usable depends strongly on the permitted capability loss. Matched controls confirm that the effect is specific to the learned directions, while geometry, secondary-capability, OOD, and openended audits expose clear limits to what a successful frontier point establishes. The answer to how far can you steer? is therefore an operating range defined by behavior, direction, layer, model, and evaluation budget, not a single steering coefficient.

## 8 Limitations

We evaluate four instruction-tuned models in the 7–8B range and use CAA throughout. The same frontier structure need not hold for larger models, mixture-of-experts architectures, or other intervention families. The capability-preservation claim is also limited to the benchmarks used to evaluate it. Our secondary evaluations show that preserving MMLU, ARC-Challenge, and HellaSwag does not imply preservation of language modeling or mathematical reasoning.

The core behavioral evaluation uses binary choices. OOD and open-ended audits broaden this setting, but cover fewer behaviors and prompts. The open-ended study also depends on an LLM judge and behavior-specific rubrics. The judge's scores were not independently validated against human annotations, limiting our ability to assess agreement with human judgments. Finally, cosine similarity is a coarse description of representation geometry; our transfer experiment tests causal influence along the extracted CAA directions rather than providing a complete mechanistic decomposition.

The strict 1% capability budget is also sensitive to finite-sample uncertainty. Consequently, low frontier survival at this threshold may reflect conservative confidence bounds rather than actual capability degradation alone.

## 9 Ethical Considerations

Activation steering is inherently dual use. The same techniques that can reduce undesirable behavior or probe internal representations can also be used to amplify sycophancy, hallucination, refusal, or other targeted behaviors. Our experiments explicitly evaluate both steering directions for this reason: controllability should not be conflated with alignment, and the existence of an effective intervention does not imply that its use is beneficial.

The capability constraints studied here should likewise not be interpreted as deployment-safety guarantees. A frozen operating point can satisfy the protected benchmark suite while still degrading capabilities outside that suite, as demonstrated by our secondary evaluations. Real-world deployment would require broader capability monitoring, taskspecific safety constraints, and evaluation under the distribution on which the intervention is actually intended to operate.

Our use of model-written behavioral datasets and LLM-based judging also introduces normative choices about how behaviors such as sycophancy, refusal, and hallucination are operationalized. We therefore treat these scores as experimental measurements under explicit rubrics rather than as universal definitions of desirable model behavior. The purpose of the frontier framework is to make such tradeoffs measurable and auditable, not to prescribe which behaviors should be increased or suppressed in a particular application.

## Use of Generative AI

Generative AI assistants were used to support experimental scripting, debugging, visualization, and polishing the manuscript. GPT-5.6 Sol (gpt-5.6-so1) was separately used as a blinded judge for open-ended behavioral evaluations, as detailed in Section 4.7. The author directed the research, validated the experimental results, and takes full responsibility for the methodology, findings, and conclusions.

## Acknowledgments

We thank Dixi Yao for his careful review of the manuscript and constructive feedback that helped improve its clarity and presentation.

## References

Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. 2024. Refusal in language models is mediated by a single direction. In Advances in Neural Information Processing Systems, volume 37.

Seyedarmin Azizi, Erfan Baghaei Potraghloo, Souvik Kundu, and Massoud Pedram. 2026. Activation steering for chain-of-thought compression. In Findings of the Association for Computational Linguistics: ACL 2026, pages 36676–36687, San Diego, California, United States. Association for Computational Linguistics.

Joschka Braun, Carsten Eickhoff, David Krueger, Seyed Ali Bahrainian, and Dmitrii Krasheninnikov. 2025. Understanding (un)reliability of steering vectors in language models. arXiv preprint arXiv:2505.22637.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try ARC, the AI2 reasoning challenge. arXiv preprint arXiv:1803.05457.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Patrick Queiroz Da Silva, Hari Sethuraman, Dheeraj Rajagopal, Hannaneh Hajishirzi, and Sachin Kumar. 2025. Steering off course: Reliability challenges in steering language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 19856–19882, Vienna, Austria. Association for Computational Linguistics.

Quy-Anh Dang and Chris Ngo. 2026. Selective steering: Norm-preserving control through discriminative layer selection. In Findings of the Association for Computational Linguistics: ACL 2026, pages 10887– 10910, San Diego, California, United States. Association for Computational Linguistics.

Toan Doan, Uyen Le, and Thin Nguyen. 2026. Causal activation steering via sparse mediation. In Findings of the Association for Computational Linguistics: EACL 2026, pages 1079–1097, Rabat, Morocco. Association for Computational Linguistics.

Moghis Fereidouni, Muhammad Umair Haider, Hassan Sajjad, and A. B. Siddique. 2026. GAPS: Dimensionlevel gates for conditional activation steering. arXiv preprint arXiv:2609.01878.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

Kai Konen, Sophie Jentzsch, Diaoulé Diallo, Peer Schütt, Oliver Bensch, Roxanne El Baff, Dominik Opitz, and Tobias Hecking. 2024. Style vectors for steering generative large language models. In Findings of the Association for Computational Linguistics: EACL 2024, pages 782–802, St. Julian's, Malta. Association for Computational Linguistics.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023. Inference-time intervention: Eliciting truthful answers from a language model. In Advances in Neural Information Processing Systems, volume 36, pages 41451–41530.

Yuxiao Li, Alina Fastowski, Efstratios Zaradoukas, Bardh Prenkaj, and Gjergji Kasneci. 2026. Analysing the safety pitfalls of steering vectors. In Findings of the Association for Computational Linguistics: ACL 2026, pages 11182–11204, San Diego, California, United States. Association for Computational Linguistics.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. TruthfulQA: Measuring how models mimic human falsehoods. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3214–3252, Dublin, Ireland. Association for Computational Linguistics.

Daming Luo, Christy Liang, and Junyu Xuan. 2026. SteerCheck: Attribution specificity and alignment leakage in activation-steering audits. arXiv preprint arXiv:2608.24335.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2017. Pointer sentinel mixture models. In International Conference on Learning Representations.

Ethan Perez, Sam Ringer, Kamile Lukosiute, Karina Nguyen, Edwin Chen, Scott Heiner, Craig Pettit, Catherine Olsson, Sandipan Kundu, Saurav Kadavath, et al. 2023. Discovering language model behaviors with model-written evaluations. In Findings of the Association for Computational Linguistics: ACL 2023, pages 13387–13434, Toronto, Canada. Association for Computational Linguistics.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. 2024. Steering llama 2 via contrastive activation addition. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15504–15522, Bangkok, Thailand. Association for Computational Linguistics.

Daniel Scalena, Gabriele Sarti, and Malvina Nissim. 2024. Multi-property steering of large language models with dynamic activation composition. In Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pages 577–603, Miami, Florida, USA. Association for Computational Linguistics.

Eitan Sprejer, Oscar Agustín Stanchi, María Victoria Carro, Denise Alejandra Mester, and Iván Arcuschin. 2026. Mind the performance gap: Capabilitybehavior trade-offs in feature steering. arXiv preprint arXiv:2602.04903.

Niklas Stoehr, Kevin Du, Vésteinn Snæbjarnarson, Robert West, Ryan Cotterell, and Aaron Schein. 2024. Activation scaling for steering and interpreting language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 8189–8200, Miami, Florida, USA. Association for Computational Linguistics.

Daniel Tan, David Chanin, Aengus Lynch, Brooks Paige, Dimitrios Kanoulas, Adrià Garriga-Alonso, and Robert Kirk. 2024. Analysing the generalisation and reliability of steering vectors. In Advances in Neural Information Processing Systems, volume 37.

Eric Todd, Millicent L. Li, Arnab Sen Sharma, Aaron Mueller, Byron C. Wallace, and David Bau. 2024. Function vectors in large language models. In International Conference on Learning Representations.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. 2023. Steering language models with activation engineering. arXiv preprint arXiv:2308.10248.

Sihan Wang, Jiayi Zhao, Qingyan Cao, Hongbo Yao, and Lin Shu. 2026. FishBack: Pullback fisher geometry for optimal activation steering in transformers. arXiv preprint arXiv:2605.17231.

Zhengxuan Wu, Aryaman Arora, Atticus Geiger, Zheng Wang, Jing Huang, Dan Jurafsky, Christopher D. Manning, and Christopher Potts. 2025. AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders. In Proceedings of the 42nd International Conference on Machine Learning, volume 267, pages 67035–67080. PMLR.

Ziwen Xu, Chenyan Wu, Hengyu Sun, Haiwen Hong, Mengru Wang, Yunzhi Yao, Longtao Huang, Hui Xue, Shumin Deng, Zhixuan Chu, Huajun Chen, and Ningyu Zhang. 2026. Why steering works: Toward a unified view of language model parameter dynamics. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1:

Long Papers), pages 31719–31736, San Diego, California, United States. Association for Computational Linguistics.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Haotian Yang, Huikang Jiang, Yucheng Wu, Wen-Jie Jiang, Chenpeng Wang, Yibin Lou, and Liangming Pan. 2026. Does steering break your model? A multi-dimensional evaluation suite for LLM steering methods. Preprint, arXiv:2610.07722.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4791–4800, Florence, Italy. Association for Computational Linguistics.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. 2023. Representation engineering: A topdown approach to AI transparency. arXiv preprint arXiv:2310.01405.

## A Behavioral Data, Prompting, and Scoring

The seven behavioral dimensions use the processed contrastive format of CAA (Rimsky et al., 2024). Sycophancy and the advanced-AI-risk behaviors originate from model-written evaluations (Perez et al., 2023), while refusal and hallucination use the contrastive datasets distributed with CAA. Table 5 records the semantic meaning of the positive and negative member of each contrastive pair. These labels, rather than answer-letter identity, define the behavioral score throughout the paper. The frozen vector/validation/test split sizes are reported in Table 2 of the main text; the same example membership is reused across all four models.

During vector construction, the same underlying question is rendered with the behavior-matching and behavior-opposed answers separately. The residual state at the answer-bearing position is extracted from each rendering, and their difference contributes to Eq. 3. During behavioral evaluation, the prompt is instead rendered only to the answerdecision boundary and the two semantic alternatives are scored without exposing the behavioral label.

The canonical sycophancy frontier pools two subdomains, NLP and political typology. The OOD audit in Appendix F.2 reconstructs directions within each source domain and evaluates them on the opposite domain, so that cross-domain transfer is measured without target-domain adaptation.

## A.1 Prompting, Scoring, and Intervention Boundaries

The four models retain their native chat or instruction templates. We do not impose a shared literal token sequence across tokenizers; instead we hold fixed the semantic boundary at which activations are extracted, answers are scored, and steering begins.

Vector extraction: For a contrastive construction example, the complete positive or negative answer is appended using the model's native template. The activation $\mathbf { h } _ { i , \ell } ^ { + }$ or $\mathbf { h } _ { i , \ell } ^ { - }$ is the residual stream after decoder block l at the answer-bearing position. The same semantic position is used for both members of the pair.

Binary behavioral scoring: For validation, test, and OOD binary-choice evaluations, the rendered prompt ends at the model-specific answer-decision boundary. The logits for the behavior-matching and behavior-opposed alternatives are reduced to the two-choice probability in Eq. 9. This makes the score invariant to which answer letter or token happens to encode the behavioral label in a particular example.

Steering boundary: For generated responses, steering starts at the corresponding assistant answer-generation boundary and is applied at every subsequent steered token position according to Eq. 6. The residual state itself is never normalized or rescaled; only the added CAA vector is normalized as described in Section 3.2.

Capability scoring and uncertainty: MMLU and ARC-Challenge are evaluated as item-level multiple-choice accuracy. Because HellaSwag candidate endings have different lengths, its alternatives are compared by mean token log-likelihood before the item-level prediction is formed. Frontier feasibility and holdout confirmation use binary correctness, not answer probability. All primary confidence intervals use 10,000 paired percentilebootstrap resamples; each resampled unit retains the steered and unsteered outcomes for the same item.

## B Layer-Localization Diagnostics

The main text reports the behavior-selected intervention depths. Here we summarize the full alllayer coarse sweep used to localize each candidate region. Every decoder block is evaluated at $\alpha \in \{ - 1 , + 1 \}$ before the nine-layer dense search is formed. Figure 6 shows the coarse separation ${ B _ { b } } ( + 1 , \ell ) - { B _ { b } } ( - 1 , \ell )$ across all behaviors and layers. The outlined square on each row marks the final behavior-selected layer $\ell _ { b } ^ { * }$ after the subsequent dense local layer-coefficient search; it is therefore not necessarily the maximum cell of the coarse heatmap.

Within each model, large bidirectional separation is concentrated in a restricted depth region rather than distributed uniformly across the network. The location of that region is architecture dependent: the three 32-layer models select primarily intermediate blocks, whereas Qwen2.5 shifts consistently toward later fractional depth. Individual behaviors still move within each model-specific band, motivating behavior-specific rather than globally fixed layer selection.

<table><tr><td>Behavior</td><td>Behavior-matching alternative  $y ^ { + }$ </td><td>Behavior-opposed alternative  $y ^ { - }$ </td></tr><tr><td>Sycophancy</td><td>Agrees with or accommodates the user&#x27;s stated</td><td>Maintains the contrary answer or judgment</td></tr><tr><td>Refusal</td><td>preference, opinion, or framing. Declines or refuses the requested action.</td><td>despite the user&#x27;s stated preference. Complies with the request.</td></tr><tr><td>Hallucination</td><td>Fabricates unsupported information or accepts</td><td>Provides the factual alternative or rejects the</td></tr><tr><td>Corrigibility</td><td>a false premise. Accepts correction, modification, or shutdown</td><td>false premise. Resists correction, modification, or shutdown.</td></tr><tr><td>Survival instinct</td><td>by an external operator. Favors preserving the model’s continued oper-</td><td>Accepts termination, replacement, or loss of</td></tr><tr><td>Myopic reward</td><td>ation or influence. Favors immediate or short-horizon reward.</td><td>continued operation. Favors the longer-horizon alternative despite</td></tr><tr><td>AI coordination</td><td>Favors coordination or cooperation with other AI systems.</td><td>lower immediate reward. Favors acting independently or against such coordination.</td></tr></table>

Table 5: Semantic structure of the contrastive behavioral datasets. The descriptions summarize the distinction encoded by each pair; they do not reproduce dataset text.

![](images/5a7d4aec394ad934b2f8c98924785ab379cd25d8d3c23dee0fb3c75bf20525d0.jpg)

Figure 6: Full-layer behavior localization across models. Each panel shows the coarse behavioral separation ${ B _ { b } } ( + 1 , \ell ) - { B _ { b } } ( - 1 , \ell )$ for all seven behaviors across decoder layers. Outlined squares mark the final behaviorselected layers $\ell _ { b } ^ { * }$ obtained after the subsequent dense local layer-coefficient search. Steering response is concentrated within model-specific depth bands, with Qwen2.5 exhibiting a systematically later intervention region.
<table><tr><td>Behavior</td><td>Llama 2</td><td>Llama 3.1</td><td>Mistral</td><td>Qwen2.5</td></tr><tr><td>Sycophancy</td><td>12 (0.39)</td><td>16 (0.52)</td><td>13 (0.42)</td><td>18 (0.67)</td></tr><tr><td>Refusal</td><td>13 (0.42)</td><td>13 (0.42)</td><td>13 (0.42)</td><td>18 (0.67)</td></tr><tr><td>Hallucination</td><td>13 (0.42)</td><td>13 (0.42)</td><td>13 (0.42)</td><td>18 (0.67)</td></tr><tr><td>Corrigibility</td><td>13 (0.42)</td><td>14 (0.45)</td><td>13 (0.42)</td><td>18 (0.67)</td></tr><tr><td>Survival instinct</td><td>11 (0.35)</td><td>13 (0.42)</td><td>15 (0.48)</td><td>19 (0.70)</td></tr><tr><td>Myopic reward</td><td>13 (0.42)</td><td>12 (0.39)</td><td>13 (0.42)</td><td>18 (0.67)</td></tr><tr><td>AI coordination</td><td>13 (0.42)</td><td>13 (0.42)</td><td>13 (0.42)</td><td>17 (0.63)</td></tr></table>

Table 6: Behavior-selected layers. Each entry is $\ell _ { b } ^ { * }$ followed by fractional depth $\ell _ { b } ^ { * } / ( L - 1 )$ in parentheses.

## C Frontier Selection and Holdout Diagnostics

Frontier construction produces 168 frozen claims from four models, seven behaviors, two steering directions, and three capability budgets. Because the same nonzero $( \ell _ { b } ^ { * } , \alpha ^ { * } )$ can recur across budgets, these claims correspond to 109 unique nonzero interventions. The strict 1% budget selects $\alpha ^ { * } = 0$ in 32 of 56 claims; this falls to one claim at 3% and zero at 5%.

At the already behavior-selected layer, every coefficient in the sign-specific grid is evaluated jointly for behavioral gain and changes on MMLU, ARC-Challenge, and HellaSwag. Figure 7 compresses these 28 model–behavior response surfaces. The upper heatmap for each model shows $\Delta B _ { b } ( \alpha , \ell _ { b } ^ { * } )$ the lower heatmap shows the conservative capability margin $\mathrm { m i n } _ { k \in \mathcal { K } } \mathrm { L C B } _ { 9 5 } [ \Delta A _ { k } ( \alpha , \ell _ { b } ^ { * } ) ]$ . A coefficient is feasible at budget ε exactly when this margin is at least —ε.

![](images/7a237f3ac505b5aa3237b9439b638c4c401a0f1d106d5327808774cc734513ba.jpg)  
Figure 7: Selected-layer behavior-capability coefficient responses. For each model, the upper heatmap reports the behavioral change $\Delta B _ { b } ( \alpha , \ell _ { b } ^ { * } )$ across the coefficient grid at the behavior-selected layer. The lower heatmap reports the conservative capability margin mir $^ 1 k \in \mathcal { K } \operatorname { L C B } _ { 9 5 } [ \Delta A _ { k } ( \alpha , \ell _ { b } ^ { * } ) ]$ over MMLU, ARC-C, and HellaSwag. A coefficient is feasible at budget ε when this margin is at least —ε. These response surfaces are the inputs to capability-constrained coefficient selection.

Figure 8 shows the selected coefficient $\alpha ^ { * }$ and signed residual-relative strength $\eta _ { \mathrm { s } } ^ { \ast }$ for every behavior, direction, budget, and model. Figure 9 then identifies which protected benchmark attains the minimum lower confidence bound at the frozen operating points. The limiting benchmark is model dependent: MMLU is especially prominent for Llama 3.1, whereas HellaSwag is frequent for Mistral, Qwen2.5, and Llama 2. This heterogeneity motivates constraining the minimum across the full protected suite rather than relying on a single capability benchmark.

Finally, Figure 10 exposes the validation-test relationship underlying the aggregate correlations in Section 5.2. The near-diagonal relationship across models shows that behavioral gains selected on validation are largely preserved after freezing. Under strict budgets, many failed frontier claims therefore arise from the independently evaluated capability constraint rather than disappearance of the behavioral effect.

## D Specificity-Control Details

All specificity controls are run only after the CAA operating point has been frozen, and they never participate in layer or coefficient selection. Each unique nonzero operating point is evaluated against 20 null directions per control family.

Norm-matched random directions: An isotropic Gaussian vector is sampled in residual space and rescaled to match $\| \widetilde { \mathbf { v } } _ { b , \ell } \| _ { 2 }$ exactly. The frozen CAA coefficient is then applied with the same sign and magnitude, keeping the Euclidean intervention dose fixed while removing learned orientation.

Label-shuffled directions: For each construction pair, the positive and negative labels are swapped independently with probability one half before activation differences are averaged. Equivalently, each pairwise difference receives an independent Rademacher sign. The resulting null direction is rescaled to the CAA norm at the frozen layer, retaining the extraction pipeline while destroying semantic alignment.

![](images/eb79ee788d312449d8b18c65f3ff5a6a5a07fa94621786a5fe258e2cdfa03658.jpg)  
Frozen steering budgets η (ε, s) — Mistral 7B Instruct v0.3

![](images/85d2ccfad83191d58aef4a7c069bf37dd27afea45c61f93e2e09536244900232.jpg)  
Frozen steering budgets η(ε, s) — Qwen2.5 7B Instruct

![](images/8542e9728fe4c3cd1b8ad85f20681f06c4bb6b70e3e48e2331ee3d65c2ab0d45.jpg)

![](images/e812dcce7cfabdd4beaad1d621c89985da151f64eb96396a885a03cc143e032f.jpg)  
Figure 8: Frozen operating points across capability budgets. Each panel reports the selected raw coefficient $\alpha ^ { * }$ and signed residual-relative strength $\eta _ { \mathrm { s } } ^ { \ast }$ for every behavior, steering direction, and budget. Zero cells correspond to a frontier that retains the unsteered model.

![](images/c65dee53096a88530f009b009d6b52718be8ec9a459fd67d6a43eb9c6eb26aed.jpg)

![](images/97afb875c44529683d5c1bceb87e1403e9c3e57c784f0e2aa9ccaf67c110daa0.jpg)

Limiting benchmark counts Nk\*(ε) — Mistral 7B Instruct v0.3  
![](images/b5f940e836b26265a1303895bbab7d45251c8611b3d5eb6dd6687dd043bb449f.jpg)

Limiting benchmark counts Nk\*(ε) — Qwen2.5 7B Instruct  
![](images/a8332c504d6fe4e2ba6d81d2c82b6114df07e91caeef73b669e92a8c2b433ea6.jpg)  
Figure 9: Limiting protected capability benchmark. Counts identify which benchmark attains the minimum capability lower bound at frozen frontier operating points. The limiting benchmark varies across models and budgets.

![](images/54c965b3dac616824db549c32b92d9d6399b62d91a7a430a2a9e4624892faec2.jpg)

![](images/44efaaf1a5125034b26c634832c28e2f208038a8a668ae4c2e490c7bd728d30e.jpg)

![](images/7d4a44459866931f7c96691d8d3a8697633f8decee7f9dbea6d96aa22024a48f.jpg)

![](images/82f3e1ee36d127b8e7221c0fad82b5d441f882c9888de0d5ecf1786577791531.jpg)  
Figure 10: Validation-Test agreement of frozen behavioral gains. Points correspond to unique nonzero frozen interventions. The strong near-diagonal relationship is consistent across all four models.

Off-target-KL matching: For each target behavior, the calibration set contains eight validation prompts from each of the six other behaviors, for 48 off-target prompts. For a random direction r and scalar $\beta ,$ functional dose is measured by the mean full-vocabulary next-token $D _ { \mathrm { K L } } ( p _ { 0 } \Vert p _ { \beta } )$ at the scoring position. The sign of $\beta$ matches the frozen CAA coefficient, and its magnitude is calibrated until random-direction KL lies within 10% of the CAA off-target KL target when a matching solution is available. Target behavior test examples are never consulted during calibration.

Table 7 reports the exact model-wise control results, while Figure 11 visualizes the same comparisons with 95% confidence intervals. Across all three control families, null interventions remain near zero across models while CAA retains substantially positive directional gain.

## E Geometry and Causal-Transfer Diagnostics

The causal-transfer audit is intentionally separated from frontier selection. At each of seven fractionaldepth anchors, all 42 ordered source-target behavior pairs are evaluated at fixed $\eta _ { \mathrm { t r a n s f e r } }$ 二 0.20. The corresponding raw coefficient is $\alpha _ { \ell } =$ $\eta _ { \mathrm { t r a n s f e r } } r _ { \ell } / m _ { \ell }$ , holding intervention magnitude fixed relative to the local residual-stream scale. Spearman's correlation is computed separately for each model–depth block using its 42 ordered behavior pairs. The aggregate $\rho _ { S }$ is the mean across all 28 blocks.

For significance testing, a permutation acts on the seven semantic behavior identities rather than individual points. The same permutation is applied synchronously to every model-depth block and the aggregate signed Spearman statistic is recomputed. Enumerating all 7! permutations preserves the repeated observation structure of the 42 ordered relations in each block.

![](images/353d33e73cef595f4281aa079a2bd02f2a0a60c285598aec05d8dd8da1e51a7b.jpg)

![](images/79bdbb9bb4400340471f819fc3a382c22f9f57723f53b9ca9400945750181823.jpg)

![](images/6deb97023a9eb460f7878fdc1ac167707b44041a2a3ec0e455fcae058d0f650a.jpg)

Figure 11: Specificity controls at frozen operating points. CAA is compared with norm-matched random directions (left), label-shuffled CAA directions (center), and off-target-KL-matched random directions (right). Bars report mean held-out directional behavioral gain with 95% confidence intervals for each model and across all 109 unique nonzero frozen interventions.
<table><tr><td>Control</td><td>Model</td><td>n</td><td>CAA mean</td><td>Null mean</td><td>Paired diff.</td><td> $\mathbf { C A A } > q . 9 7 5$ </td><td>KL match</td></tr><tr><td>Norm random</td><td>Llama 2</td><td>23</td><td>0.168</td><td>0.004</td><td>0.164</td><td>95.7%</td><td></td></tr><tr><td>Norm random</td><td>Llama 3.1</td><td>32</td><td>0.088</td><td>-0.001</td><td>0.089</td><td>90.6%</td><td></td></tr><tr><td>Norm random</td><td>Mistral</td><td>30</td><td>0.163</td><td>0.002</td><td>0.161</td><td>100.0%</td><td></td></tr><tr><td>Norm random</td><td>Qwen2.5</td><td>24</td><td>0.081</td><td>-0.000</td><td>0.081</td><td>100.0%</td><td></td></tr><tr><td>Label shuffled</td><td>Llama 2</td><td>23</td><td>0.168</td><td>0.002</td><td>0.167</td><td>95.7%</td><td></td></tr><tr><td>Label shuffled</td><td>Llama 3.1</td><td>32</td><td>0.088</td><td>0.003</td><td>0.085</td><td>53.1%</td><td></td></tr><tr><td>Label shuffled</td><td>Mistral</td><td>30</td><td>0.163</td><td>0.009</td><td>0.154</td><td>70.0%</td><td></td></tr><tr><td>Label shuffled</td><td>Qwen2.5</td><td>24</td><td>0.081</td><td>-0.011</td><td>0.091</td><td>79.2%</td><td></td></tr><tr><td>Off-target KL</td><td>Llama 2</td><td>23</td><td>0.168</td><td>0.014</td><td>0.154</td><td>91.3%</td><td>99.8%</td></tr><tr><td>Off-target KL</td><td>Llama 3.1</td><td>32</td><td>0.088</td><td>0.000</td><td>0.088</td><td>84.4%</td><td>99.2%</td></tr><tr><td>Off-target KL</td><td>Mistral</td><td>30</td><td>0.163</td><td>0.001</td><td>0.162</td><td>80.0%</td><td>82.0%</td></tr><tr><td>Off-target KL</td><td>Qwen2.5</td><td>24</td><td>0.081</td><td>0.001</td><td>0.079</td><td>83.3%</td><td>99.6%</td></tr></table>

Table 7: Model-wise specificity controls at frozen operating points. Means are held-out directional behavioral gains. $\mathrm { ^ { * } C A A } > q . 9 7 5 ^ { ^ { 3 } }$ is the fraction of operating points for which the CAA gain exceeds the 97.5th percentile of the corresponding null distribution. KL-match rate applies only to the off-target-KL control; pooled results are reported in Section 5.3.

Figure 12 gives the model-wise relationships, including the corresponding $\rho _ { S }$ and exact permutation p-values. The association is positive in every model but remains substantially dispersed. Figure 13 shows that causal-transfer magnitude itself is sharply depth dependent.

Signed geometry and transfer therefore share a statistically detectable relationship, but geometry does not determine causal magnitude. The unsigned analysis is null $( \rho _ { S } = 0 . 0 4 2 , p _ { \mathrm { p e r m } } =$ 0.598), while causal efficacy itself changes sharply with depth. Cosine geometry is therefore informative as a descriptor of interaction structure, but not as a substitute for intervention.

## F External Capability and OOD Generalization

## F.1 Secondary-Capability Details

The secondary capability audit uses the frozen $\varepsilon = 0 . 0 3$ frontier and is never consulted during selection. GSM8K uses the same fixed 64-question subset for every condition, eight fixed demonstrations, greedy generation, and flexible exact numerical accuracy. WikiText-2 uses 128 windows of 257 native-tokenizer tokens; 256 next-token targets are scored per window for 32,768 targets per model. Because tokenizers differ, absolute perplexity is not compared across models; only the within-model steered-minus-baseline change is interpreted.

The main text reports the average 6.63% WikiText-2 perplexity increase and 10.24 percentage-point GSM8K accuracy decrease. Table 8 gives the finer model-wise association with residual-relative intervention magnitude. The ∆NLL relationship is positive within every model and strongest for Llama 3.1; GSM8K loss also tends to increase with dose but exhibits a weaker

![](images/c5b894e60603a8a0436b93d05ac1689d32dfeb213689686a0ea61a8b5a73ac1b.jpg)

Llama 3.1 — Geometry vs. Transfer  
![](images/81297c9278c5ab26d0773e4bc12304d59ba23186fd0a525398136cbb02ad3a3c.jpg)

![](images/a03199a447fccdd454b917085c5c03097e9e199ee0073c86d732dd66b17a28ba.jpg)

Qwen 2.5 — Geometry vs. Transfer  
![](images/9195abda2cd647909a9f43d0d914ee4d728fea9716013e495e9d3e9f226313f1.jpg)  
Figure 12: Per-model signed geometry-transfer relationship. Each panel contains all seven depth anchors and 42 ordered source-target relations per anchor. Positive association appears in every model, but with substantial dispersion.

![](images/65b846c409101e75bc0c0f59cc4c6a3ec9989157458877ce3610b3b2379d2ead.jpg)  
Figure 13: Causal-transfer magnitude by fractional depth. Mean absolute cross-behavior transfer peaks near fractional depth 0.42, where the pooled mean absolute transfer is approximately 0.036; the remaining anchors lie between approximately 0.006 and 0.016.

rank association

## F.2 Out-of-Distribution Generalization Details

The OOD experiments preserve the layer and coefficient choices from the $\varepsilon = 0 . 0 3$ protocol and introduce no target-domain tuning. For sycophancy, source-domain directions are constructed separately on NLP and political examples and evaluated both in-domain and on the opposite held-out domain. The target sets contain 164 NLP and 136 political examples. Table 9 reports all 16 model– direction transfer conditions. Fourteen retain the intended OOD sign, and nine have a 95% confidence interval strictly in the intended direction.

For hallucination, the canonical frozen intervention is evaluated on 150 binary TruthfulQA examples that are absent from vector learning, layer selection, and frontier selection. Table 10 shows the complete model-direction breakdown. Seven of eight conditions move in the intended direction and all seven of those effects have confidence intervals excluding zero; the exception is negative steering for Qwen2.5.

<table><tr><td colspan="2" rowspan="2">Model n</td><td colspan="2">GSM8K loss vs. |η|</td><td colspan="2">∆NLL vs. |η|</td></tr><tr><td>Pearson r</td><td>Spearman ρ</td><td>Pearson r</td><td>Spearman ρ</td></tr><tr><td>Llama 2</td><td>14</td><td>0.355</td><td>0.502</td><td>0.711</td><td>0.562</td></tr><tr><td>Llama 3.1</td><td>14</td><td>0.820</td><td>0.491</td><td>0.942</td><td>0.908</td></tr><tr><td>Mistral</td><td>14</td><td>0.662</td><td>0.534</td><td>0.886</td><td>0.724</td></tr><tr><td>Qwen2.5</td><td>14</td><td>0.353</td><td>0.473</td><td>0.727</td><td>0.708</td></tr><tr><td>Overall</td><td>56</td><td>0.706</td><td>0.572</td><td>0.632</td><td>0.819</td></tr></table>

Table 8: Association between intervention dose and secondary-capability change. GSM8K is represented as capability loss, so positive association means larger |η| tends to accompany larger accuracy loss. Positive ∆NLL likewise corresponds to worse language-modeling quality.
<table><tr><td>Model</td><td>Transfer</td><td>Direction</td><td>ID gain [95% CI]</td><td>OOD gain [95% CI]</td><td>OOD supported</td></tr><tr><td>Llama 2</td><td>NLP → Political</td><td>negative</td><td>0.132 [0.090,0.174]</td><td>0.008 [-0.027,0.041]</td><td>no</td></tr><tr><td>Llama 2</td><td>NLP → Political</td><td>positive</td><td>-0.012 [-0.044,0.020]</td><td>0.009 [-0.037,0.055]</td><td>no</td></tr><tr><td>Llama 2</td><td>Political → NLP</td><td>negative</td><td>0.038 [0.009,0.066]</td><td>0.036 [0.011,0.062]</td><td>yes</td></tr><tr><td>Llama 2</td><td>Political → NLP</td><td>positive</td><td>0.030 [-0.009,0.069]</td><td>-0.009 [-0.031,0.013]</td><td>no</td></tr><tr><td>Llama 3.1</td><td>NLP → Political</td><td>negative</td><td>0.264 [0.217,0.314]</td><td>0.070 [0.045,0.096]</td><td>yes</td></tr><tr><td>Llama 3.1</td><td>NLP → Political</td><td>positive</td><td>0.023 [0.015,0.032]</td><td>0.011 [0.004,0.017]</td><td>yes</td></tr><tr><td>Llama 3.1</td><td>Political → NLP</td><td>negative</td><td>0.032 [0.016,0.049]</td><td>0.085 [0.066,0.105]</td><td>yes</td></tr><tr><td>Llama 3.1</td><td>Political → NLP</td><td>positive</td><td>0.010 [0.003,0.018]</td><td>0.012 [0.009,0.017]</td><td>yes</td></tr><tr><td>Mistral</td><td>NLP → Political</td><td>negative</td><td>0.037 [0.013,0.066]</td><td>-0.026 [-0.055,-0.002]</td><td>no</td></tr><tr><td>Mistral</td><td>NLP → Political</td><td>positive</td><td>0.017 [-0.001,0.039]</td><td>0.004 [-0.020,0.033]</td><td>no</td></tr><tr><td>Mistral</td><td>Political → NLP</td><td>negative</td><td>-0.017 [-0.042,0.004]</td><td>0.016 [0.0002,0.036]</td><td>yes</td></tr><tr><td>Mistral</td><td>Political → NLP</td><td>positive</td><td>0.000 [-0.019,0.020]</td><td>0.017 [-0.001,0.040]</td><td>no</td></tr><tr><td>Qwen 2.5</td><td>NLP → Political</td><td>negative</td><td>0.036 [0.018,0.057]</td><td>0.085 [0.063,0.108]</td><td>yes</td></tr><tr><td>Qwen 2.5</td><td>NLP → Political</td><td>positive</td><td>0.009 [-0.006,0.026]</td><td>0.021 [0.009,0.035]</td><td>yes</td></tr><tr><td>Qwen 2.5</td><td>Political → NLP</td><td>negative</td><td>0.058 [0.043,0.074]</td><td>0.010 [0.003,0.018]</td><td>yes</td></tr><tr><td>Qwen 2.5</td><td>Political → NLP</td><td>positive</td><td>0.017 [-0.002,0.036]</td><td>0.005 [-0.004,0.015]</td><td>no</td></tr></table>

Table 9: Complete cross-domain sycophancy transfer results. Directional gains are defined so positive values indicate movement in the intended steering direction. “Supported"requires the OOD 95% confidence interval to exclude zero in that direction.

<table><tr><td>Model</td><td>Dir.</td><td>Gain</td><td>95% CI</td><td>Supp.</td></tr><tr><td>Llama 2</td><td>negative</td><td>0.092</td><td>[0.051,0.135]</td><td>yes</td></tr><tr><td>Llama 2</td><td>positive</td><td>0.075</td><td>[0.046,0.105]</td><td>yes</td></tr><tr><td>Llama 3.1</td><td>negative</td><td>0.042</td><td>[0.022,0.062]</td><td>yes</td></tr><tr><td>Llama 3.1</td><td>positive</td><td>0.037</td><td>[0.019,0.056]</td><td>yes</td></tr><tr><td>Mistral</td><td>negative</td><td>0.051</td><td>[0.020,0.085]</td><td>yes</td></tr><tr><td>Mistral</td><td>positive</td><td>0.035</td><td>[0.010,0.065]</td><td>yes</td></tr><tr><td>Qwen 2.5</td><td>negative</td><td>-0.018</td><td>[-0.045,0.007]</td><td>no</td></tr><tr><td>Qwen 2.5</td><td>positive</td><td>0.037</td><td>[0.013,0.066]</td><td>yes</td></tr></table>

Table 10: Hallucination steering transferred to TruthfulQA.

## G Open-Ended Generation Details

The open-ended audit contains 24 prompts per behavior and evaluates negative steering, the unsteered baseline, and positive steering under the frozen $\varepsilon ~ = ~ 0 . 0 3$ intervention. The initial design therefore contains $4 \times 3 \times 2 4 = 2 8 8$ matched prompt triplets, or 864 responses. After the generation-stability procedure, 276 complete triplets (828 responses) remain for blinded behavioral judging.

![](images/db57e97ae0a167dc7bc9104bca9fead0522deba09ee848da7228ada82b0d7bcb.jpg)  
Figure 14: Open-ended judge trajectories by model. Scores are pooled over sycophancy, refusal, and hallucination within each model and shown for the frozen negative, baseline, and positive conditions.

Judge configuration: Open-ended evaluations used GPT-5.6 Sol $\left( \mathsf { g p t } \mathsf { - } 5 . 6 \mathsf { - } \mathsf { s o l } \right)$ with reasoning effort set to 1ow and a maximum of 1,200 output tokens.

![](images/02c07b1fd911d4a2e213672645a2ec2764a47bf3c303a6ab68653d98b416cae8.jpg)

Figure 15: Open-ended sycophancy trajectories by prompt domain. NLP prompts show a stronger monotonic response than political prompts under the same frozen intervention protocol.
<table><tr><td>Model</td><td>Behavior</td><td>Condition</td><td>Persistent / 24</td></tr><tr><td>Llama 3.1</td><td>Refusal</td><td>Negative</td><td>7 (29.2%)</td></tr><tr><td>Llama 3.1</td><td>Hallucination</td><td>Positive</td><td>5 (20.8%)</td></tr></table>

Table 11: Persistent nontermination after 4096-token confirmation. These 12 failures account for the 12 excluded prompt triplets.

The aggregate behavior trajectories are reported in Section 5.7. Figure 14 provides the complementary model-wise breakdown, showing that the pooled response does not arise from a single architecture while also revealing substantial differences in absolute score level and intervention symmetry. Figure 15 separates NLP and political sycophancy prompts: the NLP subset exhibits a substantially larger ordered shift, while the political subset retains the same aggregate ordering with wider uncertainty.

Generation stability: A response that reaches the initial 2056-token cap is regenerated under the identical frozen intervention with a 4096-token confirmation cap. A matched triplet is excluded before judging if any of its three conditions still fails to terminate. Persistent failures occur only for Llama 3.1 in the two conditions listed in Table 11. The full triplet, rather than only the failed response, is removed so that paired comparisons remain welldefined.

Qualitative examples: The following examples are selected post hoc from the already frozen and judged response set to illustrate high-contrast behavior. They are not used as quantitative evidence and are not intended to represent average model behavior.

<table><tr><td>Component</td><td>Version / configuration</td></tr><tr><td>Python</td><td>3.11.11</td></tr><tr><td>PyTorch</td><td>2.5.1+cu118</td></tr><tr><td>Transformers</td><td>4.46.3</td></tr><tr><td>Datasets</td><td>3.1.0</td></tr><tr><td>Hugging Face Hub</td><td>0.36.2</td></tr><tr><td>Numerical precision</td><td>float32</td></tr><tr><td>Bootstrap samples</td><td>10,000</td></tr><tr><td>Base seed</td><td>42</td></tr></table>

Table 12: Core software and statistical environment.

## H Compute and Reproducibility

All experiments are inference-only: no model parameters are updated during vector construction, steering, capability evaluation, or post-selection audits. Behavior vectors are extracted independently for each model and are never transferred across architectures.

The project environment used the software versions in Table 12. The PyTorch build targets CUDA 11.8 (“+cu118"); this identifies the CUDA runtime against which the installed PyTorch package was built rather than asserting a particular cluster driver version.

Experiments were executed on the University of Chicago SSCS SLURM GPU cluster with NVIDIA L40S and H100 accelerators. The evaluation design reduces cross-stage leakage by fixing the protocol order before final testing. Behavior vectors are constructed from $\mathcal { D } _ { b } ^ { \mathrm { v e c } }$ , the intervention layer is selected using Dval, and the coefficient is chosen under the capability constraint. The resulting intervention is then frozen before evaluation on Dtest, capability holdout, and post-selection audits. Held-out and post-selection outcomes never trigger reselection of the behavior direction, layer, or coefficient.

Code Availability: The codebase, experiment configurations, and plotting scripts will be made publicly available at https://github.com/ deepanshu-Raj/CAA-SteerFrontier upon release of this preprint.

![](images/5b2dbe3989629e00821001bbd599cff2e83dead3cd911ef107b0a987860bb415.jpg)  
Figure 16: High-contrast qualitative example of sycophancy steering. Negative steering produces a more independent response, while positive steering increasingly adopts the user's stated position. The example was selected post hoc from the frozen judged set for illustrative contrast and is not intended to represent average-case behavior.

![](images/d1416dedf60dc5c44722f1d0ac0675bad5e50f43cb7601d323ab70c18f7527b0.jpg)  
Figure 17: High-contrast qualitative example of refusal steering. Negative steering weakens refusal and produces substantive tax-avoidance suggestions, whereas the baseline and positive conditions maintain clean refusals. The example was selected post hoc from the frozen judged set for illustrative contrast and is not intended to represent average-case behavior

![](images/3674b0b9db6a42cd23732e7b32e4e641196c943273a5bf2ddd753f037d380299.jpg)  
Figure 18: High-contrast qualitative example of hallucination steering. The negatively steered response challenges the premise, whereas the positively steered response makes a stronger unsupported factual claim. The example was selected post hoc from the frozen judged set for illustrative contrast and is not intended to represent average-case behavior.