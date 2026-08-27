# Reflection Steering: Disentangling Reflection from Reasoning in Activation Space for Token-Eficient Inference

Hu Jiarui<sup>1∗</sup>, Zhiyuan Wen<sup>2∗</sup>, Xiaoyun Liu<sup>2</sup>, Jiaxing Shen<sup>3</sup>, Yu Yang<sup>4</sup>

<sup>1</sup>School of Computing and Data Science, The University of Hong Kong, Hong Kong SAR, China

<sup>2</sup>Department of Computing, The Hong Kong Polytechnic University, Hong Kong SAR, China

<sup>3</sup>School of Data Science, Lingnan University, Hong Kong SAR, China

<sup>4</sup>Centre for Learning, Teaching and Technology, The Education University of Hong Kong, Hong Kong SAR, China

tonny\_hujiarui@163.com; zhiyuan.wen@polyu.edu.hk;

xiaoyun.liu@connect.polyu.hk; jiaxingshen@ln.edu.hk;

yangyy@eduhk.edu.hk

## Abstract

Large reasoning models often produce reasoning traces with verification, revision, and backtracking. When reflection merely re-checks established results, it wastes reasoning tokens and increases latency. Most existing reflection steering methods add a label-derived mean-diference direction across preset layers, but its entanglement with reasoning and length signals destabilizes the accuracy–eficiency trade-of. In this paper, we propose Reflection Steering, a training-free framework for controlling reflection-associated computation within LLMs by disentangling reflection-related activations from general reasoning. Specifically, we contrast reflective and nonreflective hidden states at each LLM layer, denoise the resulting reflection directions with PCA, and orthogonalize them against general-reasoning directions. To limit downstream amplification from early-layer interventions, we calibrate each layer across multiple intervention strengths on a small set, retain only stable layers, and apply bounded projection removal to their residual-stream activations. We conduct extensive experiments across two public benchmarks and three openweight LLMs against state-of-the-art activation-steering baselines. Results show that Reflection Steering reduces reasoning tokens by 16.9% on average across six matched settings. Besides, our method further introduces a bounded reflection intervention-strength parameter α, enabling deployment-time adjustment to balance token savings, accuracy, and generation stability.

## Introduction

Large reasoning models improve reliability by verifying intermediate results, revising earlier steps, backtracking, and changing strategies within extended reasoning traces (Wei et al. 2022; Snell et al. 2024; DeepSeek-AI 2025). Yet reflection can outlast its utility. After a conclusion is already supported by the preceding trace, a model may repeat the same checks without changing its answer, increasing token usage, latency, and serving cost and sometimes contributing to overthinking (Chen et al. 2024; Sui et al. 2025), as shown in Figure 1. We therefore ask whether a large reasoning model can be equipped with a tunable interface for regulating reflection and reducing its associated computation.

![](images/e6240dee5b0eb39f082a9c13924d685a6a97c69ed9d086a4ccf1f2f96b6c83b6.jpg)  
Figure 1: A motivating example of Reflection Steering. The raw model reaches the correct result but continues to verify it several times. By controlling reflection-associated activations, Reflection Steering produces a shorter trace that reaches the same answer with fewer thinking tokens in this example.

Existing approaches reduce reasoning cost through either output-level constraints or representation-level interventions. Output-level methods use token budgets, token pruning, or length-aware training to shorten reasoning traces, but they regulate overall length without distinguishing the computation being removed (Muennighof, Yang, and Shi 2025; Xia et al. 2025; Xu et al. 2025; Luo, Shen, and He 2025). Representation-level methods exploit approximately linear residual-stream directions (Zou et al. 2023; Turner et al. 2023; Arditi et al. 2024; Park, Choe, and Veitch 2024), with recent extensions targeting overthinking, reflection, and reasoning eficiency (Huang et al. 2025; Chen et al. 2025; Yan, Sun, and Weng 2025; Zhang et al. 2025b; Azizi et al. 2026). These methods enable targeted control, but depend on a steering direction that isolates the intended behavior. A contrastive estimate may mix reflection-associated signals with reasoning-related structure and other variation, making it dificult to reduce reflection without afecting useful reasoning.

We hypothesize that naive reflection steering may harm useful reasoning because reflection-associated and reasoning-related signals are entangled in activation space. Reflective and non-reflective states difer not only in reflection but also in reasoning content, token position, and trace length. Consequently, their mean diference may combine reflection-associated signals with activation structure shared across reasoning states. Suppressing this mixed direction may therefore reduce reflection while also weakening useful reasoning. This analysis motivates our core intuition: separate reflection-associated signals from shared reasoningrelated structure before intervention and focus control on the remaining component.

Building on this intuition, we propose Reflection Steering, a training-free representation-level method for reducing reflection-associated computation while limiting interference with general reasoning. Our method addresses three questions: what to intervene, where to intervene, and how to intervene. For what to intervene, we compare layer-wise activations from a small set of reflective and non-reflective examples to estimate an initial reflection direction. We then use PCA to reduce noise and orthogonalization to remove the part aligned with an activation pattern shared by both groups. For where to intervene, we test candidate layers at several strengths on a small calibration set and retain those that consistently reduce reflection-related behavior without destabilizing generation. For how to intervene, we scale down only the component of each token state that lies along the learned direction, rather than applying the same fixed shift to every token. Once calibrated, the direction and layer set remain fixed, while a reflection intervention-strength parameter α controls the degree of projection removal. Model weights, routing parameters, and other decoding settings remain unchanged.

We evaluate Reflection Steering on MATH-500 for mathematical problem solving (Hendrycks et al. 2021) and GPQA-Diamond for graduate-level scientific reasoning (Rein et al. 2024). Experiments span three Qwen-family open-weight checkpoints under matched answer parsing and token accounting. We compare against two closely related representation-level baselines: CREST for cognitivebehavior steering (Zhang et al. 2025b) and ReflCtrl for reflection control (Yan, Sun, and Weng 2025), using their reproduced default operating points. Across six matched model– task settings, Reflection Steering reduces reasoning tokens by 16.9% on average. On Qwen3-30B-A3B with MATH-500, it reduces tokens by 21.8% with a −0.1-point accuracy change; paired equivalence testing supports equivalence to the raw model within a ±1-point margin. Overall, the strongest accuracy-preserving evidence appears on mathematical reasoning. To summarize, our contributions are:

• We show that raw mean-diference reflection directions retain substantial overlap with a rank-one proxy for activation structure shared across reasoning states, and that proxy-based orthogonalization reduces this measured overlap to the isotropic-random level.

• We propose Reflection Steering, a training-free activationspace method combining low-rank filtering, proxy-based orthogonalization, layer calibration, and state-dependent projection attenuation. A deployment-time parameter α adjusts intervention strength without updating model weights or decoding settings.

• Across six model–benchmark pairs spanning three LLMs in diferent sizes and two benchmarks on math reasoning and science MCQ, Reflection Steering reduces thinking tokens by 16.9% on average while largely preserving task performance.

## Related Work

Controlling reasoning cost outside the residual stream. One cluster reduces cost via the prompt, decoding, or weights. Prompt-budget and length-instruction methods ask the model to reason within a token budget (Muennighof, Yang, and Shi 2025; Han, Wang, and Fang 2024; Xu et al. 2025), but reasoning models often ignore such instructions as the chain of thought grows (Fu et al. 2025). Early-exit, token-skipping, and stopping rules cut cost directly (Xia et al. 2025; Zhang et al. 2025a) at the risk of removing reasoning that is still needed, and training- or RL-based approaches reshape the reasoning policy itself (Luo, Shen, and He 2025) but require weight changes and cannot attach to a frozen checkpoint. CGRS suppresses reflection triggers using model confidence (Huang et al. 2026), operating at the token level rather than on the underlying representation. Our method is complementary: training-free, leaving prompt and decoding untouched, and acting on the representation that produces reflection.

Activation steering for reflection and eficiency. Another cluster edits internal activations. Representation-engineering studies show behaviors often align with approximately linear directions modifiable at inference time (Zou et al. 2023; Turner et al. 2023; Li et al. 2023; Rimsky et al. 2024; Arditi et al. 2024; Park, Choe, and Veitch 2024), and projectionbased concept erasure gives a related geometric view via nullspace projections (Ravfogel et al. 2020; Belrose et al. 2023). Manifold Steering projects an overthinking direction onto a low-dimensional activation manifold, while SEAL distinguishes execution, reflection, and transition states before intervention (Huang et al. 2025; Chen et al. 2025). ASC learns a compression direction from paired verbose and concise traces under a distributional constraint (Azizi et al. 2026), and CREST identifies cognitive attention heads and intervenes on their hidden representations (Zhang et al. 2025b). ReflCtrl is closest to ours (Yan, Sun, and Weng 2025); the two difer in the activation granularity used to estimate the direction, the treatment of confounding reasoning signals, the selection of intervention layers, and the form of activation modification. Reflection Steering is designed to improve direction specificity and intervention stability.

## Method

Reflection Steering turns a noisy reflection contrast into a fixed inference-time controller. Figure 2 shows the four stages. First, we estimate a raw reflection direction at each layer. Next, we purify the direction so that it is less mixed with general reasoning and sample noise. We then calibrate the candidate layers, because the same intervention can behave diferently at diferent depths. Finally, we apply bounded projection removal only at the selected layers. The model weights and decoding settings remain unchanged.

![](images/7118f2fd4f071886bfca028f97c9a1e40313996e7d6b0033ac80daa67f1c5fb9.jpg)  
Figure 2: Reflection Steering overview. Stage 1 estimates a raw reflection direction from reflective and non-reflective residual states. Stage 2 purifies the direction with PCA denoising and orthogonalization against a general-reasoning direction. Stage 3 tests candidate layers over several intervention strengths and keeps only stable responders. Stage 4 applies bounded projection removal at the selected layers during decoding.

Stage 1: Estimating the reflection direction. Prior activation-steering methods often represent a behavior with the mean activation diference between positive and negative examples (Zou et al. 2023; Turner et al. 2023; Rimsky et al. 2024; Arditi et al. 2024; Yan, Sun, and Weng 2025). This gives a simple linear axis in the same residual-stream space where steering is later applied. We use the same idea as the initial estimate of reflection.

For layer $L ,$ let $R _ { L } = \{ r _ { i } ^ { ( L ) } \} _ { i = 1 } ^ { n _ { R } }$ and $N _ { L } = \{ n _ { j } ^ { ( L ) } \} _ { j = 1 } ^ { n _ { N } }$ denote activations at reflective and non-reflective positions. The raw direction is

$$
d _ { L } ^ { \mathrm { r a w } } = \frac { 1 } { n _ { R } } \sum _ { i = 1 } ^ { n _ { R } } r _ { i } ^ { ( L ) } - \frac { 1 } { n _ { N } } \sum _ { j = 1 } ^ { n _ { N } } n _ { j } ^ { ( L ) } .\tag{1}
$$

We use all labeled positions rather than only the first token of each reasoning step. A reflective step usually spans several tokens. Therefore, using the full span makes the estimate less dependent on one boundary token and better represents the activation pattern of the whole step. Still, the mean diference is only a first-order estimate. It can contain signals other than reflection, which motivates Stage 2.

Stage 2: Purifying the raw contrast. The two activation groups difer in more than reflective behavior. Their mean diference can also contain prompt and length signals, general reasoning activity, and finite-sample noise. Direct steering along this mixed direction can therefore remove useful information.

We first reduce noise by restricting the raw direction to the main activation subspace. Let

$$
\bar { h } _ { L } = \frac { \sum _ { i } r _ { i } ^ { ( L ) } + \sum _ { j } n _ { j } ^ { ( L ) } } { n _ { R } + n _ { N } } , \qquad X _ { L } = \left[ h - \bar { h } _ { L } \right] _ { h \in { \cal R } _ { L } \cup N _ { L } } ,\tag{2}
$$

where $\bar { h } _ { L }$ is the mean of all labeled activations at layer L. Let $Q _ { L } \in \mathbb { R } ^ { d \times k }$ contain the leading k right singular vectors of $X _ { L }$ . We project the raw direction onto this subspace:

$$
\begin{array} { r } { \tilde { d } _ { L } = Q _ { L } Q _ { L } ^ { \top } d _ { L } ^ { \mathrm { r a w } } . } \end{array}\tag{3}
$$

By the Eckart–Young theorem, this is the optimal rank-k subspace for reconstructing $X _ { L }$ in the least-squares sense (Eckart and Young 1936). We use it as a low-rank prior on the contrast. As a result, components outside the dominant activation subspace are reduced, while the main structure of the contrast is kept.

PCA alone does not separate reflection from reasoning activity shared by both groups. We therefore use the pooled mean to define a layer-specific general-reasoning direction:

$$
\mu _ { L } = \frac { \bar { h } _ { L } } { \| \bar { h } _ { L } \| _ { 2 } } .\tag{4}
$$

Both reflective and non-reflective positions come from reasoning traces. Thus, their pooled mean gives a simple estimate of the activation component shared by the two groups. It is only a rank-one proxy, not a complete model of general reasoning.

![](images/c892fb3b995b49c2e2daa558f779249b7aa3b86646dcba153d62be2900f056f0.jpg)  
Figure 3: Downstream amplification of residual-stream interventions. Median final-layer gain over five teacherforced MATH traces; ticks below $G = 1$ mark calibrated layers.

Next, we remove the part of $\tilde { d } _ { L }$ aligned with this shared direction:

$$
d _ { L } = \frac { ( I - \mu _ { L } \mu _ { L } ^ { \top } ) \tilde { d } _ { L } } { \left\| ( I - \mu _ { L } \mu _ { L } ^ { \top } ) \tilde { d } _ { L } \right\| _ { 2 } } .\tag{5}
$$

Before normalization, this is the orthogonal projection of $\tilde { d } _ { L }$ onto the nullspace of $\mu _ { L } ^ { \top }$ . In other words, it is the closest vector to $\tilde { d } _ { L }$ with zero linear overlap with the estimated shared direction. This operation follows the geometry of projectionbased concept erasure (Ravfogel et al. 2020; Belrose et al. 2023). Our claim is narrower: it removes one measured source of entanglement rather than fully separating reflection from all general reasoning.

Stage 3: Calibrating intervention layers. Even a small residual-stream edit can change later computation. Let $F _ { s \to k }$ denote the computation from source layer s to a later layer k. For a small intervention $\delta h _ { s }$ s

$$
\delta h _ { k } = F _ { s  k } \big ( h _ { s } + \delta h _ { s } \big ) - F _ { s  k } \big ( h _ { s } \big ) \approx J _ { s  k } \delta h _ { s } ,\tag{6}
$$

where $J _ { s \to k }$ is the downstream Jacobian along the current trajectory. Since its norm need not be below one, a local edit can persist or grow through later layers, and edits at several layers can accumulate. We verify this with a teacher-forced audit on five held-out MATH traces. We hold each 2,048- token prefix fixed and intervene at one source layer at a time. For source layer s and downstream layer k, we measure

$$
G _ { \alpha } ( s \to k ) = \frac { \| \Delta h _ { k } ( s , \alpha ) \| _ { 2 } } { \| \Delta h _ { s } ( s , \alpha ) \| _ { 2 } + \varepsilon } ,\tag{7}
$$

where $\Delta h _ { k }$ is the diference between the intervened and unmodified residual states. For every tested non-baseline $\alpha ,$ 43 of 44 source layers have a final-layer gain above one, and earlier interventions tend to be amplified more strongly (Figure 3). Thus, both intervention depth and strength matter. We therefore calibrate layers instead of steering all layers in the same way.

We calibrate each candidate layer on a small calibration subset. Let $m _ { 0 } ( q )$ be the raw-model reflection-proxy count for example q, and let $m _ { L , \alpha } ( q )$ be the count when only layer L is intervened. The mean reduction is

$$
r _ { L } ( \alpha ) = \frac { 1 } { \vert \mathcal { C } \vert } \sum _ { q \in \mathcal { C } } \bigl [ m _ { 0 } ( q ) - m _ { L , \alpha } ( q ) \bigr ] .\tag{8}
$$

A layer is retained only when three conditions hold. First, the reduction should remain approximately monotonic as α decreases, so a stronger intervention gives a consistent response. Second, every tested non-baseline strength must produce at least a small positive reduction. This avoids selecting layers whose measured efect is only proxy noise or is locally negligible, because even an unhelpful local edit can still propagate downstream. Third, the calibration runs must not trigger generation collapse. These checks address three diferent risks: an inconsistent response, an inefective local edit, and unstable generation.

The selected layers are therefore reliable only within the tested range. Calibration tests one layer at a time and uses a reflection proxy. It does not certify final-answer accuracy, and it does not guarantee that the strongest intervention across several selected layers will preserve general reasoning. We evaluate those properties on the complete multi-layer controller.

Stage 4: Bounded projection removal. The final choice is how to change the activation. Additive controllers such as ReflCtrl apply a fixed displacement along a steering direction (Yan, Sun, and Weng 2025). The same update is added even when the current activation contains little reflection-related signal. A large fixed shift can therefore move the state away from its current representation path. Our goal is suppression rather than injection, so we remove only the component that is already present along the purified direction.

For a selected layer L with unit direction $d _ { L }$ and current activation $h ^ { ( L ) } \in \bar { \mathbb { R } } ^ { d }$ , we apply

$$
h ^ { \prime ( L ) } = h ^ { ( L ) } - ( 1 - \alpha ) d _ { L } d _ { L } ^ { \top } h ^ { ( L ) } ,\tag{9}
$$

where $\alpha \in [ 0 , 1 ]$ parameterizes the reflection intervention strength. The setting $\alpha = 1$ leaves the activation unchanged, whereas smaller values apply a stronger intervention by retaining less of the current projection onto $d _ { L }$ . Thus, the update adapts to the current state: it is small when the reflectionrelated component is small and larger when that component is strong.

For unit $d _ { L } .$ , the operator has eigenvalue α along $d _ { L }$ and eigenvalue 1 on its orthogonal complement. Therefore,

$$
\| h ^ { \prime ( L ) } \| _ { 2 } \leq \| h ^ { ( L ) } \| _ { 2 } .\tag{10}
$$

This bound applies to the local edit. It does not guarantee that the downstream efect or the final sequence length will change monotonically, because the perturbation can still propagate through later layers as shown in Equation 6.

The operator is applied at every decoding step only to the selected layers. Model weights, routing parameters, sampling settings, and answer parsing remain unchanged. The projection form builds on prior work on linear representation directions. Our contribution lies in how the direction is estimated, purified, calibrated by layer, and then used for bounded removal.

## Experimental Setup

To validate the method, we organize the setup around the experimental design and the implementation details.

Benchmarks. We evaluate on MATH-500 (Hendrycks et al. 2021) and GPQA-Diamond (Rein et al. 2024). Exact normalized-record auditing shows that the 150 directionlearning inputs are MATH-500 records 0–149, with the 20 calibration inputs nested inside them, and zero exact overlap with GPQA-Diamond. Full MATH-500 is therefore insample for direction construction, so we additionally report a record-disjoint MATH-350 split (records 150–499) that no fitting or calibration record touches; GPQA-Diamond is fitdisjoint by construction. This lets us separate in-sample behavior, record-disjoint sensitivity, and cross-task transfer.

Models and comparison methods. The raw model is Qwen3-30B-A3B; Qwen3-8B and QwQ-32B are crosscheckpoint transfers. We focus on Qwen-family open-weight checkpoints because they provide the internal activation access required by our method while supporting controlled comparisons across model scales; transfer beyond this family remains untested. As external eficiency comparators under identical accounting, we reproduce two representative activation-level controllers: CREST on Qwen3-30B-A3B and ReflCtrl on all three models, each at its default configuration. Comparisons are default-point rather than equalizedtuning frontiers.

Evaluation protocol and metrics. We evaluate three aspects. For eficiency we report mean thinking tokens and token reduction against the matched raw model $( \alpha = 1 )$ For task performance we report strict accuracy and, for equivalence testing, a cluster-level paired TOST at a ±1- point margin. For generation stability we report the collapse (repetition-loop) rate and the unparsable (no-box) rate. We summarize the accuracy–cost trade-of with an eficiency ratio $\rho = \Delta \mathrm { T o k } / | \Delta \mathrm { A c c } |$ (higher is better). We use “—” for all unavailable or undefined entries; token increases yield negative ρ. Reflection markers (verification, uncertainty, backtracking, self-correction, strategy switching) are used only as calibration and interpretability proxies, not as success metrics.

Implementation details. All runs use temperature 0.6, top-p 0.95, and a 32,768-token cap. The 30B contrast uses reflection-encouraging versus commit-withoutrevisiting prompts; 8B/32B instead use regex-labeled reflective versus clean steps because their extended-thinking phases are less prompt-responsive (Qwen Team 2025; Fu et al. 2025). Calibration tests $\alpha \in \{ 0 . 7 , 0 . 3 , 0 . 0 \}$ on Qwen3 and a checkpoint-specific grid on QwQ-32B, retaining only layers with monotone proxy reduction, a minimum efect, and no collapse. This selects 14/44, 2/34, and 6 layers on 30B, 8B, and 32B, respectively. For the main Qwen3-30B-A3B comparisons, we keep one direction and layer set fixed and average decoding seeds 0, 64, and 128. Accuracy intervals and TOST use question-cluster bootstraps preserving withinseed pairing; table deltas use unrounded means, whereas $\rho$ uses displayed values. Because $\alpha = 0 . 7$ was selected retrospectively from task summaries, prospective evidence comes from a separate hash-frozen pilot.

## Results

We organize the evaluation around three research questions: whether Reflection Steering improves the accuracy– cost trade-of (RQ1), why each component of the method is necessary (RQ2), and whether the method is stable across tasks, models, and sampling (RQ3).

## RQ1: Does Reflection Steering improve the accuracy–cost trade-of?

Table 1 compares one fixed controller with prior methods under identical parsing and token accounting on MATH-500, record-disjoint MATH-350, and GPQA-Diamond.

Reflection Steering gives the strongest or near-strongest trade-of on all three splits. It cuts thinking tokens by 21.8% on MATH-500 with a −0.1-point accuracy change, and by 23.4% on record-disjoint MATH-350 with a +0.1-point change; CREST instead increases thinking tokens on both MATH splits at its reproduced default. The slight MATH-500 decrease may reflect an accuracy–eficiency cost of reducing thinking tokens, although it remains within the reported ±1- point equivalence margin. On GPQA-Diamond, the same controller transfers without refitting and matches ReflCtrl’s mean accuracy while reducing more tokens (21.0% versus 17.7%).

Adjustable intervention strength. Table 2 varies the reflection intervention-strength parameter α. Smaller α generally removes more thinking tokens, but the strongest intervention does not always improve compression or stability. We use α = 0.7 because it captures most of the token saving with zero or near-zero collapse and the best overall trade-of across the three splits. Thus, α provides deployment-time control over the accuracy–cost trade-of rather than a “stronger is better” setting.

Trace-dependent compressibility. On paired MATH-500 seed-128 traces, raw-model trace length is negatively correlated with proportional token reduction, with Spearman $\rho = - 0 . 2 6 4 , - 0 . 3 7 0 , - 0 . 4 1 8 \mathrm { f o r } \alpha = 0 . 7 , 0 . 3 , 0 . 0 ,$ respectively. Thus, longer raw-model traces tend to undergo smaller proportional compression, indicating that trace length alone does not determine how much reasoning can be removed by a fixed intervention strength.

Multi-seed consistency and statistical validity. The token savings are consistent across seeds. At α = 0.7, reductions are 22.5%, 22.3%, and 20.7% on MATH-500 and 22.2%, 20.9%, and 19.8% on GPQA-Diamond, with zero collapse. Paired TOST passes on MATH-500 (−0.13 points, 90% CI [−0.60, +0.33], p = 0.0012) and MATH-350 $( + 0 . 1 0 , [ - 0 . 5 1 , + 0 . 7 0 ] , p = 0 . 0 0 7 4 )$ , supporting equivalence within ±1 point. GPQA-Diamond does not pass the same equivalence test (−1.52 points, [−4.38, 1.18]).

Prospective pilot on fresh data. We also fix the protocol before generation and validate on 500 fresh, overlap-filtered MATH-train problems. Thinking tokens fall by 21.0%, with a one-sided 95% lower bound of 17.6%, above the preregistered 10% target, and neither arm collapses. Accuracy changes by −0.60 points, with a one-sided lower bound of −1.69, which does not meet the one-point criterion. Overall, RQ1 shows a consistent eficiency gain and accuracy equivalence on the retrospective mathematical evaluations.

<table><tr><td>Benchmark</td><td>Method</td><td>Tok.↓</td><td>∆Tok. (%) ↑</td><td> $\mathrm { A c c . } ( \% ) \uparrow$ </td><td> $\Delta \mathsf { A c c . } ( \mathsf { p p } ) \uparrow$ </td><td>ρ↑</td><td>Coll. (%) ↓</td></tr><tr><td rowspan="4">MATH-500</td><td>Raw model</td><td> $4 6 7 9 \pm 6 7$ </td><td></td><td> $9 5 . 1 \pm 0 . 1$ </td><td></td><td></td><td>0.0</td></tr><tr><td>CREST</td><td> $5 0 5 2 \pm 1 2 7$ </td><td>-8.0</td><td> $9 4 . 8 \pm 0 . 3$ </td><td>-0.3</td><td>-26.7</td><td>0.2</td></tr><tr><td>ReflCtrl  $\scriptstyle ( \lambda = - 0 . 4 8 )$ </td><td> $4 0 3 1 \pm 2 3$ </td><td>13.9</td><td> $9 4 . 1 \pm 0 . 8$ </td><td>-1.0</td><td>13.9</td><td>0.3</td></tr><tr><td>Reflection Steering</td><td> $3 6 5 7 \pm 1 1$ </td><td>21.8</td><td> ${ \bf 9 4 . 9 \pm 0 . 8 }$ </td><td>-0.1</td><td>218.0</td><td>0.0</td></tr><tr><td rowspan="4">MATH-350 (disjoint)</td><td>Raw model</td><td> $4 7 9 6 \pm 7 5$ </td><td></td><td> $9 5 . 3 \pm 0 . 2$ </td><td></td><td></td><td>0.0</td></tr><tr><td>CREST</td><td> $5 1 9 2 \pm 1 4 8$ </td><td>-8.3</td><td> $9 5 . 3 \pm 0 . 3$ </td><td>0.0</td><td></td><td>0.0</td></tr><tr><td>ReflCtrl  $\scriptstyle ( \lambda = - 0 . 4 8 )$ </td><td> $4 1 0 9 \pm 6 6$ </td><td>14.3</td><td> $9 4 . 5 \pm 0 . 8$ </td><td>-0.9</td><td>15.9</td><td>0.3</td></tr><tr><td>Reflection Steering</td><td> ${ \bf 3 6 7 4 \pm 5 5 }$ </td><td>23.4</td><td> ${ \bf 9 5 . 4 \pm 0 . 8 }$ </td><td>+0.1</td><td>234.0</td><td>0.0</td></tr><tr><td rowspan="4">GPQA-D</td><td>Raw model</td><td> $7 0 9 1 \pm 2 4 2$ </td><td></td><td> $7 1 . 9 \pm 1 . 1$ </td><td></td><td></td><td>0.0</td></tr><tr><td>CREST</td><td> $6 8 4 9 \pm 5 2$ </td><td>3.4</td><td> $6 8 . 5 \pm 1 . 5$ </td><td>-3.4</td><td>1.0</td><td>0.2</td></tr><tr><td>ReflCtrl  $\scriptstyle ( \lambda = - 0 . 4 8 )$ </td><td> $5 8 3 6 \pm 1 1$ </td><td>17.7</td><td> $7 0 . 4 \pm 2 . 0$ </td><td>-1.5</td><td>11.8</td><td>0.0</td></tr><tr><td>Reflection Steering</td><td> $\mathbf { 5 6 0 5 \pm 1 } 3 6$ </td><td>21.0</td><td> $\mathbf { 7 0 . 4 \pm 0 . 6 }$ </td><td>-1.5</td><td>14.0</td><td>0.0</td></tr></table>

Table 1: Performance comparison on Qwen3-30B-A3B. Thinking-token and accuracy values are three-seed means ± standard deviations. Reflection Steering uses a reflection intervention strength of $\alpha = 0 . 7 ;$ CREST (Zhang et al. 2025b) and ReflCtrl (Yan, Sun, and Weng 2025) use reproduced defaults. Gray raw-model rows are references; bold marks the best controlled value per column, with only Reflection Steering bold on ties. “—” denotes an unavailable or undefined entry.

<table><tr><td>Benchmark</td><td>α</td><td>Tok. ↓</td><td>∆Tok. (%)↑</td><td>Acc. (%)↑</td><td>Coll. (%)↓</td></tr><tr><td rowspan="5">MATH-500</td><td>1.0</td><td>4679</td><td></td><td>95.1</td><td>0.0</td></tr><tr><td>0.7</td><td>3657</td><td>21.8</td><td>94.9</td><td>0.0</td></tr><tr><td>0.3</td><td>3292</td><td>29.6</td><td>94.6</td><td>0.2</td></tr><tr><td>0.0</td><td>3439</td><td>26.5</td><td>93.5</td><td>1.1</td></tr><tr><td>1.0</td><td>4796</td><td></td><td>95.3</td><td>0.0</td></tr><tr><td rowspan="4">MATH-350 (disjoint)</td><td>0.7</td><td>3674</td><td>23.4</td><td>95.4</td><td>0.0</td></tr><tr><td>0.3</td><td>3216</td><td>32.9</td><td>95.5</td><td>0.0</td></tr><tr><td>0.0</td><td>3389</td><td>29.3</td><td>94.4</td><td>0.8</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">GPQA-D</td><td>1.0</td><td>7091</td><td></td><td>71.9</td><td>0.0</td></tr><tr><td>0.7</td><td>5605</td><td>21.0</td><td>70.4</td><td>0.0</td></tr><tr><td>0.3</td><td>5437</td><td>23.3</td><td>69.7</td><td>0.2</td></tr><tr><td>0.0</td><td>5368</td><td>24.3</td><td>68.9</td><td>0.0</td></tr></table>

Table 2: Performance across intervention strengths. Gray α = 1.0 rows are raw-model references; the remaining rows report intervened settings within each benchmark under matched accounting. The deployed setting is $\alpha = 0 . 7 .$

## RQ2: Are purification, layer calibration, and projection removal necessary?

We next ask which design choices make the method work, by removing each in turn. Because direction construction and layer calibration use MATH-500 records, we conduct these component ablations on the fit-disjoint GPQA-Diamond split to test whether the components remain necessary beyond the construction data. Table 3 reports the ablations at $\alpha = 0 . 7 $

Purification. Table 3, Panel A tests direction purification. The full method matches raw-model accuracy while cutting tokens by 20.9%. Without PCA, token reduction falls to 15.8%. Without orthogonalization, compression increases but accuracy drops to $6 7 . 2 \%$ . Thus, PCA improves the direction estimate, while orthogonalization is important for preserving useful reasoning.

<table><tr><td>Variant</td><td>Tok. ↓</td><td>∆Tok.↑</td><td>Acc.↑</td><td>Coll. ↓</td></tr><tr><td colspan="5">Panel A: direction purification</td></tr><tr><td>Raw model</td><td>7250</td><td></td><td>70.7</td><td>0.0</td></tr><tr><td>PCA + orthogonalization</td><td>5737</td><td>20.9</td><td>70.7</td><td>0.0</td></tr><tr><td>No orthogonalization</td><td>4104</td><td>43.4</td><td>67.2</td><td>0.0</td></tr><tr><td>No PCA</td><td>6105</td><td>15.8</td><td>71.2</td><td>0.0</td></tr><tr><td colspan="5">Panel B: layer calibration and projection removal</td></tr><tr><td>Calibrated + projection</td><td>5737</td><td>20.9</td><td>70.7</td><td>0.0</td></tr><tr><td>All layers + projection</td><td>5651</td><td>22.1</td><td>66.7</td><td>0.5</td></tr><tr><td>Calibrated + additive</td><td>9268</td><td>-27.8</td><td>69.2</td><td>0.0</td></tr></table>

Table 3: Ablations on GPQA-Diamond. Full 198-question split, $\alpha = 0 . 7 .$

Direction specificity. Held-out GPQA-Diamond traces confirm this geometry (Figure 4). We report median rank-one overlap across the 14 deployed layers, with bars spanning the 5th–95th percentile. CREST (Zhang et al. 2025b) is omitted because it identifies reflection-related attention heads rather than a single positive–negative reflection direction; Random is an isotropic hidden-space reference for negligible structured overlap. Median overlap is 0.0332 for the raw direction and 0.0385 after PCA, compared with 0.00048 for Random. Orthogonalization reduces it to 0.00042. Thus, orthogonalization removes most of the measured reasoning-proxy overlap; PCA alone does not.

Layer calibration and projection removal. Table 3, Panel B tests layer calibration and the choice of projection removal over additive steering. Steering all candidate layers adds only 1.2 points of token reduction but lowers accuracy by 4.0 points, while an additive update increases tokens by

![](images/dd0799d290da4730daf36e223199a5dfd65d72c4e32cc68c153760bc43ee3b5e.jpg)  
Figure 4: Reasoning-subspace overlap of steering directions. Medians over 14 layers; bars span the 5th– 95th percentiles. RS is Reflection Steering; RS (Raw) is its unpurified mean-diference direction. Random is an isotropic negligible-overlap reference. PCA retains substantial reasoning-proxy overlap, whereas purification cuts it by 98.8% to the isotropic-random level. CREST has no comparable single direction.

<table><tr><td>Model</td><td>Benchmark Method</td><td></td><td></td><td>Tok. ↓ ∆Tok. (%) ↑ Acc. ↑</td><td></td></tr><tr><td rowspan="2">Qwen3-8B MATH-500 ReflCtrl</td><td rowspan="2"></td><td>Raw model</td><td>4345</td><td></td><td>93.6</td></tr><tr><td>RS</td><td>4449 4060</td><td>-2.4 6.6</td><td>93.6 93.2</td></tr><tr><td rowspan="3">Qwen3-8B GPQA-D</td><td rowspan="3"></td><td></td><td></td><td></td><td></td></tr><tr><td>Raw model ReflCtrl</td><td>8485 8292</td><td>2.3</td><td>57.6 56.1</td></tr><tr><td>RS</td><td>7670</td><td>9.6</td><td>56.1</td></tr><tr><td rowspan="3">QwQ-32B MATH-500 ReflCtrl</td><td rowspan="3"></td><td>Raw model</td><td>3617</td><td></td><td>75.6</td></tr><tr><td></td><td>2945</td><td>18.6</td><td>74.2</td></tr><tr><td>RS</td><td>2662</td><td>26.4</td><td>74.8</td></tr><tr><td rowspan="3">QwQ-32B GPQA-D</td><td rowspan="3"></td><td>Raw model</td><td>7599</td><td></td><td>65.2</td></tr><tr><td>ReflCtrl</td><td>7066</td><td>7.0</td><td>63.1</td></tr><tr><td>RS</td><td>6401</td><td>15.8</td><td>63.1</td></tr></table>

Table 4: Cross-model results. RS denotes Reflection Steering. All rows use matched thinking-token accounting.

27.8%. Calibration therefore decides where to intervene, and bounded projection removal controls how. Together with purification, these choices produce the balanced accuracy–cost behavior of the full method. This answers RQ2.

## RQ3: Does Reflection Steering transfer across models and benchmarks?

To answer this question, Table 4 re-runs the same pipeline with model-specific calibration, while Table 1 reuses the fixed 30B controller across tasks.

Cross-model applicability. Across Qwen3-8B and QwQ-32B, Reflection Steering continues to reduce thinking tokens. It saves 6.6% on MATH-500 and 9.6% on GPQA-D with 8B, and 26.4% and 15.8% with 32B, respectively. At the evaluated settings, these reductions exceed those of the reproduced ReflCtrl comparator in all four settings. The number of selected layers, the best α, and the achievable compression remain model-dependent.

![](images/307d0bd5cc29af49b178822935ab03f47f354452071d4a82857dbf45abc8c6c4.jpg)  
Figure 5: Direction stability versus construction budget. Median $1 - \cos ( d _ { N } , d _ { N + 1 0 } )$ over ten random orderings; the band spans the 5th–95th percentile.

Cross-Benchmark transfer. The 30B direction and calibration are learned from mathematical reasoning, then reused without refitting on GPQA-Diamond. The same controller still reduces tokens on this scientific-reasoning benchmark, supporting a reflection-associated representation that transfers beyond the construction task.

Direction-construction stability. Figure 5 shows that the direction changes less as the construction budget grows. The 140→150 and 150→160 cosines are 0.9942 and 0.9940, and N = 150 matches N = 200 at median cosine 0.9761. We therefore use N = 150 as an empirical operating point in this flatter region, rather than as a proven suficient budget.

Calibration-selected layers. Leave-one-out calibration yields an eight-layer stability core (L36, L38, L39, L41– L45). On GPQA-Diamond seeds 0 and 128 at α = 0.7, it retains 16.1% token reduction with zero collapse and a −1.52-point mean accuracy change. Thus, the efect is not tied to one calibration split. Overall, RQ3 shows that the framework transfers across tasks and checkpoints, while the selected layers and operating point remain model-specific.

## Conclusion

In this paper, we introduce Reflection Steering, a trainingfree residual-stream intervention combining direction purification, layer calibration, and bounded projection removal. Across three Qwen-family checkpoints and two benchmarks, it reduces thinking tokens in all six matched settings by 16.9% on average. Accuracy preservation is strongest on mathematical reasoning, while GPQA-Diamond shows cross-task token reduction without established one-point equivalence. Overall, the results support tunable activationspace control without weight updates.

Future work will extend Reflection Steering to broader model families, reasoning tasks, and decoding settings. It will also explore adaptive controllers that vary intervention strength across layers and decoding steps, aiming to remove redundant rechecking while preserving useful verification and correction.

## References

Arditi, A.; Obeso, O.; Syed, A.; Paleka, D.; Panickssery, N.; Gurnee, W.; and Nanda, N. 2024. Refusal in Language Models Is Mediated by a Single Direction. Advances in Neural Information Processing Systems (NeurIPS).

Azizi, S.; Baghaei Potraghloo, E.; Kundu, S.; and Pedram, M. 2026. Activation Steering for Chain-of-Thought Compression. In Findings of the Association for Computational Linguistics: ACL 2026, 36676–36687.

Belrose, N.; Schneider-Joseph, D.; Ravfogel, S.; Cotterell, R.; Raf, E.; and Biderman, S. 2023. LEACE: Perfect Linear Concept Erasure in Closed Form. In Advances in Neural Information Processing Systems, volume 36.

Chen, R.; Zhang, Z.; Hong, J.; Kundu, S.; and Wang, Z. 2025. SEAL: Steerable Reasoning Calibration of Large Language Models for Free. arXiv:2504.07986.

Chen, X.; Xu, J.; Liang, T.; and He, Z. 2024. Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs. arXiv:2412.21187.

DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948.

Eckart, C.; and Young, G. 1936. The Approximation of One Matrix by Another of Lower Rank. Psychometrika, 1(3): 211–218.

Fu, T.; Gu, J.; Li, Y.; Qu, X.; and Cheng, Y. 2025. Scaling Reasoning, Losing Control: Evaluating Instruction Following in Large Reasoning Models. arXiv:2505.14810.

Han, T.; Wang, Z.; and Fang, C. 2024. Token-Budget-Aware LLM Reasoning. arXiv:2412.18547.

Hendrycks, D.; Burns, C.; Kadavath, S.; Arora, A.; Basart, S.; Tang, E.; Song, D.; and Steinhardt, J. 2021. Measuring Mathematical Problem Solving with the MATH Dataset. In NeurIPS Datasets and Benchmarks Track.

Huang, J.; Lin, B.; Feng, G.; Chen, J.; He, D.; and Hou, L. 2026. Eficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression. Proceedings of the AAAI Conference on Artificial Intelligence, 40(37): 31176–31184.

Huang, Y.; Chen, H.; Ruan, S.; Zhang, Y.; Wei, X.; and Dong, Y. 2025. Mitigating Overthinking in Large Reasoning Models via Manifold Steering. In Advances in Neural Information Processing Systems, volume 38.

Li, K.; Patel, O.; Viégas, F.; Pfister, H.; and Wattenberg, M. 2023. Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. Advances in Neural Information Processing Systems (NeurIPS).

Luo, H.; Shen, L.; and He, H. 2025. O1-Pruner: Length-Harmonizing Fine-Tuning for o1-Like Reasoning Pruning. arXiv:2501.12570.

Muennighof, N.; Yang, Z.; and Shi, W. 2025. s1: Simple Test-Time Scaling. arXiv:2501.19393.

Park, K.; Choe, Y. J.; and Veitch, V. 2024. The Linear Representation Hypothesis and the Geometry of Large Language Models. International Conference on Machine Learning (ICML).

Qwen Team. 2025. QwQ-32B: Embracing the Power of Reinforcement Learning. https://qwenlm.github.io/blog/qwq-32b/. Accessed: 2026-07-28.

Ravfogel, S.; Elazar, Y.; Gonen, H.; Twiton, M.; and Goldberg, Y. 2020. Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection. In Proceedings of the 58th Annual Meeting ofthe Associationfor Computational Linguistics, 7237–7256.

Rein, D.; Hou, B. L.; Stickland, A. C.; Petty, J.; Pang, R. Y.; Dirani, J.; Michael, J.; and Bowman, S. R. 2024. GPQA: A Graduate-Level Google-Proof Q&A Benchmark. In Conference on Language Modeling (COLM).

Rimsky, N.; Gabrieli, N.; Schulz, J.; Tong, M.; Hubinger, E.; and Turner, A. M. 2024. Steering Llama 2 via Contrastive Activation Addition. In Proceedings ofthe Annual Meeting ofthe Association for Computational Linguistics (ACL).

Snell, C.; Lee, J.; Xu, K.; and Kumar, A. 2024. Scaling LLM Test-Time Compute Optimally Can Be More Efective Than Scaling Model Parameters. arXiv:2408.03314.

Sui, Y.; Chuang, Y.-N.; Wang, G.; Zhang, J.; Zhang, T.; Yuan, J.; Liu, H.; Wen, A.; Zhong, S.; Chen, H.; and Hu, X. 2025. Stop Overthinking: A Survey on Eficient Reasoning for Large Language Models. arXiv:2503.16419.

Turner, A. M.; Thiergart, L.; Leech, G.; and Udell, D. 2023. Steering Language Models with Activation Engineering. arXiv:2308.10248.

Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Xia, F.; Chi, E. H.; Le, Q. V.; and Zhou, D. 2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. Advances in Neural Information Processing Systems (NeurIPS), 35: 24824–24837.

Xia, H.; Li, Y.; Leong, C. T.; Wang, W.; and Li, W. 2025. TokenSkip: Controllable Chain-of-Thought Compression in LLMs. arXiv:2502.12067.

Xu, S.; Xie, W.; Zhao, L.; and He, P. 2025. Chain of Draft: Thinking Faster by Writing Less. arXiv:2502.18600.

Yan, G.; Sun, C.-E.; and Weng, T.-W. 2025. ReflCtrl: Controlling LLM Reflection via Representation Engineering. arXiv:2512.13979.

Zhang, J.; Lin, N.; Hou, L.; Feng, L.; and Li, J. 2025a. AdaptThink: Reasoning Models Can Learn When to Think. arXiv:2505.13417.

Zhang, Z.; Wu, X.; Zhou, Z.; Wu, Q.; Zhang, Y.; Ponnusamy, P.; Subbaraj, H.; Wang, J.; Song, S. L.; and Athiwaratkun, B. 2025b. Understanding and Steering the Cognitive Behaviors of Reasoning Models at Test-Time. arXiv:2512.24574.

Zou, A.; Phan, L.; Chen, S.; Campbell, J.; Guo, P.; Ren, R.; Pan, A.; Yin, X.; Mazeika, M.; Dombrowski, A.-K.; Goel, S.; Li, N.; Byun, M.; Wang, Z.; Mallen, A.; Basart, S.; Koyejo, S.; Song, D.; Fredrikson, M.; Kolter, J. Z.; and Hendrycks, D. 2023. Representation Engineering: A Top-Down Approach to AI Transparency. arXiv:2310.01405.