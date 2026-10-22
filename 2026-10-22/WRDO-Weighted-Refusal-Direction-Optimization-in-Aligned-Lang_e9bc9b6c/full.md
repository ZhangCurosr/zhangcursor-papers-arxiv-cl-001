# WRDO: Weighted Refusal Direction Optimization in Aligned Language Models

Grigoriy Evseev<sup>1,2</sup>, Vladimir Solodkin<sup>1,2,3</sup>, Dmitry Bylinkin<sup>1,2</sup>, Daniil Medyakov<sup>1,2</sup>, Vladimir Aletov<sup>1,2</sup>, Aleksandr Beznosikov<sup>1,2,4</sup>

<sup>1</sup>Moscow Independent Research Institute of Artificial Intelligence

<sup>2</sup>Basic Research of Artificial Intelligence Laboratory (BRAIn Lab)

<sup>3</sup>T-Technologies

<sup>4</sup>Innopolis University

Aligned large language models are trained to comply with benign requests while refusing harmful ones. However, recent work shows that this refusal behavior is fragile and can often be disrupted by simple activation- or weight-space interventions. In particular, refusal has been shown to be largely mediated by a single residual-stream direction, typically computed as a diference-of-means vector between harmful and harmless prompts. However, this standard construction aggregates heterogeneous harmful prompts into one uniform average, potentially conflating distinct refusal-related phenomena. In this work, we study whether refusal interventions can be strengthened by learning how to combine prompt-specific refusal directions. We propose Weighted Refusal Direction Optimization (WRDO), a procedure that constructs a composite refusal direction from a bank of per-prompt directions using learned aggregation weights. This approach allows diferent harmful prompts to contribute unequally to the final intervention direction. We evaluate WRDO against prior single-direction methods on three open-weight language models using three guard judges and HarmBench as an external evaluator. We further measure the efect of the intervention on general model capability through standard utility benchmarks. Finally, we investigate a defense-oriented setting in which the same optimization machinery is used in reverse to restore refusal behavior in an abliterated model checkpoint. Our results show that weighted aggregation provides a more efective and flexible mechanism for manipulating refusal representations. Code is available at https://github.com/brain-lab-research/WRDO

## Introduction

Large language models deployed in user-facing settings are aligned to follow benign instructions and to refuse harmfula ones, with this behavior shaped by a combination of supervised fine-tuning, instruction tuning and preference optimization [1, 2]. The resulting refusal mechanism is, in practice, brittle under both prompt-level adversarial attacks [3, 4] and direct interventions on the model itself, which has motivated a mechanistic line of work that asks how refusal is represented internally and which components of the model implement it. A central finding of this line is that refusal is, to a first approximation, mediated by a single direction in the residual stream of the transformer [5]: ablating one diference-of-means vector, computed between activations on harmful and harmless prompts at a chosen layer, is suficient to disable refusal in a broad set of open-weight aligned models. The same work establishes an equivalence between activation ablation and weight-space orthogonalization of the attention output and feed-forward down projections, which makes the intervention a one-shot edit of the model with no inference-time overhead, and public toolkits such as Heretic [6] operationalize this recipe end-to-end.

The standard procedure averages every harmful prompt into a single axis. A disinformation prompt, a malware prompt and a privacy-violation prompt all enter the same mean, and the resulting direction is difuse in the sense that it conflates phenomena that need not be aligned in the residual stream. A natural alternative is to retain a per-prompt direction $r _ { i }$ for each training prompt and to combine them with a coeficient vector w. The simplest combination is the uniform average, $r = \mathrm { n o r m a l i z e } ( \sum _ { i } r _ { i } / N )$ , but there is no a priori reason for every prompt to contribute equally. Performance-aware weighted aggregation is known to beat uniform averaging whenever the aggregated components difer in quality, from ensembling ranked classifiers [7] to federated averaging [8] and adapter selection [9], and per-prompt refusal directions are heterogeneous in exactly that way. In this work, we research the question of whether computing refusal direction using the aggregation weights yields a stronger intervention than the single-direction recipe.

## Our Contributions:

• We propose Weighted Refusal Direction Optimization (WRDO), a procedure that obtains a refusal direction by learning aggregation weights w over a bank of per-prompt directions, together with two concrete instantiations of WRDO based on separable CMA-ES and a proposed GRPO variant with importance sampling (GRPO-IS).

• We show empirically that the proposed WRDO instantiations outperform all considered prior single-direction baselines across three open-weight models and three guard judges, and that this superiority transfers to HarmBench ruling out judge-specific overfitting.

• We obtain that WRD-GRPO-IS preserves the general capability of the intervened model substantially better than every other considered method on standard utility benchmarks.

• We additionally study a defense setup that reuses the same machinery in reverse to restore refusal on an abliterated checkpoint, illustrating an alternative use of the proposed approach.

The remainder of the paper is organized as follows. Section 2 formalizes the WRDO problem. Section 3 presents the two WRDO instantiations introduced in this work, namely an adaptation of separable CMA-ES and a proposed GRPO variant with importance sampling, together with the three single-direction baselines against which they are compared. Section 4 specifies the experimental setup, and Sections 5–6 report the empirical results, ablations and discussion.

## 2 Problem Formulation

An aligned language model is denoted as $f _ { \theta }$ , a harmful training set of prompts $\mathcal { D } _ { \mathrm { t r a i n } } ^ { \mathrm { h a r m } } = \{ x _ { i } \} _ { i = 1 } ^ { N }$ , a held-out harmful validation set $\mathcal { D } _ { \mathrm { v a l } } ^ { \mathrm { h a r m } }$ and a harmless baseline corpus $\mathcal { D } _ { \mathrm { b e n i g n } }$ . The hidden dimensionality of the residual stream is $d ,$ and L denotes the number of transformer blocks.

## 2.1 Candidate bank construction

Let $h ^ { ( l ) } ( x ) \in \mathbb { R } ^ { d }$ denote the residual-stream activation of $f _ { \theta }$ on prompt x at layer l, evaluated at the last token position. Activations are extracted at every transformer block, including the embedding output, so that each prompt is summarized by a tensor in $\mathbb { R } ^ { ( L + 1 ) \times d }$ . The mean benign activation $\mu _ { \mathrm { b e n i g n } } ^ { ( l ) } = \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } _ { \mathrm { b e n i g n } } } [ h ^ { ( l ) } ( \boldsymbol { x } ) ]$ is computed at every layer.

For each training prompt $x _ { i }$ we define a per-prompt layerwise direction by taking the diference $h ^ { ( l ) } ( x _ { i } ) - \mu _ { \mathrm { b e n i g n } } ^ { ( l ) }$ and applying $L _ { 2 }$ normalization at every layer independently,

$$
r _ { i } ^ { ( l ) } = \frac { h ^ { ( l ) } ( x _ { i } ) - \mu _ { \mathrm { b e n i g n } } ^ { ( l ) } } { \left\| h ^ { ( l ) } ( x _ { i } ) - \mu _ { \mathrm { b e n i g n } } ^ { ( l ) } \right\| _ { 2 } } ,\tag{1}
$$

and stack the per-layer vectors into a layerwise tensor $r _ { i } \in \mathbb { R } ^ { ( L + 1 ) \times d }$ . The candidate bank $\mathcal { R } = \{ r _ { i } \} _ { i = 1 } ^ { N }$ contains one such tensor per harmful training prompt, with $N = | \mathcal { D } _ { \mathrm { t r a i n } } ^ { \mathrm { h a r m } } |$ . The bank is computed once per model and is shared across all optimization methods.

## 2.2 Weighted direction

A coeficient vector $w \in \mathbb { R } ^ { N }$ aggregates the bank of Section 2.1 by a scalar weighting that is shared across layers and across hidden coordinates, followed by a per-layer $L _ { 2 }$ renormalization. Concretely, the composite direction at layer l is

$$
r ^ { ( l ) } ( w ) = \frac { \sum _ { i = 1 } ^ { N } w _ { i } r _ { i } ^ { ( l ) } } { \lVert \sum _ { i = 1 } ^ { N } w _ { i } r _ { i } ^ { ( l ) } \rVert _ { 2 } } .\tag{2}
$$

The per-layer normalization enforces $\| r ^ { ( l ) } ( w ) \| _ { 2 } = 1$ at every layer and removes the overall positive scale of $w ,$ so that only the orientation of the composite at each layer is varied. Intervention intensity is controlled separately through the layerwise schedule described in Section 2.3. The single-direction baseline of [5] is recovered, in this notation, by replacing the per-prompt $r _ { i }$ by a single diference-of-means tensor (computed between the mean harmful and mean benign activations at every layer) and setting $w _ { 1 } = 1$ with $N = 1$

## 2.3 Intervention operator

The primary intervention is weight-space orthogonalization. We adopt the intervention operator of the Heretic toolkit [6]. Given the layerwise composite direction $r ^ { ( l ) } ( w )$ of Equation 2, every attention output projection $W _ { O } ^ { ( l ) }$ and every feed-forward down-projection $W _ { \mathrm { d o w n } } ^ { ( l ) }$ at block l is replaced by

$$
\begin{array} { r } { W ^ { ( l ) \prime } = W ^ { ( l ) } - \alpha ^ { ( l ) } r ^ { ( l ) } ( w ) r ^ { ( l ) } ( w ) ^ { \top } W ^ { ( l ) } , } \end{array}\tag{3}
$$

where $\alpha ^ { ( l ) }$ is the fixed layerwise intensity schedule. The full form of $\alpha ^ { ( l ) }$ , together with the values of the auxiliary parameters that we hold fixed throughout the paper, is given in Appendix B. The only departure from the Heretic recipe is the choice of $r ^ { ( l ) } ( w )$ : Heretic plugs in the single direction of [5], whereas WRDO substitutes the per-prompt composite of Equation 2.

The composition of $f _ { \theta }$ with the operator of Equation 3, evaluated at the layerwise direction $r ( w ) = \{ r ^ { ( l ) } ( w ) \} _ { l }$ and a given $\alpha ,$ is denoted $T _ { r ( w ) } ( \theta )$ . We write $\pi _ { T _ { r ( w ) } ( \theta ) } ( \cdot \mid x )$ for the conditional distribution over generations produced by the intervened model under greedy decoding.

## 2.4 Harmfulness metric

For each prompt x and response y, every guard model used in this paper outputs a binary safe/unsafe decision $J ( x , y ) \in \{ 0 , 1 \}$ , with $J = 1$ denoting an unsafe response. The attack success rate (ASR) of the intervened model is the average of this per-prompt binary signal over the validation distribution,

$$
\mathrm { A S R } _ { J } ( w ) = \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } _ { \mathrm { v a l } } ^ { \mathrm { h a r m } } } \bigl [ J ( \boldsymbol { x } , \pi _ { T _ { r ( w ) } ( \theta ) } ( \cdot | \boldsymbol { x } ) ) \bigr ] .\tag{4}
$$

In practice, $\mathrm { A S R } _ { J }$ is estimated on a validation batch as the empirical mean of the per-prompt decisions. The optimization objective is the negative attack success rate,

$$
\begin{array} { r } { \begin{array} { r } { \mathcal { L } _ { J } ( w ) = - \mathrm { A S R } _ { J } ( w ) , } \end{array} } \end{array}\tag{5}
$$

which is shared by every WRDO method. Some optimizers may augment this objective with method-specific regularizers when needed (see WRD-GRPO-IS in Section 3.2 for a KL-regularized variant). The attack setup minimizes $\mathcal { L } _ { J }$

$$
w ^ { \star } = \arg \operatorname* { m i n } _ { w \in \mathcal { W } } \mathcal { L } _ { J } ( w ) ,\tag{6}
$$

where the feasible set $\mathcal { W } \subset \mathbb { R } ^ { N }$ is a per-coordinate box on $w ,$ with bounds and initialization detailed in Appendix A.

## 3 Methods

## 3.1 Baselines

Basic RD. We refer to the single-direction refusal-ablation procedure of [5] as Basic RD throughout the paper. A single layerwise direction is computed as the per-layer L<sub>2</sub>-normalized diference of means

$$
r _ { \mathrm { B a s i c } } ^ { ( l ) } = \frac { \mu _ { \mathrm { h a r m } } ^ { ( l ) } - \mu _ { \mathrm { b e n i g n } } ^ { ( l ) } } { \lVert \mu _ { \mathrm { h a r m } } ^ { ( l ) } - \mu _ { \mathrm { b e n i g n } } ^ { ( l ) } \rVert _ { 2 } }\tag{7}
$$

on the training split, where $\mu _ { \mathrm { h a r m } } ^ { ( l ) }$ is the mean activation of the harmful training prompts at layer l and $\mu _ { \mathrm { b e n i g n } } ^ { ( l ) }$ is the mean activation of the benign baseline corpus. The resulting layerwise direction is applied through the operator of Equation 3. There is no candidate bank and no optimization. This is the reference against which any benefit of per-prompt aggregation must be measured.

Heretic. We additionally include the public Heretic toolkit [6] as a further baseline. Heretic applies the same weight-space orthogonalization of Equation 3 to the single Basic RD direction, but tunes the layerwise intensity schedule with its own internal Bayesian search rather than using the fixed schedule of Appendix B. There is no per-prompt candidate bank and no learned w; the baseline isolates the contribution of schedule tuning alone, independent of the per-prompt aggregation introduced by WRDO. We run the released pipeline out of the box, with default hyperparameters.

RDO. We further include the gradient-based refusal-direction baseline of [10], denoted RDO. Instead of taking a diference of means as in Basic RD, RDO parameterizes a single refusal direction per layer and optimizes it directly with gradient descent against a refusal-suppression objective on the harmful training split, exploiting diferentiability through the model’s hidden activations. We use the released implementation of Wollschläger et al. [10] unchanged: the comparison against Basic RD isolates the contribution of learning the direction itself. As with Basic RD and Heretic, there is no per-prompt candidate bank and no learned aggregation w.

RDO-CONE. Wollschläger et al. [10] also argue that refusal is mediated by a cone of directions rather than by a single line, and release a concept-cone construction alongside the single-direction RDO objective. This is the prior construction closest to the motivation of WRDO, since it likewise rejects the assumption that one diference-of-means axis captures refusal, so we include it as a separate baseline denoted RDO-CONE. It difers from WRDO in where the multiplicity is placed: RDO-CONE learns a set of mutually constrained directions that span a cone and ablates the whole cone, whereas WRDO keeps a fixed bank of per-prompt directions and learns only how to aggregate them into the composite of Equation 2. We again use the released implementation of Wollschläger et al. [10] unchanged, with the authors’ default cone dimensionality.

## 3.2 WRD optimizers

For optimizing the refusal-direction weights in the WRDO pipeline, we propose two distinct procedures. The first is a WRDO-tailored adaptation of separable CMA-ES, a well-known derivative-free optimizer suited to our low-dimensional setting. The second is a novel scheme based on group-relative policy optimization with importance sampling, designed specifically for the WRDO objective.

WRD-CMAES. CMA-ES [11, 12] is a standard derivative-free optimizer for continuous black-box objectives, well suited to the noisy and gradient-free nature of the WRDO loss. In our setting we apply it to $\mathcal { L } _ { J }$ of Equation 5: the search variables are the aggregation coeficients w of the candidate bank, the search space is described in Appendix A, and each black-box query corresponds to one full WRDO evaluation, i.e. building the composite direction r(w), intervening the model through the operator of Section 2.3, and measuring the resulting ASR on the validation batch. The concrete sampler choice, hyperparameter values and search-space bounds are listed in Appendix F.

Relation to evolution strategies. Both instantiations belong to the evolution-strategy family [11, 13, 14]: a distribution over w is maintained, perturbed samples are scored by the black-box objective, and the distribution is moved towards the better-scoring ones. WRD-CMAES is its canonical member, and WRD-GRPO-IS reads as an ES with importance mixing, with the Gaussian rollout of Equation 11 as the search distribution and the weights $\widetilde { v } _ { m }$ as the mixing correction, so running both compares ES against $\mathrm { R L }$ under one objective and one budget. The analogy stops at the reward: NES-style updates see only the scalar fitness of each perturbation, whereas WRD-GRPO-IS routes it through token-level policy ratios and can therefore carry the KL term of Equation 8.

WRD-GRPO-IS. The target loss function described in 2.4 is a non-diferentiable function, so it is natural to consider a reinforcement learning approach to its optimization. To keep the intervened model close to the clean base model on benign prompts, WRD-GRPO-IS augments the bare WRDO objective of Equation 5 with a KL-divergence penalty against the policy $\pi _ { f _ { \theta } }$ evaluated on a random batch from $\mathcal { D } _ { \mathrm { b e n i g n } } .$ , yielding the method-specific objective

$$
\begin{array} { r } { \widetilde { \mathcal { L } } _ { J } ( w ) = \mathcal { L } _ { J } ( w ) + \beta \mathrm { K L } _ { w } , } \end{array}\tag{8}
$$

with $\beta \geq 0$ and $\mathrm { K L } _ { w } = \mathbb { E } _ { { x } \sim \mathcal { D } _ { \mathrm { b e n i g n } } } \bigl [ \mathrm { K L } \left( \pi _ { T _ { r ( w ) } ( \theta ) } ( \cdot | x ) \| \pi _ { f _ { \theta } } ( \cdot | x ) \right) \bigr ]$ the KL divergence of the intervened policy from the clean policy on the benign batch. The KL term is specific to WRD-GRPO-IS: the WRD-CMAES sampler consumes the ASR-only signal of Equation 5 without any regularization.

Group Relative Policy Optimization. Let us denote the language model parameterized by θ as a policy $\pi _ { \theta }$ Given a query x drawn from a query set ${ \mathcal { D } } ,$ it induces a distribution over responses y, whose likelihood factorizes as

$$
\pi _ { \boldsymbol { \theta } } ( y \mid x ) = \prod _ { t = 1 } ^ { | y | } \pi _ { \boldsymbol { \theta } } ( y _ { t } \mid x , y _ { < t } ) ,
$$

where $| y |$ denotes the number of tokens in y. A query-response pair $( x , y )$ is scored by a reward model $s ,$ yielding a per-prompt reward $S ( x , y ) \in \{ 0 , 1 \}$ ; in our setting $\boldsymbol { S } ( x , y ) = \boldsymbol { J } ( x , y )$ is the binary judge decision of Equation 4. The standard GRPO algorithm computes a relative advantage for each response within a group of $G$ responses generated for the same query from the old policy $\pi _ { \theta _ { \mathrm { o l d } } }$ . Specifically, GRPO optimizes

$$
\mathcal { I } _ { \mathrm { G R P O } } ( \theta ) = \mathbb { E } _ { \boldsymbol { x } , \{ y _ { i } \} \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot | x ) } \left[ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { | y _ { i } | } \sum _ { t = 1 } ^ { | y _ { i } | } \operatorname* { m i n } \Bigl ( \rho _ { i , t } ( \theta ) \widehat { A } _ { i } , \mathrm { c l i p } ( \rho _ { i , t } ( \theta ) , 1 - \varepsilon , 1 + \varepsilon ) \widehat { A } _ { i } \Bigr ) \right] ,\tag{9}
$$

where the token-level importance ratio is

$$
\rho _ { i , t } ( \theta ) = \frac { \pi _ { \theta } ( y _ { i , t } \mid x , y _ { i , < t } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { i , t } \mid x , y _ { i , < t } ) } ,
$$

We write importance ratios as $\rho$ throughout this section to keep them distinct from the aggregation coeficients $w \in \mathbb { R } ^ { N }$ of Section 2.2, which remain the only trainable quantity of WRDO. and all tokens in a response share the same group-relative advantage

$$
\widehat { A } _ { i } = \frac { S ( x , y _ { i } ) - \operatorname* { m e a n } ( \{ S ( x , y _ { j } ) \} _ { j = 1 } ^ { G } ) } { \operatorname* { s t d } ( \{ S ( x , y _ { j } ) \} _ { j = 1 } ^ { G } ) } .
$$

Group Relative Policy Optimization with Importance Sampling. In the standard GRPO formulation (9), the group of responses for each query is sampled from the old policy $\pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } }$ , and the clipped objective uses the token-level importance ratio between the current and old policies. Here, we consider a stratified data-collection scheme. For each query x, we sample one response from each of M behavior policies $\{ \mu _ { m } \} _ { m = 1 } ^ { M }$ that are obtained by shifting the original model weights with formula (3), i.e., $y _ { m } \sim \mu _ { m } ( \cdot \mid x )$ . As the resulting group is of-policy with respect to $\pi _ { \theta _ { \mathrm { o l d } } }$ , directly applying (9) would generally yield a biased estimate of the GRPO objective. We introduce sequence-level importance sampling weights to correct this mismatch while retaining the original ratio against $\pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } }$ Specifically, for each response $y _ { m }$ sampled from $\mu _ { m }$ , we define the sequence-level importance weight

$$
v _ { m } = { \frac { \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { m } \mid x ) } { \mu _ { m } ( y _ { m } \mid x ) } } .
$$

In practice, we stabilize the weights $\{ v _ { m } \} _ { m = 1 } ^ { M }$ by applying clipping, denoted as $\widetilde { v } _ { m }$ , and treat them as constants with respect to θ during optimization. This reweighting ensures that the overall objective remains consistent with the GRPO surrogate.

Analogous to GRPO, we assign a shared advantage to all tokens in a response. However, to preserve the controlvariate property under of-policy sampling, we compute the group baseline using the same sequence-level weights. Given the group $\{ y _ { m } \} _ { m = 1 } ^ { M }$ for query x, we first form the importance-weighted baseline

$$
b ( x ) = \frac { \sum _ { m = 1 } ^ { M } \widetilde { v } _ { m } S ( x , y _ { m } ) } { \sum _ { m = 1 } ^ { M } \widetilde { v } _ { m } } ,
$$

and define the normalized group-relative advantage as

$$
\widehat { A } _ { m } = \frac { S ( x , y _ { m } ) - b ( x ) } { \mathrm { s t d } \Big ( \{ S ( x , y _ { j } ) \} _ { j = 1 } ^ { M } \Big ) } ,
$$

where all tokens in $y _ { m }$ share the same advantage, i.e., $\widehat { A } _ { m , t } = \widehat { A } _ { m }$ for all t.

Finally, our GRPO objective with importance sampling is

$$
\mathcal { I } _ { \mathrm { G R P O - I S } } ( \theta ) = \mathbb { E } _ { x , y _ { m } \sim \mu _ { m } ( \cdot \vert x ) } \left[ \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \widetilde v _ { m } \frac { 1 } { \vert y _ { m } \vert } \sum _ { t = 1 } ^ { \vert y _ { m } \vert } \operatorname* { m i n } \Bigl ( \rho _ { m , t } ( \theta ) \widehat { A } _ { m } , \mathrm { c l i p } ( \rho _ { m , t } ( \theta ) , 1 - \varepsilon , 1 + \varepsilon ) \widehat { A } _ { m } \Bigr ) \right] ,\tag{10}
$$

where the token-level importance ratio is defined as

$$
\rho _ { m , t } ( \theta ) = \frac { \pi _ { \theta } ( y _ { m , t } \mid x , y _ { m , < t } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { m , t } \mid x , y _ { m , < t } ) } .
$$

Compared with standard GRPO, our objective (10) difers in that the group is collected by stratified sampling from multiple behavior policies $\{ \mu _ { m } \}$ , and a sequence-level importance weight $\widetilde { v } _ { m }$ is used to correct the of-policy sampling back to $\pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } }$

In our setting, the trainable parameter is the coeficient vector $w \in \mathbb { R } ^ { N }$ of Section 2.2 and the base policy $\pi _ { \theta _ { \mathrm { o l d } } }$ is the clean model $f _ { \theta }$ . At each training step, the M behavior policies $\{ \mu _ { m } \} _ { m = 1 } ^ { M }$ are constructed by perturbing the current w with Gaussian noise,

$$
w _ { m } = w + \varepsilon _ { m } , \qquad \varepsilon _ { m } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I _ { N } ) ,\tag{11}
$$

with $\varepsilon _ { 0 } = 0$ . Each $w _ { m }$ defines a layerwise composite direction $r ^ { ( l ) } ( w _ { m } )$ via Equation 2 and an intervened model $\mu _ { m } = T _ { r ( w _ { m } ) } ( \theta _ { \mathrm { o l d } } )$ via the operator of Section 2.3. Only w is updated by the surrogate of Equation 10, while the candidate bank R, the base parameters $\theta _ { \mathrm { o l d } }$ and the schedule $\alpha ^ { ( l ) }$ are held fixed throughout training.

## 4 Experimental Setup

## 4.1 Data

The harmful pool is the union of six public benchmarks, namely AdvBench [3], MaliciousInstruct [15], the TDC2023 [16], HarmBench [17], JailbreakBench [18] and StrongREJECT [19], deduplicated by exact prompt-string match and treated as a single distribution. The deduplicated pool is pre-partitioned into a training set, used only to compute the candidate bank, a validation set, used as the optimization objective in Equation 4, and a test set, used only for the final evaluation. Concrete sampling seeds and the source of the prepared splits are documented in Appendix C. The harmless baseline $\mathcal { D } _ { \mathrm { b e n i g n } }$ is a fixed-size sub-sample of Alpaca [20] used at bank-construction time.

<table><tr><td></td><td></td><td colspan="3">α = 1.0</td><td colspan="3">α = 3.0</td></tr><tr><td>Model</td><td>Method</td><td>WG</td><td>Q3G</td><td>LG3</td><td>WG</td><td>Q3G</td><td>LG3</td></tr><tr><td rowspan="7">Qwen3-8B-Base</td><td>No intervention</td><td>0.02</td><td>0.02</td><td>0.02</td><td>0.02</td><td>0.02</td><td>0.02</td></tr><tr><td>Basic RD</td><td> $0 . 2 2 \pm 0 . 0 2$ </td><td> $0 . 1 1 \pm 0 . 0 3$ </td><td> $0 . 4 4 \pm 0 . 0 2$ </td><td> $0 . 8 1 \pm 0 . 0 3$ </td><td> $0 . 8 0 \pm 0 . 0 3$ </td><td> $0 . 9 1 \pm 0 . 0 4$ </td></tr><tr><td>Heretic</td><td> $0 . 7 7 \pm 0 . 0 4$ </td><td> $0 . 8 3 \pm 0 . 0 5$ </td><td> $0 . 8 1 \pm 0 . 0 5$ </td><td> $0 . 7 7 \pm 0 . 0 4$ </td><td> $0 . 8 3 \pm 0 . 0 5$ </td><td> $0 . 8 1 \pm 0 . 0 5$ </td></tr><tr><td>RDO</td><td> $0 . 3 4 \pm 0 . 0 5$ </td><td> $0 . 4 1 \pm 0 . 0 4$ </td><td> $0 . 4 8 \pm 0 . 0 5$ </td><td> $0 . 3 4 \pm 0 . 0 5$ </td><td> $0 . 4 1 \pm 0 . 0 4$ </td><td> $0 . 4 8 \pm 0 . 0 5$ </td></tr><tr><td>Uniform-avg WRD</td><td> $0 . 0 8 \pm 0 . 0 0 1$ </td><td> $0 . 2 5 \pm 0 . 0 1$ </td><td> $0 . 3 2 \pm 0 . 0 2$ </td><td> $0 . 0 0 2 \pm 0 . 0 0 0 3$ </td><td> $0 . 8 9 \pm 0 . 0 3$ </td><td> $0 . 9 2 \pm 0 . 0 2$ </td></tr><tr><td>WRD-CMAES</td><td> $0 . 8 8 \pm 0 . 0 5$ </td><td> $0 . 8 3 \pm 0 . 0 2$ </td><td> $\mathbf { 0 . 8 9 \ \pm 0 . 0 2 }$ </td><td> $0 . 9 1 \pm 0 . 0 5$ </td><td> ${ \bf 0 . 9 9 \pm 0 . 0 0 2 }$ </td><td> $\mathbf { 0 . 9 8 \ \pm 0 . 0 0 2 }$ </td></tr><tr><td>WRD-GRPO-IS</td><td> $\mathbf { 0 . 9 1 \Pi { \pm 0 . 0 7 } }$ </td><td> ${ \bf 0 . 9 4 \ \pm 0 . 0 2 }$ </td><td> $0 . 8 7 \pm 0 . 0 2$ </td><td> ${ \bf 0 . 9 3 \ \pm 0 . 0 9 }$ </td><td> $0 . 9 5 \pm 0 . 0 2$ </td><td> $0 . 9 6 \pm 0 . 0 1$ </td></tr><tr><td rowspan="7"> $\mathrm { O L M o { - } 3 { - } 1 0 2 5 { - } 7 B }$ </td><td>No intervention</td><td>0.08</td><td>0.09</td><td>0.07</td><td>0.08</td><td>0.09</td><td>0.07</td></tr><tr><td>Basic RD</td><td>0.41 ±0.02</td><td> $0 . 5 1 \pm 0 . 0 3$ </td><td> $0 . 4 3 \pm 0 . 0 3$ </td><td> $0 . 7 6 \pm 0 . 0 2$ </td><td> $0 . 7 2 \pm 0 . 0 3$ </td><td> $0 . 7 3 \pm 0 . 0 2$ </td></tr><tr><td>Heretic</td><td> $0 . 7 6 \pm 0 . 0 3$ </td><td> $0 . 7 3 \pm 0 . 0 4$ </td><td> $0 . 7 0 \pm 0 . 0 3$ </td><td> $0 . 7 6 \pm 0 . 0 3$ </td><td> $0 . 7 3 \pm 0 . 0 4$ </td><td> $0 . 7 0 \pm 0 . 0 3$ </td></tr><tr><td>RDO</td><td> $0 . 6 1 \pm 0 . 0 4$ </td><td> $0 . 6 7 \pm 0 . 0 3$ </td><td> $0 . 7 3 \pm 0 . 0 3$ </td><td> $0 . 6 1 \pm 0 . 0 4$ </td><td> $0 . 6 7 \pm 0 . 0 3$ </td><td> $0 . 7 3 \pm 0 . 0 3$ </td></tr><tr><td>Uniform-avg WRD</td><td> $0 . 6 4 \pm 0 . 0 2$ </td><td> $0 . 6 6 \pm 0 . 0 1$ </td><td> $0 . 6 8 \pm 0 . 0 2$ </td><td> $0 . 7 7 \pm 0 . 0 3$ </td><td> $0 . 7 6 \pm 0 . 0 2$ </td><td> $0 . 7 9 \pm 0 . 0 1$ </td></tr><tr><td>WRD-CMAES</td><td> ${ \bf 0 . 8 7 \tau \pm 0 . 0 2 }$ </td><td> $\mathbf { 0 . 8 6 \ \pm 0 . 0 3 }$ </td><td> $\mathbf { 0 . 8 2 \ \pm 0 . 0 2 }$ </td><td> $0 . 8 8 \pm 0 . 0 1$ </td><td> $0 . 8 8 \pm 0 . 0 2$ </td><td> $0 . 8 5 \pm 0 . 0 3$ </td></tr><tr><td>WRD-GRPO-IS</td><td> $0 . 8 2 \pm 0 . 0 2$ </td><td> $\mathbf { 0 . 8 6 \ \pm 0 . 0 3 }$ </td><td> $0 . 7 9 \pm 0 . 0 2$ </td><td> ${ \bf 0 . 9 2 \ \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 9 7 \pm 0 . 0 3 }$ </td><td> ${ \bf 0 . 9 7 \pm 0 . 0 2 }$ </td></tr><tr><td rowspan="7">Olmo-3-7B-Instruct</td><td>No intervention</td><td>0.003</td><td>0.003</td><td>0.004</td><td>0.003</td><td>0.003</td><td>0.004</td></tr><tr><td>Basic RD</td><td> $0 . 4 0 \pm 0 . 0 2$ </td><td> $0 . 4 2 \pm 0 . 0 1$ </td><td> $0 . 4 3 \pm 0 . 0 3$ </td><td> $0 . 5 3 \pm 0 . 0 2$ </td><td> $0 . 6 1 \pm 0 . 0 3$ </td><td> $0 . 5 9 \pm 0 . 0 1$ </td></tr><tr><td>Heretic</td><td> $0 . 6 2 \pm 0 . 0 3$ </td><td> $0 . 5 8 \pm 0 . 0 4$ </td><td> $0 . 6 9 \pm 0 . 0 4$ </td><td> $0 . 6 2 \pm 0 . 0 3$ </td><td> $0 . 5 8 \pm 0 . 0 4$ </td><td> $0 . 6 9 \pm 0 . 0 4$ </td></tr><tr><td>RDO</td><td> $0 . 4 0 \pm 0 . 0 5$ </td><td> $0 . 5 2 \pm 0 . 0 4$ </td><td> $0 . 5 3 \pm 0 . 0 4$ </td><td> $0 . 4 0 \pm 0 . 0 5$ </td><td> $0 . 5 2 \pm 0 . 0 4$ </td><td> $0 . 5 3 \pm 0 . 0 4$ </td></tr><tr><td>Uniform-avg WRD</td><td> $0 . 4 9 \pm 0 . 0 2$ </td><td> $0 . 4 2 \pm 0 . 0 1$ </td><td> $0 . 4 5 \pm 0 . 0 3$ </td><td> $0 . 6 2 \pm 0 . 0 2$ </td><td> $0 . 6 7 \pm 0 . 0 1$ </td><td> $0 . 6 7 \pm 0 . 0 1$ </td></tr><tr><td>WRD-CMAES</td><td> ${ \bf 0 . 9 2 \ \pm 0 . 0 2 }$ </td><td> $\mathbf { 0 . 8 9 \ \pm 0 . 0 1 }$ </td><td> ${ \bf 0 . 9 3 \ \pm 0 . 0 1 }$ </td><td> $0 . 8 9 \pm 0 . 0 2$ </td><td> ${ \bf 0 . 9 3 \ \pm 0 . 0 2 }$ </td><td> $0 . 8 6 \pm 0 . 0 1$ </td></tr><tr><td> $\mathrm { W R D - G R P O - I S }$ </td><td> ${ \bf 0 . 9 2 \ \pm 0 . 0 1 }$ </td><td> $0 . 8 8 \pm 0 . 0 2$ </td><td> ${ \bf 0 . 9 3 \ \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 9 3 \ \pm 0 . 0 1 }$ </td><td> $0 . 9 0 \pm 0 . 0 2$ </td><td> $\mathbf { 0 . 8 8 \ \pm 0 . 0 1 }$ </td></tr></table>

Table 1: Attack setup. Test ASR (in [0, 1]) on held-out harmful prompts, reported at the two extremes of the intensity grid, $\alpha \in \{ 1 . 0 , 3 . 0 \}$ . Judges: WG = WildGuard, Q3G = Qwen3Guard-Gen-8B, LG3 = Llama-Guard-3-8B. Rows for α-independent methods are repeated across columns for layout consistency. Bold marks the best result in each column within each model block. Higher is a stronger attack. The intermediate intensities $\alpha \in \{ 1 . 5 , 2 . 0 \}$ are reported in Table 5 of Appendix D. Every intervened cell is the mean over three independent repetitions of the full pipeline, with the standard deviation over those repetitions. Cells that would round to 0.00 are reported at three decimals; none is an exact zero.

## 4.2 Models

We use three open-weight models: Qwen/Qwen3-8B-Base [21], allenai/OLMo-3-1025-7B [22] and Olmo-3- 7B-Instruct [22]. All evaluations use greedy decoding (do\_sample=False) with a maximum of 100 new tokens per completion, one completion per prompt, the default system prompt of each model and no few-shot context.

## 4.3 Judges

We use three guard models, namely allenai/wildguard [23], Qwen/Qwen3Guard-Gen-8B [24] and meta-llama/ Llama-Guard-3-8B [25]. Each guard exposes its own decoding pipeline that maps a (prompt, response) pair to a binary safe/unsafe label, which we use directly as $J ( x , y ) \in \{ 0 , 1 \}$ in Equation 4. The per-judge $\mathrm { A S R } _ { J } \in [ 0 , 1 ]$ is then obtained by averaging these per-prompt binary decisions over the test set and is reported separately for every judge. HarmBench [17] with the oficial classifier cais/HarmBench-Llama-2-13b-cls provides an additiona external evaluator that is never used during optimization.

## 4.4 Metrics

The primary metric is the ASR of Equation 4, reported separately for each of the three judges J on the held-ou test set. The secondary metrics are the utility benchmarks of Section 4.5, the HarmBench ASR of Section 5.4.

## 4.5 Utility benchmarks

Utility preservation is evaluated through the Language Model Evaluation Harness [26] on arc\_easy and arc\_challenge [27], mmlu [28], gsm8k and word-level perplexity on wikitext. Accuracy and perplexity are reported separately, with the direction of improvement made explicit in the tables.

## 5 Results

## 5.1 Attack setup

Table 1 reports the test-set ASR of every method on the three base models, evaluated separately against the three guard judges, at the two extremes of the peak intensity grid $\alpha \in \{ 1 . 0 , 3 . 0 \}$ of the layerwise schedule introduced in Section 2.3 (Equation $^ { 3 ; }$ full form in Appendix B). The intermediate intensities $\alpha \in \{ 1 . 5 , 2 . 0 \}$ are deferred to Table 5 in Appendix D, where the same conclusions hold. The no-intervention rows provide the absence-of-attack baselines on the same test set. The Heretic entries are duplicated across the α columns because Heretic does not consume this hyperparameter from the outside: it selects its own shift coeficient internally through the toolkit’s Bayesian search, so a single value is reported per model and judge and repeated across the α grid for layout consistency. The RDO rows are duplicated for the same reason: RDO does not expose α as a hyperparameter, the intensity is baked into the gradient-learned direction itself, and the resulting per-model, per-judge value is repeated across the α grid. To check whether the apparent benefit of WRDO is driven by the per-prompt candidate bank alone or by the learned aggregation on top of it, we introduce the Uniform-avg WRD sanity-check baseline in this experiment. Uniform-avg WRD uses the same candidate bank of Section 2.1 as the learned WRDO instantiations, but fixes the coeficients to the uniform vecto $w _ { i } = 1 / N$ without any optimization.

Discussion. First, in nearly every configuration the two learned WRDO instantiations (WRD-CMAES and WRD-GRPO-IS) outperform every prior single-direction baseline by a clear margin, establishing the superiority of the weighted-aggregation approach over the diference-of-means recipe, the Heretic toolkit and the gradient-learned RDO baseline. Second, the Uniform-avg WRD baseline is consistently dominated by the learned WRDO instantiations across the α grid, confirming that learning the aggregation coeficients w is what drives the gain, not the per-promp bank alone.

## 5.2 Concept cones and scale

Two comparisons fall outside the grid of Table 1 and are reported in Appendix I. Against the concept cone of Wollschläger et al. [10], which treats refusal as a subspace rather than a line, the two are close: RDO-CONE leads by 3.3 ASR points on Qwen3-8B-Base, WRD-GRPO-IS by 5.5 and 8.5 on the two OLMo checkpoints. Repeating the attack setup on the 21B openai/gpt-oss-20b [29] reproduces the ranking with wider margins.

## 5.3 Utility preservation

Table 2 reports utility benchmarks for the same $w ^ { \star }$ that produced Table 1, per base model, at the two extremes of the intensity grid $\alpha \in \{ 1 . 0 , 3 . 0 \}$ . Intermediate values $\alpha \in \{ 1 . 5 , 2 . 0 \}$ are deferred to Table 6 in Appendix E. In this experiment we restrict attention to the WildGuard-optimized runs only: $w ^ { \star }$ in every cell below is the coeficient vector obtained by optimizing against WildGuard. Accuracy is reported for ARC-easy (ARC-e), ARC-challenge (ARC-c), MMLU and GSM8K; word-level perplexity is reported for WikiText (PPL). As in Section 5.1, the Heretic entries are duplicated across the α columns because Heretic selects its own shift coeficient internally rather than consuming α from the outside, so a single value per model and metric is repeated across the α grid for layou consistency.

Discussion. First, the prior single-direction baselines degrade general capabilities substantially at the larger values of α: accuracy on ARC and GSM8K collapses and WikiText perplexity grows by orders of magnitude. Heretic is partially protected, since its Bayesian search minimizes a KL divergence to the unintervened model, and it does hold perplexity near the clean value throughout. It is not uniformly utility-preserving, however: on Qwen3-8B-Base it costs 40 MMLU and 65 GSM8K points, so its KL target constrains the benign output distribution without protecting downstream benchmarks. Second, among the learned WRDO instantiations WRD-CMAES exhibits the same degradation pattern as the prior baselines: the Optuna target consumed by the sampler is the bare ASR-only signal without a KL term, so the optimizer is free to find directions that succeed at suppressing refusal but disrupt unrelated capabilities. Third, WRD-GRPO-IS preserves utility best because its surrogate of Equation 10 optimizes the KL-regularized objective of Equation 8, keeping the edit close to the clean policy even at high α.

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="5">α = 1.0</td><td colspan="5">α = 3.0</td></tr><tr><td>ARC-e</td><td>ARC-c</td><td>MMLU</td><td>GSM8K</td><td>PPL</td><td>ARC-e</td><td>ARC-c</td><td>MMLU</td><td>GSM8K</td><td>PPL</td></tr><tr><td rowspan="6">Qwen3-8B-Base</td><td>No intervention</td><td> $0 . 6 3 \pm 0 . 0 1$ </td><td> $0 . 3 8 \pm 0 . 0 1$ </td><td> $0 . 6 7 \pm 0 . 0 0 4$ </td><td> $0 . 8 4 \pm 0 . 0 1$ </td><td>11.11</td><td> $0 . 6 3 \pm 0 . 0 1$ </td><td> $0 . 3 8 \pm 0 . 0 1$ </td><td> $0 . 6 7 \pm 0 . 0 0 4$ </td><td> $0 . 8 4 \pm 0 . 0 1$ </td><td>11.11</td></tr><tr><td>Basic RD</td><td> $0 . 5 9 \pm 0 . 0 1$ </td><td> $0 . 3 6 \pm 0 . 0 1$ </td><td> $0 . 6 4 \pm 0 . 0 0 4$ </td><td> $0 . 6 7 \pm 0 . 0 1$ </td><td>13.81</td><td> $0 . 2 6 \pm 0 . 0 1$ </td><td> $0 . 2 3 \pm 0 . 0 1$ </td><td> $0 . 2 3 \pm 0 . 0 0 4$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td><td> $> 1 0 ^ { 1 2 }$ </td></tr><tr><td>Heretic</td><td> $0 . 5 1 \pm 0 . 0 1$ </td><td> $0 . 3 0 \pm 0 . 0 1$ </td><td> $0 . 2 7 \pm 0 . 0 0 4$ </td><td> $0 . 1 9 \pm 0 . 0 1$ </td><td>13.44</td><td> $0 . 5 1 \pm 0 . 0 1$ </td><td> $0 . 3 0 \pm 0 . 0 1$ </td><td> $0 . 2 7 \pm 0 . 0 0 4$ </td><td> $0 . 1 9 \pm 0 . 0 1$ </td><td>13.44</td></tr><tr><td>RDO</td><td> $0 . 6 3 \pm 0 . 0 1$ </td><td> $0 . 3 7 \pm 0 . 0 1$ </td><td> $0 . 6 7 \pm 0 . 0 0 4$ </td><td>0.82 ± 0.01</td><td>11.16</td><td> $0 . 6 3 \pm 0 . 0 1$ </td><td> $0 . 3 7 \pm 0 . 0 1$ </td><td> $0 . 6 7 \pm 0 . 0 0 4$ </td><td> $0 . 8 2 \pm 0 . 0 1$ </td><td>11.16</td></tr><tr><td>WRD-CMAES</td><td> $0 . 5 3 \pm 0 . 0 1$ </td><td> $0 . 3 4 \pm 0 . 0 1$ </td><td> $0 . 6 7 \pm 0 . 0 0 3 \ 0 . 0 1 \pm 0 . 0 0 3$ </td><td></td><td> $1 6 1 . 1 8 \ 0 . 3 7 \pm 0 . 0 1$ </td><td></td><td> $0 . 2 9 \pm 0 . 0 1$ </td><td> $0 . 5 1 \pm 0 . 0 0 4$ </td><td> $0 . 0 0 4 \pm 0 . 0 0 2$ </td><td>&gt; 104</td></tr><tr><td>WRD-GRPO-IS</td><td> $0 . 6 3 \pm 0 . 0 1$ </td><td> $0 . 3 8 \pm 0 . 0 1$ </td><td>0.65 ± 0.004 0.83 ± 0.01</td><td></td><td>11.15</td><td> $0 . 6 4 \pm 0 . 0 1$ </td><td> $0 . 3 8 \pm 0 . 0 1$ </td><td> $0 . 6 2 \pm 0 . 0 0 4$ </td><td> $0 . 8 3 \pm 0 . 0 1$ </td><td>11.35</td></tr><tr><td rowspan="6">OLMo-3-1025-7B</td><td>No intervention</td><td> $0 . 7 9 \pm 0 . 0 1$ </td><td> $0 . 4 8 \pm 0 . 0 1 5$ </td><td> $0 . 6 2 \pm 0 . 0 0 4 \ 0 . 5 3 \pm 0 . 0 1 4$ </td><td></td><td>62.5</td><td> $0 . 7 9 \pm 0 . 0 1$ </td><td> $0 . 4 8 \pm 0 . 0 1 5$ </td><td> $0 . 6 2 \pm 0 . 0 0 4$ </td><td> $0 . 5 3 \pm 0 . 0 1 4$ </td><td>62.5</td></tr><tr><td>Basic RD</td><td> $0 . 8 0 \pm 0 . 0 1$ </td><td> $0 . 4 8 \pm 0 . 0 1$ </td><td>0.61 ± 0.004</td><td> $0 . 5 3 \pm 0 . 0 1$ </td><td>62.71</td><td> $0 . 4 2 \pm 0 . 0 1$ </td><td> $0 . 2 6 \pm 0 . 0 1$ </td><td> $0 . 3 2 \pm 0 . 0 0 4$ </td><td> $0 . 0 0 0 8 \pm 0 . 0 0 0 8$ </td><td>432.02</td></tr><tr><td>Heretic</td><td> $0 . 7 9 \pm 0 . 0 1$ </td><td> $0 . 4 8 \pm 0 . 0 1$ </td><td>0.62 ± 0.004</td><td> $0 . 5 2 \pm 0 . 0 1$ </td><td>62.61</td><td> $0 . 7 9 \pm 0 . 0 1$ </td><td> $0 . 4 8 \pm 0 . 0 1$ </td><td> $0 . 6 2 \pm 0 . 0 0 4$ </td><td> $0 . 5 2 \pm 0 . 0 1$ </td><td>62.61</td></tr><tr><td>RDO</td><td> $0 . 8 0 \pm 0 . 0 1$ </td><td> $0 . 4 8 \pm 0 . 0 1$ </td><td>0.62 ± 0.004</td><td> $0 . 5 6 \pm 0 . 0 1$ </td><td>63.32</td><td> $0 . 8 0 \pm 0 . 0 1$ </td><td> $0 . 4 8 \pm 0 . 0 1$ </td><td> $0 . 6 2 \pm 0 . 0 0 4$ </td><td> $0 . 5 6 \pm 0 . 0 1$ </td><td>63.32</td></tr><tr><td>WRD-CMAES</td><td> $0 . 7 9 \pm 0 . 0 1$ </td><td> $0 . 4 8 \pm 0 . 0 1$ </td><td>0.62 ± 0.004</td><td> $0 . 5 2 \pm 0 . 0 1$ </td><td>62.40</td><td> $0 . 7 9 \pm 0 . 0 1$ </td><td> $0 . 4 8 \pm 0 . 0 1$ </td><td> $0 . 6 1 7 \pm 0 . 0 0 4$ </td><td> $0 . 5 1 \pm 0 . 0 1$ </td><td>61.70</td></tr><tr><td>WRD-GRPO-IS</td><td> $0 . 7 9 \pm 0 . 0 1$ </td><td> $0 . 4 8 \pm 0 . 0 1 5$ </td><td>0.62 ± 0.004 0.53 ± 0.014</td><td></td><td>62.5</td><td> $0 . 7 8 \pm 0 . 0 1$ </td><td> $0 . 4 9 \pm 0 . 0 1 5$ </td><td> $0 . 6 1 \pm 0 . 0 0 4$ </td><td> $0 . 5 5 \pm 0 . 0 1 4$ </td><td>62.21</td></tr><tr><td rowspan="6">Olmo-3-7B-Instruct</td><td>No intervention</td><td> $0 . 7 6 \pm 0 . 0 1$ </td><td> $0 . 4 8 \pm 0 . 0 1$ </td><td>0.53 ± 0.004</td><td> $0 . 6 7 \pm 0 . 0 1$ </td><td>100.83 0.76 ± 0.01</td><td> $0 . 4 8 \pm 0 . 0 1$ </td><td></td><td> $0 . 5 3 \pm 0 . 0 0 4$ </td><td> $0 . 6 7 \pm 0 . 0 1$ </td><td>100.83</td></tr><tr><td>Basic RD</td><td> $0 . 7 7 \pm 0 . 0 1$ </td><td> $0 . 4 8 \pm 0 . 0 1$ </td><td> $0 . 5 4 \pm 0 . 0 0 4$ </td><td> $0 . 6 7 \pm 0 . 0 1$ </td><td>100.81</td><td> $0 . 4 2 \pm 0 . 0 1$ </td><td> $0 . 3 1 \pm 0 . 0 1$ </td><td> $0 . 1 7 \pm 0 . 0 0 4$ </td><td> $0 . 0 1 \pm 0 . 0 1$ </td><td>&gt; 105</td></tr><tr><td>Heretic</td><td> $0 . 7 6 \pm 0 . 0 1$ </td><td> $0 . 4 8 \pm 0 . 0 1$ </td><td>0.54 ± 0.004</td><td> $0 . 6 7 \pm 0 . 0 1$ </td><td>101.27 0.76 ± 0.01</td><td></td><td> $0 . 4 8 \pm 0 . 0 1$ </td><td> $0 . 5 4 \pm 0 . 0 0 4$ </td><td> $0 . 6 7 \pm 0 . 0 1$ </td><td>101.27</td></tr><tr><td>RDO</td><td> $0 . 7 6 \pm 0 . 0 1$ </td><td> $0 . 4 8 \pm 0 . 0 1$ </td><td> $0 . 5 3 \pm 0 . 0 0 4 0 . 6 5 \pm 0 . 0 1$ </td><td></td><td>101.34 0.76 ± 0.01</td><td></td><td> $0 . 4 8 \pm 0 . 0 1$ </td><td> $0 . 5 3 \pm 0 . 0 0 4$ </td><td> $0 . 6 5 \pm 0 . 0 1$ </td><td>101.34</td></tr><tr><td>WRD-CMAES</td><td> $0 . 7 6 \pm 0 . 0 1$ </td><td> $0 . 4 8 \pm 0 . 0 1$ </td><td> $0 . 5 3 \pm 0 . 0 0 4$ </td><td> $0 . 6 8 \pm 0 . 0 1$ </td><td>101.19  $0 . 7 4 \pm 0 . 0 1$ </td><td></td><td> $0 . 4 6 \pm 0 . 0 1$ </td><td> $0 . 5 0 \pm 0 . 0 0 4$ </td><td> $0 . 6 4 \pm 0 . 0 1$ </td><td>118.19</td></tr><tr><td>WRD-GRPO-IS</td><td> $0 . 7 6 \pm 0 . 0 1$ </td><td> $0 . 4 8 \pm 0 . 0 1$ </td><td> $0 . 5 3 \pm 0 . 0 0 4$ </td><td>0.67 ± 0.01 100.81</td><td> $0 . 7 6 \pm 0 . 0 1$ </td><td></td><td> $0 . 4 7 \pm 0 . 0 1$ </td><td> $0 . 5 3 \pm 0 . 0 0 4$ </td><td> $0 . 6 7 \pm 0 . 0 1$ </td><td>100.49</td></tr></table>

Table 2: Utility preservation per base model, reported at $\alpha \in \lbrace 1 . 0 , 3 . 0 \rbrace$ . Higher is better for ARC-e, ARC-c, MMLU and GSM8K (accuracy as a fraction in [0, 1]); lower is better for PPL (WikiText word-level perplexity). Rows for α-independent methods are repeated across columns for layout consistency. Intermediate values $\alpha \in$ {1.5, 2.0} are reported in Table 6 of Appendix E.

## 5.4 HarmBench

We additionally re-evaluate the considered methods through an external benchmark that was not used during optimization, in order to corroborate the significance of the attack-setup results with a training-independent evaluator and to control for potential overfitting to the guard judges of Section 4.3. Because HarmBench is designed primarily as an evaluator of instruction-tuned models, in this setup we restrict attention to the instruct variants o our model pool, additionally including Qwen/Qwen3-8B [21].

Table 3 reports HarmBench ASR. We adopt the same HarmBench evaluation setup as [5]: we take the prompt set from [17] and score the responses generated by the intervened model with the oficial HarmBench classifier cais/HarmBench-Llama-2-13b-cls. As in the utility experiment of Section 5.3, we restrict this evaluation to the WildGuard-optimized runs: for every method the intervened model evaluated by HarmBench is the one whose coeficient vector was selected against WildGuard during training. HarmBench itself is never used during optimization and serves as an external evaluator that controls for judge-specific overfitting.

Discussion. The HarmBench evaluation confirms the superiority of the WRDO approach observed in Section 5.1: the learned WRDO instantiations attain the highest HarmBench ASR across the α grid. The agreement between the guard-judge metric and a classifier that was never used during optimization confirms that the WRDO advantage over the prior single-direction recipes is a real efect rather than an artifact of the training signal.

## 5.5 Defense setup

We additionally consider a defense setup that reverses the role of WRDO: instead of unlocking harmful behavior on an aligned model, we attempt to suppress it on a model whose refusal has already been broken. Notably, we operate on a third-party abliterated checkpoint whose editing procedure is unknown to us, so we work exclusively from its weights without any knowledge of how the refusal removal was performed.

<table><tr><td rowspan="2">Method</td><td colspan="4">Qwen/Qwen3-8B</td><td colspan="4">Olmo-3-7B-Instruct</td></tr><tr><td>1.0</td><td>1.5</td><td>2.0</td><td>3.0</td><td>1.0</td><td>1.5</td><td>2.0</td><td>3.0</td></tr><tr><td>No intervention</td><td>0.07</td><td>0.07</td><td>0.07</td><td>0.07</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Basic RD</td><td>0.22</td><td>0.31</td><td>0.46</td><td>0.48</td><td>0.51</td><td>0.77</td><td>0.68</td><td>0.66</td></tr><tr><td>Heretic</td><td>0.87</td><td>0.87</td><td>0.87</td><td>0.87</td><td>0.77</td><td>0.77</td><td>0.77</td><td>0.77</td></tr><tr><td>RDO</td><td>0.39</td><td>0.39</td><td>0.39</td><td>0.39</td><td>0.20</td><td>0.20</td><td>0.20</td><td>0.20</td></tr><tr><td>WRD-CMAES</td><td>0.87</td><td>0.92</td><td>0.97</td><td>0.82</td><td>0.81</td><td>0.85</td><td>0.88</td><td>0.89</td></tr><tr><td>WRD-GRPO-IS</td><td>0.95</td><td>0.94</td><td>0.91</td><td>0.97</td><td>0.83</td><td>0.93</td><td>0.86</td><td>0.92</td></tr></table>

Table 3: HarmBench Standard ASR (in [0, 1]) computed with cais/HarmBench-Llama-2-13b-cls, varying α and model. Bold marks the best result in each column within each model block. Higher is a stronger attack.

<table><tr><td></td><td colspan="3"> $\alpha = - 1 . 0$ </td><td colspan="3"> $\alpha = - 1 . 5$ </td><td colspan="3"> $\alpha = - 2 . 0$ </td><td colspan="3"> $\alpha = - 3 . 0$ </td></tr><tr><td>Method</td><td>WG</td><td>Q3G</td><td>LG3</td><td>WG</td><td>Q3G</td><td>LG3</td><td>WG</td><td>Q3G</td><td>LG3</td><td>WG</td><td>Q3G</td><td>LG3</td></tr><tr><td>Abliterated (no defense)</td><td>0.85</td><td>0.83</td><td>0.84</td><td>0.85</td><td>0.83</td><td>0.83</td><td>0.85</td><td>0.83</td><td>0.84</td><td>0.85</td><td>0.83</td><td>0.84</td></tr><tr><td>Basic RD</td><td></td><td>0.84 ±0.02 0.85 ±0.03 0.82 ±0.02</td><td></td><td>0.79 ±0.03</td><td>0.84 ±0.02</td><td>0.79 ±0.03</td><td>0.80 ±0.03</td><td>0.87 ±0.02</td><td>0.87 ±0.03</td><td>0.63 ±0.02</td><td>0.68 ±0.03</td><td>0.74 ±0.02</td></tr><tr><td>WRD-CMAES</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.47 ±0.05 0.56 ±0.04 0.53 ±0.06 0.011 ±0.004 0.023 ±0.005 0.073 ±0.004 0.021 ±0.005 0.053 ±0.006 0.042 ±0.004 0.014 ±0.005 0.086 ±0.006 0.079 ±0.008</td><td></td><td></td><td></td><td></td></tr><tr><td>WRD-GRPO-IS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.50 ±0.08 0.52 ±0.07 0.47 ±0.09 0.014 ±0.007 0.034 ±0.008 0.047 ±0.006 0.015 ±0.007 0.014 ±0.003 0.020 ±0.005 0.019 ±0.009 0.015 ±0.003 0.022 ±0.008</td><td></td><td></td><td></td><td></td></tr></table>

Table 4: Defense setup. Test ASR (in [0, 1]) on the abliterated checkpoint, across the sign-reversed intensity α and the three judges. Lower is a stronger defense. The three intervened rows are means over three independently optimized w<sup>⋆</sup>, with the standard deviation over those runs; the Abliterated row receives no intervention and is reported from a single evaluation. Bold marks the strongest defense per column.

Table 4 reports the defense setup. As the abliterated checkpoint we take the publicly released Orenguteng/ Llama-3.1-8B-Lexi-Uncensored-V2, an uncensored derivative of Llama-3.1-8B-Instruct whose refusal behavior has already been removed by a third-party procedure. We apply the same WRDO pipeline to this checkpoint with the same α grid as in the attack setup, only with reversed sign. For Basic RD this coincides exactly with the defense recipe of Section 3.2 of [5]. The Heretic toolkit and RDO do not expose a defense mode and are therefore excluded from this experiment.

Discussion. The two learned WRDO instantiations clearly win the defense setup, driving the abliterated checkpoint’s ASR far below the no-defense baseline, while Basic RD fails to restore refusal. This shows that the proposed methods adapt to the alternative setup through a simple sign flip, with no change to the optimizer or the candidate bank.

## 5.6 Cross-judge transfer

A cross-judge transfer analysis (Appendix H) identifies WildGuard as the most reliable optimization signal among the considered guards: interventions optimized against WildGuard transfer well to the other two judges, while the reverse direction is less stable.

## 6 Conclusion

We introduced Weighted Refusal Direction Optimization (WRDO), a procedure that constructs a refusal direction by learning aggregation coeficients w over a bank of per-prompt directions, and we presented two derivative-free instantiations of the resulting black-box problem: a WRDO-tailored adaptation of separable CMA-ES and a proposed GRPO variant with importance sampling.

Across three open-weight models, three guard judges and the HarmBench classifier, the two WRDO instantiations exceed all prior single-direction baselines and transfer cleanly to the defense setup. Among them, WRD-GRPO-IS is the most reliable: it preserves the general capabilities of the intervened model substantially better than every other method considered, and the cross-judge analysis singles out WildGuard as the most reliable optimization signal, so we recommend WRD-GRPO-IS optimized against WildGuard as the default configuration for both auditing and defense.

## 7 Limitations

The full protocol of Sections 5.1–5.5 covers three open-weight checkpoints of comparable size, in the 7–8B parameter range.

A second limitation concerns the WRD-CMAES instantiation: its current implementation consumes the bare ASR-only objective of Equation 5 and does not yet support the KL regularizer of Equation 8. Extending the Optuna interface to expose a KL-aware sampling target is under active development, and we expect to incorporate this support in a future revision of the codebase.

## 8 Acknowledgments

The work of G. Evseev, D. Bylinkin, D. Medyakov, V. Aletov, A. Beznosikov was supported by the Ministry of Economic Development of the Russian Federation (agreement No. 139-15-2025-013, dated June 20, 2025, IGK 000000C313925P4B0002).

## 9 Ethical Considerations

This work is inherently dual-use: methods for identifying and modifying refusal directions can help analyze and improve model safety, but they can also be used to weaken safeguards in aligned models. In particular, interventions that suppress refusal may increase the risk of harmful compliance in domains such as cyber abuse, privacy violations, disinformation, or other unsafe requests.

We therefore view WRDO primarily as a tool for controlled mechanistic analysis and safety evaluation. Our experiments are conducted ofline on open-weight models and use automated guard judges, HarmBench-style evaluation, and utility benchmarks to study both safety and capability efects. The goal is to better understand refusal representations, not to encourage unsafe deployment or misuse.

The broader implication of our findings is that refusal behavior may be concentrated in low-dimensional internal features and can therefore be brittle under targeted interventions. This suggests that refusal alone should not be treated as a complete safety mechanism; robust deployment requires layered defenses, including safer training, adversarial evaluation, monitoring, external classifiers, and system-level policy enforcement.

At the same time, the defense experiment shows a constructive use of the same machinery: WRDO-like methods may help diagnose and partially restore refusal behavior in modified or abliterated checkpoints. Overall, we hope this work supports more robust alignment methods and encourages careful, transparent evaluation of both the risks and defensive uses of refusal-direction interventions.

## References

[1] Long Ouyang, Jef Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022.

[2] Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. Training a helpful and harmless assistant with reinforcement learning from human feedback, 2022.

[3] Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models, 2023.

[4] Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. Jailbroken: How does LLM safety training fail?, 2023.

[5] Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. Refusal in language models is mediated by a single direction, 2024. URL https://arxiv.org/abs/2406.11717

[6] Philipp Emanuel Weidmann. Heretic: Fully automatic censorship removal for language models. https: //github.com/p-e-w/heretic, 2025.

[7] Mehmet Eren Ahsen, Robert M. Vogel, and Gustavo A. Stolovitzky. Unsupervised evaluation and weighted aggregation of ranked classification predictions. Journal of Machine Learning Research, 20(166):1–40, 2019.

[8] Abdullah Abdul Sattar Shaikh, M. S. Bhargavi, and Pavan Kumar C. FedWAPR: Bridging theory and practice in probability-driven weighted aggregation for federated learning. Information Sciences, 723:122697, 2026.

[9] Andrey Veprikov, Vladimir Solodkin, Alexander Zyl, Andrey Savchenko, and Aleksandr Beznosikov. Weight-LoRA: Keep only necessary adapters. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL), pages 12419–12437, 2026.

[10] Tom Wollschläger, Jannes Elstner, Simon Geisler, Vincent Cohen-Addad, Stephan Günnemann, and Johannes Gasteiger. The geometry of refusal in large language models: Concept cones and representational independence. arXiv preprint arXiv:2502.17420, 2025.

[11] Nikolaus Hansen and Andreas Ostermeier. Completely derandomized self-adaptation in evolution strategies. Evolutionary Computation, 9(2):159–195, 2001.

[12] Nikolaus Hansen. The cma evolution strategy: A tutorial, 2023. URL https://arxiv.org/abs/1604.00772.

[13] Daan Wierstra, Tom Schaul, Tobias Glasmachers, Yi Sun, Jan Peters, and Jürgen Schmidhuber. Natural evolution strategies, 2014.

[14] Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, and Ilya Sutskever. Evolution strategies as a scalable alternative to reinforcement learning, 2017.

[15] Yangsibo Huang, Samyak Gupta, Mengzhou Xia, Kai Li, and Danqi Chen. Catastrophic jailbreak of open-source LLMs via exploiting generation, 2023.

[16] Mantas Mazeika, Andy Zou, Norman Mu, Long Phan, Zifan Wang, Chunru Yu, Adam Khoja, Fengqing Jiang, Aidan O’Gara, Zhen Xiang, Arezoo Rajabi, Dan Hendrycks, Radha Poovendran, Bo Li, and David Forsyth. TDC 2023 (LLM Edition): The trojan detection challenge. NeurIPS 2023 Competition Track, 2023. URL https://neurips.cc/virtual/2023/competition/66583

[17] Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, and Dan Hendrycks. HarmBench: A standardized evaluation framework for automated red teaming and robust refusal, 2024.

[18] Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramèr, Hamed Hassani, and Eric Wong. JailbreakBench: An open robustness benchmark for jailbreaking large language models, 2024.

[19] Alexandra Souly, Qingyuan Lu, Dillon Bowen, Tu Trinh, Elvis Hsieh, Sana Pandey, Pieter Abbeel, Justin Svegliato, Scott Emmons, Olivia Watkins, and Sam Toyer. A StrongREJECT for empty jailbreaks, 2024.

[20] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Alpaca: A strong, replicable instruction-following model, 2023.

[21] Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

[22] OLMo Team. Olmo 3, 2025. arXiv:2512.13961.

[23] Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, and Nouha Dziri. WildGuard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of LLMs, 2024.

[24] Qwen Team. Qwen3Guard technical report, 2025. arXiv:2510.14276.

[25] Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, and Madian Khabsa. LlamaGuard: LLM-based input-output safeguard for human-AI conversations, 2023.

[26] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jefrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighof, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 07 2024. URL https: //zenodo.org/records/12608602.

[27] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try ARC, the AI2 reasoning challenge, 2018.

[28] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2020.

[29] OpenAI. gpt-oss-120b and gpt-oss-20b model card, 2025.

[30] Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna: A nextgeneration hyperparameter optimization framework. In KDD, 2019.

## Appendix

Supplementary Materials for WRDO: Weighted Refusal Direction Optimization in Aligned Language Models

## A Coeficient-Vector Constraints

The coeficients $w _ { i }$ in Equation 2 are signed real numbers, with no further normalization across coordinates beyond the per-layer $L _ { 2 }$ renormalization already built into Equation 2. The feasible set W of Equation 6 is the per-coordinate box

$$
\mathcal { W } = [ w _ { \mathrm { m i n } } , w _ { \mathrm { m a x } } ] ^ { N } ,\tag{12}
$$

with bounds $w _ { \mathrm { m i n } }$ and $w _ { \mathrm { m a x } }$ fixed as hyperparameters of the search and shared across all derivative-free methods of Section 3. We do not impose a simplex constraint, a unit-norm constraint or a top-k sparsity constraint: the box of Equation 12 is the only restriction on w. All derivative-free methods are initialized at the uniform-average point $w _ { i } = 1 / N$ , which coincides with the Uniform-avg WRD sanity-check baseline of Section 5.1.

## B Layerwise Intensity Schedule

The fixed layerwise schedule $\alpha ^ { ( l ) }$ of Equation 3 is adopted verbatim from the Heretic toolkit [6]. Let $\ell ^ { \star } =$ max\_weight\_position · (L − 1) denote the absolute peak position in the stack, $\Delta = { \mathfrak { m i n } }$ \_weight\_distance · $( L - 1 )$ the half-width of the active window, $\alpha _ { \mathrm { { m i n } } }$ = min\_weight the boundary value of the schedule, and $d _ { l } = | l - \ell ^ { \star } | / \Delta$ the normalized distance of layer l from the peak. The schedule is

$$
\alpha ^ { ( l ) } = \left\{ { \begin{array} { l l } { \alpha + d _ { l } \left( \alpha _ { \mathrm { m i n } } - \alpha \right) , } & { d _ { l } \leq 1 , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } , } \end{array} } \right.\tag{13}
$$

so that the intensity decays linearly from the peak value α = max\_weight at $\ell ^ { \star }$ to $\alpha _ { \mathrm { { m i n } } }$ at the window boundary, and no intervention is applied to layers further than $\Delta$ from the peak. We follow the parameter naming of [6] verbatim. Throughout the paper we use max\_weight\_position = 0.7, min\_weight\_distance = 0.3 and $\alpha _ { \mathrm { m i n } } = 0$ leaving α = max\_weight ∈ {1.0, 1.5, 2.0, 3.0} as the only varied parameter.

## C Data Splits and Sampling Seeds

Source of the splits. All harmful-prompt splits are loaded from the prepared dataset erinya/dataset\_splits on the Hugging Face Hub, with the JSON files splits/harmful\_{train,val,test}.json merged from the six public benchmarks listed in Section 4.1 after exact-prompt-string deduplication.

Harmful train sample for the bank. At bank-construction time we sample a fixed-size subset of the harmful train split to populate the candidate bank R. The number of sampled prompts is N = 128 (ALL\_CATEGORIES\_HARMFUL\_PROMPT\_COUNT in the released code), and the sampling seed is 42 (ALL\_CATEGORIES\_HARMFUL\_PROMPT\_SEED). The same seed is reused across models and methods so that the candidate bank is built from the same N = 128 prompts in every run.

Harmful validation and test splits. Optimization uses the harmful validation split as the source of $\mathcal { D } _ { \mathrm { v a l } } ^ { \mathrm { h a r m } }$ in Equation 4, and the harmful test split as $\mathcal { D } _ { \mathrm { t e s t } } ^ { \mathrm { h a r m } }$ for the held-out evaluations reported in Section 5. Both splits are consumed in order, with a model batch size of 32 per evaluation step (BATCH\_SIZE in the released code).

<table><tr><td></td><td></td><td colspan="3">α = 1.5</td><td colspan="3">α = 2.0</td><td></td></tr><tr><td>Model</td><td>Method</td><td>WG</td><td>Q3G</td><td>LG3</td><td>WG</td><td>Q3G</td><td>LG3</td><td></td></tr><tr><td rowspan="7">Qwen3-8B-Base</td><td>No intervention</td><td>0.02</td><td>0.02</td><td>0.02</td><td>0.02</td><td>0.02</td><td>0.02</td><td></td></tr><tr><td>Basic RD</td><td> $0 . 3 3 \pm 0 . 0 1$ </td><td> $0 . 4 7 \pm 0 . 0 1$ </td><td> $0 . 6 7 \pm 0 . 0 2$ </td><td>0.39 ±0.03</td><td> $0 . 8 7 \pm 0 . 0 2$ </td><td></td><td> $0 . 9 5 \pm 0 . 0 3$ </td></tr><tr><td>Heretic</td><td> $0 . 7 7 \pm 0 . 0 4$ </td><td> $0 . 8 3 \pm 0 . 0 5$ </td><td> $0 . 8 1 \pm 0 . 0 5$ </td><td> $0 . 7 7 \pm 0 . 0 4$ </td><td> $0 . 8 3 \pm 0 . 0 5$ </td><td></td><td> $0 . 8 1 \pm 0 . 0 5$ </td></tr><tr><td>RDO</td><td> $0 . 3 4 \pm 0 . 0 5$ </td><td> $0 . 4 1 \pm 0 . 0 4$ </td><td> $0 . 4 8 \pm 0 . 0 5$ </td><td> $0 . 3 4 \pm 0 . 0 5$ </td><td> $0 . 4 1 \pm 0 . 0 4$ </td><td></td><td> $0 . 4 8 \pm 0 . 0 5$ </td></tr><tr><td>Uniform-avg WRD</td><td> $0 . 0 0 3 \pm 0 . 0 0 0 2$ </td><td> $0 . 8 3 \pm 0 . 0 2$ </td><td> $0 . 8 1 \pm 0 . 0 1$ </td><td> $0 . 0 0 2 \pm 0 . 0 0 0 2$ </td><td> $0 . 8 9 \pm 0 . 0 1$ </td><td></td><td> $0 . 8 5 \pm 0 . 0 3$ </td></tr><tr><td>WRD-CMAES</td><td> $0 . 9 1 \pm 0 . 0 5$ </td><td> ${ \bf 0 . 9 8 \ \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 9 3 \ \pm 0 . 0 2 }$ </td><td> $0 . 9 3 \pm 0 . 0 4$ </td><td> ${ \bf 0 . 9 9 \pm 0 . 0 1 }$ </td><td></td><td> $\mathbf { 0 . 9 8 \ \pm 0 . 0 2 }$ </td></tr><tr><td>WRD-GRPO-IS</td><td> ${ \bf 0 . 9 2 \ \pm 0 . 0 8 }$ </td><td> $0 . 9 6 \pm 0 . 0 2$ </td><td> ${ \bf 0 . 9 3 \ \pm 0 . 0 2 }$ </td><td> $\mathbf { 0 . 9 4 \ \pm 0 . 0 9 }$ </td><td> $0 . 9 5 \pm 0 . 0 2$ </td><td></td><td> $0 . 9 3 \pm 0 . 0 3$ </td></tr><tr><td rowspan="7"> $\mathrm { O L M o { - } 3 { - } 1 0 2 5 { - } 7 B }$ </td><td>No intervention</td><td>0.08</td><td>0.09</td><td>0.07</td><td>0.08</td><td>0.09</td><td></td><td>0.07</td></tr><tr><td>Basic RD</td><td>0.58 ±0.02</td><td>0.56 ±0.03</td><td> $0 . 4 8 \pm 0 . 0 3$ </td><td> $0 . 6 1 \pm 0 . 0 2$ </td><td> $0 . 6 2 \pm 0 . 0 2$ </td><td></td><td>0.69 ±0.03</td></tr><tr><td>Heretic</td><td> $0 . 7 6 \pm 0 . 0 3$ </td><td> $0 . 7 3 \pm 0 . 0 4$ </td><td> $0 . 7 0 \pm 0 . 0 3$ </td><td> $0 . 7 6 \pm 0 . 0 3$ </td><td> $0 . 7 3 \pm 0 . 0 4$ </td><td></td><td> $0 . 7 0 \pm 0 . 0 3$ </td></tr><tr><td>RDO</td><td> $0 . 6 1 \pm 0 . 0 4$ </td><td> $0 . 6 7 \pm 0 . 0 3$ </td><td> $0 . 7 3 \pm 0 . 0 3$ </td><td> $0 . 6 1 \pm 0 . 0 4$ </td><td> $0 . 6 7 \pm 0 . 0 3$ </td><td></td><td> $0 . 7 3 \pm 0 . 0 3$ </td></tr><tr><td>Uniform-avg WRD</td><td> $0 . 6 9 \pm 0 . 0 2$ </td><td> $0 . 5 2 \pm 0 . 0 2$ </td><td> $0 . 5 9 \pm 0 . 0 3$ </td><td> $0 . 7 3 \pm 0 . 0 2$ </td><td> $0 . 7 3 \pm 0 . 0 3$ </td><td></td><td> $0 . 7 5 \pm 0 . 0 1$ </td></tr><tr><td>WRD-CMAES</td><td> $\mathbf { 0 . 8 4 \ \pm 0 . 0 2 }$ </td><td> $0 . 7 8 \pm 0 . 0 3$ </td><td> $0 . 8 1 \pm 0 . 0 2$ </td><td> $0 . 7 4 \pm 0 . 0 2$ </td><td> $0 . 8 4 \pm 0 . 0 3$ </td><td></td><td> $0 . 8 1 \pm 0 . 0 2$ </td></tr><tr><td>WRD-GRPO-IS</td><td> $0 . 8 1 \pm 0 . 0 2$ </td><td> $\mathbf { 0 . 8 5 \ \pm 0 . 0 3 }$ </td><td> $\mathbf { 0 . 8 6 \ \pm 0 . 0 3 }$ </td><td> $\mathbf { 0 . 8 9 \ \pm 0 . 0 2 }$ </td><td> $\mathbf { 0 . 9 1 \bot 0 . 0 2 }$ </td><td></td><td> ${ \bf 0 . 9 6 \ \pm 0 . 0 1 }$ </td></tr><tr><td rowspan="7">Olmo-3-7B-Instruct</td><td>No intervention</td><td>0.003</td><td>0.003</td><td>0.004</td><td>0.003</td><td>0.003</td><td></td><td>0.004</td></tr><tr><td>Basic RD</td><td> $0 . 4 9 \pm 0 . 0 2$ </td><td> $0 . 6 9 \pm 0 . 0 1$ </td><td> $0 . 5 2 \pm 0 . 0 3$ </td><td> $0 . 7 2 \pm 0 . 0 2$ </td><td> $0 . 8 7 \pm 0 . 0 3$ </td><td></td><td> $0 . 7 5 \pm 0 . 0 2$ </td></tr><tr><td>Heretic</td><td> $0 . 6 2 \pm 0 . 0 3$ </td><td> $0 . 5 8 \pm 0 . 0 4$ </td><td> $0 . 6 9 \pm 0 . 0 4$ </td><td> $0 . 6 2 \pm 0 . 0 3$ </td><td> $0 . 5 8 \pm 0 . 0 4$ </td><td></td><td> $0 . 6 9 \pm 0 . 0 4$ </td></tr><tr><td>RDO</td><td> $0 . 4 0 \pm 0 . 0 5$ </td><td> $0 . 5 2 \pm 0 . 0 4$ </td><td> $0 . 5 3 \pm 0 . 0 4$ </td><td> $0 . 4 0 \pm 0 . 0 5$ </td><td> $0 . 5 2 \pm 0 . 0 4$ </td><td></td><td> $0 . 5 3 \pm 0 . 0 4$ </td></tr><tr><td>Uniform-avg WRD</td><td> $0 . 5 7 \pm 0 . 0 2$ </td><td> $0 . 5 3 \pm 0 . 0 2$ </td><td> $0 . 5 8 \pm 0 . 0 3$ </td><td> $0 . 5 4 \pm 0 . 0 3$ </td><td> $0 . 6 9 \pm 0 . 0 2$ </td><td></td><td> $0 . 6 1 \pm 0 . 0 1$ </td></tr><tr><td>WRD-CMAES</td><td> ${ \bf 0 . 9 8 \ \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 9 6 \ \pm 0 . 0 2 }$ </td><td> $0 . 9 2 \pm 0 . 0 1$ </td><td> $0 . 9 2 \pm 0 . 0 2$ </td><td> $0 . 8 8 \pm 0 . 0 3$ </td><td></td><td> $0 . 9 3 \pm 0 . 0 1$ </td></tr><tr><td>WRD-GRPO-IS</td><td> $0 . 9 3 \pm 0 . 0 2$ </td><td> ${ \bf 0 . 9 6 \ \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 9 5 \ \pm 0 . 0 2 }$ </td><td> ${ \bf 0 . 9 3 \ \pm 0 . 0 1 }$ </td><td> ${ \bf 0 . 9 3 \ \pm 0 . 0 2 }$ </td><td></td><td> ${ \bf 0 . 9 6 \ \pm 0 . 0 1 }$ </td></tr></table>

Table 5: Attack setup at the intermediate intensities $\alpha \in \{ 1 . 5 , 2 . 0 \}$ , complementing the main Table 1. Judges: $\mathrm { W G } =$ WildGuard, Q3G = Qwen3Guard-Gen-8B, LG3 = Llama-Guard-3-8B. Every intervened cell is a three-seed mean with the standard deviation over those repetitions; the Heretic and RDO rows repeat the values of Table 1 because neither method consumes α. Bold marks the best result in each column within each model block. Higher is a stronger attack.

Benign baseline. The harmless baseline $\mathcal { D } _ { \mathrm { b e n i g n } }$ used to compute $\mu _ { \mathrm { b e n i g n } } ^ { ( l ) }$ in Equation 1 is the first 400 prompts of the train split of mlabonne/harmless\_alpaca. A separate 50-prompt subset of the test split of the same dataset is used as the benign batch for the KL penalty in Equation 8.

Optimizer seeds. The Optuna sampler seed is fixed at 42 (OPTUNA\_SAMPLER\_SEED). For WRD-GRPO-IS, behaviorpolicy perturbations $\varepsilon _ { m } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } I _ { N } )$ are drawn from the active PyTorch RNG at training time, with σ controlled by GRPO\_NOISE\_SCALE.

## D Attack Setup at Intermediate α

Table 5 complements Table 1 by reporting the attack setup of Section 5.1 at the intermediate intensities $\alpha \in \{ 1 . 5 , 2 . 0 \}$ omitted from the main text. Column conventions, the bold-per-column rule and the three-seed reporting follow Table 1 unchanged; because Heretic and RDO do not consume α, their rows repeat the values of Table 1. The reading of Section 5.1 carries over without exception: the two learned WRDO instantiations hold the best cell in al model-judge columns of this table, and Uniform-avg WRD remains below them everywhere, so neither conclusion of Section 5.1 depends on the two intensities kept in the main text.

## E Utility Preservation at Intermediate α

Table 6 complements Table 2 by reporting the utility benchmarks at the intermediate intensity values $\alpha \in \{ 1 . 5 , 2 . 0 \}$ omitted from the main text. Column conventions, the WildGuard-only restriction and the Heretic-duplication remark of Section 5.3 apply unchanged.

<table><tr><td colspan="10">α = 1.5</td></tr><tr><td>Model</td><td>Method</td><td>ARC-e</td><td>ARC-c</td><td>MMLU</td><td>GSM8K</td><td>PPL</td><td>ARC-e</td><td>ARC-c MMLU</td><td>GSM8K</td><td>PPL</td></tr><tr><td rowspan="6">Qwen3-8B-Base</td><td>No intervention</td><td>0.63 ± 0.01</td><td>0.38 ± 0.01 0.67</td><td>± 0.004</td><td>0.84 ± 0.01</td><td>11.11</td><td>0.63 ± 0.01 0.38</td><td>± 0.01 0.67 ± 0.004</td><td>0.84 ± 0.01</td><td>11.11</td></tr><tr><td>Basic RD</td><td>0.42 ± 0.01</td><td>0.26 0.01 0.23 H</td><td>± 0.004</td><td>0.008 ± 0.002</td><td>113.98 0.29 ± 0.01</td><td>0.22 ±</td><td>0.01 0.23 ± 0.004</td><td>0.00 ± 0.00</td><td>&gt; 107</td></tr><tr><td>Heretic</td><td>0.51 ± 0.01</td><td>0.30 ± 0.01 0.27</td><td>± 0.004</td><td>0.19 ± 0.01</td><td>13.44 0.51 ± 0.01</td><td>0.30</td><td>0 ± 0.01 0.27 ± 0.004</td><td>0.19 ± 0.01</td><td>13.44</td></tr><tr><td>RDO</td><td>0.63 ± 0.01</td><td>0.37 ± 0.01</td><td>0.67 ± 0.004</td><td>0.82 ± 0.01</td><td>11.16</td><td>0.63 ± 0.01 0.37</td><td>± 0.01 0.67 ± 0.004</td><td>0.82 ± 0.01</td><td>11.16</td></tr><tr><td>WRD-CMAES</td><td>0.63 ± 0.01</td><td>0.38 ± 0.01</td><td>0.64 ± 0.004</td><td>0.84 ± 0.01</td><td>11.25</td><td>0.51 ± 0.01 0.30 ±</td><td>0.01 0.47 ± 0.004</td><td>0.04 ± 0.005</td><td>64.84</td></tr><tr><td>WRD-GRPO-IS</td><td>0.63 ± 0.01</td><td>0.38 ± 0.01 0.64</td><td> ± 0.004</td><td>0.84 ± 0.01</td><td>11.17</td><td>0.64 ± 0.01 0.38 ±</td><td>0.01 0.65 ± 0.004</td><td>0.85 ± 0.01</td><td>11.27</td></tr><tr><td rowspan="6">OLMo-3-1025-7B</td><td>No intervention</td><td>0.79 ± 0.01</td><td>0.48 ± 0.01 0.62</td><td>± 0.004</td><td>0.53 ± 0.01</td><td>62.5 0.79 ± 0.01</td><td>0.48 ±</td><td>0.01 0.62 ± 0.004</td><td>0.53 ± 0.01</td><td>62.5</td></tr><tr><td>Basic RD</td><td>0.79 ± 0.01</td><td>0.48 ± 0.01 0.62</td><td>± 0.004</td><td>0.55 ± 0.01</td><td>61.81</td><td>0.79 ± 0.01 0.47</td><td>7 ± 0.01 0.61 ± 0.004</td><td>0.50 ± 0.01</td><td>72.75</td></tr><tr><td>Heretic</td><td>0.79 ± 0.01</td><td>0.48 ± 0.01 0.62</td><td> ± 0.004</td><td>0.52 ± 0.01</td><td>62.61</td><td>0.79 ± 0.01 0.48 ± 0.01</td><td>0.62 ± 0.004</td><td>0.52 ± 0.01</td><td>62.61</td></tr><tr><td>RDO</td><td>0.80 ± 0.01</td><td>0.48 ± 0.01</td><td>0.62 ± 0.004</td><td>0.56 ± 0.01</td><td>63.32</td><td>0.80 ± 0.01 0.48</td><td>± 0.01 0.62 ± 0.004</td><td>0.56 ± 0.01</td><td>63.32</td></tr><tr><td>WRD-CMAES</td><td>0.79 ± 0.01</td><td>0.47 7 ± 0.01 0.62</td><td>± 0.004</td><td>0.52 ± 0.01</td><td>62.39</td><td>0.79 ± 0.01 0.48</td><td>± 0.01 0.62 ± 0.004</td><td>0.51 ± 0.01</td><td>62.57</td></tr><tr><td>WRD-GRPO-IS</td><td>0.79 ± 0.01</td><td>0.48 ± 0.01 0.62</td><td> ± 0.004</td><td>0.52 ± 0.01</td><td>62.88</td><td>0.79 ± 0.01 0.48</td><td>± 0.01 0.62 ± 0.004</td><td>0.53 ± 0.01</td><td>62.31</td></tr><tr><td rowspan="6">Olmo-3-7B-Instruct</td><td>No intervention</td><td>0.76 ± 0.01</td><td>0.48 ± 0.01 0.53</td><td>± 0.004</td><td>0.67 ± 0.01</td><td>100.83 0.76 ± 0.01</td><td>0.48 ± 0.01</td><td>0.53 ± 0.004</td><td>0.67 ± 0.01</td><td>100.83</td></tr><tr><td>Basic RD</td><td>0.75 ± 0.01</td><td>0.47 ± 0.01</td><td>0.53 ± 0.004</td><td>0.65 ± 0.01</td><td>102.96</td><td>0.59 ± 0.01</td><td>0.31 ± 0.01 0.42 ± 0.004</td><td>0.61 ± 0.01</td><td>&gt; 104</td></tr><tr><td>Heretic</td><td>0.76 ± 0.01</td><td>0.48 ± 0.01</td><td>0.54 ± 0.004</td><td>0.67 ± 0.01</td><td>101.27</td><td>0.76 ± 0.01</td><td>0.48 ± 0.01 0.54 ± 0.004</td><td>0.67 ± 0.01</td><td>101.27</td></tr><tr><td>RDO</td><td>0.76 ± 0.01</td><td>0.48 ± 0.01</td><td>0.53 ± 0.04</td><td>0.65 ± 0.01</td><td>101.34</td><td>0.76 ± 0.01 0.48</td><td>± 0.01 0.53 ± 0.04</td><td>0.65 ± 0.01</td><td>101.34</td></tr><tr><td>WRD-CMAES</td><td>0.77 ± 0.01</td><td>0.47 0.01 ±</td><td>0.53 ± 0.004</td><td>0.67 ± 0.01</td><td>101.19</td><td>0.76 ± 0.01</td><td>0.47 ± 0.01 0.53 ± 0.004</td><td>0.66 ± 0.01</td><td>103.42</td></tr><tr><td>WRD-GRPO-IS</td><td>0.76 ± 0.008</td><td>0.48 ± 0.01</td><td>0.53 ± 0.004</td><td>0.67 ± 0.01</td><td>100.83</td><td>0.76 ± 0.008</td><td>0.49 ± 0.01 0.52 ± 0.004</td><td>0.67 ± 0.01</td><td>100.22</td></tr></table>

Table 6: Utility preservation per base model at the intermediate intensities α ∈ {1.5, 2.0}, complementing the main Table 2. Higher is better for ARC-e, ARC-c, MMLU and GSM8K; lower is better for PPL.

## F Training Hyperparameters

This section documents the training-time hyperparameters of all six methods used in the main experiments. All values are reproduced in Table 7; the paragraphs below give a short narrative for each method. For the prior baselines (Basic RD, Heretic, RDO) we use the hyperparameter values from their original implementations without any additional tuning.

Basic RD. Basic RD performs no learned optimization. The only tunable knobs are the dataset filters used to construct the harmful/harmless splits and to select the seed diference-of-means direction: harmful prompts that the unintervened model already refuses are discarded if their refusal-induction score is below induce\_refusal\_threshold, candidate directions whose KL to the clean policy on the benign batch exceeds kl\_threshold are dropped, and the bottom prune\_layer\_percentage of layers is excluded from the layer-search. All three thresholds are taken verbatim from the reference implementation of Arditi et al. [5].

Heretic. Heretic runs an Optuna-based Bayesian search over the layerwise schedule of Equation 3. The search budget is n\_trials with n\_startup\_trials random-initialization trials, and uses a KL constraint with scale kl\_divergence\_scale and target kl\_divergence\_target. Trial selection on the Pareto front prefers the minimum-KL trial whose refusal count is at most refusal\_threshold\_count. The search ranges over max\_weight ∈ [0.8, 1.5], max\_weight\_position relative to depth in [0.6, 1.0], min\_weight ∈ [0.0, 1.0], min\_weight\_distance relative to depth in [0.05, 0.6], the direction-index fraction in [0.4, 0.9], and the schedule scope ∈ {global, per-layer}. We use the released Heretic toolkit out of the box, with the defaults of Weidmann [6] for every hyperparameter listed above.

RDO. RDO trains a single per-layer direction by gradient descent with AdamW. The training loop runs for epochs steps with patience-based early stopping and at most n\_lr\_reduce learning-rate reductions, at learning rate lr and efective batch size effective\_batch\_size accumulated from micro-batches of size batch\_size. The composite loss is a weighted sum of an ablation term (weight ablation\_lambda), an addition term (addition\_lambda) and a retain term (retain\_lambda). Target sequences for the loss are generated greedily up to num\_target\_tokens tokens in batches of target\_generation\_batch\_size. All values match the reference implementation released by Wollschläger et al. [10].

<table><tr><td>Method</td><td>Hyperparameter</td><td>Value</td></tr><tr><td>Basic RD</td><td>kl_threshold induce_refusal_threshold prune_layer_percentage</td><td>0.1 0.0 0.2</td></tr><tr><td>Heretic</td><td>n_trials /n_startup_trials kl_divergence_scale/_target refusal_threshold_count max_weight range min_weight range seed</td><td>200 / 60 1.0 / 0.01 0 [0.8, 1.5] [0.0, 1.0] 42</td></tr><tr><td>RDO</td><td>optimizer lr batch_size /effective_batch_size epochs /patience /n_lr_reduce  $( \lambda _ { \mathrm { a b l } } , \lambda _ { \mathrm { a d d } } , \lambda _ { \mathrm { r e t } } )$  num_target_tokens target_generation_batch_size seed</td><td>AdamW  $1 0 ^ { - 2 }$  1 /16  $1 ~ / ~ 5 ~ / ~ 2$  (1.0, 0.2, 1.0) 8 64 42</td></tr><tr><td>WRD-CMAES</td><td>sampler n_trials /sampler_seed  $( w _ { \mathrm { m i n } } , w _ { \mathrm { m a x } } )$  init / mode optimizer</td><td>CmaEsSampler (separable) 50/42 (−2.0, +2.0) uniform-avg / scalar AdamW  $1 0 ^ { - 3 }$  </td></tr><tr><td>WRD-GRPO-IS</td><td>learning_rate n_groups (M) n_epochs σ (noise_scale) clip_ratio /is_clip_ratio β (KL weight) ref_alpha</td><td>8 50 1 0.2 / 5.0 1000.0 1.0</td></tr></table>

Table 7: Training hyperparameters for the six methods of Section 3. Values are read verbatim from the configuration files in the released codebase. Hyperparameters shared across methods (sampling seed 42, batch size 16 or 32, maximum response length, system prompt) are listed in Appendix C.

WRD-CMAES. WRD-CMAES uses the Optuna [30] implementation of separable CMA-ES through the CmaEsSampler interface. The search variables are the coeficients $w \in \mathbb { R } ^ { N }$ of Section 2.2, constrained to the box $[ w _ { \mathrm { m i n } } , w _ { \mathrm { m a x } } ] ^ { N }$ of Appendix A. The total query budget is n\_trials, the sampler seed is sampler\_seed, the per-coordinate bounds are $w _ { \mathrm { m i n } } = - 2 . 0$ and $w _ { \mathrm { m a x } } = 2 . 0$ , and the parameterization is the scalar mode initialized at the uniform-average point $( w _ { i } = 1 / N )$ . All other sampler hyperparameters are left at library defaults.

WRD-GRPO-IS. WRD-GRPO-IS optimizes the surrogate of Equation 10 with AdamW at learning rate learning\_rate. Each training step samples a group of n\_groups behaviour policies $\{ \mu _ { m } \}$ from the Gaussian rollout of Equation 11 with noise scale σ = noise\_scale, and a fresh group is drawn every epoch over n\_epochs total epochs. The PPO/GRPO clip ratio is clip\_ratio and the importance-sampling clip ratio is is\_clip\_ratio; loss is aggregated with the loss\_agg\_mode reduction. The KL penalty of Equation 8 is realized with coeficient β = beta. The reference-policy mixture weight is ref\_alpha. The reward is the harmfulness score with the sign of Equation 6, and w is parameterized in the scalar mode initialized at the uniform-average point.

## G Compute Resources

Every single run reported in this paper was conducted on a single NVIDIA H100 GPU with 80 GB of memory.   
Table 8 reports the approximate wall-clock runtime per method for the attack-setup experiments of Section 5.1.

## H Cross-Judge Transfer

The learned WRDO instantiations consume a specific guard judge as their training signal, which raises the possibility that the resulting coeficient vector w<sup>⋆</sup> exploits idiosyncrasies of that judge rather than capturing a transferable refusal-suppression direction. We probe this potential judge overfitting through a cross-judge transfer analysis: $w ^ { \star }$ is optimized against each guard in turn and then re-evaluated against the other two, with the magnitude of the of-diagonal drop quantifying the extent to which performance is judge-specific.

<table><tr><td>Method</td><td>Approx. runtime, hours</td></tr><tr><td>Basic RD</td><td>～2</td></tr><tr><td>Heretic</td><td>~7</td></tr><tr><td>RDO</td><td>~3</td></tr><tr><td>Uniform-avg WRD</td><td>～2</td></tr><tr><td>WRD-CMAES</td><td>~4</td></tr><tr><td>WRD-GRPO-IS</td><td>~11</td></tr></table>

Table 8: Approximate wall-clock runtime per method on the attack-setup experiments of Section 5.1, on a single NVIDIA H100 80 GB GPU.

Table 9 reports the cross-judge transfer matrix obtained by optimizing $w ^ { \star }$ against one guard judge and then evaluating the resulting intervention against each of the three guard judges, for every value of $\alpha \in \{ 1 . 0 , 1 . 5 , 2 . 0 , 3 . 0 \}$ Large of-diagonal drops indicate that the optimizer has fit features specific to the training judge rather than a transferable refusal-suppression signal. The matrix is reported for the three judge-optimized methods; Basic RD, Heretic and RDO have no judge dependence and are omitted.

Discussion. The cross-judge analysis shows that learned WRDO interventions do not always transfer equally across guard judges. In several cases, optimizing against one judge yields high ASR under that same judge but noticeably weaker performance under another, indicating that part of the learned direction may depend on judge-specific decision boundaries rather than on a fully transferable refusal-suppression signal.

The most pronounced asymmetry is observed for WildGuard: interventions that perform well under WildGuard tend to transfer relatively well to the other judges, while interventions optimized for other judges can fail to preserve high WildGuard ASR. This suggests that WildGuard provides the most reliable and demanding optimization signal among the considered guards.

By contrast, Llama-Guard-3 often reports lower ASR than the other judges. We interpret this cautiously: rather than necessarily indicating weaker attacks, this may reflect lower sensitivity of Llama-Guard-3 to harmful behavior in the evaluated generations, leading to more false negatives or inconsistent harmfulness detection. As a result, lower Llama-Guard-3 scores should not be treated as definitive evidence that the intervention is less efective.

## I Concept Cones and Scale

Two comparisons fall outside the grid of Table 1. First, the closest prior work argues that refusal occupies a subspace rather than a line [10], so we evaluate its concept-cone construction RDO-CONE (Section 3.1) on the same three models and test split. Second, all three checkpoints sit in the 7–8B range, so we repeat the attack setup on openai/gpt-oss-20b [29], a 21B mixture-of-experts model with 3.6B active parameters, leaving the pipeline of Section 5.1 unchanged. Both are scored by WildGuard and reported in Table 12.

<table><tr><td></td><td></td><td></td><td colspan="3">α = 1.0</td><td colspan="3">α = 1.5</td><td colspan="3">α = 2.0</td><td colspan="3">α = 3.0</td></tr><tr><td>Model</td><td>Method</td><td>Opt. judge</td><td>WG</td><td>Q3G</td><td>LG3</td><td>WG</td><td>Q3G</td><td>LG3</td><td>WG</td><td>Q3G</td><td>LG3</td><td>WG</td><td>Q3G</td><td>LG3</td></tr><tr><td rowspan="5">Qwen3-8B-Base</td><td rowspan="2">WRD-CMAES</td><td>WG</td><td>0.89</td><td>0.89</td><td>0.92</td><td>0.91</td><td>0.92</td><td>0.95</td><td>0.95</td><td>0.96</td><td>0.96</td><td>0.90</td><td>0.93</td><td>0.92</td></tr><tr><td>Q3G</td><td>0.00</td><td>0.84</td><td>0.97</td><td>0.00</td><td>0.98</td><td>0.96</td><td>0.00</td><td>1.00</td><td>0.97</td><td>0.00</td><td>1.00</td><td>0.97</td></tr><tr><td rowspan="2"></td><td>LG3</td><td>0.00</td><td>0.73</td><td>0.88</td><td>0.02</td><td>0.94</td><td>0.93</td><td>0.16</td><td>0.96</td><td>0.98</td><td>0.01</td><td>0.98</td><td>0.98</td></tr><tr><td>WG</td><td>0.91</td><td>0.94</td><td>0.90</td><td>0.93</td><td>0.95</td><td>0.92</td><td>0.95</td><td>0.96</td><td>0.94</td><td>0.95</td><td>0.96</td><td>0.93</td></tr><tr><td rowspan="3">WRD-GRPO-IS</td><td>Q3G</td><td>0.91</td><td>0.94</td><td>0.90</td><td>0.94</td><td>0.95</td><td>0.92</td><td>0.01</td><td>0.96</td><td>0.98</td><td>0.94</td><td>0.95</td><td>0.93</td></tr><tr><td rowspan="2"></td><td>LG3</td><td>0.87</td><td>0.92</td><td>0.87</td><td>0.94</td><td>0.96</td><td>0.92</td><td>0.36</td><td>0.42</td><td>0.39</td><td>0.91</td><td>0.95</td><td>0.97</td></tr><tr><td>WG</td><td>0.86</td><td>0.94</td><td>0.50</td><td>0.83</td><td></td><td>0.20</td><td></td><td></td><td></td><td></td><td>0.49</td></tr><tr><td rowspan="5">OLMo-3-1025-7B</td><td rowspan="3">WRD-CMAES</td><td>Q3G</td><td>0.85</td><td>0.86</td><td>0.55</td><td>0.77</td><td>0.75 0.79</td><td>0.53</td><td>0.74 0.82</td><td>0.70 0.83</td><td>0.35 0.38</td><td>0.89 0.94</td><td>0.89 0.87</td><td>0.46</td></tr><tr><td>LG3</td><td>0.65</td><td>0.69</td><td>0.82</td><td>0.63</td><td>0.65</td><td>0.80</td><td>0.69</td><td>0.74</td><td>0.80</td><td>0.91</td><td>0.91</td><td>0.85</td></tr><tr><td>WG</td><td>0.83</td><td>0.76</td><td>0.42</td><td>0.80</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">WRD-GRPO-IS</td><td>Q3G</td><td>0.62</td><td>0.86</td><td>0.30</td><td>0.65</td><td>0.73 0.84</td><td>0.40 0.32</td><td>0.89 0.85</td><td>0.79 0.91</td><td>0.50 0.45</td><td>0.91 0.92</td><td>0.89 0.97</td><td>0.43 0.50</td></tr><tr><td>LG3</td><td>0.60</td><td>0.65</td><td>0.79</td><td>0.65</td><td>0.70</td><td>0.85</td><td>0.80</td><td>0.85</td><td>0.95</td><td>0.88</td><td>0.92</td><td>0.96</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="5">Olmo-3-7B-Instruct</td><td rowspan="3">WRD-CMAES</td><td>WG Q3G</td><td>0.91 0.17</td><td>0.91 0.89</td><td>0.95 0.82</td><td>0.98</td><td>0.96</td><td>0.97 0.96</td><td>0.92</td><td>0.95</td><td>0.93 0.86</td><td>0.88</td><td>0.83</td><td>0.76</td></tr><tr><td>LG3</td><td>0.17</td><td>0.96</td><td>0.92</td><td>0.29 0.22</td><td>0.95</td><td></td><td>0.62</td><td>0.87</td><td></td><td>0.76</td><td>0.93</td><td>0.88</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>0.97</td><td>0.91</td><td>0.59</td><td>0.94</td><td>0.92</td><td>0.79</td><td>0.88</td><td>0.85</td></tr><tr><td rowspan="3">WRD-GRPO-IS</td><td>WG</td><td>0.90 0.23</td><td>0.95</td><td>0.21</td><td>0.93</td><td>0.90</td><td>0.47 0.80</td><td>0.93</td><td>0.98</td><td>0.58</td><td>0.93</td><td>0.97</td><td>0.72</td></tr><tr><td>Q3G LG3</td><td>0.22</td><td>0.88 0.30</td><td>0.78 0.91</td><td>0.35 0.30</td><td>0.96 0.42</td><td>0.94</td><td>0.58 0.45</td><td>0.92 0.58</td><td>0.90 0.95</td><td>0.78 0.80</td><td>0.91 0.85</td><td>0.80 0.87</td></tr></table>

Table 9: Cross-judge transfer. Test ASR (in [0, 1]) when $w ^ { \star }$ is optimized against the row judge and evaluated against the column judge, reported per base model, for every $\alpha \in \{ 1 . 0 , 1 . 5 , 2 . 0 , 3 . 0 \}$ . Within each α block, diagonal cells coincide with the corresponding entries of Table 1 or, for $\alpha \in \{ 1 . 5 , 2 . 0 \}$ , of Table 5.

Discussion. The two constructions perform comparably, with the ordering depending on the checkpoint rather than favouring either consistently. Learned weighted aggregation is therefore competitive with an explicitly multidimensional cone while keeping the operator of Equation 3 rank-one per layer. We read this as evidence that the gain of WRDO comes from where the harmful prompts are allowed to disagree rather than from the dimensionality of the ablated subspace.

<table><tr><td>β</td><td>1</td><td>10</td><td> $1 0 ^ { 2 }$ </td><td> $1 0 ^ { 3 }$ </td><td> $1 0 ^ { 4 }$ </td></tr><tr><td>ASR (WG)</td><td>0.95</td><td>0.92</td><td>0.91</td><td>0.91</td><td>0.78</td></tr></table>

Table 10: Attack side of the $\beta$ sweep. Test ASR under WildGuard for WRD-GRPO-IS on Qwen3-8B-Base at $\alpha = 1 . 0 $ Higher is a stronger attack.

## J External Evaluation of the Defense Setup

Table 11 re-scores the defended checkpoints of Section 5.5 with the external HarmBench classifier, which plays no part in the optimization, and therefore complements the three guard judges of Table 4. At $\alpha \leq - 1 . 5$ both learned instantiations hold HarmBench ASR at or below 0.03 against 0.94 for the undefended checkpoint, whereas Basic

<table><tr><td>Method</td><td>-1.0 -1.5 -2.0 -3.0</td></tr><tr><td>Abliterated (no defense)</td><td>0.94 0.94 0.94 0.94</td></tr><tr><td>Basic RD</td><td>0.94 0.92 0.91 0.31</td></tr><tr><td>WRD-CMAES</td><td>0.47 0.03 0.00 0.03</td></tr><tr><td>WRD-GRPO-IS</td><td>0.47 0.02 0.00 0.00</td></tr></table>

Table 11: Defense setup evaluated externally. HarmBench Standard ASR computed with cais/HarmBench-Llama-2-13b-cls on the abliterated checkpoint, across the sign-reversed intensity α. Harm-Bench is never used during optimization. Lower is a stronger defense.

<table><tr><td>Model</td><td>Method</td><td>ASR</td></tr><tr><td>Qwen3-8B-Base</td><td>RDO-CONE WRD-GRPO-IS</td><td>0.953 0.93 ±0.09</td></tr><tr><td>OLMo-3-1025-7B</td><td>RDO-CONE WRD-GRPO-IS</td><td>0.865 0.92 ±0.02</td></tr><tr><td>Olmo-3-7B-Instruct</td><td>RDO-CONE WRD-GRPO-IS</td><td>0.845  $\mathbf { 0 . 9 3 \ \pm 0 . 0 1 }$ </td></tr><tr><td rowspan="4">gpt-oss-20b</td><td>No intervention</td><td>0.001</td></tr><tr><td>Basic RD</td><td>0.308</td></tr><tr><td>Heretic</td><td>0.772</td></tr><tr><td>WRD-CMAES WRD-GRPO-IS</td><td>0.923 0.931</td></tr></table>

Table 12: Concept-cone comparison. Test ASR under WildGuard; WRDO rows are taken at $\alpha = 3 . 0 \AA$ The WRD-GRPO-IS rows of the upper blocks repeat the corresponding cells of Table 1 and carry its three-seed spread; the RDO-CONE and gpt-oss-20b rows are single runs. Higher is a stronger attack.

RD stays above 0.90 until $\alpha = - 3 . 0$ , an intensity at which its utility has already collapsed in the attack direction (Table 2).

## K Sensitivity to the KL Coeficient $\beta$

The KL coeficient $\beta$ of Equation $^ 8$ is the single hyperparameter that sets where WRD-GRPO-IS sits on the attack-versus-utility trade-of, so the value reported throughout the paper is not an arbitrary default. We select it by sweeping $\beta \in \{ 1 , 1 0 , 1 0 ^ { 2 } , 1 0 ^ { 3 } , 1 0 ^ { 4 } \}$ on Qwen3-8B-Base at α = 1.0, with everything else held at the values of Table $^ { 7 , }$ and reading ASR and utility of the same run. Tables 10 and 13 report both sides of the sweep. The selection is made once, on this model, and the resulting value is reused unchanged for every other model and judge in the paper; we did not tune $\beta$ per model.

The two sides of the sweep move in opposite directions, but not at the same rate. ASR falls gradually as $\beta$ grows, staying almost flat over the middle of the range before dropping of at the largest value. Utility does not track it: it stays poor while the penalty is weak, with MMLU and GSM8K far below the clean model and perplexity an order of magnitude above ${ \mathrm { i t } } ,$ and then recovers abruptly in a single step to the unintervened level. The two regimes overlap in one place, and that is where we operate: $\beta = 1 0 ^ { 3 }$ is the only setting that combines an ASR at the top of the flat region with utility indistinguishable from the clean checkpoint, and it is the value used throughout the paper.

<table><tr><td> $\beta$  ARC-e ARC-c MMLU</td><td>GSM8K</td><td>PPL</td></tr><tr><td>1 0.52 0.34</td><td>0.25 0.38</td><td>261.32</td></tr><tr><td>10 0.51 0.36</td><td>0.34 0.43</td><td>260.13</td></tr><tr><td> $1 0 ^ { 2 }$  0.60 0.36</td><td>0.59 0.67</td><td>150.72</td></tr><tr><td> $1 0 ^ { 3 }$  0.63 0.38</td><td>0.65 0.83</td><td>11.15</td></tr><tr><td> $1 0 ^ { 4 }$  0.63 0.38</td><td>0.65</td><td>0.84 11.12</td></tr></table>

Table 13: Utility side of the $\beta$ sweep, same runs as Table 10. Higher is better for the four accuracies, lower is better for WikiText perplexity. Standard errors are 0.01 for ARC-e, ARC-c and GSM8K and 0.004 for MMLU. Bold marks the best value per column.