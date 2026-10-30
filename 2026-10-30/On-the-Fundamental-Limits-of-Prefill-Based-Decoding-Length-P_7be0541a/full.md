# On the Fundamental Limits of Prefill-Based Decoding Length Prediction

Haowen Shi<sup>1</sup>, Xuzhe Yan<sup>1</sup>, Fahao Chen<sup>2</sup>, Peng Li<sup>1</sup>

<sup>1</sup>Xi’an Jiaotong University, Xi’an, China <sup>2</sup>Shandong University, Jinan, China

HaowenShi@outlook.com, Zedekiah5571@outlook.com, chenfh@ieee.org, pengli@xjtu.edu.cn

## Abstract

Predicting how many tokens a large language model (LLM) will decode, using only prefill-phase hidden states, is a key primitive for inference scheduling. We show that current benchmarks conflate predictor quality with task hardness: when the target model, dataset, or sampling configuration changes, reported accuracy collapses even though the predictor is unchanged. We trace this instability to an overlooked error source—the inherent stochasticity of autoregressive decoding—and derive bounds that every prefill-side method must obey. Our analysis treats decoding as a stopped stochastic process. The perstep stopping probability of a response is not constant but ramps up over hundreds of steps, so the decode length inherits a central limit theorem from the many local decisions that determine where a response ends; this yields per-prompt normality with an explicit error term, together with an afine variance law $\sigma ~ = ~ \alpha \mu + b$ whose slope is the coeficient of variation of the number of response segments. Empirically, the normal law reproduces the per-prompt MAE and the central mass of the decode-length distribution to within a few percent, and the residual departures from normality (mild right skew, slight excess kurtosis) are the ones predicted by the samplingnoise correction. Under fixed model, data, and sampling conditions, a broad design space of predictors (MLP, EGTP, FiLM, soft-label, and Mixture-of-Experts variants) converges to similar performance close to the bound, confirming that system error dominates model error. We argue for a benchmarking protocol that fixes the model, data, and sampling parameters and that reports relative-error metrics alongside absolute ones. Code and analysis tools are available at https: //github.com/ShwStone/DecodingPredict.

## 1 Introduction

Large language model (LLM) inference consists of two distinct phases: a compute-bound prefill phase that processes the input prompt in parallel, and a memory-bound decode phase that generates output tokens autoregressively [Kwon et al., 2023, Yu et al., 2022]. Because the decode phase dominates end-to-end latency and energy consumption [Zhong et al., 2024], knowing how many tokens will be decoded before decoding begins is invaluable for batch scheduling, PD disaggregation, and resource provisioning [Shahout et al., 2024, Qiu et al., 2024, Zheng et al., 2023].

Motivated by this need, a growing body of work proposes to predict decoding length from prefill-phase information. Early approaches train external proxy models on surface features [Qiu et al., 2024]. More recent methods reuse the target LLM’s own internal activations: Shahout et al. [2024] schedule requests with intermediate-layer embeddings; Piotrowski et al. [2025] build a graph convolutional regressor over layer-wise hidden states; and the concurrent EGTP framework [Xie et al., 2026] introduces Entropy-Guided Token Pooling (EGTP) over on-the-fly hidden states, achieving state-of-theart results on the newly released ForeLen benchmark.

The fairness problem. Despite these advances, we observe that published numbers are dificult to compare. A predictor may achieve MAE ≈ 80 on ForeLen’s long-sequence split with Qwen2.5-7B, yet the same architecture degrades to $\mathrm { M A E } > 1 5 0$ on ShareGPT with Qwen3-32B, or to MAE > 260 on RL-sampled data. The root cause is that the inherent randomness of autoregressive generation—not predictor design—is the dominant error source. When a query has high decode-length variance, no prefill-side predictor can forecast a single deterministic length with high accuracy, because the length itself is a random variable.

Our contributions. We make three contributions.

1. A theory of where the error comes from. We model decoding as a stopping time with a non-constant hazard and show that the decode length is asymptotically normal, with an explicit Berry–Esseen error term (§3). The same analysis yields the afine variance law σ = αµ + b and identifies the fitted slope as the coeficient of variation of the number of response segments. From the normal law we then derive bounds that hold for every prefill-side predictor, including sharpened MAE and pass-rate bounds and an exact decomposition of the excess risk into a system and a model term.

2. Empirical validation, including where the law fails. On 300 ShareGPT prompts with 200 decodes each, we test the normal approximation against the quantities that the bounds actually use: the per-prompt MAE matches the Gaussian value $\sigma { \sqrt { 2 / \pi } }$ to within 1.9%, and the central mass matches to within two percentage points (§4).

The residual departures from normality (median skewness +0.27, excess kurtosis +0.35) are those predicted by the sampling-noise correction of our analysis. We also confirm that $\sigma$ follows the afine law in $\mu ,$ with coeficients $( \alpha , b )$ that vary markedly across models and posttraining recipes (Table 1).

3. Architecture-agnostic evaluation. We benchmark a wide range of predictor designs—MLP regression on concatenated hidden states, EGTP-style entropy pooling, FiLM conditioning, soft-label classification, and MoE routing—under fixed model, data, and sampling settings. All methods converge to similar performance close to the theoretical bound (§5).

Finally, we argue that existing benchmarks such as ForeLen conflate predictor quality with task hardness, and we recommend a controlled evaluation protocol that reports relative-error metrics alongside absolute ones (§6).

## 2 Preliminaries and Related Work

Problem formulation. Let a prompt x be tokenized into $x _ { 1 : L }$ , and let $\mathbf { h } _ { 1 : L } ^ { ( l ) }$ denote the hidden states of layer l, for $l \in [ L _ { \mathrm { m o d e l } } ]$ , computed during prefill. Let Y be the number of new tokens decoded before the model emits an end-of-sequence (EOS) token, and let $\hat { y } = f \big ( x , \{ \mathbf { h } _ { 1 : L } ^ { ( l ) } \} _ { l } \big )$ be an estimate of Y that uses prefill-observable information only.

Metrics. Following prior work [Xie et al., 2026], we report the mean absolute error $\begin{array} { r } { ( \mathrm { M A E } = \frac { 1 } { N } \sum _ { i } | \hat { y } _ { i } - \bar { Y _ { i } } | ) } \end{array}$ and the pass-rate accuracy at tolerance $\tau .$

$$
\begin{array} { r } { \operatorname { a c c @ } \tau : = \operatorname* { P r } \bigl ( | \hat { y } - Y | < \tau \bigr ) , } \end{array}\tag{1}
$$

that is, the fraction of test prompts whose prediction error is below τ tokens. We report acc@100, acc@200, acc@500, and $x ( p 9 0 )$ , the smallest τ for which $\mathrm { a c c @ } \tau \geq 9 0 \%$

Feature extraction. Most embedding-based methods pool the layer-wise hidden states into a fixed-dimensional vector. Common strategies are mean pooling over layers and tokens, taking the last token of the last layer, or concatenating all layers [Shahout et al., 2024, Xie et al., 2026]. Consistent with prior observations on sentence embeddings [Jiang et al., 2023, Lei et al., 2024], we find that concatenating all layers gives the richest representation for this task.

EGTP and ForeLen. Xie et al. [2026] propose Entropy-Guided Token Pooling (EGTP), which computes a per-token entropy $H _ { i }$ from the nexttoken distribution and uses softmax(H) as attention weights to pool hidden states. They introduce ForeLen, a benchmark with three splits (long sequences, chain-of-thought reasoning, and RL sampling), and report MAE reductions of up to 29.16% over baselines. EGTP is an appealing featureengineering technique, but ForeLen evaluates diferent methods on diferent model–dataset combinations, so the reported gains cannot be separated from diferences in task dificulty.

Other predictors. Piotrowski et al. [2025] treat layer-wise hidden states as a graph and apply a GCN regressor. Zheng et al. [2023] predict the length with a lightweight side network inserted at inference time. Qiu et al. [2024] train a separate proxy model on the prompt text alone. Our work difers from all of these in targeting the fundamental limits that they share.

Iterative and online prediction. All of the above methods make a single prediction before decoding begins. A complementary line of work updates the estimate as partial output becomes available. Choi et al. [2025] propose ELIS, which feeds the prompt plus the already-generated tokens into a BGEbased predictor at fixed intervals (e.g. every 50 tokens) and shows that MAE decreases monotonically as more context is revealed. Wu et al. [2026] take a diferent route: instead of predicting the length, they estimate the remaining time adaptively with a skip-join MLFQ scheduler that re-prioritizes jobs at every iteration. Neither approach contradicts our prefill-side bounds: they operate on a richer information set (x and $y _ { < t } )$ and therefore face a smaller efective σ. We discuss their implications in §6.4.

## 3 Theoretical Analysis of Prediction Error

This section isolates the part of the prediction error that no predictor can remove. We first describe decoding as a stopped stochastic process and record exact hazard identities (§3.1). We then give a mechanistic model of how a response acquires its length, from which normality of the decode length follows as a theorem rather than as an assumption (§3.2). Finally we translate that law into bounds valid for every prefill-side predictor (§3.3) and identify what controls the per-prompt standard deviation (§3.5). To keep the exposition short, all proofs and derivations are deferred to Appendix A: the main text states each result, gives a one-line proof idea where useful, and points to the full argument.

## 3.1 Decoding Length as a Stopping Time

Fix a prompt x, a target model, and a sampling configuration. Decoding draws $y _ { 1 } , y _ { 2 } , . . .$ . from the model’s next-token distribution and stops at

$$
Y \ = \ \operatorname* { m i n } \{ t \geq 1 : y _ { t } \in S \} ,\tag{2}
$$

where $\boldsymbol { \mathcal { S } }$ is the set of stop tokens (EOS together with any configured stop strings). Hence Y is a stopping time of the sampling process, and its randomness is exactly the randomness of the sampled continuation. Its law is governed by the stopping probability (discrete hazard) and the survival function

$$
p _ { t } \ = \ \operatorname* { P r } \bigl ( Y = t \mid Y \geq t , x \bigr ) , \qquad S _ { t } \ = \ \operatorname* { P r } \bigl ( Y \geq t \mid x \bigr ) \ = \ \prod _ { s = 1 } ^ { t - 1 } ( 1 - p _ { s } ) ,\tag{3}
$$

with $S _ { 1 } = 1$ and $\operatorname* { P r } ( Y = t \mid x ) = S _ { t } p _ { t }$ . The hazard is itself random, because the sampled context decides how likely the model is to stop at step t.

Appendix A.1 records the exact moment identities implied by (3) and shows that a constant hazard would make Y geometric, with $\sigma / \mu \to 1$ and hence an irreducible relative MAE of about 74%; the measured $\sigma / \mu \approx 0 . 1 5$ (§4) rules that model out. Moreover, since the indicators $1 \{ Y ~ \geq ~ t \}$ that sum to $Y$ are nested rather than independent, normality cannot be inherited from a central limit theorem over tokens: it has to come from a coarser level, which is the subject of §3.2.

## 3.2 A Renewal Model and a Central Limit Theorem

A long response is not one uninterrupted decision: it is a concatenation of segments—sentences, list items, clauses, or reasoning steps—each completed before the model decides whether to continue.

Assumption 3.1 (Segment (renewal) structure). Conditionally on x, the response splits into M segments of lengths $L _ { 1 } , \dots , L _ { M }$ such that (i) given M and $x ,$ the $L _ { i }$ are i.i.d. with mean m, variance v, and finite third absolute central moment $\rho ; \mathbf { \Gamma } ( i i )$ M is a stopping count of the segment-level continuation process, independent of the $L _ { i } ,$ with $\mathbb { E } [ M ] = n$ and Var $( M ) = c ^ { 2 } n ^ { 2 } + c _ { 1 } n$ for constants $c , c _ { 1 } \geq 0$ . Then $\begin{array} { r } { Y = \sum _ { i = 1 } ^ { M } L _ { i } } \end{array}$

Theorem 3.2 (Normality of the decode length). Under Assumption $3 . 1 ,$ let $\mu = m n , \ \sigma ^ { 2 } = v n + m ^ { 2 } \mathrm { V a r } ( M )$ , and $\rho _ { M } \ = \ \mathbb { E } | M - n | ^ { 3 } / \mathrm { V a r } ( M ) ^ { 3 / 2 }$ Suppose moreover that M satisfies a Berry–Esseen bound at rate $n ^ { - 1 / 2 } , i . e .$ $\operatorname* { s u p } _ { u } \left| \operatorname* { P r } \bigl ( ( M - n ) / \sqrt { \operatorname { V a r } ( M ) } \leq u \bigr ) - \Phi ( u ) \right| \leq C _ { M } \rho _ { M } n ^ { - 1 / 2 }$ . Then there is a constant C such that

$$
\operatorname* { s u p } _ { u \in \mathbb { R } } \Bigl | \operatorname* { P r } \Bigl ( \frac { Y - \mu } { \sigma } \le u \Bigr | x \Bigr ) - \Phi ( u ) \Bigr | \le \frac { C } { \sqrt { n } } \Bigl ( \rho _ { M } + \frac { \sqrt { v } } { m c } \Bigr ) ,\tag{4}
$$

and hence $( Y - \mu ) / \sigma \Rightarrow { \mathcal { N } } ( 0 , 1 )$ as $n \to \infty$

Assumption 3.1 is the only statistical assumption of the paper, and it is deliberately coarse: it constrains a segment count rather than token-level behaviour. Appendix A.2 proves the theorem, discusses the regimes $c = 0$ (renewal count) and $c > 0$ (proportionally fluctuating segment count), and derives the consequences used below: the afine variance law $\sigma = \alpha \mu + b$ with $\alpha = \mathrm { C V } ( M )$ (Corollary $\mathrm { A . 3 } )$ ; the aggregation index $\sigma / \mu \geq c ,$ with $k : =$ $( \mu / \sigma ) ^ { 2 }$ efective stopping opportunities (Corollary $\mathrm { A . 4 } ) { \it : }$ ; and the Gaussian– geometric shape of the residual skew (Remark A.5). The stopping ramp of Fig. 2 and the measurements of $\ S 4$ check the assumption and its consequences empirically.

## 3.3 Lower Bounds on Prediction Error

Theorem 3.2 justifies the working model

$$
Y \mid x \sim \mathcal { N } \big ( \mu ( x ) , \sigma ^ { 2 } ( x ) \big ) ,\tag{5}
$$

up to the correction quantified in (4). We now derive what this law implies for any predictor ${ \hat { y } } = f ( x )$ that sees prefill-phase information only, and is therefore independent of the decoding noise given x.

Proposition 3.3 (Irreducible MAE). For $Y \sim { \mathcal { N } } ( \mu , \sigma ^ { 2 } )$

$$
\operatorname* { i n f } _ { \hat { y } \in \mathbb { R } } \mathbb { E } \big [ | Y - \hat { y } | \big ] \ = \ \mathbb { E } \big [ | Y - \mu | \big ] \ = \ \sigma \sqrt { \frac { 2 } { \pi } } \ \approx \ 0 . 7 9 7 9 \sigma ,\tag{6}
$$

and the infimum is attained uniquely at $\hat { y } = \mu$

Proposition 3.4 (Maximal pass-rate accuracy). For every tolerance $\tau > 0$ and every $\hat { y } \in \mathbb R$ ，

$$
\operatorname { a c c @ } \tau \ : = \ : \mathrm { P r } \big ( | Y - \hat { y } | < \tau \big ) \ : \leq \ : 2 \Phi \Big ( \frac { \tau } { \sigma } \Big ) - 1 ,\tag{7}
$$

where $\Phi$ is the standard normal CDF.

Proposition 3.3 follows by diferentiating $\hat { y } \mapsto \mathbb { E } | Y - \hat { y } |$ and Proposition 3.4 from a layer-cake (rearrangement) argument over intervals of length $2 \tau ;$ both extrema are attained uniquely at $\hat { y } = \mu$ , so the statements concern the best possible predictor rather than the mean predictor (Appendix A.3).

Propositions 3.3 and 3.4 state that, once $\sigma ( x )$ is fixed, no prefill-side predictor can do better: the bounds hold equally for MLPs, graph regressors, entropy-weighted pooling, and mixtures of experts. Concretely, acc@ $1 0 0 \geq$ $9 0 \%$ for a single prompt requires $\sigma ~ \le ~ 1 0 0 / \Phi ^ { - 1 } ( 0 . 9 5 ) ~ = ~ 6 1$ tokens and acc $\textcircled { \scriptsize { 2 } } 2 0 0 \geq 9 0 \%$ requires $\sigma \leq 1 2 2$ tokens; above these thresholds high accuracy is impossible, whatever the capacity or the amount of training data.

A test set aggregates prompts with diferent σ. Because $\sigma \mapsto 2 \Phi ( \tau / \sigma ) - 1$ is strongly nonlinear, averaging the per-prompt bounds and inserting σ¯ difer materially—57.9% versus 46.2% for the test set of §5.3; we therefore always report the average of the per-prompt bounds (the curvature of the map is computed in Appendix A.3).

Proposition 3.5 (Excess risk). For any predictor ${ \hat { y } } = f ( x )$

$$
{ \mathbb E } \big [ ( Y - \hat { y } ) ^ { 2 } \big ] = \underbrace { { \mathbb E } \big [ \sigma ^ { 2 } ( x ) \big ] } _ { s y s t e m \ e r r o r } + \underbrace { { \mathbb E } \big [ ( \mu ( x ) - \hat { y } ) ^ { 2 } \big ] } _ { m o d e l \ e r r o r } ,\tag{8}
$$

$$
\mathbb { E } \big [ | Y - \hat { y } | \big ] \leq \mathbb { E } \big [ \sigma ( x ) \sqrt { 2 / \pi } \big ] + \mathbb { E } \big [ | \mu ( x ) - \hat { y } | \big ] .\tag{9}
$$

Equation (8) is the law of total variance and (9) the triangle inequality applied prompt-wise (Appendix A.3). Thus the gap between a predictor and the bound is at most its distance to the oracle mean $\mu ( x )$ , and since training labels are single decodes rather than the means $\mu ( x )$ , the regression target carries noise of variance $\sigma ^ { 2 } ( x )$ , which slows learning at finite N—the residual gap observed in §5.3.

Corollary 3.6 (Relative-error metrics are the right currency). Under the afine law (18), for long responses $( \mu \gg b / \alpha )$

$$
\frac { \mathrm { M A E } _ { \operatorname* { m i n } } } { \mu } \longrightarrow \alpha \sqrt { \frac { 2 } { \pi } } \approx 0 . 7 9 8 \alpha , \qquad \mathrm { a c c } \ @ 1 0 0 _ { \operatorname* { m a x } } \longrightarrow 2 \Phi \left( \frac { 1 0 0 } { \alpha \mu } \right) - 1 .\tag{10}
$$

Absolute MAE therefore grows without bound with the length scale of the data, whereas the relative error tends to a stable, architecture-independent constant fixed by the generation process—the formal justification for the reporting protocol of §6.1.

## 3.4 Estimating σ from Finitely Many Decodes

In practice σ is unobserved: we see only K decodes per prompt. With $Y _ { 1 } , \dots , Y _ { K } \sim { \mathcal { N } } ( \mu , \sigma ^ { 2 } )$ i.i.d. the sample variance satisfies $( K - 1 ) S ^ { 2 } / \sigma ^ { 2 } \sim$ $\chi _ { K - 1 } ^ { 2 }$ , so requiring S to estimate σ within 10% at 95% confidence amounts to the two simultaneous inequalities

$$
\chi _ { K - 1 , 0 . 0 2 5 } ^ { 2 } \geq 0 . 8 1 ( K - 1 ) \quad \mathrm { a n d } \quad \chi _ { K - 1 , 0 . 9 7 5 } ^ { 2 } \leq 1 . 2 1 ( K - 1 ) ,\tag{11}
$$

which first hold at $K = 1 9 3 \ ( { \mathrm { A p p e n d i x ~ A . 5 } } )$ ; we use $K = 2 0 0$ decodes per prompt. All bounds reported below should therefore be read as accurate to about ±10% in σ, hence $\pm 1 0 \%$ in the bound itself.

## 3.5 What Controls σ?

Empirically, σ is strongly correlated with the mean length $\mu .$ Across six models evaluated on ShareGPT prompts, the relation is well described by the afine model derived in Appendix A.2 (Corollary A.3):

$$
\sigma = \alpha \mu + b .\tag{12}
$$

Table 1: Per-model $\sigma - \mu$ law on ShareGPT (300 prompts × 200 decodes). Lower α and b indicate more stable generation. By Corollary $\mathrm { A . 3 } , \ \alpha \ =$ $\mathrm { C V } ( M )$ and $b = \kappa _ { 1 } / ( 2 \alpha )$
<table><tr><td>Model</td><td>α</td><td>μ</td><td>b</td></tr><tr><td>Qwen3-4B-no-thinking</td><td>0.2029</td><td>814.25</td><td>5.56</td></tr><tr><td>Qwen3-4B-Instruct</td><td>0.1068</td><td>1443.28</td><td>31.25</td></tr><tr><td>DeepSeek-R1-distilled-thinking</td><td>0.1156</td><td>676.00</td><td>147.60</td></tr><tr><td>Llama3-8B</td><td>0.2042</td><td>555.23</td><td>-1.39</td></tr><tr><td>Qwen3-32B</td><td>0.1486</td><td>907.92</td><td>27.36</td></tr><tr><td> $\mathrm { Q w e n 3 - 3 0 B - A 3 B - I n s t r u c t }$ </td><td>0.1682</td><td>1479.78</td><td>12.18</td></tr></table>

Table 1 reports $( \alpha , b )$ for each model. The spread is large: Qwen3-4B-Instruct is comparatively stable $( \alpha = 0 . 1 0 6 8 )$ , whereas Qwen3-4B-no-thinking is not $( \alpha = 0 . 2 0 2 9 )$ . DeepSeek-R1-distilled-thinking combines a low slope $( \alpha = 0 . 1 1 5 6 )$ with an unusually large intercept $( b = 1 4 7 . 6 )$ , reflecting the exploratory randomness intrinsic to chain-of-thought generation even when the final answer is short. Since $\alpha \approx \mathrm { C V } ( M )$ , the fitted slopes say that a long answer to a ShareGPT prompt is worth roughly $1 / \alpha ^ { 2 } \approx 2 5 \mathrm { - } 9 0$ efectively independent stopping opportunities.

## 4 Empirical Validation

## 4.1 Is the Decode Length Normal?

We sample 300 prompts from ShareGPT and decode each K = 200 times with Qwen3-32B, using the model’s recommended sampling configuration (temperature 0.6, top-k 20, top-p 0.95) so that the length law reflects deployment. Two of the 300 prompts are deterministic (all 200 decodes return the same 8-token answer); there $\sigma = 0$ and every statement below holds trivially. The remaining 298 prompts yield 200 lengths each.

A single p-value is a poor summary at $K = 2 0 0$ , since tests reject minuscule deviations once the sample is large. We therefore report three complementary diagnostics: how often normality is rejected, the size and direction of the deviations (standardized skewness and excess kurtosis), and—most importantly—how accurately the Gaussian law reproduces the quantities that the bounds actually use.

Rejection rates. Shapiro–Wilk does not reject normality at $\alpha = 0 . 0 5$ for 79 of the 298 prompts (26.5%), while Kolmogorov–Smirnov does not reject for 226 (75.8%). The gap reflects how sensitive Shapiro–Wilk is to tiny deviations at $K = 2 0 0$

Efect sizes. The deviations are small and systematic rather than arbitrary: the median standardized skewness is +0.27 (70.8% of prompts are right-skewed) and the median excess kurtosis is +0.35; 58% of prompts have $| \mathrm { s k e w } | ~ < ~ 0 . 5$ and 69% have |kurt| < 1. Pooling all 59,600 standardized lengths $( Y - \hat { \mu } ) / \hat { \sigma }$ gives skewness 0.34, excess kurtosis 1.48, and a Kolmogorov–Smirnov distance of 0.019 from $\mathcal { N } ( 0 , 1 )$ . This is precisely the correction predicted by Remark A.5: a Gaussian bulk plus a small rightskewed sampling component. Solving (32) for the observed medians gives $r = 0 . 6 0$ (from the skewness) and $r = 0 . 5 7$ (from the kurtosis), i.e. a mutually consistent decomposition with about a quarter of the variance in sampling noise.

Accuracy where it matters. The per-prompt mean absolute deviation in units of σ is $\mathbb { E } | Y - \mu | / \sigma = 0 . 7 8 3$ (median over prompts), against the Gaussian prediction ${ \sqrt { 2 / \pi } } = 0 . 7 9 8 \colon$ the bound of Proposition 3.3 is accurate to 1.9% and slightly conservative. Likewise, the empirical probability mass within $\tau / \sigma \in \{ 0 . 6 2 5 , 1 , 1 . 2 8 , 1 . 6 4 5 \}$ exceeds the Gaussian value by 1.1, 1.7, 2.0 and 1.1 percentage points, respectively, so the bound of Proposition 3.4 errs on the safe side at the tolerances of interest. The normal model is thus not a convenient fiction: it is accurate to a few percent exactly where the bounds rely on it. The most visible exception is a genuine outlier prompt (CV 0.61, skewness 9.1), a runaway generation that no prefill-side predictor can anticipate; more generally the distribution of $\sigma$ is right-tailed (10% of prompts have $\sigma > 3 0 0 )$ , which is why we report both the mean $\sigma$ (162.3) and the median (133.9) whenever ceilings are quoted.

Fig. 1(a) shows the Q–Q plot of a representative long prompt and Fig. 1(b) the pooled standardized law. Fig. 2 displays the same prompt’s survival curve on a log scale, which is markedly not the straight line implied by a constant hazard, together with the stopping ramp that the CLT argument of §3.2 relies on.

## 4.2 µ–σ Correlation

Figure 3 plots per-prompt $( \mu , \sigma )$ pairs. The positive correlation is evident (Spearman $\rho \ : = \ : 0 . 7 8 )$ : longer expected outputs carry larger variance, as Corollary A.3 predicts. Outliers above the trend line are open-ended questions (e.g. “do you have reference text book. . . ”), while outliers below are tightly constrained instructions (e.g. code refactoring with prescribed variable names).

## 4.3 MAE versus σ

We train a simple 3-layer MLP on concatenated prefill hidden states and evaluate its per-prompt MAE on the same 300 ShareGPT prompts. Figure 4 plots the test MAE against σ. The fitted slope is 0.774, close to the theoretical slope $\sqrt { 2 / \pi } \approx 0 . 7 9 8$ of Proposition 3.3. In other words, the predictor is not the bottleneck: its error is the stochastic floor plus a small

![](images/c95fbfdc1e06a6efc9dc750957716b1b4822bf69d6f7dabfab8838178e66e8ef.jpg)

![](images/0d16821dcfbcba887ee6d8d4a99b698c0a096f8d1d9503cfd6c1b462af83c011.jpg)  
Figure 1: Normality of decode length (Qwen3-32B, ShareGPT). (a) Q–Q plot for a representative prompt $( \mu = 1 9 4 6$ 2 $\sigma = 2 1 7$ $K = 2 0 0 )$ ; the deviations are confined to the extreme tails. (b) Pooled standardized lengths of all 298 non-degenerate prompts (59,600 decodes) against $\mathcal { N } ( 0 , 1 )$ : a slightly peaked, slightly right-skewed Gaussian, as predicted by Remark A.5.

ofset.

## 5 Predictor Architectures and Results

## 5.1 Experimental Setup

Primary evaluation (ShareGPT self-test). Our main experiments use ShareGPT as both training and test source. Prompts with IDs 0–18000 (≈18K prompts) form the training and validation set, and prompts 18000– 18300 (exactly 300) form the test set. All experiments fix the target model to Qwen3-32B (no-thinking mode) with its recommended sampling parameters (temperature 0.6, top-k 20, top-p 0.95), so that generation matches deployment conditions. Prefill hidden states are exported once per prompt, and decode metadata is produced with the same model configuration. When evaluating other models (Llama3-8B, Qwen3-4B, DeepSeek-R1-distilled), we likewise adopt each model’s own default sampling configuration.

Why only 300 test prompts? Testing our framework requires an accurate per-prompt estimate of σ. As shown in §3.4, estimating σ to within 10% at 95% confidence requires $K \ge 1 9 3$ repeated decodes, and we use K = 200; with 200 decodes per test prompt, the test set must stay small for the evaluation to be afordable. The training set, in contrast, uses single-shot decode labels, which sufice for fitting a predictor.

Cross-dataset generalization. Beyond the ShareGPT self-test we ran cross-dataset experiments (Appendix D). For example, training on LMSYS-200K and testing on ShareGPT 18000–18300 yields MAE 182.40, and training on LeetCode and testing on HumanEval yields MAE 29.28 with acc@100 =

![](images/bc8dcd7782563135816b362331ac27eea7c92f2324810e93a91da6da81dcbd12.jpg)

![](images/981fde1ec0c746dd236634ba2d6e1fd38525c231eb3ae5cf5557328967655417.jpg)  
Figure 2: The stopping process of the prompt in Fig. 1(a). (a) Empirical survival function on a log scale; a constant hazard would produce the dashed straight line, whereas the observed curve steepens with t, i.e. the stopping probability increases with length. (b) Bin-wise stopping probability (bin width 5% of $\mu )$ : the hazard ramps from $\approx 0$ to $\approx 1$ over roughly 350 steps, which is the aggregation that Theorem 3.2 requires and Remark A.2 rules out.

96.28%. Predictor performance is thus highly sensitive to domain alignment between training and test, while the σ-driven ceiling remains the dominant constraint in every setting.

Data generation caveat. For resource reasons, prefill hidden states and length labels were produced by a PD-disaggregated pipeline: features were computed on prefill nodes and labels on separate decode workers. Batching, and hence low-level numerical behaviour, was not strictly controlled across the two stages; we return to this limitation in §6.

## 5.2 Architectures Evaluated

We evaluate the following predictor designs, all of which see prefill hidden states only:

• MLP-Concat: concatenate the last-token hidden vector of all layers (∼100k dims) and regress through a 3-layer MLP (hidden width 128). This is our simplest baseline.

• MLP-Mean: mean-pool hidden states across layers and tokens, then MLP regression.

• EGTP-style: compute a per-token entropy from the next-token logits, softmax-normalize it into attention weights, and weighted-pool the hidden states [Xie et al., 2026]; MLP regression on the pooled vector.

![](images/7d00fe42a996a47fb955e75c4be6e0352f6bcec2ede0289eed2f5893ce26b778.jpg)

Figure 3: Per-prompt mean µ versus standard deviation σ on ShareGPT (Qwen3-32B, 300 prompts × 200 decodes). The linear trend $\sigma = \alpha \mu + b$ of Corollary A.3 is visible, with outliers reflecting prompt-specific constraints.  
![](images/0914e9a33112ddd3275464a802ffc1b2bbb9f875910e273a1fc9c07af0cbba5d.jpg)  
Figure 4: Per-prompt test MAE versus per-prompt σ (Qwen3-32B, ShareGPT). The linear fit (blue) closely tracks the theoretical lower bound (red, $\mathrm { M A E } _ { \operatorname* { m i n } } = \sigma \sqrt { 2 / \pi } )$

• FiLM: treat the prefill token count as a conditioning signal and use a FiLM generator [Perez et al., 2018] to modulate the hidden representation before regression.

• Soft-Label: discretize lengths into 50 or 100 bins, train a classification head with Gaussian-smoothed targets, and map the expected bin back to a continuous length.

• MoE-Hard: train 4 expert regressors on length-based splits, with a hard gate classifier [Shazeer et al., 2017] selecting one expert.

• MoE-Soft: the same experts, with a soft gate that averages their predictions.

• MoE-E2E: end-to-end joint training of gate and experts with an auxiliary routing loss.

Table 2: Benchmark comparison under fixed model (Qwen3-32B), dataset (ShareGPT 18000–18300), and sampling settings. All methods use prefill hidden states only. The Gaussian bound is the ceiling of Propositions 3.3 and 3.4, averaged over the 300 test prompts, i.e. $\begin{array} { r } { { \frac { 1 } { N } } \sum _ { i } \sigma _ { i } \sqrt { 2 / \pi } } \end{array}$ and $\begin{array} { r } { \frac { 1 } { N } \sum _ { i } [ 2 \Phi ( \tau / \sigma _ { i } ) - 1 ] } \end{array}$ ; no prefill-side predictor can exceed it.
<table><tr><td>Method</td><td></td><td>MAE ↓ Acc@100 ↑ Acc@200 ↑ Acc@500 ↑ x(p90) ↓</td><td></td><td></td><td></td></tr><tr><td>MLP-Mean</td><td>181.21</td><td>45.07%</td><td>68.91%</td><td>93.51%</td><td>402.78</td></tr><tr><td>MLP-Concat</td><td>169.31</td><td>48.91%</td><td>71.11%</td><td>94.35%</td><td>391.09</td></tr><tr><td>MLP-Concat +</td><td>168.13</td><td>48.63%</td><td>71.65%</td><td>94.45%</td><td>392.13</td></tr><tr><td>LayerNorm EGTP-style</td><td>466.79</td><td>14.58%</td><td>25.35%</td><td>61.86%</td><td>891.48</td></tr><tr><td>FiLM</td><td>192.69</td><td>38.30%</td><td></td><td></td><td></td></tr><tr><td>Soft-Label (50</td><td>183.24</td><td>45.05%</td><td></td><td></td><td></td></tr><tr><td>bins) Soft-Label (100</td><td>180.15</td><td>46.10%</td><td>69.83%</td><td>93.42%</td><td>413.24</td></tr><tr><td>bins) MoE-Hard</td><td>177.91</td><td>46.54%</td><td>69.77%</td><td>93.35%</td><td>418.52</td></tr><tr><td>MoE-Soft</td><td>170.70</td><td>47.86%</td><td>71.00%</td><td>94.19%</td><td>393.70</td></tr><tr><td>MoE-E2E</td><td>171.39</td><td>47.33%</td><td>71.01%</td><td>94.30%</td><td></td></tr><tr><td>Theoretical bound</td><td>129.5</td><td>57.9%</td><td>79.7%</td><td>96.5%</td><td></td></tr></table>

## 5.3 Results

Table 2 reports MAE and pass-rate metrics. Several findings stand out.

Finding 1: simple is competitive. MLP-Concat matches or beats FiLM, Soft-Label, and all MoE variants; the gap to the best MoE variant (MoE-Soft, MAE 170.70) is below 1% relative. Notably, EGTP-style entropy-guided pooling [Xie et al., 2026] degrades to MAE 466.79 on this split, far behind even the simplest MLP-Mean baseline, which suggests that entropy-weighted aggregation can inject noise when the token-level entropy distribution does not align with the length-predictive signal.

Finding 2: the ceiling is tight and the residual gap is model error. The mean per-prompt σ is 162.3 (median 133.9; the σ distribution is right-tailed, with 10% of prompts above $\sigma = 3 0 0 )$ , so the Gaussian floor is 129.5 tokens. Our best predictor achieves 168.1, an excess of 38.6 tokens. By Proposition 3.5 this excess is an upper bound on the model error $\mathbb { E } | \mu ( x ) - \hat { y } |$ i.e. at most 30% of the system error $\mathbb { E } [ \sigma ( x ) ] { \sqrt { 2 / \pi } }$ . Two efects account for it: training labels are single decodes and therefore carry noise of variance $\sigma ^ { 2 } ( x )$ (Proposition 3.5), and the test prompts are not perfectly aligned with the training distribution. Neither is architectural, which is consistent with architectures that difer by orders of magnitude in complexity landing within a few percent of each other.

Table 3: Gaussian Acc@τ ceiling $2 \Phi ( \tau / \sigma ) - 1$ (Proposition 3.4) as a function of the per-prompt standard deviation. A 90% Acc@100 ceiling requires $\sigma \leq$ 61 tokens.
<table><tr><td>σ (tokens)</td><td>60</td><td>100</td><td>150</td><td>200</td><td>250</td><td>300</td><td>400</td></tr><tr><td>Acc@100 ceiling</td><td>90.4%</td><td>68.3%</td><td>49.5%</td><td>38.3%</td><td>31.1%</td><td>26.1%</td><td>19.7%</td></tr><tr><td>Acc@200 ceiling</td><td>99.9%</td><td>95.4%</td><td>81.8%</td><td>68.3%</td><td>57.6%</td><td>49.5%</td><td>38.3%</td></tr><tr><td>Acc@500 ceiling</td><td> $> 9 9 . 9 \% > 9 9 . 9 \%$ </td><td></td><td>99.9%</td><td>98.8%</td><td>95.4%</td><td>90.4%</td><td>78.9%</td></tr></table>

Finding 3: 90% Acc@100 is unattainable, and the shortfall is quantifiable. The per-prompt-averaged Gaussian ceiling is 57.9% at $\tau = 1 0 0 ( 7 9 . 7 \%$ at $\tau = 2 0 0 , 9 6 . 5 \%$ at $\tau = 5 0 0 )$ . Measured Acc@100 tops out at 48.9%, i.e. 8.9 points (or 15% relatively) below the ceiling, and measured Acc@500 reaches 94.5% against a ceiling of 96.5%. Reporting accuracy relative to the ceiling is therefore far more informative than the bare percentage (§6.1). Inserting the mean $\bar { \sigma } \ = \ 1 6 2 . 3$ into (7) instead of averaging perprompt bounds would give 46.2% and understate the ceiling by 12 points, which is why we always average the per-prompt bounds.

The same arithmetic explains the other models. Llama3-8B $( \bar { \sigma } = 1 1 2 )$ and Qwen3-30B-A3B $( \bar { \sigma } = 2 6 1 )$ face much easier and much harder tasks respectively; Table 3 tabulates the resulting ceiling once, as a function of σ, so that any predictor can be placed against it without re-deriving the bound.

## 6 Discussion

## 6.1 Toward Fairer Benchmarks

The ForeLen benchmark [Xie et al., 2026] evaluates diferent methods on diferent model–dataset combinations (e.g. Qwen2.5-7B on long sequences, Llama3.2-3B on RL sampling). Because each combination has its own σ, the reported MAE diferences confound predictor quality with task hardness. We recommend that future benchmarks

1. fix the target LLM, the dataset, and the sampling parameters when comparing predictors;

2. report the per-prompt σ (or at least its average and median over the test set), so that absolute MAE values can be read against the bound of Propositions 3.3 and 3.4;

3. adopt relative-error metrics $( \mathrm { M A E } / \mu ,$ or pass rates expressed as a fraction of the ceiling $2 \Phi ( \tau / \sigma ) - 1 )$ alongside absolute MAE, since Corollary 3.6 shows that only the former is stable under changes of the length scale.

## 6.2 Runtime Compute Inconsistency

Our prefill features and decode labels were produced by separate pipeline stages on diferent workers, without strict control of batch size, CUDA kernel versions, or deterministic mode. The hidden states used as features and those that produced the labels may therefore difer at the floating-point level. We believe this efect is small relative to the σ-driven system error, but it remains uncontrolled. Future work should generate features and labels in the same forward pass, or at least fix all runtime determinism flags.

## 6.3 PD-Disaggregated Generation

To obtain labels at scale we used a PD-disaggregated serving setup [Zhong et al., 2024] in which prefill nodes export hidden states and KV caches while decode nodes generate completions. The separation introduces two potential biases: numerical diferences between prefill and decode workers (as above), and possibly diferent batching strategies on the two sides. Since our training labels are single decodes rather than means of 200 decodes, label noise is amplified further. These are practical constraints of large-scale data collection; they should be acknowledged and, ideally, controlled in future studies.

## 6.4 Iterative Prediction as Uncertainty Reduction

Our analysis assumes that the predictor sees prefill-phase information only. This is the hardest regime, because $\operatorname { V a r } ( Y \mid x )$ is then as large as it can be. Once decoding begins, every generated token $y _ { t }$ carries information about the remaining length, and the posterior variance can only shrink: $\operatorname { V a r } ( Y \mid x , y _ { < t } ) \leq \operatorname { V a r } ( Y \mid x )$ , strictly so for most prompts. Choi et al. [2025] exploit this by re-running a BGE-based predictor at fixed intervals during decoding and report that MAE decreases monotonically as more tokens are observed.

Theoretically, iterative prediction is the most efective way to reduce uncertainty without changing the generation process. Prefill-side feature engineering can only shrink the model error of Proposition 3.5, whereas iterative prediction shrinks the system error by reducing the conditional variance of the target. In the limit where the predictor is invoked after every token, the remaining length collapses to a near-deterministic countdown and the MAE bound tends to zero.

The price is runtime cost: each prediction requires an encoder forward pass, adding latency and memory pressure to the serving system. For scheduling applications that consume prefill-side estimates to batch requests, this overhead is often acceptable only for coarse-grained updates (e.g. every 50– 100 tokens). We therefore view prefill-only bounds and iterative refinement as complementary: the former set the baseline achievable without decodeside overhead, the latter quantify the uncertainty reduction available to systems that are willing to pay for it.

## 7 Conclusion

We have shown that decoding-length prediction from prefill hidden states is fundamentally constrained by the stochasticity of autoregressive generation. Treating decoding as a stopping time and modelling the growth of a response as a renewal process yields per-prompt normality with an explicit error term, the afine variance law $\sigma = \alpha \mu + b $ and lower bounds on MAE and pass-rate accuracy that no prefill-side predictor can beat. Empirically these bounds are tight: a simple MLP already lands within 30% of the noise floor, while more elaborate designs (EGTP, FiLM, Soft-Label, MoE) yield marginal or negative returns. The implication is that efort should shift from ever more complex predictors toward (i) controlling the variance of the generation process itself—through lower-temperature sampling, constrained decoding, or prompt design—and (ii) benchmarking protocols that fix the model, data, and sampling parameters and report relative errors against the ceiling.

Code availability. All code for data generation, model training, and evaluation, together with the prefill feature extraction utilities described above, is available at https://github.com/ShwStone/DecodingPredict.

## References

Seungbeom Choi, Jeonghoe Goo, Eunjoo Jeon, Mingyu Yang, and Minsung Jang. ELIS: Eficient LLM iterative scheduling system with response length predictor. arXiv preprint arXiv:2505.09142, 2025.

Ting Jiang, Shaohan Huang, Zhongzhi Luan, Deqing Wang, and Fuzhen Zhuang. Scaling sentence embeddings with large language models. arXiv preprint arXiv:2307.16645, 2023.

Woosuk Kwon, Zhuohan Li, Zhang Siyuan, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Eficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th ACM Symposium on Operating Systems Principles, pages 611–626, 2023.

Yibin Lei, Di Wu, Tianyi Zhou, Tao Shen, Yu Cao, Chongyang Tao, and Andrew Yates. Meta-task prompting elicits embeddings from large language models. arXiv preprint arXiv:2402.18458, 2024.

Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. FiLM: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.

Grzegorz Piotrowski, Mateusz Bystroński, Mikolaj Holysz, Jakub Binkowski, Grzegorz Chodak, and Tomasz Kajdanowicz. Graph-based approach to predicting remaining output length in large language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics: Student Research Workshop, 2025.

Haoran Qiu, Weichao Mao, Archit Patke, Shengkun Cui, Saurabh Jha, Chen Wang, Hubertus Franke, Zbigniew T Kalbarczyk, Tamer Basar, and Ravishankar K Iyer. Eficient interactive LLM serving with proxy model-based sequence length prediction. arXiv preprint arXiv:2404.08509, 2024.

Rana Shahout, Eran Malach, Chunwei Liu, Weifan Jiang, Minlan Yu, and Michael Mitzenmacher. Don’t stop me now: Embedding based scheduling for LLMs. arXiv preprint arXiv:2410.01035, 2024.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geofrey Hinton, and Jef Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.

Bingyang Wu, Yinmin Zhong, Zili Zhang, Gang Huang, Xuanzhe Liu, and Xin Jin. Fast distributed inference serving for large language models. In Proc. USENIX Symp. on Networked Systems Design and Implementation (NSDI), 2026.

Huanyi Xie, Yubin Chen, Liangyu Wang, Lijie Hu, and Di Wang. Predicting LLM output length via entropy-guided representations. arXiv preprint arXiv:2602.11812, 2026.

Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. Orca: A distributed serving system for transformer-based generative models. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22), pages 521–538, 2022.

Zangwei Zheng, Xiaozhe Ren, Fuzhao Xue, Yang Luo, Xin Jiang, and Yang You. Response length perception and sequence scheduling: An LLMempowered LLM inference pipeline. Advances in Neural Information Processing Systems, 36:65517–65530, 2023.

Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. Distserve: Disaggregating prefill and decoding for goodput-optimized large language model serving. arXiv preprint arXiv:2401.09670, 2024.

## A Detailed Derivations

## A.1 Hazard Identities and the Geometric Baseline

Lemma A.1 (Hazard identities). For any hazard sequence $\{ p _ { t } \} _ { t \ge 1 }$ with survival function $\begin{array} { r } { S _ { t } = \prod _ { s = 1 } ^ { t - 1 } ( 1 - p _ { s } ) } \end{array}$ as in (3),

$$
\begin{array} { c } { { \mathbb { E } [ Y ] = \displaystyle \sum _ { t \ge 1 } S _ { t } , \qquad \mathbb { E } [ Y ^ { 2 } ] = \displaystyle \sum _ { t \ge 1 } ( 2 t - 1 ) S _ { t } , } } \\ { { \mathrm { V a r } ( Y ) = \displaystyle \sum _ { t \ge 1 } ( 2 t - 1 ) S _ { t } - \left( \displaystyle \sum _ { t \ge 1 } S _ { t } \right) ^ { 2 } . } } \end{array}\tag{13}
$$

Proof. For an integer-valued $Y ~ \ge ~ 1$ the layer-cake identities read $Y =$ $\textstyle \sum _ { t \geq 1 } \mathbf { 1 } \{ Y \geq t \}$ and, using $\begin{array} { r } { \sum _ { t = 1 } ^ { n } ( 2 t - 1 ) = n ^ { 2 } } \end{array}$ 2

$$
Y ^ { 2 } = \sum _ { t \geq 1 } ( 2 t - 1 ) \mathbf { 1 } \{ Y \geq t \} .\tag{14}
$$

Taking expectations and using $\mathbb { E } [ { \bf 1 } \{ Y \ge t \} ] = \mathrm { P r } ( Y \ge t ) = S _ { t }$ gives (13). □

Remark A.2 (A constant hazard gives an exponential law, not a normal one). If $p _ { t } \equiv p$ then $Y \sim \operatorname { G e o m } ( p )$ and the coeficient of variation is $\sqrt { 1 - p }  1$ , so the irreducible relative error is E|Y − $\mu | / \mu  2 / e \approx 0 . 7 4$ regardless of the output length. Since the indicators ${ \bf 1 } \{ Y \ge t \}$ that sum to $Y$ are nested rather than independent, a first-passage time does not inherit the CLT of a sum: normality requires many nearly independent contributions, which enter at a coarser level than the token.

Justification. For $p _ { t } \equiv p$ we have $S _ { t } = ( 1 - p ) ^ { t - 1 }$ , so (13) gives $\mathbb { E } [ Y ] = 1 / p$ and $\operatorname { V a r } ( Y ) = ( 1 - p ) / p ^ { 2 }$ . For small $p$ the rescaled variable $p Y$ converges to a unit-rate exponential, whence

$$
{ \frac { \mathbb { E } | Y - \mathbb { E } Y | } { \mathbb { E } Y } } \longrightarrow \mathbb { E } | E - 1 | = { \frac { 2 } { e } } \approx 0 . 7 3 6 , \qquad E \sim \mathrm { E x p } ( 1 ) ,\tag{15}
$$

where the constant follows from $\begin{array} { r } { \int _ { 0 } ^ { 1 } ( 1 - x ) e ^ { - x } d x + \int _ { 1 } ^ { \infty } ( x - 1 ) e ^ { - x } d x = 2 e ^ { - 1 } } \end{array}$ An $O ( 1 )$ relative error is thus unavoidable under a constant hazard, no matter how much is known about $x ,$ which is why the measured $\sigma / \mu \approx 0 . 1 5$ excludes that model. □

## A.2 Proof of Theorem 3.2 and Its Consequences

Wald identities. Write $\begin{array} { r } { Y = \sum _ { i = 1 } ^ { M } L _ { i } } \end{array}$ with M independent of the i.i.d. segments. Then $\mathbb { E } [ Y ] = m \mathbb { E } [ M ] = { \dot { m } } { \dot { n } }$ , and by the law of total variance and Wald’s identity $\mathbb { E } [ \operatorname { V a r } ( Y \mid M ) ] = v \mathbb { E } [ M ]$ , so

$$
\mathrm { V a r } ( Y ) = v n + m ^ { 2 } \mathrm { V a r } ( M ) = v n + m ^ { 2 } \big ( c ^ { 2 } n ^ { 2 } + c _ { 1 } n \big ) \ = \ \kappa _ { 2 } \mu ^ { 2 } + \kappa _ { 1 } \mu ,\tag{16}
$$

with $\kappa _ { 2 } = c ^ { 2 }$ and $\kappa _ { 1 } = v / m + m c _ { 1 }$ . Taking square roots and expanding,

$$
\sigma = \sqrt { \kappa _ { 2 } \mu ^ { 2 } + \kappa _ { 1 } \mu } = \alpha \mu \sqrt { 1 + \frac { \kappa _ { 1 } } { \kappa _ { 2 } \mu } } = \alpha \mu + \frac { \kappa _ { 1 } } { 2 \alpha } + { \cal O } ( \mu ^ { - 1 } ) ,\tag{17}
$$

This is the afine variance law:

Corollary A.3 (Afine variance law). Under Assumption 3.1, with $\kappa _ { 2 } = c ^ { 2 }$ and $\kappa _ { 1 } = v / m + m c _ { 1 }$ ，

$$
\sigma ^ { 2 } = \kappa _ { 2 } \mu ^ { 2 } + \kappa _ { 1 } \mu , ~ h e n c e ~ \sigma = \alpha \mu + b + O \big ( \mu ^ { - 1 } \big ) ,\tag{18}
$$

where $\alpha = \sqrt { \kappa _ { 2 } } = c$ and $b = \kappa _ { 1 } / ( 2 \alpha )$ . The slope is therefore the coeficient of variation of the segment count and the intercept collects per-segment noise discounted by $1 / ( 2 \alpha )$ —properties of the generation process rather than of the predictor.

Corollary A.4 (Aggregation index). Under Assumption $3 . 1 ,$

$$
{ \frac { \sigma } { \mu } } = \sqrt { c ^ { 2 } + { \frac { v / m ^ { 2 } + c _ { 1 } } { n } } } \ \geq \ c ,\tag{19}
$$

with equality as $n  \infty$ whenever $c > 0$ . Hence $k : = ( \mu / \sigma ) ^ { 2 }$ measures how many efectively independent stopping opportunities a prompt ofers: models with a single deciding moment have $k = O ( 1 )$ , aggregated models have $k \gg 1$ For the prompts of $\ S 4$ the median of $\sigma / \mu$ is 0.145, i.e. $k \approx 4 7$

The two regimes. The variance form $\operatorname { V a r } ( M ) = c ^ { 2 } n ^ { 2 } + c _ { 1 } n$ of Assumption 3.1 covers both cases of interest. With $c = 0$ the segment count behaves like a renewal count, $\mathrm { V a r } ( M ) \propto n ,$ and the sampling and content terms of the decomposition below are of the same order. With $c > 0$ the number of segments fluctuates proportionally across decodes of the same prompt—the regime implied by a fluctuating content path—and $c = \operatorname { C V } ( M \mid x )$ , which is the regime of the measurements in $\ S 4$

The two-scale decomposition. With $\begin{array} { r } { A = \sum _ { i < M } ( L _ { i } - m ) } \end{array}$ and $B =$ $m ( M { - } n )$ we have $Y - \mu = A + B$ exactly, with $\mathbb { E } [ A ] = \mathbb { \bar { E } } [ B ] = 0 , \operatorname { V a r } ( A ) = v n$ and $\mathrm { V a r } ( B ) = m ^ { 2 } \mathrm { V a r } ( M )$ , so $\sigma ^ { 2 } = \operatorname { V a r } ( A ) + \operatorname { V a r } ( B )$ . Although A and B are not independent (both depend on $M )$ , they are conditionally independent given M, so by the tower property

$$
\varphi _ { ( Y - \mu ) / \sigma } ( t ) = \operatorname { \mathbb { E } } \Bigl [ e ^ { i t B / \sigma } \varphi _ { A / \sigma } \bigl ( t \mid M \bigr ) \Bigr ] .\tag{20}
$$

Given $M = k ,$ the variable A is a sum of k centred i.i.d. terms, so a secondorder expansion (or the Berry–Esseen bound for sums of length k) gives

$$
\Bigl \vert \varphi _ { A / \sigma } \bigl ( t \mid M = k \bigr ) - 1 \Bigr \vert \le \frac { t ^ { 2 } } { 2 } \frac { \mathrm { V a r } ( A \mid M = k ) } { \sigma ^ { 2 } } + O \Bigl ( \frac { \rho \vert t \vert ^ { 3 } k ^ { 3 / 2 } } { m ^ { 3 } \sigma ^ { 3 } } \Bigr ) = O \Bigl ( \frac { t ^ { 2 } } { n } \Bigr ) ,\tag{21}
$$

uniformly for $k = O ( n )$ , because Var $\cdot ( A \mid M = k ) = v k$ and $\sigma ^ { 2 } \geq m ^ { 2 } c ^ { 2 } n ^ { 2 }$ Since $| e ^ { i t B / \sigma } | = 1$

$$
\Bigl | \varphi _ { ( Y - \mu ) / \sigma } ( t ) - \varphi _ { B / \sigma } ( t ) \Bigr | \le { \mathbb E } \Bigl [ \bigl | \varphi _ { A / \sigma } ( t \mid M ) - 1 \bigr | \Bigr ] = O \Bigl ( \frac { t ^ { 2 } } { n } \Bigr ) ,\tag{22}
$$

i.e. the sampling term perturbs the characteristic function only at order $1 / n$ For the content term, the assumed Berry–Esseen bound for M gives, with $\sigma _ { B } = m { \sqrt { \operatorname { V a r } ( M ) } }$

$$
\left| { \varphi _ { B / \sigma } ( t ) - e ^ { - { t ^ { 2 } / 2 } } } \right| \le \left| { \varphi _ { B / \sigma _ { B } } ( t ) - e ^ { - { t ^ { 2 } / 2 } } } \right| + \frac { t ^ { 2 } } { 2 } \Big | 1 - \frac { \sigma _ { B } ^ { 2 } } { \sigma ^ { 2 } } \Big | \le C \Big ( \rho _ { M } \frac { | t | ^ { 3 } } { \sqrt { n } } + \frac { | t | ^ { 2 } } { n } \Big ) ,\tag{23}
$$

where $1 - \sigma _ { B } ^ { 2 } / \sigma ^ { 2 } = v n / \sigma ^ { 2 } = { \cal O } ( n ^ { - 1 } )$ . Combining the three displays, for every fixed $t ,$

$$
\left| \varphi _ { ( Y - \mu ) / \sigma } ( t ) - e ^ { - t ^ { 2 } / 2 } \right| \leq C ^ { \prime } \Bigl ( \rho _ { M } \frac { | t | ^ { 3 } } { \sqrt { n } } + \frac { | t | ^ { 2 } } { n } \Bigr ) \longrightarrow 0 ,\tag{24}
$$

and $\mathrm { L e v y ` s }$ continuity theorem gives $( Y - \mu ) / \sigma \Rightarrow { \mathcal { N } } ( 0 , 1 )$ . Esseen’s smoothing lemma upgrades pointwise characteristic-function convergence to the uniform CDF bound (4). □

What the theorem does and does not assume. The argument does not derive normality from scratch. It reduces normality of a token-level stopping time to normality of a segment count $M { \mathrm { - a } }$ much coarser object, with only tens of efective units—and it prices the reduction: the token-level (sampling) contribution enters the variance only at relative order $v n / \sigma ^ { 2 } =$ $O ( n ^ { - 1 } )$ , so its shape is irrelevant to leading order. In the renewal regime $( c = 0 , \operatorname { V a r } ( M ) \propto n )$ the two terms are of the same order and Anscombe’s theorem for randomly stopped sums plays the role of the content CLT; the conclusion is unchanged whenever $M / n  1$ in probability. The empirical checks of $\ S 4$ test the outcome rather than the assumption: the afine law (16), the ratio $\mathbb { E } | Y - \mu | / \sigma$ , and the skewness/kurtosis pair all match the model’s predictions.

## A.3 Proofs and Auxiliary Computations for Section 3.3

Proof of Proposition 3.3. Let $Z \sim { \mathcal { N } } ( 0 , 1 )$ and $c = ( \hat { y } - \mu ) / \sigma$ , so that $\mathbb { E } | Y - \hat { y } | = \sigma \psi ( c )$ with $\psi ( c ) = \mathbb { E } | Z - c |$ . Diferentiating under the expectation,

$$
\psi ^ { \prime } ( c ) = \mathbb { E } { \bigl [ } - \operatorname { s g n } ( Z - c ) { \bigr ] } = \operatorname* { P r } ( Z < c ) - \operatorname* { P r } ( Z > c ) = 2 \Phi ( c ) - 1 ,\tag{25}
$$

which is negative for $c < 0$ , zero at $c = 0 .$ , and positive for $c > 0 ;$ ; hence $c = 0$ is the unique minimizer. The value follows from $u = z ^ { 2 } / 2$

$$
\mathbb { E } | Z | = 2 \int _ { 0 } ^ { \infty } z \frac { e ^ { - z ^ { 2 } / 2 } } { \sqrt { 2 \pi } } d z = 2 \int _ { 0 } ^ { \infty } \frac { e ^ { - u } } { \sqrt { 2 \pi } } d u = \sqrt { \frac { 2 } { \pi } } .\tag{26}
$$

For $Y \sim { \mathcal { N } } ( \mu , \sigma ^ { 2 } )$ this gives $\mathrm { M A E } _ { \operatorname* { m i n } } = \sigma \sqrt { 2 / \pi } \approx 0 . 7 9 7 9 \sigma$

Proof of Proposition 3.4. The density $f ( y ) = \sigma ^ { - 1 } \varphi ( ( y - \mu ) / \sigma )$ is symmetric about $\mu$ and non-increasing in $| y - \mu |$ , so its superlevel sets $A _ { s } = \{ y$ : $f ( y ) \geq s \}$ are intervals centred at $\mu$ with lengths $\ell ( s )$ . By the layer-cake representation and the fact that the intersection of two intervals of lengths $2 \tau$ and $\ell ( s )$ has length at most min $\{ 2 \tau , \ell ( s ) \}$ ,

$$
\operatorname* { P r } ( Y \in I ) = \int _ { 0 } ^ { \infty } \left| I \cap A _ { s } \right| d s \leq \int _ { 0 } ^ { \infty } \operatorname* { m i n } \{ 2 \tau , \ell ( s ) \} d s = \operatorname* { P r } \left( | \mu - Y | < \tau \right)\tag{27}
$$

for every interval I of length $2 \tau ;$ the equality case is $I = [ \mu - \tau , \mu + \tau ]$ , i.e. $\hat { y } = \mu$ . Taking $I = [ \hat { y } - \tau , \hat { y } + \tau ]$ gives $( 7 )$ □

Proof of Proposition 3.5. For a predictor ${ \hat { y } } = f ( x )$ that is independent of the decoding noise given $x ,$ the tower property gives

$$
\begin{array} { r l } & { \mathbb { E } \big [ ( Y - \hat { y } ) ^ { 2 } \big ] = \mathbb { E } \Big [ \mathbb { E } \big [ ( Y - \mu ( x ) ) ^ { 2 } \mid x \big ] \Big ] + \mathbb { E } \big [ ( \mu ( x ) - \hat { y } ) ^ { 2 } \big ] } \\ & { \qquad = \mathbb { E } \big [ \sigma ^ { 2 } ( x ) \big ] + \mathbb { E } \big [ ( \mu ( x ) - \hat { y } ) ^ { 2 } \big ] , } \end{array}\tag{28}
$$

which is (8) and identifies the two terms as the irreducible (system) and the reducible (model) error. For the absolute loss, the triangle inequality together with Proposition 3.3 applied prompt-wise, $\mathbb { E } [ | Y - \hat { y } | \mid x ] \ge \sigma ( x ) \sqrt { 2 / \pi }$ gives

$$
\mathbb { E } \big [ | Y - \hat { y } | \big ] \leq \mathbb { E } \big [ \sigma ( x ) \sqrt { 2 / \pi } \big ] + \mathbb { E } \big [ | \mu ( x ) - \hat { y } | \big ] ,\tag{29}
$$

i.e. (9): the excess over the bound is at most the $L _ { 1 }$ distance from the predictor to the oracle mean. □

Proof of Corollary 3.6. Under (18), $\sigma / \mu = \alpha \sqrt { 1 + \kappa _ { 1 } / ( \kappa _ { 2 } \mu ) } \to \alpha _ { \mathrm { i } }$ so Proposition 3.3 gives $\mathrm { M A E } _ { \operatorname* { m i n } } / \mu \to \alpha \sqrt { 2 / \pi }$ and Proposition 3.4 gives $\csc \ @ \tau _ { \mathrm { m a x } } = 2 \Phi ( \tau / \sigma ) - 1 \to 2 \Phi ( \tau / ( \alpha \mu ) ) - 1$ . The bound therefore depends on $\mu$ only through the ratio $\tau / \mu .$ which is what makes relative-error metrics scale-free. □

Test-set aggregation. For $g ( \sigma ) = 2 \Phi ( \tau / \sigma ) - 1$

$$
g ^ { \prime } ( \sigma ) = - { \frac { 2 \tau \varphi ( \tau / \sigma ) } { \sigma ^ { 2 } } } , \qquad g ^ { \prime \prime } ( \sigma ) = { \frac { 2 \tau \varphi ( \tau / \sigma ) } { \sigma ^ { 3 } } } \Big ( 2 - { \frac { \tau ^ { 2 } } { \sigma ^ { 2 } } } \Big ) ,\tag{30}
$$

so $g$ is decreasing everywhere, convex for $\sigma > \tau / \sqrt { 2 }$ and concave below. Prompts with $\sigma \ll \tau$ saturate g near 1, so a test-set average is dominated by the convex region and typically exceeds $g ( \bar { \sigma } )$ : for the test set of $\ S 5 . 3$ the per-prompt average is 57.9% against 46.2% for $g ( \bar { \sigma } )$ . This is why all ceilings in this paper average the per-prompt bounds.

## A.4 The Shape of the Deviations

Remark A.5 (Variance budget and the shape of the deviations). Write the two terms of the decomposition

$$
Y - \mu = \underbrace { \sum _ { i = 1 } ^ { M } \bigl ( L _ { i } - m \bigr ) } _ { s a m p l i n g \ n o i s e } + \underbrace { m \bigl ( M - n \bigr ) } _ { c o n t e n t \ n o i s e }\tag{31}
$$

as $A$ and B, and let $r = \sigma _ { \mathrm { s a m p } } / \sigma _ { \mathrm { c o n t } }$ . If the path-wise hazard is locally flat around the crossing point, A behaves like a centred geometric variable, so the standardized law is a Gaussian–geometric convolution whose cumulants give

$$
\operatorname { s k e w } ( Y ) = { \frac { 2 r ^ { 3 } } { ( 1 + r ^ { 2 } ) ^ { 3 / 2 } } } , \qquad \operatorname { k u r t } ( Y ) = { \frac { 6 r ^ { 4 } } { ( 1 + r ^ { 2 } ) ^ { 2 } } } ,\tag{32}
$$

with kurt the excess kurtosis; $r  0$ recovers the Gaussian. The empirical medians (skewness 0.27, excess kurtosis $0 . 3 5 ; ~ \ S 4 )$ invert to $r ~ = ~ 0 . 6 0$ and $r = 0 . 5 7 -$ mutually consistent, and implying that at the segment counts $r e \mathrm { - }$ alized here $( n \approx 4 5 ,$ , far from the asymptotic regime of Theorem 3.2) about a quarter of the per-prompt variance is sampling noise. In this precise sense the decode length $^ { \circ } \mathit { i } _ { S } \mathit { ^ { \prime } } _ { } ^ { \mathrm { ~ ~ } }$ Gaussian: the location and scale of the bound are Gaus-$s i a n ,$ and the residual right skew is a small, quantifiable correction rather than a modelling failure.

Derivation. Let $X = N + r \left( E - 1 \right)$ with $N \sim \mathcal { N } ( 0 , 1 ) , E \sim \mathrm { E x p } ( 1 )$ independent, and $r \geq 0$ . Cumulants are additive, and the exponential law has $\kappa _ { 3 } = 2 , \kappa _ { 4 } = 6$ , so

$$
\mathrm { V a r } ( X ) = 1 + r ^ { 2 } , \qquad \kappa _ { 3 } ( X ) = 2 r ^ { 3 } , \qquad \kappa _ { 4 } ( X ) = 6 r ^ { 4 } ,\tag{33}
$$

whence

$$
\operatorname { s k e w } ( X ) = { \frac { 2 r ^ { 3 } } { ( 1 + r ^ { 2 } ) ^ { 3 / 2 } } } , \qquad \operatorname { k u r t } ( X ) = { \frac { 6 r ^ { 4 } } { ( 1 + r ^ { 2 } ) ^ { 2 } } } ,\tag{34}
$$

which is (32). The map $r \mapsto \operatorname { s k e w } ( X )$ is strictly increasing, so $r$ can be inverted from an observed skewness (or, redundantly, from an observed excess kurtosis); the ratio $r ^ { 2 } / ( 1 + r ^ { 2 } )$ is the share of the variance carried by the geometric component. For the empirical medians of $\ S 4$ (skew = 0.27, kurt = 0.35) the two inversions give $r = 0 . 6 0$ and $r = 0 . 5 7$ , i.e. a samplingnoise share of $2 4 \mathrm { - } 2 6 \%$ . Theorem 3.2 makes the sampling component vanish only in the limit $n  \infty ;$ at the segment counts realized here $( n \approx 4 5 )$ its share is small but not negligible, which is precisely why the residual skew is measurable.

## A.5 $\chi ^ { 2 }$ Confidence Bound for σ

Given K i.i.d. observations $Y _ { 1 } , \dots , Y _ { K } \sim { \mathcal { N } } ( \mu , \sigma ^ { 2 } )$ with sample variance $\begin{array} { r } { S ^ { 2 } = \frac 1 { K - 1 } \sum _ { i } ( Y _ { i } - \bar { Y } ) ^ { 2 } } \end{array}$ ，

$$
{ \frac { ( K - 1 ) S ^ { 2 } } { \sigma ^ { 2 } } } \sim \chi _ { K - 1 } ^ { 2 } .\tag{35}
$$

We want $S$ to estimate σ within 10% relative error at confidence 0.95, i.e. $\operatorname* { P r } ( 0 . 9 \sigma \le S \le 1 . 1 \sigma ) \ge 0 . 9 5$ , which is equivalent to

$$
\operatorname* { P r } \bigl ( 0 . 8 1 ( K - 1 ) \le \chi _ { K - 1 } ^ { 2 } \le 1 . 2 1 ( K - 1 ) \bigr ) \ge 0 . 9 5 .\tag{36}
$$

Let $\chi _ { K - 1 , \alpha } ^ { 2 }$ denote the α-quantile of the chi-squared distribution with $K - 1$ degrees of freedom. Using the two central quantiles 0.025 and 0.975,

$$
\chi _ { K - 1 , 0 . 0 2 5 } ^ { 2 } \geq 0 . 8 1 ( K - 1 ) \quad \mathrm { a n d } \quad \chi _ { K - 1 , 0 . 9 7 5 } ^ { 2 } \leq 1 . 2 1 ( K - 1 ) .\tag{37}
$$

Both inequalities hold simultaneously for the first time at $K = 1 9 3 \colon$ there $\chi _ { 1 9 2 , 0 . 0 2 5 } ^ { 2 } = 1 5 5 . 5 2 \geq 0 . 8 1 { \cdot } 1 9 2 = 1 5 5 . 5 2$ and $\chi _ { 1 9 2 , 0 . 9 7 5 } ^ { 2 } = 2 3 2 . 2 7 \leq 1 . 2 1 { \cdot } 1 9 2 =$ 232.32, whereas at $K = 1 9 2$ the corresponding values are $1 5 4 . 6 2 < 1 5 4 . 7 1$ and $2 3 1 . 1 6 > 2 3 1 . 1 1$ . We round up to $K = 2 0 0$ decodes per prompt. This requirement is the reason our test set is limited to 300 prompts: 200 decodes per prompt make an exhaustive evaluation of a larger test set prohibitively expensive.

## B Training Details

All predictors are implemented in PyTorch and trained with Adam. Unless noted otherwise:

• learning rate $1 0 ^ { - 3 }$ with cosine decay;

• batch size 32;

• up to 100 epochs, with early stopping on validation MAE (patience 10);

• inputs are prefill hidden states exported with attn\_implementation="eager";

• loss: Smooth-L1 (Huber) for regression, and a cross-entropy plus MSE mixture for soft labels.

MLP-Concat. Concatenate the last-token hidden vectors of all layers. For Qwen3-32B (64 layers, hidden size 5120) the input dimension is 327,680, and the backbone applies $3 2 7 6 8 0 \to 1 2 8 \to 1 2 8 \to 1$ linear maps with ReLU activations in between.

EGTP-style. Compute the entropy $H _ { i }$ of the next-token distribution at each prefill position, normalise to weights $w _ { i } = \mathrm { s o f t m a x } ( H ) _ { i }$ , weight-pool the hidden states per layer, and mean-pool across layers. An MLP regressor is fitted on the resulting 5120-dimensional vector.

FiLM. Condition on the prefill token count L through a small MLP that produces a scale $\gamma$ and a shift $\beta ,$ applied element-wise to the mean-pooled hidden state before the regression head.

Soft-Label. Discretize target lengths into 100 equal-width bins and smooth the target with a Gaussian kernel of $\sigma _ { \mathrm { b i n } } = 2$ bins. The loss is $\mathcal { L } = \lambda \mathcal { L } _ { \mathrm { C E } } +$ $( 1 - \lambda ) \mathcal { L } _ { \mathrm { M S E } }$ with $\lambda = 0 . 5$

MoE variants. Four experts cover four length intervals with 20% overlap, namely [0, 375), [375, 750), [750, 1149) and $\lbrack 1 1 4 9 , \infty )$ . Each expert is a 2- layer MLP, and the gate is a 2-layer network with softmax output. MoE-E2E adds an auxiliary routing loss derived from the expert prediction intervals.

## C Full Benchmark Results

Table 4 reports mean and standard deviation over 10 independent training runs.

Table 4: Mean ± std over 10 runs (Qwen3-32B, ShareGPT).
<table><tr><td>Method</td><td>MAE</td><td>Acc@100</td><td> $\operatorname { A c c } \ @ 2 0 0$ </td><td>Acc@500</td></tr><tr><td>MLP-Concat</td><td> $1 6 9 . 3 \pm 4 . 2$ </td><td> $4 8 . 9 \pm 1 . 1 \%$ </td><td> $7 1 . 1 \pm 0 . 9 \%$ </td><td> $9 4 . 4 \pm 0 . 3 \%$ </td></tr><tr><td> $\mathrm { E G T P - s t y l e }$ </td><td> $4 6 6 . 8 \pm 1 . 1$ </td><td> $1 4 . 6 \pm 1 . 0 \%$ </td><td> $2 5 . 4 \pm 3 . 0 \%$ </td><td> $6 1 . 9 \pm 3 . 3 \%$ </td></tr><tr><td> $_ \mathrm { M o E - S o f t }$ </td><td> $1 7 0 . 7 \pm 6 . 3$ </td><td> $4 7 . 9 \pm 1 . 5 \%$ </td><td> $7 1 . 0 \pm 1 . 2 \%$ </td><td> $9 4 . 2 \pm 0 . 5 \%$ </td></tr></table>

## D Cross-Dataset Generalization

Table 5 summarizes experiments that train on one dataset and test on another (all with Qwen3-32B unless noted otherwise). Absolute MAE is strongly influenced by the alignment between the training and test domains, while the σ-driven ceiling stays the dominant constraint.

When the domains are closely aligned (LeetCode → HumanEval, both code generation), the predictor achieves low MAE and high accuracy. When they difer substantially (LMSYS → ShareGPT), MAE drifts toward the ceiling implied by the test-set σ. The pattern reinforces the central claim: once the predictor is adequately trained, the residual error is dominated by the inherent stochasticity of generation rather than by architectural limitations.

Table 5: Cross-dataset generalization (train → test).
<table><tr><td>Train → Test</td><td>MAE Acc@100 Acc@200 Acc@500 x(p90)</td><td></td><td></td></tr><tr><td>LMSYS-200K →</td><td>172.60 47.22%</td><td>70.89%</td><td>94.13% 393.05</td></tr><tr><td>ShareGPT-100K LMSYS-200K →</td><td>182.40 43.75%</td><td>68.44%</td><td>92.43% 435.48</td></tr><tr><td>ShareGPT-18k-18.3k LMSYS-200K →</td><td>44.33 92.96%</td><td>99.04%</td><td>99.88% 86.23</td></tr><tr><td>GSM8K ShareGPT-100K →</td><td>132.70 57.76%</td><td>80.16%</td><td>96.68%299.96</td></tr><tr><td>ShareGPT-10K LeetCode → HumanEval</td><td>29.28</td><td></td><td>96.28%100.00%100.00%76.17</td></tr></table>

## E MoE Meta-Gate Definitions

For completeness we record the confidence-aware meta gate used in the MoE experiments. Let $K = 4$ experts, where expert i outputs a mean prediction $\mu _ { i } ( x )$ and a risk estimate $r _ { i } ( x )$

$$
\mu _ { i } ( x ) = W _ { i , \mu } h _ { i } ( x ) + b _ { i , \mu } ,\tag{38}
$$

$$
r _ { i } ( x ) = \mathrm { S o f t p l u s } \big ( W _ { i , r 2 } \mathrm { R e L U } ( W _ { i , r 1 } h _ { i } ( x ) + b _ { i , r 1 } ) + b _ { i , r 2 } \big ) .\tag{39}
$$

The risk head is trained to predict the expert’s absolute error $e _ { i } ( x , Y ) =$ $\left| \mu _ { i } ( x ) - Y \right|$ with a Smooth-L1 loss.

The meta gate concatenates the feature $x ,$ the expert predictions $\mu ( x )$ 2 the risks $r ( x )$ , and the interval distances $d ( x )$ , where $d _ { i } ( x ) = \operatorname* { m a x } ( a _ { i } -$ $\mu _ { i } , 0 ) + \mathrm { m a x } ( \mu _ { i } - b _ { i } , 0 )$ for the expert interval $[ a _ { i } , b _ { i } ]$ . It outputs a prior $\pi ( x ) = \operatorname { s o f t m a x } ( g ( z ( x ) ) )$ and scores

$$
s _ { i } ( x ) = \log \bigl ( \pi _ { i } ( x ) + \varepsilon \bigr ) - \alpha r _ { i } ( x ) - \beta d _ { i } ( x ) ,\tag{40}
$$

with routing weights $w _ { i } ( x ) = \exp ( s _ { i } ( x ) / T ) \big / \sum _ { j } \exp ( s _ { j } ( x ) / T )$ and final prediction $\begin{array} { r } { \hat { y } ( x ) = \sum _ { i } w _ { i } ( x ) \mu _ { i } ( x ) } \end{array}$

The total training loss is ${ \mathcal L } = { \mathcal L } _ { \mathrm { r e g } } + \lambda _ { \mathrm { k l } } { \mathcal L } _ { \mathrm { k l } } + \lambda _ { \mathrm { b a l } } { \mathcal L } _ { \mathrm { b a l } }$ , where $\mathcal { L } _ { \mathrm { r e g } }$ is Smooth-L1 on $\hat { y } , \mathcal { L } _ { \mathrm { k l } }$ is the KL divergence from the oracle soft routing targets $q _ { i } \propto \exp ( - | \mu _ { i } - Y | )$ , and $\mathcal { L } _ { \mathrm { b a l } }$ penalises unbalanced expert usage.