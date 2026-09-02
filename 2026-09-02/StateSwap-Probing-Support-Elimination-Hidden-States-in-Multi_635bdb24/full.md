# StateSwap: Probing Support–Elimination Hidden States in Multiple-Choice Questions

Chao Gao<sup>1</sup>, Haijiang Liu<sup>2,3</sup>, Qiyuan Li<sup>2,3</sup>, Caicai Guo<sup>2,3</sup>, Frank van Harmelen<sup>1</sup>, Jinguang Gu<sup>2,3</sup>

<sup>1</sup>Department of Computer Science, Vrije Universiteit Amsterdam, Amsterdam, The Netherlands <sup>2</sup>School of Computer Science and Technology, Wuhan University of Science and Technology, Wuhan, China <sup>3</sup>Hubei Province Key Laboratory of Intelligent Information Processing and Real-time Industrial System, Wuhan University of Science and Technology, Wuhan, China

Correspondence: c.gao@vu.nl, simon@wust.edu.cn

## Abstract

Large language models often answer the same multiple-choice question inconsistently when it is posed under support-oriented and elimination-oriented framings. We investigate whether these discrepancies arise from different internal representations induced by the two framings. We introduce a dual-framing protocol with minimally varied prompts that use either support- or elimination-oriented framing while keeping the evaluation target fixed. To probe the internal computation, we append an untrained special token, [STATE], and treat its residual-stream activation as an intervention interface. Across both models, the two framings induce separable [STATE] activations concentrated in intermediate layers. Swapping these activations between paired prompts systematically changes predictions and improves cross-framing agreement, providing intervention-based evidence that the activations are behaviorally relevant. Beyond instance-level substitution, mean-difference steering directions derived from the dual-framing contrast exhibit more bounded layer-wise responses than matched contrastive activation addition directions under the evaluated protocol. Code is available at https://github.com/Cha0Ga0/SWAPSTATE

## 1 Introduction

Recent work has shown that the process-ofelimination (PoE) strategy, widely used by humans for solving multiple-choice questions (MCQs), can improve the performance of large language models (LLMs) (Ma and Du, 2023). This approach is motivated by the fact that support-oriented (SUP) prompts (“which option is correct?”) and elimination-oriented (ELIM) prompts (“which options are incorrect?”) are logically tied to the same answer key. For example, some studies first prompt the model to identify and eliminate incorrect options before deriving the final answer (Zhu et al.,

![](images/c6bad1a47994ba15baebfaa6fb311aeba6e1ee7939504f5221f6ad1d7f377118.jpg)  
Figure 1: Overview of StateSwap. StateSwap swaps [STATE] activations between paired SUP and ELIM prompts while keeping the textual input fixed. The figure shows an idealized agreement-increasing outcome for illustration only; substitution may improve, reduce, or leave unchanged accuracy and agreement. We interpret systematic prediction changes, including but not limited to improvements, as evidence of behavioral relevance under intervention.

2025; Fu et al., 2025; Wang et al., 2026; Zhao and Zhang, 2025). However, empirical studies also report that elimination-style prompting can instead impair accuracy (Balepur et al., 2024). At first glance, this appears contradictory: if SUP and ELIM prompts encode the same task objective and semantic content for decision-making, they should not lead to different predictions. This motivates the following question: do the two framings induce distinguishable internal representations?

Inspired by recent mechanistic studies (Heimersheim and Nanda, 2024; Zhang et al., 2025; Lu et al., 2025), we investigate prompt framing under controlled inference settings, using deterministic decoding and paired prompt evaluation to isolate representation-level effects from decoding variability. We test whether framing-dependent behavioral differences are accompanied by distinguishable activation patterns in the model’s hidden-state space. Separability identifies a representational correlate of prompt framing, and activation substitution evaluates its behavioral relevance.

Based on this perspective, we investigate how SUP and ELIM prompts shape the internal representations of the model, and whether targeted activation substitution can provide intervention-based evidence on the role of these representations. Specifically, we use a newly registered, untrained special token with no pretrained lexical semantics, [STATE], to analyze the geometry of the hidden states elicited by the two framings and assess their representational distinguishability. We then introduce activation-level interventions that replace selected internal states while keeping the textual input fixed, allowing us to examine how changes in the induced representation affect the model’s output distribution over answer indices. Comparisons with dual-prompt ensemble baselines distinguish activation substitution from simple answer aggregation. Finally, we test whether the intervention remains effective when [STATE] is moved away from its canonical final position.

The main contributions of this work are threefold:

1. We identify distinguishable [STATE] representations induced by SUP and ELIM prompts that encode the same underlying decision problem.

2. We show that substituting framing-sensitive [STATE] activations changes model predictions and is associated with higher accuracy and cross-framing agreement in the evaluated settings.

3. We find that mean-difference steering directions derived from the two-framing prompt contrast exhibit lower layer-wise variability than matched contrastive activation addition directions under the evaluated protocol.

## 2 Related Work

LLM reasoning has been studied from multiple complementary perspectives, including prompting, fine-tuning, symbolic tools, model collaboration, and output verification (He et al., 2025). Sparseautoencoder analyses characterize activation patterns associated with chain-of-thought reasoning (Chen et al., 2025), while causal tracing and model editing link localized computations to factual predictions (Meng et al., 2022). Circuit-level studies localize induction-like computations to particular attention heads (Olsson et al., 2022), and activation patching provides a framework for testing the behavioral relevance of localized components (Heimersheim and Nanda, 2024). Synthetic statetracking experiments further show that transformers can develop localized mechanisms, including late-layer MLP neurons, for representing and updating latent states (Zhang et al., 2025).

This interventionist view is further supported by recent work on inference-time intervention and activation steering demonstrating that high-level behaviors in LLMs can be influenced by manipulating directions in activation space (Li et al., 2023; Turner et al., 2024; Zou et al., 2023). Methods such as contrastive activation addition (CAA) (Rimsky et al., 2024), conditional activation steering (Lee et al., 2025), and dynamic steering approaches (Li et al., 2025) show that activation shifts derived from dataset-level statistics can systematically alter model outputs, including truthfulness, refusal behavior, and bias-related responses. These results provide strong evidence that internal representations encode behaviorally meaningful information in explorable and manipulable forms.

A parallel line of work explores interface-based control of internal computation via localized representation slots. In encoder-style models, special tokens such as [CLS] serve as sequence-level interfaces to global representations (Devlin et al., 2019), while in autoregressive models, parameter-efficient methods such as prompt tuning and prefix tuning learn continuous embeddings that steer model behavior without modifying base parameters (Lester et al., 2021; Li and Liang, 2021; Liu et al., 2022). These approaches highlight the role of localized interfaces as high-leverage access points to latent representations.

Motivated by this line of work, we adopt an intervention-based perspective to investigate how prompt framing shapes internal representations. To operationalize this perspective, we use a fixed, untrained token as a minimal read–write interface to the residual stream. This setup allows us to examine whether semantically equivalent prompts with different framings give rise to distinguishable decision-related representations for the same input instance and whether these representations can be aligned through instance-level intervention.

![](images/e6ae8621565e453b785eda510a003b8488f68f23ef0f886521324c6a8845991c.jpg)  
Figure 2: Paired SUP (green) and ELIM (red) prompts place an aligned [STATE] interface (yellow) at the same position. Layer-wise diagnostics localize candidate intervention regions (Section 3.3). StateSwap then replaces the full [STATE] activation between the paired framings within the selected layers (Section 3.2).

## 3 Methodology

We investigate whether framing-conditioned [STATE] activations are behaviorally relevant under controlled substitution. Section 3.1 defines the aligned inputs and state interface, Section 3.2 formalizes cross-framing substitution, and Section 3.3 describes how the intervention layers are localized; Appendix A.1 summarizes the notation.

## 3.1 Input Construction and State Interface

Consider a causal language model $f _ { \theta }$ with L Transformer blocks. We append a newly registered [STATE] token to each prompt, yielding an input sequence $\mathbf { x } = ( x _ { 1 } , \dots , x _ { T } )$ in which [STATE] occupies the final position $t _ { S } \triangleq T$ . Let $\mathbf { h } _ { t } ^ { ( l ) } \in \mathbb { R } ^ { d }$ denote the residual-stream representation at position t after Transformer block $l ,$ where $t \in \{ 1 , \ldots , T \}$ and $l \in \{ 1 , \ldots , L \}$ . We define the corresponding [STATE] representation as

$$
\mathbf { s } ^ { ( l ) } \triangleq \mathbf { h } _ { t _ { S } } ^ { ( l ) } .
$$

Although [STATE] has no pretrained lexical semantics and receives no task-specific training, its contextualized representation can aggregate information from the preceding prompt and propagate through later layers to affect subsequent generation. We therefore use $\mathbf { s } ^ { ( l ) }$ as a single-token read–write interface for inference-time interventions. Token registration and embedding initialization are described in Appendix A.2.

For each question q, a pair of prompts is constructed, differing only in the framing instruction: eliminating incorrect options (ELIM) versus supporting correct options (SUP). Let $I \in$ {ELIM, SUP} denote the framing instruction. Each prompt ends with [STATE]:

$$
\mathbf { x } _ { I } ( q ) = \operatorname { C o n c a t } ( I , q , [ \mathsf { S T A T E } ] ) .
$$

To ensure that [STATE] occupies the same token index in both framings, instruction segments are padded to the same token length (Appendix A.2).

## 3.2 State Substitution and Prediction Sensitivity

Let $W \subseteq \{ 1 , \dots , L \}$ denote the contiguous intervention region localized by the diagnostic in Section 3.3. For framing I, let $\bar { I }$ denote its complement. Each substitution direction is run independently on ${ \bf x } _ { I } ( { q } )$ , with the cached donor state from <sup>¯</sup>I substituted over W:

$$
\mathbf { s } ^ { ( k ) } ( I , q )  \mathbf { s } _ { \mathrm { c a c h e } } ^ { ( k ) } ( \bar { I } , q ) , \quad k \in W .
$$

For each $k \in W .$ the intervention overwrites only the post-block state $\mathbf { s } ^ { ( k ) } ( I , q )$ during the prompt-processing pass; all other residualstream vectors at the intervention point remain unchanged, although downstream activations may change through residual propagation. Computation then proceeds normally from the substituted states to standard autoregressive generation. Featurewindow slices are used only for diagnostics and localization of W; the intervention itself replaces the full d-dimensional state $\mathbf { s } ^ { ( k ) }$

## 3.3 Locating a Stable Intervention Window

For $\mathbf { s } ^ { ( l ) } \in \mathbb { R } ^ { d }$ , candidate regions are identified by scanning contiguous feature windows of size w:

$$
\begin{array} { r } { \mathbf { s } _ { [ j : j + w ) } ^ { ( l ) } \in \mathbb { R } ^ { w } , \quad j \in \{ 0 , \dots , d - w \} . } \end{array}
$$

For each layer l and window $( j , w )$ paired Cohen’s d statistic is computed from the paired question-wise ELIM–SUP differences $( \mathsf { A p - }$ pendix A.3). Two scalarizations are used.

Random-direction diagnostic Once per run, a single random unit vector $\mathbf { u } \in \mathbb { R } ^ { d }$ is sampled and projected onto the corresponding window slice:

$$
\begin{array} { r } { \phi _ { j , w } ^ { \mathrm { r a n d } } ( \mathbf { z } ) = \mathbf { u } _ { [ j : j + w ) } ^ { \top } \mathbf { z } , } \end{array}
$$

yielding $d _ { l } ^ { \mathrm { r a n d } } ( j , w )$

Mean-difference diagnostic For each $( l , j , w )$ we form the mean-difference direction

$$
\mathbf { v } _ { l } ( j , w ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \mathbf { s } _ { \mathrm { E L I M } , i , [ j : j + w ) } ^ { ( l ) } - \mathbf { s } _ { \mathrm { S U P } , i , [ j : j + w ) } ^ { ( l ) } \right)
$$

stabilize its normalization and project:

$$
\begin{array} { r l } & { \phi _ { l , j , w } ^ { \mathrm { m d } } ( \mathbf { z } ) = \hat { \mathbf { v } } _ { l } ( j , w ) ^ { \top } \mathbf { z } , } \\ & { ~ \hat { \mathbf { v } } _ { l } ( j , w ) = \displaystyle \frac { \mathbf { v } _ { l } ( j , w ) } { \operatorname* { m a x } ( \lVert \mathbf { v } _ { l } ( j , w ) \rVert , \epsilon _ { v } ) } , } \end{array}
$$

where $\epsilon _ { v } > 0$ is small, yielding $d _ { l } ^ { \mathrm { m d } } ( j , w )$

Window-level separability scores are summarized into a per-layer intensity for each layer. Layers are ranked by this intensity, and adjacent selected layers are merged into a contiguous intervention region W; the aggregation and selection procedure is given in Appendix A.3.

## 4 Experimental Setup and Evaluation Protocol

## 4.1 Datasets

To study prompt-framing effects under controlled conditions, we evaluate on two multiple-choice benchmarks: (i) a reasoning-focused subset of MMLU covering 17 subject categories (MMLU-17) (Hendrycks et al., 2021), and (ii) MedQA-CH, a Chinese medical benchmark (Jin et al., 2021), which require non-trivial reasoning and allow unambiguous comparison across equivalent formulations. To avoid data leakage, we use 50 calibration examples from the training split solely to localize a candidate intervention window; these examples do not enter downstream test evaluation, and the selected window is fixed for all reported experiments. After fixing the window, we conduct a post hoc robustness analysis using 1,000 examples and 500 bootstrap resamples. Dataset statistics are provided in Appendix B.1. All downstream inference and reported evaluations use the official test splits.

## 4.2 Models and decoding

We evaluate on reasoning-focused multiple-choice benchmarks and open-weight LLMs Qwen-2.5-7B-Instruct (Yang et al., 2024) and GLM-4-9B (GLM et al., 2024) in a zero-shot setting. All generations use deterministic decoding (greedy; no sampling) to avoid confounding intervention effects with sampling variance. We fix the chat template, context length, and decoding hyperparameters across all runs; details are provided in Appendix B.2.

## 4.3 Prompt structure and [STATE]

All prompts consist of an instruction segment, the question with options, and a final [STATE] token. The exact instruction templates are provided in Appendix B.3. To verify robustness to random [STATE] initialization, we rerun our method on randomly sampled instances with different random seeds while keeping all other settings fixed, and observe negligible variation in the generated responses (Appendix B.4).

## 4.4 Inference

We consider two framings: a SUP instruction, which asks the model to predict the correct option(s), and an ELIM instruction, which asks the model to predict the incorrect option(s). For each framing, we evaluate a baseline condition and an intervention condition in which the [STATE] representation is substituted over a selected layer region W using the other framing. All other computations are held fixed. Outputs from both framings are parsed into a shared option-set representation, as detailed in Appendix B.5.1.

For reported ensemble results, we use paired deterministic prompt realizations within each framing. Thus, SUP and ELIM each consist of two promptinstantiated runs, yielding two framing-specific voting groups. We use strict-majority voting within each group.

## 4.5 Metrics

We report accuracy (ACC) for each framing and cross-framing decision overlap measured by the Jaccard index. We further analyze correctness to distinguish symmetric and asymmetric outcomes. Response-level stability is evaluated using Length-Aware Cosine Similarity (LAC) and BERTScore-F1 computed on the full generated responses under deterministic decoding. Cosine similarity captures global embedding-level alignment, while

BERTScore-F1 reflects token-level semantic consistency between responses. Metric definitions and parsing rules are provided in Appendix B.5.2.

## 5 Results

We evaluate the proposed framework through the following evaluation questions (EQs):

EQ1 (Representation) Do the two framings induce separable internal representations?

EQ2 (Effects of State Substitution) Does exchanging the final-position [STATE] representation transfer decision-relevant information between the two framings?

EQ3 (Mechanism Specificity) Are these effects specific to the [STATE] interface rather than generic perturbations?

We additionally conduct a separate extension analysis to examine whether the underlying SUP– ELIM contrast remains useful without the [STATE] interface in a population-level steering setting.

## 5.1 EQ1: Separable Decision-Related Representations

Before evaluating the downstream effects of [STATE] substitution, we examine whether the two framings produce systematically different internal representations at the [STATE] interface. In this diagnostic step, representation-level separability localizes a plausible layer window W for subsequent interventions.

Separability diagnostics Figure 3 visualizes the [STATE] representations across representative layers. The SUP and ELIM representations are largely mixed in the early layers, become clearly separated in the intermediate layers, and show weaker separation in the later layers. This pattern suggests that framing-dependent structure at the [STATE] position is concentrated in a subset of intermediate layers rather than uniformly distributed across depth. We further quantify this trend with layerwise separability diagnostics for both Qwen-2.5-7B and GLM-4-9B in Appendix D.3.

Localization of the intervention window W Based on the intermediate-layer concentration observed in Figure 3 and the layer-wise separability diagnostics, we select a compact intervention window W for [STATE] substitution. For Qwen-2.5- 7B, we use layers 11–20 as the intervention window.

![](images/25c1830e9839366082218c503d0ca8cf4f86478ae360057d2e234e283f972efb.jpg)  
Figure 3: PCA visualization of [STATE] representations for Qwen-2.5-7B at Layers 1, 11, and 27. Red and blue points denote SUP and ELIM prompts, respectively. The two framings are largely mixed in the early layers, become clearly separated in the intermediate layers, and show weaker separation in the later layers. Additional layer-wise PCA visualizations are provided in Appendix D.1.

For GLM-4-9B, the same separability-based criterion selects layers 21–40; full cross-model diagnostics are reported in Appendix D.3. These windows are determined from representation-level separability diagnostics rather than downstream task performance. A post-hoc comparison across candidate Qwen layer ranges, including the magnitude and stability of output-distribution shifts under substitution, is provided in Appendix D.2. The stability of the localized intermediate-layer region under a larger calibration sample and bootstrap resampling is evaluated in Appendix D.4.

Random-label control To assess whether the observed layer-wise separability could arise from spurious correlations or artifacts of the analysis procedure, we perform a random-label control in which the binary framing labels are randomly permuted while all representations and projection settings are held fixed. This provides a baseline for estimating the level of separability expected under chance alignment.

As shown in Figure 4, the separability obtained from 10 random label permutations remains consistently low across layers and window sizes, in contrast to the structured intermediate-layer peak observed under the true labels. The gap between the two conditions persists across all window sizes considered, indicating that the localization pattern is not an artifact of windowing, random projection, or label imbalance.

## 5.2 EQ2: Effects of State Substitution

Decision-related behavioral effects of [STATE] substitution Table 1 examines whether substituting the localized [STATE] representation leads to observable downstream changes in decision behavior. Greedy decoding removes sampling randomness from the BASE→SUB comparison. Across both models, benchmarks, and framings, [STATE] substitution consistently changes the model’s prediction outcomes. In particular, both framingspecific accuracy and cross-framing Jaccard overlap increase under substitution. These changes provide behavioral evidence that the substituted [STATE] activations affect decision-related computation in addition to exhibiting linear separability. More fine-grained accuracy breakdowns and ensemble comparisons are provided in Appendix C.1.

![](images/04b1b33b82c081e153fa1ce8eacd55cd6baa4f5e35c25df5782560aa36e2a735.jpg)  
Figure 4: Random-label control for layer-wise separability across window sizes. Solid lines: true labels. Dashed lines: shuffled-label mean across repeated shuffles. Shaded regions: ±2σ across shuffles. For each layer and feature-window size w, the separability score is the Top-K mean of the absolute paired Cohen’s d values across feature-window positions. Scores are jointly min–max normalized using the union of the true-label values and the shuffled-label mean ±2σ bounds across all feature-window sizes.

![](images/9bda44cec13f04dfcc2fbb880fffcd2046364fc592d6fac9f32d5f9ffeb60f30.jpg)

![](images/0ee71477647845871f50f69dfc8d89d615ee1f08686c5d546831ca38c2da256b.jpg)  
Figure 5: Base-to-substitution accuracy transition matrices. Each matrix shows the proportion of examples transitioning between baseline and substituted predictions, aggregated over both framings. Rows indicate baseline outcomes (Correct / Wrong), and columns indicate outcomes after [STATE] substitution. The four cells correspond to Correct→Correct, Correct→Wrong, Wrong→Correct, and Wrong→Wrong transitions. Results are shown for Qwen-2.5-7B (left) and GLM-4-9B (right).

Framing asymmetry under intervention Figure 5 summarizes the accuracy transitions induced by [STATE] substitution for both models. Each matrix reports how predictions shift between correct and incorrect outcomes when replacing the original decision state with its counterpart from the other framing.

<table><tr><td colspan="3">Setting ACC (SUP) ACC (ELIM) Jaccard</td></tr><tr><td></td><td>Qwen-2.5-7B</td><td></td></tr><tr><td>MMLU-17 (base)</td><td>65.52 65.38 66.53</td><td>75.46 76.63</td></tr><tr><td>MMLU-17 (Sub)</td><td>66.31 74.36</td><td>78.10</td></tr><tr><td>MedQA-CH (base) MedQA-CH (Sub)</td><td>77.12 80.26 77.45</td><td>81.58</td></tr><tr><td></td><td>GLM-4-9B</td><td></td></tr><tr><td></td><td></td><td></td></tr><tr><td>MMLU-17 (base) MMLU-17 (Sub)</td><td>66.47 68.74 69.21 72.36</td><td>72.96 78.21</td></tr><tr><td></td><td>75.91</td><td></td></tr><tr><td>MedQA-CH (base) MedQA-CH (Sub)</td><td>73.54 76.82 77.92</td><td>78.67 82.48</td></tr></table>

Table 1: Behavioral outcomes under cross-framing [STATE] substitution. Base denotes inference with the aligned [STATE]-augmented prompt but without activation replacement. Sub denotes cross-framing replacement over the selected layer window W. For ACC (SUP), the recipient run uses a SUP prompt and receives the cached [STATE] activation from its paired ELIM prompt; ACC (ELIM) uses the reverse direction. Each value is computed by within-framing strict-majority voting over two deterministic prompt realizations, with ties counted as incorrect. SUP and ELIM predictions are not combined. Jaccard is the intersection-over-union of the question-index sets answered correctly under SUP and ELIM. All values are percentages; higher values indicate higher accuracy or greater overlap between successful instances. All values are percentages; higher is better. Per-task MMLU accuracy is reported in Table C.3.

Across both Qwen-2.5-7B and GLM-4-9B, substantial off-diagonal mass shows that substitution systematically alters model behavior. A non-trivial fraction of previously incorrect predictions becomes correct after substitution, while a smaller fraction of correct predictions degrades. The imbalance between these transitions reveals an asymmetric behavioral effect across the two framing directions. Additional analysis in Appendix C.2 shows that [STATE] substitution more often increases than decreases eliminated-incorrect mass, suggesting enhanced error-elimination determinacy.

## 5.3 EQ3: Interface, Content, and Position Specificity

Ablation settings We conduct a series of controlled ablations to isolate which aspects of the intervention are responsible for the observed behavioral effects. For each ablation, we compare the intervened output against the model’s original greedy-decoding output without activation intervention. All interventions are applied within the same layer window W identified in Section 5.1. The ablations are organized along three orthogonal dimensions:

1. Intervention object We compare StateSwap at [STATE] with a final-content-token control that applies the same swap to the original prompt’s final content token without [STATE]. This comparison measures interface specificity.

2. Substitution content We vary what replaces the original [STATE] representation, comparing structured cross-framing substitutions with content-agnostic controls. This comparison measures the dependence of model behavior on substitution content.

3. Interface position We move [STATE] from its canonical final position to earlier positions relative to the question and option segments while keeping the substitution operation and layer window fixed. This comparison measures position dependence.

Detailed implementations of each ablation, including zeroing, random replacement, framingconditioned substitution, and alternative interface positions, are provided in Appendix E.

Interface specificity: [STATE] versus finalcontent-token substitution We assess whether the observed effects depend on the intervention location or can be reproduced by modifying latestage token representations more generally. We apply the same substitution procedure to the original final content token, while keeping the intervention window and decoding configuration fixed.

As shown in Figure 6a, substitution at the final content token produces outputs that remain close to the baseline, with similarity scores concentrated near 1.0. Under the tested conditions, the [STATE] interface is therefore more effective than the final-content-token control at altering model behavior. Quantile summaries for both conditions are reported in Table E.3.

Effect of substitution content We examine how the nature of the substituted representation affects model behavior. Figure 6b reports response-level

![](images/2b85ec8b46f2a371440b009f3f98e50aba3266cc5378fc51f6b68066c8d45fa2.jpg)

![](images/68e5f6f924274341779d9de0ff87767023c9942fd7bccf7e813f126fe2359c44.jpg)

(a) Interface specificity. Substituting [STATE] vs. the final content token under the same window W and greedy decoding.  
![](images/8824fadff7328444d2cdb0a3f79cf5710d6a3c4edc0366eb9092ece8d0687867.jpg)

![](images/262d51b54c77298841210cca6004b2c3fb7cdd3d211c857add8f63aaee93c940.jpg)  
(b) Ablations at [STATE]. Zeroing and random replacement vs. cross-framing substitution.

Figure 6: Response-level stability controls. (a) shows that late-position content-token states are substantially less effective as intervention targets than the [STATE] interface. (b) distinguishes unstructured residual injections (zero/random), which degrade stability, from structured cross-framing substitution, which preserves high-similarity regimes.

similarity between the baseline output and outputs produced under different intervention types, measured using length-aware cosine similarity (LAC) and BERTScore-F1. Quantile statistics for both metrics are provided in Table E.2.

Unstructured interventions (e.g., zeroing or random replacement) substantially reduce response similarity, whereas cross-framing substitution preserves high similarity across most examples. The distinct response regimes associate StateSwap’s effects with the content of the substituted activation.

[STATE] token ablation We also evaluate the effect of inserting the [STATE] token on MedQA-CH accuracy by comparing the original prompt with an otherwise identical prompt that includes the token. Adding [STATE] changes accuracy negligibly across both models (Appendix E.1), separating token insertion from the subsequent activationsubstitution effect.

Position sensitivity under misaligned substitution We further examine whether cross-framing substitution depends on the structural placement of [STATE]. More than 95% of generations under the tested misaligned placements either become incoherent or fail to produce a valid task-relevant response; the evaluated configurations, failure criterion, and canonical-position comparison are provided in Appendix E.3.

![](images/fbc7b43033810dabacff4c539bbda0cb340dc2431e616b6474191b5a8a10d692.jpg)  
Figure 7: Accuracy under activation steering across intermediate layers. Solid lines denote steering directions derived from the dual-framing contrast between SUP and ELIM prompts, while dashed lines denote the original CAA baseline. Red and blue curves correspond to Qwen-2.5-7B and GLM-4-9B, respectively. Horizontal dashed lines indicate unsteered baseline accuracies.

## 5.4 Dual Framing as a General Intervention Contrast

The preceding experiments characterize StateSwap within the dual-framing design. We next evaluate the SUP–ELIM contrast independently of the introduced interface by removing [STATE] and instantiating the same contrast in a populationlevel CAA-style mean-difference steering framework. This analysis compares the direction derived from matched SUP–ELIM prompts with the original CAA direction under otherwise matched settings.

As shown in Figure 7, the dual-framing direction produces systematic layer-dependent effects in both models even without the [STATE] token. Detailed aggregate statistics and complete layer-wise results, including comparisons with the original CAA direction and failures under strong coefficients, are provided in Appendix C.3.

## 6 Discussion

The choice of SUP and ELIM is central to what can be inferred from StateSwap. In preliminary analyses, we considered other complementary prompt pairs, including strictly translated Chinese and English instructions and prompts requesting a direct answer versus explicit reasoning. Such diversity can benefit prediction aggregating alternative reasoning paths improves accuracy (Wang et al., 2022; Lin et al., 2024). Language contrasts remained strongly classifiable from shallow through deep layers, plausibly because they introduce pervasive lexical and syntactic differences. Direct-answer and reasoning-first prompts were most distinguishable in intermediate and late layers, potentially reflecting different response strategies and generation plans. In both cases, a classifier could exploit factors broader than the decision distinction of interest.

The SUP–ELIM contrast was selected to reduce these confounds. The paired prompts preserve the question, options, language, decoding procedure, and answer key while changing whether the model identifies the correct option or eliminates the incorrect ones. Although this does not guarantee that framing is the only information encoded at [STATE], it supports a more specific conclusion: logically equivalent decision objectives can induce distinguishable intermediate representations whose exchange affects downstream predictions.

Additional, SUP and ELIM are broad observational labels that may each contain finer-grained states associated with domain, uncertainty, response strategy, or required deliberation. A model facing a medical question, for example, may adopt a more cautious state, potentially expressed through longer chains of thought, while remaining within the same SUP or ELIM class. Our analysis marginalizes over such within-class structure because it asks whether one controlled framing distinction is detectable and behaviorally relevant. Hierarchical or multi-factor analyses could instead separate framing from domain, confidence, and reasoning strategy, and test which finer-grained states remain independently substitutable.

## 7 Conclusion

StateSwap provides a training-free framework for analyzing framing-sensitive representations through a dedicated residual-stream interface. Across two evaluated LLMs, support- and elimination-oriented prompts induce separable intermediate-layer [STATE] activations. Exchanging these activations systematically changes downstream predictions, while the controls distinguish structured substitution from token insertion, finalcontent-token replacement, unstructured perturbations, and prompt ensembling. These findings establish the behavioral relevance of framingconditioned activations within the tested models, multiple-choice framing protocol, and deterministic decoding setting.

## Limitations

We conduct the experiments in a controlled setting with four-option multiple-choice questions and deterministic greedy decoding. This design helps isolate activation-level effects from sampling variance and makes BASE-to-SUB differences easier to attribute to the intervention. However, it also limits the scope of our claims. We do not evaluate whether the same framing-sensitive states persist under sampling-based decoding, longer multi-step generation, or open-ended tasks.

The current formulation is also tied to the structure of multiple-choice questions. In particular, SUP and ELIM prompts define complementary option sets, and our evaluation relies on parsing outputs into correct and incorrect options. Extending StateSwap beyond MCQs would require paired prompt formulations with a shared evaluation target, as well as task-specific agreement metrics for open-ended outputs.

Our method further relies on an untrained [STATE] token inserted at a fixed position. Although different random initializations of this token produce negligible variation in our experiments, the inserted token may still introduce subtle distributional or representational shifts. We therefore do not claim that the observed effects are independent of the interface design or intervention site.

StateSwap also assumes white-box access to model activations and parameters, leaving its applicability to black-box models untested. Majority voting over paraphrased prompts is used only to reduce prompt-realization noise during evaluation, and is not part of the intervention method itself.

## Ethical Statements

This work does not involve human subjects, user studies, or the collection of personal data. All experiments are conducted offline on publicly available multiple-choice benchmarks, including MMLU and MedQA-CH, using openly released language models. The generated responses are synthetic benchmark outputs and are not associated with real individuals.

StateSwap is proposed as a diagnostic method for studying framing-sensitive internal representations under controlled inference settings. Although the intervention changes intermediate activations, our experiments are limited to semantically equivalent multiple-choice prompts and deterministic decoding. We do not present the method as a deployed alignment or decision-support system.

A potential risk of activation-level interventions is that similar techniques could be adapted for broader model steering, including biased or undesired behavioral changes. We therefore restrict our claims to offline analysis and report ablations to distinguish structured state substitution from generic perturbations. Since one benchmark is medical, we emphasize that the results should not be interpreted as evidence of clinical reliability or suitability for real-world medical decision making. Any use in high-stakes settings would require separate domain validation and safety evaluation.

We used AI-assisted tools responsibly to support language editing and manuscript preparation; all research ideas, analyses, results, and conclusions were developed and verified by the authors, who take full responsibility for the content of this paper.

## References

Nishant Balepur, Shramay Palta, and Rachel Rudinger. 2024. It’s not easy being wrong: Large language models struggle with process of elimination reasoning. In Findings of the Association for Computational Linguistics: ACL 2024, pages 10143–10166.

Xi Chen, Aske Plaat, and Niki van Stein. 2025. How does chain of thought think? mechanistic interpretability of chain-of-thought reasoning with sparse autoencoding. arXiv preprint arXiv:2507.22928.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186.

Qihang Fu, Yongbin Qin, Ruizhang Huang, Yanping Chen, Yulin Zhou, and Lintao Long. 2025. Exclusion of thought: Mitigating cognitive load in large language models for enhanced reasoning in multiplechoice tasks. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 21673– 21686.

Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, et al. 2024. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793.

Feijuan He, Han Lai, Jiaqi Liu, Bo Wang, Haoran Chen, Haohan Liu, and Chenxi Zhang. 2025. Solving mathematical problems using large language models: A survey. Data Intelligence, 7(4):907–946.

Stefan Heimersheim and Neel Nanda. 2024. How to use and interpret activation patching. arXiv preprint arXiv:2404.15255.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. 2021. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences, 11(14):6421.

Bruce W. Lee, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Erik Miehling, Pierre Dognin, Manish Nagireddy, and Amit Dhurandhar. 2025. Programming refusal with conditional activation steering. In International Conference on Learning Representations.

Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3045–3059, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023. Inferencetime intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems, 36:41451–41530.

Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190.

Yichen Li, Zhiting Fan, Ruizhe Chen, Xiaotang Gai, Luqi Gong, Yan Zhang, and Zuozhu Liu. 2025. Fairsteer: Inference time debiasing for llms with dynamic activation steering. In Findings ofthe Association for Computational Linguistics: ACL 2025, pages 11293–11312.

Lei Lin, Jiayi Fu, Pengli Liu, Qingyang Li, Yan Gong, Junchen Wan, Fuzheng Zhang, Zhongyuan Wang, Di Zhang, and Kun Gai. 2024. Just ask one more time! self-agreement improves reasoning of language models in (almost) all scenarios. In Findings of the Association for Computational Linguistics: ACL 2024, pages 3829–3852.

Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Tam, Zhengxiao Du, Zhilin Yang, and Jie Tang. 2022. P-tuning: Prompt tuning can be comparable to fine-tuning across scales and tasks. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 61–68. Association for Computational Linguistics.

Wenquan Lu, Yuechuan Yang, Kyle Lee, Yanshu Li, and Enqi Liu. 2025. Latent chain-of-thought? decoding the depth-recurrent transformer. arXiv preprint arXiv:2507.02199.

Chenkai Ma and Xinya Du. 2023. Poe: Process of elimination for multiple choice reasoning. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 4487–4496.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in gpt. Advances in neural information processing systems, 35:17359–17372.

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. 2022. In-context learning and induction heads. CoRR, abs/2209.11895.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. 2024. Steering Llama 2 via contrastive activation addition. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15504–15522. Association for Computational Linguistics.

{Alexander Matt} Turner, Lisa Thiergart, Gavin Leech, David Udell, Ulisse Mini, and Monte MacDiarmid. 2024. Activation addition: Steering language models without optimization. Working paper, arXiv.org, United Kingdom.

Mingsen Wang, Zhaoqiang Li, Xuegang Zhao, and Qihan Guo. 2026. Eliminate-then-select: A humancentric reasoning framework for educational question answering with llms. Information Processing & Management, 63(2):104422.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Yifan Zhang, Wenyu Du, Dongming Jin, Jie Fu, and Zhi Jin. 2025. Finite state automata inside transformers with chain-of-thought: A mechanistic study on state tracking. arXiv preprint arXiv:2502.20129.

Qianli Zhao and Mei Zhang. 2025. Elimination-based reasoning with llm for multiple-choice educational question answering. Journal ofKing Saud University Computer and Information Sciences, 37(7):204.

Zhenhao Zhu, Bulou Liu, Qingyao Ai, and Yiqun Liu.   
2025. Option-id based elimination for multiple   
choice questions. arXiv preprint arXiv:2501.15175.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. 2023. Representation engineering: A topdown approach to ai transparency. arXiv preprint arXiv:2310.01405.

## A Methodological Details

## A.1 Notation Summary

For clarity, we summarize the StateSwap-specific notation below. Standard Transformer notation follows Section 3.1; arguments $( I , q )$ are omitted when they are clear from context.

```latex
Term / Symbol Object (this paper)
framing and instance $I \in \{ \mathrm { S U P } , \mathrm { E L I M } \} ; \bar { I }$ is the com
plementary framing and q is the
shared question
framing-conditioned input ${ \bf x } _ { I } ( { q } )$ =
Concat(I, q, [STATE])
interface position $t _ { S } \triangleq { \dot { T } } ,$ , the final position in
${ \bf x } _ { I } ( { q } )$
interface state $\mathbf { s } ^ { ( l ) } ( I , q ) = \mathbf { h } _ { t _ { S } } ^ { ( l ) } ( I , q ) \in \mathbb { R } ^ { d }$
intervention region contiguous $W \subseteq \{ 1 , \ldots , L \}$
feature-window indices length $w ,$ start $j \in \{ 0 , \ldots , d -$
w}, and evaluated size set $\dot { \mathcal Ḋ W Ḍ } \subseteq$
$\{ 1 , \ldots , d \}$
feature-window slice $\mathbf { s } _ { [ j : j + w ) } ^ { ( l ) } \in \mathbb { R } ^ { w }$
window-level diagnostic $\dot { d _ { l } } ( \boldsymbol { j } , \boldsymbol { w } )$ , the paired Cohen’s d af
ter scalarization; $d _ { l } ^ { \mathrm { r a n d } }$ and $d _ { l } ^ { \mathrm { m d } }$
denote the two diagnostics
per-size layer intensity $\tilde { d } _ { l } ( w )$ , the mean of the K largest
values in $\{ | d _ { l } ( j , w ) | \} _ { j = 0 } ^ { d - w }$
aggregated layer intensity $\begin{array} { r } { \tilde { d } _ { l } = | \mathcal { W } | ^ { - 1 } \sum _ { w \in \mathcal { W } } \tilde { \dot { d } } _ { l } ( w ) } \end{array}$
state substitution $\mathbf { s } ^ { ( k ) } ( I , q ) \  \ \mathbf { s } _ { \mathrm { c a c h e } } ^ { ( \bar { k } ) } ( \bar { I } , q )$ for
$k \in { \dot { W } }$
```

## A.2 Construction and Alignment of the [STATE] Interface

We introduce [STATE] as an additional special token in the tokenizer vocabulary. Concretely, we add [STATE] via the tokenizer interface: tokenizer.add\_special\_tokens(·). After extending the tokenizer vocabulary, we resize the model’s input embedding matrix to match the updated vocabulary size. All experiments are inference-only: model parameters are not updated, and the embedding vector corresponding to [STATE] remains at its initialization value throughout.

Token-Level Alignment Our methodology requires [STATE] to occupy the same absolute token index $t _ { S }$ across ELIM and SUP prompts for the same question q. We ensure this by token-level alignment of the instruction segment. Let the full prompt be

$$
\mathbf { x } _ { I } ( q ) = \operatorname { C o n c a t } ( I , Q , [ \mathsf { S 7 A T E } ] ) .
$$

We construct the instruction segment so that it has the same token length for all instructions under the model tokenizer. For practical guidance, please refer to Appendix B.3.

Algorithm 1 Tokenize and Align Instructions   
Require: Instructions list $\mathcal { T } = \{ I _ { 1 } , I _ { 2 } , \ldots , I _ { n } \}$   
Ensure: Aligned token lists $\begin{array} { r } { \mathcal { T } _ { \mathrm { a l i g n e d } } , } \end{array}$ padding   
sizes ${ \mathcal P } ,$ target instruction length $L _ { \mathrm { t a r g e t } }$   
1: Initialize empty lists $\mathcal { T } _ { \mathrm { r a w } }$ and $L _ { \mathrm { b a s e } }$   
2: for each instruction I in I do   
3: tokens ← Tokenize(I)   
4: Append tokens to $\mathcal { T } _ { \mathrm { r a w } }$   
5: Append |tokens| to $L _ { \mathrm { b a s e } }$   
6: end for   
7: $L _ { \mathrm { t a r g e t } }  \mathrm { m a x } ( L _ { \mathrm { b a s e } } )$   
8: Initialize empty lists $\tau _ { \mathrm { a l i g n e d } }$ and $\mathcal { P }$   
9: for $i \gets 1$ to n do   
10: $p a d _ { i } \gets L _ { \mathrm { t a r g e t } } - L _ { \mathrm { b a s e } } [ i ]$   
11: $a l i g n e d _ { i } \gets \mathsf { \bar { C } o n c a t } ( [ \mathsf { P A D } ] ^ { p a d _ { i } } , \mathsf { T } _ { \mathrm { r a w } } [ i ] )$   
12: Append aligned<sub>i</sub> to T<sub>aligned</sub>   
13: Append pad<sub>i</sub> to $\mathcal { P }$   
14: end for   
15: return $\mathcal { T } _ { \mathrm { a l i g n e d } } , \mathcal { P } , L _ { \mathrm { t a r g e t } }$

Because the question block and the final [STATE] token are identical in both framings, equalizing the instruction-segment length guarantees that the absolute index $t _ { S }$ of [STATE] is matched between ELIM and SUP for each paired question q.

## A.3 Window Diagnostics and Candidate Region Selection

For window size w, we use start indices $j \in$ $\{ 0 , \ldots , d - w \}$ and the slice notation $\left[ j : j + w \right)$ For a scalarization $\phi : \mathbb { R } ^ { w }  \mathbb { R } .$ , define paired differences for each question i:

$$
\begin{array} { r } { \Delta _ { i } ^ { ( l , j , w ) } = \phi ( \mathbf { s } _ { \mathrm { E L I M } , i , [ j : j + w ) } ^ { ( l ) } ) - \phi ( \mathbf { s } _ { \mathrm { S U P } , i , [ j : j + w ) } ^ { ( l ) } ) . } \end{array}
$$

We report the paired Cohen’s d statistic:

$$
d _ { l } ( j , w ) = \frac { \overline { { \Delta ^ { ( l , j , w ) } } } } { \mathrm { s t d } ( \Delta ^ { ( l , j , w ) } ) + \epsilon } ,\tag{1}
$$

where the mean and standard deviation are computed over $\{ \Delta _ { i } ^ { ( l , j , w ) } \} _ { i = 1 } ^ { N }$

Candidate Region Selection We compute layerlevel intensities from absolute window-level separability scores using a Top-K mean aggregation. Because both diagnostics are reported in experiments, the same procedure is applied independently to $d _ { l } ^ { \mathrm { r a n d } } ( j , w )$ and $d _ { l } ^ { \mathrm { m d } } ( j , w )$

For a given diagnostic $d _ { l } ( j , w )$ and window size w, we define

$$
\tilde { d } _ { l } ( w ) = \mathrm { T o p K M e a n } \Big ( \{ | d _ { l } ( j , w ) | \} _ { j = 0 } ^ { d - w } ; K \Big ) ,
$$

where TopKMean $( \cdot ; K )$ averages the K largest values. Unless stated otherwise, we further average over a small set of window sizes W:

$$
\tilde { d } _ { l } = \frac { 1 } { | \mathcal { W } | } \sum _ { w \in \mathcal { W } } \tilde { d } _ { l } ( w ) .
$$

Layers are then ranked by $\tilde { d } _ { l }$ . To determine a threshold for selecting candidate layers, we rely on an empirical consistency criterion guided by an independent PCA analysis. Specifically, we observe that layers whose $\tilde { d } _ { l }$ values exceed approximately 80% of the maximum separability score consistently coincide with the region showing the clearest separation in the PCA visualization. Based on this observation, we select layers satisfying

$$
\tilde { d } _ { l } \ge 0 . 8 \cdot \operatorname* { m a x } _ { l ^ { \prime } } \tilde { d } _ { l ^ { \prime } }
$$

For each evaluated model, the layers satisfying this criterion form a single contiguous interval, which we denote by W. This threshold is not tuned for downstream performance. It is fixed across models and datasets and serves only as a diagnostic criterion for localizing layers that exhibit stable, representation-level differences between the two framings.

## B Experimental and Evaluation Details

## B.1 Datasets and Data Splits

We evaluate our approach on two four-option multiple-choice benchmarks: MedQA-CH and a reasoning-focused subset of MMLU. All downstream inference and reported task evaluations are conducted on the official test splits. A separate set of 50 examples from the training split is used only to localize the candidate intervention window; these examples do not enter the reported test evaluation. Appendix D.4 provides the full calibration protocol and a larger-scale bootstrap robustness analysis.

MedQA-CH MedQA-CH is the Chinese subset of MedQA, a medical question answering benchmark designed to evaluate professional-level clinical reasoning. It consists of 1,015 multiple-choice questions covering clinical knowledge and diagnostic reasoning. All questions are presented in Chinese and evaluated independently from English benchmarks.

MMLU (Reasoning-Focused Subset) We construct a reasoning-oriented subset of MMLU covering 17 subject categories that emphasize logical reasoning, mathematical thinking, legal analysis, ethical judgment, and domain-specific inference. The selected subsets span formal logic, mathematics, law, philosophy, medicine, and social sciences. All MMLU evaluations are conducted in English using the official test splits.

Table B.1 summarizes the detailed dataset composition. In total, our evaluation comprises 5,622 multiple-choice questions. For bilingual evaluation, English and Chinese benchmarks are tested separately rather than mixed within a single evaluation setting.

<table><tr><td>Dataset / Subset</td><td># Questions</td></tr><tr><td>MedQA</td><td></td></tr><tr><td>MedQA-CH</td><td>1,015</td></tr><tr><td>MMLU (Reasoning-Focused Subset)</td><td></td></tr><tr><td>Formal Logic Logical Fallacies</td><td>125 163</td></tr><tr><td>College Math Logic</td><td>100</td></tr><tr><td>Abstract Algebra (Proofs)</td><td>100</td></tr><tr><td>High School Math</td><td>270</td></tr><tr><td>Professional Law</td><td>1,534</td></tr><tr><td>Jurisprudence</td><td>108</td></tr><tr><td>Moral Disputes</td><td>346</td></tr><tr><td>Business Ethics</td><td>100</td></tr><tr><td>Philosophy</td><td></td></tr><tr><td></td><td>311</td></tr><tr><td>Sociology</td><td>201</td></tr><tr><td>World History</td><td>237</td></tr><tr><td>U.S. History</td><td>204</td></tr><tr><td>World Religions</td><td>171</td></tr><tr><td>Bioethics</td><td>272</td></tr><tr><td>Clinical Knowledge (Reasoning)</td><td>265</td></tr><tr><td>Medical Genetics</td><td>100</td></tr><tr><td>Total (MMLU subset)</td><td>4,607</td></tr><tr><td></td><td></td></tr><tr><td>Total (All datasets)</td><td>5,622</td></tr></table>

Table B.1: Dataset statistics. All datasets are evaluated using their official test splits. MedQA-CH is evaluated in Chinese, while MMLU subsets are evaluated in English.

## B.2 Decoding and Inference Configuration

All experiments are conducted in a zeroshot setting using deterministic greedy decoding. Specifically, we set do\_sample=False and max\_new\_tokens=1024. As greedy decoding is used, sampling-related parameters such as temperature and top\_k are not applicable and are ignored by the generation backend. All random seed values are 123.

The maximum context length is fixed across all experiments, and inputs exceeding this limit are truncated accordingly. Right padding is applied with a maximum of 64 padding tokens. All generations use the official chat templates provided by the respective models without modification.

For reproducibility, all runs are executed with fixed random seeds and deterministic computation settings. Inference is performed using torch\_dtype=float16, with key–value caching enabled to ensure consistent decoding behavior across runs.

## B.3 Detailed Prompt Templates

This section provides the prompt templates used in the experiments.

Prompt Components Each input prompt consists of three components: an instruction segment I, a question segment Q that contains the question text, and an option segment O that enumerates all candidate answers in a fixed format. In addition, a special interface token [STATE] is appended to the end of the prompt to mark the position at which the model is expected to produce its output. For models that employ a beginning-of-sequence token, the tokenizer’s default behavior is preserved.

Instruction Templates We use two groups of framing-specific instruction templates with aligned task objectives. One group instructs the model to reason step by step and identify correct options (Group 1), while the other instructs the model to reason step by step and identify incorrect options (Group 2). Within each group, multiple paraphrased variants are used to reduce dependence on a specific wording. All instruction templates are fixed across datasets and models.

## Group 1 - Sup prompt

Instruction: Proceed step by step: first list the complete chain of reasoning, then pick out Correct options. Question: [QUESTION]

Instruction: Take a careful, step-by-step approach. First, reason in detail, then find Correct choices. Question: [QUESTION]

## Group 2 - Elim prompt

Instruction: Proceed step by step: first list the complete chain of reasoning, then pick out Wrong options. Question: [QUESTION]

Instruction: Take a careful, step-by-step approach. First, reason in detail, then find Incorrect choices. Question: [QUESTION]

For each framing, we instantiate two instruction templates, yielding two deterministic greedy predictions for SUP and two for ELIM. Unless otherwise specified, reported ensemble results aggregate these paired prompt realizations by strictmajority voting within the corresponding framing or StateSwap variant; a 1–1 tie is counted as an incorrect ensemble prediction. This protocol separates prompt-template variation from decoding randomness, since all runs use greedy decoding.

## B.4 Robustness to [STATE] Initialization

The purpose of this experiment is to verify that StateSwap does not rely on incidental randomness from the untrained [STATE] token. Table B.2 shows that output similarity remains near-perfect across different random initializations, supporting the claim that [STATE] functions as a deterministic state interface rather than a source of stochastic variation.

## B.5 Output Parsing and Evaluation Metrics

## B.5.1 Parsing to Option Sets

To ensure consistency across languages and facilitate automatic evaluation, model outputs are postprocessed into a standardized representation consisting of two fields: Correct options:[. . . ] and Wrong options:[. . . ]. This step preserves the semantic meaning of the original responses while enforcing a fixed, machine-readable output format.

Options Answer extraction Let O = $\{ A , B , C , D \}$ denote the option alphabet. We parse each field into a set $S \subseteq { \mathcal { O } }$ by extracting all uppercase letters in O from the field, ignoring punctuation and separators (e.g., commas, Chinese enumeration marks).

## B.5.2 Metric Definitions

Accuracy (ACC) For each question with groundtruth correct option $y \in { \mathcal { O } } .$ , the prediction under a given projection is considered correct if $y \in S$

<table><tr><td rowspan="2">Swap direction</td><td rowspan="2">Seeda</td><td rowspan="2">Seedb</td><td colspan="2">LA-cos ↑</td><td colspan="2">BERT-F1 ↑</td></tr><tr><td>mean</td><td>std</td><td>mean</td><td>std</td></tr><tr><td rowspan="10">Elim_Baseline ↓  $S u p \_ B a s e l i n e$ </td><td>1</td><td>2</td><td>9.99999986589e-01</td><td>7.81e-08</td><td>9.99999991059e-01</td><td>5.65e-08</td></tr><tr><td>1</td><td>3</td><td>9.99999986589e-01</td><td>7.81e-08</td><td>9.99999991059e-01</td><td>5.65e-08</td></tr><tr><td>1</td><td>12</td><td>9.99999989569e-01</td><td>7.38e-08</td><td>9.99999991059e-01</td><td>5.65e-08</td></tr><tr><td>1</td><td>123</td><td>9.99999989569e-01</td><td>7.38e-08</td><td>9.99999991059e-01</td><td>5.65e-08</td></tr><tr><td>2</td><td>3</td><td>9.99999986589e-01</td><td>7.81e-08</td><td>9.99999991059e-01</td><td>5.65e-08</td></tr><tr><td>2</td><td>12</td><td>9.99999989569e-01</td><td>7.38e-08</td><td>9.99999991059e-01</td><td>5.65e-08</td></tr><tr><td>2</td><td>123</td><td>9.99999989569e-01</td><td>7.38e-08</td><td>9.99999991059e-01</td><td>5.65e-08</td></tr><tr><td>3</td><td>12</td><td>9.99999989569e-01</td><td>7.38e-08</td><td>9.99999991059e-01</td><td>5.65e-08</td></tr><tr><td>3</td><td>123</td><td>9.99999989569e-01</td><td>7.38e-08</td><td>9.99999991059e-01</td><td>5.65e-08</td></tr><tr><td>12</td><td>123</td><td>9.99999991059e-01</td><td>7.34e-08</td><td>9.99999991059e-01</td><td>5.65e-08</td></tr><tr><td rowspan="10"> $S u p \_ B a s e l i n e$   $\downarrow$  Elim_Baseline</td><td>1</td><td>2</td><td>9.99999982119e-01</td><td>7.04e-08</td><td>9.99999976158e-01</td><td>8.19e-08</td></tr><tr><td>1</td><td>3</td><td>9.99999982119e-01</td><td>7.04e-08</td><td>9.99999976158e-01</td><td>8.19e-08</td></tr><tr><td>1</td><td>12</td><td>9.99999984354e-01</td><td>6.62e-08</td><td>9.99999976158e-01</td><td>8.19e-08</td></tr><tr><td>1</td><td>123</td><td>9.99999984354e-01</td><td>6.62e-08</td><td>9.99999976158e-01</td><td>8.19e-08</td></tr><tr><td>2</td><td>3</td><td>9.99999982119e-01</td><td>7.04e-08</td><td>9.99999976158e-01</td><td>8.19e-08</td></tr><tr><td>2</td><td>12</td><td>9.99999984354e-01</td><td>6.62e-08</td><td>9.99999976158e-01</td><td>8.19e-08</td></tr><tr><td>2</td><td>123</td><td>9.99999984354e-01</td><td>6.62e-08</td><td>9.99999976158e-01</td><td>8.19e-08</td></tr><tr><td>3</td><td>12</td><td>9.99999984354e-01</td><td>6.62e-08</td><td>9.99999976158e-01</td><td>8.19e-08</td></tr><tr><td>3</td><td>123</td><td>9.99999984354e-01</td><td>6.62e-08</td><td>9.99999976158e-01</td><td></td></tr><tr><td>12</td><td>123</td><td>9.99999985844e-01</td><td>6.58e-08</td><td>9.99999976158e-01</td><td>8.19e-08 8.19e-08</td></tr></table>

Table B.2: Detailed robustness results for random [STATE] initialization. We sample 40 instances and rerun inference with five random seeds (1, 2, 3, 12, 123), keeping all other settings fixed. For each swap direction and seed pair, we report the mean and standard deviation of LA-cos and BERT-F1 computed over the 40 instances, showing negligible variation across seeds.

where S denotes the set of options predicted as correct by the model. Formally, accuracy is defined as:

$$
\mathrm { A C C } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } \left[ y _ { i } \in S _ { i } \right] ,
$$

where N is the total number of questions, $y _ { i }$ is the ground-truth correct option for the i-th question, $S _ { i }$ denotes the predicted set of correct options, and I[·] is the indicator function.

Jaccard Let $C _ { \mathrm { S u p } }$ and $C _ { \mathrm { E l i m } }$ denote the sets of question indices for which the model predicts the correct answer under the support- and eliminationoriented framings, respectively. For each question, correctness is determined solely based on the Correct options field in the normalized output, as described above.

We define cross-framing agreement using the Jaccard index over these two sets:

$$
J = \frac { \lvert C _ { \mathrm { S u p } } \cap C _ { \mathrm { E l i m } } \rvert } { \lvert C _ { \mathrm { S u p } } \cup C _ { \mathrm { E l i m } } \rvert } .
$$

Intuitively, a question contributes to the intersection if and only if the correct option appears in the Correct options list under both framings. For example, if the SUP output is Correct: [1,0,0,0] and the ELIM output is also Correct: [1,0,0,0], the question is counted as overlapping;

whereas if the two outputs differ (e.g., [1,0,0,0] vs. [0,1,0,0]), it is not. This metric therefore measures agreement in successful decisions between the two framings, rather than overlap between option sets for individual questions.

Relation Between ACC and Jaccard While ACC measures whether the model produces the correct answer under a single framing, the Jaccard index captures the consistency of correct decisions between the two framings. In particular, ACC reflects per-framing performance, whereas Jaccard evaluates cross-framing agreement conditioned on correctness. A high ACC with a low Jaccard score indicates that the model can answer correctly under either framing but does so inconsistently, while a high Jaccard score reflects stable decision-making between the two framings. The two metrics therefore provide complementary views of model behavior.

Length-Aware Cosine Similarity Given a baseline response r and its perturbed counterpart ${ \tilde { r } } ,$ we compute the cosine similarity between their sentence embeddings. To account for potential length mismatches, we apply a length-aware scaling factor:

$$
\operatorname { L A C } ( r , { \tilde { r } } ) = \cos ( \mathbf { e } _ { r } , \mathbf { e } _ { \tilde { r } } ) \cdot { \frac { \operatorname* { m i n } ( | r | , | { \tilde { r } } | ) } { \operatorname* { m a x } ( | r | , | { \tilde { r } } | ) } } ,
$$

where $\mathbf { e } _ { r }$ and $\mathbf { e } _ { \widetilde { r } }$ denote normalized sentence embeddings, and |r| denotes the character length of the response.

This metric captures semantic similarity while penalizing excessive length divergence caused by perturbations. The length-aware scaling term is introduced to account for cases where perturbations to the hidden state cause the model to generate excessively long or verbose outputs that are only weakly related to the original response. In such cases, standard cosine similarity alone may overestimate semantic consistency due to shared topical content. By down-weighting similarity scores when large length discrepancies occur, this metric penalizes uncontrolled or drifting generations and better reflects whether the core semantic content is preserved.

BERTScore We additionally compute BERTScore between the perturbed and baseline outputs, reporting F1 scores. BERTScore measures token-level semantic similarity using contextualized embeddings and is sensitive to both content preservation and phrasing changes. In our experiments, the baseline output is treated as the reference and the perturbed output as the candidate.

Relation Between LAC and BERTScore These two metrics are computed by comparing each perturbed output against its corresponding baseline output without any state modification. They are therefore intended to measure the sensitivity of the generation to perturbations applied to the [STATE] representation, rather than absolute output quality.

Length-aware cosine similarity reflects global semantic consistency between the perturbed and baseline responses, while BERTScore captures finegrained lexical and contextual deviations. Importantly, higher scores do not necessarily indicate better performance: values close to 1 suggest that the perturbation has little effect on the output, whereas lower scores indicate that modifying the internal state leads to substantial changes in the generated content. Extremely low scores may further reflect degenerate or incoherent outputs caused by the perturbation.

## B.5.3 Error-Elimination Metric

For instance i and framing $p \in \{ \mathrm { S U P } , \mathrm { E L I M } \}$ , let $W _ { i } ^ { ( p ) } \subseteq \mathcal { O }$ be the options parsed from the Wrong options field. We define the eliminated-incorrect

count as

$$
E _ { i } ^ { ( p ) } = \big | W _ { i } ^ { ( p ) } \setminus \{ y _ { i } \} \big | \in \{ 0 , 1 , 2 , 3 \} .
$$

Let $E _ { i , \mathrm { b a s e } } ^ { ( p ) }$ and $E _ { i , \mathrm { s u b } } ^ { ( p ) }$ denote the values before and after substitution. We define

$$
\Delta _ { i } ^ { ( p ) } = E _ { i , \mathrm { s u b } } ^ { ( p ) } - E _ { i , \mathrm { b a s e } } ^ { ( p ) } .
$$

An instance is counted as a RIGHT SHIFT if $\Delta _ { i } ^ { ( p ) } >$ 0, and as a LEFT SHIFT if $\Delta _ { i } ^ { ( p ) } < 0$ . We report the proportions of right- and left-shifted instances under each framing and their macro-average across SUP and ELIM. A higher right-shift frequency indicates stronger incorrect-option elimination.

## C Additional Behavioral Results

## C.1 Ensemble and Per-Task Results

Our main reported results are based on deterministic ensemble inference rather than a single prompt realization. Specifically, each single-framing result is averaged over two greedy runs with different prompt realizations. To further examine whether the observed gains can be explained by simple aggregation effects, we compare several ensemble variants: single-framing baselines (SUP-only and ELIM-only), StateSwap variants (SUP(ELIM State) and ELIM(SUP State)), a naive cross-framing ensemble (SUP+ELIM), framing-plus-StateSwap ensembles, and a dual StateSwap ensemble. As shown in Table C.1, ensembles involving swapped [STATE] predictions consistently outperform ensembles formed only from the original framings. In particular, the dual StateSwap ensemble achieves the best accuracy for both Qwen-2.5-7B and GLM-4-9B, suggesting that StateSwap transfers useful decision-state information beyond ordinary variance reduction from ensembling.

Since combining two framings can produce an even number of votes, we further evaluate a weighted majority rule for tie-breaking. We use a linear vote of the form $\mathrm { v o t e } = \mathbf { S U P } + \alpha$ · ELIM and search over α for 30 trials, reporting the best achievable accuracy for each ensemble. As shown in Table C.2, even under this optimized aggregation scheme, the dual StateSwap ensemble remains stronger than direct SUP–ELIM ensembling. This indicates that the StateSwap gains cannot be explained away by a better voting heuristic alone.

<table><tr><td>Method</td><td>Qwen-2.5-7B Acc.</td><td>GLM-4-9B Acc.</td></tr><tr><td colspan="3">2-Run Ensemble</td></tr><tr><td>ELIM-only</td><td>0.7436</td><td>0.7354</td></tr><tr><td>ELIM(SUP State)</td><td>0.7745</td><td>0.7792</td></tr><tr><td>SUP-only</td><td>0.7712</td><td>0.7591</td></tr><tr><td>SUP(ELIM State)</td><td>0.8026</td><td>0.7682</td></tr><tr><td colspan="3">4-Run Ensemble</td></tr><tr><td>SUP + SUP(ELIM State)</td><td>0.7939</td><td>0.7692</td></tr><tr><td>ELIM + ELIM(SUP State)</td><td>0.7844</td><td>0.7861</td></tr><tr><td>SUP + ELIM SUP(ELIM State) +</td><td>0.8020</td><td>0.7890</td></tr><tr><td>ELIM(SUP State)</td><td>0.8156</td><td>0.7949</td></tr></table>

Table C.1: Ensemble ablation results. The 2-run setting reports deterministic greedy ensembles within a single framing or StateSwap condition. The 4-run setting combines two 2-run groups. StateSwap-based ensembles outperform the naive SUP+ELIM ensemble on both models.
<table><tr><td>Method</td><td>Qwen-2.5-7B Acc.</td><td>GLM-4-9B Acc.</td></tr><tr><td>4-Run Ensemble with Weighted Voting</td><td></td><td></td></tr><tr><td>SUP + SUP(ELIM State)</td><td>0.8008</td><td>0.7732</td></tr><tr><td>ELIM + ELIM(SUP State)</td><td>0.7943</td><td>0.7861</td></tr><tr><td>SUP + ELIM</td><td>0.8059</td><td>0.7939</td></tr><tr><td>SUP(ELIM State) + ELIM(SUP State)</td><td>0.8225</td><td>0.8018</td></tr></table>

Table C.2: Weighted-vote ensemble ablation. We search over the linear tie-breaking weight α for 30 trials and report the best accuracy. StateSwap-based ensembling remains stronger than direct SUP+ELIM aggregation.

Per-task MMLU accuracy Table C.3 reports per-task accuracy under the SUP and ELIM framings. Compared with the Base setting, the Sub variant achieves consistent or improved performance across most MMLU categories.

## C.2 Decision Determinacy Behavior

Beyond answer accuracy, we analyze whether [STATE] substitution alters the model’s determinacy in eliminating incorrect options, using the error-elimination metric defined in $\mathsf { A p - }$ pendix B.5.3.

As shown in Table C.4, right shifts consistently occur more frequently than left shifts across models and datasets, indicating that [STATE] substitution more often strengthens the model’s ability to eliminate incorrect options, rather than relaxing it.

This asymmetric shift aligns with the gains observed in ACC Elim and Jaccard (Table 1), suggesting that the observed accuracy improvements are driven by enhanced error-elimination capability, manifested as more decisive and systematic suppression of incorrect options, rather than random variation or superficial answer changes.

## C.3 Detailed Steering Results

This experiment isolates how a population-level steering direction changes when the contrast is constructed from instructions rather than answer options. It contains no [STATE] token and does not perform instance-level state substitution.

Matched contrast construction For calibration example $i ,$ let $p _ { i } ^ { \mathsf { S U P } }$ request the correct option, let $p _ { i } ^ { \mathrm { E L I M } }$ request an incorrect option, and let $y _ { i }$ be the correct answer letter. We pair $y _ { i }$ with a deterministic incorrect letter $\widetilde { y } _ { i }$ , chosen as the first letter in $\{ A , B , C , D \}$ that differs from $y _ { i }$ . Let $h _ { l } ( p , y )$ denote the residual-stream activation at the appended answer token $y$ after layer l. Standard CAA holds the instruction fixed and changes the completion:

$$
v _ { l } ^ { \mathrm { C A A } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ h _ { l } ( p _ { i } ^ { \mathrm { S U P } } , y _ { i } ) - h _ { l } ( p _ { i } ^ { \mathrm { S U P } } , \widetilde { y } _ { i } ) \right] .\tag{2}
$$

The dual-framing construction holds the completion fixed and changes the instruction:

$$
v _ { l } ^ { \mathrm { D F } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ h _ { l } ( p _ { i } ^ { \mathrm { S U P } } , y _ { i } ) - h _ { l } ( p _ { i } ^ { \mathrm { E L I M } } , y _ { i } ) \right] .\tag{3}
$$

We use N = 50 calibration examples for both directions. The two contrasts share the positive pair $( p _ { i } ^ { \mathrm { S U P } } , y _ { i } )$ . Their negative members are complementary counterfactuals: CAA preserves the instruction but forces an inconsistent option, whereas dual framing preserves the option but forces an inconsistent instruction–completion pair. Neither negative member is treated as a valid answer to its instruction. Both serve only to define a matched activation difference.

Steering and evaluation protocol We independently $\ell _ { 2 } .$ -normalize each layer-wise direction, giving $\widehat { v } _ { l } = v _ { l } / \| v _ { l } \| _ { 2 }$ . For one layer at a time, we modify the residual stream at the answer boundary as

$$
h _ { l } \gets h _ { l } + a \widehat { v } _ { l } , \qquad a \in \{ - 2 , - 1 , 0 , 1 , 2 \} .\tag{4}
$$

<table><tr><td rowspan="2">Task</td><td colspan="4">Qwen-2.5-7B</td><td colspan="4">GLM-4-9B</td></tr><tr><td>Base-Elim</td><td>Sub-Elim</td><td>Base-Sup</td><td>Sub-Sup</td><td>Base-Elim</td><td>Sub-Elim</td><td>Base-Sup</td><td>Sub-Sup</td></tr><tr><td>abstract_algebra_proofs</td><td>63.27</td><td>61.62</td><td>63.64</td><td>66.00</td><td>65.31</td><td>65.31</td><td>59.60</td><td>70.71</td></tr><tr><td>bioethics</td><td>76.01</td><td>79.04</td><td>78.31</td><td>80.88</td><td>81.18</td><td>86.40</td><td>82.35</td><td>87.50</td></tr><tr><td>business_ethics</td><td>71.72</td><td>71.00</td><td>68.00</td><td>68.00</td><td>69.39</td><td>73.47</td><td>68.69</td><td>74.75</td></tr><tr><td>clinical_knowledge</td><td>75.19</td><td>74.05</td><td>69.96</td><td>69.81</td><td>74.90</td><td>78.03</td><td>73.86</td><td>76.23</td></tr><tr><td>college_math</td><td>70.97</td><td>69.47</td><td>61.05</td><td>63.00</td><td>63.04</td><td>72.34</td><td>58.95</td><td>61.46</td></tr><tr><td>formal_logic</td><td>64.46</td><td>63.64</td><td>62.81</td><td>63.93</td><td>61.29</td><td>68.25</td><td>64.80</td><td>65.08</td></tr><tr><td>hs_math</td><td>86.92</td><td>88.67</td><td>75.00</td><td>77.61</td><td>81.42</td><td>83.90</td><td>62.20</td><td>70.45</td></tr><tr><td>jurisprudence</td><td>71.30</td><td>71.96</td><td>69.44</td><td>68.52</td><td>80.37</td><td>79.63</td><td>75.93</td><td>75.00</td></tr><tr><td>logical_fallacies</td><td>81.51</td><td>80.13</td><td>81.87</td><td>84.28</td><td>77.07</td><td>85.19</td><td>83.23</td><td>83.44</td></tr><tr><td>medical_genetics</td><td>84.38</td><td>84.85</td><td>80.00</td><td>80.00</td><td>85.00</td><td>86.00</td><td>80.00</td><td>81.00</td></tr><tr><td>moral_disputes</td><td>61.70</td><td>62.39</td><td>63.85</td><td>65.51</td><td>69.01</td><td>72.09</td><td>63.58</td><td>65.90</td></tr><tr><td>philosophy</td><td>64.61</td><td>65.48</td><td>71.70</td><td>73.31</td><td>68.93</td><td>71.06</td><td>63.67</td><td>69.45</td></tr><tr><td>professional_law</td><td>47.22</td><td>48.92</td><td>50.20</td><td>50.36</td><td>54.46</td><td>57.67</td><td>53.82</td><td>55.71</td></tr><tr><td>sociology</td><td>73.06</td><td>71.94</td><td>71.28</td><td>71.43</td><td>76.02</td><td>84.26</td><td>72.73</td><td>75.00</td></tr><tr><td>us_history</td><td>85.05</td><td>85.20</td><td>80.20</td><td>82.14</td><td>84.92</td><td>89.11</td><td>85.93</td><td>87.13</td></tr><tr><td>world_history</td><td>81.62</td><td>79.24</td><td>84.55</td><td>82.63</td><td>84.85</td><td>86.38</td><td>86.38</td><td>86.44</td></tr><tr><td>world_religions</td><td>85.71</td><td>84.52</td><td>82.53</td><td>82.63</td><td>84.02</td><td>86.90</td><td>81.66</td><td>82.25</td></tr></table>

Table C.3: Extended per-task results on the MMLU benchmark. We report accuracy under the two framings (Elim and Sup), each evaluated with the original model (Base) and with [STATE] substitution (Sub). This table complements the main results by providing fine-grained, task-level comparisons. All values are percentages(%); higher is better.

<table><tr><td>Model</td><td>Dataset</td><td>Right Shift</td><td></td><td>Left Shift</td></tr><tr><td rowspan="2">Qwen-2.5-7B</td><td>MMLU</td><td>7.25%</td><td>&gt;</td><td>7.17%</td></tr><tr><td>MedQA</td><td>11.31%</td><td>&gt;</td><td>9.29%</td></tr><tr><td rowspan="2">GLM-4-9B</td><td>MMLU</td><td>9.75%</td><td>&gt;</td><td>7.52%</td></tr><tr><td>MedQA</td><td>15.24%</td><td>&gt;</td><td>11.39%</td></tr></table>

Table C.4: Directional changes in eliminated-incorrect mass from baseline to substitution (proportions over all instances). Right Shift indicates an increase in eliminated-incorrect mass, while Left Shift indicates a decrease.

Both methods use identical hooks, prompt templates at evaluation, coefficient values, and nexttoken scoring over $\{ A , B , C , D \}$ . We evaluate all 1,015 MedQA-CH test questions with Qwen-2.5- 7B and GLM-4-9B. Accuracy at a = 0 is the unsteered reference and is omitted from Table C.5. We sweep Layers 1–26 for Qwen and Layers 1–35 for GLM. Consequently, the comparison changes only whether the calibration contrast perturbs the answer option or the instruction.

Aggregate comparison Averaged across the evaluated layers, the dual-framing direction retains higher accuracy than CAA at $a = - 2 , - 1$ , and 2. The respective dual-framing and CAA means are 59.03 vs. 37.16, 69.67 vs. 48.85, and 68.15 vs. 60.35 for GLM. They are 71.21 vs. 52.26, 81.61 vs. 75.97, and 74.77 vs. 62.90 for Qwen. At a = 1, the means are nearly identical: 74.05 vs. 74.07 for GLM and 83.37 vs. 83.52 for Qwen.

The clearest stability difference occurs at $a =$ −1. Relative to CAA, the layer-wise range narrows from 50.54 to 19.90 points for GLM and from 60.89 to 28.08 points for Qwen. At a = 1, the range is similar for GLM (10.93 for CAA and 9.16 for dual framing) and narrower under dual framing for Qwen (27.00 vs. 12.41). Strong coefficients still cause sharp failures at individual layers. Higher retained accuracy also measures robustness to intervention, not greater steering strength by itself. The evidence therefore supports a more bounded layerwise response under this matched protocol, rather than general superiority over CAA. Table C.5 reports every layer–coefficient result.

## D Representation and Window Analyses

## D.1 Additional Visualization of Layer-wise Separability

We visualize [STATE] representations using principal component analysis (PCA) to illustrate their layer-wise geometry under the two framings.

For Qwen-2.5-7B, we extract residual-stream activations at the [STATE] position using the same contiguous feature window (w = 6) as in the separability analysis. At each layer, PCA is applied independently to pooled representations from both framings, and the first three principal components are visualized.

Figure D.1 shows substantial overlap in early layers, clearer separation in intermediate layers, and reduced separation in later layers, matching the non-monotonic depth-wise trend in Figure D.2.

<table><tr><td rowspan="2">Layer</td><td colspan="4">Two-framing</td><td colspan="4">CAA</td></tr><tr><td>a = −2</td><td> $a = - 1$ </td><td>a = 1</td><td>a = 2</td><td>a = −2</td><td>a = −1</td><td>a = 1</td><td> $a = 2$ </td></tr><tr><td colspan="9">GLM-4-9B</td></tr><tr><td></td><td>74.68</td><td>74.19</td><td>74.38</td><td>74.38</td><td>21.48</td><td>46.60</td><td>69.95</td><td>26.90</td></tr><tr><td></td><td>73.4340</td><td>73.89</td><td>74.68</td><td>74.68</td><td>22.46</td><td>58.13</td><td>72.22</td><td>48.87</td></tr><tr><td></td><td></td><td>73.50</td><td>74.29</td><td>0.00</td><td>22.17</td><td>50.15</td><td>75.67</td><td>0.00</td></tr><tr><td></td><td>74.88</td><td>74.48</td><td>74.09</td><td>73.79</td><td>21.48</td><td>37.14</td><td>75.76</td><td>69.75</td></tr><tr><td></td><td>74.19</td><td>13.17</td><td>74.19</td><td>73.89</td><td>23.94</td><td>50.44</td><td>74.38</td><td>67.98</td></tr><tr><td></td><td>75.76</td><td></td><td>73.20</td><td>73.00</td><td>21.48</td><td>33.20</td><td>74.09</td><td>48.87</td></tr><tr><td></td><td>74.48</td><td>75.37</td><td>73.69</td><td>73.30</td><td>21.48</td><td>35.67</td><td>70.54</td><td>38.03</td></tr><tr><td></td><td>61.48</td><td>69.56</td><td>72.91</td><td>69.46</td><td>21.48</td><td>32.91</td><td>71.43</td><td>47.88</td></tr><tr><td></td><td>60.79</td><td>69.26</td><td>74.38</td><td>71.13</td><td>21.48</td><td>32.91</td><td>68.77</td><td>45.62</td></tr><tr><td></td><td>59.51</td><td>68.37</td><td>74.98</td><td>71.63</td><td>21.48</td><td>25.91</td><td>69.75</td><td>52.91</td></tr><tr><td>0112134151617181920</td><td>55.27</td><td>70.05</td><td>72.61</td><td>60.99</td><td>21.48</td><td>30.94</td><td>68.77</td><td>49.75</td></tr><tr><td></td><td>59.70</td><td>71.23</td><td>71.63</td><td>63.94</td><td>21.48</td><td>40.30</td><td>72.12</td><td>65.02</td></tr><tr><td></td><td>45.81</td><td>67.88</td><td>73.10</td><td>64.83</td><td>21.48</td><td>37.04</td><td>69.46</td><td>55.17</td></tr><tr><td></td><td>42.17</td><td>67.98</td><td>73.30</td><td>62.56</td><td>21.48</td><td>40.49</td><td>71.92</td><td>67.00</td></tr><tr><td></td><td>33.79</td><td>60.30</td><td>71.13</td><td>61.67</td><td>21.48</td><td>37.24</td><td>73.79</td><td>66.80</td></tr><tr><td></td><td>40.59</td><td>61.38</td><td>67.98</td><td>51.43</td><td>21.48</td><td>36.16</td><td>73.60</td><td>67.68</td></tr><tr><td></td><td>36.26</td><td>65.42</td><td>69.16</td><td>52.71</td><td>21.48</td><td>36.75</td><td>73.69</td><td>58. .33</td></tr><tr><td></td><td>45.12</td><td>67.29</td><td>70.54</td><td>47.88</td><td>21.48</td><td>35.96</td><td>73.00</td><td>61.38</td></tr><tr><td></td><td>40.00</td><td>69.06</td><td>70.05</td><td>41.87</td><td>21.58</td><td>33.30</td><td>73.10</td><td>58.82</td></tr><tr><td></td><td>28.08</td><td>56.95</td><td>71.92</td><td>64.14</td><td>21.48</td><td>32.71</td><td>72.02</td><td></td></tr><tr><td></td><td>41.38</td><td>55.47</td><td>77.04</td><td>73.99</td><td></td><td>37.04</td><td>75.47</td><td>44.53</td></tr><tr><td>21223</td><td>38.62</td><td>63.74</td><td>76.35</td><td>74.09</td><td></td><td>37.93</td><td>77.24</td><td>42.17</td></tr><tr><td></td><td>43.05</td><td>66.11</td><td>75.47</td><td>76.55</td><td></td><td>38.72</td><td>77.44</td><td>57.24</td></tr><tr><td></td><td>38.13</td><td>60.89</td><td>76.55</td><td>76.45</td><td>2142 25.42</td><td>39.90</td><td></td><td>63.35</td></tr><tr><td></td><td>55.96</td><td>68.08</td><td>77.14</td><td></td><td>40.69</td><td></td><td>79.70</td><td>61.97</td></tr><tr><td></td><td></td><td>74.09</td><td>76.65</td><td>79.11</td><td></td><td>54.48</td><td>71.13</td><td>66.80</td></tr><tr><td>5242526728293032</td><td>71.72</td><td>73.99</td><td>75.96</td><td>79.41</td><td>69.75</td><td>71.13</td><td>79.01</td><td>74.19</td></tr><tr><td></td><td>71.72</td><td></td><td>75.86</td><td>78.72</td><td>75.86</td><td>72.71</td><td>79.11</td><td>79.41</td></tr><tr><td></td><td>70.74</td><td>73.50</td><td>75.57</td><td>78.62</td><td>73.10</td><td>72.71</td><td>78.23</td><td>80.20</td></tr><tr><td></td><td>73.60</td><td>74.38</td><td></td><td>78.03</td><td>77.83</td><td>76.45</td><td>76.75</td><td>79.41</td></tr><tr><td></td><td>72.22</td><td>73.89</td><td>75.67</td><td>77.14</td><td>74.68</td><td>74.58</td><td>77.14</td><td>80.30</td></tr><tr><td></td><td>71.63</td><td>73.50</td><td>75.67</td><td>77.34</td><td>73.40</td><td>73.99</td><td>77.83</td><td>80.10</td></tr><tr><td>3233435</td><td>71.53</td><td>73.50</td><td>29</td><td>77.14</td><td>74.38</td><td>74.68</td><td>75.07</td><td>76.75</td></tr><tr><td></td><td>2</td><td>73.89</td><td></td><td>77.54</td><td>2.81 72</td><td>73.89</td><td>74.19</td><td></td></tr><tr><td></td><td></td><td>74.09</td><td></td><td>77.34</td><td>14.75</td><td>74.88</td><td>74.78</td><td>76.16 75.96</td></tr><tr><td></td><td></td><td>73.60</td><td></td><td>76.35</td><td></td><td>72.61</td><td>75.17</td><td>76.95</td></tr><tr><td colspan="9"></td></tr><tr><td></td><td>85.52</td><td>85.71</td><td>86.21</td><td>Qwen-2.5-7B 85.2.</td><td>29.66</td><td>67.19</td><td>81.28</td><td>29.85</td></tr><tr><td></td><td>85.81</td><td>86.11</td><td>85.81</td><td></td><td>37.93</td><td>67.78</td><td>79.90</td><td>35.76</td></tr><tr><td></td><td>86.21</td><td>85.91</td><td>85.71</td><td>85.02</td><td>33.20</td><td>78.13</td><td>82.76</td><td>62.17</td></tr><tr><td></td><td>86.31</td><td>86.11</td><td>84.33</td><td>83.35</td><td>44.04</td><td>80.79</td><td>84.43</td><td>73.89</td></tr><tr><td></td><td>85.62</td><td></td><td>84.93</td><td>83.94</td><td>59.51</td><td>83.74</td><td>83.98</td><td>77.34</td></tr><tr><td>-12345678911</td><td>85.22 84.33</td><td>9</td><td>85.52 84.53</td><td>84.24</td><td>58.42 54.58</td><td>82.76 81.67</td><td>82.56 83.94</td><td>73.69</td></tr></table>

Table C.5: Detailed MedQA-CH accuracy under mean-difference activation steering. DUAL holds the answer option fixed and contrasts the SUP and ELIM instructions. CAA holds the instruction fixed and contrasts the correct and paired incorrect options. Results use steering coefficients a $\in \{ - 2 , - 1 , 1 , 2 \}$ . The upper block reports GLM-4-9B results, while the lower block reports Qwen-2.5-7B results. The unsteered case $a = 0$ is omitted for clarity.

<table><tr><td>Layers</td><td>Cos↓</td><td>IQR</td><td>p90</td><td>KL↑</td><td>IQR</td><td>p90</td></tr><tr><td>3-5</td><td>0.130</td><td>0.070</td><td>0.275</td><td>1.634</td><td>1.319</td><td>6.832</td></tr><tr><td>13-15</td><td>0.108</td><td>0.085</td><td>0.295</td><td>1.000</td><td>0.943</td><td>3.745</td></tr><tr><td>11-20</td><td>0.108</td><td>0.066</td><td>0.257</td><td>1.487</td><td>1.198</td><td>6.253</td></tr><tr><td>25-28</td><td>0.131</td><td>0.069</td><td>0.275</td><td>1.636</td><td>1.317</td><td>6.832</td></tr></table>

Table D.1: Window comparison under representation substitution. Lower cosine distance indicates smaller global distortion in logit space; when paired with nontrivial KL shifts, we use it as a proxy for reduced collateral perturbation. IQR and p90 summarize dispersion and tail heaviness. Values are averaged across swap directions.

PCA is performed independently at each layer; hence the resulting axes are not comparable across layers.

## D.2 Analysis of the Intervention Window

This subsection compares candidate layer windows using output-level statistics under representation substitution as a post-hoc assessment of the separability-selected intervention window.

Analysis Metrics We measure (i) cosine distance between the original and intervened logit vectors (global geometric change), and (ii) $\mathrm { K L } ( q \| p )$ between the corresponding output distributions (probability mass reallocation). For each metric we report the mean and two stability summaries: IQR and the 90th percentile (p90), where larger p90 indicates heavier tails. p90 corresponds to the 90th percentile of per-example distances; larger p90 implies more extreme cases among the top 10% most affected examples.

Layer Windows Discussion Table D.1 aggregates results across swap directions (directionspecific trends are qualitatively consistent with the conclusions). Shallow (3–5) and late (25–28) layers behave nearly identically on both metrics, yielding larger cosine distances (0.130–0.131) and the largest KL values (1.634–1.636) with heavier KL tails (p90=6.832). This suggests strong distributional shifts accompanied by more extreme probability reallocations among the most affected examples.

The intermediate-layer peak (13–15) exhibits the largest dispersion and tail heaviness in cosine distance (IQR=0.085, p90=0.295) while producing the smallest KL magnitude and tails (KL mean=1.000, $\mathrm { p } 9 0 { = } 3 . 7 4 5 )$ , indicating a mismatch between geometric logit changes and distributional reallocation under this window.

In contrast, the intermediate window (11–20) shows the most favorable stability–effect profile under our criteria: mean cosine distance is similar for 11–20 and 13–15 (both ≈ 0.108), but 11–20 achieves the smallest cosine dispersion and lightest cosine tails $( \mathrm { I Q R } { = } 0 . 0 6 6 , \mathrm { p } 9 0 { = } 0 . 2 5 7 )$ , while maintaining substantial KL shifts (mean=1.487) and a lighter KL tail compared to 3–5/25–28 (p90 6.253 vs. 6.832). A smaller KL p90 implies less extreme probability reallocations among the top 10% most affected instances, which is desirable for a robust and predictable intervention interface.

Conclusion Overall, the separability-selected window 11–20 also provides the best stability– effect trade-off under output-level statistics: it avoids large and heavy-tailed geometric perturbations (cosine) while still inducing meaningful distributional changes (KL) without the heaviest tails observed at the layer-range extremes. This analysis is post hoc: the output-level statistics are not used to select the intervention window or tune it for task performance.

## D.3 Cross-Model Separability Diagnostics at the [STATE] Interface

To complement the PCA visualization in Section 5.1, we report the full layer-wise separability diagnostics for both Qwen-2.5-7B and GLM-4-9B. For each model, we use the same paired Sup/Elim construction, compute Cohen’s d over residual-stream features at the [STATE] position, and apply the same sliding-window procedure over feature dimensions. We report both the label-free and label-informed variants of the normalized separability heatmaps.

Figure D.2 shows that the two models exhibit a consistent layer-wise pattern. In both Qwen-2.5- 7B and GLM-4-9B, separability is weak in early layers, becomes most pronounced in intermediate layers, and attenuates toward later layers. This cross-model consistency supports the conclusion that framing-dependent structure at the [STATE] interface is concentrated in an intermediate-layer region rather than uniformly distributed across network depth.

Following the same criterion used in the main text, we select model-specific intervention windows based solely on the observed separability patterns: layers 11–20 for Qwen-2.5-7B and layers 21–40 for GLM-4-9B. These windows are not tuned based on downstream task performance;

Layer-wise 3D Representation

![](images/4fd7282f6a53e6f2e74c0439f13a9e002ffdd9b46eae848f73508767fba586cc.jpg)  
Figure D.1: Layer-wise PCA of framing-conditioned [STATE] representations. For each layer of Qwen-2.5-7B, PCA is applied to pooled representations using the same feature window (w = 6) as in the separability analysis. Points are individual instances colored by framing. Intermediate-layer representations show clearer separation than early- or late-layer representations, consistent with Figure D.2.

they are used in subsequent experiments to test whether [STATE] substitution within the separable intermediate-layer region produces systematic behavioral changes.

![](images/cd6a156a6fa1d9bfe93e30ef3a09e2be9ecaf8419fc29a12f2235916dd1ff613.jpg)  
Figure D.2: Layer-wise normalized separability at the [STATE] position for Qwen-2.5-7B (top) and GLM-4- 9B (bottom). For each model, the heatmaps report labelfree Cohen’s d and the corresponding label-informed variant. Both models show weak early-layer separability, a pronounced intermediate-layer concentration, and attenuation toward later layers.

## D.4 Robustness of Window Localization

Window localization uses a calibration set of 50 examples. Each example is instantiated with two prompt realizations for each of the SUP and ELIM framings, yielding 200 framing-conditioned representations. The resulting model-specific intervention window is fixed before evaluation on the test split.

We assess the stability of this localization for Qwen-2.5-7B on an expanded sample of 1,000 examples. We generate 500 bootstrap resamples and recompute the layer-wise separability profile for each resample. Table D.2 reports the mean normalized score and the width of the 95% bootstrap confidence interval for the ten highest-scoring layers under each diagnostic.

Under both diagnostics, the highest-scoring layers cluster in the intermediate region, with most falling between Layers 11 and 20. The bootstrap results show that localization to this region is consistent across resamples, although the exact layer rankings differ between the two diagnostics.

## E Ablation Details and Controls

This section provides implementation details for the ablation experiments described in the main text.

<table><tr><td colspan="4">Label-free separability</td><td colspan="4">Label-informed separability</td></tr><tr><td>Rank</td><td>Layer</td><td>Mean</td><td>95% CI width</td><td>Rank</td><td>Layer</td><td>Mean</td><td>95% CI width</td></tr><tr><td>1</td><td>19</td><td>0.898</td><td>0.142</td><td>1</td><td>12</td><td>0.730</td><td>0.205</td></tr><tr><td>2</td><td>16</td><td>0.891</td><td>0.139</td><td>2</td><td>14</td><td>0.716</td><td>0.181</td></tr><tr><td>3</td><td>14</td><td>0.882</td><td>0.137</td><td>3</td><td>11</td><td>0.705</td><td>0.152</td></tr><tr><td>4</td><td>20</td><td>0.868</td><td>0.142</td><td>4</td><td>18</td><td>0.705</td><td>0.200</td></tr><tr><td>5</td><td>18</td><td>0.853</td><td>0.139</td><td>5</td><td>19</td><td>0.682</td><td>0.178</td></tr><tr><td>6</td><td>12</td><td>0.839</td><td>0.135</td><td>6</td><td>16</td><td>0.674</td><td>0.175</td></tr><tr><td>7</td><td>13</td><td>0.807</td><td>0.135</td><td>7</td><td>13</td><td>0.665</td><td>0.186</td></tr><tr><td>8</td><td>9</td><td>0.775</td><td>0.145</td><td>8</td><td>20</td><td>0.660</td><td>0.180</td></tr><tr><td>9</td><td>11</td><td>0.767</td><td>0.124</td><td>9</td><td>15</td><td>0.636</td><td>0.173</td></tr><tr><td>10</td><td>17</td><td>0.747</td><td>0.123</td><td>10</td><td>17</td><td>0.635</td><td>0.174</td></tr></table>

Table D.2: Bootstrap summaries of layer-wise separability for Qwen-2.5-7B. For each diagnostic, the table lists the ten layers with the highest mean normalized scores; confidence-interval widths are estimated from 500 bootstrap resamples of the 1,000-example sample.

All interventions operate on the residual-stream activations at the [STATE] position and are applied layerwise within the intervention window W, unless otherwise specified.

## E.1 Token and Interface Controls

<table><tr><td>Setting</td><td>Correct / Total</td><td>Accuracy</td></tr><tr><td>GLM-4-9B w/o token</td><td>812 / 1015</td><td>80.00%</td></tr><tr><td>With [STATE] token</td><td>810 / 1015</td><td>79.80%</td></tr><tr><td>Qwen-2.5-7B w/o token</td><td>882 / 1015</td><td>86.90%</td></tr><tr><td>With [STATE] token</td><td>885 / 1015</td><td>87.19%</td></tr></table>

Table E.1: MedQA-CH accuracy with and without the [STATE] token.

To test interface specificity, we compare substitution at [STATE] with otherwise identical substitution at the original prompt’s final content token. Substitution content and the layer window are held fixed in the interface comparison.

## E.2 Substitution content

We consider the following substitution types when modifying the [STATE] representation:

• Zeroing. The residual vector at the [STATE] position is replaced with a zero vector at each layer in W.

• Random replacement. The residual vector is replaced by a randomly sampled vector drawn from the empirical distribution of [STATE] activations at the same layer. Sampling is performed independently for each layer, and the sampled vector is fixed across forward passes within a run.

• Framing-conditioned substitution. The residual vector at the [STATE] position is replaced with a cached activation obtained from the complementary framing (Sup↔Elim) for the same input question. Substitution is applied layerwise within W, preserving the original activation outside the intervention window.

## E.3 Interface Position

To test whether substitution depends on the structural role of the interface, we compare the canonical final-position configuration with two misaligned placements:

$$
\begin{array} { r } { \mathbf { x } _ { 1 } = [ \mathsf { P A D } ] ^ { n } + I + Q + O + [ \mathsf { S T A I E } ] , } \\ { \mathbf { x } _ { 2 } = [ \mathsf { P A D } ] ^ { n } + I + Q + [ \mathsf { S T A T E } ] + O , } \\ { \mathbf { x } _ { 3 } = [ \mathsf { P A D } ] ^ { n } + I + [ \mathsf { S T A T E } ] + Q + O , } \end{array}
$$

where I, Q, and O denote the instruction, question, and answer-option segments, respectively. For all three configurations, we apply the same framingconditioned substitution over the same layer window W.

We count a generation as failed when it is incoherent or cannot be parsed into the required tasklevel answer representation. Across the two misaligned configurations, more than 95% of generations fail this criterion. By contrast, the canonical configuration preserves coherent, task-relevant generation for the majority of inputs. This result indicates that [STATE] is not an arbitrary writable slot: its effectiveness depends on being placed after the complete question and option context.

## E.4 Quantitative Similarity Summaries for Ablations

This appendix reports distributional summaries for the response-level similarity analyses shown in Figures 6a and 6b. For each condition, we compute length-aware cosine similarity (LAC) and BERTScore-F1 between the baseline output and the output under intervention, and report the median, interquartile range (IQR), and upper quantiles (P90/P95). The number of evaluated runs is denoted by N.

Substitution content: structured vs. unstructured interventions Table E.2 summarizes the similarity distributions for different substitution contents at the [STATE] position (Figure 6b), contrasting framing-conditioned substitution with content-agnostic controls.

<table><tr><td>Metric</td><td>Median</td><td>IQR</td><td>P90</td><td>P95</td></tr><tr><td>LAC</td><td></td><td></td><td></td><td></td></tr><tr><td>zeroing</td><td>0.6620</td><td>0.4913</td><td>0.9005</td><td>0.9351</td></tr><tr><td>random</td><td>0.6242</td><td>0.4933</td><td>0.8931</td><td>0.9141</td></tr><tr><td>cross-framing</td><td>0.8766</td><td>0.1925</td><td>1.0000</td><td>1.0000</td></tr><tr><td>BERTScore-F1</td><td></td><td></td><td></td><td></td></tr><tr><td>zeroing</td><td>0.5631</td><td>0.1196</td><td>0.6954</td><td>0.7554</td></tr><tr><td>random</td><td>0.5560</td><td>0.1289</td><td>0.6855</td><td>0.7460</td></tr><tr><td>cross-framing</td><td>0.6400</td><td>0.3449</td><td>1.0000</td><td>1.0000</td></tr></table>

Table E.2: Quantile summaries of response-level similarity for different substitution contents at the [STATE] position (Figure 6b).
<table><tr><td>Metric</td><td>Median</td><td>IQR</td><td>P90</td><td>P95</td></tr><tr><td>LAC</td><td></td><td></td><td></td><td></td></tr><tr><td>[STATE]</td><td>0.8766</td><td>0.1925</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Final content token</td><td>0.9780</td><td>0.0660</td><td>1.0000</td><td>1.0000</td></tr><tr><td>BERTScore-F1</td><td></td><td></td><td></td><td></td></tr><tr><td>[STATE]</td><td>0.6400</td><td>0.3449</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Final content token</td><td>0.9444</td><td>0.2330</td><td>1.0000</td><td>1.0000</td></tr></table>

Table E.3: Quantile summaries comparing substitutions at the [STATE] position and the final content token.

Intervention location: [STATE] vs. final-contenttoken substitution Table E.3 reports quantile summaries for substitutions applied at [STATE] versus at the final content token position (Figure 6a), using the same intervention window W and decoding procedure.