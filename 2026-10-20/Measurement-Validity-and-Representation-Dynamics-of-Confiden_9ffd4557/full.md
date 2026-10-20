# Measurement Validity and Representation Dynamics of Confidence Signals in Large Language Models

Debojeet Ghosh Shreyansh Kumar Ayushman Garg

## Abstract

Whether a language model “knows what it knows” matters for trustworthy deployment and for deciding when it should abstain. Prior work usually measures this through one channel: verbalized confidence, a linear probe on hidden states, or agreement among sampled answers (semantic entropy), and often treats these as interchangeable. We compare all three in a pre-registered audit of five Qwen2.5 models (0.5B–7B, including a base/instruct pair at 7B) on factual retrieval (PopQA, SimpleQA) and multi-step reasoning (GSM8K, MATH). First, probes trained on prefill states to predict whether sampled answers will disagree become informative earlier for retrieval (about 7% of layer depth on average) than for reasoning (about 28%). The pre-registered five-layer test supported this only marginally; a finer 21- layer follow-up, designed after those results, supports it more clearly. In most retrieval cells, however, a prompt-only TF-IDF classifier matches the probe, so the early signal may reflect surface features of the question. Second, verbalized confidence is unstable: slightly diferent prompt templates yield scores that barely correlate. Third, Expected Calibration Error is misleading: verbal confidence looks well calibrated because it clusters near the base rate while discriminating poorly, whereas sampling agreement has worse ECE but about twice the resolution. Of five pre-registered hypotheses, two were falsified, one was partly supported, one was not supported, and one could not be tested because the base model’s verbal confidence was degenerate.

## 1 Introduction

Large language models (LLMs) demonstrate impressive capabilities across factual retrieval, mathematica problem-solving, and symbolic logic. However, their tendency to produce plausible but factually erroneous statements, often stated with high certainty, poses risks in high-stakes environments such as clinical decision support, legal analysis, and automated code generation. To mitigate these risks, a substantial literature has investigated methods for estimating model confidence to enable selective abstention or routing to human supervisors.

Current approaches to LLM confidence estimation fall predominantly into three methodological paradigms:

1. Verbalized Elicitation: Querying the model directly to state its confidence as a probability percentage (0 to 100%) or qualitative certainty category within the prompt context.

2. Internal Representation Probing: Training linear classifiers on the model’s residual stream hidden representations during the forward pass over the prompt.

3. Behavioral Sampling Consistency: Generating multiple temperature-perturbed rollouts and measuring the semantic entropy or answer consensus across equivalent clusters.

Empirical studies frequently treat these three channels as interchangeable estimators of a unified epistemic state. In this paper, we conduct a pre-registered, auditable empirical investigation across a 30-cell grid (5 model architectures × 6 task tiers) to test whether confidence representations are coherent, interchangeable, and mechanistically uniform.

That assumption fails on each dimension we measured. Verbal confidence does not generalize across minor prompt alterations; standard calibration metrics mask large diferences in discriminative sorting power; and the depth at which internal probes become informative difers between associative recall and multistep reasoning tasks, although for recall a prompt-only baseline explains much of the early signal. Raw verbal confidence is nearly uncorrelated with both behavioral and internal signals (pooled ρ = 0.086), while behavioral and internal signals agree moderately in most cells (Appendix A).

## 2 Related Work

Verbal Confidence & Elicitation Fragility. Eliciting verbal self-confidence via prompting has been explored extensively (Kadavath et al., 2022; Lin et al., 2022; Tian et al., 2023). Verbalized confidence is sensitive to elicitation strategy and prompt framing (Xiong et al., 2024; Geng et al., 2024), and auxiliary models can calibrate an LLM from its generations alone (Ulmer et al., 2024). Most existing benchmarks evaluate verbal confidence using a single prompt template, implicitly presuming measurement invariance across phrasing.

Probing Hidden Representations. Probing internal representations for truthfulness and confidence traces back to the geometry of hidden states (Burns et al., 2023; Azaria & Mitchell, 2023; Marks & Tegmark, 2023; Zou et al., 2023). Hidden states can carry truthfulness and answerability signals that the generated text does not express (Li et al., 2023; Slobodkin et al., 2023). Prior investigations frequently evaluate layer depth on coarse discrete grids (e.g., 25% increments).

Semantic Entropy & Behavioral Consistency. Kuhn et al. (2023) and Farquhar et al. (2024) demonstrated that semantic entropy computed across clustered stochastic generations outperforms raw token logprobabilities for hallucination detection, and Wang et al. (2023) showed that self-consistency rollouts improve reasoning accuracy. The closest prior work compares verbalized confidence, sequence probability, P(True) and hidden-state probes on shared items (Mahaut et al., 2024), and DECK (Chauhan, 2026) proposes a consistency × confidence taxonomy structurally similar to our quadrant analysis (Section 7). We position this paper as a measurement-validity study rather than a discovery: it reports where a three-channel comparison holds up and where it does not.

Proper Scoring Rules & Murphy Decomposition. In meteorological forecasting and statistical decision theory, Murphy (1973), Bröcker (2009), and Gneiting & Raftery (2007) showed that the standard mean squared error of probabilistic forecasts (Brier score) decomposes into three orthogonal components: Reliability (calibration error), Resolution (sorting power), and Uncertainty (inherent task unpredictability). Standard Expected Calibration Error (Guo et al., 2017; Nixon et al., 2019) measures only reliability, conflating well-calibrated uninformative forecasts with genuinely discriminative predictions.

## 3 Experimental Setup & Pre-Registered Protocol

To guard against post-hoc outcome selection, our experimental design, hypotheses H0–H4, pass rules and audit gates were committed in advance in a version-controlled protocol. One analysis in this paper is not pre-registered: the 21-tap depth sweep and its 2-consecutive onset rule (Section 4.1), which were specified after the pre-registered five-tap results were known. We label it as a follow-up throughout.

## 3.1 The 30-Cell Grid Architecture

The study systematically evaluates five models from the Qwen2.5 open-weights family across six benchmark tiers:

• Models (5): Qwen2.5-0.5B-Instruct (24 layers), Qwen2.5-1.5B-Instruct (28 layers), Qwen2.5-3B-Instruct (36 layers), Qwen2.5-7B-Base (28 layers), and Qwen2.5-7B-Instruct (28 layers).

• Retrieval Tiers (3): PopQA, top popularity quintile (R1); PopQA, bottom popularity quintile (R2); and SimpleQA (R3; adversarially constructed short-answer questions).

• Reasoning Tiers (3): GSM8K (C1; grade-school multi-step arithmetic), MATH levels 1–2 (C2), and MATH levels 4–5 (C3). Each cell holds 500 questions sampled at a fixed seed and split 60/20/20 into train, calibration and test.

This gives a grid of 30 cells. Some earlier studies drop cells whose accuracy falls outside a chosen range. We kept all 30 and instead recorded whether each cell’s pilot accuracy fell between 25% and 80%.

## 3.2 Probe Tap Specification & Architectural Implementation

We captured activations during a single forward pass over the prompt, before the model generated anything. For each model we read the token embeddings (0% depth) and the output of each transformer block at the chosen depths.

The probe reads the hidden state at the last prompt token, skipping any padding. The capture was switched on before the prompt was processed and switched of before decoding began, so generated tokens could not leak into the captured representations. An earlier version of the pipeline had this bug: the hooks stayed active and recorded output tokens. It was fixed, and a test in the repository now checks it. Instruct models received prompts formatted with their oficial chat template, and the base model received raw prompt text.

## 3.3 Pre-Registered Safeguards and Gates

The pipeline enforced four pre-registered audit gates:

Gate 1 (Grading Validity). A human audit of 200 graded outputs, stratified by grader (about 50 each for alias-match, NLI, numeric and symbolic grading), agreed with automated grading on 195/200 (97.5%; threshold 95%). All 5 discrepancies were the symbolic grader under-crediting correct math answers because of formatting (4 in C2, 1 in C3, 0 in retrieval), so symbolic-grader agreement was 90% and reasoning accuracy is slightly understated.

Gate 2 (Verbal Generalization). Required pairwise Spearman rank correlation $\rho \ge 0 . 6 0$ across three elicitation formats (Format A: integer 0–100; Format B: five ordinal labels; Format C: answer-orpass decision).

Gate 3 (Probe Validity). Probes were trained on the train split and evaluated on the calibration split. A cell passes if some depth reaches $\mathrm { A U R O C } \geq 0 . 6 5$ while beating both a 20-draw label-shufle null and a prompt-only TF-IDF baseline, and if the $p _ { 0 }$ negative control holds: the probe at 0% depth (the token embedding, which carries no question-specific computation at the final prompt token) must score AUROC 0.50 ± 0.10. 7b-base\_\_C3 failed the negative control $( p _ { 0 } = 0 . 6 3 4 )$ and was dropped.

Gate 4 (Model Comparison, H3). Requires matched base/instruct items carrying both a verbal and a behavioral or internal signal. Format parsing was not the problem (the base model parsed $\mathrm { { o n } \geq 9 9 \% }$ of items in every format). Its Format B answers, however, used essentially two labels (CERTAIN and NO\_IDEA), so the pre-registered verbal-degeneracy guard (≥ 3 distinct values per cell) excluded its verbal signal in all six tiers. With 0 matched rows, H3 declined to test rather than report a misleading 0.0% rate.

## 4 Representation Depth: Reasoning vs. Retrieval (H4)

Our mechanistic hypothesis (H4) was that probes begin to predict better than chance at an earlier layer for retrieval than for reasoning, and at an earlier layer in larger models. The pre-registered protocol trains probes on how consistent the model is with itself, not on whether its answer is correct. For each question we sample 10 answers and compute their semantic entropy, which measures how much the answers difer in meaning. A question is labeled “inconsistent” if its entropy is above the median for its cell and “consistent” otherwise. A probe’s onset is the first layer at which it predicts this label better than chance, using the model’s state at the last prompt token, before any answer is generated.

Pilot accuracy per cell (bold = in 25%–80% target band; dotted = out-of-band covariate)  
![](images/34e10b3ae6806e063df316a4c42f71e84251856aa2acae926bd24044c1e7f548.jpg)  
Figure 1: Complete 30-cell evaluation matrix. Pilot accuracy per (model × tier) cell. Dotted outlines indicate cells outside the 25% to 80% accuracy band.

## 4.1 Continuous 21-Tap Probe Sweeps & the 2-Consecutive Rule

Pre-registered test. The protocol specified five depth taps (0, 25, 50, 75, 100%), a single-point onset (the first tap with $\mathrm { A U R O C } \geq 0 . 6 5 )$ , and a pass rule requiring the bootstrap CI on the reasoning−retrieval onset diference to exclude zero. On the coarse run this test passed: $\Delta = + 1 3 . 6 4 \mathrm { p p } \ [ 9 5 \% \mathrm { C I } ; + 2 . 2 7 , + 2 7 . 2 7 ]$ over 22 probe-valid cells. Our internal audit judged it fragile, because onsets could take only three values ({25, 50, 75}), an exact one-sided permutation test gave $p = 0 . 0 5 2$ , and dropping the two C1 cells at 75% reduced ∆ to 6.06 pp.

Follow-up (not pre-registered). We therefore re-ran the probe stages over the same generations at 21 depth percentiles $( 0 \% , 5 \% , \ldots , 1 0 0 \% )$ , with a 2-consecutive confirmation rule specified and committed before the sweep ran, but after the coarse results were known. A depth percentile p is the onset if:

$$
\operatorname { A U R O C } ( p ) \geq 0 . 6 5 \quad { \mathrm { a n d } } \quad \operatorname { A U R O C } ( p + 5 { \% } ) \geq 0 . 6 5 .\tag{1}
$$

Across the 28 cells with probe fits (the 0.5B R1 and R2 cells had too few usable rows), 588 logistic-regression probes were fitted (28 cells × 21 depth percentiles); 24 cells passed Gate 3.

## 4.2 The Emergence Gap: Full 24-Cell Distribution

Table 1 reports all 24 cells that passed Gate 3 in the follow-up sweep (11 retrieval and 13 reasoning cells), together with the prompt-only TF-IDF baseline AUROC and whether the probe at the onset layer beats it.

The distributions are:

• Retrieval onsets ([5, 5, 5, 5, 5, 5, 5, 5, 5, 15, 15]%): nine of eleven cells trigger at 5% depth (Layer 1 or 2); mean 6.82%, median 5.0%.

• Reasoning onsets ([5, 5, 5, 5, 5, 15, 30, 40, 40, 45, 50, 60, 60]%): mean 28.08%, median 30.0%.

![](images/cb8da4b5d9ad791ff426892a07db2528129357cc5559dbf1918e79c4757bdeda.jpg)

Figure 2: AUROC across 21 layer depth percentiles. Calibration-split AUROC of linear probes predicting the median-split semantic-entropy label from prefill hidden states, faceted by model size. Retrieval tiers (cool lines) mostly cross 0.65 by 5% depth; reasoning tiers (warm lines) cross later on average. In 9 of 11 retrieval cells the onset-layer probe does not beat the prompt-only TF-IDF baseline (Table 1).  
Table 1: Onset depth for all 24 probe-valid cells (21-tap follow-up sweep).
<table><tr><td>Model</td><td>Tier</td><td>Onset (%)</td><td>Onset layer</td><td>Probe AUROC (onset)</td><td>Probe AUROC (peak)</td><td>TF-IDF AUROC</td><td></td><td>Probe beats TF-IDF at onset</td></tr><tr><td colspan="7">Panel A: Factual retrieval tiers (N = 11 cells; mean onset = 6.82%, median = 5.0%)</td><td></td><td></td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>R1 (PopQA-high)</td><td></td><td>5.0 Layer 1 / 28</td><td></td><td>0.864</td><td>0.941</td><td>0.882</td><td>No</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>R2 (PopQA-low)</td><td>5.0</td><td>Layer 1 / 28</td><td></td><td>0.877</td><td>0.935</td><td>0.889</td><td>No</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>R3 (SimpleQA)</td><td>5.0</td><td>Layer 1 / 28</td><td></td><td>0.775</td><td>0.817</td><td>0.728</td><td>Yes</td></tr><tr><td>Qwen2.5-3B-Instruct</td><td>R2 (PopQA-low)</td><td>5.0</td><td>Layer 2 / 36</td><td></td><td>0.867</td><td>0.905</td><td>0.868</td><td>No</td></tr><tr><td>Qwen2.5-3B-Instruct</td><td>R3 (SimpleQA)</td><td>15.0</td><td>Layer 5 / 36</td><td></td><td>0.654</td><td>0.784</td><td>0.670</td><td>No</td></tr><tr><td>Qwen2.5-7B-Base</td><td>R1 (PopQA-high)</td><td>5.0</td><td>Layer 1 / 28</td><td></td><td>0.787</td><td>0.925</td><td>0.791</td><td>No</td></tr><tr><td>Qwen2.5-7B-Base</td><td>R2 (PopQA-low)</td><td>5.0</td><td>Layer 1 / 28</td><td></td><td>0.905</td><td>0.936</td><td>0.883</td><td>Yes</td></tr><tr><td>Qwen2.5-7B-Base</td><td>R3 (SimpleQA)</td><td>15.0</td><td>Layer 4 / 28</td><td></td><td>0.673</td><td>0.840</td><td>0.719</td><td>No</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>R1  $\mathrm { ( P o p Q A { - } h i g h ) }$ </td><td>5.0</td><td>Layer 1 / 28</td><td></td><td>0.796</td><td>0.879</td><td>0.838</td><td>No</td></tr><tr><td>Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct</td><td>R2 (PopQA-low)</td><td>5.0</td><td>Layer 1 / 28</td><td></td><td>0.697</td><td>0.863</td><td>0.788</td><td>No</td></tr><tr><td></td><td>R3 (SimpleQA)</td><td>5.0</td><td>Layer 1 / 28</td><td></td><td>0.729</td><td>0.853</td><td>0.802</td><td>No</td></tr><tr><td colspan="9">Panel B: Multi-step reasoning tiers (N = 13 cells; mean onset = 28.08%, median = 30.0%)</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>C3 (MATH L4–5)</td><td>5.0</td><td>Layer 1 / 28</td><td></td><td>0.736</td><td>0.861</td><td>0.736</td><td>Tie</td></tr><tr><td>Qwen2.5-3B-Instruct</td><td>C2 (MATH L1–2)</td><td>5.0</td><td>Layer 2 / 36</td><td></td><td>0.691</td><td>0.853</td><td>0.662</td><td>Yes</td></tr><tr><td>Qwen2.5-3B-Instruct</td><td>C3 (MATH L4–5)</td><td>5.0</td><td>Layer 2 / 36</td><td></td><td>0.689</td><td>0.838</td><td>0.712</td><td>No</td></tr><tr><td>Qwen2.5-7B-Base</td><td>C2 (MATH L1–2)</td><td>5.0</td><td>Layer 1 / 28</td><td></td><td>0.682</td><td>0.832</td><td>0.695</td><td>No</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>C3 (MATH L4–5)</td><td>5.0</td><td>Layer 1 / 28</td><td></td><td>0.740</td><td>0.804</td><td>0.738</td><td>Yes</td></tr><tr><td>Qwen2.5-0.5B-Instruct</td><td>C3 (MATH L4–5)†</td><td>15.0</td><td>Layer 4 / 24</td><td></td><td>0.708</td><td>0.875</td><td>0.646</td><td>Yes</td></tr><tr><td>Qwen2.5-7B-Base</td><td>C1 (GSM8K)</td><td>30.0</td><td>Layer 8 / 28</td><td></td><td>0.708</td><td>0.745</td><td>0.495</td><td>Yes</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>C1 (GSM8K)</td><td>40.0</td><td>Layer 11 / 28</td><td></td><td>0.659</td><td>0.718</td><td>0.627</td><td>Yes</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>C2 (MATH L1–2)</td><td>40.0</td><td>Layer 11</td><td>/28</td><td>0.672</td><td>0.767</td><td>0.716</td><td>No</td></tr><tr><td>Qwen2.5-0.5B-Instruct</td><td>C2 (MATH L1–2)</td><td>45.0</td><td>Layer 11 </td><td>/24</td><td>0.738</td><td>0.875</td><td>0.655</td><td>Yes</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>C2 (MATH L1–2)</td><td>50.0</td><td>Layer 14</td><td>/28</td><td>0.675</td><td>0.732</td><td>0.592</td><td>Yes</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>C1 (GSM8K)</td><td>60.0</td><td>Layer 17</td><td>/28</td><td>0.716 0.670</td><td>0.716 0.676</td><td>0.519 0.577</td><td>Yes</td></tr><tr><td>Qwen2.5-3B-Instruct</td><td>C1 (GSM8K)</td><td>60.0</td><td>Layer 22 / 36</td><td></td><td></td><td></td><td></td><td>Yes</td></tr></table>

<sup>†</sup> 0.5b-instruct\_\_C3 was excluded by Gate 3 on the five-tap grid and passes on the 21-tap grid, but its probe rests on 16 calibration and 12 test items and its test-split AUROC at onset is 0.33; treat it as noise. It lowers the reasoning mean, so it works against H4. The other cell that passes only on the fine grid is 1.5b-instruct\_\_C1 (onset 60%). “Tie” means the two AUROCs are equal to numerical precision.

• Follow-up test (same estimator as the pre-registered rule): $\Delta _ { \mathrm { m e a n } } = + 2 1 . 2 6 \ : \mathrm { p p }$ [95% bootstrap CI: +9.72, +33.05]. An exact one-sided permutation test on the diference in means gives $\mathbf { \mathit { p } } = \mathbf { 0 . 0 0 4 }$ (the pipeline’s 20,000-draw approximation gives 0.0039). The median gap is $\Delta _ { \mathrm { m e d i a n } } = + 2 5 . 0 \mathrm { p p }$ (30.0% vs. 5.0%).

• Surface baseline: the onset-layer probe fails to beat the prompt-only TF-IDF baseline in 12 of 24 cells, including 9 of 11 retrieval cells. Early retrieval onsets are therefore not evidence that the network has computed anything about recall by 5% depth; a bag-of-words model of the question predicts the same label about as well.

• Scale: the pre-registered prediction that larger models onset earlier is not supported (Spearman $\rho ( \mathrm { o n s e t , p a r a m e t e r s } ) = - 0 . 2 2 ~ [ - 0 . 5 6 , + 0 . 2 1 ] , \ : p = 0 . 3 0 )$ . Peak AUROC is higher for retrieval than for reasoning (mean 0.88 vs. 0.79), as predicted.

## 4.3 Distributional Nuance

As shown in Table 1, five reasoning cells also have onsets at 5%, all of them MATH cells: $1 . 5 \mathsf { b } _ { -- } \mathsf { C } 3 , 3 \mathsf { b } _ { -- } \mathsf { C } 3$ and 7b-instruct\_\_C3 (levels 4–5), and 3b\_\_C2 and 7b-base\_\_C2 (levels 1–2). This is not a floor efect: greedy accuracy in these cells ranges from 13% to 59%. Two of the five do not beat the TF-IDF baseline at onset. We do not have a confirmed explanation for these early onsets.

By tier, GSM8K (C1) onsets are uniformly late ([30, 40, 60, 60]%), MATH levels 1–2 (C2) are mixed ([5, 5, 40, 45, 50]%), and MATH levels 4–5 (C3) are early ([5, 5, 5, 15]%). The reasoning−retrieval gap is therefore carried mainly by C1 and part of C2.

Excluding GSM8K (C1) leaves a smaller gap: $\pmb { \Delta } _ { \mathrm { n o - C 1 } } = + 1 2 . 6 3 \mathrm { p p }$ (95% bootstrap CI [+1.31, +25.76], 20,000 draws, seed 20260813), which still excludes zero but with a lower bound near it.

## 5 Prompt-Format Instability & Verbal Fragility (H0)

Hypothesis H0 tested whether verbalized confidence reflects a stable latent property invariant to elicitation formatting $( \rho \ge 0 . 6 0 )$ . We elicited verbal confidence across three distinct prompt templates:

• Format A: An integer confidence from 0 to 100 that the answer is correct.

• Format B: One of five ordinal labels: CERTAIN, FAIRLY\_CONFIDENT, SOME-WHAT\_UNSURE, MOSTLY\_GUESSING, NO\_IDEA. Because the raw Format B value was circular (the empirical accuracy of the chosen label), the canonical verbal signal is $B _ { \mathrm { f i x } }$ , a label-free fixed mapping of these labels to probabilities.

• Format C: An answer-or-pass decision under explicit scoring (+1 for a correct answer, −2 for an incorrect one, 0 for a pass).

Table 2: Verbal format correlation matrix (Gate 2 compliance).
<table><tr><td>Comparison Pair</td><td>Spearman  $\rho$ </td><td>Pre-Registered Target</td><td>Status</td></tr><tr><td>Format A vs. Format B</td><td>+0.126</td><td>≥ 0.60</td><td>FALSIFIED</td></tr><tr><td>Format A vs. Format C</td><td>-0.069</td><td>≥ 0.60</td><td>FALSIFIED</td></tr><tr><td>Format B vs. Format C</td><td>+0.247</td><td>≥ 0.60</td><td>FALSIFIED</td></tr></table>

As shown in Table 2, all pairwise correlations were well below the pre-registered 0.60 threshold, and pair A–C was slightly negative (ρ = −0.069, 95% CI [−0.111, −0.030]). Format A is close to a ceiling: 13,121 of 15,000 Format A responses state 90 or higher. Among those, the model still chose to pass in Format C 26.8% of the time. Verbalized confidence varies with the elicitation format and cannot be treated as a stable scalar property of the model.

## 6 Calibration vs. Discrimination: Murphy Decomposition (H1)

Hypothesis H1 pre-registered a calibration ranking, Behavioral > Internal > Verbal, passing only if the best-minus-worst ECE diference had a CI excluding zero.

## 6.1 The ECE Equivalence Fallacy

On the pooled test split of the pre-registered five-tap run, ECE was indistinguishable: Verbal ECE = 0.0356 [0.024, 0.061], Behavioral $\mathrm { E C E } = 0 . 0 4 2 7 \ [ 0 . 0 3 5 , 0 . 0 6 1 ]$ , Internal ECE = 0.0368 [0.031, 0.058]. The worst−best diference is 0.0071 with a 95% bootstrap CI that includes zero $[ - 0 . 0 1 6 , ~ + 0 . 0 2 7 ] .$ , so H1 is falsified under its pre-registered criterion. (The 21-tap follow-up changes only the internal row: ECE 0.0364, $N = 2 { , } 4 0 0 .$ ) These pooled numbers compare diferent question sets per signal; Section 6.2 restricts to items carrying all three.

![](images/010d1fb100fa327326aeecb8558f741d59e29f484c4c99844192d8fe7d012efe.jpg)

![](images/fbfdaf19b21de65bbc5313f40a96dc7f916bc921e01e1f10c3aede3643c57cd8.jpg)  
Figure 3: Signal calibration and Murphy decomposition on the test split. Left: Calibration curves close to the diagonal; points with $N < 1 0$ pruned to remove small-sample noise. Right: Murphy decomposition of the Brier score. All signals show small reliability error (solid bars), while behavioral consensus achieves more than double the resolution (light bars) of verbal confidence.

## 6.2 The Complete-Case Tradeof & ECE Binning Sensitivity (N = 842 Matched Rows)

Pooled ECE hides coverage imbalances (follow-up sweep: Verbal N = 1,166 across 12 cells; Internal $N =$ 2,400 across 24 cells; Behavioral $N = 2 ,$ 451 across 29 cells). For a like-for-like comparison, Table 3 uses the $N = 8 4 2$ test questions, across 10 cells, where all three signals are present (base rate $\bar { y } = 0 . 2 8 1 5 )$ , reporting ECE under 15 uniform bins (pipeline standard) and 10 uniform bins. Reliability and resolution use 10 bins.

Table 3: Complete-case Murphy decomposition and ECE sensitivity (N = 842 matched items).
<table><tr><td>Signal</td><td>ECE (15 bins) ↓</td><td>ECE (10 bins) ↓</td><td>Reliability ↓</td><td>Resolution ↑</td><td>Brier↓</td></tr><tr><td>Verbal  $\left( B _ { \mathrm { f i x } } \right)$ </td><td>0.0366</td><td>0.0326</td><td>0.00171</td><td>0.03160</td><td>0.1713</td></tr><tr><td>Internal Probe</td><td>0.0500</td><td>0.0338</td><td>0.00219</td><td>0.05154</td><td>0.1524</td></tr><tr><td>Behavioral Consensus</td><td>0.0704</td><td>0.0680</td><td>0.00808</td><td>0.06371</td><td>0.1465</td></tr></table>

Table 3 shows three patterns:

• Behavioral consensus scores worse on ECE (0.0704 / 0.0680 vs. 0.0366 / 0.0326). A plausible cause is that 10 samples at $T = 0 . 8$ give a coarse, often polarized agreement score.

• Behavioral consensus leads on resolution: 2.02× that of verbal confidence (0.06371 vs. 0.03160) and 1.24× that of the internal probe (0.05154). This sorting power outweighs the calibration penalty, giving the lowest Brier score (0.1465 vs. 0.1524 internal and 0.1713 verbal).

• Verbal confidence keeps ECE low by clustering near the base rate: in 9 of the 10 complete-case cells, calibrated verbal confidence never exceeds 0.50. Like a forecaster who always predicts the historical average, it keeps calibration error low while separating correct from incorrect answers poorly.

![](images/b717439ae3c81fc0775e50ce025bfee9bac35a405e03105030e293bca2b6c0bd.jpg)

![](images/834409372ea61bb53f4b36bbbefe38760199ced993c121efe89e5da172d347c4.jpg)  
Figure 4: Epistemic alignment quadrants and forced-answer abstention behavior. Left: Calibrated verbal vs. behavioral/internal confidence. Right: The abstention split on Format C passes (justified hedge vs. missed knowledge).

## 7 Disagreement Dynamics: Discrete Quadrant Artifact vs. Continuous Metric (H2)

Hypothesis H2 predicted that disagreements between stated verbal confidence and behavioral/internal confidence fall into distinct quadrants governed by question features, replicating across cells. H2 is not supported: the failing conjunct is replication across cells. We report the quadrant analysis descriptively.

Under the pre-registered 0.5 threshold, the five-tap run gave 90 cases of Excessive Hedging (low verbal, high behavioral) against 9 cases of Performative Certainty (high verbal, low behavioral) among $N = 8 3 1$ complete-case items (follow-up sweep: 86 vs. 9 among 842). The 10:1 ratio is largely an artifact of the cutof: in 8 of 9 cells calibrated verbal confidence never crosses 0.5, which suppresses the “high verbal” quadrants. The cases are also concentrated: all 9 Performative Certainty items come from one cell $\left( \mathtt { 7 b - i n s t r u c t \_ c 2 } \right)$ and all 90 Excessive Hedging items from two (3b-instruct\_\_C1, 7b-instruct\_\_R1).

Replacing the cutof with a continuous mismatch, ∆ = Verbal − Behavioral, gives a distribution centered at zero: Mean $( \Delta ) = - 0 . 0 0 6 , \mathrm { M e d i a n } ( \Delta ) = - 0 . 0 0 0 4 , \mathrm { S D } = 0 . 1 1 4$ . Regressing ∆ on question features (length, presence of a year, digits, multiple entities) gives no association (all $p > 0 . 3 5$ pooled; all $p > 0 . 6$ with cell fixed efects). Pooled over cells, quadrant membership is associated with some features $( \mathrm { e . g . }$ , multiple entities, $\chi ^ { 2 } = 7 9 . 6 ;$ digits, $\chi ^ { 2 } = 6 8 . 2 )$ , but no feature reaches significance within any single cell, which is consistent with these features acting as proxies for tier rather than as question-level drivers.

## 8 Limitations & Threats to Validity

• Single model family. All evaluations use Qwen2.5 (0.5B to 7B). The depth gap and format fragility need testing on other families.

• Probe label. Probes predict a median split of the model’s own semantic entropy, not correctness. H4 is about when the prompt representation predicts sampling dispersion, and should not be read as the depth at which the model “knows it is right”.

• Surface baseline. In 9 of 11 retrieval cells, the onset-layer probe does not beat a prompt-only TF-IDF baseline, so early retrieval onsets may reflect surface properties of the question.

• Pre-registration. The pre-registered five-tap H4 test passed but was fragile (permutation p = 0.052). The 21-tap sweep and 2-consecutive rule are a follow-up designed after those results were seen.

• Reasoning composition. The reasoning−retrieval gap is carried mainly by GSM8K (C1); MATH levels 4–5 (C3) onsets are early, and excluding C1 roughly halves the gap (+12.6 pp, CI lower bound +1.3).

• Grading. The symbolic grader under-credits correctly formatted math answers (90% agreement with human labels, all errors one-directional), slightly depressing measured accuracy in C2/C3.

• Sample size. 500 questions per cell (100 test items), below the protocol’s target of 2,000. Completecase analyses cover only 9–10 of 30 cells.

• H3 untested. Base-vs-instruct alignment could not be evaluated: the 7B base model’s verbal confidence was degenerate (essentially two labels) and was excluded in all six tiers.

## 9 Conclusion

Across 30 experimental cells, the three confidence channels do not behave like interchangeable readings of one quantity. Verbal self-assessment depends on the elicitation format. Sampling consensus has roughly twice the resolution of verbal confidence and the best Brier score, at about 10× the inference cost. Internal probes become informative at shallower depth for retrieval than for reasoning, but the retrieval signal is largely matched by a prompt-only baseline, and the gap rests on a follow-up analysis that was not pre-registered. On this evidence, practitioners should not route on verbal confidence alone; sampling consensus is the most informative single signal we measured.

## Broader Impact Statement

This work evaluates how far confidence signals from LLMs can be trusted for selective abstention. Its main practical implication is cautionary: verbal confidence changes with the elicitation format and should not be used on its own to gate answers in safety-relevant settings, and early-layer probe signals for factual recall are largely matched by a prompt-only baseline. The study uses public benchmarks and open-weight models only, and releases no new model.

## References

Amos Azaria and Tom Mitchell. The internal state of an LLM knows when it’s lying. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 967–976, 2023.

Jochen Bröcker. Reliability, suficiency, and the decomposition of proper scores. Quarterly Journal of the Royal Meteorological Society, 135(643):1512–1519, 2009.

Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. Discovering latent knowledge in language models without supervision. In International Conference on Learning Representations (ICLR), 2023.

M. S. Chauhan. DECK: A consistency × confidence taxonomy of LLM hallucinations. arXiv preprint arXiv:2606.02289, 2026.

Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in large language models using semantic entropy. Nature, 630:625–630, 2024.

Jiahui Geng, Fengyu Cai, Yuxia Wang, Heinz Koeppl, Preslav Nakov, and Iryna Gurevych. A survey of confidence estimation and calibration in large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2024.

Tilmann Gneiting and Adrian E. Raftery. Strictly proper scoring rules, prediction, and estimation. Journal of the American Statistical Association, 102(477):359–378, 2007.

Chuan Guo, Geof Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In International Conference on Machine Learning (ICML), pp. 1321–1330, 2017.

Saurav Kadavath et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.

Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In International Conference on Learning Representations (ICLR), 2023.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Inference-time intervention: Eliciting truthful answers from a language model. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, 2023.

Stephanie Lin, Jacob Hilton, and Owain Evans. Teaching models to express their uncertainty in words. Transactions on Machine Learning Research, 2022.

Matéo Mahaut, Laura Aina, Paula Czarnowska, Momchil Hardalov, Thomas Müller, and Lluís Màrquez. Factual confidence of LLMs: On reliability and robustness of current estimators. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), 2024.

Samuel Marks and Max Tegmark. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. arXiv preprint arXiv:2310.06824, 2023.

Allan H. Murphy. A new vector partition of the probability score. Journal of Applied Meteorology, 12(4): 595–600, 1973.

Jeremy Nixon, Michael W. Dusenberry, Linchuan Zhang, Ghassen Jerfel, and Dustin Tran. Measuring calibration in deep learning. In CVPR Workshops, 2019.

Aviv Slobodkin, Omer Goldman, Avi Caciularu, Ido Dagan, and Shauli Ravfogel. The curious case of hallucinatory (un)answerability: Finding truths in the hidden states of over-confident large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023.

Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher D. Manning. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023.

Dennis Ulmer, Martin Gubri, Hwaran Lee, Sangdoo Yun, and Seong Joon Oh. Calibrating large language models using their generations only. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL), 2024.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In International Conference on Learning Representations (ICLR), 2023.

Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. Can LLMs express their uncertainty? an empirical evaluation of confidence elicitation in LLMs. In International Conference on Learning Representations (ICLR), 2024.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. Representation engineering: A top-down approach to AI transparency. arXiv preprint arXiv:2310.01405, 2023.

## A Correlation Matrix Across Confidence Signals

Pairwise Spearman between raw signals (test split)  
![](images/dffb4f957b6362b813b0cc325c473df38227fe174ed2ad24cf6083453e0a3f0b.jpg)  
Figure 5: Cross-signal rank correlations. Per-cell Spearman correlations between the raw verbal, behavioral and internal signals, computed where both signals in a pair are present. Pooled over the test split, raw verbal and internal signals correlate at $\rho = 0 . 0 8 6$ . After per-cell calibration the pooled correlation rises to 0.83, which mostly reflects diferences in dificulty between cells.

## B Compute and Provenance

All runs used a single NVIDIA RTX PRO 6000 Blackwell GPU. The five-tap generation run did not record the code version it used, but the 21-tap follow up did. Code, raw generations and derived tables are available at https://github.com/PEEXR/Confidence-AI.