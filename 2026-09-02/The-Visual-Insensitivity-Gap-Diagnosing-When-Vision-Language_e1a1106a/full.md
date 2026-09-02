# The Visual Insensitivity Gap: Diagnosing When Vision-Language Models Fail to Use Visual Evidence

Genpei Zhang

University of Wisconsin–Madison genpei.zhang@wisc.edu

## Abstract

Vision-language models are evaluated by aggregate accuracy on multimodal benchmarks, a practice that implicitly assumes the model uses its visual input. We show this assumption fails on 40%–97% of samples across six VLMs and three perceptual benchmarks: blurring the question-relevant visual region leaves the next-token distribution nearly unchanged. We name this phenomenon the Visual Insensitivity Gap and quantify it with a per-sample Visual Sensitivity Index (VSI). The gap is a property of samples, not of models: VSI ranks correlate across models (grand-mean Spearman ρ=+0.40, permutation $p \ < \ 1 0 ^ { - 3 } )$ , so the same samples are flagged insensitive by VLMs sharing no architectural detail beyond a contrastively pretrained vision tower. The mechanism is concrete: on the insensitive samples, a linear probe on each model’s own vision tower distinguishes perturbed from clean images at 0.72–0.79 accuracy, yet the model’s argmax token changes on only 2%–11% of the same samples, an encoder– LLM gap above 0.65 on every model. Mapping VSI’s diagnostic utility cell by cell surfaces a strong regime (multichoice reasoning on capable VLMs: AUROC=0.85–0.87) and a weak regime (well-calibrated factuality, where softmax confidence already leads). VSI is not a universal best abstention signal; it is a sample-intrinsic indicator of visionignoring failure, best used as a conditional ensemble component.

## 1 Introduction

Vision-language models (VLMs) are routinely evaluated by their aggregate accuracy on multimodal benchmarks such as POPE (Li et al. 2023), MMVP (Tong et al. 2024b), HallusionBench (Guan et al. 2024), and MMStar (Chen et al. 2024). This practice that implicitly assumes the model uses its visual input. Yet across six recent VLMs spanning three architecture families (Liu et al. 2024a,b; Bai et al. 2025a,b; Laurenc¸on et al. 2024a), perturbing the questionrelevant visual region leaves the output unchanged on 40%– 97% of perceptual-benchmark samples, even when a linear probe (Alain and Bengio 2017) on the model’s own vision encoder detects the perturbation. The gap matters wherever a VLM’s answer carries decision weight (clinical triage, document-grounded question answering, embodied agents): a confidently wrong answer that ignored the visual evidence is a different failure from a hedged answer that engaged with it, yet aggregate accuracy cannot tell them apart.

We name this phenomenon the Visual Insensitivity Gap: the discrepancy between what the vision encoder represents and what the language head writes out. We quantify it with the Visual Sensitivity Index (VSI), a per-sample measure of how much the next-token distribution moves when the question-relevant visual region is blurred. VSI distributions show a consistent heavy left tail across the six models and four benchmarks we study: a sub-population of insensitive samples separates from a bulk of sensitive ones. The sub-population is sample-intrinsic: VSI ranks agree across models with a grand-mean Spearman rank correlation of ρ=+0.40 on perceptual benchmarks (permutation $p < 1 0 ^ { - 3 }$ on every tested pair), so the same samples tend to be flagged insensitive by VLMs that share no architectural detail beyond a contrastively pretrained (Radford et al. 2021) vision tower. Visual insensitivity is therefore a property of samples and of the dominant pretraining paradigm, not of any individual model.

Mechanism follows. On samples a model flags as insensitive, a linear probe on that model’s own vision tower distinguishes perturbed from unperturbed images at 0.72–0.79 accuracy (a frozen reference CLIP probe reads 0.82). The model’s argmax token, by contrast, changes on only 2%– 11% of these same samples. The encoder–decoder gap of 0.66–0.71 replicates on all six models. The vision tower sees the perturbation; the language head, on a large fraction of samples, does not propagate that signal. This measurement rests on an input-level interventional framing, which we adopt throughout: attention-based interpretability of VLMs has produced rich descriptive maps (Neo et al. 2025; Ben Melech Stan et al. 2024) but does not by itself establish whether the represented information is used in producing the output. Removing visual evidence and watching the softmax move makes the causal question concrete: would the answer change if this evidence were absent? The Visual Insensitivity Gap is what we see when, sample after sample, the answer is no.

The gap is also diagnostic. Samples with low VSI but high softmax confidence form a quadrant of confidently wrong while ignoring vision predictions; these track hallucination by every measure we examine. On multi-choice reasoning (MMStar), the math and science subcategories on Qwen2.5- VL-32B yield $\mathrm { A U R O C } _ { \mathrm { V S I } } { = } 0 . 8 5$ and 0.87 respectively, the strongest single-signal selective-generation results we observe. Yet VSI is not a universal best signal: on POPEstyle factuality, softmax confidence still leads on several model–benchmark cells. Our framing is therefore conditional: where VSI helps is itself an empirical, per-cell question.

Our thesis is that visual insensitivity is a robust, sampleintrinsic property of contemporary VLMs, and we test it in three complementary ways. We (1) formalise the Visual Insensitivity Gap and establish its heavy-tailed, crossmodel structure on a six-model, four-benchmark grid (§2); (2) trace the mechanism to an encoder–LLM disconnect that replicates on all six models (§2.4); and (3) map where the per-sample signal is diagnostically useful (strong on multichoice reasoning, weak on well-calibrated factuality), and how it combines with softmax and verbalised-confidence baselines (§3–§4). We close by discussing what these findings do, and do not, tell us about how VLMs use vision.

## 2 The Visual Insensitivity Gap

This section formalises the phenomenon and traces it to a measurable encoder–LLM disconnect.

## 2.1 The Visual Sensitivity Index

For a vision-language model f producing next-token distributions $f ( \cdot \mid x , q )$ given image x and question $q ,$ we measure visual sensitivity by perturbing the image and observing how the output distribution moves. Let $x _ { \sigma }$ denote the image with a Gaussian-blur perturbation of strength σ applied to the region most relevant to the question $q .$ The Visual Sensitivity Index is

$$
\operatorname { V S I } ( x , q ; f ) = \operatorname { K L } ( f ( \cdot \mid x , q ) \parallel f ( \cdot \mid x _ { \sigma } , q ) ) ,
$$

with $\sigma { = } 2 0$ as the default (robustness across $\sigma \in$ $\{ 1 0 , 2 0 , 4 0 \}$ in §4.3). The KL form captures the full distributional shift, not only argmax flips; localising the perturbation isolates whether the model uses the specific content the question asks about, not images in general.

The question-relevant region is located by parsing the principal noun phrase from $q ,$ querying Grounding-DINO (Liu et al. 2024d) for an open-vocabulary box, and refining the box to a pixel mask with SAM (Kirillov et al. 2023). Samples without a confident box fall back to wholeimage perturbation. §4.3 also studies a whole-image variant (whole-image VSI). The raw sample pool per benchmark is identical across models; accuracy-dependent metrics use the answer-parseable subset $( \mathrm { A p p . } \mathrm { A } . 3 )$ , with per-cell counts of 93 to 500, and all metrics are computed per (model, benchmark) cell. We report VSI on six VLMs (LLaVA-1.5- 7B (Liu et al. 2024a), LLaVA-NeXT-Mistral-7B (Liu et al. 2024b), Idefics3-8B (Laurenc¸on et al. 2024a), Qwen3-VL-8B (Bai et al. 2025a), Qwen2.5-VL-7B and Qwen2.5-VL-32B (Bai et al. 2025b)) on three perceptual benchmarks (POPE (Li et al. 2023), MMVP (Tong et al. 2024b), HallusionBench (Guan et al. 2024)) and one multi-choice reasoning benchmark (MMStar (Chen et al. 2024)).

## 2.2 A heavy left tail of insensitive samples

Figure 1 plots the per-sample VSI histogram for every (model, benchmark) cell. The shape is consistent: a heavy left tail of insensitive samples and a separated bulk of sensitive samples. The insensitive fraction, i.e. samples with $\mathrm { V S I { < } 0 . 0 5 }$ , spans 40% to 97%, with HallusionBench typically highest (figures and diagrams reduce the informativeness of local-region perturbation) and POPE typically lowest. The heavy left tail is not an artefact of any one model: every model shows it on every benchmark we tested. The fractions are not monotone in model scale either: Qwen2.5- VL-32B’s tail is no lighter than Qwen2.5-VL-7B’s. Hartigan’s dip test does not reject unimodality and silhouette scores are moderate (Table 1), so we describe the structure as a heavy left tail rather than bimodality; the distinction matters because a heavy tail offers no natural threshold, which motivates the quintile-based analyses used throughout.

The whole-image variant (Table 1, last row) is the sanity check: its insensitive fraction collapses to $\approx ~ 6 \%$ , so the localised heavy tail is not a generic “any-perturbation” artefact. Region-level VSI captures which visual content the model attends to, not whether it attends to images at all.

## 2.3 Visual insensitivity is sample-intrinsic

We call a per-sample VSI score sample-intrinsic if its rank order is preserved across models above chance: given VSI scores $\dot { V } ^ { A }$ and $V ^ { B }$ from models $A , B$ on the same pool, the Spearman rank correlation $\rho ( V ^ { A } , V ^ { B } )$ is bounded away from zero and separated from any permutation null. Without this test, the heavy-tail pattern of §2.2 could be modelidiosyncratic: each model’s tail could comprise different samples, with only the aggregate shape in common.

Over the three perceptual benchmarks, the grand-mean Spearman of region-blur VSI across the 15 model pairs is $\rho { = } \mathrm { + 0 . 4 0 }$ . Same-family pairs tend to correlate more strongly on average (pooled region-blur mean $\rho { = } 0 . 5 1$ vs 0.37 for cross-family pairs, with the same direction on every benchmark). Cross-family agreement nonetheless remains substantial: Qwen3-VL vs LLaVA-1.5 reaches $\rho { = } 0 . 4 5$ on POPE, and even the weakest pair we tested $\mathbf { ( Q w e n 2 . 5 – V L - }$ 32B vs Idefics3 on MMVP) retains $\rho { = } { + } 0 . 2 0$ , well separated from a permuted null. Cross-family pairs share no component beyond a contrastively pretrained vision tower; their agreement suggests it is the shared pretraining paradigm that shapes which samples get ignored. A practical corollary, used in §2.4, is that an insensitivity flag derived from one VLM transfers to others at lower but informative fidelity. Appendix Fig. 5 renders joint VSI quintiles for three model pairs; the high-error cluster sits in the low-VSI corner for both models.

Bootstrap 95% CIs $( B { = } 1 0 0 0 )$ lie fully above $\rho { = } 0$ for every POPE and MMVP pair; permutation tests confirm (§4.3). Cross-model agreement is benchmark-dependent: POPE yields a mean $\rho { = } 0 . 5 5$ , MMVP $\rho { = } 0 . 3 4$ , and HallusionBench $\rho { = } 0 . 3 2$ , with HallusionBench weakest because its chart-reading items compete with our region-detection pipeline. VSI thus captures a benchmark-conditional sample property, not a universal one.

## 2.4 Encoder–LLM disconnect: the mechanism

What does a vision encoder represent on insensitive samples? We answer this with a controlled intervention. The six VLMs do not share a vision tower (the LLaVA models use CLIP-ViT-L/14; Idefics3 uses SigLIP-SO400M; the Qwen models use natively trained ViTs), so we probe each model’s own tower: an $L _ { 2 } .$ -regularised logistic-regression linear probe on $\ell _ { 2 } \cdot$ -normalised final-layer features, trained to discriminate perturbed from unperturbed images with grouped 5-fold cross-validation (App. A.4). On the 83 samples flagged low-VSI by Qwen3-VL (VSI<0.05), a single shared pool evaluated on each model’s own tower, every tower linearly separates perturbed from clean images at 0.72–0.79 accuracy (Table 2); a frozen reference CLIP encoder, independent of all six models, reads $0 . 8 2 \pm 0 . 0 4$ on the same pool (App. A.4).

![](images/6f8fc4bbddbd75c243114eb87b2807d772eb10f394b625e74ebd37e25b026c20.jpg)  
Figure 1: Distribution of per-sample VSI across 6 models × 3 benchmarks. A heavy left tail of insensitive samples (VSI<0.05, red dashed line) separates from a sensitive bulk; the insensitive fraction (f) ranges from 40% to 97% across cells.

Yet the same models’ outputs barely register the perturbation: the argmax token changes on only 2%–11% of the same samples (Table 2). The encoder–LLM gap, own-tower probe accuracy minus argmax-change rate, is 0.66–0.71 on every model. The reference-encoder gap (0.72–0.80) agrees in direction and magnitude, so the disconnect does not depend on whose representation is probed. The high-VSI control shows the asymmetry is not probe weakness: there, probe accuracy rises to 0.86–0.91 and the argmax-change rate rises with it (Fig. 2). The disconnect is therefore not a property of the architecture as a whole; it is concentrated where VSI is small.

Three alternative explanations are worth ruling out, all by the same data. Too-weakperturbation: ruled out by the 0.72– 0.79 own-tower probe accuracy (0.86–0.91 on the high-VSI control), well above the 2–11% argmax-change rate. Greedy decoding hiding distributional shift: ruled out because VSI is the KL between the two full softmax distributions and is small on the low-VSI subset by construction. Uniform reliance on a fixed visual prior: ruled out because the gap is concentrated in the bottom VSI quintile and closes toward the top, so the same language head does respond to visual changes on other samples from the same distribution. The disconnect is therefore best characterised as an input–output usage gap, consistent with a routing failure rather than a representational deficit: every model’s own vision tower, and a frozen generic encoder, carries the information, and the same language head demonstrably responds to visual change on high-VSI samples; on the affected samples, the path from encoder to output is simply silent.

## 3 Diagnostic Value

A heavy left tail on its own is a phenomenon, not a diagnostic. This section examines whether VSI carries usable signal for downstream tasks.

## 3.1 Failure modes within the low-VSI quintile

For each (model, benchmark) cell we take the bottom-VSI quintile and partition it by correctness and softmax confidence relative to the cell’s median. This yields four quadrants: correct + high softmax (lucky), correct + low softmax (cautious win), wrong + low softmax (the abstention target), and wrong + high softmax, i.e. confidently wrong while ignoring vision. The four-quadrant breakdown (Appendix Fig. 4) shows that the dangerous quadrant is substantial on HallusionBench (typically 25–30% of the bottom quintile), modest on MMVP, and smallest on POPE. LLaVA-NeXT carries the largest dangerous fraction; Qwen2.5-VL-32B the smallest. VSI’s signal value is therefore largest precisely where confidence-only signals fail: when the model is confidently wrong about what is in the image.

The cell-level numbers underlying this picture are nonuniform. On Qwen3-VL POPE, the bottom-VSI quintile carries an error rate of 10.5% against a top-quintile rate of 0.0% (the decrease is monotone across quintiles). On the same model’s MMVP, the bottom-quintile error rate is 15.0% against a top-quintile 25.0%: a 0.6 ratio in the inverse direction, where samples the model finds visually salient are also samples it finds harder. Pooled across both benchmarks the ratio is 2.33 (17.9% vs 7.7%), and on the larger six-model grid the ratio averages around 1.7 on POPE and around 1.0 on MMVP. The conclusion is conditional usefulness: VSI’s quintile signal is real on factuality benchmarks like POPE and HallusionBench, but on visual-primitives benchmarks like MMVP the relationship between insensitivity and error can invert. The hallucination-flagging interpretation specifically applies to the confidently-wrong subset, where Appendix Fig. 4 shows this quadrant aligning with low VSI across all six models.

<table><tr><td rowspan="2">Variant Median</td><td rowspan="2"></td><td colspan="2">Fraction with VSI</td><td rowspan="2"></td><td rowspan="2">Dip p Silh.</td></tr><tr><td> $< 0 . 0 1$ </td><td> $< 0 . 0 5$ </td></tr><tr><td>Region-blur</td><td>0.09</td><td>24.5%</td><td>41.5%</td><td>0.99</td><td>0.57</td></tr><tr><td>Patch-occlusion</td><td>0.09</td><td>21.0%</td><td>42.0%</td><td>0.96</td><td>0.54</td></tr><tr><td>Whole-image</td><td>2.78</td><td>1.5%</td><td>6.0%</td><td>0.24</td><td>0.61</td></tr></table>

Table 1: Per-variant VSI statistics $( n { = } 2 0 0 ,$ Qwen3-VL-8B). Localised perturbations carry a heavy left tail (∼ 42% below $\mathrm { V S I { = } 0 . 0 5 ) } ;$ ; whole-image blurring almost always moves the softmax. The dip test (Hartigan and Hartigan 1985) does not reject unimodality; silhouette (Rousseeuw 1987) is moderate.
<table><tr><td rowspan="2">Model</td><td colspan="2">Own-tower probe acc</td><td colspan="2"></td></tr><tr><td>low-VSI</td><td>high-VSI</td><td>∆argmax</td><td>Gap</td></tr><tr><td>Qwen3-VL-8B</td><td> $0 . 7 3 4 { \pm } . 0 3$ </td><td>0.893</td><td>0.024</td><td>0.710</td></tr><tr><td>Qwen2.5-VL-7B</td><td> $0 . 7 6 4 { \pm } . 0 7$ </td><td>0.885</td><td>0.060</td><td>0.704</td></tr><tr><td>Qwen2.5-VL-32B</td><td> $0 . 7 4 1 { \pm } . 0 6$ </td><td>0.889</td><td>0.084</td><td>0.656</td></tr><tr><td>LLaVA-1.5-7B</td><td> $0 . 7 2 2 { \pm } . 1 1$ </td><td>0.859</td><td>0.060</td><td>0.661</td></tr><tr><td>LLaVA-NeXT-7B</td><td> $0 . 7 8 9 { \pm } . 0 4$ </td><td>0.906</td><td>0.108</td><td>0.680</td></tr><tr><td>Idefics3-8B</td><td> $0 . 7 5 3 { \pm } . 0 4 $ </td><td>0.876</td><td>0.084</td><td>0.668</td></tr></table>

Table 2: Encoder–LLM disconnect on the shared $n { = } 8 3$ low-VSI pool, probed on each model’s own vision tower $\mathrm { ( \pm = 5 \mathrm { - } f o l d }$ std); $\mathrm { \hbar ^ { * } h i g h - V S I ^ { * } }$ is the matched sensitive control. ∆argmax: fraction of samples whose top-1 token changes under perturbation. Gap = probe acc − ∆argmax, >0.65 throughout. Reference-CLIP probe: 0.82±0.04 (App. A.4).

![](images/eb662f1efc5b68fa067cba7f927993666d1037fe7debcaa7531606af13754f95.jpg)  
Figure 2: Encoder–LLM disconnect on the low-VSI subset across 6 VLMs. Blue: linear-probe accuracy on each model’s own vision tower (0.72–0.79; error bars: 5-fold std). Solid orange: argmax-change rate on the same low-VSI samples (2–11%). Light orange: argmax-change rate on the high-VSI samples (where the gap largely closes). The vertical bracket shows the encoder– LLM gap of 0.66–0.71.

In cutoff terms: on the six-model POPE grid, VSI<0.05 flags 42% of samples (Fig. 1) within which the confidentlywrong quadrant is on average 2.1× denser than in a matched-size top-VSI population. Tightening to VSI<0.01 enriches a further 1.4× on Qwen-family models but reverses on LLaVA-NeXT, where the strictest cutoff mainly draws in refusal-style answers that softmax already flags; VSI<0.05 therefore captures most of the signal at manageable falsepositive cost.

## 3.2 VSI is strongest on multi-choice reasoning

Aggregate VSI metrics on perceptual benchmarks are modest (typical $\mathrm { A U R O C _ { V S I } } \in \mathsf { \bar { [ 0 . 4 5 , 0 . 6 0 ] } } )$ , but MMStar (Chen et al. 2024) reveals a sharper picture. MMStar groups its samples into six macro-categories: coarse perception, fine-grained perception, instance reasoning, logical reasoning, math, and science. On MMStar we compute wholeimage VSI throughout: its diagram- and chart-style layouts make open-vocabulary region detection unreliable, and a localised blur would conflate insensitivity with detection failure (App. B.7). Figure 3 reports per-category $\mathrm { A U R O C _ { V S I } }$ for each of our six models, with the full 6×6 numerical matrix in Appendix Table 5. On Qwen2.5-VL-32B, the math and science-and-technology categories yield $\mathrm { A U R O C } _ { \mathrm { V S I } } { = } 0 . 8 5 1$ (95% bootstrap CI $[ 0 . 7 3 , 0 . 9 4 ] , n { = } 3 4 )$ and $\mathrm { A U R O C _ { V S I } { = } 0 . 8 6 7 \ ( [ 0 . 7 8 , 0 . 9 5 ] , } \ i { = } 4 6 )$ respectively, with Pearson $r ( - \mathrm { V S I } , \mathrm { e r r } ) { \bf o f } + 0 . 4 7 \mathrm { a n d } + 0 . 5 9 ;$ the bottomquintile error rate is 4.05× the top-quintile rate on science, and on math the top quintile has zero errors. These are the strongest signals we observe anywhere in the grid, though the per-category samples are small and the intervals correspondingly wide. Other models on the same categories yield weaker signals: LLaVA-1.5-7B’s math AUROC is 0.29 (inverse direction), with a similar inversion on logical reasoning (AUROC=0.33). The MMStar result is a modeldependent and category-dependent finding rather than a universal claim.

![](images/205fb25723c94c2d8173b891766595821209c5541bbb4122bd909bc19f60121b.jpg)  
Figure 3: Per-category $\mathrm { A U R O C _ { V S I } }$ on MMStar’s six macrocategories. Multi-choice reasoning categories (math, science) yield the strongest VSI signals on capable models, peaking at 0.85–0.87 on Qwen2.5-VL-32B.

Why does VSI sharpen on multi-choice reasoning for capable models? Two plausible mechanisms. First, the answer set is small (typically four options), so when visual evidence moves the output distribution at all, it more often flips the argmax; distributional sensitivity and answer changes align tightly. Second, math and science questions are densely tied to the image (e.g., reading values off a chart), so ignoring vision more often means ignoring the answer-determining evidence. The inverse phenomenon on LLaVA-1.5 suggests this mechanism reverses when the model’s capability ceiling on the category is already binding: errors there are dominated by reasoning failures rather than vision failures, and VSI tracks only the latter. This per-model heterogeneity is consistent with our overarching framing of conditional signal selection.

## 3.3 Sample exemplars

Appendix Fig. 6 renders six low-VSI samples (visibly present objects denied, visual primitives flipped, colours misread) that all three of our baseline models (Qwen3-

VL, Qwen2.5-VL-7B, LLaVA-1.5-7B) answer incorrectly with softmax confidence above 0.99 and VSI below 0.01. The combination is the same in every case: visual evidence is present and encoder-detectable (§2.4), and the model nonetheless produces a confident incorrect answer that does not move when the relevant region is removed. These samples are concrete instances of the confidently wrong while ignoring vision quadrant of Appendix Fig. 4, and they illustrate why an abstention signal built on softmax confidence alone cannot flag them: by that measure, these are precisely the predictions the model believes most strongly.

## 4 Conditional Signal Selection

The previous section showed VSI is a sharp diagnostic on certain (model, benchmark) cells. Is it the best confidence signal for downstream abstention or selective generation? No — not universally. VSI is instead a useful conditional component of a signal ensemble, capturing a failure mode (vision-ignoring confident errors) that softmax-based signals systematically miss.

## 4.1 VSI is not a universal best signal

Across the 18 cells of our perceptual grid (Table 3), the softmax max-probability baseline wins on 10 cells, primarily where the model is well-calibrated: Idefics3 and Qwen2.5- VL-32B on all three benchmarks, and LLaVA-NeXT on POPE (max-prob 0.85 vs region-VSI 0.54). VSI wins on the remaining 8 cells, almost always in hybrid form (seven hybrids and one standalone): a z-score ensemble that includes at least one of region-VSI, whole-image-VSI, or verbalised confidence is the best signal on cells where calibration is poor (LLaVA-1.5 on MMVP, Qwen3-VL on POPE), or where the failure mode is hallucination rather than uncertainty (LLaVA-NeXT on HallusionBench). VSI alone, without a softmax component, is the best signal on exactly one cell (LLaVA-1.5 on HallusionBench, whole-image VSI AU-ROC 0.58). The pattern, not any single cell, is the takeaway: no signal dominates across cells, but a VSI-containing ensemble wins where calibration is the bottleneck.

## 4.2 Complementarity with other confidence signals

Softmax max-probability (Hendrycks and Gimpel 2017; Guo et al. 2017) measures intrinsic prediction confidence. Verbalised confidence (Tian et al. 2023; Lin, Hilton, and Evans 2022) measures the model’s elicited self-assessment. VSI measures whether the model uses the visual input. These three signals should be largely independent. We confirm this empirically: the Pearson correlation between negative-VSI and negative-verbalised-confidence is between −0.08 and +0.05 across all six models on POPE. The (VSI, max-prob) joint quintile concentrates errors in the low-low corner (Appendix Fig. 7), suggesting an ensemble can recover errors that either signal alone misses.

An equal-weight z-score sum of region-VSI and wholeimage VSI is the best signal on 3 of the 18 perceptual cells. Concretely, on Qwen3-VL POPE the hyb(region+whole)

<table><tr><td>Model</td><td>Bench</td><td>n</td><td>Acc.</td><td>Max-prob</td><td>VSIr</td><td> $\mathrm { V S I } _ { w }$ </td><td> $\mathrm { h y b } ( r + w )$ </td><td>Best signal</td></tr><tr><td rowspan="3">Qwen3-VL-8B</td><td>POPE</td><td>359</td><td>0.88</td><td>0.544</td><td>0.591</td><td>0.619</td><td>0.636</td><td>hyb(r+w)</td></tr><tr><td>MMVP</td><td>268</td><td>0.63</td><td>0.622</td><td>0.487</td><td>0.568</td><td>0.564</td><td>hyb(r+w+mp+vb)</td></tr><tr><td>Hallu</td><td>93</td><td>0.72</td><td>0.602</td><td>0.584</td><td>0.479</td><td>0.515</td><td>mp</td></tr><tr><td rowspan="3">Qwen2.5-VL-7B</td><td>POPE</td><td>167</td><td>0.74</td><td>0.710</td><td>0.419</td><td>0.575</td><td>0.583</td><td>hyb(r+w+mp+vb)</td></tr><tr><td>MMVP</td><td>300</td><td>0.55</td><td>0.496</td><td>0.583</td><td>0.651</td><td>0.676</td><td>hyb(r+w)</td></tr><tr><td>Hallu</td><td>81</td><td>0.74</td><td>0.539</td><td>0.539</td><td>0.547</td><td>0.549</td><td>hyb(r+w+mp+vb)</td></tr><tr><td rowspan="3">Qwen2.5-VL-32B</td><td>POPE</td><td>496</td><td>0.88</td><td>0.820</td><td>0.417</td><td>0.550</td><td>0.555</td><td>mp</td></tr><tr><td>MMVP</td><td>294</td><td>0.75</td><td>0.665</td><td>0.445</td><td>0.595</td><td>0.538</td><td>mp</td></tr><tr><td>Hallu</td><td>165</td><td>0.71</td><td>0.608</td><td>0.486</td><td>0.553</td><td>0.497</td><td>mp</td></tr><tr><td rowspan="3">LLaVA-1.5-7B</td><td>POPE</td><td>500</td><td>0.83</td><td>0.796</td><td>0.570</td><td>0.470</td><td>0.490</td><td>mp</td></tr><tr><td>MMVP</td><td>280</td><td>0.53</td><td>0.484</td><td>0.546</td><td>0.586</td><td>0.593</td><td> $\mathbf { \Pi } _ { \mathrm { V S I } _ { w } } ^ { \mathrm { h y b ( } r + w \mathrm { ) } }$ </td></tr><tr><td>Hallu</td><td>179</td><td>0.50</td><td>0.488</td><td>0.506</td><td>0.581</td><td>0.564</td><td></td></tr><tr><td rowspan="3">LLaVA-NeXT-Mistral-7B</td><td>POPE</td><td>500</td><td>0.88</td><td>0.853</td><td>0.541</td><td>0.616</td><td>0.640</td><td>mp</td></tr><tr><td>MMVP</td><td>300</td><td>0.62 0.42</td><td>0.597 0.552</td><td>0.485</td><td>0.576 0.559</td><td>0.563</td><td>mp</td></tr><tr><td>Hallu</td><td>179</td><td></td><td></td><td>0.562</td><td></td><td>0.591</td><td>hyb(r+w+mp)</td></tr><tr><td rowspan="3">Idefics3-8B</td><td>POPE</td><td>499</td><td>0.89</td><td>0.657</td><td>0.475</td><td>0.508</td><td>0.515</td><td>mp</td></tr><tr><td>MMVP</td><td>300</td><td>0.51</td><td>0.583 0.575</td><td>0.458</td><td>0.412 0.497</td><td>0.416</td><td>mp</td></tr><tr><td>Hallu</td><td>178</td><td>0.47</td><td></td><td>0.477</td><td></td><td>0.491</td><td>mp</td></tr></table>

Table 3: Selective-generation AUROC across all 18 cells (6 models × 3 perceptual benchmarks). Bold / underline = best / second-best among the four displayed AUROC columns; shaded Best-signal cells contain a VSI component. n is the answerparseable subset per cell (App. A.3). No single signal dominates: max-prob wins on $1 0 / 1 8$ cells (where the model is wellcalibrated), a hybrid containing at least one VSI component wins on $7 / 1 8$ (where calibration is poor or the failure mode is hallucination), and VSI alone wins on $1 / 1 8 . \ ^ { \ast } \mathrm { V S I } _ { r } ^ { \ , , }$ = region-blur $\mathrm { V S I } ;  { \mathrm { \Omega } } ^ { \ast } \mathrm { V S I } _ { w }  { \mathrm { \Omega } } ^ { \ast } =$ whole-image VSI; “hyb(r+w)” = equal weight z-score sum of the two; $\mathrm { \tilde { \Omega } ^ { \mathrm { e } } \mathrm { m p } ^ { \mathrm { , * } } } \mathrm { v b } ^ { \mathrm { , * } } = \mathrm { m a x } \mathrm { - p r o b } \mathrm { / }$ verbalised confidence. Best signal is selected from the full signal set, including ensembles whose AUROC columns are omitted for space.

signal reaches AUROC 0.636 versus max-probability AU-ROC 0.544, a +0.09 gain (paired bootstrap $p { < } 0 . 0 1 )$ ; on Qwen2.5-VL-7B MMVP it reaches AUROC 0.676 versus max-prob 0.496, a +0.18 gain. PRR@80 (the prediction– rejection ratio at 80% coverage: the fraction of the oracle’s achievable risk reduction that the signal realises when the worst-ranked 20% of predictions are rejected) is higher for the hybrid than for every single signal in cells where calibration is poor (LLaVA-NeXT HallusionBench: hybrid PRR 0.19 vs max-prob PRR −0.05). The hybrid also has the highest minimum (worst-case) AUROC across cells of any single or hybrid signal we examined, making it a sensible default for selective generation where verbalised confidence is unavailable. The equal-weight rule is not claimed optimal: a learned logistic-regression hybrid adds +0.01–+0.03 AU-ROC on most cells (Appendix D.1) at the cost of labelled calibration data.

## 4.3 Robustness: sigma choice, threshold sensitivity, and permutation tests

Does VSI’s signal depend on the perturbation strength σ=20? Appendix Table 6 and Fig. 10 show it does not. On four representative cells we recomputed VSI at $\sigma \in$ {10, 20, 40}; $\mathrm { A U R O C _ { V S I } }$ varies by at most 0.05, and the per-sample VSI rank correlation across σ pairs reaches $\rho { = } 0 . 7 6 { - } 0 . 9 7$ (mean 0.89). The choice of σ is therefore not load-bearing. The same flatness holds for the insensitivefraction threshold: across {0.01, 0.05, 0.10, 0.20} the direction of the low-VSI vs high-VSI error difference is invariant within each cell, and the per-cell ratio of bottom-quintile to top-quintile error rates moves by at most a factor of 1.3 across thresholds on the eight cells we audited (full sweep in Appendix D.1). The headline insensitive-fraction estimate of 40%–97% is therefore not an artefact of the 0.05 cutoff.

To rule out chance for the sample-intrinsic claim of §2.3, we ran a $1 0 ^ { 5 }$ -shuffle permutation test for each of four (model-A, model-B, benchmark) pairs. Observed Spearman $\rho ~ \in ~ [ + 0 . 2 0 , + 0 . 5 5 ]$ ; three of the four pairs reach twosided $\dot { p } < 1 0 ^ { - 5 }$ , and the weakest pair (Qwen2.5-VL-32B vs Idefics3 on MMVP, $\rho { = } \mathrm { + 0 . 2 0 ) }$ reaches $p { = } 3 . 4 \times 1 0 ^ { - 4 }$ Per-pair null distributions and the full table are in App. B.2. The observed agreement is far from the permuted null in every case. We similarly bootstrapped per-signal AUROCs in Table 3 (B=1000); 95% CIs and paired-bootstrap p-values for each “hybrid vs single-signal” comparison are released with the code.

## 5 Related Work

Hallucination evaluation in VLMs. A large body of work targets multimodal hallucination through new benchmarks: POPE (Li et al. 2023) for object-existence binary questions, MMVP (Tong et al. 2024b) for CLIP-blind visualprimitive pairs, HallusionBench (Guan et al. 2024) for entangled language hallucination and visual illusion, and

MMStar (Chen et al. 2024) for reasoning-and-perception questions filtered to be visually grounded. Broader evaluation suites such as MMBench (Liu et al. 2024e) and MMMU (Yue et al. 2024) extend coverage to multidisciplinary reasoning and image-grounded knowledge but report aggregate accuracy rather than per-sample reliability. Surveys (Bai et al. 2024; Liu et al. 2024c) situate these benchmarks within broader VLM evaluation; CHAIR (Rohrbach et al. 2018) measured caption hallucination earlier. Our work uses these benchmarks as substrates rather than proposing a new one, and decomposes their aggregate scores at the sample level so that the heavy-tailed VSI distribution becomes visible.

Selective generation, abstention, and calibration. The selective-generation literature (Geifman and El-Yaniv 2017; Whitehead et al. 2022) measures models that may abstain. The standard signal is softmax max-probability (Hendrycks and Gimpel 2017) after temperature scaling (Guo et al. 2017); verbalised-confidence prompting (Tian et al. 2023; Lin, Hilton, and Evans 2022) elicits an explicit confidence score from the model itself. Both signals were developed in unimodal settings and inherit a shared blind spot in the multimodal regime: each is computed from the output distribution alone, downstream of encoding, and so cannot distinguish a confident prediction that uses the visual evidence from one that confidently ignores it. We use these baselines as components in our ensemble (§4.2), and show that VSI captures precisely the second case.

Input-level intervention vs internal probing. Two contrasting methodologies recur in the VLM analysis literature. Internal-state methods (linear probing of frozen encoders (Alain and Bengio 2017; Radford et al. 2021), compositional probes of language–image binding (Yuksekgonul et al. 2023), and attention-map analysis (Neo et al. 2025; Ben Melech Stan et al. 2024)) ask what information is present in a model’s intermediate representations. Input-level methods, including counterfactual synthesis (Vo et al. 2025) and the region perturbation used here, instead ask what information the model’s output actually uses. Deletion-based attribution methods (occlusion maps (Zeiler and Fergus 2014), randomised input sampling (Petsiuk, Das, and Saenko 2018), and perturbation-based local explanations (Ribeiro, Singh, and Guestrin 2016)) also remove input content, but their goal is per-pixel saliency for a single prediction; VSI instead compresses the output distribution’s response to a question-conditioned perturbation into a per-sample scalar whose ranking transfers across models. Relatedly, the VQA language-prior literature (Agrawal et al. 2018) established that models can answer without looking at the image; VSI turns that observation into a persample, cross-model measurement. The internal and inputlevel views are complementary: our §2.4 result requires both a probe (to show a frozen encoder represents the perturbation) and a counterfactual (to show the output does not move). Concurrent attention-probing work on perception failures (Liu et al. 2025) examines a related question on the model’s internal-state evidence, and reasoning-time grounding-decay work (Raghu and Pandey 2026) examines it on the autoregressive decode trajectory rather than at the prediction-time softmax. We see these approaches as slices of the same encoder–decoder routing problem.

Diagnostic and probing perspectives on VLMs. Closest in genre to our work, Yuksekgonul et al. (2023) show VLMs behave like bag-of-words on compositional probes; Tong et al. (2024b) trace MLLM perception failures to systematic blind spots in CLIP, and Tong et al. (2024a) extend that diagnosis to a wider vision-centric benchmark suite. Darcet et al. (2024) show that vision transformers spontaneously repurpose redundant patch tokens for global computation, a candidate explanation for how encoder information can be present yet routed away from the language head. Laurenc¸on et al. (2024b) ablate VLM design choices and find that the cross-modal projection substantially shifts downstream perception accuracy, consistent with our reading of the disconnect as a routing problem. The diagnostic framing of the present paper follows Geirhos et al. (2019), who probe CNN shape-vs-texture bias, and Schaeffer, Miranda, and Koyejo (2023), who argue that emergent capabilities are a measurement artefact rather than an architectural property. Where prior probing work is model-intrinsic, ours is sample-intrinsic: the property lives on samples, not architectures, and its ordering transfers across models sharing only a contrastively pretrained vision tower.

## 6 Discussion

Following Schaeffer, Miranda, and Koyejo (2023) and Recht et al. (2019), we are deliberate about scope. We propose no new model, training objective, or abstention method, and we do not claim VSI is a universal selective-generation signal: softmax max-probability beats it on 10 of 18 well-calibrated cells (Table 3). Four limitations bound the claims. Crossmodel agreement is benchmark-conditional (POPE mean ρ=0.55; MMVP 0.34; HallusionBench 0.32, whose chart items defeat local-region blurring). The VSI–error relationship can invert (Qwen3-VL on MMVP perception subtypes, p=0.043, uncorrected; App. D.1), so a deployed threshold needs per-cell calibration. Probe accuracies are lineardecodability lower bounds on the encoders’ information content. And we intervene at the input, so the disconnect is causal at the input–output level but does not pinpoint a structural locus within the language head.

What remains is a sample-intrinsic property replicated across six VLMs, robust to perturbation strength and threshold choice, and diagnostically strongest where the model is otherwise capable (AUROC up to 0.87 on MMStar reasoning with Qwen2.5-VL-32B). Two extensions follow naturally: activation patching across the cross-modal projection to localise the failure (Neo et al. 2025; Ben Melech Stan et al. 2024), and targeted retraining that up-weights the insensitive sub-population. For now, VSI is best read as a measurement: a signal that says, of a given prediction, whether the model used vision to produce it.

## References

Agrawal, A.; Batra, D.; Parikh, D.; and Kembhavi, A. 2018. Don’t Just Assume; Look and Answer: Overcoming Priors for Visual Question Answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Alain, G.; and Bengio, Y. 2017. Understanding Intermediate Layers Using Linear Classifier Probes. In International Conference on Learning Representations Workshop (ICLR Workshop).

Bai, S.; Cai, Y.; Chen, R.; Chen, K.; Chen, X.; Cheng, Z.; Deng, L.; Ding, W.; Gao, C.; Ge, C.; et al. 2025a. Qwen3- VL Technical Report. arXiv preprint arXiv:2511.21631.

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang, K.; Wang, P.; Wang, S.; Tang, J.; Zhong, H.; Zhu, Y.; Yang, M.; Li, Z.; Wan, J.; Wang, P.; Ding, W.; Fu, Z.; Xu, Y.; Ye, J.; Zhang, X.; Xie, T.; Cheng, Z.; Zhang, H.; Yang, Z.; Xu, H.; and Lin, J. 2025b. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923.

Bai, Z.; Wang, P.; Xiao, T.; He, T.; Han, Z.; Zhang, Z.; and Shou, M. Z. 2024. Hallucination of Multimodal Large Language Models: A Survey. arXiv preprint arXiv:2404.18930.

Ben Melech Stan, G.; Aflalo, E.; Rohekar, R. Y.; Bhiwandiwalla, A.; Tseng, S.-Y.; Olson, M. L.; Gurwicz, Y.; Wu, C.; Duan, N.; and Lal, V. 2024. LVLM-Intrepret: An Interpretability Tool for Large Vision-Language Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 8182–8187.

Chen, L.; Li, J.; Dong, X.; Zhang, P.; Zang, Y.; Chen, Z.; Duan, H.; Wang, J.; Qiao, Y.; Lin, D.; and Zhao, F. 2024. Are We on the Right Way for Evaluating Large Vision-Language Models? In Advances in Neural Information Processing Systems (NeurIPS).

Darcet, T.; Oquab, M.; Mairal, J.; and Bojanowski, P. 2024. Vision Transformers Need Registers. In International Conference on Learning Representations (ICLR).

Geifman, Y.; and El-Yaniv, R. 2017. Selective Classification for Deep Neural Networks. In Advances in Neural Information Processing Systems (NeurIPS).

Geirhos, R.; Rubisch, P.; Michaelis, C.; Bethge, M.; Wichmann, F. A.; and Brendel, W. 2019. ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. In International Conference on Learning Representations (ICLR).

Guan, T.; Liu, F.; Wu, X.; Xian, R.; Li, Z.; Liu, X.; Wang, X.; Chen, L.; Huang, F.; Yacoob, Y.; Manocha, D.; and Zhou, T. 2024. HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Guo, C.; Pleiss, G.; Sun, Y.; and Weinberger, K. Q. 2017. On Calibration of Modern Neural Networks. In International Conference on Machine Learning (ICML).

Hartigan, J. A.; and Hartigan, P. M. 1985. The Dip Test of Unimodality. The Annals ofStatistics, 13(1): 70–84.

Hendrycks, D.; and Gimpel, K. 2017. A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks. In International Conference on Learning Representations (ICLR).

Kirillov, A.; Mintun, E.; Ravi, N.; Mao, H.; Rolland, C.; Gustafson, L.; Xiao, T.; Whitehead, S.; Berg, A. C.; Lo, W.- Y.; Dollar, P.; and Girshick, R. 2023. Segment Anything. In´ Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

Laurenc¸on, H.; Marafioti, A.; Sanh, V.; and Tronchon, L. 2024a. Building and Better Understanding Vision-Language Models: Insights and Future Directions. arXiv preprint arXiv:2408.12637.

Laurenc¸on, H.; Tronchon, L.; Cord, M.; and Sanh, V. 2024b. What matters when building vision-language models? In Advances in Neural Information Processing Systems (NeurIPS).

Li, Y.; Du, Y.; Zhou, K.; Wang, J.; Zhao, W. X.; and Wen, J.- R. 2023. Evaluating Object Hallucination in Large Vision-Language Models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP).

Lin, S.; Hilton, J.; and Evans, O. 2022. Teaching Models to Express Their Uncertainty in Words. Transactions on Machine Learning Research (TMLR).

Liu, H.; Li, C.; Li, Y.; and Lee, Y. J. 2024a. Improved Baselines with Visual Instruction Tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Liu, H.; Li, C.; Li, Y.; Li, B.; Zhang, Y.; Shen, S.; and Lee, Y. J. 2024b. LLaVA-NeXT: Improved Reasoning, OCR, and World Knowledge. https://llava-vl.github.io/blog/2024-01- 30-llava-next/.

Liu, H.; Xue, W.; Chen, Y.; Chen, D.; Zhao, X.; Wang, K.; Hou, L.; Li, R.; and Peng, W. 2024c. A Survey on Hallucination in Large Vision-Language Models. arXiv preprint arXiv:2402.00253.

Liu, S.; Zeng, Z.; Ren, T.; Li, F.; Zhang, H.; Yang, J.; Jiang, Q.; Li, C.; Yang, J.; Su, H.; Zhu, J.; and Zhang, L. 2024d. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. In European Conference on Computer Vision (ECCV).

Liu, Y.; Duan, H.; Zhang, Y.; Li, B.; Zhang, S.; Zhao, W.; Yuan, Y.; Wang, J.; He, C.; Liu, Z.; Chen, K.; and Lin, D. 2024e. MMBench: Is Your Multi-Modal Model an All-Around Player? In European Conference on Computer Vision (ECCV).

Liu, Z.; Chen, Z.; Liu, H.; Luo, C.; Tang, X.; Wang, S.; Zeng, J.; Dai, Z.; Shi, Z.; Wei, T.; Dumoulin, B.; and Tong, H. 2025. Seeing but Not Believing: Probing the Disconnect Between Visual Attention and Answer Correctness in VLMs. arXiv preprint arXiv:2510.17771.

Neo, C.; Ong, L.; Torr, P.; Geva, M.; Krueger, D.; and Barez, F. 2025. Towards Interpreting Visual Information Processing in Vision-Language Models. In International Conference on Learning Representations (ICLR).

Petsiuk, V.; Das, A.; and Saenko, K. 2018. RISE: Randomized Input Sampling for Explanation of Black-box Models. In British Machine Vision Conference (BMVC).

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; Krueger, G.; and Sutskever, I. 2021. Learning Transferable Visual Models from Natural Language Supervision. In International Conference on Machine Learning (ICML), 8748– 8763. PMLR.

Raghu, S.; and Pandey, S. 2026. Don’t Blink: Evidence Collapse during Multimodal Reasoning. arXiv preprint arXiv:2604.04207.

Recht, B.; Roelofs, R.; Schmidt, L.; and Shankar, V. 2019. Do ImageNet Classifiers Generalize to ImageNet? In International Conference on Machine Learning (ICML).

Ribeiro, M. T.; Singh, S.; and Guestrin, C. 2016. “Why Should I Trust You?”: Explaining the Predictions of Any Classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

Rohrbach, A.; Hendricks, L. A.; Burns, K.; Darrell, T.; and Saenko, K. 2018. Object Hallucination in Image Captioning. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP).

Rousseeuw, P. J. 1987. Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis. Journal of Computational and Applied Mathematics, 20: 53–65.

Schaeffer, R.; Miranda, B.; and Koyejo, S. 2023. Are Emergent Abilities of Large Language Models a Mirage? In Advances in Neural Information Processing Systems (NeurIPS).

Tian, K.; Mitchell, E.; Zhou, A.; Sharma, A.; Rafailov, R.; Yao, H.; Finn, C.; and Manning, C. D. 2023. Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP).

Tong, S.; Brown, E.; Wu, P.; Woo, S.; Middepogu, M.; Akula, S. C.; Yang, J.; Yang, S.; Iyer, A.; Pan, X.; Wang, Z.; Fergus, R.; LeCun, Y.; and Xie, S. 2024a. Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs. In Advances in Neural Information Processing Systems (NeurIPS).

Tong, S.; Liu, Z.; Zhai, Y.; Ma, Y.; LeCun, Y.; and Xie, S. 2024b. Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Vo, A.; Nguyen, K.-N.; Taesiri, M. R.; Dang, V. T.; Nguyen, A. T.; and Kim, D. 2025. Vision Language Models are Biased. arXiv preprint arXiv:2505.23941.

Whitehead, S.; Petryk, S.; Shakib, V.; Gonzalez, J.; Darrell, T.; Rohrbach, A.; and Rohrbach, M. 2022. Reliable Visual Question Answering: Abstain Rather Than Answer Incorrectly. In European Conference on Computer Vision (ECCV).

Yue, X.; Ni, Y.; Zhang, K.; Zheng, T.; Liu, R.; Zhang, G.; Stevens, S.; Jiang, D.; Ren, W.; Sun, Y.; Wei, C.; Yu, B.; Yuan, R.; Sun, R.; Yin, M.; Zheng, B.; Yang, Z.; Liu, Y.; Huang, W.; Sun, H.; Su, Y.; and Chen, W. 2024. MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Yuksekgonul, M.; Bianchi, F.; Kalluri, P.; Jurafsky, D.; and Zou, J. 2023. When and Why Vision-Language Models Behave Like Bags-Of-Words, and What to Do About It? In International Conference on Learning Representations (ICLR).

Zeiler, M. D.; and Fergus, R. 2014. Visualizing and Understanding Convolutional Networks. In European Conference on Computer Vision (ECCV).

<table><tr><td>Model</td><td>Vision tower</td><td>Dim</td><td>Layers</td><td>Res.</td></tr><tr><td>LLaVA-1.5-7B</td><td>CLIP-ViT-L/14</td><td>1024</td><td>24</td><td>336</td></tr><tr><td>LLaVA-NeXT-7B</td><td>CLIP-ViT-L/14</td><td>1024</td><td>24</td><td>336</td></tr><tr><td>Idefics3-8B</td><td>SigLIP-SO400M</td><td>1152</td><td>27</td><td>364</td></tr><tr><td>Qwen3-VL-8B</td><td>native ViT</td><td>1152</td><td>27</td><td>dyn.</td></tr><tr><td>Qwen2.5-VL-7B</td><td>native ViT</td><td>1280</td><td>32</td><td>dyn.</td></tr><tr><td>Qwen2.5-VL-32B</td><td>native ViT</td><td>1280</td><td>32</td><td>dyn.</td></tr></table>

Table 4: Vision-tower specifications of the six VLMs, read from each released checkpoint’s configuration. “Dim” = final-layer feature dimensionality used by the own-tower probes (App. A.4); “dyn.” = native dynamic resolution. Only the two LLaVA models share a tower; Qwen3-VL’s tower is architecturally SigLIP-SO400M-shaped but separately trained.

## A Implementation Details

## A.1 Models, checkpoints, and vision towers

The six VLMs span three architecture families and four distinct vision towers (Table 4). This heterogeneity is load-bearing for the sample-intrinsic claim: cross-family VSI agreement (§2.3) cannot be explained by shared encoder weights, because cross-family pairs share none. Exact checkpoint identifiers and revision hashes are pinned in the released code.

## A.2 Region-detection pipeline details

The region used by region-blur VSI is produced in three stages. First, we parse the principal noun phrase from the question q using a lightweight off-the-shelf parser (no model-call cost); on POPE this is typically the queried object (“Is there a spoon in the image?” yields “spoon”), on MMVP it is the visual primitive being asked about, and on HallusionBench it is the chart, diagram, or scene element referenced in the question. Second, we query Grounding-DINO (base variant) (Liu et al. 2024d) for an open-vocabulary bounding box with a confidence threshold of 0.3. Third, we refine the box to a pixel mask using SAM-ViT-base (Kirillov et al. 2023) with the box prompt; the resulting mask is dilated by 4 pixels to avoid sharp boundary artefacts. When Grounding-DINO returns no box above threshold, the sample falls back to whole-image perturbation; the fallback rate is below 3% on POPE and MMVP, and approximately 19% on HallusionBench (chart-reading questions dominate the failures). Gaussian-blur perturbation uses a kernel size of 4σ+1 pixels.

## A.3 Answer-parsing rates and decoding degeneration

The per-cell n in Table 3 is the answer-parseable subset. The raw sample pool per benchmark is identical across models (e.g., the same 500 POPE sample IDs are presented to every model), but accuracy-dependent metrics drop samples whose generated answer cannot be parsed into a valid choice. The dominant cause is degenerate greedy decoding rather than region-detection fallback or verbalised-confidence extraction: under our greedydecoding setup, Qwen2.5-VL-7B produces repetitive degenerate output (an unbounded repetition of a single spurious token, e.g. addCriterion) on 333/500 POPE samples (67%), Qwen3-VL-8B on 141/500 (28%), Qwen2.5-VL-32B on 4/500, and the LLaVA-family models and Idefics3 on 0/500. Degenerate outputs are excluded rather than scored as errors: scoring them as wrong would conflate a decoding pathology with the visual-grounding failure this paper measures. VSI itself requires no answer parsing, so the distributional analyses of §2.2 and the cross-model Spearman correlations of §2.3 (computed on the inner join of per-pair sample IDs; $n _ { \mathrm { j o i n t } } { = } 5 0 0$ on POPE pairs) use the full pool. Per-cell parse rates are released alongside the persample CSVs.

## A.4 Encoder linear-probe protocol

All probes are L<sub>2</sub>-regularised logistic regressions (C=1.0) on ℓ -normalised final-layer image features, with labels perturbed (+1) vs unperturbed (−1) and grouped 5-fold crossvalidation that assigns both views of an image to the same fold, so no image appears in both train and test. Feature dimensionalities follow each tower (Table 4): 1024 for the CLIP-ViT-L/14 towers, 1152 for SigLIP-SO400M and Qwen3-VL’s tower, 1280 for the Qwen2.5-VL towers.

Own-tower probes (main result). For each of the six VLMs we extract pooled features from the model’s own vision tower and fit the probe on the shared low-VSI pool: the 83 images flagged by Qwen3-VL (VSI<0.05) within a 200- sample pool (100 POPE + 100 MMVP), with a matched high-VSI control from the same pool. The ∆argmax rates of Table 2 are computed on the same 83 images, so the gap is a same-subset difference. Low-VSI accuracies (fold std): Qwen3-VL 0.734±0.033; Qwen2.5-VL-7B 0.764±0.069; Qwen2.5-VL-32B 0.741±0.064; LLaVA-1.5 0.722±0.109; LLaVA-NeXT 0.789±0.040; Idefics3 0.753±0.041. On the high-VSI control the same probes reach 0.86–0.91, confirming that the low-VSI gap is not a probe-capacity ceiling. Permodel probe outputs are released with the code; the Qwen3- VL value exactly reproduces an earlier single-model pilot run, cross-validating the generalised pipeline.

Reference-CLIP probe (external witness). We additionally probe a frozen openai/clip-vit-large-patch14 (224px, feature dimension 1024), chosen independently of the VLMs under test; none of the six uses this exact encoder (the LLaVA models use the 336px variant, Idefics3 uses SigLIP-SO400M, and the Qwen models use natively trained ViTs). Because this probe sees only the images, its accuracy is a property of the sample set. On the 83 images with Qwen3-VL VSI <0.05 (each contributing a clean and a perturbed view; 166 rows), per-fold accuracies are [0.824, 0.853, 0.882, 0.781, 0.781]: mean 0.82, fold standard deviation 0.04 (95% t-interval [0.77, 0.88]). An RBF-kernel variant (γ=1/d) exceeds the linear probe by at most 0.02 on any fold, indicating the signal is linearly available rather than requiring elaborate decoding.

## A.5 Verbalised-confidence prompt template

Verbalised confidence is elicited by appending a confidence question to the model’s own answer in a second forward pass:

[question] Your answer: [answer]. On   
a scale from 0 to 100, how confident   
are you in the answer? Respond with the   
number only.

We extract the first integer in the response with a regex, divide by 100, and clip to [0, 1]. Samples on which the model emits no integer (typically because it adds qualifying text like “somewhat confident”) are dropped from the verbalisedconfidence analysis but retained for VSI and max-prob analyses. The drop rate is 4%–11% per cell. We use the same generation hyperparameters as for the main answer (greedy decoding, max tokens=8), so the verbalised score reflects what the model says when explicitly asked, not a sampled posterior.

## B Extended Results

## B.1 Low-VSI failure-mode breakdown (figure)

For each cell we sort samples by VSI, take the bottom quintile, and partition by (correct vs wrong) × (softmax confidence above vs below the cell’s median). The dangerous quadrant (wrong + high confidence) is the abstention target a softmax-only signal would miss: these are predictions the model believes most strongly. The median split is computed per cell, so “high confidence” is relative to the model’s own confidence distribution on that benchmark rather than an absolute threshold; this prevents a miscalibrated model from emptying a quadrant by construction. Its size varies systematically across cells: largest on HallusionBench (25%–30% on most models), modest on MMVP (10%–25%), and smallest on POPE (5%–15%). LLaVA-NeXT on HallusionBench carries the largest absolute dangerous fraction in the grid, consistent with its low aggregate accuracy (0.42) and weak calibration on figure-reading questions. Qwen2.5-VL-32B carries the smallest, consistent with its stronger calibration baseline.

## B.2 Cross-model VSI agreement detail

The three pairs shown are chosen to span the within-family vs cross-family spectrum: Qwen3-VL-8B vs Qwen2.5-VL-7B (same family, different tower and decoder generations), Qwen3-VL-8B vs LLaVA-1.5-7B (cross-family, disjoint vision towers), and Qwen2.5-VL-32B vs Idefics3-8B (crossfamily at the largest scale gap). On all three pairs the lowlow corner concentrates the highest joint error rates, with the diagonal also visibly elevated, so the model pair tends to err on the same samples when both flag them as visually insensitive.

Full pairwise statistics. Pairwise Spearman, Pearson, and low-VSI Jaccard tables for all 15 (model-A, model-B) pairs on each benchmark are released with the code. The grandmean Spearman across the three perceptual benchmarks (region-blur, 15 pairs) is $\rho { = } \mathrm { + 0 . 4 0 }$ , with per-benchmark means of 0.55 (POPE), 0.34 (MMVP), and 0.32 (Hallusion-Bench). Decomposing by family: same-family pairs (three Qwen pairs, one LLaVA pair) average $\rho { = } 0 . 5 9 / 0 . 4 7 / 0 . 4 6$ on POPE/MMVP/HallusionBench vs 0.54/0.29/0.27 for the eleven cross-family pairs (pooled 0.51 vs 0.37), and the same-family advantage holds in direction on every (benchmark, perturbation) combination, including whole-image VSI (0.39 vs 0.21 pooled). Note the POPE gap is small (+0.05) and several cross-family pairs rank among the strongest on that benchmark, which is why the main text claims the family effect only on average.

Permutation tests. Four representative pairs were tested against a 10<sup>5</sup>-shuffle two-sided permutation null (seed 42; per-pair null distributions released with the code):
<table><tr><td>Pair</td><td>Bench</td><td>n</td><td>ρ p</td><td></td></tr><tr><td>Qwen3 × Qwen2.5-7B</td><td>POPE</td><td>500</td><td>+0.502</td><td> $< 1 0 ^ { - 5 }$ </td></tr><tr><td>Qwen3 × LLaVA-1.5</td><td>POPE</td><td>500</td><td>+0.453</td><td> $< 1 0 ^ { - 5 }$ </td></tr><tr><td>Qwen3 × Qwen2.5-32B</td><td>POPE</td><td>500</td><td>+0.547</td><td> $< 1 0 ^ { - 5 }$ </td></tr><tr><td>Q2.5-32B × Idefics3</td><td>MMVP</td><td>300</td><td>+0.201</td><td> $3 . 4 \times 1 0 ^ { - 4 }$ </td></tr></table>

Even the weakest tested pair, the largest architectural and scale gap in our grid on the weakest-agreement benchmark, is separated from its null by more than three orders of mag nitude in p.

## B.3 Sample exemplars

We selected these six samples by intersecting the bottom VSI quintiles of the three baseline models with the subset on which all three are simultaneously wrong with high confidence. The intersection is small (n=11 samples across the three perceptual benchmarks); we display the six most diverse by question type. Each sample is exhibit-A for the sample-intrinsic claim: three independently-trained VLMs converge on the same wrong, confident, vision-ignoring answer, despite encoding the perturbed region in four distinct vision towers (Table 4). We did not cherry-pick within the intersection: the qualitative pattern (object denied, primitive flipped, colour misread) is representative of the wider low-VSI failure population.

## B.4 Signal complementarity heatmap (all six models)

If VSI and max-prob were perfectly correlated, the heatmap would be diagonal (errors concentrate along the rank-rank diagonal) and combining the two signals would carry no additional information. If they were fully independent, errors would distribute uniformly across the joint quintile grid. The observed pattern is between these extremes: errors concentrate in the bottom-left corner (both signals flag uncertainty), but the off-diagonal cells (one signal flags uncertainty, the other does not) still contain a non-trivial error mass: specifically, the cell at (low VSI, high max-prob) on the bottom row carries the confidently-wrong-while-ignoring-vision predictions that motivate the hybrid ensemble of §4.2. The pattern is qualitatively the same across all six models, with quantitative differences in mass (Qwen2.5-VL-32B’s grid is more concentrated near the corner; LLaVA-1.5-7B’s is more uniform).

![](images/79c8f9fb4f579607504ccfa3d1ac9603cbaf3a0089d24fb49097d55870f813f4.jpg)  
Figure 4: Four-quadrant classification of bottom-VSI-quintile samples across 6 models × 3 benchmarks. Red (wrong + high conf) denotes confidently-wrong while-ignoring-vision predictions; this is the hallucination-flagging target. Cross-referenced from §3.1.

![](images/cf26ece3e27d3a508ca0e8b4cace2f96c2fe251e1790d07918f7d36c6e0ba814.jpg)

![](images/b3b2181beb59394a63aa3a123f4600b5ec8b09347a0a229d07cf763e9673c220.jpg)

![](images/1b5f34688f460e1ffd2974ee757bd3156f0840576a14e32f142db206d4c1cae4.jpg)  
Figure 5: Joint VSI quintiles across three representative model pairs on POPE, coloured by joint error rate. The low-VSI corner concentrates joint errors; this is the visual form of the sample-intrinsic claim of §2.3.

## B.5 Verbalised confidence on POPE

Per-model Pearson correlation between negative VSI and negative verbalised confidence on POPE: Qwen3-VL-8B r=−0.083, Qwen2.5-VL-7B r=−0.018, Qwen2.5-VL-$3 2 \mathbf { B } \ r = + 0 . 0 4 1 , \mathrm { L L a V A - 1 . 5 - 7 } \mathbf { B } \ r = - 0 . 0 2 9 , \mathrm { L L a V A - N e X T }$ $r { = } { + } 0 . 0 1 2 .$ , Idefics3-8B $r { = } { + } 0 . 0 5 1$ . None exceeds 0.10 in absolute value. The heatmap interpretation mirrors Appendix B.4: low-low corners concentrate error mass, but the off-diagonal cells where one signal is low and the other is high contain meaningful additional error mass that justifies the hybrid ensemble. Verbalised confidence on POPE is itself only marginally informative as a standalone abstention signal (AUROC typically 0.55–0.65) but contributes to the best hybrid on the cells where the verbal score is reliably extractable.

## B.6 Reliability diagrams

For each cell we plot empirical accuracy vs predicted confidence in 10 equal-width bins. Max-prob (blue) is the bestcalibrated signal in most cells, often lying close to the diagonal. Normalised-VSI (orange) and verbalised confidence (teal) are noisier and frequently sit above the diagonal at the low-confidence end, reflecting their limited dynamic range (most VSI mass concentrates in the left tail of §2.2; verbalised scores cluster near integer multiples of 10). The bins with fewer than 10 samples are visually noisier, especially on HallusionBench where per-cell n is smaller. Expected Calibration Error is $\begin{array} { r } { \mathrm { E C E } = \sum _ { b } \frac { n _ { b } } { N } \left| \mathrm { a c c } _ { b } - \mathrm { c o n f } _ { b } \right| } \end{array}$ over the 10 bins; the Brier score is the mean squared error between stated confidence and binary correctness. Both are released per cell alongside the per-sample CSVs.

## B.7 MMStar per-category AUROC<sub>VSI</sub>

MMStar samples per macro-category range from $n { = } 2 8$ (Qwen3-VL math) to n=50 (most cells) after our filtering; cells with $n { < } 2 5$ are excluded from the headline numbers but visible in the released CSV. We restrict MMStar to whole-image perturbation: the multi-choice prompts often span a layout (diagram, chart, equation panel) for which open-vocabulary region detection is unreliable, so a localised region-blur would conflate VSI with detection failure. The capability-conditional pattern is sharpest on math and science-and-technology because these macro-categories have the largest gap between “model uses the chart and answers correctly” and “model ignores the chart and outputs an answer from prior knowledge”; perceptual macro-categories (coarse / fine-grained perception) have less of this gap to exploit and consequently weaker VSI signal.

<table><tr><td>Category</td><td>Qwen3-VL-8B</td><td>Qwen2.5-VL-7B</td><td>Qwen2.5-VL-32B</td><td>LLaVA-1.5-7B</td><td>LLaVA-NeXT</td><td>Idefics3-8B</td></tr><tr><td>Coarse perception</td><td>0.65</td><td>0.48</td><td>0.61</td><td>0.57</td><td>0.55</td><td>0.64</td></tr><tr><td>Fine-grained perception</td><td>0.42</td><td>0.56</td><td>0.59</td><td>0.54</td><td>0.37</td><td>0.58</td></tr><tr><td>Instance reasoning</td><td>0.52</td><td>0.59</td><td>0.66</td><td>0.58</td><td>0.55</td><td>0.46</td></tr><tr><td>Logical reasoning</td><td>0.55</td><td>0.66</td><td>0.62</td><td>0.33</td><td>0.56</td><td>0.67</td></tr><tr><td>Math</td><td>0.61</td><td>0.61</td><td>0.85</td><td>0.29</td><td>0.67</td><td>0.58</td></tr><tr><td>Science &amp; technology</td><td>0.61</td><td>0.63</td><td>0.87</td><td>0.59</td><td>0.51</td><td>0.51</td></tr><tr><td>ALL</td><td>0.56</td><td>0.58</td><td>0.69</td><td>0.54</td><td>0.55</td><td>0.59</td></tr></table>

Table 5: Full MMStar per-category $\mathbf { A U R O C } _ { \mathrm { V S I } }$ matrix (6 models × 6 macro-categories + overall). $\mathbf { B o l d } = \geq 0 . 6 5 ;$ italics = inverse direction $( \mathrm { A U R O C } < 0 . 5 )$ . The capability-conditional pattern of §3.2 is visible: math and science columns peak on the strongest model in our grid and invert on the weakest (LLaVA-1.5-7B math AUROC 0.29).

## C Robustness Analyses

## C.1 σ-stability figure and protocol

We selected four representative cells for the σ ablation (Qwen3-VL-8B POPE, Qwen3-VL-8B MMVP, Qwen2.5- VL-32B POPE, LLaVA-NeXT POPE) on the basis of sample-count adequacy $( n ~ \geq ~ 2 6 8$ each) and architectural diversity (two Qwen variants, two model families). For each cell we recomputed VSI from scratch at $\sigma \in \{ 1 0 , 2 0 , 4 0 \}$ (a total of 12 additional sweeps). The $\mathbf { A U R O C } _ { \mathrm { V S I } }$ delta across σ values is at most 0.048 on any cell; the per-sample VSI rank correlation across σ pairs ranges from 0.76 (Qwen3-VL MMVP, σ=10 vs 40) to 0.97 (LLaVA-NeXT POPE, σ=20 vs 40). Together these justify treating σ=20 as a non-loadbearing implementation choice rather than as a tuning hyperparameter.

## C.2 σ-robustness numerical values

The Spearman correlations should be read as: even on the most sensitive (10 vs 40) σ-pair on the most rankingsensitive cell (Qwen3-VL MMVP), 76% of the rank order is preserved. The widest AUROC swing in any cell is on Qwen3-VL-8B POPE, from 0.543 (σ=10) to 0.591 (σ=20): a swing of 0.048. By comparison, at fixed σ the AUROC ranges across the four cells of Table 6 by more than 0.17, so the σ-induced variation is small relative to cell-to-cell variation.

## C.3 Region-VSI vs whole-image-VSI correspondence

Region-blur and whole-image-blur VSI are two perturbation variants. We report their per-sample rank correlation honestly. The grand-mean Spearman across all 18 (model, benchmark) cells is $\rho { = } + 0 . 2 4$ , with per-cell values ranging from $\rho { = } - 0 . 0 0 \mathrm { ( L L a V A { - } 1 . 5 – 7 B }$ on HallusionBench) to $\rho { = } + 0 . 6 5$ (Idefics3-8B on MMVP); MMVP cells average $\rho { = } + 0 . 3 6$ , POPE averages $\rho { = } + 0 . 1 7$ , and HallusionBench averages $\rho = + 0 . 2 0$ . The two variants thus capture overlapping but distinct sample populations. This is consistent with our §4.2 finding that the equal-weight sum is the best signal on a plurality of cells: if the two variants ranked the same samples identically, their ensemble would carry no additional information. The MMVP advantage is interpretable: MMVP questions are about discrete visual primitives whose presence is captured by both localised and global blurring, so the two variants flag overlapping samples. POPE’s lower correlation reflects that object-presence questions are sensitive to local-region content (the queried object) but only weakly to global image statistics. Per-cell values are released with the code.

## D Reproducibility and Scope

## D.1 Reproducibility, hardware, and computational budget

Hardware. All experiments use BF16 precision on NVIDIA A100-80GB GPUs (single-GPU jobs except for Qwen2.5-VL-32B which uses two GPUs with naive model parallelism). Per-sample wall-clock time at σ=20 regionblur is approximately 2.3 s on Qwen2.5-VL-32B and 0.7– 1.1 s on 7B-class models. Total compute for all results in the paper is approximately 32 A100-hours, on standby-QOS shared-cluster GPUs.

Generation hyperparameters. Greedy decoding (do sample=False) for all answer generation; max new tokens=8 for binary POPE / multi-choice MMVP / MMStar, max new tokens=64 for open-ended HallusionBench. Temperature does not apply under greedy. VSI is computed from the top-50 next-token distribution after the answer-prefix tokens; KL divergence is computed on the aligned support of the top-50 candidates from both clean and perturbed forward passes.

Software. PyTorch with HuggingFace Transformers, BF16 throughout; exact package versions are pinned in the released environment file. Probes use scikit-learn 1.8.0. We pin this version deliberately: GroupKFold split assignment varies across scikit-learn releases, which moves the reference-probe mean by up to 0.01 (0.824 on 1.8.0 vs 0.813 on an adjacent release); the fold standard deviation we report (±0.04) dominates this variation, but bitwise reproduction requires the pinned version.

Statistics. Bootstrap CIs use B=1000 resamples; crossmodel Spearman permutation tests use $N { = } 1 0 ^ { 5 }$ shuffles (seed 42; per-pair nulls released with the code); pairedbootstrap signal comparisons in §4.2 use B=1000. We do not apply multiple-testing correction within Table 3 because the per-cell comparisons are interpreted as exploratory.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Bench</td><td rowspan="2">n</td><td colspan="3">AUROC @  $\sigma$ </td><td colspan="3">Spearman across σ</td></tr><tr><td>10</td><td>20</td><td>40</td><td>10-20</td><td>10-40</td><td>20-40</td></tr><tr><td>Qwen3-VL</td><td>POPE</td><td>359</td><td>0.543</td><td>0.591</td><td>0.590</td><td>0.91</td><td>0.87</td><td>0.92</td></tr><tr><td>Qwen3-VL</td><td>MMVP</td><td>268</td><td></td><td>0.4760.487</td><td>0.475</td><td>0.82</td><td>0.76</td><td>0.90</td></tr><tr><td>Qwen2.5-32B</td><td>POPE</td><td></td><td></td><td>4960.383 0.4170.424</td><td></td><td>0.91</td><td>0.88</td><td>0.95</td></tr><tr><td>LLaVA-NeXT POPE</td><td></td><td></td><td>5000.514 0.541 0.534</td><td></td><td></td><td>0.94</td><td>0.90</td><td>0.97</td></tr></table>

Table 6: σ-robustness numerical values. $\mathbf { A U R O C } _ { \mathrm { V S I } }$ varies by at most 0.05 across $\sigma \in \{ 1 0 , 2 0 , 4 0 \}$ on any cell; per-sample VSI Spearman across σ pairs is $\geq 0 . 7 6$ . The defaul $\sigma { = } 2 0$ is not load-bearing.

Data and code. Full per-cell aggregate statistics (n, accuracy, fraction VSI<0.05, AUROC, quintile ratios) are released together with all per-sample CSVs. The full grid spans 6 models × 4 benchmarks: the three perceptual benchmarks are probed with both region-blur and whole-image perturbation, while MMStar uses whole-image only, yielding 42 (model, benchmark, perturbation) cells in total. Random seeds are fixed (seed=42) throughout.

## D.2 Metric definitions

$\mathrm { A U R O C _ { V S I } }$ is computed with per-sample error as the positive class and −VSI as the ranking score, so values above 0.5 mean lower visual sensitivity predicts higher error risk; the same convention (negated score, error as positive class) applies to max-prob and verbalised confidence, making all AUROC columns of Table 3 directly comparable. PRR@80 is the prediction–rejection ratio at 80% coverage: with the worst-ranked 20% of predictions rejected, it is the achieved reduction in risk divided by the reduction an oracle achieves by rejecting only errors, where 1 is oracle-perfect, 0 is no better than random rejection, and negative values mean the signal preferentially rejects correct predictions. Hybrid signals z-normalise each component within the cell and sum with equal weights; the learned variant of §4.2 replaces the equal weights with logistic-regression coefficients fitted on a held-out calibration split.

## D.3 Limitations beyond the main paper

We surfaced four limitations in §6. Three additional ones are worth noting in the appendix. First, our analysis is restricted to next-token softmax distributions; we do not measure intermediate-layer activations, attention patterns, or KV-cache content, so the routing-failure interpretation of §2.4 is necessarily an input-output characterisation rather than a structural one. Second, the verbalised-confidence signal is sensitive to prompt phrasing (we tried two variants on Qwen3-VL POPE and observed AUROC differences up to 0.06); the values we report are for the prompt template in App. A.5 only. Third, the encoder probe uses last-layer features only; probing intermediate encoder layers might reveal where in each tower the perturbation-sensitive features live, but does not change the input-output gap we report. None of these limitations changes our headline claims, but each is a natural direction for follow-up work.

![](images/a6c8c59a41afab6d80923fbd796d39e6f8febd5492b10b6f09f0e0eb0ee22f72.jpg)

Q. Is there a tv in the image?

Gold: yes

![](images/e8019408f74aee72c00ec6d25d986826931a6adad8ada3d3d1e1e21b36ca2958.jpg)

Q. Is there a backpack in the image?

Gold: yes

![](images/b4613f0d246f42736e12aa6651d1d5a34c35a3ee35806ba02c2d575feb566959.jpg)

Q. Is it a slice of cake or a whole cake in the picture?

Gold: (a)

![](images/867dc41d7381f067d330cca5a36afad1d9c73b1c899ebf7d2cba5fc99f6a4a37.jpg)

Q. Are these oranges hanging on the tree?

Gold: (b)

HallusionBench

Q. Are two circles in the image the same color? Yes or No

![](images/cdac76ed6442407d987df2c0aadf813786d239d86c26b06b1fea352349d6a3f9.jpg)

Gold: yes

HallusionBench

Q. Are two circles in the image different color? Yes or No

![](images/2988b75e6bffcd5b5822e1d88c193c88743883888ecb7102b59dd8cbc89c9a2d.jpg)

Gold: no

Qwen3-VL-8B VSI=0.001 max-prob=0.992 wrong “Based on the image provided, there i…”

Qwen2.5-VL-7B VSI=0.001 max-prob=0.999 wrong “No, there is no TV in the image. The…”

$$
\begin{array} { r l } & { \mathrm { L L a V A - 1 . 5 - 7 B } \qquad \forall \mathsf { S l } = \mathbf { 0 . 0 3 2 } \qquad \mathsf { m a x \mathrm { - } p r o b { = } } 0 . 7 9 3 \qquad \mathsf { w r o n g } } \\ & { \qquad \mathrm { ~ } ^ { \mathrm { a } } N T \mathrm { : ~ } N o , \ t h e r e \ i s \ n o \ T V i n \ t h e \ i m a g e . . . ^ { \prime \prime } } \end{array}
$$

Qwen3-VL-8B VSI=0.000 max-prob=0.971 wrong “You are correct, there is no backpac…”

$$
\begin{array} { r l r l r } { { \mathbf { Q } } { \mathbf { w } } { \mathbf { e } } { \mathbf { n } } 2 . 5 { \mathbf { - } } { \mathbf { V } } \mathbf { L } { \mathbf { - } } 7 { \mathbf { B } } } & { } & { { \mathbf { V } } { \mathbf { S } } { \mathbf { l } } { \mathbf { = } } 0 . 0 0 9 } & { } & { { \mathrm { m a x - p r o b } } { \mathbf { = } } 0 . 6 7 3 } & { } & { { \mathbf { w } } { \mathbf { r o n g } } } \\ { a d d C r i t e r i o n ^ { \prime \prime } } \end{array}
$$

$$
\begin{array} { r } { \begin{array} { c c l } { \mathbf { L L a V } \mathbb { A - 1 . 5 - 7 B } } & { \mathbf { V S } \mathbf { \mathsf { I } } = \mathbf { 0 . 0 1 5 } } & { \mathsf { m a x . p r o b { = } } 0 . 8 3 \mathbf { 1 } } & { \mathbf { c o r r e c t } } \\ { \therefore Y e s , \ t h e r e \ i s \ a \ b a c k p a c k \ i n \ t h e \ i m . . . ^ { \prime \prime } } & \end{array} } \end{array}
$$

$$
\begin{array} { r l } { { \mathbf { Q } } { \mathbf { w } } { \mathbf { e } } { \mathbf { n } } 3 { - } { \mathbf { V } } \mathbf { L } { - } 8 \mathbf { B } \qquad } & { { } { \mathsf { V } } { \mathsf { S } } { \mathsf { I } } { = } 0 . 0 0 1 \qquad { \mathrm { m a x } } { \cdot } { \mathsf { p r o b } } { = } 0 . 9 9 1 \quad { \mathrm { w r o n g } } } \\ { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \boldsymbol { \epsilon } } { d e s s e r t ^ { \prime \prime } } \end{array}
$$

$$
\begin{array} { r l r l r l } { { \mathbf { Q } } { \mathbf { w } } { \mathbf { e } } { \mathbf { n } } 2 . 5 - \mathbf { V } \mathbf { L } - 7 \mathbf { B } } & { } & { \forall 5 | = 0 . 0 0 0 } & { } & { \mathsf { m a x } \mathsf { - p r o b } { = } 0 . 6 7 9 } & { } & { \mathsf { c o r r e c t } } \\ { a d d C r i t e r i o n ^ { \prime \prime } } \end{array}
$$

$$
\begin{array} { r l r } { \mathsf { L L a V A - 1 . 5 - 7 B } \qquad } & { { \mathsf { V S } } \mathsf { I } = 0 . 1 2 2 \qquad } & { \mathsf { m a x - p r o b } = 0 . 5 1 0 \qquad \mathsf { c o r r e c t } } \\ { ^ { \prime } \mathit { S S I S I A N T : } ( b ) \ W h o l e ^ { \prime \prime } } \end{array}
$$

$$
\begin{array} { r l r l } { { \bf Q } { \bf w } { \bf e n } 3 - { \bf V } { \bf L } - 8 { \bf B } \quad } & { } & { { } \quad \forall { \sf S I } = 0 . 0 0 0 } & { \quad \mathrm { m a x - p r o b } = 0 . 9 8 9 \quad \mathrm { w r o n g } } \\ { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \end{array}
$$

$$
\begin{array} { r l r l r l } { { \mathbf { Q } } { \mathbf { w } } { \mathbf { e } } { \mathbf { n } } 2 . 5 { \mathbf { - } } { \mathbf { V } } \mathbf { L } { \mathbf { - } } 7 { \mathbf { B } } } & { } & { { } { \mathbf { V } } { \mathbf { S } } { \mathbf { l } } { \mathbf { = } } 0 . 0 3 0 } & { } & { { } { \mathrm { ~ m a x - p r o b } } { \mathbf { = } } 0 . 6 6 5 } & { } & { { } { \mathbf { w } } { \mathbf { r } } { \mathbf { o } } { \mathbf { n } } { \mathbf { g } } } \\ { a d d C r i t e r i o n ^ { \prime \prime } } \end{array}
$$

$$
\begin{array} { r l r } { \mathsf { L L a V A - 1 . 5 - 7 B } \qquad } & { { \mathsf V S l } = 0 . 1 2 5 \qquad } & { \mathsf { m a x - p r o b } = 0 . 6 6 3 \qquad \mathsf { w r o n g } } \\ { \omega _ { O } \eta _ { Y } . } & { { \cal A S S I S T A N T : } \ N o ^ { \prime \prime } } \end{array}
$$

$$
\begin{array} { r } { \begin{array} { c c c } { \mathbf { Q } \mathbf { w } \mathbf { e } \mathbf { n } 3 - \mathbf { V } \mathbf { L } - \mathbf { 8 } \mathbf { B } \quad \quad } & { \mathbf { V } \mathbf { S } \mathbf { l } = \mathbf { 0 } . 0 0 0 \quad } & { \mathrm { m a x - p r o b } = 1 . \mathbf { 0 } 0 0 \quad \mathrm { w r o n g } } \\ { \mathbf { a } \quad } & { \mathbf { a } \quad } & { \mathrm { i n c e } ^ { \prime } } \end{array} } \end{array}
$$

$$
\begin{array} { r l r l r l } { { 9 } \mathbf { w } \mathbf { e n } 2 . 5 \mathbf { - } \mathbf { } \mathbf { \overline { { V L - 7 B } } } } & { } & { \quad \forall 5 | = 0 . 0 0 0 } & { } & { \mathrm { m a x - p r o b } = 0 . 5 6 2 } & { } & { \mathrm { c o r r e c t } } \\ { \mathbf { \overline { { \Gamma } } } \varphi _ { S } , \mathrm { ~ } } \end{array}
$$

$$
\begin{array} { r l } { { \mathbf { L } \mathbf { L } \mathbf { \overline { { a } } } \mathbf { V } \mathbf { A } \mathbf { - 1 } . 5 \mathbf { - 7 } \mathbf { B } \qquad } } & { { \mathbf { V } \mathbf { S } \mathbf { l } } \mathbf { = } \mathbf { 0 } . 0 \mathbf { 0 } \mathbf { 0 } \qquad \mathsf { m a x } \mathbf { \cdot p } \mathsf { r o b } \mathbf { - } 0 . 7 5 5 \qquad \mathsf { w r o n g } } \\ { \omega _ { C } \qquad } & { { \mathsf { M S S } } / S T S I N T ; \ N o ^ { \prime \prime } } \end{array}
$$

$$
\begin{array} { r l r l r } { \mathbf { Q } \mathbf { w } \mathbf { e n } 3 - \mathbf { V } \mathbf { L } - 8 \mathbf { B } } & { } & { \mathbf { V } \mathbf { S } \mathbf { l } = 0 . 0 0 0 } & { } & { \mathbf { m } \mathbf { a } \mathbf { x } \mathbf { - } \mathbf { p } \mathbf { r } \mathbf { o } \mathbf { b } = 0 . 9 9 9 \quad \mathrm { w r o n g } } \\ { \mathbf { \sigma } \mathbf { \sigma } \mathbf { A } \boldsymbol { n } { s } { w e r } ; \gamma e s ^ { \prime \prime } } \end{array}
$$

$$
\begin{array} { r l r l r l } { { \mathbf { Q } } { \mathbf { w } } { \mathbf { e } } { \mathbf { n } } 2 . 5 { \mathbf { - } } { \mathbf { V } } \mathbf { L } { \mathbf { - } } 7 { \mathbf { B } } } & { } & { { } { \mathbf { V } } { \mathbf { S } } { \mathbf { l } } { \mathbf { = } } 0 . 0 2 4 } & { } & { { } { \mathrm { m a x - p r o b } } { \mathbf { = } } 0 . 7 0 6 } & { } & { { } { \mathbf { w } } { \mathbf { r } } { \mathbf { o } } { \mathbf { n } } { \mathbf { g } } } \\ { a d d C r i t e r i o n ^ { \prime \prime } } \end{array}
$$

$$
\begin{array} { r l r } { \mathbf { L L } \mathbf { a } \mathbf { V } \mathbb { A } \mathbf { - 1 . 5 . 7 } \mathbf { B } \quad } & { { } \mathbf { V } \mathbf { S } \mathbf { l } \mathbf { = } \mathbf { 0 . 0 0 0 } } & { \quad \mathrm { m a x . p r o b } \mathbf { = } \mathbf { 0 . 8 8 } \mathbf { 1 } } & { \quad \mathrm { c o r r e c t } } \\ { \omega _ { O T \ : n O . } \mathbf { A } S S I S T A N T ; \ : Y e s ^ { \prime \prime } } & { { } } & { } \end{array}
$$

Figure 6: Six low-VSI samples that all three baseline VLMs (Qwen3-VL-8B, Qwen2.5-VL-7B, LLaVA-1.5-7B) answer incorrectly with softmax confidence above 0.99 and VSI < 0.01. POPE examples (rows 1–2): clearly-visible objects denied. MMVP examples (rows 3–4): visual primitives misclassified. HallusionBench examples (rows 5–6): colour-comparison errors on coloured shapes.

![](images/51ef755fd738f51d1b13723d47a870576e52703d0a767ddb2f1e4edb58a2749a.jpg)

![](images/24fac7f7aaff92ddb443559ff72ee2b45e2ef4443fd1574cde90284c45b18ec3.jpg)

![](images/2e829c253f685580a5294871e5f8fd67b93acd4ae9543d2a7a62a99fff2dadcd.jpg)

![](images/41d59a1d636023b31ab4e275ebc6140e12e127f8e55d11ccca16ef893bcfbbd0.jpg)

![](images/6583f81a40ae8b490ca068bec0bae730fa0cceb029acf0f81892832190fce867.jpg)  
Figure 7: Per-model mean error rate as a function of the joint quintile of VSI and softmax max-probability, on POPE. The lowlow corner concentrates errors across all six models; the high-high corner has lowest error rates. The two signals are partially redundant on confident-correct predictions but capture different failure modes for confident-wrong ones.

![](images/534fdd418a6ddbe6200b090bf7eb3a5605fe13355b44519055e083ba43fe61e2.jpg)

![](images/e6e6718e84aba7f0e0b49a230904b4be1662d54ca49ad2b1b2a4d6c640d8c447.jpg)

![](images/2990740ad4753a7891bba226a49c41694b81a1cb2f1ab9f24a018b18e68623a7.jpg)

![](images/df0348b1e8a60c2cc190da853399c4d189c1b35bffbdefcf04178895b3fe7e19.jpg)

![](images/90b486092e523f1d82a6c1220c7b770579c65e47d2e40e51691cf7ba689eddf4.jpg)

![](images/2e5a698f2c83da42e2bb94146827ae999c46ecb80fcea4dd6fdf9fd2a8df1cf3.jpg)  
Figure 8: Per-model joint VSI × verbalised-confidence quintile error-rate heatmaps on POPE. Pearson correlation between negative-VSI and negative-verbalised confidence is $| r | < 0 . 1 0$ on every model, supporting independence (§4.2).

![](images/5009b01852bd12f2970c9722ea9a38a04d103cfc8f19b4bdeab7f19d4d53c597.jpg)  
Figure 9: Reliability curves (per (model, benchmark) cell) for max-prob, normalised-VSI, and verbalised confidence. Max-prob is mostly aligned with the diagonal; VSI and verbalised are noisier due to limited dynamic range.

![](images/7bff3d1829ad4748c49318af49e1b8b16071cdf2f2dbe07f0e2fcbdb94048193.jpg)

![](images/c3dba66bdef8b038b2c66bd325c18f49a04d18f3152a3eb7de828138f02dfb13.jpg)  
Figure 10: Left: AUROC as a function of perturbation strength σ on four representative cells. The signal is flat within ±0.05. Right: per-sample VSI Spearman rank correlation across σ pairs reaches 0.76–0.97. Numerical values in Table 6. Cross-referenced from §4.3.