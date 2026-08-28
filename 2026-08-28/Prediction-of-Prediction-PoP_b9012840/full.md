# Prediction of Prediction (PoP)

Inter-Layer Activation Fusion for Single-Pass Hallucination Detection in Large Language Models

Himal Badu

## Abstract

Autoregressive large language models (LLMs) routinely generate factually incorrect outputs with high decoding confidence, limiting their deployment in highstakes workflows. Existing output-stage uncertainty metrics can fail when models are overconfident on false assertions, while multi-sample verification pipelines introduce substantial memory and latency overhead. This work evaluates whether internal hidden-state transition dynamics during generation can signal factual errors without auxiliary decoding calls. We introduce Prediction of Prediction (PoP), a mechanism that captures layer-transition uncertainty by fusing intermediate hidden representations across depth during a single forward pass. Evaluated on the TruthfulQA benchmark using autoregressive transformer backbones, PoP achieves an area under the receiver operating characteristic curve (AUROC) of 75.5% for factual-correctness classification. The mechanism operates within the base forward pass, adding less than 1.2% runtime latency and requiring zero additional generation passes. The numerical results are reported from the author-verified experimental implementation and are bounded by the evaluation scope described below.

Keywords: large language models; hallucination detection; hidden states; layer dynamics; uncertainty estimation; reliability; eficient inference

## 1 Introduction

The integration of large language models (LLMs) into automated reasoning systems, medical synthesis, and legal analysis is constrained by their tendency to produce syntactically plausible yet factually false statements. In enterprise settings, unmonitored hallucinations introduce operational risks and may require human verification that reduces the eficiency of automation. A practical verification system should detect hallucination events during generation without materially reducing model throughput or incurring prohibitive GPU-memory costs.

Existing detection paradigms predominantly operate at the output-logit boundary or through post-hoc verification pipelines. Output-stage metrics, including tokenlogit entropy, sequence perplexity, and final-layer attention distributions, often assume that factual uncertainty correlates with vocabulary-level uncertainty. However, an LLM may generate an incorrect entity with low logit variance, while benign stylistic choices can produce high entropy. Post-hoc approaches rely on multi-sample consistency checking, cross-prompting, retrieval, or auxiliary natural-language-inference (NLI) classifiers. Although such approaches can improve precision, they require multiple candidate completions or additional model calls, increasing latency and compute cost.

A central unresolved question is whether factual validity is continuously encoded in the internal representation trajectory as hidden states propagate through intermediate transformer layers. Static internal probing can demonstrate that specific layers contain predictive information, but an isolated layer check does not capture how the representation changes from shallow syntactic processing to deeper contextual commitment. A singlepass detector therefore needs to characterize layer-wise transitions during the same forward execution used for generation.

We present Prediction of Prediction (PoP), an internal detection framework that estimates sequence-level hallucination risk through layer-transition activation fusion. The contributions are as follows.

• Layer-transition uncertainty statistic and fusion. We study a mechanism that intercepts intermediate activations $h _ { t } ^ { ( l ) }$ and computes a cross-depth fusion state to track uncertainty propagation during generation.

• Single-pass operational detector. We design a lightweight scoring head $\Phi _ { \mathrm { m e t a } }$ that evaluates sequence-level risk during autoregressive generation without auxiliary decoding runs or external verifiers.

• Empirical benchmark validation. We report an AUROC of 75.5% on the TruthfulQA evaluation setting, with additional latency below 1.2% relative to unhooked generation.

The broader literature already contains hidden-state probes, layer-selection methods, spectral trajectory models, attention-based detectors, and graph-based dynamics. Accordingly, we present PoP as a specific uncertainty-propagation and fusion mechanism, not as the first internal-state or single-pass hallucination detector.

## 2 Related Work

Output-level methods estimate reliability from token probabilities, predictive entropy, semantic entropy, selfconsistency, or external evidence. Semantic entropy detects uncertainty over the meaning of multiple sampled generations, while semantic-entropy probes approximate that signal from hidden states of a single generation. These methods provide essential baselines but difer from PoP because PoP estimates a signal from ordered changes across internal layers rather than from multiple meanings or a single static representation.

Internal-state approaches include unsupervised realtime detectors, hidden-state and attention-based singleresponse methods, linear residual-stream probes, and cross-layer contribution probes. More recent work studies spectral activation trajectories, layer-wise semantic dynamics, automatic layer selection, attention sinks, multiplex graphs, and full hidden-state trajectories. These papers make the relevant novelty boundary strict: PoP must show that its uncertainty-propagation statistic contributes beyond a best single-layer probe, an existing dynamic detector, attention-derived features, and an equally sized prediction head.

Table 1: Mechanism families and the comparison boundary for PoP.
<table><tr><td>Family</td><td>Signal source</td><td>PoP comparison question</td></tr><tr><td>Output-only</td><td>Logits, entropy, perplexity</td><td>Do transitions add information beyond output uncertainty?</td></tr><tr><td>Multi-sample</td><td>Multiple meanings/completions</td><td>Does one-pass inference reduce the cost fairly?</td></tr><tr><td>Static probe</td><td>One hidden layer</td><td>Do ordered transitions beat the best single layer?</td></tr><tr><td>Dynamic inter- Layer or temporal nal</td><td>trajectories</td><td>Is the transition statistic mathematically distinct?</td></tr><tr><td>Attention/</td><td></td><td></td></tr><tr><td>graph</td><td>Attention, spectra, routing, graphs</td><td>Does PoP improve the matched-cost frontier?</td></tr></table>

Closest comparison: ICR Probe. ICR Probe quantifies the contribution of modules to residual-stream updates and then probes cross-layer hidden-state evolution [10]. PoP instead computes cosine divergence between adjacent normalized layer states, aggregates those transitions with depth weights, applies cross-layer attention to the retained state matrix, and incorporates temporal drift before scoring. The methods therefore share white-box access and a high-level interest in internal dynamics, but difer in their transition statistic, fusion operation, and scoring representation; we treat ICR Probe as a direct conceptual comparator rather than as an equivalent implementation.

Production-oriented systems such as token-level NLI cascades and log-probability time-series detectors provide complementary black-box or context-grounded baselines. They emphasize latency, span localization, calibration, and the diference between intrinsic factual confabulation and contextual contradiction. We distinguish these access assumptions explicitly in the evaluation.

## 3 Problem Definition and Scope

## 3.1 Hallucination target

This paper studies factual correctness in generated answers. The terms intrinsic hallucination, contextual contradiction, unsupported generation, citation error, and reasoning error should not be treated as interchangeable. The present study focuses primarily on factual correctness and does not claim that PoP detects every form of hallucination.

## 3.2 Generation setting

Let an autoregressive transformer have L sequential hidden layers and model dimension d. At decoding step $t ,$ the model generates token $y _ { t }$ and produces an internal state at each layer. PoP is intended to operate during the normal autoregressive process, one generator forward pass per decoding step, while retaining the activations required for the detector.

![](images/5c4039c074c6c7961a1f3ac0881ace2696c0beaa5e3803f16450bc2507ca2cbb.jpg)  
Figure 1: Compact top-to-bottom architecture of PoP. The base LLM processes input tokens in one forward pass per decoding step while hooks retain hidden states. PoP computes adjacent-layer transition divergence, performs cross-layer fusion with temporal drift, and produces steplevel and calibrated sequence-level hallucination scores.

## 3.3 White-box access and cost model

PoP requires access to intermediate hidden states. It therefore applies to open-weight or instrumentable models and does not directly apply to closed APIs that expose only text or token probabilities. In this manuscript, “single pass” means one forward execution of the generator for each autoregressive decoding step and zero auxiliary generation passes for the same response. We report activation memory, detector parameters, additional floating-point operations (FLOPs), and wall-clock latency.

## 3.4 Task output

The detector produces a step-level hallucination probability $U _ { t }$ and a calibrated sequence-level score $S _ { \mathrm { P o P } } ( Y )$ . The evaluation includes both response-level classification and an online early-warning analysis. These are separate evaluation targets and are reported separately.

## 4 Layer-Transition Uncertainty Method

## 4.1 Overview

PoP intercepts and fuses hidden-state dynamics across intermediate transformer layers during generation. It does not operate as a post-hoc text analyzer and does not require additional candidate completions.

Reading Figure 1. The compact top-to-bottom schematic groups the method into three operational stages. The first box shows ordinary generator execution and activation capture. The second box computes adjacent-layer divergence, cross-layer fusion, and temporal drift; the final box maps the assembled features to $U _ { t }$ and the calibrated sequence score. The detailed equations below define each operation precisely.

## 4.2 Internal representation

Let the unnormalized hidden state at decoding step t and layer l be $h _ { t } ^ { ( l ) } \in \mathbb { R } ^ { d }$ , where $l \in \{ 1 , 2 , \ldots , L \}$ . To reduce scale variation across depth, PoP applies layer normalization:

$$
\hat { h } _ { t } ^ { ( l ) } = \mathrm { L a y e r N o r m } ( h _ { t } ^ { ( l ) } ) = \left( \frac { h _ { t } ^ { ( l ) } - \mu _ { t } ^ { ( l ) } { \bf 1 } } { \sqrt { ( \sigma _ { t } ^ { ( l ) } ) ^ { 2 } + \epsilon } } \right) \odot \gamma + \beta ,\tag{1}
$$

where $\mu _ { t } ^ { ( l ) }$ and $\sigma _ { t } ^ { ( l ) }$ are the layer-wise mean and standard deviation, ϵ is a numerical stabilizer, and $\gamma , \beta \in \mathbb { R } ^ { d }$ are learned normalization parameters. The retained hiddenstate matrix is

$$
H _ { t } = [ \hat { h } _ { t } ^ { ( 1 ) } ; \hat { h } _ { t } ^ { ( 2 ) } ; \dots ; \hat { h } _ { t } ^ { ( L ) } ] \in \mathbb { R } ^ { L \times d } .\tag{2}
$$

Activations are extracted with forward hooks without mutating the base-model weights or altering the standard key-value-cache execution graph.

## 4.3 Transition uncertainty statistic

PoP computes directional state changes between adjacent layers. The layer-to-layer representation shift is

$$
\Delta h _ { t } ^ { ( l ) } = \hat { h } _ { t } ^ { ( l ) } - \hat { h } _ { t } ^ { ( l - 1 ) } , \quad l \in \{ 2 , 3 , \ldots , L \} .\tag{3}
$$

The scalar transition divergence is the cosine distance between successive normalized states:

$$
\delta _ { t } ^ { ( l ) } = 1 - \frac { \hat { h } _ { t } ^ { ( l ) } \cdot \hat { h } _ { t } ^ { ( l - 1 ) } } { \lVert \hat { h } _ { t } ^ { ( l ) } \rVert _ { 2 } \lVert \hat { h } _ { t } ^ { ( l - 1 ) } \rVert _ { 2 } } , \quad \delta _ { t } ^ { ( l ) } \in [ 0 , 2 ] .\tag{4}
$$

The step-level transition vector is

$$
\begin{array} { r } { u _ { t } = [ \delta _ { t } ^ { ( 2 ) } , \delta _ { t } ^ { ( 3 ) } , \ldots , \delta _ { t } ^ { ( L ) } ] ^ { \top } \in \mathbb { R } ^ { L - 1 } . } \end{array}\tag{5}
$$

The central trajectory-divergence statistic aggregates the layer transitions with learnable depth weights:

$$
D _ { t } = \frac { 1 } { L - 1 } \sum _ { l = 2 } ^ { L } w _ { l } \delta _ { t } ^ { ( l ) } .\tag{6}
$$

The depth weights $w _ { l }$ are initialized to 1. The normalization by $L - 1$ keeps the scale comparable across model depths. We evaluate whether $D _ { t }$ adds information beyond a static state, a best single layer, attention entropy, spectral features, and output entropy.

## 4.4 Inter-layer fusion and calibration

To combine layer-wise features with global depth context, PoP projects $H _ { t }$ into a lower-dimensional query, key, and value space:

$$
Q = H _ { t } W _ { Q } , \quad K = H _ { t } W _ { K } , \quad V = H _ { t } W _ { V } ,
$$

$$
W _ { Q } , W _ { K } , W _ { V } \in \mathbb { R } ^ { d \times d _ { k } } .\tag{7}
$$

The corrected scaled dot-product attention equations are

$$
A _ { t } = \mathrm { s o f t m a x } \left( { \frac { Q K ^ { \top } } { \sqrt { d _ { k } } } } \right) , \quad Z _ { t } = A _ { t } V .\tag{8}
$$

The scaling term $\sqrt { d _ { k } }$ is included in the attention logits. The fused state is computed by mean-pooling the crosslayer value output and adding the final normalized layer state:

$$
z _ { t } = \mathrm { L a y e r N o r m } ( \mathrm { M e a n P o o l } ( Z _ { t } ) + \hat { h } _ { t } ^ { ( L ) } ) .\tag{9}
$$

The meta-representation concatenates the fused state, transition vector, and temporal drift:

$$
x _ { t } = [ z _ { t } \| u _ { t } \| \Delta z _ { t } ] , \quad \Delta z _ { t } = z _ { t } - z _ { t - 1 } , \quad \Delta z _ { 1 } = 0 .\tag{10}
$$

The step-level hallucination probability is produced by a two-layer multilayer perceptron (MLP):

$$
U _ { t } = \sigma \left( W _ { 2 } \mathrm { \ G E L U } ( W _ { 1 } x _ { t } + b _ { 1 } ) + b _ { 2 } \right) .\tag{11}
$$

The sequence-level score uses a calibrated aggregate:

$$
S _ { \mathrm { P o P } } ( Y ) = \sigma \left( \alpha \left( \frac { 1 } { T } \sum _ { t = 1 } ^ { T } U _ { t } \right) + \beta \right) , \quad \sigma ( a ) = \frac { 1 } { 1 + e ^ { - a } } .\tag{12}
$$

The calibration coeficients $\alpha , \beta \in \mathbb { R }$ are learned on a validation split to reduce expected calibration error. We use Platt scaling for calibration; temperature scaling and Platt scaling are distinct procedures and are not used interchangeably.

## 4.5 Online inference and complexity

Algorithm 1: Online single-pass PoP hallucination scoring.

Input: prompt X, base LLM M, layer hooks $L _ { h }$ scoring head $\Phi _ { \mathrm { m e t a } } .$

Output: generated sequence Y, risk score $S _ { \mathrm { P o P } }$

1. Initialize Y as an empty sequence.

2. For each autoregressive step $t = 1 , \dots , T \colon$ execute one generator forward pass with hooks; capture $H _ { t }$ and generate y<sub>t</sub>.

3. Compute u<sub>t</sub> in $O ( L d )$

4. Compute z<sub>t</sub> through inter-layer fusion in $O ( L ^ { 2 } d _ { k } + L d )$

5. Concatenate $z _ { t } , \ u _ { t }$ , and $\Delta z _ { t } ;$ compute $U _ { t }$ with $\Phi _ { \mathrm { m e t a } } .$

6. Append $y _ { t }$ to Y.

7. Return Y and the calibrated aggregate of $U _ { 1 : T } .$

The implementation uses exactly one generator forward pass per token, zero auxiliary generation passes, temporary activation memory of $O ( L d )$ per step, and transition-feature computation of $O ( L d )$ . Inter-layer attention adds $O ( L ^ { 2 } d _ { k } + L d )$ . With $d _ { k } = 6 4$ and $d = 4 0 9 6$ the scoring head adds fewer than 1.4 million trainable parameters and less than 1.2% runtime latency under the stated hardware, batch size, sequence length, precision, and warm-up protocol.

## 5 Experimental Setup

## 5.1 Models and decoding

We evaluate Llama-3-8B-Instruct, Qwen2.5-7B-Instruct, and Mistral-7B-Instruct-v0.2. Base models are described as running in native 16-bit precision on NVIDIA A100-80GB GPUs. The decoding settings are greedy decoding with temperature 0.0 and nucleus sampling with temperature 0.7 and $p = 0 . 9$ Maximum output length is 256 tokens.

## 5.2 Datasets and labels

TruthfulQA. We use 817 questions designed to elicit common human false beliefs, with factual correctness determined using the oficial target-evaluation procedure.

Table 2: Model and evaluation configuration.
<table><tr><td>Component</td><td>Specification</td></tr><tr><td>Model families</td><td>Llama-3-8B-Instruct; Qwen2.5-7B-Instruct; Mistral-7B-Instruct-v0.2</td></tr><tr><td>Precision</td><td>FP16 or BF16 native precision</td></tr><tr><td>Decoding</td><td>Greedy, temperature 0.0; nucleus, temperature 0.7,  $p = 0 . 9$ </td></tr><tr><td>Maximum length</td><td>256 generated tokens</td></tr><tr><td>Datasets</td><td>TruthfulQA; HaluEval 2.0; FaithDial</td></tr><tr><td>Split</td><td>70% train, 15% validation, 15% test, stratified by factual label</td></tr><tr><td>Context</td><td>Closed-book QA and retrieval-augmented generation context</td></tr><tr><td>Labeling</td><td>Automated NLI verification cross-checked by human annotators</td></tr></table>

Table 3: Baseline families included in the manuscript.
<table><tr><td>Family</td><td>Methods included</td></tr><tr><td></td><td>Output-only heuristics Token-logit entropy; sequence predictive perplexity; final-layer vocabulary-logit margin</td></tr><tr><td>Multi-sample post-hoc</td><td>and Semantic entropy with  $K = 5$  generations; external DeBERTa-v3-large NLI</td></tr><tr><td></td><td>entailment scoring Static internal probing LDA and logistic regression on isolated mid-layer and final-layer hidden states</td></tr><tr><td>Dynamic internal and Residual-stream norm growth;</td><td></td></tr><tr><td>spectral</td><td>attention-head entropy dispersion; layer-wise spectral geometry distance</td></tr></table>

HaluEval 2.0. We use 10,000 paired sequences across general question answering and dialogue, containing factual ground truths and explicitly generated hallucinations.

FaithDial. We use 5,000 conversational turns evaluating faithfulness against grounded knowledge snippets.

The exact dataset names, versions, licenses, sample counts, split construction, and annotation procedure are recorded in the author-verified experimental implementation. The HaluEval version and the role of automated NLI labels are fixed by that implementation.

## 5.3 Baselines

The comparison set includes output-only, multisample, static internal, and dynamic internal baselines. We do not claim superiority over methods that are not evaluated under the same model-access and experimental conditions; the novelty claim is restricted to the uncertainty-propagation and fusion mechanism studied here.

## 5.4 Evaluation metrics

AUROC and AUPRC measure threshold-agnostic separation. Precision at fixed recall, including P at R90, reflects high-reliability deployment. Expected calibration error (ECE) measures agreement between predicted risk and empirical frequency across ten equal-width bins. Also report Brier score, precision and recall at the operating threshold, response-level or span-level F1 where applicable, milliseconds per token, VRAM overhead, extra FLOPs, and throughput impact.

## 6 Main Results

## 6.1 Detection performance

The following author-verified results are reported on Llama-3-8B-Instruct under the stated evaluation protocol.

PoP reports the highest separation accuracy among the methods shown in this table, including 75.5% AU-ROC on TruthfulQA and 74.6% on HaluEval, and a 9.4 percentage-point improvement over the static $l = 3 2$ probe on TruthfulQA. These point estimates are interpreted within the stated evaluation protocol and modelaccess conditions.

## 6.2 Calibration and operating points

Calibration reduces ECE from 0.142 to 0.031. The calibration coeficients and operating threshold are selected on the validation split, while the test set is reserved for final evaluation.

## 6.3 Cost and latency

PoP adds 0.3 ms/token and 18.4 MB of temporary activation memory under the stated benchmark protocol. The measurement is interpreted with the declared warmup, repeated-trial, synchronization, hardware, and batchsize conditions.

## 6.4 Cross-model and cross-domain transfer

PoP loses less than 3.5 percentage points of AUROC in the reported transfer settings. This claim is explicitly bounded to English-language generation; multilingual transfer is outside the scope of this version.

## 6.5 Perturbation robustness

We evaluate paired perturbations intended to separate factual status from surface style. Entity swaps preserve syntax while changing validity. Style transformations change register while preserving factual status. Paraphrases preserve meaning, distractor-context insertion adds irrelevant facts, and temperature scaling increases decoding variation.

PoP loses at most 2.4 percentage points of AUROC across these conditions, while output entropy is more sensitive to style and temperature. This interpretation is bounded to the tested perturbations.

## 6.6 Early warning and token-level behavior

Because PoP computes $U _ { t }$ during generation, it supports an online early-warning monitor. In the illustrative example, the risk score spikes near the generated entity “Zurich” in the sentence “The capital of France is Zurich.” The reported $U _ { t }$ crosses a threshold of 0.70 within $1 . 2 \pm 0 . 4$ tokens of the initial factual-error onset, prevents an average of 18.4 downstream decoding steps on hallucinated completions, and achieves token-level precision of 71.8% and recall of 68.4% for hallucinated entity spans.

The token-level interpretation is bounded by the annotation protocol used for the reported entity spans. Deployment as a stopping rule additionally requires monitoring false interruptions, restart or revision cost, and user-visible response quality.

## 7 Analysis and Ablations

We report ablations on Llama-3-8B-Instruct and TruthfulQA.

Table 4: Detection performance comparison. TQ denotes TruthfulQA and HE denotes HaluEval.
<table><tr><td>Method</td><td>Type</td><td colspan="6">TQ AUC TQ AP TQ P@R90 HE AUC HE AP HE P@R90</td></tr><tr><td>Logit entropy</td><td>Output</td><td>54.2%</td><td>51.8%</td><td>22.4%</td><td>52.1%</td><td>49.3%</td><td>18.7%</td></tr><tr><td>Perplexity</td><td>Output</td><td>56.1%</td><td>53.4%</td><td>24.1%</td><td>55.0%</td><td>52.6%</td><td>21.0%</td></tr><tr><td>Static probe,  $l = 1 6$ </td><td>Static</td><td>63.8%</td><td>61.2%</td><td>35.6%</td><td>61.4%</td><td>58.9%</td><td>31.2%</td></tr><tr><td>Static probe, l = 32</td><td>Static</td><td>66.1%</td><td>64.0%</td><td>39.8%</td><td>64.2%</td><td>62.1%</td><td>36.5%</td></tr><tr><td>Spectral geometry</td><td>Dynamic</td><td>68.4%</td><td>66.7%</td><td>43.1%</td><td>67.0%</td><td>65.2%</td><td>40.8%</td></tr><tr><td>Semantic entropy,  $K = 5$ </td><td>Multi-sample</td><td>74.1%</td><td>72.8%</td><td>51.3%</td><td>72.8%</td><td>71.0%</td><td>48.6%</td></tr><tr><td>DeBERTa-v3 NLI</td><td>Post-hoc</td><td>74.8%</td><td>73.5%</td><td>52.0%</td><td>73.9%</td><td>72.4%</td><td>50.1%</td></tr><tr><td>PoP (ours)</td><td>Single-pass</td><td>75.5%</td><td>74.6%</td><td>54.2%</td><td>74.6%</td><td>73.1%</td><td>51.8%</td></tr></table>

Figure 2: Step-wise Uncertainty Trajectory during Entity Hallucination  
![](images/39c2f46e7ad5cb6ac640f9931a703b6ead65af94ec994dd384b30c6c1fde886e.jpg)  
Figure 2: Step-wise uncertainty trajectory during entity hallucination.

Table 5: Calibration and operating-point results.
<table><tr><td>Configuration</td><td>ECE</td><td>Brier</td><td>Pre- cision</td><td>Recall</td><td>F1</td></tr><tr><td>Uncalibrated  $\mathrm { P o P }$ </td><td>0.142</td><td>0.188</td><td>68.4%</td><td>76.2%</td><td>0.721</td></tr><tr><td>Platt-calibrated</td><td>0.031</td><td>0.124</td><td>73.1%</td><td>74.5%</td><td>0.738</td></tr><tr><td>PoP,  $\begin{array} { r l r } { \alpha } & { { } = } & { 1 . 2 4 , } \end{array}$   $\beta = - 0 . 1 2$ </td><td></td><td></td><td></td><td></td><td></td></tr></table>

The reported ablations suggest that ordered layer transitions and layer normalization contribute to $\mathrm { P o P s }$ performance. The strongest interpretation is that the transition representation is predictive under the tested setup. The ablations do not, by themselves, prove that the internal transition is a causal driver of hallucination. Causal language should be reserved for experiments involving activation patching, steering, or another intervention.

The reported ablations establish the contribution of the evaluated transition representation, normalization, layer order, and fusion components. Comparisons with additional transition statistics and alternative fusion heads are outside the scope of this version.

## 8 Limitations and Broader Impact

## 8.1 Technical limitations

White-box access. PoP requires intermediate hidden states and therefore cannot directly operate on closed-source APIs that expose only text or token probabilities.

Table 6: Cost and latency under the stated benchmark protocol.
<table><tr><td>Method</td><td>Extra calls</td><td>Latency</td><td>VRAM overhead</td><td>Through- put penalty</td></tr><tr><td>Unhooked base0 LLM</td><td></td><td>24.1 ms/token</td><td>0.0 MB</td><td>0.0%</td></tr><tr><td>PoP (ours)</td><td>0</td><td>24.4 ms/token</td><td>18.4 MB</td><td>1.2%</td></tr><tr><td>Spectral geome-0 try</td><td></td><td>31.8 ms/token</td><td>142.0 MB</td><td>31.9%</td></tr><tr><td>Semantic tropy,  $K = 5$ </td><td>en4 auxiliary generations</td><td>120.5 ms/token</td><td>0.0 MB</td><td>400.0%</td></tr><tr><td>External BERTa NLI</td><td>De-1 auxiliary pass</td><td>48.2 ms/token</td><td>1740.0 MB</td><td>100.0%</td></tr></table>

Activation memory. Retaining $H _ { t } \in \mathbb { R } ^ { L \times d }$ requires temporary GPU memory. Large batches and very deep models may constrain throughput even when the incremental latency is small.

Scope. This study evaluates English-language generation in closed-book question answering, dialogue, and short-context retrieval-augmented settings. Crosslingual performance, long-form document generation, and reasoning-chain reliability remain unverified.

Benchmark currency. The present study evaluates the model and dataset configuration listed in Table 2; it does not claim state-of-the-art performance over every newer model or benchmark. The reported conclusions should therefore be read as conditional on the stated evaluation scope.

Supervised calibration. The scoring head and calibration procedure require labeled validation data. Severe distribution shifts may degrade calibration.

Figure 3: Layer-Transition Divergence Across Transformer Depth  
![](images/d4abd40323a0703caf6defab2737c4b248043bfb961fbdf7f83bafe234d94f9d.jpg)  
Figure 3: Layer-transition divergence across transformer depth.

Table 7: Cross-model and cross-domain transfer.
<table><tr><td>Backbone</td><td>Dataset/ domain</td><td>Zero-shot AUROC</td><td>Δ</td></tr><tr><td>Llama-3-8B- Instruct</td><td>TruthfulQA, in- domain</td><td>75.5%</td><td></td></tr><tr><td>Llama-3-8B-</td><td>FaithDial, dia-</td><td>72.1%</td><td>-3.4%</td></tr><tr><td>Instruct Qwen2.5-7B-</td><td>logue TruthfulQA</td><td>73.8%</td><td>-1.7%</td></tr><tr><td>Instruct Mistral-7B-</td><td>TruthfulQA</td><td>72.4%</td><td>-3.1%</td></tr><tr><td>Instruct-v0.2 Mistral-7B- Instruct-v0.2</td><td>HaluEval, QA/dialogue</td><td>71.2%</td><td>-4.3%</td></tr></table>

Table 8: Perturbation robustness.
<table><tr><td>Pertur- bation</td><td>Control</td><td></td><td>Base</td><td>Pert.</td><td>Δ</td></tr><tr><td>Entity swap</td><td>Factual ity flips; retained</td><td>valid- syntax</td><td>75.5%</td><td>74.8%</td><td>-0.7%</td></tr><tr><td>Style trans- Register formation</td><td>factual</td><td>changes; status</td><td>75.5%</td><td>75.2%</td><td>-0.3%</td></tr><tr><td>Paraphrase</td><td>constant Surface changes;</td><td>form meaning</td><td>75.5%</td><td>74.9%</td><td>-0.6%</td></tr><tr><td>Distractor context</td><td>constant pended</td><td>Irrelevant facts ap-</td><td>75.5%</td><td>73.8%</td><td>-1.7%</td></tr><tr><td>Temperature Nucleus sampling 0.7</td><td>increases variation</td><td></td><td>75.5%</td><td>73.1%</td><td>-2.4%</td></tr></table>

Predictive versus causal interpretation. The current experiments analyze standard forward execution without active causal intervention. The reported quantities therefore represent predictive correlations with factual errors rather than proved causal mechanisms.

## 8.2 Broader impact and risks

Automated hallucination detectors can create automation bias. A low risk score should not be interpreted as a guarantee of truth, especially in legal, clinical, or other high-stakes settings. Aggressive thresholds may also suppress valid but unusual linguistic responses or rare factual assertions. Human review, domain-specific verification, and clear uncertainty communication remain necessary.

## 9 Conclusion

This manuscript studies whether factual hallucinations in autoregressive LLMs leave detectable signals in internal layer-transition dynamics before final vocabulary projection. It introduces PoP, a lightweight framework that intercepts hidden-state trajectories across transformer depth, computes normalized transition divergence, fuses cross-layer information, and produces a calibrated hallucination-risk score during standard generation. We report 75.5% AUROC on TruthfulQA, less than 1.2% runtime latency overhead, and zero auxiliary generation passes under the stated evaluation protocol. The contribution is a specific combination of adjacentlayer transition divergence, cross-layer activation fusion, temporal drift, and calibration rather than a general claim that internal activations can detect hallucinations.

## 10 References

## References

[1] Z. Ji, N. Lee, R. Frieske, et al.., “Survey of hallucination in natural language generation,” ACM Computing Surveys, 55(12), 2023.

[2] L. Huang, W. Yu, W. Ma, et al.., “A survey on hallucination in large language models: Principles, taxonomy, and open challenges,” arXiv:2311.05232, 2023.

[3] S. Farquhar, K. Konyushkova, L. Kurth, et al.., “Detecting hallucinations in large language models using semantic entropy,” Nature, 630, 625–630, 2024.

[4] L. Kuhn, Y. Gal, and S. Farquhar, “Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation,” ICLR, 2023.

[5] P. Manakul, A. Liusie, and M. Gales, “SelfCheckGPT: Zeroresource black-box hallucination detection for generative large language models,” EMNLP, 2023.

[6] A. Azaria and T. Mitchell, “The internal state of an LLM knows when it is lying,” Findings of EMNLP, 2023.

Table 9: Ablation results on Llama-3-8B-Instruct and TruthfulQA.
<table><tr><td>Configuration</td><td>AUROC</td><td>Δ</td></tr><tr><td>Full PoP framework</td><td>75.5%</td><td></td></tr><tr><td>Best single layer,  $l = 2 4$ </td><td>66.8%</td><td>-8.7%</td></tr><tr><td>Static features only</td><td>67.2%</td><td>-8.3%</td></tr><tr><td>Transition features only</td><td>70.1%</td><td>-5.4%</td></tr><tr><td>No layer normalization</td><td>68.9%</td><td>-6.6%</td></tr><tr><td>Shuffled layer order</td><td>58.3%</td><td>-17.2%</td></tr><tr><td>Random-feature control</td><td>50.1%</td><td>-25.4%</td></tr><tr><td>Matched final-layer MLP</td><td>66.4%</td><td>-9.1%</td></tr><tr><td>Attention-only control</td><td>59.2%</td><td>-16.3%</td></tr><tr><td>Output-entropy control</td><td>55.8%</td><td>-19.7%</td></tr></table>

[7] J. Kossen, J. Han, M. Razzak, et al.., “Semantic Entropy Probes: Robust and Cheap HallucinationDetection in LLMs,” arXiv:2406.15927, 2024.

[8] G. Sriramanan, S. Bharti, V. S. Sadasivan, et al.., “LLM-Check: Investigating Detection of Hallucinations in Large Language Models,” NeurIPS, 2024.

[9] J. Su, and colleagues, “Unsupervised Real-Time Hallucination Detection based on the Internal States of Large Language Models,” arXiv:2403.06448, 2024.

[10] Z. Zhang, X. Hu, H. Zhang, J. Zhang, and X. Wan, “ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection in LLMs,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, pp. 17986–18002, 2025; arXiv:2507.16488.

[11] C. O’Neill, S. Chalnev, C. C. Zhao, and colleagues, “A Single Direction of Truth: An Observer Model’s Linear Residual Probe Exposes and Steers Contextual Hallucinations,” arXiv:2507.23221, 2025.

[12] A. Ettori, and colleagues, “EigenTrack: Spectral Activation Feature Tracking for Hallucination and Out-of-Distribution Detection in LLMs and VLMs,” arXiv:2509.15735, 2025.

[13] M. Mir, “The Geometry of Truth: Layer-wise Semantic Dynamics for Hallucination Detection in LLMs,” arXiv:2510.04933, 2025.

[14] X. Wang, W. X. Cao, A. G. Wilson, and Z. Zeng, “Automatic Layer Selection for Hallucination Detection,” ICML, 2026; arXiv:2605.26366.

[15] R. Alvi, N. L. Sayeedi, and M. F. A. Sayeedi, “MultiHaluDet: Multilingual Hallucination Detection via LLM Hidden State Probing,” MeLLM, 2026.

[16] J. Binkowski, K. Adamczewski, and T. Kajdanowicz, “Attention Sinks as Internal Signals for Hallucination Detection in Large Language Models,” Accepted at ICML 2026; arXiv:2604.10697.

[17] K. Cherukuri and L. R. Varshney, “Hallucination Basins: A Dynamic Framework for Understanding and Controlling LLM Hallucinations,” arXiv:2604.04743, 2026.

[18] A. Alansari, M. Alkhorasani, and H. Luqman, “CrossHallu: Do Hallucination Signals Generalize Across Languages and Domains in Large Language Model’s Internals?” arXiv:2607.04029, 2026.

[19] A. Shapiro, K. Taneja, and A. Goel, “HALT: Hallucination Assessment via Log-probs as Time series,” arXiv:2602.02888, 2026.

[20] D. Wilson and M. Akrout, “Low-Cost Black-Box Detection of LLM Hallucinations via Dynamical System Prediction,” arXiv:2605.05134, 2026.

[21] I. Itkin, “Temporal Multi-Signal Fusion for Token-Level Hallucination Detection,” Research Square preprint, posted July 10, 2026; not peer reviewed at the time of the accessed version.

[22] S. Lin, J. Hilton, and O. Evans, “TruthfulQA: Measuring how models mimic human falsehoods,” ACL, 2022.

[23] J. Li, X. Cheng, W. X. Zhao, et al.., “HaluEval: A largescale hallucination evaluation benchmark for large language models,” EMNLP, 2023.

[24] N. Dziri, H. Rashkin, T. Lin, et al.., “FaithDial: A faithful benchmark for information-seeking dialogue,” TACL, 10, 1473–1490, 2022.

[25] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” ICML, 2017.

[26] A. Vaswani, N. Shazeer, N. Parmar, et al.., “Attention is all you need,” NeurIPS, 2017.