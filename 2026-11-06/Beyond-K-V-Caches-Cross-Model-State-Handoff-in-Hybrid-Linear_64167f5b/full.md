# Beyond K/V Caches: Cross-Model State Handoff in Hybrid Linear-Attention LLMs

Jiacheng Cai Tao Zhang Siqi Liang Yun Dai Ziqian Zeng South China University of Technology, China

## Abstract

Production deployments often use different models to balance cost and quality through cascading, routing, and switching models during long conversations. Context accumulates across turns. Each switch requires the target to repeat prefill over this context. Existing methods transfer full-attention (FA) K/V caches across models. Hybrid linear-attention models also retain recurrent states (REC) and convolu tion histories (Conv). The interaction between Conv and REC complicates cross-model state transfer. Conv perturbations can change keys and values in recurrent layers and alter REC updates. Conv histories are overwritten, but changes in REC can still affect later outputs. More accurate REC reconstruction need not improve continuation. Building on this, we propose Hybrid State Transfer (HST) for all three state types. It uses structured REC maps and preserves history order during Conv alignment. We calibrate the adapters using the frozen target’s outputs during continuation. FA cache map updates are merged offline. Across five benchmarks, Qwen3.5 4B→27B and 9B→27B achieve mean task retentions of 89.88% and 94.11%, respectively. With source caches already available, separate timing on synthetic 16K inputs yields handoff body speedups of 21.66× and 21.91×.

## 1 Introduction

Production deployments often use models of different sizes within a family to balance cost and quality through cascading, routing, and switching models during long conversations (Heo et al., 2026; Ong et al., 2025). The source model may already have processed the context. In this case, target prefill repeats computation over the same tokens to build the target’s inference state. The cost of this repeated computation increases with context length. Cross-model state transfer aims to reuse the source model’s computation by converting the source state into a form usable by the target. Hybrid linearattention models also retain recurrent states (REC) and convolution histories (Conv). The interaction between Conv and REC complicates cross-model state transfer.

Existing work on cross-model state transfer focuses on the K/V caches of full-attention (FA) models. Affine mappings fitted in closed form convert these caches across models (Heo et al., 2026; Qu et al., 2026). Nonlinear correction modules are added to an affine K/V mapping and trained through a frozen target model using continuation cross-entropy and forward KL losses (Qiu et al., 2026). The hybrid linear-attention models studied here also use REC to summarize history and Conv to store projected features within a finite window (Yang et al., 2025). Source and target models may differ in layer layout, recurrent head count, and convolution channel count. Conv also affects key and value computation in recurrent layers, and these signals drive REC updates. Errors in Conv conversion can therefore propagate into REC and affect subsequent target computation.

Our controlled interventions show that the effects of Conv perturbations can persist in REC and alter later outputs after the affected history slots have been overwritten. REC reconstruction controls further show that lower local error need not improve full-model continuation. These observations motivate continuation compatibility: the target must be able to use the transferred context and update its state effectively. We propose Hybrid State Transfer (HST), a framework for transferring FA K/V caches, REC, and Conv between hybrid linearattention models. HST combines structural adaptation with functional calibration through the target model and keeps both models’ pretrained weights fixed. It uses pair-specific REC mappings with optional state-conditioned corrections and affine Conv alignment that preserves history order. For

FA, learned weight updates are merged offline into the base affine maps. All states and position metadata are installed at a consistent boundary.

We evaluate the proposed HST framework using models from the Qwen3.5 family (Qwen Team, 2026) on five benchmarks covering commonsense reasoning, knowledge-based question answering, and mathematical reasoning. For 4B→27B and 9B→27B, mean task retention reaches 89.88% and 94.11%, exceeding the strongest compared hybrid baseline by 3.56 and 7.10 percentage points, respectively. Supplementary 4B↔9B experiments examine handoff with matching state layouts. Separate timing measurements on synthetic 16K inputs with source caches already available show handoff body speedups of 21.66× and 21.91× for the two main directions, respectively. These measurements exclude source prefill, the target model’s output projection from hidden states to vocabulary scores, and answer generation.

Our main contributions are:

• Conv and REC during continuation. In controlled interventions, the effects of Conv perturbations can persist in REC and alter later outputs beyond the convolution window. Separate reconstruction diagnostics show lower REC reconstruction error alongside higher continuation KL.

• Hybrid State Transfer (HST). HST transfers FA K/V caches, REC, and Conv with position metadata at a common prefix boundary. It uses pair-specific REC maps with optional state-conditioned corrections and preserves history order during affine Conv alignment. REC mappings and FA cache maps are calibrated offline through a frozen target model, and learned updates to the FA cache maps are merged into the base affine maps before deployment.

• Task performance and handoff latency. Across five benchmarks, HST achieves higher mean task retention than the evaluated hybrid baselines for both 4B→27B and 9B→27B. Separate timings on synthetic inputs with source caches already available show lower body latency for HST than for native target prefill.

## 2 Related Work

Affine K/V mappings. Cross-Model selects supporting source layers and fits separate affine ridge maps for K and V. It removes source RoPE before K conversion and applies target RoPE afterward (Heo et al., 2026). CacheBridge constrains source-head support and weights fitting by attention sensitivity while retaining affine deployment (Qu et al., 2026). These approaches provide mappings for explicit full-attention caches. Additional persistent states require their own representations and correspondences.

Functional cache calibration. CacheLink adds nonlinear K/V corrections to a Ridge base and trains them through a frozen receiver using continuation cross-entropy and forward KL (Qiu et al., 2026). Receiver-side supervision and the distinction between reconstruction and downstream utility therefore have precedents. We examine continuation when recurrent memory and convolution histories also participate, and study full-attention cache calibration that can be folded into affine maps. The Ridge, CacheBridge, and CacheLink comparisons in Section 5 use hybrid extensions implemented for this study.

## 3 Preliminaries

## 3.1 Hybrid Linear-Attention Models

Linear attention admits recurrent computation with a fixed-size state (Katharopoulos et al., 2020). Delta-rule formulations maintain a matrix memory and update it at each token (Yang et al., 2024). The Qwen3.5 models studied here interleave fullattention (FA) layers with Gated DeltaNet (GDN) layers (Qwen Team, 2026; Yang et al., 2025).

FA layers retain K/V caches indexed by token position, and the cache size grows with prefix length. Each GDN layer retains a recurrent-state matrix (REC) and a convolution history (Conv). REC summarizes previous updates, and Conv stores a finite window of projected input features in temporal order. For a fixed model, REC and Conv sizes do not grow with prefix length. Model pairs may differ in layer layout, recurrent head count, and convolution channel count.

At each recurrent step, a causal convolution combines the current projected features with Conv history to produce query, key, and value signals. Separate input projections supply decay, write, and output gates. The previous REC matrix and the current keys and values determine the REC update under the decay and write gates. The query reads the updated matrix. Normalization, output gating, and an output projection then produce the layer’s hidden representation (Yang et al., 2025; Qwen Team, 2026). Conv therefore affects both the current readout and the information written into REC. Effects written into REC can outlast Conv’s finite window.

## 3.2 Cross-Model State Transfer

Let C denote a prefix boundary after processing tokens $x _ { 1 : C }$ . For model m, we represent the complete inference state at this boundary as

$$
\mathcal { M } _ { m } ( C ) = \left( \{ Z _ { m , j } ( C ) \} _ { j \in \mathbb { Z } _ { m } } , \pi _ { m } ( C ) \right) .\tag{1}
$$

Here, $\mathcal { I } _ { m }$ indexes the persistent state components, and $Z _ { m , j } ( C )$ denotes one component. The metadata $\pi _ { m } ( C )$ contains position indices, cache lengths, and attention masks. All components and metadata describe the same processed prefix $x _ { 1 : C }$ After subsequent REC updates, truncating the FA cache cannot recover an earlier REC matrix.

We consider source and target models s and t from the same family, using the same input sequence and aligned prefix boundaries. Both models’ pretrained weights remain fixed. A transfer function $F _ { \theta }$ converts the source state into a target state:

$$
\widehat { \mathcal { M } } _ { t } ( C ) = F _ { \theta } ( \mathcal { M } _ { s } ( C ) ) .\tag{2}
$$

Here, θ denotes the transfer parameters. The target processes the remaining task input from the installed state, followed by answer scoring or generation.

Offline fitting and diagnostics use native target states from prefill of the same prefix. Online conversion uses only the source boundary state and saved transfer parameters, avoiding target prefix recomputation.

## 3.3 Continuation Compatibility

We use continuation compatibility to describe the target model’s ability to use transferred context and update its state effectively. Task performance is the final criterion; preliminary diagnostics use forward KL from native outputs to diagnostic branches. Figure 1 and Table 1 use different model configurations and inputs.

Conv perturbations. In native Qwen3.5-9B, we zero only Conv in eight packed token blocks, each with two prefix boundaries. All branches consume the same 32 continuation tokens. After measuring step 4, separate branches restore native REC, Conv, or both; FA caches are not restored. At step 8, restoring REC yields output KL of 0.00721, compared with 0.07023 for restoring Conv. The perturbed Conv slots have been overwritten, but effects can persist in REC and alter later outputs. Appendix E examines later fluctuations and an additional FA-restoration control.

![](images/56d5201d94e3696cbfd7be5695e7490af346697cf5023811ae36ca7c8a9af843.jpg)  
Figure 1: State restoration after zeroing Conv in native Qwen3.5-9B. Curves show all 32 steps, with two prefix boundaries averaged within each of eight packed blocks before averaging across blocks. Restore both denotes REC+Conv; FA caches are not restored. The dashed line marks restoration after measuring step 4. Both axes are logarithmic; KL is from native outputs to each branch.

REC reconstruction. A separate diagnostic uses an auxiliary 4B→27B configuration and 32 development documents. These documents participated in candidate selection, so this comparison is exploratory. Base denotes the auxiliary transfer configuration before the diagnostic reconstruction adjustments. The two residual candidates modify REC while retaining Base’s FA caches and Conv histories. Both candidates are fitted to reconstruct native target REC states. For each evaluated REC layer, they use the same residual basis and predict its coefficients by ridge regression. REC residual uses summaries of source REC states; Context residual additionally uses summaries of source FA caches and Conv histories. These reconstruction candidates are separate from HST’s functionally trained conditional REC correction. Compared with Base, REC residual reduces the initial relative MSE from 1.00671 to 0.28380, but increases fullmodel continuation KL from 0.64829 to 0.69142. Lower local REC reconstruction error therefore need not produce lower continuation KL. Table 1 reports the comparison.

<table><tr><td>Configuration</td><td>Initial REC relative MSE ↓</td><td>Continuation KL↓</td></tr><tr><td>Base</td><td>1.00671</td><td>0.64829</td></tr><tr><td>REC residual</td><td>0.28380</td><td>0.69142</td></tr><tr><td>Context residual</td><td>0.24696</td><td>0.67841</td></tr></table>

Table 1: Initial REC reconstruction error and full-model continuation KL for the auxiliary 4B→27B configuration on 32 development documents used for candidate selection. Results are averaged over nine conditions per document: three prefix lengths (128, 512, and 2048 tokens) crossed with three continuation branches (natural, revisit, and topic shift). All configurations consume the same fixed continuation tokens. Lower reconstruction error need not yield lower continuation KL.

A separate state-replacement study uses 18 documents in a fixed auxiliary 4B→27B configuration (Appendix B.2). We replace selected components with native target states for the same prefix and retain the other components in their transferred form at the handoff boundary. Composite KL equally weights the output KL at the end of the prompt and its mean over the next eight positions. Replacing REC or FA reduces composite KL from 0.59291 to 0.10060 or 0.33515, respectively. Replacing both yields 0.01664, whereas replacing only Conv yields 0.61845. These results measure replacement effects within the transferred configuration and do not isolate independent component contributions. Together, the diagnostics motivate explicit adaptation of REC and Conv and calibration through the target model’s subsequent outputs. The next section presents HST.

## 4 Method

HST combines complete-state mapping with functional calibration through a frozen target model. The mappings adapt FA K/V caches, REC matrices, and Conv histories to the target’s layer, head, and channel layouts. Both models’ pretrained weights remain fixed throughout.

## 4.1 Handoff Boundary and State Consistency

Let $x _ { 1 : N }$ be the task prompt, and let r denote the number of real prompt tokens reserved for target processing after installation. The handoff boundary is $C = N - r$ . The source processes $x _ { 1 : C }$ and exports its state at $C .$ The target consumes the remaining prompt $x _ { C + 1 : N }$ from the converted state. This suffix is part of the original task input.

Figure 2 contrasts native prefill of $x _ { 1 : N }$ with handoff from an available source state at $C .$ Handoff skips target prefix prefill and resumes at $x _ { C + 1 : N }$ . Both paths reach N before scoring or generation. Their final hidden states and next-token distributions can differ.

All components and position metadata must describe the same prefix $x _ { 1 : C }$ (Section 3.2). We construct target FA, REC, and Conv states with consistent positions and cache lengths before the first target suffix token. Native target states provide offline fitting and supervision; online handoff uses the source state and trained adapters.

## 4.2 Complete-State Mapping

Source support layers are selected offline, and mapping parameters are fitted separately for each model pair. We omit the batch dimension and write feature vectors as row vectors. Figure 3 illustrates the 4B→27B configuration, including its stateconditioned REC correction. The 9B→27B configuration uses its own calibrated mappings and does not apply this correction.

FA caches. For each target FA layer $j ,$ , K and V share four selected source support layers but use separate feature vectors and affine maps. At historical position i, we first undo source RoPE on each source K head, then concatenate the resulting K features in the saved support-layer order to form $z _ { j , i } ^ { K }$ We concatenate V features independently in the same order to form $z _ { j , i } ^ { V }$ . Within each layer, features follow the KV-head order and then the coordinate order within each head. The maps produce

$$
\begin{array} { r l } & { \widehat { K } _ { j , i } = \mathcal { R } _ { t , i } \big ( \mathcal { P } _ { t } ( z _ { j , i } ^ { K } W _ { j } ^ { K } + b _ { j } ^ { K } ) \big ) , } \\ & { \widehat { V } _ { j , i } = \mathcal { P } _ { t } ( z _ { j , i } ^ { V } W _ { j } ^ { V } + b _ { j } ^ { V } ) , } \end{array}\tag{3}
$$

where $\mathcal { P } _ { t }$ reshapes outputs into target KV heads and $\mathcal { R } _ { t , i }$ applies target RoPE (Heo et al., 2026). In the studied models, RoPE rotates the first 64 coordinates of each 256-dimensional K head. The remaining coordinates are unchanged.

Centered affine ridge regression initializes the K and V maps without regularizing their biases. The weights are shared across historical positions. The saved support-layer order is retained because the fitted coefficients correspond to that feature order. Mapping K after removing source RoPE separates the representation mapping from the explicit positional transformation.

![](images/5a04133b28a72f9cc184a059036347299e01165746ee56256b00b11448d12eec.jpg)  
Figure 2: Handoff protocol and body latency definition. Given an available source state at $C = N - r$ , handoff skips target prefill of $x _ { 1 : C }$ . The timed handoff includes state conversion, installation, the REC correction when used, and target suffix processing. The native reference processes the complete prompt. Both timed paths end when the target cache and final hidden state are ready. Source prefill, the vocabulary output projection, and answer generation are excluded. Block widths are schematic.

REC matrices. Both model pairs use 16 groups defined by the native GDN Q/K heads. Each source group contains two value heads, and each target group contains three. For each target REC layer j, we select four source REC layers. At matrix coordinate (a, b), the two heads from each selected source layer contribute eight values, concatenated into ${ \pmb x } _ { j , g , a , b }$ for group g. The grouped affine map is

$$
\begin{array} { r } { \pmb { s } _ { j , g , a , b } ^ { 0 } = \pmb { x } _ { j , g , a , b } G _ { j , g } + \beta _ { j , g } , } \end{array}\tag{4}
$$

where $G _ { j , g } \in \mathbb { R } ^ { 8 \times 3 }$ and $\beta _ { j , g } \in \mathbb { R } ^ { 3 }$ . The output contains the values for the three target heads at the same matrix coordinate. Here, $\beta _ { j , g }$ is the mapping bias, separate from the target model’s recurrent gates.

The same weights and biases are applied at every coordinate of the 128 × 128 state matrices. This mapping combines selected source layers and heads without mixing matrix coordinates or native head groups. The two model pairs share this construction but use different support layers and parameters. Affine fitting initializes (G, β), and offline functional calibration subsequently updates both. We assemble $S ^ { 0 }$ using these calibrated parameters; the superscript denotes the state before the conditional correction when used.

Conv histories. For each target recurrent layer, we use the first source layer in its saved REC support list. Conv stores the projected Q/K/V features before the recurrent layer’s causal convolution. We divide these channels into Q, K, and V blocks and fit separate affine channel maps within their native groups. The same weights and biases are applied at all four history slots, without mixing slots. In the 4B→27B configuration shown in Figure 3, this maps 8,192 source channels to 10,240 target channels.

The slots retain their original order from oldest to newest. Applying a shared channel map at each slot changes its feature representation while preserving its position in the convolution history. This alignment accounts for Conv’s role in producing the keys and values used by subsequent REC updates.

## 4.3 Functional Calibration and FA Folding

Low reconstruction error does not ensure that the target will use a transferred state effectively. We therefore calibrate the adapters through target continuation in separate offline stages: updating REC mappings, training the 4B→27B REC correction, and learning an FA weight increment. The target model’s pretrained parameters, including those used for recurrent state updates and the vocabulary output projection, remain fixed.

REC mapping calibration. Starting from the fitted affine maps, functional calibration updates both $G _ { j , g }$ and $\beta _ { j , g }$ through the frozen target. Initial calibration uses KL between native and transferred output distributions, together with normalized REC reconstruction error. Later task-oriented calibration also uses natural-text continuations and candidatelevel outputs, including candidate-distribution KL and a mean-squared error term on centered candidate scores. Support layers and the grouped mapping structure remain fixed. The two model pairs are calibrated separately.

![](images/9c10e987f8ce3ad13c089ab7e9d0cb1c5fbd466738d0bf2a6e09dce29b1f6bbb.jpg)  
Figure 3: Complete-state handoff for Qwen3.5-4B→27B with a state-conditioned REC correction. Offline training calibrates the REC mappings and trains the correction before the final FA-only stage. FA increments are merged into the cache-mapping weights before handoff. $\operatorname { A t } C = N - r ,$ , the adapter converts FA K/V caches, REC matrices, and Conv histories. The REC correction produces $S ^ { + }$ from $S ^ { 0 }$ once. All target states and position metadata are ready before the target processes $x _ { C + 1 : N }$

State-conditioned REC correction. The 4B→27B configuration supplements $S ^ { 0 }$ with a correction conditioned on all 48 mapped target REC layers. At each layer, we compute row and column means, flatten each array in head order, and concatenate the two vectors. We concatenate the layer vectors in target-layer order and stop gradients through the resulting global summary. We partition it into 256-dimensional blocks, sum the blocks with alternating negative and positive signs, divide by the square root of the block count, and apply LayerNorm without trainable affine parameters. An affine projection followed by tanh produces coefficients $c _ { j , h , k }$ for each target layer $j ,$ head h, and correction term k.

Let $\bar { u } _ { j , h , k }$ and $\bar { v } _ { j , h , k }$ be trainable direction vectors after separate L2 normalization. The corrected state is

$$
S _ { j , h } ^ { + } = S _ { j , h } ^ { 0 } + \gamma d \rho _ { j , h } \sum _ { k = 1 } ^ { k _ { R } } c _ { j , h , k } \bar { u } _ { j , h , k } \bar { v } _ { j , h , k } ^ { \top } ,\tag{5}
$$

where $S _ { j , h } ^ { 0 } ~ \in ~ \mathbb { R } ^ { d \times d } , ~ d ~ = ~ 1 2 8 , ~ \gamma ~ = ~ 0 . 1$ , and $k _ { R } = 2 .$ The scale $\rho _ { j , h }$ is the RMS of the complete $d \times d$ matrix for that head, with gradients stopped through the scale computation. The added matrix has rank at most two; the base state $S ^ { 0 }$ has no imposed rank constraint.

The correction is trained separately with the REC maps, FA maps, Conv maps, and pretrained models fixed. At handoff, it is evaluated once before the target processes the prompt suffix. The target then continues with its native recurrent updates. The correction’s coefficients and scale depend nonlinearly on the current mapped state, so it cannot be merged into a fixed affine REC map. The 9B→27B configuration uses its calibrated $S ^ { 0 }$ directly.

Foldable FA calibration. Following the lowrank update parameterization of LoRA (Hu et al., 2022), the final calibration stage learns an increment to each K/V mapping weight:

$$
\begin{array} { r } { W _ { j } ^ { K / V } = W _ { j } ^ { 0 , K / V } + A _ { j } ^ { K / V } B _ { j } ^ { K / V } . } \end{array}\tag{6}
$$

Both model pairs use rank-four increments. The base FA weights and biases remain fixed. Randomly initialized $A _ { j }$ and zero-initialized $B _ { j }$ give an initial increment of zero. REC mappings, the REC correction when used, and Conv mappings are also frozen during this stage. Only the increment is rank-constrained; the full base mapping matrices

are retained.

The trained increment is independent of the transferred context. Before online handoff, it is merged into the base cache-mapping weight to obtain the $W _ { j } ^ { K / V }$ used in Equation (3). The online FA branch therefore applies one affine map for K and one for V. This merge changes the cache mapper, while both models’ pretrained weights remain fixed.

## 4.4 Training and Online Handoff

Base fitting uses paired source and native target states. REC mapping calibration and the separate 4B→27B correction training precede final FA calibration.

Both task-oriented training of the conditional REC correction and final FA calibration use forward KL over 32 continuation prediction positions for natural-text supervision. Mathematical sequences use forward KL plus token cross-entropy with coefficient 0.25 over 32 gold-solution prediction positions. Multiple-choice requests use forward KL over the complete candidate-score distribution plus correct-candidate cross-entropy with coefficient 0.25. For the multiple-choice data in these stages, each candidate score is the sum of its token log-probabilities. A softmax over the scores gives the candidate distribution. Each multi-token candidate is evaluated from an independent promptstate branch.

The native teacher is the frozen 27B target using its own prefix state. In the final FA stage of the 4B→27B configuration, the first eight mathematical prediction positions instead use a frozen transfer teacher. This teacher uses the base FA maps, calibrated REC maps, aligned Conv histories, and the trained REC correction, without the final FA increment. The remaining 24 positions use the native target teacher. All 32 positions receive equal weight, and token cross-entropy uses the same gold solution throughout. This teacher schedule applies to mathematical sequences. The 9B→27B configuration uses native target supervision at all 32 mathematical positions and has no conditional REC correction.

Online execution uses the saved parameters without optimization or teacher supervision:

1. Convert and install the boundary state. Apply the saved FA, REC, and Conv maps to the source state at C and assemble the target cache with consistent positions, cache lengths,

and metadata.

2. Apply the configured REC correction. In $4 \mathrm { B } \to 2 7 \mathrm { B }$ , update the installed REC states from $S ^ { 0 }$ to $S ^ { + }$ once using the global summary. The 9B→27B configuration retains $S ^ { 0 }$

3. Resume target computation. Process the real prompt suffix $x _ { C + 1 : N }$ using the target’s native update rules. At N, the target cache and final hidden state are ready. The languagemodeling head then projects the hidden state into vocabulary scores for subsequent answer scoring or generation.

The handoff body latency defined in Figure 2 is evaluated in Section 5.3.

FA conversion processes every retained K/V entry, so its cost grows with prefix length. REC and Conv have fixed sizes for a given model. State installation, the conditional correction when enabled, and target suffix computation also contribute to handoff cost. The latency benefit therefore depends on the target prefix computation saved and the cost of these remaining operations.

## 5 Experiments

## 5.1 Setup

Models and hardware. We evaluate Qwen3.5- 4B and Qwen3.5-9B as sources with Qwen3.5-27B as the target (Qwen Team, 2026). We use two NVIDIA A800 GPUs with 80GB of memory each. Evaluation uses batch size 1, BF16 model weights and FA caches, and FP32 REC states. Both models’ pretrained parameters remain fixed.

Tasks. We evaluate on ARC-Challenge (ARC-C) (Clark et al., 2018), HellaSwag (Zellers et al., 2019), WinoGrande (Sakaguchi et al., 2020), MMLU (Hendrycks et al., 2021), and GSM8K (Cobbe et al., 2021). We use the full test splits of ARC-C (1,172 examples), MMLU (14,042 examples across 57 subjects), and GSM8K (1,319 examples), and the full validation splits of HellaSwag (10,042 examples) and WinoGrande (1,267 examples). ARC-C, HellaSwag, and Wino-Grande use 0-shot evaluation. MMLU uses 5-shot prompts with demonstrations from each subject’s development split, and GSM8K uses 8-shot chainof-thought prompts.

<table><tr><td>Task</td><td>Ridge CacheBridge</td><td></td><td>CacheLink</td><td>Ours</td></tr><tr><td>4B→27B</td><td></td><td></td><td></td><td></td></tr><tr><td>ARC-C</td><td>76.69</td><td>75.86</td><td>90.30</td><td>92.78</td></tr><tr><td>HellaSwag</td><td>84.55</td><td>83.16</td><td>92.38</td><td>94.64</td></tr><tr><td>WinoGrande</td><td>67.13</td><td>65.93</td><td>86.87</td><td>89.27</td></tr><tr><td>MMLU</td><td>66.05</td><td>61.88</td><td>74.73</td><td>81.05</td></tr><tr><td>GSM8K</td><td>49.41</td><td>5.74</td><td>87.33</td><td>91.66</td></tr><tr><td>Avg.</td><td>68.77</td><td>58.51</td><td>86.32</td><td>89.88</td></tr><tr><td>9B→27B</td><td></td><td></td><td></td><td></td></tr><tr><td>ARC-C</td><td>96.52</td><td>94.81</td><td>95.66</td><td>103.28</td></tr><tr><td>HellaSwag</td><td>91.21</td><td>91.21</td><td>99.33</td><td>99.33</td></tr><tr><td>WinoGrande</td><td>77.36</td><td>77.36</td><td>87.28</td><td>87.28</td></tr><tr><td>MMLU</td><td>83.66</td><td>84.16</td><td>67.93</td><td>88.22</td></tr><tr><td>GSM8K</td><td>77.29</td><td>23.79</td><td>84.86</td><td>92.42</td></tr><tr><td>Avg.</td><td>85.21</td><td>74.27</td><td>87.01</td><td>94.11</td></tr></table>

Table 2: Task retention (%, higher is better) relative to native 27B. Avg. equally weights task retentions. Bold marks the highest point estimates, including ties, per task and direction. Retention above 100% indicates scores above native 27B. All methods use complete hybrid-state configurations.  
![](images/d5c283e51bf53a03bcd1549838fbc4dd6c65041b99a3498d38d4dacbd57aa363.jpg)  
Figure 4: Body latency comparison between state transfer and native target prefill for 9B→27B on synthetic contexts with an available source cache. At 16K tokens, transfer reduces latency from 8,823.37 to 402.79 ms, yielding a 21.91× speedup.

Baselines and metrics. We compare complete hybrid-state extensions of Cross-Model (Heo et al., 2026), CacheBridge (Qu et al., 2026), and Cache-Link (Qiu et al., 2026). We refer to our Cross-Model extension as Ridge. In 4B→27B, Ridge and CacheBridge use closed-form REC maps, whereas CacheLink and our method use functionally calibrated REC adaptation. All four methods use the same fitted Conv alignment maps for this direction. For the 4B→27B CacheLink extension, FA training uses native-target teachers with equally weighted token cross-entropy and forward KL; multiplechoice supervision uses token losses on gold candidate branches. Task retention is 100 $s _ { m , t } / s _ { T , t }$ where $s _ { m , t }$ and $s _ { T , t }$ are method and native-target scores. Avg. equally weights the five task retentions. Absolute accuracies, ablations and state diagnostics, and matching-layout handoffs are reported in Appendices A, B, and A.1.

## 5.2 Task Retention

Table 2 shows mean retentions of 89.88% and 94.11% for our method in 4B→27B and 9B→27B, respectively, exceeding CacheLink by 3.56 and 7.10 percentage points. The gains are taskdependent: our method outperforms CacheLink on all five tasks in 4B→27B, but ties on HellaSwag and WinoGrande in 9B→27B. The latter direction’s largest gain is on MMLU, where retention increases from 67.93% to 88.22%.

## 5.3 Handoff Body Latency with an Available Cache

We measure handoff body latency for 4B→27B and 9B→27B on synthetic contexts of 64–16K tokens with source caches already available. The timed transfer body includes state conversion, cache assembly and required copies, any REC correction, and target continuation until the target cache and hidden state are available. Timing excludes source prefill, the language modeling head, and answer generation. Speedup is computed as native target body latency divided by transfer body latency.

Figure 4 compares native target prefill and state transfer times for 9B→27B. Target prefill time grows more rapidly with context length than transfer time. At 16K, transfer reduces latency from 8,823.37 to 402.79 ms, yielding a 21.91× speedup, compared with 3.01× at 64 tokens.

Appendix C reports timing records for both directions. For 4B→27B, speedup increases from 2.78× at 64 tokens to 21.66× at 16K, where latency decreases from 8,644.23 to 399.02 ms.

## 6 Conclusion

We propose a unified framework for cross-model state handoff in the Qwen3.5 hybrid model family to reduce repeated prefill when switching models. The framework combines structural adaptation of full-attention caches, recurrent states, and convolution histories with functional calibration through the target model, including foldable linear updates for attention-cache conversion. Across five benchmarks, 4B→27B and 9B→27B achieve mean task retentions of 89.88% and 94.11%, respectively. With source caches already available, the two directions achieve handoff body speedups of 21.66× and 21.91×, respectively, on synthetic 16K contexts.

## Limitations

We evaluate HST only within the Qwen3.5 family with Gated DeltaNet layers. Transfer to other linear-attention architectures, including KDA (Zhang et al., 2025), and to MoE models remains untested. Our benchmarks also provide limited coverage of long-context understanding. Task quality and handoff body latency are measured on benchmarks and synthetic contexts, respectively. Joint evaluation on the same inputs and under consistent inference settings is needed to establish the quality–efficiency trade-off.

## References

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try ARC, the AI2 reasoning challenge. Preprint, arXiv:1803.05457.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. Preprint, arXiv:2110.14168.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Taekyung Heo, Rasoul Shafipour, Ritchie Zhao, Maximilian Golub, Mohammad Mahdi Kamani, Ritika Borkar, Makesh Tarun Chandran, Pantea Zardoshti, and Bita Darvish Rouhani. 2026. Cross-model KV cache transfer in LLM families: A closedform linear mapping for prefill reuse. Preprint, arXiv:2608.03893.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. 2020. Transformers are RNNs: Fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 5156–5165. PMLR.

Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M Waleed Kadous, and Ion Stoica. 2025. RouteLLM: Learning to route LLMs with preference data. In The Thirteenth International Conference on Learning Representations.

Guilherme Penedo, Hynek Kydlícek, Loubna Ben Al-ˇ lal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro von Werra, and Thomas Wolf. 2024. The FineWeb datasets: Decanting the web for the finest text data at scale. In Advances in Neural Information Processing Systems, volume 37.

Haomiao Qiu, Yining Tang, Zijia Song, Ruizi Han, Ziyue Qiao, and Miao Zhang. 2026. CacheLink: Functional KV transport for zero-prefill cross-model context handoff. Preprint, arXiv:2610.15509.

Xingyu Qu, Siyuan Lu, Zhiyu Chen, Sheng Wang, and Tao Lin. 2026. CacheBridge: Efficient cross-model KV cache transfer. Preprint, arXiv:2609.00891v1.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents. Official release. Accessed 2026-10- 09.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2020. WinoGrande: An adversarial Winograd schema challenge at scale. Proceedings ofthe AAAI Conference on Artificial Intelligence, 34(05):8732–8740.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. 2025. Gated delta networks: Improving Mamba2 with delta rule. In The Thirteenth International Conference on Learning Representations. Also available as arXiv:2412.06464v3.

Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. 2024. Parallelizing linear transformers with the delta rule over sequence length. In Advances in Neural Information Processing Systems, volume 37, pages 115491–115522.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting ofthe Associationfor Computational Linguistics, pages 4791–4800, Florence, Italy. Association for Computational Linguistics.

Yu Zhang, Zongyu Lin, Xingcheng Yao, Jiaxi Hu, Fanqing Meng, Chengyin Liu, Xin Men, Songlin Yang, Zhiyuan Li, Wentao Li, Enzhe Lu, Weizhou Liu, Yanru Chen, Weixin Xu, Longhui Yu, Yejie Wang, Yu Fan, Longguang Zhong, Enming Yuan, and 40 others. 2025. Kimi Linear: An expressive, efficient attention architecture. Preprint, arXiv:2510.26692.

## A Supplementary Benchmark Results

Table 3 reports absolute accuracies for native models and all evaluated handoff configurations. Macro equally averages the five task accuracies.

## A.1 Matching State Layouts

Table 4 reports supplementary handoffs with matching state layouts and separately calibrated maps.

## B Ablation and Diagnostic Results

These experiments use auxiliary 4B→27B checkpoints, separate from the main comparison. The deployment ablation uses a native GSM8K reference of 1270/1319 (96.29%); the main comparison uses 96.36%.

## B.1 Deployment Ablation

Table 5 toggles the FA increment and REC correction without retraining. The base FA, calibrated REC, and Conv maps remain fixed. Enabling both modules yields 88.32% accuracy.

## B.2 Native-State Replacement

Table 6 reports native-state replacement results on 18 documents. The selected components are replaced with native target states for the same prefix; all other components retain their transferred states at the handoff boundary. Composite KL equally weights the output KL at the end of the prompt and its mean over the next eight positions.

## C Handoff Latency and Speedup

Table 7 reports synthetic-input timings from 64 to 16K tokens. The native 27B reference is measured separately for each direction. With source caches already available, handoff body timing includes state conversion, installation and required copies, REC correction when used, and target suffix processing. Source prefill, the vocabulary output projection, and answer scoring or generation are excluded. These timing measurements are reported separately from benchmark quality results.

## D Training and Parameter Details

## D.1 Training Configuration

Experiments use two NVIDIA A800 GPUs with 80GB memory each. Evaluation uses batch size 1, BF16 model weights and FA caches, and FP32 REC states. Both pretrained backbones remain fixed.

The reported 4B→27B base-fitting recipe uses 500 FineWeb-Edu (Penedo et al., 2024) sequences of 1024 tokens, with four ordered source supports per FA and REC target layer. Conv maps share parameters across the four history slots. Both the weights and biases of the REC maps are functionally calibrated offline. The 4B→27B REC correction has rank at most 2 and uses a 256-dimensional state summary; 9B→27B uses its calibrated REC map directly. Both directions use rank-4 FA increments merged into the cache-mapping weights before online handoff.

During the final FA stage, the REC and Conv maps and the conditional REC correction, when used, remain fixed. The 4B→27B configuration uses frozen transfer-teacher supervision for the first eight gold-solution prediction positions of mathematical sequences and native-target supervision for the remaining 24. The 9B→27B configuration uses native-target supervision at all 32 mathematical prediction positions. Each final FA stage uses 384 training units and 768 updates. We use a learning rate of $3 \times 1 0 ^ { - 4 }$ and evaluate the final checkpoint.

## D.2 Final FA-Stage Parameters and Training Time

Table 8 reports the additional FA parameters, update counts, and elapsed times for the two main transfer directions. Other state adapters remain fixed during this stage. Times exclude teacher preparation, earlier state calibration, and evaluation.

## E Native Conv Restoration Diagnostic

Native Qwen3.5-9B is evaluated on eight packed 1,024-token blocks from eleven source documents, at cuts 128 and 512. We average cuts within each block, then average blocks equally. Conv is zeroed at each cut. Each branch consumes the case’s 32 true continuation tokens. After measuring step 4, selected REC and/or Conv states are restored once from the native reference at that step; FA is not restored. A further control restores REC, Conv, and FA together. The script verifies bitwise-equal native logits and zero KL for all 16×28 = 448 measurements at steps 5–32. Thus, FA caches retaining earlier perturbations can affect subsequent computation after REC and Conv have been restored. These oracle interventions are separate from transfer benchmarks.

<table><tr><td>Model / direction</td><td>Method</td><td>ARC-C</td><td>HellaSwag</td><td>WinoGrande</td><td>MMLU</td><td>GSM8K</td><td>Macro</td></tr><tr><td>4B</td><td>Native</td><td>54.27</td><td>73.12</td><td>70.24</td><td>69.72</td><td>89.39</td><td>71.35</td></tr><tr><td>9B</td><td>Native</td><td>56.14</td><td>78.08</td><td>72.85</td><td>69.92</td><td>93.40</td><td>74.08</td></tr><tr><td>27B</td><td>Native</td><td>61.52</td><td>83.37</td><td>78.77</td><td>86.51</td><td>96.36</td><td>81.31</td></tr><tr><td rowspan="4">4B→27B</td><td>Ridge</td><td>47.18</td><td>70.49</td><td>52.88</td><td>57.14</td><td>47.61</td><td>55.06</td></tr><tr><td>CacheBridge</td><td>46.67</td><td>69.33</td><td>51.93</td><td>53.53</td><td>5.53</td><td>45.40</td></tr><tr><td>CacheLink</td><td>55.55</td><td>77.02</td><td>68.43</td><td>64.65</td><td>84.15</td><td>69.96</td></tr><tr><td>Ours</td><td>57.08</td><td>78.90</td><td>70.32</td><td>70.12</td><td>88.32</td><td>72.95</td></tr><tr><td rowspan="4">9B→27B</td><td>Ridge</td><td>59.38</td><td>76.04</td><td>60.94</td><td>72.37</td><td>74.48</td><td>68.64</td></tr><tr><td>CacheBridge</td><td>58.33</td><td>76.04</td><td>60.94</td><td>72.81</td><td>22.92</td><td>58.21</td></tr><tr><td>CacheLink</td><td>58.85</td><td>82.81</td><td>68.75</td><td>58.77</td><td>81.77</td><td>70.19</td></tr><tr><td>Ours</td><td>63.54</td><td>82.81</td><td>68.75</td><td>76.32</td><td>89.06</td><td>76.10</td></tr><tr><td rowspan="2">4B→9B</td><td>Ridge</td><td>56.83</td><td>75.35</td><td>68.59</td><td>71.91</td><td>84.53</td><td>71.44</td></tr><tr><td>Ours</td><td>55.97</td><td>76.09</td><td>71.51</td><td>69.72</td><td>88.32</td><td>72.32</td></tr><tr><td>9B→4B</td><td>Ours</td><td>53.38</td><td>72.37</td><td>69.09</td><td>74.60</td><td>87.01</td><td>71.29</td></tr></table>

Table 3: Absolute benchmark accuracy (%). The native target provides the reference for each transfer direction. Macro equally averages tasks.

<table><tr><td>Compatible configuration</td><td>ARC-C</td><td>HellaSwag</td><td>WinoGrande</td><td>MMLU</td><td>GSM8K</td><td>Avg.</td></tr><tr><td>4B→9B (Ridge)</td><td>101.23</td><td>96.50</td><td>94.15</td><td>102.85</td><td>90.50</td><td>97.05</td></tr><tr><td>4B→9B (Ours)</td><td>99.70</td><td>97.45</td><td>98.16</td><td>99.71</td><td>94.56</td><td>97.92</td></tr><tr><td>9B→4B (Ours)</td><td>98.36</td><td>98.97</td><td>98.36</td><td>107.00</td><td>97.34</td><td>100.01</td></tr></table>

Table 4: Task retention (%) for matching-layout handoffs. Native 9B and native 4B are the respective references for 4B→9B and 9B→4B. Avg. equally averages task retentions.

<table><tr><td>Configuration</td><td>FA increment</td><td>REC correction</td><td>Correct / total Accuracy (%)</td><td></td></tr><tr><td>A</td><td>Off</td><td>Off</td><td>832 / 1319</td><td>63.08</td></tr><tr><td>B</td><td>Off</td><td>On</td><td>878 / 1319</td><td>66.57</td></tr><tr><td>C</td><td>On</td><td>Off</td><td>1144 / 1319</td><td>86.73</td></tr><tr><td>D</td><td>On</td><td>On</td><td>1165 / 1319</td><td>88.32</td></tr><tr><td>Native 27B</td><td></td><td></td><td>1270 / 1319</td><td>96.29</td></tr></table>

Table 5: GSM8K deployment ablation in a fixed auxiliary configuration. Modules are toggled without retraining.

<table><tr><td>Native-state replacement Composite KL ↓</td><td></td></tr><tr><td>None</td><td>0.59291</td></tr><tr><td>REC only</td><td>0.10060</td></tr><tr><td>FA only</td><td>0.33515</td></tr><tr><td>REC and FA</td><td>0.01664</td></tr><tr><td>Conv only</td><td>0.61845</td></tr></table>

Table 6: Native-state replacement on 18 documents. Only the specified components are replaced with native target states for the same prefix. Lower composite KL is better.

<table><tr><td rowspan="2">Input length</td><td colspan="3">4B→27B</td><td colspan="3">9B→27B</td></tr><tr><td>Native (ms)</td><td>Transfer (ms)</td><td>Speedup</td><td>Native (ms)</td><td>Transfer (ms)</td><td>Speedup</td></tr><tr><td>64</td><td>367.43</td><td>132.20</td><td>2.78×</td><td>365.41</td><td>121.46</td><td>3.01×</td></tr><tr><td>128</td><td>384.88</td><td>134.32</td><td>2.87×</td><td>383.93</td><td>123.83</td><td>3.10×</td></tr><tr><td>256</td><td>410.29</td><td>135.71</td><td>3.02×</td><td>410.48</td><td>126.12</td><td>3.25×</td></tr><tr><td>512</td><td>468.38</td><td>139.97</td><td>3.35×</td><td>462.58</td><td>127.43</td><td>3.63×</td></tr><tr><td>1K</td><td>589.90</td><td>147.09</td><td>4.01×</td><td>593.08</td><td>136.37</td><td>4.35×</td></tr><tr><td>2K</td><td>1068.88</td><td>162.24</td><td>6.59×</td><td>1077.29</td><td>154.58</td><td>6.97×</td></tr><tr><td>4K</td><td>2068.30</td><td>198.62</td><td>10.41×</td><td>2082.30</td><td>200.31</td><td>10.40×</td></tr><tr><td>8K</td><td>4114.88</td><td>269.78</td><td>15.25×</td><td>4143.11</td><td>275.72</td><td>15.03×</td></tr><tr><td>16K</td><td>8644.23</td><td>399.02</td><td>21.66×</td><td>8823.37</td><td>402.79</td><td>21.91×</td></tr></table>

Table 7: Native target and handoff body times in milliseconds. K denotes 1024 tokens; speedup is native time divided by transfer time. Values are rounded to two decimal places.
<table><tr><td>Configuration</td><td>Additional FA parameters (M)</td><td>Updates</td><td>Stage time (h)</td></tr><tr><td>Main 4B→27B</td><td>0.655</td><td>768</td><td>2.92</td></tr><tr><td>Main 9B→27B</td><td>0.655</td><td>768</td><td>3.36</td></tr></table>

Table 8: Final FA-stage parameters, updates, and elapsed times for the main systems. Each stage uses 384 training units. Times cover final FA updates only.

<table><tr><td>Step</td><td>REC+Conv</td><td>REC+Conv+FA</td></tr><tr><td>18</td><td>0.00703398</td><td>0.00000000</td></tr><tr><td>19</td><td>0.01039322</td><td>0.00000000</td></tr><tr><td>20</td><td>0.00508434</td><td>0.00000000</td></tr></table>

Table 9: Output KL after oracle restoration in native Qwen3.5-9B. Both columns use the same eight blocks and two cuts per block.

The REC+Conv branch reaches its highest postrestoration mean KL at step 19 (0.01039322; Table 9). Block 2 (zero-based), cut 128, has KL 0.13518570, contributing 81.29% of this mean. The mean over the remaining 15 cases is 0.00207372. The aggregate peak is dominated by this case, whose continuation remains within one source document.