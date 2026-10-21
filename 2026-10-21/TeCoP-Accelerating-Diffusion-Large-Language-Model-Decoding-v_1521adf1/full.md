# TeCoP: Accelerating Diffusion Large Language Model Decoding via Tentative Context Pipelining

Jui-Chien Tsou<sup>∗</sup> Hsuan-Chi Liu<sup>∗</sup> Wei-Chin Wang Yu-Syuan Xu Chun-Yi Lee

National Taiwan University

## Abstract

Diffusion large language models (dLLMs) predict multiple token positions per forward pass, yet block-wise decoders serialize generation and repeatedly recompute stable predictions. We attribute this inefficiency to coupling a prediction’s readiness for irreversible commitment with its utility as context. We introduce Tentative Context Pipelining (TeCoP), a trainingfree decoder that decouples these roles through staggered block-parallel pipelining and draftand-verify. TeCoP reuses stable but uncommitted predictions as provisional cross-block context, with commitment governed by confidence and stability checks. Across four LLaDA and Dream dLLMs on GSM8K, Minerva MATH, and MBPP, TeCoP reduces number of forward evaluations by up to 87% and achieves up to 7.50× throughput without KV caching and 11.77× with caching over the uncached original decoder. It outperforms Fast-dLLM and DSB in throughput across all 12 model– benchmark settings, with and without caching, while keeping average cached accuracy within one percentage point of the original decoder.

## 1 Introduction

The decoding efficiency of diffusion large language models (dLLMs) hinges on how aggressively their parallel prediction capacity is converted into committed tokens. Unlike autoregressive models that generate strictly left to right, dLLMs iteratively refine masked positions through denoising (He et al., 2023; Sahoo et al., 2024) and can predict many positions in a single forward pass. Recent large-scale instances such as LLaDA (Nie et al., 2026) and Dream (Ye et al., 2025) reach competitive accuracy on general language understanding, instruction following, and reasoning benchmarks. This parallel potential, however, remains largely unrealized in practice, and the bottleneck lies in how generation is currently scheduled. Existing decoders (Arriola et al., 2025; Wu et al., 2025b) pair block-wise semi-autoregressive schedules with conservative confidence-based commit rules: a later block waits until its predecessors are resolved, and within the active block only the positions whose confidence clears a high threshold are committed, even though every masked position is predicted in the same forward pass. The rest are masked again and recomputed on the next pass, so each pass advances only a handful of tokens and decoding degenerates toward the token-by-token generation that diffusion models were designed to escape. As a result, dLLMs spend much of their compute re-deciding already stable predictions, leaving their parallel prediction capacity underused.

![](images/5d2cc43229ea14d0924acda0c6f5089fea49720636e1ba69c3a08538857f1123.jpg)  
Figure 1: Visualization of pending-position confidence during masked diffusion decoding. Left: In the baseline, many positions outside the active block window achieve high confidence but cannot be committed. Right: Our pipeline method progressively expands the active block region, enabling these predictions to be committed earlier and reducing wasted forward passes.

Recent work has sought faster dLLM decoding through approximate caching, confidencethresholded parallel commitment, and adaptive block schedules (Wu et al., 2025b; Israel et al., 2025; Mohamed et al., 2025; Luo et al., 2026a).

Much of it rests on a shared observation that a prediction often stabilizes several steps before it crosses the commit threshold, and confidence-only commitment is therefore conservative (Wang et al., 2025a; Li et al., 2025b; Zhang et al., 2026; Luo et al., 2026b). These methods act on that signal either by committing such tokens earlier or by deferring uncertain ones, and in both cases the predictions stay outside the decoding context until finalized. The caution is deliberate, as an unreliable prediction admitted into the context could propagate its error to every position that conditions on it. The cross-block utility of uncommitted predictions nevertheless remains unexploited, since a future-block token can become predictable before its preceding block resolves, while conventional block-wise decoding masks these positions and defers later blocks. The one pipeline that crosses block boundaries, D2F (Wang et al., 2025b), obtains this ability by distilling the model to condition on partially denoised prefixes, and no training-free counterpart exists for frozen models. Beyond this cross-block loss, the same masking discards stable predictions within a block that have yet to clear the commit threshold, and forces the redundant recomputation noted above. Across these methods, an uncommitted prediction is rarely treated as more than a candidate to commit or discard, and two gaps open between what a model has already produced and what it is permitted to exploit as context, one across blocks and one within them.

Both gaps identified above trace to a single root cause: existing decoders treat two distinct properties of a prediction as one. A prediction has a readiness to be committed, which requires it to be reliable enough to finalize irreversibly, and a utility as context, which requires only that it be accurate enough to guide other positions. Since a prediction enters the working context only once it clears the commit threshold, the decoder structurally binds the second property to the first. Fig. 1 illustrates the resulting waste: under a conventional schedule, many positions outside the active block reach high confidence well before they become eligible to commit, yet remain masked and repeatedly recomputed. Separating these properties allows stable but uncommitted predictions to serve as provisional context for other blocks. Realizing this requires two components: a schedule that keeps several blocks open and a verification mechanism that checks tentative predictions before commitment.

To this end, we propose Tentative Context Pipelining (TeCoP), a training-free decoder that requires no change to the model. It realizes the schedule as a block-parallel pipeline that keeps multiple blocks active within a single forward pass, and the verification discipline as a draft-and-verify procedure that reuses uncommitted-but-stable predictions as provisional cross-block context, while commitment remains governed by confidence and cross-round stability. Fig. 2 illustrates the resulting decoding trajectories and throughput–accuracy trade-off. Our contributions are as follows:

• We identify the coupling between commitment readiness and contextual utility as a key bottleneck in dLLM decoding, which can be relaxed at inference time without retraining.

• We propose TeCoP, a training-free decoder that combines cross-block pipelining with the reuse of stable but uncommitted predictions as provisional context, committing them only after verification.

• Across four LLaDA and Dream dLLMs on GSM8K, Minerva MATH, and MBPP, TeCoP outperforms Fast-dLLM and DSB in throughput across all settings, achieving 1.03–1.31× Fast-dLLM’s throughput without KV caching and 1.26–1.58× with caching. Relative to standard block-wise decoding, it reaches up to 7.50× and 11.77× speedups without and with caching, respectively, while keeping average accuracy within one percentage point.

## 2 Related Work

Diffusion Language Models. Diffusion language models cast text generation as iterative denoising over discrete tokens, and masked diffusion has become the dominant instantiation of this paradigm (He et al., 2023; Sahoo et al., 2024). The paradigm has since expanded into a broad family of systems: open-weight models such as LLaDA and Dream address general instruction following and mathematical reasoning (Nie et al., 2026; Ye et al., 2025), while Gemini Diffusion, Mercury, and Seed Diffusion demonstrate the approach in industrial settings (Google DeepMind, 2025; Khanna et al., 2025; Song et al., 2025). Dream-Coder and DiffuCoder further specialize this line for code generation (Xie et al., 2025; Gong et al., 2025). Within this family, our work builds on masked dLLMs: the model architecture, tokenizer, and training objective remain unchanged, and the improvement lies entirely in inference-time decoding.

![](images/7f110c7ffefa8a7a9cb44d334236640e7e11b18e398c0749eb76254fd5a2a6e8.jpg)

![](images/0b7210f751c4b9b921f6964120c186b3d81778d7502fe5a37e76c388b0fd9ae0.jpg)  
Figure 2: Overview of the decoding efficiency of Tentative Context Pipelining (TeCoP). Left: Decoding trajectories of the original block-wise decoder, Fast-dLLM (Wu et al., 2025b), DSB (Luo et al., 2026a), and TeCoP, colored by the step at which each token is committed. TeCoP activates multiple blocks in parallel and finishes in fewer number of forward evaluations. Right: Throughput–accuracy trade-off on GSM8K with KV caching; speedups are relative to the uncached original decoder.

Block-wise Decoding for Diffusion Language Models. Block-wise semi-autoregressive decoding is the prevailing inference strategy for dLLMs: the output is split into contiguous blocks, and each is decoded from the committed tokens of earlier blocks (Arriola et al., 2025). This coarse left-toright structure improves reliability and forms the basis of most efficient dLLM decoders. Existing refinements target different parts of this pipeline, through confidence-based and adaptive commitment (Wu et al., 2025b; Israel et al., 2025; Mohamed et al., 2025; Wang et al., 2025a; Chen et al., 2025a; Li et al., 2025b; Zhang et al., 2026), dependency-aware token selection (Luo et al., 2026b), cache reuse across denoising steps (Wu et al., 2025a; Ma et al., 2025; Liu et al., 2025b; Jiang et al., 2025), and dynamic block schedules (Luo et al., 2026a; Mohamed et al., 2025). These refinements largely treat decoding as a selection problem over positions, thresholds, or schedules. Since their blocks still run in sequence, the decoder advances only within a narrow active region and leaves the parallel predictions already computed for later regions underused.

Accelerating Sequential Block-wise Decoding. A growing body of work relaxes the strict sequencing of block-wise decoding. Among these, the method most relevant to ours is Discrete Diffusion Forcing (Wang et al., 2025b), which pipelines blocks under staggered activation thresholds and conditions later blocks on partially denoised earlier ones. It reaches this inter-block parallelism through asymmetric distillation that retrains the model to denoise from partial prefixes, whereas TeCoP adds no training and runs on frozen dLLMs. The same relaxation appears in other cross-block and streaming schemes that decode later regions before earlier blocks are fully resolved (Liu et al., 2025a; Wu et al., 2025a), and in learned parallel decoders that distill faster certainty dynamics into the model (Chen et al., 2025b; Bao et al., 2025). Beyond these diffusion-specific schedules, sequential dependence has also been reduced at the algorithm level: Jacobi and lookahead decoding solve for several tokens at once (Santilli et al., 2023; Fu et al., 2024), and speculative decoding proposes draft tokens and verifies them against the target model (Leviathan et al., 2023; Chen et al., 2023), an idea recently carried into diffusion generation (Christopher et al., 2025; Li et al., 2025a; Chen et al., 2026; Wu and Zhang, 2025; Gao et al., 2025; Cui et al., 2026). Nearest in spirit to TeCoP is a line that treats non-final predictions as usable signal, though each method exploits it differently: Soft-Mask represents uncertain predictions as soft tokens (Hersche et al., 2025), WINO verifies or remasks unreliable ones (Hong et al., 2026), and deferred-commitment schedules wait on uncertain positions for more context (Shu et al., 2026; Luo et al., 2026b). TeCoP reuses this signal in a different way: selected uncommitted predictions are retained as discrete tentative drafts inside the same masked-dLLM state, supply provisional context across active blocks, and are committed only through a confidence or cross-round stability check. Unlike the trained pipeline of Wang et al. (2025b), TeCoP obtains this cross-block context at inference time on a frozen model. Unlike methods that defer or discard uncertain tokens, it reuses stable predictions as context before they commit.

## 3 Preliminaries

## 3.1 Masked Diffusion Language Models

TeCoP builds on the decoding process of masked diffusion language models, and this section sets up the decoding loop and notation that the method later modifies. In these models, generation reverses a forward process that masks tokens in a sequence. Specifically, at timestep s each position of a clean sequence $x _ { 0 } = ( x _ { 0 , 1 } , \dots , x _ { 0 , L } )$ is either preserved or replaced by the [MASK] symbol, as follows:

$$
x _ { s , i } = \left\{ \begin{array} { l l } { x _ { 0 , i } , } & { \mathrm { w i t h ~ p r o b a b i l i t y } ~ \rho _ { s } , } \\ { \lbrack \mathrm { M A S K } ] , } & { \mathrm { w i t h ~ p r o b a b i l i t y } ~ 1 - \rho _ { s } . } \end{array} \right.\tag{1}
$$

where $\rho _ { s }$ is the probability that a token is preserved. The model $p _ { \theta }$ is trained to recover the original token at each masked position, formulated as:

$$
p _ { \theta } ( x _ { 0 , i } \mid x _ { s } ) , \quad i \in \mathcal { M } _ { s } ,\tag{2}
$$

with $\mathcal { M } _ { s } \ = \ \{ i | x _ { s , i } \ = \ [ \mathrm { M A S K } ] \}$ the set of masked positions. At inference, decoding no longer tracks the diffusion timestep and instead proceeds in rounds indexed by r. Each round performs one forward pass and then commits a subset of positions, and the working sequence $x ^ { ( r ) }$ holds committed tokens in place while the rest stay masked. Standard block-wise decoding feeds this sequence back unchanged, with $z ^ { ( r ) } = x ^ { ( r ) }$ as the model input. From $z ^ { ( r ) }$ , every masked position is scored in parallel over the vocabulary V, and each yields a most likely token and a confidence, computed as:

$$
\begin{array} { r l } & { \hat { x } _ { i } ^ { ( r ) } = \underset { v \in \mathcal { V } } { \arg \operatorname* { m a x } } p _ { \theta } ( x _ { i } = v \mid z ^ { ( r ) } ) , } \\ & { c _ { i } ^ { ( r ) } = \underset { v \in \mathcal { V } } { \arg \operatorname* { m a x } } p _ { \theta } ( x _ { i } = v \mid z ^ { ( r ) } ) . } \end{array}\tag{3}
$$

A single forward pass therefore scores every masked position at once, and how much of this parallel capacity a schedule turns into committed tokens is what separates one decoder from another.

## 3.2 Block-wise Diffusion Decoding

This section formalizes the schedule that turns the model’s per-round predictions into a finished sequence: the block-wise, left-to-right loop that almost every dLLM decoder adopts, and the one TeCoP later reorganizes. Such a decoder splits the response into $m = \lceil L / b \rceil$ contiguous blocks of length b, defined as follows:

$$
B _ { j } = \{ ( j - 1 ) b + 1 , \ldots , \operatorname* { m i n } ( j b , L ) \} .\tag{4}
$$

These blocks are decoded from left to right, and a block is completed before the next opens. At each round, completed blocks act as fixed context, the active block is decoded in parallel, and later blocks remain masked. Within the active block $B _ { j }$ , every uncommitted position is scored in one forward pass, and a confidence gate commits the positions whose confidence reaches a threshold τ, given by:

$$
\begin{array} { r } { \mathcal { C } ^ { ( r ) } = \left\{ i \in B _ { j } \ \Big | \ x _ { i } ^ { ( r ) } = [ \mathrm { M A S K } ] , \ c _ { i } ^ { ( r ) } \geq \tau \right\} . } \end{array}\tag{5}
$$

Committed tokens are fixed and never revised, and the rest stay masked for a later round. This schedule trades parallelism for stability by deferring uncertain predictions, and its cost is that a later block cannot contribute until the blocks before it are resolved. Relaxing this constraint is what TeCoP sets out to do.

## 4 Methodology

This section presents the architecture of TeCoP, which realizes the decoupling identified in Section 1: a prediction may guide other positions as context long before it is reliable enough to commit. Fig. 3 illustrates the full design, built from two mechanisms that share one pool of active blocks. The block-parallel pipeline schedule advances several blocks at once and allows a later block to begin before its predecessor finishes. The draft-andverify procedure reuses uncommitted-but-stable predictions as provisional context to help other unresolved positions decode, while using stability as an additional signal for commitment. Algorithm 1 states the complete loop.

## 4.1 Pipelined Block Schedule

TeCoP achieves its concurrency through the pipeline schedule, which maintains a sliding window of active blocks in place of the single block of standard decoding. The window is bounded on two axes. The pipeline depth K caps its width, the largest number of blocks active together. The activation threshold α controls its advance: the next block opens once the current frontier block has committed an α-fraction of its tokens. As this window slides across the sequence, each block passes through three states: pending while it waits fully masked, active once it enters the window and joins each forward pass, and frozen after all its tokens commit and it becomes fixed context. The activation threshold α protects quality as the window advances. Although the parallel forward pass could open all K blocks at once, premature activation carries a cost: a block opened before its predecessor commits anything must predict from an almost fully masked left context, and its unreliable predictions propagate through the rounds that follow. The threshold rules this out by opening a block only after the current frontier block has committed an α-fraction of its tokens, which anchors every new block in reliable context. Under this schedule, later blocks advance while their predecessors are still resolving, and this recovers the parallelism that strict left-to-right decoding forgoes.

![](images/2c0dc23a5e8287df1088865be140cf95eaf8c2ae9c3387f76c605d322115b276.jpg)

![](images/918a5fcfb8a6f3766337c5889e62ff08acede74d62e39dbcda86ae300967a8d9.jpg)  
Figure 3: Illustration of Tentative Context Pipelining (TeCoP) mechanism. The pipelined block schedule keeps multiple blocks simultaneously active and progressively opens new blocks according to the activation threshold α (set to 75% here for illustration). The draft-and-verify procedure recycles uncommitted predictions as tentative drafts, uses them as provisional context in the next round, and commits them through a confidence or stability check.

## 4.2 Drafting with Provisional Context

With several blocks overlapping, the draft-andverify procedure makes use of the predictions they produce before those predictions are committed. A position whose confidence clears a draft threshold $\tau _ { d }$ becomes a tentative draft: it serves as provisional context in the next round and may later be committed if it remains stable across rounds and satisfies the verification threshold. This speculative path operates alongside the pipeline’s ordinary confidence gate, which retains the high-confidence safe-commit rule.

A draft is never written into the committed sequence itself. Each active position is instead one of three states, listed in Table 1: committed, tentative, or masked, with tentative drafts held in a separate buffer. This separation matters to the pipeline, since the activation rule of Section 4.1 gauges a block’s committed fraction by counting non-mask entries of the committed sequence, and routing drafts through it would inflate that count and open blocks too early.

Table 1: States in an active block.
<table><tr><td>State Symbol Meaning</td></tr><tr><td>Committed  $x _ { i }$  finalized; value locked Tentative  $\tilde { x } _ { i }$  draft from a prior round</td></tr><tr><td>Masked [M] no draft yet 60 1.0</td></tr><tr><td>Aveece corag Baseline</td></tr><tr><td>Draft and verify 0.8 40 0.6</td></tr><tr><td>0.4 20</td></tr><tr><td>Men  once Baseline oo</td></tr><tr><td>0.2 Draft and Verify 0.0 100 200 0 20 40 60 80 Token position Decoding step</td></tr></table>

![](images/772ec6b0e09a679e7599f98d9bb4be82ca5fceb719a869d1db520d817023dafa.jpg)  
Figure 4: Draft-and-verify analysis on 100 GSM8K samples. Left: commit lag by token position. Right: token confidence over decoding steps.

Within each round, a re-drafting step refreshes the drafts alongside commitment. Every stillmasked position whose confidence reaches $\tau _ { d }$ records its current prediction as a fresh draft, and any draft that failed verification is refreshed to the latest prediction. Both updates happen inside the same forward pass that performs the commit, which holds the amortized cost at one pass per round. The threshold $\tau _ { d }$ is the procedure’s main safety valve:

Algorithm 1: TeCoP Decoding   
Input: prompt, length L, block size b, depth K,   
thresholds $\tau , \tau _ { d } , \tau _ { v } , \alpha$   
Output: committed sequence x   
x ← [MASK] × L; x˜ ← ∅ × L // tentative   
drafts   
$\{ B _ { 1 } , \ldots , B _ { m } \}  \mathrm { s p l i t } ( L , b ) ;$ activate(B<sub>1</sub>)   
while not all\_frozen $\mathrm { ' } B _ { 1 } , \ldots , B _ { m } )$ do   
A ← positions of active blocks   
// build input with tentative context   
z ← x   
for $; \in \mathcal { A }$ with $x _ { i } = [ \mathrm { M A S K } ]$ do   
$\mathbf { i f } \widetilde { x } _ { i } \neq \varnothing$ then $\bar { z } _ { i } \gets \tilde { x } _ { i }$   
end   
// one forward pass; scores every masked   
$i \in \mathcal A$   
$\begin{array} { r } { \hat { x } _ { i }  \arg \operatorname* { m a x } _ { v \in \mathcal { V } } p _ { \theta } ( x _ { i } = v \mid z ) ; } \end{array}$   
$\begin{array} { r } { c _ { i } \gets \operatorname* { m a x } _ { v \in \mathcal { V } } p _ { \theta } ( x _ { i } { = } v \mid z ) } \end{array}$   
n ← num\_committed(x)   
// safe-commit or verify-commit   
for $i \in \mathcal A$ with $x _ { i } = [ \mathrm { M A S K } ]$ do   
if $c _ { i } \geq \tau$ then   
$x _ { i } \gets \hat { x } _ { i }$ // safe   
else if $\tilde { x } _ { i } \neq$ ∅ and $\hat { x } _ { i } = \tilde { x } _ { i }$ and $c _ { i } \geq \tau _ { v }$ then   
x<sub>i</sub> ← xˆ<sub>i</sub> // verified   
end   
end   
$/ /$ ensure progress if no token commits   
if num\_committed $( x ) = n$ then   
$i ^ { \star } \gets$ arg max<sub>i∈A: x =[MASK]</sub> c<sub>i</sub>   
$x _ { i ^ { \star } } \gets \hat { x } _ { i ^ { \star } }$   
end   
// refresh tentative drafts   
for $i \in \mathcal A$ with $x _ { i } = [ \mathrm { M A S K } ]$ do   
$\mathbf { i f } \widetilde { x } _ { i } \neq \varnothing$ or $c _ { i } \geq \tau _ { d }$ then $\tilde { x } _ { i } \gets \hat { x } _ { i }$   
end   
// advance pipeline   
for $B _ { j }$ in active blocks do   
if fully\_committed $\left( B _ { j } \right)$ then freeze $\left( B _ { j } \right)$   
end   
if committed\_ratio(frontier) ≥ α and   
num\_active() < K then   
activate(next\_pending())   
end   
end   
return x

it determines how reliable a prediction must be before it may act as context, and this protects later denoising from misleading drafts.

## 4.3 Verification

A draft provides provisional context before commitment, and each decoding round applies two commit rules before re-drafting.

Safe-commit. Every masked position whose confidence reaches τ is committed using the standard confidence-based criterion. Tentative context alone does not trigger commitment; a token must satisfy either this confidence criterion or the verification rule below before being finalized, apart from the progress-guaranteeing fallback.

Verify-commit. A tentative position is committed when its prediction is unchanged from the draft, $\hat { x } _ { i } = \tilde { x } _ { i }$ , and its confidence reaches a verify threshold $\tau _ { v }$ . This rule admits tokens that stay consistent across rounds, including sub-τ tokens that the confidence gate alone would discard round after round, and it rejects tokens that are only briefly confident. Fallback. If neither commit path selects a token, TeCoP commits the highest-confidence unresolved position, following the fallback used by the original decoder. This guarantees progress even when no prediction reaches either commit threshold. In our fixed-length setting, decoding terminates once all L positions are committed and all blocks are frozen.

The effect of this second path appears in commit lag, the number of consecutive steps a position predicts the same token before finalization. Fig. 4 shows that draft-and-verify reduces average lag from 11.7 to 8.9 (23.9%) on held-out GSM8K and increases token confidence earlier in decoding. Both suggest that provisional context helps unresolved predictions stabilize and commit sooner.

## 5 Experiments

Models and benchmarks. We test TeCoP on four diffusion language models from two widely used open dLLM families: LLaDA-8B-Instruct (Nie et al., 2026), LLaDA-1.5 (Zhu et al., 2026), Dream-v0-Base-7B, and Dream-v0-Instruct-7B (Ye et al., 2025). We evaluate mathematical reasoning on GSM8K (Cobbe et al., 2021) and Minerva MATH (Hendrycks et al., 2021; Lewkowycz et al., 2022), and code generation on MBPP (Austin et al., 2021). GSM8K uses 5-shot exact match, Minerva MATH uses 4-shot math\_verify, and MBPP uses 3-shot pass@1, all evaluated with lm-eval (Biderman et al., 2024).

Baselines. We compare against three decoders. Original is standard block-wise decoding with the full denoising budget and serves as the reference for speedup. Fast-dLLM (Wu et al., 2025b) commits every position whose confidence exceeds a threshold, and DSB (Luo et al., 2026a) is a training-free method that relaxes fixed block boundaries. Each method is evaluated without KV caching, to isolate reductions in number of forward evaluations, and with its own cache: DualCache for Fast-dLLM, the DSB cache for DSB, and Fast-dLLM’s block-wise prefix KV cache for TeCoP.

Table 2: Main results on GSM8K, Minerva MATH, and MBPP. Cached variants use DualCache (Fast-dLLM), DSB’s cache (DSB), and prefix KV cache (TeCoP). Spd. denotes TPS speedup over uncached Original. For Acc., bold and underline mark the top two among Original and cached variants (ties included). For NFE, TPS, and Spd., bold and underline mark the best across all methods.
<table><tr><td rowspan="2">Method</td><td colspan="4">GSM8K</td><td colspan="4">Minerva MATH</td><td colspan="4">MBPP</td></tr><tr><td>Acc.(%) NFE↓</td><td></td><td>TPS↑</td><td></td><td>Spd.↑ Acc.(%) NFE↓</td><td></td><td>TPS↑</td><td>Spd. ↑</td><td>Acc.(%)</td><td>NFE↓</td><td>TPS↑</td><td>Spd.↑</td></tr><tr><td colspan="10">LLaDA-8B-Instruct</td><td></td><td></td><td></td></tr><tr><td>Original</td><td>76.8</td><td>256.0</td><td>16.8</td><td>1.00×</td><td>33.2</td><td>256.0</td><td>24.1</td><td>1.00×</td><td>40.2</td><td>256.0</td><td>16.2</td><td>1.00×</td></tr><tr><td>Fast-dLLM</td><td>77.5</td><td>78.2</td><td>54.7</td><td>3.25×</td><td>33.0</td><td>97.1</td><td>63.5</td><td>2.63×</td><td>40.6</td><td>70.1</td><td>59.1</td><td>3.65×</td></tr><tr><td>DSB</td><td>77.7</td><td>74.5</td><td>57.4</td><td>3.41×</td><td>33.1</td><td>93.6</td><td>65.6</td><td>2.72×</td><td>41.8</td><td>67.2</td><td>60.2</td><td>3.72×</td></tr><tr><td>TeCoP</td><td>76.7</td><td>61.8</td><td>71.7</td><td>4.26×</td><td>32.5</td><td>77.4</td><td>81.8</td><td>3.40×</td><td>41.2</td><td>53.1</td><td>75.8</td><td>4.68×</td></tr><tr><td>Fast-dLLM + cache DSB + cache</td><td>77.3</td><td>85.3</td><td>90.7</td><td>5.39×</td><td>32.7</td><td>107.3</td><td>80.0</td><td>3.32×</td><td>37.8</td><td>70.2</td><td>76.0</td><td>4.70×</td></tr><tr><td>TeCoP + cache</td><td>78.4 78.1</td><td>78.4 64.1</td><td>100.8 132.4</td><td>5.99× 7.87×</td><td>33.5 32.9</td><td>98.0</td><td>88.5 122.9</td><td>3.67×</td><td>42.6</td><td>65.8</td><td>79.7</td><td>4.93× 6.85×</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>80.0</td><td></td><td>5.10×</td><td>39.0</td><td>53.7</td><td>110.9</td><td></td></tr><tr><td colspan="10">LLaDA-1.5</td><td colspan="3"></td></tr><tr><td>Original</td><td>80.3</td><td>256.0</td><td>15.7</td><td>1.00×</td><td>33.5</td><td>256.0</td><td>21.3</td><td>1.00×</td><td>39.2</td><td>256.0</td><td>5.9</td><td>1.00×</td></tr><tr><td>Fast-dLLM</td><td>80.8</td><td>75.3</td><td>52.7</td><td>3.36×</td><td>33.6</td><td>95.6</td><td>56.8</td><td>2.66×</td><td>39.2</td><td>42.0</td><td>34.4</td><td>5.80×</td></tr><tr><td>DSB</td><td>81.0</td><td>71.7</td><td>55.6 69.2</td><td>3.54×</td><td>34.2</td><td>94.4</td><td>57.0</td><td>2.67×</td><td>34.8</td><td>42.0</td><td>32.9</td><td>5.55×</td></tr><tr><td>TeCoP</td><td>79.2</td><td>59.7</td><td></td><td>4.41×</td><td>32.2</td><td>76.1</td><td>72.6</td><td>3.40×</td><td>39.8</td><td>32.4</td><td>44.4</td><td>7.50×</td></tr><tr><td>Fast-dLLM + cache DSB + cache</td><td>80.6</td><td>84.2</td><td>84.1 92.9</td><td>5.36×</td><td>32.4</td><td>105.5</td><td>70.9</td><td>3.32×</td><td>34.2</td><td>43.9</td><td>48.6</td><td>8.21×</td></tr><tr><td>TeCoP + cache</td><td>80.6 80.6</td><td>76.2 62.1</td><td>128.6</td><td>5.92× 8.20×</td><td>34.0 32.4</td><td>98.1</td><td>76.4</td><td>3.58×</td><td>37.0</td><td>38.6</td><td>47.9</td><td>8.08×</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>78.7</td><td>111.0</td><td>5.20×</td><td>38.6</td><td>32.2</td><td>69.7</td><td>11.77×</td></tr><tr><td colspan="10">Dream-v0-Base-7B</td><td colspan="3"></td></tr><tr><td>Original</td><td>75.0</td><td>256.0</td><td>10.0</td><td>1.00×</td><td>34.8</td><td>256.0</td><td>18.1</td><td>1.00×</td><td>53.0</td><td>256.0</td><td>5.3</td><td>1.00×</td></tr><tr><td>Fast-dLLM</td><td>75.4</td><td>152.6</td><td>16.7</td><td>1.67×</td><td>34.4</td><td>102.7</td><td>44.6</td><td>2.46×</td><td>53.8</td><td>90.3</td><td>14.6</td><td>2.74×</td></tr><tr><td>DSB</td><td>75.4</td><td>161.4</td><td>15.5</td><td>1.56×</td><td>34.5</td><td>98.9</td><td>45.5</td><td>2.51×</td><td>53.8</td><td>93.7</td><td>13.8</td><td>2.60×</td></tr><tr><td>TeCoP</td><td>73.6</td><td>121.1</td><td>21.3</td><td>2.14×</td><td>32.9</td><td>89.5</td><td>52.4</td><td>2.89×</td><td>53.2</td><td>80.0</td><td>16.7</td><td>3.14×</td></tr><tr><td>Fast-dLLM + cache</td><td>75.9</td><td>162.3</td><td>30.9</td><td>3.10×</td><td>33.3</td><td>121.8</td><td>57.2</td><td>3.15×</td><td>53.0</td><td>111.5</td><td>17.1</td><td>3.23×</td></tr><tr><td>DSB + cache TeCoP + cache</td><td>72.2 74.1</td><td>160.1 122.4</td><td>29.2 48.9</td><td>2.93× 4.91×</td><td>33.2 32.0</td><td>113.3 95.5</td><td>58.2 90.3</td><td>3.21× 4.98×</td><td>55.0 51.8</td><td>101.6 85.8</td><td>18.0 27.1</td><td>3.38× 5.11×</td></tr><tr><td colspan="10">Dream-v0-Instruct-7B</td></tr><tr><td></td><td></td><td></td><td>12.3</td><td></td><td></td><td></td><td></td><td>1.00×</td><td>53.6</td><td>256.0</td><td></td><td></td></tr><tr><td>Original Fast-dLLM</td><td>79.7 78.8</td><td>256.0 63.2</td><td>48.5</td><td>1.00×</td><td>37.8</td><td>256.0</td><td>20.6</td><td>2.86×</td><td>55.6</td><td>34.1</td><td>5.8 39.6</td><td>1.00× 6.77×</td></tr><tr><td>DSB</td><td>79.3</td><td>81.1</td><td>37.0</td><td>3.93× 3.00×</td><td>37.7 37.7</td><td>87.1 90.3</td><td>58.9 56.2</td><td>2.73×</td><td>50.6</td><td>45.2</td><td>28.0</td><td>4.80×</td></tr><tr><td>TeCoP</td><td>77.3</td><td>59.8</td><td>52.6</td><td>4.26×</td><td>36.2</td><td>78.8</td><td>65.0</td><td>3.16×</td><td>55.4</td><td>35.0</td><td>40.8</td><td>6.98×</td></tr><tr><td>Fast-dLLM + cache</td><td>76.6</td><td>71.8</td><td>73.3</td><td>5.94×</td><td>37.7</td><td>97.9</td><td>79.2</td><td>3.85×</td><td>53.4</td><td>43.5</td><td>45.1</td><td>7.71×</td></tr><tr><td>DSB + cache</td><td>77.8</td><td>67.5</td><td>75.4</td><td>6.11×</td><td>37.8</td><td>90.2</td><td>81.5</td><td>3.96×</td><td>57.4</td><td>36.0</td><td>50.5</td><td>8.63×</td></tr><tr><td>TeCoP + cache</td><td>77.6</td><td>60.5</td><td>97.8</td><td>7.93×</td><td>35.6</td><td>79.7</td><td>115.3</td><td>5.61×</td><td>52.6</td><td>35.2</td><td>56.9</td><td>9.73×</td></tr></table>

Metrics. We report accuracy, the average number of model forward evaluations (NFE) per problem, and throughput in tokens per second (TPS). TPS counts the generated answer tokens, excluding EOS padding, per second of generation time. Speedup (Spd.) is the unrounded TPS ratio relative to Original without caching.

Implementation details. All experiments use a single NVIDIA H200 GPU, batch size 1, and generation length 256 unless noted otherwise. We report mean TPS over three runs. With caching, TeCoP adopts Fast-dLLM’s block-wise approximate prefix KV cache (Wu et al., 2025b): prompt and frozenblock KV states are cached and refreshed when a block freezes, while active and pending blocks are recomputed each forward pass. Full configurations appear in Appendix A.

## 5.1 Quantitative Results

Table 2 shows that TeCoP improves decoding efficiency across four models and three benchmarks. Without caching, it achieves 2.14×–7.50× speedups over Original, outperforming Fast-dLLM and DSB in TPS across all 12 settings (1.03×– 1.31× over Fast-dLLM), with lower NFE in 11 and 12 settings, respectively. With caching, TeCoP achieves the highest TPS and lowest NFE among cached decoders in all 12 settings, outperforming Fast-dLLM with DualCache by 1.26–1.58× despite caching only the prefix. Table 3 shows that despite task-dependent accuracy fluctuations across decoding policies, TeCoP maintains average accuracy within one percentage point of Original across all 12 cached settings, while outperforming FastdLLM in both average accuracy and speedup. Appendix A.4 confirms that efficiency gains persist at length 512.

Table 3: Average accuracy change and geometric-mean speedup over original across 4 models and 3 benchmarks with KV caching. See Appendix Table 9 for per-benchmark results.
<table><tr><td>Method</td><td>∆Acc. (pp) Spd. ↑</td><td></td></tr><tr><td>Fast-dLLM + cache</td><td>-1.02 +0.20</td><td>4.50× 4.72×</td></tr><tr><td>DSB + cache TeCoP + cache</td><td>-0.98 6.65×</td><td></td></tr></table>

Table 4: Effect of pipeline depth K (LLaDA-8B-Instruct, GSM8K). Larger K raises throughput at the cost of accuracy. Spd. is the TPS speedup over Original.
<table><tr><td>K</td><td>Acc. (%)</td><td>NFE↓</td><td>TPS ↑ Spd. ↑</td></tr><tr><td>1 (no pipelining)</td><td>78.2</td><td>72.4</td><td>61.0 3.63×</td></tr><tr><td>2</td><td>76.7</td><td>61.8</td><td>71.7 4.26×</td></tr><tr><td>3</td><td>75.6</td><td>46.8</td><td>95.1 5.66×</td></tr><tr><td>4</td><td>74.3</td><td>41.8</td><td>105.9 6.30×</td></tr></table>

## 5.2 Ablation Studies

We ablate both components of TeCoP on LLaDA-8B-Instruct with GSM8K, varying one hyperparameter at a time with the others at their defaults. Setting K = 1 removes cross-block pipelining, and any $\tau _ { d } \geq \tau$ removes tentative drafting, since every position reaching $\tau _ { d }$ is already safe-committed.

Effect of pipeline depth. Table 4 shows the effect of varying the pipeline depth K: raising K from 1 to 2 increases the speedup from 3.63× to 4.26× at a cost of 1.5 points. Larger K raises throughput further, up to 6.30× at K = 4, but with a steeper accuracy cost, so we use K = 2.

Effect of draft threshold. Table 5 shows the effect of varying the draft threshold $\tau _ { d } \colon$ lowering $\tau _ { d }$ from τ (no drafting) to 0.85 raises the speedup from 3.94× to 4.26×. Lower thresholds admit less reliable drafts, making accuracy unstable before it drops to 75.2% at $\tau _ { d } = 0 . 6 ,$ so we keep τ<sub>d</sub> = 0.85 as the default for both model families.

Table 5: Effect of draft threshold $\tau _ { d }$ (LLaDA-8B-Instruct, GSM8K). Lower $\tau _ { d }$ raises throughput, but overly low values make accuracy unstable.
<table><tr><td>Td</td><td>Acc. (%)</td><td>NFE↓</td><td>TPS ↑</td><td>Spd. ↑</td></tr><tr><td>0.6</td><td>75.2</td><td>42.1</td><td>104.3</td><td>6.20×</td></tr><tr><td>0.7</td><td>77.1</td><td>48.7</td><td>90.9</td><td>5.41×</td></tr><tr><td>0.8</td><td>76.5</td><td>57.4</td><td>77.2</td><td>4.59×</td></tr><tr><td>0.85</td><td>76.7</td><td>61.8</td><td>71.7</td><td>4.26×</td></tr><tr><td>≥ τ (no drafting)</td><td>77.4</td><td>66.8</td><td>66.2</td><td>3.94×</td></tr></table>

Table 6: Verification diagnostic on GSM8K. Random token replacements cause most drafts to be rejected.
<table><tr><td>Slot content</td><td>Positions</td><td>Admitted</td><td>Rejected</td></tr><tr><td>Genuine draft</td><td>37,091</td><td>97.31%</td><td>2.69%</td></tr><tr><td>Random token</td><td>37,091</td><td>16.46%</td><td>83.54%</td></tr></table>

## 5.3 Validating the Verification Mechanism

To test whether verify-commit simply copies tentative context rather than verifying it, we replace each GSM8K draft with a random vocabulary token in a measurement-only forward pass that does not affect the actual decoding process.

Table 6 shows that 83.54% of random drafts are rejected, versus 2.69% of genuine drafts, indicating that verification favors drafts consistent with the model’s prediction rather than merely copying the tentative input context.

## 6 Conclusion

We introduce Tentative Context Pipelining (TeCoP), a training-free decoder that decouples contextual utility from commitment readiness through cross-block pipelining and draft-andverify. By reusing stable but uncommitted predictions as provisional context, TeCoP reduces number of forward evaluations by up to 87% and achieves up to 7.50× and 11.77× speedups over uncached standard block-wise decoding without and with KV caching, respectively. Across four dLLMs and three benchmarks, TeCoP outperforms Fast-dLLM and DSB in throughput across all 12 settings with and without caching, achieving a 6.65× geometric-mean cached speedup while keeping average accuracy within one percentage point of the standard decoder.

## Limitations

TeCoP remains unevaluated on open-ended and long-form generation benchmarks, where higher output entropy may reduce draft stability and

longer sequences may increase error propagation. Future work will study broader tasks, longer generations, and cross-domain draft stability.

## References

Marianne Arriola, Aaron Gokaslan, Justin T Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. 2025. Block diffusion: Interpolating between autoregressive and diffusion language models. In Proc. Int. Conf. on Learning Representations (ICLR).

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. 2021. Program synthesis with large language models. CoRR, abs/2108.07732.

Wenrui Bao, Zhiben Chen, Dan Xu, and Yuzhang Shang. 2025. Learning to parallel: Accelerating diffusion large language models via learnable parallel decoding. CoRR, abs/2509.25188.

Stella Biderman, Hailey Schoelkopf, Lintang Sutawika, Leo Gao, Jonathan Tow, Baber Abbasi, Alham Fikri Aji, Pawan Sasanka Ammanamanchi, Sidney Black, Jordan Clive, Anthony DiPofi, Julen Etxaniz, Benjamin Fattori, Jessica Zosa Forde, Charles Foster, Jeffrey Hsu, Mimansa Jaiswal, Wilson Y. Lee, Haonan Li, and 11 others. 2024. Lessons from the trenches on reproducible evaluation of language models.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. 2023. Accelerating large language model decoding with speculative sampling. CoRR, abs/2302.01318.

Jian Chen, Yesheng Liang, and Zhijian Liu. 2026. DFlash: Block diffusion for flash speculative decoding. CoRR, abs/2602.06036.

Kecheng Chen, Ziru Liu, Xijia Tao, Hui Liu, Xinyu Fu, Suiyun Zhang, Dandan Tu, Lingpeng Kong, Rui Liu, and Haoliang Li. 2025a. Beyond confidence: Adaptive and coherent decoding for diffusion language models. CoRR, abs/2512.02044.

Zigeng Chen, Gongfan Fang, Xinyin Ma, Ruonan Yu, and Xinchao Wang. 2025b. dParallel: Learnable parallel decoding for dLLMs. CoRR, abs/2509.26488.

Jacob K Christopher, Brian R Bartoldson, Tal Ben-Nun, Michael Cardei, Bhavya Kailkhura, and Ferdinando Fioretto. 2025. Speculative diffusion decoding: Accelerating language generation through diffusion. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL), pages 12042–12059. Association for Computational Linguistics.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. CoRR, abs/2110.14168.

Junxia Cui, Haotian Ye, Runchu Tian, Hongcan Guo, Jinya Jiang, Haoru Li, Chaojie Ren, Yiming Huang, Kaijie Zhu, Zhongkai Yu, Kun Zhou, and Jingbo Shang. 2026. SimSD: Simple speculative decoding in diffusion language models. CoRR, abs/2606.02544.

Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. 2024. Break the sequential dependency of LLM inference using lookahead decoding. In Proc. Int. Conf. on Machine Learning (ICML), pages 14060– 14079.

Yifeng Gao, Ziang Ji, Yuxuan Wang, Biqing Qi, Hanlin Xu, and Linfeng Zhang. 2025. Self speculative decoding for diffusion large language models. CoRR, abs/2510.04147.

Shansan Gong, Ruixiang Zhang, Huangjie Zheng, Jiatao Gu, Navdeep Jaitly, Lingpeng Kong, and Yizhe Zhang. 2025. DiffuCoder: Understanding and improving masked diffusion models for code generation. CoRR, abs/2506.20639.

Google DeepMind. 2025. Gemini diffusion. Technical report / model page.

Zhengfu He, Tianxiang Sun, Qiong Tang, Kuanning Wang, Xuanjing Huang, and Xipeng Qiu. 2023. DiffusionBERT: Improving generative masked language models with diffusion models. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics, pages 4521–4534. Association for Computational Linguistics.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the MATH dataset. In Proc. Conf. on Neural Information Processing Systems (NeurIPS).

Michael Hersche, Samuel Moor-Smith, Thomas Hofmann, and Abbas Rahimi. 2025. Soft-masked diffusion language models. CoRR, abs/2510.17206.

Feng Hong, Geng Yu, Yushi Ye, Haicheng Huang, Huangjie Zheng, Ya Zhang, Yanfeng Wang, and Jiangchao Yao. 2026. Wide-in, narrow-out: Revokable decoding for efficient and effective DLLMs. In Proc. Int. Conf. on Learning Representations (ICLR).

Daniel Mingyi Israel, Guy Van den Broeck, and Aditya Grover. 2025. Accelerating diffusion LLMs via adaptive parallel decoding. In Proc. Conf. on Neural Information Processing Systems (NeurIPS), volume 38.

Yuchu Jiang, Yue Cai, Xiangzhong Luo, Jiale Fu, Jiarui Wang, Chonghan Liu, and Xu Yang. 2025. d<sup>2</sup>Cache: Accelerating diffusion-based LLMs via dual adaptive caching. CoRR, abs/2509.23094.

Samar Khanna, Siddhant Kharbanda, Shufan Li, Harshit Varma, Eric Wang, Sawyer Birnbaum, Ziyang Luo, Yanis Miraoui, Akash Palrecha, Stefano Ermon, Aditya Grover, and Volodymyr Kuleshov. 2025. Mercury: Ultra-fast language models based on diffusion. CoRR, abs/2506.17298.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast inference from transformers via speculative decoding. In Proc. Int. Conf. on Machine Learning (ICML), pages 19274–19286.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. 2022. Solving quantitative reasoning problems with language models. In Proc. Conf. on Neural Information Processing Systems (NeurIPS).

Guanghao Li, Zhihui Fu, Min Fang, Qibin Zhao, Ming Tang, Chun Yuan, and Jun Wang. 2025a. DiffuSpec: Unlocking diffusion language models for speculative decoding. CoRR, abs/2510.02358.

Pengxiang Li, Yefan Zhou, Dilxat Muhtar, Lu Yin, Shilin Yan, Li Shen, Soroush Vosoughi, and Shiwei Liu. 2025b. Diffusion language models know the answer before decoding. CoRR, abs/2508.19982.

Aiwei Liu, Minghua He, Shaoxun Zeng, Sijun Zhang, Linhao Zhang, Chuhan Wu, Wei Jia, Yuan Liu, Xiao Zhou, and Jie Zhou. 2025a. WeDLM: Reconciling diffusion language models with standard causal attention for fast inference. CoRR, abs/2512.22737.

Zhiyuan Liu, Yicun Yang, Yaojie Zhang, Junjie Chen, Chang Zou, Qingyan Wei, Shaobo Wang, Yichen Zhu, and Linfeng Zhang. 2025b. dLLM-Cache: Accelerating diffusion large language models with adaptive caching. CoRR, abs/2506.06295.

Lizhuo Luo, Shenggui Li, Yonggang Wen, and Tianwei Zhang. 2026a. DSB: Dynamic sliding block scheduling for diffusion LLMs. CoRR, abs/2602.05992.

Lizhuo Luo, Zhuoran Shi, Jiajun Luo, Zhi Wang, Shen Ren, Wenya Wang, and Tianwei Zhang. 2026b. DAWN: Dependency-aware fast inference for diffusion LLMs. CoRR, abs/2602.06953.

Xinyin Ma, Runpeng Yu, Gongfan Fang, and Xinchao Wang. 2025. dKV-Cache: The cache for diffusion language models. In Proc. Conf. on Neural Information Processing Systems (NeurIPS).

Amr Mohamed, Yang Zhang, Michalis Vazirgiannis, and Guokan Shang. 2025. Fast-decoding diffusion language models via progress-aware confidence schedules. CoRR, abs/2512.02892.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. 2026. Large language diffusion models. Proc. Conf. on Neural Information Processing Systems (NeurIPS), 38:50608–50646.

Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander M Rush, and Volodymyr Kuleshov. 2024. Simple and effective masked diffusion language models. In Proc. Conf. on Neural Information Processing Systems (NeurIPS), volume 37, pages 130136– 130184.

Andrea Santilli, Silvio Severino, Emilian Postolache, Valentino Maiorca, Michele Mancusi, Riccardo Marin, and Emanuele Rodolà. 2023. Accelerating transformer inference for translation via parallel decoding. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, pages 12336–12355. Association for Computational Linguistics.

Yingte Shu, Yuchuan Tian, Chao Xu, Yunhe Wang, and Hanting Chen. 2026. Deferred commitment decoding for diffusion language models. CoRR, abs/2601.02076.

Yuxuan Song, Zheng Zhang, Cheng Luo, Pengyang Gao, Fan Xia, Hao Luo, Zheng Li, Yuehang Yang, Hongli Yu, Xingwei Qu, Yuwei Fu, Jing Su, Ge Zhang, Wenhao Huang, Mingxuan Wang, Lin Yan, Xiaoying Jia, Jingjing Liu, Wei-Ying Ma, and 3 others. 2025. Seed diffusion: A large-scale diffusion language model with high-speed inference. CoRR, abs/2508.02193.

Kangyu Wang, Zhiyun Jiang, Haibo Feng, Weijia Zhao, Lin Liu, Jianguo Li, Zhenzhong Lan, and Weiyao Lin. 2025a. CreditDecoding: Accelerating parallel decoding in diffusion large language models with trace credits. CoRR, abs/2510.06133.

Xu Wang, Chenkai Xu, Yijie Jin, Jiachun Jin, Hao Zhang, and Zhijie Deng. 2025b. Diffusion LLMs can do faster-than-ar inference via discrete diffusion forcing. CoRR, abs/2508.09192.

Chengyue Wu, Hao Zhang, Shuchen Xue, Shizhe Diao, Yonggan Fu, Zhijian Liu, Pavlo Molchanov, Ping Luo, Song Han, and Enze Xie. 2025a. FastdLLM v2: Efficient block-diffusion LLM. CoRR, abs/2509.26328.

Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. 2025b. Fast-dLLM: Training-free acceleration of diffusion LLM by enabling KV cache and parallel decoding. CoRR, abs/2505.22618.

Shutong Wu and Jiawei Zhang. 2025. Free draftand-verification: Toward lossless parallel decoding for diffusion large language models. CoRR, abs/2510.00294.

Zhihui Xie, Jiacheng Ye, Lin Zheng, Jiahui Gao, Jingwei Dong, Zirui Wu, Xueliang Zhao, Shansan Gong,

Xin Jiang, Zhenguo Li, and Lingpeng Kong. 2025. Dream-Coder 7B: An open diffusion language model for code. CoRR, abs/2509.01142.

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. 2025. Dream 7B: Diffusion large language models. CoRR, abs/2508.15487.

Haohui Zhang, Zhiye Wang, Xiaoying Gan, Xinbing Wang, and Bo Jiang. 2026. LEAP: Unlocking dLLM parallelism via lookahead early-convergence token detection. CoRR, abs/2605.10980.

Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. 2026. LLaDA 1.5: Variancereduced preference optimization for large language diffusion models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11425– 11460. Association for Computational Linguistics.

## A Experimental Details and Additional Results

This appendix provides additional experimental details and analyses. Unless otherwise specified, experiments use a generation length of 256 and batch size 1. For TeCoP, one configuration is shared within each model family and fixed across GSM8K, Minerva MATH, and MBPP. All benchmarks are in English and cover mathematical reasoning or Python code generation.

Implementation and licenses. Our implementation builds on existing open-source implementations of LLaDA, Dream, and Fast-dLLM under their respective licenses, with all required notices retained. Model weights and benchmark datasets are not redistributed.

## A.1 Hyperparameter Notation

Table 7 summarizes the decoding parameters used throughout the experiments.

Table 7: Notation of decoding parameters used in TeCoP.
<table><tr><td>Parameter</td><td>Meaning</td></tr><tr><td>b</td><td>Block size.</td></tr><tr><td>K</td><td>Max number of concurrent active blocks.</td></tr><tr><td>α</td><td>Threshold for opening the next block.</td></tr><tr><td>T</td><td>Commit threshold for safe-commit.</td></tr><tr><td> $\tau _ { d }$ </td><td>Threshold for admitting tentative drafts.</td></tr><tr><td> $\tau _ { v }$ </td><td>Threshold for committing stable tentative drafts.</td></tr></table>

## A.2 Hyperparameter Settings

Table 8 reports the decoding configurations used for TeCoP. We use one shared configuration for the two LLaDA models and another shared configuration for the two Dream models, reflecting differences in their confidence distributions and decoding behavior. Importantly, these hyperparameters are not tuned separately for each benchmark: once selected, the same model-family-level configuration is reused across GSM8K, Minerva MATH, and MBPP.

Table 8: Model-family-specific decoding configuration for TeCoP.
<table><tr><td>Model</td><td>b</td><td>K</td><td>α</td><td>T</td><td>Td</td><td>Tv</td></tr><tr><td>LLaDA-8B-Instruct</td><td>32</td><td>2</td><td>0.80</td><td>0.90</td><td>0.85</td><td>0.80</td></tr><tr><td>LLaDA-1.5</td><td>32</td><td>2</td><td>0.80</td><td>0.90</td><td>0.85</td><td>0.80</td></tr><tr><td>Dream-v0-Base-7B</td><td>16</td><td>2</td><td>0.85</td><td>0.90</td><td>0.85</td><td>0.70</td></tr><tr><td>Dream-v0-Instruct-7B</td><td>16</td><td>2</td><td>0.85</td><td>0.90</td><td>0.85</td><td>0.70</td></tr></table>

Baseline and caching configurations. All methods use greedy decoding. Original uses block size 32 and commits one position per forward pass, with low-confidence remasking for LLaDA and entropybased ordering for Dream. Fast-dLLM and DSB use a confidence threshold of 0.9 and an initial block size of 32. In the cached setting, Fast-dLLM uses DualCache, DSB uses its official cache with a prefix window of 24 for LLaDA and 4 for Dream, and TeCoP uses the block-wise prefix KV cache of Fast-dLLM, refreshed whenever a block is frozen.

## A.3 Average Metric Per Dataset

Table 9 shows the per-benchmark average accuracy change and geometric-mean speedup relative to Original, over 4 models with caching enabled.

Table 9: Average accuracy change and geometric-mean speedup relative to Original.
<table><tr><td>Method</td><td>∆Acc. (%) Spd. ↑</td></tr><tr><td>GSM8K</td><td></td></tr><tr><td>Fast-dLLM + cache DSB + cache</td><td>-0.35 4.80× -0.70 5.02×</td></tr><tr><td>TeCoP + cache -0.35</td><td>7.08×</td></tr><tr><td>Minerva MATH</td><td></td></tr><tr><td>Fast-dLLM + cache DSB + cache</td><td>-0.80 3.40× -0.20 3.59×</td></tr><tr><td>TeCoP + cache</td><td>-1.60 5.22×</td></tr><tr><td>MBPP</td><td></td></tr><tr><td>Fast-dLLM + cache DSB + cache</td><td>-1.90 5.57× +1.50 5.84×</td></tr><tr><td>TeCoP + cache</td><td>-1.00 7.96×</td></tr></table>

## A.4 Longer Generation Length

We further evaluate decoding with a generation length of 512 on GSM8K, using the same TeCoP configurations as in the main experiments without retuning. KV caching remains disabled and batch size is 1. As shown in Table 10, TeCoP retains substantial efficiency gains at the longer generation length, achieving speedups of 7.18× and 8.44× over Original on LLaDA-8B-Instruct and LLaDA-1.5, respectively. Both exceed their corresponding speedups at length 256, and TeCoP achieves the highest throughput among all compared methods on both models, demonstrating that its efficiency gains extend to longer generation lengths.

## A.5 Peak GPU Memory

We additionally measure peak allocated GPU memory on 100 GSM8K samples with batch size 1,

Table 10: Results on GSM8K with generation length 512, without KV caching. TeCoP uses the length-256 configurations without retuning. Spd. is the TPS speedup over Original, computed from unrounded TPS.
<table><tr><td>Method Acc. (%)</td><td colspan="3">NFE↓ TPS ↑ Spd. ↑</td></tr><tr><td></td><td colspan="3">LLaDA-8B-Instruct</td></tr><tr><td>Original</td><td>77.6</td><td>512.0</td><td>8.0 1.00×</td></tr><tr><td>Fast-dLLM</td><td>76.9</td><td>97.0 44.1</td><td>5.50×</td></tr><tr><td>DSB</td><td>77.7</td><td>97.8 43.2</td><td>5.39×</td></tr><tr><td>TeCoP</td><td>76.2</td><td>75.6</td><td>57.5 7.18×</td></tr><tr><td></td><td colspan="3">LLaDA-1.5</td></tr><tr><td>Original</td><td>81.6</td><td>512.0</td><td>6.1 1.00×</td></tr><tr><td>Fast-dLLM</td><td>83.0</td><td>80.3 40.0</td><td>6.57×</td></tr><tr><td>DSB</td><td>82.8</td><td>80.8</td><td>39.1 6.43×</td></tr><tr><td>TeCoP</td><td>82.5</td><td>64.3</td><td>51.4 8.44×</td></tr></table>

generation length 256, and KV caching disabled.

Table 11: Peak allocated GPU memory (GiB) on GSM8K without KV caching. Original and Fast-dLLM have identical peak memory in this setting.
<table><tr><td>Model</td><td>Original / Fast-dLLM</td><td>TeCoP</td></tr><tr><td>LLaDA-8B-Instruct</td><td>18.53</td><td>16.17</td></tr><tr><td>LLaDA-1.5</td><td>18.51</td><td>16.16</td></tr><tr><td>Dream-v0-Base-7B</td><td>15.21</td><td>16.13</td></tr><tr><td>Dream-v0-Instruct-7B</td><td>15.22</td><td>16.15</td></tr></table>

As shown in Table 11, TeCoP introduces no substantial additional peak-memory overhead. All configurations remain below 19 GiB: TeCoP uses less peak memory on LLaDA and incurs less than 1 GiB additional memory on Dream.