# Replay the Curvature: Accurate and Scalable NVFP4 Quantization for Large Language Model Inference

Ruiyi Ding<sup>1\*</sup>, Jie Li<sup>1</sup>, Kang He<sup>1</sup>, Ziyan Liu<sup>5</sup>, Chengru Song<sup>1†</sup>, Yuedong Xu<sup>2,3†</sup>, Yuan Cheng<sup>2,3,4†</sup>

<sup>1</sup>KlingAI Research

<sup>2</sup>Fudan University <sup>3</sup>Shanghai AI Incubation and Innovation Center

<sup>4</sup>Shanghai Academy of AI for Science <sup>5</sup>University of Science and Technology of China

<sup>†</sup>Corresponding authors

dingruiyi@klingai.com, songchengru@klingai.com, cheng yuan@fudan.edu.cn

## Abstract

Large language models make weight storage and memory traffic major inference costs, motivating low-precision formats that represent each weight with only a few bits. Such formats use a scale to map floating-point values into a small codebook; NVFP4 improves local range utilization by letting every 16 E2M1 weights share an E4M3 block scale. Choosing that scale is difficult in GPTQ because quantizing one column updates those that follow, so evaluating a block independently can misestimate its final reconstruction error. Large models pose a second challenge: full-precision weights, calibration activations, and second-order state cannot all remain on one accelerator, while assigning complete layers to devices leaves each time-consuming layer solve serial. We introduce Schur Replay, a scale-selection algorithm that reproduces the GPTQ updates caused by each block scale and scores the resulting block error after accounting for compensation from unquantized columns. Separately, our execution infrastructure keeps only the active layer resident, tiers activations across device, host, and disk, retires full-precision layers after export, and distributes independent output rows across tensor-parallel ranks. Together, the algorithm and infrastructure attain 99.35% and 100.84% question-weighted recovery from BF16 across seven benchmarks on Qwen3.5-397B-A17B and Llama-3.3-70B-Instruct. On the 397B model, the infrastructure reduces measured per-layer time by 15.17× over ModelOpt and 23.14× over LLM Compressor, with lower memory used per GPU.

## 1 Introduction

Modern large language models contain tens or hundreds of billions of parameters, making weight storage and memory traffic major costs during inference. Low-precision formats reduce the bits used by each weight, allowing larger models to fit in accelerator memory and reducing the data moved for matrix multiplication. To map a floating-point weight w into a small set of representable values, quantization first chooses a scale s, rounds the normalized value w/s to the nearest low-precision code, and approximately reconstructs the weight as the code times s. A common initialization sets s from the largest absolute weight so that the codebook covers the observed range. With one scale for an entire tensor, however, a few large values can leave most weights using only a small fraction of that codebook. Microscaling addresses this mismatch by assigning local scales to small blocks. NVFP4 adopts this design: a tensor-wide FP32 scale sets the global range, while every 16 E2M1 weights share an E4M3 block scale, yielding approximately 4.5 bits per value [Rouhani et al., 2023, NVIDIA et al., 2025]. This layout is efficient on modern accelerators, but makes the 16 quantization decisions interdependent: one poorly chosen block scale can distort the whole block. Post training quantization (PTQ) must choose these values from a small calibration set without retraining, and GPTQ uses activation-derived second-order information to compensate each rounding error [Frantar et al., 2023].

The difficulty is that GPTQ is sequential. After quantizing one column, it updates the columns that follow to compensate for the incurred error. Two block scales that initially look similar can therefore create different later states and different final reconstruction errors. Existing NVFP4 methods improve the representation or its initialization: Four Over Six adapts the effective E2M1 range; SOAR and ScaleSweep optimize or search scales; MR-GPTQ rotates local blocks; and ARCQuant adds residual channels [Cook et al., 2026, Bao et al., 2026, Lin and Wan, 2026, Egiazarian et al., 2026, Meng et al., 2026]. Our focus is complementary: evaluate each exportable scale in the GPTQ state that it actually creates, rather than only on the original block.

Scaling this reconstruction to hundreds of billions of parameters creates two separate execution bottlenecks. First, fullprecision weights, calibration activations, and second-order state cannot all remain in accelerator memory. Second, distributing samples, experts, or complete modules does not accelerate the reconstruction of one exceptionally large layer. Public toolchains support layer-wise loading, disk offloading, and multi-device execution [NVIDIA Corporation, 2024, Red Hat AI and vLLM Project, 2024, Gong et al., 2024, ModelCloud.ai, 2024], but their documented designs neither realize paralleled solve of each weight matrix across tp ranks, nor make use of a strict active-layer residency and a layer-scoped activation lifecycle.

We address the algorithmic bottleneck with Schur Replay, a group-scale selection algorithm. For each hardwarevalid scale, it performs the GPTQ updates that the candidate would trigger, then scores the resulting block error after accounting for the compensation still available from unquantized columns. We also formalizes this algorithm through recurrence-exact replay and a conditional Schur objective. We address the execution bottleneck with a infrastructure framework: because different output rows follow independent GPTQ recurrences, it assigns rows of the same layer to different ranks, keeps only the active weights on device, tiers activations across device, host, and disk, and retires full-precision layers after recoverable NVFP4 export.

Our main contributions are:

Recurrence-aware NVFP4 reconstruction. Schur Replay evaluates each exportable scale in the sequential GPTQ state that it creates, capturing feedback within the 16-value block.

Memory-bounded distributed execution. Active-layer residency and tiered activation storage bound device memory, while output-row parallelism engages all tensor-parallel ranks in the current layer.

Large-model evidence. Combining Schur Replay with our execution infrastructure recovers 99.35%/100.84% of BF16 across seven benchmarks on Qwen3.5-397B-A17B/Llama-3.3-70B-Instruct. On 397B, the infrastructure makes measured per-layer compute 15.17×/23.14× faster than ModelOpt/LLM Compressor at 35.0 GB peak memory per GPU; full-model totals remain projections.

## 2 Related Work

Quantization methods. PTQ converts models without retraining: AdaRound learns rounding [Nagel et al., 2020]; OBC/OBQ and GPTQ use second-order compensation [Frantar et al., 2022, 2023]; SmoothQuant, AWQ, and Omni-Quant calibrate distributions [Xiao et al., 2023, Lin et al., 2024, Shao et al., 2024]; and QuIP# and SpinQuant apply incoherence processing or rotations [Tseng et al., 2024, Liu et al., 2025]. QAT instead adapts parameters to simulated quantization noise but requires more data, compute, and memory [Bengio et al., 2013, Liu et al., 2023, Chen et al., 2025, Bondarenko et al., 2024].

NVFP4. Four Over Six adapts the E2M1 range [NVIDIA et al., 2025]; SOAR and ScaleSweep optimize or search scales [Cook et al., 2026, Bao et al., 2026]; MR-GPTQ adds micro-rotations and reordering; and ARCQuant adds residual channels [Rouhani et al., 2023, Lin and Wan, 2026, Egiazarian et al., 2026, Meng et al., 2026]. Native pretraining and quantization-aware distillation are training-based alternatives [NVIDIA et al., 2025, Xin et al., 2026]. These approaches change initialization or representation; Schur Replay instead scores each scale in its induced GPTQ recurrence.

Infrastructure. Existing stacks provide calibration, offload, framework integration, and deployment support [NVIDIA Corporation, 2024, Red Hat AI and vLLM Project, 2024, Gong et al., 2024, Advanced Micro Devices, Inc., 2024, Intel Corporation, 2021, Or et al., 2025, Microsoft Corporation, 2023, ModelCloud.ai, 2024]. LLM Compressor’s documented distributed GPTQ assigns each module to one owner rank; we found no public design fullyoutilize the layerwise and parallel feature for efficient PTQ solving.

## 3 Preliminaries

Layer reconstruction. For a linear layer $W \in \mathbb { R } ^ { M \times N }$ and calibration inputs $X \in \mathbb { R } ^ { N \times T }$ , where M, N, and T denote the output width, input width, and number of calibration positions, reconstruction-based PTQ solves

$$
\widehat { W } = \mathop { \arg \operatorname* { m i n } } _ { W _ { q } \in \mathcal { Q } } \| W X - W _ { q } X \| _ { F } ^ { 2 } .\tag{1}
$$

The Q is the set of weight matrices representable by the target quantization format. Writing $\Delta = W _ { q } - W , H = X X ^ { \top }$ and $\delta _ { m } = \Delta _ { m , : } ^ { \top } \in \mathbb { R } ^ { N }$ we obtain

$$
\| \Delta X \| _ { F } ^ { 2 } = \mathrm { t r } ( \Delta H \Delta ^ { \top } ) = \sum _ { m = 1 } ^ { M } \delta _ { m } ^ { \top } H \delta _ { m } .\tag{2}
$$

Thus all output rows share one input Hessian but contribute independent quadratic objectives. OBS and OBC/OBQ exploit this structure by compensating each forced quantization error through the remaining unconstrained coordinates [Hassibi et al., 1993, Frantar et al., 2022].

GPTQ recurrence. GPTQ scales this compensation to transformer layers by quantizing every output row in a common input-column order and reusing one Hessian factor [Frantar et al., 2023]. Let π be the chosen column permutation (the identity unless activation ordering is enabled). Applying π to the columns of W and the corresponding rows of X preserves $\dot { H } = X X ^ { \top }$ in the permuted coordinate system; henceforth W, X, H, and all column indices refer to that order. After damping and handling structurally zero columns, let

$$
H ^ { - 1 } = U ^ { \top } U ,\tag{3}
$$

where $U$ is upper triangular and $u _ { i i } = U _ { i i }$ . For a working row w in this order and column i, GPTQ quantizes the current compensated value to $q _ { i }$ and updates

$$
e _ { i } = \frac { w _ { i } - q _ { i } } { u _ { i i } } , \qquad w _ { i : }  w _ { i : } - e _ { i } U _ { i , i : } .\tag{4}
$$

The recurrence is sequential across input columns because each error changes later values, whereas output rows remain independent. NVFP4 shares one scale across consecutive columns, so selecting that scale from the group’s initial weights ignores the values produced inside the recurrence. Section 4.1 derives a conditional group cost and evaluates scales by recurrence-exact intra-group replay; Section 4.2.4 exploits output-row independence across tensor-parallel ranks.

## 4 Method

Our approach consists of two complementary components. The first is Schur Replay, which improves the PTQ objective and group-scale selection algorithm. The second is a separate scalable execution infrastructure that preserves the numerical accuracy of the underlying quantizer.

## 4.1 Better PTQ Algorithm

We introduce Schur Replay, a group-scale selection method for microscaled PTQ. Its central distinction from independent block reconstruction is that each candidate scale is evaluated under the conditional quadratic metric induced by the committed prefix and the compensation still available in the unquantized future. A closed-form update locates a promising region of the hardware-valid scale grid, after which recurrence-exact replay inside the current group selects the scale that is actually committed.

## 4.1.1 NVFP4 Group Quantization

We use the notation and GPTQ recurrence of Section 3, with Hessia matrix $H = X X ^ { \top }$ and $H ^ { - 1 } = U ^ { \top } U$ . Because Equation 4 is sequential across input columns, quantizing column i changes the values subsequently presented to columns $i + 1 , \ldots , N ;$ scale selection must therefore account for the state induced inside a shared-scale group.

For NVFP4, each group G contains $B = 1 6$ consecutive input-column indices and shares a positive scale $s .$ For a single output row, let $w _ { 0 } \in \mathbb { R } ^ { B }$ denote the compensated group values upon entry. With $\rho : \mathbb { R }  \Lambda$ denoting elementwise round-to-nearest onto the E2M1 levels $\Lambda = \{ 0 , \pm 0 . 5 , \pm 1 , \pm 1 . 5 , \pm 2 , \pm 3 , \pm 4 , \pm 6 \}$ , the group quantizer is

$$
Q _ { s } ( w ) = s \rho ( w / s ) .\tag{5}
$$

The effective multiplier s must be representable by the E4M3 block scale, and the largest reconstructed magnitude is 6s. If $\gamma > 0$ converts a stored positive E4M3 code to its effective multiplier, then

$$
S = \{ a / \gamma : a \in { \mathrm { E 4 M 3 } } ^ { + } \} .\tag{6}
$$

Searching directly on $s$ ensures that deployment-time scale alignment cannot alter the selected solution.

## 4.1.2 Conditional Group Metric

At the entrance to a group, partition the GPTQ-ordered column indices into the committed prefix weight $P ,$ current group weight $G$ of 16 columns, and unquantized future weight $F ,$ and write ${ \mathcal { R } } = G \cup F$ for the trailing indices. The prefix is fixed, and its optimal affine compensation has already been propagated into $w _ { 0 }$ . For candidate scale $s ,$ recurrence replay (defined below) produces $q ( s ) \in \mathbb { R } ^ { B }$ ; define the actual group quantization error as $\delta _ { G } ( s ) = $ $q ( s ) - w _ { 0 }$ . The future weight remains free to compensate this error; writing $\delta _ { F } \in \mathbb { R } ^ { | F | }$ for an arbitrary adjustment on the future coordinates, the candidate’s marginal reconstruction cost is

$$
\begin{array} { r } { \begin{array} { r l } { \Delta f _ { G } ( s ) = \underset { \delta _ { F } } { \operatorname* { m i n } } \left[ \begin{array} { c } { \delta _ { G } ( s ) } \\ { \delta _ { F } } \end{array} \right] ^ { \top } \left[ \begin{array} { c c } { H _ { G G } } & { H _ { G F } } \\ { H _ { F G } } & { H _ { F F } } \end{array} \right] \left[ \begin{array} { c } { \delta _ { G } ( s ) } \\ { \delta _ { F } } \end{array} \right] } & { } \\ { = \delta _ { G } ( s ) ^ { \top } S _ { G | P } \delta _ { G } ( s ) , } & { } \\ { S _ { G | P } = H _ { G G } - H _ { G F } H _ { F F } ^ { - 1 } H _ { F G } . } \end{array} } \end{array}\tag{7}
$$

Here $Z _ { A B }$ denotes a submatrix of $Z ,$ , and we abbreviate $S _ { G | P }$ as $S _ { G }$ . The displayed blocks are the trailing curvature after the committed prefix has been fixed and its affine correction absorbed into $w _ { 0 }$ . With $\Omega \ : = \ : H ^ { - 1 } \ : = \ : U ^ { \top } U$ removing the prefix from the inverse system and then eliminating the free future gives

$$
H _ { \mathcal { R R } } ^ { - 1 } = \Omega _ { \mathcal { R R } } - \Omega _ { \mathcal { R P } } \Omega _ { P P } ^ { - 1 } \Omega _ { P \mathcal { R } } = U _ { \mathcal { R R } } ^ { \top } U _ { \mathcal { R R } } ; \qquad S _ { G | P } ^ { - 1 } = \big ( H _ { \mathcal { R R } } ^ { - 1 } \big ) _ { G G } = U _ { G G } ^ { \top } U _ { G G } .\tag{8}
$$

The second identity eliminates the free future weight $F .$ . Equivalently, $S _ { G | P }$ is the conditional curvature seen by G if it is left until last among the trailing variables.

Appendix D.1 proves both identities. The reason for using the Schur complement is that scale selection should measure the loss that remains after the still-unquantized coordinates respond optimally, rather than the isolated error of the current block. Directly scoring $\| \delta _ { G } ( s ) \| _ { 2 } ^ { 2 }$ or $\delta _ { G } ( s ) ^ { \top } H _ { G G } \delta _ { G } ( s )$ implicitly sets $\delta _ { F } ~ = ~ 0$ and therefore charges the candidate for error components that later GPTQ updates can compensate. Minimizing the trailing quadratic instead gives the optimal response $\delta _ { F } ^ { \star } = - H _ { F F } ^ { - 1 } H _ { F G } \delta _ { G } ( s ) ;$ ; substituting it back yields

$$
\Delta f _ { G } ( s ) = \delta _ { G } ( s ) ^ { \top } \big ( H _ { G G } - H _ { G F } H _ { F F } ^ { - 1 } H _ { F G } \big ) \delta _ { G } ( s ) = \delta _ { G } ( s ) ^ { \top } S _ { G | P } \delta _ { G } ( s ) .\tag{9}
$$

Thus the Schur score is exactly the marginal reconstruction error made irreversible after finished quantization of current block, after the quantized prefix is fixed and the future is allowed its best continuous compensation. It accounts jointly for the realized 16-column error and conditional cross-column curvature. This interpretation also clarifies the surrogate boundary: the future is optimized continuously when candidates are ranked, whereas later blocks will eventually be quantized. We therefore use the score to compare candidate commitments at the same GPTQ state, not as a claim that it equals the final discrete layer loss. For one column, $U _ { G G } = [ u _ { i i } ]$ and $S _ { G | P } = 1 / u _ { i i } ^ { 2 }$ , so the score reduces to $\delta _ { i } ^ { 2 } / u _ { i i } ^ { 2 } \stackrel { = } { = } e _ { i } ^ { 2 }$ , the scalar OBQ/GPTQ marginal cost [Frantar et al., 2022, 2023]. The Schur quadratic is therefore its block generalization under the same future-compensation principle.

## 4.1.3 Scale Proposal by Conditional Quadratic Minimization

The discrete E2M1 codes depend on $s ,$ but if a code vector $Q \in \Lambda ^ { B }$ is held fixed, minimizing the block marginal cost in Equation 7 along the scale coordinate is a one-dimensional quadratic problem:

$$
\phi ( s ) = ( w _ { 0 } - s Q ) ^ { \top } S _ { G } ( w _ { 0 } - s Q ) , \qquad \left| s ^ { * } ( Q ) = \frac { Q ^ { \top } S _ { G } w _ { 0 } } { Q ^ { \top } S _ { G } Q } \right| .\tag{10}
$$

This update applies the same quadratic-minimization principle underlying OBS-family weight compensation [Hassibi et al., 1993, Frantar et al., 2022], but along a shared scale coordinate rather than an individual weight coordinate. Because $Q = \rho ( w _ { 0 } / s )$ , we resolve the scale–code dependence with $J = 3$ fixed-point iterations initialized by the scale $s _ { \mathrm { o l d } }$ currently assigned to the group,

$$
{ \boldsymbol { s } } ^ { ( t + 1 ) } = \frac { \rho (  { \boldsymbol { w } } _ { 0 } / s ^ { ( t ) } ) ^ { \top }  { \boldsymbol { S } } _ { G }  { \boldsymbol { w } } _ { 0 } } { \rho (  { \boldsymbol { w } } _ { 0 } / s ^ { ( t ) } ) ^ { \top }  { \boldsymbol { S } } _ { G } \rho (  { \boldsymbol { w } } _ { 0 } / s ^ { ( t ) } ) } , \qquad { \boldsymbol { s } } ^ { ( 0 ) } =  { \boldsymbol { s } } _ { \mathrm { o l d } } , \quad t = 0 , \dots , J - 1 .\tag{11}
$$

We retain $s ^ { ( t ) }$ when the denominator is numerically zero and clamp a valid update to positive values. This fixed-code step is deliberately a proposal rather than the final decision. Its quadratic update is inexpensive and curvature-aware, but changing s can change the E2M1 codes, and GPTQ compensation can then change the values rounded later in the group. Committing the proposal directly would therefore optimize an internally inconsistent snapshot. We instead use it only to center a small neighborhood on the exportable grid; the next stage evaluates that neighborhood under the actual sequential recurrence.

## 4.1.4 Recurrence-Exact Replay on the Exportable Grid

To evaluate a scale without the fixed-code approximation, we replay the actual GPTQ recurrence inside $G .$ Let $d _ { j } = ( U _ { G G } ) _ { j j }$ , let w $\boldsymbol { \jmath } _ { k } ^ { ( j ) }$ denote the working value at group coordinate k after replaying columns $1 , \ldots , j$ , and initialize $\dot { w } ^ { ( 0 ) } = w _ { 0 }$ . For candidate $s ,$

$$
\begin{array} { r l r } & { } & { q _ { j } ( s ) = s \rho \Bigl ( w _ { j } ^ { ( j - 1 ) } / s \Bigr ) , \qquad e _ { j } ( s ) = \frac { w _ { j } ^ { ( j - 1 ) } - q _ { j } ( s ) } { d _ { j } } , } \\ & { } & { w _ { k } ^ { ( j ) } = w _ { k } ^ { ( j - 1 ) } - e _ { j } ( s ) ( U _ { G G } ) _ { j k } , \qquad k = j + 1 , \dots , B . } \end{array}\tag{12}
$$

The vector $q ( s ) = ( q _ { 1 } ( s ) , \ldots , q _ { B } ( s ) ) ^ { \top }$ collects the replayed quantized values. Its actual error, sign-reversed residual, and conditional marginal score are

$$
\delta _ { G } ( s ) = q ( s ) - w _ { 0 } , \qquad r ( s ) = - \delta _ { G } ( s ) , \qquad L ( s ) = \delta _ { G } ( s ) ^ { \top } S _ { G } \delta _ { G } ( s ) = r ( s ) ^ { \top } S _ { G } r ( s ) .\tag{13}
$$

Every candidate starts from the same block-entry state $w _ { 0 }$ and uses the commit-time E2M1 rounding implementation. This common initialization is essential: otherwise candidates evaluated later would inherit corrections produced by earlier candidates and would no longer be comparable. Replay maintains a private working state for each candidate, reproduces all intra-group dependencies, and returns both the codes that would be committed and their realized error. Equation 7 then compares those realized errors under the continuous-future marginal surrogate; only the winning candidate updates the global GPTQ state.

Exhaustively replaying the full E4M3 grid is unnecessary. Order its $C \ = \ | S |$ admissible scales as ${ \mathcal S } = \{ \sigma _ { 1 } <$ $\cdots < \sigma _ { C } \}$ , let $\begin{array} { r l r } { s ^ { * } } & { { } = } & { s ^ { ( J ) } } \end{array}$ be the final proposal from Equation 11, and define the clipped insertion index $c =$ clip(searchsorte $| ( S , s ^ { * } ) ) \in \{ 1 , \ldots , C \}$ . For a chosen nonnegative window half-width $h ,$ we evaluate

$$
\mathcal { C } = \{ \sigma _ { \mathrm { c l i p } ( c + o ) } : o = - h , \ldots , h \} \cup \{ \Pi _ { S } ( s _ { \mathrm { o l d } } ) \} , \qquad \widehat { s } = \arg \operatorname* { m i n } _ { s \in \mathcal { C } } L ( s ) ,\tag{14}
$$

where clip restricts indices to $\{ 1 , \ldots , C \}$ and $\Pi _ { \mathcal { S } }$ projects to the nearest exportable scale. The fixed-code update locates a promising neighborhood but does not decide the final scale: recurrence-exact replay adjudicates among the hardware-valid candidates using the state each candidate actually creates. Figure 1 summarizes this proposal–replay– commit sequence.

![](images/92cda6e41baccea25ae361605174796cd927ef0df9adc2d444c4c88fb9819841.jpg)  
Figure 1: Schur Replay proposes a local window of exportable scales, replays each candidate from the same blockentry state, and commits the candidate minimizing $L ( s ) = \delta _ { G } ( s ) ^ { \top } S _ { G | P } \delta _ { G } ( s )$

Exhaustive-window audit. An exhaustive audit over all 126 exportable scales finds that the $h = 4$ window contains a global surrogate optimum for 83.28% of Qwen rows and 83.53% of Llama rows; $h = 8$ exceeds 99.96% for both. Appendix D.5 reports layer-wise results and the sampled-group scope.

Retaining the status-quo scale gives the per-group surrogate guarantee

$$
L ( \widehat { s } ) \leq L ( \Pi _ { \cal { S } } ( s _ { \mathrm { o l d } } ) ) .\tag{15}
$$

Ties within tolerance prefer the old scale and then the smaller scale, making selection deterministic. Because the projected old scale is always included in C, the local search cannot increase the conditional replay surrogate relative to retaining that scale, even when the fixed-point proposal lands in a poor neighborhood. This is a per-group guarantee for the calibration-defined surrogate rather than a monotonicity claim for downstream task accuracy. Setting $h \geq C$ recovers the exhaustive-grid optimum.

Algorithm 1 gives the full procedure. For $K \leq 2 h + 2$ candidates and fixed $B \ = \ 1 6 .$ , proposal and replay add $O ( J B ^ { 2 } + K B ^ { 2 } )$ work beyond the $O ( B ^ { 3 } )$ group metric. Replay is vectorized over output rows and candidates as described in Section 4.2.1. We use $h = 4$ for both model families, capping each group at ten candidates without model- or dataset-specific tuning.

## 4.2 Faster PTQ Framework

Applying reconstruction-based PTQ to large dense and mixture-of-experts models creates a systems problem: weights, activations, and second-order state can exhaust memory, while assigning complete layers to devices leaves a largelayer solve serial. Our infrastructure combines layer-wise memory virtualization with same-layer output-row-parallel $G P T Q .$ . The former manages a complete per-layer lifecycle across storage tiers; the latter distributes independent row recurrences across tensor-parallel ranks. Both preserve the mathematical row-wise solve: they change placement and scheduling, not the quantization objective or update order within a row (Figure 2).

## 4.2.1 Batched Candidate Replay and Parallel Commit

Output rows and candidate scales are independent despite the column-sequential GPTQ recurrence. We therefore solve a tensor of shape $M \times K \times B$ : each row–candidate pair owns a private trailing state, while one batched launch processes all pairs at a fixed column. The implementation still executes the B columns in their original order, so batching exposes independent work without replacing the recurrence by an independent-block approximation; it requires B sequential steps rather than KB separately launched steps.

After scale selection, the commit path quantizes only the current group and propagates only the winning error to the unquantized suffix. The trailing correction is an in-place rank-one update, and CUDA Graph capture reduces launch

![](images/48890ddbbe1a3a374350d67ff29f49854acee28ef0c421c6b32dc0774fde04d0.jpg)  
Figure 2: Scalable NVFP4 PTQ infrastructure: shared-Hessian row-parallel reconstruction (left), layer-wise activation collection and weight prefetch/retirement (center), and GPU–CPU–SSD–HDFS overflow (right).  
overhead without changing the eager kernel sequence. Algorithm 2 gives the schedule.

## 4.2.2 Aggressive Layer-Wise Memory Virtualization

GPU memory prioritizes the current layer, activation Gram state, and inverse-Hessian factor; CPU memory, rank-local SSD, and finally HDFS hold lower-priority activation and reconstruction state. This ordering keeps solver-critical tensors resident while permitting calibration activations to exceed device and host capacity. Temporary spill files are atomically published so interrupted writes cannot be consumed as valid state.

As shown in Figure 2, only layer i is installed on device during calibration; a pinned two-slot pool prefetches layer i + 1, and the full-precision layer is retired after its NVFP4 fragment becomes durable. Consequently, device weight residency is bounded by the active layer plus one prefetched layer rather than the full model, leaving memory for Hessian factorization and replay. All the prefetch and offload of activations and weights are executed asynchornously while the main solver executes on the current layer, Algorithm 3 and Appendix D.6 give the protocol.

## 4.2.3 Recoverable Streaming Export

Each calibrated layer is packed immediately into a local NVFP4 fragment. As the quantized previous layer are always immediately stored in the storage when quantization is finished, we can use these cache as kind of a checkpoint.

An atomic manifest advances only after the fragment and next-layer inputs are durable, enabling layer-grained recovery and full-precision retirement. Appendix D.6 and Algorithm 4 give the details.

## 4.2.4 Same-Layer Output-Row-Parallel GPTQ

As shown in the left of Figure 2, for a row-parallel layer, rank r holds $W _ { r } \in \mathbb { R } ^ { M \times N _ { r } }$ and $X _ { r } \in \mathbb { R } ^ { N _ { r } \times T }$ . After all-gathering $X = \mathrm { c a t } _ { 0 } ( X _ { 0 } , \ldots , X _ { P _ { \mathrm { t p } } - 1 } )$ , each rank forms $H _ { r } \ = \ X _ { r } X ^ { \top }$ ; gathering these row blocks yields the shared $H = X X ^ { \top }$ . Because the reconstruction objective separates over output rows, ranks can solve disjoint row ranges with the same activation order and inverse-Hessian factor, gather the solved rows, and re-shard along N. Thu all ranks work on the current layer rather than assigning that entire layer to one owner. This decomposition equals a single-device solve in exact arithmetic; Appendix D.6 reports floating-point validation.

Column-parallel layers reduce Hessian numerators and sample counts before renormalization; grouped experts retain separate Hessians and quantizer states. Algorithm 5 gives the schedule.

## 5 Experiments

## 5.1 Setup and Baselines

Models, calibration, and evaluation. We evaluate W4A4 NVFP4 on Qwen3.5-397B-A17B [Qwen Team, 2026] and Llama-3.3-70B-Instruct [Meta AI, 2024, Grattafiori et al., 2024]. Accuracy runs share a decontaminated 255- document science/commonsense/math JSONL (seed 42), deterministic load order, batch size 1, and maximum length 16,384; Appendices D.2 and D.3 specify the model checkpoints, calibration construction, quantization hyperparame ters, module coverage, and structural exceptions. The 397B model uses EP8/TP8 and the 70B model uses one GPU for quantization and TP2 for inference. We report single-sample greedy pass@1 (T = 0, top-p = 1, and at most 8,192 new tokens) over 15,461 questions from MMLU-Pro, GSM8K, CMath, LiveCodeBench, English MGSM, Hu manEval, and GPQA-Diamond [Wang et al., 2024, Cobbe et al., 2021, Wei et al., 2023, Jain et al., 2024, Shi et al., 2023, Chen et al., 2021, Rein et al., 2023]; Appendix D.4 gives dataset versions, prompts, grader provenance, exact counts, and paired uncertainty.

$$
R _ { \mathrm { w } } = 1 0 0 \frac { \sum _ { d } n _ { d } s _ { d } ^ { \mathrm { q u a n t } } } { \sum _ { d } n _ { d } s _ { d } ^ { \mathrm { B F 1 6 } } } ,\tag{16}
$$

where $n _ { d }$ is the number of questions in dataset d. Values slightly above 100% indicate higher aggregate accuracy than BF16 on this finite evaluation set.

Baselines. We compare GPTQ with SmoothQuant and AWQ rescaling; Four Over Six, SOAR, and ScaleSweep block-scale selection; ARCQuant residual channels; MR-GPTQ micro-rotations; and Schur Replay [Xiao et al., 2023, Lin et al., 2024, Cook et al., 2026, Bao et al., 2026, Lin and Wan, 2026, Meng et al., 2026, Egiazarian et al., 2026]. SmoothQuant, AWQ, Four Over Six, SOAR, and ScaleSweep use the same GPTQ reconstruction backend; ARCQuant and MR-GPTQ follow their format-specific reconstruction procedures.

## 5.2 W4A4 Accuracy

Tables 1 and 2 report all seven task scores. Schur Replay has the highest quantized question-weighted recovery: 99.35% on Qwen3.5-397B-A17B and 100.84% on Llama-3.3-70B-Instruct, versus ScaleSweep’s 97.28%/99.03% and MR-GPTQ’s 93.86%/91.81%.

Table 1: W4A4 results on Qwen3.5-397B-A17B. W. Rec. is the question-weighted recovery in Equation 16. Bold marks the best quantized method; BF16 provides the full-precision reference.
<table><tr><td>Method</td><td>MMLU-P</td><td>GSM8K</td><td>CMath</td><td>LiveCode</td><td>MGSM</td><td>HumEval</td><td>GPQA</td><td>W. Rec. (%)</td></tr><tr><td>BF16</td><td>76.09</td><td>92.19</td><td>91.80</td><td>44.75</td><td>92.00</td><td>91.46</td><td>55.05</td><td>100.00</td></tr><tr><td>MR-GPTQ</td><td>70.99</td><td>89.54</td><td>87.89</td><td>40.75</td><td>88.40</td><td>91.46</td><td>43.43</td><td>93.86</td></tr><tr><td>GPTQ</td><td>73.12</td><td>91.36</td><td>92.90</td><td>41.75</td><td>91.20</td><td>85.37</td><td>45.96</td><td>96.69</td></tr><tr><td>ScaleSweep</td><td>73.60</td><td>92.12</td><td>91.89</td><td>41.75</td><td>90.00</td><td>93.29</td><td>47.98</td><td>97.28</td></tr><tr><td>Four Over Six+GPTQ</td><td>73.91</td><td>90.67</td><td>91.53</td><td>40.25</td><td>89.20</td><td>87.80</td><td>52.02</td><td>97.32</td></tr><tr><td>SmoothQuant+GPTQ</td><td>74.07</td><td>90.22</td><td>91.53</td><td>43.00</td><td>89.20</td><td>92.68</td><td>52.53</td><td>97.60</td></tr><tr><td>AWQ+GPTQ</td><td>74.09</td><td>91.13</td><td>90.71</td><td>41.50</td><td>92.80</td><td>93.29</td><td>53.03</td><td>97.69</td></tr><tr><td>ARCQuant</td><td>75.17</td><td>89.99</td><td>89.80</td><td>39.75</td><td>90.80</td><td>90.24</td><td>48.48</td><td>98.34</td></tr><tr><td>SOAR+GPTQ</td><td>74.55</td><td>92.04</td><td>92.81</td><td>41.25</td><td>89.20</td><td>92.68</td><td>55.05</td><td>98.38</td></tr><tr><td>Schur Replay (ours)</td><td>75.40</td><td>93.03</td><td>92.17</td><td>42.25</td><td>92.80</td><td>92.68</td><td>52.53</td><td>99.35</td></tr></table>

Table 2: W4A4 results on Llama-3.3-70B-Instruct under the same seven-dataset protocol as Table 1.
<table><tr><td>Method</td><td>MMLU-P</td><td>GSM8K</td><td>CMath</td><td>LiveCode</td><td>MGSM</td><td>HumEval</td><td>GPQA</td><td>W. Rec. (%)</td></tr><tr><td>BF16</td><td>55.19</td><td>93.25</td><td>86.61</td><td>31.00</td><td>92.00</td><td>84.15</td><td>37.88</td><td>100.00</td></tr><tr><td>MR-GPTQ</td><td>49.33</td><td>89.76</td><td>85.70</td><td>29.75</td><td>91.20</td><td>84.76</td><td>36.87</td><td>91.81</td></tr><tr><td>GPTQ</td><td>54.37</td><td>91.81</td><td>85.97</td><td>29.25</td><td>91.20</td><td>84.15</td><td>42.42</td><td>98.67</td></tr><tr><td>AWQ+GPTQ</td><td>54.24</td><td>93.18</td><td>86.70</td><td>32.00</td><td>92.80</td><td>81.71</td><td>39.90</td><td>98.85</td></tr><tr><td>ScaleSweep</td><td>54.53</td><td>92.72</td><td>86.34</td><td>31.50</td><td>90.00</td><td>84.15</td><td>38.38</td><td>99.03</td></tr><tr><td>Four Over Six+GPTQ</td><td>54.90</td><td>91.51</td><td>84.70</td><td>29.25</td><td>90.00</td><td>82.32</td><td>44.44</td><td>99.14</td></tr><tr><td>SmoothQuant+GPTQ</td><td>54.75</td><td>91.89</td><td>86.98</td><td>30.75</td><td>92.40</td><td>82.93</td><td>37.37</td><td>99.26</td></tr><tr><td>ARCQuant</td><td>55.44</td><td>93.56</td><td>83.70</td><td>32.50</td><td>93.60</td><td>82.32</td><td>40.40</td><td>100.15</td></tr><tr><td>SOAR+GPTQ</td><td>55.68</td><td>92.19</td><td>85.79</td><td>30.00</td><td>91.20</td><td>81.71</td><td>34.34</td><td>100.20</td></tr><tr><td>Schur Replay (ours)</td><td>55.93</td><td>93.18</td><td>86.16</td><td>29.25</td><td>92.00</td><td>83.54</td><td>39.90</td><td>100.84</td></tr></table>

![](images/35f436fb02a7cc8be7490a5963820cf286531b5d92d82c90e0c3ddfa2cb5f14a.jpg)

![](images/14a086b22c9c7928660d50fe5816736e11cabe730cef2e68d39fa396de37df5a.jpg)

![](images/a18e2ef25a0787d1cd1d6de15b65468e0621ba05d2259699949a810e22ee9652.jpg)

![](images/9fb4ef70aeb807329d57428ede10363036ac0cbe7b8a7f898e314d34ed56d59f.jpg)

![](images/b7b4a79ac182d89f001656d29c91f24657efb7f91ef1adaad5cded30a8a821c3.jpg)  
Figure 3: NVFP4 quantization efficiency. Panels (a–b) report steady-layer compute, (c–d) full-model compute, and (e) 397B peak memory. Warm bars compare infrastructure under aligned GPTQ; diagonal hatching marks projected 397B GPTQ totals. Purple cross-hatched bars are separate measured Schur Replay runs: W4A16 on one GPU for 70B and on eight GPUs for 397B. The 397B GPTQ bars use W4A4, so the purple bars are contextual measurements rather than inputs to the infrastructure speedup ratios.

## 5.3 Ablations

With ten candidates (h = 4) for both models, Qwen macro recovery at h = 1, 4, 8, 16 is 97.70%, 98.92%, 98.62%, and 97.21%. Although h = 8 doubles the neighborhood and raises surrogate-optimum coverage from 83.3% to 99.96%, it does not improve recovery. Removing Schur scoring, recurrence replay, or both loses 90, 15, or 115 answers; static scoring raises held-out log-probability MSE from 1.769 to 1.842 (t = 3.2). Appendix B gives the full W4A4 ablations and W4A16 comparison.

## 5.4 Infrastructure Efficiency

All 397B GPTQ pipelines share a checkpoint; ours and ModelOpt use comparable Megatron TP8/EP8 representations, while LLM Compressor uses its Hugging Face path. Only ours parallelizes the current layer’s solve across ranks. With identical 32 × 4096 inputs, steady-layer times are 120.25, 1,824.29, and 2,782.89 seconds for ours, ModelOpt, and LLM Compressor; the 70B control gives 1.13×/2.76× speedups. Measured 70B GPTQ totals are 0.80, 0.91, and 2.09 hours, while the corresponding 397B totals (2.01, 30.63, and 46.42 hours) are projections; peak memory is 35.0, 218.6, and 86.6 GB/GPU. For reference, we additionally report calibration time for our Schur Replay algorithm: 54.91 seconds per steady layer and 1.22 hours end-to-end for 70B on one GPU, and 157.84 seconds per steady layer and 2.67 hours end-to-end for 397B on eight GPUs. Because Schur Replay performs additional per-group scale search, these measurements are not included in the infrastructure speedup ratios. Appendix D.6 gives checks.

## 6 Conclusion

Schur Replay provides recurrence-exact scale evaluation and conditional Schur scoring; our separate execution infrastructure provides memory-bounded, output-row-parallel GPTQ. Together, these components produce 397B and 70B NVFP4 models that retain 99.35% and 100.84% question-weighted BF16 recovery across seven benchmarks, with ablations supporting both replay and scoring. The infrastructure reduces measured per-layer and 70B full-model quantization compute against the evaluated public toolchains, and the 397B run peaks at 35.0 GB per GPU; 397B full-model runtimes remain layer-type-aware projections. These results show that hardware-valid NVFP4 conversion can be accurate and scalable without retraining.

The current evidence is empirical rather than universal: the exhaustive scale audit covers the sampled groups, and downstream evaluation spans two architectures and seven tasks. Future work should test whether the same reconstruction gains transfer to other microscaling formats, calibration regimes, and model families, while extending complete end-to-end measurements to the largest deployments.

## AI use statement

Generative AI tools were used to assist with code drafting, debugging, and language polishing of the manuscript. All AI-assisted code and text were reviewed and verified by the authors, who take full responsibility for the final implementation, experiments, claims, and manuscript content.

## Ethics statement

This work’s controlled quantitative experiments use publicly available models and public benchmarks and do not involve human subjects or private data. We additionally mention an anonymized, authorized integration in an internal production pipeline, but do not disclose or release the proprietary model, data, metric definitions, or raw outputs, and do not use that integration to support the paper’s quantitative claims. Reducing the cost of deploying large language models may broaden access to their capabilities. However, quantization does not remove biases, safety risks, or other limitations inherited from the underlying models, and deployments should retain the safeguards appropriate to those models and their application contexts.

## Reproducibility statement

The main paper specifies the quantization objective, scale-selection procedure, calibration protocol, evaluation metric, and infrastructure comparisons. The appendix provides complete pseudocode, derivations, model and module coverage, calibration details, ablation definitions, exhaustive-window audits, and additional system measurements. The source package contains the data-derived tables and figures used in the manuscript.

## References

Advanced Micro Devices, Inc. AMD Quark. Software documentation, 2024. URL https://quark.docs.amd. com/latest/. Accessed 2026.

Chengzhu Bao, Xianglong Yan, Zhiteng Li, Guangshuo Qin, Guanghua Yu, and Yulun Zhang. SOAR: Scale optimization for accurate reconstruction in NVFP4 quantization. arXiv preprint arXiv:2605.12245, 2026. URL https://arxiv.org/abs/2605.12245.

Yoshua Bengio, Nicholas Leonard, and Aaron Courville. Estimating or propagating gradients through stochastic ´ neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013. URL https://arxiv.org/ abs/1308.3432.

Yelysei Bondarenko, Riccardo Del Chiaro, and Markus Nagel. Low-rank quantization-aware training for LLMs. arXiv preprint arXiv:2406.06385, 2024. URL https://arxiv.org/abs/2406.06385.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Ed wards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. URL https://arxiv.org/abs/2107.03374.

Mengzhao Chen, Wenqi Shao, Peng Xu, Jiahao Wang, Peng Gao, Kaipeng Zhang, and Ping Luo. EfficientQAT: Efficient quantization-aware training for large language models. arXiv preprint arXiv:2407.11062, 2025. URL https://arxiv.org/abs/2407.11062.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. URL https://arxiv.org/abs/2110.14168.

Jack Cook, Junxian Guo, Guangxuan Xiao, Yujun Lin, Keith Wyss, Mahdi Nazemi, Asit Mishra, Carlo del Mundo, Tijmen Blankevoort, and Song Han. Four over six: More accurate NVFP4 quantization with adaptive block scaling. arXiv preprint arXiv:2512.02010, 2026. URL https://arxiv.org/abs/2512.02010.

Vage Egiazarian, Roberto L. Castro, Denis Kuznedelev, Andrei Panferov, Eldar Kurtic, Shubhra Pandit, Alexandre Marques, Mark Kurtz, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Bridging the gap between promise and performance for microscaling FP4 quantization. In International Conference on Learning Representations, 2026. URL https://arxiv.org/abs/2509.23202.

Elias Frantar, Sidak Pal Singh, and Dan Alistarh. Optimal brain compression: A framework for accurate post-training quantization and pruning. In Advances in Neural Information Processing Systems, volume 35, 2022. doi: 10.52202/068431-0323. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/hash/1caf09c9f4e6b0150b06a07e77f2710c-Abstract-Conference.html.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate post-training quantization for generative pre-trained transformers. In International Conference on Learning Representations, 2023. URL https: //arxiv.org/abs/2210.17323.

Ruihao Gong, Yang Yong, Shiqiao Gu, Yushi Huang, Chengtao Lv, Yunchen Zhang, Xianglong Liu, and Dacheng Tao. LLMC: Benchmarking large language model quantization with a versatile compression toolkit. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 132–152, 2024. URL https://aclanthology.org/2024.emnlp-industry.12.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. URL https://arxiv.org/abs/2407.21783.

Babak Hassibi, David G. Stork, and Gregory J. Wolff. Optimal brain surgeon and general network pruning. In IEEE International Conference on Neural Networks, pages 293–299, 1993. doi: 10.1109/ICNN.1993.298572.

Intel Corporation. Intel Neural Compressor. Software repository, 2021. URL https://github.com/intel/ neural-compressor. Accessed 2026.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. LiveCodeBench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024. URL https://arxiv.org/abs/2403.07974.

Jemin Lee, Sihyeong Park, Jinse Kwon, Jihun Oh, and Yongin Kwon. Exploring the trade-offs: Quantization methods, task difficulty, and model size in large language models from edge to giant. arXiv preprint arXiv:2409.11055, 2024. URL https://arxiv.org/abs/2409.11055.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. AWQ: Activation-aware weight quantization for on-device LLM compression and acceleration. In Proceedings ofMachine Learning and Systems, volume 6, 2024. URL https://arxiv.org/ abs/2306.00978.

Li Lin and Xiaojun Wan. ScaleSweep: Accurate NVFP4 post-training quantization of LLMs via block scale initialization. arXiv preprint arXiv:2606.07618, 2026. URL https://arxiv.org/abs/2606.07618.

Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman˘ Krishnamoorthi, and Vikas Chandra. LLM-QAT: Data-free quantization aware training for large language models. arXiv preprint arXiv:2305.17888, 2023. URL https://arxiv.org/abs/2305.17888.

Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, and Tijmen Blankevoort. SpinQuant: LLM quantization with learned rotations. In International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2405.16406.

Haoqian Meng, Yilun Luo, Yafei Zhao, Wenyuan Liu, Peng Zhang, and Xindian Ma. ARCQuant: Boosting NVFP4 quantization with augmented residual channels for LLMs. arXiv preprint arXiv:2601.07475, 2026. URL https: //arxiv.org/abs/2601.07475.

Meta AI. Llama 3.3-70B-Instruct model card. Hugging Face model card, 2024. URL https://huggingface. co/meta-llama/Llama-3.3-70B-Instruct.

Microsoft Corporation. Olive: Hardware-aware model optimization. Software documentation and repository, 2023. URL https://microsoft.github.io/Olive/. Accessed 2026.

ModelCloud.ai. GPTQModel. Software repository, 2024. URL https://github.com/ModelCloud/ GPTQModel. Accessed 2026.

Markus Nagel, Rana Ali Amjad, Mart Van Baalen, Christos Louizos, and Tijmen Blankevoort. Up or down? adaptive rounding for post-training quantization. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 7197–7206, 2020. URL https://proceedings.mlr.press/v119/nagel20a.html.

NVIDIA, Felix Abecassis, Anjulie Agrusa, Dong Ahn, Jonah Alben, Stefania Alborghetti, Michael Andersch, Sivakumar Arayandi, Alexis Bjorlin, Aaron Blakeman, et al. Pretraining large language models with NVFP4. arXiv preprint arXiv:2509.25149, 2025. URL https://arxiv.org/abs/2509.25149.

NVIDIA Corporation. NVIDIA Model Optimizer. Software repository, 2024. URL https://github.com/ NVIDIA/Model-Optimizer. Accessed 2026.

Andrew Or, Apurva Jain, Daniel Vega-Myhre, Jesse Cai, Charles David Hernandez, Zhenrui Zheng, Driss Guessous, Vasiliy Kuznetsov, Christian Puhrsch, Mark Saroufim, Supriya Rao, Thien Tran, and Aleksandar Samardziˇ c.´ TorchAO: Pytorch-native training-to-serving model optimization. arXiv preprint arXiv:2507.16099, 2025. URL https://arxiv.org/abs/2507.16099. ICML 2025 Workshop on Championing Open-source Development.

Qwen Team. Qwen3.5: Towards native multimodal agents. Official release and model card, 2026. URL https:// qwen.ai/blog?id=qwen3.5. Qwen3.5-397B-A17B model card: https://huggingface.co/Qwen/ Qwen3.5-397B-A17B.

Red Hat AI and vLLM Project. LLM Compressor, 2024. URL https://github.com/vllm-project/ llm-compressor.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof q&a benchmark. arXiv preprint arXiv:2311.12022, 2023. URL https://arxiv.org/abs/2311.12022.

Bita Darvish Rouhani, Ritchie Zhao, Ankit More, Mathew Hall, Alireza Khodamoradi, Summer Deng, Dhruv Choudhary, Marius Cornea, Eric Dellinger, Kristof Denolf, Stosic Dusan, Venmugil Elango, Maximilian Golub, Alexander Heinecke, Phil James-Roxby, Dharmesh Jani, Gaurav Kolhe, Martin Langhammer, Ada Li, Levi Melnick, Maral Mesmakhosroshahi, Andres Rodriguez, Michael Schulte, Rasoul Shafipour, Lei Shao, Michael Siu, Pradeep Dubey, Paulius Micikevicius, Maxim Naumov, Colin Verrilli, Ralph Wittig, Doug Burger, and Eric Chung. Microscaling data formats for deep learning. arXiv preprint arXiv:2310.10537, 2023. URL https: //arxiv.org/abs/2310.10537.

Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, and Ping Luo. OmniQuant: Omnidirectionally calibrated quantization for large language models. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2308.13137.

Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. Language models are multilingual chain-of-thought reasoners. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/ 2210.03057.

Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, and Christopher De Sa. QuIP#: Even better LLM quantization with hadamard incoherence and lattice codebooks. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, 2024. URL https://arxiv. org/abs/2402.04396.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. MMLU-Pro: A more robust and challenging multi-task language understanding benchmark. arXiv preprint arXiv:2406.01574, 2024. URL https://arxiv.org/abs/2406.01574.

Tianwen Wei, Jian Luan, Wei Liu, Shuang Dong, and Bin Wang. CMATH: Can your language model pass chinese elementary school math test? arXiv preprint arXiv:2306.16636, 2023. URL https://arxiv.org/abs/ 2306.16636.

Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. SmoothQuant: Accurate and efficient post-training quantization for large language models. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, 2023. URL https://arxiv. org/abs/2211.10438.

Meng Xin, Sweta Priyadarshi, Jingyu Xin, Bilal Kartal, Aditya Vavre, Asma Kuriparambil Thekkumpate, Zijia Chen, Ameya Sunil Mahabaleshwarkar, Ido Shahaf, Akhiad Bercovich, Kinjal Patel, Suguna Varshini Velury, Chenjie Luo, Zhiyu Cheng, Jenny Chen, Chen-Han Yu, Wei Ping, Oleg Rybakov, Nima Tajbakhsh, Oluwatobi Olabiyi, Dusan Stosic, Di Wu, Song Han, Eric Chung, Sharath Turuvekere Sreenivas, Bryan Catanzaro, Yoshi Suhara, Tijmen Blankevoort, and Huizi Mao. Quantization-aware distillation for NVFP4 inference accuracy recovery. arXiv preprint arXiv:2601.20088, 2026. URL https://arxiv.org/abs/2601.20088.

## A Code & Artifact Availability

All development in this work is built on NVIDIA ModelOpt. Upon acceptance, we will release a general-purpose patch that can be installed on top of ModelOpt to enable Schur Replay for both W4A4 and W4A16 quantization. The patch will also provide the core infrastructure contributions introduced in this paper, allowing ModelOpt to execute PTQ methods—including GPTQ, GPTAQ, and Schur Replay—with improved hardware efficiency.

We will additionally release the calibration dataset used for the public-model experiments, together with its deterministic construction metadata, and all quantized model artifacts produced for the main experiments, subject to the licenses and redistribution terms of the corresponding base models and source datasets. These artifacts are intended to support direct reproduction of the reported public-model evaluation without requiring users to repeat every large-scale quantization run.

The released implementation will support the complete four-tier GPU–CPU–SSD–HDFS storage hierarchy used in our public-model experiments when both a local SSD path and an HDFS path are configured. If either persistentstorage path is unavailable, the same implementation will automatically degrade to a three-tier cache hierarchy while preserving the quantization algorithm and execution semantics. The proprietary Prompt Enhancer, internal evaluation assets, and production-specific integration are outside the release scope.

## B Complete Ablation Tables

Table 3: Replay-window ablation on Qwen3.5-397B-A17B W4A4.
<table><tr><td>Half-width h</td><td>Default</td><td>Macro recovery (%)</td></tr><tr><td>1</td><td>no</td><td>97.70</td></tr><tr><td>4</td><td>yes</td><td>98.92</td></tr><tr><td>8</td><td>no</td><td>98.62</td></tr><tr><td>16</td><td>no</td><td>97.21</td></tr></table>

Factorial design and counting unit. We cross two residual constructions with two scoring metrics while holding the candidate grid, scale proposal, old-scale retention, tie-breaking, calibration data, and quantization backend fixed. The static construction independently rounds every column from the shared group-entry snapshot w ,

$$
q _ { j } ^ { \mathrm { s t a t i c } } ( s ) = s \rho ( w _ { 0 , j } / s ) , \qquad r ^ { \mathrm { s t a t i c } } ( s ) = w _ { 0 } - q ^ { \mathrm { s t a t i c } } ( s ) .\tag{17}
$$

Although the implementation receives $U _ { G G }$ through the common loss interface, static construction does not use it: the quantization error of column $j$ never changes the value rounded at column $j + 1$ . The replay construction instead executes the commit-time GPTQ recurrence in Equation 12. It rounds the updated state $w _ { j } ^ { ( j - 1 ) }$ and propagates each normalized error through $( U _ { G G } ) _ { j , j + 1 : B }$ before quantizing subsequent columns, producing

$$
r ^ { \mathrm { r e p l a y } } ( s ) = w _ { 0 } - q ^ { \mathrm { r e p l a y } } ( s ) .\tag{18}
$$

The residual axis therefore tests entry-snapshot RTN codes against the codes realized by the sequential GPTQ commit path. Independently, the metric axis scores either residual with the conditional Schur loss $r ^ { \top } S _ { G } r$ or the Euclidean loss $\| r \| _ { 2 } ^ { 2 }$ . The four cells are replay × Schur (ours), static × Schur (score rtn), replay × Euclidean (metric id), and static × Euclidean (pure rtn). We additionally report BF16 and replay × Schur with h = 16 (win16). Because dataset percentages have different resolution, Table 4 reports absolute correct-question counts; “1q” is the percentagepoint change caused by one question within that dataset.

All four cells are internal ablations of our own pipeline and none reproduces a published method. Every cell retains the exportable grid of Equation 6, the closed-form proposal of Equation 10, the same local candidate window, and old-scale retention; only the scored quantity changes. In particular metric id is not a prior baseline but our search with the conditional metric removed, and score rtn applies the Schur metric to a residual the commit path never produces. Baseline comparisons against published methods appear in Tables 1 and 2.

Attributing the metric main effect requires distinguishing two uses of second-order information. Established GPTQ/OBCstyle reconstruction uses $H ^ { - 1 }$ to order columns and to normalize per-column error during weight compensation. The metric axis here instead evaluates a shared block scale under $S _ { G } ,$ the Schur complement that eliminates the still unquantized future columns; Equation 8 shows this equals $( U _ { G G } ^ { \top } U _ { G G } ) ^ { - 1 }$ and reduces to the classical scalar score only at $B = 1$ . This extension of the conditional metric from single-weight decisions to shared-scale selection is introduced in this work, so the dominant metric effect measures the contribution of that extension rather than of preexisting practice.

Main effects and interaction. Starting from ours, replacing Schur with Euclidean scoring while holding replay fixed loses 90 questions (−0.747 recovery points). Replacing recurrence replay with a static residual while holding Schur fixed loses 15 questions (−0.124 points). An additive model therefore predicts a joint loss of 105 questions, or 11,863 correct and −0.871 points. The measured static × Euclidean cell has 11,853 correct (98.390% recovery), ten fewer questions than this prediction. Thus both Schur scoring and recurrence replay contribute positively in the controlled aggregate, the metric main effect is larger, and their joint removal has an additional −10-question (−0.083-point) interaction.

Table 4: Per-dataset absolute correct counts for the controlled metric × residual-construction ablation. The two panels contain the same rows and together specify every cell. Bold marks the best quantized count per dataset.
<table><tr><td>Dataset</td><td> $n _ { q }$ </td><td>Weight 1q (pp)</td><td>BF16</td><td>Ours</td></tr><tr><td>MMLU-Pro GSM8K</td><td>12,032</td><td>77.8%</td><td>0.008 9,155</td><td>9,072</td></tr><tr><td>CMath</td><td>1,319</td><td>8.5%</td><td>0.076 1,216 0.091</td><td>1,227</td></tr><tr><td>LiveCodeBench</td><td>1,098</td><td>7.1%</td><td>1,008 179</td><td>1,012</td></tr><tr><td>MGSM-English</td><td>400</td><td>2.6% 1.6%</td><td>0.250 230</td><td>169</td></tr><tr><td>HumanEval</td><td>250 164</td><td>1.1%</td><td>0.400 0.610 150</td><td>232 152</td></tr><tr><td>GPQA-Diamond</td><td>198</td><td>1.3%</td><td>0.505</td><td>104</td></tr><tr><td>Total</td><td>15,461</td><td>100%</td><td>109 12,047</td><td>11,968</td></tr></table>

<table><tr><td>Dataset</td><td>score_rtn metric_id pure_rtn win16</td><td></td><td></td><td></td></tr><tr><td>MMLU-Pro</td><td>9,079</td><td>9,010</td><td>8,979</td><td>9,093</td></tr><tr><td>GSM8K</td><td>1,218</td><td>1,206</td><td>1,212</td><td>1,216</td></tr><tr><td>CMath</td><td>1,006</td><td>1,013</td><td>1,011</td><td>1,013</td></tr><tr><td>LiveCodeBench</td><td>161</td><td>166</td><td>166</td><td>167</td></tr><tr><td>MGSM-English</td><td>225</td><td>228</td><td>229</td><td>223</td></tr><tr><td>HumanEval</td><td>157</td><td>149</td><td>154</td><td>148</td></tr><tr><td>GPQA-Diamond</td><td>107</td><td>106</td><td>102</td><td>100</td></tr><tr><td>Total</td><td>11,953</td><td>11,878</td><td>11,853</td><td>11,960</td></tr></table>

Table 5: Question-weighted recovery and deficit relative to the 12,047-correct BF16 reference.
<table><tr><td>Cell</td><td></td><td>Correct Recovery (%) ∆ questions vs. BF16</td></tr><tr><td>BF16</td><td>12,047</td><td></td></tr><tr><td>Ours (replay × Schur)</td><td>11,968</td><td>99.344 -79</td></tr><tr><td>score_rtn (static × Schur)</td><td>11,953</td><td>99.220 -94</td></tr><tr><td>win16 (replay × Schur, h = 16)</td><td>11,960</td><td>99.278 -87</td></tr><tr><td>metric_id (replay×Euclidean)</td><td>11,878</td><td>98.597 -169</td></tr><tr><td>pure_rtn (static × Euclidean)</td><td>11,853</td><td>98.390 -194</td></tr></table>

## B.1 Weight-Only NVFP4 (W4A16)

The main results quantize both weights and activations. To separate the weight-reconstruction contribution from activation quantization, Table 6 repeats the Qwen3.5-397B-A17B comparison with weights in NVFP4 and activations in BF16, holding the calibration set, evaluation suite, greedy decoding protocol, and BF16 reference fixed. Because the archived per-question scores cover the same seven non-floor datasets and 15,461 questions as Tables 1 and 2, both recovery aggregates are recomputed here on that scope rather than reusing the 10-dataset macro recorded in the source file.

Two observations carry over from W4A4. Schur Replay attains the lowest weight-reconstruction error and the highest recovery in both act-order settings, and its advantage over GPTQ is larger than the gap between the two GPTQ variants. The ordering of reconstruction error also matches the ordering of recovery across all five arms here, which the W4A4 setting does not always exhibit because activation quantization adds an error source that weight reconstruction cannot address. Act-order interacts differently with the two methods: it helps Schur Replay on both metrics but lowers GPTQ recovery from 98.84% to 96.28% despite improving its reconstruction error, so reconstruction error alone does not determine end-to-end accuracy. The single arm exceeding 100% weighted recovery reflects small gains on GSM8K and HumanEval offsetting losses on LiveCodeBench and GPQA-Diamond, and single-run greedy decoding does no resolve differences of this size; the per-dataset counts and uncertainty treatment in Appendix D.4 apply here as well.

Table 6: Weight-only NVFP4 (W4A16) on Qwen3.5-397B-A17B over the same seven datasets and 15,461 questions as the W4A4 tables. “Rel. MSE” is the relative weight-reconstruction error; “Macro” and “Weighted” are unweighted and question-weighted recovery against the 12,047-correct BF16 reference.
<table><tr><td>Method</td><td>Act-order</td><td>Rel. MSE</td><td>Correct</td><td>Macro (%)</td><td>Weighted (%)</td></tr><tr><td>BF16 reference</td><td>一</td><td>一</td><td> $^ { 1 2 , 0 4 7 }$ </td><td>100.00</td><td>100.00</td></tr><tr><td>RTN + shrink search</td><td>一</td><td>0.0832</td><td>11,656</td><td>94.65</td><td>96.75</td></tr><tr><td>GPTQ</td><td>一</td><td>0.0480</td><td>11,908</td><td>95.48</td><td>98.84</td></tr><tr><td>GPTQ</td><td>√</td><td>0.0440</td><td>11,599</td><td>94.99</td><td>96.28</td></tr><tr><td>Schur Replay</td><td>一</td><td>0.0379</td><td>11,997</td><td>98.49</td><td>99.59</td></tr><tr><td>Schur Replay</td><td>√</td><td>0.0365</td><td>12,055</td><td>99.18</td><td>100.07</td></tr></table>

## C Pseudocode

This appendix collects the complete pseudocode for Schur Replay and the execution infrastructure described in Section 4. We use AG for all-gather, $\mathrm { c a t } _ { d }$ for concatenation along dimension d, $[ n ] = \{ 0 , \ldots , n - 1 \}$ , and $\epsilon > 0$ for a numerical guard.

Schur Replay group-scale selection   
$\mathbf { I n p u t : } w _ { 0 } \in \mathbb { R } ^ { M \times B } , U _ { G G } \in \mathbb { R } ^ { B \times B } , s _ { 0 } \in \mathcal { S } ^ { M } , h , J = 3$   
$S _ { G } \gets ( U _ { G G } ^ { \top } U _ { G G } ) ^ { - 1 } ; \quad s \gets s _ { 0 }$   
for $t = 1 , \dots , J$ do   
$Q  \rho ( w _ { 0 } / s ) ; \quad d  \mathrm { d i a g } ( Q S _ { G } Q ^ { \top } )$   
$s  ( Q S _ { G } w _ { 0 } ^ { \top } ) _ { \mathrm { d i a g } } / \operatorname* { m a x } ( d , \epsilon ) \mathrm { i f } d > \epsilon$   
end for   
$c _ { m } \gets \mathrm { c l i p } ( \mathrm { s e a r c h s o r t e d } ( S , s _ { m } ) ) ; \quad \mathcal { C } _ { m } \gets \{ S _ { \mathrm { c l i p } ( c _ { m } + o ) } \} _ { o = - h } ^ { h } \cup \{ \Pi _ { S } ( s _ { 0 , m } ) \} _ { s = 0 } ^ { h } ,$   
$K \gets 2 h + 2 ; \quad \mathbf { C } \gets \mathrm { p a d } ( \{ { \mathcal { C } _ { m } } \} _ { m = 1 } ^ { M } , K ) ; \quad R \gets w _ { 0 } [ \cdot , \emptyset , \colon ] \otimes \mathbf { 1 } _ { K }$ // mask padded slots   
for $j = 1 , \dots , B$ do   
$q _ { j }  \mathbf { C } \rho ( R _ { : , : , j } / \mathbf { C } ) ; \quad e _ { j }  ( R _ { : , : , j } - q _ { j } ) / ( U _ { G G } ) _ { j j }$   
$R _ { : , : , j + 1 : B } \gets R _ { : , : , j + 1 : B } - e _ { j } \otimes ( U _ { G G } ) _ { j , j + 1 : B }$   
end for   
$q  \mathrm { s t a c k } _ { j = 1 } ^ { B } ( q _ { j } )$   
$L _ { m , k } \gets ( \bar { w } _ { 0 , m , : } - q _ { m , k , : } ) S _ { G } ( w _ { 0 , m , : } - q _ { m , k , : } ) ^ { \top }$ for valid $\mathbf { C } _ { m , k } ;$ padded losses are $+ \infty$   
$L _ { m } ( s ) \gets ( w _ { 0 , m , : } - q _ { m } ( s ) ) S _ { G } ( w _ { 0 , m , : } - q _ { m } ( s ) ) ^ { \top }$   
$\begin{array} { r } { k _ { m } ^ { * } \gets \arg \operatorname* { m i n } _ { k } L _ { m , k } ; \quad \widehat { s } _ { m } \gets \mathbf { C } _ { m , k _ { m } ^ { * } } ; } \end{array}$ commi $\mathbf { \chi } _ { \mathbf { \chi } } ( w _ { 0 } , \widehat { s } , U _ { G G } )$   
assert $L _ { m } ( \widehat { s } _ { m } ) \leq L _ { m } ( \Pi _ { S } ( s _ { 0 , m } ) ) ,$ ∀m

```latex
Batched replay and group-local GPTQ commit
Input : $\begin{array} { r } { W \in \mathbb { R } ^ { M \times N } , U \in \mathbb { R } ^ { N \times N } , \mathcal { G } = \{ g , \dotsc , g + B - 1 \} , S _ { G } \in \mathbb { R } ^ { B \times B } , S \in \mathbb { R } ^ { M \times K } } \end{array}$
$W ^ { \mathrm { i n } }  W ; \quad W _ { 0 }  W _ { : , \mathscr { G } } ; \quad U _ { G }  U _ { \mathscr { G } , \mathscr { G } } ; \quad R  W _ { 0 } [ : , \emptyset , : ] \otimes { \bf 1 } _ { K }$
for $j = 1 , \dots , B$ do
$D _ { j }  \operatorname* { m a x } ( ( U _ { G } ) _ { j , j } , \epsilon ) ; \quad \widehat { W } _ { : , : , j }  Q _ { \mathrm { N V F P 4 } } ( R _ { : , : , j } ; S )$
$E _ { j }  ( R _ { : , : , j } - \widehat { W } _ { : , : , j } ) / D _ { j }$
$R _ { : , : , j + 1 : B } \gets R _ { : , : , j + 1 : B } - E _ { j } \otimes ( U _ { G } ) _ { j , j + 1 : B }$
end for
$\mathcal { L } _ { m , k } \gets ( \widehat { W } _ { m , k , : } - W _ { 0 , m , : } ) S _ { G } ( \widehat { W } _ { m , k , : } - W _ { 0 , m , : } ) ^ { \top }$
$k _ { m } ^ { * } \gets \arg \operatorname* { m i n } _ { k \in \{ 1 , \dots , K \} } \mathcal { L } _ { m , k } ; \quad S _ { m } ^ { * } \gets S _ { m , k _ { m } ^ { * } } , \quad m = 1 , \dots , M$
for $j = 1 , \dots , B$ do; $i \gets g + j - 1$
$\widehat { G } _ { : , j } \gets Q _ { \mathrm { N V F P 4 } } ( W _ { : , i } ; S ^ { * } ) ; \quad e _ { i } \gets ( W _ { : , i } - \widehat { G } _ { : , j } ) / U _ { i , i }$
$W _ { : , i : N } \gets W _ { : , i : N } - e _ { i } \otimes U _ { i , i : N }$
end for
Invariant : $( \widehat { G } , W _ { : , g + B : N } ) \equiv _ { \mathrm { b i t } } \mathrm { C o m m i t e a g e r } \left( W ^ { \mathrm { i n } } , \mathcal { G } , S ^ { * } ; U \right)$
Algorithm 2: Candidate-scale batching exposes the independent row and candidate dimensions without changing the
column-wise GPTQ recurrence.
```

Tiered capture and asynchronous layer residency   
Input : $\overline { { \{ x _ { b } \} } } _ { b = 0 } ^ { N _ { b } - 1 }$ (activation batches), $K _ { \mathrm { g p u } } , K _ { \mathrm { h o s t } }$ (tier capacities), $\overline { { \{ \theta _ { i } \} _ { i = 0 } ^ { L - 1 } } }$   
一 $\mathrm { ' } _ { \mathrm { G P U } } , \quad b < K _ { \mathrm { g p u } } ,$   
$\tau ( b ) \gets \Big \{ \mathrm { C P U } , \quad K _ { \mathrm { g p u } } \leq b < K _ { \mathrm { g p u } } + K _ { \mathrm { h o s t } } , \quad C _ { b } \gets \mathrm { p u t } _ { \tau ( b ) } ( x _ { b } )$   
DISK, otherwise;   
$\mathcal { H }  \mathrm { p a r k } _ { \mathrm { C P U } } ( \{ \theta _ { i } \} ) ; \quad \mathcal { A }  \emptyset$   
Prefetch(i): u ← i mod 2; $P _ { u } \gets \mathrm { p i n } ( \mathcal { H } _ { i } ) ; \widetilde { \theta } _ { i } \gets \mathrm { H 2 D } _ { \mathrm { a s y n c } } ( P _ { u } )$ ; record event η<sub>i</sub>   
Onload(i): wait $( \eta _ { i } ) ; \theta _ { i } \gets \widetilde { \theta } _ { i } ; \mathcal { A } \gets \mathcal { A } \cup \{ i \}$   
Release(i): θ ← ofload(θ ) or ∅ after pac $\mathfrak { x } ( i ) ; \mathcal { A }  \mathcal { A } \setminus \{ i \}$   
Invariant : $\vert { \mathcal { A } } \vert \leq 1 , \quad { \widetilde { \theta } } _ { i + 1 } \notin { \mathcal { A } } ,$ consume $\left( C _ { b } \right) \Rightarrow C _ { b }$ ∈/ DISK   
Algorithm 3: Bounded activation storage and single-layer weight residency prevent full-model weights and the complete   
calibration cache from residing on the accelerator simultaneously.   
Recoverable layer-wise PTQ pipeline   
Input : $\overline { { \{ \ell _ { i } = ( \theta _ { i } , \cdot ) \} _ { i = 0 } ^ { L - 1 } } }$ , C<sub>0</sub> (input cache), w (commit window)   
M (manifest); i<sub>0</sub> ← resume $( \mathcal { M } ) ; \quad \mathcal { F } _ { r } \gets \emptyset$ on every rank r   
park({ℓ<sub>i</sub>}); prefetch(i<sub>0</sub>)   
for $i = i _ { 0 } , \ldots , L - 1$ do   
$( z _ { < i - 1 } , z _ { i - 1 } , z _ { i } ) \gets ( \mathrm { S K I P , R U N , C A P T U R E } )$   
$X _ { i } \gets$ materialize(C ); onload(i); prefetch $( i + 1 )$   
$( a _ { i } , H _ { i } ) \gets \mathrm { { c a l i b r a t e } } ( \ell _ { i } , \underline { { X } } _ { i } ) ; ( \widehat { \theta } _ { i } , q _ { i } , m _ { i } ) \gets \mathrm { { P T } } \mathrm { { Q } } ( \theta _ { i } ; a _ { i } , H _ { i } )$   
$( F _ { i } , h _ { i } )$ ← atomic pack $( { \widehat { \theta } } _ { i } ) ; C _ { i + 1 } \gets$ capture next $( \ell _ { i } , \widehat { \theta } _ { i } , C _ { i } )$   
${ \mathcal { F } } _ { r } \gets { \mathcal { F } } _ { r } \cup \{ F _ { i } \} ; \operatorname { s a v e } ( \widehat { \theta } _ { i } , q _ { i } , m _ { i } )$   
if $( i + 1 )$ mod $w = 0 \mathrm { o r } i = L - 1 \mathrm { : }$ commit $( \mathcal { M } , i , C _ { i + 1 } )$   
discard(θ<sub>i</sub>)   
end for   
min<sub>r</sub> $\begin{array} { r } { | \mathcal { F } _ { r } | = \operatorname* { m a x } _ { r } | \mathcal { F } _ { r } | = L ; \quad A \gets \mathrm { f i n a l i z e } ( \{ ( F _ { i } , h _ { i } ) \} _ { i = 0 } ^ { L - 1 } ) } \end{array}$   
Invariant : $\mathcal { M } . i = \operatorname* { m a x } \{ j : \{ 0 : j \} \quad$ is durable}, SHA256(F ) = h   
Algorithm 4: Atomic window commits and hash-validated streaming fragments make long-running PTQ recoverable without   
retaining the full-precision model in memory.   
Row-parallel GPTQ solve   
Input : $N _ { r } = N / P _ { \mathrm { t p } } , \ B \ | \ N _ { r } , \ \mathcal { I } _ { r } = [ r N _ { r } , ( r + 1 ) N _ { r } ) , \ B _ { r } = [ r N _ { r } / B , ( r + 1 ) N _ { r } / B ) , \ r \in [ P _ { \mathrm { t p } } ]$   
$\lambda \geq 0$ (absolute Hessian damping); $B = 1 6$   
$\begin{array} { r } { \boldsymbol { W _ { r } } \in \mathbb { R } ^ { M \times N _ { r } } , \quad \boldsymbol { A _ { r } } \in \mathbb { R } ^ { M \times { \hat { N } _ { r } } / \Breve { \boldsymbol B } } } \end{array}$ (block-scale metadata), $X _ { r } \in \mathbb { R } ^ { N _ { r } \times T }$   
$X  \mathrm { c a t } _ { 0 } ( \mathrm { A G } ( X _ { r } ) ) \in \mathbb { R } ^ { N \times T }$ // gather contractionfeatures; retain tokens   
$H _ { r } \gets X _ { r } \dot { X } ^ { \top } \dot { \in } \mathbb { R } ^ { \dot { N } _ { r } \times N }$ // local Hessian row block   
$\{ H _ { p } \} _ { p = 0 } ^ { P _ { \mathrm { t p } } - 1 }  \mathrm { A G } ( H _ { r } )$ // all-gather row blocks to every rank   
$\dot { H } \longleftarrow \cot _ { 0 } ( H _ { 0 } , \dots , \dot { H } _ { P _ { \mathrm { t p } } - 1 } ) = X X ^ { \top } \in \mathbb { R } ^ { N \times N }$   
$W \gets \mathrm { c a t } _ { 1 } ( \mathrm { A G } ( W _ { r } ) ) ; \quad A \gets \mathrm { c a t } _ { 1 } ( \mathrm { A G } ( A _ { r } ) )$   
$\pi  \operatorname { a c t o r d e r } ( H ) ; \quad H ^ { \pi }  H _ { \pi , \pi }$   
$U \gets \mathrm { c h o l } ^ { \top } ( ( \dot { H } ^ { \pi } + \lambda I _ { N } ) ^ { - 1 } )$   
$\mathcal { T } _ { r } \gets \mathrm { r o w } .$ partition $( [ M ] , r , P _ { \mathrm { t p } } )$   
$( \widehat { W } ^ { ( r ) } , \widehat { A } ^ { ( r ) } ) \gets \mathrm { G P T Q } ( W _ { \mathbb { Z } _ { r } , : } , A _ { \mathbb { Z } _ { r } , : } ; U , \pi )$   
$\widehat { \cal W } \longleftarrow \mathrm { c a t 0 } ( \mathrm { A G } ( \widehat { \cal W } ^ { ( r ) } ) ) ; \widehat { \cal A } \longleftarrow \mathrm { c a t } _ { 0 } ( \mathrm { A G } ( \widehat { \cal A } ^ { ( r ) } ) )$   
$W _ { r }  \widehat { W } _ { : , \mathcal { I } _ { r } } ; \quad A _ { r }  \widehat { A } : , B _ { r }$   
Invariant : every rank uses the same $( H ^ { \pi } , U , \pi ) ;$ ; in exact arithmetic, gathering $( \widehat W ^ { ( r ) } , \widehat A ^ { ( r ) } )$ equals $\operatorname { G P T Q } ( W , A ; U , \pi )$   
Algorithm 5: The row-parallel schedule converts otherwise idle tensor-parallel ranks into independent GPTQ workers without   
approximating the objective.

## D Additional Analysis and Infrastructure Details

## D.1 Partitioned Proof of the Conditional Metric

All blocks below are taken after applying the column permutation used by GPTQ, so the processing order is $P , G$ , F and ${ \mathcal { R } } = G \cup F$ contains the trailing indices. Let $\Omega \stackrel { \mathbf { \bar { \mathbf { \Lambda } } } } { = } H ^ { - 1 } = U ^ { \top } U$ and partition the global upper-triangular facto as

$$
U = \left[ \begin{array} { c c } { { U _ { P P } } } & { { U _ { P \mathcal { R } } } } \\ { { 0 } } & { { U _ { \mathcal { R R } } } } \end{array} \right] .\tag{19}
$$

Consequently,

$$
\Omega _ { P P } = U _ { P P } ^ { \top } U _ { P P } , \quad \Omega _ { P \mathcal { R } } = U _ { P P } ^ { \top } U _ { P \mathcal { R } } , \quad \Omega _ { \mathcal { R R } } = U _ { P \mathcal { R } } ^ { \top } U _ { P \mathcal { R } } + U _ { \mathcal { R R } } ^ { \top } U _ { \mathcal { R R } } .\tag{20}
$$

Applying the block-inverse identity to $\Omega = H ^ { - 1 }$ gives the inverse of the corresponding principal block of H as the Schur complement of $\Omega _ { P P } \colon$

$$
\begin{array} { r l } & { H _ { \mathcal { R R } } ^ { - 1 } = \Omega _ { \mathcal { R R } } - \Omega _ { \mathcal { R P } } \Omega _ { P P } ^ { - 1 } \Omega _ { P \mathcal { R } } } \\ & { \quad \quad \quad = U _ { P \mathcal { R } } ^ { \top } U _ { P \mathcal { R } } + U _ { \mathcal { R R } } ^ { \top } U _ { \mathcal { R R } } - U _ { P \mathcal { R } } ^ { \top } U _ { P P } ( U _ { P P } ^ { \top } U _ { P P } ) ^ { - 1 } U _ { P P } ^ { \top } U _ { P \mathcal { R } } } \\ & { \quad \quad \quad = U _ { \mathcal { R R } } ^ { \top } U _ { \mathcal { R R } } . } \end{array}\tag{21}
$$

(22)

(23)

The last equality uses nonsingularity of the damped factor $U _ { P P }$ , for which $\begin{array} { r } { U _ { P P } ( U _ { P P } ^ { \top } U _ { P P } ) ^ { - 1 } U _ { P P } ^ { \top } = I } \end{array}$ . Thus the contribution from rows in $P$ is present in $\Omega _ { \mathcal { R R } }$ but is exactly removed by the Schur term; no commutation between principal submatrices and inversion is assumed.

Next partition the trailing curvature $H _ { \mathcal { R R } }$ by $\mathcal { R } \stackrel { } { = } G \cup F$ . Eliminating the still-free future F gives the prefixconditional group curvature

$$
S _ { G | P } = H _ { G G } - H _ { G F } H _ { F F } ^ { - 1 } H _ { F G } ,\tag{24}
$$

whose block-inverse identity is

$$
( H _ { \mathcal { R R } } ^ { - 1 } ) _ { G G } = S _ { G | P } ^ { - 1 } .\tag{25}
$$

Because $U _ { \mathcal { R R } }$ remains upper triangular,

$$
U _ { \mathcal { R R } } = \left[ \begin{array} { c c } { U _ { G G } } & { U _ { G F } } \\ { 0 } & { U _ { F F } } \end{array} \right] , \qquad ( U _ { \mathcal { R R } } ^ { \top } U _ { \mathcal { R R } } ) _ { G G } = U _ { G G } ^ { \top } U _ { G G } .\tag{26}
$$

Combining the last two displays proves $S _ { G | P } ^ { - 1 } = U _ { G G } ^ { \top } U _ { G G }$ and hence Equation 8. The block $U _ { G G }$ is extracted from the globally permuted factor after preceding elimination.

## D.2 Calibration and Quantizer Configuration

Accuracy experiments. Qwen3.5-397B-A17B and Llama-3.3-70B-Instruct use one identical calibration set. It contains 255 multi-turn documents: 85 each from science (SciQ), commonsense (CommonsenseQA and PIQA), and math ematics (NuminaMath-CoT). The production mixture is built with seed 42; the loader then applies a deterministic seed-0 shuffle. Documents target 11,000 characters (measured median 11,327), are rendered with apply chat template without an added generation prompt, and include assistant answers. Tokenization uses batch size 1, truncation/padding at 16,384 tokens, and therefore produces 255 calibration batches per model.

Candidate records are filtered against the evaluation data using normalized 13-gram overlap over prompts and gold answers. The scan covers the registered evaluation suites, and the builder fails closed if the blocklist, domain balance, chat structure, renderability, token budget, or length-parity checks fail. This guards against calibration–evaluation contamination while keeping the two model families on the same source records. Both accuracy recipes use identity pre-transforms, NVFP4 groups of 16 weights, and the same calibration set; Qwen uses TP8/EP8 and Llama uses TP1 during quantization.

Scale and Hessian settings. For tensor-wide scale γ, let $a _ { \mathrm { m a x } } = \operatorname* { m a x } _ { m , n } | W _ { m n } |$ . The implementation maps a<sub>max</sub> to the largest product of a positive E4M3 scale code (448) and the E2M1 maximum (6):

$$
\gamma = \frac { 4 4 8 \times 6 } { a _ { \operatorname* { m a x } } } , \qquad S = \{ a / \gamma : a \in \mathrm { E 4 M } 3 ^ { + } \} .\tag{27}
$$

The fixed-code proposal runs three iterations from the incoming scale. If $Q ^ { \top } S _ { G } Q \leq 1 0 ^ { - 1 2 }$ , the update retains the incoming scale; valid updates are clamped positive. Both headline arms use the same fixed replay half-window $h = 4 ,$ an engineering operating point between replay cost and local reconstruction fidelity; it is not tuned separately by model or dataset. The incoming exportable scale is always retained, and deterministic ties prefer that scale and then the smaller candidate. Hessians are accumulated in FP32, symmetrized, and damped at 1% of the mean diagonal for the production Qwen recipe; failed Cholesky factorizations trigger increasing damping rather than an identity-Hessian substitution. The active GPTQ column permutation is applied before extracting the blocks used in Equation 8.

Systems timing calibration. All three 397B timing pipelines load the same compatible Hugging Face checkpoint. Ours converts it to Megatron TP8/EP8 grouped-expert modules before quantization, and ModelOpt performs an analogous conversion within its framework, so ModelOpt is the same-representation baseline; LLM Compressor provides no comparable conversion and runs its Hugging Face-native path. Neither baseline distributes one layer’s GPTQ solve across tensor-parallel ranks, so their Megatron-side support does not shorten that layer’s critical path. The steadystate comparison uses 32 byte-identical pretokenized sequences at length 4,096 in all three frameworks; using 255 sequences at length 16,384 made the baseline Hessian passes infeasible within the study budget. The measured 70B full-model comparison uses the complete 255-by-16,384 protocol. Pretokenized artifacts record model, calibration path, sample count, maximum length, batch size, shuffle seed, batch/token counts, tokenizer class, and vocabulary size.

## D.3 Quantized Module Coverage

Common scope. For both model families, W4A4 and W4A16 quantize the same weight-bearing modules with block-16 NVFP4 weights. W4A4 additionally quantizes their input activations to block-16 NVFP4, whereas W4A16 retains BF16 activations. Embeddings, the output language-model head, normalization layers, rotary-position components, MoE routing and gating modules, multi-token-prediction heads, and vision components remain in high precision or outside the evaluated language backbone. We do not otherwise exempt the first or last transformer block.

Qwen3.5-397B-A17B. Our module coverage follows the weight-selection policy of the official Qwen3.5-397B-A17B-FP8 release (https://huggingface.co/Qwen/Qwen3.5-397B-A17B-FP8). The 60-block hybrid backbone contains 45 Gated DeltaNet (GDN) blocks and 15 full-attention blocks, with full attention at zero-based block indices 3, 7, . . . , 59. In every block, we quantize both routed-expert projections and both shared-expert projections. In each GDN block, we quantize the fused input projection and output projection; in each full-attention block, we quantize the fused QKV projection and output projection. Under TP8/EP8, this corresponds to 360 se lected modules per rank: 120 grouped routed-expert modules, 120 shared-expert projections, 90 GDN projections, and 30 full-attention projections. Two structural exceptions are important. First, the fused GDN input projection is segmented: its query, key, value, and gate rows are quantized, whereas the small beta and alpha state-gating rows remain in high precision to avoid perturbing the recurrent decay dynamics. Second, each routed expert is calibrated and quantized independently, with its own Hessian statistics and scales; scales are not shared across experts.

Llama-3.3-70B-Instruct. The 80-block dense backbone has four selected modules per block: the fused QKV projection, attention output projection, MLP up projection, and MLP down projection, for 320 dense modules in total. It has no GDN segmentation, routed experts, shared experts, or grouped-linear special case. Quantization uses TP1; TP2 is used only for inference evaluation.

Table 7: Exact correct-question counts for Qwen3.5-397B-A17B. SOAR is the strongest external quantized baseline by question-weighted recovery.
<table><tr><td>Dataset</td><td>BF16</td><td>SOAR</td><td>Schur Replay</td></tr><tr><td>MMLU-Pro (12,032)</td><td>9,155</td><td>8,970</td><td>9,072</td></tr><tr><td>GSM8K (1,319)</td><td>1,216</td><td>1,214</td><td>1,227</td></tr><tr><td>CMath (1,098)</td><td>1,008</td><td>1,019</td><td>1,012</td></tr><tr><td>LiveCodeBench (400)</td><td>179</td><td>165</td><td>169</td></tr><tr><td>MGSM-English (250)</td><td>230</td><td>223</td><td>232</td></tr><tr><td>HumanEval (164)</td><td>150</td><td>152</td><td>152</td></tr><tr><td>GPQA-Diamond (198)</td><td>109</td><td>109</td><td>104</td></tr><tr><td>Total (15,461)</td><td>12,047</td><td>11,852</td><td>11,968</td></tr><tr><td>Accuracy (%)</td><td>77.919</td><td>76.657</td><td>77.408</td></tr><tr><td>Wilson 95% CI (%)</td><td>[77.258,78.565][75.984,77.318][76.742,78.060]</td><td></td><td></td></tr></table>

Table 8: Exact correct-question counts for Llama-3.3-70B-Instruct under the same protocol as Table 7.
<table><tr><td>Dataset</td><td>BF16</td><td>SOAR</td><td>Schur Replay</td></tr><tr><td>MMLU-Pro (12,032)</td><td>6,641</td><td>6,699</td><td>6,730</td></tr><tr><td>GSM8K (1,319)</td><td>1,230</td><td>1,216</td><td>1,229</td></tr><tr><td>CMath (1,098)</td><td>951</td><td>942</td><td>946</td></tr><tr><td>LiveCodeBench (400)</td><td>124</td><td>120</td><td>117</td></tr><tr><td>MGSM-English (250)</td><td>230</td><td>228</td><td>230</td></tr><tr><td>HumanEval (164)</td><td>138</td><td>135</td><td>137</td></tr><tr><td>GPQA-Diamond (198)</td><td>75</td><td>68</td><td>79</td></tr><tr><td>Total (15,461)</td><td>9,389</td><td>9,408</td><td>9,468</td></tr><tr><td>Accuracy (%)</td><td>60.727</td><td>60.850</td><td>61.238</td></tr><tr><td>Wilson 95% CI (%)</td><td></td><td></td><td>[59.955,61.494] [60.078,61.616] [60.467,62.003]</td></tr></table>

## D.4 Benchmark Evaluation, Provenance, and Uncertainty

Decoding and prompts. Each model–dataset pair uses one deterministic greedy completion with temperature 0, top-p = 1, and at most 8,192 new tokens; the recorded seed field is inactive under temperature-0 decoding. The seven sets contain 12,032 MMLU-Pro, 1,319 GSM8K, 1,098 CMath, 400 LiveCodeBench, 250 English MGSM, 164 HumanEval, and 198 GPQA-Diamond questions, totaling 15,461. Math prompts request step-by-step reasoning and a final answer in \boxed{}; MMLU-Pro and deterministically shuffled GPQA options request only the option letter. HumanEval requests the full Python function, and LiveCodeBench requests a complete program. Every JSONL record stores idx, dataset, model, subset, sample, seed, full prediction, and gold. Across every headline arm and the Qwen window/metric/scoring ablations, (idx,sample) is unique, sample=0, and all IDs and gold fields agree with the collection-local BF16 record.

Graders and executable tasks. Scoring uses OpenCompass commit 63b0199ba6c7f251386c04b0215ea5c3016a0407, whose GSM8K, MATH, and multiple-choice postprocessors are called by our scorer. HumanEval uses the 164- record openai/openai humaneval test split: its canonical solutions match our gold fields byte for byte. Live-CodeBench uses the 400-record test.jsonl at Hugging Face revision 0fe84c3912ea0c4d4a78037083943e8f0c4dd505, covering 2023-05-07 through 2024-03-02. The executor extracts the longest fenced code block, recognizes Python or C++, compiles C++17 with -O2, and checks HumanEval unit tests or LiveCodeBench public and private stdin/stdout cases. Re-scoring BF16, SOAR, and Schur Replay exactly reproduces every reported dataset percentage for both models. A second complete execution of both code sets gives zero changes among 6,768 repeated per-question verdicts.

Macro-average sensitivity. For completeness, Table 9 reports the unweighted arithmetic mean of the seven dataset accuracies alongside the question-weighted recovery used in the main text. The macro mean can be biased as a modellevel summary in this suite: it assigns the same weight to MMLU-Pro’s 12,032 questions and to HumanEval’s 164 or GPQA-Diamond’s 198, although the latter accuracy estimates have substantially larger sampling variance. Consequently, a few outcomes on a small dataset can move the macro ranking more than their share of the 15,461 evaluated questions. We therefore treat it as a complementary task-balanced view and retain question-weighted recovery as the headline aggregate.

Table 9: Seven-dataset macro mean and question-weighted recovery (%). Macro averages datasets uniformly; W. Rec. weights their question counts and normalizes by the corresponding BF16 total. Bold marks the best quantized method in each column.
<table><tr><td>Method Qwen Macro</td><td>Qwen W. Rec.</td><td>Llama Macro</td><td>Llama W. Rec.</td></tr><tr><td>BF16</td><td>77.62</td><td>100.00</td><td>68.58 100.00</td></tr><tr><td>GPTQ</td><td>74.52</td><td>96.69 68.45</td><td>98.67</td></tr><tr><td>SmoothQuant+GPTQ</td><td>76.18</td><td>97.60 68.15</td><td>99.26</td></tr><tr><td>AWQ+GPTQ</td><td>76.65</td><td>97.69 68.65</td><td>98.85</td></tr><tr><td>Four Over Six+GPTQ</td><td>75.05</td><td>97.32 68.16</td><td>99.14</td></tr><tr><td>SOAR+GPTQ</td><td>76.80</td><td>98.38 67.27</td><td>100.20</td></tr><tr><td>ARCQuant</td><td>74.89</td><td>98.34 68.79</td><td>100.15</td></tr><tr><td>ScaleSweep</td><td>75.80</td><td>97.28 68.23</td><td>99.03</td></tr><tr><td>MR-GPTQ</td><td>73.21</td><td>93.86 66.77</td><td>91.81</td></tr><tr><td>Schur Replay (ours)</td><td>77.27</td><td>99.35 68.57</td><td>100.84</td></tr></table>

On Qwen3.5-397B-A17B, Schur Replay leads both quantized aggregates (77.27 macro; 99.35 weighted recovery). On the smaller Llama-3.3-70B-Instruct model, the macro scores are tightly clustered: ARCQuant reaches 68.79, AWQ+GPTQ 68.65, and Schur Replay 68.57, a 0.22-point span between the first and third methods, while Schur Replay leads question-weighted recovery at 100.84. This modest separation is consistent with a prior cross-scale quantization evaluation, which reports that accuracy and method rankings depend on model size, task, and bit width and shows several smaller-model settings with close aggregate scores across quantizers [Lee et al., 2024]. It is descriptive evidence rather than a monotone scaling claim: architecture and task mix also differ between our 70B dense and 397B MoE models.

Paired uncertainty. We align methods by dataset and question ID, and bootstrap the paired correctness difference with 100,000 resamples and fixed seed 20,260,907. On Qwen, Schur Replay exceeds SOAR by 0.750 percentage points; the 95% paired-bootstrap interval is [0.220, 1.281] points, and the exact two-sided McNemar test over 786 SOAR-only versus 902 Schur-only successes gives p = 0.0051. On Llama, Schur Replay exceeds SOAR by 0.388 points, with an 80% paired-bootstrap interval of [0.013, 0.763] points. Recovery remains collection-local: exact counts give 99.344%/100.841% for Schur Replay on Qwen/Llama, consistent with the rounded headline values.

![](images/ec2024ed619cc631cf83081113c5ceb8bb9bdc5ba117228d9d17bb09b9aa0001.jpg)

![](images/045fd621c9277b7075a4059361266da380cdd97cff2cddb6abda93680eaa1ba9.jpg)

![](images/a69e17a325caff2e1219032bab60f72d155fecdf5fe3f6cd9e3db5d639919480.jpg)  
Figure 4: Exhaustive audit of the Schur Replay window over all 126 exportable scales. Percentages exclude flat rows for which every candidate ties. (a) Signed distance from the proposal center to the nearest optimum of the conditional replay surrogate. (b) Fraction of rows whose local window contains such an optimum. (c) Layer-wise coverage at $h = 4 .$ . The audit contains 143.5M live rows from Qwen3.5-397B-A17B and 85.5M from Llama-3.3-70B-Instruct.

## D.5 Exhaustive Exportable-Scale Audit

We score the complete 126-point exportable grid for every sampled group and compare its row-wise global optimum with the proposal-centered window. The audit covers all 60 decoder layers of Qwen3.5-397B-A17B (MoE, TP8/EP8; 69,016 groups and 143.5M non-flat rows) and all 80 layers of Llama-3.3-70B-Instruct (dense, TP1; 5,324 groups and 85.5M non-flat rows). Let d be the signed grid distance from proposal center c to the nearest exhaustive optimum. The optimum lies at c for 28.44%/28.90% of rows, within one step for 57.47%/58.34%, within four for 83.28%/83.53%, and within eight for 99.965%/99.989% on Qwen/Llama. At h = 4, layer-wise coverage remains between 82.79– 85.83% on Qwen and 83.10–85.06% on Llama, showing that the aggregate is not dominated by a few layers.

These coverage numbers do not order end-to-end accuracy. Widening from h = 4 to h = 8 raises coverage of the surrogate optimum from 83.28% to 99.965% and doubles the replayed neighborhood, while macro recovery changes from 98.92% to 98.62% in this sensitivity sweep. The two quantities measure different properties: coverage concerns the calibration-defined surrogate, whereas recovery is measured on downstream tasks. This audit characterizes the proposal and sensitivity around the fixed h = 4 engineering operating point; neither coverage nor downstream benchmark outcomes were used as a post-hoc selection rule. The sweep does not establish a monotone relation between search width and downstream quality. Equation 15 applies only to the per-group surrogate because the incoming scale remains available; it makes no corresponding guarantee for downstream accuracy.

## D.6 Layer-Wise Recovery and Numerical Validation

Calibration-state scaling stress test. Figure 5 evaluates activation and storage scaling on a mock Qwen3.5 transformer layer with the layer width of Qwen3.5-397B-A17B under TP8. The test uses one eight-GPU node with 72 GB of device memory per GPU, 3.5 TB of local SSD capacity, and a mounted HDFS-backed network volume. Sequence length is fixed at 16,384 while the number of calibration blocks increases from 32 to 8,192. The plotted times are the observed elapsed times of the reported runs; a cross denotes a run that did not complete under host-memory pressure and is not a timing value.

At 32 blocks, the optimized path takes 8.51 seconds, compared with 12.08 seconds for ModelOpt TP8 and 16.48 seconds for LLM Compressor. At 128 blocks, the corresponding times are 10.63, 40.53, and 46.46 seconds. At 512 blocks, ours completes in 46.89 seconds and LLM Compressor in 595.56 seconds, while ModelOpt fails under host-memory pressure. At 8,192 blocks, only ours completes, in 1,080.9 seconds; both public baselines fail under host-memory pressure. These measurements isolate a Qwen-width mock layer rather than an end-to-end model run, and support the infrastructure claim that tiered activation storage sustains calibration states beyond the host-memory envelope reached by the evaluated baselines.

The layer-wise forward uses three states. Layers older than the immediate predecessor become parameter-free skip modules that reconstruct zero-valued outputs from lightweight metadata; the preceding layer runs on cached inputs;

![](images/fa45c0ea2a018af8414259ac1260b2a82ab0b5f3d7fb380336326ccecd3605dd.jpg)

![](images/b6294ee137be9273d7913539d488b200810e94885be792a4e0a76fd110ef9d47.jpg)  
Figure 5: Scaling of a mock Qwen3.5 transformer layer with Qwen3.5-397B-A17B layer width under TP8 as calibra tion blocks increase at sequence length 16,384. The node has eight 72 GB GPUs, 3.5 TB of local SSD capacity, and mounted HDFS storage. (a) Scaling curves show completed elapsed-time measurements. (b) Grouped bars show the same measurements for direct comparison at each workload size. Both panels use logarithmic time axes; crosses mark runs that did not complete under host-memory pressure and are not assigned synthetic times or bar heights. Ours is faster for every jointly completed setting and is the only evaluated path that completes the 8,192-block stress test.

and the current layer captures its inputs and terminates the model forward. This state machine preserves the active-layer residency bound even under framework hooks that would otherwise move parameters automatically. After packing, each retired parameter is replaced by a zero-element tensor rather than a meta tensor, so accidental reuse fails instead of propagating meta-valued outputs.

After each layer is calibrated, the system saves its weights, quantizer state, and output metadata. A window is committed only after the next-layer inputs are written and an atomic manifest records the last complete boundary. Resume validates the model depth and checkpoint interval to detect configuration drift. Each real-NVFP4 fragment and its source checkpoint, when present, carry SHA-256 digests; finalization memory-maps validated fragments, rejects duplicate module keys or conflicting absorbed tensors, builds the artifact locally, and verifies its digest after transfer. Distributed minimum and maximum reductions over completed-layer counts expose rank disagreement before final ization while retaining valid fragments for a resumed run.

All relocation, batching, graph replay, spill, checkpoint, and distributed-solve paths are checked against their reference implementations. Hessian inversion symmetrizes the matrix, handles structurally zero columns, and increases diagonal damping when Cholesky factorization fails; an unresolved failure invalidates the solve rather than substituting an identity Hessian. Because fused FP32 Hessian accumulation changes summation order, validation uses tight relativeerror checks and requires identical sample counts and activation-order permutations.

## E Sample Outputs

The examples below are organized by dataset. Within each benchmark, the Qwen3.5-397B-A17B disagreement is followed immediately by the Llama-3.3-70B-Instruct disagreement, and each model block compares SOAR, Four Over Six, and Schur Replay. We preserve archived generations verbatim apart from typography and explicitly marked omissions. When a Qwen generation reaches its length limit inside the reasoning trace, we label it incomplete rather than reconstructing a final answer.

## E.1 MMLU-Pro disagreements

Qwen3.5-397B-A17B: MMLU-Pro, question 5798: reversing the good/bad percentages. The question asks what percentage of Germans described the relationship between the United States and Germany as good in 2019. The keyed option is A (34%); option C gives 64%. All three archived predictions contain closed reasoning traces.

SOAR (×), CoT excerpt. “The user is asking for a specific statistic from 2019 regarding German public opinion on the relationship between the U.S. and Germany. [source-identification discussion omitted] A key finding from the 2019 Pew survey: ‘About two-thirds of Germans (64%) say relations between the United States and Germany are good.’ [checking omitted] If 64% said good, 34% said bad. Option A is 34%. This suggests the question is derived directly from this specific dataset where the distractors include the inverse statistic. [final recheck omitted] Option A: 34% (The % who said it was bad). Option C: 64% (The % who said it was good). The question asks for ‘good’. So, C.” Emitted answer: C.

Four Over Six (×), CoT excerpt. “The user is asking for a specific statistic from 2019 regarding German public opinion on the relationship between the U.S. and Germany. [source-identification discussion omitted] In 2018, 79% said it was good. In 2019, it dropped to 64%. [checking omitted] Pew Research Center, ‘Transatlantic Relations’: 2017: 83%; 2018: 79%; 2019: 64%. This seems consistent. Therefore, the answer is C. [option-by-option check omitted] Conclusion: C is the correct answer.” Emitted answer: C.

Schur Replay (✓), CoT excerpt. “The user is asking for a specific statistic from 2019 regarding German public opinion on US–Germany relations. [source-identification discussion omitted] ‘Just 34% of Germans describe relations between the U.S. and Germany as good.’ This seems to be the specific data point. Therefore, Option A is the correct answer. [checking omitted] Conclusion: 34% is the most likely intended answer based on the prominent Pew Research data from that year. [option check omitted] Good: 34%. Bad: 64%. Answer: A.” Emitted answer: A.

Observed disagreement. The two baselines settle on the reversed 64% statistic, whereas Schur Replay distinguishes the keyed good/bad percentages and emits A.

Llama-3.3-70B-Instruct. Question 122 (business). Which of the following is not one of the underlying principles of the corporate governance Combined Code of Practice?

Options. A: Integrity; B: Adaptability; C: Accountability; D: Openness; E: Dependability; F: Acceptability.

<table><tr><td></td><td>Gold</td><td>SOAR</td><td>Four Over Six</td><td>Schur Replay</td></tr><tr><td>Emitted letter</td><td>F</td><td>E×</td><td>E×</td><td>F√</td></tr></table>

Question 6224 (health). A deficiency of which vitamin has been associated with enamel defects and increased risk of dental caries?

Options. A: Vitamin D; B: Vitamin K; C: Riboflavin; D: Niacin; E: Biotin; F: Vitamin A; G: Vitamin E; H: Vitamin C; I: Vitamin B12; J: Folic acid.

<table><tr><td></td><td>Gold</td><td>SOAR</td><td>Four Over Six</td><td>Schur Replay</td></tr><tr><td>Emitted letter</td><td>A</td><td>F×</td><td>F ×</td><td>A√</td></tr></table>

Question 9028 (physics). When the shadow of the Moon falls on Earth, we have a:

Options. A: lunar eclipse; B: lunar eclipse if daytime and solar eclipse if nighttime; C: very dangerous event; D: planetary alignment; E: celestial phenomenon; F: supernova event; G: solar eclipse if daytime and lunar eclipse if nighttime; H: solar flare event; I: solar eclipse; J: moonlight night.

<table><tr><td></td><td>Gold</td><td>SOAR</td><td>Four Over Six</td><td>Schur Replay</td></tr><tr><td>Emitted letter</td><td>I</td><td>G×</td><td>G ×</td><td>I√</td></tr></table>

Observed difference. Across three categories, both baselines converge to the same distractor while Schur Replay emits the keyed option. For these three particular Llama records, each displayed letter is the complete archived generation; no reasoning text was stored for these records. This does not imply that every Llama MMLU-Pro prediction is short, whereas the Qwen MMLU-Pro case above explicitly preserves its archived reasoning excerpts.

## E.2 GSM8K disagreements

Qwen3.5-397B-A17B: GSM8K, question 1019: whether late checkouts have departed. A hotel starts with 100 guests. Twenty-four check out early and 15 choose late checkout. In the afternoon, twice as many people check in as chose late checkout, and seven more check in after dinner. The gold answer is 98.

SOAR (×), CoT excerpt. “Initial state: Hotel has 100 guests. [intervening deliberation omitted] A ‘late checkout’ implies that these guests are staying in their rooms longer than the standard checkout time. They have not left the hote yet. [arithmetic omitted] $1 0 0 - 2 4 + 3 0 + 7 = 1 1 3 . ^ { \circ }$ Its final response repeats this interpretation and emits 113

Four Over Six (✓), CoT excerpt. “Initial state: Hotel has 100 guests. [intervening deliberation omitted] Since the timeline of the problem extends to ‘after dinner’ (evening), guests with a late checkout (typically 1 PM–4 PM) would have left by this time. [arithmetic omitted] Result: 98.” Its final response computes $1 0 0 - 2 4 - 1 5 + 3 0 + 7$ and emits 98 .

Schur Replay (✓), CoT excerpt. “Initial state: Hotel has 100 guests. [analysis omitted] 24 guests check out early. [step omitted] 15 guests check out late. Since the question asks for the count ‘now,’ after the sequence of events including the afternoon and dinner, these guests have departed.” Its final response obtains $6 1 + 3 0 + 7$ and emits 98 . Observed disagreement. SOAR treats the late-checkout group as still present at the final time and obtains 113; Four Over Six and Schur Replay account for their departure and recover the keyed 98.

Llama-3.3-70B-Instruct. Question. John buys 2 pairs of shoes for each of his 3 children. They cost \$60 each. How much did he pay?

Gold derivation. He bought $3 \times 2 = 6$ pairs of shoes, so he spent $6 0 \times 6 = \$ 36 0$

SOAR (×), complete generation. To find the total amount John paid, we need to calculate the total number of shoes he bought and then multiply it by the cost of each shoe. (1) John buys 2 pairs of shoes for each of his 3 children. Since each pair contains 2 shoes, the total number of shoes per child is $2 \times 2 = 4$ shoes. (2) John has 3 children, so the total number of shoes he buys is $4 \times 3 = 1 2$ shoes. (3) Each shoe costs \$60, so the total amount John paid is $1 2 \times \mathfrak { F } 6 0 = \mathfrak { G } 7 2 0$ . Therefore, the total amount John paid is 720 .

Four Over Six (×), complete generation. To find the total amount John paid, we need to calculate the total number of shoes he bought and then multiply it by the cost of each shoe. Step 1: John buys 2 pairs of shoes for each of his 3 children. Since each pair contains 2 shoes, the total number of shoes per child is 2 pairs × 2 shoes/pair = 4 shoes. For 3 children, the total number is 4 shoes/child × 3 children = 12 shoes. Step 2: Each shoe costs \$60. Therefore, the total amount John paid is 12 shoes × \$60/shoe = \$720. The final answer is 720 .

Schur Replay (✓), complete generation. To find the total amount John paid, we need to calculate the total number of shoes and then multiply by the cost of each pair. Step 1: He buys 2 pairs of shoes for each of his 3 children, so the total number of pairs is $2 \times 3 = 6$ pairs. Step 2: Each pair costs \$60, so the total cost is $6 \times \ S 6 0 = \ S 3 6 0$ . Therefore, the total amount John paid is \$360. The final answer is 360 .

Observed difference. The two baselines change the unit from pairs to individual shoes before applying the \$60 price, doubling the result. Schur Replay preserves the pair-level unit through both arithmetic steps.

## E.3 CMath disagreements

Qwen3.5-397B-A17B: CMath, question 1085: completed answer versus a truncated reasoning loop. The school has 893 students and each uniform costs 88 yuan; the requested estimate is rounded to the nearest ten thousand yuan. The gold answer is 80,000.

SOAR (✓), CoT excerpt. “Total Cost = (Number of Students) × (Cost per Set). Calculation: 893 × 88. [calculation omitted] 78584 → 80000. [rounding check omitted] So 80,000 is the correct answer.” Its final response shows the exact multiplication and emits “8万” (or 80000).

Four Over Six (incomplete), CoT excerpt. “Given: Number of students: 893; Cost per suit: 88 yuan. [calculation omitted] Let’s do the exact calculation to be sure, then round.” The trace reaches the same 80,000 conclusion internally, but its archived tail repeats “Wait, I’ll write 80000” until the generation limit. It contains no closing </think> and no emitted final response.

Schur Replay (✓), CoT excerpt. $^ { * * } 8 9 3 \times 8 8 = 7 8 5 8 4$ . Rounding: 78584 → 80000. Estimation: $9 0 0 \times 9 0 =$ 81000 → 80000. Both match. Final Answer: 80000.” Its final response gives both exact and estimated calculations and emits 80000

Observed disagreement. SOAR and Schur Replay finish their reasoning and emit the gold-equivalent value; Four Over Six enters a repetitive reasoning loop and exhausts the archived generation budget before producing an answer.

Llama-3.3-70B-Instruct. Question. 全球荒漠化土地的面积已占陆地总面积的四分之一，而且正在以每年约六万平方千米的速度扩展。地球陆地总面积约为三千六百万平方千米，如果不制止，多少年后地球上将连一根草都看不到？Gold: <sub>4</sub>5<sub>0</sub>。

SOAR (×), complete generation. 为了解决这个问题，我们需要计算出荒漠化的速度和全球陆地总面积之间的关系。

1. 全球陆地总面积=3600万平方千米。

2. 荒漠化土地面积占陆地总面积的1/4，即荒漠化土地面积 = 3600万 × 1/4= 900万平方千米。

3. 每年荒漠化的速度=6万平方千米（平均值）。

<sub>4.</sub> 问题中提到，如果不制止，多少年后地球上连一根草都看不到了，也就是说，所有的陆地都会变成荒漠。

要计算出需要多少年后地球上所有陆地都会变成荒漠，我们需要计算出剩余的陆地面积（非荒漠化的陆地面积）是多少，然后除以每年的荒漠化速度。剩余的陆地面积 $= 3 6 0 0 \mathcal { T } \mathrm { ~ - ~ } 9 0 0 \mathcal { T } = 2 7 0 0$ 万平方千米。年数 =<sub>2700</sub>万 <sub>/ 6</sub>万 <sub>= 4500</sub>年。因此，答案是： 4500 。

Four Over Six (×), complete generation. 为了解决这个问题，我们需要计算荒漠化的速度和全球陆地面积。  
1. 全球陆地总面积=3600万平方千米。

2. 荒漠化土地面积 = 1/4 × 3600万 = 900万平方千米。

3. 每年荒漠化的面积=6万平方千米。

现在，我们需要计算如果不制止荒漠化，多少年后地球上所有陆地都会变成荒漠。剩余陆地面积 = 全球陆地总面积 − 荒漠化土地面积 $= 3 6 0 0 \mathcal { T } - 9 0 0 \mathcal { T } = 2 7 0 0 \mathcal { T }$ 平方千米。由于每年荒漠化的面积为6万平方千米，年数 = 2700万 / 6万 = 4500年。因此，如果不制止荒漠化，4500年后地球上所有陆地都会变成荒漠。最终答案是： 4500 。

Schur Replay (✓), complete generation. 要解决这个问题，我们需要按照步骤计算出如果荒漠化以每年<sub>6</sub>万平方千米的速度扩展，需要多少年后，地球上所有陆地都会变成荒漠。

1. 计算地球陆地总面积：已知陆地总面积为3600万平方千米。

2. 计算当前荒漠化土地面积：荒漠化土地面积占陆地总面积的1/4，即 $\begin{array} { r } { \frac { 1 } { 4 } \times 3 6 0 0 \overline { { \mathcal { I } } } = 9 0 0 } \end{array}$ 万平方千米。

3. 计算尚未荒漠化的土地面积：尚未荒漠化的土地面积为总面积减去已荒漠化的面积，即 $3 6 0 0 \mathcal { F } - 9 0 0 \mathcal { F } =$

2700万平方千米。

4. 计算荒漠化扩展速度：每年扩展6万平方千米。

<sub>5.</sub> 计算需要多少年使所有陆地变成荒漠：将尚未荒漠化的土地面积除以每年扩展的面积，即 $\frac { 2 7 0 0 \mathcal { T } } { 6 \mathcal { T } } = 4 5 0$ 年。

因此，如果不采取措施制止荒漠化的扩展，450年后，地球上所有陆地都可能变成荒漠。答案： 450 。

Observed difference. All three generations set up the same remaining area and division, but both baselines introduce a factor-of-ten error in the final quotient. Schur Replay evaluates the quotient consistently.

## E.4 LiveCodeBench disagreements

Qwen3.5-397B-A17B: LiveCodeBench, question 256: direct indegree versus transitive closure. The task asks whether partial pairwise “stronger than” relations uniquely identify the strongest programmer. Because the archive contains no per-question execution verdict for this Qwen record, we report the program disagreement without assigning correctness marks.

SOAR, CoT excerpt. “The ‘strongest programmer’ is the one who is stronger than everyone else. [analysis omitted] This corresponds to nodes with an in-degree of 0. If there is exactly one such person, they must be the strongest.” The emitted implementation counts only direct incoming edges:

```python
in_degree = [0] <sub>*</sub> (N + 1)
for _ in range(M):
u = int(next(iterator))
v = int(next(iterator))
in_degree[v] += 1
candidates = []
for i in range(1, N + 1):
if in_degree[i] == 0:
candidates.append(i)
print(candidates[0] if len(candidates) == 1 else -1)
```

Four Over Six, CoT excerpt. “We need to determine if there is a unique strongest programmer based on the given information. [analysis omitted] Find all nodes with in-degree 0; if unique, print it.” Its emitted implementation likewise uses only direct indegree.

Schur Replay, CoT excerpt. “The relation ‘stronger than’ is a strict total order. [analysis omitted] We need to determine if there is a unique person who must be the strongest programmer consistent with all given information.” Its emitted implementation first computes reachability:

```python
for k in range(N):
for i in range(N):
for j in range(N):
if reach[i][k] and reach[k][j]:
reach[i][j] = True
candidates = [i + 1 for i in range(N)
if all(not reach[j][i] for j in range(N) if i != j)]
print(candidates[0] if len(candidates) == 1 else -1)
```

Observed disagreement. The baselines reason over direct indegree, whereas Schur Replay explicitly materializes the transitive closure before testing whether a unique strongest candidate remains.

Llama-3.3-70B-Instruct. Problem. There are N programmers and a strict, total, transitive superiority relation. The input provides M facts $A _ { i } > B _ { i }$ . Output the unique programmer that the supplied facts determine to be strongest; output −1 if multiple strongest programmers remain possible. Here $2 \leq N \leq 5 0$ and $0 \le M \le N ( N - 1 ) / 2$ . The public examples include a chain that uniquely identifies programmer 1 and a fork for which the answer is −1.

Acceptance condition. The complete programs below are executed against the benchmark’s public and private stdin/stdout tests. SOAR and Four Over Six fail; Schur Replay passes.

SOAR (×), complete generation.

```python
from collections import defaultdict, deque
def find_strongest_programmer(n, m, edges):
graph = defaultdict(list)
in_degree = [0] (n + 1)
for u, v in edges:
graph[u].append(v)
in_degree[v] += 1
queue = deque([i for i in range(1, n + 1)
if in_degree[i] == 0])
while queue:
u = queue.popleft()
for v in graph[u]:
in_degree[v] -= 1
if in_degree[v] == 0:
queue.append(v)
candidates = [i for i in range(1, n + 1)
if in_degree[i] == 0]
if len(candidates) > 1:
return -1
else:
return candidates[0]
n, m = map(int, input().split())
edges = [tuple(map(int, input().split())) for _ in range(m)]
print(find_strongest_programmer(n, m, edges))
```

The topological pass mutates every reachable indegree to zero, so the final candidate list no longer identifies the original unique source.

Four Over Six (×), complete generation.

```python
from collections import defaultdict
def find_strongest_programmer(n, m, relations):
graph = defaultdict(list)
for a, b in relations:
graph[a].append(b)
def dfs(node, visited):
if node in visited:
return False
visited.add(node)
for neighbor in graph[node]:
if not dfs(neighbor, visited):
return False
return True
strongest = None
for i in range(1, n + 1):
visited = set()
if dfs(i, visited) and len(visited) == n:
if strongest is not None:
return -1
strongest = i
return strongest
n, m = map(int, input().split())
relations = [tuple(map(int, input().split())) for _ in range(m)]
print(find_strongest_programmer(n, m, relations))
```

A revisit is treated as failure. Converging paths can therefore reject a valid strongest candidate.

Schur Replay (✓), complete generation.

```python
import sys
def solve():
N, M = map(int, sys.stdin.readline().split())
graph = [[False] <sub>*</sub> N for _ in range(N)]
for _ in range(M):
A, B = map(int, sys.stdin.readline().split())
A -= 1
B -= 1
graph[A][B] = True
for k in range(N):
```

```python
for i in range(N):
for j in range(N):
if graph[i][k] and graph[k][j]:
graph[i][j] = True
for i in range(N):
if all(graph[i][j] for j in range(N) if i != j):
if sum(all(graph[i][j] for j in range(N)
if i != j) for i in range(N)) > 1:
print(-1)
return
print(i + 1)
return
print(-1)
if __name__ == "__main__":
solve()
```

Observed difference. Schur Replay computes transitive reachability and tests whether a vertex reaches every other vertex, matching the task definition and passing all recovered tests.

## E.5 MGSM-English disagreements

Qwen3.5-397B-A17B: MGSM-English, question 157: completed arithmetic versus a truncated trace. Blake runs back and forth 15 times on a 100-yard field. Kelly runs back and forth once and then to the 40-yard line and back 34 times. The gold difference is 80 yards.

SOAR (✓), CoT excerpt. “One ‘back and forth’ is $1 0 0 + 1 0 0 = 2 0 0$ yards. [steps omitted] Blake’s Distance = $1 5 \times 2 0 0 = 3 0 0 0$ yards. [steps omitted] Kelly’s total distance is $2 0 0 + 3 4 \times 8 0 = 2 9 2 0 { \mathrm { y a r d s } } . ^ { \prime \prime }$ The final response emits 80 .

Four Over Six (incomplete), CoT excerpt. “Blake’s Activity: Runs back and forth 15 times. Kelly’s Activity: [problem decomposition omitted] $1 5 \times 2 0 0 = 3 0 0 0 ; 3 4 \times 8 0 = 2 7 2 0 ; 2 0 0 + 2 7 2 0 = 2 9 2 0 ; 3 0 0 0 - 2 9 2 0 = 8 0 ; ^ { \prime }$ The archive then continues checking wording and field conventions until its length limit; it contains no closing </think> and no final response.

Schur Replay (✓), CoT excerpt. “One ‘back and forth’ = 100 + 100 = 200 yards. Blake does this 15 times. [steps omitted] Kelly’s Part 2 is 34 × 80 = 2720 yards.” Its final response totals 3,000 and 2,920 yards and emits 80

Observed disagreement. SOAR and Schur Replay complete essentially the same correct calculation; Four Over Six reaches 80 inside its CoT but does not exit the reasoning channel or emit a final answer.

Llama-3.3-70B-Instruct. Question. Jerry rolls a six-sided die. How much more likely, expressed as a percentage, is rolling a number greater than 3 than rolling two even numbers in a row? Gold: 25.

SOAR (×), complete generation. Step 1: The favorable outcomes greater than 3 are 4, 5, and 6, so the probability is $3 / 6 = 1 / 2$ . Step 2: The even outcomes are 2, 4, and 6, so one even roll has probability $3 / 6 = 1 / 2$ and two independent even rolls have probability $( 1 / 2 ) ( 1 / 2 ) = 1 / 4$ . Step 3: The difference is $1 / 2 - 1 / 4 = 1 / 4$ . Step 4: To express this difference as a percentage, the generation divides it by the probability of two even rolls: $( 1 / 4 ) / ( 1 / 4 ) \times 1 0 0 = 1 0 0 \%$ Step 5: It interprets this as rolling a number greater than 3 being 100% more likely, or twice as likely. The final answer is 100 .

Four Over Six (×), complete generation. Step 1: The numbers greater than 3 are 4, 5, and 6, so their probability is $3 / 6 = 1 / 2 = 0 . 5$ . Step 2: The even numbers are 2, 4, and 6, so the probability of two even rolls is $0 . 5 \times 0 . 5 = 0 . 2 5$ Step 3: The difference is $0 . 5 - 0 . 2 5 = 0 . 2 5$ . Step 4: The generation writes $( 0 . 2 5 / 0 . 2 5 ) \times 1 0 0 = 1 0 0 \%$ , then repeats $( ( 0 . 5 - 0 . 2 5 ) / 0 . 2 5 ) \times 1 0 0 = 1 0 0 \%$ as the “correct percentage increase.” It concludes that the first event is twice as likely. The final answer is 100

Schur Replay (✓), complete generation. Step 1: The favorable outcomes greater than 3 are 4, 5, and 6. Out of six outcomes, this gives $3 / 6 = 1 / 2$ . Step 2: There are three even outcomes, so two consecutive even rolls have probability $( 3 / 6 ) ( 3 / 6 ) = 9 / 3 6 = 1 / 4 .$ . Step 3: Subtracting gives $1 / 2 - 1 / 4 = 1 / 4$ . Step 4: Expressing each probability as a percentage gives $5 0 \% - 2 5 \% = 2 5 \%$ . The final answer is 25 %.

Observed difference. All methods derive the same event probabilities. The baselines convert their absolute difference into a relative increase, whereas Schur Replay reports the percentage-point difference expected by the benchmark.

## E.6 HumanEval disagreements

Qwen3.5-397B-A17B: HumanEval, question 22: whether booleans count as integers. The task is to filter a heterogeneous Python list, retaining integer values. The canonical body is [x for x in values if isinstance(x, int)]. The disagreement concerns Python’s rule that bool is a subclass of int.

SOAR, CoT excerpt. “In Python, bool is a subclass of int, so isinstance(True, int) returns True. [deliberation omitted] To be safe and exclude booleans, I should add and not isinstance(value, bool).” It emits:

```lua
return [value for value in values
if isinstance(value, int) and not isinstance(value, bool)]
```

Four Over Six, CoT excerpt. “Check if the element is an instance of int. Important: In Python, bool is a subclass of int. [deliberation omitted] I’ll stick with isinstance.” It emits the canonical rule:

return [x for x in values if isinstance(x, int)]

Schur Replay (incomplete), CoT excerpt. “For each item, check if it is an instance of int. [deliberation omitted] What about booleans? In Python, bool is a subclass of int. [further checking omitted] I’ll stick with isinstance.” The trace remains in this repeated edge-case analysis until the archive limit and contains no closing </think> or emitted implementation.

Observed disagreement. SOAR deliberately departs from the canonical isinstance semantics by excluding booleans; Four Over Six emits the canonical implementation; Schur Replay reasons toward the canonical rule but does not complete the generation. We report this as a program/completion disagreement rather than inferring a hidden final program.

Llama-3.3-70B-Instruct. Function requirement. Implement same chars(s0, s1) so that it returns true exactly when two words contain the same distinct characters. The official prompt includes examples with unequal multiplicities, such as same chars("abcd", "dddddddabc") == True.

SOAR (×), complete implementation.

```python
def same_chars(s0: str, s1: str) -> bool:
"""Check if two words have the same characters."""
return sorted(s0) == sorted(s1)
```

Four Over Six (×), complete implementation.

```python
def same_chars(s0: str, s1: str) -> bool:
"""Check if two words have the same characters."""
return sorted(s0) == sorted(s1)
```

Schur Replay (✓), complete implementation.

```python
def same_chars(s0: str, s1: str) -> bool:
"""Check if two words have the same characters."""
return set(s0) == set(s1)
```

Official distinguishing tests.

assert candidate(’eabcdzzzz’, ’dddzzzzzzzddeddabc’) == True   
assert candidate(’abcd’, ’dddddddabc’) == True   
assert candidate(’dddddddabc’, ’abcd’) == True   
assert candidate(’eabcd’, ’dddddddabc’) == False   
assert candidate(’abcd’, ’dddddddabcf’) == False   
assert candidate(’eabcdzzzz’, ’dddzzzzzzzddddabc’) == False   
assert candidate(’aabb’, ’aaccc’) == False

Observed difference. Sorting compares both identity and multiplicity, so both baseline programs fail the positive unequal-multiplicity tests. Schur Replay implements the required set semantics and passes the complete recovered test suite.

## E.7 GPQA-Diamond disagreements

Qwen3.5-397B-A17B: GPQA-Diamond, question 101: biological regulation versus a reading-frame defect. The question asks why a Cre-dependent receptor–eGFP construct shows no green signal. The keyed option is A, a post-excision reading-frame mismatch.

SOAR (×), CoT excerpt. “The discrepancy lies between the successful protein detection in astrocytes (in vitro) and the lack of fluorescence in neural crest-derived cells (in vivo). [biological analysis omitted] Option D suggests they are in a paracrine relationship. [elimination omitted] Thus, the paracrine nature of the pair explains why forcing autocrine expression leads to signal loss.” Emitted answer: D.

Four Over Six (incomplete), CoT excerpt. “Analyze the Construct Design: [long construct analysis omitted] Why would they miss it? Because they relied on the antibody. This narrative supports Option A. [continued comparison with the paracrine option omitted]” The archived trace reaches its length limit while comparing A and D; it contains no closing </think> and no final option.

Schur Replay $( { \check { \sqrt { \mathbf { \alpha } } } } ) ,$ CoT excerpt. “After excision, one lox site (34 base pairs) remains at the junction between the Receptor and eGFP sequences. [derivation omitted] Since 34 is not divisible by 3, the remaining lox site sequence introduces a 1-base-pair frameshift. [consequence analysis omitted] Conclusion: Option A.” Its final response explains the frameshift and emits A.

Observed disagreement. SOAR selects a biological down-regulation account (D), Four Over Six never commits to a final option, and Schur Replay identifies the construct-level frame defect and emits the keyed A.

Llama-3.3-70B-Instruct. Question. Consider

$$
\rho = \frac { 1 } { 2 } \left( | 0 \rangle \langle 0 | + | 1 \rangle \langle 1 | \right) .
$$

What is its geometrical position in the qubit space?

Options. A: $r = ( 0 , 0 , 0 ) ; \mathbf { B } \colon r = ( 1 , 1 , 0 ) ; \mathbf { C } \colon r = ( 1 , 1 , 1 ) ; \mathbf { D } \colon r = ( 0 , 0 , 1 )$

Gold rationale. Since $| 0 \rangle = ( 1 , 0 ) ^ { \top }$ and $| 1 \rangle = ( 0 , 1 ) ^ { \top }$ , the state is

$$
\rho = { \frac { 1 } { 2 } } { \binom { 1 } { 0 } } \ 1 \rho \sp { 0 } ) = { \frac { 1 } { 2 } } I .
$$

Comparing this with $\rho = ( I + r \cdot \sigma ) / 2 \mathrm { g i v e s } r = ( 0 , 0 , 0 )$ , so the gold option is A.

SOAR (×), complete generation: D

Four Over Six (×), complete generation: D.

Schur Replay (✓), complete generation. Step 1: Understand the given density matrix. The given density matrix represents a mixed state where the qubit has an equal probability of being in state |0⟩ and state |1⟩.

Step 2: Recall the Bloch sphere representation. A qubit state can be represented by a point with coordinates $( x , y , z )$ related to the expectation values of the Pauli operators $\sigma _ { x } , \sigma _ { y }$ , and $\sigma _ { z }$ .

Step 3: Calculate the expectation values. For the given density matrix,

$$
\langle \sigma _ { z } \rangle = \frac { 1 } { 2 } \big ( \langle 0 | \sigma _ { z } | 0 \rangle + \langle 1 | \sigma _ { z } | 1 \rangle \big ) = \frac { 1 } { 2 } ( 1 - 1 ) = 0 .
$$

The expectation values of $\sigma _ { x }$ and $\sigma _ { y }$ are both 0 because the density matrix is diagonal in the computational basis and these operators introduce off-diagonal terms.

Step 4: Determine the geometrical position. Since the expectation values of $\sigma _ { x } , \sigma _ { y }$ , and $\sigma _ { z }$ are all $0 ,$ the point representing this state is at the center of the Bloch sphere. The final answer is $\boxed { A } .$

Observed difference. The two baselines emit the pure-state north-pole distractor D. Schur Replay retains the archived derivation from the density matrix to the vanishing Bloch vector and selects A.