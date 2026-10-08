# HySPE: Positional Encoding via Symplectic Dual Shears

Zhongping Ji

## Abstract

We introduce Hyperbolic Symplectic Positional Encoding (HySPE), grounding positional attention in noncompact symplectic transformations. While canonical Rotary Position Embedding (RoPE) parameterizes the compact, elliptic branch of Sp(2, R) via rotations, HySPE operationalizes its hyperbolic branch via a damped symmetric composition of dual shears, yielding a conformally symplectic contraction with two spectral decay rates per channel pair. To eliminate the exponential representation drift inherent to naive absolute factorizations, we diagonalize the operator in its invariant eigenbasis and introduce blockwise coordinate rebasing with adaptive centered execution. This guarantees length-independent numerical bounds while matching cached RoPE forward latency (7.21 ms on an RTX 4090). On TinyShakespeare, HySPE-UltraLong maintains an invariant perplexity of 4.810 up to 16× zero-shot extrapolation $( L = 4 0 9 6 )$ , whereas RoPE degrades to 131.198. Scaled to a 51M-parameter subword Transformer on WikiText-103 $( L _ { \mathrm { t r a i n } } = 5 1 2 )$ , HySPE closely matches RoPE in-domain while robustly extrapolating to length 8192, reducing tail perplexity by 83.9% over RoPE. While these controlled experiments establish HySPE’s extrapolation robustness and numerical stability, evaluating its scaling behavior on large-scale foundation models remains an important direction for future investigation.

## 1 Introduction

Positional encodings dictate how self-attention mechanisms in Transformers [1] represent relative token displacements. Rotary Position Embedding (RoPE) [2] establishes relative-position dependence through pairwise orthogonal rotations of query and key projections. In two dimensions, these rotations correspond to the elliptic subgroup of the real symplectic group $\operatorname { S p } ( 2 , \mathbb { R } ) \ \cong \ \operatorname { S L } ( 2 , \mathbb { R } )$ While volume-preserving and isometric, elliptic orbits are bounded and non-dissipative; consequently, extending sequence length beyond the pretraining horizon exposes unobserved phase domains, precipitating phase aliasing and attention entropy collapse [3, 4].

In this work, we propose Hyperbolic Symplectic Positional Encoding (HySPE), exploring the complementary hyperbolic branch of Sp(2, R). Motivated by dissipative dynamical systems and hyperbolic geometric flows [5, 6], HySPE constructs relative position operators via symmetric dual shears. The composition of two transverse elementary shears generates a symmetric hyperbolic operator with reciprocal positive real eigenvalues. When modulated by an exponential decay envelope, the relative attention kernel becomes contractive and conformally symplectic, satisfying $T _ { \Delta } ^ { \top } J \bar { T } _ { \Delta } = e ^ { - 2 \gamma \Delta } J$ for the standard symplectic form �. This endows each channel pair with two distinct spectral attenuation rates, providing anisotropic distance filtering directly within the representation space.

Crucially, we uncover and resolve a fundamental tension between the continuum algebraic kernel and its finite-precision realization. Although the relative operator $T _ { \Delta }$ is strictly bounded, expressing it via separated absolute query and key matrices forces the key operator to diverge exponentially as $e ^ { ( 2 \chi + \delta \bar { ) } n }$ , causing severe numerical overflow in float precision and corrupting KV-cache representation scales.

We overcome this representation divergence without modifying the mathematical kernel. By diagonalizing the dual-shear operator into its invariant eigenbasis, we factorize the bilinear interaction into decoupled decay modes and introduce spectral block rebasing. This local gauge transformation anchors intermediate scales to query-block origins, bounding maximum positive exponents to small constants regardless of sequence length. To eliminate kernel dispatch overhead, we design an adaptive centered blocking scheme that executes long-sequence attention in few kernel calls.

Across rigorous, multi-seed experiments on a byte-level Transformer trained on TinyShakespeare at length 256, HySPE achieves marked in-domain perplexity gains and zero-shot extrapolation up to 4096 tokens (16× extension) with zero quality degradation. Identical-checkpoint interventions verify that the stable spectral formulation completely recovers the predictive capacity lost to floating-point truncation in naive factorization. Furthermore, optimized HySPE forward latency reaches 7.21 ms at length 4096, matching cached RoPE (7.20 ms) on an RTX 4090. HySPE demonstrates that hyperbolic symplectic geometry provides an expressive, numerically robust alternative to canonical rotations for foundation sequence models.

## 2 Related Work

Rotary, scaled, and multi-rate position representations. In Transformer self-attention [1], relative position mechanisms modulate query-key dot products through relative displacement $\Delta = m - n$ . RoPE [2] enforces this via 2D orthogonal rotations, corresponding to the elliptic subgroup of $\mathrm { S p } ( 2 , \mathbb { R } )$ , which preserves vector $L _ { 2 }$ norms but lacks intrinsic distance damping and exhibits periodic phase oscillation under context extension. xPos [7] augments rotary transformations with empirical exponential damping $\zeta ^ { \Delta } \mathcal { R } ( \Delta \theta )$ to facilitate length extrapolation. However, xPos applies an isotropic scalar decay that uniformly contracts circular orbits. HySPE fundamentally departs from the elliptic branch: it parameterizes the hyperbolic branch of Sp(2, R) via symmetric dual shears, producing a contractive spectrum with two unequal real decay rates per channel pair.

Attractor dynamics, hyperbolic geometry, and Lorentz boosts. The formulation of HySPE is motivated by recent advances in non-Euclidean representation learning, notably contractive torus attractors in Ananke [5] and closed-form hyperbolic flows in Minkowski Attractor Networks [6]. In language modeling, HoPE [8] recently explored continuous Lorentz rotations in hyperbolic geometry to enforce monotonic attention decay. However, parameterizing continuous Lorentz boosts in Euclidean attention without coordinate stabilization introduces severe representation divergence (exponential key-norm explosion) over long contexts. In contrast, HySPE establishes an exact, discrete relative positional kernel via dual shears that preserves conformal symplectic invariants, analytically resolving representation drift through invariant eigenbasis rebasing and centered block execution.

Decayed and recurrent attention. RetNet [9] and ALiBi [10] introduce scalar distance penalties into linear retention and softmax attention, respectively. In contrast to scalar attention biases that penalize all channels indiscriminately, HySPE operates directly on 2D feature subspaces via hyperbolic shears, providing content-dependent, anisotropic decay. Moreover, HySPE retains exact quadratic causal attention rather than relying on linear-attention approximations.

Exact attention and memory-eficient execution. FlashAttention [11] highlights the primacy of tiling and IO awareness in Transformer eficiency. Our stable implementations operate within standard scaled dot-product attention (SDPA), optimizing positional coordinate scales and block scheduling to match native rotary eficiency on modern hardware.

## 3 Symplectic Dual-Shear Attention

## 3.1 Symplectic Group Perspective and Dual-Shear Construction

Linear transformations on two-dimensional phase space preserving the canonical symplectic form $J = \left( \begin{array} { l l } { 0 } & { 1 } \\ { - 1 } & { 0 } \end{array} \right)$ constitute the real symplectic group:

$$
\operatorname { S p } ( 2 , \mathbb { R } ) = \left\{ M \in \mathbb { R } ^ { 2 \times 2 } \ \middle \vert \ M ^ { \top } J M = J \right\} \cong \operatorname { S L } ( 2 , \mathbb { R } ) .\tag{1}
$$

Elements of $\mathrm { S p } ( 2 , \mathbb { R } )$ fall into three conjugacy classes based on their trace $\mathrm { T r } ( M )$

1. Elliptic $( | \mathrm { T r } ( M ) | < 2 )$ : Conjugate to standard Euclidean rotations SO(2). The eigenvalues reside on the unit circle $e ^ { \pm i { \bar { \theta } } }$ . This class underpins RoPE [2]. Its bounded orbits lack geometric dissipation.

2. Parabolic $( | \mathrm { T r } ( M ) | = 2 ) \colon$ Unipotent shears of the form $\left( { \begin{array} { l l } { 1 } & { h } \\ { 0 } & { 1 } \end{array} } \right)$ . They represent degenerate, shear-like transitions without spectral separation.

3. Hyperbolic $( | \mathrm { T r } ( M ) | > 2 ) $ Real reciprocal eigenvalues $e ^ { \pm \chi }$ . They induce distinct expansive and contractive axes, defining hyperbolic boosts.

HySPE investigates the hyperbolic symplectic branch. For a parameter $\chi \geq 0 .$ , let $h = 2 \sinh ( \chi / 2 )$ . We construct the fundamental operator � via a symmetric composition of two elementary parabolic shears:

$$
H = ( { \begin{array} { c c } { 1 } & { h } \\ { 0 } & { 1 } \end{array} } ( { \begin{array} { c c } { 1 } & { 0 } \\ { h } & { 1 } \end{array} } ) = ( { \begin{array} { c c } { 1 + h ^ { 2 } } & { h } \\ { h } & { 1 } \end{array} } ) .\tag{2}
$$

Direct calculation verifies that det $( H ) = ( 1 + h ^ { 2 } ) - h ^ { 2 } = 1$ , and consequently:

$$
H ^ { \top } J H = J \implies H \in \mathrm { S p } ( 2 , \mathbb { R } ) .\tag{3}
$$

The matrix � is symmetric positive definite with $\operatorname { T r } ( H ) = 2 + h ^ { 2 } = 2 \cosh \chi \geq 2$ . Its eigenvalues are precisely $e ^ { - \chi }$ and $e ^ { \chi }$

## 3.2 Conformally Symplectic Attention Kernel

To ensure contractive attention over arbitrary distances, we introduce dissipation rate $\gamma = \chi + \delta ,$ , where $\delta \geq 0$ . We modulate row-vector query and key representations $q _ { m } , k _ { n } \in \mathbb { R } ^ { 1 \times 2 }$ at positions � and � through dual contragredient actions:

$$
\widetilde { q } _ { m } = e ^ { - \gamma m } q _ { m } H ^ { m } , \qquad \widetilde { k } _ { n } = e ^ { \gamma n } k _ { n } ( H ^ { - n } ) ^ { \top } .\tag{4}
$$

For causal displacement $\Delta = m - n \ge 0$ , their bilinear inner product satisfies:

$$
\widetilde { q } _ { m } \widetilde { k } _ { n } ^ { \top } = q _ { m } T _ { \Delta } k _ { n } ^ { \top } , \qquad T _ { \Delta } = e ^ { - \gamma \Delta } H ^ { \Delta } .\tag{5}
$$

Design dichotomy: Absolute vs. relative shear modulations. An appealing computational property of elementary shears is that their action $( x , y ) \mapsto ( x + h y , y )$ involves only a single fused multiply-add (FMA) instruction, bypassing transcendental evaluations (cos sin cosh sinh) entirely. A natural aspiration is to directly substitute RoPE’s orthogonal rotations with such unipotent shears. However, direct symmetric application fails to produce a relative kernel. For an elementary shear $S _ { m } = \left( \begin{array} { c c } { { 1 } } & { { m h } } \\ { { 0 } } & { { 1 } } \end{array} \right)$ , its non-orthogonality $( S _ { m } ^ { \top } \neq S _ { m } ^ { - 1 } )$ implies that the symmetric bilinear form couples absolute coordinates:

$$
S _ { m } ^ { \top } S _ { n } = { \binom { 1 } { m h } } ( { 1 \atop 0 }  \ { 1 \atop 1 } ) = { \binom { 1 } { m h } }  \begin{array} { c } { { n h } } \\ { { 1 + m n h ^ { 2 } } } \end{array} ) ,\tag{6}
$$

which explicitly depends on $m \cdot n$ rather than $m - n$

This distinction leads to two divergent architectural strategies: (1) absolute shear modulation, which exploits ultra-cheap FMA operators while accepting coordinate cross-coupling, and (2) relative dual-shear modulation, which restores exact shift invariance $\Delta = m - n$ by pairing a composed symmetric shear � with its contragredient dual $( H ^ { - n } ) ^ { \top }$ . In this paper, we focus strictly on the latter, developing HySPE as a principled relative positional framework.

Proposition 1 (Conformal Symplectic Property and Contractive Spectrum). The relative positional transfer operator $T _ { \Delta }$ satisfies the conformal symplectic identity:

$$
T _ { \Delta } ^ { \top } J T _ { \Delta } = e ^ { - 2 \gamma \Delta } J .\tag{7}
$$

Furthermore, let $H = U \mathrm { d i a g } ( e ^ { - \chi } , e ^ { \chi } ) U ^ { \top }$ denote the orthogonal eigendecomposition of �. Then:

$$
T _ { \Delta } = U \mathrm { d i a g } \Big ( e ^ { - ( 2 \chi + \delta ) \Delta } , e ^ { - \delta \Delta } \Big ) U ^ { \top } , \qquad \| T _ { \Delta } \| _ { 2 } = e ^ { - \delta \Delta } \leq 1 .\tag{8}
$$

Proof. Since $H \in \mathrm { S p } ( 2 , \mathbb { R } )$ , $H ^ { \top } J H = J$ , by induction $( H ^ { \Delta } ) ^ { \top } J H ^ { \Delta } = J .$ . Thus:

$$
T _ { \Delta } ^ { \top } J T _ { \Delta } = \left( e ^ { - \gamma \Delta } H ^ { \Delta } \right) ^ { \top } J \left( e ^ { - \gamma \Delta } H ^ { \Delta } \right) = e ^ { - 2 \gamma \Delta } ( H ^ { \Delta } ) ^ { \top } J H ^ { \Delta } = e ^ { - 2 \gamma \Delta } J .
$$

Substituting $H ^ { \Delta } = U$ diag $( e ^ { - \chi \Delta } , e ^ { \chi \Delta } ) U ^ { \top }$ and $\gamma = \chi + \delta$ into Eq. (5) directly yields Eq. (8). The operator norm is given by the dominant eigenvalue $e ^ { - \delta \Delta } \leq 1$ □

For head dimension $d _ { h }$ with $K = d _ { h } / 2$ pairs, the complete attention score evaluates as:

$$
\ell _ { m n } = \frac { 1 } { \sqrt { d _ { h } } } \sum _ { j = 0 } ^ { K - 1 } q _ { m , j } T _ { \Delta , j } k _ { n , j } ^ { \top } , \qquad o _ { m } = \sum _ { n \leq m } \frac { e ^ { \ell _ { m n } } } { \sum _ { t \leq m } e ^ { \ell _ { m t } } } \nu _ { n } .\tag{9}
$$

HySPE introduces zero additional learned parameters; the values, projections, and causal softmax retain standard Transformer forms.

## 3.3 The Representation Divergence Dilemma

Proposition 1 reveals a critical numerical dichotomy. While the composite operator $T _ { \Delta }$ is contractive $( \| T _ { \Delta } \| _ { 2 } \leq 1 )$ , the naive absolute key factors in Eq. (4) diverge exponentially. Specifically, the key transformation matrix $T _ { K } ( n ) = e ^ { \gamma n } ( H ^ { - n } ) ^ { \top }$ possesses eigenvalues:

$$
\lambda ( T _ { K } ( n ) ) \in \left\{ e ^ { ( \gamma + \chi ) n } , e ^ { ( \gamma - \chi ) n } \right\} = \left\{ e ^ { ( 2 \chi + \delta ) n } , e ^ { \delta n } \right\} .\tag{10}
$$

Its spectral radius grows as $\rho ( T _ { K } ( n ) ) = e ^ { ( 2 \chi + \delta ) n }$ and its condition number scales as $\kappa ( T _ { K } ( n ) ) = e ^ { 2 \chi n }$ $\mathrm { A t } n = 4 0 9 6 .$ intermediate activations blow up beyond $1 0 ^ { 3 0 }$ , triggering immediate exponent overflow in standard 16-bit floating-point formats and severe rounding truncation in FP32. Computing relative scores through separated absolute matrices destroys precision, obscuring the true performance of the underlying mathematical kernel.

## 4 Numerically Stable Evaluation of HySPE

## 4.1 Exact Spectral Rebasing

We resolve representation divergence by eliminating absolute coordinate scaling entirely. Let $r = ( r ^ { ( 1 ) } , r ^ { ( 2 ) } ) = ( 2 \chi + \delta , \delta )$ denote the paired spectral decay rates, and let $c \in \mathbb { R }$ represent an arbitrary coordinate anchor. For each channel pair, we project queries and keys into the invariant eigenbasis � and rebase them to �:

$$
Q _ { m } ^ { ( c ) } = ( q _ { m } U ) \odot e ^ { - r ( m - c ) } , \qquad K _ { n } ^ { ( c ) } = ( k _ { n } U ) \odot e ^ { r ( n - c ) } .\tag{11}
$$

Because $e ^ { - r ( m - c ) } \odot e ^ { r ( n - c ) } = e ^ { - r ( m - n ) }$ , their inner product in the rotated frame satisfies:

$$
Q _ { m } ^ { ( c ) } \left( K _ { n } ^ { ( c ) } \right) ^ { \top } = ( q _ { m } U ) \mathrm { d i a g } \Big ( e ^ { - r ^ { ( 1 ) } ( m - n ) } , e ^ { - r ^ { ( 2 ) } ( m - n ) } \Big ) ( k _ { n } U ) ^ { \top } = q _ { m } T _ { m - n } k _ { n } ^ { \top } .\tag{12}
$$

This algebraic equivalence holds identically for any choice of � in real arithmetic.

In our stable reference implementation (�), queries are partitioned into contiguous blocks $[ a , b )$ of length at most 128, and we set $c = a$ . Each block attends to the full causal key history [0, �). The maximum positive exponent evaluated anywhere in keys is strictly bounded by $r _ { \mathrm { m a x } } ( b - a - 1 ) \leq 1 2 7 r _ { \mathrm { m a x } } \approx 2 . 5 4$ , completely independent of absolute sequence position � or �. Non-initial rectangular blocks enforce causality via explicit ofset masks.

## 4.2 Adaptive Centered Block Evaluation

While reference � provides numerical stability, invoking SDPA over 128-token blocks induces high kernel dispatch overhead for long sequences. We design an optimized fast path (�) by combining centered coordinate anchoring with adaptive block sizing:

$$
B _ { \mathrm { m a x } } = \left\lfloor \frac { E } { r _ { \mathrm { m a x } } } \right\rfloor + 1 , \qquad c = \frac { a + b - 1 } { 2 } , \qquad E = 4 0 .\tag{13}
$$

By setting the anchor � at the block midpoint, positive exponents in both queries and keys are bounded by $E / 2 = 2 0 $ . At FP32, $e ^ { 2 \bar { 0 } } \approx 4 . 8 5 \times 1 0 ^ { 8 }$ , which resides well within the normal float range without underflow or overflow.

Positional geometry and masks are precomputed and cached per sequence length. At context length 4096, HySPE-UltraLong requires only a single SDPA dispatch per layer, while HySPE-v1 requires two, compared to 32 calls in reference �. The fast path retains exact quadratic causal attention, maintaining mathematical equivalence to � while matching RoPE execution speed.

## 5 Length Extrapolation on TinyShakespeare

Model architecture and data. We evaluate HySPE on an autoregressive byte-level Transformer $( V = 2 5 6 , 4 . 8 \mathrm { M }$ parameters) featuring 6 layers, hidden dimension $d = 2 5 6$ , 4 attention heads $( d _ { h } = 6 4 )$ , SwiGLU FFN dimension $d _ { \mathrm { f f n } } = 6 8 8$ , and pre-RMSNorm with tied embeddings. Models are trained on the 90/10 contiguous byte split of the TinyShakespeare benchmark [12]. No artificial QK-normalization or external attention biases are applied.

Training configuration. All configurations are trained at context length $L _ { \mathrm { t r a i n } } = 2 5 6$ for 2,400 steps using AdamW with batch size 32, weight decay 0.01, gradient clipping at 1.0, and cosine learning-rate decay from $1 . 5 \times 1 0 ^ { - 3 }$ to $1 0 ^ { - 4 }$ . Experiments are replicated across seeds 42, 43, and 44, sharing initial weights and training sequence samplers. Computations use FP32 with TF32 disabled on an NVIDIA RTX 4090.

The frequency spectrum derives from $f _ { j } = 1 0 0 0 0 ^ { - j / 3 2 }$ and $\chi _ { j } = \chi _ { \mathrm { m a x } }$ tanh $( f _ { j } )$ for $j \in \{ 0 , \ldots , 3 1 \}$ }. We examine three spectral configurations:

• HySPE-v1: $( \chi _ { \mathrm { m a x } } , \delta ) = ( 0 . 0 0 8 , 0 . 0 0 1 0 )$ , balanced attenuation;

• HySPE-v2: $( \chi _ { \mathrm { m a x } } , \delta ) = ( 0 . 0 1 2 , 0 . 0 0 0 8 )$ , aggressive shear deformation;

• HySPE-UltraLong: $( \chi _ { \mathrm { m a x } } , \delta ) = ( 0 . 0 0 5 , 0 . 0 0 0 5 )$ , conservative long-range spectrum.

Configurations are trained under the original matrix factorization (�); HySPE-v1 and HySPE-UltraLong are additionally trained directly with stable spectral rebasing (�), yielding 18 trained checkpoints.

Evaluation metrics. Validation perplexity (PPL) is evaluated across 120 shared endpoints at context lengths � ∈ {256, 512, 1024, 2048, 4096}. We report both full-context PPL (mean NLL over all tokens) and matched-tail PPL (evaluated strictly over the final 256 tokens at each endpoint to isolate long-range context benefit). Reported metrics denote mean and sample standard deviation over three seeds.

Benchmarking and diagnostics. Latency is measured via synchronized PyTorch CUDA events on the RTX 4090 (PyTorch 2.6.0, CUDA 12.4). Measurements record median group latency across 30 iterations following 3 warmup passes. High-precision numerical audits compare FP32 implementations (�, �, �) against an independent FP64 double-precision reference (�) evaluated on promoted model parameters.

## 5.1 In-Domain Quality and Zero-Shot Length Extrapolation

As illustrated in Figure 1(a), every evaluated HySPE variant improves in-domain predictive quality over the matched RoPE baseline at $L = 2 5 6$ . The zero-shot context extension trajectory across 1× → 16× extrapolation horizons is visualized in Figure 1(b). Standard RoPE experiences immediate, catastrophic phase confusion once the sequence length exceeds the pretraining window, climbing steeply from 5.015 to 131.198 at $L = 4 0 9 6$ . In sharp contrast, HySPE variants maintain remarkably flat scaling curves. Notably, stable-trained HySPE-UltraLong preserves an invariant perplexity of 4.810 across all evaluation lengths from 256 to 4096, demonstrating that hyperbolic symplectic transformations provide an intrinsic inductive bias against long-range attention entropy explosion.

![](images/9c95dc4d7d6cc932d19ed9da15cc12a1f71dba17cbda747053f9455f12b7fcb1.jpg)

![](images/208e4a820ab095cbce310e8b6a415df5128326d94b5c62153ce04ea1b83d8f3b.jpg)  
Figure 1: Prediction quality and zero-shot context extrapolation on TinyShakespeare. All models are trained at $L _ { \mathrm { t r a i n } } = 2 5 6$ for 2,400 steps; HySPE configurations are trained and evaluated using the stable spectral formulation (�). Points and error bars indicate mean and sample standard deviation across seeds 42, 43, and 44 over 120 shared validation endpoints. (a) In-domain perplexity at length 256; percentage annotations denote relative PPL reductions over standard RoPE. (b) Full-context PPL across context lengths up to 16× $( L = 4 0 9 6 )$ on a logarithmic vertical axis. While RoPE diverges past 10<sup>2</sup>, HySPE-v1 exhibits stable scaling and HySPE-UltraLong maintains a flat trajectory near 4.81 throughout. Numerical details are in Table 1.

Table 1: Zero-shot context extrapolation perplexity up to 16× training length $( L _ { \mathrm { t r a i n } } = 2 5 6 )$ . All HySPE evaluations use the stable reference implementation (�); the training column denotes the training factorization. Values denote mean ± sample standard deviation across three random seeds.
<table><tr><td>Method</td><td>Training</td><td> $2 5 6 \left( 1 \times \right)$ </td><td>512 (2×)</td><td>1024 (4×)</td><td>2048 (8×)</td><td>4096 (16×)</td></tr><tr><td>RoPE</td><td>Standard</td><td> $5 . 0 1 5 \pm 0 . 1 4 9$ </td><td> $1 1 . 5 6 7 \pm 0 . 9 4 9$ </td><td> $3 3 . 1 2 5 \pm 5 . 3 1 1$ </td><td> $7 1 . 6 5 0 \pm 1 5 . 3 9 9$ </td><td> $1 3 1 . 1 9 8 \pm 3 4 . 8 8 2$ </td></tr><tr><td>HySPE-v1</td><td>Matrix (M)</td><td> $4 . 6 8 0 \pm 0 . 1 1 3$ </td><td> $4 . 7 6 7 \pm 0 . 1 0 1$ </td><td> $4 . 7 4 6 \pm 0 . 0 9 1$ </td><td> $4 . 7 4 1 \pm 0 . 1 0 0$ </td><td> $5 . 2 4 1 \pm 0 . 3 8 7$ </td></tr><tr><td>HySPE-v1</td><td>Stable (S)</td><td> $\mathbf { 4 . 6 8 8 \pm 0 . 0 6 3 }$ </td><td> $\mathbf { 4 . 7 7 8 \pm 0 . 0 7 1 }$ </td><td> $\mathbf { 4 . 7 6 1 \pm 0 . 0 7 0 }$ </td><td> $\mathbf { 4 . 7 5 7 \pm 0 . 0 7 2 }$ </td><td> $5 . 2 0 1 \pm 0 . 2 7 6$ </td></tr><tr><td>HySPE-v2</td><td>Matrix (M)</td><td> $4 . 7 2 6 \pm 0 . 0 9 3$ </td><td> $4 . 8 3 2 \pm 0 . 0 9 1$ </td><td> $4 . 7 8 7 \pm 0 . 0 7 8$ </td><td> $4 . 8 0 8 \pm 0 . 0 2 7$ </td><td> $5 . 0 1 5 \pm 0 . 1 5 4$ </td></tr><tr><td>HySPE-UltraLong</td><td>Matrix (M)</td><td> $4 . 7 6 2 \pm 0 . 0 3 3$ </td><td> $4 . 8 4 3 \pm 0 . 0 2 6$ </td><td> $4 . 8 1 0 \pm 0 . 0 3 5$ </td><td> $4 . 7 8 2 \pm 0 . 0 4 1$ </td><td> $\mathbf { 4 . 7 6 1 \pm 0 . 0 5 6 }$ </td></tr><tr><td>HySPE-UltraLong</td><td>Stable (S)</td><td> $4 . 8 1 0 \pm 0 . 0 3 9$ </td><td> $4 . 8 9 1 \pm 0 . 0 3 9$ </td><td> $4 . 8 5 0 \pm 0 . 0 3 4$ </td><td> $4 . 8 3 1 \pm 0 . 0 3 2$ </td><td> $\mathbf { 4 . 8 1 0 \pm 0 . 0 3 2 }$ </td></tr></table>

As shown in Table 1, every HySPE variant outperforms the baseline RoPE model in-domain $( L = 2 5 6 )$ . Stable-trained HySPE-v1 reduces perplexity from 5.015 to 4.688, representing a 6.52% relative reduction (paired seed improvements: 5.68%, 4.64%, and 9.11%). Under length extrapolation, standard RoPE exhibits severe degradation, escalating from 5.015 at $L = 2 5 6$ to 131.198 at $L = 4 0 9 6$ . In contrast, HySPE models remain completely stable. HySPE-UltraLong exhibits an entirely flat perplexity curve, recording precisely 4.810 at both length 256 and length 4096. At $L = 4 0 9 6 ,$ , its tail perplexity evaluates to $4 . 8 0 9 \pm 0 . 0 4 8$ , compared to $2 9 7 . 6 3 0 \pm 9 4 . 6 8 9$ for RoPE. Stable training confirms that this scale invariance is an intrinsic property of the hyperbolic symplectic operator, persisting when representation divergence is eliminated throughout training.

## 5.2 Identical-Checkpoint Interventions

To isolate algorithmic formulation from training dynamics, we conduct implementation interventions on fixed checkpoints (Table 2). Switching the evaluation kernel of matrix-trained HySPE-v1 from naive (�) to stable (�) drops length-4096 perplexity from 105.96 to 5.24, and tail perplexity from 436.82 to 7.68. Conversely, evaluating stable-trained HySPE-v1 with naive matrix factorization degrades perplexity from 5.20 to 111.42.

Table 2: Identical-checkpoint implementation intervention at length 4096. Evaluated on the exact same trained weights. Values denote mean ± sample standard deviation across three seeds.
<table><tr><td></td><td></td><td colspan="2">Full-context PPL (L = 4096)</td><td colspan="2">Matched-Tail PPL (L = 4096)</td></tr><tr><td>Method</td><td>Training</td><td>Naive (M) Eval</td><td>Stable (S) Eval</td><td>Naive (M) Eval</td><td>Stable (S) Eval</td></tr><tr><td>HySPE-v1</td><td>Matrix (M)</td><td> $1 0 5 . 9 6 \pm 1 3 . 3 0$ </td><td> ${ \pm } \mathbf { 0 . 2 4 } \pm \mathbf { 0 . 3 9 }$ </td><td> $4 3 6 . 8 2 \pm 7 3 . 8 0$ </td><td> ${ \pm \mathbf { 7 . 6 8 \pm 2 . 5 1 } }$ </td></tr><tr><td>HySPE-v1</td><td>Stable (S)</td><td> $1 1 1 . 4 2 \pm 1 5 . 6 1$ </td><td> ${ \pm \mathbf { 5 . 2 0 \pm 0 . 2 8 } }$ </td><td> $5 1 2 . 5 5 \pm 1 0 2 . 5 6$ </td><td> $\mathbf { 7 . 8 9 \pm 3 . 1 7 }$ </td></tr><tr><td>HySPE-v2</td><td>Matrix (M)</td><td> $1 6 5 . 8 4 \pm 2 3 . 6 7$ </td><td> ${ \bf 5 . 0 2 \pm 0 . 1 5 }$ </td><td> $4 1 8 . 4 0 \pm 1 0 1 . 8 9$ </td><td> ${ \pm } \ : 5 . 9 6 \pm 0 . 5 5$ </td></tr><tr><td>HySPE-UltraLong</td><td>Matrix (M)</td><td> $4 4 . 7 7 \pm 9 . 9 1$ </td><td> $\mathbf { 4 . 7 6 \pm 0 . 0 6 }$ </td><td> $4 2 7 . 3 9 \pm 1 3 6 . 6 6$ </td><td> ${ \bf 4 . 7 6 \pm 0 . 0 9 }$ </td></tr></table>

An operator-level numerical audit measuring dot products at position 4096 against double-precision ground truth reveals a maximum absolute error of $7 . 0 4 \times 1 0 ^ { 1 3 }$ for naive �, compared to $1 . 9 0 \times 1 0 ^ { - 7 }$ for stable �. In double precision, the output of � agrees with the exact operator to within $2 . 3 \times 1 0 ^ { - 1 5 }$ . This establishes that the observed extrapolation degradation in naive implementations is purely a numerical artifact of representation divergence, fully cured by spectral rebasing.

The catastrophic degradation observed in naive dual implementations is purely an artifact of representation divergence rather than a deficiency of the mathematical kernel. As shown in Figure 2(a) and Table 2, switching solely the evaluation path from the naive absolute factorization (�) to the stable spectral rebased kernel (�) on the exact same checkpoint drops length-4096 perplexity from 105.96 to 5.24 for HySPE-v1, and from 44.77 to 4.76 for HySPE-UltraLong. Conversely, running naive evaluation (�) on weights trained with the stable kernel (�) degrades performance to PPL 111.42, confirming that spectral rebasing is indispensable at inference time regardless of training trajectory.

![](images/bc502239b6b9cd677ed67ee3e9ce3721c8e4e84041fb11a4ae381c921f1ea0ab.jpg)

![](images/b5632dea2c0d18afc9b0a64d8f0367625f54a121f746bb18801113854c2f39fb.jpg)  
Figure 2: Numerical factorization recovery and forward execution latency. Values report mean ± sample standard deviation across three seeds. (a) Same-checkpoint intervention at context length 4096: evaluating identical trained weights with the stable spectral kernel (�, blue) entirely recovers low perplexity compared to the naive absolute matrix factorization (�, grey; vertical axis on log scale). Training modes are indicated below each model. (b) Full-model forward pass latency on an RTX 4090 $( L = 4 0 9 6 , B = 1$ FP32): centered adaptive evaluation (�) achieves an 8.50× speedup over the reference implementation (�), reaching 7.21 ms and matching cached RoPE at 7.20 ms. Detailed timing protocols and precision diagnostics are reported in Section 5 and Table 4.

## 5.3 Execution Latency and Benchmarking

Figure 2(b) benchmarks full-model forward pass latency at context length 4096 on an NVIDIA RTX 4090. While the reference implementation (�) incurs dispatch overhead due to its fine-grained 128-token query tiling (evaluating at 61.322 ms), the centered adaptive implementation (�) reduces latency to 7.212 ms. This corresponds to an 8.50× speedup, placing HySPE within 0.22% of cached RoPE (7.197 ms). As summarized in Table 3, training throughput (forward plus backward at $L = 2 5 6 , B = 3 2 )$ is similarly on par, with HySPE-UltraLong (17.558 ms) matching RoPE (17.394 ms) within 0.9%. Centered block scheduling thus efectively closes the eficiency gap between hyperbolic symplectic operators and standard rotary embeddings.

Table 3: Full-model execution latency on NVIDIA RTX 4090 in milliseconds (mean ± standard deviation across three checkpoints). Forward-backward excludes optimizer overhead.
<table><tr><td>Method / Kernel</td><td>Forward  $( L = 2 5 6 , B = 1 )$ </td><td>Forward  $( L = 4 0 9 6 , B = 1 )$ </td><td>Fwd+Bwd  $( L = 2 5 6 , B = 3 2 )$ </td></tr><tr><td>RoPE (Standard)</td><td> $2 . 8 2 6 \pm 0 . 0 2 9$ </td><td> $7 . 1 9 2 \pm 0 . 1 5 8$ </td><td> $1 7 . 4 7 3 \pm 0 . 0 3 7$ </td></tr><tr><td>RoPE (Cached)</td><td> $\mathbf { 2 . 5 7 2 \pm 0 . 1 3 0 }$ </td><td> $\mathbf { 7 . 1 9 7 \pm 0 . 0 2 1 }$ </td><td> $\mathbf { 1 7 . 3 9 4 \pm 0 . 0 3 0 }$ </td></tr><tr><td>HySPE-v1 / Reference (S)</td><td> $4 . 5 2 8 \pm 0 . 0 9 6$ </td><td> $6 1 . 2 1 1 \pm 0 . 1 7 4$ </td><td> $1 8 . 8 1 8 \pm 0 . 0 2 7$ </td></tr><tr><td>HySPE-v1 / Centered (F)</td><td> $2 . 9 0 5 \pm 0 . 0 3 7$ </td><td> $8 . 5 2 9 \pm 0 . 1 1 1$ </td><td> $1 7 . 5 8 3 \pm 0 . 0 2 0$ </td></tr><tr><td>HySPE-UltraLong / Reference (S)</td><td> $4 . 4 8 2 \pm 0 . 0 3 5$ </td><td> $6 1 . 3 2 2 \pm 0 . 0 4 9$ </td><td> $1 8 . 7 4 2 \pm 0 . 0 2 4$ </td></tr><tr><td>HySPE-UltraLong / Centered (F)</td><td> $\mathbf { 2 . 8 8 4 \pm 0 . 0 4 1 }$ </td><td> ${ \bf 7 . 2 1 2 \pm 0 . 0 8 2 }$ </td><td> $\mathbf { 1 7 . 5 5 8 \pm 0 . 0 0 9 }$ </td></tr></table>

Table 3 presents runtime benchmarks on the RTX 4090. For HySPE-UltraLong at length 4096, centered adaptive evaluation (�) reduces forward latency from 61.322 ms to 7.212 ms (an 8.50× speedup over reference �), closely matching cached RoPE (7.197 ms). Training throughput (Forward+Backward at $L = 2 5 6 , B = 3 2 )$ reaches 17.558 ms for HySPE-UltraLong, within 0.9% of cached RoPE (17.394 ms). Thus, HySPE introduces negligible runtime overhead while conferring robust length extrapolation.

## 5.4 Numerical Precision Diagnostics

Table 4: Precision diagnostics at length 4096 using stable-trained weights. � denotes FP64 double-precision reference. TV denotes token Total Variation; KL denotes Kullback-Leibler divergence. Top-1 changes evaluate across 36,864 tokens.
<table><tr><td>Method</td><td>Candidate / Reference</td><td>Max Logit Error</td><td>Mean TV</td><td>Mean KL</td><td>Top-1 Discrepancies</td></tr><tr><td>HySPE-v1</td><td>Fast (F) / Reference (S)</td><td> $1 . 1 8 1 \times 1 0 ^ { - 3 }$ </td><td> $5 . 6 5 7 \times 1 0 ^ { - 6 }$ </td><td> $3 . 1 3 4 \times 1 0 ^ { - 1 0 }$ </td><td>0 / 36, 864</td></tr><tr><td>HySPE-UltraLong</td><td>Fast (F) / Reference (S)</td><td> $1 . 8 8 1 \times 1 0 ^ { - 3 }$ </td><td> $9 . 0 6 4 \times 1 0 ^ { - 6 }$ </td><td> $7 . 3 1 9 \times 1 0 ^ { - 1 0 }$ </td><td>2 / 36, 864</td></tr><tr><td>HySPE-UltraLong</td><td>Reference (S) / FP64 (D)</td><td> $8 . 7 6 1 \times 1 0 ^ { - 4 }$ </td><td> $5 . 6 7 9 \times 1 0 ^ { - 6 }$ </td><td> $2 . 6 9 1 \times 1 0 ^ { - 1 0 }$ </td><td>0 / 36, 864</td></tr><tr><td>HySPE-UltraLong</td><td>Fast (F) / FP64 (D)</td><td> $2 . 0 4 8 \times 1 0 ^ { - 3 }$ </td><td> $8 . 4 8 8 \times 1 0 ^ { - 6 }$ </td><td> $6 . 6 2 5 \times 1 0 ^ { - 1 0 }$ </td><td>2 / 36,864</td></tr></table>

Full-dataset validation confirms that across all 45 evaluated checkpoints and context lengths, the mean NLL diference between fast path � and reference � never exceeds $2 . 1 9 8 \times 1 0 ^ { - 7 }$ . Fine-grained diagnostics at length 4096 (Table 4) confirm high fidelity: for HySPE-UltraLong, mean KL divergence between � and � is $7 . 3 1 9 \times 1 0 ^ { - 1 0 }$ , and top-1 predictions agree on 36,862 of 36,864 tokens (99.995% agreement), with a maximum logit discrepancy of $5 . 9 1 \times 1 0 ^ { - 5 }$ at the two disagreeing positions. Reference � provides maximal numerical fidelity, while centered � achieves near-RoPE runtime eficiency with indistinguishable validation loss.

## 6 Length Extrapolation on WikiText-103

## 6.1 Experimental Setup

We evaluate on WikiText-103 [13] (wikitext-103-raw-v1) using its standard splits. Text is tokenized with the GPT-2 byte-pair encoding (50,257 tokens). All models share a 51M-parameter decoder-only Transformer: 8 layers, hidden dimension 512, 8 attention heads $( d _ { h } = 6 4 )$ , SwiGLU intermediate dimension 1,376, pre-RMSNorm, and tied embeddings. Models are trained at context length 512 for 15,000 steps using AdamW $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5$ , weight decay 0.1). The learning rate peaks at $1 0 ^ { - 3 }$ after a 100-step linear warmup and decays via a cosine schedule to 10<sup>−4</sup>. Computations use BF16 autocast with torch.compile on an NVIDIA RTX 4090, while positional operations are executed in FP32.

We compare RoPE against two HySPE configurations: default HySPE with $( \chi _ { \mathrm { m a x } } , \delta ) = ( 0 . 0 0 5 , 0 . 0 0 0 5 )$ and HySPE-v2 with (0.012, 0.0008). Channel pairs derive spectral rates via Eq. (14).

$$
\omega _ { j } = 1 0 0 0 0 ^ { - j / 3 2 } , \qquad \chi _ { j } = \chi _ { \mathrm { m a x } } \operatorname { t a n h } ( \omega _ { j } ) , \qquad ( r _ { j , \mathrm { f a s t } } , r _ { j , \mathrm { s l o w } } ) = ( 2 \chi _ { j } + \delta , \delta ) , \quad j = 0 , \dots , 3 1 .\tag{14}
$$

For zero-shot length extrapolation (� ∈ {512, 1024, 2048, 4096}), we evaluate the same checkpoints across 64 fixed endpoints without position interpolation. We report both Full PPL (averaged over all � tokens) and Tail PPL (evaluated strictly on the final 512 targets to isolate the efect of extended context on identical tokens).

## 6.2 Results and Extrapolation Analysis

Table 5: In-domain results on WikiText-103 (� = 512, 15,000 steps). Validation PPL uses native BF16; standard test PPL uses FP32 stable evaluation.
<table><tr><td>Method</td><td>Xmax</td><td>δ</td><td>Validation PPL</td><td>Test PPL</td></tr><tr><td>RoPE</td><td></td><td></td><td>26.87</td><td>29.95</td></tr><tr><td>HySPE</td><td>0.005</td><td>0.0005</td><td>28.19</td><td>31.23</td></tr><tr><td>HySPE-v2</td><td>0.012</td><td>0.0008</td><td>27.73</td><td>30.78</td></tr></table>

Table 5 presents in-domain performance at context length 512. RoPE achieves a test PPL of 29.95, while default HySPE reaches 31.23 (a 4.28% gap). Increasing the shear magnitude in HySPE-v2 reduces test PPL to 30.78, narrowing the gap to 2.79%.

In-domain dynamics: Character vs. subword modeling. Unlike character-level modeling where metric dissipation acts as an immediate regularizer against high-frequency token repetition, subword modeling over dense BPE vocabularies relies heavily on RoPE’s unattenuated orthogonal phases for short-range syntactic coordinate distinction. Strengthening the shear deformation in HySPE-v2 partially recovers this local discriminability, substantially closing the in-domain gap.

Table 6: Zero-shot length extrapolation on WikiText-103 $( L _ { \mathrm { t r a i n } } = 5 1 2 )$ . Full PPL scores the entire context; Tail PPL scores the final 512 targets across all context lengths.
<table><tr><td colspan="2"></td><td colspan="2">RoPE</td><td colspan="2">HySPE</td><td colspan="2">HySPE-v2</td></tr><tr><td>L</td><td>Scale</td><td>Full</td><td>Tail</td><td>Full</td><td>Tail</td><td>Full</td><td>Tail</td></tr><tr><td>512</td><td>1x</td><td>28.10</td><td>28.10</td><td>29.10</td><td>29.10</td><td>28.81</td><td>28.81</td></tr><tr><td>1024</td><td>2x</td><td>53.55</td><td>97.68</td><td>28.00</td><td>26.00</td><td>27.84</td><td>25.82</td></tr><tr><td>2048</td><td>4x</td><td>111.15</td><td>247.33</td><td>27.02</td><td>26.21</td><td>27.10</td><td>26.58</td></tr><tr><td>4096</td><td>8x</td><td>187.83</td><td>361.59</td><td>27.15</td><td>28.07</td><td>31.01</td><td>52.97</td></tr></table>

Extrapolation up to 8× length. As context extends beyond the training horizon, the relative performance inverts decisively (Table 6). While RoPE experiences immediate phase breakdown—its Tail PPL surging to 97.68 at 2× and 361.59 at 8×—HySPE exhibits strong contractive stability.

Crucially, expanding context actively improves prediction quality for HySPE: default HySPE’s Tail PPL drops from 29.10 at � = 512 to 26.00 at � = 1024 (a 10.66% improvement on identical targets), remaining at 28.07 at � = 4096. At 8× extrapolation, default HySPE reduces Full and Tail PPL by 85.54% and 92.24% relative to RoPE. HySPE-v2 performs best up to 2× context, while default HySPE provides superior stability at 4× and 8×, reflecting the trade-of governed by the dissipation rates.

Stress test at 16× length (� = 8192). To probe the operational limits, we evaluate models on a long-document cohort up to length 8192 without position interpolation (Table 7).

At 16× extrapolation, RoPE diverges entirely (Tail PPL 502.51). In contrast, default HySPE retains strong coherence, achieving Full PPL 36.49 and Tail PPL 80.83 (an 83.91% reduction over RoPE). Although attenuation across 8,000 steps introduces moderate tail degradation, HySPE preserves predictive utility where rotary mechanisms experience catastrophic failure.

Table 7: Matched 16× stress test (� = 8192) on long document segments. Full PPL scores the complete context; Tail PPL scores the final 512 tokens.
<table><tr><td rowspan="2">L</td><td rowspan="2">Scale</td><td colspan="2">RoPE</td><td colspan="2">HySPE</td><td colspan="2">HySPE-v2</td></tr><tr><td>Full</td><td>Tail</td><td>Full</td><td>Tail</td><td>Full</td><td>Tail</td></tr><tr><td>512</td><td>1x</td><td>28.72</td><td>28.72</td><td>29.99</td><td>29.99</td><td>29.63</td><td>29.63</td></tr><tr><td>1024</td><td>2x</td><td>51.51</td><td>96.28</td><td>27.89</td><td>27.07</td><td>27.61</td><td>26.83</td></tr><tr><td>2048</td><td>4×</td><td>112.24</td><td>243.21</td><td>28.20</td><td>27.33</td><td>28.21</td><td>27.59</td></tr><tr><td>4096</td><td>8x</td><td>181.14</td><td>340.75</td><td>28.38</td><td>29.27</td><td>32.22</td><td>53.29</td></tr><tr><td>8192</td><td>16x</td><td>292.87</td><td>502.51</td><td>36.49</td><td>80.83</td><td>69.94</td><td>186.72</td></tr></table>

## 7 Discussion and Future Directions

HySPE demonstrates that positional attention benefits substantially from the rich geometric structure of non-compact symplectic groups. By deriving the contractive conformal symplectic kernel and resolving representation divergence via spectral rebasing, HySPE simultaneously achieves in-domain improvements or approximations and length extrapolation up to 16× pretraining context without learned decay parameters or heuristic windowing.

The symplectic framing provides a clear roadmap for architectural scaling:

1. Symplectic Hybrids and Complexification: In this work, the pure hyperbolic branch of Sp(2, R) demonstrates strong length extrapolation. To bridge periodic phase discrimination with monotonic dissipation, one natural progression is to compose elliptic (rotational) and hyperbolic (shear) generators within Sp(2, R). Beyond real subgroup composition, extending the framework to its complexification Sp(2, C)  SL(2, C) systematically incorporates continuous oscillatory phases into contractive shear orbits, unifying high-frequency local discriminability with invariant long-context stability.

2. Hardware-Fused Kernels: The centered adaptive block formulation maps directly onto modern tiled attention algorithms. Developing dedicated FlashAttention-style CUDA/Triton kernels that fuse the spectral basis rotation and coordinate rebasing into online SRAM loading will enable sub-quadratic KV-cache decoding.

3. Retrieval Dynamics at Scale: While perplexity remains invariant up to 4096 tokens, factual retrieval across long contexts involves distinct attention dynamics. Scaling HySPE to billion-parameter models and evaluating on extensive retrieval suites (e.g., Needle-In-A-Haystack, LongBench) will further clarify the interplay between contractive spectral rates and long-range associative recall.

## 8 Conclusion

We have presented HySPE (Hyperbolic Symplectic Positional Encoding), a principled positional encoding scheme grounded in symplectic dual shears and contractive conformal dynamics. By identifying and resolving the representation divergence inherent in naive factorizations through invariant spectral rebasing and centered block attention, HySPE reconciles non-compact geometric transformations with strict floating-point stability and hardware eficiency. Empirical evaluations demonstrate marked in-domain perplexity reductions or approximations and invariant 16× context extrapo lation at forward latencies matching standard RoPE. HySPE provides an expressive, mathematically rigorous foundation for long-context sequence modeling in modern Transformer architectures.

## References

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS), volume 30, 2017.

[2] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864, 2021. https://arxiv.org/abs/2104.09864.

[3] Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, and Siva Reddy. The Impact of Positional Encoding on Length Generalization in Transformers. In Advances in Neural Information Processing Systems (NeurIPS), 2023. Also available as arXiv:2305.19466. https://arxiv.org/abs/2305.19466.

[4] Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending Context Window of Large Language Models via Positional Interpolation. arXiv:2306.15595, 2023. https://arxiv.org/abs/2306. 15595.

[5] Zhongping Ji. Ananke: Contractive Torus Attractor Networks. arXiv:2609.24737, 2026. https://arxiv.org/ abs/2609.24737.

[6] Zhongping Ji. Minkowski Attractor Networks: Closed-Form Hyperbolic Flows for Visual Representations. arXiv:2609.37817, 2026. https://arxiv.org/abs/2609.37817.

[7] Yutao Sun, Li Dong, Barun Patra, Shuming Ma, Shaohan Huang, Alon Benhaim, Vishrav Chaudhary, Xia Song, and Furu Wei. A length-extrapolatable transformer. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14590–14604, 2023. https: //aclanthology.org/2023.acl-long.816/.

[8] Chang Dai, Hongyu Shan, Mingyang Song, and Di Liang. HoPE: Hyperbolic Rotary Positional Encoding for Stable Long-Range Dependency Modeling in Large Language Models. arXiv:2509.05218, 2025. https: //arxiv.org/abs/2509.05218.

[9] Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive Network: A Successor to Transformer for Large Language Models. arXiv:2307.08621, 2023. https://arxiv.org/abs/2307.08621.

[10] Ofir Press, Noah A. Smith, and Mike Lewis. Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation. arXiv:2108.12409, 2021. https://arxiv.org/abs/2108.12409

[11] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and Memory-Eficient Exact Attention with IO-Awareness. arXiv:2205.14135, 2022. https://arxiv.org/abs/2205.14135.

[12] Andrej Karpathy. Tiny Shakespeare corpus, char-rnn repository. https://github.com/karpathy/char-rnn/ tree/master/data/tinyshakespeare.

[13] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer Sentinel Mixture Models.arXiv:1609.07843, 2016. https://arxiv.org/abs/1609.07843.