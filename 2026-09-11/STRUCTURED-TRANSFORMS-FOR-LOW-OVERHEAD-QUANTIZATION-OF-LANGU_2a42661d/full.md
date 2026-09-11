# STRUCTURED TRANSFORMS FOR LOW-OVERHEAD QUANTIZATION OF LANGUAGE MODELS

A PREPRINT

Daria Cherniuk<sup>∗</sup> kamikazizen@gmail.com

Alexander Rudikov Institute of Numerical Mathematics

Boris Kashin Steklov Mathematical Institute

Ivan Oseledets Institute of Numerical Mathematics

September 11, 2026

## ABSTRACT

We revisit Kashin-decomposition-based weight quantization for large language models and propose an improved algorithm with stronger convergence properties and structured, efficient orthogonal transforms. The method retains the core factorization of each weight into two components – one with bounded infinity norm and the other with bounded infinity norm after an orthogonal transformation – but replaces the dense random orthogonal matrix with a sign-randomized Discrete Cosine Transform (DCT), reducing the per-iteration cost from O(N<sup>2</sup>) to O(N log N). The proposed greedy algorithm with alternating updates guarantees the four-peak distribution required for stable 2-bit clustering of each factor and admits closed-form initialization of cluster centers, removing the multi-restart k-means bottleneck of prior work. Composed with OPTQ-style sequential error compensation and QuIP-style incoherence preprocessing, the resulting JAX pipeline is competitive with OPTQ, QuIP, QuIP-RG and a fine-tuning- and vector-quantization-free variant of QuIP# at 4-bit per channel on OPT, Llama-2 and Pythia, with favorable wall-clock scaling. The bounded- $\cdot \ell _ { \infty }$ factorization is also notably robust: on stress configurations where QuIP variants diverge to four-digit perplexity (Pythia-6.9B) or abort with NaNs in LDL back-substitution (Mistral-7B), Kashin-DCT remains numerically stable and stays close to FP16 baseline. At inference time, each weight decomposes into two 2-bit factor codes per channel that are structurally suited to native-2-bit hardware.

Keywords Post-training quantization · Large language models · Kashin decomposition · Discrete cosine transform · Greedy algorithms

## 1 Introduction

Modern Large Language Models (LLMs) achieve their capabilities at the cost of weight tensors that dominate memory bandwidth and footprint at inference time. Post-training quantization (PTQ) compresses these weights to low bit-widths without re-training, making it the practical choice when the training pipeline and large-scale compute are unavailable. Lower inference precision also translates into lower energy draw per query and a correspondingly smaller carbon footprint. In deployment, the dominant scheme remains plain uniform scalar quantization with per-channel or per-group scales [Dettmers et al., 2022], sometimes with a non-uniform grid such as NF4 [Dettmers et al., 2023]. The research literature improves on this baseline first via second-order error compensation [Frantar et al., 2023, Frantar and Alistarh, 2022] and incoherence preprocessing [Chee et al., 2023, Tseng et al., 2024a], and most recently via vector quantization (VQ): AQLM [Egiazarian et al., 2024], GPTVQ [van Baalen et al., 2024], and QTIP [Tseng et al., 2024b] currently lead sub-4-bit benchmarks. VQ comes with three practical costs, however: each layer must store learned codebook entries alongside the integer indices, the strongest results require gradient-based fine-tuning, and codebook lookups do not fuse into standard matmul kernels as cleanly as scalar dequantization.

A complementary line of work [Merkulov et al., 2024] approaches low-bit quantization through Kashin decomposition: each vectorized weight matrix w is factorized as $w = \bar { u } + P ^ { T } \hat { v }$ , where $P$ is a random orthogonal matrix, u and $\hat { v } = P v$ separately have small infinity norms. Empirically, the components produced by Kashin’s greedy algorithm form distributions with four sharp symmetric peaks [Merkulov et al., 2024], which is a near-ideal target for 2-bit clustering. Despite this attractive structure, the prior application of Kashin decomposition to LLMs [Merkulov et al., 2024] suffered from three weaknesses. First, to amortize cost the authors reformulated the algorithm matrix-wise, which dropped the per-vector convergence guarantees of the original theorem [Kashin, 1977]; on a non-trivial fraction of layers the iteration failed to converge. Second, even on layers where the iteration did converge, the four-peak structure that 2-bit clustering relies on was not consistently produced: the joint u-vˆ distribution could collapse rather than separate into the four symmetric modes (Figure 3(a)); Third, the four cluster centers were recovered with multi-restart k-means, an unstructured search that dominated quantization wall-clock time.

Contributions. We revisit Kashin-decomposition-based quantization end-to-end and address all three weaknesses, while keeping the analysis at the vector level where rigorous convergence proofs are available.

• A partitioned greedy algorithm with guaranteed peaks and a structured DCT transform (Section 4). Building on the recent acceleration of Kashin’s theorem in [Kashin et al., 2025], we propose a Greedy Algorithm with Alternating Updates (Algorithm 1) that fixes the order of u- and vˆ-updates in blocks of four. The schedule guarantees four-peak distributions in both factors, and we prove the conversion guarantees. We further replace the dense random orthogonal Q by a sign-randomized Discrete Cosine Transform with O(N log N) cost and zero stored matrix.

• Closed-form cluster centers (Section 3.1). The greedy updates change the factors by some value $\pm c _ { k } ( r _ { k } )$ at each step, so after two updates the four peak locations are exactly $\pm c _ { 1 } \pm c _ { 2 } .$ , known analytically from the $r _ { k }$ residual norm. We initialize k-means with these centers and run only a handful of refinement iterations, eliminating multi-restart search and reducing quantization time.

• Adaptive integration and a JAX pipeline (Section 5). We compose Kashin decomposition with OPTQ-style sequential error compensation and QuIP-style incoherence preprocessing (Hadamard or Kronecker), and implement the full pipeline in compile-friendly JAX with multi-GPU pmap support. On OPT, Pythia, and Llama-family models at 4-bit per channel, the method matches or surpasses OPTQ, QuIP, QuIP-RG, and (fine-tuning-free, vector-quantization-free) QuIP# on WikiText-2 and C4 perplexity and on HellaSwag, PiQA, and Winogrande accuracy.

## 2 Related Work

Quantization techniques for deep neural networks can be broadly categorized into post-training quantization (PTQ) and quantization-aware training (QAT). QAT typically delivers the highest accuracy because it embeds low-precision constraints directly into the optimization loop; however, it also demands extensive GPU time and memory, especially when working with the very large models that stand to benefit most from reduced-precision inference. PTQ, in contrast, converts a pre-trained model to lower bit-widths without additional gradient-based fine-tuning, making it attractive for practitioners who lack the computational resources or training datasets.

The fastest and easiest-to-implement PTQ schemes rely on simple heuristics, such as uniform round-to-nearest or stochastic rounding applied per layer or per row Dettmers et al. [2022], or assume a particular activation/weight distribution, as in NormalFloat 4-bit (NF4) quantization Dettmers et al. [2023]. Activation- and outlier-aware method reduce the dynamic range that scalar quantization must cover, either by smoothing the activation/weight magnitudes [Xiao et al., 2023], by scaling salient channels [Lin et al., 2024a], by jointly learning weight clipping ranges and equivalent transformations [Shao et al., 2024], or by routing outliers to a dense non-uniform codebook [Kim et al., 2024].

Recent state-of-the-art methods use second-order information to minimize the quantization-induced error in the layer outputs. OPTQ Frantar et al. [2023] sequentially quantizes chunks of layer weights while compensating for the accumulated error through corrections to the yet-to-be-quantized parameters of the layer. QuIP [Chee et al., 2023] shows that OPTQ is a special case of LDLQ adaptive quantization, essentially the method introduced in QuIP without incoherence processing, in which the correction to the yet-to-be-quantized rows of the weight matrix is a linear combination of the quantization errors of the already-quantized rows. Authors further propose incoherence pre- and post-processing to bring weight matrices into the conditions of their theorem, which establishes that LDLQ quantization is a lower bound on nearest and stochastic rounding.

QuIP# [Tseng et al., 2024a] replaces multiplication by random orthogonal matrices with randomized Hadamard transforms for incoherence processing, which yields better incoherence properties and faster runtime. However, QuIP# uses vector quantization and resorts to fine-tuning. The same randomized-rotation idea is the basis for a parallel line of activation-quantization methods: QuaRot [Ashkboos et al., 2024] fuses Hadamard rotations into the residual stream so that both weights and activations are quantized in the rotated basis, SpinQuant [Liu et al., 2025] learns the rotation matrices on the calibration distribution, and DuQuant [Lin et al., 2024b] composes block-rotations with permutations to suppress outliers further.

![](images/e86f17e304341e339343a91a8820065fbdb0622488fae04aa27a2a717c5c554b.jpg)  
Figure 1: The proposed quantization pipeline consists of several stages: (a) move to the next column to quantize; (b) decompose with the proposed Greedy Algorithm with Alternating Updates (Algorithm 1) into u and vˆ; (c) quantize each factor to 2 bits via 4-peak clustering; (d) stack the quantized vectors into the output matrices U and $\hat { V }$ ; then add error compensation to the unquantized weights and return to (a).

Beyond QuIP#, a substantial line of work pushes weight compression further by switching from scalar to vector codebooks. AQLM [Egiazarian et al., 2024] learns an additive composition of small codebooks per group of weights; GPTVQ [van Baalen et al., 2024] extends OPTQ-style error compensation to multi-dimensional codebooks; and QTIP [Tseng et al., 2024b] replaces QuIP#’s lattice codebook with trellis-coded quantization on top of the same incoherence processing. These methods achieve very strong sub-4-bit compression ratios, but share two practical costs: each layer must store learned codebook entries alongside the integer indices, and reaching the reported quality typically requires gradient-based fine-tuning or extensive calibration sweeps. The Kashin decomposition we revisit in this work, by contrast, retains scalar 2-bit clusters and incurs only a handful of centroids per column (Section 3.1), with no fine-tuning step.

## 3 Problem Setting

The work of Kashin [1977] has introduced the following theorem.

Theorem 1. For each $N = 2 , 3 , \ldots$ . and for each orthogonal transformation $P \in \mathbb { O } ^ { N }$ , with the exception of a set $V \subset \mathbb { O } ^ { N } , \quad \mu _ { H } ( V ) \leqslant 2 ^ { - N }$ , thefollowing inequality holds:

$$
\operatorname* { m a x } \left\{ \| x \| _ { 1 } , \| P x \| _ { 1 } \right\} \geqslant c _ { 1 } \cdot \sqrt { N } \| x \| _ { 2 } \qquad \forall x \in \mathbb { R } ^ { N } ,\tag{1}
$$

where $c _ { 1 } > 0$ is an absolute constant.

Using (1), for each $P \notin V$ , a greedy algorithm [Temlyakov, 2011] was constructed such that, for every $x \in B _ { 2 } ^ { N } =$ $\{ x \in \mathbf { \bar { \mathbb { R } } ^ { N } } : \| x \| _ { 2 } \leqslant 1 \}$ , after k steps, it produces vectors $u _ { k }$ and $v _ { k }$ that satisfy

$$
\begin{array} { c } { \displaystyle { \| u \| _ { \infty } \leqslant \frac { c _ { 2 } } { \sqrt { N } } , \| P v \| _ { \infty } \leqslant \frac { c _ { 2 } } { \sqrt { N } } , } } \\ { \displaystyle { \| x - u _ { k } - v _ { k } \| _ { 2 } \leqslant \gamma ^ { k } , } } \end{array}\tag{2}
$$

where $\gamma < 1$ and $c _ { 2 }$ are absolute constants. See Appendix B for full algorithm. The algorithm requires storing $N ^ { 2 }$ entries of the matrix $Q$ and entails a computational complexity of $\mathcal { O } ( N ^ { 2 } )$ operations per iteration.

![](images/07da082d4a7497835096253061a39cb4759551d4e774b0e71c5dbc12da5e88fb.jpg)  
Figure 2: Convergence and factor-distribution comparison on a single random vector $\boldsymbol { x } \in \mathbb { R } ^ { N }$ with $N = 1 0 ^ { 4 }$ drawn from $\mathcal { N } ( 0 , I _ { N } )$ . Left: residual norm $\lVert x - u _ { k } - v _ { k } ^ { \bullet } \rVert _ { 2 }$ versus the number of greedy updates for the original greedy algorithm of Kashin [1977] (Algorithm 2 in Appendix B, violet) and our partitioned variant of Algorithm 1 (yellow). Center, right: empirical densities of $u _ { k }$ and $U _ { \mathcal { E } , \Phi } v _ { k }$ at convergence; the partitioned schedule produces four sharp symmetric peaks, while the original schedule leaves the second factor unimodal.

The upper bounds on the infinity norm in (2) naturally lead to using this factorization for quantization of neural network weights, since the most common uniform quantization suffers severely from outliers [Nagel et al., 2021]. An even more interesting phenomenon is that the distributions of both $u _ { k }$ and $P v _ { k }$ often form four distinctive symmetric peaks [Merkulov et al., 2024], which enables straightforward clustering-based quantization. The former work used this factorization to quantize the weights of an LLM. Their approach reformulated Kashin’s theorem for the matrix case but failed to provide a rigorous convergence analysis for that reformulation. Consequently, the approach encountered convergence failures and ill-defined peaks on a fraction of layers. Moreover, the clustering process relied on the k-means algorithm with multiple restarts, which significantly slowed quantization.

In this work, we retain the vector formulation, for which rigorous proofs are available, while alleviating the large size of the matrix Q noted in Merkulov et al. [2024] by using a sign-randomized Discrete Cosine Transform (DCT) as the orthogonal transformation, leveraging the recent acceleration of Kashin et al. [2025]; we then partition the greedy schedule so that the dominant updates land at analytically known centroid locations, which removes the multi-restart k-means step. The proposed Greedy Algorithm with Alternating Updates and its convergence analysis are presented in Section $4 ;$ the closed-form k-means initialization derived from this schedule is described in Section 3.1 below. We further combine Kashin quantization with adaptive quantization methods that quantize model weights chunk by chunk, compensating for the introduced error by adaptively updating the yet-to-be-quantized weights, in the spirit of OPTQ and QuIP; experimental details are provided in Section 5.

## 3.1 Closed-form k-means initialization

Previous work [Merkulov et al., 2024] applied k-means clustering to the four-peak distributions of u and $P v$ with random centroid initialization and multiple restarts $( n _ { \mathrm { i n i t } } = 5 0$ in their reported configuration), which dominated quantization wall-clock time. We exploit the structure of the updates in Algorithm 2: at iteration k, the update of the vector u is $\begin{array} { r } { \frac { 1 } { N } \| r _ { k } \| _ { 1 } \mathrm { s i g n } ( r _ { k } ) } \end{array}$ , which corresponds to adding $\pm c _ { k } ( r _ { k } )$ ) where $r _ { k }$ is a residual at iteration k. After two such updates, the marginal distribution of u contains four peaks at exactly $\pm c _ { 1 } \pm c _ { 2 }$ , known analytically from the residual norms. Because the residual norm decreases rapidly across iterations, these early updates already provide accurate approximations of the peak locations in the final distribution; in practice the first four iterations are sufficient. The fixed schedule of Algorithm 1 (Section 4) ensures that exactly two updates per factor land at the analytic centroids. After decomposition converges, the analytic $\pm c _ { 1 } \pm c _ { 2 }$ values are used to initialize the cluster centers and a small number of k-means refinement iterations are run; no multi-restart outer loop is required. The quantization procedure for vˆ follows the same strategy. The closed-form initialization removes the multi-restart loop entirely, reducing the clustering wall-clock time by roughly 10× relative to the multi-restart 2-D k-means baseline of [Merkulov et al., 2024] on medium-sized models.

## 4 The Greedy Algorithm with Alternating Updates

## 4.1 Background

Let $\Phi = \{ \phi _ { j } \} _ { j = 1 } ^ { N }$ be an orthonormal basis in $\mathbb { R } ^ { N }$ such that $\| \phi _ { j } \| _ { \infty } \leqslant K / \sqrt { N } , 1 \leqslant j \leqslant N$ . For the orthonormal DCT-II used throughout this $\begin{array} { r } { \mathsf { p a p e r } ^ { 2 } , \phi _ { j } ( n ) = \sqrt { 2 / N } \cos \left( \pi ( 2 n + 1 ) j / ( 2 N ) \right) \mathrm { f o r } j = 0 , \dots , N - 1 , \mathsf { s o } \| \phi _ { j } \| _ { \infty } \leqslant \sqrt { 2 / N } } \end{array}$ and the constant K in our analysis equals ${ \sqrt { 2 } } ,$ , i.e. a small absolute constant independent of N. As in Kashin et al. [2025], we consider the orthogonal operator $U _ { \mathcal { E } , \Phi } = \mathcal { F } _ { \Phi } ^ { - 1 } T _ { \mathcal { E } } \mathcal { F } _ { \Phi }$ , where the orthogonal operator $\mathcal { F } _ { \Phi }$ acts in $\mathbb { R } ^ { N }$ according to the rule $\begin{array} { r } { \mathcal { F } _ { \Phi } \big ( \sum _ { j = 1 } ^ { N } a _ { j } \phi _ { j } \big ) = \{ a _ { j } \} _ { j = 1 } ^ { N } = a \in \mathbb { R } ^ { N } } \end{array}$ , let $\mathcal { F } _ { \Phi } ^ { - 1 }$ be the inverse of $\mathcal { F } _ { \Phi }$ , and let the operator $T _ { \mathcal { E } }$ be defined for a given random set of signs $\mathcal { E } = \{ \varepsilon _ { j } \} _ { j = 1 } ^ { N } , \varepsilon _ { j } = \pm 1 , 1 \leqslant j \leqslant N$ by the relation $T _ { \mathcal { E } } ( a ) = \{ \varepsilon _ { j } a _ { j } \} _ { j = 1 } ^ { N }$

In Kashin et al. [2025], the authors leveraged the fact that for most sign sets $\mathcal { E } ,$ the following relation holds:

$$
\operatorname* { m a x } \left( \left\| x \right\| _ { 1 } , \left\| U _ { \varepsilon , \Phi } x \right\| _ { 1 } \right) \geqslant \frac { R ( N ) } { K \sqrt { 2 } } \qquad \forall x \in \mathbb { R } ^ { N } , \ \left\| x \right\| _ { 2 } = 1 ,\tag{3}
$$

where $R ( N ) = { \frac { \sqrt { N } } { c _ { 3 } { \big ( } \log N { \big ) } ^ { 1 / 2 } { \big ( } \log \log N { \big ) } ^ { 3 } } }$ , and $c _ { 3 }$ is an absolute constant. Throughout the rest of the paper we

instantiate $\mathcal { F } _ { \Phi }$ as the orthonormal DCT-II and write $P = U _ { \mathcal { E } , \Phi } ;$ with $\mathcal { F } _ { \Phi } ^ { - 1 } = \mathcal { F } _ { \Phi } ^ { \top }$ the operator $P$ admits the FFT-fast realization $P z = \mathrm { I D C T } \big ( \varepsilon \odot \mathrm { D C T } ( z ) \big )$ used in Algorithm 1.

The authors of Kashin et al. [2025], using the result from Guédon et al. [2008], propose a greedy algorithm in $N \cdot$ dimensional Euclidean space with the dictionary $S = Q _ { N } \bigcup U \varepsilon , _ { \Phi } Q _ { N }$ , where $\hat { Q _ { N } } \subset \mathbb { R } ^ { N }$ is the set of vertices of the cube $B _ { \infty } ^ { N } \mathrm { ( i . e . }$ ., the set of vectors of the form $( \delta _ { 1 } , \dots , \delta _ { N } )$ with $\delta _ { i } = \pm 1 , 1 \leqslant i \leqslant N )$ . It is shown that if (3) holds for a sign set $\mathcal { E }$ , then for an arbitrary vector $\boldsymbol { x } \in \mathbb { R } ^ { N }$ with $\| x \| _ { 2 } \leqslant 1$ , this greedy algorithm constructs vectors $u _ { k }$ and $v _ { k }$ from $\mathbb { R } ^ { \mathbb { N } }$ in k steps such that:

$$
\left\| x - u _ { k } - v _ { k } \right\| _ { 2 } \leqslant \left( 1 - \alpha ( N ) \right) ^ { k / 2 } , \qquad \alpha ( N ) = \frac { R ^ { 2 } ( N ) } { 2 K ^ { 2 } N }
$$

$$
\operatorname* { m a x } \left( \left\| u _ { k } \right\| _ { \infty } , \left\| U _ { \varepsilon , \Phi } v _ { k } \right\| _ { \infty } \right) \leqslant \frac { 4 c _ { 3 } ^ { 2 } K ^ { 2 } \log N \left( \log \log N \right) ^ { 6 } } { \sqrt { N } } .\tag{4}
$$

In this paper, based on numerical experiments, we propose a modification of the algorithm from Kashin et al. [2025] that offers several advantages over the original method. Notably, while the modification provides practical benefits, the resulting theoretical bound on the convergence rate for the new algorithm is slightly weaker than the estimate provided in Kashin et al. [2025].

## 4.2 Proposed Algorithm

The proposed modification retains the dictionary $S = Q _ { N } \cup U _ { \mathcal { E } , \Phi } Q _ { N }$ of Kashin et al. [2025], but replaces the single-step global greedy selection with a partitioned scheme that alternates between the two halves in blocks of four greedy steps: within each block, the first two atomic greedy steps draw from $Q _ { N }$ (and accumulate into $u )$ , while the last two draw from $U _ { \mathcal { E } , \Phi } Q _ { N }$ (and accumulate into $\hat { v } = U _ { \mathcal { E } , \Phi } v )$ ; see Algorithm 1. The slowdown $\beta ( N ) = \alpha ^ { 2 } ( N ) / 3 6$ established in Proposition 1 is the explicit price of fixing this schedule rather than letting each step adaptively choose the better half as in Kashin et al. [2025]; in exchange, the dominant updates of u and vˆ land at the analytic centroid locations $\pm c _ { 1 } \pm c _ { 2 }$ used by the closed-form k-means initialization of Section 3.1. By symmetry of the convergence argument the reverse schedule $( U _ { \mathcal { E , \Phi } } Q _ { N } , U _ { \mathcal { E , \Phi } } Q _ { N } , Q _ { N } , Q _ { N } )$ yields the same residual norm at every block of 4 iterations.

## 4.3 Convergence Analysis

We now estimate the convergence rate of Algorithm 1. Proposition 1 below shows that one block of four atomic greedy steps contracts the residual norm by a factor $( 1 - \beta ( N ) ) ^ { \hat { 1 } / 2 }$ with $\beta ( N ) = \alpha ^ { 2 } ( N ) / 3 6$ ; iterating yields the geometric residual decay of Theorem 2. Reaching a target residual norm therefore takes polylogarithmic factor of iteration steps of the unconstrained algorithm of Kashin et al. [2025].

Proposition 1. Let $\alpha ( N ) = R ^ { 2 } ( N ) / 2 K ^ { 2 } N$ and $\beta ( N ) = \alpha ^ { 2 } ( N ) / 3 6 .$ . For any $\boldsymbol { x } \in \mathbb { R } ^ { N }$ with $\| x \| _ { 2 } \leqslant A _ { \mathrm { \ell } }$ , one block of the partitioned greedy algorithm produces a residual $r _ { 1 }$ satisfying

$$
\lVert r _ { 1 } \rVert _ { 2 } \leqslant A \bigl ( 1 - \beta ( N ) \bigr ) ^ { 1 / 2 } .\tag{5}
$$

![](images/16a12e1dde5ff9fc0e3121e84c24e1177a418fba11e523ed0eaed0d3041a03d4.jpg)

![](images/cbbe84c26e38ad3b3d183762b9a5d3ae617518d6d76de19cc90d2863a81d6e42.jpg)  
Figure 3: Joint value distribution of u and vˆ for the fc2 layer weights of the third decoder block of OPT-125m, decomposed by three algorithms. Red crosses indicate the cluster centers found by k-means after factorization. (a) Matrix reformulation ofMerkulov et al. [2024]: the matrix-wise greedy iteration drops the per-vector convergence guarantee and the u-vˆ joint distribution collapses, leaving cluster centers ill-defined. (b) Original vector greedy algorithm [Kashin, 1977] (Algorithm 2): per-vector convergence is restored but the vˆ-axis remains spread-out, so 2-bit clustering of vˆ is degenerate. (c) Proposed Algorithm 1: the partitioned schedule and DCT-based P produce four-peaked marginals along both axes, yielding well-separated cluster centers.

Theorem 2. Let $\alpha ( N ) = R ^ { 2 } ( N ) / 2 K ^ { 2 } N$ and $\beta ( N ) = \alpha ^ { 2 } ( N ) / 3 6 .$ . For any $\boldsymbol { x } \in \mathbb { R } ^ { N }$ with $\| x \| _ { 2 } \leqslant 1$ , the partitioned greedy algorithm after k blocks ofiterations constructs $u _ { k } , v _ { k } \in \mathbb { R } ^ { N }$ such that

$$
\left\| x - u _ { k } - v _ { k } \right\| _ { 2 } \leqslant \big ( 1 - \beta ( N ) \big ) ^ { k / 2 } ,
$$

$$
\operatorname* { m a x } \Big ( \big \| u _ { k } \big \| _ { \infty } , \big \| U _ { \mathcal { E } , \Phi } v _ { k } \big \| _ { \infty } \Big ) \leqslant \frac { 4 } { \beta ( N ) \sqrt { N } } = \frac { c _ { 4 } K ^ { 4 } ( \log N ) ^ { 2 } ( \log \log N ) ^ { 1 2 } } { \sqrt { N } } ,\tag{6}
$$

where $c _ { 4 }$ is an absolute constant.

Proof sketch. The inequality (3) guarantees that, for any residual $\rho ,$ at least one of the two dictionary halves $Q _ { N }$ or $U _ { \mathcal { E } , \Phi } Q _ { N }$ is the “good” half, in the sense that a single greedy step from it contracts the squared $\ell _ { 2 }$ norm by a factor $( 1 - { \dot { \alpha } } ( N ) )$ ). The unconstrained algorithm of Kashin et al. [2025] picks the good half adaptively at every step. The partitioned algorithm cannot: it is forced to spend steps 1, 2 on $Q _ { N }$ regardless. Proposition 1 reduces to two cases. (i) If $Q _ { N }$ is the good half for the block’s input residual $\rho _ { 0 }$ , step 1 alone already gives $\| \rho _ { 1 } \| _ { 2 } \leqslant A \sqrt { 1 - \alpha } ,$ which is stronger than the claim; steps 2–4 are non-expansive. (ii) If $U _ { \mathcal { E } , \Phi } Q _ { N }$ is the good half, steps 1, 2 on $Q _ { N }$ are “wasted” but, by orthogonality of consecutive greedy updates, their combined contribution has norm $\leqslant A { \sqrt { 2 \beta ( N ) } }$ . Step 3 from $U _ { \mathcal { E } , \Phi } Q _ { N }$ is then at least as good as the best one-shot approximation $\lambda ^ { * } w ^ { * }$ of $\rho _ { 0 }$ from that half, which by the one-step estimate satisfies $\lVert \rho _ { 0 } - \lambda ^ { * } w ^ { * } \rVert _ { 2 } \leqslant A \sqrt { 1 - \alpha }$ . The triangle inequality combines these into $\Vert \rho _ { 3 } \Vert _ { 2 } \leqslant \dot { A } \sqrt { 1 - \alpha } + \dot { A } \sqrt { 2 \beta } \leqslant$ $A { \sqrt { 1 - \beta } } .$ , where the final step is exactly the algebraic identity that fixes the constant $\ddot { \beta } ( \ddot { N } ) = \alpha ^ { 2 } ( N ) / 3 6$ . Theorem 2 then iterates Proposition 1 over k blocks for the $\ell _ { 2 }$ bound, and bounds $\| u _ { k } \| _ { \infty }$ and $\| U _ { \mathcal { E } , \Phi } \dot { v } _ { k } \| _ { \infty }$ by summing the per-step coefficients $| \lambda _ { j } ^ { ( \ell ) } | \leqslant \| r _ { j - 1 } \| _ { 2 } / \sqrt { N }$ across blocks; the geometric residual decay makes that sum a convergent series of order $1 / ( \beta ( N ) \sqrt { N } )$ . Full proofs of Proposition 1 and Theorem 2 can be found in Appendix E.

## 5 Experiments

We conduct experiments on the OPT [Zhang et al., 2022], Llama-2 [Touvron et al., 2023], Mistral [Jiang et al., 2023], and Pythia [Biderman et al., 2023] families of models. We use WikiText-2 [Merity et al., 2017] and C4 [Raffel et al., 2020] for measuring perplexity, and HellaSwag [Zellers et al., 2019], PiQA [Bisk et al., 2020], and Winogrande [Sakaguchi et al., 2021] for zero-shot accuracy. All evaluations are run through the lm-evaluation-harness framework<sup>3</sup> for reproducible and reliable results, with maximum sequence length set to 2048. For all adaptive methods, the per-layer Hessian $H = X ^ { \top } .$ X is estimated on 1000 sequences of length 2048 drawn from the WikiText-2 (wikitext-2-raw-v1)

Algorithm 1 Greedy Algorithm with Alternating Updates (DCT-based realization).   
1: Input: Vector x $\in \mathbb { R } ^ { N }$ , sign mask $\varepsilon \in \{ - 1 , + 1 \} ^ { N }$ , number of blocks k   
2: Output: Vectors $u _ { k } , \hat { v } _ { k } \in \mathbb { R } ^ { N }$ such that   
x ≈ $u _ { k } + P \hat { v } _ { k } = u _ { k } + v _ { k } ,$   
where the orthogonal operator $P : \mathbb { R } ^ { N }  \mathbb { R } ^ { N }$ is defined by   
$P z : = \mathrm { I D C T } \left( \varepsilon \odot \mathrm { D C T } ( z ) \right) .$   
3: Initialize $u _ { 0 } \gets \mathbf { 0 } _ { N } , \ \hat { v } _ { 0 } \gets \mathbf { 0 } _ { N } , \ r _ { 0 } \gets x$   
4: for $t = 0 , 1 , \ldots , k - 1$ do   
5: $\rho  r _ { t } , \quad u  u _ { t } , \quad \hat { v }  \hat { v } _ { t }$   
6: for $j = 1 , 2$ do   
7: // two greedy stepsfrom $Q _ { N }$   
8: $\Delta u \gets \mathrm { s i g n } ( \rho ) \cdot \frac { \| \rho \| _ { 1 } } { N }$   
9: $u \gets u + \Delta u ,$ $\dot { \rho } \gets \rho - \Delta u$   
10: end for   
11: for $j = 3 ,$ 4 do   
12: // two greedy stepsfrom $U _ { \mathcal { E } , \Phi } Q _ { N }$   
13: $\Delta \hat { v }  ( \mathrm { s i g n } ( P \rho ) \cdot \frac { \| P \rho \| _ { 1 } } { N } )$   
14: $\hat { v }  \hat { v } + \Delta \hat { v } , \qquad \rho  \rho - ^ { \prime } P \Delta \hat { v }$   
15: end for   
16: $u _ { t + 1 } \gets u ,$ $\hat { v } _ { t + 1 } \gets \hat { v } ,$ $r _ { t + 1 } \gets \rho$   
17: end for   
18: Return: $u _ { k } , \ \hat { v } _ { k } , \ r _ { k }$ (and $v _ { k } = P \hat { v } _ { k }$ if needed, by symmetry $P ^ { 2 } = I )$

training split, and the same calibration set is used across all baselines. Unless stated otherwise, each row in Table 1 reports mean ± standard deviation over three random seeds; the seed determines the calibration data shuffle for all methods, the sign set ε for our Kashin-DCT pipeline, and the random incoherence rotation for any method that uses one. All quantization and evaluation runs are performed on a single NVIDIA H100 GPU; per-method end-to-end quantization wall-clock times are reported in Appendix D, and an inference-time analysis for the Kashin representation is given in Appendix C.

First, we demonstrate the advantages of our Greedy Algorithm with Alternating Updates over the original greedy algorithm. Figure 2 shows that, on a random vector drawn from N (0, 1), Algorithm 1 attains nearly the same convergence rate as the original vector greedy algorithm while producing markedly sharper peak definition.

Second, we show that our method does not fail on layers where the previous Kashin-decomposition-based approach encountered problems. Figure 3 compares the joint u and vˆ distributions for the matrix reformulation of Merkulov et al. [2024], the original greedy algorithm, and our updated algorithm. The proposed approach is the only one that produces distinctive peaks in both the u and $P v$ value distributions, enabling stable cluster quantization.

Finally, we compare our quantization pipelines with other PTQ methods, namely Round-To-Nearest (RTN), OPTQ [Frantar et al., 2023], QuIP [Chee et al., 2023] (with two decomposition variants, LDLQ and LDLQ-RG), and a fine-tuning- and vector-quantization-free variant of QuIP# [Tseng et al., 2024a]. For all methods, the per-outputchannel target is 4 bits per weight. The rows of Table 1 correspond to the following configurations: QuIP = LDLQ with Kronecker-of-rotations incoherence preprocessing; ${ \bf Q } { \bf u } { \bf \dot { I } P - R } { \bf G } = \mathrm { L D L Q - R G }$ with Kronecker-of-rotations incoherence preprocessing; QuIP# = LDLQ-RG with a randomized Hadamard transform for incoherence preprocessing, with end-to-end fine-tuning and the E8 lattice vector codebook of the original method disabled, since we only consider PTQ value quantization methods that do not require fine-tuning. Kashin-DCT+K and Kashin-DCT+H is Kashin decomposition with OPTQ-style sequential error compensation with the Kronecker and randomized-Hadamard incoherence preprocessing, respectively. Results are presented in Table 1.

We note that the LDLQ + Kronecker configurations (rows “QuIP” / “QuIP-RG”) exhibit larger variance and a perplexity regression on Pythia-1.4B and Llama-2-7B (marked <sup>†</sup> in Table 1), and the QuIP# row, which uses the randomized Hadamard variant, recovers the expected behavior. This QuIP instability also appears on Pythia-6.9B, where all three

Table 1: Perplexity (PPL) on WikiText-2 / C4 and accuracy (acc\_norm for HellaSwag and PiQA, acc for Winogrande) at 4-bit per-output-channel weight quantization. Best quantization result per column in bold, second best underlined (FP16 and RTN excluded from ranking). Context length 2048; each cell is mean ± std over 3 random seeds (except for FP16 and RTN). <sup>†</sup>LDLQ (QuIP) and LDLQ-RG (QuIP-RG) are unstable on Pythia-1.4B and Llama-2-7B in our reruns; see Section 5. Pythia-6.9B stress test in Appendix A.
<table><tr><td rowspan="2">Method</td><td colspan="5">Pythia-1.4B</td><td colspan="5">OPT-1.3B</td></tr><tr><td>Wiki-2 ↓</td><td>C4↓</td><td>Hella ↑</td><td>PiQA↑</td><td>Wino ↑</td><td>Wiki-2 ↓</td><td>C4↓</td><td>Hella ↑</td><td>PiQA↑</td><td>Wino ↑</td></tr><tr><td>FP16</td><td>14.72</td><td>38.76</td><td>52.12</td><td>71.16</td><td>57.30</td><td>16.47</td><td>39.41</td><td>41.36</td><td>71.00</td><td>59.75</td></tr><tr><td>RTN</td><td>18.73</td><td>49.99</td><td>49.72</td><td>69.42</td><td>55.33</td><td>29.46</td><td>73.25</td><td>34.31</td><td>67.52</td><td>53.59</td></tr><tr><td>OPTQ</td><td> ${ \bf 1 6 . 1 2 \pm 0 . 0 1 }$ </td><td> $\underline { { 4 3 . 4 7 \pm 0 . 0 2 } }$ </td><td>50.94 ± 0.27</td><td> $7 0 . 2 6 \pm 0 . 7 7$ </td><td>56.33 ± 0.63 17.63 ± 0.01 42.45 ± 0.17 40.46 ± 0.14</td><td></td><td></td><td></td><td> $7 0 . 5 1 \pm 0 . 2 8$ </td><td> $5 8 . 5 1 \pm 0 . 4 0$ </td></tr><tr><td>QulP</td><td> $1 9 . 7 6 ^ { \dag } \pm 0 . 4 4$ </td><td> $5 2 . 3 6 ^ { \dagger } \pm 0 . 9 4$ </td><td> $4 9 . 8 4 ^ { \dagger } \pm 0 . 0 3$ </td><td> $7 0 . 0 6 ^ { \dag } \pm 0 . 9 8$ </td><td> $5 6 . 7 5 ^ { \dagger } \pm 0 . 4 9$ </td><td></td><td></td><td></td><td>17.91 ± 0.08 42.83 ± 0.17 40.43 ± 0.17 70.82 ± 0.41</td><td> $5 8 . 6 9 \pm 0 . 7 2$ </td></tr><tr><td>QuIP-RG</td><td> $2 0 . 8 4 ^ { \dagger } \pm 0 . 3 9$ </td><td> $5 4 . 7 1 ^ { \dagger } \pm 0 . 7 9$ </td><td> $4 9 . 8 9 ^ { \dagger } \pm 0 . 4 3$ </td><td> $6 9 . 7 1 ^ { \dagger } \pm 0 . 3 6$ </td><td> $5 6 . 6 4 ^ { \dagger } \pm 0 . 9 1$ </td><td></td><td></td><td></td><td> $1 7 . 8 9 \pm 0 . 0 6 4 2 . 7 5 \pm 0 . 1 6 4 0 . 1 9 \pm 0 . 0 6 7 0 . 5 1 \pm 0 . 8 5$ </td><td> ${ \bf 5 9 . 2 7 \pm 0 . 9 9 }$ </td></tr><tr><td>QuIP #</td><td> $1 7 . 2 5 \pm 0 . 1 8$ </td><td> $4 5 . 9 2 \pm 0 . 4 9$ </td><td> $5 0 . 3 5 \pm 0 . 3 0$ </td><td> $\mathbf { 7 0 . 5 7 \pm 0 . 3 0 }$ </td><td> ${ \bf 5 7 . 4 3 \pm 0 . 1 6 }$ </td><td></td><td></td><td></td><td> $1 7 . 8 3 \pm 0 . 1 1 4 2 . 7 5 \pm 0 . 3 6 4 0 . 5 3 \pm 0 . 1 2 7 0 . 1 9 \pm 0 . 4 8$ </td><td> $5 8 . 0 9 \pm 1 . 1 0$ </td></tr><tr><td>Kashin-DCT+K</td><td> $1 6 . 5 7 \pm 0 . 0 4$ </td><td> $4 3 . 6 0 \pm 0 . 1 8$ </td><td> $5 0 . 0 0 \pm 0 . 1 1$ </td><td> $7 0 . 0 9 \pm 0 . 3 0$ </td><td> $5 7 . 2 2 \pm 1 . 2 4$ </td><td></td><td>17.20 ± 0.02 41.08 ± 0.04 40.63 ± 0.20 70.31 ± 0.08</td><td></td><td></td><td> $5 9 . 1 7 \pm 0 . 8 0$ </td></tr><tr><td>Kashin-DCT+H</td><td> ${ \underline { { 1 6 . 4 9 \pm 0 . 0 7 } } }$ </td><td> ${ \bf 4 3 . 4 0 \pm 0 . 2 7 }$ </td><td> $5 0 . 2 0 \pm 0 . 0 7$ </td><td> ${ \underline { { 7 0 . 4 2 \pm 0 . 1 4 } } }$ </td><td> $\overline { { 5 6 . 6 4 \pm 0 . 9 2 } }$ </td><td>17.19 ± 0.02 41.12 ± 0.02</td><td></td><td>40.59 ± 0.12</td><td>70.68 ± 0.28</td><td> $\overline { { 5 8 . 3 2 \pm 0 . 1 3 } }$ </td></tr></table>

<table><tr><td rowspan="2">Method</td><td colspan="5">Llama-2-7B</td><td colspan="5">Llama-2-13B</td></tr><tr><td> $\mathrm { \bf W i k i - } 2 \downarrow _ { - }$ </td><td>C4↓</td><td>Hella ↑</td><td>PiQA ↑</td><td>Wino ↑</td><td>Wiki-2 ↓</td><td>C4↓</td><td>Hella ↑</td><td>PiQA↑</td><td>Wino ↑</td></tr><tr><td>FP16</td><td>9.20</td><td>19.40</td><td>76.14</td><td>78.73</td><td>69.46</td><td>8.11</td><td>17.59</td><td>79.64</td><td>80.36</td><td>72.53</td></tr><tr><td>RTN</td><td>10.56</td><td>22.78</td><td>74.43</td><td>78.56</td><td>68.75</td><td>8.74</td><td>18.71</td><td>78.80</td><td>79.65</td><td>70.88</td></tr><tr><td>OPTQ</td><td> $9 . 8 0 \pm 0 . 0 1$ </td><td> $2 1 . 3 0 \pm 0 . 0 6$ </td><td> $7 4 . 7 9 \pm 0 . 1 3$ </td><td> $7 7 . 8 2 \pm 0 . 2 7$ </td><td>68.25 ± 0.51</td><td>8.48 ± 0.00</td><td>18.59 ± 0.08</td><td> $7 8 . 2 7 \pm 0 . 3 4$ </td><td>80.11 ± 0.28</td><td> $7 1 . 6 4 \pm 0 . 5 9$ </td></tr><tr><td>QuIP</td><td> $4 8 . 3 0 ^ { \dagger } \pm 2 9 . 8 4$ </td><td> $1 3 3 . 0 0 ^ { \dagger } \pm 8 3 . 5 7$ </td><td></td><td>48.10† ± 9.88 68.41† ± 3.99 57.06† ± 3.42</td><td></td><td> $8 . 6 4 \pm 0 . 0 3$ </td><td>18.80 ± 0.14</td><td> $7 7 . 7 9 \pm 0 . 3 0$ </td><td>79.78 ± 0.30</td><td> $7 2 . 2 7 \pm 0 . 2 5$ </td></tr><tr><td>QuIP-RG</td><td> $2 0 . 1 9 ^ { \dagger } \pm 4 . 2 2$ </td><td> $4 8 . 5 8 ^ { \dagger } \pm 1 2 . 6 3$ </td><td> $6 1 . 2 2 ^ { \dagger } \pm 3 . 8 5$ </td><td>73.54† ± 1.33</td><td> $6 4 . 5 6 ^ { \dagger } \pm 1 . 2 3$ </td><td> $8 . 6 3 \pm 0 . 0 7$ </td><td> $1 8 . 8 4 \pm 0 . 1 3$ </td><td></td><td>78.16 ± 0.20 79.52 ± 0.79</td><td> ${ \bf 7 3 . 2 2 \pm 0 . 6 7 }$ </td></tr><tr><td>QuIP #</td><td> ${ \bf 9 . 4 7 \pm 0 . 0 1 }$ </td><td> ${ \bf 2 0 . 1 6 \pm 0 . 1 3 }$ </td><td> ${ \underline { { 7 4 . 9 8 \pm 0 . 0 8 } } }$ </td><td> ${ \bf 7 8 . 4 9 \pm 0 . 1 4 }$ </td><td> $6 9 . 3 2 \pm 0 . 4 3$ </td><td> ${ \bf 8 . 3 1 \pm 0 . 0 0 }$ </td><td></td><td></td><td>18.19 ± 0.01 78.86 ± 0.0980.16 ± 0.16</td><td> $7 2 . 4 3 \pm 0 . 1 8$ </td></tr><tr><td>Kashin-DCT+K</td><td> $9 . 6 5 \pm 0 . 0 1$ </td><td> $2 0 . 4 5 \pm 0 . 0 9$ </td><td> $7 4 . 9 0 \pm 0 . 1 5$ </td><td> $7 8 . 0 9 \pm 0 . 2 7$ </td><td> ${ \bf 6 9 . 4 3 \pm 0 . 7 8 }$ </td><td> $8 . 3 8 \pm 0 . 0 0$ </td><td></td><td></td><td>18.19 ± 0.08 78.78 ± 0.04 80.54 ± 0.31 72.74 ± 0.55</td><td></td></tr><tr><td>Kashin-DCT+H</td><td> $\underline { { 9 . 6 0 \pm 0 . 0 2 } }$ </td><td> $\underline { { 2 0 . 3 6 \pm 0 . 0 3 } }$ </td><td> ${ \bf 7 5 . 0 4 \pm 0 . 1 6 }$ </td><td> $7 8 . 2 4 \pm 0 . 1 1$ </td><td> $6 9 . 2 7 \pm 0 . 8 7$ </td><td> $\underline { { 8 . 3 7 \pm 0 . 0 1 } }$ </td><td></td><td></td><td>18.19 ± 0.01 78.90 ± 0.23 80.12 ± 0.17</td><td> $7 2 . 5 3 \pm 0 . 2 1$ </td></tr></table>

QuIP variants we ran fail catastrophically – QuIP and QuIP-RG diverge to > 2000 WikiText-2 PPL, and even QuIP# blows up to ∼ 325 PPL – while $\mathrm { O } \bar { \mathrm { P T } } \mathrm { Q }$ stays close to its baseline at 12.02 PPL. “Kashin-DCT+H” recovers to $2 0 . 6 \pm 1 . 6$ WikiText-2 PPL, more than an order of magnitude better than QuIP# at the same bit budget; full per-task numbers are reported in Appendix A (Table 2). On Mistral-7B v0.1 the failure is sharper still: all four QuIP variants abort with NaN in the LDL back-substitution on the SwiGLU mlp.down\_proj layer, and OPTQ itself degrades to ≈ 380 Wiki-2 PPL, while Kashin-DCT remains numerically stable and stays within ∼ 0.3 Wiki-2 PPL of the FP16 reference of 8.63 (Table 3; full diagnostic in Appendix A). Across the four stress configurations we encountered — elevated variance on Llama-2-7B and Pythia-1.4B, catastrophic divergence on Pythia-6.9B, and NaN abort on Mistral-7B — Kashin-DCT is the only pipeline that remains numerically stable on every layer, which we read as evidence that the bounded- $\cdot \ell _ { \infty }$ factor decomposition with an adaptive per-column codebook is structurally more robust to weight-distribution outliers than fixed-grid rounding driven through an ill-conditioned Hessian inverse.

## 6 Conclusion

We revisited Kashin-decomposition quantization for LLMs and resolved the three weaknesses that had limited its prior application: weak convergence under matrix reformulation, the $\mathcal { O } ( N ^ { 2 } )$ cost of an explicit random orthogonal transform, and slow multi-restart k-means clustering. The proposed Greedy Algorithm with Alternating Updates retains the dictionary $Q _ { N } \cup U _ { \mathcal { E } , \Phi } Q _ { N }$ but fixes the order of u- and vˆ-updates in blocks of four, which places the dominant updates of each factor at known centroid locations $\pm c _ { 1 } \pm c _ { 2 }$ and produces the empirically four-peaked distributions that 2-bit clustering exploits; the analysis in Section 4 establishes geometric residual decay at per-block rate $( 1 - \beta ( N ) ) ^ { 1 / 2 }$ with $\beta ( N ) = \alpha ^ { 2 } ( N ) / 3 6$ , and an $\ell _ { \infty }$ bound of order $K ^ { 4 } ( \log N ) ^ { 2 } ( \log \log N ) ^ { 1 2 } / \sqrt { N }$ , i.e. within polylogarithmic factors of the unconstrained scheme of Kashin et al. [2025]. Replacing the random orthogonal Q with a sign-randomized DCT removes the dense $N \times N$ matrix entirely and brings the per-iteration cost to O(N log N), while closed-form initialization of cluster centers from the residual norms removes the k-means multi-restart outer loop, reducing k-means clustering wall-clock time. Composed with OPTQ-style error compensation and QuIP-style incoherence preprocessing, the resulting JAX pipeline is competitive with OPTQ, QuIP, QuIP-RG, and a fine-tuningand vector-quantization-free variant of QuIP# at 4-bit per channel on OPT-1.3B, Llama-2-7B/13B and Pythia-1.4B/6.9B, while admitting an inference-time decomposition into two 2-bit factor codes per channel that is structurally well-suited to native-2-bit hardware (Appendix C). The favorable scaling of Kashin-DCT+H from 7B to 13B (Appendix D) further suggests that the relative cost of the proposed pipeline becomes more attractive as model size grows, even before any kernel-level optimization is applied. A second, less expected property emerges from the stress tests: on configurations where the QuIP family diverges into four-digit perplexities (Pythia-6.9B) or aborts with NaNs in LDL back-substitution (Mistral-7B v0.1), and where OPTQ itself silently saturates to ≈ 380 Wiki-2 PPL on the same layer, Kashin-DCT+H is the only pipeline that produces robust results (Appendix A); the bounded- $\ell _ { \infty }$ factor decomposition with an adaptive per-column codebook is structurally more robust to weight-distribution outliers than fixed-grid rounding driven through an ill-conditioned Hessian inverse.

Limitations. Our results do not cover activation-aware baselines such as AWQ [Lin et al., 2024a], OmniQuant [Shao et al., 2024] or rotation-based methods [Ashkboos et al., 2024, Liu et al., 2025]. That said, rotation-based methods are orthogonal to our weight decomposition rather than competing with it: the rotation is applied to the weight matrix prior to factorization, and our Kashin-DCT+K and Kashin-DCT+H variants are already exactly this composition with Kronecker and randomized-Hadamard rotations, so substituting any other orthogonal preprocessing (e.g. the learned rotations of Liu et al. [2025] or QuaRot’s residual-stream rotation [Ashkboos et al., 2024]) leaves the rest of the pipeline unchanged. This is left for future work.

Future work. Two directions stand out. First, implementing the fused 2-bit GEMM with shared-ε DCT kernel described in Appendix C and benchmarking end-to-end latency against a tuned 4-bit baseline such as Marlin [Frantar et al., 2025] would convert the asymptotic bandwidth and arithmetic advantages of the decomposed form into a measured serving benefit, particularly on accelerators with native 2-bit tensor cores. Second, extending the decomposition beyond per-channel 4-bit — to sub-4-bit budgets via vector-quantization codebooks, or to weight-and-activation quantization by composing with learned rotations [Liu et al., 2025].

## References

Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. Llm.int8(): 8-bit matrix multiplication for transformers at scale. In Proceedings ofthe 36th International Conference on Neural Information Processing Systems, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc. ISBN 9781713871088.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Efficient finetuning of quantized LLMs. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview. net/forum?id=OUIFPHEgJU.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. OPTQ: Accurate quantization for generative pre-trained transformers. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=tcbBPnfwxS.

Elias Frantar and Dan Alistarh. Optimal brain compression: A framework for accurate post-training quantization and pruning. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors, Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=ksVGCOlOEba.

Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, and Christopher De Sa. QuIP: 2-bit quantization of large language models with guarantees. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=xrk9g5vcXR.

Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, and Christopher De Sa. QuIP#: even better llm quantization with hadamard incoherence and lattice codebooks. In Proceedings ofthe 41st International Conference on Machine Learning, ICML’24. JMLR.org, 2024a.

Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, and Dan Alistarh. Extreme compression of large language models via additive quantization. In Proceedings ofthe 41st International Conference on Machine Learning, ICML’24, 2024.

Mart van Baalen, Andrey Kuzmin, Markus Nagel, Peter Couperus, Cedric Bastoul, Eric Mahurin, Tijmen Blankevoort, and Paul Whatmough. GPTVQ: The blessing of dimensionality for LLM quantization. arXiv preprint arXiv:2402.15319, 2024.

Albert Tseng, Qingyao Sun, David Hou, and Christopher De Sa. QTIP: Quantization with trellises and incoherence processing. In Advances in Neural Information Processing Systems, 2024b.

Daniil Merkulov, Daria Cherniuk, Alexander Rudikov, Ivan Oseledets, Ekaterina Muravleva, Aleksandr Mikhalev, and Boris Kashin. Quantization of large language models with an overdetermined basis. In Negar Kiyavash and Joris M. Mooij, editors, Proceedings of the Fortieth Conference on Uncertainty in Artificial Intelligence, volume 244 of Proceedings of Machine Learning Research, pages 2527–2536. PMLR, 15–19 Jul 2024. URL https://proceedings.mlr.press/v244/merkulov24a.html.

Boris Sergeevich Kashin. Diameters of some finite-dimensional sets and classes of smooth functions. Izvestiya Rossiiskoi Akademii Nauk. Seriya Matematicheskaya, 41(2):334–351, 1977.

Boris Kashin, Ivan Oseledets, and Alexander Rudikov. Accelerated algorithm for splitting a vector into two vectors with small uniform norm. Matematicheskie Zametki, 118(3):434–442, 2025.

Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. SmoothQuant: Accurate and efficient post-training quantization for large language models. In Proceedings ofthe 40th International Conference on Machine Learning, ICML’23, 2023.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. AWQ: Activation-aware weight quantization for on-device LLM compression and acceleration. In Proceedings ofMachine Learning and Systems, 2024a.

Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Peng Gao, Yu Qiao, and Ping Luo. OmniQuant: Omnidirectionally calibrated quantization for large language models. In The Twelfth International Conference on Learning Representations, 2024.

Sehoon Kim, Coleman Hooper, Amir Gholami, Zhen Dong, Xiuyu Li, Sheng Shen, Michael W. Mahoney, and Kurt Keutzer. SqueezeLLM: Dense-and-sparse quantization. In Proceedings of the 41st International Conference on Machine Learning, ICML’24, 2024.

Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. QuaRot: Outlier-free 4-bit inference in rotated LLMs. In Advances in Neural Information Processing Systems, 2024.

Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, and Tijmen Blankevoort. Spinquant: LLM quantization with learned rotations. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=ogO6DGE6FZ.

Haokun Lin, Haobo Xu, Yichen Wu, Jingzhi Cui, Yingtao Zhang, Linzhan Mou, Linqi Song, Zhenan Sun, and Ying Wei. DuQuant: Distributing outliers via dual transformation makes stronger quantized LLMs. In Advances in Neural Information Processing Systems, 2024b.

Vladimir Temlyakov. Greedy approximation, volume 20. Cambridge University Press, 2011.

Markus Nagel, Marios Fournarakis, Rana Ali Amjad, Yelysei Bondarenko, Mart van Baalen, and Tijmen Blankevoort. A white paper on neural network quantization. ArXiv, abs/2106.08295, 2021. URL https://api.semanticscholar. org/CorpusID:235435934.

Olivier Guédon, Shahar Mendelson, Alain Pajor, and Nicole Tomczak-Jaegermann. Majorizing measures and proportional subsets of bounded orthonormal systems. Revista Matematica Iberoamericana, 24:1075–1095, 02 2008. doi:10.4171/RMI/567.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. OPT: Open pre-trained transformer language models, 2022.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models, 2023.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7B, 2023.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar van der Wal. Pythia: A suite for analyzing large language models across training and scaling. In Proceedings ofthe 40th International Conference on Machine Learning, ICML’23, 2023.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. In International Conference on Learning Representations, 2017.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Proceedings ofthe 57th Annual Meeting ofthe Associationfor Computational Linguistics, 2019.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. PIQA: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI Conference on Artificial Intelligence, 2020.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. WinoGrande: An adversarial Winograd schema challenge at scale. Communications ofthe ACM, 64(9):99–106, 2021.

Elias Frantar, Roberto L. Castro, Jiale Chen, Torsten Hoefler, and Dan Alistarh. MARLIN: Mixed-precision autoregressive parallel inference on large language models. In Proceedings ofthe 30th ACM SIGPLAN Symposium on Principles and Practice ofParallel Programming, PPoPP’25, 2025.

## A Stress Tests on Pythia-6.9B and Mistral-7B

Pythia-6.9B. Table 2 reports the full numbers for the Pythia-6.9B stress-test referenced in Section 5. Calibration setup is identical to the main table (1000 WikiText-2 train samples, sequence length 2048, three random seeds; mean ± std reported). The QuIP family fails catastrophically on this model (perplexities in the hundreds-to-thousands), while Kashin-DCT+H stays within an order of magnitude of OPTQ.

Table 2: Pythia-6.9B at 4-bit per channel. WikiText-2 / C4 perplexity (lower is better), HellaSwag / PiQA / Winogrande accuracy (higher is better). Best quantization result in bold, second best underlined (FP16/RTN excluded).
<table><tr><td>Method</td><td>Wiki-2 ↓</td><td>C4↓</td><td>Hella ↑</td><td>PiQA ↑</td><td>Wino ↑</td></tr><tr><td>FP16</td><td>11.41</td><td>30.11</td><td>63.87</td><td>76.44</td><td>61.40</td></tr><tr><td>RTN</td><td>17.55</td><td>44.59</td><td>58.86</td><td>73.01</td><td>59.12</td></tr><tr><td>OPTQ</td><td> ${ \bf 1 } 2 . { \bf 0 } 2 \pm { \bf 0 } . { \bf 0 0 }$ </td><td> ${ \bf 3 2 . 0 6 \pm 0 . 0 2 }$ </td><td> ${ \bf 6 2 . 9 5 \pm 0 . 3 2 }$ </td><td> ${ \bf 7 5 . 9 0 \pm 0 . 2 2 }$ </td><td> ${ \bf 6 0 . 7 2 \pm 0 . 1 2 }$ </td></tr><tr><td>QuIP</td><td> $2 4 8 6 . 3 9 \pm 5 6 9 . 0 0$ </td><td> $8 6 0 5 . 2 9 \pm 2 4 5 6 . 1 0$ </td><td> $3 4 . 0 1 \pm 1 . 5 6$ </td><td> $5 4 . 9 3 \pm 0 . 6 3$ </td><td> $5 3 . 0 4 \pm 0 . 9 3$ </td></tr><tr><td>QuIP-RG</td><td> $2 2 3 9 . 4 1 \pm 7 7 8 . 1 0$ </td><td> $7 3 2 3 . 5 7 \pm 2 8 3 3 . 4 0$ </td><td> $3 4 . 7 2 \pm 0 . 9 7$ </td><td> $5 5 . 5 0 \pm 0 . 3 6$ </td><td> $5 2 . 0 1 \pm 1 . 1 2$ </td></tr><tr><td>QuIP #</td><td> $3 2 4 . 9 3 \pm 4 7 . 6 3$ </td><td> $7 9 9 . 7 7 \pm 9 6 . 9 6$ </td><td> $4 1 . 1 3 \pm 0 . 4 8$ </td><td> $5 8 . 8 9 \pm 0 . 1 7$ </td><td> $5 7 . 0 1 \pm 1 . 1 5$ </td></tr><tr><td>Kashin-DCT+K</td><td> $2 5 . 9 4 \pm 0 . 7 4$ </td><td> $6 1 . 9 8 \pm 0 . 9 7$ </td><td> $5 7 . 6 4 \pm 0 . 4 1$ </td><td> $7 1 . 8 7 \pm 0 . 2 4$ </td><td> $6 0 . 3 3 \pm 0 . 2 8$ </td></tr><tr><td>Kashin-DCT+H</td><td> $2 0 . 6 4 \pm 1 . 6 3$ </td><td> $\underline { { 4 9 . 9 1 \pm 3 . 1 3 } }$ </td><td> $5 8 . 6 6 \pm 0 . 4 7$ </td><td> ${ \underline { { 7 2 . 7 2 } } } \pm 0 . 1 9$ </td><td> $\overline { { 5 9 . 9 8 \pm 0 . 0 8 } }$ </td></tr></table>

Mistral-7B v0.1. Table 3 reports per-method results on Mistral-7B v0.1 at 4-bit per channel under the same calibration setup as Table 2 (Kashin rows averaged over two seeds, GPTQ over three). The QuIP family fails to produce any output on Mistral, for the reason explained in the next paragraph. GPTQ runs to completion but degrades catastrophically (Wiki-2 PPL ≈ 380, more than 40× above the FP16 reference of 8.63), while Kashin-DCT+K and Kashin-DCT+H remain numerically stable at 0.32 and 0.29 Wiki-2 PPL above the FP16 baseline, respectively.

Table 3: Mistral-7B v0.1 at 4-bit per channel. WikiText-2 / C4 perplexity, HellaSwag / PiQA / Winogrande accuracy. “aborts” = produces NaNs and fails to complete; see paragraph below. Best non-aborting quantization result per column in bold, second best underlined.
<table><tr><td>Method</td><td>Wiki-2↓</td><td>C4↓</td><td>Hella ↑</td><td>PiQA ↑</td><td>Wino ↑</td></tr><tr><td>FP16</td><td>8.63</td><td>21.42</td><td>81.23</td><td>82.75</td><td>75.14</td></tr><tr><td>GPTQ QuIP / QuIP-RG / QuIP#</td><td> $3 7 9 . 7 3 \pm 1 5 . 7 1$ </td><td> $1 9 4 8 . 8 1 \pm 5 4 2 . 4 0$  aborts (NaN in LDL decomposition; see below)</td><td> $3 1 . 6 8 \pm 0 . 8 6$ </td><td> $6 7 . 1 7 \pm 1 . 3 4$ </td><td> $5 3 . 2 8 \pm 0 . 8 5$ </td></tr><tr><td>Kashin-DCT+K</td><td> $8 . 9 5 \pm 0 . 0 1$ </td><td> $2 2 . 2 6 \pm 0 . 0 1$ </td><td> $7 9 . 9 9 \pm 0 . 1 5$ </td><td> $8 1 . 5 9 \pm 0 . 0 4$ </td><td> $\mathbf { 7 4 . 5 5 \pm 0 . 0 6 }$ </td></tr><tr><td> $_ { \mathrm { K a s h i n - D C T + H } }$ </td><td> $\mathbf { 8 . 9 2 \pm 0 . 0 1 }$ </td><td> $\overline { { 2 2 . 2 1 \pm 0 . 0 1 } }$ </td><td> $\mathbf { \overline { { 8 0 . 0 9 \pm 0 . 0 1 } } }$ </td><td> $\overline { { 8 2 . 1 0 \pm 0 . 2 3 } }$ </td><td> ${ \underline { { 7 3 . 9 2 \pm 0 . 9 5 } } }$ </td></tr></table>

Why all OPTQ-family methods fail on Mistral. Both failures share a root cause: the SwiGLU input to Mistral’s mlp.down\_proj produces extreme channel-wise outliers, leaving the 14336 × 14336 calibration Hessian $H = X ^ { \top } X$ ill-conditioned with several eigenvalues orders of magnitude below the largest; randomized rotations do not change the spectrum, so the conditioning survives the OPTQ-default 1% diagonal damping. QuIP (LDLQ) aborts: the Cholesky factor L has near-zero diagonals, LDLQ’s unit-diagonal normalization $L \gets \dot { L } \operatorname { d i a g } ( L ) ^ { - 1 }$ amplifies off-diagonals, and the back-substitution overflows to NaN at mlp.down\_proj on both v0.1 and v0.3 (all four had/kron × ldlq/ldlqRG variants). $O P T Q$ runs to completion but silently destroys the model: the residual update

$$
W _ { : , i + 1 : } \ \gets \ W _ { : , i + 1 : } \ - \ \frac { W _ { : , i } - Q _ { : , i } } { [ H ^ { - 1 } ] _ { i i } } [ H ^ { - 1 } ] _ { i , i + 1 : }\tag{7}
$$

uses the same ill-conditioned $H ^ { - 1 }$ , swinging the yet-to-be-quantized columns far outside their pre-calibrated perchannel grid; the subsequent round-to-grid step saturates them at the endpoints (--act-order mitigates this only marginally). Kashin- $. D C T$ remains robust: the analytic centroids $\pm c _ { 1 } \pm c _ { 2 }$ scale with the actual residual norm $\| r _ { k } \| _ { 1 } / N$ so the codebook follows the inflated magnitude, and the bounded- $\boldsymbol { \cdot } \ell _ { \infty }$ factor decomposition keeps k-means wellconditioned on the same calibration data. The same mechanism is the source of stability we observe in Table 1 on the Llama-2-7B columns where LDLQ+Kron variance blows up (the daggered <sup>†</sup> cells), and in Table 2 on Pythia-6.9B where all three QuIP variants diverge into the hundreds-to-thousands PPL range while Kashin-DCT+H stays within an order of magnitude of the OPTQ baseline.

## B Original Kashin Greedy Algorithm

For completeness, we restate the original Kashin vector-decomposition greedy algorithm, on which the matrix reformu lation of Merkulov et al. [2024] and our partitioned variant in Section 4 build.

```latex
Algorithm 2 Vector Decomposition Greedy Algorithm
Input: Vector $x \in \mathbb { R } ^ { n }$ , Orthogonal matrix Q, Tolerance $\varepsilon > 0$
Output: Vectors $u , \hat { v } \in \mathbb R ^ { n }$ such that $\boldsymbol { x } \approx \boldsymbol { u } + \boldsymbol { v } = \boldsymbol { u } + Q ^ { T } \hat { \boldsymbol { v } } ,$ , and both u and hatv have small infinity norm.
Initialize $u  0 ^ { n } , \hat { v }  0 ^ { n }$
Define projection $\pi _ { x } ( y ) : = \frac { \boldsymbol { x } ^ { \top } \boldsymbol { y } } { \| \boldsymbol { y } \| _ { 2 } ^ { 2 } } \cdot \boldsymbol { y }$
while $\| x \| \geq \varepsilon$ do
if $\| \ddot { x } \| _ { 1 } > \| Q x \| _ { 1 }$ then
$\pi  \pi _ { x } ( { \mathrm { S i g n } } ( x ) )$
$u  u + \pi$
$x  x - \pi$
else
$\pi  \pi _ { x } ( \operatorname { S i g n } ( Q x ) )$
${ \hat { v } }  { \hat { v } } + \pi$
$x  x - Q ^ { T } \pi$
end if
end while
Return: $x , u , \hat { v }$
```

## C Inference

The Kashin representation $w = u + P \hat { v }$ stores each layer as two 2-bit factor matrices $U , \hat { V } \in \mathbb { R } ^ { N \times M }$ together with the per-layer sign mask $\varepsilon \in \{ \pm 1 \} ^ { N }$ that defines the orthogonal operator $P = \mathrm { I D C T o } T _ { \varepsilon } \circ \mathrm { D C T }$ . Because ε is shared across all columns of a layer (generated once from a fixed seed), a single P acts on every column of $\hat { V } .$ With the orthonormal DCT-II<sup>4</sup>, P is a symmetric involution $( P ^ { \top } = P$ and $P ^ { 2 } = I$ , since $T _ { \varepsilon }$ is diagonal with $\varepsilon _ { i } ^ { 2 } = 1 )$ . Associativity of the matmul then gives the folding identity

$$
X W = X { \big ( } U + P { \hat { V } } { \big ) } = X U + ( X P ) { \hat { V } } = { \big [ } X { \big | } X P { \big ] } { \binom { U } { \hat { V } } } ,\tag{8}
$$

where $X P$ is computed by applying the same FFT-fast ‘iDCT ◦ sign-mask ◦ DCT’ pipeline used for $P \hat { v }$ during quantization, just to each row of X rather than to a column of $\hat { V }$

Compute. The right path through (8) depends on hardware. Without native 2-bit GEMM support, it is more efficient to reconstitute the dense weight $W = U + P \hat { V }$ once at load time and run a standard fp16 (or 4-bit-dequant) matmul

XW. With native 2-bit GEMM (e.g. Hopper-class FP4/INT2 tensor cores), the fused form of (8) is preferable: keeping U and $\hat { V }$ in their 2-bit storage and computing $X U + ( X P ) { \hat { V } }$ as a single 2N-K-dimension 2-bit GEMM avoids the fp16 expansion of the weight body and preserves the on-chip footprint advantage of 2-bit codes. In both regimes the DCT cost is O(KN log N) for $\bar { K = ( \mathrm { b a t c h } ) }$ · (seq len) activation rows, several orders of magnitude below the $\mathcal { O } ( K N M )$ matmul.

Native 2-bit hardware. On accelerators with native 2-bit tensor cores, the decomposed form gains two kernel-level advantages over a standard 4-bit per-channel kernel: (i) codebook dequantization maps cleanly to the 2-bit addressing unit – the lookup c[ code ] replaces $( \mathrm { c o d e } - z _ { p } )$ · s operation of uniform quantization; (ii) each 2-bit code is the natural register-packing and shared-memory granule, so the centroid table loads contiguously without the cross-lane shuffles a 4-bit dequant with zero-point typically requires.

Memory bandwidth. At 4-bit per channel, each layer stores 4NM weight bits (2NM for each of $U$ and $\hat { V } )$ Symmetric centroids about zero in both factors permit the per-column metadata to be reduced to two signed magnitudes per factor (4M fp16 values per layer), plus a layer-shared sign vector $\varepsilon \in \{ \pm 1 \} ^ { N }$ , which can be recovered from the seed rather than stored. The effective bits-per-weight is therefore $( 4 N M + \dot { 6 } 4 \dot { M ^ { ) } } / ( N M ) = 4 + 6 4 / N ;$ ; for typical LLM hidden dimensions $N \in \left[ 4 0 9 6 , 8 1 9 2 \right]$ this metadata adds at most $6 4 / 4 0 9 6 \approx \dot { 0 } . 0 1 6$ bits per weight on top of the 4-bit weight body. By comparison, standard 4-bit per-channel quantization stores one fp16 scale per column and (optionally) one fp16 zero-point, yielding $4 + 1 6 / N$ bits per weight without zero-point and $4 + 3 2 / \bar { N }$ with it – equivalently, an additional 0.004 and 0.008 bits per weight at ${ \bar { N } } = 4 0 { \bar { 9 } } 6$

## D Quantization Runtime

Table 4 reports the end-to-end wall-clock time required to quantize Llama-2-7B and Llama-2-13B at 4 bits per channel for the three methods we ran on identical hardware: OPTQ, the fine-tuning- and vector-quantization-free variant of QuIP# (LDLQ-RG with randomized Hadamard incoherence preprocessing), and our Kashin-DCT+H pipeline. All measurements are taken on a single NVIDIA H100 GPU; mean ± standard deviation is computed over three random seeds. The reported time does not include Hessian estimation.

Table 4: Quantization wall-clock time at 4 bits per channel. Single NVIDIA H100 GPU; mean ± std over three random seeds.
<table><tr><td>Method</td><td></td><td>Llama-2-7B (s) ↓ Llama-2-13B (s) ↓</td></tr><tr><td>OPTQ</td><td> $2 2 9 . 9 \pm 2 . 2$ </td><td> $4 3 7 . 1 \pm 2 . 1$ </td></tr><tr><td>QuIP# (LDLQ-RG + Hadamard)</td><td> $7 6 3 . 0 \pm 5 . 3$ </td><td> $1 2 9 8 . 0 \pm 6 . 1$ </td></tr><tr><td>Kashin-DCT+H</td><td> $1 3 1 3 . 6 \pm 2 6 . 7$ </td><td> $1 5 4 4 . 9 \pm 1 0 . 8$ </td></tr></table>

Scaling 7B → 13B. Going from Llama-2-7B to Llama-2-13B (a ∼ 1.9× parameter increase), OPTQ and QuIP# wall-clock times scale roughly proportionally $- 1 . 9 0 \times ( 2 2 9 . 9 \to 4 3 7 . 1 { \mathrm { s } } )$ and $1 . 7 0 \times ( 7 6 3 . 0 \to 1 2 9 8 . 0 \mathrm { s } )$ , respectively – whereas Kashin-DCT+H grows only $1 . 1 8 \times ( 1 3 1 3 . 6 \to 1 5 4 4 . 9 \mathrm { s } )$ . Concretely, doubling the model roughly doubles the OPTQ and QuIP# quantization budget but adds only ∼ 18% to Kashin-DCT+H, so the relative cost of the proposed method becomes more favorable as model size grows.

## E Proof of Proposition 1

The dictionary S is partitioned into two halves:

$$
S = Q _ { N } \bigcup U \varepsilon , _ { \Phi } Q _ { N } .
$$

The partitioned greedy algorithm alternates between these halves: within each block, the first two steps use vectors from $Q _ { N }$ , and the next two steps use vectors from $U _ { \mathcal { E } , \Phi } Q _ { N }$ . To establish the convergence rate, it suffices to analyze the error reduction over a single block.

Fix one such block and denote by $\rho _ { 0 }$ the residual at its start; thus $\rho _ { 0 } = x$ and $\| \rho _ { 0 } \| _ { 2 } \leqslant A$ . For $j \in \{ 1 , 2 , 3 , 4 \}$ , let $\rho _ { j }$ be the residual after the j-th step inside this block. In the notation of Proposition 1 we have $r _ { 1 } = \rho _ { 4 }$

By (3), for $\rho _ { 0 }$ at least one of the following conditions holds:

(i)

$$
\| \rho _ { 0 } \| _ { 1 } \geqslant { \frac { R ( N ) } { K { \sqrt { 2 } } } } A ,\tag{9}
$$

(ii)

$$
\| U _ { \mathcal { E , \Phi } } \rho _ { 0 } \| _ { 1 } \geqslant \frac { R ( N ) } { K \sqrt 2 } A .\tag{10}
$$

We first record the standard estimate used in Kashin et al. [2025]. Suppose $\begin{array} { r } { \| y \| _ { 1 } \geqslant \frac { R ( N ) } { K \sqrt { 2 } } \| y \| _ { 2 } } \end{array}$ . For any $w \in Q _ { N }$ we have $\lVert \boldsymbol { w } \rVert _ { 2 } ^ { 2 } = N$ , and the maximum ma $\mathrm { x } _ { w \in Q _ { N } } | \langle y , w \rangle | = \| y \| _ { 1 }$ is attained by $w _ { j } ^ { * } = \mathrm { s i g n } ( y _ { j } )$ . Hence the greedy step satisfies

$$
\left\| y - \lambda ^ { * } w ^ { * } \right\| _ { 2 } ^ { 2 } = \| y \| _ { 2 } ^ { 2 } - \frac { \| y \| _ { 1 } ^ { 2 } } { N } \leqslant \| y \| _ { 2 } ^ { 2 } \Big ( 1 - \frac { R ^ { 2 } ( N ) } { 2 K ^ { 2 } N } \Big ) = \| y \| _ { 2 } ^ { 2 } ( 1 - \alpha ( N ) ) .\tag{11}
$$

Since $U _ { \mathcal { E } , \Phi } ^ { 2 } = I \left( \mathrm { a s } \mathcal { F } _ { \Phi } \right.$ is orthogonal transformation $. , T _ { \mathcal { E } }$ is diagonal, $T _ { \mathcal { E } } ^ { 2 } = I )$ , we have $\langle y , U \varepsilon , \Phi q \rangle = \langle U \varepsilon , \Phi y , q \rangle$ for any $q \in Q _ { N }$ , so an identical argument applies to $U _ { \mathcal { E } , \Phi } Q _ { N }$ when condition (10) holds.

Case 1: condition (9) holds. By (11), the first greedy step (from $Q _ { N } )$ gives $\| \rho _ { 1 } \| _ { 2 } \leqslant A \sqrt { 1 - \alpha ( N ) }$ . The remaining three steps are non-expansive (each is an orthogonal projection onto a one-dimensional subspace), so

$$
\| r _ { 1 } \| _ { 2 } = \| \rho _ { 4 } \| _ { 2 } \leqslant \| \rho _ { 1 } \| _ { 2 } \leqslant A \sqrt { 1 - \alpha ( N ) } \leqslant A \sqrt { 1 - \beta ( N ) } .
$$

Case 2: condition (9) fails. By (3), condition (10) must then hold.

Sub-case 2a: $\| \rho _ { 2 } \| _ { 2 } < A \sqrt { 1 - \beta ( N ) }$ . The third and fourth steps are non-expansive, hence $\| r _ { 1 } \| _ { 2 } = \| \rho _ { 4 } \| _ { 2 } \leqslant \| \rho _ { 2 } \| _ { 2 } \leqslant$ $A { \sqrt { 1 - \beta ( N ) } }$ , and (5) holds.

Sub-case 2b: $\| \rho _ { 2 } \| _ { 2 } \geqslant A \sqrt { 1 - \beta ( N ) }$ . By the property of the pure greedy algorithm $( \rho _ { j } \perp w _ { j }$ at each step j),

$$
\| x \| _ { 2 } ^ { 2 } = \| \rho _ { 2 } \| _ { 2 } ^ { 2 } + \| \lambda _ { 1 } w _ { 1 } \| _ { 2 } ^ { 2 } + \| \lambda _ { 2 } w _ { 2 } \| _ { 2 } ^ { 2 } .
$$

Combining with $\| x \| _ { 2 } \leqslant A$ and $\| \rho _ { 2 } \| _ { 2 } \geqslant A \sqrt { 1 - \beta ( N ) }$ yields

$$
\| \lambda _ { 1 } w _ { 1 } \| _ { 2 } ^ { 2 } + \| \lambda _ { 2 } w _ { 2 } \| _ { 2 } ^ { 2 } \leqslant A ^ { 2 } \beta ( N ) ,
$$

so the triangle inequality gives

$$
\lVert \lambda _ { 1 } w _ { 1 } + \lambda _ { 2 } w _ { 2 } \rVert _ { 2 } \ \leqslant \ \lVert \lambda _ { 1 } w _ { 1 } \rVert _ { 2 } + \lVert \lambda _ { 2 } w _ { 2 } \rVert _ { 2 } \ \leqslant \ A \sqrt { 2 \beta ( N ) } .
$$

Let $\lambda ^ { * } w ^ { * }$ be the best approximation of $x$ from $U _ { \mathcal { E } , \Phi } Q _ { N }$ . Since condition (10) holds, the one-step estimate gives $\| x - \lambda ^ { * } w ^ { * } \| _ { 2 } \leqslant A \sqrt { 1 - \alpha ( N ) }$ . Because the actual greedy choice at the third step is at least as good as $\lambda ^ { * } w ^ { * }$ , the triangle inequality yields

$$
\begin{array} { r l } { \| \rho _ { 3 } \| _ { 2 } \leqslant \| \rho _ { 2 } - \lambda ^ { * } w ^ { * } \| _ { 2 } \leqslant \| x - \lambda ^ { * } w ^ { * } \| _ { 2 } + \| \lambda _ { 1 } w _ { 1 } + \lambda _ { 2 } w _ { 2 } \| _ { 2 } } & { } \\ { \leqslant A \sqrt { 1 - \alpha ( N ) } + A \sqrt { 2 \beta ( N ) } } & { } \\ { \leqslant A \big ( 1 - \frac { \alpha ( N ) } { 2 } + \frac { \alpha ( N ) \sqrt { 2 } } { 6 } \big ) \leqslant A \big ( 1 - \frac { \alpha ( N ) } { 4 } \big ) \leqslant A \sqrt { 1 - \beta ( N ) } . } \end{array}\tag{12}
$$

The fourth step is non-expansive, hence

$$
\| r _ { 1 } \| _ { 2 } = \| \rho _ { 4 } \| _ { 2 } \leqslant \| \rho _ { 3 } \| _ { 2 } \leqslant A \sqrt { 1 - \beta ( N ) }\tag{13}
$$

In all cases (5) is established, completing the proof.

## F Proof of Theorem 2

We prove

$$
\| r _ { k } \| _ { 2 } ~ = ~ \| x - u _ { k } - v _ { k } \| _ { 2 } ~ \leqslant ~ \big ( 1 - \beta ( N ) \big ) ^ { k / 2 }\tag{14}
$$

by induction on k. The case $k = 0$ is immediate: $\| r _ { 0 } \| _ { 2 } = \| x \| _ { 2 } \leqslant 1 = ( 1 - \beta ( N ) ) ^ { 0 } .$

Assume (14) holds for some $k \geqslant 0$ and consider the $( k + 1 ) \cdot \mathrm { t h }$ block, which starts from $\rho _ { 0 } ~ = ~ r _ { k }$ and produces $r _ { k + 1 } = \rho _ { 4 }$ . Setting $A = ( 1 - \beta ( N ) ) ^ { k / 2 }$ in Proposition 1, we obtain

$$
\lVert r _ { k + 1 } \rVert _ { 2 } \ \leqslant \ A \sqrt { 1 - \beta ( N ) } \ = \ \bigl ( 1 - \beta ( N ) \bigr ) ^ { ( k + 1 ) / 2 } ,
$$

which closes the induction and proves the first line of (6).

Recall that within the j-th block the algorithm performs four atomic steps: two from $Q _ { N }$ (contributing to u) and two from $U _ { \mathcal { E } , \Phi } Q _ { N }$ (contributing to v). Let $w _ { j } ^ { ( 1 ) } , \bar { w _ { j } ^ { ( 2 ) } } \in Q _ { N }$ and $w _ { j } ^ { ( 3 ) } , w _ { j } ^ { ( 4 ) } \in U _ { \mathcal { E } , \Phi } Q _ { N }$ denote the dictionary elements chosen in block $j ,$ with corresponding coefficients $\lambda _ { j } ^ { ( 1 ) } , \ldots , \lambda _ { j } ^ { ( 4 ) }$ . By construction,

$$
u _ { k } \ = \ \sum _ { j = 1 } ^ { k } \sum _ { \ell \in \{ 1 , 2 \} } \lambda _ { j } ^ { ( \ell ) } w _ { j } ^ { ( \ell ) } , \quad v _ { k } \ = \ \sum _ { j = 1 } ^ { k } \sum _ { \ell \in \{ 3 , 4 \} } \lambda _ { j } ^ { ( \ell ) } w _ { j } ^ { ( \ell ) } .\tag{15}
$$

Coefficient bound. At every greedy step the chosen atom w satisfies $\lVert \boldsymbol { w } \rVert _ { 2 } ^ { 2 } = N$ and the associated coefficient equals $\lambda { = } \tilde { \langle \rho , w \rangle } / N$ , where $\rho$ is the residual entering that step. Hence by Cauchy–Schwarz,

$$
| \lambda _ { j } ^ { ( \ell ) } | \leqslant \frac { \| \rho _ { j , \ell - 1 } \| _ { 2 } \| w \| _ { 2 } } { \| w \| _ { 2 } ^ { 2 } } = \frac { \| \rho _ { j , \ell - 1 } \| _ { 2 } } { \sqrt { N } } ,\tag{16}
$$

where $\rho _ { j , \ell - 1 }$ is the residual just before step ℓ of block $j .$ Since atomic steps are non-expansive, $\| \rho _ { j , \ell - 1 } \| _ { 2 } \leqslant \| r _ { j - 1 }$ ∥<sub>2</sub>, and by part 1, $\| r _ { j - 1 } \| _ { 2 } \leqslant ( 1 - \beta ( N ) ) ^ { ( j - 1 ) / 2 }$ . Therefore

$$
| \lambda _ { j } ^ { ( \ell ) } | \ \leqslant \ \frac { ( 1 - \beta ( N ) ) ^ { ( j - 1 ) / 2 } } { \sqrt { N } } .\tag{17}
$$

Summing the contributions. Every atom $w \in Q _ { N }$ has $\| w \| _ { \infty } = 1$ . By (15) and (17),

$$
\begin{array} { r l r } {  { \| u _ { k } \| _ { \infty } \leqslant \sum _ { j = 1 } ^ { k } \sum _ { \ell \in \{ 1 , 2 \} } | \lambda _ { j } ^ { ( \ell ) } | \leqslant \frac { 2 } { \sqrt { N } } \sum _ { j = 1 } ^ { k } \big ( 1 - \beta ( N ) \big ) ^ { ( j - 1 ) / 2 } \leqslant } } \\ & { } & { \leqslant \frac { 2 } { \sqrt { N } } \cdot \frac { 1 } { 1 - \sqrt { 1 - \beta ( N ) } } \leqslant \frac { 4 } { \beta ( N ) \sqrt { N } } . } \end{array}\tag{18}
$$

The same bound holds for $\| U _ { \mathcal { E } , \Phi } v _ { k } \| _ { \infty }$ since $U _ { \mathcal { E } , \Phi } ^ { 2 } = I$ implies that the action of $U _ { \mathcal { E } , \Phi }$ on the second sum in (15) produces atoms in $Q _ { N }$ with the same $\ell _ { \infty }$ -norm bound.

Combining (15) with the fact that $\| w \| _ { \infty } ~ = ~ 1$ for every $w \in \mathcal S$ and substituting $\beta ( N ) ~ = ~ \alpha ^ { 2 } ( N ) / 3 6 ~ =$ $R ^ { 4 } ( N ) / ( \stackrel { \smile } { 1 4 4 } K ^ { 4 } N ^ { 2 } ) = 1 / ( 1 4 4 c _ { 3 } ^ { 4 } K ^ { 4 ^ { \prime } } ( \stackrel { \dots } { \log } N ) ^ { 2 } ( \log \log N ) ^ { 1 \breve { 2 } } )$ , we conclude

$$
\operatorname* { m a x } \Bigl ( \| u _ { k } \| _ { \infty } , \| U _ { \mathcal { E } , \Phi } v _ { k } \| _ { \infty } \Bigr ) \leqslant \sum _ { j , \ell } | \lambda _ { j } ^ { ( \ell ) } | \leqslant \frac { 4 } { \beta ( N ) \sqrt { N } } = \frac { c _ { 4 } K ^ { 4 } ( \log N ) ^ { 2 } ( \log \log N ) ^ { 1 2 } } { \sqrt { N } } ,
$$

where $c _ { 4 } = 5 7 6 c _ { 3 } ^ { 4 }$ is an absolute constant. This is exactly the second line of (6).