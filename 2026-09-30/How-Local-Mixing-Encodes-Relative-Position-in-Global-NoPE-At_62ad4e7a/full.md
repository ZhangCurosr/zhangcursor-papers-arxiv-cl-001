# How Local Mixing Encodes Relative Position in Global NoPE Attention

Cutter Dawes<sup>1,\*</sup>, Nick Alonso<sup>1,\*</sup>, Tom Figliolia<sup>1</sup>, Beren Millidge<sup>1</sup>

<sup>1</sup> Zyphra Research \* Equal contribution

Abstract—The attention operation is naively position invariant. However, positional information is fundamental to natural language, and therefore a variety of explicit position encodings have been developed in transformer-based models, such as rotary position encoding (RoPE). Although explicit position encodings have long been assumed to be required, recent methods that interleave local mixing layers, such as sliding window attention (SWA) and gated linear attention, while not encoding position (NoPE) in global attention layers has recently been shown to be successful at scale. How and why this approach works is not well-understood. In this paper, we develop an explanation of how hybrid models of this sort can implicitly encode position at global NoPE layers. Supported by both theoretical and empirical evidence, our central argument is that SWA and gated linear attention induce a recency bias in the residual stream that propagates to, and is selected by, the global attention logits. Moreover, in contrast to the implicit position encodings found in models with only global NoPE attention, in which positional information arises solely from the causal mask, the recency bias in hybrid models can be maintained across long sequences. In addition to deepening our understanding of how hybrid models encode position, these findings may provide insights for how to encode position in a way that can extrapolate to longer sequence lengths indefinitely.

## I. INTRODUCTION

Distinguishing tokens by their position is a vital component of language modeling (Dufter et al., 2022). Since self-attention by itself is permutation invariant, attention layers in large language models (LLMs) typically utilize position encodings (PEs), which explicitly represent either the absolute or relative position of queries and keys. However, commonly used PEs such as rotary positional encoding (RoPE; Su et al., 2024) incur computational overhead and make length extension and extrapolation difficult. Removing explicit PE (a method called NoPE) has met with mixed success: NoPE does not match the in-domain performance of the same architecture with explicit PE, although it can provide some improvements in length extrapolation whereas the performance of RoPE and other explicit position embeddings tend to collapse quickly beyond their training sequence length (Haviv et al., 2022; Kazemnejad et al., 2023).

Separately, to mitigate the computational and memory costs of attention across long sequences, so-called hybrid architectures that interleave either sliding window attention (SWA) or gated linear layers with global attention layers are increasingly used at scale. Interestingly, as hybrid models have become more widespread, there has been a shift in the global attention layers from using RoPE (Gemma Team, 2024) to p-RoPE (Qwen Team, 2025) to, most recently, NoPE. Notably,

Kimi K3 interleaved global NoPE with a gated linear variant called KDA (Kimi Team et al., 2026).

However, there are remaining questions concerning how such models utilize and represent position in their NoPE attention layers, an issue that is critical to their long context performance. To our knowledge, only one prior paper has studied how hybrid models represent position in NoPE layers, and their main finding is that they do not represent absolute position (Puvvada et al., 2025) at all. However, this result leaves open the question of whether such models may learn to encode relative position in their NoPE layers.

In this paper, we present novel theoretical and empirical evidence that hybrid architectures can and do develop relative PEs at their NoPE layers. Unlike other explicit relative PEs such as RoPE, hybrid architectures implicitly encode relative position via the interactions between local or linear layers and subsequent NoPE layers. Specifically, the SWA or gated linear layers introduce a recency bias in the residual stream that propagates to, and is selected by, the global NoPE attention logits and then accentuated through depth. We support this account in two complementary ways:

(i) Theoretically, we show mathematically how SWA approximates a moving-average convolution in expectation, inducing a recency bias in its output that transfers into the residual stream, propagates through intervening layers, and is selected by the global NoPE attention logits.

(ii) Empirically, we demonstrate the existence of the predicted recency bias in the residual stream and NoPE attention logits across depth. We show this bias is present at initialization and strengthens throughout training. We validate this in hybrid models interleaving global NoPE attention with SWA (with RoPE and NoPE) and KDA (Appendix E)

An overview of our hypothesized mechanism is provided in Figure 1. In the following sections, we first note that existing relative PEs generally induce a recency bias (Section II), then we introduce our core argument in three steps (Section III). First, SWA produces a residual stream recency bias (Section IV); second, this recency bias propagates through the intervening modules such as normalization and multi-layer perceptrons (MLPs; Section V); and third, this recency bias propagates to and is selected by the global NoPE attention logits (Section VI).

![](images/c80d2af1dc82e0c974d3cff25ec94d8e658cefa852b0ecc1c9cb9f69831cbc83.jpg)  
Figure 1: Proposed mechanism for implicit relative position encoding. (a) Tokens begin as initially uncorrelated. (b) Local mixing correlates nearby residual states. (c) Learned query and key projections can read out this lag-dependent structure as recency-biased global NoPE logits. The shading intensity denotes correlation or logit strength, while line width denotes mixing or attention weight.

## II. BACKGROUND: RECENCY BIAS AS RELATIVE POSITION ENCODING

Relative PEs, as opposed to absolute PEs, are the industry standard in LLM architectures (Shaw et al., 2018; Dufter et al., 2022; Su et al., 2024). These PEs alter attention logits, directly or indirectly, in a way that depends on the relative distance between query and key. There are a variety of methods for encoding relative positions in attention, but the majority share the following feature:

Common and state-of-the-art relative position encoding methods encode relative position by creating a recency bias in attention logits. That is, given a query q<sub>i</sub> at position i, and a key $k _ { j }$ at j, relative position encodings bias the attention logits downward as the relative distance $i - j$ increases.

Relative PEs encoding a recency bias is not strictly necessary (Chen et al., 2025). In principle, a relative PE applied to attention can induce any kind of bias in attention logits, as long as this bias depends directly on relative position. However, using a recency bias to encode relative position appears to work well in practice, because: (i) it provides a particularly simple positional signal for the model to learn and use (i.e., in which distance is represented directly by a smooth, monotonic change in the attention strength); and (ii) it fits the statistics of natural sequential data (i.e., the tokens most important for predicting

next tokens tend to be the most recent ones; Tan et al., 2025;   
Wu et al., 2025; Kim et al., 2026).

Survey of relative PEs. There are generally two ways to create a relative PE in attention logits: additively or multiplicatively (Zhang et al., 2026). Additive encodings directly add a bias, $b _ { i - j }$ , to attention logits; i.e., $s = q _ { i } ^ { \top } k _ { j } + b _ { i - j } .$ , where i and j are the position of query and key and $i - j$ the relative position. Examples include Alibi and Fire (Press et al., 2022; Li et al., 2024), which add a fixed, data-independent bias that decays linearly with relative distance, as well as the more recent Fox PE (Lin et al., 2025) that uses a data-dependent bias. All of these explicitly induce a recency bias in attention logits, such that keys farther from the query have a larger negative bias added to their attention logits. Multiplicative PEs include a matrix, $R _ { i - j }$ , in the query-key multiplication; i.e., $s = q _ { i } ^ { \top } R _ { i - j } k _ { j }$ , where $R _ { i - j }$ is unique to relative position $i - j$ . RoPE, arguably the best-known relative PE, rotates pairs of channels using rotation matrices in both the queries and keys. It has been observed that learned attention heads with RoPE either focus on very low-frequency channels with little resulting influence on logit decay, or they focus on highfrequency channels in such a way that they create a strong recency bias (Barbero et al., 2025). Other multiplicative PEs, like the recent Wall attention (Yang et al., 2025), create a similar kind of recency bias in a way that is data-dependent (Yang et al., 2025; Zhang et al., 2026).

PE in NoPE. Among previous work studying the position encoding resulting from NoPE, two papers are particularly relevant to this work. First, Puvvada et al. (2025) studied implicit position encodings in models with global NoPE attention, including hybrid architectures that interleave global NoPE with SWA. In particular, they trained probes to predict absolute position encoding in the residual stream embeddings, finding that the probes could predict absolute position in global NoPE architectures but not in hybrid architectures (see Haviv et al., 2022; Chi et al., 2023 for more evidence of absolute position encodings in global NoPE architectures). This provides evidence that hybrid NoPE architectures, unlike their global NoPE counterparts, do not encode absolute position, but leaves open the possibility that hybrid architectures could encode relative positions.

Second, Zuo et al. (2025) showed that architectures with global NoPE attention at every layer could learn a recency biasbased PE in their residual stream, created by the asymmetric token mixing effects of attention with a causal mask. However, their analysis and empirical results only demonstrate a recency bias effect on very short sequences (approximately 30 tokens), leaving open the question of how this bias may change with sequence length. Additionally, they do not explain the conditions under which attention can make use of this recency bias in the residual stream. In this work, we apply a similar analysis to hybrid architectures with SWA layers and compare to global NoPE models. Importantly, we show that SWA has a recency bias effect on the residual stream independent of total sequence length, whereas for initialized global NoPE residuals the recency bias loses resolution even at moderate sequence lengths. Further, unlike previous works, we explain the conditions required of attention parameters to make use of this recency bias in the residual stream, and we scale our empirical results to models with 350M+ parameters trained on natural language data.

III. THESIS: HYBRID ARCHITECTURES IMPLICITLY ENCODE RELATIVE POSITION AT GLOBAL NOPE LAYERS

We develop and support the following thesis about hybrid architectures with global NoPE attention:

Local layers (e.g., SWA) output a sequence of vectors which, in expectation, have a recency bias (i.e., smaller cosine similarity with greater relative distance). These outputs then create a recency bias in the post-attention residual stream that is persistent across sequence lengths. This residual stream bias can (and does after training) get transferred through the intervening layers and to attention logits at NoPE layers, thereby creating an implicit relative position encoding in NoPE layers.

We provide two lines of evidence that present complementary views of this effect. The theoretical analysis isolates the structural effect of one type of local layer (SWA) and outlines how the resulting recency bias can propagate through intervening modules and into the global NoPE logits. However, our analysis does not fully account for the variety of computation the model is being trained for, particularly the effects of contentdependent attention and MLPs. Our empirical analysis then verifies the existence of the residual stream and logit-based recency biases in multiple models, at varying stages of training, on both random and natural language data (for random-token controls, see Appendix D-B). We also show how these effects generalize to hybrid architectures that use KDA (Appendix E) as the local layer instead of SWA. Below, we first provide theoretical and then empirical preliminaries common to each step in our proposed mechanism.

## A. Theory

First, we formally introduce several notions of similarity in the residual stream, and what we mean by a recency bias. Let $X \ = \ [ x _ { 1 } , \dots , x _ { T } ] \ \in \ \mathbb { R } ^ { D \times T }$ be the residual stream at some layer. The most simple and intuitive measurement of similarity is cosine similarity; $\begin{array} { r } { \mathrm { i . e . , } c ( d ) = \mathbb { E } \big [ x _ { i } ^ { \top } x _ { i - d } \big / | | x _ { i } | | | x _ { i - d } | | \big ] } \end{array}$ However, cosine similarity captures just one slice of the full similarity structure; in particular, it does not account for crosschannel correlations that may be picked up by the global attention projections. Therefore, we also introduce the lagged cross-moment matrix, $G _ { X } ( d ) = \mathbb { E } [ x _ { i } x _ { i - d } ^ { \top } ]$ , where the $x _ { i }$ are unit-normalized.

To obtain scalar measurements of similarity, one can consider various slices of this matrix; i.e., the Frobenius inner product $\langle G _ { X } ( d ) , M \rangle _ { F } = \operatorname { t r } ( G _ { X } ( d ) ^ { \top } M )$ with some matrix M. Different slices of the lagged cross-moment correspond precisely to our similarity measures of interest: cosine similarity is the isotropic component, and the NoPE logits capture alignment with the composed query-key projection. That is,

$$
\begin{array} { r l } { \mathrm { C o s i n e : ~ } } & { { } c ( d ) = \langle G _ { X } ( d ) , I \rangle _ { F } , } \\ { \mathrm { N o P E ~ l o g i t s : ~ } } & { { } \ell ( d ) = \langle G _ { Z } ( d ) , M \rangle _ { F } . } \end{array}
$$

Here, I is the identity matrix, Z is the actual post-normalization input to a NoPE head at its (unnormalized) model scale, and $M = W _ { Q } ^ { \top } W _ { K } / \sqrt { D _ { H } }$ with head dimension $D _ { H }$ . Given one of the above scalar similarity profiles $f ( d )$ , we compute the recency gap as $f ( 1 ) - f ( L )$ , where L is the far-distance reference lag (4096 in our experiments); then, $f ( L )$ is the similarity floor, or the underlying correlation structure in the sequence present at long distances.

## B. Empirics

For the empirical analysis, we study hybrid architectures at both the 120M and 350M scales, varying the SWA window size w ∈ {64, 128, 256, 512, 1024, 2048, 4096} and comparing to a global NoPE baseline (i.e., no SWA interleave) at each scale. Unless otherwise noted, the SWA layers use RoPE and are interleaved with global NoPE at a 1:1 ratio. The main text shows results for the 350M models, and the $w = 1 2 8$ variant for single-model analyses. We train the models on the Prolong dataset (Gao et al., 2025). For evaluations periodically throughout training, we use TextbookChapters (Chevalier et al., 2024).

Using the measures defined in Section III-A, we track residual stream cosine similarity and global NoPE attention logits across training, depth, and SWA window size. For more details on the experimental setup, including training setup, statistics sampling, and head aggregation, see Appendix A.

## IV. STEP 1: SWA PRODUCES A RECENCY BIAS IN THE RESIDUAL STREAM

## A. Theory

Unlike full attention, SWA restricts its attention operation to a local window of keys and values. Here, we show that this local operation gives the attention output a distance-dependent cosine similarity profile. Furthermore, because its scale is fixed by the window size rather than the sequence length (in contrast to full attention), this effect does not dilute as the total sequence length grows.

For a single attention head, define $q _ { i } = W _ { Q } x _ { i } , k _ { i } = W _ { K } x _ { i } .$ $v _ { i } = W _ { V } x _ { i } .$ For a given $q _ { i }$ and $k _ { i - d } ,$ the attention weight at distance d is

$$
\alpha _ { i } ( d ) = \frac { \exp \bigl ( q _ { i } ^ { \top } k _ { i - d } / \sqrt { D _ { H } } \bigr ) } { \sum _ { r = 0 } ^ { w - 1 } \exp \bigl ( q _ { i } ^ { \top } k _ { i - r } / \sqrt { D _ { H } } \bigr ) } , \qquad 0 \le d < w ,
$$

where w is the sliding window size and $D _ { H }$ the head dimension. Writing $o _ { i } = W _ { O } W _ { V } x _ { i }$ , the SWA output is

$$
y _ { i } = W _ { O } \sum _ { d = 0 } ^ { w - 1 } \alpha _ { i } ( d ) v _ { i - d } = \sum _ { d = 0 } ^ { w - 1 } \alpha _ { i } ( d ) o _ { i - d } .
$$

Hence, SWA acts akin to a data-dependent convolution.

Intuition. To illustrate why such a local filter may produce a recency bias in its output, consider a simplified scenario that closely reflects $\mathbf { S W A }$ in an LLM at initialization, in which the projected values $o _ { i }$ are zero-mean, of similar magnitude, and uncorrelated across positions, while the attention weights are roughly uniform. Then we may approximate the SWA output as a moving average,

$$
y _ { i } \approx \frac { 1 } { w } \sum _ { r = 0 } ^ { w - 1 } o _ { i - r } .
$$

In this simplified scenario, the cosine similarity of two outputs, say $y _ { i } = { \textstyle \frac { 1 } { w } } ( o _ { i - w + 1 } + \cdot \cdot \cdot + o _ { i } )$ and $\begin{array} { r } { y _ { i - d } = \frac { 1 } { w } ( o _ { i - d - w + 1 } + } \end{array}$ $\cdots + o _ { i - d } )$ , depends on the fraction of projected values shared by their sums. The closer they are in position, the larger this shared fraction; $\mathrm { e . g . }$ , for a window size of $6 4 , y _ { i }$ shares 63 terms with $y _ { i - 1 } , 6 2$ with $y _ { i - 2 } ,$ and so on. Thus, the expected output cosine similarity decreases with relative distance and approaches zero once the windows no longer overlap.

Length scaling and comparison with global NoPE. As noted in Section II, Zuo et al. (2025) did a similar analysis, but for global NoPE architectures; i.e., they showed that a global NoPE attention layer could induce an adjacency pattern in its outputs. However, they considered only tiny sequences (∼ 30 tokens) and did not explore whether this effect scaled with sequence length. Here, the uniform-attention toy case helps explain why global NoPE cannot sustain a well-resolved recency bias even out to 1000-token sequences. Specifically, the output of a global NoPE layer will be

$$
y _ { i } \approx \frac { 1 } { i + 1 } \sum _ { t = 0 } ^ { i } o _ { t } .
$$

Unlike SWA, the model averages over the entire preceding sequence. Therefore, ${ \mathrm { i f ~ } } i = 1 0 0 0$ , then $y _ { i }$ and $y _ { i - 1 0 }$ share 99% of their terms, so the uniform-attention cosine changes only slightly over fixed lags. Together, these calculations predict a persistent window-scale recency profile for SWA, but not a comparably resolved fixed-lag profile for global NoPE.

To make the contrast with SWA more concrete, under the same initialization-like assumptions as above, the output’s cosine similarity with distance is:

$$
\begin{array} { l } { \displaystyle { c _ { \mathrm { { S W A } } } ( d ) \approx \big [ 1 - \frac { d } { w } \big ] _ { + } , } } \\ { \displaystyle { c _ { \mathrm { { g l o b a l } } } ( d ) \approx \sqrt { ( i - d + 1 ) / ( i + 1 ) } \propto 1 - O ( d / i ) . } } \end{array}
$$

Hence, at initialization, the strength of SWA’s recency bias decays with the window size w, with a smaller window resulting in a stronger, steeper recency bias; conversely, global attention’s positional signal decays with the sequence length $i ,$ and therefore dilutes for longer sequences. That is, SWA’s locality scale is length-independent by construction, whereas global attention’s is not.

General mean-attention case. We now describe how SWA affects the similarity structure in its output after training, where the attention profile is no longer uniform and the incoming residual stream no longer zero-mean Gaussian. Again, we approximate attention by its mean profile $a _ { d } = \mathbb { E } [ \alpha _ { i } ( d ) ]$ (estimated across some number of relevant sequences). Suppose that X has some incoming lagged cross-moment structure $G _ { X } ( d )$ let $A = W _ { O } W _ { V }$ be the composed projected-value matrix and $\begin{array} { r } { g _ { a } ( k ) = \sum _ { s } a _ { s } a _ { s + k } } \end{array}$ the mean attention autocorrelation. In this setting, the SWA output is $\begin{array} { r } { y _ { i } \approx A \sum _ { d = 0 } ^ { w - 1 } a _ { d } x _ { i - d } , } \end{array}$ , and its lagged cross-moment is

$$
G _ { Y } ( d ) \approx A \sum _ { k } g _ { a } ( k ) G _ { X } ( d - k ) A ^ { \top } = A ( g _ { a } * G _ { X } ) ( d ) A ^ { \top } ,
$$

if output norms are concentrated (see Appendix B-A for the full derivation). Thus, the similarity structure in the output is the incoming structure mixed across distances by the mean attention autocorrelation, and then mixed across channels by the composed attention matrices. The structural effect of SWA is still present, as $g _ { a } ( k ) > 0 \mathrm { ~ i f ~ } | k | < w$ , and 0 otherwise. $\mathrm { S o } ,$ the window bounds the scale of the recency bias, but its effect can be amplified or dampened by the attention profile within the window. Finally, the extent to which this mixing promotes or dilutes a recency bias in the output with respect to each similarity metric depends on its alignment to the relevant projections.

## B. Empirics

Initialization and length scaling. Empirically, we first validate the argument that SWA’s effect is structurally different from that of global NoPE, and that only the former produces a well-resolved recency bias at long sequence lengths. Figure 2 compares the isolated first attention-branch outputs before residual addition at initialization. With increasing sequence lengths (probed via query position along an 8k-token sequence), global NoPE activations become so highly correlated that positions cannot be resolved for most distances; in contrast, SWA retains a strong distance-dependent profile across the window while flattening beyond, regardless of sequence length.

![](images/20467f02e2dc9937da26a0895189ec1df0de5c7f98b4c3a5fafaa49897446eda.jpg)  
Figure 2: Attention outputs at initialization. Raw cosine of the isolated first attention-branch outputs for global NoPE and SWA (w = 128; marked by dashed line), across four fixed query-position bins. The recency bias in the SWA output is stronger and remains constant with increased sequence length, whereas the recency bias of full attention weakens significantly at longer sequences.

Emergence with training and depth. Though the analysis above corroborates the hypothesis that SWA produces a recency bias at initialization, it is possible that this effect dissipates during training as the model focuses on learning contentdependent interactions. However, this is not the case – in fact, the recency bias strengthens during training. Figure 3 visualizes the recency profile during training at the first, middle, and last SWA layers. Across depth, the trained profiles show strengthening and subsequent attenuation relative to their peak, while remaining stronger than at initialization. Note that this effect stabilized early in training (the colorbar is logarithmic in the fraction of that scale’s total training tokens). Furthermore, the recency bias strengthens significantly with increased depth. Further analysis as well as layer-by-layer profiles are shown in Appendix B-B.

![](images/293595903d31f4a50222ee3b59406b9d54ffb9a440936f47c1bff00dff3ee049.jpg)

Window-size dependence. The theoretical analysis in Section IV-A suggests that smaller SWA windows will produce a narrower and steeper recency bias in the outgoing residual stream. The distance profiles (top) and layer-wise recency gaps (bottom) in Figure 4a verify this trend; smaller windows lead to stronger recency biases across depth (for full profile comparisons, see Figure 9b in Appendix B). Further, to mitigate the confound of the strong logit-based recency bias induced by RoPE within the window (which as we discussed in Section IV-A has an important contribution to the recency bias), we also verify this for models with NoPE-in-SWA. Figure 4b shows the distance profiles and layer-wise recency gaps across window sizes for NoPE-in-SWA; in fact, the effect of window size is more pronounced, with smaller windows producing significantly stronger recency biases (for further NoPE-in-SWA results, please see Figure 12f in Appendix D).

Given the pronounced effect of SWA window size on recency bias strength, a further question one might ask is how window size affects model performance. We find that smaller windows achieve lower loss than their larger-windowed counterparts (see Figure 5 here and Figure 12g–h in the appendix, which report the late-training validation-loss trajectories; note that larger NoPE-in-SWA windows cause loss collapse). Although preliminary, this is promising evidence that the recency bias strength is important to such hybrid models’ success. In fact, the benefit of strengthening the recency bias effect by decreasing

![](images/33abc1f0dfdf2e2c6b6edda7f241235b40bc2121db9a9653b6bbc1a37c806a10.jpg)

![](images/c1dd3e106bd5f5faa6da035794e324d00ef3a317a24b52f423df29ede3bbc28b.jpg)  
Figure 3: Residual cosine during training. Post-attention cosine profiles at the first, middle, and last SWA layers (w = 128), with color denoting the fraction of training completed on a logarithmic scale. This demonstrates that residual stream recency bias increases with training and depth.

![](images/8d2f17c57717b19abf3ef3611ece150aab056780d45ea271eadb2f1b2c908029.jpg)  
Figure 4: Window dependence of residual cosine. (a) RoPE-in-SWA and (b) NoPE-in-SWA, with the top panels showing the cosine gap $c ( d ) - c ( 4 0 9 6 )$ at the middle SWA layer (L13), and the bottom panels showing the cosine gap $c ( 1 ) - c ( 4 0 9 6 )$ across depth (crosses mark SWA layers and circles global NoPE layers). With or without RoPE, smaller windows result in stronger recency biases across depth.

![](images/f18f7f26f5218aef11aedba93af87e1b950b0585c96eb44549627f58ef05efb9.jpg)  
Figure 5: Validation loss across window sizes. (a) RoPE-in-SWA and (b) NoPE-in-SWA. Mean 350M DCLM token crossentropy (nats) over the final 30% of training. With or without RoPE, smaller windows result in lower loss.

window size may outweigh the cost of the associated reduction in training compute so that the performance of the model improves on shorter window sizes. Also note that, since these experiments did not match compute across window sizes, the advantage for small windows is even stronger than shown here.

## V. STEP 2: PRESERVATION BY INTERVENING TRANSFORMATIONS

## A. Theory

Addition of the residual stream. Suppose there is some similarity structure $G _ { Y } ( d )$ in the SWA output $Y _ { \pm }$ , and $G _ { X } ( d )$ in its input X. Denote the following residual stream by $Z = X + Y ;$ we wish to understand its similarity structure $G _ { Z } ( d )$ as a function of its components. Expanding the outerproduct in $G _ { Z } ( d ) = \mathbb { E } [ z _ { i } z _ { i - d } ^ { \top } ]$ , we obtain

$$
G _ { Z } ( d ) = G _ { X } ( d ) + \mathbb { E } [ x _ { i } y _ { i - d } ^ { \top } ] + \mathbb { E } [ y _ { i } x _ { i - d } ^ { \top } ] + G _ { Y } ( d )
$$

For uncorrelated white inputs, the first and second terms are $0 ,$ and the third term becomes $a _ { d } A \Sigma$ , and so we obtain $G _ { Z } ( d ) =$ $a _ { d } A \Sigma + g _ { a } ( d ) A \Sigma A ^ { \top }$ . In either case, the similarity structure $G _ { Y } ( d )$ is carried into the residual stream, and the resulting similarity structure depends on its interactions with the existing similarity structure $G _ { X } ( d )$

Effect of normalization. The actual normalization operation, whether this is the (numerically stable, scaled by $\sqrt { D } )$ L2 normalization for RMSNorm or first subtracting the channel mean for LayerNorm, has little effect on similarity structure. RMSNorm in fact exactly preserves the unit-normalized crossmoment structure; and, though LayerNorm additionally projects prior to L2 normalization, prior work has found that they behave very similarly in practice (Zhang & Sennrich, 2019; Gupta et al., 2026).<sup>1</sup>

In contrast, the gain (and offset for LayerNorm; though we omit that here for simplicity) of the normalization module can alter the similarity structure, including its floor and gap. Here, let X denote the output of the normalization stage. Decompose $X$ into its variation around a position-independent mean ${ \bar { x } } =$ $\mathbb { E } [ x _ { i } ]$ , so that each $x _ { i } = \bar { x } + \epsilon _ { i }$ where $\mathbb { E } [ \epsilon _ { i } ] = 0$ . For per-channel gains $D _ { \gamma } = \mathrm { d i a g } ( \gamma )$ and output $Y = D _ { \gamma } X$ , we have

$$
\mathbb { E } \big [ y _ { i } y _ { i - d } ^ { \top } \big ] = \underbrace { D _ { \gamma } \bar { x } \bar { x } ^ { \top } D _ { \gamma } } _ { \mathrm { s h a r e d - m e a n ~ c o n t r i b u t i o n } } + \underbrace { D _ { \gamma } \mathbb { E } \big [ \epsilon _ { i } \epsilon _ { i - d } ^ { \top } \big ] D _ { \gamma } } _ { \mathrm { f u c t u a t i o n ~ c r o s s - m o m e n t } } .
$$

Write $M \ = \ D _ { \gamma } \bar { x } \bar { x } ^ { \top } D _ { \gamma }$ for the shared-mean contribution and $e _ { M } = \mathrm { t r } ( M ) = \| D _ { \gamma } \bar { x } \| ^ { 2 }$ for its energy, and $C ( d ) \ =$ $D _ { \gamma } \mathbb { E } [ \epsilon _ { i } \epsilon _ { i - d } ^ { \top } ] D _ { \gamma }$ for the fluctuation cross-moment and $e _ { C } =$ $\mathrm { t r } ( \bar { C ( 0 ) ) } = \mathbb { E } [ \| D _ { \gamma } \epsilon _ { i } \| ^ { 2 } ]$ for its energy. Assuming approximately stationary second moments and concentrated norms, the unit-normalized similarity gap and far-distance floor are

$$
G _ { Y } ( d ) - G _ { Y } ( L ) \approx { \frac { C ( d ) - C ( L ) } { e _ { M } + e _ { C } } } , \qquad G _ { Y } ( L ) \approx { \frac { M + C ( L ) } { e _ { M } + e _ { C } } } .
$$

How the normalization module’s gain affects the similarity gap and floor (and in turn, the cosine similarity, or another projection or magnitude of the cross-moment) depends on its effect on the shared-mean contribution and fluctuation crossmoments and their energies. The gain’s effect on each of these terms depends in turn on how the channel-pair entries are reweighted by $\gamma _ { c } \gamma _ { k }$ , with the energies depending on the diagonal weights $\gamma _ { c } ^ { 2 } .$ . In practice, the broad pattern appears to be that the normalization’s gain reduces the shared-mean energy relative to the fluctuation energy; when the fluctuation profile $C ( d ) / e _ { C }$ is approximately preserved, this can reduce the cosine similarity floor while amplifying the similarity gap.

Effect of the MLP. For normalized inputs, a wide MLP at random initialization admits a simple kernel description of its effect on cosine similarity (Daniely et al., 2016); i.e., at initialization an MLP preserves the ordering of the recency-bias profile.

Suppose the MLP branch is $y = W _ { 2 } f ( W _ { 1 } x )$ , with wide random Gaussian $W _ { 1 } , W _ { 2 }$ , and consider two inputs $x , x ^ { \prime }$ with cosine similarity $p .$ For each hidden unit, the corresponding preactivations $U = w ^ { \top } x$ and $V = w ^ { \top } x ^ { \prime }$ are jointly Gaussian with correlation p. We therefore reduce the problem to asking how the nonlinearity $f$ transforms this correlation, defining the resulting kernel $k _ { f } ( p ) ;$ the subsequent projection $W _ { 2 }$ approximately preserves this kernel at large width. Expanding $f$ in the Gaussian Hermite basis gives

$$
k _ { f } ( p ) = { \frac { \sum _ { n } a _ { n } ^ { 2 } p ^ { n } } { \sum _ { n } a _ { n } ^ { 2 } } } .
$$

The $n = 0$ term contributes a constant similarity floor, the $n = 1$ term transmits similarity linearly, and higher-order terms nonlinearly reshape the profile. Since the coefficients enter as $a _ { n } ^ { 2 } , \ k _ { f }$ is nondecreasing for $p \geq 0$ . Thus, at random initialization, a wide MLP preserves the ordering of a nonnegative recency-bias profile, although it may change its floor, magnitude, and shape.

This initialization argument does not guarantee preservation by trained MLPs. Depending on the projections and nonlinearities learned in training, the MLP can suppress, amplify, or reorient the incoming similarity structure. We therefore assess these effects empirically in Section V-B, for the similarity gap and floor each.

## B. Empirics

We observed in Section V-A that residual stream addition, normalization, and MLPs each have a complex effect on the full cross-moment similarity structure, including both its gap and floor. To empirically characterize these effects, Figure 6 measures the difference in the cosine gap and floor, before and after each module (averaged across each of its appearances in depth; the grey block summaries show the cumulative effect across each block). Proceeding through each module appearance from left to right, we generally observe the following trends: (i) normalization, including its learned affine transformation, reduces the cosine floor and increases the gap; (ii) SWA provides the strongest boost to the gap (and interestingly, reduces the floor); (iii) the residual addition reduces the gap and increases the floor; (iv) the MLP reduces the gap and increases the floor; and (v) global NoPE attention also increases the gap, though much less so than SWA. The cumulative effect across each block is much smaller, with full-attention and MLP blocks largely preserving the incoming gap.

![](images/d6fb0623460f2052ee03e0ffea390280802afb99951bc3866f47de5f59a8e225.jpg)  
Figure 6: Module changes in cosine gap and floor. After-minus-before changes in cosine gap $c ( 1 ) - c ( 4 0 9 6 )$ (top) and cosine floor c(4096) (bottom), averaged across SWA and full-attention blocks at $w = 1 2 8$ . Residual-addition steps compare with the isolated branch; shaded block summaries compare the complete update with its incoming residual. Note that the effect of each module on the similarity structure is largely as predicted by our theory.

![](images/cc75d681b58cdef118a9811389868d8b3955a0a6a116ef18d404cc3caf5b9ad5.jpg)  
Figure 7: Global NoPE logits during training. Head-mean profiles at the first, middle, and last global layers $( w = 1 2 8 )$ , with color denoting the fraction of training completed on a logarithmic scale. This shows that, through training, the global NoPE heads learn to select the residual stream recency bias and thereby develop a corresponding recency bias in their logits.

## VI. STEP 3: GLOBAL NOPE INHERITS THE RESIDUAL STREAM RECENCY BIAS

## A. Theory

Suppose the recency bias has been preserved until the input to global NoPE attention; the final step is its successful readout into the global NoPE logits. Let $x _ { i }$ denote the postnormalization input to global NoPE’s query and key projections. For a fixed attention head, let $M = W _ { Q } ^ { \top } W _ { K } / \sqrt { D _ { H } }$ and recall $G _ { X } ( d ) = \mathbb { E } [ x _ { i } x _ { i - d } ^ { \top } ]$ . The expected logit profile is

$$
\ell ( d ) = \mathbb { E } [ x _ { i } ^ { \top } M x _ { i - d } ] = \langle G _ { X } ( d ) , M \rangle _ { F } .
$$

Thus, the logit recency gap $\ell ( 1 ) - \ell ( L )$ is positive precisely when M aligns positively with $G _ { X } ( 1 ) - G _ { X } ( L )$ ; its magnitude depends on the extent of that alignment. Notably, in contrast to cosine, this similarity profile can depend on the skew component of the full cross-moment structure. Depending on the slice orientation, the projections can preserve, distort, or reverse the residual-similarity ordering (see Appendix C for further discussion).

Note also that, for independent zero-mean query and key initializations, $\mathbb { E } [ M ] = 0 ;$ ; therefore, the recency bias does not immediately propagate to the full attention logits in expectation over initialization. However, training can align $W _ { Q }$ and $W _ { K }$ allowing their logits to inherit the residual stream recency bias, which predicts that the effect should become visible in NoPE logits as training induces favorable alignment.

## B. Empirics

Emergence with training. To empirically verify this, Figure 7 tracks the global NoPE attention logits during training. Indeed, at initialization the distance dependence is weak; but, as training progresses, a recency-biased profile emerges across depth, consistent with learning the projection alignment described above. Similarly to the development of the residual stream recency bias following SWA, the recency bias in the full attention logits strengthens and stabilizes very early in training (the colorbar is logarithmic in the fraction of training completed). But, whereas the residual stream recency bias increases consistently with depth, the logit recency bias quickly strengthens then stabilizes (layer-by-layer profiles are shown in Appendix C-B).

Window-size dependence. We found in Section IV that smaller windows induce a stronger residual stream recency bias. To investigate the extent to which this trend propagates into the global NoPE logits, Figure 8 summarizes the layer and head distribution of the recency gap. In comparison to the residual stream, a few distinct patterns emerge. For both RoPE-in-SWA and NoPE-in-SWA, the logit gap remains fairly stable across depth after initially strengthening (as seen above). For RoPE-in-SWA (Figure 8a), there is also little variation by window size; however, for NoPE-in-SWA (Figure 8b), the logit gaps for larger windows swing drastically positive and negative (note that the recency gap axis is symmetric-log scaled to account for the wider variation). Further analysis as well as layer-by-layer profiles are shown in Appendix C-B.

These patterns suggest that the model seeks to learn a consistent logit recency bias across depth. However, if window size is too large (and therefore, the residual stream recency bias too weak) then the model is unable to develop a logit recency bias, resulting in performance collapse. This aligns with the spirit of our proposed mechanism: the role of local layers is to produce a residual stream recency bias available to be picked up by the global NoPE logits; within the global NoPE logits, the model learns to pick up and maintain a consistent recency bias from the residual stream across heads to serve as its position encoding in attention.

![](images/d525c2c38faf2c42736ecdaa37646909d808d7481e897c21dcef814dea9c362f.jpg)  
Figure 8: Global NoPE logit gaps. RoPE-in-SWA (top) and NoPE-in-SWA (bottom), showing the head-mean gap $\bar { \ell } ( 1 ) -$ <sup>¯</sup>ℓ(4096) (shading spans the full headwise range and vertical axes are symmetric-log). Though the window dependence is less clear in RoPE-in-SWA, it appears the logit recency bias collapses for large windows with NoPE-in-SWA.

## VII. DISCUSSION

Despite their empirical success, how hybrid models interleav ing SWA and global NoPE attention encode position in their global NoPE layers has remained a mystery. In this paper, we develop a novel mechanistic explanation for how such models accomplish this encoding. First, SWA creates a recency bias in its output by correlating neighboring embeddings, which emerges as a fundamental and in-built inductive bias present from initialization. Second, this recency bias is preserved by the intervening transformations (viz. residual stream addition, normalization, and MLPs). Third, it is selected by the global attention projections (to varying degrees across heads) and propagated to the global attention logits. Furthermore, this proposed mechanism may be extended to hybrid models that interleave linear layers like KDA in place of SWA (Appendix E). This recency bias acts as a kind of implicit relative position encoding, creating effects similar to explicit relative position encodings.

Our account of how such models encode position is particularly interesting given the special capacity of hybrid models with global NoPE attention for length extension and generalization. We hypothesize that this occurs for a few reasons. First, the window provides a finite-length local scale that does not dilute as sequence length increases, and which is invariant to the absolute sequence length. Also, the position encoding learned does not radically change at distances beyond those seen in training, as is the case with RoPE, implying that extrapolation to longer sequence lengths does not introduce any serious discontinuities. Finally, this position encoding may be uniquely flexible due to residing within the residual stream geometry, which may enable expressive and easily-learned interactions with content that may be adaptively read out by the global attention heads.

Nevertheless, there are still many unknowns about how hybrid models with SWA and global NoPE attention encode position. For example, we still have not developed a complete quantitative account of how the recency bias propagates through all modules of the model, or how this affects the attenuation of the recency bias across depth and varying sequence lengths. Nor have we established a quantitative relationship between these characteristics and the extent of length generalizability. These topics are left to future work.

## REFERENCES

Federico Barbero, Alex Vitvitskyi, Christos Perivolaropoulos, Razvan Pascanu, and Petar Velickoviˇ c. Round and round´ we go! what makes rotary positional encodings useful? In The Thirteenth International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2410.06205.

Yuhan Chen, Ang Lv, Jian Luan, Bin Wang, and Wei Liu. HoPE: A novel positional encoding without long-term decay for enhanced context awareness and extrapolation. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 23044–23056, 2025. doi: 10.18653/v1/2025.acl-long.1123. URL https://aclanthology.org/2025.acl-long.1123/.

Alexis Chevalier, Jiayi Geng, Alexander Wettig, Howard Chen, Sebastian Mizera, Toni Annala, Max Aragon, Arturo Rodriguez Fanlo, Simon Frieder, Simon Machado, Akshara Prabhakar, Ellie Thieu, Jiachen T. Wang, Zirui Wang, Xindi Wu, Mengzhou Xia, Wenhan Xia, Jiatong Yu, Junjie Zhu, Zhiyong Ren, Sanjeev Arora, and Danqi Chen. Language models as science tutors. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 8310–8335, 2024. URL https://proceedings.mlr.press/v235/ chevalier24a.html.

Ta-Chung Chi, Ting-Han Fan, Li-Wei Chen, Alexander Rudnicky, and Peter Ramadge. Latent positional information is in the self-attention variance of transformer language models without positional embeddings. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 1183– 1193, 2023. doi: 10.18653/v1/2023.acl-short.102. URL https://aclanthology.org/2023.acl-short.102/.

Amit Daniely, Roy Frostig, and Yoram Singer. Toward deeper understanding of neural networks: The power of initialization and a dual view on expres-

sivity. In Advances in Neural Information Processing Systems, volume 29. Curran Associates, Inc., 2016. URL https://proceedings.neurips.cc/paper\_files/paper/2016/ file/abea47ba24142ed16b7d8fbf2c740e0d-Paper.pdf.

Philipp Dufter, Martin Schmitt, and Hinrich Sch "utze. Position information in transformers: An overview. Computational Linguistics, 48(3):733–763, 2022. doi: 10. 1162/coli\_a\_00445.

Tianyu Gao, Alexander Wettig, Howard Yen, and Danqi Chen. How to train long-context language models (effectively). arXiv preprint arXiv:2410.02660, 2025. URL https://arxiv. org/abs/2410.02660.

Gemma Team. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024. URL https://arxiv.org/abs/2408.00118.

Akshat Gupta, Atahan Ozdemir, Caoqinwei Gong, and Gopala Anumanchipalli. Geometric interpretation of layer normalization and a comparative analysis with rmsnorm. In Findings of the Association for Computational Linguistics: EACL 2026, pp. 375–407, 2026.

Adi Haviv, Ori Ram, Ofir Press, Peter Izsak, and Omer Levy. Transformer language models without positional encodings still learn positional information. In Findings of the Association for Computational Linguistics: EMNLP 2022, pp. 1382–1390, 2022. doi: 10.18653/v1/2022.findings-emnlp.99. URL https://aclanthology.org/2022.findings-emnlp.99/.

Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, and Siva Reddy. The impact of positional encoding on length generalization in transformers. In Advances in Neural Information Processing Systems, volume 36, pp. 24892–24928, 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash 4e85362c02172c0c6567ce593122d31c-Abstract-Conference. html.

Junu Kim, Xiao Liu, Zhenghao Lin, Lei Ji, Yeyun Gong, and Edward Choi. LayerNorm induces recency bias in transformer decoders. In Findings of the Association for Computational Linguistics: ACL 2026, 2026. URL https://aclanthology.org/2026.findings-acl.1430/.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, et al. Kimi K3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026. URL https://arxiv.org/abs/2607. 24653.

Shanda Li, Chong You, Guru Guruganesh, Joshua Ainslie, Santiago Ontanon, Manzil Zaheer, Sumit Sanghai, Yiming Yang, Sanjiv Kumar, and Srinadh Bhojanapalli. Functional interpolation for relative positions improves long context transformers. In The Twelfth International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/ 2310.04418.

Zhixuan Lin, Evgenii Nikishin, Xu Owen He, and Aaron Courville. Forgetting transformer: Softmax attention with a forget gate. In The Thirteenth International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/

2503.02130.

Ofir Press, Noah A. Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. In International Conference on Learning Representations, 2022. URL https://arxiv.org/abs/2108.12409.

Krishna C. Puvvada, Faisal Ladhak, Santiago Akle Serrano, Cheng-Ping Hsieh, Shantanu Acharya, Somshubra Majumdar, Fei Jia, Samuel Kriman, Simeng Sun, Dima Rekesh, and Boris Ginsburg. SWAN-GPT: An efficient and scalable approach for long-context language modeling. arXiv preprint arXiv:2504.08719, 2025. URL https://arxiv.org/abs/2504. 08719.

Qwen Team. Qwen3-Next-80B-A3B-Instruct model card. Hugging Face, 2025. URL https://huggingface.co/Qwen/ Qwen3-Next-80B-A3B-Instruct.

Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. Selfattention with relative position representations. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 464–468, 2018. doi: 10.18653/ v1/N18-2074. URL https://aclanthology.org/N18-2074/.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. doi: 10.1016/j.neucom.2023.127063. URL https:// arxiv.org/abs/2104.09864.

Shawn Tan, Songlin Yang, Aaron Courville, Rameswar Panda, and Yikang Shen. Scaling stick-breaking attention: An efficient implementation and in-depth study. arXiv preprint arXiv:2410.17980, 2025. URL https://arxiv.org/abs/2410. 17980.

Xinyi Wu, Yifei Wang, Stefanie Jegelka, and Ali Jadbabaie. On the emergence of position bias in transformers. In Pro ceedings of the 42nd International Conference on Machine Learning, 2025. URL https://arxiv.org/abs/2502.01951.

Songlin Yang, Yikang Shen, Kaiyue Wen, Shawn Tan, Mayank Mishra, Liliang Ren, Rameswar Panda, and Yoon Kim. PaTH attention: Position encoding via accumulating householder transformations. In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/085713-2080. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 59c27bf8d56d3d50c7aeaf7535dee975-Abstract-Conference. html.

Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in neural information processing systems, 32, 2019.

Yifan Zhang, Zixiang Chen, Yifeng Liu, Zhen Qin, Huizhuo Yuan, Kangping Xu, Yang Yuan, Quanquan Gu, and Andrew Chi-Chih Yao. Group representational position encoding. In The Fourteenth International Conference on Learning Representations, 2026. URL https://arxiv.org/abs/2512.07805.

Chunsheng Zuo, Pavel Guerzhoy, and Michael Guerzhoy. Position information emerges in causal transformers without

positional encodings via similarity of nearby embeddings. In Proceedings of the 31st International Conference on Computational Linguistics, pp. 9418–9430, 2025. URL https://aclanthology.org/2025.coling-main.632/.

## Appendix

Appendix A: Measurement and Experimental Details 12   
A-A Similarity measures and estimation 12   
A-B Models, training, and evaluation inputs 12   
A-C Aggregation and uncertainty conventions 12   
Appendix B: Supplement to Step 1: SWA Produces a Recency Bias in the Residual Stream 13   
B-A Theory . 13   
B-B Empirics . 13   
Appendix C: Supplement to Step 3: Global NoPE Inherits the Residual Stream Recency Bias 13   
C-A Theory . 14   
C-B Empirics . 14   
Appendix D: Controls 15   
D-A 120M-scale robustness 15   
D-B Random-token controls 15   
Appendix E: KDA extension 21   
E-A Theory . 21   
E-B Empirics . 21

## APPENDIX A MEASUREMENT AND EXPERIMENTAL DETAILS

## A. Similarity measures and estimation

We measure the cosine similarity after attention is added to the residual stream, except for isolated attention outputs at initialization and explicitly named module states. For the matrix plots, we average outer products of unit-length states before taking norms since this captures shared structure, though sampling noise can contribute. Our cosine and logit profiles use every integer lag from 1 to 4096.

## B. Models, training, and evaluation inputs

SWA configurations. We study the 120M and 350M hybrid models described in Section III-B, alternating SWA with global NoPE attention. SWA windows are 64, 128, 256, 512, 1024, 2048, 4096; single-model comparisons use w = 128. The SWA layers use RoPE unless stated otherwise. The two model scales were trained for 12B and 24B tokens, respectively, with a sequence length of 8192. Training and natural-language evaluation text use the GPT-2 tokenizer. The model embedding vocabulary is padded to 50,304 IDs. The figures in the main text use the 350M models and TextbookChapters unless their captions state otherwise.

KDA configurations. The KDA controls interleave KDA and global NoPE layers at both model scales. The supplementary KDA figures use final-checkpoint measurements on both TextbookChapters and random-token inputs; their aggregation convention are stated below.

Evaluation support. Recency measurements use 48 sequences of length 8192. The natural-language inputs come from TextbookChapters; random-token controls sample IDs uniformly from the padded model vocabulary with a seed of 0. Queries are fixed at zero-based positions 4352–8191 for every lag $1 \leq d \leq 4 0 9 6$ , giving 184320 query–key pairs per lag. Initialization plots partition this support into four bins of 960 query positions. Validation-loss plots instead use DCLM sequences of length 4096 and show raw trajectories over the final 30% of training.

## C. Aggregation and uncertainty conventions

Global NoPE logits. In the SWA and KDA profile and gap figures, we average the actual pre-softmax logits across heads at each lag. The logit gap is <sup>¯</sup>ℓ(1) − <sup>¯</sup>ℓ(4096) from this head-mean profile. Where shown, shading spans the range of headwise gaps, not uncertainty.

Module changes and matrix measurements. Figure 6 uses the final 350M RoPE-in-SWA, w = 128 checkpoint. Each point is an after-minus-before change, averaged equally over the twelve SWA or twelve full-attention blocks. Residual addition steps compare with the isolated branch while block summaries compare the updated residual with its incoming residual. For Figure 10, matrix norms are taken after averaging matrices across queries and sequences separately at each state and layer. Query–key pairs are not treated as independent samples.

## APPENDIX B

## SUPPLEMENT TO STEP 1: SWA PRODUCES A RECENCY BIAS IN THE RESIDUAL STREAM

This appendix provides additional theoretical and empirical results that corroborate those in Section IV, using the same setting of 350M scale models using natural language inputs (TextbookChapters).

## A. Theory

Mean-attention cross-moment derivation. As in Section IV-A, we assume that the norms of vectors entering and leaving the attention branch are concentrated, and therefore we work directly on unit-normalized moments $G _ { X }$ and $G _ { Y }$ . We assume these moments are approximately stationary away from the causal boundary. For a fixed mean attention profile $a _ { d } .$ , that is zero outside, $0 \leq d < w$ , let $\textstyle y _ { i } = A \sum _ { d } a _ { d } x _ { i - d }$ . Expanding both outputs gives

$$
\begin{array} { l } { { \displaystyle G _ { Y } ( d ) = A \sum _ { r , s = 0 } ^ { w - 1 } a _ { r } a _ { s } \mathbb { E } [ x _ { i - r } x _ { i - d - s } ^ { \top } ] A ^ { \top } } } \\ { ~ } \\ { { = A \sum _ { r , s = 0 } ^ { w - 1 } a _ { r } a _ { s } G _ { X } ( d + s - r ) A ^ { \top } } } \\ { { } } \\ { { = A \displaystyle \sum _ { k = - ( w - 1 ) } ^ { w - 1 } g _ { a } ( k ) G _ { X } ( d - k ) A ^ { \top } , ~ } } \end{array}
$$

Here d is the output lag and $k = r - s$ is the difference between two filter offsets. For learned attention, taking $a _ { r } = \mathbb { E } [ \alpha _ { i } ( r ) ]$ is a fixed-profile approximation that omits weight fluctuations and their dependence on the inputs.

For uncorrelated white inputs, $G _ { X } ( d ) = \Sigma \mathbb { 1 } \{ d = 0 \}$ , and therefore $G _ { Y } ( d ) = g _ { a } ( d ) A \Sigma A ^ { \top }$ . If we further assume attention is uniform, then the triangular profile introduced in the text is a special case of the same convolution.

## B. Empirics

Depth-wise and window-size dependence. As noted in Section IV-B, the residual stream recency bias accumulates acros depth. To investigate this at finer resolution, Figure 9a shows the residual stream recency profile following every SWA layer at the last checkpoint of training, which demonstrates that the trend of increasing recency bias through depth is notably smooth across layers. Comparing Figure 9a to its 120M-scale companion Figure 12c in Appendix D-A, it is particularly notable how consistent this layer-wise recency bias trend is across model scales, not only in ordering, but also in magnitude. This supports that the residual stream recency bias provides a functional role regardless of scale.

To supplement the comparison of window sizes in Figure 4, Figure 9b displays the full cosine profiles at the first, middle, and last SWA layers of different window sizes; Figure 5 in the main text shows the validation loss curves for the same range of window sizes. Together, these results show that varying window size has a clear and predictable effect on both cosine profiles and language modeling (though we cannot fully isolate its causal role).

(a) Depth  
![](images/70ed23de56d18035de3cb5b252de68c4a1f5e77c5834fb869865c312520a5d28.jpg)

![](images/3f805caf99bb4200f09be50a5c788d791871f8eec58233bdc93763b487dfb92b.jpg)

(b) Window size  
![](images/025f4c8aa7dd140bca11cccb7170aa5423fec7ac2ed7c00f9d380f35c81bea9f.jpg)

![](images/b40983cb36eec3b3b340c98e73edf25f64b55eae544cd97d76d777d3807f6a1f.jpg)  
Figure 9: Residual cosine at 350M. Post-attention gap $c ( d ) - c ( 4 0 9 6 )$ : (a) every SWA layer at the final $w = 1 2 8$ checkpoint; (b) first, middle, and last SWA layers across windows 64–4096.

## APPENDIX C

## SUPPLEMENT TO STEP 3: GLOBAL NOPE INHERITS THE RESIDUAL STREAM RECENCY BIAS

This appendix examines how the recency structure induced by SWA and shaped by the intervening transformations analyzed in Section V is read out by global NoPE. It provides a cross-moment analysis and additional measurements supporting the readout in Section VI. We also explain how cross-moment magnitude and alignment help us understand the recency bias in both the residual stream and logits.

## A. Theory

Decomposing the general similarity structure. To analyze this readout, we study the full lagged cross-moment. Central to our understanding of this general structure is that different slices of it reduce to the cosine similarity and per-head logit curves, via the Frobenius inner product. Just as a vector inner product depends on both magnitude and alignment, it is instructive to decompose the Frobenius inner product into the contributions of cross-moment magnitude and its vectorized alignment with either the isotropic component I or the composed query-key projection $M = W _ { Q } ^ { \top } W _ { K } / \sqrt { D _ { H } }$

For nonzero matrices G and X of the same shape, the Frobenius inner product decomposes as

$$
\langle G , X \rangle _ { F } = \| G \| _ { F } \| X \| _ { F } \cos ( G , X ) , \qquad \cos ( G , X ) = { \frac { \operatorname { t r } ( G ^ { \top } X ) } { \| G \| _ { F } \| X \| _ { F } } } .
$$

Hence, the matrix cosine is just the ordinary cosine between the vectorized matrices: the two norms set the scale, while their alignment determines the sign.

To see how this pertains to a recency bias, let $B _ { X } = G _ { X } ( 1 ) - G _ { X } ( L )$ , and $B _ { Z } = G _ { Z } ( 1 ) - G _ { Z } ( L )$ for the actual post-normalization input $Z$ to global attention. Taking $X = I$ or $X = M$ gives

$$
\begin{array} { r l } & { c ( 1 ) - c ( L ) = \sqrt { D } \| B _ { X } \| _ { F } \cos ( B _ { X } , I ) , } \\ & { \ell ( 1 ) - \ell ( L ) = \| B _ { Z } \| _ { F } \| M \| _ { F } \cos ( B _ { Z } , M ) . } \end{array}
$$

Here $\| I \| _ { F } = { \sqrt { D } } ,$ , and $\| M \| _ { F }$ is the head’s readout gain. Therefore, the recency gaps for both cosine similarity and the global NoPE logits depend on the underlying matrix lag contrast of the cross-moment, which is unsigned (i.e., a large magnitude alone does not imply a positive cosine or logit gap). Meanwhile, its vectorized cosine alignment determines the sign. Next, we empirically assess its magnitude and alignment.

## B. Empirics

Cross-moment magnitude and readout alignment. For a broader view on the cross-moment magnitude and its alignment to both the isotropic component and per-head projections, Figure 10 measures the matrix lag contrast B of the cross-moment and its cosine similarity with I and M across depth. Indeed, as is expected from the separate analyses of residual stream cosine and the logits, the total similarity magnitude increases through depth along with its alignment to I (which together imply the increased recency bias through depth we see in Section IV); meanwhile, the global NoPE projections maintain a positive alignment with B that remains relatively constant, producing positive and consistent logit gaps across depth (as seen in Section VI-B).

![](images/d217327eb4aa469db2c255397df7b89323f9626955a31f0074a217a8e5ee3536.jpg)

![](images/ba61c55a3a78b75dfcb94c71e4c641770ed79f4e1cec9ead73c2b8d3982ae2fe.jpg)

![](images/9466b9c8064053f9370e00b70fafa77858bf8cb69e999c1d7939f0b02ef53aba.jpg)  
Figure 10: Cross-moment magnitude and readout alignment. Final 350M RoPE-in-SWA model $( w = 1 2 8 )$ . Left: $\| B _ { X } \| _ { F }$ at post-block residual states; crosses mark SWA and circles global NoPE layers. Middle: $\cos ( B _ { X } , I )$ at SWA layers. Right: $\cos ( B _ { Z } , M )$ at model-scale post-LN1 inputs to global heads; small points show eight heads and connected circles their unweighted mean. Magnitude uses a logarithmic axis; alignments use linear axes.

Depth-wise and window-size dependence. For a more fine-grained view of depth-wise patterns than provided in Figure 8, Figure 11a shows the mean logit profile at every global NoPE layer. As noted in Section VI-B, the pattern is different from that of the residual stream recency bias: after an initial rise, the logit recency bias is largely consistent across layers. Similarly, comparing the logit profiles for different window sizes, Figure 11b shows a weaker relationship between recency bias and window size than in the residual stream. The relative lack of variation in the logit profiles with depth and window size suggests that, as we discussed in Section VI-B, the model seeks to develop a consistent logit-based recency bias to serve as its position encoding in attention.

![](images/4d97a515fed215480585b1e013f43864a0534979f292b2c5dd64c74595bb0461.jpg)

![](images/84cdff4a9ee8e51273df5024a8a26ade554f6ff1ba0aadac36fe23a55f268ee7.jpg)  
Figure 11: Global NoPE logits at 350M. Head-mean gap <sup>¯</sup>ℓ(d) − <sup>¯</sup>ℓ(4096): (a) every global layer at the final w = 128 checkpoint; (b) first, middle, and last global layers across window sizes.

## APPENDIX D

## CONTROLS

## A. 120M-scale robustness

To verify that the 350M-scale results of the main text are robust across model scales, we train 120M models with the same configurations of SWA window size and varying RoPE and NoPE within the window, on the same dataset and for a number of tokens comparable given the scale difference (see Appendix A). These 120M-scale results corroborate the main-text 350M-scale results, with the same qualitative conclusions from each panel provided in the composite of Figure 12. That is, SWA’s recency bias is present from initialization (a), strengthens with training and depth (b–c), becomes sharper with smaller windows (d–e), and propagates to and is selected by the logits (i–l).

Furthermore, Figure 12f–h displays the cosine gap for NoPE-in-SWA, as well as a comparison of the validation losses for both RoPE and NoPE within the window. Of particular note is that, in both cases, loss decreases with smaller windows despite fewer training flops, indicating that smaller windows may be beneficial, perhaps due to their ability to resolve a strong recency bias. In further support of this, with NoPE-in-SWA loss collapses entirely for larger windows; this is suggestive that, without a sufficiently strong architectural mechanism to resolve a recency bias and therefore encode position, such models fail to learn language effectively.

## B. Random-token controls

The main analyses use natural-language sequences, as these are the natural domain of the language models we train. However, language has an underlying correlation structure that may confound the evidence for our proposed architectural mechanism. Hence, repeating the complete SWA analysis on uniformly sampled tokens from the model’s vocabulary is an important control to isolate the structural contribution from natural-language statistics. The following controls use the same query positions and lags as the TextbookChapters analyses and report results for both RoPE and NoPE within SWA (Figures 13–16; each composite groups the observables for one SWA encoding and model scale).

These controls also corroborate the main-text findings. Across all four composites, we observe the recency bias present at initialization and strengthening with training and depth (a–c). SWA’s position encoding has a larger impact on the effect of window size (d–e): for NoPE-in-SWA at both model scales, we observe the same pattern of smaller windows promoting stronger recency biases; however, for RoPE-in-SWA at both scales, the trends were inconclusive, likely due to random token altering the learned RoPE decay profile. Finally, across all four settings, the recency bias transfers to the logits particularly for smaller windows, but this transfer is less consistent than in natural language (f–i). Overall, these results are consistent with our proposed architectural mechanism functioning, but being untuned for data outside its training distribution.

(a) Initialization  
![](images/01938a2377508bbcee3bf29aa58c868b05bb0b7a21aee67b3762ad810e04819e.jpg)  
(c) Residual depth

(b) Residual training  
![](images/eeac593b6074d77254ff2b146300cc1166be1c22574d3845db67257247c0c2e8.jpg)  
(d) Residual windows

![](images/624b029052e1cc85e8dbb40133e13c3e7d3674edc4d58f2774dc095673014920.jpg)  
(e) RoPE cosine gap

![](images/fa508a9b8a908e3a7aa32be8d37ef119a120c3613a143ff9d9a6b0ea33aea084.jpg)  
(f) NoPE cosine gap

![](images/cdd673daa592bb27728c46b886ff7034e2c36090954e8aed4b361df7b19d2da9.jpg)  
(g) RoPE validation loss

![](images/de5b45c1bae750b567a62c8060ec939fd2052b1757c5a040c4c383cab536a8de.jpg)  
(h) NoPE validation loss

![](images/d295e43220736fa90db84b5f4bf2c3f746cabb1f3dcb7265498140a4bc310e43.jpg)

![](images/b8611b2d0fbc60cc4d39fb7c79811ebf477723712588f1a7875dd5f953a4f563.jpg)

(i) Logit training  
![](images/190475593e35d87faf581a6dfab2abf6d342f7dc0e72afeaf615cc8f4a81a979.jpg)

(j) Logit depth  
![](images/0859b659b8cc68935493a4a1056641dacb1adc964f44b46e20f4710d7a64f65f.jpg)

(k) Logit windows  
![](images/81603fbc007f846afd59642905b8a3afcc6d365fb240bda25d03665d9447ebfb.jpg)

(l) Head-mean logit gap  
![](images/c6fbb56c3d0253b4739f9ddc2bd7e0fec7268f27f1518a8c6650bbf1d94ddd10.jpg)  
Figure 12: 120M TextbookChapters robustness and validation loss. (a) Attention-branch cosine at initialization. (b–f) Postattention residual cosine during training, across depth and SWA windows, including RoPE- and NoPE-in-SWA gaps. (g–h) Validation loss over the final 30% of training on 4096-token sequences. (i–l) Global-NoPE head-mean logit profiles and gaps during training, across depth and windows. Profile gaps are relative to lag 4096; window sizes are 64–4096. Window comparisons do not isolate the contribution of recency to the loss.

![](images/2d9d4062f73ff5a0a9fdf5a8fe0f612774d090746b4593d2e01c989cd639375b.jpg)

![](images/9ae7727205d0dd1216ca819ceccd8bc27fc292f9a763ecc7ab77f18a3ad400b7.jpg)  
(c) Residual depth

(b) Residual training  
![](images/e4beee91c2494b3c8c7daecc894fe102d79ffee53037229d6c479f192931e115.jpg)  
(d) Residual windows

![](images/823fdeed037089bd3b6dad45d5c57f2a4898d067059bad3509266fbfbdd46680.jpg)  
(e) Residual cosine gap

![](images/cf4aa42d8cebca179046f84d6e1afb4e4bac294e083f81b09b44905acb4f6aca.jpg)  
(f) Logit training

![](images/cb20b4d6d65fbcb727039dbd7a231e5b80b7812588c270edce1a880f99f84680.jpg)  
(g) Logit depth

![](images/65be8b40be9567de5b352d08faa353555418bcb0ea3e348795b1360d01119b7a.jpg)

![](images/2997b28b3f66039f0c3a53baa73d0558935f3ae26dfd214215bc2a1a60c1c06b.jpg)

(h) Logit windows  
![](images/360a2cc0e3625aa916708b85262a49991a60c8e3a528130033658af3ddc11cdb.jpg)

(i) Head-mean logit gap  
![](images/1d1f71b07acdc532a8c273489ac7b2794a3da7d008ababc6c7a9f9e9c79c61e6.jpg)  
Figure 13: Random-token control: 350M, RoPE within SWA. (a) Attention-branch cosine at initialization. (b–e) Post-attention residual cosine during training, across depth and windows, including the depthwise gap. (f–i) Global-NoPE head-mean logi profiles and gaps during training, across depth and windows. Profile gaps are relative to lag 4096; window sizes are 64–4096.

![](images/e5f605e78f029e13f9a989e348e3d51a21f3a630a8bad34b848f72c5f0913f8e.jpg)

![](images/b84005c82c3c47f5bc61643df540fe26d1c3497a0a63aca886ae0146296a1c61.jpg)  
(c) Residual depth

(b) Residual training  
![](images/fdf25717b12ea4c79110536a366374ca15b3fb12d9ea95c83f6db39573c91253.jpg)  
(d) Residual windows

![](images/25ab2bfa73f8e84943aa91cd7dcc57987585e949d61dc2beb1360b882acd4e07.jpg)  
(e) Residual cosine gap

![](images/1ca2f2da201b052002efcdba88d628684b42feec193e14d2e8f3b9a8af64545e.jpg)  
(f) Logit training

![](images/a4e7e05a0b4e6dea73f03a97a85f6b9ef34a0ad3e94d62b462d5542f5f3ab1c7.jpg)  
(g) Logit depth

![](images/e47718d657c7e2f961018cc57b314b557e335ede734017e1a4a19d380533d39b.jpg)

![](images/6034bf0cc0fab1539d85768602aa9d2237e33be68d01b89fc0d17f8f6010aa5b.jpg)

(h) Logit windows  
![](images/3405a4c4d9d725687b35812f133f51082946d8940771e8264b110d2d238dc45c.jpg)

(i) Head-mean logit gap  
![](images/901385ea1a234c55d9c84010eb5c2a3160dfd8ef6d9ecafcda1a3f503d8dbb27.jpg)  
Figure 14: Random-token control: 350M, NoPE within SWA. (a) Attention-branch cosine at initialization. (b–e) Post-attention residual cosine during training, across depth and windows, including the depthwise gap. (f–i) Global-NoPE head-mean logit profiles and gaps during training, across depth and windows. Profile gaps are relative to lag 4096; window sizes are 64–4096.

![](images/fa08cd676331a6544e1c1968aed4f9086c8d04e6a05c4d16e8971561cc7a6d3b.jpg)

![](images/2f46d65b82dab47c586854cf2cd4cf0181b1e6ae130535a2a00806c33489b30f.jpg)  
(c) Residual depth

(b) Residual training  
![](images/81548dc0087fec58f2f9c538fbde0cc972e6e48d9f0150723ade66148dc4a9b2.jpg)  
(d) Residual windows

![](images/baac94d87774959d16ccf6665e3816616910112c2b39677ca3618c5161244742.jpg)  
(e) Residual cosine gap

![](images/5cb9fd4fb2a827abb7d9fccf36eeb9cea098c2887255804f11e7cfa7643a8971.jpg)  
(f) Logit training

![](images/36da09f2d28605d1e4b1bff71512b8c71d4aece740efc27881fabbfdba058fee.jpg)  
(g) Logit depth

![](images/abeaf467170872537e0ec0ef07f7c85b82b168df3d745b40381b3ec9178aaca8.jpg)

![](images/505e41112b68a3a47d02dd9ddbd8b958b7ad8ef7618b9a7bfcf53c15ec2e2625.jpg)

(h) Logit windows  
![](images/c0a4be21e7d27a501e3aa0a0ee823b6bd23850749574a75500bc427ded415b05.jpg)

(i) Head-mean logit gap  
![](images/b7741ef36cf996071bf94cd1316df3a61ecc036146dbd8e8a2031b2fb2c423f0.jpg)  
Figure 15: Random-token control: 120M, RoPE within SWA. (a) Attention-branch cosine at initialization. (b–e) Post-attention residual cosine during training, across depth and windows, including the depthwise gap. (f–i) Global-NoPE head-mean logi profiles and gaps during training, across depth and windows. Profile gaps are relative to lag 4096; window sizes are 64–4096.

![](images/24669bcfe2a3cf8c5e966e92cb620218b4aa68272c7ac7e7ab49ae7675531003.jpg)

![](images/91925a238558adcda3c0e1361ddf03969efe7011225f267b5393cbea0ea921ac.jpg)  
(c) Residual depth

(b) Residual training  
![](images/f836fedf2f957a3936891956fef40e94d4ad199fd7faca5ba48ee204435950b9.jpg)  
(d) Residual windows

![](images/21260a119cf7c142e281c7e6b16578bdf424a47680c48b7492f5154caa601c38.jpg)  
(e) Residual cosine gap

![](images/0a7b75ae1cad155b811ecc336f24261b6861b6e343c2003c21b86c37ef7fe83c.jpg)  
(f) Logit training

![](images/d8c0b6368826a77ed49d6ca30a6b701127478413fff3f2acf062a4a4c6ed991d.jpg)  
(g) Logit depth

![](images/4ce9cd2ed28c81bcbb218c0b10818e81ac8bcbf8ad3d524c3876a5918fe7b2dc.jpg)

![](images/7f4a9f91cb05f3032ce994e54722859aae2ac6a7a7289a4cdca87e9a0ff778b8.jpg)

(h) Logit windows  
![](images/2a4b37b8ecc153e42d254736cda440bd0ab40e05411d5e7e629118f6fbc98ee8.jpg)

(i) Head-mean logit gap  
![](images/09c5a5d0e10a59099e57b4c24fca2037e84e0f6c83b91f600734c2a17db119c9.jpg)  
Figure 16: Random-token control: 120M, NoPE within SWA. (a) Attention-branch cosine at initialization. (b–e) Post-attention residual cosine during training, across depth and windows, including the depthwise gap. (f–i) Global-NoPE head-mean logi profiles and gaps during training, across depth and windows. Profile gaps are relative to lag 4096; window sizes are 64–4096.

# APPENDIX E KDA EXTENSION

## A. Theory

KDA updates its recurrent state as

$$
\begin{array} { r } { S _ { t } = ( I - \beta _ { t } k _ { t } k _ { t } ^ { \top } ) \operatorname { D i a g } ( \alpha _ { t } ) S _ { t - 1 } + \beta _ { t } k _ { t } v _ { t } ^ { \top } . } \end{array}
$$

Here, $\alpha _ { t }$ controls how much of each state channel is retained, while the delta update revises the association with the current key. A write from position i reaches position t through the intervening transition products. We now use a simplified approximation to show how this repeated mixing can produce a recency bias, following the same intuition as for SWA.

Intuition. Consider a simplified scenario in which the projected values $o _ { t }$ are zero-mean, of similar magnitude, and uncorrelated across positions, and the effective contribution of a past value decreases by a fixed factor $0 < \lambda < 1$ at each step. Averaging over the content-dependent gates, writes, and query readout, we approximate the attention output before normalization and gating as an exponential moving average,

$$
y _ { t } \approx \lambda y _ { t - 1 } + ( 1 - \lambda ) o _ { t } = ( 1 - \lambda ) \sum _ { r = 0 } ^ { \infty } \lambda ^ { r } o _ { t - r } ,
$$

where the infinite sum describes positions sufficiently far from the start of the sequence. As in SWA, nearby outputs are similar because they combine many of the same values; consider, for example, $y _ { t }$ and $y _ { t - d } .$ In particular, all values contributing to $y _ { t - d }$ also contribute to $y _ { t } ,$ , but with their weights reduced by $\lambda ^ { d } .$ . Hence, if output norms are concentrated, their expected cosine similarity is approximately

$$
c _ { \mathrm { K D A } } ( d ) \approx \lambda ^ { d } .
$$

Thus, the same shared-values argument that gives a triangular profile for uniform SWA gives an exponential profile in thi simplified KDA case.

Length scaling. The retention factor plays a role analogous to the SWA window size: a smaller λ produces a narrower, steeper recency profile, while a larger λ retains correlations over longer distances. Critically, like SWA, this scale is length-independent and therefore remains equally well resolved with sequence length.

Correlated inputs. As in the general SWA case, the incoming similarity structure also affects the output. Let $G _ { O } ( d )$ and $G _ { Y } ( d )$ denote the lagged cross-moments of the unit-normalized projected values and outputs, respectively. Assume norms are concentrated, then for the same exponential weight $a _ { r } = ( 1 - \lambda ) \lambda ^ { r }$ , the filter calculation then gives the approximate relation

$$
G _ { Y } ( d ) \propto \sum _ { k \in \mathbb { Z } } g _ { a } ( k ) G _ { O } ( d - k ) , \qquad g _ { a } ( k ) = { \frac { 1 - \lambda } { 1 + \lambda } } \lambda ^ { | k | } .
$$

Thus, the incoming structure is mixed across distances by a smoothly decaying overlap profile. As in SWA, the subsequent transformations determine how this structure enters the residual stream and whether later NoPE projections read it out as larger logits for recent keys.

## B. Empirics

We empirically test whether our proposed mechanism extends to KDA-interleaved models at different scales, when using both natural language data and random-token controls. Figures 17 and 18 below show the KDA results; each row compares both model scales, and all logit panels use the head-mean convention specified in Appendix A-C. Indeed, the same empirical pattern appears: KDA layers primarily increase residual stream recency bias across depth (a–d); and later NoPE layers learn recency-biased logits (e–h). This indicates that the general mechanism we identify is not specific to SWA, but can apply to many local sequence mixers including the variety of increasingly common linear gated attention types.

(a) Residual profiles | 350M  
![](images/0dd57f73fc9bd0f32c9c78e50aebdfd4eb84cd7d91b91cfcf1cd140252ad4923.jpg)  
(c) Residual cosine gap | 350M

(b) Residual profiles | 120M  
![](images/b353f8d1a4a5d596ab054578d3e405aaec4709b727169f0a22e371e496908f70.jpg)  
(d) Residual cosine gap | 120M

![](images/e28794d9d5a67e3d708ddfd542cb92ad20fd521b808683e755ddbb8c236e8f46.jpg)

![](images/61b60cf5c2ca94412bd07e56aaaab753b4850537c665dd14de618afbbad2ae29.jpg)

(e) Global NoPE logits | 350M  
![](images/12495354a433809908002e8ac3d73db5d37ef7dbd92195400e5188e4303952bb.jpg)

(f) Global NoPE logits | 120M  
![](images/e4fadf4cce0a3b3fbc0b0e83ab03cd73463b0f02bf81bc0d2d83a1f3c61b81ff.jpg)

(g) Head-mean logit gap | 350M  
![](images/16470ab56d13095b00974417d2111be71e4c1121e47cd102f41454f6948a1ff1.jpg)

(h) Head-mean logit gap | 120M  
![](images/0ca1e4a072715c7a4b321d3b69002b6d9c9ab8e84958e62e63a77dbb87125d19.jpg)  
Figure 17: KDA–NoPE natural-language extension. Columns compare 350M (left) and 120M (right). (a–b) Residual cosine gap profiles across depth; (c–d) lag-1 residual gaps; (e–f) global-NoPE head-mean logit gap profiles; (g–h) lag-1 head-mean logit gaps. All gaps are relative to lag 4096.

(a) Residual profiles | 350M  
![](images/a98c911244aaf96d0020372b5fcbffeea937f32ccd1edfd60d6f8b7f51a0f030.jpg)

(b) Residual profiles | 120M  
![](images/6ba7b37775c00d825767358956fe90deedf9812d035aefd2c6aebde76f5abb0c.jpg)  
(d) Residual cosine gap | 120M

(c) Residual cosine gap | 350M  
![](images/e51016338de747b8632ef74ab97bae70ae72ac4e03d0b34275cd549cf9e3585f.jpg)

![](images/803c8daf41d04607879d56ee043402c4f3aa007acaa5e2eb3ee44705158f1d14.jpg)

(e) Global NoPE logits | 350M  
![](images/aaffc79954bb72641004d7c75191b86d708ac671e902b498fe3eb95c6a8ea3a4.jpg)

(f) Global NoPE logits | 120M  
![](images/991c55a6e3304def471f7ac88e32052cec554abd54cb14018a3b4b3ac88fa00b.jpg)

(g) Head-mean logit gap | 350M  
![](images/095a225610d469077cac218d2a0219a7ddc48b9f6c6d5e28afb55b2e8de9bfab.jpg)

(h) Head-mean logit gap | 120M  
![](images/3eb8ef14dc3396c64c9017314e9072554df6cdcb3f7bcd70adae14f8b0d9bfea.jpg)  
Figure 18: KDA–NoPE random-token extension. Columns compare 350M (left) and 120M (right). (a–b) Residual cosine gap profiles across depth; (c–d) lag-1 residual gaps; (e–f) global-NoPE head-mean logit gap profiles; (g–h) lag-1 head-mean logit gaps. All gaps are relative to lag 4096.