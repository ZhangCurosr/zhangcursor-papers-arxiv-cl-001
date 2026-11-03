# DepthGraft: Structural Regularization through Hierarchical Cross-Layer KV Reconstruction

Xiaohan Qin<sup>1,2,3</sup>, Xiangdong Zhang<sup>1,3</sup>, Yu Wang<sup>3</sup>, Huaijin Wu<sup>1,3</sup>, Zhuo Xia<sup>1,3</sup>, Yebin Yang<sup>1,3</sup>, Junchi Yan<sup>1,†</sup>

<sup>1</sup>School of AI, Shanghai Jiao Tong University <sup>2</sup>Shanghai Innovation Institute <sup>3</sup>Dots Studio, Xiaohongshu Inc.

Project page: https://github.com/jianke0604/DepthGraft

Scaling depth is a key route to stronger large language models, but Pre-LN architectures often suffer from depth-wise information dilution, limiting the effective use of deep layers. While cross-layer connectivity has proven effective at mitigating this problem, existing designs usually introduce additional computation or memory access. We revisit cross-layer KV sharing as an efficient form of such connectivity. Beyond cache compression, each KV reuse edge creates a direct gradient path from deep layers to source KV projections, turning KV sharing into an implicit structural regularizer for KV representation geometry. We provide a systematic analysis of this mechanism, showing that proper sharing topologies sharpen key retrieval directions while enriching the value content subspace, whereas existing adjacent or global-anchor designs do not fully exploit this effect. Motivated by this, we propose DepthGraft, a hierarchical KV-sharing architecture that reconstructs deep-layer KV from stride-aligned shallow– middle source pairs. DepthGraft distributes non-local gradient flow across source layers, while preserving prefill early completion and reducing KV cache by roughly one third. Experiments on dense and MoE models from 0.5B to 16B parameters show consistent downstream improvements under both GQA and MLA. Notably, on the 16B MoE model, DepthGraft achieves a 7.0% absolute improvement on MMLU-Pro, along with gains of 6.6% on CMMLU and 6.4% on C3. These results validate the potential of DepthGraft as a scalable design for large-scale pretraining. The project page is available at https://github.com/jianke0604/DepthGraft.

![](images/4a14c6a065f7bbdcaede90db895a4adf4d87bb8d4468e104f292d806b34f29cf.jpg)

![](images/1f51b5b04455d015a26b255d81a496df9290a620577e1ab6801beacb0fafafeb.jpg)

![](images/a7061f132a6a06df19b782be416a1ac1cf2b342f609caa897721a335075ae005.jpg)

![](images/96db8fa4a82e27c1cbd3e33fc1ca331f27364f6dd2b821f24948f4c14777d06d.jpg)  
Figure 1 | Results on the 16B MoE model. Left two: DepthGraft demonstrates a consistent advantage throughout training. Right two: DepthGraft reduces prefill time (TTFT) and KV cache usage by approximately one-third, while incurring almost no loss in decoding speed.

## 1. Introduction

Scaling depth has been a primary lever for improving Transformer-based large language models (LLMs) [21, 32, 42]. In modern Pre-Layer-Normalization Transformers [40], however, adding layers does not always yield the expected improvements [11, 22]. The residual stream [12] aggregates all preceding computations through a single hidden-state trajectory with fixed additive updates; as the stack grows deeper, features formed in early layers can be progressively diluted and become difficult for later layers to retrieve. This depth-wise information dilution limits the effective use of additional layers, making depth scaling less efficient than its representational capacity would suggest [9, 30].

Recent work has validated cross-layer connectivity as an effective remedy for information dilution along depth. Instead of forcing all depth history through the residual stream, recent designs allow deep layers to directly read earlier representations: Attention Residuals [33] reweights preceding hidden states to replace fixed residual accumulation, while Mixture-of-Depths Attention (MoDA) [48] augments attention with key–value pairs from previous layers. Such connections give deep layers direct access to earlier features and create shorter gradient paths to shallow layers. However, dense cross-layer access also introduces additional computation, memory traffic, and system complexity, requiring careful performance–efficiency trade-offs and sophisticated systems-level engineering.

We note that cross-layer KV sharing can also be viewed as a form of such connectivity: it provides deep layers with a channel that bypasses information dilution in the residual stream by granting direct access to shallow-layer memory representations, while gradients flow from deep layers back through the shared KV to shallow layers, supplying shallow-layer KV with optimization signals driven by deep-layer task demands—thus holding the potential to improve depth utilization. Although cross-layer KV sharing has been explored in prior work [4, 31, 49], existing methods predominantly approach it from the perspective of cache compression and inference acceleration. No prior work has systematically studied the cross-layer gradient dynamics of KV sharing, especially how such connections reshape KV representation geometry during pretraining and how the sharing topology determines this effect.

In this paper, we propose that KV sharing is not merely a cache-saving mechanism, but also a form of cross-layer gradient regularization. When a source layer’s KV is reused by a distant deep layer, the source KV receives training signals from multiple query spaces. This reshapes KV representation geometry in an asymmetric way: keys are encouraged to concentrate around shared discriminative retrieval directions, while values are pushed to cover a richer content subspace. With a proper sharing topology, this specialization sharpens memory retrieval while increasing the expressive capacity of the value content channel, yielding a useful structural bias for training. We use this lens to revisit existing KV-sharing methods and find that their topologies expose complementary limitations. Adjacent-layer sharing provides too little query-space separation, whereas global-anchor sharing concentrates heterogeneous deep-layer gradients on a few sources, preventing either design from fully exploiting the regularization effect.

Motivated by this analysis, we propose DepthGraft, a simple hierarchical KV-sharing design derived from the cross-layer regularization view. DepthGraft partitions the network into shallow, middle, and deep stages. The shallow and middle stages compute KV normally, while each deep layer reconstructs its KV from a stride-aligned shallow–middle source pair using learnable channel-wise coefficients. This construction creates non-local source–reuse connections with sufficient query-space separation, distributes cross-layer gradients across many source layers rather than a few global anchors, and gives each deep layer access to memory bases at two abstraction levels. As a result, DepthGraft preserves the efficiency benefits of KV sharing, including prefill early completion and KV-cache reduction, while turning the memory pathway into an effective regularized cross-layer connection. Training-dynamics analyses further support our theory, showing that DepthGraft reshapes KV representation geometry as predicted, and experiments across dense and MoE models demonstrate consistent performance gains at multiple scales.

Our main contributions are as follows.

1). KV sharing as cross-layer gradient regularization. We identify KV sharing as a form of cross-layer connectivity and provide a systematic analysis of how its gradient paths reshape KV geometry. Our analysis shows that proper KV sharing sharpens key retrieval directions while enriching the value content subspace, and that this effect is governed by the sharing topology.

2). Hierarchical cross-layer KV reconstruction. We propose DepthGraft, which uses a hierarchical stage layout to reconstruct deep-layer KV from stride-aligned shallow–middle source pairs. This yields a distributed non-local connectivity pattern with two abstraction-level memory bases, avoiding the concentrated-anchor structure of prior KV-sharing methods.

3). Efficient and scalable implementation. DepthGraft removes independent KV materialization from the deep stage, reducing KV cache by roughly <sup>1</sup> and allowing KV construction to finish before the final stage during prefill. Its stride-aligned source assignment is engineering-friendly and naturally compatible with PP/VPP schedules, making the design suitable for large-scale distributed training.

4). Consistent empirical gains. We validate DepthGraft on both dense and MoE models, covering model scales from 0.5B to 16B total parameters. Across scales and attention variants, DepthGraft consistently improves downstream performance while preserving inference efficiency.

## 2. Related Work

Cross-layer connectivity. A broad line of work enriches depth-wise information flow by giving each layer direct access to earlier representations [15, 17, 18]. DenseNet [15] concatenates all preceding feature maps, establishing a canonical dense cross-layer design. In Transformers, DenseFormer [23] and ANCRe [45] introduce learned layer-wise connection weights, while Value Residual Learning [47] adds targeted residual access to earlier value states. Attention Residuals [33] replaces fixed residual accumulation with learned reweighting over preceding hidden states, and MoDA [48] augments sequence attention with key–value memories from previous layers. Together, these works show that cross-layer connectivity is an effective mechanism for improving deep Transformer architectures [25, 44].

KV sharing for efficient inference. KV cache is a major memory bottleneck during autoregressive inference, motivating many methods that reduce cache storage through KV sharing [37, 38, 43]. Cross-layer KV sharing is an important direction in this line. CLA [4] and MLKV [49] reuse KV within adjacent layer groups, reducing cache but often incurring performance degradation when the reuse constraint becomes too strong. YOCO [31] shares a global KV memory for all deeper layers, enabling prefill early completion and strong cache reduction. FusedKV [20] further reconstructs deep-layer KV from two global anchors, providing more flexibility than direct global reuse. Cross-layer KV sharing is becoming an increasingly important architectural choice for large-scale LLM pretraining, as illustrated by its adoption in DeepSeek-V4.1-Flash [41] and HySparse2 [35].

Positioning. DepthGraft systematically analyzes how cross-layer KV sharing alters training dynamics, showing that the induced cross-layer gradients can regularize KV representation geometry. Guided by this analysis, DepthGraft uses stride-aligned hierarchical reconstruction to improve KV representations and model quality while preserving the KV-cache savings of sharing-based inference.

## 3. Method

## 3.1. Preliminaries

Attention and KV cache. Consider an �-layer decoder-only Transformer. We index layers by $\ell \in \{ 0 , \ldots , L - 1 \}$ and use hidden states $\mathbf { X } _ { \ell } \in \mathbb { R } ^ { T \times D }$ for a sequence of length �. Layer ℓ forms queries, keys, and values as

$$
\mathbf { Q } _ { \ell } = \mathbf { X } _ { \ell } \mathbf { W } _ { \ell } ^ { Q } , \qquad \mathbf { K } _ { \ell } = \mathbf { X } _ { \ell } \mathbf { W } _ { \ell } ^ { K } , \qquad \mathbf { V } _ { \ell } = \mathbf { X } _ { \ell } \mathbf { W } _ { \ell } ^ { V } ,\tag{1}
$$

where $\mathbf { W } _ { \rho } ^ { Q } \in \mathbb { R } ^ { D \times H _ { q } d } , \mathbf { W } _ { \rho } ^ { K } , \mathbf { W } _ { \rho } ^ { V } \in \mathbb { R } ^ { D \times H _ { k } d }$ , with $H _ { q }$ query heads, $H _ { k }$ key-value heads, and per-head dimension �. Under grouped-query attention [1], the $H _ { q }$ query heads are assigned to $H _ { k }$ KV groups. For query head $h \in \{ 0 , \ldots , H _ { q } - 1 \}$ , let $\phi ( h ) \ : = \ : \lfloor h H _ { k } / H _ { q } \rfloor$ denote its KV group. The attention output is

$$
\mathrm { A t t n } _ { \ell } = \mathrm { C o n c a t } _ { h = 0 } ^ { H _ { q } - 1 } \left( \mathrm { s o f t m a x } \Big ( \frac { \mathbf { Q } _ { \ell , h } \mathbf { K } _ { \ell , \phi ( h ) } ^ { \top } } { \sqrt { d } } \Big ) \mathbf { V } _ { \ell , \phi ( h ) } \right) \mathbf { W } _ { \ell } ^ { O } ,\tag{2}
$$

where $\mathbf { Q } _ { \ell , h } , \mathbf { K } _ { \ell , g } , \mathbf { V } _ { \ell , g } \in \mathbb { R } ^ { T \times d }$ and $\mathbf { W } _ { \ell } ^ { O } \in \mathbb { R } ^ { H _ { q } d \times D }$ . In a standard Transformer, each layer materializes its own $( \mathbf { K } _ { \ell } , \mathbf { V } _ { \ell } )$ , so autoregressive inference caches the full set $\{ ( \mathbf { K } _ { \ell } , \mathbf { V } _ { \ell } ) \} _ { \ell = 0 } ^ { L - 1 }$

Cross-layer KV sharing as memory routing. KV sharing replaces some of these independent KV tensors with KV tensors produced at other layers. We write $( \mathbf { K } _ { \rho } ^ { \mathrm { u s e } } , \mathbf { V } _ { \rho } ^ { \mathrm { u s e } } )$ for the pair actually used by layer ℓ in Eq. 2. This pair may be computed locally, $( \mathbf { K } _ { \ell } ^ { \mathrm { u s e } } , \dot { \mathbf { V } } _ { \ell } ^ { \mathrm { u s e } } ) \dot { = } ( \mathbf { K } _ { \ell } , \mathbf { V } _ { \ell } )$ , or derived from a set of source layers. Thus a KV-sharing scheme specifies a cross-layer memory-routing graph: source layers produce memory states, and reuse layers consume them. This view separates three design choices that matter for both efficiency and optimization: which layers serve as sources, how many reuse layers each source serves, and how the reused KV is constructed.

Adjacent-layer sharing. CLA [4] and MLKV [49] partition consecutive layers into groups of size �. Within each group, only the first layer computes KV and the remaining layers reuse it:

$$
\mathbf { K } _ { \ell } ^ { \mathrm { u s e } } = \mathbf { K } _ { \lfloor \ell / N \rfloor \cdot N } , \qquad \mathbf { V } _ { \ell } ^ { \mathrm { u s e } } = \mathbf { V } _ { \lfloor \ell / N \rfloor \cdot N } .\tag{3}
$$

Because source layers are interleaved throughout depth, this pattern reduces the number of cached KV tensors but does not make the cache complete before the final layer group is processed.

Global-anchor sharing. Global-anchor methods place all reusable memory before a boundary layer �. For layers $\ell > n ,$ , KV is no longer computed independently but obtained from one or more anchors. YOCO [31] uses a single anchor: ${ \bf K } _ { \rho } ^ { \mathrm { u s e } } = { \bf K } _ { n } , ~ { \bf V } _ { \rho } ^ { \mathrm { u s e } } = { \bf V } _ { n } , ~ \forall \ell > n$ . FusedKV [20] instead reconstructs deep-layer KV from two anchors through channel-wise fusion:

$$
{ \bf K } _ { \ell } ^ { \mathrm { u s e } } = \mathcal { F } _ { \ell } ^ { K } ( { \bf K } _ { 0 } , { \bf K } _ { n } ) , \qquad { \bf V } _ { \ell } ^ { \mathrm { u s e } } = \mathcal { F } _ { \ell } ^ { V } ( { \bf V } _ { 0 } , { \bf V } _ { n } ) , \qquad \forall \ell > n ,\tag{4}
$$

where $\mathcal { F } _ { \ell } ^ { K }$ and $\mathcal { F } _ { \ell } ^ { V }$ are per-layer fusion functions. Since all source KV tensors are available once layer � completes, global-anchor designs already have all the KV states needed by the remaining layers, thereby enabling prefill early completion.

Forward reuse induces backward connectivity. Although KV sharing is usually introduced as a cache-saving mechanism, it also changes the training dynamics. If layer ℓ uses KV derived from a source layer $s ,$ gradients from layer ℓ backpropagate through the reused tensors to the source projections $\pmb { W } _ { s } ^ { K }$ and ${ \mathbf W } _ { s } ^ { V }$ . Thus every memory-routing edge is also a direct cross-layer gradient path. We next analyze how these cross-layer gradient signals change the representation geometry of keys and values during training.

## 3.2. Representation Geometry of Keys and Values Under Cross-Layer Sharing

We use effective rank [27] as a compact and widely used measure of representation dimensionality. For a centered representation matrix $\mathbf { M } \in \mathbb { R } ^ { \mathbf { \tilde { N } } \times d }$ , we compute its feature covariance ${ \bf C _ { M } } =$ $\mathbf { M } ^ { \top } \mathbf { M } / N$ with eigenvalues $\{ \lambda _ { i } \} _ { i = 1 } ^ { d } ,$ , and define

$$
\operatorname { e r a n k } ( \mathbf { C } _ { \mathbf { M } } ) = \exp \Bigl ( - \sum _ { i = 1 } ^ { d } p _ { i } \log p _ { i } \Bigr ) , \qquad p _ { i } = \frac { \lambda _ { i } } { \sum _ { j = 1 } ^ { d } \lambda _ { j } } .\tag{5}
$$

Applied to $\mathbf { K } _ { \ell }$ and $\mathbf { V } _ { \ell } ,$ effective rank measures how evenly activation energy is distributed across feature directions, reflecting the dimensionality of the key retrieval geometry and value content space.

K/V geometry in standard attention. Suppressing head indices, let $\mathbf { S } _ { \ell } = \mathbf { Q } _ { \ell } \mathbf { K } _ { \ell } ^ { \top } / \sqrt { d } , \mathbf { P } _ { \ell } =$ softmax $\begin{array} { r } { ( { \bf S } _ { \ell } ) , { \bf O } _ { \ell } = { \bf P } _ { \ell } { \bf V } _ { \ell } , { \bf G } _ { \ell } = \partial \mathcal { L } / \partial { \bf S } _ { \ell } , } \end{array}$ and ${ \bf E } _ { \ell } = \partial \mathcal { L } / \partial { \bf O } _ { \ell }$ . In a standard layer,

$$
\begin{array} { r } { \nabla _ { \mathbf { K } _ { \ell } } = \frac { 1 } { \sqrt { d } } \mathbf { G } _ { \ell } ^ { \top } \mathbf { Q } _ { \ell } , \qquad \nabla _ { \mathbf { V } _ { \ell } } = \mathbf { P } _ { \ell } ^ { \top } \mathbf { E } _ { \ell } . } \end{array}\tag{6}
$$

These two expressions expose the asymmetry between K and V. Keys are optimized through the softmax logits, where the competitive normalization makes retrieval decisions depend on relative token scores. During training, key directions that repeatedly create useful logit separations receive stronger reinforcement, while directions that do not affect the induced attention distribution are less consistently updated. This produces a natural tendency for the key spectrum to concentrate along a smaller set of discriminative directions. As shown by the baseline curves in Fig. 2, the effective rank of $\mathbf { K } _ { \ell }$ decreases during training, reflecting the formation of a sharper retrieval geometry.

Values enter attention through a different pathway. Once $\mathbf { P } _ { \ell }$ is fixed, the output is a linear readout $\mathbf { O } _ { \boldsymbol { \ell } } = \mathbf { P } _ { \boldsymbol { \ell } } \mathbf { V } _ { \boldsymbol { \ell } }$ . The value gradient $\mathbf { P } _ { \ell } ^ { \top } \mathbf { E } _ { \ell }$ routes downstream output gradients back to the value matrix according to the current attention pattern. $\mathbf { A } \mathbf { s }$ the model learns to transmit richer information through attention, this linear content channel is encouraged to use more independent directions. Correspondingly, the baseline in Fig. 2 shows that the effective rank of $\mathbf { V } _ { \ell }$ increases during training.

KV sharing amplifies K/V specialization through cross-layer gradients. Consider a source layer � whose KV is reused by a distant layer ℓ. The source KV receives both its local gradient and the reuse-layer gradient:

$$
\begin{array} { r } { \nabla _ { \mathbf { K } _ { s } } ^ { \mathrm { s h a r e } } = \frac { 1 } { \sqrt { d } } \mathbf { G } _ { s } ^ { \top } \mathbf { Q } _ { s } + \frac { 1 } { \sqrt { d } } \mathbf { G } _ { \ell } ^ { \top } \mathbf { Q } _ { \ell } , \qquad \nabla _ { \mathbf { V } _ { s } } ^ { \mathrm { s h a r e } } = \mathbf { P } _ { s } ^ { \top } \mathbf { E } _ { s } + \mathbf { P } _ { \ell } ^ { \top } \mathbf { E } _ { \ell } . } \end{array}\tag{7}
$$

The same source key now serves two query spaces, and the same source value supports two readout patterns. When � and ℓ are sufficiently separated, $\mathbf { Q } _ { s }$ and $\mathbf { Q } _ { \ell }$ share task-relevant retrieval directions but also contain layer-specific components. For keys, shared discriminative directions receive coherent reinforcement, whereas layer-specific directions can interfere through the softmax competition; the source key is therefore biased toward directions useful across both layers, further sharpening the retrieval geometry. For values, the two readout terms do not compete through softmax. Different attention patterns instead route complementary gradients to the same value matrix, so the source value must cover a larger union of content directions. Thus cross-layer KV sharing amplifies the natural K/V specialization: keys become more compact retrieval indices, while values occupy richer content spaces, as observed in Fig. 2. In Appendix C, we provide a detailed theoretical analysis that further supports the mechanism above under explicit assumptions.

![](images/332e7df3d8f97e44b77d9cabe67b727cba0b227a02d31479b1c115c3610c7de8.jpg)

![](images/7fa88f96f8e2d1b66c8bca251d55bcaffd01d04b18ebc694d35e35d22a21876a.jpg)  
(a) 1.5B Dense

![](images/3f052566fa090bb009c4532d9af61631733ca27f7efd93e0273f9c17a3b2f30e.jpg)

![](images/6ec97740711db1254a5174fc14a17800d2a4defb534a44b46910d7d0dc0b5548.jpg)  
(b) 9B MoE  
Figure 2 | Evolution of KV representation geometry during training, visualized via effective rank (averaged over all layers).

## 3.3. Connectivity Structure Determines Regularization Effectiveness

Our analysis suggests that cross-layer gradients can act as a structural regularizer for KV geometry, but the regularization effect is determined by the topology of the sharing graph. We now revisit existing KV-sharing designs and show that they expose complementary limitations.

Adjacent-layer sharing: insufficient query-space separation. In CLA [4] and MLKV [49], consecutive layers are grouped (typically groups of 2–3), and all layers within a group reuse the KV computed by the first layer. For a source layer � reused by its adjacent layer �+1, the source key receives

$$
\begin{array} { r } { \nabla _ { { \bf K } _ { s } } ^ { \mathrm { a d j } } = \frac { 1 } { \sqrt { d } } { \bf G } _ { s } ^ { \top } { \bf Q } _ { s } + \frac { 1 } { \sqrt { d } } { \bf G } _ { s + 1 } ^ { \top } { \bf Q } _ { s + 1 } . } \end{array}\tag{8}
$$

This has the same form as Eq. 7, but the two query spaces are nearly adjacent in depth.

Because neighboring hidden states are highly correlated, $\mathbf { Q } _ { s }$ and $\mathbf { Q } _ { s + 1 }$ provide largely overlapping retrieval signals. The extra gradient path therefore adds little non-local pressure: it neither introduces enough queryspace mismatch to strongly compress K toward shared discriminative directions nor provides sufficiently distinct attention patterns to expand V. Fig. 3(a) is consistent with this diagnosis: adjacent sharing does not produce the pronounced K-rank concentration predicted for useful cross-layer regularization. In this regime, the representational con-

![](images/c00201c52c20fb8450b39a3cc38e6b04ce03885188e566b311a91f34be8216af.jpg)  
(a)

![](images/f126b290c5ab1c94073d0a7e8cb9e314fc9f20233b99c2ce5cb134d1d92c08f7.jpg)  
(b)  
Figure 3 | (a) Adjacent-layer sharing (CLA/M-LKV) does not reduce K erank below the baseline. (b) Global-anchor sharing concentrates gradient mass on anchor layers.

straint can dominate the weak regularization benefit: multiple layers attend to identical memory, potentially reducing the model’s effective depth.

Global-anchor sharing: gradient congestion and uneven regularization. In YOCO and FusedKV, a small number of early layers (one or two) serve as global anchors for all deep layers. For an anchor layer � shared by � deep layers, the gradient accumulates contributions from every reuse layer:

$$
\nabla _ { \mathbf { K } _ { a } } ^ { \mathrm { g l o b a l } } = \frac { 1 } { \sqrt { d } } \mathbf { G } _ { a } ^ { \top } \mathbf { Q } _ { a } + \ \sum _ { j = 1 } ^ { M } \frac { 1 } { \sqrt { d } } \operatorname { d i a g } ( \alpha _ { j } ) \mathbf { G } _ { \ell _ { j } } ^ { \top } \mathbf { Q } _ { \ell _ { j } } ,\tag{9}
$$

where $\alpha _ { j }$ are per-layer coefficients (all ones for YOCO; learnable for FusedKV). Compared with the two-way interaction in Eq. 7, the anchor must satisfy � heterogeneous query spaces at once. This creates a high-fan-in multi-objective pressure on the same KV tensor: some cross-layer interference is useful, but concentrating all deep-layer objectives on one or two anchors can make the source representation an overloaded compromise. Fig. 3(b) shows this concentrated load directly: YOCO and FusedKV place most KV-gradient mass on their anchor layers.

Global-anchor sharing also makes the regularization structurally uneven. Only the anchor layers receive direct cross-layer KV gradients from deep layers; the remaining shallow and middle layers may still receive gradients through the residual stream, but they do not participate in the direct KV-sharing path analyzed in Sec. 3.2. Fig. 5 in the appendix illustrates this effect for FusedKV: the anchor layer exhibits elevated V effective rank, whereas nearby non-anchor layers show no clear improvement over the baseline.

## 3.4. DepthGraft: Hierarchical Cross-Layer KV Reconstruction

By viewing KV-sharing designs through the lens of cross-layer gradients as structural regularization, the analysis above motivates the following design principles:

1. Sufficient cross-layer distance. Source and reuse layers must be separated by enough depth that their query spaces meaningfully diverge, enabling the gradient interference that drives K/V specialization. Adjacent-layer sharing fails this condition.

2. Distributed gradient flow with full-layer participation. Each source layer should receive gradients from only a small number of reuse layers to avoid gradient congestion, and most layers should participate in the cross-layer connectivity graph so that the regularization benefit is not confined to a few anchors.

We instantiate these requirements under an efficiency constraint: all source KV tensors should be produced before the reconstructed layers, preserving prefill early completion. Based on this, we propose DepthGraft, which introduces a hierarchical stage layout that partitions the �-layer Transformer into shallow, middle, and deep stages, each containing roughly $L / 3$ layers. The shallow and middle stages compute KV normally, while the deep stage reconstructs its KV from earlier stages. Rather than using global anchors, DepthGraft adopts stride-aligned source assignment, treating deep-layer memory as a locally reconstructed trajectory over depth. For each deep layer ℓ, DepthGraft selects one shallow source and one middle source at the same relative depth position:

$$
\begin{array} { r } { s ( \ell ) = \ell - \frac { 2 L } { 3 } , \qquad m ( \ell ) = \ell - \frac { L } { 3 } . } \end{array}\tag{10}
$$

This gives each deep layer two memory bases at different abstraction levels: shallow memory preserves lower-level lexical and token-local patterns, while middle-layer memory carries more abstract semantic states [2, 26, 46]. It then reconstructs KV from the corresponding shallow– middle source pair with learnable per-dimension coefficients:

$$
\mathbf { K } _ { \ell } = \mathbf { a } _ { \ell , 1 } \odot \mathbf { K } _ { s ( \ell ) } + \mathbf { a } _ { \ell , 2 } \odot \mathbf { K } _ { m ( \ell ) } , \qquad \mathbf { V } _ { \ell } = \mathbf { b } _ { \ell , 1 } \odot \mathbf { V } _ { s ( \ell ) } + \mathbf { b } _ { \ell , 2 } \odot \mathbf { V } _ { m ( \ell ) } ,\tag{11}
$$

where $\mathbf { a } _ { \ell , i } , \mathbf { b } _ { \ell , i }$ are channel-wise coefficients for each KV group, providing higher flexibility and expressiveness than single-source reuse. For keys, the coefficients are tied across RoPE rotation pairs, so rotary position encoding and reconstruction commute, which preserves the standard KV-cache storage logic. The stride-aligned correspondence lets each deep layer receive shallow and middle memory from the same relative depth position, preserving the ordering of representation evolution across depth. Fig. 4 in the appendix provides a workflow illustration of DepthGraft.

This design naturally yields the following advantages.

(1) Inference efficiency. Because the deep stage does not materialize independent KV, DepthGraft reduces the KV cache and prefill cost by approximately 1/3. Appendix F.1 describes the cachesaving and prefill-early-completion mechanism.

(2) Systemsfriendliness. The stride-aligned source mapping is simple to implement and naturally compatible with pipeline parallelism (PP) and virtual pipeline parallelism (VPP), avoiding additional cross-stage KV communication under an appropriate placement. Appendix F.3 provides the corresponding PP/VPP details.

(3) Regularized KV geometry. The design satisfies the two principles above: non-local source–target pairs provide query-space separation, and stride alignment distributes cross-layer gradients across source layers. As shown in Fig. 2 and Fig. 3(b), DepthGraft improves the geometry of KV representations while maintaining a healthy distributed gradient flow. The effective-rank advantage in Fig. 2 is observed across nearly all layers, rather than being confined to either source or reuse layers.

## 4. Experiments

We evaluate DepthGraft on both dense and sparse MoE Transformers, covering multiple model scales and attention variants. Our experiments are organized around three questions: (1) whether the empirical results support the analysis in Sec. 3.2, i.e., whether DepthGraft can use cross-layer gradients as a structural regularizer to improve downstream performance while preserving the KV-cache and prefill-efficiency benefits of KV sharing; (2) whether this effect scales to larger models and remains effective across different attention variants; and (3) whether the cross-layer connectivity and stage-layout choices in Sec. 3.4 are necessary, and how alternative sharing variants behave under the same training and evaluation protocol.

## 4.1. Experimental Setup

Training Settings. We conduct experiments on both dense and sparse MoE language models. For MoE models, we evaluate three scales of DeepSeekMoE-style Transformers [10] with SwiGLU feed-forward networks, ranging from 3B to 16B total parameters. For dense models, we train models from 0.5B to 1.5B parameters. The main comparisons use models trained from scratch with matched data and training recipes at each scale. Training details are provided in Appendix A, and additional experiments are provided in Appendix D.

Evaluation. We test few-shot performance on a broad downstream suite covering four capability groups: knowledge and language understanding with MMLU [13], MMLU-Pro [34], and LAM-BADA [24]; reasoning and mathematics with ARC-Challenge [6], SIQA [28], and GSM8K [7]; code generation with HumanEval [5], MBPP [3], and OpenCompass LCBench [8]; and Chinese understanding with C-Eval [16], CMMLU [19], and C3 [29].

Table 1 | Dense scaling results of baseline and DepthGraft on 0.5B–1.5B models. We report accuracy or pass rate on each benchmark and average performance over all benchmarks. The best result within each model size is shown in bold.
<table><tr><td rowspan="2">Model Method</td><td rowspan="2"></td><td colspan="3">Knowledge &amp; Understanding</td><td colspan="3">Reasoning &amp; Mathematics</td><td colspan="3">Code</td><td colspan="3">Chinese</td><td rowspan="2">Avg.</td></tr><tr><td>MMLU</td><td>M-Pro</td><td>LAMB.</td><td>ARC-C</td><td>SIQA</td><td>GSM8K</td><td>HEval</td><td>MBPP</td><td>LCBench</td><td>C-Eval</td><td>CMMLU</td><td>C3</td></tr><tr><td rowspan="2">0.5B</td><td>GQA</td><td>29.37</td><td>3.43</td><td>43.46</td><td>25.09</td><td>36.08</td><td>1.36</td><td>4.27</td><td>3.60</td><td>1.22</td><td>30.81</td><td>29.07</td><td>26.74</td><td>19.54</td></tr><tr><td>DepthGraft</td><td>30.07</td><td>4.86</td><td>43.66</td><td>28.18</td><td>34.19</td><td>1.97</td><td>6.71</td><td>5.20</td><td>0.52</td><td>29.99</td><td>30.35</td><td>27.73</td><td>20.29</td></tr><tr><td rowspan="2">1B</td><td>GQA</td><td>30.35</td><td>7.86</td><td>52.58</td><td>31.27</td><td>35.26</td><td>6.67</td><td>9.15</td><td>9.40</td><td>2.09</td><td>31.45</td><td>30.23</td><td>31.67</td><td>23.17</td></tr><tr><td>DepthGraft</td><td>32.70</td><td>8.57</td><td>54.17</td><td>27.84</td><td>38.08</td><td>6.07</td><td>9.76</td><td>12.60</td><td>2.61</td><td>31.43</td><td>30.52</td><td>33.92</td><td>24.02</td></tr><tr><td rowspan="2">1.5B</td><td>GQA</td><td>34.81</td><td>11.14</td><td>54.52</td><td>36.43</td><td>43.81</td><td>14.10</td><td>12.80</td><td>16.20</td><td>3.48</td><td>34.64</td><td>33.56</td><td>44.79</td><td>28.36</td></tr><tr><td>DepthGraft</td><td>38.14</td><td>12.57</td><td>58.25</td><td>37.11</td><td>46.67</td><td>15.01</td><td>14.63</td><td>21.40</td><td>3.83</td><td>33.74</td><td>33.81</td><td>54.25</td><td>30.78</td></tr></table>

Table 2 | MoE scaling results of baseline models and DepthGraft on 3B–16B models. We report accuracy or pass rate on each benchmark and average performance over all benchmarks. The best result within each model setting is shown in bold.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="3">Knowledge &amp; Understanding</td><td colspan="3">Reasoning &amp; Mathematics</td><td colspan="3">Code</td><td colspan="3">Chinese</td><td rowspan="2">Avg.</td></tr><tr><td>MMLU</td><td>M-Pro</td><td>LAMB.</td><td>ARC-C</td><td>SIQA</td><td>GSM8K</td><td>HEval</td><td>MBPP</td><td>LCBench</td><td>C-Eval</td><td>CMMLU</td><td>C3</td></tr><tr><td rowspan="2">3B A0.5B</td><td>GQA</td><td>31.44</td><td>7.43</td><td>51.42</td><td>26.46</td><td>35.62</td><td>6.22</td><td>12.80</td><td>14.40</td><td>1.22</td><td>32.29</td><td>28.19</td><td>28.78</td><td>23.02</td></tr><tr><td>DepthGraft</td><td>32.32</td><td>6.29</td><td>52.39</td><td>32.30</td><td>37.36</td><td>6.82</td><td>13.41</td><td>13.00</td><td>2.26</td><td>32.51</td><td>28.82</td><td>31.41</td><td>24.07</td></tr><tr><td rowspan="2">9B A1.5B</td><td>GQA</td><td>44.18</td><td>20.43</td><td>62.36</td><td>48.11</td><td>47.54</td><td>26.99</td><td>23.17</td><td>32.20</td><td>7.65</td><td>40.00</td><td>36.69</td><td>55.12</td><td>37.04</td></tr><tr><td>DepthGraft</td><td>46.53</td><td>19.71</td><td>63.48</td><td>52.92</td><td>53.58</td><td>28.81</td><td>22.56</td><td>29.60</td><td>8.17</td><td>42.65</td><td>41.26</td><td>62.63</td><td>39.33</td></tr><tr><td rowspan="2">9B A1.5B</td><td>MLA</td><td>43.30</td><td>18.29</td><td>63.48</td><td>46.74</td><td>47.75</td><td>27.52</td><td>22.56</td><td>29.60</td><td>4.00</td><td>37.85</td><td>35.02</td><td>55.95</td><td>36.01</td></tr><tr><td>DepthGraft</td><td>47.93</td><td>18.86</td><td>63.33</td><td>51.89</td><td>49.54</td><td>31.54</td><td>26.83</td><td>27.40</td><td>8.00</td><td>45.88</td><td>42.52</td><td>62.52</td><td>39.69</td></tr><tr><td rowspan="2">16B A3B</td><td>GQA</td><td>51.72</td><td>23.86</td><td>66.43</td><td>62.54</td><td>55.89</td><td>39.50</td><td>22.56</td><td>32.40</td><td>7.22</td><td>47.60</td><td>40.68</td><td>70.74</td><td>43.43</td></tr><tr><td>DepthGraft</td><td>54.21</td><td>30.86</td><td>66.78</td><td>63.57</td><td>59.26</td><td>45.34</td><td>28.05</td><td>32.60</td><td>9.39</td><td>50.00</td><td>47.29</td><td>77.15</td><td>47.04</td></tr></table>

## 4.2. Main Results

Dense models. We first evaluate DepthGraft on dense Transformers at 0.5B, 1B, and 1.5B parameters (Table 1). DepthGraft improves the average score at all three scales, by +0.75, +0.85, and +2.42 points, respectively. The gains are not confined to a single benchmark family. At 1.5B, for example, DepthGraft improves knowledge understanding (MMLU +3.33, MMLU-Pro +1.43), reasoning and mathematics (SIQA +2.86, GSM8K +0.91), code generation (MBPP +5.20), and Chinese comprehension (C3 +9.46). This consistent improvement on dense models indicates that constraining only the KV memory pathway does not reduce model capacity; instead, it provides a useful inductive bias for deep-layer memory representations.

MoE models. We next scale DepthGraft to MoE architectures from 3B to 16B total parameters (Table 2). Across GQA-based MoE models, DepthGraft consistently outperforms the corresponding baseline, improving the average score by +1.05, +2.29, and +3.61 points at 3B, 9B, and 16B, respectively. The 16B model shows especially strong gains, including MMLU-Pro (+7.00), CMMLU (+6.61), C3 (+6.41), and GSM8K (+5.84). Furthermore, the consistent gains on both 9B GQA (+2.29 Avg.) and 9B MLA (+3.68 Avg.) further show that DepthGraft is not tied to a particular attention implementation.

Taken together, the main results support the mechanism developed in Sec. 3.2: cross-layer KV reconstruction improves performance not merely by saving cache, but by reshaping the geometry of KV representations through direct cross-layer gradient regularization. The fact that the gains appear across dense models, sparse MoE models, different attention variants, and a wide range of scales suggests that the effect is a stable architectural bias rather than a scalespecific artifact. During training, we also observe that the cross-layer gradient regularization introduced by DepthGraft can weaken attention-sink behavior, further supporting the view that DepthGraft improves the organization of the memory pathway. Detailed visualization and analysis are provided in Appendix E.2.

Table 3 | Long-context inference efficiency of baseline models and DepthGraft. We report timeto-first-token (TTFT) and time-per-output-token (TPOT) from 8K to 128K context lengths.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="5">TTFT (ms)</td><td colspan="5">TPOT (ms)</td></tr><tr><td>8K</td><td>16K</td><td>32K</td><td>64K</td><td>128K</td><td>8K</td><td>16K</td><td>32K</td><td>64K</td><td>128K</td></tr><tr><td rowspan="2">9BMLA</td><td>Baseline</td><td>52.22</td><td>89.05</td><td>161.75</td><td>314.09</td><td>611.19</td><td>3.39</td><td>3.54</td><td>3.84</td><td>4.09</td><td>4.59</td></tr><tr><td>DepthGraft</td><td>43.82</td><td>69.89</td><td>122.77</td><td>233.31</td><td>447.51</td><td>3.42</td><td>3.57</td><td>3.87</td><td>4.13</td><td>4.62</td></tr><tr><td rowspan="2">16B GQA</td><td>Baseline</td><td>102.26</td><td>198.39</td><td>422.94</td><td>1000.15</td><td>2655.90</td><td>5.07</td><td>5.25</td><td>5.60</td><td>6.35</td><td>7.79</td></tr><tr><td>DepthGraft</td><td>84.44</td><td>151.13</td><td>313.31</td><td>726.11</td><td>1891.80</td><td>5.12</td><td>5.30</td><td>5.64</td><td>6.40</td><td>7.84</td></tr></table>

Efficiency Analysis. As discussed in Sec. 3.4, DepthGraft does not materialize independent KV tensors in the deep stage, yielding a fixed KV-cache reduction of roughly one third. Table 3 further reports end-to-end TTFT and TPOT on 9B MLA and 16B GQA models with a fixed 2K generation length. DepthGraft substantially reduces TTFT, and the gain becomes larger at longer contexts where prefill is increasingly compute-bound. At 128K context length, TTFT is reduced by 26.8% on 9B MLA and 28.8% on 16B GQA. In contrast, TPOT is nearly unchanged: the overhead remains below 1% across all tested context lengths. This suggests that DepthGraft preserves decoding efficiency while substantially reducing prefill latency, and the gap to the theoretical speedup can be further reduced with specialized kernel optimizations.

## 4.3. Ablation Studies

We conduct ablations on the 16B A3B MoE model to examine the roles of cross-layer gradient flow, connectivity pattern, and stage layout in DepthGraft. We further compare DepthGraft with existing KV-sharing methods under the same training and evaluation protocol.

Cross-layer gradients are the key source of the gain. Variants (2)–(4) all use the same three-stage layout, allowing us to isolate the effect of the cross-layer gradient path. Shallow-Copy directly reuses the stride-aligned shallow source from $S _ { 1 }$ , while Shallow-Copy (detach) stops gradients through the reused KV. The detached variant does not degrade relative to the GQA baseline (42.86 vs. 42.38 Avg.), indicating redundancy in deep-layer KV. Once the backward path is enabled, however, the average score increases markedly to 45.30. This controlled comparison supports our central motivation: KV sharing is useful not only as a forward memory reuse mechanism, but also because cross-layer gradients regularize earlier KV representations through deep-layer training signals.

Hierarchical reconstruction is more effective than single-source copy. Variants (3)–(5) compare single-source copy with hierarchical reconstruction under active cross-layer gradients. Shallow-Copy in Variant (3) only connects $S _ { 1 }$ to $S _ { 3 , }$ , leaving the middle stage outside the sharing graph; the 2-stage variant (5) uses full one-to-one copy, but each reuse layer still receives only a single memory source. DepthGraft instead reconstructs each deep-layer KV from a stride-aligned S<sub>1</sub>–S<sub>2</sub> source pair, improving the average score over both single-source alternatives (47.56 vs. 45.30 and 43.58). The gains on representative benchmarks such as M-Pro, C3, C-Eval, and GSM8K suggest that the benefit comes from combining broad source participation with a more flexible hierarchical memory basis, rather than from cross-layer gradient flow alone.

The three-stage layout gives the best overall trade-off. Variants (5)–(8) compare different stage partitions under full gradient flow. The reconstructed layers take stride-aligned sources and obtain their KV by either direct copy or hierarchical reconstruction. The 2-stage variant is too restrictive, while the 4-stage variants test whether using more source stages or more reuse stages is beneficial. The results suggest that reconstruction sources are not simply “the more the better”: reconstructing the last stage from three source stages underperforms the proposed two-source

Table 4 | Ablation study on the 16B A3B MoE model. We decompose DepthGraft along two axes: cross-layer gradient connectivity and stage layout. Best results within each group are in bold.
<table><tr><td>Method</td><td>Grad. Flow</td><td>Stage Layout</td><td>MMLU</td><td>M-Pro</td><td>CMMLU</td><td>C3</td><td>C-Eval</td><td>GSM8K</td><td>HEval</td><td>Avg.</td></tr><tr><td>(1) GQA Baseline</td><td></td><td>一</td><td>51.72</td><td>23.86</td><td>40.68</td><td>70.74</td><td>47.60</td><td>39.50</td><td>22.56</td><td>42.38</td></tr><tr><td colspan="9">Gradient Connectivity &amp; Hierarchical Reconstruction Variants</td><td></td></tr><tr><td>(2) Shallow-Copy (detach)</td><td>x</td><td> $S _ { 1 }  S _ { 3 }$ </td><td>49.72</td><td>24.14</td><td>42.08</td><td>70.96</td><td>44.76</td><td>41.55</td><td>26.83</td><td>42.86</td></tr><tr><td>(3) Shallow-Copy</td><td>Partial</td><td> $S _ { 1 }  S _ { 3 }$ </td><td>52.53</td><td>29.43</td><td>46.72</td><td>70.96</td><td>45.50</td><td>43.29</td><td>28.66</td><td>45.30</td></tr><tr><td>(4) DepthGraft</td><td>Full</td><td> $S _ { 1 } \oplus S _ { 2 } \to S _ { 3 }$ </td><td>54.21</td><td>30.86</td><td>47.29</td><td>77.15</td><td>50.00</td><td>45.34</td><td>28.05</td><td>47.56</td></tr><tr><td colspan="9">Stage Layout Variants</td><td></td></tr><tr><td>(5) DepthGraft (2-stage)</td><td>Full</td><td> $S _ { 1 }  S _ { 2 }$ </td><td>52.11</td><td>22.71</td><td>47.82</td><td>70.25</td><td>46.56</td><td>40.64</td><td>25.00</td><td>43.58</td></tr><tr><td>(6) DepthGraft (4-stage, 3-src)</td><td>Full</td><td> ${ \sf S } _ { 1 } \oplus { \sf S } _ { 2 } \oplus { \sf S } _ { 3 }  { \sf S } _ { 4 }$ </td><td>51.65</td><td>25.71</td><td>43.96</td><td>75.40</td><td>45.36</td><td>47.69</td><td>28.05</td><td>45.40</td></tr><tr><td>(7) DepthGraft (4-stage, 2-src)</td><td>Full</td><td> $S _ { 1 } \oplus S _ { 2 } \to S _ { 3 } , S _ { 4 }$ </td><td>53.26</td><td>27.14</td><td>47.43</td><td>72.05</td><td>50.48</td><td>43.14</td><td>23.78</td><td>45.33</td></tr><tr><td>(8) DepthGraft</td><td>Full</td><td> $S _ { 1 } \oplus S _ { 2 } \to S _ { 3 }$ </td><td>54.21</td><td>30.86</td><td>47.29</td><td>77.15</td><td>50.00</td><td>45.34</td><td>28.05</td><td>47.56</td></tr></table>

Table 5 | Comparison with prior KV sharing methods on the 16B A3B MoE model. Type indicates the sharing topology. Best results are in bold.
<table><tr><td>Method</td><td>Type</td><td>MMLU</td><td>M-Pro</td><td>CMMLU</td><td>C3</td><td>C-Eval</td><td>GSM8K</td><td>HEval</td><td>Avg.</td></tr><tr><td>GQA Baseline</td><td>一</td><td>51.72</td><td>23.86</td><td>40.68</td><td>70.74</td><td>47.60</td><td>39.50</td><td>22.56</td><td>42.38</td></tr><tr><td>CLA / MLKV</td><td>Adjacent</td><td>48.81</td><td>20.14</td><td>40.64</td><td>58.25</td><td>40.50</td><td>37.91</td><td>20.12</td><td>38.05</td></tr><tr><td>YOCO</td><td>Global-1</td><td>50.28</td><td>27.57</td><td>43.87</td><td>67.73</td><td>46.59</td><td>42.91</td><td>24.39</td><td>43.33</td></tr><tr><td>FusedKV</td><td>Global-2</td><td>53.86</td><td>27.71</td><td>46.48</td><td>70.90</td><td>50.66</td><td>44.88</td><td>25.00</td><td>45.64</td></tr><tr><td>DepthGraft</td><td>Hierarchical</td><td>54.21</td><td>30.86</td><td>47.29</td><td>77.15</td><td>50.00</td><td>45.34</td><td>28.05</td><td>47.56</td></tr></table>

DepthGraft layout, and the 4-stage two-source variant also remains below the three-stage design. Thus, a three-stage partition provides a stronger operating point, balancing two complementary source abstraction levels, distributed gradient flow, and the intended final-stage cache-saving region.

DepthGraft outperforms prior KV-sharing topologies. Table 5 compares DepthGraft with existing KV-sharing designs under the same 16B A3B MoE setting. The results further support the topology analysis in Sec. 3.3. Adjacent sharing substantially underperforms the baseline, supporting the view that reusing nearby KV reduces the model’s effective depth. Global-anchor sharing performs better than adjacent sharing and can improve over the baseline, but shows limited gains because the gradient regularization is concentrated on a small set of anchors. By distributing non-local source–reuse connections across layers, DepthGraft achieves the best overall performance.

## 4.4. Long-Context Evaluation

We further evaluate whether cross-layer KV reconstruction preserves long-context retrieval. Both the baseline and DepthGraft 16B A3B models undergo continued training at a 32K context length, with the RoPE base adjusted for length extension. Table 6 reports ten RULER [14] tasks at context lengths from 4K to 32K, covering needle-in-a-haystack retrieval, variable tracking, frequent-word extraction, and question answering.

DepthGraft achieves a higher average at every evaluated context length, including 94.8 vs. 86.8 at 4K and 82.4 vs. 76.2 at 32K. The gains extend beyond simple retrieval: at 32K, variable tracking improves from 37.8 to 76.6 and frequent-word extraction from 64.0 to 74.7. It also matches or exceeds the baseline on most reported NIAH subtasks. These results show that reconstructing deep-layer KV can retain and improve long-context capabilities without storing independent KV at every layer.

Table 6 | Long-context performance on RULER after context extension. NIAH denotes needle-ina-haystack retrieval; VT, FWE, and QA denote variable tracking, frequent-word extraction, and SQuAD-based question answering.
<table><tr><td rowspan="2">Length</td><td rowspan="2">Method</td><td colspan="7">NIAH</td><td rowspan="2">VT</td><td rowspan="2">FWE</td><td rowspan="2">QA</td><td rowspan="2">Avg.</td></tr><tr><td>s1</td><td>s2</td><td>s3</td><td>mk1</td><td>mk3</td><td>mv</td><td>mq</td></tr><tr><td rowspan="2">4K</td><td>Baseline</td><td>100</td><td>100</td><td>75.0</td><td>100</td><td>99.0</td><td>90.5</td><td>95.5</td><td>43.2</td><td>80.7</td><td>84.0</td><td>86.8</td></tr><tr><td>DepthGraft</td><td>100</td><td>100</td><td>100</td><td>100</td><td>99.0</td><td>98.3</td><td>98.5</td><td>81.6</td><td>84.3</td><td>86.0</td><td>94.8</td></tr><tr><td rowspan="2">8K</td><td>Baseline</td><td>100</td><td>100</td><td>70.0</td><td>97.0</td><td>96.0</td><td>58.3</td><td>95.3</td><td>44.0</td><td>65.0</td><td>66.0</td><td>79.2</td></tr><tr><td>DepthGraft</td><td>100</td><td>100</td><td>100</td><td>94.0</td><td>100</td><td>98.0</td><td>94.5</td><td>80.0</td><td>78.7</td><td>61.0</td><td>90.6</td></tr><tr><td rowspan="2">16K</td><td>Baseline</td><td>100</td><td>100</td><td>82.0</td><td>96.0</td><td>93.0</td><td>48.3</td><td>91.5</td><td>45.6</td><td>66.0</td><td>63.0</td><td>78.5</td></tr><tr><td>DepthGraft</td><td>100</td><td>100</td><td>96.0</td><td>92.0</td><td>96.0</td><td>87.5</td><td>86.0</td><td>77.6</td><td>78.3</td><td>63.0</td><td>87.6</td></tr><tr><td rowspan="2">32K</td><td>Baseline</td><td>100</td><td>100</td><td>99.0</td><td>95.0</td><td>61.0</td><td>66.3</td><td>87.8</td><td>37.8</td><td>64.0</td><td>51.0</td><td>76.2</td></tr><tr><td>DepthGraft</td><td>100</td><td>100</td><td>100</td><td>94.0</td><td>76.0</td><td>73.5</td><td>76.5</td><td>76.6</td><td>74.7</td><td>53.0</td><td>82.4</td></tr></table>

## 5. Conclusion and Future Work

We presented DepthGraft, a hierarchical cross-layer KV reconstruction method that turns KV sharing from a cache-saving mechanism into a regularized memory pathway. Our analysis shows that KV reuse creates direct cross-layer gradient paths that reshape KV representation geometry, and that the effectiveness of this regularization depends on the topology of the sharing graph. Motivated by this view, DepthGraft reconstructs deep-layer KV from stride-aligned shallow–middle source pairs, preserving the efficiency benefits of KV sharing while distributing non-local gradient flow across source layers. Experiments on dense and MoE models from 0.5B to 16B show consistent downstream gains, substantial reductions in KV cache and prefill cost, and negligible overhead during decoding. Future work can develop a more rigorous theoretical understanding of how cross-layer KV sharing improves representation geometry, explore more effective sharing topologies, and further scale this direction to larger pretraining regimes.

## References

[1] Joshua Ainslie, James Lee-Thorp, Michiel De Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4895–4901, 2023.

[2] Nura Aljaafari, Danilo S Carvalho, and André Freitas. The mechanics of conceptual interpretation in gpt models: Interpretative insights. arXiv preprint arXiv:2408.11827, 2024.

[3] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

[4] William Brandon, Mayank Mishra, Aniruddha Nrusimha, Rameswar Panda, and Jonathan Ragan-Kelley. Reducing transformer key-value cache size with cross-layer attention. Advances in Neural Information Processing Systems, 37:86927–86957, 2024.

[5] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

[6] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.

[7] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[8] OpenCompass Contributors. Opencompass: A universal evaluation platform for foundation models. https://github.com/open-compass/opencompass, 2023.

[9] Róbert Csordás, Christopher D Manning, and Christopher Potts. Do language models use their depth efficiently? arXiv preprint arXiv:2505.13898, 2025.

[10] Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Yu Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixture-ofexperts language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1280–1297, 2024.

[11] Andrey Gromov, Kushal Tirumala, Hassan Shapourian, Paolo Glorioso, and Daniel A Roberts. The unreasonable ineffectiveness of the deeper layers. arXiv preprint arXiv:2403.17887, 2024.

[12] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

[13] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

[14] Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654, 2024.

[15] Gao Huang, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger. Densely connected convolutional networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4700–4708, 2017.

[16] Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Yao Fu, et al. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models. Advances in neural information processing systems, 36:62991–63010, 2023.

[17] Suzana Ilic, Edison Marrese-Taylor, Jorge Balazs, and Yutaka Matsuo. Deep contextualized word representations for detecting sarcasm and irony. In Proceedings of the 9th workshop on computational approaches to subjectivity, sentiment and social media analysis, pages 2–7, 2018.

[18] Jonas Knupp, Jan Hendrik Metzen, Jeremias Bohn, Georg Groh, and Kristian Kersting. Depthrecurrent attention mixtures: Giving latent reasoning the attention it deserves. arXiv preprint arXiv:2601.21582, 2026.

[19] Haonan Li, Yixuan Zhang, Fajri Koto, Yifei Yang, Hai Zhao, Yeyun Gong, Nan Duan, and Timothy Baldwin. Cmmlu: Measuring massive multitask language understanding in chinese. In Findings of the Associationfor Computational Linguistics: ACL 2024, pages 11260–11285, 2024.

[20] Hongzhan Lin, Zhiqi Bai, Xinmiao Zhang, Sen Yang, Xiang Li, Siran Yang, Yunlong Xu, Jiaheng Liu, Yongchi Zhao, Jiamang Wang, et al. Reconstructing kv caches with cross-layer fusion for enhanced transformers. arXiv preprint arXiv:2512.03870, 2025.

[21] Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, et al. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025.

[22] Xin Men, Mingyu Xu, Qingyu Zhang, Qianhao Yuan, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, and Weipeng Chen. Shortgpt: Layers in large language models are more redundant than you expect. In Findings of the Association for Computational Linguistics: ACL 2025, pages 20192– 20204, 2025.

[23] Matteo Pagliardini, Amirkeivan Mohtashami, François Fleuret, and Martin Jaggi. Denseformer: Enhancing information flow in transformers via depth weighted averaging. In Advances in Neural Information Processing Systems, 2024.

[24] Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc-Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. The lambada dataset: Word prediction requiring a broad discourse context. In Proceedings ofthe 54th annual meeting ofthe association for computational linguistics (volume 1: Long papers), pages 1525–1534, 2016.

[25] Xiaohan Qin, Xiangdong Zhang, Tuo Dai, Sunan Zou, Huaijin Wu, Yebin Yang, Zhuo Xia, Lin Yao, Yuliang Liu, and Junchi Yan. Multi-stream attention residuals. arXiv preprint arXiv:2610.13630, 2026.

[26] Elisabetta Rocchetti and Alfio Ferrara. How instruction-tuning imparts length control: A crosslingual mechanistic analysis. arXiv preprint arXiv:2509.02075, 2025.

[27] Olivier Roy and Martin Vetterli. The effective rank: A measure of effective dimensionality. In 2007 15th European signal processing conference, pages 606–610. IEEE, 2007.

[28] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Social iqa: Commonsense reasoning about social interactions. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pages 4463–4473, 2019.

[29] Kai Sun, Dian Yu, Dong Yu, and Claire Cardie. Investigating prior knowledge for challenging chinese machine reading comprehension. Transactions of the Association for Computational Linguistics, 8:141–155, 2020.

[30] Wenfang Sun, Xinyuan Song, Pengxiang Li, Lu Yin, Yefeng Zheng, and Shiwei Liu. The curse of depth in large language models. arXiv preprint arXiv:2502.05795, 2025.

[31] Yutao Sun, Li Dong, Yi Zhu, Shaohan Huang, Wenhui Wang, Shuming Ma, Quanlu Zhang, Jianyong Wang, and Furu Wei. You only cache once: Decoder-decoder architectures for language models. Advances in Neural Information Processing Systems, 37:7339–7361, 2024.

[32] Kimi Team, Yifan Bai, Yiping Bao, Y Charles, Cheng Chen, Guanduo Chen, Haiting Chen, Huarong Chen, Jiahao Chen, Ningxin Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.

[33] Kimi Team, Guangyu Chen, Yu Zhang, Jianlin Su, Weixin Xu, Siyuan Pan, Yaoyu Wang, Yucheng Wang, Guanduo Chen, Bohong Yin, et al. Attention residuals. arXiv preprint arXiv:2603.15031, 2026.

[34] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multitask language understanding benchmark. Advances in Neural Information Processing Systems, 37: 95266–95290, 2024.

[35] Jianyu Wei, Yizhao Gao, Qihao Zhang, Shimao Chen, Zhengju Tang, Yu Cheng, Shengjie Zhou, Zihan Jiang, Yifan Song, Hailin Zhang, et al. HySparse2: Hybrid sparse attention with two-level KV sharing. arXiv preprint arXiv:2609.26368, 2026.

[36] Kaiyue Wen, Zhiyuan Li, Jason Wang, David Hall, Percy Liang, and Tengyu Ma. Understanding warmup-stable-decay learning rates: A river valley loss landscape perspective. arXiv preprint arXiv:2410.05192, 2024. URL https://arxiv.org/abs/2410.05192.

[37] Haoyi Wu and Kewei Tu. Layer-condensed kv cache for efficient inference of large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11175–11188, 2024.

[38] Huaijin Wu, Lianqiang Li, Hantao Huang, Yi Tu, Jihang Zhang, Minghui Yu, and Junchi Yan. HShare: Fast LLM decoding by hierarchical key-value sharing. In International Conference on Learning Representations, volume 2025, pages 89412–89428, 2025.

[39] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.

[40] Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tieyan Liu. On layer normalization in the transformer architecture. In International conference on machine learning, pages 10524–10533. PMLR, 2020.

[41] Anyi Xu, B Li, Bangcai Lin, Bing Xue, BingCheng Xian, Bingzheng Xu, Bochao Wu, Bowei Zhang, Boyi Deng, CC Yu, et al. DeepSeek-V4.1-Flash: Pushing the limits of KV cache compression. arXiv preprint arXiv:2609.19969, 2026.

[42] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[43] Yifei Yang, Zouying Cao, Qiguang Chen, Libo Qin, Dongjie Yang, Hai Zhao, and Zhi Chen. Kvsharer: Efficient inference via layer-wise dissimilar kv cache sharing. arXiv preprint arXiv:2410.18517, 2024.

[44] Xiangdong Zhang, Xiaohan Qin, Sunan Zou, Tuo Dai, Xiaoming Shi, Huaijin Wu, Yebin Yang, Zhuo Xia, Shaofeng Zhang, Lin Yao, et al. xhc: Expanded hyper-connections. arXiv preprint arXiv:2607.14530, 2026.

[45] Yilang Zhang et al. Ancre: Adaptive neural connection reassignment for efficient depth scaling. arXiv preprint arXiv:2602.09009, 2026.

[46] Zifan Zheng, Yezhaohui Wang, Yuxin Huang, Shichao Song, Mingchuan Yang, Bo Tang, Feiyu Xiong, and Zhiyu Li. Attention heads of large language models: A survey. arXiv preprint arXiv:2409.03752, 2024.

[47] Zhanchao Zhou et al. Value residual learning. In Proceedings of the Annual Meeting of the Association for Computational Linguistics, pages 28341–28356, 2025.

[48] Lianghui Zhu, Yuxin Fang, Bencheng Liao, Shijie Wang, Tianheng Cheng, Zilong Huang, Chen Chen, Lai Wei, Yutao Zeng, Ya Wang, et al. Mixture-of-depths attention. arXiv preprint arXiv:2603.15619, 2026.

[49] Zayd Muhammad Kawakibi Zuhri, Muhammad Farid Adilazuarda, Ayu Purwarianti, and Alham Fikri Aji. Mlkv: Multi-layer key-value heads for memory efficient transformer decoding. In Findings ofthe Associationfor Computational Linguistics: NAACL 2025, pages 5516–5525, 2025.

## A. Experimental Details

Model configurations. Tables 7 and 8 summarize the model configurations and training hyperparameters used in our dense and MoE experiments. All models are decoder-only Transformers with SwiGLU feed-forward layers, RMSNorm, and RoPE positional embeddings. The dense models use GQA throughout, while the MoE experiments cover both GQA and MLA variants. All MoE backbones follow a DeepSeekMoE-style layout [10]: the first Transformer layer is dense and the remaining layers are MoE layers.

Table 7 | Hyperparameters and model configurations for dense-model pretraining.
<table><tr><td>Group</td><td>Parameter</td><td>0.5B</td><td>1B</td><td>1.5B</td></tr><tr><td rowspan="7">Training</td><td>lr schedule</td><td>WSD</td><td>WSD</td><td>WSD</td></tr><tr><td>lr</td><td>8.0×10-4</td><td>6.0×10-4</td><td>6.0×10-4</td></tr><tr><td>warmup steps</td><td>2,000</td><td>2,000</td><td>2,000</td></tr><tr><td>decay ratio</td><td>0.2</td><td>0.2</td><td>0.2</td></tr><tr><td>AdamW-β</td><td>(0.9, 0.95)</td><td>(0.9, 0.95)</td><td>(0.9, 0.95)</td></tr><tr><td>weight decay</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>grad clip</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td rowspan="10">Model</td><td>global batch size</td><td>512</td><td>512</td><td></td></tr><tr><td>hidden dim.</td><td>1024</td><td>1024</td><td>1024</td></tr><tr><td># layers</td><td>9</td><td>18</td><td>28</td></tr><tr><td>head dim.</td><td>128</td><td>128</td><td>128</td></tr><tr><td># query heads</td><td>8</td><td>8</td><td>8</td></tr><tr><td># KV heads</td><td>4</td><td>4</td><td>4</td></tr><tr><td>context length</td><td>8192</td><td>8192</td><td>8192</td></tr><tr><td>dense FFN size</td><td>10944</td><td>10944</td><td>10944</td></tr><tr><td>DepthGraft reuse layers</td><td>3/9</td><td>6/18</td><td>9/28</td></tr><tr><td>Normalization</td><td></td><td>RMSNorm (€ = 10−5)</td><td></td></tr><tr><td rowspan="2">Data</td><td>Position Embedding</td><td></td><td>RoPE (θ = 50000)</td><td></td></tr><tr><td>tokenizer # tokens</td><td>qwen2</td><td>qwen2</td><td>qwen2</td></tr></table>

Table 8 | Hyperparameters and model configurations for MoE-model pretraining.
<table><tr><td>Group</td><td>Parameter</td><td>3B A0.5B GQA</td><td>9B A1.5B GQA</td><td>9B A1.5B MLA</td><td>16B A3B GQA</td></tr><tr><td rowspan="7">Training</td><td>lr schedule</td><td>WSD</td><td>WSD</td><td>WSD</td><td>WS</td></tr><tr><td>lr</td><td>4.82×10 -4</td><td>4.82×10⁻4</td><td>4.82×10⁻4</td><td> $4 . 2 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>warmup steps</td><td>2,000</td><td>2,000</td><td>2,000</td><td>750</td></tr><tr><td>decay ratio</td><td>0.2</td><td>0.2</td><td>0.2</td><td></td></tr><tr><td>AdamW-β</td><td>(0.9, 0.95)</td><td>(0.9, 0.95)</td><td>(0.9, 0.95)</td><td>(0.9, 0.95)</td></tr><tr><td>weight decay</td><td>0.1</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>grad clip global batch size</td><td>1.0 512</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>一</td></tr><tr><td rowspan="14">Model</td><td>hidden dim.</td><td>768</td><td>2080</td><td>2080</td><td>2048</td></tr><tr><td># layers</td><td>15</td><td>15</td><td>15</td><td>28</td></tr><tr><td># dense layers</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td># MoE layers</td><td>14</td><td>14</td><td>14</td><td>27</td></tr><tr><td>head dim.</td><td>128</td><td>128</td><td>一</td><td>128</td></tr><tr><td># query heads</td><td>8</td><td>16</td><td>16</td><td>16</td></tr><tr><td># KV heads</td><td>4</td><td>8</td><td></td><td>8</td></tr><tr><td>MLA q-lora rank</td><td>一</td><td>一</td><td>1536</td><td>一</td></tr><tr><td>MLA kv-lora rank</td><td>一</td><td>一</td><td>384</td><td>一</td></tr><tr><td>MLA qk head dim.</td><td>一</td><td>一</td><td>80+48</td><td>一</td></tr><tr><td>MLA v head dim.</td><td></td><td></td><td>128</td><td></td></tr><tr><td>context length</td><td>8192</td><td>8192</td><td>8192</td><td>8192</td></tr><tr><td>dense FFN size</td><td>10944</td><td>6144</td><td>6144</td><td>10944</td></tr><tr><td># routed experts</td><td>144</td><td>144</td><td>144</td><td>64</td></tr><tr><td>topK route</td><td>8</td><td>8</td><td>8</td><td>6</td></tr><tr><td>MoE FFN size</td><td>544</td><td>704</td><td>704</td><td>1408</td></tr><tr><td>shared exp. size</td><td>544</td><td>704</td><td>704</td><td>2816</td></tr><tr><td>DepthGraft reuse layers</td><td>5/15</td><td>5/15</td><td>5/15</td><td>9/28</td></tr><tr><td>Normalization</td><td></td><td></td><td>RMSNorm (€ = 10−5)</td><td></td></tr><tr><td>Position Embedding</td><td colspan="4">RoPE (θ = 50000)</td></tr><tr><td rowspan="3">Data</td><td>tokenizer</td><td></td><td>qwen2</td><td>qwen2</td><td>qwen2</td></tr><tr><td></td><td>qwen2</td><td></td><td></td><td></td></tr><tr><td># tokens</td><td></td><td></td><td></td><td></td></tr></table>

Training hyperparameters. All comparisons at the same model scale use identical training recipes and token budgets, so that differences can be attributed to the KV-sharing design rather than optimization settings. For the prior KV-sharing baselines in Table 5, CLA/MLKV use threelayer sharing groups, while YOCO and FusedKV start reusing KV after the middle stage, i.e., after the first 2�/3 layers, as illustrated in Fig. 3(b). This matches the reuse ratio of DepthGraft for YOCO and FusedKV, allowing us to compare performance under different connectivity patterns. For our 28-layer models, we leave layer 0 unshared and apply the three-stage reconstruction to the remaining 27 layers, resulting in three 9-layer stages. We use AdamW with $\beta _ { 1 } { = } 0 . 9 ,$ $\beta _ { 2 } { = } 0 . 9 5$ , weight decay 0.1, and gradient clipping at 1.0. Models smaller than 16B use a warmup– stable–decay schedule [36]: the final 20% of training tokens use exponential decay to reduce the learning rate to 0.1 of its peak value. For the 16B MoE model, we use a warmup–stable schedule without LR decay and with batch-size ramp-up. All models use the qwen2 tokenizer with a vocabulary size of 152,064 and are trained with a context length of 8192. Our pretraining corpus is primarily proprietary and private, built from web text, code, math, and multilingual sources, with exact/fuzzy deduplication, rule-based filtering, and LLM-based quality filtering.

Evaluation protocol. We evaluate all models with the same few-shot protocol under OpenCompass. The benchmark suite covers knowledge and language understanding with MMLU (5-shot),

MMLU-Pro (5-shot), and LAMBADA (0-shot); reasoning and mathematics with ARC-Challenge (25-shot), SIQA (5-shot), and GSM8K (4-shot); code generation with HumanEval (0-shot), MBPP (3-shot), and OpenCompass LCBench2023 (5-shot); and Chinese understanding with C-Eval (5-shot), CMMLU (5-shot), and C3 (3-shot).

## B. Method Details and Additional Illustrations

![](images/476086f9f65e8184a07dcd828851b9540f15226e2c67cdb8ec532203be466aa5.jpg)  
Figure 4 | DepthGraft workflow. The shallow and middle stages compute and cache KV normally, while deep-stage KV is reconstructed from stride-aligned shallow–middle source pairs. The deep stage therefore does not materialize independent KV cache, while gradients from deep-layer attention flow back to the source KV projections through the reconstruction paths.

RoPE-pair tying for K coefficients. Rotary position embeddings apply a block-diagonal rotation to each two-dimensional channel pair of the query and key vectors. For the NeoX-style layout used in our implementation, dimensions � and $j + d / 2$ form one RoPE pair. If a channel-wise coefficient assigns different values to the two dimensions in the same pair, then applying RoPE before and after reconstruction would no longer be equivalent. Consistent with FusedKV [20], we therefore tie the K reconstruction coefficients within each RoPE pair. Concretely, for each source coefficient $\mathbf { a } _ { \ell , i } ,$ we store a half-sized tensor $\mathbf { a } _ { \ell , i } \in \mathbb { R } ^ { H _ { k } \times d / 2 }$ and expand it to $\tilde { \mathbf { a } } _ { \ell , i } \in \mathbb { R } ^ { H _ { k } \times d }$ by

$$
\tilde { \mathbf { a } } _ { \ell , i } [ h , j ] = \tilde { \mathbf { a } } _ { \ell , i } [ h , j + d / 2 ] = \mathbf { a } _ { \ell , i } [ h , j ] , \qquad j = 1 , \dotsc , d / 2 .\tag{12}
$$

Let $\mathbf { R } _ { p }$ denote the RoPE rotation matrix at position �. Because each expanded coefficient is constant within every rotation pair, it commutes with the RoPE rotation:

$$
\begin{array} { r } { \mathbf { R } _ { p } ( \tilde { \mathbf { a } } _ { \ell , i } \odot \mathbf { K } ) = \tilde { \mathbf { a } } _ { \ell , i } \odot \mathbf { R } _ { p } ( \mathbf { K } ) . } \end{array}\tag{13}
$$

Therefore, for the reconstructed key,

$$
\begin{array} { r } { \mathbf { R } _ { p } \bigl ( \tilde { \mathbf { a } } _ { \ell , 1 } \odot \mathbf { K } _ { s ( \ell ) } + \tilde { \mathbf { a } } _ { \ell , 2 } \odot \mathbf { K } _ { m ( \ell ) } \bigr ) = \tilde { \mathbf { a } } _ { \ell , 1 } \odot \mathbf { R } _ { p } \bigl ( \mathbf { K } _ { s ( \ell ) } \bigr ) + \tilde { \mathbf { a } } _ { \ell , 2 } \odot \mathbf { R } _ { p } \bigl ( \mathbf { K } _ { m ( \ell ) } \bigr ) . } \end{array}\tag{14}
$$

This means that key reconstruction can be performed either before or after RoPE without changing the result, preserving the standard KV-cache storage logic. Value vectors do not use RoPE, so the V coefficients $\mathbf { b } _ { \ell , i }$ are stored at full channel dimension and do not require pair tying.

Coefficient initialization. All reconstruction coefficients $\mathbf { a } _ { \ell , 1 } , \mathbf { a } _ { \ell , 2 } , \mathbf { b } _ { \ell , 1 } , \mathbf { b } _ { \ell , 2 }$ are initialized to 0.5. At the start of training, each deep layer therefore receives an equal-weight reconstruction from its shallow and middle sources. This provides a stable interpolation between the two memory bases before the model learns layer- and channel-specific preferences. As shown in Appendix E.4, the coefficients later deviate from this symmetric initialization and learn different source preferences for K and V.

## C. Theoretical Analysis

A conditional spectral view of cross-layer KV regularization. We provide a local spectral analysis to support the effective-rank diagnostic in Eq. 5. For a centered key or value activation matrix M, denote its feature covariance by $\mathbf { C _ { M } } = \mathbf { M } ^ { \top } \mathbf { M } / N$ . The effective rank is high when the covariance spectrum is close to uniform, and low when activation energy is concentrated in a few feature directions. Our goal is not to prove that cross-layer KV sharing always changes effective rank in a fixed direction. Instead, we show that the observed $\mathrm { K } / \mathrm { V }$ effective-rank trends follow under a natural spectral-alignment condition on the cross-layer update covariance.

Lemma 1 (spectral injection and effective rank). Consider a local update of the activation matrix

$$
\mathbf { M } ^ { + } = \mathbf { M } + \eta \Delta ,\tag{15}
$$

where � is the induced activation-space update and � is the learning rate. The covariance update admits the exact decomposition

$$
\begin{array} { c } { \displaystyle \mathbf { C } _ { \mathbf { M } ^ { + } } = \frac { 1 } { N } ( \mathbf { M } + \eta \mathbf { \Delta } \mathbf { \Delta } ) ^ { \top } ( \mathbf { M } + \eta \mathbf { \Delta } \mathbf { \Delta } ) } \\ { \mathbf { \Delta } = \mathbf { C } _ { \mathbf { M } } + \eta \mathbf { A } _ { \mathbf { M } , \Delta } + \eta ^ { 2 } \boldsymbol { \Sigma } _ { \Delta } , } \end{array}\tag{16}
$$

where

$$
\mathbf { A } _ { \mathbf { M } , \Delta } = \frac { 1 } { N } ( \mathbf { M } ^ { \top } \Delta + \Delta ^ { \top } \mathbf { M } ) , \qquad \Sigma _ { \Delta } = \frac { 1 } { N } \Delta ^ { \top } \Delta .\tag{17}
$$

We analyze the directional trend over a short training window, where stochastic mini-batch fluctuations make the first-order alignment term $\mathbf { A _ { M , \Delta } }$ average to a small or low-rank correction that does not reverse the spectral ordering induced by $\Sigma _ { \Delta }$ . Equivalently, for the purpose of identifying which feature directions receive persistent update energy, we approximate

$$
\mathbb { E } [ \mathbf { C _ { M ^ { + } } } ] \approx \mathbf { C _ { M } } + \eta ^ { 2 } \mathbb { E } [ \Sigma _ { \Delta } ] .\tag{18}
$$

Let $\begin{array} { r } { { \bf { C } } _ { \bf { M } } = { \bf U } \mathrm { d i a g } ( \lambda _ { 1 } , \dots , \lambda _ { d } ) { \bf U } ^ { \top } } \end{array}$ with $\lambda _ { 1 } \geq \cdots \geq \lambda _ { d } \geq 0$ . Assume that the update covariance is approximately aligned with the current representation geometry:

$$
\mathbf { U } ^ { \top } \mathbb { E } [ \Sigma _ { \Delta } ] \mathbf { U } = \mathrm { d i a g } ( \mu _ { 1 } , \dots , \mu _ { d } ) + \mathbf { R } , \qquad \| \mathbf { R } \| _ { 2 } \ll \| \mathrm { d i a g } ( \mu ) \| _ { 2 } .\tag{19}
$$

Ignoring the small off-diagonal residual R, the eigenvalues evolve as

$$
\lambda _ { i } ^ { + } = \lambda _ { i } + \eta ^ { 2 } \mu _ { i } = \lambda _ { i } ( 1 + \eta ^ { 2 } \rho _ { i } ) , \qquad \rho _ { i } = \mu _ { i } / \lambda _ { i } .\tag{20}
$$

After normalization, the effective-rank weights become

$$
p _ { i } ^ { + } = \frac { \lambda _ { i } ^ { + } } { \sum _ { j } \lambda _ { j } ^ { + } } = \frac { \lambda _ { i } ( 1 + \eta ^ { 2 } \rho _ { i } ) } { \sum _ { j } \lambda _ { j } ( 1 + \eta ^ { 2 } \rho _ { j } ) } .\tag{21}
$$

If the relative injection $\rho _ { i }$ is larger on the leading eigenspace than on the trailing eigenspace, then probability mass shifts toward already-dominant components, so $\mathfrak { p } ^ { + }$ majorizes p. Since Shannon entropy is Schur-concave, $H ( \mathbf { p } ^ { + } ) < H ( \mathbf { p } )$ whenever the shift is nonzero, and therefore

$$
\mathrm { e r a n k } ( \mathbf { C _ { M ^ { + } } } ) < \mathrm { e r a n k } ( \mathbf { C _ { M } } ) .\tag{22}
$$

Conversely, if $\rho _ { i }$ is larger on trailing or previously underused directions, the normalized spectrum becomes more uniform and

$$
\mathrm { e r a n k } ( \mathbf { C _ { M ^ { + } } } ) > \mathrm { e r a n k } ( \mathbf { C _ { M } } ) .\tag{23}
$$

This establishes that effective-rank changes are controlled by relative spectral injection, not merely by the absolute magnitude of the update covariance.

Cross-layer KV sharing. We now apply the lemma to a source layer � whose KV activations are reused by a non-adjacent layer ℓ. Suppressing head indices and reconstruction coefficients for clarity, the source KV gradients take the form

$$
\nabla _ { \mathbf { K } _ { s } } ^ { \mathrm { s h a r e } } = \frac { 1 } { \sqrt { d } } \left( \mathbf { G } _ { s } ^ { \top } \mathbf { Q } _ { s } + \mathbf { G } _ { \ell } ^ { \top } \mathbf { Q } _ { \ell } \right) , \qquad \nabla _ { \mathbf { V } _ { s } } ^ { \mathrm { s h a r e } } = \mathbf { P } _ { s } ^ { \top } \mathbf { E } _ { s } + \mathbf { P } _ { \ell } ^ { \top } \mathbf { E } _ { \ell } ,\tag{24}
$$

where

$$
\mathbf { G } _ { t } = \frac { \partial \mathcal { L } } { \partial ( \mathbf { Q } _ { t } \mathbf { K } _ { s } ^ { \top } / \sqrt { d } ) } , \qquad \mathbf { E } _ { t } = \frac { \partial \mathcal { L } } { \partial ( \mathbf { P } _ { t } \mathbf { V } _ { s } ) } , \qquad t \in \{ s , \ell \} .\tag{25}
$$

Under a local linearization of the projection update, we assume that the induced activation update covariance follows the covariance of these activation gradients. The key-side covariance contains local and cross-layer terms:

$$
\begin{array} { r l } & { \mathbf { \Sigma } _ { \mathbf { { Z } } } ^ { \mathrm { { s h a r e } } } \propto \mathbb { E } \left[ \mathbf { \mathbf { Q } } _ { s } ^ { \top } \mathbf { \mathbf { G } } _ { s } \mathbf { \mathbf { G } } _ { s } ^ { \top } \mathbf { Q } _ { s } + \mathbf { Q } _ { \ell } ^ { \top } \mathbf { \mathbf { G } } _ { \ell } \mathbf { \mathbf { G } } _ { \ell } ^ { \top } \mathbf { Q } _ { \ell } \right. } \\ & { \qquad \left. + \mathbf { Q } _ { s } ^ { \top } \mathbf { \mathbf { G } } _ { s } \mathbf { \mathbf { G } } _ { \ell } ^ { \top } \mathbf { Q } _ { \ell } + \mathbf { Q } _ { \ell } ^ { \top } \mathbf { \mathbf { G } } _ { \ell } \mathbf { \mathbf { G } } _ { s } ^ { \top } \mathbf { Q } _ { s } \right] . } \end{array}\tag{26}
$$

The value-side covariance similarly expands as

$$
\begin{array} { r l } & { \Sigma _ { V } ^ { \mathrm { s h a r e } } \propto \mathbb { E } \biggl [ \mathbf { E } _ { s } ^ { \top } \mathbf { P } _ { s } \mathbf { P } _ { s } ^ { \top } \mathbf { E } _ { s } + \mathbf { E } _ { \ell } ^ { \top } \mathbf { P } _ { \ell } \mathbf { P } _ { \ell } ^ { \top } \mathbf { E } _ { \ell } } \\ & { \qquad + \mathbf { E } _ { s } ^ { \top } \mathbf { P } _ { s } \mathbf { P } _ { \ell } ^ { \top } \mathbf { E } _ { \ell } + \mathbf { E } _ { \ell } ^ { \top } \mathbf { P } _ { \ell } \mathbf { P } _ { s } ^ { \top } \mathbf { E } _ { s } \biggr ] . } \end{array}\tag{27}
$$

Proposition 1 (asymmetric effective-rank pressure from cross-layer KV sharing). Suppose that (i) cross-layer key gradients share a low-dimensional task-relevant retrieval subspace $\mathcal { U } _ { K }$ aligned with the leading eigenspace of $\mathbf { C } _ { \mathbf { K } _ { s } } ,$ , and their cross-covariance is coherent inside $\mathcal { U } _ { K }$ but weak outside it; and (ii) value readout gradients from the source and reuse layers contain complementary content directions with non-negligible projection onto trailing or underused eigendirections of ${ \mathbf { C } } { \mathbf { v } } _ { s }$ . Then, under the spectral approximation in Eqs. 18–19, cross-layer KV sharing locally induces

$$
\mathrm { e r a n k } ( { \mathbf { C } } _ { \mathbf { K } _ { s } ^ { + } } ) < \mathrm { e r a n k } ( { \mathbf { C } } _ { \mathbf { K } _ { s } } ) , \qquad \mathrm { e r a n k } ( { \mathbf { C } } _ { { \mathbf { V } } _ { s } ^ { + } } ) > \mathrm { e r a n k } ( { \mathbf { C } } _ { { \mathbf { V } } _ { s } } ) .\tag{28}
$$

Proof. For keys, Eq. 26 shows that cross-layer sharing adds covariance terms coupling $\mathbf { G } _ { s } ^ { \top } \mathbf { Q } _ { s }$ and $\mathbf { G } _ { \ell } ^ { \top } \mathbf { Q } _ { \ell }$ . By assumption, these terms reinforce one another inside $\mathcal { U } _ { K } ,$ , producing a positive spectral increment on directions shared by the two query spaces. Outside $\mathcal { U } _ { K }$ , the cross-layer components are weakly correlated, so their contribution to trailing eigenvalues is small relative to the coherent leading injection. Therefore the relative increments satisfy

$$
\frac { \mu _ { i } ^ { K } } { \lambda _ { i } ^ { K } } > \frac { \mu _ { j } ^ { K } } { \lambda _ { j } ^ { K } } , \qquad i \in \mathcal { I } _ { \mathrm { l e a d } } ( \mathbf { K } _ { s } ) , \ j \in \mathcal { I } _ { \mathrm { t r a i l } } ( \mathbf { K } _ { s } ) ,\tag{29}
$$

where $\mu _ { i } ^ { K }$ denotes the key update-covariance eigenvalue in the approximate eigenbasis of ${ \bf C } _ { { \bf K } _ { s } }$ By Lemma 1, the normalized key covariance spectrum becomes more concentrated, yielding

$$
\mathrm { e r a n k } ( \mathbf { C } _ { \mathbf { K } _ { s } ^ { + } } ) < \mathrm { e r a n k } ( \mathbf { C } _ { \mathbf { K } _ { s } } ) .\tag{30}
$$

For values, Eq. 27 shows that the reuse layer routes an additional readout gradient $\mathbf { P } _ { \ell } ^ { \top } \mathbf { E } _ { \ell }$ into the same source value matrix. Unlike keys, values are read out linearly once the attention pattern is fixed; distinct attention patterns can therefore provide complementary content gradients rather than competing through the softmax logits. By assumption, these additional content gradients have non-negligible projection onto underused value directions, so

$$
\frac { \mu _ { i } ^ { V } } { \lambda _ { i } ^ { V } } < \frac { \mu _ { j } ^ { V } } { \lambda _ { j } ^ { V } } , \qquad i \in \mathcal { I } _ { \mathrm { l e a d } } ( \mathbf { V } _ { s } ) , \ j \in \mathcal { I } _ { \mathrm { t r a i l } } ( \mathbf { V } _ { s } ) .\tag{31}
$$

Lemma 1 then implies that the normalized value covariance spectrum becomes more uniform:

$$
\mathrm { e r a n k } ( \mathbf { C } _ { \mathbf { V } _ { s } ^ { + } } ) > \mathrm { e r a n k } ( \mathbf { C } _ { \mathbf { V } _ { s } } ) .\tag{32}
$$

Combining the two parts proves Eq. 28.

Interpretation and scope. The proposition is conditional: it does not claim that effective rank must change monotonically under arbitrary training dynamics. It states that the K/V effectiverank trends follow when the induced update covariance is organized as suggested by the attention mechanism: keys receive coherent reinforcement in shared retrieval directions, while values receive complementary content gradients. The topology of the sharing graph affects whether these assumptions are plausible. Adjacent sharing may make $\mathbf { Q } _ { s } \approx \mathbf { Q } _ { \ell }$ and $\mathbf { P } _ { s } \approx \mathbf { P } _ { \ell } ,$ yielding mostly redundant gradients. Global-anchor sharing can overload a few sources with many heterogeneous objectives. Stride-aligned non-local sharing is more likely to satisfy the assumptions because it provides sufficient cross-layer distance while distributing the gradient load across source layers.

## D. Additional Experimental Results

## D.1. Retention of Gains after Supervised Fine-tuning

We apply supervised fine-tuning (SFT) to the baseline and DepthGraft 16B A3B pretrained models using the same data, learning-rate schedule, and update counts, and evaluate both with the same downstream protocol. Table 9 shows that the pretraining advantage persists after SFT: DepthGraft achieves an average score of 52.00, compared with 48.52 for the baseline. It improves on 11 benchmarks and matches the baseline on LCBench, with notable gains on MBPP (+7.00), HumanEval (+5.49), ARC-C (+4.81), and GSM8K (+2.96). The gains from KV reconstruction therefore carry through supervised adaptation rather than being erased by post-training.

Table 9 | Downstream performance after SFT on the 16B A3B models. Both models use the same fine-tuning recipe.
<table><tr><td>Method</td><td>MMLU</td><td>M-Pro</td><td>LAMB.</td><td>ARC-C</td><td>SIQA</td><td>GSM8K</td><td>HEval</td><td>MBPP</td><td>LCBench</td><td>C-Eval</td><td>CMMLU</td><td>C3</td><td>Avg.</td></tr><tr><td>Baseline + SFT</td><td>53.72</td><td>35.00</td><td>60.77</td><td>67.01</td><td>59.83</td><td>52.08</td><td>37.80</td><td>35.80</td><td>14.96</td><td>49.46</td><td>43.47</td><td>72.33</td><td>48.52</td></tr><tr><td>DepthGraft + SFT</td><td>56.84</td><td>37.57</td><td>62.28</td><td>71.82</td><td>62.54</td><td>55.04</td><td>43.29</td><td>42.80</td><td>14.96</td><td>52.03</td><td>47.28</td><td>77.59</td><td>52.00</td></tr></table>

## D.2. Compatibility with Sliding-Window Attention

DepthGraft changes the source of deep-layer KV, while sliding-window attention (SWA) restricts the positions that each query can attend to. We test their combination on a 12-layer MoE model with 8B total and approximately 1B activated parameters, using a 3:1 ratio of SWA to full-attention layers and a window size of 512. KV reuse is restricted to layers of the same attention type: SWA layers reuse SWA sources, and full-attention layers reuse full-attention sources. This avoids mixing the two types, whose KV-gradient norms differ in our experiments, while preserving prefill early completion because all sources remain in the first two stages.

Table 10 | Compatibility with SWA on an 8B MoE model. The two variants share the same attention-mask pattern and training recipe.
<table><tr><td>Method</td><td>MMLU</td><td>M-Pro</td><td>LAMB.</td><td>ARC-C</td><td>SIQA</td><td>GSM8K</td><td>HEval</td><td>MBPP</td><td>LCBench</td><td>C-Eval</td><td>CMMLU</td><td>C3</td><td>Avg.</td></tr><tr><td>SWA baseline</td><td>37.05</td><td>11.71</td><td>60.26</td><td>21.62</td><td>43.67</td><td>16.38</td><td>21.34</td><td>22.20</td><td>3.13</td><td>34.34</td><td>34.32</td><td>31.45</td><td>28.12</td></tr><tr><td>SWA + DepthGraft</td><td>38.25</td><td>13.57</td><td>61.43</td><td>33.33</td><td>40.31</td><td>19.50</td><td>18.90</td><td>23.00</td><td>2.57</td><td>33.57</td><td>36.54</td><td>35.63</td><td>29.72</td></tr></table>

As shown in Table 10, DepthGraft improves the average score from 28.12 to 29.72, with gains on ARC-C, GSM8K, and C3 among other tasks. These results support combining cross-layer KV reconstruction with local attention, extending its applicability beyond full-attention backbones. An alternative extension is a causal encoder–decoder design inspired by DeepSeek-V4.1-Flash [41], applying KV reconstruction only to full-attention layers while allowing SWA layers to retain their own KV. This gives local representations greater freedom to evolve across layers and could preserve the benefits of prefill early completion through SWA bounded replay over a short suffix of the prompt.

## E. Additional Training Dynamics and Representation Analysis

## E.1. Effective Rank Dynamics: Per-Layer Analysis

Sec. 3.3 argues that global-anchor sharing, such as FusedKV, induces structurally uneven regularization: only the designated anchor layers participate in cross-layer gradient flow, while nearby non-anchor layers receive no direct gradient benefit from KV reuse. We provide per-layer empirical evidence for this effect by tracking the value effective rank erank(Vℓ), defined in Eq. 5, at individual layers throughout training.

Figure 5 shows V effective-rank trajectories for layers 0–3 under the 9B GQA baseline, FusedKV, and DepthGraft. For FusedKV, only layer 0, the bottom anchor, exhibits a clear increase above the baseline. Layers 1–3 do not serve as anchors and therefore receive no direct cross-layer gradient through reused KV tensors; correspondingly, their V effective ranks show no clear improvement over the baseline. This supports the topology analysis in Sec. 3.3: global-anchor sharing confines the representation benefit to a small set of source layers.

In contrast, DepthGraft elevates V effective rank across the shown source layers. Because its stride-aligned source assignment distributes reuse edges across the shallow and middle stages, the gradient-mediated regularization described in Sec. 3.2 is spread over many source layers rather than concentrated on a few anchors. The per-layer effective-rank measurements therefore provide additional evidence that distributed cross-layer connectivity is important for turning KV sharing into an effective representation regularizer.

![](images/4aa034059e245d523f6f363eec90313e3db0b133c76619a5cc686b9aa8fc78c4.jpg)  
(a) Layer 0

![](images/f002d460e4ec054e25c289f50519edf8a8968291dd17c85c0b55d93a9cad8cc7.jpg)  
(b) Layer 1

![](images/3669d86cd4e779b2311cef1123b8595bb4fae8d75b3b4d3edd937caf9390a53f.jpg)  
(c) Layer 2

![](images/38c5fdf9cc9921da8c037817cb5c8c9e1d478b4d868d3320fee233970c4bcb7d.jpg)  
(d) Layer 3  
Figure 5 | Per-layer V effective rank during training for the 9B GQA baseline, FusedKV, and DepthGraft. FusedKV improves the anchor layer but leaves nearby non-anchor layers close to the baseline, whereas DepthGraft yields consistent rank elevation across the shown source layers due to distributed stride-aligned connectivity.

## E.2. Attention Sink Analysis

A common pathology in deep Transformers is the attention sink phenomenon, where a disproportionate amount of attention mass is assigned to the first token regardless of its semantic relevance [39]. In our setting, this behavior provides an additional diagnostic of key geometry: if a layer develops extreme key directions, the softmax distribution can become dominated by a small number of sink positions.

DepthGraft mitigates this tendency through cross-layer gradient regularization. Each shallow or middle source serves its own attention layer while also being reused by a stride-aligned deep layer, so its KV projections receive optimization signals from multiple query spaces. This structure can regularize the source key geometry and make it less likely for a layer to develop sink-prone key directions that dominate the softmax. This is consistent with the analysis in Sec. 3.2, where useful cross-layer KV sharing makes keys form a more compact retrieval geometry.

We verify this empirically by tracking the average attention score assigned to the first token throughout training. Figure 6 reports this metric for both 9B GQA and 9B MLA models. In both cases, DepthGraft consistently lowers first-token attention concentration relative to the baseline, indicating reduced attention-sink behavior.

## E.3. Training Loss Curves

Figure 7 compares the training loss curves of DepthGraft and the corresponding baselines across representative dense and MoE settings. DepthGraft consistently achieves lower training loss across model scales and attention variants, indicating that the proposed cross-layer KV reconstruction does not introduce optimization instability. The magnitude of the loss improvement, however, is moderate: the relative loss reduction is roughly 0.5%.

This behavior is consistent with the role of DepthGraft as a structural regularizer. Unlike crosslayer designs whose primary effect is to substantially alter the residual computation pathway, DepthGraft constrains only the KV memory pathway and even removes independent deep-layer KV projections. Consequently, its training-loss advantage is not as large as that reported for some broader cross-layer connectivity mechanisms, but it yields much larger downstream gains than the loss difference alone would suggest. This supports our interpretation that DepthGraft improves the geometry and generalization of KV representations, rather than merely optimizing the pretraining loss more aggressively.

![](images/d38b479f1f6aad2e12308a25e6fce7f450c3bdbc9181628f39a6c9e5f309efe0.jpg)  
(a) 9B GQA model.

![](images/0257dba432c7a0114c16d8646dedf89d3c88aedb91aa137ff5e470c592a7b5a1.jpg)  
(b) 9B MLA model.  
Figure 6 | Average attention score assigned to the first token over training steps. DepthGraft consistently reduces first-token attention concentration compared with the baseline on both GQA and MLA models.

![](images/c5cb4c7a309420fb12a587cb683b21a722f157bdf830100c23d080fa26b473d4.jpg)  
(a) 1.5B Dense.

![](images/09432fb83a6eda8dc90ff9957e612795456439592ab1b4248fdddedb3182f973.jpg)  
(b) 9B MoE MLA.

![](images/d00911ab124acd2b9bba614b61134e0eeff1ad6fc2445acf453839036c3d531b.jpg)  
(c) 16B MoE GQA.  
Figure 7 | Training loss curves across representative model scales and attention variants. Depth-Graft consistently improves training loss, but the gain is modest relative to its downstream benchmark improvements, supporting its interpretation as a KV-geometry regularizer.

## E.4. Reconstruction Coefficient Dynamics

We further inspect the learned channel-wise reconstruction coefficients. For each reused layer, DepthGraft reconstructs deep-layer KV from a shallow source (src1) and a middle source (src2). Table 11 reports the final coefficient norm normalized by its initialization norm; values larger than 1 indicate that the corresponding source receives increased weight during training.

The learned coefficients reveal a consistent K/V asymmetry. For keys, the middle source is generally upweighted relative to the shallow source, suggesting that deep-layer retrieval benefits more from the more abstract middle-stage key geometry. For values, the shallow source is more strongly upweighted, indicating that lower-level content features remain useful when reconstructing the deep-layer memory content. This qualitative pattern is consistent with the observation behind FusedKV. However, the comparison in Table 5 shows that FusedKV still underperforms DepthGraft under the same 16B A3B MoE setting. This indicates that the benefit does not come only from two-source reconstruction; distributing the source–reuse connections across layers, and thereby spreading the cross-layer gradient regularization over many participating source layers, is also important.

Table 11 | Final reconstruction-coefficient norm relative to initialization. Src1 denotes the shallow source and src2 denotes the middle source. Values larger than 1 indicate increased reliance on the corresponding source after training.
<table><tr><td>Layer</td><td>K src1</td><td>K src2</td><td>V src1</td><td>V src2</td></tr><tr><td>10</td><td>0.6905</td><td>1.1955</td><td>2.1655</td><td>0.7594</td></tr><tr><td>11</td><td>0.4738</td><td>1.1936</td><td>1.6323</td><td>1.4713</td></tr><tr><td>12</td><td>0.5894</td><td>1.2603</td><td>2.4572</td><td>1.2290</td></tr><tr><td>13</td><td>0.7178</td><td>1.0534</td><td>2.6295</td><td>0.8892</td></tr><tr><td>14</td><td>0.7208</td><td>0.9557</td><td>1.8670</td><td>0.6671</td></tr></table>

## F. Efficiency and System Analysis

## F.1. KV Cache and Prefill Cost

DepthGraft partitions the network into shallow, middle, and deep stages of approximately equal depth. The shallow and middle stages materialize KV normally, while the deep stage reconstructs its KV from stride-aligned shallow–middle sources. Therefore, no independent KV cache needs to be stored for the deep stage. For an �-layer model with identical KV dimensionality across layers, the baseline stores KV for all � layers, whereas DepthGraft stores KV only for roughly 2�/3 source layers:

$$
{ \frac { { \mathrm { K V C a c h e } } _ { D e p t h G r a f t } } { { \mathrm { K V C a c h e } } _ { \mathrm { b a s e l i n e } } } } \approx { \frac { 2 L / 3 } { L } } = { \frac { 2 } { 3 } } .\tag{33}
$$

This gives an approximately one-third reduction in KV-cache memory during inference.

The same structure also enables prefill early completion. In standard autoregressive inference, the prefill phase processes the full prompt through all layers to produce the KV cache and the hidden state needed to generate the first response token. In DepthGraft, the purpose of prefill with respect to KV construction is completed once the shallow and middle stages have processed the full prompt, because all KV tensors needed by future decoding steps have already been materialized. The deep stage does not need to process the entire prompt to create additional KV cache. Instead, after the shallow and middle KV caches are available, we only forward the final prompt token through the deep stage, using the stored shallow–middle KV caches to compute attention and produce the first response token.

Thus, the prefill saving comes from avoiding full-sequence computation in the deep stage, while retaining the necessary computation for the last prompt token that initializes decoding. As context length and model size increase, the full-prompt prefill computation becomes increasingly dominant, and the cost of forwarding only the final token through the deep stage becomes a smaller fraction of total prefill cost. This explains the trend in Table 3: the measured TTFT reduction grows at longer context lengths and approaches the theoretical one-third saving.

## F.2. Training FLOPs and Parameter Count

DepthGraft does not introduce additional training FLOPs in the way that many explicit crosslayer connectivity mechanisms do. The shallow and middle stages are computed normally, and the deep-stage KV reconstruction only requires element-wise linear combinations of two cached source tensors. This mixing cost is negligible compared with the matrix multiplications in attention and FFN layers. Moreover, because the deep stage does not require independent K/V projections, an implementation can skip the corresponding K/V projection computation in reused layers, slightly reducing training FLOPs. In our large-scale runs, the observed training throughput is essentially identical to the baseline.

We also reduce the activated parameter count by removing the K and V projection weights from reused layers. For a GQA layer with hidden dimension $d _ { \mathrm { m o d e l } } , n _ { \mathrm { k v } }$ KV heads, and head dimension �ℎ, the K/V projection parameters per layer are

$$
P _ { \mathrm { K V } } ^ { \mathrm { l a y e r } } = 2 d _ { \mathrm { m o d e l } } n _ { \mathrm { k v } } d _ { h } ,\tag{34}
$$

where the factor of 2 accounts for $\mathbf { W } ^ { K }$ and $\pmb { W } ^ { V }$ . If DepthGraft reuses KV in $L _ { \mathrm { r e u s e } }$ layers, the removed activated parameters are

$$
\Delta P _ { \mathrm { a c t i v e } } = L _ { \mathrm { r e u s e } } \cdot 2 d _ { \mathrm { m o d e l } } n _ { \mathrm { k v } } d _ { h } .\tag{35}
$$

For the 16B A3B GQA MoE model, $d _ { \mathrm { m o d e l } } = 2 0 4 8 , n _ { \mathrm { k v } } = 8 , d _ { h } = 1 2 8$ , and $L _ { \mathrm { r e u s e } } { = } 9$ . This gives

$$
\Delta P _ { \mathrm { a c t i v e } } = 9 \times 2 \times 2 0 4 8 \times 8 \times 1 2 8 = 3 7 . 7 \mathrm { M } .\tag{36}
$$

Relative to the 3B activated-parameter budget, this corresponds to approximately 1.3% fewer activated parameters. Thus, DepthGraft obtains its downstream gains while slightly reducing, rather than increasing, the active model capacity.

## F.3. Parallel Training Compatibility

Pipeline parallelism (PP) is essential for training large Transformers whose parameters and activations cannot fit on a single device. PP partitions the model along depth and assigns different layer blocks to different physical pipeline ranks. Virtual pipeline parallelism (VPP) further splits each physical stage into smaller virtual chunks, typically at layer granularity, and interleaves their execution to reduce pipeline bubbles and improve hardware utilization. For cross-layer KV sharing, the main systems concern is whether a reuse layer needs to fetch source KV tensors from a different PP rank; such accesses would introduce additional cross-stage communication and complicate activation lifetime management.

DepthGraft is designed to avoid this issue. Because each deep layer reuses a stride-aligned shallow–middle source pair, the source and reuse layers can be co-located on the same physical PP rank under an appropriate VPP partition. Table 12 gives a balanced example for a 36-layer model with PP size 4. Using zero-based layer indices, the network is divided into three 12-layer stages: layers 0–11, 12–23, and 24–35. Each deep layer � + 24 reconstructs KV from the stridealigned pair (�, � + 12), and each triple $( i , i + 1 2 , i + 2 4 )$ is placed on the same physical PP rank. Thus, cross-layer KV production and consumption remain local to one device.

This layout preserves the natural forward order as three repeated PP passes over shallow, middle, and deep depth bands, while keeping the non-local KV dependency local within each physical rank. With layer-granular VPP chunks, no additional inter-rank KV transfer is required for DepthGraft under this placement. In this balanced setting, every PP rank owns exactly three shallow layers, three middle layers, and three deep reuse layers, so the per-rank layer count is identical. For practical model depths that are not perfectly divisible by the number of PP ranks, the placement may introduce mild imbalance, but the final load remains close. This locality property is important for scaling DepthGraft to large distributed pretraining runs without changing the communication pattern of the baseline PP/VPP system.

Table 12 | A balanced PP/VPP-compatible layer placement for a 36-layer model with PP size 4. Each stride-aligned source–reuse triple is placed on the same physical PP rank.
<table><tr><td>Depth band</td><td>PP rank 0</td><td>PP rank 1</td><td>PP rank 2</td><td>PP rank 3</td></tr><tr><td>Shallow sources</td><td>0,1,2</td><td> $3 , 4 , 5$ </td><td> $6 , 7 , 8$ </td><td>9,10, 11</td></tr><tr><td>Middle sources</td><td>12, 13, 14</td><td>15, 16, 17</td><td>18, 19, 20</td><td>21,22, 23</td></tr><tr><td>Deep reuse</td><td>24, 25,26</td><td>27,28,29</td><td>30, 31, 32</td><td>33,34,35</td></tr></table>