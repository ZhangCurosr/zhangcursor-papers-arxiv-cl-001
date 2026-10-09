# BRIDGING KV-CACHE QUANTIZATION AND LINEAR ATTENTION: FROM THEORY TO PRETRAINED WEIGHT MIGRATION

Kaicheng Xiao<sup>∗</sup>, Liran Dong<sup>∗</sup>, Haotian Li, Guoliang Xing

The Chinese University of Hong Kong

{xk023, dl123, lh023, glxing}@ie.cuhk.edu.hk

## ABSTRACT

KV-cache quantization and linear attention are two representative approaches to tackling the storage and computational costs of Transformers. KV-cache quantization compresses individual KV entries into discrete codes but retains all entries, whereas linear attention recurrently aggregates multiple historical KV contributions into a fixed-size continuous state but can introduce interference. This contrast raises the question of whether per-KV compression and multi-KV aggregation can be bridged within a single mechanism for efficient attention. We identify RAM-Net as such a bridge through soft assignments over a discrete address space. These assignments determine recurrent updates to the continuous slot state associated with each address. Under a restricted RAM-Net construction, we prove that soft address assignments extend hard quantized matching to a separable read-write overlap that locally approximates full-attention similarity and supports recurrent aggregation. These connections further enable Transformer-to-RAM-Net weight migration through a new path based on a soft-quantized intermediate construction. Across nine pretrained Transformer models from 0.3B to 7B parameters, RAM-Net recovers an average of 87.1% of the teachers’ accuracy gains over random guessing across six commonsense and knowledge tasks using only a 500M-token budget per model.

## 1 INTRODUCTION

Transformer models have become the dominant architecture for language modeling (Vaswani et al., 2017; Brown et al., 2020). However, the memory use and per-query computation of full attention grow with context length because all historical keys and values are stored in the KV cache and accessed for each query. Two representative approaches reduce these costs in different ways. KVcache quantization compresses individual KV entries into discrete codes (Liu et al., 2024; Savkin et al., 2025), but retains all entries, so memory still grows with context length. Linear attention instead recurrently aggregates multiple historical KV contributions into a fixed-size continuous state (Katharopoulos et al., 2020; Choromanski et al., 2021), while this aggregation can cause interference among them. This contrast raises the question of whether per-KV compression and multi-KV aggregation can be bridged within a single mechanism for efficient attention. We identify RAM-Net (Xiao et al., 2026) as such a bridge, combining a discrete address space with recurrent continuous slot states. This bridge provides a new view of efficient attention and a new path for weight migration.

We theoretically explain how RAM-Net’s soft address assignments connect discrete-code quantization, full attention, and linear attention within a restricted subset of RAM-Net. These assignments generalize hard code selection in KV quantization, recovering quantized-key matching in the hard write limit while reads remain soft. We further prove that the overlap between soft read and write assignments locally approximates the exponential dot-product similarity of full attention without reconstructing keys from discrete codes. Moreover, the address overlap is exactly separable into read and write terms. This separability enables historical KV contributions to be recurrently aggregated into continuous slot states, yielding the linear-style continuous state construction.

These theoretical connections further enable practical Transformer-to-RAM-Net migration across a broad range of pretrained Transformers. Unlike prior Transformer-to-linear migration methods (Goldstein et al., 2025; Bick et al., 2024), our migration introduces a soft-quantized intermediate attention model that replaces exponential dot-product matching with separable address overlap. To construct the soft-quantized intermediate attention model, we first compress and align the Trans former’s query and key projections to match its attention patterns. We then fold a codebook into the aligned projections to implement soft quantization, while copying the remaining shared parameters from the Transformer. Finally, we initialize RAM-Net with the intermediate model’s parameters and progressively fine-tune it to compensate for approximation errors and architectural differences.

We evaluate the migration design on nine publicly available pretrained Transformer models covering five model families, multi-head attention (MHA) and grouped-query attention (GQA) (Ainslie et al., 2023) configurations, and scales from 0.3B to 7B parameters. With a migration budget of only 500M tokens per model, RAM-Net recovers an average of 87.1% of the teachers’ accuracy gains over random guessing across six commonsense and knowledge tasks. Ablations on the intermediate construction show that the theory-guided design improves attention alignment with the teacher and generally leads to lower output error during head-level recovery. Together with the downstream gains from staged recovery under the same total token budget, these findings support the practical value of our theory-guided migration design.

## 2 BACKGROUND

In sequence modeling, attention uses learnable projections to map an input hidden state $\mathbf { h } _ { t }$ to a query $\mathbf { q } _ { t } ,$ , a key $\mathbf { k } _ { t }$ , and a value $\mathbf { v } _ { t }$ . Under the test-time training (TTT) view (Sun et al., 2025), attention constructs a query-to-value predictor $f _ { t }$ from historical KV samples $\{ ( \mathbf { k } _ { s } , \mathbf { v } _ { s } ) \} _ { s \leq t }$ and evaluates it on the current query as $\mathbf { o } _ { t } = f _ { t } ( \mathbf { q } _ { t } )$ . The predictor $f _ { t }$ is therefore defined over the query-key (QK) matching space.

For causal full attention (Vaswani et al., 2017), its predictor forms a weighted sum of historical values using softmax-normalized dot-product similarities between the query and historical keys:

$$
f _ { t } ( \mathbf { q } ) = \sum _ { s \leq t } \alpha _ { t , s } ( \mathbf { q } ) \mathbf { v } _ { s } , \quad \alpha _ { t , s } ( \mathbf { q } ) = \frac { \exp ( \mathbf { q } ^ { \top } \mathbf { k } _ { s } / \tau ) } { \sum _ { j \leq t } \exp ( \mathbf { q } ^ { \top } \mathbf { k } _ { j } / \tau ) } .\tag{1}
$$

Here $\tau > 0$ is the softmax temperature, and $\alpha _ { t , s } ( \mathbf { q } )$ are the softmax scores. Since softmax assigns larger weights to keys with larger dot-product similarities, full attention can be viewed as a soft nearest-neighbor (Soft-NN) predictor in QK space. This Soft-NN predictor uses the complete set of historical KV samples, so its storage and per-query computation grow with context length. Fig. 1 illustrates how full attention is modified by KV-cache quantization and linear attention, and how RAM-Net connects these two directions, as detailed in the following three subsections.

## 2.1 QUANTIZED KV CACHES: DISCRETE-CODE SOFT-NN PREDICTOR

KV-cache quantization compresses historical KV samples into a finite set of numerical levels or codewords. Quantizing keys partitions QK space into regions whose keys share the same reconstruction. For example, scalar quantization encodes individual coordinates (Liu et al., 2024; Micikevicius et al., 2022), whereas vector quantization jointly encodes coordinate groups (Savkin et al., 2025; Son et al., 2025). Let ${ \mathcal { Q } } _ { \mathrm { K } }$ and $\mathcal { Q } _ { \mathrm { V } }$ denote the quantize-dequantize operators for keys and values. This yields a discrete-code Soft-NN predictor:

$$
f _ { t } ( \mathbf { q } ) = \sum _ { s < t } \alpha _ { t , s } ( \mathbf { q } ) \mathcal { Q } _ { \mathrm { V } } ( \mathbf { v } _ { s } ) , \quad \alpha _ { t , s } ( \mathbf { q } ) = \frac { \exp \left( \mathbf { q } ^ { \top } \mathcal { Q } _ { \mathrm { K } } ( \mathbf { k } _ { s } ) / \tau \right) } { \sum _ { j \leq t } \exp \left( \mathbf { q } ^ { \top } \mathcal { Q } _ { \mathrm { K } } ( \mathbf { k } _ { j } ) / \tau \right) } .\tag{2}
$$

This predictor still stores all historical KV samples, but quantization reduces the storage required for each sample. The resulting matching quality depends on how accurately the reconstructed keys preserve the original similarities between query and keys.

![](images/b3c368576539e6d2d2071a4f2511570041389747e538d4dd666f9e2dbb82d4c3.jpg)

![](images/3284f645fb597e532f966c7a35ba7e86289c6e7f5fc36eb6c8e50452a194447a.jpg)  
Figure 1: Comparison of attention predictors in QK space. For each sub-plot, the left shows how historical KV information is stored or aggregated, while the right visualizes the resulting $f _ { t } ( \mathbf { q } )$ over QK space. RAM-Net retains the discrete-code representation of KV-cache quantization and the continuous-state aggregation of linear attention.

## 2.2 LINEAR ATTENTION: LINEAR PREDICTOR WITH CONTINUOUS FEATURES

Kernelized linear attention replaces the exponential dot-product similarity in Eq. (1) with a separable query-key similarity given by $\phi ( \mathbf { q } ) \phi ( \mathbf { k } ) ^ { \dagger }$ (Katharopoulos et al., 2020; Choromanski et al., 2021). This separable form allows historical KV contributions to be recurrently aggregated into a fixed-size and continuous-feature state $\mathbf { S } _ { t }$ . It yields a linear predictor:

$$
f _ { t } ( \mathbf { q } ) \propto \phi ( \mathbf { q } ) \mathbf { S } _ { t } , \quad \mathbf { S } _ { t } = \Lambda _ { t } \mathbf { S } _ { t - 1 } + \phi ( \mathbf { k } _ { t } ) ^ { \top } \mathbf { v } _ { t } .\tag{3}
$$

Here, $\Lambda _ { t }$ controls how the previous state is retained or modified (Yang et al., 2024; Gu & Dao, 2024; Yang et al., 2025). Maintaining this fixed-size state makes storage and per-query computation independent of context length. However, the fixed state size limits the predictor’s capacity for finegrained matching to historical keys in the QK space.

## 2.3 RAM-NET: DISCRETE-ADDRESS LINEAR-INTERPOLATION PREDICTOR

RAM-Net (Xiao et al., 2026) consists of a finite set of discrete addresses ${ \mathbf { a } } \in { \mathcal { A } } ,$ each indexing a slot that stores a continuous state ${ \bf s } _ { t } ( { \bf a } )$ . Keys and queries produce soft write and read assignments over discrete addresses, respectively, determining which slots are updated and read. The discrete addresses act as codes, and the assignments softly partition QK space. Each slot stores a continuous state that recurrently aggregates historical KV contributions as in linear attention. Specifically, k<sub>t</sub> and $\mathbf { q } _ { t }$ are each split into $U$ groups of address logits, and Product Softmax combines their group-wise softmax probabilities into write and read distributions over addresses. Only the Top-K probabilities are retained for efficiency, which yields sparse write weights $w _ { t } ( \mathbf { a } )$ and read weights $r _ { t } ( \mathbf { a } )$ . Gated Sparse Update (GSU) then recurrently updates the selected slots with $\mathbf { v } _ { t } .$ . The predictor and GSU rule are given by

$$
f _ { t } ( \mathbf { q } _ { t } ) \propto \sum _ { \mathbf { a } \in \mathcal { A } } r _ { t } ( \mathbf { a } ) \mathbf { s } _ { t } ( \mathbf { a } ) , \quad \left\{ \begin{array} { l l } { m _ { t } ( \mathbf { a } ) = \sigma ( \mathrm { l o g i t } ( 1 - w _ { t } ( \mathbf { a } ) ) - \gamma _ { t } ) \cdot m _ { t - 1 } ( \mathbf { a } ) + w _ { t } ( \mathbf { a } ) } \\ { \mathbf { s } _ { t } ( \mathbf { a } ) = \mathbf { s } _ { t - 1 } ( \mathbf { a } ) + \frac { w _ { t } ( \mathbf { a } ) } { m _ { t } ( \mathbf { a } ) + \epsilon } \left( \mathbf { v } _ { t } - \mathbf { s } _ { t - 1 } ( \mathbf { a } ) \right) } \end{array} \right. ,\tag{4}
$$

Here, $f _ { t } ( \mathbf { q } _ { t } )$ is defined by the address overlap of write and read weights, $\gamma _ { t }$ is an input-dependent scalar that modulates the forgetting tendency across all slots, $m _ { t } ( \mathbf { a } )$ is a per-slot scalar that tracks the slot’s inertia against new writes, and ϵ stabilizes the denominator. The read weights linearly combine the continuous states stored at discrete addresses to produce the output, defining a finite discrete-address linear-interpolation predictor. RAM-Net therefore combines the discrete-code representation of KV quantization with the recurrent aggregation of linear attention. Sec. 3 establishes this bridge theoretically, showing how a separable address kernel locally approximates Soft-NN similarity and how GSU extends cumulative averaging with forgetting while retaining linear aggregation.

![](images/db77bbc3fe03375a76837f724f87b10ed68a935f0570d7d32796f03a8bf19df3.jpg)  
Finite-Code Soft-NN Predictor  
Figure 2: Overview of Sec. 3. RAM-Net connects discrete-code Soft-NN and linear predictors through the separable overlap between read and write distributions built on circular soft addressing, which locally approximates Soft-NN matching.

## 3 BRIDGING DISCRETE-CODE SOFT-NN AND CONTINUOUS LINEAR PREDICTORS

This section establishes how the address space of RAM-Net connects discrete-code Soft-NN predictors and continuous-feature linear predictors, as illustrated in Fig. 2. Sec. 3.1 first introduces soft assignment distributions over a finite address space as a discrete-code representation of queries and keys and adopts a restricted circular construction for theoretical analysis. Sec. 3.2 then shows that the overlap between soft read and write distributions locally approximates exponential dotproduct similarity, enabling discrete-code Soft-NN matching in the address space without explicit reconstruction. The address overlap is also separable into read and write terms. Building on this separability, Sec. 3.3 shows how a linear predictor aggregates historical KV contributions into recurrent slot states and how GSU extends the cumulative updates with retention.

## 3.1 DISCRETE-CODE REPRESENTATION VIA CIRCULAR SOFT ADDRESSING

Address space. In RAM-Net, we define the address space as $\mathcal { A } = \{ 0 , \ldots , d _ { p } - 1 \} ^ { U }$ , where each address $\mathbf { a } = ( a ^ { ( 0 ) } , \dots , a ^ { ( U - 1 ) } )$ ) consists of U factors, and each factor $a ^ { ( u ) }$ takes one of $d _ { p }$ possible code indices. Each address $\mathbf { a } \in \mathcal { A }$ indexes a slot. The read and write projections W and $\mathbf { \bar { W } } _ { w }$ each map the hidden state $\mathbf { h } _ { t }$ to $U$ groups of logits, one for each factor. Product Softmax normalizes each group and multiplies the resulting factor-wise probabilities to form a distribution over A.

Restricted projection. For the u-th factor, let $\mathbf { W } _ { r } ^ { ( u ) }$ and $\mathbf { W } _ { w } ^ { ( u ) }$ denote the read and write projection blocks that produce the corresponding logits in Product Softmax. We restrict these projection blocks to the forms $\mathbf { W } _ { r } ^ { ( u ) } = \beta \mathbf { C } ^ { \top } \widetilde { \mathbf { W } } _ { Q } ^ { ( u ) }$ and $\mathbf { W } _ { w } ^ { ( u ) } = \beta \mathbf { C } ^ { \top } \widetilde { \mathbf { W } } _ { K } ^ { ( u ) }$ , respectively. Here, $\mathbf { C } \in \mathbb { R } ^ { 2 \times d _ { p } }$ is a shared and fixed codebook, and the scalar $\beta \geq 0$ controls the concentration of the code assignments. This restricted parameterization defines a special case of RAM-Net in which each read and write projection block has rank at most two. This parameterization introduces two-dimensional intermediate query and key vectors, with $\widetilde { \mathbf { W } } _ { Q } ^ { ( u ) } , \widetilde { \mathbf { W } } _ { K } ^ { ( u ) } \in \mathbb { R } ^ { 2 \times d _ { \mathrm { m o d e l } } }$ and $\mathbf { q } _ { t } ^ { ( u ) } , \mathbf { k } _ { s } ^ { ( u ) } \in \mathbb { R } ^ { 2 } . \widetilde { \mathbf { W } } _ { Q } ^ { ( u ) }$ and ${ \widetilde { \mathbf { W } } } _ { K } ^ { ( u ) }$ map the hidden states to $\mathbf { q } _ { t } ^ { ( u ) } = \widetilde { \mathbf { W } } _ { Q } ^ { ( u ) } \mathbf { h } _ { t }$ and $\mathbf { k } _ { s } ^ { ( u ) } = \widetilde { \mathbf { W } } _ { K } ^ { ( u ) } \mathbf { h } _ { s }$ . This two-dimensional structure also supports an approximate correspondence between RoPE rotations and RAM-Net’s cyclic address shifts (CAPE), as described in Appendix A.6 and used for weight migration in Sec. 4.

Circular code with soft assignments. For our theoretical construction, we choose C as a circular codebook with $d _ { p }$ codes uniformly spaced on the unit circle. Each intermediate vector $\mathbf { q } _ { t } ^ { ( u ) } , \mathbf { k } _ { s } ^ { ( u ) } \in$ $\mathbb { R } ^ { 2 }$ is scored against these shared codes. For any $\mathbf { x } \in \mathbb { R } ^ { 2 }$ , the code vectors and the corresponding soft assignment are defined as

$$
\mathbf { c } _ { j } = \left( \cos \frac { 2 \pi j } { d _ { p } } , \sin \frac { 2 \pi j } { d _ { p } } \right) ^ { \top } , \quad p _ { \beta } ( j | \mathbf { x } ) = \frac { \exp ( \beta \mathbf { c } _ { j } ^ { \top } \mathbf { x } ) } { \sum _ { \ell = 0 } ^ { d _ { p } - 1 } \exp ( \beta \mathbf { c } _ { \ell } ^ { \top } \mathbf { x } ) } , \quad j = 0 , \ldots , d _ { p } - 1 .\tag{5}
$$

The circular codebook is therefore $\mathbf { C } = \left\lceil \mathbf { c } _ { 0 } \quad \cdots \quad \mathbf { c } _ { d _ { p } - 1 } \right\rceil$ , with each scaled codeword $\beta \mathbf { c } _ { j }$ having norm $\beta .$ . Product Softmax combines the assignments from the U factors to define the read and write distributions over A as $\begin{array} { r } { r _ { t } ( \mathbf { a } ) = \prod _ { u = 0 } ^ { U - 1 } p _ { \beta } ( a ^ { ( u ) } \ | \ \mathbf { q } _ { t } ^ { ( u ) } ) } \end{array}$ and $\begin{array} { r } { w _ { s } ( \mathbf { a } ) = \prod _ { u = 0 } ^ { U - 1 } p _ { \beta } ( a ^ { ( u ) } \ | \ \mathbf { k } _ { s } ^ { ( u ) } ) } \end{array}$ . As $\beta  \infty ,$ , each factor-wise assignment concentrates all probability mass on its highest-scoring code. Product Softmax therefore concentrates on the address formed by these factor-wise maximizing code indices. Thus, hard code assignment in discrete-code quantization is a limiting case of our soft address construction, as shown in Appendix A.5.

## 3.2 APPROXIMATION TO SOFT-NN PREDICTORS VIA ADDRESS OVERLAP

The circular soft addressing distribution in Sec. 3.1 provides a discrete-code representation of queries and keys. We write $\mathbf { q } _ { t } = ( \mathbf { q } _ { t } ^ { ( 0 ) } , \dots , \mathbf { q } _ { t } ^ { ( U - 1 ) } )$ and ${ \bf k } _ { s } = ( { \bf k } _ { s } ^ { ( 0 ) } , \ldots , { \bf k } _ { s } ^ { ( U - \bar { 1 } ) } )$ for the 2Udimensional representations. We show how the address distribution overlap can approximate the exponential dot-product similarity exp $\left( \mathbf { q } _ { t } ^ { \top } \mathbf { k } _ { s } / \tau \right)$ used by the Soft-NN predictors of full attention and KV-cache quantization. The following theorem establishes this approximation in a local regime. Theorem 1 (Local approximation of Soft-NN matching). Consider the circular soft-addressing construction in Sec. 3.1. Forfixed U and $d _ { p } \geq 4 ,$ , in the local regime ma $\tau _ { u } \beta ( \| \mathbf { q } _ { t } ^ { ( u ) } \| _ { 2 } + \| \mathbf { k } _ { s } ^ { ( u ) } \| _ { 2 } ) $ 0, the address overlap satisfies

$$
\exp \biggl ( \frac { \beta ^ { 2 } } { 2 } \mathbf { q } _ { t } ^ { \top } \mathbf { k } _ { s } \biggr ) \approx \kappa _ { A } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) = | A | \sum _ { \mathbf { a } \in \mathcal { A } } r _ { t } ( \mathbf { a } ) w _ { s } ( \mathbf { a } ) .\tag{6}
$$

Proofsketch. Define $\begin{array} { r } { Z _ { d _ { p } } ( { \bf x } ) = \frac { 1 } { d _ { p } } \sum _ { j = 0 } ^ { d _ { p } - 1 } \exp ( { \bf c } _ { j } ^ { \top } { \bf x } ) } \end{array}$ . Using the factorized Product Softmax distributions defined in Sec. 3.1, the address overlap can be written as

$$
\kappa _ { \mathcal { A } } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) = \prod _ { u = 0 } ^ { U - 1 } \left[ d _ { p } \sum _ { j = 0 } ^ { d _ { p } - 1 } p _ { \beta } ( j | \mathbf { q } _ { t } ^ { ( u ) } ) p _ { \beta } ( j | \mathbf { k } _ { s } ^ { ( u ) } ) \right] = \prod _ { u = 0 } ^ { U - 1 } \frac { Z _ { d _ { p } } \left( \beta ( \mathbf { q } _ { t } ^ { ( u ) } + \mathbf { k } _ { s } ^ { ( u ) } ) \right) } { Z _ { d _ { p } } ( \beta \mathbf { q } _ { t } ^ { ( u ) } ) Z _ { d _ { p } } ( \beta \mathbf { k } _ { s } ^ { ( u ) } ) } .\tag{7}
$$

As $d _ { p }  \infty ,$ , the uniformly spaced circular codes approach a continuous unit circle, giving $Z _ { d _ { p } } ( \dot { \mathbf { x } } )  I _ { 0 } ( \| \mathbf { x } \| _ { 2 } )$ , where $I _ { 0 }$ is the modified Bessel function of the first kind of order zero. In the local regime, substituting log $I _ { 0 } ( z ) = z ^ { 2 } / 4 + O ( z ^ { 4 } )$ into the logarithm of Eq. (7) in this limit and expanding the squared norms gives

$$
\log \kappa _ { \infty } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) = \operatorname* { l i m } _ { d _ { p } \to \infty } \log \kappa _ { A } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) = \frac { \beta ^ { 2 } } { 2 } \mathbf { q } _ { t } ^ { \top } \mathbf { k } _ { s } + O \left( \beta ^ { 4 } \sum _ { u = 0 } ^ { U - 1 } ( \| \mathbf { q } _ { t } ^ { ( u ) } \| _ { 2 } + \| \mathbf { k } _ { s } ^ { ( u ) } \| _ { 2 } ) ^ { 4 } \right)\tag{8}
$$

Appendix A.1 provides explicit error bounds and shows that, the finite-codebook correction is absorbed by the remainder in Eq. (8). The same expansion therefore holds for log $\kappa _ { \mathcal { A } } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } )$ . Exponentiating yields Eq. (6). □

The scaling factor |A| is introduced for the kernel comparison in the proof and does not affect the normalized attention weights. Intuitively, even after rescaling, the address overlap remains bounded by |A|, whereas the exponential dot-product kernel is unbounded, motivating a local approximation. Matching the local expansion to the target kernel gives $\beta ^ { 2 } / 2 = 1 / \tau$ and hence $\beta ^ { \star } = \sqrt { 2 / \tau }$ . For $\tau = \sqrt { d _ { \mathrm { h e a d } } }$ , this becomes $\beta ^ { \star } = \sqrt { 2 } d _ { \mathrm { h e a d } } ^ { - 1 / 4 }$ We use $\beta ^ { \star }$ to guide weight migration in Sec. 4. Together, the local approximation in Theorem 1 and the separability of the address overlap establish RAM-Net’s connections to continuous Soft-NN, discrete-code Soft-NN, and linear predictors. The local approximation allows RAM-Net to preserve Soft-NN matching in the discrete-code address space without the key reconstruction used in KV-cache quantization. The address overlap is exactly separable into read and write terms, enabling the recurrent linear aggregation developed in Sec. 3.3.

## 3.3 LINEAR PREDICTOR VIA SLOT AGGREGATION

Using the local approximation in Theorem 1 over historical keys and the separable feature form for linear predictors provided by the address overlap in Eq. (6), we aggregate historical KV contributions by address, giving

$$
\frac { \sum _ { s \leq t } \exp \left( \beta ^ { 2 } / 2 \cdot \mathbf { q } _ { t } ^ { \top } \mathbf { k } _ { s } \right) \mathbf { v } _ { s } } { \sum _ { s \leq t } \exp \left( \beta ^ { 2 } / 2 \cdot \mathbf { q } _ { t } ^ { \top } \mathbf { k } _ { s } \right) } \approx \frac { \sum _ { s \leq t } \kappa _ { \mathcal { A } } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) \mathbf { v } _ { s } } { \sum _ { s \leq t } \kappa _ { \mathcal { A } } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) } = \frac { \sum _ { \mathbf { a } } r _ { t } ( \mathbf { a } ) m _ { t } ( \mathbf { a } ) \mathbf { s } _ { t } ( \mathbf { a } ) } { \sum _ { \mathbf { a } } r _ { t } ( \mathbf { a } ) m _ { t } ( \mathbf { a } ) } .\tag{9}
$$

Here, $\begin{array} { r } { m _ { t } ( \mathbf { a } ) = \sum _ { s < t } w _ { s } ( \mathbf { a } ) } \end{array}$ is the cumulative write mass of slot a, and $\begin{array} { r } { \mathbf { s } _ { t } ( \mathbf { a } ) = \frac { \sum _ { s \leq t } w _ { s } ( \mathbf { a } ) \mathbf { v } _ { s } } { m _ { t } ( \mathbf { a } ) } } \end{array}$ is its cumulative weighted mean of values, detail in Appendix A.2. The rightmost expression defines a mass-aware readout whose read coefficients are proportional to $r _ { t } ( \mathbf { a } ) m _ { t } ( \mathbf { a } )$ , so the contribution of each slot depends on both its query read score and accumulated write mass.

The GSU rule in Eq. (4) extends this cumulative form by introducing $\gamma _ { t }$ to control the retention of historical contributions. With $m _ { 0 } ( { \bf a } ) = 0 , { \bf s } _ { 0 } ( { \bf a } ) = { \bf 0 }$ , and $\epsilon = 0$ , taking $\gamma _ { t } \to - \infty$ at every step removes this forgetting effect and reduces GSU to cumulative averaging:

$$
m _ { t } ( \mathbf { a } ) = m _ { t - 1 } ( \mathbf { a } ) + w _ { t } ( \mathbf { a } ) , \qquad \mathbf { s } _ { t } ( \mathbf { a } ) = { \frac { m _ { t - 1 } ( \mathbf { a } ) \mathbf { s } _ { t - 1 } ( \mathbf { a } ) + w _ { t } ( \mathbf { a } ) \mathbf { v } _ { t } } { m _ { t } ( \mathbf { a } ) } } .\tag{10}
$$

For finite $\gamma _ { t } .$ , GSU retains its forgetting mechanism, where $m _ { t }$ and $\mathbf { s } _ { t }$ track the retained write mass and its corresponding weighted value mean. The resulting predictor remains linear in the read address features, while GSU updates the slot states online with controlled retention.

Original RAM-Net reads these slot states using $\begin{array} { r } { \sum _ { \mathbf { a } } r _ { t } ( \mathbf { a } ) \mathbf { s } _ { t } ( \mathbf { a } ) } \end{array}$ , omitting the write-mass reweighting. The mass-aware readout ablation in Appendix B.3(Tables 7 and 8) shows that the mass-aware readout generally matches the Transformer’s attention more closely before head-level recovery, but tends to yield slightly higher perplexity after migration than the original readout. It may be related to the coupling of read weights to forgetting dynamics and a potential mismatch between mass reweighting and Top-K selection based only on $r _ { t } ( \mathbf { a } )$

## 4 TRANSFORMER-TO-RAM-NET WEIGHT MIGRATION

The connections established in Sec. 3 provide a basis for migrating pretrained Transformer weights to RAM-Net. The migration proceeds in two stages. First, we adapt the teacher’s query-key matching to soft-quantized intermediate attention model, aiming to preserve its token-to-token attention patterns. Second, we enable RAM-Net’s recurrent updates and progressively fine-tune the model to reduce the remaining output discrepancies.

## 4.1 MIGRATION TO SOFT-QUANTIZED INTERMEDIATE ATTENTION

Soft-quantized intermediate attention uses the overlap of circular read and write address assignments in place of exponential dot-product similarity (Secs. 3.1 and 3.2), allowing us to align token-totoken matching before GSU. We construct it by performing low-rank and positional alignment of the teacher’s query and key projections, then folding them into read and write projections.

Low-rank and positional alignment. The circular construction in Sec. 3.1 uses U twodimensional factors. Each two-dimensional factor can serve as a RoPE rotation plane, with rotations at frequencies $\omega _ { u } = 2 \pi / d _ { p } ^ { u + 1 }$ for $u = 0 , \ldots , U - 1$ approximately corresponding to CAPE’s cyclic address shifts (Appendix A.6). Since the address count $d _ { p } ^ { U }$ grows exponentially with U, we compress each teacher head’s query and key vectors from their original dimension $d _ { k }$ to 2U, motivated by evidence that these representations admit low-rank approximations (Saxena et al., 2024). We introduce adapters $\mathbf { P } _ { Q } , \dot { \mathbf { P } _ { K } } \in \mathbb { R } ^ { 2 U \times d _ { k } }$ to implement this reduction, aiming to preserve positiondependent token matching. For each head, we consider both RoPE with the target frequencies $\omega _ { u }$ and no positional encoding (NoPE). Using the Fourier representation of rotary matching (Hua et al., 2025), we obtain closed-form adapter initializations from calibration statistics through frequency projection and low-rank factorization (Appendix A.7). After refining the RoPE and NoPE candidates against the teacher’s attention maps using forward Kullback-Leibler divergence (KL), we retain RAM-Net’s original CAPE pattern across layers and assign heads to CAPE or NoPE based on their relative alignment errors.

Circular soft quantization. After selecting the adapters and positional modes, we map the aligned query and key representations to soft circular address assignments. Let $\widetilde { \mathbf { C } } _ { U } = \mathrm { D i a g } ( \mathbf { C } , \dots , \mathbf { C } )$ denote the block-diagonal expansion of the circular codebook with U copies. Given the teacher projections $\mathbf { W } _ { Q } ^ { t }$ and $\mathbf { W } _ { K } ^ { t }$ , we combine low-rank alignment, the circular construction, and temperature matching to obtain the student projections $\mathbf { W } _ { r } ^ { s } = \beta ^ { \star } \widetilde { \mathbf { C } } _ { U } ^ { \top } \mathbf { P } _ { Q } \mathbf { W } _ { Q } ^ { t }$ and $\mathbf { W } _ { w } ^ { s } = \beta ^ { \star } \widetilde { \mathbf { C } } _ { U } ^ { \top } \mathbf { P } _ { K } \mathbf { W } _ { K } ^ { t }$ , where $\beta ^ { \star }$ is the assignment concentration defined in Sec. 3.2 with $\tau = { \sqrt { 2 U } }$ . Product Softmax then yields address assignments whose overlap defines the intermediate attention similarity.

## 4.2 RAM-NET CONVERSION AND PROGRESSIVE RECOVERY

Recovery proceeds at the head, attention-block, and full-model levels. We convert the intermediate model in Sec. 4.1 to RAM-Net and optimize its RAM-Net-specific parameters to compensate for output differences introduced by Top-K slot selection and GSU. Notably, we optimize the read and write projection matrices W<sup>s</sup> and $\mathbf { W } _ { w } ^ { s }$ without the low-rank and shared-codebook constraints used in Sec. 4.1. For head-level recovery, we adapt address routing and GSU updates to match the teacher’s per-head outputs. For attention-block recovery, we train the attention branch to match the teacher’s residual updates. Both stages use normalized mean squared error (NMSE), computed on head outputs before output projection and on attention-branch outputs before residual addition, respectively. For full-model recovery, we unfreeze and fine-tune all student parameters using nexttoken cross-entropy as the main objective. Appendix B.2 provides the complete training details.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETUP

We evaluate Transformer-to-RAM-Net weight migration on pretrained models ranging from 340M to 7B parameters. We use nine publicly available pretrained teacher models, whose architectures are summarized in Table 5. Each RAM-Net matches its teacher’s architecture and tokenizer, except for the attention mechanism. Unless otherwise specified, we use $d _ { p } = 4 , U = 5 .$ , and $\mathrm { T o p } { \cdot } K = 3 2$ For GQA (Ainslie et al., 2023), each KV head is repeated over its query-head group to form the equivalent multi-head attention (MHA) weights used for migration. All migration experiments use the same total budget of 500M tokens, with per-stage token allocations, learning rates, and other training details provided in Table 6 and Appendix B.2. The tokens come from the 10BT sample of the FineWeb-Edu dataset HuggingFaceFW (2024). Full-model recovery uses two NVIDIA H200 GPUs with FSDP2, while all other stages use a single GPU.

## 5.2 LANGUAGE MODEL EVALUATION BENCHMARKS

Table 1 reports language modeling performance and accuracy on six common-knowledge tasks. Across the nine teachers, the Wiki PPL gap between RAM-Net and its teacher ranges from 1.22 to 5.09, while the average-accuracy gap across the six tasks ranges from 1.8 to 4.5 percentage points. We use the relative score $\begin{array} { r } { R = \bar { \frac { s - r } { t - r } } } \end{array}$ to avoid overstating retention when both models perform near chance; s, t, and r denote RAM-Net, teacher, and chance accuracies. Across teachers, R ranges from 79.2% to 95.0%, with a mean of 87.1%. These results show that the migration procedure retains most of the teachers’ performance on these benchmarks across attention types, model families, and scales. We also report the mass-aware readout variant (Sec. 3.3) in Appendix B.3, which shows similar overall trends. Table 3 and Table 10 in the appendix further show that RAM-Net partially recovers teacher performance on code completion and summarization tasks from LongBench.

## 5.3 ABLATION STUDIES

Low-rank alignment and circular codebook. We ablate the low-rank alignment and circular codebook in Sec. 4.1, measuring the Jensen–Shannon divergence (JS) between teacher and RAM-Net attention distributions after intermediate attention construction and before head-level recovery. The r/w projection and kq projection adapter omit learned low-rank alignment, learning projections from scratch and transferring teacher QK weights through random adapters, respectively. The codebook baselines retain learned low-rank alignment to isolate the effects of codebook design. The random codebook tests the circular construction against shared random codes. Inspired by FAVOR+ (Choromanski et al., 2021), the orthogonal random codebook replaces circular codes with randomly scaled two-dimensional directions obtained by QR decomposition of Gaussian matrices. We also compare learned codebooks (Alonso et al., 2026; van den Dool et al., 2026) with the fixed circular construction: shared k-means learns one shared $d _ { p } .$ -centroid codebook from calibration using query and key across factors, while per-factor $P Q$ learns one per factor (Jegou et al., 2011). Table 2 shows that´ the circular codebook yields lower JS than all alternatives across the nine teachers. In the appendix, Fig. 7 further shows that the circular codebook generally yields lower head NMSE during recovery; Fig. 6 and Table 11 evaluate the effects of varying $d _ { p }$ and $U .$

Table 1: Zero-shot language modeling and commonsense reasoning performance of Teacher-to-RAM-Net migration.
<table><tr><td>Family</td><td>Teacher</td><td>Model</td><td>Tokens Ctx.</td><td>Wiki PPL</td><td>ARC-E↑</td><td></td><td>ARC-C ↑ PIQA ↑ SciQ ↑</td><td></td><td></td><td>COPA↑</td><td></td><td>Hella. ↑ Avg. ↑ Avg. R ↑</td><td></td></tr><tr><td rowspan="6">FLA</td><td rowspan="2">transformer-340M-Teacher 10B</td><td></td><td>10B 0.5B</td><td>4K 2K</td><td>30.05 35.14</td><td>57.4 54.7</td><td>24.1 23.7</td><td>65.6 64.7</td><td>82.5 80.2</td><td>73.0 65.0</td><td>32.1 31.0</td><td>55.8 53.2</td><td rowspan="2">86.5</td></tr><tr><td>RAM-Net</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">transformer-1.3B- 100B</td><td>Teacher</td><td>100B</td><td>2K</td><td>17.66</td><td>56.1</td><td>23.9</td><td>70.1</td><td>85.3</td><td>75.0</td><td>38.5</td><td>58.1</td><td rowspan="2">86.4</td></tr><tr><td>RAM-Net</td><td>0.5B</td><td>2K</td><td>21.93</td><td>52.7</td><td>24.8</td><td>68.3</td><td>78.8</td><td>71.0</td><td>35.6</td><td>55.2</td></tr><tr><td rowspan="2">transformer-2.7B- 100B</td><td>Teacher</td><td>100B</td><td>2K</td><td>15.23</td><td>59.4</td><td>26.5</td><td>71.9</td><td>87.9</td><td>75.0</td><td>42.0</td><td>60.4</td><td rowspan="2">89.2</td></tr><tr><td>RAM-Net</td><td>0.5B</td><td>2K</td><td>19.82</td><td>56.5</td><td>24.6</td><td>70.3</td><td>79.9</td><td>74.0</td><td>38.3</td><td>57.3</td></tr><tr><td rowspan="5">Qwen2.5</td><td rowspan="2">0.5B-Base</td><td>Teacher</td><td>一</td><td>32K</td><td>18.46</td><td>64.4</td><td>29.3</td><td>70.6</td><td>93.1</td><td>74.0</td><td>40.7</td><td>62.0</td><td rowspan="2">79.2</td></tr><tr><td>RAM-Net</td><td>0.5B</td><td>4K</td><td>23.35</td><td>63.6</td><td>26.4</td><td>68.5</td><td>81.9</td><td>73.0</td><td>36.9</td><td>58.4</td></tr><tr><td rowspan="2">1.5B-Base</td><td>Teacher</td><td>1</td><td>32K</td><td>12.60</td><td>75.2</td><td>41.1</td><td>75.6</td><td>94.2</td><td>83.0</td><td>50.2</td><td>69.9</td><td rowspan="2">95.0</td></tr><tr><td>RAM-Net</td><td>0.5B</td><td>4K</td><td>16.33</td><td>75.3</td><td>41.5</td><td>74.3</td><td>89.8</td><td>82.0</td><td>45.6</td><td>68.1</td></tr><tr><td rowspan="2">3B-Base</td><td>Teacher</td><td></td><td>32K</td><td>12.41</td><td>77.6</td><td>45.0</td><td>78.6</td><td>96.1</td><td>86.0</td><td>55.0</td><td>73.0</td><td rowspan="2">91.5</td></tr><tr><td>RAM-Net</td><td>0.5B</td><td>4K</td><td>13.63</td><td>77.2</td><td>43.7</td><td>77.0</td><td>92.9</td><td>79.0</td><td>50.7</td><td>70.1</td></tr><tr><td rowspan="2"></td><td rowspan="2">RedPajama INCITE-Base-3B-v1</td><td>Teacher</td><td>800B</td><td>2K</td><td>11.76</td><td>68.0</td><td>31.7</td><td>74.8</td><td>90.4</td><td>77.0</td><td>47.6</td><td>64.9</td><td rowspan="2">79.7</td></tr><tr><td>RAM-Net</td><td>0.5B</td><td>2K</td><td>16.65</td><td>63.6</td><td>27.7</td><td>72.2</td><td>80.6</td><td>77.0</td><td>41.9</td><td>60.5</td></tr><tr><td rowspan="2">Mistral</td><td rowspan="2">7B-v0.1</td><td>Teacher</td><td>1</td><td>8K</td><td>8.22</td><td>80.3</td><td>50.9</td><td>80.9</td><td>96.4</td><td>92.0</td><td>61.5</td><td>77.0</td><td rowspan="2">89.4</td></tr><tr><td>RAM-Net</td><td>0.5B</td><td>4K</td><td>10.14</td><td>77.2</td><td>44.1</td><td>79.7</td><td>91.5</td><td>87.0</td><td>58.0</td><td>72.9</td></tr><tr><td rowspan="2">LLaMA</td><td rowspan="2">2-7B-hf</td><td>Teacher</td><td>2T</td><td>4K</td><td>8.76</td><td>75.3</td><td>42.7</td><td>77.7</td><td>94.8</td><td>88.0</td><td>57.1</td><td>72.6</td><td rowspan="2">87.2</td></tr><tr><td>RAM-Net</td><td>0.5B</td><td>4K</td><td>11.84</td><td>71.6</td><td>39.0</td><td>76.2</td><td>89.2</td><td>81.0</td><td>51.6</td><td>68.1</td></tr></table>

Avg. is the arithmetic mean of the six task accuracies. Relative score Avg. R gives Avg. $( ( s - r ) / ( t - r ) )$ over tasks for which t $> r$ with 95% confidence.

Code assignment concentration $\beta .$ The concentration $\beta$ controls how the address overlap approximates exponential dot-product similarity. We sweep $\beta / \beta ^ { \star }$ , where $\beta ^ { \star }$ is derived in Sec. 3.2. For each teacher, all sweep configurations use the same low-rank adapters. We report normalized JS (Fig. 3) and normalized head NMSE (Fig. 5 in the appendix) before head-level recovery, each normalized by its value at $\beta ^ { \star }$ for the same teacher. Across teachers, $\beta ^ { \star }$ yields near-minimal JS and head NMSE, although some minima occur around 1.58β<sup>⋆</sup>. For well-aligned querykey pairs, higher-order terms can reduce the overlap relative to the local approximation, so a slightly larger β may partly compensate by sharpening code assignments.

Migration pipeline ablation. We assess the value of staged recovery in Sec. 4 on Qwen2.5-1.5B under a fixed 500M-token budget (Table 4). The complete pipeline achieves lower Wiki PPL and higher downstream accuracy than full-model only, which allocates the entire budget to full-model recovery. Removing either headlevel recovery or attention-block recovery lowers downstream accuracy, even when its tokens are reallocated to full-model recovery. To assess the preset CAPE pattern, pattern-free CAPE selection determines CAPE head counts solely from relative RoPE alignment errors. It slightly lowers logit KL and Wiki PPL but also lowers downstream accuracy relative to the preset pattern. Appendix Table 12 and Fig. 8 provide intermediate results and stage-wise errors, respectively.

![](images/baeedd80a590da5b590686fa1cb273d75a296270096ddb8f945d9388a97dd012.jpg)  
Figure 3: Attn. JS under different circu lar assignment concentrations $\beta .$

Table 2: Attention JS divergence of intermediate attention construction.
<table><tr><td>Method</td><td>FLA-0.3B</td><td>FLA-1.3B</td><td>FLA-2.7B</td><td>Qwen-0.5B</td><td>Qwen-1.5B</td><td>Qwen-3B</td><td>Red.-3B</td><td>Mist.-7B</td><td>LLaMA-7B</td></tr><tr><td>r/w projection</td><td>0.561</td><td>0.563</td><td>0.568</td><td>0.625</td><td>0.617</td><td>0.601</td><td>0.611</td><td>0.619</td><td>0.629</td></tr><tr><td>QK projection adapter</td><td>0.570</td><td>0.558</td><td>0.571</td><td>0.627</td><td>0.629</td><td>0.623</td><td>0.601</td><td>0.615</td><td>0.615</td></tr><tr><td>Random codebook</td><td>0.441</td><td>0.440</td><td>0.437</td><td>0.486</td><td>0.476</td><td>0.498</td><td>0.443</td><td>0.430</td><td>0.389</td></tr><tr><td>Orth. Rand. codebook</td><td>0.487</td><td>0.515</td><td>0.525</td><td>0.574</td><td>0.566</td><td>0.560</td><td>0.563</td><td>0.568</td><td>0.586</td></tr><tr><td>Shared k-means codebook</td><td>0.415</td><td>0.412</td><td>0.419</td><td>0.466</td><td>0.453</td><td>0.481</td><td>0.421</td><td>0.411</td><td>0.377</td></tr><tr><td>Per-factor PQ codebook</td><td>0.412</td><td>0.423</td><td>0.431</td><td>0.487</td><td>0.476</td><td>0.487</td><td>0.446</td><td>0.445</td><td>0.407</td></tr><tr><td>Circular codebook</td><td>0.398</td><td>0.394</td><td>0.402</td><td>0.452</td><td>0.444</td><td>0.470</td><td>0.412</td><td>0.389</td><td>0.359</td></tr></table>

Table 3: LongBench results.
<table><tr><td>Model</td><td>Method</td><td>LCC</td><td>MultiN.</td><td>SAMSum</td><td>Avg. ↑</td></tr><tr><td rowspan="2">FLA-1.3B</td><td>Teacher</td><td>48.3</td><td>6.6</td><td>6.2</td><td>20.3</td></tr><tr><td>RAM-Net</td><td>36.4</td><td>7.2</td><td>9.3</td><td>17.6</td></tr><tr><td rowspan="2">Qwen-3B</td><td>Teacher</td><td>14.1</td><td>24.5</td><td>45.5</td><td>28.0</td></tr><tr><td>RAM-Net</td><td>15.5</td><td>15.9</td><td>29.5</td><td>20.3</td></tr><tr><td rowspan="2">Mistral-7B</td><td>Teacher</td><td>70.0</td><td>15.5</td><td>28.2</td><td>37.9</td></tr><tr><td>RAM-Net</td><td>33.4</td><td>12.1</td><td>22.1</td><td>22.5</td></tr></table>

Table 4: Pipeline ablations on Qwen2.5-1.5B.
<table><tr><td>Method</td><td>Logit KL ↓</td><td>Wiki PPL ↓</td><td>Avg. ↑</td><td>R↑</td></tr><tr><td>Teacher</td><td>一</td><td>12.60</td><td>69.9</td><td></td></tr><tr><td>Complete</td><td>0.164</td><td>16.33</td><td>68.1</td><td>95.0</td></tr><tr><td>Pattern-free CAPE select.</td><td>0.162</td><td>16.17</td><td>67.6</td><td>93.7</td></tr><tr><td>w/o Head Rec.</td><td>0.166</td><td>16.31</td><td>66.9</td><td>90.8</td></tr><tr><td>w/o Attn.-Block Rec.</td><td>0.181</td><td>16.64</td><td>66.9</td><td>90.2</td></tr><tr><td>Full-Model only</td><td>0.197</td><td>16.82</td><td>66.6</td><td>89.7</td></tr></table>

## 6 RELATED WORK

## 6.1 LINEAR-TIME ATTENTION VIA QUANTIZATION AND KERNEL APPROXIMATION

Vector-quantized attention reduces attention cost by replacing individual keys with a finite codebook. Transformer-VQ (Lingle, 2024) uses quantized keys with recurrent per-code statistics, whose connection to our framework through hard writes and soft reads is derived in Appendix A.5. AVQ-Attention (van den Dool et al., 2026) adaptively refines the codebook according to attention importance, while online VQ attention (Alonso et al., 2026) uses sparse online updates to maintain a larger fixed-size memory. Kernelized attention instead uses separable features to approximate softmax matching. Performer/FAVOR+ (Choromanski et al., 2021) approximates the exponential dot-product kernel with positive orthogonal random features. Our construction connects these two views by using discrete-code soft assignments whose separable read and write overlap locally approximates exponential dot-product similarity and supports recurrent aggregation.

## 6.2 TRANSFORMER-TO-LINEAR WEIGHT MIGRATION

Prior work has migrated pretrained Transformers to various linear or recurrent architectures, including recurrent attention in SUPRA (Mercat et al., 2024), kernelized linear attention in Hedgehog (Zhang et al., 2024) and LoLCATs (Zhang et al., 2025), Mamba models in MOHAWK (Bick et al., 2024) and Llamba (Bick et al., 2025), RWKV variants in RADLADS (Goldstein et al., 2025), gated recurrent models in Liger (Lan et al., 2025), and linearization for Gated DeltaNet (Kuzina et al., 2026). These methods generally convert directly to the target architectures. Attention-to-Mamba (Moudgil et al., 2026) instead uses kernelized linear attention as an intermediate before conversion to Mamba. Unlike these Transformer-to-linear methods, our migration proposes a distinct path by introducing an intermediate construction derived from quantized Transformers.

## 7 CONCLUSION AND LIMITATIONS

In this paper, we establish RAM-Net as a theoretical bridge between discrete-code quantization and linear attention, and use this connection to enable efficient Transformer-to-RAM-Net weight migration. One limitation is that our derivation is local and assumes a fixed circular codebook with projection blocks of rank at most two, while practical RAM-Net uses unrestricted projections. Our migration experiments are also conducted under a fixed token budget without further tuning. We leave these limitations for future work.

## REFERENCES

Joshua Ainslie, James Lee-Thorp, Michiel De Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit´ Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. In Proceedings of the 2023 conference on empirical methods in natural language processing, pp. 4895–4901, 2023.

Nick Alonso, Tomas Figliolia, and Beren Millidge. Online vector quantized attention. arXiv preprint arXiv:2602.03922, 2026. URL https://arxiv.org/abs/2602.03922.

Aviv Bick, Kevin Y. Li, Eric P. Xing, J. Zico Kolter, and Albert Gu. Transformers to SSMs: Distilling quadratic knowledge to subquadratic models. arXiv preprint arXiv:2408.10189, 2024. URL https://arxiv.org/abs/2408.10189.

Aviv Bick, Tobias Katsch, Nimit Sohoni, Arjun Desai, and Albert Gu. Llamba: Scaling distilled recurrent models for efficient language processing. arXiv preprint arXiv:2502.14458, 2025. URL https://arxiv.org/abs/2502.14458.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html.

Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, and Adrian Weller. Rethinking attention with performers. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id= Ua6zuk0WRH.

Carl Eckart and Gale Young. The approximation of one matrix by another of lower rank. Psychometrika, 1(3):211–218, 1936. doi: 10.1007/BF02288367.

FLA Hub. FLA Hub Models. https://huggingface.co/fla-hub/models. Accessed: 2026-09-26.

Daniel Goldstein, Eric Alcaide, Janna Lu, and Eugene Cheah. RADLADS: Rapid attention distillation to linear attention decoders at scale. arXiv preprint arXiv:2505.03005, 2025. URL https://arxiv.org/abs/2505.03005.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id= tEYskw1VY2.

Ermo Hua, Che Jiang, Xingtai Lv, Kaiyan Zhang, Youbang Sun, Yuchen Fan, Xuekai Zhu, Biqing Qi, Ning Ding, and Bowen Zhou. Fourier position embedding: Enhancing attention’s periodic extension for length generalization. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 24932–24949. PMLR, 2025. URL https://proceedings.mlr.press/v267/hua25b.html.

HuggingFaceFW. fineweb-edu (revision 22b0aca), 2024. URL https://huggingface.co/ datasets/HuggingFaceFW/fineweb-edu.

Herve J´ egou, Matthijs Douze, and Cordelia Schmid. Product quantization for nearest neighbor´ search. IEEE Transactions on Pattern Analysis and Machine Intelligence, 33(1):117–128, 2011. doi: 10.1109/TPAMI.2010.57.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lelio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril,´ Thomas Wang, Timothee Lacroix, and William El Sayed. Mistral 7b, 2023. URL´ https: //arxiv.org/abs/2310.06825.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc¸ois Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pp. 5156–5165. PMLR, 2020.

Anna Kuzina, Paul N. Whatmough, and Babak Ehteshami Bejnordi. The key to going linear: Analysis-driven transformer linearization. arXiv preprint arXiv:2607.07706, 2026. URL https://arxiv.org/abs/2607.07706.

Disen Lan, Weigao Sun, Jiaxi Hu, Jusen Du, and Yu Cheng. Liger: Linearizing large language models to gated recurrent structures. arXiv preprint arXiv:2503.01496, 2025. URL https: //arxiv.org/abs/2503.01496.

Lucas D. Lingle. Transformer-VQ: Linear-time transformers via vector quantization. In International Conference on Learning Representations, 2024. URL https://openreview.net/ forum?id=oDdzXQzP2F.

Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. KIVI: A tuning-free asymmetric 2bit quantization for KV cache. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 32332–32344, 2024. URL https://proceedings.mlr. press/v235/liu24bz.html.

Jean Mercat, Igor Vasiljevic, Sedrick Keh, Kushal Arora, Achal Dave, Adrien Gaidon, and Thomas Kollar. Linearizing large language models. arXiv preprint arXiv:2405.06640, 2024. URL https://arxiv.org/abs/2405.06640.

Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey, Richard Grisenthwaite, Sangwon Ha, Alexander Heinecke, Patrick Judd, John Kamalu, Naveen Mellempudi, Stuart Oberman, Mohammad Shoeybi, Michael Siu, and Hao Wu. FP8 formats for deep learning. arXiv preprint arXiv:2209.05433, 2022. doi: 10.48550/arXiv.2209.05433. URL https://arxiv.org/abs/2209.05433.

L. Mirsky. Symmetric gauge functions and unitarily invariant norms. The Quarterly Journal of Mathematics, 11(1):50–59, 1960. doi: 10.1093/qmath/11.1.50.

Abhinav Moudgil, Ningyuan Huang, Eeshan Gunesh Dhekane, Pau Rodr´ıguez, Luca Zappella, and Federico Danieli. Attention to mamba: A recipe for cross-architecture distillation. arXiv preprint arXiv:2604.14191, 2026.

R. Penrose. On best approximate solutions of linear matrix equations. Mathematical Proceedings ofthe Cambridge Philosophical Society, 52(1):17–19, 1956. doi: 10.1017/S0305004100030929.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.

Semyon Savkin, Eitan Porat, Or Ordentlich, and Yury Polyanskiy. NestQuant: Nested lattice quantization for matrix products and LLMs. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 53042–53062. PMLR, 2025. URL https://proceedings.mlr.press/v267/savkin25a.html.

Utkarsh Saxena, Gobinda Saha, Sakshi Choudhary, and Kaushik Roy. Eigen attention: Attention in low-rank space for KV cache compression. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 15332–15344. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.findings-emnlp.899.

Donghyun Son, Euntae Choi, and Sungjoo Yoo. NSNQuant: A double normalization approach for calibration-free low-bit vector quantization of KV cache. In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/085713-1436. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 3d8ee933c215fbb7b4d1948ff4906299-Abstract-Conference.html.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864, 2021. URL https://arxiv.org/abs/2104.09864.

Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, and Carlos Guestrin. Learning to (Learn at test time): RNNs with expressive hidden states. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 57503–57522. PMLR, 2025. URL https://proceedings.mlr.press/ v267/sun25h.html.

Together Computer. RedPajama-INCITE-Base-3B-v1. https://huggingface.co/ togethercomputer/RedPajama-INCITE-Base-3B-v1, 2023. Hugging Face model repository.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023. URL https://arxiv.org/abs/2307.09288.

Winfried van den Dool, Patrick Forre, Amir Habibian, Yuki M. Asano, and Max Welling. AVQ-´ attention: Adaptive vector-quantized attention. arXiv preprint arXiv:2607.12789, 2026. URL https://arxiv.org/abs/2607.12789.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Kaicheng Xiao, Haotian Li, Liran Dong, and Guoliang Xing. RAM-Net: Expressive linear attention with selectively addressable memory. arXiv preprint arXiv:2602.11958, 2026. URL https: //arxiv.org/abs/2602.11958.

Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention transformers with hardware-efficient training. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 56501– 56523, 2024. URL https://proceedings.mlr.press/v235/yang24ab.html.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving Mamba2 with delta rule. In International Conference on Learning Representations, 2025. URL https:// openreview.net/forum?id=r8H7xhYPwz.

Michael Zhang, Kush Bhatia, Hermann Kumbong, and Christopher Re. The hedgehog & the porcu- ´ pine: Expressive linear attentions with softmax mimicry. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2402.04347.

Michael Zhang, Simran Arora, Rahul Chalamala, Alan Wu, Benjamin Spector, Aaryan Singhal, Krithik Ramesh, and Christopher Re. LoLCATs: On low-rank linearizing of large lan-´ guage models. In International Conference on Learning Representations, 2025. URL https: //arxiv.org/abs/2410.10254.

## A ADDITIONAL PROOFS AND ANALYSIS

## A.1 FULL PROOF AND EXTENSIONS OF THEOREM 1

We provide a complete and generalized proof of Theorem 1, allowing separate read and write assignment concentrations (i.e., scaled codeword norms) and deriving an explicit approximation error bound stated in Theorem 2.

Theorem 2 (Error bound with separate read and write concentrations and finite codebooks). Under the circular-codebook andfactorized read and write setup of Theorem 1, let $\beta _ { r } , \beta _ { w } \ge 0$ denote the read and write codeword norms. With $\| \mathbf { c } _ { j } \| _ { 2 } = 1$ , the corresponding codewords are $\beta _ { r } \mathbf { c } _ { j }$ and $\beta _ { w } \mathbf { c } _ { j }$ respectively. Let $M = \operatorname* { m a x } _ { 0 \leq u < U }$ max $\{ \beta _ { r } \| \mathbf { q } _ { t } ^ { ( u ) } \| _ { 2 } , \beta _ { w } \| \mathbf { k } _ { s } ^ { ( u ) } \| _ { 2 } \}$ and $\xi _ { d _ { p } , M } = 2 e ^ { M } M ^ { d _ { p } } / d _ { p } !$ . The log-overlap satisfies

$$
\left| \log \kappa _ { \boldsymbol { A } } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) - \frac { \beta _ { r } \beta _ { w } } { 2 } \mathbf { q } _ { t } ^ { \top } \mathbf { k } _ { s } \right| \leq U \left[ \frac { M ^ { 4 } } { 4 } + \frac { 3 \xi _ { d _ { p } , M } } { 1 - \xi _ { d _ { p } , M } } \right] , \qquad \xi _ { d _ { p } , M } < 1 .\tag{11}
$$

Proof. We first derive the continuous-codebook limit and its dot-product approximation, and then bound the dot-product approximation error and finite-codebook error separately.

For the uniform circular codewords $\mathbf { c } _ { j } ~ { = } ~ ( \cos ( 2 \pi j / d _ { p } ) , \sin ( 2 \pi j / d _ { p } ) ) ^ { \top }$ , introduce the partition function

$$
Z _ { d _ { p } } ( { \bf x } ) = \frac { 1 } { d _ { p } } \sum _ { j = 0 } ^ { d _ { p } - 1 } \exp ( { \bf c } _ { j } ^ { \top } { \bf x } ) .
$$

Here the read and write scales are absorbed into the argument of $Z _ { d _ { p } }$ . Since $\mathcal { A } = \{ 0 , \ldots , d _ { p } - 1 \} ^ { U }$ the product structure of the read and write distributions allows the overlap to factorize over the U factors:

$$
\begin{array} { r l } { \kappa _ { \perp } ( { \bf q } _ { \parallel } , { \bf k } _ { \mathrm { s } } ) = | { \cal A } | \sum _ { i ^ { \prime } = 1 } ^ { n } \exp _ { s } ( { \bf a } ) } \\ & { = \frac { \kappa _ { \perp } ( { \bf k } _ { \parallel } ) } { n ^ { 3 } } \sum _ { i ^ { \prime } = 1 } ^ { n } \frac { \exp ( \beta _ { i ^ { \prime } } \mathrm { c } _ { i ^ { \prime } } ^ { \top } ( { \bf a } ) \mathrm { q } _ { i } ^ { ( \mathrm { i n } ) } ) } { \sin ( ( \beta _ { i ^ { \prime } } \mathrm { c } _ { i ^ { \prime } } ^ { \top } ) \mathrm { q } _ { i ^ { \prime } } } \frac { U _ { i ^ { \prime } - 1 } ^ { - 1 } } { \exp ( \beta _ { i ^ { \prime } } \mathrm { c } _ { i ^ { \prime } } ^ { \top } ) \mathrm { k } _ { i ^ { \prime } } ^ { \top ( { \mathrm { i n } ) } } } \times } \\ &  = \frac { d _ { i ^ { \prime } } ^ { \top } } { n ^ { 3 } } \frac { \kappa _ { \perp } ( { \bf k } _ { \parallel } ) } { \sin ( ( \frac { \alpha _ { i ^ { \prime } } } { \beta _ { i ^ { \prime } } } \mathrm { c } _ { i ^ { \prime } } ^ { \top } - 1 ) \exp ( \beta _ { i ^ { \prime } } \mathrm { c } _ { i ^ { \prime } } ^ { \top } ) \mathrm { q } _ { i ^ { \prime } } ^ { ( \mathrm { i n } ) } } \cdot \prod _ { i = 0 } ^ { n } \frac { \exp ( \beta _ { i ^ { \prime } } \mathrm { c } _ { i ^ { \prime } } ^ { \top } \mathrm { k } _ { i ^ { \prime } } ^ { ( \mathrm { i n } ) } ) }  \sin ( ( \beta _ { i ^ { \prime } } \mathrm { c } _ { j ^ { \prime } } ^ { \top } ) \mathrm { k } _ { i ^ { \prime } } ^  ( \mathrm { i n }  \end{array}\tag{12}
$$

Let $Z _ { \infty } ( \mathbf { x } ) = \operatorname* { l i m } _ { d _ { p } \to \infty } Z _ { d _ { p } } ( \mathbf { x } )$ and $\begin{array} { r } { \kappa _ { \infty } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) = \operatorname* { l i m } _ { d _ { v } \to \infty } \kappa _ { \mathcal { A } } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) } \end{array}$ . The circular Riemann sum converges to an integral; substituting this limit into Eq. (12) gives

$$
\begin{array} { l } { { \displaystyle Z _ { \infty } ( { \bf x } ) = \operatorname* { l i m } _ { d _ { p } \to \infty } \frac { 1 } { d _ { p } } \sum _ { j = 0 } ^ { d _ { p } - 1 } \exp \Bigl ( \| { \bf x } \| _ { 2 } \cos \Bigl ( \frac { 2 \pi j } { d _ { p } } - \arg { \bf x } \Bigr ) \Bigr ) } } \\ { ~ } \\ { { \displaystyle \qquad = \frac { 1 } { 2 \pi } \int _ { 0 } ^ { 2 \pi } \exp ( \| { \bf x } \| _ { 2 } \cos \theta ) d \theta } } \\ { { \displaystyle \qquad = I _ { 0 } ( \| { \bf x } \| _ { 2 } ) } } \\ { { \displaystyle \qquad \quad \kappa _ { \infty } ( { \bf q } _ { t } , { \bf k } _ { s } ) = \prod _ { u = 0 } ^ { U - 1 } \frac { I _ { 0 } \Bigl ( \| \beta _ { r } { \bf q } _ { t } ^ { ( u ) } + \beta _ { w } { \bf k } _ { s } ^ { ( u ) } \| _ { 2 } \Bigr ) } { I _ { 0 } \Bigl ( \| \beta _ { r } { \bf q } _ { t } ^ { ( u ) } \| _ { 2 } \Bigr ) I _ { 0 } \Bigl ( \| \beta _ { w } { \bf k } _ { s } ^ { ( u ) } \| _ { 2 } \Bigr ) } } . } \end{array}\tag{13}
$$

Here $I _ { 0 }$ is the modified Bessel function of the first kind of order zero.

To identify the leading term of the log-kernel, use $\begin{array} { r } { I _ { 0 } ( \rho ) = 1 + \frac { \rho ^ { 2 } } { 4 } + O ( \rho ^ { 4 } ) } \end{array}$ and $\log ( 1 + x ) =$ $\textstyle x - { \frac { x ^ { 2 } } { 2 } } + O ( x ^ { 3 } )$ as $x \to 0$

$$
\log { I _ { 0 } ( \rho ) } = \left( { \textstyle { \frac { \rho ^ { 2 } } { 4 } } } + O ( \rho ^ { 4 } ) \right) - { \textstyle { \frac { 1 } { 2 } } } \left( { \textstyle { \frac { \rho ^ { 2 } } { 4 } } } + O ( \rho ^ { 4 } ) \right) ^ { 2 } + O ( \rho ^ { 6 } ) = \frac { \rho ^ { 2 } } { 4 } + O ( \rho ^ { 4 } ) .
$$

Substituting this expansion into the logarithm of Eq. (13) as $M  0$ and using the squared-norm identity $\| \beta _ { r } \mathbf { q } _ { t } ^ { ( u ) } + \beta _ { w } \mathbf { k } _ { s } ^ { ( u ) } \| _ { 2 } ^ { 2 } = \beta _ { r } ^ { 2 } \| \mathbf { q } _ { t } ^ { ( u ) } \| _ { 2 } ^ { 2 } + 2 \beta _ { r } \beta _ { w } \mathbf { q } _ { t } ^ { ( u ) } \mathbf { \Sigma } \mathbf { k } _ { s } ^ { ( u ) } + \beta _ { w } ^ { 2 } \| \mathbf { k } _ { s } ^ { ( u ) } \| _ { 2 } ^ { 2 }$ yields

$$
\begin{array} { r l } & { \log \mathcal { R } _ { \infty } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) = \displaystyle \sum _ { u = 0 } ^ { U - 1 } \left[ \log { \cal I } _ { 0 } \Big ( \Big \| \beta _ { r } \mathbf { q } _ { t } ^ { ( u ) } + \beta _ { u } \mathbf { k } _ { s } ^ { ( u ) } \Big \| _ { 2 } \Big ) \right. } \\ & { \qquad \left. - \log { \cal I } _ { 0 } \Big ( \Big \| \beta _ { r } \mathbf { q } _ { t } ^ { ( u ) } \Big \| _ { 2 } \Big ) - \log { \cal I } _ { 0 } \Big ( \Big \| \beta _ { u } \mathbf { k } _ { s } ^ { ( u ) } \Big \| _ { 2 } \Big ) \right] } \\ & { \qquad = \displaystyle \frac { 1 } { 4 } \sum _ { u = 0 } ^ { U - 1 } \Big [ \Big \| \beta _ { r } \mathbf { q } _ { t } ^ { ( u ) } + \beta _ { u } \mathbf { k } _ { s } ^ { ( u ) } \Big \| _ { 2 } ^ { 2 } - \beta _ { r } ^ { 2 } \Big \| \mathbf { q } _ { t } ^ { ( u ) } \Big \| _ { 2 } ^ { 2 } - \beta _ { u } ^ { 2 } \Big \| \mathbf { k } _ { s } ^ { ( u ) } \Big \| _ { 2 } ^ { 2 } \Big ] } \\ & { \qquad \quad + O \left( \displaystyle \sum _ { u = 0 } ^ { U - 1 } \Big ( \beta _ { r } \Big \| \mathbf { q } _ { t } ^ { ( u ) } \Big \| _ { 2 } + \beta _ { u } \Big \| \mathbf { k } _ { s } ^ { ( u ) } \Big \| _ { 2 } \Big ) ^ { 4 } \right) } \\ & { \qquad = \displaystyle \frac { \beta _ { r } \beta _ { u } } { 2 } \mathbf { q } _ { t } ^ { \top } \mathbf { k } _ { s } + O \Big ( \Big ( \beta _ { r } \Big \| \mathbf { q } _ { t } \Big \| _ { 2 } + \beta _ { w } \Big \| \mathbf { k } _ { s } ^ { ( u ) } \Big \| _ { 2 } \Big ) ^ { 4 } \Big ) } \endarray \end{array}\tag{14}
$$

The last step uses a conservative remainder bound in terms of the norms of the full query and key vectors by the triangle inequality. Eq. (14) gives the continuous-codebook version of the dot-product approximation in Theorem 1, generalized to separate read and write assignment concentrations $\beta _ { r }$ and $\beta _ { w }$

We now replace this asymptotic remainder with an explicit bound. With M as defined in Theorem 2, the triangle inequality gives $\| \boldsymbol { \beta } _ { r } \mathbf { q } _ { t } ^ { ( u ) } + \boldsymbol { \beta } _ { w } \mathbf { k } _ { s } ^ { ( u ) } \| _ { 2 } \leq 2 M$ for every u. It therefore suffices to control the scalar approximation error of log $I _ { 0 }$ on these argument ranges.

For $\rho \geq 0$ , set $t = \rho ^ { 2 } / 4$ , giving

$$
\left( 1 + \frac { t } { 2 } \right) ^ { 2 } = 1 + t + \frac { t ^ { 2 } } { 4 } \leq I _ { 0 } ( \rho ) = \sum _ { m = 0 } ^ { \infty } \frac { t ^ { m } } { ( m ! ) ^ { 2 } } \leq \sum _ { m = 0 } ^ { \infty } \frac { t ^ { m } } { m ! } = e ^ { t } .
$$

To obtain a lower bound for log $I _ { 0 } ,$ , we use the following integral representation and the inequality $1 - s \leq 1 / ( 1 + s )$ for $s \geq 0 \colon$

$$
( t / 2 ) - { \frac { ( t / 2 ) ^ { 2 } } { 2 } } = \int _ { 0 } ^ { \frac { t } { 2 } } ( 1 - s ) d s \leq \int _ { 0 } ^ { \frac { t } { 2 } } { \frac { 1 } { 1 + s } } d s = \log ( 1 + { \frac { t } { 2 } } ) .
$$

Taking logarithms of the power-series bounds and applying this inequality yields

$$
t - { \frac { t ^ { 2 } } { 4 } } \leq 2 \log \left( 1 + { \frac { t } { 2 } } \right) \leq \log I _ { 0 } ( \rho ) \leq t .
$$

Rearranging isolates the quadratic approximation error $0 \leq t - \log I _ { 0 } ( \rho ) \leq t ^ { 2 } / 4$ . Define this scalar remainder by

$$
\eta ( \rho ) = \frac { \rho ^ { 2 } } { 4 } - \log I _ { 0 } ( \rho )
$$

and substitute $t = \rho ^ { 2 } / 4$ to obtain

$$
0 \leq \eta ( \rho ) \leq \frac { \rho ^ { 4 } } { 6 4 } .\tag{15}
$$

Substituting log $\begin{array} { r } { I _ { 0 } ( \rho ) = \frac { \rho ^ { 2 } } { 4 } - \eta ( \rho ) } \end{array}$ into the logarithm of Eq. (13) gives the exact remainder:

$$
\begin{array} { r } { \log \kappa _ { \infty } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) - \frac { \beta _ { r } \beta _ { w } } { 2 } \mathbf { q } _ { t } ^ { \top } \mathbf { k } _ { s } = \displaystyle \sum _ { u = 0 } ^ { U - 1 } \Big [ \eta \Big ( \beta _ { r } \| \mathbf { q } _ { t } ^ { ( u ) } \| _ { 2 } \Big ) + \eta \Big ( \beta _ { w } \| \mathbf { k } _ { s } ^ { ( u ) } \| _ { 2 } \Big ) } \\ { - \eta \Big ( \| \beta _ { r } \mathbf { q } _ { t } ^ { ( u ) } + \beta _ { w } \mathbf { k } _ { s } ^ { ( u ) } \| _ { 2 } \Big ) \Big ] . } \end{array}
$$

By Eq. (15), the first two terms in each summand lie in $[ 0 , M ^ { 4 } / 6 4 ]$ ], while the subtracted term lies in $[ 0 \dot { 0 } , ( 2 \dot { M } ) ^ { 4 } / 6 4 ]$ . Summing these bounds gives

$$
- \frac { U M ^ { 4 } } { 4 } \leq \log \kappa _ { \infty } ( { \bf q } _ { t } , { \bf k } _ { s } ) - \frac { \beta _ { r } \beta _ { w } } { 2 } { \bf q } _ { t } ^ { \top } { \bf k } _ { s } \leq \frac { U M ^ { 4 } } { 3 2 } .\tag{16}
$$

Thus the continuous-kernel approximation error satisfies

$$
\left| \log \kappa _ { \infty } ( { \bf q } _ { t } , { \bf k } _ { s } ) - \frac { \beta _ { r } \beta _ { w } } { 2 } { \bf q } _ { t } ^ { \top } { \bf k } _ { s } \right| \leq \frac { U M ^ { 4 } } { 4 } .\tag{17}
$$

Now consider the error between the finite and continuous log-kernels, $\vert \log \kappa _ { \mathcal { A } } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) - \log \kappa _ { \infty } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) \vert$ Set $\begin{array} { r l r } { \rho } & { { } = } & { \| \mathbf { x } \| _ { 2 } } \end{array}$ and let $I _ { n }$ denote the modified Bessel function of the first kind of order n. Applying the Fourier-Bessel expansion exp(ρ cos $\begin{array} { r } { \ \phi ) \ = \ I _ { 0 } ( \rho ) + 2 \sum _ { n = 1 } ^ { \infty } I _ { n } ( \rho ) } \end{array}$ cos(nϕ) to $Z _ { d _ { p } } ( \mathbf { x } )$ shows that only frequencies divisible by $d _ { p }$ survive the discrete average:

$$
\begin{array} { l } { { \displaystyle Z _ { d _ { p } } ( { \bf x } ) = I _ { 0 } ( \rho ) + 2 \sum _ { n = 1 } ^ { \infty } I _ { n } ( \rho ) \left[ \frac { 1 } { d _ { p } } \sum _ { j = 0 } ^ { d _ { p } - 1 } \cos \Bigl ( n \left( \frac { 2 \pi j } { d _ { p } } - \arg { \bf x } \right) \Bigr ) \right] } } \\ { { \displaystyle \qquad = Z _ { \infty } ( { \bf x } ) + 2 \sum _ { \ell = 1 } ^ { \infty } I _ { \ell d _ { p } } ( \rho ) \cos ( \ell d _ { p } \arg { \bf x } ) } . } \end{array}
$$

These additional modes account for the finite-codebook error. Dividing by $Z _ { \infty } ( \mathbf { x } ) = I _ { 0 } ( \rho ) > 0$ and using | cos $\theta | \leq 1$ gives

$$
\left| \frac { Z _ { d _ { p } } ( \mathbf { x } ) } { Z _ { \infty } ( \mathbf { x } ) } - 1 \right| \le \frac { 2 \sum _ { \ell = 1 } ^ { \infty } I _ { \ell d _ { p } } ( \rho ) } { I _ { 0 } ( \rho ) } .
$$

To bound this tail explicitly, compare the power series of $I _ { n }$ and use $( m + n ) ! \geq$ m!n!:

$$
I _ { n } ( \rho ) = \left( \frac { \rho } { 2 } \right) ^ { n } \sum _ { m = 0 } ^ { \infty } \frac { ( \rho ^ { 2 } / 4 ) ^ { m } } { m ! ( m + n ) ! } \leq \frac { ( \rho / 2 ) ^ { n } } { n ! } \sum _ { m = 0 } ^ { \infty } \frac { ( \rho ^ { 2 } / 4 ) ^ { m } } { ( m ! ) ^ { 2 } } = \frac { ( \rho / 2 ) ^ { n } } { n ! } I _ { 0 } ( \rho ) .
$$

Apply the resulting bound $I _ { n } ( \rho ) / I _ { 0 } ( \rho ) \leq ( \rho / 2 ) ^ { n } / n ! .$ , and use $( d _ { p } + m ) ! \geq d _ { p } ! m ! \colon$

$$
\begin{array} { r l r } {  { \frac { 2 \sum _ { \ell = 1 } ^ { \infty } I _ { \ell d _ { p } } ( \rho ) } { I _ { 0 } ( \rho ) } \le 2 \sum _ { \ell = 1 } ^ { \infty } \frac { ( \rho / 2 ) ^ { \ell d _ { p } } } { ( \ell d _ { p } ) ! } \le 2 \sum _ { n = d _ { p } } ^ { \infty } \frac { ( \rho / 2 ) ^ { n } } { n ! } = 2 ( \frac { \rho } { 2 } ) ^ { d _ { p } } \sum _ { m = 0 } ^ { \infty } \frac { ( \rho / 2 ) ^ { m } } { ( d _ { p } + m ) ! } } } \\ & { } & { \le \frac { 2 ( \rho / 2 ) ^ { d _ { p } } } { d _ { p } ! } \sum _ { m = 0 } ^ { \infty } \frac { ( \rho / 2 ) ^ { m } } { m ! } = \frac { 2 e ^ { \rho / 2 } ( \rho / 2 ) ^ { d _ { p } } } { d _ { p } ! } . ~ } \end{array}
$$

Since every argument norm in the kernel factors is at most 2M, combining the preceding bounds gives, uniformly for $\| \mathbf { x } \| _ { 2 } \leq 2 M$

$$
\frac { 2 \sum _ { \ell = 1 } ^ { \infty } I _ { \ell d _ { p } } ( \rho ) } { I _ { 0 } ( \rho ) } \leq \xi _ { d _ { p } , M } , \qquad \xi _ { d _ { p } , M } = \frac { 2 e ^ { M } M ^ { d _ { p } } } { d _ { p } ! } .\tag{18}
$$

To compare the log-kernels, we convert this relative error bound into a logarithmic one. Assume $\xi _ { d _ { p } , M } < 1$ and $\| \mathbf { x } \| _ { 2 } \leq 2 M$ , and set $s = Z _ { d _ { p } } ( \mathbf { x } ) / I _ { 0 } ( \| \mathbf { x } \| _ { 2 } ) - 1$ . Eq. (18) gives $| s | \leq \xi _ { d _ { p } , M }$ , and every v between 0 and s satisfies $1 + v \ge 1 - \dot { \xi } _ { d _ { p } , M } > 0 .$ Consequently,

$$
| \log ( 1 + s ) | = \left| \int _ { 0 } ^ { s } \frac { 1 } { 1 + v } d v \right| \leq \left| \int _ { 0 } ^ { s } \frac { 1 } { 1 - \xi _ { d _ { p } , M } } d v \right| \leq \frac { \xi _ { d _ { p } , M } } { 1 - \xi _ { d _ { p } , M } } .
$$

Substituting the definition of s yields, uniformly for $\| \mathbf { x } \| _ { 2 } \leq 2 M$

$$
\left| \log \frac { Z _ { d _ { p } } ( \mathbf { x } ) } { Z _ { \infty } ( \mathbf { x } ) } \right| \le \frac { \xi _ { d _ { p } , M } } { 1 - \xi _ { d _ { p } , M } } .\tag{19}
$$

By Eqs. (12) and (13), the logarithm of each factor of $\kappa _ { \mathcal { A } } / \kappa _ { \infty }$ contains one numerator correction and two denominator corrections, all with argument norms at most 2M. Applying Eq. (19) to these $3 U$ terms and using the triangle inequality gives

$$
\left| \log \frac { \kappa _ { \mathcal { A } } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) } { \kappa _ { \infty } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) } \right| \leq \frac { 3 U \xi _ { d _ { p } , M } } { 1 - \xi _ { d _ { p } , M } } .\tag{20}
$$

Combining the finite-codebook bound in Eq. (20) with the continuous-kernel approximation bound in Eq. (17) proves Eq. (11):

$$
\begin{array} { l } { \displaystyle \left. \log \kappa _ { \boldsymbol { A } } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) - \frac { \beta _ { r } \beta _ { w } } { 2 } \mathbf { q } _ { t } ^ { \top } \mathbf { k } _ { s } \right. } \\ { \displaystyle \leq \left. \log \kappa _ { \boldsymbol { A } } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) - \log \kappa _ { \infty } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) \right. + \left. \log \kappa _ { \infty } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) - \frac { \beta _ { r } \beta _ { w } } { 2 } \mathbf { q } _ { t } ^ { \top } \mathbf { k } _ { s } \right. } \\ { \displaystyle \leq U \left[ \frac { M ^ { 4 } } { 4 } + \frac { 3 \xi _ { d _ { p } , M } } { 1 - \xi _ { d _ { p } , M } } \right] } \end{array}
$$

For fixed $d _ { p } , \xi _ { d _ { p } , M } = { \cal O } ( M ^ { d _ { p } } )$ as $M  0 .$ , so $\xi _ { d _ { p } , M } < 1$ holds for sufficiently small M. The total error is therefore $O \big ( U ( M ^ { 4 } + M ^ { d _ { p } } ) \big )$ , and hence $O ( U M ^ { 4 } )$ whenever $d _ { p } \geq 4 .$ For fixed U, setting $\beta _ { r } = \beta _ { w } = \beta$ and exponentiating gives

$$
\kappa _ { \cal A } ( { \bf q } _ { t } , { \bf k } _ { s } ) = \exp ( \frac { \beta ^ { 2 } } { 2 } { \bf q } _ { t } ^ { \top } { \bf k } _ { s } ) [ 1 + O ( U M ^ { 4 } ) ] , \qquad M  0 ,
$$

which recovers Theorem 1.

## A.2 FROM SOFTMAX ATTENTION TO MASS-AWARE RECURRENT READOUT

We approximate causal softmax attention by a mass-aware readout over address states. When the local regime holds uniformly over $s \leq t ,$ , the first step follows from Theorem 2; the remaining equalities are exact. Canceling |A| and exchanging the sums over tokens and addresses yields a readout whose address weights are proportional to the read probability multiplied by the accumulated write mass:

$$
\begin{array} { r l } { \frac { \sum _ { s \leq t } \exp \Big ( \frac { \partial \nu _ { s } \rho _ { s } } { \rho _ { s } } \mathbf { u } _ { s } \Big ) \mathbf { v } _ { s } } { \sum _ { s \leq t } \exp \Big ( \frac { \partial \nu _ { s } \rho _ { s } } { \rho _ { s } } \mathbf { u } _ { t } ^ { \intercal } \mathbf { k } _ { s } \Big ) } \approx } & { \frac { \sum _ { s \leq t } \kappa _ { A } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) \mathbf { v } _ { s } } { \sum _ { s \leq t } \kappa _ { A } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) } } \\ & { = \frac { { \left| A \right| } \sum _ { s \leq t } \sum _ { \mathbf { n } \in A } r _ { i } ( \mathbf { a } ) \mathbf { v } _ { s } ( \mathbf { a } ) \mathbf { v } _ { s } } { \left| A \right| \sum _ { s \leq t } \kappa _ { B } \varepsilon _ { A } \gamma _ { t } ( \mathbf { b } ) \left| \mathcal { U } _ { s } \left( \mathbf { b } \right) \right| } } \\ & { = \frac { \sum _ { s \in A } \chi _ { t } ( \mathbf { n } ) \sum _ { s \leq t } w _ { s } ( \mathbf { a } ) \mathbf { v } _ { s } } { \sum _ { \mathbf { n } \in A } \gamma _ { t } ( \mathbf { b } ) \left| \mathcal { U } _ { s } \left( \mathbf { b } \right) \right| } } \\ & { = \frac { \sum _ { s \in A } \chi _ { t } ( \mathbf { n } ) \sum _ { s \leq t } w _ { s } ( \mathbf { b } ) } { \sum _ { s \in A } \chi _ { t } ( \mathbf { b } ) \left| \mathcal { D } _ { t } \left( \mathbf { b } \right) \right| } \frac { \sum _ { s \leq t } w _ { s } ( \mathbf { a } ) \mathbf { v } _ { s } } { m _ { t } ( \mathbf { b } ) } } \\ &  = \frac { r } { \sin \left( \mathbf { A } \right) } \frac { r _ { t } ( \mathbf { a } ) m _ { t } ( \mathbf { a } ) }  \ \end{array}\tag{21}
$$

Here $m _ { t } ( \mathbf { a } )$ is the cumulative write mass, and ${ \bf s } _ { t } ( { \bf a } )$ is the write-weighted mean of values at address a:

$$
m _ { t } ( \mathbf { a } ) = \sum _ { s < t } w _ { s } ( \mathbf { a } ) , \qquad \mathbf { s } _ { t } ( \mathbf { a } ) = \frac { \sum _ { s \leq t } w _ { s } ( \mathbf { a } ) \mathbf { v } _ { s } } { m _ { t } ( \mathbf { a } ) } .
$$

Appendix A.3 shows how GSU recovers these cumulative statistics in its cumulative-averaging limit.

## A.3 GSU DYNAMICS AND THE CUMULATIVE-AVERAGING LIMIT

We show that GSU recovers the cumulative write mass and weighted value mean derived in Appendix A.2. For fixed soft write weights $0 < w _ { t } ( \mathbf { a } ) < 1$ , taking $\gamma _ { t } \to - \infty$ at every step makes the retention gate tend to one:

$$
\sigma ( \mathrm { l o g i t } ( 1 - w _ { t } ( \mathbf { a } ) ) - \gamma _ { t } ) \to 1
$$

In this limit, the mass update retains all previous mass and adds the current write weight. Starting from $m _ { 0 } ( { \bf a } ) = 0 \mathrm { a n d } { \bf s } _ { 0 } ( { \bf a } ) = { \bf 0 }$ , the accumulated mass is therefore

$$
m _ { t } ( \mathbf { a } ) = m _ { t - 1 } ( \mathbf { a } ) + w _ { t } ( \mathbf { a } ) = \sum _ { s \leq t } w _ { s } ( \mathbf { a } ) .
$$

We set $\epsilon = 0$ in the state update. Substituting the mass recurrence then shows by induction that the state is the cumulative write-weighted mean. The fourth equality uses the induction hypothesis; the base case follows from the zero initialization:

$$
\begin{array} { r l } & { \mathbf { s } _ { t } ( \mathbf { a } ) = \mathbf { s } _ { t - 1 } ( \mathbf { a } ) + \frac { w _ { t } ( \mathbf { a } ) } { m _ { t } ( \mathbf { a } ) } \left( \mathbf { v } _ { t } - \mathbf { s } _ { t - 1 } ( \mathbf { a } ) \right) } \\ & { \quad \quad = \frac { \left( m _ { t } ( \mathbf { a } ) - w _ { t } ( \mathbf { a } ) \right) \mathbf { s } _ { t - 1 } ( \mathbf { a } ) + w _ { t } ( \mathbf { a } ) \mathbf { v } _ { t } } { m _ { t } ( \mathbf { a } ) } } \\ & { \quad \quad = \frac { m _ { t - 1 } ( \mathbf { a } ) \mathbf { s } _ { t - 1 } ( \mathbf { a } ) + w _ { t } ( \mathbf { a } ) \mathbf { v } _ { t } } { m _ { t } ( \mathbf { a } ) } } \\ & { \quad \quad = \frac { \sum _ { s \in t } w _ { s } ( \mathbf { a } ) \mathbf { v } _ { s } + w _ { t } ( \mathbf { a } ) \mathbf { v } _ { t } } { m _ { t } ( \mathbf { a } ) } } \\ & { \quad \quad = \frac { \sum _ { s \in t } w _ { s } ( \mathbf { a } ) \mathbf { v } _ { s } } { \sum _ { s \in t } w _ { s } ( \mathbf { a } ) } } \end{array}
$$

Thus GSU recovers cumulative weighted averaging in this limit. Combined with the mass-aware readout in Eq. (21), it exactly reproduces the normalized address-kernel attention.

## A.4 EFFECTS OF CODEBOOK SIZE, SPARSITY, AND READOUT

Building on the attention approximation in Appendix A.1, we examine how codebook resolution, sparse address selection, and readout weighting affect the resulting value field. Fig. 4 compares these configurations with full attention, using a shared read and write codeword norm $\beta _ { r } = \beta _ { w } = \beta$

The codeword norm $\beta$ controls how strongly reads and writes concentrate on nearby addresses. Top-K truncation introduces larger artifacts at small $\beta ,$ where diffuse address weights leave more probability mass outside the selected slots. Increasing $d _ { p }$ refines the angular grid and expands the address space to $d _ { p } ^ { U }$ slots. At moderate $\beta ,$ the finer codebook reduces coarse grid artifacts and produces smoother transitions that more closely resemble the full-attention reference.

Standard RAM-Net uses $\begin{array} { r } { \sum _ { \mathbf { a } } r _ { t } ( \mathbf { a } ) \mathbf { s } _ { t } ( \mathbf { a } ) } \end{array}$ , whereas the mass-aware readout in Eq. (21) replaces the coefficients with $r _ { t } ( \mathbf { a } ) m _ { t } ( \mathbf { a } ) \big / \sum _ { \mathbf { b } } r _ { t } ( \mathbf { b } ) m _ { t } ( \mathbf { b } )$ ). The two agree when the masses are equal and positive on the read support and $\begin{array} { r } { \sum _ { \mathbf { a } } ^ { \mathbf { \tilde { \alpha } } } r _ { t } ( \mathbf { a } ) = 1 } \end{array}$ . Otherwise, mass weighting changes the relative influence of slots and shifts the boundaries between value regions. When these masses differ, mass weighting changes the relative influence of slots and can shift the boundaries between value regions. Because the address masses depend jointly on the codebook factors, this weighting can introduce oblique transitions, as illustrated by the last two columns.

Although mass-aware readout recovers normalized address-kernel attention in the cumulativeaveraging limit, its practical benefit may be limited by interactions with forgetting and sparse selection. Appendix B.3 compares the two readouts empirically. First, forgetting can reduce a slot’s mass relative to other slots, lowering its mass-aware readout weight even when the read scores remain unchanged. This couples forgetting to both the state dynamics and the readout, potentially suppressing slots that remain relevant to the query. Second, the efficient Top-K decoder selects addresses using the factorized score, whereas the history-dependent mass generally breaks this factorization in $r _ { t } ( \mathbf { a } ) m _ { t } ( \mathbf { a } )$ . Reweighting only the selected addresses can therefore miss slots that would rank highly under the mass-aware score.

![](images/5bf40387c3a8d1b0e490efa58f0684889faeb39ae27077f3b4e5de3c1d4641a9.jpg)  
Figure 4: Value fields under different RAM-Net configurations. Keys and queries lie on $S ^ { 1 } \times S ^ { 1 } \subset$ $\mathbb { R } ^ { \tilde { 4 } }$ , with the plotted axes representing the angular coordinates of the two factors. Marker colors encode the stored values, while the background shows the predicted value $f _ { t } ( \mathbf { q } )$ . Full attention provides the reference on the left. Rows vary the codeword norm around the reference setting $\beta ^ { \star }$ and columns vary Top- $K .$ , $d _ { p }$ , and mass-aware readout.

## A.5 HARD-WRITE LIMIT AND VECTOR-QUANTIZED ATTENTION

We derive the hard-write limit from the exact address overlap in Eq. (12). We keep the read codeword norm $\beta _ { r }$ fixed and finite while taking the write codeword norm $\beta _ { w } \ \to \ \infty$ , yielding hard write assignments with soft reads.

For each key block, assume a unique maximizing code and define

$$
j _ { s } ^ { ( u ) } = \underset { 0 \leq j < d _ { p } } { \arg \operatorname* { m a x } } { \mathbf { c } _ { j } ^ { \top } \mathbf { k } _ { s } ^ { ( u ) } } = \underset { 0 \leq j < d _ { p } } { \arg \operatorname* { m i n } } \| \mathbf { k } _ { s } ^ { ( u ) } - \mathbf { c } _ { j } \| _ { 2 } ^ { 2 } .
$$

The second equality follows from $\| \mathbf { c } _ { j } \| _ { 2 } = 1$ , so the assignment is nearest-code vector quantization. For nonzero key blocks with continuously distributed angles, ties occur with probability zero, so the maximizing code is unique with probability one.

Dividing the numerator and denominator of each factor-wise write probability by the largest exponential makes every nonmaximizing term vanish as $\beta _ { w }  \infty$ . Consequently,

$$
\frac { \exp ( \beta _ { w } \mathbf { c } _ { j } ^ { \top } \mathbf { k } _ { s } ^ { ( u ) } ) } { \sum _ { i = 0 } ^ { d _ { p } - 1 } \exp ( \beta _ { w } \mathbf { c } _ { i } ^ { \top } \mathbf { k } _ { s } ^ { ( u ) } ) } \longrightarrow \mathbf { 1 } _ { \{ j = j _ { s } ^ { ( u ) } \} } .
$$

Let $\mathbf { a } _ { s } = ( j _ { s } ^ { ( 0 ) } , \dots , j _ { s } ^ { ( U - 1 ) } )$ be the selected product address; then $w _ { s } ( \mathbf { a } ) \longrightarrow \mathbf { 1 } _ { \{ \mathbf { a } = \mathbf { a } _ { s } \} }$ . Define the reconstructed key by $\widehat { \mathbf { k } } _ { s } = ( \mathbf { c } _ { j _ { s } ^ { ( 0 ) } } ^ { \top } , \ldots , \mathbf { c } _ { j _ { s } ^ { ( U - 1 ) } } ^ { \top } ) ^ { \top }$ . The hard write selects a single term in the overlap, giving

$$
\operatorname* { l i m } _ { \beta _ { w }  \infty } \kappa _ { \mathcal { A } } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) = d _ { p } ^ { U } \prod _ { u = 0 } ^ { U - 1 } \frac { \exp ( \beta _ { r } \mathbf { c } _ { j ( u ) } ^ { \top } \mathbf { q } _ { t } ^ { ( u ) } ) } { \sum _ { j = 0 } ^ { d _ { p } - 1 } \exp ( \beta _ { r } \mathbf { c } _ { j } ^ { \top } \mathbf { q } _ { t } ^ { ( u ) } ) } = \frac { \exp ( \beta _ { r } \mathbf { q } _ { t } ^ { \top } \widehat { \mathbf { k } } _ { s } ) } { \prod _ { u = 0 } ^ { U - 1 } Z _ { d _ { p } } ( \beta _ { r } \mathbf { q } _ { t } ^ { ( u ) } ) } .\tag{22}
$$

The denominator depends only on the query and therefore cancels when the attention weights are normalized over tokens:

$$
\operatorname* { l i m } _ { \beta _ { w } \to \infty } \frac { \sum _ { s \leq t } \kappa _ { \mathscr { A } } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) \mathbf { v } _ { s } } { \sum _ { s \leq t } \kappa _ { \mathscr { A } } ( \mathbf { q } _ { t } , \mathbf { k } _ { s } ) } = \frac { \sum _ { s \leq t } \exp ( \beta _ { r } \mathbf { q } _ { t } ^ { \top } \widehat { \mathbf { k } } _ { s } ) \mathbf { v } _ { s } } { \sum _ { s \leq t } \exp ( \beta _ { r } \mathbf { q } _ { t } ^ { \top } \widehat { \mathbf { k } } _ { s } ) } .\tag{23}
$$

Eq. (23) gives softmax attention over product-quantized keys (Jegou et al., 2011), with each of the´ $U$ blocks quantized using a fixed circular codebook of $d _ { p }$ codewords. Taking this hard-write limit after the cumulative-averaging limit in Appendix A.3 makes each occupied slot’s mass and state the count and mean of its assigned values. With the mass-aware readout in Eq. (21), this recovers the per-code aggregation underlying Transformer-VQ (Lingle, 2024).

## A.6 CORRESPONDENCE BETWEEN ROPE AND CAPE

Consider the full soft address distributions of the circular construction in Sec. 3.1 before Top-K truncation. We flatten each address $\mathbf { a } \in { \mathcal { A } }$ into a slot index $i \in \{ 0 , \ldots , d _ { p } ^ { U } - 1 \}$ , with $a ^ { ( 0 ) }$ as the least significant digit:

$$
i ( { \bf a } ) = \sum _ { u = 0 } ^ { U - 1 } a ^ { ( u ) } d _ { p } ^ { u } , \quad \quad a ^ { ( u ) } = \left\lfloor { \frac { i } { d _ { p } ^ { u } } } \right\rfloor { \bmod { d _ { p } } } , \quad \quad u = 0 , \dots , U - 1 .\tag{24}
$$

For each two-dimensional input $\mathbf { x } ,$ the circular assignment has the following log probability, using the partition function $Z _ { d _ { p } }$ defined in Appendix $\mathrm { A . 1 }$ :

$$
\begin{array} { l } { \log p _ { \beta } ( j \mid \mathbf { x } ) = \beta \mathbf { c } _ { j } ^ { \top } \mathbf { x } - \log \left( d _ { p } Z _ { d _ { p } } ( \beta \mathbf { x } ) \right) } \\ { \quad \quad = \beta \| \mathbf { x } \| _ { 2 } \cos \left( \displaystyle \frac { 2 \pi } { d _ { p } } j - \arg \mathbf { x } \right) - \log \left( d _ { p } Z _ { d _ { p } } ( \beta \mathbf { x } ) \right) . } \end{array}
$$

For a vector $\mathbf { x } = ( \mathbf { x } ^ { ( 0 ) } , \ldots , \mathbf { x } ^ { ( U - 1 ) } )$ with two-dimensional blocks, we remove the normalization term, which is independent of i, to define the factor logit waveform:

$$
\Phi _ { \mathbf { x } } ^ { ( u ) } ( i ) = \log p _ { \beta } ( a ^ { ( u ) } \mid \mathbf { x } ^ { ( u ) } ) + \log \Big ( d _ { p } Z _ { d _ { p } } ( \beta \mathbf { x } ^ { ( u ) } ) \Big ) = \beta \| \mathbf { x } ^ { ( u ) } \| _ { 2 } \cos \bigg ( \frac { 2 \pi } { d _ { p } } a ^ { ( u ) } - \arg \mathbf { x } ^ { ( u ) } \bigg )\tag{25}
$$

The read and write distributions are obtained by superposing their respective factor logit waves and applying a softmax over addresses:

$$
r _ { t } ( \mathbf { a } ) \propto \exp \left( \sum _ { u = 0 } ^ { U - 1 } \Phi _ { \mathbf { q } _ { t } } ^ { ( u ) } ( i ( \mathbf { a } ) ) \right) \qquad w _ { s } ( \mathbf { a } ) \propto \exp \left( \sum _ { u = 0 } ^ { U - 1 } \Phi _ { \mathbf { k } _ { s } } ^ { ( u ) } ( i ( \mathbf { a } ) ) \right)
$$

By Eq. (24), each factor waveform has period $d _ { p } ^ { u + 1 }$ and is constant over blocks of $d _ { p } ^ { u }$ slots. It can be obtained by sampling and holding the following continuous wave, where $s \in$ R denotes a continuous slot coordinate:

$$
\widetilde { \Phi } _ { \mathbf { x } } ^ { ( u ) } ( s ) = \beta \| \mathbf { x } ^ { ( u ) } \| _ { 2 } \cos \left( \frac { 2 \pi } { d _ { p } ^ { u + 1 } } s - \arg \mathbf { x } ^ { ( u ) } \right) .\tag{26}
$$

Evaluating this continuous wave at the left endpoint of each block of $d _ { p } ^ { u }$ slots exactly recovers the discrete waveform in Eq. (25):

$$
\Phi _ { \mathbf { x } } ^ { ( u ) } ( i ) = \widetilde { \Phi } _ { \mathbf { x } } ^ { ( u ) } \left( d _ { p } ^ { u } \left\lfloor \frac { i } { d _ { p } ^ { u } } \right\rfloor \right) , \qquad u = 0 , \dots , U - 1 .\tag{27}
$$

RoPE preserves each block’s norm and rotates it by $\nu ^ { ( u ) } t ,$ where $\nu ^ { ( u ) }$ is the angular frequency of factor u. Substituting this change into Eq. (26) gives

$$
\begin{array} { l } { \displaystyle \beta \| { \bf x } ^ { ( u ) } \| _ { 2 } \cos \biggl ( \frac { 2 \pi } { d _ { p } ^ { u + 1 } } s - \arg { { \bf x } ^ { ( u ) } } - \nu ^ { ( u ) } t \biggr ) } \\ { = \beta \| { \bf x } ^ { ( u ) } \| _ { 2 } \cos \biggl [ \frac { 2 \pi } { d _ { p } ^ { u + 1 } } \left( s - \frac { d _ { p } ^ { u + 1 } \nu ^ { ( u ) } } { 2 \pi } t \right) - \arg { { \bf x } ^ { ( u ) } } \biggr ] } \\ { = \widetilde { \Phi } _ { \bf x } ^ { ( u ) } \left( s - \frac { d _ { p } ^ { u + 1 } \nu ^ { ( u ) } } { 2 \pi } t \right) . } \end{array}
$$

This corresponds to translation at speed $d _ { p } ^ { u + 1 } \nu ^ { ( u ) } / ( 2 \pi )$ slots per token. Under the cyclic-shift convention $f ( i ) \mapsto f ( ( i - t )$ mod $d _ { p } ^ { U } )$ , setting the translation speed to one slot per token gives

$$
\nu ^ { ( u ) } = \frac { 2 \pi } { d _ { p } ^ { u + 1 } } , \qquad u = 0 , \ldots , U - 1 .\tag{28}
$$

With Eq. (28), every continuous factor wave becomes $\widetilde { \Phi } _ { \mathbf { x } } ^ { ( u ) } ( s - t )$ , so their superposition also translates by t slots. Since softmax commutes with cyclic slot permutations, CAPE equivalently shifts the sampled-and-held logits, whereas RoPE translates the continuous waves before sampling and holding. By Eq. (27), the two operations give the following factor logits at slot i:

$$
\mathrm { C A P E : } \quad \widetilde { \Phi } _ { \mathbf { x } } ^ { ( u ) } \left( d _ { p } ^ { u } \left\lfloor \frac { i - t } { d _ { p } ^ { u } } \right\rfloor \right) ,
$$

$$
\mathrm { R o P E : } \quad \widetilde { \Phi } _ { \mathbf { x } } ^ { ( u ) } \left( d _ { p } ^ { u } \left\lfloor \frac { i } { d _ { p } ^ { u } } \right\rfloor - t \right) .
$$

This difference in the order of operations introduces a discretization error into the resulting address distributions.

## A.7 LOW-RANK ADAPTER CONSTRUCTION

We derive the initialization in Sec. 4.1 by matching the query-key interactions induced by the teacher projections. Consider one attention head and omit layer and head indices. Under the column-vector convention used in the main text, the teacher projections are $\mathbf { W } _ { Q } , \mathbf { W } _ { K } \in \mathbb { R } ^ { d _ { k } \times d _ { \operatorname* { m o d e l } } }$ , and the target projections are $\widetilde { \mathbf { W } } _ { Q } , \widetilde { \mathbf { W } } _ { K } \in \mathbb { R } ^ { 2 U \times d _ { \mathrm { m o d e l } } }$ . Let $\Delta = i - j$ be the key position i minus the query position $j ,$ and let ${ \bf R } _ { T } ( \Delta )$ and ${ \bf R } _ { S } ( \Delta )$ denote the relative teacher and target rotations. The unscaled teacher logit is ${ \bf h } _ { j } ^ { \top } { \bf B } _ { T } ( \Delta ) { \bf h } _ { i }$ , where

$$
\begin{array} { r } { { \bf B } _ { T } ( \Delta ) = { \bf W } _ { Q } ^ { \top } { \bf R } _ { T } ( \Delta ) { \bf W } _ { K } , \qquad { \bf B } _ { S } ( \Delta ) = \widetilde { { \bf W } } _ { Q } ^ { \top } { \bf R } _ { S } ( \Delta ) \widetilde { { \bf W } } _ { K } . } \end{array}\tag{29}
$$

These bilinear operators describe the interactions that determine attention logits. Matching them therefore targets the joint behavior of the query and key projections rather than approximating each projection independently.

Let $p ( \Delta )$ be the empirical relative-position distribution weighted by teacher attention. Define the hidden-state second moments $\pmb { \Sigma } _ { Q } \dot { = } \mathbb { E } _ { j } [ \mathbf { h } _ { j } \mathbf { h } _ { i } ^ { \top } ]$ and $\pmb { \Sigma } _ { K } \equiv \mathbb { E } _ { i } [ \mathbf { h } _ { i } \mathbf { \bar { h } } _ { i } ^ { \top } ]$ to weight operator errors. Here, $\Sigma _ { Q }$ and $\Sigma _ { K }$ reflect the different positional sampling distributions of queries and keys under causal attention. Approximating squared logit error with factorized query and key second moments that do not depend on $\Delta$ gives the objective

$$
\begin{array} { r } { \mathcal { L } = \mathbb { E } _ { \Delta \sim p } \left\| \boldsymbol { \Sigma } _ { Q } ^ { 1 / 2 } \big [ \mathbf { B } _ { T } ( \Delta ) - \mathbf { B } _ { S } ( \Delta ) \big ] \boldsymbol { \Sigma } _ { K } ^ { 1 / 2 } \right\| _ { F } ^ { 2 } . } \end{array}\tag{30}
$$

Write $\widehat { \bf B } _ { T } ( \Delta ) = \Sigma _ { Q } ^ { 1 / 2 } { \bf B } _ { T } ( \Delta ) \Sigma _ { K } ^ { 1 / 2 }$ for the weighted teacher operator.

CAPE initialization. Each teacher RoPE plane contributes cosine and sine terms to ${ \bf B } _ { T } ( \Delta )$ (Su et al., 2021). We approximate these positional interactions using the target frequencies $\nu ^ { ( u ) }$ $u = 0 , \ldots , U - 1$ , specified in Eq. (28). Collect their cosine and sine functions in $\phi ( \Delta ) \ =$ $( \cos ( \nu ^ { ( 0 ) } \Delta ) , \sin ( \nu ^ { ( 0 ) } \bar { \Delta } ) , \dots , \cos ( \nu ^ { ( \bar { U } - 1 ) } \Delta ) , \sin ( \nu ^ { ( U - 1 ) } \Delta ) ) ^ { \top }$ . We first relax the two-dimensional projection structure by allowing $2 U$ unrestricted coefficient matrices $\mathbf { X } _ { a } \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } \times d _ { \mathrm { m o d e l } } }$ . The relaxed objective is

$$
\mathcal { L } = \mathbb { E } _ { \Delta \sim p } \left\| \widehat { \mathbf { B } } _ { T } ( \Delta ) - \sum _ { a = 0 } ^ { 2 U - 1 } \phi _ { a } ( \Delta ) \mathbf { X } _ { a } \right\| _ { F } ^ { 2 } .
$$

Differentiating the least-squares objective gives

$$
\frac { \partial \mathcal { L } } { \partial \mathbf { X } _ { a } } = 2 \left( \sum _ { b } \mathbb { E } _ { \Delta \sim p } [ \phi _ { a } ( \Delta ) \phi _ { b } ( \Delta ) ] \mathbf { X } _ { b } - \mathbb { E } _ { \Delta \sim p } [ \phi _ { a } ( \Delta ) \widehat { \mathbf { B } } _ { T } ( \Delta ) ] \right) .
$$

Let $G _ { a b } = \mathbb { E } _ { \Delta \sim p } [ \phi _ { a } ( \Delta ) \phi _ { b } ( \Delta ) ]$ denote the entries of the Gram matrix G. A solution of the normal equations is

$$
\mathbf { X } _ { a } ^ { \star } = \sum _ { b = 0 } ^ { 2 U - 1 } ( \mathbf { G } ^ { \dagger } ) _ { a b } \mathbb { E } _ { \Delta \sim p } [ \phi _ { b } ( \Delta ) \widehat { \mathbf { B } } _ { T } ( \Delta ) ] .\tag{31}
$$

Here, $\mathbf { G } ^ { \dagger }$ is the Moore-Penrose pseudoinverse. This gives the minimum-norm solution of the unconstrained frequency projection (Penrose, 1956), with G accounting for correlations among the target basis functions. We next seek target projections $\widetilde { \mathbf { W } } _ { Q }$ and $\widetilde { \mathbf { W } } _ { K }$ whose induced weighted operator approximates the frequency expansion from Eq. (31):

$$
\Sigma _ { Q } ^ { 1 / 2 } \widetilde { \mathbf { W } } _ { Q } ^ { \top } \mathbf { R } _ { S } ( \Delta ) \widetilde { \mathbf { W } } _ { K } \boldsymbol { \Sigma } _ { K } ^ { 1 / 2 } \approx \sum _ { a = 0 } ^ { 2 U - 1 } \phi _ { a } ( \Delta ) \mathbf { X } _ { a } ^ { \star } .
$$

For each target frequency $\nu ^ { ( u ) }$ , we approximate the corresponding cosine and sine coefficients using a single rotation plane. Let the rows of $\widetilde { \mathbf { W } } _ { Q } ^ { ( u ) } \Sigma _ { Q } ^ { 1 / 2 }$ and $\widetilde { \mathbf { W } } _ { K } ^ { ( u ) } \bar { \mathbf { { Z } } } _ { K } ^ { 1 / 2 }$ be $\mathbf { a } _ { u , 0 } ^ { \top } , \mathbf { a } _ { u , 1 } ^ { \top }$ and $\mathbf { b } _ { u , 0 } ^ { \top } , \mathbf { b } _ { u , 1 } ^ { \top } ,$ respectively. Then

$$
\begin{array} { r l } & { \boldsymbol { \Sigma } _ { Q } ^ { 1 / 2 } \left( \widetilde { \mathbf { W } } _ { Q } ^ { ( u ) } \right) ^ { \top } \mathbf { R } _ { S } ^ { ( u ) } ( \Delta ) \widetilde { \mathbf { W } } _ { K } ^ { ( u ) } \boldsymbol { \Sigma } _ { K } ^ { 1 / 2 } } \\ & { = \left[ \mathbf { a } _ { u , 1 } ^ { \top } \right] ^ { \top } \left[ \cos ( \nu ^ { ( u ) } \Delta ) \right. \left. - \sin ( \nu ^ { ( u ) } \Delta ) \right] \left[ \mathbf { b } _ { u , 1 } ^ { \top } \right] } \\ & { \approx \cos ( \nu ^ { ( u ) } \Delta ) \mathbf { X } _ { 2 u } ^ { \star } + \sin ( \nu ^ { ( u ) } \Delta ) \mathbf { X } _ { 2 u + 1 } ^ { \star } . } \end{array}\tag{32}
$$

Based on Eq. (32), define $\mathbf { a } _ { u } = \mathbf { a } _ { u , 0 } + i \mathbf { a } _ { u , 1 } , \mathbf { b } _ { u } = \mathbf { b } _ { u , 0 } + i \mathbf { b } _ { u , 1 }$ , and $\mathbf { M } _ { u } = \mathbf { X } _ { 2 u } ^ { \star } + i \mathbf { X } _ { 2 u + 1 } ^ { \star }$ , where i is the imaginary unit:

$$
\begin{array} { r l } & { \mathrm { R e } [ e ^ { - i \nu _ { u } \Delta } \mathbf { a } _ { u } \mathbf { b } _ { u } ^ { H } ] = \cos ( \nu ^ { ( u ) } \Delta ) ( \mathbf { a } _ { u , 0 } \mathbf { b } _ { u , 0 } ^ { \top } + \mathbf { a } _ { u , 1 } \mathbf { b } _ { u , 1 } ^ { \top } ) + \sin ( \nu ^ { ( u ) } \Delta ) ( \mathbf { a } _ { u , 1 } \mathbf { b } _ { u , 0 } ^ { \top } - \mathbf { a } _ { u , 0 } \mathbf { b } _ { u , 1 } ^ { \top } ) } \\ & { \quad \mathrm { R e } [ e ^ { - i \nu _ { u } \Delta } \mathbf { M } _ { u } ] = \cos ( \nu ^ { ( u ) } \Delta ) \mathbf { X } _ { 2 u } ^ { \star } + \sin ( \nu ^ { ( u ) } \Delta ) \mathbf { X } _ { 2 u + 1 } ^ { \star } } \end{array}
$$

Minimizing the sum of the squared Frobenius errors of the cosine and sine coefficients is equivalent to approximating $\mathbf { M } _ { u }$ by $\mathbf { a } _ { u } \mathbf { b } _ { u } ^ { H }$ , whose rank is at most one. By the Eckart-Young-Mirsky theorem (Eckart & Young, 1936; Mirsky, 1960), the best rank-one approximation of $\mathbf { M } _ { u }$ in Frobenius norm is given by its leading singular component. We apply singular value decomposition (SVD) to ${ { \bf { M } } _ { u } }$ and retain its leading singular component, $\mathbf { M } _ { u } \approx \sigma _ { u } ^ { \phantom { \dagger } } \mathbf { 1 } _ { u } \mathbf { r } _ { u } ^ { H }$ . Taking $\mathbf { a } _ { u } = \sqrt { \sigma _ { u } } \mathbf { l } _ { u }$ and $\mathbf { b } _ { u } = \sqrt { \sigma _ { u } } \mathbf { r } _ { u }$ gives

$$
\widetilde { \mathbf { W } } _ { Q } ^ { ( u ) } = \sqrt { \sigma } _ { u } \left[ ( \mathrm { R e } \mathbf { l } _ { u } ) ^ { \top } \right] \boldsymbol { \Sigma } _ { Q } ^ { - 1 / 2 } , \qquad \widetilde { \mathbf { W } } _ { K } ^ { ( u ) } = \sqrt { \sigma } _ { u } \left[ ( \mathrm { R e } \mathbf { r } _ { u } ) ^ { \top } \right] \boldsymbol { \Sigma } _ { K } ^ { - 1 / 2 } .\tag{33}
$$

Stacking the U two-row blocks in Eq. (33) gives the 2U-dimensional projections required by Sec. 3.1. The frequency projection is optimal before imposing the plane structure, and each rank-one truncation is optimal for its coefficient matrix. Because the basis functions can be correlated and unequally weighted under $p ( \Delta )$ , the separate truncations do not generally minimize the joint constrained objective in Eq. (30). They instead provide the structured initialization refined by forward KL alignment in Sec. 4.1.

NoPE initialization. For NoPE, $\mathbf { R } _ { S } ( \Delta ) = \mathbf { I }$ , so the target operator is independent of relative position. Under Eq. (30), its optimal unconstrained value in the weighted space is $\widehat { \bf B } _ { 0 } = \mathbb { E } _ { \Delta \sim p } [ \widehat { \bf B } _ { T } ( \Delta ) ]$ expanding the squared error around this mean leaves a constant term plus the squared distance to $\widehat { \mathbf { B } } _ { 0 }$ Since the target operator has rank at most $2 U ,$ , its optimal constrained approximation is the truncated SVD of this mean. Let $\mathbf { U } _ { r } \mathbf { S } _ { r } \mathbf { V } _ { r } ^ { \top }$ denote the truncation with $r = \operatorname* { m i n } ( 2 U , d _ { k } )$ . The corresponding projections are

$$
\widetilde { \bf W } _ { Q } = { \bf S } _ { r } ^ { 1 / 2 } { \bf U } _ { r } ^ { \top } { \bf \Sigma } _ { { \Sigma } _ { Q } ^ { - 1 / 2 } } , \qquad \widetilde { \bf W } _ { K } = { \bf S } _ { r } ^ { 1 / 2 } { \bf V } _ { r } ^ { \top } { \bf \Sigma } _ { { \cal K } } ^ { - 1 / 2 } .\tag{34}
$$

If $r < 2 U$ , zero rows complete the required dimensions. This construction averages the teacher’s positional interactions under $p ( \Delta )$ rather than simply removing its RoPE rotations.

Both constructions are compatible with the adapter in Sec. 4.1. The weighted teacher operators have column space contained in that of $\boldsymbol { \Sigma } _ { \boldsymbol { Q } } ^ { 1 / 2 } \mathbf { W } _ { \boldsymbol { Q } } ^ { \top }$ and row space contained in that of ${ \bf W } _ { K } { \pmb \Sigma } _ { K } ^ { 1 / 2 }$

Table 5: Architectures of the teacher models.
<table><tr><td>Family</td><td>Teacher</td><td>Layer</td><td>Hidden size</td><td>Head</td><td>KV Head</td><td> $d _ { \mathrm { h e a d } }$ </td><td>Attn.</td><td>Native ctx.</td></tr><tr><td rowspan="3">FLA</td><td>transformer-340M-10B</td><td>24</td><td>1024</td><td>32</td><td>32</td><td>32</td><td>MHA</td><td>4K</td></tr><tr><td>transformer-1.3B-100B</td><td>24</td><td>2048</td><td>32</td><td>32</td><td>64</td><td>MHA</td><td>2K</td></tr><tr><td>transformer-2.7B-100B</td><td>32</td><td>2560</td><td>20</td><td>20</td><td>128</td><td>MHA</td><td>2K</td></tr><tr><td rowspan="3">Qwen2.5</td><td>0.5B-Base</td><td>24</td><td>896</td><td>14</td><td>2</td><td>64</td><td>GQA</td><td>32K</td></tr><tr><td>1.5B-Base</td><td>28</td><td>1536</td><td>12</td><td>2</td><td>128</td><td>GQA</td><td>32K</td></tr><tr><td>3B-Base</td><td>36</td><td>2048</td><td>16</td><td>2</td><td>128</td><td>GQA</td><td>32K</td></tr><tr><td>RedPajama</td><td>INCITE-Base-3B-v1</td><td>32</td><td>2560</td><td>32</td><td>32</td><td>80</td><td>MHA</td><td>2K</td></tr><tr><td>Mistral</td><td>7B-v0.1</td><td>32</td><td>4096</td><td>32</td><td>8</td><td>128</td><td>GQA/SWA</td><td>8K</td></tr><tr><td>LLaMA</td><td>Llama-2-7b-hf</td><td>32</td><td>4096</td><td>32</td><td>32</td><td>128</td><td>MHA</td><td>4K</td></tr></table>

Table 6: Overview of Transformer-to-RAM-Net migration. The total training budget is 500M tokens per teacher.
<table><tr><td>Phase</td><td>Stage</td><td>Main Loss</td><td>Trainable Parameters</td><td>Tokens Warmup LR</td><td></td><td></td></tr><tr><td>Intermediate</td><td>Low-rank &amp; positional alignment Attention KL Circular soft quantization</td><td></td><td>QK adapters, log-scales</td><td>7.209M 0.360M 0 -</td><td></td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td rowspan="4">Recovery</td><td>Head-level</td><td>Head-output NMSE</td><td>Wr, Ww , scales, γt</td><td>49.152M 1.638M</td><td></td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Attn.-block local</td><td>Residual-update NMSE Attention branch</td><td></td><td>61.440M2.048M</td><td></td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Attn.-block stack</td><td>Residual-update NMSE Attention branch</td><td></td><td>36.864M1.638M</td><td></td><td> $3 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Full-model</td><td> $0 . 2 5 \mathrm { L o g i t } \mathrm { K L } + \mathrm { C E }$ </td><td>All parameters</td><td>345.375M0.819M</td><td></td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr></table>

Frequency projection preserves these containments, as do the retained nonzero singular components. After inverse weighting, the constructed query and key projections therefore lie in the row spaces of $\mathbf { W } _ { Q }$ and ${ \bf W } _ { K }$ , respectively. Consequently, adapters realizing these projections are $\mathbf { P } _ { Q } = \widetilde { \mathbf { W } } _ { Q } \mathbf { W } _ { Q } ^ { \dagger }$ and $\mathbf { P } _ { K } = \widetilde { \mathbf { W } } _ { K } \mathbf { W } _ { K } ^ { \dagger }$ , which satisfy $\widetilde { \mathbf { W } } _ { Q } = \mathbf { P } _ { Q } \mathbf { W } _ { Q }$ and $\widetilde { \mathbf { W } } _ { K } = \mathbf { P } _ { K } \mathbf { W } _ { K }$

Finally, the derivation matches unscaled interactions. Including the factor $( 2 U / d _ { k } ) ^ { 1 / 4 }$ in both adapters preserves the teacher’s dot-product coefficient because $( 2 U / d _ { k } ) ^ { 1 / 2 } / \sqrt { 2 U } = 1 / \sqrt { d _ { k } }$ . The resulting projections provide the initialization for forward KL alignment and subsequent circular address folding.

## B EXPERIMENTS

## B.1 TEACHER MODELS

We evaluate the generality of our migration method on nine teacher models from five families with different architectural configurations, as summarized in Table 5. The five families are FLA FLA Hub, Qwen2.5 Qwen et al. (2025), RedPajama Together Computer (2023), Mistral Jiang et al. (2023), and LLaMA-2 Touvron et al. (2023). All teachers are loaded from their corresponding publicly available pretrained checkpoints on Hugging Face. We use the 10BT sample of the FineWeb-Edu dataset HuggingFaceFW (2024) for training. Their different head dimensions, numbers of query and KV heads, and attention configurations lead to different low-rank adapter shapes and head-wise CAPE/NoPE assignments during migration.

## B.2 MIGRATION DETAILS

RAM-Net configurations. Unless otherwise specified, RAM-Net uses $d _ { p } = 4 , U = 5 .$ , and Top-$K = 3 2$ , and the retained address probabilities are not renormalized after Top-K. Output normalization is disabled throughout migration because the Transformer teacher has no corresponding operation. The sigmoid output gate is introduced only at the beginning of attention-block recovery. We initialize its projection weight to zero and bias to $b _ { g } = 3 ,$ giving a constant gain $g _ { 0 } = \sigma ( 3 )$ and rescale the copied output projection as $W ^ { o }  W ^ { o } \big / g _ { 0 }$ . This preserves the attention output at insertion while allowing the gate to become learnable in subsequent recovery.

Table 7: Mass-aware readout results. Values in parentheses indicate the score differences relative to RAM-Net in Table 1.
<table><tr><td>Family</td><td>Teacher</td><td>Wiki PPL ↓</td><td>ARC-E↑</td><td>ARC-C ↑</td><td>PIQA ↑</td><td>SciQ ↑</td><td>COPA↑</td><td>Hella. ↑</td><td>Avg. ↑</td><td>R↑</td></tr><tr><td rowspan="3">FLA</td><td>transformer-340M-10B</td><td>35.28 (+0.1)</td><td>55.8(+1.1)</td><td>23.9(+0.2)</td><td>64.3 (-0.4)</td><td>80.3 (+0.1)</td><td>66.0 (+1.0)</td><td>31.1 (+0.1)</td><td>53.6(+0.4)</td><td>87.7 (+1.2)</td></tr><tr><td>transformer-1.3B-100B</td><td>21.83 (-0.1)</td><td>52.3 (-0.4)</td><td>23.7(-1.1)</td><td>67.7(-0.6)</td><td>78.0(-0.8)</td><td>72.0 (+1.0)</td><td>35.8 (+0.2)</td><td>54.9 (-0.3)</td><td>86.4(-0.1)</td></tr><tr><td>transformer-2.7B-100B</td><td>20.09(+0.3)</td><td>56.8 (+0.3)</td><td>25.4(+0.8)</td><td>69.6(-0.7)</td><td>79.4(-0.5)</td><td>76.0 (+2.0)</td><td>37.8 (-0.5)</td><td>57.5(+0.2)</td><td>89.6(+0.4)</td></tr><tr><td rowspan="3">Qwen2.5</td><td>0.5B-Base</td><td>23.55(+0.2)</td><td>64.1 (+0.5)</td><td>28.8 (+2.4)</td><td>68.4(-0.1)</td><td>81.9(0.0)</td><td>69.0(-4.0)</td><td>36.8 (-0.1)</td><td>58.2 (-0.2)</td><td>86.1 (+6.9)</td></tr><tr><td>1.5B-Base</td><td>17.30(+1.0)</td><td>74.8 (-0.5)</td><td>40.2(-1.3)</td><td>74.7(+0.4)</td><td>88.4(-1.4)</td><td>77.0(-5.0)</td><td>44.7(-0.9)</td><td>66.6(-1.5)</td><td>90.2(-4.8)</td></tr><tr><td>3B-Base</td><td>13.89 (+0.3)</td><td>77.2(0.0)</td><td>44.2(+0.5)</td><td>77.0(0.0)</td><td>91.6(-1.3)</td><td>81.0(+2.0)</td><td>50.3 (-0.4)</td><td>70.2(+0.1)</td><td>92.3 (+0.8)</td></tr><tr><td colspan="2">RedPajama INCITE-Base-3B-v1</td><td>17.71 (+1.1)</td><td>63.9(+0.3)</td><td>29.8(+2.1)</td><td>71.4(-0.8)</td><td>81.1 (+0.5)</td><td>74.0(-3.0)</td><td>41.0(-0.9)</td><td>60.2 (-0.3)</td><td>82.1 (+2.4)</td></tr><tr><td>Mistral</td><td>7B-v0.1</td><td>10.32(+0.2)</td><td>76.7 (-0.5)</td><td>44.7(+0.6)</td><td>79.5 (-0.2)</td><td>92.3 (+0.8)</td><td>88.0(+1.0)</td><td>57.4(-0.6)</td><td>73.1 (+0.2)</td><td>89.8(+0.4)</td></tr><tr><td>LLaMA</td><td>Llama-2-7b-hf</td><td>12.98(+1.1)</td><td>71.2(-0.4)</td><td>39.4(+0.4)</td><td>76.2(0.0)</td><td>89.0(-0.2)</td><td>76.0(-5.0)</td><td>50.4(-1.2)</td><td>67.0 (-1.1)</td><td>84.6(-2.6)</td></tr></table>

Optimizer. Table 6 summarizes the complete training configuration. All stages use AdamW with zero weight decay and gradient clipping at 1. Forward and backward computation use BF16 with FP32 master weights. Each optimizer step processes 16K tokens. Low-rank alignment, head-level recovery, and attention-block recovery use linear warmup followed by cosine decay, while fullmodel recovery uses a short warmup followed by a constant learning rate. After each training stage, we restore the checkpoint with the lowest validation loss before proceeding to the next stage.

Overall workflow. During low-rank and positional alignment, only the adapters and head-wise log-scales are optimized, while the teacher and backbone remain frozen. Circular soft quantization analytically folds the selected adapters into RAM-Net read and write projections and requires no training. Head-level recovery then trains the RAM routing parameters and dynamic forgetting parameters on the native Top-K path while keeping the copied V/O projections and backbone frozen.

Attention-block recovery contains two consecutive steps. The local step feeds each student block the corresponding teacher hidden state, while the stack step propagates student hidden states through the complete stack. Both optimize the residual-update objective $\mathcal { L } _ { \mathrm { b l o c k } } ~ =$ $\begin{array} { r } { \frac { 1 } { L } \sum _ { \ell = 0 } ^ { L - 1 } \mathrm { N M S E } \left( \Delta z _ { \ell } ^ { S } , \Delta z _ { \ell } ^ { T } \right) } \end{array}$ , where $\Delta z _ { \ell } ~ = ~ z _ { \ell } - u _ { \ell } ,$ and $u _ { \ell }$ and $z _ { \ell }$ denote the block input and output. Both steps train the complete attention branch, including RAM routing, dynamic forgetting, V/O projections, and the output gate, while MLP, normalization, embedding, and language-model-head parameters remain frozen. The copied V/O projections use 0.1× the base learning rate.

Full-model recovery finally unfreezes all parameters and optimizes ${ \mathcal { L } } _ { \mathrm { f u l l } }$ . RAM-specific parameters use the base learning rate, while parameters copied from the teacher use 0.1× the base learning rate.

For both attention-block recovery steps, copied V/O parameters use 0.1× the listed learning rate. During fullmodel recovery, all parameters copied from the teacher similarly use 0.1× the listed rate, while RAM-specific parameters use the full rate.

## B.3 PERFORMANCE OF MASS-AWARE READOUT

Tables 7 and 8 compare mass-aware and standard RAM-Net readouts. Consistent with the theoretical derivation in Sec. 3.3, mass-aware readout more closely matches the Transformer before head-level recovery, yielding lower attention JS in six of nine models. However, after migration, it generally gives higher Wiki PPL while maintaining comparable common-knowledge performance. This gap may be related to the coupling of read weights to forgetting dynamics and a potential mismatch between mass reweighting and Top-K selection based only on $r _ { t } ( \mathbf { a } )$

![](images/624031d3555959895ac022c543349fa366ef859a50ef9e9189a6d809f8dec65c.jpg)  
Figure 5: Head NMSE under different circular assignment concentrations $\beta$

Table 8: JS divergence with and without mass-aware readout after enabling GSU.
<table><tr><td>Readout rule</td><td>FLA-0.3B</td><td>FLA-1.3B</td><td>FLA-2.7B</td><td>Qwen-0.5B</td><td>Qwen-1.5B</td><td>Qwen-3B</td><td>Red.-3B</td><td>Mist.-7B</td><td>LLaMA-7B</td></tr><tr><td>w/o</td><td>0.277</td><td>0.286</td><td>0.308</td><td>0.292</td><td>0.293</td><td>0.312</td><td>0.312</td><td>0.283</td><td>0.253</td></tr><tr><td>w/</td><td>0.259</td><td>0.273</td><td>0.300</td><td>0.290</td><td>0.297</td><td>0.311</td><td>0.319</td><td>0.280</td><td>0.257</td></tr></table>

Table 9: Train-from-scratch results for 340M models with and without mass-aware readout.
<table><tr><td colspan="4">Wiki PPL↓ ARC-E↑ ARC-C ↑</td><td>PIQA↑</td><td>SciQ↑</td><td>COPA↑</td><td></td><td>Hella. ↑ Avg. ↑</td></tr><tr><td>w/o</td><td>26.89</td><td>59.1</td><td>24.7</td><td>66.9</td><td>82.5</td><td>67.0</td><td>33.4</td><td>55.6</td></tr><tr><td>w/</td><td>27.10</td><td>59.1</td><td>26.1</td><td>67.5</td><td>82.1</td><td>73.0</td><td>33.3</td><td>56.8</td></tr></table>

Table 9 further compares the two readouts when training a 340M model from scratch with the same FineWeb-Edu 10BT dataset. Mass-aware readout again gives slightly higher Wiki PPL, while improving the average common-knowledge score from 55.6 to 56.8. This trend is consistent with the migration results.

## B.4 ADDITIONAL EXPERIMENTAL RESULTS

Detailed results on LongBench. Table 10 reports code completion and summarization results in LongBench for all teacher and RAM-Net pairs. RAM-Net recovers teacher performance more closely for smaller models, while the gap generally becomes larger at higher model scales. Since all models use the same 500M-token migration budget, these results suggest that larger teachers may require a larger recovery budget to better transfer their downstream capabilities.

Head NMSE with concentration $\beta .$ Fig. 5 shows a trend consistent with the attention JS results in Fig. 3. Head NMSE is minimized near $\beta ^ { \star }$ across teachers, with some minima around 1.58β<sup>⋆</sup>, further supporting $\beta ^ { \star }$ as a balanced default setting.

Head-recovery loss curves. Fig. 7 extends the low-rank alignment and circular codebook comparison in Table 2 to head-level recovery. Consistent with the lower attention JS reported in the main text, the circular codebook also starts from the lowest NMSE and shows the fastest error reduction during recovery. Together, these results show that the circular construction improves both the initial attention alignment and the efficiency of subsequent recovery.

Per-stage breakdown in migration. Using FLA-2.7B as an example, Fig. 8 shows how different migration stages affect attention matching and output recovery. Circular soft quantization introduces approximation error, while Top-K truncation causes the largest increase in attention KL. This is expected because the address-overlap approximation in Sec. 3.2 is defined over the complete soft read and write distributions, whereas Top-K discards part of the overlap and sparsifies the induced token-to-token matching. Despite this large attention mismatch, head-level recovery rapidly reduces the output NMSE while the attention KL remains high. This indicates that RAM-Net can recover teacher head outputs through its sparse address routing and recurrent updates without reproducing the teacher attention distributions exactly. Attention-block and full-model recovery further reduce the remaining output discrepancy. The slight increase in NMSE during full-model recovery reflects the shift from intermediate-output matching to the final logit KL and cross-entropy objectives.

Different $d _ { p }$ and U. Fig. 6 evaluates the effect of $d _ { p }$ and U during intermediate construction. Increasing U generally reduces the forward KL and head NMSE by preserving a higher-dimensional QK representation, while a larger $d _ { p }$ provides finer angular resolution for the circular codebook. However, lower intermediate error does not necessarily translate into better final migration performance. Since the number of RAM-Net slots grows as $\check { d } _ { p } ^ { U }$ , increasing either parameter also enlarges the address space and can reduce overlap mass and effective slot utilization, as observed in the orig inal RAM-Net study. Table 11 reflects this trade-off, where $U = 5$ achieves the best final $\operatorname { A v g }$ and relative score despite U = 6 having lower low-rank KL. We therefore use $d _ { p } = 4$ and $U = 5$ as a balanced setting between intermediate approximation quality and final language-model performance.

Table 10: LongBench results for all teacher and RAM-Net pairs.
<table><tr><td>Metric</td><td>Method</td><td>FLA-0.3B</td><td>FLA-1.3B</td><td>FLA-2.7B</td><td>Qwen-0.5B</td><td>Qwen-1.5B</td><td>Qwen-3B</td><td>Red.-3B</td><td>Mist.-7B</td><td>LLaMA-7B</td></tr><tr><td rowspan="2">LCC ↑</td><td>Teacher</td><td>17.0</td><td>48.3</td><td>50.7</td><td>10.0</td><td>14.7</td><td>14.1</td><td>44.7</td><td>70.0</td><td>66.7</td></tr><tr><td>RAM-Net</td><td>19.6</td><td>36.4</td><td>36.4</td><td>18.5</td><td>12.7</td><td>15.5</td><td>32.7</td><td>33.4</td><td>24.8</td></tr><tr><td rowspan="2">MultiNews ↑</td><td>Teacher</td><td>10.5</td><td>6.6</td><td>6.3</td><td>21.4</td><td>25.2</td><td>24.5</td><td>20.1</td><td>15.5</td><td>13.8</td></tr><tr><td>RAM-Net</td><td>9.6</td><td>7.2</td><td>8.4</td><td>12.9</td><td>13.9</td><td>15.9</td><td>8.6</td><td>12.1</td><td>11.5</td></tr><tr><td rowspan="2">SAMSum ↑</td><td>Teacher</td><td>6.6</td><td>6.2</td><td>4.9</td><td>37.3</td><td>42.4</td><td>45.5</td><td>0.0</td><td>28.2</td><td>31.4</td></tr><tr><td>RAM-Net</td><td>8.0</td><td>9.3</td><td>10.3</td><td>20.3</td><td>28.7</td><td>29.5</td><td>0.1</td><td>22.1</td><td>23.9</td></tr><tr><td rowspan="2">Avg. ↑</td><td>Teacher</td><td>11.3</td><td>20.3</td><td>20.6</td><td>22.9</td><td>27.4</td><td>28.0</td><td>21.6</td><td>37.9</td><td>37.3</td></tr><tr><td>RAM-Net</td><td>12.4</td><td>17.6</td><td>18.3</td><td>17.3</td><td>18.4</td><td>20.3</td><td>13.8</td><td>22.5</td><td>20.1</td></tr></table>

![](images/4effbc704f01fca6224a04b6fca4e46c1c55eeaae0288ca3fa381dbfd81dc7a0.jpg)  
Figure 6: Effect of $d _ { p }$ and order U on intermediate forward KL and head NMSE.

Ablations on staged migration. Table 12 provides a detailed comparison of the migration pipeline. Complete pipeline denotes the full staged procedure, while full-model only performs only full-model recovery. The “matched” checkpoints use the same cumulative token budgets as the corresponding stages of the complete pipeline. Pattern-free CAPE selection determines CAPE heads directly from their relative alignment errors without the preset pattern. w/o head-level recovery and w/o attention-block recovery omit the corresponding recovery stage and reallocate its training tokens to full-model recovery.

The full-model only baseline achieves higher task scores at the early matched-head budget, but the complete pipeline overtakes it after attention-block recovery and reaches higher final Avg. and relative score R under the same 500M-token budget. Removing either head-level or attention-block recovery also lowers the final Avg. and R, even when Wiki PPL remains similar. These results show that the intermediate recovery stages improve teacher-relative downstream performance beyond what can be achieved by allocating the same training budget directly to full-model recovery.

Table 11: Effect of order U on Qwen2.5-1.5B. Logit KL is measured after full-model recovery.
<table><tr><td rowspan="2">Order</td><td rowspan="2">Low-rank KL↓</td><td rowspan="2">Quant. JS↓</td><td rowspan="2">Head NMSE↓</td><td rowspan="2">Logit KL↓</td><td rowspan="2">Wiki PPL↓</td><td rowspan="2"></td><td rowspan="2">ARC-E↑ ARC-C ↑ PIQA ↑ SciQ ↑ COPA ↑ Hella. ↑ Avg. ↑ R ↑</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"></td></tr><tr><td></td></tr><tr><td>4 (256 slots)</td><td>1.957</td><td>0.253</td><td>0.177</td><td>0.500</td><td>16.58</td><td>73.8</td><td>38.9</td><td>73.6</td><td>87.4</td><td>82.0</td><td>45.6</td><td>66.9</td><td>90.8</td></tr><tr><td>5 (1024 slots)</td><td>1.748</td><td>0.293</td><td>0.197</td><td>0.465</td><td>16.33</td><td>75.3</td><td>41.5</td><td>74.3</td><td>89.8</td><td>82.0</td><td>45.6</td><td>68.1</td><td>95.0</td></tr><tr><td>6 (4096 slots)</td><td>1.555</td><td>0.339</td><td>0.209</td><td>0.476</td><td>16.01</td><td>74.4</td><td>41.6</td><td>74.0</td><td>89.2</td><td>77.0</td><td>45.4</td><td>66.9</td><td>91.8</td></tr></table>

Table 12: Ablation of the staged migration pipeline on Qwen2.5-1.5B.
<table><tr><td>Method</td><td>Checkpoint</td><td>Tokens (M) Logit KL ↓</td><td></td><td>Wiki ↓</td><td>ARC-E</td><td>ARC-C</td><td>PIQA SciQ</td><td></td><td>COPA Hella.</td><td></td><td>Avg. ↑</td><td>R↑</td></tr><tr><td>Teacher</td><td>Original</td><td>一</td><td></td><td>12.60</td><td>75.2</td><td>41.1</td><td>75.6</td><td>94.2</td><td>83.0</td><td>50.2</td><td>69.9</td><td></td></tr><tr><td>Complete</td><td>Head-Rec. end</td><td>56.4</td><td></td><td>77.56</td><td>47.1</td><td>25.1</td><td>64.2</td><td>69.9</td><td>69.0</td><td>35.2</td><td>51.7 43.8</td><td></td></tr><tr><td></td><td>Block-Rec. end</td><td>154.7</td><td></td><td>19.44</td><td>72.1</td><td>37.6</td><td>73.6 87.2</td><td></td><td>79.0</td><td>44.1</td><td>65.6 86.3</td><td></td></tr><tr><td></td><td>Final</td><td>500.0</td><td>0.164</td><td>16.33</td><td>75.3</td><td>41.5</td><td>74.3</td><td>89.8</td><td>82.0</td><td>45.6</td><td>68.1 95.0</td><td></td></tr><tr><td>Full-Model only</td><td>Matched head</td><td>56.4</td><td>0.419</td><td>23.90</td><td>65.1</td><td>31.5</td><td>70.2</td><td>83.0</td><td>77.0</td><td>40.7</td><td>61.2 71.1</td><td></td></tr><tr><td></td><td>Matched block</td><td>154.7</td><td>0.278</td><td>19.35</td><td>70.1</td><td>35.7</td><td>73.687.3</td><td></td><td>78.0</td><td>43.2</td><td>64.7 82.6</td><td></td></tr><tr><td></td><td>Final</td><td>500.0</td><td>0.197</td><td>16.82</td><td>71.8</td><td>39.2</td><td>74.3</td><td>89.5</td><td>80.0</td><td>44.6</td><td>66.6 89.7</td><td></td></tr><tr><td>Pattern-free CAPE Final</td><td></td><td>500.0</td><td>0.162</td><td>16.17</td><td>74.3</td><td>41.1</td><td>74.3</td><td>89.1</td><td>81.0</td><td>45.8</td><td>67.6 93.7</td><td></td></tr><tr><td>w/o head Rec.</td><td>Final</td><td>500.0</td><td>0.166</td><td>16.31</td><td>72.8</td><td>39.7</td><td>74.4</td><td>89.2</td><td>80.0</td><td>45.2</td><td></td><td>66.9 90.8</td></tr><tr><td>w/o block Rec.</td><td>Final</td><td>500.0</td><td>0.181</td><td>16.64</td><td>74.3</td><td>39.2</td><td>73.9</td><td>89.4</td><td>79.0</td><td>45.4</td><td></td><td>66.9 90.2</td></tr></table>

![](images/2afbdbe514d8545b72e312977f52b6fe6c1912f716bd019ffc5690094bda8607.jpg)  
Figure 7: Head-level recovery NMSE under different low-rank adapter and codebook constructions.

![](images/526cf064009d8b7de33caad1e4eca6f4567e92b30d22effd7413302e8832b0bc.jpg)  
Figure 8: Stage-wise evolution throughout the migration pipeline of FLA-2.7B.