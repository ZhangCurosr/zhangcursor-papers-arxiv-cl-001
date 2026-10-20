# SPECMRL: REDESIGN MATRYOSHKA REPRESENTA-TION LEARNING WITHIN PARAMETRIC SPECTRUM

Zhongxin Chen<sup>1∗</sup>, Songhao Wu<sup>1∗</sup>, Heng Cui<sup>3</sup>, Cong Li<sup>3</sup>, Xiao Zhou<sup>1†</sup>, Rui Yan<sup>2†</sup>

<sup>1</sup> Gaoling School of Artificial Intelligence, Renmin University of China

<sup>2</sup> Lenovo Group <sup>3</sup> School of Computer Science, Wuhan University

{chenzhongxin, songhaowu, xiaozhou}@ruc.edu.cn rui.yan@whu.edu.cn

## ABSTRACT

Matryoshka Representation Learning (MRL) is designed to support flexible representation sizes by jointly optimizing nested prefixes. While effective, this design lacks an explicit constraint to ensure that the leading portion provides a suitable compact encoding. To address this, we propose SpecMRL, which reorganizes the feature space to enable more effective prefix readout. Specifically, we extend MRL with a parameter-derived spectral basis organized with feature importance. SpecMRL otherwise follows standard MRL, with prefix readout performed in the learned spectral space. Experiments on MTEB show that SpecMRL outperforms MRL at all evaluated reduced dimensions, with larger gains under more aggressive compression. These results suggest that the learned spectral basis better prioritizes informative directions, enabling more effective representation compression. Notably, SpecMRL extends the performance frontier of MRL, surpassing full-dimensional MRL with only half the embedding dimensions.<sup>1</sup>

## 1 INTRODUCTION

Representation learning has been a well-established research topic in the ML community (Chen et al., 2020; Gao et al., 2021b) and underpins a wide range of downstream applications (Reimers & Gurevych, 2019; Karpukhin et al., 2020). The growing applications call for more flexible representation schemes that can adapt to varying latency and storage constraints across use cases. Matryoshka Representation Learning (MRL, Kusupati et al., 2022) was introduced to address this need, offering an approach that can be easily adopted in existing representation learning frameworks.

The central idea of MRL is to extend full-dimensional representation learning to a set of nested prefixes. In practice, each nested readout is constructed from the leading portion of the full representation. This design enables representations at multiple output dimensions to be learned within a single training run. While simple and effective, it does not explicitly ensure that earlier dimensions are better suited for nested readout, motivating a more principled design for MRL.

For such prefix readout to be effective, the leading dimensions should ideally concentrate the information most useful for compact representations. However, standard MRL provides no explicit constraint for establishing this ordering. Earlier studies (LI et al., 2025; Huy et al., 2026) have sought to improve low-dimensional readouts through cross-dimensional supervision. Yet, these approaches keep the original coordinate system unchanged, which may itself be ill-suited to effective prefix truncation. Figure 1 depicts how a more suitable basis can better organize semantic information and improve the quality of low-dimensional representations. This motivates us to rethink how the representation space itself should be organized to better support nested prefix readout in MRL.

To this end, we propose SpecMRL, a parameter-derived spectral redesign of MRL that organizes the representation space by feature importance. Such an organization better aligns with the prefixbased design of MRL and provides a natural foundation for dimension reduction. Starting from a backbone representation, SpecMRL differs from standard MRL only in its choice of basis. Figure 1 presents a comparative overview of MRL and SpecMRL, highlighting different organizations of representation space for nested readout. Specifically, SpecMRL changes only the representation basis, with standard MRL subsequently applied in the resulting space.

![](images/083573e2663d1bfbf5f7bdca9ae50287c1d0012e2e6cd97efc3578af4805318f.jpg)  
Figure 1: Schematic comparison of standard MRL and SpecMRL. Text embeddings are shown as points in the middle, with colors and marker shapes denoting distinct semantic groups. MRL retains coordinate prefixes in the original basis $\left\{ \boldsymbol { e } _ { i } \right\}$ as compact representations, whereas SpecMRL performs the readout in an ordered spectral basis $\{ u _ { i } \}$ derived from the parameter structure of $W _ { \mathrm { P r o j } }$ The 2D views and enlarged neighborhoods of $q$ (red star) illustrate how this change better organizes the semantic space and supports more effective low-dimensional representations.

Concretely, we introduce a matrix $W _ { \mathrm { P r o j } }$ and derive from its Gram matrix an ordered spectral basis that serves as the desired coordinate system. Although this is simple in design, a direct implementation poses two challenges: (1) the computational overhead of standard eigen solvers and (2) the numerical instability of eigen vector gradients. To address these challenges, we introduce several technical refinements and develop a power-iteration-based alternative to approximate the eigen decomposition of the Gram matrix. Our empirical observations further confirm that this design avoids training collapse due to numerical instability while achieving throughput comparable to vanilla MRL.

We conduct experiments to validate the effectiveness of SpecMRL and provide further insights into its design. Specifically, we evaluate SpecMRL on widely used MTEB datasets (Muennighoff et al., 2022) with backbones from multiple model families. The empirical results show that SpecMRL substantially improves performance at reduced dimensions, outperforming MRL by up to 3.80 points under 16× compression. Beyond these gains, SpecMRL further extends the performance frontier of MRL, surpassing its full-capacity performance with only half the dimensions. Further validation shows that these gains persist with larger backbones and longer training, confirming the potential of its further scaling. Our analyses also provide initial insights into why SpecMRL is effective. Taken together, these results motivate a rethink of the design principles of MRL and we hope it can inspire improvements for future representation learning.

## 2 BACKGROUND: MATRYOSHKA REPRESENTATION LEARNING (MRL)

We briefly review how sentence embeddings are obtained and trained with LLMs before the introduction of MRL (Kusupati et al., 2022). Given an input sequence s, the LLM encodes it into contextualized hidden states and aggregates them into a dense representation X:

$$
\begin{array} { r } { \pmb { X } = \pmb { \operatorname { \mathbb { P } } } \left( \operatorname { L L M } \left( \left[ \pmb { \mathscr { s } } _ { 1 } , \pmb { \mathscr { s } } _ { 2 } , \ldots , \pmb { \mathscr { s } } _ { n } \right] \right) \right) . } \end{array}
$$

Given a training batch B, X is typically optimized via the following contrastive loss:

$$
\mathcal { L } \left( X \right) = - \frac { 1 } { \vert \mathcal { B } \vert } \sum _ { X \in \mathcal { B } } \log \frac { \exp \left( \sin ( X , X ^ { + } ; T ) \right) } { \exp \left( \sin ( X , X ^ { + } ; T ) \right) + \sum _ { X ^ { - } \in \mathcal { N } } \exp \left( \sin ( X , X ^ { - } ; T ) \right) } ,\tag{1}
$$

where $X ^ { + }$ and $X ^ { - }$ denote the positive and negative samples of $X ,$ and $T$ controls the temperature of the similarity function sim $( \cdot , \cdot ; T )$ . MRL extends this contrastive learning over a set of nested dimensions $\mathcal { M } .$ , with loss applied at each dimension. Specifically, for any nested dimension $m \in \mathcal { M }$ MRL retains the first m entries of X as the reduced representation:

$$
X _ { \mathrm { M R L } } ^ { ( m ) } = X [ : , : m ] ,\tag{2}
$$

and aggregates the losses across nested dimensions into a weighted objective:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { M R L } } \ = \ \sum _ { m \in \mathcal { M } } w _ { m } \mathcal { L } \left( X _ { \mathrm { M R L } } ^ { ( m ) } \right) \ , } \end{array}\tag{3}
$$

where $w _ { m }$ denotes the weight assigned to each dimension $m$ . This nested structure motivates the name, Matryoshka Representation Learning (MRL). Here, the “Matryoshka” metaphor is motivated by Russian nesting dolls, reflecting the intuition that more informative features should occupy earlier dimensions. Despite this motivation, MRL does not explicitly ensure such an ordering, leaving the distribution of information across dimensions unspecified. Prior work (LI et al., 2025; Yang & Zhao, 2026; Zhang et al., 2025) has also considered this issue, but mainly introduces additional constraints across nested representations instead of directly ordering the feature dimensions. In this paper, we seek to establish refined design principles toward better dimension reduction. Below, we present SpecMRL as a refined perspective on how the core motivation behind MRL can be realized.

## 3 METHODOLOGY

## 3.1 SPECMRL: PARAMETRIC SPECTRAL DESIGN FOR IMPROVED MRL

In this section, we first present the implementation of SpecMRL and defer its design principles to later sections. By design, MRL relies on a heuristic prior that favors earlier dimensions to construct nested representations. SpecMRL breaks this paradigm and recasts MRL’s predefined dimension ordering as a learnable spectral organization. Specifically, it encodes this latent structure in $W _ { \mathrm { P r o j } } \in$ $\mathbb { R } ^ { D \times R }$ and recovers an ordered basis from the spectrum of the corresponding Gram matrix:

$$
U = \mathrm { \bf E i g e n } \big ( { \cal W } _ { \mathrm { P r o j } } { \cal W } _ { \mathrm { P r o j } } ^ { \top } \big ) ,\tag{4}
$$

where Eigen(·) returns the basis U, with eigen values ordered as $\lambda _ { 1 } \ \geq \ \lambda _ { 2 } \ \geq \ \cdot \cdot \cdot \geq \ \lambda _ { D } \ \geq \ 0$ Keeping the leading m columns $U _ { : , 1 : m } \in \mathbb { R } ^ { D \times m }$ as the nested readout basis, we define

$$
\pmb { X } _ { \mathrm { S p e c } } ^ { ( m ) } = \pmb { X } \pmb { U } _ { : , 1 : m } \in \mathbb { R } ^ { m }\tag{5}
$$

where $X _ { \mathrm { S p e c } } ^ { ( m ) }$ denotes the projection of $\pmb { X } \in \mathbb { R } ^ { D }$ onto the leading m basis vectors. With this definition, SpecMRL is a drop-in modification to MRL, replacing the nested representation $X ^ { ( m ) }$ with the spectral prefix $X _ { \mathrm { S p e c } } ^ { ( m ) }$ in each loss term while leaving the rest of the pipeline unchanged.

Given the constructed spectrum, the training objective of SpecMRL can be formulated as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S p e c } } \ = \ \sum _ { m \in \mathcal { M } } w _ { m } \mathcal { L } \left( X _ { \mathrm { S p e c } } ^ { ( m ) } \right) \ . } \end{array}\tag{6}
$$

This substitution changes only the construction of nested representations while preserving the vanilla MRL framework, and requires no additional loss terms or modifications to the training pipeline. Figure 2 provides an overview of the key stages in the SpecMRL pipeline.

We present a parameter-space view to help interpret our SpecMRL design. Conceptually, the learned parameters of $W _ { \mathrm { P r o j } }$ can be viewed as an encoding of the embedding-space geometry. From this perspective, SpecMRL may organize feature channels in a more principled manner than vanilla MRL, offering one possible explanation for the observed performance gains. We provide additional empirical observations in Appendix C to further support this interpretation.

![](images/cebf31ab8c6210dcaed5043fa3f949ad1b57bedfd61cd8a5e2e403936e8f73d7.jpg)  
(a) Spectral readout.

```python
# W: learnable factor [D, R]
# U: cached spectral basis [D, D]
def specmrl(q, p, W, U, y):
# Warm−started spectral update
G = W @ W.T
U, = torch.linalg.qr(G @ U.detach())
# Spectral readout
q, p = q @ U, p @ U
# Original Matryoshka loss
loss = mrl loss(q, p, y)
return loss, U.detach()
```  
(b) PyTorch-style pseudocode.  
Figure 2: Illustration and pseudocode of SpecMRL. (a) Embeddings are rotated into a learned spectral basis before prefix readouts. (b) The basis is updated by one warm-started orthogonal iteration per training step and cached for the next step. mrl loss denotes the original Matryoshka objective.

## 3.2 SPECMRL WITH EFFICIENT AND SCALABLE SPECTRAL SOLVER

While SpecMRL is simple in design, its naive implementation can involve several subtle technical challenges that are easy to overlook. A straightforward approach is to directly decompose the Gram matrix using off-the-shelf numerical tools, such as the built-in eigen solvers in PyTorch (Paszke et al., 2019), to obtain the spectral basis. However, this is (1) computationally expensive and (2) sensitive to ill-conditioned eigenvalue distribution, making it unsuitable in practice.<sup>2</sup>

To address these challenges, we approximate the ordered basis in Eq. 4 with a streaming poweriteration design. Specifically, we first initialize the basis with the ordered eigen vectors of the initial Gram matrix. Starting from this initialization, each subsequent step updates the previous basis through a single matrix multiplication followed by QR orthogonalization (Householder, 1958):

$$
\begin{array} { r } { \pmb { U } ^ { ( t ) } = \mathrm { Q R } \left( \pmb { W } _ { \mathrm { P r o j } } ^ { ( t ) } \pmb { W } _ { \mathrm { P r o j } } ^ { ( t ) \top } \pmb { U } ^ { ( t - 1 ) } \right) , } \end{array}\tag{7}
$$

where QR returns the orthogonal factor. Gradients propagate through the Gram matrix and QR factorization, while basis $U ^ { ( \overline { { t } } - 1 ) }$ is held fixed. Similar approaches have been explored in optimizer designs to reduce decomposition overhead (Ahn et al., 2025; Vyas et al., 2025). Figure 2 (b) provides PyTorch-style pseudocode for this design. We further validate the convergence of this solver in Appendix C by comparing its solved basis against the exact eigen decomposition solution.

The benefits of our power iteration design are empirically validated in practice. In particular, our power-iteration design avoids training collapse across all experimental settings with only modest computational overhead. Once training is complete, the spectral basis is fixed and no longer requires recomputation from $W _ { \mathrm { P r o j } } .$ As a result, SpecMRL incurs only the cost of a $D \times D$ linear projection at inference time. We provide a detailed analysis of both training and inference costs in Section 4.1.3 to further assess the practical efficiency of SpecMRL.

## 4 EXPERIMENT

## 4.1 METHOD ANALYSIS: CLARIFYING DESIGN PRINCIPLES FOR SPECMRL

We begin with ablation studies to shed light on the research questions regarding SpecMRL, seeking to validate its effectiveness and provide insights into the design principles behind it.

## 4.1.1 HOW DOES SPECMRL PERFORM RELATIVE TO TRADITIONAL MRL?

We train SpecMRL alongside MRL on backbones from various model families and evaluate their performance on MTEB (Muennighoff et al., 2022). Table 1 summarizes the results, while detailed experimental setups are provided in Appendix A. SpecMRL highlights the following advantages:

Table 1: Performance of SpecMRL averaged across MTEB tasks, with task-level breakdown provided in Table 15. τ controls the reduction ratio, reducing the output dimension to $1 / \tau$ of its original size. Colored entries highlight improvements over vanilla MRL, while bold text marks the best result for each backbone. Parenthetical values denote the gains of SpecMRL over MRL.
<table><tr><td></td><td colspan="2">Qwen3-0.6B</td><td colspan="2">Qwen2-0.5B</td><td colspan="2">Llama3.2-1B</td></tr><tr><td>T</td><td>MRL</td><td>SpecMRL</td><td>MRL</td><td>SpecMRL</td><td>MRL</td><td>SpecMRL</td></tr><tr><td>1</td><td>60.97</td><td>60.94 (−0.03)</td><td>58.96</td><td>58.74 (−0.22)</td><td>61.92</td><td>61.87 (−0.05)</td></tr><tr><td>2 4</td><td>60.54 59.40 57.88</td><td>61.61 (+1.07) 61.28 (+1.88) 60.46 (+2.58)</td><td>58.67 58.32 57.26</td><td>59.44 (+0.77) 59.20 (+0.88) 58.44 (+1.18)</td><td>61.74 61.36 60.62</td><td>62.42 (+0.68) 62.23 (+0.87) 61.80 (+1.18)</td></tr></table>

(1) SpecMRL outperforms MRL across all reduced dimensions. The observed improvements are consistent across backbones and compression ratios, while SpecMRL preserves the nested structure of MRL. Importantly, the performance gap widens under stronger compression. The larger gains at lower dimensions suggest that SpecMRL more effectively prioritizes informative directions for low-dimensional readouts, which may explain its advantage over MRL.

(2) SpecMRL extends the frontier of MRL, achieving superior performance using a compact representation. As shown in Table 1, SpecMRL achieves its best performance with only half of the full dimension. This reduction lowers retrieval costs, with negligible overhead and no compromise in downstream performance. We further analyze this dimension–performance trade-off, including the slight degradation observed at the full dimension, in Section 4.3.

We argue that these findings position SpecMRL as a promising alternative to standard MRL, pushing the performance frontier toward more compact representations.

## 4.1.2 WHAT MOTIVATES THE USE OF PARAMETRIC SPECTRUM FOR SPECMRL?

The spectral basis in SpecMRL can, in principle, be derived from sources beyond the learned parameter matrix. Below, we introduce several alternatives to clarify the rationale behind our choice.

Parameter-Derived Spectrum Proves Effective. One natural alternative to SpecMRL’s parameter-derived spectrum is to construct the spectral basis directly from the input features. Prior work (LI et al., 2025) has also explored exploiting feature-space structure to guide MRL, making feature-derived basis a plausible choice. We further discuss the relation to ESE in Section 4.4.

Based on this motivation, we introduce a feature-derived variant FeatMRL for comparison:

$$
U _ { \mathrm { F e a t } } ^ { ( t ) } = \mathbf { E i g e n } \Big ( X ^ { ( t ) \top } X ^ { ( t ) } \Big ) ,\tag{8}
$$

where the basis ${ U _ { \mathrm { F e a t } } ^ { ( t ) } }$ is obtained directly from the Gram matrix of the current input batch $X ^ { ( t ) }$ We further consider a momentum-based variant to mitigate potential batch-to-batch variations:

$$
\begin{array} { r } { U _ { \mathrm { F e a t } } ^ { ( t ) } = \mathbf { E i g e n } \left( M ^ { ( t ) } \right) , \quad \mathrm { w h e r e } \quad M ^ { ( t ) } = \beta M ^ { ( t - 1 ) } + \left( 1 - \beta \right) { \mathbfcal { X } } ^ { ( t ) \top } { \mathbfcal { X } } ^ { ( t ) } . } \end{array}\tag{9}
$$

Table 2: Comparison of SpecMRL and FeatMRL on MTEB. Bold values indicate the best performance, while parenthetical values denote performance drops relative to SpecMRL.
<table><tr><td>T</td><td>SpecMRL</td><td>FeatMRL (β = 0)</td><td>FeatMRL (β = 0.1)</td><td>FeatMRL (β = 0.5)</td><td>FeatMRL (β = 0.9)</td></tr><tr><td>1</td><td>60.94</td><td>60.72 (−0.22)</td><td>60.68 (–0.26)</td><td>60.69 (−0.25)</td><td>60.38 (−0.56)</td></tr><tr><td>2</td><td>61.61</td><td>60.45 (−1.16)</td><td>60.36 (-1.25)</td><td>60.42 (−1.19)</td><td>60.01 (-1.60)</td></tr><tr><td>4</td><td>61.28</td><td>59.99 (−1.29)</td><td>59.84 (-1.44)</td><td>59.91 (-1.37)</td><td>59.35 (-1.93)</td></tr><tr><td>8</td><td>60.46</td><td>58.82 (−1.64)</td><td>58.68 (−1.78)</td><td>58.75 (-1.71)</td><td>58.14 (-2.32)</td></tr><tr><td>16</td><td>58.53</td><td>55.83 (-2.70)</td><td>55.62 (-2.91)</td><td>55.90 (-2.63)</td><td>55.15 (-3.38)</td></tr></table>

Table 2 reports the results, where $\beta = 0$ denotes vanilla FeatMRL. Detailed results for all evaluated β values are provided in the Table 16. FeatMRL improves over vanilla MRL at several reduced dimensions, but remains below SpecMRL at every evaluated dimension. Momentum smoothing does not close this gap. These results favor the learned parameter-derived basis over the featurederived alternatives evaluated here.

Over-Parameterization of $W _ { \mathrm { P r o j } }$ Brings Additional Gains. In principle, any choice of $R \geq D$ is sufficient for $W _ { \mathrm { P r o j } }$ to attain full row rank. However, SpecMRL adopts an over-parameterized design with $R \gg D$ . This is motivated by our observation that increasing R beyond this threshold can further improve the performance of SpecMRL.

To study the effect of $R ,$ we conduct a hyperparameter search over R on Qwen3-0.6B and report the performance on MTEB. As shown in Table 3, we observe that (1) over-parameterizing $W _ { \mathrm { P r o j } }$ improves performance, and (2) larger values of R continue to improve performance, while the benefit of further scaling decreases. Based on these observations, we use $R \ =$ 128D throughout this paper.

Table 3: Hyperparameter search for the over-parameterization scale of $W _ { \mathrm { P r o j } }$ . Darker shades indicate better relative performance; the top-two results are shown in bold and underline.
<table><tr><td>T</td><td>D</td><td>4D</td><td>16D</td><td>64D</td><td>128D</td><td>256D</td></tr><tr><td>1</td><td>60.77</td><td>60.73</td><td>60.92</td><td>60.72</td><td>60.94</td><td>60.90</td></tr><tr><td>2</td><td>61.02</td><td>61.27</td><td>61.34</td><td>61.45</td><td>61.61</td><td>61.55</td></tr><tr><td>4</td><td>60.65</td><td>60.84</td><td>61.04</td><td>61.14</td><td>61.28</td><td>61.26</td></tr><tr><td>8</td><td>59.64</td><td>60.00</td><td>60.27</td><td>60.43</td><td>60.46</td><td>60.44</td></tr><tr><td>16</td><td>57.80</td><td>58.12</td><td>58.30</td><td>58.38</td><td>58.53</td><td>58.41</td></tr><tr><td>AVG.</td><td>59.98</td><td>60.19</td><td>60.37</td><td>60.42</td><td>60.56</td><td>60.51</td></tr></table>

One possible explanation for this advantage is that SpecMRL makes it feasible to extend MRL to higher-dimensional spaces. $W _ { \mathrm { P r o j } }$ expands the feature space from $\mathbb { R } ^ { D }$ to $\mathbb { R } ^ { R } .$ , while SpecMRL avoids the need to directly perform MRL at dimension R. This enables finer-grained similarity modeling at a manageable computational cost, thereby extending the model capacity beyond that of traditional MRL. Similar benefits of over-parameterized designs have also been observed in prior representation learning studies (Chen et al., 2020).

## 4.1.3 EFFICIENCY AND NUMERICAL STABILITY

The over-parameterized design, together with the complexity of eigen decomposition, naturally raises concerns about SpecMRL’s end-to-end efficiency. In practice, power iteration confines most of the additional computation to standard matrix operations supported by modern hardware, keeping the overhead affordable. Moreover, this overhead is inputindependent and remains small relative to the backbone computation. We provide a more comprehensive complexity analysis in Appendix E.

Table 4: Training throughput degradation of SpecMRL with different eigen solvers.
<table><tr><td>Solver</td><td>Qwen3</td><td>Qwen2</td><td>Llama3</td></tr><tr><td>SVD Eigh</td><td>-4.34% -4.59%</td><td>-3.36% -3.52%</td><td>-8.35% -4.57%</td></tr><tr><td>Power</td><td>-0.62%</td><td>-0.53%</td><td>-0.74%</td></tr></table>

Table 4 reports SpecMRL’s training throughput drop relative to MRL for different eigen solvers in the main experiments. Specifically, we benchmark SVD and Eigh using torch.linalg.svd and torch.linalg.eigh, respectively. For Eigh, we use the MAGMA backend following PyTorch’s guidance on numerical stability (Paszke et al., 2019). As shown in Table 4, our design incurs less overhead than established eigen solvers, achieving throughput nearly identical to that of MRL. Beyond efficiency, these eigen solvers are also unstable when the eigen gap $| \lambda _ { i } - \lambda _ { j } |$ vanishes.

Formally, for any eigenvector ${ \bf { u } } _ { i } ,$ its gradient with respect to the Gram matrix takes the form:

$$
\frac { \partial { \pmb u } _ { i } } { \partial { \pmb A } } [ \Delta { \pmb A } ] = \sum _ { j \neq i } \frac { { \pmb u } _ { j } { \pmb u } _ { j } ^ { \top } \Delta { \pmb A } { \pmb u } _ { i } } { \lambda _ { i } - \lambda _ { j } } , \quad \mathrm { w h e r e } \quad { \pmb A } = W _ { \mathrm { P r o j } } W _ { \mathrm { P r o j } } ^ { \top } .\tag{10}
$$

Our implementation avoids this gradient dependence, thereby mitigating sensitivity to the illconditioned eigenvalues and improving numerical stability of SpecMRL.

At inference time, once the basis is fixed, SpecMRL incurs only the overhead of a $D \times D$ matrix multiplication. Meanwhile, it halves the dimension without compromising performance, leading

to faster vector retrieval. Our evaluation confirms that SpecMRL incurs negligible overhead while providing a retrieval speedup of up to 1.86×. Section 4.6 provides the detailed experimental setup.

## 4.2 ABLATION STUDIES

We conduct ablation studies to further validate the effectiveness of the design choices behind SpecMRL and better understand why they lead to improved performance.

Table 5: Results of ablation studies. Parenthetical values denote drops relative to SpecMRL.
<table><tr><td rowspan=1 colspan=3>T   MRL  SpecMRL</td><td rowspan=1 colspan=2>(a) MRL w. Gram  (b) MRL w. Orth</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>60.97</td><td rowspan=1 colspan=1>60.94</td><td rowspan=1 colspan=1>61.35 (+0.41)</td><td rowspan=1 colspan=1>60.59 (-0.35)</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>60.54</td><td rowspan=1 colspan=1>61.61</td><td rowspan=2 colspan=1>60.80 (−0.81)59.64 (-1.64)</td><td rowspan=4 colspan=1>60.91 (−0.70)60.24 (−1.04)59.28 (−1.18)57.19 (-1.34)</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>59.40</td><td rowspan=1 colspan=1>61.28</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>57.88</td><td rowspan=1 colspan=1>60.46</td><td rowspan=2 colspan=1>57.70 (-2.76)54.11 (−4.42)</td></tr><tr><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>54.73</td><td rowspan=1 colspan=1>58.53</td></tr></table>

## 4.2.1 SPECTRUM CONTRIBUTES BEYOND GRAM-MATRIX OVER-PARAMETERIZATION.

Compared with vanilla MRL, SpecMRL introduces a learnable factor $W _ { \mathrm { P r o j } } \in \mathbb { R } ^ { D \times R }$ and derives its ordered spectral basis from the corresponding Gram matrix. To examine whether the improvement can be explained merely by this additional parameterization, we construct an ablated variant that retains the same learnable factor but removes spectral decomposition.

For a nested dimension m, let $W _ { m } = W _ { \mathrm { P r o j } } [ : m , : ]$ . This variant retains the original coordinate prefixes $X ^ { ( m ) }$ and evaluates similarity after the implicit lifting $X ^ { ( m ) } W _ { m }$ . Equivalently, its similarity is induced by the Gram matrix $\dot { G } _ { m } = W _ { m } W _ { m } ^ { \top }$ . We therefore refer to this variant as MRL w. Gram. The full formulation and details are provided in Appendix A.3.

As shown in Table 5, the Gram-based variant provides only limited improvements over vanilla MRL and degrades rapidly under stronger compression. In contrast, SpecMRL consistently performs better at reduced dimensions. These results show that introducing the same learnable factor through Gram-based similarity does not reproduce the gains of SpecMRL, supporting the proposed spectral readout over this alternative. Extra comparison with a learnable linear is provided in Appendix B.

## 4.2.2 EIGEN DECOMPOSITION PROVIDES MORE THAN LEARNABLE ORTHOGONALITY

Eigen decomposition organizes the learned structure into an ordered set of orthogonal directions. This suggests two possible sources of improvement: learning a favorable orthogonal basis and imposing a spectral ordering. To disentangle their effects, we construct an ablated variant that retains only learnable orthogonality. This allows us to assess whether the learned spectral ordering benefits MRL’s nested readouts. Specifically, we replace the orthogonal basis in Eq. 4 as:

$$
U _ { \mathrm { O r t h } } = \mathbf { Q } \mathbf { R } \big ( W _ { \mathrm { P r o j } } W _ { \mathrm { P r o j } } ^ { \top } \big ) ,\tag{11}
$$

where QR is applied to the Gram matrix alone to retain orthogonality. Table 5 (b) reports its results. Compared with vanilla MRL, learned orthogonality provides a clear advantage, but remains inferior to the full SpecMRL design. This further highlights the value of ordering feature directions by learned importance, which is naturally compatible with MRL’s nested readout design.

## 4.3 VALIDATE SPECMRL AT LARGER DIMENSIONS AND TRAINING SCALES

We scale both model width and training scale to further evaluate SpecMRL from two perspectives: (1) whether its effectiveness persists in larger representation spaces, and (2) whether its gains truly extend the performance frontier of MRL rather than merely resulting from faster convergence.

Empirical Observations at Larger Representation Dimensions. We first increase the model width and scale the Qwen3 backbone from 0.6B to 4B to extend our evaluation. Table 6 compares

Table 6: Comparison between MRL and SpecMRL on Qwen3-4B. SpecMRL remains effective as the representation space scales up.
<table><tr><td rowspan=1 colspan=4>Qwen3-4B</td></tr><tr><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>Bands</td><td rowspan=1 colspan=1>MRL|</td><td rowspan=1 colspan=1>SpecMRL</td></tr><tr><td rowspan=1 colspan=1>2560</td><td rowspan=1 colspan=1>0:20</td><td rowspan=1 colspan=1>64.71</td><td rowspan=1 colspan=1>64.29 (−0.42)</td></tr><tr><td rowspan=1 colspan=1>2432</td><td rowspan=1 colspan=1>0:19</td><td rowspan=1 colspan=1>64.73</td><td rowspan=1 colspan=1>64.98 (+0.25)</td></tr><tr><td rowspan=1 colspan=1>1024</td><td rowspan=1 colspan=1>0：8</td><td rowspan=1 colspan=1>64.38</td><td rowspan=1 colspan=1>65.06 (+0.68)</td></tr><tr><td rowspan=1 colspan=1>512</td><td rowspan=1 colspan=1>0：4</td><td rowspan=1 colspan=1>63.51</td><td rowspan=1 colspan=1>64.86 (+1.35)</td></tr><tr><td rowspan=1 colspan=1>256</td><td rowspan=1 colspan=1>0：2</td><td rowspan=1 colspan=1>62.40</td><td rowspan=1 colspan=1>64.49 (+2.09)</td></tr><tr><td rowspan=1 colspan=1>128</td><td rowspan=1 colspan=1>0:1</td><td rowspan=1 colspan=1>60.20</td><td rowspan=1 colspan=1>63.51 (+3.31)</td></tr></table>

![](images/7f6592fdbe66d793df9f7ceb11eb311f34f9b40189bbeacdc4ff7c27a692e11b.jpg)  
Figure 3: MTEB performance trends of SpecMRL across nested dimensions. The final spectral band introduces an unexpected performance drop.

SpecMRL with MRL on Qwen3-4B using MTEB. The relative gains of SpecMRL over MRL follow trends similar to those observed on the 0.6B model in the main results reported in Table 1.

Leveraging the larger hidden size of Qwen3-4B, we further conduct a fine-grained sweep over nested dimensions to investigate SpecMRL’s behavior under different spectral truncations. Specifically, we partition the spectral basis into 20 bands and report the average MTEB score over a subset of tasks with the first n bands retained. Figure 3 visualizes the resulting trend, where SpecMRL exhibits a pattern similar to that of MRL: enlarging the representation leads to improved performance for most retained spectral bands. Moreover, the learned spectral basis remains effective across unseen dimensions, indicating stability at dimensions unseen during training.

However, we observe an unexpected degradation when the final spectral bands are retained. We therefore remove these bands in other experimental settings to examine whether the same effect persists. Table 7 summarizes the results, where i : j denotes retaining bands i through j − 1.

Table 7: Comparison of SpecMRL performance with and without the final spectral bands.
<table><tr><td></td><td colspan="2">Qwen3-0.6B</td><td colspan="2">Qwen2-0.5B</td><td colspan="2">Llama3.2-1B</td></tr><tr><td>Bands</td><td>MRL</td><td>SpecMRL</td><td>MRL</td><td>SpecMRL</td><td>MRL</td><td>SpecMRL</td></tr><tr><td>0：20</td><td>60.97</td><td>60.94 (−0.03)</td><td>58.96</td><td>58.74 (−0.22)</td><td>61.92</td><td>61.87 (−0.05)</td></tr><tr><td>0：19</td><td>60.94</td><td>61.35 (+0.41)</td><td>59.00</td><td>59.20 (+0.20)</td><td>61.93</td><td>62.37 (+0.44)</td></tr><tr><td>0:10</td><td>60.54</td><td>61.61 (+1.07)</td><td>58.67</td><td>59.44 (+0.77)</td><td>61.74</td><td>62.42 (+0.68)</td></tr></table>

The degradation in the smallest-singular-value bands remains consistent across different settings. Similar behavior was observed for the spectrum of the LLM unembedding matrix (Wu et al., 2026). Our further analysis attributes this phenomenon to disproportionate projection onto the corresponding spectral subspace, which partially disrupts isotropy. We defer this analysis to Appendix D and hope these findings may motivate more principled designs for embedding training.

SpecMRL Extends the Performance Frontier Beyond Faster Convergence. We next investigate whether SpecMRL’s gains persist beyond faster convergence at larger training scales. Therefore, we extend Qwen3-0.6B training beyond main experiments and monitor its downstream performance relative to MRL in an over-training regime. As shown in Figure 4, SpecMRL not only retains its advantage over MRL throughout training but also attains a higher performance ceiling. Table 17 reports the full MTEB evaluation in an over-training regime.

![](images/2a28258298696c7253a5ce27988d90bea8713c8e324cdae9b4c0b8a4f98f4948.jpg)  
Figure 4: MTEB subset results evaluated every 100 steps. The advantage of SpecMRL is consistent throughout the training process.

Taken together, these results highlight the potential of SpecMRL toward training with larger models and longer horizons. We hope to further validate this scalability under larger computational budgets.

## 4.4 COMPARISON OF SPECMRL WITH EXISTING IMPROVEMENTS TO MRL

Table 8: Comparison of SpecMRL with MRL variants on Qwen3-0.6B.
<table><tr><td>T</td><td>MRL</td><td>ESE (L2C)</td><td>MIPIC (SIA)</td><td>SpecMRL</td></tr><tr><td>1</td><td>60.97</td><td>57.97 (-3.00)</td><td>60.42 (−0.55)</td><td>60.94 (−0.03)</td></tr><tr><td>2</td><td>60.54</td><td>57.26 (-3.28)</td><td>59.93 (–0.61)</td><td>61.61 (+1.07)</td></tr><tr><td>4</td><td>59.40</td><td>56.38 (-3.02)</td><td>59.08 (-0.32)</td><td>61.28 (+1.88)</td></tr><tr><td>8</td><td>57.88</td><td>55.18 (-2.70)</td><td>57.74 (-0.14)</td><td>60.46 (+2.58)</td></tr><tr><td>16</td><td>54.73</td><td>50.26 (-4.47)</td><td>55.09 (+0.36)</td><td>58.53 (+3.80)</td></tr></table>

We also compare SpecMRL with other MRL variants (ESE, LI et al., 2025; MIPIC, Huy et al., 2026) to further validate its effectiveness. Both ESE and MIPIC are designed to enhance information retention at lower dimensions through additional supervision. Specifically, they augment the standard MRL objective with different loss designs for cross-dimensional supervision. Given that, we restrict our comparison to their reduced-dimension variants, ESE (L2C) and MIPIC (SIA), excluding other design elements. Further discussion of their full designs is provided in Section 5.

We compare SpecMRL with these baselines on Qwen3-0.6B using the same setup as in our main experiments. Table 8 presents the comparative results, where SpecMRL achieves the best overall performance among the methods considered. Notably, ESE (L2C) shows a performance collapse on classification tasks. Therefore, we also report the average over the non-classification subset in Table 18, where the same conclusion holds. Beyond these gains, SpecMRL provides additional advantage in practice, since it only changes the prefix-readout basis. As a consequence, this makes SpecMRL a compatible extension to MRL, requiring no extra hyperparameter tuning while remaining robust across diverse tasks. Further details for baselines implementation and additional evaluation results are provided in Appendix A.2 for clarity.

## 4.5 ROBUSTNESS TO MRL CONFIGURATIONS

We further evaluate SpecMRL under different MRL settings to assess whether its advantages are robust across configurations. Specifically, we conduct two experiments on Qwen3-0.6B: (1) setting all nested loss weights to 1, and (2) adopting sparse supervision by dropping the last two dimensions (128 and 64) from the nested training set. All other training settings are kept unchanged.

Table 9 compares MRL and SpecMRL under different configurations. Across these configurations, SpecMRL consistently preserves the low-dimensional advantage observed in the main experi ments, indicating robustness to MRL training settings. Notably, under sparse supervision, SpecMRL achieves larger gains over MRL at the two prefixes without dedicated loss terms, including a 7.29- point improvement at 64 dimensions. This suggests that SpecMRL depends less on direct supervision at individual prefixes to preserve low-dimensional information, enabling effective readout under sparser nested supervision and demonstrating robustness to the choice of training dimensions.

## 4.6 INFERENCE AND RETRIEVAL EFFICIENCY

Embedding Encoding. We compare MRL and SpecMRL with Qwen3-0.6B on the same 100,000 passages. The measured time includes the forward pass and the additional projection in SpecMRL. With a precomputed spectral basis, SpecMRL performs a full 1024 × 1024 projection yet increases total encoding time from 312.85 s for MRL to 313.99 s, corresponding to an overhead of only 0.36%.

Vector Retrieval. We randomly select 8,192 passages from one million MS MARCO passages as queries and use the remainder as the retrieval database. Using FAISS’s GpuIndexFlatIP on a single NVIDIA A100 PCIe 40GB GPU, we perform FP32 exact top-100 retrieval. Indices are prebuilt and remain in GPU memory. We measure retrieval throughput in queries per second (QPS) across different dimensions and query batch sizes. As shown in Table 10, speedups generally increase with query batch size before leveling off. Using half the dimensions, SpecMRL achieves the best score (Table 1) and up to a 1.86× retrieval speedup, improving both quality and efficiency.

Table 9: Robustness to different MRL training configurations.
<table><tr><td colspan="2">Default</td><td></td><td colspan="2">Uniform Weights</td><td colspan="2">Sparse Supervision</td></tr><tr><td>T</td><td>MRL</td><td>SpecMRL</td><td>MRL</td><td>SpecMRL</td><td>MRL</td><td>SpecMRL</td></tr><tr><td>1</td><td>60.97</td><td>60.94 (−0.03)</td><td>60.35</td><td>60.37 (+0.02)</td><td>60.73</td><td>60.76 (+0.03)</td></tr><tr><td>2</td><td>60.54</td><td>61.61 (+1.07)</td><td>60.09</td><td>61.79 (+1.70)</td><td>60.11</td><td>61.53 (+1.42)</td></tr><tr><td>4</td><td>59.40</td><td>61.28 (+1.88)</td><td>59.31</td><td>61.60 (+2.29)</td><td>58.76</td><td>61.20 (+2.44)</td></tr><tr><td>8</td><td>57.88</td><td>60.46 (+2.58)</td><td>58.38</td><td>61.04 (+2.66)</td><td>55.88</td><td>60.15 (+4.27)</td></tr><tr><td>16</td><td>54.73</td><td>58.53 (+3.80)</td><td>56.48</td><td>59.30 (+2.82)</td><td>49.94</td><td>57.23 (+7.29)</td></tr></table>

Table 10: Retrieval throughput (QPS) across compression ratios τ and query batch sizes B, with output dimension 1024/τ. Parentheses show speedups over τ = 1 at the same B. Bold highlight the retrieval speedup of the representation with the highest average MTEB score in Table 1.
<table><tr><td>T</td><td>B = 1</td><td>B = 64</td><td>B = 256</td><td>B = 1024</td><td>B = 2048</td></tr><tr><td>1</td><td>140 (1.00×)</td><td>5,207 (1.00×)</td><td>7,103 (1.00×)</td><td>7,859 (1.00×)</td><td>7,961 (1.00×)</td></tr><tr><td>2</td><td>176 (1.26×)</td><td>7,713 (1.48×)</td><td>12,558 (1.77×)</td><td>14,605 (1.86×)</td><td>14,791 (1.86×)</td></tr><tr><td>4</td><td>205 (1.46×)</td><td>9,891 (1.90×)</td><td>20,174 (2.84×)</td><td>25,508 (3.25×)</td><td>25,906 (3.25×)</td></tr><tr><td>8</td><td>231 (1.65×)</td><td>11,414 (2.19×)</td><td>28,986 (4.08×)</td><td>40,594 (5.17×)</td><td>41,282 (5.19×)</td></tr><tr><td>16</td><td>352 (2.52×)</td><td>12,319 (2.37×)</td><td>36,712 (5.17×)</td><td>56,884 (7.24×)</td><td>57,814 (7.26×)</td></tr></table>

## 5 RELATED WORK

Modern applications use models to encode data into embeddings for a wide range of downstream tasks. Despite continued improvements in quality, the storage and computational costs of fixed, high-dimensional embeddings remain a practical bottleneck. MRL (Kusupati et al., 2022) addresses this limitation through joint supervision of nested prefixes, enabling flexible output dimensions. While MRL is simple, several methods seek further improvements by introducing auxiliary objectives. ESE (LI et al., 2025) introduces a learn-to-compress (L2C) objective that supervises truncated prefixes with PCA-based compression targets. Similarly, MIPIC (Huy et al., 2026) aligns tokenlevel relations across dimensions through self-distilled intra-relational alignment (SIA), matching token-importance distributions with KL divergence and inter-token geometry with centered kernel alignment. Both methods derive alignment targets from full-dimensional embeddings to supervise low-dimensional prefixes, while also incorporating intermediate-layer supervision to improve embeddings at shallower depths. In contrast to alignment-based supervision, MIC (Hong et al., 2026) introduces geometric regularization to reduce redundancy between prefix and residual dimensions.

MRL has also been extended to frozen models by adding lightweight trainable layers, supporting effective prefix truncation at multiple dimensions without backbone update. Following this idea, Matryoshka-Adaptor (Yoon et al., 2024) learns nested embeddings through lightweight adapters, while SMEC (Zhang et al., 2025) further improves this adaptation process through sequential training and adaptive dimension selection. More recently, AdaWidth (Yang & Zhao, 2026) learns an orthogonal rotation of frozen embeddings, providing a learnable orthogonal basis without explicit spectral ordering. This design is conceptually closely related to the learnable-basis ablation considered in our analysis, which underperforms the proposed spectral basis.

## 6 CONCLUSION

We present SpecMRL, a parametric-spectral redesign of MRL that addresses the limitations of fixed basis-prefix readout. By exploiting the ordered orthogonal structure of the learned spectrum, SpecMRL reorganizes representations into a form naturally suited to nested truncation. Across diverse experimental settings, SpecMRL consistently outperforms MRL at reduced dimensions with negligible overhead, while further extending the performance frontier. These findings highlight the readout space as an important learning target in nested representation. We hope this perspective motivates further exploration of information organization and inspires more principled MRL designs.

## REFERENCES

Kwangjun Ahn, Byron Xu, Natalie Abreu, Ying Fan, Gagik Magakyan, Pratyusha Sharma, Zheng Zhan, and John Langford. Dion: Distributed orthonormalized updates, 2025. URL https: //arxiv.org/abs/2504.05295.

Vera Boteva, Demian Gholipour, Artem Sokolov, and Stefan Riezler. A full-text learning to rank dataset for medical information retrieval. In European Conference on Information Retrieval, 2016. URL https://api.semanticscholar.org/CorpusID:14355670.

Inigo Casanueva, Tadas Tem˜ cinas, Daniela Gerz, Matthew Henderson, and Ivan Vuliˇ c. Efficient´ intent detection with dual sentence encoders. In Tsung-Hsien Wen, Asli Celikyilmaz, Zhou Yu, Alexandros Papangelis, Mihail Eric, Anuj Kumar, Inigo Casanueva, and Rushin Shah (eds.),˜ Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI, pp. 38–45, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.nlp4conva i-1.5. URL https://aclanthology.org/2020.nlp4convai-1.5.

Daniel Cer, Mona Diab, Eneko Agirre, I nigo Lopez-Gazpio, and Lucia Specia. SemEval-2017 task 1: Semantic textual similarity multilingual and crosslingual focused evaluation. In Steven Bethard, Marine Carpuat, Marianna Apidianaki, Saif M. Mohammad, Daniel Cer, and David Jurgens (eds.), Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017), pp. 1–14, Vancouver, Canada, August 2017a. Association for Computational Linguistics. doi: 10.18653/v1/S17-2001. URL https://aclanthology.org/S17-2001.

Daniel Cer, Mona Diab, Eneko Agirre, Inigo Lopez-Gazpio, and Lucia Specia. SemEval-2017 ˜ task 1: Semantic textual similarity multilingual and crosslingual focused evaluation. In Steven Bethard, Marine Carpuat, Marianna Apidianaki, Saif M. Mohammad, Daniel Cer, and David Jurgens (eds.), Proceedings ofthe 11th International Workshop on Semantic Evaluation (SemEval-2017), pp. 1–14, Vancouver, Canada, August 2017b. Association for Computational Linguistics. doi: 10.18653/v1/S17-2001. URL https://aclanthology.org/S17-2001/.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In Proceedings of the 37th International Conference on Machine Learning, ICML’20. JMLR.org, 2020.

Arman Cohan, Sergey Feldman, Iz Beltagy, Doug Downey, and Daniel Weld. SPECTER: Document-level representation learning using citation-informed transformers. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault (eds.), Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 2270–2282, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.207. URL https://aclanthology.org/2020.acl-main.207.

Jack FitzGerald, Christopher Hench, Charith Peris, Scott Mackie, Kay Rottmann, Ana Sanchez, Aaron Nash, Liam Urbach, Vishesh Kakarala, Richa Singh, Swetha Ranganath, Laurie Crist, Misha Britan, Wouter Leeuwis, Gokhan Tur, and Prem Natarajan. Massive: A 1m-example multilingual natural language understanding dataset with 51 typologically-diverse languages, 2022.

Luyu Gao, Yunyi Zhang, Jiawei Han, and Jamie Callan. Scaling deep contrastive learning batch size under memory limited setup. In Anna Rogers, Iacer Calixto, Ivan Vulic, Naomi Saphra, Nora ´ Kassner, Oana-Maria Camburu, Trapit Bansal, and Vered Shwartz (eds.), Proceedings of the 6th

Workshop on Representation Learning for NLP (RepL4NLP-2021), pp. 316–321, Online, August 2021a. Association for Computational Linguistics. doi: 10.18653/v1/2021.repl4nlp-1.31. URL https://aclanthology.org/2021.repl4nlp-1.31/.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. SimCSE: Simple contrastive learning of sentence embeddings. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (eds.), Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 6894–6910, Online and Punta Cana, Dominican Republic, November 2021b. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.552. URL https://aclanthology.org/2021.emnlp-main.552/.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Dang Nguyen Hong, Nhi Ngoc-Yen Nguyen, and Huy-Hieu Pham. Mic: Maximizing informational capacity in adaptive representations via isotropic subspace alignment, 2026. URL https://ar xiv.org/abs/2605.29987.

Alston S. Householder. Unitary triangularization of a nonsymmetric matrix. J. ACM, 5(4):339–342, October 1958. ISSN 0004-5411. doi: 10.1145/320941.320947. URL https://doi.org/10 .1145/320941.320947.

Xinshuo Hu, Zifei Shan, Xinping Zhao, Zetian Sun, Zhenyu Liu, Dongfang Li, Shaolin Ye, Xinyuan Wei, Qian Chen, Baotian Hu, Haofen Wang, Jun Yu, and Min Zhang. Kalm-embedding: Superior training data brings a stronger embedding model, 2025. URL https://arxiv.org/abs/ 2501.01028.

Phung Gia Huy, Hai An Vu, Minh-Phuc Truong, Thang Duc Tran, Linh Ngo Van, Thanh Hong Nguyen, and Trung Le. MIPIC: Matryoshka representation learning via self-distilled intrarelational and progressive information chaining. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens (eds.), Findings of the Association for Computational Linguistics: ACL 2026, pp. 13824–13838, San Diego, California, United States, July 2026. Association for Computational Linguistics. ISBN 979-8-89176-395-1. doi: 10.18653/v1/2026.findings-acl.676. URL https://aclanthology.org/2026.findings-acl.676/.

Jared Kaplan, Sam McCandlish, Thomas Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeff Wu, and Dario Amodei. Scaling laws for neural language models. ArXiv, abs/2001.08361, 2020. URL https://api.semanticscholar.org/Corpus ID:210861095.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu (eds.), Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6769–6781, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.550. URL https://aclanthology.org/2020.emnlp-main.550/.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2017. URL https://arxiv.org/abs/1412.6980.

Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham Kakade, Prateek Jain, et al. Matryoshka representation learning. In Proceedings of the 36th International Conference on Neural Information Processing Systems, pp. 30233–30249, 2022.

Ken Lang. Newsweeder: Learning to filter netnews. In Armand Prieditis and Stuart Russell (eds.), Machine Learning Proceedings 1995, pp. 331–339. Morgan Kaufmann, San Francisco (CA), 1995. ISBN 978-1-55860-377-6. doi: https://doi.org/10.1016/B978-1-55860-377-6.50048-7. URL https://www.sciencedirect.com/science/article/pii/B978155860 3776500487.

Xianming LI, Zongxi Li, Jing Li, Haoran Xie, and Qing Li. ESE: Espresso sentence embeddings. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=plgLA2YBLH.

Xueqing Liu, Chi Wang, Yue Leng, and ChengXiang Zhai. Linkso: a dataset for learning to retrieve similar question answer pairs on software development forums. Proceedings of the 4th ACM SIGSOFT International Workshop on NLP for Software Engineering, 2018. URL https://ap i.semanticscholar.org/CorpusID:53111679.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019. URL https: //arxiv.org/abs/1711.05101.

Marco Marelli, Stefano Menini, Marco Baroni, Luisa Bentivogli, Raffaella Bernardi, and Roberto Zamparelli. A SICK cure for the evaluation of compositional distributional semantic models. In Nicoletta Calzolari, Khalid Choukri, Thierry Declerck, Hrafn Loftsson, Bente Maegaard, Joseph Mariani, Asuncion Moreno, Jan Odijk, and Stelios Piperidis (eds.), Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC’14), pp. 216– 223, Reykjavik, Iceland, May 2014. European Language Resources Association (ELRA). URL http://www.lrec-conf.org/proceedings/lrec2014/pdf/363\_Paper.pdf.

Niklas Muennighoff, Nouamane Tazi, Lo¨ıc Magne, and Nils Reimers. Mteb: Massive text embed ding benchmark. arXiv preprint arXiv:2210.07316, 2022. doi: 10.48550/ARXIV.2210.07316. URL https://arxiv.org/abs/2210.07316.

Tri Nguyen, Mir Rosenberg, Xia Song, Jianfeng Gao, Saurabh Tiwary, Rangan Majumder, and Li Deng. MS MARCO: A human generated machine reading comprehension dataset. CoRR, abs/1611.09268, 2016. URL http://arxiv.org/abs/1611.09268.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Ed-¨ ward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library, 2019. URL https://arxiv.org/abs/1912.01703.

Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 3982– 3992, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1410. URL https://aclanthology.org/D19-1410/.

Elvis Saravia, Hsien-Chi Toby Liu, Yen-Hao Huang, Junlin Wu, and Yi-Shin Chen. CARER: Contextualized affect representations for emotion recognition. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (eds.), Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 3687–3697, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1404. URL https://aclanthology.org/D18-1404.

Darsh Shah, Tao Lei, Alessandro Moschitti, Salvatore Romeo, and Preslav Nakov. Adversarial domain adaptation for duplicate question detection. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (eds.), Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 1056–1063, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1131. URL https://aclanthology.org/D18-1131.

Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, and Sham M. Kakade. SOAP: Improving and stabilizing shampoo using adam for language modeling. In The Thirteenth International Conference on Learning Representations, 2025. URL https: //openreview.net/forum?id=IDxZhXrpNf.

Henning Wachsmuth, Shahbaz Syed, and Benno Stein. Retrieval of the best counterargument with out prior topic knowledge. In ACL, 2018.

David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. Fact or fiction: Verifying scientific claims. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 7534–7550, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emn lp-main.609. URL https://aclanthology.org/2020.emnlp-main.609.

Songhao Wu, Zhongxin Chen, Yuxuan Liu, Heng Cui, Cong Li, and Rui Yan. Your unembedding matrix is secretly a feature lens for text embeddings. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, KDD ’26, pp. 5454–5463, New York, NY, USA, 2026. Association for Computing Machinery. ISBN 9798400722592. doi: 10.1145/37 70855.3818204. URL https://doi.org/10.1145/3770855.3818204.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report, 2024. URL https://arxiv.org/abs/2407.10671.

Shubing Yang and Dongfang Zhao. Adawidth: Query-adaptive embedding width for dense retrieval, 2026. URL https://arxiv.org/abs/2608.23862.

Jinsung Yoon, Rajarishi Sinha, Sercan O Arik, and Tomas Pfister. Matryoshka-adaptor: Unsupervised and supervised tuning for smaller embedding dimensions. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 10318–10336, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.576. URL https://aclanthology.org/2024.emnlp-main.576/.

Biao Zhang, Lixin Chen, Tong Liu, and Bo Zheng. SMEC:rethinking matryoshka representation learning for retrieval embedding compression. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 26209–26222, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.1332. URL https://aclanthology.org/2025.emnl p-main.1332/.

## A IMPLEMENTATION DETAILS

## A.1 GENERAL SETUP

Training Setup. We conduct the main experiments reported in Table 1 using backbones from widely used model families, including Qwen3-0.6B-Base (Team, 2025), Qwen2-0.5B (Yang et al., 2024), and Llama-3.2-1B (Grattafiori et al., 2024). We further include Qwen3-4B-Base in the scal ing experiments to evaluate performance at a larger model scale. Unless otherwise specified, we instantiate P as mean pooling to derive the embeddings. For MRL training setup, we follow the setup of Hu et al. (2025) and summarize the configuration for each backbone in Table 13, where m denotes the i-th nested dimension and $w _ { i }$ denotes its corresponding loss weight. For brevity, τ denotes the compression ratio rounded to the nearest power of two throughout the tables.

We provide further details on our training setup. For data preparation, we use the publicly available KaLM-embedding-finetuning-data dataset<sup>3</sup> (Hu et al., 2025). By default, we train on a randomly sampled subset of 1,152,000 examples, approximately 1/5 of the full dataset, to keep the experimental cost manageable. Specifically, each training example consists of one query, one positive example, and seven hard negatives. We also use in-batch negatives across GPUs. For the long-run experiments in Section 4.3, we use the full dataset to evaluate performance under extended training.

Regarding the training hyperparameters, the per-device batch size is set to 96 queries, resulting in a global batch size of 384 across four GPUs. We adopt GradCache (Gao et al., 2021a) to reduce memory usage. The maximum sequence length is set to 512 tokens for both queries and passages. We use the AdamW optimizer (Kingma & Ba, 2017; Loshchilov & Hutter, 2019) with a learning rate of $1 0 ^ { - 4 }$ , a warmup ratio of 0.1 followed by linear decay, and a weight decay of 0.01. For contrastive learning, the temperature is set to 0.01 across all experimental settings. We initialize the entries of $W _ { \mathrm { P r o j } } \mathrm { \bar { f } r o m } \mathcal { N } ( 0 , 0 . 0 2 ^ { 2 } )$ and obtain the initial spectral basis via the eigenvectors of its Gram matrix. More implementation details and reproducibility information are provided in our codebase.

Evaluation. We conduct experiments on the widely used English MTEB benchmark (Muennighoff et al., 2022), which comprises 56 datasets spanning seven task categories: Semantic Textual Similarity (STS.), Classification (Class.), Clustering (Cluster.), Pair Classification (PairClass.), Reranking (Rerank.), Retrieval (Retr.), and Summarization (Sum.). For experiments presented in Figures 3 and 4, we report the average score over a 15-dataset evaluation subset detailed in Table 14.

For sentence encoding, both MRL and SpecMRL follow KaLM’s task-specific inference instructions (Hu et al., 2025). For SpecMRL, we perform one orthogonal iteration in FP32 using the final parameter Gram matrix, starting from the spectral basis saved in the checkpoint. The resulting ba sis remains fixed throughout evaluation. For Qwen2-0.5B, we use the same nested dimensions for training and evaluation, following KaLM (Hu et al., 2025). We use unnormalized embeddings for classification and apply $L _ { 2 }$ normalization at each target dimension for all other tasks.

## A.2 IMPLEMENTATION OF ESE AND MIPIC

To focus on dimensional compression and facilitate a controlled comparison with MRL and SpecMRL, we adapt ESE and MIPIC to use only the final-layer representations of Qwen3-0.6B. Method-specific hyperparameters are selected following the original papers (LI et al., 2025; Huy et al., 2026). The implementations of all baselines are also available in our code repository. Detailed evaluation results for all compared methods are presented in Table 18.

ESE (L2C). We retain the learn-to-compress objective and apply it to the final-layer sentence embeddings, omitting intermediate-layer supervision. Contrastive losses are applied to the full 1024- dimensional representation and its 128-dimensional prefix. The latter additionally receives alignment supervision from the compression targets. Representations at other dimensions are obtained by prefix truncation at evaluation time, without additional training losses at those dimensions.

MIPIC (SIA). We combine the MRL objective with SIA applied to final-layer token representations. The nested dimensions and MRL loss weights follow Table 13. Using mean-pooled sentence embeddings as anchors, SIA aligns token-importance distributions across dimensions through KL divergence. Linear CKA further aligns the relational structures of full-dimensional and truncated representations on token subsets selected by full-dimensional importance scores.

## A.3 IMPLEMENTATION OF THE GRAM-MATRIX ABLATION

In this ablated variant, we retain the same learnable projection factor $W _ { \mathrm { P r o j } }$ as in SpecMRL. For each nested dimension $m ,$ , we define $W _ { m } = \left. W _ { \mathrm { P r o j } } [ : m , : ] \right.$ and $G _ { m } = \mathbf { \bar { W } } _ { m } W _ { m } ^ { \top }$ , and let $\pmb { X } ^ { ( m ) } , ~ \pmb { Y } ^ { ( m ) } \in \mathbb { R } ^ { 1 \times m }$ denote the corresponding truncated backbone embeddings. To investigate whether the performance gains of SpecMRL arise solely from the introduction of additional learnable parameters, we directly apply $W _ { m }$ to the corresponding truncated backbone embeddings, without performing any spectral decomposition. This produces the resulting R-dimensional representations $X ^ { ( m ) } W _ { m }$ and $Y ^ { ( m ) } W _ { m }$ , whose cosine similarity can be computed implicitly as:

$$
\sin ( \pmb { X } ^ { ( m ) } \pmb { W } _ { m } , \pmb { Y } ^ { ( m ) } \pmb { W } _ { m } ) = \frac { \pmb { X } ^ { ( m ) } \pmb { G } _ { m } ( \pmb { Y } ^ { ( m ) } ) ^ { \top } } { \sqrt { \pmb { X } ^ { ( m ) } \pmb { G } _ { m } ( \pmb { X } ^ { ( m ) } ) ^ { \top } } \sqrt { \pmb { Y } ^ { ( m ) } \pmb { G } _ { m } ( \pmb { Y } ^ { ( m ) } ) ^ { \top } } } ,\tag{12}
$$

At evaluation, we compute the Cholesky factorization $G _ { m } = L _ { m } L _ { m } ^ { \top }$ for each target dimension. Each truncated embedding $X ^ { ( m ) }$ is mapped to the m-dimensional embedding $\pmb { X } ^ { ( m ) } \pmb { L } _ { m }$ . This preserves the inner products and norms of the lifted representations, and hence their cosine similarities.

## B COMPARISON WITH A LEARNABLE LINEAR READOUT

To assess whether a generic learnable readout can reproduce the gains of SpecMRL, we augment vanilla MRL with a $D \times D$ linear layer as the readout. We jointly train this layer with the backbone and denote the resulting as MRL w. Linear.

As shown in Table 11, although both readouts can integrate all original channels, MRL w. Linear underperforms SpecMRL at every evaluated dimension, with the gap widening as the dimension decreases. In principle, an unconstrained linear layer can represent the orthogonal transformation used

Table 11: Comparison with learnable readout.
<table><tr><td rowspan=1 colspan=1> $\tau$ </td><td rowspan=1 colspan=1>SpecMRL</td><td rowspan=1 colspan=1>MRL w. Linear</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>60.94</td><td rowspan=1 colspan=1>60.92 (−0.02)</td></tr><tr><td rowspan=2 colspan=1>24</td><td rowspan=1 colspan=1>61.61</td><td rowspan=4 colspan=1>60.72 (−0.89)60.14 (−1.14)59.13 (−1.33)57.03 (-1.50)</td></tr><tr><td rowspan=1 colspan=1>61.28</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>60.46</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>58.53</td></tr></table>

by SpecMRL. However, under the same MRL objective, this expressive capacity does not translate into comparable prefix quality, further supporting the value of SpecMRL’s ordered spectral basis.

## C PARAMETER SPACE REFLECTS EMBEDDING GEOMETRY

From a parametric perspective, each column of $W _ { \mathrm { P r o j } }$ can be interpreted as an embedding-space vector, making the weight matrix structurally analogous to a cached collection of embeddings. We contend that $W _ { \mathrm { P r o j } }$ encodes high-level semantic information, such that their collective organization in parameter space reflects the geometry of the learned embedding space during training.

We present our empirical observations to substantiate this structural correspondence. Specifically, we randomly sample $N = 1 0 ^ { 6 }$ sentences from the training set and encode them separately using MRL and SpecMRL to obtain calibration embeddings $\hat { \mathcal { E } } _ { \mathrm { c a l } } \doteq \mathbb { R } ^ { N \times D }$ . Using these embeddings as a reference, we extract their principal directions $U _ { \hat { \varepsilon } }$ and compare the resulting leading k-dimensional subspace with the corresponding readout subspace using principal-angle overlap:

$$
S _ { k } = \frac { 1 } { k } \left\| \left[ \left( U _ { \hat { \varepsilon } } \right) _ { : , 1 : k } \right] ^ { \top } U _ { : , 1 : k } \right\| _ { F } ^ { 2 } ,\tag{13}
$$

where U denotes the readout basis of the model. The score lies in [0, 1], with higher values indicating greater alignment. For random k-dimensional subspaces in $\mathbb { R } ^ { D }$ , the expected overlap is $k / D$

As shown in Table 12, both the overlap scores of MRL and SpecMRL exceed the random baseline, suggesting that the MRL objective encourages alignment between readout subspaces and embedding geometry. Viewed as a collection of embedding vectors, the $D \times R$ parameter matrix can be interpreted as a learned proxy for the full population of sentence embeddings, fitting their geometry during training. Compared with the fixed coordinate basis used by MRL, the spectral readout subspaces of SpecMRL exhibit stronger alignment with the dominant subspaces of sentence embeddings. This closer alignment may help preserve informative embedding structure under dimensional readout, thereby partially explaining the improved low-dimensional performance of SpecMRL.

Table 12: Readout subspace alignment.
<table><tr><td>k</td><td>Rand</td><td>MRL</td><td>SpecMRL</td></tr><tr><td>64</td><td>0.0628</td><td>0.2522</td><td>0.3025 (+ 5.03%)</td></tr><tr><td>128</td><td>0.1237</td><td>0.2783</td><td>0.4295 (+ 15.12%)</td></tr><tr><td>256</td><td>0.2513</td><td>0.3171</td><td>0.4715 (+ 15.44%)</td></tr><tr><td>512</td><td>0.4999</td><td>0.5197</td><td>0.5518 (+ 3.21%)</td></tr></table>

For SpecMRL, the readout basis used is obtained by power iteration, following the procedure described in Appendix A. We further verify this approximation against the ordered eigenbasis computed directly from the same Gram matrix. At the final checkpoint, their subspace overlap exceeds 99.9% for all evaluated k, indicating that the two bases span nearly identical leading subspaces.

## D ENERGY CONCENTRATION IN THE TAILING SPECTRAL SUBSPACE

We conduct further analyses to investigate why dropping the final spectral bands can improve SpecMRL’s downstream performance. Motivated by the analysis in (Wu et al., 2026), we examine the behavior of SpecMRL-derived representations when projected onto different spectral bands.

Specifically, we first partition the spectrum into 20 consecutive bands ordered by eigenvalue. We then sample $N = 1 0 ^ { \bar { 6 } }$ passages from the MS MARCO corpus (Nguyen et al., 2016) to obtain the calibration embeddings. Let $z _ { i j } = \mathbf { u } _ { i } ^ { \top } X _ { [ i ] }$ denote the projection of $\mathbf { X } _ { [ i ] }$ onto the j-th spectral direction. The average energy fraction of band $B _ { b }$ over the N calibration embeddings is:

$$
E _ { b } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { \sum _ { j \in B _ { b } } z _ { i j } ^ { 2 } } { \sum _ { j = 1 } ^ { 1 0 2 4 } z _ { i j } ^ { 2 } } .\tag{14}
$$

Figure 5 shows that the final band accounts for a large fraction of energy (21.72%), despite spanning only 5% of the spectral directions. Ideally, the energy should decay smoothly across the ordered spectral bands, with the first band accounting for the largest share, consistent with its intended role in capturing the most informative semantic directions in MRL. The decay pattern reflects the decreasing semantic importance of later spectral bands, while the smoothness helps maintain isotropy by avoiding concentration in a small subset of directions. In turn, the disproportionate concentration in the final band indicates increased anisotropy, suggesting a departure from the expected geometry. Similar observations have also been reported by Wu et al. (2026). Although the cause of this phenomenon remains unclear, we hope this analysis provides an initial explanation for the observed improvement and motivates further investigation.

![](images/b180c22a71d5b38cdc6b7b9ac7d05cc4423362ed73f5afa1c13853df94f994e7.jpg)  
Figure 5: Average projection energy fractions.

## E COMPUTATIONAL COMPLEXITY

This section provides a supplementary analysis of SpecMRL’s computational overhead relative to MRL. The additional computations consist of Gram matrix construction and orthogonal basis update, followed by projecting a batch of $B _ { \mathrm { p r o j } }$ embeddings of dimension D. As shown in Fig. 2, the orthogonal basis update involves only one matrix multiplication and one QR factorization (Householder, 1958). Counting a multiply-add as two FLOPs, the matrix multiplication requires $2 D ^ { 3 }$ FLOPs, while QR factorization costs ${ \scriptstyle { \frac { 8 } { 3 } } } D ^ { 3 }$ FLOPs in total: ${ \scriptstyle { \frac { 4 } { 3 } } } D ^ { 3 }$ for factorization and another ${ \scriptstyle { \frac { 4 } { 3 } } } D ^ { 3 }$ for explicitly forming the full orthogonal matrix. The additional forward cost is therefore given as:

$$
F _ { \mathrm { e x t r a } } \approx \underbrace { 2 R D ^ { 2 } } _ { \mathrm { G r a m \ c o n s t r u c t i o n } } + \underbrace { \frac { 1 4 } { 3 } D ^ { 3 } } _ { \mathrm { b a s i s \ u p d a t e } } + \underbrace { 2 B _ { \mathrm { p r o j } } D ^ { 2 } } _ { \mathrm { e m b e d d i n g \ p r o j e c t i o n } } .\tag{15}
$$

For comparison, let P denote the number of linear weights, and T the number of tokens processed per device. Following the approximation in (Kaplan et al., 2020), the backbone forward cost is:

$$
F _ { \mathrm { b a c k b o n e } } \approx 2 T P .\tag{16}
$$

Substituting the Qwen3-0.6B model configuration, the additional forward cost remains below 1% of the backbone forward FLOPs, demonstrating the low overhead of our power-iteration design. This computational efficiency is further validated by the empirical results presented in Section 4.1.3.

Table 13: MRL training configurations for different backbone models.
<table><tr><td>一</td><td colspan="7">MRL Setup</td></tr><tr><td>Checkpoint [HF]</td><td> $m _ { 0 } \mathrm { ~ } ( w _ { 0 } )$ </td><td>m1 (w1)</td><td></td><td>m2 (w2)</td><td></td><td> $m _ { 3 } \ ( w _ { 3 } )$ </td><td> $m _ { 4 } \ ( w _ { 4 } )$ </td></tr><tr><td>Qwen3-0.6B-Base 2</td><td>1024 (1.0)</td><td>512</td><td>(0.3)</td><td>256</td><td>(0.2)</td><td>128 (0.1)</td><td>64 (0.1)</td></tr><tr><td>Qwen2-0.5B</td><td>896 (1.0)</td><td>512</td><td>(0.3)</td><td>256</td><td>(0.2)</td><td>128 (0.1)</td><td>64 (0.1)</td></tr><tr><td>Llama-3.2-1B</td><td>2048 (1.0)</td><td>1024</td><td>(0.3)</td><td>512</td><td>(0.2)</td><td>256 (0.1)</td><td>128 (0.1)</td></tr><tr><td>Qwen3-4B-Base</td><td>2560 (1.0)</td><td>1024</td><td>(0.3)</td><td>512</td><td>(0.2)</td><td>256 (0.1)</td><td>128 (0.1)</td></tr></table>

Table 14: MTEB evaluation metrics and the 15-dataset subset used for scaling experiments.
<table><tr><td>Category</td><td>Metric</td><td>Dataset</td></tr><tr><td rowspan="3">STS. (3)</td><td rowspan="3">Spearman</td><td>STS17 (Cer et al., 2017a)</td></tr><tr><td>SICK-R (Marelli et al., 2014)</td></tr><tr><td>STSBenchmark (Cer et al., 2017b)</td></tr><tr><td>Class. (3)</td><td>Accuracy</td><td>Banking77Classification (Casanueva et al., 2020) EmotionClassification (Saravia et al., 2018) MassiveIntentClassification (FitzGerald et al., 2022)</td></tr><tr><td rowspan="2">Cluster. (3)</td><td rowspan="2">V-measure</td><td>BiorxivClusteringS2S (Muennighoff et al., 2022)</td></tr><tr><td>MedrxivClusteringS2S (Muennighoff et al., 2022)</td></tr><tr><td>PairClass. (1)</td><td>AP</td><td>TwentyNewsgroupsClustering (Lang, 1995) SprintDuplicateQuestions (Shah et al., 2018)</td></tr><tr><td>Rerank. (2)</td><td>MAP</td><td>SciDocsRR (Cohan et al., 2020)</td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td>StackOverflowDupQuestions (Liu et al., 2018)</td></tr><tr><td></td></tr><tr><td rowspan="2">Retr. (3)</td><td rowspan="2">nDCG@10</td><td>SciFact (Wadden et al., 2020)</td></tr><tr><td>ArguAna (Wachsmuth et al., 2018) NFCorpus (Boteva et al., 2016)</td></tr></table>

Table 15: Task-level breakdown of MTEB results in Table 1. Shaded rows denote SpecMRL, bold values indicate better overall results, parenthetical values report gains over MRL under each setup.
<table><tr><td></td><td>STS.</td><td>Class.</td><td>Cluster.</td><td>PairClass.</td><td>Rerank.</td><td>Retr.</td><td>Sum.</td><td>Avg</td></tr><tr><td># Datasets 10</td><td>12</td><td></td><td>11</td><td>3</td><td>4</td><td>15</td><td>1</td><td>56</td></tr><tr><td colspan="9">Qwen3-0.6B</td></tr><tr><td rowspan="2">1024 (τ = 1)</td><td>80.77</td><td>74.25</td><td>48.72</td><td>72.74</td><td>48.93</td><td>49.02</td><td>30.38</td><td>60.97</td></tr><tr><td>80.24</td><td>74.45</td><td>49.06</td><td>72.44</td><td>47.67</td><td>49.30</td><td>29.66</td><td>60.94 (−0.03)</td></tr><tr><td rowspan="2">512 (τ = 2)</td><td>80.87</td><td>73.99</td><td>48.43</td><td>72.09</td><td>48.25</td><td>48.10</td><td>29.92</td><td>60.54</td></tr><tr><td>80.83</td><td>75.11</td><td>49.43</td><td>74.49</td><td>50.98</td><td>49.24</td><td>30.46</td><td>61.61 (+1.07)</td></tr><tr><td rowspan="2">256 (τ = 4)</td><td>80.21</td><td>73.45</td><td>47.89</td><td>71.42</td><td>47.11</td><td>45.62</td><td>29.00</td><td>59.40</td></tr><tr><td>81.04</td><td>74.98</td><td>49.44</td><td>74.46</td><td>50.75</td><td>48.08</td><td>30.30</td><td>61.28 (+1.88)</td></tr><tr><td rowspan="2">128 (τ = 8)</td><td>79.49</td><td>72.99</td><td>47.43</td><td>68.29</td><td>45.47</td><td>42.18</td><td>29.24</td><td>57.88</td></tr><tr><td>80.76</td><td>74.56</td><td>49.28</td><td>73.90</td><td>49.95</td><td>45.93</td><td>30.74</td><td>60.46 (+2.58)</td></tr><tr><td rowspan="2">64 (τ = 16)</td><td>78.09</td><td>72.11</td><td>46.54</td><td>63.87</td><td>42.52</td><td>34.38</td><td>29.63</td><td>54.73</td></tr><tr><td>80.18</td><td>73.89</td><td>48.89</td><td>72.43</td><td>48.07</td><td>40.79</td><td>29.86</td><td>58.53 (+3.80)</td></tr><tr><td colspan="9">Qwen2-0.5B</td></tr><tr><td rowspan="2">896 (τ = 1)</td><td>78.57</td><td>71.82</td><td>46.99</td><td>71.52</td><td>46.68</td><td>47.04</td><td>30.59</td><td>58.96</td></tr><tr><td>78.72</td><td>71.80</td><td>46.59</td><td>71.33</td><td>46.24</td><td>46.75</td><td>27.87</td><td>58.74 (−0.22)</td></tr><tr><td rowspan="2">512 (τ = 2)</td><td>78.49</td><td>72.10</td><td>46.80</td><td>70.91</td><td>46.13</td><td>46.14</td><td>31.13</td><td>58.67</td></tr><tr><td>79.25</td><td>73.07</td><td>47.05</td><td>73.22</td><td>48.98</td><td>46.51</td><td>28.31</td><td>59.44 (+0.77)</td></tr><tr><td rowspan="2">256 (τ = 4)</td><td>78.41</td><td>72.39</td><td>46.82</td><td>71.23</td><td>46.49</td><td>44.57</td><td>30.18</td><td>58.32</td></tr><tr><td>79.40</td><td>72.94</td><td>47.11</td><td>73.07</td><td>48.85</td><td>45.64</td><td>28.37</td><td>59.20 (+0.88)</td></tr><tr><td rowspan="2">128 (τ = 8)</td><td>77.71</td><td>72.26</td><td>46.68</td><td>69.69</td><td>45.03</td><td>41.94</td><td>30.67</td><td>57.26</td></tr><tr><td>79.29</td><td>72.79</td><td>46.96</td><td>72.25</td><td>48.53</td><td>43.34</td><td>28.52</td><td>58.44 (+1.18)</td></tr><tr><td rowspan="2">64 (τ = 16)</td><td>76.41</td><td>71.86</td><td>46.42</td><td>67.69</td><td>43.74</td><td>36.48</td><td>30.47</td><td>55.23</td></tr><tr><td>78.33</td><td>72.37</td><td>46.72</td><td>70.00</td><td>46.55</td><td>38.72</td><td>27.59</td><td>56.61 (+1.38)</td></tr><tr><td colspan="9"></td></tr><tr><td rowspan="2">2048 (τ = 1)</td><td>80.91</td><td>75.77</td><td>50.76</td><td>75.43</td><td>Llama3.2-1B 47.71</td><td>49.60</td><td>29.64</td><td>61.92</td></tr><tr><td>80.71</td><td>76.05</td><td>50.28</td><td>76.02</td><td>48.19</td><td>49.39</td><td>30.18</td><td>61.87 (-0.05)</td></tr><tr><td rowspan="2">1024 (τ = 2)</td><td>80.79</td><td>75.84</td><td>50.75</td><td>75.22</td><td>46.97</td><td>49.17</td><td>30.06</td><td>61.74</td></tr><tr><td>81.09</td><td>76.61</td><td>50.60</td><td>76.70</td><td>51.27</td><td>49.53</td><td>30.35</td><td>62.42 (+0.68)</td></tr><tr><td rowspan="2">512 (τ = 4)</td><td>80.55</td><td>75.73</td><td>50.64</td><td>74.77</td><td>46.63</td><td>48.30</td><td>29.40</td><td>61.36</td></tr><tr><td>81.22</td><td>76.59</td><td>50.67</td><td>76.62</td><td>51.06</td><td>48.80</td><td>30.05</td><td>62.23 (+0.87)</td></tr><tr><td rowspan="2">256 (τ = 8)</td><td>80.31</td><td>75.40</td><td>50.49</td><td>74.00</td><td>45.30</td><td>46.59</td><td>29.54</td><td>60.62</td></tr><tr><td>81.28</td><td>76.52</td><td>50.53</td><td>76.29</td><td>50.57</td><td>47.52</td><td>30.19</td><td>61.80 (+1.18)</td></tr><tr><td rowspan="2">128 (τ = 16)</td><td>79.88</td><td>74.79</td><td>50.34</td><td>73.20</td><td>44.88</td><td>43.21</td><td>30.29</td><td>59.42</td></tr><tr><td>81.32</td><td>76.28</td><td>50.48</td><td>75.40</td><td>49.69</td><td>44.74</td><td>30.91</td><td>60.91 (+1.49)</td></tr></table>

Table 16: Task-level breakdown of FeatMRL variants. Bold and underlined average scores indicate the best and second-best FeatMRL variants within each group.
<table><tr><td></td><td>STS. Class.</td><td>Cluster.</td><td></td><td>PairClass.</td><td>Rerank. Retr.</td><td>Sum.</td><td>Avg</td></tr><tr><td># Datasets</td><td>10</td><td>12</td><td>11</td><td>3</td><td>4 15</td><td>1</td><td>56</td></tr><tr><td>τ = 1</td><td>80.24</td><td>74.45</td><td>49.06</td><td>72.44</td><td>47.67 49.30</td><td>29.66</td><td>60.94</td></tr><tr><td>β= 0</td><td>80.70</td><td>73.08</td><td>48.86</td><td>72.31</td><td>49.35 48.99</td><td>29.80</td><td>60.72 (−0.22)</td></tr><tr><td>β= 0.1</td><td>80.73</td><td>72.54</td><td>48.77</td><td>72.27 49.82</td><td>49.10</td><td>31.06</td><td>60.68 (−0.26)</td></tr><tr><td>β = 0.5</td><td>80.65</td><td>72.30</td><td>48.79</td><td>72.87</td><td>49.54 49.29</td><td>31.65</td><td>60.69 (-0.25)</td></tr><tr><td>β = 0.9</td><td>80.78</td><td>71.97</td><td>48.85</td><td>72.37 48.71</td><td>48.72</td><td>29.69</td><td>60.38 (–0.56)</td></tr><tr><td>τ = 2</td><td>80.83</td><td>75.11</td><td>49.43</td><td>74.49</td><td>50.98 49.24</td><td>30.46</td><td>61.61</td></tr><tr><td>β=0</td><td>80.85</td><td>72.86</td><td>48.79</td><td>71.78</td><td>48.62 48.43</td><td>29.41</td><td>60.45 (-1.16)</td></tr><tr><td>β = 0.1</td><td>80.86</td><td>72.37</td><td>48.68</td><td>71.72 49.22</td><td>48.31</td><td>30.96</td><td>60.36 (−1.25)</td></tr><tr><td>β = 0.5</td><td>80.79</td><td>71.98</td><td>48.56</td><td>72.45 48.96</td><td>48.85</td><td>31.85</td><td>60.42 (-1.19)</td></tr><tr><td>β = 0.9</td><td>80.97</td><td>71.69</td><td>48.69</td><td>71.77 47.89</td><td>47.92</td><td>29.47</td><td>60.01 (-1.60)</td></tr><tr><td>τ = 4</td><td>81.04</td><td>74.98</td><td>49.44</td><td>74.46</td><td>50.75 48.08</td><td>30.30</td><td>61.28</td></tr><tr><td>β= 0</td><td>80.98</td><td>72.73</td><td>48.66</td><td>70.87</td><td>47.60 47.28</td><td>29.31</td><td>59.99 (−1.29)</td></tr><tr><td>β = 0.1</td><td>80.98</td><td>71.98</td><td>48.46</td><td>70.98 48.32</td><td>47.20</td><td>29.82</td><td>59.84 (-1.44)</td></tr><tr><td>β = 0.5</td><td>80.79</td><td>71.69</td><td>48.51</td><td>71.68 48.02</td><td>47.66</td><td>31.25</td><td>59.91 (-1.37)</td></tr><tr><td>β = 0.9</td><td>80.99</td><td>71.32</td><td>48.48</td><td>70.76 46.82</td><td>46.34</td><td>30.29</td><td>59.35 (-1.93)</td></tr><tr><td>τ = 8</td><td>80.76</td><td>74.56</td><td>49.28</td><td>73.90</td><td>49.95 45.93</td><td>30.74</td><td>60.46</td></tr><tr><td>β=0</td><td>80.54</td><td>72.27</td><td>48.47</td><td>68.55</td><td>45.75 44.56</td><td>30.96</td><td>58.82 (-1.64)</td></tr><tr><td>β = 0.1</td><td>80.49</td><td>71.67</td><td>48.17</td><td>68.77</td><td>46.44 44.58</td><td>30.41</td><td>58.68 (-1.78)</td></tr><tr><td>β = 0.5</td><td>80.39</td><td>71.21</td><td>48.22</td><td>69.37</td><td>46.24 45.12</td><td>31.54</td><td>58.75 (-1.71)</td></tr><tr><td>β = 0.9</td><td>80.45</td><td>70.89</td><td>48.24</td><td>68.63</td><td>44.96 43.54</td><td>31.35</td><td>58.14 (-2.32)</td></tr><tr><td>τ = 16</td><td>80.18</td><td>73.89</td><td>48.89</td><td>72.43</td><td>48.07 40.79</td><td>29.86</td><td>58.53</td></tr><tr><td>β=0</td><td>78.85</td><td>71.60</td><td>47.86</td><td>63.11</td><td>42.51 37.47</td><td>30.86</td><td>55.83 (-2.70)</td></tr><tr><td>β = 0.1</td><td>78.73</td><td>70.95</td><td>47.58</td><td>63.54</td><td>43.23 37.26</td><td>29.82</td><td>55.62 (−2.91)</td></tr><tr><td>β = 0.5</td><td>78.71</td><td>70.78</td><td>47.76</td><td>63.82</td><td>43.15 38.35</td><td>29.59</td><td>55.90 (-2.63)</td></tr><tr><td>β = 0.9</td><td>78.82</td><td>70.28</td><td>47.79</td><td>63.63</td><td>41.91 36.18</td><td>30.12</td><td>55.15 (–3.38)</td></tr></table>

Table 17: Task-level breakdown of MTEB in an over-training regime on Qwen3-0.6B, with training steps extended from 3K to 15K. The improvement of SpecMRL remains consistent with Table 1, suggesting that the gains of SpecMRL persist beyond faster convergence.
<table><tr><td></td><td>STS.</td><td>Class.</td><td>Cluster.</td><td>PairClass.</td><td>Rerank.</td><td>Retr.</td><td>Sum.</td><td>Avg</td></tr><tr><td># Datasets</td><td>10</td><td>12</td><td>11</td><td>3</td><td>4</td><td>15</td><td>1</td><td>56</td></tr><tr><td rowspan="2">1024 (τ = 1)</td><td>80.52</td><td>81.40</td><td>52.79</td><td>70.19</td><td>48.99</td><td>50.69</td><td>28.91</td><td>63.54</td></tr><tr><td>80.74</td><td>81.15</td><td>52.36</td><td>68.12</td><td>47.86</td><td>50.59</td><td>30.38</td><td>63.25 (-0.29)</td></tr><tr><td rowspan="2">512 (τ = 2)</td><td>80.53</td><td>81.34</td><td>52.74</td><td>69.44</td><td>48.43</td><td>49.80</td><td>28.28</td><td>63.19</td></tr><tr><td>81.35</td><td>81.50</td><td>52.64</td><td>71.39</td><td>50.95</td><td>50.59</td><td>29.75</td><td>63.88 (+0.69)</td></tr><tr><td rowspan="2">256 (τ = 4)</td><td>80.20</td><td>81.12</td><td>52.49</td><td>68.01</td><td>46.83</td><td>47.52</td><td>27.82</td><td>62.23</td></tr><tr><td>81.55</td><td>81.41</td><td>52.60</td><td>71.44</td><td>50.85</td><td>49.69</td><td>29.50</td><td>63.64 (+1.41)</td></tr><tr><td rowspan="2">128  $( \tau = 8 )$ </td><td>79.50</td><td>80.91</td><td>52.28</td><td>64.48</td><td>45.29</td><td>44.91</td><td>28.53</td><td>61.03</td></tr><tr><td>81.23</td><td>81.09</td><td>52.52</td><td>70.75</td><td>50.42</td><td>47.33</td><td>28.13</td><td>62.77 (+1.74)</td></tr><tr><td rowspan="2">64  $( \tau = 1 6 )$ </td><td>78.21</td><td>80.29</td><td>51.88</td><td>61.40</td><td>43.08</td><td>39.03</td><td>28.71</td><td>58.69</td></tr><tr><td>80.45</td><td>80.23</td><td>52.28</td><td>69.22</td><td>49.26</td><td>42.12</td><td>28.85</td><td>60.85 (+2.16)</td></tr></table>

Table 18: Task-level breakdown of MTEB results for SpecMRL and existing MRL variants. Column $\operatorname { A v g } _ { \mathrm { w / o } \ \mathrm { C l s } }$ denotes the average MTEB results over non-classification MTEB datasets.
<table><tr><td></td><td>Method</td><td>STS.</td><td>Class.</td><td>Cluster.</td><td>PairClass.</td><td>Rerank.</td><td>Retr.</td><td>Sum.</td><td> $\mathrm { A v g } _ { \mathrm { w / o C l s } }$ </td></tr><tr><td rowspan="4">1024  $( \tau = { \bf 1 } )$ </td><td>MRL</td><td>80.77</td><td>74.25</td><td>48.72</td><td>72.74</td><td>48.93</td><td>49.02</td><td>30.38</td><td>57.35</td></tr><tr><td>ESE</td><td>81.12</td><td>59.88</td><td>48.61</td><td>70.96</td><td>49.12</td><td>49.49</td><td>29.93</td><td>57.44 (+0.10)</td></tr><tr><td>MIPIC</td><td>79.88</td><td>73.05</td><td>48.75</td><td>69.69</td><td>48.51</td><td>49.21</td><td>30.77</td><td>56.98 (−0.37)</td></tr><tr><td>SpecMRL</td><td>80.24</td><td>74.45</td><td>49.06</td><td>72.44</td><td>47.67</td><td>49.30</td><td>29.66</td><td>57.26 (-0.09)</td></tr><tr><td rowspan="4">512  ${ \bf \rho } ( \tau = { \bf 2 } )$ </td><td>MRL</td><td>80.87</td><td>73.99</td><td>48.43</td><td>72.09</td><td>48.25</td><td>48.10</td><td>29.92</td><td>56.87</td></tr><tr><td>ESE</td><td>80.92</td><td>59.12</td><td>48.47</td><td>69.47</td><td>48.49</td><td>48.23</td><td>28.85</td><td>56.75 (−0.12)</td></tr><tr><td>MIPIC</td><td>80.02</td><td>72.82</td><td>48.56</td><td>68.49</td><td>47.94</td><td>48.03</td><td>30.12</td><td>56.41 (−0.45)</td></tr><tr><td>SpecMRL</td><td>80.83</td><td>75.11</td><td>49.43</td><td>74.49</td><td>50.98</td><td>49.24</td><td>30.46</td><td>57.92 (+1.05)</td></tr><tr><td rowspan="4">256  $( \tau = 4 )$ </td><td>MRL</td><td>80.21</td><td>73.45</td><td>47.89</td><td>71.42</td><td>47.11</td><td>45.62</td><td>29.00</td><td>55.57</td></tr><tr><td>ESE</td><td>80.40</td><td>58.10</td><td>48.17</td><td>71.04</td><td>48.07</td><td>46.06</td><td>30.02</td><td>55.91 (+0.35)</td></tr><tr><td>MIPIC</td><td>79.87</td><td>72.47</td><td>48.23</td><td>67.35</td><td>46.39</td><td>46.03</td><td>31.57</td><td>55.43 (−0.14)</td></tr><tr><td>SpecMRL</td><td>81.04</td><td>74.98</td><td>49.44</td><td>74.46</td><td>50.75</td><td>48.08</td><td>30.30</td><td>57.55 (+1.98)</td></tr><tr><td rowspan="4">128  $( \tau = \mathbf { 8 } )$ </td><td>MRL</td><td>79.49</td><td>72.99</td><td>47.43</td><td>68.29</td><td>45.47</td><td>42.18</td><td>29.24</td><td>53.76</td></tr><tr><td>ESE</td><td>79.79</td><td>57.19</td><td>47.81</td><td>68.43</td><td>47.33</td><td>43.76</td><td>29.12</td><td>54.63 (+0.88)</td></tr><tr><td>MIPIC</td><td>79.36</td><td>72.07</td><td>47.96</td><td>63.52</td><td>44.77</td><td>43.13</td><td>30.51</td><td>53.82 (+0.07)</td></tr><tr><td>SpecMRL</td><td>80.76</td><td>74.56</td><td>49.28</td><td>73.90</td><td>49.95</td><td>45.93</td><td>30.74</td><td>56.61 (+2.85)</td></tr><tr><td rowspan="4">64  $\mathbf { \Psi } ( \tau = \mathbf { 1 6 } )$ </td><td>MRL</td><td>78.09</td><td>72.11</td><td>46.54</td><td>63.87</td><td>42.52</td><td>34.38</td><td>29.63</td><td>50.00</td></tr><tr><td>ESE</td><td>77.59</td><td>54.63</td><td>46.45</td><td>61.87</td><td>43.13</td><td>32.43</td><td>27.75</td><td>49.07 (−0.92)</td></tr><tr><td>MIPIC</td><td>78.08</td><td>71.20</td><td>47.20</td><td>59.58</td><td>42.22</td><td>36.88</td><td>29.96</td><td>50.70 (+0.70)</td></tr><tr><td>SpecMRL</td><td>80.18</td><td>73.89</td><td>48.89</td><td>72.43</td><td>48.07</td><td>40.79</td><td>29.86</td><td>54.34 (+4.34)</td></tr></table>