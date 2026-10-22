# A KAN-based Non-linear Factorization for Embedding Layer Compression

Oliver Giudice, Anna Russo Russo

Applied Research Team, IT Dept. Banca d’Italia \*− oliver.giudice@bancaditalia.it, anna.russorusso@bancaditalia.it

## Abstract

Compressing the embedding layer in Natural Language Processing (NLP) models is a critical yet challenging problem: embeddings often account for a substantial fraction of the model parameters while being highly sensitive to information loss. On the one hand effective compression can significantly reduce memory footprint and improve deployment efficiency, but on the other hand even small distortions in the embedding space may propagate through the entire neural network thus degrading downstream performance.

Among existing embedding compression approaches, low-rank factorization provides a simple and mathematically principled solution by approximating the embedding matrix with a reduced-rank decomposition. However, its practical effectiveness is often limited, as the resulting linear bottleneck constrains representational capacity. To overcome these limitations, we propose an enhanced factorization scheme that increases expressive power by introducing learnable non-linear projections within the compressed embedding layer. Our method is inspired by the Kolmogorov–Arnold Networks (KAN) paradigm, which replaces fixed linear transformations with learnable univariate functions. Despite promising results in other domains, applications of KANs in NLP remain scarce and have so far been largely limited to final or task-specific layers, rather than core representational components such as embedding layers.

We evaluate the proposed approach by integrating the compressed embedding layer into the Skip-gram framework and comparing it against a standard low-rank baseline under matched parameter budgets on the Word Analogy and Semantic Similarity benchmarks. We further introduce an effective initialization strategy from a pre-trained, uncompressed embedding layer. The behavior of the model is analyzed both immediately after initialization and after a brief post-initialization refinement phase, considering training with the standard Skip-gram objective as well as variants augmented with distillation from the original embedding layer. Overall, our results show that augmenting low-rank factorization with learnable non-linear projections effectively mitigates the limitations of linear bottlenecks, yielding more faithful compressed embeddings under fixed dimensionality and suggesting a promising direction for embedding layer compression beyond purely linear methods

Keywords: Natural Language Processing, Embedding Compression, Low-Rank Factorization, Kolmogorov-Arnold Networks

## 1. Introduction

Word embeddings have become an essential building block of modern Natural Language Processing (NLP), enabling the representation of words as dense vectors that capture semantic and syntactic relationships. However, as vocabulary sizes grow, the parameter count and memory footprint of standard embedding layers become a significant bottleneck, limiting scalability and hindering deployment in memory-constrained environments. To address these challenges, a broad family of compressed embedding techniques has been developed, among which low-rank factorization (Lan et al., 2019; Lioutas et al., 2020), hashing (Chen et al., 2015), quantization (Shi and Yu, 2018; Shu and Nakayama, 2017), distillation learning (Gao et al., 2023), pruning or weight sharing strategies (Chen et al., 2018; Suzuki and Nagata, 2016) and others (Bałazy et al., 2021). Within this landscape, low-rank factorization provides a principled and widely used baseline, being broadly adopted in NLP and other machine learning settings beyond embedding compression. In the standard approach, the factorization is strictly linear, as it decomposes the embedding matrix into two smaller matrices.

In this work we propose a novel low-rank factorization technique based on Kolmogorov–Arnold Networks (KANs). KANs, recently popularized by Liu et al. (2024b), are neural architectures that employ learnable, non-linear, univariate activation functions. Compared to traditional Multi-Layer Perceptrons (MLPs), this architecture appears to exhibit more favorable scalinglaw constants and has been empirically shown to achieve higher accuracy and improved interpretability at a fixed parameter budget. The proposed factorization technique aims at overcoming the implicit linear bottleneck of the standard low-rank factorization by replacing the second linear projection with a KAN-based mapping in which each output dimension is computed as a sum of learned non-linear transformations of the individual input coordinates.

To the best of our knowledge, KAN-style learnable activation functions have not been explored as a mechanism for embedding compression. The low-rank factorization setting provides a clean way to isolate their contribution. To this end, we introduce a KAN-based factorized embedding layer along with two initialization schemes and we evaluate it within the Skipgram framework against a standard low-rank factorization.

The paper is structured as follows: Section 2. discusses related work on KANs, while Section 3. provides preliminary background and notation; Section 4. explains our original KAN-based factorization technique. The experimental framework used for evaluation is described in Section 5., while the experimental results are reported and discussed in Section 6.. Lastly, Section 7. offers final considerations and future research directions for the employment of KANs in NLP.

## 2. Related Work

Since their introduction, KANs have been explored across a wide range of domains, including time-series (Han et al., 2025; Vaca-Rubio et al., 2024; Xu et al., 2024), computer vision (Cheon, 2024; Yu et al., 2025), reinforcement learning (Guo et al., 2024; Kich et al., 2024), speech (Xu et al., 2025a), and physics-informed modeling (Howard et al., 2024; So and Yung, 2024; Wang et al., 2025), often as replacements for fully-connected blocks within standard architectures (e.g., CNNs (Bodner et al., 2025), RNNs/LSTMs (Genet and Inzirillo, 2024b), GNNs (Xu et al., 2025b; Zhang and Zhang, 2024), Transformers (Genet and Inzirillo, 2024a; Yang and Wang, 2024), and U-Nets (Li et al., 2024)). A substantial body of follow-up work has focused on modifying the functional form of the learnable univariate components, which are implemented as spline-based parametrizations in the original formulation. Among the proposed replacement, we mention rational functions (Aghaei, 2024), radial basis functions (Li, 2024), sinusoidal functions (Reinhardt et al., 2025), Rectified Linear Units (Qiu et al., 2024).Alternative aggregation functions in place of sums have also been tested (Altarabichi, 2024b; Liu et al., 2024a), as well as regularization and parameter-reduction strategies, such as pruning, dropout, parameter sharing, bottleneck designs, Mixture of Experts (Altarabichi, 2024a; Drokin, 2024; Fan et al., 2025; Yang and Wang, 2024). At the same time, recent studies (Hou et al., 2025; Yu et al., 2024; Zeng et al., 2024) have raised concerns about the fairness of empirical comparisons and the scope of KAN advantages, suggesting that performance gains may be confined to settings involving smooth, regular functions, small-to-medium model sizes, or the use of symbolic regression.

To the best of our knowledge, applications of KANs to the field of Natural Language Processing (NLP) are still scarce in literature; probably because the lack of smoothness in the involved inputs and the training instability of KANs make it hard to deploy them in state-of-the-art models. Among the existing applications, Imran and Ishmam (2024) highlight the advantages of employing a KAN-based classification head on top of different language models for sentiment analysis, topic classification, question classification and language identification tasks.

## 3. Background

Kolmogorov-Arnold Networks (KANs) are neural architectures inspired by the Kolmogorov-Arnold representation theorem. The theorem states that any smooth function can be represented as a finite composition of continuous, but not necessarily smooth, univariate functions combined through summation. Building on this idea, Liu et al. (2024b) proposed a neural network architecture in which each layer output is computed as a sum of learnable smooth univariate functions.

A KAN layer with index l maps an input vector $x ^ { ( l ) } \in \mathbb { R } ^ { n _ { l } }$ to an output vector $\boldsymbol { x } ^ { ( l + 1 ) } \in \mathbb { R } ^ { n _ { l + 1 } }$ by associating a learnable univariate function $\phi _ { i j } ^ { ( l ) } : \mathbb { R }  \mathbb { R }$ to each directed connection from input coordinate i to output coordinate j, and aggregating contributions by summation:

$$
x _ { j } ^ { ( l + 1 ) } = \sum _ { i = 1 } ^ { n _ { l } } \phi _ { i j } ^ { ( l ) } ( x _ { i } ^ { ( l ) } ) , \qquad j = 1 , \ldots , n _ { l + 1 } .\tag{1}
$$

Equivalently, the layer can be viewed as an $n _ { l } \times n _ { l + 1 }$ matrix of univariate functions $\Phi ^ { ( l ) } = ( \phi _ { i j } ^ { ( l ) } )$ applied coordinate-wise and summed across inputs. Consequently, a KAN with L layers is obtained through composition:

$$
\mathbf { K A N } ( x ) = \left( \Phi ^ { ( L - 1 ) } \circ \Phi ^ { ( L - 2 ) } \circ \cdots \circ \Phi ^ { ( 0 ) } \right) ( x ) ,\tag{2}
$$

where each $\Phi ^ { ( l ) }$ denotes the non-linear mapping defined by Eq. (1). In the original formulation, each activation function $\phi _ { i j } ^ { ( l ) }$ is parametrized as the sum of a base function and a spline component: $\phi _ { i j } ^ { ( l ) } ( x _ { i } ^ { ( l ) } ) = \gamma _ { i j } ^ { ( l ) } \tilde { f ( x _ { i } ^ { ( l ) } ) } + \mathcal { S } _ { i j } ^ { ( l ) } ( x _ { i } ^ { ( l ) } )$ , where f is a fixed basis function (e.g., a sigmoid-like function), $\gamma _ { i j } ^ { ( l ) }$ is a learnable coefficient, and $\mathcal { S } _ { i j } ^ { ( l ) }$ is a learnable spline of order k defined on a grid of size $g .$ . However, the definition of KAN layers does not depend on a specific functional family and, as highlighted in Section 2., several different parametrizations are possible.

## 4. Methodology

Given a vocabulary V and embedding size d, a standard embedding matrix $E \in \mathbb { R } ^ { | V | \times d }$ can be approximated by a rank-r factorization $E \approx A B$ , where $A \in \mathbb { R } ^ { | V | \times r }$ and $\boldsymbol { B } \in \mathbb { R } ^ { r \times d }$ with $r \ll d .$ . In our method, we retain this factorized structure but replace the second linear mapping with the learnable non-linear transformation provided by a KAN layer of the same shape, namely with r inputs and d outputs:

$$
E x \approx K A N ( W x )
$$

![](images/fc1fddbc63b0caf5a50c33e046aa15d02ee3a7b3011dc3f7476df372111c463e.jpg)  
(a)

![](images/5c514f88a372e8e85d07beb2cfabd3564ebca39263e1a3a3422ab1d85dce7630.jpg)  
(b)

![](images/97b202ea32a3936ba84aa41db85f2b370a042ae884bc6cc5dee0d7a3d0db186d.jpg)  
(c)  
Figure 1: Embeddings layers to embed vocabulary V into a space of size d: (a) a standard full-rank embedding layer; (b) a low-rank factorized embedding layer of rank $r ;$ (c) a KAN-based factorized embedding layer of rank r.

where x denotes the one-hot vector of a vocabulary item and $W \in \mathbb { R } ^ { | V | \times r }$ plays the role of the first factor in the low-rank decomposition. Figure 1 compares visually the proposed factorized KAN-based embedding layer against a standard embedding layer and a low-rank factorized embedding layer.

A practical advantage of low-rank factorized embeddings is that they admit a straightforward initialization. Given a pre-trained embedding matrix $E \in \mathbb { R } ^ { | V | \times d }$ , the low-rank decomposition with minimum reconstruction error can be obtained through a singular-value decomposition (SVD), yielding matrices $A \in \mathbb { R } ^ { | V | \times r }$ and $\boldsymbol { B } \in \mathbb { R } ^ { r \times d }$ .To enable an analogous initialization mechanism for our KAN-based factorized embeddings, we develop two strategies, both starting from the SVD of E. In each case, the matrix A is used to initialize the linear transformation encoded in matrix W that precedes the KAN layer. Since KAN activation functions operate within a fixed activation interval, defined by the splines grid, we first apply a scaling transformation controlled by a constant $\alpha > 0$ , deriving $A ^ { \prime } = \alpha A$ and $B ^ { \prime } = B / \alpha$ . Then we initialize W from $A ^ { \prime }$ . The two methods differ in how the parameters of the KAN layer itself are initialized.

KAN-SVD initialization The first strategy aims to closely reproduce the linear behaviour of the SVD-based low-rank factorization at initialization. After setting $W = A ^ { \prime }$ , each KAN activation function $\phi _ { i j }$ is initialized so that, within the activation interval, it approximates a linear function with slope given by the corresponding entry of $B ^ { \prime }$ , i.e. $\phi _ { i j } ( y ) \approx b _ { i j } ^ { \prime } \cdot y$ . This can be achieved, for instance, by performing a least-squares regression over a small set of points within the activation interval. We refer to this initialization as KAN-SVD.

KAN-BCD initialization The second strategy adopts a block coordinate descent (BCD) scheme that alternates between updating the parameters of the KAN layer and the linear layer W, minimizing the reconstruction loss over the vocabulary V:

$$
\mathcal { L } = \sum _ { x \in V } \Vert E \cdot x - K A N ( W \cdot x ) \Vert ^ { 2 }
$$

At each iteration, we freeze W and update the KAN parameters by solving a ridge-regularized least-squares problem. This problem admits a closed-form solution because the spline parameters enter linearly in the output of the KAN layer. Then, we freeze the KAN layer and optimize W to minimize $\mathcal { L }$ via stochastic gradient descent over a fixed number of steps. The process continues until the relative improvement in $\mathcal { L }$ between consecutive iterations falls below a tolerance threshold. Note that this strategy does not rely on the matrix B, but explores a more flexible non-linear reconstruction regime. We denote this method as KAN-BCD.

## 5. Experimental setup

We compare our KAN-based factorized embeddings against low-rank factorized embeddings of the same rank within the task of learning Skipgram-like word embeddings. The pre-trained embedding vectors to be compressed are derived by training a standard full-rank Skipgram model. We assess information loss after compression using two intrinsic evaluation tasks: word analogies and semantic similarity. Both tasks are evaluated (i) immediately after compression, and (ii) after a short post-compression adaptation phase, in which the factorized model is finetuned on the same data used to train the full-rank embeddings. Following Lioutas et al. (2020), we compare two adaptation strategies: training with a standard Skipgram loss or including a feature distillation term from the original full-rank embeddings.

Throughout the rest of the paper, we denote low-rank factorization as LF, and the two KANbased factorized embeddings as KF-SVD and KF-BCD, corresponding respectively to the KAN-SVD and KAN-BCD initialization strategies. When discussing the effect of the post-compression adaptation phase, we use the suffix + to distinguish the distilled models.

## 5.1. Training data

We train a standard Skipgram model with negative sampling following the original formulation of Mikolov et al. (2013b): this is a binary classifier trained to recognize whether two tokens are likely to be in the same context or not. We use a corpus consisting of a 120M words subset of Wikipedia. After removing punctuation and stop words and discarding tokens with fewer than 100 occurrences, we obtain a vocabulary of approximately 37k tokens. The training dataset is generated by scanning the corpus and collecting up to four target–context pairs per occurrence, using a context window of size 2. Each target–context pair observed within the context window constitutes a positive training example, while negative examples are obtained by pairing the same target token with tokens randomly sampled from the vocabulary, through the frequency-based sampling distribution recommended in Mikolov et al. (2013b). For each positive target-context pair, three negative examples are generated.

Table 1: Number of learnable parameters and associated compression rate with respect to the full-rank embeddingsfor the tested models.
<table><tr><td>Models</td><td>N. parameters</td><td>Compression rate</td></tr><tr><td> $\overline { { L F } }$ </td><td>935,150</td><td>2.00x</td></tr><tr><td>KF-SVD, KF-BCD</td><td>946,400</td><td>1.97x</td></tr></table>

## 5.2. Hyperparameters and implementation details

We compress embeddings of size $d = 5 0$ with factorized embeddings of rank $r = 2 5$ . As commonly done in Skipgram models, the same embedding layer is shared between target and context representations.

For the implementation of the KAN-based factorized embedding, we use the fastkan library, which employs Radial Basis Functions (RBFs) in place of splines. This choice is supported by Li (2024), who shows that RBFs can approximate cubic B-splines, while being substantially more efficient. In this implementation, a KAN layer takes the form:

$$
\Phi ( x ) = \sum _ { m = 1 } ^ { g } C ^ { m } \psi ( \left| \left| x - p _ { m } \right| \right| )\tag{3}
$$

where ψ is a Gaussian kernel, $p _ { m }$ are fixed grid points, and $C ^ { m } \in \mathbb { R } ^ { r \times d }$ are learnable parameters. We define a grid of $g = 1 0$ points spaced uniformly in the activation interval $[ - 1 , 1 ]$ . The number of learnable parameters in each model is reported in Table 1.

Initialization The matrices A,B obtained via SVD of the full-rank embedding matrix E, are scaled by a constant factor α set to the inverse of the 99-percentile of $| A |$

For the KAN-SVD strategy, the KAN layer is initialized to emulate a linear transformation with weight matrix $B ^ { \prime }$ . The RBF-based implementation that we use allows to do that by interpolating the identity function at grid points $p _ { 1 } , \ldots , p _ { g }$ . Following the definition given in Eq. (3), the interpolating weights $c ^ { * }$ can be computed by solving the linear system $\boldsymbol { Q } \boldsymbol { c } ^ { * } = \boldsymbol { \nu } ,$ , where $Q _ { h k } =$ $\psi ( \left| \left| p _ { h } - p _ { k } \right| \right| )$ and $\nu _ { k } = p _ { k }$ for $h , k = 1 \ldots g$ . Such defined $Q$ is non-singular (Micchelli, 1986), thus the system is guaranteed to admit a solution. We then set $C _ { m } = c _ { m } ^ { * } B ^ { \prime }$ for all $m = 1 , \ldots , g$

In the context of the KAN-BCD initialization, the KAN parameters are optimized with a ridge coefficient equal to $1 0 ^ { - 4 }$ , while the linear layer is updated through the Adam optimizer with learning rate $\bar { 1 } 0 ^ { - 3 }$ and for 20 steps. We follow the BCD scheme up to 200 iterations or until the relative improvement in reconstruction loss falls below $1 0 ^ { - 4 }$

Post-compression adaptation We fine-tune the factorized models using the Adam optimizer with learning rate $1 0 ^ { - \bar { 4 } }$ . When feature distillation is enabled, we add a penalty based on the cosine similarity between the current embedding $w _ { t }$ and the original embedding $w _ { t } ^ { \prime } \colon \mathcal { L } _ { F D } ( t ) =$ $1 - \frac { w _ { t } \cdot w _ { t } ^ { \prime } } { \| w _ { t } \| \| w _ { t } ^ { \prime } \| }$ , for token t. The loss for a target-context pair $( t , c )$ becomes:

$$
( 1 - \lambda ) \mathcal { L } _ { S G } ( t , c ) + \frac { \lambda } { 2 } ( \mathcal { L } _ { F D } ( t ) + \mathcal { L } _ { F D } ( c ) )
$$

where $\mathcal { L } _ { S G }$ is the regular Skipgram loss function and $\lambda = 0 . 3$ , following (Lioutas et al., 2020).

## 5.3. Embedding evaluation

We evaluate analogy performance on the Google Analogy dataset (Mikolov et al., 2013a), containing 19 544 questions across semantic and syntactic relations, using the standard 3CosAdd formulation. Semantic similarity is assessed on WordSim-353 (Finkelstein et al., 2002) and SimLex-999 (Hill et al., 2015), using Spearman correlation.

## 5.4. Hardware and software environment

All experiments were conducted on a single NVIDIA V100 GPU (16 GB) on the Ubuntu Linux operating system. Models were implemented in Python 3.10 with PyTorch 2.3.0, Hugging Face Transformers 4.45.2 and CUDA 12.4.

## 6. Results

Table 2 summarizes the performance of the evaluated models immediately after compression. As expected, low-rank factorization (LF) entails a substantial loss of information: its accuracy on word analogy tasks drops by almost 50% relative to the full-rank baseline, and the overall average reduction is 28.8%. KF-SVD performs even worse (31.2% average reduction), which is consistent with its design: since it is initialized to closely replicate the behaviour of the compressed LF embedding, it inherits and potentially amplifies its errors. In contrast, KF-BCD embeddings retain a considerably larger fraction of semantic structure. Their average performance reduction is only 16.4%, nearly halving the degradation observed for LF. On analogy tasks and WordSim-353, the same pattern emerges: LF suffers a 39% reduction on average, whereas KF-BCD exhibits only 19%. A different trend is observed on SimLex-999. Both LF and KF-SVD achieve slightly higher correlation scores than the full-rank model. This suggests that SimLex may emphasize semantic relations that are underrepresented in the corpus used to train the original full-rank Skipgram embeddings. Notably, this behaviour persists after adaptation: post-compression fine-tuning does not increase correlation on SimLex (Table 3), supporting the hypothesis that the discrepancy stems from corpus–dataset mismatch rather than from compression effects.

Table 2: Scores achieved immediately after compression on the evaluation datasets. Columns syn. analogies accuracy and sem. analogies accuracy report the accuracy scored on the Google Analogy datasets, respectively for syntactical and semantical relations; columns wordsim correlation and simlex correlation report the Spearman correlation scored respectively on the WordSim-353 and SimLex-999 datasets. The last column displays the average relative reduction in performance with respect to the full-rank model.

<table><tr><td></td><td>syn. analogies</td><td>sem. analogies</td><td>wordsim</td><td>simlex</td><td>avg. performance</td></tr><tr><td>Model</td><td>accuracy</td><td>accuracy</td><td>correlation</td><td>correlation</td><td>reduction (%)</td></tr><tr><td>full-rank</td><td>0.203</td><td>0.224</td><td>0.671</td><td>0.225</td><td></td></tr><tr><td>LF</td><td>0.100</td><td>0.103</td><td>0.585</td><td>0.230</td><td>28.8</td></tr><tr><td>KF-SVD</td><td>0.093</td><td>0.091</td><td>0.585</td><td>0.229</td><td>31.2</td></tr><tr><td>KF-BCD</td><td>0.149</td><td>0.161</td><td>0.652</td><td>0.207</td><td>16.4</td></tr></table>

Table 3 reports the best performance achieved within three epochs of post-compression adaptation. Overall, the adaptation phase reduces the average gap to the full-rank baseline across all models, although the gains are not uniform. In the end, all KAN-based variants perform better on average than the standard low-rank factorization, with KF-SVD+ attaining the lowest overall average reduction (11.7%) despite starting from the weakest post-compression initialization.

These results suggest that the additional functional expressivity introduced by the KAN layer provides a more effective capacity for recovering task-relevant structure during fine-tuning. In contrast, KF-BCD and KF-BCD+ show only moderate improvements: this may reflect a less favourable optimization landscape induced by the BCD initialization. The distillation loss is generally beneficial for both KAN-based variants, whereas it consistently harms adaptation for LF embeddings.

Table 3: Best scores achieved over 3 epochs of post-compression adaptation. Columns as in Table 2.
<table><tr><td rowspan=2 colspan=1>Model</td><td rowspan=2 colspan=1>syn. analogiesaccuracy</td><td rowspan=2 colspan=1>sem. analogiesaccuracy</td><td rowspan=1 colspan=1>wordsim</td><td rowspan=2 colspan=1>simlexcorrelation</td><td rowspan=2 colspan=1>avg. perf.reduction (%)</td></tr><tr><td rowspan=1 colspan=1>correlation</td></tr><tr><td rowspan=1 colspan=1>full-rank</td><td rowspan=1 colspan=1>0.203</td><td rowspan=1 colspan=1>0.224</td><td rowspan=1 colspan=1>0.671</td><td rowspan=1 colspan=1>0.225</td><td rowspan=1 colspan=1>一</td></tr><tr><td rowspan=1 colspan=1>LF</td><td rowspan=1 colspan=1>0.155</td><td rowspan=1 colspan=1>0.134</td><td rowspan=1 colspan=1>0.625</td><td rowspan=1 colspan=1>0.230</td><td rowspan=1 colspan=1>17.1</td></tr><tr><td rowspan=1 colspan=1>LF+</td><td rowspan=1 colspan=1>0.111</td><td rowspan=1 colspan=1>0.103</td><td rowspan=1 colspan=1>0.590</td><td rowspan=1 colspan=1>0.230</td><td rowspan=1 colspan=1>27.3</td></tr><tr><td rowspan=1 colspan=1>KF-SVD</td><td rowspan=1 colspan=1>0.163</td><td rowspan=1 colspan=1>0.128</td><td rowspan=1 colspan=1>0.629</td><td rowspan=1 colspan=1>0.229</td><td rowspan=1 colspan=1>16.8</td></tr><tr><td rowspan=1 colspan=1>KF-SVD+</td><td rowspan=1 colspan=1>0.152</td><td rowspan=1 colspan=1>0.174</td><td rowspan=1 colspan=1>0.663</td><td rowspan=1 colspan=1>0.229</td><td rowspan=1 colspan=1>11.7</td></tr><tr><td rowspan=1 colspan=1>KF-BCD</td><td rowspan=1 colspan=1>0.149</td><td rowspan=1 colspan=1>0.159</td><td rowspan=1 colspan=1>0.653</td><td rowspan=1 colspan=1>0.214</td><td rowspan=1 colspan=1>15.8</td></tr><tr><td rowspan=1 colspan=1>KF-BCD+</td><td rowspan=1 colspan=1>0.149</td><td rowspan=1 colspan=1>0.166</td><td rowspan=1 colspan=1>0.652</td><td rowspan=1 colspan=1>0.207</td><td rowspan=1 colspan=1>15.8</td></tr></table>

## 7. Conclusion and future work

Our results indicate that KAN-based factorized embeddings constitute a viable and competitive alternative to traditional low-rank factorizations, as learnable non-linearities introduced by KAN layers enable to recover a substantial portion of the expressiveness lost through compression.

The two initialization schemes we proposed exhibit complementary advantages. KAN-SVD excels at recovering information directly from the full-rank embedding matrix, while KAN-BCD offers a more effective starting point for post-compression adaptation or, possibly, downstream fine-tuning. At the same time, the heterogeneous behaviour observed across the two post-compression adaptation strategies highlights that training stability remains a central challenge for KAN-based embedding layers. These findings point to the need for a more systematic understanding of how initialization, optimization hyperparameters, and fine-tuning procedures interact with KAN architectures.

Future work will extend this study along two directions. First, evaluating KAN-based embeddings on downstream NLP tasks will clarify whether the advantages observed in intrinsic evaluation translate into gains in practical applications. Second, integrating KAN-based factorized embeddings into larger architectures, most notably Transformer models, will shed light on their compatibility with modern NLP pipelines.

Overall, our study suggests that KAN-based non-linear factorization is a promising direction for making embedding compression more expressive without sacrificing parameter efficiency.

## References

Aghaei A. A. (2024). rKAN: Rational Kolmogorov-Arnold Networks. arXiv:2406.14495.

Altarabichi M. G. (2024a). DropKAN: Regularizing KANs by masking post-activations. arXiv:2407.13044.

Altarabichi M. G. (2024b). Rethinking the Function of Neurons in KANs. arXiv:2407.20667.

Bałazy K., Banaei M., Lebret R., Tabor J., and Aberer K. (2021). Direction is what you need: Improving Word Embedding Compression in Large Language Models. In Proceedings ofthe 6th Workshop on Representation Learning for NLP, pp. 322–330. Association for Computational Linguistics.

Bodner A. D., Tepsich A. S., Spolski J. N., and Pourteau S. (2025). Convolutional Kolmogorov-Arnold Networks. arXiv:2406.13155.

Chen P. H., Si S., Li Y., Chelba C., and jui Hsieh C. (2018). Groupreduce: Block-wise low-rank approximation for neural language model shrinking. arXiv:1806.06950.

Chen W., Wilson J. T., Tyree S., Weinberger K. Q., and Chen Y. (2015). Compressing Neural Networks with the Hashing Trick. arXiv:1504.04788.

Cheon M. (2024). Demonstrating the Efficacy of Kolmogorov-Arnold Networks in Vision Tasks. arXiv:2406.14916.

Drokin I. (2024). Kolmogorov-Arnold Convolutions: Design Principles and Empirical Studies. arXiv:2407.01092.

Fan W., Wang C., Li S., and Liu N. (2025). Shift-Invariant Attribute Scoring for Kolmogorov-Arnold Networks via Shapley Value. arXiv:2510.01663.

Finkelstein L., Gabrilovich E., Matias Y., Rivlin E., Solan Z., Wolfman G., and Ruppin E. (2002). Placing search in context: the concept revisited. ACM Trans. Inf. Syst., 20(1):116–131.

Gao C., Wu X., Wang P., Wang J., Zang L., Wang Z., and Hu S. (2023). DistilCSE: Effective Knowledge Distillation For Contrastive Sentence Embeddings. arXiv:2112.05638.

Genet R. and Inzirillo H. (2024a). A Temporal Kolmogorov-Arnold Transformer for Time Series Forecasting. arXiv:2406.02486.

Genet R. and Inzirillo H. (2024b). TKAN: Temporal Kolmogorov-Arnold Networks. arXiv:2405.07344.

Guo H., Li F., Li J., and Liu H. (2024). KAN v.s. MLP for Offline Reinforcement Learning. arXiv:2409.09653.

Han X., Zhang X., Wu Y., Zhang Z., and Wu Z. (2025). Are KANs Effective for Multivariate Time Series Forecasting? arXiv:2408.11306.

Hill F., Reichart R., and Korhonen A. (2015). SimLex-999: Evaluating semantic models with (genuine) similarity estimation. Computational Linguistics, 41(4):665–695.

Hou Y., Ji T., Zhang D., and Stefanidis A. (2025). Kolmogorov-Arnold Networks: A Critical Assessment of Claims, Performance, and Practical Viability. arXiv:2407.11075.

Howard A. A., Jacob B., Murphy S. H., Heinlein A., and Stinis P. (2024). Finite basis Kolmogorov-Arnold networks: domain decomposition for data-driven and physics-informed problems. arXiv:2406.19662.

Imran A. A. and Ishmam M. F. (2024). Fourierkan outperforms MLP on Text Classification Head Finetuning. arXiv:2408.08803.

Kich V. A., Bottega J. A., Steinmetz R., Grando R. B., Yorozu A., and Ohya A. (2024). Kolmogorov-Arnold Network for Online Reinforcement Learning. arXiv:2408.04841.

Lan Z., Chen M., Goodman S., Gimpel K., Sharma P., and Soricut R. (2019). Albert: A lite bert for self-supervised learning of language representations. arXiv:1909.11942.

Li C., Liu X., Li W., Wang C., Liu H., Liu Y., Chen Z., and Yuan Y. (2024). U-KAN Makes Strong Backbone for Medical Image Segmentation and Generation. arXiv:2406.02918.

Li Z. (2024). Kolmogorov-Arnold Networks are Radial Basis Function Networks. arXiv:2405.06721.

Lioutas V., Rashid A., Kumar K., Haidar M. A., and Rezagholizadeh M. (2020). Improving Word Embedding Factorization for Compression Using Distilled Nonlinear Neural Decomposition. In Cohn T., He Y., and Liu Y., editors, Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 2774–2784. Association for Computational Linguistics.

Liu Z., Ma P., Wang Y., Matusik W., and Tegmark M. (2024a). KAN 2.0: Kolmogorov-Arnold Networks Meet Science. arXiv:2408.10205.

Liu Z., Wang Y., Vaidya S., Ruehle F., Halverson J., Soljaciˇ c M., Hou T. Y., and Tegmark M. (2024b).´ KAN: Kolmogorov-Arnold Networks. arXiv:2404.19756.

Micchelli C. A. (1986). Interpolation of scattered data: Distance matrices and conditionally positive definite functions. Constructive Approximation, 2.

Mikolov T., Chen K., Corrado G., and Dean J. (2013a). Efficient estimation of word representations in vector space. arXiv:1301.3781.

Mikolov T., Sutskever I., Chen K., Corrado G., and Dean J. (2013b). Distributed representations of words and phrases and their compositionality. arXiv:1310.4546.

Qiu Q., Zhu T., Gong H., Chen L., and Ning H. (2024). ReLU-KAN: New Kolmogorov-Arnold Networks that Only Need Matrix Addition, Dot Multiplication, and ReLU. arXiv:2406.02075.

Reinhardt E. A. F., Dinesh P. R., and Gleyzer S. (2025). SineKAN: Kolmogorov-Arnold Networks Using Sinusoidal Activation Functions. Frontiers in Artificial Intelligence, 7.

Shi K. and Yu K. (2018). Structured word embedding for low memory neural network language model. In Interspeech 2018, pp. 1254–1258.

Shu R. and Nakayama H. (2017). Compressing word embeddings via deep compositional code learning. arXiv:1711.01068.

So C. C. and Yung S. P. (2024). Higher-order-ReLU-KANs (HRKANs) for solving physics-informed neural networks (PINNs) more accurately, robustly and faster. arXiv:2409.14248.

Suzuki J. and Nagata M. (2016). Learning compact neural word embeddings by parameter space sharing. In Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence, IJCAI’16, p. 2046–2052. AAAI Press.

Vaca-Rubio C. J., Blanco L., Pereira R., and Caus M. (2024). Kolmogorov-Arnold Networks (KANs) for Time Series Analysis. arXiv:2406.02918.

Wang Y., Sun J., Bai J., Anitescu C., Eshaghi M. S., Zhuang X., Rabczuk T., and Liu Y. (2025). Kolmogorov Arnold Informed neural network: A physics-informed deep learning framework for solving forward and inverse problems based on Kolmogorov Arnold Networks. Computer Methods in Applied Mechanics and Engineering, 433.

Xu A., Zhang B., Kong S., Huang Y., Yang Z., Srivastava S., and Sun M. (2025a). Effective Integration of KAN for Keyword Spotting. In ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5.

Xu J., Chen Z., Li J., Yang S., Wang W., Hu X., and Ngai E. (2025b). Enhancing Graph Collaborative Filtering with FourierKAN Feature Transformation. arXiv:2406.01034.

Xu K., Chen L., and Wang S. (2024). Kolmogorov-Arnold Networks for Time Series: Bridging Predictive Power and Interpretability. arXiv:2406.02496.

Yang X. and Wang X. (2024). Kolmogorov-Arnold Transformer. arXiv:2409.10594.

Yu R., Yu W., and Wang X. (2024). KAN or MLP: A Fairer Comparison. arXiv:2407.16674.

Yu R. C., Wu S., and Gui J. (2025). Residual Kolmogorov-Arnold Network for Enhanced Deep Learning. arXiv:241.

Zeng C., Wang J., Shen H., and Wang Q. (2024). KAN versus MLP on Irregular or Noisy Functions. arXiv:2408.07906.

Zhang F. and Zhang X. (2024). GraphKAN: Enhancing Feature Extraction with Graph Kolmogorov Arnold Networks. arXiv:2406.13597.