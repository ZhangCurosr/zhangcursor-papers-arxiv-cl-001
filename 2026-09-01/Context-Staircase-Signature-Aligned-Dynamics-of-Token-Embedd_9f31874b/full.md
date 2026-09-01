# Context Staircase: Signature-Aligned Dynamics of Token Embeddings under Small Initialization

Junjie Yao School of Mathematical Sciences Shanghai Jiao Tong University Shanghai 200240, China

Liangkai Hang School of Mathematical Sciences Shanghai Jiao Tong University Shanghai 200240, China

Zhi-Qin John Xu

Institute of Natural Sciences, School of Mathematical Sciences

Shanghai Jiao Tong University

Shanghai 200240, China

Corresponding author

yaojj22@sjtu.edu.cn

hangliangkai@sjtu.edu.cn

xuzhiqin@sjtu.edu.cn

## Abstract

Token embeddings are the basic representational units that connect discrete tokens with continuous computation in language models. Although modern language models learn embeddings from random initialization through gradient-based training, the dynamical mechanism by which meaningful embedding structures emerge remains unclear. In this work, we identify that the evolving embedding structures are closely related to token-conditioned label and contextual distributions, which we formalize as probability signatures. We ob serve a progressive learning process, which we term Context Staircase: embeddings learn the low-order statistic signatures of the data before the high-order ones. More specifically, we observe that early in training they align with the simplest, context-free signature linking a token to its label, and as training proceeds, they progressively reflect signatures involving more and more context tokens. We then analyze the gradient flow of embeddings under small initialization to explain this phenomenon, deriving embedding evolution equations for feed-forward and self-attention architectures. We further extend these observations to real language-model training. Finally, we show that these embedding structures play an important role in both task learning and the incorporation of semantic structure into the embedding space. Overall, our results provide a dynamic explanation of how data statistics and architecture jointly shape token embeddings in language models, and reveal an implicit bias in the space of data statistics: training proceeds from simpler, low-order statistical relations toward increasingly complex, context-dependent ones.

Keywords: token embeddings, training dynamics, gradient flow

## 1 Introduction

Large language models (LLMs) have become powerful general-purpose systems, achieving strong performance across a wide range of language tasks (Brown et al., 2020; Touvron et al., 2023; OpenAI, 2023; Guo et al., 2025). At the foundation of these systems lies the token embedding space, which connects discrete symbols to continuous computation. Early static word representations were often constructed from distributional statistics, such as word co-occurrence, or through matrix factorization (Pennington et al., 2014; Levy and Goldberg, 2014). Predictive approaches such as CBOW and Skip-gram instead learned word embeddings through dedicated context-prediction objectives, by predicting a target word from its surrounding context or vice versa (Mikolov et al., 2013a,b). Later approaches further incorporated subword information into these representations (Bojanowski et al., 2017). Modern language models initialize token embeddings randomly and optimize them jointly with the rest of the model under the language modeling objective (Vaswani et al., 2017; Radford et al., 2019; Brown et al., 2020). This raises a basic question: what statistical structures are written into token embeddings during training, and how do these structures evolve over time?

Prior work has shown that learned representations contain meaningful semantic organization (Ethayarajh, 2019; Hewitt and Manning, 2019; Gurnee and Tegmark, 2024; Han et al., 2024). Most of these studies analyze representations after training or probe what information they contain. Much less is known about how these structures emerge in the embedding. In particular, it remains unclear whether the geometry observed at diferent stages of training can be related to identifiable statistics of the data and the model architecture. Recent studies have further shown that initialization can substantially afect the evolution and geometry of token embeddings (Zhang et al., 2024, 2025; Yao et al., 2025). In particular, small initialization tends to produce more structured embedding spaces that better reflect the underlying relations of the task, and such embedding structures have been closely associated with improved reasoning ability.

Motivated by these observations, we systematically investigate the evolution of token embeddings under small initialization. We introduce a hierarchy of token-conditioned statistical quantities, called probability signatures, which describe the joint distributions among a token, the target label, and the context tokens. Across controlled sequence-prediction tasks, we identify a characteristic evolution that we term Context Staircase: the early embedding geometry is closely associated with low-order signatures, while structures associated with higher-order signatures become increasingly visible as training proceeds. As illustrated by the addition task in Figure 1, the embeddings first form a clear linear structure, closely matching that of the low-order signature. Later in training, the embedding geometry develops a pronounced odd–even separation, which is likewise reflected in the higher-order signature. More broadly, Context Staircase reveals an implicit bias in the space of data statistics: training preferentially incorporates simpler, low-order statistical relations involving fewer context variables before progressively incorporating more complex, higher-order relations involving richer context.

To explain these empirical observations, we analyze the gradient flow of token embeddings under small initialization. We show that the embedding gradient can be decomposed into contributions associated with label signatures of diferent orders. Small initialization strongly suppresses higher-order contributions at the beginning of training, explaining the early association with low-order signatures. As training proceeds and the parameter scale grows, contributions from all accessible signature orders reach the same leading scale and can jointly afect the embedding dynamics, with the highest-order contribution being the largest under our late-stage conditions. Thus, small initialization provides a dynamical mechanism for the observed implicit bias, inducing an ordered progression from simple to increasingly context-dependent statistical structures.

For self-attention models, the corresponding signature terms are position-dependent and modulated by attention scores, which control how strongly and how rapidly diferent tokens afect the embedding dynamics. We further connect these analyses to distinct computational paths in a full decoder-only Transformer and test the resulting hypotheses on a rea language-model training run.

![](images/1080d18ad7548542f42b28cd4f529febde7e273142132727a5982bec6f4f57e2.jpg)  
Figure 1: Under small initialization, the embedding geometry evolves progressively during training.

Finally, we investigate the functional role of these embedding structures and show that they can directly facilitate model training and task learning. Using a simple numerical comparison task, we demonstrate that once the embedding space acquires the structure encoded by the corresponding signature, the model can naturally learn the underlying comparison rule, suggesting that signature-aligned geometry provides a useful inductive basis for solving the task. We further examine signatures derived from real-world language corpora and find that they already exhibit meaningful semantic organization, including structured relations among numbers, cyclic relations among weekdays, and category-level clustering of semantically related nouns. These results suggest that signatures capture statistical structures in language that can serve as a basis for the emergence of semantic organization in the learned embedding space.

Our main contributions are as follows:

1. We identify a progressive evolution of token-embedding structures, which we term Context Staircase: embeddings are closely associated with low-order signatures in the early stage of training, while higher-order signature structures gradually emerge later.

This reveals an implicit bias in the space of data statistics, from simpler, low-order relations to increasingly complex, context-dependent ones.

2. We provide a dynamical explanation of this phenomenon through gradient-flow analysis. We show that signatures of diferent orders explicitly enter the embedding updates, and that small initialization induces their ordered influence over training.

3. We extend the analysis from feed-forward networks to self-attention architectures and further to real language-model training.

4. We investigate the role of the signature-aligned embedding structures, showing that they have important efects on task learning and the incorporation of semantic structure into the embedding space.

## 2 Related Work

Token embeddings and representation geometry. Early studies of word embeddings focused on constructing static vector representations that encode the distributional properties of words. Neural language models introduced distributed word representations as trainable parameters learned from language modeling objectives (Bengio et al., 2003), while word2vec showed that eficient local context-prediction objectives can produce word vectors with strong semantic and syntactic regularities (Mikolov et al., 2013b). GloVe further connected word embeddings to global word–word co-occurrence statistics through a matrix-factorization-style objective (Pennington et al., 2014), and fastText enriched static embeddings with subword information by representing words as sums of character n-gram vectors (Bojanowski et al., 2017). These works established the classical view that embedding geometry is closely tied to contextual distributions in text.

With the rise of contextualized and large language models, subsequent studies shifted from constructing embeddings to analyzing what linguistic and semantic structures are encoded in learned representations. Contextual representations in ELMo, BERT, and GPT-2 were shown to exhibit nontrivial geometric properties such as anisotropy and context dependence (Ethayarajh, 2019), and syntactic structures were found to be recoverable from neural word representation spaces through structural probes (Hewitt and Manning, 2019). More recent work has shown that modern large language models organize world knowledge, semantic concepts, and output word embeddings through structured directions or regions in representation space (Gurnee and Tegmark, 2024; Han et al., 2024; Park et al., 2025).

A closely related line of recent work further shows that the geometry of learned representations can depend strongly on initialization. In particular, studies of Transformer models have found that diferent initialization scales can lead to qualitatively diferent representation structures and learning behaviors (Zhang et al., 2024, 2025; Yao et al., 2025). Small initialization tends to produce more structured embedding spaces that more clearly reflect the underlying relations of the task, and these structures have been associated with improved reasoning and compositional generalization. These findings highlight initialization as an important factor in the emergence of embedding structure during training.

However, it remains unclear what data-dependent statistical structures these evolving embeddings correspond to and how such structures emerge progressively through gradientbased training. Our work addresses this question by directly analyzing the training dynamics of token embeddings under small initialization and relating their evolving geometry to a hierarchy of probability signatures derived from the data.

Training dynamics and gradient-flow analyses. A complementary line of work studies deep learning through the dynamics induced by gradient descent or its continuous-time limit, gradient flow. Early analyses of deep linear networks showed that even linear input– output maps can exhibit highly nontrivial learning dynamics, including plateaus and rapid transitions (Saxe et al., 2014). Kernel-based analyses, such as the neural tangent kernel framework, characterize training in the lazy regime where feature representations change only weakly (Jacot et al., 2018). More relevant to our setting are recent studies of the feature-learning regime under small initialization. These works show that small initialization can induce qualitatively diferent training dynamics, and that gradient flow may drive parameters toward structured or low-complexity configurations through phenomena such as condensation (Xu et al., 2025; Zhou et al., 2022; Boursier et al., 2022). Such results suggest that small initialization provides a particularly useful regime for studying how nontrivial representation structures emerge during training.

For Transformer models, the dynamical efects of initialization and feature learning are only beginning to be understood. Mean-field analysis has been used to study gradient scales and training stability in Transformers (Xiong et al., 2020), while other works analyze how attention-based architectures acquire in-context learning or token-level dependencies through gradient descent and gradient-flow dynamics (von Oswald et al., 2023; Yang et al., 2024b, 2025). Under small initialization, Yao et al. (2025) analyzes the early-stage training dynamics of language models and characterizes the coupled evolution of token embeddings and self-attention during training. More recently, Chen et al. (2026) develops a stage-wise gradient-flow analysis of attention learning, revealing a multi-stage process in which projection parameters first undergo condensation, attention subsequently becomes activated and develops structured focus. Most closely related to our work, Im et al. (2026) analyzes how attention-based language models learn token associations from natural language data by approximating early-stage gradients with their leading terms. They derive closed-form characterizations of Transformer weights in terms of corpus-dependent basis functions, including bigram, token-interchangeability, and context mappings, thereby connecting parameter formation with interpretable data statistics. Their analysis, however, focuses primarily on parameters inside Transformer blocks, such as attention- and MLPrelated weights, whereas the evolution of the token embedding matrix itself remains largely unexplored. In contrast, we focus on token embeddings under small initialization: starting from their empirically observed geometric evolution, we use gradient-flow analysis to identify the data-dependent probability signatures that enter the embedding updates through diferent computational paths.

## 3 Preliminary

## 3.1 Prediction Problem

We consider a token prediction problem over a finite vocabulary V, with vocabulary size $| \nu | = V$ . Each sample consists of an input sequence X and a target token y:

$$
X = ( x _ { 1 } , x _ { 2 } , \ldots , x _ { L } ) \in \mathcal { V } ^ { L } , \qquad y \in \mathcal { V } .\tag{1}
$$

For each token $\alpha \in \nu$ , we use $e _ { \alpha } \in \mathbb { R } ^ { V }$ to denote its one-hot vector. Consider a model $f :$ $\mathcal { V } ^ { L } \to \mathbb { R } ^ { V }$ , which contains an input embedding matrix $W ^ { E } \in \mathbb { R } ^ { d \times V }$ . The embedding vector of token $\alpha \in \mathcal V$ is denoted by ${ w _ { \alpha } ^ { E } \in \mathbb { R } ^ { d } }$ . The model also contains an output unembedding matrix $\pmb { W } ^ { U } \in \mathbb { R } ^ { V \times d }$ and $\pmb { g } ( X ) \in \mathbb { R } ^ { d }$ , which satisfies

$$
\pmb { \mathscr { f } } ( X ) = \pmb { W } ^ { U } \pmb { g } ( X ) \in \mathbb { R } ^ { V } .
$$

The predicted distribution is $\pmb { p } ( X ) = \mathrm { s o f t m a x } ( \pmb { f } ( X ) )$ , and training objective is the standard cross-entropy loss $\ell ( X , y ) = - \log p ( X ) _ { y }$

Throughout this work, we study the gradient-flow dynamics of the input embedding vectors:

$$
\begin{array} { r l } & { \displaystyle \frac { d w _ { \alpha } ^ { E } } { d t } = - \mathbb { E } _ { \mathcal { D } } \left[ \frac { \partial \ell ( X , y ) } { \partial w _ { \alpha } ^ { E } } \right] = - \mathbb { E } _ { \mathcal { D } } \left[ \left( \frac { \partial f ( X ) } { \partial w _ { \alpha } ^ { E } } \right) ^ { T } \frac { \partial \ell ( X , y ) } { \partial f ( X ) } \right] } \\ & { \quad \quad = \mathbb { E } _ { \mathcal { D } } \left[ \left( \frac { \partial f ( X ) } { \partial w _ { \alpha } ^ { E } } \right) ^ { T } ( e _ { y } - p ( X ) ) \right] = \mathbb { E } _ { \mathcal { D } } \left[ \left( \frac { \partial f ( X ) } { \partial w _ { \alpha } ^ { E } } \right) ^ { T } e _ { y } \right] - \mathbb { E } _ { \mathcal { D } } \left[ \left( \frac { \partial f ( X ) } { \partial w _ { \alpha } ^ { E } } \right) ^ { T } p ( X ) \right] } \\ & { \quad : = \displaystyle \frac { d w _ { \alpha } ^ { E } } { d t } ^ { y } - \frac { d w _ { \alpha } ^ { E } } { d t } ^ { p } . } \end{array}\tag{2}
$$

Our goal is to characterize how the data distribution and model components shape the embedding updates during training, and to relate these updates to the empirical evolution of embedding geometry.

## 3.2 Token-conditioned Signatures

In language modeling, the meaning of a token is often characterized by the distributional context in which it appears (Harris, 1954; Firth, 1957; Turney and Pantel, 2010; Lenci, 2018). Motivated by this distributional view, we introduce a hierarchy of token-conditioned distributions for our sequence prediction setting (1). For a token $\alpha \in \mathcal { V }$ , we denote its occurrence times in X by $N _ { \alpha } ( X )$ Let $I \sim \mathrm { U n i f } ( [ L ] )$ be sampled independently of the training sample $( X , y )$ , and condition on the event $x _ { I } = \alpha$ We define the occurrence probability of α by

$$
r _ { \alpha } : = \mathbb { P } ( x _ { I } = \alpha ) = \frac { 1 } { L } \mathbb { E } [ N _ { \alpha } ( X ) ] .
$$

For notational simplicity, throughout this paper we write $\mathbb { P } ( \cdot \mid \alpha )$ as shorthand for $\mathbb { P } ( \cdot \mid$ $x _ { I } = \alpha )$ , and use the same shorthand for conditional expectations. The most basic statistical object associated with α is the conditional label distribution

$$
\mathbb { P } \left( y = \nu \mid \alpha \right) , \qquad \nu \in \mathcal { V } .
$$

Conditioned on $\alpha ,$ we remove the occurrence at position I from the sequence and write the remaining context tokens as

$$
X _ { - \alpha } = ( x _ { 1 } , x _ { 2 } , \dots , x _ { L - 1 } ) .
$$

For $m \in \{ 1 , \ldots , L - 1 \}$ , let

$$
{ \mathcal { T } } _ { m } : = \left\{ ( j _ { 1 } , \ldots , j _ { m } ) \in [ L - 1 ] ^ { m } : j _ { a } \neq j _ { b } { \mathrm { ~ f o r ~ a l l ~ } } a \neq b \right\} ,
$$

so that $| \mathcal { Z } _ { m } | = ( L - 1 ) _ { m } = ( L - 1 ) ( L - 2 ) \cdot \cdot \cdot ( L - m )$ . We sample an ordered tuple of distinct context positions

$$
( J _ { 1 } ^ { ( m ) } , \ldots , J _ { m } ^ { ( m ) } ) \sim \mathrm { U n i f } ( \mathbb { Z } _ { m } ) ,
$$

independently of $( X , y , I )$ . The order-m co-occurrence and label signatures are defined componentwise by

$$
\varphi _ { \alpha } ^ { m } ( \beta _ { 1 } , \dots , \beta _ { m } ) : = \mathbb { P } \left( x _ { J _ { 1 } ^ { ( m ) } } = \beta _ { 1 } , \dots , x _ { J _ { m } ^ { ( m ) } } = \beta _ { m } \mid \alpha \right) ,
$$

$$
\varphi _ { \alpha } ^ { y ; m } ( \nu , \beta _ { 1 } , \dots , \beta _ { m } ) : = \mathbb { P } \left( y = \nu , x _ { _ { J _ { 1 } ^ { ( m ) } } } = \beta _ { 1 } , \dots , x _ { _ { J _ { m } ^ { ( m ) } } } = \beta _ { m } \mid \alpha \right) .
$$

Equivalently,

$$
\varphi _ { \alpha } ^ { y ; m } ( \nu , \beta _ { 1 } , \ldots , \beta _ { m } ) = { \frac { 1 } { ( L - 1 ) _ { m } } } \sum _ { ( j _ { 1 } , \ldots , j _ { m } ) \in { \cal { T } } _ { m } } \mathbb { P } \left( y = \nu , x _ { j _ { 1 } } = \beta _ { 1 } , \ldots , x _ { j _ { m } } = \beta _ { m } \mid \alpha \right) ,
$$

with the analogous expression for $\varphi _ { \alpha } ^ { m }$ . For $m = 0$ , we define

$$
\varphi _ { \alpha } ^ { y ; 0 } ( \nu ) : = \mathbb { P } \left( y = \nu \mid \alpha \right) .
$$

The corresponding tensors are

$$
\varphi _ { \alpha } ^ { m } : = \sum _ { \beta _ { 1 } , \dots , \beta _ { m } \in \mathcal { V } } \varphi _ { \alpha } ^ { m } ( \beta _ { 1 } , \dots , \beta _ { m } ) \bigotimes _ { i = 1 } ^ { m } e _ { \beta _ { i } } ,
$$

$$
\varphi _ { \alpha } ^ { y ; m } : = \sum _ { \nu , \beta _ { 1 } , \dots , \beta _ { m } \in \mathcal { V } } \varphi _ { \alpha } ^ { y ; m } ( \nu , \beta _ { 1 } , \dots , \beta _ { m } ) \boldsymbol { e } _ { \nu } \otimes \bigotimes _ { i = 1 } ^ { m } \boldsymbol { e } _ { \beta _ { i } } .
$$

We refer to $\varphi _ { \alpha } ^ { m }$ and $\varphi _ { \alpha } ^ { y ; m }$ as the order-m co-occurrence signature and the order-m label signature of $\alpha ,$ respectively. Each signature is a normalized probability tensor. Moreover, the hierarchy is marginally consistent: marginalizing any context mode of an order-m signature recovers the corresponding order-(m − 1) signature. We formalize this property below and use it both in the small-initialization analysis and in the frequency interpretation of the signature hierarchy.

In this work, we first focus on sequence prediction settings in which the context tokens are exchangeable, so that their statistical role does not depend on their specific positions in the sequence. This abstraction is also consistent with classical permutation-invariant embedding models such as CBOW, where context tokens are aggregated without distinguishing their order. We later relax this assumption in our analysis of self-attention, where diferent token positions are explicitly distinguished and position-dependent signatures are considered.

## 3.3 Parameter Initialization

Throughout this paper, all parameter matrices are initialized as

$$
W _ { i j } \sim { \mathcal { N } } ( 0 , d _ { \mathrm { i n } } ^ { - 2 \gamma } ) ,
$$

where $d _ { \mathrm { i n } }$ is the input dimension of the matrix. Equivalently, each coordinate has standard deviation $d ^ { - \gamma }$ . The hyperparameter $\gamma$ controls the initialization scale: larger $\gamma$ corresponds to smaller initialization. We use the small-initialization regime $\begin{array} { r } { \gamma > \frac { 1 } { 2 } } \end{array}$ when analyzing the relative scale of diferent signature contributions at initialization. This regime is closely related to the condensed/nonlinear training regime studied in prior work (Luo et al., 2021; Zhou et al., 2022).

## 3.4 Main Result: Context Staircase

With the hierarchy of signatures defined above, we now introduce the central phenomenon studied in this work. We find that small initialization induces a characteristic progressive evolution of token embeddings, which we term Context Staircase. Under small initialization, the embedding geometry is initially dominated by low-order signatures, especially the zeroth-order label signature. As training proceeds, the influence of higher-order signatures gradually increases, and richer contextual structures become increasingly visible in the embedding space.

Figure 1 illustrates this phenomenon in the addition task. At the early stage of training, the number embeddings form a smooth ordered structure that closely matches the zeroth-order signature. Later in training, an additional odd–even structure emerges, which is contained in the higher-order signature. Similar transitions are observed across the controlled tasks studied in the following sections.

The theoretical results of this paper provide a simple explanation for this progressive evolution.

Proposition (Low-order dominance under small initialization, informal). Under small initialization, higher-order signatures have a much weaker influence on the embedding dynamics than the zeroth-order signature. Consequently, the embedding space is shaped predominantly by low-order statistical structure at the beginning of training.

The intuition is that higher-order signature contributions involve more embedding factors. When the embeddings are initialized at a suficiently small scale, these additional factors strongly suppress the corresponding contributions. This creates a natural hierarchy among signature orders at the beginning of training.

Proposition (Emergence of higher-order signatures during training, informal). As training proceeds and the parameter scale grows, the initial suppression of higher-order signatures gradually disappears. Higher-order signatures can then have a substantial influence on the embedding dynamics together with the lower-order signatures. In the late-stage regime considered in our analysis, the highest accessible signature order has the strongest influence among them.

Together, these two results explain the basic mechanism of Context Staircase: small initialization first favors simple, low-order statistical relations, while training progressively allows richer, higher-order contextual statistics to shape the embedding space. Importantly, higher-order signatures do not necessarily replace lower-order ones. Rather, multiple signature orders can jointly influence the late-stage embedding geometry, with the highest accessible order becoming the most prominent under the conditions of our analysis.

The following sections provide the detailed empirical evidence and formal analysis for this phenomenon. We first study Context Staircase in a controlled feed-forward model, and then examine how it extends to self-attention architectures and real language-model training.

## 4 Signature-Aligned Dynamics in Feed-Forward Models

We first study a controlled feed-forward model. This setting allows us to present the empirical signature-alignment phenomenon in its simplest form and then derive a gradient-flow explanation for why the same statistical structures enter the embedding updates.

## 4.1 Model Definition

We first consider the feed-forward network and study how the embedding space develops. Given an input sequence $X = ( x _ { 1 } , \dots , x _ { L } )$ , we define that:

$$
\pmb { f } _ { \mathrm { f f n } } ( \boldsymbol { X } ) : = \boldsymbol { W } ^ { U } \pmb { g } _ { \mathrm { f f n } } ( \boldsymbol { X } ) = \boldsymbol { W } ^ { U } \boldsymbol { \sigma } \left( \sum _ { i = 1 } ^ { L } \pmb { w } _ { x _ { i } } ^ { E } \right) ,\tag{3}
$$

where $\sigma$ denotes an element-wise activation. This model can be viewed as a CBOW-type architecture (Mikolov et al., 2013a). We use this controlled model first to expose the empirical signature-alignment phenomenon and then to derive the gradient-flow mechanism that can generate it.

## 4.2 Empirical Signature Alignment

We begin with the central empirical observation of this work. In controlled sequenceprediction tasks, the geometry of the learned token embeddings evolves through a hierarchy of token-conditioned signatures. The early embedding space is well explained by the zeroorder label signature, whereas the efects of higher-order signatures become jointly visible later in training. We first document this empirical change and then use gradient-flow analysis to explain why diferent signature orders enter the embedding dynamics and how their relative scales evolve during training.

Alignment with $\varphi _ { \alpha } ^ { y ; 0 }$ in the early stage. We first examine how the embedding space is organized during the early stage of training, when the model has only begun to extract statistical regularities from the data. Across a series of controlled tasks, we observe that the emerging embedding structures are not arbitrary, but exhibit a close and systematic relation to the zeroth-order label signature $\varphi _ { \alpha } ^ { y ; 0 }$ . We study this phenomenon from two complementary perspectives: the embedding structure of an individual token and the geometric organization of the embedding space across multiple tokens. In the first class of tasks, we prescribe a classical label distribution for a specific token a and compare the resulting early-stage embedding of a with this distribution. In the second class of tasks, we construct datasets in which the zeroth-order signatures of diferent tokens exhibit predefined geometric patterns, and examine whether the corresponding embeddings develop analogous structures.

We construct the first class of tasks as follow:

Task 1. Let V be a finite vocabulary and let Y be a finite label space. We fix a single anchor token $a \in \nu$ and a predefined context-token set ${ \mathcal { X } } \subseteq { \mathcal { V } } \setminus \{ a \}$ . The input space associated with the anchor token a is defined as

$$
{ \mathcal { D } } _ { a } : = \{ ( a , x _ { 1 } , x _ { 2 } ) : x _ { 1 } , x _ { 2 } \in \chi \} .
$$

For each sequence $( a , x _ { 1 } , x _ { 2 } ) \in \mathcal { D } _ { a }$ , we assign its target label by independently sampling

$$
y ( a , x _ { 1 } , x _ { 2 } ) \sim p _ { a } ,
$$

where $p _ { a } \in \Delta ( \mathcal { V } )$ is a prescribed label distribution associated with the anchor token a.

We compare $\varphi _ { a } ^ { y ; 0 }$ with the distribution induced by the embedding logits, softmax $\left( W ^ { U } w _ { a } ^ { E } \right)$ , under four prescribed label distributions: uniform, normal, Poisson, and exponential.

As shown in Figure 2A, after a short period of training, the distribution induced by the embedding logits of token a closely matches its zeroth-order label signature $\varphi _ { a } ^ { y ; 0 }$ . This correspondence is consistently observed across all four target distributions, as reflected by the high correlation and small distributional distance between softmax $\left( W ^ { U } w _ { a } ^ { E } \right)$ and $\varphi _ { a } ^ { y ; 0 }$ (In Figure 2A, r denotes the Pearson correlation coeficient, KL denotes the Kullback–Leibler divergence, and JSD denotes the Jensen–Shannon divergence between the two distributions). These results show that the statistical profile encoded by the early-stage embedding of token a is closely associated with its zeroth-order label distribution.

Then we construct the second type of tasks as follow:

Task 2. Let $\mathcal { X } = \{ 1 , \ldots , N \}$ be a finite set of number tokens, we consider input sequences of length two,

$$
X = ( x _ { 1 } , x _ { 2 } ) , \qquad x _ { 1 } , x _ { 2 } \in \mathcal { X } .
$$

For each input sequence $( x _ { 1 } , x _ { 2 } )$ , the target label is determined by an arithmetic rule

$$
y = F ( x _ { 1 } , x _ { 2 } ) ,
$$

where F is chosen from one of the following operations:

$$
F _ { \mathrm { a d d } } ( x _ { 1 } , x _ { 2 } ) = x _ { 1 } + x _ { 2 } , \quad F _ { \mathrm { s u b } } ( x _ { 1 } , x _ { 2 } ) = | x _ { 1 } - x _ { 2 } | , \quad F _ { \mathrm { m u l } } ( x _ { 1 } , x _ { 2 } ) = x _ { 1 } x _ { 2 } .
$$

In addition, we construct the task $F _ { \mathrm { m o d } } ( x _ { 1 } , x _ { 2 } ) = x _ { 1 } + x _ { 2 } + 1$ mod N, which will be used for the analysis in the next section.

These tasks are chosen because they induce distinctive and interpretable label-distribution structures over number tokens, which makes the comparison between embedding geometry and signature geometry more direct.

Let ${ \pmb w } _ { a } ^ { E , ( t ) }$ denote the learned input embedding of token a at epoch t. For every pair of number tokens $a , b \in { \mathcal { X } }$ , we compute the pairwise embedding similarity

$$
S _ { a , b } ^ { \mathrm { e m b } , ( t ) } = \frac { \langle { \pmb w } _ { a } ^ { E , ( t ) } , { \pmb w } _ { b } ^ { E , ( t ) } \rangle } { \| { \pmb w } _ { a } ^ { E , ( t ) } \| _ { 2 } \| { \pmb w } _ { b } ^ { E , ( t ) } \| _ { 2 } } ,\tag{4}
$$

and the pairwise signature similarity

$$
S _ { a , b } ^ { \mathrm { s i g , ( 0 ) } } = \frac { \langle \varphi _ { a } ^ { y ; 0 } , \varphi _ { b } ^ { y ; 0 } \rangle } { \| \varphi _ { a } ^ { y ; 0 } \| _ { 2 } \| \varphi _ { b } ^ { y ; 0 } \| _ { 2 } } .\tag{5}
$$

We denote $S ^ { \mathrm { e m b , } ( t ) }$ as the cosine similarity matrix of all tokens, and similar with the $S ^ { \mathrm { s i g , ( 0 ) } }$ We then quantify the agreement between these two geometries by computing the correlation between their of-diagonal entries:

$$
\begin{array} { r } { \rho _ { ( 0 ) } ^ { ( t ) } = \mathrm { C o r r } \left( \{ S _ { a , b } ^ { \mathrm { e m b , ( } t ) } : a , b \in \mathcal { X } , a < b \} , \{ S _ { a , b } ^ { \mathrm { s i g , ( 0 ) } } : a , b \in \mathcal { X } , a < b \} \right) . } \end{array}\tag{6}
$$

A larger value of $\rho$ indicates that the learned embedding geometry is more consistent with the signature geometry.

As shown in Figure 2 B-D, the S<sup>emb,(t)</sup> exhibit highly similar structural patterns to the corresponding $\overrightharpoon { S } ^ { \mathrm { s i g } , ( 0 ) }$ across all three arithmetic tasks. For the addition task, both matrices display a smooth diagonal band structure, reflecting the gradual change of label distributions across nearby number tokens. For the absolute-subtraction task, both matrices show a symmetric “X”-shaped pattern.

We further visualize the structures of embedding space and signature space by the lens of PCA projection in Figure 2 E, F, respectively. The PCA projections provide a more intuitive visualization of the strong correspondence between the two spaces. In the addition and subtraction tasks, both the embedding space and the signature space exhibit the same ordered geometric structure. In the multiplication task, the numbers instead form distinct clusters according to their odd components. For example, {1, 2, 4, 8, 16} form one cluster; {3, 6, 12} form another; and {5, 10, 20} form a third. These results establish the first part of the empirical phenomenon: during early training, numerical embeddings acquire spatial structures that closely match the geometry of the zero-order label signatures.

Emergence of higher-order signature efects. We next examine how the embedding geometry continues to evolve beyond the early stage of training. A natural question is whether the embedding geometry remains confined to the structure associated with the zeroth-order signature throughout training, or whether additional structures emerge as training proceeds. To investigate this question, we consider two complementary settings: one in which the zeroth-order signature already contains suficient information for solving the task, and another in which it cannot distinguish diferent input tokens or capture the structure required by the task. This comparison allows us to examine whether the emergence of additional embedding structures depends on their necessity for task learning.

To illustrate this process, we compare two arithmetic tasks: $F _ { \mathrm { a d d } } \left( x _ { 1 } , x _ { 2 } \right) = x _ { 1 } + x _ { 2 }$ and $F _ { \mathrm { m o d } } \left( x _ { 1 } , x _ { 2 } \right) = x _ { 1 } + x _ { 2 } + 1$ mod N in Task 2. Figure 3 A,B show how the training loss and accuracy evolve over the training for the two tasks, together with the learned embeddings at the early and late stages of training. These two tasks difer significantly in their early stage. In $F _ { \mathrm { a d d } }$ , diferent number tokens induce distinct label distributions. Therefore, the embeddings form an ordered geometry. In contrast, for $F _ { \mathrm { m o d } }$ , all number tokens have identical label distributions. As a result, the early embeddings are almost aligned in the same direction. In this case, the zero-order signature provides almost no useful information for distinguishing diferent number tokens, and the model fails to learn the task during the early stage. Despite these diferent early-stage behaviors, both tasks exhibit a clear change in embedding geometry as training continues. At the late stage, the embeddings no longer simply reflect the zero-order signatures, but a more complex structure. In the addition task, the embeddings of diferent number tokens exhibit an odd-even separated structure in the late stage of training. In the modular-addition task, the embeddings of diferent number tokens form a modulo-19 ring structure: the successor of token 1 is (1 + 19) mod $N = 2 0$ and the next token is (20 + 19) mod $N = 3 9$

![](images/9bafbd427a42f7f5bc8d867eb4e78f86c53b02643023c06463931c7f09c79fb8.jpg)  
Figure 2: A: Comparison between the $\varphi _ { a } ^ { y ; 0 }$ and $\mathrm { s o f t m a x } ( \pmb { W } ^ { U } \pmb { w } _ { a } ^ { E } )$ in Task 1 under four target label distributions: uniform, normal, Poisson, and exponential. Here, r denotes the Pearson correlation coeficient, KL denotes the Kullback–Leibler divergence, and JSD denotes the Jensen–Shannon divergence between the two distributions. B: Heatmap of $S ^ { \mathrm { e m b , } ( t ) }$ for the addition, absolute-subtraction, and multiplication tasks, with $t = 1 0 0$ . C: Heatmap of $S ^ { \mathrm { s i g } }$ for the three arithmetic tasks. D: Pearson and Spearman correlations between the of-diagonal entries of $S ^ { \mathrm { e m b , } ( t ) }$ and $S ^ { \mathrm { s i g } }$ across the three arithmetic tasks, where $t = 1 0 0$ . E: PCA projections of the learned token embeddings for the three arithmetic tasks at epoch 100. F: PCA projections of the corresponding zero-order label signatures for the three arithmetic tasks.

To quantify this process, we measure the correlation between $S ^ { \mathrm { s i g , ( 0 ) } }$ and $S ^ { \mathrm { e m b , } ( t ) }$ across t (see (4),(5) and (6)). Furthermore, we define that

$$
S _ { a , b } ^ { \mathrm { s i g , ( 1 ) } } = \frac { \langle \varphi _ { a } ^ { y ; 1 } \pmb { v } , \varphi _ { b } ^ { y ; 1 } \pmb { v } \rangle } { \lVert \varphi _ { a } ^ { y ; 1 } \pmb { v } \rVert _ { 2 } \lVert \varphi _ { b } ^ { y ; 1 } \pmb { v } \rVert _ { 2 } } ,\tag{7}
$$

where $\boldsymbol { v } \in \mathbb { R } ^ { | \mathcal { X } | }$ is the probe vector which is obtained through optimizing the v by maximizing the correlation between $S ^ { \mathrm { s i g , ( 1 ) } }$ and S<sup>emb,(t)</sup> for each t. Figure 3 C exhibits the correlation and MSE between $S ^ { \mathrm { e m b , } ( t ) }$ and $S ^ { \mathrm { s i g , ( 0 ) } } , \ S ^ { \mathrm { s i g , ( 1 ) } }$ . It’s noted that in Modular addition task, $S ^ { \mathrm { s i g , ( 0 ) } }$ is the matrix whose all elements equal to 1 and the pearson correlation between S<sup>sig,(0)</sup> and S<sup>emb</sup> is invalid. So we measure the correlation here by $1 - \operatorname { M S E } ( S ^ { \mathrm { s i g } , ( 0 ) } , S ^ { \mathrm { e m b } } )$ As shown, S<sup>sig,(0)</sup> explains the embedding geometry almost perfectly at the beginning of training, achieving correlation close to 1 and near-zero MSE. As training proceeds, however, the correlation drops substantially. In contrast, $S ^ { \mathrm { s i g , ( 1 ) } }$ becomes a much better predictor of the embedding geometry after the early stage. It maintains a high correlation and achieves consistently lower prediction error than S<sup>sig,(0)</sup> throughout later training.

These results reveal a progressive change in the statistical structure reflected by the embeddings. During the early stage of training, the embedding geometry is closely associated with the zeroth-order signature. As training proceeds, this association gradually weakens, while structures related to the first-order signature become increasingly visible. The latestage behavior depends on the information contained in the zeroth-order signature. When it is already informative for the task, part of the corresponding structure can remain visible in the embedding space. In contrast, when the zeroth-order signature cannot distinguish diferent input tokens, the late-stage embedding geometry exhibits little relation to it and is instead much more closely associated with the first-order signature. Since the sequences considered here have length two, the first-order signature is the only accessible higher-order signature. The joint efects of multiple higher-order signatures are further examined in Appendix C.

## 4.3 Gradient-Flow Explanation

We now analyze the gradient flow of the embeddings to explain the evolution of embedding structures observed above, from their early association with the zeroth-order signature to

![](images/836d6eaf025e78647619b58409c09f269ddf940c1cd2198af5ebaac83b7d4336.jpg)

![](images/ac97f7787d1f40d3d65b559a1d7848641bbb490b432e03fdddd93c8532a5d34c.jpg)

![](images/88ed8473a44d810323ac91db7495d1c990ff940463d63ebdb86f2e27a6b7c945.jpg)

![](images/268b1109092f379b1772bcd820f6b5485253a8b9bf90424923f4fea633faa796.jpg)

![](images/6bff35dbbdd4cbc12d2f6c5ed169e0ea70c2633f7da4e5fb7f745fd7500e733a.jpg)

![](images/296e4f3226856583646343b21c53489b349dc07f0d7bc937d36fb1de1c6ce6f5.jpg)

![](images/86c72d8129ffe336f49d45cb1047a10f63cea6b875fa08b89e7523c5670a7c77.jpg)

![](images/a9aa60f326b89eb9249fe39e434ff29092ca6e43afa65e8e22eded4944047bf8.jpg)

![](images/08d79e48d96586ea7309c613e253eb7adbd66ecf67f1a073df17b733ea7c6146.jpg)

![](images/7f0a751d35912e44f3f9a5d977a4c010a34eecb7df68e1e59f6a5cd5b2d02927.jpg)

![](images/a2e7368f92c795406090b060a2addca36ffe6afed1a9aed660d18ea73f6b5ba5.jpg)

![](images/e75e9e50265d3c0e8662e93eff959c6d750f3897d836fdb9f59a7808c76ebac6.jpg)

![](images/545c2b2c33a1cf319c8b5d4eb60399222cc91b0f584442228039645797c2c1fa.jpg)

![](images/0b95b00dbc462825cfe918e431e51bd1350f641f1eecc189ad08f5e7135e7043.jpg)  
Figure 3: A: Training loss and training accuracy over epochs for the addition task and the modular-addition task. The dashed vertical lines mark the early and late epochs used for visualization. B: Cosine-similarity matrices and PCA projections of the learned token embeddings at the early and late epochs for the addition task and the modular-addition task. C: Temporal evolution of the correlation and MSE between the $S ^ { \mathrm { e m b , } ( t ) }$ and the $S ^ { \mathrm { s i g , ( 0 ) } }$ , as well as the optimized $\widetilde { S } ^ { \mathrm { s i g } , ( 1 ) }$ , for the addition task and the modular-addition task.

the emergence of higher-order signature efects later in training. We first establish the following assumation

Assumption 1. We assume that the activation function σ can be approximated by a coordinate-wise polynomial:

$$
\sigma ( z ) = \sum _ { k = 1 } ^ { K } C _ { k } z ^ { \odot k } ,\tag{8}
$$

where max<sub>k</sub> ${ \mathrm { : } } \left| C _ { k } \right| < \infty$ and $z ^ { \odot k }$ denotes the coordinate-wise k-th power.

We then compute the Jacobian of $f _ { \mathrm { f f n } } ( X )$ with respect to the embedding vector $\boldsymbol { w _ { \alpha } ^ { E } }$

$$
\frac { \partial { f _ { \mathrm { { f f n } } } ( X ) } } { \partial { w _ { \alpha } ^ { E } } } = N _ { \alpha } ( X ) { W ^ { U } } { \mathrm { D i a g } } \left( \sigma ^ { \prime } \left( \sum _ { i = 1 } ^ { L } w _ { x _ { i } } ^ { E } \right) \right) .\tag{9}
$$

By the definition of $I ,$ for any integrable function $H ( X , y )$ of the training sample,

$$
\begin{array} { r } { \mathbb { E } \left[ N _ { \alpha } ( X ) \pmb { H } ( X , y ) \right] = L r _ { \alpha } \mathbb { E } \left[ \pmb { H } ( X , y ) \mid \alpha \right] . } \end{array}
$$

Recall (2), then we have that

$$
\frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } = \frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } \bigg | ^ { y } - \frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } \bigg | ^ { p } ,
$$

where

$$
\frac { d \pmb { w } _ { \alpha } ^ { E } } { d t } \bigg | ^ { y } : = L r _ { \alpha } \mathbb { E } \left[ \sigma ^ { \prime } \left( \sum _ { i = 1 } ^ { L } \pmb { w } _ { x _ { i } } ^ { E } \right) \odot \pmb { W } ^ { U , T } \pmb { e } _ { y } \mid \alpha \right] ,
$$

$$
\frac { d \pmb { w } _ { \alpha } ^ { E } } { d t } \bigg | ^ { p } : = L r _ { \alpha } \mathbb { E } \left[ \sigma ^ { \prime } \left( \sum _ { i = 1 } ^ { L } \pmb { w } _ { x _ { i } } ^ { E } \right) \odot \pmb { W } ^ { U , T } \pmb { p } ( X ) \mid \alpha \right] .
$$

For the label-driven term $\left. \frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } \right| ^ { y }$ , we have the following result:

Theorem 1. Define that $\mathcal { W } _ { \alpha } ^ { y , m , k } \in \ L ( \mathbb { R } ^ { d } ) \frac { V \times \cdot \cdot \cdot \times V } { m + 1 }$ is a tensor with order $m + 1$ , whose elements is a d-dimensional vector with

$$
\mathcal { W } _ { \alpha } ^ { y , m , k } \left( \nu , \beta _ { 1 } , \cdot \cdot \cdot , \beta _ { m } \right) = \binom { L - 1 } { m } \sum _ { \stackrel { l \geq 0 , n _ { 1 } , \cdots , n _ { m } \geq 1 } { l \leq \sum _ { r = 1 } ^ { m } n _ { r } = k - 1 } } \frac { ( k - 1 ) ! } { l ! \prod _ { r = 1 } ^ { m } n _ { r } ! } ( w _ { \alpha } ^ { E } ) ^ { \odot l } \odot w _ { \nu } ^ { U } \odot \bigcup _ { r = 1 } ^ { m } ( w _ { \beta _ { r } } ^ { E } ) ^ { \odot n _ { r } } .\tag{10}
$$

With the Assumption 1, we have that

$$
\frac { d { w } _ { \alpha } ^ { E } } { d t } \Big | ^ { y } = L r _ { \alpha } \sum _ { m = 0 } ^ { \operatorname* { m i n } \{ L - 1 , K - 1 \} } \mathcal { A } _ { \alpha , m } \cdot \varphi _ { \alpha } ^ { y ; m } ,\tag{11}
$$

where $\begin{array} { r } { \mathcal { A } _ { \alpha , m } = \sum _ { k = m + 1 } ^ { K } k C _ { k } \mathcal { W } _ { \alpha } ^ { y , m , k } , C _ { k } } \end{array}$ is the approximation constant in (8). Here $\mathcal { A } _ { \alpha , m }$ $\varphi _ { \alpha } ^ { y ; m }$ denotes the contraction over the label index and the m context-token indices, resulting in a vector in $\mathbb { R } ^ { d }$

Theorem 1 provides the central mechanism behind the empirical observations: label signatures from order 0 up to the largest order appear explicitly in the embedding gradient. The result identifies the statistical components available to drive the embedding updates; the geometric alignment documented above is the corresponding empirical manifestation.

For the prediction-induced term $\begin{array} { r } { \frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } \bigg | ^ { p } } \end{array}$ , to have a similar result, we use the Taylor expansion of the softmax around the origin and keep its leading constant term: $p ( X ) =$ ${ \frac { 1 } { V } } \mathbf { 1 } + O ( \| f ( X ) \| )$ . We have the following result:

Theorem 2. Define that $\mathcal { U } _ { \alpha , 0 } ^ { m , k } \in ( \mathbb { R } ^ { d } ) ^ { \underbrace { V \times \cdots \times V } _ { m } }$ is a tensor with order m, whose element is a d-dimensional vector with

$$
\mathcal { U } _ { \alpha , 0 } ^ { m , k } \left( \beta _ { 1 } , \dots , \beta _ { m } \right) : = \binom { L - 1 } { m } \sum _ { l \ge 0 , \ n _ { 1 } , \dots , n _ { m } \ge 1 } \frac { ( k - 1 ) ! } { l ! \prod _ { r = 1 } ^ { m } n _ { r } ! } b _ { 0 } \odot ( w _ { \alpha } ^ { E } ) ^ { \odot l } \odot \binom { m } { r = 1 } ( w _ { \beta _ { r } } ^ { E } ) ^ { \odot n _ { r } } ,\tag{12}
$$

where $b _ { 0 } : = \mathbf { \Omega } _ { V } ^ { 1 } W ^ { U , T } \mathbf { 1 }$ . Under the Assumption 1 and $\pmb { p } ( X ) = \frac { 1 } { V } \pmb { 1 } + O ( \| \pmb { f } ( X ) \| )$ , then we have

$$
\frac { d { w } _ { \alpha } ^ { E } } { d t } \Big | ^ { p } = L r _ { \alpha } \sum _ { m = 1 } ^ { \operatorname* { m i n } \{ L - 1 , K - 1 \} } { \mathcal { B } } _ { \alpha , m } \cdot \varphi _ { \alpha } ^ { m } + R _ { \alpha } ,\tag{13}
$$

where $\begin{array} { r } { B _ { \alpha , m } = \sum _ { k = m + 1 } ^ { K } k C _ { k } \mathcal { U } _ { \alpha , 0 } ^ { m , k } , R _ { \alpha } } \end{array}$ denotes the remainder term from the softmax expansion.

Remark 1. In Theorem ${ \mathcal { Q } } ,$ we only keep the first-order expansion of the softmax probability $p ( X )$ around the origin. This leads to the $\varphi _ { \alpha } ^ { m }$ up to order min $\{ L - 1 , K - 1 \}$ . If p(X) is expanded to higher orders, then higher-order $\varphi _ { \alpha } ^ { m }$ will be introduced. For example, if the second-order term in the softmax expansion is retained, the highest order of $\varphi _ { \alpha } ^ { m }$ $m = \operatorname* { m i n } { \{ L - 1 , 2 K - 1 \} }$ . Consequently, for any fixed activation order $K _ { \cdot }$ , higher-order terms in the softmax expansion can in principle introduce $\varphi _ { \alpha } ^ { m }$ of any order up to $L - 1$ . In Appendix $B . { \mathcal { Q } } ,$ we provide the explicit form obtained by retaining the second-order term in the softmax expansion.

Remark 2. Although the above analysis is formulated using a polynomial activation, this assumption mainly serves to make the contributions associated with diferent signature orders explicit. Common activation functions used in neural networks, including ReLU and other standard nonlinearities, can be approximated on a bounded neighborhood of the origin by polynomials of suficiently high degree. The polynomial degree K can therefore be chosen much larger than the sequence length L, so that all signature orders accessible to a sequence of length L are included in the expansion.

Our analysis is concerned with how signatures of diferent orders enter and afect the embedding dynamics, rather than with the quantitative approximation error of the activation function itself. We therefore omit the approximation residual and use the polynomial representation as an analytical device throughout the following discussion. Unless otherwise specified, we henceforth consider the regime K $\gg L$

However, in many exchangeable sequence-prediction tasks, the co-occurrence signatures $\varphi _ { \alpha } ^ { m }$ carry little discriminative information about the token α. Intuitively, after conditioning on a selected occurrence of $\alpha ,$ the remaining context tokens are often sampled from a distribution that does not depend on the identity of $\alpha .$ More formally, suppose that the tokens in $X _ { - \alpha }$ are sampled independently from position-dependent distributions $\pi _ { i } ,$ and that each $\pi _ { i }$ is independent of the conditioned token α. Then, for any index subset $j _ { 1 } , \ldots , j _ { m } \subseteq$ $[ L - 1 ]$ , we have

$$
\mathbb { P } \left( x _ { j _ { 1 } } = \beta _ { 1 } , \cdot \cdot \cdot , x _ { j _ { m } } = \beta _ { m } \mid \alpha \right) = \mathbb { P } \left( x _ { j _ { 1 } } = \beta _ { 1 } , \cdot \cdot \cdot , x _ { j _ { m } } = \beta _ { m } \right) , \qquad \beta _ { 1 } , \cdot . . . , \beta _ { m } \in \mathcal { V } .
$$

Therefore, the $\varphi _ { \alpha } ^ { m }$ are identical across diferent α, that is, $\varphi _ { \alpha _ { 1 } } ^ { m } = \varphi _ { \alpha _ { 2 } } ^ { m }$ for any $\alpha _ { 1 } , \alpha _ { 2 } \in \mathcal { V }$ Such signatures do not separate diferent tokens. For this reason, we do not further analyze their efect in the main text, and instead focus on the label signatures $\varphi _ { \alpha } ^ { y ; m }$

## 4.4 Order Hierarchy under Small Initialization

Theorem 1 identifies the signature components that can enter the embedding gradient, but it does not by itself determine their relative scales. We next show that small initialization creates a pronounced asymmetry at the beginning of training: higher-order terms contain more small embedding factors and are therefore suppressed relative to the zero-order term.

Proposition 1. Assume that L and K are fixed and that the model is initialized with scale $d ^ { - \gamma }$ for some $\gamma > 1 / 2$ . Then, for every token α with $r _ { \alpha } = \mathbb { P } \left( x _ { I } = \alpha \right) > 0$ and every fixed $m \geq 1$ 2

$$
\frac { \left( \mathbb { E } _ { \mathrm { i n i t } } \left. \boldsymbol { A } _ { \alpha , m } \cdot { \varphi } _ { \alpha } ^ { y ; m } \right. _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } } { \left( \mathbb { E } _ { \mathrm { i n i t } } \left. \boldsymbol { A } _ { \alpha , 0 } \cdot { \varphi } _ { \alpha } ^ { y ; 0 } \right. _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } } = O \left( d ^ { - m \gamma } \right) = o ( 1 ) .
$$

Proposition 1 indicates that in root-mean-square scale, the zero-order label signature gives the leading contribution to the label-driven embedding gradient at initialization. This hierarchy provides a mechanistic explanation for the robust early-stage zero-order alignment observed in Figures 2 and 3.

Proposition 1 explains why the zeroth-order signature is most visible at the beginning of training. Under small initialization, higher-order signature terms are strongly suppressed because they involve additional powers of the small parameter scale. As training proceeds and the parameter scales grow, this suppression becomes weaker. Consequently, higherorder signatures can increasingly contribute to the embedding dynamics together with lowerorder signatures. Proposition 2 formalizes this late-stage behavior.

Proposition 2. Assume that K is suficiently large relative to L, and that $C _ { K } \neq 0$ . Let $s _ { E } ( t )$ and $s _ { U } ( t )$ denote the typical scales of the embedding and unembedding vectors at time t. Suppose that $0 < s _ { E } ( t ) , s _ { U } ( t ) < \infty$ , that $s _ { E } ( t )$ becomes suficiently large during training, and that the normalized contractions and label signatures are uniformly non-degenerate. Then, in this late-stage regime, all accessible signature-order contributions have the same leading scale:

$$
\| A _ { \alpha , m } ( t ) \cdot \varphi _ { \alpha } ^ { y ; m } \| _ { 2 } \asymp s _ { U } ( t ) s _ { E } ( t ) ^ { K - 1 } , \qquad 0 \leq m \leq L - 1 ,
$$

where the comparison constants are independent of t. Thus, no signature order is suppressed by an additional power of the parameter scale. Moreover, among these same-scale contributions, the highest-order term is the largest:

$$
\begin{array} { r } { \big \| \boldsymbol A _ { \alpha , L - 1 } ( t ) \cdot \boldsymbol \varphi _ { \alpha } ^ { y ; L - 1 } \big \| _ { 2 } \geq \| \boldsymbol A _ { \alpha , m } ( t ) \cdot \boldsymbol \varphi _ { \alpha } ^ { y ; m } \| _ { 2 } , \qquad 0 \leq m < L - 1 . } \end{array}
$$

The mechanism can be seen directly from the definition $\begin{array} { r } { \mathcal { A } _ { \alpha , m } = \sum _ { k = m + 1 } ^ { K } k C _ { k } \mathcal { W } _ { \alpha } ^ { y ; m , k } } \end{array}$ For a fixed signature order $m ,$ the summation starts from degree $m + 1$ and ends at the common maximum degree K. The degree-k term contains one unembedding factor and k − 1 embedding factors, and therefore has a typical scale proportional to $s _ { U } ( t ) s _ { E } ( t ) ^ { k - 1 }$ This simple structure explains the diferent behaviors at the beginning and later stages of training.

At small initialization, $s _ { E } \ll 1$ , so the lowest-degree term in each $\mathcal { A } _ { \alpha , m }$ is the largest. Since the lowest possible degree is $k = m + 1$ , the order-m contribution starts at the scale $s _ { U } s _ { E } ^ { m }$ . Thus, increasing the signature order by one introduces one additional factor of the small embedding scale. This produces the hierarchy $m = 0 > m = 1 > m = 2 > \cdots$ in magnitude at initialization, which is precisely the suppression described in Proposition 1.

As training proceeds and $s _ { E } ( t )$ becomes large, the opposite end of the sum becomes dominant. For every accessible signature order $m _ { \colon }$ the largest degree is now the same, namely $k = K$ . Consequently, the dominant term in every $\mathcal { A } _ { \alpha , m }$ contains the same number $K - 1$ of embedding factors, and all signature orders therefore share the same leading parameter scale $s _ { U } ( t ) s _ { E } ( t ) ^ { K - 1 }$ The scale separation between diferent signature orders thus disappears: higher-order signatures are no longer intrinsically suppressed and can act together with lower-order signatures on the embedding dynamics.

After this common parameter scale is factored out, the remaining diference between signature orders comes from their order-dependent combinatorial prefactors (see (10)). For the degree-K contribution, this prefactor contains $B _ { m } ( K ) = ( m + 1 ) ! \left\{ { K - 1 \atop m + 1 } \right\}$ , which grows as $B _ { m } ( K ) \sim ( m + 1 ) ^ { K - 1 }$ for fixed m and large K. Hence, when $K \gg L .$ the highest accessible order $m = L - 1$ has the largest prefactor. This explains why the late-stage dynamics are jointly influenced by signatures of diferent orders, while the highest-order contribution can be the strongest among them.

## 4.5 Relations across Signature Orders from a Fourier Frequency Perspective

Proposition 2 shows that, once the initialization-induced suppression is lifted, all accessible signature-order contributions have the same leading parameter scale and can jointly afect the embedding dynamics. Although the highest-order contribution is the largest among them, this result does not specify how the statistical structures associated with diferent orders are related. In particular, it leaves open whether diferent signature orders introduce mutually independent structures or whether their efects follow an intrinsic organization. We next examine this question from a frequency perspective.

Signatures of diferent orders are not independent statistical objects. By Lemma 1, an order- $( m - 1 )$ signature is obtained by marginalizing any context mode of an order-m signature. For a fixed token $\alpha .$ , the first two label signatures satisfy

$$
\varphi _ { \alpha } ^ { y ; 0 } ( y ) = \mathbb { P } \left( Y = y \mid \alpha \right) , \qquad \varphi _ { \alpha } ^ { y ; 1 } ( y , x ) = \mathbb { P } \left( Y = y , x _ { J _ { 1 } ^ { ( 1 ) } } = x \mid \alpha \right) ,
$$

and hence

$$
\varphi _ { \alpha } ^ { y ; 0 } ( y ) = \sum _ { x \in \mathcal { V } } \varphi _ { \alpha } ^ { y ; 1 } ( y , x ) .
$$

Denote the real Fourier basis by $\begin{array} { r } { \pmb { c } _ { k } ^ { \mathrm { c o s } } = \left( \cos \frac { 2 \pi k j } { p } \right) _ { i = 0 } ^ { p - 1 } } \end{array}$ , where $k = 0 , 1 , \ldots , p / 2$ is the frequency and $p = | \mathcal { V } |$ in our settings. Since $\pmb { c } _ { 0 } ^ { \mathrm { c o s } } = \mathbf { 1 }$ , the marginalization above can be written as

$$
\varphi _ { \alpha } ^ { y ; 0 } = \varphi _ { \alpha } ^ { y ; 1 } c _ { 0 } ^ { \mathrm { c o s } } .
$$

More generally, contracting any context mode of an order-m signature with $\boldsymbol { c } _ { 0 } ^ { \mathrm { c o s } }$ gives the corresponding order- $( m - 1 )$ signature. In particular, contracting the last context mode yields

$$
\varphi _ { \alpha } ^ { y ; m - 1 } = \varphi _ { \alpha } ^ { y ; m } \pmb { c } _ { 0 } ^ { \mathrm { c o s } } , \qquad m = 1 , 2 , \ldots , L - 1 .
$$

Thus, lower-order signatures are already contained in the zero-frequency projections of higher-order signatures, while the nonzero-frequency components encode the additional structures available at higher orders. The efects of diferent signature orders therefore need not be interpreted as the appearance of unrelated structures.

This relation suggests a possible temporal organization of the empirical dynamics: the embedding may first reflect the zero-frequency structure shared with lower-order signatures and later incorporate additional task-dependent nonzero-frequency components. This temporal statement is not implied by Proposition $2 ;$ we test it empirically by analyzing the frequency components of the first-order signature in $F _ { \mathrm { a d d } }$ and $F _ { \mathrm { m o d } }$ . For each number token α and each frequency $k ,$ we define

$$
\pmb { h } _ { \alpha , k } ^ { ( 1 ) } = \varphi _ { \alpha } ^ { y ; 1 } \pmb { c } _ { k } ^ { \mathrm { c o s } } , \qquad k = 0 , 1 , \dots , p / 2 .
$$

We then compute the pairwise cosine-similarity matrix induced by this frequency component:

$$
S _ { k ; a , b } ^ { \mathrm { s i g } , ( 1 ) } = \frac { \langle h _ { a , k } ^ { ( 1 ) } , h _ { b , k } ^ { ( 1 ) } \rangle } { \| h _ { a , k } ^ { ( 1 ) } \| _ { 2 } \| h _ { b , k } ^ { ( 1 ) } \| _ { 2 } } , \qquad a , b \in \mathcal { X } .
$$

Finally, for each frequency $k ,$ we measure the agreement

$$
\rho _ { ( 1 ) , k } ^ { ( t ) } = \mathrm { C o r r } \left( \{ S _ { k ; a , b } ^ { \mathrm { s i g } , ( 1 ) } : a < b \} , \{ S _ { a , b } ^ { \mathrm { e m b } , ( t ) } : a < b \} \right) .
$$

We track $\rho _ { ( 1 ) , k } ^ { ( t ) }$ over training in $F _ { \mathrm { a d d } }$ and $F _ { \mathrm { m o d } }$ . Figure 4 A, B reveal a clear frequencywise transition. At the earliest stage, the embedding similarity structure is most strongly correlated with the $k = 0$ component and several low-frequency components. As training proceeds, the correlation with the $k = 0$ component decreases, while correlations with nonzero components increase, including the task-specific components at $k = 2 0$ in $F _ { \mathrm { a d d } }$ and $k = 1 9$ in $F _ { \mathrm { m o d } }$

Figure 4 C, D further compare the number embeddings with the corresponding $h _ { \alpha , k } ^ { ( 1 ) }$ at the early checkpoint (epoch 100) and the final checkpoint (epoch 1000). Early in training, the embedding space closely resembles the zero-frequency projection of the first-order signature. Later, the nonzero-frequency components also become strongly expressed and closely match the more complex embedding geometry. These results indicate that the efects of diferent signature orders do not emerge as unrelated structures. Instead, the empirical transition from lower-order to higher-order signature efects can be interpreted as a progressive spectral refinement from the shared zero-frequency structure to additional task-specific nonzero-frequency components.

A particularly interesting phenomenon is that changing only the random seed can lead to substantially diferent embedding geometries, even when the dataset and training configuration remain unchanged. We find that this variation can be explained by the embeddings aligning with diferent frequency components of the signatures. The corresponding results are presented in Appendix D.

![](images/c1e68de6aec1e327b7ec5cbf39488265e76317d8de8cdadabc1ce1d2696c4bf4.jpg)

![](images/df19bfbd4b0dc241761f83384374125f7ce44710e79d01f2cf8d81fb721625e3.jpg)

C  
![](images/c7e84a1319650606ff5dd5ea39b24a4fbbd624cb56c453bf162ceb91a1e769a1.jpg)

![](images/0e549d9a3b28e5d1d59c02e5c97b8f9681ad8cbb575a9304dcd41b3bb76c66a5.jpg)

D  
![](images/a3bc724cd43c472fb9922ecdf25e7d11f5507e92397fb1e975fa9685f9272ba0.jpg)

![](images/f111a85dbe245ea06bc149e995dfdf7c4bd00983a640b5beee5a2976e281d0d6.jpg)

![](images/35576f98239f11f1c95410ccc46719a4f316ca1e5acc59a1318d01496d2cc27d.jpg)

![](images/b2b0e1c6cd442927a2850b3a1ca702fd758df81455493c0fd7236d1ea8598a95.jpg)

![](images/908e69e5d86f80e8ef24812ed11fc7e59aff6370a7b821195d88644bb8cc9265.jpg)

![](images/45caa402535262136fededc3a2c4568237e96a2735e013a6968fb3bfefe8d69c.jpg)  
Figure 4: A: Heatmap of $\rho _ { ( 1 ) , k } ^ { ( t ) }$ across training epochs t and Fourier frequencies k for the addition task. B: The same correlation heatmap for the modular-addition task. C: PCA projections for the addition task, showing $\boldsymbol { w _ { \alpha } ^ { E } }$ at epochs 100 and 1000 together with $h _ { \alpha , k } ^ { ( 1 ) }$ at frequencies $k ~ = ~ 0$ and $k \ : = \ : 2 0$ , respectively. D: PCA projections for the modular-addition task, showing $\boldsymbol { w _ { \alpha } ^ { E } }$ at epochs 100 and 1000 together with $h _ { \alpha , k } ^ { ( 1 ) }$ at frequencies $k = 0$ and $k = 1 9$ , respectively.

## 4.6 A Possible Benefit of Context Staircase.

Context Staircase suggests a particular learning order induced by small initialization. At the beginning of training, low-order signatures have a stronger efect on the embedding dynamics, while higher-order signatures become increasingly important as training proceeds. The zeroth-order signature only describes the relation between a token and the label. The firstorder signature additionally includes one context token, and each higher order incorporates one more context token. Therefore, Context Staircase gradually introduces statistical relations involving increasingly more contextual information, from simple token–label relations to more complex relations involving the full context.

Importantly, this ordering does not imply faster learning in every task. Its efect depends on whether the low-order signatures already contain information useful for the task.

For example, in the modular-addition task, all number tokens have the same zeroth-order signature. The statistical structure emphasized at the beginning of training therefore cannot distinguish diferent tokens or support the solution of the task. Learning only begins to make substantial progress when higher-order signatures become influential. In such a case, small initialization can actually delay task learning at the early stage.

The possible benefit of small initialization should therefore be understood as imposing an ordered rather than an immediately task-optimal learning process. Instead of allowing statistical relations of diferent orders to afect the embeddings at comparable scales from the beginning, it induces Context Staircase, in which simpler relations are learned first and more context-dependent relations are incorporated later. This simple-to-complex ordering may provide a useful inductive bias for representation learning, although whether it accelerates or delays optimization depends on which signature orders contain the information required by the task.

This perspective also suggests an analogy between Context Staircase and the frequency principle of neural networks. The frequency principle states that neural networks tend to learn low-frequency components of a target function before higher-frequency ones during training (Xu et al., 2020, 2019, 2024; Rahaman et al., 2019). Although the notion of complexity is diferent in the two settings, the learning order is conceptually similar. The frequency principle orders structures by their frequency, whereas Context Staircase orders statistical relations by the amount of contextual information they involve. Both phenomena therefore reflect an implicit preference of neural-network training dynamics for learning relatively simple structures before more complex ones. From this perspective, Context Staircase may be viewed as an analogous implicit bias in the space of contextual statistics, providing a possible statistical counterpart to the low-to-high frequency learning behavior observed in neural networks.

## 4.7 Activation Order Controls the Accessible Signature Order

Theorem 1 shows that an activation of order K exposes label signatures only up to order min $\{ L - 1 , K - 1 \}$ . This motivates a direct architectural test: in a construction where lowerorder signatures cannot distinguish the number tokens, successful learning should emerge only when the activation can expose the required full-context signature. We consider length-$L \ F _ { \mathrm { m o d } }$ tasks, where lower-order signatures are identical across diferent number tokens and only the highest-order signature $\varphi _ { \alpha } ^ { y ; L - 1 }$ can distinguish them. In this construction, the gradient-flow decomposition therefore predicts a sharp change when the activation order reaches the sequence length. To test this prediction, we employ the activation $\sigma ( z ) = z ^ { \odot K }$ and vary both the sequence length L and the activation order K. For each configuration, we record the maximum training accuracy and visualize the cosine-similarity matrix of the learned token embeddings. As shown in Figure 5, when the activation order is smaller than the sequence length, the model is unable to learn the task at all; correspondingly, the embeddings of diferent numerical tokens collapse into the same direction. In contrast, once the activation order reaches or exceeds the sequence length, the model can successfully learn the task, and the embedding similarity matrix develops a clear nontrivial structure, consistent with the signature orders accessible in the gradient-flow decomposition.

![](images/f0feb38c0c188cc7c59d20d077db4d1ba7e2d05c8713bef16a2537e4e12c389e.jpg)

B  
![](images/96559f445575e34de995e16f43cc18f38592bf1e79f9ef0fb4a4df5559e2be14.jpg)  
Figure 5: A: Heatmap of the maximum training accuracy achieved for modular-addition tasks with diferent sequence lengths and activation orders. B: Cosine-similarity matrices of the learned token embeddings for the corresponding task settings with diferent sequence lengths and activation orders.

## 5 Attention-Modulated Signature Dynamics

The feed-forward experiments establish that embedding geometry can progressively align with a hierarchy of signatures, while the gradient-flow decomposition explains why those signatures enter the updates. We next ask how self-attention changes this picture. The central diference is that attention routes statistical information non-uniformly: the contribution of a token depends not only on its contextual distribution, but also on the attention mass assigned to it during prediction.

## 5.1 Model Definition

Recall that the input embedding matrix for a sequence $X = ( x _ { 1 } , \dots , x _ { L } )$ is

$$
\begin{array} { r } { W _ { X } ^ { E } = [ \pmb { w } _ { x _ { 1 } } ^ { E } , \pmb { w } _ { x _ { 2 } } ^ { E } , \pmb { \mathrm { \ldots } } , \pmb { w } _ { x _ { L } } ^ { E } ] \in \mathbb { R } ^ { d \times L } . } \end{array}
$$

Denote that $\pmb { W } ^ { Q } , \pmb { W } ^ { K } , \pmb { W } ^ { V } \in \mathbb { R } ^ { d _ { k } } \times d$ and $W ^ { O } \in \mathbb { R } ^ { d \times d _ { k } }$ . For each position $s ,$ define the query, key, and value vectors as

$$
\begin{array} { r } { \pmb { q } _ { s } : = \pmb { W } ^ { Q } \pmb { w } _ { x _ { s } } ^ { E } , \qquad \pmb { k } _ { s } : = \pmb { W } ^ { K } \pmb { w } _ { x _ { s } } ^ { E } , \qquad \pmb { v } _ { s } : = \pmb { W } ^ { V } \pmb { w } _ { x _ { s } } ^ { E } . } \end{array}\tag{14}
$$

We consider that the model utilizes the last position to output the prediction. Denote that $W ^ { Q K } : = W ^ { K , T } W ^ { Q }$ and $W ^ { O V } : = W ^ { O } W ^ { V }$ , then the attention logit from the last position to position s is $z _ { s } = \frac { ( \pmb { w } _ { x _ { s } } ^ { E } ) ^ { T } \pmb { W } ^ { Q K } \pmb { w } _ { x _ { L } } ^ { E } } { \sqrt { d } }$ . The last-row attention weights are

$$
a _ { s } : = \frac { \exp ( z _ { s } ) } { \sum _ { r = 1 } ^ { L } \exp ( z _ { r } ) } , \qquad s = 1 , \ldots , L .
$$

Then the attention model $f _ { \mathrm { a t t n } }$ is

$$
\begin{array} { r l } & { { \pmb g } _ { \mathrm { a t t n } } ( { \boldsymbol X } ) : = { \pmb W } ^ { O } \displaystyle \sum _ { s = 1 } ^ { L } a _ { s } { \pmb v } _ { s } : = { \pmb W } ^ { O V } \displaystyle \sum _ { s = 1 } ^ { L } a _ { s } { \pmb w } _ { x _ { s } } ^ { E } , } \\ & { { \pmb f } _ { \mathrm { a t t n } } ( { \boldsymbol X } ) = { \pmb W } ^ { U } { \pmb g } _ { \mathrm { a t t n } } ( { \boldsymbol X } ) . } \end{array}\tag{15}
$$

## 5.2 Gradient-flow Analysis

We now analyze how the attention mechanism afects the embedding dynamics. By (1), the embedding gradient can be decomposed into a label-driven term and prediction-induced term:

$$
\frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } = \frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } \bigg | ^ { y } - \frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } \bigg | ^ { p } .
$$

The previous section shows that the label-driven term is the main source through which label signatures enter the embedding dynamics. We therefore focus on $\left. \frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } \right| ^ { y }$ for the attention model. (14) indicates that in $f _ { \mathrm { a t t n } }$ , the gradient with respect to the embedding $\boldsymbol { w _ { \alpha } ^ { E } }$ arises through three components: the value vector, query vector and the key vector. Therefore, we decompose that

$$
\frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } \bigg | ^ { y } = \frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } \bigg | _ { \mathrm { \scriptsize { v a l u e } } } ^ { y } + \frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } \bigg | _ { \mathrm { \scriptsize { q u e r y } } } ^ { y } + \frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } \bigg | _ { \mathrm { \scriptsize { k e y } } } ^ { y } .
$$

As will be seen from the gradient-flow decomposition below, the attention model still expresses the embedding dynamics as a combination of label signatures. However, there are two significant diferences between the $f _ { \mathrm { a t t n } }$ and the $f _ { \mathrm { f f n } }$ model. First, self-attention is position-sensitive. In $f _ { \mathrm { a t t n } }$ , distinct context positions play diferent roles in the computation. The signatures defined above condition on a sampled occurrence of $\alpha ,$ but aggregate the remaining context positions without distinguishing their locations. Here, however, we need to consider the distributions of these tokens at diferent positions. For any positions $s _ { 1 } , s _ { 2 } \in$ [L], we define $\varphi _ { \alpha } ^ { y ; x _ { s _ { 1 } } } \in \mathbb { R } ^ { V \times V }$ and $\varphi _ { \alpha } ^ { y ; x _ { s _ { 1 } } , x _ { s _ { 2 } } } \in \mathbb { R } ^ { V \times V \times V }$ by

$$
\begin{array} { r l } & { \varphi _ { \alpha } ^ { y ; x _ { s _ { 1 } } } ( \nu , \beta _ { s _ { 1 } } ) : = \mathbb { P } ( y = \nu , x _ { s _ { 1 } } = \beta _ { s _ { 1 } } \mid \alpha ) , } \\ & { \varphi _ { \alpha } ^ { y ; x _ { s _ { 1 } } , x _ { s _ { 2 } } } ( \nu , \beta _ { s _ { 1 } } , \beta _ { s _ { 2 } } ) : = \mathbb { P } ( y = \nu , x _ { s _ { 1 } } = \beta _ { s _ { 1 } } , x _ { s _ { 2 } } = \beta _ { s _ { 2 } } \mid \alpha ) , } \end{array}\tag{16}
$$

where $\nu , \beta _ { s _ { 1 } } , \beta _ { s _ { 2 } } \in \mathcal { V }$ . Moreover, the position of α itself should also be taken into account, especially whether it appears at the last position of the sequence. For any positions $s _ { 0 } , s _ { 1 } , s _ { 2 } \in [ L ]$ , we define $\varphi _ { \alpha , x _ { s _ { 0 } } } ^ { y ; x _ { s _ { 1 } } ^ { \star } } \in \mathbb { R } ^ { V \times V }$ and $\varphi _ { \alpha , x _ { s _ { 0 } } } ^ { \hat { y } ; x _ { s _ { 1 } } , x _ { s _ { 2 } } } \in \mathbb { R } ^ { V \times V \times \hat { V } }$ by

$$
\begin{array} { r l } & { \varphi _ { \alpha , x _ { s _ { 0 } } } ^ { y ; x _ { s _ { 1 } } } ( \nu , \beta _ { s _ { 1 } } ) : = \mathbb { P } \left( y = \nu , x _ { s _ { 1 } } = \beta _ { s _ { 1 } } \mid x _ { s _ { 0 } } = \alpha \right) , } \\ & { \varphi _ { \alpha , x _ { s _ { 0 } } } ^ { y ; x _ { s _ { 1 } } , x _ { s _ { 2 } } } ( \nu , \beta _ { s _ { 1 } } , \beta _ { s _ { 2 } } ) : = \mathbb { P } \left( y = \nu , x _ { s _ { 1 } } = \beta _ { s _ { 1 } } , x _ { s _ { 2 } } = \beta _ { s _ { 2 } } \mid x _ { s _ { 0 } } = \alpha \right) , } \end{array}\tag{17}
$$

where $\nu , \beta _ { s _ { 1 } } , \beta _ { s _ { 2 } } \in \mathcal { V }$ . These quantities record the label and contextual distributions associated with α while keeping track of the positions that enter the attention computation.

Second, attention does not transmit these signatures uniformly. Instead, the contribution of each token is multiplied by its attention score. Under the occurrence-conditioned

construction above, $a _ { I }$ is the attention score assigned to the selected occurrence of α. We also define the total attention mass received by token α in X as

$$
A _ { \alpha } ( X ) : = \sum _ { s = 1 } ^ { L } a _ { s } \mathbf { 1 } \{ x _ { s } = \alpha \} .
$$

This quantity measures how much attention is routed through all occurrences of α in the sequence. Correspondingly, for any positions $s _ { 1 } , s _ { 2 } \in [ L - 1 ]$ , we introduce the following attention-weight factors:

$$
\begin{array} { r l } & { \psi _ { \alpha } ^ { y } ( \nu ) : = \mathbb { E } [ a _ { I } \mid y = \nu , \alpha ] , } \\ & { \psi _ { \alpha } ^ { y , x _ { L } } ( \nu , \beta _ { L } ) : = \mathbb { E } [ a _ { I } \mid y = \nu , x _ { L } = \beta _ { L } , \alpha ] , } \\ & { \psi _ { \alpha } ^ { y , x _ { L } , x _ { s _ { 1 } } } ( \nu , \beta _ { L } , \beta _ { s _ { 1 } } ) : = \mathbb { E } [ a _ { I } a _ { s _ { 1 } } \mid y = \nu , x _ { L } = \beta _ { L } , x _ { s _ { 1 } } = \beta _ { s _ { 1 } } , \alpha ] , } \\ & { \psi _ { \alpha , x _ { L } } ^ { y ,  x _ { s _ { 1 } } } ( \nu , \beta _ { s _ { 1 } } ) : = \mathbb { E } [ a _ { s _ { 1 } } \mid y = \nu , x _ { s _ { 1 } } = \beta _ { s _ { 1 } } , x _ { L } = \alpha ] , } \\ & { \psi _ { \alpha , x _ { L } } ^ { y ,  x _ { s _ { 1 } } , x _ { s _ { 2 } } } ( \nu , \beta _ { s _ { 1 } } , \beta _ { s _ { 2 } } ) : = \mathbb { E } [ a _ { s _ { 1 } } a _ { s _ { 2 } } \mid y = \nu , x _ { s _ { 1 } } = \beta _ { s _ { 1 } } , x _ { s _ { 2 } } = \beta _ { s _ { 2 } } , x _ { L } = \alpha ] . } \end{array}\tag{18}
$$

With these definitions, the attention-induced embedding dynamics can be interpreted as encoding position-specific signatures modulated by attention-dependent weights. Formally, we have the following result

Theorem 3. Consider the attention model $f _ { \mathrm { a t t n } }$ under the last-token prediction setting. For any token $\alpha \in \nu$ , the label-driven component of the embedding gradient can be decomposed into value, key, and query contributions:

$$
\frac { d w _ { \alpha } ^ { E } } { d t } \bigg | _ { \mathrm { a t t n } } ^ { y } = \frac { d w _ { \alpha } ^ { E } } { d t } \bigg | _ { \mathrm { v a l u e } } ^ { y } + \frac { d w _ { \alpha } ^ { E } } { d t } \bigg | _ { \mathrm { k e y } } ^ { y } + \frac { d w _ { \alpha } ^ { E } } { d t } \bigg | _ { \mathrm { q u e r y } } ^ { y } .
$$

More explicitly, these three terms are given by

$$
\begin{array} { l } { \displaystyle \frac { d w _ { \alpha } ^ { E } } { d t } \bigg | _ { \mathrm { v a l u e } } ^ { y } = L r _ { \alpha } \left( W ^ { U } W ^ { O V } \right) ^ { T } \left( \varphi _ { \alpha } ^ { y _ { \alpha } ^ { 0 } \odot } \varphi _ { \alpha } ^ { y } \right) , } \\ { \displaystyle \frac { d w _ { \alpha } ^ { E } | ^ { y } } { d t } = \frac { L r _ { \alpha } } { \sqrt { d } } W ^ { Q K } \left[ \widetilde { W } _ { \alpha } ^ { \mathrm { l e v } } \cdot ( \psi _ { \alpha } ^ { y , x _ { L } } \odot \varphi _ { \alpha } ^ { y , x _ { L } } ) - \widetilde { W } ^ { \mathrm { l e v } } \cdot \sum _ { s _ { 1 } = 1 } ^ { L } \psi _ { \alpha } ^ { y , x _ { L } , x _ { s _ { 1 } } } \odot \varphi _ { \alpha } ^ { y , x _ { L } , x _ { s _ { 1 } } } \right] , } \\ { \displaystyle \frac { d w _ { \alpha } ^ { E } } { d t } \bigg | _ { \mathrm { q u e r y } } ^ { y } = \frac { r _ { \alpha } ^ { x _ { L } } } { \sqrt { d } } W ^ { Q K } \left[ \widetilde { W } ^ { \mathrm { q u e r y , 1 } } \cdot \sum _ { s _ { 1 } = 1 } ^ { L } \psi _ { \alpha , x _ { L } } ^ { y , - x _ { s _ { 1 } } } \odot \varphi _ { \alpha , x _ { L } } ^ { y , x _ { s _ { 1 } } } - \widetilde { W } ^ { \mathrm { q u e r y , 2 } } \cdot \sum _ { s _ { 1 } , s _ { 2 } = 1 } ^ { L } \psi _ { \alpha , x _ { L } } ^ { y , - x _ { s _ { 1 } } , x _ { s _ { 2 } } } \odot \varphi _ { \alpha , x _ { L } } ^ { y ; x _ { s _ { 1 } } , x _ { s _ { 2 } } } \right] . } \end{array}\tag{19}
$$

Here ⊙ denotes element-wise multiplication. The coeficient tensors are defined as $f o l l o w s .$ For the key contribution, $\widetilde { W } _ { \alpha } ^ { \mathrm { k e y } } \in \left( \mathbb { R } ^ { d } \right) ^ { V \times V }$ and $\widetilde { W } ^ { \mathrm { k e y } } \in \left( \mathbb { R } ^ { d } \right) ^ { V \times V \times V }$ with element

$$
\begin{array} { r } { \widetilde { W } _ { \alpha } ^ { \mathrm { k e y } } \left( \nu , \beta _ { 1 } \right) = \left( ( \pmb { w } _ { \nu } ^ { U } ) ^ { T } \pmb { W } ^ { O V } \pmb { w } _ { \alpha } ^ { E } \right) \pmb { w } _ { \beta _ { 1 } } ^ { E } , } \\ { \widetilde { W } ^ { \mathrm { k e y } } \left( \nu , \beta _ { 1 } , \beta _ { 2 } \right) = \left( ( \pmb { w } _ { \nu } ^ { U } ) ^ { T } \pmb { W } ^ { O V } \pmb { w } _ { \beta _ { 1 } } ^ { E } \right) \pmb { w } _ { \beta _ { 2 } } ^ { E } , } \end{array}
$$

where $\nu , \beta _ { 1 } , \beta _ { 2 } \ \in \ \mathcal { V }$ Similarly, for the query contribution, Wf <sup>query,1</sup> $\mathbf { \Sigma } \in \mathbf { \Sigma } \left( \mathbb { R } ^ { d } \right) ^ { V \times V }$ and $\widetilde { W } ^ { \mathrm { q u e r y , 2 } } \in \left( \mathbb { R } ^ { d } \right) ^ { V \times V \times V }$ with element

$$
\widetilde { W } ^ { \mathrm { q u e r y } , 1 } \left( \nu , \beta _ { 1 } \right) = \left( ( \pmb { w } _ { \nu } ^ { U } ) ^ { T } \pmb { W } ^ { O V } \pmb { w } _ { \beta _ { 1 } } ^ { E } \right) \pmb { w } _ { \beta _ { 1 } } ^ { E } ,
$$

$$
\widetilde { W } ^ { \mathrm { q u e r y } , 2 } \left( \nu , \beta _ { 1 } , \beta _ { 2 } \right) = \left( ( \pmb { w } _ { \nu } ^ { U } ) ^ { T } \pmb { W } ^ { O V } \pmb { w } _ { \beta _ { 1 } } ^ { E } \right) \pmb { w } _ { \beta _ { 2 } } ^ { E } .
$$

Theorem 3 indicates that the main efect of self-attention is to modulate the signature terms by attention scores. The embedding dynamics are still expressed as a combination of signatures, but each signature is now weighted according to how strongly the corresponding tokens are attended to.

It is worth noting that Theorem 3 keeps the attention-score term in its full form, so that the role of attention in the embedding dynamics remains explicit and interpretable. As a consequence, the resulting expression involves label signatures of at most second order. Higher-order and more complex signatures could be obtained by further expanding the attention-score term, in a manner similar to the derivation of Theorem 1. We do not pursue this direction here, as our main goal is to characterize the direct efect of attention scores on the embedding dynamics.

## 5.3 Empirical Consequences of Attention Modulation

We examine two empirical consequences of the attention decomposition. First, we test whether the shift from early zero-order dominance to later higher-order influence observed in the feed-forward model also appears in an attention-only model. Second, we isolate the role of attention scores in controlling the rate and magnitude of embedding evolution.

To examine the first perspective, we again use the length-2 tasks $F _ { \mathrm { a d d } }$ and $F _ { \mathrm { m o d } }$ . In these tasks, the attention scores assigned to the two number tokens are uniform, so the embedding dynamics are primarily governed by the label signatures. Figure 6A–H shows the embedding structures of the two tasks at the early and late stages of training, revealing a phenomenon similar to that observed in the FFN analysis. Specifically, the embedding space is first reshaped by the zero-order label signatures in the early stage, while the firstorder signature efect becomes clearly visible later in training. These results show that the progressive signature-alignment phenomenon is not specific to the feed-forward model: an attention-only architecture also exhibits early zero-order dominance followed by the emergence of higher-order structure.

To investigate the efect of the attention score, we construct the following tasks.

Task 3. Given anchor set A and noise set R with $\mathcal { A } \cap \mathcal { R } = \emptyset$ , the sequence X is formulated as

$$
X = \left[ r _ { 1 } , \ldots r _ { m - 1 } , a , r _ { m + 1 } \ldots , r _ { L } \right] ,
$$

where $r _ { i } \in \mathcal { R }$ and $a \in { \mathcal { A } }$ . the label $y = 0 \ i f a$ mod $2 = 0$ else 1.

In this task, we set $A = \{ 1 1 , \ldots , 2 0 \}$ and $\mathcal { R } = \{ 2 1 , . . . , 6 0 \}$ At initialization, the attention scores assigned to diferent tokens are approximately uniform, so the embedding dynamics are primarily driven toward the corresponding signature directions. During training, however, the attention scores assigned to the r tokens rapidly decay to nearly zero. Consequently, the associated gradient magnitudes become negligible, and the embeddings of the r tokens exhibit little further evolution. Figure 6 I,J exhibits the structure of the anchor token $a \in { \mathcal { A } }$ and the noise token $r \in \mathcal { R }$ . Anchor token a are divided into two group by the parity, and the noise token are aligned in the one direction, which is consistent with their label signature. The key diference is that the rate of structure formation is controlled by attention. Figure 6 K displays the average received attention score to the anchor tokens and the noise tokens during training, indicating that the anchor tokens a receive substantially larger average attention scores than the noise tokens r. Correspondingly, their embedding geometry changes much more rapidly, as reflected by the faster increase of embedding norm over training in Figure 6 L. In contrast, the noise tokens receive much smaller attention scores, and their embeddings evolve more slowly.

A  
![](images/ba61e6229ae9ec9da666229b348cccfc98e133cad7b51d001dec11f09e63632c.jpg)

B  
![](images/4aee9bc9e9bd2c80a08087bb557438840a1f175cf3437b81bab85fd29f54a39d.jpg)

![](images/ff5b974051bf4f5e9080b1a17dfa930a8a030600e616260a171488ad0f36ba64.jpg)

D  
![](images/3f2781a0ae4b29a3af81be02bcc90726df8d2a45db8ffbcd5da942e0d9b9e04e.jpg)

![](images/cfe560f61234d6ffde0dfe108a9495c177764c137c8081bb6c63a49b29ca47db.jpg)

![](images/8bff5a8c99facff29b15ec39f84a6db85058b6e1df65658b18d9e8e22bf313c7.jpg)

![](images/16c45838d1c082f62e212ccf0548fc4f5f2b949e59e95bf612f7cd507009c24e.jpg)

H  
![](images/8ea6f21b66450fa53feb15aa104d00d0631dfecf231d2598be89ea4fcaeaf984.jpg)

![](images/853a3728c75750317979024a4e39a388ba9a9030bd049a53d575a9f5a004ca45.jpg)

K  
![](images/a7478aff8052549c2ce83e823c5518525b67b42e9681240341a3c9a34d1898f7.jpg)

![](images/0fda4f41ad1521083da584bddac0eb8d9ea760f098cd004edd8b77dd260a1e93.jpg)

L  
![](images/7e2e7bd363c4bc524733248ec72b8d873bdc47cb70524cc62010ea00bd52253d.jpg)  
Figure 6: A–H: Embedding cosine-similarity matrices and PCA projections for the addition and modular-addition tasks at early and late training stages. A,B show earlystage addition; C,D show late-stage addition; E,F show early-stage modular addition; G,H show late-stage modular addition. I,J: Embedding cosine-similarity matrices of anchor tokens and noise tokens in the parity task. K: Mean attention dynamics for anchor tokens and noise tokens. L: Embedding norm dynamics for anchor tokens and noise tokens.

## 6 Path-Specific Signatures in Transformer Training

The controlled analyses above identify two complementary mechanisms: nonlinear feedforward paths expose signatures of diferent orders, while attention paths modulate positionspecific signatures through learned routing weights. We now use these results to formulate path-specific hypotheses for a full decoder-only Transformer and evaluate them on a real language-model training run.

Consider a decoder-only Transformer with $N _ { \mathrm { l a y e r } }$ layers and sequence length L. Let $s \in \{ 1 , \ldots , L - 1 \}$ denote a prediction position, so that the target token is $x _ { s + 1 }$ . We write

the final residual stream at position s as

$$
\pmb { h } _ { s } ^ { N _ { \mathrm { l a y e r } } } = \pmb { h } _ { s } ^ { 0 } + \sum _ { l = 0 } ^ { N _ { \mathrm { l a y e r } } - 1 } \pmb { a } _ { s } ^ { l } + \sum _ { l = 0 } ^ { N _ { \mathrm { l a y e r } } - 1 } \pmb { m } _ { s } ^ { l } ,
$$

where $\pmb { a } _ { s } ^ { l }$ and $m _ { s } ^ { l }$ denote the residual updates produced by the attention module and the FFN module in layer l, respectively. This notation keeps the full layer dependence inside $\pmb { a } _ { s } ^ { l }$ and $m _ { s } ^ { l }$ , and only uses the additive structure of the residual stream.

We focus on the label-dependent part of the gradient flow. Taking the expectation over training sequences and prediction positions, we have

$$
\frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } \bigg | ^ { y } = \mathbb { E } \left[ \left( \frac { \partial { \pmb h } _ { s } ^ { N _ { \mathrm { l a y e r } } } } { \partial { \pmb w } _ { \alpha } ^ { E } } \right) ^ { T } { \pmb W } ^ { U , T } { \pmb e } _ { x _ { s + 1 } } \right] .
$$

Using the residual decomposition above, this gradient can be decomposed into three types of path contributions:

$$
\frac { d w _ { \alpha } ^ { E } } { d t } \bigg | ^ { y } = G _ { \alpha } ^ { \mathrm { d i r e c t } } + \sum _ { l = 0 } ^ { N _ { \mathrm { l a y e r } } - 1 } G _ { \alpha } ^ { \mathrm { a t t n } , l } + \sum _ { l = 0 } ^ { N _ { \mathrm { l a y e r } } - 1 } G _ { \alpha } ^ { \mathrm { f f n } , l } ,
$$

where

$$
G _ { \alpha } ^ { \mathrm { d i r e c t } } : = \mathbb { E } \left[ \left( \frac { \partial h _ { s } ^ { 0 } } { \partial w _ { \alpha } ^ { E } } \right) ^ { T } W ^ { U , T } e _ { x _ { s + 1 } } \right] , \quad G _ { \alpha } ^ { \mathrm { a t t n } , l } : = \mathbb { E } \left[ \left( \frac { \partial a _ { s } ^ { l } } { \partial w _ { \alpha } ^ { E } } \right) ^ { T } W ^ { U , T } e _ { x _ { s + 1 } } \right] ,
$$

and

$$
G _ { \alpha } ^ { \mathrm { f i n } , l } : = \mathbb { E } \left[ \left( \frac { \partial { \bf m } _ { s } ^ { l } } { \partial { \bf w } _ { \alpha } ^ { E } } \right) ^ { T } { \bf W } ^ { U , T } e _ { x _ { s + 1 } } \right] .
$$

Thus, the embedding gradient in a full Transformer is an additive sum of the direct path, attention paths, and FFN paths. This identity provides the bridge between the modular analyses above and the embedding dynamics of a full language model.

Direct path. We first analyze the direct contribution $G _ { \alpha } ^ { \mathrm { d i r e c t } }$ . Since the input residual stream at position s contains the embedding of the current token $x _ { s }$ , we have

$$
\frac { \partial h _ { s } ^ { 0 } } { \partial \pmb { w } _ { \alpha } ^ { E } } = \mathbf { 1 } _ { { x } _ { s } = \alpha } \pmb { I } .
$$

Substituting this identity into the direct gradient contribution gives

$$
G _ { \alpha } ^ { \mathrm { d i r e c t } } = \mathbb { E } \left[ \mathbf { 1 } _ { { \boldsymbol { x } } _ { s } = \alpha } W ^ { U , T } e _ { x _ { s + 1 } } \right] .
$$

Let $r _ { \alpha } ^ { \mathrm { n t } } = \mathbb { P } ( x _ { s } = \alpha )$ be the probability that token α appears at the prediction position, and define the next-token signature of α as $\varphi _ { \alpha } ^ { y ; \mathrm { n t } } \in \mathbb { R } ^ { V }$ whose ν-th element is

$$
\varphi _ { \alpha } ^ { y ; \mathrm { n t } } ( \nu ) = \mathbb { P } ( x _ { s + 1 } = \nu \mid x _ { s } = \alpha ) .
$$

Then the direct contribution can be rewritten as

$$
G _ { \alpha } ^ { \mathrm { d i r e c t } } = { r } _ { \alpha } ^ { \mathrm { n t } } W ^ { U , T } \varphi _ { \alpha } ^ { y ; \mathrm { n t } } .
$$

Therefore, the direct embedding-readout path updates $\boldsymbol { w _ { \alpha } ^ { E } }$ according to the distribution of tokens that follow α. In other words, before considering the indirect efects of attention and FFN modules, the most immediate statistical signal received by the embedding of $\alpha$ is its next-token signature.

FFN path. The FFN module has already been analyzed in the previous sections. There, we showed that a position-wise nonlinear module can introduce signature information into the embedding dynamics, and that the order of the induced signature depends on the efective input structure and the activation order. In the full Transformer, however, the FFN module receives the residual stream at a single position. Therefore, at the beginning of training, the input to the FFN module at position s is dominated by the residual direct term:

$$
\begin{array} { r } { h _ { s } ^ { \ell } \approx h _ { s } ^ { 0 } = w _ { x _ { s } } ^ { E } . } \end{array}
$$

Thus, from the perspective of the FFN module, the efective input sequence has length one. Consequently, in the early stage, the FFN path only induces zero-order label signatures which is the next-token signature under the next-token prediction.

As training proceeds, the attention outputs and the residual stream become comparable in scale. For example, after the first attention module, the input to the following FFN at position s can be written schematically as

$$
\pmb { h } _ { s } ^ { 0 } + \pmb { a } _ { s } ^ { 0 } \approx \pmb { w } _ { x _ { s } } ^ { E } + \pmb { W } ^ { O V , 0 } \sum _ { j \le s } A _ { s , j } ^ { 0 } \pmb { w } _ { x _ { j } } ^ { E } .
$$

Once $\left\| a _ { s } ^ { 0 } \right\|$ becomes comparable to $\left\| h _ { s } ^ { 0 } \right\|$ , the FFN input becomes a weighted combination of the embeddings of the previous tokens $x _ { 1 } , \ldots , x _ { s }$ . Therefore, by the FFN analysis in the previous section, the FFN path can then introduce higher-order signature information into the embedding gradient. These analysis indicate that $G _ { \alpha } ^ { \mathrm { f f n } , l }$ could be formulated as a combination of signatures with diferent orders.

Attention path. The efect of the attention path follows directly from the analysis in the previous section. In particular, Theorem 3 shows that, at the early stage of training, the leading structure introduced by the attention path is an attention-weighted zero-order label signature. For a token $\alpha ,$ let $I _ { s } \sim \mathrm { U n i f } ( [ s ] )$ be sampled independently of the training sequence, and condition on $x _ { I _ { s } } = \alpha$ . In this prefix setting, we use $\mathbb { P } ( \cdot \mid \alpha )$ as shorthand for $\mathbb { P } ( \cdot \mid x _ { I _ { s } } = \alpha )$ . The total attention mass at prediction position s is

$$
A _ { \alpha } ( X _ { : s } ) = \sum _ { j \leq s } a _ { s , j } { \bf 1 } \{ x _ { j } = \alpha \} .
$$

Then the corresponding signature takes the form $\varphi _ { \alpha } ^ { y ; 0 } \odot \psi _ { \alpha } ^ { y }$ , where $\varphi _ { \alpha } ^ { y ; 0 } ( \nu ) = \mathbb { P } ( x _ { s + 1 } = \nu \mid \alpha )$ and $\psi _ { \alpha } ^ { y } ( \nu ) = \mathbb { E } [ a _ { s , I _ { s } } \mid x _ { s + 1 } = \nu , \alpha ]$ . Therefore, the attention path contains the same zeroorder token–label statistics as before, but reweights them according to how much attention is assigned to occurrences of $\alpha$ . Tokens receiving larger attention mass consequently exert a stronger influence on the corresponding embedding update.

As training proceeds, the key- and query-dependent terms derived in Theorem 3 introduce additional position-specific and higher-order signatures. We do not repeat their derivation here and focus below on the early-stage attention-weighted zero-order signature.

## 6.1 Experimental Validation

We next investigate how the statistical structures encoded by token embeddings evolve during real language-model training. Our theoretical analysis predicts that the embedding geometry is initially dominated by low-order signatures, while richer higher-order signatures gradually become encoded as training proceeds. We first verify this progression from the next-token signature to the first-order signature. We then provide an additional verification of the attention-path prediction derived from the path analysis.

We train a 0.7B-parameter decoder-only Transformer on 36B tokens. Let $\tau$ denote the set of the 200 most frequent eligible tokens in the training corpus. For each token $\alpha \in { \mathcal { T } }$ we first estimate its empirical next-token signature $\varphi _ { \alpha } ^ { y ; \mathrm { n t } }$ , and construct the corresponding similarity matrix

$$
S _ { \alpha , \beta } ^ { \mathrm { n e x t } } = \frac { \left. \varphi _ { \alpha } ^ { y ; \mathrm { n t } } , \varphi _ { \beta } ^ { y ; \mathrm { n t } } \right. } { \left. \varphi _ { \alpha } ^ { y ; \mathrm { n t } } \right. _ { 2 } \left. \varphi _ { \beta } ^ { y ; \mathrm { n t } } \right. _ { 2 } } , \qquad \alpha , \beta \in \mathcal { T } .
$$

For each epoch t, we compute the embedding similarity matrix

$$
S _ { \alpha , \beta } ^ { \mathrm { e m b } , ( t ) } = \frac { \left. { \pmb w } _ { \alpha } ^ { E , ( t ) } , { \pmb w } _ { \beta } ^ { E , ( t ) } \right. } { \left| { \pmb w } _ { \alpha } ^ { E , ( t ) } \right| _ { 2 } \left| { \pmb w } _ { \beta } ^ { E , ( t ) } \right| _ { 2 } } , \qquad \alpha , \beta \in { \cal T } ,
$$

and measure the agreement between the embedding geometry and the next-token signature geometry by

$$
\rho _ { \mathrm { n e x t } } ^ { ( t ) } = \mathrm { C o r r } \left( \left\{ S _ { \alpha , \beta } ^ { \mathrm { e m b } , ( t ) } : \alpha < \beta \right\} , \left\{ S _ { \alpha , \beta } ^ { \mathrm { n e x t } } : \alpha < \beta \right\} \right) .
$$

To examine whether the embedding geometry progressively captures richer statistical structures, we further estimate the first-order signature. For an occurrence of α at position s, we record whether another token $\beta$ has appeared anywhere in the prefix $X _ { < s }$ , regardless of its position or multiplicity. The resulting first-order signature is

$$
\varphi _ { \alpha } ^ { y ; 1 } ( \nu , \beta ) = \mathbb { P } \left( x _ { s + 1 } = \nu , \beta \in X _ { < s } \mid x _ { s } = \alpha \right) , \qquad \nu , \beta \in \mathcal { V } .\tag{20}
$$

Since $\varphi _ { \alpha } ^ { y ; 1 }$ contains an additional context-token dimension, it cannot be compared directly with the embedding geometry. We learn a shared projection along the context-token dimension together with a linear readout. Specifically, at epoch t, the probe produces an embedding-like representation

$$
\begin{array} { r } { \widehat { \pmb w } _ { \alpha } ^ { E , ( t ) } = \varphi _ { \alpha } ^ { y ; 1 } \mathbf { v } ^ { ( t ) } , } \end{array}\tag{21}
$$

where $\mathbf { v } ^ { ( t ) }$ is a learned probe over the context-token dimension. We randomly split the anchor tokens into a fixed training set and a held-out test set. At each epoch, the probe

parameters are learned using only the training tokens, while evaluation is performed exclusively on similarities between held-out tokens. Let

$$
\widehat { S } _ { \alpha , \beta } ^ { ( 1 ) , ( t ) } = \frac { \left. \widehat { \pmb { w } } _ { \alpha } ^ { E , ( t ) } , \widehat { \pmb { w } } _ { \beta } ^ { E , ( t ) } \right. } { \left| \widehat { \pmb { w } } _ { \alpha } ^ { E , ( t ) } \right| _ { 2 } \left| \widehat { \pmb { w } } _ { \beta } ^ { E , ( t ) } \right| _ { 2 } }
$$

denote the similarity predicted by the first-order signature probe. The probe is trained directly in the similarity space by minimizing the MSE between $\widehat { S } _ { \alpha , \beta } ^ { ( 1 ) , ( t ) }$ and $S _ { \alpha , \beta } ^ { \mathrm { e m b } , ( t ) }$

Figures 7A and B verify the transition from low-order to higher-order signatures. Figure 7A shows that $\rho _ { \mathrm { n e x t } } ^ { ( t ) }$ rises rapidly from nearly zero, reaches a pronounced peak at the beginning of training, and then gradually decreases to a stable nonzero level. This indicates that the embedding geometry is initially dominated by the next-token signature, while additional statistical structures emerge during subsequent training. Figure 7B reports the held-out similarity MSE of the first-order signature probe. Although the embedding geometry changes rapidly during the earliest stage of optimization, the probe error quickly decreases and remains consistently low throughout the remainder of training. Since the probe is trained only on one subset of anchor tokens and evaluated exclusively on unseen tokens, this result shows that a shared rank-one projection of the first-order signature generalizes across tokens and captures a substantial fraction of the learned embedding geometry.

As an additional verification of the path analysis, we also examine the attention path. For layer ℓ and epoch t, let $\bar { A } _ { s , j } ^ { \ell , ( t ) }$ denote the attention weight from position s to position $j ,$ averaged over attention heads. Let $I _ { s } \sim \mathrm { U n i f } ( [ s ] )$ be independent of the training sequence, and use the same occurrence-conditioned shorthand $\mathbb { E } [ \cdot \mid \alpha ] = \mathbb { E } [ \cdot \mid x _ { I _ { s } } = \alpha ]$ . The attentionpath signature associated with token α is estimated as

$$
\varphi _ { \alpha } ^ { y ; \mathrm { a t t n } , \ell , ( t ) } ( \nu ) = \mathbb { E } \left[ \bar { A } _ { s , I _ { s } } ^ { \ell , ( t ) } \mathbf { 1 } _ { \{ x _ { s + 1 } = \nu \} } \mid \alpha \right] , \qquad \nu \in \mathcal { V } .\tag{22}
$$

We then construct

$$
S _ { \alpha , \beta } ^ { \mathrm { a t t n - s i g } , \ell , ( t ) } = \frac { \left. \varphi _ { \alpha } ^ { y ; \mathrm { a t t n } , \ell , ( t ) } , \varphi _ { \beta } ^ { y ; \mathrm { a t t n } , \ell , ( t ) } \right. } { \left. \varphi _ { \alpha } ^ { y ; \mathrm { a t t n } , \ell , ( t ) } \right. _ { 2 } \left. \varphi _ { \beta } ^ { y ; \mathrm { a t t n } , \ell , ( t ) } \right. _ { 2 } } , \qquad \alpha , \beta \in \mathcal { T } .
$$

For each layer ℓ and epoch t, we define

$$
z _ { \alpha } ^ { \ell , ( t ) } = W ^ { O , \ell , ( t ) } W ^ { V , \ell , ( t ) } w _ { \alpha } ^ { E , ( t ) } ,
$$

construct

$$
S _ { \alpha , \beta } ^ { \mathrm { a t t n - l o g i t s } , \ell , ( t ) } = \frac { \Bigl \langle z _ { \alpha } ^ { \ell , ( t ) } , z _ { \beta } ^ { \ell , ( t ) } \Bigr \rangle } { \Bigl \vert z _ { \alpha } ^ { \ell , ( t ) } \Bigr \vert _ { 2 } \Bigl \vert z _ { \beta } ^ { \ell , ( t ) } \Bigr \vert _ { 2 } } ,
$$

and compute

$$
\rho _ { \mathrm { a t t n } } ^ { \ell , ( t ) } = \mathrm { C o r r } \left( \left\{ S _ { \alpha , \beta } ^ { \mathrm { a t t n - l o g i t s } , \ell , ( t ) } : \alpha < \beta \right\} , \left\{ S _ { \alpha , \beta } ^ { \mathrm { a t t n - s i g } , \ell , ( t ) } : \alpha < \beta \right\} \right) .
$$

Figure 7C shows that the similarity structure induced by the attention-path representations exhibits a clear early-stage correlation with the corresponding attention-weighted signature, providing empirical support for the path-specific prediction derived in the theoretical analysis.

![](images/3d750a95f697317d612a3b541c1286778b2046e46076907da271726e35481d18.jpg)

![](images/b3d96140668215686d22bcc3a7da5ab23af9700f33b4f393da93e381084fb36b.jpg)

![](images/5589e5652615f53e5f593c556583193786a57e7b1395b8950ee62cd33ae3de79.jpg)  
Figure 7: A: Correlation between the embedding similarity matrix $S ^ { \mathrm { e m b , ( } t ) }$ and the nexttoken signature similarity matrix $S ^ { \mathrm { n e x t } }$ over training steps, computed on the 200 most frequent tokens. B: held-out similarity MSE between the pairwise cosine similarities predicted by the rank-one first-order-signature probe and those of the token embeddings across training epochs; the broken horizontal axis separates the early-training epochs from the later epochs. C: Layer-wise heatmap of the correlation between the attention-path similarity matrix $S ^ { \mathrm { a t t n - l o g i t s } , \ell , ( t ) }$ and the attention-weighted signature similarity matrix $S ^ { \mathrm { a t t n - s i g } , \ell , ( t ) }$ over training steps.

## 7 The Role of Signatures in Model Learning

The preceding sections establish that learned embeddings can align with identifiable statistical signatures and that these signatures appear explicitly in the embedding updates. We now ask a separate question: does such a structure actually help the model learn the task?

## 7.1 Relation between Signature and Learning

The empirical results above show that the embedding space can reflect a hierarchy of signatures, while the gradient-flow analysis identifies these signatures as components of the embedding update. A natural question is whether learning such signature structures actually facilitates the learning of the task itself. In particular, does the emergence of a mathematically meaningful embedding geometry necessarily imply that the model has learned the corresponding task?

We investigate this question from three complementary perspectives. We first show that encoding a signature structure is not by itself suficient for solving a task. We then test whether directly providing task-relevant signature information can accelerate learning. Finally, we show that the order at which discriminative information first appears in the signature hierarchy can strongly afect when the task is learned.

Signature structure alone is not suficient for task learning. We first illustrate this point using the binary modular-addition task $F _ { \mathrm { m o d } }$ (see Task 2). Previous studies on modular addition often associate the emergence of structured token representations with successful task learning (Liu et al., 2022; Nanda et al., 2023; Gromov, 2023; Mallinar et al., 2025). Consistent with this picture, in the FFN model, once the embeddings become aligned with the first-order signatures and form the corresponding modular structure, the model rapidly learns the task (Figure 8 A, B).

The attention-only model provides a contrasting example (Figure 8 A, C). Its learnable embeddings also become dominated by first-order signature information and exhibit a clear mathematical structure, yet the model fails to solve the task. Thus, a meaningful signaturealigned embedding geometry is not suficient by itself: the downstream architecture must also be able to transform this representation into the target operation. In this setting, the attention-only model is not expressive enough to efectively exploit the learned signature structure.

A  
![](images/ae4fa0928b439fdae7bb2e3770bf400351676110fe6f93f0c83b4d3ea5f6a76f.jpg)

B  
![](images/b79b5c06641f5da30bbb50c488783b569801bb0b6ee17e425dcb0949ef15febe.jpg)

C  
![](images/ddcb418c11fbdc2dd2ed9674ab01a9cf95085bd16f5b293d38996425a6d0ce12.jpg)

D  
![](images/677e000d6b92000e0bf5778a3f309454f0d3b003a398c78131e409899ea12485.jpg)  
Figure 8: A: Training accuracy of the FFN model and the attention-only model. B: PCA projection of the final FFN token embeddings. C: PCA projection of the final attention-only token embeddings. D: Training accuracy of the FFN model with fixed first-order signature embeddings, trainable embeddings, and fixed random embeddings.

Task-relevant signatures can accelerate learning. We next test whether the signature itself contains information that can directly facilitate optimization. We train the FFN model with a fixed first-order signature embedding, where each token is represented by a random linear transformation of $\varphi _ { a } ^ { y ; 1 }$ (see Appendix for details). The resulting model converges substantially faster than the standard trainable-embedding baseline (Figure 8 D). This confirms that first-order signatures contain task-relevant information that can directly facilitate learning.

Interestingly, fixed random embeddings also outperform trainable embeddings at the early stage of this modular-addition task. In this setting, all number tokens have the same zeroth-order label signature. The early embedding updates therefore contain little information for distinguishing diferent tokens, whereas fixed random embeddings are approximately pairwise orthogonal and preserve token distinctions from the beginning. This suggests that not only the presence of signature information, but also the order at which discriminative information appears in the signature hierarchy, can afect the learning dynamics.

Signature order controls the onset of task learning. We test this prediction more systematically by considering binary modular addition $F _ { \mathrm { m o d } }$

$$
y = \left( x _ { 1 } + x _ { 2 } \right) { \bmod { M } } , \qquad x _ { 1 } , x _ { 2 } \in \{ 1 , \ldots , N \} ,
$$

and varying the modulus M while keeping N fixed.

The $\varphi _ { \alpha } ^ { y ; 0 }$ in this task have a simple dependence on M and N. When M divides $N ,$ all tokens have identical zeroth-order label signatures. Their pairwise signature similarities are therefore all equal to one. In this case, the $\varphi _ { \alpha } ^ { y ; 0 }$ provides no information for distinguishing each token required by the task, and the model must rely on higher-order signature information to learn the modular rule.

When M does not divide N, the residue classes contain unequal numbers of tokens. As a result, the zeroth-order signatures are no longer identical across all tokens and already provide discriminative information at the beginning of training. Their similarity matrices consequently exhibit a regular residue-dependent structure rather than being uniformly one.

We verify this prediction with $N = 2 0$ and vary M across a range of values. Figure 9 reports the first epoch at which the training accuracy reaches 95%, together with the pairwise similarity matrices of the $\varphi _ { \alpha } ^ { y ; 0 }$ . A striking slowdown occurs precisely when M divides N: for $M = 4 , 5 , 1 0$ , and 20, the similarity matrices are uniformly one and the model requires thousands of epochs to reach high accuracy. For the other tested values of M, the $\varphi _ { \alpha } ^ { y ; 0 }$ already distinguish diferent residue classes, and the model typically learns the task within only a few hundred epochs.

![](images/3a98013018233d4fdaaafa960714b9c6d8ba85181fcfee7ab75dcd75901c9da3.jpg)  
Figure 9: Learning speed of $F _ { \mathrm { m o d } }$ under diferent moduli M, with $N = 2 0$ . The vertical axis shows the first epoch at which training accuracy reaches 95%; gray points denote runs where $M \vert N$ . The accompanying heatmaps show the pairwise similarities between zeroth-order signatures.

These results establish a direct connection between the signature hierarchy and task learning. Signatures can provide useful representations for solving a task, but whether and when this information becomes useful depends both on the architecture that processes the embeddings and on the order at which task-relevant distinctions emerge in the signature hierarchy.

## 7.2 An Example of Task Solving via Signatures

We then use a simple order-selection task to illustrate how signatures can support task solving when the model is able to learn the task successfully. Let the label space be $\mathcal { V } =$ $\{ 1 , \ldots , N \}$ . Each input sequence has the form

$$
S = ( c , x _ { 1 } , x _ { 2 } ) , \qquad c \in \{ a , b \} , \quad x _ { 1 } , x _ { 2 } \in \mathcal { V } ,
$$

where $x _ { 1 }$ and $x _ { 2 }$ are numerical tokens and c is an anchor token that determines the selection rule. We take c uniformly from $\{ a , b \}$ and $x _ { 1 } , x _ { 2 }$ independently and uniformly from $\mathcal { V } .$ The target is defined as

$$
Y = f ( S ) = { \left\{ \begin{array} { l l } { \operatorname* { m a x } ( x _ { 1 } , x _ { 2 } ) , } & { c = a , } \\ { \operatorname* { m i n } ( x _ { 1 } , x _ { 2 } ) , } & { c = b . } \end{array} \right. }
$$

Thus, the task is not simply to compare two numbers. Instead, the model must select one of the two order statistics according to the contextual anchor token $c .$ This task is motivated by the familiar ambiguity in comparing numbers such as 3.9 and 3.11. When they are interpreted as decimal numbers, $3 . 9 > 3 . 1 1$ . However, when they are interpreted as version numbers, $3 . 1 1 > 3 . 9$ (Xie, 2024; Yang et al., 2024a; Chen et al., 2025).

We now show that the superposition of zero-order signatures is suficient to solve the task. For a sequence $( c , x _ { 1 } , x _ { 2 } )$ , we define the signature-induced score for label $\nu$ as

$$
\Phi _ { c , x _ { 1 } , x _ { 2 } } ( y ) = \varphi _ { c } ^ { y ; 0 } ( y ) + \varphi _ { x _ { 1 } } ^ { y ; 0 } ( y ) + \varphi _ { x _ { 2 } } ^ { y ; 0 } ( y ) .
$$

The prediction induced by zero-order signature superposition is

$$
\hat { y } = \arg \operatorname* { m a x } _ { y \in \mathcal { Y } } \Phi _ { c , x _ { 1 } , x _ { 2 } } ( y ) .
$$

We first consider the contribution of the two numerical tokens. Fix a numerical token x and denote the other numerical token by $z ,$ which is uniformly sampled from $\mathcal { V } .$ . There are three cases. $\operatorname { I f } z < x ,$ , the output is $x$ when the anchor is $^ { a , }$ which occurs with probability $1 / 2$ . If $z > x$ , the output is $x$ when the anchor is $b ,$ again with probability $1 / 2$ . Thus, whenever $z \neq x ,$ , one of the two equally likely anchor choices outputs $x ,$ while the other outputs z. If $z = x$ , both the maximum and minimum are equal to $x ,$ so the output is always x.

Equivalently, we can decompose the label distribution into two parts. One half of the probability mass is always assigned to $x ,$ corresponding to the anchor choice that selects $x$ when $z \neq x .$ . The other half follows the value of $z .$ Since z is uniformly distributed over $\mathcal { V }$ this second part contributes $1 / 2 N$ to every label, including $y = x$ . Therefore,

$$
\varphi _ { x } ^ { y ; 0 } = \frac { 1 } { 2 } { \bf 1 } _ { y = x } + \frac { 1 } { 2 N } .
$$

Therefore, the sum of the two numerical signatures is

$$
\varphi _ { x _ { 1 } } ^ { y ; 0 } + \varphi _ { x _ { 2 } } ^ { y ; 0 } = \frac { 1 } { N } + \frac { 1 } { 2 } { \bf 1 } _ { y = x _ { 1 } } + \frac { 1 } { 2 } { \bf 1 } _ { y = x _ { 2 } } .
$$

Thus, the two numerical tokens create two equal candidate peaks at $y = x _ { 1 }$ and $y = x _ { 2 }$ 2 while all non-candidate labels only receive the constant background value $1 / N$ . Since the numerical signatures contribute an additional peak of height $1 / 2$ at $x _ { 1 }$ and $x _ { 2 } .$ , whereas the anchor signatures vary only at scale $O ( 1 / N )$ , the maximizer of $\Phi _ { c , x _ { 1 } , x _ { 2 } } ( y )$ lies among the two candidate labels.

Assume without loss of generality that $x _ { 1 } < x _ { 2 }$ . When $c = a .$ , the anchor signature is

$$
\varphi _ { a } ^ { y ; 0 } ( y ) = \frac { 2 y - 1 } { N ^ { 2 } } ,
$$

which is monotonically increasing in y. The total signature score becomes

$$
\Phi _ { a , x _ { 1 } , x _ { 2 } } ( y ) = \frac { 2 y - 1 } { N ^ { 2 } } + \frac 1 N + \frac 1 2 { \bf 1 } _ { y = x _ { 1 } } + \frac 1 2 { \bf 1 } _ { y = x _ { 2 } } .
$$

The two candidate labels have scores

$$
\Phi _ { a , x _ { 1 } , x _ { 2 } } ( x _ { 1 } ) = \frac { 2 x _ { 1 } - 1 } { N ^ { 2 } } + \frac { 1 } { N } + \frac { 1 } { 2 } , \qquad \Phi _ { a , x _ { 1 } , x _ { 2 } } ( x _ { 2 } ) = \frac { 2 x _ { 2 } - 1 } { N ^ { 2 } } + \frac { 1 } { N } + \frac { 1 } { 2 } .
$$

Since $x _ { 2 } > x _ { 1 }$ , we have

$$
\Phi _ { a , x _ { 1 } , x _ { 2 } } ( x _ { 2 } ) > \Phi _ { a , x _ { 1 } , x _ { 2 } } ( x _ { 1 } ) .
$$

Hence the maximum is attained at the larger candidate:

$$
\arg \operatorname* { m a x } _ { y \in \mathcal { V } } \Phi _ { a , x _ { 1 } , x _ { 2 } } ( y ) = x _ { 2 } = \operatorname* { m a x } ( x _ { 1 } , x _ { 2 } ) .
$$

When $c = b .$ , the anchor signature is

$$
\varphi _ { b } ^ { y ; 0 } ( y ) = \frac { 2 ( N - y ) + 1 } { N ^ { 2 } } ,
$$

which is monotonically decreasing in y. The total signature score becomes

$$
\Phi _ { b , x _ { 1 } , x _ { 2 } } ( y ) = \frac { 2 ( N - y ) + 1 } { N ^ { 2 } } + \frac { 1 } { N } + \frac { 1 } { 2 } { \bf 1 } _ { y = x _ { 1 } } + \frac { 1 } { 2 } { \bf 1 } _ { y = x _ { 2 } } .
$$

The two candidate labels have scores

$$
\Phi _ { b , x _ { 1 } , x _ { 2 } } ( x _ { 1 } ) = \frac { 2 ( N - x _ { 1 } ) + 1 } { N ^ { 2 } } + \frac { 1 } { N } + \frac { 1 } { 2 } , \qquad \Phi _ { b , x _ { 1 } , x _ { 2 } } ( x _ { 2 } ) = \frac { 2 ( N - x _ { 2 } ) + 1 } { N ^ { 2 } } + \frac { 1 } { N } + \frac { 1 } { 2 } .
$$

Since $x _ { 1 } < x _ { 2 }$ , we have

$$
\Phi _ { b , x _ { 1 } , x _ { 2 } } ( x _ { 1 } ) > \Phi _ { b , x _ { 1 } , x _ { 2 } } ( x _ { 2 } ) .
$$

Hence the maximum is attained at the smaller candidate:

$$
\arg \operatorname* { m a x } _ { y \in \mathcal { V } } \Phi _ { b , x _ { 1 } , x _ { 2 } } ( y ) = x _ { 1 } = \operatorname* { m i n } ( x _ { 1 } , x _ { 2 } ) .
$$

Therefore, the min–max task can be solved by the additive superposition of zero-order signatures. This construction does not claim that every trained model implements exactly this score; rather, it demonstrates that the statistical information contained in the signatures is suficient to support the required prediction rule when the architecture can access it.

## 7.3 Signatures Encode Semantic Structure

We further find that signatures themselves already contain interpretable semantic structures, suggesting that they provide an important statistical basis from which embeddings acquire semantic organization. Previous studies have extensively documented that learned word and language-model representations exhibit rich semantic geometry, including linear relational structures, semantic clustering, and low-dimensional organization of real-world concepts (Mikolov et al., 2013c; Pennington et al., 2014; Yaghoobzadeh et al., 2019; Gurnee and Tegmark, 2024). Most of these studies, however, focus on characterizing or probing the resulting representation geometry. Here, we instead ask whether such structures can already be traced back to identifiable statistics of the training data.

![](images/901228f6081388843cfdd704202b7b5fe608dd2a44afb56533f8baa6e910d587.jpg)  
Figure 10: Mechanism of zero-order signature superposition in the min–max task.

To this end, we compute the next-token signatures φ<sub>α</sub> y;nt directly from the same 36Btoken corpus used to train our language model and visualize the signatures of several groups of tokens using PCA. As shown in Fig. 11 A, the signatures of numerical tokens form a continuous trajectory that approximately follows their numerical order. Figure 11 B presents the signatures of the seven weekdays, which exhibit a cyclic organization consistent with their periodic semantic relationship. Beyond ordered structures, signatures also encode categorical information. In Fig. 11 C, we jointly project the signatures of tokens from several semantic categories, including colors, family roles, countries, and programming languages. Tokens belonging to the same semantic category are naturally grouped together, while diferent categories occupy distinct regions of the projected space.

These results show that semantic geometries commonly observed in learned representations are already present in the token-conditioned statistics of the training corpus. Combined with our dynamical analysis showing how these signatures progressively enter the embedding updates, this provides a possible statistical origin for the emergence of semantic organization in the learned embedding space.

## 8 Conclusion

In this work, we identified a progressive signature-alignment phenomenon in token-embedding dynamics. Across controlled sequence-prediction tasks, early embedding geometries are well explained by simple token-conditioned label distributions. Later in training, the efects of multiple higher-order contextual signatures become jointly visible rather than following a strict one-at-a-time replacement, with the highest accessible order providing the strongest contribution under the late-stage conditions of our analysis. The same evolution admits a spectral interpretation: low-order structure corresponds to zero-frequency projections of higher-order signatures, while task-specific nonzero-frequency components emerge later in training. This perspective also accounts for the distinct geometries obtained from diferent random seeds as diferent spectral selections within a shared signature hierarchy.

![](images/129d352a4e86e95b04e9d7df5d787b06e8862e46d08cb69615de0e95f1dab718.jpg)

![](images/532d43c540791e8a35aff7d5a0d87e0b3fb614e2e041e94544151041fbad8c18.jpg)

![](images/0106b996326288e952f35dcf51de5312875d09ff36ef32efce214c6af0ef52bb.jpg)  
Figure 11: A: PCA projection of the signatures of numerical tokens, connected according to their numerical order. B: PCA projection of the signatures of weekdays, connected according to their cyclic order. C: Joint PCA projection of the signatures of tokens from diferent semantic categories, including colors, family roles, countries, and programming languages.

We used gradient-flow analysis to provide a mechanistic explanation for these observations. In feed-forward models, signatures of diferent orders appear explicitly in the embedding gradient, and small initialization suppresses higher-order contributions at the beginning of training. In self-attention models, position-specific signatures are modulated by attention scores, so learned routing determines which token statistics have the strongest efect on the embeddings. These results explain how the data distribution and computational architecture jointly determine the statistical components available to shape embedding updates, without requiring the learned geometry to be treated as a purely post-hoc object.

The controlled analyses further motivate path-specific hypotheses in full Transformers. In a real language-model training run, we find that early embedding geometry tracks next-token signatures associated with the direct and FFN paths, while attention-path representations track attention-weighted signatures across layers. The correlations subsequently decay as additional structures enter the model, consistent with the progressive nature of the signature-alignment picture.

Finally, we showed that learning a meaningful signature-aligned geometry is not by itself suficient for solving a task. Whether the encoded structure is useful depends on the computations available after the embedding lookup. When the architecture can exploit the relevant signature information, signature-based embeddings can directly support prediction and accelerate optimization; when it cannot, a structured embedding space may emerge without successful task learning.

Overall, our results connect an empirical phenomenon in embedding geometry with an interpretable gradient-flow mechanism. The signature perspective provides a unified language for describing what statistical structures appear in token embeddings, how those structures change during training, how architectural paths modulate them, and when they become useful for task solving.

## Acknowledgments and Disclosure of Funding

This work is sponsored by the National Key R&D Program of China Grant No. 2022YFA100 8200 (Z. X.), the National Natural Science Foundation of China Grant No. 92570001 (Z. X.), 12371511 (Z. X.), 12422119 (Z. X.), 2025 Key Technology R&D Program “New Generation Information Technology” Project 25511103100 of Shanghai Municipal Science and Technology Commission (Z. X.).

## Appendix A. Experimental Configurations

This section provides the complete experimental configurations for the controlled FFN experiments, the attention-only experiments, the signature-based embedding experiments, and the real language-model training experiment. Unless otherwise specified, all reported results are obtained using the common settings described below.

## A.1 FFN Experiments

## A.1.1 FFN Architecture

For the controlled FFN experiments, we set $d = 1 2 8$ unless otherwise specified. The model contains no hidden bias and no output bias. No normalization layer, residual connection, dropout, or positional embedding is used.

For Figures 1–3 and Figure 7, the activation function is $\sigma ( z ) ~ = ~ \mathrm { R e L U } ( z )$ , applied coordinate-wise. For the activation-order experiment in Figure 4, we instead use the monomial activation $\sigma ( z ) = z ^ { \odot K }$

## A.1.2 Prescribed Label-Distribution Experiment

This experiment corresponds to Task 1 and Figure 1A. We use one anchor token $a = 1 0 0$ and the context-token set $\mathcal { X } = \{ 0 , 1 , \ldots , 5 0 \}$ . Each input sequence has length three and takes the form

$$
X = ( a , x _ { 1 } , x _ { 2 } ) , \qquad x _ { 1 } , x _ { 2 } \in \mathcal { X } .\tag{23}
$$

The complete input set is

$$
\mathcal { D } _ { a } = \left\{ \left( a , x _ { 1 } , x _ { 2 } \right) : x _ { 1 } , x _ { 2 } \in \mathcal { X } \right\} ,\tag{24}
$$

which contains $5 1 ^ { 2 } = 2 6 0 1$ distinct input sequences.

For every input $X \in { \mathcal { D } } _ { a } .$ , its target label is sampled from a prescribed distribution $p _ { a } \colon$

$$
y ( X ) \sim p _ { a } .\tag{25}
$$

The sampled labels are sampled once before training and kept fixed. The label vocabulary is $\mathcal { V } = \mathcal { X }$

We consider the following four prescribed distributions.

Uniform distribution. The uniform distribution is defined by

$$
p _ { a } ( y ) = \frac { 1 } { | y | } , \qquad y \in \mathcal { V } .\tag{26}
$$

Normal distribution. The unnormalized probability mass is defined as

$$
\widetilde { p } _ { a } ( y ) = \exp \left( - \frac { ( y - \mu ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) ,\tag{27}
$$

where $\mu = 2 5 , \sigma = 1$ . The discrete distribution is obtained by normalization:

$$
p _ { a } ( y ) = \frac { \widetilde { p } _ { a } ( y ) } { \sum _ { \nu \in y } \widetilde { p } _ { a } ( \nu ) } .\tag{28}
$$

Poisson distribution. The unnormalized probability mass is

$$
\widetilde { p } _ { a } ( y ) = \frac { \lambda ^ { y } e ^ { - \lambda } } { y ! } ,\tag{29}
$$

where $\lambda = 1 0$ . The distribution is truncated to Y and renormalized:

$$
p _ { a } ( y ) = \frac { \widetilde { p } _ { a } ( y ) } { \sum _ { \nu \in y } \widetilde { p } _ { a } ( \nu ) } .\tag{30}
$$

Exponential distribution. The unnormalized probability mass is

$$
\widetilde { p } _ { a } ( y ) = \exp ( - \lambda y ) ,\tag{31}
$$

where $\lambda = 5$ . The distribution is normalized over Y:

$$
p _ { a } ( y ) = \frac { \widetilde { p } _ { a } ( y ) } { \sum _ { \nu \in y } \widetilde { p } _ { a } ( \nu ) } .\tag{32}
$$

Figure 1A uses the checkpoint at epoch 200. The embedding-induced label distribution is computed as

$$
\widehat { p } _ { a } = \mathrm { s o f t m a x } \left( \pmb { w } ^ { U } \pmb { w } _ { a } ^ { E } \right) .\tag{33}
$$

The Pearson correlation coeficient is computed between $p _ { a }$ and $\widehat { p } _ { a }$ . The Kullback–Leibler divergence is computed as

$$
D _ { \mathrm { K L } } ( p _ { a } \| \widehat { p } _ { a } ) = \sum _ { y \in \mathcal { V } } p _ { a } ( y ) \log \frac { p _ { a } ( y ) } { \widehat { p } _ { a } ( y ) } ,\tag{34}
$$

and the Jensen–Shannon divergence is

$$
D _ { \mathrm { J S } } ( p _ { a } , \widehat { p } _ { a } ) = \frac { 1 } { 2 } D _ { \mathrm { K L } } ( p _ { a } \| m ) + \frac { 1 } { 2 } D _ { \mathrm { K L } } ( \widehat { p } _ { a } \| m ) , \qquad m = \frac { p _ { a } + \widehat { p } _ { a } } { 2 } .\tag{35}
$$

## A.1.3 Arithmetic Tasks

For the arithmetic experiments, the input number-token set is

$$
\mathcal { X } _ { N } = \{ 1 , \ldots , N \} .\tag{36}
$$

Each input is an ordered pair

$$
X = ( x _ { 1 } , x _ { 2 } ) , \qquad x _ { 1 } , x _ { 2 } \in \mathcal { X } _ { N } .\tag{37}
$$

We use the complete input set

$$
\mathcal { D } = \mathcal { X } _ { N } \times \mathcal { X } _ { N } ,\tag{38}
$$

containing $N ^ { 2 }$ samples. For Figure 1, we set $N = 2 0$ . and uses the checkpoint at epoch 100.

## A.1.4 Addition and Modular-Addition Dynamics

For Figures 2 and 3, we set $N = 4 0$ . The input set is again

$$
\mathcal { X } _ { 4 0 } = \{ 1 , \ldots , 4 0 \} .\tag{39}
$$

The addition model and the modular-addition model are trained for 1200 epochs. The early and late epochs shown in Figure 2 are 100 and 1100. The epochs used in Figure 3 are 100 and 1000.

## A.1.5 Optimized First-Order Signature Projection

For each token $\alpha ,$ the first-order label signature is represented as a matrix

$$
\begin{array} { r } { \varphi _ { \alpha } ^ { y ; 1 } \in \mathbb { R } ^ { | \mathcal { V } | \times | \mathcal { X } | } . } \end{array}\tag{40}
$$

Given a probe vector

$$
\mathbf { v } \in \mathbb { R } ^ { | \mathcal { X } | } ,\tag{41}
$$

we define the projected signature

$$
\begin{array} { r } { \mathbf { h } _ { \alpha } ^ { ( 1 ) } ( \mathbf { v } ) = \varphi _ { \alpha } ^ { y ; 1 } \mathbf { v } \in \mathbb { R } ^ { | \mathcal { V } | } . } \end{array}\tag{42}
$$

The induced pairwise similarity is

$$
S _ { a , b } ^ { \mathrm { s i g , ( 1 ) } } ( \mathbf { v } ) = \frac { \left. \mathbf { h } _ { a } ^ { ( 1 ) } ( \mathbf { v } ) , \mathbf { h } _ { b } ^ { ( 1 ) } ( \mathbf { v } ) \right. } { \left\| \mathbf { h } _ { a } ^ { ( 1 ) } ( \mathbf { v } ) \right\| _ { 2 } \left\| \mathbf { h } _ { b } ^ { ( 1 ) } ( \mathbf { v } ) \right\| _ { 2 } } .\tag{43}
$$

For every epoch t, the probe vector is optimized independently by solving

$$
\mathbf { v } ^ { ( t ) } = \arg \operatorname* { m a x } _ { \mathbf { v } } \mathrm { C o r r } \left( \mathrm { v e c } _ { \triangle } \left( \mathbf { S } ^ { \mathrm { s i g } , ( 1 ) } ( \mathbf { v } ) \right) , \mathrm { v e c } _ { \triangle } \left( \mathbf { S } ^ { \mathrm { e m b } , ( t ) } \right) \right) ,\tag{44}
$$

where ${ \mathrm { v e c } } \triangle$ extracts the strictly upper-triangular entries of a symmetric matrix.

The correlation objective is Pearson. The probe vector is initialized as

$$
v _ { i } \sim { \mathcal { N } } ( 0 , | { \mathcal { X } } | ^ { - 0 . 5 } ) .\tag{45}
$$

It is optimized using Adam with learning rate 0.01 for 300 steps. The probe vector is trained using five-fold cross-validation over token pairs. For each epoch, it is optimized on four folds and evaluated on the remaining fold, and the reported results are averaged over the five held-out folds.

## A.1.6 Fourier Analysis of the First-Order Signature

For Figure 3, we use the real cosine basis

$$
\mathbf { c } _ { k } ^ { \cos } = \left( \cos \left( { \frac { 2 \pi k j } { p } } \right) \right) _ { j = 0 } ^ { p - 1 } , \qquad k = 0 , 1 , \ldots , { \frac { p } { 2 } } ,\tag{46}
$$

where

$$
p = N = 4 0 .\tag{47}
$$

For each token α and frequency k, the projected signature is

$$
\mathbf { h } _ { \alpha , k } ^ { ( 1 ) } = \varphi _ { \alpha } ^ { y ; 1 } \mathbf { c } _ { k } ^ { \mathrm { { c o s } } } .\tag{48}
$$

Its pairwise cosine-similarity matrix is

$$
S _ { k ; a , b } ^ { \mathrm { s i g } , ( 1 ) } = \frac { \left. { \bf h } _ { a , k } ^ { ( 1 ) } , { \bf h } _ { b , k } ^ { ( 1 ) } \right. } { \left\| { \bf h } _ { a , k } ^ { ( 1 ) } \right\| _ { 2 } \left\| { \bf h } _ { b , k } ^ { ( 1 ) } \right\| _ { 2 } } .\tag{49}
$$

The agreement with the embedding geometry is measured by

$$
\rho _ { ( 1 ) , k } ^ { ( t ) } = \mathrm { C o r r } \left( \mathrm { v e c } _ { \triangle } \left( \mathbf { S } _ { k } ^ { \mathrm { s i g } , ( 1 ) } \right) , \mathrm { v e c } _ { \triangle } \left( \mathbf { S } ^ { \mathrm { e m b } , ( t ) } \right) \right) .\tag{50}
$$

We use Pearson correlation. The heatmaps include epochs sampled every 100 epochs.

For the PCA visualizations, the selected late-stage frequencies are $k = 2 0$ for the addition task and $k = 1 9$ for the modular-addition task. These frequencies are selected because they maximize the correlation at the displayed epoch.

We analyze only the cosine components because the zero-frequency sine basis is identically zero and therefore cannot represent the required zero-frequency projection.

## A.1.7 Activation-Order Sweep

This experiment corresponds to Figure 4. We consider modular-addition sequences of length $L \in \{ 2 , 3 , 4 \}$ , and use the monomial activation $\sigma ( z ) = z ^ { \odot K } , K \in \{ 1 , 2 , 3 , 4 , 5 , 6 \}$ . For a sequence

$$
X = ( x _ { 1 } , \ldots , x _ { L } ) , \qquad x _ { i } \in \mathcal { X } _ { N } ,\tag{51}
$$

the target is

$$
y = 1 + \left( \sum _ { i = 1 } ^ { L } x _ { i } \right) { \bmod { N } } ,\tag{52}
$$

where $N = 5 0$ . For each pair $( L , K )$ , the training set contains all $N ^ { L }$ sequences. Each model is trained for 1000 epochs.

## A.2 Attention-Only Experiments

## A.2.1 Attention A<sub>r</sub>chitect<sub>ur</sub>e

We use $d = 1 2 8 , d _ { k } = 6 4$ . The attention model uses causal masking. It contains learned positional embeddings. It contains no residual connection, no normalization, and no FFN module. The attention and output layers use no bias.

## A.2.2 Attention Arithmetic Experiments

The addition and modular-addition tasks in Figure 5A–H use $N = 2 0$ . The input dataset contains all $N ^ { 2 }$ ordered pairs. The models are trained for 10000 epochs. The early and late epochs visualized in Figure 5 are 100 and 10000.

## A.2.3 Anchor–Noise Parity Task

This experiment corresponds to Figure 5I–L. We use the anchor-token set

$$
\mathcal { A } = \{ 1 1 , 1 2 , \dots , 2 0 \}\tag{53}
$$

and the noise-token set

$$
{ \mathcal { R } } = \{ 2 1 , 2 2 , \ldots , 6 0 \} .\tag{54}
$$

Each sequence has length 8. A sequence contains one anchor token $a \in { \mathcal { A } }$ and $L - 1$ noise tokens sampled from $\textstyle { \mathcal { R } } \colon$

$$
X = ( r _ { 1 } , \ldots , r _ { m - 1 } , a , r _ { m + 1 } , \ldots , r _ { L } ) .\tag{55}
$$

The anchor position m is sampled uniformly from $\{ 1 , \ldots , L \} ]$ . The anchor token is sampled uniformly from A. Noise tokens are sampled independently with replacement from $\mathcal { R }$

The training set is generated once with 10000 samples. The model is trained for 200 epochs.

For a sequence $X ,$ , the attention mass assigned to the anchor token is

$$
A _ { \mathrm { a n c h o r } } ( X ) = \sum _ { s = 1 } ^ { L } a _ { s } \mathbf { 1 } _ { \{ x _ { s } \in \mathcal { A } \} } ,\tag{56}
$$

and the attention mass assigned to the noise tokens is

$$
A _ { \mathrm { n o i s e } } ( X ) = \sum _ { s = 1 } ^ { L } a _ { s } \mathbf { 1 } _ { \{ x _ { s } \in \mathcal { R } \} } .\tag{57}
$$

The curves in Figure 5K report the average over the full dataset of these quantities. For a single-anchor sequence, $A _ { \mathrm { a n c h o r } } ( X )$ is the attention weight assigned to the anchor position.

The average embedding norms are computed as

$$
E _ { \mathrm { a n c h o r } } ( t ) = { \frac { 1 } { | { \cal A } | } } \sum _ { \alpha \in { \cal A } } \left\| { \pmb w } _ { \alpha } ^ { E , ( t ) } \right\| _ { 2 } ,\tag{58}
$$

and

$$
E _ { \mathrm { n o i s e } } ( t ) = \frac { 1 } { | \mathcal { R } | } \sum _ { \alpha \in \mathcal { R } } \left\| \pmb { w } _ { \alpha } ^ { E , ( t ) } \right\| _ { 2 } .\tag{59}
$$

## A.3 Signature-Based Embedding Experiments

This section provides the details of the fixed signature-embedding experiment shown in Figure 7D.

## A.3.1 Fixed First-Order Signature Embedding

For every token $\alpha \in { \mathcal { X } }$ , we first compute its empirical first-order label signature

$$
\begin{array} { r } { \varphi _ { \alpha } ^ { y ; 1 } \in \mathbb { R } ^ { | \mathcal { V } | \times | \mathcal { X } | } . } \end{array}\tag{60}
$$

The fixed signature embedding is constructed using a random linear projection:

$$
\begin{array} { r } { { \pmb w } _ { \alpha } ^ { E , \mathrm { s i g } } = \lambda { \bf R } \varphi _ { \alpha } ^ { y ; 1 } { \pmb v } , } \end{array}\tag{61}
$$

where

$$
\lambda \in \mathbb { R } , \mathbf { R } \in \mathbb { R } ^ { d \times | \mathcal { V } | } , \pmb { v } \in \mathbb { R } ^ { | \mathcal { X } | } .\tag{62}
$$

The entries of R and v are sampled independently and fixed during training. The scale parameter λ are trainable.

## A.3.2 Baselines

We compare the following two embedding settings.

Trainable embedding. The input embedding matrix is initialized according to the common initialization rule and optimized jointly with the remaining model parameters.

Fixed random embedding. Each token is assigned a random embedding

$$
\begin{array} { r } { { \pmb w } _ { \alpha } ^ { E , \mathrm { r a n d } } = \lambda { \pmb v } _ { \alpha } , } \end{array}\tag{63}
$$

where ${ \pmb v } _ { \alpha } \in \mathbb { R } ^ { d }$ remains fixed throughout training and the scale parameter λ is trainable.

## A.3.3 FFN and Attention Comparison in Figure 7

The experiment in Figure 7A–C uses the task $F _ { \mathrm { m o d } }$ with $L = 2 , N = 4 0$ . The models are trained for 1000 epochs using identical optimization settings. The PCA projections in Figure 7B and Figure 7C use the final epochs at epoch 1000.

## A.4 LLMs

Model Architecture We use a 0.7B dense decoder-only Transformer throughout the language-model experiments. The model follows a standard decoder-only Transformer architecture with a pre-norm design. Compared with the original GPT-3 architecture Brown et al. (2020), we replace LayerNorm with RMSNorm and the standard MLP with SwiGLU. The model consists of 24 Transformer layers with hidden size 1024, 16 attention heads, attention head dimension 96, and an MLP intermediate dimension of 4096. Rotary Position Embeddings (RoPE) are adopted for positional encoding, and the vocabulary size is 60,416. The model contains approximately 680M non-embedding parameters.

Training Methods We train the model using the standard next-token prediction objective with Megatron-LM Shoeybi et al. (2019) and the AdamW optimizer. We use a learning-rate ratio $r _ { \mathrm { l r } } ~ = ~ 0 . 1$ , a warm-up ratio $r _ { \mathrm { w a r m u p } } = 0 . 0 3$ , a peak learning rate of $\eta _ { \mathrm { p e a k } } = 2 . 5 \times 1 0 ^ { - 4 }$ , and a global batch size of $B = 2 5 6$

Training is performed using bfloat16 mixed precision, while gradient accumulation is carried out in fp32 precision. Both attention dropout and hidden dropout are disabled throughout training.

Data The experiments were conducted on a high-quality bilingual (Chinese–English) corpus containing 36B tokens and spanning multiple domains, including 13.5B web tokens, 9B Wikipedia tokens, 4.5B code tokens, and 9B mathematics tokens.

## Appendix B. Theoretical Details

## B.1 Proof of Theorem 1,2

Recalling that

$$
\pmb { f } _ { \mathrm { f f n } } ( \boldsymbol { X } ) : = \boldsymbol { W } ^ { U } \pmb { g } _ { \mathrm { f f n } } ( \boldsymbol { X } ) = \boldsymbol { W } ^ { U } \boldsymbol { \sigma } \left( \sum _ { i = 1 } ^ { L } \pmb { w } _ { x _ { i } } ^ { E } \right) .
$$

With the Assumption 1, we have that

$$
\sigma ^ { \prime } \left( \sum _ { i = 1 } ^ { L } \pmb { w } _ { x _ { i } } ^ { E } \right) = \sum _ { k = 1 } ^ { K } k C _ { k } \left( \sum _ { i = 1 } ^ { L } \pmb { w } _ { x _ { i } } ^ { E } \right) ^ { \odot \left( k - 1 \right) } .
$$

Therefore,

$$
\frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } \bigg | ^ { y } = L r _ { \alpha } \sum _ { k = 1 } ^ { K } k C _ { k } { \pmb J } _ { \alpha , k } ^ { y } , \qquad \frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } \bigg | ^ { p } = L r _ { \alpha } \sum _ { k = 1 } ^ { K } k C _ { k } { \pmb J } _ { \alpha , k } ^ { p } ,
$$

where

$$
\boldsymbol { J } _ { \alpha , k } ^ { y } : = \mathbb { E } \left[ \left( \sum _ { i = 1 } ^ { L } \boldsymbol { w } _ { x _ { i } } ^ { E } \right) ^ { \odot ( k - 1 ) } \odot \boldsymbol { W } ^ { U , T } \boldsymbol { e } _ { y } \mid \alpha \right] ,
$$

and

$$
J _ { \alpha , k } ^ { p } : = \mathbb { E } \left[ \left( \sum _ { i = 1 } ^ { L } \pmb { w } _ { x _ { i } } ^ { E } \right) ^ { \odot ( k - 1 ) } \odot \pmb { W } ^ { U , T } \pmb { p } ( X ) \mid \alpha \right] .
$$

Label-driven term. Conditioned on α, we write the as

$$
\sum _ { i = 1 } ^ { L } \pmb { w } _ { x _ { i } } ^ { E } = \pmb { w } _ { \alpha } ^ { E } + \sum _ { j = 1 } ^ { L - 1 } \pmb { w } _ { z _ { j } } ^ { E } .
$$

For the k-th polynomial term, the coordinate-wise multinomial expansion gives

$$
\left( \sum _ { i = 1 } ^ { L } w _ { x _ { i } } ^ { E } \right) ^ { \odot ( k - 1 ) } = \sum _ { m = 0 } ^ { \operatorname* { m i n } \{ L - 1 , k - 1 \} } \sum _ { \substack { l \ge 0 , n _ { 1 } , \cdots , n _ { m } \ge 1 } } \frac { ( k - 1 ) ! } { l ! \prod _ { j \in J } n _ { j } ! } ( w _ { \alpha } ^ { E } ) ^ { \odot l } \odot \sum _ { \substack { J = \{ j _ { 1 } , \cdots , j _ { m } \} \subseteq [ L - 1 ] ^ { m } = 1 } } \stackrel { m } { \longrightarrow } ( w _ { z _ { j _ { r } } } ^ { E } ) ^ { \odot n _ { r } } .\tag{64}
$$

Let $\pmb { w } _ { \nu } ^ { U } : = \pmb { W } ^ { U , T } \pmb { e } _ { \nu } \in \mathbb { R } ^ { d }$ denote the unembedding vector of token ν. Substituting the above expansion into $J _ { \alpha , k } ^ { y }$ yields

$$
\begin{array} { r } { J _ { \alpha , k } ^ { y } = \displaystyle \sum _ { m = 0 } ^ { \operatorname* { m i n } \{ L - 1 , k - 1 \} } \displaystyle \sum _ { \stackrel { l \geq 0 , n _ { 1 } , \cdots , n _ { m } \geq 1 } { l + \sum _ { r = 1 } ^ { m } n _ { r } = k - 1 } } \frac { ( k - 1 ) ! } { l ! \prod _ { r = 1 } ^ { m } n _ { r } ! } ( { \pmb w } _ { \alpha } ^ { E } ) ^ { \odot l } \odot } \\ { \displaystyle \sum _ { \substack { J = \{ j _ { 1 } , \cdots , j _ { m } \} \subseteq [ L - 1 ] } } { \mathbb { E } } \left[ { \pmb w } _ { \nu } ^ { U } \odot \bigodot _ { r = 1 } ^ { m } ( { \pmb w } _ { z _ { j _ { r } } } ^ { E } ) ^ { \odot n _ { r } } \mid { \alpha } \right] . } \end{array}\tag{65}
$$

For $J = \{ j _ { 1 } , \dots , j _ { m } \}$ , it’s noted that

$$
\mathbb { E } \left[ w _ { \nu } ^ { U } \odot \bigodot _ { r = 1 } ^ { m } ( w _ { z _ { j _ { r } } } ^ { E } ) ^ { \odot n _ { r } } \mid \alpha \right] = \sum _ { \nu , \beta _ { r } \in \mathcal { V } } \mathbb { P } \left( y = \nu , z _ { j _ { r } } = \beta _ { r } \mid \alpha \right) w _ { \nu } ^ { U } \odot \bigodot _ { r = 1 } ^ { m } ( w _ { \beta _ { r } } ^ { E } ) ^ { \odot n _ { r } } .
$$

After summing over all exponent assignments, the resulting kernel is symmetric in its m context indices. Hence averaging over ordered distinct tuples is equivalent to averaging over unordered subsets, and the sum over the $\binom { L - 1 } { m }$ subsets can be written as

$$
J _ { \alpha , k } ^ { y } = \sum _ { m = 0 } ^ { \operatorname* { m i n } \{ L - 1 , k - 1 \} } \binom { L - 1 } { m } \sum _ { \nu , \beta _ { 1 } , \dots , \beta _ { m } \in \mathcal { V } } \varphi _ { \alpha } ^ { y ; m } ( \nu , \beta _ { 1 } , \dots , \beta _ { m } )
$$

$$
\times \left[ \sum _ { l \geq 0 , n _ { 1 } , \ldots , n _ { m } \geq 1 } \frac { ( k - 1 ) ! } { l ! \prod _ { r = 1 } ^ { m } n _ { r } ! } ( { \pmb w } _ { \alpha } ^ { E } ) ^ { \odot l } \odot { \pmb w } _ { \nu } ^ { U } \odot \bigodot _ { r = 1 } ^ { m } ( { \pmb w } _ { \beta _ { r } } ^ { E } ) ^ { \odot n _ { r } } \right] .
$$

Define that $\mathcal { W } _ { \alpha } ^ { y , m , k } \in \mathbf { \Omega } ( \mathbb { R } ^ { d } ) ^ { \frac { V \times \cdot \cdot \cdot \times V } { m + 1 } }$ is a tensor with order $m + 1$ , whose elements is a d-dimensional vector with

$$
\mathcal { W } _ { \alpha } ^ { y , m , k } \left( \nu , \beta _ { 1 } , \cdot \cdot \cdot , \beta _ { m } \right) = \binom { L - 1 } { m } \sum _ { \stackrel { l \geq 0 , n _ { 1 } , \cdots , n _ { m } \geq 1 } { l \neq \sum _ { r = 1 } ^ { m } n _ { r } = k - 1 } } \frac { ( k - 1 ) ! } { l ! \prod _ { r = 1 } ^ { m } n _ { r } ! } ( w _ { \alpha } ^ { E } ) ^ { \odot l } \odot w _ { \nu } ^ { U } \odot \bigcup _ { r = 1 } ^ { m } ( w _ { \beta _ { r } } ^ { E } ) ^ { \odot n _ { r } } ,
$$

then we have that

$$
J _ { \alpha , k } ^ { y } = \sum _ { m = 0 } ^ { \operatorname* { m i n } \{ L - 1 , k - 1 \} } \psi _ { \alpha } ^ { y , m , k } \cdot \varphi _ { \alpha } ^ { y ; m } , \quad \frac { d w _ { \alpha } ^ { E } } { d t } \bigg | ^ { y } = L r _ { \alpha } \sum _ { k = 1 } ^ { K } k C _ { k } J _ { \alpha , k } ^ { y } = L r _ { \alpha } \sum _ { m = 0 } ^ { \operatorname* { m i n } \{ L - 1 , K - 1 \} } A _ { \alpha , m } \cdot \varphi _ { \alpha } ^ { y ; m } ,
$$

which adimits the Theorem 1.

Prediction-induced term. We next derive the operator associated with the predictioninduced term:

$$
\pmb { J } _ { \alpha , k } ^ { p } = \mathbb { E } \left[ \left( \sum _ { i = 1 } ^ { L } \pmb { w } _ { x _ { i } } ^ { E } \right) ^ { \odot ( k - 1 ) } \odot \pmb { W } ^ { U , T } \pmb { p } ( X ) \mid \alpha \right] .
$$

At the early stage of training, the logits are small. Therefore, we use the first-order softmax expansion around the origin: $\pmb { p } ( X ) = \frac { 1 } { V } \pmb { 1 } + O ( \| \pmb { f } ( X ) \| )$ Since $\pmb { \mathscr { f } } ( X ) = \pmb { W } ^ { U } \sigma ( \pmb { h } ( X ) )$ , we obtain

$$
\pmb { W } ^ { U , T } \pmb { p } ( X ) = \pmb { b } _ { 0 } + O ( \| \pmb { f } ( X ) \| ) ,
$$

where $\pmb { b } _ { 0 } : = \frac { 1 } { V } \pmb { W } ^ { U , T } \pmb { 1 }$ . Using the same multinomial expansion of $h ( X ) ^ { \odot ( k - 1 ) }$ , we get

$$
\begin{array} { l } { { \displaystyle { J _ { \alpha , k } ^ { p } = \sum _ { m = 0 } ^ { \operatorname* { m i n } \{ L - 1 , k - 1 \} } \sum _ { l \geq 0 , \ n _ { 1 } , \cdots , n _ { m } \geq 1 } \sum _ { J = \{ j _ { 1 } , \cdots , j _ { m } \} \subseteq [ L - 1 ] } \frac { ( k - 1 ) ! } { l ! \prod _ { r = 1 } ^ { m } n _ { r } ! } } } } \\ { { \displaystyle { \mathbb { E } \left[ b _ { 0 } \odot ( { \pmb w } _ { \alpha } ^ { E } ) ^ { \odot l } \odot \bigodot \bigodot \bigodot ( \boldsymbol { \varpi } _ { r } ^ { E } ) ^ { \odot n _ { r } } \mid \alpha \right] } . } } \end{array}
$$

For $J = \{ j _ { 1 } , \dots , j _ { m } \}$

After the same symmetrization over context positions, this becomes

$$
\begin{array} { l } { { \displaystyle { \cal J } _ { \alpha , k } ^ { p } = \sum _ { m = 0 } ^ { \operatorname* { m i n } \{ L - 1 , k - 1 \} } \left( { \displaystyle L - 1 } \right) \sum _ { \beta _ { 1 } , \dots , \beta _ { m } \in \mathcal { V } } \varphi _ { \alpha } ^ { m } ( \beta _ { 1 } , \dots , \beta _ { m } ) } } \\ { { \displaystyle \left[ \sum _ { l \ge 0 , \ n _ { 1 } , \dots , n _ { m } \ge 1 } \frac { ( k - 1 ) ! } { l ! \prod _ { r = 1 } ^ { m } n _ { r } ! } b _ { 0 } \odot ( { \pmb w } _ { \alpha } ^ { E } ) ^ { \odot l } \odot \bigcap _ { r = 1 } ^ { m } ( { \pmb w } _ { \beta _ { r } } ^ { E } ) ^ { \odot n _ { r } } \right] } . } \end{array}
$$

Therefore, define $\mathcal { U } _ { \alpha } ^ { m , k } \in \mathsf { \Gamma } ( \mathbb { R } ^ { d } ) \frac { V \times \cdots \times V } { m }$ is a tensor with order $m ,$ , whose element is a d-dimensional vector with

$$
\mathcal { U } _ { \alpha } ^ { m , k } \left( \beta _ { 1 } , \dots , \beta _ { m } \right) : = \binom { L - 1 } { m } \sum _ { l \ge 0 , \ n _ { 1 } , \dots , n _ { m } \ge 1 } \frac { ( k - 1 ) ! } { l ! \prod _ { r = 1 } ^ { m } n _ { r } ! } b _ { 0 } \odot ( w _ { \alpha } ^ { E } ) ^ { \odot l } \odot \binom { m } { r = 1 } ( w _ { \beta _ { r } } ^ { E } ) ^ { \odot n _ { r } } .
$$

Then we have

$$
J _ { \alpha , k } ^ { p } = \sum _ { m = 0 } ^ { \operatorname* { m i n } \{ L - 1 , k - 1 \} } \mathcal { U } _ { \alpha } ^ { m , k } \cdot \varphi _ { \alpha } ^ { m } , \quad \frac { d w _ { \alpha } ^ { E } } { d t } \Bigg | ^ { p } = L r _ { \alpha } \sum _ { m = 1 } ^ { \operatorname* { m i n } \{ L - 1 , K - 1 \} } \mathcal { B } _ { \alpha } ^ { m } \cdot \varphi _ { \alpha } ^ { m } + R _ { \alpha } .
$$

where $\begin{array} { r } { B _ { \alpha , m } = \sum _ { k = 1 } ^ { K } k C _ { k } \mathcal { U } _ { \alpha } ^ { m , k } } \end{array}$

## B.2 Supplementary of Remark 1

In this section, we will further discuss the formulation of $\begin{array} { r } { \frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } \bigg | ^ { p } } \end{array}$ in $f _ { \mathrm { f f n } }$ , with an expansion to the linear term of the softmax function and we will show that more higher-order signatures are induced. Recall that:

$$
\boldsymbol { J } _ { \alpha , k } ^ { p } = \mathbb { E } \left[ \left( \sum _ { i = 1 } ^ { L } \boldsymbol { w } _ { x _ { i } } ^ { E } \right) ^ { \odot ( k - 1 ) } \odot \boldsymbol { W } ^ { U , T } \boldsymbol { p } ( \boldsymbol { X } ) \mid \alpha \right] .
$$

At the early stage of training, we use the first-order softmax expansion around the origin:

$$
p ( X ) = \frac { 1 } { V } \mathbf { 1 } + \frac { 1 } { V } { \pmb { H } } { \pmb { f } } ( X ) + O ( \| { \pmb { f } } ( X ) \| ^ { 2 } ) ,
$$

where

$$
\pmb { H } : = \pmb { I } _ { V } - \frac { 1 } { V } \pmb { 1 } \pmb { 1 } ^ { T } .
$$

Since $\pmb { f } ( \boldsymbol { X } ) = \pmb { W } ^ { U } \sigma ( \pmb { h } ( \boldsymbol { X } ) )$ , we obtain

$$
W ^ { U , T } \pmb { p } ( X ) = b _ { 0 } + \frac { 1 } { V } M \sigma \left( \sum _ { i = 1 } ^ { L } \pmb { w } _ { \pmb { x } _ { i } } ^ { E } \right) + O ( \Vert \pmb { f } ( X ) \Vert ^ { 2 } ) ,
$$

where

$$
b _ { 0 } : = \frac { 1 } { V } W ^ { U , T } { \bf 1 } , \qquad M : = W ^ { U , T } H W ^ { U } .
$$

Thus $J _ { \alpha , k } ^ { p }$ decomposes into a uniform-prediction part and a linearized-logit part:

$$
J _ { \alpha , k } ^ { p } = J _ { \alpha , k } ^ { p , 0 } + J _ { \alpha , k } ^ { p , 1 } + R _ { \alpha , k } ^ { p } ,
$$

where

$$
\boldsymbol { J } _ { \alpha , k } ^ { p , 0 } : = \mathbb { E } \left[ \left( \sum _ { i = 1 } ^ { L } \boldsymbol { w } _ { x _ { i } } ^ { E } \right) ^ { \odot ( k - 1 ) } \odot \boldsymbol { b } _ { 0 } \mid \alpha \right] ,
$$

$$
J _ { \alpha , k } ^ { p , 1 } : = \frac { 1 } { V } \mathbb { E } \left[ \left( \sum _ { i = 1 } ^ { L } w _ { x _ { i } } ^ { E } \right) ^ { \odot ( k - 1 ) } \odot M \sigma ( \pmb { h } ( X ) ) \mid \alpha \right] ,
$$

and $R _ { \alpha , k } ^ { p }$ contains the higher-order softmax remainder. The analysis on $J _ { \alpha , k } ^ { p , 0 }$ has been completed in the previous section.

Linearized-logit part. Using $\begin{array} { r } { \sigma \left( \sum _ { i = 1 } ^ { L } \pmb { w } _ { x _ { i } } ^ { E } \right) = \sum _ { q = 1 } ^ { K } C _ { q } \left( \sum _ { i = 1 } ^ { L } \pmb { w } _ { x _ { i } } ^ { E } \right) ^ { \odot q } } \end{array}$ , we have

$$
J _ { \alpha , k } ^ { p , 1 } = \frac { 1 } { V } \sum _ { q = 1 } ^ { K } C _ { q } \mathbb { E } \left[ \left( \sum _ { i = 1 } ^ { L } \pmb { w } _ { x _ { i } } ^ { E } \right) ^ { \odot ( k - 1 ) } \odot \pmb { M } \left( \sum _ { i = 1 } ^ { L } \pmb { w } _ { x _ { i } } ^ { E } \right) ^ { \odot q } | \alpha \right] .
$$

It’s noted that

$$
\begin{array} { r l } & { \mathbb { E } [ ( \displaystyle \sum _ { i = 1 } ^ { L } w _ { x _ { i } } ^ { E } ) ^ { \odot ( k - 1 ) } \odot M ( \displaystyle \sum _ { i = 1 } ^ { L } w _ { x _ { i } } ^ { E } ) ^ { \odot q } \mid \alpha ] } \\ & { = \mathrm { m i n } \{ L - 1 , k - 1 + q \} } \\ & { = 0 \qquad \displaystyle \sum _ { m = 0 } ^ { \operatorname* { m i n } \{ L - 1 , k - 1 \} } \sum _ { \substack { \substack { l _ { 1 } , l _ { 2 } \geq 0 , n _ { 1 } ^ { ( 1 ) } , \ldots , n _ { m } ^ { ( 1 ) } , n _ { 1 } ^ { ( 1 ) } , \ldots , n _ { m } ^ { ( 2 ) } \geq 1 } } } \sum _ { \substack { l _ { 1 } = \{ l _ { 1 } ^ { ( 1 ) } , \ldots , \ldots , l _ { m } ^ { ( 1 ) } \geq [ L - 1 ] \} } } \frac { ( k - 1 ) ! } { l _ { 1 } ! \prod _ { r = 1 } ^ { m _ { 1 } } n _ { r } ^ { ( 1 ) } ! l _ { 2 } ! \prod _ { r = 1 } ^ { m _ { 2 } } n _ { r } ^ { ( 2 ) } ! } } \\ & { \qquad l _ { 1 } l _ { 2 } + l _ { 2 } + \sum _ { r = 1 } ^ { m _ { 1 } } n _ { r } ^ { ( 1 ) } + \sum _ { r = 1 } ^ { m _ { 2 } } \frac { ( l _ { 1 } ^ { ( 2 ) } ) - \zeta _ { l } ^ { ( 2 ) } ( [ l - 1 ] ) } { l _ { r } ^ { ( 1 ) } + \zeta _ { l } ^ { ( 1 ) } } [ \alpha ] \sum _ { \substack { l _ { 1 } \geq j _ { 2 } \geq 0 } } ^ { \infty } [ l - 1 ] \} } \\ &  \mathbb { E } [ ( w _ { \alpha } ^ { E } ) ^ { l _ { 1 } } \odot \displaystyle \sum _ { r = 1 } ^ { m _ { 1 } } ( W _ { \Sigma _ { r , r } ^ { ( 1 ) } } ^ { E } ) ^  \odot n _ { r } \end{array}
$$

since

$$
\begin{array} { r l } & { \mathbb { E } \left[ ( { \pmb w } _ { \alpha } ^ { E } ) ^ { l _ { 1 } } \odot \overset { m _ { 1 } } { \underset { r = 1 } { \overset { m _ { 1 } } { \prod } } } ( { \pmb W } _ { z _ { j _ { r } } } ^ { E } ) ^ { \odot n _ { r } ^ { ( 1 ) } } \odot M \left( ( { \pmb w } _ { \alpha } ^ { E } ) ^ { l _ { 2 } } \odot \overset { m _ { 2 } } { \underset { r = 1 } { \overset { m _ { 2 } } { \prod } } } ( { \pmb W } _ { z _ { j _ { r } } } ^ { E } ) ^ { \odot n _ { r } ^ { ( 2 ) } } \right) \mid \alpha \right] } \\ & { = \displaystyle \sum _ { \beta _ { 1 } , \beta _ { 2 } , \dots , \beta _ { m } \in \mathcal { V } } \mathbb { P } \left( z _ { j _ { 1 } } ^ { } ( \beta _ { 1 } , \dots , z _ { j _ { m _ { 1 } } } ^ { } = \beta _ { m _ { 1 } } , z _ { j _ { 1 } ^ { ( 2 ) } } = \beta _ { m _ { 1 } } + 1 , \dots , z _ { j _ { m _ { 2 } } ^ { ( 2 ) } } = \beta _ { m } \mid \alpha \right) } \\ & { \qquad ( { \pmb w } _ { \alpha } ^ { E } ) ^ { l _ { 1 } } \odot \overset { m _ { 1 } } { \underset { r = 1 } { \overset { m _ { 1 } } { \bigodot } } } ( { \pmb W } _ { \beta _ { r } } ^ { E } ) ^ { \odot n _ { r } ^ { ( 1 ) } } \odot M \left( ( { \pmb w } _ { \alpha } ^ { E } ) ^ { l _ { 2 } } \odot \overset { m _ { 2 } } { \underset { r = 1 } { \overset { m _ { 2 } } { \bigodot } } } ( { \pmb W } _ { \beta _ { m _ { 1 } + r } } ^ { E } ) ^ { \odot n _ { r } ^ { ( 2 ) } } \right) , } \end{array}
$$

Define that $\mathcal { U } _ { \alpha , 1 } ^ { m , k } \in ( \mathbb { R } ^ { d } ) ^ { \underbrace { V \times \dots \times V } _ { m } }$ with element

$$
\begin{array} { r l } { \mathcal { U } _ { \alpha , 1 } ^ { m , k } \left( \beta _ { 1 } , \dots , \beta _ { m } \right) = \left( \displaystyle L - 1 \right) \frac { 1 } { V } \displaystyle \sum _ { \substack { q = \operatorname* { m a x } \{ 1 , m - k + 1 \} } } ^ { K } } &  \displaystyle \sum _ { \substack { l _ { 1 } , l _ { 2 } \geq 0 , n _ { 1 } ^ { ( 1 ) } \} } ^ { \sum } \frac { ( k - 1 ) ! } { l _ { 1 } ! \prod _ { r = 1 } ^ { m } n _ { 1 } ^ { ( 2 ) } ! } \frac { q ! } { l _ { 2 } ! \prod _ { r = 1 } ^ { m } n _ { \tau } ^ { ( 2 ) } ! } } \\ { \displaystyle \qquad } & { \displaystyle \qquad i _ { 1 } + l _ { 2 } + \sum _ { r = 1 } ^ { m } n _ { r } ^ { ( 1 ) } + \sum _ { r = 2 } ^ { m } n _ { r } ^ { ( 2 ) } } \\ { \displaystyle \qquad } & { \displaystyle \{ w _ { \alpha } ^ { E } \} ^ { l _ { 1 } } \odot \bigtriangleup _ { \gamma } ^ { ( } W _ { \beta _ { r } } ^ { E ) \odot n _ { r } ^ { ( 1 ) } } \odot M \left( ( w _ { \alpha } ^ { E } ) ^ { l _ { 2 } } \odot \bigtriangleup _ { \gamma = 1 } ^ { ( 1 ) } ( W _ { \beta _ { m + 1 } } ^ { E } ) ^ { \odot n _ { r } ^ { ( 2 ) } } \right) \in \mathbb { R } ^ { d } , } \end{array}
$$

then we have

$$
\begin{array} { c } { { \displaystyle { \cal J } _ { \alpha , k } ^ { p , 1 } = \sum _ { m = 0 } ^ { \operatorname* { m i n } \{ L - 1 , k - 1 + K \} } { \mathcal { U } _ { \alpha , 1 } ^ { m , k } \cdot \varphi _ { \alpha } ^ { m } } } } \end{array}
$$

Combining the uniform-prediction and linearized-logit parts, we write

$$
\begin{array} { r } { { \cal J } _ { \alpha , k } ^ { p } = \displaystyle \sum _ { m = 0 } ^ { \operatorname* { m i n } \{ L - 1 , k - 1 \} } \mathcal { U } _ { \alpha , 0 } ^ { m , k } \cdot \varphi _ { \alpha } ^ { m } + \sum _ { m = 0 } ^ { \operatorname* { m i n } \{ L - 1 , k - 1 + K \} } \mathcal { U } _ { \alpha , 1 } ^ { m , k } \varphi _ { \alpha } ^ { m } + R _ { \alpha } ^ { p , k } . } \end{array}
$$

Summation of k, we obtain

$$
\begin{array} { l } { \displaystyle \frac { d w _ { \alpha } ^ { E } } { d t } \bigg | ^ { p } = L r _ { \alpha } \sum _ { k = 1 } ^ { K } k C _ { k } \left( \sum _ { m = 0 } ^ { \operatorname* { m i n } \{ L - 1 , k - 1 \} } \mathcal { U } _ { \alpha , 0 } ^ { m , k } \cdot \varphi _ { \alpha } ^ { m } + \sum _ { m = 0 } ^ { \operatorname* { m i n } \{ L - 1 , k - 1 + K \} } \mathcal { U } _ { \alpha , 1 } ^ { m , k } \varphi _ { \alpha } ^ { m } + R _ { \alpha } ^ { p , k } \right) } \\ { \displaystyle \qquad = L r _ { \alpha } \left( \sum _ { m = 0 } ^ { \operatorname* { m i n } \{ L - 1 , K - 1 \} } B _ { \alpha } ^ { m } \cdot \varphi _ { \alpha } ^ { m } + \sum _ { m = 0 } ^ { \operatorname* { m i n } \{ L - 1 , 2 K - 1 \} } \mathcal { C } _ { \alpha } ^ { m } \cdot \varphi _ { \alpha } ^ { m } \right) + \mathcal { R } _ { \alpha } . } \end{array}
$$

where $\begin{array} { r } { \mathcal { C } _ { \alpha , m } = \sum _ { k = 1 } ^ { K } k C _ { k } \mathcal { U } _ { \alpha , 1 } ^ { m , k } } \end{array}$

## B.3 Proof of Proposition 1

Lemma 1. For any token α with $r _ { \alpha } > 0$ and any $1 \leq m \leq L - 1$ , the label signatures satisfy

$$
\sum _ { \nu , \beta _ { 1 } , \dots , \beta _ { m } \in \mathcal { V } } \varphi _ { \alpha } ^ { y ; m } ( \nu , \beta _ { 1 } , \dots , \beta _ { m } ) = 1 .
$$

Moreover, for every context mode $q \in [ m ]$ 2

$$
\sum _ { \beta _ { q } \in \mathcal { V } } \varphi _ { \alpha } ^ { y ; m } ( \nu , \beta _ { 1 } , \ldots , \beta _ { m } ) = \varphi _ { \alpha } ^ { y ; m - 1 } ( \nu , \beta _ { 1 } , \ldots , \beta _ { q - 1 } , \beta _ { q + 1 } , \ldots , \beta _ { m } ) .
$$

Consequently,

$$
\begin{array} { r } { \| \varphi _ { \alpha } ^ { y ; m } \| _ { F } \leq \| \varphi _ { \alpha } ^ { y ; m - 1 } \| _ { F } \leq \cdot \cdot \cdot \leq \| \varphi _ { \alpha } ^ { y ; 0 } \| _ { F } . } \end{array}
$$

The same normalization, marginal-consistency, and norm-monotonicity properties hold for the co-occurrence signatures after omitting the label index.

Proof. By definition, $\varphi _ { \alpha } ^ { y ; m }$ is the joint distribution of the label and the tokens at an ordered tuple of m distinct context positions sampled uniformly from $\mathcal { I } _ { m }$ . Its entries are therefore nonnegative and sum to one.

Fix $q \in [ m ]$ . After deleting the q-th coordinate from a uniformly sampled tuple in $\mathcal { T } _ { m } .$ the remaining ordered $( m - 1 )$ )-tuple is uniform on $\mathcal { T } _ { m - 1 }$ . Indeed, every fixed element of $\mathcal { T } _ { m - 1 }$ has exactly $L - m$ preimages in $\mathcal { I } _ { m }$ , and

$$
( L - 1 ) _ { m } = ( L - 1 ) _ { m - 1 } ( L - m ) .
$$

Summing over the token value at the deleted coordinate therefore gives precisely the order-$( m - 1 )$ marginal, proving the stated consistency relation.

To prove the norm inequality, fix all indices except $\beta _ { q }$ and use nonnegativity:

$$
\sum _ { \beta _ { q } \in \mathcal { V } } \varphi _ { \alpha } ^ { y ; m } ( \nu , \beta _ { 1 } , \dots , \beta _ { m } ) ^ { 2 } \leq \left( \sum _ { \beta _ { q } \in \mathcal { V } } \varphi _ { \alpha } ^ { y ; m } ( \nu , \beta _ { 1 } , \dots , \beta _ { m } ) \right) ^ { 2 } .
$$

Summing this inequality over all remaining indices and applying the marginal-consistency identity yields

$$
\| \varphi _ { \alpha } ^ { y ; m } \| _ { F } ^ { 2 } \leq \| \varphi _ { \alpha } ^ { y ; m - 1 } \| _ { F } ^ { 2 } .
$$

Iterating over m proves the result. The co-occurrence case is identical.

Proposition 1. We first estimate the size of each contracted order-m operator at initialization. Recall that the order-m label-driven operator can be decomposed according to the polynomial degree:

$$
\mathcal { A } _ { \alpha , m } = \sum _ { k = m + 1 } ^ { K } k C _ { k } \mathcal { W } _ { \alpha } ^ { y ; m , k } .
$$

The lowest possible degree is $k = m + 1$ . For fixed m and $k ,$ each entry of ${ \mathcal W } _ { \alpha } ^ { y ; m , k }$ is a finite linear combination of coordinate-wise products containing one unembedding factor and $k - 1$ embedding factors. Since each factor has coordinate scale $d ^ { - \gamma }$ , each entry has second moment at most of order $d ^ { - 2 k \gamma }$ . The numbers of terms in these finite sums are bounded because L and K are fixed.

Let $\psi _ { m }$ be any deterministic tensor with the same shape as the order-m signature. For each coordinate $a \in [ d ]$ , the a-th coordinate of $\mathcal { W } _ { \alpha } ^ { y ; m , k } \cdot \psi _ { m }$ is a weighted sum of entries of ${ \mathcal W } _ { \alpha } ^ { y ; m , k }$ . By the above moment estimate and Cauchy’s inequality,

$$
\begin{array} { r } { \mathbb { E } \left\| \left[ \mathcal { W } _ { \alpha } ^ { y ; m , k } \cdot \psi _ { m } \right] _ { a } \right\| ^ { 2 } \lesssim d ^ { - 2 k \gamma } \| \psi _ { m } \| _ { F } ^ { 2 } . } \end{array}
$$

Summing over $a \in [ d ]$ gives

$$
\begin{array} { r } { \left( \mathbb { E } \left\| \mathcal { W } _ { \alpha } ^ { y ; m , k } \cdot \psi _ { m } \right\| _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } \lesssim d ^ { 1 / 2 - k \gamma } \| \psi _ { m } \| _ { F } . } \end{array}
$$

Since $k \geq m + 1$ , the largest contribution at initialization comes from the lowest-degree term $k = m + 1$ , while all higher-degree terms contain additional factors of size $d ^ { - \gamma }$ . Hence

$$
\begin{array} { r } { \left( \mathbb { E } \left\| \mathscr { A } _ { \alpha , m } \cdot \psi _ { m } \right\| _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } \lesssim d ^ { 1 / 2 - ( m + 1 ) \gamma } \| \psi _ { m } \| _ { F } . } \end{array}
$$

For $m = 0$ , the leading term is the degree-one term:

$$
\begin{array} { r } { A _ { \alpha , 0 } ^ { \mathrm { l e a d } } = C _ { 1 } { \mathcal W } _ { \alpha } ^ { y ; 0 , 1 } . } \end{array}
$$

For any deterministic vector $\psi _ { 0 }$ , this term satisfies

$$
\left[ \mathcal { W } _ { \alpha } ^ { y ; 0 , 1 } \cdot \psi _ { 0 } \right] _ { a } = \sum _ { \ell \in \mathcal { V } } \psi _ { 0 } ( \ell ) \pmb { w } _ { \ell } ^ { U } ( a ) .
$$

Since the unembedding coordinates are independent with variance of order $d ^ { - 2 \gamma }$ , we have

$$
\mathbb { E } \left\| \left[ \mathscr { W } _ { \alpha } ^ { y ; 0 , 1 } \cdot \psi _ { 0 } \right] _ { a } \right\| ^ { 2 } \asymp d ^ { - 2 \gamma } \| \psi _ { 0 } \| _ { F } ^ { 2 } .
$$

After summing over $a \in [ d ]$ , this yields

$$
\left( \mathbb { E } \left\| \boldsymbol { A } _ { \alpha , 0 } \cdot \boldsymbol { \psi } _ { 0 } \right\| _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } \asymp d ^ { 1 / 2 - \gamma } \| \boldsymbol { \psi } _ { 0 } \| _ { F } ,
$$

where the higher-degree terms in $\mathcal { A } _ { \alpha , 0 }$ are smaller by additional powers of $d ^ { - \gamma } .$

Now take $\psi _ { m } = \varphi _ { \alpha } ^ { y ; m }$ . For every fixed $m \geq 1$ , the ratio between the root-mean-square scales of the order-m and order-0 contractions is bounded by

$$
\frac { \Big ( \mathbb { E } \left\| \boldsymbol { A } _ { \alpha , m } \cdot \boldsymbol { \varphi } _ { \alpha } ^ { y ; m } \right\| _ { 2 } ^ { 2 } \Big ) ^ { 1 / 2 } } { \Big ( \mathbb { E } \left\| \boldsymbol { A } _ { \alpha , 0 } \cdot \boldsymbol { \varphi } _ { \alpha } ^ { y ; 0 } \right\| _ { 2 } ^ { 2 } \Big ) ^ { 1 / 2 } } \lesssim d ^ { - m \gamma } \frac { \| \boldsymbol { \varphi } _ { \alpha } ^ { y ; m } \| _ { F } } { \| \boldsymbol { \varphi } _ { \alpha } ^ { y ; 0 } \| _ { F } } .
$$

By Lemma 1,

$$
\frac { \| \varphi _ { \alpha } ^ { y ; m } \| _ { F } } { \| \varphi _ { \alpha } ^ { y ; 0 } \| _ { F } } \le 1 .
$$

The factor $\binom { L - 1 } { m }$ contained in $\mathcal { A } _ { \alpha , m }$ is independent of d because L is fixed. Therefore,

$$
\frac { \left( \mathbb E \left\| A _ { \alpha , m } \cdot \varphi _ { \alpha } ^ { y ; m } \right\| _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } } { \left( \mathbb E \left\| A _ { \alpha , 0 } \cdot \varphi _ { \alpha } ^ { y ; 0 } \right\| _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } } \overset { < } { \sim } d ^ { - m \gamma } = o ( 1 ) .
$$

Thus every fixed higher-order label-signature contraction is asymptotically smaller than the zero-order contraction at initialization.

## B.4 Proof of Proposition 2

For each signature order m, the label-driven contribution can be decomposed according to the polynomial degree:

$$
\mathcal { A } _ { \alpha , m } ( t ) \cdot \varphi _ { \alpha } ^ { y ; m } = \sum _ { k = m + 1 } ^ { K } T _ { \alpha , m , k } ( t ) , \qquad T _ { \alpha , m , k } ( t ) : = k C _ { k } \mathcal { W } _ { \alpha } ^ { y ; m , k } ( t ) \cdot \varphi _ { \alpha } ^ { y ; m } .
$$

We write

$$
\begin{array} { r } { { \pmb w } _ { \beta } ^ { E } ( t ) = s _ { E } ( t ) { \pmb { \bar { w } } } _ { \beta } ^ { E } ( t ) , \qquad { \pmb w } _ { \nu } ^ { U } ( t ) = s _ { U } ( t ) { \pmb { \bar { w } } } _ { \nu } ^ { U } ( t ) . } \end{array}
$$

The degree-k term contains one unembedding factor and $k - 1$ embedding factors. Hence, under the assumed non-degeneracy of normalized contractions, its contracted norm has the scale

$$
\begin{array} { r } { \| T _ { \alpha , m , k } ( t ) \| _ { 2 } \asymp | k C _ { k } | \widetilde { B } _ { m } ( k ) s _ { U } ( t ) s _ { E } ( t ) ^ { k - 1 } \| \varphi _ { \alpha } ^ { y ; m } \| _ { F } , } \end{array}
$$

where the combinatorial coeficient is

$$
B _ { m } ( k ) : = \sum _ { \stackrel { h > 0 , ~ n _ { 1 } , \ldots , n _ { m } \geq 1 } { h + \sum _ { r = 1 } ^ { m } n _ { r } = k - 1 } } \frac { ( k - 1 ) ! } { h ! \prod _ { r = 1 } ^ { m } n _ { r } ! } , \qquad \widetilde { B } _ { m } ( k ) : = \binom { L - 1 } { m } B _ { m } ( k ) .
$$

Here $\widetilde { B } _ { m } ( \boldsymbol { k } )$ is independent of t and incorporates the deterministic position-averaging factor induced by the normalized signature definition.

We first compare diferent degrees for a fixed m. For any $k < K$ , we have

$$
\frac { \| T _ { \alpha , m , k } ( t ) \| _ { 2 } } { \| T _ { \alpha , m , K } ( t ) \| _ { 2 } } \lesssim \frac { | k C _ { k } | \widetilde B _ { m } ( k ) } { | K C _ { K } | \widetilde B _ { m } ( K ) } s _ { E } ( t ) ^ { k - K } .
$$

Since $k - K < 0 .$ , this ratio converges to zero when $s _ { E } ( t )$ becomes suficiently large. Therefore, for every fixed $m$ , the highest-degree term $T _ { \alpha , m , K } ( t )$ dominates all lower-degree terms. Because there are only finitely many $m = 0 , \ldots , L - 1$ , there exists a common threshold S<sub>∗</sub> such that whenever $s _ { E } ( t ) \geq S _ { * }$

$$
\| A _ { \alpha , m } ( t ) \cdot \varphi _ { \alpha } ^ { y ; m } \| _ { 2 } \asymp \| T _ { \alpha , m , K } ( t ) \| _ { 2 } , \qquad 0 \leq m \leq L - 1 .
$$

Since K and $L$ are fixed, the coeficients $| K C _ { K } | \widetilde { B } _ { m } ( K )$ are independent of t. Together with the uniform non-degeneracy of the normalized contractions and signatures, this gives

$$
\| A _ { \alpha , m } ( t ) \cdot \varphi _ { \alpha } ^ { y ; m } \| _ { 2 } \asymp s _ { U } ( t ) s _ { E } ( t ) ^ { K - 1 } , \qquad 0 \leq m \leq L - 1 .
$$

Hence all signature orders have the same late-stage scaling in the parameter magnitudes. Unlike at initialization, their hierarchy is no longer determined by diferent powers of $s _ { E } ( t )$

It remains only to compare the order-dependent prefactors of these same-scale terms. At degree K, their relative magnitudes are governed by the combinatorial coeficient $\widetilde { B } _ { m } ( K )$ together with the normalized contractions and signature factors. Since the factor $\binom { L - 1 } { m }$ is independent of K, the exponential dependence on K is determined by $B _ { m } ( K )$ , which satisfies

$$
B _ { m } ( K ) = ( m + 1 ) ! \left\{ K - 1 \atop { m + 1 } \right\} ,
$$

where $\textstyle { \left\{ { \begin{array} { l } { n } \\ { r } \end{array} } \right\} }$ is the Stirling number of the second kind. For fixed m and large K,

$$
B _ { m } ( K ) \sim ( m + 1 ) ^ { K - 1 } .
$$

Therefore, for any $m < L - 1$

$$
\frac { \widetilde { B } _ { L - 1 } ( K ) } { \widetilde { B } _ { m } ( K ) } \sim \frac { 1 } { \binom { L - 1 } { m } } \left( \frac { L } { m + 1 } \right) ^ { K - 1 } \to \infty , \qquad K \to \infty .
$$

Since the label signatures and normalized contractions are uniformly non-degenerate, choosing K suficiently large gives

$$
\| T _ { \alpha , L - 1 , K } ( t ) \| _ { 2 } \geq C \| T _ { \alpha , m , K } ( t ) \| _ { 2 } , \qquad 0 \leq m < L - 1 ,
$$

for a suficiently large constant $C > 1$

Having established that all signature orders share the same late-stage scale, we now compare their prefactors. When $s _ { E } ( t ) \geq S _ { * }$ and K is suficiently large, the highest-order prefactor is the largest. Therefore, the full highest-order contribution is the largest among the same-scale signature-order contributions:

$$
\begin{array} { r } { \left\| \boldsymbol A _ { \alpha , L - 1 } ( t ) \cdot \boldsymbol \varphi _ { \alpha } ^ { y ; L - 1 } \right\| _ { 2 } \geq \| \boldsymbol A _ { \alpha , m } ( t ) \cdot \boldsymbol \varphi _ { \alpha } ^ { y ; m } \| _ { 2 } , \qquad 0 \leq m < L - 1 . } \end{array}
$$

Therefore, once $s _ { E } ( t ) \geq S _ { * }$ and K is suficiently large, the order- $( L - 1 )$ term is the largest among all signature contributions. This completes the proof.

## B.5 Proof of Theorem 3

We first compute the value-path contribution.

$$
\frac { d { w _ { \alpha } ^ { E } } } { d t } \bigg | _ { \mathrm { v a l u e } } ^ { y } = \mathbb { E } \left[ \sum _ { s = 1 } ^ { L } a _ { s } \mathbf { 1 } \{ x _ { s } = \alpha \} \boldsymbol { W } ^ { O V , T } \boldsymbol { W } ^ { U , T } \boldsymbol { e } _ { y } \right] .\tag{66}
$$

Define the total attention mass received by token α as

$$
A _ { \alpha } ( X ) : = \sum _ { s = 1 } ^ { L } a _ { s } \mathbf { 1 } \{ x _ { s } = \alpha \} .
$$

Then we have

$$
\frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } \bigg | _ { \mathrm { v a l u e } } ^ { y } = L r _ { \alpha } { \pmb W } ^ { \prime T } \mathbb { E } \left[ a _ { I } { \pmb e } _ { y } \mid \alpha \right] .
$$

We recall the label signature

$$
\varphi _ { \alpha } ^ { y ; 0 } ( \nu ) : = \mathbb { P } ( y = \nu \mid \alpha ) ,
$$

and the attention-weighting vector

$$
\psi _ { \alpha } ^ { y } ( \nu ) : = \mathbb { E } [ a _ { I } \ | \ y = \nu , \alpha ] .
$$

Then the attention-weighted label distribution satisfies

$$
\mathbb { E } [ a _ { I } \pmb { e } _ { y } \mid \alpha ] = \varphi _ { \alpha } ^ { y ; 0 } \odot \psi _ { \alpha } ^ { y } .
$$

Therefore,

$$
\frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } \bigg | _ { \mathrm { v a l u e } } ^ { y } = L r _ { \alpha } { \pmb W } ^ { \prime T } ( \varphi _ { \alpha } ^ { y ; 0 } \odot \psi _ { \alpha } ^ { y } ) .
$$

This is the first signature term. It has the same form as a label-signature update, except that the label signature is reweighted by the attention mass assigned to token α.

We next compute the contribution from the derivative of the attention scores. Define the attention-weighted value average by

$$
\bar { \pmb w } _ { X } ^ { E } : = \sum _ { s = 1 } ^ { L } a _ { s } { \pmb w } _ { x _ { s } } ^ { E } .
$$

Then

$$
\pmb { g } _ { \mathrm { a t t n } } ( \pmb { X } ) = \pmb { W } ^ { O V } \bar { \pmb { w } } _ { X } ^ { E } .
$$

The softmax derivative satisfies

$$
\frac { \partial a _ { s } } { \partial z _ { u } } = a _ { s } ( { \bf 1 } \{ s = u \} - a _ { u } ) .
$$

Therefore,

$$
\begin{array} { c } { \displaystyle \frac { \partial \bar { \boldsymbol { w } } _ { X } ^ { E } } { \partial z _ { u } } = \sum _ { s = 1 } ^ { L } \frac { \partial a _ { s } } { \partial z _ { u } } \boldsymbol { w } _ { x _ { s } } ^ { E } } \\ { \displaystyle = \sum _ { s = 1 } ^ { L } a _ { s } ( \mathbf { 1 } \{ s = u \} - a _ { u } ) \boldsymbol { w } _ { x _ { s } } ^ { E } . } \end{array}\tag{67}
$$

The score-path contribution to the embedding gradient is therefore

$$
\left. \frac { d w _ { \alpha } ^ { E } } { d t } \right| _ { \mathrm { s c o r e } } ^ { y } = \mathbb { E } \left[ \sum _ { u = 1 } ^ { L } \left. W ^ { O V , T } W ^ { U , T } e _ { y } , \sum _ { s = 1 } ^ { L } a _ { s } ( \mathbf { 1 } \{ s = u \} - a _ { u } ) w _ { x _ { s } } ^ { E } \right. \frac { \partial z _ { u } } { \partial w _ { \alpha } ^ { E } } \right] .
$$

It remains to compute $\partial z _ { u } / \partial w _ { \alpha } ^ { E }$ . the embedding $\pmb { w } _ { \alpha } ^ { E }$ afects $z _ { u }$ in two ways. If $x _ { L } = \alpha$ 2 it afects the last query vector. If $x _ { u } = \alpha$ , it afects the key vector at position u. Therefore,

$$
\frac { \partial z _ { u } } { \partial w _ { \alpha } ^ { E } } = \frac { 1 } { \sqrt { d } } \mathbf { 1 } \{ x _ { u } = \alpha \} W ^ { Q K } w _ { x _ { L } } ^ { E } + \frac { 1 } { \sqrt { d } } \mathbf { 1 } \{ x _ { L } = \alpha \} W ^ { Q K , T } w _ { x _ { u } } ^ { E } .
$$

Substituting this into the score-path contribution gives

$$
\begin{array} { l } { \displaystyle \frac { d w _ { \alpha } ^ { E } } { d t } \bigg | _ { \mathrm { s c o r e } } ^ { y } = \frac { 1 } { \sqrt { d } } \mathbb { E } \left[ \sum _ { u = 1 } ^ { L } \bigg \langle W ^ { O V , T } W ^ { U , T } e _ { y } , \sum _ { s = 1 } ^ { L } a _ { s } ( \mathbf { 1 } \{ s = u \} - a _ { u } ) w _ { x _ { s } } ^ { E } \bigg \rangle \mathbf { 1 } \{ x _ { u } = \alpha \} W ^ { Q K } w _ { x _ { L } } ^ { E } \right] } \\ { \displaystyle \qquad + \frac { 1 } { \sqrt { d } } \mathbb { E } \left[ \sum _ { u = 1 } ^ { L } \bigg \langle W ^ { O V , T } W ^ { U , T } e _ { y } , \sum _ { s = 1 } ^ { L } a _ { s } ( \mathbf { 1 } \{ s = u \} - a _ { u } ) w _ { x _ { s } } ^ { E } \bigg \rangle \mathbf { 1 } \{ x _ { L } = \alpha \} W ^ { Q K , T } w _ { x _ { u } } ^ { E } \right] . } \end{array}\tag{68}
$$

We call the first term the key-score contribution and the second term the query-score contribution:

$$
\left. \frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } \right| _ { \mathrm { s c o r e } } ^ { y } = \left. \frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } \right| _ { \mathrm { k e y } } ^ { y } + \left. \frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } \right| _ { \mathrm { q u e r y } } ^ { y } .
$$

Consider the term $\left. \frac { d { \pmb w } _ { \alpha } ^ { E } } { d t } \right| _ { \mathrm { k e y } } ^ { y }$ , we have that

$$
\begin{array} { r l r } {  { \frac { d w _ { \alpha } ^ { E } } { d t } \bigg | _ { \mathrm { l e y } } ^ { y } = \frac { 1 } { \sqrt { d } } \mathbb { E } [ \sum _ { u = 1 } ^ { L } W ^ { Q K } w _ { x _ { L } } ^ { E } w _ { \nu } ^ { U , T } W ^ { O V } ( \sum _ { s = 1 } ^ { L } a _ { s } ( \mathbf { 1 } \{ s = u \} - a _ { u } ) w _ { x _ { s } } ^ { E } ) \mathbf { 1 } \{ x _ { u } = \alpha \} ] } } \\ & { } & { = \frac { 1 } { \sqrt { d } } \mathbb { E } [ W ^ { Q K } w _ { x _ { L } } ^ { E } w _ { \nu } ^ { U , T } W ^ { O V } ( \sum _ { u = 1 } ^ { L } ( \sum _ { s = 1 } ^ { L } a _ { s } ( \mathbf { 1 } \{ s = u \} - a _ { u } ) w _ { x _ { s } } ^ { E } ) \mathbf { 1 } \{ x _ { u } = \alpha \} ) ] . } \end{array}
$$

It’s noted that

$$
\begin{array} { l } { { \displaystyle \sum _ { u = 1 } ^ { L } \left( \displaystyle \sum _ { s = 1 } ^ { L } a _ { s } ( \mathbf { 1 } \{ s = u \} - a _ { w } ) w _ { x _ { s } } ^ { E } \right) \mathbf { 1 } \{ x _ { u } = \alpha \} } } \\ { { = \displaystyle \sum _ { u = 1 } ^ { L } \sum _ { s = 1 } ^ { L } \mathbf { 1 } \{ x _ { u } = \alpha \} \mathbf { 1 } \{ s = u \} a _ { s } w _ { x _ { s } } ^ { E } - \displaystyle \sum _ { u = 1 } ^ { L } \sum _ { s = 1 } ^ { L } \mathbf { 1 } \{ x _ { u } = \alpha \} a _ { u } a _ { s } w _ { x _ { s } } ^ { E } } } \\ { { = \displaystyle \sum _ { u = 1 } ^ { L } \mathbf { 1 } \{ x _ { u } = \alpha \} a _ { u } w _ { \alpha } ^ { E } - \displaystyle \sum _ { u = 1 } ^ { L } \mathbf { 1 } \{ x _ { u } = \alpha \} a _ { u } \displaystyle \sum _ { s = 1 } ^ { L } a _ { s } w _ { x _ { s } } ^ { E } } } \\ { { = \displaystyle \left( \displaystyle \sum _ { u = 1 } ^ { L } \mathbf { 1 } \{ x _ { u } = \alpha \} a _ { u } \right) \left( w _ { \alpha } ^ { E } - w _ { x } ^ { E } \right) . } } \end{array}
$$

Then we have

$$
\begin{array} { r l } & { \displaystyle \frac { d w _ { \alpha } ^ { E } } { d t } \bigg | _ { \mathrm { k e y } } ^ { y } = \frac { 1 } { \sqrt { d } } \mathbb { E } \left[ W ^ { Q K } w _ { x _ { L } } ^ { E } \left( \sum _ { u = 1 } ^ { L } \mathbf { 1 } \{ x _ { u } = \alpha \} a _ { u } \right) w _ { \nu } ^ { U , T } W ^ { O V } \left( w _ { \alpha } ^ { E } - \bar { w } _ { X } ^ { E } \right) \right] } \\ & { \quad \quad \quad = \frac { L r _ { \alpha } } { \sqrt { d } } W ^ { Q K } \mathbb { E } \left[ a _ { I } w _ { x _ { L } } ^ { E } w _ { \nu } ^ { U , T } W ^ { O V } \left( w _ { \alpha } ^ { E } - \bar { w } _ { X } ^ { E } \right) \mid \alpha \right] . } \end{array}
$$

Conditioned on $\alpha ,$ this term depends on the joint distribution of the label y and the last token $x _ { L }$ . Define the label-last-token signature

$$
\varphi _ { \alpha } ^ { y , x _ { L } } ( \nu , \beta ) : = \mathbb { P } ( y = \nu , x _ { L } = \beta \mid \alpha ) , \quad \varphi _ { \alpha } ^ { y , x _ { L } , x _ { s } } ( \nu , \beta _ { L } , \beta _ { s } ) : = \mathbb { P } ( y = \nu , x _ { L } = \beta _ { L } , x _ { s } = \beta _ { s } \mid \alpha ) .
$$

For each pair $( \nu , \beta )$ , define the key-side attention-derivative weight

$$
\begin{array} { r l } & { \psi _ { \alpha } ^ { y , x _ { L } , ( 1 ) } ( \nu , \beta ) : = \operatorname { \mathbb { E } } \left[ a _ { I } \mid y = \nu , x _ { L } = \beta , \alpha \right] . } \\ & { \psi _ { \alpha } ^ { y , x _ { L } , x _ { s } , ( 1 ) } ( \nu , \beta _ { L } , \beta _ { s } ) : = \operatorname { \mathbb { E } } \left[ a _ { I } a _ { s } \mid y = \nu , x _ { L } = \beta _ { L } , x _ { s } = \beta _ { s } , \alpha \right] , \qquad s = 1 , 2 , \cdots , L . } \end{array}
$$

and $\widetilde { W } ^ { \mathrm { k e y } } \in \left( \mathbb { R } ^ { d } \right) ^ { V \times V } , \widetilde { W } ^ { \mathrm { k e y } , s } \in \left( \mathbb { R } ^ { d } \right) ^ { V \times V \times V }$ , where

$$
\begin{array} { r } { \widetilde { W } ^ { \mathrm { k e y } } \left( \nu , \beta _ { L } \right) = \pmb { w } _ { \beta _ { L } } ^ { E } \pmb { w } _ { \nu } ^ { U , T } \pmb { W } ^ { O V } \pmb { w } _ { \alpha } ^ { E } , \qquad \widetilde { W } ^ { \mathrm { k e y , s } } \left( \nu , \beta _ { L } , \beta _ { s } \right) = \pmb { w } _ { \beta _ { L } } ^ { E } \pmb { w } _ { \nu } ^ { U , T } \pmb { W } ^ { O V } \pmb { w } _ { \beta _ { s } } ^ { E } . } \end{array}
$$

Then

$$
\left. \frac { d w _ { \alpha } ^ { E } } { d t } \right| _ { \mathrm { k e y } } ^ { y } = \frac { L r _ { \alpha } } { \sqrt { d } } W ^ { Q K } \left( \widetilde { W } ^ { \mathrm { k e y } } \cdot \left( \psi _ { \alpha } ^ { y , x _ { L } , ( 1 ) } \odot \varphi _ { \alpha } ^ { y , x _ { L } } \right) - \sum _ { s = 1 } ^ { L } \widetilde { W } ^ { \mathrm { k e y } , s } \cdot \left( \psi _ { \alpha } ^ { y , x _ { L } , x _ { s } , ( 1 ) } \odot \varphi _ { \alpha } ^ { y , x _ { L } , x _ { s } } \right) \right) .
$$

We now rewrite the query-score term in signature form. The label-driven query-score term is

$$
\frac { d w _ { \alpha } ^ { E } } { d t } \bigg | _ { \mathrm { q u e r y } } ^ { y } : = \frac { 1 } { \sqrt { d } } \mathbb { E } \left[ { \bf 1 } \{ x _ { L } = \alpha \} { \cal W } ^ { Q K , T } \sum _ { u = 1 } ^ { L } w _ { x _ { u } } ^ { E } w _ { y } ^ { U , T } { \cal W } ^ { O V } \sum _ { s = 1 } ^ { L } a _ { s } ( { \bf 1 } \{ s = u \} - a _ { u } ) w _ { x _ { s } } ^ { E } \right] .
$$

It’s noted that

$$
\begin{array} { l } { \displaystyle \sum _ { u = 1 } ^ { L } w _ { x u } ^ { E } w _ { y } ^ { U , T } W ^ { O V } \sum _ { s = 1 } ^ { L } a _ { s } ( \mathbf { 1 } \{ s = u \} - a _ { u } ) w _ { x s } ^ { E } } \\ { = \displaystyle \sum _ { u = 1 } ^ { L } w _ { x u } ^ { E } w _ { y } ^ { U , T } W ^ { O V } a _ { u } w _ { x u } ^ { E } - \sum _ { u = 1 } ^ { L } w _ { x u } ^ { E } w _ { y } ^ { U , T } W ^ { O V } a _ { u } \sum _ { s = 1 } ^ { L } a _ { s } w _ { x s } ^ { E } } \\ { = \displaystyle \sum _ { u = 1 } ^ { L } a _ { u } w _ { x u } ^ { E } w _ { y } ^ { U , T } W ^ { O V } w _ { x _ { u } } ^ { E } - \left( \sum _ { u = 1 } ^ { L } a _ { u } w _ { x u } ^ { E } \right) w _ { \nu } ^ { U , T } W ^ { O V } \left( \sum _ { u = 1 } ^ { L } a _ { u } w _ { x u } ^ { E } \right) . } \end{array}
$$

Let φ<sup>y;xs</sup><sub>α,xL</sub>(ν, β<sub>s</sub>) = P (y = ν, x<sub>s</sub> = β<sub>s</sub> | x<sub>L</sub> = α) , φ<sup>y;xs,xu</sup><sub>α,xL</sub> (ν, β<sub>s</sub>, β<sub>u</sub>) = P (y = ν, x<sub>s</sub> = β<sub>s</sub>, x<sub>u</sub> = β<sub>u</sub> | x<sub>L</sub> = α) and define that

$$
\begin{array} { r l } & { \psi _ { \alpha , x _ { L } } ^ { y ,  x _ { s } } ( \nu , \beta _ { s } ) = \mathbb { E } [ a _ { s } \middle | y = \nu , x _ { s } = \beta _ { s } , x _ { L } = \alpha ] , } \\ & { \psi _ { \alpha , x _ { L } } ^ { y ,  x _ { s } , x _ { u } } ( \nu , \beta _ { s } , \beta _ { u } ) = \mathbb { E } [ a _ { s } a _ { u } \middle | y = \nu , x _ { s } = \beta _ { s } , x _ { u } = \beta _ { u } , x _ { L } = \alpha ] , } \end{array}
$$

and

$$
\widetilde { W } _ { \alpha } ^ { \mathrm { q u e r y } , u } \left( \nu , \beta _ { u } \right) = w _ { \beta _ { u } } ^ { E } w _ { \nu } ^ { U , T } W ^ { O V } w _ { \beta _ { u } } ^ { E } , \qquad \widetilde { W } _ { \alpha } ^ { \mathrm { Q u e r y } , u , s } \left( \nu , \beta _ { u } , \beta _ { s } \right) = w _ { \beta _ { u } } ^ { E } w _ { \nu } ^ { U , T } W ^ { O V } w _ { \beta _ { s } } ^ { E } .
$$

Then we have

$$
\begin{array} { l } { \displaystyle \frac { d w _ { \alpha } ^ { E } } { d t } \Bigg | _ { \mathrm { q u e r y } } ^ { y } = \frac { r _ { \alpha } ^ { x _ { L } } } { \sqrt { d } } W ^ { Q K } } \\ { \displaystyle ( \sum _ { u = 1 } ^ { L } \widetilde { W } _ { \alpha } ^ { \mathrm { q u e r y } , u } \cdot ( \varphi _ { \alpha , x _ { L } } ^ { y ; x _ { s } } \odot \psi _ { \alpha , x _ { L } } ^ { y ,  x _ { s } } ) - \sum _ { u , s = 1 } ^ { L } \widetilde { W } _ { \alpha } ^ { \mathrm { q u e r y } , u , s } \cdot ( \varphi _ { \alpha , x _ { L } } ^ { y ; x _ { s } , x _ { u } } \odot \psi _ { \alpha , x _ { L } } ^ { y ,  x _ { s } , x _ { u } } ) ) . } \end{array}
$$

## Appendix C. Multi-stage embedding evolution in length-three sequences.

The experiments above consider sequence-length-two tasks, where only the zeroth- and firstorder signatures are accessible. To examine the late-stage regime with multiple higher-order contributions, we now consider a sequence-length-three task in which both first- and secondorder signatures can influence the embedding dynamics. Proposition 2 suggests that, once the initialization-induced suppression is lifted, these accessible orders have the same leading parameter scale and can therefore contribute jointly, while the second-order contribution is expected to be the largest among them.

Task 4. Let A and X be finite sets of integers, where A is the set of anchor tokens. We consider input sequences

$$
X = ( a _ { 1 } , a _ { 2 } , x ) , \qquad a _ { 1 } , a _ { 2 } \in \mathcal { A } , \quad x \in \mathcal { X } ,
$$

with input space $\mathcal { D } = \mathcal { A } \times \mathcal { A } \times \mathcal { X }$ . The target label is defined as

$$
y = F ( a _ { 1 } , a _ { 2 } , x ) = ( a _ { 1 } + a _ { 2 } + x ) { \bmod { | x | } } .
$$

The anchor tokens and the contextual token are drawn from disjoint token ranges. For any fixed anchor token, marginalizing over the remaining inputs produces the same label distribution. Consequently, all anchor tokens have identical zeroth-order label signatures. In contrast, the joint token–label statistics retained by the first- and second-order signatures depend on the anchor identity and can therefore distinguish diferent anchor tokens.

Figure 12 shows the evolution of the anchor-token embedding geometry. At the early stage, the cosine similarities are nearly uniform and the PCA projection collapses into a compact cluster, consistent with the identical zeroth-order signatures. At the intermediate stage, a banded similarity pattern emerges and the PCA projection forms a smooth onedimensional curve, indicating that contextual signature efects have begun to appear. At the late stage, the similarity matrix becomes less purely banded and the PCA projection develops a more complex geometry, consistent with the joint influence of the first- and second-order signatures rather than the replacement of one order by the other.

To quantify this evolution, let $S ^ { ( m ) }$ denote the pairwise similarity structure induced by the mth-order signatures. We compare the of-diagonal entries of $S ^ { ( m ) }$ with those of the embedding cosine-similarity matrix. The correlation with $S ^ { ( 0 ) }$ briefly approaches one at the beginning of training, but then becomes strongly negative and gradually moves back toward zero. In contrast, the correlations with both $S ^ { ( 1 ) }$ and $S ^ { ( 2 ) }$ rapidly approach one and remain high throughout training. Thus, the late-stage geometry simultaneously reflects both accessible higher-order signatures. Moreover, $S ^ { ( 2 ) }$ provides a consistently stronger fit than $S ^ { ( 1 ) }$ in the later stage, matching the theoretical picture that the same-scale higher-order contributions coexist while the highest-order contribution is the largest. The zeroth-order geometry is therefore transient, whereas the first- and second-order statistics jointly shape the later embedding space.

## Appendix D. Diferent Seeds, Diferent Embedding

Even with the dataset, architecture, and training configuration fixed, changing only the random seed can lead to visibly diferent final embedding geometries. We observe this

![](images/de9a9ecb057bcbd2e0792a372ce09d19fd3635e5f4714044bfe38318810e76a7.jpg)  
Figure 12: The top panel shows the training loss and accuracy over 1,000 epochs, with dashed vertical lines marking the early, middle, and late checkpoints at epochs 10, 100, and 1,000. The three lower columns show the anchor-token cosinesimilarity matrices and their two-dimensional PCA projections at the corresponding checkpoints. Token indices are displayed directly in the PCA projections.

phenomenon in both the addition and modular-addition tasks. Rather than being arbitrary, these diferences can be explained by the frequency components of the higher-order signatures.

For each seed, we compare the pairwise similarity structure of the final embeddings with that induced by each Fourier component of the corresponding signature. Specifically, we identify

$$
k ^ { * } = \arg \operatorname* { m a x } _ { k > 1 0 } \mathrm { C o r r } \bigl ( S _ { \mathrm { e m b } } , S _ { \mathrm { s i g } } ^ { ( k ) } \bigr ) ,
$$

![](images/00791ad98a22e6fb3610a7f4051c65c07b94db047de256becf64a049ee31fe87.jpg)  
Figure 13: Correlation between the of-diagonal entries of the anchor-token embedding cosine-similarity matrix and the similarity matrices induced by the zeroth-, first-, and second-order signatures, denoted by $S ^ { ( 0 ) }$ , S<sup>(1)</sup>, and $S ^ { ( 2 ) }$ , respectively, over training epochs. The horizontal dashed line marks zero correlation.

where $S _ { \mathrm { e m b } }$ is the embedding similarity matrix and $S _ { \mathrm { s i g } } ^ { ( k ) }$ is the similarity matrix induced by frequency k. As shown in Figures 15 and 14, the selected frequency $k ^ { * }$ varies across seeds. Correspondingly, the PCA geometry of the embeddings closely matches the projection of the signature component at the selected frequency. Thus, diferent initializations lead the embeddings to align with diferent spectral components of the same higher-order signature, producing distinct but signature-governed geometries.

![](images/806ae6e62aac97b8f5753a989901cc6eb6af00106dc1a90d5abbb715240020ef.jpg)  
Figure 14: Results for ten random seeds at epoch 1,000. For each seed, the top panel shows the Pearson correlation between the embedding similarity structure and that induced by each Fourier frequency k of the first-order signature, with the red marker and dashed line indicating the frequency of maximum correlation. The middle and bottom panels show the two-dimensional PCA projections of the learned embeddings and the first-order signature component at the selected frequency, respectively. Token indices are displayed directly in the PCA projections.

![](images/1c5e890c6f0677b7888e330eb9a983ee97242456403cc82a59fc33a19600a976.jpg)  
Figure 15: Results for ten random seeds at epoch 1,000. For each seed, the top panel shows the Pearson correlation between the embedding similarity structure and that induced by each Fourier frequency k of the first-order signature, with the red marker and dashed line indicating the frequency of maximum correlation. The middle and bottom panels show the two-dimensional PCA projections of the learned embeddings and the first-order signature component at the selected frequency, respectively. Token indices are displayed directly in the PCA projections.

## References

Yoshua Bengio, R´ejean Ducharme, Pascal Vincent, and Christian Jauvin. A neural probabilistic language model. Journal of Machine Learning Research, 3:1137–1155, 2003.

Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. Enriching word vectors with subword information. Transactions of the Association for Computational Linguistics, 5:135–146, 2017.

Etienne Boursier, Loucas Pillaud-Vivien, and Nicolas Flammarion. Gradient flow dynamics of shallow ReLU networks for square loss and orthogonal inputs. In Advances in Neural Information Processing Systems, volume 35, 2022.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901, 2020.

Yongchao Chen, Harsh Jhamtani, and Srinagesh Sharma. Steering large language models between code execution and textual reasoning. In International Conference on Learning Representations, 2025.

Zheng-An Chen, Pengxiao Lin, Zhi-Qin John Xu, and Tao Luo. Focus and dilution: The multi-stage learning process of attention. In Forty-third International Conference on Machine Learning, 2026.

Kawin Ethayarajh. How contextual are contextualized word representations? comparing the geometry of BERT, ELMo, and GPT-2 embeddings. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pages 55–65. Association for Computational Linguistics, 2019.

John R. Firth. Papers in Linguistics 1934–1951. Oxford University Press, London, 1957.

Andrey Gromov. Grokking modular arithmetic. arXiv preprint arXiv:2301.02679, 2023.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645:633–638, 2025. doi: 10.1038/s41586-025-09422-z.

Wes Gurnee and Max Tegmark. Language models represent space and time. In International Conference on Learning Representations, 2024.

Chi Han, Jialiang Xu, Manling Li, Yi Fung, Chenkai Sun, Nan Jiang, Tarek Abdelzaher, and Heng Ji. Word embeddings are steers for language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, pages 16410–16430. Association for Computational Linguistics, 2024.

Zellig S. Harris. Distributional structure. Word, 10(2–3):146–162, 1954. doi: 10.1080/ 00437956.1954.11659520.

John Hewitt and Christopher D. Manning. A structural probe for finding syntax in word representations. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4129–4138. Association for Computational Linguistics, 2019.

Shawn Im, Changdae Oh, Zhen Fang, and Sharon Li. How do transformers learn to associate tokens: Gradient leading terms bring mechanistic interpretability. In International Conference on Learning Representations, 2026.

Arthur Jacot, Franck Gabriel, and Cl´ement Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In Advances in Neural Information Processing Systems, volume 31, 2018.

Alessandro Lenci. Distributional models of word meaning. Annual Review of Linguistics, 4:151–171, 2018. doi: 10.1146/annurev-linguistics-030514-125254.

Omer Levy and Yoav Goldberg. Neural word embedding as implicit matrix factorization. In Advances in Neural Information Processing Systems, volume 27, 2014.

Ziming Liu, Ouail Kitouni, Niklas Nolte, Eric J. Michaud, Max Tegmark, and Mike Williams. Towards understanding grokking: An efective theory of representation learning. In Advances in Neural Information Processing Systems, volume 35, 2022.

Tao Luo, Zhi-Qin John Xu, Zheng Ma, and Yaoyu Zhang. Phase diagram for two-layer ReLU neural networks at infinite-width limit. Journal of Machine Learning Research, 22 (71):1–47, 2021.

Neil Rohit Mallinar, Daniel Beaglehole, Libin Zhu, Adityanarayanan Radhakrishnan, Parthe Pandit, and Mikhail Belkin. Emergence in non-neural models: Grokking modular arithmetic via average gradient outer product. In International Conference on Machine Learning, 2025.

Tomas Mikolov, Kai Chen, Greg Corrado, and Jefrey Dean. Eficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781, 2013a.

Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S. Corrado, and Jefrey Dean. Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems, volume 26, 2013b.

Tomas Mikolov, Wen-tau Yih, and Geofrey Zweig. Linguistic regularities in continuous space word representations. In Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 746–751, Atlanta, Georgia, June 2013c. Association for Computational Linguistics.

Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability. In International Conference on Learning Representations, 2023.

OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Kiho Park, Yo Joong Choe, Yibo Jiang, and Victor Veitch. The geometry of categorical and hierarchical concepts in large language models. In International Conference on Learning Representations, 2025.

Jefrey Pennington, Richard Socher, and Christopher D. Manning. Glove: Global vectors for word representation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, pages 1532–1543. Association for Computational Linguistics, 2014.

Alec Radford, Jefrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. OpenAI Technical Report, 2019.

Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 5301–5310. PMLR, 09–15 Jun 2019.

Andrew M. Saxe, James L. McClelland, and Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks. In International Conference on Learning Representations, 2014.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth’ee Lacroix, Baptiste Rozi‘ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. LLaMA: Open and eficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Peter D. Turney and Patrick Pantel. From frequency to meaning: Vector space models of semantics. Journal of Artificial Intelligence Research, 37:141–188, 2010.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017.

Johannes von Oswald, Eyvind Niklasson, Ettore Randazzo, Jo˜ao Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In Proceedings of the 40th International Conference on Machine Learning. PMLR, 2023.

Zikai Xie. Order matters in hallucination: Reasoning order as benchmark and reflexive prompting for large-language-models. arXiv preprint arXiv:2408.05093, 2024.

Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tie-Yan Liu. On layer normalization in the transformer architecture. In Proceedings of the 37th International Conference on Machine Learning, pages 10524–10533. PMLR, 2020.

Zhi-Qin John Xu, Yaoyu Zhang, and Yanyang Xiao. Training behavior of deep neural network in frequency domain. In 26th International Conference on Neural Information Processing, volume 11953, pages 264–274, 2019.

Zhi-Qin John Xu, Yaoyu Zhang, Tao Luo, Yanyang Xiao, and Zheng Ma. Frequency principle: Fourier analysis sheds light on deep neural networks. Communications in Computational Physics, 28(5):1746–1767, 2020. doi: 10.4208/cicp.OA-2020-0085. URL https://doi.org/10.4208/cicp.OA-2020-0085.

Zhi-Qin John Xu, Yaoyu Zhang, and Tao Luo. Overview frequency principle/spectral bias in deep learning. Communications on Applied Mathematics and Computation, 7(3):827–864, 2024.

Zhi-Qin John Xu, Yaoyu Zhang, and Zhangchen Zhou. An overview of condensation phenomenon in deep learning, 2025. URL https://arxiv.org/abs/2504.09484.

Yadollah Yaghoobzadeh, Katharina Kann, T. J. Hazen, Eneko Agirre, and Hinrich Sch¨utze. Probing for semantic classes: Diagnosing the meaning content of word embeddings. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 5740–5753. Association for Computational Linguistics, July 2019. doi: 10.18653/ v1/P19-1574.

Haotong Yang, Yi Hu, Shijia Kang, Zhouchen Lin, and Muhan Zhang. Number cookbook: Number understanding of language models and how to improve it. arXiv preprint arXiv:2411.03766, 2024a.

Hongru Yang, Bhavya Kailkhura, Zhangyang Wang, and Yingbin Liang. Training dynamics of transformers to recognize word co-occurrence via gradient flow analysis. In Advances in Neural Information Processing Systems, volume 37, pages 46047–46117, 2024b.

Hongru Yang, Zhangyang Wang, Jason D. Lee, and Yingbin Liang. Transformers provably learn two-mixture of linear classification via gradient flow. In International Conference on Learning Representations, 2025.

Junjie Yao, Zhongwang Zhang, and Zhi-Qin John Xu. An analysis for reasoning bias of language models with small initialization. In International Conference on Machine Learning, pages 71834–71864. PMLR, 2025.

Zhongwang Zhang, Pengxiao Lin, Zhiwei Wang, Yaoyu Zhang, and Zhi-Qin John Xu. Initialization is critical to whether transformers fit composite functions by reasoning or memorizing. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

Zhongwang Zhang, Pengxiao Lin, Zhiwei Wang, Yaoyu Zhang, and Zhi-Qin John Xu. Complexity control facilitates reasoning-based compositional generalization in transformers. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

Hanxu Zhou, Qixuan Zhou, Tao Luo, Yaoyu Zhang, and Zhi-Qin John Xu. Towards understanding the condensation of neural networks at initial training. In Advances in Neural Information Processing Systems, volume 35, pages 2184–2196, 2022.