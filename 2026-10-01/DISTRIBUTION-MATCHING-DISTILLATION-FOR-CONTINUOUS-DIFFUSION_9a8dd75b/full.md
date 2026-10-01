# DISTRIBUTION MATCHING DISTILLATION FOR CONTINUOUS DIFFUSION LANGUAGE MODELS

Paul Le Van Kiem<sup>∗</sup> Inria, PSL Research University paul.le-van-kiem@inria.fr

Umut Simsekli Inria, PSL Research University umut.simsekli@inria.fr

Dario Shariatian<sup>∗</sup>   
Cohere   
dario.shariatian@cohere.com   
Alain Durmus   
CMAP, Ecole Polytechnique   
alain.durmus@polytechnique.edu

## ABSTRACT

Continuous diffusion language models generate all tokens in parallel, yet highquality generation can still require hundreds of network evaluations (NFEs). We study how distributional distillation can reduce this cost by exploiting the student’s probabilistic token outputs. Our unified formulation connects the student’s output parameterization to the resulting gradient estimators and yields two methods with the same student architecture and reverse-KL matching objective: Simplex-DMD uses continuous token relaxations and pathwise gradients, while Reinforce-DMD uses categorical sampling and REINFORCE with a learned density ratio. We develop both methods for multi-step generation and investigate the training and sampling choices associated with each parameterization. On OpenWebText, for sequences of 1,024 tokens, Simplex-DMD achieves a generative perplexity of 45.6 at a unigram entropy of 5.44 nats in just 4 NFEs, a 49% reduction relative to the strongest evaluated diffusion baseline at matched entropy and sampling budget. Reinforce-DMD improves the frontier at larger budgets, reaching a generative perplexity of 14.9 at an entropy of 5.00 nats with 256 NFEs, a 20% reduction under the same comparison protocol.

## 1 INTRODUCTION

Diffusion language models generate text either through diffusion directly in discrete token space (Austin et al., 2021; Lou et al., 2024; Sahoo et al., 2024) or through Gaussian diffusion in a continuous embedding space (Dieleman et al., 2022). Recent advances in continuous diffusion language models have shown that the latter approach is competitive with its discrete counterparts (Chen et al., 2026b; Yang et al., 2026; Hu et al., 2026). Although both approaches update all sequence positions in parallel, generating high-quality text can still require hundreds of sequential network evaluations. This sampling cost motivates diffusion distillation, which aims to reduce the number of evaluations while preserving generation quality, using a pretrained teacher to supervise a student to generate samples in fewer steps.

Two broad approaches differ in the teacher supervision they provide. Trajectory-based methods supervise student predictions or transitions along the teacher’s dynamics. For continuous data, these include consistency and flow map methods (Song et al., 2023; Boffi et al., 2025b), and for discrete data they have been developed for both discrete and continuous diffusion models (Deschenaux & Gulcehre, 2025; Hayakawa et al., 2025; Roos et al., 2026; Potaptchik et al., 2026; Lee et al., 2026).

Distributional methods instead use the teacher to guide the student toward the data distribution, without prescribing its generation trajectories. Developed for continuous data in both one-step and multi-step regimes (Luo et al., 2023; Yin et al., 2024; Salimans et al., 2024), these approaches have recently been explored for discrete data, like language, with discrete (Hoogeboom et al., 2026; Li et al., 2026; Zhu et al., 2025) and continuous diffusion models (Chen et al., 2026a).

![](images/a73906c6a48801b9e9ab4d56628a7d57922816aa2330bac8d7286dfd641f2d03.jpg)  
(a) Few-step budgets, matched to Simplex-DMD.

![](images/7dea7d61feec1148252ff7861e7942148933f9a15ff98497f876aadfbe5e1758.jpg)  
(b) Multi-step budgets, matched to Reinforce-DMD.  
Figure 1: Gen PPL on OpenWebText against number of function evaluations (NFE), at the matched unigram entropy $H _ { \mathrm { N F E } }$ (top axis), defined as in Table 2. Each method’s log Gen PPL is linearly interpolated at $\bar { H } _ { \mathrm { N F E } }$ between the two sampling temperatures whose entropies bracket it. LangFlow and MDLM are undistilled models; FMLM, ReDi and D-MMD are distillation methods. A method that does not reach $H _ { \mathrm { N F E } }$ is shown at the closest measured entropy, with its value in parentheses. The dashed purple line marks the Gen PPL of the data, with its entropy in parentheses.

We study distributional distillation for continuous diffusion models on discrete data (CDMd). Their denoising networks predict a probability distribution over clean tokens at each position, given noisy embeddings and a noise level. Retaining these outputs for the student leaves a natural choice: use the probability vectors directly as continuous token representations, or sample discrete tokens from them. In both cases, we embed and noise the student outputs and match their distribution to the corresponding noised data distribution under a reverse-KL objective. These two parameterizations yield different training procedures. Simplex Distribution Matching Distillation (Simplex-DMD) uses pathwise gradients through continuous token relaxations, with an auxiliary denoiser estimating the score of the noised student distribution, following Diff-Instruct and DMD (Luo et al., 2023; Yin et al., 2024). REINFORCE Distribution Matching Distillation (Reinforce-DMD) uses categorical samples and a REINFORCE estimator (Williams, 1992), with a discriminator estimating the density ratio between noised student and data distributions. We extend both methods to multi-step generation and study their training behavior and quality-diversity tradeoffs across sampling budgets.

## Our contributions are as follows:

• We introduce a unified framework for distributional distillation that compares noised student and data distributions through a generic discrepancy. This formulation recovers objectives used in prior work as particular choices of discrepancy and extends to multi-step generation.

• We specialize this framework to reverse-KL matching and derive two distillation methods: Simplex-DMD and Reinforce-DMD. Together, they provide continuous-relaxation and categorical-sampling approaches to training the same student architecture for multi-step generation.

• We distill the same LangFlow teacher (Chen et al., 2026b) (170M parameters) with both methods and evaluate generation of 1,024-token sequences on OpenWebText using generative perplexity– entropy frontiers (Pynadath et al., 2026). The methods improve on the evaluated baselines at complementary sampling budgets: Simplex-DMD at 2–16 network evaluations and Reinforce-DMD at 256 evaluations, matching the best baseline at 128 (Figure 1). At four evaluations and unigram entropy 5.44, Simplex-DMD reduces Gen PPL from 90.2 for ReDi to $4 5 . 6 ;$ at 256 evaluations and entropy 5.00, Reinforce-DMD reduces it from 18.6 for D-MMD to 14.9.

Notation. Let K denote the vocabulary size, let $\mathsf { V } = \{ e ^ { 1 } , \ldots , e ^ { K } \}$ be the canonical basis of $\mathbb { R } ^ { K }$ and $\mathsf { X } = \mathsf { V } ^ { L }$ the space of sequences of length L. Capital letters denote random variables and lower-case letters their realizations. We represent a discrete sample by a matrix $\mathbf { x } \in \mathbb { R } ^ { L \times K }$ with rows $\mathbf { x } ^ { \ell } \in \mathsf { V }$ and entries ${ \bf x } ^ { \ell k }$ . We write $\Delta _ { K } = \{ u \in \mathbb { R } ^ { K } : u ^ { k } \geq 0 , \sum _ { k = 1 } ^ { K } u ^ { k } = 1 \}$ for the probability simplex. For $\mathsf { S } \subseteq \mathbb { R } ^ { n } , \mathsf { S } ^ { L }$ denotes the matrices with $L$ rows in ${ \mathsf S } .$ . The embedding matrix $\mathbf { E } \in \mathbb { R } ^ { K \times d }$ maps x to xE. We denote the stop-gradient operator by sg(·).

## 2 PRELIMINARIES: CONTINUOUS DIFFUSION FOR DISCRETE DATA

Let $p _ { \mathrm { d a t a } }$ be a distribution on X. Continuous diffusion models for discrete data (CDMd) map each sample x to an embedding $\mathbf { x } \mathbf { E } \in \mathbb { R } ^ { L \times d }$ , where E is fixed or learned jointly with the model (Strudel et al., 2022; Li et al., 2022). They learn to reverse a Gaussian corruption process in this embedding space, generating samples by progressively denoising Gaussian noise before applying a categorical decoder to produce a token sequence. We describe the forward process and the resulting generative model within the Variational Diffusion Model (VDM) framework (Kingma et al., 2021).

Given $\mathbf { X } \sim p _ { \mathrm { d a t a } } .$ the forward process $( Z _ { t } ) _ { t \in [ 0 , 1 ] }$ progressively corrupts the embedding XE with Gaussian noise according to a decreasing schedule $( \alpha _ { t } ) _ { t \in [ 0 , 1 ] } ,$ , with $\alpha _ { 1 } = 0 _ { : }$ , and $\sigma _ { t } ^ { 2 } = 1 - \alpha _ { t } ^ { 2 }$ . It is a Markov process in $\mathbb { R } ^ { L \times d }$ with conditional laws $q _ { t | \mathbf { x } } ( \cdot \mid \mathbf { x } ) = \mathrm { N } ( \alpha _ { t } \mathbf { x } \mathbf { E } , \sigma _ { t } ^ { 2 } \mathrm { I d } )$ . For $0 \leq s < t \leq 1$ , its transition kernels are $q _ { t | s } ( \cdot \mid z _ { s } ) = \mathrm { N } ( \alpha _ { t | s } z _ { s } , \sigma _ { t | s } ^ { 2 } \mathrm { I d } )$ , where $\alpha _ { t | s } = \alpha _ { t } / \alpha _ { s }$ and $\sigma _ { t | s } ^ { 2 } = \sigma _ { t } ^ { 2 } - \alpha _ { t | s } ^ { 2 } \sigma _ { s } ^ { 2 }$ Here, Id is the identity on vectorized latent matrices, and we denote by $p _ { t }$ the marginal density of $Z _ { t }$ Fix a grid $0 = t _ { 0 } < \cdots < t _ { n } = 1$ and write $s _ { i } = t _ { i - 1 }$ . The reverse generative model is given by

$$
p ^ { \theta } ( \mathbf { x } ) = \int p _ { 1 } ( z _ { 1 } ) p _ { \mathbf { x } | 0 } ^ { \theta } ( \mathbf { x } \mid z _ { 0 } ) \prod _ { i = 1 } ^ { n } p _ { s _ { i } | t _ { i } } ^ { \theta } ( z _ { s _ { i } } \mid z _ { t _ { i } } ) \operatorname { d } z _ { t _ { 0 } : t _ { n } } .
$$

The transitions in this model are given using a denoising network $\mathbf { x } ^ { \theta } ( z _ { t } , t ) \in \Delta _ { K } ^ { L }$ by

$$
p _ { s | t } ^ { \theta } ( z _ { s } \mid z _ { t } ) = q _ { s | t , \mathbf { x } } \left( z _ { s } \mid z _ { t } , \mathbf { x } ^ { \theta } ( z _ { t } , t ) \right) ,
$$

where $q _ { s \mid t , \mathbf { x } }$ is the conditional density of $Z _ { s }$ given $Z _ { t } = z _ { t }$ and $\mathbf { X } = \mathbf { x }$ , that we extend to simplexvalued inputs. By Bayes’ rule,

$$
q _ { s | t , \mathbf { x } } ( z _ { s } \mid z _ { t } , \mathbf { x } ) = { \frac { q _ { t | s } ( z _ { t } \mid z _ { s } ) q _ { s | \mathbf { x } } ( z _ { s } \mid \mathbf { x } ) } { q _ { t | \mathbf { x } } ( z _ { t } \mid \mathbf { x } ) } } = \mathrm { N } { \left( z _ { s } ; { \frac { \alpha _ { t | s } \sigma _ { s } ^ { 2 } } { \sigma _ { t } ^ { 2 } } } z _ { t } + { \frac { \alpha _ { s } \sigma _ { t | s } ^ { 2 } } { \sigma _ { t } ^ { 2 } } } \mathbf { x } \mathbf { E } , { \frac { \sigma _ { s } ^ { 2 } \sigma _ { t | s } ^ { 2 } } { \sigma _ { t } ^ { 2 } } } \operatorname { I d } \right) }\tag{1}
$$

The same network sets the conditional likelihood as $\begin{array} { r } { p _ { \mathbf { x } | 0 } ^ { \theta } ( \mathbf { x } \mid z _ { 0 } ) = \prod _ { \ell = 1 } ^ { L } \langle \mathbf { x } ^ { \theta , \ell } ( z _ { 0 } , 0 ) , \mathbf { x } ^ { \ell } \rangle } \end{array}$ . The generative model is fitted by minimizing the negative evidence lower bound (NELBO); as the time-grid mesh tends to zero, it takes the form (Kingma et al., 2021; Gulrajani & Hashimoto, 2023):

$$
\begin{array} { r l } &  { \displaystyle \mathsf { L } _ { \mathrm { c } } ( { \bf x } ; \theta ) = \mathbb { E } _ { Z _ { 0 } \sim q _ { 0 \mid { \bf x } } ( \cdot \mid { \bf x } ) } \left[ - \sum _ { \ell = 1 } ^ { L } \log \langle { \bf x } ^ { \theta , \ell } ( Z _ { 0 } , 0 ) , { \bf x } ^ { \ell } \rangle \right] } \\ & { \qquad + \displaystyle \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } [ - \mathrm { S N R } ^ { \prime } ( t ) ] \mathbb { E } _ { Z _ { t } \sim q _ { t \mid { \bf x } } ( \cdot \mid { \bf x } ) } \left[ \big \| \left( { \bf x } ^ { \theta } ( Z _ { t } , t ) - { \bf x } \right) { \bf E } \big \| _ { \mathrm { F } } ^ { 2 } \right] \mathrm { d } t , } \end{array}
$$

where $\mathrm { S N R } ( t ) = \alpha _ { t } ^ { 2 } / \sigma _ { t } ^ { 2 }$ . The diffusion loss is such that, when $t > 0 .$ , the embedded denoiser output $\mathbf { x } ^ { \theta } \mathbf { E }$ targets the posterior mean in embedding space $( t , z _ { t } ) \mapsto m _ { t } ( z _ { t } ) \mathbf { E }$ , where $m _ { t } ( z _ { t } ) = \mathbb { E } [ { \bf X } | Z _ { t } ^ { \bar { } } =$ $z _ { t } ] \in \bar { \Delta _ { K } ^ { L } }$ is the conditional mean of X given $Z _ { t }$ . For $t = 0$ , the reconstruction term ensures that the denoiser network’s target is $m _ { 0 }$ . Alternatively, the denoiser network can be trained directly by cross-entropy (Dieleman et al., 2022; Chen et al., 2026b):

$$
\mathsf { L } _ { \mathrm { C E } } ( \mathbf { x } ; \theta ) = \mathbb { E } _ { t \sim \operatorname { U n i f } ( 0 , 1 ) } \left[ - \sum _ { \ell = 1 } ^ { L } \log \langle \mathbf { x } ^ { \theta , \ell } ( Z _ { t } , t ) , \mathbf { x } ^ { \ell } \rangle \right] \ ,
$$

for which the target at optimum is $m _ { t }$ . Although not equivalent to the NELBO above, the crossentropy loss can yield a likelihood bound with a suitable weighting (Davis et al., 2026). Embedding normalization, self-conditioning, and learned noise schedules are further design choices detailed in Section B.1.

## 3 DISTRIBUTION MATCHING DISTILLATION FOR CONTINUOUS DIFFUSION ON DISCRETE DATA

Distributional distillation trains a student by matching the distribution of its generated samples to the data distribution. In our setting, the student takes as input a latent $Z _ { T }$ at noise level $T \in { \bar { ( 0 , 1 ] } }$ and produces a distribution over clean token representations. These outputs are then renoised with the Gaussian forward process of Section 2, and their distributions are matched with the corresponding data marginals $p _ { t }$ at lower noise levels $t < T$ . For the approaches considered, the matching objective involves the fixed pretrained CDMd $p ^ { \theta }$ , used as the distillation teacher.

More precisely, given $Z _ { T } = z _ { T }$ , the student network outputs $\mathbf { x } ^ { \eta } ( z _ { T } , T ) \in \Delta _ { K } ^ { L }$ , which parameterizes a conditional kernel $\mathsf { k } _ { T } ^ { \eta } ( \cdot \mid z _ { T } )$ . We consider two parameterizations of this kernel. The first uses the probability vectors themselves as simplex-valued token representations, while the second samples one categorical token at each sequence position:

$$
\begin{array} { r } { { \sf k } _ { T } ^ { \eta } ( \mathrm { d } { \bf x } \mid z _ { T } ) = \delta _ { { \bf x } ^ { \eta } ( z _ { T } , T ) } ( \mathrm { d } { \bf x } ) , \quad { \bf x } \in \Delta _ { K } ^ { L } , } \end{array}
$$

$$
( \mathrm { c o n t i n u o u s ~ r e l a x a t i o n } ) ,\tag{2}
$$

$$
\mathsf { k } _ { T } ^ { \eta } ( \mathbf { x } \mid z _ { T } ) = \prod _ { \ell = 1 } ^ { L } \left. \mathbf { x } ^ { \ell } , \mathbf { x } ^ { \eta , \ell } ( z _ { T } , T ) \right. , \quad \mathbf { x } \in \mathsf { X } ,
$$

$$
( \mathrm { d i s c r e t e \ s a m p l i n g } ) .\tag{3}
$$

These two parameterizations share the same distribution-matching objective but lead to different gradient estimators. We formulate this objective for one-step generation in a framework unifying existing methods by considering a general class of matching objectives, extend it to multi-step generation and derive Simplex-DMD for the continuous relaxation and Reinforce-DMD for discrete sampling.

## 3.1 A UNIFIED VIEW OF ONE-STEP DISTRIBUTION MATCHING

We first consider one-step generation and isolate a common formulation of distributional distillation. Existing distribution-matching methods can be viewed as comparing the noised student and data distributions across noise levels, while differing in the discrepancy used to compare these marginals. We formalize this view through a generic discrepancy D, before specializing it to the reverse KL used by our methods.

In the one-step setting, $T = 1$ and the student starts from $Z _ { 1 } \sim p _ { 1 } = \mathrm { N } ( 0 , \mathrm { I d } )$ . Its clean output $\mathbf { X } ^ { \eta }$ is sampled according to $\mathsf { k } _ { 1 } ^ { \eta } ( \cdot \mid Z _ { 1 } )$ , inducing the marginal distribution $\begin{array} { r } { p _ { \mathbf { x } } ^ { \dot { \eta } } = \int \mathsf { k } _ { 1 } ^ { \eta } ( \cdot \mid z _ { 1 } ) p _ { 1 } ( \bar { z _ { 1 } } ) \mathrm { d } z _ { 1 } } \end{array}$

Applying the Gaussian forward process of Section 2 to $\mathbf { X } ^ { \eta } \sim p _ { \mathbf { x } } ^ { \eta }$ gives a latent process $( Z _ { t } ^ { \eta } ) _ { t \in [ 0 , 1 ] }$ We denote its marginals by $p _ { t } ^ { \eta }$ , with the corresponding data marginal being $p _ { t }$ . Comparing $p _ { \mathbf { x } } ^ { \eta }$ with $p _ { \mathrm { d a t a } }$ is difficult; Gaussian noising smooths both distributions in the same embedding space, where the diffusion teacher $p ^ { \theta }$ provides quantities that can be leveraged by the loss objectives. Matching these marginals across noise levels defines the generic one-step distribution-matching objective:

$$
\mathsf { L } _ { \mathrm { d i s t i l l } } ( \eta ) = \int _ { 0 } ^ { 1 } \mathbb { E } _ { Z _ { t } ^ { \eta } \sim p _ { t } ^ { \eta } } [ \mathsf { D } ( p _ { t } ^ { \eta } , p _ { t } , Z _ { t } ^ { \eta } ) ] \mathrm { ~ d } t ,\tag{4}
$$

where D denotes a local discrepancy between student and data marginals evaluated at $z _ { t }$ . Different choices of D recover distribution-matching objectives used in prior work, as summarized in Table 1; full objectives and derivations are given in Section C. The gradient of (4) has two contributions: (A) from the dependence of the sampling law on $\eta ,$ and (B) from the dependence of D on $p _ { t } ^ { \eta }$

$$
\nabla _ { \eta } \mathsf { L } _ { \mathrm { d i s t i l l } } ( \eta ) = \int _ { 0 } ^ { 1 } \left[ \underbrace { \int \mathsf { D } ( p _ { t } ^ { \eta } , p _ { t } , z _ { t } ) \nabla _ { \eta } p _ { t } ^ { \eta } ( z _ { t } ) \mathrm { d } z _ { t } } _ { \qquad \mathrm { ~ } \forall \mathrm { ~ } \eta \in \Bigl \{ \vphantom { \int \eta } p _ { t } ^ { \eta } ( z _ { t } ) \nabla _ { \eta } \mathsf { D } ( p _ { t } ^ { \eta } , p _ { t } , z _ { t } ) \mathrm { d } z _ { t } \right\} } \mathrm { d } t .\tag{5}
$$

Following Diff-Instruct and DMD $( \mathrm { L u o } \operatorname { e t } \mathrm { a l }$ ., 2023; Yin et al., 2024), we focus in this work on the reverse KL, choosing $\mathsf { D } ( p _ { t } ^ { \eta } , p _ { t } , z _ { t } ) = \log \big ( p _ { t } ^ { \eta } ( z _ { t } ) / p _ { t } ( z _ { t } ) \big )$ . The loss thus becomes:

$$
\mathsf { L } _ { \mathrm { D M D } } ( \eta ) = \int _ { 0 } ^ { 1 } \mathrm { K L } ( p _ { t } ^ { \eta } \| p _ { t } ) \ \mathrm { d } t .\tag{6}
$$

In this case, (B) integrates to zero (Section D.1), leaving only the dependence of the sampled student marginal on η in (A). Estimating this term depends on the student parameterization: the continuous

<table><tr><td>Criterion</td><td>Local discrepancy D</td><td>Representative methods</td></tr><tr><td>Continuous diffusion</td><td></td><td></td></tr><tr><td>Reverse KL</td><td> $\log p _ { t } ^ { \eta } ( z _ { t } ) - \log p _ { t } ( z _ { t } )$ </td><td>Diff-Instruct (Luo et al., 2023), DMD (Yin et al., 2024) SiD (Zhou et al., 2024); FGM (Huang</td></tr><tr><td>Score / Velocity / Moment matching</td><td> $\left. \nabla _ { z _ { t } } \log p _ { t } ^ { \eta } ( z _ { t } ) - \nabla _ { z _ { t } } \log p _ { t } ( z _ { t } ) \right. ^ { 2 }$ </td><td>et al., 2024); Moment matching (Salimans et al., 2024)</td></tr><tr><td>Discrete diffusion</td><td></td><td></td></tr><tr><td>Bregman divergence</td><td> $\sum _ { \mathbf { H } ( z ^ { \prime } , z _ { t } ) = 1 } d _ { F } \left( \frac { p _ { t } ^ { \eta } ( z ^ { \prime } ) } { p _ { t } ^ { \eta } ( z _ { t } ) } , \frac { p _ { t } ( z ^ { \prime } ) } { p _ { t } ( z _ { t } ) } \right)$   $d _ { \mathrm { H } } ( z ^ { \prime } , z _ { t } ) { = } 1$ </td><td>IDLM (Li et al., 2026); D-MMD (Hoogeboom et al., 2026)</td></tr><tr><td>Posterior f-divergence</td><td> $\sum _ { \ell } D _ { f } \left( p _ { \mathbf { x } | t } ^ { \eta , \ell } ( \cdot \mid z _ { t } ) , p _ { \mathbf { x } | t } ^ { \ell } ( \cdot \mid z _ { t } ) \right)$ </td><td>DiMO (Zhu et al., 2025)</td></tr></table>

Table 1: Local discrepancies for distributional distillation. Score, velocity and moment discrepancies agree up to time-dependent factors. Definitions and derivations are given in Section C.

relaxation admits a pathwise gradient, whereas discrete sampling requires a score function estimator.   
We derive the two estimators in Sections 3.3 and 3.4.

The previous objective (4) matches the student and data distributions only after embedding and Gaussian noising. It is therefore not immediate that exact matching of the noised marginals recovers the original discrete data distribution. The following result shows that this is nevertheless the case, under a simple geometric condition on the embedding matrix that we satisfy.

Theorem 1. Assume that the rows of E are distinct and have a common Euclidean norm. Then

$$
\mathsf { L } _ { \mathrm { D M D } } ( \eta ) = 0 \qquad \Longleftrightarrow \qquad p _ { \bf x } ^ { \eta } = p _ { \mathrm { d a t a } } .
$$

We give the proof in Section D.2.

## 3.2 MULTI-STEP GENERATION

The one-step construction of Section 3.1 learns to transport the terminal marginal $p _ { 1 }$ to the data distribution. In order to extend the method to multi-step generation, we adapt this construction to arbitrary starting noise levels $T \in ( 0 ,$ 1]: given $Z _ { T } \sim p _ { T }$ , the student, conditioned on T, produces a clean sample $\mathbf { \breve { X } } ^ { \eta , T } \sim \mathsf { k } _ { T } ^ { \eta } ( \cdot \mid Z _ { T } )$ , and we denote its marginal by $p _ { \mathbf { x } } ^ { \eta , T }$ . Renoising this sample to a noise level $t < T$ with the forward process of Section 2 yields the latent states $( Z _ { t } ^ { \eta , T } )$ ), inducing marginals $p _ { t } ^ { \eta , T }$ , which we match to the data marginal $p _ { t }$ using the same reverse-KL objective as in the one-step case (6). Varying T yields the multi-step objective:

$$
\mathsf { L } _ { \mathrm { D M D } } ^ { \mathrm { m s } } ( \eta ) = \mathbb { E } _ { ( t , T ) \sim \nu } \left[ \mathrm { K L } \Bigl ( p _ { t } ^ { \eta , T } \Big \| p _ { t } \Bigr ) \right] \mathrm { ~ , ~ }\tag{7}
$$

where the joint distribution ν over (t, T) is defined as $\nu ( \mathrm { d } t , \mathrm { d } T ) = \nu _ { T } ( \mathrm { d } t ) \nu _ { 1 } ( \mathrm { d } T )$ , with $\nu _ { s } =$ $\mathrm { U n i f } ( 0 , s ) \stackrel { \cdot } { \mathrm { f o r } } s \in ( 0 , 1 ]$ . Learning these transports for all starting levels enables multi-step generation by composing them along decreasing noise levels $1 = T _ { n } > \bar { T } _ { n - 1 } > \cdot \cdot \cdot > T _ { 0 } = 0$ . Starting from $\bar { \widehat { Z } } _ { T _ { n } } \sim \bar { p } _ { 1 }$ , at each level $T _ { i }$ the student draws $\widehat { \mathbf { X } } ^ { \eta , T _ { i } } \sim \mathsf { k } _ { T _ { i } } ^ { \eta } ( \cdot \mid \widehat { Z } _ { T _ { i } } )$ and maps it to the next level $T _ { i - 1 }$ . We explore the following family of DDIM-inspired transitions (Song et al., 2021), a standard approach for exploring multi-step performance, which controls the stochasticity of the sampling procedure:

$$
p _ { t | T } ^ { \eta } ( z _ { t } \mid z _ { T } ) = \int q _ { t | \mathbf { x } , T } ^ { \zeta } ( z _ { t } \mid \mathbf { x } , z _ { T } ) \mathbf { k } _ { T } ^ { \eta } ( \mathrm { d } \mathbf { x } \mid z _ { T } ) ~ , \qquad t < T ,
$$

where, for $0 \le \zeta _ { t , T } \le \sigma _ { t }$

$$
q _ { t | \mathbf { x } , T } ^ { \zeta } ( z _ { t } \mid \mathbf { x } , z _ { T } ) = \mathrm { N } \biggl ( z _ { t } ; \alpha _ { t } \mathbf { x } \mathbf { E } + \sqrt { \sigma _ { t } ^ { 2 } - \zeta _ { t , T } ^ { 2 } } \frac { z _ { T } - \alpha _ { T } \mathbf { x } \mathbf { E } } { \sigma _ { T } } , \zeta _ { t , T } ^ { 2 } \mathrm { I d } \biggr ) \ .\tag{8}
$$

The parameter $\zeta _ { t , T }$ interpolates between retaining the noise inferred from the current latent and injecting fresh Gaussian noise. $\mathrm { A t } \ \zeta _ { t , T } \ = \ 0$ , the update is deterministic conditional on x. At $\zeta _ { t , T } = \sigma _ { t }$ , it discards the current latent noise and recovers the forward-renoising transition used to define $Z _ { t } ^ { \eta , T }$ , whereas $\zeta _ { t , T } = { \sigma _ { t } \sigma _ { T | t } } / { \sigma _ { T } }$ recovers the Gaussian bridge (1). Finally, the generative model takes a final step from ${ \widehat { Z } } _ { 0 }$ by applying the categorical decoder parameterized by $\mathbf { x } ^ { \eta } ( \widehat { Z } _ { 0 } , 0 )$ . In the forward renoising $\zeta _ { t , T } = \sigma _ { t }$ case, these transitions recover the correct diffusion marginals $p _ { t }$ if $\widehat { Z } _ { T } \sim p _ { T }$ and the induced clean student marginal equals $p _ { \mathrm { d a t a } } ;$ in general, the diffusion marginals are also recovered if the student kernel equals the exact posterior $p _ { \mathbf { x } | T } ( \cdot \mid z _ { T } )$ , see Section D.5.

## 3.3 SIMPLEX-DMD: CONTINUOUS RELAXATION

Under the continuous relaxation (2), the student’s probability vectors are used as the clean sample: $\mathbf { X } ^ { \eta , T } = \mathbf { x } ^ { \eta } ( Z _ { T } , T )$ . The noising process is differentiable with respect to the student parameters, allowing us to optimize the multi-step distribution-matching objective with a pathwise gradient.

Student gradient. We first derive the gradient of the matching objective (7). Define the reparameterized forward-noising map $\pi _ { t } : ( \mathbf { x } , \varepsilon ) \mapsto \alpha _ { t } \mathbf { x } \mathbf { E } + \sigma _ { t } \varepsilon$ , such that $Z _ { t } ^ { \eta , T } \overset { d } { = } \pi _ { t } ( \mathbf { x } ^ { \eta } ( Z _ { T } , T ) , \varepsilon )$ for $Z _ { T } \sim p _ { T }$ and $\varepsilon \sim \mathrm { N } ( 0 , \mathrm { I d } )$ . This makes explicit the dependence of the noised latent on η and how the reparameterization trick (Kingma & Welling, 2014) applies to (7), similarly to Diff-Instruct and DMD (Luo et al., 2023; Yin et al., 2024):

$$
\nabla _ { \eta } \mathrm { L } _ { \mathrm { D M D } } ^ { \mathrm { m s } } ( \eta ) = \mathbb { E } _ { ( t , T ) \sim \nu , \zeta _ { T } \sim p _ { T } , \varepsilon \sim \mathrm { N } ( 0 , \mathrm { I d } ) } \left[ \left( s ^ { \eta , T } ( Z _ { t } ^ { \eta , T } , t ) - s ( Z _ { t } ^ { \eta , T } , t ) \right) ^ { \top } \frac { \partial Z _ { t } ^ { \eta , T } } { \partial \eta } \right] ,\tag{9}
$$

where $s ^ { \eta , T } ( z , t ) = \nabla _ { z } \log p _ { t } ^ { \eta , T } ( z )$ and $s ( z , t ) = \nabla _ { z } \log p _ { t } ( z )$ are respectively the student and data scores (see Section D.1). Thus, the pathwise update requires evaluating the difference between the score of the noised student distribution and that of the data distribution.

Auxiliary score estimation. Neither score in (9) is available exactly, but under Gaussian noising, each is determined by the posterior mean of its corresponding clean representation. This relation is given by Tweedie’s formula (Efron, 2011):

$$
s ( z , t ) = \frac { \alpha _ { t } m _ { t } ( z ) { \bf E } - z } { \sigma _ { t } ^ { 2 } } , \qquad s ^ { \eta , T } ( z , t ) = \frac { \alpha _ { t } m _ { t } ^ { \eta , T } ( z ) { \bf E } - z } { \sigma _ { t } ^ { 2 } } ,\tag{10}
$$

where $m _ { t } ^ { \eta , T } ( z ) = \mathbb { E } \left\lceil \mathbf { X } ^ { \eta , T } \mid Z _ { t } ^ { \eta , T } = z \right\rceil$ is the posterior mean of the student distribution. The frozen teacher denoiser $\mathbf { x } ^ { \theta } ( z , t )$ provides an estimate of the data posterior mean, and hence an estimate $s ^ { \theta } ( z , t )$ of $s ( z , t )$ . We train an auxiliary denoiser $\mathbf { x } ^ { \phi } ( z , t , \bar { T } ) \in \Delta _ { K } ^ { L }$ to target the posterior mean $m _ { t } ^ { \eta , T }$ of the student distribution, by minimizing the following soft-target cross-entropy:

$$
\mathsf { L } _ { \mathrm { a u x } } ( \phi ) = \mathbb { E } _ { ( t , T ) \sim \nu , Z _ { T } \sim p _ { T } , \varepsilon \sim \mathrm { N } ( 0 , \mathrm { I d } ) } \left[ \sum _ { \ell = 1 } ^ { L } \mathrm { C E } \big ( \mathbf { X } ^ { \eta , T , \ell } , \mathbf { x } ^ { \phi , \ell } ( Z _ { t } ^ { \eta , T } , t , T ) \big ) \right] .\tag{11}
$$

We alternate $n _ { \mathrm { a u x } }$ auxiliary updates on detached student outputs with one student update.

## 3.4 REINFORCE-DMD: DISCRETE SAMPLING

The student clean sample $\mathbf { X } ^ { \eta , T }$ is now drawn from a discrete distribution given by the transition $\mathsf { k } _ { T } ^ { \eta } ( \cdot \mid Z _ { T } )$ of (3), preventing the direct pathwise differentiation used by Simplex-DMD.

Student gradient. The gradient of the multi-step objective (7) admits the same decomposition as that of the one-step objective derived in (5). In our reverse KL matching setting, contribution (B) vanishes, so it remains to differentiate the conditional law of $\mathbf { X } ^ { \eta , T }$ which gives the exact score-function identity (Williams, 1992):

$$
\begin{array} { r } { \nabla _ { \eta } \mathsf { L } _ { \mathrm { D M D } } ^ { \mathrm { m s } } ( \eta ) = \mathbb { E } _ { \mathbf { \phi } _ { T } \sim \nu _ { 1 } , Z _ { T } \sim p _ { T } } \left[ C _ { T } ( \mathbf { X } ^ { \eta , T } ) \nabla _ { \eta } \log \mathsf { k } _ { T } ^ { \eta } ( \mathbf { X } ^ { \eta , T } \mid Z _ { T } ) \right] \ , } \\ { \mathbf { X } ^ { \eta , T } \sim \mathsf { k } _ { T } ^ { \eta } ( \cdot \vert Z _ { T } ) \qquad } \end{array}
$$

$$
C _ { T } ( \mathbf { X } ^ { \eta , T } ) = \mathbb { E } _ { \boldsymbol { Z } _ { t } ^ { \eta , T } \sim q _ { t \mid \mathbf { x } } ( \cdot \vert \mathbf { X } ^ { \eta , T } ) } \left[ \log \frac { p _ { t } ^ { \eta , T } ( Z _ { t } ^ { \eta , T } ) } { p _ { t } ( Z _ { t } ^ { \eta , T } ) } \right] .
$$

The cost $C _ { T }$ is the expected log-ratio of the student and data marginals at time t. Unlike the scores in the Simplex-DMD gradient, which diffusion denoisers can estimate, this log-ratio is not directly available and requires another estimator.

Auxiliary estimation. In order to estimate this log-ratio of interest, we train a discriminator $D _ { \psi } ( z _ { t } , t , T ) \in ( 0 , 1 )$ to distinguish noised data from noised student samples:

$$
\begin{array} { r l } & { \mathrm { L } _ { \mathrm { d i s c } } ( \psi ) = - \displaystyle \int \left[ \mathbb { E } _ { Z _ { t } \sim p _ { t } } \log D _ { \psi } ( Z _ { t } , t , T ) + \mathbb { E } _ { Z _ { t } ^ { \eta , T } \sim p _ { t } ^ { \eta , T } } \log \left( 1 - D _ { \psi } ( Z _ { t } ^ { \eta , T } , t , T ) \right) \right] \nu ( \mathrm { d } t , \mathrm { d } T ) . } \end{array}\tag{12}
$$

For a fixed student, the population optimum is $D ^ { \star } ( z , t , T ) = p _ { t } ( z ) / ( p _ { t } ( z ) + p _ { t } ^ { \eta , T } ( z ) )$ , whose negative logit equals log $( p _ { t } ^ { \eta , T } / p _ { t } )$ (Section D.3). For a pair $0 < t < T \leq 1$ and independent noise $\varepsilon \sim \mathrm { N } ( 0 , \mathrm { I d } )$ , the pointwise cost estimate is:

$$
\widehat { C } _ { \psi } ( { \bf X } ^ { \eta , T } ; t , T , \varepsilon ) = \log \frac { 1 - D _ { \psi } ( \pi _ { t } ( { \bf X } ^ { \eta , T } , \varepsilon ) , t , T ) } { D _ { \psi } ( \pi _ { t } ( { \bf X } ^ { \eta , T } , \varepsilon ) , t , T ) } .
$$

Variance reduction and teacher supervision. For each $( Z _ { T } , T )$ , we draw $G \geq 2$ independent triplets $( \mathbf { X } _ { g } ^ { \eta , T } , t _ { g } , \varepsilon _ { g } )$ with $\mathbf { X } _ { g } ^ { \eta , T } \sim \mathsf { k } _ { T } ^ { \hat { \eta } } ( \cdot \mid Z _ { T } ) , t _ { g } \sim \nu _ { T }$ , and $\varepsilon _ { g } \sim \mathrm { N } ( 0 , \mathrm { I d } )$ . We compute the costs $\widehat { C } _ { g } = \widehat { C } _ { \psi } ( \mathbf { X } _ { g } ^ { \eta , T } ; t _ { g } , T , \varepsilon _ { g } )$ and the leave-one-out baselines $\begin{array} { r } { \widehat { b } _ { g } = ( G - 1 ) ^ { - 1 } \sum _ { g ^ { \prime } \neq g } \widehat { C } _ { g ^ { \prime } } } \end{array}$ (Kool et al., 2019). Inspired by the group-based baseline in GRPO, we center each cost by subtracting $\widehat { b } _ { g }$ (Shao et al., 2024), and include a KL anchor required for stable training in our experiments (Section 4.2), yielding the per-group surrogate:

$$
\widehat { \mathrm { L } } _ { \mathrm { R - D M D } } ( \eta ) = \frac { 1 } { G } \sum _ { g = 1 } ^ { G } \mathrm { s g } ( \widehat { C } _ { g } - \widehat { b } _ { g } ) \log { \sf k } _ { T } ^ { \eta } ( { \bf X } _ { g } ^ { \eta , T } \mid Z _ { T } ) + \beta \sum _ { \ell = 1 } ^ { L } \mathrm { K L } \big ( { \bf x } ^ { \eta , \ell } ( Z _ { T } , T ) \big | \big | { \bf x } ^ { \theta , \ell } ( Z _ { T } , T ) \big ) ,\tag{13}
$$

where $\beta \geq 0$ . We average this surrogate over sampled $( Z _ { T } , T )$ and their groups. The leave-one-out baseline preserves the expected cost-based update. We use $G = 4 .$ , requiring one student evaluation and G cost evaluations per group.

We alternate $n _ { \mathrm { a u x } }$ discriminator updates on detached student samples with one student update using (13). During student updates, the discriminator, costs and baselines are detached.

## 3.5 RELATED WORK

Continuous diffusion for discrete data. Diffusion-LM jointly learns embeddings and a diffusion model (Li et al., 2022), SED uses self-conditioning in embedding diffusion (Strudel et al., 2022), and Plaid develops a likelihood-based training framework (Gulrajani & Hashimoto, 2023). Alongside these developments, CDCD introduced score interpolation, parameterizing the score through an estimate of the posterior mean (Dieleman et al., 2022). More recently, LangFlow (Chen et al., 2026b) and RePlaid (Yang et al., 2026) refined noise scheduling and training, achieving language modeling performance competitive with that of discrete diffusion models.

Trajectory distillation. Consistency models learn to map states along a probability-flow trajectory to a common endpoint, enabling generation with one or a few model evaluations (Song et al., 2023; Song & Dhariwal, 2024). Flow map matching generalizes this construction to maps between arbitrary times, with objectives for distilling pretrained models as well as training directly from data through self-distillation (Boffi et al., 2025a;b). Categorical flow maps, including CFM (Roos et al., 2026), DFM (Potaptchik et al., 2026), and FMLM (Lee et al., 2026), build on CDMd to extend these ideas to discrete data, enabling one- and few-step generation. For discrete diffusion, SDTT matches predictions along teacher rollouts (Deschenaux & Gulcehre, 2025), while Di4C distills transition compositions into mixture models that capture correlations across dimensions (Hayakawa et al., 2025). A complementary approach is taken by Rectified Flow (Liu et al., 2023) and a discrete adaptation ReDi (Yoo et al., 2025), which iteratively rectifies the source-target coupling in flow matching to accelerate sampling.

Distributional distillation. Distributional distillation methods differ primarily in their choice of local discrepancy and how they estimate the loss gradient. Table 1 summarizes these methods, and Section C provides their derivations.

## 4 EXPERIMENTS

We evaluate whether Simplex-DMD and Reinforce-DMD preserve generation quality and diversity when distilling a continuous diffusion language model to reduced sampling budgets. We compare them with diffusion and distillation baselines across network evaluation budgets, and ablate the main training and sampling choices.

We distill the released LangFlow model (Chen et al., 2026b) on OpenWebText, using sequences of length 1024 over the GPT-2 (Radford et al., 2019) vocabulary $( K = 5 0 , 2 5 7 )$ . We keep the teacher embedding matrix E fixed and initialize both students and their auxiliary networks from the teacher. Training runs for 10,000 total iterations with $n _ { \mathrm { a u x } } = 4$ auxiliary updates per student update, i.e., 2,000 student updates, requiring approximately 40 H100 GPU hours.

We evaluate generation quality with generative perplexity (Gen PPL, lower is better), computed by GPT-2 Large, and diversity with unigram entropy (higher is better). Since lower perplexity can result from reduced diversity, we compare Gen PPL–entropy frontiers obtained by varying the sampling temperature (Pynadath et al., 2026). Baselines include the LangFlow teacher, the continuous-diffusion distillation method FMLM, the discrete diffusion models and distillation methods MDLM, D-MMD, IDLM, SDTT, and ReDi (see Section 3.5), and the autoregressive references GPT-2 (124M) and OPT (125M) (Zhang et al., 2022). For the autoregressive baselines, we use checkpoints from Hugging Face with parameter counts closest to teacher’s. All methods use a common evaluation protocol. Table 3 reports model sizes and training budgets, while Section E and Table 4 provide further evaluation and configuration details.

## 4.1 GENERATION QUALITY ACROSS SAMPLING BUDGETS

Figure 2 compares generative frontiers at representative low and high sampling budgets. At 4 network evaluations, Simplex-DMD achieves lower Gen PPL than the evaluated diffusion baselines throughout the overlapping entropy range. At 256 evaluations, Reinforce-DMD achieves the lowest Gen PPL among the diffusion methods at matched entropy around 5.0. The two methods therefore improve generation quality in complementary sampling regimes: Simplex-DMD at low budgets and Reinforce-DMD at larger budgets.

![](images/657ca3e4fee7f7432fbd01defe73b7264f1e06026beff4882c8ee8ef6eb23059.jpg)

![](images/db100837de85a8a20efad6ff1aff40fe95a416341a1fe879a0d5bc9d214310e9.jpg)  
Figure 2: Generative frontiers on OpenWebText at $\mathrm { N F E } = 4$ and 256. Temperature varies from 0.8 to 1.1; stars mark temperature 1.0 and diamonds the data. Gen PPL uses an inverted log scale. Dashed GPT-2 and OPT use 1,024 evaluations. Simplex-DMD is omitted at $\mathrm { N F E } = 2 5 6$ due to low diversity. Full sweeps are in Section E.2.

At ${ \mathrm { N F E } } = 4 .$ , matching the entropy at 5.44, Simplex-DMD reaches a Gen PPL of 45.6, compared with 90.2 for ReDi, corresponding to a 49% reduction. At $\mathrm { N F E } = 2 5 6$ and matched entropy 5.00, Reinforce-DMD reaches a Gen PPL of 14.9, compared with 18.6 for D-MMD, a 20% reduction.

<table><tr><td rowspan="2">Method</td><td colspan="6">Network evaluations</td></tr><tr><td>2</td><td>4</td><td>8</td><td>16</td><td>128</td><td>256</td></tr><tr><td> $H _ { \mathrm { N F E } }$ </td><td>5.65</td><td>5.44</td><td>5.38</td><td>5.20</td><td>5.27</td><td>5.00</td></tr><tr><td>LangFlow (Chen et al., 2026b) MDLM (Sahoo et al., 2024)</td><td>1077 1909</td><td>317 678</td><td>147 267</td><td>58.6 72.8</td><td>39.1 44.3</td><td>21.3 23.8</td></tr><tr><td>FMLM (Lee et al., 2026) D-MMD (Hoogeboom et al., 2026)</td><td>142 (5.32) 1752</td><td>128 (5.40) 431</td><td>94.7 135</td><td>55.2 41.7</td><td>22.6 (4.87) 27.3</td><td>13.5 (4.50) 18.6</td></tr><tr><td>IDLM (Li et al., 2026) SDTT (Deschenaux &amp; Gulcehre, 2025)</td><td>955</td><td>145</td><td>48.4</td><td>24.3</td><td>11.6(4.75)</td><td>9.14(4.27)</td></tr><tr><td>ReDi (oo et al., 2025)</td><td>1496 261 (5.46)</td><td>372 90.2</td><td>103 49.4</td><td>39.4 32.4 (5.32)</td><td>32.6 30.7</td><td>20.1</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>25.0 (5.19)</td></tr><tr><td>Simplex-DMD Reinforce-DMD</td><td>178 1112</td><td>45.6 556</td><td>26.7 347</td><td>17.0 68.8</td><td>3.60 (3.82) 26.9</td><td>2.51 (3.31) 14.9</td></tr></table>

Table 2: Gen PPL on OpenWebText at matched unigram entropy H<sub>NFE</sub>; lower is better. H<sub>NFE</sub> is the Simplex-DMD entropy closest to the data for $\mathrm { N F E } = 2 \mathrm { - } 1 6$ , and the corresponding Reinforce-DMD entropy for NFE = 128, 256. Log Gen PPL is linearly interpolated at H<sub>NFE</sub>. If a curve does not reach the target entropy, its closest point is reported with entropy in parentheses. Bold indicates the lowest matched value.

With just 4 network evaluations, Simplex-DMD approaches the quality-diversity frontier of OPT-125M which uses 1,024 NFEs, bringing autoregressive-level performance within reach of few-step diffusion generation. At matched unigram entropy $H \approx 5 . 5 \bar { 4 }$ , their Gen PPL values are 54.6 and 52.1, respectively. Reinforce-DMD also compares favorably with OPT: at $H \approx 5 . 0 0$ , it achieves Gen PPL 14.9 versus 17.9, using 256 evaluations.

To compare methods across sampling budgets more systematically, Table 2 reports Gen PPL at a budget-specific matched-entropy target $H _ { \mathrm { N F E } } .$ We linearly interpolate log Gen PPL between measured frontier points that bracket each target. Simplex-DMD obtains the lowest matched-entropy Gen PPL at 2, 4, 8, and 16 network evaluations, whereas Reinforce-DMD matches D-MMD at 128 evaluations and obtains the lowest value at 256. At NFE = 2, FMLM reaches a lower raw Gen PPL than Simplex-DMD (142 versus 178), but only at a substantially lower entropy (5.32 versus 5.65), and is therefore not a matched-diversity comparison.

## 4.2 TRAINING AND SAMPLING CHOICES

We detail the design choices made to obtain the results presented in the previous subsection, and the related ablations validating them.

Teacher supervision stabilizes Reinforce-DMD. Figure 3 examines the effect of the teacher KL anchor introduced in (13) at $\mathrm { N F E } = 1 2 8$ Removing the anchor $( \beta ~ = ~ 0 )$ leads to collapse, with essentially zero unigram entropy. Positive anchor weights and an annealed schedule prevent collapse and yield similar generative perplexity– entropy frontiers, with $\beta = 0 . 5$ providing a modest improvement over the other tested settings. These results indicate that teacher supervision is needed for stable Reinforce-DMD training in the tested configurations, while performance is relatively insensitive to the anchor weight within the positive range considered. We therefore use $\beta = 0 . 5$ in our default configuration.

![](images/b1def795962d5ab542da61366019eb6d9b079d35cd6175a3af4f883cd2586169.jpg)  
Figure 3: Effect of the KL anchor weight β on Reinforce-DMD at $\mathrm { N F E } = 1 2 8$ . Without the anchor, generation collapses $( H = 0 ,$ , off-scale).

![](images/a702e7b15bb5410cc45bbed585a774d5c01faeda13de5e66b9caafd2044783fd.jpg)  
(a) Simplex-DMD, NFE = 4 and 8

![](images/224d48ed35a1a88caad5f60f6cee535c7612001e41fd2548d3d8a53134468b1b.jpg)  
(b) Reinforce-DMD, NFE = 128 and 256

![](images/f05e6774698e2d26b7fd88270eb96a222ed95c7e1b3dc8a465acce8797c527cc.jpg)  
Figure 4: Sampler comparison for c ∈ {0, 0.25, 0.5, 0.75, 1, 1.5} and forward renoising.

Forward renoising performs best among the tested samplers. The multi-step construction of Section 3.2 defines a family of sampling transitions through the noise coefficient $\zeta _ { t , T }$ . We compare

$$
\zeta _ { t , T } = c \frac { \sigma _ { t } \sigma _ { T | t } } { \sigma _ { T } } , \qquad c \in \{ 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 , 1 . 5 \} ,
$$

clipping c to respect the constraints on $\zeta _ { t , T }$ if necessary, with forward renoising, $\zeta _ { t , T } = \sigma _ { t }$ , where a maximum amount of fresh Gaussian noise is added at each sampling step. Across the swept budgets, forward renoising gives the best observed Gen PPL–entropy frontier for both Simplex-DMD and Reinforce-DMD (Figure 4).

Additional training choices. Further ablations in Sections E.3 and E.5 support the remaining choices of our default configuration. Initializing the student from scratch leads to collapse in the tested configurations, motivating initialization from the teacher. Self-conditioning improves Simplex-DMD but provides no benefit for Reinforce-DMD in our experiments, so we use it only for Simplex-DMD. For Simplex-DMD, a bridge-based training construction inspired by moment matching (Salimans et al., 2024) underperforms the forward-renoising training objective; we discuss its auxiliary score approximation in Section D.4.

## 5 CONCLUSION

We introduced a distribution-matching framework for distilling continuous diffusion models on discrete data and derived two methods from a common reverse-KL objective: Simplex-DMD, based on simplex-valued outputs and pathwise gradients, and Reinforce-DMD, based on categorical sampling and score-function estimation. On OpenWebText, they improve generation at complementary sampling budgets, with Simplex-DMD strongest at low budgets and Reinforce-DMD at larger budget among the evaluated distillation methods. Like other distribution-matching approaches, our methods require a pretrained teacher and an auxiliary model during training. Our experiments are also limited to academic model scale; scaling to substantially larger language models will require studying thi training overhead and extending evaluation beyond the Gen PPL–entropy protocol used here.

## AI USE STATEMENT

Generative AI tools assisted with writing and editing the manuscript, including improving clarity and structure and providing feedback on the presentation and interpretation of experimental results. They also assisted with coordinating experiment scripts, cluster jobs, and data collection. The authors reviewed and revised all AI-assisted suggestions and take full responsibility for the scientific claims and final content of the paper.

## REFERENCES

Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

Nicholas M. Boffi, Michael S. Albergo, and Eric Vanden-Eijnden. How to build a consistency model: Learning flow maps via self-distillation. In Advances in Neural Information Processing Systems (NeurIPS), volume 38, 2025a.

Nicholas M. Boffi, Michael S. Albergo, and Eric Vanden-Eijnden. Flow map matching with stochastic interpolants: A mathematical framework for consistency models. Transactions on Machine Learning Research, 2025b.

Tianqi Chen, Shujian Zhang, and Mingyuan Zhou. Dlm-one: Diffusion language models for one-step sequence generation, 2026a. URL https://arxiv.org/abs/2506.00290.

Ting Chen, Ruixiang Zhang, and Geoffrey Hinton. Analog bits: Generating discrete data using diffusion models with self-conditioning. In International Conference on Learning Representations (ICLR), 2023.

Yuxin Chen, Chumeng Liang, Hangke Sui, Ruihan Guo, Chaoran Cheng, Jiaxuan You, and Ge Liu. LangFlow: Continuous diffusion rivals discrete in language modeling. arXiv preprint arXiv:2604.11748, 2026b.

Oscar Davis, Anastasiia Filippova, Pierre Ablin, Victor Turrisi, Amitis Shidani, Marco Cuturi, and Louis Bethune. Scaling categorical flow maps.´ arXiv preprint arXiv:2605.07820, 2026.

Justin Deschenaux and Caglar Gulcehre. Beyond autoregression: Fast LLMs via self-distillation through time. In International Conference on Learning Representations (ICLR), 2025.

Sander Dieleman, Laurent Sartran, Arman Roshannai, Nikolay Savinov, Yaroslav Ganin, Pierre H. Richemond, Arnaud Doucet, Robin Strudel, Chris Dyer, Conor Durkan, Curtis Hawthorne, Remi´ Leblond, Will Grathwohl, and Jonas Adler. Continuous diffusion for categorical data. arXiv preprint arXiv:2211.15089, 2022.

Bradley Efron. Tweedie’s formula and selection bias. Journal of the American Statistical Association, 106(496):1602–1614, 2011.

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in Neural Information Processing Systems (NeurIPS), 2014.

Samson Gourevitch, Yazid Janati, Dario Shariatian, Umut Simsekli, Eric Moulines, Eric P. Xing, and Alain Durmus. Uniform diffusion models revisited: Leave-one-out denoiser and absorbing state reformulation. arXiv preprint arXiv:2605.22765, 2026.

Ishaan Gulrajani and Tatsunori B. Hashimoto. Likelihood-based diffusion language models. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

Satoshi Hayakawa, Yuhta Takida, Masaaki Imaizumi, Hiromi Wakaki, and Yuki Mitsufuji. Distillation of discrete diffusion through dimensional correlations. In International Conference on Machine Learning (ICML), 2025.

Emiel Hoogeboom, David Ruhe, Jonathan Heek, Thomas Mensink, and Tim Salimans. Beyond single tokens: Distilling discrete diffusion models via discrete MMD. arXiv preprint arXiv:2603.20155, 2026.

Keya Hu, Linlu Qiu, Yiyang Lu, Hanhong Zhao, Tianhong Li, Yoon Kim, Jacob Andreas, and Kaiming He. ELF: Embedded language flows. arXiv preprint arXiv:2605.10938, 2026.

Zemin Huang, Zhengyang Geng, Weijian Luo, and Guo-jun Qi. Flow generator matching. arXiv preprint arXiv:2410.19310, 2024.

Diederik P. Kingma and Max Welling. Auto-encoding variational Bayes. In International Conference on Learning Representations (ICLR), 2014.

Diederik P. Kingma, Tim Salimans, Ben Poole, and Jonathan Ho. Variational diffusion models. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 REINFORCE samples, get a baseline for free! In ICLR 2019 Deep Reinforcement Learning meets Structured Prediction Workshop, 2019.

Chanhyuk Lee, Jaehoon Yoo, Manan Agarwal, Sheel Shah, Jerry Huang, Aditi Raghunathan, Seunghoon Hong, Nicholas M. Boffi, and Jinwoo Kim. Flow map language models: One-step language modeling via continuous denoising. arXiv preprint arXiv:2602.16813, 2026.

David Li, Nikita Gushchin, Dmitry Abulkhanov, Eric Moulines, Ivan Oseledets, Maxim Panov, and Alexander Korotin. IDLM: Inverse-distilled diffusion language models. arXiv preprint arXiv:2602.19066, 2026.

Xiang Lisa Li, John Thickstun, Ishaan Gulrajani, Percy Liang, and Tatsunori B. Hashimoto. Diffusion-LM improves controllable text generation. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with Rectified Flow. In International Conference on Learning Representations (ICLR), 2023.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. In International Conference on Machine Learning (ICML), 2024.

Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhihua Zhang. Diff-Instruct: A universal approach for transferring knowledge from pre-trained diffusion models. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

Peter Potaptchik, Jason Yim, Adhi Saravanan, Peter Holderrieth, Eric Vanden-Eijnden, and Michael S. Albergo. Discrete flow maps. arXiv preprint arXiv:2604.09784, 2026.

Patrick Pynadath, Jiaxin Shi, and Ruqi Zhang. Generative frontiers: Why evaluation matters for diffusion language models. arXiv preprint arXiv:2604.02718, 2026.

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. Technical report, OpenAI, 2019.

Daan Roos, Oscar Davis, Floor Eijkelboom, Michael Bronstein, Max Welling, <sup>˙</sup>Ismail <sup>˙</sup>Ilkan Ceylan, Luca Ambrogioni, and Jan-Willem van de Meent. Categorical flow maps. arXiv preprint arXiv:2602.12233, 2026.

Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T. Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

Subham Sekhar Sahoo, Justin Deschenaux, Aaron Gokaslan, Guanghan Wang, Justin Chiu, and Volodymyr Kuleshov. The diffusion duality. In International Conference on Machine Learning (ICML), 2025.

Tim Salimans, Thomas Mensink, Jonathan Heek, and Emiel Hoogeboom. Multistep distillation of diffusion models via moment matching. In Advances in Neural Information Processing Systems (NeurIPS), 2024.

Yair Schiff, Subham Sekhar Sahoo, Hao Phung, Guanghan Wang, Alexander Rush, Volodymyr Kuleshov, Hugo Dalla-Torre, Sam Boshar, Bernardo P. de Almeida, and Thomas Pierrot. Simple guidance mechanisms for discrete diffusion models. In International Conference on Learning Representations (ICLR), 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations (ICLR), 2021.

Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. In International Conference on Learning Representations (ICLR), 2024.

Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In International Conference on Machine Learning (ICML), 2023.

Robin Strudel, Corentin Tallec, Florent Altche, Yilun Du, Yaroslav Ganin, Arthur Mensch, Will´ Grathwohl, Nikolay Savinov, Sander Dieleman, Laurent Sifre, and Remi Leblond. Self-conditioned´ embedding diffusion for text generation. arXiv preprint arXiv:2211.04236, 2022.

Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8:229–256, 1992.

Zhihan Yang, Wei Guo, Shuibai Zhang, Subham Sekhar Sahoo, Yongxin Chen, Arash Vahdat, Morteza Mardani, and John Thickstun. Continuous diffusion scales competitively with discrete diffusion for language. arXiv preprint arXiv:2605.18530, 2026.

Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shechtman, Fr¨ edo Durand, William T. Freeman,´ and Taesung Park. One-step diffusion with distribution matching distillation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

Jaehoon Yoo, Wonjung Kim, and Seunghoon Hong. ReDi: Rectified discrete flow. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer, Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. Opt: Open pre-trained transformer language models, 2022. URL https://arxiv.org/abs/ 2205.01068.

Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In International Conference on Machine Learning (ICML), 2024.

Yuanzhi Zhu, Xi Wang, Stephane Lathuili ´ ere, and Vicky Kalogeiton. Di[M]O: Distilling masked \` diffusion models into a one-step generator. arXiv preprint arXiv:2503.15457, 2025.

## CONTENTS

A Notation 15   
B Background: objectives and parameterizations 16   
B.1 Continuous diffusion on discrete data: objective and design choices 16   
B.2 Discrete diffusion: the bound, the plug-in and the concrete score 18   
C Distributional distillation: discrepancies and gradients 19   
C.1 Continuous distillation baselines: SiD, FGM and moment matching 19   
C.2 Discrete distillation as a Bregman divergence between concrete scores 21   
C.3 Masked Diffusion Distillation: collapse to a posterior KL 22   
D Method: derivations and proofs 22   
D.1 Distribution matching distillation: gradient derivation 22   
D.2 Recovering the discrete data distribution 23   
D.3 The optimal discriminator and the log-ratio 25   
D.4 Auxiliary score estimation with bridge renoising . 25   
D.5 Multi-step generation 26   
E Supplementary experiments 27   
E.1 Implementation details 27   
E.2 Full generative frontiers . 28   
E.3 Design space 28   
E.4 Auxiliary loss 31   
E.5 Bridge renoising for Simplex-DMD 31   
E.6 Generated samples 32

## A NOTATION

Data, representations, and dimensions. The vocabulary is V with K tokens, and $\mathsf { X } = \mathsf { V } ^ { L }$ is the space of sequences of length L. We identify token k with the canonical basis vector $\mathbf { e } _ { k } \in \mathbb { R } ^ { K }$ and discrete sequences with matrices of one-hot rows. The probability simplex is $\Delta _ { K } = \{ u \in \mathbb { R } ^ { K }$ $\begin{array} { r } { u _ { k } \geq 0 , \sum _ { k } ^ { } u _ { k } = 1 \big \} } \end{array}$ ; relaxed sequences and collections of token probabilities belong to $\Delta _ { K } ^ { L }$ . We write $\mathbf { x } ^ { \ell }$ for row ℓ and $x _ { k } ^ { \ell }$ for its k-th coordinate. A collection of token marginals does not by itself specify a joint sequence distribution. The clean data law is $p _ { \mathrm { d a t a } }$ . The embedding matrix $\mathbf { E } \in \mathbb { R } ^ { K \times d }$ has rows $E _ { k }$ , where d is the embedding dimension, and maps x to $\mathbf { x } \mathbf { E } \in \mathbb { R } ^ { L \times d }$ . Uppercase symbols such as X and $Z _ { t }$ denote random variables; lowercase symbols denote their realizations.

Gaussian diffusion and noise levels. The data forward process has marginals $p _ { t }$ and conditional laws

$$
Z _ { t } = \alpha _ { t } { \bf X } { \bf E } + \sigma _ { t } \varepsilon , \qquad q _ { t | { \bf x } } ( \cdot \mid { \bf x } ) = \mathrm { N } ( \alpha _ { t } { \bf x } { \bf E } , \sigma _ { t } ^ { 2 } \mathrm { I d } ) , \qquad \varepsilon \sim \mathrm { N } ( 0 , \mathrm { I d } ) .
$$

Here Id is the identity on vectorized latent matrices, $\sigma _ { t } ^ { 2 } = 1 - \alpha _ { t } ^ { 2 }$ , and the idealized endpoints are $( \alpha _ { 0 } , \sigma _ { 0 } ) = ( 1 , 0 )$ and $( \alpha _ { 1 } , \sigma _ { 1 } ) = ( 0 , 1 )$ , so $p _ { 1 } = \mathrm { N } ( 0 , \mathrm { I d } )$ . For $s < t , q _ { t \mid s }$ denotes the forward transition, with $\alpha _ { t | s } = \alpha _ { t } / \alpha _ { s }$ and $\sigma _ { t | s } ^ { 2 } = \sigma _ { t } ^ { 2 } - \alpha _ { t | s } ^ { 2 } \sigma _ { s } ^ { 2 }$ . The Gaussian bridge $q _ { s \mid t , \mathbf { x } }$ conditions on both the later latent and the clean representation; $p _ { s | t } ^ { \theta }$ denotes a learned teacher reverse transition.

Posteriors, denoisers, and scores. The joint data posterior is $p _ { \mathbf { x } | t } ( \cdot \mid z _ { t } )$ , with token marginal $p _ { \mathbf { x } | t } ^ { \ell } ( \cdot \mid z _ { t } )$ and mean $m _ { t } ( z _ { t } ) = \mathbb { E } [ \mathbf { X } \mid Z _ { t } = z _ { t } ] \in \Delta _ { K } ^ { L }$ . The pretrained teacher has parameters θ, denoiser $\mathbf { x } ^ { \theta } ( z _ { t } , t )$ , and generative law $p ^ { \theta } ; p _ { t }$ always denotes the data forward marginal. Student parameters are $\eta .$ auxiliary-denoiser parameters are $\phi ,$ , and discriminator parameters are ψ. The auxiliary denoiser $\mathbf { x } ^ { \phi } ( z , t , T )$ targets $\bar { m } _ { t } ^ { \eta , T } ( z ) = \mathbb { E } [ \mathbf { X } ^ { \eta , T } \mid Z _ { t } ^ { \eta , T } = z ]$ . The data and student scores are $s ( z , t ) = \nabla _ { z } \log p _ { t } ( z )$ and $s ^ { \eta , T } ( z , t ) = \nabla _ { z } \log p _ { t } ^ { \eta , T } ( z ) ; s ^ { \theta }$ denotes the teacher estimate of the data score. The scalar time s in a transition is distinct from the score function $s ( z , t )$

Student outputs and training marginals. Given $Z _ { T }$ , the student predicts $\mathbf { x } ^ { \eta } ( Z _ { T } , T ) \in \Delta _ { K } ^ { L }$ , which parameterizes the clean-output kernel $\mathsf { k } _ { T } ^ { \eta } ( \cdot \mid Z _ { T } )$ . For Simplex-DMD this kernel is the point mass $\delta _ { \mathbf { x } ^ { \eta } ( Z _ { T } , T ) } ;$ for Reinforce-DMD it is the product of categorical laws with these token probabilities. We write $\mathbf { X } ^ { \eta , T } \sim \mathsf { k } _ { T } ^ { \eta } ( \cdot \mid Z _ { T } )$ and $p _ { \mathbf { x } } ^ { \eta , T }$ for its marginal when $Z _ { T } \sim p _ { T }$ . Forward noising gives $Z _ { t } ^ { \eta , T }$ with law $p _ { t } ^ { \eta , T }$ , where $\pi _ { t } ( \mathbf { x } , \varepsilon ) = \alpha _ { t } \mathbf { x } \mathbf { E } + \sigma _ { t } \varepsilon$ denotes the reparameterized noising map. The one-step notation $\mathbf { X } ^ { \eta } , p _ { \mathbf { x } } ^ { \eta }$ , and $p _ { t } ^ { \eta }$ omits the fixed starting level $T = \dot { 1 }$

Time sampling and matching objectives. In the multi-step objective, T is the starting noise level and $t < \bar { T }$ is the level at which noised distributions are compared. We sample $( t , T ) \sim \nu ,$ , with $\nu ( \mathrm { d } t , \mathrm { d } T ) = \nu _ { T } ( \mathrm { d } t ) \nu _ { 1 } ( \mathrm { d } T )$ and $\nu _ { s } = \mathrm { U n i f } ( 0 , s )$ . The local discrepancy is D, and ${ \mathsf { L } } _ { \mathrm { d i s t i l l } }$ denotes the generic distribution-matching objective. For reverse $\mathrm { K L } .$ , the one-step and multi-step losses are

$$
\mathsf { L } _ { \mathrm { D M D } } ( \eta ) = \int _ { 0 } ^ { 1 } \mathrm { K L } ( p _ { t } ^ { \eta } \parallel p _ { t } ) \mathrm { d } t , \qquad \mathsf { L } _ { \mathrm { D M D } } ^ { \mathrm { m s } } ( \eta ) = \mathbb { E } _ { ( t , T ) \sim \nu } [ \mathrm { K L } ( p _ { t } ^ { \eta , T }  p _ { t } ) ] .
$$

The auxiliary-denoiser and discriminator objectives are ${ \mathsf { L } } _ { \mathrm { a u x } }$ and ${ \mathsf { L } } _ { \mathrm { d i s c } } ,$ , respectively; $n _ { \mathrm { a u x } }$ is the number of auxiliary updates per student update.

Discrete-sampling cost and variance reduction. The discriminator $D _ { \psi } ( z , t , T ) \in ( 0 , 1 )$ estimates the probability that a noised sample comes from data rather than the student. Its negative logit, $\log ( ( 1 - D _ { \psi } ) / D _ { \psi } )$ , estimates the log-ratio log $( p _ { t } ^ { \eta , T } / p _ { t } )$ . The expected log-ratio cost is $C _ { T }$ , and $\widehat { C } _ { \psi }$ denotes its pointwise estimate using a sampled time and Gaussian noise. For a group of G samples sharing $( Z _ { T } , T )$ , g indexes a sample, ${ \widehat { C } } _ { g }$ is its estimated cost, and $\begin{array} { r } { \widehat { b } _ { g } = ( G - 1 ) ^ { - 1 } \sum _ { q ^ { \prime } \neq q } \widehat { C } _ { g ^ { \prime } } } \end{array}$ is its leave-one-out baseline. The coefficient $\beta \geq 0$ weights the tokenwise KL anchor from the student to the frozen teacher.

Multi-step sampling and evaluation. The inference grid is $1 = T _ { n } > T _ { n - 1 } > \cdot \cdot \cdot > T _ { 0 } = 0$ Hatted variables $\widehat { Z } _ { T _ { i } }$ and $\widehat { \mathbf X } ^ { \eta , T _ { i } }$ denote states and clean proposals along the composed sampler; their laws need not equal the corresponding training marginals. The student transition is $p _ { t | T } ^ { \eta }$ , formed by composing $\mathsf { k } _ { T } ^ { \eta }$ with the DDIM-inspired kernel $q _ { t | \mathbf { x } , T } ^ { \zeta }$ . The noise standard deviation $\zeta _ { t , T }$ controls fresh noise: 0 gives a deterministic update conditional on the clean proposal, $\sigma _ { t } \sigma _ { T | t } / \sigma _ { T }$ gives the Gaussian bridge, and $\sigma _ { t }$ gives forward renoising. The sampler experiments use the multiplier c in $\zeta _ { t , T } = c \sigma _ { t } \sigma _ { T | t } / \sigma _ { T }$ . Sampling temperature rescales output logits and is separate from the starting noise level $T$ (the sample appendix also uses $T$ locally for temperature). NFE denotes the number of network evaluations per generated sample. Gen PPL is generative perplexity under GPT-2-large; entropy is the mean per-sequence Shannon entropy of empirical token frequencies, measured in nats.

Operators and appendix conventions. We use $\operatorname { K L } ( P \| Q )$ for Kullback–Leibler divergence, $\begin{array} { r } { \mathrm { C E } ( u , v ) = - \sum _ { k } u _ { k } } \end{array}$ log $v _ { k }$ for cross-entropy, $\operatorname { s g } ( \cdot )$ for stop-gradient, $\delta _ { x }$ for a point mass, and $\circeq$ for equality in distribution. In Section D.2, V is the set of one-hot sequences, $\bar { \mathcal { E } } ( \mathbf { x } ) = \mathbf { x } \mathbf { E } , \rho$ is a clean law, and $Q _ { t } \rho$ is its embedded, noised law; $f _ { \# } \rho$ denotes the law of $f ( X )$ for $X \sim \rho .$ In the comparison of continuous distillation objectives in Section $\mathrm { C } ,$ clean samples are written directly in Euclidean space, so $m _ { t }$ there is a mean in that space; $s _ { t }$ and $v _ { t }$ denote the score and velocity fields. Other locally defined symbols retain the meanings specified in their respective derivations.

## B BACKGROUND: OBJECTIVES AND PARAMETERIZATIONS

## We expand the continuous and discrete formulations recalled in Section 2.

## B.1 CONTINUOUS DIFFUSION ON DISCRETE DATA: OBJECTIVE AND DESIGN CHOICES

We review four design choices for continuous diffusion on token embeddings: training objectives, embedding normalization, self-conditioning and noise schedules, highlighting how existing models implement them.

The objective. We distinguish likelihood-based training, used by Diffusion-LM (Li et al., 2022), Plaid (Gulrajani & Hashimoto, 2023), and RePlaid (Yang et al., 2026), from direct cross-entropy training. In the notation of Section 2, the continuous-time negative evidence lower bound consists of prior, reconstruction, and diffusion terms:

$$
\begin{array} { r l } & { \mathsf { L } _ { \mathrm { c } } ( \mathbf { x } ; \theta ) = \mathsf { L } _ { \mathrm { p r i o r } } ( \mathbf { x } ) + \mathbb { E } _ { Z _ { 0 } \sim q _ { 0 } | \mathbf { x } } ( \cdot | \mathbf { x } ) \left[ - \displaystyle \sum _ { \ell = 1 } ^ { L } \log \langle \mathbf { x } ^ { \theta , \ell } ( Z _ { 0 } , 0 ) , \mathbf { x } ^ { \ell } \rangle \right] } \\ & { \qquad + \displaystyle \frac { 1 } { 2 } \int _ { 0 } ^ { 1 } \left[ - \mathrm { S N R } ^ { \prime } ( t ) \right] \mathbb { E } _ { Z _ { t } \sim q _ { t } | \mathbf { x } } ( \cdot | \mathbf { x } ) \left[ \big \| \big ( \mathbf { x } ^ { \theta } ( Z _ { t } , t ) - \mathbf { x } \big ) \mathbf { E } \big \| _ { \mathrm { F } } ^ { 2 } \right] \mathrm { d } t , } \end{array}
$$

where $\mathrm { S N R } ( t ) = \alpha _ { t } ^ { 2 } / \sigma _ { t } ^ { 2 }$ . For a standard Gaussian generative prior, the prior term is explicit:

$$
\mathsf { L } _ { \mathrm { p r i o r } } ( \mathbf { x } ) = \mathrm { K L } \big ( q _ { 1 | \mathbf { x } } ( \cdot \mid \mathbf { x } ) \big | \big | \mathrm { N } ( 0 , \mathrm { I d } ) \big ) ~ .
$$

Since $\alpha _ { 1 } = 0$ and $\sigma _ { 1 } = 1$ in Section $2 , q _ { 1 | \mathbf { x } } ( \cdot \mid \mathbf { x } ) = \mathrm { N } ( 0 , \mathrm { I d } )$ and this term vanishes. It is nonzero only for schedules with a finite terminal signal-to-noise ratio. The reconstruction term trains the categorical decoder at time zero, while the diffusion term penalizes denoising errors in embedding space.

Alternatively, direct cross-entropy training supervises the clean token at every sampled noise level:

$$
\mathsf { L } _ { \mathrm { C E } } ( \mathbf { x } ; \theta ) = \mathbb { E } _ { t \sim \mathrm { U n i f } ( 0 , 1 ) } \left[ - \sum _ { \ell = 1 } ^ { L } \log \langle \mathbf { x } ^ { \theta , \ell } ( Z _ { t } , t ) , \mathbf { x } ^ { \ell } \rangle \right] .
$$

This approach is used by CDCD (Dieleman et al., 2022) and LangFlow (Chen et al., 2026b), with model-specific representations and parameterizations. Analog Bits uses squared-error regression by default, but also investigates sigmoid and softmax cross-entropy variants (Chen et al., 2023, Appendix B). SED instead combines a mean-squared error diffusion loss in a fixed embedding space with a separate reconstruction cross-entropy loss for its trainable readout (Strudel et al., 2022, Section 3, Eqs. (4)–(5)). Both objectives above are averaged over $\mathbf { x } \sim p _ { \mathrm { d a t a } }$ during training. Unlike the diffusion term of the NELBO, this cross-entropy directly supervises token probabilities rather than their embedded mean; the unweighted objective above is not the NELBO.

Embedding normalization. These objectives also put different pressures on learned embeddings. The diffusion term alone vanishes if all token embeddings coincide, whereas cross-entropy favors distinguishable tokens and can encourage increasing their separation relative to the noise. To control this scale, embeddings are constrained to a sphere, with each row of E having norm 1 or ${ \sqrt { d } } ,$ , where d is the embedding dimension (Dieleman et al., 2022; Chen et al., 2026b; Yang et al., 2026). This prevents the embedding norms from vanishing or diverging, although normalization alone does not prevent different tokens from sharing the same embedding.

Self-conditioning. Self-conditioning lets the denoiser refine an earlier prediction of the clean sample alongside the current noisy latent. Introduced in Analog Bits (Chen et al., 2023) and adopted for token embeddings in SED (Strudel et al., 2022), it conditions on predicted analog-bit vectors and continuous embeddings, respectively. In our token-probability parameterization, we express this mechanism by augmenting the network input as $\mathbf { x } ^ { \theta } ( z _ { t } , \dot { t } , \hat { \mathbf { x } } )$ , where xˆ is a previous prediction in $\Delta _ { K } ^ { L }$ or a zero matrix indicating that no prediction is available. At sampling time, the prediction from the preceding denoising step is reused without an additional network evaluation. Along the descending grid $t _ { n } > \cdots > t _ { 0 }$ , this gives

$$
\hat { \mathbf { x } } ^ { ( i ) } = \mathbf { x } ^ { \theta } \big ( z _ { t _ { i } } , t _ { i } , \hat { \mathbf { x } } ^ { ( i + 1 ) } \big ) , \qquad \hat { \mathbf { x } } ^ { ( n + 1 ) } = 0 .
$$

For $i \geq 1$ , this prediction parameterizes the transition from $t _ { i }$ to $t _ { i - 1 }$ and is passed to the next network evaluation.

During training, a noisy state is sampled directly, so no preceding prediction is available. On a randomly selected subset of training examples, a preliminary forward pass supplies the conditioning input:

$$
\hat { \mathbf { x } } = \mathrm { s g } ( \mathbf { x } ^ { \theta } ( z _ { t } , t , 0 ) ) .
$$

The training loss is then evaluated using $\mathbf x ^ { \theta } ( z _ { t } , t , \hat { \mathbf x } )$ , with gradients flowing only through this second pass. The remaining examples use $\hat { \mathbf { x } } = 0$ , so the network also learns to denoise without a previous prediction, as required at the first sampling step. The fraction of self-conditioned training examples is a model-specific choice.

The noise schedule. The noise schedule determines how quickly the forward process removes information from the clean embeddings. We parameterize it by the increasing negative log signal-tonoise ratio $\gamma ( t ) = - \log \mathrm { S N R } ( t )$ , so that

$$
\alpha _ { t } = \sqrt { \mathrm { s i g m o i d } ( - \gamma ( t ) ) } , \qquad \sigma _ { t } = \sqrt { \mathrm { s i g m o i d } ( \gamma ( t ) ) } .
$$

Thus increasing $\gamma ( t )$ decreases the signal and increases the noise in $Z _ { t } = \alpha _ { t } \mathbf { X } \mathbf { E } + \sigma _ { t } \varepsilon$ . The endpoint $\alpha _ { 1 } = 0$ of Section 2 correspond to $\gamma ( 1 ) = + \infty ;$ practical schedules clip $\gamma$ to finite values γ<sub>min</sub>, γ<sub>max</sub>. In practice, the networks receive the noise level as $\gamma ( t )$ rather than t: the notation $\mathbf { x } ^ { \theta } ( z _ { t } , t )$ of the main text stands for $\mathbf { x } ^ { \theta } ( z _ { t } , \gamma ( t ) )$ ), and likewise for the student and auxiliary networks.

LangFlow (Chen et al., 2026b) learns a schedule that aims to distribute information loss evenly over time. It fits a scaled Gumbel cumulative distribution function to the observed token cross-entropy as a function of $\gamma _ { : }$ , learning its location $\mu ,$ scale $b > 0$ , and amplitude by squared-error regression. The corresponding quantile schedule is

$$
\gamma ( t ) = \mu - b \log ( - \log t ) , \qquad 0 < t < 1 .\tag{14}
$$

This allocates more noise levels to regions where the fitted cross-entropy changes rapidly; in practice, the noise range is clipped.

RePlaid (Yang et al., 2026) instead learns two endpoints and a monotone interpolation between them:

$$
\gamma ( t ) = \gamma _ { 0 } + ( \gamma _ { 1 } - \gamma _ { 0 } ) \tilde { \gamma } ( t ) , \qquad \tilde { \gamma } ( 0 ) = 0 , \quad \tilde { \gamma } ( 1 ) = 1 .
$$

The endpoints are optimized using the diffusion loss, while the monotone network $\tilde { \gamma }$ is trained to reduce the Monte Carlo variance of its estimator. For fixed endpoints and a denoiser taking $\gamma ( t )$ as input, the continuous-time bound is invariant to the interior schedule, so its shape can be adjusted to improve estimation efficiency (Kingma et al., 2021).

In our experiments, we reuse the teacher’s pretrained noise schedule and keep it fixed throughout distillation and sampling.

## B.2 DISCRETE DIFFUSION: THE BOUND, THE PLUG-IN AND THE CONCRETE SCORE

Discrete diffusion progressively corrupts a token sequence and learns to reverse this process to generate data. We consider two choices of the replacement distribution π: a point mass on an additional absorbing [MASK] token for masked diffusion, or the uniform distribution over the vocabulary for uniform diffusion. The process starts directly from the clean sequence, $Z _ { 0 } = \mathbf { X } , \mathbf { s o } p _ { 0 } = p _ { \mathrm { d a t a } } .$ . We describe the forward process, the reverse model and its training bound, then explain how denoiser predictions parameterize reverse transitions and how this relates to concrete scores in continuous time.

Forward process. The forward process corrupts each position independently by replacing its token with a draw from π. Let $( \alpha _ { t } ) _ { t \in [ 0 , 1 ] }$ be a decreasing schedule with $\alpha _ { 0 } = 1$ and $\alpha _ { 1 } = 0$ . Between times $s < t ,$ a token is retained with probability $\alpha _ { t } / \alpha _ { s }$ <sub>s</sub> and otherwise resampled from π, giving the transition

$$
\begin{array} { r } { q _ { t | s } ^ { \ell } ( z _ { t } ^ { \ell } \mid z _ { s } ^ { \ell } ) = \mathrm { C a t } \Big ( z _ { t } ^ { \ell } ; \frac { \alpha _ { t } } { \alpha _ { s } } z _ { s } ^ { \ell } + \big ( 1 - \frac { \alpha _ { t } } { \alpha _ { s } } \big ) \pi \Big ) , \qquad 0 \le s < t \le 1 . } \end{array}\tag{15}
$$

In particular, setting $s = 0$ gives the conditional law given the clean token:

$$
q _ { t | \mathbf { x } } ^ { \ell } ( z _ { t } ^ { \ell } \mid \mathbf { x } ^ { \ell } ) = \mathrm { C a t } \big ( z _ { t } ^ { \ell } ; \alpha _ { t } \mathbf { x } ^ { \ell } + ( 1 - \alpha _ { t } ) \pi \big ) ,\tag{16}
$$

These conditional laws and the forward transitions determine the bridges $q _ { s \mid t , \mathbf { x } }$ by Bayes’ rule. At $t = 1$ , every position follows $\operatorname { C a t } ( \pi )$ independently of the clean sequence, so the terminal distribution is the product prior $p _ { 1 } = \mathrm { C a t } ( \pi ) \dot { \otimes } \dot { L }$

Generative model and KL bound. Generation starts from this prior and applies learned reverse transitions along a grid $0 = t _ { 0 } < \cdots < t _ { n } = 1$ , with $s _ { i } = t _ { i - 1 }$ . Each transition predicts the positions independently conditional on the full current sequence, yielding

$$
p ^ { \theta } ( { \bf x } ) = \sum _ { z _ { t _ { 1 } : t _ { n } } } p _ { 1 } ( z _ { 1 } ) \prod _ { i = 1 } ^ { n } p _ { s _ { i } | t _ { i } } ^ { \theta } ( z _ { s _ { i } } \mid z _ { t _ { i } } ) , \qquad p _ { s | t } ^ { \theta } ( z _ { s } \mid z _ { t } ) = \prod _ { \ell = 1 } ^ { L } p _ { s | t } ^ { \theta , \ell } ( z _ { s } ^ { \ell } \mid z _ { t } ) ,\tag{17}
$$

where $z _ { s _ { 1 } } = \mathbf { x }$ is the generated sequence at time zero. To relate the generated distribution to the data distribution, let $p _ { s \mid t }$ denote the exact reverse transition induced by the forward process with $\mathbf { X } \sim p _ { \mathrm { d a t a } }$ . The exact and learned reverse chains share the same terminal prior. Applying the chain rule for KL divergence to their trajectory laws, then marginalizing to time zero, bounds the discrepancy between data and generated samples by the accumulated transition errors:

$$
\mathrm { K L } \bigl ( p _ { \mathrm { d a t a } } \bigr | \bigr | p ^ { \theta } \bigr ) \leq \sum _ { i = 1 } ^ { n } \mathbb { E } _ { Z _ { t _ { i } } \sim p _ { t _ { i } } } \Bigl [ \mathrm { K L } \Bigl ( p _ { s _ { i } | t _ { i } } ( \cdot \mid Z _ { t _ { i } } ) \Bigr | \Bigr | p _ { s _ { i } | t _ { i } } ^ { \theta } ( \cdot \mid Z _ { t _ { i } } ) \Bigr ) \Bigr ] .\tag{18}
$$

In general, the exact reverse transition contains dependencies between positions that the factorized model cannot represent. For a fixed $z _ { t } ,$ each transition error separates into tokenwise marginal errors and a term accounting for these dependencies:

$$
\mathrm { K L } \Big ( p _ { s | t } ( \cdot \ | \ z _ { t } ) \ \Big | \Big | p _ { s | t } ^ { \theta } ( \cdot \ | \ z _ { t } ) \Big ) \ = \ \sum _ { \ell = 1 } ^ { L } \mathrm { K L } \Big ( p _ { s | t } ^ { \ell } ( \cdot \ | \ z _ { t } ) \ \Big | \Big | \ p _ { s | t } ^ { \theta , \ell } ( \cdot \ | \ z _ { t } ) \Big ) \ + \ { \mathcal T } _ { s | t } ( z _ { t } ) \ ,\tag{19}
$$

where $\begin{array} { r } { p _ { s | t } ^ { \ell } ( \cdot  { \mid } z _ { t } ) = \sum _ { z _ { s } ^ { - \ell } } p _ { s | t } ( z _ { s }  { \mid } z _ { t } ) } \end{array}$ is the ℓ-th token marginal of the exact transition and $\begin{array} { r } { \mathcal { T } _ { s \mid t } ( z _ { t } ) = \mathrm { K L } \Big ( p _ { s \mid t } \big ( \cdot \mid z _ { t } \big ) \Big \| \prod _ { \ell } p _ { s \mid t } ^ { \ell } \big ( \cdot \mid z _ { t } \big ) \Big ) } \end{array}$ measures the conditional dependence between positions under the exact reverse transition. Only the marginal errors depend on θ. Thus, within the factorized family, the optimal transition matches each exact token marginal; the remaining term $\mathcal { T } _ { s \mid t } ( z _ { t } )$ is the irreducible error due to factorization at this time step. This motivates parameterizing the model through predictions that recover these marginals.

The bridge plug-in. The bound asks for a reverse transition, whereas the network produces a point of $\bar { \Delta _ { K } ^ { L } }$ . Nearly every work joins the two by plugging in, evaluating the bridge as though the prediction were the clean token (Sahoo et al., 2024; Schiff et al., 2025),

$$
p _ { s | t } ^ { \theta , \ell } ( \cdot \mid z _ { t } ) = q _ { s | t , \mathbf { x } } ^ { \ell } \big ( \cdot \mid z _ { t } ^ { \ell } , \mathbf { x } ^ { \theta , \ell } ( z _ { t } , t ) \big ) .\tag{20}
$$

Whether this costs anything is the question of whether the token marginals of (19) are themselves of that form, and they are.

Proposition 1 (Gourevitch et al., 2026). $L e t \mathbf { x } ^ { \mathrm { { l o o } , \boldsymbol { \ell } } } ( z _ { t } , t ) = \mathbb { E } [ \mathbf { x } ^ { \ell } \mid z _ { t } ^ { - \ell } ]$ be the leave-one-out denoiser, and let the bridge be extended to a simplex-valuedfirst argument by its Bayes-ratio form, as both processes admit. Then,for every position $\ell ,$

$$
p _ { s | t } ^ { \ell } ( \cdot \mid z _ { t } ) = q _ { s | t , \mathbf { x } } ^ { \ell } \left( \cdot \mid z _ { t } ^ { \ell } , \mathbf { x } ^ { \mathrm { l o o } , \ell } ( z _ { t } , t ) \right) .
$$

When π is uniform this is the only simplex-valued field with that property; when π is the mask, the leave-one-out denoiser agrees with the denoiser $\dot { \mathbb { E } } [ { \mathbf { x } } ^ { \ell } \mid z _ { t } ]$ at every masked position, and the latter has it too.

The plug-in at $\mathbf { x } ^ { \theta } = \mathbf { x } ^ { \mathrm { { l o o } } }$ therefore cancels the sum in (19), leaving only the factorization error. The object it fits is the leave-one-out denoiser, which does not read the position it predicts.

Continuous time. Refining the grid removes the factorization error, two positions both jumping within one interval being a second-order event, so (18) becomes tight over the family (17) in the limit. That limit is written on ratios rather than on transitions. For a state $z _ { t } ,$ , a position ℓ and a token $k ,$ write $z _ { t } ^ { \ell \to k }$ for $z _ { t }$ with its ℓ-th token replaced by k; these are the states at Hamming distance one from $z _ { t }$ . The concrete score of the marginal $p _ { t }$ collects the corresponding ratios,

$$
s _ { t } ^ { \ell } ( z _ { t } ) _ { k } = \frac { p _ { t } ( z _ { t } ^ { \ell  k } ) } { p _ { t } ( z _ { t } ) } ,\tag{21}
$$

and the same ratios for $q _ { t | \mathbf { x } }$ are closed-form, only the ℓ-th factor of (16) changing,

$$
s _ { t } ^ { \ell } ( z _ { t } \mid \mathbf { x } ) _ { k } = \frac { \left. \alpha _ { t } \mathbf { x } ^ { \ell } + ( 1 - \alpha _ { t } ) \pi , e _ { k } \right. } { \left. \alpha _ { t } \mathbf { x } ^ { \ell } + ( 1 - \alpha _ { t } ) \pi , z _ { t } ^ { \ell } \right. } .\tag{22}
$$

The second is defined for any point of $\Delta _ { K } ^ { L }$ in place of x, so the plug-in of (20) carries over to it,

$$
s _ { t } ^ { \theta , \ell } ( z _ { t } ) = s _ { t } ^ { \ell } \big ( z _ { t } \mid \mathbf { x } ^ { \theta } ( z _ { t } , t ) \big ) ,\tag{23}
$$

and returns (21) exactly at $\mathbf { x } ^ { \theta } = \mathbf { x } ^ { \mathrm { { l o o } } }$ , as in Proposition 1 (Gourevitch et al., 2026). The limit of (18) compares the two scores through a Bregman divergence, weighted by the rate at which the reverse process leaves $Z _ { t } ^ { \ell }$ (Lou et al., 2024),

$$
\mathsf { L } _ { \mathrm { d } } ( \theta ) = \int _ { 0 } ^ { 1 } R _ { t } \mathbb { E } _ { Z _ { t } \sim p _ { t } } \bigg [ \sum _ { \ell = 1 } ^ { L } \big \langle \pi , Z _ { t } ^ { \ell } \big \rangle \sum _ { e _ { k } \neq Z _ { t } ^ { \ell } } d _ { F } \Big ( s _ { t } ^ { \ell } ( Z _ { t } ) _ { k } , ~ s _ { t } ^ { \theta , \ell } ( Z _ { t } ) _ { k } \Big ) \bigg ] \mathrm { d } t ,\tag{24}
$$

with $\begin{array} { r l r } { R _ { t } } & { { } = } & { - \alpha _ { t } ^ { \prime } / \alpha _ { t } } \end{array}$ the substitution rate of (15), the reverse rate towards $z _ { t } ^ { \ell \to k }$ being $R _ { t } \langle \pi , z _ { t } ^ { \ell } \rangle s _ { t } ^ { \ell } ( z _ { t } ) _ { k } .$ , generated by $F ( u ) = u \log u - u ,$ so that $d _ { F } ( a , b ) = a \log ( a / b ) - a + \overset { \cdot } { b }$ . The data score (21) is not available, but it is the average of (22) over the denoising posterior, and a Bregman divergence is insensitive to that substitution up to a constant in θ: drawing $\mathbf { X } \sim p _ { \mathbf { x } | t } ( \cdot \mid Z _ { t } )$ and scoring against $s _ { t } ^ { \ell } ( Z _ { t } \mid \mathbf { X } )$ is the form actually minimized. Nothing in (24) requires $s _ { t } ^ { \theta }$ to come from a denoiser: parameterizing the concrete score directly by a network is SEDD (Lou et al., 2024).

## C DISTRIBUTIONAL DISTILLATION: DISCREPANCIES AND GRADIENTS

This appendix derives the objectives summarized in Table 1, the gradient each method descends, and the identities relating them. All methods are considered in the one-step setting of Section 3.1: $T = 1$ the student starts from $Z _ { 1 } \sim p _ { 1 }$ , the terminal law of the diffusion considered, and its clean output $\mathbf { X } ^ { \eta } \sim \mathsf { k } _ { 1 } ^ { \eta } ( \cdot \mid Z _ { 1 } )$ has law $p _ { \mathbf { x } } ^ { \eta } ,$ whose noised marginals are $p _ { t } ^ { \eta }$

## C.1 CONTINUOUS DISTILLATION BASELINES: SID, FGM AND MOMENT MATCHING

Score identity distillation, flow generator matching and moment matching distill continuous diffusion models on continuous data, the case ${ \bf E } = \mathrm { I d }$ of Section 2. Each is stated in a different field — scores, velocities and conditional means — and each estimates the gradient of the same objective differently. We restate them on one path and identify what each one computes.

Path and fields. The student is deterministic, $\mathbf { X } ^ { \eta } = \mathbf { x } ^ { \eta } ( Z _ { 1 } , 1 ) \in \mathbb { R } ^ { d }$ with $Z _ { 1 } \sim p _ { 1 } = \mathrm { N } ( 0 , \mathrm { I d } )$ and its output is corrupted along the path of Section 2,

$$
Z _ { t } ^ { \eta } = \alpha _ { t } { \bf X } ^ { \eta } + \sigma _ { t } \varepsilon , \qquad \varepsilon \sim \mathrm { N ( 0 , I d ) } ,
$$

with $p _ { t } ^ { \eta }$ the law of $Z _ { t } ^ { \eta }$ and $p _ { t }$ that of the data under the same kernel $q _ { t | \mathbf { x } }$ . At each noise level the student law is described by any one of three fields, its conditional mean, its score and its velocity,

$$
\begin{array} { r l r } & { m _ { t } ^ { \eta } ( z _ { t } ) = \mathbb { E } \big [ \mathbf { X } ^ { \eta } \mid Z _ { t } ^ { \eta } = z _ { t } \big ] , } & { s _ { t } ^ { \eta } ( z _ { t } ) = \nabla _ { z _ { t } } \log p _ { t } ^ { \eta } ( z _ { t } ) , } \\ & { v _ { t } ^ { \eta } ( z _ { t } ) = \mathbb { E } \big [ \dot { \alpha } _ { t } \mathbf { X } ^ { \eta } + \dot { \sigma } _ { t } \varepsilon \mid Z _ { t } ^ { \eta } = z _ { t } \big ] , } & \end{array}
$$

and $m _ { t } , s _ { t } , v _ { t }$ denote the same fields under the data law. Tweedie’s identity, $s _ { t } ^ { \eta } = ( \alpha _ { t } m _ { t } ^ { \eta } - z _ { t } ) / \sigma _ { t } ^ { 2 }$ and its velocity counterpart, $v _ { t } ^ { \eta } = \dot { \alpha } _ { t } m _ { t } ^ { \eta } + \dot { \sigma } _ { t } \big ( z _ { t } - \alpha _ { t } m _ { t } ^ { \eta } \big ) \big / \sigma _ { t }$ , make the three differences proportional at each noise level,

$$
s _ { t } ^ { \eta } - s _ { t } = \frac { \alpha _ { t } } { \sigma _ { t } ^ { 2 } } \bigl ( m _ { t } ^ { \eta } - m _ { t } \bigr ) , \qquad v _ { t } ^ { \eta } - v _ { t } = \Bigl ( \dot { \alpha } _ { t } - \alpha _ { t } \frac { \dot { \sigma } _ { t } } { \sigma _ { t } } \Bigr ) \bigl ( m _ { t } ^ { \eta } - m _ { t } \bigr ) ,
$$

so matching scores, velocities or conditional means is one criterion under three names, up to a weight in t. We work with the score.

The objective and its two gradient terms. The squared discrepancy between the two fields, integrated over noise levels, is

$$
\mathsf { L } ( \eta ) = \int _ { 0 } ^ { 1 } w ( t ) \mathbb { E } _ { Z _ { t } \sim p _ { t } ^ { \eta } } \Bigl [ \bigl \| s _ { t } ^ { \eta } ( Z _ { t } ) - s _ { t } ( Z _ { t } ) \bigr \| ^ { 2 } \Bigr ] \mathrm { d } t .\tag{25}
$$

Here η enters twice, through the law of the state and through the field evaluated at it. Freezing one at a time gives two losses,

$$
\begin{array} { r l } & { \mathsf { L } _ { 1 } ( \eta ) = \displaystyle \int _ { 0 } ^ { 1 } w ( t ) \mathbb { E } _ { Z _ { 1 } , \varepsilon } \Big [ \big \| \mathsf { s g } ( s _ { t } ^ { \eta } ) ( Z _ { t } ^ { \eta } ) - s _ { t } ( Z _ { t } ^ { \eta } ) \big \| ^ { 2 } \Big ] \mathrm { d } t , } \\ & { \mathsf { L } _ { 2 } ( \eta ) = \displaystyle \int _ { 0 } ^ { 1 } w ( t ) \mathbb { E } _ { Z _ { t } \sim \mathrm { s g } ( p _ { t } ^ { \eta } ) } \Big [ \big \| s _ { t } ^ { \eta } ( Z _ { t } ) - s _ { t } ( Z _ { t } ) \big \| ^ { 2 } \Big ] \mathrm { d } t , } \end{array}\tag{26}
$$

where $\operatorname { s g } ( \cdot )$ freezes the parameters of a field, or of the law a state is drawn from, but not the value at which the field is read. Both equal $\mathsf { L } ( \eta )$ , and their gradients are the two terms of (5),

$$
\nabla _ { \eta } \mathsf { L } _ { 1 } = \left( A \right) , \qquad \nabla _ { \eta } \mathsf { L } _ { 2 } = \left( B \right) , \qquad \nabla _ { \eta } \mathsf { L } = \left( A \right) + \left( B \right) .
$$

For the reverse KL the second vanishes, $\begin{array} { r } { \mathbb { E } _ { p _ { t } ^ { \eta } } [ \nabla _ { \eta } \log p _ { t } ^ { \eta } ] = \nabla _ { \eta } \int p _ { t } ^ { \eta } = 0 } \end{array}$ . Here it does not, and $\mathsf { L } _ { 2 }$ cannot be differentiated as it stands: the auxiliary network estimates $s _ { t } ^ { \eta }$ at the current parameters, not its derivative $\nabla _ { \eta } { s _ { t } ^ { \eta } }$

Trading the student field for the sample that produced it. The unknown field is removed from one side of the squared norm by an identity on the student’s own path.

Lemma 1. Let $\sigma _ { t } > 0$ on the support ofw. Then $\mathsf { L } ( \eta ) = \mathsf { L } ^ { \mathrm { p r o j } } ( \eta )$ , where

$$
\lfloor \mathrm { \mathrm { { p r o j } } } ( \eta ) = \int _ { 0 } ^ { 1 } w ( t ) \mathbb { E } _ { Z _ { 1 } , \varepsilon } \Bigl [ \bigl ( s _ { t } ^ { \eta } - s _ { t } \bigr ) ^ { \top } \bigl ( s _ { t } ( \cdot \mid \mathbf { X } ^ { \eta } ) - s _ { t } \bigr ) \bigl ( Z _ { t } ^ { \eta } \bigr ) \Bigr ] \mathrm { d } t ,\tag{27}
$$

and $s _ { t } ( z _ { t } \mid x ) = \nabla _ { z _ { t } } \log q _ { t | \mathbf { x } } ( z _ { t } \mid x ) = ( \alpha _ { t } x - z _ { t } ) / \sigma _ { t } ^ { 2 }$ is the conditional score.

Proof. $\begin{array} { r } { p _ { t } ^ { \eta } ( z _ { t } ) s _ { t } ^ { \eta } ( z _ { t } ) \ = \ \nabla _ { z _ { t } } p _ { t } ^ { \eta } ( z _ { t } ) \ = \ \int \nabla _ { z _ { t } } q _ { t | \mathbf { x } } ( z _ { t } \ \vert \ x ) p _ { \mathbf { x } } ^ { \eta } ( x ) \mathrm { d } x \ = \ \int q _ { t | \mathbf { x } } ( z _ { t } \ \vert \ x ) s _ { t } ( z _ { t } \ \vert \ ) \mathrm { d } z _ { t } \ = \ \int \eta _ { t } ( z _ { t } \ \vert \ x ) p _ { t } ( z _ { t } \ \vert \ x ) p _ { t } ( x ) \mathrm { d } z _ { t } } \end{array}$ $x ) p _ { \mathbf { x } } ^ { \eta } ( x )$ dx, so that $\mathbb { E } _ { Z _ { t } \sim p _ { t } ^ { \eta } } [ f ( Z _ { t } ) ^ { \top } s _ { t } ^ { \eta } ( Z _ { t } ) ] ~ = ~ \mathbb { E } _ { Z _ { 1 } , \varepsilon } [ f ( Z _ { t } ^ { \eta } ) ^ { \top } s _ { t } ( Z _ { t } ^ { \eta } ~ | ~ \mathbf { X } ^ { \eta } ) ]$ for any squareintegrable $f .$ Take $f = s _ { t } ^ { \eta } - s _ { t }$ and subtract $\mathbb { E } [ f ^ { \top } s _ { t } ]$ from both sides. □

The marginal field on the left is unknown; the conditional field on the right is explicit, and differentiable in η through the sample that produced the state. Equation (27) is the form SiD calls projected

score matching (Zhou et al., 2024). Freezing the field in it, as in (26), isolates the part of its gradient due to the sampling operation,

$$
\mathsf { L } _ { 1 } ^ { \mathrm { p r o j } } ( \eta ) = \int _ { 0 } ^ { 1 } w ( t ) \mathbb { E } _ { Z _ { 1 } , \varepsilon } \Big [ \big ( \mathrm { s g } ( s _ { t } ^ { \eta } ) - s _ { t } \big ) ^ { \top } \big ( s _ { t } ( \cdot \vert \mathbf { X } ^ { \eta } ) - s _ { t } \big ) ( Z _ { t } ^ { \eta } ) \Big ] \mathrm { d } t ,\tag{28}
$$

where η now reaches the objective twice over, through the state and through $\mathbf { X } ^ { \eta }$ inside the conditional score. That second route is what the swap has bought: $\mathsf { L } _ { 1 } ^ { \mathrm { p r o j } }$ and L<sub>1</sub> agree in value, both being $\mathsf { L } ( \eta )$ but not in gradient, part of $( B )$ having moved into the sampling operation. The gap between them is exactly the missing term.

Proposition 2 (Huang et al., 2024). $\nabla _ { \eta } \mathsf { L } _ { 2 } ( \eta ) = 2 \nabla _ { \eta } \bigl ( \mathsf { L } _ { 1 } ^ { \mathrm { p r o j } } ( \eta ) - \mathsf { L } _ { 1 } ( \eta ) \bigr )$

FGM states this in the velocity field and proves it from the identity of Lemma 1 (Huang et al., 2024). The exact gradient of (25) is therefore the gradient of a computable loss,

$$
\nabla _ { \eta } \mathsf { L } ( \eta ) = \nabla _ { \eta } \bigl ( 2 \mathsf { L } _ { 1 } ^ { \mathrm { p r o j } } ( \eta ) - \mathsf { L } _ { 1 } ( \eta ) \bigr ) ,\tag{29}
$$

in which the student field appears only frozen, at the current parameters, which is what the auxiliary network provides.

What each method computes. The three baselines descend $\mathsf { L } _ { 1 } ^ { \mathrm { p r o j } }$ and $\mathsf { L } _ { 1 }$ in different proportions. FGM takes the combination (29) and so descends the exact gradient (Huang et al., 2024). SiD descends $\mathsf { L } _ { 1 } ^ { \mathrm { p r o j } } - \alpha \mathsf { L }$ <sub>1</sub> for a scalar α (Zhou et al., 2024), whose gradient is, by Proposition 2,

$$
\nabla _ { \eta } \bigl ( \mathsf { L } _ { 1 } ^ { \mathrm { p r o j } } - \alpha \mathsf { L } _ { 1 } \bigr ) = \bigl ( 1 - \alpha \bigr ) ( A ) + \textstyle { \frac { 1 } { 2 } } \left( B \right) .
$$

FGM is therefore SiD at $\alpha = 1 / 2$ , doubled, the factor being absorbed in the step size. The values reported as best for SiD, $\alpha \in [ 0 . 7 5 , 1 . 2 ]$ (Zhou et al., 2024), sit past that point; at $\alpha = 1$ the pathwise term disappears and only $\textstyle { \frac { 1 } { 2 } } ( B )$ is left, and beyond it the pathwise term is descended with the opposite sign. The one-step version of moment matching descends (28), the case $\alpha = 0$ , with $Z _ { t } ^ { \eta }$ frozen in addition (Salimans et al., 2024). The three are one family, read off a single pair of losses.

## C.2 DISCRETE DISTILLATION AS A BREGMAN DIVERGENCE BETWEEN CONCRETE SCORES

D-MMD (Hoogeboom et al., 2026) and IDLM (Li et al., 2026) distill discrete diffusion models, and their few-step constructions differ: D-MMD draws the next state from the bridge, IDLM rather uses the forward process. At one step they descend the same objective, derived once here.

The objective. The student draws $Z _ { 1 } \sim p _ { 1 }$ and emits $\mathbf { X } ^ { \eta } \sim \mathsf { k } _ { 1 } ^ { \eta } ( \cdot \mid Z _ { 1 } )$ , of sequence law $p _ { \mathbf { x } } ^ { \eta }$ , and the criterion is the reverse Kullback–Leibler divergence K $\varphi _ { \mathbf { x } } ^ { \eta } \parallel p _ { \mathrm { d a t a } } )$ . Diffuse both laws along (15). Their reverse chains are Markov on the same grid and agree at $t = 1$ , so (18) holds with the two exchanged,

$$
\displaystyle \mathrm { K L } ( p _ { \mathbf { x } } ^ { \eta } \parallel p _ { \mathrm { d a t a } } ) \leq \sum _ { i = 1 } ^ { n } \mathbb { E } _ { Z _ { t _ { i } } \sim p _ { t _ { i } } ^ { \eta } } [ \mathrm { K L } ( p _ { s _ { i } | t _ { i } } ^ { \eta } ( \cdot \mid Z _ { t _ { i } } )  p _ { s _ { i } | t _ { i } } ( \cdot \mid Z _ { t _ { i } } ) ) ] ,
$$

and Proposition 1 applies to both sides, each token marginal being a bridge with a leave-one-out denoiser plugged in — the student’s on the left, the data’s on the right. The passage to continuous time of Section B.2 runs unchanged and returns a Bregman divergence between the two concrete scores, read at the states the student visits,

$$
\mathrm { K L } ( p _ { \mathbf { x } } ^ { \eta } \parallel p _ { \mathrm { d a t a } } ) \leq \int _ { 0 } ^ { 1 } R _ { t } \mathbb { E } _ { Z _ { t } \sim p _ { t } ^ { \eta } } \biggl [ \sum _ { \ell = 1 } ^ { L } \langle \pi , Z _ { t } ^ { \ell } \rangle \sum _ { e _ { k } \neq Z _ { \ell } ^ { \ell } } d _ { F } \Bigl ( s _ { t } ^ { \eta , \ell } ( Z _ { t } ) _ { k } , ~ s _ { t } ^ { \ell } ( Z _ { t } ) _ { k } \Bigr ) \biggr ] \mathrm { d } t .\tag{30}
$$

Neither score is available in distillation: the teacher stands in for $s _ { t } ,$ and an auxiliary model, fitted on the student’s own samples, for $s _ { t } ^ { \eta }$

Why a difference of two training losses computes it. Neither method evaluates (30). Both subtract two per-sample training losses of the kind Section B.2 ends on, evaluated on student samples. For a simplex-valued field and the concrete score sˆ it induces through (23), that loss is (Lou et al., 2024)

$$
\ell _ { t } ( \hat { s } ; z _ { t } , \mathbf { x } ) = \sum _ { \ell = 1 } ^ { L } \big \langle \pi , z _ { t } ^ { \ell } \big \rangle \sum _ { e _ { k } \neq z _ { t } ^ { \ell } } \Big [ \hat { s } ^ { \ell } ( z _ { t } ) _ { k } - s _ { t } ^ { \ell } ( z _ { t } \mid \mathbf { x } ) _ { k } \log \hat { s } ^ { \ell } ( z _ { t } ) _ { k } \Big ] ,
$$

the integrand of (24) up to a term free of ${ \hat { s } } { \mathrm { : } }$ an evidence bound rather than a divergence, minimized over $\mathbf { X } { \ ' } \sim p _ { \mathbf { x } \mid t } ^ { \eta } ( { \cdot } \mid z _ { t } )$ at $\hat { s } = s _ { t } ^ { \eta }$ . Subtracting two of them removes that free term and leaves the divergence,

$$
\mathbb { E } _ { \mathbf { X } \sim p _ { \mathrm { x } | t } ^ { \eta } ( \cdot | z _ { t } ) } \Big [ \ell _ { t } ( s _ { t } ^ { \theta } ; z _ { t } , \mathbf { X } ) - \ell _ { t } ( s _ { t } ^ { \eta } ; z _ { t } , \mathbf { X } ) \Big ] = \sum _ { \ell = 1 } ^ { L } \big \langle \pi , z _ { t } ^ { \ell } \big \rangle \sum _ { e _ { k } \neq z _ { t } ^ { \ell } } d _ { F } \Big ( s _ { t } ^ { \eta , \ell } ( z _ { t } ) _ { k } , ~ s _ { t } ^ { \theta , \ell } ( z _ { t } ) _ { k } \Big ) ,
$$

which is the integrand of (30) with the teacher in place of the data. IDLM writes it as the teacher’s loss on student samples minus its minimum over auxiliary fields (Li et al., 2026); D-MMD adds a term pulling the auxiliary towards the teacher, which leaves the fixed point where it is (Hoogeboom et al., 2026). Under the mask this divergence takes a simpler form still, derived in Section C.3.

## C.3 MASKED DIFFUSION DISTILLATION: COLLAPSE TO A POSTERIOR KL

Take $\pi = e _ { m }$ , the point mass on the mask symbol. The weight $\langle \pi , z _ { t } ^ { \ell } \rangle$ of (30) is then 1 at masked positions and 0 elsewhere, so only masked positions are scored. At one of them, for every $k \neq m ,$ the concrete score is the denoiser up to a factor that depends on t alone,

$$
s _ { t } ^ { \ell } ( z _ { t } ) _ { k } = \frac { \alpha _ { t } } { 1 - \alpha _ { t } } m _ { t } ^ { \ell } ( z _ { t } ) _ { k } ,
$$

where $m _ { t } ^ { \ell } ( z _ { t } ) = \mathbb { E } [ { \mathbf { X } } ^ { \ell } \mid Z _ { t } = z _ { t } ]$ is the data denoiser, equal at masked positions to the leave-one-out denoiser of Proposition 1. The same holds for the student score with $m _ { t } ^ { \eta , \ell } ( z _ { t } ) = \mathbb { E } [ \mathbf { X } ^ { \eta , \ell } \mid Z _ { t } ^ { \eta } = z _ { t } ]$ the denoiser of the noised student distribution, not the student generator. Replacing the student and data scores in (30) accordingly leaves the loss

$$
\mathrm { K L } ( p _ { \mathbf { x } } ^ { \eta } \parallel p _ { \mathrm { d a t a } } ) \leq \int _ { 0 } ^ { 1 } \frac { - \alpha _ { t } ^ { \prime } } { 1 - \alpha _ { t } } \mathbb { E } _ { Z _ { t } \sim p _ { t } ^ { \eta } } \bigg [ \sum _ { \ell : Z _ { t } ^ { \ell } = e _ { m } } \mathrm { K L } \Big ( m _ { t } ^ { \eta , \ell } ( Z _ { t } ) \left. m _ { t } ^ { \ell } ( Z _ { t } ) \right) \Big ] \mathrm { d } t ,
$$

where $m _ { t } ^ { \eta }$ and $m _ { t }$ are approximated in practice by an auxiliary denoiser and the teacher $\mathbf { x } ^ { \theta }$ , yielding the discrepancy Table 1 lists for DiMO (Zhu et al., 2025) and the loss D-MMD implements in its masked instance (Hoogeboom et al., 2026). Nothing of the sort happens under the uniform law, where every position is scored and the denominator of (22) keeps the head in it.

## D METHOD: DERIVATIONS AND PROOFS

The derivations behind Section 3: the gradient the objective is trained with, the proof that the vertex constraint is recovered, the discriminator that supplies the log-ratio, and the alternative objective we distill as a baseline.

## D.1 DISTRIBUTION MATCHING DISTILLATION: GRADIENT DERIVATION

We derive the pathwise gradient used by Simplex-DMD in (9). The key observation is that differentiating the reverse KL produces both a contribution through the generated latent and an explicit density derivative; the latter has zero expectation. Throughout, the embedding matrix E, the noise schedule, and the sampling distribution ν are fixed. We assume sufficient regularity to interchange differentiation and integration.

Reparameterized student samples. For $( t , T ) \sim \nu ,$ , draw $Z _ { T } \sim p _ { T }$ and independent noise $\varepsilon \sim \mathrm { N } ( 0 , \mathrm { I d } )$ . Under the continuous relaxation,

$$
{ \bf X } ^ { \eta , T } = { \bf x } ^ { \eta } ( Z _ { T } , T ) , \qquad Z _ { t } ^ { \eta , T } = \pi _ { t } ( { \bf X } ^ { \eta , T } , \varepsilon ) = \alpha _ { t } { \bf x } ^ { \eta } ( Z _ { T } , T ) { \bf E } + \sigma _ { t } \varepsilon .
$$

This construction realizes the marginal $p _ { t } ^ { \eta , T }$ on a sampling space whose law does not depend on η. The multi-step objective is therefore

$$
\mathsf { L } _ { \mathrm { D M D } } ^ { \mathrm { m s } } ( \eta ) = \mathbb { E } _ { \boldsymbol { Z } _ { T } \sim p _ { T } , \boldsymbol { \varepsilon } \sim \mathrm { N } ( \boldsymbol { 0 } , \mathrm { I d } ) } \left[ \log \frac { p _ { t } ^ { \eta , T } ( Z _ { t } ^ { \eta , T } ) } { p _ { t } ( Z _ { t } ^ { \eta , T } ) } \right] .
$$

Differentiating the reverse KL. Fix (t, T) and write $s ^ { \eta , T } ( z , t ) = \nabla _ { z } \log p _ { t } ^ { \eta , T } ( z )$ and $s ( z , t ) =$ $\nabla _ { z } \log { p _ { t } ( z ) }$ . Holding $Z _ { T }$ and ε fixed, the chain rule gives

$$
\nabla _ { \boldsymbol { \eta } } \log \frac { p _ { t } ^ { \eta , T } ( Z _ { t } ^ { \eta , T } ) } { p _ { t } ( Z _ { t } ^ { \eta , T } ) } = \Big ( s ^ { \eta , T } ( Z _ { t } ^ { \eta , T } , t ) - s ( Z _ { t } ^ { \eta , T } , t ) \Big ) ^ { \top } \frac { \partial Z _ { t } ^ { \eta , T } } { \partial \eta }
$$

$$
+ \left. \nabla _ { \eta } \log p _ { t } ^ { \eta , T } ( z ) \right| _ { z = Z _ { t } ^ { \eta , T } } .
$$

In the last term, the density is differentiated at a fixed argument z. Its expectation vanishes because $p _ { t } ^ { \eta , T }$ integrates to one:

$$
\begin{array} { r l } { \mathbb { E } _ { Z _ { t } ^ { \eta , T } \sim p _ { t } ^ { \eta , T } } \left[ \left. \nabla _ { \eta } \log p _ { t } ^ { \eta , T } ( z ) \right| _ { z = Z _ { t } ^ { \eta , T } } \right] = \int _ { \mathbb { R } ^ { L \times d } } \nabla _ { \eta } p _ { t } ^ { \eta , T } ( z ) \mathrm { d } z } & { } \\ { = \nabla _ { \eta } 1 = 0 . } \end{array}
$$

This is the cancellation of contribution (B) in (5); it also holds for the categorical student, although that parameterization does not admit the pathwise derivative.

Averaging the remaining term yields

$$
\nabla _ { \eta } \mathrm { L } _ { \mathrm { D M D } } ^ { \mathrm { m s } } ( \eta ) = \mathbb { E } \frac { ( t , T ) \sim \nu } { Z _ { T } \sim p _ { T } , \varepsilon \sim \mathrm { N } ( 0 , \mathrm { I d } ) } \left[ \left( s ^ { \eta , T } ( Z _ { t } ^ { \eta , T } , t ) - s ( Z _ { t } ^ { \eta , T } , t ) \right) ^ { \top } \frac { \partial Z _ { t } ^ { \eta , T } } { \partial \eta } \right] ,
$$

which is (9). Latent coordinates are flattened when forming the score–Jacobian product. The latent Jacobian includes the embedding map and the signal coefficient:

$$
\frac { \partial Z _ { t } ^ { \eta , T } } { \partial \eta } = \alpha _ { t } \frac { \partial [ \mathbf { x } ^ { \eta } ( Z _ { T } , T ) \mathbf { E } ] } { \partial \eta } .
$$

The one-step gradient follows by fixing $T = 1$ , sampling t uniformly, and identifying $p _ { t } ^ { \eta , 1 }$ with $p _ { t } ^ { \eta }$

Estimated scores. The identity above uses exact scores. In practice, the frozen teacher and the auxiliary denoiser provide their estimates through (10). During a student update, the estimated score difference is detached and gradients flow only through the reparameterized latent. Thus the implemented update approximates the exact pathwise gradient, with accuracy determined by the score estimates.

## D.2 RECOVERING THE DISCRETE DATA DISTRIBUTION

This subsection proves Theorem 1: matching the embedded, noised marginals recovers the clean data distribution, even when the student outputs continuous probability vectors. We then explain how the same argument identifies the transports from noisy data marginals to the clean data distribution used for multi-step generation.

Clean and noised distributions. We use the clean student law p<sup>η</sup> from Section 3.1 and identify discrete sequences with the one-hot vertex set $V = \{ \mathbf { e } _ { 1 } , \dots , \mathbf { e } _ { K } \} ^ { \bar { L } }$ of $\Delta _ { K } ^ { L }$ . The data law $p _ { \mathrm { d a t a } }$ is supported on V, whereas a relaxed student can place mass throughout $\Delta _ { K } ^ { L }$ . For a clean distribution $\rho$ on this simplex, let $Q _ { t } \rho$ denote the law of

$$
Z _ { t } = \alpha _ { t } { \bf X } { \bf E } + \sigma _ { t } \varepsilon , ~ { \bf X } \sim \rho , ~ \varepsilon \sim \mathrm { N ( 0 , I d ) } ,
$$

with independent Gaussian noise. Thus $Q _ { t } p _ { \mathbf { x } } ^ { \eta } = p _ { t } ^ { \eta }$ and $Q _ { t } p _ { \mathrm { d a t a } } = p _ { t }$ . Writing the objective as a function of the clean law gives

$$
\mathsf { L } _ { \mathrm { D M D } } ( \rho ) = \int _ { 0 } ^ { 1 } \mathrm { K L } ( Q _ { t } \rho \parallel p _ { t } ) \mathrm { d } t , \qquad \mathsf { L } _ { \mathrm { D M D } } ( \eta ) = \mathsf { L } _ { \mathrm { D M D } } ( p _ { \mathbf { x } } ^ { \eta } ) ,
$$

which is exactly (6) for the student law. Here $\rho$ denotes a clean distribution; ν retains its meaning as the time-sampling distribution in the multi-step objective. For the proofs, write $\mathcal { E } ( \mathbf { x } ) = \mathbf { x } \mathbf { E }$ for the position-wise embedding map. We assume that $\alpha _ { t }$ and $\sigma _ { t }$ are measurable, finite, and strictly positive for $t \in ( 0 , 1 )$ ); endpoint values do not affect the integral, and the loss may be infinite.

Why equal-norm embeddings suffice. Constraining the token embeddings to a common sphere is standard in the lineage of Section 2: CDCD L -normalizes them before every use and backpropagates through the normalization (Dieleman et al., 2022), LangFlow places them on a sphere of radius $\sqrt { d }$ (Chen et al., 2026b), and RePlaid constrains each row to unit length (Yang et al., 2026). If the rows are also distinct, this normalization ensures that no token embedding is a convex combination of the others, as the following proposition shows.

Proposition 3. Distinct rows sharing a common Euclidean norm are in convex position.

Proof. Let $\| E _ { k } \| = R$ for all k and suppose $\begin{array} { r } { E _ { i } = \sum _ { k \neq i } a _ { k } E _ { k } } \end{array}$ with $a \in \Delta _ { K - 1 }$ . For $k \neq i .$ $\| E _ { i } - E _ { k } \| ^ { 2 } = 2 R ^ { 2 } - 2 \langle E _ { i } , E _ { k } \rangle$ is positive by distinctness, so $\langle E _ { i } , E _ { k } \rangle < R ^ { 2 }$ . Taking the inner product of the supposed combination with $E _ { i }$ gives $\begin{array} { r } { R ^ { 2 } = \sum _ { k \neq i } \dot { a _ { k } } \big \langle \dot { E _ { i } } , \dot { E _ { k } } \big \rangle < R ^ { 2 } } \end{array}$ □

More generally, the argument below only requires distinct rows in convex position; it does not require the embedding map to be injective on the whole simplex.

Proof of the theorem. We first show that a token embedding can only come from its one-hot vector.   
We then undo Gaussian noising to identify the clean distribution.

Lemma 2. Under the assumption of Theorem 1, $x \in \Delta _ { K }$ and $x { \bf E } = E _ { i }$ imply $x = \mathbf { e } _ { i }$ . Hence $\mathcal { E } ^ { - 1 } ( \mathcal { E } ( V ) ) = V$ , and E is injective on V .

Proof. From $\begin{array} { r } { E _ { i } = x _ { i } E _ { i } + \sum _ { k \neq i } x _ { k } E _ { k } , \mathrm { i f } x _ { i } < 1 } \end{array}$ then dividing by $1 - x _ { i }$ gives $\begin{array} { r } { E _ { i } = \sum _ { k \neq i } \frac { x _ { k } } { 1 - x _ { i } } E _ { k } } \end{array}$ a convex combination of the other rows, contradicting convex position. Hence $x _ { i } = 1$ . Applying this at every position gives the second claim, and injectivity on $\mathbf { \bar { \rho } } _ { V }$ follows from distinctness of the rows. □

Lemma 3 (Identification at one noise level). Under the assumption ofTheorem 1,for any probability measure ρ on $\Delta _ { K } ^ { L }$ and any $t \in ( 0 , 1 ) , Q _ { t } \rho = Q _ { t } p _ { \mathrm { d a t a } }$ holds ifand only $i f \rho = p _ { \mathrm { d a t a } } .$

Proof. Only the forward implication needs an argument. Both measures are Gaussian smoothings, so their characteristic functions factor as $\widehat { ( \alpha _ { t } \mathcal { E } ) _ { \# } } \rho \cdot \hat { g }$ with $\hat { g } ( u ) = e ^ { - \sigma _ { t } ^ { 2 } \| u \| ^ { 2 } / 2 }$ nowhere zero. Dividing by $\hat { g }$ and using $\alpha _ { t } \neq 0$ gives $\mathcal { E } _ { \# } \rho = \mathcal { E } _ { \# } p _ { \mathrm { d a t a } }$ . That measure is carried by ${ \mathcal { E } } ( V )$ , so $\rho$ is carried by $\mathcal { E } ^ { - 1 } ( \mathcal { E } ( V ) ) = V$ by Lemma 2. On that finite set $\mathcal { E }$ is injective, so equality of the embedded laws is equality of the mass at every vertex sequence. □

Proof of Theorem 1. We prove that $\mathsf { L } _ { \mathrm { D M D } } ( \rho ) = 0$ if and only if $\rho ~ = ~ p _ { \mathrm { d a t a } }$ , for any probability measure $\rho$ on $\Delta _ { K } ^ { L }$ ; the theorem is the case $\rho = p _ { \mathbf { x } } ^ { \eta }$ . The reverse implication is immediate, $Q _ { t } \rho$ being a function of ρ alone. Conversely, nonnegativity of the KL divergence forces $Q _ { t } \rho = Q _ { t } p _ { \mathrm { d a t a } }$ for almost every $t \in ( 0 , 1 )$ , and any such t with Lemma 3 closes the argument. □

Implications for multi-step generation. In Section $3 . 2 ,$ , the student starts from $Z _ { T } \sim p _ { T }$ and produces a clean sample with law $p _ { \mathbf { x } } ^ { \eta , T }$ . For each fixed $T > 0 ,$ , Lemma 3 applies to this law just as in the one-step case: matching $p _ { t } ^ { \eta , T }$ to $p _ { t }$ at any noise level $0 < t < T$ implies $p _ { \mathbf { x } } ^ { \eta , T } = p _ { \mathrm { d a t a } }$ Consequently, a zero multi-step objective (7) gives, for $\nu _ { 1 }$ -almost every starting level $T$ ,

$$
\int \mathsf { k } _ { T } ^ { \eta } ( \mathrm { d } \mathbf { x } \mid z _ { T } ) p _ { T } ( \mathrm { d } z _ { T } ) = p _ { \mathrm { d a t a } } ( \mathrm { d } \mathbf { x } ) .
$$

Indeed, nonnegativity of the KL divergence implies matching for $\nu _ { T }$ -almost every $t < T$ , and any such level identifies the clean law. Thus the objective learns a family of transports from $p _ { T } \ \mathrm { t o } p _ { \mathrm { d a t a } } ,$ indexed by the starting noise level.

## D.3 THE OPTIMAL DISCRIMINATOR AND THE LOG-RATIO

We show that the population optimum of (12) recovers the log-density ratio required by Reinforce-DMD. This is the standard optimal discriminator argument of Goodfellow et al. (2014), applied to the noised data and student distributions at each pair (t, T).

Proposition 4. Fix the student parameters η and a pair $0 < t < T \leq 1$ with $\sigma _ { t } \ > \ 0$ . Over measurablefunctions $D : \mathbb { R } ^ { L \times d } \overset { \cdot } {  } ( 0 , 1 )$ , the objective

$$
\ell _ { t , T } ( D ) = - \mathbb { E } _ { Z _ { t } \sim p _ { t } } [ \log D ( Z _ { t } ) ] - \mathbb { E } _ { Z _ { t } ^ { \eta , T } \sim p _ { t } ^ { \eta , T } } \Big [ \log \big ( 1 - D ( Z _ { t } ^ { \eta , T } ) \big ) \Big ]
$$

has the unique minimizer, up to sets of Lebesgue measure zero,

$$
D ^ { \star } ( z , t , T ) = \frac { p _ { t } ( z ) } { p _ { t } ( z ) + p _ { t } ^ { \eta , T } ( z ) } .
$$

Consequently,

$$
\log \frac { 1 - D ^ { \star } ( z , t , T ) } { D ^ { \star } ( z , t , T ) } = \log \frac { p _ { t } ^ { \eta , T } ( z ) } { p _ { t } ( z ) } .
$$

Proof. Gaussian noising with $\sigma _ { t } > 0$ makes both densities strictly positive. Writing the objective as

$$
\ell _ { t , T } ( D ) = \int _ { \mathbb { R } ^ { L \times d } } \left[ - p _ { t } ( z ) \log D ( z ) - p _ { t } ^ { \eta , T } ( z ) \log \left( 1 - D ( z ) \right) \right] \mathrm { d } z ,
$$

we can minimize the integrand separately at each z. Set $a = p _ { t } ( z ) > 0$ and $b = p _ { t } ^ { \eta , T } ( z ) > 0$ . For $u \in ( 0 , 1 )$ ), let $f ( u ) = - a \log u - b \log ( 1 - u )$ . Its derivatives are

$$
f ^ { \prime } ( u ) = - \frac { a } { u } + \frac { b } { 1 - u } , \qquad f ^ { \prime \prime } ( u ) = \frac { a } { u ^ { 2 } } + \frac { b } { ( 1 - u ) ^ { 2 } } > 0 .
$$

Thus $f$ is strictly convex, and its unique minimizer solves $f ^ { \prime } ( u ) = 0 \ :$ , giving $u = a / ( a + b )$ Substitution yields $D ^ { \star }$ and $( 1 - D ^ { \star } ) / \bar { D ^ { \star } } = b / a$ , which proves the log-ratio identity. □

Averaging these objectives over $( t , T ) \sim \nu$ gives (12). Hence an unrestricted discriminator conditioned on both noise levels has the stated optimum for ν-almost every pair.

Leave-one-out baseline. For a fixed discriminator, the leave-one-out baseline $\widehat { b } _ { g }$ is independent of $\mathbf { X } _ { q } ^ { \eta , T }$ conditionally on $( Z _ { T } , T )$ . Since the conditional score function has zero expectation, subtracting this baseline preserves the expected cost-based update (Kool et al., 2019).

## D.4 AUXILIARY SCORE ESTIMATION WITH BRIDGE RENOISING

The multi-step formulation in Section 3.2 can also use the bridge $q _ { t | T , \mathbf { x } }$ in place of forward noising. For Simplex-DMD, consider $Z _ { T } \sim p _ { T }$ and $Z _ { t } \sim q _ { t | T , \mathbf { x } } ( \cdot \mid Z _ { T } , \mathbf { x } ^ { \eta } ( \dot { Z } _ { T } , T ) )$ ), and denote the resulting marginal by $p _ { t } ^ { \eta , T \to t }$ . Taking the conditional expectation of the Gaussian bridge score, with the coefficients of (1), gives

$$
\nabla _ { z _ { t } } \log p _ { t } ^ { \eta , T  t } ( z _ { t } ) = \frac { \alpha _ { t } \sigma _ { T | t } ^ { 2 } \mathbb { E } [ \mathbf { x } ^ { \eta } ( Z _ { T } , T ) \mid Z _ { t } = z _ { t } ] \mathbf { E } + \alpha _ { T | t } \sigma _ { t } ^ { 2 } \mathbb { E } [ Z _ { T } \mid Z _ { t } = z _ { t } ] - \sigma _ { T } ^ { 2 } z _ { t } } { \sigma _ { t } ^ { 2 } \sigma _ { T | t } ^ { 2 } } ,
$$

for positive bridge variance, where both expectations are under this joint law. Unlike forward noising, the exact bridge score therefore involves the additional conditional mean $\mathbb { E } [ Z _ { T } \mid Z _ { t } ]$ , which is not estimated by the auxiliary token denoiser.

In practice, we retain the usual auxiliary score estimator by approximating the conditional law of $Z _ { T }$ given $Z _ { t } = z _ { t }$ with the forward Gaussian transition $\mathcal { N } ( \dot { \alpha } _ { T | t } z _ { t } , \sigma _ { T | t } ^ { 2 } \mathrm { I d } )$ . Although this conditional law does not generally hold under bridge renoising, it gives $\mathbb { E } [ \dot { Z } _ { T } \mid Z _ { t } = z _ { t } ] \approx \alpha _ { T | t } z _ { t }$ . Using $\sigma _ { T } ^ { 2 } = \alpha _ { T | t } ^ { 2 } \sigma _ { t } ^ { 2 } + \sigma _ { T | t } ^ { 2 }$ , the score expression then reduces to the usual Tweedie form,

$$
\nabla _ { z _ { t } } \log p _ { t } ^ { \eta , T  t } ( z _ { t } ) \approx \frac { \alpha _ { t } \mathbb { E } [ { \mathbf { x } } ^ { \eta } ( Z _ { T } , T ) \mid Z _ { t } = z _ { t } ] \mathbf { E } - z _ { t } } { \sigma _ { t } ^ { 2 } } ,
$$

whose conditional token mean is estimated by the auxiliary denoiser. Thus, retaining this estimator under bridge renoising introduces an approximation, whereas it follows exactly from the forwardnoising construction.

Estimating the exact bridge score would require an additional prediction task for $\mathbb { E } [ Z _ { T } \mid Z _ { t } ]$ , for example through a separate network or output head, or an auxiliary model predicting the combined bridge mean directly. We leave such estimators to future work. Related experiments on continuous data, in Appendix D of Salimans et al. (2024), reported poor results when fitting the exact bridge score. We do not evaluate bridge renoising during Reinforce-DMD training.

## D.5 MULTI-STEP GENERATION

We justify here the DDIM-inspired family of transitions introduced in Section 3.2. The following proposition gives two conditions under which a transition preserves the exact diffusion marginals.

Proposition 5. Fix $0 \leq t < T \leq 1$ and suppose that $\widehat { Z } _ { T } \sim p _ { T }$ . Given $\widehat { Z } _ { T } ,$ , sample

$$
\widehat { \mathbf { X } } \sim \mathsf { k } _ { T } ^ { \eta } ( \cdot \mid \widehat { Z } _ { T } ) ,
$$

and then

$$
\widehat { Z } _ { t } \sim q _ { t | \mathbf { x } , T } ^ { \zeta } ( \cdot \mid \widehat { \mathbf { X } } , \widehat { Z } _ { T } ) .
$$

Then either of the following conditions implies $\widehat { Z } _ { t } \sim p _ { t }$

1. $\zeta _ { t , T } = \sigma _ { t }$ and the marginal distribution of $\widehat { \mathbf X }$ is $p _ { \mathrm { d a t a } } ,$

2. $\mathsf { k } _ { T } ^ { \eta } ( \cdot \mid z _ { T } ) = p _ { \mathbf { x } | T } ( \cdot \mid z _ { T } )$ for p -almost every $z _ { T }$ . In this case the conclusion holds for every $0 \leq \zeta _ { t , T } \leq \sigma _ { t }$

Proof. For $\zeta _ { t , T } = \sigma _ { t }$ , the coefficient multiplying the noise inferred from $\widehat { Z } _ { T }$ vanishes, and the transition becomes

$$
\widehat { Z } _ { t } = \alpha _ { t } \widehat { \mathbf { X } } \mathbf { E } + \sigma _ { t } \varepsilon , \qquad \varepsilon \sim \mathrm { N ( 0 , I d ) } .
$$

If $\widehat { \mathbf { X } } \sim p _ { \mathrm { d a t a } }$ , this is exactly the forward construction defining $p _ { t }$ .

For the second statement, sampling $\widehat { \mathbf X }$ from the exact posterior given $\widehat { Z } _ { T } \sim p _ { T }$ reconstructs the exact joint law of $( \mathbf { X } , Z _ { T } )$ under the forward process. Hence we may write

$$
\widehat { Z } _ { T } = \alpha _ { T } \widehat { \mathbf { X } } \mathbf { E } + \sigma _ { T } \varepsilon _ { T } , \qquad \varepsilon _ { T } \sim \mathrm { N ( 0 , I d ) } ,
$$

with $\varepsilon _ { T }$ independent of $\widehat { \mathbf { X } } . \mathbf { A }$ draw from (8) can therefore be written as

$$
\widehat { Z } _ { t } = \alpha _ { t } \widehat { \mathbf { X } } \mathbf { E } + \sqrt { \sigma _ { t } ^ { 2 } - \zeta _ { t , T } ^ { 2 } } \varepsilon _ { T } + \zeta _ { t , T } \varepsilon ,
$$

where $\varepsilon \sim \mathrm { N } ( 0 , \mathrm { I d } )$ is independent of $( \widehat { \mathbf { X } } , \varepsilon _ { T } )$ . The two Gaussian noise terms combine into a $\mathrm { N } ( 0 , \sigma _ { t } ^ { 2 } \operatorname { I d } )$ variable independent of $\widehat { \mathbf { X } }$ , so that

$$
\widehat { Z } _ { t } \mid \widehat { \mathbf { X } } \sim \mathrm { N } \big ( \alpha _ { t } \widehat { \mathbf { X } } \mathbf { E } , \sigma _ { t } ^ { 2 } \mathrm { I d } \big ) .
$$

Since $\widehat { \mathbf { X } } \sim p _ { \mathrm { d a t a } } .$ , marginalizing over $\widehat { \mathbf X }$ gives $\widehat { Z } _ { t } \sim p _ { t }$

Applying Proposition 5 recursively along a decreasing grid shows that, if its assumptions hold at every transition, then

$$
\widehat { Z } _ { T _ { i } } \sim p _ { T _ { i } } , \qquad i = 0 , \ldots , n .
$$

In particular, for forward renoising, exact distribution matching at each starting level implies the required clean-marginal condition under the geometric assumption of Theorem 1.

## E SUPPLEMENTARY EXPERIMENTS

## E.1 IMPLEMENTATION DETAILS

Training budgets and model sizes. Table 3 separates teacher pretraining from the additional updates used to train the distilled students. Both the student and auxiliary models retain the LangFlow architecture (Chen et al., 2026b); we use an embedding-space diffusion transformer with 12 blocks, hidden dimension 768, 12 attention heads and rotary position embeddings, conditioned on the noise level through adaLN modulation, with an output head over the vocabulary that is not tied to E. Each network has 170.8M parameters, of which the 38.6M of E are frozen and shared with the teacher. For Reinforce-DMD, the discriminator keeps this trunk and replaces the vocabulary head with a scalar output per position. Optimization-step counts alone do not imply equal training compute: batch sizes and the number of network evaluations per update can differ, and our methods alternate auxiliary and student updates. Each student is trained for 10,000 optimizer updates in total, comprising 8,000 auxiliary updates and 2,000 student updates. Our training GPU budget comprises four nodes with eight NVIDIA H100 GPUs each (32 GPUs in total).

<table><tr><td>Method</td><td>Teacher</td><td>Teacher updates</td><td>Distillation updates</td><td>Student params.</td></tr><tr><td>SDTT (Deschenaux &amp; Gulcehre, 2025)</td><td>MDLM</td><td>1M</td><td>r × 10k  $r = 1 , \ldots , 7$ </td><td>≈170M</td></tr><tr><td>IDLM (Li et al., 2026)</td><td>MDLM</td><td>1M</td><td>1M</td><td>169.63M</td></tr><tr><td>FMLM (Lee et al., 2026)</td><td>FLM</td><td>1.5M</td><td>1M</td><td>≈170M</td></tr><tr><td>ReDi (Yoo et al., 2025)</td><td>DUO</td><td>~500k</td><td>≤1M</td><td>≈165M</td></tr><tr><td>Simplex-DMD (ours)</td><td>LangFlow</td><td>1M</td><td>10k</td><td>≈170M</td></tr><tr><td>Reinforce-DMD (ours)</td><td>LangFlow</td><td>1M</td><td>10k</td><td>≈170M</td></tr></table>

Table 3: Training budgets and model sizes for the distilled checkpoints. Teacher updates are pretraining steps; distillation updates are additional optimizer updates. For our methods, the 10k total includes 8k auxiliary and 2k student updates. SDTT uses successive round checkpoints, indexed by r. The ReDi student budget is an upper bound. Parameter counts refer to one sampling model, including embeddings, and exclude auxiliary training networks.

For SDTT, the teacher is the MDLM teacher released with SDTT; IDLM uses kuleshov-group/mdlm-owt and a learning rate of 10<sup>−6</sup>. Both teachers are MDLM models (Sahoo et al., 2024). The FMLM checkpoint is distilled from its flow language model (FLM) teacher (Lee et al., 2026). ReDi uses the DUO (Sahoo et al., 2025) checkpoint s-sahoo/duo (duo.ckpt). Both of our students use the same released LangFlow checkpoint (Chen et al., 2026b). For all the baselines, we use their default samplers presented in their associated paper.

Training and sampling configuration. Table 4 summarizes the training and sampling configuration of each student; the ablations of Section 4.2 and below vary its components. At inference, we divide the temporal interval [0, 1] uniformly, using $T _ { i } = i / n \mathrm { f o r } i = 1 , . . . , n .$ , and traverse this grid from $T _ { n } = \bar { 1 \mathrm { t o } T _ { 1 } }$ and apply a final discrete decoder step at time $T _ { 1 }$ , yielding n total NFEs. Sampling temperature is applied at the network output: we divide the token logits by the temperature before the softmax that produces the clean token probabilities.

Discriminator architecture. The Reinforce-DMD discriminator uses the same transformer backbone as the teacher, including rotary positional embeddings and noise-level conditioning through adaptive layer normalization. It operates directly on noised latents in the teacher’s embedding space, omitting the token-embedding layer and replacing the vocabulary prediction head with a scalar head. We initialize the backbone from the teacher checkpoint and initialize the scalar head separately. The head applies noise-conditioned adaptive layer normalization followed by a projection to one logit per position; averaging these logits over sequence positions yields a single scalar logit per sequence.

<table><tr><td>Component</td><td>Simplex-DMD</td><td>Reinforce-DMD</td></tr><tr><td>Student initialization</td><td>teacher</td><td>teacher</td></tr><tr><td>Auxiliary initialization</td><td>teacher</td><td>teacher</td></tr><tr><td>Renoising kernel</td><td>qt|x</td><td>qt|x</td></tr><tr><td>Self-conditioning</td><td>on</td><td>off</td></tr><tr><td>KL anchor β</td><td></td><td>0.5</td></tr><tr><td>Sampler</td><td>forward</td><td>forward</td></tr></table>

Table 4: Training and sampling configuration of each student. A dash marks a component the objective does not have. The last row is applied at inference.

Optimization. We use a global training batch size of 256 and AdamW at a constant learning rate of $3 \times 1 0 ^ { - 4 }$ after 2500 warmup steps, without weight decay, in bf16, with gradients clipped at 1.0 and an exponential moving average of decay 0.9999. We use $n _ { \mathrm { a u x } } = 4 \colon$ four auxiliary updates on detached student samples, followed by one student update. At each optimization step, we sample the time pair $( t , T ) \sim \nu .$

Evaluation harness. We evaluate on OpenWebText with sequences of length 1024, sweeping sampling temperatures {0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10} applied to the output logits and sampling budgets $\{ 1 , 2 , 4 , \dot { \ldots } , 1 0 2 4 \}$ , with the autoregressive references evaluated only at 1024 NFE. Generative perplexity is computed by exponentiating the mean token negative log-likelihood under GPT-2-large. We evaluate with a single random seed, as in the default experimental setup of Gourevitch et al. (2026, Appendix I.1). We pair perplexity with diversity measured as the mean per-sequence Shannon entropy of empirical GPT-2 token frequencies, in nats, to obtain a quality–diversity frontier at each sampling budget, following Pynadath et al. (2026).

## E.2 FULL GENERATIVE FRONTIERS

Figure 5 extends the method comparison of Figure 2 to all evaluated sampling budgets.

## E.3 DESIGN SPACE

Figure 6 places the variants of each student on the frontier, every variant differing from Table 4 in a single component. The bridge-renoising ablation for Simplex-DMD is reported separately in Section E.5.

Simplex-DMD variants. The top row compares the recipe with three variants. No sc trains the student without self-conditioning, the reuse of the previous clean prediction as an extra network input described in Section B.1. No tsched replaces the teacher’s learned noise schedule, which the recipe reuses (Section B.1, (14)), with a default schedule $\gamma ( t ) = \mu - b \log ( - \log t )$ , with $\mu = 4 . 7 2 3$ and $b = 0 . 8 5 2$ being hyperparameters. No student init initializes the student from scratch instead of from the teacher. The no-tsched variant lies on the recipe’s curve and is told apart only by where along the curve its temperature-1.0 point sits; without self-conditioning, the curve reaches a slightly higher Gen PPL at the same entropy. The student trained from scratch collapses to near-uniform tokens, with unigram entropy about 6.9 and Gen PPL above $6 \times 1 0 ^ { 4 }$ ; it is marked in the bottom-right corner, outside the plotted range.

Reinforce-DMD variants. The bottom row varies the weight $\beta$ of the KL anchor toward the teacher in (13): $\beta \in \{ 0 , 0 . 2 , 0 . 5 \}$ , and a schedule annealing β from 0.5 to 0.05 over training. An additional variant trains Reinforce-DMD with self-conditioning, which the recipe does not use for this student. With $\beta = 0 ;$ , the student collapses to zero unigram entropy; this variant is marked in the top-left corner, outside the plotted range. The positive weights are separated by little, and $\beta = 0 . 5$ is the best of them. Self-conditioning moves the frontier to a higher Gen PPL at every entropy.

![](images/dc60b54602119ada4eaa5a2252f013d143135efeef9595fe95def5e882061a6c.jpg)  
Figure 5: Generative frontier on OpenWebText at every step budget, read as in Figure 2.

![](images/589c25c0720f355f14d3be01e534ba842c755c43ffded95c62dd443bdd31cb69.jpg)  
(a) Simplex-DMD, NFE = 4

![](images/e9145771436d2303c8047fbf8c1690601a59d42add33c143a7a79ab4dc7a4dbd.jpg)  
(b) Simplex-DMD, NFE = 8

![](images/04afe54e0f8872acff65ab79d36de42d1855086a96d627758231e847b1bc5180.jpg)  
(c) Reinforce-DMD, NFE = 128

![](images/0359167af3262fe329fe11d5d9a7cdbd9098aaf3fdb2012793abace6dafef40f.jpg)  
(d) Reinforce-DMD, NFE = 256  
Figure 6: Design space on OpenWebText, one component changed per variant, read as in Figure 2. Each student is shown at the two budgets where it is competitive.

## E.4 AUXILIARY LOSS

Equation (11) trains the auxiliary denoiser with cross-entropy against the student’s relaxed output. Two alternatives have the same population minimizer, the conditional mean $m _ { t } ^ { \eta , T }$ , and so leave the gradient identity (9) unchanged. The first regresses the auxiliary on that output under squared error,

$$
\mathsf { L } _ { \mathrm { a u x } } ^ { \mathrm { L } 2 } ( \phi ) = \mathbb { E } _ { ( t , T ) \sim \nu , Z _ { T } \sim p _ { T } , \varepsilon } \left[ \sum _ { \ell = 1 } ^ { L } \left\| \mathbf { X } ^ { \eta , T , \ell } - \mathbf { x } ^ { \phi , \ell } ( Z _ { t } ^ { \eta , T } , t , T ) \right\| ^ { 2 } \right] ,\tag{31}
$$

and the second keeps cross-entropy but replaces the simplex target by a token drawn at each position, $Y ^ { \ell } \sim \mathrm { C a t } ( \mathbf { X } ^ { \eta , T , \ell } )$ , written $\mathbf { e } _ { Y ^ { \ell } }$ for its one-hot vector,

$$
\mathsf { L } _ { \mathrm { a u x } } ^ { \mathrm { h a r d } } ( \phi ) = \mathbb { E } _ { ( t , T ) \sim \nu , Z _ { T } \sim p _ { T } , \varepsilon , Y } \left[ \sum _ { \ell = 1 } ^ { L } \mathrm { C E } \Big ( \mathbf { e } _ { Y ^ { \ell } } , \mathbf { x } ^ { \phi , \ell } ( Z _ { t } ^ { \eta , T } , t , T ) \Big ) \right] .\tag{32}
$$

Since $\mathbb { E } [ { \mathbf e } _ { Y ^ { \ell } } \mid Z _ { t } ^ { \eta , T } ] = \mathbb { E } [ { \mathbf X } ^ { \eta , T , \ell } \mid Z _ { t } ^ { \eta , T } ]$ , (32) estimates the same quantity as (11) with the extra variance of the categorical draw.

Figure 7 compares the three as Simplex-DMD training variants. They do not move the frontier; they move the temperature-1.0 point along it. At both budgets, the sampled-token target reaches the lowest Gen PPL, but at the lowest entropy of the three, and squared error sits at the opposite end, more diverse and less fluent, with the soft target between them. The variants also differ in how fast entropy erodes: from NFE = 32 to NFE = 128 the soft target loses 1.03 nats of unigram entropy against 1.06 for squared error and 1.17 for the sampled-token target. Since Gen PPL falls with entropy along this frontier, a variant that collapses faster reports the better perplexity while generating less diverse text. We keep the soft target for that reason.

![](images/58966ca51a930a1aa41a6d56afd5ecf90eed18ad23615f99da16be61f2103582.jpg)  
(a) NFE = 32

![](images/61651b687c9d3621e0ced6cb0a8c25cb7bf37e518688bb3ce9310895b7c19564.jpg)  
(b) NFE = 128  
Figure 7: Auxiliary loss for Simplex-DMD, read as in Figure 2: the soft target of (11) (CE soft), the squared error of (31) (L2) and the sampled-token target of (32) (CE hard). The three variants trace one frontier and differ in where their temperature-1.0 point sits on it, the sampled-token target lowest in entropy and squared error highest.

## E.5 BRIDGE RENOISING FOR SIMPLEX-DMD

We compare forward noising $q _ { t | \mathbf { x } }$ with the bridge $q _ { t | T , \mathbf { x } }$ during Simplex-DMD training (Figure 8). Each variant is a separate training run; forward noising is the default in Table 4. At every displayed budget, forward noising reaches a lower Gen PPL than the bridge at the same entropy, with the gap narrowing as the budget grows. Section D.4 discusses the auxiliary score approximation used for the bridge variant. We do not evaluate bridge renoising during Reinforce-DMD training.

## Sample 1

game, and they are much more powerful and terrifying than ones you play.”

All of that. That’s why elves fight against the ash, and why they do in combat and express their interests. The most baffest answer to this question, though, is whether cooperative action can do anything to an otherwise beautiful species.

There are powerful heroes like Balondakland. Magic’s heroes are beautiful and powerful, and they make them so much better. Attending resistance to powerful heroes, like almost about every single character, defeated or defeated in the game, is mind-shacking. Magic is about strategy, not a competitive strategy that is solely focused on how you play and fight. You have to win powerful battles. Why isn’t anyone going to tell you that advice? <eos> Imagine walking through every major venue in the United States every weekend and being told that British music is revered and charing kids to believe. You know how they treat kids? You just want to tell their kids, because they will always play. Are you kidding?

It is your job to do it like that: Your kids . . .

## Sample 2

Times. You may opt-out at any time. agree agree to receive occasional updates and special offers for The New York Times’s products and services. Thank you for subscribing. An error has occurred. Please try again later. View all New York Times newsletters. <eos> The increasingly technical aspect of chemical attack has started to wreak havoc in the United States. Even as the amount of bombs dropped and vast amounts of gas has soared, military officials said the problem was pervasive. And chemical attack threats aren’t new in the United States, said of physicist Dr. Mark Col. Salff of Columbia University. Experts believe they are common in countries that don’t yet know much — and tend to detonate civilian infrastructure behind a chemical attack.

Indeed, the United States’ government conducted a simulated chemical attack attack last year at Fort McMourstal, Va., where although it was the Pentagon’s first interagency Air Air Weapons Center (IAF) in the early 1990s, the airborne alert didn’t require the Bush administration to start experimenting with chemical attacks. Now, experts say the chemical attack’s capabilities isn’t compared with many . . .

about Hillary Clinton — and Trump refused to vilify her.

It’s pretty rare example of a woman who donated to Trump’s re-election campaign. But She Doesn’t Really Support The Public

Continue Reading Below Below Bagale Morinas

Handousee-stick speech.

But this woman’s lawyer is of no: She’s giving public officials a history of her work to the presidency.

Nobody without dating a woman has a certain course of conduct — there’s one reason why divorce lawyers find it very difficult: Anyone using public money to attack his or her spouse just gets a de-fated.

The fact is, there’s a good chance that this divorce or the whole thing could be worse if that it makes Donald Trump look worse. But supporting Trump is just another candidate or vice candidate.

We aren’t so sure if this has anything to do with Melania. But to be sure, it doesn’t mean Melania the mind-blung or most stack of any political conversations we’ve ever seen. When Melania gets on jumpballs, automated drivers crash into Melania’s emails, phone numbers, phone numbers, and private apps.

Of course, . . .

## Reinforce-DMD

NFE = 128, T = 1.10 · Gen PPL 26.9, entropy 5.27

## Sample 1

decreasing; changes in pattern of hacerenial events do not improve after use by patients without required DCI criteria; changes in markers of hacerenial events do not improve after use in patients with nonmenidiients; lesora conditions do not differentiate as an increase in patients with hacerenial events over time (9%); GRAs do not affect as an increase in patients with hacerenial events over time (9%); there has been an increase in GRAs and high blood pressure in patients with hacerenial events regardless of BMI (10%); there has been an increase in MII after signs of hypertension were reported in patients with high blood pressure (10%); in patients with hacerenial events over time (15%); GRAs have been either decreased in patients with hacerenial events in baseline and without exercise (16); there has been an increase in obesity and high blood pressure in diabetes (15) in patients with hacerenial events without diarrhea (15%); in all cases, there was no detectable cardiovascular symptoms after onset of autoimmune events; patients with inflammatory bowel sensitivity were found to show a reduction in patients with hacerenial events . . .

## Sample 2

emio build=true <class# rm appemio.init” - create “/tmp/retockbyC” JAX 15.0 appemio.init” - create “/tmp/retockbyC” JAX 15.0

1 2 3 4 5 6 7 8 rm appemio build=true <class# remove docker appemio.init” - create “/tmp/retockbyC” JAX 15.0 appemio.init” - create “/tmp/retockbyC” JAX 15.0 <class# remove docker./ appemio.init” - create

“/tmp/retockbyC” JAX 14.0 <class# remove docker appemio.init” - create “/tmp/retockbyC” JAX 15.0 <class# rm appemio build=true

This won’t work automatically; it still works by default inside your ’r80 in your build directory before it actually works. So we want to live everything on your ’r80.

Setting Up into the Appemio Environments

To start running appemio, you need to set up a new, new build inside make all configuration files.

You will start with appemio/esuse-build, which will create a new configurationfile that is called appemio/esuse, but it’s already available automatically:

1 cp /var/.share/appemio/esuse

which will build inside add appemio/esuse, but that’s located in lib/build/esuse.module.conf :

1 cp /var/.share/appemio/esuse

will create a new configurationfile so also download appemio/esuse that’s already available. It creates appemio/esuse.conf, which is file appemio-build and

which is located in lib/build/appemio/esuse.conf :

## Sample 3

ands, we will continue to seek new coal sources,” Bowman said in his budget. “As a rural country, we have a great need for our provinces, our bodies, and our provinces and communities to have the edge in demonstrating this wave of growth.”

Story continues below advertisement

Under Mr. Trudeau’s plan, each province has limited greenhouse gas emissions by nearly 50 megawatts. More than 50 states in the U.S. have set rules to address fossil fuels. The two largest jurisdictions in the states of Vermont, North Dakota and Washington, D.C. have reduced greenhouse gas emissions by 20 per cent year-per-year.

Those reductions overlap far beyond Prime Minister Justin Trudeau’s pledge to spend billions of dollars on the province’s dirty tape.

Low-emissions costs of coal projects increase by more than 15 per cent year-per-year, while reductions now account for 85 per cent of the cost of low-carbon infrastructure, according to a Bloomberg Institute analysis Monday. More than half of those costs will come from low-emissions and natural gas.

In July, Mr. Trudeau unveiled plans to erect 30 megawatts of new oilsands . . .

## LangFlow (teacher)

## NFE = 1024, T = 1.10 · Gen PPL 85.6, entropy 5.49

Sample 1

2 hits were not (corrected)

8/9/25: Meat (@Hot Alert) has been picked up for three hours.. 2 kills were not (corrected) 11 hits were not (corrected)

2/7/29/25: Burn from Asgard (@Hot Alert) has been picked up for six hours. 4 hits were not (corrected) OTHER ERROR/INCOMPLETE: Lowened 13 stars on my playlist 3/17/03/27: Las Mountains (@Hot Alert) has been picked up for eight hours.. 3 hits were not (corrected) 5/3/11/30: The Future Problem (@Hot Alert) was picked up for over a hour. 2 hits were not (corrected) <eos> If Golson turns home and you come there, you can put him on the roster. A wanna-play clown in your system? — leke Moslon (@Soslon ESPN) November 19, 2015

AFNDLEVILLE – The life of recruiting draft ranting is a nightmare.

Luckily for a core scorer who accepted draft year after drowning school had spent its highest semester, Tymontise Golson is secretly living in town for another day.

The former Ohio State guard, who appeared on Charlie Troke’s draft earlier this month, took to that point Wednesday night meeting with a group of league reporters.

## Sample 2

of skill and education plans for secondary sailing across Europe. Of the 352 charter boats there are 3,200 different applications available around the world with each one recognised in quantity.

David McConnell, national accommodation on the invaluable Rolic Drum Local Focus, which provides qualification guidance for all single luxury consultation applications, said: “We have seen sailing fishing stocks increase shortages due to the growing level of ocean education in demand.”

Sinking expert Roger Dixon said: “These results show that the UK fishing market has a powerful programme of skills and education initiatives available to children and kids around the world. <eos> [Editor Side Note: The following first marks our former September 16th Guest’s honour.]

Even today, one of music’s most evocative terms is Tavement Line’s “See That I Grived.” A Athoppersinging blend of lyrics that were inherited from his nearly haphazard and petulous live records in 1987, Six Flags Alegs eight years later. In celebration of it allure, Tavement Line has nicely prime prop reissues but part of their numerous compilation. After associating their last album of single “Kids I Still Walk” . . .

## Sample 3

a weakness,” he told breaking-well for six minutes in his dissertation, “and sometimes you good both very quickly.” <eos> RAR (IAM)) [LANAP] – Israeli forces launched three separate missile attacks on the country’s border towards Israel on Wednesday, including what Palestinian targeted witnesses say is the first retaliatory clashes.

The raids took place at a checkpoint on both sides of the Palestinian town of Ash Sabh, where Todi Palm flew heavy bombs on Lod al Sala, but were identified by a border insurgent, according to local residents.

A Palestinian militant investigating Outside Gaza said civilians died from the attack Wednesday, but others gave vague of other missile intercepts at the sight. No casualties could be immediately reported.

The missile bombardations by Israel came less than two weeks after its security forces raided a heavily populated fence and fence carrying anti-tank rockets. Two people were killed and an Israeli Israeli plane was intercepted near Ramte.

Anshara was seized by a five-day truce mediated by Israel and Iranian-led rebels, during which the rebels wanted in their area to be backed with militant terror groups . . .

## MDLM

NFE = 1024, T = 0.95 · Gen PPL 65.5, entropy 5.49

## Sample 1

civil rights profession say the allegations are “serious and troubling.”

“It’s clear that very little of the jurors have acknowledged wrongdoing or wrongdoing,” Toohey said, echoing a refrain common in the civil rights movement. “These are very serious allegations.”

Chris Douglass, a United States Department of National Security Policy lawyer at United States Witness in Florida, explained that the case often disadvantaged millennials who are pressured not to take part in the investigation and their reporting completed.

“They’re exploited,” Toohey said.

United States Witness names 105 officers involved in Miami civil rights cases that have been identified. Miami is a unique for civil rights attorneys all across America because corruption, says Lee Grenner, a former federal prosecutor of color, makes it difficult for attorneys who can initiate cases in foreign countries before the Supreme Court (also known as the American “African Supreme Court”) can call upon outside legal counsel, thus making it more effective and expensive to investigate black rights cases. <eos> Update: A population center, 18 for Malta, released a paper encouraging more condomless pills. The paper, which was authored as . . .

## Sample 2

for money. In fact, the whole system will regulate our feelings in ways that only make sense for us to really address how it treats our selves equally. For a start, some studies point out that one system away from we altogether can actually allow some people to feel more about loss than what might go into it, though the system is likely also a poor measure of value, resulting in a lack of emotional profits. Although it does not do any better than normal savings accounts, it offers some possible benefits.

By the way, history shows that if we misheat ourselves we can do the impossible for ourselves, whether everything is on going. But if we more emphasize the concern of diminishing return, we may simply drop everything, and we “blow up a cascade of misery and despair. And indeed the path to collapse teaches our system of the most intricate ways we manage for money: How do we make joy, dispassion, love, compassion, and even loss easy? Here’s what happens. <eos> McGlorady’s husband and her room throw the weight of . . .

## Sample 3

.2 million repeats) now in its third quarter, which would be tied with the primetime telecast for longestrunning programming in network history.

Disney’s ABC came in at 5.6 million repeats, possibly 17% higher than NBC’s previous 5.6– and close to breaking even with cable history in the share of live programming on network television.

10 p.m. ET/6’30 ABC was ticklled this week, and the network is expected to become the seventh ABC network to win post WWF slot at 10 p.m. ET/7’30.

Chuck Licht missed out

In an appearance at the 2011 Republican National Convention last Sunday, Roberts said his network drew the least share of the Los Angeles market in all-day coverage. The 2010 Comcast MediaStar convention averaged 1.7 million viewers on channel 88, the network’s highest number among all telecasts, up more than the network’s 4,5,000 watched by Comcast cable network Fox’s \$2.9 million teleday draw, and NBC.

Cost Mario Roberts said the Tampa Bay Times dressing is the direct result of the rise of cable “puttinging people for the great particular class of being rebuffed by the . . .