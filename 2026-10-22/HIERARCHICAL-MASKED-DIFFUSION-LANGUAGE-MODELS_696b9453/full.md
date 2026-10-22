# HIERARCHICAL MASKED DIFFUSION LANGUAGE MODELS

Parul Gupta<sup>1</sup> Satwik<sup>1</sup> Avni Aggarwal<sup>1∗</sup> Bhavesh Gurnani<sup>1∗</sup> Arnab Kumar Mondal<sup>2</sup> Dinesh Khandelwal<sup>2</sup> Dinesh Garg<sup>2</sup> Parag Singla<sup>2</sup> <sup>1</sup>Indian Institute of Technology Delhi <sup>2</sup>IBM Research

## ABSTRACT

Masked diffusion language models have emerged as an alternative to autoregressive models, enabling parallel token generation. However, standard masked diffusion language models use a shared noise schedule across positions without explicitly encoding which positions should be resolved first. Many tasks exhibit a natural priority among positions: document titles and section headings can guide the generation of body text, while Sudoku cells determined directly from the initial clues can help resolve more difficult cells. To incorporate this notion of priority (or hierarchy) among token positions, we introduce the Hierarchical Masked Diffusion Language Model (HMDLM), which augments each token position with a hierarchy level available as part of the training data, and requires zero or minimal additional annotations. It factors generation into two passes: the first predicts a hierarchy distribution over masked positions, the second predicts token distributions conditioned on it. Giving each hierarchy level its own noise schedule induces a soft, controllable preference for revealing higher-priority positions earlier while keeping decoding fully parallel. Our formulation is mathematically grounded, and we re-derive the forward and reverse processes and the NELBO for this factorization. Our evaluation shows that for the task of language generation on LM1B, HMDLM reduces perplexity by more than 1.5 points over Anchored Diffusion Language Model (ADLM), a recently proposed strong diffusion baseline; on WikiHow, we see a drop in perplexity of upto 4.4 pts on ADLM and upto 7 pts against MDLM (Masked Diffusion Language Model). A LLM judge on WikiHow, prefers HMDLM generations (the two variants combined) more than 2× as often as the baseline. On the task of solving Sudoku puzzles requiring multi-step reasoning, it improves solution accuracy by up to 8.6% absolute points over existing diffusion baselines.

## 1 INTRODUCTION

Large language models (LLMs) have revolutionized the AI space, from general-purpose text generation to solving complex problems such as mathematical reasoning and theorem proving. Despite their success, limitations remain. One of the criticisms of modern-day LLMs, which are largely auto-regressive in nature, is that they generate text only in a strictly left-to-right manner. It has been argued that this style of decoding has specific disadvantages, such as not being able to correct past mistakes, and involving large computational costs. As a remedy, recent literature has proposed another class of models, referred to as Diffusion Models (Austin et al., 2021; Lou et al., 2024; Sahoo et al., 2024; Nie et al., 2025c), which do not subscribe to the left-to-right generation style, but rather generate multiple tokens in parallel. The formulation is based on a forward diffusion process, which translates the text into noise, and on learning the reverse process, can generate the desired text by iteratively denoising from complete noise. This line of work has shown significant promise for the future of language generation, both for increasing computational efficiency and solving complex tasks (e.g., solving Sudoku or generating structured text).

A key open question in masked diffusion language models is how to choose the order in which positions should be denoised. Standard masked diffusion treats token positions symmetrically: with a shared masking schedule and no task-dependent notion of priority. Recent work shows that token ordering and decoding strategies can substantially affect generation quality (Kim et al., 2025; Garg et al., 2025; Hong et al., 2026; Fox et al., 2026). Recent approaches address this by identifying important anchor tokens (Rout et al., 2025; Xue et al., 2026), or adding planning-aware training and decoding mechanisms (Peng et al., 2025; Israel et al., 2025). These methods suggest that not all positions should be treated equally, but none provides a general hierarchy-aware forward and backward diffusion process. Such priority is often available: many problems carry a natural hierarchy over positions. In document generation, titles and section headings or summary often provide highlevel semantic structure before the surrounding body text. In Sudoku, more constrained cells are easier to fill first, and their priority belongs to the position, not the digit it will contain.

We introduce HMDLM, a hierarchical extension of masked diffusion language models that associates each token position with a hierarchy variable. During the forward process, both the token and its hierarchy state are diffused toward a masked state. During the reverse process, the model first predicts the hierarchy for masked positions and then predicts token identities conditioned on the predicted hierarchy. This two-stage factorization allows the model to learn which hierarchy level each position belongs to before decoding its token value. Unlike anchor-based methods (Rout et al., 2025), which tie importance to token identities, HMDLM assigns hierarchy to positions, with a separate noise schedule per level to control how fast each level is unmasked (see Section 4.3.2). We also derive the negative evidence lower bound (NELBO) for this process which serves as our training objective 4.

We experimentally validate HMDLM across several benchmark domains. On language modeling benchmarks like LM1B, we train the model from scratch and compare with diffusion baselines (like MDLM (Sahoo et al., 2024), SEDD (Lou et al., 2024), the Anchored Diffusion Language Model (ADLM) (Rout et al., 2025), and DiffusionBERT (He et al., 2023)) on both perplexity and generation quality. On Wikihow, we fine-tune on pretrained checkpoints and compare with ADLM and MDLM on perplexity and qualitative generation. On Sudoku, we train from scratch, and more reliably produce accurate solutions than the diffusion baselines (like MGDM (Ye et al., 2025) and MDLM). These results demonstrate that explicitly modeling hierarchy over token positions can improve masked diffusion generation across both natural language and structured domains.

Our work makes the following contributions: (a) We present a novel formulation for incorporating the notion of hierarchy in the discrete diffusion models, referred to as HMDLM (b) We re-formulate the standard diffusion formulation to have a two stage training and inference process, where the first stage predicts distribution over hierarchy levels, and second one predicts token distribution given the hierarchy prediction. (c) We re-derive the ELBO to incorporate the notion of hierarchy in the forward and backward diffusion process. (d) We present experiments showing gains over masked diffusion baselines in perplexity and generation quality on LM1B and WikiHow, and report accuracy on Sudoku. We also plan to release the code for HMDLM on paper acceptance.

## 2 RELATED WORK

Masked diffusion language models: Diffusion models have recently been adapted to language generation as an alternative to left-to-right autoregressive modeling. Early approaches explored diffusion in continuous word-vector spaces, as in Diffusion-LM (Li et al., 2022), or directly over discrete tokens, as in D3PMs (Austin et al., 2021) with structured transition matrices and absorbing [MASK] states. More recent masked diffusion language models formulate generation as iterative denoising from partially masked sequences: the forward process gradually replaces tokens with [MASK], and the reverse model reconstructs the sequence from bidirectional context. SEDD (Lou et al., 2024), reparameterized discrete diffusion (Zheng et al., 2023), DiffusionBERT (He et al., 2023), MD4 (Shi et al., 2024) and MDLM (Sahoo et al., 2024) improve the objectives and sampling procedures for this setting, while large-scale systems such as MDM (Nie et al., 2025a), LLaDA (Nie et al., 2025b) and LLaDA2.0 (Bie et al., 2025) show that masked diffusion can support instructionfollowing LLMs. However, these models typically employ a uniform masking schedule, ignoring the inherent semantic or structural hierarchy of the task.

Generation order and decoding strategies: Masked diffusion is attractive because generation need not proceed left-to-right; tokens can be revealed in arbitrary or partially parallel orders. This flexibility also makes the choice of denoising order important. Prior work shows that adaptive or learned ordering can significantly improve masked diffusion, through inference-time token ordering (Kim et al., 2025), planner-aware training (Peng et al., 2025), planned generation where the model first constructs a high-level plan and then denoises the planned spans in parallel (Israel et al., 2025), variational order learning (Fox et al., 2026), or unified learnable-order formulations (Hong et al., 2026). Other methods (Wang et al., 2025; Huang et al., 2025) modify sampling trajectories to address irreversibility or decoding bias. Our work is complementary: rather than learning a free-form position policy or adding an external planner, HMDLM encodes a task-dependent hierarchy in the diffusion state itself and uses per-level schedules to induce a controllable soft generation order.

Anchoring, planning, and structured priors: Several recent works also argue that diffusion language models benefit from identifying important tokens or high-level structure before completing the full sequence. Anchored Diffusion Language Models (Rout et al., 2025) observe that masking important words early can remove useful context, and use a two-stage model that predicts anchor-token distributions before predicting missing tokens. Planned Diffusion (Israel et al., 2025) first constructs an explicit plan and then generates spans in parallel. In code generation, AnCoder (Xue et al., 2026) uses syntax-tree priors to prioritize keywords and identifiers, and code-focused diffusion models such as DiffuCoder and Dream-Coder further show the value of domain-specific structure (Gong et al., 2025; Xie et al., 2025). These works share our motivation that not all sequence positions should be treated equally. However, HMDLM differs by attaching hierarchy to positions rather than only token identities, supporting multiple task-defined hierarchy levels, and using these levels to control both the forward corruption process and the reverse denoising dynamics through level-specific schedules.

## 3 BACKGROUND

## 3.1 NOTATIONS

Let x denote a sequence of discrete tokens each token $x ^ { i }$ is represented as one-hot column vector, and $\begin{array} { r } { \mathcal { V } = \{ x ^ { i } \in \{ 0 , 1 \} ^ { D } : \sum _ { j = 1 } ^ { D } x ^ { i } [ j ] = 1 \} } \end{array}$ at position i as the set of all such vectors. Define $C a t ( . ; \pi )$ as the categorical distribution over V classes, where $\boldsymbol { \pi } \in \mathbb { R } ^ { D }$ and $\textstyle \sum _ { d } \pi [ d ] = 1$ . Also we assume a special $[ M A S K ]$ token, $m \in \mathcal { V }$ and treat it as the $D ^ { t h }$ category, $\begin{array} { r } { \mathbf { i } . \mathbf { e } . , m [ D ] = 1 } \end{array}$ . Let $x _ { t }$ denote the state of the diffusion process at time step t. The following description follows the treatment provided in Sahoo et al. (2024).

## 3.2 DISCRETE DIFFUSION MODELS

Forward process. Discrete diffusion process is defined using a forward process that gradually corrupts data over T time steps, and replaces each token with a special MASK token. Let $t \in$ $\{ 0 , 1 , \ldots , T \}$ , with $t = 0$ corresponding to clean data and $t = T$ to fully masked data, and $x _ { t } ^ { i } \in \mathcal V$ denote progressively noisier latent variables where the timestep t ranges from 0 (zero noise) to $T$ (maximum noise). The forward noising process q at time t and position i is defined as:

$$
q ( x _ { t } ^ { i } \mid x _ { 0 } ^ { i } ) = \operatorname { C a t } \left( x _ { t } ^ { i } ; \alpha _ { t } x _ { 0 } ^ { i } + ( 1 - \alpha _ { t } ) m \right) ,\tag{1}
$$

where $x _ { 0 } ^ { i }$ is the clean state, $\alpha _ { t }$ is the masking schedule where $\alpha _ { t } \in [ 0 , 1 ]$ is a strictly decreasing function in $t ,$ with $\alpha _ { 0 } = 1$ and $\alpha _ { T } = 0$ controlling the proportion of unmasked tokens at time t. The corresponding posterior transition with $\begin{array} { r } { \lambda _ { t } = \frac { \alpha _ { t - 1 } - \alpha _ { t } } { 1 - \alpha _ { t } } } \end{array}$ is:

$$
\begin{array} { r } { q ( x _ { t - 1 } ^ { i } \mid x _ { t } ^ { i } , x _ { 0 } ^ { i } ) = \left\{ \begin{array} { l l } { \mathrm { C a t } ( x _ { t - 1 } ^ { i } ; x _ { t } ^ { i } ) , } & { x _ { t } ^ { i } \neq m , } \\ { \mathrm { C a t } \big ( x _ { t - 1 } ^ { i } ; ( 1 - \lambda _ { t } ) m + \lambda _ { t } x _ { 0 } ^ { i } \big ) , } & { x _ { t } ^ { i } = m . } \end{array} \right. } \end{array}\tag{2}
$$

Reverse process. The reverse process iteratively denoises the data from timestep t to $t - 1$ until the sequence becomes fully unmasked. Diffusion model aims to approximate the reverse process by learning a backward model $p _ { \theta } ( x _ { t - 1 } ^ { i } | x _ { t } ^ { i } )$ . Since the true posterior depends on the unknown clean token $x _ { 0 } ^ { i } .$ , it is approximated with a neural network $x _ { \theta } ^ { i } ( x _ { t } , t ) \in \Delta ^ { \nu }$ , which predicts the clean token distribution from the noisy input $\ v { x } _ { t } ^ { i }$ at time t. During the reverse diffusion process, unmasked tokens stay fixed, and clean inputs are never masked:

$$
\begin{array} { r } { p _ { \theta } ( x _ { t - 1 } ^ { i } \mid x _ { t } ) = \left\{ \begin{array} { l l } { \mathrm { C a t } ( x _ { t - 1 } ^ { i } ; x _ { t } ^ { i } ) , } & { x _ { t } ^ { i } \neq m , } \\ { \mathrm { C a t } \big ( x _ { t - 1 } ^ { i } ; ( 1 - \lambda _ { t } ) m + \lambda _ { t } x _ { \theta } ^ { i } ( x _ { t } , t ) \big ) , } & { x _ { t } ^ { i } = m . } \end{array} \right. } \end{array}\tag{3}
$$

NELBO. The denoising model $x _ { \theta } ( x _ { t } , t )$ is trained using the following Negative ELBO:

$$
\mathcal { L } _ { \mathrm { N E L B O } } ( x ; \theta ) = \mathbb { E } _ { q , t } \bigg [ - \log p _ { \theta } ( x _ { 0 } \mid x _ { 1 } ) + T \left( \frac { \alpha _ { t } - \alpha _ { t - 1 } } { 1 - \alpha _ { t } } \log \langle x _ { \theta } ( x _ { t } , t ) , x _ { 0 } \rangle \right) \bigg ]\tag{4}
$$

## 4 HIERARCHICAL MASKED DIFFUSION LANGUAGE MODEL

In this section we introduce the Hierarchical Masked Diffusion Language Model (HMDLM), which extends discrete diffusion models by incorporating a notion of hierarchy into both the forward and backward processes. Our framework augments each token position with a hierarchy level. Let $h _ { 0 } ^ { i }$ denote the gold hierarchy at position $i ,$ represented as a one-hot vector over K levels, i.e., $h _ { 0 } ^ { i } \in \{ 0 , 1 \} ^ { K }$ . We introduce a distinct hierarchy mask state $h _ { \mathrm { m } } ,$ giving $K { + 1 }$ possible hierarchy states. Positions to be unmasked at time step t are governed by the level dependent masking schedule $\alpha _ { k } ( t )$ defined for time step $t \in \{ 0 , \ldots , \bar { T } \}$ and level $k \in \dot { \{ 0 , \ldots , K - 1 \} }$ which is a decreasing function. If a position i at time step t is sampled or is already unmasked, we then set $b _ { t } ^ { i } = 0$ and $b _ { t } ^ { i } = 1$ otherwise.

$$
s _ { t } ^ { i } = ( x _ { t } ^ { i } , h _ { t } ^ { i } ) = ( x _ { 0 } ^ { i } , h _ { 0 } ^ { i } ) \mathrm { ~ i f ~ } b _ { t } ^ { i } = 0 \mathrm { ~ ( r e v e a l e d ) , } \qquad ( m , h _ { \mathrm { m } } ) \mathrm { ~ i f ~ } b _ { t } ^ { i } = 1 \mathrm { ~ ( m a s k e d ) . }\tag{5}
$$

Since $b _ { t } ^ { i }$ is shared, a token and its hierarchy level are always masked or revealed together, ruling out decoupled states such as a masked token with a revealed level.

## 4.1 FORWARD AND REVERSE PROCESS

Forward Process The forward process is a fixed Markov chain, that progressively masks tokens from the initial state $s _ { 0 }$ to a noisy state $s _ { t }$ at each time step t over $T$ time steps, according to the hierarchy conditioned noise schedule $\alpha _ { k } ( t )$ . Tokens at lower hierarchy levels are masked more aggressively at early time-steps, than tokens at higher hierarchy levels. The forward marginal at a single token position i with hierarchy k and joint state $s _ { \mathrm { t } } ^ { i } = ( x _ { \mathrm { t } } ^ { i } , h _ { \mathrm { t } } ^ { i } )$ is given by:

$$
q ( s _ { t } ^ { i } \mid s _ { 0 } ^ { i } ) = \operatorname { C a t } \left( s _ { t } ^ { i } ; \alpha _ { h _ { 0 } ^ { i } } ( t ) s _ { 0 } ^ { i } + ( 1 - \alpha _ { h _ { 0 } ^ { i } } ( t ) ) s _ { m } \right)\tag{6}
$$

At $t \ : = \ : T$ , all tokens are fully masked regardless of the hierarchy level, i.e. $\alpha _ { k } ( T ) = 0$ . Let $\begin{array} { r } { \lambda _ { k } ( t ) = \frac { \alpha _ { k } ( t - 1 ) - \alpha _ { k } ( t ) } { 1 - \alpha _ { k } ( t ) } } \end{array}$ denote the probability that a masked level-k position is revealed at step $t ,$ with $\lambda _ { k } ( t ) = 0$ when $\alpha _ { k } ( t ) = 1$ . The true joint posterior is:

$$
\begin{array} { r } { q ( { s } _ { t - 1 } ^ { i } \mid { s } _ { t } ^ { i } , { s } _ { 0 } ^ { i } ) = \left\{ \begin{array} { l l } { \mathrm { C a t } ( { s } _ { t - 1 } ^ { i } ; { s } _ { t } ^ { i } ) , } & { { s } _ { t } ^ { i } \ne { { \bf s } _ { m } } , } \\ { \mathrm { C a t } \Big ( { s } _ { t - 1 } ^ { i } ; ( 1 - \lambda _ { h _ { 0 } ^ { i } } ( t ) ) { { \bf s } _ { m } } + \lambda _ { h _ { 0 } ^ { i } } ( t ) { s } _ { 0 } ^ { i } \Big ) , } & { { s } _ { t } ^ { i } = { \bf s } _ { m } . } \end{array} \right. } \end{array}\tag{7}
$$

## Reverse Process

The reverse process aims to recover the clean sequence $s _ { 0 }$ from a fully masked sequence s over $T$ denoising steps. We approximate the reverse process $p _ { \theta } ( s _ { t - 1 } \mid s _ { t } , t )$ using the neural network $s _ { \theta } ( s _ { t } , t )$ , which is composed of two joined networks - the hierarchy head, $h _ { \theta } ( x _ { t } , h _ { t } , t )$ and the token head, $x _ { \theta } ( x _ { t } , h _ { t - 1 } , t )$ . It takes as input the noisy state $s _ { t } { = } ( x _ { t } , h _ { t } )$ at timestep t and predicts the clean joint state $s _ { 0 }$

Revealed positions are carried over unchanged, $p _ { \theta } \big ( s _ { t - 1 } ^ { i } \ | \ s _ { t } ^ { i } \big ) = \operatorname { C a t } \bigl ( s _ { t - 1 } ^ { i } ; s _ { t } ^ { i } \bigr ) \ \mathrm { i f } \ s _ { t } ^ { i } \neq \mathbf { s } _ { m }$ . For a masked position $( s _ { t } ^ { i } = \mathbf { s } _ { m } )$ , the model first draws the hierarchy state and then the token conditioned on it:

$$
\begin{array} { r l r l } & { ~ p _ { \theta } \left( h _ { t - 1 } ^ { i } = k \ | \ x _ { t } , h _ { t } \right) = \lambda _ { k } ( t ) g _ { k } ^ { i } ( t ) , } & & { k \in \{ 0 , \ldots , K - 1 \} , } \\ & { ~ p _ { \theta } \left( h _ { t - 1 } ^ { i } = h _ { \operatorname* { m } } \ | \ x _ { t } , h _ { t } \right) = 1 - \pi _ { i } ( t ) , } \\ & { ~ p _ { \theta } \left( x _ { t - 1 } ^ { i } \ | \ x _ { t } , h _ { t - 1 } \right) = \mathrm { C a t } \big ( x _ { t - 1 } ^ { i } ; \ f ^ { i } ( t ) \big ) , } & & { h _ { t - 1 } ^ { i } \neq h _ { \operatorname* { m } } , } \\ & { ~ p _ { \theta } \left( x _ { t - 1 } ^ { i } = m \ | \ x _ { t } , h _ { t - 1 } \right) = 1 , } & & { h _ { t - 1 } ^ { i } = h _ { \operatorname* { m } } , } \end{array}\tag{8}
$$

where $g ^ { i } ( t ) \ = \ \mathrm { s o f t m a x } { \left( h _ { \theta } ^ { i } ( x _ { t } , h _ { t } , t ) \right) }$ is the hierarchy head’s distribution over the K levels, $f ^ { i } ( t ) \ = \ \mathrm { s o f t m a x } \big ( x _ { \theta } ^ { i } ( x _ { t } , h _ { t - 1 } , t ) \big )$ is the token head’s distribution over the vocabulary, and $\begin{array} { r } { \pi _ { i } ( t ) = \sum _ { k = 0 } ^ { K - 1 } \lambda _ { k } ( t ) g _ { k } ^ { i } ( t ) = \mathbb { E } _ { k \sim g ^ { i } ( t ) } [ \lambda _ { k } ( t ) ] } \end{array}$ , is the total probability that position i is revealed at step t. Since the gold hierarchy $h _ { 0 } ^ { i }$ is unknown in the reverse process, $\pi _ { i } ( t )$ replaces the true reveal probability $\lambda _ { h _ { \mathrm { } } ^ { i } } \left( t \right)$ of equation 7 by its expectation under the predicted hierarchy distribution. Because $\lambda _ { k ^ { \prime } } ( t ) \geq \lambda _ { k } ( t )$ for $k ^ { \prime } > k$ , a masked position is more likely to be revealed at a higher level than at a lower one even when $g ^ { i } ( t )$ is uniform, which is what induces the soft generation order (Appendix D).

## 4.2 ELBO

We derive the discrete time ELBO 9 and its continuous-time limit 76 for the generative process $p _ { \theta } ( s _ { 0 } )$ . The detailed derivation is given in the Appendix A. The discrete ELBO is given as:

$$
\log p _ { \theta } ( \mathbf { s } _ { 0 } ) \ \geq \ \underbrace { \mathbb { E } _ { q } \big [ \log p _ { \theta } ( \mathbf { s } _ { 0 } \mid \mathbf { s } _ { 1 } ) \big ] } _ { L _ { 0 } } - \sum _ { t = 2 } ^ { T } \underbrace { \mathbb { E } _ { q } \Big [ D _ { \mathrm { K L } } \big ( q ( \mathbf { s } _ { t - 1 } \mid \mathbf { s } _ { t } , \mathbf { s } _ { 0 } ) \mid \mid p _ { \theta } ( \mathbf { s } _ { t - 1 } \mid \mathbf { s } _ { t } ) \big ) \Big ] } _ { L _ { t } } .\tag{9}
$$

Each $L _ { t }$ is a KL over the state $\mathbf { \Psi } ( \mathbf { s } _ { t } ) = \mathbf { \Psi } ( x _ { t } , h _ { t } )$ . By the chain rule of KL divergence, $L _ { t }$ can be expanded as:

$$
\begin{array} { r l } & { L _ { t } = \underbrace { D _ { \mathrm { K L } } \big ( q ( \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } , \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) \mid \big \| \ p _ { \theta } ( \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } ) \big ) } _ { \mathrm { h i e r a r i b y K L ( P a s s 1 ) } } } \\ & { \qquad + \underbrace { { \mathbb { E } } _ { q ( \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } , \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) } \Big [ D _ { \mathrm { K L } } \big ( q ( \mathbf { x } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t - 1 } , \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) \mid \big \| \ p _ { \theta } ( \mathbf { x } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t - 1 } ) \big ) \Big ] } _ { \mathrm { t o k e n ~ K L ( P a s s 2 ) } } . } \end{array}\tag{10}
$$

The KL divergence splits into two terms 10, which we model using two stages in our architecture introduced in 4.3.

## 4.3 MODEL ARCHITECTURE

## 4.3.1 TWO STAGE MODEL

![](images/b7ed769e9b42555dd409ccb54fcec41afaa9f3a5e9080949d643d4caf7228304.jpg)  
Figure 1: HMDLM inference example. Positions $i = 0 , 3 , 5$ are masked in $\left( { { x } _ { t } } , { { h } _ { t } } \right)$ . Step t: Stage 1 reveals $h _ { t - 1 } ^ { 5 } = 1$ , then Stage 2 predicts $x _ { t - 1 } ^ { 5 } = { \mathsf { b a l l } }$ . Step $t - 1 \colon$ Stage 1 reveals $h _ { t - 2 } ^ { 0 } = 0$ and $h _ { t - 2 } ^ { 3 } = 0$ , then Stage 2 predicts $x _ { t - 2 } ^ { 0 } = \mathrm { T }$ he and $x _ { t - 2 } ^ { 3 } = { \mathrm { t h e } }$

To model the factorized posterior, we introduce the following two stage architecture 1.

First Stage: Given the partially denoised sequence $( x _ { t } , h _ { t } )$ at time step t, the first stage predicts the hierarchy distribution $g ^ { i } ( t )$ at every masked position $i \in \mathcal { M } _ { t } .$ , where $\mathcal { M } _ { t }$ is the set of masked positions at time $t ;$ together with the noise schedule, this defines $p _ { \theta } \big ( h _ { t - 1 } ^ { i } \ | \ x _ { t } , h _ { t } \big )$ in equation 8. Masked positions are fed the masked state ${ \bf s } _ { m }$ as input to Stage I during both training and inference. During training, unmasked positions are provided with their gold pairs $( x _ { 0 } ^ { i } , h _ { 0 } ^ { i } )$ via teacher forcing. During inference, they are provided with the token and hierarchy sampled at the step when the position was first revealed, which remain fixed thereafter. At each inference step t, every masked position i draws its next hierarchy state from equation 8: it is revealed at level $k \in \{ 0 , \ldots , K - 1 \}$ with probability $\lambda _ { k } ( t ) g _ { k } ^ { i } ( t )$ and stays masked with probability $1 - \pi _ { i } ( t )$ . This single draw decides both whether position i is revealed at this step and at which level, which forms the set of revealed positions $\mathcal { U } _ { t } \subseteq \mathcal { M } _ { t }$

Second Stage: The second stage predicts the token distribution at each masked position i at time step t: $p _ { \theta } ( x _ { t - 1 } ^ { i } \mid x _ { t } , h _ { t - 1 } )$ . For masked positions, it receives as input $\left( x _ { t } , h _ { t - 1 } \right)$ with the hierarchy at the reveal set filled in: positions $i \in \mathcal { U } _ { t }$ are assigned gold hierarchy $h _ { 0 } ^ { i } , ( m , h _ { 0 } ^ { i } )$ during training and the Stage-I sample $( m , \bar { h _ { t - 1 } ^ { i } } )$ during inference. For remaining masked positions, we pass the mask state $s _ { m } = ( \mathbf { m } , h _ { m } )$ as input. For unmasked positions, the token and hierarchy are copied directly from $x _ { t }$ and $h _ { t }$ without modification: $p _ { \theta } ( s _ { t - 1 } ^ { i } \bar  \mid s _ { t } ^ { i } ) = \mathrm { C a t } \left( s _ { t - 1 } ^ { i } ; s _ { t } ^ { i } \right)$ if $s _ { t } ^ { i } \neq { \bf s _ { m } }$ during inference and passed $s _ { 0 } ^ { i } { = } ( x _ { 0 } ^ { i } , h _ { 0 } ^ { i } )$ during training. Both stages use weight tying, with separate output heads (hierarchy predictor head and token predictor head) attached to the final hidden layer and a softmax applied at the end of each head. This design reduces the parameter count while ensuring that both stages benefit from the same rich contextual representation.

Note: Across iterations, the token and hierarchy at each position, $s _ { t } ^ { i } = ( x _ { t } ^ { i } , h _ { t } ^ { i } )$ , are masked and unmasked jointly through a shared Bernoulli variable 5 . Within a single iteration, however, Stage 1 predicts the hierarchy before Stage 2 predicts the token. The partially unmasked state $( m , h _ { t - 1 } ^ { i } )$ with $h _ { t - 1 } ^ { i } \neq h _ { \mathrm { m } }$ therefore arises only between the two stages of the same iteration.

## 4.3.2 PER-LEVEL NOISE SCHEDULE AND HIERARCHY EMBEDDINGS

We use a different masking schedule for each hierarchy level k. Tokens that are more informative are given a higher hierarchy level and a slower masking schedule. Consequently, during the forward noising process, these tokens are masked more slowly than lower level tokens, preserving them for longer. They are denoised faster than other tokens during the reverse de-noising process. These tokens are unmasked earlier than lower-hierarchy tokens, allowing the model to condition on higher level tokens when predicting the remaining masked positions.

The hierarchy of a position is represented by a row of the learnable hierarchy embedding matrix $W \in \mathbb { R } ^ { ( K + 1 ) \times H }$ , where K is the number of hierarchy levels and H is the hidden dimension. Row $W _ { k }$ is the embedding of level $k \in \{ 0 , \ldots , K - 1 \}$ , and the last row $W _ { K }$ is the embedding of the hierarchy mask state $h _ { \mathrm { m } }$ . The hierarchy embedding for each position i is added to the token embedding at position i and fed as input to the backbone.

## 4.4 TRAINING AND INFERENCE

Training The HMDLM model is trained end-to-end across both the passes. We sample a time step t, and mask each token independently according to the token’s hierarchy level masking schedule. The input to the Stage 1 is $\left( \boldsymbol { x } _ { t } , \boldsymbol { h } _ { t } \right)$ and it predicts the hierarchy probabilities $- p _ { \theta } ( h _ { t - 1 } ^ { i } \mid x _ { t } , h _ { t } )$ , we pass updated $\left( x _ { t } , h _ { t - 1 } \right)$ to Stage 2 to get token distribution - $\cdot p _ { \theta } ( x _ { t - 1 } ^ { i } \mid x _ { t } , h _ { t - 1 } )$ for each masked position. The final training loss is governed by:

$$
\mathcal { L } = \mathbb { E } _ { t , q } \Bigg [ \sum _ { i = 1 } ^ { L } { \mathbf { 1 } [ x _ { t } ^ { i } = m ] \left( \lambda _ { h _ { 0 } ^ { i } } ^ { \prime } ( t ) \left( - \log f _ { x _ { 0 } ^ { i } } ^ { i } ( t ) - \beta _ { \mathrm { h } } \log g _ { h _ { 0 } ^ { i } } ^ { i } ( t ) \right) + \beta _ { \mathrm { e } } \left( \sum _ { k } g _ { k } ^ { i } ( t ) \lambda _ { k } ^ { \prime } ( t ) - \lambda _ { h _ { 0 } ^ { i } } ^ { \prime } ( t ) \right) \right) } \Bigg ]\tag{11}
$$

where $\beta _ { h }$ and $\beta _ { e }$ are tunable hyperparameters, $\begin{array} { r } { \lambda _ { k } ^ { \prime } ( t ) = \frac { - \alpha _ { k } ^ { \prime } ( t ) } { 1 - \alpha _ { k } ( t ) } } \end{array}$ , and $h _ { 0 } ^ { i }$ is the gold hierarchy and $x _ { 0 } ^ { i }$ is the gold token, $g _ { k } ^ { i } ( t )$ and $f _ { v } ^ { i } ( t )$ are the hierarchy-head and token-head probabilities of equation 8 at position i. The full training procedure is given in Algorithm 1.

Inference During the reverse process, we start from a fully masked sequence and iteratively unmask tokens at each time step t. At time step $t = T$ all the tokens are $[ M A S K ]$ tokens and hence are assigned masked state, $s _ { m }$ . In Stage I, we sample hierarchy value for each masked token position i at time step t from the predicted hierarchy probabilities as:

$$
\begin{array} { r } { h _ { t - 1 } ^ { i } \sim \operatorname { C a t } ( \lambda _ { 0 } ( t ) g _ { 0 } ^ { i } ( t ) , \dots , \lambda _ { K - 1 } ( t ) g _ { K - 1 } ^ { i } ( t ) , \underbrace { 1 - \pi _ { i } ( t ) } _ { \mathrm { s t a y m a s k e d } } ) } \end{array}\tag{12}
$$

with $g ^ { i } ( t )$ and $\pi _ { i } ( t )$ as in equation $^ { 8 , }$ , and $\lambda _ { k } ( t )$ the per-step reveal probability of level k. Higher levels are masked more slowly, and for the schedules we use ${ \lambda } _ { k } ( t ) \dot { { \bf \phi } } \le { \lambda } _ { k ^ { \prime } } ( t )$ for $k < k ^ { \prime }$ at every step at which level $k ^ { \prime }$ is still masked (Appendix D). A position predicted to be at level $k ^ { \prime }$ therefore receives a larger reveal mass $\lambda _ { k ^ { \prime } } ( t ) g _ { k ^ { \prime } } ^ { i } ( \bar { t } )$ and is revealed earlier. The full inference algorithm is given in the Appendix (Algorithm 2).

## 5 EXPERIMENTS

In this section, we evaluate the proposed Hierarchical Masked Diffusion Language Model (HMDLM) along two axes: (1) its ability to model language distributions, measured via perplexity, and (2) its effectiveness as a puzzle solver, measured by accuracy. We consider three representative benchmarks: the One Billion Word Benchmark (LM1B) (Chelba et al., 2013) and WikiHow (Koupaee & Wang, 2018) for generative modeling, and Sudoku (Palm et al., 2018)

## 5.1 LANGUAGE GENERATION TASKS

Setup. For both LM1B and WikiHow, HMDLM adopts the MDLM backbone (Sahoo et al., 2024): 12 transformer blocks, hidden size 768, 12 attention heads, rotary positional embeddings (Su et al., 2024), and the dataset-dependent tokenizer for each corpus: BERT-base-uncased for LM1B and GPT-2 for WikiHow. The hierarchy-specific components (K = 2 levels) are (i) a shared embedding matrix $W \in \mathbb { R } ^ { ( K + 1 ) \times H } = \mathbb { R } ^ { 3 \times 7 6 \bar { 8 } }$ , with K learnable level embeddings plus a learnable [MASK]- hierarchy row for unrevealed positions, and (ii) a hierarchy-prediction head that mirrors the token head’s adaLN-modulated (Peebles & Xie, 2023) final layer, mapping $\mathbb { R } ^ { 7 6 8 }  \mathbb { R } ^ { 2 }$ . Together these add ≈ 0.2M parameters, about 0.15% of the 139M-parameter LM1B backbone and 0.12% of the 170M-parameter WikiHow backbone. The Sudoku architecture differs and is described in Section 5.2.

Metrics. We evaluate generation quality with four complementary measures. Model PPL assesses density modeling. Generative PPL assesses fluency of sampled text, but can be artificially lowered by leaning on a few high-likelihood phrases, producing repetitive, low-entropy output. We therefore read it alongside two diversity measures: entropy, where a value close to the reference or a high value indicates the model has not collapsed onto safe phrases, and MAUVE, which measures the similarity between the model-generated and reference text distributions. To corroborate the above metrics, we also run a pairwise LLM-as-judge study (GPT-5 judge) that reports how often each model’s generations are preferred.

## 5.1.1 LM1B DATASET

Task Description: We evaluate the generative modeling capability of HMDLM by training it from scratch on LM1B, a standard dataset for language modeling. We compare test perplexity against several diffusion-based baselines: MDLM (Sahoo et al., 2024), SEDD (Lou et al., 2024), the Anchored Diffusion Language Model (ADLM) (Rout et al., 2025), and DiffusionBERT (He et al., 2023).

Setup: Using the shared language backbone (Section 5.1), we set a context length of 128 and train HMDLM from scratch for 1.25M and 2.5M steps (≈ 33B and ≈ 65B tokens) on LM1B (see Section B).

Hierarchy Assignment: We define two hierarchy levels on LM1B by in-sentence word frequency (τ) following Rout et al. (2025). Tokens appearing at most τ times in a sample are assigned to level 1 (important words) and the rest to level 0 (common words). Although ADLM reports numbers with τ = 5, this setting provides limited hierarchy separation: majority of the tokens get hierarchy 1 (99.16%). Under $\tau = 1$ , around 79% tokens get level 1, and the rest get assigned level 0 - yielding a more meaningful hierarchy.

Noise Schedule: We use a linear denoising schedule (identical across all models for a fair PPL comparison) with a level offset $\delta = 0 . 2 0$ , defined for HMDLM as $\alpha _ { 0 } ( t ) =$ $1 - ( 1 - \varepsilon ) t$ and $\alpha _ { 1 } ( t ) = 1$ for $t \in [ 0 , \delta ]$ while $\begin{array} { r } { \alpha _ { 1 } ( t ) = 1 - ( 1 - \varepsilon ) \frac { t - \delta } { 1 - \delta } } \end{array}$ for $t \in$ (δ, 1].

Table 1: Test perplexities (PPL ↓) on LM1B. Best value is bolded.
<table><tr><td>Model PPL (↓)</td></tr><tr><td>33B tokens Diffusion BERT (He et al., 2023) ≤ 63.78 SEDD (Lou et al., 2024) &lt; 32.79 MDLM (Sahoo et al., 2024) 7 27.04 ADLM (Rout et al., 2025) 26.40</td></tr><tr><td>HMDLM ≤ 24.30 HMDLM</td></tr><tr><td>65B tokens MDLM (Sahoo et al., 2024) &lt; 25.49</td></tr><tr><td>ADLM (Rout et al., 2025) &lt; 24.46 ≤ 22.88</td></tr></table>

Table 2: LM1B generative evaluation on 65B checkpoints. Mean ± std over 3 seeds.
<table><tr><td>Model</td><td>Gen. PPL (↓) MAUVE* (↑)</td><td></td><td>Entropy</td></tr><tr><td>MDLM</td><td> $1 7 7 . 9 4 { \pm } 0 . 8 2$ </td><td> $0 . 6 6 { \scriptstyle \pm 0 . 0 3 }$ </td><td> $6 . 0 2 2 { \scriptstyle \pm 0 . 0 0 3 }$ </td></tr><tr><td>HMDLM</td><td>83.35±0.83</td><td> $\mathbf { 0 . 7 6 { \scriptstyle \pm 0 . 0 6 } }$ </td><td> $5 . 8 7 6 { \scriptstyle \pm 0 . 0 1 4 }$ </td></tr></table>

<sup>⋆</sup>MAUVE is reported here in %.

Results: We generate 1000 samples across

three seeds for each model (128 token length and 256 steps). As can be seen from the Table 1 and Table 2, HMDLM attains the best model PPL and generative perplexity (measured using GPT-2 Large) among the diffusion baselines at the 65B-token budget, cutting generative PPL by nearly 95 points relative to MDLM <sup>1</sup>. The reduction in generative PPL is accompanied by only a small decrease in entropy (5.876 vs. 6.022), providing no clear evidence of severe diversity collapse. MAUVE scores, though low, are higher for HMDLM as compared to MDLM. An independent LLM-as-judge evaluation (prompt: H) is consistent with the results: in blinded pairwise comparisons with randomized A/B assignment (100 samples per seed, 3 seeds, 2 runs each; 600 comparisons total), the judge preferred HMDLM to MDLM 56% to 44%, with no ties. Under the $\tau = 5$ setting, HMDLM performs similar to MDLM, as nearly all real tokens are assigned to the same hierarchy level, leaving little scope for the hierarchical mechanism to provide an advantage.

## 5.1.2 WIKIHOW DATASET

Task Description: We next evaluate generative modeling on Wiki-How. We initialize HMDLM from the publicly released MDLM checkpoint pretrained on OpenWebText (OWT) (Gokaslan & Cohen, 2019), which we refer to as MDLM-OWT, and fully fine-tune it on WikiHow. $\mathbf { A } s$

Table 3: Model PPL (↓) across sequence lengths on WikiHow. Best value is in bold.
<table><tr><td> $L$ </td><td>MDLM</td><td> $\mathrm { H M D L M - P ^ { \dagger } }$ </td><td>ADLM</td><td>HMDLM-F‡</td></tr><tr><td>256</td><td> $2 4 . 6 0 \pm 0 . 2 2$ </td><td> $2 3 . 8 9 \pm 0 . 1 1$ </td><td> $2 2 . 0 5 \pm 0 . 2 1$ </td><td>17.62 ± 0.12</td></tr><tr><td>512</td><td> $2 1 . 0 4 \pm 0 . 0 6$ </td><td> $2 1 . 7 6 \pm 0 . 1 5$ </td><td> $2 0 . 1 5 \pm 0 . 1 0$ </td><td> ${ \bf 1 6 . 6 4 \pm 0 . 0 6 }$ </td></tr><tr><td>1024</td><td> $2 2 . 6 5 \pm 0 . 0 4$ </td><td> $2 2 . 5 8 \pm 0 . 0 4$ </td><td> $2 0 . 9 2 \pm 0 . 0 5$ </td><td>16.90 ± 0.09</td></tr></table>

<sup>†</sup>HMDLM-P: position-based hierarchy. <sup>‡</sup>HMDLM-F: frequency-based hierarchy.

baselines, we fine-tune the MDLM-OWT and ADLM-OWT checkpoints on WikiHow under an identical setting.

Setup: We reuse the shared language backbone (Section 5.1), fully fine-tuning from the MDLM-OWT checkpoint. We experiment with maximum sequence lengths of $L \in \{ 2 \bar { 5 } 6 , 5 1 2 , 1 0 2 4 \}$ , filtering WikiHow to corresponding token length. The train, val, and test set corresponding to each token length L is kept consistent across all models for fair comparison. During inference, we use 256 sampling steps for all three sequence lengths. All the baseline models are trained for 30 epochs. Both the HMDLM variants are trained slightly longer because of lower expected masking fraction. Detailed discussion in Appendix C.

Hierarchy Assignment: We consider two hierarchy-assignment strategies. In the position-based scheme, we assign level 1 to summary tokens and level 0 to the rest; here we compare directly against the fine-tuned MDLM model, since ADLM defines its anchor tokens by frequency rather than position and is therefore not comparable under this assignment. In the Frequency-based scheme, we reuse

Table 4: Generative evaluation on WikiHow across sequence lengths $L ~ \in ~ \{ 2 5 6 , 5 1 2 , 1 0 2 4 \}$ HMDLM-P/F: position-/frequency-based hierarchy. Gold is the reference data.
<table><tr><td></td><td colspan="3">Gen. PPL</td><td colspan="3">MAUVE (↑)</td><td colspan="3">Entropy</td></tr><tr><td>Model</td><td>256</td><td>512</td><td>1024</td><td>256</td><td>512</td><td>1024</td><td>256</td><td>512</td><td>1024</td></tr><tr><td>Gold</td><td>25.2</td><td>21.0</td><td>20.8</td><td>1.00</td><td>1.00</td><td>1.00</td><td>5.86</td><td>6.42</td><td>6.70</td></tr><tr><td>MDLM</td><td>76.0</td><td>67.4</td><td>62.7</td><td>0.33</td><td>0.11</td><td>0.12</td><td>6.47</td><td>7.10</td><td>7.60</td></tr><tr><td>ADLM</td><td>84.0</td><td>82.1</td><td>93.81</td><td>0.40</td><td>0.21</td><td>0.08</td><td>6.49</td><td>7.17</td><td>7.77</td></tr><tr><td>HMDLM-P</td><td>23.1</td><td>17.6</td><td>14.6</td><td>0.38</td><td>0.13</td><td>0.07</td><td>5.91</td><td>6.26</td><td>6.45</td></tr><tr><td>HMDLM-F</td><td>29.2</td><td>25.4</td><td>19.92</td><td>0.52</td><td>0.41</td><td>0.20</td><td>5.98</td><td>6.60</td><td>6.83</td></tr></table>

the frequency rule from LM1B, assigning level 1 to tokens occurring at most once in a sentence and level 0 otherwise; this is directly comparable with ADLM, as we apply the same rule to define its anchor positions.

Noise Schedule: We use the power-function schedule $\alpha _ { h } ( t ) = 1 - ( 1 - \epsilon ) t ^ { \omega _ { h } }$ with $\omega _ { h } [ 0 ] = 1 . 0$ and $\omega _ { h } [ 1 ] = 2 . 0$ . Setting $\omega _ { h } [ 0 ] = 1 . 0$ recovers the log-linear schedule of MDLM.

Results: HMDLM-F (Frequency based hierarchy) achieves the best model PPL and its generative PPL is closest to reference distribution. HMDLM-P (Position based hierarchy) achieves the best gen PPL, on all three generation lengths (Tables 3 and 4), cutting gen PPL by nearly 50 points relative to the baselines. The improvement is not repetition-driven: MAUVE is higher for HMDLM-F at every length, and entropy stays close to that of the reference dataset, indicating a closer match in terms of token distribution. LLM-as-judge evaluation (prompt: H) is consistent with the results: in blinded comparisons (400 samples, 2 runs each; 800 comparisons total), the judge preferred HMDLM-P 40.75% times, HMDLM-F 31.88% times, ADLM 15.25%, and MDLM 12.12%. Therefore, HMDLM combined was preferred more than twice the times baselines were selected. We also find that, for this dataset, HMDLM-F has lower model PPL on test set, better MAUVE scores and entropy closer to that of reference data than the HMDLM-P for all lengths. Refer to Section G in the Appendix for a qualitative analysis

Position- vs. frequency-based hierarchy: On this dataset, HMDLM-F outperforms HMDLM-P in model PPL, MAUVE, and entropy (closer to reference distribution) at all lengths. We hypothesize that generating key tokens first matters more than their positions, since these tokens provide enough context to decode the rest of the text. This is domain-dependent: for tasks like Sudoku, position-based hierarchy is likely essential. Both variants improve over MDLM, and ADLM, which is implicitly frequency-driven, is consistently outperformed by HMDLM-F. This highlights the benefit of our principled treatment of hierarchy in diffusion models.

## 5.2 SUDOKU DATASET

Task Description: Sudoku is a useful test case for structured reasoning. Some cells can be filled directly from the given numbers, while others can be resolved only after several other cells have been filled; revealing the easy cells first therefore makes the harder cells easier to predict. In contrast, the current diffusion models (MDLM and MGDM (Ye et al., 2025)) reveal cells without exploiting such an order. Each puzzle has 81 cells and some given cells. We use the Sudoku dataset of Palm et al. (2018) with 180k training, 18k validation, and 18k test puzzles. The given cells are never masked and are excluded from the loss.

Hierarchy labels. Starting from the given cells, we repeatedly fill the masked cells that can be determined directly. A cell is assigned a level according to the round in which it is first determined. We use six levels $( K =$ 6): level 5 contains cells determined directly from the given cells, while lower levels contain cells that require more rounds of solving. Level 0 contains cells that are determined only after six or more rounds, or that are never determined by these rules and therefore require search. At test time, the levels of blank cells are predicted by the hierarchy head. Architecture details for both variants (25M and 6M) are provided in Appendix F.

Table 5: Sudoku-hard results: 25M parameter models under stochastic & greedy decoding.
<table><tr><td>Metric</td><td>Model</td><td>stochastic Greedy</td><td></td></tr><tr><td rowspan="2">Board Acc.</td><td>MDLM</td><td>0.81</td><td>0.83</td></tr><tr><td>HMDLM</td><td>0.86</td><td>0.88</td></tr><tr><td rowspan="2">Cell Acc.</td><td>MDLM</td><td>0.94</td><td>0.94</td></tr><tr><td>HMDLM</td><td>0.96</td><td>0.97</td></tr></table>

Results. HMDLM consistently outperforms the baseline in both solve rate and cell accuracy: for the 25M model it wins under both stochastic and greedy decoding (Table 5), and the same advantage holds for the compact 6M model (Table 6). We also analyze the cell accuracy for each hierarchy level (Table 8), HMDLM shows the largest gains, up to 8% over MDLM for level 0 cells (or the cells that are most difficult to fill). Additionally, we ran a qualitative analysis on Sudoku puzzles and found that, for the 25M model, among 287 puzzles solved by HMDLM but not MDLM across all three seeds, 45.6% showed MDLM making an early commitment to a search cell, while HMDLM delayed filling the same cell until fewer candidates remained. Details in Appendix F

Table 6: Sudoku-hard test set (18k puzzles). Results are from 6M-parameter models trained for 100k steps with $T = 8 1$ and confidence-based sampling. Mean ± std over 3 sampling seeds.
<table><tr><td>Model</td><td>Board Acc.</td><td>Cell Acc.</td></tr><tr><td>MGDM</td><td> $6 3 . 0 7 \pm 0 . 1 8$ </td><td> $8 5 . 7 9 \pm 0 . 1 0$ </td></tr><tr><td>HMDLM</td><td> ${ \bf 7 1 . 6 9 \pm 0 . 2 6 }$ </td><td> ${ \bf 9 1 . 1 3 \pm 0 . 0 7 }$ </td></tr></table>

## 6 CONCLUSION AND LIMITATIONS

We introduced HMDLM, a hierarchical extension of masked diffusion language models that assigns priority to token positions and incorporates it into both the forward and reverse processes. By using level-specific noise schedules and a two-stage architecture that first predicts hierarchy and then tokens, HMDLM enables structured and controllable generation while remaining lightweight and modular. Our model consistently outperforms strong diffusion baselines by decoding semantically important tokens early and using them to better condition the rest of the sequence. While this improvement comes at the cost of additional computation due to the two-stage design, our results highlight hierarchy-aware generation as a promising direction for diffusion-based language models.

## REFERENCES

Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. Structured denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993, 2021.

Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, Mingliang Gong, Zhuochen Gong, Yanmei Gu, Jiaqi Hu, Zenan Huang, Zhenzhong Lan, et al. Llada2. 0: Scaling up diffusion language models to 100b. arXiv preprint arXiv:2512.15745, 2025.

Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn, and Tony Robinson. One billion word benchmark for measuring progress in statistical language modeling. arXiv preprint arXiv:1312.3005, 2013.

David Fox, Sam Bowyer, Song Liu, Laurence Aitchison, Raul Santos-Rodriguez, and Mengyue Yang. Learning generation orders for masked discrete diffusion models via variational inference. In 2nd Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Efficacy, ICLR 2026, 2026.

Prateek Garg, Bhavya Kohli, and Sunita Sarawagi. Masked diffusion models are secretly learnedorder autoregressive models. In EurIPS 2025 Workshop on Principles of Generative Modeling (PriGM), 2025.

Aaron Gokaslan and Vanya Cohen. Openwebtext corpus. http://Skylion007.github.io/ OpenWebTextCorpus, 2019.

Shansan Gong, Ruixiang Zhang, Huangjie Zheng, Jiatao Gu, Navdeep Jaitly, Lingpeng Kong, and Yizhe Zhang. Diffucoder: Understanding and improving masked diffusion models for code generation. arXiv preprint arXiv:2506.20639, 2025.

Zhengfu He, Tianxiang Sun, Qiong Tang, Kuanning Wang, Xuan-Jing Huang, and Xipeng Qiu. Diffusionbert: Improving generative masked language models with diffusion models. In Proceedings of the 61st annual meeting ofthe associationfor computational linguistics (volume 1: Long papers), pp. 4521–4534, 2023.

Chunsan Hong, Sanghyun Lee, and Jong Chul Ye. Unifying masked diffusion models with various generation orders and beyond. arXiv preprint arXiv:2602.02112, 2026.

Pengcheng Huang, Shuhao Liu, Zhenghao Liu, Yukun Yan, Shuo Wang, Zulong Chen, and Tong Xiao. Pc-sampler: Position-aware calibration of decoding bias in masked diffusion models. arXiv preprint arXiv:2508.13021, 2025.

Daniel Israel, Tian Jin, Ellie Cheng, Guy Van den Broeck, Aditya Grover, Suvinay Subramanian, and Michael Carbin. Planned diffusion. arXiv preprint arXiv:2510.18087, 2025.

Jaeyeon Kim, Kulin Shah, Vasilis Kontonis, Sham M Kakade, and Sitan Chen. Train for the worst, plan for the best: Understanding token ordering in masked diffusions. In International Conference on Machine Learning, pp. 30749–30768. PMLR, 2025.

Mahnaz Koupaee and William Yang Wang. Wikihow: A large scale text summarization dataset, 2018. URL https://arxiv.org/abs/1810.09305.

Xiang Li, John Thickstun, Ishaan Gulrajani, Percy S Liang, and Tatsunori B Hashimoto. Diffusion-lm improves controllable text generation. Advances in neural information processing systems, 35: 4328–4343, 2022.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. In Proceedings of the 41st International Conference on Machine Learning, pp. 32819–32848, 2024.

Shen Nie, Fengqi Zhu, Chao Du, Tianyu Pang, Qian Liu, Guangtao Zeng, Min Lin, and Chongxuan Li. Scaling up masked diffusion models on text. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 82974–82997, 2025a. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ ce1c1ff5d94079dea348a2317a889281-Paper-Conference.pdf.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025b.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models, 2025c. URL https: //arxiv.org/abs/2502.09992.

Rasmus Palm, Ulrich Paquet, and Ole Winther. Recurrent relational networks. In S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 31, pp. 3372–3382. Curran Associates, Inc., 2018. URL http://nips.cc.

William Peebles and Saining Xie. Scalable diffusion models with transformers, 2023. URL https: //arxiv.org/abs/2212.09748.

Fred Zhangzhi Peng, Zachary Bezemek, Jarrid Rector-Brooks, Shuibai Zhang, Anru R Zhang, Michael Bronstein, Alexander Tong, and Avishek Joey Bose. Planner aware path learning in diffusion language models training. arXiv preprint arXiv:2509.23405, 2025.

Litu Rout, Constantine Caramanis, and Sanjay Shakkottai. Anchored diffusion language model, 2025. URL https://arxiv.org/abs/2505.18456.

Subham S Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems, 37:130136–130184, 2024.

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis Titsias. Simplified and generalized masked diffusion for discrete data. Advances in neural information processing systems, 37: 103131–103167, 2024.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Guanghan Wang, Yair Schiff, Subham Sahoo, and Volodymyr Kuleshov. Remasking discrete diffusion models with inference-time scaling. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=IJryQAOy0p.

Zhihui Xie, Jiacheng Ye, Lin Zheng, Jiahui Gao, Jingwei Dong, Zirui Wu, Xueliang Zhao, Shansan Gong, Xin Jiang, Zhenguo Li, et al. Dream-coder 7b: An open diffusion language model for code. arXiv preprint arXiv:2509.01142, 2025.

Anton Xue, Litu Rout, Constantine Caramanis, and Sanjay Shakkottai. Ancoder: Anchored code generation via discrete diffusion models. arXiv preprint arXiv:2602.17688, 2026.

Jiacheng Ye, Jiahui Gao, Shansan Gong, Lin Zheng, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Beyond autoregression: Discrete diffusion for complex reasoning and planning. In International Conference on Learning Representations (ICLR), 2025.

Lin Zheng, Jianbo Yuan, Lei Yu, and Lingpeng Kong. A reparameterized discrete diffusion model for text generation. arXiv preprint arXiv:2302.05737, 2023.

## A DISCRETE-TIME ELBO

## A.1 JOINT GENERATIVE MODEL $p _ { \theta } ( \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } )$

Hierarchy is jointly modelled with tokens. We represent the state of the model at time step t as $\mathbf { s _ { t } } = ( \mathbf { x _ { t } } , \mathbf { h _ { t } } )$ . When token i is masked $( x _ { t } ^ { i } = M )$ , both the token and its hierarchy level are masked: the token is replaced by [MASK] and $h _ { t } ^ { i }$ is set to masked hierarchy state $( h _ { M A S K } )$ . HMDLM therefore targets the joint distribution:

$$
\log p _ { \theta } ( \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) .\tag{13}
$$

For $s < t$ and a masked position $( x _ { t } ^ { i } = m )$ , both $x _ { 0 } ^ { i }$ and $h _ { 0 } ^ { i }$ are revealed together upon unmasking, with the hierarchy fixed consistently to $h _ { 0 } ^ { i } = f ( x _ { 0 } ^ { i } )$

$$
( x _ { s } ^ { i } , h _ { s } ^ { i } ) = \left\{ \begin{array} { l l l } { ( x _ { 0 } ^ { i } , h _ { 0 } ^ { i } ) } & { \mathrm { w i t h ~ p r o b . } } & { \displaystyle \frac { \alpha _ { h _ { 0 } ^ { i } } ( s ) - \alpha _ { h _ { 0 } ^ { i } } ( t ) } { 1 - \alpha _ { h _ { 0 } ^ { i } } ( t ) } } & { \mathrm { ( j o i n t l y ~ r e v e a l e d ) } } \\ { ( m , \ h _ { m a s k } ) } & { \mathrm { w i t h ~ p r o b . } } & { \displaystyle \frac { 1 - \alpha _ { h _ { 0 } ^ { i } } ( s ) } { 1 - \alpha _ { h _ { 0 } ^ { i } } ( t ) } } & { \mathrm { ( j o i n t l y ~ r e m a i n ~ m a s k e d ) } } \end{array} \right.\tag{14}
$$

## A.2 STEP 1 — INTRODUCE THE JOINT DIFFUSION PROCESS

We target the joint distribution log $p _ { \theta } ( \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } )$ . The joint $\left( \mathbf { x } _ { t } , \mathbf { h } _ { t } \right)$ is the diffusion state at time t: when $x _ { t } ^ { i } = \bar { M }$ , both the token and its hierarchy label are masked and unobserved; when $x _ { t } ^ { i } \neq M ,$ , both are revealed together. Introduce the latent noisy sequence of joint states $\left( \mathbf { x } _ { 1 : T } , \mathbf { h } _ { 1 : T } \right)$ at $T$ discrete time steps $t _ { 1 } < \cdots < t _ { T } = 1$

$$
\log p _ { \theta } ( \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) = \log \int p _ { \theta } ( \mathbf { x } _ { 0 : T } , \mathbf { h } _ { 0 : T } ) \ d \mathbf { x } _ { 1 : T } \ d \mathbf { h } _ { 1 : T } .\tag{15}
$$

Because each denoising step depends only on the current state and not on the whole history, the joint model breaks into a product of one-step reverse predictions:

$$
p _ { \theta } ( \mathbf { x } _ { 0 : T } , \mathbf { h } _ { 0 : T } ) = p ( \mathbf { x } _ { T } , \mathbf { h } _ { T } ) \prod _ { t = 1 } ^ { T } p _ { \theta } ( \mathbf { x } _ { t - 1 } , \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } )\tag{16}
$$

We now split each reverse step into two parts: first predict the hierarchy, then predict the token given that hierarchy. We also drop $\mathbf { h } _ { t }$ from the token prediction, since $\mathbf { h } _ { t - 1 }$ already carries the hierarchy information needed. Finally, since the last state is fully masked and fixed, its prior splits into a product $p ( \mathbf { x } _ { T } ) p ( \mathbf { h } _ { T } )$ . This gives:

$$
p _ { \theta } ( \mathbf { x } _ { 0 : T } , \mathbf { h } _ { 0 : T } ) = p ( \mathbf { x } _ { T } ) p ( \mathbf { h } _ { T } ) \prod _ { t = 1 } ^ { T } \underbrace { p _ { \theta } ( \mathbf { h } _ { t - 1 } \mid \mathbf { h } _ { t } , \mathbf { x } _ { t } ) } _ { \mathrm { p a s ~ 1 : h i e r a c h y } } \cdot \underbrace { p _ { \theta } ( \mathbf { x } _ { t - 1 } \mid \mathbf { h } _ { t - 1 } , \mathbf { x } _ { t } ) } _ { \mathrm { P a s s 2 : t o k e n s } } .\tag{17}
$$

Multiply and divide by the forward process $q ( \mathbf { x } _ { 1 : T } , \mathbf { h } _ { 1 : T } \mid \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } )$ in 15:

$$
\log p _ { \theta } ( \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) = \log \mathbb { E } _ { q ( \mathbf { x } _ { 1 : T } , \mathbf { h } _ { 1 : T } | \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) } \biggl [ \frac { p _ { \theta } ( \mathbf { x } _ { 0 : T } , \mathbf { h } _ { 0 : T } ) } { q ( \mathbf { x } _ { 1 : T } , \mathbf { h } _ { 1 : T } \mid \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) } \biggr ] .\tag{18}
$$

## A.3 STEP $2 - \mathrm { A }$ PPLY JENSEN’S INEQUALITY

Since log is concave, Jensen’s inequality gives:

$$
\begin{array} { r } { \log p _ { \theta } ( \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) \ \geq \ \mathbb { E } _ { q ( \mathbf { x } _ { 1 : T } , \mathbf { h } _ { 1 : T } | \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) } \left[ \log \frac { p _ { \theta } ( \mathbf { x } _ { 0 : T } , \mathbf { h } _ { 0 : T } ) } { q ( \mathbf { x } _ { 1 : T } , \mathbf { h } _ { 1 : T } \mid \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) } \right] . } \end{array}\tag{19}
$$

## A.4 STEP 3 — EXPAND THE LOG OF THE JOINT MODEL

The forward process factorises as:

$$
q ( \mathbf { x } _ { 1 : T } , \mathbf { h } _ { 1 : T } \mid \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) = \prod _ { t = 1 } ^ { T } q ( \mathbf { x } _ { t } , \mathbf { h } _ { t } \mid \mathbf { x } _ { t - 1 } , \mathbf { h } _ { t - 1 } ) .\tag{20}
$$

Taking the log of equation 17 gives:

$$
\begin{array} { r } { \log p _ { \theta } ( \mathbf { x } _ { 0 : T } , \mathbf { h } _ { 0 : T } ) = \log p ( \mathbf { x } _ { T } ) + \log p ( \mathbf { h } _ { T } ) + \underbrace { \underset { t = 1 } { \overset { T } { \sum } } \log p _ { \theta } ( \mathbf { h } _ { t - 1 } \mid \mathbf { h } _ { t } , \mathbf { x } _ { t } ) } _ { \mathrm { P a s s ~ 1 : ~ h i e r a c h y } } } \\ { + \underbrace { \underset { t = 1 } { \overset { T } { \sum } } \log p _ { \theta } ( \mathbf { x } _ { t - 1 } \mid \mathbf { h } _ { t - 1 } , \mathbf { x } _ { t } ) } _ { \mathrm { p a s s 2 : t o k e n s } } . } \end{array}\tag{21}
$$

Substituting into equation 19:

$$
\begin{array} { r l } { \log p _ { \theta } ( \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) \geq } & { \mathbb { E } _ { q } \Big [ \log p ( \mathbf { x } _ { T } ) + \log p ( \mathbf { h } _ { T } ) } \\ & { \qquad + \displaystyle \sum _ { t = 1 } ^ { T } \log p _ { \theta } ( \mathbf { h } _ { t - \cdot } \mid \mathbf { h } _ { t } , \mathbf { x } _ { t } ) } \\ & { \qquad + \displaystyle \sum _ { t = 1 } ^ { T } \log p _ { \theta } ( \mathbf { x } _ { t - \cdot } \mid \mathbf { h } _ { t - \cdot } , \mathbf { x } _ { t } ) } \\ & { \qquad - \displaystyle \sum _ { t = 1 } ^ { T } \log q ( \mathbf { x } _ { t } , \mathbf { h } _ { t } \mid \mathbf { x } _ { t - \cdot } , \mathbf { h } _ { t - \cdot } ) \Big ] . } \end{array}\tag{22}
$$

## A.5 STEP 4 — TELESCOPE THE FORWARD TERMS

Key identity. By Bayes’ rule on the joint forward process:

$$
q ( \mathbf { x } _ { t - 1 } , \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } , \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) = { \frac { q ( \mathbf { x } _ { t } , \mathbf { h } _ { t } \mid \mathbf { x } _ { t - 1 } , \mathbf { h } _ { t - 1 } ) \ q ( \mathbf { x } _ { t - 1 } , \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) } { q ( \mathbf { x } _ { t } , \mathbf { h } _ { t } \mid \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) } } .\tag{23}
$$

Rearranging:

$$
\begin{array} { r l } & { \log q ( \mathbf { x } _ { t } , \mathbf { h } _ { t } \mid \mathbf { x } _ { t - 1 } , \mathbf { h } _ { t - 1 } ) = \log q ( \mathbf { x } _ { t - 1 } , \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } , \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) } \\ & { \qquad + \log q ( \mathbf { x } _ { t } , \mathbf { h } _ { t } \mid \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) - \log q ( \mathbf { x } _ { t - 1 } , \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) . } \end{array}\tag{24}
$$

Telescoping. Substituting equation 24 into $\begin{array} { r } { \sum _ { t = 1 } ^ { T } \log q ( \mathbf x _ { t } , \mathbf h _ { t } \mid \mathbf x _ { t - 1 } , \mathbf h _ { t - 1 } ) } \end{array}$ and using $q ( \mathbf { x } _ { 0 }$ , h<sub>0</sub> | $\mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) = 1 \colon$

$$
\sum _ { t = 1 } ^ { T } \log q ( \mathbf { x } _ { t } , \mathbf { h } _ { t } \mid \mathbf { x } _ { t - 1 } , \mathbf { h } _ { t - 1 } ) = \sum _ { t = 1 } ^ { T } \log q ( \mathbf { x } _ { t - 1 } , \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } , \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) + \log q ( \mathbf { x } _ { T } , \mathbf { h } _ { T } \mid \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) .\tag{25}
$$

Substitute back. Applying equation 25 to equation 22, grouping $t = 1$ separately, and identifying each $( \log p \theta - \log q )$ pair $\begin{array} { r } { \mathrm { a s } { \bar { \mathbf { \Theta } } } - { \bar { D } } _ { \mathrm { K L } } ( q \| p ) } \end{array}$ :

$$
\begin{array} { r l } { \log p _ { \theta } ( \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) } & { \geq \underbrace { \mathbb { E } _ { q } \bigl [ \log p _ { \theta } ( \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } \mid \mathbf { x } _ { 1 } , \mathbf { h } _ { 1 } ) \bigr ] } _ { L _ { 0 } } } \\ & { \phantom { = } - \underbrace { T } _ { t = 2 } \underbrace { \mathbb { E } _ { q } \Bigl [ D _ { \mathrm { K L } } \bigl ( q ( \mathbf { x } _ { t - 1 } , \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } , \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) \mid \mid p _ { \theta } ( \mathbf { x } _ { t - 1 } , \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } ) \bigr ) \Bigr ] } _ { L _ { t } } } \\ & { \phantom { = } - \underbrace { D _ { \mathrm { K L } } \bigl ( q ( \mathbf { x } _ { T } , \mathbf { h } _ { T } \mid \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) \mid \mid p ( \mathbf { x } _ { T } , \mathbf { h } _ { T } ) \bigr ) } _ { L _ { T } } . } \end{array}\tag{26}
$$

Here $p ( \mathbf { x } _ { T } , \mathbf { h } _ { T } ) = p ( \mathbf { x } _ { T } ) p ( \mathbf { h } _ { T } )$ since the terminal prior factorises. The reverse model factorises as

$$
p _ { \theta } ( \mathbf { x } _ { t - 1 } , \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } ) = p _ { \theta } ( \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } ) \cdot p _ { \theta } ( \mathbf { x } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t - 1 } )\tag{27}
$$

We approximate the reverse process $p _ { \theta } ( \mathbf { x } _ { t - 1 } , \mathbf { h } _ { t - 1 } | \mathbf { x } _ { t } , \mathbf { h } _ { t } )$ using the neural network $s _ { \theta } ( \mathbf { x } _ { t } , \mathbf { h } _ { t } , t )$ which is composed of two joined networks $- \ p _ { \theta } ( \mathbf h _ { t - 1 } \ \mid \ \mathbf x _ { t } , \mathbf h _ { t } )$ is modeled as $h _ { \theta } ( \mathbf { x } _ { t } , \mathbf { h } _ { t } , t )$ and $p _ { \theta } ( \mathbf { x } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t - 1 } )$ is modeled as $x _ { \theta } ( \mathbf { x } _ { t } , \mathbf { h } _ { t - 1 } , t )$

## A.6 STEP 5 — DECOMPOSE THE JOINT KL TERM $L _ { t }$ (FULL DERIVATION)

Each $L _ { t }$ is a KL over the joint state $\left( \mathbf { x } _ { t - 1 } , \mathbf { h } _ { t - 1 } \right)$ , so that the term of interest is $\begin{array} { r l } { L _ { t } } & { { } = } \end{array}$ $D _ { \mathrm { K L } } \big ( q ( \mathbf { x } _ { t - 1 } , \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } , \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) \mid \mid p _ { \theta } ( \mathbf { x } _ { t - 1 } , \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } ) \big )$

Chain rule of KL divergence. For any two joint distributions $q ( a , b \mid c )$ and $p ( a , b \mid d )$ that are each factorised in the same variable order, the KL divergence obeys the exact identity

$$
D _ { \mathrm { K L } } \big ( q ( a , b \mid c ) \| p ( a , b \mid d ) \big ) = D _ { \mathrm { K L } } \big ( q ( a \mid c ) \| p ( a \mid d ) \big ) + \mathbb { E } _ { q ( a \mid c ) } \Big [ D _ { \mathrm { K L } } \big ( q ( b \mid a , c ) \| p ( b \mid a , d ) \big ) \Big ] .\tag{28}
$$

Hierarchy-first factorisation. We apply equation 28 with $a = \mathbf { h } _ { t - 1 }$ (hierarchy) and $b = { \bf x } _ { t - 1 }$ (token), matching the two-stage structure of the model. The true posterior factorises by the product rule ,

$$
q ( \mathbf { x } _ { t - 1 } , \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } , \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) = q ( \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } , \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) q ( \mathbf { x } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t - 1 } , \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) ,\tag{29}
$$

Note: We drop $h _ { t }$ since $h _ { t - 1 }$ contains the latest state of the model.

The reverse model is defined to factorise in the same order - Pass 1 predicts $\mathbf { h } _ { t - 1 }$ and Pass 2 predicts $\mathbf { x } _ { t - 1 }$ conditioned on it

$$
p _ { \theta } ( \mathbf { x } _ { t - 1 } , \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } ) = \underbrace { p _ { \theta } ( \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } ) } _ { \mathrm { P a s s ~ 1 : ~ h i e r a r c h y ~ h e a d } } \underbrace { p _ { \theta } ( \mathbf { x } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t - 1 } ) } _ { \mathrm { P a s s ~ 2 : ~ t o k e n ~ h e a d } } .\tag{30}
$$

Apply the chain rule. Substituting the factorisations equation 29 and equation 30 into the identity equation 28 yields the two-term decomposition

$$
\begin{array} { r l } & { D _ { \mathrm { K L } } \big ( q ( \mathbf { x } _ { t - 1 } , \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } , \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) \mid \mid p _ { \theta } ( \mathbf { x } _ { t - 1 } , \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } ) \big ) } \\ & { \quad = \underbrace { D _ { \mathrm { K L } } \big ( q ( \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } , \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) \mid \mid p _ { \theta } ( \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } ) \big ) } _ { \mathrm { h i e r a c h y ~ K L ~ ( P a s ~ 1 ) } } } \\ & { \quad \quad + \underbrace { \mathbb { E } _ { q ( \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } , \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) } \Big [ D _ { \mathrm { K L } } \big ( q ( \mathbf { x } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t - 1 } , \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) \mid \mid p _ { \theta } ( \mathbf { x } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t - 1 } ) \big ) \Big ] } _ { \mathrm { t o k e n ~ K L ~ ( P a s s ~ 2 ) } } . } \end{array}\tag{31}
$$

## A.7 STEP 6 — SIMPLIFY THE TERMINAL TERM $L _ { T }$

$\mathbf { A } \mathbf { t } \ t = T$ , every token and hierarchy label is fully masked with probability $1 - \alpha _ { k } ( 1 ) \approx 1$ . The joint prior $p ( \mathbf { x } _ { T } , \mathbf { h } _ { T } )$ is therefore also the fully-masked state, so

$$
L _ { T } = D _ { \mathrm { K L } } \big ( q ( \mathbf { x } _ { T } , \mathbf { h } _ { T } \mid \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) \mid \mid p ( \mathbf { x } _ { T } , \mathbf { h } _ { T } ) \big ) \approx 0 .\tag{32}
$$

We drop this term.

## A.8 STEP 7 — FULL JOINT DISCRETE ELBO

$$
\begin{array} { r l } { \mathrm { { D r o p p i n g ~ } } L _ { T } = 0 ; } \\ { \log p _ { \theta } ( \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) } & { \geq \underbrace { { \mathbb { E } } _ { \boldsymbol { q } } [ \log p _ { \theta } ( \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } \mid \mathbf { x } _ { 1 } , \mathbf { h } _ { 1 } ) ] } _ { L _ { 0 } } } \\ & { \phantom { m m m m m m m m m m } - \underbrace { T } _ { t = 2 } ^ { T } { \mathbb { E } } _ { \boldsymbol { q } } [ D _ { \mathrm { { K L } } } ( q ( \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } , \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) \parallel \mathcal { p } _ { \theta } ( \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } ) ) ] } \\ & { - \underbrace { T } _ { t = 2 } ^ { T } { \mathbb { E } } _ { \boldsymbol { q } } [ \mathbf { \overline { { B _ { \boldsymbol { q } } ( \mathbf { H } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } , \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) } [ D _ { \mathrm { { K L } } } ( q ( \mathbf { x } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t - 1 } , \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) ) ] } \ \lVert \ { p _ { \theta } } ( \mathbf { x } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t - 1 } ) ) ] } _ { L _ { t } ^ { \mathrm { s t } } ( \mathrm { s t a r e a t i v i t y ~ h a r d } ) } } \end{array}\tag{33}
$$

## A.9 STEP 8 — EXPAND THE HIERARCHY KL (PASS 1)

From equation 33, the hierarchy contribution at reverse step t is:

$$
L _ { t } ^ { \mathrm { h i e r } } = \mathbb { E } _ { q } [ D _ { \mathrm { K L } } ( q ( \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } , \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) \parallel p _ { \theta } ( \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } ) ) ] .\tag{34}
$$

Since the forward process factorizes independently across sequence positions, and the Pass-1 hierarchy head predicts each position independently given the noisy state $\left( \mathbf { x } _ { t } , \mathbf { h } _ { t } \right)$

$$
q ( \mathbf { h } _ { t - 1 } \mid \cdots ) = \prod _ { i = 1 } ^ { L } q ( h _ { t - 1 } ^ { i } \mid \cdots ) ,\tag{35}
$$

$$
p _ { \theta } ( \mathbf { h } _ { t - 1 } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } ) = \prod _ { i = 1 } ^ { L } p _ { \theta } { \big ( } h _ { t - 1 } ^ { i } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } { \big ) } .\tag{36}
$$

The joint KL therefore decomposes into a sum of independent per-position KLs,

$$
L _ { t } ^ { \mathrm { h i e r } } = \mathbb { E } _ { q } \left[ \sum _ { i = 1 } ^ { L } D _ { \mathrm { K L } } \big ( q ( h _ { t - 1 } ^ { i } \mid \cdots ) \parallel p _ { \theta } ( h _ { t - 1 } ^ { i } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } ) \big ) \right] .\tag{37}
$$

Notation. Let

$$
g _ { h } ^ { i } ( t ) : = [ h _ { \theta } ^ { i } ( \mathbf { x } _ { t } , \mathbf { h } _ { t } , t ) ] _ { h } , \qquad h \in \{ 1 , \ldots , K \} , \qquad \sum _ { h = 1 } ^ { K } g _ { h } ^ { i } ( t ) = 1 ,\tag{38}
$$

denote the Pass-1 softmax distribution over the K hierarchy classes at position i. For convenience we write

$$
\lambda \equiv \lambda _ { h _ { 0 } ^ { i } } ( t ) , \qquad \pi \equiv \pi _ { i } ( t ) = \sum _ { k = 1 } ^ { K } g _ { k } ^ { i } ( t ) \lambda _ { k } ( t ) ,\tag{39}
$$

where $\lambda _ { h _ { 0 } ^ { i } } \left( t \right)$ is the gold reveal probability associated with the true hierarchy class $h _ { 0 } ^ { i }$ , and $\lambda _ { k } ( t )$ is the (schedule-determined) reveal probability of class k. The quantity $\pi _ { i } ( t )$ is the model’s total predicted reveal probability, obtained by averaging the class-dependent reveal probability $\lambda _ { k } ( t )$ under the Pass-1 distribution $g ^ { i } ( t )$

Forward posterior at masked positions. At an unmasked position $( s _ { t } ^ { i } \neq s _ { m } )$ we have $s _ { t - 1 } ^ { i } = s _ { 0 } ^ { i }$ deterministically, contributing zero KL. At a masked position the absorbing-state posterior is

$$
\small q ( s _ { t - 1 } ^ { i } \mid s _ { t } ^ { i } , s _ { 0 } ^ { i } ) = \left\{ \begin{array} { l l } { \mathrm { C a t } \big ( s _ { t - 1 } ^ { i } ; ~ s _ { t } ^ { i } \big ) , } & { s _ { t } ^ { i } \ne \mathbf { s _ { m } } , } \\ { \mathrm { C a t } \left( s _ { t - 1 } ^ { i } ; \frac { \left( 1 - \alpha _ { h _ { 0 } ^ { i } } ( t - 1 ) \right) \mathbf { s _ { m } } + \left( \alpha _ { h _ { 0 } ^ { i } } ( t - 1 ) - \alpha _ { h _ { 0 } ^ { i } } ( t ) \right) s _ { 0 } ^ { i } } { 1 - \alpha _ { h _ { 0 } ^ { i } } ( t ) } \right) , } & { s _ { t } ^ { i } = \mathbf { s _ { m } } . } \end{array} \right.\tag{40}
$$

Reverse hierarchy model (Pass 1). The head first assigns position i to class h with probability $g _ { h } ^ { i } ( t )$ , and each class then reveals at its own known schedule probability $\lambda _ { h } ( t )$ . The reverse model is thus

$$
\begin{array} { r } { p _ { \theta } ( h _ { t - 1 } ^ { i } = h \vert \mathbf { x } _ { t } , \mathbf { h } _ { t } ) = \lambda _ { h } ( t ) g _ { h } ^ { i } ( t ) , \qquad h \in \{ 1 , \ldots , K \} , } \end{array}\tag{41}
$$

$$
p _ { \theta } ( h _ { t - 1 } ^ { i } = m \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } ) = 1 - \pi , \qquad \pi _ { i } ( t ) = \sum _ { k = 1 } ^ { K } g _ { k } ^ { i } ( t ) \lambda _ { k } ( t ) .\tag{42}
$$

Each class here reveals at its own probability $\lambda _ { h } ( t )$ , and on the gold class this gives the gold probability directly,

$$
\begin{array} { r } { p _ { \theta } ( h _ { t - 1 } ^ { i } = h _ { 0 } ^ { i } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } ) = \lambda _ { h _ { 0 } ^ { i } } ( t ) g _ { h _ { 0 } ^ { i } } ^ { i } ( t ) = \lambda g _ { h _ { 0 } ^ { i } } ^ { i } ( t ) . } \end{array}\tag{43}
$$

Per-position KL.

$$
\frac { h _ { t - 1 } ^ { i } } { h _ { 0 } ^ { i } } \qquad \lambda \qquad \lambda g _ { h _ { 0 } ^ { i } } ^ { i } ( t )
$$

The KL divergence is

$$
D _ { \mathrm { K L } } ( q \Vert p _ { \theta } ) = \sum _ { v } q ( v ) \log \frac { q ( v ) } { p _ { \theta } ( v ) } , \qquad v \in \{ 1 , \dots , K , h _ { \mathrm { m a s k } } \} .\tag{44}
$$

Since $q ( v ) = 0$ for every incorrect hierarchy class $( h \neq h _ { 0 } ^ { i } )$ , all such terms vanish — the off-gold entries $\dot { \lambda _ { h } ( t ) } g _ { h } ^ { i } ( t )$ never enter the KL and serve only to normalize $p _ { \theta }$ . Only the gold-class and mask rows remain,

$$
D _ { \mathrm { K L } } = \lambda \log \frac { \lambda } { \lambda g _ { h _ { 0 } ^ { i } } ^ { i } ( t ) } + ( 1 - \lambda ) \log \frac { 1 - \lambda } { 1 - \pi } .\tag{45}
$$

In the gold-class row the reveal rate λ cancels between numerator and denominator, leaving a pure cross-entropy on $g ^ { i } \colon$

$$
\begin{array} { l } { { \displaystyle { D _ { \mathrm { K L } } = \lambda \log \frac { 1 } { g _ { h _ { 0 } ^ { i } } ( t ) } + ( 1 - \lambda ) \log \frac { 1 - \lambda } { 1 - \pi } } } } \\ { { \displaystyle { } } } \\ { { \displaystyle { = - \lambda \log g _ { h _ { 0 } ^ { i } } ^ { i } ( t ) + ( 1 - \lambda ) \log \frac { 1 - \lambda } { 1 - \pi } } . } } \end{array}\tag{46}
$$

Hence

$$
D _ { \mathrm { K L } } ^ { \mathrm { h i e r , i } } = \underbrace { - \lambda \log g _ { h _ { 0 } ^ { i } } ^ { i } ( t ) } _ { \mathrm { r e v e a l - w e i g h t e d c r o s s - e n t r o p y } } + \underbrace { ( 1 - \lambda ) \log \frac { 1 - \lambda } { 1 - \pi } } _ { \mathrm { e x t r a - t e r m } }\tag{47}
$$

Summing over all the positions,

$$
L _ { t } ^ { \mathrm { h i e r } } = \sum _ { i = 1 } ^ { L } \mathbb { E } _ { q } \left[ - \lambda _ { h _ { 0 } ^ { i } } ( t ) \log g _ { h _ { 0 } ^ { i } } ^ { i } ( t ) + \left( 1 - \lambda _ { h _ { 0 } ^ { i } } ( t ) \right) \log \frac { 1 - \lambda _ { h _ { 0 } ^ { i } } ( t ) } { 1 - \pi _ { i } ( t ) } \right] .\tag{48}
$$

## A.10 STEP 9 — EXPAND THE TOKEN KL (PASS 2)

We expand the per-position token contribution

$$
{ \cal L } _ { t } ^ { \mathrm { t o k } , i } = \mathbb { E } _ { q ( h _ { t - 1 } ^ { i } \mid \mathbf { x } _ { t } , \mathbf { h } _ { t } , \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) } \Big [ D _ { \mathrm { K L } } \big ( q ( x _ { t - 1 } ^ { i } \mid h _ { t - 1 } ^ { i } , \cdot \cdot \cdot , \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } ) \big \mid \big \Vert p _ { \theta } ( x _ { t - 1 } ^ { i } \mid h _ { t - 1 } ^ { i } , \mathbf { x } _ { t } ) \big ) \Big ] ,\tag{49}
$$

By the shared reveal variable ( equation 5), position i is governed by a single Bernoulli $b _ { t } ^ { i } \sim$ Bernoull $\left( 1 - \alpha _ { h _ { 0 } ^ { i } } ( t ) \right)$ , and the joint state $s _ { t } ^ { i } = \bar { ( x _ { t } ^ { i } , h _ { t } ^ { i } ) }$ is a deterministic push-forward of $b _ { t } ^ { i } \colon$ both streams reveal to $( x _ { 0 } ^ { i } , h _ { 0 } ^ { i } )$ when the position is kept and mask to $( m , h _ { \mathrm { m a s k } } )$ otherwise. Consequently the joint reverse posterior over $( x _ { t - 1 } ^ { i } , h _ { t - 1 } ^ { i } )$ is equation 40.

Branch decomposition. The outer expectation in equation 49 runs over the two values $h _ { t - 1 } ^ { i } \in$ $\{ h _ { 0 } ^ { i } , h _ { \mathrm { m a s k } } \}$ , with marginal weights λ and $1 - \lambda$ from equation 40. We treat each branch in turn.

Revealed branch $( h _ { t - 1 } ^ { i } = h _ { 0 } ^ { i }$ , weight λ).

$$
\begin{array} { r } { p _ { \theta } ( x _ { t - 1 } ^ { i } = v \mid h _ { 0 } ^ { i } , x _ { t } , h _ { t - 1 } ) = f _ { v } ^ { i } ( t ) \mathrm { ~ w i t h ~ } f _ { v } ^ { i } ( t ) : = [ x _ { \theta } ( x _ { t } , h _ { t - 1 } , t ) ] _ { v } \mathrm { ~ a n d ~ } \sum _ { v = 1 } ^ { V } f _ { v } ^ { i } ( t ) = 1 \times [ h _ { 0 } , h _ { t - 1 } , h _ { t } ] _ { v } } \end{array}
$$

$$
\begin{array} { r } { \frac { x _ { t - 1 } ^ { i } } { x _ { 0 } ^ { i } } \displaystyle \lvert \begin{array} { l } { q \rvert \quad p _ { \theta } } \\ { 1 \ u { \rvert } f _ { x _ { 0 } ^ { i } } ^ { i } ( t ) } \end{array} \quad \quad \implies \quad \quad D _ { \mathrm { K L } } ( q \parallel p _ { \theta } ) \vert _ { \mathrm { r e v } } = \sum _ { v } q _ { v } \log \frac { q _ { v } } { p _ { \theta , v } } = - \log f _ { x _ { 0 } ^ { i } } ^ { i } ( t ) . } \\ { m \quad \quad 0 \quad \quad 0 } \end{array}\tag{50}
$$

Masked branch $( h _ { t - 1 } ^ { i } = h _ { \operatorname* { m a s k } }$ , weight $1 - \lambda )$ .

$$
\begin{array} { r l r } { \frac { x _ { t - 1 } ^ { i } } { x _ { 0 } ^ { i } } \frac { \mid q \mid p _ { \theta } } { 0 } } & { { } \implies } & { { } \quad D _ { \mathrm { K L } } \big ( q \| p _ { \theta } \big ) \big | _ { \mathrm { m a s k } } = 0 . } \\ { \quad \left. v \neq x _ { 0 } ^ { i } \right| 0 \left| \begin{array} { l l l } { 0 } & { } & { \implies } \\ { 1 } \end{array} \right. } & { } & { { } \Longrightarrow \qquad D _ { \mathrm { K L } } \big ( q \| p _ { \theta } \big ) \big | _ { \mathrm { m a s k } } = 0 . } \end{array}\tag{51}
$$

Taking the outer expectation.

$$
\begin{array} { r l } & { L _ { t } ^ { \mathrm { t o k } , i } = \mathbb { E } _ { q } \Big [ \underbrace { \lambda } _ { P ( h _ { t - 1 } ^ { i } = h _ { 0 } ^ { i } ) } \cdot \big ( - \log f _ { x _ { 0 } ^ { i } } ^ { i } ( t ) \big ) \ + \ \underbrace { \big ( 1 - \lambda \big ) } _ { P ( h _ { t - 1 } ^ { i } = h _ { \mathrm { m a s k } } ) } \cdot 0 \Big ] } \\ & { = - \mathbb { E } _ { q } \big [ \lambda _ { h _ { 0 } ^ { i } } ( t ) \log f _ { x _ { 0 } ^ { i } } ^ { i } ( t ) \big ] . } \end{array}\tag{52}
$$

Summing over position,

$$
L _ { t } ^ { \mathrm { t o k } } = \sum _ { i = 1 } ^ { L } L _ { t } ^ { \mathrm { t o k } , i } = - \sum _ { i = 1 } ^ { L } \mathbb { E } _ { q } \big [ \lambda _ { h _ { 0 } ^ { i } } ( t ) \log f _ { x _ { 0 } ^ { i } } ^ { i } ( t ) \big ] .\tag{53}
$$

The token loss is thus a reveal-weighted cross-entropy on the Pass-2 softmax: each position contributes the negative log-likelihood of its clean token $x _ { 0 } ^ { i }$ , weighted by the class-dependent probability $\lambda _ { h _ { 0 } ^ { i } } \left( t \right)$ that the position is revealed at step $t - 1$

## DISCRETE → CONTINUOUS

Setup. Let $\begin{array} { r } { s = t - \frac { 1 } { T } } \end{array}$ , and let $\alpha _ { k } ( t )$ be the class-k survival (unmasked) probability, decreasing with $\alpha _ { k } ( 0 ) = 1 , \alpha _ { k } ( 1 ) = \dot { 0 }$

Step A — per-step masked-position KL

$$
L _ { t } ^ { \mathrm { t o k } , i } = - \frac { \alpha _ { h _ { 0 } ^ { i } } ( s ) - \alpha _ { h _ { 0 } ^ { i } } ( t ) } { 1 - \alpha _ { h _ { 0 } ^ { i } } ( t ) } \log f _ { x _ { 0 } ^ { i } } ^ { i } ( t ) .\tag{54}
$$

Step B — sum over timesteps.

$$
\mathcal { L } ^ { \mathrm { t o k } } = \sum _ { t } L _ { t } ^ { \mathrm { t o k } } = - \sum _ { t } \sum _ { i } \mathbb { E } _ { q } \left[ \frac { \alpha _ { h _ { 0 } ^ { i } } ( s ) - \alpha _ { h _ { 0 } ^ { i } } ( t ) } { 1 - \alpha _ { h _ { 0 } ^ { i } } ( t ) } \log f _ { x _ { 0 } ^ { i } } ^ { i } ( t ) \right] .\tag{55}
$$

Step C — expose the mask indicator.

$$
\mathcal { L } ^ { \mathrm { t o k } } = - \sum _ { t } \mathbb { E } _ { q } \left[ \sum _ { i } 1 [ x _ { t } ^ { i } = M ] \frac { \alpha _ { h _ { 0 } ^ { i } } ( s ) - \alpha _ { h _ { 0 } ^ { i } } ( t ) } { 1 - \alpha _ { h _ { 0 } ^ { i } } ( t ) } \log f _ { x _ { 0 } ^ { i } } ^ { i } ( t ) \right] .\tag{56}
$$

Step D — Riemann rescaling $( \sum _ { t }  \mathbb { E } _ { t } )$ .

$$
\sum _ { t } [ \cdot ] = \underbrace { \begin{array} { l l } { \frac { 1 } { T } \sum _ { t } } & { \underbrace { T [ \cdot ] } _ { \begin{array} { l } { \scriptstyle \mathscr { D } ( 1 ) } \end{array} } , } \end{array} } _ { \mathbb { E } _ { t } \in \{ 1 / T , \dots , 1 \} }\tag{57}
$$

$$
\mathcal { L } ^ { \mathrm { t o k } } = - \mathbb { E } _ { t } \mathbb { E } _ { q } \left[ \sum _ { i } { 1 } [ x _ { t } ^ { i } = M ] \left( T \cdot \frac { \alpha _ { h _ { 0 } ^ { i } } ( s ) - \alpha _ { h _ { 0 } ^ { i } } ( t ) } { 1 - \alpha _ { h _ { 0 } ^ { i } } ( t ) } \right) \log f _ { x _ { 0 } ^ { i } } ^ { i } ( t ) \right] .\tag{58}
$$

Step E — limit $T \to \infty :$ weight → reveal hazard.

$$
T \big ( \alpha _ { h _ { 0 } ^ { i } } ( s ) - \alpha _ { h _ { 0 } ^ { i } } ( t ) \big ) = \frac { \alpha _ { h _ { 0 } ^ { i } } ( t - \frac { 1 } { T } ) - \alpha _ { h _ { 0 } ^ { i } } ( t ) } { 1 / T } \xrightarrow [ T  \infty ] - \alpha _ { h _ { 0 } ^ { i } } ^ { \prime } ( t ) ,\tag{59}
$$

$$
T \cdot \frac { \alpha _ { h _ { 0 } ^ { i } } ( s ) - \alpha _ { h _ { 0 } ^ { i } } ( t ) } { 1 - \alpha _ { h _ { 0 } ^ { i } } ( t ) } \longrightarrow \frac { - \alpha _ { h _ { 0 } ^ { i } } ^ { \prime } ( t ) } { 1 - \alpha _ { h _ { 0 } ^ { i } } ( t ) } = : \lambda _ { h _ { 0 } ^ { i } } ^ { \prime } ( t ) .\tag{60}
$$

Step $\mathbf { F } -$ continuous-time loss.

$$
\mathcal { L } _ { \mathrm { L M } } = \mathbb { E } _ { t , q } \left[ \sum _ { i } 1 [ x _ { t } ^ { i } = M ] \lambda _ { h _ { 0 } ^ { i } } ^ { \prime } ( t ) \big ( - \log f _ { x _ { 0 } ^ { i } } ^ { i } ( t ) \big ) \right] .\tag{61}
$$

Step G — Hierarchy head (same derivation, $f  g )$ .

$$
\mathcal { L } _ { \mathrm { h i e r } } = \mathbb { E } _ { t , q } \left[ \sum _ { i } 1 [ x _ { t } ^ { i } = M ] \lambda _ { h _ { 0 } ^ { i } } ^ { \prime } ( t ) \big ( - \log g _ { h _ { 0 } ^ { i } } ^ { i } ( t ) \big ) \right] .\tag{62}
$$

Let $\begin{array} { r } { s = t - \frac { 1 } { T } } \end{array}$ , and

$$
\mathcal { E } _ { t } ^ { i } = \left( 1 - \lambda _ { h _ { 0 } ^ { i } } ( t ) \right) \log \frac { 1 - \lambda _ { h _ { 0 } ^ { i } } ( t ) } { 1 - \pi _ { i } ( t ) } ,\tag{61}
$$

Both $\lambda _ { h _ { 0 } ^ { i } } \left( t \right)$ and every $\lambda _ { k } ( t )$ are $O ( 1 / T )$ , hence so is $\pi _ { i } ( t )$

Step H — expose the mask indicator. Only masked positions contribute (unmasked positions give zero KL, cf. equation 49):

$$
\mathcal { E } _ { t } ^ { \mathrm { h i e r } } = \mathbb { E } _ { q } \left[ \sum _ { i } { 1 [ x _ { t } ^ { i } = M ] \left( 1 - \lambda _ { h _ { 0 } ^ { i } } ( t ) \right) \log \frac { 1 - \lambda _ { h _ { 0 } ^ { i } } ( t ) } { 1 - \pi _ { i } ( t ) } } \right] .\tag{62}
$$

Step I — small-step expansion. Write $\lambda \equiv \lambda _ { h _ { 0 } ^ { i } } ( t )$ and $\pi \equiv \pi _ { i } ( t )$ , both $O ( 1 / T )$ . Using log(1 − $\begin{array} { r } { z ) = - z - \frac { z ^ { 2 } } { 2 } - \cdot \cdot \cdot } \end{array}$

$$
\log \frac { 1 - \lambda } { 1 - \pi } = ( \pi - \lambda ) - \textstyle \frac { 1 } { 2 } \big ( \lambda ^ { 2 } - \pi ^ { 2 } \big ) + O ( T ^ { - 3 } ) .\tag{63}
$$

Multiplying by (1 − λ) and collecting terms by order,

$$
{ \mathcal E } _ { t } ^ { i } = \underbrace { ( \pi - \lambda ) } _ { O ( 1 / T ) } + \underbrace { \frac { 1 } { 2 } ( \lambda - \pi ) ^ { 2 } } _ { O ( 1 / T ^ { 2 } ) } + O ( T ^ { - 3 } ) .\tag{64}
$$

Step J — Riemann rescaling $( \sum _ { t }  \mathbb { E } _ { t } )$ .

$$
\sum _ { t } [ \cdot ] = \underbrace { \frac { 1 } { T } \sum _ { t } } _ { \mathbb { E } _ { t \in \{ 1 / T , \ldots , 1 \} } } \underbrace { T [ \cdot ] } _ { O ( 1 ) } , T ^ { \dot { \varepsilon } _ { t } ^ { i } } = \underbrace { T ( \pi - \lambda ) } _ { O ( 1 ) } + \underbrace { \frac { T } { 2 } ( \lambda - \pi ) ^ { 2 } } _ { O ( 1 / T ) \to 0 } .\tag{65}
$$

Step K — limit $T \to \infty :$ rate → reveal hazard. For each class $k ,$ as in Step E of the token derivation,

$$
T \lambda _ { k } ( t ) = \frac { \alpha _ { k } \big ( t - \frac { 1 } { T } \big ) - \alpha _ { k } ( t ) } { 1 / T } \cdot \frac { 1 } { 1 - \alpha _ { k } ( t ) } \xrightarrow [ T  \infty ] { } \frac { - \alpha _ { k } ^ { \prime } ( t ) } { 1 - \alpha _ { k } ( t ) } = : \lambda _ { k } ^ { \prime } ( t ) .\tag{66}
$$

Applying this to λ and, term by term, to π,

$$
T \lambda = T \lambda _ { h _ { 0 } ^ { i } } ( t ) \to \lambda _ { h _ { 0 } ^ { i } } ^ { \prime } ( t ) , \qquad T \pi = \sum _ { k } g _ { k } ^ { i } ( t ) T \lambda _ { k } ( t ) \to \sum _ { k } g _ { k } ^ { i } ( t ) \lambda _ { k } ^ { \prime } ( t ) ,\tag{67}
$$

so that

$$
T ( \pi - \lambda ) \xrightarrow [ T \to \infty ] { } \sum _ { k } g _ { k } ^ { i } ( t ) \lambda _ { k } ^ { \prime } ( t ) - \lambda _ { h _ { 0 } ^ { i } } ^ { \prime } ( t ) ,\tag{68}
$$

while the quadratic remainder vanishes.

Now, the $1 - \lambda _ { k } ( t )$ term,

$$
1 - \lambda _ { k } ( t ) = \frac { 1 - \alpha _ { k } \big ( t - \frac { 1 } { T } \big ) } { 1 - \alpha _ { k } ( t ) } .\tag{69}
$$

Since $\alpha _ { k }$ is continuous, $\alpha _ { k } \big ( t - \frac { 1 } { T } \big )  \alpha _ { k } ( t )$ as $T \to \infty$ . Moreover, $t > 0$ implies $1 - \alpha _ { k } ( t ) > 0$ so the denominator is nonzero. Therefore

$$
1 - \lambda _ { k } ( t ) \xrightarrow [ T \to \infty ] { } \frac { 1 - \alpha _ { k } ( t ) } { 1 - \alpha _ { k } ( t ) } = 1 .\tag{70}
$$

The only singular point is $t = 0 \colon$ since $1 - \alpha _ { k } ( 0 ) = 0$

$$
\lambda _ { k } ^ { \prime } ( t ) \xrightarrow [ t \to 0 ^ { + } ] { } \infty ,\tag{71}
$$

while $1 - \lambda _ { k } ( t ) \in ( 0 , 1 ]$ remains bounded and is undefined only at $t = 0$ itself.

In pratice, we sample $t \sim \mathcal { U } [ \varepsilon , 1 ]$ , and therefore never sample $t = 0$ and hence the expression always stays bounded, where $\varepsilon = 1 0 ^ { - 3 }$

Step L — continuous-time extra term.

$$
\mathcal { L } _ { \mathrm { c o n s } } = \mathbb { E } _ { t , q } \left[ \sum _ { i } { 1 [ x _ { t } ^ { i } = M ] } \Big ( \underbrace { \sum _ { k } g _ { k } ^ { i } ( t ) \lambda _ { k } ^ { \prime } ( t ) } _ { \mathrm { p r e d i c t e d r e v e a t ~ h a r a r d } } - \lambda _ { h _ { 0 } ^ { i } } ^ { \prime } ( t ) \Big ) \right] , \qquad \lambda _ { k } ^ { \prime } ( t ) = \frac { - \alpha _ { k } ^ { \prime } ( t ) } { 1 - \alpha _ { k } ( t ) } .\tag{72}
$$

## A.11 STEP $1 0 - \mathrm { F I N A I }$ L TRAINING LOSS

Training losses. The final training loss consists of the following three components:

$$
\mathcal { L } _ { \mathrm { L M } } = \mathbb { E } _ { t , q } \left[ \sum _ { i = 1 } ^ { L } \mathbf { 1 } [ x _ { t } ^ { i } = M ] \lambda _ { h _ { 0 } ^ { i } } ^ { \prime } ( t ) \left( - \log f _ { x _ { 0 } ^ { i } } ^ { i } ( t ) \right) \right] ,\tag{73}
$$

$$
\mathcal { L } _ { \mathrm { h i e r } } = \mathbb { E } _ { t , q } \left[ \sum _ { i = 1 } ^ { L } \mathbf { 1 } [ x _ { t } ^ { i } = M ] \lambda _ { h _ { 0 } ^ { i } } ^ { \prime } ( t ) \left( - \log g _ { h _ { 0 } ^ { i } } ^ { i } ( t ) \right) \right] ,\tag{74}
$$

$$
\mathcal { L } _ { \mathrm { c o n s } } = \mathbb { E } _ { t , q } \Bigg [ \sum _ { i = 1 } ^ { L } \mathbf { 1 } [ x _ { t } ^ { i } = M ] \left( \sum _ { k } g _ { k } ^ { i } ( t ) \lambda _ { k } ^ { \prime } ( t ) - \lambda _ { h _ { 0 } ^ { i } } ^ { \prime } ( t ) \right) \Bigg ] ,\tag{75}
$$

In practice we optimise a reweighted sum of the same three terms, replacing the tied weights above by tunable coefficients:

$$
{ \mathcal { L } } _ { \mathrm { H M D L M } } = { \mathcal { L } } _ { \mathrm { L M } } ~ + ~ { \beta } _ { \mathrm { h i e r } } { \mathcal { L } } _ { \mathrm { h i e r } } ~ + ~ { \beta } _ { \mathrm { c o n s } } { \mathcal { L } } _ { \mathrm { c o n s } } .\tag{76}
$$

```latex
Algorithm 1 HMDLM training<sup>2</sup>
Require: data D of clean pairs $\left( \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } \right)$ , with $x _ { 0 , i } \in \{ 1 , \ldots , V \} , h _ { 0 } ^ { i } \in \{ 1 , \ldots , K \}$ ; per-class keep
schedules $\{ \alpha _ { k } ( \cdot ) \}$ with reveal probabilities $\lambda _ { k } ( \cdot ) ;$ hierarchy head $h _ { \theta }$ (Pass 1), token head $x _ { \theta }$
(Pass 2); number of inference time steps T, weights $\beta _ { \mathrm { h i e r } } , \beta _ { \mathrm { c o n s } } ;$ learning rate η.
1: while not converged do
2: sample $( \mathbf { x } _ { 0 } , \breve { \mathbf { h } _ { 0 } } ) \sim \mathcal { D } , \quad t \sim \mathcal { U } ( \epsilon , 1 )$
// Forward corruption to level t
3: for $i = 1 , \dots , L$ do
4: $m _ { i } \sim \mathrm { B e r n o u l l i } \big ( 1 - \alpha _ { h _ { 0 } ^ { i } } ( t ) \big )$ ▷ $m _ { i } = 1 \colon$ masked at t
5: $( x _ { t } ^ { i } , h _ { t } ^ { i } ) \gets ( M , h _ { \mathrm { m a s k } } ) \operatorname { i f } m _ { i } { = } 1$ else $( x _ { 0 } ^ { i } , h _ { 0 } ^ { i } )$
6: end for
Pass 1 — hierarchy.
7: $g _ { i } ( t ) \gets \mathrm { s o f t m a x } \big ( \bar { h _ { \theta } ^ { i } } ( { \mathbf x } _ { t } , { \mathbf h } _ { t } , t ) \big ) , \quad \lambda _ { i } ^ { \prime } \gets \lambda _ { h _ { 0 } ^ { i } } ^ { \prime } ( t ) ,$ for all i
8: $\begin{array} { r } { \mathcal { L } _ { \mathrm { h i e r } }  \sum _ { i } m _ { i } \lambda _ { i } ^ { \prime } ( - \log g _ { h _ { \mathrm { } } ^ { i } } ^ { i } ( t ) ) } \end{array}$
9: $\begin{array} { r } { \mathcal { L } _ { \mathrm { c o n s } } \gets \sum _ { i } m _ { i } \left( \sum _ { k } g _ { k } ^ { i } ( t ) \mathop { \lambda _ { k } ^ { \prime } } ( t ) - \lambda _ { h _ { 0 } ^ { i } } ^ { \prime } ( t ) \right) } \end{array}$
10: $s \gets \operatorname* { m a x } ( t - 1 / T , 0 )$ ▷ one reverse step on the inference grid
11: for i with $m _ { i } = 1$ do
$\left( \alpha _ { h _ { 0 } ^ { i } } ( s ) - \alpha _ { h _ { 0 } ^ { i } } ( t ) \right)$
$\alpha _ { h _ { 0 } ^ { i } } ( t ) < 1 ,$
12: $\lambda _ { i }   \overline { { \mathrm { ~ \ ~ 1 ~ } - \alpha _ { h _ { 0 } ^ { i } } ( t ) } } $ ▷ per-step reveal probability
0, otherwise
13: $b _ { i } \sim$ Bernoulli(λ<sub>i</sub>) ▷ $b _ { i } = 1 \colon$ hierarchy revealed at s
14: end for
15: $\mathcal { R }  \{ i : m _ { i } = 1 , b _ { i } = 1 \}$
16: $h _ { s } ^ { i } \gets \bar { h } _ { 0 } ^ { i }$ for $i \in \mathcal { R } ; \ h _ { s } ^ { i } \gets \ h _ { t } ^ { i }$ otherwise ▷ Pass-2 context only: gold hierarchy at $\mathcal { R } , h _ { \mathrm { m a s k } }$
elsewhere
Pass 2 — token.
17: $f _ { i } ( t ) \gets \mathrm { s o f t m a x } \big ( x _ { \theta } ^ { i } ( \mathbf { x } _ { t } , \mathbf { h } _ { s } , t ) \big )$ for all i
18: $\begin{array} { r } { \mathcal { L } _ { \mathrm { L M } }  \sum _ { i } m _ { i } \lambda _ { i } ^ { \prime } ( - \log f _ { x _ { 0 } ^ { i } } ^ { i } ( t ) ) } \end{array}$ ▷ closed-form token CE over all masked positions
// Combine and update
19: $\mathcal { L } \gets ( \mathcal { L } _ { \mathrm { L M } } + \beta _ { \mathrm { h i e r } } ^ { \mathrm { ~ } } \mathcal { L } _ { \mathrm { h i e r } } + \beta _ { \mathrm { c o n s } } \mathcal { L } _ { \mathrm { c o n s } } ) / L$
20: $\theta \gets \theta - \eta \nabla _ { \theta } \mathcal { L }$
21: end while
```

## B EXPECTED MASK FRACTION UNDER HIERARCHICAL LOG-LINEAR SCHEDULE

We derive the expected fraction of masked tokens per training step under our hierarchical masking schedule, which is needed to compute the effective training-token budget and compare fairly against baseline diffusion language models (Sahoo et al., 2024).

Setup. Let K denote the number of hierarchy levels and let offse $\mathfrak { t } _ { k } \in [ 0 , 1 )$ denote the per-level time offset. The effective time at level k is

$$
t _ { \mathrm { e f f } , k } = \mathrm { c l i p } \left( \frac { t - \mathrm { o f f s e t } _ { k } } { 1 - \mathrm { o f f s e t } _ { k } } , 0 , 1 \right) ,\tag{77}
$$

and the log-linear schedule for level k is

$$
\alpha _ { k } ( t ) = 1 - ( 1 - \varepsilon ) t _ { \mathrm { e f f } , k } , \varepsilon = 1 0 ^ { - 3 } .\tag{78}
$$

The probability that a token at level k is masked at time t is therefore

$$
p _ { k } ( t ) \ : = \ : 1 - \alpha _ { k } ( t ) \ : = \ : ( 1 - \varepsilon ) t _ { \mathrm { e f f } , k } .\tag{79}
$$

Algorithm 2 HMDLM ancestral sampling (reverse process)   
Require: hierarchy head $h _ { \theta }$ (Pass 1), token head $x _ { \theta }$ (Pass 2); per-class keep schedules $\{ \alpha _ { k } ( \cdot ) \} _ { k = 1 } ^ { K } ;$   
number of steps $T ;$ optional token temperature τ.   
1: $x ^ { i }  M , \quad h ^ { i }  h _ { \mathrm { m a s k } } \quad \mathrm { f o r \ a l l } \ i = 1 , \dots , L$ ▷ t = 1: fully masked   
2: for $n = T , T - 1 , . . . , 1$ do   
3: $t \gets n / T , \qquad s \gets ( n - 1 ) / T$ ▷ reverse step $t  s ; \mathbf { x } .$ , h persist across iterations   
Pass 1 — hierarchy.   
4: $g ^ { i } \gets$ softmax $\big ( h _ { \theta } ^ { i } ( \mathbf { \dot { x } } , \mathbf { h } , t ) \big )$ for all i with $h ^ { i } = h _ { \mathrm { m a s k } }$   
5: ${ \mathcal { R } } \gets \emptyset$   
6: for i with $h ^ { i } = h _ { \mathrm { m a s k } }$ do   
7: $\lambda _ { k }  { \frac { \alpha _ { k } ( s ) - \alpha _ { k } ( t ) } { \prime \uparrow } }$ for $k = 1 , \ldots , K$ ▷ per-step reveal probability of class k   
$1 - \alpha _ { k } ( t )$   
8: sample $\begin{array} { r } { \hat { h } ^ { i } \sim \operatorname { C a t } \ ( \lambda _ { 1 } g _ { 1 } ^ { i } , \dots , \lambda _ { K } g _ { K } ^ { i } , \underbrace { 1 - \sum _ { k } \lambda _ { k } g _ { k } ^ { i } } ) ; h ^ { i }  \hat { h } ^ { i } } \end{array}$ ▷ commit the sampled   
stay masked   
class (or $h _ { \mathrm { m a s k } } )$ into h   
9: i $\because h ^ { i } \ne h _ { \mathrm { m a s k } }$ then   
10: $\mathcal { R }  \mathcal { R } \cup \{ i \}$ ▷ position revealed this step (shared across both streams)   
11: end if   
12: end for   
Pass 2 — token. Fill tokens at ${ \mathcal { R } } ,$ conditioned on the just-sampled hierarchy $\hat { h } ^ { i }$   
13: $f ^ { i } \gets$ softmax $( x _ { \theta } ^ { i } ( \mathbf { x } , \mathbf { h } , t ) )$ for all $i \in \mathcal { R }$ ▷ h carries $\hat { h } ^ { i } \mathrm { a t } i \in \mathcal { R }$ from line 9   
14: for $i \in \mathcal { R }$ do   
15: sample $x ^ { i } \sim \operatorname { C a t } ( f ^ { i } )$ ▷ reveal token; masked positions keep $x ^ { i } = M$   
16: end for   
17: freeze $( x ^ { i } , h ^ { i } )$ for all $i \in \mathcal { R } ;$ carry all of $( \mathbf { x } , \mathbf { h } )$ into step $n - 1$ ▷ committed positions are   
never revisited   
18: end for   
19: return $( \mathbf { x } , \mathbf { h } )$ ▷ clean sample $\left( \mathbf { x } _ { 0 } , \mathbf { h } _ { 0 } \right)$

Per-level expected mask probability. We compute $\mathbb { E } _ { t \sim \mathcal { U } [ 0 , 1 ] } [ p _ { k } ( t ) ] = ( 1 - \varepsilon ) \mathbb { E } [ t _ { \mathrm { e f f } , k } ]$ . Splitting the integral at the clamp boundary $t = \mathsf { o f f s e t } _ { k }$

$$
\mathbb { E } [ t _ { \mathrm { e f f } , k } ] = \int _ { 0 } ^ { \mathrm { o f f s e t } _ { k } } 0 d t + \int _ { \mathrm { o f f s e t } _ { k } } ^ { 1 } \frac { t - \mathrm { o f f s e t } _ { k } } { 1 - \mathrm { o f f s e t } _ { k } } d t .\tag{80}
$$

With the substitution $u = t - \mathsf { o f f s e t } _ { k }$ , the second integral becomes

$$
\int _ { 0 } ^ { 1 - \mathrm { o f f s e t } _ { k } } \frac { u } { 1 - \mathrm { o f f s e t } _ { k } } d u = \frac { ( 1 - \mathrm { o f f s e t } _ { k } ) ^ { 2 } } { 2 ( 1 - \mathrm { o f f s e t } _ { k } ) } = \frac { 1 - \mathrm { o f f s e t } _ { k } } { 2 } .\tag{81}
$$

Hence,

$$
\mathbb { E } _ { t \sim \mathcal { U } [ 0 , 1 ] } [ p _ { k } ( t ) ] \ = \ ( 1 - \varepsilon ) \cdot \frac { 1 - \mathrm { o f f s e t } _ { k } } { 2 } .\tag{82}
$$

Plugging in our configuration. With $K = 2$ , level offsets offset $_ \mathrm { ~ 1 ~ } = ~ 0 . 2 5$ (rare tokens) and $\mathrm { { o f f s e t } _ { 0 } = 0 . 0 }$ (frequent tokens), and $\varepsilon = 1 0 ^ { - 3 }$

$$
\begin{array} { r } { \mathbb { E } [ p _ { 1 } ( t ) ] = 0 . 9 9 9 \cdot \frac { 1 - 0 . 2 5 } { 2 } = 0 . 9 9 9 \cdot 0 . 3 7 5 = 0 . 3 7 4 6 2 5 , } \end{array}\tag{83}
$$

$$
\begin{array} { r } { \mathbb { E } [ p _ { 0 } ( t ) ] = 0 . 9 9 9 \cdot \frac { 1 - 0 . 0 } { 2 } = 0 . 9 9 9 \cdot 0 . 5 0 0 = 0 . 4 9 9 5 0 0 . } \end{array}\tag{84}
$$

Overall expected mask fraction. Let $f _ { k }$ denote the fraction of real token positions assigned to level k, with $f _ { 0 } + f _ { 1 } = 1$ . The overall expected mask fraction is

$$
\mathbb { E } [ \mathrm { m a s k } \mathrm { f r a c t i o n } ] = f _ { 0 } \cdot \mathbb { E } [ p _ { 0 } ( t ) ] + f _ { 1 } \cdot \mathbb { E } [ p _ { 1 } ( t ) ] .\tag{85}
$$

For LM1B with the bert-base-uncased tokenizer and context length $L = 1 2 8 .$ , we assign a token to level 1 if it occurs at most τ times within its own sequence (at $\tau = 1$ , exactly once).

Measured over the full training split (30,301,028 sequences, 922,347,589 real tokens), level 1 holds 737,634,266 real tokens at $\tau = 1$ and 914,638,961 at $\tau = 5 ,$ so

$$
\tau = 1 : \quad f _ { 1 } = 0 . 7 9 9 7 3 6 , \quad f _ { 0 } = 0 . 2 0 0 2 6 4 ,\tag{86}
$$

$$
\tau = 5 : \quad f _ { 1 } = 0 . 9 9 1 6 4 2 , \quad f _ { 0 } = 0 . 0 0 8 3 5 8 .\tag{87}
$$

Substituting into equation 85:

$$
\tau = 1 : { \mathbb { E } } [ \mathrm { m a s k } \operatorname { f r a c t i o n } ] = 0 . 2 0 0 2 6 4 ( 0 . 4 9 9 5 0 0 ) + 0 . 7 9 9 7 3 6 ( 0 . 3 7 4 6 2 5 )
$$

$$
= 0 . 1 0 0 0 3 2 + 0 . 2 9 9 6 0 1 \approx 0 . 3 9 9 6 3 3 ,\tag{88}
$$

$$
{ \begin{array} { r l } { \tau = 5 : } & { \mathbb { E } [ { \mathrm { m a s k ~ f r a c t i o n } } ] = 0 . 0 0 8 3 5 8 ( 0 . 4 9 9 5 0 0 ) + 0 . 9 9 1 6 4 2 ( 0 . 3 7 4 6 2 5 ) } \\ & { = 0 . 0 0 4 1 7 5 + 0 . 3 7 1 4 9 4 \approx 0 . 3 7 5 6 6 9 . } \end{array} }\tag{89}
$$

Effective training-token budget. The budget is denominated in supervised positions, i.e. those that are both masked and real. Padding is excluded from the loss.<sup>3</sup> Given training steps S, batch size $B ,$ and mean real length ${ \bar { R } } ,$ the expected number of supervised tokens is

$$
\mathbb { E } [ \mathrm { s u p e r v i s e d t o k e n s } ] \ = \ S \cdot B \cdot \bar { R } \cdot \mathbb { E } [ \mathrm { m a s k f r a c t i o n } ] .\tag{90}
$$

With $B = 5 1 2$ and $\bar { R } = 3 0 . 4 4$ , each step presents $B \cdot \bar { R } = 1 5 { , } 5 8 5$ real positions and therefore supervises $1 5 , 5 8 5 \cdot 0 . 3 9 9 6 3 3 \approx 6 , 2 2 8 . 3$ tokens at $\tau = 1$ and $1 5 , 5 8 5 \cdot 0 . 3 7 5 6 6 9 \approx 5 , 8 5 4 . 8 \mathrm { a t } \tau = 5 ,$ against MDLM’s $1 5 , 5 8 5 \cdot 0 . 4 9 9 5 0 0 \approx 7 , 7 8 4 . 7$ . To reach a target budget N, the required number of steps is

$$
S ^ { \star } = \frac { N } { B \cdot \bar { R } \cdot \mathbb { E } [ \mathrm { m a s k f r a c t i o n } ] } .\tag{91}
$$

We instantiate this at two budgets. MDLM’s 33B tokens on LM1B counts masked positions over the padded context $( 1 0 ^ { 6 } \cdot 6 5 , 5 3 6 \cdot 0 . 4 9 9 5 = 3 2 . 7 4 \mathrm { { B } ) ; }$ of these, a fraction $\bar { R } / L = 0 . 2 3 \bar { 7 } 8$ are real, so the same run supervises $N _ { 3 3 } = 7 . 7 8 4 7 \mathrm { B }$ tokens. Matching it requires

$$
S _ { 3 3 } ^ { \star } = \frac { 7 . 7 8 4 7 \times 1 0 ^ { 9 } } { 1 5 , 5 8 5 \cdot 0 . 3 9 9 6 3 3 } \approx 1 . 2 4 9 9 \times 1 0 ^ { 6 } ~ \mathrm { s t e p s }\tag{92}
$$

$$
S _ { 3 3 } ^ { \star } = \frac { 7 . 7 8 4 7 \times 1 0 ^ { 9 } } { 1 5 , 5 8 5 \cdot 0 . 3 7 5 6 6 9 } \approx 1 . 3 2 9 6 \times 1 0 ^ { 6 } \mathrm { s t e p s }\tag{93}
$$

For the extended budget, twice as large at $N _ { 6 5 } = 1 5 . 5 6 9 4 \mathrm { B }$ supervised tokens (65.47B masked positions),

$$
S _ { 6 5 } ^ { \star } = \frac { 1 5 . 5 6 9 4 \times 1 0 ^ { 9 } } { 1 5 , 5 8 5 \cdot 0 . 3 9 9 6 3 3 } \approx 2 . 4 9 9 8 \times 1 0 ^ { 6 } \mathrm { s t e p s }\tag{94}
$$

$$
S _ { 6 5 } ^ { \star } = \frac { 1 5 . 5 6 9 4 \times 1 0 ^ { 9 } } { 1 5 , 5 8 5 \cdot 0 . 3 7 5 6 6 9 } \approx 2 . 6 5 9 3 \times 1 0 ^ { 6 } \mathrm { s t e p s }\tag{95}
$$

## C EXPECTED MASKING FRACTION FOR WIKIHOW

Under the hierarchical power schedule, a response token at level ℓ with exponent $w _ { \ell }$ is masked at diffusion time $t \sim \mathcal { U } ( 0 , 1 )$ with probability $\left( 1 - \epsilon \right) t ^ { w _ { \ell } }$ . Its expected masking rate is therefore

$$
\bar { m } _ { \ell } = ( 1 - \epsilon ) \int _ { 0 } ^ { 1 } t ^ { w _ { \ell } } d t = \frac { 1 - \epsilon } { w _ { \ell } + 1 } .\tag{96}
$$

Let $g _ { \ell }$ be the fraction of tokens assigned to level $\ell ,$ pooled over the training set and counted over diffused positions only (the prompt and padding are excluded). The expected masking rate of HMDLM is then

$$
\bar { m } _ { \mathrm { H M D L M } } = ( 1 - \epsilon ) \sum _ { \ell } \frac { g _ { \ell } } { w _ { \ell } + 1 } .\tag{97}
$$

Table 7: Step-matched training budgets on WikiHow at context lengths 512 and 1024. $g _ { 1 }$ is the fraction of tokens at level 1. Under the summary rule, level 1 is the article summary. Under the count rule (τ=1), level 1 is any token occurring at most once in its own sample.
<table><tr><td>Model</td><td> $g _ { 1 }$ </td><td> $\rho = 1 - g _ { 1 } / 3$ </td><td>Steps</td><td>Extra steps (epochs)</td></tr><tr><td colspan="5"> $L = 5 1 2$ </td></tr><tr><td>MDLM</td><td></td><td>1.000</td><td>28,650</td><td></td></tr><tr><td>HMDLM (summary)</td><td>22.34%</td><td>0.926</td><td>30,955</td><td> $+ 2 , 3 0 5 \ : ( + 2 . 4 0 )$ </td></tr><tr><td>HMDLM (τ=1)</td><td>30.82%</td><td>0.897</td><td>31,930</td><td> $+ 3 , 2 8 0 ( + 3 . 4 1 )$ </td></tr><tr><td colspan="5"> $L = 1 0 2 4$ </td></tr><tr><td>MDLM</td><td></td><td>1.000</td><td>41,137</td><td></td></tr><tr><td>HMDLM (summary)</td><td>16.41%</td><td>0.945</td><td>43,517</td><td> $+ 2 , 3 8 0 \left( + 1 . 7 2 \right)$ </td></tr><tr><td>HMDLM (τ=1)</td><td>27.12%</td><td>0.910</td><td>45,225</td><td> $+ 4 , 0 8 8 \left( + 2 . 9 6 \right)$ </td></tr></table>

MDLM corresponds to a single level with $w = 1$ , giving $\bar { m } _ { \mathrm { M D L M } } = ( 1 - \epsilon ) / 2$ . With two levels and exponents $w _ { 0 } = 1 , w _ { 1 } = 2$

$$
\bar { m } _ { \mathrm { H M D L M } } = ( 1 - \epsilon ) \left[ \frac { 1 - g _ { 1 } } { 2 } + \frac { g _ { 1 } } { 3 } \right] = ( 1 - \epsilon ) \left( \frac { 1 } { 2 } - \frac { g _ { 1 } } { 6 } \right) ,\tag{98}
$$

so the ratio to MDLM, in which ϵ cancels, is

$$
\rho \ = \ \frac { \bar { m } _ { \mathrm { H M D L M } } } { \bar { m } _ { \mathrm { M D L M } } } \ = \ 1 - \frac { g _ { 1 } } { 3 } .\tag{99}
$$

To match the expected number of masked (loss-bearing) tokens seen by MDLM in $S _ { \mathrm { M D L M } }$ steps at the same batch size, HMDLM is trained for

$$
S _ { \mathrm { H M D L M } } ~ = ~ { \frac { S _ { \mathrm { M D L M } } } { \rho } } ~ = ~ { \frac { 3 ~ S _ { \mathrm { M D L M } } } { 3 - g _ { 1 } } } , ~ \Delta S ~ = ~ S _ { \mathrm { H M D L M } } - S _ { \mathrm { M D L M } } ~ = ~ S _ { \mathrm { M D L M } } ~ { \frac { g _ { 1 } } { 3 - g _ { 1 } } }\tag{100}
$$

steps.

On WikiHow, MDLM is trained for ≈ 29.8 epochs at each context length, with a global batch of $1 2 8 \colon S _ { \mathrm { M D L M } } = 2 8 { , } 6 5 0$ steps at $L = 5 1 2 ( 1 2 3 , 0 4 6$ training examples, 961.3 steps per epoch) and $S _ { \mathrm { M D L M } } = 4 1 , 1 3 7$ steps at $\dot { L } = 1 0 2 4$ (176,696 training examples, 1,380.4 steps per epoch). Table 7 gives the resulting budgets.

## D PER-LEVEL SCHEDULES AND THE INDUCED REVEAL ORDER

Levels are indexed so that a higher level is masked more slowly. Formally, for any two levels $k < k ^ { \prime }$ we require

$$
\alpha _ { k } ( t ) \ \leq \ \alpha _ { k ^ { \prime } } ( t ) \quad { \mathrm { f o r ~ a l l ~ } } t ,\tag{101}
$$

with strict inequality at every step at which level $k ^ { \prime }$ is active, i.e. $\alpha _ { k ^ { \prime } } ( t ) < 1$ . Since $\alpha _ { k } ( 0 ) = 1$ and $\alpha _ { k } ( T ) = 0$ for every k, the inequality is necessarily an equality at the two endpoints. Fewer level-k positions are therefore masked at any interior point of the forward process.

The reverse dynamics are governed by the per-step reveal probability $\lambda _ { k } ( t )$ , and equation 101 does not by itself imply an ordering of the $\lambda _ { k }$ . We therefore state the property we use directly: for $k < k ^ { \prime }$

$$
\lambda _ { k } ( t ) \ \leq \ \lambda _ { k ^ { \prime } } ( t ) \qquad \mathrm { w h e n e v e r } \ \alpha _ { k ^ { \prime } } ( t ) < 1 .\tag{102}
$$

Outside that interval, $\alpha _ { k ^ { \prime } } ( t ) = 1$ , no $\mathrm { l e v e l } { - } k ^ { \prime }$ position is masked and $\lambda _ { k ^ { \prime } } ( t ) = 0 ;$ the ordering is vacuous there, since there is nothing left at level $\overline { { k ^ { \prime } } }$ to reveal. Both schedules used in our experiments satisfy equation 102, as the continuous-time hazards $\lambda _ { k } ^ { \prime } ( t ) = - \alpha _ { k } ^ { \prime } ( t ) / ( 1 - \alpha _ { k } ( t ) )$ show:

• Log-linear with per-level offset equation 78, used on LM1B: $\lambda _ { k } ^ { \prime } ( t ) = 1 / ( t - \delta _ { k } )$ for $t > \delta _ { k }$ and 0 otherwise. For $\delta _ { k } < \delta _ { k ^ { \prime } }$ and $t > \delta _ { k ^ { \prime } }$ we have $t - \delta _ { k } > t - \delta _ { k ^ { \prime } } > 0$ , hence $\lambda _ { k } ^ { \prime } ( t ) < \lambda _ { k ^ { \prime } } ^ { \prime } ( t )$

• Power schedule $\alpha _ { k } ( t ) = 1 - ( 1 - \varepsilon ) t ^ { \omega _ { k } }$ , used on WikiHow: $\lambda _ { k } ^ { \prime } ( t ) = \omega _ { k } / t$ , so the ordering holds exactly when $\omega _ { k } < \omega _ { k ^ { \prime } }$ . We use $\omega = ( 1 . 0 , 2 . 0 )$ , and $\omega _ { k } = 1$ recovers the log-linear schedule of MDLM.

The discrete $\lambda _ { k } ( t )$ inherit the ordering on the step grids used in our experiments $( T = 2 5 6$ on both LM1B, and WikiHow): the only steps at which $\lambda _ { 0 } ( t ) > \lambda _ { 1 } ( t )$ are the 64 LM1B steps with $t \leq \delta _ { 1 }$ where level 1 is inactive and $\lambda _ { 1 } ( t ) = 0$

Consequently, while level $k ^ { \prime }$ is still being revealed, a position that Stage I predicts to be at level $k ^ { \prime }$ receives a larger reveal mass $\lambda _ { k ^ { \prime } } ( t ) g _ { k ^ { \prime } } ^ { i } \bar { ( t ) }$ in equation 8 and is revealed earlier, which is the soft generation order HMDLM is designed to induce.

## E PERPLEXITY: CONTINUOUS-TIME ELBO WEIGHT

We evaluate test-set perplexity under the continuous-time limit of the HMDLM ELBO. For a single hierarchy level k, the masking schedule is linear with offset $\delta _ { k }$

$$
\alpha _ { k } ( t ) = \left\{ \begin{array} { l l } { { 1 } } & { { t \leq \delta _ { k } , } } \\ { { 1 - ( 1 - \varepsilon ) \displaystyle \frac { t - \delta _ { k } } { 1 - \delta _ { k } } } } & { { \delta _ { k } < t \leq 1 , } } \end{array} \right. \delta _ { 0 } = 0 , \delta _ { 1 } = 0 . 2 5 ,\tag{103}
$$

where $\alpha _ { k } ( t )$ is the survival (clean) probability of a level-k token at time $t .$

Per-level continuous-time weight. The standard absorbing-state ELBO weight is

$$
\lambda _ { k } ^ { \prime } ( t ) ~ = ~ - \frac { \alpha _ { k } ^ { \prime } ( t ) } { 1 - \alpha _ { k } ( t ) } ,\tag{104}
$$

For the schedule in equation 103, on the active interval $t > \delta _ { k }$

$$
\alpha _ { k } ^ { \prime } ( t ) = - \frac { 1 - \varepsilon } { 1 - \delta _ { k } } , 1 - \alpha _ { k } ( t ) = ( 1 - \varepsilon ) \frac { t - \delta _ { k } } { 1 - \delta _ { k } } .\tag{105}
$$

Substituting into equation 104,

$$
w _ { k } ( t ) = \frac { ( 1 - \varepsilon ) / ( 1 - \delta _ { k } ) } { ( 1 - \varepsilon ) ( t - \delta _ { k } ) / ( 1 - \delta _ { k } ) } = \frac { 1 } { t - \delta _ { k } } .\tag{106}
$$

For $t \leq \delta _ { k }$ the level is fully unmasked $( 1 - \alpha _ { k } ( t ) = 0 )$ and contributes no loss; we therefore extend $\lambda _ { k } ^ { \prime }$ by zero in that regime:

$$
\lambda _ { k } ^ { \prime } ( t ) : = { \left\{ \begin{array} { l l } { { \frac { 1 } { t - \delta _ { k } } } } & { t > \delta _ { k } , } \\ { 0 } & { t \leq \delta _ { k } . } \end{array} \right. }\tag{107}
$$

$\delta _ { 0 } = 0$ recovers the MDLM weight $1 / t$ used by Sahoo et al. (2024), so equation 107 is a strict generalisation.

Posterior-weighted per-token weight. Since at evaluation, the gold hierarchy is not available, at position i, we replace the gold-hierarchy weight $\lambda _ { h _ { 0 } ^ { i } } ^ { \prime } ( t )$ with its expectation $r _ { i } ( t )$ under the model’s posterior over hierarchy levels at time t:

$$
r _ { i } ( t ) = \sum _ { k = 0 } ^ { K - 1 } [ h _ { \theta } ^ { i } ( x _ { t } , h _ { t } , t ) ] _ { h } \lambda _ { k } ^ { \prime } ( t ) ,\tag{108}
$$

where $h _ { \theta } ^ { i }$ is the Pass-1 hierarchy distribution at position $i .$

Token-level perplexity. Let N denote the total number of tokens (non-padding) in the test set, and let $\mathbf { x } _ { t } .$ , h<sub>t</sub> be sampled by the joint forward process at time $t \sim \mathcal { U } ( \varepsilon , 1 )$ , with $\varepsilon = 1 0 ^ { - 3 }$ . The per-token negative log-likelihood is

$$
{ \mathrm { N L L } } _ { \mathrm { t o k } } = { \frac { 1 } { N } } \mathbb { E } _ { t , \mathbf { x } _ { t } } { \Bigg [ } \sum _ { i : x _ { t } ^ { i } = M } r _ { i } ( t ) \left( - \log p _ { \theta } ( x _ { 0 } ^ { i } \mid \mathbf { x } _ { t } ) \right) { \Bigg ] } ,\tag{109}
$$

and the reported test-set perplexity is

$$
\mathrm { { P P L } = \exp ( N L L _ { \mathrm { { t o k } } } ) . }\tag{110}
$$

Numerical handling of $t \to \delta _ { k } ^ { + } . \quad \mathrm { A t } \ t = \delta _ { k }$ is bounded in code by a denominator floor:

$$
\lambda _ { k } ( t ) \approx \frac { 1 } { \operatorname* { m a x } ( t - \delta _ { k } , \eta ) } , \qquad \eta = 1 0 ^ { - 8 } ,
$$

which leaves the weight unchanged everywhere except in a vanishingly small neighbourhood of $\delta _ { k }$ and prevents single-sample gradient spikes from the Monte-Carlo estimator.

## F ARCHITECTURE DETAILS AND QUALITATIVE ANALYSIS ON SUDOKU

## F.1 ARCHITECTURE DETAILS

For the 6M MGDM and HMDLM models, we use a compact 3-layer GPT-2 Transformer ( 5.5M parameters) with hidden size 384, 12 attention heads of dimension 32, a 1536-dimensional GELU feed-forward layer, learned absolute position embeddings, and a 31-token character-level vocabulary whose output head shares weights with the input embedding and bidirectional attention. The hierarchical variant (HMDLM) adds only a level embedding and a level-prediction head, about 5K extra parameters.

For the 25M MDLM and HMDLM models, we use a 12-layer diffusion transformer (DiT, 25.0M parameters) with hidden size 384, 6 attention heads of dimension 64, a 1536-dimensional GELU feed-forward layer, rotary position embeddings, and bidirectional attention over a fixed 81-token sequence (one token per cell). Both the models are trained from scratch. The hierarchical variant (HMDLM, K=6 levels) adds a 7x384 level embedding (6 levels plus a [MASK] level), which is summed with the token embedding, and a level-prediction head that has the same form as the token output head. Together they add about 104K parameters (25.10M versus 24.99M).

## F.2 QUALITATIVE ANALYSIS

For the sudoku 25M hmdlm and mdlm models - we identified 287 puzzles where MDLM failed to solve across all three seeds but HMDLM solved correctly across all three. We then performed 8 additional inference runs per model on these puzzles. We pick some of these examples and trace their unmasking order to show how HMDLM helped in getting the right answer:

Example 1: 50607080040000001600000000000200000307005040000090002096   
0000500030002007000100000   
Example 2: 008100026000807000900000000010008900030000010000450000000309800   
200000400500020300

![](images/3e26cc46d03527c65dfef94628cfbf976ff130cab9c7e26aefcd9ff0af005c0a.jpg)  
Figure 2: Example 1, Start.

![](images/563796b0e67819c0c59642147308a037547d63ac7a0b658e35321aa9f97f1a4d.jpg)  
Round 2. MDLM predicts the R9C6 cell and gets it wrong. There are multiple possible values for the cell 3,4,5,6,7,8,9. HMDLM on the other hand, pushes the unmasking of this cell to later rounds.  
Figure 3: Example 1, round 2.

![](images/61662263a48d8ae5fbd58701551b07ed4c5421b6e2a95e1a4ec67b6ed359da44.jpg)  
Round 25. HMDLM defers R9C6 and fills other cells first.  
Figure 4: Example 1, round 25.

![](images/8611be314228ca6491451dfb29af0e57cad1af2b19788d4e030f8080b401491c.jpg)  
Figure 5: Example 1, round 26.

![](images/8cf7a3e0ad09a74f659c22a780a1a78b5224e270a58ad3ec2142400788f12a90.jpg)  
Figure 6: Example 2, Start.  
3 gold level of a blank (blue = forced, grey = search)

![](images/6bbeadaa0655e6594c960783b5c19c5760ca222a1b8964c08c093f6c0f98c70a.jpg)  
Round 1. MDLM predicts the R7C1 cell and gets it wrong. There are multiple possible values for the cell 1,4,6,7. HMDLM on the other hand, pushes the unmasking of this cell to later rounds.

Figure 7: Example 2, round 1.  
![](images/64f57e18505124d851e570c2e44f1ed8e1a668fa652a6f215ec261b496b05bf6.jpg)  
Figure 8: Example 2, round 29.

![](images/e65706338fd7819f5d9f5bb8cfeefa50b1f9e6e2710e8542b7af3afb202a6ce7.jpg)  
Figure 9: Example 2, round 36.

Both the samples show the same pattern, MDLM tends to unmask ambiguous cells in the beginning and gets the prediction wrong. On the other hand, HMDLM tends to defer the unmasking of such cells to later rounds until neighbouring cells get filled first. Across 287 puzzles (where HMDLM passed and MDLM failed) with 8 samples per model, we find 131 cases where MDLM’s first error is an early commit to a search cell and HMDLM fills that same cell only once a single candidate remains.

## F.3 CELL ACCURACY PER HIERARCHY LEVEL

Table 8: Cell accuracy (%) by hierarchy level.
<table><tr><td>Level</td><td>Cells</td><td>MDLM</td><td>HMDLM</td></tr><tr><td>5</td><td>givens + round 0</td><td>99.76</td><td>100.00</td></tr><tr><td>4</td><td>round 1</td><td>99.59</td><td>99.99</td></tr><tr><td>3</td><td>round 2</td><td>99.29</td><td>99.98</td></tr><tr><td>2</td><td>round 3</td><td>98.73</td><td>99.92</td></tr><tr><td>1</td><td>rounds 4–5</td><td>96.77</td><td>99.45</td></tr><tr><td>0</td><td>round 6+ / never forced</td><td>78.78</td><td>86.57</td></tr></table>

Each cell is assigned a level by the constraint-propagation round in which it becomes forced: level 5 cells are givens or forced immediately (round 0), while level 0 cells are forced only after 6+ rounds or never, and so require search. We see that HMDLM improves the accuracy for the most difficult cells (label 0) by up to 8% than MDLM.

## G QUALITATIVE ANALYSIS ON WIKIHOW

In this section we look at the generative quality of samples from HMDLM, MDLM, and ADLM models.

## G.1 PROMPT: HOW TO REPLACE THE REAR SHOCK ON AN 1150RT

MDLM Find the parts that you need to build in order not to replace the rear shock. diagnose the problem. assemble your latest model. that will take you to points where you want to attach ground cable. shop older models with an antenna kit — the replacement might come with 80mm screw for a zoom lens. when you find your pictures of the 110054ares, youll see an obvious slot on the shelf. disconnect the screws. you may disconnect them above the antenna mast if you connect the kit on the antenna that acts as the zoom lens. tighten up the screws. the wiring actually begins. enjoy wiring it. also make sure that the carbon fiber shaft meets the rest of the parts. that then download the latest version of the wirelesss and chargers. so by now, you will complete your replacement order. youll need to separate the stand like the new shock like your christmas boots from the cables that you use to replace your rear and front shocks. on model ebola suggest that the connection and the handle on disengage. then you should isolate the wires from a stainless steel. set up your stepper mount separately. it identifies the wattage and this one turns on the wire. you will need wiring

ADLM Remove the clutch and cut the 17 braided cable from valvages up to about 12th inch into the driver shock. insert the valve valve in a branch of the comparison valve about 1 mm in the bottom lip. insert in the valve valve with a minimal cussist pressure about 5, 000remp. push your gently next to a pump station, four thumbs up and the lever about two inches down. insert one of the bearings into the hard valve, which usually fits into the internal part. along with pulling on a slow wheel, the vr should be left into the quick valve. insert a pin, but novel adjustment is an option in the bottom lip. make sure it stills your heel while pushing the valve, as found in normal mode. insert in two screws into the vr and tighten the jl band. begin twisting the traction steel - socket into the rac socket, lid of the driver or shock. push through the valve flex and, using a cork, or cauterized rubber band, remove the slide from the inside., the traction valve cuts the force of the cussist when the valve is closed.

HMDLM Check the components you need to help you replace the rear shock. unscrew the rear bolt to the rear. put the shocks on the ground. remove the shock bolt you have. now you can replace the front shocks. remove the rear bolt with the nut that the rear shock bolt is attached to. with the wheel on the back on the open side, pull. you should pull this with the nut that the front bolt is near it. you should do this in the direction on the rear bolt shown with the bolt. when you do this, you should notice how the pressure is applied. when you do this, you should find that the rear bolt, shock bolt are in place. after this, find that the rear bolt is in place with the front shock bolt. do this a couple of times until you can the find the replacement parts.

Conclusion: Among the three samples, HMDLM produces the most faithful response: it stays on the task of replacing a rear shock throughout and refers to relevant components such as shocks, bolts and nuts, although its step order is repetitive and somewhat confused. ADLM maintains a mechanical register but is largely incoherent, with invented terms (e.g., “valvages”, “cussist”) and physically implausible instructions. MDLM drifts off topic almost immediately, moving to antenna kits, zoom lenses and wiring, and never addresses the shock replacement. Overall, HMDLM shows the strongest prompt adherence and topical consistency, ADLM is second, and MDLM is the weakest.

## G.2 PROMPT: HOW TO BUY A CUSTOM FIT CAR COVER

MDLM Find a vanity cover if there are none on your list. celebrate who is going to drive. buy the car cover you need. get it framed. label the cover. make sure that it isnt just a side cover but if you still have one, get a frame and create some polo paint. coat your cover with black to make it, drywall post paper or more ply window covers for material covers. get a cheap cover. these will be cheaper than your usual store window cover. since youre planning to sell the car youre going to build you either choose a cover with floor windows. if you do not intend to build it as is, be sure to buy an inexpensive cover. car cover cant go over the years, unless you have a plan to make your best youll always need to make up new windows going forward. tips are useful, so select a cover and not a framing cover as will help sell the idea that it works for that purpose. accessories on the cover cover will free up cover wastage, if you cant find the little scratches in the car you need to wipe them out before youve made your final cover.

ADLM , first, make sure the you have the nail links. go down the edge. look at the details on top. put detail here and think some thoughts on if your arms fit correctly. look toward cushier areas where you would like you get a good fit like the pieces under the jeep under the convenience button. frames is fine, but less than works. where you find where you think you are more fit, good. look over the shoulders. the rest of the seats unless you get at the side of the neck try some chair. find the front cover and try the front wheel. put a few inserts near the peephole. add some glosses or any more products in the top section. look over the interior of the cover, slip or pop the back. are you wearing a high top, or just would like to see the final mix out dress options couch pants did just get too tight for the shoulders are you fit do you want the skin soft or tight attention to face, hair, etc. are you sure of movement a little caution.

HMDLM Look at the records of most retailers. locate the cover that fits the vehicle youre looking to buy. look for car covers online. locate the stores you can find your custom fit car covers at. look for different types of car covers from depending on what youre buying or looking for specific vehicles. visit a local shop and call and ask them if they will get what you are looking for. pay for or buy custom covers. some stores may keep car bins or keep a box of cars that will fit the type of cover that you can get from the shop. if you want medium - sized vehicles to be as large as you want, you can be able to buy it . however, it is usually not much more than what you want. look for a car type store or a retailer that tailor a car cover to you for a cheaper price.

Conclusion HMDLM produces the most relevant response: it stays focused on buying a custom-fit car cover and follows a sensible sequence of steps (checking retailers, searching online, contacting local shops and comparing prices), although a few sentences are garbled (e.g., keep a box of cars). MDLM repeatedly mentions car covers but quickly drifts into unrelated and contradictory instructions such as framing the cover, applying "polo paint" and adding windows. ADLM departs furthest from the prompt, shifting from car covers to garment fitting (shoulders, pants, dress options) with largely incoherent instructions. Overall, HMDLM shows the strongest prompt adherence and coherence, followed by MDLM, with ADLM the weakest.

## H PROMPT FOR LLM-AS-JUDGE

## H.1 LM1B PROMPT

For evaluating the LM1B generated samples under LLM-as-judge, we pass the following prompt:

"""You are an expert evaluator judging two pieces of machine-generated English text. Your task is to determine which text is BETTER in overall quality. You must choose exactly ONE of:

## A - Sample A is better

B - Sample B is better

tie - The two samples are approximately equal in quality

There is NO prompt or instruction the text is responding to; these are standalone generated passages. Evaluate each text holistically, considering:

1. Fluency: Is the text grammatical and natural English?

2. Coherence: Do the sentences follow each other in a logical, connected way?

3. Consistency: Does the text stay on a consistent topic without contradictions or abrupt, incoherent drift?

4. Overall readability: Does the passage read like plausible, well-formed human-written text? SCORING RULES:

1. Judge ONLY the two texts.

2. Do NOT assume that either sample is better because it is longer.

3. Do NOT penalize a response merely for being brief or terse.

4. Judge each text independently on its own quality.

5. Do NOT use any information about the source/model that generated the text.

6. The labels A and B are randomly assigned and have no meaning.

7. Choose tie only when the two texts are genuinely comparable in overall quality.

Return ONLY valid JSON: "winner": "A" | "B" | "tie" """

## H.2 WIKIHOW PROMPT

On similar lines, for running the LLM-as-judge on WikiHow dataset we use the following prompt:

"""You are an expert evaluator judging four responses to the same how-to prompt.

Your task is to determine which ONE response is the BEST for the given prompt.

You must choose exactly ONE of: A, B, C, D.

Evaluate the responses holistically, considering:

1. Adherence: Does the response address the specific topic in the prompt, or does it drift?

2. Coherence: Do the steps follow a logical, connected order?

3. Fluency: Is the response grammatical and natural English?

4. Overall usefulness: Would the response plausibly help someone accomplish what the prompt asks?

SCORING RULES:

1. Judge ONLY the prompt and the four responses. 2. Do NOT assume that a sample is better because it is longer. 3. Do NOT penalize a response merely for being brief or terse if it adequately answers the prompt. 4. Judge each response independently against the prompt. 5. Do NOT compare based on writing style alone; prioritize usefulness and correctness for the prompt. 6. Do NOT use any information about the source/model that generated the responses. 7. The labels A, B, C and D are randomly assigned and have no meaning. 8. If several responses are close, still pick the single best one.

Return ONLY valid JSON:

"winner": "A" | "B" | "C" | "D" """

## I SOCIETAL IMPACT OF THE MODEL

The proposed HMDLM advances masked diffusion language modeling by introducing hierarchical decoding, which improves reasoning and text generation quality. The positive societal impacts include more efficient and accurate language models that can benefit downstream NLP applications such as question answering, text generation, and reasoning tasks. On the negative side, as with all pre-trained language models, HMDLM could potentially be misused to generate misleading or harmful text.

## J COMPUTATIONAL RESOURCES

For language modelling, all experiments were conducted on NVIDIA A100 80GB GPUs. The HMDLM model was trained for approximately four days on 8 GPUs with a batch size of 512, completing 2.5M ( 65B tokens) training steps on LM1B dataset.

For WikiHow, All experiments were conducted on NVIDIA A100 80GB GPUs. The HMDLM model was trained for approximately 8 hours on 4 GPUs.

For Sudoku, again we used the same machine and trained on 4GPUs for 4 hours.