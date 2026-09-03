# Do Large Language Models Capture the Diversity in their Training Data?

Youqi Wu<sup>∗</sup>, Farzan Farnia<sup>†</sup>

## Abstract

Large language models are trained to model conditional distributions over text, yet it remains inadequately understood whether they capture the full diversity of plausible outputs present in their training data. We study this question through an information-theoretic lens by comparing the conditional entropy of model-generated outputs with that of the corresponding training data. Given paired inputoutput samples, we use conditional entropy and its matrix-based analogue based on von Neumann entropy to measure output variability beyond what is explained by the conditioning input, without requiring multiple reference outputs for the same prompt. Across LLM families with publicly available training data, including OLMo, Pythia, and GPT-Neo, we consistently find that model-generated outputs exhibit lower conditional entropy than their training data, across diferent model scales, sequence lengths, and decoding strategies. We observe a similar conditional diversity gap beyond language modeling, including class-conditioned ImageNet generators and text-conditioned models trained on MS-COCO. To address this gap, we propose a post-hoc correction mechanism that generates multiple outputs for each input and reweights them through a matrix-entropy projection, increasing conditional diversity while remaining close to the original model distribution. We prove the concavity of the matrix-based conditional entropy functional, which makes the resulting entropy-constrained projection a convex optimization problem, and develop a scalable mirror-descent algorithm for its implementation. Our results reveal a systematic conditional diversity gap between modern generative models and their training data, and provide an information-theoretic framework for measuring and mitigating this gap.

## 1 Introduction

Large language models are increasingly used as conditional distribution models: given a prompt, instruction, context, or document, they are expected to produce not only a single plausible completion, but samples from a rich set of valid outputs. The same viewpoint underlies many other conditional generative systems, including text-to-image models, image captioning systems, and class-conditioned image generators [1, 2, 3, 4]. In these settings, the conditioning variable rarely determines a unique output. A prompt may admit multiple correct phrasings, reasoning paths, levels of detail, or stylistic choices; an image may admit several faithful captions; and a class label may correspond to a broad range of visual instances. A central question is therefore whether modern conditional generative models capture the full range of outputs supported by the data distribution.

Most existing evaluations do not directly answer this question. Standard language-modeling metrics such as likelihood or perplexity measure average predictive fit, while many generation benchmarks focus on accuracy, alignment, and semantic consistency. For image and multimodal generation, metrics such as CLIPScore [5], MAUVE [6], and conditional Fréchet-type distances [7] provide valuable tools for comparing generated and reference samples. However, a model could still perform well under such metrics while still concentrating its probability mass on a narrower subset of valid outputs. This is important for LLMs and prompt-guided generation, where the goal is not merely to produce one acceptable answer, but to represent the conditional variability present in natural text data.

A direct measurement of conditional output variability is statistically challenging. In many datasets, we observe paired samples (X, Y), where X is the conditioned variable and Y is the corresponding output, but we do not observe many independent human outputs for the exact same condition. Thus, estimating conditional entropy separately for each fixed prompt is usually infeasible. To address this, we study conditional entropy through paired input-output samples. At the classical level, Shannon conditional entropy measures the uncertainty in Y that remains after observing X. At the kernel level, we use a matrix-based analog buil from a product kernel on pairs (X, Y) and subtract the entropy of the conditioning variable. This gives a way to measure output variability beyond the variability already explained by the inputs, without requiring repeated reference outputs for every condition.

Using this viewpoint, we find a consistent conditional output-range gap in modern generative models. Across LLM families with publicly accessible training data, including OLMo [8], Pythia [9], and GPT-Neo [10], we empirically evaluate that model-generated continuations exhibit lower conditional entropy than the corresponding training data. We observe the same phenomenon beyond text, including class-conditioned ImageNet generation and text-conditioned image models trained on MS-COCO. These results suggest that the issue is not specific to one architecture or training approach. Rather, conditional generative models appear to represent a narrowed version of the conditional distribution present in their data, as their conditional entropy underestimates that of the training data.

We formalize this phenomenon using conditional entropy and matrix-based conditional entropy, which extends the recent diversity bias analysis in [11] from unconditional generative models to conditional generative AI and LLM models. The resulting score compares the entropy of the joint input-output distribution with the entropy of the input marginal, thereby isolating the part of the variability attributable to the conditiona output law. This formulation will be natural for LLMs and general conditional generation systems, because it evaluates paired samples directly, rather than requiring multiple ground-truth responses for each prompt. It also connects the empirical measurement problem to matrix-based quantum entropy functionals.

On the technical side, we prove that the proposed matrix-based conditional entropy is a concave functional of the joint input-output distribution. This structural property has two consequences. It yields finite-sample monotonicity and underestimation properties analogous to the classical behavior of the classical Shannon entropy. More importantly, the concavity of the matrix-based quantum entropy implies that projecting a generated conditional distribution toward a higher conditional-entropy region can be formulated as a convex optimization problem.

Following the analysis, we propose a post-hoc sample-space reweighting method for LLMs and conditional generators. Given prompts $x _ { 1 } , \ldots , x _ { N }$ , we generate and consider multiple outputs $y _ { i , 1 } , \ldots , y _ { i , m }$ for each prompt x<sub>i</sub> and jointly optimize weights over these generated candidates so as to improve the overall conditional variability. The key constraint is that the total weight assigned to each prompt remains fixed, so the input marginal is preserved and only the conditional distribution over generated outputs is adjusted. The projection then finds the closest reweighted empirical distribution whose matrix-based conditional entropy exceeds a target level. In this way, the method does not retrain the model or synthesize new samples; instead, it asks whether the model has already generated valid but underweighted outputs, and redistributes probability mass toward a broader conditional empirical distribution.

For scalability, we develop an eficient mirror-descent implementation over a product of simplices, one simplex for the candidates generated from each condition. This extends exponentiated-gradient reweighting from the unconditional setting to conditional generation while exactly preserving the prompt marginal. Since the joint input-output feature space is a tensor-product space and can be prohibitively high-dimensional, we implement the method using sketched joint features. For finite-dimensional embeddings, we use random projections of Kronecker-product features; for shift-invariant kernels, we use direct joint random Fourier features for the product kernel. These approximations avoid explicitly forming the full tensor-product representation and reduce the cost of each optimization step to depend on the sketch dimension rather than the raw product dimension. The contributions of this work can be summarized as:

• We formulate the problem of whether LLMs and conditional generative models capture the full range of valid training outputs using conditional entropy and its matrix-based counterpart.

• We empirically demonstrate a consistent conditional output-range gap across LLMs with publicly available training data, class-conditioned ImageNet and MS-COCO generators.

• We derive concavity and finite-sample properties of the applied matrix-based conditional entropy functional, and show that entropy-constrained reweighting can be formulated as a convex program.

• We develop a scalable post-hoc reweighting algorithm based on product-simplex mirror descent and sketched joint features for adjusting conditional output distributions.

## 2 Preliminaries

We consider a conditional generation problem with conditioning variable $X \in { \mathcal { X } }$ and output variable $Y \in \mathcal { V }$ . The data distribution is denoted by $P _ { X Y }$ , with marginal $P _ { X }$ and conditional law $P _ { Y \mid X \cdot { \mathrm { ~ A ~ } } }$ conditional generative model specifies a conditional distribution $Q _ { Y \mid X }$ , which together with the same input marginal induces the joint model distribution $Q _ { X Y } = P _ { X } Q _ { Y \mid X }$

Let $k _ { X } : \mathcal { X } \times \mathcal { X } \to \mathbb { R }$ and $k _ { Y } : \mathcal { V } \times \mathcal { V }  \mathbb { R }$ be positive semidefinite kernels. Throughout the paper, we use normalized kernels which satisfy $k _ { X } ( x , x ) = 1$ and $k _ { Y } ( y , y ) = 1$ for every $x \in \mathcal { X }$ and $y \in \mathcal { V } . ^ { 1 }$ For paired samples, we use the product kernel

$$
k _ { X Y } \big ( [ x , y ] , [ x ^ { \prime } , y ^ { \prime } ] \big ) = k _ { X } ( x , x ^ { \prime } ) \cdot k _ { Y } ( y , y ^ { \prime } ) .
$$

Given samples $( x _ { 1 } , y _ { 1 } ) , \dotsc , ( x _ { n } , y _ { n } )$ , let $K _ { X }$ and $K _ { Y }$ be the corresponding Gram matrices. The product kernel has Gram matrix where ⊙ denotes the Hadamard product: $K _ { X Y } = K _ { X } \odot K _ { Y }$

For a positive semidefinite matrix A with unit-trace $\operatorname { T r } ( A ) = 1$ , its von Neumann entropy is defined as

$$
H _ { \mathrm { v N } } ( A ) = - \operatorname { T r } ( A \log A )
$$

Following the matrix-based entropy framework of [12, 13], we measure conditional output uncertainty by the diference between the joint input–output entropy and the input entropy:

$$
H _ { \mathrm { v N } } { \big ( } Y | X ; { \widehat { P } } _ { n } { \big ) } : = H _ { \mathrm { v N } } { \big ( } { \frac { 1 } { n } } K _ { X Y } { \big ) } - H _ { \mathrm { v N } } { \big ( } { \frac { 1 } { n } } K _ { X } { \big ) } .
$$

We note that the exponential of the above function is the same as the Conditional Vendi score, which is introduced and analyzed by Jalali et al. [13]. This quantity is the kernel analogue of conditional entropy: it measures the uncertainty contributed by the output after accounting for the entropy already present in the conditioning variable.

We also use the order-2 matrix entropy $H _ { 2 } ( A ) = - \log \operatorname { T r } ( A ^ { 2 } )$ , which gives the empirical conditional order-2 entropy as follows where $\| \cdot \| _ { F }$ denotes the Frobenius norm:

$$
H _ { 2 } \big ( Y \mid X ; \widehat { P } _ { n } \big ) : = H _ { 2 } \big ( \frac { 1 } { n } K _ { X Y } \big ) - H _ { 2 } \big ( \frac { 1 } { n } K _ { X } \big ) = \log \frac { \| K _ { X } \| _ { F } ^ { 2 } } { \| K _ { X } \odot K _ { Y } \| _ { F } ^ { 2 } } .
$$

This order-2 version is useful as a computationally simpler counterpart to von Neumann entropy, as formulated and framed as the Conditional RKE score in [13]. Population-level operator definitions and additional details are deferred to Appendix B.

## 3 Conditional output-range gaps in open language models

We first evaluate whether open language models match the conditional output range of their training data. We focus on OLMo, Pythia, and $\mathrm { G P T - N e o }$ , for which the training corpora are publicly available or reconstructable. Each example is represented as a paired sample $( X , Y )$ , where X is a prefix with $l _ { p } \in \{ 3 , 4 , 5 \}$ tokens and Y is the corresponding continuation with $l _ { c } \in \{ 2 , 3 , 4 \}$ tokens. For the training baseline, Y is the continuation observed in the corpus; for model samples, Y is generated from the same prefix using greedy decoding, nucleus sampling, or ancestral sampling. We then compare the paired training and model distributions using conditional von Neumann entropy (VNE), conditional order-2 matrix entropy, and lexica Distinct-2 and Distinct-3 scores.

In all numerical experiments and tables, we report the exponentiated forms $\exp ( H _ { \mathrm { v N } } )$ and $\exp ( H _ { 2 } )$ so that the reported values can be interpreted as efective numbers of conditionally distinguishable outputs under the corresponding entropy measures. Table 1 reports the results. The gap is defined as $\Delta =$ $\mathrm { M e t r i c _ { t r a i n } - M e t r i c _ { m o d e l } ; }$ therefore, positive values indicate that model-generated continuations have a smaller measured conditional output range than the training data. The pattern is consistent across all three model families and all decoding methods: the reported gap is positive.

Table 1: Diversity metrics across models and generation strategies with sample size 20K. For each model, we report conditional VNE, conditional RKE, and Distinct-2 as mean ± standard deviation.
<table><tr><td>Model</td><td>Group</td><td>Exp.Cond.VNE</td><td>ΔVNE</td><td>Exp.Cond.RKE</td><td>ΔRKE</td><td>Distinct-2</td><td>ΔD-2</td></tr><tr><td rowspan="4">OO</td><td>Training Data</td><td>338.58±0.69</td><td></td><td> $6 7 . 4 3 { \pm } 0 . 6 3$ </td><td></td><td>0.709±0.004</td><td></td></tr><tr><td>Greedy Decoding</td><td>218.88±0.34</td><td>119.70</td><td> $4 6 . 1 7 { \pm } 0 . 1 9$ </td><td>21.26</td><td>0.276±0.001</td><td>0.433</td></tr><tr><td>Nucleus Sampling</td><td> $2 8 7 . 3 1 { \pm } 0 . 9 8 $ </td><td>51.27</td><td> $5 5 . 0 8 { \pm } 0 . 2 7 $ </td><td>12.35</td><td>0.516±0.003</td><td>0.193</td></tr><tr><td>Ancestral Sampling</td><td>297.31±0.78</td><td>41.27</td><td> $5 6 . 8 1 { \pm } 0 . 2 2 $ </td><td>10.62</td><td>0.558±0.003</td><td>0.151</td></tr><tr><td rowspan="4">Pfthia</td><td>Training Data</td><td>262.55±0.47</td><td></td><td>53.44±0.38</td><td></td><td>0.701±0.006</td><td></td></tr><tr><td>Greedy Decoding</td><td>176.39±0.27</td><td>86.16</td><td>40.98±0.64</td><td>12.46</td><td>0.266±0.001</td><td>0.435</td></tr><tr><td>Nucleus Sampling</td><td>232.24±0.78</td><td>30.31</td><td> $4 7 . 5 1 { \pm } 0 . 2 7 $ </td><td>5.93</td><td>0.525±0.005</td><td>0.176</td></tr><tr><td>Ancestral Sampling</td><td>241.79±0.23</td><td>20.76</td><td> $4 9 . 0 0 { \pm } 0 . 4 2 \ $ </td><td>4.44</td><td>0.564±0.006</td><td>0.137</td></tr><tr><td rowspan="4">GT-eo</td><td>Training Data</td><td>260.01±0.52</td><td></td><td> $5 2 . 9 8 { \pm } 0 . 2 3 $ </td><td></td><td>0.701±0.008</td><td></td></tr><tr><td>Greedy Decoding</td><td>179.36±0.88</td><td>80.65</td><td> $4 3 . 3 4 { \pm } 0 . 9 3 $ </td><td>9.64</td><td>0.284±0.003</td><td>0.417</td></tr><tr><td>Nucleus Sampling</td><td>229.62±0.42</td><td>30.39</td><td> $4 8 . 4 8 { \pm } 0 . 6 7$ </td><td>4.50</td><td>0.526±0.002</td><td>0.175</td></tr><tr><td>Ancestral Sampling</td><td>238.50±0.61</td><td>21.51</td><td> $4 9 . 5 9 { \pm } 0 . 3 2 $ </td><td>3.39</td><td>0.570±0.002</td><td>0.131</td></tr></table>

The lexical Distinct-n scores provide a complementary surface-level check. They also show positive gaps in every setting, with the largest deficits again occurring under greedy decoding. However, Distinct-n does not explicitly account for the conditioning prefix and only measures n-gram variation in the generated continuations. We therefore use it as supporting evidence, while the main comparison is based on conditional entropy over paired prefix–continuation samples. The confidence intervals, computed over five independent evaluations, are small relative to the observed gaps, indicating that the efect is stable rather than a fluctuation of a single sample draw.

These experiments motivate two questions. First, what structural property makes conditional von Neumann entropy a principled analogue of Shannon conditional entropy? Second, if a model has already generated multiple candidate outputs for each input, can we reweight those candidates to obtain a higher-entropy conditional empirical distribution while remaining close to the original model samples? The next section answers the first question through a concavity result. The following section uses this result to formulate post-hoc conditional reweighting as a convex projection problem.

## 4 Theoretical Features of Kernel-induced Conditional von Neumann Entropy

We next establish the structural property needed for our proposed optimization of conditional von Neumann entropy. For Shannon entropy, we note that conditional entropy is well-known to be concave in the joint distribution $P _ { X Y }$ . In this section, we prove that the kernel matrix-based (quantum) conditional entropy defined in [12] used in our formulation is also a concave functional of the joint distribution. This property gives a convex-analytic justification for the projection method developed in Section 5. We begin by reviewing the corresponding known fact for the classical Shannon entropy.

Proposition 1 (Concavity of classical conditional Shannon entropy). For finite alphabets X and $\mathcal { V } _ { i }$ , the Shannon conditional entropy $P _ { X Y } \longmapsto H ( Y | X )$ is concave in the joint distribution $P _ { X Y }$

Proof. This proposition is known in the literature. For completeness, we give a proof in the Appendix.

Next, we highlight that the concavity of the kernel-induced matrix-based (quantum) conditional entropy in [12] does not follow from this result or a similar proof. This is because unlike the known identity for classical Shannon entropy that $H ( Y | X ) = \mathbb { E } _ { x \sim P _ { X } } [ H ( Y | X = x ) ]$ , the kernel-induced conditional von Neumann entropy in [12] does not satisfy the same identity. In the proof of the following proposition, we utilize the strong subadditivity of the quantum entropy to prove the concavity of the quantum conditional entropy functional as defined in [12]:

Proposition 2 (Concavity of kernel-induced conditional von Neumann entropy). Let $k _ { X }$ and $k _ { Y }$ be normalized positive semidefinite kernels. Then, the conditional von Neumann entropy defined in $\it { 1 2 }$

$$
P _ { X Y } \longmapsto { \mathcal { H } } _ { \mathrm { v N } } ( Y | X ; P _ { X Y } )
$$

is a concave functional of the joint distribution $P _ { X Y }$

Proof. We provide the proof in the Appendix.

Our shown Proposition 2 highlights the key structural property needed by the correction method we propose in the next section. Since superlevel sets of concave functions are convex sets, for every threshold $\rho$ the set

$$
\left\{ Q : { \mathcal { H } } _ { \mathrm { v N } } ( Y | X ; Q ) \geq \rho \right\}
$$

is a convex set. Therefore, if we search over empirical distributions supported on a fixed set of generated samples, the constraint that the reweighted distribution has conditional entropy at least $\rho$ is a convex constraint. As a result, the conditional von Neumann entropy is suitable not only as a diagnostic, but further as the basis for a tractable post-hoc reweighting procedure.

## 5 Kernel-based Entropy-Constrained Projection for Conditional Generators

The empirical gaps in Section 3 motivate a post-hoc correction problem. Suppose a conditional generator has already produced multiple candidate outputs for each input. We ask whether one can reweight these candidates so that the resulting conditional distribution moves toward a higher conditional-entropy region while remaining close to the original model distribution. The concavity of conditional von Neumann entropy gives a projection framework for this question.

Fix an input marginal $P _ { X }$ . For a conditional law $R _ { Y \mid X }$ , define

$$
\mathcal { H } _ { \mathrm { v N } } \big ( R _ { Y | X } ; P _ { X } \big ) : = \mathcal { H } _ { \mathrm { v N } } \big ( Y | X ; P _ { X } R _ { Y | X } \big ) ,
$$

where $P _ { X } R _ { Y \mid X }$ denotes the joint law induced by $P _ { X }$ and $R _ { Y \mid X }$ . For a threshold $\rho ,$ let

$$
\mathcal { C } _ { \rho } ( P _ { X } ) = \big \{ R _ { Y | X } : \mathcal { H } _ { \mathrm { v N } } ( R _ { Y | X } ; P _ { X } ) \geq \rho \big \} .
$$

By Proposition $2 , { \mathcal { C } } _ { \rho } ( P _ { X } )$ is a convex set.

We compare conditional laws through divergences lifted to joint laws with the same input marginal. Given a Bregman divergence $D _ { \Phi }$ on joint laws, define

$$
D _ { \Phi | P _ { X } } \bigl ( R _ { Y | X } , Q _ { Y | X } \bigr ) : = D _ { \Phi } \bigl ( P _ { X } R _ { Y | X } , P _ { X } Q _ { Y | X } \bigr ) .
$$

For example, when $D _ { \Phi }$ is KL divergence, this gives the P<sub>X</sub>-averaged conditional KL. When $D _ { \Phi }$ is the squared Hilbertian distance between kernel mean embeddings of joint pairs $( X , Y )$ , it gives the conditional MMD with the input marginal held fixed.

Theorem 1. Let $\rho = \mathcal { H } _ { \mathrm { v N } } \big ( P _ { Y | X } ; P _ { X } \big )$ , and let $Q _ { Y | X } ^ { \star }$ be the $D _ { \Phi | P _ { X } }$ -Bregman projection of $Q _ { Y \mid X }$ onto $\mathcal { C } _ { \rho } ( P _ { X } )$ Then, we have

$$
D _ { \Phi | P _ { X } } \left( P _ { Y | X } , Q _ { Y | X } ^ { \star } \right) \leq D _ { \Phi | P _ { X } } \left( P _ { Y | X } , Q _ { Y | X } \right) - D _ { \Phi | P _ { X } } \left( Q _ { Y | X } ^ { \star } , Q _ { Y | X } \right)
$$

The theorem gives a geometric interpretation of the projection step. If the entropy threshold is chosen so that the data conditional law is feasible, then the projection cannot increase the Bregman discrepancy from the data law. The second term is the Pythagorean decrease obtained by replacing $Q _ { Y \mid X }$ with its high-entropy projection. The proof is provided in Appendix C.

We now instantiate this principle on generated samples. Let $x _ { 1 } , \ldots , x _ { N }$ be input prompts, and let $y _ { i , 1 } , \ldots , y _ { i , m }$ be m generated outputs for prompt $x _ { i }$ . We assign weights $p _ { i } \in \Delta _ { m }$ to the candidates generated from $x _ { i }$ and place joint mass $\begin{array} { r } { q _ { i , j } = \frac { 1 } { N } p _ { i , j } \mathrm { ~ o n ~ } ( x _ { i } , y _ { i , j } ) } \end{array}$ . Hence each input keeps total mass $1 / N$ , and the reweighting modifies only the conditional distribution over generated outputs.

Let $K \in \mathbb { R } ^ { N m \times N m }$ be the product-kernel matrix over generated pairs,

$$
K _ { ( i , j ) , ( i ^ { \prime } , j ^ { \prime } ) } = k _ { X } ( x _ { i } , x _ { i ^ { \prime } } ) k _ { Y } ( y _ { i , j } , y _ { i ^ { \prime } , j ^ { \prime } } ) .
$$

Let $u _ { i , j } = 1 / ( N m )$ represent the uniform weight distribution (original baseline). We formulate the finitesupport projection as

$$
\begin{array} { r l } { \underset { q \in \mathbb { R } ^ { N _ { m } } } { \operatorname* { m i n } } } & { ( q - u ) ^ { \top } K ( q - u ) } \\ { \mathrm { s . t . } } & { \mathcal { H } _ { \mathrm { v N } } ( q ) \geq \rho , } \\ & { q _ { i , j } \geq 0 , \quad \displaystyle \sum _ { j ^ { \prime } = 1 } ^ { m } q _ { i , j ^ { \prime } } = \frac { 1 } { N } \quad \mathrm { f o r ~ a l l } i \in \{ 1 , \ldots , N \} , j \in \{ 1 , \ldots , m \} . } \end{array}\tag{1}
$$

The objective is the finite-sample conditional MMD between the reweighted empirical distribution and the original uniformly weighted model samples. The entropy term is the empirical conditional von Neumann entropy of the weighted paired sample. Since the block constraints fix the empirical input marginal, the input-entropy term is constant over the feasible set.

Proposition 3. The optimization problem equation 1 is a convex program, and its objective is the finite-support instance of the lifted squared-MMD discrepancy $D _ { \Phi | P _ { X } }$

For large sample sets, we solve a covariance-space approximation of equation 1. Let $\tilde { z } _ { i , j } \in \mathbb { R } ^ { r }$ be normalized sketched features for the pair $( x _ { i } , y _ { i , j } )$ , chosen so that their inner products approximate the product kernel. Define

$$
\widetilde { C } ( \boldsymbol { p } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { m } p _ { i , j } \widetilde { z } _ { i , j } \widetilde { z } _ { i , j } ^ { \top } .
$$

We optimize the penalized covariance-space objective

$$
\operatorname* { m i n } _ { p _ { 1 } , . . . , p _ { N } \in \Delta _ { m } } \quad F ( p ) : = \Big \| \frac { 1 } { N } \sum _ { i , j } ( p _ { i , j } - 1 / m ) \tilde { z } _ { i , j } \Big \| _ { 2 } ^ { 2 } - \lambda H _ { \mathrm { v N } } ( \widetilde { C } ( p ) ) .\tag{2}
$$

Proposition 4. The optimization problem equation 2 is a convex program. It is the covariance-space analogue of the kernel projection equation 1 under the sketched product-kernel representation.

We propose the algorithm Conditional Entropy Projection by Block Exponentiated Gradient (CEP-BEG) for solving equation 2. As described in Algorithm 1, CEP-BEG performs block mirror descent over the product simplex $\Delta _ { m } ^ { N } .$ with separated exponentiated-gradient (EG) normalization per input block. Note that we include this block normalization to preserve the empirical input marginal. Theorem 2 provides a convergence guarantee for the averaged iterate in CEP-BEG.

Theorem 2 (Convergence of CEP-BEG). Assume F is convex and that the CEP-BEG gradients satisfy $\begin{array} { r } { \sum _ { i = 1 } ^ { N } \| g _ { i } ^ { ( t ) } \| _ { \infty } ^ { 2 } \leq G ^ { 2 } } \end{array}$ for all t where $g _ { i } ^ { ( t ) } = ( g _ { i , 1 } ^ { ( t ) } , \ldots , g _ { i , m } ^ { ( t ) } )$ . Let $\begin{array} { r } { \bar { p } ^ { ( T ) } = \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } p ^ { ( t ) } } \end{array}$ . Then, choosing $\eta =$ $\sqrt { 2 N \log m } / ( G \sqrt { T } )$ , for any minimizer $p ^ { \star }$ of equation 2 we have

$$
F ( \bar { p } ^ { ( T ) } ) - F ( p ^ { \star } ) \leq G \sqrt { \frac { 2 N \log m } { T } } .
$$

The sketched features can be constructed in two complementary ways. For finite-dimensional embeddings, Gaussian random projection of the tensor-product feature gives a Johnson–Lindenstrauss-type guarantee on the generated sample set. For shift-invariant kernels, direct joint random Fourier features approximate the product kernel without forming tensor products. With sketch dimension $r ,$ each CEP-BEG iteration costs $O ( N m r ^ { 2 } + r ^ { 3 } )$ using a full eigendecomposition of the $r \times r$ covariance matrix. The proofs, gradient and feature constructions, and concentration bounds are in Appendix C.

```latex
Algorithm 1 CEP-BEG: Conditional Entropy Projection by Block Exponentiated Gradient
Require: Inputs $\{ x _ { i } \} _ { i = 1 } ^ { N }$ , generated outputs $\{ y _ { i , j } \} _ { j = 1 } ^ { m } ,$ sketch dimension r, entropy weight λ, step size η,
iterations $T .$
1: Construct normalized sketched joint features $\{ \tilde { z } _ { i , j } \in \mathbb { R } ^ { r } \}$
2: Initialize $p _ { i , j } ^ { ( 0 ) } = 1 / m$ for all $i , j .$
3: for $t = 0 , \ldots , T - 1$ do
4: Form the weighted covariance $\widetilde { C } ( \boldsymbol { p } ^ { ( t ) } )$
5: Compute a gradient or subgradient $\overset { \cdot } { g } ^ { ( t ) }$ of $F$ at $p ^ { ( t ) }$
6: Update each input block by $\begin{array} { r } { p _ { i , j } ^ { ( t + 1 ) } \stackrel {  } { = } p _ { i , j } ^ { ( t ) } \exp ( - \stackrel { \cdot } { \eta } g _ { i , j } ^ { ( t ) } ) / \big ( \sum _ { \ell = 1 } ^ { m } p _ { i , \ell } ^ { ( t ) } \exp ( - \eta g _ { i , \ell } ^ { ( t ) } ) \big ) } \end{array}$
7: end for
8: return Weights $q _ { i , j } = p _ { i , j } ^ { ( T ) } / N$ on the generated pairs.
```

## 6 Numerical Results

## 6.1 Experimental Setup

Models and datasets. Section 3 studies the conditional output-range gap for three fully open language models: OLMo [8], Pythia [9], and GPT-Neo [10]. Specifically, OLMo is trained on Dolma [14], a largescale open text corpus designed to support transparent language-model pretraining. Pythia and GPT-Neo are trained on The Pile [15], a diverse collection of English text from multiple domains, including web text, books, academic papers, code, and other curated sources. Beyond language models, we evaluate two conditional image-generation settings. For class-conditioned ImageNet generation, we follow the DGM-Eval benchmark [16] and use generated samples from LDM [17], ADM [18], BigGAN [19], and DiT-XL-2 [20], with ImageNet [21] training images as the real-data reference. For text-conditioned MS-COCO generation, we evaluate U-ViT [22], SDXL [23], and PixArt [24], using MS-COCO 2014 [25] training set as the reference distribution.

Tasks and evaluation protocol. We first evaluate conditional diversity gaps across language and image generation tasks, and then study two applications of the proposed conditional diversity objective. For language modeling, following Section 3, we compare training-corpus continuations with model continuations generated from the same prefixes using greedy decoding, nucleus sampling with $p = 0 . 9$ , and ancestral sampling. For image generation, we compare real and generated images under matched conditions, including class labels on ImageNet and captions on MS-COCO. We then evaluate the proposed objective in two application settings. For language models, we apply entropy-projected reweighting to multiple candidate continuations generated for each prefix, while preserving the prefix marginal. For text-to-image generation, we incorporate the conditional diversity objective into the difusion sampling process as a sampling-time guidance term, encouraging higher-diversity outputs under the same text condition on MS-COCO dataset.

Feature representations and kernels. We extract features using four fixed pretrained models: Qwen3- Embedding [26], CLIP [27], T5 [28], and DINOv2 [29]. Then we construct kernels on the resulting features and evaluate three kernel families: Gaussian, cosine, and degree-3 polynomial kernels. All kernels are normalized before computing conditional VNE and conditional RKE.

## 6.2 Conditional Diversity Gaps in Language Models

Figure 1 reports conditional VNE across sample sizes from 5K to 20K for OLMo, Pythia, and GPT-Neo. We also observe that conditional VNE increases with sample size for both training and generated continuations. However, the training-data curve grows faster than the generated-sample curves, so the measured diversity gap generally becomes larger as more samples are included. The gap is not restricted to the approximately 1B-parameter models used in the main experiments: we additionally evaluated OLMo-3-7B, Pythia-2.8B, and Pythia-6.9B, and the conditional output-range gap remains substantial across all three larger models; the complete sample-size curves are reported in Appendix E.10. We also perform a human-interpretable lexica audit counting distinct semantic categories in training versus generated continuations, which recovers the same ordering as conditional VNE; the details are reported in Appendix E.9.

Table 2: Conditional diversity gaps for long prefix–continuation sequences. We report the exponential of conditional VNE for the training reference and generated continuations under greedy decoding, nucleus sampling, and ancestral sampling, using prefix lengths {8, 16, 32} and continuation lengths {16, 32, 64}.
<table><tr><td>Model</td><td>Prefix</td><td>Continuation</td><td>Training</td><td>Greedy</td><td>Nucleus</td><td>Ancestral</td><td>Avg. Gap</td></tr><tr><td rowspan="12">GPT0</td><td>8</td><td>16</td><td>456.9</td><td>366.1</td><td>414.5</td><td>418.5</td><td>57.2</td></tr><tr><td>8</td><td>32</td><td>511.4</td><td>394.5</td><td>452.5</td><td>458.4</td><td>76.3</td></tr><tr><td>8</td><td>64</td><td>558.7</td><td>416.7</td><td>485.1</td><td>494.9</td><td>93.1</td></tr><tr><td>16</td><td>16</td><td>482.0</td><td>438.4</td><td>456.9</td><td>460.0</td><td>30.2</td></tr><tr><td>16</td><td>32</td><td>520.4</td><td>460.1</td><td>483.0</td><td>488.2</td><td>43.3</td></tr><tr><td>16</td><td>64</td><td>558.8</td><td>479.5</td><td>504.9</td><td>512.3</td><td>59.9</td></tr><tr><td>32</td><td>16</td><td>513.8</td><td>494.9</td><td>502.6</td><td>505.5</td><td>12.8</td></tr><tr><td>32</td><td>32</td><td>536.6</td><td>507.3</td><td>517.6</td><td>519.9</td><td>21.7</td></tr><tr><td>32</td><td>64</td><td>563.9</td><td>519.3</td><td>530.6</td><td>535.5</td><td>35.4</td></tr><tr><td>8</td><td>16</td><td>462.9</td><td>410.3</td><td>433.0</td><td>439.4</td><td>35.3</td></tr><tr><td>8 8</td><td>32</td><td>520.0</td><td>456.5</td><td>484.9</td><td>489.0</td><td>43.2</td></tr><tr><td></td><td>64</td><td>575.6</td><td>502.1</td><td>534.0</td><td>541.2</td><td>49.8</td></tr><tr><td>16 OO</td><td>16</td><td>487.8</td><td>456.9</td><td>470.5</td><td>474.0</td><td>20.7</td></tr><tr><td>16</td><td>32</td><td>530.4</td><td>488.0</td><td>506.0</td><td>510.4</td><td>28.9</td></tr><tr><td>16</td><td>64</td><td>576.9</td><td>521.7</td><td>545.4</td><td>549.2</td><td>38.1</td></tr><tr><td>32</td><td>16</td><td>524.0</td><td>506.1</td><td>513.6</td><td>516.0</td><td>12.1</td></tr><tr><td>32 32</td><td>32</td><td>551.4</td><td>524.7</td><td>534.8</td><td>538.1</td><td>18.9</td></tr><tr><td></td><td>64</td><td>585.2</td><td>546.9</td><td>561.1</td><td>564.2</td><td>27.8</td></tr><tr><td rowspan="10">Pyfthhia</td><td>8</td><td>16</td><td>463.3</td><td>374.9</td><td>423.0</td><td>429.7</td><td>54.1</td></tr><tr><td>8</td><td>32</td><td>519.2</td><td>415.3</td><td>468.2</td><td>475.2</td><td>66.3</td></tr><tr><td>8</td><td>64</td><td>568.2</td><td>452.9</td><td>506.7</td><td>509.5</td><td>78.5</td></tr><tr><td>16</td><td>16</td><td>488.6</td><td>449.1</td><td>465.9</td><td>469.3</td><td>27.2</td></tr><tr><td>16</td><td>32</td><td>527.7</td><td>480.1</td><td>496.6</td><td>499.3</td><td>35.7</td></tr><tr><td>16</td><td>64</td><td>567.7</td><td>510.6</td><td>519.8</td><td>523.4</td><td>49.8</td></tr><tr><td>32</td><td>16</td><td>521.6</td><td>503.4</td><td>511.5</td><td>513.7</td><td>12.1</td></tr><tr><td>32</td><td>32</td><td>545.0</td><td>519.5</td><td>528.3</td><td>530.2</td><td>19.0</td></tr><tr><td>32</td><td>64</td><td>572.8</td><td>536.4</td><td>539.8</td><td>542.7</td><td>33.2</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 6.3 Long-Sequence Conditional Diversity Gaps

To test whether the gap persists beyond the short prefix–continuation lengths in Section 3, we extend the experiments to prefix lengths {8, 16, 32} and continuation lengths {16, 32, 64} tokens across GPT-Neo, OLMo, and Pythia, representing each sequence as a concatenation of 2-token segment embeddings; the construction details and a chunk-size ablation are reported in Appendix E.6. Table 2 reports the exponentiated conditional VNE for the training reference and the three decoding strategies, along with the average gap. The training data exhibit higher conditional entropy than every decoding strategy in all evaluated configurations, indicating that the conditional diversity shortfall is a robust phenomenon across sequence lengths.

## 6.4 Conditional Diversity Gaps Beyond Language Models

Figure 2 and Figure 3 report conditional VNE across sample sizes from 2.5K to 20K for class-conditioned ImageNet generation and text-conditioned MS-COCO generation, respectively. In both settings, the real-data reference consistently achieves higher conditional VNE than the generated samples under matched conditions. We also observe that the gap generally increases with sample size, as the real-data conditional VNE grows faster than the generated-sample conditional VNE. This trend is consistent with the LLM experiments in Section 3.

![](images/84cfb2d399ed9c0ea29d6e7726bca02ed84c97f10ac90f03ec3566309c39bcce.jpg)  
Figure 1: Conditional diversity gaps between training data and generated data in language models.

Table 3: Efect of entropy-projected reweighting on conditional diversity. We report exponential of conditional VNE and conditional RKE at sample sizes 10K and 20K, averaged over 5 runs.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Setting</td><td colspan="2">Sample Size = 10K</td><td colspan="2">Sample Size = 20K</td></tr><tr><td>Exp.Cond.VNE</td><td>Exp.Cond.RKE</td><td>Exp.Cond.VNE</td><td>Exp.Cond.RKE</td></tr><tr><td rowspan="2">OLMo</td><td>w/o Projection</td><td>228.17±2.54</td><td>222.55±4.28</td><td>366.35±2.01</td><td>241.14±4.27</td></tr><tr><td>w/ / Projection</td><td>233.88±1.89</td><td>257.25±6.37</td><td>376.77±1.90</td><td>284.92±5.34</td></tr><tr><td rowspan="2">Pythia</td><td>w/o Projection</td><td>230.55±1.46</td><td>217.22±6.02</td><td>371.72±1.63</td><td>236.24±3.98</td></tr><tr><td>w/ Projection</td><td>236.94±1.84</td><td>254.40±5.69</td><td>382.74±1.38</td><td>285.05±4.11</td></tr><tr><td rowspan="2">GPT-Neo</td><td>w/o Projection</td><td>211.18±1.92</td><td>225.51±4.85</td><td>340.90±1.48</td><td>246.13±4.26</td></tr><tr><td>w/Projection</td><td>219.08±0.70</td><td>268.23±6.24</td><td>353.68±2.20</td><td>296.02±3.01</td></tr></table>

## 6.5 Entropy-Projected Reweighting for Conditional Generation

We next evaluate the proposed entropy-projected reweighting method on language generation. For each prefix, we generate 10 candidate continuations and reweight them using the conditional entropy projection in Section 5. Table 3 compares the original uniformly weighted candidates, denoted as “w/o Projection”, with the reweighted candidates, denoted as “w/ Projection”. These results indicate that the proposed projection can increase the measured conditional diversity of generated samples without retraining the model or generating new outputs.

We also compare the proposed projection with a temperature-scaling baseline and conduct a qualitative probe of the reweighting behavior; both are reported in Appendices E.7 and E.8. The temperature comparison shows that matching the conditional entropy alone is insuficient, and the probe confirms that the entropy gain comes from de-peaking valid modes rather than promoting low-quality outputs.

## 6.6 Conditional VNE Guidance for Difusion Sampling

We further evaluate whether the conditional VNE score can be used not only as an evaluation metric, but also as a sampling-time guidance signal for conditional generation. We apply conditional VNE guidance to SDXL on MS-COCO captions. Detailed implementation is provided in Appendix E.4. Table 4 reports the results. SDXL with conditional VNE guidance achieves higher conditional VNE than SDXL without guidance across all evaluated sample sizes.

## 6.7 Downstream Application: Conditional VNE-Weighted MBR Decoding

We next evaluate the proposed reweighting framework on a downstream task, model-based minimum Bayes risk (MBR) decoding [30], where replacing the original candidate weights with the conditional entropy projection yields a conditional-Vendi-weighted variant (CVS-MBR) that we compare with Monte Carlo MBR (MC-MBR), model-based MBR (MBMBR), and its length-normalized variant (MBMBR-L) on three summarization settings; implementation details are provided in Appendix D.3. Table 5 reports conditional VNE together with ROUGE-L, BERTScore, Distinct-2, and Self-BLEU-2. CVS-MBR achieves the highest conditional VNE and Distinct-2, as well as the lowest Self-BLEU-2, across all three settings, while a paired bootstrap shows no statistically significant degradation in ROUGE-L or BERTScore relative to MBMBR-L, demonstrating a diversity-oriented quality–diversity trade-of.

Table 4: Text-conditioned image generation on MS-COCO using SDXL with and without the guidance. We report exponential of conditional VNE as mean ± standard deviation over 5 independent runs.
<table><tr><td>Data / Model</td><td>10K</td><td>12.5K</td><td>15K</td><td>17.5K</td><td>20K</td></tr><tr><td>Reference MS-COCO</td><td> $2 8 . 5 1 \pm 0 . 0 6$ </td><td> $3 1 . 9 3 \pm 0 . 0 6$ </td><td> $3 5 . 0 3 \pm 0 . 0 3$ </td><td> $3 7 . 9 2 \pm 0 . 0 4$ </td><td> $4 0 . 5 2 \pm 0 . 0 3$ </td></tr><tr><td>SDXL w/o Guidance</td><td> $2 2 . 5 1 \pm 0 . 0 9$ </td><td> $2 4 . 7 6 \pm 0 . 0 7$ </td><td> $2 6 . 7 6 \pm 0 . 0 7$ </td><td> $2 8 . 5 8 \pm 0 . 0 4$ </td><td> $3 0 . 2 5 \pm 0 . 0 4$ </td></tr><tr><td>SDXL w/ Guidance</td><td> $2 3 . 7 5 \pm 0 . 0 9$ </td><td> $2 6 . 2 1 \pm 0 . 0 6$ </td><td> $2 8 . 4 4 \pm 0 . 0 7$ </td><td> $3 0 . 4 3 \pm 0 . 0 4$ </td><td> $3 2 . 2 7 \pm 0 . 0 2$ </td></tr></table>

Table 5: Conditional VNE-weighted MBR decoding on three summarization settings. We report conditional VNE, ROUGE-L, BERTScore, Distinct-2, and Self-BLEU-2. Arrows indicate the desirable direction for each metric. CVS-MBR denotes the proposed conditional-Vendi-weighted variant.
<table><tr><td>Setting</td><td>Method</td><td>Cond.VNE ↑</td><td>ROUGE-L ↑</td><td>BERTScore ↑</td><td>Distinct-2 ↑</td><td>Self-BLEU-2 ↓</td></tr><tr><td rowspan="4">XSum BART-large-XSum</td><td>MC-MBR</td><td>27.88</td><td>0.335</td><td>0.915</td><td>0.936</td><td>0.209</td></tr><tr><td>MBMBR</td><td>4.93</td><td>0.346</td><td>0.918</td><td>0.929</td><td>0.212</td></tr><tr><td>MBMBR-L</td><td>26.44</td><td>0.350</td><td>0.917</td><td>0.938</td><td>0.207</td></tr><tr><td>CVS-MBR</td><td>28.33</td><td>0.331</td><td>0.915</td><td>0.945</td><td>0.197</td></tr><tr><td rowspan="4">CNN/DailyMail BART-large-CNN</td><td>MC-MBR MBMBR</td><td>33.00</td><td>0.268</td><td>0.879</td><td>0.944</td><td>0.186</td></tr><tr><td></td><td>3.35</td><td>0.273</td><td>0.880</td><td>0.945</td><td>0.184</td></tr><tr><td>MBMBR-L CVS-MBR</td><td>28.80</td><td>0.266</td><td>0.879</td><td>0.948</td><td>0.174</td></tr><tr><td></td><td>33.34</td><td>0.272</td><td>0.879</td><td>0.951</td><td>0.165</td></tr><tr><td rowspan="4">CNN/DailyMail Mistral-7B-Instruct</td><td>MC-MBR</td><td>12.20</td><td>0.256</td><td>0.880</td><td>0.919</td><td>0.233</td></tr><tr><td>MBMBR</td><td>4.62</td><td>0.254</td><td>0.881</td><td>0.924</td><td>0.229</td></tr><tr><td>MBMBR-L</td><td>13.15</td><td>0.261</td><td>0.881</td><td>0.920</td><td>0.234</td></tr><tr><td>CVS-MBR</td><td>14.09</td><td>0.261</td><td>0.881</td><td>0.926</td><td>0.224</td></tr></table>

Ablation Studies. To examine the robustness of the conditional diversity gaps, we conduct ablation studies under diferent experimental choices. Specifically, we vary the evaluation sample size, the kernel function, and the embedding model. For the language-model experiments, we further vary the prefix– continuation construction by considering prefix token lengths in {3, 4, 5} and continuation token lengths in {2, 3, 4}. For the long-sequence experiments, we additionally perform a chunk-size ablation over chunk sizes {1, 2, 4} used in the segment concatenation representation. For the sample-size analysis, we draw subsamples from 2.5k to 20k with diferent random seeds out of 30k samples. Full results are provided in the Appendix.

## 7 Conclusion and Limitations

We studied whether LLMs and conditional generative models capture the full range of valid outputs present in their training distributions. Using conditional entropy and its matrix-based von Neumann analogue, we observed a consistent conditional output-range gap across open language models and conditional imagegeneration settings. We specifically showed the concavity of kernel-induced conditional von Neumann entropy, supporting both its use as a diagnostic and the convexity of entropy-constrained projection. Building on this structure, we proposed a post-hoc reweighting method that preserves the input marginal while redistributing mass across generated candidates, with an eficient product-simplex mirror-descent implementation using sketched joint features.

The proposed entropy-based view complements standard evaluations of factuality, alignment, and semantic correctness. Our direct training-data comparison focuses on models for which the relevant training data are publicly available or reconstructable; for many widely used LLMs and generative AI models, the training data are not publicly accessible, making the analysis more dificult but potentially valuable. These considerations suggest natural extensions to closed-source systems and alternative feature choices.

## References

[1] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving language understanding by generative pre-training. 2018.

[2] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

[3] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.

[4] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. BLIP-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 19730–19742. PMLR, 2023.

[5] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. CLIPScore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 7514–7528. Association for Computational Linguistics, 2021.

[6] Krishna Pillutla, Swabha Swayamdipta, Rowan Zellers, John Thickstun, Sean Welleck, Yejin Choi, and Zaïd Harchaoui. MAUVE: Measuring the gap between neural text and human text using divergence frontiers. In Advances in Neural Information Processing Systems, volume 34, pages 4816–4828, 2021.

[7] Jaywon Koo, Jeferson Hernandez, Moayed Haji-Ali, Ziyan Yang, and Vicente Ordóñez. Evaluating text-to-image synthesis with a conditional fréchet distance. CoRR, abs/2503.21721, 2025.

[8] Dirk Groeneveld, Iz Beltagy, Evan Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David Atkinson, Russell Authur, Khyathi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar, Yuling Gu, Jack Hessel, Tushar Khot, William Merrill, Jacob Morrison, Niklas Muennighof, Aakanksha Naik, Crystal Nam, Matthew Peters, Valentina Pyatkin, Abhilasha Ravichander, Dustin Schwenk, Saurabh Shah, William Smith, Emma Strubell, Nishant Subramani, Mitchell Wortsman, Pradeep Dasigi, Nathan Lambert, Kyle Richardson, Luke Zettlemoyer, Jesse Dodge, Kyle Lo, Luca Soldaini, Noah Smith, and Hannaneh Hajishirzi. OLMo: Accelerating the science of language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15789–15809. Association for Computational Linguistics, August 2024.

[9] Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, Usvsn Sai Prashanth, Edward Raf, Aviya Skowron, Lintang Sutawika, and Oskar Van Der Wal. Pythia: A suite for analyzing large language models across training and scaling. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 2397–2430. PMLR, 23–29 Jul 2023.

[10] Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. GPT-Neo: Large scale autoregressive language modeling with meshtensorflow, October 2021.

[11] Farzan Farnia, Mohammad Jalali, and Azim Ospanov. Exposing diversity bias in deep generative models: Statistical origins and correction of diversity error. In ICLR 2026 2nd Workshop on Deep Generative Model in Machine Learning: Theory, Principle and Eficacy, 2026.

[12] Luis G. Sanchez Giraldo, Murali Rao, and Jose C. Principe. Measures of entropy from data using infinitely divisible kernels. IEEE Transactions on Information Theory, 61(1):535–548, 2015.

[13] Mohammad Jalali, Azim Ospanov, Amin Gohari, and Farzan Farnia. Conditional vendi score: Promptaware diversity evaluation for text-guided generative ai models. In Proceedings of The 29th International Conference on Artificial Intelligence and Statistics, Proceedings of Machine Learning Research. PMLR, pages 02–05, 2026.

[14] Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, Ananya Harsh Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Nathan Lambert, Ian Magnusson, Jacob Morrison, Niklas Muennighof, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang Shen, Emma Strubell, Nishant Subramani, Oyvind Tafjord, Pete Walsh, Luke Zettlemoyer, Noah A. Smith, Hannaneh Hajishirzi, Iz Beltagy, Dirk Groeneveld, Jesse Dodge, and Kyle Lo. Dolma: An open corpus of three trillion tokens for language model pretraining research. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15725–15788. Association for Computational Linguistics, 2024.

[15] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. The Pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.

[16] George Stein, Jesse Cresswell, Rasa Hosseinzadeh, Yi Sui, Brendan Ross, Valentin Villecroze, Zhaoyan Liu, Anthony L Caterini, Eric Taylor, and Gabriel Loaiza-Ganem. Exposing flaws of generative mode evaluation metrics and their unfair treatment of difusion models. In Advances in Neural Information Processing Systems, volume 36, 2023.

[17] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.

[18] Prafulla Dhariwal and Alexander Quinn Nichol. Difusion models beat gans on image synthesis. In Advances in Neural Information Processing Systems, volume 34, pages 8780–8794, 2021.

[19] Andrew Brock, Jef Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. In International Conference on Learning Representations, 2019.

[20] William Peebles and Saining Xie. Scalable difusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.

[21] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale hierarchical image database. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009.

[22] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A ViT backbone for difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22669–22679, 2023.

[23] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: Improving latent difusion models for high-resolution image synthesis. In International Conference on Learning Representations, 2024.

[24] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. PIXART-Σ: Weak-to-strong training of difusion transformer for 4k text-to-image generation. In European Conference on Computer Vision, pages 74–91. Springer, 2024.

[25] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO: Common objects in context. In European Conference on Computer Vision, pages 740–755. Springer, 2014.

[26] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 Embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025.

[27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR, 2021.

[28] Colin Rafel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020.

[29] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024.

[30] Yuu Jinnai, Yasuhito Yasui, Kosuke Koyama, Hidetoshi Shimizu, and Tohru Nishikawa. Model-based minimum Bayes risk decoding for text generation. In Proceedings of the International Conference on Machine Learning, 2024.

[31] Yushi Hu, Benlin Liu, Jungo Kasai, Yizhong Wang, Mari Ostendorf, Ranjay Krishna, and Noah A. Smith. TIFA: Accurate and interpretable text-to-image faithfulness evaluation with question answering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

[32] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. In Advances in Neural Information Processing Systems, volume 36, 2023.

[33] Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and Bill Dolan. A diversity-promoting objective function for neural conversation models. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 110–119. Association for Computational Linguistics, 2016.

[34] Yaoming Zhu, Sidi Lu, Lei Zheng, Jiaxian Guo, Weinan Zhang, Jun Wang, and Yong Yu. Texygen: A benchmarking platform for text generation models. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval, pages 1097–1100. ACM, 2018.

[35] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. BLEU: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318. Association for Computational Linguistics, 2002.

[36] Ramakrishna Vedantam, C. Lawrence Zitnick, and Devi Parikh. CIDEr: Consensus-based image description evaluation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 4566–4575, 2015.

[37] Peter Anderson, Basura Fernando, Mark Johnson, and Stephen Gould. SPICE: Semantic propositional image caption evaluation. In Computer Vision – ECCV 2016, volume 9909 of Lecture Notes in Computer Science, pages 382–398. Springer, 2016.

[38] Thomas Unterthiner, Sjoerd van Steenkiste, Karol Kurach, Raphaël Marinier, Marcin Michalski, and Sylvain Gelly. FVD: A new metric for video generation. In ICLR 2019 Workshop on Deep Generative Models for Highly Structured Data, 2019.

[39] Jay Zhangjie Wu, Guian Fang, Haoning Wu, Xintao Wang, Yixiao Ge, Xiaodong Cun, David Junhao Zhang, Jia-Wei Liu, Yuchao Gu, Rui Zhao, Weisi Lin, Wynne Hsu, Ying Shan, and Mike Zheng Shou. Towards a better metric for text-to-video generation. CoRR, abs/2401.07781, 2024.

[40] Yuanxin Liu, Lei Li, Shuhuai Ren, Rundong Gao, Shicheng Li, Sishuo Chen, Xu Sun, and Lu Hou. FETV: A benchmark for fine-grained evaluation of open-domain text-to-video generation. CoRR, abs/2311.01813, 2023.

[41] Azim Ospanov, Mohammad Jalali, and Farzan Farnia. Scendi score: Prompt-aware diversity evaluation via schur complement of clip embeddings. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 16927–16937. IEEE, 2025.

[42] Mohammad Jalali, Haoyu Lei, Amin Gohari, and Farzan Farnia. Sparke: Scalable prompt-aware diversity guidance in difusion models via RKE score. CoRR, abs/2506.10173, 2025.

[43] Mischa Dombrowski, Weitong Zhang, Sarah Cechnicka, Hadrien Reynaud, and Bernhard Kainz. Image generation diversity issues and how to tame them. CoRR, abs/2411.16171, 2025.

[44] Longfei Yun, Chenyang An, Zilong Wang, Letian Peng, and Jingbo Shang. The price of format: Diversity collapse in LLMs. CoRR, abs/2505.18949, 2025.

[45] Qingzhong Wang and Antoni B. Chan. Describing like humans: On diversity in image captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4195–4203, 2019.

[46] Nicholas Carlini, Florian Tramèr, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Úlfar Erlingsson, Alina Oprea, and Colin Rafel. Extracting training data from large language models. In 30th USENIX Security Symposium (USENIX Security 21), pages 2633–2650. USENIX Association, 2021.

[47] Nicholas Carlini, Jamie Hayes, Milad Nasr, Matthew Jagielski, Vikash Sehwag, Florian Tramèr, Borja Balle, Daphne Ippolito, and Eric Wallace. Extracting training data from difusion models. CoRR, abs/2301.13188, 2023.

[48] Xiang Gu, Yaxing Wang, Yueyuan Li, Xiaochuan Li, Tim Salimans, Chao Zhang, and Yang Song. On memorization in difusion models. In International Conference on Learning Representations, 2024.

[49] Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In International Conference on Learning Representations, 2023.

[50] Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630, 2024.

[51] Ziniu Li, Congliang Chen, Tian Xu, Zeyu Qin, Jiancong Xiao, Zhi-Quan Luo, and Ruoyu Sun. Preserving diversity in supervised fine-tuning of large language models. In International Conference on Learning Representations, 2025.

[52] Han Shen. On entropy control in llm-rl algorithms. In International Conference on Learning Representations, 2026.

[53] Alexander Nikitin, Jannik Kossen, Yarin Gal, and Pekka Marttinen. Kernel language entropy: Finegrained uncertainty quantification for llms from semantic similarities. In Advances in Neural Information Processing Systems, volume 37, 2024.

[54] Dan Friedman and Adji Bousso Dieng. The vendi score: A diversity evaluation metric for machine learning. Transactions on Machine Learning Research, 2023.

[55] Azim Ospanov, Jingwei Zhang, Mohammad Jalali, Xuenan Cao, Andrej Bogdanov, and Farzan Farnia. Towards a scalable reference-free evaluation of generative models. Advances in Neural Information Processing Systems, 37:120892–120927, 2024.

[56] Mohammad Jalali, Cheuk Ting Li, and Farzan Farnia. An information-theoretic evaluation of generative models in learning multi-modal distributions. Advances in Neural Information Processing Systems, 36:9931–9943, 2023.

[57] Lai Wei, Zhiquan Tan, Chenghai Li, Jindong Wang, and Weiran Huang. Dif-erank: A novel rank-based metric for evaluating large language models. In Advances in Neural Information Processing Systems, volume 37, 2024.

[58] Md Rifat Arefin, Gopeshh Subbaraj, Nicolas Gontier, Yann LeCun, Irina Rish, Ravid Shwartz-Ziv, and Christopher Pal. Seq-vcr: Preventing collapse in intermediate transformer representations for enhanced reasoning. In International Conference on Learning Representations, 2025.

[59] Gabriel Pereyra, George Tucker, Jan Chorowski, Łukasz Kaiser, and Geofrey Hinton. Regularizing neural networks by penalizing confident output distributions. CoRR, abs/1701.06548, 2017.

[60] Christian Szegedy, Vincent Vanhoucke, Sergey Iofe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2818–2826, 2016.

[61] Joshua C. Peterson, Ruairidh M. Battleday, Thomas L. Grifiths, and Olga Russakovsky. Human uncertainty makes classification more robust. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9616–9625, 2019.

[62] Esla Timothy Anzaku, Laurent Obongo, Paul Bogolin, and Hisham Cholakkal. Re-assessing imagenet: How aligned is its single-label assumption with its multi-label nature? CoRR, abs/2412.18409, 2024.

[63] Elliott H. Lieb and Mary Beth Ruskai. Proof of the strong subadditivity of quantum-mechanical entropy. Physical Review Letters, 30(10):434–436, 1973.

[64] Avery Katz, Akshay Jaiswal, Michelle Nesca, and Evangelos Milios. Identifying risk factors associated with lower back pain in electronic medical record free text: Deep learning approach using clinical note annotations. JMIR Medical Informatics, 2023.

[65] Akshay Jaiswal and Evangelos Milios. Breaking the token barrier: Chunking and convolution for eficient long text classification with BERT. arXiv preprint arXiv:2304.11547, 2023.

[66] Michael Günther, Isabelle Mohr, Daniel J. Williams, Bo Wang, and Han Xiao. Late chunking: Contextua chunk embeddings using long-context embedding models. arXiv preprint arXiv:2409.03753, 2024.

[67] Sai Zhang, Yizhuo Liang, Ming Gong, Daxin Jiang, and Nan Duan. Multi-view document representation learning for open-domain dense retrieval. In Proceedings of the ACM Web Conference, 2022.

[68] Keshav Santhanam, Omar Khattab, Jon Saad-Falcon, Christopher Potts, and Matei Zaharia. ColBERTv2: Efective and eficient retrieval via lightweight late interaction. In Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics, 2022.

[69] Thomas Le Bronnec, Armand Yang, Jean-Clédent Renault, Gilles Biau, and Emmanuel Vincent. Exploring precision and recall to assess the quality and diversity of LLMs. In Proceedings of the Annual Meeting of the Association for Computational Linguistics, 2024.

## Appendix A Related Work

Evaluation of conditional generative AI. Evaluation in conditional generation is shaped primarily by fidelity, alignment, and preference-based criteria rather than by intrinsic variety under fixed conditions. In text-to-image generation, representative examples include CLIPScore [5], TIFA [31], ImageReward [32], and more recent conditional fidelity metrics such as cFreD [7]. In open-ended text generation, MAUVE [6] has become a prominent distributional metric, while lexical diversity diagnostics such as distinct-n [33] and Self-BLEU [34] remain common. In image captioning, standard evaluation still centers on BLEU [35], CIDEr [36], and SPICE [37]. In text-to-video generation, common practice combines video-quality metrics such as FVD [38] with newer alignment-oriented metrics such as T2VScore [39] and evaluation suites such as FETV [40]. Also, the Scendi score [41] and Conditional Vendi/RKE scores [13, 42] provide conditional diversity scores for prompt-aware diversity quantification. These works have substantially advanced benchmarking for conditional generation, but they are not designed to isolate the variety that a model contributes under a fixed condition. Our work complements this literature by focusing specifically on conditional variety as a first-class evaluation target.

Variety shortfall, novelty, and memorization in modern generative AI. A growing empirical literature suggests that diversity shortfall is a real phenomenon in modern generative systems. Specifically, Farnia et al. [11] conduct a diversity gap analysis for unconditional generative models, numerically showing a von Neumann entropy gap between standard generative models and their training data, which we extend to the conditional generation case by analyzing the conditional von Neumann entropy function. In image generation, Dombrowski et al. [43] report that current state-of-the-art models cover only a limited fraction of the diversity present in the training distribution. In LLMs, Yun et al. [44] identify diversity collapse, where formatting and instruction structure drive semantically similar outputs even under decoding regimes intended to encourage variety. In image captioning, Wang and Chan [45] show that models optimized for standard captioning metrics often produce generic captions and remain far below human performance in diversity. Related concerns arise in the memorization and novelty literature. Carlini et al. [46] demonstrate that large language models can regurgitate memorized training data, while Carlini et al. [47] show analogous extraction phenomena for difusion models. More recent empirical studies also investigate memorization behavior in difusion models directly [48]. Our work difers in that it uses the conditional kernel entropy to analyze a shortfall relative to the empirical training distribution.

Entropy-based analysis of LLMs. Entropy has recently become a useful tool for studying uncertainty, reliability, and training dynamics in LLMs. Semantic entropy [49, 50] estimates uncertainty over meanings rather than surface forms and has been used for hallucination detection and reliability assessment. Other work uses entropy directly in training or fine-tuning: Li et al. [51] connect diversity preservation in supervised fine-tuning to entropy-regularized objectives, while Shen [52] studies entropy control in LLM reinforcement learning. These works use entropy mainly for uncertainty estimation, hallucination detection, or training-time regularization. Our focus is diferent: we compare the conditional entropy of model-generated outputs against the corresponding training distribution and study the resulting conditional output-range gap.

Matrix-entropy methods for LLMs. Closely related to our use of matrix-based entropy are recent works applying von Neumann or spectral entropy to LLM outputs and representations. Nikitin et al. [53] propose Kernel Language Entropy (KLE), a method for uncertainty quantification that builds positive semidefinite unit-trace kernels over LLM outputs and measures semantic uncertainty using von Neumann entropy. Vendi score [54, 55] uses von Neumann entropy of the kernel matrices for evaluating the variety of the generated data, which has been similarly formulated for order-2 Rényi entropy as the RKE score [56]. Wei et al. [57] use matrix-entropy-inspired representation spectra for LLM evaluation, showing that rank/entropy-based representation measures correlate with model scale and standard performance indicators. Arefin et al. [58] use matrix-based entropy to diagnose representation collapse in intermediate Transformer layers and propose a regularizer to increase representation richness for reasoning. These works demonstrate that spectral entropy is informative for LLM uncertainty, representation quality, and training behavior. However, they primarily analyze unconditional output sets or internal representations. On the other hand our work studies conditional entropy and compares model outputs directly to their training data.

Entropy regularization and predictive uncertainty. Finally, our work is related in spirit to entropy-promoting regularization in supervised learning. Confidence-penalty regularization [59] and label smoothing [60] both discourage overly sharp predictive distributions and have been shown to improve deep models across tasks. More recent work has emphasized that real-world supervised datasets may contain substantial ambiguity even when only a single hard label is observed, as reflected for example in human label distributions [61] and critiques of the single-label assumption in ImageNet-style benchmarks [62]. Although these works are not about generative diversity evaluation, they share a common theme with our analysis: finite datasets can under-represent the uncertainty or variability of the underlying task, and entropy-promoting mechanisms can partially compensate for this mismatch.

## Appendix B Extended Preliminaries

This appendix records the population-level operator definitions corresponding to the empirical quantities in Section 2. Throughout, k<sub>X</sub> and $k _ { Y }$ are normalized positive semidefinite kernels with feature maps

$$
\phi _ { X } : \mathcal { X } \to \mathcal { H } _ { X } , \qquad \phi _ { Y } : \mathcal { Y } \to \mathcal { H } _ { Y } ,
$$

so that $k _ { X } ( x , x ^ { \prime } ) = \langle \phi _ { X } ( x ) , \phi _ { X } ( x ^ { \prime } ) \rangle _ { \mathcal { H } _ { X } } , k _ { Y } ( y , y ^ { \prime } ) = \langle \phi _ { Y } ( y ) , \phi _ { Y } ( y ^ { \prime } ) \rangle _ { \mathcal { H } _ { Y } }$ , and

$$
\| \phi _ { X } ( x ) \| _ { \mathcal { H } _ { X } } ^ { 2 } = 1 , \qquad \| \phi _ { Y } ( y ) \| _ { \mathcal { H } _ { Y } } ^ { 2 } = 1 .
$$

The product kernel on paired samples is induced by the tensor-product feature map

$$
\psi ( x , y ) = \phi _ { X } ( x ) \otimes \phi _ { Y } ( y ) ,
$$

since

$$
\langle \psi ( x , y ) , \psi ( x ^ { \prime } , y ^ { \prime } ) \rangle _ { \mathcal { H } _ { X } \otimes \mathcal { H } _ { Y } } = k _ { X } ( x , x ^ { \prime } ) k _ { Y } ( y , y ^ { \prime } ) .
$$

For a joint distribution $P _ { X Y }$ , define the joint and marginal covariance operators

$$
C _ { X Y } ( P ) = \mathbb { E } _ { ( X , Y ) \sim P } \big [ \psi ( X , Y ) \psi ( X , Y ) ^ { * } \big ] , \qquad C _ { X } ( P _ { X } ) = \mathbb { E } _ { X \sim P _ { X } } \big [ \phi _ { X } ( X ) \phi _ { X } ( X ) ^ { * } \big ] .
$$

The normalization of the kernels implies

$$
\mathrm { T r } ( C _ { X Y } ( { \cal P } ) ) = 1 , \qquad \mathrm { T r } ( C _ { X } ( { \cal P } _ { X } ) ) = 1 ,
$$

so these are trace-one positive semidefinite operators whenever the expectations are well defined. The population conditional von Neumann entropy is

$$
\mathcal { H } _ { \mathrm { v N } } ( Y \mid X ; P ) : = H _ { \mathrm { v N } } ( C _ { X Y } ( P ) ) - H _ { \mathrm { v N } } ( C _ { X } ( P _ { X } ) ) .
$$

This is the population operator analogue of the empirical quantity

$$
H _ { \mathrm { v N } } \big ( \frac { 1 } { n } K _ { X } \odot K _ { Y } \big ) - H _ { \mathrm { v N } } \big ( \frac { 1 } { n } K _ { X } \big ) .
$$

Indeed, for the empirical distribution $\begin{array} { r } { \widehat { P } _ { n } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \delta _ { ( x _ { i } , y _ { i } ) } } \end{array}$ , the nonzero eigenvalues of the empirical covariance operator

$$
C _ { X Y } ( \widehat { P } _ { n } ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \psi ( x _ { i } , y _ { i } ) \psi ( x _ { i } , y _ { i } ) ^ { * }
$$

coincide with the nonzero eigenvalues of ${ \scriptstyle { \frac { 1 } { n } } } K _ { X Y }$ , where $K _ { X Y } = K _ { X } \odot K _ { Y }$ . Similarly, the nonzero eigenvalues of $C _ { X } ( \widehat { P } _ { n , X } )$ coincide with those of ${ \scriptstyle { \frac { 1 } { n } } } K _ { X }$ . Thus the Gram-matrix formula in the main text is exactly the finite-sample representation of the operator definition.

The order-2 version is defined at the population level by

$$
\displaystyle \mathcal { H } _ { 2 } ( Y \mid X ; P ) : = H _ { 2 } ( C _ { X Y } ( P ) ) - H _ { 2 } ( C _ { X } ( P _ { X } ) ) , \qquad H _ { 2 } ( A ) : = - \log \operatorname { T r } ( A ^ { 2 } ) .
$$

For empirical samples, this becomes

$$
H _ { 2 } \big ( \frac { 1 } { n } K _ { X Y } \big ) - H _ { 2 } \big ( \frac { 1 } { n } K _ { X } \big ) = \log \frac { \| K _ { X } \| _ { F } ^ { 2 } } { \| K _ { X } \odot K _ { Y } \| _ { F } ^ { 2 } } .
$$

The equality follows from

$$
\operatorname { T r } \bigl ( \bigl ( \frac 1 n K _ { X } \bigr ) ^ { 2 } \bigr ) = \frac 1 { n ^ { 2 } } \| K _ { X } \| _ { F } ^ { 2 } , \qquad \operatorname { T r } \bigl ( \bigl ( \frac 1 n K _ { X Y } \bigr ) ^ { 2 } \bigr ) = \frac 1 { n ^ { 2 } } \| K _ { X } \odot K _ { Y } \| _ { F } ^ { 2 } .
$$

The normalized-kernel assumption is used throughout to ensure that the covariance operators have unit trace and hence can be interpreted as density operators for von Neumann entropy. If a kernel k satisfies $k ( z , z ) > 0$ , the normalized kernel

$$
\widetilde { k } ( z , z ^ { \prime } ) = \frac { k ( z , z ^ { \prime } ) } { \sqrt { k ( z , z ) k ( z ^ { \prime } , z ^ { \prime } ) } }
$$

is positive semidefinite and satisfies $\widetilde { k } ( z , z ) = 1$ . In infinite-dimensional settings, the definitions above are understood for trace-class covariance operators with finite entropy; all empirical Gram-matrix quantities used in the experiments are finite-dimensional and therefore well defined.

## Appendix C Proofs and Additional Theoretical Results

## C.1 Proof of Proposition 1

Proof. Let $P _ { X Y } ^ { ( 0 ) }$ and $P _ { X Y } ^ { ( 1 ) }$ be two joint distributions on finite alphabets X and $\mathcal { V } .$ and let

$$
P _ { X Y } ^ { \lambda } = ( 1 - \lambda ) P _ { X Y } ^ { ( 0 ) } + \lambda P _ { X Y } ^ { ( 1 ) } , \qquad \lambda \in [ 0 , 1 ] .
$$

We use the convention 0 log $0 = 0$ . For any joint law $P _ { X Y }$ ,

$$
H ( Y \mid X ; P ) = - \sum _ { x , y } P _ { X Y } ( x , y ) \log { \frac { P _ { X Y } ( x , y ) } { P _ { X } ( x ) } } .
$$

Fix $x \in \mathcal { X }$ . Applying the log-sum inequality to the two collections

$$
a _ { y } = ( 1 - \lambda ) P _ { X Y } ^ { ( 0 ) } ( x , y ) , \qquad b _ { y } = \lambda P _ { X Y } ^ { ( 1 ) } ( x , y )
$$

gives

$$
\sum _ { y } P _ { X Y } ^ { \lambda } ( x , y ) \log { \frac { P _ { X Y } ^ { \lambda } ( x , y ) } { P _ { X } ^ { \lambda } ( x ) } } \leq ( 1 - \lambda ) \sum _ { y } P _ { X Y } ^ { ( 0 ) } ( x , y ) \log { \frac { P _ { X Y } ^ { ( 0 ) } ( x , y ) } { P _ { X } ^ { ( 0 ) } ( x ) } } + \lambda \sum _ { y } P _ { X Y } ^ { ( 1 ) } ( x , y ) \log { \frac { P _ { X Y } ^ { ( 1 ) } ( x , y ) } { P _ { X } ^ { ( 1 ) } ( x ) } } .
$$

Multiplying by −1 and summing over x yields

$$
H ( Y | X ; P ^ { \lambda } ) \ge ( 1 - \lambda ) H ( Y | X ; P ^ { ( 0 ) } ) + \lambda H ( Y | X ; P ^ { ( 1 ) } ) ,
$$

which proves concavity.

## C.2 Proof of Proposition 2

We first make explicit the operator representation used in the proof. Let $k _ { X }$ and $k _ { Y }$ be normalized positive semidefinite kernels with feature maps

$$
\phi _ { X } : \mathcal { X } \to \mathcal { H } _ { X } , \qquad \phi _ { Y } : \mathcal { Y } \to \mathcal { H } _ { Y } ,
$$

satisfying

$$
\| \phi _ { X } ( x ) \| _ { \mathcal { H } _ { X } } ^ { 2 } = k _ { X } ( x , x ) = 1 , \qquad \| \phi _ { Y } ( y ) \| _ { \mathcal { H } _ { Y } } ^ { 2 } = k _ { Y } ( y , y ) = 1 .
$$

For a joint distribution $P _ { X Y }$ , define

$$
C _ { X Y } ( P ) = \mathbb { E } _ { ( X , Y ) \sim P } \Big [ ( \phi _ { X } ( X ) \otimes \phi _ { Y } ( Y ) ) ( \phi _ { X } ( X ) \otimes \phi _ { Y } ( Y ) ) ^ { * } \Big ] ,
$$

and

$$
C _ { X } ( P _ { X } ) = \mathbb { E } _ { X \sim P _ { X } } \Big [ \phi _ { X } ( X ) \phi _ { X } ( X ) ^ { * } \Big ] .
$$

The normalized-kernel assumption gives

$$
\mathrm { T r } ( C _ { X Y } ( P ) ) = 1 , \qquad \mathrm { T r } ( C _ { X } ( P _ { X } ) ) = 1 .
$$

Thus, these covariance operators are density operators whenever the relevant entropies are finite. The conditional von Neumann entropy used in the paper is

$$
{ \mathcal { H } } _ { \operatorname { v N } } ( Y \mid X ; P ) = H _ { \operatorname { v N } } ( C _ { X Y } ( P ) ) - H _ { \operatorname { v N } } ( C _ { X } ( P _ { X } ) ) .
$$

For empirical Gram matrices and finite-dimensional sketched features, all operators below are finite-dimensional. For general kernels, the same argument applies under the standard trace-class and finite-entropy conditions for the corresponding density operators.

Proof. Let

$$
\rho _ { X Y } ( P ) : = C _ { X Y } ( P ) .
$$

We first identify the correct marginal operator. For a rank-one tensor term,

$$
( \phi _ { X } ( x ) \otimes \phi _ { Y } ( y ) ) ( \phi _ { X } ( x ) \otimes \phi _ { Y } ( y ) ) ^ { * } ,
$$

taking the partial trace over $\mathcal { H } _ { Y }$ gives

$$
\mathrm { T r } \Big [ \big ( \phi _ { X } ( x ) \otimes \phi _ { Y } ( y ) \big ) \big ( \phi _ { X } ( x ) \otimes \phi _ { Y } ( y ) \big ) ^ { * } \Big ] = \| \phi _ { Y } ( y ) \| _ { \mathcal { H } _ { Y } } ^ { 2 } \phi _ { X } ( x ) \phi _ { X } ( x ) ^ { * } .
$$

Since $k _ { Y } ( y , y ) = 1$ , this equals $\phi _ { X } ( x ) \phi _ { X } ( x ) ^ { * }$ . Taking expectation over $( X , Y ) \sim P$ yields

$$
\mathrm { T r } _ { Y } \rho _ { X Y } ( P ) = C _ { X } ( P _ { X } ) .
$$

Therefore, we have

$$
\mathcal { H } _ { \mathrm { v N } } ( Y \mid X ; P ) = H _ { \mathrm { v N } } ( \rho _ { X Y } ( P ) ) - H _ { \mathrm { v N } } ( \mathrm { T r } _ { Y } \rho _ { X Y } ( P ) ) .
$$

Now take two joint distributions $P _ { X Y } ^ { ( 0 ) }$ and $P _ { X Y } ^ { ( 1 ) }$ , and define

$$
P _ { X Y } ^ { \lambda } = ( 1 - \lambda ) P _ { X Y } ^ { ( 0 ) } + \lambda P _ { X Y } ^ { ( 1 ) } , \qquad \lambda \in [ 0 , 1 ] .
$$

The covariance map is afine, so

$$
\rho _ { X Y } ( P ^ { \lambda } ) = ( 1 - \lambda ) \rho _ { X Y } ( P ^ { ( 0 ) } ) + \lambda \rho _ { X Y } ( P ^ { ( 1 ) } ) .
$$

We set the following:

$$
\rho _ { X Y } ^ { ( 0 ) } : = \rho _ { X Y } ( P ^ { ( 0 ) } ) , \qquad \rho _ { X Y } ^ { ( 1 ) } : = \rho _ { X Y } ( P ^ { ( 1 ) } ) , \qquad \rho _ { X Y } ^ { \lambda } : = ( 1 - \lambda ) \rho _ { X Y } ^ { ( 0 ) } + \lambda \rho _ { X Y } ^ { ( 1 ) } .
$$

We consider an auxiliary two-dimensional classical register R with orthonormal basis vectors $e _ { 0 } , e _ { 1 }$ , and define the block-diagonal density operator

$$
\Omega _ { R X Y } = ( 1 - \lambda ) ( e _ { 0 } e _ { 0 } ^ { * } ) \otimes \rho _ { X Y } ^ { ( 0 ) } + \lambda ( e _ { 1 } e _ { 1 } ^ { * } ) \otimes \rho _ { X Y } ^ { ( 1 ) } .
$$

Its XY marginal is $\Omega _ { X Y } = \rho _ { X Y } ^ { \lambda }$ . Hence,

$$
H _ { \mathrm { v N } } ( Y \mid X ) _ { \Omega } = H _ { \mathrm { v N } } ( \rho _ { X Y } ^ { \lambda } ) - H _ { \mathrm { v N } } ( \mathrm { T r } _ { Y } \rho _ { X Y } ^ { \lambda } ) = \mathcal { H } _ { \mathrm { v N } } ( Y \mid X ; P ^ { \lambda } ) .
$$

We next compute the conditional entropy after additionally conditioning on the classical register R. Since $\Omega _ { R X Y }$ is block diagonal,

$$
H _ { \mathrm { v N } } ( \Omega _ { R X Y } ) = h ( \lambda ) + ( 1 - \lambda ) H _ { \mathrm { v N } } ( \rho _ { X Y } ^ { ( 0 ) } ) + \lambda H _ { \mathrm { v N } } ( \rho _ { X Y } ^ { ( 1 ) } ) ,
$$

where

$$
h ( \lambda ) = - ( 1 - \lambda ) \log ( 1 - \lambda ) - \lambda \log \lambda .
$$

Similarly,

$$
\Omega _ { R X } = ( 1 - \lambda ) ( e _ { 0 } e _ { 0 } ^ { * } ) \otimes \mathrm { T r } _ { Y } \rho _ { X Y } ^ { ( 0 ) } + \lambda ( e _ { 1 } e _ { 1 } ^ { * } ) \otimes \mathrm { T r } _ { Y } \rho _ { X Y } ^ { ( 1 ) } ,
$$

and therefore

$$
H _ { \mathrm { v N } } ( \Omega _ { R X } ) = h ( \lambda ) + ( 1 - \lambda ) H _ { \mathrm { v N } } ( \mathrm { T r } _ { Y } \rho _ { X Y } ^ { ( 0 ) } ) + \lambda H _ { \mathrm { v N } } ( \mathrm { T r } _ { Y } \rho _ { X Y } ^ { ( 1 ) } ) .
$$

Subtracting the last two equations results in

$$
H _ { \mathrm { v N } } \big ( Y \mid X R \big ) _ { \Omega } = ( 1 - \lambda ) \Big [ H _ { \mathrm { v N } } \big ( \rho _ { X Y } ^ { ( 0 ) } \big ) - H _ { \mathrm { v N } } \big ( \mathrm { T r } _ { Y } \rho _ { X Y } ^ { ( 0 ) } \big ) \Big ] + \lambda \Big [ H _ { \mathrm { v N } } \big ( \rho _ { X Y } ^ { ( 1 ) } \big ) - H _ { \mathrm { v N } } \big ( \mathrm { T r } _ { Y } \rho _ { X Y } ^ { ( 1 ) } \big ) \Big ] .
$$

Equivalently,

$$
H _ { \mathrm { v N } } ( Y \mid X R ) _ { \Omega } = ( 1 - \lambda ) { \mathcal { H } } _ { \mathrm { v N } } ( Y \mid X ; P ^ { ( 0 ) } ) + \lambda { \mathcal { H } } _ { \mathrm { v N } } ( Y \mid X ; P ^ { ( 1 ) } ) .
$$

Using the strong subadditivity of von Neumann entropy [63] implies the monotonicity of quantum conditional entropy under additional conditioning:

$$
H _ { \mathrm { v N } } ( Y | X ) _ { \Omega } \geq H _ { \mathrm { v N } } ( Y | X R ) _ { \Omega } .
$$

Combining the identities above gives

$$
\mathcal { H } _ { \mathrm { v N } } ( \boldsymbol { Y } \mid \boldsymbol { X } ; P ^ { \lambda } ) \geq ( 1 - \lambda ) \mathcal { H } _ { \mathrm { v N } } ( \boldsymbol { Y } \mid \boldsymbol { X } ; P ^ { ( 0 ) } ) + \lambda \mathcal { H } _ { \mathrm { v N } } ( \boldsymbol { Y } \mid \boldsymbol { X } ; P ^ { ( 1 ) } ) ,
$$

which finishes the concavity proof for the kernel-induced matrix-based conditional entropy.

## C.3 Additional Theoretical Results on Finite-Sample Conditional Entropy

Corollary 1 (Finite-sample underestimation and monotonicity for Shannon conditional entropy). Let ${ \widehat { P } } _ { n }$ be the empirical joint distribution of n i.i.d. samples from a distribution $P _ { X Y }$ on finite alphabets. Then

$$
\mathbb { E } [ H ( Y | X ; { \widehat { P } } _ { n } ) ] \leq H ( Y | X ; P ) .
$$

Moreover, for all integers $m \leq n$

$$
\mathbb { E } [ H ( Y | X ; { \widehat { P } } _ { m } ) ] \leq \mathbb { E } [ H ( Y | X ; { \widehat { P } } _ { n } ) ] .
$$

Proof. The first inequality follows from Jensen’s inequality and Proposition 1:

$$
\mathbb { E } [ H ( Y | X ; { \widehat { P } } _ { n } ) ] \leq H ( Y | X ; \mathbb { E } [ { \widehat { P } } _ { n } ] ) = H ( Y | X ; P ) .
$$

For monotonicity, let $Z _ { 1 } , \ldots , Z _ { n }$ be the paired samples, where $Z _ { i } = ( X _ { i } , Y _ { i } )$ . Conditional on $Z _ { 1 } , \ldots , Z _ { n }$ draw a subset $S \subseteq \{ 1 , \dots , n \}$ of size m uniformly without replacement, and let $\widehat { P } _ { S }$ be the empirical distribution of the selected samples. Then

$$
\mathbb { E } [ \widehat { P } _ { S } | Z _ { 1 } , \ldots , Z _ { n } ] = \widehat { P } _ { n } .
$$

The concavity results in the following:

$$
H ( Y \mid X ; \widehat { P } _ { n } ) = H \Big ( Y \mid X ; \mathbb { E } [ \widehat { P } _ { S } \mid Z _ { 1 } , \ldots , Z _ { n } ] \Big ) \geq \mathbb { E } \Big [ H ( Y \mid X ; \widehat { P } _ { S } ) \mid Z _ { 1 } , \ldots , Z _ { n } \Big ] .
$$

Taking expectations gives

$$
\mathbb { E } [ H ( Y | X ; { \widehat { P } } _ { n } ) ] \geq \mathbb { E } [ H ( Y | X ; { \widehat { P } } _ { S } ) ] .
$$

The unconditional law of $\widehat { P } _ { S }$ is the same as the empirical distribution of m i.i.d. samples from $P _ { X Y }$ . Therefore

$$
\mathbb { E } [ H ( Y | X ; { \widehat { P } } _ { n } ) ] \geq \mathbb { E } [ H ( Y | X ; { \widehat { P } } _ { m } ) ] .
$$

Theorem 3 (First-order bias of empirical Shannon conditional entropy). Assume X and Y take values in finite alphabets, and define

$$
{ \cal S } _ { X Y } = \{ ( x , y ) : { \cal P } _ { X Y } ( x , y ) > 0 \} , { \cal S } _ { X } = \{ x : { \cal P } _ { X } ( x ) > 0 \} .
$$

Using natural logarithms,

$$
H ( Y \mid X ; P ) - \mathbb { E } [ H ( Y \mid X ; { \widehat { P } } _ { n } ) ] = { \frac { | S _ { X Y } | - | S _ { X } | } { 2 n } } + O { \Big ( } { \frac { 1 } { n ^ { 2 } } } { \Big ) } .
$$

Proof. For a discrete distribution P with support size s and strictly positive probabilities on its support, the Miller–Basharin expansion gives

$$
H ( P ) - \mathbb { E } [ H ( { \widehat { P } } _ { n } ) ] = { \frac { s - 1 } { 2 n } } + O { \Big ( } { \frac { 1 } { n ^ { 2 } } } { \Big ) } .
$$

Applying this expansion to $P _ { X Y }$ gives

$$
H ( X , Y ; P ) - \mathbb { E } [ H ( X , Y ; { \widehat { P } } _ { n } ) ] = { \frac { | S _ { X Y } | - 1 } { 2 n } } + O { \Big ( } { \frac { 1 } { n ^ { 2 } } } { \Big ) } .
$$

Applying the same expansion to the marginal $P _ { X }$ gives

$$
H ( X ; P _ { X } ) - \mathbb { E } [ H ( X ; \widehat { P } _ { n , X } ) ] = \frac { | \mathcal { S } _ { X } | - 1 } { 2 n } + O \Big ( \frac { 1 } { n ^ { 2 } } \Big ) .
$$

Since

$$
H ( Y | X ; P ) = H ( X , Y ; P ) - H ( X ; P _ { X } )
$$

and

$$
H ( Y \mid X ; \widehat { P } _ { n } ) = H ( X , Y ; \widehat { P } _ { n } ) - H ( X ; \widehat { P } _ { n , X } ) ,
$$

subtracting the marginal expansion from the joint expansion yields

$$
H ( Y \mid X ; P ) - \mathbb { E } [ H ( Y \mid X ; { \widehat { P } } _ { n } ) ] = { \frac { | { \mathcal { S } } _ { X Y } | - | { \mathcal { S } } _ { X } | } { 2 n } } + O { \Big ( } { \frac { 1 } { n ^ { 2 } } } { \Big ) } .
$$

Corollary 2 (Finite-sample underestimation and monotonicity for conditional von Neumann entropy). Let ${ \widehat { P } } _ { n }$ be the empirical distribution of n i.i.d. paired samples from $P _ { X Y }$ . Assume the relevant conditional von Neumann entropies are finite. Then

$$
\mathbb { E } [ \mathcal { H } _ { \mathrm { v N } } ( Y | X ; \widehat { P } _ { n } ) ] \leq \mathcal { H } _ { \mathrm { v N } } ( Y | X ; P ) .
$$

Moreover, for all integers $m \leq n$

$$
\mathbb { E } [ \mathcal { H } _ { \mathrm { v N } } ( Y | X ; \widehat { P } _ { m } ) ] \leq \mathbb { E } [ \mathcal { H } _ { \mathrm { v N } } ( Y | X ; \widehat { P } _ { n } ) ] .
$$

Proof. The first inequality follows from Jensen’s inequality and Proposition 2:

$$
\mathbb { E } [ \mathcal { H } _ { \mathrm { v N } } ( Y \mid X ; \widehat { P } _ { n } ) ] \leq \mathcal { H } _ { \mathrm { v N } } ( Y \mid X ; \mathbb { E } [ \widehat { P } _ { n } ] ) = \mathcal { H } _ { \mathrm { v N } } ( Y \mid X ; P ) .
$$

For monotonicity, condition on n paired samples $Z _ { 1 } , \ldots , Z _ { n }$ and draw a subset S of size m uniformly without replacement. As above,

$$
\mathbb { E } [ \widehat { P } _ { S } | Z _ { 1 } , \ldots , Z _ { n } ] = \widehat { P } _ { n } .
$$

By concavity,

$$
\mathcal { H } _ { \mathrm { v N } } ( Y | X ; \widehat { P } _ { n } ) \geq \mathbb { E } [ \mathcal { H } _ { \mathrm { v N } } ( Y | X ; \widehat { P } _ { S } ) | Z _ { 1 } , \dots , Z _ { n } ] .
$$

Taking expectations and using the fact that $\widehat { P } _ { S }$ has the same law as an empirical distribution formed from m i.i.d. samples proves the result. □

## C.4 Additional Theoretical Results on Lifted Bregman Divergences

Fix an input marginal $P _ { X }$ . The map

$$
R _ { Y \mid X } \longmapsto P _ { X } R _ { Y \mid X }
$$

embeds a conditional law into a joint law whose input marginal is fixed. Given a Bregman divergence $D _ { \Phi }$ on a convex domain of joint laws, define

$$
D _ { \Phi \mid P _ { X } } ( R _ { Y \mid X } , Q _ { Y \mid X } ) = D _ { \Phi } ( P _ { X } R _ { Y \mid X } , P _ { X } Q _ { Y \mid X } ) .
$$

If $D _ { \Phi }$ is the KL divergence, then

$$
D _ { \Phi \mid P _ { X } } \left( R _ { Y \mid X } , Q _ { Y \mid X } \right) = \operatorname { \mathbb { E } } _ { X \sim P _ { X } } \left[ \operatorname { K L } \left( R _ { Y \mid X } ( \cdot \mid X ) \parallel Q _ { Y \mid X } ( \cdot \mid X ) \right) \right] ,
$$

whenever $P _ { X } R _ { Y \mid X }$ is absolutely continuous with respect to $P _ { X } Q _ { Y \mid X }$

If $\psi ( x , y )$ is a feature map on joint pairs and

$$
\mu _ { P _ { X } R } = \mathbb { E } _ { X \sim P _ { X } , Y \sim R ( \cdot \mid X ) } [ \psi ( X , Y ) ]
$$

is the kernel mean embedding of $P _ { X } R _ { Y \mid X }$ , then

$$
D _ { \mathrm { M M D } \mid P _ { X } } ( R _ { Y \mid X } , Q _ { Y \mid X } ) = \| \mu _ { P _ { X } R } - \mu _ { P _ { X } Q } \| _ { \mathcal { H } } ^ { 2 } .
$$

This is the Bregman divergence generated by $\Phi ( \mu ) = \| \mu \| _ { \mathcal { H } } ^ { 2 }$ on the Hilbert space of mean embeddings, since

$$
D _ { \Phi } ( \mu , \nu ) = \| \mu - \nu \| _ { \mathcal { H } } ^ { 2 } .
$$

## C.5 Proof of Theorem 1

For $\rho \in \mathbb { R } .$ , we consider the definition:

$$
\mathcal { C } _ { \rho } ( P _ { X } ) = \Big \{ R _ { Y \mid X } : \mathcal { H } _ { \operatorname { v N } } \big ( R _ { Y \mid X } ; P _ { X } \big ) \geq \rho \Big \} .
$$

Lemma 1 (Convexity of conditional entropy superlevel sets). The set $\mathcal { C } _ { \rho } ( P _ { X } )$ is convex.

Proof. Let $R _ { Y \mid X } ^ { ( 0 ) } , R _ { Y \mid X } ^ { ( 1 ) } \in \mathcal { C } _ { \rho } ( P _ { X } )$ and $\lambda \in [ 0 , 1 ]$ . Define

$$
R _ { Y \mid X } ^ { \lambda } = ( 1 - \lambda ) R _ { Y \mid X } ^ { ( 0 ) } + \lambda R _ { Y \mid X } ^ { ( 1 ) } .
$$

Because the input marginal is fixed,

$$
P _ { X } R _ { Y \mid X } ^ { \lambda } = ( 1 - \lambda ) P _ { X } R _ { Y \mid X } ^ { ( 0 ) } + \lambda P _ { X } R _ { Y \mid X } ^ { ( 1 ) } .
$$

By Proposition 2,

$$
\mathcal { H } _ { \mathrm { v N } } \big ( R _ { Y | X } ^ { \lambda } ; P _ { X } \big ) \geq ( 1 - \lambda ) \mathcal { H } _ { \mathrm { v N } } \big ( R _ { Y | X } ^ { ( 0 ) } ; P _ { X } \big ) + \lambda \mathcal { H } _ { \mathrm { v N } } \big ( R _ { Y | X } ^ { ( 1 ) } ; P _ { X } \big ) \geq \rho .
$$

Thus $R _ { Y \mid X } ^ { \lambda } \in { \mathcal { C } } _ { \rho } ( P _ { X } )$

We also use the following lemma on Bregman divergences. Let Ω be a convex domain, let $\Phi : \Omega  \mathbb { R }$ be diferentiable and strictly convex, and define

$$
D _ { \Phi } ( A , B ) = \Phi ( A ) - \Phi ( B ) - \langle \nabla \Phi ( B ) , A - B \rangle .
$$

Lemma 2. Let ${ \mathcal { C } } \subseteq \Omega$ be nonempty and convex. Suppose

$$
\begin{array} { r } { Q ^ { \star } \in \underset { R \in \mathcal { C } } { \mathrm { a r g m i n } } D _ { \Phi } ( R , Q ) . } \end{array}
$$

Then, for every $P \in { \mathcal { C } }$

$$
D _ { \Phi } ( P , Q ^ { \star } ) + D _ { \Phi } ( Q ^ { \star } , Q ) \leq D _ { \Phi } ( P , Q ) .
$$

Proof. The first-order optimality condition for the projection gives

$$
\langle \nabla \Phi ( Q ^ { \star } ) - \nabla \Phi ( Q ) , P - Q ^ { \star } \rangle \geq 0 , \qquad P \in { \mathcal { C } } .
$$

The three-point identity for Bregman divergences gives

$$
D _ { \Phi } ( P , Q ) = D _ { \Phi } ( P , Q ^ { \star } ) + D _ { \Phi } ( Q ^ { \star } , Q ) + \langle \nabla \Phi ( Q ^ { \star } ) - \nabla \Phi ( Q ) , P - Q ^ { \star } \rangle .
$$

Combining the two displays yields the inequality.

Proof of Theorem 1. The choice

$$
\rho = \mathcal { H } _ { \mathrm { v N } } \big ( P _ { Y | X } ; P _ { X } \big )
$$

implies $P _ { Y \mid X } \in { \mathcal { C } } _ { \rho } ( P _ { X } )$ . By Lemma 1, $\mathcal { C } _ { \rho } ( P _ { X } )$ is convex. Applying Lemma 2 to the lifted joint laws $P _ { X } P _ { Y \mid X } , \dot { P } _ { X } Q _ { Y \mid X } , P _ { X } Q _ { Y \mid X } ^ { \star }$ over the convex set $\{ P _ { X } R _ { Y \mid X } : R _ { Y \mid X } \in \mathcal { C } _ { \rho } ( P _ { X } ) \}$ results in

$$
D _ { \Phi \mid P _ { X } } \bigl ( P _ { Y \mid X } , Q _ { Y \mid X } ^ { \star } \bigr ) + D _ { \Phi \mid P _ { X } } \bigl ( Q _ { Y \mid X } ^ { \star } , Q _ { Y \mid X } \bigr ) \leq D _ { \Phi \mid P _ { X } } \bigl ( P _ { Y \mid X } , Q _ { Y \mid X } \bigr ) .
$$

Therefore, the proof is complete.

## C.6 Additional Theoretical Results on Finite-Support Conditional MMD Projection

Let $x _ { 1 } , \ldots , x _ { N }$ be input prompts, and let $y _ { i , 1 } , \ldots , y _ { i , m }$ be generated candidates for $x _ { i }$ . Let $M = N r$ and index pairs by $a = ( i , j )$

$$
\begin{array} { r } { s _ { a } = ( x _ { i } , y _ { i , j } ) . } \end{array}
$$

Let $q \in \mathbb { R } ^ { M }$ be a distribution over the generated pairs and let $u \in \mathbb { R } ^ { M }$ be the uniform baseline,

$$
u _ { i , j } = \frac { 1 } { N m } .
$$

The conditional block constraints are

$$
q _ { i , j } \geq 0 , \qquad \sum _ { j = 1 } ^ { m } q _ { i , j } = \frac { 1 } { N } , \qquad i = 1 , \dots , N .
$$

Equivalently, $q _ { i , j } = p _ { i , j } / N$ with $p _ { i } \in \Delta _ { m }$

Let

$$
k ( ( x , y ) , ( x ^ { \prime } , y ^ { \prime } ) ) = k _ { X } ( x , x ^ { \prime } ) k _ { Y } ( y , y ^ { \prime } )
$$

be the product kernel, and let $K \in \mathbb { R } ^ { M \times M }$ be the Gram matrix over the generated pairs. Then the squared MMD between the reweighted distribution q and the uniform baseline u is

$$
\Big \| \sum _ { a } ( q _ { a } - u _ { a } ) \psi ( s _ { a } ) \Big \| _ { \mathcal { H } } ^ { 2 } = ( q - u ) ^ { \top } K ( q - u ) .
$$

Thus, the objective in equation 1 is the finite-support instance of the lifted conditional MMD with empirical input marginal.

The weighted joint covariance operator is

$$
C _ { X Y } ( q ) = \sum _ { a } q _ { a } \psi ( s _ { a } ) \psi ( s _ { a } ) ^ { * } .
$$

Because the product kernel is normalized and $\textstyle \sum _ { a } q _ { a } = 1$ , this operator has trace one. The empirical input covariance is

$$
C _ { X } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \phi _ { X } ( x _ { i } ) \phi _ { X } ( x _ { i } ) ^ { * } .
$$

The block constraints make $C _ { X }$ independent of $q ,$ so

$$
{ \mathcal { H } } _ { \mathrm { v N } } ( q ) = H _ { \mathrm { v N } } ( C _ { X Y } ( q ) ) - H _ { \mathrm { v N } } ( C _ { X } ) .
$$

The nonzero eigenvalues of $C _ { X Y } ( q )$ coincide with the nonzero eigenvalues of the weighted Gram matrix

$$
K _ { q } = \operatorname { d i a g } ( { \sqrt { q } } ) K \operatorname { d i a g } ( { \sqrt { q } } ) .
$$

Indeed, writing $C _ { X Y } ( q ) = A A ^ { * }$ with columns ${ \sqrt { q _ { a } } } \psi ( s _ { a } )$ gives $K _ { q } = A ^ { * } A$ , and $A A ^ { * }$ and $A ^ { * } A$ have the same nonzero spectrum.

## C.7 Proof of Proposition 3

Proof. The objective $( q - u ) ^ { \top } K ( q - u )$ is convex because K is positive semidefinite. The nonnegativity and block-marginal constraints are linear. The map $q \mapsto C _ { X Y } ( q )$ is afine, and $H _ { \mathrm { v N } }$ is concave on positive trace-one operators. Since $C _ { X }$ is fixed over the feasible set, $q \mapsto \mathcal { H } _ { \mathrm { v N } } ( q )$ is concave. Hence, the entropy superlevel constraint is convex, and the program is convex. □

## C.8 Proof of Proposition 4

Proof. The squared norm term is convex because it is the squared norm of an afine function of $p .$ The map $p \mapsto \widetilde { C } ( p )$ is afine, and von Neumann entropy is concave on trace-one positive semidefinite matrices. Therefore $p \mapsto - H _ { \mathrm { v N } } ( \widetilde { C } ( p ) )$ is convex. The constraints $p _ { i } \in \Delta _ { m }$ are linear, so the problem is convex. □

## C.9 Proof of Theorem 2

CEP-BEG uses the product-simplex negative-entropy mirror map

$$
\Psi ( p ) = \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { m } p _ { i , j } \log p _ { i , j } .
$$

The associated Bregman divergence is

$$
D _ { \Psi } ( p , p ^ { \prime } ) = \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { m } p _ { i , j } \log \frac { p _ { i , j } } { p _ { i , j } ^ { \prime } } .
$$

The mirror step is

$$
p ^ { ( t + 1 ) } = \operatorname * { a r g m i n } _ { p _ { 1 } , \ldots , p _ { N } \in \Delta _ { m } } \Big \{ \langle g ^ { ( t ) } , p \rangle + \frac { 1 } { \eta } D _ { \Psi } ( p , p ^ { ( t ) } ) \Big \} ,
$$

which yields the block exponentiated-gradient update in Algorithm 1.

Proof. For each block i, the negative-entropy mirror descent inequality gives

$$
\langle g _ { i } ^ { ( t ) } , p _ { i } ^ { ( t ) } - p _ { i } \rangle \leq \frac { D _ { \mathrm { K L } } ( p _ { i } , p _ { i } ^ { ( t ) } ) - D _ { \mathrm { K L } } ( p _ { i } , p _ { i } ^ { ( t + 1 ) } ) } { \eta } + \frac { \eta } { 2 } \| g _ { i } ^ { ( t ) } \| _ { \infty } ^ { 2 } .
$$

Summing over $i = 1 , \ldots , N$ yields

$$
\langle g ^ { ( t ) } , p ^ { ( t ) } - p \rangle \leq \frac { D _ { \Psi } ( p , p ^ { ( t ) } ) - D _ { \Psi } ( p , p ^ { ( t + 1 ) } ) } { \eta } + \frac { \eta } { 2 } \sum _ { i = 1 } ^ { N } \Vert g _ { i } ^ { ( t ) } \Vert _ { \infty } ^ { 2 } .
$$

Taking $p = p ^ { \star }$ and using convexity of F,

$$
F ( p ^ { ( t ) } ) - F ( p ^ { \star } ) \leq \langle g ^ { ( t ) } , p ^ { ( t ) } - p ^ { \star } \rangle .
$$

Summing over $t = 0 , \ldots , T - 1$ gives

$$
\sum _ { t = 0 } ^ { T - 1 } \left( F ( p ^ { ( t ) } ) - F ( p ^ { \star } ) \right) \leq \frac { D _ { \Psi } ( p ^ { \star } , p ^ { ( 0 ) } ) } { \eta } + \frac { \eta T G ^ { 2 } } { 2 } .
$$

Since $p _ { i , j } ^ { ( 0 ) } = 1 / m$ and $p _ { i } ^ { \star } \in \Delta _ { m }$ ，

$$
D _ { \Psi } ( p ^ { \star } , p ^ { ( 0 ) } ) = \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { m } p _ { i , j } ^ { \star } \log ( m p _ { i , j } ^ { \star } ) \leq N \log m .
$$

Dividing by $T$ and using Jensen’s inequality for the convex function $F _ { \mathrm { { ; } } }$

$$
F ( \bar { p } ^ { ( T ) } ) \leq \frac { 1 } { T } \sum _ { t = 0 } ^ { T - 1 } F ( p ^ { ( t ) } ) ,
$$

we obtain

$$
F ( \bar { p } ^ { ( T ) } ) - F ( p ^ { \star } ) \leq \frac { N \log m } { \eta T } + \frac { \eta G ^ { 2 } } { 2 } .
$$

The stated choice of $\eta = \sqrt { 2 N \log m } / ( G \sqrt { T } )$ minimizes this upper bound which becomes the following and finishes the proof:

$$
F ( \bar { p } ^ { ( T ) } ) - F ( p ^ { \star } ) \leq G \sqrt { \frac { 2 N \log m } { T } } .
$$

The proof is hence complete.

For the regularized entropy implementation, a concrete gradient bound follows under the assumption of a spectral floor. If $\| \tilde { z } _ { i , j } \| _ { 2 } = 1$ and the eigenvalues of the matrices used inside the logarithm are bounded below by $\tau > 0$ , then

$$
\| g _ { i } ^ { ( t ) } \| _ { \infty } \leq \frac { 4 + \lambda ( 1 + | \log \tau | ) } { N } \quad \mathrm { f o r ~ a l l ~ } i , t .
$$

Consequently, one may take

$$
G = \frac { 4 + \lambda ( 1 + | \log \tau | ) } { \sqrt { N } } .
$$

The bound follows from $\| \widetilde { r } ( p ) \| _ { 2 } \leq 2$ and

$$
\| \log \widetilde { C } ( p ) + I \| _ { \mathrm { o p } } \leq 1 + | \log \tau |
$$

on the spectral interval $[ \tau , 1 ]$

## Appendix D Implementation Details

## D.1 LLM Data Construction and Sample Generation

We construct paired prefix–continuation samples from the corresponding open training corpora of the evaluated language models. For OLMo, we use the Dolma corpus; for Pythia and GPT-Neo, we use The Pile. Each example is represented as a pair (X, Y), where X is a short prefix used as the conditioning input and Y is the following continuation. The same prefix set is used for both the training-data reference and the model-generated continuations, so the input marginal is matched across groups.

For OLMo, we use the allenai/OLMo-1B-hf checkpoint and its associated tokenizer. The tokenizer padding token is set to the end-of-sequence token when no padding token is provided, and left padding is used for batched autoregressive generation. The model is loaded with automatic device mapping and half precision. To construct the training reference for OLMo, we use the Hugging Face Dolma dataset. We load the downloaded Dolma files with the Hugging Face datasets library and process each document into paragraph-level candidates. We split documents by blank lines, normalize whitespace, and keep only paragraphs with at least 500 characters and at least 3 sentence-ending marks. From each document, we uniformly sample the candidate paragraphs. We then tokenize each candidate paragraph and keep the first paragraph that contains at least the required number of tokens. Unless otherwise specified, we use a prefix length of 3 tokens and a continuation length of 2 tokens. The prefix tokens are used as the condition X, and the immediately following tokens are used as the training continuation $Y _ { \mathrm { t r a i n } }$ . To examine the efect of token-length choices, we additionally conduct an ablation over all combinations of prefix lengths in {3, 4, 5} and continuation lengths in {2, 3, 4}. We also conduct a sample-size ablation by drawing subsamples of diferent sizes under multiple random seeds and reporting the resulting variation across runs. Documents without a valid paragraph of suficient length are skipped.

For each prefix, we generate model continuations using three decoding strategies. Greedy decoding selects the most likely token at each step. Nucleus sampling samples from the smallest token set whose cumulative probability mass exceeds $p = 0 . 9$ . Ancestral sampling samples from the full next-token distribution, implemented by setting top\_p=1.0. All strategies use the same maximum number of newly generated tokens as the training continuation length. We store the generated data in JSON format with fields for the example id, prefix, training continuation, greedy decoding output, nucleus sampling output, and ancestral sampling output. In the main experiments, we sample up to 30K prefix–continuation pairs and evaluate conditional diversity at diferent sample sizes using subsets of these generated records. For Pythia and GPT-Neo, we follow the same construction using The Pile as the training-data reference. We use the Hugging Face checkpoints EleutherAI/pythia-1b and EleutherAI/gpt-neo-1.3B, together with their corresponding tokenizers. For each model, we tokenize documents from The Pile, extract matched prefix–continuation pairs, and generate continuations from the same prefix set under greedy decoding, nucleus sampling with the same $p = 0 . 9$ and ancestral sampling. This ensures that the comparison between training and generated groups is always performed under a shared empirical input marginal.

Long-sequence construction. For the long-sequence experiments in Section 6.3, we extend the prefix length to {8, 16, 32} tokens and the continuation length to {16, 32, 64} tokens. To represent a long sequence while preserving local information, we split each sequence into 2-token segments, compute the Qwen3- Embedding of each segment independently, and concatenate the resulting embeddings into a single feature vector. The same chunking construction has been widely used in the long-text literature [64, 65, 66, 67, 68]. The same prefix set is used for the training-data reference and the model-generated continuations. We additionally perform a chunk-size ablation over chunk sizes {1, 2, 4} in Appendix E.6. The entropy computation depends primarily on the number of samples and the representation dimension rather than directly on the number of tokens; longer sequences mainly increase the cost of constructing their representations, after which the same kernel-based estimator and optimization procedure apply.

In Table 2 we observe two consistent trends in the gap. First, for a fixed prefix length, the gap generally grows with the continuation length, because a longer continuation leaves more room for valid variation that the model under-covers. Second, for a fixed continuation length, the gap generally shrinks as the prefix length increases, because stronger conditioning reduces the range of plausible continuations. Nevertheless, no configuration eliminates the gap.

## D.2 Image Data Construction and Sample Generation

We construct paired condition–image samples for two conditional image-generation settings: classconditioned ImageNet generation and text-conditioned MS-COCO generation. Each example is represented as a pair (X, Y), where X is the conditioning variable and Y is the corresponding image. For ImageNet, X is the class label and Y is the image. For MS-COCO, X is the text caption and Y is the image. In both settings, we compare generated images with real-data references under matched conditions.

For the class-conditioned ImageNet experiments, we follow the DGM-Eval benchmark and use the generated image samples provided for several class-conditioned generators, including ADM, LDM, BigGAN, and DiT-XL-2. The real-data reference is constructed from ImageNet training images with their corresponding class labels. For each generated sample, we use its class label as the condition and the generated image as the output. We then form paired samples $( X _ { i } , Y _ { i } )$ for both the real ImageNet reference and each generator. Conditional diversity is evaluated by comparing the real and generated paired distributions under the same class-conditioned protocol.

For the text-conditioned MS-COCO experiments, we use captions sampled from the MS-COCO dataset as conditioning inputs. Given a caption $X _ { i } ,$ , we generate an image $Y _ { i }$ using each text-to-image model. We evaluate U-ViT-S/2 and its corresponding Deep variant, as well as SDXL-Base-1.0 and PixArt-Σ. For the real-data reference, we use the MS-COCO image associated with the sampled caption. This gives paired caption–image samples for the real MS-COCO reference and for each generated model group. The same caption set is used when comparing the reference and generated samples, so the empirical input margina over captions is matched.

For all image-generation experiments, conditional VNE and conditional RKE are computed from paired condition–image samples using the product kernel between condition features and image features. In the ImageNet setting, the condition feature is derived from the class label; in the MS-COCO setting, the condition feature is derived from the caption. Unless otherwise specified, we use the gaussian kernel with bandwidth selection following the literature.

For sample-size analysis, we evaluate conditional diversity over subsets of diferent sizes. In the ImageNet and MS-COCO gap experiments, we report the exponential of conditional VNE over sample sizes ranging from 2.5K to 20K. For each sample size, we draw subsamples under multiple random seeds and report the mean and standard deviation across evaluations. This protocol is applied consistently to the real-data reference and to each generated model group.

## D.3 Evaluation Settings and Hyperparameters

For the LLM diversity-gap experiments, the default text representation is Qwen3-Embedding. We compute embeddings for the prefix and continuation texts separately, construct a product kernel over the paired prefix–continuation samples, and evaluate conditional VNE and conditional RKE on the resulting normalized kernel matrices. Unless otherwise stated, we use the Gaussian kernel with bandwidth selected by the median heuristic on the corresponding feature distances. We also report ablations with T5 embeddings, cosine kernels, and degree-3 polynomial kernels with $\gamma = 1$ in Tables 13, 14, and 15.

For the image experiments, image features are extracted using the default visual encoder DINOv2 and condition text features are extracted by CLIP model. We use the same normalized product-kernel construction for all conditional VNE and conditional RKE evaluations, so diferences in the reported scores reflect the conditional output distributions under matched input conditions.

For entropy-projected reweighting, each prefix is associated with 10 generated candidate continuations. The optimization is performed over a product of simplices, with one simplex for the candidates associated with each prefix, so the empirical prefix marginal is fixed throughout optimization. We initialize the weights uniformly, use the conditional entropy objective described in Section 5, and report the reweighted empirica distribution using the same evaluation pipeline as the unweighted generated samples. For scalability, the joint product features are computed through the sketched representation described in the method section.

For conditional VNE guidance in difusion sampling, we use SDXL on MS-COCO captions and set the guidance scale to $\eta = 0 . 0 3$ . The guidance step is applied during denoising after the standard SDXL update. All guided and unguided comparisons use the same caption set and the same evaluation features, and conditional VNE is reported as mean ± standard deviation over independent subsamples. All experiments were run on 2 RTX-4090 GPUs.

For the MBR decoding experiments in Section 6.7, we follow the candidate-generation protocol of Jinnai et al. [30], using nucleus sampling with $p = 0 . 9$ for a controlled comparison. We evaluate XSum with BARTlarge-XSum, CNN/DailyMail with BART-large-CNN, and CNN/DailyMail with Mistral-7B-Instruct-v0.1 using the CNN/DM prompt from their Appendix G. For each of 50 test inputs, we compare Monte Carlo MBR, model-based MBR with and without length normalization, and the proposed conditional-Vendi-weighted variant (CVS-MBR). All methods operate on the same candidate pool and utility function and difer only in the reference weighting used in the MBR expectation.

The methods compared in Table 5 represent diferent operating points. MBMBR and MBMBR-L use model-probability weights and are mainly designed to improve expected generation quality; accordingly, they achieve higher ROUGE-L and BERTScore, with length normalization mitigating sequence-length bias. In contrast, CVS-MBR explicitly reduces concentration in the reference distribution and therefore achieves the highest conditional VNE and Distinct-2, as well as the lowest Self-BLEU-2, across all three settings. To examine the quality–diversity trade-of, we performed a paired bootstrap over the 50 test inputs with 2000 resamples, comparing CVS-MBR against MBMBR-L. The 95% confidence intervals for the ROUGE-L and BERTScore diferences contain zero in all three settings, indicating no statistically significant degradation in quality while CVS-MBR consistently attains the highest conditional VNE.

## Appendix E Additional Numerical Results

## E.1 Additional LLM Diversity Gap Results

Tables 17 and 18 provide the full LLM diversity-gap results at sample sizes 10K and 20K. Across OLMo, Pythia, and GPT-Neo, the training continuations consistently achieve higher conditional VNE and conditional RKE than model-generated continuations from the same prefixes. The ordering among decoding strategies is also stable: greedy decoding has the largest gap, nucleus sampling reduces the gap, and ancestral sampling is usually the closest to the training reference. For example, at 20K samples, the conditional VNE gap for OLMo is 119.70 under greedy decoding, 51.27 under nucleus sampling, and 41.27 under ancestral sampling.

The lexical diversity metrics show the same qualitative pattern. Distinct-2 and Distinct-3 are substantially lower for greedy decoding than for the training continuations, while stochastic decoding recovers much of the surface-level n-gram diversity. However, even when Distinct scores become relatively close to the training reference, the conditional VNE and conditional RKE gaps remain positive. This indicates that the proposed kernel-based conditional diversity metrics capture conditional output-range diferences that are not fully explained by local n-gram variety.

Comparing 10K and 20K samples further shows that the measured conditional diversity gap becomes more pronounced as the evaluation set grows. The training reference gains conditional VNE faster than the generated groups, which is consistent with the sample-size curves in Figure 1. This behavior suggests that the training data continues to reveal additional valid conditional continuations as more examples are included, whereas the generated samples cover a narrower conditional range.

## E.2 Additional Image Diversity Gap Results

Figure 2 and Table 20 report the ImageNet class-conditioned results. The ImageNet reference has the highest conditional VNE at every sample size. Among the evaluated generators, DiT-XL-2 and LDM are closest to the reference, while ADM and BigGAN exhibit much larger gaps. At 20K samples, ImageNet reaches 54.15, compared with 49.95 for DiT-XL-2, 47.86 for LDM, 22.47 for ADM, and 17.04 for BigGAN. Thus, the strongest difusion and transformer-based generators approach the reference more closely, but they still do not fully match the conditional diversity of real images under the same class labels.

Figure 3 and Table 21 show the text-conditioned MS-COCO results. The MS-COCO reference again achieves the highest conditional VNE across all sample sizes. SDXL is the strongest among the evaluated generated groups in this experiment, followed closely by U-ViT Deep S/2 and U-ViT S/2, while PixArt-Σ has the largest gap. At 20K samples, the reference reaches 40.52, whereas SDXL reaches 30.25, U-ViT Deep S/2 reaches 29.94, U-ViT S/2 reaches 29.50, and PixArt-Σ reaches 25.25.

The image results mirror the LLM results: the real-data reference grows faster with sample size than the generated groups, so the diversity gap becomes clearer as more samples are evaluated. Because the comparisons are made under matched class labels or captions, the gap reflects diferences in conditional output variability rather than a mismatch in the empirical input marginal.

## E.3 Additional Reweighting Results

The entropy-projected reweighting results in Table 3 show that conditional diversity can be increased post hoc by redistributing probability mass over already generated candidates. The reweighting improves conditional VNE for all three LLMs at both evaluated sample sizes. At 20K samples, conditional VNE increases from 366.35 to 376.77 for OLMo, from 371.72 to 382.74 for Pythia, and from 340.90 to 353.68 for GPT-Neo.

The efect is especially strong for conditional RKE. At 20K samples, RKE increases from 241.14 to 284.92 for OLMo, from 236.24 to 285.05 for Pythia, and from 246.13 to 296.02 for GPT-Neo. These gains are obtained without retraining the model and without adding new generated continuations. Since the projection preserves the total mass assigned to each prefix, the improvement comes from changing the conditional distribution over available continuations rather than altering the input distribution.

These results support the interpretation that generated candidate sets often contain underweighted alternatives. The projection can recover part of this latent conditional diversity by moving mass toward candidates that increase the joint condition–output entropy while remaining within the fixed candidate support.

## E.4 Additional Conditional VNE Guidance Results

We further evaluate whether the conditional VNE score can be used not only as an evaluation metric, but also as a sampling-time guidance signal for conditional generation. The goal is to encourage higher conditional diversity during sampling while keeping the text condition fixed.

Let $c ^ { ( n ) }$ denote the caption for the n-th generated image, and let $z _ { t } ^ { ( n ) }$ be its latent variable at difusion step t. Given previously generated caption-image pairs $\{ ( c ^ { ( i ) } , z ^ { ( i ) } ) \} _ { i = 1 } ^ { n - 1 }$ and the current pair $( c ^ { ( n ) } , z _ { t } ^ { ( n ) } )$ , we construct a caption kernel $K _ { C }$ and a latent-image kernel $K _ { Z }$ . The conditional VNE guidance score is

$$
\widehat { H } _ { \mathrm { v N } } ( Z \mid C ) = H _ { \mathrm { v N } } \left( \frac { 1 } { n } ( K _ { C } \odot K _ { Z } ) \right) - H _ { \mathrm { v N } } \left( \frac { 1 } { n } K _ { C } \right) ,
$$

where $\odot$ denotes the Hadamard product. Since the caption kernel $K _ { C }$ is fixed during the latent update, the guidance acts through the gradient of the joint conditional term with respect to the current latent variable. After the standard SDXL denoising update, we apply an additional guidance step

$$
z _ { t - 1 } ^ { ( n ) } \gets z _ { t - 1 } ^ { ( n ) } + \eta \nabla _ { z _ { t - 1 } ^ { ( n ) } } \widehat H _ { \mathrm { v N } } ( Z \mid C ) ,
$$

where η is the guidance scale. For the experiment, we use captions from the MS-COCO dataset and generate 30K images with ${ \mathrm { S D X L } } ,$ both with and without the proposed guidance, and the guidance scale we used is 0.03. We then evaluate conditional VNE at sample sizes from 10K to 20K reported in Table 4. SDXL with conditional VNE guidance consistently obtains higher conditional VNE than SDXL without guidance at every evaluated sample size. For example, at 20K samples, the conditional VNE increases from 30.25 without guidance to 32.27 with guidance. The guided samples still remain below the reference MS-COCO data, which reaches 40.52 at the same sample size, but the guidance reduces the measured gap between generated and reference samples.

## E.5 Ablation Study Analysis

Tables 13, 14, and 15 examine robustness to embedding and kernel choices. The absolute scale of conditional VNE and conditional RKE changes substantially across feature maps and kernels, as expected, but the qualitative conclusion is unchanged: the training reference has the highest conditional diversity, greedy decoding has the largest gap, and stochastic decoding reduces but does not eliminate the gap. This consistency indicates that the observed conditional diversity gap is not an artifact of a single embedding model or kernel family.

Table 13 repeats the LLM analysis with T5 embeddings. Even under this alternative text representation, all generated groups remain below the corresponding training reference. At 20K samples, the conditional VNE gaps under ancestral sampling are 10.11 for OLMo, 6.94 for Pythia, and 5.63 for GPT-Neo, while greedy decoding has much larger gaps. Tables 14 and 15 show the same ranking under degree-3 polynomial and cosine kernels, respectively, although the numerical scale is compressed relative to the default Gaussian-kernel evaluation.

Table 16 studies the efect of increasing the prefix length to 4 tokens with 2 continuation tokens. The training reference still has higher conditional VNE and conditional RKE than all generated groups for all three models. The gaps are generally smaller than in the default 3-token-prefix setting, which is consistent with stronger conditioning reducing the range of plausible short continuations. Nevertheless, the decoding-order pattern remains unchanged: greedy decoding is least diverse, followed by nucleus sampling and ancestral sampling.

Table 6: Chunk-size ablation for long-sequence conditional VNE on GPT-Neo with chunk size = 1. We report the exponential of conditional VNE for the training reference and generated continuations under greedy decoding, nucleus sampling, and ancestral sampling, using prefix lengths {8, 16, 32} and continuation lengths {16, 32, 64}.
<table><tr><td>Prefix</td><td>Continuation</td><td>Training</td><td>Greedy</td><td>Nucleus</td><td>Ancestral</td></tr><tr><td>8</td><td>16</td><td>437.3</td><td>337.3</td><td>400.7</td><td>410.2</td></tr><tr><td>8</td><td>32</td><td>474.5</td><td>351.8</td><td>425.2</td><td>436.9</td></tr><tr><td>8</td><td>64</td><td>503.7</td><td>365.3</td><td>446.2</td><td>462.3</td></tr><tr><td>16</td><td>16</td><td>431.3</td><td>380.3</td><td>409.3</td><td>415.4</td></tr><tr><td>16</td><td>32</td><td>455.6</td><td>389.7</td><td>422.9</td><td>432.9</td></tr><tr><td>16</td><td>64</td><td>477.9</td><td>400.8</td><td>435.4</td><td>447.5</td></tr><tr><td>32</td><td>16</td><td>432.9</td><td>406.6</td><td>420.3</td><td>424.9</td></tr><tr><td>32</td><td>32</td><td>446.3</td><td>409.3</td><td>427.8</td><td>433.7</td></tr><tr><td>32</td><td>64</td><td>462.1</td><td>413.6</td><td>434.4</td><td>442.3</td></tr></table>

Table 7: Chunk-size ablation for long-sequence conditional VNE on GPT-Neo with chunk size = 4. We report the exponential of conditional VNE for the training reference and generated continuations under greedy decoding, nucleus sampling, and ancestral sampling, using prefix lengths {8, 16, 32} and continuation lengths {16, 32, 64}.
<table><tr><td>Prefix</td><td>Continuation</td><td>Training</td><td>Greedy</td><td>Nucleus</td><td>Ancestral</td></tr><tr><td>8</td><td>16</td><td>454.7</td><td>382.5</td><td>430.5</td><td>433.4</td></tr><tr><td>8</td><td>32</td><td>515.0</td><td>415.2</td><td>479.8</td><td>482.0</td></tr><tr><td>8</td><td>64</td><td>572.7</td><td>435.4</td><td>518.1</td><td>525.0</td></tr><tr><td>16</td><td>16</td><td>455.0</td><td>428.2</td><td>443.1</td><td>443.1</td></tr><tr><td>16</td><td>32</td><td>496.6</td><td>454.4</td><td>475.9</td><td>477.6</td></tr><tr><td>16</td><td>64</td><td>541.3</td><td>474.1</td><td>503.7</td><td>507.4</td></tr><tr><td>32</td><td>16</td><td>459.8</td><td>450.8</td><td>455.8</td><td>456.5</td></tr><tr><td>32</td><td>32</td><td>484.1</td><td>468.0</td><td>475.5</td><td>475.4</td></tr><tr><td>32</td><td>64</td><td>514.8</td><td>483.0</td><td>493.9</td><td>495.1</td></tr></table>

Table 19 reports the full prefix–continuation length ablation for conditional VNE. Increasing continuation length tends to increase the absolute conditional VNE for both training and generated continuations because the output space becomes larger. The average gap remains positive in every tested configuration, confirming that the conditional diversity gap persists across the evaluated tokenization choices. The gap is often larger for shorter prefixes and longer continuations, where the condition leaves more uncertainty about valid outputs and the model must cover a broader conditional continuation space.

## E.6 Chunk-Size Ablation for Long-Sequence Representations

The long-sequence experiments in Section 6.3 represent each sequence as a concatenation of 2-token segment embeddings. To examine the sensitivity of the reported gap to this choice, we vary the chunk size over {1, 2, 4} tokens per segment on GPT-Neo. Since chunk size = 2 corresponds to the configuration already reported in Table 2, here we report the two remaining cases: Tables 6 and 7 show the exponentiated conditional VNE for chunk sizes 1 and 4, respectively. Across all three chunk sizes, the training data consistently achieve higher conditional VNE than the generated continuations under every decoding strategy and every prefix–continuation configuration. The absolute scale of the conditional VNE changes with the chunk size, but the qualitative ordering training > ancestral > nucleus > greedy and the positivity of the gap are preserved. This confirms that the observed long-sequence conditional diversity gap is not an artifact of the specific chunk size used in the main long-sequence analysis.

Table 8: Comparison of temperature scaling and the proposed entropy projection on OLMo. Precision measures generated probability mass within the estimated support of the training continuations [69]. External conditional NLL is computed using an independent language model conditioned on the same prefix. The temperature T ≈ 1.9 is tuned to approximately match the training-data conditional entropy.
<table><tr><td>Method</td><td>Precision ↑</td><td>Ext. Cond. NLL ↓</td></tr><tr><td>Training data</td><td>1.000</td><td>4.679</td></tr><tr><td>Temperature  $T \approx 1 . 9$ </td><td>0.930</td><td>4.938</td></tr><tr><td>Entropy projection (ours)</td><td>0.979</td><td>3.713</td></tr></table>

## E.7 Comparison with Temperature Scaling

A natural question is whether the same diversity gain can be achieved by simply raising the sampling temperature. Theorem 1 does not necessarily apply to temperature scaling, because changing the temperature need not produce the closest distribution satisfying the entropy constraint. To examine this baseline empirically, we repeated the OLMo experiment using ancestral sampling with diferent temperatures. Increasing the temperature raises conditional entropy but does not eliminate the gap within the evaluated range; even at a relatively high temperature, the conditional VNE remains below the training-data value. We then searched for a temperature that matches the training-data conditional entropy and found that approximately $T \approx 1 . 9$ achieves this target. However, matching entropy alone did not match the underlying data distribution.

Table 8 reports two independent quality-oriented measures: precision [69], measuring generated probability mass within the estimated support of the training continuations, and external conditional negative log likelihood, computed using an independent language model conditioned on the same prefix. Although T ≈ 1.9 approximately matches the reference entropy, it yields worse precision and worse external NLL than the training data. In contrast, the proposed projection achieves substantially higher precision and lower external NLL. These results support the distinction motivated by Theorem 1: simply matching entropy is insuficient, and the adjustment should be performed through the closest-distribution projection rather than an arbitrary entropy increase.

## E.8 Qualitative Probe of the Reweighting Behavior

To inspect how the projection redistributes probability mass, we conduct a controlled probe. We take two prompts, “A vibrant city in Northern America is \_” and “A renowned celebrity is \_\_\_”, sample 1000 continuations for each, and reweight them using the conditional entropy projection. The per-output mass shift reveals a clean de-peaking pattern: weight leaves the over-represented dominant modes and moves to equally valid but under-represented alternatives. Table 9 reports the before/after weights for the top dominant modes and selected up-weighted tail candidates. Quantitatively, the discrete entropy and the corresponding efective number of distinct continuations both increase: for the city probe, the discrete entropy rises from 2.587 to 2.853 and the efective number of distinct continuations rises from 13.29 to 17.34; for the celebrity probe, the entropy rises from 3.979 to 4.398 and the efective number rises from 53.46 to 81.25. These examples make the practical efect of the projection transparent: it reduces excessive concentration on dominant generated answers and increases the representation of plausible alternatives already present in the candidate pool, rather than promoting low-quality or implausible outputs.

## E.9 Lexical Audit of Conditional Diversity

To examine whether the observed conditional-entropy ordering also reflects in a directly human-interpretable form, we perform a word-level audit using the same OLMo setting as the main text with 20,000 samples. We count exact, case-insensitive occurrences from three fixed and enumerable semantic lexicons: countries (192 terms), sports (80 terms), and occupations (118 terms). For each category, we normalize the term counts into a categorical distribution and report the number of distinct terms observed as well as the efective number of equally probable lexical choices ${ \bar { \mathbf { \theta } } } _ { 2 } H$ , where H is the Shannon entropy of the normalized distribution in bits.

Table 9: Qualitative probe of the entropy projection on two prompts. We report the probability mass before and after projection for the top down-weighted dominant modes and selected up-weighted tail candidates. “Before” and “After” denote the mass assigned by the original uniformly weighted candidates and the reweighted candidates, respectively.
<table><tr><td>Prompt</td><td>Down-weighted (dominant modes): Before → After</td><td>Up-weighted (plausible tail): Before → After</td></tr><tr><td>“A vibrant city in Northern America is &quot;9</td><td>Toronto: 0.206 → 0.177 Vancouver: 0.198 → 0.137 New York: 0.096 → 0.068</td><td>Miami: 0.018 → 0.046 Denver: 0.020 → 0.035 Philadelphia: 0.010 → 0.022</td></tr><tr><td>“A renowned celebrity is 7</td><td>Oprah Winfrey: 0.134 → 0.064 Madonna: 0.068 → 0.036 Michael Jackson: 0.052 → 0.020 Tom Cruise:  $0 . 0 3 2  0 . 0 2 0$ </td><td>Jackie Chan: 0.010 → 0.019 Kate Middleton: 0.007 → 0.013 Queen Victoria: 0.004 → 0.015 J. K. Rowling: 0.002 → 0.012</td></tr></table>

Table 10: Lexical audit of conditional diversity on OLMo at sample size 20K. For each semantic category, we report the number of distinct terms observed and the efective number of equally probable lexical choices $2 ^ { H }$ where H is the Shannon entropy of the normalized term-frequency distribution in bits. The ordering across all three categories matches the conditional VNE ordering: training > ancestral > nucleus > greedy.
<table><tr><td>Group</td><td>Cond.VNE</td><td>Countries: distinct</td><td>Countries:  $\overline { { 2 ^ { H } } }$ </td><td>Sports: distinct</td><td>Sports: 2H</td><td>Occupations: distinct</td><td>Occupations: 2H</td></tr><tr><td>Training data</td><td>414.2</td><td>95</td><td>54.4</td><td>35</td><td>22.4</td><td>75</td><td>49.9</td></tr><tr><td>Greedy</td><td>369.4</td><td>82</td><td>26.9</td><td>20</td><td>13.6</td><td>66</td><td>35.9</td></tr><tr><td>Nucleus</td><td>390.8</td><td>82</td><td>37.3</td><td>25</td><td>17.9</td><td>69</td><td>42.6</td></tr><tr><td>Ancestral</td><td>394.9</td><td>83</td><td>41.9</td><td>24</td><td>18.4</td><td>68</td><td>44.7</td></tr></table>

Table 10 reports the results. Importantly, all three independent lexical audits recover the same ordering as conditional VNE: training > ancestral > nucleus > greedy. The efect is also concrete at the level of frequencies. For example, in country mentions, greedy decoding assigns 20.0% of its mass to “UK” and 19.8% to “United States”, compared with only 8.4% and 4.3% in the training continuations. This indicates that the conditional VNE gap corresponds to a measurable shortfall in the semantic variety of generated continuations, not merely a change in the kernel-derived score.

## E.10 Conditional Diversity Gaps in Larger Language Models

The main experiments evaluate approximately 1B-parameter models. To examine whether the observed gap is restricted to smaller or earlier models, we additionally evaluate OLMo-3-7B and extend the Pythia experiments to 2.8B and 6.9B parameters under the same setting. Tables 11 and 12 report the exponentiated conditional VNE across sample sizes from 2.5K to 20K.

For OLMo-3-7B, the relative reductions at 20K are approximately 33.0%, 16.2%, and 12.0% under greedy, nucleus, and ancestral decoding, respectively. For Pythia-2.8B and Pythia-6.9B, the corresponding gaps also remain substantial. These results indicate that the conditional output-range gap is not restricted to the earlier approximately 1B-parameter models and does not disappear merely by increasing model size.

Table 11: Conditional diversity scores for OLMo-3-7B across sample sizes. We report the exponential of conditional VNE for the training reference and generated continuations under greedy decoding, nucleus sampling, and ancestral sampling.
<table><tr><td>Group</td><td>2.5K</td><td>5K</td><td>7.5K</td><td>10K</td><td>12.5K</td><td>15K</td><td>17.5K</td><td>20K</td></tr><tr><td>Training data</td><td>86.92</td><td>138.84</td><td>183.04</td><td>220.30</td><td>254.52</td><td>284.20</td><td>312.61</td><td>339.17</td></tr><tr><td>Greedy</td><td>71.92</td><td>107.87</td><td>135.95</td><td>158.50</td><td>178.47</td><td>196.05</td><td>212.69</td><td>227.40</td></tr><tr><td>Nucleus</td><td>79.55</td><td>125.16</td><td>161.26</td><td>191.71</td><td>217.79</td><td>241.40</td><td>264.24</td><td>284.58</td></tr><tr><td>Ancestral</td><td>81.80</td><td>128.47</td><td>166.06</td><td>198.25</td><td>226.70</td><td>252.94</td><td>276.80</td><td>298.40</td></tr></table>

Table 12: Conditional diversity scores for Pythia-2.8B and Pythia-6.9B across sample sizes. We report the exponential of conditional VNE for the training reference and generated continuations under greedy decoding, nucleus sampling, and ancestral sampling.
<table><tr><td>Model</td><td>Group</td><td>2.5K</td><td>5K</td><td>7.5K</td><td>10K</td><td>12.5K</td><td>15K</td><td>17.5K</td><td>20K</td></tr><tr><td>Pyt2.8B</td><td>Training data</td><td>72.60</td><td>114.80</td><td>149.39</td><td>177.89</td><td>203.14</td><td>225.89</td><td>247.28</td><td>265.87</td></tr><tr><td></td><td>Greedy</td><td>58.49</td><td>86.69</td><td>108.65</td><td>125.62</td><td>140.05</td><td>152.45</td><td>163.21</td><td>172.54</td></tr><tr><td></td><td>Nucleus</td><td>66.07</td><td>101.37</td><td>131.11</td><td>155.42</td><td>176.17</td><td>194.62</td><td>211.69</td><td>226.57</td></tr><tr><td></td><td>Ancestral</td><td>68.98</td><td>105.77</td><td>136.45</td><td>161.83</td><td>182.24</td><td>201.95</td><td>220.13</td><td>236.08</td></tr><tr><td>PYt69B</td><td>Training data</td><td>72.60</td><td>114.80</td><td>149.39</td><td>177.89</td><td>203.14</td><td>225.89</td><td>247.28</td><td>265.87</td></tr><tr><td></td><td>Greedy</td><td>58.18</td><td>86.95</td><td>109.39</td><td>126.75</td><td>141.22</td><td>153.96</td><td>164.73</td><td>174.13</td></tr><tr><td></td><td>Nucleus</td><td>67.94</td><td>105.16</td><td>135.18</td><td>159.59</td><td>179.85</td><td>199.01</td><td>215.33</td><td>230.26</td></tr><tr><td></td><td>Ancestral</td><td>68.79</td><td>106.71</td><td>138.51</td><td>164.17</td><td>185.90</td><td>206.12</td><td>224.19</td><td>239.50</td></tr></table>

Table 13: Ablation study with T5 embeddings for kernel-based conditional diversity metrics at sample size 20K. The standard deviations of conditional VNE and conditional RKE are below 0.01 across 5 independent runs. The gap is computed as $\Delta = \mathrm { M e t r i c } _ { \mathrm { t r a i n } } - \mathrm { M e t r i c } _ { \mathrm { g r o u p } }$ . Positive gaps indicate lower diversity than the training data.
<table><tr><td rowspan="2">Model</td><td>Group</td><td>Metric</td><td>Value</td><td>Δ</td></tr><tr><td>Training Data</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>49.74 4.25 0.769 0.964</td><td></td></tr><tr><td rowspan="3">OO</td><td>Greedy Decoding</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>28.22 3.34 0.337 0.538</td><td>21.52 0.91 0.432 0.426</td></tr><tr><td>Nucleus Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>37.94 3.74 0.588 0.889</td><td>11.81 0.51 0.181 0.075</td></tr><tr><td>Ancestral Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>39.64 3.81 0.627 0.917</td><td>10.11 0.44 0.142 0.047</td></tr><tr><td rowspan="4">Pftthia</td><td>Training Data</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>43.06 4.15 0.755 0.930</td><td></td></tr><tr><td>Greedy Decoding</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>26.36 3.43 0.325 0.497</td><td>16.70 0.72 0.430 0.433</td></tr><tr><td>Nucleus Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>34.31 3.76 0.597 0.866</td><td>8.75 0.39 0.158 0.064</td></tr><tr><td>Ancestral Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>36.12 3.83 0.632 0.890</td><td>6.94 0.32 0.123 0.040</td></tr><tr><td rowspan="4">GT-To</td><td>Training Data</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>43.07 4.15 0.755 0.929 27.18</td><td>14.49</td></tr><tr><td>Greedy Decoding</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>3.51 0.344 0.510 34.21</td><td>0.64 0.411 0.419 7.46</td></tr><tr><td>Nucleus Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>3.78 0.599 0.856</td><td>0.37 0.156 0.073 5.63</td></tr><tr><td>Ancestral Sampling</td><td>Exp.Cond.VNE BAp.Cond.RKE Distinct-2 Distinct-3</td><td>36.04 3.85 0.638 0.882</td><td>0.30 0.117 0.047</td></tr></table>

Table 14: Ablation study with a degree-3 polynomial kernel for kernel-based conditional diversity metrics at sample size 20K. The standard deviations of conditional VNE and conditional RKE are below 0.01 across 5 independent runs. The gap is computed as $\Delta = \mathrm { M e t r i c } _ { \mathrm { t r a i n } } - \mathrm { M e t r i c } _ { \mathrm { g r o u p } }$ . Positive gaps indicate lower diversity than the training data.
<table><tr><td>Model</td><td>Group</td><td>Metric Exp.Cond.VNE</td><td>Value</td><td>Δ</td></tr><tr><td rowspan="4">OO</td><td>Training Data</td><td>Exp.Cond.RKE Distinct-2 Distinct-3</td><td>8.71 2.54 0.769 0.964</td><td></td></tr><tr><td>Greedy Decoding</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>6.28 2.41 0.337 0.538</td><td>2.43 0.13 0.432 0.426</td></tr><tr><td>Nucleus Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>7.56 2.46 0.588 0.889</td><td>1.15 0.08 0.181 0.075</td></tr><tr><td>Ancestral Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>7.68 2.47 0.627 0.917</td><td>1.03 0.07 0.142 0.047</td></tr><tr><td rowspan="4">Pftthia</td><td>Training Data</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>7.64 2.50 0.755 0.930</td><td></td></tr><tr><td>Greedy Decoding</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>5.82 2.26 0.325 0.497</td><td>1.82 0.24 0.430 0.433</td></tr><tr><td>Nucleus Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>7.14 2.36 0.597 0.866</td><td>0.50 0.14 0.158 0.064</td></tr><tr><td>Ancestral Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>7.21 2.37 0.632 0.890</td><td>0.43 0.13 0.123 0.040</td></tr><tr><td rowspan="4">GT-To</td><td>Training Data</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>7.64 2.50 0.755 0.929</td><td>1.92</td></tr><tr><td>Greedy Decoding</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>5.72 2.24 0.344 0.510 7.01</td><td>0.26 0.411 0.419 0.63</td></tr><tr><td>Nucleus Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>2.33 0.599 0.856</td><td>0.17 0.156 0.073 0.59</td></tr><tr><td>Ancestral Sampling</td><td>Exp.Cond.VNE Bxp.Cond.RKE Distinct-2 Distinct-3</td><td>7.05 2.34 0.638 0.882</td><td>0.16 0.117 0.047</td></tr></table>

Table 15: Ablation study with a cosine kernel for kernel-based conditional diversity metrics at sample size 20K. The standard deviations of conditional VNE and conditional RKE are below 0.01 across 5 independent runs. The gap is computed as $\Delta = \mathrm { M e t r i c } _ { \mathrm { t r a i n } } - \mathrm { M e t r i c } _ { \mathrm { g r o u p } }$ . Positive gaps indicate lower diversity than the training data.
<table><tr><td>Model</td><td>Group</td><td>Metric</td><td>Value</td><td>Δ</td></tr><tr><td rowspan="4">OO</td><td>Training Data</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>2.72 1.35 0.769 0.964</td><td></td></tr><tr><td>Greedy Decoding</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>2.15 1.24 0.337 0.538</td><td>0.57 0.11 0.432 0.426</td></tr><tr><td>Nucleus Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>2.38 1.28 0.588 0.889</td><td>0.34 0.07 0.181 0.075</td></tr><tr><td>Ancestral Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>2.43 1.29 0.627 0.917</td><td>0.29 0.06 0.142 0.047</td></tr><tr><td rowspan="4">Pftthia</td><td>Training Data</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>2.67 1.34 0.755 0.930</td><td></td></tr><tr><td>Greedy Decoding</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>2.19 1.25 0.325 0.497</td><td>0.47 0.09 0.430 0.433</td></tr><tr><td>Nucleus Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>2.43 1.29 0.597 0.866</td><td>0.24 0.05 0.158 0.064</td></tr><tr><td>Ancestral Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>2.46 1.30 0.632 0.890</td><td>0.20 0.04 0.123 0.040</td></tr><tr><td rowspan="4">GT-To</td><td>Training Data</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>2.67 1.34 0.755 0.929 2.27</td><td>0.41</td></tr><tr><td>Greedy Decoding</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3 Exp.Cond.VNE</td><td>1.27 0.344 0.510 2.46</td><td>0.07 0.411 0.419 0.22</td></tr><tr><td>Nucleus Sampling</td><td>Exp.Cond.RKE Distinct-2 Distinct-3</td><td>1.29 0.599 0.856</td><td>0.05 0.156 0.073 0.17</td></tr><tr><td>Ancestral Sampling</td><td>Exp.Cond.VNE B6p.Cond.RKE Distinct-2 Distinct-3</td><td>2.50 1.30 0.638 0.882</td><td>0.04 0.117 0.047</td></tr></table>

Table 16: Ablation study on prefix token length, with prefix token length set to 4 at sample size 20K. The gap is computed as $\Delta = \mathrm { M e t r i c } _ { \mathrm { t r a i n } } - \mathrm { M e t r i c } _ { \mathrm { g r o u p } } .$ Positive gaps indicate lower diversity than the training data.
<table><tr><td>Model</td><td>Group</td><td>Metric</td><td>Value</td><td>Δ</td></tr><tr><td rowspan="4">OO</td><td>Training Data</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>310.48 100.06 0.709 0.947</td><td></td></tr><tr><td>Greedy Decoding</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>245.72 72.08 0.323 0.545</td><td>64.77 27.98 0.386 0.401</td></tr><tr><td>Nucleus Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>281.28 83.67 0.534 0.858</td><td>29.20 16.39 0.175 0.089</td></tr><tr><td>Ancestral Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>287.63 86.29 0.572 0.892</td><td>22.85 13.77 0.137 0.055</td></tr><tr><td rowspan="6">Pythia GTTeo</td><td>Training Data</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>240.47 84.79 0.708 0.918</td><td></td></tr><tr><td>Greedy Decoding</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3 Exp.Cond.VNE</td><td>193.58 67.28 0.310 0.504 223.86</td><td>46.88 17.51 0.398 0.414 16.61</td></tr><tr><td>Nucleus Sampling</td><td>Exp.Cond.RKE Distinct-2 Distinct-3 Exp.Cond.VNE</td><td>76.50 0.537 0.834 229.11</td><td>8.29 0.171 0.084 11.36</td></tr><tr><td>Ancestral Sampling</td><td>Exp.Cond.RKE Distinct-2 Distinct-3 Exp.Cond.VNE</td><td>78.60 0.578 0.865 240.47</td><td>6.19 0.130 0.053</td></tr><tr><td>Training Data</td><td>Exp.Cond.RKE Distinct-2 Distinct-3 Exp.Cond.VNE</td><td>84.79 0.709 0.917 194.98</td><td>45.49</td></tr><tr><td>Greedy Decoding</td><td>Exp.Cond.RKE Distinct-2 Distinct-3 Exp.Cond.VNE</td><td>70.01 0.330 0.510 223.74</td><td>14.78 0.379 0.407 16.73</td></tr><tr><td rowspan="2"></td><td>Nucleus Sampling</td><td>Exp.Cond.RKE Distinct-2 Distinct-3</td><td>77.72 0.547 0.831</td><td>7.07 0.162 0.086</td></tr><tr><td>Ancestral Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Bistinct-3</td><td>229.45 79.69 0.582 0.863</td><td>11.02 5.10 0.127 0.055</td></tr></table>

Table 17: Ablation study on sample size for diversity metrics across models and generation strategies at sample size 10K. The gap is computed as $\Delta = \mathrm { M e t r i c } _ { \mathrm { t r a i n } } - \mathrm { M e t r i c } _ { \mathrm { g r o u p } } .$ Positive gaps indicate lower diversity than the training data.
<table><tr><td>Model</td><td>Group</td><td>Metric</td><td>Value</td><td>Δ</td></tr><tr><td rowspan="4">OO</td><td>Training Data</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>221.31 67.33 0.769 0.964</td><td></td></tr><tr><td>Greedy Decoding</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3 Exp.Cond.VNE</td><td>154.97 46.19 0.337 0.538</td><td>66.34 21.14 0.432 0.426</td></tr><tr><td>Nucleus Sampling</td><td>Exp.Cond.RKE Distinct-2 Distinct-3</td><td>193.58 55.44 0.588 0.889</td><td>27.73 11.89 0.181 0.075</td></tr><tr><td>Ancestral Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3 Exp.Cond.VNE</td><td>198.51 56.92 0.627 0.917 176.51</td><td>22.80 10.41 0.142 0.047</td></tr><tr><td rowspan="4">Pythia</td><td>Training Data</td><td>Exp.Cond.RKE Distinct-2 Distinct-3</td><td>52.89 0.755 0.930</td><td></td></tr><tr><td>Greedy Decoding</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3 Exp.Cond.VNE</td><td>127.46 40.37 0.325 0.497 159.36</td><td>49.05 12.52 0.430 0.433 17.15</td></tr><tr><td>Nucleus Sampling</td><td>Exp.Cond.RKE Distinct-2 Distinct-3 Exp.Cond.VNE</td><td>46.84 0.597 0.866 165.10</td><td>6.05 0.158 0.064 11.41</td></tr><tr><td>Ancestral Sampling</td><td>Exp.Cond.RKE Distinct-2 Distinct-3 Exp.Cond.VNE</td><td>48.27 0.632 0.890 175.98</td><td>4.62 0.123 0.040</td></tr><tr><td rowspan="4">GT-T-0</td><td>Training Data</td><td>Exp.Cond.RKE Distinct-2 Distinct-3 Exp.Cond.VNE</td><td>52.60 0.755 0.929 130.11</td><td>45.87</td></tr><tr><td>Greedy Decoding</td><td>Exp.Cond.RKE Distinct-2 Distinct-3 Exp.Cond.VNE</td><td>43.03 0.344 0.510 159.35</td><td>9.57 0.411 0.419 16.63</td></tr><tr><td>Nucleus Sampling</td><td>Exp.Cond.RKE Distinct-2 Distinct-3 Exp.Cond.VNE</td><td>47.93 0.599 0.856 163.39</td><td>4.67 0.156 0.073 12.59</td></tr><tr><td>Ancestral Sampling</td><td>Exp.Cond.RKE B&amp;stinct-2 Distinct-3</td><td>48.71 0.638 0.882</td><td>3.89 0.117 0.047</td></tr></table>

Table 18: Ablation study on sample size for diversity metrics across models and generation strategies at sample size 20K. The gap is computed as $\Delta = \mathrm { M e t r i c } _ { \mathrm { t r a i n } } - \mathrm { M e t r i c } _ { \mathrm { g r o u p } } .$ Positive gaps indicate lower diversity than the training data.
<table><tr><td>Model</td><td>Group</td><td>Metric</td><td>Value</td><td>Δ</td></tr><tr><td rowspan="4">OO</td><td>Training Data</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>338.58 67.43 0.709 0.946</td><td></td></tr><tr><td>Greedy Decoding</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>218.88 46.17 0.276 0.465</td><td>119.70 21.26 0.433 0.481</td></tr><tr><td>Nucleus Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>287.31 55.08 0.516 0.847</td><td>51.27 12.35 0.193 0.099</td></tr><tr><td>Ancestral Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>297.31 56.81 0.558 0.883</td><td>41.27 10.62 0.151 0.063</td></tr><tr><td rowspan="5">Ptthia</td><td>Training Data</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>262.55 53.44 0.701 0.913</td><td></td></tr><tr><td>Greedy Decoding</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>176.39 40.98 0.266 0.423</td><td>86.16 12.46 0.435 0.490</td></tr><tr><td>Nucleus Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>232.24 47.51 0.525 0.826</td><td>30.31 5.93 0.176 0.087</td></tr><tr><td>Ancestral Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>241.79 49.00 0.564 0.857</td><td>20.76 4.44 0.137 0.056</td></tr><tr><td>Training Data</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>260.01 52.98 0.701 0.912</td><td></td></tr><tr><td rowspan="4">GPTT-0</td><td>Greedy Decoding</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2 Distinct-3</td><td>179.36 43.34 0.284 0.437</td><td>80.65 9.64 0.417 0.475</td></tr><tr><td>Nucleus Sampling</td><td>Exp.Cond.VNE Exp.Cond.RKE Distinct-2</td><td>229.62 48.48 0.526 0.812</td><td>30.39 4.50 0.175 0.100</td></tr><tr><td>Ancestral Sampling</td><td>Distinct-3 Exp.Cond.VNE Exp.Cond.RKE</td><td>238.50 49.59</td><td>21.51 3.39</td></tr><tr><td></td><td>B30tinct-2 Distinct-3</td><td>0.570 0.848</td><td>0.131 0.064</td></tr></table>

Table 19: Full prefix and continuation length ablation for LLM conditional VNE. We report the exponential of conditional VNE for the training reference and generated continuations under greedy decoding, nucleus sampling, and ancestral sampling. The average gap is computed across the three decoding strategies.
<table><tr><td>Model</td><td>Prefix</td><td>Continuation</td><td>Training</td><td>Greedy</td><td>Nucleus</td><td>Ancestral</td><td>Avg. Gap</td></tr><tr><td rowspan="6">OLMo</td><td>3</td><td>2</td><td>338.58</td><td>218.88</td><td>287.31</td><td>297.30</td><td>70.75</td></tr><tr><td>3</td><td>3</td><td>617.63</td><td>404.37</td><td>564.52</td><td>581.67</td><td>100.78</td></tr><tr><td>3</td><td>4</td><td>741.13</td><td>517.42</td><td>716.87</td><td>726.35</td><td>87.58</td></tr><tr><td>4</td><td>2</td><td>310.48</td><td>245.72</td><td>281.27</td><td>287.63</td><td>38.94</td></tr><tr><td>4</td><td>3</td><td>470.69</td><td>388.95</td><td>450.11</td><td>455.52</td><td>39.16</td></tr><tr><td>4</td><td>4</td><td>532.24</td><td>463.86</td><td>525.02</td><td>529.51</td><td>26.11</td></tr><tr><td rowspan="6">Pythia</td><td>3</td><td>2</td><td>262.55</td><td>176.39</td><td>232.24</td><td>241.79</td><td>45.74</td></tr><tr><td>3</td><td>3</td><td>492.00</td><td>301.65</td><td>439.49</td><td>456.91</td><td>92.65</td></tr><tr><td>3</td><td>4</td><td>604.57</td><td>397.22</td><td>575.02</td><td>589.29</td><td>84.06</td></tr><tr><td>4</td><td>2</td><td>240.46</td><td>193.58</td><td>223.85</td><td>229.10</td><td>24.95</td></tr><tr><td>4</td><td>3</td><td>370.43</td><td>285.74</td><td>345.38</td><td>352.88</td><td>42.43</td></tr><tr><td>4</td><td>4</td><td>423.95</td><td>342.08</td><td>408.47</td><td>414.80</td><td>35.50</td></tr><tr><td rowspan="6">GPT-Neo</td><td>3</td><td>2</td><td>262.55</td><td>179.36</td><td>229.62</td><td>238.49</td><td>46.73</td></tr><tr><td>3</td><td>3</td><td>491.53</td><td>301.45</td><td>432.98</td><td>452.09</td><td>96.02</td></tr><tr><td>3</td><td>4</td><td>607.21</td><td>391.38</td><td>561.90</td><td>579.97</td><td>96.13</td></tr><tr><td>4</td><td>2</td><td>240.47</td><td>194.97</td><td>223.74</td><td>229.45</td><td>24.42</td></tr><tr><td>4</td><td>3</td><td>371.87</td><td>284.75</td><td>341.55</td><td>352.06</td><td>45.75</td></tr><tr><td>4</td><td>4</td><td>426.57</td><td>339.42</td><td>404.99</td><td>412.19</td><td>41.04</td></tr></table>

Table 20: Conditional diversity scores for class-conditioned ImageNet generation across sample sizes. Rows correspond to fixed sample sizes, and columns correspond to the ImageNet reference data and evaluated generative models. We report the exponential of conditional VNE as mean ± standard deviation over 5 independent runs.
<table><tr><td>Sample Size</td><td><img src="images/6787689d3e52bb52d9f2fe2fa795a6015fff7ccf928a76d768a4523db63983bf.jpg"/></td><td><img src="images/b6d37ca952e84d8e22b1d4353b2e8e48cee0f0cabc2686a0e3ece16526794f43.jpg"/></td><td><img src="images/5a285f6ecb60451ef5300c54a50050b152937477988b6bc064d934dac0a556e5.jpg"/></td><td><img src="images/c31b9befddab28dd880a0573885cb56fe4297f8c8c9399696153a03105937260.jpg"/></td><td><img src="images/120e9ded526e8af32c80f8e7e856600130096adaecb491b863b42244243f3e66.jpg"/></td></tr><tr><td>2.5K</td><td> $1 7 . 5 3 \pm 0 . 0 9$ </td><td> $1 6 . 6 6 \pm 0 . 1 1$ </td><td> $5 . 7 8 \pm 0 . 0 2$ </td><td> $5 . 3 2 \pm 0 . 0 3$ </td><td> $1 6 . 9 6 \pm 0 . 0 6$ </td></tr><tr><td>5K</td><td> $2 6 . 4 1 \pm 0 . 1 2$ </td><td> $2 4 . 5 0 \pm 0 . 0 8$ </td><td> $9 . 2 0 \pm 0 . 0 4$ </td><td> $8 . 0 2 \pm 0 . 0 5$ </td><td> $2 5 . 1 4 \pm 0 . 1 0$ </td></tr><tr><td>7.5K</td><td> $3 3 . 0 1 \pm 0 . 1 0$ </td><td> $3 0 . 2 7 \pm 0 . 1 9$ </td><td> $1 2 . 0 7 \pm 0 . 0 3$ </td><td> $1 0 . 1 4 \pm 0 . 0 6$ </td><td> $3 1 . 2 5 \pm 0 . 1 7$ </td></tr><tr><td>10K</td><td> $3 8 . 5 7 \pm 0 . 0 8$ </td><td> $3 4 . 9 0 \pm 0 . 1 9$ </td><td> $1 4 . 5 6 \pm 0 . 0 4$ </td><td> $1 1 . 8 9 \pm 0 . 0 5$ </td><td> $3 6 . 1 8 \pm 0 . 2 2$ </td></tr><tr><td>12.5K</td><td> $4 3 . 1 5 \pm 0 . 1 8$ </td><td> $3 8 . 7 5 \pm 0 . 1 8$ </td><td> $1 6 . 8 0 \pm 0 . 0 4$ </td><td> $1 3 . 4 0 \pm 0 . 0 5$ </td><td> $4 0 . 2 4 \pm 0 . 2 0$ </td></tr><tr><td>15K</td><td> $4 7 . 2 3 \pm 0 . 2 3$ </td><td> $4 2 . 1 4 \pm 0 . 1 7$ </td><td> $1 8 . 8 4 \pm 0 . 0 6$ </td><td> $1 4 . 7 5 \pm 0 . 0 5$ </td><td> $4 3 . 8 6 \pm 0 . 1 9$ </td></tr><tr><td>17.5K</td><td> $5 0 . 6 3 \pm 0 . 1 7$ </td><td> $4 5 . 1 3 \pm 0 . 1 7$ </td><td> $2 0 . 7 2 \pm 0 . 0 4$ </td><td> $1 5 . 9 5 \pm 0 . 0 4$ </td><td> $4 7 . 0 8 \pm 0 . 2 4$ </td></tr><tr><td>20K</td><td> $5 4 . 1 5 \pm 0 . 1 4$ </td><td> $4 7 . 8 6 \pm 0 . 1 7$ </td><td> $2 2 . 4 7 \pm 0 . 0 3$ </td><td> $1 7 . 0 4 \pm 0 . 0 2$ </td><td> $4 9 . 9 5 \pm 0 . 2 1$ </td></tr></table>

![](images/48bc233d3aebffede759ea1cc7cd82281b72d2469c6208f1922a3c06d4fd7e04.jpg)  
Figure 2: Conditional diversity gap on class-conditioned ImageNet generation. We plot conditional VNE as a function of sample size for the ImageNet training-data reference and generated samples from ADM, $\mathrm { D i T - X L - 2 } ,$ LDM, and BigGAN under matched class labels. Across all evaluated sample sizes, the reference ImageNet data achieves higher conditional VNE than the generated samples, and the gap generally increases as the sample size grows.

![](images/3e37e8333d939703d5e74e49874ea4f1721bca925466800190d42a99db01f3ab.jpg)

Deep U-ViT  
![](images/730d2e7cc47b9d7beb44961f4cb55ebf370b821f62a7ae10e7f0d841d53e29df.jpg)

![](images/ad8a0d9a7e1cc993dcf3f6983a4e165c55654778899de965fd23e87c77eb04dc.jpg)

U-ViT  
![](images/268df60a01c6dd91712b9c4696ede271c2858dcdfb508be66708ef0d09bf02cb.jpg)  
Figure 3: Conditional diversity gap on text-conditioned MS-COCO generation. We plot conditional VNE as a function of sample size for the MS-COCO real-data reference and generated samples from SDXL, PixArt, U-ViT, and Deep U-ViT under matched text captions. Across all evaluated sample sizes, the real-data reference achieves higher conditional VNE than the generated samples, and the gap generally becomes larger with increasing sample size.

Table 21: Conditional diversity scores for text-conditioned MS-COCO generation across sample sizes. Rows correspond to fixed sample sizes, and columns correspond to the MS-COCO reference data and evaluated text-to-image generative models. We report the exponential of conditional VNE as mean ± standard deviation over independent runs.
<table><tr><td>Sample Size</td><td><img src="images/9de0fec5ee934ba9c68c8577900dbc8dae34be7f57df4753063fcc550ce0f8bb.jpg"/></td><td><img src="images/20dc8a18974afb4375748d8d293d3b0081dde9dfe3615e2f27f777a30a9c4a7f.jpg"/></td><td><img src="images/d31d5dfc7701ea6e84649c7c4c8c47a9068d84e37445322d1a09d3dbf38fb57d.jpg"/></td><td><img src="images/4561fda8f8cc795f63c4b652662fd4acda6805ef4be2a38445e3c2a4ffe28026.jpg"/></td><td><img src="images/7241838bf75205c8a99c23b0de15c6893ae76e48a4723907486eff397d01fc56.jpg"/></td></tr><tr><td>2.5K</td><td> $1 3 . 8 8 \pm 0 . 0 7$ </td><td> $1 1 . 6 3 \pm 0 . 0 6$ </td><td> $1 1 . 5 1 \pm 0 . 0 4$ </td><td> $1 2 . 0 2 \pm 0 . 0 2$ </td><td> $1 0 . 8 7 \pm 0 . 0 4$ </td></tr><tr><td>5K</td><td> $1 9 . 9 1 \pm 0 . 0 4$ </td><td> $1 6 . 0 7 \pm 0 . 0 6$ </td><td> $1 5 . 8 9 \pm 0 . 0 9$ </td><td> $1 6 . 4 9 \pm 0 . 0 3$ </td><td> $1 4 . 5 4 \pm 0 . 0 5$ </td></tr><tr><td>7.5K</td><td> $2 4 . 6 2 \pm 0 . 0 5$ </td><td> $1 9 . 3 3 \pm 0 . 0 7$ </td><td> $1 9 . 0 9 \pm 0 . 0 9$ </td><td> $1 9 . 7 5 \pm 0 . 1 4$ </td><td> $1 7 . 1 5 \pm 0 . 0 3$ </td></tr><tr><td>10K</td><td> $2 8 . 5 1 \pm 0 . 0 6$ </td><td> $2 2 . 0 3 \pm 0 . 0 5$ </td><td> $2 1 . 7 3 \pm 0 . 0 5$ </td><td> $2 2 . 5 1 \pm 0 . 0 9$ </td><td> $1 9 . 3 0 \pm 0 . 0 1$ </td></tr><tr><td>12.5K</td><td> $3 1 . 9 3 \pm 0 . 0 6$ </td><td> $2 4 . 3 6 \pm 0 . 0 5$ </td><td> $2 4 . 0 3 \pm 0 . 0 6$ </td><td> $2 4 . 7 6 \pm 0 . 0 7$ </td><td> $2 1 . 0 6 \pm 0 . 0 3$ </td></tr><tr><td>15K</td><td> $3 5 . 0 3 \pm 0 . 0 3$ </td><td> $2 6 . 3 8 \pm 0 . 0 6$ </td><td> $2 6 . 0 0 \pm 0 . 0 6$ </td><td> $2 6 . 7 6 \pm 0 . 0 7$ </td><td> $2 2 . 6 2 \pm 0 . 0 2$ </td></tr><tr><td>17.5K</td><td> $3 7 . 9 2 \pm 0 . 0 4$ </td><td> $2 8 . 2 4 \pm 0 . 0 5$ </td><td> $2 7 . 8 4 \pm 0 . 0 5$ </td><td> $2 8 . 5 8 \pm 0 . 0 4$ </td><td> $2 4 . 0 1 \pm 0 . 0 0$ </td></tr><tr><td>20K</td><td> $4 0 . 5 2 \pm 0 . 0 3$ </td><td> $2 9 . 9 4 \pm 0 . 0 3$ </td><td> $2 9 . 5 0 \pm 0 . 0 4$ </td><td> $3 0 . 2 5 \pm 0 . 0 4$ </td><td> $2 5 . 2 5 \pm 0 . 0 1$ </td></tr></table>