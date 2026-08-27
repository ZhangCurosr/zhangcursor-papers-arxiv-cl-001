# GGSS: Geodesic-Gated Spherical Steering for Inference-Time Debiasing of Generative Vision–Language Models

Yiqun Sun<sup>1</sup>, Junyu Chen<sup>2</sup>, Pengfei Wei<sup>1</sup>\*, Lawrence B. Hsieh<sup>1</sup> <sup>1</sup>Magellan Technology Research Institute (MTRI), <sup>2</sup>National University of Singapore {duke.sun, pengfei.wei, lawrence.hsieh}@mtri.co.jp, chenjunyu@u.nus.edu

## Abstract

Generative vision–language models (VLMs) are increasingly used in human-centered settings, yet they can produce demographically biased outputs even when images differ only in controlled attributes such as perceived race or gender. However, existing inference-time debiasers were largely designed for static embeddings or CLIP-like models rather than generative VLMs. We propose GGSS—Geodesic-Gated Spherical Steering—a norm-preserving intervention that discovers a counterfactual bias subspace on the unit hypersphere, steers visual tokens along geodesic arcs, and uses an adaptive gate to focus correction on tokens that carry stronger demographic signal. We evaluate four generative VLMs against ten adapted inferencetime debiasing baselines and prompt-based mitigation under a single operating-point protocol across categorical, pairwise, and occupationgender bias tests, while also measuring general visual-language capability. GGSS achieves the lowest average bias on all four models, significant on three of four backbones under paired permutation tests, while preserving MMStar accuracy within ±0.6 p.p. of the unsteered baseline. Code is available at https://github.c om/dukesun99/GGSS.

## 1 Introduction

Vision–language models (VLMs) now appear in human-centered, and sometimes high-stakes, decision-making settings, including recruitment and decision support (Liu et al., 2023; Wu et al., 2024; Banerjee et al., 2024). As they become more capable and more widely deployed, a growing body of work has documented that modern VLMs produce systematically different outputs across demographic groups even when the underlying visual content is controlled (Chen et al., 2026; Howard et al., 2024; Huang et al., 2025). Debiasing such models at inference time (Figure 1) is attractive because it avoids retraining, applies to frozen checkpoints, and can be reconfigured as the target notion of bias changes (Chuang et al., 2023; Gerych et al., 2024, 2026). However, many existing inferencetime debiasing methods were developed for CLIPlike (Radford et al., 2021) settings where an input is summarized by a single global embedding. INLP, R-LACE, and LEACE erase concepts through linear subspace operations (Ravfogel et al., 2020, 2022; Belrose et al., 2023). Classifier-guided methods apply mean-shift corrections. BendVLM performs retrieval-based equalization in CLIP-style embedding spaces (Gerych et al., 2024; Radford et al., 2021). Modern generative VLMs (Bai et al., 2025; Liu et al., 2024) instead encode an image as many visual tokens that are fused with a decoderonly language model. Reusing CLIP-era methods in this setting can be overly destructive and, as our experiments show, can even be dominated by the

![](images/d6893c5f9db2ec14848a564e2f021e3c8a5566c48284ac1644a9fb6815b9b9ef.jpg)  
Figure 1: Motivation for inference-time debiasing. Instead of retraining a VLM with additional data and heavy GPU resources, GGSS inserts a lightweight debiasing hook during inference to reduce demographic bias efficiently while keeping the original model frozen.

unsteered baseline.

We identify three concrete challenges that a successful inference-time debiaser for generative VLMs must address. First, geometry preservation: visual activations in recent VLMs enter the language model through a small set of projection layers whose outputs are later consumed by attention and decoding. Euclidean subtraction of a bias direction distorts both directions and norms; empirically this matters increasingly with model scale, causing sharp drops in general quality and over-steering failure modes that on-sphere steering avoids (§4.3). Second, token-level bias heterogeneity: demographic signal in a generative VLM is concentrated in a small fraction of visual tokens (e.g., face or clothing tokens) while many tokens encode pose, background, or textual cues and carry essentially none. Hard subspace projection applied uniformly over-corrects the latter and silently degrades task performance. Third, singledirection rigidity: many existing methods assume a binary protected attribute and a single “bias direction,” but race-like attributes in practice have multi-class structure and must be handled as a lowdimensional subspace (Manzini et al., 2019).

We propose GGSS (Geodesic-Gated Spherical Steering), an inference-time debiasing framework that matches these three challenges with three coordinated design choices. First, we discover a bias subspace $\mathbf { V } _ { \mathrm { b i a s } }$ from counterfactual image sets that vary demographic attributes while holding other visual factors as fixed as possible. We collect pooled visual activations and apply singular value decomposition to tangent-space shifts around per-group Fréchet means on the unit hypersphere, yielding a multi-dimensional and norm-aware estimate of demographic variation. Second, at inference time we replace hard null-space projection with spherical linear interpolation (Shoemake, 1985) between the original token direction and a debiased target, producing a smooth geodesic rotation that preserves norms exactly. Third, we add an adaptive confidence gate g per token, calibrated from the distribution of bias-projection magnitudes observed during discovery. This gate applies little or no steering to low-bias tokens and stronger steering to tokens with unusually large bias components, which makes the intervention more selective than a uniform projection.

We evaluate GGSS on four generative VLMs against ten adapted inference-time debiasing baselines, a prompt-based mitigation family, and four structural ablations. The evaluation covers raceand gender-related bias tests together with MMStar capability preservation, and all methods are compared under a fair protocol, with bootstrap confidence intervals and paired permutation tests for every reported comparison. Experiments show GGSS attains the lowest average bias on all four models, with statistically significant reductions on three of the four backbones and per-task reductions up to 96%, 84%, and 61%, while keeping MMStar accuracy within ±0.6 p.p. of baseline. These results support our central claim: for generative VLMs, adaptive geodesic steering is a better primitive for inference-time bias mitigation than hard, uniform projection. Matched-gate ablations attribute robustness at scale to on-sphere steering and selectivity to the gate.

Our contributions are fourfold:

• We identify why hard, token-uniform, singledirection debiasing operators under-perform on generative VLMs, and propose GGSS as a norm-preserving geodesic alternative.

• We introduce an adaptive gate that uses the discovery-set bias-norm distribution to assign per-token steering strengths without additional training.

• We port representative CLIP- and LM-era debiasers to the same generative-VLM intervention layer, producing a unified comparison suite.

• We evaluate four VLMs under one best-avg-α protocol; GGSS achieves the lowest average bias on all four while preserving MMStar capability.

## 2 Related Work

Social biases in vision–language models. Prior work shows that VLMs encode and express social biases across representations, benchmarks, and downstream behaviors. Grounded vision-language embeddings inherit demographic associations from earlier representation models (Ross et al., 2021; Srinivasan and Bisk, 2022), and later studies measure disparities across gender, race, age, nationality, religion, and other attributes (Zhou et al., 2022; Janghorbani and De Melo, 2023; Ruggeri and Nozza, 2023). These biases also surface in captioning, visual question answering, retrieval, open-ended prompting, counterfactual evaluation, and real-image settings (Hirota et al., 2022; Hall et al., 2023; Ghate et al., 2025; Howard et al., 2024; Wang et al., 2024; Huang et al., 2025; Narnaware et al., 2025). This motivates inference-time mitigation methods that can be applied to frozen models without retraining.

VLM debiasing: training-based vs. inferencetime. Most VLM debiasing methods target CLIP-style dual encoders, not generative VLMs. Training-based approaches learn lightweight transformations, pruning or imputation modules, causal adjustments, bias-corpus disentanglement, LLMguided projections, or joint image–text corrections (Seth et al., 2023; Jung et al., 2024; Pang et al., 2025; Jang et al., 2025; Molahasani et al., 2025; Zhang et al., 2025). Inference-time methods instead modify prompts or embedding geometry at test time, including calibrated prompt-defined projection, query-specific local debiasing, and rotationbased correction (Chuang et al., 2023; Gerych et al., 2024, 2026). Post-hoc transformations of embedding geometry also serve instruction following (Feng et al., 2025) and concept suppression in text-to-image generation (Chen et al., 2025; Li et al., 2025). Debiasing for generative VLMs is less explored, with recent work moving toward internal activation intervention and monitoring (Ratzlaff et al., 2024; An et al., 2026; Huben et al., 2024; Cheng et al., 2025; Singh et al., 2026; Wang et al., 2026). GGSS follows this activation-level direction, but focuses on generative VLM on token activations.

Concept erasure and activation steering. Our baselines draw on linear concept erasure and inference-time activation steering. INLP, R-LACE, and LEACE remove protected information by learning or solving for linear subspace projections (Ravfogel et al., 2020, 2022; Belrose et al., 2023). These classical erasers are useful reference points, but they were designed for static embeddings or encoder representations and typically act as hard, token-uniform projections. Activation steering methods such as ActAdd, Representation Engineering, and Inference-Time Intervention instead manipulate hidden states directly at test time (Turner et al., 2023; Zou et al., 2023; Li et al., 2023). GGSS shares the inference-time spirit of this line, but replaces additive Euclidean steering with spherical geometry. The unit hypersphere and spherical interpolation are standard tools in directional statistics and computer graphics (Mardia and Jupp, 2000; Pennec, 2006; Shoemake, 1985), and related geometric views of concept directions have appeared in representation analysis and null-space steering (Park et al., 2023; Sheng et al., 2026; Sun et al., 2025a). We instantiate these ideas for imageside activations in generative VLMs and add a biasnorm-calibrated gate that makes steering tokenadaptive.

## 3 The GGSS Framework

We propose GGSS—Geodesic-Gated Spherical Steering—a two-stage framework for debiasing generative VLMs, summarized in Figure 2. Offline discovery passes counterfactual images through the frozen VLM to estimate the steering quantities: a protected-attribute basis, a global spherical reference point µ, and gate calibration statistics $( \tilde { b } , \sigma _ { b } )$ Online inference is separate: a new test input produces an activation tensor, and GGSS steers its visual tokens using the discovered quantities. The learned quantities are reused for new inputs at inference time.

Notation. Let

$$
\mathbb { S } ^ { D - 1 } = \left\{ \mathbf { x } \in \mathbb { R } ^ { D } : \| \mathbf { x } \| _ { 2 } = 1 \right\}
$$

be the unit sphere in $\mathbb { R } ^ { D }$ , and let $\begin{array} { r } { \nu ( \mathbf { x } ) = \frac { \mathbf { x } } { \| \mathbf { x } \| _ { 2 } } } \end{array}$ be the normalization operator. For $\mathbf { p } , \mathbf { q } \in \mathbb { S } ^ { D - 1 }$ define $d ( \mathbf { p } , \mathbf { q } ) = \operatorname { a r c c o s } \langle \mathbf { p } , \mathbf { q } \rangle$ and

$$
\begin{array} { r l } & { \log _ { \mathbf { p } } ( \mathbf { q } ) = d ( \mathbf { p } , \mathbf { q } ) \frac { \mathbf { q } - \mathbf { et { } { ' } } ( \mathbf { p } , \mathbf { q } ) \mathbf { p } } { \| \mathbf { q } - \mathbf { et { } { ' } } \mathbf { p } , \mathbf { q }  \mathbf { p } \| _ { 2 } } , } \\ & { \exp _ { \mathbf { p } } ( \mathbf { t } ) = \cos ( \| \mathbf { t } \| _ { 2 } ) \mathbf { p } + \sin ( \| \mathbf { t } \| _ { 2 } ) \frac { \mathbf { t } } { \| \mathbf { t } \| _ { 2 } } , } \end{array}
$$

with the standard conventions $\log _ { \mathbf { p } } ( \mathbf { p } ) = \mathbf { 0 }$ and $\exp _ { \mathbf { p } } ( \mathbf { 0 } ) = \mathbf { p }$ . For $\mathbf { p } , \mathbf { q } \in \mathbb { S } ^ { D - 1 }$ with $\mathbf p \neq \mathbf q$ and $\mathbf { q } \neq - \mathbf { p } .$ , define spherical linear interpolation by

$$
\begin{array} { c } { { \mathrm { S l e r p } ( { \bf p } , { \bf q } ; \beta ) = \displaystyle \frac { \sin ( ( 1 - \beta ) \theta ) } { \sin \theta } { \bf p } + \displaystyle \frac { \sin ( \beta \theta ) } { \sin \theta } { \bf q } , } } \\ { { \theta = \displaystyle \ a r c c o s \langle { \bf p } , { \bf q } \rangle . } } \end{array}
$$

When $\mathbf { p } \ = \ \mathbf { q } ,$ we use the continuous extension ${ \mathrm { S l e r p } } ( \mathbf { p } , \mathbf { p } ; { \boldsymbol { \beta } } ) = \mathbf { p }$ . For $0 \leq \beta \leq 1$ , Slerp interpolates between p and q; for $\beta > 1$ , it continues along the same geodesic beyond the target.

The rest of this section follows the two stages of GGSS. Section 3.1 presents the offline discovery stage, where counterfactual activations are used to estimate a protected-attribute subspace, a global reference point, and gate calibration statistics. Section 3.2 presents the inference-time steering stage, where each visual token is projected away from the learned protected-attribute subspace, updated along the sphere, and rescaled to its original norm.

![](images/9c02c787856871b9ed387c97aeacd99113399de32afef6f8bca1114e85728adb.jpg)  
Figure 2: Overview of the GGSS framework. The offline stage estimates a spherical bias subspace from counterfactual image activations and calibrates a bias-sensitive token gate. At inference time, each visual token is mapped to tangent space, decomposed into bias and clean components, gated according to its bias magnitude, geodesically rotated toward a debiased direction via Slerp, and finally norm-restored before being passed to the LLM decoder of the VLM.

## 3.1 Counterfactual Bias Subspace Discovery

Counterfactual representatives. We write the discovery images as

$$
\mathbf { x } _ { u , a } , \qquad u \in \mathcal { U } , \quad a \in \mathcal { A } .
$$

Here a indexes the protected attribute value, such as a race category, while u collects the fixed factors, such as identity, occupation, gender, pose, clothing, background, and lighting. This abstracts the concrete occupation/identity/race/gender indexing specified in Appendix A.1. Running the frozen model on ${ \bf x } _ { u , a }$ with a fixed discovery prompt, the text instruction paired with each discovery image (“Describe this image in detail.”), gives the targetlayer activation

$$
\mathbf { H } _ { u , a } \in \mathbb { R } ^ { T \times D } .
$$

The choice of discovery prompt is immaterial at our intervention layer: the vision-to-language projection computes its visual-token activations before any interaction with the text prompt, so the prompt architecturally cannot influence them, and re-running discovery with four alternative prompts yields subspaces identical to numerical precision (principal-angle cosines $\ge 0 . 9 9 9 7 )$ and identical end-to-end results (Appendix B.13). We summarize each image by the normalized pooled representative

$$
\mathbf { y } _ { u , a } = \nu \left( \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbf { H } _ { u , a , t } \right) \in \mathbb { S } ^ { D - 1 } .
$$

Within a fixed context u, varying a is designed to emphasize protected-attribute variation while keeping other visual factors fixed. Pooling yields one stable spherical representative per counterfactual image.

Subspace estimation. For each fixed context $u ,$ let the spherical Fréchet mean be

$$
\mathbf { c } _ { u } \in \arg \operatorname* { m i n } _ { \mathbf { z } \in \mathbb { S } ^ { D - 1 } } \sum _ { a \in \mathcal { A } } d ( \mathbf { z } , \mathbf { y } _ { u , a } ) ^ { 2 } .
$$

The protected-attribute shift for value a is

$$
\begin{array} { r } { { \bf s } _ { u , a } = \log _ { { \bf c } _ { u } } ( { \bf y } _ { u , a } ) . } \end{array}
$$

Stacking the shifts $\{ \mathbf { s } _ { u , a } : u \in \mathcal { U } , a \in \mathcal { A } \}$ as rows gives the shift matrix

$$
\mathbf { S } \in \mathbb { R } ^ { N _ { \mathrm { s h i f t s } } \times D } , \qquad N _ { \mathrm { s h i f t s } } = | \mathcal { U } | | \mathcal { A } | .
$$

Each row of $\mathbf { S }$ is one tangent shift induced by changing the protected attribute within a fixed context. We then apply SVD:

$$
\mathbf { S } = \mathbf { U } \Sigma \mathbf { V } ^ { \top } , \qquad \mathbf { V } _ { \mathrm { b i a s } } = [ \pmb { v } _ { 1 } , \dots , \pmb { v } _ { k } ] \in \mathbb { R } ^ { D \times k } .\tag{1}
$$

Here $v _ { j }$ is the j-th column of V. Thus $\mathbf { V } _ { \mathrm { b i a s } }$ is the matrix of the top-k right singular vectors of $\mathbf { S } _ { \mathbf { \partial } }$ and span $\left( \mathbf { V } _ { \mathrm { b i a s } } \right)$ is the learned protected-attribute subspace. Since the columns of V are orthonormal, $\mathbf { V } _ { \mathrm { b i a s } } ^ { \top } \mathbf { V } _ { \mathrm { b i a s } } = I _ { k }$ . We set $k = | \mathcal { A } | - 1$ by default. The global reference point is the Fréchet mean of all discovery representatives:

$$
\pmb \mu \in \arg \operatorname* { m i n } _ { \mathbf z \in \mathbb { S } ^ { D - 1 } } \sum _ { u \in \mathcal { U } } \sum _ { a \in \mathcal { A } } d ( \mathbf z , \mathbf y _ { u , a } ) ^ { 2 } .
$$

Together, $\mathbf { V } _ { \mathrm { b i a s } }$ and $\pmb { \mu }$ parameterize the steering geometry. Unlike classifier-based discovery methods, the subspace is estimated directly from counterfactual activation differences and requires no supervised probe.

Gate calibration statistics. For each discovery representative $\mathbf { c } \in \{ \mathbf { y } _ { u , a } : u \in \mathcal { U } , a \in \mathcal { A } \}$ , compute

$$
b ( \mathbf { c } ) = \| \mathbf { V } _ { \mathrm { b i a s } } ^ { \top } \log _ { \pmb { \mu } } ( \mathbf { c } ) \| _ { 2 } ,
$$

where $\mathbf { V } _ { \mathrm { b i a s } } ^ { \top } \log _ { \mu } ( \mathbf { c } )$ gives the coordinates of $\mathbf { c } _ { \mathrm { : } }$ after mapping to the reference point $\mathbf { \mu } _ { \mu } ,$ along the learned protected-attribute basis. Thus $b ( \mathbf { c } )$ is the protected-coordinate magnitude of the discovery representative. We store

$$
\tilde { b } = \mathrm { m e d i a n } \{ b ( { \bf c } ) \} , \qquad \sigma _ { b } = \mathrm { s t d } \{ b ( { \bf c } ) \} .
$$

These scalars calibrate the token gate at inference time.

## 3.2 Token-Level Geodesic Steering

At inference time, the new input produces a targetlayer activation tensor

$$
\mathbf { Z } \in \mathbb { R } ^ { n _ { \mathrm { b a t c h } } \times T ^ { \prime } \times D } .
$$

We flatten it to token vectors $\{ { \bf h } _ { i } \} _ { i = 1 } ^ { N }$ , with $N =$ $n _ { \mathrm { b a t c h } } T ^ { \prime }$ , and process each nonzero token independently. Let

$$
r _ { i } = \| \mathbf { h } _ { i } \| _ { 2 } , \qquad \widehat { \mathbf { h } } _ { i } = \nu ( \mathbf { h } _ { i } ) , \qquad \mathbf { t } _ { i } = \log _ { \mu } ( \widehat { \mathbf { h } } _ { i } ) .
$$

The radius $r _ { i }$ is kept outside the debiasing step and restored at the end.

Projection and target direction. Because $\mathbf { V } _ { \mathrm { b i a s } }$ has orthonormal columns, $\mathbf { V } _ { \mathrm { b i a s } } \mathbf { V } _ { \mathrm { b i a s } } ^ { \top }$ is the orthogonal projector onto span $\left( \mathbf { V } _ { \mathrm { b i a s } } \right)$ . The protectedcoordinate component and its orthogonal complement are

$$
\begin{array} { r } { \mathbf { p } _ { i } = \mathbf { V } _ { \mathrm { b i a s } } \mathbf { V } _ { \mathrm { b i a s } } ^ { \top } \mathbf { t } _ { i } , \quad } \\ { \mathbf { t } _ { i } ^ { \mathrm { c l e a n } } = ( I - \mathbf { V } _ { \mathrm { b i a s } } \mathbf { V } _ { \mathrm { b i a s } } ^ { \top } ) \mathbf { t } _ { i } . } \end{array}\tag{2}
$$

Here $\mathbf { p } _ { i }$ is the component removed by GGSS, while $\mathbf { t } _ { i } ^ { \mathrm { c l e a n } }$ is the steering coordinate after zeroing the learned protected-attribute coordinates. Mapping the projected tangent vector back to the sphere gives the target direction

$$
\widehat { \mathbf { h } } _ { i } ^ { \mathrm { t a r g e t } } = \mathrm { e x p } _ { \mu } ( \mathbf { t } _ { i } ^ { \mathrm { c l e a n } } ) .
$$

This defines the target direction by first projecting in the steering coordinates and then mapping the projected coordinate back to the sphere.

Calibrated token gate. The token-specific gate is

$$
\begin{array} { r l } & { z _ { i } = \frac { \| \mathbf { V } _ { \mathrm { b i a s } } ^ { \top } \mathbf { t } _ { i } \| _ { 2 } - \tilde { b } } { \sigma _ { b } + \varepsilon } , } \\ & { g _ { i } = g _ { \mathrm { f l o o r } } + ( 1 - g _ { \mathrm { f l o o r } } ) \mathrm { s i g m o i d } ( \kappa z _ { i } ) , } \end{array}\tag{3}
$$

where $\varepsilon > 0$ avoids division by zero, κ controls gate sharpness, and $g _ { \mathrm { f l o o r } }$ sets the minimum steering strength. Since $\mathbf { V } _ { \mathrm { b i a s } }$ has orthonormal columns, $\lVert \mathbf { V } _ { \mathrm { b i a s } } ^ { \top } \mathbf { t } _ { i } \rVert _ { 2 } ~ = ~ \lVert \mathbf { p } _ { i } \rVert _ { 2 }$ Thus the gate compares the token’s protected-coordinate magnitude with the discovery statistics $( \tilde { b } , \sigma _ { b } )$ . Tokens with larger protected-coordinate magnitude receive stronger steering, while low-bias tokens stay closer to their original directions.

Geodesic update. Set $\beta _ { i } = \alpha g _ { i }$ and update the token direction by

$$
\widehat { \mathbf { h } } _ { i } ^ { \mathrm { s t e e r e d } } = \mathrm { S l e r p } ( \widehat { \mathbf { h } } _ { i } , \widehat { \mathbf { h } } _ { i } ^ { \mathrm { t a r g e t } } ; \beta _ { i } ) ,\tag{4}
$$

then restore the original radius:

$$
\mathbf { h } _ { i } ^ { \mathrm { s t e e r e d } } = r _ { i } \widehat { \mathbf { h } } _ { i } ^ { \mathrm { s t e e r e d } } .\tag{5}
$$

When $\beta _ { i } = 0$ the direction is unchanged, while $\beta _ { i } = 1$ reaches the projected target. Larger values continue along the same geodesic, allowing the steering strength α to control how aggressively the protected-coordinate component is reduced.

The following proposition records the mechanical guarantees of the update: GGSS removes the learned protected-attribute coordinates by an orthogonal projection in the steering coordinates, and the final spherical update preserves the original token norm exactly. Whether the steering reduces bias is an empirical question, addressed in §4.

Proposition 1. For the basis $\mathbf { V } _ { \mathrm { b i a s } }$ defined in $E q . ( 1 )$ and the token update defined in Eqs. $( 2 ) \mathrm { - } ( 5 ) .$ the projected coordinate $\mathbf { t } _ { i } ^ { \mathrm { c l e a n } }$ is the Euclidean projection $o f \mathbf { t } _ { i }$ onto the set ofvectors whose protectedattribute coordinates are zero:

$$
\mathbf { t } _ { i } ^ { \mathrm { c l e a n } } = \arg \operatorname* { m i n } _ { \mathbf { z } \in \mathbb { R } ^ { D } } \left\{ \| \mathbf { z } - \mathbf { t } _ { i } \| _ { 2 } ^ { 2 } : \mathbf { V } _ { \mathrm { b i a s } } ^ { \top } \mathbf { z } = \mathbf { 0 } \right\} .
$$

Moreover, $i f \widehat { \mathbf { h } } _ { i } ^ { \mathrm { t a r g e t } } \neq - \widehat { \mathbf { h } } _ { i } ,$ , with the convention $\mathrm { S l e r p } ( \mathbf { p } , \mathbf { p } ; \beta ) = \mathbf { p } ,$ , then

$$
\| \mathbf { h } _ { i } ^ { \mathrm { s t e e r e d } } \| _ { 2 } = \| \mathbf { h } _ { i } \| _ { 2 } .
$$

The proof is given in Appendix $\mathsf { A } . 3$

The proposition shows that GGSS removes the learned protected-attribute coordinates by an orthogonal projection and preserves the token norm after the spherical update.

The steered tokens are reshaped back into the activation tensor and passed to the remaining layers of the frozen VLM. Because the gate is tokendependent, low-bias tokens are changed less, while tokens with larger protected-coordinate magnitude are steered more strongly.

## 4 Experiments

## 4.1 Setup

Models. We evaluate GGSS on four generative VLMs spanning three model families and a range of parameter scales: Pixtral-12B (12B) (Agrawal et al., 2024), LLaVA-1.6-Vicuna-7B (7B), LLaVA-1.6-Mistral-7B (7B) (Liu et al., 2024), and Qwen3- VL-4B-Instruct (4B) (Bai et al., 2025). For every method, we intervene at the same late vision-tolanguage projection layer so that differences reflect the steering rule rather than the hook location. Checkpoint identifiers, exact hook paths, activations, and results for additional hook sites appear in Appendix B.3.

Counterfactual images. The counterfactual image sets are not generated by us: we use the 480 real-photograph, face-only counterfactual images released with the REFLECT/FOCUS dataset (Chen et al., 2026), covering 6 occupations × 8 source identities × 5 perceived races × 2 genders. Each identity’s variants are pixel-aligned, so activation differences isolate the protected attribute; discovery occupations are disjoint from the evaluated occupation (Protocol below).

Tasks and metrics. We evaluate with three bias protocols on categorical and pairwise attributes. For each counterfactual image, the probes ask:

• MCQ (race, multiple choice; REFLECT (Chen et al., 2026)): a salary/education multiplechoice question about the pictured person, scored by the Jensen–Shannon divergence of answer distributions across races. The metric is mean\_race\_jsd×10<sup>3</sup>.

• 2AFC (race, pairwise; REFLECT (Chen et al., 2026)): a two-alternative forced choice (“which of two versions of the same person has higher income”) on race-paired images. The metric is race\_bias\_std.

• Nurse/Doctor (gender, classification): a singleword occupation query on nurse/doctor images. The metric is the gender gap in correct classification, |P(nurse|man) − P(nurse|woman)|.

Two of the four models (LLaVA-1.6-Vicuna-7B and LLaVA-1.6-Mistral-7B) exhibit zero baseline

2AFC bias. Steered values are reported for completeness but do not contribute to the average. We also report accuracy on MMStar (Chen et al., 2024), a 1,500-question multiple-choice benchmark of general multimodal capability spanning six dimensions from coarse/fine-grained perception to reasoning, math, and science, scored by exact matching with VLMEvalKit (Duan et al., 2024); perdimension breakdown in Appendix B.6.

Baselines. This work re-implements and adapts existing inference-time debiasers from CLIP- and LM-era settings to generative VLMs. These methods were originally developed for static word embeddings, BERT-style encoders, or CLIP dualencoders; here, we make them target the same late vision-to-language projection layer for a controlled comparison. Concretely, we compare GGSS against ten external steering baselines spanning four families:

• INLP (Ravfogel et al., 2020): iterative linearprobe null-space stacking, ported from wordembedding debiasing, with Euclidean and spherical variants.

• MeanDiff: classifier-guided mean-shift correction in the style of Zou et al. (2023) probing. We use both RBF-SVM and multinomial logistic-regression probes, each with Euclidean and spherical variants trained on pooled visual activations.

• BendVLM (Gerych et al., 2024): we adapt BendVLM’s CLIP-embedding retrieval and Lagrangian equalization from the shared text– image embedding space to generative-VLM hidden activations, evaluating both Euclidean and spherical variants.

• LEACE (Belrose et al., 2023): the closed-form least-squares concept eraser, fit with the authors’ official implementation, as a direct spherical port and combined with our calibrated gate (Appendix A.5).

We additionally evaluate a prompt-space family: the identical probes with a fairness instruction appended to every prompt, with no steering (§4.2). All adapted implementations are released together with this paper as a benchmark suite for inferencetime debiasing of generative VLMs.

Protocol. For every (method, model) pair we sweep α ∈ {0.25, 0.5, 0.75, 1.0, 1.5} across the bias tasks and pick a single α that minimizes the unweighted mean per-task % change relative to the unsteered baseline (“best-avg-α” protocol). Only tasks with non-zero baselines contribute to the average. All bias metrics, the $\mathrm { A v g } \Delta \%$ column, and MMStar for that method and model are reported at this selected operating point. For GGSS we fix κ = 5 and $g _ { \mathrm { f l o o r } } = 0 . 3$ throughout, with sensitivity analyses in §4.3 and Appendix B.2.

<table><tr><td></td><td colspan="4">Pixtral-12B</td><td colspan="4">LLaVA-1.6-Vicuna-7B</td><td colspan="4">LLaVA-1.6-Mistral-7B</td><td colspan="4">Qwen3-VL-4B</td></tr><tr><td>Method</td><td>MCQ  $\times 1 0 ^ { 3 } \downarrow$ </td><td>2AFC ↓</td><td>N/D ↓</td><td> $\mathbf { A v } \mathbf { g }$  ∆%↓</td><td> $\mathbf { M C Q }$   $\times 1 0 ^ { 3 } \downarrow$ </td><td>2AFC ↓</td><td>N/D ↓</td><td>Avg ∆%↓</td><td>MCQ ×103 ↓</td><td>2AFC ↓</td><td>N/D ↓</td><td> $\mathbf { A v } \mathbf { g }$  ∆%↓</td><td>MCQ  $\times 1 0 ^ { 3 } \downarrow$ </td><td>2AFC ↓</td><td>N/D ↓</td><td>Avg ∆%↓</td></tr><tr><td>Baseline</td><td>25.75</td><td>0.243</td><td>0.425</td><td>0%</td><td>8.70</td><td>0.000</td><td>0.600</td><td>0%</td><td>26.87</td><td>0.000</td><td>0.625</td><td>0%</td><td>14.65</td><td>0.368</td><td>0.400</td><td>0%</td></tr><tr><td colspan="2">INLP (Ravfogel et al., 2020)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>10.52</td><td></td><td></td><td></td><td>5.71</td><td></td><td></td><td></td></tr><tr><td>INLP (Eucl.)</td><td>15.05 -42%</td><td>0.131 -46%</td><td>0.350 -18%</td><td>-35%</td><td>1.87 -79%</td><td>0.000</td><td>0.625 +4%</td><td>-37%</td><td>-61%</td><td>0.000</td><td>0.375 -40%</td><td>-50%</td><td>-61%</td><td>0.279 -24%</td><td>0.375 -6%</td><td>-31%</td></tr><tr><td>INLP (sph.)</td><td>0.47 -98%</td><td>0.297 +22%</td><td>0.350 -18%</td><td>-31%</td><td>2.05 -76%</td><td>0.000</td><td>0.025 -96%</td><td>-86%</td><td>9.87 -63%</td><td>0.000</td><td>0.050 -92%</td><td>-78%</td><td>11.00 -25%</td><td>0.238 -35%</td><td>0.050 -88%</td><td>-49%</td></tr><tr><td colspan="2">MeanDiff</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MeanDiff-SVM (Eucl.)</td><td>25.75 +0%</td><td>0.247 +2%</td><td>0.425 +0%</td><td>+1%</td><td>0.86 -90%</td><td>0.000</td><td>0.600 +0%</td><td>-45%</td><td>11.05 -59%</td><td>0.000</td><td>0.625 +0%</td><td>-29%</td><td>5.24 -64%</td><td>0.368 +0%</td><td>0.375 -6%</td><td>-23%</td></tr><tr><td>MeanDiff-SVM (sph.)</td><td>17.73 -31%</td><td>0.248 +2%</td><td>0.425 +0%</td><td>-10%</td><td>2.76 -68%</td><td>0.000</td><td>0.625 +4%</td><td>-32%</td><td>13.67 -49%</td><td>0.000</td><td>0.625 +0%</td><td>-25%</td><td>14.02 -4%</td><td>0.412 +12%</td><td>0.375 -6%</td><td>+0%</td></tr><tr><td>MeanDiff-LR (Eucl.)</td><td>25.75 +0%</td><td>0.247 +2%</td><td>0.425 +0%</td><td>+1%</td><td>3.09 -64%</td><td>0.000</td><td>0.600 +0%</td><td>-32%</td><td>12.14 -55%</td><td>0.000</td><td>0.625 +0%</td><td>-27%</td><td>5.77 -61%</td><td>0.368 +0%</td><td>0.375 -6%</td><td>-22%</td></tr><tr><td>MeanDiff-LR (sph.)</td><td>25.75 +0%</td><td>0.239 -2%</td><td>0.425 +0%</td><td>-1%</td><td>6.15 -29%</td><td>0.000</td><td>0.625 +4%</td><td>-13%</td><td>21.21 -21%</td><td>0.000</td><td>0.625 +0%</td><td>-11%</td><td>13.56 -7%</td><td>0.400 +9%</td><td>0.400 +0%</td><td>+0%</td></tr><tr><td colspan="2">BendVLM (Gerych et al., 2024)</td><td>0.267</td><td>0.125</td><td></td><td></td><td></td><td></td><td></td><td>10.81</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BendVLM (Eucl.)</td><td>13.03 -49%</td><td>+10%</td><td>-71%</td><td>-37%</td><td>2.71 -69%</td><td>0.000</td><td>0.625 +4%</td><td>-32%</td><td>-60%</td><td>0.000</td><td>0.625 +0%</td><td>-30%</td><td>5.31 -64%</td><td>0.374 +2%</td><td>0.350 -13%</td><td>-25%</td></tr><tr><td>BendVLM (sph.)</td><td>112.03 +335%</td><td>0.000 -100%</td><td>0.000 -100%</td><td>+45%</td><td>0.00 -100%</td><td>0.452</td><td>*</td><td>-100%†</td><td>2.83 -89%</td><td>0.000</td><td>0.550 -12%</td><td>-51%</td><td>2.86 -80%</td><td>0.152 -59%</td><td>0.425 +6%</td><td>-44%</td></tr><tr><td>GGSS (Ours)</td><td>16.85 -35%</td><td>0.096 -61%</td><td>0.125 -71%</td><td>-55%</td><td>1.36 -84%</td><td>0.000</td><td>0.025 -96%</td><td>-90%</td><td>8.48 -68%</td><td>0.000</td><td>0.050 -92%</td><td>-80%</td><td>5.10 -65%</td><td>0.154 -58%</td><td>0.175 -56%</td><td>-60%</td></tr></table>

Table 1: Main results against external baselines. Each (method, model) row uses a single best-avg-α selected from {0.25, 0.5, 0.75, 1.0, 1.5}. A gray “–” on the second line means the baseline is zero, so percent change is undefined. <sup>∗</sup> marks runs where the steered model produced no parseable answers (BendVLM (sph.) collapses generation there, consistent with Table 2; likely a difficulty of porting a CLIP-space method to generative activations, not a flaw of the original method). <sup>†</sup> average over a single task, excluded from cross-method comparison. Avg ∆% averages over non-zero-baseline tasks: three on Pixtral-12B and Qwen3-VL-4B, two on the LLaVA models, so it is not strictly comparable across models.
<table><tr><td></td><td colspan="4">Race-task steering</td><td colspan="4">Gender-task steering</td><td></td></tr><tr><td>Method</td><td>Pixtral</td><td>Vicuna</td><td>Mistral</td><td>Qwen3</td><td>Pixtral</td><td>Vicuna</td><td>Mistral</td><td>Qwen3</td><td>Avg.</td></tr><tr><td>Baseline</td><td>53.6 0.0%</td><td>37.3 0.0%</td><td>38.5 0.0%</td><td>61.5 0.0%</td><td>53.6 0.0%</td><td>37.3 0.0%</td><td>38.5 0.0%</td><td>61.5 0.0%</td><td>47.7 0.0%</td></tr><tr><td>INLP (Eucl.)</td><td>52.1 -1.5%</td><td>37.2 -0.1%</td><td>36.4 -2.1%</td><td>62.1 +0.6%</td><td>53.3 -0.3%</td><td>37.6 +0.3%</td><td>38.0 -0.5%</td><td>62.0 +0.5%</td><td>47.3 -0.4%</td></tr><tr><td>INLP (sph.)</td><td>47.5 -6.1%</td><td>36.8 -0.5%</td><td>38.4 -0.1%</td><td>62.4 +0.9%</td><td>50.3 -3.3%</td><td>37.1 -0.2%</td><td>38.6 +0.1%</td><td>61.3 -0.2%</td><td>46.5 -1.2%</td></tr><tr><td>MeanDiff-SVM (Eucl.)</td><td>53.3 -0.3%</td><td>37.1 -0.2%</td><td>39.0 +0.5%</td><td>61.4 -0.1%</td><td>53.1 -0.5%</td><td>37.4 +0.1%</td><td>38.2 -0.3%</td><td>61.9 +0.4%</td><td>47.7 -0.1%</td></tr><tr><td>MeanDiff-SVM (sph.)</td><td>53.2 -0.4%</td><td>37.3 +0.0%</td><td>39.2 +0.7%</td><td>61.8 +0.3%</td><td>53.1 -0.5%</td><td>37.2 -0.1%</td><td>39.0 +0.5%</td><td>61.8 +0.3%</td><td>47.8 +0.1%</td></tr><tr><td>MeanDiff-LR (Eucl.)</td><td>53.6 +0.0%</td><td>36.9 -0.4% 37.1</td><td>38.7 +0.2%</td><td>61.4 -0.1%</td><td>53.7 +0.1%</td><td>37.2 -0.1%</td><td>38.8 +0.3%</td><td>61.8 +0.3%</td><td>47.8 +0.0% 47.7</td></tr><tr><td>MeanDiff-LR (sph.)</td><td>53.0 -0.6% 53.0</td><td>-0.2% 37.1</td><td>39.4 +0.9% 39.0</td><td>61.7 +0.2% 61.6</td><td>53.2 -0.4% 45.1</td><td>36.6 -0.7% 37.8</td><td>38.8 +0.3% 38.6</td><td>61.9 +0.4% 61.7</td><td>+0.0% 46.7</td></tr><tr><td>BendVLM (Eucl.)</td><td>-0.6% 6.9</td><td>-0.2% 2.9</td><td>+0.5% 39.5</td><td>+0.1% 59.0</td><td>-8.5% 5.3</td><td>+0.5% 2.6</td><td>+0.1% 39.8</td><td>+0.2% 59.3</td><td>-1.0% 26.9</td></tr><tr><td></td><td>-46.7% 47.3</td><td>-34.4% 35.0</td><td>+1.0% 38.9</td><td>-2.5% 60.8</td><td>-48.3% 46.5</td><td>-34.7% 37.5</td><td>+1.3% 39.0</td><td>-2.2% 61.8</td><td>-20.8% 45.9</td></tr><tr><td>Pooled SVD (Eucl.) Pooled SVD (sph.)</td><td>-6.3% 53.5</td><td>-2.3% 38.2</td><td>+0.4% 38.5</td><td>-0.7% 61.7</td><td>-7.1% 53.3</td><td>+0.2% 37.2</td><td>+0.5% 38.4</td><td>+0.3% 61.8</td><td>-1.9% 47.8</td></tr><tr><td>Per-token SVD (Eucl.)</td><td>-0.1% 42.0</td><td>+0.9% 37.2</td><td>+0.0% 37.7</td><td>+0.2% 60.2</td><td>-0.3% 22.1</td><td>-0.1% 36.9</td><td>-0.1% 38.5</td><td>+0.3% 61.0</td><td>+0.1% 42.0</td></tr><tr><td>Per-token SVD (sph.)</td><td>-11.6% 50.1 -3.5%</td><td>-0.1% 36.2 -1.1%</td><td>-0.8% 38.1</td><td>-1.3% 61.1</td><td>-31.5% 50.9</td><td>-0.4% 37.4</td><td>+0.0% 38.2</td><td>-0.5% 61.2</td><td>-5.8% 46.6</td></tr><tr><td>GGSS (ours)</td><td>53.2 -0.4%</td><td>37.8 +0.5%</td><td>-0.4% 38.9 +0.4%</td><td>-0.4% 61.6 +0.1%</td><td>-2.7% 53.3</td><td>+0.1% 37.1 -0.2%</td><td>-0.3% 38.4 -0.1%</td><td>-0.3% 62.1</td><td>-1.1% 47.8</td></tr></table>

Table 2: MMStar capability preservation at best-avg-α. Best-avg-α choices are listed in Appendix B.2.

<table><tr><td colspan="3">Component ablation on Qwen3-VL-4B, MCQ (race)</td></tr><tr><td>Variant</td><td>JSD×1k↓</td><td>Bias Red. % ↑</td></tr><tr><td>Baseline (unsteered)</td><td>14.646</td><td></td></tr><tr><td>Pooled SVD (Eucl.; hard proj., no gate)</td><td>4.261</td><td>70.9</td></tr><tr><td>Pooled SVD (sph.; hard proj., no gate)</td><td>5.612</td><td>61.7</td></tr><tr><td>GGSS w/o gate  $( g _ { i } \equiv 1 )$ </td><td>5.612</td><td>61.7</td></tr><tr><td>GGSS w/o Slerp (hard proj. + gate)</td><td>4.533</td><td>69.1</td></tr><tr><td>GGSS (full)</td><td>2.414</td><td>83.5</td></tr></table>

Table 3: Component ablation of GGSS on Qwen3-VL-4B, MCQ salary (race) at $\alpha = 1 . 0 $

<table><tr><td colspan="5">Sensitivity to  $( \kappa , g _ { \mathrm { f l o o r } } )$  on Qwen3-VL-4B, MCQ</td></tr><tr><td>gfloor \κ</td><td>1</td><td>2</td><td>5</td><td>10</td></tr><tr><td>0.0</td><td>4.851</td><td>4.578</td><td>10.971</td><td>9.196</td></tr><tr><td>0.3</td><td>4.578</td><td>5.781</td><td>2.414</td><td>4.533</td></tr><tr><td>0.5</td><td>6.128</td><td>6.714</td><td>4.578</td><td>4.069</td></tr></table>

Table 4: Sensitivity of GGSS to gate sharpness κ and floor $g _ { \mathrm { f l o o r } }$ on Qwen3-VL-4B, MCQ salary (race) at $\alpha = 1 . 0$ . Entries are JSD×1k; lower is better.

Discovery uses counterfactual image sets whose identities are held out from the evaluation occupation. For the race-based tasks we discover on {cook, doctor, lawyer, nurse, teacher}, and for the gender-based Nurse/Doctor task we discover on {cook, lawyer, teacher} while holding out both nurse and doctor. MCQ and 2AFC are scored on the SocialCounterfactuals probe suite (Howard et al., 2024), which does not share identities with the discovery pool.

Reading the tables. Tables 1 and 2 should be read as operating-point comparisons: for each method and model, the same selected α is used for all bias metrics and for MMStar. They separate two questions that are often conflated: whether a steering rule reduces demographic sensitivity, and whether the same operating point preserves general multimodal reasoning. A useful method must do both.

## 4.2 Main Results

Table 1 reports best-avg-α metrics for every method across all models and bias tasks; GGSS is the single most reliable method in the comparison.

Ranking. GGSS attains the lowest Avg ∆% among all external baselines on all 4 models. The gaps are −55% vs. −37% on Pixtral-12B, where BendVLM (Eucl.) is the strongest external baseline, then −90% vs. −86% on LLaVA-1.6-Vicuna-7B, −80% vs. −78% on LLaVA-1.6-Mistral-7B, and −60% vs. −49% on Qwen3-VL-4B, all three against INLP (sph.), the strongest competitor overall.

Magnitude. GGSS produces substantial bias reductions: up to −96% on Nurse/Doctor (LLaVA-$1 . 6 \mathrm { { - V i c u n a - } } 7 \mathrm { { B } \colon 0 . 6 0 0  0 . 0 2 5 ) }$ , −84% on MCQ (LLaVA-1.6-Vicuna- $7 \mathrm { B } \colon 8 . 7 0  1 . 3 6 )$ , and −61% on 2AFC (Pixtral-12B: 0.243 → 0.096).

MCQ and Nurse/Doctor measure different output formats—categorical answer distributions and binary occupation classification—yet GGSS reduces both without changing the selected operating point for a given model; on the two LLaVA models the zero-baseline 2AFC entries serve as sanity checks.

Capability preservation (MMStar). Table 2 reports MMStar results at each method’s best-avg-α (the same α used to report its bias metrics in Table 1) across all steering settings. GGSS stays within ±0.6 p.p. of baseline across these evaluations: adaptive geodesic steering preserves general capability. A steering method that reduces benchmark bias by collapsing general VLM capability is not useful as a deployment primitive, and several baselines suffer exactly such drops; Bend-VLM (sph.), for example, collapses on Pixtral and LLaVA-Vicuna. GGSS’s MMStar changes remain small under both race and gender steering: the gate confines the intervention to directions that carry the targeted bias and leaves the remaining visual tokens largely untouched.

<table><tr><td>Model</td><td>GGSS vs. unsteered (p)</td><td>MMStar ∆pp (race / gender)</td><td>McNemar p (race / gender)</td></tr><tr><td>Pixtral-12B</td><td> $< 1 0 ^ { - 4 }$ </td><td> $- 0 . 4 0 / - 0 . 3 3$ </td><td>0.58/0.49</td></tr><tr><td>LLaVA-Vicuna-7B</td><td> $0 . 0 6 8 ^ { a }$ </td><td> $+ 0 . 4 7 / - 0 . 2 0$ </td><td>0.58 / 0.77</td></tr><tr><td>LLaVA-Mistral-7B</td><td>0.007</td><td> $+ 0 . 4 0 / - 0 . 0 7$ </td><td>0.51/1.00</td></tr><tr><td>Qwen3-VL-4B</td><td>0.002</td><td> $+ 0 . 1 3 / + 0 . 6 0$ </td><td>0.86 / 0.09</td></tr></table>

Table 5: Significance summary at the paper’s operating points. Column 2: paired sign-flip permutation test of GGSS vs. the unsteered model, combined across non-zero-baseline bias tasks (<sup>a</sup>on LLaVA-Vicuna the Nurse/Doctor task alone has $p < 1 0 ^ { - 4 } )$ . Columns 3–4: MMStar paired per-question statistics under race / gender steering. Protocol details in Appendix B.4.

Statistical significance. Decoding is greedy and deterministic, so the relevant uncertainty is sampling over evaluation items. For every reported cell we compute 95% bootstrap CIs (10,000 resamples, over items and identities), paired sign-flip permutation tests, and exact McNemar tests on MMStar (Appendix B.4). Table 5 summarizes: GGSS’s reductions are significant on three of four backbones (on LLaVA-Vicuna the Nurse/Doctor reduction alone has $p < 1 0 ^ { - 4 } )$ , and its MMStar changes are statistically indistinguishable from the unsteered model in all eight runs, whereas the strongest competitor, INLP (sph.), significantly damages Pixtral’s MMStar (−6.1 pp race, $p < 1 0 ^ { - 4 } )$ .

<table><tr><td>Method</td><td>Pix- tral</td><td>Qwen3- VL</td><td>Mis- tral</td><td>Vi- cuna</td><td>mean</td><td>worst</td></tr><tr><td>GGSS (paper)</td><td>-55.3</td><td>-59.9</td><td>-80.2</td><td>-90.1</td><td>-71.4</td><td>-55.3</td></tr><tr><td>LEACE + our gate</td><td>-72.2</td><td>-30.8</td><td>-66.9</td><td>-83.9</td><td>-63.5</td><td>-30.8</td></tr><tr><td>INLP (sph.)</td><td>-31.2</td><td>-49.3</td><td>-77.6</td><td>-86.1</td><td>-61.1</td><td>-31.2</td></tr><tr><td>LEACE (sph.)</td><td>-70.6</td><td>-26.4</td><td>-51.5</td><td>-45.4</td><td>-48.5</td><td>-26.4</td></tr></table>

Table 6: Added baselines. Avg ∆% at each method’s best-avg-α (bias tasks with non-zero baseline; lower is better), with the four-model mean and worst case. INLP (sph.) is the strongest baseline from Table 1, shown for reference. Full details in Appendix B.10.

Held-out α selection. Table 1 selects α on the probe suite it reports, identically for every method, so no method is advantaged, but absolute reductions may be optimistic. Under a cross-fitted protocol selecting α on one identity fold and evaluating on the other (Appendix B.5), GGSS stays near the top on every model, matches the paper’s α on six of eight folds, and reduces bias in every held-out fold.

Additional baseline families. Table 6 adds LEACE (Belrose et al., 2023) at the same layer under the same protocol. Ported this way, LEACE is a serious baseline (it wins the Pixtral column while preserving MMStar there) but is less consistent across backbones (−26% to −31% on Qwen3- VL), while GGSS keeps the best four-model mean and worst case. Prompt-space mitigation (a fairness instruction appended to every prompt) is inconsistent and can even amplify bias (e.g., Vicuna MCQ +85%); full results in Appendix B.9.

Debiasing, not blanket removal. A steering rule could reduce measured bias by deleting demographic perception outright; GGSS does not. Steering one attribute leaves recognition of the other attribute intact on all four backbones, all six MM-Star dimensions stay within binomial noise, and MME (Qwen3-VL) moves only −0.4% / −1.8% with no subcategory collapsing. The trade-off with the targeted attribute is controlled by α: at $\alpha = 0 . 5$ on Qwen3-VL, race recognition stays within 4 pp of baseline while 55% of the MCQ reduction is already realized, so tasks that need the attribute can run at moderate α or gate steering off (Appendix B.8).

## 4.3 Ablations and Sensitivity

Component and sensitivity. Table 3 isolates the contribution of each GGSS ingredient on a representative setting (Qwen3-VL-4B, MCQ salary, $\alpha { = } 1 . 0 )$ . Disabling the gate (g<sub>i</sub> ≡ 1) gives the same JSD as the ungated spherical hard-projection variant (5.61). Adding the gate without Slerp improves this value from 5.61 to 4.53 (−69% from baseline), showing that selective gating alone already helps. The full method, combining gating and Slerp interpolation, achieves 2.41 (−84%), a further 14 pp improvement in this setting.

Table 4 sweeps gate sharpness $\kappa \in \{ 1 , 2 , 5 , 1 0 \}$ and floor $g _ { \mathrm { f l o o r } } \in \{ 0 . 0 , 0 . 3 , 0 . 5 \}$ . All twelve settings reduce bias well below the unsteered baseline $( 1 4 . 6 5 )$ ; the best reduction (−84%) occurs at $\kappa = 5 , \ g _ { \mathrm { f l o o r } } = 0 . 3$ . High κ with $g _ { \mathrm { f l o o r } } = 0$ can overselect, but a positive floor recovers performance.

Geometry isolated at matched gate. Toggling Euclidean versus spherical steering at matched gate condition on all four backbones, with paired signflip tests (Appendix B.11), shows the toggle is within sampling noise at the 4B–7B scales but decisive at 12B: on Pixtral the spherical form is significantly better under both toggles $( p = 0 . 0 3 1 /$ 0.015), both Euclidean forms significantly damage MMStar there $( - 6 . 3 \mathrm { p p } , p \ < \ 1 0 ^ { - 8 } )$ while the spherical forms preserve it everywhere, and over-steering exposes failure modes the geodesic never showed (9× overshoot; a degenerate cell with 72/80 unparseable). The ablations thus attribute robustness that grows with model scale to on-sphere steering, and selectivity and controllability to the gate.

## 5 Conclusion

We introduced GGSS, an inference-time debiasing framework for generative VLMs that discovers counterfactual spherical bias subspaces, applies a calibrated token gate, and steers with normpreserving Slerp. Across four generative VLMs and three bias protocols, GGSS attains the lowest Avg ∆% on all 4 models against ten external steering baselines and prompt-based mitigation, significant on three of four backbones, while keeping MMStar indistinguishable from the unsteered model. Matched-gate ablations attribute robustness at scale to on-sphere steering and controllable selectivity to the gate, supporting adaptive geodesic steering as a practical primitive for inference-time bias mitigation.

## Limitations

Scope. Our evaluation covers four generative VLMs, three demographic-bias protocols, and MM-Star, but does not establish that GGSS generalizes to all architectures, languages, domains, or bias types. We focus on perceived race and gender in image-conditioned settings; other protected attributes, intersectional groups, multilingual prompts, and open-ended uses remain future work. The subspace construction may also extend to nondemographic bias axes such as political leaning, whose embedding-space structure has been studied in text models (Sun et al., 2025b).

Calibration. GGSS still requires selecting a steering strength. We choose one best-avg-α per (method, model) from {0.25, 0.5, 0.75, 1.0, 1.5} using held-out bias measurements, and the optimal value is model-dependent. A tuning-free rule, potentially using the bias-norm statistics <sup>˜</sup>b and $\sigma _ { b }$ to set token-level strengths, remains an open direction.

Deployment. Lower benchmark bias should not be interpreted as complete fairness or safety. GGSS changes activations at inference time, but does not remove biased knowledge from model parameters or guarantee robustness under distribution shift. Misconfigured steering may also suppress useful demographic information or affect tasks beyond our capability checks: at the bias-minimizing steering strength, the targeted attribute’s reportability on face-only close-ups is strongly reduced (Appendix B.8), so tasks that legitimately require the attribute, such as requested demographic descriptions or clinical documentation, should operate at a moderate α chosen from the trade-off curve, or gate steering off. Deployment should include broader auditing, human review where appropriate, and monitoring for residual or newly introduced harms.

## Ethics Statement

Intended use and dual use. GGSS is designed to reduce demographic bias in deployed VLMs without retraining. The same mechanism, however, is dual-use: a steering subspace can be aimed at amplifying demographic sensitivity as easily as at reducing it, and inference-time manipulation of model behavior is demonstrably exploitable (Liu et al., 2026). At high steering strength the intervention also shades from debiasing into attribute removal (Appendix B.8). We release the tradeoff measurements needed to choose an operating point deliberately; deployments should measure both bias and attribute retention on the target task before fixing a steering strength.

Demographic labels. The counterfactual datasets we use annotate perceived race and binary gender as assigned by dataset curators. These labels are operational constructs for measuring output disparities; they do not capture self-identification, intersectional identity, or the full diversity of human appearance, and results should not be read as claims about any individual’s identity.

Data and licensing. All images come from publicly released research datasets (RE-FLECT/FOCUS, SocialCounterfactuals) used within their stated licenses; we generate no new images of people and release no new personal data. Artifact documentation and intended-use notes accompany the released benchmark suite (Appendix B.14).

Residual risk. Reduced benchmark bias is not fairness. Steering does not remove biased knowledge from model parameters, may behave differently under distribution shift, and covers only the attributes and tasks we measure. Systems using GGSS in consequential settings should retain human oversight and independent auditing.

## References

Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Baptiste Bout, Devendra Chaplot, Jessica Chudnovsky, Diogo Costa, Baudouin De Monicault, Saurabh Garg, Theophile Gervet, Soham Ghosh, Amélie Héliou, Paul Jacob, Albert Q. Jiang, Kartik Khandelwal, Timothée Lacroix, Guillaume Lample, Diego Las Casas, Thibaut Lavril, and 23 others. 2024. Pixtral 12b. arXiv preprint arXiv:2410.07073.

Na Min An, Yoonna Jang, Yusuke Hirota, Ryo Hachiuma, Isabelle Augenstein, and Hyunjung Shim. 2026. Interpretable debiasing of vision-language models for social fairness. arXiv preprint arXiv:2602.24014.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, and 45 others. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.

Debodeep Banerjee, Stefano Teso, Burcu Sayin, and Andrea Passerini. 2024. Learning to guide human decision makers with vision-language models. arXiv preprint arXiv:2403.16501.

Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, and Stella Biderman. 2023. LEACE: Perfect linear concept erasure in closed form. Advances in Neural Information Processing Systems, 36:66044–66063.

Die Chen, Zhiwen Li, Mingyuan Fan, Cen Chen, Wenmeng Zhou, Yanhao Wang, and Yaliang Li. 2025. Growth inhibitors for suppressing inappropriate image concepts in diffusion models. In International Conference on Learning Representations, volume 2025, pages 79164–79184.

Haodong Chen, Qiang Huang, Jiaqi Zhao, Qiuping Jiang, Xiaojun Chang, and Jun Yu. 2026. Measuring social bias in vision-language models with faceonly counterfactuals from real photos. arXiv preprint arXiv:2601.06931.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. 2024. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330.

Harry Cheng, Yangyang Guo, Qingpei Guo, Ming Yang, Tian Gan, Weili Guan, and Liqiang Nie. 2025. Social debiasing for fair multi-modal llms. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1740–1750.

Ching-Yao Chuang, Varun Jampani, Yuanzhen Li, Antonio Torralba, and Stefanie Jegelka. 2023. Debiasing vision-language models via biased prompts. arXiv preprint arXiv:2302.00070.

Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, Dahua Lin, and Kai Chen. 2024. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models. In Proceedings ofthe 32nd ACM International Conference on Multimedia, MM ’24, page 11198–11201.

Yingchaojie Feng, Yiqun Sun, Yandong Sun, Minfeng Zhu, Qiang Huang, Anthony Kum Hoe Tung, and Wei Chen. 2025. Don’t reinvent the wheel: Efficient instruction-following text embedding based on guided space transformation. In Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 24511–24525.

Walter Gerych, Cassandra Parent, Quinn Perian, Rafiya Javed, Justin Solomon, and Marzyeh Ghassemi. 2026. Wring out the bias: A rotation-based alternative to projection debiasing. In The Fourteenth International Conference on Learning Representations.

Walter Gerych, Haoran Zhang, Kimia Hamidieh, Eileen Pan, Maanas K Sharma, Tom Hartvigsen, and Marzyeh Ghassemi. 2024. Bendvlm: Test-time debiasing of vision-language embeddings. Advances in Neural Information Processing Systems, 37:62480– 62502.

Kshitish Ghate, Tessa Charlesworth, Mona Diab, and Aylin Caliskan. 2025. Biases propagate in encoderbased vision-language models: A systematic analysis from intrinsic measures to zero-shot retrieval outcomes. In Findings of the Association for Computational Linguistics: ACL 2025, pages 18562–18580.

Melissa Hall, Laura Gustafson, Aaron Adcock, Ishan Misra, and Candace Ross. 2023. Vision-language models performing zero-shot tasks exhibit disparities between gender groups. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, pages 2778–2785.

Yusuke Hirota, Yuta Nakashima, and Noa Garcia. 2022. Quantifying societal bias amplification in image captioning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13440–13449.

Phillip Howard, Avinash Madasu, Tiep Le, Gustavo Lujan Moreno, Anahita Bhiwandiwalla, and Vasudev Lal. 2024. Socialcounterfactuals: Probing and mitigating intersectional social biases in vision-language models with counterfactual examples. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11975–11985.

Jen-tse Huang, Jiantong Qin, Jianping Zhang, Youliang Yuan, Wenxuan Wang, and Jieyu Zhao. 2025. Visbias: Measuring explicit and implicit social biases in vision language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 17981–18004.

Robert Huben, Hoagy Cunningham, Logan Riggs Smith, Aidan Ewart, and Lee Sharkey. 2024. Sparse autoencoders find highly interpretable features in language models. Proceedings ofthe International Conference on Learning Representations (ICLR).

Taeuk Jang, Hoin Jung, and Xiaoqian Wang. 2025. Target bias is all you need: Zero-shot debiasing of visionlanguage models with bias corpus. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 1935–1946.

Sepehr Janghorbani and Gerard De Melo. 2023. Multimodal bias: Introducing a framework for stereotypical bias assessment beyond gender and race in vision– language models. In Proceedings of the 17th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics, pages 1725–1735.

Hoin Jung, Taeuk Jang, and Xiaoqian Wang. 2024. A unified debiasing approach for vision-language models across modalities and tasks. Advances in Neural Information Processing Systems, 37:21034–21058.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023. Inferencetime intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems, 36:41451–41530.

Zhiwen Li, Die Chen, Mingyuan Fan, Cen Chen, Yaliang Li, Yanhao Wang, and Wenmeng Zhou. 2025. Responsible diffusion models via constraining text embeddings within safe regions. In Proceedings of the ACM on Web Conference 2025, pages 1588–1601.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. 2024. Llavanext: Improved reasoning, ocr, and world knowledge.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Advances in neural information processing systems, 36:34892– 34916.

Yuansen Liu, Yixuan Tang, and Anthony Kum Hoe Tung. 2026. Reasoning hijacking: The fragility of reasoning alignment in large language models. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 36646–36665.

Thomas Manzini, Lim Yao Chong, Alan W Black, and Yulia Tsvetkov. 2019. Black is to criminal as caucasian is to police: Detecting and removing multiclass bias in word embeddings. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 615–621.

Kanti V. Mardia and Peter E. Jupp. 2000. Directional statistics. In Wiley Series in Probability and Statistics. Wiley.

Mahdiyar Molahasani, Azadeh Motamedi, Michael Greenspan, Il-Min Kim, and Ali Etemad. 2025. Prism: Reducing spurious implicit biases in visionlanguage models with llm-guided embedding projection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 688–697.

Vishal Narnaware, Ashmal Vayani, Rohit Gupta, Sirnam Swetha, and Mubarak Shah. 2025. Bbq-v: Benchmarking visual stereotype bias in large multimodal models. Preprint, arXiv:2502.08779.

Bo Pang, Tingrui Qiao, Caroline Walker, Chris Cunningham, and Yun Sing Koh. 2025. Cabin: Debiasing vision-language models using backdoor adjustments. In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, IJCAI-25, pages 484–492.

Kiho Park, Yo Joong Choe, and Victor Veitch. 2023. The linear representation hypothesis and the geometry of large language models. arXiv preprint arXiv:2311.03658.

Xavier Pennec. 2006. Intrinsic statistics on Riemannian manifolds: Basic tools for geometric measurements. Journal ofMathematical Imaging and Vision, 25(1):127–154.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. In Proceedings ofthe 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 8748–8763. PMLR.

Neale Ratzlaff, Matthew Lyle Olson, Musashi Hinck, Shao-Yen Tseng, Vasudev Lal, and Phillip Howard. 2024. Debiasing large vision-language models by ablating protected attribute representations. arXiv preprint arXiv:2410.13976.

Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, and Yoav Goldberg. 2020. Null it out: Guarding protected attributes by iterative nullspace projection. In Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics, pages 7237–7256.

Shauli Ravfogel, Michael Twiton, Yoav Goldberg, and Ryan Cotterell. 2022. Linear adversarial concept erasure. In International Conference on Machine Learning, pages 18400–18421.

Candace Ross, Boris Katz, and Andrei Barbu. 2021. Measuring social biases in grounded vision and language embeddings. In Proceedings ofthe 2021 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 998–1008.

Gabriele Ruggeri and Debora Nozza. 2023. A multidimensional study on bias in vision-language models. In Findings of the Association for Computational Linguistics: ACL 2023, pages 6445–6455.

Ashish Seth, Mayur Hemani, and Chirag Agarwal. 2023. Dear: Debiasing vision-language models with additive residuals. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6820–6829.

Leheng Sheng, Changshuo Shen, Weixiang Zhao, Junfeng Fang, Xiaohao Liu, Zhenkai Liang, Xiang Wang, An Zhang, and Tat-Seng Chua. 2026. AlphaSteer: Learning refusal steering with principled null-space constraint. In The Fourteenth International Conference on Learning Representations.

Ken Shoemake. 1985. Animating rotation with quaternion curves. In Proceedings ofthe 12th Annual Conference on Computer Graphics and Interactive Techniques (SIGGRAPH), pages 245–254.

Himanshu Singh, Ziwei Xu, AV Subramanyam, and Mohan Kankanhalli. 2026. Do prompts guarantee safety? mitigating toxicity from llm generations through subspace intervention. arXiv preprint arXiv:2602.06623.

Tejas Srinivasan and Yonatan Bisk. 2022. Worst of both worlds: Biases compound in pre-trained vision-andlanguage models. In Proceedings of the 4th Workshop on Gender Bias in Natural Language Processing (GeBNLP), pages 77–85.

Yandong Sun, Qiang Huang, Ziwei Xu, Yiqun Sun, Yixuan Tang, and Anthony KH Tung. 2025a. One swallow does not make a summer: Understanding semantic structures in embedding spaces. arXiv preprint arXiv:2512.00852.

Yiqun Sun, Qiang Huang, Anthony Kum Hoe Tung, and Jun Yu. 2025b. Prism: A framework for producing interpretable political bias embeddings with politicalaware cross-encoder. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 27719– 27733.

Alex Turner, Lisa Thiergart, David Udell, Gavin Leech, Ulisse Mini, and Monte MacDiarmid. 2023. Activation addition: Steering language models without optimization. arXiv preprint arXiv:2308.10248.

Hao Wang, Yiqun Sun, Pengfei Wei, Lawrence B Hsieh, and Daisuke Kawahara. 2026. Sparse autoencoders as plug-and-play firewalls for adversarial attack detection in vlms. arXiv preprint arXiv:2605.07447.

Sibo Wang, Xiangkui Cao, Jie Zhang, Zheng Yuan, Shiguang Shan, Xilin Chen, and Wen Gao. 2024. Vlbiasbench: A comprehensive benchmark for evaluating bias in large vision-language model. arXiv preprint arXiv:2406.14194.

Xing Wu, Kehong Liu, Jianjia Wang, Junfeng Yao, Bin Deng, Rongqi Lv, and Jun Song. 2024. Candidate evaluation with multimodal data-driven for recruitment. In International Conference on Pattern Recognition, pages 81–96. Springer.

Haoyu Zhang, Yangyang Guo, and Mohan Kankanhalli. 2025. Joint vision-language social bias removal for clip. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 4246–4255.

Kankan Zhou, Eason Lai, and Jing Jiang. 2022. Vlstereoset: A study of stereotypical bias in pre-trained vision-language models. In Proceedings ofthe 2nd Conference ofthe Asia-Pacific Chapter ofthe Associationfor Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 527–538.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, and 2 others. 2023. Representation engineering: A top-down approach to AI transparency. arXiv preprint arXiv:2310.01405.

## A Implementation Details

## A.1 Notation

The main text uses the compact notation $\mathbf { x } _ { u , a } .$ where a is the protected attribute value and u collects the visual factors held fixed. In the racedebiasing implementation, $a = r \in \mathcal { R }$ and $u =$ $( o , b , g )$ , where $o \in \mathcal { O }$ is the occupation, $b \in B _ { o }$ is the base identity, and $g \in { \mathcal { G } }$ is the gender label. Thus

$$
\mathbf { x } _ { u , a } = I _ { o , b } ^ { ( r , g ) } .
$$

The corresponding activation matrix is

$$
\mathbf { H } _ { u , a } = \mathbf { H } _ { o , b } ^ { ( r , g ) } \in \mathbb { R } ^ { T \times D } ,
$$

where $D$ is the hidden dimension, $T$ is the number of discovery visual tokens, and the t-th token activation is $\mathbf { h } _ { o , b , t } ^ { ( r , g ) } \in \mathbb { R } ^ { D }$ . The normalized pooled representative in the main text is

$$
\mathbf { y } _ { u , a } = \nu \left( \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbf { H } _ { u , a , t } \right) = \widehat { \bar { \mathbf { h } } } _ { o , b } ^ { ( r , g ) } .
$$

The within-context Fréchet mean and shift are similarly related by

$$
\begin{array} { r } { \begin{array} { r } { \mathbf { c } _ { u } = \pmb { c } _ { o , b , g } , } \\ { \mathbf { s } _ { u , a } = \mathbf { s } _ { o , b } ^ { ( r , g ) } \quad \mathrm { w h e n } u = ( o , b , g ) , a = r . } \end{array} } \end{array}
$$

For gender debiasing, the roles of r and $g$ are exchanged: $a = g$ and $\boldsymbol { u } = \left( o , b , r \right)$ . The learned basis is denoted by $\mathbf { V } _ { \mathrm { b i a s } }$ throughout the paper. At inference time, the hooked activation tensor has shape

$$
\mathbf { Z } \in \mathbb { R } ^ { n _ { \mathrm { b a t c h } } \times T ^ { \prime } \times D } ,
$$

where $T ^ { \prime }$ is the sequence length at the intervention layer. We use $n _ { \mathrm { b a t c h } }$ , rather than B, to denote batch size. The steering strength is denoted by $\alpha \in \mathbb { R }$

## A.2 Implementation Details of GGSS

## A.2.1 Discovery data and activation extraction

Discovery occupations are held out on a pertask basis so that the occupation used to score the downstream bias metric never appears in the discovery pool. For the racebased tasks (MCQ, 2AFC) we discover on {cook, doctor, lawyer, nurse, teacher}. For the gender-based Nurse/Doctor task we discover on {cook, lawyer, teacher} and hold out both nurse and doctor. For each base identity in the active discovery pool, we use all $K = 5$ race variants and both gender variants, yielding 10 counterfactual images per identity. All images within the same counterfactual group are required to have identical pixel dimensions. This is enforced by a runtime check before activation extraction to guarantee a consistent number of visual tokens.

For each image, we run a single forward pass with the fixed discovery prompt

$$
^ { ^ { \scriptstyle ( * ) } } \mathrm { D e s c r i b e ~ t h i s ~ i m a g e ~ i n ~ d e t a i l . } ^ { , }
$$

and capture the output of the target module using register\_forward\_hook. If the hooked module returns a tuple, we cache the first element. Otherwise, we cache the output tensor directly. The captured tensor is reshaped into

$$
\mathbf { H } _ { o , b } ^ { ( r , g ) } \in \mathbb { R } ^ { T \times D } .
$$

When T exceeds a configurable cap max\_tokens (default: 2048), we draw a deterministic random subset of token indices using seed 42. The same token indices are reused for all $( r , g )$ variants of the same base identity $b ,$ so token positions remain aligned across counterfactual images. GGSS then mean-pools the retained tokens to obtain one vector per image.

## A.2.2 Numerical spherical primitives

Section 3 defines the ideal spherical operations ν, log, exp, and Slerp. In Appendix A, we use Normalize<sub>ε</sub>, Log, and Exp to denote their numerical implementations. Specifically,

$$
\operatorname { N o r m a l i z e } _ { \varepsilon } ( \mathbf { x } ) = { \frac { \mathbf { x } } { \| \mathbf { x } \| _ { 2 } + \varepsilon } } , \qquad \varepsilon = 1 0 ^ { - 1 0 } .
$$

Inner products passed to arccos are clamped to $[ - 1 + 1 0 ^ { - 7 } , 1 - 1 0 ^ { - 7 } ]$ Degenerate cases with geodesic distance below $\varepsilon _ { \mathrm { g e o } } = 1 0 ^ { - 8 }$ return the zero tangent vector for the logarithmic map, and tangent vectors with norm below $\varepsilon _ { \mathrm { g e o } }$ return the base point under the exponential map. For Slerp, when the angle is below $\varepsilon _ { \mathrm { g e o } }$ , we use the continuous small-angle limit. All spherical operations are implemented in vectorized form over tokens.

## A.2.3 Pooled counterfactual bias subspace discovery

For each image, we first mean-pool the token activations and project onto the unit sphere:

$$
\widehat { \bar { \mathbf { h } } } _ { o , b } ^ { ( r , g ) } = \mathrm { N o r m a l i z e } _ { \varepsilon } \left( \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbf { h } _ { o , b , t } ^ { ( r , g ) } \right) .\tag{6}
$$

For race debiasing, $\boldsymbol { u } = \left( o , b , g \right)$ and $a = r .$ . Thus the main-body center $\mathbf { c } _ { u }$ is written concretely as $\scriptstyle c _ { o , b , g } .$ . For each counterfactual group $( o , b , g )$ , this center is computed by iterative Fréchet (Karcher) mean. Starting from the normalized Euclidean mean $\pmb { \eta } ^ { ( 0 ) }$ , the update at iteration m is

$$
\begin{array} { r l } & { \bar { \mathbf { t } } ^ { ( m ) } = \displaystyle \frac { 1 } { K } \sum _ { r } \mathrm { L o g } _ { \eta ^ { ( m ) } } \left( \widehat { \bar { \mathbf { h } } } _ { o , b } ^ { ( r , g ) } \right) , } \\ & { \eta ^ { ( m + 1 ) } = \mathrm { N o r m a l i z e } _ { \varepsilon } \left( \mathrm { E x p } _ { \eta ^ { ( m ) } } \big ( \bar { \mathbf { t } } ^ { ( m ) } \big ) \right) . } \end{array}\tag{7}
$$

The iteration terminates when $\lVert \bar { \mathbf t } ^ { ( m ) } \rVert _ { 2 } ~ < ~ 1 0 ^ { - 7 }$ or after 100 iterations. We denote the converged center by $\scriptstyle { c _ { o , b , g } } .$

The main-body shift $\mathbf { s } _ { u , a }$ is written concretely as $\mathbf { s } _ { o , b } ^ { ( r , g ) }$

$$
\begin{array} { r } { \mathbf { s } _ { u , a } = \mathbf { s } _ { o , b } ^ { ( r , g ) } = \log _ { \mathbf { c } _ { u } } ( \mathbf { y } _ { u , a } ) = \log _ { c _ { o , b , g } } \left( \widehat { \bar { \mathbf { h } } } _ { o , b } ^ { ( r , g ) } \right) . } \end{array}\tag{8}
$$

Concatenating across occupations, identities, genders, and races yields a shift matrix

$$
\mathbf { S } \in \mathbb { R } ^ { N _ { \mathrm { s h i f t s } } \times D } , \qquad N _ { \mathrm { s h i f t s } } = \sum _ { o \in O } | \boldsymbol { \mathcal { B } } _ { o } | | \boldsymbol { \mathcal { G } } | | \mathcal { R } | .\tag{9}
$$

The $\mathbf { S } \mathbf { V } \mathbf { D } \mathbf { S } = \mathbf { U } \pmb { \Sigma } \mathbf { V } ^ { \top }$ yields the bias subspace

$$
\mathbf { V } _ { \mathrm { b i a s } } = [ \pmb { v } _ { 1 } \ | \ \cdots \ | \ \pmb { v } _ { k } ] \in \mathbb { R } ^ { D \times k } ,\tag{10}
$$

with $k = K - 1 = 4$ by default. The global reference point is the Fréchet mean of all pooled unit vectors, computed with the same iteration as Eq. (7).

## A.2.4 Gate calibration statistics

For each normalized pooled discovery representative $\mathrm { ~ \bf ~ c ~ } = \mathrm { ~ \bf ~ y ~ } _ { u , a } .$ , we compute the protectedcoordinate magnitude

$$
b ( \mathbf { c } ) = \| \mathbf { V } _ { \mathrm { b i a s } } ^ { \top } \log _ { \pmb { \mu } } ( \mathbf { c } ) \| _ { 2 } .
$$

Equivalently, in the concrete implementation notation, $\mathbf { c } = \widehat { \bar { \mathbf { h } } } _ { o , b } ^ { ( r , g ) }$ . We cache

$$
\tilde { b } = \mathrm { m e d i a n } \{ b ( { \bf c } ) \} , \qquad \sigma _ { b } = \mathrm { s t d } \{ b ( { \bf c } ) \} .
$$

These scalars are loaded at inference time to compute the calibrated token gate.

## A.2.5 Inference-time geodesic-gated steering

At inference time, a forward hook intercepts the output activation tensor $\mathbf { Z } \in \mathbb { R } ^ { n _ { \mathrm { b a t c h } } \times T ^ { \prime } \times \mathbf { \dot { D } } }$ , flattens it into $N = n _ { \mathrm { b a t c h } } T ^ { \prime }$ token vectors, and processes each nonzero token independently. For token $\mathbf { h } _ { i } ,$ we record its norm $r _ { i } = \| \mathbf { h } _ { i } \| _ { 2 }$ and normalized direction $\widehat { \mathbf { h } } _ { i } = \mathrm { N o r m a l i z e } _ { \varepsilon } ^ { \mathrm { " } } ( \mathbf { h } _ { i } )$ . The implemented token update is

$$
\begin{array} { r } { \mathbf { t } _ { i } = \mathrm { L o g } _ { \mu } ( \widehat { \mathbf { h } } _ { i } ) , \quad \quad \quad \quad \quad } \\ { \mathbf { p } _ { i } = \mathbf { V } _ { \mathrm { b i a s } } \mathbf { V } _ { \mathrm { b i a s } } ^ { \top } \mathbf { t } _ { i } , \quad \quad \quad \quad } \\ { \mathbf { t } _ { i } ^ { \mathrm { c l e a n } } = ( I - \mathbf { V } _ { \mathrm { b i a s } } \mathbf { V } _ { \mathrm { b i a s } } ^ { \top } ) \mathbf { t } _ { i } . } \end{array}\tag{11}
$$

The target direction is

$$
\widehat { \mathbf { h } } _ { i } ^ { \mathrm { t a r g e t } } = \mathrm { N o r m a l i z e } _ { \varepsilon } \left( \mathrm { E x p } _ { \mu } ( \mathbf { t } _ { i } ^ { \mathrm { c l e a n } } ) \right) .\tag{12}
$$

The per-token gate uses the discovery statistics:

$$
\begin{array} { r l } & { z _ { i } = \frac { \| \mathbf { V } _ { \mathrm { b i a s } } ^ { \top } \mathbf { t } _ { i } \| _ { 2 } - \tilde { b } } { \sigma _ { b } + \varepsilon } , } \\ & { g _ { i } = g _ { \mathrm { f l o o r } } + ( 1 - g _ { \mathrm { f l o o r } } ) \ \mathrm { s i g m o i d } ( \kappa z _ { i } ) , } \end{array}\tag{13}
$$

Since V<sub>bias</sub> has orthonormal columns, $\lVert \mathbf { V } _ { \mathrm { b i a s } } ^ { \top } \mathbf { t } _ { i } \rVert _ { 2 } ~ = ~ \lVert \mathbf { p } _ { i } \rVert _ { 2 } .$ For the main experiments, we use $\kappa = 5$ and $g _ { \mathrm { f l o o r } } = 0 . 3 ;$ other $g _ { \mathrm { { f l o o r } } }$ values are used only in the sensitivity analysis. Slerp then rotates toward the target:

$$
\widehat { \mathbf { h } } _ { i } ^ { \mathrm { s t e e r e d } } = \mathrm { S l e r p } ( \widehat { \mathbf { h } } _ { i } , \widehat { \mathbf { h } } _ { i } ^ { \mathrm { t a r g e t } } ; \alpha g _ { i } ) ,\tag{14}
$$

and norm restoration produces the final output:

$$
\mathbf { h } _ { i } ^ { \mathrm { s t e e r e d } } = r _ { i } \widehat { \mathbf { h } } _ { i } ^ { \mathrm { s t e e r e d } } .\tag{15}
$$

The steered tensor is reshaped to its original layout and cast to the input dtype before being returned by the hook.

## A.2.6 Saved artifacts

The discovery stage saves a .pt checkpoint containing $( \mathbf { V } _ { \mathrm { b i a s } } , \mu , k , \tilde { b } , \sigma _ { b } )$ together with metadata: singular values $\sigma _ { j }$ , explained-variance ratios $\rho _ { j } =$ $\sigma _ { j } ^ { 2 } / \sum _ { \ell } \sigma _ { \ell } ^ { 2 }$ , the full protected-coordinate magnitude distribution $\{ b ( \mathbf { c } ) \}$ (for diagnostic histograms, see Appendix B.2), and discovery-set sizes. This checkpoint is sufficient for inference-time loading.

## A.3 Proof of Proposition 1

Proof. By Eq. (1), the columns of $\mathbf { V } _ { \mathrm { b i a s } }$ are right singular vectors of S. Hence $\mathbf { V } _ { \mathrm { b i a s } } ^ { \top } \mathbf { V } _ { \mathrm { b i a s } } \ =$ $I _ { k } .$ . Therefore $\mathbf { V } _ { \mathrm { b i a s } } \mathbf { V } _ { \mathrm { b i a s } } ^ { \top }$ is symmetric and idempotent, so it is the orthogonal projector onto $\operatorname { s p a n } ( \mathbf { V } _ { \mathrm { b i a s } } )$ . Consequently,

$$
I - \mathbf { V } _ { \mathrm { b i a s } } \mathbf { V } _ { \mathrm { b i a s } } ^ { \top }
$$

is the orthogonal projector onto

$$
\mathrm { s p a n } ( \mathbf { V } _ { \mathrm { b i a s } } ) ^ { \perp } = \{ \mathbf { z } \in \mathbb { R } ^ { D } : \mathbf { V } _ { \mathrm { b i a s } } ^ { \top } \mathbf { z } = \mathbf { 0 } \} .
$$

Using Eq. (2), we have

$$
\mathbf { t } _ { i } ^ { \mathrm { c l e a n } } = ( I - \mathbf { V } _ { \mathrm { b i a s } } \mathbf { V } _ { \mathrm { b i a s } } ^ { \top } ) \mathbf { t } _ { i } ,
$$

which proves the projection claim.

It remains to prove norm preservation. Let $\mathbf { p } =$ $\widehat { \mathbf { h } } _ { i } , \mathbf { q } = \widehat { \mathbf { h } } _ { i } ^ { \mathrm { t a r g e t } }$ , and $\theta = \operatorname { a r c c o s } \langle \mathbf { p } , \mathbf { q } \rangle$ . If p $\neq \mathbf q$ and $\mathbf { q } \neq - \mathbf { p }$ , define

$$
\mathbf { u } = \frac { \mathbf { q } - \cos ( \theta ) \mathbf { p } } { \sin ( \theta ) } .
$$

Then u has unit norm and is orthogonal to p. The Slerp formula can be written as

$$
{ \mathrm { S l e r p } } ( \mathbf { p } , \mathbf { q } ; \beta _ { i } ) = \cos ( \beta _ { i } \theta ) \mathbf { p } + \sin ( \beta _ { i } \theta ) \mathbf { u } ,
$$

which has unit norm. If $\mathbf { p } = \mathbf { q } ,$ the convention ${ \mathrm { S l e r p } } ( \mathbf { p } , \mathbf { p } ; \beta _ { i } ) = \mathbf { p } $ also gives a unit vector. Therefore $\widehat { \mathbf { h } } _ { i } ^ { \mathrm { s t e e r e d } }$ has unit norm under the stated cases. By Eq. (5) and $r _ { i } = \| \mathbf { h } _ { i } \| _ { 2 }$

$$
\lVert \mathbf { h } _ { i } ^ { \mathrm { s t e e r e d } } \rVert _ { 2 } = \lVert r _ { i } \mathbf { \widehat { h } } _ { i } ^ { \mathrm { s t e e r e d } } \rVert _ { 2 } = r _ { i } = \lVert \mathbf { h } _ { i } \rVert _ { 2 } .
$$

This proves the result.

## A.4 Ablation Variants

We evaluate four SVD ablations that isolate individual design choices of GGSS.

Pooled SVD (Eucl.). This removes both spherical geometry and the Slerp/gate. Images are Euclidean-pooled, per-group Euclidean means are subtracted, and SVD on the resulting shift matrix yields a Euclidean bias basis. Inference subtracts $\alpha \mathbf { V } _ { \mathrm { b i a s } } \mathbf { V } _ { \mathrm { b i a s } } ^ { \top } ( \mathbf { h } _ { i } - \pmb { \mu } )$ from every token. Because it is Euclidean, this variant does not preserve token norms.

Pooled SVD (sph.). Retains spherical geometry (Fréchet means, tangent shifts, exp-map projection, norm restoration) but uses hard null-space projection in tangent space:

$$
\mathbf { t } _ { i } ^ { \mathrm { s t e e r e d } } = \mathbf { t } _ { i } - \alpha \mathbf { V } _ { \mathrm { b i a s } } \mathbf { V } _ { \mathrm { b i a s } } ^ { \top } \mathbf { t } _ { i } .
$$

This is identical to GGSS with the Slerp step replaced by hard projection and the gate disabled $( g _ { i } \equiv 1 )$ .

Per-token SVD (Eucl.) and Per-token SVD (sph.). These variants compute per-token centers $\scriptstyle c _ { o , b , g , t }$ and per-token shifts before SVD, preserving token position information in discovery. The inference step then applies standard null-space projection. Per-token SVD is more expensive and, on most (model, task) cells, dominated by the pooled variant.

<table><tr><td></td><td colspan="4">Pixtral-12B</td><td colspan="4">LLaVA-1.6-Vicuna-7B</td><td colspan="4">LLaVA-1.6-Mistral-7B</td><td colspan="4">Qwen3-VL-4B</td></tr><tr><td>Method</td><td>MCQ  $\times 1 0 ^ { 3 } \downarrow$ </td><td>2AFC ↓</td><td>N/D ↓</td><td> $\mathbf { A v } \mathbf { g }$  △%↓</td><td> $\mathbf { M C Q }$   $\times 1 0 ^ { 3 } \downarrow$ </td><td>2AFC ↓</td><td>N/D ↓</td><td>Avg △%↓</td><td>MCQ  $\times 1 0 ^ { 3 } \downarrow$ </td><td>2AFC ↓</td><td>N/D ↓</td><td>Avg △%↓</td><td>MCQ  $\times 1 0 ^ { 3 } \downarrow$ </td><td>2AFC ↓</td><td>N/D ↓</td><td>Avg △%↓</td></tr><tr><td>Baseline</td><td>25.75</td><td>0.243</td><td>0.425</td><td>0%</td><td></td><td>8.70 0.000</td><td>0.600</td><td>0%</td><td>26.87</td><td>0.000</td><td>0.625</td><td>0%</td><td>14.65</td><td>0.368</td><td>0.400</td><td>0%</td></tr><tr><td colspan="2">Hard projection, no Slerp, no gate</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Pooled SVD (Eucl.)</td><td>10.30 -60%</td><td>0.173 -29%</td><td>0.425 +0%</td><td>-30%</td><td>9.91 +14%</td><td>0.000</td><td>0.025 -96%</td><td>-41%</td><td>5.81 -78%</td><td>0.000</td><td>0.300 -52%</td><td>-65%</td><td>6.49 -56%</td><td>0.190 -48%</td><td>0.100 -75%</td><td>-60%</td></tr><tr><td>15.54 -40%</td><td>0.098 -60%</td><td>0.125 -71%</td><td>-57%</td><td>1.36 -84%</td><td>0.000</td><td>0.025 -96%</td><td>-90%</td><td>11.20 -58%</td><td>0.000</td><td>0.050 -92%</td><td>-75%</td><td>8.37 -43%</td><td>0.173</td><td>0.150 -62%</td><td>-53%</td></tr><tr><td colspan="2">Per-token hard projection</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>-53%</td><td></td><td></td></tr><tr><td>Per-token SVD (Eucl.)</td><td>24.59</td><td>0.172 -29%</td><td>0.522</td><td>-4%</td><td>4.26</td><td>0.000</td><td>0.600</td><td>-25%</td><td>2.08</td><td>0.000</td><td>0.625</td><td>-46%</td><td>4.35</td><td>0.245</td><td>0.475</td><td>-28%</td></tr><tr><td>Per-token SVD (sph.)</td><td>-4% 32.84</td><td>0.227 -7%</td><td>+23% 0.400</td><td>+5%</td><td>-51% 2.05</td><td>0.000</td><td>+0% 0.625</td><td>-36%</td><td>-92% 2.74</td><td>0.000</td><td>+0% 0.625</td><td>-45%</td><td>-70% 4.19</td><td>-33% 0.286</td><td>+19% 0.450</td><td>-27%</td></tr><tr><td>GGSS (Ours)</td><td>+28% 16.85</td><td>0.096</td><td>-6% 0.125</td><td>-55%</td><td>-76%</td><td>一 1.36 0.000</td><td>+4% 0.025</td><td>-90%</td><td>-90% 8.48</td><td>0.000</td><td>+0% 0.050</td><td>-80%</td><td>-71% 5.10</td><td>-22% 0.154</td><td>+12% 0.175</td><td>-60%</td></tr></table>

Table 7: SVD ablations of GGSS across all models and bias tasks. Rows remove the gate and Slerp, and per-token rows additionally replace pooled SVD with per-token SVD. Results use the same best-avg-α protocol as Table 1; per-column best values within this ablation set are bolded.

## A.5 Baseline Methods

## A.5.1 Classifier-Guided Mean-Shift Correction (MeanDiff)

We adapt classifier-based debiasing to the generative VLM setting using a single image-level representation per image. For each discovery image, the activation matrix is mean-pooled to obtain $\bar { \mathbf { h } } \in \mathbb { R } ^ { D }$ A race classifier is then trained on these pooled activations. We consider two classifiers: an RBFkernel SVM (sklearn.svm.SVC(kernel="rbf"), C=1, γ=scale) and a multinomial logistic regression (GPU-accelerated cuML where available, otherwise scikit-learn). For the Euclidean variant, the class mean $\pmb { \mu } _ { r }$ and global mean $\pmb { \mu } _ { \mathrm { g l o b a l } }$ yield the class-specific shift $\delta _ { r } = \mu _ { r } - \mu _ { \mathrm { g l o b a l } }$ . At inference time the pooled activation is classified and $\alpha \pmb { \delta } _ { \hat { r } }$ is subtracted from every token. The spherical variant uses Fréchet means and tangent-space subtraction with norm restoration.

## A.5.2 INLP

We implement Iterative Null-space Projection (Ravfogel et al., 2020) in both Euclidean and spherical variants. Let $\mathbf { X } \in \mathbb { R } ^ { N \times D }$ and $y \in \{ 1 , \ldots , K \} ^ { N }$ The iteration is $\mathbf { X } ^ { ( 0 ) } = \mathbf { X }$ . For $m = 0 , 1 , \ldots$ we fit a multinomial logistic-regression classifier $f ^ { ( m ) }$ on $\mathbf { X } ^ { ( m ) }$ . If accuracy falls to within $1 0 ^ { - 6 }$ of the uniform-chance baseline $1 / K$ , the iteration halts. Otherwise, the orthonormalized rows of the weight matrix are appended to the bias basis and $\mathbf { X } ^ { ( m + 1 ) }$ is obtained by projecting $\mathbf { X } ^ { ( m ) }$ into their null space. We use at most 10 iterations and $C { = } 1 . 0$ . After convergence, stacking and orthonormalizing the extracted directions yields $\mathbf { V } _ { \mathrm { b i a s } }$ . The Euclidean variant applies standard null-space steering. The spherical variant normalizes, computes a Fréchet global mean, and projects in the tangent space before exp-mapping and norm-restoring.

## A.5.3 LEACE

We additionally evaluate LEACE (Belrose et al., 2023), which provides a closed-form least-squares concept eraser. Let $\pmb { \Sigma } _ { \mathbf { x } }$ denote the covariance of pooled discovery activations h<sup>¯</sup> and let $\pmb { \Sigma } _ { \mathbf { x } \mathbf { z } }$ denote the cross-covariance with the one-hot race labels. LEACE computes a whitening $\mathbf { W } = \pmb { \Sigma } _ { \mathbf { x } } ^ { - 1 / 2 }$ takes the top-K−1 left singular vectors of $\mathbf { W } \pmb { \Sigma } _ { \mathbf { x } \mathbf { z } } .$ and applies the affine projection $P ( \mathbf { x } ) \ = \ \mathbf { x } \ - \quad$ $\mathbf { W } ^ { - 1 } \mathbf { U } \mathbf { U } ^ { \top } \mathbf { W } ( \mathbf { x } - \mu _ { \mathrm { g l o b a l } } )$ . We evaluate two variants: spherical LEACE, applied in tangent space, and gated geodesic LEACE, which uses LEACE’s U as $\mathbf { V } _ { \mathrm { b i a s } }$ inside the GGSS pipeline. The latter serves as a drop-in test of whether LEACE’s closed-form subspace transfers under the Slerp + gate framework.

## A.5.4 BendVLM-style Retrieval Equalization

Following Gerych et al. (2024), we represent each discovery image by a single global activation e = vec(H) $\bar { \in } \mathbb { R } ^ { T \bar { D } }$ and store L2-normalized reference vectors with their race labels. We replace Bend-VLM’s text-side augmentation with an SVD-based demographic projector computed from race-center differences. At inference time we retrieve the 10 nearest references per race via cosine similarity, average to per-race prototypes $\pi _ { r }$ , and solve for the minimum-norm correction that equalizes cosine distance to all prototypes. The Euclidean variant uses a linear equidistance constraint. The spherical variant lifts everything to the tangent space at the global Fréchet mean, solves the equidistance system there, and maps back to the sphere with norm restoration. The correction tensor is cached after the first decoding step of each image because visual tokens remain unchanged across autoregressive steps.

## A.6 Model-Specific Hook Locations and Runtime Setup

All methods intervene at the layer denoted by projection-mlp2, which resolves to the final nn.Linear layer of the vision-to-language projection MLP. For Pixtral-12B, this is the second linear in the multi-modal projector. For LLaVA-1.6-Vicuna-7B and LLaVA-1.6-Mistral-7B, it is model.model.mm\_projector[2]. For Qwen3- VL-4B-Instruct, it is model.model.visual .merger.linear\_fc2. Models are loaded with device\_map="auto" and run in bfloat16. All steering computations use float32 for numerical stability, and outputs are cast back to the input dtype before being returned by the hook. All stochastic components use seed 42. Inner products passed to arccos are clamped to $[ - 1 + 1 0 ^ { - 7 } , 1 - 1 0 ^ { - 7 } ]$ , and geodesic distances below $1 0 ^ { - 8 }$ are treated as zero.

## B Additional Experimental Details

## B.1 Protocols

Sweep. All methods are evaluated at $\alpha \in \{ 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 . 0 , 1 . 5 \}$ on three bias tasks. We report the best-α metric per (method, model, task). Discovery occupations are held out on a pertask basis as described in §A.2: race-task discovery uses {cook, doctor, lawyer, nurse, teacher}, and gender-task discovery uses {cook, lawyer, teacher} with nurse and doctor held out.

Decoding. Greedy decoding (do\_sample=False). Default budget max\_new\_tokens= 2048. For MCQ we use 8, since the output is constrained to a single letter.

CEO Salary. We sample $N _ { \mathrm { b i o s } } = 1 0$ biographies, replace name placeholders with “the candidate,” and ask the model to output only an integer salary. Parsing first searches for a 4–9 digit integer with commas/dollar signs stripped, then falls back to any integer $\geq 1 0 0 0$ . The baseline maximum salary is used as an outlier cap for steered methods. Total queries per (method,α): $B _ { \mathrm { c e o } } \cdot N _ { \mathrm { b i o s } } \cdot 5 \cdot 2 .$ , where $B _ { \mathrm { c e o } } = 8$ . Evidential role. With only 10 biographies this probe is small, and we treat it as directional, corroborating evidence, not a stand-alone result; no headline claim rests on it. On both backbones where the probe elicits varying salaries, the point estimates move toward parity under GGSS (Qwen3-VL gender $\mathrm { g a p - 8 8 k  - 5 5 k ; }$ Pixtral 144k → 33k), the same direction as the statistically significant reductions on the larger probes, but the bio-cluster bootstrap CIs are wide and overlap (Qwen3-VL: [−213k, 0] unsteered vs. [−165k, 0] steered).

MMStar. MMStar is evaluated with VLMEvalKit using exact matching rather than LLM-based judging to avoid judge-dependent confounds. We report $\Delta _ { \mathrm { a c c } } = \mathrm { A c c } ^ { \mathrm { m e t h o d } } ( \alpha ) - \mathrm { A c c } ^ { \mathrm { b a s e l i n e } }$

## B.2 Hyperparameters

Table 8 summarizes the three GGSS-specific hyperparameters and the ranges we explored. Best-α is selected once per (method, model) pair under the best-avg-α protocol. κ and $g _ { \mathrm { f l o o r } }$ are fixed across all main-table experiments.

<table><tr><td>Hyperparameter</td><td>Default</td><td>Range explored</td></tr><tr><td>α (strength)</td><td>best-α</td><td>{0.25, 0.5, 0.75, 1.0, 1.5}</td></tr><tr><td>κ (gate sharpness)</td><td>5</td><td>{1, 2, 5, 10}</td></tr><tr><td>gfloor (gate floor)</td><td>0.3</td><td>{0.0, 0.3, 0.5}</td></tr><tr><td>k (subspace dim)</td><td> $K - 1$ </td><td>fixed</td></tr></table>

Table 8: GGSS hyperparameters and explored ranges.

<table><tr><td>Method</td><td>Pixtral</td><td>Vicuna</td><td>Mistral</td><td>Qwen3</td></tr><tr><td>INLP (Eucl.)</td><td>1.0</td><td>0.25</td><td>1.0</td><td>1.5</td></tr><tr><td>INLP (sph.)</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.5</td></tr><tr><td>MeanDiff-SVM (Eucl.)</td><td>0.25</td><td>0.5</td><td>0.75</td><td>1.0</td></tr><tr><td>MeanDiff-SVM (sph.)</td><td>0.25</td><td>0.25</td><td>0.75</td><td>0.5</td></tr><tr><td>MeanDiff-LR (Eucl.)</td><td>0.75</td><td>0.5</td><td>1.0</td><td>1.0</td></tr><tr><td>MeanDiff-LR (sph.)</td><td>0.75</td><td>1.0</td><td>0.75</td><td>1.0</td></tr><tr><td>BendVLM (Eucl.)</td><td>1.0</td><td>0.75</td><td>0.75</td><td>1.5</td></tr><tr><td>BendVLM (sph.)</td><td>0.25</td><td>0.5</td><td>0.25</td><td>1.5</td></tr><tr><td>Pooled SVD (Eucl.)</td><td>0.5</td><td>1.0</td><td>0.75</td><td>1.5</td></tr><tr><td>Pooled SVD (sph.)</td><td>0.75</td><td>1.0</td><td>1.0</td><td>1.5</td></tr><tr><td>Per-token SVD (Eucl.)</td><td>0.75</td><td>0.25</td><td>0.5</td><td>1.0</td></tr><tr><td>Per-token SVD (sph.)</td><td>1.0</td><td>1.0</td><td>0.25</td><td>1.5</td></tr><tr><td>GGSS</td><td>0.75</td><td>1.0</td><td>1.0</td><td>1.5</td></tr></table>

Table 9: Best-avg-α choices used in Tables 1 and 2. Each value is the single operating point selected for that (method, model) pair.

## B.3 Per-Model Full Results and MMStar

Full per-(method, α) tables are reported in the project repository (results\_report.md). Table 1 in the main paper reports best-α values. Layer comparisons (other than projection-mlp2) were preliminary and preserved unchanged from earlier work, and we do not recommend reading headline numbers from them.

MMStar preservation. Table 2 reports the compact MMStar summary for all paper-relevant methods at each method’s best-avg-α. The project repository contains the corresponding evaluation artifacts and per-run outputs.

## B.4 Statistical Significance Protocol

Decoding is greedy and deterministic with a fixed seed, so repeated runs are bit-identical; the relevant uncertainty is sampling over evaluation items. For every cell of Table 1 we computed: (i) 95% bootstrap confidence intervals (10,000 resamples), over items and over base identities, for the metric and for the paired %-change vs. the unsteered baseline on the same resampled items; and (ii) paired sign-flip permutation tests between methods, which are valid regardless of the plug-in metric’s smallsample bias because both methods are evaluated on the same items. MMStar is compared per-question with exact McNemar tests. Table 5 in the main text summarizes the outcomes.

The largest individual claims hold on their own: the LLaVA-Vicuna Nurse/Doctor reduction $( - 9 6 \% )$ has $p < 1 0 ^ { - 4 }$ with an item-level CI of $[ - 1 1 8 \% , - 5 7 \% ]$ of the baseline gap, and the MM-Star “within $\pm 0 . 6 \mathrm { p . p . } ^ { , \mathrm { \prime } }$ claim is backed by perquestion paired CIs (all within ±1.9 pp) and exact McNemar tests (all $p \geq 0 . 0 9 )$ . By contrast, INLP (sph.), the strongest bias-side competitor, significantly damages Pixtral’s MMStar (−6.1 pp race / −3.3 pp gender, McNemar $p < 1 0 ^ { - 4 } )$

## B.5 Held-Out α Selection (Cross-Fitted)

Probes are split by base identity into two folds; α is selected on one fold with the paper’s best-avg-α rule and metrics are reported on the disjoint fold, cross-fitted, applied symmetrically to all methods. Table 10 reports GGSS: the selected α matches the paper’s on six of eight folds, and the test-fold reduction is negative in every fold on every model.

<table><tr><td>Model</td><td>selected α, fold A / B (paper α)</td><td>test-fold Avg ∆%, fold A /B</td></tr><tr><td>Pixtral-12B</td><td>0.75 / 1.0 (0.75)</td><td>-27.4% / -33.8%</td></tr><tr><td>LLaVA-Vicuna-7B</td><td>1.0 / 0.25 (1.0)</td><td>-50.5% /-50.0%</td></tr><tr><td>LLaVA-Mistral-7B</td><td>1.0 /1.0 (1.0)</td><td>-77.8% / -93.5%</td></tr><tr><td>Qwen3-VL-4B</td><td>1.5 /1.5 (1.5)</td><td>-34.4% / -40.8%</td></tr></table>

Table 10: Held-out α protocol for GGSS: α selected on fold A (identities 1–4) is evaluated on fold B (identities 5–8) and vice versa.

## B.6 Per-Dimension MMStar Results

Table 11 breaks MMStar down by capability dimension at the deployed steering strength. All six dimensions, including fine-grained perception, stay within the ±6 pp binomial noise band $( n = 2 5 0$ per dimension) of the unsteered model on all four backbones.

## B.7 MME Results

As a second capability benchmark beyond MM-Star, we evaluated MME on Qwen3-VL-4B under GGSS at the deployed strength: the perception total moves $1 6 9 3 . 9  1 6 8 5 . 6 ~ ( - 0 . 4 \% )$ and the reasoning total $6 3 1 . 1  6 1 7 . 1 ( - 1 . 8 \% )$ , with no subcategory collapsing.

## B.8 Demographic-Recognition Probes and the α Dial

To test whether steering debiases demographic perception or deletes it, the steered model is explicitly asked the perceived gender and race of the held-out counterfactual images. Table 12 shows the intervention is attribute-specific: steering one attribute leaves recognition of the other attribute intact on all four backbones. Table 13 shows the removalvs-retention trade-off is controllable through α: at the bias-minimizing operating point the targeted attribute’s reportability on face-only close-ups is strongly reduced, which is the expected behaviour of subspace removal, while at moderate strength the model retains the attribute with much of the debiasing already realized. The knee of this trade-off is attribute-dependent, so α can be chosen per attribute and task; tasks that legitimately require the attribute can operate at moderate α or gate steering off, without retraining.

## B.9 Fairness-Instruction Prompting Baseline

The prompt-space baseline runs the identical probes, with no steering, after appending a fairness instruction to every prompt: “do NOT let the person’s perceived race, ethnicity, or gender influence your answer; base your answer only on task-relevant visual evidence”. Table 14 shows instruction prompting is inconsistent across models and tasks, and can even amplify bias, whereas GGSS reduces bias consistently on every model.

## B.10 LEACE Baseline Results

LEACE (Belrose et al., 2023) is fit with the authors’ official concept-erasure implementation and applied at the same projection layer as every other method (implementation details in Appendix A.5), in two variants: a direct spherical port, and LEACE’s closed-form subspace combined with our calibrated gate. Both were evaluated on all four backbones under the paper’s best-avg-α protocol; Table 6 in the main text reports the results. Ported this way, LEACE is a serious baseline: it wins the Pixtral column at its operating point and preserves Pixtral MMStar there (−0.8/−0.9 pp for the two variants, McNemar $p \ge 0 . 1 5 )$ , so its results do not stem from a broken port. It is, however, less consistent across backbones (−26% to −31% on Qwen3-VL), while GGSS keeps the best four-backbone mean and worst case and preserves MMStar throughout.

<table><tr><td>Model</td><td>Steering</td><td>CP</td><td>FP</td><td>IR</td><td>LR</td><td>math</td><td>S&amp;T</td></tr><tr><td>Pixtral-12B</td><td>race</td><td> $6 8 . 8  6 9 . 2 $ </td><td> $4 4 . 0  4 1 . 6 $ </td><td> $6 8 . 8  7 1 . 2 $ </td><td> $5 7 . 2  5 5 . 6$ </td><td> $4 5 . 6 \to 4 4 . 4$ </td><td> $3 7 . 2  3 7 . 2 $ </td></tr><tr><td>Pixtral-12B</td><td>gender</td><td> $6 8 . 8  6 8 . 0$ </td><td> $4 4 . 0  4 4 . 4 $ </td><td> $6 8 . 8 \to 6 8 . 8$ </td><td> $5 7 . 2  5 5 . 6$ </td><td> $4 5 . 6 \to 4 5 . 2$ </td><td> $3 7 . 2  3 7 . 6$ </td></tr><tr><td>LLaVA-Vicuna-7B</td><td>race</td><td> $5 9 . 2  5 8 . 0 $ </td><td> $3 2 . 8  3 8 . 0$ </td><td> $4 6 . 0  4 5 . 6$ </td><td> $3 1 . 2  3 2 . 8 $ </td><td> $2 7 . 2  2 5 . 6$ </td><td> $2 7 . 6  2 6 . 8$ </td></tr><tr><td>LLaVA-Vicuna-7B</td><td>gender</td><td> $5 9 . 2  5 9 . 2 $ </td><td> $3 2 . 8 \to 3 2 . 4$ </td><td> $4 6 . 0  4 6 . 8$ </td><td> $3 1 . 2  3 0 . 4$ </td><td> $2 7 . 2  2 6 . 8$ </td><td>27.6→27.2</td></tr><tr><td>LLaVA-Mistral-7B</td><td>race</td><td> $6 2 . 0  6 2 . 0$ </td><td> $3 4 . 8 \to 3 6 . 4$ </td><td> $4 8 . 0  4 6 . 4 $ </td><td> $3 5 . 6 \to 3 5 . 2$ </td><td> $2 7 . 2  2 7 . 6$ </td><td> $2 3 . 2  2 5 . 6$ </td></tr><tr><td>LLaVA-Mistral-7B</td><td>gender</td><td> $6 2 . 0  6 2 . 4$ </td><td> $3 4 . 8 \to 3 4 . 8$ </td><td> $4 8 . 0  4 7 . 6 $ </td><td> $3 5 . 6 \to 3 4 . 0$ </td><td> $2 7 . 2  2 7 . 2$ </td><td> $2 3 . 2  2 4 . 4$ </td></tr><tr><td>Qwen3-VL-4B</td><td>race</td><td> $7 2 . 4  7 2 . 4$ </td><td> $5 6 . 8 \to 5 7 . 2$ </td><td> $7 1 . 6  7 1 . 2$ </td><td> $6 0 . 4  6 1 . 2$ </td><td> $5 9 . 6 \to 6 0 . 0$ </td><td> $4 8 . 4  4 8 . 0$ </td></tr><tr><td>Qwen3-VL-4B</td><td>gender</td><td> $7 2 . 4  7 2 . 8$ </td><td> $5 6 . 8 \to 5 6 . 4$ </td><td> $7 1 . 6  7 2 . 8$ </td><td> $6 0 . 4  6 1 . 6$ </td><td> $5 9 . 6 \to 6 0 . 0$ </td><td> $4 8 . 4  4 9 . 2 $ </td></tr></table>

Table 11: MMStar accuracy per capability dimension, unsteered → GGSS at the deployed α (CP coarse perception, FP fine-grained perception, IR instance reasoning, LR logical reasoning, S&T science & technology).

<table><tr><td>Model</td><td>gender recognition, under race steering under gender steering</td><td>race recognition,</td></tr><tr><td>Pixtral-12B</td><td> $8 8 . 7  8 8 . 7 $ </td><td> $7 0 . 0  7 5 . 0$ </td></tr><tr><td>Qwen3-VL-4B</td><td> $9 7 . 5  9 6 . 3 $ </td><td> $8 0 . 0  8 2 . 5$ </td></tr><tr><td> $\mathrm { L L a V A – V i c u n a  – 7 B }$ </td><td> $9 3 . 8 \to 8 1 . 2$ </td><td> $6 6 . 2  6 6 . 2$ </td></tr><tr><td>LLaVA-Mistral-7B</td><td> $9 8 . 8  1 0 0 . 0$ </td><td> $7 2 . 5  7 2 . 5$ </td></tr></table>

Table 12: Recognition accuracy (%) of the nontargeted attribute, unsteered → steered at the paper’s α: the attribute that steering is not aimed at remains recognizable.
<table><tr><td>α</td><td>race recog. (%)</td><td> $\mathbf { M C Q } \ \mathrm { r a c e \ b i a s }$   $( \mathrm { J S D } \times 1 0 ^ { 3 } )$ </td><td>gender  $\mathrm { r e c o g . } \left( \% \right)$ </td><td>Nurse/Doctor gap</td></tr><tr><td>0 (unsteered)</td><td>80.0</td><td>14.65</td><td>97.5</td><td>0.400</td></tr><tr><td>0.5</td><td>76.2</td><td> $6 . 5 4 ( - 5 5 \% )$ </td><td>96.3</td><td> $0 . 3 7 5 \ : ( - 6 \% )$ </td></tr><tr><td>1.5 (paper)</td><td>20.0</td><td> $5 . 1 0 ( - 6 5 \% )$ </td><td>42.5</td><td> $0 . 1 7 5 \ : ( - 5 6 \% )$ </td></tr></table>

Table 13: The α dial on Qwen3-VL-4B: recognition of the targeted attribute vs. bias reduction under the corresponding steering mode. Most of the MCQ (race) reduction is realized by $\alpha = 0 . 5$ with recognition nearly intact; the Nurse/Doctor (gender) gap responds mainly between $\alpha = 0 . 5$ and the paper’s operating point.

## B.11 Matched-Gate Geometry Ablation

Table 15 toggles Euclidean versus spherical steering at matched gate condition on all four backbones under the paper’s full protocol (all bias tasks, bestavg-α applied identically to every variant), with paired sign-flip tests on every geometry contrast. At 4B and 7B the geometry toggle is within sampling noise in both gate conditions, and on Qwen3- VL the point estimates favor the Euclidean form; at the single Table 3 cell, a Euclidean update with the same gate plus renormalization posts the lowest value we measured (1.51 vs. GGSS’s 2.41, paired $p \approx 0 . 7 5 ;$ ; renormalization returns the update to the token’s norm sphere, so this variant is a chordal discretization of the same on-sphere move). Geometry becomes decisive with scale: on Pixtral-12B the spherical form is significantly better under both toggles (no gate $p = 0 . 0 3 1$ ; gate fixed $p = 0 . 0 1 5 )$ both Euclidean forms significantly damage MM-Star there $( - 6 . 3 \mathsf { p p } .$ , McNemar $p < 1 0 ^ { - 8 } )$ while both spherical forms preserve it on every backbone, and over-steering exposes failure modes the geodesic form did not show in any of our runs: the off-sphere update overshoots to roughly 9× the unsteered bias at $\alpha = 1 . 0 $ , and the renormalized variant degenerates outright there (72 of 80 responses unparseable).

## B.12 Gate Calibration: Pooled vs. Per-Token Statistics

The gate statistics $\tilde { b } , \sigma _ { b }$ are calibrated on pooled discovery representatives, while steering acts on individual tokens. We compared the two distributions directly under the paper’s pooled subspace: the per-token bias-norm distribution is right-shifted relative to the pooled representatives (Qwen3-VL: median 0.196 vs. 0.113), so with pooled calibration the gate operates in its upper range on real tokens: mean gate 0.85, with 72% of tokens gated above 0.9 and the least-biased 14% suppressed below 0.4. Pooled calibration therefore gives an operating point that steers most tokens and spares the low-bias tail. Re-running GGSS with the gate statistics recalibrated on the per-token distribution (same subspace, same α) reaches the same bias reduction at the paper’s operating point (−65% MCQ on Qwen3-VL, identical to pooled), the same or better reductions at lower α (−91% at α = 0.5), equally preserves MMStar (61.8 vs. baseline 61.5, McNemar $p = 0 . 5 6 )$ , and under the full protocol ends within a point of pooled GGSS on Qwen3-VL (−60.4% vs. −59.9%) and at −84.6% on LLaVA-Mistral. The choice of calibration set does not change the paper’s conclusions.

<table><tr><td>Model</td><td>MCQ salary  $\mathrm { ( J S D \times 1 0 ^ { 3 } ) }$ </td><td>2AFC income (std)</td><td>Nurse/Doctor gap</td><td>GGSS avg</td></tr><tr><td>Pixtral-12B</td><td> $2 5 . 7 5 \to 1 8 . 8 8 ( - 2 7 \% )$ </td><td> $0 . 2 4 3 \to 0 . 2 8 1 ( + 1 6 \% )$ </td><td> $0 . 4 2 5  0 . 4 2 5 ( 0 \% )$ </td><td>-55%</td></tr><tr><td>Qwen3-VL-4B</td><td> $1 4 . 6 5 \to 1 4 . 2 9 ( - 2 \% )$ </td><td> $0 . 3 6 8 \to 0 . 2 9 9 ( - 1 9 \% )$ </td><td> $0 . 4 0 0  0 . 4 5 0 ( + 1 3 \% )$ </td><td>-60%</td></tr><tr><td> $\mathrm { L L a V A – V i c u n a – 7 B }$ </td><td> $8 . 7 0  1 6 . 0 6 ~ ( + 8 5 \% )$ </td><td> $0  0 ( - )$ </td><td> $0 . 6 0 0 \to 0 . 3 5 0 ( - 4 2 \% )$ </td><td>-90%</td></tr><tr><td>LLaVA-Mistral-7B</td><td> $2 6 . 8 7  9 . 0 4 ( - 6 6 \% )$ </td><td> $0  0 ( - )$ </td><td> $0 . 6 2 5  0 . 5 7 5 ( - 8 \% )$ </td><td>-80%</td></tr></table>

Table 14: Fairness-instruction prompting (no steering): bias metric before → after adding the instruction (% change). GGSS’s average reduction shown for reference.
<table><tr><td>Variant</td><td>Pixtral-12B</td><td> ${ \mathrm { Q w e n 3 - V L - 4 B } }$ </td><td>LLaVA-Mistral-7B</td><td>LLaVA-Vicuna-7B</td></tr><tr><td>Euclidean hard projection (no gate)</td><td>-29.6%</td><td>-59.7%</td><td>-65.2%</td><td>-41.0%</td></tr><tr><td>Spherical hard projection (no gate)</td><td>-56.6%</td><td>-52.8%</td><td>-75.2%</td><td>-90.1%</td></tr><tr><td>p (geometry, no gate)</td><td>0.031</td><td>0.30</td><td>0.60</td><td>0.50</td></tr><tr><td>Euclidean + calibrated gate</td><td>-18.3%</td><td>-62.4%</td><td>-67.6%</td><td>-86.1%</td></tr><tr><td>GGSS (spherical + calibrated gate)</td><td>-55.3%</td><td>-59.9%</td><td>-80.2%</td><td>-90.1%</td></tr><tr><td>p (geometry, gate fixed)</td><td>0.015</td><td>0.81</td><td>0.41</td><td>0.66</td></tr></table>

Table 15: Geometry toggled at matched gate condition, Avg ∆% under the paper’s protocol (each variant at its own best-avg-α; lower is better). $" p '$ rows: paired sign-flip test of the geometry contrast directly above, combined across tasks. For the two LLaVA models the average covers MCQ and Nurse/Doctor only (their 2AFC baseline is zero, so ∆% is undefined; the convention is identical for all methods, as in Table 1).

## B.13 Discovery-Prompt Invariance

The steered layer is the vision-to-language projection: its visual-token activations are computed before any interaction with the text prompt, so the discovery prompt architecturally cannot influence them. We verified this empirically: re-running discovery with four alternative prompts yields subspaces identical to numerical precision (principalangle cosines ≥ 0.9997) and identical end-to-end steering results (Table 16).

<table><tr><td>Discovery prompt</td><td>steered JSD×10³</td></tr><tr><td>&quot;Describe this image in detail.&quot; (paper)</td><td>5.10</td></tr><tr><td>&quot;What do you see in this image?&quot;</td><td>5.10</td></tr><tr><td>&quot;Caption this image.&quot;</td><td>5.10</td></tr><tr><td>&quot;Briefly describe the person in the photo.&quot;</td><td>5.10</td></tr><tr><td>&quot;Write a short story inspired by this image.&quot;</td><td>5.10</td></tr></table>

Table 16: Discovery-prompt invariance: Qwen3-VL-4B MCQ race bias $( \mathrm { J S D } \times 1 0 ^ { 3 }$ , unsteered 14.65) when $\mathbf { V } _ { \mathrm { b i a s } }$ is discovered with different prompts; steering applied at the paper’s operating point.

## B.14 Artifact Documentation, Licensing, and Intended Use

We use existing artifacts only for research evaluation and do not redistribute third-party datasets or model weights. REFLECT/FOCUS is used for controlled race- and gender-counterfactual bias evaluation; its repository is released under GPL-3.0 and documents 480 face-only counterfactual images across six occupations, eight source identities per occupation, and five race by two gender variants. SocialCounterfactuals is used consistently with its intended purpose of probing social bias in VLMs and is released under the MIT license. MMStar is used only as a general VLM capability benchmark through VLMEvalKit; VLMEvalKit is Apache-2.0 licensed. For MMStar, we rely on the original benchmark release and do not redistribute its data, since we could not verify an explicit dataset license from the public repository or dataset card.

Our released artifacts consist of GGSS code, configuration files, steering checkpoints, and aggregate evaluation outputs. These artifacts are intended for research on inference-time debiasing and auditing of frozen generative VLMs, not for making deployment claims without additional human review and domain-specific safety testing. The artifact documentation specifies the evaluated models, hook locations, bias tasks, capability benchmark, demographic attributes, occupations, held-out discovery/evaluation splits, and hyperparameter settings.