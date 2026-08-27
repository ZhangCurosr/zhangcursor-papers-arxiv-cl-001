# SHORT HORIZONS AND SPARSE CONCEPTS: A MATH-EMATICAL VIEW OF THE READOUT IN THE J-LENS

Shi-Qi Yan<sup>∗</sup>, Kai-Xuan Ding<sup>∗</sup>, Chao-Hong Tan, Qian Chen,

Wen Wang, Xiangang Li, Zhen-Hua Ling

Alibaba Token Hub, Alibaba Group

<sup>∗</sup>Equal Contribution

sqyan01@mail.ustc.edu.cn

## ABSTRACT

The Jacobian lens (J-lens) has been proposed as a way to read verbalizable representations from language models. However, its principle and meaning lack a detailed and theoretical discussion. We provide a mathematical view of this interpretation and of its assumed causal structure. Besides treating the J-lens as a heuristic probe, we further regard it as a first-order causal transfer operator from intermediate activations to expected future readouts. We study the Jacobian matrix as the optimal local linear approximation of the downstream mapping, analyze its global approximation behavior and bias, and identify its mathematical meaning as an expectation over anticipated future readouts. Further analysis of the Jacobian energy distribution reveals that its causal geometry is highly sparse. The energy decays with depth, concentrates in an extremely small proportion, and decomposes into diagonal pathways and specific critical positions. This decomposition further resolves the expectation of the J-lens over future outputs into short-horizon and sparse concept predictions, providing a more intuitive attribution and explanation for the ability of the J-lens to visualize concepts during the thinking process. Based on the theory, we propose a simple but effective improvement strategy and decoupling method for the J-lens, which significantly enhances the ability of the J-lens to read out correct intermediate concepts.

## 1 INTRODUCTION

Most computation in large language models (LLMs) occurs in the background (Brown et al., 2020; Wei et al., 2022). A prompt may elicit a brief answer, but this answer arises from a sequence of complex intermediate transformations whose content cannot be directly observed. Mechanistic interpretability attempts to open this black box by reading internal states, intervening on them, and relating them to behavior (Sharkey et al., 2025; Alain & Bengio, 2017; Hewitt & Liang, 2019). A central difficulty is that intermediate-layer representations are not naturally expressible in the vocabulary of human concepts, and their connection to the output layer is indirect. The residual stream is a high-dimensional vector space, whereas interpretation requires a structure that is sparse, reportable, and causally meaningful.

The recently proposed Jacobian lens, or J-lens, addresses this difficulty by propagating a hidden state along the downstream computation of the model and measuring how its perturbation affects the final-layer state (Gurnee et al., 2026). The resulting J-space is described as sparse and verbalizable: hidden states can be approximated by nonnegative combinations of a few J-lens vectors that support reporting, modulation, and selective computation (Olsson et al., 2022). However, it lacks a clear and detailed mathematical interpretation. The reasons why averaging Jacobian matrices over future positions yields valid tokens, and how the resulting representations relate to the internal thoughts of the model remain unknown. The J-lens claims that the first-order sensitivity of future outputs to hidden states is itself a carrier of verbalizable tendencies. Evaluating this claim requires establishing in what sense the Jacobian matrix is optimal, what objective it approximates, and whether the causal influence behind the average is uniformly distributed or concentrated at specific positions.

To address this issue, we provide a clear mathematical interpretation of the J-lens and empirically validate its effectiveness and identify its practical deviations. Given an input $h _ { t , l } .$ , the J-lens can be viewed as a global fit of the first-order linear approximation to the latent mapping $\mathbb { E } _ { t ^ { \prime } > t } [ h _ { t ^ { \prime } , L } ]$ . Conceptually, we regard the J-lens as an averaged first-order causal transfer operator. In a given context, the Jacobian matrix is the optimal local linear approximation of how a small change in an intermediate activation affects subsequent outputs. Under this formulation, the residual vanishes to first order. Averaging this operator over samples yields the practical $J _ { \ell } ,$ which provides the globally optimal approximation under the minimum mean squared error criterion. We use a Stein-based identity to justify the passage from local approximation to global fitting, estimate the potential theoretical bias, and provide experimental evidence that corroborates this conclusion from the reverse direction.

Building on this, we further provide an attribution of the causal mapping fitted by the J-lens. The averaged operator $J _ { \ell }$ compresses all position-to-position Jacobian matrices into a single matrix, but the underlying sensitivity is not uniform. We therefore study the Jacobian energy $E _ { L } ( t , t ^ { \prime } )$ between a source position t and a target position $t ^ { \prime } .$ . Experiments validate the sparsity of the Jacobian energy: on the one hand, the energy tends to be larger in earlier layers and much smaller near the output; on the other hand, within a single layer, the Jacobian energy exhibits strong concentration, and a small number of pairs carry most of the total energy. Through visualization, we observe that the concentration of energy can be grouped into two patterns: the diagonal pattern $\mathbf { \Psi } ( t \approx t ^ { \prime } )$ and the horizontal or vertical pattern $( t = t ^ { * } \mathrm { o r } t ^ { \prime } = t ^ { * } )$ . They represent two effective outcomes of the J-lens mapping in practice: short-horizon token prediction and highly sensitive tokens at critical positions. We refer to these two outcomes as the short horizons and sparse concepts. The readout results of the practical J-lens also effectively corroborate our conclusion.

Based on the above theoretical and experimental results, we design experiments to further validate the effectiveness of our interpretation by improving the J-lens. First, the concentration effect of the Jacobian energy indicates that the predictions of the J-lens naturally align with short-horizon tokens and sparse concepts. Averaging the Jacobian matrices over all positions turns unimportant positions into noise and affects the practical representations of the J-lens. We therefore introduce a simple but effective filtering strategy that uses only a subset of positions with the highest energy in the J-lens construction. Experiments show that using only the Jacobian matrices from the top 10% to 20% of positions by energy significantly improves the capability of the J-lens. Second, we also attempt to decouple the two representations of the J-lens (Ghandeharioun et al., 2024; Conmy et al., 2023). By designing masks targeting the diagonal (Pˆıslar et al., 2025; Meng et al., 2022), the two capabilities are decoupled by removing or retaining only the Jacobian matrices on diagonal entries. However, experiments show that the coupling between the two capabilities is stronger than expected, and the strengthening or weakening of the two capabilities exhibits a positive correlation. As this work is still in progress, this becomes a goal of our future research.

In summary, this work makes three contributions. First, we provide a mathematical foundation for the J-lens and establish it as the expectation over anticipated outputs. Through local linearization, global least-squares projection, and a Stein-type bridge, we connect the J-lens to the optimal linear readout and explicitly characterize non-Gaussian and nonlinear deviations. Second, we present experimental analyses and visualization studies that cluster the expectation over anticipated outputs into short-horizon token predictions and critical concept predictions. Third, we design a lens improving and decoupling method that further validates the reliability and effectiveness of our conclusions.

## 2 BACKGROUND: J-LENS AND GLOBAL WORKSPACE

A transformer language model maintains, at each token position, a residual-stream vector that is read from and written to by every layer. Early states are close to the input token, whereas final states are directly mapped to vocabulary scores by the unembedding matrix $W _ { U }$ . Intermediate states are therefore computational, but not directly verbal. The logit lens reads them as if all layers shared one coordinate system; this works near the output but often fails earlier, where representations have not yet been rotated into output coordinates (nostalgebraist, 2020; Belrose et al., 2023; Geva et al., 2022).

The Jacobian lens (J-lens) instead reads an intermediate activation by its average first-order effect on present and future outputs (Gurnee et al., 2026). For layer ℓ, it averages the Jacobian from a source activation to later final-layer states over source positions, target positions, and prompts:

$$
J _ { \ell } = \mathbb { E } _ { t , t ^ { \prime } \geq t , \mathrm { p r o m p t } } \left[ \frac { \partial h _ { \mathrm { f i n a l } , t ^ { \prime } } } { \partial h _ { \ell , t } } \right] , \qquad \mathrm { l e n s } ( h _ { \ell } ) = \mathrm { s o f t m a x } ( W _ { U } \mathrm { n o r m } ( J _ { \ell } h _ { \ell } ) ) .\tag{1}
$$

The top tokens of this readout describe what an activation is disposed to say across contexts, rather than what one context happens to say. The rows of $W _ { U } J _ { \ell }$ define token-associated readout directions; their sparse nonnegative combinations form the J-space.

Functionally, it supports five properties corresponding to conscious access: verbal report, directed modulation, internal reasoning, flexible generalization, and selectivity. However, a more detailed theoretical account is missing, and the J-lens itself is approximate and limited in applications. This paper takes this challenge seriously. We investigate why the averaged Jacobian matrix can approximate the optimal readout, how the approximation is affected by nonlinear and non-Gaussian bias, and what sparse causal structure over positions underlies this interpretation.

## 3 MATHEMATICAL EXPLANATION OF J-LENS: EXPECTATION OF THE FUTURE READOUTS

Given a layer ℓ and position t. We study the relationship between the J-lens and the mapping function of the averaged future readouts:

$$
y _ { t } = f _ { \ell } ( h _ { \ell , t } ) : = \mathbb { E } _ { t ^ { \prime } \geq t } [ h _ { \mathrm { f i n a l } , t ^ { \prime } } ] , \qquad f _ { \ell } : \mathbb { R } ^ { d } \to \mathbb { R } ^ { d } ,\tag{2}
$$

where $h _ { \ell , t }$ is the hidden states at layer $\ell$ and position $t ,$ and $y _ { t }$ is an averaged future final-layer responses $h _ { \mathrm { f i n a l } , t ^ { \prime } } | _ { t ^ { \prime } \geq t }$ . The function $f _ { \ell }$ is implicitly defined by the remaining network and is generally nonlinear. In this section, we systematically analyze the mathematical principles underlying the Jlens, corroborate its first-order approximation to the nonlinear mapping $f _ { \ell } ,$ , and confirm its meaning. Specifically, given the local function-fitting capability of the Jacobian matrix, we mathematically interpret the construction of the J-lens, i.e., the statistical expectation of the Jacobian over the training distribution, as a process of fitting $f _ { \ell }$ from local approximation to global (Stein, 1981). In addition, we analyze its approximation bias from both theoretical and experimental perspectives, validating the reliability and effectiveness of the theoretical interpretation in this section while providing a new perspective on the failure of the J-lens in early layers.

## 3.1 LOCAL VIEW: TANGENT LINEARIZATION

For a smooth map $f : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$ , the Jacobian $J _ { f } ( h _ { 0 } ) \in \mathbb { R } ^ { d \times d }$ has entries $[ J _ { f } ( h _ { 0 } ) ] _ { i j } = \partial f _ { i } / \partial h _ { j }$ at $h _ { 0 }$ . The first-order Taylor expansion is

$$
f ( h _ { 0 } + \delta ) = f ( h _ { 0 } ) + J _ { f } ( h _ { 0 } ) \cdot \delta + o ( \| \delta \| ) .\tag{3}
$$

Thus the Jacobian can be regarded as the local sensitivity map: a small intervention δ at $h _ { \ell , t }$ changes the predicted future response, to first order, by $J _ { f _ { \ell } } ( h _ { \ell , t } ) \cdot \delta$ . The approximation is asymptotically exact as $\| \delta \|  0$ , but its usable radius shrinks where curvature is large, and the matrix is pointwise rather than global. Therefore, a single Jacobian matrix solely estimates the first-order approximation with tangent linearization in a local range.

## 3.2 GLOBAL VIEW: LEAST-SQUARES READOUT

The global view fixes one affine map for the whole distribution. Let $h \sim p , y = f ( h ) , \mu = \mathbb { E } [ h ]$ , and $\Sigma = \operatorname { C o v } ( h )$ , with $\Sigma \succ 0$ . Writing $\tilde { h } = h - \mu$ and $\tilde { y } = y - \mathbb { E } [ y ]$ , a global first-order approximation of the function f can be treated as a least-squares estimation:

$$
\operatorname* { m i n } _ { W , b } \ { \mathbb { E } } \| y - ( W h + b ) \| ^ { 2 }\tag{4}
$$

with the normal-equation solution

$$
\begin{array} { r } { \left\{ \begin{array} { l l } { W ^ { * } = \mathbb { E } [ \tilde { y } \tilde { h } ^ { \top } ] \Sigma ^ { - 1 } = \operatorname { C o v } ( y , h ) \operatorname { C o v } ( h ) ^ { - 1 } } \\ { b ^ { * } = \mathbb { E } [ y ] - W ^ { * } \mu . } \end{array} \right. } \end{array}\tag{5}
$$

The residual is orthogonal to all affine functions of $h ,$ so $W ^ { * } h + b ^ { * }$ is the $L ^ { 2 }$ projection of the regression function $\grave { m ( h ) } = \mathbb { E } [ y \mid h ]$ onto affine functions. This optimum requires no Gaussianity and no differentiability; it is a statistical compromise slope over the data distribution.

## 3.3 SCORE IDENTITY AND THE STEIN BRIDGE

The link between local derivatives and global least squares is integration by parts. Assume that h has a smooth positive density p on $\mathbb { R } ^ { d }$ , that $f : \mathbb { R } ^ { d }  \mathbf { \bar { \mathbb { R } } } ^ { d }$ is componentwise differentiable, and that $f$ grows slowly enough relative to p that boundary terms vanish (Hyvarinen & Dayan, 2005). The¨ score function is

$$
s ( h ) : = \nabla \log p ( h ) = \frac { \nabla p ( h ) } { p ( h ) } .\tag{6}
$$

For one component $f _ { i }$ and one input coordinate $h _ { j }$

$$
\mathbb { E } [ \partial _ { j } f _ { i } ( h ) ] = \int _ { \mathbb { R } ^ { d } } \partial _ { j } f _ { i } ( h ) p ( h ) d h = - \int _ { \mathbb { R } ^ { d } } f _ { i } ( h ) \partial _ { j } p ( h ) d h ,\tag{7}
$$

where the boundary term vanishes under the growth assumption, where $f _ { i } ( h ) p ( h ) \to 0 \mathrm { a s } \| h \| \to \infty$ for every component i. Since $\partial _ { j } p = p \partial _ { j } \log p = p s _ { j }$ as shown in Eq. 6,

$$
\mathbb { E } [ \partial _ { j } f _ { i } ( h ) ] = - \mathbb { E } [ f _ { i } ( h ) s _ { j } ( h ) ] .\tag{8}
$$

Stacking over i and j gives the multivariate score identity:

$$
\mathbb { E } [ J _ { f } ( h ) ] = - \mathbb { E } \big [ f ( h ) s ( h ) ^ { \top } \big ] .\tag{9}
$$

The left side averages local slopes. The right side says that this average is determined by correlations between function values and the local tilt of the density. Derivative information is therefore encoded in how the density changes, not only in how the function bends.

Equation equation 9 is distribution-free once the regularity conditions hold. The Gaussian case is special because the score is affine. If $h \sim { \mathcal { N } } ( { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } )$ with $\Sigma ^ { ' } \succ 0$ , then

$$
\log p ( h ) = \mathsf { c o n s t - } \frac { 1 } { 2 } ( h - \mu ) ^ { \top } \Sigma ^ { - 1 } ( h - \mu ) , \qquad s ( h ) = - \Sigma ^ { - 1 } ( h - \mu ) .\tag{10}
$$

Substituting into equation 9,

$$
\begin{array} { r } { \mathbb { E } [ J _ { f } ( h ) ] = - \mathbb { E } \Big [ f ( h ) \left( - \Sigma ^ { - 1 } ( h - \mu ) \right) ^ { \top } \Big ] = \mathbb { E } \big [ f ( h ) ( h - \mu ) ^ { \top } \big ] \Sigma ^ { - 1 } . } \end{array}\tag{11}
$$

Because $\mathbb { E } [ s ( h ) ] = 0$ , replacing f by $\tilde { f } = f - \mathbb { E } [ f ( h ) ]$ does not change the right-hand side. Hence

$$
\begin{array} { r } { \mathbb { E } [ J _ { f } ( h ) ] = \mathbb { E } [ \tilde { f } ( h ) \tilde { h } ^ { \top } ] \Sigma ^ { - 1 } = \operatorname { C o v } ( f ( h ) , h ) \Sigma ^ { - 1 } = W ^ { * } . } \end{array}\tag{12}
$$

This is the Stein bridge: under Gaussian inputs, the average Jacobian is exactly the population leastsquares slope.

Theorem 1 (Stein bridge (Stein, 1981)). Let $h \sim { \mathcal { N } } ( { \boldsymbol { \mu } } , { \boldsymbol { \Sigma } } )$ with $\Sigma \succ 0 ,$ , and let f satisfy the mild growth and differentiability conditions above. Then

$$
\mathbb { E } [ J _ { f } ( h ) ] = \operatorname { C o v } ( f ( h ) , h ) \operatorname { C o v } ( h ) ^ { - 1 } = W ^ { * } .\tag{13}
$$

Under Gaussian inputs, the averaged Jacobian matrix is exactly equal to the population least-squares slope. Therefore, the J-lens can be understood as an approximation to the optimal linear readout. When the input distribution is close to Gaussian, this approximation is accurate; when the input distribution deviates substantially from Gaussian, an additional residual term is introduced into the score function, inducing a systematic bias between the averaged Jacobian and the optimal linear readout. In the subsequent section, we will analyze this bias and evaluate the layer-level valid range of our theoretical interpretation.

## 3.4 BIAS ANALYSIS

The J-lens involves two approximations. The first replaces the nonlinear mapping with a first-order linear transfer matrix, and the second approximates the globally optimal transfer matrix by the expectation of local Jacobians. Nonlinear and non-Gaussian biases are therefore systematically introduced and amplified as the distance between source and target layers increases. In this section, we analyze in detail the specific effects of these biases from both theoretical and experimental perspectives and provide a new explanation for the early failure of the J-lens.

![](images/a76d16413fc907721ab16051efa467fcc4ede7f6f765adf502489c3f2aa3fd04.jpg)

![](images/6946dd7d843f85df3fa78778e9cf093c9820832ef04da7f87e9885287554d366.jpg)  
Figure 1: Layer-wise bias analysis of J-lens for Qwen3-8B.

## 3.4.1 NONLINEAR BIAS

The Stein bridge compares two slopes, $W ^ { * }$ and $\mathbb { E } [ J _ { f } ( h ) ]$ ]. A separate error remains even if the best possible slope is used: a linear map cannot reproduce curvature. Let $\mu = \mathbb { E } [ h ] , \delta = h - \mu ,$ $J = \bar { J } _ { f } ( \mu )$ , and let $H _ { i } = \nabla ^ { 2 } f _ { i } ( \mu )$ be the Hessian of output component i. Write $\bar { H } [ \delta , \delta ]$ for the vector with components $\delta ^ { \top } H _ { i } \delta$ and $H [ \Sigma ] = \mathbb { E } [ H [ \delta , \delta ] ]$ , whose i-th component is $\operatorname { t r } ( H _ { i } \Sigma )$ . The intrinsic nonlinear error of the optimal affine readout is

$$
\varepsilon _ { \mathrm { l i n } } ^ { 2 } = \mathbb { E } \big \| f ( h ) - \big ( \mathbb { E } [ f ( h ) ] + W ^ { * } ( h - \mu ) \big ) \big \| ^ { 2 } .\tag{14}
$$

This is the residual that no choice of linear slope can remove.

## 3.4.2 NON-GAUSSIAN BIAS

Away from Gaussianity, decompose the score into its Gaussian part plus a residual:

$$
s ( h ) = - \Sigma ^ { - 1 } ( h - \mu ) + r ( h ) , \qquad r ( h ) = 0 \iff h { \mathrm { ~ i s ~ G a u s s i a n } } .\tag{15}
$$

Substitution into the score identity gives the exact bias formula

$$
\begin{array} { r } { \epsilon _ { \mathrm { G a u } } : = W ^ { * } - \mathbb { E } [ J _ { f } ( h ) ] = \mathbb { E } \Big [ \tilde { f } ( h ) r ( h ) ^ { \top } \Big ] . } \end{array}\tag{16}
$$

The gap is therefore not controlled by nonlinearity alone. It is the covariance between the centered output fluctuation and the pointwise non-Gaussianity of the density. A standard Cauchy–Schwarz bound yields

$$
\| \epsilon _ { \mathrm { G a u } } \| _ { F } \leq \sqrt { \mathbb { E } \| \tilde { f } ( h ) \| ^ { 2 } } \sqrt { \mathbb { E } \| r ( h ) \| ^ { 2 } }\tag{17}
$$

where the second factor is a distribution-level non-Gaussianity measure.

## 3.4.3 OVERALL BIAS

As a cascade bias system without correlation, the overall system bias can be calculated by a sum of the squared biases:

$$
\epsilon _ { \bar { J } } ^ { 2 } = \epsilon _ { \mathrm { { l i n } } } ^ { 2 } + \| \epsilon _ { \mathrm { { G a u } } } \| _ { F } ^ { 2 } .\tag{18}
$$

To verify the consistency between theory and experiments, we perform a layer-wise quantitative comparison between the theoretical error and the empirical error of the J-lens. The results show that both the nonlinear bias and the non-Gaussian error decrease as the source layer approaches the target layer, and that empirical measurements of the overall error also validate our theory. This provides an effective validation of our mathematical interpretation of the J-lens and explains the early failure of the J-lens from the perspective of mathematical bias: large nonlinear and distributional biases cause the mapped results at the target layer to deviate substantially from the confidence interval and lose their intended meaning.

![](images/fbaeb7fbabf272b00cd1a998694bb567317b933985569eab37ec1de323449fdd.jpg)

![](images/053efac51c1af2701773fe40d2540211a8ecd549c08d82ac7aa40cc88f44adad.jpg)  
Figure 2: Jacobian energy analysis along the layers. The left figure indicates the relationship between absolute energy and the layers. The right figure indicates the energy concentration across layers, where the top x% of elements accumulate y% of the total energy of the layer.

## 4 JACOBIAN ENERGY STRUCTURE IN J-LENS: SHORT HORIZON AND SPARSE CONCEPTS

This section moves from the validity of the averaged operator $J _ { \ell }$ to the positional structure that the average hides. The theory above treats $J _ { \ell }$ as an aggregate first-order readout, while we further ask which source-target pairs actually carry that aggregate.

## 4.1 SETUP

The positional Jacobian energy is adopted as a guide quantity, which measures how strongly a perturbation at one position propagates to a later final-layer position. For layer ℓ, source position t, and target position $t ^ { \prime } ,$ , define Jacobian energy as:

$$
E _ { \ell } ( t , t ^ { \prime } ) = \left\| \frac { \partial h _ { \mathrm { f i n a l } , t ^ { \prime } } } { \partial h _ { \ell , t } } \right\| ^ { 2 } .\tag{19}
$$

Since the J-lens is the average of the Jacobian matrices over all positions within a layer, analyzing the Jacobian energy at each position is important for understanding the representational tendency of the J-lens. It reveals which positions receive more attention from the J-lens, and the content at these positions with higher energy would influence the readout of the J-lens more strongly.

## 4.2 LAYER-WISE DECAY AND CONCENTRATION

As shown in Figure 2, two regularities across the layers appear before structural analysis. First, total energy decays with depth. Early layers sit upstream of many subsequent transformations and therefore have larger influences; late layers are closer to the output and act through shorter residual pathways. Second, energy is heavy-tailed across position pairs. A small top fraction of pairs ac counts for most of the mass, and the concentration increases with depth. Thus the averaged object $J _ { \ell }$ is not an even superposition of weak effects but the domination of a minority of position pairs. This also explains why the J-lens, although constructed from the output expectation over all future positions, can stably read out valid concepts.

## 4.3 JACOBIAN STRUCTURE: SHORT HORIZONS AND SPARSE CONCEPT

To further investigate the distributional pattern of the highly concentrated Jacobian energy, we conduct visualization studies to analyze the structure of the J-lens and the causal implications of its actual representations. Specifically, we compute the Jacobian energy for every position pair in each layer during the construction of the J-lens and plot a three-dimensional heatmap over source posi tions, target positions, and layers. Since an overly dense point cloud is not suitable for display, and the concentration effect of energy distribution indicates that a small number of high-energy points are sufficient to reveal the structure, we display the 400 (approximately the top 13%) position pairs with the highest energy in each layer, and hide the remaining points.

![](images/659745f1a04330e473302cc97ccaf53af23aab563ffd8a793d95f20441434566.jpg)  
Figure 3: A 3D heatmap for the energy distribution of the Jacobian at each layer and position.

Figure 3 demonstrates that high-energy pairs separate into two patterns. The first is the diagonal one: $E _ { \ell } ( t , t )$ is large, meaning that a position primarily preserves or prepares the next token. Diagonal energy is especially prominent in later layers, where the remaining computation is short and output-oriented. Since the elements on the diagonal and adjacent positions represent short-term expressions, these causal results are defined as short horizons. The second pattern is horizontal or vertical positions. Some source positions influence many future positions, behaving as broadcast origins; some target positions receive influence from many sources, behaving as integration points. When these positions are stable across prompts and attach to interpretable tokens, we group them as the sparse concept positions.

According to the visualization results above, let $t ^ { * }$ denote the positions of sparse concept, the formalization of J-lens can be approximated as:

$$
J _ { \ell } = \mathbb { E } _ { t , t ^ { \prime } \ge t , \mathrm { p r o m p t } } \left[ \frac { \partial h _ { \mathrm { f i n a l } , t ^ { \prime } } } { \partial h _ { \ell , t } } \right] \approx \mathbb { E } _ { t , t ^ { \prime } \approx t , \mathrm { p r o m p t } } \left[ \frac { \partial h _ { \mathrm { f i n a l } , t ^ { \prime } } } { \partial h _ { \ell , t } } \right] + \mathbb { E } _ { t , t ^ { \prime } = t ^ { * } , \mathrm { p r o m p t } } \left[ \frac { \partial h _ { \mathrm { f i n a l } , t ^ { \prime } } } { \partial h _ { \ell , t } } \right] \mathrm { . }\tag{20}
$$

Moreover, based on the derivation that interprets the J-lens as the aggregate expectation over future outputs in Section 3, we can reduce this future output to two positional modes: short horizons and sparse concepts. This is precisely the core understanding and interpretation of the meaning of the J-lens readout in this paper.

## 5 EXPERIMENTS

Based on the theoretical and experimental study above, we propose improvements to the J-lens to further validate our theory. Since experiments show that the Jacobian energy exhibits a pronounced concentration pattern, and that its concentration in the short horizon and sparse concept modes is precisely the design goal of the J-lens, the Jacobians at the remaining positions act as noise under this view. We therefore propose a simple but effective entry filtering strategy: only the Jacobian matrices at the top-j fraction of position pairs $( \boldsymbol { \mathrm { p } } , \boldsymbol { \mathrm { p } } ^ { \prime } )$ with the largest energy are used to construct the J-lens. We also attempt to decouple the two readout modes of the J-lens, short horizon and sparse concept. Since heatmaps show that the Jacobian energy concentrates on the diagonal and on horizontal and vertical lines at specific positions, we apply masks with specific shapes to decouple the J-lens. For example, we apply a mask matrix that sets the diagonal to zero and keeps the remaining entries as one, with the expectation of retaining only sparse concepts.

## 5.1 EXPERIMENTS SETTINGS

We utilized Qwen3-8B (Team, 2025) in the experiments. Following the previous study, Wikitext (Merity et al., 2017) is introduced for J-lens construction, and six tasks including association, multihop reasoning, multilingual, ordering operations, poetry, and typos are evaluated (Gurnee et al., 2026).

Table 1: Main results across several tasks. Bold denotes the best. Vanilla means the original J-lens. All J-lenses are trained on the same 200 examples with the same configuration.
<table><tr><td></td><td colspan="2">association</td><td colspan="2">multihop</td><td colspan="2">multilingual</td><td colspan="2">order-ops</td><td colspan="2">poetry</td><td colspan="2">typo</td><td colspan="2">avg</td></tr><tr><td>Method</td><td>SHL</td><td>ICR</td><td>SHL</td><td>ICR</td><td>SHL</td><td>ICR</td><td>SHL</td><td>ICR</td><td>SHL</td><td>ICR</td><td>SHL</td><td>ICR</td><td>SHL</td><td>ICR</td></tr><tr><td></td><td colspan="10">Qwen3-8B</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla</td><td>1.01</td><td>4.70</td><td>1.65</td><td>16.58</td><td>0.66</td><td>40.07</td><td>0.66</td><td>12.82</td><td>0.99</td><td>19.54</td><td>0.62</td><td>14.40</td><td>0.94</td><td>18.71</td></tr><tr><td> $\mathrm { F i l t e r } _ { \mathrm { t o p j } = 0 . 5 }$ </td><td>1.04</td><td>4.93</td><td>1.71</td><td>15.58</td><td>0.68</td><td>38.53</td><td>0.68</td><td>12.56</td><td>1.00</td><td>20.17</td><td>0.67</td><td>15.69</td><td>0.98</td><td>18.60</td></tr><tr><td> $\mathrm { F i l t e r } _ { \mathrm { t o p j = 0 . 2 } }$ </td><td>1.05</td><td>6.36</td><td>1.73</td><td>16.35</td><td>0.69</td><td>48.40</td><td>0.67</td><td>17.36</td><td>1.06</td><td>21.13</td><td>0.69</td><td>19.03</td><td>1.00</td><td>22.14</td></tr><tr><td> $\mathrm { F i l t e r } _ { \mathrm { t o p j = 0 . 1 } }$ </td><td>1.06</td><td>6.44</td><td>1.72</td><td>15.78</td><td>0.68</td><td>47.38</td><td>0.68</td><td>18.26</td><td>1.04</td><td>21.17</td><td>0.72</td><td>19.25</td><td>1.00</td><td>22.00</td></tr><tr><td> $\mathrm { F i l t e r } _ { \mathrm { t o p j = 0 . 0 1 } }$ </td><td>1.05</td><td>4.28</td><td>1.69</td><td>13.26</td><td>0.67</td><td>31.56</td><td>0.69</td><td>11.82</td><td>1.03</td><td>18.96</td><td>0.69</td><td>15.73</td><td>0.98</td><td>16.45</td></tr><tr><td> $\mathrm { F i l t e r _ { w / o } D i a g }$ </td><td>0.59</td><td>4.20</td><td>1.15</td><td>16.75</td><td>0.50</td><td>30.44</td><td>0.38</td><td>6.24</td><td>0.77</td><td>16.91</td><td>0.33</td><td>10.78</td><td>0.63</td><td>15.02</td></tr><tr><td> $\mathrm { F i l t e r } _ { \mathrm { D i a g } }$ </td><td>1.08</td><td>4.72</td><td>1.70</td><td>13.28</td><td>0.68</td><td>33.61</td><td>0.74</td><td>12.75</td><td>1.05</td><td>18.84</td><td>0.69</td><td>16.11</td><td>1.00</td><td>17.07</td></tr></table>

## 5.2 METRICS

In this paper, we design two metrics named short-horizon layers (SHL) and intermediate-concept recall rate (ICR) to evaluate the capability of J-lens on next-token prediction and intermediate concept recalling individually. Both metrics operate on the full position-layer lens readout: for a prompt of T tokens, the lens yields a vocabulary distribution $\mathbf { y } _ { t } ^ { ( l ) }$ at every layer $l \in \{ 1 , \ldots , L - 1 \}$ and position $t \in \{ 1 , \ldots , T \}$

Short-horizon layers. For each position t with true next token $x _ { t + 1 }$ , we count the length of the trailing run of layers whose lens top-1 prediction matches $x _ { t + 1 } { : }$

$$
S H L _ { t } = \operatorname* { m a x } \{ r : \arg \operatorname* { m a x } _ { \boldsymbol { v } } q _ { t , \boldsymbol { v } } ^ { ( l ) } = x _ { t + 1 } \forall l \in \{ L - r + 1 , \ldots , L \} \} .\tag{21}
$$

Averaging $S H L _ { t }$ over positions measures how early the lens readout converges to the model’s next token prediction, demonstrating the short-horizon ability of a J-lens.

Intermediate-concept recall rate. For prompts annotated with intermediate concepts $\begin{array} { r l } { { \mathcal { Z } } } & { { } = } \end{array}$ $\{ w _ { 1 } , \ldots , w _ { m } \}$ , we count grid cells whose lens top- $K \left( K = 1 0 \right)$ contains any token of concept:

$$
I C R = \big | \big \{ ( l , t ) : \mathbb { Z } \cap \mathrm { t o p } { \bf - } K \big ( \mathbf { q } _ { t } ^ { ( l ) } \big ) \neq \emptyset \big \} \big | ,\tag{22}
$$

which measures how broadly the J-lens reads the required sparse concept.

## 5.3 RESULTS

Results in Table 1 presents the results for methods on several benchmarks in this paper. We can conclude the following finding. First, since the filtering methods with topj further strengthens the concentration of the Jacobian energy in the J-lens and enables its representational capacity to exceed that of the unfiltered baseline, J-lens can focus more on the highlight sparse concept and outperform the vanilla on most tasks. Second, experiments on the diagonal filtering reveal that the coupling between the two patterns is deeper than the visualized results demonstrate: the ability of the J-lens to predict the next token is weakened but does not disappear, while its performance in generating sparse concepts also declines. This points to a direction for future work.

## 6 CONCLUSION

We interpret the J-lens as an averaged first-order transfer operator from intermediate activations to expected future readouts: the Jacobian matrix is the locally optimal linearization, least squares gives the globally optimal affine readout, and the Stein bridge makes them coincide under Gaussian inputs. Nonlinear curvature and non-Gaussian score residuals explain systematic bias and provide an explanation for the early failure of the J-lens. We then further analyze the structure of the Jacobian energy. The Jacobian energy decays with depth, concentrates in a small number of source-target pairs, and splits into diagonal short horizons and sparse concept positions. Finally, we introduce improvements and decoupling methods for the J-lens. Energy filtering preserves or improves nexttoken consistency and the recall rate of intermediate concepts, thereby validating our interpretation of the meaning of the J-lens readout. Finally, we acknowledge that this work still has limitations and is still in progress, which does not represent the final results.

## REFERENCES

Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Workshop Track Proceedings. OpenReview.net, 2017. URL https: //openreview.net/forum?id=HJ4-rAVtl.

Nora Belrose, Zach Furman, Logan Smith, Danny Halawi, Igor Ostrovsky, Lev McKinney, Stella Biderman, and Jacob Steinhardt. Eliciting latent predictions from transformers with the tuned lens. CoRR, abs/2303.08112, 2023. doi: 10.48550/ARXIV.2303.08112. URL https://doi. org/10.48550/arXiv.2303.08112.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin (eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html.

Arthur Conmy, Augustine N. Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adria\` Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine (eds.), Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper\_files/paper/2023/hash/ 34e1dbe95d34d7ebaf99b9bcaeb5b2be-Abstract-Conference.html.

Mor Geva, Avi Caciularu, Kevin Ro Wang, and Yoav Goldberg. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7- 11, 2022, pp. 30–45. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022. EMNLP-MAIN.3. URL https://doi.org/10.18653/v1/2022.emnlp-main.3.

Asma Ghandeharioun, Avi Caciularu, Adam Pearce, Lucas Dixon, and Mor Geva. Patchscopes: A unifying framework for inspecting hidden representations of language models. In Ruslan Salakhutdinov, Zico Kolter, Katherine A. Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, volume 235 of Proceedings of Machine Learning Re search, pp. 15466–15490. PMLR / OpenReview.net, 2024. URL https://proceedings. mlr.press/v235/ghandeharioun24a.html.

Wes Gurnee, Nicholas Sofroniew, Adam Pearce, Mateusz Piotrowski, Isaac Kauvar, Runjin Chen, Anna Soligo, Paul C. Bogdan, Euan Ong, Rowan Wang, Ben Thompson, David Abrahams, Sub hash Kantamneni, Emmanuel Ameisen, Joshua Batson, and Jack Lindsey. Verbalizable representations form a global workspace in language models. CoRR, abs/2607.15495, 2026. doi: 10. 48550/ARXIV.2607.15495. URL https://doi.org/10.48550/arXiv.2607.15495.

John Hewitt and Percy Liang. Designing and interpreting probes with control tasks. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, EMNLP-IJCNLP 2019, Hong Kong, China, November 3-7, 2019, pp. 2733–2743. Association for Computational Linguistics, 2019. doi: 10.18653/V1/D19-1275. URL https://doi.org/10.18653/v1/D19-1275.

Aapo Hyvarinen and Peter Dayan. Estimation of non-normalized statistical models by score match-¨ ing. Journal ofMachine Learning Research, 6(4):695–709, 2005.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper\_files/paper/2022/ hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017. URL https: //openreview.net/forum?id=Byj72udxe.

nostalgebraist. Interpreting gpt: The logit lens. LessWrong, 2020. https://www.lesswrong. com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens.

Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. In-context learning and induction heads. CoRR, abs/2209.11895, 2022. doi: 10. 48550/ARXIV.2209.11895. URL https://doi.org/10.48550/arXiv.2209.11895.

Theodora-Mara Pˆıslar, Sara Magliacane, and Atticus Geiger. Combining causal models for more accurate abstractions of neural networks. In Biwei Huang and Mathias Drton (eds.), Causal Learning and Reasoning, Lausanne, Switzerland, 7-9 May 2025, volume 275 of Proceedings of Machine Learning Research, pp. 114–138. PMLR, 2025. URL https://proceedings. mlr.press/v275/pislar25a.html.

Lee Sharkey, Bilal Chughtai, Joshua Batson, Jack Lindsey, Jeffrey Wu, Lucius Bushnaq, Nicholas Goldowsky-Dill, Stefan Heimersheim, Alejandro Ortega, Joseph Isaac Bloom, Stella Biderman, Adria Garriga-Alonso, Arthur Conmy, Neel Nanda, Jessica Rumbelow, Martin Wattenberg, Nandi\` Schoots, Joseph Miller, William Saunders, Eric J. Michaud, Stephen Casper, Max Tegmark, David Bau, Eric Todd, Atticus Geiger, Mor Geva, Jesse Hoogland, Daniel Murfet, and Tom McGrath. Open problems in mechanistic interpretability. Trans. Mach. Learn. Res., 2025, 2025. URL https://openreview.net/forum?id=91H76m9Z94.

Charles M. Stein. Estimation of the mean of a multivariate normal distribution. The Annals of Statistics, 9(6):1135–1151, 1981.

Qwen Team. Qwen3 technical report. CoRR, abs/2505.09388, 2025. doi: 10.48550/ARXIV.2505. 09388. URL https://doi.org/10.48550/arXiv.2505.09388.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. Emergent abilities of large language models. Trans. Mach. Learn. Res., 2022, 2022. URL https://openreview.net/forum?id= yzkSU5zdwD.