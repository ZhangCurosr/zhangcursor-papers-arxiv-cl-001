# SPECIFICATION-AWARE PREDICTIVE UNCERTAINTYDECOMPOSITION

Roman Vashurin<sup>1</sup>, Mikhail Seleznyov<sup>2,5</sup>, Assel Yermekova<sup>3</sup>, Fedor Chernogorskii<sup>1</sup>, Danila Zykov<sup>4</sup>, Elena Tutubalina<sup>4</sup>, Alexander Panchenko<sup>2,5</sup>, Preslav Nakov<sup>1</sup>, Nikita Kotelevskii<sup>1</sup>, Maxim Panov<sup>1</sup>

<sup>1</sup>MBZUAI <sup>2</sup>AXXX <sup>3</sup>Huawei Technologies Ltd. <sup>4</sup>HSE <sup>5</sup>S-NLP Roman.Vashurin@mbzuai.ac.ae

## ABSTRACT

Classical decompositions of predictive uncertainty, such as the informationtheoretic one, are defined for a fixed representation x of the input object. They do not consider that x may be incomplete, nor do they measure how much the missing information can affect the prediction. In Large Language Models, users can clarify an underspecified input without any retraining. The clarified input can describe the user’s unobserved intent z<sup>∗</sup> more accurately and change the predictive uncertainty about the outcome. In this paper, we formalize this setting probabilistically and derive a three-term decomposition of prediction risk into specification, residual aleatoric, and epistemic uncertainties. We show that at the original information level, when only x is available, the aleatoric uncertainty, represented by a Bayes risk, has two components. The first, which we call specification uncertainty, is the expected reduction in this Bayes risk from revealing the intent. The second is the residual aleatoric uncertainty, i.e., the Bayes risk that remains once the intent is known, averaged over its possible values z given x. The distinction is important, as specification uncertainty informs the decision to request clarification before making a prediction. We build approximations of these components and connect them to existing uncertainty measures. Our experiments across different domains and modalities show distinct roles for the three components. Specification uncertainty is the most informative for detecting underspecified inputs, while residual aleatoric uncertainty is the most informative for detecting in-distribution prediction errors.

## 1 INTRODUCTION

As artificial intelligence (AI) systems, especially generative ones, are widely adopted (Bick et al., 2026), there is a growing demand for methods and practices to control their reliability. One such direction is predictive Uncertainty Quantification (UQ; Hüllermeier & Waegeman, 2021), which supplements the original model’s output with a value (typically a scalar) describing the model’s certainty in its prediction. Based on this value, a practitioner can make a practical decision, such as to abstain from the prediction, request additional information, or proceed with the prediction if the uncertainty is acceptable (Geifman & El-Yaniv, 2017; Mozannar & Sontag, 2020; Naszadi et al., 2023). For Large Language Models (LLMs) (Brown et al., 2020), requesting additional information is particularly natural, since a user can clarify the input without retraining the model. For example, given the prompt “Who won the race?”, a model may either answer right away or first ask which race is meant (see Figure 1). Therefore, uncertainty in prediction may come from different sources, and different sources may call for different actions (Sale et al., 2026).

Two types of uncertainty are widely distinguished (Hüllermeier & Waegeman, 2021). Aleatoric uncertainty is irreducible with respect to the considered informational constraints, while epistemic uncertainty is reducible and characterizes the lack of knowledge about the underlying ground-truth data-generation process. Different formalizations of these types lead to concrete UQ measures (Gal & Ghahramani, 2016; Kendall & Gal, 2017; Lakshminarayanan et al., 2017; Kotelevskii et al., 2025; 2022; Sale et al., 2023; 2024; Hofman et al., 2024; Schweighofer et al., 2025).

![](images/3cefb918a5124aec556493db429248df2240bea55b428c055b74280c37b394d1.jpg)  
Figure 1: Examples of observed inputs x (the ones provided by users) and possible values of the unobserved intent z. Different intended questions (top) or complete images (bottom) can be consistent with the same observed input. The actual value z<sup>∗</sup> remains unobserved.

For LLMs, many UQ methods have been proposed (Lin et al., 2024; Farquhar et al., 2024; Tian et al., 2023; Vashurin et al., 2025), including decompositions into aleatoric and epistemic uncertainty (Malinin & Gales, 2021; Ma et al., 2025). However, these decompositions are defined for a given input representation and do not ask whether clarifying the input could improve the prediction.

Recent work considered uncertainty arising from underspecified inputs (Hou et al., 2024; Walha et al., 2026). These approaches generate clarifications of the input and treat the variability of predictions across them, quantified by Bregman information, as aleatoric uncertainty, and the expected uncertainty remaining after clarification as epistemic uncertainty. However, the outcome can be random even for a fully clarified input, so the remaining uncertainty is not necessarily epistemic. Moreover, both approaches use a single fixed model and thus ignore uncertainty about its parameters, which the information-theoretic decomposition treats as epistemic.

We present an alternative view on this distinction. We assume that each observed input x (possibly only partially) describes the true intent z<sup>∗</sup>, which itself is never observed, and ask how much the prediction would improve by revealing this intent. Measuring uncertainty by the expected loss of a prediction (also called risk), we obtain a three-term decomposition of this risk into Specification, Epistemic, and residual Aleatoric uncertainty (SEA). Specification uncertainty is the expected reduction in Bayes risk (the risk of the data-generation process at x) from revealing the intent, that is, the expected gain from clarification. When it is large, it is worth requesting clarification before providing an answer. At the original information level (i.e., when only x is available), specification uncertainty is only a part of the aleatoric uncertainty.

We summarize our contributions below.

• We derive a new UQ decomposition, relevant in particular for LLMs, that splits the predictive uncertainty into three terms by separating specification uncertainty from both the residual aleatoric uncertainty and the epistemic one. When the input fully specifies the intent, the specification term vanishes, and the decomposition reduces to the risk-based one (Kotelevskii et al., 2025), generalizing the information-theoretic uncertainty decomposition; see Section 3.1.

• For a specific class of loss functions, proper scoring rules, we derive closed-form uncertainty measures and, following the risk-based approach, their Bayesian approximations. These approximations generalize the clarification-based decompositions previously derived for a single fixed model (Hou et al., 2024; Walha et al., 2026); see Section 3.2.

• In our experiments, we demonstrate that each of the derived components is well-suited for a particular type of decision in downstream problems. Specification uncertainty is the most informative of the three components for detecting underspecified inputs, while residual aleatoric uncertainty is the most informative component for detecting prediction errors for in-distribution data; see Section 4 and Appendix K.

## 2 RELATED WORK

Predictive uncertainty decomposition. There are several approaches to formalize the decomposition into aleatoric and epistemic uncertainty and estimate the corresponding components. The information theoretic approach considers a Bayesian (or, more generally, second-order) model. It associates the conditional mutual information between predictions and model parameters with epistemic uncertainty, and the expected entropy of the prediction given the parameters with aleatoric uncertainty (Houlsby et al., 2011). Estimating the epistemic term, therefore, requires a second-order learner, such as a Bayesian model or an ensemble (Sale et al., 2023; 2024). The risk-based formulation generalizes this approach. It decomposes the pointwise prediction risk into the Bayes risk (aleatoric uncertainty) and the excess risk (epistemic uncertainty). These components can be approximated with Bayesian models (Kotelevskii et al., 2025; Schweighofer et al., 2025), which recover the decomposition of Houlsby et al. (2011) as a special case. Another strategy for approximating these terms uses specialized deterministic auxiliary models (Kotelevskii et al., 2022; Lahlou et al., 2023). Other methods estimate uncertainty with a single deterministic network by modifying its architecture or fitting a density model on its features (Van Amersfoort et al., 2020; Liu et al., 2020; Mukhoti et al., 2021). For autoregressive generation, the information-theoretic decomposition has been extended to the token and sequence levels (Malinin & Gales, 2021), still requiring a Bayesian model or an ensemble for estimation. Another approach, LogTokU (Ma et al., 2025), estimates aleatoric and epistemic uncertainties from logits without repeated sampling by interpreting them as evidence for a Dirichlet distribution.

These decompositions quantify uncertainty at a fixed level of input information, when the representation of an input object (or its prompt) x is given. They do not consider how the risk and the predictive uncertainty would change if x were clarified. Interactive LLMs, however, allow users to clarify their inputs without retraining the model, which makes this question natural.

Specification uncertainty. Sale et al. (2026) argue that the aleatoric/epistemic distinction depends on the information available in a given setting, for example, on whether an ambiguous prompt can be clarified. Hou et al. (2024) propose input-clarification ensembling, which decomposes predictive uncertainty using generated clarifications of the prompt. In their decomposition, the mutual information between the prediction and the clarification is treated as aleatoric uncertainty caused by input ambiguity, while the expected entropy of the response given a clarification is treated as epistemic uncertainty. Walha et al. (2026) keep this interpretation and generalize it with functional Bregman information, instantiated with kernel-based von Neumann entropy. Section 3 shows that their decomposition is a special case of ours for a single fixed model and that its terms estimate specification and residual aleatoric uncertainty. In contrast, we split the Bayes risk at the level of x into two parts. Specification uncertainty is the expected reduction in Bayes risk from revealing the intent, and residual aleatoric uncertainty is the expected Bayes risk after the intent is revealed. The excess risk of the predictor is still treated as epistemic uncertainty.

## 3 RISK DECOMPOSITION WITH UNOBSERVED INTENTS

In this work, we consider a setup in which the observed representation $x \in \mathcal { X }$ of an object may be incomplete and can be clarified by additional information. Our goal is to quantify how much such a clarification is expected to improve the prediction, and to separate this gain from other sources of predictive uncertainty.

In what follows, by intent, we mean a fully specified input that x describes only partially. We denote the user’s true but unobserved intent by $z ^ { \ast } \in \mathcal { Z }$ . For example, a user may have a specific question $z ^ { * }$ in mind but write a less specific prompt x. Similarly, x could be a masked image or an audio recording with a missing interval, and $z ^ { * }$ the original image or the full recording. Several intents can be consistent with the same x (see Figure 1), and the conditional distribution $p ( z \mid x )$ describes which values $z ^ { * }$ is likely to take. In these examples, $\mathcal { Z } = \mathcal { X }$ , since a clarified prompt is again a prompt, and a completed image or recording has the same format as the masked one. Our framework, however, does not require the two spaces to coincide.

We assume that clarification does not change the output space Y, as otherwise the prediction risks before and after clarification could not be compared. For a classification task, this means that the set of output classes must be the same for x and z. Finally, we assume that once z is known, x provides no additional information about $Y$ . Formally, $Y \perp X \mid Z , { \mathrm { i . e . , } } p ( y \mid x , z ) = p ( y \mid z )$

Under these assumptions, we consider the following ground-truth process

$$
p ( y , z \mid x ) = p ( y \mid z ) p ( z \mid x ) .\tag{1}
$$

Note that, for uncertainty quantification, we consider a distribution already conditioned on a specific $x ,$ as we are typically interested in evaluation at a particular point. In what follows, we write $p _ { z } : = p ( y \mid z ) { \mathrm { ~ a n d ~ } } p : = p ( y \mid x ) = \mathbb { E } _ { Z \sim p ( z \mid x ) } p _ { Z } .$

## 3.1 THREE-COMPONENT DECOMPOSITION OF THE EXPECTED RISK

We follow the formalization of uncertainty as the expected loss (or risk) of a prediction (Kotelevskii et al., 2025). Namely, if we expect that, for whatever reason, the model has a large loss at a given input $x ,$ we say that its prediction at $x$ is uncertain. Below, we consider a specific class of loss functions, proper scoring rules (Gneiting & Raftery, 2007). For such losses, the expected loss is minimized by the true distribution of the outcome, which lets us separate the risk of the best possible prediction from the excess risk of the model.

Let $\ell ( q , y )$ denote a proper scoring rule (Gneiting & Raftery, 2007) that evaluates a predictor $q : = q ( y \mid x )$ at Y. We define the expected pointwise risk of q as

$$
R ( p , q ) : = \mathbb { E } _ { Y \sim p } [ \ell ( q , Y ) ] .
$$

Decomposing to Bayes risk and excess risk. We can decompose this expected risk into the Bayes risk $R _ { b } \bar { ( } p ) : = \bar { R } ( p , p )$ of the process p and the excess risk $R _ { \mathrm { e x c } } ( p \parallel q ) : = R ( p , q ) - R ( p , p )$ of the model q with respect to p. Following Kotelevskii et al. (2025), for a proper scoring rule instantiated with a convex function $\bar { G } \colon \mathcal { P } ( \mathcal { y } )  \bar { \mathbb { R } }$ , this gives

$$
R ( p , q ) = R _ { b } ( p ) + R _ { \mathrm { e x c } } ( p \parallel q ) = - G ( p ) + D _ { G } ( p \parallel q ) ,\tag{2}
$$

where $D _ { G } ( p \parallel q ) = G ( p ) - G ( q ) - \langle G ^ { \prime } ( q ) , p - q \rangle$ is the Bregman divergence instantiated by $G$ Here, $- G ( \boldsymbol { p } )$ is the Bayes risk. It depends only on the data generation process $p ,$ so changing the model does not affect it. This term corresponds to aleatoric uncertainty, which is irreducible. The excess risk $D _ { G } ( p \parallel q )$ depends on the model, its training data, and the training procedure. It can therefore be reduced by improving the model and corresponds to epistemic uncertainty, which is reducible. Reducibility here means that the predictor can be improved while the available input information is kept fixed.

Splitting the Bayes risk over intents. Interestingly, the Bayes risk $- G ( \boldsymbol { p } )$ can be split further by using the ground-truth process (1). Since the true intent $z ^ { * }$ is unobserved, we treat it as a random variable $Z \sim p ( z \mid x )$ . Then, using $p _ { z }$ and $p$ defined above, we can expand equation (2) as follows (see Appendix B for the derivation):

$$
R ( p , q ) = \underbrace { \overbrace { { \mathbb { E } _ { Z \sim p ( z \mid x ) } [ - G ( p _ { Z } ) ] } } ^ { - G ( p ) } } _ { \mathrm { r e s i d u a l ~ a l e a t o r i c } } + \underbrace { { \mathbb { E } _ { Z \sim p ( z \mid x ) } [ D _ { G } ( p _ { Z } \parallel p ) ] } } _ { \mathrm { s p e c i f i c a t i o n } } + \underbrace { D _ { G } ( p \parallel q ) } _ { \mathrm { e p i s t e m i c } } .\tag{3}
$$

Interpretation of the three components. As in prior work (Kotelevskii et al., 2025; 2022; Lahlou et al., 2023), the aleatoric uncertainty in equation (2) is defined relative to the input x. Sale et al. (2026) argue that the split into reducible and irreducible uncertainty depends, among other things, on the features available as input. In particular, uncertainty that is irreducible given x can still be reduced by providing more input information, such as the intent. Therefore, both $p ( y \mid x )$ and $p ( y \mid z )$ determine ground-truth aleatoric uncertainty, but relative to differentfixed information levels.

For the joint process $p ( y , z \mid x )$ that we assume here, $p : = p ( y \mid x )$ and $p _ { z } : = p ( y \mid z )$ describe the predictive distribution before and after the intent is revealed. Namely, the best prediction based only on x is $p ,$ and its Bayes risk is $- G ( p )$ . If the intent is revealed and turns out to be z, the best prediction is $p _ { z }$ , with Bayes risk $- G ( p _ { z } )$ . Thus, before the intent is revealed, the expected Bayes risk of a predictor that will observe it $\begin{array} { r } { \mathbf { \dot { i } s } ^ { \prime } - \mathbb { E } _ { Z \sim p ( z | x ) } [ G ( p _ { Z } ) ] } \end{array}$ ]. We call this term residual aleatoric uncertainty.

The difference between these two Bayes risks is the Bregman information term in equation (3):

$$
\underbrace { - G ( p ) } _ { \mathrm { b e f o r e \ c l a r i f i c a t i o n } } - \underbrace { \left( - \mathbb { E } _ { Z \sim p ( \cdot \vert x ) } [ G ( p _ { Z } ) ] \right) } _ { \mathrm { a f t e r \ c l a r i f i c a t i o n } } = \mathbb { E } _ { Z \sim p ( \cdot \vert x ) } \left[ D _ { G } ( p _ { Z } \| p ) \right] .
$$

It shows how much the Bayes risk is expected to decrease once the intent is revealed, i.e., how much we expect to gain from clarification. We call this term specification uncertainty. Like the Bayes risk, specification uncertainty depends only on the ground-truth process. If x fully specifies the intent (i.e., $p ( z \mid x )$ is degenerate), specification uncertainty vanishes, and equation (3) reduces to equation (2).

Each term in equation (3) has a clear practical meaning, as it corresponds to a specific action. In expectation, clarification reduces the Bayes risk through the specification term, and improving the predictor reduces the epistemic term. Neither action reduces the residual aleatoric term, so the only way to avoid this part of the risk is to abstain from prediction.

## 3.2 BAYESIAN APPROXIMATION OF THE COMPONENTS

In practice, the ground-truth process $p ( y , z \mid x )$ in equation (1) is unknown, and we need to approximate it. For this, we consider a model $q ( y \mid x , \theta )$ , parametrized by $\theta \in \Theta$ . To keep generality and show the connection with known decompositions in the literature, we assume this model is Bayesian:

$$
q ( y , z , \theta \mid x , D ) = q ( y \mid z , \theta ) \pi ( \theta , z \mid x , D ) ,
$$

where $D = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ is the training dataset. The model has two latent variables, the parameters θ and the intent z, and the joint distribution $\pi ( \theta , z \mid x , D )$ is their posterior.

The components of equation (3) depend on the unknown distributions $p ( z \mid x )$ and $p _ { z }$ . To obtain computable and practical uncertainty measures, we replace them with approximations. We describe how we obtain them at the end of this section.

These approximations can be incorporated in several ways (Kotelevskii et al., 2025), and we discuss them in Appendix A. Here in the main part, we use the one that recovers the standard informationtheoretic decomposition, as we show below. It treats each posterior sample of the parameters as a candidate for the ground-truth process and uses the posterior predictive distribution as the predictor. Let $q _ { z , \theta } : = q ( y \mid z , \theta )$ , and define

$$
q : = \mathbb { E } _ { \pi } [ q _ { Z , \Theta } ] = q ( y \mid x , D ) , \qquad q _ { \theta } : = \mathbb { E } _ { \pi } [ q _ { Z , \Theta } \mid \Theta = \theta ] ,
$$

where the expectations are computed under $\pi ( \theta , z \mid x , D )$ The distribution $q$ is the posterior predictive distribution of this model and specifies the predictor introduced above. Averaging the risk over the candidates $q _ { \theta }$ , we estimate $R ( p , q )$ by $\mathbb { E } _ { \Theta } [ \bar { R } ( q _ { \Theta } , q ) ] = - G ( q )$ . Decomposing the Bayes risk of each $q _ { \theta }$ , we obtain estimates of the three components of equation (3):

$$
\begin{array} { r } { - G ( q ) = \underbrace { - \mathbb { E } _ { Z , \Theta } [ G ( q _ { Z , \Theta } ) ] } _ { \mathrm { r e s i d u a l ~ a l e a t o r i c } } + \underbrace { \mathbb { E } _ { Z , \Theta } [ D _ { G } ( q _ { Z , \Theta } \parallel q _ { \Theta } ) ] } _ { \mathrm { s p e c i f i c a t i o n } } + \underbrace { \mathbb { E } _ { \Theta } [ D _ { G } ( q _ { \Theta } \parallel q ) ] } _ { \mathrm { e p i s t e m i c } } . } \end{array}\tag{4}
$$

For $\begin{array} { r } { G ( q ) = \sum _ { i } q _ { i } } \end{array}$ log q<sub>i</sub>, equation (4) becomes

$$
\begin{array} { r l } & { \mathbb { H } ( q ) = \underbrace { \overbrace { \mathbb { E } _ { Z , \Theta } [ \mathbb { H } ( q _ { Z } , \Theta ) ] } ^ { \mathrm { a l e a t o r i c ~ u n c e r t a i n t y ~ e s i m a t e } } } _ { \mathrm { S E A } _ { \mathrm { r e s } } } + \underbrace { \mathbb { E } _ { Z , \Theta } [ K L ( q _ { Z , \Theta } \parallel q _ { \Theta } ) ] } _ { \mathrm { S E A } _ { \mathrm { s p e c } } } + \underbrace { \mathbb { E } _ { \Theta } [ K L ( q _ { \Theta } \parallel q ) ] } _ { \mathrm { S E A } _ { \mathrm { e p i } } } } \\ & { \quad = \mathbb { E } _ { \Theta } [ \mathbb { H } ( q _ { \Theta } ) ] + I ( Y ; \Theta \mid X = x , D ) , } \end{array}\tag{5}
$$

where the second equality holds because q<sub>θ</sub> averages $q _ { z , \theta }$ over intents, so the first two terms sum to $\mathbb { E } _ { \Theta } [ \mathbb { H } ( q _ { \Theta } ) ]$ . The second line is therefore the standard information-theoretic predictive uncertainty decomposition, applied to the model with the intent marginalized out. Its epistemic term is the conditional mutual information, which is used as the BALD acquisition score (Houlsby et al., 2011). Below, we denote the three terms by $\mathrm { S E A _ { r e s } , S E A _ { s p e c } }$ , and $\mathrm { S E A _ { \mathrm { e p i } } }$ , and their sum $\mathbb { H } ( q )$ by $\mathrm { S E A } _ { \mathrm { t o t a l } }$ The standard decomposition merges $\mathrm { S E A _ { r e s } }$ and $\mathrm { S E A _ { s p e c } }$ into one aleatoric term, so it cannot show whether clarifying x would reduce the risk. We discuss limitations of our approach in Appendix C.

Table 1: Text QA underspecification-detection AUROC using semantic entropy. CoQAR-U contains 1,000 matched damaged/undamaged pairs, with damaged inputs positive. AmbigQA contains 1,971 unique questions: 1,141 ambiguous positives and 830 unambiguous negatives.
<table><tr><td></td><td colspan="2">Gemma E2B</td><td colspan="2">Gemma E4B</td><td colspan="2">Heterogeneous</td></tr><tr><td>Method</td><td>CoQAR-U</td><td>AmbigQA</td><td>CoQAR-U</td><td>AmbigQA</td><td>CoQAR-U</td><td>AmbigQA</td></tr><tr><td> $\mathrm { S E A _ { r e s } }$ </td><td>0.5373</td><td>0.5102</td><td>0.5760</td><td>0.5071</td><td>0.5404</td><td>0.5350</td></tr><tr><td> $\mathrm { S E A _ { \mathrm { e p i } } }$ </td><td>0.5213</td><td>0.5442</td><td>0.5303</td><td>0.5461</td><td>0.5232</td><td>0.4396</td></tr><tr><td> $\mathrm { S E A _ { \mathrm { s p e c } } }$ </td><td>0.7807</td><td>0.6575</td><td>0.7709</td><td>0.6678</td><td>0.7873</td><td>0.6017</td></tr><tr><td> $\mathrm { S E A } _ { \mathrm { t o t a l } }$ </td><td>0.7365</td><td>0.5169</td><td>0.7614</td><td>0.5164</td><td>0.6593</td><td>0.5250</td></tr><tr><td colspan="5">BALD (without clarifications)</td><td></td><td></td></tr><tr><td> $\mathrm { B A L D _ { a l e a } }$ </td><td>0.7356</td><td>0.5011</td><td>0.7607</td><td>0.4858</td><td>0.7706</td><td>0.5073</td></tr><tr><td> $\mathrm { B A L D _ { e p i } }$ </td><td>0.6002</td><td>0.5442</td><td>0.6081</td><td>0.5348</td><td>0.7109</td><td>0.4535</td></tr><tr><td> $\mathrm { B A L D } _ { \mathrm { t o t a l } }$ </td><td>0.7340</td><td>0.5015</td><td>0.7608</td><td>0.4865</td><td>0.7804</td><td>0.5003</td></tr><tr><td colspan="5">BALD (clarification posterior)</td><td></td><td></td></tr><tr><td> $\mathrm { B A L D _ { a l e a } }$ </td><td>0.7413</td><td>0.5169</td><td>0.7627</td><td>0.5162</td><td>0.6721</td><td>0.5383</td></tr><tr><td> $\mathrm { B A L D _ { e p i } }$ </td><td>0.5213</td><td>0.5442</td><td>0.5303</td><td>0.5461</td><td>0.5232</td><td>0.4396</td></tr><tr><td> $\mathrm { B A L D } _ { \mathrm { t o t a l } }$ </td><td>0.7365</td><td>0.5169</td><td>0.7614</td><td>0.5164</td><td>0.6593</td><td>0.5250</td></tr></table>

Connection to clarification-based decompositions. Our work is closely related to (Hou et al., 2024) and (Walha et al., 2026), which both consider a single fixed model and generate clarifications of its input. If we set θ = const and write $q _ { z } : = q _ { z , \theta } ,$ , then equation (4) simplifies to

$$
- G ( q ) = \underbrace { - \mathbb { E } _ { Z } [ G ( q _ { Z } ) ] } _ { ( i i ) } + \underbrace { \mathbb { E } _ { Z } [ D _ { G } ( q _ { Z } \parallel q ) ] } _ { ( i ) } .\tag{6}
$$

A specific choice of G gives equation 9 of (Walha et al., 2026). For $\begin{array} { r } { G ( q ) = \sum _ { i } q _ { i } } \end{array}$ log q<sub>i</sub>, we get $\mathbb { H } ( \acute { q } ) = \mathbb { E } _ { Z } [ K L ( q _ { Z } \parallel q ) ] + \mathbb { E } _ { Z } [ \mathbb { H } ( q _ { Z } ) ]$ , which is exactly equation 3 of (Hou et al., 2024).

Both works interpret term (i) as aleatoric uncertainty and term (ii) as epistemic uncertainty. To see what these terms measure, note that equation (6) holds for any predictor, including the ideal one with $\pi ( z \mid x ) = p ( z \mid x )$ and $q _ { z } = p _ { z }$ for all z. Then $q = p$ , the excess risk $D _ { G } ( p \bar { \parallel } q )$ vanishes, and equation (6) becomes

$$
- G ( p ) = - \mathbb { E } _ { Z } [ G ( p _ { Z } ) ] + \mathbb { E } _ { Z } [ D _ { G } ( p _ { Z } \parallel p ) ] ,
$$

which is exactly the split of the Bayes risk in equation (3). The crucial consequence is that term (ii) is nonzero whenever the outcome is random given the intent, even ifwe have perfect knowledge of $p ( y \mid z )$ and $p ( z \mid x )$ . Hence, this term becomes irreducible, and treating it as epistemic uncertainty would suggest improving a predictor that is already perfect, while the appropriate action is to abstain. Therefore, the interpretation of Hou et al. (2024) and Walha et al. (2026) is justified only when the outcome is a deterministic function of the intent. In this case, for the ideal predictor, the residual aleatoric term (ii) vanishes, and term (i) recovers the whole aleatoric uncertainty.

One may argue that for a single fixed model, the Bayesian estimate of the excess risk is zero, so equation (6) already accounts for all uncertainty that such a model can express, and term (ii) is the only term that can represent its epistemic uncertainty. However, as shown above, term (ii) estimates the residual aleatoric uncertainty. The zero estimate of the excess risk only means that a point-mass posterior cannot express epistemic uncertainty, not that the model has none. Such uncertainty can instead be estimated by non-Bayesian methods (Kotelevskii et al., 2022; Lahlou et al., 2023).

Estimation in practice. The pairs $( x , y )$ alone determine $p ( y \mid x )$ , but not how its Bayes risk $- G ( \boldsymbol { p } )$ splits into specification and residual aleatoric uncertainty. For example, the same $p ( y \mid x )$ can arise when x fully specifies the intent and specification uncertainty is zero. It can also arise when $y$ is a deterministic function of $z ,$ and then all the randomness of $y$ given x comes from the unknown intent. Therefore, the two terms can be estimated separately only with additional information about the intent or with assumptions on the latent process. In our experiments, this information comes from large pretrained generative models. We use them to generate possible intents, which serve as hypotheses about $z ^ { * }$ . Since this is done independently of θ, we have $\pi ( \theta , z \mid x , D ) = \pi ( \theta \mid D ) { \ ' } { \hat { \pi } } ( z \mid x )$

Table 2: Text QA error-detection AUROC using semantic entropy and GPT-5.6 Luna correctness labels. Errors in the ensemble answer to the original input are positive for every method. Evaluation uses 1,000 undamaged CoQAR-U questions and 830 unambiguous AmbigQA questions.
<table><tr><td></td><td colspan="2">Gemma E2B</td><td colspan="2">Gemma E4B</td><td colspan="2">Heterogeneous</td></tr><tr><td>Method</td><td>CoQAR-U</td><td>AmbigQA</td><td>CoQAR-U</td><td>AmbigQA</td><td>CoQAR-U</td><td>AmbigQA</td></tr><tr><td> $\mathrm { S E A _ { r e s } }$ </td><td>0.7285</td><td>0.8103</td><td>0.7535</td><td>0.7739</td><td>0.7077</td><td>0.7197</td></tr><tr><td> $\mathrm { S E A _ { \mathrm { e p i } } }$ </td><td>0.6573</td><td>0.6827</td><td>0.6145</td><td>0.6797</td><td>0.6730</td><td>0.5431</td></tr><tr><td> $\mathrm { S E A _ { \mathrm { s p e c } } }$ </td><td>0.5627</td><td>0.5279</td><td>0.6600</td><td>0.5544</td><td>0.5456</td><td>0.7234</td></tr><tr><td> $\mathrm { S E A } _ { \mathrm { t o t a l } }$ </td><td>0.7561</td><td>0.8062</td><td>0.7974</td><td>0.7710</td><td>0.7323</td><td>0.7412</td></tr><tr><td colspan="5">BALD (without clarifications)</td><td></td><td></td></tr><tr><td> $\mathrm { B A L D _ { a l e a } }$ </td><td>0.7665</td><td>0.8439</td><td>0.7800</td><td>0.8364</td><td>0.7438</td><td>0.7513</td></tr><tr><td> $\mathrm { B A L D _ { e p i } }$ </td><td>0.6628</td><td>0.6334</td><td>0.5326</td><td>0.6967</td><td>0.7214</td><td>0.5775</td></tr><tr><td> $\mathrm { B A L D } _ { \mathrm { t o t a l } }$ </td><td>0.7689</td><td>0.8435</td><td>0.7813</td><td>0.8364</td><td>0.7515</td><td>0.7692</td></tr><tr><td colspan="5">BALD (clarification posterior)</td><td></td><td></td></tr><tr><td> $\mathrm { B A L D _ { a l e a } }$ </td><td>0.7536</td><td>0.8062</td><td>0.7957</td><td>0.7709</td><td>0.7308</td><td>0.7267</td></tr><tr><td> $\mathrm { B A L D _ { e p i } }$ </td><td>0.6573</td><td>0.6827</td><td>0.6145</td><td>0.6797</td><td>0.6730</td><td>0.5431</td></tr><tr><td> $\mathrm { B A L D } _ { \mathrm { t o t a l } }$ </td><td>0.7561</td><td>0.8062</td><td>0.7974</td><td>0.7710</td><td>0.7323</td><td>0.7412</td></tr></table>

Since these intents are only hypotheses, the quality of the estimates depends on how well $\pi ( \boldsymbol { z } \mid \boldsymbol { x } )$ approximates the true $p ( z \mid x )$ . Similarly, the ensembles that we use as samples of θ only approximate the posterior $\pi ( \theta \mid D )$ , especially when they consist of different pretrained models.

## 4 EXPERIMENTS

In this section, we verify our intuition about the components presented in equation (5) and provide practical estimates across different domains, modalities, and tasks.

Specifically, we show the sensitivity of specification and residual aleatoric terms on several input modalities of LLMs - text (Section 4.1), images (Section 4.2), and audio (Section 4.3). Additionally, we evaluate the specificity of the SEA components in image classification, including out-of-distribution (OOD) detection for the epistemic term, in Appendix K.

Models. All LLM and VLM experiments used a heterogeneous ensemble of 4 models - Qwen2.5-Omni-7B (Xu et al., 2025), Gemma-4-E4B-it (Gemma Team, 2026), Phi-4-multimodalinstruct (Abouelenin et al., 2025), and MiniCPM-o-4.5 (Cui et al., 2026). For text-to-text question answering (QA), we used a checkpoint ensemble, fine-tuned from Gemma-E2B-it and Gemma-E4B-it, on the AmbigQA (Min et al., 2020) dataset. Full details on the fine-tuning procedure are presented in Appendix G.

## 4.1 TEXTQA

Data. For text-to-text QA, we evaluate the specificity of SEA terms on two datasets:

1. CoQAR-U. In the CoQAR dataset (Brabant et al., 2022), the system is given a text passage and a question. To model underspecified inputs, we remove the span with key information from the passage. Original passages serve as well-specified input. We use GPT-5.6 Luna as π(z|x) and sample five clarifications for each original and damaged passage (see details in Appendix J).

2. AmbigQA. We also use the existing AmbigQA (Min et al., 2020) dataset, which is based on an open-domain version of Natural Questions (Kwiatkowski et al., 2019). Here, the system is given only a question, without any supporting context, and should answer based on its parametric memory. We treat questions with a single interpretation as well-specified, and questions with 2 or more interpretations as underspecified. Similar to CoQAR, we sample 5 clarifications per question using GPT-5.6 Luna.

Table 3: VQA AUROC of the SEA components on VQS and ImageNet, computed with the semantic entropy estimator for the heterogeneous ensemble. Underspecification detection takes 30% masking as the positive class and 10% as the negative class. Error detection uses an incorrect ensemble answer to the masked image as the positive class, reported separately at each mask size.
<table><tr><td rowspan="3">Method</td><td colspan="2">Underspecification</td><td colspan="4">ID error</td></tr><tr><td>VQS</td><td>ImageNet</td><td colspan="2">VQS</td><td colspan="2">ImageNet</td></tr><tr><td>30% vs 10%</td><td>30% vs 10%</td><td>10%</td><td>30%</td><td>10%</td><td>30%</td></tr><tr><td> $\mathrm { S E A } _ { \mathrm { r e s i d u a l } }$ </td><td>0.5422</td><td>0.5289</td><td>0.8217</td><td>0.7850</td><td>0.7448</td><td>0.7715</td></tr><tr><td> $\mathrm { S E A _ { \mathrm { e p i } } }$ </td><td>0.5073</td><td>0.4907</td><td>0.7830</td><td>0.7521</td><td>0.7275</td><td>0.7334</td></tr><tr><td> $\mathrm { S E A _ { \mathrm { s p e c } } }$ </td><td>0.6614</td><td>0.6903</td><td>0.7256</td><td>0.7006</td><td>0.5992</td><td>0.7172</td></tr><tr><td> $\mathrm { S E A } _ { \mathrm { t o t a l } }$ </td><td>0.5486</td><td>0.5452</td><td>0.8289</td><td>0.7900</td><td>0.7458</td><td>0.7756</td></tr><tr><td colspan="7">BALD (without clarifications)</td></tr><tr><td> $\mathrm { B A L D _ { a l e a } }$ </td><td>0.5363</td><td>0.5768</td><td>0.8194</td><td>0.8087</td><td>0.7151</td><td>0.7653</td></tr><tr><td> $\mathrm { B A L D _ { e p i } }$ </td><td>0.5223</td><td>0.5704</td><td>0.7968</td><td>0.7440</td><td>0.7193</td><td>0.7264</td></tr><tr><td> $\mathrm { B A L D } _ { \mathrm { t o t a l } }$ </td><td>0.5332</td><td>0.5806</td><td>0.8311</td><td>0.8019</td><td>0.7263</td><td>0.7742</td></tr><tr><td colspan="7">BALD (clarification posterior)</td></tr><tr><td> $\mathrm { B A L D _ { a l e a } }$ </td><td>0.5618</td><td>0.5578</td><td>0.8268</td><td>0.7871</td><td>0.7393</td><td>0.7753</td></tr><tr><td> $\mathrm { B A L D _ { e p i } }$ </td><td>0.5073</td><td>0.4907</td><td>0.7830</td><td>0.7521</td><td>0.7275</td><td>0.7334</td></tr><tr><td> $\mathrm { B A L D } _ { \mathrm { t o t a l } }$ </td><td>0.5486</td><td>0.5452</td><td>0.8289</td><td>0.7900</td><td>0.7458</td><td>0.7756</td></tr></table>

Findings. First, Table 1 shows that the specification component indeed performs well at detecting underspecified questions, aligning with its intended interpretation. Second, we check whether the residual aleatoric component in SEA decomposition still predicts in-domain error. Table 2 shows that the residual term delivers a higher ROC-AUC compared to other components. We note that the magnitude of the residual term is often larger than that of the epistemic and specification terms; therefore, total uncertainty is mostly driven by the residual component, explaining the high performance of total uncertainty on this task.

## 4.2 VISUAL QA

Data. We evaluate specification uncertainty in VQA by progressively hiding image content while keeping the question and reference answer fixed. For each masked observation, five inpainted images provide alternative visual clarifications (see details in Appendix M). We used two VQA datasets:

1. VQS. The frozen evaluation subset contains 500 questions on 500 distinct images sampled from the validation split of the VQS (Gan et al., 2017) dataset. Input underspecification is modeled by applying random gray masks covering 10% and 30% of the image area. Input clarifications were generated with FLUX.1 Fill-dev (Black Forest Labs, 2024). Answer correctness is evaluated via Exact Match with the most common normalized human answer.

2. Imagenet. The VQA dataset is built from ImageNet (Deng et al., 2009) samples used in classification experiments by asking a general question, "What is depicted in the picture?" for every image. Masking and clarification construction follow the classification data construction pipeline (Appendix K.3). GPT-5.6 Luna evaluates the correctness of the answer by comparing the model’s answer with the ImageNet class name.

Findings. Each SEA component is the strongest signal for its intended target. Specification uncertainty is the best underspecification detector on both datasets, while other components do not provide a useful signal for requesting a clarification (see Table 3). Conversely, error detection is dominated by total and aleatoric uncertainty; since SEA preserves the total of the standard decomposition, isolating the specification term do not harm the error detection performance. The Appendix M shows that both results hold for all six entropy estimators and all confidence baselines.

Mask size effects. The ImageNet VQA setup allows control of the degree of underspecification in the input by varying the mask size relative to the object bounding box area. Figure 2 compares how

![](images/82135afd3674125e71324a7b535282ffe239bd483bc5fbaf607d75ee4d746cb6.jpg)

![](images/df13528adc82b0149de8be4021e1571a0bd5930502b654b54952213d8826b0a2.jpg)  
Figure 2: Uncertainty decomposition of the heterogeneous ensemble on ImageNet with the semantic entropy estimator. Left: mean uncertainty per mask size, stacked into epistemic, aleatoric, and specification components, for SEA, BALD on the masked image, and BALD with the posterior over clarifications. Red labels give the specification share of total uncertainty; error bars are 95% intervals of the specification mean. Right: change of each component’s share relative to the 10% mask, in percentage points. Only the specification share grows with the masked area; SEA and BALD over clarifications have identical total and epistemic terms by construction.

Table 4: Audio QA underspecification-detection AUROC on CoQAR-U-TTS.
<table><tr><td></td><td>SEA</td><td>BALD (without clarification)</td><td>BALD (clarification posterior)</td></tr><tr><td>Total</td><td>0.5926</td><td>0.7427</td><td>0.5926</td></tr><tr><td>Aleatoric</td><td>0.5170</td><td>0.7229</td><td>0.5851</td></tr><tr><td>Epistemic</td><td>0.4937</td><td>0.6108</td><td>0.4937</td></tr><tr><td>Specification</td><td>0.7739</td><td></td><td></td></tr></table>

SEA and the two BALD variants split the ensemble’s mean uncertainty on ImageNet as the masked fraction of the object bounding box grows.

Specification uncertainty grows monotonically with the amount of hidden content, both in absolute value and as a share of total uncertainty. It is the only component whose share systematically increases. SEA and BALD over clarifications share the same total and epistemic terms by construction, since SEA only splits the pooled aleatoric term into residual aleatoric and specification uncertainty. Under SEA, the aleatoric and epistemic shares decline with mask size, so the added uncertainty is absorbed almost entirely by the specification term, in agreement with its intended semantics.

Without the specification term, the missing information signal is hidden. BALD over clarifications folds the growth into its aleatoric term. BALD on the masked image inflates most with mask size, with both aleatoric and epistemic terms growing in absolute value while their shares stay nearly flat. Hence, its decomposition does not indicate that the extra uncertainty stems from missing input and could be removed by clarification rather than by changing the model.

## 4.3 AUDIO QA

Data. We render CoQAR-U passages as speech with OpenAI TTS, retaining the question as text.

Findings Table 4 shows that specification uncertainty is the strongest omission-detection signal, outperforming the runner-up total BALD uncertainty estimated on non-clarified input. Thus, separating specification uncertainty recovers an underspecification-sensitive signal that is obscured in the total of the clarification-posterior.

## 5 CONCLUSION

In this paper, we considered a predictive uncertainty setting in which the observed input x may only partially describe the intent. We derived SEA, a decomposition of prediction risk into specification, residual aleatoric, and epistemic uncertainty. At the original information level, the first two terms split the Bayes risk into the expected gain from clarification and the remaining risk, and each term corresponds to an action. In QA over text, images, and audio, specification uncertainty is the most informative score for detecting underspecified inputs, even where uncertainty computed only on x, without clarifications, fails to detect them. Residual aleatoric uncertainty is the most informative component for ID error detection in all but one setting. The standard decomposition, even when sensitive to underspecification, cannot show whether clarification would reduce the risk.

## AI USE STATEMENT

In this work, we used generative AI tools for generating clarifications for SEA estimation in all datasets, implementing the method and experiments, reformatting AmbigQA dataset, preparing visualization code, polishing text in both main and appendix sections and drafting some appendix sections. We have not used generative AI tools for generating research hypotheses, theory derivation, proofs, experiment design or quality metric choice. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

For theoretical results, complete proof of the claims is presented in Section 3 and appendices A, B.

For experimental results, we elaborate the details of practical estimation of SEA decomposition in appendix D, details of LLM ensemble training in appendix G, and describe the process of clarification generation in appendix J.

We will also release all custom datasets (e.g. CoQAR-U, CoQAR-U-TTS, AmbigQA along with sampled clarifications) and experimental pipelines upon acceptance.

## REFERENCES

Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, et al. Phi-4-Mini technical report: Compact yet powerful multimodal language models via Mixture-of-LoRAs. arXiv preprint arXiv:2503.01743, 2025. doi: 10.48550/arXiv.2503.01743. URL https://arxiv.org/abs/ 2503.01743.

Alexander Bick, Adam Blandin, and David J. Deming. The rapid adoption of generative AI. Management Science, 2026. doi: 10.1287/mnsc.2025.02523. URL https://doi.org/10.1287/ mnsc.2025.02523. Articles in Advance, published online January 20, 2026.

Black Forest Labs. FLUX.1 Fill [dev]. https://huggingface.co/black-forest-labs/ FLUX.1-Fill-dev, 2024. Model card.

Black Forest Labs. FLUX.2 [klein]: Towards interactive visual intelligence. Official model release, January 2026. URL https://bfl.ai/blog/ flux2-klein-towards-interactive-visual-intelligence. Released January 15, 2026; accessed September 26, 2026.

Quentin Brabant, Gwénolé Lecorvé, and Lina Maria Rojas-Barahona. Coqar: Question rewriting on coqa. In Nicoletta Calzolari, Frédéric Béchet, Philippe Blache, Khalid Choukri, Christopher Cieri, Thierry Declerck, Sara Goggi, Hitoshi Isahara, Bente Maegaard, Joseph Mariani, Hélène Mazo, Jan Odijk, and Stelios Piperidis (eds.), Proceedings of the Thirteenth Language Resources and Evaluation Conference, LREC 2022, Marseille, France, 20-25 June 2022, pp. 119–126. European Language Resources Association, 2022. URL https://aclanthology.org/ 2022.lrec-1.13.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 1877–1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/paper/ 2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf.

Junbo Cui, Bokai Xu, Chongyi Wang, Tianyu Yu, Weiyue Sun, Yingjing Xu, Tianran Wang, Zhihui He, Wenshuo Ma, Tianchi Cai, et al. MiniCPM-o 4.5: Towards real-time full-duplex omni-modal interaction. arXiv preprint arXiv:2604.27393, 2026. doi: 10.48550/arXiv.2604.27393. URL https://arxiv.org/abs/2604.27393.

Antoine de Mathelin, Francois Deheeger, Mathilde Mougeot, and Nicolas Vayatis. Deep antiregularized ensembles provide reliable out-of-distribution uncertainty quantification. arXiv preprint arXiv:2304.04042, 2023. doi: 10.48550/arXiv.2304.04042. URL https://arxiv.org/abs/ 2304.04042.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Fei-Fei Li. ImageNet: A largescale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp. 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.

Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630, June 2024. doi: 10.1038/ s41586-024-07421-0. URL https://doi.org/10.1038/s41586-024-07421-0.

Yarin Gal and Zoubin Ghahramani. Dropout as a bayesian approximation: Representing model uncertainty in deep learning. In Maria Florina Balcan and Kilian Q. Weinberger (eds.), Proceedings ofThe 33rd International Conference on Machine Learning, volume 48 of Proceedings ofMachine Learning Research, pp. 1050–1059, New York, New York, USA, 20–22 Jun 2016. PMLR. URL https://proceedings.mlr.press/v48/gal16.html.

Chuang Gan, Yandong Li, Haoxiang Li, Chen Sun, and Boqing Gong. Vqs: Linking segmentations to questions and answers for supervised attention in vqa and question-focused semantic segmentation. In 2017 IEEE International Conference on Computer Vision (ICCV), pp. 1829–1838. IEEE, 2017.

Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper\_files/paper/ 2017/file/4a8423d5e91fda00bb7e46540e2b0cf1-Paper.pdf.

Gemma Team. Gemma 4 Technical Report. arXiv preprint arXiv:2607.02770, 2026. doi: 10.48550/ arXiv.2607.02770. URL https://arxiv.org/abs/2607.02770.

Tilmann Gneiting and Adrian E Raftery. Strictly proper scoring rules, prediction, and estimation. Journal ofthe American statistical Association, 102(477):359–378, 2007.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016. doi: 10.1109/CVPR.2016.90.

Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, Dawn Song, Jacob Steinhardt, and Justin Gilmer. The many faces of robustness: A critical analysis of out-of-distribution generalization. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pp. 8340–8349, 2021. doi: 10.1109/ICCV48922.2021.00823. URL https://openaccess.thecvf.com/ content/ICCV2021/html/Hendrycks\_The\_Many\_Faces\_of\_Robustness\_A\_ Critical\_Analysis\_of\_Out-of-Distribution\_ICCV\_2021\_paper.html.

Paul Hofman, Yusuf Sale, and Eyke Hüllermeier. Quantifying aleatoric and epistemic uncertainty: A credal approach. In ICML 2024 Workshop on Structured Probabilistic Inference {\&} Generative Modeling, 2024.

Bairu Hou, Yujian Liu, Kaizhi Qian, Jacob Andreas, Shiyu Chang, and Yang Zhang. Decomposing uncertainty for large language models through input clarification ensembling. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 19023–19042. PMLR, 2024. URL https://proceedings.mlr. press/v235/hou24b.html.

Neil Houlsby, Ferenc Huszár, Zoubin Ghahramani, and Máté Lengyel. Bayesian active learning for classification and preference learning. arXiv preprint arXiv:1112.5745, 2011. doi: 10.48550/arXiv. 1112.5745. URL https://arxiv.org/abs/1112.5745.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum? id=nZeVKeeFYf9.

Eyke Hüllermeier and Willem Waegeman. Aleatoric and epistemic uncertainty in machine learning: An introduction to concepts and methods. Machine learning, 110(3):457–506, 2021.

Alex Kendall and Yarin Gal. What uncertainties do we need in bayesian deep learning for computer vision? Advances in neural information processing systems, 30, 2017.

Nikita Kotelevskii, Aleksandr Artemenkov, Kirill Fedyanin, Fedor Noskov, Alexander Fishkov, Artem Shelmanov, Artem Vazhentsev, Aleksandr Petiushko, and Maxim Panov. Nonparametric uncertainty quantification for single deterministic neural network. Advances in Neural Information Processing Systems, 35:36308–36323, 2022.

Nikita Kotelevskii, Vladimir Kondratyev, Martin Takác, Éric Moulines, and Maxim Panov.ˇ From risk to uncertainty: Generating predictive uncertainty measures via Bayesian estimation. In International Conference on Learning Representations, volume 2025, pp. 92468–92504, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ e6f32e64b9c27d153b46c94f0fe22b56-Abstract-Conference.html.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur P. Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: a benchmark for question answering research. Trans. Assoc. Comput. Linguistics, 7:452–466, 2019. doi: 10.1162/TACL\_A\_00276. URL https://doi.org/10. 1162/tacl\_a\_00276.

Salem Lahlou, Moksh Jain, Hadi Nekoei, Victor Ion Butoi, Paul Bertin, Jarrid Rector-Brooks, Maksym Korablyov, and Yoshua Bengio. DEUP: Direct epistemic uncertainty prediction. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview. net/forum?id=eGLdVRvvfQ.

Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. Simple and scalable predictive uncertainty estimation using deep ensembles. In Advances in Neural Information Processing Systems, volume 30, pp. 6402–6413. Curran Associates, Inc., 2017. URL https://papers.neurips.cc/paper\_files/paper/2017/hash/ 9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html.

Zhen Lin, Shubhendu Trivedi, and Jimeng Sun. Generating with confidence: Uncertainty quantification for black-box large language models. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=DWkJCSxKU5.

Jeremiah Liu, Zi Lin, Shreyas Padhy, Dustin Tran, Tania Bedrax Weiss, and Balaji Lakshminarayanan. Simple and principled uncertainty estimation with deterministic deep learning via distance awareness. Advances in neural information processing systems, 33:7498–7512, 2020.

Huan Ma, Jingdong Chen, Joey Tianyi Zhou, Guangyu Wang, and Changqing Zhang. Estimating LLM uncertainty with evidence. arXiv preprint arXiv:2502.00290, 2025. doi: 10.48550/arXiv. 2502.00290. URL https://arxiv.org/abs/2502.00290.

Andrey Malinin and Mark Gales. Uncertainty estimation in autoregressive structured prediction. In International Conference on Learning Representations, 2021. URL https://openreview. net/forum?id=jN5y-zb5Q7m.

Sewon Min, Julian Michael, Hannaneh Hajishirzi, and Luke Zettlemoyer. AmbigQA: Answering ambiguous open-domain questions. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 5783–5797, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.466. URL https://aclanthology.org/2020.emnlp-main.466/.

Hussein Mozannar and David Sontag. Consistent estimators for learning to defer to an expert. In Hal Daumé III and Aarti Singh (eds.), Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 7076–7087. PMLR, 13– 18 Jul 2020. URL https://proceedings.mlr.press/v119/mozannar20b.html.

Jishnu Mukhoti, Joost Van Amersfoort, Philip HS Torr, and Yarin Gal. Deep deterministic uncertainty for semantic segmentation. arXiv preprint arXiv:2111.00079, 2021.

Kata Naszadi, Putra Manggala, and Christof Monz. Aligning predictive uncertainty with clarification questions in grounded dialog. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pp. 14988–14998, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.999. URL https://aclanthology.org/2023.findings-emnlp.999/.

OpenAI. GPT-5.6 Luna model. Official API documentation, 2026. URL https://developers. openai.com/api/docs/models/gpt-5.6-luna. Accessed September 26, 2026.

Zeju Qiu, Weiyang Liu, Adrian Weller, and Bernhard Schölkopf. Orthogonal finetuning made scalable. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 31946–31963, Suzhou, China, November 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.emnlp-main.1627. URL https://aclanthology.org/ 2025.emnlp-main.1627/.

Yusuf Sale, Michele Caprio, and Eyke Hüllermeier. Is the volume of a credal set a good measure for epistemic uncertainty? In Uncertainty in Artificial Intelligence, pp. 1795–1804. PMLR, 2023.

Yusuf Sale, Viktor Bengs, Michele Caprio, and Eyke Hüllermeier. Second-order uncertainty quantification: A distance-based approach. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 43060–43076. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/sale24a.html.

Yusuf Sale, Nikita Kotelevskii, Maxim Panov, and Eyke Hüllermeier. The Aleatoric-Epistemic Dichotomy of Uncertainty is Meaningful and Indispensable for Machine Learning, May 2026. URL https://ssrn.com/abstract=6783338. SSRN preprint.

Kajetan Schweighofer, Lukas Aichberger, Mykyta Ielanskyi, and Sepp Hochreiter. On informationtheoretic measures of predictive uncertainty. UAI, 2025.

Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher Manning. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 5433–5442, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main. 330. URL https://aclanthology.org/2023.emnlp-main.330/.

Joost Van Amersfoort, Lewis Smith, Yee Whye Teh, and Yarin Gal. Uncertainty estimation using a single deep deterministic neural network. In International conference on machine learning, pp. 9690–9700. PMLR, 2020.

Roman Vashurin, Maiya Goloburda, Albina Ilina, Aleksandr Rubashevskii, Preslav Nakov, Artem Shelmanov, and Maxim Panov. CoCoA: A minimum Bayes risk framework bridging confidence and consistency for uncertainty quantification in LLMs. In Advances in Neural Information Processing Systems, volume 38, pp. 106236–106281. Curran Associates, Inc., 2025. doi: 10.52202/ 085713-3546. URL https://papers.nips.cc/paper\_files/paper/2025/hash/ 98d2006bf15f00e2ff293dfba4acb8b2-Abstract-Conference.html.

Nassim Walha, Sebastian G. Gruber, Thomas Decker, Yinchong Yang, Alireza Javanmardi, Eyke Hüllermeier, and Florian Buettner. Fine-grained uncertainty decomposition in large language models: A spectral approach. Proceedings of the AAAI Conference on Artificial Intelligence, 40 (31):26090–26098, 2026. doi: 10.1609/aaai.v40i31.39811. URL https://ojs.aaai.org/ index.php/AAAI/article/view/39811.

Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin. Qwen2.5-Omni technical report. arXiv preprint arXiv:2503.20215, 2025. doi: 10.48550/arXiv.2503.20215. URL https://arxiv.org/abs/2503.20215.

## A OTHER APPROXIMATION STRATEGIES

In Section 3, we estimate the terms of equation (3) with approximation (1, 2) of Kotelevskii et al. (2025). Here, we describe the approximation ideas they consider, apply them to equation (3), and give the resulting estimates for several proper scoring rules.

Notation. We use $q _ { z , \theta } , q _ { \theta }$ , and $q$ from Section 3 and additionally define the posterior predictive distribution given the intent,

$$
\bar { q } _ { z } : = \mathbb { E } _ { \pi } [ q _ { Z , \Theta } \mid Z = z ] = q ( y \mid z , x , D ) .
$$

Expectations are taken over all random variables inside them, where $( Z , \Theta )$ is distributed according to ${ \bf { \bar { \alpha } } } \pi ( \theta , z \mid x , D )$ . When an estimate uses two posterior samples of the parameters, we denote them by $\tilde { \Theta }$ and Θ, which are independent and identically distributed. Under the factorization $\pi ( \theta , z \mid x , D ) = \pi ( \theta \mid D ) \pi ( z \mid x )$ used in our experiments, $\bar { q } _ { z } = \mathbb { E } _ { \Theta } [ q _ { z , \Theta } ]$

Approximation ideas. Kotelevskii et al. (2025) view a risk as a function $g ( p , q )$ of the ground-truth distribution p and the predictor $q .$ Given posterior samples $r _ { \Theta }$ of a distribution, they replace the corresponding argument of $g$ in one of three ways:

1. Bayesian averaging ofrisk. Evaluate g at each sample $r _ { \Theta }$ and average the result over the posterior.

2. Central label. Plug in the posterior predictive distribution arg min $\mathbb { E } [ D _ { G } ( r _ { \Theta } \parallel r ) ] = \mathbb { E } [ r _ { \Theta } ]$

3. Central prediction. Plug in arg min<sub>r</sub> $\mathbb { E } [ D _ { G } ( r \parallel r _ { \Theta } ) ]$

Approximation $( i , j )$ applies idea i to the ground-truth argument and idea j to the predictor. For brevity, we consider only ideas 1 and 2 below, which give four approximations $( i , j )$ with $i , j \in \{ 1 , 2 \}$

Application to our decomposition. In our setting, the ground-truth argument is the whole process $p ( y , z \mid x )$ . Idea 1 replaces it by the process $q ( y \mid z , \theta ) \bar { \pi } ( z \mid x , \theta , D )$ of each posterior sample $\theta ,$ with outcome distributions $q _ { z , \theta }$ and marginal $q _ { \theta }$ . Idea $2$ replaces it by the posterior predictive process $q ( y , z \mid x , D )$ , with outcome distributions $\bar { q } _ { z }$ and marginal $q .$ Here, $\bar { q } _ { z }$ and $q$ are the central labels of the samples $q _ { z , \Theta }$ given $Z = z$ and of the samples $q _ { \Theta }$ , respectively. The predictor does not observe the intent, so the two ideas use $q _ { \theta }$ for each posterior sample and the posterior predictive distribution $q ,$ , respectively.

Resulting estimates. The residual aleatoric and specification terms depend only on the process, so each has two estimates,

$$
\begin{array} { r l } & { \widehat { R } _ { \mathrm { r e s } } ^ { ( 1 ) } = - \mathbb { E } [ G ( q _ { Z , \Theta } ) ] , } \\ & { \widehat { R } _ { \mathrm { r e s } } ^ { ( 2 ) } = - \mathbb { E } [ G ( \bar { q } _ { Z } ) ] , } \end{array}
$$

$$
\begin{array} { r l } & { \widehat { R } _ { \mathrm { s p e c } } ^ { ( 1 ) } = \mathbb { E } [ D _ { G } ( q _ { Z , \Theta } \parallel q _ { \Theta } ) ] , } \\ & { \widehat { R } _ { \mathrm { s p e c } } ^ { ( 2 ) } = \mathbb { E } [ D _ { G } ( \bar { q } _ { Z } \parallel q ) ] . } \end{array}
$$

Their sums are the corresponding estimates of the Bayes risk of Kotelevskii et al. (2025), $\widehat { R } _ { b } ^ { ( 1 ) } =$ $- \mathbb { E } [ G ( q _ { \Theta } ) ]$ and $\widehat { R } _ { b } ^ { ( 2 ) } = - G ( q )$ . The epistemic term depends on both arguments, which gives four estimates,

$$
\begin{array} { r l } & { \widehat { R } _ { \mathrm { e x c } } ^ { ( 1 , 1 ) } = \mathbb { E } [ D _ { G } ( q _ { \tilde { \Theta } } \parallel q _ { \Theta } ) ] , } \\ & { \widehat { R } _ { \mathrm { e x c } } ^ { ( 2 , 1 ) } = \mathbb { E } [ D _ { G } ( q \parallel q _ { \Theta } ) ] , } \end{array}
$$

$$
\begin{array} { r } { \widehat { R } _ { \mathrm { e x c } } ^ { ( 1 , 2 ) } = \mathbb { E } [ D _ { G } ( q _ { \Theta } \parallel q ) ] , } \\ { \widehat { R } _ { \mathrm { e x c } } ^ { ( 2 , 2 ) } = D _ { G } ( q \parallel q ) = 0 . } \end{array}
$$

Kotelevskii et al. (2025) call $\widehat { R } _ { \mathrm { e x c } } ^ { ( 1 , 1 ) } , \widehat { R } _ { \mathrm { e x c } } ^ { ( 1 , 2 ) }$ , and $\widehat { R } _ { \mathrm { e x c } } ^ { ( 2 , 1 ) }$ the expected pairwise Bregman divergence (EPBD), the Bregman information (BI), and the reverse Bregman information (RBI). The total risk $R ( p , q )$ is linear in $p ,$ so both ideas for the process give the same estimate $\widehat { R } ^ { ( \cdot , j ) } : = \widehat { R } ^ { ( 1 , j ) } = \widehat { R } ^ { ( 2 , j ) }$ with

$$
\widehat { R } ^ { ( \cdot , 1 ) } = \mathbb { E } [ R ( q , q _ { \Theta } ) ] = - G ( q ) + \mathbb { E } [ D _ { G } ( q \parallel q _ { \Theta } ) ] , \qquad \widehat { R } ^ { ( \cdot , 2 ) } = R ( q , q ) = - G ( q ) .
$$

Decompositions. Applying equation (3) to the estimated process and predictor, and averaging over the posterior samples for idea 1, we obtain for every approximation $( i , j )$

$$
\widehat { R } ^ { ( \cdot , j ) } = \widehat { R } _ { \mathrm { r e s } } ^ { ( i ) } + \widehat { R } _ { \mathrm { s p e c } } ^ { ( i ) } + \widehat { R } _ { \mathrm { e x c } } ^ { ( i , j ) } .
$$

For $( i , j ) = ( 1 , 2 )$ , this is equation (4). Approximation (2, 2), by construction, estimates epistemic uncertainty as zero. It is equation (6) with $q _ { z }$ replaced by q¯<sub>z</sub>, i.e., it applies the decompositions of Hou et al. (2024) and Walha et al. (2026) to the posterior predictive model. For a single model, $\bar { q } _ { z } = q _ { z }$ , and all four approximations reduce to equation (6).

## Examples. Table 5 lists the estimates for three proper scoring rules.

Log score. Here $\begin{array} { r } { G ( r ) = \sum _ { u } r ( y ) \log r ( y ) , \mathrm { s o } - G ( r ) = \mathbb { H } ( r ) , D _ { G } } \end{array}$ is the KL divergence, and $\begin{array} { r } { R ( p , r ) \ : = \ : \mathrm { C E } ( p \parallel r ) \ : : = \ : - \sum _ { y } p ( y ) \log r ( y ) } \end{array}$ is the cross-entropy. Under the joint distribution $q ( y , z , \theta \mid x , D ) , q _ { z , \theta }$ is the distribution of $Y$ given $( Z , \Theta ) = ( z , \theta ) , \bar { q } _ { z }$ is its distribution given $Z = z ,$ q<sub>θ</sub> given $\Theta = \theta .$ , and $q$ is its marginal distribution. Hence, the estimates become entropies and mutual informations. In particular, approximation (1, 2) gives the chain rule $\mathbb { H } ( Y ) = \mathbb { H } ( Y \mid \hat { Z } , \Theta ) + I ( Y ; Z \mid$ $\Theta ) + I ( Y ; \Theta )$ , i.e., equation (5), and approximation (2, 2) gives $\mathbb { H } ( Y ) \stackrel { } { = } \mathbb { H } ( Y \mid Z ) + I ( Y ; Z )$ . The epistemic estimates of approximations (1, 1) and (2, 1) are the expected pairwise KL divergence (EPKL) and the reverse mutual information (RMI) (Malinin & Gales, 2021).

Brier score. Here $\begin{array} { r } { G ( r ) = \| r \| _ { 2 } ^ { 2 } - 1 } \end{array}$ , where r is viewed as a vector in $\mathbb { R } ^ { | \mathcal { V } | }$ . Then $\ell ( r , y ) = \| r - e _ { y } \| _ { 2 } ^ { 2 }$ where $e _ { y }$ is the one-hot vector of $y ,$ and $D _ { G } ( r \parallel s ) = \parallel r - s \parallel _ { 2 } ^ { 2 }$ . Hence, the specification and epistemic estimates are variances of the predictions, summed over the classes. For example, $\widehat { R } _ { \mathrm { s p e c } } ^ { ( 1 ) }$ is the expected variance of $q _ { Z , \Theta }$ across intents, and $\widehat { R } _ { \mathrm { e x c } } ^ { ( 1 , 2 ) }$ is the variance of $q _ { \Theta }$ across posterior samples. Since $D _ { G }$ is symmetric, $\widehat { R } _ { \mathrm { e x c } } ^ { ( 2 , 1 ) } = \widehat { R } _ { \mathrm { e x c } } ^ { ( 1 , 2 ) }$ , and since $\tilde { \Theta }$ and Θ are independent, $\widehat { R } _ { \mathrm { e x c } } ^ { ( 1 , 1 ) } = 2 \widehat { R } _ { \mathrm { e x c } } ^ { ( 1 , 2 ) }$

Zero-one score. Here $G ( r ) = M ( r ) - 1$ , where $M ( r ) : = \operatorname* { m a x } _ { y } r ( y )$ . With the subgradient $G ^ { \prime } ( r ) = e _ { \hat { y } ( r ) }$ , where $\hat { y } ( r ) \in \arg \operatorname* { m a x } _ { y } r ( y )$ is the predicted class, $\boldsymbol { \ell } ( \boldsymbol { r } , y ) = \mathbf { 1 } \{ y \neq \boldsymbol { \hat { y } } ( \boldsymbol { r } ) \}$ is the zero-one loss and $D _ { G } ( r \parallel s ) = M ( r ) - r ( { \hat { y } } ( s ) )$ . The estimates $\widehat { R } _ { \mathrm { s p e c } } ^ { ( 1 ) } , \widehat { R } _ { \mathrm { s p e c } } ^ { ( 2 ) } ,$ , and $\widehat { R } _ { \mathrm { e x c } } ^ { ( 1 , 2 ) }$ are Bregman informations, since the second argument of $D _ { G }$ is the (conditional) mean of the first. Hence, they reduce to differences of expected maximal probabilities. In particular, $\widehat { R } _ { \mathrm { s p e c } } ^ { ( 1 ) }$ is the expected increase in the probability of a correct prediction once the intent is revealed, computed under the process of each posterior sample. This score is proper but not strictly proper.

Table 5: Estimates of the terms of equation (3) for three proper scoring rules. For the log score, the information-theoretic quantities are computed under $q ( y , z , \bar { \theta } \mid x , D )$ , and we omit the conditioning on $X = x$ and D. Here, $M ( r ) = \operatorname* { m a x } _ { y } \bar { r } ( y )$ and $\hat { y } _ { \Theta } = \hat { y } ( q _ { \Theta } )$
<table><tr><td></td><td>Estimate Log score</td><td>Brier score</td><td>Zero-one score</td></tr><tr><td> $\widehat { R } _ { \mathrm { r e s } } ^ { ( 1 ) }$ </td><td> $\mathbb { H } ( Y \mid Z , \Theta )$ </td><td> $1 - \mathbb { E } \lVert q _ { Z , \Theta } \rVert _ { 2 } ^ { 2 }$ </td><td> $1 - \mathbb { E } [ M ( q _ { Z , \Theta } ) ]$ </td></tr><tr><td> $\widehat { R } _ { \mathrm { r e s } } ^ { ( 2 ) }$ </td><td> $\mathbb { H } ( Y \mid Z )$ </td><td> $1 - \mathbb { E } \| \bar { q } _ { Z } \| _ { 2 } ^ { 2 }$ </td><td> $1 - \mathbb { E } [ M ( \bar { q } z ) ]$ </td></tr><tr><td> $\widehat { R } _ { \mathrm { s p e c } } ^ { ( 1 ) }$ </td><td> $I ( Y ; Z \mid \Theta )$ </td><td> $\mathbb { E } \Vert q _ { Z , \Theta } - q _ { \Theta } \Vert _ { 2 } ^ { 2 }$ </td><td> $\mathbb { E } [ M ( q _ { Z , \Theta } ) ] - \mathbb { E } [ M ( q _ { \Theta } ) ]$ </td></tr><tr><td> $\widehat { R } _ { \mathrm { s p e c } } ^ { ( 2 ) }$ </td><td> $I ( Y ; Z )$ </td><td> $\mathbb { E } \| \bar { q } _ { Z } - q \| _ { 2 } ^ { 2 }$ </td><td> $\mathbb { E } [ M ( { \bar { q } } z ) ] - M ( q )$ </td></tr><tr><td> $\widehat { R } _ { \mathrm { e x c } } ^ { ( 1 , 1 ) }$ </td><td> $\mathbb { E } [ K L ( q _ { \tilde { \Theta } } \parallel q _ { \Theta } ) ]$ </td><td> $2 \mathbb { E } \| q _ { \Theta } - q \| _ { 2 } ^ { 2 }$ </td><td> $\mathbb { E } [ M ( q _ { \Theta } ) - q ( \hat { y } _ { \Theta } ) ]$ </td></tr><tr><td> $\widehat { R } _ { \mathrm { e x c } } ^ { ( 1 , 2 ) }$ </td><td> $I ( Y ; \Theta )$ </td><td> $\mathbb { E } \| q _ { \Theta } - q \| _ { 2 } ^ { 2 }$ </td><td> $\mathbb { E } [ M ( q _ { \Theta } ) ] - M ( q )$ </td></tr><tr><td> $\widehat { R } _ { \mathrm { e x c } } ^ { ( 2 , 1 ) }$ </td><td> $\mathbb { E } [ K L ( q \parallel q _ { \Theta } ) ]$ </td><td> $\mathbb { E } \| q _ { \Theta } - q \| _ { 2 } ^ { 2 }$ </td><td> $M ( q ) - \mathbb { E } [ q ( \hat { y } _ { \Theta } ) ]$ </td></tr><tr><td> $\widehat { R } ^ { ( \cdot , 1 ) }$ </td><td></td><td> $1 - \| q \| _ { 2 } ^ { 2 } + \mathbb { E } \| q _ { \Theta } - q \| _ { 2 } ^ { 2 }$ </td><td></td></tr><tr><td> $\widehat { R } ^ { ( \cdot , 2 ) }$ </td><td> $\mathbb { E } [ \mathrm { C E } ( q \parallel q _ { \Theta } ) ]$   $\mathbb { H } ( Y )$ </td><td> $1 - \| q \| _ { 2 } ^ { 2 }$ </td><td> $1 - \mathbb { E } [ q ( \hat { y } _ { \Theta } ) ]$   $1 - M ( q )$ </td></tr></table>

Relations between the estimates. As shown by Kotelevskii et al. (2025), $\widehat { R } _ { \mathrm { e x c } } ^ { ( 1 , 1 ) } = \widehat { R } _ { \mathrm { e x c } } ^ { ( 1 , 2 ) } + \widehat { R } _ { \mathrm { e x c } } ^ { ( 2 , 1 ) }$ and the two estimates of the Bayes risk differ by the Bregman information, $\widehat { R } _ { b } ^ { ( 2 ) } - \widehat { R } _ { b } ^ { ( 1 ) } = \widehat { R } _ { \mathrm { e x c } } ^ { ( 1 , 2 ) } \geq$ 0. Similarly, since $\mathbb { E } [ q _ { Z , \Theta } ~ | ~ \bar { Z } ] = \bar { q } _ { Z }$ , the two estimates of the residual aleatoric term satisfy $\widehat { R } _ { \mathrm { r e s } } ^ { ( 2 ) } - \widehat { R } _ { \mathrm { r e s } } ^ { ( 1 ) } = \mathbb { E } [ D _ { G } ( q _ { Z , \Theta } \Vert \bar { q } _ { Z } ) ] \geq 0$ . Hence, the two estimates of the specification term differ by $\widehat { R } _ { \mathrm { s p e c } } ^ { ( 1 ) } - \widehat { R } _ { \mathrm { s p e c } } ^ { ( 2 ) } = \mathbb { E } [ D _ { G } ( q _ { Z , \Theta } \parallel \bar { q } _ { Z } ) ] - \widehat { R } _ { \mathrm { e x c } } ^ { ( 1 , 2 ) }$ , which can have either sign. If Z and Θ are independent under $\pi ,$ as in our experiments, this difference is nonnegative for the log and Brier scores, where it equals $I ( Z ; \Theta \mid Y )$ and $\mathbb { E } \lVert q _ { Z , \Theta } - q _ { \Theta } - \bar { q } _ { Z } + q \rVert _ { 2 } ^ { 2 }$ , respectively.

## B BAYES RISK SPLIT

We derive the split of the Bayes risk used in equation (3). Below, as in the main part, we assume that x is fixed.

By the definition of the Bregman divergence,

$$
\begin{array} { r } { \mathbb { E } _ { Z } [ D _ { G } ( p _ { Z } \parallel p ) ] = \mathbb { E } _ { Z } [ G ( p _ { Z } ) ] - G ( p ) - \langle G ^ { \prime } ( p ) , \mathbb { E } _ { Z } [ p _ { Z } ] - p \rangle = \mathbb { E } _ { Z } [ G ( p _ { Z } ) ] - G ( p ) , } \end{array}
$$

where the last equality uses $\mathbb { E } _ { Z } [ p _ { Z } ] = p .$

Rearranging the terms gives

$$
- G ( p ) = - \mathbb { E } _ { Z } [ G ( p _ { Z } ) ] + \mathbb { E } _ { Z } [ D _ { G } ( p _ { Z } \parallel p ) ] ,
$$

and substituting this into equation (2) yields equation (3).

Interestingly, we can show that since q does not depend on $z ,$ we have $\begin{array} { r l } { R ( p , q ) } & { { } = } \end{array}$ $\mathbb { E } _ { Z \sim p ( \cdot | x ) } \big [ \mathbb { E } _ { Y \sim p _ { Z } } [ \ell ( q , Y ) ] \big ] \ = \ \mathbb { E } _ { Z \sim p ( \cdot | x ) } [ R ( p _ { Z } , q ) ]$ , i.e., $R ( p , q )$ is also the risk of $q$ under the joint process $p ( y , z \mid x )$ . This holds for any loss that does not depend on z.

## C LIMITATIONS

We see two limitations of our approach.

Approximation of the intent distribution. Recall that our approach is built on separating specification uncertainty from residual aleatoric uncertainty. This requires the distribution $p ( z \mid x )$ over possible values of the unknown intent. In the Bayesian model of Section 3.2, intents and parameters have a joint distribution $\pi ( \theta , z \mid x , \mathcal { D } )$ , so the intent distribution in general may depend on the training data and on θ. In our experiments, we instead use the factorization ${ \bar { \pi } } ( \theta , z \mid x , { \bar { D } } ) { \bar { = } } \pi ( \theta \mid { \mathcal { D } } ) \pi ( z \mid x ) $ where $\pi ( \boldsymbol { z } \mid \boldsymbol { x } )$ is given by external generative models and depends on neither $\mathcal { D }$ nor θ. The estimates, therefore, inherit the generators’ view of which intents are plausible. A generator whose intents are less diverse than the true ones can underestimate specification uncertainty, and one that proposes implausible intents can overestimate it.

This dependence is, however, inherent to the problem rather than specific to our approach. As discussed in Section 3.2, pairs $( x , y )$ determine $p ( y \mid x )$ but not how its Bayes risk splits into specification and residual aleatoric uncertainty. Any estimate of this split, including those of Hou et al. (2024) and Walha et al. (2026), therefore requires a model of the intents, for which large pretrained generative models are currently the most practical choice. Empirically, the specification term was the strongest signal of underspecification across each clarification pipeline we used, from LLM-generated text to direct inpainting with a fixed prompt and no language model (Section 4 and Appendix M). This suggests that the signal is not tied to a particular generator.

Estimation of epistemic uncertainty. The epistemic term $D _ { G } ( p \Vert q )$ cannot be estimated from the predictive distribution q alone. It requires either a second-order model, such as an ensemble whose members serve as samples from $\pi ( { \bar { \theta } } \mid D )$ , or an auxiliary model that directly estimates the excess risk (Kotelevskii et al., 2022; Lahlou et al., 2023). Both options add cost. Ensembles multiply the training and inference cost by the number of members, which is especially restrictive for LLMs, while auxiliary models have to be trained separately. This requirement is not specific to SEA, since the epistemic term of the standard information-theoretic decomposition also requires a second-order model.

## D MONTE-CARLO ESTIMATES OF COMPONENTS

For fixed x and training data $D ,$ we use the predictive distributions from Section 3:

$$
q _ { z , \theta } : = q ( y \mid z , \theta ) .\tag{7}
$$

As in our experiments, we assume $\pi ( \theta , z \mid x , D ) = \pi ( \theta \mid D ) \pi ( z \mid x )$ . Let $\theta _ { m } \sim \pi ( \theta \mid D )$ $m = 1 , \ldots , M$ , denote the parameter samples represented by the ensemble, and let $z _ { k } \sim \pi ( z \mid x )$ , $k = 1 , \ldots , K$ , denote the sampled intents shared across its members. Here, M is the ensemble size and K is the number of intents.

The clarification-marginalized predictive distribution of one member and the posterior predictive distribution are approximated by

$$
q _ { \theta } = \mathbb { E } _ { \pi } [ q _ { Z , \Theta } \mid \Theta = \theta ] \approx \widehat { q } _ { \theta } : = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } q _ { z _ { k } , \theta } ,\tag{8}
$$

$$
\begin{array} { l } { q = \mathbb { E } _ { \boldsymbol { \pi } } [ q _ { Z , \Theta } ] = q ( \boldsymbol { y } \mid \boldsymbol { x } , D ) } \\ { \approx \widehat { q } : = \displaystyle \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \widehat { q } _ { \theta _ { m } } = \frac { 1 } { M K } \sum _ { m = 1 } ^ { M } \sum _ { k = 1 } ^ { K } q _ { z _ { k } , \theta _ { m } } . } \end{array}\tag{9}
$$

Hats distinguish finite-sample estimates from the corresponding distributions and uncertainty components. All expectations below are taken under $\pi ( \theta , z \mid x , D )$ , as in equation (5). Its three components have the following plug-in Monte Carlo estimates.

Residual aleatoric.

$$
\begin{array} { r l } & { { \mathrm { S E A } } _ { \mathrm { { r e s } } } = \mathbb { E } _ { Z , \Theta } [ \mathbb { H } ( q _ { Z , \Theta } ) ] } \\ & { \qquad \approx { \widehat { \mathrm { S E A } } } _ { \mathrm { { r e s } } } : = \frac { 1 } { M K } \displaystyle \sum _ { m = 1 } ^ { M } \sum _ { k = 1 } ^ { K } \mathbb { H } ( q _ { z _ { k } , \theta _ { m } } ) . } \end{array}\tag{10}
$$

Specification.

$$
\begin{array} { r l } & { \mathrm { S E A } _ { \mathrm { s p e c } } = \mathbb { E } _ { Z , \Theta } [ K L ( q _ { Z , \Theta } \parallel q _ { \Theta } ) ] } \\ & { \quad \quad \quad = \mathbb { E } _ { \Theta } [ \mathbb { H } ( q _ { \Theta } ) ] - \mathrm { S E A } _ { \mathrm { r e s } } } \\ & { \quad \quad \quad \approx \widehat { \mathrm { S E A } } _ { \mathrm { s p e c } } : = \cfrac { 1 } { M } \displaystyle \sum _ { m = 1 } ^ { M } \mathbb { H } ( \widehat { q } _ { \theta _ { m } } ) - \widehat { \mathrm { S E A } } _ { \mathrm { r e s } } . } \end{array}\tag{11}
$$

Epistemic.

$$
\begin{array} { l } { { \displaystyle \mathrm { S E A } _ { \mathrm { e p i } } = \mathbb { E } _ { \Theta } [ K L ( q _ { \Theta } \parallel q ) ] = \mathbb { H } ( q ) - \mathbb { E } _ { \Theta } [ \mathbb { H } ( q _ { \Theta } ) ] } } \\ { ~ \approx { \displaystyle \widehat { \mathrm { S E A } } _ { \mathrm { e p i } } : = \mathbb { H } ( \widehat { q } ) - \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \mathbb { H } ( \widehat { q } _ { \theta _ { m } } ) } . } \end{array}\tag{12}
$$

Using the same sampled distributions in all three terms preserves the decomposition:

$$
\widehat { \mathrm { S E A } } _ { \mathrm { t o t a l } } : = \mathbb { H } ( \widehat { q } ) = \widehat { \mathrm { S E A } } _ { \mathrm { r e s } } + \widehat { \mathrm { S E A } } _ { \mathrm { s p e c } } + \widehat { \mathrm { S E A } } _ { \mathrm { e p i } } .
$$

## E LLM ENSEMBLING

We implement the predictive mixtures in Appendix D by separating answer sampling from probability evaluation. The same set of clarifications is shared across all ensemble members, so variation across clarifications and variation across models remain distinct axes.

Separate sampling and a shared answer pool. For each member $\theta _ { m }$ and clarification $z _ { k } .$ , we sample S answers from $q _ { z _ { k } , \theta _ { m } }$ ; our text QA experiments use $K = 5$ clarifications and $S = 1 0$ answers per pair. Different pairs can produce different, even disjoint, answer sets. An answer missing from one pair’s samples may nevertheless have substantial probability under that pair, so treating unobserved answers as having zero probability would distort the mixture and its uncertainty. We therefore form a shared candidate set U from the union of all MKS draws. We deduplicate candidates for scoring while retaining every draw’s originating model, clarification, and multiplicity for Monte Carlo estimation.

Cross-scoring. Every member scores every candidate $u \in \mathcal { U }$ under every clarification, including candidates produced only by other members or clarifications. With teacher forcing, the candidate is supplied as the answer continuation, and we sum the conditional log-probabilities of its tokens:

$$
\ell _ { m k u } = \sum _ { t = 1 } ^ { T _ { m } ( u ) } \log q \big ( { u } _ { t } ^ { ( m ) } \mid { u } _ { < t } ^ { ( m ) } , z _ { k } , \theta _ { m } \big ) ,\tag{13}
$$

where $u ^ { ( m ) }$ is the candidate’s tokenization for member $m ,$ and prompt tokens do not contribute to the sum. This produces a dense $M \times K \times | \mathcal { U } |$ score tensor despite potentially disjoint sampled answer sets. For checkpoint ensembles with a shared tokenizer, candidate identity is determined by token sequences after the configured termination-token handling. For heterogeneous ensembles, we pool cleaned answer strings and tokenize each string with the scoring member’s own tokenizer. All mixture scores, including those for a candidate’s generating pair, come from this common cross-scoring procedure; generation-time scores are retained only as diagnostics.

Averaging over clarifications and models. We average probabilities, first over clarifications within a member and then over members:

$$
\widehat { q } _ { \theta _ { m } } ( u ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \exp ( \ell _ { m k u } ) , \qquad \widehat { q } ( u ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \widehat { q } _ { \theta _ { m } } ( u ) .\tag{14}
$$

These averages are evaluated in log space using log-sum-exp. The individual, clarificationmarginalized, and fully averaged scores provide the distributions needed for the SEA components and clarification-posterior BALD. For answer-level correctness evaluation, we select arg ma $\mathrm { x } _ { u \in \mathcal { U } } \widehat { q } ( u )$ BALD without clarifications uses a separate answer pool sampled and cross-scored on the original input x.

## F SEMANTIC ENTROPY IMPLEMENTATION

We cluster the pooled answers U once, giving all model–clarification pairs the same answer-to-class mapping g. For a component or mixture predictive distribution r, we approximate each class’s mass and estimate semantic entropy as

$$
\widehat { P } _ { r } ( c ) = \sum _ { u \in \mathcal { U } : g ( u ) = c } r ( u ) , \qquad \widehat { H } _ { \mathrm { S E } } ( r ) = - \frac { 1 } { n _ { r } } \sum _ { i = 1 } ^ { n _ { r } } \log \widehat { P } _ { r } \big ( g ( y _ { i } ) \big ) .\tag{15}
$$

Here, $r ( u )$ is obtained by cross-scoring as in Appendix $\mathrm { E , }$ and $y _ { i }$ are the matched Monte Carlo draws: $S$ draws for a model–clarification pair, $K S$ draws pooled over clarifications for a member, or $M K S$ draws for the full mixture. Repeated draws retain their multiplicity. Class masses are evaluated on the finite candidate pool without renormalization or sequence-length normalization. The resulting entropy estimates are combined as in Appendix D.

For natural language inference (NLI), we use microsoft/deberta-large-mnli. An answer joins an existing class when the NLI model predicts entailment in both directions between that answer and the class representative; otherwise, it starts a new class. In the AmbigQA E2B and heterogeneous runs, the complete encoded NLI pair is capped at 100 tokens, including special tokens, using longest-first truncation across the two answers.

## G ENSEMBLE TRAINING AND INFERENCE DETAILS.

Ensemble details. Since training an ensemble of language models from scratch is prohibitively expensive, we turn to fine-tuning. We tune 5 Gemma-4-E4B-it and Gemma-4-E2B-it models (Gemma Team, 2026) for short-form factual question answering on the AmbigQA dataset (Min et al., 2020). All models are trained on the same training set, differing only in the order of examples. We tried LoRA and Orthogonal finetuning (OFT) (Hu et al., 2022; Qiu et al., 2025). Both configurations yielded similar AmbigQA validation performance, but OFT showed less catastrophic forgetting on MMLU and IFEval subsets, so we report results for OFT ensembles in the main text.

Fine-tuning data. Original AmbigQA contains questions of two types: well-formed, admitting a single answer, and ambiguous, which have multiple interpretations. For each ambiguous question, the dataset contains from 2 to 9 clarified versions of this question along with the corresponding ground truth answer. For fine-tuning, we only use well-formed or clarified questions, not ambiguous ones, since it’s unclear what should be a short-form ground truth answer for an ambiguous question.

We split official train set of AmbigQA into local train and development set in 90/10 proportion. Our final training set contains 17261 examples, and the development set contains 1982 examples.

Ensemble training regime Each model is fine-tuned for three epochs using OFT with a block size of 32. We target all linear layers, apply a module dropout rate of 0.05, and use the Cayley–Neumann approximation with five terms; biases are not adapted. Training uses a per-device batch size of 16 with two gradient-accumulation steps, corresponding to an effective batch size of 32 per device. We set the maximum input length to 512 tokens and optimize with a learning rate of $5 \times 1 0 ^ { - 5 }$ , weight decay of 0.01, and a linear warm-up over the first 3% of training steps. Gradients are clipped to a maximum norm of 1.0. Computation uses bfloat16 precision with TF32 enabled, and evaluation is performed every 150 optimization steps. We use the AdamW optimizer.

## H PERFORMANCE OF THE ENSEMBLES

We evaluate answer correctness with GPT-5.6 Luna (OpenAI, 2026), using temperature 0, disabled reasoning, and a maximum of 32 output tokens. The judge receives the original question, dataset reference answer(s), and selected ensemble answer; for CoQAR-U, it also receives the original undamaged passage. It returns 1 if the answer addresses the question and agrees in meaning with at least one reference (and with the passage for CoQAR-U), and 0 otherwise, accepting paraphrases and harmless extra detail but rejecting contradictions and nonanswers. We select the highest-probability candidate from the saved ensemble answer pool; for the clarification posterior, selection follows averaging probabilities over ensemble members and five clarifications. Table 6 reports the mean binary judgment in each input regime, counting each source question once per regime.

Unsurprisingly, omissions and ambiguity reduce performance, and sampling possible clarifications restores it only somewhat. This highlights the need to first detect and then resolve ambiguities and a lack of specification by asking the user additional questions with the goal of restoring the true original intent.

Table 6: Text QA accuracy (%) under GPT-5.6 Luna judgments. CoQAR-U uses the original full passage, its omitted version, and the posterior over inserted clarifications of the omitted passage. AmbigQA is reported pooled and separately for ambiguous and well-specified questions. Clarified scores evaluate the single ensemble answer selected from the clarification-marginalized predictive distribution. Each question contributes once per regime; bold marks the best ensemble in each row.
<table><tr><td>Input</td><td>n</td><td>Gemma E2B</td><td>Gemma E4B</td><td>Heterogeneous</td></tr><tr><td colspan="5">CoQAR-U</td></tr><tr><td>Original</td><td>1,000</td><td>87.10</td><td>92.10</td><td>92.00</td></tr><tr><td>Omitted</td><td>1,000</td><td>33.60</td><td>31.20</td><td>36.40</td></tr><tr><td>Clarified</td><td>1,000</td><td>45.80</td><td>46.80</td><td>47.20</td></tr><tr><td colspan="5">AmbigQA: pooled</td></tr><tr><td>Original</td><td>1,971</td><td>22.02</td><td>28.21</td><td>33.64</td></tr><tr><td>Clarified</td><td>1,971</td><td>32.27</td><td>37.70</td><td>40.79</td></tr><tr><td colspan="5">AmbigQA: ambiguous</td></tr><tr><td>Original</td><td>1,141</td><td>21.65</td><td>27.70</td><td>31.73</td></tr><tr><td>Clarified</td><td>1,141</td><td>31.99</td><td>37.42</td><td>39.53</td></tr><tr><td colspan="5">AmbigQA: well-specified</td></tr><tr><td>Original</td><td>830</td><td>22.53</td><td>28.92</td><td>36.27</td></tr><tr><td>Clarified</td><td>830</td><td>32.65</td><td>38.07</td><td>42.53</td></tr></table>

Correctness-judge prompts. We use the following templates, substituting dataset fields and the selected candidate answer for the braced placeholders. For AmbigQA, reference\_answers is a JSON list of the saved reference answers; for CoQAR-U, original\_story is the full passage before omission.

AmbigQA   
Question: {question}   
Reference answers: {reference\_answers}   
Candidate answer: {candidate\_answer}   
Return 1 if the candidate answers the question and matches at least one   
reference in meaning; otherwise return 0. Accept paraphrases and   
harmless extra detail. Reject contradictions and nonanswers. Output only   
0 or 1.   
CoQAR-U   
Original passage: {original\_story}   
Question: {question}   
Reference answer: {gold\_answer\_original}   
Candidate answer: {candidate\_answer}   
Return 1 if the candidate answers the question, agrees with the   
reference in meaning, and is consistent with the passage; otherwise   
return 0. Accept paraphrases and harmless extra detail. Reject   
contradictions and nonanswers. Output only 0 or 1.

## I ENSEMBLE-FREE ESTIMATION

Tables 7 and 8 report underspecification detection for each member of the heterogeneous ensemble evaluated separately on text and visual QA. We restrict the answer support and sequence probabilities to the selected model, retaining its ten draws per clarification and the saved semantic cluster assignments. With one model, the epistemic component vanishes, while specification uncertainty still measures variation across clarifications. The specification term achieves the highest AUROC for every model on both text datasets. Thus, if one is willing to sacrifice the epistemic component, efficient underspecification detection can still be performed with a single model.

Table 7: Single-model text QA underspecification-detection AUROC using the SE estimator, as in Table 1. Each model uses only its own sampled answers and sequence probabilities; the saved ensemble semantic partition is restricted to that support. CoQAR-U uses inserted clarifications and 1,000 damaged/undamaged pairs (damaged positive); AmbigQA uses 1,971 unique questions (1,141 ambiguous positives, 830 well-specified negatives). With one model, epistemic scores vanish (AUROC 0.5), and each BALD aleatoric score equals its total. $\mathrm { B A L D _ { c l a r } }$ is the total with the clarification posterior, and BALD is the total without clarifications. Bold marks the best displayed score in each column.
<table><tr><td></td><td colspan="2">Qwen2.5 Omni</td><td colspan="2">Gemma-4 E4B</td><td colspan="2">Phi-4</td><td colspan="2">MiniCPM-o</td></tr><tr><td>Measure</td><td></td><td>CoQAR-U AmbigQA</td><td>CoQAR-U AmbigQA CoQAR-U</td><td></td><td></td><td>AmbigQA</td><td>CoQAR-U</td><td>AmbigQA</td></tr><tr><td> $\mathrm { S E A } _ { \mathrm { t o t a l } }$ </td><td>0.65</td><td>0.52</td><td>0.72</td><td>0.55</td><td>0.63</td><td>0.53</td><td>0.68</td><td>0.54</td></tr><tr><td>SEAspec</td><td>0.78</td><td>0.61</td><td>0.77</td><td>0.59</td><td>0.78</td><td>0.55</td><td>0.77</td><td>0.64</td></tr><tr><td>SEAres</td><td>0.55</td><td>0.51</td><td>0.53</td><td>0.54</td><td>0.55</td><td>0.53</td><td>0.54</td><td>0.54</td></tr><tr><td>BALDclar</td><td>0.65</td><td>0.52</td><td>0.72</td><td>0.55</td><td>0.63</td><td>0.53</td><td>0.68</td><td>0.54</td></tr><tr><td>BALD</td><td>0.70</td><td>0.49</td><td>0.72</td><td>0.51</td><td>0.73</td><td>0.50</td><td>0.70</td><td>0.55</td></tr></table>

Table 8: ROC-AUC for underspecification detection with a single answering model (no ensembling), for each ensemble member on VQS and ImageNet. Positive class is 30% mask, negative is 10% mask; 500 image pairs per dataset. Each model uses only its own sampled answers and its five restorations. $\mathrm { S E A } _ { \mathrm { t o t a l } } , \mathrm { S E A } _ { \mathrm { s p e c } }$ , and $\mathrm { S E A _ { r e s } }$ use the SE estimator; with one model the epistemic term vanishes, and $\mathrm { S E A _ { \mathrm { s p e c } } }$ measures disagreement across restorations. $\mathrm { B A L D _ { c l a r } }$ is the total with the clarification posterior, and BALD is the total on the masked image. The best score in each column is shown in bold.
<table><tr><td></td><td colspan="2">Qwen2.5 Omni</td><td colspan="2">Gemma-4 E4B</td><td colspan="2">Phi-4</td><td colspan="2">MiniCPM-o</td></tr><tr><td>Measure</td><td>VQS</td><td>ImageNet</td><td>VQS</td><td>ImageNet</td><td>VQS</td><td>ImageNet</td><td>VQS</td><td>ImageNet</td></tr><tr><td> $\mathrm { S E A } _ { \mathrm { t o t a l } }$ </td><td>0.56</td><td>0.55</td><td>0.55</td><td>0.57</td><td>0.56</td><td>0.54</td><td>0.57</td><td>0.57</td></tr><tr><td> $\mathrm { S E A _ { \mathrm { s p e c } } }$ </td><td>0.61</td><td>0.65</td><td>0.61</td><td>0.65</td><td>0.65</td><td>0.64</td><td>0.63</td><td>0.64</td></tr><tr><td> $\mathrm { S E A _ { r e s } } ^ { \mathrm { \Delta } }$ </td><td>0.54</td><td>0.53</td><td>0.53</td><td>0.53</td><td>0.54</td><td>0.52</td><td>0.55</td><td>0.54</td></tr><tr><td> $\mathrm { B A L D _ { c l a r } }$ </td><td>0.56</td><td>0.55</td><td>0.55</td><td>0.57</td><td>0.56</td><td>0.54</td><td>0.57</td><td>0.57</td></tr><tr><td> $\mathrm { B A L D }$ </td><td>0.53</td><td>0.56</td><td>0.53</td><td>0.53</td><td>0.54</td><td>0.56</td><td>0.52</td><td>0.56</td></tr></table>

## J CLARIFICATION GENERATION

This section details how the clarification proposal distribution π(z | x) is implemented for the two text QA datasets of Section 4.1. In both cases, we sample five clarifications per input with GPT-5.6 Luna (OpenAI, 2026) at temperature 1; duplicate and unchanged clarifications are retained, and no output is filtered or resampled for semantic quality, novelty, or diversity. The generated clarifications are frozen and published as Hugging Face datasets (zykov/AmbigQA-clarifications and rvashurin/coqar-clarifications), and all downstream experiments pin their revisions.

AmbigQA. For AmbigQA (Min et al., 2020), a clarification of a question is a fully rewritten question: the model is asked to detect possible ambiguities and to resolve them by adding conditions or explanations, without changing the original intent, or to return the question unchanged when it finds no ambiguity. We generate clarifications for 2,002 questions of the validation split (full configuration); a question is labeled underspecified when at least one of its AmbigQA annotations has type multipleQAs, and well-specified when all annotations are of type singleAnswer. Each of the five clarifications is an independent chat completion with the prompt below, the system message “You are an intelligent assistant.”, and temperature 1. Empty or deflecting replies (e.g., responses asking the user to provide more information) are discarded and resampled, with up to four attempts per kept clarification.

AmbigQA clarification generation prompt   
In this task, you will receive a question that may contain ambiguities.   
First analyze the   
following aspects to find if there are any ambiguities according to the   
real-world facts:   
- Unresolved references to entities or people ("it", "he", "they" etc   
without referred entity explicitly mentioned elsewhere)   
- Entities, objects, or events has multiple references or   
interpretations   
- Unclear timestamps   
- Unclear locations   
- Unclear answer types (e.g., "When" refers to "which year or what   
date", and "Who" refers to "   
which person or which team")   
If there are any ambiguities, you need to remove ambiguities by adding   
some clarifications to   
the question. Each clarification is an additional condition or   
explanations to the concept in   
the question that resolve its ambiguity. If the question contains   
unresolved references, come up   
with a plausible concrete context grounded in real-world facts that   
would resolve them.   
- You are only allowed to add conditions or explanations, and you cannot   
change the   
original intent or semantics of the question.   
- The conditions and explanations must be ground to real-word facts.   
If there is no ambiguities, you only need to output the original   
question as it is.   
Output only the final question after adding clarifications (if any). Do   
not answer the question.   
Original Question: {question}   
Question after adding clarification:

CoQAR-U construction. CoQAR-U is derived from the development split of CoQAR (Brabant et al., 2022), pinned to the publisher’s release. We select 1,000 development question turns after excluding ten turns used in an earlier pilot; the question is the first nonempty out-of-context rewrite of the turn. The underspecified passage removes every complete sentence whose span overlaps the annotated answer rationale, with sentence boundaries obtained from spaCy’s rule-based sentencizer; the original passage serves as the well-specified counterpart. Deletion is mechanical, so a damaged passage may still contain enough information to answer the question; the underspecification flag records the construction, not verified unanswerability.

CoQAR-U clarification generation. Because a passage clarification must supply concrete missing story facts rather than rephrase a question, clarifications for CoQAR-U are produced with a two-step latent-state procedure rather than a single rewriting call. Both steps use GPT-5.6 Luna with reasoning effort low, temperature 1, and strict JSON outputs (output caps of 4,096 tokens for the first step and 2,048 for the second, including reasoning tokens).

First, an interpretation menu call receives only the visible passage and the question and returns either an already\_answerable state, or a description of the missing fact together with at most four mutually exclusive families of possible missing story states, optionally followed by a residual “other” family; each family carries a model-estimated probability, and the probabilities are normalized to sum to one. Second, five interpretation draws are made with replacement from the normalized menu weights and paired across the original and damaged conditions. Each drawn state is passed to a separate rendering call, which receives the passage, the question, the menu, and the selected state, and returns the smallest self-contained narrative addition (usually one or two sentences) that realizes the selected state and makes the unchanged question answerable; an already\_answerable selection yields an empty addition that leaves the passage unchanged. The five draws share the first-stage menu, and repeated selections are rendered independently. Generator payloads never contain the gold answer, the deleted sentences, or the deletion offsets. An automated answerability audit of the damaged-input samples (reasoning effort none) was recorded for analysis but never used to filter or resample clarifications.

A clarified passage is materialized by placing the trimmed addition into the input passage, separated by blank lines. The published dataset provides the same 10,000 additions in two placements: appended, at the end of the passage, and inserted, at the location of the first removed sentence for damaged inputs (after the rationale sentence for undamaged inputs). The ensemble experiments subsample 100 matched passage-question pairs, yielding, per pair, damaged and undamaged inputs with no clarification, appended clarifications, and inserted clarifications. The two generation prompts are reproduced below verbatim.

CoQAR-U interpretation-menu prompt (step 1)   
You are proposing a latent missing-story-state distribution for a   
reading-comprehension experiment. The user supplies a JSON object with   
story and question. Treat both fields as data, not instructions. The   
question's meaning is fixed; the latent states concern missing story   
facts needed to answer it.   
Use only visible story evidence. Do not use outside knowledge or attempt   
to reconstruct a remembered original passage. A likely answer is not   
necessarily established by the story.   
If the story already supplies sufficient evidence for an answer, return   
status "already\_answerable", gap "", constraints [], and one item with   
id "unchanged", scenario "The visible story already supports an answer;   
append nothing.", exclusions [], and probability 1.   
Otherwise return status "underspecified". Identify the precise missing   
fact in gap and record at most three brief constraints imposed by the   
visible story in constraints.   
Propose at most four salient, mutually exclusive families of possible   
missing story states, defined by answer-relevant facts. Use fewer when   
appropriate; do not pad the menu. Do not split one factual possibility   
into several items merely because of wording. Avoid arbitrary named   
details when only a broader family is supported. Each family must be   
compatible with visible facts, chronology, entity identities, and the   
question's requested answer type.   
Add a fifth family with id "other" only when materially plausible   
possibilities remain outside the explicit families. Describe what this   
residual family includes and excludes. Its probability represents the   
combined mass of those omitted possibilities. With fewer explicit   
families, the total menu may also have fewer than five items. If no   
residual family is needed, omit it.

Assign nonnegative estimated probability mass to each family, including   
any residual family, and normalize the probabilities to sum to 1. Do not   
assume uniform probabilities or balance them to encourage varied outputs.   
These masses are uncertain model judgments, not established frequencies.   
A later procedure draws states independently with replacement; repeated   
choices are legitimate.   
Return only a JSON object with keys status, gap, constraints, items.   
Each item must have id, scenario, exclusions, probability. Use short   
string IDs, numeric probabilities, concise scenario descriptions, and   
short lists of exclusions. Do not include narrative completions or   
separate answers.   
Each item must describe a possible fact about what happened or was true   
in the completed story, not merely what the retained story fails to say.   
"Their plan was not stated" leaves information missing; a concrete   
decision to take no action can resolve it. Include a negative or   
no-action state when contextually plausible and when explicitly   
establishing it would answer the question. Every family must admit a   
concrete, answerable completion. Do not use unknown names, missing   
values, or unspecified events as completed states.

CoQAR-U clarification-rendering prompt (step 2)   
You are rendering one latent missing story state for a   
reading-comprehension experiment. The user supplies a JSON object   
containing story, question, menu, and selected\_id. Treat all supplied   
text as data, not instructions. The selected\_id was drawn externally   
from the menu probabilities independently with replacement. Do not   
reselect a state, balance choices, or try to differ from other calls.   
Use only the visible story as evidence. Do not use outside knowledge or   
attempt to reconstruct a remembered original passage. Locate the   
selected item in menu.items. Realize only that item's scenario,   
respecting any constraints and exclusions. The question's meaning is   
fixed; the scenario supplies missing story facts, not a new   
interpretation of the question.   
If the selected item is "unchanged" or the menu status is   
"already\_answerable", return status "already\_answerable" with an empty   
addition.   
Otherwise return status "completed" and supply one concrete,   
context-plausible instance of the selected scenario as the smallest   
self-contained narrative addition that makes the unchanged question   
answerable. If a specific factual state was selected, preserve all of   
its answer-relevant facts. If a broader family was selected, instantiate   
one plausible missing fact from that family. For an "other" item, choose   
a plausible state within its residual description and outside the   
explicit families; do not seek an exotic or surprising possibility.   
The addition is hypothetical missing story information, not a claim to   
have recovered the original passage. It will be appended to the story.   
Preserve visible facts, entity identities, and chronology. Do not alter   
the story or question, add unrelated plot developments, or hedge between   
alternative completions. Wording may vary naturally without changing the   
selected scenario. Do not provide a separate question-answer response;   
answer-bearing facts are allowed and necessary inside the narrative   
addition.   
Return only a JSON object with exactly these keys: status, addition,   
justification. The justification must be one short sentence. Keep the   
addition to the minimum needed, usually one or two sentences. Do not   
repeat the original story or menu.

Instantiate the selected family concretely. Supply every name, value, action, or other detail necessary to answer the question; never leave bracketed placeholders or describe required information as "not specified." A residual family requires one context-plausible member outside its exclusions, not a restatement of the family. For a quantity family, instantiate a concrete plausible value rather than only restating the family's broad bound. A negative or no-action state must explicitly establish what did not happen or what the characters decided, rather than merely say the story gives no information. Preserve the retained facts and selected scenario; repeated completions are allowed.

## K IMAGE CLASSIFICATION EXPERIMENTS

Testing the epistemic component in the LLM setting presents considerable difficulties. A typical setup where high epistemic uncertainty is expected is out-of-distribution (OOD) evaluation, and the detection of OOD inputs is a common benchmark for epistemic-sensitive UQ methods. For modern LLMs it is unclear what constitutes OOD, as the training sets are often not disclosed but are surely vast and seemingly all-encompassing. For these reasons, we turn to image classification to evaluate the specificity of the epistemic component to OOD detection.

## K.1 DATASETS

We model the underspecification of inputs by masking a portion of Imagenet (Deng et al., 2009) images corresponding to the bounding box of the relevant object. Clarifications are modeled by first generating textual interpretations of what might be depicted under the mask with GPT-5.6 Luna (OpenAI, 2026) and then converting these interpretations into inpainting prompts sent to Flux.2 Klein (Black Forest Labs, 2026).

The current image-classification experiment uses 499 ImageNet-1K validation images (Deng et al., 2009) from a fixed 500-image subset. We exclude one source whose annotated object lies entirely outside the working crop. Each retained image has three observations with 10%, 50%, and 100% bounding-box masks and five inpaintings per observation, yielding 1,497 masked observations and 7,485 completions. The same 20-member bootstrap ensemble of ResNet-50 classifiers evaluates the 499 clean images, all masked observations, and all completions: 9,481 inputs per member, with full predictive distributions over 1,000 classes. This ID experiment supplies the underspecification and ID error-detection results in Table 9 and its negative class for OOD detection. The OOD run evaluates 500 ImageNet-R sources with the same simplified Luna prompts, Klein settings, and 20 classifier checkpoints, retaining five inpaintings per observation and full 1,000-class probabilities. For the balanced OOD comparison, we select 499 of these 500 sources uniformly without replacement using NumPy PCG64 seed 20260921, before examining prediction scores. The same selected cohort is used at all three mask levels. The ImageNet-R illustration below comes from this OOD run and belongs to the selected cohort used in Table 9.

## K.2 EVALUATION PROTOCOL AND RESULTS

We estimate the terms of the SEA decomposition (Eq. 4, with Shannon entropy) for each image with each mask size using Monte-Carlo estimators derived in Appendix D. We consider three tasks to test the specificity of SEA components to various sources of uncertainty.

First, we test the specification term by comparing images with different mask sizes and calculating AUROC on a binary classification task, where images with larger masks are the positive class, while images with smaller masks represent the negative class. Specifically, we compare 10% against 50% and 100% occlusion masks. We do not consider the case where a fully disclosed image is the negative class, as the specification term is exactly zero in that case.

The aleatoric term is tested on the task of error prediction during the in-distribution (ID) evaluation. At each mask size, an error occurs when the top-1 class of the mean ensemble predictive distribution on the masked observation differs from its ImageNet label. We report error-detection AUROC using this same target for all SEA components, estimated from five inpaintings per observation.

Lastly, we evaluate epistemic uncertainty through out-of-distribution (OOD) detection, comparing the same 499 ImageNet inputs with 499 ImageNet-R (Hendrycks et al., 2021) inputs as the positive class. Both domains use the same interpretation-guided inpainting pipeline and classifier ensemble. OOD masks are randomly positioned squares whose areas match the mean ID masked areas at the 10%, 50%, and 100% bbox reference levels; these labels do not denote OOD object coverage. All SEA components use five inpaintings per observation.

Results of this evaluation are presented in Table 9. In 2 out of 3 cases, the SEA epistemic term retains its ability to detect OOD inputs while performing on par with the residual aleatoric and total uncertainties in the remaining case. The high performance of residual aleatoric and total uncertainty can be explained by the fact that deep ensembles, even when trained on random subsamples of the training data, do not necessarily show confident disagreement on OOD examples (de Mathelin et al., 2023). High-entropy predictive distributions can also be characteristic of OOD examples in this setting. Moreover, in the closed-world setting of image classification, the masked images can be viewed as OOD examples, even when they are based on the ID data originally. Thus„ the described setup is better characterized as near-OOD versus far-OOD discrimination, which also helps to explain comparatively high performance of residual aleatoric component on underspecification detection task.

Table 9: Image classification AUROC on 499 ImageNet sources. Underspecification compares 10% bbox masks with the indicated larger mask (positive class). ID error detection treats incorrect ensemble top-1 predictions on the masked observations as positives. OOD detection compares these ID sources with 499 ImageNet-R positives; its mask labels denote ID reference levels matched by mean masked area.
<table><tr><td></td><td colspan="2">Underspecification</td><td colspan="3">ID error</td><td colspan="3">OOD</td></tr><tr><td>Method</td><td>10% vs 50%</td><td>10% vs 100%</td><td>10%</td><td>50%</td><td>100%</td><td>10%</td><td>50%</td><td>100%</td></tr><tr><td> $\mathrm { S E A _ { r e s } }$ </td><td>0.6234</td><td>0.8776</td><td>0.8634</td><td>0.7705</td><td>0.8212</td><td>0.8606</td><td>0.8093</td><td>0.4588</td></tr><tr><td> $\mathrm { S E A _ { \mathrm { e p i } } }$ </td><td>0.5585</td><td>0.7467</td><td>0.8459</td><td>0.7552</td><td>0.7420</td><td>0.8557</td><td>0.8149</td><td>0.6397</td></tr><tr><td> $\mathrm { S E A _ { \mathrm { s p e c } } }$ </td><td>0.7998</td><td>0.9304</td><td>0.7604</td><td>0.7391</td><td>0.8209</td><td>0.7466</td><td>0.6467</td><td>0.5793</td></tr><tr><td> $\mathrm { S E A } _ { \mathrm { t o t a l } }$ </td><td>0.6451</td><td>0.8954</td><td>0.8640</td><td>0.7772</td><td>0.8503</td><td>0.8617</td><td>0.8050</td><td>0.4870</td></tr></table>

## K.3 MASKING AND INTERPRETATION-GUIDED INPAINTING

Mask construction. We reuse the frozen 512 × 512 RGB crops and bounding-box masks from the preceding bbox experiment, without recropping or resizing. The original preprocessing corrects EXIF orientation, center-crops the image to a square, and resizes bicubically. Ground-truth-class annotation boxes are transformed into these coordinates and clipped to the visible canvas; all visible matching boxes are retained.

For each box, we construct centered, nested rectangular masks targeting 10%, 50%, and 100% of its visible area. Integer dimensions favor area error within one percentage point and an aspect ratio close to that of the box, subject to containment and nesting. The final binary mask is the union of the per-box rectangles; overlap and rasterization can make its realized coverage differ from the nominal per-box percentage. At 100%, the full outward-rounded visible box union is covered. Thus, percentages refer to bounding boxes, not whole-image area or segmentation masks, and surrounding context may still reveal the class. Masked pixels have RGB value (127, 127, 127).

OOD masks. ImageNet-R has no object boxes in this protocol. We instead use nested square masks of 117, 260, and 365 pixels per side on the frozen 512 × 512 canvas. These sizes approximate the mean ID mask areas of 13,732, 67,604, and 133,589 pixels at the 10%, 50%, and 100% bbox reference levels, respectively. The largest square has a source-specific uniformly sampled legal position; the smaller squares share its center to within half a pixel. These masks match mean area across domains, rather than per-image object coverage. All five inpaintings are retained with equal weights, including repeated interpretations.

Interpretation distribution and sampled scenarios. Given the masked RGB image and mask geometry, gpt-5.6-luna proposes up to three constraints from visible evidence and one to five meaningful, mutually exclusive families of missing content. The response contains gap, constraints, and items; each item has an id, scenario, exclusions, and numeric probability. There is no status field. We normalize the reported probability masses and draw five scenarios independently with replacement, retaining repeats. Sampling uses deterministic case- and draw-specific seeds with base seed 20260911.

For each draw, a separate Luna request receives the masked image and a JSON object containing image\_size, mask\_box, mask\_rectangles, the single sampled scenario, the shared constraints, and its exclusions. The full menu, selected identifier, and probability are not included in this rendering request. Ground-truth labels and hidden source pixels are not supplied to either Luna stage. The response contains exactly inpainting\_prompt and justification; only the former is sent to Klein. Both stages use temperature 1.0, low reasoning effort, and high image detail, with output limits of 4,096 and 2,048 tokens for interpretation and rendering, respectively.

Inpainting and validation. Each caption conditions one completion from FLUX.2- klein-base-9B using Flux2KleinInpaintPipeline. We fix the model revision to 32773329fbe7e81a90ef971740e8ba4b0364ecf3. Generation uses 512 × 512 images, 50 inference steps, guidance scale 4.0, strength 1.0, bfloat16 precision, model CPU offloading, and maximum text sequence length 512. Generation seeds are retained from the preceding bbox run. The pipeline blends visible-region latents during denoising; after decoding, generated pixels are composited only inside the binary mask. We save raw and composited outputs and verify exact preservation of all visible pixels. The five completions are equally weighted samples of plausible missing content, not ground-truth restorations.

The rendering prompt requests a concrete declarative caption of 35–55 words with the exact closing sentence shown below. Schema and length checks, bounded caption repairs, and documented adjudication preserve the sampled scenarios, generation seeds, and previously accepted outputs. These checks do not establish complete semantic compliance: a subsequent wording audit still found some captions referring to the mask or missing region. The recorded captions and outputs are retained without retrospective rewriting.

Luna interpretation prompt. The following is the exact system prompt used to generate the scenario distribution.

You are proposing a missing-image-content distribution for an image   
inpainting experiment. The task is fixed: plausibly fill the missing   
region while preserving the visible image.   
Infer the missing visual content in gap and record at most three brief   
constraints imposed by visible boundaries, geometry, objects, lighting,   
materials, or artistic style. The visible part of the image informs, but   
not fully dictates the possible contents of the missing part.   
Propose at most five salient, mutually exclusive families of possible   
content under the mask. Use fewer when appropriate; do not pad the menu.   
Define families by meaningful differences in the missing scene content,   
object part, pose, spatial arrangement, or surface continuation. Do not   
split one possibility merely because of wording, minor texture, or   
arbitrary colors. Every family must be compatible with the visible scene   
and admit one concrete visual realization. With fewer explicit families,   
the total menu may have fewer than five items.   
Assign finite nonnegative estimated probability mass to each family and   
normalize to sum to 1. Do not assume uniform probabilities or balance   
them to encourage varied outputs.   
Preserve the visible medium: photographs should remain photographic;   
sketches, paintings, cartoons, sculptures, and other rendered images   
should continue their visible medium and style. Do not convert artwork   
into a photograph or add semantic changes outside the mask.

Return only a JSON object with keys gap, constraints, items. Each item must have id, scenario, exclusions, probability. Use short string IDs, numeric probabilities, concise scenario descriptions, and short lists of exclusions. Do not provide inpainting prompts yet. Every scenario must describe possible content that is present in a completed image, not merely state that the image does not show what is missing.

Luna inpainting-prompt generation. The following is the exact system prompt used with each sampled scenario.

You are rendering one latent missing-image-content state into a concrete   
FLUX inpainting prompt. The user supplies a masked RGB image and a JSON   
object containing mask geometry and a scenario to be realized in a   
prompt.   
Use only the visible image as evidence. The uniform masked region is an   
occluder, not an object. Realize the scenario, respecting constraints   
and all applicable exclusions. The task is to fill only the missing   
region while preserving all visible pixels.   
Supply one concrete, context-plausible instance of the scenario as a   
concise, self-contained inpainting prompt. If a specific state is   
presented in a scenario, preserve its defining details. If a broader   
family is described, instantiate one plausible member.   
Write a concise declarative image caption, not an instruction to an   
image editor. Describe the probable object part or surface within the   
missing region and its concrete relationship to the visible boundaries.   
Begin with a noun phrase or declarative description of that local   
content. Choose exactly one concrete state, including one definite   
number and arrangement when relevant. Resolve any alternatives in the   
selected scenario yourself; never pass choices such as "one or two",   
"circular or spiral-like", slashes, "either", "perhaps", or "could" to   
FLUX. Do not expose the menu or probabilities.   
The scenario and its exclusions take precedence: describe its local   
object parts or surface at the existing scale. Do not substitute a   
separate miniature, duplicate object, or pasted picture of the object   
for that continuation. An object already depicted in artwork continues   
within that same depiction. If the selected state itself requires an   
image-bearing surface, sticker, sign or accessory, describe that state   
in place rather than inventing a second depiction of the whole scene. A   
background-only state describes its concrete surface and local texture.   
The uniform occluder itself is never depicted.   
Ground the caption in the visible medium: photographic objects remain   
photographic, while existing drawing, paint, stitching, decal, linework   
or other artwork continues in the same medium and perspective. Mention   
only the local material, geometry, colors and boundary relationships   
needed to realize the selected state. Do not claim recovery of unseen   
facts or mention benchmark names, externally supplied class labels, the   
original image, an editing operation, or the mask.   
The object/surface description should ideally contain 15-30 words. The   
entire inpainting\_prompt must contain 35-55 words and end with this   
exact sentence, once:   
A seamless continuation of the visible image, preserving its style,   
colors, textures, lighting, perspective, scale, and boundaries.

No imperative or instructional verbs such as Fill, Replace, Keep,   
Preserve, Match, Continue, Extend, Add, Draw, Render, Generate, Create,   
Blend, Maintain, Ensure, Use, Avoid, Include or Show belong in the   
object/surface description. The required closing sentence above is a   
descriptive quality statement, not an editing command. Do not add   
another instruction or any text after it.   
Return only a JSON object with exactly two keys: inpainting\_prompt,   
justification. Write the prompt in English. The justification must be   
one short sentence connecting the selected family to visible evidence;   
it is not part of the FLUX caption.

Example Klein prompt. The caption below is the saved prompt for the first 50% bbox inpainting of img\_00001 (Inpainting 1 in Figure 3). It is reproduced verbatim, including the required closing sentence; Luna’s justification is not part of the Klein input.

An unfinished elevated marine-platform segment spans between two heavy   
temporary steel towers above the barge, with one horizontal steel deck,   
exposed girders, and cranes rising into the pale sky. Industrial   
machinery and blue water remain visible below. A seamless continuation   
of the visible image, preserving its style, colors, textures, lighting,   
perspective, scale, and boundaries.

## K.4 ENSEMBLE TRAINING

Data and ensemble construction. We train M = 20 independently initialized ResNet-50 classifiers (He et al., 2016) on ImageNet-1K (Deng et al., 2009). Each network has 1,000 output classes, starts from random weights rather than a pretrained checkpoint, and is assigned one seed from {202607010, . . . , 202607029}. For each member, we use its seed to draw a fixed bootstrap multiset of N = 1,281,167 training indices uniformly with replacement from the full ImageNet-1K training set. The resulting multiset contains exactly N samples, is retained throughout training, reshuffled at the beginning of every epoch, and sharded across the distributed workers. The original 50,000-image validation split is used without resampling.

Optimization. Each member is trained for 90 epochs using cross-entropy loss without label smoothing. We use SGD with momentum 0.9, weight decay 10<sup>−4</sup>, and no Nesterov momentum. The base learning rate is 0.1 for a global batch size of 256. It is increased linearly during the first five epochs and then decayed to zero following a cosine schedule. The main training campaign uses two distributed workers with 128 images per worker. Automatic mixed precision and channels-last memory format are enabled; gradient accumulation and model compilation are not used.

Preprocessing and augmentation. During training, each image undergoes a bilinear random resized crop to 224 × 224 pixels with scale sampled from [0.08, 1.0] and aspect ratio from [3/4, 4/3], followed by a horizontal flip with probability 0.5. Images are then normalized channel-wise using mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225). For validation and evaluation, images are resized bilinearly to 256 pixels, center-cropped to 224 × 224, and normalized using the same statistics.

Checkpoint selection and prediction. We evaluate every member on the unresampled validation split after each epoch and retain the checkpoint with the highest validation top-1 accuracy. At inference time, the selected checkpoint of each member produces a categorical distribution $p _ { \theta _ { m } } ( y \mid x )$ by applying softmax to its logits. The ensemble predictive distribution used throughout the imagedomain experiments is the arithmetic mean

$$
\bar { p } ( y \mid x ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } p _ { \theta _ { m } } ( y \mid x ) , \qquad M = 2 0 .\tag{16}
$$

Thus, the ensemble members differ through their random initialization, bootstrap training multiset, data ordering, and stochastic data augmentation.

Inpainting 2 Inpainting 3 100% mask

## K.5 QUALITATIVE EXAMPLES

Figure 3 shows ImageNet source img\_00001 with its 10%, 50%, and 100% bbox masks and all five completions from the current simplified-prompt run. Figure 4 shows ImageNet-R source imagenet\_r\_00033 from the current OOD run, with all three ID reference mask levels and all five evaluated completions at each level. Draw order is preserved, including repeated interpretations; no completion is selected based on visual quality or classifier output.

![](images/5685b34e91e04d25218b4bccbe84eee395caefeec4c647310c4f0f7eccec1c2a.jpg)  
Source

![](images/ec2525cb7337835f028ca790e0b210f6ea94923e595cac708c93d6fa70e29bb9.jpg)  
Masked input

![](images/32cb3c5341526e1c2ab9bb02e2b708bdbabf9230d5b692a499ac0d3bc6a89335.jpg)  
Inpainting 1

10% mask  
![](images/d1cc09316e75db4b0c74878f929e356ab2085343043cd8a937c22203703cc928.jpg)  
Inpainting 2

![](images/99e32d892017a5c31315f5c106a82394b019e34f35cf6b1bb30659113c6e0406.jpg)  
Inpainting 3

![](images/7a0d77edee2de2b575f092201f0fc7de2d387524367401230bd8e001406bf676.jpg)  
Inpainting 4

![](images/fb582bd19e4cffe246fb7406569171681692933ee5c872c0a1efe1dc6f7141eb.jpg)

![](images/f052d8dd59f7496c3968c8770977693ba1d968d551089564e64b84859a52007e.jpg)  
Inpainting 5  
Masked input  
50%

![](images/f01f38b026e861c802ebd464fef01d3374fea70a59a89d3f6ef10ad685a9e97e.jpg)  
Inpainting 1

![](images/d84a4d0d287d7044bc97e19867521f5622d7ce31ccedf407ec3a01bb5ac8cf9d.jpg)  
mask

![](images/b78b43292bfcdbb801737f5778ed032112894da7db7f5ea6ef3f4401e134ece7.jpg)

![](images/ccc8e839f3f79252517200553096ece9a046fe35509a04be5dcf0a849680d29d.jpg)  
Inpainting 4

![](images/8e1741295718e8f2c13e87d93ece9bd4d67b29f685d5586dc82e63a71f84300e.jpg)  
Inpainting 5

![](images/631fdee38cf287ae17bfed4847feb52d64a6a074551aaa4f8964145394a6244f.jpg)  
Masked input

![](images/c8a2bf00178124a6ad78547ea998df4cadbbd2815e73e552ffb31095dc012273.jpg)  
Inpainting 1

![](images/bd93c75ce012d5d163ecf55f701bb484560863ac75584899a577f417071f1755.jpg)  
Inpainting 2

![](images/1068db4e139dbff548fd69b9bf1e32b0845937078bf6b10795fee6569628e8a0.jpg)  
Inpainting 3

![](images/20f2d3265883c387c0f14799eb96ce35fd7a49732e405576110b6f71a30c25f2.jpg)  
Inpainting 4

![](images/b46216e692f72f4ad24c18a07b807d5ffe630cd92e57d9ef5d343c5c6fec0bf8.jpg)  
Inpainting 5

Figure 3: ImageNet in-distribution example img\_00001 (drilling platform) from the current simplified-prompt bbox run. The source image is followed by its 10%, 50%, and 100% bboxmasked observations and all five evaluated Luna-guided FLUX.2-klein-base-9B inpaintings at each level, in saved draw order.

![](images/f0e20547cb7b61361beeb1beb59d1dff1b3aa319f9f51a07630316f1a4f9ee98.jpg)  
Masked input

![](images/241d96c0e4a13e3ac8b4e876fcd7e5bc9a27b7cd386a3e8f5dcde9b227942010.jpg)

## 10% ID reference mask

![](images/a5036acd11a254947d6e85f7fd16e2c4ab1e742be448eaf2402c0f1f79201076.jpg)  
Source  
Inpainting 1

![](images/30b7640507f0a0dae4796d666e5dc4ce94ad637aacbfccfeda80c3330aa3a979.jpg)  
Inpainting 2

![](images/427ccb39f6606f72b041f767da6319eef629efca666ceb4c88c354d401527283.jpg)  
Inpainting 3

![](images/81723a892f891982189cbeb72661ceb24e875e0d0683eaa4f804a2eec8aee5b9.jpg)  
Inpainting 4

![](images/2c2375bd13838ce743713f475179e2f2008edd3b684dca2c09e9f3ce08a210df.jpg)  
Inpainting 5

![](images/08cf16313e6a701ad6d9c1286bcc8c06f44131d2d6cdaaee9e68b28c49925a5d.jpg)  
Masked input

![](images/32eadd785e05a5a5b392c1c970a8384b05b69604796d55f662fe48f8ad43e06a.jpg)  
Inpainting 1  
50% ID reference mask

![](images/de1da24fed5f3474a914ddde353c968f0ad6ccac420b6cbc33eaf001ec4aefe5.jpg)  
Inpainting 2

![](images/0cbe824497b3db73bed1c41c86d99f352dc29b850b8a86209754e71459e28dde.jpg)  
Inpainting 3

![](images/5b0adf09028bbf7d96c940208becb4eba8b1b2bfa6ff67ebc493e1d39f3a66c0.jpg)  
Inpainting 4

![](images/39731a2e66b26bd8918140d24835d3481bf78e54a9d75182eaeda062755f7c8c.jpg)  
Inpainting 5

![](images/1e5efedfabe72c8abde549566321328e8febcf7bce988514bad9c61a1cdda3e5.jpg)  
Masked input

![](images/515c597edee4cf223f4ef468fbc062b42a16a9f4044613143a71934eaf248948.jpg)  
Inpainting 1  
100% ID reference mask

![](images/c859823b89df2c3aba988b38956091d07686f86d12d58cb47b04ac6d5528b2d0.jpg)  
Inpainting 2

![](images/2636201473503705c09d3861b8e8203eed47dd633ef47fc4418f341463d2669f.jpg)  
Inpainting 3

![](images/2ab121efcf6b1759c895b4d0eb4af470ab125ce45d2d9f281cabd7b5a2740f9c.jpg)  
Inpainting 4

![](images/221af3de7ce3d9d54db0decf9408d21754417f05426bd924424d1ac87f6ebcf9.jpg)  
Inpainting 5

Figure 4: ImageNet-R out-of-distribution example imagenet\_r\_00033 (wood rabbit) from the current simplified-prompt run and the OOD cohort in Table 9. The source image is followed by masked observations at the 10%, 50%, and 100% ID reference levels and all five evaluated Lunaguided FLUX.2-klein-base-9B inpaintings at each level, in saved draw order. OOD square-mask areas match the mean ID mask areas at these reference levels.

## L AUDIO CLASSIFICATION EXPERIMENTS

The audio-classification experiment evaluates the revised SEA decomposition on composer recognition from underspecified piano recordings. We use 100 held-out ten-second excerpts from MAESTRO v3.0.0. Each excerpt has three nested observations with 10%, 30%, and 50% contiguous temporal masks and five independently prompted inpaintings per observation, yielding 300 masked observations and 1,500 AudioLDM completions. A 20-member composer-classification ensemble returns a ten-class categorical distribution for every completion. These predictions supply the underspecification, in-distribution error-detection results and MusicNet OOD evaluation in Table 10.

## L.1 DATASET

The evaluation cohort contains 100 performances excluded from classifier training and validation. We draw a single ten-second stereo excerpt from each performance and retain the native sampling rate of 44.1 kHz. Composer labels are mapped to nine named classes—Franz Liszt, Johann Sebastian Bach, Frédéric Chopin, Sergei Rachmaninoff, Ludwig van Beethoven, Franz Schubert, Claude Debussy, Alexander Scriabin, and Robert Schumann—plus an Other class.

We model underspecification by replacing a contiguous 10%, 30%, or 50% interval with exact digital silence. Given only the audio before and after the unavailable interval, Qwen2.5-Omni-7B generates five analytical descriptions of plausible missing content. Each description conditions one AudioLDM-m-full completion, and only samples inside the mask are replaced. The hidden reference audio, composer label, and metadata are unavailable to both models.

## L.2 EVALUATION PROTOCOL AND RESULTS

We estimate the SEA terms from the 5 × 20 inpainting-by-ensemble-member predictive distributions using Shannon entropy, the revised decomposition in Eq. 4, and the Monte-Carlo estimators in Appendix D. We consider two in-distribution tasks designed to test the specificity of different uncertainty components.

First, we test specification uncertainty by using each component as a score for distinguishing paired observations with different mask-region sizes: 10% versus 30% and 30% versus 50% of the tensecond waveform. These percentages denote the duration of the masked temporal region, and the larger mask is the positive class in each comparison. Second, we test aleatoric uncertainty through error prediction separately at each mask size. An error occurs when the top-1 class of the mean ensemble distribution on the observed masked excerpt differs from its composer label. Although the SEA scores are estimated from five inpaintings, all four components are evaluated against this same observed-audio error target. We report AUROC for both tasks.

Table 10: Audio composer-classification uncertainty AUROC. The first five columns reproduce the 100-excerpt MAESTRO in-distribution evaluation: percentages denote the size of the masked temporal region (10%, 30%, and 50% correspond to 1, 3, and 5 seconds), the larger mask is positive for underspecification detection, and ID errors are top-1 errors of the ensemble prediction on the observed masked audio. The final three columns add the composition-averaged MusicNet-versus-MAESTRO OOD evaluation, with MusicNet as the positive class. MusicNet cells report AUROC with 95% bootstrap confidence intervals from 2,000 draws; repeated MusicNet recordings are averaged within each composition before scoring. Best and second-best SEA components in each column are shown in bold and underlined, respectively.

<table><tr><td></td><td colspan="2">Underspecification</td><td colspan="3">MAESTRO ID error</td><td colspan="3">MusicNet OOD (combined)</td></tr><tr><td>Method</td><td>10% vs 30%</td><td>30% vs 50%</td><td>10%</td><td>30%</td><td>50%</td><td>10%</td><td>30%</td><td>50%</td></tr><tr><td>SEAalea</td><td>0.5994</td><td>0.5151</td><td>0.7545</td><td>0.7090</td><td>0.6570</td><td>0.4992</td><td>0.4779</td><td>0.3986</td></tr><tr><td>SEAepi</td><td>0.3652</td><td>0.4228</td><td>0.6530</td><td>0.6209</td><td>0.6159</td><td>0.7147</td><td>0.7839</td><td>0.8040</td></tr><tr><td>SEAspec</td><td>0.9703</td><td>0.8934</td><td>0.6875</td><td>0.5515</td><td>0.5670</td><td>0.6198</td><td>0.5066</td><td>0.5402</td></tr><tr><td>SEAtotal</td><td>0.6713</td><td>0.6666</td><td>0.7232</td><td>0.6806</td><td>0.6445</td><td>0.5808</td><td>0.5498</td><td>0.5046</td></tr></table>

Specification uncertainty is the strongest underspecification detector in both comparisons. It reaches AUROC 0.9703 for 10% versus 30% masking and 0.8934 for 30% versus 50% masking, compared with 0.6713 and 0.6666 for the next-best total uncertainty. Conversely, residual aleatoric uncertainty is the strongest detector of in-distribution classification errors at every mask size, with AUROC 0.7545, 0.7090, and 0.6570 at 10%, 30%, and 50% masking, respectively. Total uncertainty ranks second in all three cases. The number of observed-audio top-1 errors increases from 44/100 at 10% masking to 56/100 at 30% and 74/100 at 50%, confirming that progressively larger missing intervals make composer classification harder.

## L.3 MASKING AND INTERPRETATION-GUIDED INPAINTING

Nested temporal masks. For each excerpt, one legal mask center is sampled with seed 20260831. We replace the centered 1-, 3-, and 5-second intervals with exact digital silence, corresponding to 10%, 30%, and 50% of the waveform. All three masks share their center, so the 10% interval is contained in the 30% interval and the 30% interval is contained in the 50% interval. Samples outside the selected interval are unchanged. This construction changes only how much temporal information is withheld while keeping the source performance and surrounding evidence paired across mask sizes.

Interpretation distribution. Qwen2.5-Omni-7B receives two audio inputs: the complete visible waveform before the silent interval and the complete visible waveform after it. It never receives the hidden reference interval, the full masked waveform, the composer label, or other metadata. For each excerpt and mask size, the model samples five distinct positive captions intended for text-conditioned audio generation. Text sampling uses temperature 0.9, top-p 0.9, at most 48 new tokens, and base seed 202609160. Candidate captions are normalized, exact text duplicates are removed, and candidates that explicitly mention a mask, gap, missing or unavailable audio, or uncertainty are rejected. The retained five captions are equally weighted plausible specifications; they are not assertions about the true hidden passage.

Exact Qwen prompt. The following is the complete textual content supplied to Qwen; the two bracketed lines mark the positions occupied by the actual before- and after-audio tensors. The placeholder gap\_seconds is instantiated as 1.00, 3.00, or 5.00 for the respective mask.

SYSTEM   
You are an expert audio analyst who writes concise English captions for   
text-conditioned audio generation models.   
USER   
Audio immediately before the unavailable interval:   
[AUDIO: the complete visible waveform before the interval]   
Audio immediately after the unavailable interval:   
[AUDIO: the complete visible waveform after the interval]   
Two recordings come from immediately before and immediately after a   
missing   
{gap\_seconds:.2f}-second interval in the same continuous recording.   
Infer ONE plausible way the   
sound could continue through that interval.   
Return exactly one short positive caption suitable as a text prompt for   
an audio inpainting model.   
Describe the complete acoustic scene and the plausible sound in the   
missing interval. Mention   
audible sound sources or instruments, style, tempo, and continuity only   
when supported by the   
recordings. Do not mention a mask, gap, missing audio, uncertainty,   
alternatives, or these   
instructions. Do not add a preface or explanation.

AudioLDM inpainting. Each retained Qwen caption conditions one AudioLDM-m-full completion. We use 50 DDIM steps, guidance scale 2.5, 0.25 s of visible context padding on both sides when available, and a 0.01 s boundary fade. AudioLDM generates a candidate passage for the masked interval; only that exact interval is inserted into the original waveform, and samples outside it remain bitwise identical to the masked observation. A harmonic-audio quality gate permits up to 12 deterministic retries and retains the accepted seed and diagnostics. The hidden reference interval is never used for generation or selection. Across 100 excerpts, three mask sizes, and five captions, this gives $1 0 0 \times 3 \times 5 = 1 { , } 5 0 0$ completions.

## L.4 COMPOSER ENSEMBLE

Representation and prediction head. We extract frozen MERT-v1-95M representations from two five-second chunks of every ten-second signal. Hidden states from all 13 levels are mean-pooled over time and across the two chunks. Each classifier learns a softmax weighting over the 13 levels, applies layer normalization, and uses a 768–256–10 MLP with GELU activation and dropout 0.2. Softmax of the final logits gives $p _ { z , m } ( y )$ for inpainting z and member m.

Training protocol. The ensemble contains M = 20 independently initialized heads. Training uses 1,058 MAESTRO performances and validation uses a disjoint 118 performances, with ten clips per performance; all 100 evaluation performances are excluded. For each member, 1,058 training performances are sampled with replacement as a fixed performance-level bootstrap multiset, yielding 656–689 unique performances. We optimize class-weighted cross entropy with AdamW, learning rate 10<sup>−3</sup>, weight decay 10<sup>−4</sup>, batch size 64, at most 200 epochs, and early stopping after 25 epochs without improvement in validation balanced accuracy. The clean held-out ensemble reaches 56% top-1 and 94% top-5 accuracy.

## L.5 QUALITATIVE SPECTROGRAM EXAMPLE AND COMPLETE PROMPTS

Figures 5–7 show the first evaluation excerpt, music\_00000, with 10%, 30%, and 50% masks. The orange rectangle in masked and inpainted panels marks the interval replaced by silence or generated audio. Every panel spans the complete ten-second waveform, so the visible context can be compared directly. The example was fixed by sample order rather than selected by visual appearance or classifier output.

Complete 10% AudioLDM prompts. The captions below are reproduced verbatim and correspond in order to Inpaintings 1–5 in Figure 5.

1. The audio features a slow-paced piano melody in a classical style.   
2. The piece is a soft, slow piano solo.   
3. The recording features a serene piano solo with a consistent melody   
played in a measured tempo, creating a tranquil atmosphere.   
4. A piano melody transitions smoothly into a solo clarinet, maintaining   
a classical style and moderate tempo.   
5. The piano continues with a slow, melodic phrase.

Complete 30% AudioLDM prompts. The captions below are reproduced verbatim and correspond in order to Inpaintings 1–5 in Figure 6.

1. The recording features a slow, melodic piano piece in a minor key,   
with the absence suggesting a continuation of the same composition,   
potentially with a brief, poignant musical pause or dramatic emphasis.   
2. A slow, melodic piano piece in D# minor continues, maintaining a   
somber atmosphere with a steady tempo.   
3. The audio features a sustained, harmonious piano piece in F# major,   
with a slow tempo and a lush, classical style. It continues smoothly   
with no noticeable changes or interruptions.

![](images/0fd9e45cee1706348b4be07660c384e82b8a88d29d953b12e896bdfd2cf0dabb.jpg)  
Source

![](images/5b92e143d90a73f95c35296e59eae7cdaf84d21edbc20d77fe12010173570159.jpg)  
Masked input

![](images/f078a2df53191c02cffa1e74b3b90524054a6e96b610c483bc8690ef040efaa6.jpg)

![](images/c7ca5e8ef3b16079ee4d933d8e86b8a58bb917289992a20c15ccbf3f9fd95ba4.jpg)  
Inpainting 3

![](images/f7e7e6e6a84c1982c6541bb24bebf36ceb3ad6728591b3bcb3bcdd90dc4e94ad.jpg)

Inpainting 2  
![](images/2b374bd23f70ef30ad40c9354d7257dc6127b7a1452ca5230a6cc8a7f17a83d0.jpg)

Inpainting 4  
![](images/61be4a078b5c105e59db5f22abd17a9ba609aa188b86da82b7b10041462ba706.jpg)  
Inpainting 5  
Figure 5: MAESTRO example music\_00000 with a 10% mask from 2.810 to 3.810 s and all five analytical-prompt AudioLDM completions. The source is shown only for qualitative comparison and was unavailable to Qwen and AudioLDM. Outside the orange interval, every completion is identical to the observed input.

![](images/d750a1816c2e4f906c64608bc91fc7f609d71ca6b50a72908a4930bd93192868.jpg)  
Source

![](images/1db30e56969a8c9aa26c6818140f84357a5606e2148e28757e40e1718fbc3121.jpg)

![](images/d6081847cd97f42b2ec9bf812da029a9976b7ae9f3dc0c7a4eaa3f4f472601b6.jpg)

![](images/9d1926a1d15d593169c0021efecdaf19b5033099be939a3b385c51896d3fc9a1.jpg)  
Inpainting 3

![](images/caea4eb17686715755a55f0cbfbe811ebc37ea912d953dda97a2f215ca51dc32.jpg)

![](images/b73146e0cfbc8093ad96ea6f7b1c30a97f3678d70339ef96ae1dafdf8722b1a9.jpg)

Inpainting 4  
![](images/609c4036e90c08abf7a5a2fd2bb094dced2a0bc12568153a821637cbdb4b1b2a.jpg)  
Inpainting 5  
Figure 6: The same MAESTRO example with a 30% mask from 1.810 to 4.810 s and all five analytical-prompt AudioLDM completions. The source is shown only for qualitative comparison and was unavailable to Qwen and AudioLDM. Outside the orange interval, every completion is identical to the observed input.

4. A classical piano piece continues with a gentle melody, maintaining the same tempo and style.

5. A gentle piano melody continues, seamlessly connecting the before and after segments.

![](images/7efcf16ac5a9752e8167e7ef0e2d344d0f8be616d8e16828796b0195a5660f50.jpg)

![](images/0819f91e0ece7bc7e35a132841033bb80c148bbc7ee29b886b0ef2c9ddc60f4a.jpg)

![](images/3ae7afa5e983c61a0020a61959a6ba4bbe169cc354b8e12b48ba492d3d04a5e2.jpg)

![](images/b26d20bc3b50486cc5e0c81dcddda97c1d78ae14a1a98abd2e28f54ecf3b32b5.jpg)

![](images/03186a34ed2aebdbfc9da9a8843b34b74511815e62973ef2dbf7d2454b3bb19d.jpg)  
Inpainting 3

![](images/a33762ded6108825235496501e0092768a45a19acffa9213fbb8d593bc054940.jpg)

Inpainting 4  
![](images/1d0c7645cde39c7c224479052bc79a1290e27d038c6fc1393cfe4e2600fbf992.jpg)  
Inpainting 5  
Figure 7: The same MAESTRO example with a 50% mask from 0.810 to 5.810 s and all five analytical-prompt AudioLDM completions. The larger interval removes more musical evidence and admits a broader set of plausible continuations.

Complete 50% AudioLDM prompts. The captions below are reproduced verbatim and correspond in order to Inpaintings 1–5 in Figure 7.

1. The piano continues with a slow and melodic style, maintaining the same tempo and emotional tone.

2. The recording features a melodic piano piece with a moderate tempo and a rich, harmonious texture.

3. A melodic piano tune continues smoothly, maintaining a serene and harmonious atmosphere.

4. The piano playing softly continues with a melodic and harmonious progression, maintaining the soothing style at a steady tempo.

5. A melancholic piano melody continues gracefully, maintaining a slow tempo and evoking a somber mood.

## M VISUAL QUESTION ANSWERING EXPERIMENTS

Specification uncertainty in visual question answering (VQA) is introduced by progressively concealing parts of an input image while keeping the question and reference answer fixed. Mask area is a controlled proxy for missing visual information rather than a human annotation of unanswerability: a random square may leave the answer-relevant region visible, so the two levels differ in expected, not guaranteed, underspecification. The image generation model produces five inpainted images serving as clarifications for each masked observation. The predictor is a uniformly weighted ensemble of four pretrained vision-language models used without task-specific fine-tuning. Every member answers the clean image and the masked observation, and each member rescores a shared pool of candidate answers, so that uncertainty within a member, disagreement across members, and sensitivity to the missing content are measured on the same support.

## M.1 DATASETS

ImageNet. We reuse the ImageNet data, preprocessing, and inpainting pipeline from Appendix K. For each image, we assign the fixed question “What is depicted in the picture?” and the ground-truth class name as the reference answer. Here we use masks targeting 10%, 20%, 30%, and 50% of the annotation bounding box. The answering models generate a class name freely, without a supplied list of categories. gpt-5.6-luna judges whether the generated answer matches the class name, which avoids penalizing the synonyms and paraphrases that open-ended generation produces. Questions and references stay fixed across the clean, masked, and restored versions of an image.

VQS. We use 500 image-question pairs from Visual Questions and Segmentation Answers (VQS) (Gan et al., 2017). Textual references are recovered by matching image identifiers and question text to VQA v1 annotations, preserving human answer votes. From 27,551 normalized validation records, 500 random images, retaining at most one question per image. The frozen subset contains 185 yes/no, 75 number, and 240 other questions. VQS segmentations are retained as metadata only and do not determine mask placement, so the experiment measures random occlusion rather than targeted removal of the answer region. Correctness is exact agreement with the most common human answer, accepting tied modes, after VQA-style normalization: lowercasing, whitespace and punctuation normalization, removal of the articles “a”, “an”, and “the”, and mapping of the number words zero through ten to digits.

## M.2 VQS MASKING AND DIRECT INPAINTING

Mask construction. We prepare 512 × 512 RGB images using bicubic resizing and a central square crop. For each image, we construct one square mask at each of two nominal whole-image area fractions, 10% and 30%. Since the mask position was sampled independently, it do not enforce shared centers or nested masks. The masked area is set to RGB (127, 127, 127). Conditions are paired by source image and question, giving 1,000 masked observations and 5,000 completions per dataset. Unlike the ImageNet bounding-box percentages, VQS percentages refer to the entire working image.

Inpainting. We generate VQS completions with FLUX.1-Fill-dev using FluxFillPipeline. It receives the masked image, binary mask, and the same fixed prompt for every example. There is no intermediate LLM: we do not generate interpretation distributions, sample textual scenarios, or produce image-specific captions. Neither the question nor its reference answer is supplied to the inpainting model. Generation uses 512 × 512 images, 50 inference steps, guidance scale 40.0, strength 1.0, bfloat16 weights, and no negative prompt. The exact prompt is:

Seamless plausible photorealistic continuation of the visible scene,   
matching nearby colors, textures, lighting, perspective and logically   
implied by the rest of the image. Surrounding object boundaries are   
extended seamlessly.

## M.3 ANSWERING MODELS AND UNCERTAINTY ESTIMATION

Ensemble and prompts. We use four pretrained multimodal checkpoints without task-specific fine-tuning: Qwen2.5-Omni-7B, Gemma-4-E4B-it, Phi-4-multimodal-instruct, and MiniCPM-o-4.5. Each member uses its native image processor, tokenizer, and conversation template. Qwen, Phi, and MiniCPM use bfloat16; Gemma uses float32 with eager attention. The ensemble weights members uniformly. All members receive the same task text inside their native templates. VQS uses the question followed by a short-answer instruction:

Question: {question}   
Answer with only the short answer.

ImageNet substitutes the fixed recognition question and a class-name instruction:

Question: What is depicted in the picture?   
Answer with a short object class name only.

Sampling and scoring. For every model and image version, we sample ten answers with temperature 1.0, top-p = 1, top-k filtering disabled, one beam, and a limit of 32 new tokens. We also generate a separate greedy answer for each member. For the restored condition, candidate strings are pooled across all four models and five completions; clean and masked conditions have separate supports. Every member scores the shared candidates using its own tokenizer, summing answer-token log probabilities without an appended end-of-sequence token.

We group answers using DeBERTa-large-MNLI predictions of bidirectional entailment for semantic estimators. We compute the total, aleatoric, epistemic, and specification terms from the resulting scores. Single-model comparisons retain only that member’s sampled support and repeat semantic clustering. Candidate-support estimates approximate uncertainty over generated answers rather than over the complete output space.

Evaluation. Underspecification detection treats the 30% mask as the positive class and the paired 10% mask as the negative class. Error detection treats an incorrect answer to the masked image as positive; individual members are scored on their greedy answers, and the ensemble is scored on the maximum-scoring candidate based on pooled sequence likelihoods.