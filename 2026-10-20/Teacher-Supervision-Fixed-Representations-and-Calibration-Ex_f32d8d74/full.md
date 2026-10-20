# Teacher Supervision, Fixed Representations and Calibration: Exact Risk Calculations for Candidate-Scoring Decision Models

Agus Sudjianto

Denis Burakov

October 2026

## Abstract

Candidate-scoring decision models return a probability distribution over a presented set of answers. Public documentation of one such system, Jev, and of an open counterpart, JevAny, describes three training ingredients that determine what these probabilities estimate: soft targets from larger models, a frozen pretrained backbone and temperature calibration. We analyze each ingredient in Gaussian surrogates where the finite-sample risk can be computed exactly. A student trained on teacher targets inherits the teacher’s bias, which more student data does not remove and agreement with the teacher does not reveal. A fixed representation lowers the risk only below an explicit sample-size threshold, and it passes on as bias only the part of the teacher’s error that lies in the retained directions. In a probit model the exact recalibration map is a temperature scaling. It removes all excess risk only when the teacher’s coeficient has the correct direction and otherwise leaves a residual, given in closed form, that calibration diagnostics do not detect. Under Luce’s choice axiom, candidate scores are coherent across refinements of an event if and only if they aggregate by log-sum-exp. Simulations agree with the formulas under Gaussian design and show that they understate the risk under heavy-tailed design at small samples. On 231,521 credit card complaints from the Consumer Financial Protection Bureau, where the decision is whether a complaint ends in monetary relief, a student trained on a logistic teacher’s probabilities ends at the teacher’s decision cost while a student trained on outcomes goes 8% below it, recalibration of a 30B model read zero-shot corrects its calibration and leaves it worse than the teacher, Turing’s calibration check passes on a constant forecast and on a model that ranks at chance, and the 30B model departs from the coherence condition by 17.5 decibans. The probabilities of a candidate-scoring model estimate what its training targets estimate. A pretrained representation can reduce the amount of labeled data required and calibration can correct the scale of the probabilities, but neither corrects a biased teacher, so the probabilities have to be checked against realized outcomes.

Keywords: knowledge distillation; soft labels; random-design regression; inverse Wishart distribution; probit model; probability calibration; temperature scaling; Brier score; choice axiom; language-model classifiers; consumer complaints; credit decisions.

## 1 Introduction

A candidate-scoring decision model receives a state, a question and a set of candidate answers, and it returns a probability for each candidate. When the candidates are a fixed set of class labels the model is a classifier with a text interface. The probability it returns estimates a conditional distribution, and the training targets determine which distribution that is. The probability that an event occurs, the probability that an annotator selects a candidate and the probability that a reference model assigns to a candidate are diferent conditional distributions, and a model can estimate one of them well while estimating the others poorly.

Jev is a candidate-scoring model released by TypeSafe [1]. The announcement describes it as a model for structured decisions that returns all outputs in a single query, trained with a method called reinforcement learning for calibrated decisions; for high-cardinality choices it scores candidates independently and then makes an explicit choice. The announcement also states that its workflow evaluations use the predictions of large external models as reference probabilities. The training objective itself is not specified in detail. The JevAny documentation [14] describes an openly documented system of the same type. Its released models keep the backbone weights frozen and train low-rank (LoRA, rank 8) adapters together with a scoring head; training uses either a hard target or a normalized soft target distribution over the candidates; and a temperature fitted on a separate calibration set is applied to the candidate scores. Jev also appears as one of the AI judges in the adapted imitation game of Burakov [4, Chap. 1].

These descriptions identify three mechanisms that bear on the meaning of the output probabilities. Soft targets and reference probabilities make the output of a teacher the quantity that the student learns to reproduce [12]. A frozen backbone restricts the student to functions of a fixed representation. Temperature calibration changes the scale of the scores without changing their ranking [11]. Because the full Jev objective is not public, we do not model Jev. We analyze each mechanism in a Gaussian surrogate in which the finite-sample risk and the calibration curve can be computed exactly, and we state which conclusions depend on the Gaussian assumptions.

Section 3 treats least squares with Gaussian random design. The outcome-supervised risk is classical [3]; the teacher-target and mixed-supervision risks follow from the same inverse-Wishart calculation and are included for completeness. Section 4 gives the exact risk of a student restricted to a fixed linear representation and trained on a mixture of outcomes and teacher targets, in the case where the teacher depends on directions that the representation discards and the two target errors are correlated (Theorem 4.3). From it we obtain the optimal teacher weight and a projection identity for the teacher bias that the compressed student inherits. Section 5 turns to a binary outcome under a Gaussian-probit model. It gives the calibration curve of a teacher with an arbitrary coeficient distortion (Theorem 5.3), shows that the exact recalibration map is a temperature scaling and gives the residual Brier risk after recalibration in closed form (Corollary 5.4). Section 6 states the coherence condition for candidate scores, Section 7 reports Monte Carlo checks, including designs that violate the Gaussian assumption, and Section 8 tests the predictions on credit card complaints, where the evidence is text and the true probability is unknown.

## 2 Estimands

Let $X \in \mathbb { R } ^ { d }$ denote the observed predictors, Y a realized outcome and T the output of a teacher. For a binary outcome the target of an outcome-supervised predictor is $\eta ( x ) = \operatorname* { P r } ( Y = 1 \mid X = x )$ whereas a student trained on teacher outputs estimates $t ( x ) = \mathbb { E } [ T \mid X = x ]$ . A strictly proper scoring rule is minimized in expectation by the conditional distribution of the training target [9], so a low proper score against teacher targets establishes that the student reproduces t; it does not establish that $t = \eta$

A candidate-choice model may estimate $\operatorname* { P r } ( A = a \mid X = x , Q = q , { \mathcal { C } } )$ , where A is the candidate selected by an annotator, a reference model or a policy, Q is the question and C is the presented candidate set. This probability equals $\operatorname* { P r } ( Y = a \mid X = x )$ only under a stated relationship between A and Y . A related distinction applies to evidential readings of model scores. Good [10] describes Turing’s weight of evidence as the logarithm of a Bayes factor, a ratio of likelihoods under two specified hypotheses. A logit or a log-ratio of softmax outputs has this interpretation only when its numerator and denominator are such likelihoods.

The exact calculations use two surrogates: linear regression with squared error (Sections 3 and 4) and a binary probit model (Section 5). The representation is a fixed linear map chosen before the labeled sample is drawn. This corresponds to a frozen backbone with a linear head. Adapting the backbone, for example through LoRA adapters, changes the representation with the labels and lies outside the exact results.

## 3 Least squares with Gaussian random design

## 3.1 Outcome supervision

Let $( X _ { i } , Y _ { i } ) , i = 1 , \dots , n$ , be independent with

$$
X _ { i } \sim N _ { d } ( 0 , \Sigma ) , \qquad Y _ { i } = X _ { i } ^ { \top } \beta + \varepsilon _ { i } , \qquad \varepsilon _ { i } \sim N ( 0 , \sigma ^ { 2 } ) ,\tag{1}
$$

where $\Sigma \succ 0$ and the errors are independent of the design and of each other. Let X be the $n \times d$ design matrix and $X _ { 0 }$ an independent test input from the same distribution. For a coeficient estimator $\widehat { \theta }$ the excess prediction risk is

$$
\mathcal { E } ( \widehat { \theta } ) = \mathbb { E } \big [ ( X _ { 0 } ^ { \top } \widehat { \theta } - X _ { 0 } ^ { \top } \beta ) ^ { 2 } \big ] = \mathbb { E } \big [ ( \widehat { \theta } - \beta ) ^ { \top } \Sigma ( \widehat { \theta } - \beta ) \big ] ,\tag{2}
$$

and the mean squared error for a fresh outcome is $\sigma ^ { 2 } + \mathcal { E } ( \widehat { \theta } )$

Lemma 3.1 (Outcome-supervised least squares [3]). $I f n > d + 1$ , the estimator ${ \widehat { \beta } } _ { O } = ( \mathbf { X } ^ { \top } \mathbf { X } ) ^ { - 1 } \mathbf { X } ^ { \top } \mathbf { Y }$ satisfies

$$
\mathcal { E } ( \widehat { \beta } _ { O } ) = \frac { \sigma ^ { 2 } d } { n - d - 1 } .\tag{3}
$$

Proof. Conditionally on X, $\widehat { \beta } _ { O } - \beta \sim N ( 0 , \sigma ^ { 2 } ( \mathbf { X } ^ { \top } \mathbf { X } ) ^ { - 1 } )$ . Since $\mathbf { X } ^ { \top } \mathbf { X } \sim W _ { d } ( \Sigma , n )$ , its inverse has mean $\Sigma ^ { - 1 } / ( n - d - 1 )$ for $n > d + 1 \ [ 2 , 1 8 ]$ . Taking the expectation of $\sigma ^ { 2 } \operatorname { t r } \{ \Sigma ( \mathbf { X } ^ { \top } \mathbf { X } ) ^ { - 1 } \}$ gives (3). □

## 3.2 Teacher supervision

Suppose a teacher supplies $T _ { i } = X _ { i } ^ { \top } ( \beta + \delta ) + \xi _ { i }$ , where $\xi _ { i } \sim N ( 0 , \tau ^ { 2 } )$ is independent of the design. The vector δ is the systematic teacher error, and the student fits least squares to $T$ in place of $Y$

Proposition 3.2 (Teacher-target risk). For $n > d + 1$ the estimator ${ \widehat { \beta } } _ { T } = ( \mathbf { X } ^ { \top } \mathbf { X } ) ^ { - 1 } \mathbf { X } ^ { \top } \mathbf { T }$ satisfies

$$
\mathcal { E } ( \widehat { \beta } _ { T } ) = \delta ^ { \top } \Sigma \delta + \frac { \tau ^ { 2 } d } { n - d - 1 } .\tag{4}
$$

Hence $\mathcal { E } ( \widehat { \beta } _ { T } ) < \mathcal { E } ( \widehat { \beta } _ { O } )$ if and only if

$$
\delta ^ { \top } \Sigma \delta < \frac { d ( \sigma ^ { 2 } - \tau ^ { 2 } ) } { n - d - 1 } .\tag{5}
$$

Proof. The coeficient error is $\widehat { \beta } _ { T } - \beta = \delta + ( \mathbf { X } ^ { \top } \mathbf { X } ) ^ { - 1 } \mathbf { X } ^ { \top } \pmb \xi$ . The second term has conditional mean bzero, so the cross term vanishes, and its Σ-weighted second moment follows from the calculation in Lemma 3.1. □

A fixed $\delta \neq 0$ leaves the error floor $\delta ^ { \top } \Sigma \delta$ as $n \to \infty$ , whereas $\mathcal { E } ( \widehat { \beta } _ { O } )  0$ for fixed d. A teacher with $\tau ^ { 2 } < \sigma ^ { 2 }$ btherefore gives lower risk than outcome supervision only for n below the crossover $d + 1 + d ( \sigma ^ { 2 } - \tau ^ { 2 } ) / \delta ^ { \top } \Sigma \delta$

## 3.3 Mixed supervision

Let ${ \widehat { \beta } } _ { \lambda }$ minimize $( 1 - \lambda ) \| \mathbf { Y } - \mathbf { X } b \| ^ { 2 } + \lambda \| \mathbf { T } - \mathbf { X } b \| ^ { 2 }$ for $\lambda \in [ 0 , 1 ]$ . Allow the pairs $( \varepsilon _ { i } , \xi _ { i } )$ to be jointly bGaussian with variances $\sigma ^ { 2 } , \tau ^ { 2 }$ and covariance $\omega .$ , independent across i and of the design. Put $B = \delta ^ { \top } \Sigma \delta$ and $C = d / ( n - d - 1 )$

Proposition 3.3 (Mixed supervision with correlated errors). For $n > d + 1$

$$
\mathcal { E } ( \widehat { \beta } _ { \lambda } ) = \lambda ^ { 2 } B + C \big [ ( 1 - \lambda ) ^ { 2 } \sigma ^ { 2 } + \lambda ^ { 2 } \tau ^ { 2 } + 2 \lambda ( 1 - \lambda ) \omega \big ] .\tag{6}
$$

If $B + C ( \sigma ^ { 2 } + \tau ^ { 2 } - 2 \omega ) > 0$ , the unconstrained minimizer is

$$
\lambda _ { \mathrm { u n c } } = \frac { C ( \sigma ^ { 2 } - \omega ) } { B + C ( \sigma ^ { 2 } + \tau ^ { 2 } - 2 \omega ) } ,\tag{7}
$$

and the minimizer over [0, 1] is $\lambda ^ { * } = \operatorname* { m i n } \{ 1 , \operatorname* { m a x } \{ 0 , \lambda _ { \mathrm { u n c } } \} \}$ . In particular $\lambda _ { \mathrm { u n c } } < 0$ if and only if $\omega > \sigma ^ { 2 }$ , and $\lambda _ { \mathrm { u n c } } > 1$ if and only $i f \omega > \tau ^ { 2 } + B / C$ . For $\omega = 0$ and $\sigma ^ { 2 } > 0 , \lambda ^ { * } = C \sigma ^ { 2 } / \{ B + C ( \sigma ^ { 2 } +$ $\tau ^ { 2 } ) \} \in ( 0 , 1 ]$

Proof. The objective is least squares on the target $( 1 - \lambda ) \mathbf { Y } + \lambda \mathbf { T }$ , so the coeficient error is $\lambda \delta + ( \mathbf { X } ^ { \top } \mathbf { X } ) ^ { - 1 } \mathbf { X } ^ { \top } \{ ( 1 - \lambda ) \varepsilon + \lambda \xi \}$ . The noise term has conditional mean zero and conditional covariance $\{ ( 1 - \lambda ) ^ { 2 } \sigma ^ { 2 } + \lambda ^ { 2 } \tau ^ { 2 } + 2 \lambda ( 1 - \lambda ) \omega \} ( \mathbf { X } ^ { \top } \mathbf { X } ) ^ { - 1 }$ , which gives (6). The risk is a quadratic in λ with leading coeficient $B + C \operatorname { V a r } ( \varepsilon - \xi ) \geq 0 ;$ setting the derivative to zero gives (7), and the projection onto [0, 1] minimizes a convex quadratic over the interval. The two boundary conditions follow by comparing the numerator of (7) with zero and with the denominator. □

Since $\omega \leq \sigma \tau \leq \operatorname* { m a x } \{ \sigma ^ { 2 } , \tau ^ { 2 } \}$ , at most one of the two boundary conditions can hold. If the leading coeficient vanishes, then $B = 0$ and $\varepsilon = \xi$ almost surely, the linear coeficient $C ( \sigma ^ { 2 } - \omega )$ also vanishes and the risk does not depend on λ. For $B > 0$ the optimal weight tends to zero as $n \to \infty$ because $C  0$

## 4 Fixed representations

## 4.1 Outcome supervision through a representation

Let $A \in \mathbb { R } ^ { r \times d }$ have full row rank, $1 \leq r \leq d ,$ and let $Z = A X$ with covariance $C _ { A } = A \Sigma A ^ { \top }$ . The map A is fixed before the labeled sample is drawn. The best linear predictor of Y from Z has coeficient $w _ { * } = C _ { A } ^ { - 1 } A \Sigma \beta$ , and the part of the signal that Z does not capture has coeficient and variance

$$
\begin{array} { r } { \boldsymbol { u } _ { \beta } = \beta - A ^ { \top } \boldsymbol { w } _ { * } , \qquad \boldsymbol { B } _ { A } = \boldsymbol { u } _ { \beta } ^ { \top } \boldsymbol { \Sigma } \boldsymbol { u } _ { \beta } = \beta ^ { \top } \big ( \boldsymbol { \Sigma } - \boldsymbol { \Sigma } A ^ { \top } \boldsymbol { C } _ { A } ^ { - 1 } A \boldsymbol { \Sigma } \big ) \beta . } \end{array}\tag{8}
$$

We call $B _ { A }$ the representation loss.

Proposition 4.1 (Risk through a fixed representation). Let $\widehat { w } _ { O }$ be the least-squares coeficient of Y on Z with $n > r + 1$ . Its excess risk relative to $X ^ { \top } \beta$ is

$$
\mathcal { E } _ { A } = B _ { A } + \frac { r ( \sigma ^ { 2 } + B _ { A } ) } { n - r - 1 } .\tag{9}
$$

Proof. Since $\operatorname { C o v } ( Z , X ^ { \top } u _ { \beta } ) = A \Sigma u _ { \beta } = 0$ and $( Z , X ^ { \top } u _ { \beta } )$ is jointly Gaussian, $X ^ { \top } u _ { \beta }$ is independent of Z with variance $B _ { A }$ . Therefore $Y = Z ^ { \top } w _ { * } + e$ with $e = X ^ { \top } u _ { \beta } + \varepsilon$ independent of Z and $\mathrm { V a r } ( e ) = \sigma ^ { 2 } + B _ { A }$ The prediction error at a fresh point is $Z _ { 0 } ^ { \top } ( \widehat { w } _ { O } - w _ { * } ) - X _ { 0 } ^ { \top } u _ { \beta } ;$ the two terms are uncorrelated, the second has second moment $B _ { A }$ band the first has expected square $r ( \sigma ^ { 2 } + B _ { A } ) / ( n - r - 1 )$ by Lemma 3.1 applied in dimension r. □

Corollary 4.2 (Compression threshold). For $n > d + 1 , \mathcal { E } _ { A } < \mathcal { E } ( \widehat { \beta } _ { O } )$ if and only if

$$
B _ { A } < \frac { \sigma ^ { 2 } ( d - r ) } { n - d - 1 } .\tag{10}
$$

Proof. Write $\mathcal { E } _ { A } = \{ ( n - 1 ) B _ { A } + r \sigma ^ { 2 } \} / ( n - r - 1 )$ and compare it with $\sigma ^ { 2 } d / ( n - d - 1 )$ . The identity $d ( n - r - 1 ) - r ( n - d - 1 ) = ( d - r ) ( n - 1 )$ reduces the comparison to (10). □

The threshold decreases to zero as n grows, and it is zero for $r = d .$ . A representation with $B _ { A } > 0$ is therefore preferable to the full predictors only below the sample size $d + 1 + \sigma ^ { 2 } ( d - r ) / B _ { A }$

## 4.2 Mixed supervision through a representation

A teacher may depend on directions of X that the student’s representation discards. Let $\gamma = \beta + \delta$ and $T = X ^ { \top } \gamma + \xi$ , and define

$$
\begin{array} { r } { \boldsymbol { w } _ { \gamma } = C _ { A } ^ { - 1 } A \boldsymbol { \Sigma } \gamma , \qquad \boldsymbol { u } _ { \gamma } = \gamma - A ^ { \top } \boldsymbol { w } _ { \gamma } , \qquad \boldsymbol { D } _ { A } = ( \boldsymbol { w } _ { \gamma } - \boldsymbol { w } _ { \ast } ) ^ { \top } C _ { A } ( \boldsymbol { w } _ { \gamma } - \boldsymbol { w } _ { \ast } ) , } \end{array}\tag{11}
$$

together with $G _ { A } = u _ { \gamma } ^ { \top } \Sigma u _ { \gamma }$ and $S _ { A } = u _ { \beta } ^ { \top } \Sigma u _ { \gamma }$ . The error pairs $( \varepsilon _ { i } , \xi _ { i } )$ are as in Proposition 3.3. The student fits a linear head on $Z$ to the targets $V _ { \lambda } = ( 1 - \lambda ) Y + \lambda T$

Theorem 4.3 (Exact risk with representation and mixed supervision). For $n > r { + } 1$ the least-squares head ${ \widehat { w } } _ { \lambda } ~ o f ~ V _ { \lambda }$ on $Z$ satisfies

$$
\begin{array} { l } { { \displaystyle { \mathcal E } _ { A , \lambda } = B _ { A } + \lambda ^ { 2 } D _ { A } + \frac { r } { n - r - 1 } \Big [ ( 1 - \lambda ) ^ { 2 } ( B _ { A } + \sigma ^ { 2 } ) } } \\ { { \displaystyle ~ + \lambda ^ { 2 } ( G _ { A } + \tau ^ { 2 } ) + 2 \lambda ( 1 - \lambda ) ( S _ { A } + \omega ) \Big ] . } } \end{array}\tag{12}
$$

Proof. Let $w _ { \lambda } = ( 1 - \lambda ) w _ { \ast } + \lambda w _ { \gamma }$ . Then $V _ { \lambda } = Z ^ { \top } w _ { \lambda } + e _ { \lambda }$ with $\boldsymbol { e } _ { \lambda } = ( 1 - \lambda ) ( X ^ { \top } u _ { \beta } + \varepsilon ) + \lambda ( X ^ { \top } u _ { \gamma } + \xi )$ Both $X ^ { \top } u _ { \beta }$ and $X ^ { \top } u _ { \gamma }$ are uncorrelated with $Z$ , because $A \Sigma u _ { \beta } = A \Sigma u _ { \gamma } = 0$ , and the errors are independent of X. By joint Gaussianity $e _ { \lambda }$ is independent of $Z ,$ and its variance is the bracket in (12), since Cov $\cdot ( X ^ { \top } u _ { \beta } + \varepsilon , X ^ { \top } u _ { \gamma } + \xi ) = S _ { A } + \omega$ . The prediction error at a fresh point is

$$
Z _ { 0 } ^ { \top } ( \widehat { w } _ { \lambda } - w _ { \lambda } ) + \lambda Z _ { 0 } ^ { \top } ( w _ { \gamma } - w _ { * } ) - X _ { 0 } ^ { \top } u _ { \beta } .
$$

The last term is uncorrelated with $Z _ { 0 }$ and independent of the training sample, and the first term has mean zero given the training design. The three terms are therefore uncorrelated, with expected squares equal to the inverse-Wishart variance term, $\lambda ^ { 2 } D _ { A }$ and $B _ { A }$ □

Equation (12) is not the sum of a representation loss, a teacher error and an estimation variance that could each be computed separately. The term $S _ { A }$ couples the outcome and teacher residuals, and the unrepresented teacher signal $G _ { A }$ enters the variance term rather than the bias term.

Corollary 4.4 (Reduction to Proposition 3.3). If $u _ { \beta } = u _ { \gamma } = 0$ , then $B _ { A } = G _ { A } = S _ { A } = 0$ and $D _ { A } = \delta ^ { \top } \Sigma \delta$ , so (12) coincides with (6) with d replaced by r.

Corollary 4.5 (Projected teacher bias). With $\delta = \gamma - \beta$

$$
\begin{array} { r } { D _ { A } = \delta ^ { \top } \Sigma A ^ { \top } ( A \Sigma A ^ { \top } ) ^ { - 1 } A \Sigma \delta \leq \delta ^ { \top } \Sigma \delta . } \end{array}\tag{13}
$$

Equality holds if and only if $\Sigma ^ { 1 / 2 } \delta$ lies in the row space of $A \Sigma ^ { 1 / 2 }$ , and $D _ { A } = 0$ if and only if $A \Sigma \delta = 0$

Proof. Since $w _ { \gamma } - w _ { * } = C _ { A } ^ { - 1 } A \Sigma \delta$ , the first equality follows from the definition of $D _ { A }$ . The matrix $P = \Sigma ^ { 1 / 2 } A ^ { \top } ( A \Sigma A ^ { \top } ) ^ { \top } { } ^ { 1 } A \Sigma ^ { 1 / 2 }$ is the orthogonal projector onto the row space of $A \Sigma ^ { 1 / 2 }$ , and $D _ { A } = \| P \Sigma ^ { 1 / 2 } \delta \| ^ { 2 } \leq \| \Sigma ^ { 1 / 2 } \delta \| ^ { 2 }$ , with equality exactly when $\Sigma ^ { 1 / 2 } \delta$ lies in the range of $P .$ Finally $P \Sigma ^ { 1 / 2 } \delta = 0$ if and only if $( A \Sigma A ^ { \top } ) ^ { - 1 } A \Sigma \delta = 0$ , that is, $A \Sigma \delta = 0$ □

The projection removes the component of the teacher bias that the student cannot represent from the coeficient bias. That component remains in the training targets, however, and it contributes to $G _ { A }$ and $S _ { A }$ and hence to the variance of the fitted head. Compression thus converts part of the systematic teacher error into target noise.

Corollary 4.6 (Optimal teacher weight under compression). Let $C _ { r } = r / ( n - r - 1 ) , U = B _ { A } + \sigma ^ { 2 }$ $V = G _ { A } + \tau ^ { 2 }$ and $H = S _ { A } + \omega . \ I f \ D _ { A } + C _ { r } ( U + V - 2 H ) > 0$ , the unconstrained minimizer of (12) is

$$
\lambda _ { A , \mathrm { u n c } } = \frac { C _ { r } ( U - H ) } { D _ { A } + C _ { r } ( U + V - 2 H ) } ,\tag{14}
$$

and the minimizer over [0, 1] is its projection onto that interval. If the denominator is zero, the risk does not depend on λ.

Proof. Expanding (12) gives $B _ { A } + C _ { r } U - 2 C _ { r } ( U - H ) \lambda + \{ D _ { A } + C _ { r } ( U + V - 2 H ) \} \lambda ^ { 2 }$ . The leading coeficient is nonnegative because $U + V - 2 H = \mathrm { V a r } \{ ( X ^ { \top } u _ { \beta } + \varepsilon ) - ( X ^ { \top } u _ { \gamma } + \xi ) \}$ . If it vanishes, the two residuals are almost surely equal, so $U = V = H$ and the linear coeficient vanishes as well.

Compression changes $B _ { A } , D _ { A } , G _ { A } , S _ { A }$ and $C _ { r }$ together, so it has no fixed efect on the preferred teacher weight. In the example of Section 7 the optimal weights with and without compression are 0.4218 and 0.4221.

## 5 Binary outcomes under a Gaussian-probit model

## 5.1 Excess Brier risk and calibration of a distorted teacher

Let $X \sim N _ { d } ( 0 , \Sigma ) , \varepsilon \sim N ( 0 , \sigma ^ { 2 } )$ independent of X and $\begin{array} { r } { Y = \mathbf { 1 } \{ X ^ { \top } \beta + \varepsilon > 0 \} } \end{array}$ . Then

$$
\eta ( x ) = \operatorname* { P r } ( Y = 1 \mid X = x ) = \Phi ( { a } ^ { \top } x ) , \qquad { a } = \beta / \sigma .\tag{15}
$$

A teacher reports $t ( x ) = \Phi ( b ^ { \top } x )$ . In the population, minimizing cross-entropy against these soft targets returns t whenever the student’s function class contains it, and the excess Brier risk of t relative to η is $\mathbb { E } [ \{ t ( X ) - \eta ( X ) \} ^ { 2 } ]$ . Throughout we write $\boldsymbol { v } _ { a } = \boldsymbol { a } ^ { \top } \boldsymbol { \Sigma } \boldsymbol { a } , \boldsymbol { v } _ { b } = \boldsymbol { b } ^ { \top } \boldsymbol { \Sigma } \boldsymbol { b }$ and $c = a ^ { \top } \Sigma b$

Proposition 5.1 (Excess Brier risk of a probit teacher).

$$
\mathbb { E } [ \{ t ( X ) - \eta ( X ) \} ^ { 2 } ] = \frac { 1 } { 2 \pi } \left[ \arcsin \frac { v _ { a } } { 1 + v _ { a } } + \arcsin \frac { v _ { b } } { 1 + v _ { b } } - 2 \arcsin \frac { c } { \sqrt { ( 1 + v _ { a } ) ( 1 + v _ { b } ) } } \right] .\tag{16}
$$

Proof. Let (U, V) be centered jointly Gaussian with variances $s _ { U } ^ { 2 } , s _ { V } ^ { 2 }$ and covariance s , and let $E _ { 1 } , E _ { 2 }$ be independent standard normal variables independent of $( U , V )$ . Then $\mathbb { E } [ \Phi ( U ) \Phi ( V ) ] =$ $\operatorname* { P r } ( U - E _ { 1 } \ge 0 , V - E _ { 2 } \ge 0 )$ , and the bivariate orthant probability [22] gives $\mathbb { E } [ \Phi ( U ) \Phi ( V ) ] =$ ${ \frac { 1 } { 4 } } + ( 2 \pi ) ^ { - 1 }$ arcsin $\{ s _ { U V } / \sqrt { ( 1 + s _ { U } ^ { 2 } ) ( 1 + s _ { V } ^ { 2 } ) } \}$ . Applying this to the pairs $( a ^ { \top } X , a ^ { \top } X ) , ( b ^ { \top } X , b ^ { \top } X )$ and $( a ^ { \top } X , b ^ { \top } X )$ and expanding the square gives (16); the constants $\textstyle { \frac { 1 } { 4 } }$ cancel. □

Proposition 5.2 (Calibration under coeficient scaling). Suppose $v _ { a } > 0$ and $b = k a$ with $k > 0$ and let $P = \Phi ( k a ^ { \top } X )$ . For $p \in ( 0 , 1 )$ ,

$$
\operatorname* { P r } ( Y = 1 \mid P = p ) = \Phi \left( { \frac { \Phi ^ { - 1 } ( p ) } { k } } \right) .\tag{17}
$$

The forecast P ranks observations as $\eta ( X )$ does and has the same $A U C ,$ and it is calibrated if and only if $k = 1$ . Dividing the probit score by the temperature k recovers $\eta ( X )$ exactly.

Proof. The map $u \mapsto \Phi ( k u )$ is strictly increasing, so $P = p$ is equivalent to $a ^ { \top } X = \Phi ^ { - 1 } ( p ) / k$ Since P is a function of X, $\operatorname* { P r } ( Y = 1 \mid P ) = \mathbb { E } [ \eta ( X ) \mid P ] = \Phi ( a ^ { \top } X )$ , which gives (17). A strictly increasing transformation leaves the ranking unchanged, and (17) equals $p$ for all $p$ only when $k = 1$ □

Theorem 5.3 (Calibration under arbitrary coeficient distortion). Assume $v _ { b } > 0$ and let $P =$ $\Phi ( b ^ { \top } X )$ . Then

$$
\operatorname* { P r } ( Y = 1 \mid P = p ) = \Phi ( \kappa \Phi ^ { - 1 } ( p ) ) , \qquad \kappa = \frac { c / v _ { b } } { \sqrt { 1 + v _ { a } - c ^ { 2 } / v _ { b } } } .\tag{18}
$$

The result covers rotations of the coeficient vector as well as changes of its length, and it reduces to (17) when $b = k a , k > 0$

Proof. Conditionally on $b ^ { \top } X = s$ , the variable $a ^ { \top } X$ is Gaussian with mean $c s / v _ { b }$ and variance $v _ { a } - c ^ { 2 } / v _ { b }$ . For $U \sim N ( m , v )$ and an independent standard normal $E , \mathbb { E } [ \Phi ( U ) ] = \operatorname* { P r } ( E \leq U ) =$ $\Phi ( m / \sqrt { 1 + v } )$ . Substituting $s = \Phi ^ { - 1 } ( p )$ gives (18). For b = ka one has $c / v _ { b } = 1 / k$ and $c ^ { 2 } / v _ { b } = v _ { a } .$ so $\kappa = 1 / k$ □

The calibration curve (18) is linear on the probit scale with zero intercept. When $c = 0$ , the scores $a ^ { \top } X$ and $b ^ { \top } X$ are independent, P carries no information about Y and the curve is constant at $\frac { 1 } { 2 }$ .

## 5.2 Recalibration

Corollary 5.4 (Residual risk after exact recalibration). Under the assumptions of Theorem 5.3, the Brier-optimal forecast among functions of P is

$$
P _ { \mathrm { c a l } } = \mathbb { E } [ Y \mid P ] = \Phi ( \kappa b ^ { \top } X ) ,\tag{19}
$$

that $i s ,$ the teacher’s probit score divided by the temperature $1 / \kappa$ when $c > 0$ . Its excess Brier risk is

$$
\mathbb { E } [ \{ P _ { \mathrm { c a l } } - \eta ( X ) \} ^ { 2 } ] = \frac { 1 } { 2 \pi } \left[ \arcsin \frac { v _ { a } } { 1 + v _ { a } } - \arcsin \frac { c ^ { 2 } } { v _ { b } ( 1 + v _ { a } ) } \right] ,\tag{20}
$$

which is zero $i f$ and only if b is a nonzero multiple of a. For $c > 0$ the recalibration map is strictly increasing and preserves the ranking; for $c < 0$ it is strictly decreasing and reverses the ranking; for $c = 0$ the recalibrated forecast is constant at $\frac { 1 } { 2 }$

Proof. The first display follows from Theorem 5.3, and the conditional expectation minimizes ${ \mathbb E } [ \{ f ( P ) - Y \} ^ { 2 } ]$ over functions $f .$ . Because $P _ { \mathrm { c a l } } = \mathbb { E } [ \eta ( X ) \mid P ]$ , the cross term in the expansion of the square equals $\mathbb { E } [ P _ { \mathrm { c a l } } ^ { 2 } ]$ , so $\mathbb { E } [ ( P _ { \mathrm { c a l } } - \eta ) ^ { 2 } ] = \mathbb { E } [ \eta ^ { 2 } ] - \mathbb { E } [ P _ { \mathrm { c a l } } ^ { 2 } ]$ . The orthant identity in the proof of Proposition 5.1 gives $\begin{array} { r } { \mathbb { E } [ \eta ^ { 2 } ] = \frac { 1 } { 4 } + ( 2 \pi ) ^ { - 1 } } \end{array}$ arcsin $\{ v _ { a } / ( 1 + v _ { a } ) \}$ and $\begin{array} { r } { \mathbb { E } [ P _ { \mathrm { c a l } } ^ { 2 } ] = \frac { 1 } { 4 } + ( 2 \pi ) ^ { - 1 } } \end{array}$ arcsin $\{ \kappa ^ { 2 } v _ { b } / ( 1 +$ $\kappa ^ { 2 } v _ { b } ) \}$ From the definition of $\kappa , ~ 1 + \kappa ^ { 2 } v _ { b } = ( 1 + v _ { a } ) / ( 1 + v _ { a } - c ^ { 2 } / v _ { b } ) , ~ \mathrm { s o } ~ \kappa ^ { 2 } v _ { b } / ( 1 + \kappa ^ { 2 } v _ { b } ) =$ $c ^ { 2 } / \{ v _ { b } ( 1 + v _ { a } ) \}$ , which gives (20). The diference is zero if and only if $c ^ { 2 } = v _ { a } v _ { b }$ , which by the Cauchy–Schwarz inequality in the Σ inner product holds if and only if b and a are collinear. The statements about monotonicity follow from the sign of $\kappa .$ □

Temperature scaling therefore corrects a teacher whose coeficient has the right direction and the wrong length, but it leaves the excess (20) when the direction is wrong. Any function of $P$ has resolution, in the sense of Murphy’s decomposition of the Brier score [19], no greater than that of $\mathbb { E } [ Y \mid P ]$ , so no recalibration of $P$ can recover the information about $a ^ { \top } X$ that is lost in projecting onto $b ^ { \top } X$ . The statement that the exact map is a temperature scaling depends on the probit link. Under a logistic link the conditional mean of a logistic function of a Gaussian variable is not itself a logistic function of that variable, and temperature scaling on the logit scale [11] is then an approximation to the recalibration map.

## 5.3 Evaluation against teacher probabilities

When a student f is evaluated against reference probabilities t instead of outcomes, the quantity measured is $\mathbb { E } [ \{ f ( X ) - t ( X ) \} ^ { 2 } ]$ . The triangle inequality in $L ^ { 2 }$ gives

$$
\left| \mathbb { E } [ \{ f ( X ) - \eta ( X ) \} ^ { 2 } ] ^ { 1 / 2 } - \mathbb { E } [ \{ t ( X ) - \eta ( X ) \} ^ { 2 } ] ^ { 1 / 2 } \right| \leq \mathbb { E } [ \{ f ( X ) - t ( X ) \} ^ { 2 } ] ^ { 1 / 2 } .\tag{21}
$$

Agreement with the teacher therefore bounds the student’s excess Brier risk only to within the teacher’s own excess, which is given by (16) in the probit model. A student that reproduces the teacher exactly inherits that excess.

## 5.4 Information retained by a representation

For $Z = A X$ , Gaussian conditioning gives $X ^ { \top } \beta \mid Z = z \sim N ( z ^ { \top } w _ { * } , B _ { A } )$ with $w _ { * }$ and $B _ { A }$ as in (8). Therefore

$$
\operatorname* { P r } ( Y = 1 \mid Z = z ) = \Phi \left( { \frac { z ^ { \top } w _ { * } } { { \sqrt { \sigma ^ { 2 } + B _ { A } } } } } \right) ,\tag{22}
$$

so the discarded linear signal acts as additional latent noise. Equation (22) describes the best forecast available from the representation. It is not a finite-sample risk for a fitted probit model.

## 6 Coherence of candidate scores

In fixed-label classification the event, and hence the target probability, does not depend on how the candidates are presented. For a candidate-scoring model the wording and membership of the candidate set can change the output. A change is legitimate when it reflects a change in the event being asked about, and it is an inconsistency when the same event receives incompatible probabilities.

Let $q _ { \mathcal { P } } ( F \mid x )$ denote the probability that the model assigns to candidate F when the mutually exclusive and exhaustive candidate set $\mathcal { P }$ is presented with input x. Let $E = E _ { 1 } \cup E _ { 2 }$ with $E _ { 1 } \cap E _ { 2 } = \emptyset$ , and let $E ^ { c }$ be the complement of E. The model is coherent under this refinement if

$$
q _ { \{ E , E ^ { c } \} } ( E \mid x ) = q _ { \{ E _ { 1 } , E _ { 2 } , E ^ { c } \} } ( E _ { 1 } \mid x ) + q _ { \{ E _ { 1 } , E _ { 2 } , E ^ { c } \} } ( E _ { 2 } \mid x ) .\tag{23}
$$

Similarly, a relabeling or reordering of the candidates should permute the probability vector without changing the probability of any event. A softmax layer guarantees normalization within each presented set, but it does not impose (23) across diferent sets.

Proposition 6.1 (Coherence of independently computed scores). Suppose each candidate event F receives a positive weight $w ( F ) = \exp s ( F )$ that does not depend on the other candidates presented, and the probabilities are obtained by normalizing the weights over the presented set. Then (23) holds if and only if

$$
\exp s ( E ) = \exp s ( E _ { 1 } ) + \exp s ( E _ { 2 } ) , e q u i v a l e n t l y \quad s ( E ) = \log \{ \exp s ( E _ { 1 } ) + \exp s ( E _ { 2 } ) \} .\tag{24}
$$

Proof. Write $u = w ( E ) , v = w ( E _ { 1 } ) + w ( E _ { 2 } )$ and $h = w ( E ^ { c } ) > 0$ . Condition (23) reads $u / ( u + h ) =$ $v / ( v + h )$ , which is equivalent to uh = vh and hence to $u = v .$ □

The normalization in Proposition 6.1 is Luce’s choice model [16], and the assumption that weights do not depend on the other candidates is his choice axiom. Under that assumption, invariance to reordering holds automatically, and (24) is the additional condition required for refinement. The Jev announcement describes independent scoring of candidates for high-cardinality choices [1]. If candidate scores are instead computed within a shared context that contains all candidates, they can depend on the other candidates and on their order, and both reordering invariance and (23) become empirical questions that can be tested by posing the same event under diferent candidate sets.

## 7 Monte Carlo experiments

## 7.1 Design

All experiments use the parameters in Table 1. For the regression results, $\Sigma = I _ { 5 }$ and $A = [ I _ { 3 } \ : 0 ]$ , so the test-point expectation in (2) is available in closed form given the fitted coeficients, and the Monte Carlo average is taken over 20,000 independent training samples per configuration. These parameters give $B = 0 . 2 3 7 8 , B _ { A } = 0 . 0 8 0 0 , D _ { A } = 0 . 0 7 5 3 , G _ { A } = 0 . 4 6 2 5$ and $S _ { A } = 0 . 1 9 0 0$ . For the probit results, $\Sigma = I _ { 2 } , a = ( 1 , 0 ) ^ { \top }$ and $\boldsymbol { b } = ( \cos \theta , \sin \theta ) ^ { \top }$ , so that $v _ { a } = v _ { b } = 1$ and $c = \cos \theta$ . Calibration curves are estimated from $2 \times 1 0 ^ { 6 }$ simulated binary outcomes per angle in 20 equal-width forecast bins, and Brier excesses are estimated from $1 0 ^ { 8 }$ draws per angle. The code is in experiments.py, which runs on a single GPU.

Table 1: Simulation parameters.  
Quantity Value   
Design $d = 5 , r = 3 , \Sigma = I _ { 5 } , A = [ I _ { 3 } 0 ]$   
Outcome coeficient $\beta = ( 0 . 9 , - 0 . 6 , 0 . 5 , 0 . 2 , - 0 . 2 ) ^ { \top }$   
Teacher distortion $\delta = ( \mathrm { 0 . 2 2 , - 0 . 1 3 , 0 . 1 0 , 0 . 3 5 , - 0 . 2 0 } ) ^ { \top }$   
Errors $\sigma ^ { 2 } = 1 . 3 , \tau ^ { 2 } = 0 . 7 , \omega = 0 . 2 2$   
Sample-size sweep $n \in \{ 1 2 , 1 5 , 2 0 , 2 7 , 4 0 , 6 0 , 9 5 , 1 5 0 , 2 5 0 \} ; \lambda = 0 . 4 5$ for the mixed head   
Teacher-weight sweep $n = 2 7 ; 3 1$ equally spaced values of λ in [0, 1]   
Replications 20,000 training samples per configuration   
Probit calibration $\theta \in \{ 0 ^ { \circ } , 3 0 ^ { \circ } , 6 0 ^ { \circ } \}$   
Random seed 20261016 (PyTorch generator)

## 7.2 Gaussian design

Figure 1 compares the formulas with simulation. In panel (a), 34 of the 36 Monte Carlo means lie within their 95% intervals of the formulas (3), (4), (9) and (12); the two exceptions lie at 1.01 and 1.49 interval half-widths. Teacher supervision has lower risk than outcome supervision below $n = 1 8 . 6$ , and the compressed outcome model has lower risk than the full model below $n = 3 8 . 5 ,$ as predicted by (5) and (10). The teacher-trained model approaches its floor $B = 0 . 2 3 7 8 .$ , and the compressed models approach $B _ { A } = 0 . 0 8$ and $B _ { A } + \lambda ^ { 2 } D _ { A } = 0 . 0 9 5 2$ . In panel (b) all 62 Monte Carlo means lie within their 95% intervals. The formulas give optimal weights $\lambda ^ { * } = 0 . 4 2 2 1$ for the full representation and 0.4218 for the compressed one, and the grid minimizers are both 0.433, the grid point nearest to them. At $n = 2 7$ the minimum risk is 0.2010 with the full representation and 0.2066 with the compressed one, so compression does not reduce the risk of the optimally mixed estimator at this sample size. In panel (c) the binned event frequencies follow the calibration curves of Theorem 5.3, with κ = 1, 0.775 and 0.378.

![](images/8b8613a2646cefac9424e1d2bcaf104ba9fc4b4494d8a611937df78c62c78e41.jpg)

![](images/439d8ac30e23fc89eace60516ea01ec62deab4c8fbd55ba4e506e780ac560ccf.jpg)

![](images/bc0305f217ba64565f82bc5894be4ac7850cc559c6a51162705b8c66cb964f83.jpg)  
Figure 1: Exact formulas (curves) and Monte Carlo estimates (points with $9 5 \%$ intervals, mostly smaller than the markers). (a) Excess prediction risk against training size for outcome and teacher supervision with all predictors, outcome supervision through the compressed representation and mixed supervision with $\lambda = 0 . 4 5$ through the compressed representation. (b) Excess risk against teacher weight at $n = 2 7 ;$ dotted lines mark the minimizers from Proposition 3.3 and Corollary 4.6. (c) Calibration curves of rotated probit teachers (Theorem 5.3) and binned event frequencies; the dashed line is the diagonal.

Table 2 reports the Brier excesses. The simulated values agree with Proposition 5.1 and Corollary 5.4 to within 1.3 standard errors, and two-dimensional Gauss–Hermite quadrature reproduces both formulas to $1 0 ^ { - 1 5 }$ . At $\theta = 6 0 ^ { \circ }$ recalibration reduces the excess Brier risk from 0.0862 to 0.0634, and the remainder is the cost of the wrong coeficient direction.

Table 2: Excess Brier risk relative to η for rotated probit teachers, before and after exact recalibration. Monte Carlo standard errors in parentheses $( 1 0 ^ { 8 }$ draws per angle).
<table><tr><td rowspan="3">θ</td><td colspan="2">Teacher forecast  $P$ </td><td colspan="2">Recalibrated  $P _ { \mathrm { c a l } }$ </td></tr><tr><td>Formula (16)</td><td>Monte Carlo</td><td>Formula (20)</td><td>Monte Carlo</td></tr><tr><td> $0 ^ { \circ }$ </td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td> $3 0 ^ { \circ }$ </td><td>0.024117</td><td>0.024121 (0.000004)</td><td>0.022155</td><td>0.022158 (0.000003)</td></tr><tr><td> $6 0 ^ { \circ }$ </td><td>0.086236</td><td>0.086223 (0.000011)</td><td>0.063387</td><td>0.063378 (0.000007)</td></tr></table>

## 7.3 Non-Gaussian design

The inverse-Wishart moments and the independence of the residual from Z both use the Gaussian design. To measure the efect of departing from it, we drew $X = \Sigma ^ { 1 / 2 } W$ with an equicorrelated $\Sigma ~ ( \Sigma _ { j j } = 1 , \Sigma _ { j k } = 0 . 5 )$ and three laws for W, each with identity covariance: standard Gaussian, independent uniform coordinates and a multivariate t distribution with five degrees of freedom. The covariance of X is the same under all three laws, so the formulas give the same values. Under the uniform and multivariate-t laws the residua $X ^ { \top } u _ { \beta }$ remains uncorrelated with Z but is no longer independent of it. Table 3 reports the ratio of the simulated risk to the formula for outcome supervision with all predictors, (3), and for the compressed mixed head with $\lambda = 0 . 4 5$ , (12).

Table 3: Ratio of simulated excess risk to the Gaussian-design formula, equicorrelated Σ with $\rho = 0 . 5$ , 20,000 training samples per entry. The 95% Monte Carlo interval half-widths of the ratios are at most 0.022.
<table><tr><td></td><td colspan="3">Outcome OLS, all predictors</td><td colspan="3">Mixed head, compressed</td></tr><tr><td>Design law</td><td> $n = 1 2$ </td><td> $n = 2 7$ </td><td> $n = 9 5$ </td><td> $n = 1 2$ </td><td> $n = 2 7$ </td><td> $n = 9 5$ </td></tr><tr><td>Gaussian</td><td>1.005</td><td>1.003</td><td>1.000</td><td>1.011</td><td>1.006</td><td>1.000</td></tr><tr><td>Uniform coordinates</td><td>0.903</td><td>0.951</td><td>0.979</td><td>0.928</td><td>0.971</td><td>0.994</td></tr><tr><td>Multivariate  $t _ { 5 }$ </td><td>1.354</td><td>1.197</td><td>1.070</td><td>1.233</td><td>1.127</td><td>1.054</td></tr></table>

The uniform design, whose coordinates have lighter tails than the Gaussian, gives risks below the formulas, and the multivariate-t design, a scale mixture of Gaussians with heavier tails, gives risks above them. The discrepancies shrink with n. They are larger for the full model, whose risk consists entirely of estimation variance, than for the compressed head, whose risk includes the bias terms $B _ { A }$ and $\lambda ^ { 2 } D _ { A }$ that do not depend on the design law. At $n = 9 5$ the formulas are within 7% of simulation under both non-Gaussian laws, and at $n = 1 2$ they understate the risk by 26% of its simulated value under the multivariate-t law.

## 8 Application: relief decisions on credit card complaints

We tested the predictions of Sections 3–6 on a credit decision in which the evidence is text and the true probability is unknown, and the direction of every prediction held. When a consumer complains about a credit card, the issuer closes the complaint with monetary relief, with non-monetary relief or with an explanation. A bank can estimate the probability of monetary relief from the complaint and escalate the likely cases. Missing such a case costs more than reviewing one that did not need it, so the decision is taken at a low cutof.

## 8.1 Data and models

The Consumer Complaint Database of the Consumer Financial Protection Bureau (CFPB) publishes complaints with the consumer’s narrative where the consumer consented, with personal details removed [5]. The CFPB announced on 14 August 2026 that it would stop publishing narratives; we use the archive of all narratives received up to that date [6]. We keep credit card complaints closed with one of the three responses and split them by the year received: 37,520 from 2015–2018 train the teacher, 97,835 from 2019–2023 train the students, 34,957 from 2024 serve for calibration and 61,209 from 2025–2026 for testing, after removing 1,088 narratives that occur in more than one split. These are pools. Of the 97,835 student complaints, 4,000 are held out for early stopping and each student is trained on $n = 6 4 { , } 0 0 0$ drawn from the other 93,835. The recalibration curves are fitted on 5,000 of the 34,957 calibration complaints, and every model is evaluated on the same 10,000 of the 61,209 test complaints, a fixed set that makes the bootstrap comparisons paired and keeps the cost of scoring the students down. The share with monetary relief falls from 0.206 in the teacher years to 0.166 in the test years. Every model sees the company, the issue and the narrative. The label is the company’s decision, and the models predict that decision, as a triage model would.

The teacher is a multinomial logistic regression on word and bigram TF-IDF features of the narrative with indicators for the company and the issue, fitted on the teacher years only. It plays the part of the scorecard a bank already has. Because it was fitted on 2015–2018 and the students on 2019–2023, it is fixed relative to the students’ sample, as Section 9 requires. Two zero-shot readings ask a model to choose one lettered option and renormalize its next-token distribution over the letters: Qwen3-30B-A3B-Instruct [21], served with vLLM [15], and the method published as a local Jev [8], with Qwen2.5-0.5B-Instruct [20] and a fixed threshold. Each complaint is read with three options {monetary relief, non-monetary relief, explanation only} and with two options {monetary relief, no monetary relief}, each in two orders, and each format is averaged over its orders. The students follow the JevAny design [14] (Figure 2): the input is the company, the issue and the narrative, cut at 384 tokens, followed by the question and the candidates in brackets; Qwen2.5-0.5B [20] with rank-8 LoRA adapters [13] on its seven projection matrices reads it, and a linear score at each candidate’s closing bracket is normalized over the candidates shown. One student is trained on the companies’ responses and one on the teacher’s probabilities, each on $n = 6 4 { , } 0 0 0$ complaints from 2019–2023 with soft cross-entropy, questions with two or three options in random order and early stopping on 4,000 held-out complaints. Two fixed representations sit between the teacher and the students, each with a logistic regression fitted to the companies’ responses on the same 64,000 complaints as the students: the mean of the last hidden layer of Qwen2.5-0.5B with its weights frozen, which is the frozen-backbone design of Section 4, and the mean of word2vec vectors [17] trained without labels on the 2015–2018 narratives, so that the representation is fixed before the students’ sample is drawn. Recalibration uses the robust logit curve of Tasche [23, eq. A.8], $p ( x ) = 1 / [ 1 + \exp \{ - \alpha - \beta \Phi ^ { - 1 } ( \tilde { F } _ { N } ( x ) ) \} ]$ , where $\tilde { F } _ { N }$ is the distribution function of the forecast among complaints without monetary relief, averaged over its left- and right-continuous versions. The parameters are chosen by quasi moment matching [23, appendix $\mathrm { A . 2 } ]$ : the curve reproduces the relief rate of the 5,000 calibration complaints from 2024 and, through the accuracy ratio implied by the curve, their observed accuracy ratio. The curve depends on the forecast only through its rank, so, like the temperature scaling of Corollary 5.4, it leaves the ranking unchanged. All models are evaluated on the same 10,000 test complaints, of which 16.9% ended in monetary relief. The code for this section, including the recalibration (recal.py), is in the cfpb directory that accompanies experiments.py; the CFPB files are listed there with their checksums.

## 8.2 Measurement without η

For any forecast $q , \mathbb { E } [ ( q - Y ) ^ { 2 } ] = \mathbb { E } [ \{ q ( X ) - \eta ( X ) \} ^ { 2 } ] + \mathbb { E } [ \eta ( X ) \{ 1 - \eta ( X ) \} ]$ , and the second term does not depend on the model, so the diference in Brier score between two models on the same complaints estimates their diference in excess risk. We report 95% intervals from a paired bootstrap over complaints. For calibration we use the identity behind Turing’s check of weight-of-evidence calculations, $\mathbb { E } ( 1 / F \mid H ) = 1 \ [ 1 0$ , secs. 7–8]; [4, Sec. 1.4] derives it and notes that a forecast that always gives the base rate satisfies it: with $V = ( 1 - q _ { Y } ) / q _ { Y }$ the odds against the realized outcome, a forecast with $\mathbb { E } [ Y \mid q ] = q$ has $\mathbb { E } [ V \mid q ] = 1$ . The decision rule escalates when $q > c ; { \mathrm { a n } }$ unneeded escalation costs c and a missed relief case $1 - c ,$ , and we report the cost per 1,000 complaints. Coherence under the refinement of “no monetary relief” into its two subevents is measured by the mean absolute diference, in decibans, between the log-odds of monetary relief under the two candidate sets; condition (23) requires zero.

LoRA adds a rank-8 update BA to each of the seven projection matrices; the pretrained weights stay fixed  
![](images/80902f2ae0086408c07c88f17d1e50458042f603f45271b40414c6a760986a68.jpg)  
Figure 2: The student model for credit card complaints. (a) The input: the company, the issue and the consumer’s narrative, followed by the question and two or three candidates in random order. (b) One of the 24 blocks of Qwen2.5-0.5B; the pretrained weights stay fixed, and LoRA trains a rank-8 update BA to each projection matrix. (c) The candidate-scoring head reads the hidden state at each candidate’s closing bracket and normalizes the scores over the presented candidates.

## 8.3 Results

Teacher floor. The student trained on the teacher’s probabilities ends where its teacher is (Table 4). Its decision cost at c = 0.2 difers from the teacher’s by −0.4 per 1,000 (interval −3.1 to 2.1), and its Brier score is lower by 0.0029 (0.0015 to 0.0044). The student trained on outcomes goes past the teacher: its Brier score is lower by 0.0049 (0.0026 to 0.0073), its AUC is 0.80 against 0.77 and its decision cost at c = 0.2 is lower by 7.3 per 1,000 (3.8 to 10.9), or 8%. This is the pattern of Proposition 3.2 and of (21) at a sample size past the crossover. Against the teacher-trained student, the outcome student’s Brier score is lower by 0.0020 with an interval from −0.0041 to 0.0001, so the separation of the two students at this n rests on one training run each.

Fixed representations. The frozen Qwen2.5-0.5B backbone with a logistic head, trained on outcomes, has a decision cost lower than the teacher’s by 4.6 per 1,000 at c = 0.2 (1.4 to 7.9), and its Brier score is lower by 0.0022, with an interval from 0.0000 to 0.0044 that reaches zero; word2vec does no better than the teacher on either measure. Adapting the same backbone with LoRA lowers the Brier score by a further 0.0027 (0.0011 to 0.0044) and the decision cost by 2.7 per 1,000, an interval from −0.2 to 5.4 that just includes zero. The frozen representation therefore discards some signal about the decision that adaptation recovers, the representation loss $B _ { A }$ of Proposition 4.1. The training target matters more than the backbone here: the frozen backbone trained on outcomes has a decision cost lower by 4.2 per 1,000 (0.9 to 7.6) than the LoRA student trained on the teacher.

Table 4: Forecasts of monetary relief on 10,000 test complaints received in 2025–2026; the share with relief is 0.169. Turing’s mean V is 1 for a calibrated forecast. Models trained on the 2019–2023 responses (the two fixed representations and the students) are reported without recalibration; the teacher and Qwen3-30B are recalibrated on 2024. The last column measures the departure from (23). <sup>a</sup>Uncalibrated forecast; the mean is dominated by confident forecasts that turned out wrong.
<table><tr><td rowspan="2">Model</td><td colspan="3">mean</td><td rowspan="2">Turing</td><td colspan="2">cost per 1,000</td><td rowspan="2">departure</td></tr><tr><td>forecast</td><td>Brier</td><td>AUC mean V</td><td>c = 0.2</td><td>c = 0.3</td></tr><tr><td>Constant at the 2024 rate</td><td>0.171</td><td>0.141</td><td>0.50</td><td>0.99</td><td>135.4</td><td>118.4</td><td></td></tr><tr><td>Letter reading, 0.5B, as published</td><td>0.262</td><td>0.160</td><td>0.48</td><td>0.95</td><td>164.3</td><td>165.1</td><td>5.0</td></tr><tr><td>Qwen3-30B zero-shot</td><td>0.509</td><td>0.364</td><td>0.73</td><td>219ª</td><td>100.2</td><td>136.7</td><td>17.5</td></tr><tr><td>Qwen3-30B, recalibrated</td><td>0.204</td><td>0.137</td><td>0.73</td><td>0.96</td><td>100.7</td><td>116.0</td><td></td></tr><tr><td>Logistic teacher, recalibrated</td><td>0.183</td><td>0.124</td><td>0.77</td><td>1.12</td><td>90.0</td><td>100.1</td><td></td></tr><tr><td>word2vec + logistic</td><td>0.175</td><td>0.123</td><td>0.77</td><td>1.05</td><td>90.7</td><td>103.7</td><td></td></tr><tr><td>Frozen Qwen2.5-0.5B + logistic</td><td>0.186</td><td>0.121</td><td>0.79</td><td>0.95</td><td>85.4</td><td>99.8</td><td></td></tr><tr><td>Student on the teacher</td><td>0.143</td><td>0.121</td><td>0.78</td><td>1.32</td><td>89.6</td><td>100.4</td><td>0.20</td></tr><tr><td>Student on outcomes</td><td>0.194</td><td>0.119</td><td>0.80</td><td>1.70</td><td>82.7</td><td>97.3</td><td>0.69</td></tr></table>

Recalibration. Read zero-shot, Qwen3-30B ranks complaints usefully (AUC 0.73) but puts the probability of monetary relief at 0.51 on average, and in the three deciles it rates highest, where its forecast is close to 1, between 28% and 33% of complaints received relief (Figure 3a). Its calibration slope [7], the coeficient of a logistic regression of the outcome on the logit of the forecast in 2024, is 0.128, so its log-odds are about eight times too extreme; the slope describes the miscalibration, and the recalibration itself uses the rank-based curve above. Recalibration lowers its Brier score from 0.364 to 0.137 and leaves its AUC unchanged, and the recalibrated model remains worse than the teacher by 0.0133 in Brier score (0.0101 to 0.0166) and by 10.7 per 1,000 in decision cost at c = 0.2 (6.6 to 15.1). As Corollary 5.4 states for the probit model, recalibration corrects the scale of a forecast and leaves the excess that comes from its ranking.

Calibration diagnostics. Turing’s mean V is close to 1 for the recalibrated Qwen3-30B (0.96), for the published letter reading, which ranks complaints no better than chance (0.95), and for the constant forecast (0.99). The best model, the student trained on outcomes, gives 1.70, because the 1% of complaints with the largest V , confident forecasts that turned out wrong, make up two thirds of its sum. The check measures calibration only, and a validator who relied on it would accept the constant forecast and reject the best model. This is the statement after Corollary 5.4 that calibration diagnostics do not detect the residual (20).

Coherence. Qwen3-30B, which reads all candidates in one context, departs from (23) by 17.5 decibans on average: its probability of monetary relief averages 0.51 with three candidates and 0.31 with two, and many complaints receive a probability near 1 under one set and near 0 under the other (Figure 3c). The published letter reading departs by 5.0 decibans, and reversing the order of its options moves its answer by 0.40 on average. The trained students depart by 0.20 and 0.69 decibans, and reversing the order moves their answers by 0.007 and 0.012. The students were trained with candidate sets and orders drawn at random, and they come close to (23) although their architecture does not impose (24).

![](images/e295c80b120d54e97aceeaeca81e635d7f547ec74cd7a9bb27c2383dfad886f9.jpg)

![](images/d4b360bd8ab8a5632a2cbdcf6ce50577713ba6004de8f3cc25f909097288a895.jpg)

![](images/2dc804fca13f97ff9ad05dffc10f6a7deeb881761c6657153cd6209f63c312fd.jpg)  
Figure 3: Credit card complaints, 10,000 test complaints. (a) Observed share with monetary relief against the forecast, by deciles of each forecast. (b) Decision cost per 1,000 complaints against the cutof c. (c) Probability of monetary relief under the three-option and the two-option candidate sets for 1,500 complaints; a model satisfying (23) lies on the diagonal.

## 9 Scope and limitations

A deterministic linear teacher corresponds to $\tau = 0$ and is covered. A deterministic nonlinear teacher has residuals that are uncorrelated with the features in the population but in general depend on X and are heteroscedastic given $Z ,$ so the independence step in Theorem 4.3 fails. A teacher trained on records that overlap the student’s sample makes $\delta$ and the teacher errors depend on that sample. Training LoRA adapters, as in the JevAny models, makes the representation depend on the labels and is not covered.

Fitting an intercept changes the degrees of freedom. With centered Gaussian predictors and an estimated intercept, the centered Gram matrix is distributed as $W _ { d } ( \Sigma , n - 1 )$ , so the slope-risk factors $d / ( n - d - 1 )$ and $r / ( n - r - 1 )$ become $d / ( n - d - 2 )$ and $r / ( n - r - 2 )$ , for $n > d + 2$ and $n > r + 2$ respectively; the intercept estimate contributes a further term to the prediction error.

The probit results describe fixed teacher forecasts and population recalibration. They do not give the finite-sample distribution of a fitted nonlinear classifier or of a temperature estimated on a finite calibration set.

Section 8 carries out the second of the following tests on credit card complaints, a version of the first, and, with the frozen-backbone and word2vec representations, a version of the third. The first perturbs the serialization of a fixed tabular record (field order, number formatting and label wording) and measures the variation of the output probability within an observation. The second divides one candidate into mutually exclusive subcandidates and measures the departure from (23). The third compares a candidate-scoring model with a frozen pretrained embedding followed by a conventional calibrated classifier or kernel method, both trained on the same observations and evaluated with proper scores on held-out outcomes over paired splits and seeds. The present analysis gives the statistical tradeofs that such a comparison involves but does not predict its outcome. In Section 8 the perturbation is the wording and order of the candidates, and the comparison is between a logistic teacher, zero-shot readings, two fixed representations with logistic heads and LoRA students on the same complaints. That application uses one training run per student and one training size, so the crossover sample size was not located; its label is a company’s decision, which encodes the company’s policy as well as the content of the complaint; and its models lie outside the Gaussian, probit and fixed-representation assumptions, so it tests the direction of the results; their values do not carry over.

Finally, predicting an outcome and selecting an action are diferent problems. A recommendation label may encode a human policy, a teacher’s judgment or an optimization rule, and the appropriate target and loss depend on which of these is to be predicted. When actions change outcomes, causal assumptions are required in addition to the statistical ones used here.

## 10 Conclusion

Under Gaussian surrogates the three training ingredients documented for candidate-scoring models have exact consequences. Training on teacher targets leaves a risk floor equal to the teacher’s coeficient bias, so it improves on outcome supervision only below a crossover sample size. A fixed representation trades the representation loss $B _ { A }$ against the estimation variance; it also projects the teacher bias onto the retained directions and moves the unrepresented teacher signal into the variance. The optimal teacher weight depends on the sample size, the teacher distortion and the correlation of the target errors. In the probit model, temperature scaling corrects a teacher whose coeficient has the right direction, and a residual excess, given in closed form, remains when the direction is wrong. The regression formulas assume a Gaussian design; under a multivariate-t design the simulated risk exceeds them by up to 35% at n = 12, and the discrepancy shrinks as n grows. On credit card complaints the direction of each result held outside these assumptions: a student trained on a teacher stayed at the teacher’s decision cost while a student trained on outcomes went past it, recalibration left a large zero-shot model worse than a logistic regression, Turing’s calibration check accepted a constant forecast and rejected the best model, and a model reading all candidates in one context departed from the coherence condition by 17.5 decibans.

For practitioners the results suggest five checks.

1. Training target. A model trained on teacher or reference probabilities estimates the teacher’s conditional distribution. Its error against outcomes includes a teacher bias that more student data does not remove (Proposition 3.2) and that agreement with the teacher does not reveal (21).

2. Evaluation. Proper scores should be computed on realized outcomes. The model should be compared with a conventional classifier fitted to the same outcomes and with one fitted to the same teacher probabilities, so that the efect of the architecture is separated from the efect of the training target.

3. Pretrained representations. A fixed representation reduces estimation variance. It lowers the risk only while the signal it discards is below the threshold (10), and the threshold decreases as the sample grows.

4. Calibration. Recalibration corrects the scale of a forecast. A calibrated forecast can retain the excess risk (20) of a wrong coeficient direction, and calibration diagnostics do not detect it.

5. Coherence. The same event should receive the same probability under diferent candidate sets (23), which can be tested by posing the event under several sets.

A model that predicts which candidate an annotator, a teacher or a past policy would select has not been shown to select the action with the best outcome. Establishing that requires a model of outcomes given actions and, when actions change outcomes, causal assumptions.

## References

[1] D. Almeida, “Introducing System One Models & Jev,” TypeSafe blog, 2026. https://typesafe. ai/blog/introducing-system-one-models-and-jev.

[2] T. W. Anderson, An Introduction to Multivariate Statistical Analysis, 3rd ed. Hoboken, NJ: Wiley, 2003.

[3] L. Breiman and D. Freedman, “How Many Variables Should Be Entered in a Regression Equation?” Journal of the American Statistical Association, 78(381):131–136, 1983.

[4] D. Burakov, Inference Machines: What Language Models Believe, and How They Change Their Minds, 2026. Companion repository: https://github.com/deburky/ inference-machines-book.

[5] Consumer Financial Protection Bureau, “Consumer Complaint Database,” 2026. https: //www.consumerfinance.gov/data-research/consumer-complaints/. Downloaded 18 October 2026.

[6] Consumer Financial Protection Bureau, “CFPB Consumer Complaint Database Narratives Archive,” FOIA electronic reading room, 2026. https: //www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/ cfpb-consumer-complaint-database-narratives-archive/.

[7] D. R. Cox, “Two Further Applications of a Model for Binary Regression,” Biometrika, 45(3/4):562–565, 1958.

[8] D. García, “Build your own Jev (100% local),” Towards Dev, 21 September 2026.

[9] T. Gneiting and A. E. Raftery, “Strictly Proper Scoring Rules, Prediction, and Estimation,” Journal of the American Statistical Association, 102(477):359–378, 2007.

[10] I. J. Good, “Studies in the History of Probability and Statistics. XXXVII. A. M. Turing’s Statistical Work in World War II,” Biometrika, 66(2):393–396, 1979.

[11] C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, “On Calibration of Modern Neural Networks,” in Proceedings of the 34th International Conference on Machine Learning, PMLR 70:1321–1330, 2017.

[12] G. Hinton, O. Vinyals and J. Dean, “Distilling the Knowledge in a Neural Network,” arXiv:1503.02531, 2015.

[13] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang and W. Chen, “LoRA: Low-Rank Adaptation of Large Language Models,” in International Conference on Learning Representations, 2022.

[14] JevAny contributors, “Algorithm,” JevAny documentation, 2026. https://simplejev.org/ JevAny/docs/ALGORITHM.html.

[15] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang and I. Stoica, “Eficient Memory Management for Large Language Model Serving with PagedAttention,” in Proceedings of the 29th Symposium on Operating Systems Principles, 2023.

[16] R. D. Luce, Individual Choice Behavior: A Theoretical Analysis. New York: Wiley, 1959.

[17] T. Mikolov, I. Sutskever, K. Chen, G. Corrado and J. Dean, “Distributed Representations of Words and Phrases and their Compositionality,” in Advances in Neural Information Processing Systems 26, 2013.

[18] R. J. Muirhead, Aspects of Multivariate Statistical Theory. New York: Wiley, 1982.

[19] A. H. Murphy, “A New Vector Partition of the Probability Score,” Journal of Applied Meteorology, 12:595–600, 1973.

[20] Qwen Team, “Qwen2.5 Technical Report,” arXiv:2412.15115, 2024.

[21] Qwen Team, “Qwen3 Technical Report,” arXiv:2505.09388, 2025.

[22] W. F. Sheppard, “On the Application of the Theory of Error to Cases of Normal Distribution and Normal Correlation,” Philosophical Transactions of the Royal Society of London, Series A, 192:101–167, 1899.

[23] D. Tasche, “The Art of Probability-of-Default Curve Calibration,” Journal of Credit Risk, 9(4):63–103, 2013.