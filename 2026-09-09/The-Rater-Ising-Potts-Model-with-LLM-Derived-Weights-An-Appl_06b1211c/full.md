# The Rater Ising-Potts Model with LLM-Derived Weights: An Application to Multi-Category Scoring Reliability

Matthias von Davier<sup>∗</sup>

September 9, 2026

## Abstract

The Ising model is extended to the Potts model for multinomial data. We introduce a Rater Ising-Potts model that uses agreement indicators between pairs of raters and category labels, with weights derived from LLM embeddings. The model does not presuppose ordered category thresholds or equidistant scoring; instead, it focuses directly on pairwise agreement among raters and assigns category-specific positive weights, making it particularly suited for multi-category scoring reliability when raters evaluate responses using a scoring guide. We demonstrate the model’s efectiveness on diverse constructed-response tasks, including balanced short-answer items and more challenging, imbalanced essay prompts from the AERA dataset. Across these settings, the model achieves strong agreement with human scores, with the vast majority of misclassifications occurring between adjacent score levels, confirming its ability to preserve the ordinal structure of scoring rubrics without imposing rigid assumptions. A practical similarity normalization and optional power transformation is introduced as a tunable preprocessing step that sharpens semantic distinctions and can be adapted to diferent datasets. These findings suggest that LLM-derived semantic similarities, combined with this parsimonious Potts-type formulation and flexible similarity scaling, ofer a robust and

interpretable framework for reliability auditing in educational assessment contexts. Extensions to multiple raters and hierarchical rating processes are discussed.

## 1 Introduction

We introduce a Rater Ising–Potts model that leverages agreement indicators between pairs of raters and category labels, with weights derived from LLM embeddings. This paper extends an Ising Rater model for binary items (von Davier, 2026). In contrast to conventional polytomous item response models— such as the graded response model or the generalized partial credit model—our approach does not presuppose ordered category thresholds. Instead, it focuses directly on pairwise agreement among raters and assigns category-specific positive weights, making it particularly suited for multi-category scoring reliability when raters evaluate responses using a scoring guide.

To assess the practical utility of the proposed framework, we conduct two empirical applications that span diverse response characteristics. The first uses a custom dataset of $K = 7 5 0$ short answers to an open-ended question about plant growth, scored on a three-level rubric. The second applies the model to Essay 1 of the AERA dataset (Li et al., 2023, Li, 2024), which comprises $N = 1 { , } 3 3 8$ longer, more complex responses scored on a four-point scale. These two examples difer markedly in response length, number of categories, and scoring dificulty, providing a robust test of the model’s flexibility and performance. As we will demonstrate, the PARM achieves strong agreement with human scores in both settings, with near-agreement rates (predictions within ±1 category) of 1.000 and 0.890, respectively. These results indicate that the model captures the ordinal structure of scoring rubrics even when exact agreement is challenging, and they suggest that LLM-derived semantic similarities, combined with a parsimonious Potts-type formulation, ofer a flexible and interpretable framework for reliability auditing in educational assessment contexts. We also discuss extensions to multiple raters and hierarchical rating processes.

## 1.1 Notation

Assume a k-dimensional $\pmb { X } = ( X _ { 1 } , \ldots , X _ { K } )$ with im $\mathbf { \xi } ( X ) = \left\{ 0 , \ldots , C \right\} ^ { K }$ and $C \in \mathbb { N }$ . A realization of this random variable will be denoted by $( x _ { 1 } , \ldots , x _ { K } )$

and we assume a joint distribution

$$
P \left( x _ { 1 } , \dots , x _ { K } \right) = \exp \left[ g \left( x _ { 1 } , \dots , x _ { K } ; \Theta \right) - Z _ { g } \right] ,
$$

where the partition function is

$$
{ \cal Z } _ { g } = \ln \left( \sum _ { \left( x _ { 1 } ^ { \prime } , \ldots , x _ { K } ^ { \prime } \right) \in \{ 0 , \ldots , C \} ^ { K } } \exp \left[ g \left( x _ { 1 } ^ { \prime } , \ldots , x _ { K } ^ { \prime } ; \Theta \right) \right] \right) .
$$

## 1.2 Ising or Quadratic Exponential Model

The Ising model is defined by C = 1 and $\Theta = ( w , \pmb { v } )$ , with

$$
g \left( x _ { 1 } , \ldots , x _ { K } ; \Theta \right) = \frac 1 2 \sum _ { i , j } w _ { i j } ^ { * } x _ { i } ^ { * } x _ { j } ^ { * } + \sum _ { i } v _ { i } ^ { * } x _ { i } ^ { * } ,
$$

where $x _ { i } ^ { * } = 2 x _ { i } - 1 \in \{ - 1 , 1 \} , w _ { i j } ^ { * } = w _ { j i } ^ { * }$ , and $w _ { i i } ^ { * } = 0$ . This is the usual spin representation. Note that the Ising model was not only reinvented in the domain of artificial neural networks as described by Hopfield (1982) but also was championed by psychometricians as Network Psychometrics (Epskamp et al. 2018; von Davier, 2018). Molenaar (2004) pointed out that the Ising network is (almost) isomorph to the Rasch model (e.g., Rasch, 1960; von Davier, 2016) for binary data.

We now show the equivalence of the magnetic spin notation of binary states as $\{ - 1 , + 1 \}$ to a binary variable {0, 1} representation. Since $x _ { i } ^ { * } = 2 x _ { i } - 1$ , we have

$$
w _ { i j } ^ { * } x _ { i } ^ { * } x _ { j } ^ { * } = 4 w _ { i j } ^ { * } x _ { i } x _ { j } - 2 w _ { i j } ^ { * } x _ { i } - 2 w _ { i j } ^ { * } x _ { j } + w _ { i j } ^ { * } .
$$

Thus

$$
\frac { 1 } { 2 } \sum _ { i , j } w _ { i j } ^ { * } x _ { i } ^ { * } x _ { j } ^ { * } = 2 \sum _ { i , j } w _ { i j } ^ { * } x _ { i } x _ { j } - \sum _ { i , j } w _ { i j } ^ { * } x _ { i } - \sum _ { i , j } w _ { i j } ^ { * } x _ { j } + \frac { 1 } { 2 } \sum _ { i , j } w _ { i j } ^ { * } .
$$

Because w $\boldsymbol { \mathbf { \Phi } } _ { i j } ^ { * } = \boldsymbol { w } _ { j i } ^ { * }$ , the two linear sums are equal, so

$$
- \sum _ { i , j } w _ { i j } ^ { * } x _ { i } - \sum _ { i , j } w _ { i j } ^ { * } x _ { j } = - 2 \sum _ { i , j } w _ { i j } ^ { * } x _ { i } = - 2 \sum _ { i } x _ { i } \sum _ { j } w _ { i j } ^ { * } .
$$

Also,

$$
\sum _ { i } v _ { i } ^ { * } x _ { i } ^ { * } = 2 \sum _ { i } v _ { i } ^ { * } x _ { i } - \sum _ { i } v _ { i } ^ { * } .
$$

Therefore

$$
g \left( x _ { 1 } , \ldots , x _ { K } ; \Theta \right) = 2 \sum _ { i , j } w _ { i j } ^ { * } x _ { i } x _ { j } + \sum _ { i } \left( 2 v _ { i } ^ { * } - 2 \sum _ { j } w _ { i j } ^ { * } \right) x _ { i } + \left( \frac { 1 } { 2 } \sum _ { i , j } w _ { i j } ^ { * } - \sum _ { i } v _ { i } ^ { * } \right) .
$$

Define

$$
w _ { i j } = 2 w _ { i j } ^ { * }
$$

and

$$
v _ { i } = 2 v _ { i } ^ { * } - 2 \sum _ { j } w _ { i j } ^ { * } = 2 \left( v _ { i } ^ { * } - \sum _ { j } w _ { i j } ^ { * } \right) .
$$

The constant term

$$
\frac { 1 } { 2 } \sum _ { i , j } w _ { i j } ^ { * } - \sum _ { i } v _ { i } ^ { * }
$$

is absorbed by the normalizing constant of the exponential family, so the model can be written equivalently as

$$
g \left( x _ { 1 } , \ldots , x _ { K } ; \Theta \right) = \sum _ { i , j } w _ { i j } x _ { i } x _ { j } + \sum _ { i } v _ { i } x _ { i } ,
$$

with $x _ { i } \in \{ 0 , 1 \} , w _ { i j } = w _ { j i }$ , and $w _ { i i } = 0$ . Thus the two parameterizations lead to identical model based probabilities.

The model parameters are the $K \left( K - 1 \right) / 2$ weights $w _ { i j }$ for the products $x _ { i } x _ { j }$ and the $K$ parameters $v _ { i }$ for the linear terms $x _ { i }$

## 1.3 Potts Model

For multi-category variables $X _ { i } \in \{ 0 , \ldots , C \}$ , the Potts model (1952) generalizes the Ising (1925) model. The unconstrained Potts model has the form

$$
P \left( x _ { 1 } , \ldots , x _ { K } \right) = \exp \left[ \sum _ { i < j } w _ { i j } \left( x _ { i } , x _ { j } \right) + \sum _ { i } v _ { i } \left( x _ { i } \right) - Z \right] ,
$$

where the partition function is given by

$$
Z = \ln \left( \sum _ { \left( x _ { 1 } , \ldots , x _ { K } \right) \in \{ 0 , \ldots , C \} ^ { K } } \exp \left[ \sum _ { i < j } w _ { i j } \left( x _ { i } , x _ { j } \right) + \sum _ { i } v _ { i } \left( x _ { i } \right) \right] \right) .
$$

Alternatively, as before, one may use

$$
w _ { i j } ^ { \# } = \frac { 1 } { 2 } w _ { i j }
$$

and set $w _ { i i } ^ { \# } = 0$ so that

$$
\sum _ { i < j } w _ { i j } \left( x _ { i } , x _ { j } \right) = \sum _ { i = 1 } ^ { K } \sum _ { j = 1 } ^ { K } w _ { i j } ^ { \# } \left( x _ { i } , x _ { j } \right) .
$$

The $w _ { i j } \left( x _ { i } , x _ { j } \right)$ assign a weight to each of the $\left( C + 1 \right) ^ { 2 }$ possible pairs of categories for variables i and $j$ . Without further constraints, this requires up to

$$
\left( C + 1 \right) ^ { 2 } { \frac { K \left( K - 1 \right) } { 2 } }
$$

interaction parameters $w _ { i j } \left( x _ { i } , x _ { j } \right)$ , plus up to $K \left( C + 1 \right)$ main-efect parameters $v _ { i } \left( x _ { i } \right)$ . Identifiability constraints, such as setting one category as a baseline for each variable, are needed in practice.

## 1.4 Potts-Ising Model

A parsimonious ordinal version can be obtained by assuming that the categories are ordered and can be represented by numeric scores. Let

$$
x _ { i } ^ { * } = 2 x _ { i } - C ,
$$

so that $x _ { i } ^ { * } \in \{ - C , - C + 2 , . . . , C \}$ is symmetric around zero. Define

$$
w _ { i j } \left( x _ { i } , x _ { j } \right) = w _ { i j } x _ { i } ^ { * } x _ { j } ^ { * }
$$

and

$$
v _ { i } \left( x _ { i } \right) = v _ { i } x _ { i } ^ { * } .
$$

Then the Potts-Ising model is given by

$$
P \left( x _ { 1 } , \dots , x _ { K } \right) = \exp \left[ \sum _ { i < j } w _ { i j } x _ { i } ^ { * } x _ { j } ^ { * } + \sum _ { i } v _ { i } x _ { i } ^ { * } - Z \right] ,
$$

with

$$
Z = \ln \left( \sum _ { \left( x _ { 1 } , \ldots , x _ { K } \right) \in \{ 0 , \ldots , C \} ^ { K } } \exp \left[ \sum _ { i < j } w _ { i j } x _ { i } ^ { * } x _ { j } ^ { * } + \sum _ { i } v _ { i } x _ { i } ^ { * } \right] \right) .
$$

This reduces the number of parameters to $\frac { K ( K - 1 ) } { 2 } + K$ . A further reduction can be obtained by setting

$$
w _ { i j } = u _ { i } u _ { j } ,
$$

which requires only $K + K = 2 K$ parameters in total: K for the $u _ { i }$ and K for the $v _ { i }$ .

A related approach was introduced by Razaee & Amini (2020), who adapted ideas that reminisc from polytomous item response models such as the graded response model (Samejima, 1969) and the generalized partial credit model (Muraki, 1992). Specifically, they set

$$
w _ { i j } \left( x _ { i } , x _ { j } \right) = w _ { i j } x _ { i } ^ { * } x _ { j } ^ { * }
$$

with $\boldsymbol { x } _ { i } ^ { * }$ defined by a linear transformation of the category labels. However, this model depends on the choice of category scores. Diferent score assignments, such as $\{ 1 , \ldots , C \} , \left\{ 1 , 4 , 9 , \ldots , C ^ { 2 } \right\}$ , or $\{ 0 , \ldots , 0 , 1 \}$ , can lead to diferent predictions. To avoid this dependence, we introduce a model based directly on agreement indicators.

## 1.5 A Potts Model for Scoring Agreement

In this section, a model is defined that is based only on the agreement between two responses $x _ { i }$ and $x _ { j }$ If there is agreement with respect to category $c ,$ then a weight $w _ { i j c }$ is applied. If there is no agreement $x _ { i } \neq x _ { j }$ , there is no (direct) contribution to the probability function. Non-agreement contributes indirectly through the absence of terms for pairs that do not agree, and the normalization of the argument involving Z.

As a first step, the weights and the function of the score tupels are separated.

We define category-specific agreement indicators

$$
\alpha _ { c } \left( x _ { i } , x _ { j } \right) = 1 _ { \left\{ ( c , c ) \right\} } \left[ \left( x _ { i } , x _ { j } \right) \right] = 1 _ { \left\{ 0 \right\} } \left( \left| x _ { i } - c \right| + \left| x _ { j } - c \right| \right) ,
$$

which equals 1 if both $x _ { i }$ and $x _ { j }$ equal category $c ,$ and is 0 otherwise. Alternatively, for a simplified model, one may define an overall agreement indicator

$$
\alpha \left( x _ { i } , x _ { j } \right) = 1 _ { \left\{ 0 \right\} } \left( \left| x _ { i } - x _ { j } \right| \right) ,
$$

which equals 1 if $x _ { i } = x _ { j }$ , regardless of the response categories.

A category-specific agreement model is obtained by setting

$$
w _ { i j } \left( x _ { i } , x _ { j } \right) = \sum _ { c = 0 } ^ { C } w _ { i j c } \alpha _ { c } \left( x _ { i } , x _ { j } \right) ,
$$

with $w _ { i j c } = w _ { j i c }$ and $w _ { i i c } = 0$ . Note that $\begin{array} { r } { \sum _ { c = 0 } ^ { C } \alpha _ { c } \left( x _ { i } , x _ { j } \right) \le 1 } \end{array}$ as there is at most one $c \in \{ 0 , \ldots , C \}$ where the two response scores $x _ { i } = x _ { j } = c .$ . This means the baseline is for all $x _ { i } \neq x _ { j }  \forall c : \alpha _ { c } ( x _ { i } , x _ { j } ) = 0$ . Hence, all $C + 1$ parameters $\beta _ { c } , c \in \{ 0 , \ldots , C \}$ are identified as we have the baseline defined by non-agreement of responses. The main efects (biases) are defined as

$$
v _ { i } \left( x _ { i } \right) = \sum _ { c = 0 } ^ { C } v _ { i c } 1 _ { \{ c \} } \left( x _ { i } \right)
$$

where we set $v _ { i 0 } = 0$ . The resulting model is

$$
P \left( x _ { 1 } , \dots , x _ { K } \right) = \exp \left[ \sum _ { i < j } \sum _ { c = 0 } ^ { C } w _ { i j c } \alpha _ { c } \left( x _ { i } , x _ { j } \right) + \sum _ { i } \sum _ { c = 0 } ^ { C } v _ { i c } 1 _ { \left\{ c \right\} } \left( x _ { i } \right) - Z \right] ,
$$

with Z defined accordingly. Without constraints, this model has

$$
\left( C + 1 \right) \frac { K \left( K - 1 \right) } { 2 } + K \left( C + 1 \right)
$$

parameters. Identifiability constraints such as $v _ { i 0 } = 0$ for all i should be imposed.

## 2 A Potts Model for Reliability Auditing

Assume the response scores $X _ { i }$ are associated with natural language responses $Y _ { i } ,$ , which are mapped to embeddings $e _ { i } = \cosh \left( y _ { i } \right)$ in a high-dimensional space. Define the cosine similarity between pairs of embeddings as

$$
s \left( e _ { i } , e _ { j } \right) = \frac { 1 } { K } \frac { e _ { i } \cdot e _ { j } } { \left\| e _ { i } \right\| \left\| e _ { j } \right\| } = s \left( e _ { j } , e _ { i } \right) .
$$

## 2.1 Optional Similarity Normalization and Power Transformation

Let $s _ { i j }$ denote the raw cosine similarity between items i and $j ,$ computed from their sentence embeddings, with the diagonal set to zero $( s _ { i i } = 0 )$ . To place all similarities on a common scale, we first apply min–max normalization using the minimum $m = \operatorname* { m i n } _ { i \neq j } s _ { i j }$ and maximum $M = \operatorname* { m a x } _ { i \neq j } s _ { i j }$ of the of-diagonal entries:

$$
\tilde { s } _ { i j } = \frac { s _ { i j } - m } { M - m } ,
$$

which maps every similarity into the interval [0, 1]. Optionally, a power transformation is then applied to adjust the distribution of the similarities:

$$
\begin{array} { r } { s _ { i j } ^ { \prime } = ( \tilde { s } _ { i j } ) ^ { p } , } \end{array}
$$

where $p$ is a positive constant (default $p = 1 )$ . For $p > 1$ , the transformation emphasizes the strongest similarities and suppresses moderate ones; for $0 < p <$ 1, it has the opposite efect. After the transformation, the diagonal is again set to zero, so that self-similarities do not contribute to the feature computation. The resulting matrix $S ^ { \prime } = ( s _ { i j } ^ { \prime } )$ is used to construct the total similarity predictor $\begin{array} { r } { S _ { i } = \sum _ { j } s _ { i j } ^ { \prime } } \end{array}$ and the category-specific similarity sums $T _ { i } ( c )$

Using these similarities, we can specify a parsimonious agreement model. Let

$$
w _ { i j c } = \beta _ { c } s \left( e _ { i } , e _ { j } \right)
$$

and define the total similarity for each response as

$$
S _ { i } = \sum _ { j \neq i } s \left( e _ { i } , e _ { j } \right) .
$$

We then include an additional main-efect term depending on the total similarity:

$$
v _ { i } \left( x _ { i } \right) = \mu _ { x _ { i } } + \gamma _ { x _ { i } } S _ { i } ,
$$

where $\mu _ { c }$ and $\gamma _ { c }$ are category-specific parameters. For identifiability we set $\beta _ { 0 } = \gamma _ { 0 } = \mu _ { 0 } = 0$ . The model becomes

$$
P \left( x _ { 1 } , \dots , x _ { K } \right) = \exp \left[ \sum _ { i < j } ^ { C } \sum _ { c = 1 } ^ { C } \beta _ { c } s \left( e _ { i } , e _ { j } \right) \alpha _ { c } \left( x _ { i } , x _ { j } \right) + \sum _ { i = 1 } ^ { K } \left( \mu _ { x _ { i } } + \gamma _ { x _ { i } } S _ { i } \right) - Z \right] ,
$$

with

$$
Z = \ln \left( \sum _ { \left( x _ { 1 } , \ldots , x _ { K } \right) \in \{ 0 , \ldots , C \} ^ { K } } \exp \left[ \sum _ { i < j } \sum _ { c = 1 } ^ { C } \beta _ { c } s \left( e _ { i } , e _ { j } \right) \alpha _ { c } \left( x _ { i } , x _ { j } \right) + \sum _ { i = 1 } ^ { K } \left( \mu _ { x _ { i } } + \gamma _ { x _ { i } } S _ { i } \right) \right] \right) .
$$

This model has 3C free parameters: $\beta _ { 1 } , \dots , \beta _ { C } , \gamma _ { 1 } , \dots , \gamma _ { C }$ , and $\mu _ { 1 } , \ldots , \mu _ { C }$

## 3 Relationship to Multinomial Logistic Regression Models

For the k-th scored response $x _ { k }$ and response categories $b , c \in \{ 0 , \ldots , C \}$ , consider the conditional ratio

$$
R \left( c , b \right) = \frac { P \left( x _ { 1 } , \ldots , x _ { k - 1 } , c , x _ { k + 1 } , \ldots , x _ { K } \right) } { P \left( x _ { 1 } , \ldots , x _ { k - 1 } , b , x _ { k + 1 } , \ldots , x _ { K } \right) } .
$$

In the Potts auditing-reliability model (PARM), the exponent for $X _ { k } = c$ is

$$
\beta _ { c } T _ { k } \left( c \right) + \gamma _ { c } S _ { k } + \mu _ { c } ,
$$

where

$$
{ { T } _ { k } } \left( c \right) = \sum _ { i = 1 , i \ne k } ^ { K } { s \left( { { e } _ { i } } , { { e } _ { k } } \right) { { 1 } _ { \left\{ c \right\} } } \left( { { x } _ { i } } \right) }
$$

and

$$
S _ { k } = \sum _ { j \neq k } s \left( e _ { k } , e _ { j } \right) .
$$

Therefore

$$
R \left( c , b \right) = \frac { \exp \left[ \beta _ { c } T _ { k } \left( c \right) + \gamma _ { c } S _ { k } + \mu _ { c } \right] } { \exp \left[ \beta _ { b } T _ { k } \left( b \right) + \gamma _ { b } S _ { k } + \mu _ { b } \right] } = \exp \left[ \left( \beta _ { c } T _ { k } \left( c \right) + \gamma _ { c } S _ { k } + \mu _ { c } \right) - \left( \beta _ { b } T _ { k } \left( b \right) + \gamma _ { b } S _ { k } + \mu _ { b } \right) \right] .
$$

The conditional probability of $X _ { k } = c$ given all other variables is

$$
P \left( X _ { k } = c \mid x _ { j } : j \neq k \right) = \frac { \exp { \left[ \beta _ { c } T _ { k } \left( c \right) + \gamma _ { c } S _ { k } + \mu _ { c } \right] } } { \sum _ { a = 0 } ^ { C } \exp { \left[ \beta _ { a } T _ { k } \left( a \right) + \gamma _ { a } S _ { k } + \mu _ { a } \right] } } .
$$

With $\beta _ { 0 } = \gamma _ { 0 } = \mu _ { 0 } = 0$ , the denominator includes the baseline category. We can write

$$
P \left( X _ { k } = c \mid x _ { j } : j \neq k \right) = \exp \left[ \beta _ { c } T _ { k } \left( c \right) + \gamma _ { c } S _ { k } + \mu _ { c } - Q _ { k } \right] ,
$$

where

$$
Q _ { k } = \ln \left( \sum _ { a = 0 } ^ { C } \exp \left[ \beta _ { a } T _ { k } \left( a \right) + \gamma _ { a } S _ { k } + \mu _ { a } \right] \right) .
$$

Thus each variable $X _ { k }$ is associated in PARM with a multinomial logistic regression on the similarities with the other variables.

## 4 Estimation

Estimation can proceed by maximizing the pseudo-likelihood under the working assumption that the variables are conditionally independent given the others.

$$
L = \sum _ { i = 1 } ^ { K } \sum _ { c = 0 } ^ { C } 1 _ { \left\{ c \right\} } \left( x _ { i } \right) \ln P \left( X _ { i } = c \mid x _ { j } : j \neq i \right) .
$$

Equivalently,

$$
L = \sum _ { i = 1 } ^ { K } \sum _ { c = 0 } ^ { C } 1 _ { \left\{ c \right\} } \left( x _ { i } \right) \left[ \beta _ { c } T _ { i } \left( c \right) + \gamma _ { c } S _ { i } + \mu _ { c } - Q _ { i } \right] ,
$$

where

$$
T _ { i } \left( c \right) = \sum _ { j = 1 , j \neq i } ^ { K } s \left( e _ { i } , e _ { j } \right) 1 _ { \left\{ c \right\} } ( x _ { j } )
$$

and

$$
S _ { i } = \sum _ { j \neq i } s \left( e _ { i } , e _ { j } \right) .
$$

As mentioned in the previous section, the normalizing term is

$$
Q _ { i } = \ln \left( \sum _ { a = 0 } ^ { C } \exp \left[ \beta _ { a } T _ { i } \left( a \right) + \gamma _ { a } S _ { i } + \mu _ { a } \right] \right) .
$$

## 4.1 Derivatives with respect to the model parameters Define

$$
p _ { i c } = P \left( X _ { i } = c \mid x _ { j } : j \neq i \right) = \frac { \exp \left[ \beta _ { c } T _ { i } \left( c \right) + \gamma _ { c } S _ { i } + \mu _ { c } \right] } { \sum _ { a = 0 } ^ { C } \exp \left[ \beta _ { a } T _ { i } \left( a \right) + \gamma _ { a } S _ { i } + \mu _ { a } \right] } .
$$

For $c = 1 , \ldots , C$ , the first derivatives of the pseudo-likelihood are

$$
\frac { \partial L } { \partial \beta _ { c } } = \sum _ { i = 1 } ^ { K } \left( 1 _ { \{ c \} } \left( x _ { i } \right) - p _ { i c } \right) T _ { i } \left( c \right) ,
$$

$$
\frac { \partial L } { \partial \gamma _ { c } } = \sum _ { i = 1 } ^ { K } \left( 1 _ { \{ c \} } \left( x _ { i } \right) - p _ { i c } \right) S _ { i } ,
$$

and

$$
\frac { \partial L } { \partial \mu _ { c } } = \sum _ { i = 1 } ^ { K } \left( 1 _ { \{ c \} } \left( x _ { i } \right) - p _ { i c } \right) .
$$

For $c = 1 , \ldots , C$ , the second derivatives are

$$
\frac { \partial ^ { 2 } L } { \partial \beta _ { c } ^ { 2 } } = - \sum _ { i = 1 } ^ { K } p _ { i c } \left( 1 - p _ { i c } \right) \left[ T _ { i } \left( c \right) \right] ^ { 2 } ,
$$

$$
\frac { \partial ^ { 2 } L } { \partial \gamma _ { c } ^ { 2 } } = - \sum _ { i = 1 } ^ { K } p _ { i c } \left( 1 - p _ { i c } \right) \left[ S _ { i } \right] ^ { 2 } ,
$$

$$
\frac { \partial ^ { 2 } L } { \partial \mu _ { c } ^ { 2 } } = - \sum _ { i = 1 } ^ { K } p _ { i c } \left( 1 - p _ { i c } \right) ,
$$

$$
\frac { \partial ^ { 2 } L } { \partial \beta _ { c } \partial \gamma _ { c } } = - \sum _ { i = 1 } ^ { K } p _ { i c } \left( 1 - p _ { i c } \right) T _ { i } \left( c \right) S _ { i } ,
$$

$$
\frac { \partial ^ { 2 } L } { \partial \beta _ { c } \partial \mu _ { c } } = - \sum _ { i = 1 } ^ { K } p _ { i c } \left( 1 - p _ { i c } \right) T _ { i } \left( c \right) ,
$$

$$
\frac { \partial ^ { 2 } L } { \partial \gamma _ { c } \partial \mu _ { c } } = - \sum _ { i = 1 } ^ { K } p _ { i c } \left( 1 - p _ { i c } \right) S _ { i } .
$$

For $c , d \in \{ 1 , \ldots , C \}$ with $c \neq d ,$

$$
\frac { \partial ^ { 2 } L } { \partial \beta _ { c } \partial \beta _ { d } } = \sum _ { i = 1 } ^ { K } p _ { i c } p _ { i d } T _ { i } \left( c \right) T _ { i } \left( d \right) ,
$$

$$
\frac { \partial ^ { 2 } L } { \partial \gamma _ { c } \partial \gamma _ { d } } = \sum _ { i = 1 } ^ { K } p _ { i c } p _ { i d } \left[ S _ { i } \right] ^ { 2 } ,
$$

$$
\frac { \partial ^ { 2 } L } { \partial \mu _ { c } \partial \mu _ { d } } = \sum _ { i = 1 } ^ { K } p _ { i c } p _ { i d } ,
$$

$$
\frac { \partial ^ { 2 } L } { \partial \beta _ { c } \partial \gamma _ { d } } = \sum _ { i = 1 } ^ { K } p _ { i c } p _ { i d } T _ { i } \left( c \right) S _ { i } ,
$$

$$
\frac { \partial ^ { 2 } L } { \partial \beta _ { c } \partial \mu _ { d } } = \sum _ { i = 1 } ^ { K } p _ { i c } p _ { i d } T _ { i } \left( c \right) ,
$$

$$
\frac { \partial ^ { 2 } L } { \partial \gamma _ { c } \partial \mu _ { d } } = \sum _ { i = 1 } ^ { K } p _ { i c } p _ { i d } S _ { i } .
$$

These derivative expressions can be used with standard Newton-type optimization routines.

## 4.2 Implementation and Computational Details

The parameter estimation described above is implemented in a $\mathrm { P y }$ thon script that follows the pseudo-likelihood approach. The script uses the sentencetransformers library (Reimers $\&$ Gurevych, 2019) to generate text embeddings from the constructed responses, and computes pairwise cosine similarities, scaled by $1 / K$ as defined earlier. The features $T _ { i } ( c )$ and $S _ { i }$ are constructed from these similarities. The optimization is performed using the L-BFGS-B algorithm (Byrd et al., 1995) as implemented in the scipy.optimize.minimize function (Virtanen et al., 2020). The objective function is the negative log pseudo-likelihood, and its gradient is computed analytically and supplied to the optimizer.

## 4.2.1 Tuning the Similarity Power Transformation

As an optional preprocessing step (described in Section 2.1), the raw cosine similarities are min–max normalized to [0, 1] and then raised to a power p to adjust the discrimination between score levels. In the implementation, the exponent p is treated as a tunable hyperparameter. Two optimization strategies are provided:

• Grid search: the user supplies a list of candidate values $( \mathrm { e . g . } , \ p \ \in$ {0.5, 1.0, 2.0, 4.0}). The model is estimated for each value, and the one that maximizes a pre-specified performance metric is selected.

• Continuous optimization: the user supplies an interval $[ p _ { \mathrm { m i n } } , p _ { \mathrm { m a x } } ]$ and a tolerance. A bounded one-dimensional optimizer (Brent’s method, as implemented in scipy.optimize.minimize\_scalar) then searches for the p that maximizes the chosen metric. This typically converges in 20-40 evaluations and often finds a more precise optimum than a coarse grid.

The objective metric is computed on the training data (the same data used for estimation) and can be set to any of the evaluation measures available in the software, such as overall accuracy, Cohen’s kappa, or the near-agreement rate (predictions within ±1 category). This choice is specified in the configuration file (defaults.json), along with the power setting and the tuning method. Once the optimal p is found, the final model is re-estimated with that value, and all outputs (predictions, features, and parameter estimates) are generated accordingly.

This tuning step is particularly useful when the raw similarities do not adequately separate the scoring categories; the power transformation sharpens the semantic signal, as demonstrated empirically in Section 5.4. The flexible tuning mechanism allows the PARM to adapt to diferent response distributions and scoring rubrics, making it a robust tool for reliability auditing across diverse assessment contexts.

## 4.2.2 Weighted Estimation and Stability

To address class imbalance in the rating categories, the script employs balanced class weights. Each observation is assigned a weight inversely proportional to the frequency of its category, so that the total contribution of each category to the pseudo-likelihood is approximately equal. This weighting improves the model’s ability to predict minority categories, at a possible cost in overall accuracy.

Numerical stability is ensured by clipping the predicted probabilities to a small positive value (e.g., 10<sup>−15</sup>) before taking logarithms, and by bounding the parameters to a reasonable range (e.g., [−1000, 1000]). These measures prevent underflow and divergence during optimization.

The implementation makes use of several Python libraries: NumPy (Harris et al., 2020) for array operations, pandas (McKinney, 2010) for data handling, and scikit-learn (Pedregosa et al., 2011) for evaluation metrics. The script was developed using a “vibe-coding” approach, in which a large language model (LLM) assisted in writing and refining the code based on high-level descriptions of the statistical model and desired functionality.

## 5 Example Data Analysis

The following section shows two example data analyses with the proposed PARM model. The first is a more standard dataset that includes LLM simulated responses to a short answer item that is scores in three categories 0, 1, 2. The second example comes from a published dataset with substantially longer responses, using responses to essay 1, as described in Li et al. (2023). These two examples show the utility of the proposed model for short answer constructed responses. Examples with more data are possible, the python script available from the author was tested with up to 6000 responses.

## 5.1 Example 2-Point Item Data

To illustrate the proposed Potts auditing-reliability model (PARM), we constructed a dataset of K = 750 natural-language responses to the open-ended question: “Describe two things that any plant needs to grow.” Each response was independently scored by a human rater using a three-level rubric:

• Score 0: completely incorrect answer (no valid plant need mentioned);

• Score 1: answer mentions exactly one distinct correct need, or repeats the same need twice;

• Score 2: answer provides two distinct correct needs (e.g., water and sunlight, soil and air, nutrients and warmth, etc.).

The resulting score distribution was nearly balanced: 252 responses were scored 0, 248 scored 1, and 250 scored 2, providing a challenging multi-class classification setting for reliability auditing. The responses had an average length of 46.2 characters (SD = 23.2), reflecting the brief but varied nature of the open-ended answers.

Text embeddings were generated using the Sentence-BERT model distiluse-base-multilingual-cased-v2 (Reimers & Gurevych, 2019), yielding a 750 × 512 embedding matrix. Pairwise cosine similarities (scaled by 1/K) showed a mean of 0.2935 and a standard deviation of 0.2090, indicating moderate semantic overlap among the responses.

The PARM was estimated via pseudo-likelihood optimization with balanced class weights; the optimization converged in approximately 0.05 seconds. With category 0 as the baseline, the estimated parameters were:

$$
\begin{array} { c c c } { \beta _ { 1 } = 0 . 2 2 6 , } & { \gamma _ { 1 } = - 0 . 0 6 5 , } & { \mu _ { 1 } = - 1 . 1 3 7 , } \\ { \beta _ { 2 } = 0 . 5 8 4 , } & { \gamma _ { 2 } = - 0 . 2 2 3 , } & { \mu _ { 2 } = - 7 . 1 3 7 . } \end{array}
$$

Table 1 presents the confusion matrix between the observed human scores and the predicted scores from the PARM. The model achieved outstanding predictive performance: overall accuracy was 0.971, with macro-averaged precision, recall, and F1 score all at 0.971, and a Cohen’s kappa of 0.956. Notably, the nearagreement rate, defined as predictions falling within ±1 category of the true score, was perfect (1.000). This indicates that while occasional misclassifications occur, they are almost exclusively between adjacent score levels (e.g., 0 vs. 1 or 1 vs. 2), preserving the ordinal structure of the scoring rubric.

Table 1: Confusion matrix of observed (rows) vs. predicted (columns) scores.
<table><tr><td rowspan="2">Observed</td><td colspan="3">Predicted</td><td rowspan="2">Total</td></tr><tr><td>0</td><td>1</td><td>2</td></tr><tr><td>0</td><td>244</td><td>8</td><td>0</td><td>252</td></tr><tr><td>1</td><td>6</td><td>235</td><td>7</td><td>248</td></tr><tr><td>2</td><td>0</td><td>1</td><td>249</td><td>250</td></tr></table>

The high accuracy, excellent kappa, and perfect near-agreement demonstrate that the PARM framework, when equipped with LLM-derived semantic similarities, can efectively recover human-like scoring patterns. This makes it a promising tool for reliability auditing in educational and assessment contexts where open-ended responses are common.

## 5.2 Example: AERA Essay 1 Data

We further evaluated the PARM on the AERA (Automated Essay scoring and Reasoning Assessment) dataset (REFERENCE), specifically focusing on responses to Essay 1. The task involves scoring student answers to a science or reasoning prompt on a 4-point rubric, with categories ranging from 0 (lowest quality) to 3 (highest quality). The dataset contains N = 1,338 responses, with an average answer length of 257.6 characters (SD = 123.7). This represents substantially more elaborate and content-rich writing compared to the brief plant-growth responses in the previous example.

Text embeddings were generated using the same Sentence-BERT model,

distiluse-base-multilingual-cased-v2, yielding a 1,338 × 512 embedding matrix. Pairwise cosine similarities averaged 0.455 (SD = 0.183), indicating greater semantic coherence among the longer, more content-dense responses relative to the previous dataset. The smaller s.d. means that the responses are not as widely distributed as in example 1, likely making discrimination between the four response categories more challenging.

The PARM was estimated via pseudo-likelihood optimization with balanced class weights. With category 0 as the baseline, the estimated parameters were:

$$
\begin{array} { c c c } { \beta _ { 1 } = 0 . 0 9 2 , } & { \gamma _ { 1 } = - 0 . 0 1 6 , } & { \mu _ { 1 } = - 3 . 7 4 6 , } \\ { \beta _ { 2 } = 0 . 1 3 0 , } & { \gamma _ { 2 } = - 0 . 0 3 5 , } & { \mu _ { 2 } = - 6 . 0 2 9 , } \\ { \beta _ { 3 } = 0 . 2 3 8 , } & { \gamma _ { 3 } = - 0 . 0 4 3 , } & { \mu _ { 3 } = - 7 . 4 7 6 . } \end{array}
$$

The increasing $\beta _ { c }$ estimates across categories suggest that higher-scoring responses rely more strongly on semantic similarity with other responses that share the same score level, which is consistent with a well-calibrated scoring rubric.

Table 2 presents the confusion matrix between the observed human scores and the predicted scores. Predictive performance, while lower than in the simpler 3-category task, remained informative: exact accuracy reached 0.476, with a macro-averaged precision of 0.479, recall of 0.496, and F1 of 0.479, alongside a Cohen’s kappa of 0.304. Crucially, the near-agreement rate, defined as predictions falling within ±1 category of the true score, was substantial at 0.890.

Inspection of the confusion matrix confirms that most misclassifications occur between adjacent score levels (e.g., 0 vs. 1, 1 vs. 2, or 2 vs. 3), with only 110 out of 1,338 predictions (8.2%) difering by more than one category. This pattern demonstrates that the PARM framework captures the ordinal structure of the scoring rubric even when exact agreement is more challenging due to the greater subtlety and length of the responses.

Table 2: Confusion matrix for AERA Essay 1: observed (rows) vs. predicted (columns) scores.
<table><tr><td rowspan="2">Observed</td><td colspan="4">Predicted</td><td rowspan="2">Total</td></tr><tr><td>0</td><td>1</td><td>2</td><td>3</td></tr><tr><td>0</td><td>188</td><td>84</td><td>22</td><td>10</td><td>304</td></tr><tr><td>1</td><td>78</td><td>111</td><td>89</td><td>66</td><td>344</td></tr><tr><td>2</td><td>26</td><td>70</td><td>156</td><td>167</td><td>419</td></tr><tr><td>3</td><td>11</td><td>12</td><td>66</td><td>182</td><td>271</td></tr></table>

## 5.3 Example: AERA Essay 6 Data

We further evaluated the PARM on a diferent prompt from the AERA dataset, specifically Essay 6 (Li et al., 2023). This task also uses a 4-point scoring rubric with categories ranging from 0 (lowest quality) to 3 (highest quality). The dataset contains N = 1,438 responses, with an average answer length of 141.4 characters (SD = 126.4), reflecting moderately detailed constructed responses.

A notable feature of this dataset is the severe class imbalance in the observed scores: category 0 contains 1,212 responses (84.3%), while category 3 contains only 41 (2.9%). This imbalance poses a substantial challenge for any classification or reliability model, as minority categories contribute very little to the unweighted likelihood.

Text embeddings were generated using the same Sentence-BERT model,

distiluse-base-multilingual-cased-v2, yielding a 1,438 × 512 embedding matrix. Pairwise cosine similarities averaged 0.285 (SD = 0.181), which is somewhat lower than in the previous examples, indicating more heterogeneous response content. The PARM was estimated via pseudo-likelihood optimization with balanced class weights to counteract the severe imbalance. With category

0 as the baseline, the estimated parameters were:

$$
\begin{array} { c c c } { \beta _ { 1 } = 0 . 2 0 1 , } & { \gamma _ { 1 } = - 0 . 0 1 9 , } & { \mu _ { 1 } = - 2 . 2 4 7 , } \\ { \beta _ { 2 } = 0 . 4 3 2 , } & { \gamma _ { 2 } = - 0 . 0 1 5 , } & { \mu _ { 2 } = - 4 . 5 3 6 , } \\ { \beta _ { 3 } = 0 . 8 0 1 , } & { \gamma _ { 3 } = - 0 . 0 1 8 , } & { \mu _ { 3 } = - 8 . 5 6 8 . } \end{array}
$$

The strictly increasing $\beta _ { c }$ estimates again confirm the ordinal nature of the rubric, with higher categories showing stronger reliance on semantic similarity with responses sharing the same score.

Table 3 presents the confusion matrix between the observed human scores and the predicted scores. The model achieved an overall accuracy of 0.809 and a near-agreement rate (predictions within ±1 category) of 0.956. However, due to the severe class imbalance, the macro-averaged precision, recall, and F1 were 0.457, 0.567, and 0.493, respectively, with a Cohen’s kappa of 0.467. Inspection of the confusion matrix reveals that while the model correctly classifies the vast majority of low-scoring (0) responses, it struggles with the minority categories (especially categories 2 and 3), where many predictions are shifted to adjacent lower categories. Despite this, the high near-agreement rate demonstrates that the PARM framework captures the ordinal structure of the scoring rubric even under extreme imbalance, making it a robust tool for reliability auditing in realistic, skewed scoring settings.

Table 3: Confusion matrix for AERA Essay 6: observed (rows) vs. predicted (columns) scores.
<table><tr><td rowspan="2">Observed</td><td colspan="4">Predicted</td><td rowspan="2">Total</td></tr><tr><td>0</td><td>1</td><td>2</td><td>3</td></tr><tr><td>0</td><td>1047</td><td>124</td><td>33</td><td>8</td><td>1212</td></tr><tr><td>1</td><td>17</td><td>79</td><td>15</td><td>17</td><td>128</td></tr><tr><td>2</td><td>0</td><td>12</td><td>17</td><td>28</td><td>57</td></tr><tr><td>3</td><td>0</td><td>5</td><td>16</td><td>20</td><td>41</td></tr></table>

Comparing these results to those from AERA Essay 1, we observe that Essay 6 yielded substantially higher exact accuracy (0.809 vs. 0.476) and nearagreement (0.956 vs. 0.890), despite the severe class imbalance. This suggests that the rubric for Essay 6 may be applied more consistently, or that the response content is more easily diferentiated by the semantic embeddings. These findings further underscore the flexibility of the PARM framework: it performs reliably across diverse scoring tasks, from balanced short-answer items to imbalanced, longer constructed responses.

## 5.4 Impact of Similarity Transformation: AERA Essay Set 5

The practical utility of the similarity normalization and power transformation is demonstrated by comparing two runs of the PARM on AERA Essay Set 5. The first run used raw cosine similarities without any scaling $( p = 1$ , no min-max normalization), while the second applied min-max normalization followed by a strong power transformation $( p = 8 )$ . As shown in Table 4, the transformation markedly improved every performance metric. Accuracy increased from 0.544 to 0.709, Cohen’s kappa nearly doubled from 0.179 to 0.345, and the macro F1 score rose from 0.296 to 0.462. The near-agreement rate also improved substantially, from 0.915 to 0.966.

Beyond the overall performance gains, the transformation produced notable changes in the estimated $\beta _ { c }$ parameters (Table 5). In the raw run, the parameters were $\beta _ { 1 } = - 0 . 0 1 3 , \beta _ { 2 } = 0 . 3 5 0$ , and $\beta _ { 3 } = 0 . 6 8 7$ , indicating that agreement on category 1 had a slightly negative association while higher categories showed positive associations. After applying the power transformation, the magnitudes increased substantially to $\beta _ { 1 } = 0 . 1 3 1 , \beta _ { 2 } = 1 . 1 0 0$ , and $\beta _ { 3 } = 2 . 0 1 5$ . This increase in magnitude suggests that the transformation, by amplifying strong similarities and suppressing moderate ones, sharpens the semantic signal and allows the model to assign more distinctive weights to agreement on diferent score levels. Importantly, the PARM does not assume or require any ordering constraint on the $\beta _ { c }$ parameters; they are estimated freely and simply reflect the empirical agreement patterns in the data. The fact that the transformation yields larger absolute parameter values indicates improved separation between categories, which likely contributes to the observed gains in predictive accuracy and Cohen’s kappa.

A practical implication of this comparison is that the power constant p can be treated as a tunable hyperparameter. In practice, one may select p via cross-validation or based on criteria such as maximizing near-agreement rate or minimizing classification error. This flexibility allows the PARM to adapt to diferent datasets and response distributions, making it a more robust tool for reliability auditing across diverse assessment contexts.

Table 4: Performance comparison on AERA Essay Set 5 with and without similarity transformation.
<table><tr><td>Specification</td><td>Accuracy</td><td>Kappa</td><td>Macro F1</td><td>Near-Agreement</td></tr><tr><td>Raw  $( p = 1 , \mathrm { n o \ n o r m } )$ </td><td>0.544</td><td>0.179</td><td>0.296</td><td>0.915</td></tr><tr><td>Normalized  $( p = 8 )$ </td><td>0.709</td><td>0.345</td><td>0.462</td><td>0.966</td></tr></table>

Table 5: Estimated $\beta _ { c }$ parameters for AERA Essay Set 5 under the two specifications.
<table><tr><td>Specification</td><td> $\overline { { \beta _ { 1 } } }$ </td><td> $\overline { { \beta _ { 2 } } }$ </td><td> $\beta _ { 3 }$ </td></tr><tr><td>Raw  $( p = 1$  , no norm) Normalized  $( p = 8 )$ </td><td>-0.013 0.131</td><td>0.350 1.100</td><td>0.687 2.015</td></tr></table>

## 6 Conclusions

This paper introduced the Potts auditing-reliability model (PARM), a framework for multi-category scoring reliability that leverages LLM-derived semantic similarities. The model defines a joint distribution over scored responses using category-specific agreement indicators and embedding-based pairwise weights. Its formulation naturally accommodates any number of ordered or unordered rating categories, making it broadly applicable to scoring tasks with multiple grade levels.

We estimated the model via pseudo-likelihood with balanced class weights and analytical gradients, enabling eficient L-BFGS-B optimization. Empirical evaluation on two datasets demonstrated its practical utility. On a three-category plant-growth task, the PARM achieved excellent performance (accuracy = 0.971, $\kappa = 0 . 9 5 6$ , near-agreement = 1.000). On the more challenging four-category AERA Essay 1 dataset, it remained informative (accuracy = 0.476, $\kappa = 0 . 3 0 4$ near-agreement = 0.890). In both cases, the vast majority of misclassifications occurred between adjacent score levels, confirming that the PARM captures the ordinal structure of rubrics without imposing rigid equidistance assumptions.

The model parameters are interpretable: $\beta _ { c }$ reflects the influence of agreement on category c, while $\gamma _ { c }$ and $\mu _ { c }$ capture total semantic similarity efects and category intercepts. In both examples, $\beta _ { c }$ increased with higher categories, suggesting that high-scoring responses exhibit stronger semantic coherence with responses sharing the same score.

A practical preprocessing step involves normalizing the raw cosine similarities via min–max scaling to the unit interval, followed by an optional power transformation with exponent p. This transformation allows the model to emphasize or downweight moderate similarities, providing additional flexibility in capturing the semantic structure of the response corpus. The choice of p can be treated as a hyperparameter, and its efect on predictive performance and parameter interpretability warrants further investigation.

The PARM provides a flexible and extensible framework for reliability auditing. Its formulation in terms of pairwise agreements and category-specific weights readily generalizes to multiple raters by treating each rater’s score as a separate variable, and to hierarchical rating designs through random efects for raters or items. Future work will explore these extensions, along with applications to larger datasets and tasks requiring fine-grained semantic distinctions.

## References

Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). A limited memory algorithm for bound constrained optimization. SIAM Journal on Scientific Computing, 16(5), 1190–1208.

Epskamp, S., Maris, G., Waldorp, L. J., & Borsboom, D. (2018). Network psychometrics. In P. Irwing, T. Booth, & D. J. Hughes (Eds.), The Wiley handbook of psychometric testing: A multidisciplinary reference on survey, scale and test development (pp. 953–986). Wiley Blackwell. DOI:10.1002/9781118489772.ch30

Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. Nature, 585(7825), 357–362.

Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. Proceedings of the National Academy of Sciences, 79(8), 2554–2558.

Ising, E. (1925). Beitrag zur Theorie des Ferromagnetismus. Zeitschrift für Physik 31.1 (1925), pp. 253– 258.

Li, J., Gui, L., Zhou, Y., West, D., Aloisi, C., & He, Y. (2023). Distilling Chat-GPT for explainable automated student answer assessment. In Findings of the Association for Computational Linguistics: EMNLP 2023 (pp. 6007–6026). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.findingsemnlp.399

Li, J. (2024). AERA: A dataset to enable LLMs for explainable student answer scoring [Data set]. Hugging Face. https://huggingface.co/datasets/jiazhengli/AERA

McKinney, W. (2010). Data structures for statistical computing in Python. Proceedings of the 9th Python in Science Conference, 445, 51–56.

Molenaar, P. C. M. (2004). A Manifesto on Psychology as Idiographic Science: Bringing the Person Back Into Scientific Psychology, This Time Forever. Measurement: Interdisciplinary Research and Perspectives, 2(4), 201–218. DOI:10.1207/s15366359mea0204\_1

Rasch, G. (1960). Probabilistic models for some intelligence and attainment tests. Danish Institute for Educational Research.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. arXiv preprint arXiv:1908.10084.

Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.

Potts, R.B., (1952). Some generalized order-disorder transformations. Mathematical proceedings of the cambridge philosophical society. Vol. 48. 1. Cambridge University Press. 1952, pp. 106–109.

Razaee, Z. S., & Amini, A. A. (2020). The Potts-Ising model for discrete multivariate data. Advances in Neural Information Processing Systems, 33, 13727– 13737.

https://proceedings.neurips.cc/paper/2020/hash/9e5f64cde99af96fdca0e02a3d24faec-Abstract.html

Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature Methods, 17(3), 261–272.

von Davier, M. (2016). Rasch model. In W. J. van der Linden (Ed.), Handbook of item response theory: Volume one: Models (pp. 31–48). Chapman and Hall/CRC.

von Davier, M. (2018). Diagnosing Diagnostic Models: From von Neumann’s Elephant to Model Equivalencies and Network Psychometrics. Measurement: Interdisciplinary Research and Perspectives, 16(1), 59–70. DOI:10.1080/15366367.2018.1436827

von Davier, M. (2026). Integrating network psychometrics and LLMs: The Ising-Embeddings-Model applied to reliability auditing (Version 2). arXiv. https://doi.org/10.48550/arXiv.2608.26790