# EXACT SEMANTIC READOUT FROM COMPRESSED VECTOR REPRESENTATIONS<sup>∗</sup>

PREPRINT

Daniel Quigley   
Center for Possible Minds   
Indiana University Bloomington   
Bloomington, IN 47408   
dgquigle@iu.edu

September 17, 2026

## ABSTRACT

We characterize when compressed vector representations admit exact linear or affine readouts of a finite lexicon’s truth conditions: one fixed map per predicate, sending each entity vector to the corresponding truth vector. A necessary and sufficient row-space condition determines existence; the augmented truth matrix has rank r, giving minimum dimension r in the linear case, and r − 1 in the affine. Exact readouts return values in a shared truth basis on which Boolean connectives act unchanged; separability alone requires an intervening threshold. For binary relations, exact bilinear readout of identity or strict total order requires linearly independent entity vectors. Experiments with GloVe and word2vec distinguish exact affine recovery, linear separability, and held-out prediction: most predicates are strictly separable, but none admits an exact affine readout from the pretrained embeddings. Supervised transductive training attains exact affine recovery to numerical precision at every tested dimension meeting the bound. At the embeddings’ original dimension, geometries constrained to exact linear recovery retain 98–99 percent of the pretrained variance on the feature norms, and 80–83 percent on the WordNet lexicon.

Keywords formal semantics · vector logic · semantic space · encoding · embedding · mathematical linguistics

## 1 Introduction

Montagovian semantics represents entities and predicates in typed domains, with composition governed by the model’s functions [20, 9]. Distributional embeddings represent lexical items as vectors, whose geometry reflects patterns of use [4]. When these vectors represent the entities of a finite semantic model, which predicates can be recovered by linear readouts, and what does compression prevent?

A vector logic for formal semantics [25, 26] supplies an exact construction: each primitive domain element receives its own basis vector, and semantic functions extend linearly from free carriers, preserving composition along primitive intermediate types. Linear independence makes these lifts possible. Learned embeddings generally have fewer dimensions than entities, and, therefore, introduce linear dependences that may obstruct the lifts.

An embedding lookup with matrix $\mathbf { E } \in \mathbb { R } ^ { V \times d }$ is equivalent, on one-hot inputs, to a bias-free linear map [27]. It, therefore, factors through the free entity carrier as $\mathbf { E } ^ { \dagger } : \mathbb { R } ^ { V } \xrightarrow { } \mathbb { R } ^ { d }$ . This identifies two regimes: afree geometry has linearly independent entity vectors, and admits every semantic lift; a compressed geometry has dependent vectors, and admits only those lifts compatible with its dependences. Compression need not erase entity identity: distinct columns still permit arbitrary predicate lookup on a finite domain; we are interested, then, in which predicates remain linearly recoverable.

For a monadic lexicon with truth matrix T, we show that exact readouts into a shared truth basis exist precisely when the geometry’s row space contains row[T; 1]. The rank of this augmented truth matrix gives the minimum dimension, while its kernel specifies the admissible dependences among entity vectors. A restricted lexicon can, therefore, admit exact compression, although requiring every predicate forces the free regime. Once atomic readouts return the truth basis, Boolean connectives act unchanged.

Exact readout is stronger than linear separability: a score may distinguish true instances from false ones without, itself, returning their truth values. This distinction connects linear probing [3] to the vector logic. Whether such a readout exists, and whether it can be learned from a subset of entities, are separate questions. We test exact recovery, separability, and held-out prediction on distributional embeddings, then examine how much pretrained variance survives supervised training toward exact recovery. Figure 1 summarizes the program and the dependencies among these questions.

![](images/89943a37bf47af5876a02308d4ba788a0907399fab80ecfabda665696459164c.jpg)  
Figure 1: Dependencies among the formal results and empirical analyses.

## 2 Vector logic

We recall from [25] the definitions and the homomorphism theorem, and from [26] the index-sort material. We refrain from re-proving the relevant content; see, instead, papers proper therein.

## 2.1 Extensional models and free carrier

Types are generated from e and t by $\langle \sigma , \tau \rangle$ . A typed extensional model $\mathcal { M } _ { e x t } = \langle ( D _ { \tau } ) _ { \tau } , \mathcal { T } \rangle$ has entity domain $\mathcal { D } _ { \epsilon }$ , truth domain $\bar { \mathcal { D } _ { t } } = \{ 1 , 0 \}$ , function domains $\dot { \mathcal { D } } _ { \langle \sigma , \tau \rangle } \dot { = } \mathcal { D } _ { \tau } ^ { \mathcal { D } _ { c } }$ , and an interpretation $\boldsymbol { \mathcal { T } } ;$ denotations · follow the recursion of [9]. Throughout, let $\mathcal { D } _ { e }$ be $\mathrm { f i n i t e } ^ { 2 }$ , with $| \mathcal { D } _ { e } | = V$ and elements $d _ { 1 } , \ldots , d _ { V }$

The vector space model $\mathcal { M } _ { \mathcal { S } }$ assigns to each domain a real vector space $\boldsymbol { S _ { D _ { \tau } } }$ and an injection $h _ { \tau } : { \mathcal { D } } _ { \tau } \to S _ { \mathcal { D } _ { \tau } }$ <sub>τ</sub> . The construction takes the form

$$
h _ { e } ( d _ { i } ) = \mathbf { e } _ { i } \in \mathbb { R } ^ { V } , \qquad h _ { t } ( 1 ) = \mathbf { b } _ { 1 } = \left[ 0 \right] , \qquad h _ { t } ( 0 ) = \mathbf { b } _ { 0 } = \left[ 0 \right] ,
$$

where $\mathbf { e } _ { i }$ is the i-th standard basis vector, and for a function type sends $f \in \mathcal { D } _ { \langle \sigma , \tau \rangle }$ to the linear map $L _ { f } ,$ , determined on the basis $\{ \mathbf { e } _ { a } \} _ { a \in \mathcal { D } _ { c } }$ of the free carrier of $\sigma$ by $L _ { f } \mathbf { e } _ { a } = h _ { \tau } ( f ( a ) )$ ; a linear map is determined by its values on a basis, so this fixes $L _ { f }$ uniquely. Note that [25] embeds every domain, function-type domains included, by basis vectors, and defines the lift $h _ { f }$ pointwise on the image of $h _ { \sigma }$ through the left inverse $h _ { \sigma } ^ { - 1 }$ ; [26] attaches to each type a free carrier $\mathcal { F } _ { \tau }$ , with basis indexed by $\mathcal { D } _ { \tau }$ , and an operator carrier $\bar { \mathcal { S } } _ { \tau }$ , with $S _ { \langle \sigma , \tau \rangle } = \mathrm { H o m } ( \mathcal { F } _ { \sigma } , S _ { \tau } )$ , so that the free carrier stands in argument position, and the two carriers coincide on primitive types. We adopt the second convention, since the results below concern linear readouts on the entity space, whose type is primitive, and we call the family $\{ \mathcal { F } _ { \tau } \}$ the free carrier: elements of primitive domains go to basis vectors, and the primitive spaces are free on their domains. In particular, a monadic predicate $P \in \mathcal { D } _ { \langle e , t \rangle }$ becomes the $2 \times V$ matrix

$$
\mathbf { M } _ { P } = \big [ h _ { t } ( P ( d _ { 1 } ) ) \ \cdots \ h _ { t } ( P ( d _ { V } ) ) \big ] ,
$$

whose i-th column is the truth vector that P assigns to $d _ { i } ,$ , and $\mathbf { M } _ { P } \mathbf { e } _ { i } = h _ { t } ( P ( d _ { i } ) )$ is functional application. Truthfunctional connectives become fixed matrices on tensor powers of the truth space, an n-ary connective c acting as a $2 \times 2 ^ { n }$ matrix $\mathbf { M } _ { c }$ on $\otimes _ { i } h _ { t } ( t _ { i } )$ ; for instance

$$
\mathbf { M } _ { \neg } = \left[ \mathbf { 0 } \quad 1 \right] , \qquad \mathbf { M } _ { \wedge } = \left[ \mathbf { 1 } \quad 0 \quad 0 \quad 0 \right] ,
$$

in the ordering $\mathbf { b } _ { 1 } \otimes \mathbf { b } _ { 1 } , \mathbf { b } _ { 1 } \otimes \mathbf { b } _ { 0 } , \mathbf { b } _ { 0 } \otimes \mathbf { b } _ { 1 } , \mathbf { b } _ { 0 }$ ⊗ b<sub>0</sub> of the tensor basis, after [19, 31].

## 2.2 Homomorphism theorem

## Theorem 2.1

For every extensional model $\mathcal { M } _ { e x t }$ , there exist injections $\{ h _ { \tau } \}$ into vector spaces $\{ S _ { D _ { \tau } } \}$ such that every semantic function $f : { \mathcal { D } } _ { \sigma } \to { \mathcal { D } } _ { \mathit { \Pi } }$ has a unique linear lift $\overset { \cdot \mathrm { ~ \textstyle ~ - ~ } } { L _ { f } } : \mathcal { F } _ { \sigma } \overset { } { \to } \overset { \cdot \mathrm { ~ \textstyle ~ - ~ } } { S _ { D _ { \tau } } }$ from the free carrier of σ, with $L _ { f } ( \mathbf { b } _ { a } ) \stackrel { \cdot } { = } h _ { \tau } ( f ( a ) )$ for every $a \in \mathcal { D } _ { \sigma }$ , where $ { \mathbf { b } } _ { a }$ is the free encoding of $^ { a , }$ which coincides with $h _ { \sigma } ( a )$ when σ is primitive; n-ary functions lift to multilinear maps on the free carriers, and composition of semantic functions corresponds to composition of lifts along primitive intermediate types.

The square

$$
\begin{array} { r } { \mathcal { D } _ { \sigma } \xrightarrow { f \big . } \mathcal { D } _ { \tau } } \\ { \left. h _ { \sigma } \right\downarrow \overline { { \left. \begin{array} { r l r } { h _ { \tau } } & { } & { } \end{array} \right| h _ { \tau } } } } \\ { \mathcal { S } _ { \mathcal { D } _ { \sigma } } \xrightarrow { h _ { \sigma } } \overline { { L _ { f } \big . } } } & { \mathcal { S } _ { \mathcal { D } _ { \tau } } } \end{array}
$$

commutes for every f with primitive argument type, which covers every function below, so the recursive evaluation of any expression in $\mathcal { M } _ { e x t }$ has a step-by-step counterpart in $\mathcal { M } _ { \mathcal { S } }$ once arguments are carried in their free encodings. The injections are non-surjective by design: im $\left( h _ { t } \right) = \mathbf { \bar { \{ b } }  _ { 1 } , \mathbf { b } _ { 0 } \mathbf  \bar { \} }$ is a proper subset of $\mathbb { R } ^ { 2 }$ , and denotation is confined to the images.

Theorem 2.1 is existential: the free carrier is one family of injections for which the lifts exist, and other families are the subject of Section 4. The commuting square for a single function is [25], where the lift is defined on the image alone; linearity on the span for primitive argument types, the multilinear lift of n-ary functions, and the operator carriers of function types follow [26], whose descent theorems determine when a functional of a function domain acts linearly on operator encodings as well: on a power set, exactly the constants, the ultrafilter indicators, and their complements, which on a finite domain are the constants, Montague’s individuals $\lambda P . P ( d )$ , and their negations.

## 2.3 Index sorts

The intensional layer adjoins index sorts (worlds and times, among others), collected in a compound index space $\begin{array} { r } { S = \prod _ { \sigma } \mathcal { D } _ { \sigma } } \end{array}$ , with its own free carrier $h _ { S } ( s ) = { \bf e } _ { s }$ , the carrier $\mathcal { F } _ { s }$ of the compound index type s. An intension $g : S  { \mathcal { D } } ,$ <sub>τ</sub> becomes the linear operator ${ \cal S } _ { S }  { \cal S } _ { D _ { \tau } }$ , with $\mathbf { e } _ { s } \mapsto h _ { \tau } ( g ( s ) )$ ; a proposition $\varphi$ becomes $\mathbf { P } _ { \varphi } \in \mathbb { R } ^ { 2 \times | S | }$ with truth profile $\mathbf { v } ( \varphi ) \in \{ 0 , 1 \} ^ { | S | }$ its top row. Over a single discrete sort of worlds $W = \{ w _ { 1 } , \ldots , w _ { n } \}$ with accessibility matrix $\mathbf { A } \in \{ 0 , \overset { \cdot } { 1 } \} ^ { n \times n }$ , the modal operators of [26] read

$$
( \Omega \varphi ) ( w _ { i } ) = 1 \iff ( \mathbf { A } \left( \mathbf { 1 } - \mathbf { v } ( \varphi ) \right) ) _ { i } = 0 , \qquad ( \bigotimes \varphi ) ( w _ { i } ) = 1 \iff ( \mathbf { A } \mathbf { v } ( \varphi ) ) _ { i } > 0 ,\tag{1}
$$

a linear accumulation followed by a decision; when the out-degree de $\begin{array} { r } { \mathrm { g } _ { \mathbf { A } } ( w _ { i } ) = \sum _ { j } { \mathbf { A } } _ { i j } } \end{array}$ is finite, as it is here, the first condition is equivalent to $( \mathbf { A } \mathbf { v } ( \varphi ) ) _ { i } = \mathrm { d e g } _ { \mathbf { A } } ( w _ { i } )$ . Further, operators are defined over measure frames, in which the counting measure of the discrete case is but one choice among others; a finite geometry matrix carries the discrete case with finitely many indices, the case to which we restrict ourselves in Section 7.

```perl
Proposition 3.1 (single token)
For every $i \in \{ 1 , \ldots , V \} , L _ { \mathbf { E } } ( \mathbf { e } _ { i } ) = \ell _ { \mathbf { E } } ( i )$
```

## 3 Embedding lookup on free carrier

The observation that an embedding layer is a linear map on one-hot inputs is due to [27, 28], who also explains why frameworks introduce a transpose and why implementations use the lookup form at all. We develop that here in detail, and apply it to our vector logic.

## 3.1 Setup

Let $V \geq 1$ be the vocabulary size, and $d \geq 1$ the embedding dimension. An embedding matrix is any $\mathbf { E } \in \mathbb { R } ^ { V \times d }$ with rows $\mathbf { E } _ { i } \in \mathbb { R } ^ { 1 \times d }$ and entries $E _ { i j }$ . The lookup is $\ell _ { \mathbf { E } } ( i ) = \mathbf { E } _ { i }$ for $i \in \{ 1 , \ldots , V \}$ , and the bias-free linear layer with weight E is $L _ { \mathbf { E } } ( x ) = x ^ { \top }$ E for $\overset { \cdot } { x } \in \mathbb { R } ^ { V }$ . Outputs are row vectors, such that batches stack vertically. Under the reading of the vocabulary as the entity domain, ${ \bf e } _ { i } = h _ { e } ( d _ { i } )$ and $L _ { \mathbf { E } } ( \mathbf { e } _ { i } ) = ( \mathbf { E } ^ { \top } \mathbf { e } _ { i } ) ^ { \top }$ ; the composite $\mathbf { E } ^ { \top } \circ h _ { e }$ is a second injection of $\mathcal { D } _ { e }$ into a vector space, whenever the rows of E are distinct.

Remark 3.1. The vocabulary of a language model and the entity domain of a model are different objects; the identification ${ \bf e } _ { i } = h _ { e } ( d _ { i } )$ is the vocabulary as a set ofnames, one per entity. Everything below concerns the geometry matrix of Section 4, and holds for whatever the columns index; the vocabulary reading is the instance in which the geometry is a trained embedding matrix.

## 3.2 Forward pass

Proof. Fix a column j. By the definition of the matrix product,

$$
( { \bf e } _ { i } ^ { \top } { \bf E } ) _ { j } = \sum _ { k = 1 } ^ { V } ( { \bf e } _ { i } ) _ { k } E _ { k j } = \sum _ { k = 1 } ^ { V } \delta _ { i k } E _ { k j } = E _ { i j } ,
$$

the jth entry of $\mathbf { E } _ { i }$ . Since j was arbitrary, $\mathbf { e } _ { i } ^ { \top } \mathbf { E } = \mathbf { E } _ { i }$

Corollary 3.1 (batch)

Let $i _ { 1 } , \ldots , i _ { n } \in \left\{ 1 , \ldots , V \right\}$ and let $\mathbf { X } \in \mathbb { R } ^ { n \times V }$ have rows $\mathbf { e } _ { i _ { 1 } } ^ { \top } , \ldots , \mathbf { e } _ { i _ { n } } ^ { \top }$ . Then row t of XE equals $\mathbf { E } _ { i _ { t } }$ for every t, so XE is the vertical stack of $\ell _ { \mathbf { E } } ( i _ { 1 } ) , \ldots , \ell _ { \mathbf { E } } ( i _ { n } )$

Proof. Row t of XE is $\mathbf { e } _ { i _ { t } } ^ { \top } \mathbf { E } = \mathbf { E } _ { i }$ by Proposition 3.1, applied to each row independently; repeated identifiers among $i _ { 1 } , \dots , i _ { n }$ are, likewise, covered. □

Remark 3.2. We should note that, for implementation’s sake, frameworks orient the weight of a linear layer differently. In PyTorch,for example, a bias-free linear layer, with input dimension V and output dimension d, stores W $\doteq \mathbb { R } ^ { d \times \check { V } }$ and computes $x \mapsto \bar { x } \mathbf { W } ^ { \top }$ [24]; [27] identifies this transpose as the usual source of confusion in reproducing the identity in code.

Corollary 3.2 (orientation)

With $\mathbf { W } = \mathbf { E } ^ { \top }$ , the map $x \mapsto x \mathbf { W } ^ { \top }$ agrees with $L _ { \mathbf { E } }$ on $\mathbb { R } ^ { V }$ and, hence, with $\ell _ { \mathbf { E } }$ on one-hot inputs.

Proof. $\mathbf { W } ^ { \top } = \mathbf { E } , \thinspace \thinspace \mathbf { s o } \ x \mathbf { W } ^ { \top } =$ xE for every row vector x; apply Corollary 3.1.

$\mathbf { W } = \mathbf { E } ^ { \top }$ is the geometry matrix: its columns are the entity vectors $\mathbf { E } _ { i } ^ { \top }$ , and the stored parameter of the framework’s linear layer is the geometry itself.

## 3.3 Backward pass

When considering the backward pass, we are, essentially, encountering gradients. The identity extends to gradients with respect to the parameters, which makes the implementations interchangeable during training, and covers accumulation on repeated identifiers. Let L be a scalar loss depending on the parameters only by $\mathbf { Y } \overset { = } \mathbf { X } \mathbf { E } \in \mathbb { R } ^ { n \times d }$ , and write $\mathbf { G } = \^ { \bullet } \partial \mathcal { L } / \partial \mathbf { Y } \in \mathbb { R } ^ { n \times d }$ with rows $\mathbf { G } _ { 1 } , \ldots , \mathbf { G } _ { n }$ . In the lookup, the backward pass is defined as a scatter-add: the gradient with respect to E is initialized to zero, and $\mathbf { G } _ { t }$ is added to row $i _ { t }$ for each t.

Proposition 3.2 (gradient)

Under the linear-layer parameterization, $\partial \mathcal { L } / \partial \mathbf { E } = \mathbf { X } ^ { \top } \mathbf { G }$ , whose row k equals $\textstyle \sum _ { t : i _ { t } = k } \mathbf { G } _ { t }$ , with the empty sum being zero.

Proposition 3.2 coincides with the scatter-add gradient of the lookup implementation.

Proof. Since $\begin{array} { r } { Y _ { t j } = \sum _ { k } X _ { t k } E _ { k j } } \end{array}$ , the chain rule gives

$$
{ \frac { \partial { \mathcal { L } } } { \partial E _ { k j } } } = \sum _ { t = 1 } ^ { n } { \frac { \partial { \mathcal { L } } } { \partial Y _ { t j } } } { \frac { \partial Y _ { t j } } { \partial E _ { k j } } } = \sum _ { t = 1 } ^ { n } G _ { t j } X _ { t k } = ( \mathbf { X } ^ { \top } \mathbf { G } ) _ { k j } .
$$

Because $X _ { t k } = \delta _ { i _ { t } k }$ , the sum over t retains exactly the indices with $i _ { t } = k ,$ , so row k of $\mathbf { X } ^ { \top } \mathbf { G }$ is $\textstyle \sum _ { t : i _ { t } = k } \mathbf { G } _ { t }$ . The scatter-add produces the same row by construction: it deposits $\mathbf { G } _ { t }$ into row $i _ { t }$ for each t, and leaves the remaining rows at zero. □

Under Corollary 3.2, the gradient with respect to $\mathbf { W } = \mathbf { E } ^ { \top }$ is $( \mathbf { X } ^ { \top } \mathbf { G } ) ^ { \top } = \mathbf { G } ^ { \top } \mathbf { X } ;$ the identification is a transpose, and the row-selection structure is unchanged. Training moves only the columns of the geometry, indexed by tokens observed in the batch, so any property of the trained geometry is a property of the training distribution and the objective. The linear-layer backward materializes $\mathbf { X } ^ { \top } \mathbf { G } \in \mathbb { R } ^ { V \times d }$ densely, with, at most, n nonzero rows, whereas the scatter-add sees only those rows; the free carrier is a mathematical object that implementations never materialize; Table 1 shows.

<table><tr><td>Quantity</td><td>One-hot times E Lookup</td><td></td></tr><tr><td>Input storage</td><td> $\Theta ( n V )$ </td><td> $\Theta ( n )$ </td></tr><tr><td>Forward cost</td><td> $\Theta ( n V { \dot { d } } )$ </td><td> $\Theta ( n d )$ </td></tr><tr><td>Gradient storage</td><td> $\Theta ( V d )$  dense</td><td> $\Theta ( \operatorname* { m i n } ( n , V ) d )$  sparse</td></tr><tr><td>Parameters</td><td> $V d$ </td><td> $V d$ </td></tr></table>

Table 1: Cost comparison for a batch of n tokens.

## 3.4 Identifier bias and linearity

Now, [27] states the identity for a bias-free layer. The restriction concerns the identification of parameters, and leaves the function class untouched.

Let $b \in \mathbb { R } ^ { d }$ , and consider $x \mapsto x ^ { \top } \mathbf { E } + b ^ { \top }$ . On the input $\mathbf { e } _ { i } ,$ , this equals $\mathbf E _ { i } + b ^ { \top }$ , which is row i of $\mathbf { E } ^ { \prime } = \mathbf { E } + \mathbf { 1 } b ^ { \intercal }$ with $\mathbf { 1 } \in \mathbb { R } ^ { V }$ the all-ones vector. A linear layer with bias, restricted to one-hot inputs, is, again, a lookup, with table E<sup>′</sup>; the set of functions on one-hot inputs realized with bias equals the set realized without, and the parameterization with bias has a d-dimensional redundancy, since (E, b) and $( \mathbf { E } + \mathbf { 1 } c ^ { \top } , b - c )$ realize the same function for every $c \in \mathbb { R } ^ { d }$ . We see, in Section 5.3, the same redundancy, only from the side of the readout instead, where it becomes the all-ones row of the rank criterion.

$L _ { \mathbf { E } }$ is linear on $\mathbb { R } ^ { V }$ by construction. The composite $i \mapsto \ell _ { \mathbf { E } } ( i )$ on the integers admits a linear extension only when $\mathbf { E } _ { i } = i c$ for a fixed $c \in \mathbb { R } ^ { 1 \times d }$ and all i, which fails for generic E, as any E with $\mathbf { E } _ { 2 } \neq 2 \mathbf { E } _ { 1 }$ shows; and the encoding $i \mapsto \mathbf { e } _ { i }$ is, itself, incompatible with integer addition, since ${ \bf e } _ { 1 } + { \bf e } _ { 2 }$ lies outside the set of one-hot vectors. Token identifiers are labels; in the vector logic, this is the statement that the free carrier encodes entities as atoms: every relation among is via the geometry; we see in Section 5 which relations a geometry must preserve.

## 4 Regimes

The extensional theorem is proved for the free carrier. We see in Section 3 that the free carrier is the input to every embedding matrix; any other injection of $\mathcal { D } _ { e }$ into a real vector space is a candidate carrier, and which of them admit the lifts that Theorem 2.1 guarantees for the free one is decided by the geometry matrix of Definition 4.1.

## Definition 4.1 (geometry matrix)

Let $h : \mathcal { D } _ { e }  \mathbb { R } ^ { d }$ be any map; its geometry matrix is $\mathbf { H } \in \mathbb { R } ^ { d \times V }$ , with columns $\mathbf { H e } _ { i } = h ( d _ { i } )$ , such that $h = \mathbf { H } \circ h _ { e } .$ where $h _ { e }$ is the free carrier.

The free carrier itself has $\mathbf { H } = I _ { V } ;$ a trained embedding has $\mathbf { H } = \mathbf { E } ^ { \top }$ . We call h free when the columns of H are linearly independent, and compressed otherwise.

Every map out of $\mathcal { D } _ { e }$ into a vector space factors through the free carrier in this way, which is the universal property that justifies the name in the first place; the geometry matrix is the linear map through which it factors. Injectivity of h is, technically, a weaker condition than freeness: distinct columns suffice for injectivity; a compressed geometry may well have distinct columns.

![](images/eba19dbf67b846bd2f6066c854f8f22cd1d629c377675c3324c076e770bcbba0.jpg)

Figure 2: The dimension axis for a geometry $\mathbf { H } \in \mathbb { R } ^ { d \times V }$ over a lexicon with truth matrix T. Below rank[T; 1], no geometry carries the whole lexicon; at and above V with independent columns, every semantic function lifts; between them, exact lift is the row-space condition of Theorem 5.1, and is where the geometries of Section 8.4 are. One instance of each regime is drawn in Figure 4.

## 4.1 Free regime

## Proposition 4.1 (reparameterization)

Let $\{ h _ { \tau } \}$ be the free carrier with lifts $\{ L _ { f } \}$ , and, for each type, let $T _ { \tau }$ be an injective linear map on $\boldsymbol { S _ { D _ { \tau } } }$ Set $h _ { \tau } ^ { \prime } \doteq \check { T _ { \tau } } \circ h _ { \tau }$ . Then, for every semantic function $f : \mathcal { D } _ { \sigma } \overset { \cdot } {  } \mathcal { D } _ { \widehat { \ast } }$ with σ primitive, there is a linear $L _ { f } ^ { \prime }$ , with $L _ { f } ^ { \prime } \circ h _ { \sigma } ^ { \prime } = h _ { \tau } ^ { \prime } \circ f$ , and composition along primitive types is preserved.

In particular, every free geometry of Definition 4.1 admits lifts for every semantic function.

Proof. Since $T _ { \sigma }$ is injective, it has a linear left inverse $T _ { \sigma } ^ { - }$ with $T _ { \sigma } ^ { - } T _ { \sigma } = I$ on $\smash { S _ { D _ { c } } }$ . Define $L _ { f } ^ { \prime } = T _ { \tau } L _ { f } T _ { \sigma } ^ { - }$ . For $a \in \mathcal { D } _ { \sigma }$

$$
L _ { f } ^ { \prime } h _ { \sigma } ^ { \prime } ( a ) = T _ { \tau } L _ { f } T _ { \sigma } ^ { - } T _ { \sigma } h _ { \sigma } ( a ) = T _ { \tau } L _ { f } h _ { \sigma } ( a ) = T _ { \tau } h _ { \tau } ( f ( a ) ) = h _ { \tau } ^ { \prime } ( f ( a ) ) .
$$

For composition, if $g : { \mathcal { D } } _ { \tau } \to { \mathcal { D } } _ { \rho }$ with τ primitive, then

$$
L _ { g } ^ { \prime } L _ { f } ^ { \prime } = T _ { \rho } L _ { g } T _ { \tau } ^ { - } T _ { \tau } L _ { f } T _ { \sigma } ^ { - } = T _ { \rho } L _ { g } L _ { f } T _ { \sigma } ^ { - } = T _ { \rho } L _ { g \circ f } T _ { \sigma } ^ { - } = L _ { g \circ f } ^ { \prime } ,
$$

using Theorem 2.1 for ${ \cal L } _ { g } { \cal L } _ { f } = { \cal L } _ { g \circ f }$ . Finally, a free geometry $\mathbf { H } \in \mathbb { R } ^ { d \times V }$ has linearly independent columns, hence is an injective linear map $\mathbb { R } ^ { V } \to \mathbb { R } ^ { d }$ ; take $T _ { e } = \mathbf { H }$ and $T _ { \tau } = I$ for the other types. □

Here, we see that the orthonormal basis of the free carrier is a convenience: any linearly independent family of entity vectors will do. Every function lifts here, so every measurement on a free geometry passes the homomorphism conditions. The function-type spaces are left as they were, so a predicate still lives in Hom $( \mathring { \mathbb { R } } ^ { V } , \mathbb { R } ^ { 2 } )$ , while entities live in $\mathbb { R } ^ { d } ;$ the readout picture of Section $5 ,$ in which a predicate over a geometry H is a linear map $L _ { P } : \mathbb { R } ^ { d }  \mathbb { R } ^ { 2 }$ , is the specialization $L _ { P } = \mathbf { M } _ { P } \mathbf { H } ^ { - }$ , with $\mathbf { H } ^ { - }$ a left inverse of H.

## 4.2 Compressed regime

A trained embedding matrix has $\mathbf { E } ^ { \top } \in \mathbb { R } ^ { d \times V }$ with d in the hundreds (or low thousands), and V in the tens of thousands, so rank $\mathbf { E } ^ { \top } \leq d < \mathbf { \bar { \Gamma } }$ , and the geometry is compressed by a wide margin. Proposition 4.1 requires injective T, and, therefore, leaves it be. A criterion on ker H replaces the proposition, which is the space of linear dependences among the entity vectors: a vector $\mathbf { c } \in \mathbb { R } ^ { V }$ lies in ker H exactly when $\begin{array} { r } { \sum _ { i } c _ { i } h ( d _ { i } ) = 0 } \end{array}$ . We must show that lifts exist exactly when these dependences lie in the kernel of the lexicon’s augmented truth matrix.

## 5 Rank criterion for monadic predicates

Fix a geometry $\mathbf { H } \in \mathbb { R } ^ { d \times V }$ and a finite lexicon $\mathcal { P } \subseteq \mathcal { D } _ { \langle e , t \rangle }$ of monadic predicates. For $P \in \mathcal { P }$ , write $\mathbf { t } _ { P } \in \{ 0 , 1 \} ^ { 1 \times V }$ for its truth row, $( \mathbf { t } _ { P } ) _ { i } = P ( d _ { i } )$ , and let $\mathbf { T } \in \{ 0 , 1 \} ^ { | \mathcal { P } | \times V }$ be the truth matrix with rows $\mathbf { t } _ { P }$ . Write $\left[ \mathbf { T } ; \mathbf { 1 } \right]$ for T augmented by the all-ones row. Row spaces row(·) are subspaces of $\mathbb { R } ^ { 1 \times V }$

## 5.1 Exact lifts

An exact lift of P over H is a linear map $L _ { P } : \mathbb { R } ^ { d }  \mathbb { R } ^ { 2 }$ with $L _ { P } h ( d _ { i } ) = h _ { t } ( P ( d _ { i } ) )$ for every $i ,$ the truth basis $\mathbf { b } _ { 1 } , \mathbf { b } _ { 0 }$ being shared across the lexicon, such that the connective matrices of Section 2 act on outputs unchanged. In matrix form, $L _ { P } \in \mathbb { R } ^ { 2 \times d }$ , and the condition is

$$
{ \cal L } _ { P } { \bf H } = { \bf M } _ { P } ,\tag{2}
$$

with ${ \bf M } _ { P }$ the predicate matrix of the free carrier. Over the free carrier, $\mathbf { H } = I _ { V }$ and $L _ { P } = \mathbf { M } _ { P }$ solves $( 2 ) ;$ over a compressed geometry, (2) is a linear system in the unknown $L _ { P }$ that may well fail to be solvable.

![](images/c6a8201eee2f21151806adf173e39493312699f9030cd785054599d6da88e87f.jpg)

Figure 3: The left square commutes for every predicate by Theorem 2.1, with ${ \bf M } _ { P } = { \bf b } _ { 1 } { \bf t } _ { P } + { \bf b } _ { 0 } ( { \bf 1 } - { \bf t } _ { P } )$ the free readout; the dashed map exists when ${ \bf M } _ { P }$ factors through H, which is condition $( 2 )$ . When H is free (injective), the factorization is $L _ { P } = \dot { \mathbf { M } } _ { P } \mathbf { H } ^ { - }$ ; when H is compressed, it exists only if ker $\mathbf { H } \subseteq \ker \mathbf { M } _ { P }$ , which is a rank condition as Theorem 5.1.

Lemma 5.1   
For every $P , \mathrm { r o w } ( { \bf M } _ { P } ) = \mathrm { s p a n } \{ { \bf t } _ { P } , { \bf 1 } \}$

Proof. The rows of ${ \bf M } _ { P }$ are $\mathbf { t } _ { P }$ and ${ \bf 1 } - { \bf t } _ { P } $ , since column i is $\mathbf { b } _ { 1 }$ when $P ( d _ { i } ) = 1$ and $\mathbf { b } _ { 0 }$ otherwise. The spans of $\left\{ \mathbf { t } _ { P } , \mathbf { \bar { 1 } } - \mathbf { t } _ { P } \right\}$ and $\{ \mathbf { t } _ { P } , \mathbf { 1 } \}$ coincide. □

```latex
Theorem 5.1 (rank criterion)
The following are equivalent.
1. Every $P \in \mathcal { P }$ has an exact lift over $\mathbf { H } .$
2. row $( \mathbf { H } ) \supseteq \operatorname { r o w } [ \mathbf { T } ; \mathbf { 1 } ] .$
3. ker $\mathbf { H } \subseteq \ker [ \mathbf { T } ; \mathbf { 1 } ]$ ; that is, every linear dependence $\begin{array} { r } { \sum _ { i } c _ { i } h ( d _ { i } ) = 0 } \end{array}$ among the entity vectors satisfies
$\textstyle \sum _ { i } c _ { i } P ( d _ { i } ) { \dot { = } } 0$ for every $P \in { \dot { \mathcal { P } } }$ and $\textstyle \sum _ { i } { c _ { i } } = 0$
```

Proofsketch. The equation $L \mathbf { H } = \mathbf { M }$ has a solution L if and only if every row of M is a linear combination of the rows of H, that is, $\dot { \mathrm { r o w } } ( \mathbf { M } ) \subseteq \mathrm { r o w } ( \mathbf { H } )$ . By Lemma 5.1, (2) is solvable for $P$ if and only if $\mathbf { t } _ { P } , \mathbf { 1 } \in \mathrm { r o w } ( \mathbf { H } )$ , and solvability for all of $\mathcal { P }$ is row $[ \mathbf { T } ; \mathbf { 1 } ] \subseteq \operatorname { r o w } ( \mathbf { H } )$ , which is the equivalence of (1) and (2).

For (2) and (3), the row space and kernel of a matrix are orthogonal complements in $\mathbb { R } ^ { V }$ , so row $( \mathbf { H } ) \supseteq \mathrm { r o w } [ \mathbf { T } ; \mathbf { 1 } ]$ if and only if ker ${ \mathbf H } \subseteq \ker [ \dot { \mathbf T } ; { \mathbf 1 } ] ;$ ; and $\mathbf { c } \in \mathrm { k e r } [ \mathbf { T } ; \mathbf { 1 } ]$ to $\mathbf { t } _ { P } \mathbf { c } = 0$ for each P and $\mathbf { 1 c } = 0$

From (2) to (1) also admits a direct construction, and the direction from (1) to (3) a direct computation. Given (2), choose row vectors $\mathbf { w } _ { P } , \mathbf { w } _ { P } ^ { \prime } \in \mathbb { R } ^ { 1 \times d }$ with $\mathbf { w } _ { P } \mathbf { H } = \mathbf { t } _ { P }$ and $\mathbf { w } _ { P } ^ { \prime } \mathbf { H } = \mathbf { 1 } - \mathbf { t } _ { P }$ , and set ${ \cal L } _ { P } = { \bf \bar { b } } _ { 1 } { \bf w } _ { P } + { \bf b } _ { 0 } { \bf w } _ { P } ^ { \prime } ,$ , so that $L _ { P } x = \left( { \bf w } _ { P } x \right) { \bf b } _ { 1 } + ( \dot { \bf w } _ { P } ^ { \prime } x ) { \bf b } _ { 0 }$ . Then $L _ { P } h ( d _ { i } ) = P ( d _ { i } ) \mathbf { b _ { 1 } } ^ { \prime } + ( 1 - P ( d _ { i } ) ) \mathbf { b _ { 0 } } = h _ { t } ( P ( d _ { i } ) )$ . Conversely, given a lift $L _ { P }$ and $\mathbf { c \in }$ ker H, apply $L _ { P }$ to $\begin{array} { r } { \sum _ { i } c _ { i } h ( d _ { i } ) = 0 } \end{array}$ and expand: $\begin{array} { r } { \sum _ { i } c _ { i } \big ( P ( d _ { i } ) \mathbf { b } _ { 1 } + ( 1 - P ( d _ { i } ) ) \mathbf { b } _ { 0 } \big ) = 0 } \end{array}$ , and independence of $\mathbf { b } _ { 1 } , \mathbf { b } _ { 0 }$ forces $\begin{array} { r } { \sum _ { i } c _ { i } P ( d _ { i } ) \overset { \cdot } { = } 0 } \end{array}$ and $\textstyle \sum _ { i } c _ { i } ( 1 - P ( d _ { i } ) ) = { \dot { 0 } }$ , whose sum is $\textstyle \sum _ { i } c _ { i } = 0 .$ □

## 5.2 Corollaries

Here we take a brief tour of some nice properties of the lifts.

Corollary 5.1 (minimal dimension)   
The least d for which some $\mathbf { H } \in \mathbb { R } ^ { d \times V }$ carries exact lifts of all of $\mathcal { P }$ is rank $\mathbf { \mathbf { \cdot } } \left[ \mathbf { T } ; \mathbf { 1 } \right]$ , attained by any H, whose rows   
form a basis of row[T; 1].

Proof sketch. By (2) in Theorem 5.1, row(H) must contain a subspace of dimension rank[T; 1], so d $\geq$ rank H ≥ rank[T; 1]; taking the rows of H to be a basis of row[T; 1] gives equality. 口

Corollary 5.2 (closure forces the free regime)   
$\mathrm { I f } \mathcal { P } = \mathcal { D } _ { \langle e , t \rangle }$ , then exact lifts of all of P exist only over free geometries; in particular, $d \geq V .$

Proof sketch. $\mathcal { D } _ { \langle e , t \rangle }$ contains the singleton predicates $P _ { i }$ with $\mathbf { t } _ { P _ { i } } = \mathbf { e } _ { i } ^ { \top }$ , so $\operatorname { r o w } [ \mathbf { T } ; \mathbf { 1 } ] = \mathbb { R } ^ { 1 \times V }$ , and (2) in Theorem 5.1 forces rank H = V, which is linear independence of the V columns. □

The extensional theorem embeds every domain $\mathcal { D } _ { \langle e , t \rangle }$ in full; Corollary 5.2 requires a free geometry for it. Linear independence of the entity vectors is, thereby, derived from predicate closure, and the one-hot construction is one coordinate choice among the free geometries, which are its injective linear images by Definition 4.1, each admitting every lift by Proposition 4.1; the equal pairwise distances of the one-hot basis are a feature of that choice alone. The theorem embeds an object, the full function space $\mathcal { D } _ { \langle e , t \rangle }$ , that lies beyond what any learned system represents; the results that follow concern the compressed regime, where closure fails by construction, and the conditions have content.

## Corollary 5.3 (compressibility and the lexicon)

A dependence $\mathbf { c } \in \mathbb { R } ^ { V }$ among entity vectors is admissible, in the sense that some geometry with $\mathbf { c \in }$ ker H carries exact lifts of P, if and only if c ⊥ row[T; 1]; the admissible dependences form the orthogonal complement of the augmented row space of the lexicon, of dimension $V - \operatorname { r a n k } [ \bar { \mathbf { T } } ; \mathbf { 1 } ]$

Proof. Immediate from (3) in Theorem 5.1, taking H with row space exactly row $\left[ \mathbf { T } ; \mathbf { 1 } \right]$ for the converse.

A geometry may well identify entity vectors up to a dependence exactly when every predicate in the lexicon, and the constant predicate, assign that dependence weight zero. The dimension d of a learned embedding bounds its rank; whether d is compatible with exactness depends on the lexicon, which lexicalizes a minuscule fraction of the $2 ^ { V }$ available predicates, and Corollary 5.1 computes the exchange rate between rank and lexicon.

## Corollary 5.4 (indiscernibility)

Let H have row $\mathbf { \partial } ( \mathbf { H } ) = \operatorname { r o w } [ \mathbf { T } ; \mathbf { 1 } ]$ . Then $h ( d _ { i } ) = h ( d _ { j } )$ if and only if $P ( d _ { i } ) = P ( d _ { j } )$ for every $P \in \mathcal { P }$ . Hence, a minimal exact geometry is injective on $\mathcal { D } _ { e }$ if and only if $\mathcal { P }$ separates entities.

Proof sketch. $h ( d _ { i } ) = h ( d _ { j } )$ if and only if $\mathbf { e } _ { i } - \mathbf { e } _ { j } \in$ ker ${ \bf H } = \mathrm { r o w } [ { \bf T } ; { \bf 1 } ] ^ { \perp }$ , if and only if $\mathbf { t } _ { P } ( \mathbf { e } _ { i } - \mathbf { e } _ { j } ) = 0$ for all P (the condition $\mathbf { 1 } ( \mathbf { e } _ { i } - \mathbf { e } _ { j } ) \dot { = } 0$ holding automatically), which is $P ( d _ { i } ) = \mathbf { \bar { { } } } P ( d _ { j } )$ for all ${ \bf \dot { \boldsymbol { P } } } .$ □

Remark 5.1 (canonical geometry). The operator carrier of the predicate type is, itself, a compressed geometry. Let the $2 ^ { V }$ predicates $o f \mathcal { D } _ { \langle e , t \rangle }$ play the role ofentities, and Montague’s V individuals $\lambda P . P ( d _ { i } )$ the role ofa lexicon, with truth rows $\mathbf { t } _ { d _ { i } } ( P ) = P ( d _ { i } )$ . The comparison map $\varepsilon : \mathbb { R } ^ { 2 ^ { V } } \to \mathrm { H o m } ( \mathbb { R } ^ { V } , \mathbb { R } ^ { 2 } )$ of[26], which sends the basis vector ofP to ${ \bf M } _ { P } ,$ has, as its matrix, the $2 V \times 2 ^ { \stackrel { \prime } { V } }$ array, whose rows are $\mathbf { t } _ { d _ { i } }$ and $\mathbf { 1 } - \mathbf { t } _ { d _ { i } } , s o \operatorname { r o w } ( { \boldsymbol { \varepsilon } } ) = \operatorname { r o w } [ \mathbf { T } _ { \mathrm { i n d } } ; \mathbf { 1 } ]$ ], ofrank $V + 1$ $B y$ Corollary $5 . l ,$ ε is a minimal exact geometry for the lexicon of individuals, and, by Theorem $5 . l ,$ a Boolean-valued functional ofpredicates lifts over it when its truth row lies in the affine span ofthe coordinatefunctions on the cube $\{ 0 , 1 \} ^ { V }$ , which,for Boolean values, means a constant, a coordinate $P \mapsto P ( d )$ , or a negated coordinate. On operator encodings, Montague’s individuals and their negations are the only nonconstant linearfunctionals ofpredicates, and every determiner over a restrictor oftwo or more elements, together with every modal or attitude operator over two or more indices, lies outside the linear regime. Corollary 5.4 says of ε that it identifies two predicates when they agree on every individual, so that ε is injective on predicates while its kernel as a linear map is the parallelogram space of Proposition 8.3, ofdimension $2 ^ { V } - V - 1$

## 5.3 Affine readout

We see the all-ones row in Theorem 5.1 through the shared truth basis: the second row of ${ \bf M } _ { P }$ is ${ \bf 1 } - { \bf t } _ { P } ;$ a linear $L _ { P }$ must produce it. If $L _ { P }$ is permitted to be affine of the form $L _ { P } x = A x + \mathbf { c } _ { P }$ , then the requirement changes.

## Proposition 5.1 (affine lifts)

Affine exact lifts of all of $\mathcal { P }$ over H exist if and only if $\operatorname { r o w } ( \mathbf { T } ) \subseteq \operatorname { r o w } ( \mathbf { H } ) + \operatorname { s p a n } \{ \mathbf { 1 } \}$ ; equivalently, if and only if linear exact lifts exist over the homogenized geometry $\mathbf { H } ^ { + } = [ \mathbf { H } ; \mathbf { 1 } ] \in \bar { \mathbb { R } } ^ { ( d + 1 ) \times V }$

Proof sketch. An affine map on $\mathbb { R } ^ { d }$ is a linear map on $\mathbb { R } ^ { d + 1 }$ , restricted to the affine hyperplane of vectors with last coordinate 1, and $h ^ { + } ( d _ { i } ) = { \bf \ddot { \Phi } } ( h ( d _ { i } ) , 1 )$ has geometry matrix H<sup>+</sup>; the equivalence of affine lifts over H and linear lifts over $\mathbf { H } ^ { + }$ is this identification. Theorem 5.1, applied to $\mathbf { H } ^ { + }$ , requires row $[ \mathbf { T } ; \mathbf { 1 } ] \subseteq \operatorname { r o w } ( \mathbf { H } ^ { + } ) = \operatorname { r o w } ( \mathbf { H } ) + \operatorname { s p a n } \{ \mathbf { 1 } \}$ , in which the 1 requirement is automatic, leaving row $( \mathbf { T } ) \subseteq \operatorname { r o w } ( \mathbf { H } ) + \operatorname { s p a n } \{ \mathbf { 1 } \}$ □

## Corollary 5.5 (minimal affine dimension)

Write $\mathbf { T } _ { c } = \mathbf { T } - \bar { \mathbf { t } } \mathbf { 1 } ^ { \top }$ for the truth matrix with its row means removed. Then rank $\mathbf { T } _ { c } = \mathrm { r a n k } [ \mathbf { T } ; \mathbf { 1 } ] - 1$ , and the least d for which some $\mathbf { H } \in \mathbb { R } ^ { d \times V }$ carries affine exact lifts of all of $\mathcal { P }$ is rank $[ \mathbf { T } ; \mathbf { 1 } ] - 1$ , attained by any H whose rows form a basis of row $\left( \mathbf { T } _ { c } \right)$

Proof. Each row of $\mathbf { T } _ { c }$ sums to zero, so row $( \mathbf { T } _ { c } ) \subseteq \mathbf { 1 } ^ { \perp }$ and $\mathbf { 1 } \not \in \mathrm { r o w } ( \mathbf { T } _ { c } )$ , while row $[ \mathbf { T } ; \mathbf { 1 } ] = \operatorname { r o w } \left( \mathbf { T } _ { c } \right) + \operatorname { s p a n } \{ \mathbf { 1 } \}$ giving the rank identity. By Proposition 5.1, affine exact lifts exist if and only if ro $\begin{array} { r } { \boldsymbol { \mathrm { \Sigma } } ^ { } ( \mathbf { T } ) \subseteq \mathrm { r o w } ( \mathbf { H } ) + \mathrm { s p a n } \{ \mathbf { 1 } \} } \end{array}$ equivalently row $( \mathbf { T } _ { c } ) \dot { \subseteq } \mathrm { r o w } ( \mathbf { H } ) \dot { + } \mathrm { s p a n } \{ \mathbf { 1 } \}$ . The right side has dimension at most $d + 1$ and must contain $\mathrm { r o w } ( \mathbf { T } _ { c } ) \dot { + }$ span{1} of dimension rank[T; 1], whence $\begin{array} { r } { \dot { d } \geq \operatorname { r a n k } [ \mathbf { T } ; \mathbf { 1 } ] - 1 } \end{array}$ , with equality when row $( \mathbf { H } ) = \operatorname { r o w } ( \mathbf { T } _ { c } )$ □

Appending a constant coordinate converts affine readouts into linear ones; the bias supplies the constant row required by the shared truth basis. Over the free carrier, linear and affine readouts realize the same predicate extensions, since $\mathbf { 1 } \in \operatorname { r o w } ( I _ { V } )$ . We retain the linear formulation as primary, and obtain affine readouts by homogenization. Once either readout returns the truth basis, the Boolean connective matrices act unchanged.

## 5.4 Connectives and sentences

Once every predicate in $\mathcal { P }$ lifts exactly, so does every sentence built from atomic predications by truth-functional connectives, with the same connective matrices as over the free carrier.

## Proposition 5.2

Suppose every $P \in { \mathcal { P } }$ has an exact lift $L _ { P }$ over H. Then, for every Boolean combination Φ of atomic predications $P ( d _ { i } )$ , the vector computed by applying M<sub>c</sub> to tensor products of lifted outputs equals $h _ { t } ( \mathbb { I P } ] )$

Proof. We proceed by induction on Φ. For the simple case: $L _ { P } h ( d _ { i } ) = h _ { t } ( P ( d _ { i } ) )$ by hypothesis. Inductive case: if the immediate subformulas evaluate to $h _ { t } ( t _ { 1 } ) , \ldots , h _ { t } \mathbf { \bar { ( } } t _ { n } )$ , then ${ \bf M } _ { c } \big ( h _ { t } ( t _ { 1 } ) \otimes h _ { t } ( t _ { 2 } ) \otimes \cdots \otimes h _ { t } ( t _ { n } ) \big ) = h _ { t } \big ( c ( t _ { 1 } , \dots , t _ { n } ) \big )$ by the definition of $\mathbf { M } _ { c }$ over the truth space, which is unchanged by compression of the entity space. 口

Compression of the entity carrier is, therefore, confined to the leaves of a derivation; the connective level of the vector logic is insensitive to it. This is a first indication of where the compressed regime places its constraints: on the geometry of atoms and their readouts; everything from the truth space upward is unchanged. The compound predicates themselves need no lifts of their own, and in general have none: given the 1 row, the exact lexicon is closed under negation, since ${ \mathbf 1 } - { \mathbf t } _ { P } \in \mathrm { r o w } ( { \mathbf H } )$ , while $\mathbf { t } _ { P } \odot \mathbf { t } _ { Q }$ , the truth row of ${ \check { P } } \land Q$ as a monadic predicate, may lie outside row(H), as it does for the two predicates of Example 5.1 at $V = 4$ , where the conjunction is the singleton $\{ d _ { 1 } \}$ } that Figure 4 excludes. Proposition 5.2 evaluates $P ( d _ { i } ) \land Q ( d _ { i } )$ exactly all the same, because the tensor product of the two readouts is quadratic in $h ( d _ { i } )$ ), and the connective matrix acts on that product.

Example 5.1. Take $\mathcal { D } _ { e } = \{ \mathrm { D A N I E I }$ , THOMAS}, so $V = 2 ,$ , and the lexicon $\mathcal { P } = \{ \mathrm { w r i t e } , \mathrm { p u b l i s h e d } \}$ with ${ \bf t } _ { \mathrm { w r i t e } } =$ $( 1 , 0 )$ and $\mathbf { t } _ { \mathrm { p u b l i s h e d } } = ( 0 , 0 )$ , so that at the world ofevaluation Daniel writes and neither is published. Then $\left[ \mathbf { T } ; \mathbf { 1 } \right]$ has rows $( 1 , 0 ) , \dot { ( 0 , 0 ) } , ( 1 , 1 )$ and rank $2 = V ;$ by Corollary 5.1, the minimal exact dimension is 2, so this lexicon already forces the free regime on two entities. Extend to $V = 4$ , with ${ \bf t } _ { \mathrm { w r i t e } } = ( 1 , 1 , 0 , 0 )$ and $\mathbf { t } _ { \mathrm { p u b l i s h e d } } = \left( 1 , 0 , 1 , 0 \right)$ : now rank $[ \mathbf { T } ; \mathbf { 1 } ] = 3 < 4 ,$ exact compression to $d = 3$ exists, and Section 8.3 identifies the one admissible dependence.

![](images/7142bdca3529da533d890075728a85d30a8c621c67b37b4507e4f4758b604934.jpg)  
free, V = 2

![](images/fb57506d60aa8bb4fe7c505c36fcbde79854871041d76ede621e5adc4c250f95.jpg)  
compressed, V = 4, d = 3  
Figure 4: Both regimes on the lexicon of Example 5.1, in minimal exact geometries, whose rows are truth rows and 1, the zero row of published at $V = 2$ dropped; each coordinate reads off a predicate (or the constant); every entity vector reaches the affine slice at height one. Left: $V = 2 \colon \operatorname { r a n k } [ \mathbf { T } ; \mathbf { 1 } ] = 2 = { \bar { V } }$ , the lexicon forces the free regime, and the two entity vectors are independent; the sole dependence $\begin{array} { r } { \sum _ { i } c _ { i } \dot { h } ( d _ { i } ) = 0 } \end{array}$ is $c = 0$ . Right: $V = 4 :$ rank $\left[ \mathbf { T } ; \mathbf { 1 } \right] = 3$ , and the four entity vectors satisfy the single admissible dependence $\dot { h } ( \dot { d } _ { 1 } ) - h ( d _ { 2 } ) - h ( \bar { d } _ { 3 } ) + h ( d _ { 4 } ) = 0 , \dot { c }$ a parallelogram as in Section 8.3. Every truth row of the lexicon annihilates it, so ${ \bf w } _ { \mathrm { w r i t e } } = ( 1 , 0 , 0 )$ and $\mathbf { w } _ { \mathrm { p u b l i s h e d } } = ( 0 , 1 , 0 )$ are exact; the singleton $\{ d _ { 1 } \}$ assigns it weight 1, and (3) of Theorem 5.1 excludes its exact lift over this geometry.

## 5.5 Determiners

Once the atomic lexicon lifts exactly, quantification over entities computes in the compressed space as well. A determiner meaning that is conservative, extension-invariant, and isomorphism-invariant depends on its restrictor $P$ and scope Q only through the pair $( | P \setminus Q | , | P \cap Q | ) [ 3 0 , 1 1 ]$ , the tree of numbers, so that every is $| P \setminus Q | = 0$ , some is $| P \cap Q | \geq 1$ , most is $| P \cap Q | > | P \setminus Q |$ , and at least n is $| P \cap Q | \geq n$

## Proposition 5.3 (compressed determiners)

Suppose every $P \in \mathcal { P }$ has an exact lift over H, with readouts $\mathbf { w } _ { P } \mathbf { H } = \mathbf { t } _ { P }$ and $\mathbf { w _ { 1 } H } = \mathbf { 1 }$ , and let $\mathbf { G } = \mathbf { H } \mathbf { H } ^ { \top } \in$ $\mathbb { R } ^ { d \times d }$ . Then, for all $P , Q \in { \mathcal { P } }$

$$
| P \cap Q | = \mathbf { w } _ { P } \mathbf { G } \mathbf { w } _ { Q } ^ { \top } , \qquad | P \setminus Q | = \mathbf { w } _ { P } \mathbf { G } \left( \mathbf { w } _ { 1 } - \mathbf { w } _ { Q } \right) ^ { \top } ,
$$

so every conservative, extension-invariant, isomorphism-invariant determiner evaluates on $P$ and Q by two $d \times d$ bilinear accumulations followed by its decision on the tree of numbers.

Proof. $\left| P \cap Q \right| = \mathbf { t } _ { P } \mathbf { t } _ { Q } ^ { \top } = \mathbf { w } _ { P } \mathbf { H } \mathbf { H } ^ { \top } \mathbf { w } _ { Q } ^ { \top } \mathrm { ~ a n d ~ } | P \setminus Q | = \mathbf { t } _ { P } ( \mathbf { 1 } - \mathbf { t } _ { Q } ) ^ { \top } = \mathbf { w } _ { P } \mathbf { H } \mathbf { H } ^ { \top } ( \mathbf { w } _ { 1 } - \mathbf { w } _ { Q } ) ^ { \top }$ ; the rest is the cited classification. □

The Gram matrix G is formed once from the geometry and shared across the lexicon. The quantified sentence is a decision on bilinear forms in the readouts and, by the descent theorem of [26], a linear readout of no single vector. The two accumulations have the form of the modal accumulation (1) with the restrictor in place of the accessible set, which is the sense in which modals quantify over worlds [13]; the compressed regime places one constraint on both, and leaves the connective and quantifier levels as they are. Beyond the exact regime, the counts inherit the defect of Section 8 through $\mathbf { t } _ { P } - \mathbf { w } _ { P } \mathbf { H }$ , and the decisions then act on approximate counts, which we leave to future work.

## 6 Relations

A binary relation $R \subseteq { \mathcal { D } } _ { e } \times { \mathcal { D } } _ { e }$ is, at type $\langle e , \langle e , t \rangle \rangle$ , a curried function, and the Hom construction of [26] lifts it to a linear map $S _ { D _ { e } } $ Hom $( S _ { D _ { e } } , S _ { D _ { t } } )$ , equivalently, a bilinear map $S _ { D _ { e } } \times S _ { D _ { e } } \to S _ { D _ { t } }$ , equivalently, a linear map on the tensor square. Over a compressed geometry, the corresponding object is a linear $L _ { R } : \mathbb { R } ^ { d } \otimes \mathbb { R } ^ { d }  \mathbb { R } ^ { 2 }$ with

$$
L _ { R } \left( h ( d _ { i } ) \otimes h ( d _ { j } ) \right) = h _ { t } ( R ( d _ { i } , d _ { j } ) ) \quad { \mathrm { f o r ~ a l l ~ } } i , j .\tag{3}
$$

Write $\mathbf { T } _ { R } \in \{ 0 , 1 \} ^ { V \times V }$ for the truth matrix of R, $( { \bf T } _ { R } ) _ { i j } = R ( d _ { i } , d _ { j } )$

## 6.1 Bilinear criterion

Theorem 6.1 (relations)

An exact lift (3) of R over H exists if and only if $\mathbf { 1 } \in \operatorname { r o w } ( \mathbf { H } )$ , and there is $\mathbf { A } _ { R } \in \mathbb { R } ^ { d \times d }$ with

$$
\mathbf { T } _ { R } = \mathbf { H } ^ { \top } \mathbf { A } _ { R } \mathbf { H } .
$$

Proof sketch. Bookkeeping is an irritant and laborious here. We proceed cautiously.

The tensor square $h ( d _ { i } ) \otimes h ( d _ { j } )$ is column $( i , j )$ of $\mathbf { H } \otimes \mathbf { H } \in \mathbb { R } ^ { d ^ { 2 } \times V ^ { 2 } }$ , so (3) is $L _ { R } ( \mathbf { H } \otimes \mathbf { H } ) = \mathbf { M } _ { R }$ , with $\mathbf { M } _ { R } \in \mathbb { R } ^ { 2 \times V ^ { 2 } }$ having columns $h _ { t } ( R ( d _ { i } , \dot { d } _ { j } ) )$ in the same ordering of pairs $( i , j )$ as the Kronecker product; as in Lemma 5.1, $\mathrm { r o w } ( \mathbf { M } _ { R } ) = \mathrm { s p a n } \{ \mathrm { v e c } ( \mathbf { T } _ { R } ) ^ { \top } , \mathbf { 1 } _ { V ^ { 2 } } ^ { \top } \}$ with vec taken in that ordering, and solvability is row $\left( \mathbf { M } _ { R } \right) \subseteq$ row(H ⊗ H). The rows of H ⊗ H are the Kronecker products $\mathbf { H } _ { k } \otimes \mathbf { H } _ { l }$ of pairs of rows of H, so row(H ⊗ H) = row(H) ⊗ row(H); under the identification of $\mathbb { R } ^ { 1 \times V } \otimes \dot { \mathbb { R } } ^ { 1 \times V }$ with $V \times V$ matrices, row $( \mathbf { H } ) \otimes \mathrm { r o w } ( \mathbf { H } ) = \{ \mathbf { H } ^ { \top } \mathbf { A } \mathbf { H } : \mathbf { A } \in \mathbb { R } ^ { d \times d } \}$ since row $\mathbf { ( H ) } = \{ \mathbf { a } ^ { \top } \mathbf { H } \}$ and $( \mathbf { a } ^ { \top } \mathbf { H } ) ^ { \top } ( \mathbf { a } ^ { \prime \top } \mathbf { H } ) = \mathbf { H } ^ { \top } ( \mathbf { a } \mathbf { a } ^ { \prime \top } ) \mathbf { H }$ , with sums of such rank-one terms filling out all A. So $\mathrm { v e c } ( \mathbf { T } _ { R } ) ^ { \top } \in \mathrm { r o w } ( \mathbf { H } \otimes \mathbf { H } )$ is $\mathbf { T } _ { R } = \mathbf { H } ^ { \top } \mathbf { A } _ { R } \mathbf { H }$ for some $\mathbf { A } _ { R } ,$ , and $\mathbf { 1 } _ { V ^ { 2 } } ^ { \top } = \mathbf { 1 } _ { V } ^ { \top } \otimes \mathbf { 1 } _ { V } ^ { \top } \in \mathrm { r o w } ( \mathbf { H } ) \otimes \mathbf { \bar { r } } \mathrm { o w } ( \mathbf { H } )$ if and only if $\mathbf { 1 } _ { V } ^ { \top } \in \operatorname { r o w } ( \mathbf { H } )$ , since $\mathbf { 1 1 } ^ { \top } = \mathbf { H } ^ { \top } \mathbf { A } \mathbf { H }$ requires the rank-one matrix $\mathbf { 1 1 ^ { \top } }$ to have column space inside $\mathrm { c o l } ( \mathbf { H } ^ { \top } ) = \mathrm { r o w } ( \mathbf { H } ) ^ { \top }$ , and conversely $\mathbf { 1 } _ { V } ^ { \top } = \mathbf { a } ^ { \top }$ H gives $\mathbf { 1 1 } ^ { \top } = \mathbf { H } ^ { \top } \mathbf { a a } ^ { \top } \mathbf { H }$ □

![](images/74cc7e450aac0a01f3568db9d067c3072596dccc2008822e9b63274fef6173ce.jpg)  
Figure 5: Factorization $\mathbf { T } _ { R } = \mathbf { H } ^ { \top } \mathbf { A } _ { R } \mathbf { H }$ , with d $\ll V$ . The shaded row of $\mathbf { H } ^ { \top }$ is $h ( d _ { i } ) ^ { \top }$ and the shaded column of H is $h ( d _ { j } )$ ; the light bands in $\mathbf { T } _ { R }$ are row i and column $j ,$ , and their crossing is the entry $( { \bf T } _ { R } ) _ { i j } = R ( d _ { i } , d _ { j } )$ , the readout $\bar { h ( d _ { i } ) ^ { \top } } \mathbf { A } _ { R } h ( d _ { j } )$ . The inner dimension is $d ,$ giving rank $\mathbf { T } _ { R } \leq d ,$ the obstruction of Corollary 6.1; the second condition of the theorem, 1 ∈ row(H), constrains H alone.

The lifted relation<sup>3</sup> is a bilinear form $\mathbf { A } _ { R }$ on the compressed space, and $R ( d _ { i } , d _ { j } )$ is read off as $h ( d _ { i } ) ^ { \top } \mathbf { A } _ { R } h ( d _ { j } )$

## 6.2 Obstructions

## Corollary 6.1 (rank obstruction)

If R lifts exactly over H, then rank $\mathbf { T } _ { R } \leq$ rank $\mathbf { H } \leq d .$ In particular, the identity relation $\{ ( d _ { i } , d _ { i } ) \}$ , with $\mathbf { T } _ { = } = I _ { V }$ lifts exactly only over free geometries.

Proof. rank $( \mathbf { H } ^ { \top } \mathbf { A } _ { R } \mathbf { H } ) \leq$ rank H; and rank $I _ { V } = V$ forces rank $\mathbf { H } = V .$

The rank bound is a lower bound only; the factorization constrains both argument positions through the same geometry, and the exact value follows from the proof of Theorem 6.1.

## Corollary 6.2 (minimal dimension for a relation)

The least d for which some $\mathbf { H } \in \mathbb { R } ^ { d \times V }$ carries an exact lift of R is dim span $\left( \{ \mathbf { 1 } \} \cup \operatorname { r o w } ( \mathbf { T } _ { R } ) \cup \operatorname { r o w } ( \mathbf { T } _ { R } ^ { \top } ) \right)$ , attained by any H whose rows form a basis of that span.

Proof. By the identification in the proof of Theorem 6.1, $\{ \mathbf { H } ^ { \top } \mathbf { A } \mathbf { H } : \mathbf { A } \in \mathbb { R } ^ { d \times d } \}$ is the set of $V \times V$ matrices whose rows lie in row(H) and whose columns lie in $\mathrm { r o w } ( \mathbf { H } ) ^ { \top }$ , so $\mathbf { T } _ { R } = \mathbf { H } ^ { \top } \mathbf { A } _ { R } \mathbf { H }$ is solvable if and only if row $( \mathbf { T } _ { R } ) \subseteq \mathrm { r o w } ( \mathbf { H } )$ and row $( \mathbf { T } _ { R } ^ { \top } ) \subseteq \operatorname { r o w } ( \mathbf { H } )$ . With the requirement $\mathbf { 1 } \in \operatorname { r o w } ( \mathbf { H } )$ , exact lift is containment of the stated span in row(H), whence $d \geq$ rank $\mathbf { H } \geq$ dim span $( \{ \mathbf { 1 } \} \cup \operatorname { r o w } ( \mathbf { T } _ { R } ) \cup \operatorname { r o w } ( \mathbf { T } _ { R } ^ { \top } ) )$ , with equality when the rows of H are a basis of the span. □

A single monadic predicate has rank $[ { \bf t } _ { P } ; { \bf 1 } ] \le 2 ,$ and lifts exactly in dimension two, so forcing high dimension at type $\left. { e , t } \right.$ requires a lexicon, and Corollary 5.2 uses the whole closed such one. A single binary relation can force the free regime by itself, and the relation that does so is identity, the denotation of the copula in Daniel is Daniel. A strict total order does so as well, one dimension above its rank: $\mathbf { T } _ { < }$ is strictly upper triangular with ones above the diagonal, of rank $V - 1$ , while row $( \mathbf { T } _ { < } ) = \operatorname { s p a n } \{ \mathbf { e } _ { 2 } ^ { \top } , \ldots , \mathbf { e } _ { V } ^ { \top } \}$ and $\operatorname { i r o w } ( \mathbf { T } _ { < } ^ { \top } ) ^ { \top } = \operatorname { s p a n } \{ \mathbf { e } _ { 1 } ^ { \top } , \dots , \mathbf { e } _ { V - 1 } ^ { \top } \}$ together span $\mathbb { R } ^ { 1 \times V }$ for $V \geq 2$ , so Corollary 6.2 returns V . Equivalence relations with k classes have rank $\mathbf { T } _ { R } = k$ , and, since $\mathbf { T } _ { R }$ is symmetric with 1 in its row space, compress to dimension k exactly. The rank of a relation’s truth matrix bounds the dimension of any exact carrier from below, in the same way that rank[T; 1] bounds the dimension for a monadic lexicon, and Corollary 6.2 gives the value.

Remark 6.1 (higher arities). An n-ary relation imposes its condition on $\mathbf { H } ^ { \otimes n }$ , with rank $( \mathbf { H } ^ { \otimes n } ) = ( \mathrm { r a n k } \mathbf { H } ) ^ { n }$ , and the lift is an n-linearform; theflattening ranks ofthe truth tensor bound rank Hfrom below in the same way. Theorem 6.1 is the $n = 2$ case.

## 7 Index sorts

The intensional layer places its free carrier on the index space, $h _ { S } ( s ) = { \bf e } _ { s } ;$ for a discrete sort with finitely many indices, the results of Sections 5 and 6 apply to any compressed geometry $\mathbf { H } _ { W } \in \mathbb { R } ^ { d \times n }$ of the world sort with the same proofs<sup>4</sup>, once the objects are identified. Here, propositions play the role of monadic predicates: $\varphi$ has truth profile $\mathbf { v } ( \varphi ) ^ { \top }$ as its truth row over $W ,$ , and a lexicon of propositions $\mathcal { P } _ { W }$ has truth matrix $\mathbf { T } _ { W }$ . Accessibility plays the role of a binary relation, with truth matrix A.

## Proposition 7.1 (compressed worlds)

Let $\mathbf { H } _ { W } \in \mathbb { R } ^ { d \times n }$ be a geometry of $W .$

1. Every $\varphi \in { \mathcal { P } } _ { W }$ has an exact lift ${ \cal L } _ { \varphi } { \bf H } _ { W } = { \bf P } _ { \varphi }$ if and only $\mathrm { i f } \mathrm { r o w } ( { \bf H } _ { W } ) \supseteq \mathrm { r o w } [ { \bf T } _ { W } ; { \bf 1 } ] ;$ the minimal exact dimension is $\mathrm { r a n k } [ \mathbf { T } _ { W } ; \mathbf { 1 } ]$

2. Accessibility lifts exactly as a bilinear form if and only if $\mathbf { 1 } \in \mathrm { r o w } ( \mathbf { H } _ { W } )$ and $\mathbf { A } = \mathbf { H } _ { W } ^ { \top } \widehat { \mathbf { A } } \mathbf { H } _ { W }$ for some $\widehat { \mathbf { A } } \in \mathbb { R } ^ { d \times d }$ ; in particular, rank $\mathbf { A } \leq d .$

3. Under (1) and (2) the modal accumulation of (1) computes in the compressed space: with $\mathbf { v } ( \varphi ) =$ $( { \bf w } _ { \varphi } { \bf H } _ { W } ) ^ { \top }$ and $\mathbf { 1 } = ( \mathbf { w } _ { \mathbf { 1 } } \mathbf { H } _ { W } ) ^ { \top }$ for the readouts of (1),

$$
\mathbf { A } \mathbf { v } ( \varphi ) = \mathbf { H } _ { W } ^ { \top } \widehat { \mathbf { A } } \mathbf { H } _ { W } \mathbf { H } _ { W } ^ { \top } \mathbf { w } _ { \varphi } ^ { \top } , \qquad \mathbf { A } \left( \mathbf { 1 } - \mathbf { v } ( \varphi ) \right) = \mathbf { H } _ { W } ^ { \top } \widehat { \mathbf { A } } \mathbf { H } _ { W } \mathbf { H } _ { W } ^ { \top } \left( \mathbf { w } _ { \mathbf { 1 } } - \mathbf { w } _ { \varphi } \right) ^ { \top } ,
$$

each a d $\times d$ computation followed by one expansion through ${ \bf { H } } _ { W } ^ { \top }$ , after which the decisions of (1) apply unchanged.

Proof. (1) and (2) are simply Theorems 5.1 and 6.1, with W for $\mathcal { D } _ { e } . ~ ( 3 )$ is substitution.

![](images/e78c31f6608a4e10b0cc03509cff4c39a2b48ad4b71817e4dc7aead1f450d1f8.jpg)  
Figure 6: The compressed modal accumulation of (3), $\mathbf { A } \mathbf { v } ( \varphi ) = \mathbf { H } _ { W } ^ { \top } \widehat { \mathbf { A } } \left( \mathbf { H } _ { W } \mathbf { H } _ { W } ^ { \top } \right) \mathbf { w } _ { \varphi } ^ { \top }$ , with $d \ll n$ . To the right of ${ \bf { H } } _ { W } ^ { \top }$ , every block is d × d or $d \times 1$ , so the accumulation is a $d \times d$ computation, and one expansion through $\mathbf { H } _ { W } ^ { \top }$ ; the threshold checks of (1) then apply to the expanded vector unchanged. The $\mathbf { H } _ { W } \mathbf { H } _ { W } ^ { \top }$ is as a single $d \times d$ block, because it is formed once from the geometry and shared across propositions $\varphi .$

The reflexive and transitive frames of the modal logics that concern the intensional layer have accessibility matrices of varying rank; the empty relation has rank zero, and imposes only the 1 requirement, a universal relation has rank one and compresses to a single dimension, a strict linear order, of rank $n - 1$ , forces the free regime by Corollary 6.2, its row and column spaces together spanning $\mathbb { R } ^ { 1 \times n }$ , as does a reflexive linear order, whose matrix is upper triangular with unit diagonal, and the identity relation (the frame of the trivial modality), by Corollary 6.1; the strict future accessibility of a discrete time sort, therefore, admits exact lift only over the free carrier of that sort. Since the operators of (1) read A only through its Boolean support, an approximate carrier need only reproduce the support of A for the modal verdicts to survive, a weaker requirement than exact bilinear factorization<sup>5</sup>.

## 8 Relaxation

Exact lift is a subspace condition that either holds or fails outright; learned geometries will fail it for most predicates, so we might wonder, then how far is a geometry from carrying a predicate at all? The direct construction in the proof of Theorem 5.1 suggests that exact lift places $\mathbf { t } _ { P }$ in row(H), and the distance from $\mathbf { t } _ { P }$ to row(H) is a least-squares quantity.

## 8.1 Defect

Definition 8.1 (defect)

For a predicate $P$ with truth row $\mathbf { t } _ { P }$ and a geometry H, the defect is

$$
\delta ( P ; \mathbf { H } ) = \operatorname* { m i n } _ { \mathbf { w } \in \mathbb { R } ^ { 1 \times d } } \left\| \mathbf { w } \mathbf { H } - \mathbf { t } _ { P } \right\| _ { 2 } .
$$

![](images/7b828acdad560b43d47b9c2cd2c1fb76744ed95be31e36e3b5dfcfdc06130f31.jpg)  
Figure 7: The defect of Definition 8.1, in which the truth row $\mathbf { t } _ { P }$ is projected onto $\operatorname { r o w } ( \mathbf { H } )$ ; the length is $\delta ( P ; { \bf H } )$ and the angle θ is the principal angle between $\mathbf { t } _ { P }$ and the row space, with $\delta = \lVert \mathbf { t } _ { P } \rVert$ sin θ. Exact lift is $\theta = 0$ . The 1 row is drawn nearly in the plane, as Section 8.4 finds it for trained geometries; the principal angles of Table 2 are the angles θ taken jointly over row $\left[ \mathbf { T } ; \mathbf { 1 } \right]$

## Proposition 8.1

$\delta ( P ; \mathbf { H } ) = \| \mathbf { t } _ { P } ( I _ { V } - \mathbf { H } ^ { \dagger } \mathbf { H } ) \| _ { 2 }$ , where $\mathbf { H } ^ { \dagger }$ is the Moore–Penrose pseudoinverse, and $\mathbf { H } ^ { \dagger } \mathbf { H }$ is the orthogonal projector onto row(H); the minimizer is $\mathbf { w } = \mathbf { t } _ { P } \mathbf { H } ^ { \dagger }$ . Moreover, $\delta ( P ; { \bf H } ) = 0$ if and only if $\mathbf { t } _ { P } \in \mathrm { r o w } ( \mathbf { H } )$ , so that if $\bar { \mathbf { 1 } } \in \operatorname { r o w } ( \mathbf { H } )$ , exact lift of P is $\delta ( P ; { \mathbf { H } } ) = 0 .$

Proofsketch. {wH} is $\operatorname { r o w } ( \mathbf { H } )$ , and the closest point of a subspace to $\mathbf { t } _ { P }$ is its orthogonal projection $\mathbf { t } _ { P } \mathbf { H } ^ { \dagger } \mathbf { H }$ , with $\mathbf { t } _ { P } ( I - \mathbf { H } ^ { \dagger } \mathbf { H } )$ vanishes exactly on the subspace. The final clause is Theorem 5.1. □

The defect is computable on any trained embedding by one least-squares solve per predicate, with predicate extensions supplied by lexical resources or feature norms [15]; it is a statistic of the trained geometry and a property of the training data and objective, which is unconstrained; the condition is, therefore, a specification, and the defect is a measurement.

## 8.2 Threshold and separability

Conceptual spaces [8] and degree semantics [12] treat graded predicates as regions in a structured space, with the classical predicate recovered by a threshold, $[ \bar { P } ] ( x ) = \bar { \mathbf { 1 } } [ d ( x , \bar { R _ { P } } ) \leq \theta _ { P } ] ;$ its simplest instance is a half-space, what the exact lift becomes when the equality in (2) is weakened to a sign condition.

## Definition 8.2 (thresholded lift)

P is linearly separable over H if there are w $\in \mathbb { R } ^ { 1 \times d }$ and $\theta \in \mathbb { R }$ with $\mathbf { w } h ( d _ { i } ) > \theta$ when $P ( d _ { i } ) = 1$ and $\mathbf { w } h ( d _ { i } ) < \theta$ when $P ( d _ { i } ) = 0$

## Proposition 8.2

If P has an exact lift over H, then $P$ is linearly separable over H; the converse fails.

Proof. With $\mathbf { w } = \mathbf { w } _ { P }$ from the direct construction, $\mathbf { w } h ( d _ { i } ) = P ( d _ { i } ) \in \{ 0 , 1 \}$ , and $\textstyle \theta = { \frac { 1 } { 2 } }$ separates. For the converse, take $d = 1 , \mathbf { H } = [ 1 2 3 4 ]$ ], and $\mathbf { t } _ { P } = ( 0 , 0 , 1 , 1 )$ : the threshold $\textstyle \theta = { \frac { 5 } { 2 } }$ separates, while $\mathbf { t } _ { P } \in \mathrm { s p a n } \{ \mathbf { H } , \mathbf { 1 } \}$ would require $a + c = 0$ and 2a $+ c = 0$ from the first two coordinates, forcing $a = c = 0 ,$ , which contradicts the third coordinate; exact lift, therefore, fails, in the affine sense of Proposition 5.1, as well as the linear one. □

A linear probe tests the linear separability of a predicate over a learned geometry: [1] keep the probe linear, so that accuracy tracks the representation, and [10] caution that it tracks the capacity of the probe as well; we must account for this, so we do so with held-out scoring throughout, and a Gaussian null on the held-out error of Section 8.4. Proposition 8.2 is the probe in the vector logic: thresholded relaxation of the homomorphism condition, strictly weaker than the exact condition, and the defec $\delta ( \bar { P } ; { \bf { H } } )$ is the condition’s own measure. Probing practice fits each predicate independently; the rank criterion adds that a shared readout basis across the lexicon requires the 1-row, or, equivalently, a bias, and that the ranks of Corollaries 5.1 and 6.1 bound what any geometry of a given dimension can carry, before any probe is fit.

![](images/01c3dbb8e9209138dd4803dbc94362aa7131a482444ae86ffcf0b18b383e6939.jpg)  
Figure 8: The example of Proposition 8.2: four entities on a line, $\mathbf { t } _ { P } = ( 0 , 0 , 1 , 1 )$ . No affine function of position takes the values 0, 0, 1, 1, so the affine defect is positive, while the threshold at $5 / 2$ separates the extension exactly. The measurements of Section 8.4 find learned geometries in this position for nearly every predicate: high separability, positive defect.

## 8.3 Parallelograms

Let us now return to the four-entity lexicon of Example 5.1: $\mathbf { t } _ { \mathrm { w r i t e } } = ( 1 , 1 , 0 , 0 ) , \mathbf { t } _ { \mathrm { p u b l i s h e d } } = ( 1 , 0 , 1 , 0 ) , \mathbf { 1 } =$ (1, 1, 1, 1). The three rows are independent, so rank[T; 1] = 3, and, by Corollary 5.3, the admissible dependences form a one-dimensional space (the orthogonal complement of the row space), spanned by

$$
\mathbf { c } = ( 1 , - 1 , - 1 , 1 ) ^ { \top } : \qquad \mathbf { t } _ { \mathrm { w r i t e } } \mathbf { c } = 1 - 1 = 0 , \quad \mathbf { t } _ { \mathrm { p u b l i s h e d } } \mathbf { c } = 1 - 1 = 0 , \quad \mathbf { 1 } \mathbf { c } = 0 .
$$

Every minimal exact geometry for this lexicon, therefore, satisfies exactly one dependence,

$$
h ( d _ { 1 } ) - h ( d _ { 2 } ) = h ( d _ { 3 } ) - h ( d _ { 4 } ) ,
$$

which form a parallelogram, with the interpretation: the difference between a writer who is published and one who is unpublished equals the difference between a nonwriter who is published and one who is unpublished. The simplest nontrivial solutions of the constraints the rank criterion imposes are analogy structures of the kind reported for word embeddings since the classic [18].

![](images/7e5238672affd4aba421e15d20ac127e4d3e7601cc5d7566adf242928fcaa85a.jpg)

Figure 9: The parallelogram forced on every minimal exact geometry for the four-entity lexicon (simplified from Figure 4): $h ( d _ { 1 } \bar { ) } - h ( d _ { 2 } \bar { ) } = h ( d _ { 3 } ) - h ( d _ { 4 } )$ , so the displacement for published is the same, whether taken from a writer or from a nonwriter, and likewise for write. The single admissible dependence $\mathbf { c } = ( 1 , - 1 , - 1 , 1 )$ is this figure.

Remark 8.1 (A note on parallelogram). Corollary 5.3 gives a parallelogram kernel to every minimal exact geometry for a two-feature lexicon onfour entities, an exact geometry ofhigher rank having kernel zero, and the empirical literature says trained geometries (approximately) have parallelogram kernelsforfeature pairs ofthis shape; the two routes to the parallelogram are distinct: the present derivation applies to exact geometries, and the measured geometries of Section 8.4 are inexact, so the analogies observed in trained embeddings are accountedforfrom the training objective, through the co-occurrence statistics itfactorizes [6, 2].

In general, the admissible dependences are row $[ \mathbf { T } ; \mathbf { 1 } ] ^ { \perp }$ , and integer vectors in that complement with two entries +1 and two entries −1 are exactly the parallelograms the lexicon permits at all. This is not deep, but follows from pairs of entity pairs that agree on feature difference.

## Proposition 8.3 (Boolean cube)

Let P consist of m binary features on the $V = 2 ^ { m }$ entities of $\{ 0 , 1 \} ^ { m }$ , feature k having truth row $x \mapsto x _ { k }$ . Then $\operatorname { r a n k } [ \mathbf { T } ; \mathbf { 1 } ] = m + 1$ , and the space of admissible dependences $\mathrm { r o w } [ \mathbf { T } ; \mathbf { 1 } ] ^ { \perp }$ , of dimension $2 ^ { m } - m - 1$ , is spanned by the parallelogram vectors ${ \bf e } _ { x } - { \bf e } _ { x + { \bf e } _ { k } } - { \bf e } _ { z } + { \bf e } _ { z + { \bf e } _ { k } }$ over coordinates k and points $x , z$ with $x _ { k } = z _ { k } = 0$

Proof. The rows $x \mapsto x _ { k }$ and $x \mapsto 1$ are the affine functions’ basis on the cube, and are independent, giving the rank. Each parallelogram vector is orthogonal to every affine function $\begin{array} { r } { f ( x ) = a _ { 0 } + \sum _ { k } a _ { k } x _ { k } } \end{array}$ , since $f ( x ) - f ( x + \mathbf { e } _ { k } ) -$ $f ( z ) + f ( z + { \bf e } _ { k } ) = - a _ { k } + a _ { k } = 0 ,$ , so the parallelogram span lies in row $[ \mathbf { T } ; \mathbf { 1 } ] ^ { \perp }$

Conversely, let $f \in \mathbb { R } ^ { V }$ be orthogonal to every parallelogram vector; then $f ( x + \mathbf { e } _ { k } ) - f ( x ) = f ( z + \mathbf { e } _ { k } ) - f ( z )$ for all x, z with $x _ { k } = z _ { k } = 0$ , so the increment along coordinate k is a constant $a _ { k } .$ , and induction on the number of nonzero coordinates gives $\begin{array} { r } { f ( x ) = f ( 0 ) + \sum _ { k } a _ { k } x _ { k } } \end{array}$ , an affine function. The orthogonal complement of the parallelogram span is, therefore, the space of affine functions, and the parallelogram span is its complement. □

![](images/e1cb7191ff378f7862f0ce227f2be99e93e56e500e9f6aa7737c132a0206c5ae.jpg)  
Figure 10: The Boolean cube of Proposition 8.3 for $m = 3 { : }$ eight entities, three features, and a minimal exact geometry in $\overline { { \mathbb { R } ^ { 4 } } }$ (drawn in three dimensions, with the affine offset suppressed). Every minimal exact geometry is an affine image of the cube, so each face is a parallelogram; the four independent faces span the admissible dependences, of dimension $2 ^ { 3 } - 3 - 1 = 4 .$ , and the affine functions $\begin{array} { r } { a _ { 0 } + \sum _ { k } a _ { k } x _ { k } } \end{array}$ are licensed by the lexicon.

In a minimal exact geometry for a full factorial feature lexicon, every coordinate difference is, thus, a constant vector, which is the setting in which analogy by vector arithmetic is exact; an exact geometry of higher rank satisfies a subset of these equalities, and the free geometry keeps the $2 ^ { m }$ entity vectors affinely independent.

## 8.4 Measurement

We turn, now, to implementation and experiment<sup>6</sup>. We computed a held-out error ${ \hat { \delta } } ,$ written so as to keep it apart from the defect δ of Definition 8.1, the probe, the principal angles, and the dimension sweep of Corollary 5.1, on two classic lexicons against two likewise classic embeddings. The geometries are a 300-dimensional GloVe embedding [23] and the 300-dimensional word2vec embedding of [17]; each is used as its geometry matrix H without centering, whitening, normalization, or truncation. The projection residual of Proposition 8.1 depends on row(H) alone; the held-out error below depends on the coordinates through its penalty, and is reported on the coordinates as published.

The first lexicon uses the McRae feature norms [15]. Following the norms’ own inclusion threshold, a feature holds of a concept when at least five participants produced it. Entries below the threshold enter T as false, so the truth-conditional reading treats nonproduction as a negative judgment, which is an assumption about the norms rather than a datum in them.

Concepts the norms distinguish by sense, such as bat in its animal and baseball senses, share one word vector and receive the union of their features. The 541 concepts collapse to 532 words, all present in GloVe and 531 in word2vec. Retaining features assigned to at least fifteen words, and to at most fifteen fewer than all of them, gives 76 predicates and $\mathrm { r a n k } [ \mathbf { T } ; \mathbf { 1 } ] = 7 7$ in both embeddings.

The second lexicon uses WordNet hypernyms [7]. We select monosemous nouns among each embedding’s twenty thousand most frequent tokens. Hypernyms of depth at least four with 30–500 members become predicates. GloVe supplies 6271 nouns and 171 predicates, with 165 distinct extensions and rank $[ \mathbf { T } ; \mathbf { 1 } ] = 1 6 6$ ; for word2vec the figures are 5257 nouns and 143 predicates, with 139 distinct extensions and $\mathrm { r a n k } [ \mathbf { T } ; \mathbf { 1 } ] = 1 3 9$ . Predicates with identical extensions remain separate rows.

In every case, rank $[ \mathbf { T } ; \mathbf { 1 } ] < d = 3 0 0 < V$ , which places all four cells in the compressed regime. The available dimension therefore permits exact monadic compression, although whether a pretrained geometry realizes it remains to be tested.

The full-domain quantity is the affine defect of Proposition 5.1, normalized by the centered norm of the truth row,

$$
\delta ^ { + } ( P ; \mathbf { H } ) = \frac { \left\| \mathbf { t } _ { P } \left( I _ { V } - ( \mathbf { H } ^ { + } ) ^ { \dagger } \mathbf { H } ^ { + } \right) \right\| _ { 2 } } { \left\| \mathbf { t } _ { P } - \bar { t } _ { P } \mathbf { 1 } \right\| _ { 2 } } , \qquad \bar { t } _ { P } = | P | / V ,
$$

so that a constant readout scores 1 and an exact affine lift scores 0. The held-out error ${ \hat { \delta } } ( P ; { \mathbf { H } } )$ is a predictive quantity, and differs from it in target, in regularization, and in normalization. The entities are split into five folds $F _ { 1 } , \ldots , F _ { 5 } ;$ for each $k ,$ a readout $\big ( \mathbf { w } ^ { ( k ) } , b ^ { ( k ) } \big )$ is fit by ridge regression over $\mathbf { H } ^ { + }$ on the entities outside $F _ { k }$ , with the intercept unpenalized, and

$$
\widehat { \delta } ( P ; \mathbf { H } ) = \left( \frac { \sum _ { k } \sum _ { i \in F _ { k } } \left( \mathbf { w } ^ { ( k ) } h ( d _ { i } ) + b ^ { ( k ) } - \mathbf { t } _ { P } ( i ) \right) ^ { 2 } } { \sum _ { k } \sum _ { i \in F _ { k } } \left( \mathbf { t } _ { P } ( i ) - \bar { t } _ { P } ^ { ( k ) } \right) ^ { 2 } } \right) ^ { 1 / 2 } ,
$$

with $\bar { t } _ { P } ^ { ( k ) }$ the mean of $\mathbf { t } _ { P }$ over $F _ { k } ,$ so predicting the fold mean scores 1. The penalty is chosen per predicate from $\{ 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , 1 , 1 0 , 1 0 ^ { 2 } , 1 0 ^ { 3 } \}$ by the minimum of this same held-out error, a selection that biases $\hat { \delta }$ downward; choosing it on inner folds of the training data instead moves the medians below by at most 0.01. The null is the same statistic on a Gaussian matrix of the same shape. A positive $\hat { \delta }$ is, therefore, consistent with an exact affine lift on the full domain, which a penalized fit to a subset need not recover, and the two quantities are reported side by side. The probe is a logistic regression with unit inverse regularization and balanced class weights, its probabilities cross-fitted over stratified five-fold splits and scored by the area under the curve. Ranks are taken at the default tolerance of the numerical library, the largest singular value times the larger matrix dimension times machine precision; principal angles are the arccosines of the singular values of $Q _ { U } ^ { \top } Q _ { W }$ , with $Q _ { U }$ and $Q _ { W }$ orthonormal bases of row(H) and row[T; 1] from singular value decompositions at a relative cutoff of $1 0 ^ { - 1 0 }$ . Table 2 gives medians over predicates.

<table><tr><td>Lexicon</td><td>Geometry</td><td>V</td><td>rank[T; 1]</td><td> $\delta ^ { + } \left( \mathbf { m i n } \right)$ </td><td> $\hat { \delta }$ </td><td> $\hat { \delta }$  null</td><td>AUC</td><td>Angle band</td></tr><tr><td>McRae</td><td>GloVe</td><td>532</td><td>77</td><td>0.51 (0.25)</td><td>0.85</td><td>1.03</td><td>0.95</td><td>10–18 (20–26)</td></tr><tr><td>McRae</td><td>word2vec</td><td>531</td><td>77</td><td>0.53 (0.25)</td><td>0.89</td><td>1.03</td><td>0.96</td><td>10–20 (21–27)</td></tr><tr><td>WordNet</td><td>GloVe</td><td>6271</td><td>166</td><td>0.89 (0.66)</td><td>0.93</td><td>1.02</td><td>0.95</td><td>32–45 (68–70)</td></tr><tr><td>WordNet</td><td>word2vec</td><td>5257</td><td>139</td><td>0.86 (0.54)</td><td>0.92</td><td>1.02</td><td>0.96</td><td>25–44 (67–69)</td></tr></table>

Table 2: Median full-domain affine defect $\overline { { \delta ^ { + } } }$ , with its minimum over predicates in parentheses; median held-out error ${ \hat { \delta } } ,$ with Gaussian null; median probe AUC; and the second through tenth principal angles in degrees between row(H) and row[T; 1] (null given in parentheses). The first principal angle is 1.1, 2.1, 2.1, and $5 . 2$ degrees; the angle between the 1 row itself and row(H) is 1.3, 2.6, 2.1, and 5.4 degrees, so the first principal direction lies near the 1 row.

No predicate admits an exact affine lift in any of the four cells: the median full-domain defect ranges from 0.51 to 0.89 (see Table 2), and the smallest over all predicates is 0.25. The embeddings, nevertheless, have lower defects than Gaussian controls. For a centered truth row and a random d-dimensional subspace of $\mathbf { 1 } ^ { \perp }$ , the expected squared defect is $1 - d / ( V - 1 )$ , and the observed control medians match its square root to two decimals.

Jointly over the lexicon, the principal angles give the same account of the obstruction: the lexical row space lies closer to row(H) than the Gaussian controls do, and the constant row is nearly contained in it, while the smallest principal angle stays positive in every cell, beyond numerical tolerance. Writing

$$
U = \mathrm { r o w } ( { \bf H } ) , \qquad S = \mathrm { r o w } [ { \bf T } ; { \bf 1 } ] ,
$$

we, therefore, have

$$
U \cap S = \{ 0 \} , \qquad \left( U + \mathrm { s p a n } \{ { \bf 1 } \} \right) \cap S = \mathrm { s p a n } \{ { \bf 1 } \} .
$$

The second equality follows because $\mathbf { 1 } \in S \colon \mathrm { i f ~ } u + c \mathbf { 1 } \in S$ with $u \in U$ , then $u \in U \cap S$ . Thus no predicate of the lexicon admits an exact linear readout, and none other than a constant one an exact affine readout, agreeing with the individual defects.

Under the held-out error, recovery is imperfect as well: only one McRae predicate, musical instrument, has $\hat { \delta } < 0 . 5$ , and none does on WordNet. Held-out error and probe AUC are strongly negatively correlated on McRae $( \rho = - 0 . 8 6$ and

−0.82), so predicates with better discrimination generally have lower prediction error. High AUC, however, establishes neither exact affine recovery nor strict separability.

We test separability directly by seeking w and b such that

$$
\mathbf w h ( d _ { i } ) + b \geq 1 \quad \mathrm { i f } P ( d _ { i } ) = 1 , \qquad \mathbf w h ( d _ { i } ) + b \leq - 1 \quad \mathrm { o t h e r w i s e } .
$$

Every McRae predicate is separable in both embeddings and their Gaussian controls, so this test does not distinguish the geometries there. On WordNet, GloVe separates 159 of 171 predicates and word2vec 136 of 143, compared with 87 and 80 in the controls.

Size accounts for the controls, which separate no predicate above 63 members in the GloVe cell or 61 in the word2vec cell. The geometries separate every predicate up to 100 members, 19 of 22 and 21 of 24 between 100 and 200, and 3 of 12 and 3 of 7 above; among the predicates larger than any sampled control managed, 62 of 74 over GloVe and 53 of 60 over word2vec are half-spaces. Failures include action, activity, and content, whereas city, municipality, and urban area are separable in both embeddings. Urban area gives a direct instance of Proposition 8.2, strictly separable at affine defects of 0.68 and 0.54; district and administrative district give the converse, the two predicates with the lowest held-out error over GloVe being among its failures.

Replacing first-sense WordNet labels with monosemous assignments raises median AUC from 0.89 to 0.95, and lowers held-out error only from 0.94 to 0.93. Retaining progressively more principal directions of H lowers held-out error gradually, without a pronounced transition at the lexicon’s rank, which Corollary 5.1 permits: the bound guarantees that some geometry of sufficient dimension carries the lexicon, not that a truncation of a pretrained geometry does.

Recovery varies by predicate type as well: on McRae, taxonomic predicates have median held-out errors of 0.65 and 0.71, against 0.87 and 0.90 for attributive predicates, with color and size worst; on WordNet, administrative and geographic categories and substance nouns are predicted best, and abstract nouns such as idea and information worst.<sup>7</sup>

![](images/a7a7702e4add04d6cd9bd9511b052d24aa542c50cabdcb9cd014c0e398dc65f8.jpg)  
Figure 11: Per-predicate held-out error $\hat { \delta }$ under the isotonic readout against cross-validated probe AUC, drawn from the per-predicate tables. Predicates concentrate at high AUC and high error: discriminable and poorly recovered. The lower right corner, low error at high AUC, is nearly empty.

Monotone transformations of the affine score lower the held-out error to between 0.77 and 0.86, against Gaussian controls near 1.00, a gain over the ridge readout of 0.03 to 0.06 on McRae and 0.10 to 0.15 on WordNet. Both are fit within training folds, isotonic regression on cross-fitted training scores.

Recovery is much better in a small tail: for musical instrument, on eighteen concepts, isotonic error falls to 0.16 and 0.05, and birds and their parts, fruit, cities, and countries follow between 0.40 and 0.60, at AUC above 0.98. The tenth percentile of the isotonic error lies between 0.54 and 0.68, so substantial error remains for most predicates.

A two-layer readout with 64 hidden units (validated on synthetic data requiring nonlinear recovery) raises median error on McRae by 0.03 and 0.04 relative to isotonic regression, and lowers it on WordNet by 0.06 and $0 . 0 7 ;$ , to 0.77 and $0 . 7 5$ , while its AUC does not exceed the logistic probe’s. These are results about held-out entities: on the finite domain itself, distinct entity vectors permit recovery of every predicate by an unrestricted decoder.

No predicate of either lexicon admits an exact linear or affine lift, so the geometries lie outside the exact regime of Theorem 5.1. Many predicates are, nevertheless, strictly separable, placing the WordNet cells inside the thresholded regime of Proposition 8.2 well beyond the sampled controls, and the McRae cells inside it at a rate the controls match. Held-out recovery varies across predicates and remains imperfect under every readout family examined. Exact representability, separability, and predictive performance, therefore, give distinct assessments of one geometry.

## 8.5 Training toward exact regime

Whether the exact regime is reachable at the dimensions current embeddings use, and at what distributional cost, is the question of this section.

We train a geometry $\mathbf { H } \in \mathbb { R } ^ { d \times V }$ under a mixed objective. Let $\mathbf { E } \in \mathbb { R } ^ { 3 0 0 \times V }$ be the pretrained geometry with its row means removed, and $\mathbf { T } _ { c }$ the truth matrix with its row means removed. For readouts $\mathbf { \bar { W } } \in \mathbb { R } ^ { 3 0 0 \times d }$ and $\dot { \mathbf { W } } _ { T } \in \mathbb { R } ^ { | \mathcal { P } | \times }$ ×d and an intercept $\mathbf { w } _ { 0 } \in \mathbb { R } ^ { | \mathcal { P } | }$

$$
\mathcal { L } ( \mathbf { H } ) = ( 1 - \lambda ) \operatorname* { m i n } _ { \mathbf { W } } \frac { \| \mathbf { W } \mathbf { H } - \mathbf { E } \| _ { F } ^ { 2 } } { \| \mathbf { E } \| _ { F } ^ { 2 } } + \lambda \operatorname* { m i n } _ { \mathbf { W } _ { T } , \mathbf { w } _ { 0 } } \frac { \| \mathbf { W } _ { T } \mathbf { H } + \mathbf { w } _ { 0 } \mathbf { 1 } ^ { \top } - \mathbf { T } \| _ { F } ^ { 2 } } { \| \mathbf { T } _ { c } \| _ { F } ^ { 2 } } ,\tag{4}
$$

so the distributional term is the squared relative error of reconstructing the pretrained vectors linearly from H, and the truth term is the squared relative residual of the affine lift of Proposition 5.1, taken jointly over the lexicon. The objective is minimized in these squared terms; the numbers reported below, and plotted in Figures 12 and $^ { 1 3 , }$ are their square roots, written ϵ(H) for the distributional reconstruction error and $\delta _ { \mathcal { P } } ( \mathbf { H } )$ for the joint truth defect, so that both are relative norms on the scale of $\delta ^ { + }$ . Both inner minima are least-squares problems with closed-form solutions, and $\mathcal { L }$ is minimized by alternating least squares between H and the readouts, with a ridge of $1 0 ^ { - 8 }$ on the H solve and the rows of H renormalized after each step, since $\mathcal { L }$ is invariant under $G L ( d )$ acting on $\mathbf { H } ;$ each run starts from a Gaussian H with a fixed seed and takes forty iterations. Every entity is a column of $\begin{array} { r } { \check { \bf H } , } \end{array}$ so the experiment is transductive, as an embedding layer is. $\mathrm { A t } \lambda = 0 .$ , training reconstructs the pretrained geometry; at $\lambda = 1$ , the truth term alone is minimized, the surplus directions of H above the rank are untrained, and the distributional error at $\lambda = 1$ carries no information, so the frontier is read at $\lambda \in ( 0 , 1 )$ ).

Two quantities are available in closed form.

1. The minimum of the truth term over all d-dimensional geometries is $\begin{array} { r } { \left( \sum _ { k > d } \sigma _ { k } ^ { 2 } \right) ^ { 1 / 2 } / \| \mathbf { T } _ { c } \| _ { F } } \end{array}$ for the singular values $\sigma _ { k }$ of $\mathbf { T } _ { c } ,$ since the intercept absorbs the row means; this floor is positive for $d \mathbf { \Sigma } < \mathrm { \ r a n k } { \bf T } _ { c } \mathbf { \Sigma } =$ $\mathrm { r a n k } [ { \bf T } ; { \bf 1 } ] - 1$ by Corollary 5.5, and zero from there on, which is Corollary 5.1 in the affine form of Proposition 5.1, and it fixes the location of the exactness transition in advance.

2. The second is a linear-exact benchmark: the variance retained at dimension d under the constraint row $\mathbf { ( H ) \supseteq }$ row $\lceil \mathbf { T } ; \mathbf { 1 } \rceil$ of Theorem 5.1 is the fraction of $\| \mathbf { E } \| _ { F } ^ { 2 }$ captured by the best d-dimensional row space containing row $[ \mathbf { T } ; \mathbf { 1 } ]$ , namely row[T; 1] together with the top $\bar { d \mathrm { - } \mathrm { r a n k } [ \mathbf { T } ; \mathbf { 1 } ] }$ principal directions of E projected off it, and at a least-squares optimum the retained variance is $1 - \epsilon ^ { 2 }$

The benchmark is one dimension more constrained than the training criterion, which is affine, and needs row(H) to contain a complement of 1 in row[T; 1], such as row $( \mathbf { T } _ { c } ) ;$ ; the affine benchmark, with row $\left( \mathbf { T } _ { c } \right)$ in place of row $[ \mathbf { T } ; \mathbf { 1 } ]$ retains at least as much, and the two differ by at most the variance of one direction.

The grid runs over $d \ \in \ \{ 1 0 , 2 0 , 4 0 , 8 0 , 1 2 0 , 1 6 0 , 2 0 0 , 3 0 0 \}$ and $\lambda \in \{ 0 , 0 . 1 , 0 . 5 , 0 . 9 , 1 \}$ . A predicate counts as recovered within tolerance when its affine defect on the trained geometry, $\| { \bf w } _ { P } { \bf H } + b _ { P } { \bf 1 } ^ { \top } - { \bf t } _ { P } \| _ { 2 } / \| { \bf t } _ { P } - { \bar { t } } _ { P } { \bf 1 } \| _ { 2 }$ with the readout refit by least squares, is below 0.05; Figure 12 plots this fraction. The McRae cell with word2vec in this was matched to the embedding by the training loader, which looks tokens up as given, whereas the diagnostics loader of Section 8.4 adds a case fallback; that holds $\stackrel { \cdot } { V } = 5 2 9$ , one predicate falls below fifteen positives, and it trains on 75 predicates with rank $[ \mathbf { T } ; \mathbf { 1 } ] = 7 6 $ , while Table 2 reports the diagnostics cell with 76 predicates and rank $7 7 ;$ the dotted line of Figure 12 for that cell sits at 75 accordingly.

$\mathbf { A } \mathbf { t } \ \lambda = 1$ , the optimizer attains the closed-form floor of $\delta _ { \mathcal { P } }$ at every grid point: to four decimals where the floor is positive (on McRae 0.702, 0.554, and 0.350 at $d = 1 0 , 2 0$ , 40; on WordNet with GloVe 0.014 at $d = 1 6 0 ;$ with word2vec 0.058 at $d = 1 2 0 )$ , and to $1 0 ^ { - 1 3 }$ where the floor is zero, which is every grid point at or above ran $\mathbf { \tau } _ { \ [ \mathbf { T } ; \mathbf { 1 } ] } - 1$ The affine exact regime is, therefore, reached at numerical tolerance at $d = 8 0$ and above on McRae, at $d = 2 0 0$ and above on WordNet with GloVe, and at $d = 1 6 0$ and above with word2vec, and the location of the transition is the floor’s, rank $[ { \bf T } ; { \bf 1 } ] - 1$ , with the grid serving to check that the optimizer finds it. The per-predicate fraction within tolerance at $\lambda = 1$ is 1 at those points, 0.947 at $d = 1 6 0 < 1 6 5$ on WordNet with GloVe, and 0.706 at $d = 1 2 0 < 1 3 8$ with word2vec, which is the approximate carriage a positive floor admits: below the rank no geometry carries the whole lexicon, and most of it can still lie within tolerance.

At $d = 3 0 0$ , the constraint is relatively cheap on McRae and relatively affordable on WordNet: the linear-exact benchmark retains 99.2 and 98.5 percent of the pretrained variance on McRae and 82.5 and 79.5 percent on WordNet, the constraint having rank 166 and 139 over a spectral tail at $V \approx 6 0 0 0$ . On the frontier at $\lambda = 0 . 9 , \delta _ { \mathcal { P } }$ is 0.003 and 0.004 on McRae at $\epsilon = 0 . 0 8 5$ and 0.121 (retention 99.3 and 98.5 percent), with every predicate within tolerance; on WordNet it is 0.046 and 0.044 at $\epsilon = 0 . 3 6 5$ and 0.415 (retention 86.7 and 82.8 percent), with 0.75 of the predicates within tolerance after forty iterations.

The affine exact regime is, therefore, reached at numerical tolerance at $d = 8 0$ and above on McRae, at $d = 2 0 0$ and above on WordNet with GloVe, and at $d = 1 6 0$ and above with word2vec, which are the grid points at or above the minimal affine dimension of Corollary 5.5.

![](images/abc3f63246d3c6a69abaeefeb4234427e0e9ef72e868cb962780c3d56ad110fd.jpg)  
Figure 12: Fraction of predicates with affine defect below 0.05 on the trained geometry (recovered within tolerance), after alternating least squares at $\lambda = 1$ , against the trained dimension $d \in \{ 1 0 , 2 0 , 4 0 , 8 0 , 1 2 0 , 1 6 0 , 2 0 0 , 3 0 0 \}$ , drawn from the run’s output. Dotted lines mark the minimal affine dimension rank $\left[ \mathbf { T } ; \mathbf { 1 } \right] - 1$ of Corollary 5.5: 76 and 75 (McRae; the word2vec training cell holds 75 predicates, see the text), 165 (WordNet, GloVe), 138 (WordNet, word2vec). Wherever the fraction reads 1, the joint residual is below $1 0 ^ { - 1 3 }$ ; at the two grid points just below that dimension it equals the closed-form minimum, 0.014 and 0.058.

Reaching the floor on the training entities shows that an embedding layer attains the affine exact regime under the truth objective; the frontier shows that most of the distributional variance survives within tolerance of it; exact realizability at $d \geq \mathrm { r a n k } [ \mathbf { T } ; \mathbf { 1 } ]$ is Corollary 5.1, and the experiment shows that the optimizer finds it. The result is transductive: generalization of the truth-conditional structure to unseen entities, and acquisition of the same geometry from distributional training alone, remain open.

## 9 Discussion

Consider the distinction between representability and representation. The free construction establishes when a truthconditional lexicon can be carried exactly by a finite-dimensional vector space; what follows is the empirical analysis, in which we ask how closely existing representations approach that construction. The defect introduced above makes this quantitative: zero defect means exact, while positive defect measures difference from it.

We showed that ordinary distributional embeddings already lie substantially closer to the truth-conditional geometry than a random subspace of the same dimension; this is consistent with the fact that semantic features are often linearly represented in learned vector spaces [22]; it also gives a geometric interpretation of superposition [5]: when the available dimension is below the minimum required for exactness, several truth conditions must share dimensions, and the resulting dependencies appear as nonzero defect. The principal angles (as reported in Table 2), therefore, quantify the extent to which a distributional geometry already contains the structure required by the lexicon, as opposed to mere generic similarity.

![](images/eab9a2bb23e8da6e24fedaca2cb077b96cdace05ea58defd8d8f1a432b45c534.jpg)  
Figure 13: The frontier at $d = 3 0 0 \colon$ joint truth defect $\delta _ { \mathcal { P } }$ against distributional reconstruction error ϵ, both relative, for $\lambda \in \{ 0 . 1 , 0 . 5 , 0 . 9 \}$ after forty iterations, all four cells. On McRae, δ<sub>P</sub> reaches 0.003 and 0.004 at $\epsilon = 0 . 0 8 5$ and 0.121; on WordNet, 0.046 and 0.044 at 0.365 and 0.415, where the optimizer is still descending at $\lambda = 0 . 9$

Distributional learning, by itself, leaves every predicate of both lexicons outside the exact regime, by the affine defects and the principal angles, and the held-out error orders the predicates: concrete category predicates are recovered far better than attributive and abstract predicates, and monotone (or nonlinear) readouts recover much of the same ordering. This is compatible with the literature on conceptual spaces [8], while also placing a limit on a purely geometric account: a region in a conceptual space need not constitute an exact extension. Human categorization provides an independent reason not to expect exact linear separability as a universal property [16].

At d = 300, the truth-conditional constraint can be imposed, while preserving most of the variance of the original embeddings. On McRae, the lexicon is brought within tolerance with very little distributional loss; on WordNet, the constraint is more expensive, and remains compatible with substantial retention. Under the truth term alone, the optimizer reaches the affine exact regime at numerical tolerance at every trained dimension from rank[T; 1] − 1 upward, where the closed-form floor is zero, and matches the positive floor below it. Dimension is thereby eliminated as the reason the pretrained embeddings fail to carry the lexicon; the objective, the corpus, and the labels remain as candidates, which the present experiments leave apart from one another.

This has a useful consequence for the relation between distributional and truth-conditional semantics: the two need not compete for representation space. A single geometry can retain substantial distributional structure, while also carrying a truth-conditional lexicon. The experiments leave open whether language exposure alone produces such a geometry. Skip-gram’s relation to shifted PMI factorization [14] gives the distributional geometry a corpus-level interpretation, but there is nothing in that objective that requires the resulting space to satisfy the truth-conditional constraints. The present results, therefore, support a weaker (and more precise) claim: distributional learning supplies informationfrom which truth-conditional structure may be (partially) recovered; exactness requires either an additional constraint or some mechanism that supplies equivalent information.

The framework also clarifies the status of compositionality. Once the leaves of a derivation are represented exactly, the homomorphism conditions determine the corresponding Boolean composition without further learning. Outside the exact regime, each leaf is only approximately represented, so compositional error can accumulate. A system may, therefore, perform well on individual semantic probes, while failing a composed entailment. Good distributional similarity alone, consequently, leaves exact logical behavior undetermined.

The empirical scope of the present study is deliberately narrower than the formal framework we are likewise developing; here, the measurements concern one-place predicates over static entity geometries. Relations, represented bilinearly on H ⊗ H, and higher arities follow the same geometric strategy, but were not evaluated here, which we leave for future work. Quantifiers are treated in Proposition 5.3 for the exact regime, where they compute by bilinear accumulation on the readouts, and their behavior under positive defect remains to be measured. Likewise, the formal conditions governing computation after the embedding layer raise a separate question. Proposition 5.2 confines the present compression result to the leaves of a derivation; whether attention and feed-forward computation can, themselves, realize the required multilinear maps remains open.

The vector logic supplies a specification of what a representation carrying a truth-conditional structure has to satisfy, and the defect turns the specification into a measurable property of an empirical geometry, so that the question whether a learned representation carries such a structure has a computable answer. That is the principal role of the framework; the origin of the gradient observed here, and the emergence of truth-conditional structure from language exposure alone, lie outside what it decides.

## 10 Conclusion

A truth-conditional lexicon imposes linear constraints on the representation space; the rank of those constraints gives the minimum dimension for exactness, while the defect measures how closely a lower-dimensional (or otherwise unconstrained) representation approaches that ideal. The empirical results show that existing distributional embeddings contain substantial structure relevant to the lexicon, while every predicate of both lexicons fails the exact criterion, linear and affine, over both geometries. Once the truth-conditional constraint enters the objective, the affine exact regime is attained at numerical tolerance from the predicted dimension rank $[ \mathbf { T } ; \mathbf { 1 } ] - 1$ upward, and, at $d = 3 0 0$ , the lexicon comes within tolerance of it while most of the original distributional variance is retained; distributional training, by itself, leaves exactness to a further constraint.

The formal framework extends beyond the monadic case studied here. Relations and higher-arity predicates can be treated on tensor-product spaces, quantifiers require corresponding operators on compressed readouts, and contextual representations can be evaluated layer by layer. The most direct empirical continuation is to replace the pretrained distributional matrix in the mixed objective of Section 8.5 with corpus statistics to test whether exact truth-conditional structure can emerge from co-occurrence information alone, and at what distributional cost; we expect not.

What does it mean for a vector geometry to carry a truth-conditional semantics compositionally is answered here for finite monadic lexicons and binary relations; how far does an empirical geometry fall short of that condition is measured for two lexicons and two embeddings; how a learning system might acquire such a representation is another problem.

## Acknowledgments

The observation that an embedding layer is a linear map on one-hot inputs, and its pedagogical framing, are due entirely to Sebastian Raschka, which set this paper in motion.

## References

[1] Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes, 2018.

[2] Carl Allen and Timothy Hospedales. Analogies explained: Towards understanding word embeddings. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 223–231. PMLR, 2019.

[3] Yonatan Belinkov. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1):207–219, March 2022.

[4] Gemma Boleda. Distributional semantics and linguistic theory. Annual Review of Linguistics, 6:213–234, 2020.

[5] Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy models of superposition, 2022.

[6] Kawin Ethayarajh, David Duvenaud, and Graeme Hirst. Towards understanding linear word analogies. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 3253–3262, Florence, 2019. Association for Computational Linguistics.

[7] Christiane Fellbaum, editor. WordNet: An Electronic Lexical Database. MIT Press, Cambridge, MA, 1998.

[8] Peter Gärdenfors. Conceptual Spaces: The Geometry ofThought. The MIT Press, 2000.

[9] Irene Heim and Angelika Kratzer. Semantics in Generative Grammar. Blackwell, 1998.

[10] John Hewitt and Percy Liang. Designing and interpreting probes with control tasks, 2019.

[11] Edward L. Keenan and Jonathan Stavi. A semantic characterization of natural language determiners. Linguistics and Philosophy, 9(3):253–326, 1986.

[12] Christopher Kennedy. Vagueness and grammar: the semantics of relative and absolute gradable adjectives. Linguistics and Philosophy, 30(1):1–45, February 2007.

[13] Angelika Kratzer. Modality. In Arnim von Stechow and Dieter Wunderlich, editors, Semantics: An International Handbook of Contemporary Research, pages 639–650. de Gruyter, Berlin, 1991.

[14] Omer Levy and Yoav Goldberg. Neural word embedding as implicit matrix factorization. In Z. Ghahramani, M. Welling, C. Cortes, N. Lawrence, and K. Weinberger, editors, Advances in Neural Information Processing Systems, volume 27. Curran Associates, Inc., 2014.

[15] Ken McRae, George S. Cree, Mark S. Seidenberg, and Chris Mcnorgan. Semantic feature production norms for a large set of living and nonliving things. Behavior Research Methods, 37(4):547–559, November 2005.

[16] D. Medin and P. Schwanenflugel. Linear separability in classification learning. Journal of Experimental Psychology: Human Learning and Memory, 7(5):355–368, 1981.

[17] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems 26, pages 3111–3119, 2013.

[18] Tomas Mikolov, Wen-tau Yih, and Geoffrey Zweig. Linguistic regularities in continuous space word representations. In Proceedings ofNAACL-HLT 2013, pages 746–751, Atlanta, GA, 2013.

[19] Eduardo Mizraji. Vector logics: The matrix-vector representation of logical calculus. Fuzzy Sets and Systems, 50(2):179–185, 1992.

[20] Richard Montague. English as a formal language. In Richmond Thomason, editor, Formal Philosophy: Selected Papers ofRichard Montague, pages 188–221. Yale University Press, New Haven, CT, 1974.

[21] Maximilian Nickel, Volker Tresp, and Hans-Peter Kriegel. A three-way model for collective learning on multirelational data. In Proceedings ofthe 28th International Conference on International Conference on Machine Learning, ICML’11, page 809–816, Madison, WI, USA, 2011. Omnipress.

[22] Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 39643–39666. PMLR, 2024.

[23] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. GloVe: Global vectors for word representation. In Empirical Methods in Natural Language Processing (EMNLP), pages 1532–1543, 2014.

[24] PyTorch Contributors. torch.nn.linear. https://pytorch.org/docs/stable/generated/torch.nn. Linear.html. Accessed 2026-08-15.

[25] Daniel Quigley. A vector logic for extensional formal semantics. Journal of Logic, Language and Information, 34(5):557–599, 2025.

[26] Daniel Quigley. A vector logic for intensional formal semantics, 2026. Under review, Journal of Logic, Language and Information; arXiv:2602.02940.

[27] Sebastian Raschka. Why can an embedding layer be interpreted as a linear layer applied to one-hot encoded tokens? https://sebastianraschka.com/faq/docs/embedding-linear-onehot.html. Accessed 2026-08-15.

[28] Sebastian Raschka. Build a Large Language Model (From Scratch). Manning, 2024.

[29] Dana Rubinstein, Effi Levi, Roy Schwartz, and Ari Rappoport. How well do distributional models capture different types of semantic knowledge? In Proceedings ofthe 53rd Annual Meeting ofthe Associationfor Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 726–730. Association for Computational Linguistics, 2015.

[30] Johan van Benthem. Essays in Logical Semantics, volume 29 of Studies in Linguistics and Philosophy. Reidel, Dordrecht, 1986.

[31] Jonathan Westphal and Jim Hardy. Logic as a vector system. Journal ofLogic and Computation, 15(5):751–765, 2005.