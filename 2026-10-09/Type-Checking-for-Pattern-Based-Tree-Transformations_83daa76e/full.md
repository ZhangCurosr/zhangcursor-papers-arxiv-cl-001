# Type-Checking for Pattern-Based Tree Transformations

C. Aiswarya #

Chennai Mathmatical Institute, India CNRS IRL ReLaX, India

Sahil Mhaskar<sup>1</sup> #

Chennai Mathematical Institute, India

M. Praveen # Chennai Mathmatical Institute, India CNRS IRL ReLaX, India

## Abstract

We introduce and study pattern-based tree transformations. As an illustrating example, consider a source pattern $( x \cdot y ) + ( x \cdot z )$ and a target pattern $x \cdot ( y + z )$ as a pair. This source pattern matches any expression e of the form $( e _ { 1 } \cdot e _ { 2 } ) + ( e _ { 1 } \cdot e _ { 3 } )$ (by substituting x with $e _ { 1 } , y$ with $_ { e _ { 2 } , }$ and z with e<sub>3</sub>) and the pair transforms it into the expression $e _ { 1 } \cdot ( e _ { 2 } + e _ { 3 } )$ as dictated by the target pattern. Note that in this example, the set of expressions that match the source pattern is not a regular tree language.

We propose a model of tree transformations given by a finite representation of a (possibly infinite) set of such (source pattern, target pattern) pairs. The expressive power of this model comes at the cost of undecidability of checking equivalence. Nevertheless, we show that the type-checking problem is decidable for our model of pattern-based tree transformations. The type-checking problem asks whether applying a given transformation to trees having a given regular property (type) preserves the property. Our decision procedure is by a reduction to the emptiness problem of alternating tree automata.

2012 ACM Subject Classification Theory of computation → Tree languages; Theory of computation → Transducers

Keywords and phrases Tree Transducers, Type-checking, Alternating Tree Automata, Patternmatching

Related Version Full version of a preprint accepted at FSTTCS 2026

## 1 Introduction

Tree transformations in various forms of machines and logics have been studied for applications such as XML processing, programming languages, and HTML sanitisation [5, 11, 12, 14]. Classical formalisms for specifying tree transductions such as finite-state top-down and bottom-up tree transducers, macro tree transducers, attribute grammars and MSO (Monadic Second-Order Logic)-definable graph transductions typically aim for robustness with respect to operations involving regular tree languages and decidability of the equivalence problem.

We introduce a model for tree transductions targeting scenarios where the domain of the transduction is not necessarily regular, and undecidability of equivalence does not render the model useless. Expanding on the example from the abstract, consider expressions of the form $t \cdot e _ { 1 } + t \cdot e _ { 2 } + \cdot \cdot \cdot + t \cdot e _ { n }$ , where n is any number. An equivalent expression is $t \cdot ( e _ { 1 } + e _ { 2 } + \cdot \cdot \cdot + e _ { n } )$ . Transformations like this are routinely used in Horner’s method, an eficient way to evaluate polynomials. We consider the corresponding expression trees and model this transformation by matching the source tree with a source pattern and the target tree with a target pattern. The patterns are themselves trees, which can have placeholder variables to represent subtrees, like t in the above example, which must repeat in multiple positions. The number n can be arbitrary, so we need infinitely many pairs of patterns. For the example above, for each $n > 0$ we need $\left( x \cdot e _ { 1 } + x \cdot e _ { 2 } + \cdot \cdot \cdot + x \cdot e _ { n } , \quad x \cdot \left( e _ { 1 } + e _ { 2 } + \cdot \cdot \cdot \cdot + e _ { n } \right) \right)$ as a (source pattern, target pattern) pair, where x is a placeholder variable intended to match any expression t. We represent such infinite sets (of pairs of patterns) using (standard) tree transductions. We allow placeholder variables in patterns to be substituted with any tree, as long as it satisfies a guard condition associated with the variable. The guard condition is given by a regular tree language called the guard language. It allows, for example, to restrict the expression t above to be a valid expression. In some situations we may want to restrict it further, say, to not contain any products inside. In this example, the set of source trees to which the transformation is applicable needs to have the same subtree t repeat in multiple positions. Thus the set of source trees on which transformations are defined need not be a regular tree language in general, which makes our model diferent from classical tree transductions.

The backbone of our model is a finite representation of the infinite set of (source pattern, target pattern) pairs. Such a finite representation can be any formalism for tree transductions; in this paper we have chosen to use bimorphisms for technical convenience. Bimorphisms are given by a regular seed language S and two morphisms $\phi _ { s r c } , \phi _ { t g t }$ . The transduction is defined as the relation $\{ ( \phi _ { s r c } ( t ) , \phi _ { t g t } ( t ) ) \mid t \in S \}$ . Over words, bimorphisms are equivalent to rational transductions [15]. Over trees, bimorphisms capture subclasses of tree transducers (cf. [4, Section 6.5] and [6]).

For example, consider the seed language given by ab1 $( 0 + 1 ) ^ { * } ( \oplus b 1 ( 0 + 1 ) ^ { * } ) ^ { * } c$ . This is a regular word language over the alphabet $\{ a , b , c , \oplus , 0 , 1 \}$ . We define the source and target morphisms $\phi _ { s r c } , \phi _ { t g t } \colon \{ a , b , c , \oplus , 0 , 1 \} \to \{ 0 , 1 , + , \cdot , x , ( , ) \} ^ { * }$ as given by the following: $\phi _ { s r c } : = \{ a \mapsto \varepsilon , b \mapsto x \cdot , c \mapsto \varepsilon , \oplus \mapsto + , 0 \mapsto 0 , 1 \mapsto 1 \}$ and similarly we define: $\phi _ { t g t } : =$ $\{ a \mapsto x \cdot ( , b \mapsto \varepsilon , c \mapsto ) , \oplus \mapsto + , 0 \mapsto 0 , 1 \mapsto 1 \}$ . We note that this bimorphism generates pairs of the form $\left( x \cdot e _ { 1 } + x \cdot e _ { 2 } + \cdots + x \cdot e _ { n } , \quad x \cdot \left( e _ { 1 } + e _ { 2 } + \cdots + e _ { n } \right) \right)$ where $e _ { i }$ is a nonzero binary number, from the seed word abe<sub>1</sub> ⊕ be<sub>2</sub> $\oplus \cdot \cdot \cdot \oplus b e _ { n } c .$ We may add a guard to x saying that x can take values only from $1 ( 0 + 1 ) ^ { * }$ , capturing completely the distributivity of multiplication over addition for binary numbers. Note that this only requires bimorphisms over words and defines transduction over words. If instead we allow the guard language of x to be any syntactically valid expression tree, this now defines transformations from trees to trees, though given by a bimorphism over words. In fact, if we wish $e _ { 1 } , e _ { 2 } \ldots e _ { n }$ to be also expressions, we would require a bimorphism over trees. This case is detailed as an example in Section 2.

Notice that bimorphisms over words which do not produce placeholder variables anywhere already capture rational relations over words. Since equivalence-checking is undecidable for rational word relations [8, 10], it is undecidable for our model as well. However we show that the type-checking problem is decidable, even for transductions defined by bimorphisms over trees and potentially producing variables in the output of both morphisms. We first motivate the need for type-checking in the next paragraph, and give insights to our algorithm in the paragraphs following that.

Continuing with our example from above, we may want to check that applying the transformation does not increase the alternation between the operators · and +. This can be checked by verifying that if the transformation is applied to any tree from a language L consisting of trees with alternation depth bounded by a constant c, the resulting tree is in

L as well. On the other hand, for reasons of numerical stability, we may be interested in ensuring that size of subtrees made up of only the + operator is bounded. It can be checked that this regular property is not preserved by the transformation. Instead of arithmetic expressions, if we consider Boolean expressions made up of $\wedge$ and ∨ operators, we can reason about expressions being in normal forms like conjunctive normal form (CNF), disjunctive normal form (DNF) etc. We consider the type-checking problem in general where the source trees have a property represented by a regular language $L _ { s r c }$ and output trees are required to have a property represented by another (possibly diferent) regular language $\boldsymbol { L } _ { t g t }$

The main idea of our decision procedure for the type-checking problem is the following observation: if the given instance of the type-checking problem is a negative instance, a counterexample is a single tree in the seed language S, satisfying multiple conditions under multiple transformations. On the source side, applying the morphism $\phi _ { s r c }$ to the seed tree and then applying a substitution for the placeholder variables should result in a tree in $L _ { s r c } .$ On the output side, applying the morphism $\phi _ { t g t }$ to the same seed tree and then applying the same substitution for the placeholder variables should result in a tree not in $\boldsymbol { L } _ { t g t }$ . These conditions can be checked by an alternating tree automaton running on trees in the seed language, thus reducing the type-checking problem to the emptiness-checking of alternating tree automata.

Along the way, we consider some related/simpler problems. The simple question of whether a given tree matches a given pattern is NP-complete, a result known already for words [4]. The special case of the type-checking problem where the transformation is given by a single pair of patterns is EXPTime-complete. This generalises the work of pattern-based word transformations [1] to trees.

## Related work

Pattern-based word transformations were considered in [1], but these consisted only of a single pair of patterns. We extend this in two ways — by having transformations of trees, and by having a formalism that consists of potentially infinitely many pairs of patterns.

There are many transducer models for trees. Our transformations subsume classical topdown and bottom-up tree transducers, even without any variables, thanks to the bimorphism feature. In fact it subsumes any relation on trees definable by bimorphisms [4, Chapter 6]. However it is incomparable to streaming tree transducers [2] or macro tree transducers [7]. The domains of transformations defined by the latter are regular tree languages, which need not be the case in our model due to the presence of variables. On the other hand our model cannot recognize transformations which reverse a word (a tree with unary alphabet and an end marker as a leaf), while macro tree transducers and streaming tree transducers can. For exactly the same reasons, our model is incomparable with k-Pebble Transducers [14]. See Appendix A for details.

A model of transition systems was studied in [9] where states are unranked, unordered forests and edges are given by tree rewriting using a pattern-based query language. These patterns are weaker than our patterns since a variable can represent only a single node in the tree and not a subtree like in our patterns. Reachability in this transition system is undecidable, but if the depth of every forest in the transition system is bounded it becomes decidable though nonelementary. Our type-checking problem is not addressing reachability in a transition system, but only a one-step rewriting. While the decidability result in [9] is limited to bounded depth trees, our model can handle unrestricted trees. Further, all our algorithms are of elementary complexity.

## Paper organisation

In Section 2, we illustrate the correspondence between the artefacts of our model and the parts of trees and transformations that the model represents. We recall some preliminaries about trees, contexts, nondeterministic tree automata and regular tree languages in Section 3.1 and formally define the model in Section 3.2. In Section 4, we give some examples of possible applications, which motivate our model as well as the type-checking problem that we consider, which is formally defined in Section 5. We give some intuition underlying the decision procedure in Section 6, motivating the concept of alternating tree automata and relational lifts that are formally defined in Section 7. The proof of decidability is put together in Section 8. Some of the detailed proofs are moved to the appendix due to space constraints.

## 2 Illustrating Example

As mentioned in the introduction, our objective of this section is to illustrate our model through an example. We want to capture transformations of the form depicted in Figure 1. We will define a suitable seed language and give bimorphisms that attain this.

![](images/cb912f9db401b0a5de477a286de14e0953c61204e36d0dc1e2793e4a7adbe865.jpg)  
Figure 1 The transformation of an expression following distributivity. Here $f , e _ { 1 } , \ldots e _ { 6 }$ are arbitrary expressions. If f is a variable instead of an expression, we get a (source pattern, target pattern) pair that captures this transformation.

Consider the transformation depicted in Figure 1. The blue shaded parts in the source expression and the target expression remain unchanged. Expressions are formed by the alphabet $\{ + , \times , 0 , 1 \}$ . The patterns would require a placeholder symbol $y$ in addition to the above alphabet. In order to generate the (source pattern, target pattern) pair capturing this transformation, we require a seed tree which carries this blue shaded part intact. We expect the morphisms $\phi _ { s r c }$ and $\phi _ { t g t }$ to be identity on this part. However, ${ _ y } ^ { \prime } { } ^ { \times } ,$ gets inserted above $e _ { i }$ in the source pattern, and at the root in the target pattern. In order to handle this we add new unary symbols a and b in the seed tree.

The seed tree and the morphisms $\phi _ { s r c } , \phi _ { t g t }$ for this transformation are depicted in Figure 2.

Note that, we require one such seed tree for each possible choice of $e _ { i }$ and the shape of the $ { \mathrm { ~  ~ \omega ~ } } ^ { \omega } + -  { \mathrm { k e r n e l } } ^ { 5 }$ , thus giving rise to an infinite seed language and hence an infinite set of (source pattern, target pattern) pairs capturing the transformation. Thankfully, the set of seed trees is tree-regular (need to check that the root is an a, every path has exactly one $b ,$ and the segment between a and b has only +s).

We now give an example of type-checking problem on this transformation. We define the product-depth of an expression as follows. The product-depth (pd) of a branch is the number of occurrences of × in that branch. The product-depth of an expression is the maximum of the product-depths of its branches. The classes of k-pd-bounded expressions are those with product-depth at most k. We may be interested in checking whether k-pd-boundedness is preserved by the above transformation. This is an instance of type-checking problem, since k-pd-bounded expressions form a regular set of trees.

![](images/834d04bafbb175f2582c010abadcacaa17c5cbc1511fc414925360d1afce0770.jpg)  
Figure 2 The seed tree and the morphisms generating the (source pattern, target pattern) pair capturing the transformation depicted in Figure 1.

Further examples, in particular the ones where we need multiple variables, are presented after formally defining our model, which is done next.

## 3 Formal Model

## 3.1 Preliminaries

## Sets and Relations

Let $\mathbb { N } : = \{ 1 , 2 , 3 , . . . \}$ be the set of strictly positive integers and let $ { \mathbb { N } } _ { 0 } : =  { \mathbb { N } } \cup \lbrace 0 \rbrace$ . For $n \in \mathbb { N } .$ we denote the set $\{ 1 , \ldots , n \}$ by [n]. As a convention, we let $[ 0 ] : = \emptyset$ . For a set X and $n \in \mathbb { N } .$ $X ^ { n }$ denotes the usual Cartesian power of X, consisting of all n-tuples of elements in X. As a convention, we let $X ^ { 0 } : = \{ \natural \}$ , a special singleton set.

For $n \in  { \mathbb { N } } _ { 0 }$ , an n-ary relation over X is a subset $R \subseteq X ^ { n }$ . We denote the set of all n-ary relations on X by Relations $( X , n )$ . Finally, we denote $\textstyle \bigcup _ { k = 0 } ^ { \infty }$ Relations $( X , k )$ by Relations(X).

We consider automata over finite, ranked trees, following definitions from [4].

▶ Definition 1 (Alphabet). A finite set $\Sigma : = \{ a _ { 1 } / k _ { 1 } , \ldots , a _ { n } / k _ { n } \}$ is called a finite ranked alphabet or alphabet, where a<sub>i</sub> are called the letters, and $k _ { i } = : a r i t y ( a _ { i } )$ are nonnegative integers called their corresponding arities for all $i \in [ n ]$ . We define pairwise disjoint sets $\Sigma _ { k } : = \{ a \in \Sigma \mid a r i t y ( a ) = k \} \subseteq \Sigma$ for all $k \in  { \mathbb { N } } _ { 0 }$ . We require that $\Sigma _ { 0 } \neq \emptyset$ . We also define the arity of $\Sigma : = \operatorname* { m a x } _ { i } k _ { i }$

▶ Definition 2 (Term, Tree). A term or a tree over a ranked alphabet Σ is a partial function $t : \mathbb { N } ^ { * } \to \Sigma$ with the domain Positions(t) satisfying:

$\begin{array} { r l } { = } & { { } P o s i t i o n s ( t ) } \end{array}$ is nonempty and $_ { p r e f i x - c l o s e d . }$

$$
\begin{array} { r } { = \forall u \in P o s i t i o n s ( t ) , t ( u ) \in \Sigma _ { k } , k \in \mathbb { N } _ { 0 } \implies \{ j | u j \in P o s i t i o n s ( t ) \} = [ k ] . } \end{array}
$$

We say that a tree t is finite if Positions(t) is finite.

In this paper, we will only deal with finite trees. We denote the set of all finite trees over Σ by Trees (Σ). A tree language is any subset of Trees (Σ). Regular tree languages are those recognised by tree automata as defined below.

▶ Definition 3 (Nondeterministic Bottom-up Tree {Transition System, Automaton}). Let Σ be a finite alphabet. A Nondeterministic Bottom-up Tree Transition System or NBTTS over Σ is defined as $\mathcal { T } : = ( Q , \Sigma , \Delta )$ , where $Q$ is a finite set of states, and $\Delta : \Sigma  R e l a t i o n s ( Q )$ is a function such that $\Delta ( a / k ) \in R e$ lations $( Q , k + 1 )$ , for all $a / k \in \Sigma$ . A Nondeterministic Bottom-up Tree Automaton or an NBTA is a tuple $\mathscr { A } : = ( Q , \Sigma , \Delta , F )$ , where $\left( Q , \Sigma , \Delta \right)$ is a NBTTS over Σ and $F \subseteq Q$ is a set of final states. If for each $a / k \in \Sigma$ we have that $\Delta ( a / k )$ is such that for all $q _ { 1 } , \dotsc , q _ { k } \in Q$ , there exists a unique $q \in Q$ such that $( q _ { 1 } , \dots , q _ { k } , q ) \in \Delta ( a )$ then the corresponding NBTTS is called a Deterministic Bottom-up Tree Transition System $o r$ DBTTS, and the NBTA is called a Deterministic Bottom-up Tree Automaton or DBTA.

▶ Definition 4 ({Run, Result, Acceptance} for NBTA). Given an NBTTS $\mathcal { T } : = ( Q , \Sigma , \Delta )$ and a tree $t \in T r e e s \left( \Sigma \right)$ , a run of T on t is defined as a tree $\rho : P o s i t i o n s ( t )  Q$ which is ‘compatible’ with ∆, i.e., for every position u ∈ Positions(t), if $t ( u ) = a / k , \rho ( u ) = q$ and $\rho ( u i ) : = q _ { i }$ for all $i \in [ k ]$ , then, $( q _ { 1 } , \dots , q _ { k } , q ) \ \in \ \Delta ( a )$ . The result of a run $\rho$ is resul $t ( \rho ) : = \rho ( \varepsilon )$ . Given an NBTA $\ A : = \ ( Q , \Sigma , \Delta , F )$ , and a tree t ∈ Trees (Σ), we say that t is accepted by A if and only if there exists a run ρ of $\left( Q , \Sigma , \Delta \right)$ on t such that resul $ , t ( \rho ) \in F$ . The set of all trees t accepted by a NBTA A is called the language recognised $b y \ A ,$ , and is denoted by $L ( \mathcal { A } ) \subseteq T r e e s \left( \Sigma \right)$

Just like a word w can be extended by appending another word to its end, a tree can be extended by appending other trees to designated leaves. This is formalised below, where ⊎ denotes disjoint union.

▶ Definition 5 (Context). A context with arity n or an n-context over $\Sigma$ is a tree over the alphabet Σ ⊎ $\{ \bigstar _ { 1 } / 0 , \ldots , \bigstar _ { n } / 0 \}$ such that each letter □<sub>i</sub> appears exactly once in the tree, for all $i \in [ n ]$ . We refer to the positions occupied $b y \square _ { i }$ as the holes of the context for all $i \in [ n ]$

In this paper, we shall only deal with finite contexts. Intuitively, the leaves labeled with $\Pi _ { 1 } , \ldots , \Pi _ { n }$ , or the holes of the context are the designated positions where other trees can be appended. We denote the set of all finite n-contexts over Σ by $\mathsf { C o n t e x t s } _ { n } \left( \Sigma \right)$ . We denote $\textstyle \bigcup _ { n = 0 } ^ { \infty }$ Context ${ \mathfrak { s } } _ { n } \left( \Sigma \right)$ by Contexts (Σ), where we let Context $\mathbf { s } _ { 0 } \left( \Sigma \right) : = \mathrm { T r e e s } \left( \Sigma \right)$

## 3.2 Tree Transformations

In this section, we define our tree transformation model and describe some associated problems that we wish to study. We start by defining and describing tree homomorphisms, which are essentially extensions of homomorphisms from words to trees. We more or less follow the definition as presented in [4], with minor modifications for ease of writing and mathematical rigour.

▶ Definition 6 (Tree Homomorphism). Let Σ, Γ be two finite ranked alphabets, possibly not disjoint. Let $X : = \{ x _ { i } / 0 \mid i \in \mathbb { N } \}$ be a countable set of variables of arity zero that is pairwise disjoint from Σ, Γ. For each $n \in  { \mathbb { N } } _ { 0 }$ , define the section of X by n by $X _ { n } : = \{ x _ { i } / 0 \mid i \in [ n ] \}$ Note that with this definition, we also get that $X _ { 0 } = \varnothing$

Let $h : \Sigma \to T r e e s \left( \Gamma \uplus X \right)$ be a mapping such that $h ( f ) \in T r e e s \left( \Gamma \psi X _ { a r i t y ( f ) } \right)$ for all $f \in \Sigma$ . Then, we define the tree homomorphism $\widetilde { h } : T r e e s ( \Sigma )  T r e e s ( \Gamma )$ determined by h as follows:

$= { \overset { \sim } { h } } ( a ) : = h ( a ) \in$ Trees (Γ) for each $a \in \Sigma$ with arity $\it 0 .$

$\begin{array} { r l } { = } & { { } \tilde { h } ( f ( t _ { 1 } , \dots , t _ { n } ) ) : = h ( f ) \{ x _ { i } \gets \tilde { h } ( t _ { i } ) \mid i \in [ n ] \} } \end{array}$ for each $f / n \in \Sigma$ , where the right-hand side is the result of replacing all occurences of the symbol $x _ { i }$ in $h ( f )$ by the term $\widetilde { h } ( t _ { i } )$ for all $i \in [ n ]$

By abuse of notation, whenever there is no ambiguity, we shall drop the tilde and just refer to the tree homomorphism $\widetilde { h }$ determined by h just as simply h.

We let Σ be a finite ranked alphabet. We will need trees, some parts of which are not completely specified and we use variables to denote those parts. Let Var be a countable set of ranked variables or variables. For a variable $x \in \mathtt { V a r } .$ , ari $\mathtt { t y } ( x ) \in \mathbb { N } _ { 0 }$ denotes its arity. We define pairwise disjoint sets $\mathsf { V a r } _ { k } : = \{ x \in \mathsf { V a r } | \mathsf { a r i t y } ( x ) = k \} \subseteq \mathsf { V a r }$ for all $k \in \mathbb N$ . Further we also define the degree of a variable x as $\deg ( x ) : = 1 + \arcsin ( x )$

▶ Definition 7 (Pattern). A tree pattern or simply a pattern over $( \Sigma , V a r )$ is defined to be a tree ${ \mathsf { P a t } } \in T r e e s \left( \Sigma \not \uplus V a r \right)$ . We also let Var(Pat) be the set of all variables appearing in Pat. Note that $V a r ( \mathsf { P a t } ) \subseteq V a r$ is necessarily finite. We also define the degree of a pattern Pat denoted by $\begin{array} { r } { \deg ( \mathsf { P a t } ) : = \operatorname* { m a x } _ { x \in V a r ( \mathsf { P a t } ) } \deg ( x ) } \end{array}$

A pattern can represent multiple trees by substituting variables in the pattern by contexts of the same arity.

▶ Definition 8 (Substitution). A substitution over Σ is a partial function $\sigma : \ : V a r  \ :$ Contexts (Σ) such that for all $x \in D o m a i n ( \sigma )$ , we have that $\sigma ( x ) \in C o n t e x t s _ { k } \left( \Sigma \right)$ , where $k = a r i t y ( x )$

Given a substitution σ, we can naturally extend it to the set of patterns over Σ as follows. Define a tree homomorphism $\widetilde { \sigma }$ : Trees $( \Sigma \uplus \mathsf { D o m a i n } ( \sigma ) ) \to \mathtt { T r e e s } ( \Sigma )$ by setting

$$
{ \widetilde { \sigma } } ( x ) : = { \left\{ \begin{array} { l l } { x } & { { \mathrm { ~ i f ~ } } x \in \Sigma ; } \\ { \sigma ( x ) } & { { \mathrm { ~ i f ~ } } x \in \mathsf { D o m a i n } ( \sigma ) } \end{array} \right. }
$$

and extending it homomorphically. Since $\widetilde { \sigma } | _ { \mathsf { D o m a i n } ( \sigma ) } = \sigma$ , by abuse of notation we shall write $\widetilde { \sigma }$ as σ in places where the distinction is irrelevant.

We would like to control the set of contexts that can be substituted for variables in patterns, for which we use guards. A guard over a finite alphabet Σ is a partial function $\mathcal { G } : \mathsf { V a r } \to 2 ^ { \mathtt { T r e e s } ( \Sigma ) }$ , which assigns to each variable in its domain, a tree language over Σ. The following formally defines when a pattern matches a tree.

▶ Definition 9 (Match). Let Pat be a pattern over (Σ, Var) and $t \in T r e e s \left( \Sigma \right)$ be a tree. Let G be a guard such that Var $\mathsf { \tilde { \Gamma } } ( \mathsf { P a t } ) \subseteq D o m a i n ( \mathcal { G } )$ . We say that t matches pattern Pat under G if and only if there exists a substitution σ such that

$$
\begin{array} { r l } { = } & { { } V a r ( \mathsf { P a t } ) \subseteq D o m a i n ( \sigma ) \subseteq D o m a i n ( \mathcal { G } ) . } \end{array}
$$

$\mathbf { \Psi } = \sigma ( x ) \in \mathcal { G } ( x )$ for each $x \in V a r ( \mathsf { P a t } )$

$$
\mathbf { \tau } = \mathbf { \nabla } \sigma ( \mathsf { P a t } ) = t .
$$

In such a case, σ is called a matching substitution, and we say the substitution σ matches the tree t to the pattern Pat under the guard G.

There may exist more than one matching substitution for a given match. Eg., if $a / 1 , b / 1 , c / 0$ are letters and $x / 1 , y / 0$ are variables, the tree $a - b - b - c$ matches the pattern $x - b - y$ with 2 distinct substitutions $\sigma _ { 1 } , \sigma _ { 2 }$ given by $\sigma _ { 1 } ( x ) : = a , \sigma _ { 1 } ( y ) : = b - c$ whereas $\sigma _ { 2 } ( x ) : = a - b , \sigma _ { 2 } ( y ) : = c$

The following formalises the tree transformations defined by our model.

▶ Definition 10 (Atomic Transformation). Let Σ be a finite alphabet and Var be a countable set of variables. An atomic tree transformation or simply, an atomic transformation is an ordered triple $T : = ( \mathsf { P a t } _ { s r c } , \mathsf { P a t } _ { t g t } , \mathcal { G } )$ , where $\mathsf { P a t } _ { s r c } , \mathsf { P a t } _ { t g t }$ are patterns over $( \Sigma , V a r )$ called

# Type-Checking for Pattern-Based Tree Transformations

the source pattern and target pattern respectively, and G is a guard over Σ such that we have $V a r ( T ) : = V a r ( \mathsf { P a t } _ { s r c } ) \cup V a r ( \mathsf { P a t } _ { t g t } ) \subseteq D o m a i n ( \mathcal { G } )$ . The relation over trees induced by T is

$$
\begin{array} { r c l } { R e l ( T ) } & { : = } & { \{ ( t , t ^ { \prime } ) \mid \exists \sigma \ s u c h \ t h a t \ \sigma \ m a t c h e s \ t \ t o \ \ \mathsf { P a t } _ { s r c } \ u n d e r \ \mathcal { G } } \end{array}
$$

$$
a n d \sigma \ m a t c h e s \ t ^ { \prime } \ t o \ P a \mathrm { t } _ { t g t } \ u n d e r \ { \mathcal G } \}
$$

We denote by $T ( t ) = \{ t ^ { \prime } \mid ( t , t ^ { \prime } ) \in R e l ( T ) \} \subseteq { \mathrm { T r e e s } } ( \Sigma )$ the efect of T on the tree t. We naturally extend the efect of $T$ to tree languages L by setting $T ( L ) : = \textstyle \bigcup _ { t \in L } T ( t )$ . We also define $\deg ( T ) : = \operatorname* { m a x } _ { x \in \mathsf { V a r } ( T ) }$ deg(x). Our model consists of a possibly infinite set of atomic transformations, represented by a bimorphism.

▶ Definition 11 (Transformation). Let Σ, Γ be two finite alphabets. Let Var be a countably infinite set of variables. A transformation T is a tuple $\mathbf { T } : = ( S , \phi _ { s r c } , \phi _ { t g t } , \mathcal { G } )$ where

$= \ S \subseteq T r e e s \left( \Gamma \right)$ is called the seed language.

ϕ<sub>src</sub>, ϕ<sub>tgt</sub> : Trees (Γ) → Trees (Σ ⊎ Var) are tree homomorphisms.

$\circeq \mathcal G$ is a guard over Σ.

We define the relation induced by a transformation $\textbf { T } : = ( S , \phi _ { s r c } , \phi _ { t g t } , \mathcal { G } )$ as follows: for each seed tree $s \in S$ , we define the atomic transformation $T _ { s } : = ( \phi _ { s r c } ( s ) , \phi _ { t g t } ( s ) , \mathcal { G } )$ . Then $\begin{array} { r } { R e l ( \mathbf { T } ) : = \bigcup _ { s \in S } R e l ( T _ { s } ) } \end{array}$

For a tree t and a tree language $L , \mathbf { T } ( t ) = \{ t ^ { \prime } \mid ( t , t ^ { \prime } ) \in R e l ( \mathbf { T } ) \} \subseteq \operatorname { T r e e s } \left( \Sigma \right)$ and $\mathbf { T } ( L ) : =$ $\textstyle \bigcup _ { t \in L } \mathbf { T } ( t )$ as before. We further define $\begin{array} { r } { \mathsf { V a r } ( \mathbf { T } ) = \bigcup _ { \gamma \in \Gamma } \mathsf { V a r } ( \phi _ { s r c } ( \gamma ) ) \cup \mathsf { V a r } ( \phi _ { t g t } ( \gamma ) ) } \end{array}$ and $\begin{array} { r } { \deg ( \mathbf { T } ) = \operatorname* { m a x } _ { x \in \mathsf { V a r } ( \mathbf { T } ) } \deg ( x ) } \end{array}$

## 4 Example - Code Snippet Translation

As a motivating example, we consider a piece of code, written in a language like $\mathrm { C } + + \mathrm { o r }$ Java, containing a certain bounded number of for loops. We would like to transform this into a corresponding piece of code with while loops. A concrete example is depicted in Figure 3.

Let us bound the number of for loops in our programs by some fixed $n \in \mathbb { N }$ . Consider the alphabet $\Sigma : = \Sigma _ { \mathrm { c o r e } } \uplus \{ \mathsf { f o r } / 5 , \mathsf { w h i l e } / 3 , \mathsf { n u l l } / 0 \}$ where the unary $\Sigma _ { \mathrm { c o r e } }$ refers to the alphabet used for programming, which for instance, could be all the characters appearing on a standard keyboard. With this alphabet, the source and target codes are seen as trees, as depicted in Figure 4. Long chains of characters from $\Sigma _ { \mathrm { c o r e } }$ are written as mere sequences wrapped in a box for the sake of readability and space.

Target code   
int[] A = new int[m];   
Source code int i=0;   
int[] A = new int[m]; while (i<A.length) {   
for(int i=0;i<A.length;i++) { A[i] = getInputFromUser();   
A[i] = getInputFromUser(); i++;   
} }   
for(int j=0;j<A.length;j++) { int j=0;   
int minIndex = j; while (j<A.length) {   
int minValue = A[minIndex]; int minIndex = j;   
for(int k=j;k<A.length;k++) { int minValue = A[minIndex];   
(A[ ] ) { int k=j;   
minIndex = k; while(k<A.length) {   
minValue = A[k]; if (A[k] < minValue) {   
} minIndex = k;   
} minValue = A[k];   
int temp = A[j]; }   
int A[j] = A[k]; k++;   
int A[k] = temp; }   
} int temp = A[j];   
return A; int A[j] = A[k];   
int A[k] = temp;   
j++;   
}   
return A;

Figure 3 The source and target code snippets for Selection Sort, before and after transforming for loops to while loops.

![](images/026302f88e1691aa487f727f40fd5731129e6755f4ef5c5ee13c90c444b0527b.jpg)  
Figure 4 Source and target code snippets from Fig 3 as trees.

We would like to capture these codes with patterns containing variables, which will allow us to move around the sections of code as needed. Let $\mathsf { V a r } : = \left\{ x _ { i } ^ { \mathrm { i n i t } } / 1 , x _ { i } ^ { \mathrm { c o n d } } / 1 , x _ { i } ^ { \mathrm { i n c r } } / 1 \mid i \in [ n ] \right\}$ be the set of variables. Recall that n, bound on the number of for loops in our programs, is a constant and hence we have only a fixed finite number of variables.

![](images/7170dcc8a488df6736b8c206e8dd60351ff8de6850349ff64a1dc86162d5145b.jpg)  
Figure 5 The seed tree generating the source and target patterns in Figure 6.

The source and target patterns are given in Figure 6 and a seed tree generating these patterns is given in Figure 5. Note that the seed alphabet is given by $\Gamma : = \Sigma _ { \mathrm { c o r e } }$ ⊎ $\{ | \mathsf { o o p } _ { i } / 2$ , endLoo ${ \mathsf { p } } _ { i } / 0 \mid i \in [ n ] \} \not \uplus \{ { \mathsf { n u l l } } / 0 \}$

In general, the seed language will contain all trees in Trees (Γ) which contain each of the letters ${ \mathsf { l o o p } } _ { i }$ and endLoop at most once and are well-bracketed. Let us take a detailed look at how the language looks like. Note that we a priori limit the scope of code we transform to instances having at most n loops, where n is a fixed natural number. We start with the various alphabets we use:

1 Tree alphabet $\Sigma : = \Sigma _ { \mathrm { c o r e } } \uplus \left\{ \mathsf { f o r } / 5 , \mathsf { w h i l e } / 3 , \mathsf { n u l l } / 0 \right\}$ , where $\Sigma _ { \mathrm { c o r e } }$ is, for example, all the symbols available on a standard keyboard, each with an arity of 1.

Pattern alphabet $\textstyle \sum \Theta \ V \mathsf { a r }$ , where Var := $\left\{ { x _ { i } ^ { \mathrm { i n i t } } / 1 , x _ { i } ^ { \mathrm { c o n d } } / 1 , x _ { i } ^ { \mathrm { i n c r } } / 1 \mid i \in [ n ] } \right\}$

Seed alphabet $\Gamma : = \Sigma _ { \mathrm { c o r e } } \not \in \{ | 0 \circ { \mathsf { p } } _ { i } / 2$ , endL ${ \mathsf { o o p } } _ { i } / 0 \mid i \in [ n ] \}$ ⊎ {null/0}.

In the seed alphabet, we have binary letters loop for all $i \in [ n ]$ . We interpret this as follows: the letter loop is a placeholder for the $i ^ { t h }$ loop in the program. The left child of the letter will be a tree representing the part of the code that is inside the loop, ending with an $\mathsf { e n d L o o p } _ { i } ,$ which is a leaf letter to signify the end of the loop. The right child will be a tree representing the part of the code that follows $a f t e r$ the loop, eventually ending with a null leaf letter. Hence in the loop language, we would want that each pair of letters $( \mathsf { l o o p } _ { i } , \mathsf { e n d L o o p } _ { i } )$ which signify loop initiation and loop termination respectively to appear at most once. This is a necessary ‘hygiene’ condition upon which we will build more structure.

![](images/0c9d8d58ac4c142cb2ff8915c9231f39cb9607b422255e4136cf298ba58ab382.jpg)

![](images/e828f155299e3f6bf6393b8dfe209d18bf8754cb101b576657268bfb9ab09272.jpg)  
Figure 6 Source and target patterns. The source and target morphisms are identity everywhere, except for loop and endLoop , which are indicated by colors blue and red respectively.

For this condition to hold, we define for each $i \in [ n ]$ , a finite-state tree automaton

$$
\begin{array} { r l } & { \mathcal { U } _ { i } : = \left( \left\{ q _ { 0 } , q _ { 1 } , q _ { 2 } , q _ { 3 } , q _ { 4 } \right\} , \Gamma , \Delta _ { i } , \left\{ q _ { 0 } , q _ { 3 } \right\} \right) \mathrm { ~ w h e r e ~ } \Delta _ { i } \mathrm { ~ i s ~ g i v e n ~ b y } } \\ & { \qquad \Delta _ { i } : = \left\{ \left( \mathrm { e n d L o o p } _ { i } , q _ { 1 } \right) , \left( \mathrm { n u l l } , q _ { 0 } \right) \right\} \cup \left\{ \left( \mathrm { e n d L o o p } _ { j } , q _ { 0 } \right) \bigm | j \in [ n ] , j \neq i \right\} } \\ & { \qquad \cup \left\{ \left( q _ { x } , q _ { y } , \log _ { i } , q _ { \operatorname* { m i n } ( 4 , x + y + 2 ) } \right) \right\} \cup \left\{ \left( q _ { x } , a , q _ { x } \right) \bigm | x \in \left\{ 0 , 1 , 2 , 3 , 4 \right\} \right\} } \\ & { \qquad \cup \left\{ \left( q _ { x } , q _ { y } , \log _ { j } , q _ { \operatorname* { m i n } ( 4 , x + y ) } \right) \bigm | j \in [ n ] , j \neq i , ( x , y ) \neq ( 1 , 1 ) \right\} } \\ & { \qquad \cup \left\{ \left( q _ { 1 } , q _ { 1 } , \log _ { j } , q _ { 4 } \right) \right\} } \end{array}
$$

We see that each of the automata $\mathcal { U } _ { i }$ accepts all trees s over Γ such that the letters loop , endLoop both appear either once each, or not at all in s. Thus having $S \subseteq [ \bigcap _ { i = 1 } ^ { n } L ( { \mathcal { U } } _ { i } ) ]$ implies that every tree s contains the pair of letters ${ \mathsf { l o o p } } _ { i } ,$ endL ${ \mathsf { . o o p } } _ { i }$ at most once.

On top of this requirement, we also need that every tree in the seed language be “well-nested” in terms of the containment of loops. For this, we define an automaton $\mathscr { A } : = ( Q , \Gamma , \Delta , F )$ as:

$$
\begin{array} { r l } & { Q : = \{ q _ { i } \mid i \in [ n ] \} \cup \{ q _ { 0 } \} } \\ & { F : = \{ q _ { 0 } \} } \\ & { \Delta : = \{ ( \mathsf { n u l l } , q _ { 0 } ) \} \cup \{ ( \mathsf { e n d L o o p } _ { i } , q _ { i } ) \mid i \in [ n ] \} \cup } \\ & { \qquad \{ ( q _ { i } , q _ { j } , \mathsf { l o o p } _ { i } , q _ { j } ) \mid i , j \in [ n ] \cup \{ 0 \} \} \{ ( q , a , q ) \mid q \in Q , a \in \Sigma _ { \mathrm { c o r e } } \} } \end{array}
$$

We demonstrate the working of this automaton on an example word we would like to have in the seed language. Below, we consider a tree s over the seed alphabet Γ for the case $n = 9$ along with a run of the automaton A shown in parenthesis, in blue. Note that the dashed edges represent the presence of any number of unary letters from $\Sigma _ { \mathrm { c o r e } } ;$ since they “bubble $\mathrm { u p } ^ { \mathrm { \prime } }$ any state, they can be represented as such for purposes of better understanding.

![](images/9ec66a5e89e5f20ecb799d89671661c56c41c7bd439dabfa8e101cadad6ea912.jpg)

Finally, we define the seed language to be $S : = L ( A ) \cap [ \bigcap _ { i = 1 } ^ { n } L ( \mathcal { U } _ { i } ) ]$ ], which is a regular language.

In general, we see that for any program with no more than n for loops, we will be able to find a seed tree, which will give us a source pattern and a target pattern through homomorphisms $\phi _ { s r c } , \phi _ { t g t }$ respectively. The source pattern will match the program code with not more than n for loops with a substitution, which when applied to the target pattern will give us the transformed version of the program code which has all for loops transformed into while loops.

We may want to verify that applying a transformation to source trees belonging to some regular language (such as syntactically correct programs that use for loops) will never result in target trees violating desired properties (such as syntactically correct programs that use while loops). This motivates the problem we consider for our model, defined next.

## 5 Decision Problems for Tree Transformations

A simple question is to check whether there is a matching.

<table><tr><td colspan="2">Matching Problem</td></tr><tr><td>Input: Question:</td><td>Pattern Pat, tree t, guard G. // The guard languages are represented by NBTA.</td></tr></table>

We will show that this problem is already NP-complete.

▶ Theorem 12 ([13]). The Matching Problem is NP-complete.

Proof. (Upper Bound) Let $( \mathsf { P a t } , t , \mathcal { G } )$ be an instance of Matching Problem, where Pat is a pattern over $( \Sigma , \mathsf { V a r } ) , t \in \mathsf { T r e e s } \left( \Sigma \right)$ and $\mathcal { G } : \mathsf { V a r } \to 2 ^ { \mathtt { T r e e s } ( \Sigma ) }$ where each value of the function is given by an NBTA. Let the size of the input be $N : = | \mathsf { P a t } | + | t | + | \mathcal { G } |$ . Given a certificate, which is a particular substitution whose size is bounded by |t|, we can verify whether $\sigma ( \mathsf { P a t } ) = t$ in $\mathcal { O } ( | t | )$ time, which is linear in the size of the input N. We can also verify that for every $x \in \mathsf { V a r } ( \mathsf { P a t } ) , \sigma ( x ) \in \mathcal { G } ( x )$ . Since there exists a polynomial-time verifier algorithm for the problem, we see that Matching Problem is in NP.

(Lower Bound) For this proof, we present a version adapted from [13]. The proof proceeds by establishing a reduction from 1-in-3 SAT, which is defined below.

<table><tr><td colspan="2">1-in-3 SAT</td></tr><tr><td>Input:</td><td>3CNF formula φ without negated variables.</td></tr><tr><td>Question:</td><td>Is there a truth assignment to the variables of φ such that every clause contains exactly one literal that is true and φ evaluates to true?</td></tr></table>

We know from [16] that 1-in-3 SAT is NP-complete. Consider the following instance $\{ J ( v _ { f ( 3 i - 2 ) } , v _ { f ( 3 i - 1 ) } , v _ { f ( 3 i ) } ) \mid i \in [ n ] \}$ of 1-in-3 SAT, where $V : = \{ v _ { 1 } , \ldots , v _ { k } \}$ is a finite set of variables and $f : [ 3 n ]  [ k ]$ . We also have J to be a function which evaluates to true if exactly one of its arguments is set to true, and false otherwise.

Let $\Sigma : = \{ a / 1 , \# / 1 , \lambda / 0 \}$ be a ranked alphabet and Var $\mathrel { \mathop : } = \{ x _ { i } / 1 \mid i \in \mathbb { N } \}$ be a set of ranked variables. Consider a pattern Pat over (Σ, Var) given by

$$
x _ { f ( 1 ) } x _ { f ( 2 ) } x _ { f ( 3 ) } \# \ldots \# x _ { f ( 3 i - 2 ) } x _ { f ( 3 i - 1 ) } x _ { f ( 3 i ) } \# \ldots \# x _ { f ( 3 n - 2 ) } x _ { f ( 3 n - 1 ) } x _ { f ( 3 n ) } \lambda .
$$

and a tree over Σ given by $\scriptstyle t : = a \# a \# \dots \# a \# \dots \# a$ where the number of a’s in the tree is n. We see that the only substitution $\sigma : V  \{ a , \varepsilon \}$ which matches t to α is the one which assigns each true variable to a and every false variable to ε. The guard languages can be specified easily by a simple NBTA. Hence we get that a solution for the original instance of 1-in-3 SAT exists if and only if a solution for the corresponding Matching Problem has a solution. This completes the reduction. ◀

Next we consider the reachability problem for atomic transformations and establish it is EXPTime-complete.

<table><tr><td colspan="2">Atomic Transformation Reachability Problem</td></tr><tr><td>Input:</td><td>Atomic transformation  $T : = ( \mathsf { P a t } _ { s r c } , \mathsf { P a t } _ { t g t } , \mathcal { G } )$  , source language  ${ \cal L } _ { s r c } ,$  target lan- guage Ltgt. // The languages are represented by NBTA.</td></tr><tr><td>Question:</td><td>Do we have that  $T ( L _ { s r c } ) \cap L _ { t g t } \neq \emptyset ?$ </td></tr></table>

<sup>▶</sup> Theorem 13. The Atomic Transformation Reachability Problem is <sup>EXPT</sup>ime-complete.

We give a proof which is a special case of the Transformation Reachability Problem. Clearly, it is easy to see that this is indeed a case of Transformation Reachability Problem when the seed language is singleton. If A is such that $L ( \mathcal { A } ) = \{ \tau \}$ , then we just define/rename patterns $\mathsf { P a t } _ { s r c } : = \phi _ { s r c } ( \tau ) , \mathsf { P a t } _ { t g t } : = \phi _ { t g t } ( \tau )$

Proof. (Upper Bound) Let $T : = ( \mathsf { P a t } _ { s r c } , \mathsf { P a t } _ { t g t } , \mathcal { G } )$ be an atomic action, and let $\boldsymbol { B } _ { s r c } , \boldsymbol { B } _ { t g t }$ be NBTAs with $L _ { s r c } : = L ( B _ { s r c } )$ and $L _ { t g t } : = L ( B _ { t g t } )$ . To determine whether $T [ L _ { s r c } ] \cap L _ { t g t } \neq \emptyset$ we need to find a substitution σ such that $\sigma ( \mathsf { P a t } _ { s r c } ) \in L _ { s r c }$ and $\sigma ( \mathsf { P a t } _ { t g t } ) \in L _ { t g t }$ under G. We further simplify the problem at hand. Consider a bigger alphabet $\Gamma : = \Sigma \not \Cup \{ \# / 2 \}$ . Let Pat be a pattern over Γ and Var given by $\mathsf { P a t } : = \# ( \mathsf { P a t } _ { s r c } , \mathsf { P a t } _ { t g t } )$ . Let B be an NBTA over Γ which recognises all trees of the form $\# ( p _ { s r c } , p _ { t g t } )$ with $p _ { s r c } \in L _ { s r c } , p _ { t g t } \in L _ { t g t }$ . Then Atomic Transformation Reachability Problem now reduces to finding a substitution σ over Γ conforming to G such that $\sigma ( \mathsf { P a t } ) \in L : = L ( B )$

Let $M : = \operatorname* { m a x } \{ \operatorname { a r i t y } ( x _ { i } ) \mid x _ { i } \in \operatorname { V a r } ( \operatorname { P a t } ) \}$ . Let us assume that each variable $x / k \in$ $\mathtt { V a r } ( \mathsf { P a t } )$ appears $b ( x )$ number of times in Pat. By Theorem 20, for each variable x, we choose a pair $( R ( x ) , c ( x ) )$ such that $R ( x ) \subseteq Q ^ { k + 1 }$ is a relation realised by context $c ( x ) \in \tt C o n t e x t s \exp _ { k } \left( \Gamma \right)$ on B under $\mathcal G ( x )$ with the added condition that $| R ( x ) | \leq b ( x )$ . Once we have all such ordered pairs for individual variables, we simply use the substitution $\sigma : = c$ and check whether the resulting tree belongs to T. This can be done in time given by

$$
\mathcal { O } ( | \mathcal { B } | \cdot \operatorname* { m a x } _ { x } | \mathcal { G } ( x ) | + | \mathsf { P a t } | \cdot 2 ^ { | \mathcal { B } | \cdot \operatorname* { m a x } _ { x } | \mathcal { G } ( x ) | } )
$$

The number of times we have to perform this check operation is bounded above by the value $\begin{array} { r } { \left( \operatorname* { m a x } _ { x } \Big | \binom { Q ^ { \mathrm { d e g } ( x ) } } { b ( x ) } \Big | \right) ^ { \big | \mathrm { V a r } \left( \mathsf { P a t } \right) \big | } } \end{array}$ . Hence the total time required for this is given by

$$
\left( \operatorname* { m a x } _ { x } \left| \left( { Q } _ { b ( x ) } ^ { \mathrm { d e g } ( x ) } \right) \right| \right) ^ { | \mathrm { V a r } ( \mathsf { P a t } ) | } \cdot { \mathcal { O } } ( | \mathcal { B } | \cdot \operatorname* { m a x } _ { x } | { \mathcal { G } } ( x ) | + | \mathsf { P a t } | \cdot 2 ^ { | \mathcal { B } | \cdot \operatorname* { m a x } _ { x } | { \mathcal { G } } ( x ) | } )
$$

which we can see is exponential in the size of the input.

(Lower Bound) We show a reduction from Intersection Nonemptiness of DBTA to Atomic Transformation Reachability Problem. Since we know that Intersection Nonemptiness of DBTA is <sup>EXPT</sup>ime-hard, we will conclude that Atomic Transformation Reachability Problem is also <sup>EXPT</sup>ime-hard. Let $\mathcal { A } _ { 1 } , \ldots , \mathcal { A } _ { n }$ be an instance of the Intersection Nonemptiness of DBTA, where each $\mathscr { A } _ { i } : = \left( Q _ { i } , \Sigma , \Delta _ { i } , F _ { i } \right)$ is DBTA over the alphabet Σ for all $i \in [ n ]$ . We construct a corresponding instance of Atomic Transformation Reachability Problem as follows. Let $\Gamma : = \Sigma \not \Psi \left\{ \# / n , \flat / 0 \right\}$ be a larger alphabet. Consider the atomic action $T : = ( \# ( \underline { { x } } , x , \dots , x ) , \flat , \mathcal { G } )$ with $\mathcal { G } ( x ) : = \mathrm { T r e e s } \left( \Sigma \right)$ . Let $\boldsymbol { B } _ { s r c }$ be a NBTA n such that $L _ { s r c } : = L ( \boldsymbol { \mathcal { B } } _ { s r c } ) : = \{ \# ( t _ { 1 } , t _ { 2 } , \dots , t _ { n } ) \ | \ t _ { i } \in$ Trees (Σ) , $\forall i \in [ n ] \}$ . Let $B _ { t g t }$ be an NBTA such that $L _ { t g t } : = L ( B _ { t g t } ) : = \{ \boldsymbol { \flat } \}$

We can see that $T [ L _ { s r c } ] \cap L _ { t g t } \neq \emptyset$ if and only if there exists $t \in \mathrm { T r e e s } \left( \Sigma \right)$ such that $t \in L ( \mathcal { A } _ { i } )$ for all $i \in [ n ]$ . This completes the reduction, and thus the proof. ◀

Our central problem of study is whether a tree in a regular target language can be reached by a given transformation, starting from a tree in the regular source language, which we show is decidable.

<table><tr><td colspan="2">Transformation Reachability Problem</td></tr><tr><td>Input:</td><td>Transformation  $\mathbf { T } : = ( S , \phi _ { s r c } , \phi _ { t g t } , \mathcal { G } )$  , source language  ${ \cal L } _ { s r c } ,$  target language  $L _ { t g t } .$  // The languages are represented by NBTA.</td></tr><tr><td>Question:</td><td>Do we have that  $\mathbf { T } ( L _ { s r c } ) \cap L _ { t g t } \neq \emptyset ?$ </td></tr></table>

<sup>▶</sup> Theorem 14. The Transformation Reachability Problem is in ${ \mathcal { Q } } { - } E X P T I M E$ . More precisely, it can be solved in time $2 ^ { \mathsf { p o l y } ( | \mathsf { i n p u t } | ) }$ , where poly is a polynomial of degree deg(T).

Since NBTA are closed under complementation (with an exponential blow-up, see [4]) we get that the type-checking problem defined below is in 3-EXPTime.

<table><tr><td colspan="2">Type-Checking Problem</td></tr><tr><td>Input:</td><td>Transformation  $\mathbf { T } : = ( S , \phi _ { s r c } , \phi _ { t g t } , \mathcal { G } )$  , source language  ${ \cal L } _ { s r c } ,$  target language  $L _ { t g t } .$  // The languages are represented by NBTA.</td></tr><tr><td>Question:</td><td>Do we have that  $\mathbf { T } ( L _ { s r c } ) \subseteq L _ { t g t } ?$ </td></tr></table>

## ▶ Corollary 15. The Type-Checking Problem is in 3-EXPTime.

Our decision procedure for the Transformation Reachability Problem is automatatheoretic. We first give the main ideas behind our decision procedures in the next section.

## 6 Core Idea Behind the Automata Construction

![](images/97a3c39941bb4bcbacf883bc8bd5cd4f93caeb8b32d09032c3955292945519ec.jpg)  
Figure 7 The source tree is annotated with states of $\boldsymbol { \mathcal { A } } _ { L _ { s r c } } .$ . We want to simulate the efect of running $\mathcal { A } _ { L _ { s r c } }$ on the source tree in the seed tree itself. This means, when processing η, it should guess and validate the potential transformation of the tuple $( q _ { 7 } , q _ { 8 } , q _ { 9 } )$ to $q _ { 1 0 }$ by a context (colored blue) in the source tree that matches $\phi _ { s r c } ( \eta )$ , and simultaneously make sure that there is a substitution for x that preserves the transformations $\left( q _ { 1 } , q _ { 2 } \right)$ to $q _ { 3 }$ as well as $( q _ { 4 } , q _ { 5 } )$ to $q _ { 6 }$ . An alternating tree automaton on the seed tree can achieve this.

To solve the Transformation Reachability Problem, we need to check for the existence of a seed tree s giving rise to an atomic transformation $T _ { s }$ , a source tree $t _ { s r c } ,$ a target tree $t _ { t g t }$ and a substitution σ satisfying two constraints. First, the source tree $t _ { s r c }$ should be in $L _ { s r c }$ and second, the target tree $t _ { t g t }$ should be in $L _ { t g t }$ . The source tree is obtained from the seed tree by first applying the homomorphism $\phi _ { s r c }$ and then applying the substitution $\sigma .$ Suppose a node η in the seed tree is transformed into a context as shown in the middle of Fig. 7. Suppose a variable x is substituted with the tree $\sigma ( x )$ as shown in the right side of Fig. 7. Consider a run of $\mathcal { A } _ { L _ { s r c } }$ on the annotated source tree, in particular the state transformation $\left( \left( q _ { 7 } , q _ { 8 } , q _ { 9 } \right) \right.$ to $\left( q _ { 1 0 } \right) )$ when the run parses the portion of the source tree that comes from the node η of the seed tree. The idea of our decision procedure is to design a tree automaton (we will call this the far-sighted automaton in the next two paragraphs) that parses the seed tree, and when it traverses the node $\eta ,$ , it simulates $A _ { L _ { s r c } } \mathrm { ' s }$ state transformation $( q _ { 7 } , q _ { 8 } , q _ { 9 } )$ to $\left( q _ { 1 0 } \right)$ . To determine that $( q _ { 7 } , q _ { 8 } , q _ { 9 } )$ is transformed to $q _ { 1 0 }$ , the far-sighted automaton needs to know that $\left( q _ { 1 } , q _ { 2 } \right)$ is transformed to $q _ { 3 }$ by $\sigma ( x )$

The far-sighted automaton doesn’t know what is $\sigma$ (in fact, our goal is to check if a suitable σ exists). There might be infinitely many substitutions σ out of which one might work; but our far-sighted automaton is supposed to be a finite state automaton, incapable of picking one choice from infinitely many. To work around this, we observe that the actual tree $\sigma ( x )$ is not important; the important thing is that it transforms $\left( q _ { 1 } , q _ { 2 } \right)$ to $\left( q _ { 3 } \right)$ . Any other $\sigma ^ { \prime } ( x )$ which does the same transformation will work equally well in place of $\sigma ( x )$ The number of such transformations is finite (since they are transformations on a finite set of states) and the far-sighted automaton only needs to check if one of these finitely many transformations work.

One thing we ignored in the above explanation is that the variable x may occur multiple times in the source pattern. In the run of $\mathcal { A } _ { L _ { s r c } }$ on the source tree, one occurrence of $\sigma ( x )$ may encounter $\left( q _ { 1 } , q _ { 2 } \right)$ (which is transformed to $q _ { 3 } )$ . Some other occurrence of $\sigma ( x )$ may encounter some other pair $( q _ { 4 } , q _ { 5 } )$ (which is transformed to some other state, say q<sub>6</sub>). What we need is a substitution that transforms $\left( q _ { 1 } , q _ { 2 } \right)$ to $q _ { 3 }$ and also transforms $( q _ { 4 } , q _ { 5 } )$ to q<sub>6</sub> (and does other transformations too, if there are more occurrences of x in the source pattern). To capture all these transformations simultaneously, we introduce the concept of relational lifts that is formally defined in the next section. The far-sighted automaton nondeterministically picks out one such possible set of transformations to check that the source tree is in $L _ { s r c }$ . It similarly needs to check that the target tree is in $L _ { t g t }$ . Alternating tree automata ofer a convenient way to achieve all these tasks of the far-sighted automaton. The proof idea is to design an alternating tree automaton whose nonemptiness is equivalent to the existence of a seed tree that satisfies the two required constraints.

## 7 Alternating Tree Automata and Relational Lifts

We introduce concepts that will be essential for the final proof of the theorem.

▶ Definition 16 (Alternating Top-down Tree Automata). An Alternating Top-down Tree Automaton or ATTA over the alphabet Σ is a tuple $\mathscr { A } : = ( Q , \Sigma , \Delta , I )$ , where Q is a finite set of states, $I \subseteq Q$ is a set of initial states, and $\Delta : Q \times \Sigma \to B ^ { + } ( Q \times \mathbb { N } )$ such that $\Delta ( q , a / k ) \in { \mathcal { B } } ^ { + } ( Q \times [ k ] )$ , for all $q \in Q , a / k \in \Sigma$ . Here, for any set $X , B ^ { + } ( X )$ denotes the set of all positive boolean combinations of elements in $X$

The semantics of ATTA are defined using runs, which are themselves trees satisfying some conditions.

▶ Definition 17 (Run, Acceptance for ATTA). Given a tree $t \in$ Trees (Σ) and an ATTA A over $\Sigma ,$ , we define a run of A on t to be a tree $\rho$ over $Q \times \mathbb { N } ^ { * }$ such that $\rho ( \varepsilon ) = ( q , \varepsilon ) ~ f o r$ some state q and every position $u \in P o s i t i o n s ( \rho )$ satisfies the following condition: if $\rho ( u ) =$ $( q , x ) , t ( x ) = a / k , \Delta ( q , a ) = \phi .$ , then there is a subset $S : = \{ ( q _ { 1 } , i _ { 1 } ) , \dots ( q _ { n } , i _ { n } ) \} \subseteq Q \times [ k ]$ such that $S \models \phi$ , the successor positions of u in $\rho$ are $\{ u 1 , \ldots , u n \}$ , and $\rho ( u j ) = ( q _ { j } , x i _ { j } )$ for all $j \in [ n ]$

A run is successful if $\rho ( \varepsilon ) = ( q , \varepsilon )$ for some initial state $q \in I .$ . A tree t is accepted by ATTA A if there exists at least one successful run of A on t. The set of all trees accepted by ATTA A is called the tree language recognised by A, and denoted by $L ( \mathcal { A } ) \subseteq T r e e s \left( \Sigma \right)$ . A tree language $L \subseteq T r e e s \left( \Sigma \right)$ is called regular if there exists an ATTA A such that $L = L ( \mathcal { A } )$ The set of all regular tree languages over Σ is denoted by $R e g u l a r ( \Sigma ) \subseteq 2 ^ { T r e e s ( \Sigma ) }$

Suppose A is a word automaton with set of states Q and transition relation given by $\Delta : \Sigma $ Relations(Q, 2), we can extend it to $\tilde { \Delta } : \Sigma ^ { * } \to$ Relations(Q, 2) such that $( q _ { 1 } , q _ { 2 } ) \in \widetilde \Delta ( w )$ if A has a run on the word w starting from $q _ { 1 }$ and ending at $q _ { 2 }$ . The following defines a similar extension for NBTTS.

▶ Definition 18 (Extended Transition Relation(ETR)). Let $\mathcal { T } : = ( Q , \Sigma , \Delta )$ be a NBTTS. We extend the transition function $\Delta$ to a larger function ∆e : Contexts $( \Sigma )  R e l a t i o n s ( Q )$ called the Extended Transition Relation or ETR as follows. Let $c \in$ Context $s _ { k } \left( \Sigma \right)$ . For a tuple $\overline { { \pmb { q } } } : = \left( q _ { 1 } , q _ { 2 } , \ldots , q _ { k } \right)$ of states, define the NBTTS $\mathcal { T } _ { \overline { { q } } } : = ( Q , \Sigma \uplus \{ \Pi _ { 1 } / 0 , \Pi _ { 2 } / 0 , \dots , \Pi _ { k } / 0 \} , \Delta$ ⊎ $\{ ( \prod _ { i } , \{ q _ { i } \} ) \big | i \in [ k ] \} )$ . Let us define the set of results to be the set given by Resul $t s ( \overline { { { \bf q } } } ) : =$ $\{ ( \overline { { { \pmb q } } } , q ) \in Q ^ { k + 1 } \ | \ q$ is the result of a run of $\mathcal { T } _ { \overline { { \boldsymbol { q } } } }$ on $c . \}$ . Finally, we also define the set $\Delta ( c ) : =$ $\bigcup _ { \overline { { q } } \in Q ^ { k } }$ Results(q). We also refer to the relation $\widetilde { \Delta } ( c )$ as the relation induced by the context c on the states of the NBTTS T. Note that $\widetilde { \Delta } ( a ) = \Delta ( a ) , \forall a \in \Sigma$

Suppose a word $w _ { 1 }$ is extended by appending another word w<sub>2</sub> to its end. Now, $\tilde { \Delta } ( w _ { 1 } \cdot w _ { 2 } )$ is obtained by simply composing the relations $\tilde { \Delta } ( w _ { 1 } )$ and $\tilde { \Delta } ( w _ { 2 } )$ . For trees, it is more complicated: a tree $t _ { 1 }$ can have multiple “end points” where it can be extended by appending other trees. We will need a more complicated way of composing relations.

▶ Definition 19 (Relational Lifts). Let X be a set and $n \in  { \mathbb { N } } _ { 0 }$ be a nonnegative integer. Let $o p \subseteq X ^ { n + 1 }$ be a relation. We define the relational lift of op as a new relation ${ \widetilde { o p } } \subseteq R e l a t i o n s ( X ) ^ { n + 1 }$ given as follows: For $R 1 , \dots , R _ { n } , R \in R e l a t i o n s ( X )$ , we say that $( R _ { 1 } , R _ { 2 } , \ldots , R _ { n } , R ) \in { \widetilde { o p } } \ i f$ and only if

1 R ∈ Relations $\textstyle \left( X , 1 + \sum _ { j = 1 } ^ { n } i _ { j } \right)$ , where $i _ { j } \in  { \mathbb { N } } _ { 0 }$ for all $j \in [ n ]$ are such that $R _ { j } \in$ $R e l a t i o n s ( X , i _ { j } + 1 )$

1 For all $j \in [ n ]$ , for all ${ \overline { { \pmb { x } } } } _ { j } \in \ { \cal X } ^ { i _ { j } }$ , we have $( { \overline { { \pmb { x } } } } _ { 1 } , { \overline { { \pmb { x } } } } _ { 2 } , \dots , { \overline { { \pmb { x } } } } _ { n } , y ) \ \in \ R$ if there exist $y _ { 1 } , y _ { 2 } , \dotsc , y _ { n } \in X$ such that $( { \overline { { x } } } _ { j } , y _ { j } ) \in R _ { j }$ and $( y _ { 1 } , y _ { 2 } , \dotsc , y _ { n } , y ) \in o p$

We can think about relational lift as a procedure to compose arbitrarily many relations. Another, more abstract way to think about it is as a way of $\mathrm { \ddot { \Delta } l i f t i n g \ ' }$ an operator over a set to an operator over relations over that same set. In case of automata running on words, for any $a \in \Sigma$ and any $w _ { 1 } \in \Sigma ^ { * }$ , if we set $\mathsf { o p } : = \Delta ( a )$ , then $( \widetilde { \Delta } ( w _ { 1 } ) , \widetilde { \Delta } ( w _ { 1 } \cdot a ) ) \in \widetilde { \mathsf { o p } }$ . In case of NBTTS running on contexts, suppose $c _ { 1 } , \ldots , c _ { k }$ are the children of a node labeled by the letter $a / k$ and we set $\mathsf { o p } : = \Delta ( a ) \subseteq Q ^ { k + 1 }$ , then $( \widetilde \Delta ( c _ { 1 } ) , \ldots , \widetilde \Delta ( c _ { k } ) , R ) \in \widetilde { \circ \mathbf { p } }$ implies that $R = \widetilde { \Delta } ( a ( c _ { 1 } , \ldots , c _ { k } ) )$ .

The following problem and its complexity is an important intermediate technical lemma used later.

<table><tr><td colspan="2">Relation Realisability Problem</td></tr><tr><td>Input: Question:</td><td>NBTTS  $\overline { { \boldsymbol { \mathscr { T } } : = \left( Q , \Sigma , \Delta \right) } }$  , relation R ∈ Relations(Q), NBTA D. Is R realisable by  $\tau$  under  $\mathcal { D } \mathrm { ~ i . e . }$  , Does there exist c ∈ Contexts (Σ) ∩ L(D) such that  $\widetilde { \Delta } ( c ) = R ?$ </td></tr></table>

As a matter of nomenclature, if the above condition holds for a given $\tau , R , { \mathcal { D } } ,$ , then we say that the relation R is realisable by $\tau$ under D.

## <sup>▶</sup> Theorem 20. The Relation Realisability Problem is <sup>EXPT</sup>ime-complete.

Proof. (Lower Bound) We show a reduction from Intersection Nonemptiness of DBTA to Relation Realisability Problem. We recall the problem Intersection Nonemptiness of DBTA as:

<table><tr><td colspan="2">Intersection Nonemptiness of DBTA</td></tr><tr><td>Input: DBTAs</td><td> $\mathcal { A } _ { 1 } , \mathcal { A } _ { 2 } , \ldots , \mathcal { A } _ { n }$ </td></tr><tr><td>Question:</td><td>Do we have that  $\textstyle \bigcap _ { i = 1 } ^ { n } L ( { \mathcal { A } } _ { i } ) \neq \varnothing ?$ </td></tr></table>

We know from [4] that Intersection Nonemptiness of DBTA is EXPTime-complete. Let Σ be a given finite alphabet, and let $\mathcal { A } _ { 1 } , \ldots , \mathcal { A } _ { n }$ be an instance of Intersection Nonemptiness of DBTA, where $\mathscr { A } _ { i } : = ( Q _ { i } , \Sigma , \Delta _ { i } , F _ { i } )$ are DBTAs. If $F _ { i } = \emptyset$ for some $i \in [ n ]$ we must have $L ( \mathcal { A } _ { i } ) = \emptyset$ and hence the intersection $\textstyle \bigcap _ { i = 1 } ^ { n } L ( { \mathcal { A } } _ { i } ) = \varnothing$ . Hence we assume that $F _ { i } \neq \varnothing$ for all $i \in [ n ]$ Consider NBTA $\mathcal { B } : = ( ( \left. \mathscr { + } \right. Q _ { i } ) \cup \{ \top \} , \Gamma : = \Sigma \uplus \{ \# / 2 \} , \Delta , \{ \top \} )$ , with $\Delta : \Sigma \uplus \{ \# / 2 \} \to$ Relations(Q) is defined as

$$
\Delta ( a ) : = \left\{ \begin{array} { l l } { \mathsf { \Gamma } | \mathsf { \Gamma } | = 1 \Delta _ { i } ( a ) } & { \mathrm { ~ i f ~ } a \in \Sigma ; } \\ { \{ ( q , q ^ { \prime } , \top ) \mid q , q ^ { \prime } \in F _ { i } \mathrm { ~ f o r ~ s o m e ~ } i \in [ n ] \} } & { \mathrm { ~ i f ~ } a = \# } \end{array} \right.
$$

Let $R \in { \tt R e l a t i o n s } ( Q , 2 )$ be given by $R : = \{ ( f _ { i } , \top ) \mid i \in [ n ] \}$ , where each $f _ { i }$ is an arbitrarily chosen but fixed final state in $F _ { i }$ for each $i \in [ n ]$ . Also consider DBTA Shape which recognises all contexts of the form $\# ( t , \boldsymbol { \Pi } )$ where t ∈ Trees (Σ). Consider the instance of the realisability problem given by (B, R, Shape).

We can see that R is realisable by $B \cap$ Shape if and only if there exists a tree $t \in$ $L ( A _ { 1 } ) \cap \cdots \cap L ( A _ { n } )$ . Clearly, R is realisable by B ∩ Shape $\Longleftrightarrow$ there exists a context $c : = \# ( t , \bigsqcup )$ which induces the relation $R$ on the states of $\boldsymbol { B }$ under Shape $\Longleftrightarrow$ for each $i \in [ n ]$ and the labelling of the hole □ with $f _ { i }$ , the result of a run of B on t yields $\mathrm { a }$ final state $f _ { i } ^ { \prime } \in F _ { i } \iff t$ is recognised by $A _ { i }$ for each $i \in [ n ] \iff t \in \bigcap _ { i = 1 } ^ { n } L ( { \mathcal { A } } _ { i } )$ . Since we know that Intersection Nonemptiness of DBTA is EXPTime-hard, we conclude that the realisability problem is also EXPTime-hard.

(Upper Bound) Let NBTTS $\tau : = ( Q , \Sigma , \Delta )$ be a transition system, relation $R \in$ Relations $( Q , k + 1 ) \subseteq$ Relations(Q), and NBTA D be a given instance of Relation Realisability Problem. We reduce the problem of finding a context $c \in$ Contexts<sub>k</sub> (Σ) ∩ $L ( \mathcal D )$ such that ${ \widetilde \Delta } ( c ) : = R$ to the problem of NBTA nonemptiness. We will define an NBTA A with relations over $Q$ as states such that for any context $^ { c , }$ the result of a run of A on c is precisely the relation induced by $c .$ More formally, we define a NBTA A as: $\textstyle A : = ( \bigcup _ { i = 1 } ^ { k + 1 }$ Relations $( Q , i ) , \Sigma$ ⊎ $\{ \Pi _ { i } / 0 \mid i \in [ k ] \} , \Delta ^ { \prime } , \{ R \} )$ , where for all $a / p \in \Sigma$ we set $\Delta ^ { \prime } ( a ) : = { \widehat { \Delta } } ( { \overline { { a } } } )$ , the relational lift of $\Delta ( a )$ . We also set $\Delta ^ { \prime } ( \beth _ { i } ) : = Q$ for all $i \in [ k ]$

A straightforward induction on the height of a tree gives us that for a context $c \in$ Contexts<sub>k</sub> $( \Sigma ) \cap L ( \mathcal { D } )$ , we have $\widetilde { \Delta } ( c ) = R$ if and only if $c \in L ( A ) \cap L ( { \mathcal { D } } )$ . Hence to find a context which ‘realises’ the relation $R ,$ all we need to do is check $\boldsymbol { A } \cap \mathcal { D }$ for nonemptiness. Let automaton $\mathcal { D }$ have m states. We know from [4] that checking this NBTA for nonemptiness can be performed in time which is a polynomial in |Q|. Since we have that $m | Q | : =$ $m | \cup _ { i = 1 } ^ { k + 1 }$ Relations $\begin{array} { r } { ( Q , i ) | = m \sum _ { i = 1 } ^ { k + 1 } } \end{array}$ |Relations $\begin{array} { r } { ( Q , i ) | = m \sum _ { i = 1 } ^ { k + 1 } 2 ^ { | Q | ^ { i } } } \end{array}$ , we see that given $\mathcal { T } , R , \mathcal { D }$ , the problem of Relation Realisability Problem can be solved in <sup>EXPT</sup>ime. <sup>◀</sup>

In the proof for the above result, we also get that for each relation which we are checking for realisability, we can also get, in EXPTime, a context over the alphabet which realises it(if it is realisable). In such a case, we will say that the relation R is realised by the context c on the NBTTS $\tau ,$ under D. We also say that c induces R on T under D. For a given pair $( \mathcal { T } , \mathcal { D } )$ with NBTTS T and NBTA D, we define the set IndRel $( \mathcal { T } , \mathcal { D } ) : = \{ ( R , c ) \mid$ R is realised by c on T under ${ \mathcal { D } } \}$

## 8 Deciding Transformation Reachability

Let $\mathbf { T } : = ( ( S , \phi _ { s r c } , \phi _ { t g t } , \mathcal { G } ) , L _ { s r c } , L _ { t g t } )$ be a given instance of Transformation Reachability Problem, where

1 S is the seed language over Γ given by the NBTA $\mathscr { A } : = ( P , \Gamma , \Theta , E )$

$\phi _ { s r c } , \phi _ { t g t }$ : Trees (Γ) → Trees (Σ ⊎ Var) are tree homomorphisms.

G is a guard over Σ such that $\mathsf { V a r } ( \phi _ { s r c } ) \mathsf { U V a r } ( \phi _ { t g t } ) \subseteq \mathsf { D o m a i n } ( \mathcal { G } )$ , where we define(by abuse of notation) $\begin{array} { r } { \mathsf { V a r } ( \phi _ { s r c } ) : = \bigcup _ { \gamma \in \Gamma } \mathsf { V a r } ( \phi _ { s r c } ( \gamma ) ) } \end{array}$ and also $\begin{array} { r } { \mathsf { V a r } ( \phi _ { t g t } ) : = \bigcup _ { \gamma \in \Gamma } \mathsf { V a r } ( \phi _ { t g t } ( \gamma ) ) } \end{array}$   
$L _ { s r c } , L _ { t g t }$ are the source and target languages respectively, which are regular tree languages given by the NBTAs $\boldsymbol { { B } } _ { i } : = ( Q _ { i } , \Sigma , \Delta _ { i } , F _ { i } )$ for $i \in \{ s r c , t g t \}$

We see that solving the Transformation Reachability Problem is equivalent to finding whether there exists $\tau \in S$ and a substitution σ such that $\sigma ( \phi _ { i } ( \tau ) ) \in L _ { i }$ for $i \in \{ s r c , t g t \}$ Let $J \subseteq S$ be given by $J : = \{ \tau \in S \mid \exists o$ such that $\sigma ( \phi _ { i } ( \tau ) ) \in L _ { i } , i \in \{ s r c , t g t \} \}$ . We will construct a number of ATTAs such that the union of the languages they recognise will be equal to J. This will imply that solving Transformation Reachability Problem will be a matter of checking each of the correspondingly constructed ATTAs for nonemptiness.

For each $x \in \mathsf { V a r } ( \phi _ { s r c } ) \setminus \mathsf { V a r } ( \phi _ { t g t } )$ , using Theorem 20 we compute a priori, the set of relations IndRel $( B _ { s r c } , \mathcal { G } ( x ) )$ of all the relations and corresponding contexts realisable by $\boldsymbol { B _ { s r c } }$ under $\mathcal { G } ( x )$ . Similarly for each variable $y \in \mathsf { V a r } ( \phi _ { t g t } ) \setminus \mathsf { V a r } ( \phi _ { s r c } )$ , we compute IndRel $( B _ { t g t } , \mathcal { G } ( y ) )$ . For all variables $z \in \mathsf { V a r } ( \phi _ { s r c } ) \cap \mathsf { V a r } ( \phi _ { t g t } )$ , we compute the set

$$
\mathrm { I n d R e 1 } \left( \mathcal { B } _ { s r c } \cap \mathcal { B } _ { t g t } , \mathcal { G } ( z ) \right) \subseteq \mathrm { R e l a t i o n s } \left( Q _ { s r c } \times Q _ { t g t } \right) \times \mathrm { C o n t e x t s } \left( \Sigma \right) .
$$

From this set, we obtain another set IndRel $( \boldsymbol { B } _ { s r c } , \boldsymbol { B } _ { t g t } , \mathcal { G } ( z ) )$ of pairs of relations that are simultaneously realisable by $\boldsymbol { B } _ { s r c }$ and $B _ { t g t }$ along with a context $c \in L ( \mathcal G ( z ) )$ that realises both of them. The idea is that given an element $( ( \hat { q } _ { s r c } ^ { ( 1 ) } , q _ { t q t } ^ { ( 1 ) } ) , ( q _ { s r c } ^ { ( 2 ) } , q _ { t q t } ^ { ( 2 ) } ) , \dots , ( q _ { s r c } ^ { ( k + 1 ) } , q _ { t q t } ^ { ( k + 1 ) } ) ) \in R _ { 1 }$ we ‘split’ it into two components $( q _ { s r c } ^ { ( 1 ) } , q _ { s r c } ^ { ( 2 ) } , \ldots , q _ { s r c } ^ { ( k + 1 ) } )$ and $( q _ { t g t } ^ { ( 1 ) } , q _ { t g t } ^ { ( 2 ) } , \dots , q _ { t g t } ^ { ( k + 1 ) } )$ . We do this for all elements in R to get a pair of relations $R _ { s r c } , R _ { t g t }$ over $Q _ { s r c } , Q _ { t g t }$ respectively. Then, we define IndRel $( \boldsymbol { B } _ { s r c } , \boldsymbol { B } _ { t g t } , \mathcal { G } ( z ) )$ as

$$
: = \{ ( ( R _ { s r c } , R _ { t g t } ) , c ) \mid ( R , c ) \in \mathrm { I n d R e l } ( \mathcal { B } _ { s r c } \cap \mathcal { B } _ { t g t } , \mathcal { G } ( z ) ) \} .
$$

In other words, we have IndRel $( \mathcal { B } _ { s r c } , \mathcal { B } _ { t g t } , \mathcal { G } ( z ) ) : = \{ ( ( R _ { s r c } , R _ { t g t } ) , c ) ~ | ~ c \in L ( \mathcal { G } ( z ) ) , \widetilde { \Delta _ { i } } ( c ) =$ $R _ { i } , i \in \{ s r c , t g t \} \}$

We fix a pair of choice functions of realisable relations $R _ { s r c } , R _ { t g t }$ over $\mathsf { V a r } ( \phi _ { s r c } ) , \mathsf { V a r } ( \phi _ { t g t } )$ respectively such that

$$
\begin{array} { r } { R _ { s r c } ( z ) : = \left\{ \begin{array} { l l } { R \in \pi _ { 1 } \big [ \mathrm { T n d R e 1 } \left( \mathcal { B } _ { s } , \mathcal { G } ( z ) \right) \big ] } & { \mathrm { ~ i f ~ } z \in \mathrm { V a r } ( \phi _ { s r c } ) \setminus \mathrm { V a r } ( \phi _ { t g t } ) ; } \\ { R ^ { \prime } \in \pi _ { 1 } \big [ \pi _ { 1 } \big [ \mathrm { T n d R e 1 } \left( \mathcal { B } _ { s r c } , \mathcal { B } _ { t g t } , \mathcal { G } ( z ) \right) \big ] \big ] } & { \mathrm { ~ i f ~ } z \in \mathrm { V a r } ( \phi _ { s r c } ) \cap \mathrm { V a r } ( \phi _ { t g t } ) } \end{array} \right. } \end{array}
$$

$$
R _ { t g t } ( z ) : = \left\{ \begin{array} { l l } { R \in \pi _ { 1 } \big [ \mathrm { T n d R e 1 } \left( \mathcal { B } _ { t g t } , \mathcal { G } ( z ) \right) \big ] } & { \mathrm { ~ i f ~ } z \in \mathrm { V a r } \big ( \phi _ { t g t } \big ) \setminus \mathrm { V a r } \big ( \phi _ { s r c } \big ) ; } \\ { R ^ { \prime \prime } \in \pi _ { 2 } \big [ \pi _ { 1 } \big [ \mathrm { T n d R e 1 } \left( \mathcal { B } _ { s r c } , \mathcal { B } _ { t g t } , \mathcal { G } ( z ) \right) \big ] \big ] } & { \mathrm { ~ i f ~ } z \in \mathrm { V a r } \big ( \phi _ { s r c } \big ) \cap \mathrm { V a r } \big ( \phi _ { t g t } \big ) } \end{array} \right.
$$

and $( R ^ { \prime } , R ^ { \prime \prime } ) \in \pi _ { 1 }$ [IndRel $( \boldsymbol { B } _ { s r c } , \boldsymbol { B } _ { t g t } , \mathcal { G } ( z ) ) ]$ for all $z \in \mathsf { V a r } ( \phi _ { s r c } ) \cap \mathsf { V a r } ( \phi _ { t g t } )$ . Note that this also provides us with corresponding contexts which realise the chosen relations.

Armed with this, we compute the ETR s $\widetilde { \Delta _ { i } } ( \phi _ { i } ( \gamma ) )$ for all $\gamma \in \Gamma$ and $i \in \{ s r c , t g t \}$ by simply considering each $x \in \mathtt { V a r } ( \mathbf { T } )$ as a separate letter with the ‘transition relation’ given by the corresponding $R _ { i } ( x )$ and running the corresponding automaton on it.

▶ Lemma 21. For each pair of choice functions $R _ { s r c } , R _ { t g t }$ , there exists an ATTA $\mathcal { Z } _ { R _ { s r c } , R _ { t g t } }$ of size O(max $\left| Q _ { s r c } \right| , \left| Q _ { t g t } \right| )$ such that

$$
L ( \mathcal { Z } _ { R _ { s r c } , R _ { t g t } } ) = \{ \tau \mid \sigma ( \phi _ { s r c } ( \tau ) ) \in L _ { s r c } , \sigma ( \phi _ { t g t } ( \tau ) ) \in L _ { t g t } \}
$$

where σ is a function which assigns to each variable x, a context that induces the relation $R _ { i } ( x ) ~ f o r ~ i \in \{ s r c , t g t \}$ and satisfies the guard G.

We first give an explicit construction of the ATTA $\mathcal { Z } _ { R _ { s r c } , R _ { t g t } }$ . To describe it succinctly, we define a few related concepts.

▶ Definition 22 (Destination Function). Let S, L be tree languages over finite ranked alphabets Γ, Σ respectively and let ϕ : Trees (Γ) → Trees (Σ) be a tree homomorphism. For each letter $\gamma / k \in \Gamma$ with m $: = a r i t y ( \phi ( \gamma ) )$ , we define a function $D _ { \phi } ( \gamma ) : [ k ]  2 ^ { [ m ] }$ as follows. Since $\phi ( \gamma )$ is a context of arity m, we label the holes $o f \phi ( \gamma )$ by the set [m] according to the precedence in the inorder traversal of ϕ(γ). Then, $D _ { \phi } ( \gamma ) ( i ) : = \{ j \in [ m ] \ | \ j ^ { t h }$ hole of $\phi ( \gamma )$ is labelled $u _ { i } \}$ for all $i \in [ k ]$

We illustrate this with an example. Let $\gamma / 5 \in \Gamma$ be a letter and let $\phi ( \gamma / 5 )$ be given as in Figure 8. The numbers in blue denote the order of the vertex in the inorder traversal of $\phi ( \gamma )$

![](images/956a966185a64a9e4e767d115ddcda7efdf84eb0138377d940ba11bcc8bfb15e.jpg)  
Figure 8 Example computation of $D _ { \phi } ( \gamma )$

<table><tr><td rowspan=1 colspan=6>Then $D _ { \phi } ( \gamma ) : [ 5 ]  2 ^ { [ 7 ] }$ would be given by</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1> $D _ { \phi } ( \gamma ) ( i )$ </td><td rowspan=1 colspan=1>{1, 3, 6}</td><td rowspan=1 colspan=1>{4}</td><td rowspan=1 colspan=1>{2,7}</td><td rowspan=1 colspan=1>{5}</td><td rowspan=1 colspan=1>0</td></tr></table>

Note that as demonstrated by this example, in each case we must necessarily have that the set $\{ D _ { \phi } ( \gamma ) ( i ) \mid i \in [ k ] \}$ is a partition of $\left[ \mathsf { a r i t y } ( \phi ( \gamma ) ) \right]$ . Intuitively, we also consider the function $D _ { \phi } ( \gamma )$ as a kind of “inverse” of the origin function as mentioned in [3].

Coming to the construction of the $\mathrm { A T T A }$ , for each pair of choice functions $R _ { s r c } , R _ { t g t }$ , we define ATTA $\mathcal { Z } _ { R _ { s r c } , R _ { t g t } }$ given by

$$
\mathcal { Z } _ { R _ { s r c } , R _ { t g t } } : = \Big ( ( P \big | \dot { \mathbf { z } } \big | Q _ { s r c } \big | \dot { \mathbf { z } } \big | Q _ { t g t } \big | \dot { \mathbf { z } } \big | \{ \mathrm { s t a r t } \} ) , \Gamma , I : = \{ \mathrm { s t a r t } \} , \Delta ^ { \prime } \Big )
$$

where we define $\Delta ^ { \prime }$ as follows. For a fixed letter $\gamma \in \Gamma$

For a state $p \in P$ , we set

$$
\Delta ^ { \prime } ( p , \gamma / k ) : = \bigvee _ { ( p _ { 1 } , p _ { 2 } , \ldots , p _ { k } , p ) \in \Theta ( \gamma ) } ( p _ { 1 } , 1 ) \wedge ( p _ { 2 } , 2 ) \wedge \cdots \wedge ( p _ { k } , k )
$$

This transition basically ensures that the ATTA $\mathcal { Z } _ { R _ { s r c } , R _ { t g t } }$ simulates A.

For a state $q \in Q _ { s r c }$ , we set

$$
\Delta ^ { \prime } ( q , \gamma / k ) : = \bigvee _ { ( q _ { 1 } , \dots , q _ { m } , q ) \in \Delta _ { s r c } ( \phi _ { s r c } ( \gamma ) ) } \bigwedge _ { i \in [ k ] } \bigwedge _ { j \in D _ { \phi _ { s r c } } ( i ) } ( q _ { j } , i ) \big ]
$$

Intuitively, this transition simulates the running of $\boldsymbol { B } _ { s r c }$ on $\phi _ { s r c } ( \tau )$

Similarly, for a state $q \in Q _ { t g t }$ , we set

$$
\Delta ^ { \prime } ( q , \gamma / k ) : = \bigvee _ { ( q _ { 1 } , \dots , q _ { m } , q ) \in \widetilde { \Delta _ { t g t } } ( \phi _ { t g t } ( \gamma ) ) } \bigwedge _ { i \in [ k ] } \bigwedge _ { j \in D _ { \phi _ { t g t } } ( i ) } ( q _ { j } , i ) \big ]
$$

Intuitively, this transition simulates the running of $B _ { t g t }$ on $\phi _ { t g t } ( \tau )$

We also have an ε-transition $\Delta ^ { \prime } ( \mathsf { s t a r t } , \varepsilon / 1 ) : = \bigvee _ { \substack { p \in E , q _ { s r c } \in F _ { s r c } , q _ { t g t } \in F _ { t g t } } } ( ( p , 1 ) \wedge ( q _ { s r c } , 1 ) \wedge \bigtriangledown$ $( q _ { t g t } , 1 ) )$

We illustrate this with an example. Let $q \in Q _ { s r c }$ , and $\gamma / 5 \in \Gamma$ , i.e., γ is a letter with arity 5. Also let $\phi _ { s r c } ( \gamma )$ be defined as in Figure 8. Note that we have already completely computed the extended transition relation $\bar { \Delta _ { s r c } } ( \phi _ { s r c } ( \gamma ) )$ given the choice function $R _ { s r c }$ . In such a case we define $\Delta ^ { \prime } ( q , \gamma / 5 ) : =$

$$
\begin{array} { r l } & { \bigcup } \\ & { \underset { \substack { ( q _ { 1 } , \dotsc \dotsc , q _ { 7 } , q ) } } { \bigcup } \ [ \underset { { D _ { \phi _ { \mathrm { s r c } } } ( \phi _ { \mathrm { s r c } } ( \gamma ) ) } } { ( q _ { 1 } , 1 ) \wedge ( q _ { 3 } , 1 ) \wedge ( q _ { 6 } , 1 ) }  \ \wedge \ \underset { { D _ { \phi _ { \mathrm { s r c } } } ( 2 ) = \{ 4 \} } } { ( q _ { 4 } , 2 ) } \wedge \ \underset { { D _ { \phi _ { \mathrm { s r c } } } ( 3 ) = \{ 2 , 7 \} } } { ( q _ { 2 } , 3 ) \wedge ( q _ { 7 } , 3 ) } \ \wedge \ \underset { { D _ { \phi _ { \mathrm { s r c } } } ( 4 ) = \{ 5 \} } } { ( q _ { 5 } , 4 ) } ] } \\ & { \in \widehat { \Delta } _ { \mathrm { s r c } } ( \phi _ { \mathrm { s r c } } ( \gamma ) ) } \end{array}
$$

Proof. For the given pair of choice functions $R _ { s r c } , R _ { t g t }$ , let $C _ { R _ { s r c } , R _ { t q t } }$ be a function that assigns to each $x \in \mathsf { V a r } ( \phi _ { s r c } ) \cup \mathsf { V a r } ( \phi _ { t g t } )$ , a context that induces the relation $R _ { i } ( x )$ for $i \in \{ s r c , t g t \}$ and satisfies the guard $\mathcal { G }$

$( \Leftarrow )$ Let $\sigma : = C _ { R _ { s r c } , R _ { t g t } }$ such that $\sigma ( \phi _ { s r c } ( \tau ) ) \in L _ { s r c }$ and $\sigma ( \phi _ { t g t } ( \tau ) ) ~ \in ~ L _ { t g t }$ for some $\tau \in$ Trees (Γ). This implies that for an accepting run of $\boldsymbol { B _ { s r c } }$ on $\sigma ( \phi _ { s r c } ( \tau ) )$ , the result of that run is some final state $f _ { s r c } \in F _ { s r c }$ . Similarly for an accepting run of $B _ { t g t }$ on $\sigma ( \phi _ { t g t } ( \tau ) )$ , the result of that run is some final state $f _ { t g t } \in F _ { t g t }$ . Also, since $\tau \in S$ , there is an accepting run of A on τ such that the result of that run is $e \in E$ . Then we can see that the set $\{ e , f _ { s r c } , f _ { t g t } \} \vdash I$ , the starting formula for $\mathcal { Z } _ { R _ { s r c } , R _ { t g t } }$ . From here the run of $\mathcal { Z } _ { R _ { s r c } , R _ { t g t } }$ simulates the run of A on τ, and the runs of $B _ { i }$ on $\phi _ { i } ( \tau )$ for $i \in \{ s r c , t g t \}$ conforming to the transition $C _ { R _ { s r c } , R _ { t g t } }$ . Since all of these are successful runs, the run of $\mathcal { Z } _ { R _ { s r c } , R _ { t g t } }$ we described on $\tau$ is also a successful run, and hence $\tau \in L ( \mathcal { Z } _ { R _ { s r c } , R _ { t q t } } )$

(⇒) Conversely, let $\tau \in L ( \mathcal { Z } _ { R _ { s r c } , R _ { t g t } } )$ . This implies that there is an accepting run of $\mathcal { Z } _ { R _ { s r c } , R _ { t g t } }$ on τ. Given the nature of the starting formula I, we must have an $e \in E , f _ { s r c } \in$ $F _ { s r c } , f _ { t g t } \in F _ { t g t }$ such that $\{ e , f _ { s r c } , f _ { t g t } \} \vdash I$ . Since $\mathcal { Z } _ { R _ { s r c } , R _ { t g t } }$ simulates A on τ and $e \in E .$ , we see that $\tau \in L ( \mathcal { A } )$ . Since $\mathcal { Z } _ { R _ { s r c } , R _ { t g t } }$ also simulates $B _ { i }$ on $\phi _ { i } ( \tau )$ conforming to $C _ { R _ { s r c } , R _ { t g t } }$ for all $i \in \{ s r c , t g t \}$ , we see that $\sigma ( \phi _ { i } ( \tau ) ) \in L _ { i }$ for $i \in \{ s r c , t g t \}$ . This completes the proof. ◀

We know from [4] that the emptiness problem for ATTA is EXPTime-complete in the size of the automaton. Note that this computation has been performed for a specific choice of $R _ { s r c } , R _ { t g t }$ . Let R be the set of all pairs of choice functions. To complete the solution of the problem, we need to perform this computation for each pair of choice functions $( R _ { s r c } , R _ { t g t } ) \in \mathcal { R }$ . We know that $| \mathcal { R } |$ is bounded above by $\begin{array} { r } { \prod _ { x \in \mathsf { V a r } ( \phi _ { s r c } ) } 2 ^ { \left| Q _ { s r c } \right| ^ { \mathbf { d e g } ( x ) } } } \end{array}$ $\begin{array} { r } { \prod _ { y \in \mathsf { V a r } ( \phi _ { t g t } ) } 2 ^ { \vert Q _ { t g t } \vert ^ { \mathbf { d e g } ( y ) } } } \end{array}$ . Let stateSize $\mathbf { \Psi } : = \operatorname* { m a x } \{ | Q _ { s r c } | , | Q _ { t g t } | \}$ . This gives an upper bound $\begin{array} { r } { | \mathcal { R } | \leq \prod _ { x \in \mathrm { V a r } ( \phi _ { s r c } ) } 2 ^ { \mathsf { s t a t e S i z e } ^ { \mathrm { d e g } ( \mathbf { T } ) } } \cdot \prod _ { y \in \mathsf { V a r } ( \phi _ { t g t } ) } 2 ^ { \mathsf { s t a t e S i z e } ^ { \mathrm { d e g } ( \mathbf { T } ) } } } \end{array}$ which eventually simplifies to the bound $| \mathcal { R } | \leq \Big ( 2 ^ { \mathrm { s t a t e S i z e } ^ { \mathrm { d e g } ( \mathbf { T } ) } } \Big ) ^ { \vert \mathsf { V a r } ( \mathbf { T } ) \vert }$ . We also know from Lemma 21 that the size of the ATTA $\mathcal { Z } _ { R _ { s r c } , R _ { t g t } }$ is bounded above by O(stateSize) and hence each nonemptiness check requires time given by $2 ^ { \mathcal { O } ( \mathsf { s t a t e S i z e } ) }$ . Hence the total time required for all the nonemptiness checks is given by:

Total running time is in 2<sup>O(stateSize)</sup> · 2<sup>|Var(T)|·stateSizedeg(T)</sup>

Since $| \mathtt { V a r } ( \mathbf { T } ) | + \mathtt { s t a t e S i z e } \le | \mathsf { i n p u t } | .$ , on simplifying the above, we get that the running time of the algorithm we have provided is 2<sup>poly(|input|)</sup>, where poly is a polynomial of degree deg(T). This completes the proof of Theorem 14.

## 9 Conclusion

We have introduced a formal model of tree transformations which is highly expressive. While the equivalence checking of tree transformations is undecidable we show that the type-checking problem is in 3-EXPTime by giving an automata-theoretic algorithm.

A future direction is to look at other variants of expressive tree transformations where the (source pattern, target pattern) pair is generated by more generic versions of tree transducers and study the decidability of type-checking problem.

## References

1 C. Aiswarya, Sahil Mhaskar, and M. Praveen. Checking regular invariance under tightlycontrolled string modifications. In Developments in Language Theory: 26th International Conference, DLT 2022, Tampa, FL, USA, May 9–13, 2022, Proceedings, page 57–68, Berlin, Heidelberg, 2022. Springer-Verlag. doi:10.1007/978-3-031-05578-2\_4.

2 Rajeev Alur and Loris D’antoni. Streaming tree transducers. J. ACM, 64(5), aug 2017. doi:10.1145/3092842.

3 Mikołaj Bojańczyk. Transducers with origin information. In Javier Esparza, Pierre Fraigniaud, Thore Husfeldt, and Elias Koutsoupias, editors, Automata, Languages, and Programming, pages 26–37, Berlin, Heidelberg, 2014. Springer Berlin Heidelberg.

4 Hubert Comon, Max Dauchet, Rémi Gilleron, Florent Jacquemard, Denis Lugiez, Christof Löding, Sophie Tison, and Marc Tommasi. Tree Automata Techniques and Applications. 2008. URL: https://inria.hal.science/hal-03367725.

5 Loris D’antoni, Margus Veanes, Benjamin Livshits, and David Molnar. Fast: A transducerbased language for tree manipulation. ACM Trans. Program. Lang. Syst., 38(1), oct 2015. doi:10.1145/2791292.

6 Joost Engelfriet. Bottom-up and top-down tree transformations - a comparison. Mathematical Systems Theory, 9:198–231, 06 1975. doi:10.1007/BF01704020.

7 Joost Engelfriet and Heiko Vogler. Macro tree transducers. Journal of Computer and System Sciences, 31(1):71–146, 1985. URL: https://www.sciencedirect.com/science/article/ pii/0022000085900662, doi:10.1016/0022-0000(85)90066-2.

8 Patrick C. Fischer and Arnold L. Rosenberg. Multitape one-way nonwriting automata. Journal of Computer and System Sciences, 2(1):88–101, 1968. URL: https://www.sciencedirect. com/science/article/pii/S0022000068800066, doi:10.1016/S0022-0000(68)80006-6.

9 Blaise Genest, Anca Muscholl, Olivier Serre, and Marc Zeitoun. Tree pattern rewriting systems. In Sungdeok (Steve) Cha, Jin-Young Choi, Moonzoo Kim, Insup Lee, and Mahesh Viswanathan, editors, Automated Technology for Verification and Analysis, pages 332–346, Berlin, Heidelberg, 2008. Springer Berlin Heidelberg.

10 T. V. Grifiths. The unsolvability of the equivalence problem for λ-free nondeterministic generalized machines. J. ACM, 15(3):409–413, jul 1968. doi:10.1145/321466.321473.

11 Haruo Hosoya. Foundations of XML Processing: The Tree-Automata Approach. Cambridge University Press, 2010.

12 Armin Kühnemann. Benefits of tree transducers for optimizing functional programs. In Vikraman Arvind and Sundar Ramanujam, editors, Foundations of Software Technology and Theoretical Computer Science, pages 146–157, Berlin, Heidelberg, 1998. Springer Berlin Heidelberg.

13 Florin Manea and Markus L. Schmid. Matching patterns with variables. In Robert Mercaş and Daniel Reidenbach, editors, Combinatorics on Words, pages 1–27, Cham, 2019. Springer International Publishing.

14 Tova Milo, Dan Suciu, and Victor Vianu. Typechecking for xml transformers. In Proceedings of the Nineteenth ACM SIGMOD-SIGACT-SIGART Symposium on Principles of Database Systems, PODS ’00, page 11–22, New York, NY, USA, 2000. Association for Computing Machinery. doi:10.1145/335168.335171.

15 Maurice Nivat. Transductions des langages de Chomsky. Annales de l’Institut Fourier, 18(1):339–455, 1968. URL: http://www.numdam.org/articles/10.5802/aif.287/, doi:10. 5802/aif.287.

16 Thomas J. Schaefer. The complexity of satisfiability problems. In Richard J. Lipton, Walter A. Burkhard, Walter J. Savitch, Emily P. Friedman, and Alfred V. Aho, editors, Proceedings of the 10th Annual ACM Symposium on Theory of Computing, May 1-3, 1978, San Diego, California, USA, pages 216–226. ACM, 1978. doi:10.1145/800133.804350.

## A Comparison with k-Pebble Transducers

We use the notion of k-Pebble Tree Transducer as defined in [14]. For the sake of completeness, we include the definition here.

▶ Definition 23. A k-Pebble Tree Transducer is $T : = ( \Sigma , \Sigma ^ { \prime } , Q , s , \Delta )$ , where

Σ, Σ<sup>′</sup> are ranked input and output alphabets respectively.

Q is a finite set of states.

$s \in Q$ is the initial state.

∆ is a finite set of transitions, each of which has one of the following forms:

$$
\begin{array} { r l r } { \mathrm { ~  ~ \psi ~ } } & { { } = } & { ( a , \vec { b } , q )  ( q ^ { \prime } , s t a y ) , } \end{array}
$$

$$
\begin{array} { r l r } { \mathrm { ~  ~ \psi ~ } } & { { } = } & { ( a , \vec { b } , q )  ( q ^ { \prime } , d o w n L e f t ) , } \end{array}
$$

$$
\begin{array} { r l r } { \mathrm { ~  ~ \psi ~ } } & { { } = } & { ( a , \vec { b } , q )  ( q ^ { \prime } , d o w n R i g h t ) , } \end{array}
$$

$$
\begin{array} { r l r } { \mathrm { ~  ~ \psi ~ } } & { { } = } & { ( a , \vec { b } , q )  ( q ^ { \prime } , u p L e f t ) , } \end{array}
$$

$$
\begin{array} { r l r } { \mathrm { ~  ~ \psi ~ } } & { { } = } & { ( a , \vec { b } , q )  ( q ^ { \prime } , u p R i g h t ) , } \end{array}
$$

$$
\begin{array} { r l r } { \mathrm { ~  ~ \psi ~ } } & { { } = } & { ( a , \vec { b } , q )  ( q ^ { \prime } , a d d P e b b l e ) , } \end{array}
$$

$$
\begin{array} { r l } { = } & { { } ( a , \vec { b } , q )  ( q ^ { \prime } , r e m o v e P e b b l e ) , } \end{array}
$$

$$
\mathbf { \Phi } = \mathbf { \Phi } ( a , \vec { b } , q ) \to ( a ^ { \prime } , o u t p u t 0 ) ,
$$

$$
\begin{array} { r l } { = } & { { } ( a , \vec { b } , q )  ( a ^ { \prime } ( q _ { 1 } , q _ { 2 } ) , o u t p u t 2 ) , } \end{array}
$$

$$
w h e r e \ a \in \Sigma , a ^ { \prime } \in \Sigma ^ { \prime } , \vec { b } \in \{ 0 , 1 \} ^ { i - 1 } , q , q ^ { \prime } , q _ { 1 } , q _ { 2 } \in Q .
$$

Let Γ be a finite alphabet. Let Reverse $: \Gamma ^ { * }  \Gamma ^ { * }$ be the function defined as follows: For each $w : = a _ { 1 } a _ { 2 } \dots a _ { n } \in \Gamma ^ { * }$ with $a _ { i } \in \Gamma , \ \forall i \in [ n ]$ , we let Reverse $( w ) : = a _ { n } a _ { n - 1 } \ldots a _ { 2 } a _ { 1 }$ . Let $\Sigma : = \Gamma \uplus \{ \# \}$ be a finite ranked alphabet with the ranking function rank : $\Sigma   { \mathbb { N } } _ { 0 }$ given by

$$
\operatorname { r a n k } ( a ) : = { \left\{ \begin{array} { l l } { 2 } & { { \mathrm { ~ i f ~ } } a \in \Gamma ; } \\ { 0 } & { { \mathrm { ~ i f ~ } } a = \# } \end{array} \right. }
$$

We define the tree-representation-of-string function $\phi : \Gamma ^ { * } \to \Gamma$ Trees (Σ) given by defining for $w : = a _ { 1 } a _ { 2 } \dots a _ { n } \in \Gamma ^ { * }$ with $a _ { i } \in \Gamma , \phi ( w )$ as

![](images/10b8398f74da7f02dd48bd86278538f93b8435512ef1ada5633d00deb3676072.jpg)

Note that $\phi$ as defined above is an injective function. We define a function Reverse<sup>^</sup> : $\phi [ \Gamma ^ { * } ] \to \phi [ \Gamma ^ { * } ]$ by setting Reverse <sup>^</sup> := ϕ ◦ Reverse ◦ $\phi ^ { - 1 }$ . By abuse of notation, we shall write <sup>^</sup> Reverse as just Reverse when there is no need to distinguish between the two.

▶ Theorem 24. [14] There exists a 2-Pebble Tree Transducer T such that $[ [ T ] [ t ) : = R e v e r s e ( t )$ for all $t \in \phi [ \Gamma ^ { * } ]$

Proof. Consider the 2-Pebble Tree Transducer $T$ given by $T : = ( \Sigma , \Sigma , Q , q _ { 0 } , \Delta )$ where we have $Q : = \{ q _ { 0 } , q _ { u p } , q _ { d o w n } , q _ { o u t } , q _ { l l e a f } , q _ { r l e a f } \}$ . Since there are only 2 pebbles, the bit-vector $\vec { b }$ has atmost one dimension, which is the presence or absence of the first pebble. Hence we shall denote it as ⊤ or ⊥ respectively whenever applicable. The transition rules in $\Delta$ are as follows:

$$
\begin{array} { r l } { = { } } & { { } ( a , \emptyset , q _ { 0 } )  ( q _ { d o w n } , a d d P e b b l e ) , \forall a \in \Gamma , } \end{array}
$$

(#, ∅, q<sub>0</sub>) → (#, output0),

$$
\begin{array} { r l } { = } & { { } ( a , \top , q _ { d o w n } )  ( q _ { d o w n } , d o w n R i g h t ) , \forall a \in \Gamma , } \end{array}
$$

$$
\begin{array} { r l } { = { } } & { { } ( a , \bot , q _ { d o w n } )  ( q _ { d o w n } , d o w n R i g h t ) , \forall a \in \Gamma , } \end{array}
$$

$$
\begin{array} { r l r } { \mathrm { ~ } } & { { } = } & { ( \# , \bot , q _ { d o w n } )  ( q _ { o u t } , u p L e f t ) , } \end{array}
$$

$$
\begin{array} { r l } { = } & { { } ( a , \bot , q _ { o u t } )  ( a ( q _ { l l e a f } , q _ { u p } ) , o u t p u t 2 ) , \forall a \in \Gamma , } \end{array}
$$

$$
\begin{array} { r l } { = { } } & { { } ( a , \bot , q _ { u p } )  ( q _ { o u t } , u p L e f t ) , \forall a \in \Gamma , } \end{array}
$$

$$
\begin{array} { r l } { = } & { { } ( a , \bot , q _ { l l e a f } )  ( q _ { o u t } , d o w n L e f t ) , \forall a \in \Gamma , } \end{array}
$$

$$
\begin{array} { r l r } { \mathrm { ~ } } & { { } = } & { ( \# , \bot , q _ { o u t } )  ( \# , o u t p u t 0 ) , } \end{array}
$$

$$
\begin{array} { r l } { = } & { { } ( a , \top , q _ { u p } )  ( a ( q _ { l l e a f } , q _ { r l e a f } ) , o u t p u t 2 ) , \ \forall a \in \Gamma , } \end{array}
$$

$$
\begin{array} { r l } { = } & { { } ( a , \top , q _ { r l e a f } )  ( q _ { o u t } , d o w n R i g h t ) , \forall a \in \Gamma , } \end{array}
$$

We can observe that the above 2-Pebble Tree Transducer T does reverse strings, $\mathrm { i . e . , } \mathbb { [ } T ] ( t ) =$ Reverse(t) for all $t \in \phi [ \Gamma ^ { * } ]$ ◀

▶ Theorem 25. There does not exist a triple $( L , \psi _ { 1 } , \psi _ { 2 } )$ where L is a regular tree language over ranked alphabet Λ and $\psi _ { 1 } , \psi _ { 2 }$ : Trees $( \Lambda ) $ Trees (Γ) are tree homomorphisms and a subset $X \subseteq T r e e s \left( \Lambda \right)$ such that $\{ ( \psi _ { 1 } ( t ) , \psi _ { 2 } ( t ) ) \mid t \in X \} \subseteq R e \upsilon e r s e .$

Proof. We present a proof by contradiction. Let us assume to the contrary that there exists a triple $( L , \psi _ { 1 } , \psi _ { 2 } )$ as specified above such that for a subset $X \subseteq \mathrm { T r e e s } \left( \Lambda \right)$ , we have that $\{ ( \psi _ { 1 } ( t ) , \psi _ { 2 } ( t ) ) \mid t \in X \} \subseteq$ Reverse. Then we see that when ψ , ψ as homomorphisms are restricted to the set $\Lambda , \psi _ { 1 } ( x ) , \psi _ { 2 } ( x )$ would necessarily have to be trees of height 1, for all $x \in \Lambda$ . In other words, we see that $\psi _ { 1 } , \psi _ { 2 }$ are delabelings. Hence, from Nivat’s Theorem for tree transductions as cited in [4], we know that the bimorphism $\left( L , \psi _ { 1 } , \psi _ { 2 } \right)$ is equivalent to a bottom-up tree transducer over the alphabet. Since this transducer is by default a finite-state transducer, we see that it cannot realise Reverse as a transduced function, since the height of the tree to be reversed is unbounded. This contradicts our assumption about the existence of a bimorphism equivalent to Reverse, and completes the proof. ◀

▶ Theorem 26. There does not exist an action $\mathbf { T } : = ( S , \psi _ { s r c } , \psi _ { t g t } , \mathcal { G } )$ such that $\mathbf { T } = R$ everse on $\phi [ \Gamma ^ { * } ]$

Proof. We prove the theorem by contradiction. If possible, let T be such that $\mathbf { T } =$ Reverse on $\phi [ \Gamma ^ { * } ]$ . We know that by definition, $\textstyle \mathbf { T } ( t ) : = \bigcup _ { s \in S } T _ { s } ( t )$ . Let $t _ { 0 } \in \phi [ ( a + b ) ^ { * } ]$ . Since $\mathbf { T } ( t _ { 0 } ) \ = \ \mathtt { R e v e r s e } ( t _ { 0 } ) \ \neq \ \varnothing$ , let $s _ { 0 } ~ \in ~ S$ be such that $\psi _ { s r c } ( s _ { 0 } )$ matches $t _ { 0 }$ . Let σ be a substitution witnessing the match, $\mathrm { \ i . e . , } \sigma ( \psi _ { s r c } ( s _ { 0 } ) ) = t _ { 0 }$

If there no common variable appearing in $\psi _ { s r c } ( s _ { 0 } ) , \psi _ { t g t } ( s _ { 0 } )$ , then we can substitute multiple contexts from the guard language for a variable appearing in the target pattern $\psi _ { t g t } ( s _ { 0 } )$ for the same context substituted in place of a variable in the source pattern $\psi _ { s r c } ( s _ { 0 } )$ This would contradict the fact that Reverse is a bijection.

Let x be a variable in $\psi _ { s r c } ( s _ { 0 } )$ that also appears in $\psi _ { t g t } ( s _ { 0 } )$ . Then we see that $\sigma ( x ) \in \Sigma$ $\mathrm { I f } ~ \sigma ( x ) \in \mathrm { c o n t e x t s } \left( \Sigma \right)$ has height at least 2, then since x also appears in $\psi _ { t g t } ( s _ { 0 } ) , \sigma ( \psi _ { t g t } ( s _ { 0 } ) )$ can never be Reverse $\left( t _ { 0 } \right)$ . This implies that σ is a letter-to-letter substitution, i.e., $\psi _ { s r c } ( s _ { 0 } )$ is a relabeling of $t _ { 0 }$ , and consequently, $\psi _ { t g t } ( s _ { 0 } )$ is the ‘reverse’ of $\psi _ { s r c } ( s _ { 0 } )$ . Since this is true for every element in $\phi [ ( a + b ) ^ { * } ]$ , we must have that the bimorphism $( S , \psi _ { s r c } , \psi _ { t g t } )$ is such that $\{ ( \psi _ { s r c } ( s ) , \psi _ { t g t } ( s ) ) \mid s \in S \} \subseteq$ Reverse, which contradicts Theorem 25. ◀