# A generalized transition framework for identifying reduction domains in non-binary constituency parsing

Jungyeul Park   
KAIST   
South Korea   
jungyeul@kaist.ac.kr   
Zihao Huang Eunkyul Leah Jo UBC UBC Canada Canada   
{hzhhzh1999,eunkyul}@student.ubc.ca

## Abstract

Direct non-binary constituency parsing must identify an ordered sequence of children at each phrase-level reduction. We extend our earlier delimiter-guided bottom-up construction to a common calculus for top-down, in-order, and bottom-up parsing. At reduction, the rightmost delimiter marks the left boundary, the nonterminal marker supplies the mother label, and the completed items in the delimited segment determine the children and arity. We use a stack invariant to prove soundness and completeness and derive the exact length of the static oracle’s action sequence. A projection-and-lifting theorem shows that each delimiter-guided system and its canonical counterpart generate the same set of trees under both the structural and grammarlicensed transition relations. In controlled experiments on PTB and CTB, the delimiter-guided parsers achieve similar labeled accuracy to their canonical counterparts across all traversal orders. For bottom-up parsing, delimiter guidance reduces the number of action classes and increases the number of supervised decisions per tree.

## Contents

## 1 Introduction

2 Background and notation 4   
2.1 Canonical direct non-binary systems . . . . 5   
2.2 Structural and grammar-licensed transitions . 6   
2.3 Relation to our earlier bottom-up system 7   
3 Generalized delimiter-guided transition operators 8   
3.1 Operator inventory . . . 9   
3.2 Traversal order and static-oracle choices 12   
3.3 Reachability and stack invariant 14   
5 Experiments 23   
5.1 Experimental setup 24   
5.2 Main results . 24   
6 Analysis and discussion 25   
6.1 Action inventory and derivation workload 25   
6.2 Decoding speed . 26   
6.3 Accuracy by constituent arity 28   
6.4 Sentence length and agreement between parsers 30   
6.5 Tree-language equivalence and learned decisions . 31   
7 Conclusion 32   
A Canonical transition systems 38   
B Illustrative oracle derivations 40   
C Experimental details 42

## 1 Introduction

A transition system for constituency parsing must specify how terminals are consumed, how nonterminal labels are introduced, and how partial structures are composed into larger constituents. In the binary shift-reduce tradition, binary reductions combine the two topmost stack items, while separate unary actions construct unary constituents; the action may also specify the mother label and, in lexicalized variants, the head direction. Early classifier-based and deterministic parsers adopted this architecture through actions such as Shift, Reduce-Unary-X, and Reduce-Left/Right-X (Sagae and Lavie, 2005; Wang et al., 2006; Zhang and Clark, 2009). Later neural and globally trained variants retained the same binary operational core while changing the scoring, representation, or search procedure (Zhu et al., 2013; Watanabe and Sumita, 2015; Liu and Zhang, 2017b).

Binary parsing obtains a compact local reduction operation by transforming a non-binary constituent into a sequence of binary structures. The parser consequently operates over intermediate nodes that must later be removed. Direct non-binary systems instead construct each annotated constituent from its complete child sequence in one phrase-level reduction. Top-down parsing introduces the mother label before its children are processed (Dyer et al., 2016); in-order parsing introduces it after the left corner has been completed (Liu and Zhang, 2017a); and bottom-up parsing introduces it after all children are available, conventionally selecting them through an action of the form Reduce-X#k (Fernández-González and Gómez-Rodríguez, 2019).

Despite these diferent traversal orders, every phrase-level reduction constructs the same kind of object: a mother X over an ordered, nonempty sequence of completed children $\langle \alpha _ { 1 } , \ldots , \alpha _ { k } \rangle$ We call this sequence the reduction domain and its cardinality the reduction arity. The systems difer in how they identify that domain. Top-down parsing places a nonterminal marker before the children, in-order parsing places it after the left corner, and arity-specific bottom-up parsing selects a stack sufix through the action parameter k.

Our earlier bottom-up system showed that a delimiter can identify the child sequence of a reduction and thereby determine its arity (Park et al., 2026). We take that construction as the bottom-up case and ask whether the same representation preserves the tree-building capacity of each canonical traversal. The generalized calculus represents a pending phrase-level reduction by a delimiter D, a marker nt(X), and an ordered sequence of completed children. The traversal parameter $\tau \in \{ \mathrm { T D } , \mathrm { I O } , \mathrm { B U } \}$ places the marker before all children, after the left corner, or after the complete child sequence. In each case the rightmost delimiter identifies the beginning of the relevant stack segment, the marker supplies the mother label, and the completed items determine the ordered reduction domain.

We define structural applicability independently of grammar licensing. The structural relation controls the arrangement of terminals, completed items, delimiters, and markers, whereas its grammar-licensed restriction additionally requires every constructed node to instantiate a production of a specified context-free grammar. A stack invariant characterizes the ordering, segmentation, and delimiter discipline of every reachable configuration. Its preservation supports soundness and completeness for all three traversal orders. The static oracle is determined recursively by the gold tree and has $3 n + 2 m + 1$ actions for a tree with n terminals and m phrasal nodes.

The principal formal result is a global correspondence between each delimiter-guided system and its canonical counterpart. We define projections from complete successful delimiter-guided derivations and operational liftings of complete successful canonical derivations. For top-down parsing, lifting inserts a delimiter with each phrasal marker. For in-order and bottom-up parsing, a delimiter is introduced at a terminal exactly when that terminal is the leftmost terminal of at least one phrasal constituent; reductions along the resulting first-child chain retain the delimiter until its outermost constituent is completed. The projection and lifting preserve terminal yields, mother labels, and ordered child sequences, establishing equality of the generated tree languages under both the structural and grammar-licensed relations.

We evaluate the three delimiter-guided instantiations against canonical top-down and in-order parsing and arity-specific bottom-up parsing on the Penn Treebank (Marcus et al., 1993) and Penn Chinese Treebank (Xue et al., 2005). Within each traversal order, the systems share the data, preprocessing, neural architecture, optimization procedure, and decoding strategy. The accompanying analyses examine action inventory, constituent arity, and the distinct representational choice made by binary parsing.

The resulting account identifies the reduction domain, rather than an arity-labeled action, as the central object of direct non-binary composition. Once a parser configuration identifies a mother label and an ordered child sequence, arity follows from that sequence and need not be represented as an independent transition parameter.

## 2 Background and notation

Let $w = w _ { 1 } \cdot \cdot \cdot w _ { n }$ be an input sentence. A constituency tree over w is a rooted ordered tree whose leaves are the terminals $w _ { 1 } , \ldots , w _ { n }$ . Each terminal is immediately dominated by a preterminal labeled with a part-of-speech category, and each higher internal node is labeled with a phrasal nonterminal. We assume projectivity, so the yield of every node is contiguous.

Let N and P be the finite sets of phrasal and preterminal labels, respectively, with $\mathcal { N } \cap \mathcal { P } = \mathcal { O }$ A completed preterminal item has the form $P ( w _ { i } )$ , where $P \in \mathcal { P }$ . A completed phrasal item has the form

$$
X ( \alpha _ { 1 } , \ldots , \alpha _ { k } ) ,
$$

where $X \in \mathcal { N } , \mathit { k } \geq 1$ , and each $\alpha _ { j }$ is a completed preterminal or phrasal item. The sequence

$\alpha _ { 1 } , \ldots , \alpha _ { k }$ is contiguous when its members cover adjacent, nonoverlapping yields in left-to-right order.

In the stack schemas below, σ denotes an arbitrary stack prefix and | denotes stack concatenation. The transition $\operatorname { N T } ( X )$ introduces a marker nt(X). We use $\operatorname { N T } ( X )$ for the transition and $\operatorname { N T } ( X )$ for the stack symbol that it introduces.

Definition 1 (Reduction domain and arity). For a phrase-level construction $X ( \alpha _ { 1 } , \ldots , \alpha _ { k } )$ , the ordered nonempty sequence

$$
\langle \alpha _ { 1 } , \ldots , \alpha _ { k } \rangle
$$

is its reduction domain. Its left boundary is the left edge of the yield of $\alpha _ { 1 }$ , and its arity is $k .$

## 2.1 Canonical direct non-binary systems

Top-down, in-order, and bottom-up non-binary parsing construct the same phrase-level object but encode its reduction domain diferently. Their characteristic reducible stack sufixes and reduction efects are

$$
\mathrm { T D } : \qquad \sigma \mid \operatorname { N T } ( X ) \mid \alpha _ { 1 } \mid \cdots \mid \alpha _ { k } \qquad \xrightarrow { \mathrm { R E D U C E } } \qquad \sigma \mid X ( \alpha _ { 1 } , \ldots , \alpha _ { k } ) ,
$$

$$
\operatorname { I O } : \sigma \mid \alpha _ { 1 } \mid \operatorname { N T } ( X ) \mid \alpha _ { 2 } \mid \cdots \mid \alpha _ { k } \quad { \xrightarrow { \operatorname { R e n v o c e } } } \qquad \sigma \mid X ( \alpha _ { 1 } , \ldots , \alpha _ { k } ) ,
$$

$$
\mathrm { B U : } \qquad \sigma \mid \alpha _ { 1 } \mid \cdots \mid \alpha _ { k } \qquad \xrightarrow { \mathrm { R E D U C E - } X \# k } \quad \sigma \mid X ( \alpha _ { 1 } , \ldots , \alpha _ { k } ) ,
$$

where $X \in \mathcal { N }$ and $k \geq 1$ . For $k = 1$ , the sequence $\alpha _ { 2 } , \ldots , \alpha _ { k }$ in the in-order schema is empty. These schemas display only the stack efects relevant to phrase-level reduction; the complete canonical configurations, transitions, and applicability conditions are specified in Appendix A.

In top-down parsing, the mother label is introduced before any child is constructed (Dyer et al., 2016). The unmatched marker supplies X and identifies the beginning of the reduction domain. The arity is therefore the number of completed children above that marker.

In in-order parsing, the marker is introduced after the completed left corner $\alpha _ { 1 }$ and before the remaining children (Liu and Zhang, 2017a). The reduction domain consists of the left corner immediately below the marker and the completed children above it. Top-down and in-order parsing thus recover arity from their marker configurations rather than from an arity-specific action.

In arity-specific bottom-up parsing, all children are completed before the mother label is introduced (Fernández-González and Gómez-Rodríguez, 2019). Because no marker identifies the left boundary of the child sequence, the parameter k in Reduce-X#k selects the top k completed stack items. The action therefore jointly specifies the mother label and reduction arity.

## 2.2 Structural and grammar-licensed transitions

Let G denote the context-free grammar induced by the treebank, with production set R and start symbol $S \in { \mathcal { N } }$ . A completed preterminal $P ( w _ { i } )$ instantiates a production $P  w _ { i }$ . A completed phrasal item $X ( \alpha _ { 1 } , \ldots , \alpha _ { k } )$ instantiates

$$
X  Y _ { 1 } \cdot \cdot \cdot Y _ { k } ,
$$

where $Y _ { j }$ is the label of $\alpha _ { j }$

For each traversal order $\tau \in \{ \mathrm { T D } , \mathrm { I O } , \mathrm { B U } \}$ , the structural relation $\Rightarrow \tau$ enforces the operational conditions of the transition system: label typing, input and completion conditions, traversal-specific stack patterns, delimiter and marker placement, and the contiguity and ordering of completed children. It does not require a constructed item to instantiate a production in R.

The grammar-licensed relation $\Rightarrow _ { \tau , G }$ restricts $\Rightarrow \tau$ by requiring every constructed item to be licensed by R. Formally, for any configurations c and $c ^ { \prime } .$

$$
c \Rightarrow _ { \tau , G } c ^ { \prime } \quad \Longrightarrow \quad c \Rightarrow _ { \tau } c ^ { \prime } ,
$$

or equivalently,

$$
\Rightarrow _ { \tau , G } \subseteq \Rightarrow _ { \tau } .
$$

A preterminal construction requires $P  w _ { i } \in R .$ , and a phrase-level construction requires $X $ $Y _ { 1 } \cdot \cdot \cdot Y _ { k } \ \in \ R$ . Nonconstructive transitions coincide under the two relations. This distinction separates structural well-formedness and projectivity from membership in the treebank-induced

grammar. The neural decoders select actions using implementation-specific legality checks without consulting R. The formal results concern $\Rightarrow \tau$ and $\Rightarrow _ { \tau , G }$ . Appendix C specifies where the implemented checks difer from those relations.

## 2.3 Relation to our earlier bottom-up system

Our earlier work introduced delimiter-guided non-binary bottom-up parsing (Park et al., 2026). Its phrase-level reducible sufix has the form

$$
D \mid \alpha _ { 1 } \mid \cdots \mid \alpha _ { k } \mid \operatorname { N T } ( X ) .
$$

The delimiter D identifies the left boundary, the marker supplies the mother label, and the completed items between them determine the ordered reduction domain and its arity. The actions d-Shift and d-Reduce respectively introduce this boundary and preserve it across constituents sharing the same left edge. The earlier experiments established the accompanying trade-of between fewer action classes and longer bottom-up derivations.

Yang and Tu (2022) also identify constituent boundaries without predicting a child count. Their bottom-up parser traverses spans in postorder and predicts the other endpoint of the next span from a cursor at a boundary shared with the preceding span. In the delimiter calculus, bottom-up parsing establishes a boundary when shifting its first word and can retain that boundary across multiple reductions. The reduction reads the completed children from the resulting stack segment. The separation of boundary placement, mother-label introduction, and reduction lets us compare their timing across traversal orders.

The present calculus retains this separation and changes when the boundary and label are introduced. The formal question is whether those choices admit a constructive correspondence with the canonical derivations of each traversal order.

Canonical top-down transition systems do not require an independent delimiter because an unmatched nonterminal marker already identifies the left boundary of the reduction domain. In the generalized calculus, a phrasal Nt(X) explicitly separates these roles by jointly introducing

$D \ \mid \ \operatorname { N T } ( X )$ : D records the boundary, while nt(X) records the mother label. The delimiter is not introduced by a separate predicted action and is consumed together with its marker by the corresponding Reduce; erasing D recovers the canonical top-down stack schema. By contrast, in-order and bottom-up parsing introduce the mother marker only after part or all of the child sequence has been constructed, so d-Shift must establish the boundary earlier and d-Reduce may preserve it across constituents sharing that boundary.

## 3 Generalized delimiter-guided transition operators

For each traversal order $\tau \in \{ \mathrm { T D } , \mathrm { I O } , \mathrm { B U } \}$ , a parser configuration is $c = \langle \sigma , i , f \rangle$ , where σ is a stack, $i \in \{ 1 , \ldots , n + 1 \}$ is the index of the next input token, and $f \in \{ 0 , 1 \}$ is the completion flag. The initial configuration is $\langle \epsilon , 1 , 0 \rangle$ . A successful final configuration has the form $\langle T , n + 1 , 1 \rangle$ , where T is a completed tree rooted in the start symbol S with yield $w _ { 1 } \cdots w _ { n }$

The stack may contain delimiters D, markers $\mathrm { N T } ( L )$ with $L \in \mathcal { N } \cup \mathcal { P }$ , bare terminals, and completed tree items. Each action below is a partial function on configurations, and the union of these functions defines the structural relation ⇒<sub>τ</sub>. The grammar-licensed $\mathrm { r e l a t i o n } \Rightarrow _ { \tau , G }$ additionally imposes the production conditions stated below. The top-down system excludes d-Shift and d-Reduce; the in-order and bottom-up systems admit both. Every action requires $f = 0$ , and no action is applicable after $f = 1$

In every transition schema, σ denotes an arbitrary stack prefix and the displayed elements constitute the complete sufix above it. In an in-order or bottom-up phrasal Nt schema, the displayed $D$ is the rightmost active delimiter and the sufix beginning at D contains no other delimiter or marker. In a phrase-level reduction schema, the displayed D is likewise the rightmost active delimiter, and its sufix contains no other delimiter and no marker other than the one displayed. The items $\alpha _ { 1 } , \ldots , \alpha _ { k }$ are completed tree items with nonempty consecutive yields in left-to-right order. The delimiter introduced by a top-down phrasal Nt action is fresh.

An unfinished preterminal construction must be completed before any other construction begins. In the top-down order, a stack-final $\mathrm { N T } ( P )$ admits only the corresponding Shift, and the resulting sufix $\mathrm { N T } \left( P \right) \mid w _ { i - 1 }$ admits only u-Reduce. In the in-order and bottom-up orders, a stack-final bare terminal $w _ { i - 1 }$ admits only an instance of $\operatorname { N T } ( P )$ , and the resulting sufix $w _ { i - 1 } \mid \operatorname { N T } ( P )$ admits only u-Reduce. Thus no transition places a delimiter, marker, terminal, or completed item above an unfinished preterminal construction.

## 3.1 Operator inventory

Shift For $i \leq n .$ , Shift consumes $w _ { i }$ without introducing a delimiter:

$$
\begin{array} { l l l l } { \mathrm { T D : } } & { \langle \sigma \mid \mathrm { N T } ( P ) , i , 0 \rangle } & { \Rightarrow _ { \mathrm { T D } } } & { \langle \sigma \mid \mathrm { N T } ( P ) \mid w _ { i } , i + 1 , 0 \rangle , } \\ & { \mathrm { I O } / \mathrm { B U : } } & { \langle \sigma , i , 0 \rangle } & { \Rightarrow _ { \tau } } & { \langle \sigma \mid w _ { i } , i + 1 , 0 \rangle , \qquad \tau \in \{ \mathrm { I O , B U } \} . } \end{array}
$$

In the top-down schema, $P \in \mathcal { P }$ is the pending preterminal label.

d-Shift For the in-order and bottom-up systems, d-Shift is structurally applicable when the derivation is unfinished $( f = 0 )$ , a word remains in the input $( i \leq n )$ , and no preterminal construction is pending. It pushes a fresh delimiter followed by the bare terminal $w _ { i }$ and advances the input by one position.

$$
\langle \sigma , i , 0 \rangle \Rightarrow _ { \tau } \langle \sigma \mid D \mid w _ { i } , i + 1 , 0 \rangle .
$$

The ensuing $\operatorname { N T } ( P )$ and u-Reduce leave the delimiter immediately below the completed preterminal. On the gold derivation, the oracle chooses d-Shift exactly when $w _ { i }$ is the leftmost terminal of at least one phrasal constituent, and chooses plain Shift elsewhere. A single delimiter serves all phrases that begin at that word, even when an earlier delimiter is still active. Top-down parsing introduces each delimiter with a phrasal Nt action before building its children.

$\mathbf { N T } ( L )$ For $L \in \mathcal { N } \cup \mathcal { P } , \operatorname { N T } ( L )$ introduces the marker $\mathrm { N T } ( L )$ . The expressions $\operatorname { N T } ( P )$ and $\operatorname { N T } ( X )$ below are typed schema instances of this single label-parameterized family, with $P \in \mathcal { P }$ and $X \in \mathcal { N }$ Each concrete label L nevertheless instantiates a distinct predicted action type.

For a preterminal label $P \in \mathcal { P }$ , the marker position depends on the traversal order:

$$
\begin{array} { l l l l } { \mathrm { T D : } } & { \langle \sigma , i , 0 \rangle } & { \Rightarrow _ { \mathrm { T D } } } & { \langle \sigma \mid \mathrm { N T } ( P ) , i , 0 \rangle , } & { i \leq n , } \\ { \mathrm { I O / B U : } } & { \langle \sigma \mid w _ { i - 1 } , i , 0 \rangle } & { \Rightarrow _ { \tau } } & { \langle \sigma \mid w _ { i - 1 } \mid \mathrm { N T } ( P ) , i , 0 \rangle , } \\ & & { \tau \in \{ \mathrm { I O } , \mathrm { B U } \} , } & { 2 \leq i \leq n + 1 . } \end{array}
$$

Thus, top-down parsing introduces the preterminal label before consuming its terminal, whereas in-order and bottom-up parsing introduce it afterward. Preterminal marker introduction does not introduce a delimiter.

For a phrasal label $X \in { \mathcal { N } } ,$ , the traversal order determines how the delimiter is obtained and where the marker is placed:

$$
\begin{array} { l l l l } { { \mathrm { T D } : } } & { { \langle \sigma , i , 0 \rangle } } & { { \Rightarrow _ { \mathrm { T D } } } } & { { \langle \sigma \mid D \mid \mathrm { N T } ( X ) , i , 0 \rangle , } } & { { i \leq n , } } \\ { { \mathrm { I O } : } } & { { \langle \sigma \mid D \mid \alpha _ { 1 } , i , 0 \rangle } } & { { \Rightarrow _ { \mathrm { I O } } } } & { { \langle \sigma \mid D \mid \alpha _ { 1 } \mid \mathrm { N T } ( X ) , i , 0 \rangle , } } \\ { { \mathrm { B U } : } } & { { \langle \sigma \mid D \mid \alpha _ { 1 } \mid \cdots \mid \alpha _ { k } , i , 0 \rangle } } & { { \Rightarrow _ { \mathrm { B U } } } } & { { \langle \sigma \mid D \mid \alpha _ { 1 } \mid \cdots \mid \alpha _ { k } \mid \mathrm { N T } ( X ) , i , 0 \rangle , } } & { { k \geq 1 . } } \end{array}
$$

In the top-down system, $\operatorname { N T } ( X )$ jointly introduces a fresh delimiter and the marker $\operatorname { N T } ( X )$ before any child is constructed. The corresponding Reduce removes both symbols, so neither d-Shift nor d-Reduce is required. In the in-order and bottom-up systems, the active delimiter has already been introduced by d-Shift, possibly retained by d-Reduce, and $\operatorname { N T } ( X )$ introduces only the mother-label marker after the left corner or the complete child sequence, respectively.

u-Reduce The action u-Reduce completes a preterminal:

$$
\mathrm { T D : } \qquad \langle \sigma \mid \mathrm { N T } ( P ) \mid w _ { i - 1 } , i , 0 \rangle \quad \Rightarrow _ { \mathrm { T D } } \quad \langle \sigma \mid P ( w _ { i - 1 } ) , i , 0 \rangle ,
$$

$$
\begin{array} { r l r l } & { \mathrm { I O / B U : } \quad \langle \sigma \mid w _ { i - 1 } \mid \operatorname { N T } ( P ) , i , 0 \rangle } & { \Rightarrow _ { \tau } } & & { \langle \sigma \mid P ( w _ { i - 1 } ) , i , 0 \rangle , } \\ & { } & & { } \\ & & { } & & { \tau \in \{ \mathrm { I O } , \mathrm { B U } \} , \qquad 2 \leq i \leq n + 1 . } \end{array}
$$

Under $\Rightarrow _ { \tau , G }$ , the construction additionally requires $P  w _ { i - 1 } \in R$ . The action applies only to $P \in { \mathcal { P } } $ ; a unary phrasal constituent is constructed by a phrase-level reduction with one completed

child.

Reduce Let $X \ \in \ { \mathcal { N } } ,$ let $k \geq 1$ , and let $Y _ { j }$ be the root label of $\alpha _ { j }$ . The traversal-specific reductions are

$$
\mathrm { T D : } \quad \langle \sigma \mid D \mid \operatorname { N T } ( X ) \mid \alpha _ { 1 } \mid \cdots \mid \alpha _ { k } , i , 0 \rangle \qquad \Rightarrow _ { \mathrm { T D } } \quad \langle \sigma \mid X ( \alpha _ { 1 } , \ldots , \alpha _ { k } ) , i , 0 \rangle ,
$$

$$
\mathrm { I O : } \quad \langle \sigma \mid D \mid \alpha _ { 1 } \mid \operatorname { N T } ( X ) \mid \alpha _ { 2 } \mid \cdots \mid \alpha _ { k } , i , 0 \rangle \quad \Rightarrow _ { \mathrm { I O } } \quad \langle \sigma \mid X ( \alpha _ { 1 } , \ldots , \alpha _ { k } ) , i , 0 \rangle ,
$$

$$
\begin{array} { r } { \mathrm { { B U : } } \quad \langle \sigma \mid D \mid \alpha _ { 1 } \mid \cdots \mid \alpha _ { k } \mid \operatorname { N T } ( X ) , i , 0 \rangle \qquad \Rightarrow _ { \mathrm { { B U } } } \quad \langle \sigma \mid X ( \alpha _ { 1 } , \ldots , \alpha _ { k } ) , i , 0 \rangle . } \end{array}
$$

For $k \ = \ 1$ , the sequence following the marker in the in-order schema is empty. Each reduction removes the displayed delimiter and marker and replaces the ordered reduction domain with $X ( \alpha _ { 1 } , \ldots , \alpha _ { k } )$ . Under $\Rightarrow _ { \tau , G }$ , it additionally requires

$$
X \to Y _ { 1 } \cdot \cdot \cdot Y _ { k } \in R .
$$

d-Reduce For the in-order and bottom-up systems, d-Reduce has the same structural preconditions as phrase-level Reduce. The sufix above the rightmost active delimiter must consist of one phrasal marker and one or more completed children whose yields are consecutive from left to right. The in-order marker follows exactly one child, with the remaining children above it. The bottom-up marker follows all children and is at the top of the stack. The action removes the marker, replaces the children with the new phrase, and leaves the same delimiter immediately below it. The input position stays fixed.

$$
\begin{array} { r l } & { \mathrm { I O : \quad } \langle \sigma \mid D \mid \alpha _ { 1 } \mid \operatorname { N T } ( X ) \mid \alpha _ { 2 } \mid \cdots \mid \alpha _ { k } , i , 0 \rangle \quad \Rightarrow _ { \mathrm { I O } } \quad \langle \sigma \mid D \mid X ( \alpha _ { 1 } , \ldots , \alpha _ { k } ) , i , 0 \rangle , } \\ & { \mathrm { B U : \quad } \langle \sigma \mid D \mid \alpha _ { 1 } \mid \cdots \mid \alpha _ { k } \mid \operatorname { N T } ( X ) , i , 0 \rangle \qquad \Rightarrow _ { \mathrm { B U } } \quad \langle \sigma \mid D \mid X ( \alpha _ { 1 } , \ldots , \alpha _ { k } ) , i , 0 \rangle , \qquad k \geq 1 . } \end{array}
$$

The two reductions construct the same phrase and difer only in whether the delimiter remains available for a later reduction. On the gold derivation, the oracle chooses d-Reduce when a higher phrase shares the same left boundary. This occurs exactly when the new phrase is the first child of its parent. The oracle chooses Reduce at the outermost phrase beginning at that boundary.

Under $\Rightarrow _ { \tau , G }$ , both actions require $X \to Y _ { 1 } \cdot \cdot \cdot Y _ { k } \in R$

Whenever Reduce or d-Reduce is structurally applicable, the rightmost active delimiter fixes the beginning of the reduction sufix, its single marker determines the mother label, and all completed items in that sufix determine the ordered child sequence and its arity. Grammar licensing may reject the resulting production, but it does not alter the reduction domain.

Finish After all input has been consumed, Finish applies exactly when the stack contains one completed tree rooted in S:

$$
\langle S ( \alpha _ { 1 } , \ldots , \alpha _ { k } ) , n + 1 , 0 \rangle \Rightarrow _ { \tau } \langle S ( \alpha _ { 1 } , \ldots , \alpha _ { k } ) , n + 1 , 1 \rangle , \qquad k \geq 1 .
$$

The same condition holds under $\Rightarrow _ { \tau , G }$ . The target configuration is final and admits no further transition.

## 3.2 Traversal order and static-oracle choices

For a constituent with $k \geq 1$ children, define

$$
\begin{array} { r } { \mathrm { O r d e r } _ { \tau } ( k ) = \left\{ \begin{array} { l l } { 0 , } & { \tau = \mathrm { T D } , } \\ { } & { } \\ { 1 , } & { \tau = \mathrm { I O } , } \\ { } & { } \\ { k , } & { \tau = \mathrm { B U } . } \end{array} \right. } \end{array}
$$

This value is the number of completed children preceding the mother-label marker. Let $r =$ Order (k). The complete reducible sufix for $X ( \alpha _ { 1 } , \ldots , \alpha _ { k } )$ has the uniform form

$$
D \mid \alpha _ { 1 } \mid \cdots \mid \alpha _ { r } \mid \operatorname { N T } ( X ) \mid \alpha _ { r + 1 } \mid \cdots \mid \alpha _ { k } ,
$$

where an empty child subsequence is omitted. Thus, the marker precedes all children in the topdown order, follows the left corner in the in-order order, and follows all children in the bottom-up order. The delimiter precedes the complete reduction domain in every order.

For the in-order and bottom-up static oracles, let $p _ { i }$ be the preterminal dominating $w _ { i } ,$ and let $b ( w _ { i } )$ be the highest phrasal ancestor of $p _ { i }$ reached by following first-child edges only. The value $b ( w _ { i } )$ is defined exactly when $w _ { i }$ is the leftmost terminal of at least one phrasal constituent. Define the terminal-consuming action

$$
A _ { i } = \left\{ \begin{array} { l l } { \mathrm { D - S H I F T } , } & { b ( w _ { i } ) \ \mathrm { i s ~ d e f i n e d } , } \\ { \mathrm { S H I F T } , } & { b ( w _ { i } ) \ \mathrm { i s ~ u n d e f i n e d } . } \end{array} \right.
$$

When $A _ { i } = \mathrm { D } { \cdot } \mathrm { S } \mathrm { H I F T }$ , the introduced delimiter marks the common left boundary of the phrasal constituents on the first-child path from $p _ { i }$ to $b ( w _ { i } )$ . The boundary target $b ( w _ { i } )$ is used only to define the static oracle and is not stored in the parser configuration. The in-order and bottom-up oracle code also checks whether $w _ { i }$ is sentence-initial or its preterminal and the preceding word’s preterminal have diferent parents. This check follows from the existence of $b ( w _ { i } )$ on a well-typed tree. If $w _ { i }$ is later in the sentence and $b ( w _ { i } )$ exists, $p _ { i }$ is the first child of its parent and the preceding word lies outside that parent’s yield.

For a phrasal node occurrence $v = X ( T _ { 1 } , \dots , T _ { k } )$ , let $w _ { \ell ( v ) }$ be its leftmost terminal. Its staticoracle reduction action is

$$
R _ { v } = \left\{ \begin{array} { l l } { \mathrm { R E D U C E , ~ } } & { v = b ( w _ { \ell ( v ) } ) , } \\ { \qquad \mathrm { ~ } } \\ { \mathrm { D \mathrm { - } R E D U C E , ~ } } & { v \mathrm { ~ i s ~ a ~ p r o p e r ~ d e s c e n d a n t ~ o f ~ } b ( w _ { \ell ( v ) } ) \mathrm { ~ o n ~ t h e ~ s a m e ~ f i r s t \mathrm { - } c h i l d ~ p a t h . } } \end{array} \right.
$$

These cases are exhaustive because v lies on the first-child path from the preterminal over $w _ { \ell ( v ) }$ to $b ( w _ { \ell ( v ) } )$ . Reductions below the boundary target retain the delimiter, whereas reduction of the boundary target consumes it. The in-order and bottom-up static oracles use the same $A _ { i } , b ( w _ { i } )$ ， and node-indexed reduction actions $R _ { v } ;$ they difer only in whether $\operatorname { N T } ( X )$ is introduced after the left corner or after the complete child sequence. At test time the decoder predicts the shift and reduction variants from the current configuration under its own legality checks (Appendix C).

In Figure 2, d-Shift on $M s .$ establishes the shared boundary of $N P _ { 1 } , \ S$ , and T OP. Plain Shift on Haag leaves that boundary in place. Applying ordinary Reduce at $N P _ { 1 }$ would discard the boundary still needed by S and TOP. The oracle therefore uses d-Reduce at $N P _ { 1 }$ and $S ,$ then consumes the delimiter with Reduce at TOP.

## 3.3 Reachability and stack invariant

A configuration is structurally reachable if it is the initial configuration or follows from a structurally reachable configuration by one step of $\Rightarrow \tau$ . Unless stated otherwise, reachable means structurally reachable. Because $\Rightarrow _ { \tau , G } \subseteq \Rightarrow \tau ;$ , every configuration reachable under the grammar-licensed relation is also structurally reachable.

A delimiter is active while it remains on the stack. Each active delimiter determines a delimiter segment: the portion of the stack above that delimiter and below the next active delimiter, or below the top of the stack if no delimiter occurs above it. Its boundary is the input boundary at which it was introduced; this analytical attribute is not part of the parser configuration or action alphabet. The stack invariant consists of the following conditions.

1. After delimiters and markers are ignored, the yields of the completed tree items, together with an optional bare terminal, form a disjoint left-to-right partition of the consumed input prefix $w _ { 1 } \cdot \cdot \cdot w _ { i - 1 }$ . Every completed tree item has a nonempty contiguous yield.

2. At most one preterminal construction is unfinished. In a configuration whose next-input index is $i ,$ its stack-final form is $\operatorname { N T } ( P )$ $\mathrm { N T } ( P ) \mid w _ { i - 1 }$ , for $\tau = \mathrm { T D }$ , and $w _ { i - 1 } , \mathrm { o r } w _ { i - 1 } | \mathrm { N T } ( P )$ 2 for $\tau \in \{ \mathrm { I O } , \mathrm { B U } \}$ , with the first top-down form occurring before the corresponding terminal is consumed. No other bare terminal or preterminal marker occurs on the stack.

3. Within every delimiter segment, completed tree items occur in left-to-right order and have consecutive yields. The segment contains at most one phrasal marker. In the top-down order, this marker immediately follows the delimiter and precedes every completed item in the segment. In the in-order order, exactly one completed item precedes the marker. In the bottom-up order, a nonempty sequence of completed items precedes the marker; completed items may also follow it.

4. The boundaries of active delimiters are nondecreasing in bottom-to-top stack order. Every completed item below an active delimiter has yield strictly to the left of its boundary. If a delimiter segment contains a yield-bearing symbol, its leftmost such symbol begins at the delimiter’s boundary.

## 4 Formal properties

Lemma 1 (Invariant preservation). For every $\tau \in \{ \mathrm { T D } , \mathrm { I O } , \mathrm { B U } \}$ , every configuration reachable under $\Rightarrow \tau$ satisfies the stack invariant. Consequently, every configuration reachable under $\Rightarrow _ { \tau , G }$ also satisfies it.

Proof. The proof is by induction on derivation length. The initial configuration $\langle \epsilon , 1 , 0 \rangle$ satisfies all four conditions. Suppose that a reachable configuration c satisfies the invariant and that an applicable action maps c to $c ^ { \prime } .$

A Shift consumes the next terminal and places it in the traversal-specific unfinished preterminal configuration. A d-Shift has the same terminal efect and introduces a delimiter immediately before the terminal, above every active delimiter. Neither action changes the yields or relative order of existing completed items. The applicability conditions for unfinished preterminal constructions ensure that no unrelated action can intervene before the bare terminal is completed.

A preterminal $\operatorname { N T } ( P )$ places its marker before the terminal in the top-down order and after the terminal in the in-order and bottom-up orders. The corresponding u-Reduce replaces the terminal and marker with $P ( w _ { i - 1 } )$ , whose yield is the singleton $w _ { i - 1 }$ . If a delimiter immediately precedes the terminal, both actions leave it in place, and it remains anchored immediately before the resulting preterminal. Thus the consumed-prefix partition and delimiter boundaries are preserved and the unique unfinished preterminal construction is removed.

A top-down phrasal $\operatorname { N T } ( X )$ introduces $D \ | \ \operatorname { N T } ( X )$ at the current input boundary, opening a segment whose marker precedes every completed item subsequently constructed in that segment. A d-Shift opens an in-order or bottom-up segment at the same boundary as its terminal. Both actions place the new delimiter above every active delimiter, and their boundary cannot precede the boundary of any delimiter below it. An in-order phrasal marker follows the single completed left corner of its segment, whereas a bottom-up phrasal marker follows a nonempty sequence of completed items. A phrase-level reduction replaces the marker and its ordered completed children with their mother item. Their consecutive yields have a contiguous union. Reduce removes the delimiting $D ;$ d-Reduce retains it immediately before a completed item with the same left boundary. These efects preserve the ordering and anchoring of all remaining delimiters.

Finish changes only the completion flag. Every applicable transition therefore preserves the invariant. The grammar-licensed claim follows from $\Rightarrow _ { \tau , G } \subseteq \Rightarrow _ { \tau }$ □

Theorem 1 (Soundness). For every traversal order $\tau ,$ every successful derivation under $\Rightarrow \tau$ yields a well-typed projective constituency tree rooted in S whose yield is $w _ { 1 } \cdots w _ { n }$ . If the derivation is under $\Rightarrow _ { \tau , G }$ , the resulting tree is licensed by G.

Proof. Induct on construction actions. A u-Reduce constructs $P ( w _ { i - 1 } )$ , where $P \in \mathcal { P }$ , with singleton yield $w _ { i - 1 }$ . If $\alpha _ { 1 } , \ldots , \alpha _ { k }$ are well-typed projective trees with consecutive yields, a phraselevel reduction with $X \in \mathcal { N }$ constructs $X ( \alpha _ { 1 } , \ldots , \alpha _ { k } )$ . Lemma 1 ensures that the children occur in left-to-right order over consecutive yields, so their mother is also well typed and projective.

Terminal-consuming actions read each token exactly once and in input order, and no construction action deletes, duplicates, or reorders a terminal yield. A successful Finish requires complete input consumption and a stack containing exactly one completed tree rooted in $S ;$ its yield is therefore $w _ { 1 } \cdots w _ { n }$ . Under $\Rightarrow _ { \tau , G }$ , each preterminal construction requires $P \to w _ { i - 1 } \in R ,$ and each phraselevel construction requires $X \to Y _ { 1 } \cdot \cdot \cdot Y _ { k } \in R$ , where $Y _ { j }$ is the root label of the j-th child. Induction over construction actions consequently licenses every node of the final tree. □

Definition 2 (Static oracle). Fix a projective gold tree T. Using the terminal actions $A _ { i }$ and the node-indexed phrasal reduction actions $R _ { v }$ defined in Section 3, define the node-level oracle

sequences recursively by

$$
o _ { \mathrm { T D } } ( P ( w _ { i } ) ) = N T ( P ) \cdot S H I F T \cdot U \ – { R E D U C E } ,
$$

$$
\begin{array} { r } { o _ { \mathrm { T D } } ( X ( T _ { 1 } , \dots , T _ { k } ) ) = N T ( X ) \cdot o _ { \mathrm { T D } } ( T _ { 1 } ) \cdot \cdot \cdot o _ { \mathrm { T D } } ( T _ { k } ) \cdot R E D U C E , } \end{array}
$$

$$
o _ { \mathrm { I O } } ( P ( w _ { i } ) ) = A _ { i } \cdot N T ( P ) \cdot \upsilon \ – R E D U C E ,
$$

$$
\begin{array} { r } { o _ { \mathrm { I O } } ( X ( T _ { 1 } , \dots , T _ { k } ) ) = o _ { \mathrm { I O } } ( T _ { 1 } ) \cdot N T ( X ) \cdot o _ { \mathrm { I O } } ( T _ { 2 } ) \cdot \cdot \cdot o _ { \mathrm { I O } } ( T _ { k } ) \cdot R _ { X ( T _ { 1 } , \dots , T _ { k } ) } , } \end{array}
$$

$$
o _ { \mathrm { B U } } ( P ( w _ { i } ) ) = A _ { i } \cdot N T ( P ) \cdot \mathit { U } \mathrm { - } R E D U C E ,
$$

$$
o _ { \mathrm { B U } } ( X ( T _ { 1 } , \dots , T _ { k } ) ) = o _ { \mathrm { B U } } ( T _ { 1 } ) \cdot \cdot \cdot o _ { \mathrm { B U } } ( T _ { k } ) \cdot N T ( X ) \cdot R _ { X ( T _ { 1 } , \dots , T _ { k } ) } .
$$

Here · denotes sequence concatenation, and the sequence $o _ { \mathrm { I O } } ( T _ { 2 } ) \cdot \cdot \cdot o _ { \mathrm { I O } } ( T _ { k } )$ is empty when $k = 1$ The complete oracle sequence is

$$
{ \cal O } _ { \tau } ( T ) = o _ { \tau } ( T ) \cdot F I N I S H .
$$

For a fixed projective tree $T ,$ its ordered node structure determines every $A _ { i }$ and every nodeindexed reduction action $R _ { v } { , }$ , so these recursive clauses uniquely determine $O _ { \tau } ( T )$

Lemma 2 (Boundary lifecycle). Let $\tau \in \{ \mathrm { I O } , \mathrm { B U } \}$ , and construct a projective tree T in recursive in-order when $\tau = \mathrm { I O }$ or recursive postorder when $\tau = \mathrm { B U }$ . Let $w _ { i }$ be the leftmost terminal of at least one phrasal constituent. The phrasal constituents with ${ \it l e f t }$ boundary i form a chain $T _ { 1 } , \dots , T _ { q } ,$ ordered from innermost to outermost. If $w _ { i }$ is consumed by d-Shift, the resulting delimiter is the rightmost active delimiter when each $T _ { j }$ is reduced. It is retained by d-Reduce for $j ~ < ~ q$ and consumed by Reduce for $j = q ;$ every delimiter introduced strictly inside the yield of $T _ { j }$ is consumed before $T _ { j }$ is reduced.

Proof. Constituent yields are contiguous, so any two phrasal constituents with the same left boundary are nested and therefore form a first-child chain. The d-Shift consuming $w _ { i }$ places a delimiter immediately before that terminal, and completion of its preterminal leaves the delimiter immediately before the resulting item.

Prove the claim simultaneously for all boundaries by induction on the yield length of the outermost member of the chain. If that yield contains one terminal, no later child can introduce a delimiter; the traversal-specific schemas successively reduce the unary chain and establish the claim. Otherwise, a child not lying on the first-child chain has a strictly later left boundary and a strictly shorter yield. $\mathrm { B y }$ the induction hypothesis, every delimiter opened within that child is removed by the outermost constituent sharing its boundary before the traversal returns from the child. Conse quently, when $T _ { j }$ is ready for reduction, no delimiter introduced inside a later child remains above the delimiter at i. The latter is therefore the rightmost active delimiter, and its segment contains exactly the completed children of $T _ { j }$ , with the marker in the position prescribed by the traversal order. Thus d-Reduce is applicable for $j < q$ and retains the delimiter immediately before $T _ { j }$ . The same argument then applies to $T _ { j + 1 }$ . At $T _ { q }$ , Reduce is applicable to the same delimited domain and removes the delimiter. □

Theorem 2 (Completeness). Let T be a well-typed projective constituency tree over $w _ { 1 } \cdots w _ { n }$ , with preterminal labels in ${ \mathcal { P } } ,$ phrasal labels in ${ \mathcal { N } } ,$ and root label S. For every traversal order $\tau , O _ { \tau } ( T )$ is a successful derivation under $\Rightarrow \tau$ . $I f T$ is licensed by $G ,$ the same sequence is successful under $\Rightarrow _ { \tau , G }$

Proof. We prove by structural induction that $o _ { \tau } ( U )$ constructs every subtree U over its complete yield. For $\tau \in \{ \mathrm { I O } , \mathrm { B U } \}$ , the induction claim additionally records the delimiter at the left boundary of $U { : }$ every delimiter introduced strictly inside the yield of U has been consumed when U is completed, while the left-boundary delimiter remains immediately below $U$ precisely when $U$ is a proper descendant of its boundary target. It is consumed when U is that target.

For a preterminal $P ( w _ { i } )$ , the top-down sequence

$$
\mathrm { N T } ( P ) \cdot \mathrm { S H I F T } \cdot \mathrm { U } – \mathrm { R E D U C E }
$$

constructs $P ( w _ { i } )$ . In the in-order and bottom-up orders,

$$
A _ { i } \cdot \mathrm { N T } ( P ) \cdot \mathrm { U } \mathrm { - } \mathrm { R E D U C E }
$$

constructs the same item. If $A _ { i } = \operatorname { S H I F T }$ , no delimiter is introduced. If $A _ { i } = \mathrm { D } { \cdot } \mathrm { S } \mathrm { H I F T }$ , the result is

$D \mid P ( w _ { i } )$ , and D is retained for the phrasal first-child chain beginning at $w _ { i }$ . This establishes the strengthened base case.

Let $U = X ( U _ { 1 } , \dots , U _ { k } )$ , and assume the claim for its children. In the top-down order, $\operatorname { N T } ( X )$ introduces $D \ | \ \operatorname { N T } ( X )$ before their oracle sequences. Once all children have been completed, the stack has sufix

$$
D \mid \operatorname { N T } ( X ) \mid U _ { 1 } \mid \cdots \mid U _ { k } ,
$$

so Reduce constructs $U$ and removes the delimiter.

In the in-order order, $U _ { 1 }$ is completed before $\operatorname { N T } ( X )$ is introduced, and $U _ { 2 } , \dots , U _ { k }$ are completed afterward. In the bottom-up order, all k children are completed before marker introduction. Delimiters opened within children $U _ { j }$ for $j > 1$ are consumed within those children because their left boundaries difer from that of $U$ . By Lemma 2, the delimiter at the left boundary of $U$ is the rightmost active delimiter when $R _ { U }$ is applied. The reduction domain therefore has the required traversal-specific form. If $U$ is below its boundary target, d-Reduce constructs $U$ and retains the delimiter immediately below it; if $U$ is the boundary target, Reduce constructs $U$ and consumes the delimiter. The strengthened induction claim follows.

The root is the outermost constituent at its left boundary, so its reduction consumes the last retained delimiter. After $o _ { \tau } ( T )$ , all input has been consumed and the stack contains only $T ;$ Finish is therefore applicable. If T is licensed by $G ,$ every construction action instantiates its corresponding production in $R ,$ so the same sequence is licensed under $\Rightarrow _ { \tau , G }$ □

Proposition 1 (Oracle action count). Let $T$ contain n terminals, n preterminals, and m phrasal nodes. For every traversal order $\tau ,$

$$
| O _ { \tau } ( T ) | = 3 n + 2 m + 1 .
$$

The top-down oracle introduces exactly m delimiters, whereas the in-order and bottom-up oracles each introduce

$$
| \{ i : b ( w _ { i } ) \ i s \ d e f i n e d \} | \le n
$$

delimiters. Every introduced delimiter is consumed by exactly one Reduce.

The canonical top-down and in-order oracles have the same action count, whereas the arityspecific canonical bottom-up oracle has $2 n + m + 1$ actions.

Proof. Each terminal and its preterminal contribute one terminal-consuming action, one $\operatorname { N T } ( P )$ and one u-Reduce, giving 3n actions. Each phrasal node contributes one $\operatorname { N T } ( X )$ and one Reduce or d-Reduce, giving 2m actions. Finish contributes one action. The canonical top-down and in-order systems make the same three terminal-level and two phrase-level decisions. In the canonical bottom-up system, each terminal contributes Shift and Reduce- $P \# 1$ , each phrasal node contributes one arity-specific reduction, and Finish contributes one action, giving $2 n + m + 1$

Every top-down phrasal Nt introduces a delimiter, giving m delimiters. In the other two orders, a delimiter is introduced exactly when $b ( w _ { i } )$ is defined. Lemma 2 shows that reductions below the associated boundary target retain it and that reduction of the target consumes it. □

Definition 3 (Canonical projection). For $\tau \in \{ \mathrm { T D } , \mathrm { I O } , \mathrm { B U } \}$ , let D<sub>τ</sub> denote the delimiter-guided system and $\mathcal { C } _ { \tau }$ its canonical counterpart in Appendix A. In each canonical system, a reduction over a bare terminal is a preterminal construction labeled by $P \in \mathcal { P }$ , a reduction over completed tree items is a phrasal construction labeled by $X \in \mathcal { N } _ { : }$ , and an unfinished preterminal construction must be completed before any other construction begins. For top-down and in-order action–configuration sequences, $\Pi _ { \tau }$ erases delimiters from every configuration and transforms actions as follows. In the top-down order, it preserves $N T ( L )$ , Shift, and Finish, and maps both u-Reduce and phrase-level Reduce to canonical Reduce. In the in-order order, it maps both Shift and d-Shift to canonical Shift, preserves $N T ( L )$ and Finish, and maps u-Reduce, Reduce, and d-Reduce to canonical Reduce.

In the bottom-up order, $\Pi _ { \mathrm { B U } }$ is a macro projection on successful action–configuration sequences. At completed-construction boundaries it erases delimiters from the stack, and it contracts

(Shift or d-Shift) · Nt(P) · u-Reduce

to canonical Shift · Reduce- $P \# 1$ , and contracts

$$
N T ( X ) \cdot ( R E D U C E ~ o r ~ D – R E D U C E )
$$

to Reduce-X#k, where k is the number of completed children in the delimited reduction domain.   
It preserves Finish.

Theorem 3 (Global derivational correspondence). For every traversal order $\tau \in \{ \mathrm { T D } , \mathrm { I O } , \mathrm { B U } \} , \Pi _ { \tau }$ maps every successful derivation of $\mathcal { D } _ { \tau }$ to a successful derivation of $\scriptstyle { \mathcal { C } } _ { \tau }$ with the same input and final constituency tree. Conversely, every successful derivation $\delta$ of $ { \mathcal { C } } _ { \tau }$ has a successful delimiter-guided lifting $\Lambda _ { \tau } ( \delta )$ that produces the same tree and satisfies

$$
\Pi _ { \tau } \big ( \Lambda _ { \tau } ( \delta ) \big ) = \delta .
$$

Both directions hold for the structural relations and their grammar-licensed restrictions. Writing $\mathcal { L }$ and $\mathcal { L } _ { G }$ for the sets of final trees generated under the structural and grammar-licensed relations, respectively,

$$
{ \mathcal { L } } ( { \mathcal { D } } _ { \tau } ) = { \mathcal { L } } ( { \mathcal { C } } _ { \tau } ) \qquad { a n d \qquad } { \mathcal { L } } _ { G } ( { \mathcal { D } } _ { \tau } ) = { \mathcal { L } } _ { G } ( { \mathcal { C } } _ { \tau } ) .
$$

Proof. Consider first a successful delimiter-guided derivation. In the top-down order, erasing D from a phrasal source sufix

$$
D \mid \operatorname { N T } ( X ) \mid \alpha _ { 1 } \mid \cdots \mid \alpha _ { k }
$$

gives exactly the canonical top-down reduction domain. A u-Reduce likewise has the same stack efect as the canonical reduction that constructs its preterminal. All other projected actions have identical marker, terminal, or completion efects. Induction on the derivation therefore gives a successful canonical derivation with the same completed items and final tree.

In the in-order order, Shift and d-Shift have the same efect after delimiter erasure. Erasing D from

$$
D \mid \alpha _ { 1 } \mid \mathrm { N T } ( X ) \mid \alpha _ { 2 } \mid \cdots \mid \alpha _ { k }
$$

gives the canonical in-order reduction domain with the same mother label and ordered children, whether the delimiter-guided reduction retains or removes D. Preterminal reductions project in the same way. Induction again preserves every completed item and the final tree.

In a successful bottom-up derivation, the applicability conditions force every consumed bare terminal to be completed by the following $\operatorname { N T } ( P )$ ·u-Reduce. Once a bottom-up phrasal marker has been introduced, any intervening action would place material above it and make its reduction schema permanently unavailable; hence every successful derivation discharges that marker immediately by Reduce or d-Reduce. The derivation therefore factors into the blocks specified by $\Pi _ { \mathrm { B U } }$ . Each projected preterminal block constructs the same $P ( w _ { i } )$ , and each projected Reduce- $X \# k$ consumes the same ordered children and constructs the same mother item. Induction over these blocks yields a successful canonical derivation with the same final tree

For the converse, fix a successful typed canonical derivation and its final tree. Terminals are shifted from left $\tan$ right, and every construction consumes a complete top sufix. The canonical stack schemas therefore force the nodes of the fixed tree to be constructed in the corresponding recursive order: top-down marker introduction, recursive in-order construction, or recursive postorder construction. Bypassing a required marker or child would bury it below a yield-bearing item outside its reduction domain, after which it could not occupy the required sufix. The following constructions define $\Lambda _ { \tau }$ . In the top-down order, retain every canonical preterminal construction, replacing its closing Reduce with u-Reduce. For every phrasal $\operatorname { N T } ( X )$ , insert D immediately before its marker and use phrase-level Reduce to remove that delimiter when the canonical marker is closed. The canonical nesting of markers makes each inserted delimiter the rightmost active delimiter at the matching reduction. Erasing the delimiters and identifying the two delimiter-guided reduction types with canonical Reduce recovers the original derivation.

In the in-order order, replace the canonical Shift consuming $w _ { i }$ with d-Shift exactly when $w _ { i }$ is the leftmost terminal of at least one phrasal constituent in the final tree; otherwise retain Shift. Preserve each $\mathrm { { N T } } ( L )$ , replace a canonical reduction over a bare terminal with u-Reduce, and replace each phrase-level reduction by d-Reduce for a non-outermost member of its commonleft-boundary chain and by Reduce for the outermost member. The canonical schemas impose the same node-construction order as the recursive in-order traversal. Lemma 2 therefore ensures that every selected reduction is applicable and that every introduced delimiter is consumed. Projection action by action recovers the canonical derivation.

In a successful typed canonical bottom-up derivation, a Shift introducing $w _ { i }$ is immediately followed by Reduce-P#1: if another item were pushed first, the bare terminal could no longer form the top reduction domain required for its preterminal. Replace this pair with Shift or d-Shift, followed by $\operatorname { N T } ( P )$ · u-Reduce, using d-Shift exactly when $w _ { i }$ is the leftmost terminal of at least one phrasal constituent. Replace every phrase-level Reduce-X#k with $\operatorname { N T } ( X )$ followed by the d-Reduce or Reduce selected from its common-left-boundary chain. The canonical bottom-up reduction order is the recursive postorder used in Lemma $2 ;$ all lifted blocks are therefore applicable, and no delimiter remains after the root is completed. Each block projects to its original canonical action or action pair, establishing $\Pi _ { \mathrm { B U } } ( \Lambda _ { \mathrm { B U } } ( \delta ) ) = \delta$

Projection and lifting preserve every terminal yield, mother label, and ordered child sequence. They consequently preserve the production instantiated by each construction. The three arguments therefore apply unchanged to the grammar-licensed restrictions, and the two tree-language equalities follow. □

## 5 Experiments

We compare six direct non-binary parsers using phrase-structure annotations from the Wall Street Journal portion of Penn Treebank 3 (PTB) (Marcus et al., 1993; Taylor et al., 2003) and from Chinese Treebank 5.1 (CTB) (Xue et al., 2005). For each traversal order, we train the canonical system and its delimiter-guided counterpart on the same trees with the same neural architecture, optimization protocol, and decoding strategy. The systems difer in how they encode the reduction domain and in the corresponding action inventory, applicability checks, and oracle sequence. The comparison tests whether replacing the canonical domain encoding incurs a loss in labeled constituency accuracy under static-oracle training.

## 5.1 Experimental setup

For each of the three traversal orders—top-down, in-order, and bottom-up—we compare a canonical parser with its delimiter-guided counterpart. All six parsers construct non-binary branching structures directly. We implement their transitions and static oracles in a shared Stanza training and decoding pipeline (Qi et al., 2020; Bauer and Manning, 2025). All parsers learn from the resulting gold action sequences by teacher forcing. At test time, each parser greedily selects the highestscoring action permitted by its implementation’s structural checks, with no production lookup in the treebank grammar. Appendix C specifies the training and decoding protocol.

Each system–treebank pair has three independent runs (seeds 1111, 2222, and 3333), for a total of 36 runs. We select a checkpoint for each run by labeled development-set $F _ { 1 }$ . We score its test predictions with evalb (Black et al., 1991) and jp-evalb (Jo et al., 2024; Park et al., 2024). Both scorers receive the same saved gold and predicted trees, with matching token yields and reference POS labels in the exported files. The former uses the supplied COLLINS.prm on both corpora. The latter uses its default alignment mode, which retains punctuation. We compute corpus $F _ { 1 }$ independently for each seed and report the mean and sample standard deviation of the three scores.

## 5.2 Main results

The absolute diference between paired canonical and delimiter-guided means is at most 0.32 $F _ { 1 }$ points across the twelve comparisons in Table 1, with diferences computed before rounding. Topdown diferences stay within 0.11 points. In-order means are lower with delimiters by 0.14–0.20 points across the two corpora and scorers. For bottom-up parsing, the PTB means difer by less than 0.07 points, while the CTB means rise by 0.31 points under evalb and 0.32 under jp-evalb. The bottom-up pair thus preserves similar labeled constituency accuracy while changing the way the parser identifies a child sequence. These diferences describe the observed runs. The experiment specifies no statistical equivalence margin.

Table 1: Labeled test $F _ { 1 }$ on PTB and CTB. The published evalb scores come from a generative top-down parser, a greedy in-order parser trained with a static oracle, and a greedy bottom-up parser trained with a dynamic oracle. These scores reflect the training and evaluation settings of the original studies. All six systems in this study use a shared static-oracle training protocol and greedy decoding. Their scores are means with sample standard deviations across three seeds. Dashes indicate unreported jp-evalb scores.
<table><tr><td></td><td colspan="2">PTB</td><td colspan="2">CTB</td></tr><tr><td>System</td><td>EVALB</td><td>JP-EVALB</td><td>EVALB</td><td>JP-EVALB</td></tr><tr><td>Previously reported canonical non-binary systems</td><td></td><td></td><td></td><td></td></tr><tr><td>Top-down (Dyer et al., 2016)</td><td>92.40</td><td></td><td>82.70</td><td></td></tr><tr><td>In-order (Liu and Zhang, 2017a)</td><td>91.80</td><td></td><td>86.10</td><td></td></tr><tr><td>Bottom-up (Fernández-González and Gómez-Rodríguez, 2019)</td><td>91.70</td><td></td><td>86.80</td><td></td></tr><tr><td>Canonical non-binary systems, reimplemented</td><td></td><td></td><td></td><td></td></tr><tr><td>Top-down</td><td> $9 4 . 8 6 \pm 0 . 0 7$ </td><td> $9 4 . 2 7 \pm 0 . 0 9$ </td><td> $8 9 . 8 3 \pm 0 . 2 8 $ </td><td> $8 8 . 0 9 \pm 0 . 3 1$ </td></tr><tr><td>In-order</td><td> $9 5 . 0 4 \pm 0 . 0 7$ </td><td> $9 4 . 3 5 \pm 0 . 0 8$ </td><td> $9 0 . 6 2 \pm 0 . 1 9$ </td><td> $8 8 . 8 9 \pm 0 . 2 0$ </td></tr><tr><td>Bottom-up</td><td> $9 4 . 9 0 \pm 0 . 0 4$ </td><td> $9 4 . 2 7 \pm 0 . 0 9$ </td><td> $9 0 . 6 3 \pm 0 . 2 7$ </td><td> $8 8 . 8 9 \pm 0 . 2 6$ </td></tr><tr><td>Delimiter-guided systems</td><td></td><td></td><td></td><td></td></tr><tr><td>Top-down</td><td> $9 4 . 8 9 \pm 0 . 1 0$ </td><td> $9 4 . 2 5 \pm 0 . 1 1$ </td><td> $8 9 . 9 0 \pm 0 . 3 3 $ </td><td> $8 8 . 2 0 \pm 0 . 3 5$ </td></tr><tr><td>In-order</td><td> $9 4 . 8 4 \pm 0 . 0 5$ </td><td> $9 4 . 1 8 \pm 0 . 0 4$ </td><td> $9 0 . 4 8 \pm 0 . 3 8$ </td><td> $8 8 . 7 4 \pm 0 . 3 4$ </td></tr><tr><td>Bottom-up</td><td> $9 4 . 9 0 \pm 0 . 0 2$ </td><td> $9 4 . 2 1 \pm 0 . 0 4$ </td><td> $9 0 . 9 4 \pm 0 . 3 3$ </td><td> $8 9 . 2 0 \pm 0 . 3 3 $ </td></tr></table>

## 6 Analysis and discussion

## 6.1 Action inventory and derivation workload

We count oracle actions on the same gold test trees for each system, including explicit preterminal construction, the ROOT node, and the final Finish. Counting actions on a common output separates derivation length from diferences in the structures produced by greedy decoding. Table 2 pairs these counts with the action classes observed in the filtered, retagged training data.

Top-down and in-order parsing already locate the reduction domain through a nonterminal marker. Adding delimiters preserves their oracle lengths and adds a small number of classes for preterminal construction and, in in-order parsing, for shifts and reductions with a delimiter efect. The bottom-up change is larger. Canonical bottom-up assigns a separate class to each observed label–arity pair. Delimiter-guided bottom-up uses a label marker and recovers the child sequence from the stack. The vocabulary contracts from 251 to 82 classes on PTB and from 227 to 68 on CTB. Each preterminal and phrasal node now requires a marker action, as in Proposition 1. The resulting gold derivations are 64% longer on PTB and 67% longer on CTB.

The narrower output layer and the longer derivation afect diferent parts of the scoring workload.

Table 2: Action classes observed during training and mean scoring workload per gold test sentence. Actions count oracle decisions. MACs are millions of multiply–accumulate operations in the two afine layers of the action scorer.
<table><tr><td></td><td colspan="3">PTB</td><td colspan="3">CTB</td></tr><tr><td>Order</td><td>Classes</td><td>Actions</td><td>MACs (M)</td><td>Classes</td><td>Actions</td><td>MACs (M)</td></tr><tr><td>Canonical</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Top-down</td><td>79</td><td>110.04</td><td>17.21</td><td>65</td><td>122.15</td><td>18.67</td></tr><tr><td>In-order</td><td>79</td><td>110.04</td><td>17.21</td><td>65</td><td>122.15</td><td>18.67</td></tr><tr><td>Bottom-up</td><td>251</td><td>67.25</td><td>13.48</td><td>227</td><td>73.08</td><td>14.20</td></tr><tr><td colspan="7">Delimiter-guided</td></tr><tr><td>Top-down</td><td>80</td><td>110.04</td><td>17.24</td><td>66</td><td>122.15</td><td>18.70</td></tr><tr><td>In-order</td><td>82</td><td>110.04</td><td>17.30</td><td>68</td><td>122.15</td><td>18.76</td></tr><tr><td>Bottom-up</td><td>82</td><td>110.04</td><td>17.30</td><td>68</td><td>122.15</td><td>18.76</td></tr></table>

The decoder computes a score for every class before checking legality. For bottom-up parsing, the number of output scores along a gold derivation falls by 47% on PTB and 50% on CTB. The first afine layer, however, maps 532 features to 256 hidden units at every decision in all 36 checkpoints. Counting both afine layers therefore gives the opposite direction. The mean number of multiply– accumulate operations in the bottom-up action scorer rises by 28% on PTB and 32% on CTB. The extra decisions repeat the hidden computation as well as the smaller output projection.

The workload calculation uses formal oracle lengths, including for test trees that require an action class absent from the learned inventory. Encoding, composition, stack operations, and legality checks contribute separately to decoding cost. In particular, several delimiter checks scan the current stack, so the number of actions does not determine the cost of those checks.

## 6.2 Decoding speed

We benchmark greedy decoding at batch size one on AMD EPYC 7352 cores. For each corpus we sample 20 test sentences without replacement from each of five token-length bands (1–10, 11–20, 21–30, 31–40, and 41 or more). All 36 checkpoints receive the same 100 sentences per corpus, in the same order and with the same predicted POS tags. We load the checkpoints in one Stanza source tree supporting all six systems. Each corpus–seed block runs its six parsers sequentially, pinned to one core with one inference thread. The six blocks run concurrently on cores with separate L3 caches. We rotate the traversal order across seeds and alternate which encoding runs first within a pair.

For each checkpoint we warm up on five additional sentences, one per length band, and time one pass over the 100-sentence sample. The wall-clock interval covers encoding of the pretagged sentence, action scoring, legality checks, and tree construction. POS tagging, checkpoint loading, warm-up, export, and evaluation are outside the timed interval. All decodes completed with the reference token yield. After restoring reference POS tags for comparison, 3,599 of the 3,600 parses matched their archived predictions exactly.

Table 3: Single-core greedy decoding throughput on the length-balanced test samples, in sentences per second. For each checkpoint, throughput is the number of timed sentences divided by their total elapsed parsing time. Entries give the mean and sample standard deviation across the three seeds.
<table><tr><td></td><td colspan="2">PTB</td><td colspan="2">CTB</td></tr><tr><td>Order</td><td>Canonical</td><td>Delimiter-guided</td><td>Canonical</td><td>Delimiter-guided</td></tr><tr><td>Top-down</td><td> $0 . 9 2 \pm 0 . 0 4$ </td><td> $0 . 9 0 \pm 0 . 0 3$ </td><td> $1 . 6 1 \pm 0 . 0 1$ </td><td> $1 . 4 9 \pm 0 . 0 4$ </td></tr><tr><td>In-order</td><td> $0 . 9 2 \pm 0 . 1 0$ </td><td> $0 . 9 3 \pm 0 . 0 9$ </td><td> $1 . 5 8 \pm 0 . 0 5$ </td><td> $1 . 5 4 \pm 0 . 0 6$ </td></tr><tr><td>Bottom-up</td><td> $1 . 0 0 \pm 0 . 0 6$ </td><td> $0 . 9 1 \pm 0 . 0 6$ </td><td> $1 . 8 3 \pm 0 . 0 5$ </td><td> $1 . 5 5 \pm 0 . 0 3$ </td></tr></table>

Table 3 reports throughput for the length-balanced samples. On PTB, top-down and in-order throughput changes by at most 0.02 sentences per second when delimiters are introduced. On CTB, the mean falls from 1.61 to 1.49 sentences per second for top-down parsing and from 1.58 to 1.54 for in-order parsing. Canonical bottom-up has the highest mean throughput on both corpora. Delimiter-guided bottom-up reduces throughput by 9% on PTB and 15% on CTB relative to its canonical counterpart.

Mean decoding time increases with sentence length for all six parsers on both corpora (Figure 1). Delimiter-guided bottom-up takes longer than canonical bottom-up in every length band. For sentences with 41 or more tokens, its mean time rises from 1.56 to 1.76 seconds on PTB and from 0.97 to 1.16 seconds on CTB. These gaps have the same direction as the scorer workload diferences in Table 2. The measured time includes encoding and all decoding operations, so it does not isolate the contribution of any one component.

![](images/f257083c30738869120679faa12296f6fd8d7176df49dffb3089c03c04eed910.jpg)  
Figure 1: Single-core decoding time by sentence length. We average the wall-clock time over the same 20 sentences per band for each checkpoint. Points show the means across seeds and error bars show one sample standard deviation. The two corpus rows have separate vertical scales.

## 6.3 Accuracy by constituent arity

We compare the six systems on their recovery of constituents of diferent arities. For this diagnostic we match phrasal occurrences within each test sentence by label, terminal span, and exact immediate child count. We retain punctuation, omit preterminals and the outer ROOT wrapper, and count repeated occurrences with multiplicity. Exact arity is required even in the group with five or more children. We sum matches and gold and predicted occurrences within each group to compute corpus $F _ { 1 }$ per seed.

All six systems recover unary and binary constituents more accurately than constituents with three or more children. The efect of delimiter guidance varies with traversal order. On PTB, the mean for the group with at least five children rises for each order, by 0.33–0.77 points. On CTB, the four-child mean rises by 2.09 points for top-down parsing and falls by 2.22 for in-order parsing. These two directions hold in all three CTB seed pairs. The bottom-up mean for five or more children falls by 1.57 points. These CTB groups contain 184 and 235 gold occurrences, respectively. The arity results thus identify where the systems difer even when their aggregate scores are close. The criterion jointly measures span, label, and child count.

Table 4: Exact-arity labeled test $F _ { 1 }$ by immediate child count. Gold counts give the number of reference phrasal occurrences in each group. Results are means ± sample standard deviations across three seeds. C denotes canonical and D delimiter-guided.
<table><tr><td>System</td><td>1</td><td>2</td><td>3</td><td>4</td><td>≥5</td></tr><tr><td colspan="6">PTB</td></tr><tr><td>Gold count</td><td>8,879</td><td>23,840</td><td> $^ { 7 , 3 8 1 }$ </td><td> $^ { 2 , 3 1 7 }$ </td><td>1,859</td></tr><tr><td>C Top-down</td><td> $9 3 . 4 8 \pm 0 . 0 7$ </td><td> $9 2 . 8 8 \pm 0 . 1 1$ </td><td> $8 8 . 4 6 \pm 0 . 2 9$ </td><td> $8 1 . 2 9 \pm 0 . 5 9$ </td><td> $8 1 . 2 8 \pm 0 . 6 2$ </td></tr><tr><td>C In-order</td><td> $9 3 . 6 5 \pm 0 . 1 3$ </td><td> $9 3 . 1 0 \pm 0 . 0 6$ </td><td> $8 8 . 5 7 \pm 0 . 1 6$ </td><td> $8 1 . 4 2 \pm 0 . 1 2$ </td><td> $8 1 . 6 4 \pm 0 . 4 3$ </td></tr><tr><td>C Bottom-up</td><td> $9 3 . 5 3 \pm 0 . 0 4$ </td><td> $9 2 . 9 3 \pm 0 . 1 2$ </td><td> $8 8 . 4 3 \pm 0 . 2 1$ </td><td> $8 1 . 8 2 \pm 0 . 5 4$ </td><td> $8 1 . 7 7 \pm 0 . 1 3$ </td></tr><tr><td>D Top-down</td><td> $9 3 . 4 9 \pm 0 . 0 7$ </td><td> $9 2 . 8 6 \pm 0 . 1 4$ </td><td> $8 8 . 5 0 \pm 0 . 3 8$ </td><td> $8 1 . 1 1 \pm 0 . 2 7$ </td><td> $8 2 . 0 5 \pm 0 . 5 8$ </td></tr><tr><td>D In-order</td><td> $9 3 . 4 4 \pm 0 . 1 1$ </td><td> $9 2 . 7 8 \pm 0 . 0 6$ </td><td> $8 8 . 2 7 \pm 0 . 1 5$ </td><td> $8 1 . 7 0 \pm 0 . 2 1$ </td><td> $8 2 . 0 6 \pm 0 . 6 8$ </td></tr><tr><td>D Bottom-up</td><td> $9 3 . 4 6 \pm 0 . 0 9$ </td><td> $9 2 . 8 5 \pm 0 . 0 4$ </td><td> $8 8 . 1 9 \pm 0 . 3 5$ </td><td> $8 1 . 4 8 \pm 0 . 3 7$ </td><td> $8 2 . 1 0 \pm 0 . 7 3$ </td></tr><tr><td colspan="6">CTB</td></tr><tr><td>Gold count</td><td> $^ { 3 , 4 8 2 }$ </td><td>3,698</td><td> $^ \mathrm { 1 , 1 2 1 }$ </td><td>184</td><td>235</td></tr><tr><td>C Top-down</td><td> $8 9 . 8 2 \pm 0 . 1 8$ </td><td> $8 7 . 3 3 \pm 0 . 2 3$ </td><td> $8 0 . 2 5 \pm 1 . 3 0$ </td><td> $6 0 . 4 6 \pm 1 . 5 3$ </td><td> $6 8 . 3 1 \pm 2 . 2 8$ </td></tr><tr><td>C In-order</td><td> $9 0 . 4 8 \pm 0 . 1 5$ </td><td> $8 8 . 4 6 \pm 0 . 2 9$ </td><td> $8 1 . 4 0 \pm 1 . 0 4$ </td><td> $6 9 . 6 9 \pm 2 . 3 5$ </td><td> $7 4 . 1 9 \pm 1 . 7 4$ </td></tr><tr><td>C Bottom-up</td><td> $9 0 . 5 3 \pm 0 . 1 2$ </td><td> $8 8 . 4 2 \pm 0 . 6 1$ </td><td> $8 1 . 8 0 \pm 0 . 8 4$ </td><td> $7 0 . 7 7 \pm 1 . 1 8$ </td><td> $7 4 . 4 4 \pm 0 . 4 8$ </td></tr><tr><td>D Top-down</td><td> $8 9 . 6 2 \pm 0 . 5 2 $ </td><td> $8 7 . 6 0 \pm 0 . 3 3$ </td><td> $8 0 . 1 6 \pm 0 . 3 9$ </td><td> $6 2 . 5 5 \pm 2 . 6 6$ </td><td> $6 9 . 0 1 \pm 1 . 2 8$ </td></tr><tr><td>D In-order</td><td> $9 0 . 1 8 \pm 0 . 6 0$ </td><td> $8 8 . 3 4 \pm 0 . 2 1$ </td><td> $8 0 . 6 3 \pm 0 . 6 6$ </td><td> $6 7 . 4 7 \pm 1 . 3 9$ </td><td> $7 2 . 6 8 \pm 1 . 3 7$ </td></tr><tr><td>D Bottom-up</td><td> $9 0 . 6 2 \pm 0 . 2 8$ </td><td> $8 8 . 8 0 \pm 0 . 6 1$ </td><td> $8 2 . 0 8 \pm 0 . 7 6$ </td><td> $6 9 . 1 0 \pm 2 . 3 3 $ </td><td> $7 2 . 8 7 \pm 1 . 5 8$ </td></tr></table>

The canonical bottom-up inventory restricts which label–arity pairs the decoder can predict. Two gold phrasal occurrences on PTB test, an eight-child NX and an eight-child PRN, require label– arity pairs absent from the canonical bottom-up training inventory. No CTB test occurrence has this property. Neither PTB occurrence was recovered by the delimiter-guided bottom-up system in any seed. The development predictions contain examples of successful recovery of unseen pairs. Delimiter-guided bottom-up recovers the 19-child PTB NP and 64-child CTB NP in all three seeds, the 12-child PTB VP in two, and the 15-child CTB QP in none. These examples use the development data also used for checkpoint selection. They illustrate that reusing a known label at a new arity is possible, while accurate recovery still depends on the other construction decisions.

## 6.4 Sentence length and agreement between parsers

We also group the test sentences by token count and score exact label–span matches, with the same tree treatment as in the arity analysis and with child count omitted from the match. Table 5 gives corpus $F _ { 1 }$ within each group for both encodings. The shortest group has higher $F _ { 1 }$ than the longest group for every system. On PTB, the six means span 95.53–96.05 in the shortest group and 91.99–92.46 in the longest. The corresponding CTB ranges are 93.83–94.88 and 85.66–87.84. For sentences of 41 tokens or more, switching the domain encoding changes the mean by at most 0.29 points on PTB and 0.44 points on CTB within a traversal order. The overall association with sentence length is larger than the observed diferences between encodings within this group. Gold derivation workload also varies substantially with length. In the shortest and longest groups, respectively, delimiter-guided bottom-up adds 12.4 and 84.0 oracle actions per sentence on PTB and 11.1 and 118.8 on CTB.

Table 5: Labeled bracket $F _ { 1 }$ by test sentence length in tokens. Counts give the number of test sentences in each group. Results are means ± sample standard deviations across three seeds. C denotes canonical and D delimiter-guided.
<table><tr><td>System</td><td>1-10</td><td>11-20</td><td>21-30</td><td>31-40</td><td>≥ 41</td></tr><tr><td colspan="6">PTB</td></tr><tr><td>Sentences</td><td>270</td><td>764</td><td>778</td><td>433</td><td>171</td></tr><tr><td>C Top-down</td><td> $9 5 . 6 5 \pm 0 . 1 9$ </td><td> $9 5 . 2 1 \pm 0 . 2 0$ </td><td> $9 4 . 3 2 \pm 0 . 1 2$ </td><td> $9 3 . 3 9 \pm 0 . 1 4$ </td><td> $9 2 . 4 1 \pm 0 . 4 3$ </td></tr><tr><td>C In-order</td><td> $9 5 . 9 9 \pm 0 . 2 9$ </td><td> $9 5 . 1 7 \pm 0 . 0 9$ </td><td> $9 4 . 4 9 \pm 0 . 1 2$ </td><td> $9 3 . 5 3 \pm 0 . 1 4$ </td><td> $9 2 . 2 8 \pm 0 . 1 2$ </td></tr><tr><td>C Bottom-up</td><td> $9 5 . 5 3 \pm 0 . 4 7$ </td><td> $9 5 . 0 6 \pm 0 . 1 2$ </td><td> $9 4 . 6 0 \pm 0 . 0 6$ </td><td> $9 3 . 1 9 \pm 0 . 1 6$ </td><td> $9 2 . 3 4 \pm 0 . 3 9$ </td></tr><tr><td>D Top-down</td><td> $9 6 . 0 5 \pm 0 . 0 9$ </td><td> $9 5 . 1 6 \pm 0 . 0 7$ </td><td> $9 4 . 2 1 \pm 0 . 1 5$ </td><td> $9 3 . 4 3 \pm 0 . 2 3$ </td><td> $9 2 . 4 6 \pm 0 . 0 6$ </td></tr><tr><td>D In-order</td><td> $9 5 . 9 4 \pm 0 . 4 8$ </td><td> $9 5 . 1 0 \pm 0 . 0 8$ </td><td> $9 4 . 3 3 \pm 0 . 1 5$ </td><td> $9 3 . 2 8 \pm 0 . 1 4$ </td><td> $9 1 . 9 9 \pm 0 . 0 9$ </td></tr><tr><td>D Bottom-up</td><td> $9 5 . 6 7 \pm 0 . 2 3$ </td><td> $9 5 . 0 7 \pm 0 . 0 9$ </td><td> $9 4 . 3 6 \pm 0 . 1 3$ </td><td> $9 3 . 2 8 \pm 0 . 0 8$ </td><td> $9 2 . 2 4 \pm 0 . 1 1$ </td></tr><tr><td colspan="6">CTB</td></tr><tr><td>Sentences</td><td>96</td><td>81</td><td>84</td><td>38</td><td>49</td></tr><tr><td>C Top-down</td><td> $9 4 . 1 1 \pm 0 . 7 4$ </td><td> $9 1 . 6 4 \pm 0 . 2 5$ </td><td> $9 1 . 5 0 \pm 0 . 7 9$ </td><td> $8 9 . 6 0 \pm 0 . 9 8$ </td><td> $8 5 . 9 9 \pm 1 . 3 2$ </td></tr><tr><td>C In-order</td><td> $9 4 . 8 8 \pm 0 . 8 7$ </td><td> $9 1 . 9 8 \pm 0 . 7 0$ </td><td> $9 2 . 2 1 \pm 0 . 5 0$ </td><td> $9 0 . 4 7 \pm 0 . 7 2$ </td><td> $8 7 . 1 2 \pm 0 . 1 3$ </td></tr><tr><td>C Bottom-up</td><td> $9 3 . 8 3 \pm 0 . 6 5$ </td><td> $9 2 . 5 7 \pm 0 . 5 8$ </td><td> $9 1 . 8 1 \pm 0 . 3 9$ </td><td> $9 0 . 3 4 \pm 0 . 4 7$ </td><td> $8 7 . 4 0 \pm 0 . 4 8$ </td></tr><tr><td>D Top-down</td><td> $9 4 . 2 7 \pm 0 . 4 0$ </td><td> $9 2 . 1 0 \pm 0 . 2 0$ </td><td> $9 1 . 8 4 \pm 0 . 5 9$ </td><td> $8 9 . 7 1 \pm 0 . 4 8$ </td><td> $8 5 . 6 6 \pm 0 . 5 0$ </td></tr><tr><td>D In-order</td><td> $9 4 . 2 5 \pm 0 . 4 4$ </td><td> $9 2 . 2 6 \pm 0 . 5 5$ </td><td> $9 1 . 8 3 \pm 0 . 8 2$ </td><td> $9 0 . 7 2 \pm 1 . 1 6$ </td><td> $8 6 . 8 2 \pm 0 . 2 8$ </td></tr><tr><td> $\mathrm { \Delta D \ B o t t o m { - } u p }$ </td><td> $9 4 . 7 1 \pm 0 . 6 1$ </td><td> $9 2 . 6 1 \pm 0 . 5 0$ </td><td> $9 1 . 9 9 \pm 0 . 3 4$ </td><td> $9 0 . 7 7 \pm 0 . 1 2$ </td><td> $8 7 . 8 4 \pm 1 . 0 6$ </td></tr></table>

Close corpus scores allow substantial disagreement on individual sentences. We pair runs with the same seed within each traversal order and test whether the predicted multisets of labeled brackets coincide. Agreement ranges from 60.00% to 64.67% across orders on PTB and from 50.86% to 53.16% on CTB (Table 6). Under sentence-level $F _ { 1 }$ , both encodings outperform their counterpart on some sentences. For example, on CTB the delimiter-guided bottom-up predictions score higher on an average of 24.04% of sentences, and the canonical predictions score higher on 23.85%.

Table 6: Sentence-level agreement and relative accuracy for matched seed pairs (%, mean ± sample standard deviation). Same requires identical labeled bracket multisets. D higher and C higher count sentences with higher bracket $F _ { 1 }$ for the delimiter-guided and canonical predictions, respectively.
<table><tr><td rowspan="2">Order</td><td colspan="3">PTB</td><td colspan="3">CTB</td></tr><tr><td>Same</td><td>D higher</td><td>C higher</td><td>Same</td><td>D higher</td><td>C higher</td></tr><tr><td>Top-down</td><td> $6 4 . 6 7 \pm 2 . 0 6$ </td><td> $1 6 . 9 0 \pm 1 . 5 2$ </td><td>16.63 ± 0.93</td><td> $5 2 . 7 8 \pm 1 . 3 0$ </td><td> $2 2 . 7 0 \pm 1 . 0 0$ </td><td> $2 3 . 0 8 \pm 0 . 8 8$ </td></tr><tr><td>In-order</td><td> $6 2 . 1 6 \pm 1 . 2 7$ </td><td> $1 7 . 4 0 \pm 0 . 8 7$ </td><td> $1 8 . 5 0 \pm 1 . 5 1$ </td><td> $5 3 . 1 6 \pm 2 . 8 7$ </td><td> $2 0 . 3 1 \pm 2 . 8 8$ </td><td> $2 4 . 8 1 \pm 3 . 0 2$ </td></tr><tr><td>Bottom-up</td><td> $6 0 . 0 0 \pm 1 . 5 3$ </td><td> $1 8 . 6 5 \pm 0 . 5 6$ </td><td> $1 9 . 2 3 \pm 1 . 0 2$ </td><td> $5 0 . 8 6 \pm 1 . 9 9$ </td><td> $2 4 . 0 4 \pm 2 . 9 1$ </td><td> $2 3 . 8 5 \pm 3 . 8 9$ </td></tr></table>

We further count gold bracket occurrences recovered when matching only spans and when requiring labels as well. Across the six systems, 5.35–5.56% of PTB gold occurrences and 8.34–9.76% of CTB gold occurrences lack a matching predicted span after matching with multiplicity. Requiring the label removes a further 0.80–0.86% of gold occurrences on PTB and 1.36–1.59% on CTB. Missed spans therefore account for the larger share of the observed recall deficit in both encodings. These measurements describe completed trees. They leave the transition at which a parser first diverged from its gold derivation unresolved.

## 6.5 Tree-language equivalence and learned decisions

Theorem 3 equates the final tree languages of the formal systems by projecting and lifting successful derivations. The lifting uses the completed tree to choose where to place delimiters. For example, the canonical bottom-up parser can shift the second word of a sentence and leave open whether a later constituent begins there. In the formal delimiter-guided system, a plain Shift at that word rules out a phrase beginning there, because no later action can introduce a delimiter at a boundary the parser has already passed. In-order parsing makes the same early commitment, while top-down parsing introduces the boundary jointly with the mother label. A misplaced delimiter can afect several constituents with the same left edge if successive reductions retain it. The bidirectional encoder supplies sentence context in every order, while the traversal order determines which children are complete when the mother label is chosen. Static supervision trains these decisions on gold configurations.

Across all six systems and three seeds, the audit found zero structural departures in 43,488 PTB parses and 6,264 CTB parses after accounting for top-down marker timing. The relaxed in-order domain and the bottom-up fallback each occurred zero times at marker introduction and reduction. For each of those systems this covers 7,248 PTB parses and 1,044 CTB parses. Delimiter-guided top-down parsing delayed delimiter placement at every phrasal opening. The audit places each delimiter at its corresponding opening action before checking the structural schemas (Appendix C).

Binarization also changes the derivation while allowing the original branching structure to be recovered. It imposes a sequence of local compositions and introduces intermediate training targets (Sagae and Lavie, 2005; Watanabe and Sumita, 2015; Liu and Zhang, 2017b; Qi et al., 2026). The direct systems compose the entire annotated child sequence at each phrase-level reduction. The earlier binary bottom-up experiments of Park et al. (2026) used a diferent model configuration and training schedule. Estimating an efect of binarization against the six present systems would require a binary run with the same protocol.

## 7 Conclusion

This paper identifies the reduction domain as the common structural object of direct non-binary top-down, in-order, and bottom-up parsing. In an applicable phrase-level reduction, the rightmost delimiter identifies the relevant stack segment, the phrasal marker supplies the mother label, and the completed items in that segment supply the ordered children. The reduction arity is therefore the cardinality of an already identified child sequence rather than an independent parameter of the reduction action.

The formal results separate this representational claim from grammar licensing. A stack invariant is preserved throughout every structural derivation and supports soundness and completeness under both the structural and grammar-licensed relations. The recursively specified static oracle has an exact action count of $3 n + 2 m + 1$ . More centrally, the global projection-and-lifting theorem ranges over complete successful derivations of all three traversal orders. It shows that each delimiter-guided system and its canonical counterpart construct the same final trees, and hence generate the same tree language, under both relations. Delimiter guidance and canonical domain encoding thus change the derivation representation without changing its tree-building capacity.

The paired experiments show that the factorization can support accurate greedy parsing under static-oracle training. In bottom-up parsing, deriving arity from the stack reduces the learned action inventory by more than two thirds on both treebanks and increases the number of decisions in a gold derivation.

## Declaration of Competing Interest

The authors declare no competing interests.

## Code availability

The Stanza implementations of all six parsers, along with the notebooks, experiment scripts, and results, are available at https://github.com/jungyeul/generalized-non-binary.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

The authors used OpenAI ChatGPT and Codex to edit the language, reorganize parts of the manuscript, and clarify the technical and mathematical exposition. The authors reviewed and edited the resulting text, verified the formal arguments and implementations, and take full responsibility for the article.

## Acknowledgments

This work was supported by the Institute for Information & Communications Technology Planning & Evaluation (RS-2025-25461932, the Top-Tier AI Global HRD invitation program, and RS-2022- II220078, Explainable Logical Reasoning for Medical Knowledge Generation).

## References

John Bauer and Christopher D. Manning. High-Accuracy Transition-Based Constituency Parsing. In Kenji Sagae and Stephan Oepen, editors, Proceedings of the 18th International Conference on Parsing Technologies (IWPT, SyntaxFest 2025), pages 26–39, Ljubljana, Slovenia, 8 2025. Association for Computational Linguistics. ISBN 979-8-89176-294-7. URL https: //aclanthology.org/2025.iwpt-1.4/.

Ezra Black, Steven P. Abney, Dan Flickinger, Claudia Gdaniec, Ralph Grishman, Phil Harrison, Don Hindle, Robert Ingria, Frederick Jelinek, Judith Klavans, Mark Liberman, Mitch Marcus, Salim Roukos, Beatrice Santorini, and Tomek Strzalkowski. A Procedure for Quantitatively Comparing the Syntactic Coverage of English Grammars. In Proceedings of a Workshop on Speech and Natural Language, pages 306–311, Pacific Grove, California, 2 1991. Morgan Kaufmann. URL https://aclanthology.org/H91-1060/.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota, 6 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1423. URL https://aclanthology.org/N19-1423/.

Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith. Recurrent Neural Network Grammars. In Kevin Knight, Ani Nenkova, and Owen Rambow, editors, Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics:

Human Language Technologies, pages 199–209, San Diego, California, 6 2016. Association for Computational Linguistics. doi: 10.18653/v1/N16-1024. URL https://aclanthology.org/ N16-1024.

Daniel Fernández-González and Carlos Gómez-Rodríguez. Faster shift-reduce constituent parsing with a non-binary, bottom-up strategy. Artificial Intelligence, 275:559–574, 2019. ISSN 0004- 3702. doi: https://doi.org/10.1016/j.artint.2019.07.006. URL https://www.sciencedirect. com/science/article/pii/S000437021830540X.

Eunkyul Leah Jo, Angela Yoonseo Park, and Jungyeul Park. A Novel Alignment-based Approach for PARSEVAL Measures. Computational Linguistics, 50(3):1181–1190, 9 2024. URL https: //aclanthology.org/2024.cl-3.10.

Jiangming Liu and Yue Zhang. In-Order Transition-based Constituent Parsing. Transactions of the Association for Computational Linguistics, 5:413–424, 2017a. doi: 10.1162/tacl{\_}a{\_}00070. URL https://aclanthology.org/Q17-1029.

Jiangming Liu and Yue Zhang. Shift-Reduce Constituent Parsing with Neural Lookahead Features. Transactions of the Association for Computational Linguistics, 5:45–58, 2017b. ISSN 2307-387X. URL https://transacl.org/ojs/index.php/tacl/article/view/927.

Mitchell P. Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. Building a Large Annotated Corpus of English: The Penn Treebank. Computational linguistics, 19(2):313–330, 1993. URL https://aclanthology.org/J93-2004.

Jungyeul Park, Junrui Wang, Eunkyul Jo, and Angela Park. jp-evalb: Robust Alignment-based PARSEVAL Measures. In Kai-Wei Chang, Annie Lee, and Nazneen Rajani, editors, Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 3: System Demonstrations), pages 70–77, Mexico City, Mexico, 6 2024. Association for Computational Linguistics. URL https://aclanthology.org/2024.naacl-demo.7.

Jungyeul Park, Eunkyul Leah Jo, and Zihao Huang. Non-binary bottom-up constituency parsing without arity actions. In Hung-Yi Lee, Sunipa Dev, Jessy Li, and Desmond Elliott, editors, Findings of the Association for Computational Linguistics: EMNLP 2026, pages 12869–12879, Budapest, Hungary, 10 2026. Association for Computational Linguistics. ISBN 979-8-89176-462- 0. URL https://aclanthology.org/2026.findings-emnlp.611/.

Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton, and Christopher D. Manning. Stanza: A Python Natural Language Processing Toolkit for Many Human Languages. In Asli Celikyilmaz and Tsung-Hsien Wen, editors, Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pages 101–108, Online, 7 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-demos.14. URL https://aclanthology.org/2020.acl-demos.14.

Zeyao Qi, Yige Chen, Eitan Klinger, Vivaan Wadhwa, and Jungyeul Park. Better heads do not guarantee better binarized constituency parsing, 2026. URL https://arxiv.org/abs/2605. 28131.

Kenji Sagae and Alon Lavie. A Classifier-Based Parser with Linear Run-Time Complexity. In Proceedings of the Ninth International Workshop on Parsing Technology (IWPT2005), pages 125–132, Vancouver, British Columbia, 2005. Association for Computational Linguistics. URL http://www.aclweb.org/anthology/W/W05/W05-1513.

Ann Taylor, Mitchell Marcus, and Beatrice Santorini. The Penn Treebank: An Overview. In Anne Abeillé, editor, Treebanks: Building and Using Parsed Corpora, pages 5–22. Springer Netherlands, Dordrecht, 2003. ISBN 978-94-010-0201-1. doi: 10.1007/978-94-010-0201-1{\_}1. URL https: //doi.org/10.1007/978-94-010-0201-1\_1.

Mengqiu Wang, Kenji Sagae, and Teruko Mitamura. A Fast, Accurate Deterministic Parser for Chinese. In Nicoletta Calzolari, Claire Cardie, and Pierre Isabelle, editors, Proceedings of the 21st International Conference on Computational Linguistics and 44th Annual Meeting of the Association for Computational Linguistics, pages 425–432, Sydney, Australia, 7 2006. Association

for Computational Linguistics. doi: 10.3115/1220175.1220229. URL https://aclanthology.   
org/P06-1054.

Taro Watanabe and Eiichiro Sumita. Transition-based Neural Constituent Parsing. In Chengqing Zong and Michael Strube, editors, Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1169–1179, Beijing, China, 7 2015. Association for Computational Linguistics. doi: 10.3115/v1/P15-1113. URL https://aclanthology.org/P15-1113.

Nianwen Xue, Fei Xia, Fu-dong Chiou, and Marta Palmer. The Penn Chinese TreeBank: Phrase Structure Annotation of a Large Corpus. Natural Language Engineering, 11(2):207–238, 6 2005. ISSN 1351-3249. doi: 10.1017/S135132490400364X. URL https://doi.org/10.1017/ S135132490400364X.

Songlin Yang and Kewei Tu. Bottom-Up Constituency Parsing and Nested Named Entity Recognition with Pointer Networks. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2403–2416, Dublin, Ireland, 5 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.171. URL https://aclanthology.org/ 2022.acl-long.171.

Yue Zhang and Stephen Clark. Transition-Based Parsing of the Chinese Treebank using a Global Discriminative Model. In Harry Bunt and Eric de La Clergerie, editors, Proceedings of the 11th International Conference on Parsing Technologies (IWPT’09), pages 162–171, Paris, France, 10 2009. Association for Computational Linguistics. URL https://aclanthology.org/W09-3825.

Muhua Zhu, Yue Zhang, Wenliang Chen, Min Zhang, and Jingbo Zhu. Fast and Accurate Shift-Reduce Constituent Parsing. In Hinrich Schuetze, Pascale Fung, and Massimo Poesio, editors, Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 434–443, Sofia, Bulgaria, 8 2013. Association for Computational Linguistics. URL https://aclanthology.org/P13-1043.

## A Canonical transition systems

For each traversal order $\tau \in \{ \mathrm { T D } , \mathrm { I O } , \mathrm { B U } \}$ , the canonical system $\mathcal { C } _ { \tau }$ has configurations $c = \langle \sigma , i , f \rangle$ ， where $\sigma$ is a stack, $i \in \{ 1 , \ldots , n + 1 \}$ is the next-input index, and $f \in \{ 0 , 1 \}$ is the completion flag. The initial configuration is $\langle \epsilon , 1 , 0 \rangle$ . A successful final configuration has the form $\langle T , n + 1 , 1 \rangle$ ， where $T$ is a completed tree rooted in S with yield $w _ { 1 } \cdots w _ { n }$ . Top-down and in-order stacks may contain markers $\operatorname { N T } ( L )$ , bare terminals, and completed tree items; bottom-up stacks may contain bare terminals and completed tree items.

Each action below is a partial function on configurations, and their union defines the canonical structural relation $\Rightarrow _ { \tau } ^ { \mathcal { C } } .$ . In every schema, σ denotes an arbitrary stack prefix and the displayed elements constitute its complete sufix. The variables $P \in \mathcal { P }$ and $X \in \mathcal { N }$ distinguish preterminal and phrasal instances of the unified label variable $L \in { \mathcal { P } } \cup { \mathcal { N } } .$ A bare terminal can be reduced only to a preterminal, whereas a nonempty sequence of completed tree items can be reduced only to a phrasal constituent. Every action requires $f = 0$ , and no action is applicable after $f = 1$

An unfinished preterminal construction must be completed before any other transition is taken. In the top-down system, a stack-final $\mathrm { N T } ( P )$ admits only Shift, and the resulting sufix $\mathrm { N T } ( P ) \mid$ $w _ { i - 1 }$ admits only Reduce. In the in-order system, a stack-final bare terminal $w _ { i - 1 }$ admits only $\operatorname { N T } ( P )$ , and the resulting sufix $w _ { i - 1 } \mid \operatorname { N T } ( P )$ admits only Reduce. In the bottom-up system, a stack-final bare terminal $w _ { i - 1 }$ admits only Reduce- $P \# 1$

Top-down The canonical top-down action inventory consists of $\mathrm { { N T } } ( L )$ , Shift, Reduce, and Finish. The expressions $\operatorname { N T } ( P )$ and $\operatorname { N T } ( X )$ below are typed instances of the single action family

Nt(L), and both reduction schemas are instances of the single action Reduce:

$$
\begin{array} { r l } { \mathrm { N T } ( P ) : } & { { } \quad \langle \sigma , i , 0 \rangle } \end{array}
$$

$$
\begin{array} { r l } { \Rightarrow _ { \mathrm { T D } } ^ { \mathcal { C } } } & { { } \langle \sigma \mid \mathrm { N T } ( P ) , i , 0 \rangle , \qquad i \leq n , } \end{array}
$$

$$
\operatorname { N T } ( X ) : \qquad \langle \sigma , i , 0 \rangle
$$

$$
\begin{array} { r l r l } { \Rightarrow _ { \mathrm { T D } } ^ { \mathcal { C } } } & { { } \langle \sigma \mid \mathrm { N T } ( X ) , i , 0 \rangle , } & { } & { { } i \leq n , } \end{array}
$$

$$
\begin{array} { r l r } { \mathrm { S H I F T } : } & { { } } & { \langle \sigma \mid \mathrm { N T } ( P ) , i , 0 \rangle } \end{array}
$$

$$
\begin{array} { r l } { \Rightarrow _ { \mathrm { T D } } ^ { \mathcal { C } } } & { { } \langle \sigma \mid \mathrm { N T } ( P ) \mid w _ { i } , i + 1 , 0 \rangle , \qquad i \leq n , } \end{array}
$$

$$
\begin{array} { r l } { \operatorname { R E D U C E } : } & { { } \langle \sigma \mid \operatorname { N T } ( P ) \mid w _ { i - 1 } , i , 0 \rangle } \end{array}
$$

$$
\begin{array} { r l } { \Rightarrow _ { \mathrm { T D } } ^ { \mathcal { C } } } & { { } \langle \sigma \mid P ( w _ { i - 1 } ) , i , 0 \rangle , \qquad 2 \leq i \leq n + 1 , } \end{array}
$$

$$
\begin{array} { r } { \mathrm { R E D U C E : } \quad \langle \sigma \mid \operatorname { N T } ( X ) \mid \alpha _ { 1 } \mid \cdot \cdot \cdot \mid \alpha _ { k } , i , 0 \rangle \quad \Rightarrow _ { \mathrm { T D } } ^ { \mathcal { C } } \quad \langle \sigma \mid X ( \alpha _ { 1 } , \ldots , \alpha _ { k } ) , i , 0 \rangle , \qquad k \geq 1 . } \end{array}
$$

In a phrase-level Reduce, the displayed nt(X) is the topmost unmatched marker, and $\alpha _ { 1 } , \ldots , \alpha _ { k }$ are completed tree items with nonempty consecutive yields in left-to-right order. The marker supplies the mother label, and the complete sufix above it supplies the ordered child sequence.

In-order The canonical in-order action inventory likewise consists of Shift, Nt(L), Reduce, and Finish. Its typed transition schemas are

$$
\begin{array} { r l r } { \mathrm { S H I F T } : } & { { } } & { \left. \sigma , i , 0 \right. } \end{array}
$$

$$
\Rightarrow _ { \mathrm { I O } } ^ { \mathcal { C } } \quad \langle \sigma \mid w _ { i } , i + 1 , 0 \rangle , \qquad i \leq n ,
$$

$$
\begin{array} { r l } { \mathrm { N T } ( P ) : } & { { } \langle \sigma \mid w _ { i - 1 } , i , 0 \rangle } \end{array}
$$

$$
\begin{array} { r l } { \Rightarrow _ { \mathrm { I O } } ^ { \mathcal { C } } } & { { } \langle \sigma \mid w _ { i - 1 } \mid \mathrm { N T } ( P ) , i , 0 \rangle , \qquad 2 \leq i \leq n + 1 , } \end{array}
$$

$$
\begin{array} { r l } { \mathrm { R E D U C E : } } & { { } \langle \sigma \mid w _ { i - 1 } \mid \mathrm { N T } ( P ) , i , 0 \rangle } \end{array}
$$

$$
\begin{array} { r l } { \Rightarrow _ { \mathrm { I O } } ^ { \mathcal { C } } } & { { } \langle \sigma \mid P ( w _ { i - 1 } ) , i , 0 \rangle , \qquad 2 \leq i \leq n + 1 , } \end{array}
$$

$$
\mathrm { N T } ( X ) : \qquad \langle \sigma \mid \alpha _ { 1 } , i , 0 \rangle
$$

$$
\begin{array} { r l } { \Rightarrow _ { \mathrm { I O } } ^ { \mathcal { C } } } & { { } \langle \sigma \mid \alpha _ { 1 } \mid \mathrm { N T } ( X ) , i , 0 \rangle , } \end{array}
$$

$$
\begin{array} { r } { \mathrm { R E D U C E : } \quad \langle \sigma \mid \alpha _ { 1 } \mid \operatorname { N T } ( X ) \mid \alpha _ { 2 } \mid \cdot \cdot \cdot \mid \alpha _ { k } , i , 0 \rangle \quad \Rightarrow _ { \mathrm { I O } } ^ { c } \quad \langle \sigma \mid X ( \alpha _ { 1 } , . . . , \alpha _ { k } ) , i , 0 \rangle , \qquad k \geq 1 . } \end{array}
$$

In a phrase-level Reduce, the displayed marker is the topmost unmatched marker, and $\alpha _ { 1 } , \ldots , \alpha _ { k }$ are completed tree items with nonempty consecutive yields in left-to-right order. The sequence $\alpha _ { 2 } , \ldots , \alpha _ { k }$ is empty when k = 1, so unary phrasal constituents remain distinct from preterminal constructions by the type of their sole child.

Arity-specific bottom-up The canonical arity-specific bottom-up action inventory consists of Shift, Reduce-L#k, and Finish. The preterminal and phrasal schemas are typed instances of

the single arity-specific reduction family:

$$
\begin{array} { l l l l } { { \mathrm { S H I F T : } } } & { { \langle \sigma , i , 0 \rangle } } & { { \Rightarrow _ { \mathrm { B U } } ^ { \mathcal { C } } } } & { { \langle \sigma \mid w _ { i } , i + 1 , 0 \rangle , } } & { { i \leq n , } } \\ { { \mathrm { R E D U C E - } P \# 1 : } } & { { \langle \sigma \mid w _ { i - 1 } , i , 0 \rangle } } & { { \Rightarrow _ { \mathrm { B U } } ^ { \mathcal { C } } } } & { { \langle \sigma \mid P ( w _ { i - 1 } ) , i , 0 \rangle , } } & { { 2 \leq i \leq n + 1 , } } \\ { { \mathrm { R E D U C E - } X \# k : } } & { { \langle \sigma \mid \alpha _ { 1 } \mid \cdots \mid \alpha _ { k } , i , 0 \rangle } } & { { \Rightarrow _ { \mathrm { B U } } ^ { \mathcal { C } } } } & { { \langle \sigma \mid X ( \alpha _ { 1 } , \ldots , \alpha _ { k } ) , i , 0 \rangle , } } & { { k \geq 1 . } } \end{array}
$$

In the phrase-level schema, $\alpha _ { 1 } , \ldots , \alpha _ { k }$ are completed tree items with nonempty consecutive yields in left-to-right order. The parameter k selects the complete top sufix and records its arity. In particular, Reduce-P #1 applies only to a bare terminal, whereas Reduce-X#1 applies only to a completed tree item.

Completion and grammar licensing For every traversal order, Finish applies exactly when all input has been consumed and the stack contains one completed tree rooted in S:

$$
\begin{array} { r } { \mathrm { F I N I S H : } \quad \langle T , n + 1 , 0 \rangle \quad \Rightarrow _ { \tau } ^ { \mathcal { C } } \quad \langle T , n + 1 , 1 \rangle , \qquad \mathrm { r o o t } ( T ) = S . } \end{array}
$$

The canonical grammar-licensed relation $\Rightarrow _ { \tau , G } ^ { \mathcal { C } } \subseteq \Rightarrow _ { \tau } ^ { \mathcal { C } }$ requires every constructed constituent to instantiate a production in R. A preterminal reduction constructing $P ( w _ { i - 1 } )$ requires $P  w _ { i - 1 } \in$ R. A phrase-level reduction constructing $X ( \alpha _ { 1 } , \ldots , \alpha _ { k } )$ requires $X \to Y _ { 1 } \cdot \cdot \cdot Y _ { k } \in R$ , where $Y _ { j }$ is the root label of $\alpha _ { j }$ . All nonconstructive transitions coincide under the structural and grammar-licensed relations.

## B Illustrative oracle derivations

Figure 2 gives the target tree for Ms. Haag plays Elianti., with T OP treated as a phrasal nonterminal. The tree has five terminals, five preterminals, and five phrasal nodes, with phrase-level arities one, two, and three.

Table 7 gives the static oracle sequence for each traversal order. Every sequence contains 26 actions, including Finish, and is a successful derivation of the tree in Figure 2 under ⇒<sub>τ</sub>. If the displayed tree is licensed by $G ,$ the same sequences are derivations under $\Rightarrow _ { \tau , G }$ . In each phrase-level construction, the marker follows exactly $\mathrm { O r d e r } _ { \tau } ( k )$ completed children. In the top-down column, each phrasal $\operatorname { N T } ( X )$ is a single action whose stack efect introduces $D \mid \operatorname { N T } ( X )$ ; hence a delimiter is present even though no d-Shift action appears. Preterminal $\operatorname { N T } ( P )$ introduces only $\operatorname { N T } ( P )$ and no delimiter.

![](images/800b7ad1dc9debcca50cbe6538f182a4f6dfb4d4b712d521b9be94b94a70aa98.jpg)  
Figure 2: Target constituency tree for Ms. Haag plays Elianti.

For the in-order and bottom-up sequences, the boundary targets are

$$
b ( M s . ) = T O P , \qquad b ( p l a y s ) = V P , \qquad b ( E l i a n t i ) = N P _ { 2 } ,
$$

while $b ( H a a g )$ and the boundary target of the final punctuation are undefined. Hence, both sequences introduce delimiters before $M s . , p l a y s ,$ and Elianti. The delimiter before Ms. is preserved when $N P _ { 1 }$ and $S$ are constructed and consumed when its boundary target $T O P$ is constructed. The other two delimiters are consumed when VP and $N P _ { 2 }$ , respectively, are constructed.

Immediately before constructing the ternary $S ,$ the three reducible sufixes are

$$
\mathrm { T D } : \ D \mid \mathrm { N T } ( S ) \mid N P _ { 1 } \mid V P \mid .
$$

$$
\mathrm { I O } : \quad D \mid N P _ { 1 } \mid \mathrm { N T } ( S ) \mid V P \mid .
$$

$$
\mathrm { B U } : \textit { D } | \mathit { N P } _ { 1 } | \mathit { V P } | . | \mathrm { N T } ( \boldsymbol { S } )
$$

Table 7: Delimiter-guided static-oracle sequences for the tree in Figure 2. Tokens shown with Shift and d-Shift identify the consumed input token; they are not action parameters. The reduction actions have neither label nor arity parameters.
<table><tr><td>Step</td><td>Top-down</td><td>In-order</td><td>Bottom-up</td></tr><tr><td>1</td><td>NT(TOP)</td><td>D-SHIFT(Ms.)</td><td>D-SHIFT(Ms.)</td></tr><tr><td>2</td><td>NT(S)</td><td>NT(NNP)</td><td>NT(NNP)</td></tr><tr><td>3</td><td>NT(NP1)</td><td>U-REDUCE</td><td>U-REDUCE</td></tr><tr><td>4</td><td>NT(NNP)</td><td>NT(NP1)</td><td>SHIFT(Haag)</td></tr><tr><td>5</td><td>SHIFT(Ms.)</td><td>SHIFT(Haag)</td><td>NT(NNP)</td></tr><tr><td>6</td><td>U-REDUCE</td><td>NT(NNP)</td><td>U-REDUCE</td></tr><tr><td>7</td><td>NT(NNP)</td><td>U-REDUCE</td><td> $\mathrm { { N r } ( \mathrm { { N P } _ { 1 } ) } }$ </td></tr><tr><td>8</td><td>SHIFT(Haag)</td><td>D-REDUCE</td><td>D-REDUCE</td></tr><tr><td>9</td><td>U-REDUCE</td><td>NT(S)</td><td>D-SHIFT(plays)</td></tr><tr><td>10</td><td>REDUCE</td><td>D-SHIFT(plays)</td><td>NT(VBZ)</td></tr><tr><td>11</td><td>NT(VP)</td><td>NT(VBZ)</td><td>U-REDUCE</td></tr><tr><td>12</td><td>NT(VBZ)</td><td>U-REDUCE</td><td>D-SHIFT(Elianti)</td></tr><tr><td>13</td><td>SHIFT(plays)</td><td>NT(VP)</td><td>NT(NNP)</td></tr><tr><td>14</td><td>U-REDUCE</td><td>D-SHIFT(Elianti)</td><td>U-REDUCE</td></tr><tr><td>15</td><td>NT(NP2)</td><td>NT(NNP)</td><td>NT(NP2)</td></tr><tr><td>16</td><td>NT(NNP)</td><td>U-REDUCE</td><td>REDUCE</td></tr><tr><td>17</td><td>SHIFT(Elianti)</td><td>NT(NP2)</td><td>NT(VP)</td></tr><tr><td>18</td><td>U-REDUCE</td><td>REDUCE</td><td>REDUCE</td></tr><tr><td>19</td><td>REDUCE</td><td>REDUCE</td><td>SHIFT(.)</td></tr><tr><td>20</td><td>REDUCE</td><td>SHIFT(.)</td><td>NT(·)</td></tr><tr><td>21</td><td> $\operatorname { N T } ( . )$ </td><td> $\mathrm { { N T } } ( . )$ </td><td>U-REDUCE</td></tr><tr><td>22</td><td> $\operatorname { S H I F T } ( . )$ </td><td>U-REDUCE</td><td>NT(S)</td></tr><tr><td>23</td><td>U-REDUCE</td><td>D-REDUCE</td><td>D-REDUCE</td></tr><tr><td>24</td><td>REDUCE</td><td>NT(TOP)</td><td>NT(TOP)</td></tr><tr><td>25</td><td>REDUCE</td><td>REDUCE</td><td>REDUCE</td></tr><tr><td>26</td><td>FINISH</td><td>FINISH</td><td>FINISH</td></tr></table>

Deleting the delimiter and marker yields the same ordered reduction domain

$$
\langle N P _ { 1 } , V P , . \rangle
$$

in all three cases, and hence the same recovered arity k = 3. The top-down sequence constructs S with Reduce; the in-order and bottom-up sequences use d-Reduce because the active delimiter must remain available for the subsequent construction of T OP.

## C Experimental details

Data and preprocessing The PTB split assigns WSJ sections 02–21 to training, section 22 to development, and section 23 to testing. The prepared CTB 5.1 splits follow the basic partition, which assigns documents 001–270 and 440–1151 to training, 301–325 to development, and 271–300 to testing. The runners consume prepared split files. They prune null elements, discard any tree made empty by pruning, simplify functional and coindexation labels, and standardize the outer label to ROOT with one child. The resulting train, development, and test files contain 39,832, 1,700, and 2,416 trees for PTB and 17,544, 352, and 348 for CTB. The trainer removes duplicate trees from the training and development files, giving 39,514 and 1,696 trees for PTB and 16,676 and 326 for CTB. The prepared files for each split have identical content hashes across systems and seeds. The test runs score all 2,416 PTB trees and all 348 CTB trees.

Encoder and action scorer The experiments use a modified Stanza 1.14.0 parser with Python 3.10.12, PyTorch 2.14.1+cu130 and Transformers 4.57.6. The word encoder concatenates pretrained word vectors, trainable word and POS embeddings, forward and backward character language model features, and a learned linear mixture of the final four hidden layers of frozen BERT (Devlin et al., 2019). PTB uses bert-large-cased, Stanza’s conll17 vectors, and the 1billion character models. CTB uses bert-base-chinese, fasttext157 vectors, and the gigaword character models. A two-layer bidirectional LSTM encodes the words with 256 units per direction. The constituent and transition stacks use two-layer LSTMs with hidden sizes 256 and 20, respectively. Transition and POS embeddings have 20 dimensions, and the trainable word embeddings have 100. Constituent composition takes a coordinatewise maximum over child vectors, followed by a learned transformation. Two afine layers with a ReLU hidden layer of width 256 score the actions. Word, prediction, and LSTM input dropout are 0.2. The models use no additional attention layers.

Training and model selection Before oracle construction, Stanza’s default POS pipelines assign XPOS tags to the supplied tokens (combined\_charlm for English and gsdsimp\_charlm for Chinese). These tags serve as encoder features and as preterminal labels in the training trees. The transition sequences shift bare words and predict the preterminal label through the respective construction actions. Training minimizes the sum of action cross-entropies over each minibatch, with the softmax taken over the learned action inventory. Each of the 200 training iterations takes the next 5,000 trees from shufled passes through the training set, carrying unused trees forward between iterations.

Training batches contain 16 trees and evaluation batches contain 32. We use AdamW with initial learning rate $0 . 0 0 1 , \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ , and weight decay 0.05 (zero for LSTM biases) in a single training stage. A development-score plateau scheduler multiplies the learning rate by 0.6 after a patience of five iterations, with a ten-iteration cooldown and a floor of $2 \times 1 0 ^ { - 5 }$ . We retain the earliest checkpoint attaining the highest native labeled development $F _ { 1 }$ , computed with CoreNLP 4.6.0 on the filtered, retagged development trees. The canonical top-down PTB runs resumed from optimizer and scheduler checkpoints after 114, 132, and 132 iterations for seeds 1111, 2222, and 3333, respectively. The sampler and random-number generators were reinitialized on resume. All three runs continued through iteration 200.

Implementation checks at decoding The learned policy scores the observed training action classes and uses order-specific legality checks, including root restrictions and safeguards against unbounded action sequences. The unary limits derived from training are 4 for PTB and 6 for CTB. Both top-down implementations resolve a pending marker’s role from the next executed action. A shift identifies a preterminal marker, and a child marker identifies a phrasal marker. In the delimiter-guided implementation, the latter action also materializes the parent’s delimiter. The delimiter-guided in-order checks allow a phrasal marker with multiple completed items to its left in the active segment. The delimiter-guided bottom-up checks allow the entire completed stack as the reduction domain when no delimiter is active. All reported test predictions cover the full reference token sequences.

When no delimiter is active, the in-order decoder allows d-Shift only from an empty stack. The bottom-up decoder disallows d-Shift while a marker is at the top of the stack. In both orders, d-Reduce requires an active delimiter and a nonroot label. A unary d-Reduce is subject to the decoder’s unary guard. When the delimiter is at the bottom of the in-order stack, only a root reduction may consume it.

Audit of executed transitions We rerun greedy decoding for the 36 test checkpoints on an NVIDIA A100 with length-sorted inputs and the original evaluation batch size of 32. We record the selected actions and replay them against the structural schemas in Section 3 and Appendix A. For top-down traces, we assign each marker its resolved type at its opening action and, in the delimiterguided variant, place the delimiter at the same action. This moves 139,023 PTB delimiters and 26,917 CTB delimiters one action earlier in the recorded histories. For delimiter-guided in-order and bottom-up traces, we check for the respective relaxed domain at both marker introduction and reduction. The symbolic replay reconstructs every decoded tree from a total of 5,177,168 actions. All 49,752 decoded trees match the archived predictions after reference POS tags are restored.

Test scoring We restore the reference POS labels after decoding and retain the saved ROOT wrappers for both scorers. We run evalb with the COLLINS.prm file distributed with jp-evalb, applying the same label exclusions and equivalences to PTB and CTB. We run jp-evalb in its default alignment mode, which retains punctuation. The scoring records identify the exact revisions of both scorers. For each seed, we compute corpus-level $F _ { 1 }$ from the matched, predicted, and gold bracket totals. We round the mean and sample standard deviation only after aggregating across seeds.

Data provenance The Linguistic Data Consortium distributes both corpora under license. We use Penn Treebank 3 (LDC99T42) and the Chinese Treebank 5.1 update to LDC2005T01.