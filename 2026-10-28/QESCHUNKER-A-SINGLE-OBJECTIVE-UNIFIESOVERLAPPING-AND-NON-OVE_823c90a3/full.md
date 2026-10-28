# QESCHUNKER: A SINGLE OBJECTIVE UNIFIESOVERLAPPING AND NON-OVERLAPPING CHUNKINGFOR RAG

Yifan Zhao<sup>1</sup>, Qianyang Zhao<sup>1</sup>, Xinyu Guan<sup>1</sup>, kai wei<sup>2</sup>, Yuming Deng<sup>1</sup>

<sup>1</sup>Alibaba Group (alibaba-inc.com)

<sup>2</sup>Taobao (China) Software Co., Ltd. (taobao.com)

## ABSTRACT

Chunking is a fundamental preprocessing step in retrieval-augmented generation (RAG), yet most existing chunkers rely primarily on heuristics. We introduce the Question–Evidence-Supervised Chunker (QESChunker), to our knowledge the first method for constructing static chunk indices from question–evidence supervision. QESChunker constructs document-specific chunk indices by optimizing over human-annotated or LLM-generated question–evidence pairs, without access to held-out test queries. We formulate non-overlapping and overlapping index construction through a unified question–evidence utility-maximization objective, turning chunking into a supervised combinatorial optimization problem. Under the non-overlap constraint, the objective decomposes over chunks and can be optimized exactly in polynomial time by dynamic programming. When overlap is allowed, the objective becomes monotone submodular; exact optimization is <sub>NP-hard,</sub> <sub>but</sub> <sub>our</sub> <sub>greedy</sub> <sub>achieves</sub> <sub>a</sub> <sub>(1</sub> <sub>−</sub> <sub>1/</sub>√<sub>e)-approximation.</sub> <sub>Across</sub> <sub>two</sub> <sub>self-</sub> constructed and three public datasets, both variants outperform existing chunking methods in complete-evidence recall in nearly all evaluated settings, under the same overlap regime and a fixed hybrid retriever. These results recast chunking as supervised, budget-aware index construction for evidence retrieval.

## 1 INTRODUCTION

Retrieval-augmented generation (RAG) conditions a language model on passages retrieved from an external corpus (Lewis et al., 2020; Gao et al., 2023). Before any retrieval can occur, each document must be segmented into chunks that are embedded and indexed. Where the chunk boundaries fall largely determines whether the evidence for a question can be retrieved completely.

Despite its importance, chunking is overwhelmingly heuristic in practice. The baselines evaluated in this paper span fixed-length and fixed-overlap passages used by retrieval systems (Lewis et al., 2020; Guu et al., 2020; Karpukhin et al., 2020), natural paragraphs, lexical-cohesion segmentation with TextTiling (Hearst, 1997), embedding-based ClusterSemanticChunker (Smith & Troynikov, 2024), LLM-predicted boundaries with LumberChunker (Duarte et al., 2024), sentence-window retrieval (LlamaIndex, 2024b), and hierarchical Parent-Document and Auto-Merging retrieval (LlamaIndex, 2024a). These practical alternatives cover non-overlapping, overlapping, semantic, LLMbased, and hierarchical designs, but none optimizes an explicit question–evidence objective.

We cast chunking as an explicit optimization problem and show that a single objective, subjected to two different structural constraints, yields two algorithms with complementary guarantees. We call the resulting framework QESCHUNKER (Question–Evidence Supervised Chunker): questions and their evidence blocks supervise where boundaries should fall, and the same objective is then solved exactly or near-optimally under each constraint. Figure 1 summarizes the framework.

Our contributions are:

1. A unified objective. We model a document as a sequence of atomic blocks and define a two-term objective comprising a coverage term and a structural term (§3).

2. Two algorithms under different constraints. Under a non-overlap constraint—that is, a contiguous partition into multiple chunks—the objective becomes additive over chunks, and an exact dynamic program returns the global optimum in polynomial time (§4). Under an overlap constraint, an index budget over a candidate chunk family must be satisfied. We prove that exact optimization of the objective is NP-hard by a reduction from Grid Segment Covering (§5), and give a benefit-cost greedy algorithm that attains a $( 1 - 1 / { \sqrt { e } } ) \approx { \bar { 0 } } . 3 9 3$ guarantee for budgeted monotone submodular maximization under the corrected analysis of Tang et al. (2021).

![](images/10056f5639fc8c28248068cb586a6b3f6cf5e14dfee8eaf5860235d88176eb36.jpg)  
Figure 1: Overview of QESCHUNKER. A shared question–evidence objective supports exact dynamic programming for non-overlapping partitions and benefit–cost greedy selection for overlapping indices.

3. Broad wins on multiple datasets. Across two self-constructed Chinese policy corpora and three public English corpora (QASPER, PeerQA, and PrivacyQA), under a fixed hybrid retriever, our non-overlap DP ranks first in 9/10 corpus×tier display cells and our overlap method in 10/10 (Tables 3 and 4).

## 2 RELATED WORK

Chunking and adjacent segmentation. Our evaluated baselines are fixed-length, naturalparagraph, TextTiling, ClusterSemanticChunker, LumberChunker, fixed-window, Sentence-Window, Parent-Document, and Auto-Merging. Classical topic and discourse segmenters such as C99 and supervised neural boundary models optimize segmentation labels rather than RAG evidence recall (Choi, 2000; Koshorek et al., 2018; Lukasik et al., 2020). Other adjacent systems change more than chunk boundaries: Dense X replaces passages with generated propositions, Late Chunking changes contextual embedding computation, and RAPTOR introduces recursive summaries and hierarchical retrieval (Chen et al., 2024; Gunther et al., 2024; Sarthi et al., 2024). We therefore¨ exclude them from matched comparisons that isolate boundary selection under a fixed retriever and original-atom budget.

Submodular optimization. Our overlap solver draws on classical monotone-submodular maximization (Nemhauser et al., 1978; Krause & Golovin, 2014); the corresponding non-overlapping partition instead admits an exact dynamic program.

## 3 PROBLEM SETUP AND UNIFIED OBJECTIVE

## 3.1 ATOMIC BLOCKS, EVIDENCE, AND HITS

A document is atomized into a sequence $u _ { 1 } , \ldots , u _ { n }$ of atomic blocks by splitting on minimal punctuation; blocks are the common accounting unit for both indexing cost and returned context. Let $B \subseteq [ n ]$ be the set of block indices that terminate natural paragraphs. We are given m fitting questions. For question j:

• $R _ { j }$ is the set of gold evidence atomic blocks;

$H _ { j }$ is the hit set: the top-κ atomic blocks most similar to the question embedding, i.e., the blocks that a first-stage retriever would actually surface.

Let $S$ denote a set of chunks (either a partition or an overlapping set). A chunk $C = [ a , b ]$ covers blocks $a , \ldots , b$

## 3.2 HIT-GATED WEIGHTED COVERAGE WITH A TIERED STRUCTURAL PRIOR

Treat each pair “question $j ,$ evidence block $r ^ { \ast }$ as an element to be covered, with weight

$$
w ( j , r ) = \frac { 1 } { m | R _ { j } | } , \qquad \sum _ { j } \sum _ { r \in R _ { j } } w ( j , r ) = 1 .\tag{1}
$$

A single chunk C covers the element set

$$
\Gamma ( C ) = \{ ( j , r ) : \ r \in C \cap R _ { j } , \ C \cap H _ { j } \neq \emptyset \} .\tag{2}
$$

The hit gate $C \cap H _ { j } \neq \emptyset$ encodes retrieval reality: a chunk that contains none of question $j ^ { \dagger } \mathbf { s }$ hit blocks will not be surfaced for $j ,$ so it cannot contribute coverage for $j$ no matter how much gold text it holds.

The unified objective adds a tiered structural prior over chunk right-endpoints $b _ { C } { \mathrm { : } }$

$$
\boxed { f ( S ) = \sum _ { ( j , r ) \in \bigcup _ { C \in S } \Gamma ( C ) \atop \mathrm { c o v e r a g e t e r m } } w ( j , r ) + \underbrace { \sum _ { C \in S } \lambda ( b _ { C } ) } _ { \mathrm { s t r u c t u r a l t e r m } } }\tag{3}
$$

where a chunk’s structural term is determined by the terminal punctuation of its right block:

$$
\lambda ( b _ { C } ) = \lambda \times \left\{ \begin{array} { l l } { \rho _ { 1 } , } & { b _ { C } \mathrm { ~ e n d s ~ a ~ p a r a g r a p h } , } \\ { \rho _ { 2 } , } & { b _ { C } \mathrm { ~ e n d s ~ a ~ s e n t e n c e ~ b u t ~ n o t ~ a ~ p a r a g r a p h } , } \\ { \rho _ { 3 } , } & { b _ { C } \mathrm { ~ e n d s ~ a ~ c l a u s e ~ ( s e m i c o l o n ~ o r ~ c o l o n ) } , } \\ { \rho _ { 4 } , } & { b _ { C } \mathrm { ~ e n d s ~ a t ~ a ~ c o m m a ~ o r ~ a n o t h e r ~ p o s i t i o n } , } \end{array} \right.
$$

$$
\rho _ { 1 } \geq \rho _ { 2 } \geq \rho _ { 3 } \geq \rho _ { 4 } \geq 0 .\tag{4}
$$

The parameter $\lambda$ controls the contribution of the structural term. In this paper, the tier weights $\rho _ { 1 } , \rho _ { 2 } , \rho _ { 3 } , \rho _ { 4 }$ are set to 0.1, 0.005, 0.001, and 0, respectively.

Why hierarchical boundary rewards. Paragraph endings usually mark the strongest semantic boundaries; when a cut must fall inside a paragraph, sentence endings are preferable to clause endings, and clause endings are preferable to arbitrary positions. The tier weights encode this hierarchy while allowing question–evidence supervision to trade structural preference against evidence coverage.

## 3.3 PARTITION DECOMPOSITION

When $S$ is a partition, its chunks are disjoint, so $\textstyle \bigcup _ { C \in S } \Gamma ( C )$ is a disjoint union and $f$ decomposes as a sum of per-chunk values $g ( C )$ —exactly the condition that makes dynamic programming valid (§4). When overlap is allowed this decomposition breaks and $f$ must be treated as a set function, which is where submodularity earns its keep (§5). Both settings therefore share the same $f$ and, under the evaluation convention of §6.1, directly comparable results.

## 4 SETTING 1: NON-OVERLAP PARTITION

Given the prior parameters K, $L _ { \mathrm { m i n } } .$ , and $L _ { \mathrm { m a x } }$ , we seek a contiguous partition of $u _ { 1 } , \ldots , u _ { n }$ into exactly $K$ chunks such that every chunk C satisfies $L _ { \mathrm { m i n } } \leq | \bar { C } | \leq \bar { L } _ { \mathrm { m a x } }$ or is admitted by the whole-paragraph exception below:

$$
\operatorname* { m a x } _ { S } f ( S ) \quad { \mathrm { s . t . } } \quad S { \mathrm { ~ i s ~ a ~ c o n t i g u o u s ~ } } K { \mathrm { - p a r t i t i o n ~ s a t i s f y i n g ~ t h e ~ l e n g t h ~ r u l e . } }\tag{5}
$$

Whole-paragraph exception. Because a natural paragraph is often a semantic unit, an entire paragraph longer than $L _ { \mathrm { m a x } }$ may form one chunk; oversized chunks that cross paragraph boundaries remain forbidden.

For every feasible chunk C, define its per-chunk utility as

$$
g ( C ) = \sum _ { ( j , r ) \in \Gamma ( C ) } w ( j , r ) + \lambda ( b _ { C } ) .\tag{6}
$$

Let $f ( i , k )$ be the maximum utility of a feasible partition of the prefix $u _ { 1 } , \ldots , u _ { i }$ into exactly k chunks, with value −∞ when no such partition exists. The sequential structure gives the recurrence

$$
f ( i , k ) = \operatorname* { m a x } _ { t \in \mathcal { T } ( i ) } \Big \{ f ( t , k - 1 ) + g \big ( [ t + 1 , i ] \big ) \Big \} , \qquad f ( 0 , 0 ) = 0 ,\tag{7}
$$

where the predecessor set adjoins the whole-paragraph exception to the usual length window:

$$
\begin{array} { r l } & { \mathcal { T } ( i ) = \underbrace { \left\{ t : i - L _ { \mathrm { m a x } } \leq t \leq i - L _ { \mathrm { m i n } } , \ t \geq 0 \right\} } _ { \mathrm { l e n g t h w i n d o w } } } \\ & { \qquad \cup \underbrace { \left\{ \mathrm { s t a r t } ( i ) - 1 \ \middle | \ i \in B , \ i - \mathrm { s t a r t } ( i ) + 1 > L _ { \mathrm { m a x } } \right\} } _ { \mathrm { w h o l e - p a r a g r a p h e x c e p t i o n } } . } \end{array}\tag{8}
$$

with start(i) the first block of the paragraph ending at i. The optimum is $f ( n , K )$ ; cut points are recovered by back-pointers.

Time complexity analysis. Let $W = L _ { \mathrm { m a x } } - L _ { \mathrm { m i n } } + 1$ . After interval gains are computed incrementally with block-to-question inverted lists, each DP state examines at most W length-feasible predecessors plus one whole-paragraph predecessor. The running time is therefore $O ( \bar { n } K W )$ and the table with back-pointers uses $\bar { O } ( n K )$ space.

Algorithm 1 Exact DP for non-overlapping chunking   
1: Input: $u _ { 1 : n } , ( R _ { j } , H _ { j } ) _ { j = 1 } ^ { m } , K , L _ { \operatorname* { m i n } } , L _ { \operatorname* { m a x } } , B$   
2: Initialize $f ( 0 , 0 ) \gets 0$ and all other states $\mathrm { t o } - \infty .$   
3: for $i = 1 , \ldots , \stackrel { \cdot } { n }$ do   
4: Form $\dot { \tau } ( i )$ and compute $g ( [ t { + } 1 , i ] )$ incrementally for $t \in \mathcal { T } ( i )$   
5: for $k = 1 , \ldots , K$ do   
6: Update $f ( i , k )$ by equation 7 and store its maximizing predecessor.   
7: end for   
8: end for   
9: return the partition obtained by backtracking from $f ( n , K )$

## 5 SETTING 2: OVERLAP SELECTION

Given a prescribed index budget $B _ { \mathrm { i n d e x } }$ , which caps the total number of indexed atomic blocks, and prior length bounds $L _ { \mathrm { m i n } }$ and $L _ { \mathrm { m a x } } .$ let

$$
\mathcal V = \{ [ a , b ] : 1 \leq a \leq b \leq n , L _ { \operatorname* { m i n } } \leq b - a + 1 \leq L _ { \operatorname* { m a x } } \}
$$

be the candidate family. Allowing overlap, the index is a set $S \subseteq \mathcal { V }$ rather than a partition, and we solve

$$
\operatorname* { m a x } _ { S \subseteq \mathcal { V } } f ( S ) \quad \mathrm { s . t . } \quad \sum _ { C \in S } | C | \leq B _ { \mathrm { i n d e x } } .\tag{9}
$$

Here each selected chunk incurs cost |C|, and the structural prior is instantiated as a paragraphend reward (the sentence and clause tiers are zero). Unlike the partition setting, the overlapping candidate family uses only the prescribed length bounds and does not invoke the whole-paragraph exception.

## 5.1 MONOTONE SUBMODULARITY

Proposition 1 (Monotone submodularity). The set function f in equation 3 is monotone and submodular over chunks drawn from any candidate family.

Proofsketch. The coverage term is a weighted coverage function: each element $( j , r )$ contributes $w ( j , r )$ once, the first time it is covered by a selected chunk. Weighted coverage is monotone submodular (Nemhauser et al., 1978). The prior term is non-negative and modular because each summand depends only on one chunk’s right endpoint. Their sum is therefore monotone submodular. Appendix A gives the full proof. □

## 5.2 EXACT OPTIMIZATION IS NP-HARD

We set all structural-prior weights to zero, so the result applies directly to the coverage term of equation 3 and implies hardness of the full objective when the prior weights are allowed to be zero.

Theorem 1 (NP-hardness for contiguous chunks). Let the maximum chunk length L be part of the input and let

$$
\mathcal { V } _ { L } = \{ [ a , b ] : 1 \leq a \leq b \leq n , b - a + 1 \leq L \} .
$$

Deciding whether there exists $S \subseteq \mathcal { V } _ { L }$ with $\begin{array} { r } { \sum _ { C \in S } \left| C \right| \le B _ { \mathrm { i n d e x } } } \end{array}$ and $f _ { \mathrm { c o v } } ( S ) = 1$ is NP-complete, even when $| R _ { j } | = 1$ and $| H _ { j } | = 2$ for every question, the weights are $v ( j , r ) = 1 / m$ , and all structural-prior weights are zero. Consequently, exact maximization of the coverage-only objective over contiguous chunks is NP-hard. The same conclusion holds when $\gamma _ { L }$ consists of all intervals of length exactly L.

Proofsketch. We reduce from GRID SEGMENT COVERING, which is NP-complete by Bergren et al. (2020, Theorem 7). Its input is a set of m vertical unit segments $e _ { j } = [ ( x _ { j } , y _ { j } ) , ( x _ { j } , y _ { j } + 1 ) ]$ on an integer grid of side $q = O ( m )$ and an integer k; a horizontal length-2 segment covers $e _ { j }$ if it contains either endpoint. We may assume $1 \leq k \leq m$

Set $D = q + 2 m + 3$ and encode $( x , y )$ by $D y + x + 3$ . The endpoints of $e _ { j }$ then become $\{ p _ { j } , p _ { j } + D \}$ where $p _ { j } = D y _ { j } + x _ { j } + 3$ . Encoded points on distinct rows are more than 2 apart, so the grid instance is coverable by k horizontal length-2 segments exactly when k length-2 intervals on the line contain at least one point from every pair $\{ p _ { j } , p _ { j } + D \}$ . Such line intervals may be assumed to have integer left endpoints.

Let $p _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { j } p _ { j }$ and construct $n = p _ { \mathrm { m a x } } + 2 D + 2$ atomic blocks. For each $j ,$ , define

$$
R _ { j } = \{ p _ { j } + D \} , \qquad H _ { j } = \{ p _ { j } , p _ { j } + 2 D \} .\tag{10}
$$

Give its sole evidence pair weight $1 / m$ , set all prior weights to zero, and choose $L = D + 3$ and $B _ { \mathrm { i n d e x } } = k L$ . For a length-L chunk $C _ { t } = [ t , t + D + 2 ]$ 」,

$$
\begin{array} { r } { ( j , p _ { j } + D ) \in \Gamma ( C _ { t } ) \iff t \in [ p _ { j } - 2 , p _ { j } ] \\\cup [ p _ { j } + D - 2 , p _ { j } + D ] } \\ { \iff [ t , t + 2 ] \cap \{ p _ { j } , p _ { j } + D \} \neq \emptyset . } \end{array}
$$

Thus $C _ { t }$ covers question $j$ exactly when the corresponding length-2 interval covers an encoded endpoint of $e _ { j }$

This correspondence maps every grid cover of size at most k to a feasible chunk family with $f _ { \mathrm { c o v } } =$ 1. Conversely, every chunk with nonempty coverage spans an evidence block and a hit block D positions apart, and hence costs at least $D + 1$ . Since $( k + 1 ) ( D + 1 ) - k L = D + 1 - 2 k > 0 ,$ a feasible solution contains at most k such chunks. Extending each to length $L$ preserves coverage, after which the same correspondence yields a grid cover of size at most k. Appendix B gives both directions in full, including the normalization and boundary arguments.

The construction is polynomial because $D , L = O ( m ) , n = O ( m ^ { 2 } )$ , and $| \mathcal { V } _ { L } | = O ( n L ) = O ( m ^ { 3 } ) ;$ feasibility and coverage are also polynomially verifiable. This proves NP-completeness. If only length-L chunks are allowed, the same construction applies and the budget directly limits the solution to k chunks, proving the final claim. □

## 5.3 A $( 1 - 1 / \sqrt { e } )$ COST-BENEFIT GREEDY

By Proposition 1, f is monotone submodular, so equation 9 is budgeted monotone submodular maximization, for which the cost-benefit greedy with a singleton safeguard (MGREEDY) attains the following conservative guarantee under the corrected analysis of Tang et al. (2021).

Algorithm 2 Lazy cost-benefit greedy with singleton safeguard   
1: Input: V, $B _ { \mathrm { i n d e x } } ,$ and objective f   
2: $S  \varnothing ;$ heapify V by $\dot { \Delta ( C \mid S ) ^ { } } / | C |$   
3: while the heap is non-empty and $\begin{array} { r } { \sum _ { C \in S } \vert C \vert < B _ { \mathrm { i n d e x } } } \end{array}$ do   
4: Pop C and recompute $\Delta ( C \mid S ) { \big / } | { \bar { C } } |$   
5: if C is feasible, has positive gain, and remains heap-maximal then   
6: $S  S \cup \{ C \}$   
7: else if C is feasible and $\Delta ( C \mid S ) > 0$ then   
8: Reinsert C with its updated density.   
9: else   
10: Discard C.   
11: end if   
12: end while   
13: $s ^ { \star }  \arg \operatorname* { m a x } _ { C \in \mathcal { V } : | C | \leq B _ { \mathrm { i n d e x } } } f ( \{ C \} )$   
14: return arg $\operatorname* { m a x } _ { T \in \{ S , \{ s ^ { \star } \} \} } f ( T )$

Theorem 2 (Approximation guarantee (Tang et al., 2021)). Let f be monotone submodular over V with $f ( \varnothing ) \ge 0 \sp { \ - } \mathrm { { } } i n d \mathrm { { O P T } } = \operatorname* { m a x } \{ f ( S ) : S \stackrel {  } { \subseteq } \mathcal { V } , \sum _ { C \in S } | C | \le B _ { i n d e x } \}$ . Algorithm 2 returns S with

$$
\begin{array} { r } { f ( S ) \geq \left( 1 - \frac { 1 } { \sqrt { e } } \right) \mathrm { O P T } \approx 0 . 3 9 3 \mathrm { O P T } . } \end{array}\tag{11}
$$

Tang et al. (2021) close a gap in the earlier analysis of Khuller et al. (1999) and prove the stronger 0.405 approximation factor for the same modified greedy algorithm; we retain the weaker $( 1 - 1 / { \sqrt { e } } )$ bound here. Each round selects the interval with maximum marginal gain per unit cost,

$$
C ^ { \star } = \operatorname * { a r g m a x } _ { C \in \mathcal { V } } \frac { \Delta ( C \mid S ) } { \mid C \mid } , \qquad \Delta ( C \mid S ) = f ( S \cup \{ C \} ) - f ( S ) ,\tag{12}
$$

and the singleton safeguard compares the greedy’s result with the best single feasible chunk, returning the better. Reaching the $( 1 \bar { - } 1 / e ) \approx 0 . 6 \dot { 3 } 2$ guarantee instead requires enumerating all initial subsets of size $\leq 3$ (Sviridenko, 2004), which is $| \mathcal { V } | ^ { 3 } \approx 1 0 ^ { 1 4 }$ at our scale and infeasible; and for coverage objectives under cardinality constraints no polynomial algorithm can beat $( 1 - 1 / e )$ unless ${ \mathrm { P } } = { \mathrm { N P } }$ (Feige, 1998), so the residual $\sqrt { e }$ gap is the price of tractability. Lazy evaluation makes the greedy practical: by submodularity a cached marginal gain is an upper bound on the current one, so a max-heap of stale densities needs only tens of exact re-evaluations per accepted chunk, and it returns exactly the same set as the eager cost-benefit greedy. Algorithm 2 summarizes the procedure.

In practice, the empirical gap between the algorithm’s value and the optimum is generally much smaller than $1 - { \frac { 1 } { \sqrt { e } } }$

## 6 EXPERIMENTS

## 6.1 EXPERIMENTAL SETUP

Datasets and supervision. Table 1 summarizes the corpora and supervision protocols.

Table 1: Dataset statistics and supervision.
<table><tr><td>Dataset</td><td>Type</td><td>Docs</td><td>Blocks</td><td>Paragraphs</td><td>Questions</td></tr><tr><td>Party Congress report</td><td>Self-built</td><td>1</td><td>2,842</td><td>147</td><td>642</td></tr><tr><td>Public-health white paper</td><td>Self-built</td><td>1</td><td>2,884</td><td>232</td><td>683</td></tr><tr><td>QASPER (Dasigi et al., 2021)</td><td>Public</td><td>185</td><td>103,024</td><td>12,548</td><td>532</td></tr><tr><td>PeerQA (Baumgärtner et al., 2025)</td><td>Public</td><td>70</td><td>43,656</td><td>6,222</td><td>136</td></tr><tr><td>PrivacyQA (Ravichander et al., 2019)</td><td>Public</td><td>34</td><td>4,931</td><td>4,931</td><td>1,276</td></tr></table>

Supervision protocol. Both self-built corpora use eight-fold cross-validation, with boundary fitting and testing separated in every fold. For each public document containing n atomic blocks, the

Greedy paragraph-end bonus λ (symmetric-log)

qwen3.8-max API generates $n / 5$ synthetic questions exclusively for boundary fitting; contiguousevidence and cross-paragraph-evidence questions are generated in a 1:1 ratio. Evaluation then uses only the datasets’ native public questions and evidence annotations. We compare against the nonoverlap and overlap baseline families described in §2, using identical retrieval and returned-context budgets.

Automatic checks validate the generated fitting data. PrivacyQA treats every atom as a paragraph, so the natural-paragraph baseline is not applicable.

Evaluation convention. @k denotes complete-evidence recall when the deduplicated returned context contains k atomic blocks. @k% denotes the same metric when the context budget is k% of the document’s atomic blocks.

All methods use the same embedding–BM25 retriever with section-title prefixes. Coverage is aggregated per question; self-built results are averaged over the eight folds. The parameter study below fixes operating points before the cross-corpus comparison.

## 6.2 PARAMETER SWEEPS

We vary one factor at a time on the public-health white paper while holding the retriever, returnedatom budget, and eight folds fixed.

## 6.2.1 STRUCTURAL WEIGHT

Figure 2 shows complementary trends. For DP, coverage rises as small positive structural weights are introduced and then reaches a broad plateau around $\lambda = 1 0 ^ { - 2 }$ ; both display budgets follow the same pattern. Greedy is stable near zero through small bonuses but declines once λ becomes large, sharply so at $\lambda \geq 1 0 ^ { \frac { - } { - 2 } }$ , indicating that an over-weighted paragraph prior displaces evidence-bearing chunks.

Nested length bounds [L<sub>min</sub>, L<sub>max</sub>]  
![](images/f8ace8561c275b719246710bd33e598c5ddcad8559dc8fb50cdf77a9fa5bd1e6.jpg)  
DP tier scale λ (symmetric-log)

![](images/13ed4e7db9c053b607deaf7dc4001ee635a3043c2e8138bf14e5c2544e75276c.jpg)  
Figure 2: Coverage over the λ sweep at @100 and $@ 1 0 \%$

## 6.2.2 LENGTH GEOMETRY

Figure 3 shows that both variants change smoothly as the feasible interval contracts rather than exhibiting a narrow optimum. DP is nearly flat over broad ranges and degrades only near the fixedlength limit; greedy is similarly stable at @10%, with a modest decline at @100. At the rightmost point $L _ { \mathrm { m i n } } = L _ { \mathrm { m a x } }$ , so the admissible chunks are exactly those of equal-length splitting and the two settings are fully equivalent.

![](images/0609856fa2032a5f4598a6bc37fba7f271b6e0c457f6fb4619850a1745b2f765.jpg)

![](images/753a36f356699af7de1c0ee0761ce2ebc025fe694e8937e539616041059fb8a7.jpg)  
Figure 3: Coverage as candidate-length intervals contract.

## 6.2.3 WHOLE-PARAGRAPH EXCEPTION

Table 2 shows that the whole-paragraph exception helps at small returned-context budgets but provides no advantage at large budgets. For fairness in the subsequent comparison, we enable the exception in every non-overlap main experiment.

Table 2: White-paper coverage with the whole-paragraph exception disabled or enabled; bold marks the better result.
<table><tr><td></td><td colspan="5">Tight budgets (atoms)</td><td colspan="6">Larger budgets (atoms)</td></tr><tr><td>Exception</td><td>50</td><td>58</td><td>100</td><td>144</td><td>200</td><td>288</td><td>305</td><td>433</td><td>577</td><td>721</td><td>865</td></tr><tr><td>Disabled</td><td>0.7076</td><td>0.7361</td><td>0.8142</td><td>0.8598</td><td>0.8972</td><td>0.9310</td><td>0.9378</td><td>0.9595</td><td>0.9714</td><td>0.9828</td><td>0.9863</td></tr><tr><td>Enabled</td><td>0.7177</td><td>0.7490</td><td>0.8216</td><td>0.8631</td><td>0.8980</td><td>0.9285</td><td>0.9340</td><td>0.9587</td><td>0.9703</td><td>0.9798</td><td>0.9853</td></tr><tr><td>Enabled — disabled (pp)</td><td>+1.01</td><td>+1.29</td><td>+0.73</td><td>+0.33</td><td>+0.08</td><td>-0.25</td><td>-0.38</td><td>-0.08</td><td>-0.12</td><td>-0.29</td><td>-0.10</td></tr></table>

## 6.3 MAIN EXPERIMENTS

Based on §6.2, we fix all operating points before the cross-corpus comparison. For non-overlap, we set $\kappa = 5 , \lambda = 0 . 0 1$ , take $[ L _ { \mathrm { m i n } } , L _ { \mathrm { m a x } } ]$ as the 10th–90th percentiles of paragraph length, and set K to the number of paragraphs. For overlap, we set $\lambda = \dot { 1 } 0 ^ { - 4 }$ , use the same percentile-based length bounds, and set the total budget to three times the document length. We then compare nonoverlapping methods under a common partition constraint and overlapping methods under the same returned-context convention.

Residual-budget concentration (overlap only). When the primary coverage objective saturates while index budget remains, we use

$$
f _ { 2 } ( S ) = \frac { 1 } { m } \sum _ { j } \operatorname* { m a x } _ { C \in S , \ C \cap H _ { j } \neq \emptyset } \frac { | C \cap R _ { j } | } { | R _ { j } | } ,\tag{13}
$$

and continue selection with $f + \delta f _ { 2 }$ , where $\delta = 0 . 0 5$ , which favors candidates that concentrate more of a question’s evidence in one retrievable chunk.

Table 3: Non-overlap setting: complete-evidence coverage at the two display tiers (@100 fixed atoms; @10% of document atoms). On PrivacyQA the natural-paragraph baseline is degenerate (one atomic block per paragraph) and is therefore marked “–”.
<table><tr><td></td><td colspan="2">Congress report</td><td colspan="2">White paper</td><td colspan="2">QASPER</td><td colspan="2">PeerQA</td><td colspan="2">PrivacyQA</td></tr><tr><td>Method</td><td>@100</td><td>@10%</td><td>@100</td><td>@10%</td><td>@100</td><td>@10%</td><td>@100</td><td>@10%</td><td>@100</td><td>@10%</td></tr><tr><td>QESCHUNKER (exact DP)</td><td>0.9025</td><td>0.9781</td><td>0.8216</td><td>0.9285</td><td>0.7588</td><td>0.6029</td><td>0.7341</td><td>0.6589</td><td>0.8687</td><td>0.3166</td></tr><tr><td>Segment-by-paragraph</td><td>0.9020</td><td>0.9734</td><td>0.7818</td><td>0.8975</td><td>0.7425</td><td>0.5837</td><td>0.7399</td><td>0.6483</td><td></td><td></td></tr><tr><td>Equal-length split</td><td>0.8570</td><td>0.9669</td><td>0.7587</td><td>0.8946</td><td>0.6565</td><td>0.5018</td><td>0.6583</td><td>0.5424</td><td>0.8585</td><td>0.2975</td></tr><tr><td>TextTiling</td><td>0.8054</td><td>0.9424</td><td>0.7376</td><td>0.8801</td><td>0.6560</td><td>0.4842</td><td>0.6395</td><td>0.5604</td><td>0.8642</td><td>0.2956</td></tr><tr><td>ClusterSemanticChunker</td><td>0.8462</td><td>0.9494</td><td>0.7499</td><td>0.8871</td><td>0.6511</td><td>0.4540</td><td>0.6527</td><td>0.5357</td><td>0.8607</td><td>0.3039</td></tr><tr><td>LumberChunker (LLM)</td><td>0.8789</td><td>0.9694</td><td>0.7700</td><td>0.9024</td><td>0.6915</td><td>0.5064</td><td>0.6659</td><td>0.5406</td><td>0.8601</td><td>0.3001</td></tr></table>

Table 4: Overlap setting: complete-evidence coverage at the two display tiers, same corpora and convention as Table 3.
<table><tr><td></td><td colspan="2">Congress report</td><td colspan="2">White paper</td><td colspan="2">QASPER</td><td colspan="2">PeerQA</td><td colspan="2">PrivacyQA</td></tr><tr><td>Method</td><td>@100</td><td>@10%</td><td>@100</td><td>@10%</td><td>@100</td><td>@10%</td><td>@100</td><td>@10%</td><td>@100</td><td>@10%</td></tr><tr><td>QESCHUNKER (greedy)</td><td>0.9165</td><td>0.9781</td><td>0.8656</td><td>0.9545</td><td>0.7412</td><td>0.5786</td><td>0.7399</td><td>0.6483</td><td>0.8747</td><td>0.3302</td></tr><tr><td>Fixed window  $\scriptstyle L = { \bar { b } } l , { \mathrm { o v . } } 2 0 \%$ </td><td>0.8708</td><td>0.9626</td><td>0.7266</td><td>0.8846</td><td>0.6807</td><td>0.5296</td><td>0.6939</td><td>0.5456</td><td>0.8581</td><td>0.3138</td></tr><tr><td>Fixed window L=2bl, ov. 20%</td><td>0.8778</td><td>0.9609</td><td>0.7889</td><td>0.9148</td><td>0.6554</td><td>0.4721</td><td>0.6901</td><td>0.5672</td><td>0.8686</td><td>0.2947</td></tr><tr><td>Sentence-Window ±1</td><td>0.8312</td><td>0.9538</td><td>0.7469</td><td>0.8937</td><td>0.7086</td><td>0.5206</td><td>0.6862</td><td>0.5555</td><td>0.8515</td><td>0.2739</td></tr><tr><td>Sentence-Window ±2</td><td>0.8464</td><td>0.9672</td><td>0.7733</td><td>0.9079</td><td>0.6830</td><td>0.5033</td><td>0.6600</td><td>0.5436</td><td>0.8535</td><td>0.2639</td></tr><tr><td>Parent-Document</td><td>0.8615</td><td>0.9579</td><td>0.7713</td><td>0.8909</td><td>0.7095</td><td>0.5238</td><td>0.6830</td><td>0.5595</td><td>0.8461</td><td>0.2945</td></tr><tr><td>Auto-Merging (4 sent.)</td><td>0.8232</td><td>0.9561</td><td>0.7559</td><td>0.8947</td><td>0.6801</td><td>0.4881</td><td>0.6590</td><td>0.5538</td><td>0.8509</td><td>0.2650</td></tr></table>

Non-overlap. The exact DP leads in $9 / 1 0$ displayed cells. Its only loss is PeerQA at @100, where the paragraph baseline is ahead by just 0.58 points and the evidence is often already paragraphaligned. The largest gains occur on the white paper (+3.98 points at @100, +2.61 at @10%); nearties on the Congress report and PrivacyQA indicate less room when simple segmentation already matches the evidence geometry.

Overlap. Greedy leads in all 10/10 cells, by 0.61–8.11 points over the strongest baseline. The largest gaps occur on the white paper at @100 and PeerQA at @10%; gains remain positive on PrivacyQA. Since all rows share the same overlap and returned-context conventions, the pattern reflects evidence-aware selection rather than unequal evaluation budgets.

## 6.4 BUDGET ANALYSIS

Figure 4 traces the white-paper budget range; the other four datasets appear in Appendix C. Coverage rises monotonically for every method, but QESChunker separates at the smallest budgets and remains ahead in both settings. The gap is widest before the curves flatten, when only a small document fraction can be returned.

![](images/dbfa23ab0c0b3d8fce735eb5e563b7ed017b8e97ac2c94d39372df241b32800d.jpg)

![](images/ff3b79d9633b3b87c53457a38c47140a1b34552f544a8a7f7adafdf0f8aa1dc5.jpg)  
Figure 4: Coverage versus returned-atom budget on the white paper. Vertical lines mark @100 and @10%.

The appendix curves repeat this pattern with different strength: Congress methods converge near saturation; QASPER and PeerQA retain wider overlap-side gaps at low and medium budgets; and PrivacyQA has tightly clustered non-overlap curves but clear overlap separation. Thus more context helps every method while reducing the marginal value of optimized boundaries near the coverage ceiling.

## 7 LIMITATIONS

On one hand, our gains are measured at the retrieval level as complete-evidence recall. Providing more complete and effective evidence can plausibly improve downstream generation, but the rate at which recall gains translate into final answer quality remains unknown. This conversion depends on the generator, prompting, context ordering, evidence redundancy, and answer-level evaluation criteria; we therefore do not claim a fixed or proportional end-to-end improvement.

On the other hand, boundary fitting depends heavily on the quality and coverage of existing question–evidence supervision. It is thus best suited to documents that are queried repeatedly and have accumulated sufficient supervision—for example, policy reports, FAQs, or frequently discussed papers. For one-off, cold-start, or sparsely annotated documents, such supervision may be unreliable or unavailable, and performance may be driven mainly by the structural prior and the retriever rather than by learned boundaries.

## 8 CONCLUSION

QESChunker casts chunk construction as supervised, budget-aware optimization rather than heuristic preprocessing. A shared hit-gated coverage objective yields two complementary solvers: exact dynamic programming for non-overlapping partitions and benefit-cost submodular greedy for overlapping indices. Across five corpora, the two variants improve complete-evidence recall in nearly all displayed settings, with the largest gains appearing at tight returned-context budgets and smaller margins near saturation. The parameter sweeps further show that performance remains stable across broad structural-weight and candidate-length ranges. Together, these results indicate that question– evidence supervision can translate recurring retrieval demand into principled, document-specific static indices under either overlap regime.

## AI USE STATEMENT

Generative AI was used to generate fitting questions and evidence IDs and to judge auxiliary outputs. Data passed legality, membership, length, hallucination, and duplication checks; public evaluation used only human or expert labels. Judging was anonymized, order-swapped, and conducted per fact. AI was not used for theoretical development or solver implementation. The authors reviewed all outputs and accept responsibility.

## ETHICS STATEMENT

This work studies document segmentation for retrieval-augmented generation and uses publicly available scientific QA data (QASPER, PeerQA), a public privacy-policy QA corpus (PrivacyQA), and public government policy documents. It involves no human subjects, no private data, and no personal identifiers. The main dual-use consideration is that better evidence retrieval can improve both beneficial and harmful downstream generation; we mitigate fabrication risk in our own evaluation via hallucination screening, and we report negative and boundary results (scope, oracle ceiling, cost accounting) to avoid overstating capability. LLM-annotated evidence on the self-built corpora may inherit annotator-model biases; we flag this as a limitation and rely on human-annotated QASPER and author-annotated PeerQA for our highest-confidence claims.

## REPRODUCIBILITY STATEMENT

The method is fully specified in §3–§5: the objective equation 3 with weights equation 1 and tier ratios equation 4, the DP recurrence and predecessor set equation 7–equation 8 with the whole paragraph exception (Algorithm 1), the NP-hardness reduction (Theorem 1), and the greedy with its guarantee, lazy evaluation, singleton safeguard (Algorithm 2, Theorem 2). The residual-budget concentration term equation 13, evaluation convention, and sensitivity analyses are given in §6.3, §6.1, and §6.2, respectively. Correctness claims are backed by brute-force enumeration for the DP and by the certificate for the greedy.

## REFERENCES

Tim Baumgartner et al. PeerQA: A scientific question answering dataset from peer reviews. In¨ Proceedings of the North American Chapter of the Association for Computational Linguistics (NAACL), 2025. Author list to be completed for camera-ready.

Dan Bergren, Eduard Eiben, Robert Ganian, and Iyad Kanj. On covering segments with unit intervals. In 37th International Symposium on Theoretical Aspects of Computer Science (STACS 2020), volume 154 of Leibniz International Proceedings in Informatics (LIPIcs), pp. 13:1–13:17. Schloss Dagstuhl–Leibniz-Zentrum fur Informatik, 2020. doi: 10.4230/LIPIcs.STACS.2020.13.¨

Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao, Hongming Zhang, and Dong Yu. Dense x retrieval: What retrieval granularity should we use? In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 15159–15177. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.emnlp-main.845.

Freddy Y. Y. Choi. Advances in domain independent linear text segmentation. In Proceedings of the 1st Meeting of the North American Chapter of the Association for Computational Linguistics, 2000. URL https://aclanthology.org/A00-2004/.

Pradeep Dasigi, Kyle Liu, Ana Marasovic, Noah A. Smith, and Matt Gardner. A dataset of information-seeking questions and answers anchored in research papers. In Proceedings of the North American Chapter of the Association for Computational Linguistics (NAACL), 2021.

Andre V. Duarte, Jo´ ao Marques, Miguel Grac¸a, Miguel Freire, Lei Li, and Arlindo L. Oliveira.˜ LumberChunker: Long-form narrative document segmentation. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 6473–6486, 2024. doi: 10.18653/v1/2024. findings-emnlp.377.

Uriel Feige. A threshold of ln n for approximating set cover. Journal of the ACM, 45(4):634–652, 1998.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2023.

Michael Gunther, Isabelle Mohr, Daniel James Williams, Bo Wang, and Han Xiao. Late chunk- ¨ ing: Contextual chunk embeddings using long-context embedding models. arXiv preprint arXiv:2409.04701, 2024.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. REALM: Retrievalaugmented language model pre-training. In International Conference on Machine Learning (ICML), 2020.

Marti A. Hearst. TextTiling: Segmenting text into multi-paragraph subtopic passages. Computational Linguistics, 23(1):33–64, 1997.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi˘ Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020.

Samir Khuller, Anna Moss, and Joseph (Seffi) Naor. The budgeted maximum coverage problem. Information Processing Letters, 70(1):39–45, 1999.

Omri Koshorek, Adir Cohen, Noam Mor, Michael Rotman, and Jonathan Berant. Text segmentation as a supervised learning task. In Proceedings ofNAACL-HLT, Volume 2 (Short Papers), pp. 469– 473. Association for Computational Linguistics, 2018. doi: 10.18653/v1/N18-2075.

Andreas Krause and Daniel Golovin. Submodular function maximization. In Tractability: Practical Approaches to Hard Problems. Cambridge University Press, 2014.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨ aschel, Sebastian Riedel, and Douwe¨ Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

LlamaIndex. Auto merging retriever. https://docs.llamaindex.ai/, 2024a. LlamaIndex documentation.

LlamaIndex. Metadata replacement and node sentence window. https://docs.llamaindex. ai/, 2024b. LlamaIndex documentation.

Michal Lukasik, Boris Dadachev, Kishore Papineni, and Gonc¸alo Simoes. Text segmentation by˜ cross segment attention. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 4707–4716. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020.emnlp-main.380.

George L. Nemhauser, Laurence A. Wolsey, and Marshall L. Fisher. An analysis of approximations for maximizing submodular set functions—I. Mathematical Programming, 14(1):265–294, 1978.

Abhilasha Ravichander, Alan W. Black, Shomir Wilson, Thomas Norton, and Norman Sadeh. Ques tion answering for privacy policies: Combining computational and legal perspectives. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 4947– 4958. Association for Computational Linguistics, 2019. doi: 10.18653/v1/D19-1500.

Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D. Manning. RAPTOR: Recursive abstractive processing for tree-organized retrieval. In International Conference on Learning Representations, 2024.

Brandon Smith and Anton Troynikov. Evaluating chunking strategies for retrieval. Technical report, Chroma, July 2024. URL https://www.trychroma.com/research/ evaluating-chunking.

Maxim Sviridenko. A note on submodular maximization with knapsack constraints. Operations Research Letters, 32(4):309–313, 2004.

Jing Tang, Xueyan Tang, Andrew Lim, Kai Han, Chongshou Li, and Junsong Yuan. Revisiting modified greedy algorithm for monotone submodular maximization with a knapsack constraint. Proceedings ofthe ACM on Measurement and Analysis ofComputing Systems, 5(1):1–22, 2021.

## A PROOF OF PROPOSITION 1

Proof. Write $f ( S ) = f _ { \mathrm { c o v } } ( S ) + f _ { \mathrm { p r i } } ( S )$ with $\begin{array} { r } { f _ { \mathrm { c o v } } ( S ) = \sum _ { ( j , r ) \in \bigcup _ { C \in S } \Gamma ( C ) } w ( j , r ) \mathrm { ~ a n d ~ } f _ { \mathrm { p r i } } ( S ) = } \end{array}$ $\textstyle \sum _ { C \in S } \lambda ( b _ { C } )$

$f _ { \mathrm { c o v } }$ is monotone submodular. Each element $e = ( j , r )$ with weight $w ( e ) \geq 0$ is covered by S iff some $C \in S$ has $e \in \Gamma ( C )$ ; thus $\begin{array} { r } { f _ { \mathrm { c o v } } ( S ) = \sum _ { e } w ( e ) } \end{array}$ 1[e covered by S] is a weighted coverage function, which is monotone (adding chunks cannot uncover) and submodular (the marginal contribution of a chunk is the total weight of elements it covers for the first time, which is non-increasing as S grows).

$f _ { \mathrm { p r i } }$ is modular and non-negative. Each summand $\lambda ( b _ { C } ) \ge 0$ depends only on the single chunk $C ;$ adding a chunk $C$ to any S increases $f _ { \mathrm { p r i } }$ by exactly $\lambda ( b _ { C } )$ , independent of S. A modular function is submodular with equality, and monotone since the increment is non-negative.

Sum. The sum of two monotone submodular functions is monotone submodular, so f is monotone submodular. The concentration term $f _ { 2 }$ in equation 13 is a sum over j of a max over selected chunks of modular-in-C terms gated by $C \cap H _ { i } \neq \emptyset ;$ a non-negative weighted max over selected sets is monotone submodular, so $f + \bar { \delta } f _ { 2 }$ with $\overset { \cdot } { \delta } \geq 0$ preserves monotone submodularity. □

On the partition decomposition. If S is a partition, its chunks are pairwise disjoint, so no element $( j , r )$ can be covered by two different chunks (each block r lies in exactly one chunk). Hence $\textstyle \bigcup _ { C \in S } \Gamma ( C )$ is a disjoint union and $\begin{array} { r } { f _ { \mathrm { c o v } } ( S ) = \sum _ { C \in S } g _ { \mathrm { c o v } } ( C ) } \end{array}$ with $\begin{array} { r } { g _ { \mathrm { c o v } } ( C ) = \sum _ { ( j , r ) \in \Gamma ( C ) } w ( j , r ) } \end{array}$ Together with the already-additive prior, $\begin{array} { r } { f ( S ) = \sum _ { C \in S } g ( C ) } \end{array}$ is additive over chunks, which is precisely the optimal-substructure exploited by equation 7.

## B FULL PROOF OF THEOREM 1

We use the construction and notation from the proof sketch.

Proof. Forward implication. Suppose the grid instance has a cover with at most k horizontal length-2 segments. After encoding the rows on the line, discard any interval containing no encoded endpoint. Shift every remaining interval rightward until its left endpoint is its smallest contained encoded point. This preserves all covered endpoints and gives an integer left endpoint. We therefore obtain at most k intervals $[ t , t + 2 ]$ that contain at least one point from every pair $\{ p _ { j } , p _ { j } + D \}$ .

Each normalized t is an encoded endpoint, so

$$
3 \leq t \leq p _ { \operatorname* { m a x } } + D = n - L + 1 .
$$

Replace each $[ t , t + 2 ]$ by $C _ { t } = [ t , t + D + 2 ]$ . These are legal chunks, their total cost is at most $k L = B _ { \mathrm { i n d e x } }$ , and the coverage correspondence gives $f _ { \mathrm { c o v } } ( S ) \mathbf { \bar { = } } 1$

Reverse implication. Suppose $S \subseteq \mathcal { V } _ { L }$ satisfies the budget and $f _ { \mathrm { c o v } } ( S ) = 1$ . Remove chunks with $\Gamma ( C ) = \emptyset$ , preserving coverage. Every remaining chunk covers some question $j ,$ so it contains $p _ { j } + D$ and at least one point in $\{ p _ { j } , p _ { j } + 2 D \}$ . These positions differ by $D ;$ hence each such chunk costs at least $D + 1$ . Since $k \leq m$ and $D = q + 2 m + 3 .$

$$
( k + 1 ) ( D + 1 ) - k L = D + 1 - 2 k > 0 .
$$

Thus at most k nonempty-coverage chunks remain.

For each remaining chunk $[ a , b ]$ , form $\begin{array} { r } { C _ { a } = [ a , a + D + 2 ] } \end{array}$ . This extension stays inside the document because the original chunk contains an evidence block, so

$$
a \le p _ { j } + D \le p _ { \mathrm { m a x } } + D = n - L + 1 .
$$

Since $b - a + 1 \leq L , C _ { a }$ contains the original chunk and preserves all its covered pairs. The resulting family has at most k length-L chunks and still covers every question. The coverage correspondence now shows that the intervals $[ a , a + 2 ]$ contain at least one point from every pair $\{ p _ { j } , p _ { j } + D \}$ As encoded points on distinct rows are more than 2 apart, these intervals map back to at most k horizontal length-2 segments covering the grid instance.

Polynomial size and NP membership. Because $q ~ = ~ O ( m )$ , we have $D , L \ = \ O ( m )$ and $n \stackrel { . } { = } O ( m ^ { 2 } )$ . The full candidate family has size ${ \tilde { O ( n L ) } } = \mathrm { \dot { O } } ( \mathrm { m } ^ { 3 } )$ , so the reduction is polynomial. A selected family is a polynomial-size certificate, and its cost and coverage are polynomially checkable. This proves NP-completeness.

If candidates are restricted to length exactly L, the forward direction is unchanged. In the reverse direction, the budget immediately permits at most k chunks, so the short-chunk argument is unnecessary. □

## C ADDITIONAL BUDGET CURVES

We report the full returned-context budget curves for the four datasets not shown in Figure 4. Each panel follows the same non-overlap/overlap convention as the white-paper analysis.

![](images/d939d024839b57aa576472576836f658545fd689bdaa30459ade6096921b7b6b.jpg)

![](images/b1ce72c46527722cf7400abbe041da1364d5eff3e1574eb2db3831ccd181dcba.jpg)

Figure 5: Budget curves on the Party Congress report. Left: non-overlap; right: overlap.  
![](images/8baacb06a0c846fd42e1d0900714fc2335e1127673ed5dd93e295200dbf70cef.jpg)

![](images/50b7f80efc230204b1e12ebba02336b28f4feb55d3cd142b698cdb0a4ba0a662.jpg)  
Figure 6: Budget curves on QASPER. Left: non-overlap; right: overlap.

![](images/22417f692c947f76443b3ee1c6c910b978f41889371ed0edc839326463617446.jpg)

![](images/c15ff6bba6971eb36165e9c672bce212d17afd0d15860f5ac4339e0415d04f11.jpg)

Figure 7: Budget curves on PeerQA. Left: non-overlap; right: overlap.  
![](images/33ef3e045e9fd5443b7e778c0f0186ac12555dbe282bfbe4e35fad8b18b1d9a9.jpg)

![](images/af06a1f65446c8bba83cd3730b744481cee38818aa5a40313c826f1f240cf138.jpg)  
Figure 8: Budget curves on PrivacyQA. Left: non-overlap; right: overlap.