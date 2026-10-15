# Beyond Reachability: A Relation-Semantic Reasoning Framework for Biomedical Multi-hop QA

Ziyang Ling zaneling@mail.ustc.edu.cn University of Science and Technology of China Hefei, China

Ronald X.Xu chenziyi@mail.ustc.edu.cn University of Science and Technology of China Hefei, China

Ziyi Chen chenziyi@mail.ustc.edu.cn University of Science and Technology of China Hefei, China

Mingzhai Sun chenziyi@mail.ustc.edu.cn University of Science and Technology of China Hefei, China

## Abstract

Biomedical multi-hop reasoning over knowledge graphs requires identifying a semantically valid relation sequence that supports the answer. Existing methods can retrieve correct answers, yet often rank them poorly within a large pool of seemingly relevant candidates. Even when a correct answer is retrieved, its associated evidence and reasoning path may fail to support the relation composition required by the question. We term this disconnect between answer reachability and evidential validity the reachablebut-invalid problem. We present ReSemReason, which shares a question-induced distribution over typed, directed relation schemas between candidate discovery and evidence selection. Schema compatibility guides graph search; ReSemAttn then uses relationconditioned key/value projections and an uncertainty-weighted path prior to attend to candidate-specific evidence. A separate absolute-support score complements normalized attention when all retrieved paths are weak. The reported experiments on BioHopR, PrimeKGQA, and MedReason show improvements in answer ranking and answer-set quality. Path-discrimination analyses examine the distinction between endpoint correctness and evidential support. We also specify matched controls for search, attention, and schema uncertainty. Code: https://anonymous.4open.science/r/ReSemAttn.

## 1 Introduction

Biomedical multi-hop reasoning over knowledge graphs requires identifying a semantically valid relation sequence that supports the answer. A question may connect a drug to its target proteins and then to associated diseases, requiring several facts to be com posed under specific relational constraints. The knowledge needed for such reasoning is distributed across scientific publications and biomedical databases available on the Web. Knowledge graphs such as PrimeKG integrate heterogeneous resources into typed entities and directed relations [1], making this knowledge accessible through a shared, machine-interpretable structure. For biomedical search and question answering, efective access to this structure requires interpreting how relations compose into evidence for a user’s question.

This creates a semantic knowledge-access challenge for the Web: how can systems retrieve and combine interlinked biomedical facts while respecting the relation composition requested by a user? A path can connect relevant entities, and even reach the correct answer, without providing the required evidence. We call this the reachable-but-invalid problem. As illustrated in Figure 1, distinguishing valid support from misleading connectivity requires attention to relation types, traversal directions, and intermediate entity types. We address this challenge by using question-induced relation schemas to guide both candidate discovery and evidence selection. Our contribution concerns the semantic consumption of integrated biomedical knowledge which making explicit relation semantics op erational in retrieval and reasoning, and assessing supporting paths alongside answer correctness. This connects biomedical multi-hop reasoning to the broader goal of reliable knowledge access for Web search, question answering, and knowledge-grounded agents.

![](images/76e4364024eb919058d15ec59185f27ce34700f8fc58d9db2933f64fb718ae1f.jpg)  
Figure 1: Two paths reach correct answer chronic myeloid leukemia (CML), but only the ABL1 path supports the question. Bottom: reported BioHopR Coverage, H@1, and sameanswer PairAcc.

Recent research has increasingly combined large language models (LLMs) with structured biomedical knowledge. A survey of biomedical and healthcare LLM–KG integration identifies KG-guided reasoning and verification as important directions for improving factual grounding and interpretability [14]. More broadly, a recent survey of retrieval-augmented generation and reasoning emphasizes their complementary roles that retrieval supplies external premises [4]. This interaction is particularly relevant to biomedical questions, where several individually relevant facts may still fail to form an appropriate answer-supporting chain. Benchmarks and methods develop this interaction from diferent perspectives. BioHopR evaluates multi-hop, multi-answer biomedical reasoning [3], while MedReason uses knowledge graphs to construct grounded medical reasoning data [13]. More recently, eGoT combines iterative reasoning with biomedical knowledge retrieval to build evidence graphs for question answering and hypothesis generation [9]. In general KG question answering, RoG uses relation paths to organize planning, retrieval, and reasoning [5]; GNN-RAG uses graph neural retrieval to identify relevant candidates and extract paths for LLM reasoning [7]; and GCR constrains decoding with a KG-derived path index [6]. These advances establish the value of structured evidence and relation-aware reasoning.

These advances motivate examining both answer ranking and the evidence supporting a prediction. A correct answer may be retrieved within a large pool of seemingly relevant candidates yet remain poorly ranked. Even when ranking succeeds, answer accuracy alone cannot distinguish valid supporting evidence from a semantically incompatible path to the same endpoint. Evaluating the complete relation composition is therefore necessary to assess whether the retrieved evidence actually answers the question. Consider the running question, Which diseases are associated with proteins targeted by imatinib? As shown in Figure 1, the path from imatinib through ABL1 to chronic myeloid leukemia (CML) follows the requested drug–target-protein–disease composition. A contraindication branch reaches the same CML endpoint without establishing the required target-protein connection. Holding the answer entity fixed isolates the evidence-selection challenge: the system must distinguish these paths by their relation semantics.

Addressing this problem requires coordinating two stages. During candidate discovery, semantically incompatible branches can consume a limited search budget and displace valid evidence. A downstream reranker cannot recover candidates or paths that retrieval has already discarded. During evidence selection, several paths to the same candidate may compete for influence, although only some match the requested relation composition. Moreover, normalized attention expresses relative preference among retrieved paths: it can concentrate on one path even when every available path provides weak support. Candidate availability, relative evidence preference, and absolute support therefore need to be distinguished.

Question interpretation introduces an additional dificulty. Biomedical terminology and relation descriptions can admit several plausible schemas. Committing to one predicted schema can prematurely suppress a valid branch, whereas ignoring schema information leaves search and reasoning without an explicit representation of the requested composition. This motivates retaining a distribution over typed, directed relation schemas and using it consistently across the two stages, while accounting for uncertainty when it influences evidence attention.

We present ReSemReason, a relation-semantic reasoning framework that shares a question-induced schema distribution between candidate discovery and neural evidence selection. As shown in Figure 2, the framework first predicts several plausible schemas specifying relation sequences, traversal directions, and entity types.

Their compatibility with partial paths guides graph search, prioritizing relevant branches while preserving alternative interpretations through soft penalties. The resulting candidate-specific paths form the evidence memory for ReSemAttn. Relation-conditioned key/value projections shape how paths are represented, and an uncertainty-weighted compatibility prior guides token-to-path attention. Finally, an absolute-support term complements normalized attention, retaining information about whether the retrieved paths provide strong support at all. Together, these components connect the semantics used to discover evidence with those used to incorporate it into answer prediction.

Our contributions are (1) We formulate the reachable-but-invalid problem, distinguishing candidate availability and answer correctness from whether a supporting path satisfies the question’s relation composition. (2) We introduce ReSemReason, which shares soft relation schemas between candidate discovery and evidence selection, and ReSemAttn, which combines relation-conditioned representations, uncertainty-weighted attention, and absolute support.

## 2 Related Work

Biomedical multi-hop reasoning. MedHop studies evidence composition across biomedical documents [12]. More recent resources emphasize structured reasoning: BioHopR evaluates multi-answer reasoning over PrimeKG paths, PrimeKGQA provides executable graph queries, and MedReason constructs KG-grounded medical reasoning data [3, 8, 13]. Recent work further explores biomedical LLM–KG integration [14], including iterative evidence retrieval and graph-based reasoning in eGoT [9].

Retrieval and graph reasoning. Earlier approaches combine question-conditioned retrieval with graph and language representations [2, 10, 15, 16]. Recent methods advance iterative graph exploration, relation-path planning, graph neural retrieval, and graph-constrained decoding [5–7, 11], reflecting the growing integration of retrieval and reasoning [4]. ReSemReason shares soft, typed, directed relation schemas across candidate discovery and evidence selection.

## 3 Method

## 3.1 Task and Framework

Given question � and directed KG $\mathcal { G } = ( \mathcal { V } , \mathcal { R } , \mathcal { E } )$ , the system discovers candidate entities $C _ { q } ,$ , ranks them, and returns an answer set $\widehat { \mathcal { A } } _ { q }$ with supporting paths. For a path �, three events difer: the path exists in the KG; its endpoint belongs to the reference set ${ \mathcal { A } } _ { q } ;$ and its relation composition supports that endpoint as an answer to $q .$ We write the third event as Valid(�, �) and evaluate it independently of the predicted schema.

For answer $a _ { i : }$ , the selected explanation $\pi _ { i } ^ { * }$ must end at $a _ { i }$ . Success thus requires both availability, meaning that suitable candidates and evidence survive retrieval, and selection, meaning that the reasoner uses them correctly. Figure 2 follows the same imatinib case through grounding and schema prediction, candidate discovery, and pathbased reasoning. Appendix Sections A–C provide the evaluation and implementation details.

An availability ceiling. For methods whose output is restricted to $C _ { q } ,$ let $A _ { q }$ denote top-answer correctness and $G _ { q } = 1 \big [ \mathcal { A } _ { q } \cap C _ { q } \neq \emptyset \big ]$

![](images/d94c8ec0fb24db2f52980254c0832a66443578ec4ff44f2f5bef30a6f0957811.jpg)  
Figure 2: Overview of ReSemReason pipeline. Left: the two-layer ReSemAttn computation. Middle: input, schema-guided discovery, attention updates, and the CML answer with its ABL1 support. Right: shared schemas, relation-conditioned attention, and absolute support.

Then $A _ { q } \leq G _ { q } ,$ , giving

$$
\mathrm { H } @ 1 = \mathrm { C o v e r a g e } \cdot \mathrm { C o n d . H } @ 1 , \qquad \mathrm { H } @ 1 \leq \mathrm { C o v e r a g e } ,\tag{1}
$$

when conditional accuracy is defined on questions with $G _ { q } = 1$ This identity is per run on a common question set. It explains why reranking cannot compensate for missing candidates and motivates reporting both terms in E1. For multi-answer questions, Coverage is weaker than candidate recall: finding one answer can satisfy the former while leaving most of the answer set unavailable.

## 3.2 Grounding and Question-Only Schemas

We retain the top-� anchor entities under $p _ { \phi } ( e _ { 0 } \mid q )$ , rather than committing to a single lexical match. The reference grounder applies a softmax to cosine similarities between pooled question and entitytext representations, with temperature $T _ { g } = 0 . 1$ . It scores the loaded entity inventory before truncation. A schema describes hop count, entity types, relation families, and directions:

$$
S = ( L , \tau _ { 0 } , r _ { 1 } , d _ { 1 } , \tau _ { 1 } , . . . , r _ { L } , d _ { L } , \tau _ { L } ) .\tag{2}
$$

Independent heads over the pooled question representation score these components. A bounded enumeration retains the top-� hypotheses, whose scores $\ell _ { m }$ are renormalized:

$$
\rho _ { m } = \frac { \exp \ell _ { m } } { \sum _ { j = 1 } ^ { M } \exp \ell _ { j } } , \qquad S _ { q } = \{ S _ { q } ^ { ( m ) } \} _ { m = 1 } ^ { M } .\tag{3}
$$

The inference interface uses the question rather than gold test schemas. Schema enumeration and the model configuration are specified in Appendix Sections B.2 and C.

The confidence gate measures concentration within the retained hypotheses:

$$
c _ { q } = \left\{ 1 , \atop 1 - H ( \rho ) / \log M , \quad M > 1 , \quad  H ( \rho ) = - \sum _ { m } \rho _ { m } \log \rho _ { m } . \right.\tag{4}
$$

Uniform hypotheses give $c _ { q } \ = \ 0 .$ . This gate does not measure whether the correct schema survived truncation. For the running case, keeping several interpretations allows a target-protein branch to remain competitive even when it is not associated with the highest-scoring schema.

Uncertainty has two sources. High entropy can reflect several plausible interpretations, whereas low entropy can still reflect a confidently wrong retained set. The gate cannot recover a schema omitted from the top-� hypotheses. Appendix Section E.5 evaluates confidence calibration for hop, relation-sequence, and full-schema prediction.

## 3.3 Relation-Guided Candidate Discovery

Starting from each retained anchor, search expands directed edges and normalized inverse traversals. For an extension $\pi ^ { \prime }$ ending at $e ^ { \prime }$

its score is

$$
\begin{array} { l } { { \displaystyle s _ { \mathrm { s e a r c h } } ( \pi ^ { \prime } ) = \lambda _ { g } \log p _ { \phi } ( e _ { 0 } \mid q ) + \lambda _ { s } { \mathrm { C o m p a t } } ( q , \pi ^ { \prime } ) } } \\ { { \displaystyle ~ + \lambda _ { q } { \mathrm { S e m } } ( q , e ^ { \prime } ) - \lambda _ { h } h ( e ^ { \prime } ) , } } \\ { { \displaystyle { \mathrm { C o m p a t } } ( q , \pi ^ { \prime } ) = \log \sum _ { m } \rho _ { m } \exp [ - D _ { \mathrm { p r e f i x } } ( S _ { q } ^ { ( m ) } , \pi ^ { \prime } ) ] . } } \end{array}\tag{5}
$$

Here $D _ { \mathrm { p r e f i x } }$ sums position-aligned relation, direction, and destination-type mismatches over the overlapping prefix. Endpoint relevance is cosine similarity and $h ( e ^ { \prime } ) = \log ( 1 + \deg ( e ^ { \prime } ) )$ . The reference coeficients are $( 1 , 2 , 0 . 5 , 0 . 1 5 ) _  $ ; the explicit mismatch and search procedure appear in Appendix Section B.3.

Search rejects repeated nodes and retains the top-� continuing paths at each depth. Paths reaching a retained schema’s hop count enter the completed collection before the next beam is pruned. Completed paths are grouped by endpoint, with at most � retained per candidate. Thus $C _ { q }$ is discovered during traversal rather than supplied to a reranker. Compatibility penalties are soft: matching a lower-probability schema can preserve a branch, although fi nite budgets and grounding errors can still remove valid evidence. Search is discrete and uses fixed coeficients in the reference implementation; ReSemAttn does not feed scores back into expansion.

Following the running case. The target-protein branch matches the dominant schema; the contraindication branch mismatches both relation and bridge type. The protein-to-disease hop then completes the requested composition. Endpoint relevance alone cannot distinguish these routes. Soft mismatch penalties change search priority before pruning, while independent annotation determines whether the completed evidence is valid.

## 3.4 ReSemAttn: Relation-Semantic Path Attention

For candidate $a _ { i } ,$ , let $\Pi _ { i } = \left\{ \pi _ { i k } \right\}$ be its retained paths. Each memory combines entity-type, directed-relation, path-summary, and topology features:

$$
\begin{array} { r l } & { f _ { i k } = [ \phi _ { E } ( E _ { i k } ) ; \phi _ { R } ( R _ { i k } ) ; \phi _ { P } ( P _ { i k } ) ; \phi _ { T } ( T _ { i k } ) ] , } \\ & { z _ { i k } = \mathrm { L N } \big ( \mathrm { D r o p o u t } ( \mathrm { G E L U } ( W _ { z } f _ { i k } ) ) \big ) . } \end{array}\tag{6}
$$

The reference encoder pools type and relation embeddings and uses numerical path and degree features. Relation order is compared explicitly in the schema mismatch, rather than preserved by mean pooling. These features do not estimate edge-provenance reliability. Semantic compatibility. For each schema, the mismatch vector contains type disagreement, directed-relation disagreement, length/closure mismatch, and mean log-degree. With nonnegative weights � = softplus(�),

$$
\begin{array} { l } { { \displaystyle { \mathcal E } _ { i k } ^ { ( m ) } = \beta ^ { \top } { \bf d } _ { i k } ^ { ( m ) } } , } \\ { { \displaystyle u _ { i k } = \log \sum _ { m } \rho _ { m } \exp ( - { \mathcal E } _ { i k } ^ { ( m ) } ) , \qquad \ p _ { i k } = \mathrm { s o f t m a x } _ { k } ( u _ { i k } ) } . } \end{array}\tag{7}
$$

For a path of length �, let $v _ { j }$ be its �th entity and $( r _ { j } , d _ { j } )$ its incoming relation and direction. Barred symbols denote a schema of length �. Define alignment lengths $J = \operatorname* { m i n } ( L , H )$ and $J _ { e } = \operatorname* { m i n } ( L + 1 , H + 1 )$

The mismatch vector is

$$
\begin{array} { r l } & { D _ { E } = J _ { \mathrm { c } } ^ { - 1 } \displaystyle \sum _ { j = 0 } ^ { J _ { E } - 1 } 1 \big [ \tau ( \boldsymbol { v } _ { j } ) \neq \bar { \tau } _ { j } \big ] , } \\ & { D _ { R } = \frac { \displaystyle \sum _ { j = 1 } ^ { J } \big ( 1 \big [ r _ { j } \neq \bar { r } _ { j } \big ] + 0 . 7 5 1 \big [ d _ { j } \neq \bar { d } _ { j } ] \big ) } { \operatorname* { m a x } ( 1 , J ) } , } \\ & { D _ { P } = \big | L - H \big | + 1 \big [ v _ { 0 } = v _ { L } \big ] , } \\ & { D _ { T } = ( L + 1 ) ^ { - 1 } \displaystyle \sum _ { j = 0 } ^ { L } \log ( 1 + \deg ( v _ { j } ) ) , } \\ & { { \mathrm { ~ d } = [ D _ { E } , D _ { R } , D _ { P } , D _ { T } ] ^ { \top } . } } \end{array}\tag{8}
$$

Entity matching compares types; directed-relation matching preserves positional order. The path term penalizes length and closure, and the topology term measures graph degree rather than provenance quality. Since search rejects repeated nodes, closure is normally zero. Nonnegative energies and normalized schema weights imply $u _ { i k } \ \leq \ 0$ ; increasing a mismatch cannot improve compatibility. This makes the score inspectable without equating schema agreement with independently annotated validity.

Search sums mismatches over incomplete prefixes; the evidence prior normalizes aligned type/relation terms for completed paths. They share schema content and directionality, while scoring diferent decisions: which extension to explore and how much a retained explanation should contribute.

Relation-conditioned representation and selection. Each path’s relation embedding conditions low-rank mixtures for keys and values,

$$
k _ { i k } ^ { \ell } = W _ { K } ^ { \ell } ( r _ { i k } ) z _ { i k } , \qquad v _ { i k } ^ { \ell } = W _ { V } ^ { \ell } ( r _ { i k } ) z _ { i k } .\tag{9}
$$

For the pooled relation representation $r _ { i k }$ and $B _ { r }$ bases, the key projection is

$$
\begin{array} { c } { { \omega _ { i k } ^ { K } = \mathrm { s o f t m a x } ( W _ { \omega } ^ { K } r _ { i k } + b _ { \omega } ^ { K } ) , } } \\ { { { } } } \\ { { W _ { K } ( r _ { i k } ) = W _ { K } ^ { ( 0 ) } + \displaystyle \sum _ { b = 1 } ^ { B _ { r } } \omega _ { i k , b } ^ { K } U _ { K , b } V _ { K , b } ^ { \top } . } } \end{array}\tag{10}
$$

Here $U _ { K , b } , V _ { K , b } ~ \in ~ \mathbb { R } ^ { d \times r }$ give rank-� updates. The value branch has separate mixture weights and bases, and the token query is $\mathbf { q } _ { t i } ^ { \ell } = W _ { O } ^ { \ell } h _ { t i } ^ { \ell }$ . Sharing the bases across paths limits parameters while allowing diferent relation compositions to change the features used for matching and the evidence passed onward. At evidenceattention layer ℓ, token � selects paths using

$$
\alpha _ { t i k } ^ { \ell } = \mathrm { s o f t m a x } _ { k } \left( \frac { ( \mathbf { q } _ { t i } ^ { \ell } ) ^ { \top } k _ { i k } ^ { \ell } } { \sqrt { d } } + \lambda _ { \ell } c _ { q } \log ( p _ { i k } + \epsilon ) \right) .\tag{11}
$$

The dot product supplies language–memory compatibility, while the second term introduces schema compatibility. The resulting context is fused through a learned gate:

$$
\begin{array} { r l } & { \displaystyle \boldsymbol { c } _ { t i } ^ { \ell } = \sum _ { k } \alpha _ { t i k } ^ { \ell } \boldsymbol { v } _ { i k } ^ { \ell } , } \\ & { \displaystyle \boldsymbol { g } _ { t i } ^ { \ell } = \sigma \Big ( W _ { g } ^ { \ell } [ h _ { t i } ^ { \ell } ; \boldsymbol { c } _ { t i } ^ { \ell } ; h _ { S } ; h _ { a _ { i } } ] \Big ) , } \\ & { \displaystyle h _ { t i } ^ { \ell + 1 } = \mathrm { L N } \big ( h _ { t i } ^ { \ell } + \mathrm { D r o p o u t } ( g _ { t i } ^ { \ell } \odot \boldsymbol { c } _ { t i } ^ { \ell } ) \big ) . } \end{array}\tag{12}
$$

Here $h _ { a _ { i } }$ is the pooled candidate-text embedding; ℎ� projects the encoding of the serialized schemas and their probabilities. The ref erence model stacks two such layers after the backbone’s returned hidden states. Relation-conditioned projections determine evidence representation; the prior determines relative attention. The gate modulates evidence without guaranteeing its validity.

Matrix computation. For a candidate with � retained paths, stack their memories in $Z _ { i } ~ \in ~ \mathbb { R } ^ { K \times d }$ and � token states in $\mathbf { \bar { \boldsymbol { H } } } _ { i } \in \mathbb { R } ^ { T \times d }$ Relation-conditioned projections produce $K _ { i } , V _ { i } \in \mathbb { R } ^ { K \times d }$ , and $Q _ { i } =$ $H _ { i } W _ { Q } ^ { \top }$ . The operation in the left panel of Figure 2 is

$$
\begin{array} { r l r } & { } & { A _ { i } = \mathrm { s o f t m a x } _ { \mathrm { r o w } } \displaystyle \left( \frac { Q _ { i } K _ { i } ^ { \top } } { \sqrt { d } } + 1 _ { T } \eta _ { i } ^ { \top } \right) , } \\ & { } & { \eta _ { i k } = \lambda c _ { q } \log ( { \ p _ { i k } + \epsilon } ) , \qquad C _ { i } = A _ { i } V _ { i } . } \end{array}\tag{13}
$$

The semantic bias is shared across token rows, while the query–key term varies with the token. Thus a common question-level schema preference coexists with token-specific evidence selection. In the imatinib example, the two columns correspond to the target-protein and contraindication paths; the matrices show the computation, not measured attention values. A stronger path prior can favor the former without forcing all tokens to attend identically.

## 3.5 Answer Scoring and Absolute Support

Let $\bar { h } _ { i }$ denote the mean updated token states. We combine the attended question, candidate, and original question representations with normalized multi-path support:

$$
\begin{array} { r l } & { b _ { i } = \displaystyle \log \sum _ { k = 1 } ^ { | \Pi _ { i } | } e ^ { u _ { i k } } - \log | \Pi _ { i } | , } \\ & { ~ s _ { i } = \displaystyle \mathrm { M L P } ( [ \bar { h } _ { i } ; h _ { a _ { i } } ; h _ { q } ] ) + \mu b _ { i } , } \\ & { \pi _ { i } ^ { * } = \displaystyle \arg \operatorname* { m a x } _ { \pi _ { i k } \in \Pi _ { i } } u _ { i k } . } \end{array}\tag{14}
$$

We use $b _ { i }$ for scalar candidate support to distinguish it from the value vector $v _ { i k } ^ { \ell }$ . Explanations are selected by $u _ { i k }$ , not by aggregating attention weights.

Normalized attention only compares paths within a candidate. Adding a common constant � to their scores leaves $\phi _ { i k }$ unchanged but changes $b _ { i }$ by �. With one path, $p _ { i 1 } = 1$ regardless of its quality. The absolute-support term therefore retains information about uniformly weak evidence that the attention normalization removes. Log-mean-exp is invariant to duplicating the entire path multiset, but not to selectively duplicating a strong path; it does not establish evidence independence.

Interpreting a returned path. The explanation identifies the most compatible retained path; the candidate score also uses language features and aggregate support. A correct endpoint can therefore score highly while its best path is invalid. Reporting the endpoint, selected path, and independent validity label exposes this failure; attention or within-pool compatibility alone cannot establish support.

## 3.6 Training and Output Protocols

For candidate labels $y _ { i } = 1 [ a _ { i } \in \mathcal { A } _ { q } ]$ and training logits $\widetilde { s _ { i } } ,$ the reference objective combines binary cross-entropy, positive-set

ranking, and probability calibration:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { B C E } } + 0 . 5 \mathcal { L } _ { \mathrm { l i s t } } + 0 . 1 \mathcal { L } _ { \mathrm { B r i e r } } , } \\ { \mathcal { L } _ { \mathrm { l i s t } } = \log \displaystyle \sum _ { i } e ^ { \widetilde { s } _ { i } } - \log \displaystyle \sum _ { i : y _ { i } = 1 } e ^ { \widetilde { s } _ { i } } , } \\ { \mathcal { L } _ { \mathrm { B r i e r } } = | C _ { q } | ^ { - 1 } \displaystyle \sum _ { i } ( \sigma ( \widetilde { s _ { i } } ) - y _ { i } ) ^ { 2 } . } \end{array}\tag{15}
$$

The listwise term is omitted when no positive candidate is retrieved; questions without candidate paths are skipped. Search remains discrete. The loss combines candidate discrimination, ranking over the available candidates, and probability calibration. The scoring and output interfaces use the candidate representation and path support defined above.

Output interfaces. Ranking sorts candidates by �<sub>�</sub> and attaches $\pi _ { i } ^ { * }$ to each returned endpoint. Multi-answer decoding applies a sigmoid-score threshold selected on development macro F1. PrimeKGQA additionally returns directed relation outputs; MedReason maps candidate scores to answer options or normalized answer strings. Appendix Section A.3 specifies these task interfaces.

## 4 Experiments

## 4.1 Setup and Evaluation

Datasets and baselines. We study biomedical reasoning using BioHopR, PrimeKGQA, and MedReason. The original BioHopR release contains 10,127 unique questions, including 2,494 one-hop and 7,633 two-hop questions with multiple reference answers [3]. PrimeKGQA contains 83,999 question–answer pairs accompanied by SPARQL queries [8]. MedReason contains 32,682 medical question–answer pairs with step-by-step reasoning explanations [13]. These figures describe the released resources. Task-specific output handling is specified in Appendix A.3. Baselines cover prompting, retrieval augmentation, biomedical encoders, graph neural reasoning, and KG exploration (Table 1). Controlled variants use fixed candidates, one hard schema, or the full soft-schema design.

Configuration. The reported experiments use Llama-3.1-8B-Instruct and five seeds on a workstation with four 24-GB RTX 4090 GPUs. The search setting uses $N = 8 , M = 4 , B = 3 2$ , at most three hops, and $K = 8$ . Appendix Section C gives the model configuration and seed protocol.

Answer and evidence metrics. BioHopR uses H@1/5/10, P@10, R@10, and MRR; PrimeKGQA uses answer-set EM/F1/recall and relation F1; MedReason uses multiple-choice accuracy/MRR and open-answer EM/token F1. Coverage requires at least one gold entity in the retrieved pool; candidate recall averages the fraction of each gold answer set retrieved. On a common annotated cohort, let $A _ { q }$ denote top-answer correctness and $V _ { q }$ the validity of its selected supporting path. Then

$$
\begin{array} { r l } & { \mathrm { V a l i d P a t h } @ \boldsymbol { 1 } = \mathbb { E } _ { q } [ V _ { q } ] , \qquad \mathrm { J o i n t } @ \boldsymbol { 1 } = \mathbb { E } _ { q } [ A _ { q } V _ { q } ] , } \\ & { \qquad \mathrm { W P R } = \frac { \sum _ { q } { A } _ { q } ( 1 - V _ { q } ) } { \sum _ { q } { A } _ { q } } . } \end{array}\tag{16}
$$

A missing path has $V _ { q } = 0 . \mathrm { W P R }$ is undefined when no top-ranked answer is correct. For each run, H@1, ValidPath@1, Joint@1, and WPR use the same evaluation questions and path-validity labels;

consequently, Joint@1 = H@1(1 − WPR). Appendix Section A.2 gives the ranking and aggregation conventions.

## 4.2 Answer Quality Across Datasets

Table 1 reports the strongest answer performance for the softschema model across all listed metrics. On BioHopR, it improves H@1 from 0.719 to 0.772 and R@10 from 0.556 to 0.648 over Fixed Candidate ReSemAttn. The larger recall gain is consistent with improved candidate discovery. PrimeKGQA answer F1 rises from 0.661 with a hard schema to 0.761 with soft schemas; MedReason reaches 0.691 multiple-choice accuracy and 0.749 open-answer F1. BioHopR emphasizes recovery of multiple gold entities, PrimeKGQA eval uates answer sets and relations, and MedReason separates option selection from open-answer overlap. Gains across these formats support relation-semantic evidence; absolute metric values are not comparable across datasets. Appendix Section E.7 retains all precision, recall, relation-F1, and open-answer metrics. Baseline config urations and adaptations are described in Appendix D.

Variation across seeds. The five-seed BioHopR H@1 means and standard deviations are 0.772 ± 0.006 for Full, 0.748 ± 0.007 for Hard-Schema, and $0 . 7 1 9 \pm 0 . 0 0 8$ for Fixed-Candidate. The corresponding Joint@1 values are $0 . 6 0 2 \pm 0 . 0 0 7 , 0 . 5 4 8 \pm 0 . 0 0 9$ , and $0 . 4 9 5 \pm 0 . 0 1 0$ These summaries describe run variation; aggregate means alone do not establish paired significance. Appendix Section C.2 gives the seed-summary protocol.

## 4.3 Core Components Across Datasets

Table 2 separates the core components. Removing relation-guided search lowers BioHopR H@1 by 5.3 percentage points; removing the attention prior lowers it by 6.6 points, the largest decrease among these core ablations. Both relation-conditioned projections and soft schemas contribute across datasets. Aggregation also improves answer-set quality, consistent with combining evidence for multiple answers.

Grounding and search determine available evidence; projections and the prior determine how it enters the representation. Hard schemas afect both stages. Because end-to-end ablations may change path pools, they show component sensitivity; frozen-pool comparisons isolate efects on identical evidence.

## 4.4 Detailed BioHopR Ablation

Figure 3 reports all seven metrics: H@1/5/10 assess answer ranking, P@10 and R@10 assess answer-set recovery, MRR emphasizes the first correct answer, and Joint@1 additionally requires valid evidence. Their joint interpretation separates endpoint ranking from evidence selection.

Availability and evidence factors. V1–V4 vary grounding, search, and schema uncertainty; V5–V8 remove schema-matching factors. Declines without relation compatibility or path consistency show that endpoint type is insuficient. V9–V10 isolate the attention prior and relation-conditioned projections within the full pipeline. These end-to-end changes complement, rather than replace, matched-pool controls.

Combining paths. V11 removes aggregation. The aggregation comparison reports H@1/R@10 of 0.772/0.648 for normalized Log-SumExp, 0.754/0.619 for max pooling, and 0.739/0.607 for mean pooling. Normalized aggregation improves answer-set recovery here without establishing independence of paths sharing edges. Appendix Section E.3 retains all variants, including unnormalized LogSumExp and learned gated pooling.

## 4.5 Correct Endpoints and Valid Evidence

The reachable-but-invalid challenge compares real KG paths containing relation swaps, direction errors, bridge confounders, or right-answer/wrong-path alternatives. Paths are matched by hop count, endpoint type, degree bucket, and initial retrieval-score bucket. Blinded annotation judges the complete question–path relation, rather than agreement with the model’s predicted schema; construction and agreement measurements are retained in Appendix Section D.

Two annotators label each path Valid, Invalid, or Insuficient/Ambiguous; a third adjudicates disagreements. They see the question, candidate, directed path, relation definitions, and available provenance, with model identity, scores, and predicted schemas hidden. Reported agreement is 0.884 overall $( \kappa = 0 . 7 6 1 )$ and 0.901 on correct-answer paths $\left( \kappa = 0 . 7 9 2 \right)$ . The latter subset directly tests whether endpoint correctness masks disagreement about the evidence. A legal inverse traversal is not itself an invalid direction reversal.

In the running imatinib case, the two paths share the CML endpoint but express diferent relation compositions: the ABL1 route follows the requested target-protein relation, whereas the contraindication route does not.

Figure 4 separates availability from evidence quality. With the same Cross-Encoder, relation-guided search raises Coverage from 0.861 to 0.912 and H@1 from 0.641 to 0.694. Fixed-Candidate ReSemAttn reaches H@1 of 0.719 with Coverage of 0.752, illustrating how strong ranking can coexist with a limited candidate pool. The full model reports Coverage 0.948, H@1 0.772, ValidPath@1 0.634, and Joint@1 0.602. Because the compared systems do not share identical pools, these results motivate the frozen-pool control rather than replacing it.

Figure 5 separates candidate discovery from ranking (Coverage– H@1), then incorporates evidence validity (ValidPath@1–Joint@1). Method IDs match the heatmap. Path-free methods remain in the answer comparison but are omitted from path metrics; missing values are not zeros.

On same-answer valid/invalid pairs, PairAcc requires a strictly higher within-model score for the valid path; ties receive no credit. Reported values are 0.493 for BFS + CE, 0.658 for Fixed-Candidate ReSemAttn, 0.708 for Hard-Schema, and 0.806 for Full. Holding the gold endpoint fixed isolates path preference. Appendix Section D describes the negative-path construction.

Where evidence remains dificult. Among 125 manually categorized failures, 27 involve search pruning, 23 a missing top-� schema, and 19 grounding. These three categories account for 55.2% of the analyzed failures and locate bottlenecks in candidate discovery. Appendix Section F gives the complete category counts.

Robustness beyond the standard questions. Template-disjoint H@1/Joint@1 is 0.741/0.563, versus 0.772/0.602 on the standard split; combined paraphrase and alias perturbations yield 0.731/0.548.

<table><tr><td></td><td colspan="6">BioHopR</td><td colspan="2">PrimeKGQA</td><td colspan="2">MedReason</td></tr><tr><td>Method</td><td>H@1</td><td>H@5 H@10 P@10</td><td></td><td>R@10</td><td>MRR</td><td>EM</td><td>F1</td><td>Rec.</td><td></td><td>MCQ Acc. Open F1</td></tr><tr><td>Zero-shot</td><td>0.221 0.352</td><td>0.411</td><td>0.176</td><td>0.286</td><td>0.294</td><td>0.342</td><td>0.487</td><td>0.522</td><td>0.512</td><td>0.538</td></tr><tr><td>CoT</td><td>0.468</td><td>0.474 0.515</td><td>0.313</td><td>0.367</td><td>0.486</td><td>0.368</td><td>0.516</td><td>0.548</td><td>0.548</td><td>0.566</td></tr><tr><td>Dense RAG</td><td>0.403</td><td>0.558 0.606</td><td>0.337</td><td>0.421</td><td>0.471</td><td>0.394</td><td>0.543</td><td>0.579</td><td>0.563</td><td>0.581</td></tr><tr><td>GraphRAG</td><td>0.548</td><td>0.641 0.674</td><td>0.402</td><td>0.478</td><td>0.601</td><td>0.427</td><td>0.574</td><td>0.612</td><td>0.586</td><td>0.604</td></tr><tr><td>GNN-RAG</td><td>0.587</td><td>0.672 0.708</td><td>0.425</td><td>0.501</td><td>0.635</td><td>0.444</td><td>0.591</td><td>0.629</td><td>0.604</td><td>0.621</td></tr><tr><td>BioLinkBERT</td><td>0.579</td><td>0.661 0.697</td><td>0.420</td><td>0.491</td><td>0.624</td><td>0.443</td><td>0.589</td><td>0.627</td><td>0.581</td><td>0.599</td></tr><tr><td>PullNet</td><td>0.574 0.604</td><td>0.631</td><td>0.414</td><td>0.493</td><td>0.596</td><td>0.451</td><td>0.596</td><td>0.632</td><td>0.591</td><td>0.609</td></tr><tr><td>GreaseLM</td><td>0.626 0.702</td><td>0.745</td><td>0.451</td><td>0.522</td><td>0.684</td><td>0.476</td><td>0.617</td><td>0.653</td><td>0.611</td><td>0.629</td></tr><tr><td>UniKGQA</td><td>0.643 0.709</td><td>0.748</td><td>0.466</td><td>0.536</td><td>0.696</td><td>0.488</td><td>0.628</td><td>0.664</td><td>0.620</td><td>0.638</td></tr><tr><td>Think-on-Graph</td><td>0.655 0.718</td><td>0.756</td><td>0.472</td><td>0.542</td><td>0.704</td><td>0.495</td><td>0.634</td><td>0.671</td><td>0.626</td><td>0.644</td></tr><tr><td>Fixed-Candidate ReSemAttn</td><td>0.719 0.736</td><td>0.743</td><td>0.582</td><td>0.556</td><td>0.726</td><td>0.508</td><td>0.642</td><td>0.678</td><td>0.637</td><td>0.655</td></tr><tr><td>ReSemReason w/ Hard Schema</td><td>0.748 0.789</td><td>0.811</td><td>0.598</td><td>0.621</td><td>0.765</td><td>0.531</td><td>0.661</td><td>0.696</td><td>0.652</td><td>0.670</td></tr><tr><td>ReSemReason w/ Soft Schema</td><td>0.772 0.816</td><td>0.842</td><td>0.617</td><td>0.648</td><td>0.790</td><td>0.652</td><td>0.761</td><td>0.794</td><td>0.691</td><td>0.749</td></tr></table>

Table 1: Main results across three datasets. Bold denotes the best reported value in each column.

<table><tr><td rowspan="2">Variant</td><td colspan="2">BioHopR</td><td colspan="3">PrimeKGQA</td><td colspan="2">MedReason</td></tr><tr><td>H@1 MRR</td><td></td><td></td><td></td><td></td><td></td><td>Ans. EM Ans. F1 Rel. F1 MCQ Acc. Open F1</td></tr><tr><td>w/o Grounding Distribution</td><td>0.742</td><td>0.764</td><td>0.521</td><td>0.651</td><td>0.708</td><td>0.642</td><td>0.659</td></tr><tr><td>w/o Relation-Guided Search</td><td>0.719</td><td>0.744</td><td>0.503</td><td>0.633</td><td>0.684</td><td>0.631</td><td>0.648</td></tr><tr><td>w/ Hard Question Schema</td><td>0.748</td><td>0.765</td><td>0.531</td><td>0.661</td><td>0.714</td><td>0.652</td><td>0.670</td></tr><tr><td>w/o Schema Entropy Gate</td><td>0.758</td><td>0.776</td><td>0.541</td><td>0.671</td><td>0.729</td><td>0.661</td><td>0.680</td></tr><tr><td>w/o Attention-Prior Injection</td><td>0.706</td><td>0.739</td><td>0.494</td><td>0.625</td><td>0.675</td><td>0.620</td><td>0.641</td></tr><tr><td>w/o Relation-Conditioned KV</td><td>0.731</td><td>0.756</td><td>0.514</td><td>0.644</td><td>0.694</td><td>0.638</td><td>0.657</td></tr><tr><td>w/o Candidate Path Aggregation</td><td>0.745</td><td>0.768</td><td>0.526</td><td>0.655</td><td>0.706</td><td>0.648</td><td>0.666</td></tr><tr><td>Full ReSemReason</td><td>0.772</td><td>0.790</td><td>0.652</td><td>0.761</td><td>0.770</td><td>0.691</td><td>0.749</td></tr></table>

Table 2: Core ablation across the three datasets.

![](images/d34b2b440c7a1815693dd637d8052bd627d8deed71e9f6c7045c590e37ae5dcc.jpg)

![](images/9d3fca9468616bae75d4740d3e6c5be9b5c50d0726c0e8c3dd6ac832495cdaae.jpg)

![](images/055611ccf3035fb9b876e86a1ef986c68bfa0f51886c5abf60c9808968f3a354.jpg)

![](images/c4df04480778f5869a7d20cc1c6a290f68ad087fa66a819b4981dff014898c68.jpg)

![](images/5d75ffaabfa31aa8e7615699bad9913ed04806fa032f823687823e00b94cb15f.jpg)

![](images/f4f35316a6dd49353a735e1b418053093dd38749ffe8d106ffab89921a0c8326.jpg)

![](images/130e81c1478ea7fcdb93bf147262530d46a23f7aac90ab1847306d85f16d7dfb.jpg)  
Figure 3: Per-metric ablation results on BioHopR. V1–V11 denote the eleven ablated variants, and Full denotes the complete ReSemReason model. V1 = w/o Grounding Distribution V2 = w/o Relation-Guided Search V3 = w/ Hard Question Schema V4 = w/o Schema Entropy Gate V5 = w/o Entity Anchoring V6 = w/o Relation Compatibility $\mathbf { V } \mathbf { } \mathbf { } 7 \mathbf { \Psi } = \mathbf { w } / \mathbf { o }$ Path Consistency $\mathbf { V 8 } = \mathbf { w } / \mathbf { o }$ Topology Penalty V9 = w/o Attention Prior V10 = w/o Relation-Conditioned KV V11 = w/o Path Aggregation. Higher is better for all metrics.

These results expose sensitivity to wording and grounding. Template separation does not by itself establish generalization to unseen relation compositions.

With 10% type-compatible distractors, Coverage/H@1/Joint@1 falls from 0.948/0.772/0.602 to 0.931/0.748/0.567, consistent with stronger evidence competition as well as candidate loss. Removing 10% of non-gold edges yields 0.934/0.754/0.579; it does not test loss of required gold evidence.

## 4.6 Mechanism Controls

The existing prior-by-KV factorial experiment reports H@1/ MRR/Joint@1 of 0.772/0.790/0.602 with both mechanisms and 0.681/0.716/0.462 with neither; all four settings remain in Appendix Section E.2. This pattern supports complementary contributions but does not establish a statistical interaction.

Schema prediction is imperfect: reported accuracy is 0.912 for hop count, 0.847 for relation sequences, and 0.781 for full typed schemas, with expected calibration error 0.028, 0.041, and 0.057. These results motivate preserving alternatives and evaluating confidence separately from top-� coverage.

E1. Frozen candidate/path pools
<table><tr><td>Retriever</td><td>Reasoner</td><td>Coverage</td><td>CandRec</td><td>H@1</td><td>Cond. H@1</td><td>ValidPath@1</td><td>Joint@1</td></tr><tr><td>Unguided</td><td>CE</td><td>0.860</td><td>0.691</td><td>0.639</td><td>0.743</td><td>0.520</td><td>0.462</td></tr><tr><td>Unguided</td><td>ReSemAttn</td><td>0.860</td><td>0.691</td><td>0.701</td><td>0.815</td><td>0.646</td><td>0.597</td></tr><tr><td>Relation-guided</td><td>CE</td><td>0.947</td><td>0.792</td><td>0.692</td><td>0.731</td><td>0.585</td><td>0.541</td></tr><tr><td>Relation-guided</td><td>ReSemAttn</td><td>0.947</td><td>0.792</td><td>0.766</td><td>0.809</td><td>0.714</td><td>0.681</td></tr></table>

E2. Relation integration
<table><tr><td>Integration</td><td>H@1</td><td>PairAcc</td><td>Joint@1</td></tr><tr><td>Labels in text</td><td>0.696</td><td>0.625</td><td>0.545</td></tr><tr><td>Final-score only</td><td>0.718</td><td>0.696</td><td>0.579</td></tr><tr><td>Prior + shared KV</td><td>0.744</td><td>0.763</td><td>0.634</td></tr><tr><td>Prior + relation KV</td><td>0.766</td><td>0.819</td><td>0.681</td></tr></table>

E5. Absolute support
<table><tr><td>Variant</td><td>Normal H@1</td><td>Invalid FSR↓</td><td>One-path FSR↓</td></tr><tr><td>Without  $\mu b _ { i }$ </td><td>0.758</td><td>0.407</td><td>0.456</td></tr><tr><td>Full</td><td>0.766</td><td>0.169</td><td>0.203</td></tr></table>

Table 3: Consolidated mechanism controls. Cond. H@1 conditions on a retrieved gold answer; PairAcc compares same-answer valid/invalid paths. FSR is acceptance of an all-invalid candidate pool; the one-path condition also uses invalid evidence.

<table><tr><td rowspan=5 colspan=1>1 Dense RAG2 GraphRAG3 GNN-RAG4 PullNet5 GreaseLM</td><td rowspan=1 colspan=1>0.738</td><td rowspan=1 colspan=1>0.403</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>0.824</td><td rowspan=1 colspan=1>0.548</td><td rowspan=1 colspan=1>0.351</td><td rowspan=1 colspan=1>0.469</td><td rowspan=1 colspan=1>0.291</td></tr><tr><td rowspan=1 colspan=1>0.851</td><td rowspan=1 colspan=1>0.587</td><td rowspan=1 colspan=1>0.387</td><td rowspan=1 colspan=1>0.443</td><td rowspan=1 colspan=1>0.327</td></tr><tr><td rowspan=1 colspan=1>0.836</td><td rowspan=1 colspan=1>0.574</td><td rowspan=1 colspan=1>0.369</td><td rowspan=1 colspan=1>0.456</td><td rowspan=1 colspan=1>0.312</td></tr><tr><td rowspan=1 colspan=1>0.842</td><td rowspan=1 colspan=1>0.626</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>6 UniKGQA</td><td rowspan=1 colspan=1>0.867</td><td rowspan=1 colspan=1>0.643</td><td rowspan=1 colspan=1>0.441</td><td rowspan=1 colspan=1>0.398</td><td rowspan=1 colspan=1>0.387</td></tr><tr><td rowspan=1 colspan=1>7 Think-on-Graph</td><td rowspan=1 colspan=1>0.882</td><td rowspan=1 colspan=1>0.655</td><td rowspan=1 colspan=1>0.463</td><td rowspan=1 colspan=1>0.373</td><td rowspan=1 colspan=1>0.411</td></tr><tr><td rowspan=1 colspan=1>8 BFS + CE</td><td rowspan=1 colspan=1>0.861</td><td rowspan=1 colspan=1>0.641</td><td rowspan=1 colspan=1>0.438</td><td rowspan=1 colspan=1>0.404</td><td rowspan=1 colspan=1>0.382</td></tr><tr><td rowspan=1 colspan=1>9 Rel.-Guided + CE</td><td rowspan=1 colspan=1>0.912</td><td rowspan=1 colspan=1>0.694</td><td rowspan=1 colspan=1>0.501</td><td rowspan=1 colspan=1>0.353</td><td rowspan=1 colspan=1>0.449</td></tr><tr><td rowspan=3 colspan=1>10 Fixed-Candidate11 Hard-Schema12 ReSemReason</td><td rowspan=1 colspan=1>0.752</td><td rowspan=1 colspan=1>0.719</td><td rowspan=1 colspan=1>0.542</td><td rowspan=1 colspan=1>0.312</td><td rowspan=1 colspan=1>0.495</td></tr><tr><td rowspan=1 colspan=1>0.931</td><td rowspan=1 colspan=1>0.748</td><td rowspan=1 colspan=1>0.586</td><td rowspan=1 colspan=1>0.267</td><td rowspan=1 colspan=1>0.548</td></tr><tr><td rowspan=1 colspan=1>0.948</td><td rowspan=1 colspan=1>0.772</td><td rowspan=1 colspan=1>0.634</td><td rowspan=1 colspan=1>0.220</td><td rowspan=1 colspan=1>0.602</td></tr></table>

Figure 4: Answer availability and evidence quality. Rows 1– 12 identify methods; CE denotes Cross-Encoder. Higher is better except for WPR. A dash denotes an unreported path metric.  
![](images/145616d106a51b12345d3dcb3419e7e8a5f3280324f801bdf8d0d592c5620b76.jpg)

![](images/e77b8542f33e71aa3704a43d7645a2650cb5236bc095a91f332b87b1306832a7.jpg)  
Figure 5: Candidate availability and evidence quality. Numeric method IDs follow Figure 4; methods 1 (Dense RAG) and 5 (GreaseLM) do not return paths and are omitted on the right.

Table 3 presents values for three matched-control designs. E1 freezes each retriever's candidate/path pool before either reasoner is applied, so Coverage and candidate recall are shared within a retriever block. E2 fixes relation-guided pools and varies where relation information enters, using same-answer valid/invalid pairs for PairAcc. E5 removes only the absolute-support term and evaluates both normal and all-invalid pools, including a one-path condition. These controls separate retrieval availability, relation-conditioned representation, and absolute support.

## 4.7 Eficiency

Profiling uses one RTX 4090, batch size one, 50 warm-up and 500 timed queries, excluding loading. Full ReSemReason uses 276 ms/query and 760 expanded nodes (Table 4). Appendix Section G relates the measured end-to-end latency to the computational structure.

<table><tr><td>Mean</td><td colspan="3">Peak Expanded</td></tr><tr><td>Method</td><td>ms/q↓</td><td>GB↓ Nodes↓</td><td>H@1↑</td></tr><tr><td>Dense RAG</td><td>118</td><td>8.6</td><td>一 0.403</td></tr><tr><td>GraphRAG</td><td>242</td><td>10.1 1280</td><td>0.548</td></tr><tr><td>GNN-RAG</td><td>286</td><td>11.8 1460</td><td>0.587</td></tr><tr><td>PullNet</td><td>218</td><td>9.7 1320</td><td>0.574</td></tr><tr><td>GreaseLM</td><td>171</td><td>12.5</td><td>一 0.626</td></tr><tr><td>UniKGQA</td><td>205</td><td>10.9</td><td>980 0.643</td></tr><tr><td>Think-on-Graph</td><td>514</td><td>14.2</td><td>1730 0.655</td></tr><tr><td>Full ReSemReason</td><td>276</td><td>19.8</td><td>760 0.772</td></tr></table>

Table 4: End-to-end eficiency on BioHopR. All systems are evaluated with batch size one on identical hardware. A dash indicates that expanded-node counts are not directly applicable or exposed.

For � candidates, � paths each, � tokens, and hidden width �, token-to-path attention costs �(����) per layer, excluding backbone encoding and projections. Appendix Section G accounts for these costs and distinguishes edge attempts from unique-node expansions.

Increasing beam width from 8 to 32 raises Coverage from 0.883   
to 0.948 and H@1 from 0.721 to 0.772, while latency rises from 188

to 276 ms. Beam 64 gives Coverage 0.956 and H@1 0.776 at 391 ms, showing diminishing returns from additional search in this configuration.

## 5 Limitations

Missing or ambiguous KG relations, incomplete grounding, omitted schemas, and beam pruning can prevent valid evidence from reaching the reasoner. Mean-pooled relation features do not them selves preserve relation order, and the topology penalty is not a provenance-quality model. Retrieved paths and attention weights are inspectable model evidence, not causal explanations or clinical validation.

## 6 Conclusion

ReSemReason makes relation composition explicit by sharing question-induced schemas between candidate discovery and path attention. Relation-conditioned projections encode evidence, the prior steers selection, and absolute support preserves information lost by normalization. Reported answer, ablation, and path-discrimination results motivate evaluating availability and validity separately. The separation between candidate availability, answer correctness, and supporting-path validity provides a basis for evaluating biomedical multi-hop reasoning.

## References

[1] Payal Chandak, Kexin Huang, and Marinka Zitnik. 2023. Building a knowledge graph to enable precision medicine. Scientific Data 10, 1 (2023), 67. doi:10.1038/ s41597-023-01960-3

[2] Jinhao Jiang, Kun Zhou, Wayne Xin Zhao, and Ji-Rong Wen. 2023. UniKGQA: Unified Retrieval and Reasoning for Solving Multi-hop Question Answering Over Knowledge Graph. In The Eleventh International Conference on Learning Representations.

[3] Yunsoo Kim, Yusuf Abdulle, and Honghan Wu. 2025. BioHopR: A Benchmark for Multi-Hop, Multi-Answer Reasoning in Biomedical Domain. In Findings ofthe Association for Computational Linguistics: ACL 2025. Association for Computational Linguistics, Vienna, Austria, 12894–12908. doi:10.18653/v1/2025.findings-acl.668

[4] Yangning Li, Weizhi Zhang, Yuyao Yang, Wei-Chieh Huang, Yaozu Wu, Junyu Luo, Yuanchen Bei, Henry Peng Zou, Xiao Luo, Yusheng Zhao, Chunkit Chan, Yankai Chen, Zhongfen Deng, Yinghui Li, Hai-Tao Zheng, Dongyuan Li, Renhe Jiang, Ming Zhang, Yangqiu Song, and Philip S. Yu. 2025. A Survey of RAG-Reasoning Systems in Large Language Models. In Findings ofthe Association for Computational Linguistics: EMNLP 2025. Association for Computational Linguis tics, 12120–12145. doi:10.18653/v1/2025.findings-emnlp.648

[5] Linhao Luo, Yuan-Fang Li, Reza Hafari, and Shirui Pan. 2024. Reasoning on Graphs: Faithful and Interpretable Large Language Model Reasoning. In The Twelfth International Conference on Learning Representations. https://openreview. net/forum?id=ZGNWW7xZ6Q

[6] Linhao Luo, Zicheng Zhao, Gholamreza Hafari, Yuan-Fang Li, Chen Gong, and Shirui Pan. 2025. Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models. In Proceedings ofthe 42nd International Conference on Machine Learning (Proceedings of Machine Learning Research, Vol. 267). PMLR, 41540–41565. https://proceedings.mlr.press/v267/ luo25t.html

[7] Costas Mavromatis and George Karypis. 2025. GNN-RAG: Graph Neural Retrieval for Eficient Large Language Model Reasoning on Knowledge Graphs. In Findings ofthe Association for Computational Linguistics: ACL 2025. Association for Computational Linguistics, 16682–16699. doi:10.18653/v1/2025.findings-acl.856

[8] PrimeKGQA Contributors. 2024. PrimeKGQA: A Biomedical Knowledge Graph Question Answering Dataset Sourced from PrimeKG. doi:10.5281/zenodo. 13829395

[9] Nihar Sanda, Benjamin M. Gyori, Vito Quaranta, Auroop Ganguly, and Ayan Paul. 2026. eGoT: Enhanced Graph-of-Thoughts for Multi-Hop Knowledge Retrieva and Hypothesis Generation in Biomedicine. Bioinformatics 42, Supplement\_1 (2026), btag216. doi:10.1093/bioinformatics/btag216

[10] Haitian Sun, Tania Bedrax-Weiss, and William W. Cohen. 2019. Open Domain Question Answering with Iterative Retrieval on Knowledge Bases and Text. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2380–2390. doi:10.18653/ v1/D19-1242

[11] Jiashuo Sun, Chengjin Xu, Luming Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel M. Ni, Heung-Yeung Shum, Jian Guo, and Nan Duan Jiang. 2024. Think-on-Graph: Deep and Responsible Reasoning ofLarge Language Model on Knowledge Graph. arXiv:2307.07697 [cs.CL]

[12] Johannes Welbl, Pontus Stenetorp, and Sebastian Riedel. 2018. Constructing Datasets for Multi-hop Reading Comprehension Across Documents. Transactions ofthe Association for Computational Linguistics 6 (2018), 287–302. doi:10.1162/ tacl\_a\_00021

[13] Juncheng Wu, Wenlong Deng, Xingxuan Li, Sheng Liu, Taomian Mi, Yifan Peng, Ziyang Xu, Yi Liu, Hyunjin Cho, Chang-In Choi, et al. 2025. MedReason: Eliciting Factual Medical Reasoning Steps in Large Language Models via Knowledge Graphs. arXiv:2504.00993 [cs.CL] https://arxiv.org/abs/2504.00993

[14] Ran Xu, Patrick Jiang, Linhao Luo, Cao Xiao, Adam Cross, Shirui Pan, Jimeng Sun, and Carl Yang. 2025. A Survey on Unifying Large Language Models and Knowl edge Graphs for Biomedicine and Healthcare. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2. Association for Computing Machinery, 6195–6205. doi:10.1145/3711896.3736556

[15] Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure Leskovec. 2021. QA-GNN: Reasoning with Language Models and Knowledge Graphs for Question Answering. In Proceedings ofthe 2021 Conference ofthe North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies. Association for Computational Linguistics, 535–546. doi:10.18653/v1/2021.naacl-main.45

[16] Xikun Zhang, Antoine Bosselut, Michihiro Yasunaga, Hongyu Ren, Percy Liang, Christopher D. Manning, and Jure Leskovec. 2022. GreaseLM: Graph REASoning Enhanced Language Models for Question Answering. In International Conference on Learning Representations. https://openreview.net/forum?id=41e9o6cQPj

## A Data, Metrics, and Outputs

A.1 Data sources. BioHopR provides 10,127 unique questions over PrimeKG, covering one-hop and two-hop multi-answer reasoning [3]. PrimeKGQA is obtained from the release identified by DOI 10.5281/zenodo.13829395, whose dataset description reports 83,999 question–answer pairs with associated SPARQL queries [8]. MedReason provides 32,682 medical question–answer pairs with reasoning explanations [13]. These are source-dataset totals, not counts of candidate entities, supporting paths, or expanded training instances.

A.2 Evaluation metrics. For canonical-ID rankings, let $n _ { q }$ be the number of returned candidates. Precision and recall are

$$
\begin{array} { r l } & { P @ K ( q ) = \frac { \vert \mathrm { T o p K } ( q ) \cap \mathcal { A } _ { q } \vert } { \operatorname* { m i n } ( K , n _ { q } ) } , } \\ & { R @ K ( q ) = \frac { \vert \mathrm { T o p K } ( q ) \cap \mathcal { A } _ { q } \vert } { \vert \mathcal { A } _ { q } \vert } . } \end{array}\tag{A1}
$$

Empty rankings and missing reciprocals score zero. Set metrics are macro-averaged across questions.

On � evaluation questions, answer and evidence metrics are

$$
\begin{array} { c } { { \displaystyle { \mathrm { H } ( \varpi 1 = \frac { 1 } { N } \sum _ { q } A _ { q } , } } } \\ { { \displaystyle { \mathrm { V a l i d P a t h } ( \varpi 1 = \frac { 1 } { N } \sum _ { q } V _ { q } , } } } \\ { { \displaystyle { \mathrm { J o i n t @ 1 } ( \varpi 1 = \frac { 1 } { N } \sum _ { q } A _ { q } V _ { q } , } } } \\ { { \displaystyle { \mathrm { W P R } = \frac { \sum _ { q } A _ { q } ( 1 - V _ { q } ) } { \sum _ { q } A _ { q } } } . } } \end{array}\tag{A2}
$$

Missing paths have $V _ { q } = 0 ;$ WPR is undefined when its denominator is zero. The identity in Section 4.1 holds per run:

$$
\mathrm { J o i n t } @ 1 = \mathrm { H } @ 1 \big ( 1 - \mathrm { W P R } \big ) .\tag{A3}
$$

Metrics are computed separately for each seed before reporting their means and standard deviations.

A.3 Output adapters. PrimeKGQA accepts one-variable SE-LECT chains up to three hops and compares directed relation-family multisets; branching, aggregation, OPTIONAL, and UNION form separate query categories. MedReason uses source grouping before an 80/10/10 split (seed 42), option mapping, and fixed answer normalization. Multi-answer sigmoid thresholds are selected on development macro F1; canonical IDs and path keys break ties.

## B Implementation Details

B.1 Grounding and features. Grounding is gradient-free. Path memory pools types and relation/direction embeddings with hop count, unique-node ratio, closure, search score, mean, maximum, and endpoint log-degree, and the fraction above log 10. Mean pooling loses relation order; Eq. (8) compares positions explicitly.

B.2 Schema enumeration. Independent heads’ log probabilities sum over hops, endpoint/intermediate types, relations, and directions. Enumeration permits two types per position, three rela tions per hop, both directions, and three hops. The Llama configu ration retains $M = 4$ schemas using a heap of size 32. No schema temperature is fitted.

(a) BioHopR seed means and standard deviations
<table><tr><td>Method</td><td>H@1</td><td>H@5</td><td>H@10</td><td>R@10</td><td>MRR</td><td>Joint@1</td></tr><tr><td>Fixed-Cand.</td><td> $0 . 7 1 9 \pm 0 . 0 0 8$ </td><td> $0 . 7 3 6 \pm 0 . 0 0 6$ </td><td> $0 . 7 4 3 \pm 0 . 0 0 6$ </td><td> $0 . 5 5 6 \pm 0 . 0 0 9$ </td><td> $0 . 7 2 6 \pm 0 . 0 0 7$ </td><td> $0 . 4 9 5 \pm 0 . 0 1 0$ </td></tr><tr><td>Hard-Schema</td><td> $0 . 7 4 8 \pm 0 . 0 0 7$ </td><td> $0 . 7 8 9 \pm 0 . 0 0 5$ </td><td> $0 . 8 1 1 \pm 0 . 0 0 5$ </td><td> $0 . 6 2 1 \pm 0 . 0 0 8$ </td><td> $0 . 7 6 5 \pm 0 . 0 0 6$ </td><td> $0 . 5 4 8 \pm 0 . 0 0 9$ </td></tr><tr><td>Full</td><td> $\mathbf { 0 . 7 7 2 \pm 0 . 0 0 6 }$ </td><td> $0 . 8 1 6 \pm 0 . 0 0 4$ </td><td> $0 . 8 4 2 \pm 0 . 0 0 4$ </td><td> $\pm 0 . 6 4 8 \pm 0 . 0 0 6$ </td><td> $\mathbf { 0 . 7 9 0 \pm 0 . 0 0 5 }$ </td><td> $\mathbf { 0 . 6 0 2 \pm 0 . 0 0 7 }$ </td></tr></table>

(b) Independently labeled path discrimination
<table><tr><td>Method</td><td>Rel. Swap</td><td>Direction</td><td>Bridge</td><td>Same answer</td><td>Macro</td><td>AUPRC</td></tr><tr><td> $\mathrm { B F S } + \mathrm { C E }$ </td><td>0.642</td><td>0.581</td><td>0.557</td><td>0.493</td><td>0.568</td><td>0.527</td></tr><tr><td>Guided + CE</td><td>0.716</td><td>0.668</td><td>0.641</td><td>0.587</td><td>0.653</td><td>0.604</td></tr><tr><td>Fixed-Cand.</td><td>0.751</td><td>0.729</td><td>0.706</td><td>0.658</td><td>0.711</td><td>0.672</td></tr><tr><td>Hard-Schema</td><td>0.801</td><td>0.783</td><td>0.751</td><td>0.708</td><td>0.761</td><td>0.731</td></tr><tr><td>Full</td><td>0.862</td><td>0.851</td><td>0.817</td><td>0.806</td><td>0.834</td><td>0.798</td></tr></table>

Table A1: Seed variation and path validity. Same answer denotes right-answer/wrong-path pairs; Macro is macro PairAcc. Seeds are {13, 21, 42, 87, 100}. Entries are experimental measurements.

B.3 Prefix scoring. For $J = \operatorname* { m i n } ( L , H )$ , the search mismatch is

$$
\begin{array} { l } { { \displaystyle { D _ { \mathrm { p r e f i x } } = \sum _ { j = 1 } ^ { J } \left( 1 \big [ r _ { j } \neq \bar { r } _ { j } \big ] + 0 . 7 5 1 \big [ d _ { j } \neq \bar { d } _ { j } \big ] \right. } } } \\ { { \displaystyle ~ \left. + 0 . 5 1 \big [ \tau \big ( v _ { j } \big ) \neq \bar { \tau } _ { j } \big ] \right) . } } \end{array}\tag{A4}
$$

Partial search omits the helper’s 0.5|� − �| term. Scores are recomputed, not accumulated. Completed paths are collected before beam truncation; there is no extra candidate cap, explicit deduplication, or attention-to-search feedback. Degree includes generated inverse records. The prior starts at $\gamma = 0 ,$ , so $\beta = \mathrm { s o f t p l u s } ( \gamma )$

## C Training and Reproducibility

C.1 Experimental configuration. The environment uses Python 3.10.14, PyTorch 2.3.1, Transformers 4.43.3, CUDA 12.1, cuDNN 9.1, and PrimeKG 2023-01. Experimental settings are $( N , M , B , K ) = ( 8 , 4 , 3 2 , 8 )$ , three hops, eight relation bases, rank 16, $\mu = 0 . 3 5 ,$ , learning rate $2 \times 1 0 ^ { - 5 }$ , batch 8, and five epochs on four 24-GB RTX 4090s. The optimizer is AdamW with weight decay 0.01 and gradient-norm clipping at 1.

C.2 Statistical protocol. Table A1(a) reports means and standard deviations across seeds {13, 21, 42, 87, 100}. Each seed is evaluated independently using the metric definitions in Appendix Section A.2. Standard deviations summarize variation across runs.

## D Unified Baseline Evaluation

Implementation and shared conditions. All results in Table 1 are produced under our unified protocol rather than copied from the original papers. Methods use identical dataset splits, reference answers, canonical entity IDs, and evaluation scripts. KG-based methods share PrimeKG 2023-01 and the entity-linking interface. LLMbased methods use Llama-3.1-8B-Instruct; architecture-specific baselines retain their declared encoders. The code address reported in the paper is https://anonymous.4open.science/r/ReSemAttn.

The reported traversal configuration uses eight retained anchors, a maximum of three hops, beam width 32, and at most eight paths per candidate, without an additional candidate cap. Shared resource limits do not imply identical retrieved pools.

Output handling and tuning. Methods share the task-specific output adapters in Appendix A.3, including entity canonicalization, answer normalization, option mapping, and empty-output handling. Trainable methods use the same permitted training data. The reported training configuration is AdamW with learning rate $2 \times 1 0 ^ { - 5 }$ , batch size 8, five epochs, weight decay 0.01, and gradientnorm clipping at 1. Multi-answer thresholds are selected using development macro F1. The selection procedure is shared, while selected threshold values may difer. Per-baseline hyperparameter ranges, tuning-trial counts, and selected configurations require the actual development-search records.

Controlled comparisons and uncertainty. Table 1 measures complete-system performance under our implementation and protocol. Module-level comparisons use frozen candidate/path pools in E1 and fixed-pool relation-integration variants in E2 (Table 3). In E1, replacing CE with ReSemAttn changes H@1 from 0.639 to 0.701 under unguided retrieval and from 0.692 to 0.766 under relationguided retrieval.

The reported seeds are {13, 21, 42, 87, 100}. Five-seed BioHopR H@1 is $0 . 7 7 2 \pm 0 . 0 0 6$ for Full, 0.748 ± 0.007 for Hard-Schema, and $0 . 7 1 9 \pm 0 . 0 0 8$ for Fixed-Candidate. These standard deviations describe run variation. Paired confidence intervals and adjusted �- values require matched question-level predictions and are not supplied by the reported aggregate statistics. Performance claims therefore concern the evaluated implementations under this protocol.

## E Extended Empirical Results

E.1 Search–reasoning systems. Table A2(a) compares changing pools; Table 3 E1 specifies the frozen-pool control.

E.2 Attention components. Table A2(c) favors both terms, without establishing interaction.

E.3 Aggregation. Table A2(d) favors normalized LogSumExp; selective-duplication invariance is not established. Figure 3 retains seven-metric ablations.

(a) Search and reasoning systems
<table><tr><td>Search / Reasoner</td><td>Cov.</td><td>H@1</td><td>Valid WPR↓</td><td>Joint</td></tr><tr><td>Unguided / CE</td><td>0.861</td><td>0.641 0.438</td><td>0.404</td><td>0.382</td></tr><tr><td>Guided / CE</td><td>0.912</td><td>0.694 0.501</td><td>0.353</td><td>0.449</td></tr><tr><td>Unguided / Attn</td><td>0.873</td><td>0.708 0.553</td><td>0.285</td><td>0.506</td></tr><tr><td>Guided / Attn</td><td>0.948</td><td>0.772 0.634</td><td>0.220</td><td>0.602</td></tr></table>

(c) Prior and relation-conditioned KV

(b) End-to-end cost
<table><tr><td>Variant</td><td>Nodes</td><td>Total ms</td><td>Joint@1</td></tr><tr><td>BFS + CE</td><td>1510</td><td>233</td><td>0.382</td></tr><tr><td>Rel.-G. + CE</td><td>820</td><td>171</td><td>0.449</td></tr><tr><td>Fixed-Cand.</td><td>一</td><td>201</td><td>0.495</td></tr><tr><td>Hard-Schema</td><td>690</td><td>280</td><td>0.548</td></tr><tr><td>Full</td><td>760</td><td>276</td><td>0.602</td></tr></table>

<table><tr><td>Prior</td><td>Rel.-KV</td><td>H@1</td><td>MRR</td><td>Joint@1</td></tr><tr><td>Yes</td><td>Yes</td><td>0.772</td><td>0.790</td><td>0.602</td></tr><tr><td>Yes</td><td>No</td><td>0.731</td><td>0.756</td><td>0.534</td></tr><tr><td>No</td><td>Yes</td><td>0.706</td><td>0.739</td><td>0.501</td></tr><tr><td>No</td><td>No</td><td>0.681</td><td>0.716</td><td>0.462</td></tr></table>

(d) Path aggregation
<table><tr><td>Aggregation</td><td>H@1 R@10</td><td>MRR</td><td>Joint@1</td></tr><tr><td>Max path</td><td>0.754</td><td>0.619 0.773</td><td>0.571</td></tr><tr><td>Mean path</td><td>0.739</td><td>0.607 0.759</td><td>0.552</td></tr><tr><td>LogSumExp</td><td>0.763</td><td>0.639 0.783</td><td>0.588</td></tr><tr><td>Norm. LogŠumExp</td><td>0.772</td><td>0.648 0.790</td><td>0.602</td></tr><tr><td>Gated pooling</td><td>0.768</td><td>0.644 0.786</td><td>0.595</td></tr></table>

(e) Beam width
<table><tr><td>Beam</td><td>Nodes</td><td>Cov.</td><td>H@1</td><td>Joint@1</td><td>ms/q</td></tr><tr><td>8</td><td>238</td><td>0.883</td><td>0.721</td><td>0.531</td><td>188</td></tr><tr><td>16</td><td>431</td><td>0.921</td><td>0.754</td><td>0.576</td><td>226</td></tr><tr><td>32</td><td>760</td><td>0.948</td><td>0.772</td><td>0.602</td><td>276</td></tr><tr><td>64</td><td>1378</td><td>0.956</td><td>0.776</td><td>0.606</td><td>391</td></tr></table>

(f) Schemas and path memories
<table><tr><td>M K</td><td>H@1</td><td>R@10</td><td>Joint@1</td></tr><tr><td>1</td><td>8</td><td>0.748 0.621</td><td>0.548</td></tr><tr><td>2</td><td>8 0.763</td><td>0.637</td><td>0.581</td></tr><tr><td>4</td><td>8 0.772</td><td>0.648</td><td>0.602</td></tr><tr><td>8</td><td>16 0.773</td><td>0.651</td><td>0.603</td></tr></table>

Table A2: Systems, mechanisms, and budgets on BioHopR. Attn denotes ReSemAttn; Valid/Joint are ValidPath@1/Joint@1. Pools in (a) difer between reasoners. Nodes count unique expanded nodes; total latency is measured end to end. The last row of (f) changes both � and �.

(a) Surface variation
<table><tr><td>Input / split</td><td>H@1</td><td>Joint@1</td><td>Consist.</td></tr><tr><td>Standard</td><td>0.772</td><td>0.602</td><td>1.000</td></tr><tr><td>Template-disjoint</td><td>0.741</td><td>0.563</td><td>0.883</td></tr><tr><td>Paraphrased</td><td>0.758</td><td>0.584</td><td>0.914</td></tr><tr><td>Entity aliases</td><td>0.749</td><td>0.572</td><td>0.901</td></tr><tr><td>Paraphrase + aliases</td><td>0.731</td><td>0.548</td><td>0.862</td></tr></table>

(b) Graph perturbations
<table><tr><td>Perturbation</td><td>H@1</td><td>Cov.</td><td>Joint@1</td></tr><tr><td>None</td><td>0.772</td><td>0.948</td><td>0.602</td></tr><tr><td>+5%distractors</td><td>0.763</td><td>0.942</td><td>0.588</td></tr><tr><td>+10% distractors</td><td>0.748</td><td>0.931</td><td>0.567</td></tr><tr><td>-5% non-gold edges</td><td>0.766</td><td>0.943</td><td>0.594</td></tr><tr><td>-10% non-gold edges</td><td>0.754</td><td>0.934</td><td>0.579</td></tr></table>

(c) Schema calibration
<table><tr><td>Target</td><td>Accuracy</td><td>ECE↓</td><td>Brier↓</td></tr><tr><td>Hop count</td><td>0.912</td><td>0.028</td><td>0.071</td></tr><tr><td>Relation seq.</td><td>0.847</td><td>0.041</td><td>0.109</td></tr><tr><td>Full typed schema</td><td>0.781</td><td>0.057</td><td>0.146</td></tr></table>

(d) Annotation agreement
<table><tr><td>Subset</td><td>Agreement</td><td>K</td></tr><tr><td>All paths</td><td>0.884</td><td>0.761</td></tr><tr><td>Correct-answer</td><td>0.901</td><td>0.792</td></tr><tr><td>Challenge</td><td>0.867</td><td>0.733</td></tr></table>

Table A3: Robustness, schema calibration, and annotation agreement. Distractors are type-compatible; removed edges are non-gold. Lower ECE and Brier are better.

<table><tr><td></td><td colspan="5">PrimeKGQA</td><td colspan="4">MedReason</td></tr><tr><td>Method</td><td>Ans. EM</td><td>Precision</td><td>Recall</td><td>Ans. F1</td><td>Rel. F1</td><td>MCQ Acc.</td><td>MRR</td><td>Open EM</td><td>Open F1</td></tr><tr><td>Zero-shot</td><td>0.342</td><td>0.456</td><td>0.522</td><td>0.487</td><td>0.412</td><td>0.512</td><td>0.627</td><td>0.421</td><td>0.538</td></tr><tr><td>CoT</td><td>0.368</td><td>0.491</td><td>0.548</td><td>0.516</td><td>0.447</td><td>0.548</td><td>0.654</td><td>0.446</td><td>0.566</td></tr><tr><td>Dense RAG</td><td>0.394</td><td>0.512</td><td>0.579</td><td>0.543</td><td>0.483</td><td>0.563</td><td>0.669</td><td>0.461</td><td>0.581</td></tr><tr><td>GraphRAG</td><td>0.427</td><td>0.541</td><td>0.612</td><td>0.574</td><td>0.526</td><td>0.586</td><td>0.691</td><td>0.482</td><td>0.604</td></tr><tr><td>GNN-RAG</td><td>0.444</td><td>0.559</td><td>0.629</td><td>0.591</td><td>0.548</td><td>0.604</td><td>0.707</td><td>0.501</td><td>0.621</td></tr><tr><td>GreaseLM</td><td>0.476</td><td></td><td>0.653</td><td>0.617</td><td></td><td>0.611</td><td>0.716</td><td>0.508</td><td>0.629</td></tr><tr><td>UniKGQA</td><td>0.488</td><td>0.597</td><td>0.664</td><td>0.628</td><td>0.601</td><td>0.620</td><td>0.724</td><td>0.516</td><td>0.638</td></tr><tr><td>Think-on-Graph</td><td>0.495</td><td>0.608</td><td>0.671</td><td>0.634</td><td>0.615</td><td>0.626</td><td>0.731</td><td>0.522</td><td>0.644</td></tr><tr><td>Fixed-Cand.</td><td>0.508</td><td>0.611</td><td>0.678</td><td>0.642</td><td>0.641</td><td>0.637</td><td>0.742</td><td>0.536</td><td>0.655</td></tr><tr><td>Hard-Schema</td><td>0.531</td><td>0.630</td><td>0.696</td><td>0.661</td><td>0.714</td><td>0.652</td><td>0.758</td><td>0.551</td><td>0.670</td></tr><tr><td>Full</td><td>0.652</td><td>0.731</td><td>0.794</td><td>0.761</td><td>0.770</td><td>0.691</td><td>0.797</td><td>0.624</td><td>0.749</td></tr></table>

Table A4: Additional answer-set, relation, multiple-choice, and open-answer metrics. A dash indicates an unreported result; no missing result is treated as zero.

E.4 Budgets. Table A2(e–f) shows diminishing beam gains; the final row varies both � and �.

E.5 Calibration. Table A3(c) separates hop/relation/schema prediction; confidence does not measure top-� coverage.

E.6 Robustness. Table A3(a–b) tests surface/graph perturbations, not unseen compositions or missing essential edges.

E.7 Additional task metrics. Table A4 reports additional task metrics under their respective output protocols.

## F Cases and Failure Analysis

Figures 1–2 fix CML as the endpoint and compare an ABL1 route with a contraindication route. Only the former follows the requested drug–target–disease composition. Holding the endpoint fixed separates entity relevance from relation-semantic support.

Two attention layers. The three planes show discovery and two attention layers. Equation (13) forms a 3 × 2 submatrix with diseases/proteins/targeted token rows and ABL1/contraindication path columns. Gated fusion updates $H ^ { ( 0 ) }  H ^ { ( 1 ) }  H ^ { ( 2 ) }$ , reusing paths and schemas while recomputing queries with layer-specific projections and gates. Updated tokens and absolute support determine the candidate score; arg max� �<sub>�</sub>� selects its explanation.

125 failures (count; percent). Grounding 19 (15.2), omitted schema 23 (18.4), pruned path 27 (21.6), missing edge 16 (12.8), relation ambiguity 14 (11.2), underweighted valid path 13 (10.4), thresholding 8 (6.4), annotation ambiguity 3 (2.4), and other 2 (1.6). These frequencies diagnose failure locations, not causal module efects.

## G Computational Accounting

Table A2(b) reports end-to-end latency and unique expanded nodes. Search attempts scale as �( (� + (� − 1)�) <sup>¯</sup>�) before schema scoring. Per layer, attention, dense KV, and low-rank updates cost �(����), �(���<sup>2</sup>), and �(���<sub>�</sub>��), respectively; path states require �(���) memory. Backbone encoding and memory projection add computation. Profiling uses one RTX 4090, batch one, 50 warmup queries, and 500 timed queries.