# Towards Automatic Evolution Tree Generation from Citation Graphs

Zexing Zhao<sup>1</sup>, Yuntong Hu<sup>2</sup>, Liang Zhao<sup>2†</sup>

<sup>1</sup>School of Mechanical Engineering, Georgia Institute of Technology

<sup>2</sup>Department of Computer Science, Emory University

alfred.zhao@gatech.edu, {yuntong.hu, liang.zhao}@emory.edu

## Abstract

Surveys remain the primary way researchers grasp the lineage of methods within an AI subfield, but they scale poorly against the current rate of publication. Existing taxonomyinduction methods are largely leaf-bound and time-agnostic; they tend to force transitional papers into mature leaves and can create topological inversions between ancestors and descendants. We propose EvoTree, a staged framework that decouples conceptual backbone learning from temporal refinement: a graph-aware encoder with distribution-based hierarchical clustering yields a stable taxonomy backbone; temporal fine-tuning then re-attaches marginal papers to internal nodes under monotonic-path constraints; a final LLM pass labels concepts without altering the topology. We release the first annotated benchmark for this task across 11 AI subfields. EvoTree attains the highest NMI and citation-direction accuracy among all baselines and the best concept purity on the annotated benchmark, and is the only method with non-trivial marginal-paper detection on the annotated set.

## 1 Introduction

The rapid growth of scientific literature has fundamentally challenged human-centered knowledge organization and field understanding. Open scholarly infrastructures such as OpenAlex (Priem et al., 2022) now index over 470 million scientific works together with continuously expanding citation networks. As research fields evolve at an unprecedented pace, researchers increasingly struggle to efficiently understand the historical evolution, conceptual lineage, and paradigm transitions of rapidly developing domains (Fortunato et al., 2018), creating an increasing need for AI-assisted methods to organize, synthesize, and understand scientific progress (Le et al., 2026; Hu et al., 2025).

Among various forms of scientific organization, evolution trees and hierarchical lineage diagrams are particularly effective because they explicitly reveal conceptual inheritance, branching, and progressive refinement across generations of work. Such structures are widely adopted in modern survey papers to summarize the evolution of research paradigms, model families, and emerging subfields (Zhang et al., 2018; Wang et al., 2024; Hu et al., 2024; Liang et al., 2025). However, constructing these evolution trees remains largely manual, requiring substantial domain expertise and intensive literature analysis, while quickly becoming outdated as new papers continuously emerge (Wang et al., 2024; Liang et al., 2025). This motivates a fundamental research question:

## Can we automatically generate scientific evolution trees from citation graphs?

Addressing this problem is highly challenging due to several underexplored issues. First, scientific evolution trees are hierarchical, temporal, and semantically structured objects, whereas citation graphs are large-scale sparse relational networks. Bridging these two fundamentally different structures requires learning evolution-aware representations that preserve both citation dependencies and temporal scientific progression, which is not directly addressed by existing taxonomy induction methods (Zhang et al., 2018; Shen et al., 2020; Yu et al., 2020) or temporal topic modeling approaches (Blei and Lafferty, 2006; Wang and Mc-Callum, 2006; Wang et al., 2008). Second, scientific evolution is inherently dynamic: research paradigms continuously branch, merge, and refine over time, making it difficult to construct stable and interpretable lineage structures from evolving citation graphs. Existing citation and co-citation analyses in science-of-science research primarily characterize global scientific trends rather than inducing explicit hierarchical evolutionary structures (Fortunato et al., 2018). Third, existing survey papers provide only limited supervision signals, requiring models to generalize from scarce high-quality evolution annotations. Moreover, marginal or transitional papers often lie between major paradigms, making them difficult to position within conventional taxonomy induction frameworks without introducing chronological inconsistencies.

To address those challenges, we propose a structurally staged and temporally constrained framework that unifies taxonomy induction, temporal admissibility, and marginal-paper re-attachment into a cohesive pipeline. Specifically, our framework first constructs a stable, time-agnostic core taxonomy via distribution-based hierarchical clustering on graph-aware node embeddings. Subsequently, it performs temporal refinement to model evolutionary dynamics and handle marginal papers by attaching them to broader internal concepts, while enforcing temporal consistency and structural legality constraints (e.g., path monotonicity). This staged approach ensures that the conceptual backbone remains robust while temporal dynamics are accurately and logically mapped.

Our contributions are summarized as follows:

• Novel Problem Formulation. We formulate evolutionary tree construction from citation graphs, introducing a dynamic perspective for scientific literature mining.

• Staged & Constrained Framework. We propose a staged framework that separates taxonomy initialization from temporal refinement, preserving citation-aware structures while filtering peripheral papers from core trajectories.

• Evolutionary Dynamics Modeling. We incorporate temporal and structural constraints with marginal-paper detection and internal-node reattachment, alleviating chronological distortions in traditional taxonomy generation.

• Comprehensive Benchmark. We construct the first benchmark for this task across 11 AI subfields, where extensive experiments show consistent gains over taxonomy, citation-based, and LLM baselines in evolution-structure quality and annotated concept purity, together with the only non-trivial marginal-paper detection.

## 2 Related Work

## 2.1 Automatic Taxonomy Induction

Automatic taxonomy induction studies how to organize terms, entities, or topics into hierarchical structures such as trees or DAGs. Prior work has shown that taxonomy quality requires more than isolated pairwise hypernym prediction (Snow et al., 2006; Bansal et al., 2014; Mao et al., 2018). Most existing approaches focus on taxonomy expansion or completion given an existing hierarchy (Shen et al., 2020; Yu et al., 2020; Mishra et al., 2025; Jiang et al., 2022), or extend the hierarchy formalism itself (Lee et al., 2022; Lu et al., 2024; Kargupta et al., 2025). A smaller line of work constructs topic or concept hierarchies directly from corpora. TaxoGen (Zhang et al., 2018) recursively builds topic taxonomies through adaptive spherical clustering and local embeddings, while HiExpan (Shen et al., 2018) expands user-guided seed hierarchies using weakly supervised relation extraction. A recent line of work applies language models to scholarly taxonomy construction. TaxoAlign (Lahiri et al., 2025) generates taxonomies for scientific domains by aligning LM-induced hierarchies to reference structures, and Zhu et al. (2025) induce hierarchical taxonomies for scientific papers via LLM-guided multi-aspect clustering. However, these methods are primarily designed for text corpora and cannot effectively model the structural dependencies. Moreover, they generally assume static taxonomies and therefore fail to capture temporally constrained scientific evolution, including paradigm transitions and lineage progression across generations of work.

## 2.2 Knowledge Evolution Modeling

Knowledge evolution modeling studies how scientific concepts, research topics, and paradigms emerge, evolve, and transition over time. Early work primarily focused on temporal topic modeling, including Dynamic Topic Models (Blei and Lafferty, 2006; Wang et al., 2008; Bhadury et al., 2016) and Topics over Time (Wang and McCallum, 2006), which capture topic drift and temporal word distributions across document collections. Another line of research investigates scientific evolution through citation and co-citation analysis within the science-of-science community (Fortunato et al., 2018). More recently, LLM-based survey systems such as AutoSurvey (Wang et al., 2024) and SurveyX (Liang et al., 2025) retrieve papers, construct outlines, and generate survey-style text, but primarily optimize textual synthesis rather than explicit structural evolution modeling. Although several recent works summarize papers into taxonomy-like structures (Hu et al., 2024; Zhu et al., 2025), none directly addresses scientific evolution tree generation. Existing methods lack temporal constraints over evolutionary paths, and mechanisms for handling marginal papers or constructing survey-style evolution structures.

## 3 Problem Formulation

Citation graph. Given a survey paper, let $\mathcal { G } =$ $( \nu , \mathcal { E } _ { \mathcal { G } } )$ denote the citation graph constructed from its reference set, where each node $v _ { i } ~ \in ~ \mathcal { V }$ is a referenced paper with text content $s _ { i }$ (title and abstract) and publication year $\tau ( v _ { i } )$ . A directed edge $( v _ { i } , v _ { j } ) \in \mathcal { E } _ { \mathcal { G } }$ indicates that paper $v _ { i }$ cites paper $v _ { j }$ For the main evaluation, the target survey itself is excluded from $\mathcal { G }$

Taxonomy tree. A taxonomy tree organizes papers by conceptual inclusion. We denote it as $\mathcal { T } ^ { \mathrm { t a x } } = ( \mathcal { C } ^ { \mathrm { t a x } } , \mathcal { V } , \mathcal { E } ^ { \mathrm { t a x } } , \phi ^ { \mathrm { t a x } } )$ , where $\mathcal { C } ^ { \mathrm { t a x } }$ is the set of topic nodes, V is the set of paper nodes, and ${ \mathcal { E } } ^ { \mathrm { t a x } } \subseteq { \mathcal { C } } ^ { \mathrm { t a x } } \times ( { \mathcal { C } } ^ { \mathrm { t a x } } \cup { \mathcal { V } } )$ forms a rooted tree. Each topic node $c \in \mathcal { C } ^ { \mathrm { t a x } }$ represents a semantic concept with label $\phi ^ { \mathrm { t a x } } ( c )$ , and each paper node $v _ { i } \in \mathcal V$ appears only as a terminal leaf (i.e., for every paper node, its parent must be a concept node rather than another paper node, and this parent corresponds to the lowest-level concept cluster that directly contains the paper). Thus, edges in $\mathcal { T } ^ { \mathrm { t a x } }$ encode conceptual refinement only, with no temporal semantics.

Evolutionary tree. An evolutionary tree $\tau =$ $( \mathcal { C } , \mathcal { V } , \mathcal { E } _ { C } , \mathcal { E } _ { P } , \phi )$ extends this organization with temporally admissible development relations. $\mathcal { E } _ { C }$ forms a rooted tree over concepts C with labels $\phi ,$ and $\mathcal { E } _ { P } = \{ ( a ( v _ { i } ) , v _ { i } ) \}$ attaches each paper to a concept via the attachment map $a : \mathcal { V } \to \mathcal { C }$ . Unlike a taxonomy tree, papers may attach to internal concept nodes when better explained at a coarser abstraction level.

For each concept $c ,$ let $\mathcal { V } _ { \mathrm { a t t } } ( c ) = \{ v _ { i } : a ( v _ { i } ) =$ 11 c} denote the papers attached directly to $c ,$ and $\mathcal { V } ( c )$ the papers in its full subtree. We define two temporal statistics over publication years $\tau ( \cdot )$

$$
\tau ^ { \mathrm { a t t } } ( c ) = \frac { 1 } { | \mathcal { V } _ { \mathrm { a t t } } ( c ) | } \sum _ { v \in \mathcal { V } _ { \mathrm { a t t } } ( c ) } \tau ( v ) , \bar { \tau } ( c ) = \frac { 1 } { | \mathcal { V } ( c ) | } \sum _ { v \in \mathcal { V } ( c ) } \tau ( v ) .\tag{1}
$$

$\tau ^ { \mathrm { a t t } }$ supports structural constraints: because the attachment sets of a parent and its children are disjoint, the constraint it induces is non-trivial. $\bar { \tau }$ summarizes the full concept body and is used for path-level evaluation.

The tree must satisfy (i) Rooted Tree; (ii) Temporal Consistency, for each concept edge $( c _ { p } , c _ { c } ) \in \mathcal { E } _ { C }$ from parent $c _ { p }$ to child $c _ { c }$

$$
\tau ^ { \mathrm { a t t } } ( c _ { p } ) \leq \tau ^ { \mathrm { a t t } } ( c _ { c } ) + \epsilon ;\tag{2}
$$

and (iii) Sibling Ordering by non-decreasing $\tau ^ { \mathrm { a t t } }$ Eq. 2 requires each root-to-leaf path to proceed from earlier, broader concepts to later, more specialized ones, with the slack ϵ absorbing noise in publication dates so that a single mis-dated paper does not invalidate an otherwise valid edge.

Marginal papers. A marginal paper has weak compatibility with all leaf-level concepts. Given a continuous compatibility score $m ( v _ { i } , c ) \in [ 0 , 1 ]$ (instantiated in §4.4.3), the marginal set is

$$
\mathcal { M } = \{ v _ { i } \in \mathcal { V } : \operatorname* { m a x } _ { \ell \in \mathrm { L e a v e s } ( \mathcal { T } ) } m ( v _ { i } , \ell ) < \eta \} ,\tag{3}
$$

where $\eta$ is a threshold. Marginal papers are not discarded; they may attach to internal nodes to represent isolated, transitional, or weakly continued contributions.

Challenges and design overview. This formalism poses four challenges that shape our method. (C1) The discrete structure $( \mathcal { C } , \mathcal { E } _ { C } , a )$ and continuous embeddings $\left\{ \mathbf { e } _ { i } \right\}$ cannot be jointly optimized end-to-end, motivating alternating refinement between structure and representation (§4.4.1). (C2) Temporal consistency (Eq. 2) cannot emerge from a time-agnostic clustering loss, motivating a temporal merge filter on $\tau ^ { \mathrm { a t t } }$ (§4.4.1). (C3) Marginal papers fall outside leaf-only partitions, motivating explicit internal-node re-attachment (§4.4.3). (C4) Temporal refinement risks collapsing the conceptual organization, motivating a staged design that establishes a stable backbone $\mathcal { T } ^ { \mathrm { t a x } } \left( \ S 4 . 3 \right)$ before applying temporal pressure (§4.4).

## 4 Methodology

## 4.1 Overview

EvoTree (Fig. 1) addresses challenges C1–C4 through three stages over a shared backbone. The input of EvoTree (Fig. 1A) is a citation graph in which each node carries paper text and a timestamp, encoded by SPECTER2. The backbone (§4.2) couples a graph-aware encoder with a distributional tree builder that summarizes each cluster as a diagonal Gaussian. Stage I (§4.3) learns a time-agnostic taxonomy backbone T<sup>tax</sup> under section-derived weak supervision. Stage II (§4.4) refines $\mathcal { T } ^ { \mathrm { t a x } }$ into the evolutionary tree T through alternating structure–representation updates with temporal admissibility and marginal-paper handling. Stage III (§4.5) calibrates against reference trees via fewshot metric learning, and a final LLM pass (§4.6) verbalizes concept labels without altering topology. The main text is self-contained for the design rationale of each component; appendices provide derivations, loss formulations, and implementation details.

![](images/bd56dd7ed14c5424e2e0939a71840bf7a7f4c6ef3f616fcc8241c9125ae5ece7.jpg)  
Figure 1: Overview of the EvoTree pipeline. (A) Input citation graph encoded by SPECTER2. (B) Shared backbone with graph-aware encoder and distributional tree builder. (C) Stage I taxonomy backbone learning under ${ \mathcal { L } } ^ { \mathrm { t a x } }$ . (D) Stage II evolutionary refinement via alternating updates under $\mathcal L ^ { \mathrm { e v o } }$ . (E) Stage III few-shot calibration. (F) LLM concept labeling.

## 4.2 Shared Backbone

The pipeline rests on two reusable modules: a graph-aware paper encoder and a distributional tree builder.

## 4.2.1 Graph-aware Encoder

The encoder maps each paper $v _ { i }$ to a clustering embedding $\mathbf { e } _ { i } \in \mathbb { R } ^ { d }$ from text and citation context. A pretrained scientific encoder SPECTER2 (Singh et al., 2022) produces a timeagnostic vector $\mathbf { x } _ { i } ,$ from which a semantic stream $\mathbf { z } _ { i } ^ { \mathrm { s e m } } = \mathrm { M L P } _ { \mathrm { s e m } } ( \mathbf { x } _ { i } )$ and a graph stream ${ \bf h } _ { i } ^ { ( 0 ) }$ = $\mathrm { M L P } _ { \mathrm { g r a p h } } ( \mathbf { x } _ { i } )$ are derived. The graph stream propagates through L layers with anchored residuals and PairNorm (Zhao and Akoglu, 2020):

$$
\begin{array} { r l } & { \mathbf { h } _ { i } ^ { ( l + 1 ) } = } \\ & { \mathrm { P a i r N o r m } \Bigl ( \alpha \mathbf { x } _ { i } + ( 1 - \alpha ) \left[ \beta \mathbf { h } _ { i } ^ { ( l ) } + ( 1 - \beta ) \mathbf { m } _ { i } ^ { ( l ) } \right] \Bigr ) , } \end{array}\tag{4}
$$

where $\mathbf { m } _ { i } ^ { ( l ) }$ aggregates messages from citation neighbors (set to 0 for isolated nodes). The anchored term mitigates over-smoothing while PairNorm stabilizes propagation under uneven density. The two streams fuse via a connectivity-biased gate:

$$
\begin{array} { r } { \mathbf { z } _ { i } ^ { \mathrm { f u s e d } } = \lambda _ { i } \mathbf { h } _ { i } ^ { ( L ) } + ( 1 - \lambda _ { i } ) \mathbf { z } _ { i } ^ { \mathrm { s e m } } , } \\ { \lambda _ { i } = \sigma \big ( \mathbf { W } _ { g } [ \cdot ] + \delta { \mathcal { k } } [ \mathrm { i s o } ( v _ { i } ) ] \big ) , } \end{array}\tag{5}
$$

with $\delta \ < \ 0$ biasing isolated nodes toward the semantic stream. The final embedding is $\mathbf { e } _ { i } ~ =$ $\mathrm { L } _ { 2 } \mathrm { N o r m } ( \mathbf { z } _ { i } ^ { \mathrm { f u s e d } } )$ ; a separate projection $\mathbf { p } _ { i }$ is reserved for contrastive learning (Chen et al., 2020). Full details in Appendix F.

## 4.2.2 Distributional Tree Builder

Given embeddings $\{ \mathbf { e } _ { i } \}$ , the builder constructs a hierarchical tree through distributional clustering. HDBSCAN with soft membership (Campello et al., 2013) yields base clusters $\{ C _ { k } \} _ { k = 1 } ^ { K }$ and weights $w _ { i k }$ , each summarized as $\boldsymbol { C } _ { k } \sim \mathcal { N } ( \pmb { \mu } _ { k } , \mathrm { d i a g } ( \pmb { \sigma } _ { k } ^ { 2 } ) )$ Higher-level concepts arise from bottom-up agglomeration under the closed-form 2-Wasserstein distance:

$$
d ^ { 2 } ( C _ { p } , C _ { q } ) = \| \pmb { \mu } _ { p } - \pmb { \mu } _ { q } \| _ { 2 } ^ { 2 } + \| \pmb { \sigma } _ { p } - \pmb { \sigma } _ { q } \| _ { 2 } ^ { 2 } .\tag{6}
$$

The parent Gaussian is re-estimated from union members after each merge, reducing level-wise drift. The same builder is used in Stages I and II; what changes between them is the supervision and whether temporal admissibility is enforced.

## 4.3 Stage I: Taxonomy Backbone Learning

Stage I learns a time-agnostic backbone $\mathcal { T } ^ { \mathrm { t a x } }$ that captures conceptual inclusion. Inputs are the citation graph, paper texts, and weak hierarchical labels derived from the survey section structure (each paper inherits its section path). We pretrain the graph-aware paper encoder (§4.2.1) under three complementary signals: section supervision (multi-label classification and level prediction over the section hierarchy), citation consistency (link prediction and contrastive ranking on citation edges), and cluster geometry (centroid pulling for labeled papers):

$$
\begin{array} { r } { \mathcal { L } ^ { \mathrm { t a x } } = \mathcal { L } _ { \mathrm { s e c } } + \mathcal { L } _ { \mathrm { c i t e } } + \mathcal { L } _ { \mathrm { g e o } } . } \end{array}\tag{7}
$$

Applying the distributional tree builder to the pretrained embeddings yields $\mathcal { T } ^ { \mathrm { t a x } }$ , which serves as the structural prior for Stage II. Component formulations are given in Appendix H.

We emphasize that section structures serve as weak supervision rather than ground-truth concept hierarchies: headings reflect expository choices as much as conceptual organization, and their granularity varies across surveys. Their influence is correspondingly bounded. They contribute only one of the three terms in Eq. 7, shaping the initialization jointly with the citation and geometry objectives rather than acting as a direct optimization target, and are unavailable at inference.

## 4.4 Stage II: Evolutionary Refinement

Stage II transforms $\mathcal { T } ^ { \mathrm { t a x } }$ into the evolutionary tree T , driven by three signals absent from Stage I: temporal admissibility (Eq. 2), citation directionality (citing papers attach deeper than cited ones), and marginal-paper re-attachment. The heavy components (SPECTER2, GNN weights) remain frozen; only lightweight adapters (projection MLP, fusion gate, marginal head, cluster Gaussians) are fine-tuned.

## 4.4.1 EM-style Refinement

Because discrete tree construction and continuous adapter updates cannot be jointly optimized endto-end (C1), we adopt an alternating procedure inspired by EM. We label the two phases as E-step and M-step for brevity, without claiming a strict EM optimization of a probabilistic likelihood.

E-step — structure update. With adapters fixed, we re-apply the distributional tree builder to the current embeddings. Unlike Stage I, agglomeration is filtered by temporal admissibility: a candidate merge producing parent $c _ { p }$ from children $c _ { a } , c _ { b }$ is rejected if

$$
\tau ^ { \mathrm { a t t } } ( c _ { p } ) > \operatorname* { m i n } \left( \tau ^ { \mathrm { a t t } } ( c _ { a } ) , \tau ^ { \mathrm { a t t } } ( c _ { b } ) \right) + \epsilon ,\tag{8}
$$

which uses the same $\tau ^ { \mathrm { a t t } }$ statistic as the formal admissibility condition (Eq. 2). This step yields an updated tree $\mathscr { T } ^ { ( t ) }$ and a set of low-membership marginal candidates.

M-step — representation update. With $\mathscr { T } ^ { ( t ) }$ fixed, we update the adapters under three selfsupervised objectives:

$$
\mathcal { L } ^ { \mathrm { e v o } } = \mathcal { L } _ { \mathrm { s e m } } + \mathcal { L } _ { \mathrm { t e m p } } + \mathcal { L } _ { \mathrm { m a r g } } ,\tag{9}
$$

where $\mathcal { L } _ { \mathrm { s e m } }$ preserves parent–child semantic coherence, $\mathcal { L } _ { \mathrm { t e m p } }$ encourages temporal and citationdirection consistency, and ${ \mathcal { L } } _ { \mathrm { m a r g } }$ trains the marginal detector (§4.4.3). Detailed formulations in Appendix I.

## 4.4.2 Structural Admissibility Constraints

In addition to the differentiable objectives, the Estep filters candidate trees using admissibility constraints. These constraints prevent the evolutionary refinement from producing structures that are temporally invalid or that destroy the semantic backbone learned in Stage I.

Temporal consistency. This constraint follows Eq. 2, directly rejecting violating merges.

Backbone preservation. This retains the major branches of $\mathcal { T } ^ { \mathrm { t a x } }$ via the Major Branch Count:

$$
\operatorname { M B C } ( { \mathcal { T } } ) = { \big | } { \{ c \in \operatorname { c h i l d r e n } ( r ) : | { \mathcal { V } } ( c ) | \geq \rho | { \mathcal { V } } | \} } { \big | } ,
$$

requiring MBC(T) ≥ κ MBC(T<sup>tax</sup>).

(10)

Tree legality. The output must remain a valid rooted tree, with a single root, no cycles, and a unique parent for each non-root concept node. Detailed constraints are described in Appendix J.

## 4.4.3 Marginal Paper Handling

Stage II explicitly handles marginal papers: papers poorly explained by all fine-grained leaf concepts, which would blur concept boundaries and weaken temporal coherence if forced into leaves. We separate handling into leaf-level detection and all-node re-attachment.

Detection. For each paper v<sub>i</sub>, we compute its best leaf compatibility $\begin{array} { r l r } { A _ { i } ^ { \mathrm { l e a f } } } & { { } = } & { \operatorname* { m a x } _ { c \in \mathrm { L e a v e s } ( \mathcal { T } ^ { ( t ) } ) } m ( v _ { i } , c ) } \end{array}$ where $m ( v _ { i } , c ) ~ \in ~ [ 0 , 1 ]$ combines semantic similarity, citation-neighbor overlap, and temporal compatibility (Appendix K). A paper is marginal when $A _ { i } ^ { \mathrm { l e a f } } < \eta$ , providing pseudo-labels for a marginal classifier (optionally reinforced by HDBSCAN low-membership outliers). Intuitively, a paper is marginal when it is not well explained by any single leaf: its compatibility with even its best-matching concept falls below η, typically because its citations span multiple sibling branches rather than concentrating in one.

Re-attachment. Each flagged marginal paper is re-attached to the concept whose Gaussian best explains its embedding:

$$
a ( v _ { m } ) = \arg \operatorname* { m a x } _ { c \in \mathcal { C } ^ { ( t ) } } \log \mathcal { N } \big ( \mathbf { e } _ { m } ; \pmb { \mu } _ { c } , \mathrm { d i a g } ( \pmb { \sigma } _ { c } ^ { 2 } ) \big ) .\tag{11}
$$

Unlike detection, re-attachment searches over all concept nodes, allowing papers that fit no finegrained leaf to attach to broader internal concepts.

## 4.5 Stage III: Few-shot Metric Learning

To inject structural priors that self-supervised signals alone cannot recover, we calibrate the model on a small set of reference evolutionary trees via few-shot metric learning. Only a lightweight adaptation layer $( \sim 1 7 \%$ of parameters) is updated under a small learning rate (5e-6) for 20 epochs over the reference trees (10 per leave-one-domain-out fold) — orders of magnitude smaller than standard supervised fine-tuning. The calibration losses are purely metric-learning: they shape embedding geometry to reflect concept clusters and evolution directions, without predicting discrete labels.

$$
\begin{array} { r } { \mathcal { L } ^ { \mathrm { c a l } } = \mathcal { L } _ { \mathrm { m a r g } } ^ { \mathrm { c a l } } + \mathcal { L } _ { \mathrm { c o n c e p t } } ^ { \mathrm { c a l } } + \mathcal { L } _ { \mathrm { e d g e } } ^ { \mathrm { c a l } } . } \end{array}\tag{12}
$$

Weight and learning-rate choices ensure the reference signal refines rather than overrides earlier structure. Full formulations in Appendix L.

## 4.6 Concept Labeling

The tree T encodes structure but no humanreadable labels. A single LLM pass traverses $\tau$

bottom-up:

$$
\phi ( c ) = \operatorname { L L M } \bigl ( \Pi ( c , \mathcal { V } ( c ) , \{ \phi ( c ^ { \prime } ) : c ^ { \prime } \in \operatorname { c h i l d r e n } ( c ) \} ) \bigr ) ,\tag{13}
$$

where Π is a structured prompt. Leaf prompts request a precise method name; internal-node prompts require a more abstract noun phrase that generalizes over (and does not repeat) child labels. The LLM is used only for concept verbalization and does not modify topology or paper attachments. Prompt templates in Appendix C.

## 5 Experiments

## 5.1 Datasets

Survey-reference graph dataset. We construct 411 ego-graphs from arXiv survey papers. Hierarchical weak labels are derived from the survey section structure: a referenced paper inherits the section path in which it appears, e.g., Chapter 2 $ 2 . 2  2 . 2 . 3$ . We split graphs, rather than individual nodes, into train/validation/test sets with a 70/15/15 ratio. Detailed statistics are shown in Table 6.

Few-shot (FS) calibration dataset. For direct evaluation against annotated labels, we curate 11 reference evolutionary trees spanning major AI subfields, including computer vision, natural language processing, graph learning, reinforcement learning, and large language models. Each graph contains manual annotations for concept membership (FS\_concept), marginalpaper labels (FS\_is\_marginal), evolution edges (FS\_tree\_edges), and node depth (FS\_depth). Detailed statistics are shown in Table 7. Calibration follows an 11-fold leave-one-domain-out protocol (Appendix B.3).

## 5.2 Clustering Semantic Quality and Evolution Structure Quality

## 5.2.1 Metrics & Baselines

We evaluate two aspects of tree quality. For semantic clustering, we report Leaf Purity (LP) and Path Purity (PP) to measure topic consistency at the leaf and root-to-leaf path levels, respectively, and NMI to measure agreement between the tree-induced partition and topic labels. For evolution structure, we report Citation Direction Accuracy (CDA), which checks whether cited papers are placed at shallower depths than citing papers, and Path Monotonicity Rate (PMR), which measures whether publication years are non-decreasing along root-to-leaf paths.

We compare EvoTree with representative flat, hierarchical, citation-aware, and taxonomy-induction baselines, including HAC-SPECTER2 (Murtagh and Contreras, 2012), KMeans-flat (Ahmed et al., 2020), HAC-4level (Murtagh and Contreras, 2012), CitRank+HAC (Woods, 2024), TaxoGenstyle (Zhang et al., 2018), Hu-CiteTaxo (Hu et al., 2024), TaxoAlign (Lahiri et al., 2025), and Context-Aware (Zhu et al., 2025). We also report EvoTree (Pre-train), which disables temporal fine-tuning, to isolate the effect of representation–structure refinement.

## 5.2.2 Results

Semantic clustering quality. Taxonomygeneration methods remain competitive here: Context-Aware leads on LP (0.713) and PP (0.742), exceeding EvoTree by 6.0 and 3.9 points. EvoTree attains the highest NMI (0.526). These metrics reward flat leaf coherence alone; EvoTree additionally constrains its hierarchy to be temporally admissible, trading some semantic compactness for evolution-consistent structure.

Citation- and time-consistent evolution structure. Hu-CiteTaxo attains the highest PMR (0.322); its CDA is not comparable, as its output does not expose the directional edges the metric evaluates. The more informative distinction is structural: none of the three taxonomy-generation baselines emits evolution edges or marginal scores, so they are not evaluated on the FS-aligned metrics (§5.3). The indirect metrics above cover only the semantic objective that taxonomy generation shares with evolution-tree induction.

Effect of temporal fine-tuning. Comparing EvoTree with EvoTree (Pre-train) isolates the effect of the temporal fine-tuning stage. Full EvoTree improves LP and CDA, showing that joint representation–structure refinement improves both semantic coherence and citation-direction consistency.

## 5.3 Evaluation on Few-shot Test

## 5.3.1 Metrics & Baselines

We evaluate against the annotated few-shot dataset using three metrics: FS-CP, which measures concept purity within predicted leaf nodes; FS-EDA, which measures whether annotated evolutionary descendants are placed deeper than their predecessors; and FS-Mar, which evaluates marginal-paper detection by AUROC.

We compare EvoTree with clustering baselines from §5.2 and LLM-based baselines that prompt GPT-4o to generate hierarchies from titles, titles plus years, or titles plus citation-graph context. We again include EvoTree (Pre-train) to measure the effect of temporal fine-tuning.

## 5.3.2 Results

Alignment with annotated concepts. On the few-shot dataset, EvoTree achieves the highest FS-CP, though the gap is narrow. Its improvement over the strongest LLM baseline is modest, but it consistently outperforms clustering-based baselines and substantially improves over EvoTree (Pre-train), indicating that temporal fine-tuning and tree refinement help align predicted clusters with groundtruth concepts.

Evolution-direction accuracy. EvoTree is on par with the strongest LLM baseline on FS-EDA. The strong performance of LLM-title suggests that large language models can often infer local predecessor–successor relations from method names alone. However, LLM-only methods do not provide explicit structure-level constraints or marginality estimates, which are required for controlled, audit-friendly tree construction.

Marginal-paper detection. FS-Mar reflects a structural capability that, to our knowledge, no other method evaluated in this work exposes: standard clustering and LLM baselines produce only hard assignments and therefore admit no continuous marginality score, while EvoTree directly outputs marginality scores from its hierarchical structure. EvoTree obtains a non-trivial FS-Mar AUROC and improves over its pre-training-only variant, supporting the value of modeling boundary papers rather than forcing every paper into a fine-grained leaf cluster.

## 5.4 Ablation Studies

## 5.4.1 Contribution of Each Training Stage

EvoTree is trained in three stages: Pre-train only (Stage I) uses multi-task supervision with section structure as weak labels; + Post-train (Stage II) adds self-supervised evolutionary losses; + FScorrection (Stage III few-shot calibration; our full model) injects structural priors via few-shot metric learning on the reference trees. We evaluate on test graphs and on the FS dataset (Table 3) to verify that gains generalize beyond the FS-correction training set.

Table 1: Clustering semantic quality and evolution structure quality. Dashes indicate that the method does not produce the required output. Results are mean ± standard deviation over 5 runs. Best results are in bold.
<table><tr><td rowspan="2">Method</td><td colspan="3">Clustering</td><td colspan="2">Evolution</td></tr><tr><td>LP↑</td><td>PP↑</td><td>NMI↑</td><td>CDA↑</td><td>PMR↑</td></tr><tr><td>HAC-SPECTER2</td><td> $0 . 5 2 6 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 6 0 7 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 3 9 9 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td></td><td></td></tr><tr><td>KMeans-flat</td><td> $0 . 5 2 4 { \pm } 0 . 0 1 0$ </td><td> $0 . 6 0 3 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 3 9 2 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td></td><td></td></tr><tr><td>HAC-4level</td><td> $0 . 5 8 6 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 6 6 5 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 4 3 7 { \pm } 0 . 0 1 0$ </td><td>1.000</td><td>0.293±0.014</td></tr><tr><td>CitRank+HAC</td><td> $0 . 4 9 8 { \pm } 0 . 0 1 1$ </td><td> $0 . 5 7 4 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td> $0 . 3 4 6 { \pm } 0 . 0 1 3$ </td><td>1.000</td><td> $0 . 2 5 0 { \pm } 0 . 0 1 5$ </td></tr><tr><td>TaxoGen-style</td><td> $0 . 6 0 1 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 6 7 4 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 5 1 1 { \scriptstyle \pm 0 . 0 1 0 }$ </td><td> $0 . 8 9 2 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $0 . 2 6 5 { \pm } 0 . 0 1 3$ </td></tr><tr><td>Hu-CiteTaxo</td><td> $0 . 6 6 4 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 7 1 8 { \pm } 0 . 0 0 8$ </td><td> $0 . 5 1 9 { \pm } 0 . 0 0 5$ </td><td></td><td> $\mathbf { 0 . 3 2 2 { \scriptstyle \pm 0 . 0 0 7 } }$ </td></tr><tr><td>TaxoAlign</td><td> $0 . 6 5 0 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 6 9 4 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 4 8 0 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 8 8 3 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 3 0 5 { \scriptstyle \pm 0 . 0 0 8 }$ </td></tr><tr><td>Context-Aware</td><td> $\mathbf { 0 . 7 1 3 { \overset { . } { = } } 0 . 0 0 5 }$ </td><td> $\mathbf { 0 . 7 4 2 { \scriptstyle \pm 0 . 0 0 7 } }$ </td><td> $0 . 5 1 2 { \scriptstyle \pm 0 . 0 0 6 }$ </td><td> $0 . 8 2 1 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 3 1 8 { \pm } 0 . 0 0 4$ </td></tr><tr><td>EvoTree (Pre-train)</td><td> $0 . 5 9 1 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 6 8 9 { \pm } 0 . 0 0 9$ </td><td> $0 . 4 9 4 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 8 2 9 { \pm } 0 . 0 1 2$ </td><td> $0 . 2 8 4 { \pm } 0 . 0 1 3$ </td></tr><tr><td>EvoTree</td><td> $0 . 6 5 3 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 7 0 3 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $\mathbf { 0 . 5 2 6 { \overset { . } { = } } 0 . 0 0 9 }$ </td><td> $\mathbf { 0 . 8 9 3 { \scriptstyle \pm 0 . 0 1 1 } }$ </td><td> $0 . 3 1 2 { \scriptstyle \pm 0 . 0 1 2 }$ </td></tr></table>

Table 2: Evaluation on the few-shot benchmark. Results are mean ± standard deviation over 5 runs. Best results are in bold.
<table><tr><td>Method</td><td> ${ \mathrm { F S - C P } } \uparrow$ </td><td> $\mathrm { F S - E D A \uparrow }$ </td><td> $\mathrm { F S - M a r \uparrow }$ </td></tr><tr><td>HAC-SPECTER2</td><td> $0 . 6 0 0 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 8 3 8 { \pm } 0 . 0 1 3$ </td><td></td></tr><tr><td>CitRank+HAC</td><td> $0 . 7 3 9 { \pm } 0 . 0 1 0$ </td><td>1.000</td><td></td></tr><tr><td>TaxoGen-style</td><td> $0 . 6 6 1 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 9 2 9 { \pm } 0 . 0 1 2$ </td><td></td></tr><tr><td>LLM (title)</td><td> $0 . 7 6 7 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 9 6 4 { \scriptstyle \pm 0 . 0 1 0 }$ </td><td></td></tr><tr><td>LLM (title + year)</td><td> $0 . 7 5 9 { \pm } 0 . 0 1 0$ </td><td> $0 . 9 3 7 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td></td></tr><tr><td>LLM (citation graphs)</td><td> $0 . 7 6 9 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 9 5 2 { \scriptstyle \pm 0 . 0 1 0 }$ </td><td></td></tr><tr><td>EvoTree (Pre-train)</td><td> $0 . 6 5 3 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 9 0 4 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td> $0 . 6 6 2 { \scriptstyle \pm 0 . 0 1 3 }$ </td></tr><tr><td>EvoTree</td><td> $\mathbf { 0 . 7 7 5 { \scriptstyle \pm 0 . 0 0 8 } }$ </td><td> $0 . 9 4 9 { \pm } 0 . 0 1 0$ </td><td> $\mathbf { 0 . 7 2 6 { \overset { . } { = } } 0 . 0 1 1 }$ </td></tr></table>

Both later stages deliver substantial gains in leaf purity and concept-level alignment (FS-CP +0.104 after post-training and a further +0.118 after FScorrection), consistent with the self-supervised evolutionary losses injecting temporal structure that label supervision cannot capture, and with the few-shot priors sharpening concept boundaries. Post-training yields the largest gain in marginalpaper recognition $( \mathrm { F S - M a r } \ 0 . 6 6 2  0 . 7 5 0 )$ , most of which FS-correction preserves (0.726). These gains also transfer to test graphs unseen during FS-correction training, mitigating concerns about overfitting to the reference trees. The dip in FS-EDA after post-training reflects the tension between self-supervised temporal refinement and strict depth ordering; FS-correction recovers most of it $( 0 . 8 8 0  0 . 9 4 9 )$

the citation graph.

Removing the citation graph causes an even larger gap in validation score than at inference, indicating that the graph contributes signal throughout optimization. The MLP variant’s slightly higher PMR is a degenerate artifact: without citation constraints, tree construction relies on the temporal distribution of embeddings alone, trivially producing monotonic paths — mirroring the CDA artifact for HAC-4level in §5.2. Together with the training-stage ablation (§5.4.1), these results confirm that EvoTree relies jointly on the citation graph as a structural backbone and on the staged training framework that progressively injects supervised, self-supervised, and correction signals.

## 5.4.2 Necessity of Graph Structure

To verify that the GNN exploits the citation graph rather than merely propagating SPECTER2 embeddings, we construct Pre-train (MLP) — identical to Pre-train (GNN) in hyperparameters and training objective but with the citation edge set replaced by an empty set, so any gap is attributable solely to

## 5.5 Generation Quality

The metrics in Tables 1 and 2 measure agreement with reference partitions and citation ordering, but not whether a structure reads as a coherent account of how a field developed. We therefore evaluate the generated structures directly under two independent protocols: an LLM judge, instantiated with the same model as concept labeling (§4.6) and run multiple times per structure, and three CS

Table 3: Ablation study on the contribution of each training stage. Left: clustering and evolution-structure metrics on test graphs; right: FS-aligned metrics on the 11 annotated domains. Results are mean ± standard deviation over 5 runs. Best results are in bold.
<table><tr><td rowspan="2">Variant</td><td rowspan="2">LP↑</td><td rowspan="2">Clustering PP↑</td><td rowspan="2">NMI↑</td><td colspan="2">Evolution</td><td colspan="3">FS-aligned</td></tr><tr><td>CDA↑</td><td>PMR↑</td><td>FS-CP↑</td><td> $\mathrm { F S - E D A \uparrow }$ </td><td> $\mathrm { F S - M a r \uparrow }$ </td></tr><tr><td>Pre-train only</td><td> $0 . 5 4 1 { \scriptstyle \pm 0 . 0 1 0 }$ </td><td> $0 . 6 2 9 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $\mathbf { 0 . 5 3 0 { \scriptstyle \pm 0 . 0 1 1 } }$ </td><td> $0 . 8 2 9 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $0 . 2 4 0 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td> $0 . 5 5 3 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $\mathbf { 0 . 9 5 4 } \pm \mathbf { 0 . 0 1 } 2$ </td><td> $0 . 6 6 2 { \scriptstyle \pm 0 . 0 1 3 }$ </td></tr><tr><td>+ Post-train</td><td> $0 . 5 9 8 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 6 6 5 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 4 5 5 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 8 5 7 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td> $0 . 2 7 0 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $0 . 6 5 7 { \scriptstyle \pm 0 . 0 1 0 }$ </td><td> $0 . 8 8 0 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td> $\mathbf { 0 . 7 5 0 { \scriptstyle \pm 0 . 0 1 2 } }$ </td></tr><tr><td>+FS-correction</td><td> $\mathbf { 0 . 6 5 3 { \scriptstyle \pm 0 . 0 0 7 } }$ </td><td> $\mathbf { 0 . 7 0 3 { \scriptstyle \pm 0 . 0 0 8 } }$ </td><td> $0 . 5 2 6 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $\mathbf { 0 . 8 9 3 { \scriptstyle \pm 0 . 0 1 1 } }$ </td><td> $\mathbf { 0 . 3 1 2 { \scriptstyle \pm 0 . 0 1 2 } }$ </td><td> $\mathbf { 0 . 7 7 5 { \scriptstyle \pm 0 . 0 0 8 } }$ </td><td> $0 . 9 4 9 { \scriptstyle \pm 0 . 0 1 0 }$ </td><td> $0 . 7 2 6 { \scriptstyle \pm 0 . 0 1 1 }$ </td></tr></table>

Table 4: Ablation study on the necessity of the citation graph. “Val. score” is the validation score during training. Best in bold. Mean ± std over 5 runs.
<table><tr><td>Variant</td><td>LP↑</td><td>PP↑</td><td>NMI↑</td><td>PMR↑</td><td>| Val. score ↑</td></tr><tr><td>Pre-train (MLP, no graph)</td><td> $0 . 4 5 0 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td> $0 . 5 3 6 { \pm } 0 . 0 1 3$ </td><td> $0 . 3 6 3 { \pm } 0 . 0 1 3$ </td><td> $\mathbf { 0 . 3 0 6 { \scriptstyle \pm 0 . 0 1 3 } }$ </td><td> $0 . 5 1 4 { \pm } 0 . 0 1 4$ </td></tr><tr><td>Pre-train (GNN, with graph)</td><td> $\mathbf { 0 . 5 4 1 { \pm 0 . 0 1 0 } }$ </td><td> $\mathbf { 0 . 6 2 9 } \pm \mathbf { 0 . 0 1 1 }$ </td><td> $\mathbf { 0 . 4 6 4 } \pm \mathbf { 0 . 0 1 1 }$ </td><td> $0 . 2 4 0 { \scriptstyle \pm 0 . 0 1 4 }$ </td><td> $\mathbf { 0 . 7 8 0 { \overset { . } { = } } 0 . 0 1 0 }$ </td></tr></table>

Table 5: Generation-quality evaluation on the FS domains under two independent protocols, LLM-as-a-Judge and Human evaluation. Scores are on a 1–10 scale (mean ± standard deviation over LLM runs or human evaluators). Best results are in bold.
<table><tr><td rowspan="2">Method</td><td colspan="4">LLM-as-a-Judge</td><td colspan="4">Human evaluation</td></tr><tr><td>Concept</td><td></td><td>Evolution Transitional</td><td>Overall</td><td>Concept</td><td></td><td>Evolution Transitional</td><td>Overall</td></tr><tr><td>CitRank+HAC</td><td> $4 . 2 1 { \pm } 0 . 1 2$ </td><td> $3 . 5 7 { \pm } 0 . 1 4$ </td><td> $2 . 8 2 { \pm } 0 . 0 6$ </td><td> $4 . 0 8 { \pm } 0 . 0 4$ </td><td> $4 . 7 7 { \pm } 0 . 0 7$ </td><td> $4 . 3 8 { \pm } 0 . 0 5$ </td><td> $4 . 9 0 { \pm } 0 . 0 9$ </td><td> $4 . 2 1 { \pm } 0 . 0 9$ </td></tr><tr><td>Hu-CiteTaxo</td><td> $4 . 5 5 { \pm } 0 . 1 1$ </td><td> $4 . 7 3 { \pm } 0 . 0 7$ </td><td> $3 . 5 7 { \pm } 0 . 1 3$ </td><td> $4 . 6 4 { \pm } 0 . 1 4$ </td><td> $6 . 6 5 { \pm } 0 . 1 5$ </td><td> $5 . 8 7 { \pm } 0 . 1 0 $ </td><td> $5 . 8 2 { \pm } 0 . 0 1$ </td><td> $6 . 1 9 { \pm } 0 . 0 5$ </td></tr><tr><td>TaxoAlign</td><td> $5 . 4 4 { \pm } 0 . 0 5$ </td><td> $4 . 8 9 { \pm } 0 . 1 1$ </td><td> $4 . 2 2 { \pm } 0 . 0 5$ </td><td> $4 . 6 7 { \pm } 0 . 0 6$ </td><td> $5 . 9 4 \pm 0 . 0 5$ </td><td> $5 . 6 1 { \pm } 0 . 0 8$ </td><td> $5 . 2 6 { \pm } 0 . 0 5$ </td><td> $6 . 4 4 { \pm } 0 . 0 4$ </td></tr><tr><td>Context-Aware</td><td> $5 . 6 0 { \pm } 0 . 1 4$ </td><td> $5 . 2 0 { \pm } 0 . 0 4$ </td><td> $4 . 4 0 { \pm } 0 . 0 7$ </td><td> $5 . 1 0 { \pm } 0 . 1 2$ </td><td> $6 . 8 1 { \pm } 0 . 0 4 $ </td><td> $6 . 3 2 { \pm } 0 . 1 4$ </td><td> $5 . 9 7 { \pm } 0 . 0 1 $ </td><td> $6 . 6 7 { \pm } 0 . 0 9$ </td></tr><tr><td> $\mathrm { T a x o G e n – s t y l e }$ </td><td> ${ \bf 7 . 1 4 \pm 0 . 0 5 }$ </td><td> $5 . 6 6 { \pm } 0 . 1 1$ </td><td> $4 . 9 7 { \pm } 0 . 0 4$ </td><td> $6 . 2 7 { \pm } 0 . 1 2$ </td><td> $\mathbf { 8 . 6 4 \pm 0 . 0 5 }$ </td><td> $6 . 1 8 { \pm } 0 . 0 3$ </td><td> $5 . 9 3 { \pm } 0 . 1 3$ </td><td> $7 . 4 0 { \pm } 0 . 1 2$ </td></tr><tr><td>EvoTree</td><td> $6 . 0 3 { \pm } 0 . 1 3$ </td><td> ${ \bf 7 . 5 1 { \pm 0 . 0 4 } }$ </td><td> ${ \bf 6 . 8 5 { \pm } 0 . 1 0 }$ </td><td> ${ \bf 6 . 8 2 \pm 0 . 0 5 }$ </td><td> $7 . 1 6 { \pm } 0 . 0 8$ </td><td> ${ \bf 8 . 6 5 \pm 0 . 1 5 }$ </td><td> ${ \bf 6 . 4 7 \pm 0 . 1 1 }$ </td><td> $\mathbf { 8 . 0 5 } \pm \mathbf { 0 . 0 3 }$ </td></tr></table>

## 6 Conclusion

graduate students distinct from the FS annotators, scoring independently. Both apply the same four criteria—conceptual organization, scientific evolution, transitional-paper placement, and overall quality—to anonymized structures in randomized order. Protocol details are in Appendix N.

Both protocols rank EvoTree first on overall quality and produce the same overall ordering of all six methods (Table 5). EvoTree leads on scientific evolution and transitional-paper placement, the two dimensions taxonomy induction does not target, while TaxoGen-style leads on conceptual organization. The latter follows from temporal admissibility: when a method family develops over a long span, enforcing Eq. 2 separates papers that are semantically adjacent but temporally distant, at some cost to leaf-level compactness—the same trade-off behind EvoTree’s lower LP and PP. The rubric is generic across hierarchy-generation methods and does not reward EvoTree-specific mechanisms such as temporal constraints or internal-node re-attachment.

We introduced EvoTree, the first framework for automatically inducing scientific evolution trees from citation graphs, and formalized the task with an annotated benchmark covering 11 AI subfields. By decoupling conceptual backbone learning from temporal refinement, EvoTree first constructs a stable taxonomy via graph-aware encoding and distribution-based hierarchical clustering, then reattaches marginal papers to internal nodes under monotonic-path constraints, and finally invokes an LLM solely for concept verbalization without altering topology. Against taxonomy-induction, citation-only, and LLM-based baselines, EvoTree attains the best NMI, citation-direction accuracy, and annotated concept purity, is preferred by both LLM and human judges on overall quality, and is the only method that recovers transitional papers with non-trivial accuracy; ablations confirm that both the citation graph and the staged training framework are jointly necessary. We view evolution-tree induction as a step toward scalable, temporally faithful organization of scientific literature.

## Limitations

Scope of domains. Our 411 ego-graphs and 11 reference trees are drawn from AI surveys. We have not verified transfer to disciplines whose citation conventions and paradigm cadences differ substantially from AI.

Scale of the few-shot reference set. The FS dataset comprises 11 curated reference trees and 352 papers, sufficient to demonstrate the few-shot regime but limiting fine-grained per-domain analyses; a larger curated set would strengthen alignment evaluation.

Reliance on survey section structure. Stage I assumes that section organization reflects a coherent conceptual taxonomy. Surveys organized by application area or chronology may provide weaker supervision than those organized by methodology.

Temporal signal noise. Publication years conflate arXiv preprint, conference, and journal dates, introducing noise into the temporal-admissibility constraint and PMR. This contributes to the modest absolute PMR values observed in our experiments.

External LLM dependence. Concept labels are generated by GPT-4.1-mini. Although the LLM is used only for verbalization and does not alter topology, this introduces a dependency on a closedsource model; substituting an open-source LLM is straightforward but may change labeling style.

## References

Mohiuddin Ahmed, Raihan Seraj, and Syed Mohammed Shamsul Islam. 2020. The k-means algorithm: A comprehensive survey and performance evaluation. Electronics, 9(8).

Omri Avrahami, Thomas Hayes, Oran Gafni, Sonal Gupta, Yaniv Taigman, Devi Parikh, Dani Lischinski, Ohad Fried, and Xi Yin. 2023. Spatext: Spatiotextual representation for controllable image generation. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 18370–18380.

Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, and 1 others. 2022. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324.

Mohit Bansal, David Burkett, Gerard De Melo, and Dan Klein. 2014. Structured learning for taxonomy induction with belief propagation. In Proceedings

of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1041–1051.

Arnab Bhadury, Jianfei Chen, Jun Zhu, and Shixia Liu. 2016. Scaling up dynamic topic models. In Proceedings ofthe 25th International Conference on World Wide Web, page 381–390.

David M. Blei and John D. Lafferty. 2006. Dynamic topic models. In Proceedings of the 23rd International Conference on Machine Learning, page 113–120.

Ricardo JGB Campello, Davoud Moulavi, and Joerg Sander. 2013. Density-based clustering based on hierarchical density estimates. In Pacific-Asia Conference on Knowledge Discovery and Data Mining, pages 160–172. Springer.

Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. 2023. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM transactions on Graphics (TOG), 42(4):1–10.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 2020. A simple framework for contrastive learning of visual representations. arXiv preprint arXiv:2002.05709.

Guillaume Couairon, Marlene Careil, Matthieu Cord, Stéphane Lathuiliere, and Jakob Verbeek. 2023. Zero-shot spatial layout conditioning for text-toimage diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2174–2183.

Santo Fortunato, Carl Bergstrom, Katy Borner, James Evans, Dirk Helbing, Stasa Milojevic, Alexander Petersen, Filippo Radicchi, Roberta Sinatra, Brian Uzzi, Alessandro Vespignani, Ludo Waltman, Dashun Wang, and Albert-Laszlo Barabasi. 2018. Science of science. Science, 359:eaao0185.

Yuntong Hu, Zhihan Lei, Zhongjie Dai, Allen Zhang, Abhinav Angirekula, Zheng Zhang, and Liang Zhao. 2025. Cg-rag: Research question answering by citation graph retrieval-augmented llms. In Proceedings ofthe 48th international ACM SIGIR conference on research and development in information retrieval, pages 678–687.

Yuntong Hu, Zhuofeng Li, Zheng Zhang, Chen Ling, Raasikh Kanjiani, Boxin Zhao, and Liang Zhao. 2024. Taxonomy tree generation from citation graph.

Minhao Jiang, Xiangchen Song, Jieyu Zhang, and Jiawei Han. 2022. Taxoenrich: Self-supervised taxonomy completion via structure-semantic representations. In Proceedings of the ACM web conference 2022, pages 925–934.

Priyanka Kargupta, Nan Zhang, Yunyi Zhang, Rui Zhang, Prasenjit Mitra, and Jiawei Han. 2025.

Taxoadapt: Aligning llm-based multidimensional taxonomy construction to evolving research corpora. pages 29834–29850.

Avishek Lahiri, Yufang Hou, and Debarshi Kumar Sanyal. 2025. Taxoalign: Scholarly taxonomy generation using language models. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 30191–30211.

Xiaoying Le, Pengfei Qian, Yuanzhao Zhai, Xu Zhang, Qian Liu, Feng Dawei, and Bo Ding. 2026. Evonarrator: Modeling scientific evolution for feasible hypothesis generation. In Proceedings ofthe 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11846– 11865.

Dongha Lee, Jiaming Shen, SeongKu Kang, Susik Yoon, Jiawei Han, and Hwanjo Yu. 2022. Taxocom: Topic taxonomy completion with hierarchical discovery of novel topic clusters. In Proceedings ofthe ACM Web Conference 2022, pages 2819–2829.

Xun Liang, Jiawei Yang, Yezhaohui Wang, Chen Tang, Zifan Zheng, Simin Niu, Shichao Song, Hanyu Wang, Bo Tang, Feiyu Xiong, Keming Mao, and Zhiyu Li. 2025. Surveyx: Academic survey automation via large language models.

Yuyin Lu, Hegang Chen, Pengbo Mao, Yanghui Rao, Haoran Xie, Fu Lee Wang, and Qing Li. 2024. Selfsupervised topic taxonomy discovery in the box embedding space. Transactions of the Association for Computational Linguistics, 12:1401–1416.

Yuning Mao, Xiang Ren, Jiaming Shen, Xiaotao Gu, and Jiawei Han. 2018. End-to-end reinforcement learning for automatic taxonomy induction. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2462–2472.

Sahil Mishra, Kumar Arjun, and Tanmoy Chakraborty. 2025. Rank, chunk and expand: Lineage-oriented reasoning for taxonomy expansion. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 12935–12953.

Fionn Murtagh and Pedro Contreras. 2012. Algorithms for hierarchical clustering: an overview. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 2.

Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. 2021. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741.

William Peebles and Saining Xie. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. 2024. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In International Conference on Learning Representations, volume 2024, pages 1862–1874.

Jason Priem, Heather Piwowar, and Richard Orr. 2022. Openalex: A fully-open index of scholarly works, authors, venues, institutions, and concepts. arXiv preprint arXiv:2205.01833.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695.

Jiaming Shen, Zhihong Shen, Chenyan Xiong, Chi Wang, Kuansan Wang, and Jiawei Han. 2020. Taxoexpan: Self-supervised taxonomy expansion with position-enhanced graph neural network. In Proceedings ofthe web conference 2020, pages 486–497.

Jiaming Shen, Zeqiu Wu, Dongming Lei, Chao Zhang, Xiang Ren, Michelle T Vanni, Brian M Sadler, and Jiawei Han. 2018. Hiexpan: Task-guided taxonomy construction by hierarchical tree expansion. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2180–2189.

Amanpreet Singh, Mike D’Arcy, Arman Cohan, Doug Downey, and Sergey Feldman. 2022. Scirepeval: A multi-format benchmark for scientific document representations. In Conference on Empirical Methods in Natural Language Processing.

Rion Snow, Dan Jurafsky, and Andrew Y Ng. 2006. Semantic taxonomy induction from heterogenous evidence. In Proceedings ofthe 21st international conference on computational linguistics and 44th annual meeting ofthe associationfor computational linguistics, pages 801–808.

Chong Wang, David Blei, and David Heckerman. 2008. Continuous time dynamic topic models. In Proceedings of the Twenty-Fourth Conference on Uncertainty in Artificial Intelligence, page 579–586.

Xuerui Wang and Andrew McCallum. 2006. Topics over time: a non-markov continuous-time model of topical trends. In Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, page 424–433.

Yidong Wang, Qi Guo, Wenjin Yao, Hongbo Zhang, Xin Zhang, Zhen Wu, Meishan Zhang, Xinyu Dai, Min Zhang, Qingsong Wen, Wei Ye, Shikun Zhang, and Yue Zhang. 2024. Autosurvey: large language models can automatically write surveys. In Proceedings ofthe 38th International Conference on Neural Information Processing Systems.

Stephen Woods. 2024. A programmatic approach to journal use and citation analysis. portal: Libraries and the Academy, 24:177 – 200.

Yongliang Wu, Shiji Zhou, Mingzhuo Yang, Lianzhe Wang, Heng Chang, Wenbo Zhu, Xinting Hu, Xiao Zhou, and Xu Yang. 2025. Unlearning concepts in diffusion model via concept domain correction and concept preserving gradient. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 8496–8504.

Yijun Yang, Ruiyuan Gao, Xiao Yang, Jianyuan Zhong, and Qiang Xu. 2024. Guardt2i: Defending text-toimage models from adversarial prompts. Advances in neural information processing systems, 37:76380– 76403.

Yue Yu, Yinghao Li, Jiaming Shen, Hao Feng, Jimeng Sun, and Chao Zhang. 2020. Steam: Self-supervised taxonomy expansion with mini-paths. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 1026–1035.

Chao Zhang, Fangbo Tao, Xiusi Chen, Jiaming Shen, Meng Jiang, Brian Sadler, Michelle Vanni, and Jiawei Han. 2018. Taxogen: Unsupervised topic taxonomy construction by adaptive term embedding and clustering. In Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, pages 2701–2709.

Chenyu Zhang, Mingwang Hu, Wenhui Li, and Lanjun Wang. 2025. Adversarial attacks and defenses on text-to-image diffusion models: A survey. Information Fusion, 114:102701.

Yimeng Zhang, Xin Chen, Jinghan Jia, Yihua Zhang, Chongyu Fan, Jiancheng Liu, Mingyi Hong, Ke Ding, and Sijia Liu. 2024. Defensive unlearning with adversarial training for robust concept erasure in diffusion models. Advances in neural information processing systems, 37:36748–36776.

Lingxiao Zhao and Leman Akoglu. 2020. Pairnorm: Tackling oversmoothing in {gnn}s. In International Conference on Learning Representations.

Kun Zhu, Lizi Liao, Yuxuan Gu, Lei Huang, Xiaocheng Feng, and Bing Qin. 2025. Context-aware hierarchical taxonomy generation for scientific papers via LLM-guided multi-aspect clustering. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 15616–15634.

## A Case Study: Evolution Tree on Adversarial Attacks on Text-to-Image Diffusion Models

## A.1 Setup

We instantiate EvoTree on the reference corpus of Adversarial Attacks and Defenses on Text-to-Image Diffusion Models: A Survey (Zhang et al., 2025)<sup>1</sup>, which serves as both the ego node and the topical anchor of the resulting tree. The survey was selected because (i) it provides a recent, curated, and topically coherent snapshot of an actively evolving subfield, (ii) it spans both attack-side and defenseside work, allowing the tree to surface convergent research lines, and (iii) the ego node itself appears as paper [20] in the corpus, providing a natural reference point for the green branch. Unlike the experiments setting, we retain the survey node in this case-study visualization to anchor the visual layout and to illustrate how the ego work would be positioned by EvoTree in its own subfield.

After deduplication and minor filtering of offtopic citations, the corpus contains 61 papers published between 2021 and 2024. EvoTree organizes them into 12 leaf clusters grouped under 4 top-level branches, as visualized in Figure 2. The bracketed indices [1]–[61] refer exclusively to the corpus entries and are independent of the bibliography numbering in the main paper.

## A.2 Branch and Cluster Overview

The four top-level branches partition the corpus along complementary dimensions of the research landscape:

• Adversarial Threats and Defenses in Generative Models (19 papers, blue) — work targeting T2I diffusion models with adversarial prompts, red-teaming protocols, and safety filter evaluation.

• Adversarial Robustness and Defense Methods (3 papers, green) — robustness evaluation studies and the ego survey itself.

• Text-Conditioned Image Diffusion Models (19 papers, red) — the foundation-model lineage on which downstream attack/defense work depends, from unconditional DDPMclass models through latent and spatially conditioned T2I systems.

• Concept Manipulation and Optimization (20 papers, purple) — concept erasure, model editing, and pruning techniques that intersect heavily with the safety agenda.

## A.3 Analysis and Discussion

EvoTree recovers three structural properties of the subfield. (i) A clear foundation–application stratification: the red branch (DiT [24] (Peebles and Xie, 2023), latent diffusion [31](Rombach et al., 2022), GLIDE [32](Nichol et al., 2021), SDXL [27](Podell et al., 2024), and spatially conditioned variants [38]–[41](Couairon et al., 2023; Chefer et al., 2023; Avrahami et al., 2023; Balaji et al., 2022)) temporally and conceptually precedes the attack and editing work in the blue and purple branches. (ii) The concept-erasure cluster (entries [42]–[55](Wu et al., 2025; Zhang et al., 2024)) is the densest single thematic block, marking it as the dominant defense paradigm in this timeframe— ahead of input-side filtering or output-side detection. (iii) The blue branch is essentially confined to 2023 and later, placing adversarial-attack research on T2I (Yang et al., 2024) roughly two years behind the foundation work.

## B Details of Datasets

## B.1 Main Dataset

We construct 411 ego-graphs from arXiv survey papers. Each graph is centered on one survey paper: its references form the graph nodes, and citations among these references form the directed edges. The survey node itself is excluded. Each paper node carries a title, an abstract, a publication year, and a 768-dimensional SPECTER2 (Singh et al., 2022) embedding. Weak hierarchical labels are inherited from the survey’s section structure: a referenced paper takes on the section path in which it appears, e.g., Chapter $2  2 . 2  2 . 2 . 3$ . We split at the graph level, rather than the node level, into train/validation/test sets with a 70/15/15 ratio.

Table 6: Statistics of the survey-reference graph dataset.
<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Graphs</td><td>411</td></tr><tr><td>Paper nodes</td><td>44,812</td></tr><tr><td>Citation edges</td><td>205,725</td></tr><tr><td>Avg. nodes / graph</td><td>109.0</td></tr><tr><td>Avg. edges / graph Node embedding dim.</td><td>500.5 768</td></tr></table>

![](images/e1b2a6265d70bdc8c8332e1c4bdbe88164388287daecedcbefd909f780cf89d4.jpg)  
Figure 2: Evolution tree generated by EvoTree for the topic Adversarial Attacks on Text-to-Image Diffusion Models, instantiated on the reference corpus of the survey by Zhang et al. (2025). Cluster labels are produced by an LLM; the layout is rendered from the induced tree.

## B.2 Few-shot Dataset

Table 7: Statistics of the FS dataset.
<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Domains</td><td>11</td></tr><tr><td>Paper nodes</td><td>352</td></tr><tr><td>FS evolution edges</td><td>90</td></tr><tr><td>FS marginal papers</td><td>24</td></tr><tr><td>Avg. nodes / domain</td><td>32.0</td></tr></table>

Annotation schema. Each domain graph contains four types of annotations: (i) FS\_concept, which assigns each paper to a defined method concept; (ii) FS\_is\_marginal, which marks papers that lie on topic boundaries or cannot be confidently assigned to a fine-grained concept; (iii) FS\_tree\_edges, which records directed method-evolution relations, each typed as extends, improves, or (cross-domain) adapts, where the source paper is regarded as an extension or improvement of the target paper; and (iv) FS\_depth, which indicates the depth of a paper in the evolution structure.

Domain coverage. The FS dataset contains 11 high-quality annotated AI subfields. We retain domains with sufficiently informative annotations for concept membership, evolution relations, and marginal-paper evaluation, and exclude domains whose evolution-edge annotations are too sparse for reliable FS-based comparison. Table 8 reports the per-domain statistics.

FS evaluation protocol. The FS benchmark participates in Stage III calibration under an 11-fold leave-one-domain-out protocol. In each fold, calibration uses the 10 remaining domains and evaluation is performed on the held-out domain; the heldout domain’s concept labels, marginal-paper labels, evolution edges, and depth annotations are used neither for calibration nor for validation, threshold selection, or hyperparameter tuning. Each of the 11 domains serves once as the held-out domain; FSaligned metrics are averaged over the 11 held-out evaluations, and we report the mean and standard deviation of this average over 5 runs.

Table 8: Per-domain statistics of the annotated FS dataset. N denotes the number of papers in each domain. “Concepts” denotes the number of annotated method concepts. “FS edges” denotes the number of annotated method-evolution edges. “Marginal” denotes the number of papers annotated as marginal or boundary papers.
<table><tr><td>Domain</td><td>N</td><td>Concepts</td><td>FS edges</td><td>Marginal</td></tr><tr><td>3D Vision</td><td>25</td><td>5</td><td>11</td><td>4</td></tr><tr><td>CV ConvNets</td><td>40</td><td>7</td><td>4</td><td>1</td></tr><tr><td>Vision Transformers</td><td>27</td><td>4</td><td>2</td><td>3</td></tr><tr><td>GAN Image Synthesis</td><td>23</td><td>6</td><td>10</td><td>3</td></tr><tr><td>LLM Alignment</td><td>55</td><td>5</td><td>12</td><td>1</td></tr><tr><td>Meta / Few-shot</td><td>27</td><td>5</td><td>13</td><td>2</td></tr><tr><td>NLP Transformers</td><td>35</td><td>6</td><td>3</td><td>2</td></tr><tr><td>Object Detection</td><td>38</td><td>6</td><td>3</td><td>1</td></tr><tr><td>Self-supervised Learning</td><td>29</td><td>5</td><td>11</td><td>2</td></tr><tr><td>Semantic Segmentation</td><td>37</td><td>5</td><td>14</td><td>2</td></tr><tr><td>Video Understanding</td><td>16</td><td>6</td><td>7</td><td>3</td></tr><tr><td>Total</td><td>352</td><td>60</td><td>90</td><td>24</td></tr><tr><td>Average</td><td>32.0</td><td>5.5</td><td>8.2</td><td>2.2</td></tr></table>

## B.3 FS Cross-domain Overlap

The leave-one-domain-out protocol assumes that the held-out domain is genuinely unseen during calibration. Because the 11 FS domains are drawn from surveys within a single broad field, this assumption could be weakened if the same papers recur across domains: a paper appearing in both the calibration and held-out sets would leak its annotation indirectly. We therefore quantify paper-level overlap across all domain pairs.

Table 9 reports the number of papers shared by each pair of domains, with domain sizes on the diagonal. Overlap is sparse throughout: the largest shared count is 3 papers, or 8.6% when normalized by the smaller domain in the pair, and most pairs share at most one or two papers. The non-negligible overlaps concentrate between semantically adjacent domains (D2/D6/D8: CV ConvNets, Object Detection, and Semantic Segmentation; D10/D11: NLP Transformers and LLM Alignment), which is expected since neighboring subfields cite a common set of foundational works. Under this level of overlap, a held-out domain’s evaluation set is composed almost entirely of papers absent from the corresponding calibration folds, so the FS-aligned results are unlikely to reflect withinset memorization.

We note that overlap counts alone do not exclude all forms of information sharing—semantically similar papers may appear under different identifiers across surveys. The reported figures bound direct duplication, not conceptual proximity between domains.

<table><tr><td colspan="12">D1 D2 D3 D4 D5 D6 D7 D8 D9 D10 D11</td></tr><tr><td>D1</td><td>25</td><td>1</td><td>1</td><td>0</td><td>0 1</td><td></td><td>0 1</td><td>0</td><td>0</td><td>0</td></tr><tr><td>D2</td><td>1</td><td>40</td><td>2</td><td>1</td><td>2</td><td>3 2</td><td>3</td><td>2</td><td>0</td><td>0</td></tr><tr><td>D3</td><td>1</td><td>2</td><td>27</td><td>1</td><td>1</td><td>1 2</td><td>1</td><td>1</td><td>2</td><td>1</td></tr><tr><td>D4</td><td>0</td><td>1</td><td>1</td><td>23</td><td>0</td><td>0 1</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>D5</td><td>0</td><td>2</td><td>1</td><td>0</td><td>27</td><td>1 0</td><td>0</td><td>1</td><td>0</td><td>0</td></tr><tr><td>D6</td><td>1</td><td>3</td><td>1</td><td>0</td><td>1</td><td>38 1</td><td>3</td><td>2</td><td>0</td><td>0</td></tr><tr><td>D7</td><td>0</td><td>2</td><td>2</td><td>1</td><td>0</td><td>1 29</td><td>1</td><td>0</td><td>0</td><td>0</td></tr><tr><td>D8</td><td>1</td><td>3</td><td>1</td><td>0</td><td>0</td><td>3 1</td><td>37</td><td>1</td><td>0</td><td>0</td></tr><tr><td>D9</td><td>0</td><td>2</td><td>1</td><td>0</td><td>1</td><td>2 0</td><td>1</td><td>16</td><td>0</td><td>0</td></tr><tr><td>D10</td><td>0</td><td>0</td><td>2</td><td>0</td><td>0</td><td>0 0</td><td>0</td><td>0</td><td>35</td><td>3</td></tr><tr><td>D11</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0 0</td><td>0</td><td>0</td><td>3</td><td>55</td></tr></table>

Table 9: Pairwise paper-level overlap across the 11 FS domains. Diagonal entries (bold) give domain sizes; off-diagonal entries give the number of papers shared by the two domains.

## B.4 FS Benchmark Construction and Annotation

The 11 reference evolution trees were curated from survey papers that explicitly present evolution diagrams or method-lineage organizations, rather than topic-only taxonomies. This requirement restricts the candidate pool but ensures each reference tree reflects an author-endorsed account of how the field developed, which we standardize into a unified annotation format.

Three graduate students annotated four aspects independently, following a shared guideline: concept membership (which concept node each paper belongs to), evolution edges (directed development relations between concepts), marginal-paper labels (whether a paper is better explained at a coarser abstraction level), and paper depth. Annotators worked from paper titles, abstracts, publication years, and the citation graph, and had access to the source survey.

## C Concept-Labeling Prompt Templates

This appendix provides the full prompt templates used in the bottom-up concept generation pass described in §4.6. Both templates are instantiated per node and consumed by GPT-4.1-mini at temperature 0.1.

## C.1 Leaf Prompt

A leaf node represents a specific research microtopic. The prompt supplies all paper titles with publication years and instructs the model to produce a precise 3–6 word noun phrase naming the shared method or technique, explicitly discouraging generic openers (e.g., Advanced, Large).

## C.2 Internal-Node Prompt

An internal node must generalize over its children. The prompt provides the already-assigned child concept names (bottom-up guarantee) and a sample of representative papers. The model is explicitly required to produce a label that (i) is more abstract than any single child, and (ii) does not repeat any child label verbatim.

## Leaf-node prompt

You are assigning a precise concept label to a leaf cluster in an academic evolutionary taxonomy. The cluster groups closely related papers that share a specific research method, task, or technique.

Papers in this cluster: [list of titles with years]

Note: these papers span YYYY–YYYY. Reflect the temporal focus if it is distinctive.

Requirements:

1. Use 3–6 words; noun phrase, no verbs.

2. Be specific — prefer method/technique names over broad area names.

3. Do NOT start with ‘Advanced’, ‘Large’, or ‘Modern’.

Reply with ONLY the concept name, nothing else.

## Internal-node prompt

You are assigning a concept label to an internal node in an academic evolutionary taxonomy. This node is the common parent of several sub-topics and must be named at a higher level of abstraction than its children.

Child sub-topics: [• child\_label\_1, child\_label\_2, . . . ]

This subtree covers YYYY–YYYY.

Representative papers from this subtree (sample): [up to 5 titles with years]

Requirements:

1. Use 3–6 words; noun phrase, no verbs.

2. The name must generalize over ALL child subtopics listed above.

3. It must be more abstract than any single child concept.

4. Do NOT repeat a child concept name verbatim.

Reply with ONLY the concept name, nothing else.

## D Evaluation Metrics

This appendix specifies the eight evaluation metrics used in §5. We organize them into two parts: indirect metrics computed on citation graphs without annotation, and few-shot metrics computed on the 11 annotated graphs. Table 10 summarizes the role of each metric.

Throughout this appendix, $\tau$ denotes a predicted evolutionary tree with leaf set Leaves $( \mathcal { T } ) =$ $\{ L _ { 1 } , \ldots , L _ { K } \}$ . For each leaf $L _ { k } , \mathcal { M } _ { k }$ denotes the set of papers attached to $L _ { k }$ (i.e., $\mathcal { V } ( L _ { k } )$ restricted to direct attachment). Each paper $v _ { i }$ carries a publication year $\tau ( v _ { i } )$ , a section-label set $\mathbf { y } _ { i } \subseteq \mathcal { V }$ (multihot weak labels from the survey vocabulary), and a section-path set $\mathbf { h } _ { i } \subseteq \mathcal { H }$ from the pretrained section taxonomy. For papers in the annotated set $\mathcal { V } ^ { \mathrm { f s } }$ additional labels include a concept $g _ { i } \in \mathcal { G } ^ { \mathrm { f s } }$ (singlelabel, deterministic), a marginal flag $y _ { i } ^ { \mathrm { m , f s } } \in \{ 0 , 1 \}$ and an evolution edge set $\mathcal E ^ { \mathrm { e v o } }$ over papers. We write $\delta ( v _ { i } )$ for the depth of the concept node to which $v _ { i }$ is attached — whether that node is a leaf or an internal node (root depth = 0). For methods that produce only leaf-level partitions (all baselines in this work), $\delta ( v _ { i } )$ is trivially the depth of the assigned leaf.

## D.1 Indirect metrics on held-out test graphs

LP — Leaf Label Purity. The label purity of a leaf $L _ { k }$ is the fraction of its labeled papers covered

Table 10: Summary of evaluation metrics. $\mathbf { \tilde { \Sigma } } ^ { 6 6 } \mathbf { F } \mathbf { S } ^ { \ast }$ indicates whether the metric requires annotation.
<table><tr><td>Metric</td><td>What it measures</td><td>Signal source</td><td> $\mathbf { E q } .$ </td><td>Stage</td><td>FS?</td></tr><tr><td>LP</td><td>Leaf-cluster topic homogeneity</td><td>Section labels</td><td>Eq. 15</td><td>1</td><td></td></tr><tr><td>PP</td><td>Leaf-cluster path consistency</td><td>Section paths</td><td>Eq. 16</td><td>1</td><td></td></tr><tr><td>NMI</td><td>Cluster-class agreement at depth 2</td><td>Section labels</td><td>Eq. 17</td><td>1</td><td></td></tr><tr><td>CDA</td><td>Citation direction vs. tree depth</td><td>Citation graph</td><td>Eq.18</td><td>2</td><td></td></tr><tr><td>PMR</td><td>Whole-path temporal monotonicity</td><td>Publication years</td><td>Eq. 20</td><td>2</td><td></td></tr><tr><td>FS-CP</td><td>Leaf vs. concept</td><td>Concepts</td><td>Eq. 21</td><td>3</td><td>√</td></tr><tr><td>FS-EDA</td><td>Evolution direction vs. tree depth</td><td>Evolution edges</td><td>Eq. 22</td><td>3</td><td>√</td></tr><tr><td>FS-Mar</td><td>Marginal-paper detection (AUROC)</td><td>Marginal flags</td><td>Eq.23</td><td>3</td><td>√</td></tr></table>

by its single most frequent section label:

$$
\operatorname { P u r i t y } ( L _ { k } ) = { \frac { \operatorname* { m a x } _ { \ell \in \mathcal { N } } { \big | } \{ v _ { i } \in { \mathcal { M } } _ { k } : \ell \in \mathbf { y } _ { i } \} { \big | } } { { \big | } \{ v _ { i } \in { \mathcal { M } } _ { k } : \mathbf { y } _ { i } \neq \emptyset \} { \big | } } } .\tag{14}
$$

LP averages this across leaves:

$$
\mathrm { L P } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathrm { P u r i t y } ( L _ { k } ) .\tag{15}
$$

LP measures topic homogeneity within each leaf cluster.

PP — Leaf Path Purity. PP applies the same definition but operates over section-path labels $\mathbf { h } _ { i }$ instead of label sets ${ \bf y } _ { i } \mathbf { \cdot } \mathbf { \cdot } \mathbf { y }$

$$
\mathrm { P P } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { \operatorname* { m a x } _ { h \in \mathcal { H } } \left| \left\{ v _ { i } \in \mathcal { M } _ { k } : h \in \mathbf { h } _ { i } \right\} \right| } { \left| \left\{ v _ { i } \in \mathcal { M } _ { k } : \mathbf { h } _ { i } \neq \varnothing \right\} \right| } .\tag{16}
$$

PP is stricter than LP: papers grouped in the same leaf must agree not only on labels but on the full hierarchical path leading to them in the survey taxonomy.

NMI — Normalized Mutual Information. To compare predicted trees against flat-clustering baselines under equal granularity, we truncate $\tau$ at depth 2, obtaining a per-paper cluster assignment $\hat { c } _ { i } \in \{ 1 , \ldots , K ^ { \prime } \}$ . The reference class for paper v<sub>i</sub> is its lexicographically smallest label $y _ { i } = \operatorname* { m i n } ( \mathbf { y } _ { i } )$ Restricting to papers with $\mathbf { y } _ { i } \neq \boldsymbol \emptyset$ and $\hat { c } _ { i }$ defined,

$$
\mathrm { N M I } = \frac { 2 I ( \hat { C } ; Y ) } { H ( \hat { C } ) + H ( Y ) } ,\tag{17}
$$

where $I ( \cdot ; \cdot )$ is mutual information and $H ( \cdot )$ is entropy, following the arithmetic-mean normalization in scikit-learn.

CDA — Citation Direction Accuracy. For a citation edge $( v _ { s } , v _ { d } ) \in \mathcal { E } _ { \mathcal { G } }$ in which $v _ { s }$ cites $v _ { d } , v _ { s }$ is typically the more recent work building on $v _ { d }$ . We expect $v _ { s }$ to attach at a depth no shallower than $v _ { d }$ To suppress citations within the same year (which carry weak directional information), we restrict to edges with year gap $\tau ( v _ { s } ) - \tau ( v _ { d } ) \geq 1$

$$
\begin{array} { r l r } & { } & { \mathrm { C D A } = } \\ & { } & { \qquad \left| \left\{ \left( v _ { s } , v _ { d } \right) \in \mathcal { E } _ { \mathcal { G } } : \delta ( v _ { s } ) \ge \delta ( v _ { d } ) , \tau ( v _ { s } ) - \tau ( v _ { d } ) \ge 1 \right\} \right| } \\ & { } & { \qquad \left| \left\{ \left( v _ { s } , v _ { d } \right) \in \mathcal { E } _ { \mathcal { G } } : \tau ( v _ { s } ) - \tau ( v _ { d } ) \ge 1 \right\} \right| } \end{array}\tag{18}
$$

CDA assesses whether the predicted depth structure aligns with the directionality of citations.

PMR — Path Monotonicity Rate. PMR evaluates whether root-to-leaf paths reflect a temporally forward-progressing trajectory at the level of the entire concept body, including descendants. For each concept node $^ { c , }$ we use the subtree-mean time $\begin{array} { r } { \bar { \tau } _ { \mathcal { T } } ( c ) = \frac { 1 } { | \mathcal { V } ( c ) | } \sum _ { v _ { i } \in \mathcal { V } ( c ) } \tau ( v _ { i } ) } \end{array}$ as the path statistic (Eq. 1). Let Π be the set of root-to-leaf paths in $\tau$ containing at least two nodes with valid τ¯. A path $\pi = ( c _ { 0 } , c _ { 1 } , \ldots , c _ { L } ) \in \Pi$ is temporally monotone if

$$
\bar { \tau } _ { \mathcal { T } } ( c _ { 0 } ) \leq \bar { \tau } _ { \mathcal { T } } ( c _ { 1 } ) \leq \cdot \cdot \cdot \leq \bar { \tau } _ { \mathcal { T } } ( c _ { L } ) .\tag{19}
$$

PMR is the fraction of monotone paths:

$$
\mathrm { P M R } = \frac { \left| \left\{ { \pi } \in \Pi : { \pi } \mathrm { ~ i s ~ t e m p o r a l l y ~ m o n o t o n e }  \right\right|}\}   { \left| \Pi \right| } .\tag{20}
$$

## D.2 Few-shot metrics on annotated graphs

FS-CP — FS Concept Purity. FS-CP replicates LP using single-label concepts $g _ { i }$ in place of multilabel section labels:

$$
\mathrm { F S - C P } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { \operatorname* { m a x } _ { g \in \mathcal { G } ^ { \mathrm { f s } } } \left| \left\{ v _ { i } \in \mathcal { M } _ { k } : g _ { i } = g \right\} \right| } { \left| \mathcal { M } _ { k } \right| } .\tag{21}
$$

FS-CP directly measures how well the predicted leaf clusters align with research concepts and is the primary structural-alignment metric in our evaluation.

FS-EDA — FS Evolutionary Direction Accuracy. FS-EDA mirrors CDA but uses annotated evolution edges $\mathcal E ^ { \mathrm { e v o } }$ , in which $( v _ { s } , v _ { d } )$ indicates that $v _ { s }$ is an evolutionary descendant (typically more recent) of v<sub>d</sub>:

$$
\mathrm { F S - E D A } = \frac { \left| \left\{ ( v _ { s } , v _ { d } ) \in \mathcal { E } ^ { \mathrm { e v o } } : \delta ( v _ { s } ) \geq \delta ( v _ { d } ) \right\} \right| } { | \mathcal { E } ^ { \mathrm { e v o } } | } .\tag{22}
$$

Compared with CDA, FS-EDA replaces the noisy citation graph with a clean signal: an edge in $\mathcal E ^ { \mathrm { e v o } }$ reflects $^ { * } v _ { s }$ extends/improves $\boldsymbol { v } _ { d } ^ { \prime \prime }$ , whereas a citation in $\mathcal { E } _ { \mathcal G }$ may indicate any of comparison, background, or methodological reuse.

FS-Mar — FS Marginal AUROC. Our model produces a per-paper marginality score $h _ { i } \in [ 0 , 1 ]$ (the marginal-head output from Eq. 53). With $y _ { i } ^ { \mathrm { m , f s } } \in \{ \overline { { 0 } } , 1 \}$ as ground truth,

$$
\operatorname { F S - M a r } = { \frac { \left| \left\{ \left( v _ { i } , v _ { j } \right) : y _ { i } ^ { \mathrm { m , f s } } = 1 , \ y _ { j } ^ { \mathrm { m , f s } } = 0 , \ h _ { i } > h _ { j } \right\} \right| } { \left| \left\{ \left( v _ { i } , v _ { j } \right) : y _ { i } ^ { \mathrm { m , f s } } = 1 , \ y _ { j } ^ { \mathrm { m , f s } } = 0 \right\} \right| } } ,\tag{23}
$$

which equals the standard AUROC of h against $y ^ { \mathrm { m , f s } }$ . Marginal papers are those occupying domain boundaries: relevant by topic, but whose research focus deviates from the core methodological lineage. FS-Mar evaluates a capability unique to our framework—neither standard clustering baselines nor LLM baselines produce a marginality score (reported as “–” in Table 2; a constant score would give the trivial AUROC of 0.500).

## E Notation Summary

Table 11 summarizes the notation used throughout the paper. Symbols are grouped by category for ease of reference.

## F Graph-Aware Paper Encoder

This appendix expands the abstract encoder $f _ { \theta }$ introduced in §4.2.1 into concrete forward computations and justifies the three design choices discussed in the main text.

## F.1 Forward computation

A pretrained scientific encoder (SPECTER2 (Singh et al., 2022)) produces a fixed semantic representation

$$
\begin{array} { r } { \mathbf { x } _ { i } = f _ { \mathrm { L M } } ( s _ { i } ) \in \mathbb { R } ^ { d _ { \mathrm { L M } } } , } \end{array}\tag{24}
$$

which is then passed through two parallel branches.

Semantic branch. A lightweight two-layer MLP produces the semantic stream

$$
\begin{array} { r } { \mathbf { z } _ { i } ^ { \mathrm { s e m } } = \mathbf { W } _ { 2 } \mathrm { G E L U } ( \mathbf { W } _ { 1 } \mathbf { x } _ { i } ) , } \end{array}\tag{25}
$$

where $\mathbf { W } _ { 1 } \in \mathbb { R } ^ { d \times d _ { \mathrm { L M } } }$ and $\mathbf { W } _ { 2 } \in \mathbb { R } ^ { d \times d }$

Graph branch. A stack of L controlled propagation blocks computes the graph stream. With ${ \bf h } _ { i } ^ { ( 0 ) } = \mathrm { M L P } _ { \mathrm { g r a p h } } ( { \bf x } _ { i } )$ , each layer $l = 0 , \ldots , L - 1$ updates

$$
\mathbf { m } _ { i } ^ { ( l ) } = \frac { 1 } { \left| \mathcal { N } _ { \mathscr { G } } ( v _ { i } ) \right| } \sum _ { j \in \mathcal { N } _ { \mathscr { G } } ( v _ { i } ) } \mathbf { W } _ { \mathrm { m s g } } \mathbf { h } _ { j } ^ { ( l ) } ,\tag{26}
$$

$$
\mathbf { u } _ { i } ^ { ( l ) } = \beta \mathbf { h } _ { i } ^ { ( l ) } + \left( 1 - \beta \right) \mathbf { m } _ { i } ^ { ( l ) } ,\tag{27}
$$

$$
\mathbf { h } _ { i } ^ { ( l + 1 ) } = \mathrm { P a i r N o r m } \Big ( \alpha \mathbf { x } _ { i } + \left( 1 - \alpha \right) \mathbf { u } _ { i } ^ { ( l ) } \Big )\tag{28}
$$

For isolated nodes $v _ { i } \in \mathcal V _ { o }$ where $| { \mathcal { N } } _ { \mathcal { G } } ( v _ { i } ) | = 0 .$ neighborhood aggregation is undefined; we set $\mathbf { m } _ { i } ^ { ( \bar { l } ) } \ = \ \mathbf { 0 }$ so that the propagation reduces to $\mathbf { h } _ { i } ^ { ( \bar { l } + 1 ) } = \mathrm { P a i r N o r m } ( \alpha \mathbf { x } _ { i } + ( 1 - \alpha ) \beta \mathbf { h } _ { i } ^ { ( l ) } )$ , i.e., a pure semantic residual. The final graph-stream output is ${ \bf z } _ { i } ^ { \mathrm { g r a p h } } = { \bf h } _ { i } ^ { ( L ) }$

Adaptive fusion gate. The two streams are combined through a learnable connectivity-biased gate:

$$
\lambda _ { i } = \sigma \Big ( \mathbf { W } _ { g } [ \mathbf { z } _ { i } ^ { \mathrm { g r a p h } } \left. \mathbf { z } _ { i } ^ { \mathrm { s e m } } \right. + \delta \mathcal { N } [ \mathrm { i s o } ( v _ { i } ) ] \Big ) ,\tag{29}
$$

$$
\begin{array} { r } { { \bf z } _ { i } ^ { \mathrm { f u s e d } } = \lambda _ { i } { \bf z } _ { i } ^ { \mathrm { g r a p h } } + \left( 1 - \lambda _ { i } \right) { \bf z } _ { i } ^ { \mathrm { s e m } } . } \end{array}\tag{30}
$$

The clustering embedding is $\mathbf { e } _ { i } = \mathrm { L _ { 2 } N o r m } ( \mathbf { z } _ { i } ^ { \mathrm { f u s e d } } )$ An independent two-layer MLP produces the projection representation ${ \bf p } _ { i } = \mathrm { M L P } _ { \mathrm { p r o j } } ( { \bf z } _ { i } ^ { \mathrm { f u s e d } } )$ , used exclusively by the contrastive objective in §4.3 and never as input to clustering or downstream attachment.

The three design choices flagged in the main text correspond to specific equations above:

• Anchored residual: the $\alpha { \mathbf { x } } _ { i }$ term in Eq. 28 preserves the original semantic vector at every layer, preventing the propagation from collapsing distinct papers to similar representations in shallow citation graphs.

• Density-aware normalization: PairNorm in Eq. 28 centers features and rescales pairwise distances, accommodating the highly uneven cluster densities induced by hub papers.

Table 11: Notation summary. Symbols introduced in the problem formulation (§3) are used consistently across the Methodology and Appendix.
<table><tr><td>Symbol</td><td>Meaning</td><td>Introduced</td></tr><tr><td colspan="3">Input citation graph</td></tr><tr><td> $\mathcal { G } = ( \nu , \mathcal { E } _ { \mathcal { G } } )$ </td><td>Citation graph from survey reference set</td><td>§3</td></tr><tr><td> $v _ { i } \in \mathcal V$ </td><td>A paper node</td><td>§3</td></tr><tr><td> $s _ { i }$ </td><td>Title and abstract text of paper  $v _ { i }$ </td><td>§3</td></tr><tr><td> $\tau ( v _ { i } )$ </td><td>Publication year of paper  $v _ { i }$ </td><td>§3</td></tr><tr><td> $\mathcal { N } _ { \mathcal { G } } ( v _ { i } )$ </td><td>Citation neighborhood of  $v _ { i }$ </td><td> $\mathrm { A p p . F }$ </td></tr><tr><td colspan="3">Trees and concepts</td></tr><tr><td> $\mathcal { T } ^ { \mathrm { t a x } }$ </td><td>Time-agnostic taxonomy backbone</td><td>§3</td></tr><tr><td> $\tau$ </td><td>Target evolutionary tree</td><td>§3</td></tr><tr><td> $\mathcal { T } ^ { ( t ) } , \mathcal { C } ^ { ( t ) }$ </td><td>Tree and concept set at EM iteration t</td><td>§4.4.1</td></tr><tr><td> $c \in { \mathcal { C } }$ </td><td>Concept node (also indexes a cluster)</td><td>§3</td></tr><tr><td> $r$ </td><td>Root concept node</td><td>§4.4.2</td></tr><tr><td> $\mathcal { E } _ { C } , \mathcal { E } _ { P }$ </td><td>Concept edges, paper-attachment edges</td><td>§3</td></tr><tr><td> $\mathcal { V } ( c )$ </td><td>Papers attached to c or its descendants</td><td>§3</td></tr><tr><td> $\mathrm { c h i l d r e n } ( c )$ </td><td>Direct children of c in  $\tau$ </td><td>§4.6</td></tr><tr><td> $\operatorname { L e a v e s } ( \tau )$ </td><td>Leaf concept nodes of T</td><td>§4.4.3</td></tr><tr><td> $\phi ( c )$ </td><td>Concept label of c</td><td>§3</td></tr><tr><td> $\tau ^ { \mathrm { a i t } } { \dot { ( } } c { \bf ) } , { \bar { \tau } } ( c )$ </td><td>Attached-mean / subtree-mean publication year of c</td><td>§3</td></tr><tr><td> $\epsilon$ </td><td>Temporal tolerance margin</td><td>§3</td></tr><tr><td colspan="3">Representations</td></tr><tr><td> $\mathbf { x } _ { i }$ </td><td>Pretrained LM embedding of  $s _ { i }$ </td><td>§4.2.1</td></tr><tr><td> ${ \bf z } _ { i } ^ { \mathrm { s e m } } , { \bf z } _ { i } ^ { \mathrm { g r a p h } } ( = { \bf h } _ { i } ^ { ( L ) } )$ </td><td>Semantic / graph stream output</td><td>§4.2.1</td></tr><tr><td> $\mathbf { e } _ { i }$ </td><td>Fused clustering embedding</td><td>§4.2.1</td></tr><tr><td> $\mathbf { p } _ { i }$ </td><td>Projection representation (contrastive only)</td><td>§4.2.1</td></tr><tr><td> $\mu _ { c } , \sigma _ { c } ^ { 2 }$ </td><td>Diagonal Gaussian parameters of concept c</td><td>§4.2.2</td></tr><tr><td> $w _ { i c }$ </td><td>Soft membership  $p ( v _ { i } \in C _ { c } )$ </td><td>App. G</td></tr><tr><td colspan="3">Marginal mechanism</td></tr><tr><td> $\mathcal { M }$ </td><td>Marginal paper set</td><td>§3</td></tr><tr><td> $m ( v _ { i } , c )$ </td><td>Paper-to-concept compatibility score</td><td>§3</td></tr><tr><td> $A _ { i } ^ { \mathrm { l e a f } }$ </td><td>Maximum leaf compatibility for vi</td><td>§4.4.3</td></tr><tr><td> $\eta$ </td><td>Marginal threshold</td><td> $\ S 3$ </td></tr><tr><td> $s _ { \mathrm { s e m } } , s _ { \mathrm { g r a p h } } , s _ { \mathrm { t i m e } }$ </td><td>Compatibility-score components</td><td>App. K</td></tr><tr><td> $\gamma _ { 1 } , \gamma _ { 2 } , \gamma _ { 3 }$ </td><td>Compatibility-score weights</td><td>App. K</td></tr><tr><td colspan="3">Constraints and constants</td></tr><tr><td>ρ</td><td>Major-branch minimum size ratio</td><td>§4.4.2</td></tr><tr><td> $\operatorname { M B C } ( \tau )$ </td><td>Major Branch Count of  $\tau$ </td><td>§4.4.2</td></tr><tr><td> $[ d _ { \operatorname* { m i n } } , \dot { d } _ { \operatorname* { m a x } } ]$ </td><td>Drift-window bounds</td><td> $\mathrm { A p p . I }$ </td></tr></table>

• Connectivity-biased fusion gate: the term $\delta \mathbb { H } [ \mathrm { i s o } ( v _ { i } ) ]$ in Eq. 29, with δ initialized $\mathrm { t o } - 2 . 0 $ drives $\lambda _ { i }$ toward 0 for isolated nodes so that ${ \bf z } _ { i } ^ { \mathrm { f u s e d } } \approx { \bf z } _ { i } ^ { \mathrm { s e m } }$ at initialization, while remaining learnable.

## G Distributional Tree Builder

## G.1 Base clustering with soft membership

HDBSCAN is applied to the L2-normalized embeddings $\{ \mathbf { e } _ { i } \}$ to identify dense regions. To obtain soft membership, we use the algorithm’s native extension based on the mutual reachability tree (Campello et al., 2013), which yields

$$
w _ { i c } = p ( v _ { i } \in C _ { c } ) \in [ 0 , 1 ] , \quad c = 1 , \ldots , K ,\tag{31}
$$

where K is the number of discovered base clusters. Each cluster’s diagonal Gaussian is fitted by

weighted maximum likelihood:

$$
\mu _ { c } = \frac { \sum _ { i } w _ { i c } { \bf e } _ { i } } { \sum _ { i } w _ { i c } } ,\tag{32}
$$

$$
\sigma _ { c , d } ^ { 2 } = \operatorname* { m a x } \left( \frac { \sum _ { i } w _ { i c } \left( e _ { i , d } - \mu _ { c , d } \right) ^ { 2 } } { \sum _ { i } w _ { i c } } , \sigma _ { \operatorname* { m i n } } ^ { 2 } \right) ,\tag{33}
$$

where d indexes embedding dimensions and $\sigma _ { \mathrm { m i n } } ^ { 2 }$ is a numerical floor preventing degenerate Gaussians.

## G.2 2-Wasserstein distance between diagonal Gaussians

For two diagonal Gaussians $C _ { p } \sim$ $\mathcal { N } ( \mu _ { p } , \mathrm { d i a g } ( \sigma _ { p } ^ { 2 } ) )$ and $C _ { q } \sim \mathcal { N } ( \mu _ { q } , \mathrm { d i a g } ( \pmb { \sigma } _ { q } ^ { 2 } ) )$ the squared 2-Wasserstein distance has a closed form:

$$
{ \mathcal W } _ { 2 } ^ { 2 } ( C _ { p } , C _ { q } ) = \| { \pmb { \mu } } _ { p } - { \pmb { \mu } } _ { q } \| _ { 2 } ^ { 2 } + \sum _ { d } ( \sigma _ { p , d } - \sigma _ { q , d } ) ^ { 2 } .\tag{34}
$$

## G.3 Bottom-up agglomeration

Starting from the K base clusters, we greedily merge the pair $\begin{array} { r l } { ( C _ { p } ^ { \star } , C _ { q } ^ { \star } ) } & { { } = } \end{array}$ arg min $_ { p \neq q } \mathcal { W } _ { 2 } ( C _ { p } , C _ { q } )$ at each step. The merged parent’s Gaussian is reestimated from the union of member papers $\begin{array} { r l r } { { \cal M } } & { { } = } & { \mathrm { m e m b e r s } ( C _ { p } ^ { \star } ) } \end{array}$ ∪ members $( C _ { q } ^ { \star } )$ via unweighted maximum likelihood:

$$
\begin{array} { l } { \displaystyle \mu _ { \mathrm { p a r e n t } } = \frac { 1 } { | M | } \sum _ { i \in M } \mathbf { e } _ { i } , } \\ { \displaystyle \sigma _ { \mathrm { p a r e n t } , d } ^ { 2 } = } \\ { \displaystyle \operatorname* { m a x } \left( \frac { 1 } { | M | } \sum _ { i \in M } ( e _ { i , d } - \mu _ { \mathrm { p a r e n t } , d } ) ^ { 2 } , \sigma _ { \mathrm { m i n } } ^ { 2 } \right) . } \end{array}\tag{35}
$$

Re-estimation relies only on running sums $\textstyle \sum _ { i \in M } \mathbf { e } _ { i }$ and sum of squares $\textstyle \sum _ { i \in M } \mathbf { e } _ { i } ^ { \odot ^ { \bar { 2 } } }$ , which can be incrementally maintained as sufficient statistics during agglomeration. The procedure terminates when a single root cluster remains. Algorithm 1 summarizes the full builder.

## H Stage I Loss Details

This appendix expands the three pre-training loss groups summarized in §4.3:

$$
\begin{array} { r } { \mathcal { L } ^ { \mathrm { t a x } } = \mathcal { L } _ { \mathrm { s e c } } + \mathcal { L } _ { \mathrm { c i t e } } + \mathcal { L } _ { \mathrm { g e o } } . } \end{array}
$$

Algorithm 1 Distributional tree builder.   
Require: Embeddings $\{ \mathbf { e } _ { i } \} ;$ HDBSCAN parameters; (op  
tional) temporal admissibility filter $\mathcal { F } _ { \epsilon } .$   
1: {w<sub>ic</sub>}, K ← HDBSCAN\_SoftMembership $\left( \left\{ \mathbf { e } _ { i } \right\} \right)$   
2: for ${ \dot { c } } = 1 , \dots , K$ do   
3: Compute $\mu _ { c } , \sigma _ { c } ^ { 2 }$ via Eq. 32–33.   
4: end for   
5: ${ \mathcal { C } } \gets \{ C _ { 1 } , \ldots , C _ { K } \} _ { }$ ; record member papers per cluster.   
6: while $| { \mathcal { C } } | > 1$ do   
7: Compute $\mathcal { W } _ { 2 } ( C _ { p } , C _ { q } )$ for all pairs (Eq. 34).   
8: Select $( C _ { p } ^ { \star } , C _ { q } ^ { \star } ) { \dot {  } } \arg$ min $\hat { \mathcal { W } } _ { 2 }$ satisfying $\mathcal { F } _ { \epsilon }$ if pro  
vided.   
9: Re-estimate parent via Eq. 35; remove children, add   
parent.   
10: end while   
11: return Hierarchical tree with $( \mu _ { c } , \sigma _ { c } ^ { 2 } )$ at every node.

## H.1 Section supervision $\mathcal { L } _ { \mathrm { s e c } }$

Survey section structure supplies weak labels: each paper $v _ { i }$ inherits a multi-hot section vector $\mathbf { y } _ { i } ^ { \mathrm { { \bar { l b l } } } ^ { - } } \in \quad \{ 0 , 1 \} ^ { | S | }$ over the union of section labels ${ \mathcal { S } } ,$ plus a hierarchical level $y _ { i } ^ { \mathrm { l v l } } \ \in$ $\{ 1 , \ldots , L _ { \operatorname* { m a x } } \}$ . Let $\hat { \mathbf { y } } _ { i } ^ { \mathrm { l b l } } = \mathrm { s i g m o i d } ( \mathbf { W } _ { \mathrm { l b l } } \mathbf { e } _ { i } )$ and $\hat { \mathbf { y } } _ { i } ^ { \mathrm { l v l } } = \mathrm { s o f t m a x } ( \mathbf { W } _ { \mathrm { l v l } } \mathbf { e } _ { i } )$ . We use masked binary cross-entropy and categorical cross-entropy over labeled nodes $\begin{array} { r } { \nu _ { \mathrm { l b l } } \subseteq \mathcal { V } : } \end{array}$

$$
\mathcal { L } _ { \mathrm { l a b e l } } = - \frac { 1 } { | \mathcal { V } _ { \mathrm { l b l } } | } \sum _ { i \in \mathcal { V } _ { \mathrm { l b l } } } \mathrm { B C E } \left( \hat { \mathbf { y } } _ { i } ^ { \mathrm { l b l } } , \mathbf { y } _ { i } ^ { \mathrm { l b l } } \right) ,\tag{36}
$$

$$
\mathcal { L } _ { \mathrm { l e v e l } } = - \frac { 1 } { \vert \mathcal { V } _ { \mathrm { l b l } } \vert } \sum _ { i \in \mathcal { V } _ { \mathrm { l b l } } } \log \hat { y } _ { i , y _ { i } ^ { \mathrm { l v l } } } ^ { \mathrm { l v l } } ,\tag{37}
$$

$$
\mathcal { L } _ { \mathrm { s e c } } = w _ { \mathrm { l b l } } \mathcal { L } _ { \mathrm { l a b e l } } + w _ { \mathrm { l v l } } \mathcal { L } _ { \mathrm { l e v e l } } .\tag{38}
$$

## H.2 Citation consistency $\mathcal { L } _ { \mathrm { c i t e } }$

Two terms preserve citation structure: link prediction on $\mathbf { e } _ { i }$ and contrastive ranking on $\mathbf { p } _ { i }$

Link prediction. For each observed edge $( v _ { i } , v _ { j } ) \in \mathcal { E } _ { \mathcal { G } }$ (positive) and randomly sampled nonedge $( v _ { i } , v _ { j ^ { - } } )$ (negative), define the logit $\hat { y } _ { i j } =$ $\tau _ { \mathrm { l n k } } \mathbf { e } _ { i } ^ { \top } \mathbf { e } _ { j }$ with temperature $\tau _ { \mathrm { l n k } }$ (we use 10 to scale cosine similarities into a useful logit range):

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { l i n k } } = \mathrm { B C E } ( \mathrm { s i g m o i d } ( \hat { y } _ { i j } ) , y _ { i j } ) , } \end{array}\tag{39}
$$

where $y _ { i j } \in \{ 0 , 1 \}$ indicates edge existence. We use one negative per positive.

Contrastive ranking on $\mathbf { p } _ { i }$ . For two papers $v _ { i } , v _ { j }$ sharing a section label, they form a positive pair. Hard negatives $v _ { k ^ { - } }$ are mined as the most similar paper to $v _ { i }$ not sharing any label of $v _ { i }$ . With cosine similarity sim $( a , b ) = \mathbf { p } _ { a } ^ { \top } \mathbf { p } _ { b } / ( \| \mathbf { p } _ { a } \| \| \mathbf { p } _ { b } \| )$

and temperature $\tau _ { \mathrm { c t r } } \colon$

$$
\begin{array} { r l } {  { \mathcal { L } _ { \mathrm { c o n t r a s t } } = } } \\ & { - \log \frac { \exp ( \sin ( v _ { i } , v _ { j } ) / \tau _ { \mathrm { c t r } } ) } { \sum _ { k \in \{ j \} \cup K _ { i } ^ { - } } \exp ( \sin ( v _ { i } , v _ { k } ) / \tau _ { \mathrm { c t r } } ) } , } \end{array}\tag{40}
$$

with $\kappa _ { i } ^ { - }$ <sup>−</sup> the hard-negative set $( | K _ { i } ^ { - } | = 8 )$ . Note that $\mathcal { L } _ { \mathrm { { c o n t r a s t } } }$ acts on $\mathbf { p } _ { i }$ and never on $\mathbf { e } _ { i } .$ , keeping the clustering geometry untouched.

Combination.

$$
\mathcal { L } _ { \mathrm { c i t e } } = w _ { \mathrm { l n k } } \mathcal { L } _ { \mathrm { l i n k } } + w _ { \mathrm { c t r } } \mathcal { L } _ { \mathrm { c o n t r a s t } } .\tag{41}
$$

## H.3 Cluster geometry $\mathcal { L } _ { \mathrm { g e o } }$

A prototype-anchoring term pulls each labeled node toward its class centroid. For each label $s \in S$ , let $\mathcal { V } _ { s } = \{ v _ { i } : s \in \mathbf { y } _ { i } ^ { \mathrm { l b l } } \}$ and define the stop-gradient centroid

$$
\bar { \bf e } _ { s } = \mathrm { s g } \left( \frac { 1 } { | \mathcal { V } _ { s } | } \sum _ { i \in \mathcal { V } _ { s } } \mathbf { e } _ { i } \right) .\tag{42}
$$

The anchor loss is:

$$
\mathcal { L } _ { \mathrm { a n c h o r } } = \frac { 1 } { | \mathcal { S } | } \underset { s \in \mathcal { S } } { \sum } \frac { 1 } { | \mathcal { V } _ { s } | } \underset { i \in \mathcal { V } _ { s } } { \sum } \Bigl ( 1 - \mathbf { e } _ { i } ^ { \top } \bar { \mathbf { e } } _ { s } \Bigr ) .\tag{43}
$$

L<sub>geo</sub> = w<sub>anc</sub>L<sub>anchor</sub>.

## I Stage II Loss Details

This appendix expands the three evolution loss groups summarized in §4.4.1:

$$
\mathcal { L } ^ { \mathrm { e v o } } = \mathcal { L } _ { \mathrm { s e m } } + \mathcal { L } _ { \mathrm { t e m p } } + \mathcal { L } _ { \mathrm { m a r g } } .
$$

All terms operate on the current tree $\mathscr { T } ^ { ( t ) }$ with concept Gaussians $\{ ( \pmb { \mu } _ { c } , \pmb { \sigma } _ { c } ^ { 2 } ) \} _ { c \in \mathcal { C } ^ { ( t ) } }$ . Let $p ( c )$ denote the parent of concept c and $p ( i )$ the parent concept of paper $v _ { i }$ under attachment a.

## I.1 Semantic evolution consistency $\mathcal { L } _ { \mathrm { s e m } }$

Three sub-terms enforce parent–child semantic coherence.

Node-to-parent pull. Each paper is pulled toward the centroid of its parent concept:

$$
\mathcal { L } _ { \mathrm { s e m - n } } = \frac { 1 } { | \mathcal { V } | } \sum _ { i \in \mathcal { V } } \Bigl ( 1 - \cos ( \mathbf { e } _ { i } , \pmb { \mu } _ { p ( i ) } ) \Bigr ) .\tag{44}
$$

Cluster-to-parent pull. Each concept is pulled toward the centroid of its parent concept:

$$
\frac { \mathcal { L } _ { \mathrm { s e m - c } } = } { | \mathcal { C } ^ { ( t ) } \setminus \{ r \} | _ { c \in \mathcal { C } ^ { ( t ) } \setminus \{ r \} } } \Big ( 1 - \cos ( \pmb { \mu } _ { c } , \pmb { \mu } _ { p ( c ) } ) \Big ) .\tag{45}
$$

Drift window. Let $\delta _ { c } = \| \pmb { \mu } _ { c } - \pmb { \mu } _ { p ( c ) } \| _ { 2 }$ . A twosided hinge confines $\delta _ { c }$ to $[ d _ { \operatorname* { m i n } } , d _ { \operatorname* { m a x } } ] \colon$

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { d r i f t } } = } \\ { \displaystyle \sum _ { c \in \mathcal { C } ^ { ( t ) } \setminus \{ r \} } \big [ \operatorname* { m a x } ( 0 , d _ { \operatorname* { m i n } } - \delta _ { c } ) + \operatorname* { m a x } ( 0 , \delta _ { c } - d _ { \operatorname* { m a x } } ) \big ] . } \end{array}\tag{46}
$$

Combination.

$$
\mathcal { L } _ { \mathrm { s e m } } = w _ { \mathrm { s e m n } } \mathcal { L } _ { \mathrm { s e m - n } } + w _ { \mathrm { s e m c } } \mathcal { L } _ { \mathrm { s e m - c } } + w _ { \mathrm { d r i f t } } \mathcal { L } _ { \mathrm { d r i f t } } .\tag{47}
$$

## I.2 Temporal coherence $\mathcal { L } _ { \mathrm { t e m p } }$

Three sub-terms enforce forward-progressing evolution.

Path smoothness. For a root-to-leaf concept path $P = ( c _ { 0 } , c _ { 1 } , \dots , c _ { L } )$ , define velocity $\Delta _ { l } = \pmb { \mu } _ { c _ { l } } -$ $\mu _ { c _ { l - 1 } }$ . We penalize second-order changes:

$$
\mathcal { L } _ { \mathrm { s m o o t h } } = \frac { 1 } { | \mathcal { P } | } \sum _ { P \in \mathcal { P } } \frac { 1 } { L - 1 } { \sum _ { l = 2 } ^ { L } } \Vert \Delta _ { l } - \Delta _ { l - 1 } \Vert _ { 2 } ^ { 2 } ,\tag{48}
$$

where $\mathcal { P }$ is the set of root-to-leaf paths in $\mathscr { T } ^ { ( t ) }$

Path monotonicity. Let $\begin{array} { r } { \tau _ { T } ( c ) = \operatorname* { m i n } _ { v \in \mathcal { V } ( c ) } \tau ( v ) } \end{array}$ denote the earliest publication year in the subtree of c (also used in Appendix J). A hinge encourages child concepts to be no earlier than parents:

$$
\mathcal { L } _ { \mathrm { p a t h - m o n o } } = \sum _ { ( c _ { p } , c _ { c } ) \in \mathcal { E } _ { C } } \operatorname* { m a x } ( 0 , \tau _ { T } ( c _ { p } ) - \tau _ { T } ( c _ { c } ) - \epsilon ) .\tag{49}
$$

Citation directionality. For each citation $( v _ { i } , v _ { j } ) \in \mathcal { E } _ { \mathcal { G } }$ with $v _ { i }$ citing $v _ { j }$ , the citing paper should attach to a deeper node. Let $q _ { i , c }$ be the soft attachment probability of $v _ { i }$ to concept c:

$$
q _ { i , c } = \frac { \exp \bigl ( - \| \mathbf { e } _ { i } - \pmb { \mu } _ { c } \| _ { 2 } ^ { 2 } / \tau _ { \mathrm { a t t } } \bigr ) } { \sum _ { c ^ { \prime } \in \mathcal { C } ^ { ( t ) } } \exp \bigl ( - \| \mathbf { e } _ { i } - \pmb { \mu } _ { c ^ { \prime } } \| _ { 2 } ^ { 2 } / \tau _ { \mathrm { a t t } } \bigr ) } .\tag{50}
$$

Define the expected depth $\bar { d } ( v _ { i } )$ = $\textstyle \sum _ { c } q _ { i , c }$ depth(c). The citation directionality loss is:

$$
\mathcal { L } _ { \mathrm { c i t e - d i r } } = \sum _ { ( v _ { i } , v _ { j } ) \in \mathcal { E } _ { \mathcal { G } } } \operatorname* { m a x } ( 0 , \bar { d } ( v _ { j } ) - \bar { d } ( v _ { i } ) + \xi ) ,\tag{51}
$$

penalizing cases where the cited paper $v _ { j }$ ends up at least $\xi$ levels deeper than the citing v<sub>i</sub>.

Combination.

$$
\mathcal { L } _ { \mathrm { t e m p } } = \mathcal { L } _ { \mathrm { s m o o t h } } + \mathcal { L } _ { \mathrm { p a t h - m o n o } } + \mathcal { L } _ { \mathrm { c i t e - d i r } } .\tag{52}
$$

## I.3 Marginal identification and re-attachment ${ \mathcal { L } } _ { \mathrm { m a r g } }$

Let $h _ { i } = \mathrm { M L P } _ { \operatorname* { m a r } } ( \mathbf { e } _ { i } ) \in [ 0 , 1 ]$ be the marginalhead output. Pseudo-labels are $\tilde { y } _ { i } ^ { \mathrm { m } } = { \nVdash } [ A _ { i } ^ { \mathrm { l e a f } } ~ <$ $\eta ]$

Marginal classification.

$$
\mathcal { L } _ { \mathrm { m a r } } = - \frac { 1 } { | \mathcal { V } | } \sum _ { i \in \mathcal { V } } \bigl [ \tilde { y } _ { i } ^ { \mathrm { m } } \log h _ { i } + ( 1 - \tilde { y } _ { i } ^ { \mathrm { m } } ) \log ( 1 - h _ { i } ) \bigr ] .\tag{53}
$$

Attachment ranking. For each flagged marginal paper $v _ { m } \in \mathcal { M } ^ { ( t ) }$ , let $c _ { m } ^ { + } = a ( v _ { m } )$ (the optimal attachment from Eq. 11) and c<sup>−</sup> a candidate concept that fails the temporal admissibility check. With score

$$
\begin{array} { r l r } {  { s _ { \mathrm { a t t } } ( c , m ) = \log \mathcal { N } \big ( \mathbf { e } _ { m } ; \pmb { \mu } _ { c } , \mathrm { d i a g } ( \pmb { \sigma } _ { c } ^ { 2 } ) \big ) , } } \\ & { } & { \ \mathcal { L } _ { \mathrm { a t t } } ^ { \mathrm { r a n k } } = } \\ & { } & { \ \displaystyle \sum _ { v _ { m } \in \mathcal { M } ( { t } ) c ^ { - } } \operatorname* { m a x } \big ( 0 , s _ { \mathrm { a t t } } ( c ^ { - } , m ) - s _ { \mathrm { a t t } } ( c _ { m } ^ { + } , m ) + \zeta \big ) , } \end{array}
$$

with margin $\zeta .$

(54)

Centroid repulsion. To prevent internal concept centroids from collapsing, sibling concepts are encouraged to maintain pairwise separation. For each internal concept c with children children(c):

$$
\sum _ { \stackrel { c } { c } } \sum _ { \stackrel { c _ { a } , c _ { b } \in \operatorname { c h i l d r e n } ( c ) } } ^ { \mathcal { L } _ { \mathrm { r e p u l s i o n } } } \operatorname* { m a x } \big ( 0 , \zeta _ { r } - \| \pmb { \mu } _ { c _ { a } } - \pmb { \mu } _ { c _ { b } } \| _ { 2 } \big ) ,\tag{55}
$$

with separation margin $\zeta _ { r }$

Combination.

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { m a r g } } = \mathcal { L } _ { \mathrm { m a r } } + \mathcal { L } _ { \mathrm { a t t } } ^ { \mathrm { r a n k } } + \mathcal { L } _ { \mathrm { r e p u l s i o n } } . } \end{array}\tag{56}
$$

## J Structural Constraints

The main text describes the three most consequential constraints (temporal consistency, backbone preservation, tree legality). Here we formalize the two remaining constraints deferred from §4.4.2.

## J.1 Path temporal consistency

For every root-to-leaf concept path $( c _ { 0 } , c _ { 1 } , \dots , c _ { L } )$ in T ,

$$
\tau _ { T } ( c _ { 0 } ) \leq \tau _ { T } ( c _ { 1 } ) + \epsilon \leq \cdot \cdot \cdot \leq \tau _ { T } ( c _ { L } ) + L \epsilon ,\tag{57}
$$

where $\begin{array} { r } { \tau _ { T } ( c ) = \operatorname* { m i n } _ { v \in \mathcal { V } ( c ) } \tau ( v ) } \end{array}$ and ϵ is the peredge tolerance margin. This rules out global temporal inversions in which local pairs satisfy the temporal order but the path as a whole accumulates a violation.

## J.2 Path membership coherence

For every concept $c \in \mathcal { C }$ , its evolutionary ancestor chain $\operatorname { a n c } ^ { \mathrm { e v o } } ( c ) = ( c , p ( c ) , p ( p ( c ) ) , \dots , r )$ must retain non-trivial distributional overlap with its taxonomy-ancestor chain $\mathrm { a n c } ^ { \mathrm { t a x } } ( c )$ . Formally, defining overlap as the maximum normalized intersection between any ancestor in the two chains,

$$
\Omega ( c ) = \operatorname* { m a x } _ { \substack { c _ { e } \in \mathrm { a n c } ^ { \mathrm { e v o } } ( c ) , c _ { t } \in \mathrm { a n c } ^ { \mathrm { t a x } } ( c ) } } \frac { | \mathcal { V } ( c _ { e } ) \cap \mathcal { V } ( c _ { t } ) | } { \operatorname* { m i n } ( | \mathcal { V } ( c _ { e } ) | , | \mathcal { V } ( c _ { t } ) | ) } ,\tag{58}
$$

and requiring $\Omega ( c ) \geq \theta _ { \mathrm { o v } }$ . We use $\theta _ { \mathrm { o v } } = 0 . 3$ . This constraint prevents the temporal reorientation in Stage II from rewiring a concept into a semantically unrelated branch.

## K Marginal Compatibility Score

This appendix specifies the compatibility score $m ( v _ { i } , c )$ in $\ S 4 . 4 . 3$

For paper $v _ { i }$ and concept c with Gaussian $( \mu _ { c } , \sigma _ { c } ^ { 2 } )$ and member set $\mathcal { V } ( c )$

Semantic similarity.

$$
s _ { \mathrm { s e m } } ( i , c ) = \frac { 1 } { 2 } ( 1 + \cos ( { \mathbf e } _ { i } , \mu _ { c } ) ) \in [ 0 , 1 ] .\tag{59}
$$

Citation-neighborhood overlap.

$$
s _ { \mathrm { g r a p h } } ( i , c ) = \frac { | \mathcal { N } _ { \mathcal { G } } ( v _ { i } ) \cap \mathcal { V } ( c ) | } { \operatorname* { m a x } ( | \mathcal { N } _ { \mathcal { G } } ( v _ { i } ) | , 1 ) } \in [ 0 , 1 ] .\tag{60}
$$

Temporal compatibility. Let $\begin{array} { r l r l } { \bar { \tau } _ { c } } & { { } } & { = } & { } \end{array}$ $\begin{array} { r } { | \mathcal { V } ( c ) | ^ { - 1 } { \sum _ { v \in \mathcal { V } ( c ) } } \tau ( v ) } \end{array}$ be the nominal time of c. With time-scale $\sigma _ { t } .$

$$
s _ { \mathrm { t i m e } } ( i , c ) = \exp \left( - \frac { | \tau ( v _ { i } ) - \bar { \tau } _ { c } | } { \sigma _ { t } } \right) \in ( 0 , 1 ] .\tag{61}
$$

Combined compatibility.

$$
\begin{array} { r } { \begin{array} { c } { m ( v _ { i } , c ) = } \\ { \gamma _ { 1 } s _ { \mathrm { s e m } } ( i , c ) + \gamma _ { 2 } s _ { \mathrm { g r a p h } } ( i , c ) + \gamma _ { 3 } s _ { \mathrm { t i m e } } ( i , c ) , } \end{array} } \end{array}\tag{62}
$$

with $\gamma _ { 1 } + \gamma _ { 2 } + \gamma _ { 3 } = 1 { \mathrm { a n d } } m ( v _ { i } , c ) \in [ 0 , 1 ] .$

## L Stage III Calibration Loss Details

This appendix expands the three calibration loss terms summarized in §4.5:

$$
\begin{array} { r } { \mathcal { L } ^ { \mathrm { c a l } } = \mathcal { L } _ { \mathrm { m a r g } } ^ { \mathrm { c a l } } + \mathcal { L } _ { \mathrm { c o n c e p t } } ^ { \mathrm { c a l } } + \mathcal { L } _ { \mathrm { e d g e } } ^ { \mathrm { c a l } } . } \end{array}
$$

The FS dataset (§5.1) provides per-paper labels $y _ { i } ^ { \mathrm { m , f s } } \in \{ 0 , 1 \}$ (marginal), $g _ { i } ~ \in ~ \mathcal { G } ^ { \mathrm { f s } }$ (concept), and the evolution edge set $\mathcal E ^ { \mathrm { e v o } }$ over papers $( \mathsf { A p - }$ pendix D).

## L.1 FS marginal classification

Class-weighted BCE to handle the imbalance between core and marginal papers:

$$
\begin{array} { r l } { \displaystyle \mathcal { L } _ { \mathrm { m a r g } } ^ { \mathrm { c a l } } = - \frac { 1 } { | \mathcal { V } ^ { \mathrm { f s } } | } \sum _ { i \in \mathcal { V } ^ { \mathrm { f s } } } \big [ w ^ { + } y _ { i } ^ { \mathrm { m , f s } } \log h _ { i } } & { } \\ { \displaystyle + w ^ { - } ( 1 - y _ { i } ^ { \mathrm { m , f s } } ) \log ( 1 - h _ { i } ) \big ] , } \end{array}\tag{63}
$$

with $w ^ { + } = ( 1 - \bar { y } ^ { \mathrm { m } } ) / \bar { y } ^ { \mathrm { m } } , w ^ { - } = 1$ , and $\bar { y } ^ { \mathrm { m } }$ the marginal fraction in the FS set.

## L.2 FS concept alignment

For each FS concept $g \in \mathcal { G } ^ { \mathrm { f s } }$ , define its empirical centroid

$$
\bar { \bf e } _ { g } = \mathrm { s g } \left( \frac { 1 } { | \mathcal { V } _ { g } ^ { \mathrm { f s } } | } \sum _ { i \in \mathcal { V } _ { g } ^ { \mathrm { f s } } } \mathbf { e } _ { i } \right) ,\tag{64}
$$

with $\mathcal { V } _ { g } ^ { \mathrm { f s } } = \{ v _ { i } : g _ { i } = g \}$ . An InfoNCE objective aligns each paper with its FS centroid:

$$
\mathcal { L } _ { \mathrm { c o n c e p t } } ^ { \mathrm { c a l } } = - \frac { 1 } { \vert \mathcal { V } ^ { \mathrm { f s } } \vert } \sum _ { i \in \mathcal { V } ^ { \mathrm { f s } } } \log \frac { \exp ( \mathbf { e } _ { i } ^ { \top } \bar { \mathbf { e } } _ { g _ { i } } / \tau _ { \mathrm { i c } } ) } { \sum _ { g ^ { \prime } \in \mathcal { G } ^ { \mathrm { f s } } } \exp ( \mathbf { e } _ { i } ^ { \top } \bar { \mathbf { e } } _ { g ^ { \prime } } / \tau _ { \mathrm { i c } } ) } .\tag{65}
$$

## L.3 FS edge directionality

For each annotated edge $( v _ { s } , v _ { d } ) \in \mathcal { E } ^ { \mathrm { e v o } } , v _ { s }$ is the evolutionary descendant of $v _ { d }$ . A triplet-margin loss requires the predicted depth of $v _ { s }$ to exceed that of $v _ { d } .$ , with cross-domain adapts relations receiving a smaller margin:

$$
\mathcal { L } _ { \mathrm { e d g e } } ^ { \mathrm { c a l } } = \sum _ { ( v _ { s } , v _ { d } ) \in \mathcal { E } ^ { \mathrm { e v o } } } \operatorname* { m a x } \bigl ( 0 , \bar { d } ( v _ { d } ) - \bar { d } ( v _ { s } ) + \mu ( v _ { s } , v _ { d } ) \bigr ) ,\tag{66}
$$

where $\bar { d } ( \cdot )$ is the expected depth from Eq. 50, and

$$
\mu ( v _ { s } , v _ { d } ) = \left\{ { \begin{array} { l l } { \mu _ { \mathrm { s t r i c t } } } & { \mathrm { e x t e n d s / \ i m p r o v e s , } } \\ { \mu _ { \mathrm { r e l a x } } } & { \mathrm { a d a p t s . } } \end{array} } \right.
$$

## M EM-style Optimization

This appendix details the E-step / M-step alternation summarized in §4.4.1.

E-step frequency. In practice we run the E-step once per epoch. Running it more frequently (e.g., every 100 steps) did not improve performance and increased compute by $\sim 3 \times$

Temporal admissibility filter $\begin{array} { r l } { \mathcal { F } _ { \epsilon } . ~ \mathsf { A } } \end{array}$ candidate merge of clusters $C _ { p } , C _ { q }$ is admissible if for each subsequent merging step the resulting parent does not violate parental temporal ordering by more than $\epsilon = 1$ year, where the nominal time of a cluster is $\tau ^ { \mathrm { a t t } }$ (Eq. 1), the mean publication year of its directly attached papers, as in Eq. 8.

Algorithm 2 Stage II EM-style optimization.   
Require: Pre-trained encoder $f _ { \theta }$ with adapters $\theta _ { a } ;$ taxonomy   
backbone $\mathcal { T } ^ { \mathrm { t a x } }$ ; epochs $T .$   
1: Initialize $\mathcal { T } ^ { ( 0 ) }  \mathcal { T } ^ { \mathrm { t a x } } .$   
2: for $t = 1 , \dots , T$ do   
3: E-step:   
4: Freeze $\theta _ { a } ;$ compute $\left\{ \mathbf { e } _ { i } \right\}$ via current encoder.   
5: Run Algorithm 1 with temporal admissibility filter   
${ \mathcal { F } } _ { \epsilon } .$   
6: Obtain $\mathcal { T } ^ { ( t ) } , \mathcal { C } ^ { ( t ) }$ and identify marginal candidates   
$\mathcal { M } ^ { ( t ) } .$   
7: Generate pseudo-labels $\{ \tilde { y } _ { i } ^ { \mathrm { m } } \}$ via Eq. 3.   
8: M-step:   
9: Freeze $\mathcal { T } ^ { ( t ) } ;$ update $\theta _ { a }$ by one epoch of SGD on   
${ \mathcal { L } } ^ { \mathrm { e v o } } .$   
10: end for   
11: return Adapters $\theta _ { a } ,$ final tree $\mathcal { T } = \mathcal { T } ^ { ( T ) }$

## N Generation-quality Evaluation Protocol

## N.1 Evaluation Setup

Both protocols receive identical supporting information: paper titles, publication years, citation relations, and the anonymized generated hierarchy. Method names, model descriptions, and metric results are withheld. To reduce presentation bias, structures are assigned randomized anonymous identifiers and presented in randomized order. The LLM judge is the same model used for concept labeling, evaluating each structure over multiple independent runs under a fixed prompt and decoding configuration. Human evaluation is conducted by three CS graduate students who did not participate in constructing the FS annotations; they evaluate each structure independently and do not discuss individual cases before submitting their judgments.

## N.2 Criteria and Aggregation

Both protocols score four dimensions—conceptual organization, scientific evolution, transitionalpaper placement, and overall explanatory quality— on a 1–10 scale under the same rubric, each accompanied by a brief evidence-based justification. Scores are averaged over evaluation runs (LLM) or evaluators (human), then over the FS domains, and we report the mean and standard deviation per method and dimension in Table 5.

The following prompt is given to the LLM judge; human evaluators receive the same criteria in written form.

## Generation-quality evaluation prompt

You are an expert researcher. Given a citation graph and a generated hierarchy, evaluate the hierarchy on the following aspects.

Citation graph: [titles, publication years, citation relations]

Generated hierarchy: [anonymized structure]

Aspects:

1. Conceptual Organization — Are papers grouped into semantically coherent concepts? Are concept names accurate?

2. Scientific Evolution — Does the hierarchy reflect the historical development of the field? Are parent–child relationships reasonable?

3. Transitional Papers — Are transitional / bridge papers attached at an appropriate position?

4. Overall Quality — How well does the hierarchy explain the organization and evolution of this research field?

Give a score from 1–10 for each aspect and provide a brief explanation.