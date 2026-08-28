# BLANC: Discovering Patent White Space via Changes in Normalized Pointwise Mutual Information Between Multi-View Clusters

Shuichi Miyazawa<sup>a,∗</sup>, Kensuke Fujii<sup>b</sup>

<sup>a</sup>Digital & Innovation Management Division, AGC Inc., Tokyo, Japan

<sup>b</sup>Intellectual Property Division, AGC Inc., Tokyo, Japan

## A R T I C L E I N F O

Keywords:   
Patent analysis   
Patent landscaping   
White space analysis   
Technology opportunity discovery   
Topic modeling   
Co-occurrence analysis   
Document clustering

## A BS T RA C T

Identifying white space, the unexplored but potentially valuable regions of a patent landscape, is essential for strategic R&D planning, yet existing methods either rely on manual patent mapping or apply single-view clustering without quantitative gap detection. We propose BLANC (Blank Landscape Analysis through NPMI Conditioning), a three-phase pipeline that combines (1) multiview neural topic modeling along three semantic dimensions (application/use, novelty, inventive step); (2) Normalized Pointwise Mutual Information (NPMI) to quantify cross-dimensional cluster association; and (3) conditional detection that flags combinations whose NPMI drops when the corpus is filtered by a user-specified keyword. The drop is captured by a new metric, ΔNPMI, which identifies combinations “established globally, unexplored locally.” Because white space has no ground truth in principle, we evaluate BLANC on two public USPTO corpora—machine learning/AI (5,417 patents, CPC G06N) and glass compositions (1,982 patents, CPC C03C)—by artificially depleting known technology combinations and testing whether BLANC recovers them. When three-quarters of a target pair’s documents are removed, BLANC recovers 34.1% (ML/AI) and 27.3% (glass) of the depleted combinations, whereas size-matched removals not aimed at them (random documents, or those of a diferent established combination) essentially never do: the target is never recovered in 191 decoy trials. Collapsing the three semantic views into one recovers nothing, while prior cooccurrence measures also flag the target under random removal, ofering no specificity. Furthermore, a proprietary-data case (302 float glass / glass-ceramics patents) shows that, with the keyword “fluorine,” BLANC reveals a fluorine surface treatment × warpage suppression candidate (ΔNPMI up to 0.48) that experts had independently identified. The method enables eficient, reproducible preparation for patent brainstorming in R&D and IP departments.

## 1. Introduction

In industrial R&D and intellectual property (IP) departments, brainstorming sessions for new patent application themes are a critical activity. However, the preparatory process of reading large volumes of related patents sufers from several ineficiencies:

1. reading patents without a clear objective is laborintensive and dificult to motivate;

2. brainstorming quality depends heavily on individual domain expertise, leading to inconsistent outcomes across participants;

3. traditional patent maps<sup>1</sup> are time-consuming to create, require group efort, and often yield few actionable insights, since a blank region may be technologically meaningless; and

4. standard AI-based classification approaches do not incorporate an organization’s own technical knowledge, yielding only generic perspectives.

This paper addresses the problem of automatically finding “what should exist but doesn’t” in patent landscapes. We propose BLANC (Blank Landscape Analysis through NPMI Conditioning), a system that discovers technology theme combinations that are well-represented in the broader patent population but absent or underrepresented when restricted to the user’s keyword of interest. These weakened associations represent candidate themes for new patent applications: the white space.

Prior work approaches patent white space along three lines. Early morphological and map-based methods flag empty cells in keyword or classification matrices [1, 2, 3]; cluster- and function-based methods infer gaps from inter-cluster connectivity or Subject–Action–Object (SAO) structures (subject–verb–object triples extracted from patent text) [4, 5, 6]; and recent neural approaches apply transformerbased topic models such as BERTopic [7] to technologyopportunity discovery [8, 9]. A parallel line quantifies how strongly technology pairs co-occur, using Pointwise Mutual Information (PMI) and its normalized form NPMI [10, 11] or IPC co-classification statistics [12, 13].

Yet no method combines the three ingredients we argue are jointly necessary: multi-view clustering, a normalized cross-dimensional co-occurrence measure, and keywordconditional gap detection. The closest precursors each address only part of this: Kim et al. [4] use traditional clustering without a normalized association measure, while Jeon et al. [8] use single-view clustering without statistical gap detection (Table 1). BLANC closes this gap.

The approach consists of three phases:

Phase 1 (Multi-view clustering): Patent documents are independently clustered along three semantic dimensions: application/use, novelty, and inventive step. The BERTopic neural topic modeling pipeline performs the clustering, embedding documents with Sentence-BERT, reducing their dimensionality with UMAP, and grouping them with HDB-SCAN.<sup>2</sup>

![](images/a3cef850a125ca48ba864cbf4cee7c448f37ef9e4d3f86707ab5a37caf234705.jpg)  
Figure 1: System architecture and data flow of the BLANC pipeline. Phase 1 independently clusters patents along three semantic dimensions using BERTopic. Phase 2 constructs a 3D co-occurrence tensor and computes pairwise NPMI matrices. Phase 3 filters by a user keyword, recomputes conditional NPMI, and ranks cluster pairs by ΔNPMI.

Phase 2 (Co-occurrence quantification): Statistical association between clusters from diferent dimensions is computed via Normalized Pointwise Mutual Information (NPMI), producing pairwise NPMI matrices that capture strongly related theme combinations.

Phase 3 (White space detection): When the user specifies a keyword, the corpus is filtered and conditional NPMI is recomputed. Cluster pairs whose NPMI drops markedly are flagged as white space using a new metric, the NPMI drop (ΔNPMI).

The contributions of this paper are threefold:

1. Methodological: We propose the BLANC pipeline, the first to integrate multi-view BERTopic clustering with NPMI co-occurrence analysis for patent white space detection.

2. Conceptual: We introduce the NPMI drop (ΔNPMI), a metric that operationalizes white space as “established globally, unexplored locally” by computing the diference in NPMI between the full corpus and a keyword-filtered subcorpus.

3. Practical: We evaluate BLANC on two public USPTO corpora spanning distinct technology domains (ML/AI and glass compositions) under a circularity-robust, removal-driven protocol. BLANC recovers injected white space, whereas size-matched control removals (random documents, or the documents of a diferent established combination) essentially do not. Under the identical protocol, BLANC surpasses external prior methods in specificity and coverage: unnormalized inter-cluster connectivity [4] shows spurious recovery under random removal, and a CPC co-classification substitute [13] exposes substantially fewer testable combinations. A proprietary case further illustrates recovery of an expert-identified candidate.

## 2. Related Work

The proposed system integrates techniques from three research streams. This section reviews each in detail; the gap relative to the closest prior art was established in Section 1.

## 2.1. Neural topic modeling and patent-specific embeddings

BERTopic [7] is a modular framework that generates document embeddings with pre-trained transformers, reduces dimensionality with UMAP [14], clusters with HDB-SCAN [15], and produces topic representations via classbased Term Frequency–Inverse Document Frequency (c-TF-IDF). It consistently outperforms Latent Dirichlet Allocation (LDA) [16] and Non-negative Matrix Factorization (NMF) [17] on coherence and diversity benchmarks. For patent text, domain-adapted embeddings improve clustering quality: Sentence-BERT [18] enables large-scale semantic clustering, while PatentBERT [19] and PatentSBERTa [20] further specialize embeddings to the patent domain. Recent surveys document the transition to patent analysis based on large language models (LLMs) and note that such applications remain underexplored [21, 22].

Table 1  
Comparison with closest prior work. ✓ = present; — = absent.
<table><tr><td>Study</td><td>Clustering</td><td>Co-occ. measure</td><td>Multi-view</td><td>Cond. detection</td></tr><tr><td>Kim et al. (2014)</td><td>Traditional</td><td>Connectivity†</td><td></td><td></td></tr><tr><td>Choi &amp; Jun (2014)</td><td>Bayesian</td><td></td><td></td><td></td></tr><tr><td>Jeon et al. (2023)</td><td>BERTopic</td><td></td><td></td><td></td></tr><tr><td>Yang &amp; Chen (2025)</td><td>BERTopic</td><td></td><td></td><td></td></tr><tr><td>This work</td><td>BERTopic</td><td>NPMI</td><td>√</td><td>√</td></tr></table>

<sup>†</sup>Unnormalized inter-cluster connectivity, not a normalized co-occurrence association measure.

## 2.2. PMI, NPMI, and co-occurrence measures in patent analysis

PMI [10] quantifies how much the co-occurrence of two events exceeds independence, and its normalized form NPMI [11] scales values to [−1, +1], with −1 indicating complete absence of co-occurrence, 0 independence, and +1 perfect co-occurrence. This bounded normalization is critical for comparing association strengths across cluster pairs of diferent frequencies. In technology mapping, Jafe’s [12] seminal use of patent classification vectors established the economic rationale for co-occurrence-based proximity. Breschi et al. [13] and Curran and Leker [23] formalized International Patent Classification (IPC) coclassification as a test of whether technology pairs co-occur more than by chance. This is the same logic that NPMI applies to cluster pairs.

## 2.3. Patent white space analysis and technology gap detection

White space analysis has evolved through three generations. Morphological approaches [1, 24] organize patent keywords into 2D maps (morphology matrices or principal component analysis projections) and flag unoccupied cells or sparse regions as unexplored combinations. Son et al. [2] and Yoon and Magee [3] advanced vacancy detection with Generative Topographic Mapping, shifting from detection to probabilistic prediction. Cluster-based and function-based methods followed. Kim et al. [4] measured inter-cluster connectivity on text and citation networks; this is the closest precursor to computing co-occurrence between cluster pairs, although it uses traditional clustering without normalized association measures. Yoon et al. [5] used SAO structures to identify unexploited technology–product pairings, and Choi and Jun [6] applied Bayesian clustering for the same goal. Lee and Lee [25] grounded white space analysis in recombinant search theory, which holds that innovation arises from novel combinations of existing components. Transformerbased work is more recent. Jeon et al. [8] applied BERTopic with PatentSBERTa to USPTO data for technology opportunity discovery; this is the most directly analogous prior work, although it uses single-view clustering without statistical co-occurrence measures. Yang and Chen [9] integrated BERTopic with logistic growth models for lifecycle analysis. Across these generations, however, the same documents have not been clustered along several independent semantic dimensions; a plausible reason is that obtaining perspectiveseparated patent texts at scale was, until recently, impractical. The structured fields exposed by modern patent corpora (e.g., abstract, claims, and summary) now make this feasible; where such fields are absent, generative-AI extraction from the full text serves the same purpose.

## 3. Proposed Method: BLANC

Fig. 1 shows the BLANC architecture. The pipeline processes a patent corpus through three phases: multi-view clustering, co-occurrence quantification, and conditional white space detection. Each patent contributes three dimensionspecific texts (application/use, novelty, and inventive step), taken from native structured fields (e.g., abstract, claims, and summary) or, where these are absent, extracted by a generative AI model; field-specific preprocessing of these texts and the document-term matrix (DTM) used for keyword filtering in Phase 3 are detailed in Appendix A.

## 3.1. Multi-view clustering via BERTopic

The key idea is to treat diferent text fields as independent semantic views, analogous to multi-view learning, where the same dataset is analyzed from multiple representations independently. We define � = 3 dimensions: (1) application/use text, capturing where the invention is used; (2) novelty text, capturing what is new; and (3) inventive step text, capturing what benefit it provides. These three views are not arbitrary, but mirror the canonical structure of a patent specification. Its statement of industrial applicability describes where the invention is used, its claims describe what is new, and its account of the technical efect or problem solved describes why it is useful. Each view therefore corresponds to an established documentary distinction rather than an ad-hoc split. Other decompositions (e.g., technical field, problem addressed) are possible; we adopt � = 3 for this correspondence and leave a systematic study of the number and choice of views to future work (§5.4).

For each dimension � ∈ {1, 2, 3}, the BERTopic pipeline is executed independently:

1. Embedding: Dimension-specific text is encoded into dense vectors using a pre-trained Sentence-BERT model [18].

2. Dimensionality reduction: UMAP [14] projects embeddings into a lower-dimensional space.

Algorithm 1 BLANC: Patent White Space Detection Pipeline   
Require: Patent corpus �, semantic dimensions $k \in \{ 1 , \ldots , K \}$ , association threshold $\theta _ { \mathrm { { m i n } } }$ (default 0.3)   
Ensure: Ranked white space candidates   
1: for each dimension � do   
2: Extract dimension-specific texts (raw fields or generative AI)   
3: Run BERTopic: Sentence-BERT → UMAP → HDBSCAN → Keywords   
4: Assign cluster labels $c _ { k } ( d )$ for all $d \in D$   
5: end for   
6: Construct 3D co-occurrence tensor $C [ i , j , k ]$ ⊳ Eq. (1)   
7: Marginalize to three 2D matrices; compute global NPMI ⊳ Eqs. (2)–(5)   
8: User inspects the global NPMI landscape and specifies a keyword � of interest ⊳ interactive   
9: Filter corpus: $D _ { q } \gets \{ d \ | \ \mathrm { D T M } [ d , q ] > 0 \}$   
10: Compute conditional tensor and $\mathrm { N P M I } _ { q }$ over $D _ { q }$   
11: for each cluster pair $( i , j )$ across dimension combinations do   
12: $\Delta \mathrm { N P M I } ( i , j ; q ) \gets \mathrm { N P M I } ( i , j ) - \mathrm { N P M I } _ { q } ( i , j )$ ⊳ Eq. (6)   
13: if $\Delta \mathrm { N P M I } > 0$ and $\mathrm { N P M I } ( i , j ) \ge \theta _ { \mathrm { m i n } }$ and pair present in $D _ { q }$ (co-occurrence $\geq 1 )$ then   
14: Add $( i , j )$ to candidate list with $\Delta \mathrm { N P M I } ( i , j ; q )$   
15: end if   
16: end for   
17: return the top-20 candidates per dimension combination, ranked by ΔNPMI   
Micro-level drill-down (optional, per candidate):   
18: Extract document subset matching candidate cluster pair $( i , j )$   
19: Optionally intersect with $D _ { q }$ for keyword-focused analysis   
20: Build phrase co-occurrence matrix; compute pairwise NPMI   
21: Cluster phrases via Ward linkage on $1 - \mathrm { N P M I }$ distance   
22: Rank phrases by $\mathrm { N P M I } ( w , q )$ if keyword filter is active

3. Clustering: HDBSCAN [15] identifies dense clusters; documents not fitting any cluster are classified as noise (cluster ID = −1).

4. Topic representation: Each cluster is assigned representative keywords via a composite scoring method (Appendix A).

Each document � is thus assigned a cluster triple $d \ \to$ $( c _ { 1 } , c _ { 2 } , c _ { 3 } )$ , where $c _ { k }$ is the cluster assignment for dimension �, or −1 if the document is noise in that dimension.

## 3.2. Co-occurrence quantification: 3D tensor and NPMI

With � = 3 dimensions having $N _ { 1 } , N _ { 2 } , N _ { 3 }$ clusters respectively, we construct a 3D co-occurrence tensor:

$$
C [ i , j , k ] = \left| D _ { i } ^ { ( 1 ) } \cap D _ { j } ^ { ( 2 ) } \cap D _ { k } ^ { ( 3 ) } \right|\tag{1}
$$

counting documents simultaneously assigned to cluster � in dimension 1, cluster � in dimension 2, and cluster � in dimension 3. Documents that HDBSCAN leaves unassigned (noise, label −1) in any dimension are excluded from the tensor. A pairwise count obtained by marginalization can therefore be smaller than the number of documents sharing that cluster pair in the two dimensions concerned, since the latter does not constrain the third dimension. For pairwise analysis, the tensor is marginalized over each dimension to produce three 2D matrices $\begin{array} { r } { { \bf ( e . g . , } C ^ { ( 1 \cdot 2 ) } [ i , j ] = \sum _ { k } C [ i , j , k ] ) } \end{array}$

For each 2D co-occurrence matrix � of size $M \times N .$ NPMI quantifies the association strength between cluster pairs. Let $\begin{array} { r } { T = \sum _ { i , j } C _ { i j } } \end{array}$ be the total co-occurrences. Then:

$$
P ( x _ { i } ) = \sum _ { j } C _ { i j } / T , \quad P ( y _ { j } ) = \sum _ { i } C _ { i j } / T\tag{2}
$$

$$
P ( x _ { i } , y _ { j } ) = C _ { i j } / T\tag{3}
$$

$$
\operatorname { P M I } ( x _ { i } , y _ { j } ) = \log _ { 2 } { \frac { P ( x _ { i } , y _ { j } ) } { P ( x _ { i } ) \cdot P ( y _ { j } ) } }\tag{4}
$$

$$
\mathsf { N P M I } ( x _ { i } , y _ { j } ) = \frac { \mathsf { P M I } ( x _ { i } , y _ { j } ) } { - \log _ { 2 } P ( x _ { i } , y _ { j } ) }\tag{5}
$$

where a small constant is added to each probability for numerical stability.

## 3.3. Conditional white space detection

White space detection flags cluster pairs whose NPMI is high in the full corpus but drops when the corpus is restricted to documents containing a user-specified keyword.

Given a user keyword � and the DTM, the filtered corpus is $D _ { q } ~ = ~ \{ d ~ | ~ \mathrm { D T M } [ d , q ] ~ > ~ 0 \}$ . The conditional cooccurrence tensor and conditional NPMI values $\mathrm { N P M I } _ { q } ( i , j )$ are computed over $D _ { q }$ using the same formulas as Eqs. (2)– (5). We quantify white space for a cluster pair (�, �) by the NPMI drop, denoted ΔNPMI:

$$
\Delta \mathrm { N P M I } ( i , j ; q ) = \mathrm { N P M I } ( i , j ) - \mathrm { N P M I } _ { q } ( i , j )\tag{6}
$$

![](images/f4fe4f278d30bfb304ad746b0d6656104d3721ef949058819a413cf6d2871cca.jpg)

![](images/e42662761ac8fc29fcbb0c208c839adcc9204482060b1039dfecbbc6ff004722.jpg)

![](images/2c7f8a4efdf7e8c586c4fb5cc6584ddad3e5a461a136e57b4c0976519e9fbe22.jpg)  
Figure 2: Cluster size distribution for the G06N corpus after field-specific preprocessing. Each dimension resolves into many clusters $\left( 6 2 / 4 7 / 5 1 \right)$ ; the application dimension is finely graded, while novelty and inventive step each retain one larger leading cluster but still spread across dozens of themes rather than collapsing into a single dominant cluster as they do without preprocessing, enabling meaningful cross-dimensional NPMI analysis. Note: Y-axis scales difer across panels to accommodate the diferent cluster size ranges.

A positive value indicates that the combination is underrepresented among patents containing keyword �; that is, it is established globally but unexplored locally.

The metric is retained only if: (a) the filtered corpus contains at least one document assigned to both clusters jointly, i.e. conditional co-occurrence ≥ 1; (b) $\Delta \mathrm { N P M I } > 0 ;$ and (c) the original NPMI(�, $j ) \ge \theta _ { \mathrm { m i n } }$ (default: 0.3).

Condition (c) is essential. For a pair whose global NPMI is already low, ΔNPMI can turn positive under any unrelated keyword, since the filtered corpus retains only loosely related documents, which mechanically weakens an already weak association—a trivial gap, not white space. Restricting to pairs above $\theta _ { \mathrm { m i n } }$ isolates only the surprising cases, i.e., pairs that are strongly coupled across the full corpus yet decoupled once the corpus is narrowed to the keyword. The top-20 pairs from each dimension combination are presented as candidates.

Algorithm 1 summarizes the full pipeline.

## 3.4. Micro-level phrase analysis (optional)

In addition to the macro-level cluster-pair analysis, BLANC ofers an optional drill-down for inspecting individual candidates. Given a candidate pair (�, �), the subset of documents satisfying both cluster constraints is extracted, and a phrase co-occurrence matrix is built from the most frequent phrases, on which pairwise NPMI is computed. Hierarchical agglomerative clustering on the 1 − NPMI distance, using Ward linkage [26] (the criterion that merges clusters to minimize the increase in within-cluster variance), reveals phrase theme groups; the heatmap is reordered by group assignment to visualize within-group coherence and between-group separation. When a keyword filter � is active, phrases are additionally ranked by NPMI(�, �) computed with the same formula as Eq. (5) but over phrase–keyword co-occurrence. This ranking shows which phrases within the underexplored combination are most associated with the keyword of interest.

Computational complexity. Post-embedding operations are inexpensive: tensor construction is $O ( N _ { 1 } N _ { 2 } N _ { 3 } | D | )$ NPMI is �(��) per 2D matrix, and keyword filtering is �(|�|). For typical corpora of thousands of documents and tens of clusters per dimension, all post-embedding steps complete in seconds; the BERTopic embedding step is the dominant cost and scales linearly with corpus size.

## 4. Experiments

We evaluate BLANC on two public USPTO corpora and one proprietary corpus, covering clustering (§4.2), global NPMI (§4.3), white space case studies (§4.4), industrial validation on proprietary data (§4.5), and baselines (§4.6.3); a theme-keyword sensitivity diagnostic is reported in Appendix B. All software, hyperparameters, preprocessing, and reproducibility settings are detailed in Appendix A.

## 4.1. Dataset

We use the Harvard USPTO Patent Dataset (HUPD) [27], selecting two technologically distinct Cooperative Patent Classification (CPC) subclasses to evaluate cross-domain generalizability:

The G06N corpus contains 5,417 applications in CPC subclass G06N (computing arrangements based on specific computational models), covering machine learning, neural networks, and genetic algorithms, filed 2013–2016. The C03C corpus contains 1,982 applications in CPC subclass C03C (chemical composition of glasses, glazes, or vitreous enamels), covering optical glass, glass fiber, coatings, and chemical strengthening, filed 2012–2017. For both corpora, abstracts, claims, and summaries are mapped to the application, novelty, and inventive step dimensions respectively (Table 2).

On the G06N corpus, without the field-specific preprocessing detailed in Appendix A, claims collapse to 2 clusters and summaries to 8 clusters (one dominant cluster holding 80% of documents); after preprocessing they produce 47 and 51 clusters (§4.6.1). DTM construction applies a patentdomain stop-word list, in addition to the standard English stop words, to remove non-discriminative terms. The C03C corpus yields a larger DTM vocabulary (80,244 vs. 65,606) despite fewer documents, reflecting rich chemical nomenclature such as $\mathrm { \mathrm { ^ { 6 4 } S i O _ { 2 } \mathrm { - A l _ { 2 } O _ { 3 } \mathrm { - L i _ { 2 } O ^ { 3 } } } } }$

![](images/f20f34ce85edf5b9c577c128e62b87b8886cc65fc5133fa32e6fd4cdb4a3a256.jpg)  
Figure 3: NPMI heatmap for application×novelty cluster pairs (G06N corpus, top-20 application clusters by co-occurrence; verticalaxis numbers are the original cluster IDs and are therefore non-consecutive, since only these top-20 clusters are shown). Dark-red cells mark cluster pairs whose application and novelty themes co-occur far more than chance (coherent technology combinations), whereas most cells are weak (blue), indicating that the majority of cross-dimensional combinations are only loosely associated.

Table 2  
Dataset statistics for the two HUPD corpora.
<table><tr><td></td><td>G06N (ML/AI)</td><td>C03C (Glass)</td></tr><tr><td>Documents</td><td>5,417</td><td>1,982</td></tr><tr><td>Filing years</td><td>2013-16</td><td>2012-17</td></tr><tr><td>Avg. words (abstract)</td><td>114</td><td>100</td></tr><tr><td>Avg. words (claims)</td><td>1,216</td><td>764</td></tr><tr><td>Avg. words (summary)</td><td>684</td><td>729</td></tr><tr><td>Empty rate (summary)</td><td>1.6%</td><td>14.9%</td></tr><tr><td>DTM vocabulary</td><td>65,606</td><td>80,244</td></tr></table>

## 4.2. Clustering results

For G06N, BERTopic identified 62, 47, and 51 clusters for the application, novelty, and inventive step dimensions (noise rates 36.8%, 33.2%, 28.3%); the comparatively high application-dimension rate reflects HDBSCAN’s intentional exclusion of documents too short or too generic to form coherent clusters. The application dimension spans diverse themes including quantum computing, question answering, social networking, autonomous driving, anomaly detection, spiking neural networks, sentiment analysis, and agricultural AI. Fig. 2 shows the resulting size distributions, which are finely graded in the application dimension, with a larger leading cluster in novelty and inventive step but no collapse into a single dominant cluster. For C03C, BERTopic produced 63, 63, and 48 clusters (noise rates 21.3%, 21.3%, 17.3%), confirming that the pipeline scales to a substantially diferent technology domain.

## 4.3. Co-occurrence and global NPMI

The application×novelty NPMI matrix (62 × 47) reveals the associative structure of the ML/AI landscape; Fig. 3 visualizes the top-20 application rows and Table 3 lists the top-10 pairs by NPMI.

The strongest association (NPMI of 0.978) is between quantum computing clusters in both dimensions, reflecting the tight coupling between quantum processor applications and quantum-specific technical claims. Other highly associated pairs include decision trees (NPMI of 0.958), finite automata (0.948), time series forecasting (0.939), and social networking (0.930). These high-NPMI pairs represent wellestablished technology combinations in the ML/AI domain.

## 4.4. White space detection: case studies

To demonstrate the white space detection capability, we selected the keyword “neural” as the user query, targeting neural network (NN) technologies within the broader ML/AI corpus. Filtering the corpus by this keyword reduced the document count from 5,417 to 1,014 (18.7% of the original).

![](images/e4f95b0e834efd140518baa811f0aa30889fc9dfa388756decc5024304accec8.jpg)  
Figure 4: Top-5 white space candidates for keyword “neural” (G06N, application×novelty; same candidates as Table 4). Blue bars show global NPMI; salmon bars show conditional NPMI after keyword filtering. The gap (ΔNPMI, annotated) represents the degree of underrepresentation among “neural”-keyword patents.

Table 3  
Top-10 app×novelty pairs by global NPMI. Co.: co-occurrence count.
<table><tr><td>Application theme</td><td>Novelty theme</td><td>NPMI</td><td> $\mathsf { C o . }$ </td></tr><tr><td>Quantum comp.</td><td>Quantum proc.</td><td>0.978</td><td>106</td></tr><tr><td>Decision tree</td><td>Decision tree</td><td>0.958</td><td>33</td></tr><tr><td>Finite automaton</td><td>DFA/NFA</td><td>0.948</td><td>30</td></tr><tr><td>Time series</td><td>Forecast model</td><td>0.939</td><td>44</td></tr><tr><td>Social network</td><td>Social graph</td><td>0.930</td><td>61</td></tr><tr><td>Energy cons.</td><td>Energy anomaly</td><td>0.926</td><td>60</td></tr><tr><td>User expertise</td><td>Expertise inf.</td><td>0.919</td><td>8</td></tr><tr><td>Multi-obj opt.</td><td>Pareto sol.</td><td>0.916</td><td>55</td></tr><tr><td>Question ans.</td><td>Cand. answers</td><td>0.906</td><td>83</td></tr><tr><td>Anomaly det.</td><td>Anomaly score</td><td>0.878</td><td>26</td></tr></table>

Fig. 4 and Table 4 present the top white space candidates ranked by ΔNPMI for the application×novelty dimension pair.

The top candidate pairs neuromorphic hardware (application cluster 19: “neuromorphic synapse”) with spiking NN claims (novelty cluster 1: “synaptic / output neuron”); its global NPMI of 0.417 falls to a conditional NPMI of 0.079, an NPMI drop of 0.338. This combination is well established overall but underrepresented among “neural”- keyword patents, since neuromorphic patents emphasize circuit-level rather than neural-network framing.

Four of five candidates share the spiking-NN novelty cluster, pairing it with distinct application clusters: neuromorphic hardware, spiking networks, NN augmentation, and neurosynaptic cores. The fifth (ensemble learning × classification) rests on a single co-occurring document $( n _ { q } =$ 1) and is reported only for completeness. This concentration identifies spiking/neuromorphic computing as a coherent area systematically underrepresented in “neural”- keyword patents. The gap is terminological, since neuromorphic patents tend to use “synaptic”, “spike”, or “membrane”. This shows that BLANC can surface non-obvious vocabulary-divergence gaps that support brainstorming.

C03C case study. The keyword “abbe” targets optical glass via the Abbe number, a measure of optical dispersion, and filters C03C to 50 documents (2.5%). BLANC returns two high-global/near-zero-conditional candidates: optical glass × optical glass molding (ΔNPMI of 0.770) and optical glass × La/Gd rare-earth composition (ΔNPMI of 0.654). Both indicate that “abbe” patents focus on optical-property specification rather than molding or rare-earth composition. This is a potentially actionable gap for optical glass manufacturers.

Table 4  
Top-5 white space candidates for keyword “neural” (app×novelty). NPMI: global association; NPMI : conditional association on the keyword-filtered subcorpus; $\Delta = \mathrm { N P M I } - \mathrm { N P M I } _ { q } ; n _ { q } \colon$ conditional co-occurrence count.
<table><tr><td>Application</td><td>Novelty</td><td>NPMI</td><td> $\mathrm { N P M I } _ { q }$ </td><td> $\Delta$ </td><td> $n _ { q }$ </td></tr><tr><td>Neuromorphic</td><td>Spiking NN</td><td>0.417</td><td>0.079</td><td>0.338</td><td>19</td></tr><tr><td>Ensemble learn.</td><td>Classif.</td><td>0.480</td><td>0.168</td><td>0.312</td><td>1</td></tr><tr><td>Spiking NN</td><td>Spiking NN</td><td>0.395</td><td>0.089</td><td>0.306</td><td>37</td></tr><tr><td>NN augment.</td><td>Spiking NN</td><td>0.412</td><td>0.115</td><td>0.298</td><td>68</td></tr><tr><td>Neurosynaptic</td><td>Spiking NN</td><td>0.375</td><td>0.090</td><td>0.285</td><td>28</td></tr></table>

## 4.5. Industrial validation on proprietary data

We further apply BLANC to a proprietary corpus of 302 patents in float glass / glass-ceramics, filed 2019–2023. Unlike HUPD, each document carries pre-structured fields (application, solution, problem) mapped directly to the three dimensions; embeddings use a multilingual Sentence-BERT model, with UMAP (n\_neighbors=10, n\_components=5) and HDBSCAN (min\_cluster\_size=8, min\_samples=3) on application and inventive step dimensions. The novelty dimension required a �-means (� = 10) fallback because HDB-SCAN produced only 2 clusters. BLANC yielded 15, 10, and 11 clusters (noise rates 12.6%, 0%, 15.9%) spanning themes such as coating, laminated glass, solar cells, float strengthening, strengthening/warpage suppression, lithium aluminosilicate (LAS) crystallization, and cooking appliances. $\theta _ { \mathrm { { m i n } } }$ was relaxed to 0.2 because the default 0.3 leaves too few candidates on this small corpus.

Prior to the analysis, IP experts had already identified “fluorine surface treatment” × “warpage suppression” as an underexplored filing opportunity. With the keyword “fluorine” (14 matches, 4.6%), BLANC returned the flat/floatstrengthened × strengthening/warpage suppression pair as the top candidate in all three dimension combinations: ΔNPMI of 0.464 for application×novelty and 0.480 for application×inventive step. BLANC thus identified the same combination the experts had flagged. Because that pair was known to exist beforehand, we present this as a confirmatory single case demonstrating consistency with expertjudgment, not a blinded test of independent discovery. The conditional estimate rests on only 14 documents (a caveat revisited in §5.4); the candidate is nonetheless consistent across all three dimension pairs. The micro-level phrase analysis (§3.4) further confirms the gap. Phrases around warpage control (chemical strengthening, ion exchange, stress profile) and around fluorine treatment (surface treatment, fluorination, concentration profile) form two weakly connected theme groups.

## 4.6. Quantitative evaluation

## 4.6.1. Sensitivity to preprocessing and hyperparameters

The 2→47 and 8→51 cluster jumps reported in Section 4.1 for claims and summaries respectively quantify the impact of field-specific preprocessing. Without boilerplate removal, the claims and summary dimensions collapse to a handful of clusters dominated by legal formulaic language and figure-reference boilerplate, and multi-view analysis becomes impossible. A systematic sensitivity analysis of hyperparameters $( \theta _ { \mathrm { m i n } } ,$ HDBSCAN min\_cluster\_size, the keyword-scoring weight � defined in Appendix A, the number of views �, and embedding model choice) is deferred to future work; the removal-driven recovery results in Section 4.6.2 provide indirect evidence of robustness, as a consistent targeted-versus-random separation was obtained with a single parameter configuration across two technologically distinct corpora.

## 4.6.2. White space injection by removal-driven recovery

Because no ground-truth white space labels exist for patent corpora, we evaluate by injection, artificially depleting a known combination and testing whether BLANC recovers it. To make recovery the quantity under test, rather than keyword targeting, we condition on a keyword that is independent of the target pair’s specific theme, so the pair is not flagged before injection and the only mechanism that can surface it is the removal itself. (The alternative of matching the keyword to the target pair is confounded by the keyword itself; we analyze it separately as a metricsensitivity diagnostic in Appendix B.)

The protocol proceeds in four steps.

1. Select established combinations. We consider each application×novelty pair with global $\mathrm { N P M I } \ge \theta _ { \mathrm { m i n } }$ and co-occurrence $\geq 1 2$

2. Assign an independent keyword. For each pair we select a keyword by a fixed, theme-blind rule that touches the pair’s documents only through a coverage constraint. Candidates are the master-DTM terms of at least three characters that occur in at least 10% of the corpus and in at least 60% of the pair’s documents.<sup>3</sup> The master DTM contains unigrams to trigrams over the two text fields, with English and patent-domain stop words removed, df $\ge ~ 5$ on G06N and $\geq ~ 3$ on C03C, and $\mathrm { ~ \textit ~ { ~ d ~ f ~ } ~ } \leq \mathrm { ~ \ } 8 0 \%$ (Appendix A). Among these candidates we take the term with the highest corpus-wide document frequency, breaking ties alphabetically. “The pair’s documents” here means every document assigned to both clusters in the application and novelty dimensions, leaving the third dimension unconstrained (§3.2). The coverage constraint is what guarantees that removing documents from pair $\cap D _ { q }$ actually depletes the pair inside the filtered corpus, while ranking by corpus-wide frequency makes the term as generic as the constraint allows. The rule is not injective, and in practice it returns a handful of common words unrelated to any pair’s specific theme (“computer” for 37 of the 44 G06N pairs; “surface” or “temperature” for 15 of the 22 C03C pairs; Appendix C).

3. Keep only initially undetected pairs. Because such a broad term spans many clusters, the target pair is only a small fraction of $D _ { q }$ and its marginals are not inflated, so the pair lies outside the top-20 before any removal. We retain only these initially undetected pairs: on G06N, 46 high-NPMI pairs yield 45 with a qualifying keyword and 44 initially undetected; on C03C, 23 yield 23 and 22. Across the retained pairs the target accounts for a median 0.56% of $D _ { q }$ (maximum 7.0%) on G06N and 1.5% (maximum 3.3%) on C03C, confirming that the keyword does not preferentially select the pair.

4. Inject and measure recovery. We remove a fraction � of the documents in pair ∩ $D _ { q }$ and record whether the pair newly enters the top-20. Each run is matched by three controls that remove the same number of documents under the same fixed-seed protocol, difering only in which documents are removed: (i) corpusrandom, drawn uniformly from the entire corpus, and (ii) $D _ { q }$ -random, drawn uniformly from $D _ { q }$ excluding the pair. The third control, (iii) decoy, removes documents of a diferent established pair inside the same $D _ { q } .$ —the one whose |pair ∩ $D _ { q } |$ is closest to the target’s among those still undetected before removal. For the decoy control we record both whether the target is spuriously detected and whether the depleted decoy itself is. Controls (i) and (ii) are invariance tests. They perturb the corpus by the same amount but touch pair ∩ $D _ { q }$ only incidentally (an expected 0.06– 0.19 documents at $\delta \ : = \ : 0 . 7 5 )$ , so they test whether ΔNPMI reacts to removal in general. Control (iii) is the specificity test, applying a structurally equivalent targeted depletion to another pair.

Removal-driven injection: percentage of established, initially undetected pairs newly entering the top-20 after targeted remova versus three size-matched controls. “Decoy” depletes a diferent established pair in the same $D _ { q } ;$ its two columns report whether the target is spuriously detected (specificity) and whether the depleted decoy is (sensitivity). Decoy denominators are smaller where no size-matched decoy exists. Because the decoy arm removes as many documents as the target arm, a decoy larger than the target is only partly depleted and can still be recovered, which is why the decoy column stays non-zero at $\delta = 1 . 0 0$ while the target column is 0%. Depletion is complete for 22 of 42 G06N decoys and 6 of 20 on C03C at that setting, and none of those fully depleted decoys is recovered, reproducing the total-absence regime on the decoy side.
<table><tr><td colspan="3"></td><td colspan="2">Random</td><td colspan="2">Decoy</td></tr><tr><td>Corpus</td><td>δ</td><td>Targeted</td><td>corpus</td><td> $D _ { q }$ </td><td>target</td><td>decoy</td></tr><tr><td rowspan="3">G06N (n = 44)</td><td>0.50</td><td>13.6%</td><td>0%</td><td>0%</td><td>0%</td><td>25.6%</td></tr><tr><td>0.75</td><td>34.1%</td><td>0%</td><td>0%</td><td>0%</td><td>40.5%</td></tr><tr><td>1.00</td><td>0%</td><td>0%</td><td>2.3%</td><td>0%</td><td>35.7%</td></tr><tr><td rowspan="3">C03C (n = 22)</td><td>0.50</td><td>9.1%</td><td>0%</td><td>0%</td><td>0%</td><td>36.4%</td></tr><tr><td>0.75</td><td>27.3%</td><td>0%</td><td>0%</td><td>0%</td><td>54.5%</td></tr><tr><td>1.00</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>30.0%</td></tr></table>

Table 5 reports the outcome. At $\delta \ = \ 0 . 7 5 ,$ , targeted removal newly surfaces 34.1% (G06N) and 27.3% (C03C) of pairs, while all three controls stay at or near zero. Corpusrandom and $D _ { q }$ -random are at 0% in every cell but one (a single G06N case, 2.3%), and depleting a size-matched decoy pair never brings the target into the top-20 (0 of 191 decoy trials across both corpora and all three values of �). The decoy control is the informative one, because it applies the same targeted depletion to a structurally equivalent pair. In the very same runs the decoy enters the top-20 in 25.6– 40.5% (G06N) and 30.0–54.5% (C03C) of trials. The detector therefore responds to depletion of whichever pair was depleted, and not to depletion in general. The separation is thus a specificity result, not an artifact of a control that could never succeed. The recovery rates themselves (34.1% and 27.3%) are estimated less precisely than the targetedversus-control separation. A bootstrap over targets (10,000 resamples) gives 95% intervals of [20.5%, 47.7%] (G06N) and [9.1%, 45.5%] (C03C), but targets sharing a keyword share a filtered corpus and a top-20 competition, so those intervals understate the variance. Resampling keyword groups instead gives [0.0%, 41.5%] on G06N (� = 5 groups, one holding 37 of 44 targets, so a resample can omit it entirely) and [8.3%, 42.9%] on C03C (� = 6). We therefore read the recovery rates as indicative rather than precisely estimated, and rest the claim on the controls being exactly zero. The efect is non-monotone in �. At � = 1.0 the pair’s conditional co-occurrence reaches zero and it is excluded by filter (a) of §3.3 (the total-absence regime), so recovery peaks at partial removal, consistent with BLANC’s design target of underrepresentation rather than absence. Because targets, keyword, and ground truth are all fixed before detection, this measure is free of evaluation circularity.

## 4.6.3. Baseline comparison

We compare BLANC under the identical removal-driven protocol $( \delta = 0 . 7 5 )$ against both internal component ablations and external prior methods, replacing one element at a time so each comparison is on equal footing (Table 6).

Multi-view decomposition is essential. Replacing the three dimension-specific clusterings with a single BERTopic run on concatenated text (abstract + claims + summary) reproduces the single-view opportunity-discovery setting of Jeon et al. [8]. This yields 0% recovery on both corpora, because a single view captures each theme within one cluster but cannot express cross-dimensional combinations, leaving no pairs to deplete.

NPMI provides specificity; unnormalized connectivity does not. Replacing NPMI with Jaccard (|� ∩ �|∕|� ∪ �|), cosine similarity, or raw inter-cluster connectivity (the unnormalized co-occurrence count used by the closest precursor, Kim et al. [4]), while keeping the multi-view clusters, does not cleanly isolate the injected gap. Although cosine attains a higher absolute recovery rate (52.3% / 45.5%), every unnormalized or set-normalized measure also “recovers” the pair under random removal, and the spurious-recovery rate grows as marginal normalization is removed: Jaccard 15.9% / 27.3%, cosine 22.7% / 13.6%, and inter-cluster connectivity 29.5% / 54.5%. For the glass corpus, connectivity recovers the target more often under random than under targeted removal. NPMI is the only measure whose randomremoval control is 0% on both corpora, because its marginalfrequency normalization makes the conditional score react to depletion of the specific combination rather than to any change in cluster sizes. We therefore characterize NPMI’s contribution as specificity (zero spurious recovery) rather than raw sensitivity.

Learned clustering exposes far more combinations than a standard taxonomy. As an external substitute for the learned application axis, we replace it with third-party CPC subgroup codes (main\_cpc\_label), in the spirit of IPC/CPC co-classification gap analysis [13, 23], keeping the learned novelty clustering and the conditional detector fixed. The diference is one of coverage rather than per-pair recovery. Substituting CPC codes collapses the number of established cross-dimensional combinations available to test roughly fourfold (high-NPMI pairs fall from 46 to 10 on G06N and from 23 to 11 on C03C), because the of-the-shelf taxonomy aligns poorly with the corpus’s actual technology couplings. On the few combinations CPC does expose, the conditional NPMI detector retains its specificity (targeted recovery 50.0% and 20.0% against a 0% random control), confirming that the specificity property is intrinsic to NPMI and independent of the clustering source, but the candidate pool is too small to support detection on a larger scale.

Baseline comparison on the removal-driven protocol $\begin{array} { r } { ( \delta = 0 . 7 5 ) \mathrm { : } } \end{array}$ : percentage of initially undetected target pairs newly entering the top-20 under targeted versus size-matched random removal, the latter drawn from the whole corpus (the corpus-random control of §4.6.2). Only NPMI exhibits zero spurious recovery under random removal; a useful detector needs Targeted ≫ Random.
<table><tr><td rowspan="2">Method</td><td colspan="2">G06N  $( \mathsf { M L } / \mathsf { A l } )$ </td><td colspan="2">C03C (Glass)</td></tr><tr><td>Targeted</td><td>Random</td><td>Targeted</td><td>Random</td></tr><tr><td>Multi-view + NPMI (BLANC)</td><td>34.1%</td><td>0.0%</td><td>27.3%</td><td>0.0%</td></tr><tr><td> $\mathsf { M u l t i - v i e w } + \mathsf { J a c c a r d }$ </td><td>29.5%</td><td>15.9%</td><td>31.8%</td><td>27.3%</td></tr><tr><td> $\mathsf { M u l t i - v i e w } + \mathsf { C o s i n e }$ </td><td>52.3%</td><td>22.7%</td><td>45.5%</td><td>13.6%</td></tr><tr><td>Multi-view + connectivity [4]</td><td>29.5%</td><td>29.5%</td><td>22.7%</td><td>54.5%</td></tr><tr><td> $\mathsf { S i n g l e - v i e w + N P M I \left[ 8 \right] }$ </td><td>0.0%</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr></table>

All quantitative experiments use publicly available data (HUPD [27]), enabling full reproducibility; the industrial validation uses proprietary data described in Section 4.5.

## 5. Discussion

## 5.1. Practical implications for patent brainstorming

BLANC turns brainstorming preparation from unstructured reading into a focused, data-driven process. The threestep workflow (review clusters → set keyword → explore white space) runs in minutes once the corpus is prepared, compared with the weeks of manual efort required to build traditional patent maps. The multi-view decomposition also lets practitioners articulate gaps as specific application– novelty or application–inventive step combinations rather than blank regions on a 2D map.

Keyword selection is a critical user skill. Domainspecific terms (“antimicrobial”, “abbe”) substantially outperform generic ones (“glass”, “layer”), so we recommend starting from Phase-1 cluster keywords, preferring multiword technical nomenclature, and consulting CPC subclass definitions. This dependence on expertise does not reintroduce challenge (2) from §1. That challenge was that the quality of unaided brainstorming varies with each participant’s domain knowledge, leaving the search unstructured and hard to reproduce. BLANC instead requires expertise only to choose a keyword. Once the keyword is chosen,

BLANC structures and accelerates the exploration, makes each query reproducible, and lets practitioners cheaply try several keywords and compare the resulting candidates, turning open-ended reading into a guided, repeatable process rather than removing the need for expertise altogether.

BLANC is a support tool rather than a replacement for experts. Candidates are starting points for discussion, and domain expertise is required to interpret why a given combination is underexplored. A gap may persist for several reasons: limited technical feasibility, weak market demand, deliberate competitive avoidance, or merely terminological divergence, as in the neuromorphic×spiking gap of §4.4, where “neural” patents simply tend to use diferent vocabulary. Only the last is a detection artifact, and disentangling a genuine opportunity from these alternatives is precisely where the practitioner’s judgment is indispensable. The proprietary float glass case (§4.5) exemplifies this support role, in that the surfaced candidate matched one that experts had independently identified, focusing theirjudgment rather than replacing it.

## 5.2. Theoretical positioning and generalizability

The multi-view approach operationalizes recombinant search theory [25] in the patent domain. Its three dimensions of application/use, novelty, and inventive step are complementary facets of technological knowledge, and crosstabulating them via NPMI reveals recombinant opportunities invisible to single-view analysis. The NPMI drop metric extends co-word analysis (a bibliometric technique that maps a field through term co-occurrence) from absolute to conditional co-occurrence, adding a user-specific lens. ΔNPMI measures the “surprise” of a combination’s absence in a domain-specific context, paralleling the informationtheoretic interpretation of PMI [10].

While demonstrated on patents, BLANC is, in principle, applicable to any document collection with multiple classifiable fields. The cross-domain evaluation supports this. Despite fundamental technological diferences between ML/AI and glass compositions, BLANC shows the same qualitative behavior in the removal-driven test (comparable targeted recovery, 34.1% vs. 27.3% at $\delta = 0 . 7 5$ , against a 0% random-removal control in both) with no tuning beyond an HDBSCAN size adjustment, and the same preprocessing patterns proved efective in both domains.

## 5.3. Integration with LLMs

A natural extension is to pass white space candidates to a large language model for explanation generation, articulating why a combination is underexplored and what approaches might fill it, thus connecting detection to idea generation. Recent work on patent-specific LLMs [28, 29] suggests this is feasible, and preliminary experiments with commercial LLMs yield plausible explanations from clusterkeyword and co-occurrence context, though systematic evaluation remains future work.

## 5.4. Limitations

Several limitations deserve mention. Statistical stability: when the filtered corpus is small, NPMI estimates become unstable, and minimum-occurrence thresholds only partially mitigate this; we report bootstrap confidence intervals on the removal-driven recovery rate (§4.6.2), but these intervals are wide given the modest number of eligible targets, and the experiments confirm that pairs with very small co occurrence counts $( \le ~ 1 2 )$ are more likely to be missed. Preprocessing and dimension choice: multi-view clustering depends on structured text fields, and patent claims in particular require domain-aware boilerplate removal (§4.6.1); the three-dimension decomposition (application, novelty, inventive step) is also not the only possible choice, and a formal sensitivity analysis to the number of views � and to alternative dimensions (e.g., technical field, prob lem addressed) remains future work. Objectivity of cluster estimation: we prefer automatic HDBSCAN cluster esti mation precisely because it avoids analyst-chosen cluster counts; on the small proprietary corpus, however, the nov elty dimension fell back to a manually specified �-means (� = 10) because HDBSCAN underclustered that field, which lowers the objectivity of that one case and argues for automatic estimation as the default. Keyword and temporal dynamics: the system does not automatically suggest optimal keywords (Appendix B) and does not account for temporal dynamics; periodic re-analysis is recommended. Nature of the evaluation: because white space is defined by absence, no objective ground truth exists, so synthetic injection and the single expert-confirmed industrial case serve only as proxies for true discovery performance. Synthetic injection in particular detects the reduction of known cluster pairs rather than the emergence of genuinely novel combinations; truly absent combinations can instead be identified by di rect inspection of the co-occurrence tensor. Independence across targets: the keyword rule of §4.6.2 is deterministic but not injective, and 37 of the 44 G06N targets share the keyword “computer”. The targets are therefore conditioned on a handful of terms rather than being independent queries, which is why we report keyword-group as well as target level bootstrap intervals there and treat the recovery rates as indicative. Regime covered by the evaluation: the removal driven protocol selects deliberately broad keywords, so its filtered corpora are large (a median 76% of the corpus on G06N and 55% on C03C). Intended use, by contrast, is narrow: “neural” retains 18.7% of G06N and “fluorine” 4.6% of the proprietary corpus. The primary evaluation thus does not probe the small- $D _ { q }$ regime where NPMI estimates are least stable (§5.4, statistical stability). Running the same protocol with narrow keywords does not solve this problem, because the pair would often be detected before removal, reinstating the circularity the protocol was designed to avoid. That regime is covered only by the diagnostic of Appendix B and the industrial case of §4.5. Scope of industrial validation: the proprietary-data validation (§4.5) rests on a single expertconfirmed candidate; broader validation across additional fields and expert panels would strengthen the evidence.

## 6. Conclusion

We have presented BLANC, a three-phase pipeline for patent white space detection that integrates multi-view BERTopic clustering, NPMI-based co-occurrence analysis, and conditional gap detection via a new NPMI drop (ΔNPMI) metric. BLANC identifies technology combinations that are well-established in the general patent landscape but underexplored for a user-specified keyword, presenting them as actionable candidates for new patent applications.

A removal-driven injection evaluation on two public USPTO corpora, comprising 5,417 ML/AI patents in CPC G06N and 1,982 glass composition patents in CPC C03C, tests whether BLANC recovers artificially depleted technology combinations under a protocol conditioned on a keyword independent of the target. When three-quarters of a target pair’s documents are removed, BLANC recovers 34.1% (G06N) and 27.3% (C03C) of them, while sizematched control removals essentially never do. Depleting a diferent established combination never brings the target into the top-20 (0 of 191 trials), while random removals leave it undetected in 11 of 12 cells. Comparisons on the same protocol against component ablations and external prior methods establish three things: multi-view decomposition is indispensable (single-view 0%); among all association measures tested, including the closest precursor (unnormalized inter-cluster connectivity), NPMI is the only one with zero spurious recovery under random removal; and the learned application clustering exposes several times as many established cross-dimensional combinations as a standard CPC taxonomy. The consistency of this behavior across two technologically distinct domains supports the generalizability of the framework. A proprietary-data case on 302 float glass / glass-ceramics patents further illustrates that, with the keyword “fluorine,” BLANC reveals a fluorine surface treatment × warpage suppression candidate (ΔNPMI up to 0.48) that IP experts had independently identified by hand—a single confirmatory case.

A key practical finding is that field-specific text preprocessing is essential for efective multi-view clustering; without it, dimensions collapse into uninformative clusters. Future work includes temporal analysis of white space evolution, multi-keyword conditioning, statistical confidence scoring via bootstrap, and LLM-based explanation generation for white space candidates.

## Data availability

Quantitative experiments use the publicly available Harvard USPTO Patent Dataset (HUPD) [27], accessible at https://huggingface.co/datasets/HUPD/hupd. The proprietary float glass patent data used for industrial validation were retrieved from a commercial patent database under institutional license; neither the records nor the underlying search query can be redistributed. The corpus composition (size, field structure, and filing window) is described in Section 4.5.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper. Part of the methodology described in this paper is the subject of a Japanese patent application (Publication No. JP2025148721A; Miyazawa, S., Fujii, K., Fujii, Y., Tomiyori, Y., “Information processing device, information processing method, and program,” filed 2024-03-26, published 2025-10-08; Assignee: AGC Inc.).

## CRediT authorship contribution statement

Shuichi Miyazawa: Conceptualization, Methodology, Software, Validation, Visualization, Writing – original draft. Kensuke Fujii: Conceptualization, Validation, Writing – review & editing.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the authors used Claude Code (Anthropic) to assist with language editing, restructuring of paragraphs, and consistency checks across sections of the manuscript. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

## A. Public data experiment parameters

Both corpora use the Harvard USPTO Patent Dataset (HUPD) [27].

All experiments were implemented in Python using BERTopic, Sentence-Transformers, UMAP, and HDBSCAN. Because the clustering relies on these libraries’ numerical routines, the exact NPMI values are sensitive to their versions, whereas the rank-based recovery results of §4.6.2 are robust. All stochastic components (UMAP, �-means, and the injection-experiment random number generator) use fixed random seeds (HDBSCAN is deterministic given its inputs), so the reported numbers are deterministic for a given corpus and configuration.

Numerical stability. The small constant added to each probability is $\varepsilon \ = \ 1 0 ^ { - 8 }$ for the cluster-pair NPMI of Eqs. (2)–(5), and $\varepsilon = 1 0 ^ { - 1 2 }$ for the term–cluster NPMI in the keyword score of Eq. (7).

## G06N (ML/AI) corpus

5,417 patent applications with CPC subclass G06N, filed 2013–2016.

Text field mapping:

• Application/use dimension: abstract

• Novelty dimension: claims (with legal boilerplate removed)

• Inventive step dimension: summary (with figure descriptions and boilerplate removed)

Claims preprocessing removes, via regular expressions, claim numbering (“1. A method. . . ”), back-references (“The method of claim 1”), and common formulaic patterns (“comprising”, “wherein”, “configured to”, “non-transitory computer-readable medium”, etc.). Efect: novelty clustering improved from 2 clusters to 47 clusters.

Summary preprocessing removes section headers (the <SOH>. . . <EOH> markup that delimits headings in HUPD’s summary field), entire FIG. sentences, and common formulaic patterns (“in one embodiment”, “the present disclosure”, etc.). Efect: inventive step clustering improved from 8 clusters (80% in one dominant cluster) to 51 clusters.

Cluster keyword scoring. Rather than BERTopic’s built-in c-TF-IDF, which favors high-frequency terms in specialized patent text, each cluster’s representative keywords are ranked by a composite score that balances within-cluster frequency and corpus-level distinctiveness, weighted by a cluster-level inverse document frequency (cluster-IDF):

$$
\operatorname { s c o r e } ( w , i ) = \left[ \alpha \cdot \mathrm { N P M I } ( w , i ) + ( 1 - \alpha ) \cdot P ( w \mid i ) \right] \cdot \left( 1 + \operatorname { c I D F } ( w ) \right)\tag{7}
$$

where $P ( w \mid i )$ is the fraction of documents in cluster � containing term �, NPMI(�, �) is the normalized PMI between term and cluster, $\alpha ~ \in ~ [ 0 , 1 ]$ is a mixing weight (default 0.5), and $\mathrm { c I D F } ( w ) = \log ( N _ { \mathrm { c } } / | \{ i : w \in \mathrm { v o c a b } ( i ) \} | )$ is the cluster-IDF with $N _ { \mathrm { c } }$ the total number of clusters in the dimension (distinct from the number of views �). A term appearing in every cluster receives cIDF = 0; a term unique to one cluster receives cIDF = log $N _ { \mathrm { c } }$ , substantially boosting its rank, which suppresses legal boilerplate and structural terms such as “comprising” and “plurality” that otherwise dominate keyword lists.

Greedy diversified keyword selection. Candidates ranked by Eq. (7) are chosen sequentially with three rules: (1) if a candidate’s stemmed tokens are a subset of an alreadyselected longer term, skip; (2) if a candidate is a longer phrase that subsumes an already-selected shorter term, it replaces the shorter term (e.g., “acid compound” replaces “acid”); (3) otherwise skip if more than half of the candidate’s stemmed tokens already appear among selected keywords. All overlap checks use Snowball (Porter2) [30], a word-stemming algorithm, so that inflectional variants (“generate”/“generating”) are recognized as redundant.

BERTopic parameters:

• Embedding: all-MiniLM-L6-v2 (384-dim)

• UMAP: n\_neighbors=15, n\_components=5, min\_dist=0

• HDBSCAN: min\_cluster\_size=20, min\_samples=5

• Keywords: � = 0.5, cluster-IDF, Snowball, top-10

• White space detection: $\theta _ { \mathrm { m i n } } = 0 . 3 , \mathrm { t o p } { - 2 0 }$

• Stop words: standard English list augmented with patent-domain terms

Clustering results (after preprocessing): Application dimension: 62 clusters (noise rate 36.8%); Novelty dimension: 47 clusters (noise rate 33.2%); Inventive step dimension: 51 clusters (noise rate 28.3%). The application×novelty combination yields $6 2 \times 4 7 = 2 { , } 9 1 4$ cluster pairs, of which 58 have $\mathrm { N P M I } \ge \ 0 . 3$ and co-occurrence $\geq ~ 8$ (46 at the stricter $\geq 1 2$ co-occurrence floor used by the removal-driven evaluation, §4.6.2).

## C03C glass corpus

1,982 patent applications with CPC subclass C03C (chemical composition of glasses, glazes, or vitreous enamels), filed 2012–2017. The same text field mapping and claims preprocessing are applied; the summary (inventive step) field is left unprocessed, as the glass-domain vocabulary clusters well without boilerplate removal. BERTopic parameters: identical to G06N except HDBSCAN min\_cluster min\_samples=3 (adjusted for the smaller corpus). Clustering results: 63, 63, and 48 clusters (noise rates 21.3%, 21.3%, 17.3%). The application×novelty combination yields 59 pairs with $\mathrm { N P M I } \ge ~ 0 . 3$ and co-occurrence ≥ 5 (23 at the $\ge ~ 1 2$ floor used by the removal-driven evaluation). The application×inventive step combination yields 54 pairs with $\mathrm { N P M I } \ge 0 . 3$ and co-occurrence ≥ 5.

## B. Theme-keyword sensitivity diagnostic

Our initial protocol paired each target with a keyword semantically related to it: given a cluster pair $( c _ { a } , c _ { b } )$ with high global NPMI, remove a fraction � of its documents, recompute NPMI on the reduced corpus, and run white space detection with a related keyword �; a successful detection places the target in the top-20. We repeat this for 16 diverse themes per corpus on both HUPD corpora under identical protocols, with Hit Rate the fraction of experiments with the target in the top-20, MRR the mean of 1∕rank, and Precision@� the fraction within rank �. We retain this protocol as a diagnostic of ΔNPMI’s sensitivity, not as a performance claim; the controls at the end of this section show why a pair-correlated keyword cannot, on its own, evidence gap recovery, which is why the removal-driven evaluation (§4.6.2) is our primary quantitative result.

Theme selection procedure. To avoid arbitrary selection, we rank all cluster pairs in the application×novelty dimension by global NPMI (with a minimum co-occurrence filter of 8 for G06N / 5 for C03C) and go through the ranked list from the top, selecting pairs that admit an unambiguous domain-specific keyword and are not semantically redundant with an already-selected pair; selection stops at 16 pairs. On G06N this covers ranks 1–20 (4 skipped); on C03C, ranks 1–25 (9 skipped). Skipped pairs typically involve generic descriptors (e.g., “cognitive”, “coating”) that cannot be targeted by a single keyword, or that duplicate the semantic content of a higher-ranked pair.

Results. Tables 7 and 8 give per-theme outcomes. On G06N $( \delta = 0 . 5 0 )$ , BLANC achieves Hit Rate 93.8% with 5 of 16 targets at rank 1, MRR 0.469, P@5 68.8%; the one miss (ontology) has a keyword that matches documents across many cluster pairs, diluting the signal. On C03C with domain-specific keywords (“antimicrobial”, “abbe”, “optical fiber”), BLANC achieves Hit Rate 81.2% and MRR 0.594; the three misses involve cluster pairs with very small co-occurrence counts (≤ 12).

Third dimension validation. Repeating the injection on the application×inventive step pair confirms the diagnostic transfers to the third dimension: Hit Rate 87.5% (MRR 0.457) on C03C. The same holds on G06N once the summary field receives field-specific preprocessing, which raises the inventive-step dimension from 8 collapsed clusters to 51 and restores detection.

Impact of keyword specificity. Rerunning the 16 glass themes with an automatically generated keyword (the first word of the corresponding application cluster’s top keyword) drops Hit Rate to 56.2% (MRR 0.287) from 81.2% size=10<sup>,</sup>(0.594) with domain-specific keywords, because generic terms (“glass”, “layer”) match many cluster pairs and dilute the conditional signal. BLANC therefore performs best with specific technical terms; the generalizability claim itself rests on the removal-driven evaluation (§4.6.2), which shows the same targeted-versus-random behavior in both corpora.

Removal fraction sweep. We sweep $\delta \in \{ 0 . 1 0 , 0 . 2 5 \}$ 0.50, 0.75, 0.90, 1.00} on the quantum-computing pair (application cluster 2 × novelty cluster 4, global NPMI of 0.978, 106 documents) in G06N (Table 9). BLANC detects the target at rank 1 for every $\delta \ < \ 1 . 0 .$ , with ΔNPMI rising monotonically from 0.726 to 0.786. $\mathrm { A t } \delta = 1 . 0$ (complete removal), the target is excluded by the filtering criterion (§3.3) because conditional co-occurrence drops to zero. BLANC is designed for underrepresentation, not total absence.

Controls: what document removal contributes. The protocol above pairs each target with a keyword semantically related to it, so we add two controls to test whether detection is driven by the injection or by the keyword. Under a noremoval baseline (the keyword applied to the unperturbed corpus), the hand-selected target already appears in the top-20 for 100% (16/16, G06N) and 87.5% (14/16, C03C) of

Table 7  
Synthetic injection results on G06N (� = 0.50, app×novelty).
<table><tr><td>Theme</td><td>KW</td><td>Rem</td><td>Rank</td><td>Δ</td></tr><tr><td>Quantum comp.</td><td>quantum</td><td>54</td><td>1</td><td>0.733</td></tr><tr><td>Autonomous drv.</td><td>vehicle</td><td>21</td><td>1</td><td>0.463</td></tr><tr><td>Anomaly det.</td><td>anomaly</td><td>19</td><td>1</td><td>0.404</td></tr><tr><td>QA system</td><td>question</td><td>48</td><td>1</td><td>0.247</td></tr><tr><td>Finite automaton</td><td>automaton</td><td>16</td><td>1</td><td>0.103</td></tr><tr><td>Sentiment</td><td>sentiment</td><td>10</td><td>2</td><td>0.359</td></tr><tr><td>Rule engine</td><td>rule</td><td>28</td><td>3</td><td>0.188</td></tr><tr><td>Recommendation</td><td>recommend</td><td>26</td><td>3</td><td>0.135</td></tr><tr><td>Multi-obj opt.</td><td>optimization</td><td>29</td><td>3</td><td>0.097</td></tr><tr><td>Time series</td><td>forecast</td><td>25</td><td>4</td><td>0.061</td></tr><tr><td>Energy cons.</td><td>energy</td><td>32</td><td>4</td><td>0.023</td></tr><tr><td>Graph NN</td><td>graph</td><td>15</td><td>6</td><td>0.187</td></tr><tr><td>Social network</td><td>social</td><td>33</td><td>8</td><td>0.129</td></tr><tr><td>Decision tree</td><td>dec. tree</td><td>20</td><td>8</td><td>0.051</td></tr><tr><td>Predictive model</td><td>predictive</td><td>11</td><td>11</td><td>0.124</td></tr><tr><td>Ontology</td><td>ontology</td><td>20</td><td>MISS</td><td></td></tr><tr><td colspan="2">Hit Rate: 15/16 (93.8%)</td><td></td><td colspan="2">MRR: 0.469</td></tr></table>

P@1: 31.2% / P@5: 68.8% / P@10: 87.5%

## Table 9

� sweep for the quantum-computing pair (app×nov). Removal counts are taken over all 109 documents assigned to the pair in the application and novelty dimensions. The cooccurrence count of 106 in Table 3 is smaller because three of these documents are unassigned (noise) in the inventive step dimension and are thus excluded from the tensor (§3.2).
<table><tr><td>δ</td><td>Removed</td><td>Rank</td><td>ΔNPMI</td></tr><tr><td>0.10</td><td>10</td><td>1</td><td>0.726</td></tr><tr><td>0.25</td><td>27</td><td>1</td><td>0.728</td></tr><tr><td>0.50</td><td>54</td><td>1</td><td>0.733</td></tr><tr><td>0.75</td><td>81</td><td>1</td><td>0.747</td></tr><tr><td>0.90</td><td>98</td><td>1</td><td>0.786</td></tr><tr><td>1.00</td><td>109</td><td>MISS</td><td></td></tr></table>

themes, and a size-matched random-removal control (removing the same number of randomly chosen documents) reproduces these rates exactly. Removing the target pair’s documents lowers the hit rate only slightly (G06N 93.8%, C03C 81.2%, the latter agreeing with Table 8) and shifts only a few targets between adjacent ranks. The reason is structural. Conditioning on a keyword that correlates with a pair’s clusters inflates those clusters’ marginal probabilities within the small filtered corpus $D _ { q } ,$ which deflates the conditional NPMI through the normalization in Eq. (5) even when the conditional co-occurrence count is undiminished. The hit rates of Tables 7–8 therefore measure BLANC’s intended production behavior (surfacing the cluster pair associated with a keyword) rather than recovery of an artificially injected gap.

Table 8  
C03C injection (� = 0.50, app×nov, domain KWs).
<table><tr><td>Theme</td><td>KW</td><td>Rem</td><td>Rank</td><td>Δ</td></tr><tr><td>Antimicrobial</td><td>antimicrobial</td><td>9</td><td>1</td><td>0.592</td></tr><tr><td>Solar glazing</td><td>glazing</td><td>12</td><td>1</td><td>0.252</td></tr><tr><td>FTIR optical</td><td>ftir</td><td>7</td><td>1</td><td>0.803</td></tr><tr><td>Solar transm.</td><td>dom. wavel.</td><td>15</td><td>1</td><td>0.620</td></tr><tr><td>Optical fiber</td><td>opt. fiber</td><td>14</td><td>1</td><td>0.800</td></tr><tr><td>Sizing comp.</td><td>sizing</td><td>6</td><td>1</td><td>0.515</td></tr><tr><td>Li silicate</td><td>Li silicate</td><td>19</td><td>2</td><td>0.102</td></tr><tr><td>Silica-titania</td><td>titania</td><td>4</td><td>2</td><td>0.174</td></tr><tr><td>Glass fiber</td><td>fiber comp.</td><td>20</td><td>2</td><td>0.212</td></tr><tr><td>Optical (Abbe)</td><td>abbe</td><td>22</td><td>2</td><td>0.709</td></tr><tr><td>Tempered glass</td><td>tempered</td><td>9</td><td>2</td><td>0.162</td></tr><tr><td>Etching</td><td>etching</td><td>13</td><td>2</td><td>0.262</td></tr><tr><td>Induction seal</td><td>sealing</td><td>7</td><td>2</td><td>0.430</td></tr><tr><td>LCD matrix</td><td>liq. crystal</td><td>6</td><td>MISS</td><td></td></tr><tr><td>Windshield</td><td>windshield</td><td>5</td><td>MISS</td><td></td></tr><tr><td>Deep compr.</td><td>compr. str.</td><td>9</td><td>MISS</td><td></td></tr><tr><td colspan="5">Hit Rate: 13/16 (81.2%) MRR: 0.594 P@1: 37.5% / P@3: 81.2% / P05: 81.2%</td></tr></table>

Table 10

Keywords assigned by Step 2 of §4.6.2. Pairs: targets receiving the keyword. $| D _ { q } | \colon$ documents selected by it. Share: |pair ∩ $D _ { q } | / | D _ { q } |$ over those targets.
<table><tr><td colspan="3"></td><td colspan="3">Share of  $D _ { q }$ </td></tr><tr><td>Corpus</td><td>Keyword</td><td>Pairs</td><td> $| D _ { q } |$ </td><td>median</td><td>max</td></tr><tr><td>G06N</td><td>computer</td><td>37</td><td>4,118</td><td>0.56%</td><td>2.21%</td></tr><tr><td></td><td>data</td><td>4</td><td>3,789</td><td>0.50%</td><td>0.69%</td></tr><tr><td></td><td>information</td><td>1</td><td>2,417</td><td>0.54%</td><td>0.54%</td></tr><tr><td></td><td>memory</td><td>1</td><td>2,307</td><td>1.34%</td><td>1.34%</td></tr><tr><td></td><td>state</td><td>1</td><td>984</td><td>7.01%</td><td>7.01%</td></tr><tr><td>C03C</td><td>surface</td><td>8</td><td>1,159</td><td>1.51%</td><td>1.64%</td></tr><tr><td></td><td>temperature</td><td>7</td><td>1,086</td><td>1.29%</td><td>3.31%</td></tr><tr><td></td><td>layer</td><td>3</td><td>966</td><td>2.07%</td><td>2.59%</td></tr><tr><td></td><td>sio2</td><td>2</td><td>879</td><td>2.45%</td><td>3.30%</td></tr><tr><td></td><td>composition</td><td>1</td><td>863</td><td>1.27%</td><td>1.27%</td></tr><tr><td></td><td>forming</td><td>1</td><td>756</td><td>1.72%</td><td>1.72%</td></tr></table>

## C. Keyword assignment in the removal-driven protocol

The rule of Step 2 in §4.6.2 is deterministic but not injective, so a few generic terms cover all targets. Table 10 lists every assigned keyword with the size of the corpus it selects and the share of that corpus occupied by the target pairs it serves. The shares are uniformly small, at most 7.0%, which is what makes the pairs undetectable before removal; the concentration on “computer” is the basis of the independence caveat in §5.4.

## References

[1] B. Yoon, Y. Park, A systematic approach for identifying technology opportunities: Keyword-based morphology analysis, Technological Forecasting and Social Change 72 (2005) 145–160. doi:10.1016/j. techfore.2004.08.011.

[2] C. Son, Y. Suh, J. Jeon, Y. Park, Development of a GTM-based patent map for identifying patent vacuums, Expert Systems with Applications 39 (2012) 2489–2500. doi:10.1016/j.eswa.2011.08.101.

[3] B. Yoon, C. L. Magee, Exploring technology opportunities by visualizing patent information based on generative topographic mapping and link prediction, Technological Forecasting and Social Change 132 (2018) 105–117. doi:10.1016/j.techfore.2018.01.019.

[4] B. Kim, G. Gazzola, J.-M. Lee, D. Kim, K. Kim, M. K. Jeong, Inter-cluster connectivity analysis for technology opportunity discovery, Scientometrics 98 (2014) 1811–1825. doi:10.1007/ s11192-013-1097-2.

[5] J. Yoon, H. Park, W. Seo, J.-M. Lee, B.-Y. Coh, J. Kim, Technology opportunity discovery (TOD) from existing technologies and products: A function-based TOD framework, Technological Forecasting and Social Change 100 (2015) 153–167. doi:10.1016/j.techfore. 2015.04.012.

[6] S. Choi, S. Jun, Vacant technology forecasting using new Bayesian patent clustering, Technology Analysis & Strategic Management 26 (2014) 241–251. doi:10.1080/09537325.2013.850477.

[7] M. Grootendorst, BERTopic: Neural topic modeling with a classbased TF-IDF procedure, arXiv preprint (2022). doi:10.48550/arXiv. 2203.05794. arXiv:2203.05794.

[8] E. Jeon, N. Yoon, S. Y. Sohn, Exploring new digital therapeutics technologies for psychiatric disorders using BERTopic and PatentS-BERTa, Technological Forecasting and Social Change 186 (2023) 122130. doi:10.1016/j.techfore.2022.122130.

[9] H. Yang, S. Chen, Industry 5.0: Life-cycle mapping of sustainable technologies using BERTopic-driven patent analytics, World Patent Information 83 (2025) 102406. doi:10.1016/j.wpi.2025.102406.

[10] K. W. Church, P. Hanks, Word association norms, mutual information, and lexicography, Computational Linguistics 16 (1990) 22–29.

[11] G. Bouma, Normalized (pointwise) mutual information in collocation extraction, in: Proceedings of the International Conference of the German Society for Computational Linguistics and Language Technology (GSCL), 2009, pp. 31–40.

[12] A. B. Jafe, Technological opportunity and spillovers of R&D: Evidence from firms’ patents, profits, and market value, American Economic Review 76 (1986) 984–1001.

[13] S. Breschi, F. Lissoni, F. Malerba, Knowledge-relatedness in firm technological diversification, Research Policy 32 (2003) 69–87. doi:10.1016/S0048-7333(02)00004-5.

[14] L. McInnes, J. Healy, J. Melville, UMAP: Uniform manifold approximation and projection for dimension reduction, arXiv preprint (2018). doi:10.48550/arXiv.1802.03426. arXiv:1802.03426.

[15] L. McInnes, J. Healy, S. Astels, HDBSCAN: Hierarchical density based clustering, Journal of Open Source Software 2 (2017) 205. doi:10.21105/joss.00205.

[16] D. M. Blei, A. Y. Ng, M. I. Jordan, Latent Dirichlet allocation, Journal of Machine Learning Research 3 (2003) 993–1022.

[17] D. D. Lee, H. S. Seung, Learning the parts of objects by non-negative matrix factorization, Nature 401 (1999) 788–791. doi:10.1038/44565.

[18] N. Reimers, I. Gurevych, Sentence-BERT: Sentence embeddings using Siamese BERT-networks, in: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP), 2019, pp. 3982–3992. doi:10.18653/v1/D19-1410.

[19] J.-S. Lee, J. Hsiang, Patent classification by fine-tuning BERT language model, World Patent Information 61 (2020) 101965. doi:10. 1016/j.wpi.2020.101965.

[20] H. Bekamiri, D. S. Hain, R. Jurowetzki, PatentSBERTa: A deep NLP based hybrid model for patent distance and classification using augmented SBERT, Technological Forecasting and Social Change 206 (2024) 123536. doi:10.1016/j.techfore.2024.123536.

[21] R. Krestel, R. Chikkamath, C. Hewel, J. Risch, A survey on deep learning for patent analysis, World Patent Information 65 (2021) 102035. doi:10.1016/j.wpi.2021.102035.

[22] L. Jiang, S. M. Goetz, Natural language processing in the patent domain: A survey, Artificial Intelligence Review 58 (2025) Article 214. doi:10.1007/s10462-025-11168-z.

[23] C.-S. Curran, J. Leker, Patent indicators for monitoring convergence – examples from NFF and ICT, Technological Forecasting and Social Change 78 (2011) 256–273. doi:10.1016/j.techfore.2010.06.021.

[24] S. Lee, B. Yoon, Y. Park, An approach to discovering new technology opportunities: Keyword-based patent map approach, Technovation 29 (2009) 481–497. doi:10.1016/j.technovation.2008.10.006.

[25] C. Lee, G. Lee, Technology opportunity analysis based on recombinant search: Patent landscape analysis for idea generation, Scientometrics 121 (2019) 603–632. doi:10.1007/s11192-019-03224-7.

[26] J. H. Ward, Hierarchical grouping to optimize an objective function, Journal of the American Statistical Association 58 (1963) 236–244. doi:10.1080/01621459.1963.10500845.

[27] M. Suzgun, L. Melas-Kyriazi, S. K. Sarkar, S. D. Kominers, S. M. Shieber, The Harvard USPTO patent dataset: A large-scale, wellstructured, and multi-purpose corpus of patent applications, in: Advances in Neural Information Processing Systems (NeurIPS), volume 36, 2023.

[28] J.-S. Lee, J. Hsiang, Patent claim generation by fine-tuning OpenAI GPT-2, World Patent Information 62 (2020) 101983. doi:10.1016/j. wpi.2020.101983.

[29] L. Jiang, C. Zhang, P. A. Scherz, S. Goetz, Can large language models generate high-quality patent claims?, in: Findings of the Association for Computational Linguistics: NAACL 2025, 2025, pp. 1272–1287. doi:10.18653/v1/2025.findings-naacl.70.

[30] M. F. Porter, Snowball: A language for stemming algorithms, Technical Report, Snowball, 2001. URL: https://snowballstem.org/texts/ introduction.html, accessed 2026-04-12.