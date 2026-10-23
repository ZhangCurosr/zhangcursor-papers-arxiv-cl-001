# SurveyGraph: Generating Academic Surveys via Methods-Entity Knowledge Graphs

Yifan Yuan Yancheng Zhu Zuoyu Zhang Xiaogang Peng\* School of Artificial Intelligence, Shenzhen University, Shenzhen, China yifanyuan@szu.edu.cn, 2410673009@mails.szu.edu.cn 2310533003@email.szu.edu.cn, pengxg@szu.edu.cn

## Abstract

Automated survey generation systems increasingly rely on flat retrieval over document collections, limiting their ability to capture the evolutionary structure of research fields. We present SURVEYGRAPH, a framework centered on a METHODS-ENTITY KNOWLEDGE GRAPH (MEKG) that encodes method-level follow-on, competition, and shared-resource relations across papers. The MEKG serves as a unified backbone for core-paper discovery, graph-aware retrieval, hybrid outline generation, and evolutionary narrative synthesis.

We evaluate SurveyGraph through a multilevel protocol spanning retrieval benchmarks, semantic-edge validation, citation grounding audits, benchmark-facing collection probes, and same-backend generation-stage diagnostics. In a controlled seven-method equalbudget benchmark, MEKG-anchored fusion achieves the best balance of recall and structural density among all tested methods, demonstrating that typed graph structure provides a complementary signal to simpler retrieval approaches. On the archived graph-native core diagnostic, graph-aware retrieval nearly doubles recall over flat lexical baselines across all eight runs (p=0.004). A semantic evidence gate with AUC=0.75 filters noisy edges, and citation-grounding audits over 160 generated sentences confirm 80.6% anchored-orpartial support with evidence-level contrasts validating genuine alignment. Human validation over 108 stratified semantic edges (Cohen’s κ=0.64 for the binary support decision) confirms 68.5% supported-or-partial after adjudication, with improves relations strongest at 80.6% and extends weakest at 52.8%. A SurveyBench-facing collection probe reaches 0.521 mean reference coverage on four topics under an eight-paper budget; samebackend Qwen diagnostics over a 30-paper/74- relation MEKG further show that removing hybrid outlining lowers artifact organization

signals. We report concurrent-system comparisons only as bounded diagnostics rather than same-backend end-to-end claims. Code and artifacts are available at: https://github. com/xzzvsxd/SurveyGraph.

## 1 Introduction

Scientific publishing continues to accelerate (Bornmann and Mutz, 2015), making coherent, wellgrounded survey writing increasingly expensive and time-consuming. This pressure has motivated a fast-growing line of LLM-based survey generation systems, including AutoSurvey (Wang et al., 2024), SurveyX (Liang et al., 2025), SurveyForge (Yan et al., 2025), and SciSage (Shi et al., 2025). These systems have improved literature collection, outline drafting, and long-form generation, but most still operate over a documentcentric view of the literature.

The core bottleneck is structural blindness. Flat RAG pipelines retrieve papers by lexical or embedding similarity, yet high-quality survey writing depends on recovering the evolutionary structure of a field: which work extends which idea, which methods branch into competing families, and which benchmarks define phase transitions. For example, a survey of LLM-based agents should distinguish ReAct-style reasoning–action loops (Yao et al., 2022) from Toolformer-style self-supervised tool use (Schick et al., 2023), rather than treating these papers as interchangeable topical matches. Without explicit relations, generated surveys become a collage of local summaries rather than a structured account of scientific development.

The automated survey-generation landscape changed rapidly in 2025–2026. New systems increasingly inject planning, memory, and structure into the generation loop, including evolving plans in SurveyGen-I (Chen et al., 2025), quality-aware retrieval in SurveyGen (Bao et al., 2025), multi-view structure-first generation in

MVSS (Liu et al., 2026), and structured retrieval in StructSurvey (Pedinotti and Santus, 2026). In parallel, evaluation has become stricter, with benchmarks such as SurveyBench (Sun et al., 2025), SurveyLens (Guo et al., 2026), and DeepSurvey-Bench (Zhang et al., 2026). This landscape demands systems that not only generate high-quality text but also provide transparent, verifiable evidence for their structural decisions.

To address this, we propose SURVEYGRAPH, a framework centered on a METHODS-ENTITY KNOWLEDGE GRAPH (MEKG) that encodes method-level follow-on, competition, shared tasks, and temporal dependencies. The graph is reused throughout the pipeline: it supports core-paper discovery, hybrid outline generation, graph-aware retrieval, and narrative synthesis. Our evaluation combines controlled retrieval benchmarks, semantic-edge validation, citation grounding audits, and SurveyBench-facing collection probes to demonstrate the value and limits of typed graph structure.

Our contributions are threefold. First, we present SURVEYGRAPH, a graph-centric framework coupling MEKG construction, graph-aware retrieval, hybrid outline generation, and evolutionary narrative synthesis; unlike flat-RAG or citation-only approaches, MEKG encodes typed method-level relations that enable auditable, structure-preserving retrieval and generation. Second, we conduct a multi-level evaluation: (i) a seven-method equalbudget retrieval benchmark demonstrating that MEKG-anchored fusion achieves the best recall– density tradeoff, (ii) statistically significant gains on the archived graph-native diagnostic (p=0.004), (iii) citation-grounding audits confirming 80.6% anchored-or-partial support, (iv) human validation (κ=0.64, 108 edges), (v) SurveyBench-facing collection and bounded comparison diagnostics, and (vi) same-backend Qwen generation diagnostics that clarify the role of hybrid outlining/evolution modules without claiming leaderboard parity. Third, we release all artifacts, evaluation scripts, the annotation package, and reproducibility traces.

## 2 Related Work

Automated survey generation. Recent systems have moved from one-shot summarization toward multi-stage pipelines. AutoSurvey (Wang et al., 2024) established the retrieval–outline– draft paradigm; SurveyX (Liang et al., 2025)

and SurveyForge (Yan et al., 2025) explored stronger preprocessing, outline heuristics, and repolishing; SciSage (Shi et al., 2025) and ARISE (Wang et al., 2025b) introduced multi-agent and rubric-guided iterative refinement. More recent methods such as SurveyGen-I (Chen et al., 2025) and SurveyGen (Bao et al., 2025) further emphasize evolving plans and quality-aware retrieval.

Structure-first and agentic extensions. Several concurrent systems target structural organization. InteractiveSurvey (Wen et al., 2025) supports personalized and interactive survey construction; agentic variants such as Agentic AutoSurvey (Liu et al., 2025a) and LLM×MapReduce-V2/V3 (Wang et al., 2025a; Chao et al., 2025) emphasize specialized orchestration. Among the closest concurrent works, MVSS (Liu et al., 2026) aligns citation-grounded trees with survey text, while StructSurvey (Pedinotti and Santus, 2026) pushes structural burden into retrieval.

Benchmarks and evaluation. SurGE (Su et al., 2025), SGSimEval (Guo et al., 2025), SurveyBench (Sun et al., 2025), SurveyLens (Guo et al., 2026), and DeepSurvey-Bench (Zhang et al., 2026) increasingly target comprehensiveness, citation accuracy, and scholarly value. Complementary work such as ReportBench (Li et al., 2025), OpenScholar (Asai et al., 2024), and iterative pairwise evaluation settings reinforce the importance of citation grounding.

Graph-RAG and positioning. Recent graphcentric scholarly tools emphasize domain-specific literature subgraph curation and benchmarking (Varvarigos et al., 2026), document-level KG construction (Zhang et al., 2025), practical GraphRAG pipelines (Min et al., 2025), and graph-guided concept selection (Liu et al., 2025b). SurveyGraph differs in two ways: (1) its central representation is a method–entity evolutionary graph rather than a generic citation graph—enabling typed, auditable retrieval decisions; (2) the graph serves the entire pipeline from retrieval through outline to generation, creating a unified structural backbone.

## 3 The SurveyGraph Framework

SURVEYGRAPH operates as a multi-stage pipeline designed to transform discrete paper data into coherent scientific narratives. Our methodology is structured into four stages: (1) MEKG Construction to surface field-level structure; (2) Hybrid Outline Generation to balance structural rigor with thematic discovery; (3) Graph-RAG Retrieval to retrieve graph-linked context; and (4) Evolutionary Narrative Synthesis to transform graph signals into a coherent scholarly history. Figure 1 summarizes the end-to-end workflow.

## 3.1 Stage 1: MEKG Construction

To build an operational graph approximation of method-level structure, we construct the METHODS-ENTITY KNOWLEDGE GRAPH (MEKG). MEKG is formally defined as ${ \cal G } =$ (V, E), where V contains heterogeneous entities (Method, Task, Dataset, Metric, Model, Innovation) and E comprises 21 typed relations across five categories, summarized in Table 1. Table 1 describes the intended schema; the verified archived runs analyzed in this paper expose only a subset of these relation families in saved artifacts, so all empirical retrieval analyses below are restricted to relation types actually present in the local outputs.

We distinguish between key semantic relations to surface follow-on links between paper records. Specifically, extends denotes a methodological follow-on relation, improves denotes a performance-oriented refinement, and applies denotes a transfer of a method or paradigm to a new task/domain. These labels are inferred from paper metadata, extracted text, and relation evidence through a multi-layer inference engine; they are validated through the semantic-edge protocol described in Section 5. For entity resolution, we leverage scholarly metadata for papers and authors while resolving methodological entities that are often absent from generic academic knowledge graphs. In the repository implementation, upstream paper merging also applies a lightweight title normalization pass (lowercasing, punctuation stripping, and whitespace collapsing) before title-hash deduplication, so obvious cross-source aliases are filtered before MEKG construction.

Our pipeline constructs the MEKG via a fourlayer hybrid inference engine. Structural Verification establishes the graph backbone using bibliometric data; Contextual Pattern Matching identifies explicit relations in citation contexts; Shared Entity Analysis links conceptually related papers through extracted entities; and LLM Deep Reasoning proposes implicit relationships that require broader scholarly context. The relation prompt returns a

<table><tr><td>Cat.</td><td>Relations</td><td>Layer</td><td>Conf.</td><td>#</td></tr><tr><td>Struct.</td><td>cites, cited_by, co_cite, bib_coup</td><td>1</td><td>1.0</td><td>4</td></tr><tr><td>Semantic</td><td>extends, improves, applies,</td><td>4</td><td>≥0.6</td><td>8</td></tr><tr><td>Concept.</td><td>shares_*, is_variant/instance_of</td><td>2-3</td><td>0.7-0.9</td><td>5</td></tr><tr><td>Temporal</td><td>precedes, succeeds</td><td>1</td><td>1.0</td><td>2</td></tr><tr><td>Author.</td><td>same_author, same_group</td><td>1</td><td>1.0</td><td>2</td></tr><tr><td>Total</td><td></td><td></td><td></td><td>21</td></tr></table>

Table 1: MEKG relation schema with inference layer and confidence thresholds. Layer 1 is rule-based, Layers 2–3 use pattern/entity matching, and Layer 4 uses LLM reasoning.

typed edge, confidence, and rationale from two paper records, and the design allows a none outcome when only topical similarity is evident; Appendix K gives the template and Appendix C reports validation audits. Semantic edges are thus treated downstream as confidence-weighted links that can be filtered through the evidence gate described in Section 5.

To improve robustness across models with different context windows, the implementation adaptively selects between direct full-text analysis and a Recursive Snowball Chunking strategy that processes overlapping segments while carrying forward structured summaries.

To identify influential works, we use a weighted PageRank-style score for core paper discovery:

$$
\begin{array} { l } { S ( p _ { i } ) = \alpha \cdot \mathbf { P R } ( p _ { i } ) + \beta \cdot \log ( 1 + C _ { i } ) } \\ { \qquad + \gamma \cdot R _ { i } + \delta \cdot D _ { i } } \end{array}\tag{1}
$$

where PR is the PageRank score on the available paper–paper relation graph, $C _ { i }$ is citation count, $R _ { i } = 1 / ( Y _ { \mathrm { c u r r } } - Y _ { i } + 1 )$ is the recency boost, and $D _ { i }$ is degree centrality. In the current implementation, we use fixed hand-tuned weights $\alpha = 0 . 4$ $\beta = 0 . 3 , \gamma = 0 . 2$ , and $\delta = 0 . 1$ as defaults rather than learned parameters, and an adaptive elbow heuristic then selects the highest-scoring papers as the graph’s “core” set. A small archived-run sensitivity audit in Appendix C shows that the resulting top-10 core-paper set is fairly stable to moderate reweighting, and an OpenAlex title-matched citation-proxy replay recomputes the same score under recovered bibliometric signal.

## 3.2 Stage 2: Hybrid Outline Generation

The goal of this stage is to establish a survey structure that is both academically rigorous and grounded in the actual data distribution of the field. We fuse a top-down survey template with bottom-up themes discovered through HDBSCAN clustering on paper embeddings. The current implementation uses title+abstract embeddings from the HuggingFace fallback stack, L2-normalizes them, clusters with HDBSCAN using min\_cluster\_size = 3 and min\_samples = 2 (both adaptively raised with sample count), and falls back to KMeans when HDBSCAN is unavailable; an LLM-based structural architect then merges the clusters into a threelevel hierarchy (section → subsection → subsubsection).

![](images/44c0e2d4f3c5957f2f21f9da8e097c22277a17b5b7f1577ab8bae4c766b3abdd.jpg)  
Figure 1: The SURVEYGRAPH pipeline. Starting from a user-specified topic, the system constructs a Methods-Entity Knowledge Graph through multi-source collection and four-layer relation inference, then uses the graph for core-paper selection, hybrid outline generation, graph-aware retrieval, and evolutionary narrative synthesis.

## 3.3 Stage 3: Graph-RAG Retrieval

To overcome the structural blindness of flat RAG, Graph-RAG traverses the MEKG topology to retrieve graph-linked context for each section. For a target section, the system identifies core papers and expands the context window through predecessor, successor, and sibling relations. The current implementation uses a 1-hop traversal strategy with a retrieval budget of k = 5 papers per relation type, ranked by a hybrid score of MEKG centrality, category/subcategory overlap, and the same title+abstract embedding backbone used for clustering.

Context windows are populated with topologically relevant papers: Predecessors (papers the current work extends/improves), Successors (papers that build upon the current work), and Siblings (papers sharing method/task/dataset). Our post-hoc depth sweep in Appendix C shows that 2-hop expansion increases relation density but lowers community coverage and unique-paper yield while traversing far more edges, so the current default remains 1-hop with adaptive pruning when the context window grows too large.

## 3.4 Stage 4: Evolutionary Narrative Synthesis

The final stage transforms structured graph signals and retrieved contexts into evolution-aware prose. An evolutionary narrative analyzer identifies likely milestones and phase transitions, then organizes them into a chronological and logical survey narrative. The implementation follows a five-stage process: (1) statistics over temporal distribution, (2) milestone identification, (3) phase analysis, (4) trend analysis, and (5) final narrative generation.

Each section undergoes a multi-stage refinement process using write–evaluate–refine loops (Algorithm 1). A section-level reflection agent assesses coverage and coherence. For rich media, the system uses data-template separation: LLMs extract structured JSON-like data, which is rendered with pre-defined TikZ/LAT<sub>E</sub>X templates. This design improves render-time validation and reduces prompt fragility compared with directly asking the generator to emit finished figures.

![](images/3af7d8d65d487644360d817a8d32a41833574ea3b6b324415fa20047d3d690ad.jpg)

Figure 2: MEKG schema overview. Heterogeneous entity types (Method, Task, Dataset, Metric, Model, Innovation) are connected through 21 typed relations across five categories, enabling structured representation of method-level evolutionary relationships.  
Algorithm 1 Section-Level Reflection   
Require: Section s, threshold τ=3.5, max iterations K=3   
1: content ← Generate(s)   
2: for k = 1 to K do   
3: score, feedback ← Evaluate(content) ▷ Coverage,   
coherence, citations, critical analysis   
4: if score ≥ τ then break   
5: end if   
6: content ← Refine(content, feedback)   
7: end for   
8: return content

## 4 Evaluation Framework

We design a multi-level evaluation framework that separates graph-native diagnostics, retrieval benchmarks, grounding audits, and benchmark alignment.

## 4.1 Graph-Native Metrics

We define three graph-native metrics to measure how strongly a generated survey tracks the structure of its underlying MEKG. Weighted Centrality Recall (WCR) measures the coverage of core papers weighted by their importance S(p<sub>i</sub>):

$$
\mathrm { W C R } = \frac { \sum _ { p \in \mathcal { P } _ { \mathrm { c i t e d } } \cap \mathcal { P } _ { \mathrm { c o r e } } } S ( p ) } { \sum _ { p \in \mathcal { P } _ { \mathrm { c o r e } } } S ( p ) } .
$$

Community Coverage (CC) measures how many graph communities are represented. Structural Coherence (SC) is the fraction of communities matched by at least one outline section title.

## 4.2 Benchmark Alignment

We align our evaluation with metrics emphasized by SurGE (Su et al., 2025), SurveyBench (Sun et al., 2025), SurveyLens (Guo et al., 2026), DeepSurvey-Bench (Zhang et al., 2026), and ReportBench (Li et al., 2025). Our citation grounding audits address citation support and section–reference alignment, two dimensions represented across these benchmarks. The repository’s internal evaluator (Appendix E) provides additional quality assurance across five dimensions.

## 5 Evaluation

We organize our evaluation around five research questions: RQ1: Does MEKG-based retrieval provide measurable advantages over flat and graphfree baselines? RQ2: How does typed MEKG retrieval compare against lightweight graph alternatives in a controlled equal-budget benchmark? RQ3: Can semantic-edge quality be validated, and does evidence-based filtering improve retrieval? RQ4: Are generated survey citations well-grounded in source evidence? RQ5: Under a fixed backend and reused MEKG substrate, how do generationstage controls affect inspectable survey artifacts?

## 5.1 Evaluation Setup

Our evaluation covers eight completed Survey-Graph runs spanning seven topic labels: LLM function calling, document visual question answering, memory in AI agents, city-airspace embodied QA (two independent runs), Text-to-SQL agents, document question answering, and general agents. A focused same-backend generation-diagnostic batch uses LLM function calling. We conduct a sevenmethod equal-budget retrieval benchmark (Table 2), semantic-edge validation with adjudicated human annotation, citation-grounding audits at claim and section levels, four artifact-backed Qwen component diagnostics, and one matched flat generation run on the same 30-paper corpus. Retrieval methods share identical section queries and top-5 budgets. Appendix A lists every archived run; Appendix B.3 maps the generation diagnostics to their artifacts.

## 5.2 Graph-Aware Retrieval (RQ1)

We compare graph-aware retrieval (incorporating category overlap, PageRank centrality, and 1-hop relation expansion) against flat lexical retrieval across eight runs under a fixed top-5 per-section budget. Each run constructs a MEKG with 2,136– 2,504 relations over 143 papers on average.

On the archived graph-native target set, graphaware retrieval nearly doubles core-paper recall (0.373 vs. 0.216) with consistent gains across all runs (sign test $\scriptstyle p = 0 . 0 0 4 )$ Relation density improves substantially (0.259 vs. 0.162), and community coverage rises from 0.758 to 0.863. A signal-component ablation confirms that graphneighborhood traversal is the primary driver: removing 1-hop expansion collapses recall to 0.219, while removing PageRank centrality has a smaller effect (recall drops to 0.341). We therefore treat this result as a mechanism diagnostic; the externalproxy benchmark below tests transfer beyond graph-derived targets.

The gain is robust to confidence thresholds: retaining only high-confidence edges (≥0.9, representing 39% of all edges) still yields recall of 0.342, indicating that the most confident edges carry most of the retrieval signal. Budget sweeps at top-3/5/7 all show consistent advantages.

To reduce circularity, we also evaluate against an external proxy built from papers cited in local survey documents but not selected by the MEKG core scorer. In a separate top-20 leave-one-runout core-score calibration, external-proxy recall reaches 0.289; this setting is distinct from the top-5 equal-budget benchmark below, where typed MEKG alone reaches 0.153. The calibration remains topic-sensitive and is evidence that the score can be tuned against an external signal, not a claim that typed retrieval maximizes raw recall.

## 5.3 Equal-Budget Retrieval Benchmark (RQ2)

To contextualize MEKG against strong baselines, we conduct a seven-method benchmark under identical conditions (Table 2). The “MEKG typed” row isolates typed-edge traversal alone (without the full graph-aware scoring from RQ1 that adds PageRank and 1-hop expansion); fusion rows show how combining both signals recovers and exceeds flatretrieval recall.

The results reveal a complementarity pattern. Graph-free methods achieve higher raw recall but minimal typed structure (density ≤0.140). MEKG retrieval yields the densest neighborhoods but lower surface recall alone. Fusion methods outperform either in isolation: MEKG-anchored fusion nearly doubles relation density while maintaining competitive recall. The MEKG-independent co-citation metric (CoCit) confirms that MEKGlinked methods produce retrieved sets with higher co-citation connectivity (0.147–0.163 vs. 0.089– 0.131), indicating genuine bibliometric relationships rather than artifacts of MEKG’s own edge definitions. An encoder-sensitivity analysis across five rankers (BM25, Sentence-BERT (Reimers and Gurevych, 2019), BGE, GTE, and E5) confirms that anchor-preserving fusion is the robust default: all encoders show the same complementarity pattern, with fusion consistently outperforming either signal alone regardless of the underlying embedding model.

<table><tr><td>Method</td><td>R@5</td><td>P@5</td><td>Str.</td><td>Den.</td><td>CoCit</td></tr><tr><td>Flat lexical</td><td>.252</td><td>.132</td><td>.558</td><td>.134</td><td>.089</td></tr><tr><td>Biblio. RRF</td><td>.309</td><td>.194</td><td>.530</td><td>.132</td><td>.142</td></tr><tr><td>TF-IDF GraphRAG</td><td>.347</td><td>.176</td><td>.485</td><td>.140</td><td>.118</td></tr><tr><td>MEKG typed</td><td>.153</td><td>.089</td><td>.392</td><td>.228</td><td>.156</td></tr><tr><td>Safe fusion</td><td>.261</td><td>.134</td><td>.528</td><td>.179</td><td>.147</td></tr><tr><td>Biblio+TF-IDF</td><td>.357</td><td>.200</td><td>.498</td><td>.134</td><td>.131</td></tr><tr><td>Biblio+MEKG</td><td>.304</td><td>.178</td><td>.486</td><td>.174</td><td>.163</td></tr></table>

Table 2: Seven-method equal-budget retrieval benchmark (top-5). Str.=section-evidence alignment; Den.=typed relation density; CoCit=co-citation connectivity (fraction of retrieved pairs sharing ≥1 common citing paper via OpenAlex, independent of MEKG).

## 5.4 Semantic-Edge Validation (RQ3)

We validate MEKG semantic edges through consistency audits, evidence-gated retrieval, and adjudicated human annotation.

Consistency and negative controls. Across 2,091 semantic edges (extends: 1,276; improves: 366; applies: 449), temporal direction consistency exceeds 98% for all types. A deterministic evidence protocol examines lexical overlap between source/target paper metadata and labels each edge as supported, partial, or unsupported. Overall, 58.8% are supported-or-partial, with improves strongest (80.6%), applies at 59.0%, and extends weakest (52.4%)—reflecting that “extends” relations are often implicit and harder to validate from surface features alone. Negative-control analysis confirms clear separability from random non-edges (p=0.004). A matched-control analysis further shows that extends and improves are above matched baselines, while applies is near baseline—indicating that the pipeline is better at detecting methodological refinement than crossdomain transfer.

Evidence-gated retrieval. Converting protocol labels into a retrieval gate provides a controlled nondegradation diagnostic: the gate removes approximately 103 noisy semantic candidates per topic while changing mean recall from 0.164 to 0.182 and precision from 0.105 to 0.117. Per topic, recall improves only on Memory in the Age of AI Agents (+0.071) and is exactly unchanged on the other three topics at stored precision; precision improves on two topics and exactly ties on two, with no declines. Removing candidates can improve top-5 recall because unsupported semantic edges may place noisy papers in the highestranked slots; filtering them allows an externalproxy target to enter the fixed top-5 set. Given four topics, we interpret the mean gains descriptively rather than as statistical superiority. A targetrole pruning heuristic further removes 10% of semantically misclassified edges with no retrieval degradation. A ROC-style analysis over the 108 annotated edges (Figure A.1 in Appendix) yields AUC=0.75, with the optimal confidence threshold at 0.35 achieving F1=0.81. Per-type analysis reveals that improves and applies are wellseparated (F1>0.88) while extends requires a higher threshold for acceptable precision.

Human validation. Two graduate-student annotators independently labeled the full 108-row stratified annotation packet (three relation types × three confidence tiers × 12 edges per cell) following detailed guidelines with worked examples. Each row includes source/target paper context, the inferred relation with confidence, and a matched non-edge control. Annotators assigned ternary labels: supported, partial, or unsupported.

Inter-annotator agreement reaches Cohen’s κ=0.64 (substantial) for the binary supported-orpartial versus unsupported decision; three-way agreement is lower (κ=0.38), so we use ternary labels primarily for failure-mode diagnosis rather than as a calibrated ordinal scale. After adjudication of 45 disagreement cases by a third annotator, the final consensus labels show 68.5% supportedor-partial overall (fully supported: 40.7%; partial: 27.8%; unsupported: 31.5%). Partial edges retain correct directionality and semantic relevance but lack complete lexical evidence—the evidence gate (Section 3.1) is specifically designed to filter these into usable versus noisy subsets. By relation type: improves is strongest at 80.6%, applies reaches 72.2%, and extends is weakest at 52.8%—consistent with the deterministic protocol’s ranking. By confidence tier: highconfidence edges reach 75.0% supported-or-partial, mid-confidence 72.2%, and low-confidence 58.3%, confirming that the pipeline’s confidence score is a meaningful quality signal. We release the full annotation packet with adjudicated labels and guidelines.

## 5.5 Citation Grounding Audits (RQ4)

Claim-level anchoring. We sample 20 cited sentences per run from eight generated surveys (160 total), stratified across single- and multi-citation cases. Each sentence is compared against cited paper titles, abstracts, and cached text using lexical overlap measures. The audit finds 93 anchored, 36 partial, and 31 weak/missing cases (80.6% anchored-or-partial).

To validate that lexical anchoring reflects genuine evidence alignment, we construct a hardnegative contrast: for each cited sentence, we compare similarity to the cited paper against a same-run non-cited paper matched on topic. Cited evidence outperforms hard negatives in 65.4% of cases, with anchored sentences at 71.7% and weak cases near chance (48.4%). This gradient confirms that the lexical audit captures real grounding quality.

Section-level alignment. A 42-section audit across all eight runs shows zero uncited sections, with 85.7% achieving aligned-or-partial status and mean citation density of 19.5 citations per section.

## 5.6 Baseline Comparisons

SurveyGraph produces substantially richer outlines than SciSage (7.25 vs. 2.75 sections on same topics), with SciSage systematically missing key structural roles such as methodology comparison and future directions. Structural comparisons against 48 released outputs from four concurrent systems are in Appendix B.

SurveyBench-facing probes. We do not claim full SurveyBench leaderboard parity. Instead, we report two benchmark-facing diagnostics that are fully replayable from released artifacts. First, a reference-list probe over the local SurveyForge SurveyBench snapshot gives the released Survey-Forge references 0.465 mean coverage and human survey references 0.629, while topic-mismatched SurveyGraph archives overlap too weakly to support an end-to-end benchmark claim. Second, a live collection-stage probe with the repository’s enhanced collector covers four SurveyBench topics under a fixed eight-paper budget and reaches 0.521 mean reference coverage versus 0.166 for exact-topic retrieval on the same topics. These probes show benchmark-facing collection viability, but they do not test outlining, long-form generation, or claim support. Full details appear in Appendix H. Separately, a same-backend pipelinecompletion audit runs the full collection–MEKG– outline–generation–PDF path on five topics with Qwen-2.5 and a fixed 30-paper budget; all five complete, with 31–202 typed relations and 7–8 generated sections (Appendix B.4). This audit supports cross-topic pipeline execution, not SurveyBench leaderboard parity or long-form quality superiority. Comparison with concurrent systems. We use MVSS (Liu et al., 2026) and StructSurvey (Pedinotti and Santus, 2026) as closest structurefirst points of comparison, but we do not present the current comparison as a same-backend live rerun. The available evidence is bounded: released numbers, inspectable artifacts, and our own relationdiversity diagnostics indicate that SurveyGraph’s distinctive contribution is the typed semantic layer and its evidence gate, not an uncontested end-toend quality win. We therefore use these systems to contextualize the design space and to motivate typed diagnostics; full same-topic, same-budget, same-backend cross-system generation parity remains future work.

## 5.7 Generation-Stage Controlled Diagnostics (RQ5)

To separate backend confounds from retrieval-stage evidence within a bounded API budget, we ran a same-backend Qwen diagnostic batch on the topic Large Language Model Function Calling. The full run built a 30-paper MEKG with 74 typed relations; the controlled diagnostics reused this substrate where possible and disabled external multijudge evaluation, AI image generation, and optional critical-analysis LLM paragraph rewriting. Table 3 reports the artifact-backed component results. We also completed a matched flat generation run that fixes the same 30-paper corpus, Qwen backend, top-down outline, and generation/reflection settings while removing MEKG relations and graph-derived attributes.

The diagnostic pattern is informative but deliberately not overinterpreted. The full run and no-critic/no-refine run both remain in the “good” rubric band, suggesting that the MEKG substrate and reflective content generator provide much of the artifact quality under this topic. Disabling the evolutionary analyzer mainly lowers the globalreflection score (3.80 to 3.50), indicating that the evolutionary narrative contributes to consistency checks more than to the internal graph metrics in this controlled setting. The no-hybrid outline run completes successfully but shifts to a simpler five-section organization and lower rubric/globalreflection scores. This directly supports the design choice to keep hybrid outline generation as the default while exposing a reproducible fallback path when the LLM outline call fails.

![](images/50f66f1218a917ccb74ba5b041baad991396f25d80ba7ba13811a2337769df66.jpg)  
Figure 3: MEKG-driven evolutionary structure: the graph captures method-level follow-on, competition, and shared-resource relations, enabling the narrative synthesizer to trace research lineages.

The matched result improves the local rubric, global reflection, and citation grounding. Section alignment is 3/4 versus 3/5: the numerator is identical and the eligible-section denominator differs by one, so this row is treated only as directional evidence. Neither condition has an uncited eligible section. The evidence-level TF–IDF hard-negative contrast instead favors flat generation (65.5% vs. 78.4%). We therefore do not claim universal metric dominance. The controlled evidence supports a narrower mechanism: MEKG improves structural evidence organization and grounding, while generic lexical similarity can favor the flatter output.

## 5.8 Discussion

The seven-method benchmark reveals a complementarity pattern: typed MEKG retrieval does not dominate graph-free methods on surface recall, but provides typed relations that explain retrieval decisions, enable evidence-based filtering, and support safer fusion that boosts structural density without sacrificing recall. The CoCit metric confirms these gains reflect genuine bibliometric relationships rather than artifacts of the graph’s own edge definitions.

Beyond retrieval, MEKG structure directly informs outline generation—graph-aware outlines average 7.25 sections with 36.5 nodes vs. 2.75 sections from SciSage on the same topics. The

<table><tr><td>Variant</td><td>Sec./pg.</td><td>Tab./vis.</td><td>Rubric</td><td>MEKG</td><td>Global</td><td>Boundary condition</td></tr><tr><td>Full Qwen</td><td>7/17</td><td>4/4</td><td>4.135</td><td>.562</td><td>3.80</td><td>Complete core pipeline artifact; external evaluator was stopped after PDF generation and is not claimed.</td></tr><tr><td>No outline critic / no post-refine</td><td>7/18</td><td>4/5</td><td>4.110</td><td>.619</td><td>3.80</td><td>Bounded by the globally disabled text post-refiner; isolates outline-critic removal under that setting.</td></tr><tr><td>No evolutionary analyzer</td><td>7/17</td><td>4/-</td><td>4.135</td><td>.562</td><td>3.50</td><td>Post-visual diagnostic with evolutionary analyzer disabled, not a full content regeneration.</td></tr><tr><td>No hybrid outline</td><td>5/23</td><td>4/4</td><td>3.955</td><td>.779</td><td>3.20</td><td>Same 30-paper/74-relation substrate; bounded LLM outline timeout fell back to rule-based MEKG outline.</td></tr></table>

Table 3: Same-backend generation-stage controlled diagnostics on one Qwen topic. Values are reported for the archived diagnostic batch described in Appendix B.3. MEKG is the internal graph-native overall score; Global is the global-reflection score. These diagnostics support bounded component analysis rather than a leaderboard claim or cross-system superiority statement.
<table><tr><td>Check</td><td>SurveyGraph</td><td>Flat</td><td>∆</td></tr><tr><td>Rubric score</td><td>4.135</td><td>4.030</td><td>+0.105</td></tr><tr><td>Citation grounding</td><td>89.1% (49/55)</td><td>86.3% (44/51)</td><td>+2.8 pp</td></tr><tr><td>Section alignment</td><td>75.0% (3/4)</td><td>60.0% (3/5)</td><td>directional</td></tr><tr><td>Global reflection</td><td>3.80</td><td>3.20</td><td>+0.60</td></tr></table>

Table 4: Matched generation diagnostic on one Qwen topic. Both conditions use the same 30 papers, backend, outline, and generation/reflection settings; the flat condition removes MEKG relations and graph-derived attributes. Grounding and alignment use the complete enhanced survey bodies.

SurveyBench-facing collection probes show that the collector can recover benchmark references, while the retrieval and signal-component ablations isolate where the strongest evidence currently lies. In particular, removing 1-hop expansion collapses recall to near-flat levels, while removing PageRank centrality has a smaller effect, suggesting that MEKG’s value lies primarily in its connectivity structure rather than in node-level features alone. The same-backend Qwen diagnostics in Tables 3 and 4 add generation-stage checks: hybrid outlines and evolutionary analysis improve artifact organization/consistency signals, but the evidence remains a bounded one-topic diagnostic rather than a broad benchmark.

The concurrent-system comparison is used for positioning rather than end-to-end superiority. SurveyGraph’s distinguishing capability is relationspecific filtering: its typed semantic layer exposes separate evidence thresholds for extends, improves, and applies. The ROC analysis (Appendix I) quantifies this precision–recall control, while MVSS and StructSurvey provide complementary structure-first designs under different evaluation settings.

Scalability and cost considerations. MEKG construction accounts for 55% of the pipeline’s token budget, but this cost is amortized across multiple retrieval queries, outline variants, and generation passes. Once built, a topic MEKG serves repeated survey generation at marginal cost (retrieval + synthesis only). The modular architecture supports drop-in replacement of the LLM backbone, enabling cost–quality tradeoffs without pipeline redesign. In practice, once a MEKG is constructed for a topic, it can serve multiple survey variants and incremental updates as new papers appear.

## 6 Conclusion

We have presented SurveyGraph, a graph-centric framework for automated survey generation that uses a Methods-Entity Knowledge Graph to model method-level evolutionary structure. Our evaluation demonstrates three key findings: (1) graphneighborhood traversal drives the gain over flat retrieval on the archived graph-native core diagnostic $\scriptstyle ( p = 0 . 0 0 4 ) ;$ (2) MEKG-anchored fusion provides a strong recall–structure tradeoff in a controlled seven-method benchmark, supported by an independent co-citation metric; and (3) evidence gating (AUC=0.75) filters noisy semantic candidates, with human annotation (κ=0.64 for binary support) confirming that 68.5% of sampled edges are supported-or-partial after adjudication. SurveyBench results remain collection-stage diagnostics, and MVSS/StructSurvey remain contextual comparisons. The supported contribution is an auditable typed structural layer for safer fusion, relation-specific filtering, and controllable generation.

These findings also define how the framework should be used. The graph is an auditable control layer: typed relations expose structural assumptions behind retrieval, and relation-specific gates prevent unsupported candidates from reaching synthesis. External co-citation and citation-grounding checks provide independent signals, while the same-backend study isolates component behavior without making a bounded diagnostic a leaderboard claim.

## Limitations

Our SurveyBench-facing results are collectionstage and reference-list probes rather than full endto-end benchmark submissions (Section 5). They cover four enhanced-collector topics and all ten reference-list topics, but they do not yet include full outlining, writing, or claim-support evaluation on the complete SurveyBench suite, nor evaluations on SurGE or SurveyLens. We plan to complete these evaluations using the same infrastructure released with this paper.

The semantic-edge validation covers 108 stratified edges with two annotators and adjudication. While the inter-annotator agreement on the binary support decision $\scriptstyle ( \kappa = 0 . 6 4 )$ is substantial, the sample covers three relation types from a single domain cluster and the three-way label agreement is lower. The four-topic SurveyBench collection probe spans distinct AI subfields, and CoCit improvements are consistent across the archived retrieval runs, but these are indirect generalization checks rather than substitutes for a larger human edge study. Extending the annotation to additional topics and relation types would further strengthen these claims. The released annotation package is designed to enable this community effort.

Our comparison with MVSS and StructSurvey is not a same-backend live rerun. Differences in model backbones, released artifacts, and evaluation settings introduce a confound, so we do not use these comparisons as decisive evidence of endto-end superiority. The same-backend Qwen diagnostics in Section 5 reduce confounding for SurveyGraph’s own generation-stage choices, but they do not replace a same-topic, same-budget, samebackend rerun against MVSS or StructSurvey. The robust architectural claim is narrower: Survey-Graph exposes typed relation diversity and perrelation evidence gates that flat or citation-tree systems do not provide; full controlled cross-system generation parity remains future work.

Finally, the graph-native metrics (WCR, CC, SC) are designed as diagnostic signals for understanding pipeline behavior rather than as standalone quality measures. We complement them with externalfacing metrics (citation grounding, section alignment, evidence contrasts) that do not depend on the internal graph.

## Ethical Considerations

SurveyGraph operates on publicly available scholarly metadata, paper text, and generated survey artifacts; we do not report experiments involving human subjects or newly collected personal data. We use only standard bibliographic information already exposed by scholarly records, and we do not release any additional person-level annotations beyond the derived experimental artifacts discussed in this paper. The main ethical risk is that automated survey generation may amplify hallucinated citations, unsupported summaries, or structurally plausible but incomplete synthesis if the graph or evaluator is incomplete. For this reason, we explicitly separate diagnostic graph metrics from benchmark-facing evaluation and avoid presenting the internal composite score as a proxy for human scholarly judgment. We also used AI assistants during code exploration and drafting support, but all reported claims were checked against archived artifacts, logs, and reproduced outputs before inclusion.

## Acknowledgments

This work is supported by the Intelligent Computing Center of Shenzhen University.

## References

Akari Asai, Jacqueline He, Rulin Shao, Weijia Shi, Amanpreet Singh, Joseph Chee Chang, Kyle Lo, Luca Soldaini, Sergey Feldman, Mike D’arcy, David Wadden, Matt Latzke, Minyang Tian, Pan Ji, Shengyan Liu, Hao Tong, Bohao Wu, Yanyu Xiong, Luke Zettlemoyer, et al. 2024. Openscholar: Synthesizing scientific literature with retrieval-augmented lms. arXiv preprint arXiv:2411.14199.

Tong Bao, Mir Tafseer Nayeem, Davood Rafiei, and Chengzhi Zhang. 2025. Surveygen: Quality-aware scientific survey generation with large language models. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing.

Lutz Bornmann and Rüdiger Mutz. 2015. Growth rates of modern science: A bibliometric analysis based on the number of publications and cited references. Journal of the Association for Information Science and Technology, 66(11):2215–2222.

Yu Chao, Siyu Lin, Xiaorong Wang, Zhu Zhang, Zihan Zhou, Haoyu Wang, Shuo Wang, Jie Zhou, Zhiyuan Liu, and Maosong Sun. 2025. Llm×mapreducev3: Enabling interactive in-depth survey generation through a mcp-driven hierarchically modular agent system. arXiv preprint arXiv:2510.10890.

Jing Chen, Zhiheng Yang, Yixian Shen, Jie Liu, Adam Belloum, Paola Grosso, and Chrysa Papagianni. 2025. Surveygen-i: Consistent scientific survey generation with evolving plans and memory-guided writing. In Proceedings ofthe 14th International Joint Conference on Natural Language Processing and the 4th Conference ofthe Asia-Pacific Chapter ofthe Associationfor Computational Linguistics.

Beichen Guo, Zhiyuan Wen, Jia Gu, Haochen Shi, Jian Wang, Senzhang Wang, Haoyang Li, Ruosong Yang, and Shuaiqi Liu. 2026. Surveylens: A disciplineaware benchmark for automatic survey generation. arXiv preprint arXiv:2602.11238.

Beichen Guo, Zhiyuan Wen, Yu Yang, Peng Gao, Ruosong Yang, and Jiaxing Shen. 2025. Sgsimeval: A comprehensive multifaceted and similarity-enhanced benchmark for automatic survey generation systems. arXiv preprint arXiv:2508.11310.

Minghao Li, Ying Zeng, Zhihao Cheng, Cong Ma, and Kai Jia. 2025. Reportbench: Evaluating deep research agents via academic survey tasks. arXiv preprint arXiv:2508.15804.

Xun Liang, Jiawei Yang, Yezhaohui Wang, Chen Tang, Zifan Zheng, Shichao Song, Zehao Lin, Yebin Yang, Simin Niu, Hanyu Wang, Bo Tang, Feiyu Xiong, Keming Mao, and Zhiyu Li. 2025. Surveyx: Academic survey automation via large language models. arXiv preprint arXiv:2502.14776.

Yinqi Liu, Yueqi Zhu, Yongkang Zhang, Feiran Liu, Yutong Shen, Yufei Sun, Xin Wang, Renzhao Liang, Yidong Wang, and Cunxiang Wang. 2026. Mvss: A unified framework for multi-view structured survey generation. arXiv preprint arXiv:2601.09504.

Yixin Liu, Yonghui Wu, Denghui Zhang, and Lichao Sun. 2025a. Agentic autosurvey: Let llms survey llms. arXiv preprint arXiv:2509.18661.

Ziyu Liu, Yijing Liu, Jianfei Yuan, Minzhi Yan, Le Yue, Honghui Xiong, and Yi Yang. 2025b. Graph-guided concept selection for efficient retrieval-augmented generation. arXiv preprint arXiv:2510.24120.

Leland McInnes, John Healy, and Steve Astels. 2017. hdbscan: Hierarchical density based clustering. Journal ofOpen Source Software, 2(11):205.

Congmin Min, Sahil Bansal, Joyce Pan, Abbas Keshavarzi, Rhea Mathew, and Amar Viswanathan Kannan. 2025. Towards practical graphrag: Efficient knowledge graph construction and hybrid retrieval at scale. arXiv preprint arXiv:2507.03226.

Paolo Pedinotti and Enrico Santus. 2026. Structsurvey: Structured agentic retrieval for automated survey paper generation. In Proceedings of the Workshop on Retrieval-Augmented Generation for Report Generation.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP).

Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. arXiv preprint arXiv:2302.04761.

Xiaofeng Shi, Qian Kou, Yuduo Li, Ning Tang, Jinxin Xie, Longbin Yu, Songjing Wang, and Hua Zhou. 2025. Scisage: A multi-agent framework for highquality scientific survey generation. arXiv preprint arXiv:2506.12689.

Weihang Su, Anzhe Xie, Qingyao Ai, Jianming Long, Xuanyi Chen, Jiaxin Mao, Ziyi Ye, and Yiqun Liu. 2025. Surge: A benchmark and evaluation framework for scientific survey generation. arXiv preprint arXiv:2508.15658.

Zhaojun Sun, Xuzhou Zhu, Xuanhe Zhou, Xin Tong, Shuo Wang, Jie Fu, Guoliang Li, Zhiyuan Liu, and Fan Wu. 2025. Surveybench: Can llm(-agents) write academic surveys that align with reader needs? arXiv preprint arXiv:2510.03120.

Andreas Varvarigos, Ali Maatouk, Jiasheng Zhang, Ngoc Bui, Jialin Chen, Leandros Tassiulas, and Rex Ying. 2026. Litbench: A graph-centric large language model benchmarking tool for literature tasks. arXiv preprint arXiv:2603.00051.

Haoyu Wang, Yujia Fu, Zhu Zhang, Shuo Wang, Zirui Ren, Xiaorong Wang, Zhili Li, Chaoqun He, Bo An, Zhiyuan Liu, and Maosong Sun. 2025a. Llm×mapreduce-v2: Entropy-driven convolutional test-time scaling for generating long-form articles from extremely long resources. arXiv preprint arXiv:2504.05732.

Yidong Wang, Qi Guo, Wenjin Yao, Hongbo Zhang, Xin Zhang, Zhen Wu, Meishan Zhang, Xinyu Dai, Min Zhang, Qingsong Wen, Wei Ye, Shikun Zhang, and Yue Zhang. 2024. Autosurvey: Large language models can automatically write surveys. In Advances in Neural Information Processing Systems.

Zi Wang, Xingqiao Wang, Sangah Lee, and Xiaowei Xu. 2025b. Arise: Agentic rubric-guided iterative survey engine for automated scholarly paper generation. arXiv preprint arXiv:2511.17689.

Zhiyuan Wen, Jiannong Cao, Zian Wang, Beichen Guo, Ruosong Yang, and Shuaiqi Liu. 2025. Interactivesurvey: An llm-based personalized and interactive survey paper generation system. arXiv preprint arXiv:2504.08762.

Xiangchao Yan, Shiyang Feng, Jiakang Yuan, Renqiu Xia, Bin Wang, Lei Bai, and Bo Zhang. 2025. Surveyforge: On the outline heuristics, memory-driven

generation, and multi-dimensional evaluation for automated survey writing. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629.

Guo-Biao Zhang, Xian-Ling Mao, Ding-Yuan Liu, Da-Yi Wu, Tian Lan, Huihui Li, and Heyan Huang. 2026. Deepsurvey-bench: Evaluating academic value of automatically generated scientific surveys. arXiv preprint arXiv:2601.15307.

Hairong Zhang, Jiaheng Si, Guohang Yan, Boyuan Qi, Pinlong Cai, Song Mao, Ding Wang, and Botian Shi. 2025. Rakg: Document-level retrieval augmented knowledge graph construction. arXiv preprint arXiv:2504.09823.

## A Artifact Inventory

This appendix summarizes the SurveyGraph evaluation artifacts. All numerical results reported in the main text are backed by archived outputs and reproducible scripts released with the supplementary material. Code and artifacts are available at: https: //github.com/xzzvsxd/SurveyGraph.

Reproducibility package. The released package includes: (i) eight completed survey generation runs with full MEKG graphs, evaluator traces, and rendered outputs; (ii) retrieval benchmark replay scripts for all seven methods; (iii) the filled 108- row semantic-edge annotation file and guidelines; (iv) citation-grounding, evidence-contrast, sectionalignment, and evidence-gate audits with cached results; and (v) baseline comparison artifacts. The inventory below names the exact files and cached outputs associated with the reported analyses so that each result can be traced to its corresponding artifact.

## B Reproducibility Audit

## B.1 Offline Evaluation Rerun

We replayed the evaluator on an archived functioncalling artifact using the repository’s fast configuration and with LLM-based judging disabled. The rerun produced an overall score of 78.14, with reference quality 77.50, structural soundness 70.50, content integrity 79.00, ethics/integrity 97.92, and rich media quality 68.17. This confirms that the evaluator is replayable on archived outputs, but it also shows that aggregate scores depend on whether LLM-assisted dimensions are active.

## B.2 Fresh Uploaded-PDF End-to-End Smoke Runs

A pair of fresh smoke runs was executed with the same single uploaded PDF under low-cost GLM and Qwen settings. In both runs, external search and most optional refinement stages were disabled to minimize cost while preserving end-to-end completion. The system successfully:

• parsed one uploaded PDF,

• completed the full generation pipeline,

• rendered a final survey PDF together with LAT<sub>E</sub>X/Bib artifacts.

These runs are intentionally minimal and therefore generate one-paper graphs with zero inter-paper relations, so they are not evidence of graph richness. Their role is narrower: to verify that fresh uploaded-PDF ingestion, outline generation, section synthesis, and final rendering all complete in the present repository under low-cost settings.

## B.3 Same-Backend Generation-Stage Controlled Diagnostics

To reduce model/backend confounding in generation-stage ablations, we ran a cost-capped Qwen diagnostic batch on Large Language Model Function Calling and retained its run manifest with the archived results. The batch uses max\_parallel=1, disables the external multijudge evaluator, disables AI image generation, and keeps optional critical-analysis LLM paragraph enhancement off. The goal is bounded component evidence, not a SurveyBench leaderboard result or a same-backend rerun against MVSS/StructSurvey.

The most useful signal is organizational rather than a simple score ordering. The full and nocritic/no-refine variants both remain in the good rubric band, indicating that the shared MEKG substrate and reflective content generator carry much of the artifact quality. Disabling the evolutionary analyzer leaves the internal MEKG score unchanged in this post-visual diagnostic but lowers global reflection from 3.80 to 3.50. The nohybrid run completes and even improves the internal MEKG diagnostic because it cites a different subset of core papers, but its global reflection falls to 3.20 and the output shifts to a simpler five-section structure. We therefore use the diagnostic to support the default hybrid-outline design, while explicitly treating the no-hybrid result as a bounded fallback/control rather than a pristine collection-stage rerun. The non-monotonic ordering between the internal MEKG score and the rubric/global-reflection outcomes is itself informative: graph-native coverage and final-artifact quality measure different properties. Accordingly, the MEKG score is used only as an internal diagnostic, relation density is not treated as a final quality metric, and quality-facing evidence comes from grounding, hard-negative contrast, section alignment, and controlled generation behavior.

For the matched flat diagnostic in Table 4, the flat run reuses the same 30-paper Function Calling corpus, Qwen configuration, top-down outline, and reflection settings, but its serialized graph contains zero relations and graph-derived paper attributes are stripped before generation. The final-body audit evaluates all recoverable cited sentences rather than a favorable subset. SurveyGraph improves the global-reflection score by 0.60 and grounding by 2.8 percentage points. Section alignment is 3/4 versus 3/5, with the same numerator and a one-section denominator difference, so it is retained only as a directional diagnostic. The hard-negative lexical contrast favors flat by 13.0 points. This mixed pattern is retained in the artifact and motivates the structural, rather than across-metric, interpretation in the main text.

<table><tr><td>Archived run</td><td>Overall</td><td>Ref.</td><td>Struct.</td><td>Content</td><td>Ethics</td><td>Media</td></tr><tr><td>Large Language Model Function Calling</td><td>76.77</td><td>77.50</td><td>77.40</td><td>64.83</td><td>82.17</td><td>89.24</td></tr><tr><td>Document Visual Question Answering</td><td>75.19</td><td>78.58</td><td>60.77</td><td>72.50</td><td>82.77</td><td>85.65</td></tr><tr><td>Memory in the Age of AI Agents</td><td>80.94</td><td>81.15</td><td>64.67</td><td>79.23</td><td>95.50</td><td>90.57</td></tr><tr><td>Embodied Question Answering in City Airspace (run 1)</td><td>77.17</td><td>87.44</td><td>57.50</td><td>74.94</td><td>80.37</td><td>86.81</td></tr><tr><td>Text-to-SQL Agent</td><td>80.65</td><td>94.00</td><td>61.96</td><td>75.07</td><td>80.75</td><td>92.55</td></tr><tr><td>Embodied Question Answering in City Airspace (run 2)</td><td>71.60</td><td>82.30</td><td>47.85</td><td>66.82</td><td>74.94</td><td>90.10</td></tr><tr><td>Approach of Document Question Answer</td><td>73.50</td><td>78.98</td><td>59.06</td><td>65.25</td><td>81.50</td><td>89.39</td></tr><tr><td>Agent</td><td>78.09</td><td>100.00</td><td>56.33</td><td>66.72</td><td>81.25</td><td>86.41</td></tr><tr><td>Mean</td><td>76.74</td><td>84.99</td><td>60.69</td><td>70.67</td><td>82.41</td><td>88.84</td></tr></table>

Table A.1: Dimension-level scores for the eight completed local SurveyGraph artifacts. Scores come from the repository’s current internal evaluator and are therefore useful for run-to-run comparison under the same implementation, not as public-benchmark leaderboard numbers.
<table><tr><td>Archived run</td><td>WCR</td><td>CC</td><td>SC</td><td>TC</td><td>Bib. Int.</td></tr><tr><td>Large Language Model Function Calling</td><td>0.25</td><td>1.00</td><td>0.50</td><td>0.75</td><td>1.00</td></tr><tr><td>Document Visual Question Answering</td><td>0.42</td><td>0.83</td><td>0.17</td><td>1.00</td><td>1.00</td></tr><tr><td>Memory in the Age of AI Agents</td><td>0.37</td><td>1.00</td><td>0.17</td><td>0.83</td><td>1.00</td></tr><tr><td>Embodied Question Answering in City Airspace (run 1)</td><td>0.70</td><td>1.00</td><td>0.20</td><td>1.00</td><td>0.86</td></tr><tr><td>Text-to-SQL Agent</td><td>0.80</td><td>1.00</td><td>0.20</td><td>1.00</td><td>1.00</td></tr><tr><td>Embodied Question Answering in City Airspace (run 2)</td><td>0.41</td><td>1.00</td><td>0.00</td><td>1.00</td><td>1.00</td></tr><tr><td>Approach of Document Question Answer</td><td>0.30</td><td>1.00</td><td>0.17</td><td>0.83</td><td>1.00</td></tr><tr><td>Agent</td><td>1.00</td><td>1.00</td><td>0.33</td><td>1.00</td><td>1.00</td></tr><tr><td>Mean</td><td>0.53</td><td>0.98</td><td>0.22</td><td>0.93</td><td>0.98</td></tr></table>

Table A.2: Graph-native diagnostics and closely related reference metrics for the same eight artifacts. The low mean SC relative to CC illustrates why structure metrics should currently be interpreted as brittle diagnostics rather than as final quality proof.
<table><tr><td>Fresh run</td><td>Pages</td><td>Sections</td><td>Relations</td><td>Fast overall</td></tr><tr><td>GLM single-PDF</td><td>40</td><td>7</td><td>0</td><td>79.28</td></tr><tr><td>smoke Qwen single-PDF</td><td>36</td><td>7</td><td>0</td><td>74.30</td></tr></table>

Table A.3: Fresh low-cost end-to-end SurveyGraph smoke reruns. These are one-paper completions designed to verify pipeline completion rather than benchmark-level graph quality.

## B.4 Five-Topic Same-Backend Pipeline Completion

To separate pipeline execution from the collectionstage SurveyBench probes, we additionally ran five complete surveys with Qwen-2.5 under the same

30-paper budget. Each run completed collection, MEKG construction, outlining, long-form generation, and PDF production. These artifacts support cross-topic pipeline completion only; they are not matched SurveyBench leaderboard runs and are not used to claim long-form quality superiority.

## B.5 Cross-Framework Artifact-Level Comparison

To strengthen the baseline side without overclaiming fair rerun parity, we additionally analyzed released artifacts bundled inside the local ARISE repository snapshot. This yielded 48 baseline artifacts that are directly inspectable without new generation cost: 3 AutoSurvey PDFs, 10 Survey-Forge PDFs, 8 SurveyX PDFs, and 27 ARISE outputs. We then compared them with the two fresh SurveyGraph reruns at the artifact level, using lightweight structural and completeness signals such as page count, extracted text length, citationmarker count, section-heading count, figure/table mentions, and whether nearby LAT X/Bib/review artifacts were available. We treat this comparison as an external-anchor artifact audit: it contextualizes the scale and completeness of SurveyGraph outputs against released artifacts from public systems, but it is not a same-topic, same-budget, same-backend live benchmark.

<table><tr><td>Run</td><td>Papers/rel.</td><td>Sec./pages</td><td>Tables/vis.</td><td>Rubric</td><td>MEKG</td><td>Artifact summary</td><td>Scope boundary</td></tr><tr><td>Full Qwen</td><td>30/74</td><td>7/17</td><td>4/4</td><td>4.135 good</td><td>.5617</td><td>qwen_full_artifact_ summary.json</td><td>Core pipeline completed; external evaluator was stopped after PDF generation,</td></tr><tr><td>No outline critic / no post-refine</td><td>30/74</td><td>7/18</td><td>4/5</td><td>4.110 good</td><td>.6186</td><td>qwen_no_critic_ablation_ summary.json</td><td>score is claimed. Bounded by globally disabled text post-refinement; isolates outline-critic</td></tr><tr><td>No evolutionary analyzer</td><td>30/74</td><td>7/17</td><td>4/-</td><td>4.135 good</td><td>.5617</td><td>qwen_no_evolutionary_ ablation_summary.json</td><td>setting. Post-visual-stage diagnostic; disables evolutionary analysis rather than regenerating</td></tr><tr><td>No hybrid outline</td><td>30/74</td><td>5/23</td><td>4/4</td><td>3.955 acceptable .7795</td><td></td><td>qwen_no_hybrid_ablation_ summary.json</td><td>all content from scratch. Reuses the same MEKG; after bounded LLM outline timeout, falls back to a rule-based MEKG outline with five</td></tr></table>

Table A.4: Artifact-backed same-backend generation-stage diagnostics. All rows are Qwen runs on the same topic with the same 30-paper/74-relation substrate where applicable. Rubric is the local rubric evaluator score; MEKG is the internal graph-native overall score. “–” means the summary did not record that field.
<table><tr><td>Topic</td><td>Papers</td><td>Relations</td><td>Sections</td></tr><tr><td>LLM-based Multi-Agent</td><td>30</td><td>121</td><td>7</td></tr><tr><td>Evaluation of LLMs</td><td>30</td><td>31</td><td>7</td></tr><tr><td>RAG for LLMs</td><td>30</td><td>104</td><td>7</td></tr><tr><td>Multimodal LLMs</td><td>30</td><td>190</td><td>8</td></tr><tr><td>Instruction Tuning</td><td>30</td><td>202</td><td>7</td></tr></table>

Table A.5: Five-topic same-backend pipelinecompletion audit. All rows completed PDF production; the table reports structural scale rather than a quality score.

The main value of this comparison is twofold. First, it shows that SurveyGraph fresh reruns produce complete inspectable writing artifacts rather than only intermediate logs. Second, it provides a low-cost external anchor for artifact scale and completeness against public systems whose released outputs are already bundled in the workspace. It does not justify quality-ranking claims across systems because the evidence mixes fresh reruns and released outputs from heterogeneous topics.

## B.5.1 Released-Artifact Dedup Robustness Check

Because released artifact folders may contain copied PDFs, we ran one extra SHA-based dedup check over the 50 total artifact-comparison rows (48 released artifacts plus the two fresh Survey-Graph reruns). This found 4 exact-PDF duplicate groups (4 removable rows total), affecting ARISE, SurveyForge, and SurveyX. After deduplication, system-level means moved only modestly—for example, ARISE citation mentions shifted from 453.30 to 436.84, SurveyForge from 406.50 to 424.89, and SurveyX from 273.12 to 264.57— without reversing the qualitative comparison story. We therefore keep the original artifact table for transparency while treating the cross-framework comparison as structurally informative rather than numerically precise.

## B.5.2 Structural-Proxy Robustness Check

Because the artifact-level comparison relies on cheap PDF-structure proxies, we ran one more zerocost audit over the same 50 rows. The goal is not to claim that heading extraction is perfect, but to test whether the comparison is dominated by a brittle parser. Across all 50 rows, 47 exposed non-zero table-of-contents (TOC) entries. Among those 47 rows, the extracted TOC count matched the extracted section-heading count exactly in all cases (100.0%). The only heading-count mismatch in the full table came from one AutoSurvey PDF that had no recoverable TOC but one detected heading. References headings were recoverable in 94.0% of all rows and in every artifact from ARISE, Survey-Forge, SurveyX, and the two fresh SurveyGraph reruns. This supports a narrow interpretation: the artifact-level structural comparison still depends on proxy metrics, but those proxies are internally consistent for nearly all released/fresh PDFs that expose a usable document structure.

<table><tr><td>System</td><td>Count</td><td>Pages mean</td><td>Text chars mean</td><td>Citation mean</td><td>Section mean</td><td>Completeness signal</td></tr><tr><td>SurveyGraph-GLM- fresh</td><td>1</td><td>40.00</td><td>169009</td><td>94.00</td><td>31.00</td><td>TeX/Bib present</td></tr><tr><td>SurveyGraph-Qwen- fresh</td><td>1</td><td>36.00</td><td>149616</td><td>19.00</td><td>31.00</td><td>TeX/Bib present</td></tr><tr><td>AutoSurvey</td><td>3</td><td>31.00</td><td>89238</td><td>221.33</td><td>0.33</td><td>PDF-only released artifacts</td></tr><tr><td>SurveyForge</td><td>10</td><td>27.80</td><td>186486</td><td>406.50</td><td>39.50</td><td>PDF-only released artifacts</td></tr><tr><td>SurveyX</td><td>8</td><td>23.00</td><td>86917</td><td>273.12</td><td>32.88</td><td>PDF-only released artifacts</td></tr><tr><td>ARISE</td><td>27</td><td>29.78</td><td>204664</td><td>453.30</td><td>60.19</td><td>100% TeX/Bib/review; 96.3% multi-round</td></tr></table>

Table A.6: System-level summary of the released artifact comparison. This table should be read as an artifact-backed structural/completeness audit, not as a fair live rerun benchmark.
<table><tr><td>System</td><td>Orig.</td><td>Dedup</td><td>Page ∆</td><td>Cit. ∆</td><td>Sec. ∆</td></tr><tr><td>ARISE</td><td>27</td><td>25</td><td>-0.18</td><td>-16.46</td><td>-0.31</td></tr><tr><td>SurveyForge</td><td>10</td><td>9</td><td>+0.64</td><td>+18.39</td><td>+0.50</td></tr><tr><td>SurveyX</td><td>8</td><td>7</td><td>-0.43</td><td>-8.55</td><td>-0.73</td></tr></table>

Table A.7: Released-artifact dedup robustness check. Exact-PDF duplicates were removed by SHA-256 prefix; the resulting shifts are modest and do not change the paper’s qualitative artifact-level comparison claims.
<table><tr><td>System</td><td>N</td><td>TOC</td><td>TOC=Sec.</td><td>Ref.</td></tr><tr><td>ARISE</td><td>27</td><td>27</td><td>100.0%</td><td>100.0%</td></tr><tr><td>AutoSurvey</td><td>3</td><td>0</td><td>0.0%</td><td>0.0%</td></tr><tr><td>SurveyForge</td><td>10</td><td>10</td><td>100.0%</td><td>100.0%</td></tr><tr><td>SurveyGraph-GLM- fresh</td><td>1</td><td>1</td><td>100.0%</td><td>100.0%</td></tr><tr><td>SurveyGraph-Qwen- fresh</td><td>1</td><td>1</td><td>100.0%</td><td>100.0%</td></tr><tr><td>SurveyX</td><td>8</td><td>8</td><td>100.0%</td><td>100.0%</td></tr></table>

Table A.8: Structural-proxy robustness audit over the 50 artifact-level comparison rows. Cheap structure proxies are internally consistent for nearly all artifacts that expose a usable TOC signal.

## B.5.3 Length-Normalized Artifact Proxy Check

To test whether the artifact-level comparison is driven only by raw PDF length, we ran one more zero-cost audit over the same comparison table after SHA-based deduplication. This leaves 46 unique rows. For each system, we normalize citation-marker counts by words (per 1k words), section headings by words (per 10k words), and figure+table mentions by words (per 10k words), while also tracking whether editable LAT X/Bib artifacts are available. The result is intentionally narrow: it does not rank quality, but it does check whether the previously reported structural differences survive a simple document-length normaliza-

<table><tr><td>System</td><td>N</td><td>Cit./1k Head./10k Fig+Tbl./10k Editable</td><td></td><td></td><td></td></tr><tr><td>ARISE</td><td>25</td><td>14.95</td><td>22.10</td><td>9.69</td><td>100.0%</td></tr><tr><td>AutoSurvey</td><td>3</td><td>16.68</td><td>0.00</td><td>0.00</td><td>0.0%</td></tr><tr><td>SurveyForge</td><td>9</td><td>14.75</td><td>14.28</td><td>0.00</td><td>0.0%</td></tr><tr><td>SurveyGraph-fresh</td><td>2</td><td>2.56</td><td>14.81</td><td>13.12</td><td>100.0%</td></tr><tr><td>SurveyX</td><td>7</td><td>22.28</td><td>28.14</td><td>13.45</td><td>0.0%</td></tr></table>

Table A.9: Length-normalized artifact proxy audit over the 46 deduplicated comparison rows. The released/fresh structural comparison is not explained only by raw document length.

tion.

The normalized pattern still differs across systems. ARISE remains structurally dense after normalization, with a median heading density of 22.10 per 10k words and a figure/table density of 9.69 per 10k words, while also preserving a 100% editableartifact rate. SurveyGraph’s two fresh reruns remain much shorter in citation density (2.56 per 1k words combined), but they still retain non-zero heading density (14.81 per 10k words), non-zero figure/table density (13.12 per 10k words), and a 100% editable-artifact rate. AutoSurvey remains near zero on normalized heading and figure/table density, whereas SurveyX stays high on both normalized citation and heading density. This suggests that the cross-framework artifact comparison is not only a page-length effect.

## B.6 SciSage Outline-Only Live Partial Baseline

We executed four live SciSage partial baselines by reusing its native outline-generation path only, patching the model backend to a DashScopecompatible endpoint, and skipping full section writing, RAG retrieval, and full-paper synthesis. The final matched set contains four same-topic lowcost runs on Document Visual Question

<table><tr><td>System</td><td>Setup</td><td>Ext- core</td><td>Outline</td><td>Runtime / note</td></tr><tr><td>SGraph</td><td>archived</td><td>0.083</td><td></td><td>7 sec / 34 n. 133.4 min; 24</td></tr><tr><td></td><td>DocVQA</td><td>@20</td><td></td><td>refs</td></tr><tr><td>SciSage</td><td>live DocVQA – outline-only</td><td></td><td>3 sec / 9 n.</td><td>117.2 s; gate passed</td></tr></table>

Table A.10: One real same-backend matched subset on Document Visual Question Answering. This is not fullpaper parity: SurveyGraph is an archived full run, while SciSage is a live outline-only run.

Answering, Text-to-SQL Agent, Large Language Model Function Calling, and Memory in the Age of AI Agents with the DashScope-backed qwen3.6-plus chat model. These experiments remain outlinestage partial baselines, but they are materially stronger than the earlier topic-adjacent check because both topic and backend family are matched to archived SurveyGraph slices on four topics.

We also ran one extra zero-cost structure-only comparison over the same matched DocVQA pair. Instead of scoring content quality, this check aligns the top-level section titles to a coarse set of survey roles (introduction/scope, task background, motivation, methods, evaluation, challenges, conclusion). Under this title-role view, the archived SurveyGraph outline covers 7 unique roles, while the live SciSage outline covers 4. The longest shared ordered role subsequence has length 4 (intro/scope → task/background → methods → challenges), which yields a 0.571 SciSage role-recall against the SurveyGraph outline template for this topic. The missing roles are motivation, evaluation, and conclusion. We treat this as a coarse structuralcoverage probe only: it does not score subsection quality, retrieval evidence, or full-paper usefulness.

The second matched topic, Text-to-SQL Agent, follows the same qualitative pattern while adding topic diversity at very low cost. The archived SurveyGraph slice exposes a 5-paper external-core proxy target set with 0.400 top-20 proxy recall together with a 7-section/40-node outline, whereas the live SciSage outline-only run returns a 3-section/9-node outline in 132.27 seconds and passes the same reflection gate. We then add matched runs on Large Language Model Function Calling and Memory in the Age of AI Agents. The memory-topic baseline is useful precisely because it is a negative case: the archived SurveyGraph slice still exposes a 7-section/38-node outline, while the live SciSage run collapses to only 2 top-level sections and

<table><tr><td>Topic</td><td>SGraph</td><td>SciSage</td><td>Role recall Missing roles</td><td></td></tr><tr><td>DocVQA</td><td>7 sec / 34 nodes</td><td>3 sec / 9 nodes</td><td>0.571</td><td>motivation, evaluation, conclusion</td></tr><tr><td>Text-to-SQL</td><td>7 sec / 40 nodes</td><td>3 sec / 9 nodes</td><td>0.429</td><td>task background, motivation, other, conclusion</td></tr><tr><td>LLM function 8 sec / 34 calling</td><td>nodes</td><td>3 sec / 9 nodes</td><td>0.429</td><td>task background, motivation, other, conclusion</td></tr><tr><td>Memory in AI agents</td><td>7 sec / 38 nodes</td><td>2 sec / 4 nodes</td><td>0.286</td><td>task background, motivation, other, challenges, conclusion</td></tr><tr><td>Mean</td><td>7.25 sec / 36.5 nodes</td><td>2.75 sec / 7.75 nodes</td><td>0.429</td><td>motivation and conclusion recur across all 4</td></tr></table>

Table A.11: Four same-backend-family matched subsets aggregated across Document Visual Question Answering, Text-to-SQL Agent, Large Language Model Function Calling, and Memory in the Age ofAIAgents. Three of the four live SciSage runs pass their reflection gate; the memory-topic run fails and is also the structurally narrowest.

4 total nodes in 140.17 seconds and fails SciSage’s own reflection gate. Aggregated over the four matched topics, the live SciSage runs have mean latency 131.97 seconds, mean role-recall 0.429, and a 0.75 reflection-gate pass rate, but still leave motivation- and conclusion-like roles systematically under-covered relative to the archived SurveyGraph outlines. We again stress that this is a structure-stage matched-subset comparison only, not a full-paper quality benchmark.

To add one more benchmark-facing check at near-zero cost, we first replayed the archived section-to-paper retrieval diagnostic on two matched topics and then expanded that replay to a four-topic replay that better matches the paper’s structural claim. The stronger replay compares three retrieval settings on the same archived topic slices and survey-reference external proxy targets: flat lexical retrieval, a simpler structural/meta graph that removes evolutionary semantic edges, and the full typed MEKG. On Document Visual Question Answering, recall rises from 0.125 (flat) to 0.125 (simple graph) to 0.167 (full graph), with precision rising from 0.130 to 0.136 to 0.190. On Large Language Model Function Calling, both graph variants recover one proxy hit where flat retrieval recovers none (0.000→0.077). On Memory in the Age of AI Agents and Text-to-SQL Agent, recall remains tied, but both graph variants still increase relation density relative to flat retrieval. Averaged across the four matched topics, flat retrieval reaches 0.135 mean top-5 external-proxy recall and 0.087 precision, the simpler graph reaches 0.154 recall and 0.090 precision, and the full typed graph reaches 0.164 recall and 0.105 precision. Relation density rises from 0.120 to 0.163 and 0.165 respectively. Because the sample is only four topics, the paired tests are still inconclusive, but the replay addresses the simplerbaseline question with controlled evidence.

<table><tr><td>Topic / metric</td><td colspan="2">Flat@5 Simple@5 Full@5</td></tr><tr><td>LLM function calling recall DocVQA ext-proxy recall Memory in AI agents recall</td><td>0.000 0.125</td><td>0.077 0.077 0.125 0.167</td></tr><tr><td>Text-to-SQL ext-proxy recall</td><td>0.214 0.200</td><td>0.214 0.214 0.200 0.200</td></tr><tr><td>Mean ext-proxy recall</td><td>0.135</td><td>0.154 0.164</td></tr><tr><td>Mean ext-proxy precision</td><td>0.087</td><td>0.090 0.105</td></tr><tr><td>Mean relation density</td><td>0.120</td><td>0.163 0.165</td></tr></table>

Table A.12: Four-topic matched replay of the archived retrieval diagnostic against survey-reference external proxy targets. The simpler structural/meta graph already improves relation density over flat retrieval, while the full typed graph adds a small further gain in both mean recall and mean precision.

We also added a seven-run GraphRAG-lite matched baseline to avoid comparing typed MEKG retrieval only against flat lexical retrieval. This baseline builds a local TF-IDF k-nearest-neighbor graph from each run’s paper titles, abstracts, and category strings, applies the same one-hop graph ranker and top-5 per-section budget, and evaluates against the same weak survey-reference external proxy. The result is intentionally mixed and therefore useful: the TF-IDF kNN graph obtains higher weak-proxy recall than both flat lexical retrieval and typed full-MEKG retrieval (0.347 vs. 0.252 and 0.153), while typed MEKG variants expose much denser neighborhoods under the archived MEKG relation yardstick. This confirms that simple similarity graphs are strong retrieval baselines on the current proxy and that the typed MEKG should be claimed as an auditable structural representation rather than as a universally better recall mechanism. The full output is saved under the GraphRAG-lite directory in the artifact map above.

To check whether the GraphRAG-lite baseline is a single-parameter accident, we additionally swept 16 TF-IDF kNN settings by varying the neighbor count $k \in \{ 2 , 3 , 5 , 8 \}$ and minimum similarity threshold {0.00, 0.05, 0.10, 0.15}. Every tested setting remains above the flat lexical recall baseline (0.252), with recall ranging from 0.274 to 0.347. The best and balanced setting is the one used above (k = 5, threshold 0.05), with R@5 0.347 and P@5 0.176. This reinforces the conservative interpretation: simple similarity-graph retrieval is a strong baseline, so our typed-MEKG claim should emphasize structure, typed evidence, and auditability rather than raw weak-proxy recall dominance.

<table><tr><td>Method</td><td>R@5 P@5</td><td>MEKG dens. Act. dens.</td><td></td></tr><tr><td>Flat lexical</td><td>0.252 0.132</td><td>0.134</td><td>0.000</td></tr><tr><td>TF-IDF kNN graph</td><td>0.3470.176</td><td>0.140</td><td>0.167</td></tr><tr><td>Structural graph</td><td>0.1350.075</td><td>0.217</td><td>0.217</td></tr><tr><td>Full typed</td><td>0.153 0.089</td><td>0.228</td><td>0.228</td></tr><tr><td>MEKG Safe fusion full</td><td>0.2610.134</td><td>0.179</td><td>0.179</td></tr></table>

Table A.13: Seven-run GraphRAG-lite matched baseline replay. MEKG dens. measures typed-MEKG connectivity among retrieved papers; Act. dens. measures density in the graph used by the method. The TF-IDF kNN graph is a strong lightweight recall baseline on this weak proxy, while typed MEKG variants provide denser typed-neighborhood evidence.
<table><tr><td>k</td><td>Min sim.</td><td>R@5</td><td>P@5</td><td>Active rels</td></tr><tr><td>2</td><td>0.05</td><td>0.300</td><td>0.139</td><td>245.4</td></tr><tr><td>3</td><td>0.05</td><td>0.291</td><td>0.134</td><td>358.6</td></tr><tr><td>5</td><td>0.05</td><td>0.347</td><td>0.176</td><td>569.7</td></tr><tr><td>8</td><td>0.05</td><td>0.342</td><td>0.165</td><td>859.1</td></tr><tr><td>Sweep min</td><td>一</td><td>0.274</td><td>0.133</td><td>27.0</td></tr><tr><td>Sweep max</td><td>一</td><td>0.347</td><td>0.176</td><td>914.3</td></tr></table>

Table A.14: GraphRAG-lite sensitivity sweep over 16 TF-IDF kNN settings. The table shows the threshold-0.05 slice plus the overall sweep range; all tested settings exceed the flat lexical recall baseline on the same weak external proxy.

Quality-aware and matched graph minibenchmark. We also add an OpenAlex-backed bibliometric rerank baseline that uses title-matched cited\_by\_count, recency, and a lightweight paper-role prior, without using MEKG edges. Under the same seven-run weak survey-reference proxy and top-5 per-section budget, the strongest graph-free quality variant is lexical–bibliometric reciprocal-rank fusion (R@5/P@5 0.309/0.194), improving over flat lexical retrieval (0.252/0.132). This result is important because it shows that part of the retrieval gain can be obtained without typed relations; SurveyGraph’s safer claim is therefore not raw recall superiority, but auditability and controllable fusion.

We then combined the strongest graph-free quality baseline with TF-IDF GraphRAG-lite and typed-MEKG retrieval in one equal-budget minibenchmark. Table A.16 uses the same section queries, same top-5 budget, and same weak proxy for all methods. The best weak-proxy method is bibliometric-anchor + TF-IDF fusion (R@5/P@5 0.357/0.200). Full typed MEKG alone has lower weak-proxy recall but much higher typedneighborhood density, while safe fusion recovers part of the recall without abandoning typed structure. This directly motivates our final positioning: simple quality/similarity baselines are strong, and typed MEKG is most useful when used as an auditable structure and fusion signal.

<table><tr><td>Method</td><td>R@5 P@5</td><td>Quality Cites</td></tr><tr><td>Flat lexical</td><td>0.252 0.132</td><td>0.369 6.6</td></tr><tr><td>OA quality only</td><td>0.2100.457</td><td>0.670 115.5</td></tr><tr><td>Top-20 rerank</td><td>0.2800.156</td><td>0.393 11.7</td></tr><tr><td>Flat + quality</td><td>0.2800.150</td><td>0.384 11.1</td></tr><tr><td>Lexical–biblio RRF 0.309 0.194</td><td></td><td>0.452 17.3</td></tr></table>

Table A.15: OpenAlex-backed quality-aware retrieval baseline over the same seven-run weak survey-reference proxy. Quality is the normalized citation/recency/role score among retrieved papers; Cites is mean OpenAlex cited\_by\_count. This is a matched retrieval-stage baseline, not a full reproduction of quality-aware survey generation systems.
<table><tr><td>Method</td><td>R@5 P@5 Align Dens.</td></tr><tr><td>Flat lexical</td><td>0.252 0.1320.5580.134</td></tr><tr><td>Bibliometric RRF0.309 0.1940.5300.132 TF-IDF</td><td></td></tr><tr><td>GraphRAG-lite</td><td>0.3470.1760.4850.140</td></tr><tr><td>SurveyGraph MEKG</td><td>0.153 0.089 0.392 0.228</td></tr><tr><td>Safe fusion MEKG</td><td>0.2610.1340.5280.179</td></tr><tr><td>Biblio + TF-IDF fusion</td><td>0.357 0.2000.4980.134</td></tr><tr><td>Biblio + MEKG fusion</td><td>0.3040.1780.4860.174</td></tr></table>

Table A.16: Seven-run matched quality/graph minibenchmark. All methods use the same archived section queries and top-5 retrieval budget. Align is a lexical section-evidence alignment proxy, and Dens. is typed-MEKG relation density among retrieved papers.

## B.7 Baseline Audit Summary

Together, these checks justify a narrow claim set: SurveyGraph has real archived outputs, fresh lowcost end-to-end completions, same-backend internal generation diagnostics, and a partially replayable comparison story that now includes released cross-framework artifacts plus four sametopic live partial baselines. Full same-topic, samebudget, same-backend cross-system baseline parity remains future work.

<table><tr><td>System</td><td>Status</td><td>Decisive interpretation</td></tr><tr><td>SurveyForge Partial</td><td></td><td>Released artifacts analyzable; live rerun still blocked by miss- ing local database/ and</td></tr><tr><td>SurveyX</td><td></td><td>embedding assets Analyzable Released artifacts available, but no same-environment live</td></tr><tr><td></td><td></td><td>rerun claimed AutoSurvey Analyzable Released artifacts available, but no same-environment live</td></tr><tr><td>SciSage</td><td>Partial</td><td>rerun claimed Same-topic outline-only live baseline succeeds; full-paper</td></tr><tr><td>ARISE</td><td>Partial</td><td>parity still unverified Released artifacts analyzable; code path still expects extra provider/runtime setup for live rerun</td></tr></table>

Table A.17: Updated baseline audit summary after adding released artifact analysis and four same-topic live partial baselines. None of these systems should yet be counted as fully and fairly reproduced under a unified environment.
<table><tr><td>Source</td><td>N Cov.</td><td>Match</td></tr><tr><td>SurveyForge released refs</td><td>10 0.465</td><td>50.0</td></tr><tr><td>Human survey refs</td><td>100.629</td><td>148.8</td></tr><tr><td>SurveyGraph archive overlaps</td><td>150.078</td><td>3.9</td></tr></table>

Table A.18: SurveyBench reference-list probe using the local SurveyForge SurveyBench snapshot. SurveyForge and human rows are true benchmark-format referencelist evaluations. The SurveyGraph rows are only topoverlap checks between existing archived corpora and SurveyBench topics, not end-to-end benchmark runs.

## B.8 SurveyBench Reference-List Probe

We also run a zero-API probe against the local SurveyBench snapshot bundled with the Survey-Forge repository. We reused the public benchmark script’s citation-coverage definition: after date filtering, the score is the fraction of generated reference IDs that appear in the topic’s benchmark reference set. This is still not an end-to-end SurveyGraph benchmark run, because our archived SurveyGraph topics are not the ten SurveyBench topics and the SurveyForge local code path requires missing database/embedding assets for a live matched rerun. It does, however, calibrate what the benchmark-format reference metric looks like on the released SurveyForge and human reference lists, and it quantifies how little of the present SurveyGraph archive can be reused as a proxy.

The released SurveyForge references obtain nontrivial but topic-variable coverage: 0.857 on 3D Gaussian Splatting, 0.598 on Generative Diffusion Models, 0.435 on Retrieval-Augmented Generation for Large Language Models, and 0.296 on LLMbased Multi-Agent. By contrast, the strongest SurveyGraph archive-to-benchmark overlaps are 11 references for LLM-based Multi-Agent from the function-calling archive and 10 references for Evaluation of Large Language Models; these are far too small to support a benchmark claim. We include the result as a reproducible calibration and as a guardrail: future benchmark evaluation must run SurveyGraph on the SurveyBench topics with matched retrieval budgets and reference extraction, rather than treating unrelated archives as benchmark evidence.

To add one small amount of genuinely live benchmark-facing evidence without paying for new long-form generation, we also ran an exact-topic arXiv retrieval probe through the repository’s own retriever over all 10 SurveyBench topics, capping each topic at 50 returned papers and then scoring the retrieved arXiv IDs with the same date-filtered overlap rule. This collection-stage probe is deliberately narrower than an end-to-end survey benchmark: it does not test outlining, writing, or claim support. It nevertheless shows that the repository retriever can recover nontrivial benchmark references from live search alone, reaching 0.245 mean overlap coverage and 5.1 matched references per topic. The strongest topics are Hallucination in Large Language Models (0.500; 6/12 valid targets matched), 3D Gaussian Splatting (0.412; 7/17), and Multimodal Large Language Models (0.357; 5/14), while Graph Neural Networks remains low (0.051; 2/39) and LLM-based Multi-Agent still yields 0/3 matched targets under the benchmark’s date filter. We therefore use this probe only to demonstrate live benchmark-facing retrieval viability, not end-to-end benchmark parity.

To further strengthen benchmark-facing evidence without paying for full survey generation, we ran a second live collection-stage probe using the repository’s EnhancedPaperCollector rather than exact-topic search alone. This probe covers four SurveyBench topics (LLM-based Multi-Agent, Evaluation of Large Language Models, Retrieval-Augmented Generation for Large Language Models, and Multimodal Large Language Models) under a fixed 8-paper collection budget. Mean benchmark-reference coverage reaches 0.521, compared with 0.166 for the exact-topic probe on the same four topics, and all four topics improve. The largest gain is on Retrieval-Augmented Generation for Large Language Models (0.188→1.000), while LLM-based Multi-Agent rises from 0.000 to 0.333. Because this probe still measures only collection-stage overlap, we treat it as stronger retrieval evidence rather than as end-toend benchmark success.

<table><tr><td>Topic</td><td>Budget Enhanced Exact-topic</td><td></td><td></td><td>∆</td></tr><tr><td>LLM-based Multi-Agent</td><td>8</td><td>0.333</td><td>0.000</td><td>+0.333</td></tr><tr><td>Evaluation of LLMs</td><td>8</td><td>0.250</td><td>0.120</td><td>+0.130</td></tr><tr><td>RAG for LLMs</td><td>8</td><td>1.000</td><td>0.188</td><td>+0.812</td></tr><tr><td>Multimodal LLMs</td><td>8</td><td>0.500</td><td>0.357</td><td>+0.143</td></tr><tr><td>Mean</td><td>8</td><td>0.521</td><td>0.166</td><td>+0.355</td></tr></table>

Table A.19: Live four-topic SurveyBench-facing collection-stage probe using the repository EnhancedPaperCollector. Even under a small 8-paper budget, the enhanced collector outperforms exact-topic retrieval on all four matched topics.

## C Additional Low-Cost Diagnostics

To strengthen the evidence without additional longform generation, we ran fourteen extra post-hoc analyses over the already saved artifacts: the original ten local diagnostics, plus an encodersensitivity replay, a graph-safe fusion sweep, a semantic-gate error analysis, and a local cost-stage replay. All fourteen analyses are fully local and reproducible from the current workspace, except for already completed live collection probes explicitly marked as live in Appendix B.

## C.1 Offline Retrieval-Side Graph Diagnostic

We constructed a fixed-budget diagnostic ablation over the eight archived runs. For each nonintroduction/non-conclusion top-level section, we retrieved the top-5 papers with (i) a flat lexical section-to-paper ranker and (ii) a graph-aware ranker that adds category/subcategory overlap, PageRank centrality, and 1-hop relation expansion from the strongest lexical seeds. We then compared the union of retrieved papers under graph-native targets: core-paper recall, community coverage, average PageRank exposure, and relation density among retrieved papers. This experiment is intentionally diagnostic and should not be read as a public benchmark.

An exact paired sign-flip check over the same eight archived run pairs shows that the retrievalside gain is not carried by only one or two outliers. Core-paper recall, average PageRank exposure, and relation density are all positive in 8/8 non-zero pairs, each with one-sided $p = 0 . 0 0 3 9$ and two-sided $p \ = \ 0 . 0 0 7 8$ . Community coverage is directionally positive but weaker: 5 positive pairs, 1 negative pair, and 2 ties, giving one-sided $p = 0 . 0 7 8 1$ and two-sided $p = 0 . 1 5 6 3$ . We report these values as compact diagnostic evidence only, not as benchmark-grade significance claims.

<table><tr><td>Metric</td><td>Flat</td><td>Graph aware</td><td> $\Delta$ </td><td>Graph better</td></tr><tr><td>Core recall</td><td>0.216</td><td>0.373</td><td>+0.157</td><td>8/8</td></tr><tr><td>Comm. cov.</td><td>0.758</td><td>0.863</td><td>+0.104</td><td>5/8</td></tr><tr><td>Avg PageRank</td><td>0.481</td><td>0.630</td><td>+0.149</td><td>8/8</td></tr><tr><td>Relation dens.</td><td>0.162</td><td>0.259</td><td>+0.098</td><td>8/8</td></tr><tr><td>Unique papers</td><td>21.38</td><td>21.50</td><td>+0.13</td><td>4/8</td></tr></table>

Table A.20: Offline retrieval-side diagnostic ablation on the eight archived artifacts. Under the same top-5 section budget, the graph-aware ranker retrieves more central and more mutually connected paper sets on average.

<table><tr><td>Variant</td><td>Core rec.</td><td>Avg PR</td><td>Rel. dens.</td></tr><tr><td>Full graph-aware</td><td>0.373</td><td>0.630</td><td>0.259</td></tr><tr><td>No overlap</td><td>0.373</td><td>0.629</td><td>0.259</td></tr><tr><td>No centrality</td><td>0.364</td><td>0.621</td><td>0.254</td></tr><tr><td>No expansion</td><td>0.219</td><td>0.486</td><td>0.161</td></tr><tr><td>Flat lexical</td><td>0.216</td><td>0.481</td><td>0.162</td></tr></table>

Table A.21: Signal-component ablation inside the archived retrieval diagnostic. Removing 1-hop expansion nearly collapses the graph-aware gains back to the flat lexical baseline, while removing overlap is almost neutral on the current archive slice.

We also ran one small signal-component ablation on the same archived retrieval setup to separate the contributions of category/subcategory overlap, PageRank centrality, and 1-hop relation expansion. The most important result is that 1-hop expansion carries most of the observed retrieval-side lift: removing expansion drops mean core recall from 0.373 to 0.219 and relation density from 0.259 to 0.161, nearly matching the flat lexical baseline (0.216/0.162). By contrast, removing centrality only weakens mean core recall to 0.364 and relation density to 0.254, while removing category/subcategory overlap is nearly neutral on the current archived runs. We therefore treat the present graphaware retrieval gain as primarily an expansion effect, with centrality as a secondary preference signal and overlap as a weak tie-breaker under this archive slice.

<table><tr><td>Budget</td><td>Flat rec.</td><td>Graph rec.</td><td> $\Delta$ </td><td>Wins</td></tr><tr><td>top-3</td><td>0.113</td><td>0.288</td><td>+0.176</td><td>7/8</td></tr><tr><td>top-5</td><td>0.216</td><td>0.373</td><td>+0.157</td><td>8/8</td></tr><tr><td>top-7</td><td>0.288</td><td>0.505</td><td>+0.217</td><td>8/8</td></tr></table>

Table A.22: Retrieval budget sweep over the eight archived runs. The graph-aware advantage is stable across modest top-k budgets, so the top-5 choice in the main text is a conservative middle point rather than a cherry-picked setting.
<table><tr><td>Thr.</td><td>Keep rel.</td><td>Graph rec.</td><td>Rec. ∆</td><td>Graph dens.</td></tr><tr><td>≥0.5</td><td>99.0%</td><td>0.373</td><td>+0.157</td><td>0.259</td></tr><tr><td>≥0.7</td><td>80.7%</td><td>0.336</td><td>+0.120</td><td>0.248</td></tr><tr><td>≥0.9</td><td>39.1%</td><td>0.342</td><td>+0.126</td><td>0.245</td></tr></table>

Table A.23: Confidence-threshold robustness sweep over the eight archived runs under the same top-5 section budget. Even after keeping only higher-confidence relations, graph-aware retrieval remains stronger than the flat baseline on the main retrieval-side diagnostics.

## C.2 Retrieval Budget Sweep

To check whether the graph-aware gain depends on a single retrieval setting, we replayed the same diagnostic at three small budgets: top-3, top-5, and top-7 papers per section. The main pattern is that graph-aware retrieval stays ahead of flat lexical retrieval on core-paper recall and relation density at all three budgets, so the top-5 choice in the main text is best understood as a conservative middle point rather than a cherry-picked setting.

## C.2.1 Confidence-Threshold Robustness Sweep

To test whether the graph-aware gain depends mainly on low-confidence edges, we replayed the same top-5 retrieval diagnostic after pruning the archived MEKG to only relations with confidence at least 0.5, 0.7, or 0.9. The gain weakens mildly under stricter pruning, but it remains positive on both core-paper recall and relation density even at the highest threshold. This makes the current retrieval result more robust than a purely weak-edge explanation would suggest.

## C.2.2 Depth Sweep: 1-Hop vs. 2-Hop

To test whether a deeper traversal helps, we replayed the same retrieval diagnostic with 1-hop and 2-hop expansion. The 2-hop variant does retrieve denser neighborhoods, but it also increases search breadth substantially, lowers unique-paper yield, and slightly weakens community coverage on average. This supports keeping 1-hop as the default while treating 2-hop as an exploratory mode.

<table><tr><td>Metric</td><td>Flat</td><td>1-hop</td><td>2-hop</td></tr><tr><td>Core-paper recall</td><td>0.216</td><td>0.373</td><td>0.388</td></tr><tr><td>Community coverage</td><td>0.758</td><td>0.863</td><td>0.817</td></tr><tr><td>Average PageRank</td><td>0.481</td><td>0.630</td><td>0.659</td></tr><tr><td>Relation density</td><td>0.162</td><td>0.259</td><td>0.309</td></tr><tr><td>Unique papers</td><td>21.38</td><td>21.50</td><td>17.00</td></tr><tr><td>Expanded nodes</td><td></td><td>32.02</td><td>73.78</td></tr><tr><td>Trâversed edges</td><td>一</td><td>243.55</td><td>2358.56</td></tr></table>

Table A.24: 1-hop vs. 2-hop depth sweep over the same eight archived runs. 2-hop expansion adds coverage inside the graph, but it also pulls in many more nodes and edges and reduces the number of unique papers retained under the fixed retrieval budget.
<table><tr><td>Variant</td><td>Core rec.</td><td>Comm. cov.</td><td>Avg PR</td><td>Rel. dens.</td></tr><tr><td>Full graph</td><td>0.373</td><td>0.863</td><td>0.630</td><td>0.259</td></tr><tr><td>No</td><td>0.389</td><td>0.912</td><td>0.633</td><td>0.246</td></tr><tr><td>support Struct.-</td><td>0.389</td><td>0.863</td><td>0.621</td><td>0.256</td></tr><tr><td>only Sem.- only</td><td>0.291</td><td>0.829</td><td>0.545</td><td>0.115</td></tr></table>

Table A.25: Relation-family retrieval ablation over the eight archived runs. Removing only generic supports edges changes the averages little, while a semantic-only graph loses both core-paper recall and relation density.

## C.2.3 Relation-Family Ablation

To isolate which edge families matter most under the same top-5 section budget, we replayed the 1-hop graph-aware ranker with four relation inventories: the full graph, a no-support variant that removes only generic supports edges, a structural-only variant that keeps support/meta relations but removes evolutionary semantic edges, and a semantic-only variant that keeps only extends/improves/applies/refutes.

The result is mixed but informative: dropping generic support edges changes retrieval only slightly, whereas restricting the graph to semantic edges alone degrades both core recall and relation density. This supports the interpretation that the current retrieval gain comes from the combination of structural and semantic links rather than one relation family in isolation.

## C.3 Core-Score Weight Sensitivity Audit

Because the default core-paper score uses fixed hand-tuned weights, we first ran a local sensitivity audit over the same eight archived MEKGs. We replayed the score with four moderate reweightings (PageRank-heavy, citation-heavy, recency-heavy, and degree-heavy) and found that the overlap with the default top-10 remains 9.0–9.75 papers on average across variants. To address the saved artifacts’ missing citation signal, we then queried OpenAlex for 1,060 unique archived titles, accepted title matches with similarity at least 0.86 and compatible publication year, and recomputed the default score with OpenAlex cited\_by\_count as a citation proxy. The mean OpenAlex match rate is 92.7%, all eight runs recover nonzero citation signal, and the enriched top-10 overlaps the storedfield top-10 by 6.00 papers on average (range 4–7), indicating that citations materially adjust but do not completely replace the structural/recency ranking.

<table><tr><td>Variant vs. default top-10</td><td>Mean overlap</td><td>Median</td><td>Range</td></tr><tr><td rowspan="3">PageRank-heavy Citation-heavy Recency-heavy Degree-heavy</td><td>9.50</td><td>9.50</td><td>9-10</td></tr><tr><td>9.75</td><td>10.00</td><td>9-10</td></tr><tr><td>9.75 9.00</td><td>10.00 9.00</td><td>9-10 8-10</td></tr><tr><td rowspan="4">Default core-set size Elbow-selected size Archived citation fields populated</td><td>16.12</td><td></td><td></td></tr><tr><td>17.38</td><td></td><td></td></tr><tr><td>0/8</td><td></td><td></td></tr><tr><td>8/8</td><td></td><td></td></tr><tr><td>OpenAlex non-zero signal OpenAlex top-10 overlap</td><td>6.00</td><td>6.50</td><td>4-7</td></tr></table>

Table A.26: Archived core-score sensitivity and OpenAlex citation-proxy audit. Moderate weight perturbations preserve most of the default top-10 core papers, while OpenAlex title matching recovers a nonzero citation proxy for all eight runs and shows that bibliometric enrichment changes, but does not erase, the archived structural ranking.

We then replayed the survey-reference externalcore proxy under two stronger OpenAlex-backed settings. Keeping the original score weights but replacing dormant local citation fields with OpenAlex cited\_by\_count raises mean top-20 proxy recall from 0.162 to 0.189. A second leave-one-runout coarse-grid calibration over the same proxy reaches 0.289 mean top-20 recall, typically favoring citation-heavy weights (most often 0.7–0.8 on the citation term). The gain is real but not yet decisive: the paired sign test is still weak, and Text-to-SQL Agent collapses under the citation-only optimum selected on the remaining six runs. We therefore treat this replay as evidence that the present score form can be externally calibrated, not as proof that the current scorer is solved.

<table><tr><td>Calibration</td><td>Mean</td><td>Median</td><td>Note</td></tr><tr><td>Archived default score</td><td>0.162</td><td>0.077</td><td>local citation field dormant</td></tr><tr><td>OpenAlex</td><td>0.189</td><td>0.125</td><td>same score form,</td></tr><tr><td>default weights LOO tuned</td><td>0.289</td><td>0.308</td><td>citation proxy repaired coarse PR/cit/rec/deg grid</td></tr></table>

Table A.27: OpenAlex-backed external-core proxy calibration over the seven archived runs with usable survey-reference targets. Replacing dormant local citation fields with OpenAlex cited\_by\_count gives a small default gain, while leave-one-run-out coarse-grid tuning yields a larger but still explicitly proxy-bound improvement.

## C.4 Archived Input / Title-Integrity Audit

Because graph size or relation counts could potentially be inflated by trivial upstream duplication, we ran an additional audit directly over the archived MEKG paper/relation stores. This check is intentionally narrow: it does not claim full scholarly canonicalization, but it asks whether the saved runs exhibit obvious failure modes that should have been caught before any semantic interpretation.

Across the eight archived runs, the saved MEKG inputs contain 1,146 paper records and 18,160 relation records. Using the same lightweight title normalization rule implemented in the repository’s online fetcher (lowercasing, punctuation stripping, and whitespace collapsing before titlehash deduplication), we found zero within-run exact title duplicates and zero within-run normalizedtitle duplicates. We also found zero missing values for the core paper fields we rely on in this paper (id, title, abstract, category, subcategory, year, source), zero missing source/target/type/confidence fields in the saved relation records, and zero dangling relation endpoints whose source or target paper ID is absent from the corresponding archived paper store. Relation confidence values in this archive slice range from 0.4 to 1.0, with a global mean of 0.806.

To clarify why some smaller runs still expose more than 2,300 relations, we also grouped the archived artifacts by corpus size and summarized relation density. The key pattern is that the 50- paper runs are denser in relations per paper, not because of duplicate paper rows, but because compact corpora accumulate many repeated structural/meta links. In particular, the three high-frequency nonsemantic types supports, same\_author, and shares\_architecture already account for roughly 69% of edges in both small and large groups, while semantic edges remain a minority.

<table><tr><td>Run group</td><td></td><td></td><td>Papers Relations Rel./paper Sem. share</td><td></td></tr><tr><td>50-paper runs</td><td>50.0</td><td>2314.67</td><td>46.29</td><td>10.1%</td></tr><tr><td>~200- paper runs</td><td>199.2</td><td>2243.20</td><td>11.26</td><td>12.4%</td></tr></table>

In both groups, supports + same\_author + shares\_architecture account for about 69% of all edges, and structural/meta relation families exceed 87% overall.

Table A.28: Grouped relation-density summary for the archived MEKG artifacts. Smaller 50-paper runs are denser in raw relations per paper, but the density is dominated by repeated structural/meta links rather than by a larger semantic-edge share.
<table><tr><td>Relation</td><td>Count Temp.</td><td>Cue</td><td>Conf.</td></tr><tr><td>extends</td><td>127699.8%24.4%0.673</td><td></td><td></td></tr><tr><td>improves</td><td>36699.7%65.8% 0.675</td><td></td><td></td></tr><tr><td>applies</td><td>44998.0%24.7% 0.620</td><td></td><td></td></tr></table>

Table A.29: Archived semantic-edge consistency audit over 2,091 MEKG edges. High temporal-direction consistency supports the intended evolutionary orientation, while the lower cue-support rates indicate that many saved edges rely on contextual inference rather than only keyword matching.

This audit should be interpreted conservatively. It does not show that all synonymy or cross-run aliasing problems are solved, and it does not validate the semantic correctness of the edges themselves. What it does show is that, within each archived run used in this paper, the graph is not being trivially enlarged by exact duplicate rows, normalized-title duplicates, or broken relation references.

## C.5 Semantic-Edge Consistency Audit

We further audited the semantic edge types most central to MEKG’s evolutionary claims: extends, improves, and applies. For each archived edge, we checked whether the source paper is temporally at least as recent as the target paper, whether the source title/abstract contains an explicit lexical cue compatible with the relation type, and what confidence score was saved in the archived relation record. This remains a consistency audit rather than a human-annotated precision/recall study.

<table><tr><td>Control</td><td>Edge</td><td>Control</td><td> $\Delta$ </td><td>Better</td></tr><tr><td>Matched non-edge</td><td>0.0748</td><td>0.0733</td><td>+0.0015</td><td>1109/2091</td></tr><tr><td>Easy non-edge</td><td>0.0748</td><td>0.0249</td><td>+0.0500</td><td>2079/2091</td></tr></table>

Table A.30: Semantic-edge negative-control audit using lexical overlap. Observed semantic edges are clearly separated from arbitrary low-overlap nonedges, but they are nearly tied with temporally and target-role matched non-edge controls. This negative result supports our conservative interpretation that lexical evidence alone cannot validate extends/improves/applies edges.

## C.5.1 Semantic-Edge Negative-Control Audit

The consistency checks above still do not answer whether the saved semantic edges are distinguishable from plausible non-edges. We therefore added a deterministic negative-control audit over the same 2,091 archived extends/improves/applies edges. For each observed edge, we compared source–target lexical overlap against two controls from the same archived run: a matched non-edge with compatible temporal direction and target-role class, and an intentionally easy non-edge chosen to have minimal lexical overlap with the source. This design uses no new model calls and is deliberately conservative: the matched control is meant to be hard, not a strawman.

At the run level, the easy-control contrast is positive in 8/8 runs (one-sided sign-flip $p = 0 . 0 0 3 9 )$ , while the matched-control contrast is positive in only 5/8 runs (one-sided $p = 0 . 1 7 1 9 )$ . By relation type, extends and improves are slightly above matched controls on average, whereas applies is slightly below matched controls. We therefore do not use this audit to claim semantic precision. Its value is diagnostic: it rules out completely arbitrary low-overlap links, but confirms that a benchmarkgrade edge study needs human labels, citationcontext spans, and negative sampling beyond local lexical overlap.

## C.5.2 Confidence-Stratified Semantic-Edge Profile

To test whether the archived confidence scores align with local textual evidence, we add one more zero-cost profile over the same 2,091 semantic edges. We pruned the archived semantic layer to edges with confidence at least 0.5, 0.7, or 0.9 and recomputed the same consistency statistics. The result is modest but still informative: higher-confidence subsets retain near-perfect temporal direction, and explicit source-cue support rises slightly from 32.0% $\mathrm { a t } \geq 0 . 5$ to 33.2% at $\geq 0 . 7$ before becoming sample-limited $\mathrm { a t } \geq 0 . 9$ (52 edges total). We therefore interpret the saved confidence field as a useful but still imperfect diagnostic priority signal rather than as calibrated probabilistic correctness.

<table><tr><td>Min conf.</td><td>Count</td><td>Temp.</td><td>Cue</td><td>Avg conf.</td></tr><tr><td>≥0.5</td><td>2035</td><td>99.6%</td><td>32.0%</td><td>0.668</td></tr><tr><td>≥0.7</td><td>1078</td><td>99.7%</td><td>33.2%</td><td>0.754</td></tr><tr><td>≥0.9</td><td>52</td><td>100.0%</td><td>30.8%</td><td>0.907</td></tr></table>

Table A.31: Confidence-stratified semantic-edge consistency profile over the archived MEKG. Higherconfidence subsets retain near-perfect temporal direction, while explicit lexical cue support rises only modestly, which supports treating confidence as a diagnostic ranking signal rather than as a calibrated correctness score.
<table><tr><td>Relation</td><td></td><td></td><td>Supported Partial Unsupported Total</td><td></td></tr><tr><td>extends</td><td>5</td><td>6</td><td>13</td><td>24</td></tr><tr><td>improves</td><td>3</td><td>14</td><td>7</td><td>24</td></tr><tr><td>applies</td><td>4</td><td>12</td><td>8</td><td>24</td></tr><tr><td>Overall</td><td>12</td><td>32</td><td>28</td><td>72</td></tr></table>

Table A.32: Stratified repeated-protocol audit over 72 stratified semantic-edge rows. The protocol is deterministic and stricter than the earlier author-side spot audit, so the lower support rate should be read as a more conservative view of semantic-edge reliability rather than as contradictory evidence.

## C.5.3 Repeated-Protocol Semantic-Edge Audit

To add an extra sanity check without running a full human study, we expanded the semantic-edge check into a stratified repeated-protocol audit with 72 rows stratified by relation type and confidence bucket (24 each for extends, improves, and applies; 24 each for low/mid/high confidence). Unlike the earlier 36-row author-side audit, this extension uses a deterministic protocol over archived artifacts: each sampled row is assigned a support tier from temporal direction, source-side lexical cue support, target-role admissibility, and whether the observed edge beats a temporally and role-matched non-edge control in lexical overlap. This remains a zero-API deterministic audit rather than a human precision/recall study, but it is stricter and more reproducible than the earlier spot check.

In this stricter 72-row audit, the supported-orpartial rate is 61.1% overall. By relation type, improves is the strongest at 70.8%, applies reaches 66.7%, and extends is weakest at 45.8%.

The dominant failure modes are missing explicit source cues (18 rows), weak lineage signal under the matched-control contrast (17), lexical overlap not better than the matched control despite otherwise admissible roles/time (14), and targetrole mismatch (10). Only one sampled row is a temporal reversal. By confidence bucket, the low and high buckets each reach 66.7% supported-orpartial, while the mid bucket is weaker at 54.2%, reinforcing our interpretation that the archived confidence field is at best a rough priority signal rather than a calibrated correctness estimate.

To make the protocol less abstract, Table A.33 shows representative passed, partial, and failed cases with snippets extracted from paper metadata fields. These examples are not additional human labels; they illustrate why we treat semantic edges as candidate evidence. Supported cases usually contain source-side method or adaptation cues and beat the matched non-edge control, while failed cases often point from a method paper to a benchmark/dataset target or have only generic topical overlap.

## C.5.4 Generated-Claim Citation-Support and Lexical Anchor Audits

Because grounding ultimately depends on generated survey statements rather than only graph edges, we expanded the earlier two-topic spot check into a deterministic lexical-anchor audit over eight archived SurveyGraph outputs. The expanded audit samples 20 cited body sentences per run (160 sentences total), including both single- and multicitation sentences, and compares each sentence with cited paper titles, abstracts, and cached text. The labels are anchored, partial, or weak/missing; they are lexical grounding triage rather than blinded human citation-faithfulness annotation.

For continuity with earlier checks, we also keep the smaller two-topic author-side spot audit: 24 single-citation body sentences from the archived DocQA and Text-to-SQL artifacts yielded 15 supported, 7 partial, and 2 unsupported cases (91.7% supported-or-partial). The expanded 160-sentence audit is intentionally stricter and broader, so we use the 80.6% lexical anchored-or-partial rate as the safer headline. Together, the audits suggest that many generated cited statements have local evidence anchors, while over-specific implementation or formalization claims remain the dominant risk when retrieved evidence is thin.

We further ran a zero-API semantic-support probe over the same 160 cited sentences using a locally cached bge-small-en-v1.5 encoder. A title-only semantic threshold is too permissive: all 160 sentences are labelled semantic-supportedor-partial when compared only with their cited paper titles. To make the check harder, we add a hard-negative control that compares each sentence against non-cited titles from the same topic pool. Under this control, the cited title beats the hard negative in only 72/160 cases (45.0%), with a mean cited-minus-negative margin of -0.003. This negative result is useful: title-level semantic similarity is not a reliable substitute for citation faithfulness, so the lexical audit should remain a triage tool and future evaluation should use citation-context spans, full abstracts, NLI, or human labels.

Because title-only contrast is weak, we also ran an evidence-level hard-negative probe. For each cited sentence with a recoverable arXiv identifier, we matched the citation key back to the archived paper record and compared the generated sentence against the cited paper’s title, abstract, and first local text span. The hard negative is drawn from non-cited papers in the same archived run. This stronger local-evidence contrast covers 159/160 sampled sentences. Overall, the cited evidence beats the hard negative in 104/159 cases (65.4%), with the anchored lexical group strongest (71.7%) and weak/missing lexical rows near chance (48.4%). This supports the lexical audit’s ranking signal while still falling short of a full faithfulness judgment.

## C.5.5 Section-Reference Alignment Audit

To complement claim-level checks and the brittle graph-internal structural-coherence metric, we added a section-reference alignment diagnostic over the same archived generated surveys. For each top-level generated section, the script aggregates the locally cached evidence behind that section’s citations and computes a conservative lexical-overlap precision. This is not semantic entailment, but it checks whether sections carry local citation evidence instead of being uncited narrative filler.

## C.6 Survey-Reference External-Core Audit

The archived retrieval diagnostic risks circularity because it evaluates core-paper recall against a core set derived from the same MEKG. To partially reduce this coupling without paying for new publicbenchmark runs, we built one weak external-core proxy directly from the archived corpora. For each archived run, we scanned local survey/review paper JSON files, extracted arXiv identifiers mentioned in their reference text, and intersected those identifiers with the same run’s archived paper pool. This yields a survey-reference-backed target set that is not chosen by the MEKG core scorer itself, although it remains an archive-internal proxy rather than a public benchmark.

<table><tr><td>Rel.</td><td>Label</td><td>Source</td><td>Target</td><td>Failure / support mode</td><td>Local source snippet</td></tr><tr><td>extends</td><td>supported</td><td>Adapting While Learning: Grounding LLMs for Scientific Tool Use</td><td>Confucius: Iterative Tool Learning from Introspection Feedback by Easy-to-Difficult Curriculum</td><td>well-supported protocol case</td><td>&quot;Tool Usage Adaptation (TUA)&quot; categorizes problems by model accuracy and trains</td></tr><tr><td>extends</td><td>unsupported</td><td>Adapting While Learning: Grounding LLMs for Scientific Tool Use</td><td>API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs</td><td>target-role mismatch</td><td>on adapted tool-use settings. The source discusses tool-use adaptation, but the target is a benchmark resource rather than</td></tr><tr><td></td><td>improves supported</td><td>LaTr: Layout-Aware Transformer for Scene-Text VQA</td><td>Spatially Aware Multimodal Transformers for TextVQA</td><td>well-supported protocol case</td><td>a precursor method. The source explicitly proposes a layout-aware multimodal architecture for scene-text VQA.</td></tr><tr><td></td><td></td><td>improves unsupported LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding</td><td>DocVQA: A Dataset for VQA on Document Images</td><td>target-role mismatch</td><td>The source describes pre-training objectives, but the target is a dataset paper, so the relation is not a direct method</td></tr><tr><td>applies</td><td>supported</td><td>Advancing SLM Tool-Use Capability using Reinforcement Learning</td><td>Language Models</td><td>MetaTool Benchmark for Large well-supported protocol case</td><td>refinement. The source discusses reinforcement learning for tool-augmented agents in a setting tied to tool-use</td></tr><tr><td>applies</td><td>partial</td><td>CheMatAgent: Enhancing LLMs Toolformer: Language Models for Chemistry and Materials</td><td>Can Teach Themselves to Use Tools</td><td>lexical-overlap not better than matched control</td><td>evaluation. The source is a domain application of tool-use ideas, but the local evidence is not enough to establish a direct transfer path.</td></tr></table>

Table A.33: Representative semantic-edge examples from the snippet audit. The table illustrates both positive and negative protocol decisions and motivates the evidence-gated replay in Table A.41.
<table><tr><td colspan="6">Topic Cand. Samp. Anch. Part. Weak</td></tr><tr><td>LLM function</td><td>242</td><td>20</td><td>11</td><td>7</td><td>2</td></tr><tr><td>calling Doc. visual QA</td><td>232</td><td>20</td><td>13</td><td>4</td><td>3</td></tr><tr><td>Memory in AI</td><td>288</td><td>20</td><td>14</td><td>2</td><td>4</td></tr><tr><td>agents Embodied QA run 1</td><td>175</td><td>20</td><td>11</td><td>3</td><td></td></tr><tr><td>Text-to-SQL agent</td><td>202</td><td>20</td><td>5</td><td>8</td><td>6 7</td></tr><tr><td>Embodied QA run 2</td><td>217</td><td>20</td><td>13</td><td>6</td><td>1</td></tr><tr><td>Approach of</td><td>215</td><td>20</td><td>12</td><td>4</td><td>4</td></tr><tr><td>DocQA</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Agent All</td><td>138 1709</td><td>20 160</td><td>14 93</td><td>2 36</td><td>4 31</td></tr></table>

Table A.34: Expanded claim-to-citation lexical anchor audit over eight archived generated surveys. The overall anchored-or-partial rate is $1 2 9 / 1 6 0 = 8 0 . 6 \%$ with 103 single-citation and 57 multi-citation sentences. Weak/missing-anchor rows mark higher-risk claims for manual review rather than automatic hallucination labels.

Seven archived runs expose at least three such target papers. Across those seven runs, the mean proxy size is 12.43 papers (median 13), with 7.43 local survey/review sources contributing on average. The default hybrid core score recovers only 5.8% of these targets in its top-10 and 16.2% in its top-20. For context, pure recency reaches 17.9% top-20 recall and pure PageRank 17.3%. In other words, this audit does not show external-core strength; rather, it quantifies that the current corepaper scorer still aligns weakly with even a weak survey-reference proxy. We include this negative result because it directly answers the circularity criticism with real local evidence instead of leaving the issue implicit.

<table><tr><td>Lexical label</td><td>N</td><td>Cited sim. Neg. sim. Cited &gt; neg.</td><td></td><td></td></tr><tr><td>Anchored</td><td>93</td><td>0.694</td><td>0.689</td><td>0.527</td></tr><tr><td>Partial anchor</td><td>36</td><td>0.680</td><td>0.686</td><td>0.444</td></tr><tr><td>Weak/missing</td><td>31</td><td>0.651</td><td>0.672</td><td>0.226</td></tr><tr><td>All</td><td>160</td><td>0.682</td><td>0.685</td><td>0.450</td></tr></table>

Table A.35: Citation semantic-support probe with hard negatives. Scores use a locally cached BGE encoder over generated cited sentences and cited/non-cited paper titles. The hard-negative result prevents us from treating title-level semantic similarity as citation faithfulness.
<table><tr><td>Lexical label</td><td>N</td><td></td><td></td><td>Cited ev. Neg. ev. Cited &gt; neg.</td></tr><tr><td>Anchored</td><td>92</td><td>0.220</td><td>0.160</td><td>0.717</td></tr><tr><td>Partial anchor</td><td>36</td><td>0.199</td><td>0.158</td><td>0.639</td></tr><tr><td>Weak/missing</td><td>31</td><td>0.138</td><td>0.138</td><td>0.484</td></tr><tr><td>All</td><td>159</td><td>0.199</td><td>0.155</td><td>0.654</td></tr></table>

Table A.36: Evidence-level citation contrast using archived cited-paper title, abstract, and local text against same-run hard negatives. The contrast is stronger than title-only matching and shows that lexical weak/missing rows are also weaker under evidence-level comparison.

We also replayed the relation-family ablation against this weaker external proxy rather than only against the internal MEKG-derived core set. The result is mixed but informative. On mean top-5 recall, the full graph reaches 0.153, no-support reaches 0.221, structural-only reaches 0.135, and semantic-only reaches 0.266. However, semantic-only retrieval sharply lowers relation density from 0.230 to 0.091 (paired twosided $p = 0 . 0 1 5 6 )$ , while no-support also lowers density slightly but consistently (0.230→0.215; $p = 0 . 0 4 6 9 )$ . We therefore do not claim a single winning family: the weak external proxy suggests that semantic edges contribute much of the targethit signal, whereas structural/meta edges preserve denser and more internally connected retrieved sets.

<table><tr><td>Topic</td><td>Sec.</td><td>Cited Weak</td><td></td><td>Prec. Cites</td></tr><tr><td>LLM function calling</td><td>6</td><td>6</td><td>1</td><td>0.315 20.7</td></tr><tr><td>Doc. visual QA</td><td>5</td><td>5</td><td>0</td><td>0.272 24.4</td></tr><tr><td>Memory in AI</td><td>5</td><td>5</td><td>1</td><td>0.323 25.2</td></tr><tr><td>agents Embodied QA run 1</td><td>5</td><td>5</td><td>1</td><td>0.290 15.8</td></tr><tr><td>Text-to-SQL agent</td><td>5</td><td>5</td><td>1</td><td>0.197 19.8</td></tr><tr><td>Embodied QA run 2</td><td>5</td><td>5</td><td>0</td><td>0.409 22.6</td></tr><tr><td>Approach of</td><td>5</td><td>5</td><td>0</td><td>0.407 16.4</td></tr><tr><td>DocQA Agent</td><td>6</td><td>6</td><td>2 0.239</td><td>12.3</td></tr><tr><td>All</td><td>42</td><td>42</td><td>6</td><td>0.305 19.5</td></tr></table>

Table A.37: Section-reference alignment audit over 42 top-level sections from eight archived generated surveys. All sections have citations; 85.7% are aligned-or-partial under the conservative lexical proxy, and the uncitedsection rate is 0.0%.
<table><tr><td>Metric</td><td>Base@10</td><td>Base@20</td><td>Recency@20 PR@20</td><td></td></tr><tr><td>Mean recall</td><td>0.058</td><td>0.162</td><td>0.179</td><td>0.173</td></tr><tr><td>Runs used</td><td></td><td>7</td><td></td><td></td></tr><tr><td>Mean target size</td><td></td><td>12.43</td><td></td><td></td></tr><tr><td>Median</td><td></td><td></td><td>13.00</td><td></td></tr><tr><td>target size</td><td></td><td></td><td></td><td></td></tr></table>

Table A.38: Survey-reference external-core proxy audit over seven archived runs. The proxy is built from local survey/review references rather than from the MEKG core scorer, so the default recall remains low and makes the current circularity limitation explicit; the OpenAlex calibration above shows that some of this gap is tied to dormant citation fields rather than to graph structure alone.

The extended seven-run replay in Table A.40 folds the relation-family ablation and evidence gates into a single external-proxy diagnostic. It gives a more conservative picture than the fourtopic matched gate below: supported/partial gating removes 102.6 semantic edges per run on average and slightly raises precision, but it does not reproduce the four-topic recall gain. This is why the main text treats semantic gating as useful pruning rather than as a universal recall improver.

<table><tr><td>Variant</td><td>Recall@5</td><td>Prec.@5</td><td>Rel. dens. Active rel.</td><td></td></tr><tr><td>Full graph</td><td>0.153</td><td>0.088</td><td>0.230</td><td>2265.1</td></tr><tr><td>No</td><td>0.221</td><td>0.108</td><td>0.215</td><td>1690.3</td></tr><tr><td>supports Structural-</td><td>0.135</td><td>0.074</td><td>0.221</td><td>2014.0</td></tr><tr><td>only Semantic- only</td><td>0.266</td><td>0.139</td><td>0.091</td><td>251.1</td></tr></table>

Table A.39: Relation-family replay against the weaker survey-reference external proxy. Semantic-only retrieval gives the highest weak-target hit rate, but it also sharply reduces relation density and the size of the active neighborhood, which supports a mixed-family interpretation of the current retrieval scaffold.
<table><tr><td>Mode</td><td>R@5 P@5 Dens. Drop</td></tr><tr><td>Flat lexical</td><td>0.252 0.1320.134 0.0</td></tr><tr><td>Structural-only</td><td>0.1350.0740.221 0.0</td></tr><tr><td>Semantic-only</td><td>0.266 0.1390.091 0.0</td></tr><tr><td>No supports</td><td>0.221 0.1080.215 0.0</td></tr><tr><td>Full graph</td><td>0.153 0.0880.230 0.0</td></tr><tr><td>Supp./partial gate</td><td>0.151 0.089 0.224 102.6</td></tr><tr><td>Supported-only gate</td><td>0.145 0.081 0.224 209.4</td></tr></table>

Table A.40: Extended seven-run external-proxy relationfamily and semantic-gate replay. Semantic-only retrieval gives the highest weak-target recall but the lowest relation density; supported/partial gating removes noisy semantic candidates while preserving weak-proxy behavior within small margins of the full graph.

## C.7 Semantic Evidence-Gated Retrieval Replay on Four Matched Topics

The repeated-protocol audit above gives labels but does not by itself show whether those labels are useful operationally. We therefore reran the four-topic matched retrieval-stage comparison after converting the deterministic protocol labels into a cheap evidence gate. Non-semantic edges are kept; semantic edges are either kept only when labelled supported-or-partial, kept only when labelled supported, or dropped entirely in the structural-only variant. This replay uses the same archived outlines, same top-5 per-section budget, and same survey-reference external proxy as the matched simpler-graph baseline.

The gate result is directionally useful but deliberately modest. It does not prove that the deterministic protocol is a calibrated semantic-edge verifier. Instead, it shows that the failure modes identified by the audit are not merely descriptive: filtering edges that fail the protocol can improve the same retrieval-stage proxy. This strengthens the paper’s narrower claim that MEKG relations are useful when treated as ranked, auditable candidate evidence rather than as unquestioned lineage facts.

<table><tr><td>Mode</td><td>Rec.</td><td>Prec.</td><td>Dens. Drop</td></tr><tr><td>Full</td><td>0.164</td><td>0.105</td><td>0.165 0.0</td></tr><tr><td>Supp./partial gate</td><td>0.182</td><td>0.117</td><td>0.166 103.0</td></tr><tr><td>Supp.-only gate</td><td>0.172</td><td>0.103</td><td>0.166 207.2</td></tr><tr><td>Structural</td><td>0.154</td><td>0.090</td><td>0.163</td></tr></table>

Table A.41: Semantic evidence-gated retrieval replay over four matched topics. In this four-topic matched slice, the supported/partial gate drops an average of 103.0 candidate semantic edges per topic and improves the weak external-proxy recall and precision while preserving relation density; Table A.40 reports the more conservative seven-run replay. This supports using semantic-edge audits as a lightweight verification gate, but the four-topic sample remains too small for a benchmark claim.

We also applied the same deterministic protocol to all 2,091 archived semantic candidate edges, not only to the stratified 72-row sample. This fullprotocol replay is still automatic, but it clarifies which errors the gate is actually suppressing at scale. Overall, 338 edges are labelled supported, 891 partial, and 862 unsupported, for a 58.8% supported-or-partial rate. The weakest relation is extends: it has 1,276 candidates, only 52.4% supported-or-partial, and most unsupported cases are weak-lineage or target-role errors. improves is stronger (80.6% supported-or-partial), while applies remains intermediate (59.0%). Among unsupported cases, weak lineage signal accounts for 616 rows, target-role mismatch for 234, and temporal reversal for 12. This explains why the supported/partial retrieval gate improves the matched proxy: it removes a large set of noisy semantic candidates while leaving non-semantic structure untouched. In a fixed top-5 ranking, removing a high-ranked unsupported candidate can increase recall by freeing a slot for a proxy-target paper; this occurs on the memory topic, while recall is exactly tied on the other three topics.

The released annotation artifacts separate two files: (i) a reusable blinded packet with blank label columns and guidelines, and (ii) the filled adjudicated-label file used for the reported 108- edge human validation. Table A.43 reports the filled adjudicated labels, not the deterministic reference labels; the blank packet is included only to support independent reuse of the annotation protocol.

<table><tr><td>Relation</td><td></td><td></td><td></td><td>Total Supp. Partial Unsupp.</td><td>S/P</td></tr><tr><td>extends</td><td>1276</td><td>159</td><td>510</td><td>607</td><td>0.524</td></tr><tr><td>impr.</td><td>366</td><td>118</td><td>177</td><td>71</td><td>0.806</td></tr><tr><td>applies</td><td>449</td><td>61</td><td>204</td><td>184</td><td>0.590</td></tr><tr><td>All</td><td>2091</td><td>338</td><td>891</td><td>862</td><td>0.588</td></tr></table>

Table A.42: Full deterministic semantic-gate replay over all archived semantic candidate edges. The protocol is automatic rather than human annotation, but it identifies the same brittleness identified by the audits: extends is the noisiest semantic relation, whereas improves has stronger local evidence.
<table><tr><td>Relation</td><td>N</td><td></td><td></td><td>Supp. Partial Unsupp.</td><td>S/P</td></tr><tr><td>extends</td><td>36</td><td>6</td><td>13</td><td>17</td><td>0.528</td></tr><tr><td>improves</td><td>36</td><td>20</td><td>9</td><td>7</td><td>0.806</td></tr><tr><td>applies</td><td>36</td><td>18</td><td>8</td><td>10</td><td>0.722</td></tr><tr><td>All</td><td>108</td><td>44</td><td>30</td><td>34</td><td>0.685</td></tr></table>

Table A.43: Adjudicated human labels for the 108- edge semantic-edge validation packet. S/P denotes supported-or-partial. Binary support agreement is Cohen’s κ=0.64; three-way labels are used mainly for failure-mode analysis.

## C.8 Encoder-Sensitivity and Graph-Safe Retrieval Fusion

To test encoder sensitivity, we replayed the seven-run survey-reference external proxy with five base rankers: the repository’s existing lexical scorer, word TF-IDF, character TF-IDF, locally cached all-MiniLM-L6-v2, and locally cached bge-small-en-v1.5. For each base ranker, we evaluated (i) the base ranker alone and (ii) the same 1-hop graph expansion used by the archived diagnostic. This is a retrieval-stage sensitivity replay only; it does not regenerate surveys or tune prompts.

Because this replay revealed an over-expansion risk, we ran a second sweep that preserves the top base-ranked anchors before filling the rest of the top-5 budget with graph-neighbor candidates. Table A.45 reports the best high-density setting whose recall stays within 95% of the base-only recall. The result is useful but nuanced. Anchorpreserving graph fusion is helpful for the existing lexical scorer and BGE, and less useful for the two TF-IDF variants where the base ranker already dominates this weak proxy. This gives a concrete default for future retrieval: graph expansion should be anchor-preserving or budget-aware, not an unconstrained replacement for the base ranker.

<table><tr><td>Ranker</td><td>Base R</td><td>Graph R Base D</td><td>Graph D</td></tr><tr><td>Existing lexical</td><td>0.252</td><td>0.153 0.134</td><td>0.230</td></tr><tr><td>TF-IDF word</td><td>0.435</td><td>0.121 0.134</td><td>0.267</td></tr><tr><td>TF-IDF char</td><td>0.459</td><td>0.120 0.133</td><td>0.292</td></tr><tr><td>MiniLM</td><td>0.206</td><td>0.129 0.138</td><td>0.313</td></tr><tr><td>BGE-small</td><td>0.247</td><td>0.160 0.133</td><td>0.293</td></tr></table>

Table A.44: Seven-run encoder-sensitivity replay against the weak survey-reference external proxy. Unconstrained 1-hop graph expansion consistently increases relation density, but it can lower proxy recall by over-expanding away from strong lexical or dense anchors. This is a cautionary result, not a failure to report.
<table><tr><td>Base ranker</td><td>Safe mode</td><td>R</td><td>Dens.</td><td>Base R</td></tr><tr><td>BGE-small</td><td>preserve 1</td><td>0.245</td><td>0.258</td><td>0.247</td></tr><tr><td>MiniLM</td><td>preserve 4</td><td>0.212</td><td>0.173</td><td>0.206</td></tr><tr><td>Existing lexical</td><td>preserve 3</td><td>0.261</td><td>0.181</td><td>0.252</td></tr><tr><td>TF-IDF char</td><td>base only</td><td>0.459</td><td>0.133</td><td>0.459</td></tr><tr><td>TF-IDF word</td><td>base only</td><td>0.435</td><td>0.134</td><td>0.435</td></tr></table>

Table A.45: Anchor-preserving graph-fusion sweep. The reported mode is the highest-density setting whose recall remains at least 95% of the base-only recall. For BGE and the existing lexical ranker, safe graph fusion raises relation density substantially without materially reducing the weak external-proxy recall.

## C.9 Semantic Target-Role Pruning Audit

The semantic audit highlights a concrete failure mode: some extends/improves edges appear to point to surveys, benchmarks, or datasets rather than to precursor methods. To test whether this issue materially affects the retrieval-side story, we ran one simple heuristic replay over the eight archived runs, reported in the released audit package and local diagnostics. We pruned only extends and improves edges whose target titles match lightweight survey/benchmark/- dataset regex cues (e.g., survey, review, benchmark, dataset), kept all other relation families unchanged, and reran the same 1-hop graph-aware retrieval diagnostic.

This heuristic removes 10.0% of semantic edges on average across runs. By relation type, it prunes 12.5% of extends edges (160/1276), 4.4% of improves edges (16/366), and none of the applies edges. Importantly, the replay does not hurt the archived retrieval-side picture on average: mean graph core-paper recall rises from 0.373 to 0.395, and mean relation density rises from 0.259 to 0.263. The effect is modest and not yet statistically persuasive under exact sign-flip counting (core-recall deltas: 2 positive, 1 negative, 5 ties), so we do not present this as a full repair. Instead, it is evidence that the target-role mismatch identified above is real, measurable, and partially suppressible with even a very cheap heuristic.

<table><tr><td>Metric</td><td>Original</td><td>Pruned</td><td>∆</td></tr><tr><td>Core-paper recall</td><td>0.373</td><td>0.395</td><td>+0.022</td></tr><tr><td>Relation density</td><td>0.259</td><td>0.263</td><td>+0.003</td></tr><tr><td>Pruned semantic-edge share</td><td></td><td>0.100</td><td></td></tr><tr><td>Runs used</td><td></td><td>8</td><td></td></tr></table>

Table A.46: Semantic target-role pruning replay over the eight archived runs. Pruning likely survey/benchmark/- dataset targets from extends/improves removes about one tenth of semantic edges while leaving the retrieval-side diagnostic slightly better on average.

## C.10 Runtime and Token/Cost Traces

Finally, we aggregated the saved stage timers and token monitors. These traces answer a narrower reproducibility question than a full efficiency benchmark: how long do the currently inspectable runs take, which stage dominates the latency, and what token/cost traces are exposed for the fresh lowcost reruns? The archived full runs have a mean recorded stage sum of 118.91 minutes (median 123.17; range 28.10–262.69), with content generation contributing 92.5% of stage time on average. For the fresh low-cost smoke reruns, the saved token monitors expose provider-side token and cost estimates under the respective run-time pricing configurations. Table A.48 additionally normalizes the archived traces by paper count. We report these numbers mainly to contextualize why the diagnostic additions above were kept fully local and post hoc rather than expanded into another long-form generation campaign.

Before the stage-level timing profile, Table A.49 reports a variant-level quality/cost replay over the seven external-proxy runs. The local timings include artifact loading/context construction, graph preparation when applicable, and top-5 retrieval; they exclude LLM relation induction, PDF processing, and long-form generation. Thus the table is a diagnostic overhead check, not a full pipeline cost model.

To better isolate the cost of the new diagnostic replays, we also profiled a local replay of graph preparation, deterministic semantic-gate filtering, and graph retrieval over the same eight archived runs. This profile reloads archived MEKGs, constructs the graph, recomputes PageRank, applies the semantic gate, and replays top-5 retrieval; it does not re-run LLM relation induction or longform generation. The local replay takes 0.148 seconds on average, split almost evenly between graph preparation and retrieval, with deterministic semantic gating adding only 0.010 seconds. In contrast, the archived content generation/reflection stage averages 110.02 minutes. This profile supports the narrow operational point that post-hoc diagnostics and evidence gating are cheap once artifacts exist, while full generation and relation induction remain the expensive components.

<table><tr><td>Run / signal</td><td>Runtime</td><td>Token trace</td><td>Cost / note</td></tr><tr><td>Archived full mean</td><td>118.91 min</td><td></td><td>mixed histori- cal back-</td></tr><tr><td>GLM smoke</td><td>55.22 min</td><td>525,681 in / 124,106 out</td><td>ends cost 0.1533</td></tr><tr><td>Qwen smoke</td><td>50.47 min</td><td>459,211 in / 137,744 out</td><td>cost 0.1515</td></tr><tr><td>SciSage</td><td>132.07 s</td><td></td><td>live</td></tr><tr><td>partial</td><td></td><td></td><td>outline- only base- line</td></tr></table>

Table A.47: Compact runtime and token/cost trace summary. The fresh rerun cost values are the provider-side totals recorded by the saved monitors under their original pricing configuration; we do not use them for crossbackend cost ranking.
<table><tr><td>Group</td><td>Runs</td><td>Min</td><td>Min/100</td><td>Tok./100</td></tr><tr><td>50-paper</td><td>3</td><td>64.8</td><td>129.6</td><td>5.02M</td></tr><tr><td>196-200-paper</td><td>5</td><td>151.4</td><td>76.0</td><td>3.31M</td></tr><tr><td>All</td><td>8</td><td>118.9</td><td>96.1</td><td>3.95M</td></tr></table>

Table A.48: Per-100-paper normalization from archived runtime and cost traces. These historical logs time full generation/reflection rather than isolating graph construction alone; the nonlinearity between 50-paper and 200-paper groups reflects fixed generation overheads and different backend settings.

## D Metric Design Notes

Graph dependence. WCR and CC are computed relative to the constructed MEKG. They measure how well the generation pipeline leverages the graph structure and are complemented by external-facing metrics (citation grounding, section alignment) in the main evaluation.

Structural coherence. SC relies on matching section topics to graph communities via lexical overlap. Across runs, SC ranges from 0.00 to 0.50, reflecting variation in topic–community alignment. We report it as one of several diagnostic signals rather than as a standalone quality indicator.

<table><tr><td>Variant</td><td>R@5 P@5 Dens. Sec.</td></tr><tr><td>Flat lexical</td><td>0.252 0.1320.1340.166</td></tr><tr><td>Structural-only</td><td>0.135 0.074 0.221 0.177</td></tr><tr><td>Full graph</td><td>0.153 0.0880.2300.183</td></tr><tr><td>Supp./partial gate</td><td>0.151 0.0890.2240.177</td></tr><tr><td>Supported-only</td><td>0.145 0.081 0.2240.179</td></tr><tr><td>gate</td><td></td></tr></table>

Table A.49: Retrieval-stage quality/cost tradeoff over seven survey-reference external-proxy runs. Graph variants increase relation density but do not dominate flat retrieval on this weak proxy; all local replays complete in roughly 0.17–0.18 seconds once artifacts exist.
<table><tr><td>Replay stage</td><td>Mean sec.</td><td>Local share</td></tr><tr><td>Graph construction + PageRank</td><td>0.066</td><td>44.7%</td></tr><tr><td>Semantic evidence gate</td><td>0.010</td><td>6.5%</td></tr><tr><td>Graph retrieval replay</td><td>0.072</td><td>48.7%</td></tr><tr><td>Local diagnostic replay total</td><td>0.148</td><td>100.0%</td></tr><tr><td>Archived generation/reflection</td><td>6601.2</td><td></td></tr></table>

Table A.50: Local cost-stage replay over eight archived runs. The replay isolates artifact-local graph preparation, semantic gating, and retrieval overhead; it should not be read as the cost of re-inducing all semantic relations with LLM calls.

Citation grounding. The citation grounding audits (Section 5) use deterministic lexical matching and evidence-level contrasts. They provide reproducible triage of citation support quality and are designed to complement future NLI-based or human evaluation.

Internal QA score. The aggregate overall score combines multiple dimensions for internal run comparison. In the main text, we report controlled retrieval benchmarks and grounding audits as primary evidence.

## E Current Evaluation Implementation

The current repository implementation aggregates five dimensions: reference quality, structural soundness, content integrity, ethics/integrity, and rich media quality. When LLM-backed evaluation is enabled, the code uses a weighted ensemble over four heterogeneous judge models (MiniMax-M2.1, Kimi-K2-Thinking, Gemini-3.0- Flash-Preview, and GLM-4.7). These weights are engineering heuristics encoded in the current implementation rather than human-calibrated benchmark

<table><tr><td>Component</td><td>Current default / note</td></tr><tr><td>Core-paper score</td><td> $0 . 4 \cdot \mathrm { P R } + 0 . 3 \cdot \log ( 1 + C ) + 0 . 2 \cdot R +$  0.1 · D; weights are fixed hand-tuned defaults, not learned parameters. The stored artifact records lack populated citation fields, but the OpenAlex title-matched replay above provides an external citation proxy for the archived</td></tr><tr><td>Embeddings</td><td>preserve this enrichment directly. Title + abstract text; HuggingFace-based sentence embeddings with L2 normalization; the repository falls back to a local sentence-transformer-style model when</td></tr><tr><td>Clustering</td><td>the preferred remote embedding is unavailable. HDBSCAN with min_cluster_size = 3and min_samples = 2, adaptively raised as sample count grows; fallback to KMeans when HDBSCAN is</td></tr><tr><td>KMeans fallback</td><td>unavailable (McInnes et al., 2017; Reimers and Gurevych, 2019).  $\mathtt { n \_ c l u s t e r s } =$  min(configured,  $\lfloor n / 3 \rfloor )$  2  $\begin{array} { r } { \mathfrak { n } _ { - } \mathrm { i } \mathrm { i } \mathrm { n i t } = 1 0 , \mathrm { m a x } _ { - } \mathrm { i } \mathrm { t } \in \Sigma = 3 0 0 . } \end{array}$ </td></tr><tr><td>Graph-RAG ranker</td><td>Lexical section-to-paper score + category/subcategory overlap + PageRank + 1-hop relation expansion from the top lexical seeds; semantic</td></tr><tr><td>SC definition</td><td>relations receive a larger expansion bonus than structural ones. Fraction of graph communities whose names match at least one section title in the outline; section headings are matched as plain strings, which explains brittleness.</td></tr></table>

Table A.51: Implementation defaults that govern the reported diagnostics. These are engineering settings from the repository, not tuned research hyperparameters.

weights.

This design is useful for internal QA because it surfaces multiple failure modes at once: missing references, weak section structure, unsupported analysis, poor safety discussion, or low-quality figures/tables. In the main text, we complement these internal signals with controlled retrieval benchmarks, citation grounding audits, and semanticedge validation that provide external-facing evaluation evidence.

## F Implementation Details for Reproducibility

Table A.51 collects the implementation defaults that are easy to miss in prose but matter for reproducing the reported diagnostics.

<table><tr><td>Stage</td><td>Model</td><td>Context Tokens/run</td></tr><tr><td>Paper collection</td><td></td><td>0</td></tr><tr><td>Relation inference</td><td>GLM-4-Flash 128K</td><td>1.82M</td></tr><tr><td>Core-paper scoring</td><td>– (rule-based)</td><td>0</td></tr><tr><td>Outline generation</td><td>GLM-4-Flash 128K</td><td>0.31M</td></tr><tr><td>Section retrieval</td><td>− (local)</td><td>0</td></tr><tr><td>Narrative synthesis</td><td>GLM-4-Flash 128K</td><td>0.94M</td></tr><tr><td>Evaluation/QA</td><td>GLM-4-Flash 128K</td><td>0.24M</td></tr><tr><td>Total</td><td></td><td>3.31M</td></tr></table>

Table A.52: LLM backend and per-run token consumption. Relation inference dominates at 55% of total tokens. All stages use GLM-4-Flash (free-tier API) for the archived runs; the pipeline is backend-agnostic and supports GPT-4o, Claude, Qwen, and DeepSeek via unified API routing. Mean wall-clock time per full run is 151.4 minutes for 196–200 paper corpora.

## G LLM Backend and Cost Statistics

Table A.52 summarizes the LLM backends, context windows, and aggregate cost statistics for the main pipeline stages across the eight evaluation runs. The cost breakdown confirms that relation inference is the dominant expense (55% of tokens), followed by narrative synthesis (28%) and outline generation (9%). Because the pipeline uses a free-tier API (GLM-4-Flash) for the reported experiments, the direct monetary cost is zero; equivalent runs on commercial APIs (GPT-4o-mini at \$0.15/1M input tokens) would cost approximately \$0.50 per full survey generation. The MEKG construction cost is amortized across downstream stages: once built, the graph supports unlimited retrieval queries and outline variants without additional LLM calls.

## H SurveyBench-Facing Collection-Stage Probe

Table A.53 reports the benchmark-facing collection-stage probe used in the main text. This is deliberately not a full SurveyBench leaderboard result: it measures reference-list overlap after paper collection only and does not evaluate outlining, long-form writing, or claim-level support.

## I ROC-Style Evidence Gate Analysis

Figure A.1 shows the precision–recall–F1 tradeoff as the confidence threshold for edge inclusion varies. The analysis uses the 108 annotated edges with human-adjudicated labels as ground truth.

The per-relation-type analysis reveals that improves and applies edges are wellseparated from noise (F1>0.88 at their respective optimal thresholds), while extends requires a higher threshold (0.80) for acceptable precision (F1=0.59). This is consistent with the observation that implicit methodological lineage is harder to validate from surface features.

<table><tr><td>Topic</td><td></td><td>Budget Enhanced</td><td>Exact-topic</td><td>∆</td></tr><tr><td>LLM-based Multi-Agent</td><td>8</td><td>.333</td><td>.000</td><td>+.333</td></tr><tr><td>Evaluation of LLMs</td><td>8</td><td>.250</td><td>.120</td><td>+.130</td></tr><tr><td>RAG for LLMs</td><td>8</td><td>1.000</td><td>.188</td><td>+.812</td></tr><tr><td>Multimodal LLMs</td><td>8</td><td>.500</td><td>.357</td><td>+.143</td></tr><tr><td>Mean</td><td>8</td><td>.521</td><td>.166</td><td>+.355</td></tr></table>

Table A.53: Live four-topic SurveyBench-facing collection-stage probe using the repository’s enhanced collector under a fixed eight-paper budget. The metric is date-filtered reference-list overlap with the local Survey-Bench snapshot; it should not be read as full end-to-end survey-generation quality.  
![](images/d39bcdf543b6afd75589c0e6c2866c63749f8e20cd384177aa53b4b7ae484161.jpg)

![](images/271f19670db97a373633cf58a29e37ef5b803981eeefd2ad86c64d13fb9205ec.jpg)  
Figure A.1: Left: ROC curve for the evidence gate (AUC=0.75). Right: Precision, recall, and F1 as a function of the confidence threshold. The optimal threshold (0.35) retains 74% of edges while achieving F1=0.81.

## J MVSS and StructSurvey Comparison Boundaries

Table A.54 records how we use concurrent systems in this paper. We keep these comparisons for positioning and design context, but we do not report them as a same-topic, same-budget, same-backend live rerun or as evidence of statistically significant end-to-end superiority. The Qwen diagnostics in Appendix B.3 are same-backend controls among SurveyGraph variants only; they do not remove backend confounds against MVSS or StructSurvey.

Published-results context. Prior structure- and coverage-oriented systems, including StructSurvey and SurveyForge, report that structural or coverage-aware retrieval can help over flat baselines (Pedinotti and Santus, 2026; Yan et al., 2025). SurveyX likewise emphasizes citation-quality evaluation (Liang et al., 2025). These works motivate our typed design, but their corpora and evaluation protocols differ from ours; we therefore treat them as related-work context and do not merge their metrics with SurveyGraph’s diagnostics.

<table><tr><td>System</td><td>Available evidence</td><td>Main limitation</td><td>Use in this paper</td></tr><tr><td>MVSS</td><td>Published and released structure-first</td><td>Different backend and evaluation setting; no live survey-generation same-backend</td><td>Closest multi-view structural baseline; motivates typed semantic</td></tr><tr><td>StructSurvey</td><td>results Published and released structured</td><td>rerun Retrieval and generation setup differs from</td><td>diagnostics Complementary retrieval design; could benefit from</td></tr><tr><td>SurveyForge</td><td>retrieval results Local SurveyBench</td><td>SurveyGraph Reference-list probe is not full</td><td>typed MEKG edges Benchmark calibration and</td></tr><tr><td>SciSage</td><td>reference lists and SurveyGraph released artifacts Four same-topic outline-only live partial runs</td><td>generation Outline-only, not full-paper parity</td><td>collection-stage comparison Structural sanity check for outline richness</td></tr></table>

Table A.54: Boundary conditions for concurrent-system comparisons. These rows are used to contextualize SurveyGraph’s design, not as a same-backend end-to-end benchmark or significance test.

## K Prompts Used

We retain two representative prompt templates because they are directly tied to the current implementation. They are included here as implementation evidence rather than as proof of prompt-level optimization.

MEKG Relation Inference Prompt   
You are an expert researcher. Analyze the relationship   
between two research papers based on their details.   
Paper A (Source):   
- Title: {title\_a}   
- Method: {method\_a}   
- Innovation: {innovation\_a}   
- Abstract: {abstract\_a}   
Paper B (Target):   
- Title: {title\_b}   
- Method: {method\_b}   
- Innovation: {innovation\_b}   
- Abstract: {abstract\_b}   
Determine the semantic relationship from Paper A to Paper   
B and return a structured record with: relation type, confi  
dence, short rationale, and whether the relation should be   
none if the evidence only shows topical similarity. Can  
didate labels include extends, improves, applies,   
compares, supports, refutes, and none.

For the semantic-edge audits, we used the following labeling rule over saved candidate edges: supported requires direct abstract/title evidence for the specific relation; partial means compatible topic, method, or downstream-use evidence but no explicit lineage statement; unsupported covers temporal reversal, target-role mismatch, cross-task mismatch, or a claim stronger than the local evidence. The negative-control audit additionally samples same-run non-edge paper pairs so that none-style cases are explicitly represented in the evidence rather than only in the prompt.

<table><tr><td>MEKG-Driven Outline Generation Prompt</td></tr><tr><td>You are a senior academic survey expert. Generate a hier- archical survey outline grounded in the provided MEKG structure. ## Major Research Communities {communities} ## Core Papers {core_papers} ## Research Evolution Timeline</td></tr></table>

## L Generated Survey Examples

We present excerpts from three surveys generated end-to-end by SurveyGraph on distinct research topics. Each row shows three consecutive pages (title page, introduction, and body) to illustrate the system’s ability to produce coherent multi-page academic documents with structured abstracts, logical section flow, and domain-appropriate content organization. The excerpts are shown in a compact form for readability; complete surveys and generation artifacts are available in the public repository listed above.

Survey 1: LLM Function Calling. A 40-page survey covering tool-use paradigms, API integration patterns, and evaluation frameworks for function-calling capabilities in large language models.

Survey 2: Document Visual Question Answering. A 42-page survey spanning OCR-based methods, multimodal transformers, and benchmark datasets for visual document understanding.

Survey 3: Memory in AI Agents. A 48-page survey covering memory architectures, retrievalaugmented generation, and long-term knowledge management in autonomous agent systems.

![](images/93c97ec862b313c3452f10d33868aaf6262750f205f1f5c4937260164bf60b82.jpg)  
Figure A.2: Generated survey on LLM function calling: title page (left), introduction (center), and body text (right).

![](images/8605bd8020f9581e9b7cab013325f141f503c3a8e3350824998755e22940f65d.jpg)

![](images/f4373780945629c7ae2a60909d0bdf11efae515bd3f93f84c6d2b5aaedc31726.jpg)  
Figure A.3: Generated survey on document visual QA: title page (left), introduction (center), and body text (right).

![](images/a09566a8e2fdb13371466d5b3899e06bc42f423b85380d171a98229871118e5c.jpg)  
Figure A.4: Generated survey on memory in AI agents: title page (left), introduction (center), and body text (right).