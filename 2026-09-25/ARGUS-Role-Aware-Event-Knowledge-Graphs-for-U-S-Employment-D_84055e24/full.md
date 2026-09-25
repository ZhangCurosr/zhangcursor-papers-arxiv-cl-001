# ARGUS: Role-Aware Event Knowledge Graphs for U.S. Employment-Discrimination Complaints

Sriram Kannan<sup>1</sup> Swetha Saseendran<sup>1</sup> Vishnu Vardhan Reddy Kandi<sup>1</sup>

Leslie Barrett<sup>2</sup> Madhavan Seshadri<sup>2</sup> Enrico Santus<sup>2</sup>

<sup>1</sup>University of Massachusetts Amherst

<sup>2</sup>Bloomberg

{sriramkannan, ssaseendran, vkandi}@umass.edu {lbarrett4, mseshadri, esantus}@bloomberg.net

## Abstract

U.S. employment-discrimination complaints describe complex event sequences that are not explicitly captured by lexical or embeddingbased representations alone. We present AR-GUS, a source-grounded pipeline that combines a 5W1H-inspired schema, legal-domain models, and LLM-based structured generation to construct document-level Event Knowledge Graphs (EKGs) from CourtListener complaints. ARGUS extracts fact-bearing statements, builds chunk-level event graphs with participant, temporal, and causal structure, and merges them into document-level representations. We evaluate graph quality through human and multi-model assessment and test downstream utility on claim classification and legal QA. The graph-structured classifier outperforms raw and linearized baselines on the held-out set, and EKG-only retrieval improves document-scoped QA, while open-retrieval gains remain limited by low first-stage candidate recall. These results suggest that EKGs are most useful for organizing and reasoning over evidence once relevant material has been retrieved.

## 1 Introduction

Legal practitioners often need to interpret and compare cases whose underlying events are similar even when described differently. Lexical and embedding-based retrieval methods can identify semantically relevant passages, but return largely unstructured text and do not explicitly represent who did what to whom, when, and why. This structure is important in legal narratives, where participant roles and temporal or causal relations can determine whether two cases are meaningfully comparable.

We study this problem in U.S. employmentdiscrimination complaints. We present ARGUS, a pipeline that converts complaint narratives into source-grounded Event Knowledge Graphs (EKGs). Rather than representing a case only as text or a set of entities, ARGUS models events together with their participants, legal roles, source evidence, and supported temporal and causal relations.

![](images/f89dfb6bb42f60d48782f38ee991d0c8fa52908f71e28a1120d4efa291ffa638.jpg)  
Figure 1: Multi-stage pipeline for EKG construction from federal employment-discrimination complaint narratives.

ARGUS follows a staged construction process. Starting from CourtListener complaints,<sup>1</sup> it identifies fact-bearing statements, groups them into semantic chunks, extracts chunk-level event graphs using a 5W1H-inspired representation (Hamborg et al., 2019), and merges them into a documentlevel EKG through deterministic and LLM-assisted cross-chunk resolution. Figure 1 provides an overview; implementation and evaluation details are given in Section 3 and the Appendix.

We investigate three questions: (i) Can the staged pipeline produce reliable, source-grounded legal event graphs? (ii) Do these graphs improve legal classification and document-scoped question answering relative to text-based representations? (iii) Do these benefits extend to external and openretrieval legal QA settings?

We evaluate graph construction at multiple stages, including fact extraction, chunk-level graph extraction, and document-level merging, using automatic measures, human annotation, and multimodel judging. We then assess downstream utility through claim-type classification and documentscoped FAQ question answering, followed by bar-exam experiments testing generalization to multiple-choice QA and open retrieval. Exploratory case-clustering analyses are reported in Appendix C.

Our contributions are threefold. (1) We introduce a source-grounded, role-aware EKG pipeline for U.S. employment-discrimination complaints that represents events with participant, temporal, and causal structure. (2) We introduce a multi-stage evaluation protocol for legal event graphs, combining human annotation with complementary automatic and LLM-based evaluation.<sup>2</sup> (3) We show that the graph-structured classifier outperforms raw and linearized baselines on held-out claim classification and that EKG-based retrieval improves document-scoped QA; in open retrieval, gains remain limited when relevant evidence is absent from the first-stage candidate set.

## 2 Related Work

ARGUS intersects three lines of work: legal retrieval and representation, event extraction and temporal reasoning, and graph-augmented retrieval.

Legal retrieval and representation. Legal case retrieval has progressed from lexical matching toward neural interaction and representation learning (Feng et al., 2024). BERT-PLI models paragraph-level interactions between cases (Shao et al., 2020), while legal-domain models such as LEGAL-BERT (Chalkidis et al., 2020) and models trained on LeXFiles (Chalkidis et al., 2023) provide domain-adapted representations for legal NLP. The reasoning-focused legal retrieval benchmark used in our bar-exam experiments separates retrieval quality from downstream answer selection (Zheng et al., 2025).

Event extraction and temporal structure. General information-extraction systems such as Dy-GIE++ (Wadden et al., 2019), MetaGraph (Pedinotti et al., 2026) and OneIE (Lin et al., 2020) jointly model entities, relations, and/or events. In the legal domain, Filtz et al. (2020) extract event types, participants, and temporal information from ECHR decisions, while LexTime evaluates temporal ordering of event pairs in U.S. federal complaints (Barale et al., 2025). The 5W1H framework provides a compact representation of core event information (Hamborg et al., 2019). REGen combines exact, relaxed, and LLM-based matching for generative event-argument evaluation and validates these measures against human judgments (Sharif et al., 2025). ARGUS applies a 5W1Hinspired schema to source-grounded mini-graphs and merges them into document-level EKGs with legal participant roles and temporal and causal relations.

Legal knowledge graphs and graph-augmented retrieval. Li et al. (2024) construct a Chinese legal knowledge graph from criminal-law materials using a knowledge-enhanced LLM. More broadly, GraphRAG constructs entity-based graph representations for query-focused summarization (Edge et al., 2024), while KG<sup>2</sup>RAG uses knowledgegraph relations to expand and organize retrieved chunks (Zhu et al., 2025). EventRAG builds event knowledge graphs for multi-event reasoning (Yang et al., 2025). In the legal domain, Kordjamshidi et al. (2026) investigate structured representations and their efficacy in tax-law reasoning for QA, and LegalGraphRAG (Chen et al., 2026) combines hierarchical legal graphs with multi-agent retrieval and verification. ARGUS instead focuses on source-grounded, document-level EKGs for U.S. employment-discrimination complaints, preserving legal participant roles and temporal and causal relations through staged document-level merging.

## 3 Method

ARGUS converts the FACTS narrative of a U.S. employment-discrimination complaint into a source-grounded, document-level Event Knowledge Graph (EKG). The pipeline has four stages: (1) sentence-level fact extraction, (2) semantic chunking, (3) two-stage mini-graph extraction, and (4) deterministic and LLM-assisted document-level merging. Figure 1 summarizes the pipeline. We evaluate each component separately to distinguish errors introduced during fact selection, event and relation extraction, and graph merging.

Fact extraction. We define facts as event-bearing statements describing concrete actions, communications, decisions, or outcomes, distinguishing them from procedural or rhetorical text. Each sentence s is classified as Fact or Non-Fact using a fine-tuned LEGAL-BERT or LexLM encoder (Chalkidis et al., 2020, 2023). Let $p ( s )$ denote the classifier’s maximum softmax probability. Predictions above a confidence threshold τ are retained directly; sentences with $p ( s ) < \tau$ are passed to an LLM, which returns the identifiers of sentences judged to be facts. Only sentences ultimately labeled as facts are passed to graph construction.

Semantic chunking. Retained fact sentences preserve their original identifiers and are formatted as sentence-labeled input ([S1], [S2], . . . ). An LLM groups them into coherent event sequences, topics, or narrative phases. The prompt favors boundaries associated with changes in theme, time, or actor, avoids splitting an event across chunks, requires coverage of all retained sentences, and introduces approximately two to three sentences of overlap.

Event representation. ARGUS uses a 5W1Hinspired representation (Hamborg et al., 2019) within a structured event-extraction formulation (Srivastava et al., 2025). Entity records contain an identifier, name, entity kind, canonical legal role, and optional aliases. Event records contain an identifier, event type, main verb, trigger, participants, temporal information, source evidence, and confidence. Participants reference entity identifiers and carry event-specific roles, while evidence retains supporting sentence identifiers.

Temporal information may be explicit, relative, or unknown, following the broader problem of event-pair temporal relation modeling (Ning et al., 2018). The extraction schema supports BEFORE, AFTER, OVERLAP, and SAME-TIME temporal relations and CAUSES, ENABLES, and PREVENTS causal relations. ARGUS retains relations only when supported by the source text. For graphbased classification, events and participant entities become nodes; event–entity edges represent participation, and temporal and causal relations remain event–event edges. An example extracted minigraph is shown in Appendix A, Figure 4; the full downstream GNN architecture is provided in Appendix B.3.

Two-stage mini-graph extraction. Each chunk is processed through two structured-generation stages. Stage 1 extracts grounded entities and events and is explicitly instructed not to produce temporal or causal edges. Stage 2 receives the original chunk and the complete Stage 1 JSON output and adds only source-supported temporal and causal relations. Both stages require strict JSON, preservation of sentence identifiers, and no unsupported entities, events, dates, or relations.

The extraction configuration is optimized with GEPA (Agrawal et al., 2026). A candidate specifies the Stage 1 and Stage 2 models, prompts, retry instructions, and extraction settings. Candidates are evaluated using structural checks—including schema validity, reference integrity, temporal and causal consistency, and coverage—together with an LLM judge that scores factual accuracy, completeness, relevance, faithfulness, coherence, and overall quality. Cost is measured using token counts when available and API-call count otherwise. GEPA uses the metrics and observed failures to propose mutations and retains non-dominated configurations on a quality–cost Pareto frontier. The two-stage extractor and GEPA optimization loop are shown in Appendix Figures 2 and 3.

We also evaluate a Generative FrameNet-guided configuration (Tayyar Madabushi et al., 2025), using frame hints such as hiring, retaliation, and termination to guide event typing.

Document-level graph merging. Mini-graphs are first combined through deterministic merging. Entity identifiers are remapped into a documentlevel namespace, and mentions with the same lowercased, whitespace-normalized name are assigned to the same entity. Events are deduplicated using a structured key containing event type, normalized main verb, temporal value, participant entity–role pairs, and supporting sentence identifiers. Duplicate records accumulate their source references, and existing intra-chunk edges are remapped to the resulting document-level event identifiers.

ARGUS then performs LLM-assisted crosschunk resolution. Candidate pairs are restricted to events originating from different chunks and ranked by similarity over their event descriptions and contextual evidence. For each candidate, the LLM determines whether the records describe the same occurrence and whether a supported temporal or causal relation should be added. The prompt favors precision and returns no relation when evidence is insufficient. Duplicate events may be collapsed, after which existing edges are remapped and accepted cross-chunk relations are added.

Component-level evaluation. Fact extraction is evaluated by comparing LEGAL-BERT and LexLM with a prompted GPT-5 baseline, using Fact-class F1 as the primary metric and accuracy, precision, and recall as supporting measures. Minigraph extraction is evaluated on a fixed 24-chunk set using automatic structural and semantic measures and human annotation. Four annotators inspect each graph with its source text and assign Aligned, Partially Aligned, or Misaligned, together with an error tag and explanation when applicable. We report percent agreement, Fleiss’ κ, and Krippendorff’s α.

Document-level construction is compared with a one-shot Claude Sonnet 4.5 baseline that generates a complete EKG directly from the full FACTS narrative; the exact prompt is provided in Appendix B.1. Three LLM judges—DeepSeek-R1, Qwen3-Coder, and Mistral-Large—compare the staged and one-shot graphs on event grounding, evidence grounding, edge correctness, and overall quality. We additionally report inter-judge agreement, Gwet’s AC1 (Gwet, 2008), and stability under temperature variation (Appendix B.2, Table 11). Multiple judges complement human annotation and reduce reliance on a single evaluator (Zheng et al., 2023). U.S. holdout results are reported in Appendix B.4, Table 12.

## 4 Experimental Setup

## 4.1 Datasets

We collect 153 employment-discrimination complaints (Nature of Suit 442, Civil Employment Cases) from CourtListener,<sup>3</sup> an open legal-search repository maintained by the Free Law Project.

A complaint is the initiating pleading filed by a plaintiff and describes allegations rather than facts established by a court. Accordingly, the events represented by ARGUS reflect the allegations contained in the complaint and should not be interpreted as adjudicated findings.

The raw filings are OCR-derived scanned court documents with no gold labels for facts, events, or case similarity. We retrieve them through a Selenium-based pipeline and preprocess them using OCR-to-text conversion, header/footer and boilerplate removal, section detection, sentence segmentation, and normalization. Section detection isolates the FACTS narrative from jurisdictional and prayer-for-relief text.

The corpus covers five U.S. district courts to include jurisdiction-specific writing variation (Table 1): Eastern District of Pennsylvania (EAPD), Eastern District of Kentucky (KYED), Southern

<table><tr><td>District</td><td>Docs</td><td>Chunks</td><td>Avg/Doc</td><td>Sentences</td><td>Fact Rate</td></tr><tr><td>EAPD</td><td>20</td><td>87</td><td>4.35</td><td></td><td></td></tr><tr><td>KYED</td><td>31</td><td>138</td><td>4.45</td><td>1,155</td><td>81.7%</td></tr><tr><td>NYSD</td><td>46</td><td>318</td><td>6.91</td><td>4,808</td><td>92.7%</td></tr><tr><td>PAWD</td><td>28</td><td>165</td><td>5.89</td><td>1,607</td><td>93.0%</td></tr><tr><td>RID</td><td>28</td><td>118</td><td>4.21</td><td>1,631</td><td>90.9%</td></tr></table>

Table 1: Dataset statistics across districts. Fact Rate is the fraction of sentences labeled Fact; sentence-level annotation is not yet complete for EAPD.
<table><tr><td>Model</td><td>Acc.</td><td>Prec.</td><td>Rec.</td><td>F1</td></tr><tr><td>LEGAL-BERT</td><td>0.9400</td><td>0.8889</td><td>0.6154</td><td>0.7273</td></tr><tr><td>LexLM</td><td>0.9300</td><td>0.8000</td><td>0.6154</td><td>0.6957</td></tr><tr><td>GPT-5</td><td>0.8500</td><td>0.4615</td><td>0.9231</td><td>0.6154</td></tr></table>

Table 2: Fact-extraction results on a fixed 100-sentence U.S. gold sample. LEGAL-BERT and LexLM are finetuned legal-domain encoders; GPT-5 is a prompted baseline. Fact is the positive class.

District of New York (NYSD), Western District of Pennsylvania (PAWD), and District of Rhode Island (RID). Documents vary substantially in length and structure. Where sentence-level annotation is complete, the fact rate ranges from 81.7% (KYED) to 93.0% (PAWD). All splits are defined at the document level to prevent boilerplate and near-duplicate leakage; sentence-level fact extraction inherits its parent document split, while mini-graph extraction and merging are inference-only.

## 4.2 Evaluation of Graph Construction

Fact extraction. We fine-tune LEGAL-BERT and LexLM for sentence-level Fact/Non-Fact classification and compare them with a prompted GPT-5 baseline on a fixed 100-sentence U.S. gold sample (Table 2). Fact-class F1 is the primary metric, with accuracy, precision, and recall also reported. LEGAL-BERT achieves the highest F1 (0.7273) and precision (0.8889), while GPT-5 achieves substantially higher recall (0.9231). This precision– recall complementarity motivates the hybrid fallback described in Section 3. On the U.S. holdout, classifier-only and hybrid modes perform similarly (F1 ≈ 0.87 for LEGAL-BERT and ≈ 0.86 for LexLM), whereas LLM-only extraction is weaker (F1 ≈ 0.71–0.72; see Appendix B.4, Table 12).

Mini-graph extraction. We evaluate mini-graph extraction on a fixed 24-chunk set. Claude Sonnet 4.5 + GEPA achieves a quality score of 0.896, with schema validity and coverage of 1.0; causal self-loops are the main residual structural error. Among open-source alternatives (Table 3), Qwen achieves higher semantic quality and produces more events and relations per chunk, while GPT-OSS-120B scores slightly higher on structural and temporal-consistency measures.

<table><tr><td>Metric</td><td>GPT-OSS</td><td>Qwen</td></tr><tr><td>Quality Score</td><td>0.8272</td><td>0.8852</td></tr><tr><td>Judge Score</td><td>0.6642</td><td>0.7966</td></tr><tr><td>Structural Score</td><td>0.9902</td><td>0.9739</td></tr><tr><td>Temporal Consistency</td><td>0.9118</td><td>0.7647</td></tr><tr><td>Events/Chunk</td><td>7.09</td><td>7.50</td></tr><tr><td>Temporal Edges/Chunk</td><td>2.15</td><td>3.12</td></tr><tr><td>Causal Edges/Chunk</td><td>1.00</td><td>2.29</td></tr><tr><td>Schema Validity</td><td>1.00</td><td>1.00</td></tr></table>

Table 3: Open-source model comparison for mini-graph extraction.
<table><tr><td>Metric</td><td>Claude</td><td>Qwen</td></tr><tr><td>Aligned Partially Aligned</td><td>78.30% 17.40%</td><td>85.20% 14.80%</td></tr><tr><td>Misaligned Percent Agreement</td><td>4.30% 69.60%</td><td>0.00% 91.70%</td></tr><tr><td>Fleiss&#x27;κ</td><td>0.143</td><td>0.669</td></tr><tr><td>Krippendorff&#x27;s α</td><td>0.153</td><td>0.673</td></tr></table>

Table 4: Human evaluation of mini-graph alignment with the source text.

For human evaluation, four team members independently review each mini-graph with its source text and assign Aligned, Partially Aligned, or Misaligned, plus a free-text explanation and, when applicable, an error tag such as missing event, wrong role, or hallucinated detail. We report percent agreement, Fleiss’ κ, and Krippendorff’s α (Table 4). The FrameNet-guided Qwen configuration reaches 85.2% Aligned and Fleiss’ κ = 0.669, compared with 78.3% and κ = 0.143 for the Claude configuration. Because model choice and FrameNet guidance vary together, this is a configuration-level comparison rather than an isolated FrameNet ablation.

Document-level graph merging. We first measure how document-level merging changes graph size on five validation documents. Table 5 shows that event counts are almost entirely preserved: only one event is removed across the five cases. Entity counts decrease substantially in four cases as repeated mentions are consolidated into documentlevel entities. This provides a structural check before comparison with a one-shot baseline.

<table><tr><td>Case</td><td>Pre-Ev</td><td>Post-Ev</td><td>Loss</td><td>Pre-En</td><td>Post-En</td></tr><tr><td>Miczulski v. Alix</td><td>27</td><td>26</td><td>1</td><td>31</td><td>13</td></tr><tr><td>Poole v. Ampler</td><td>40</td><td>40</td><td>0</td><td>34</td><td>17</td></tr><tr><td>Sermarini v. Step Up</td><td>35</td><td>35</td><td>0</td><td>26</td><td>10</td></tr><tr><td>Bleiler v. Chester</td><td>27</td><td>27</td><td>0</td><td>24</td><td>13</td></tr><tr><td>Bacon v. M.A.G.</td><td>5</td><td>5</td><td>0</td><td>6</td><td>6</td></tr></table>

Table 5: Event and entity counts before and after document-level merging on five validation cases.

<table><tr><td>Case</td><td>Events</td><td>Temporal</td><td>Causal</td></tr><tr><td>Miczulski v. Alix Inc</td><td>21</td><td>19</td><td>6</td></tr><tr><td>Poole v. Ampler Pizza</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Sermarini v. A Step Up</td><td>16</td><td>14</td><td>20</td></tr><tr><td>Bleiler v. Chester Co.</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Bacon v. M.A.G. Ent.</td><td>17</td><td>16</td><td>13</td></tr></table>

Table 6: One-shot LLM baseline event-graph extraction on the five merger-validation documents.

We then compare ARGUS with a one-shot baseline in which Claude Sonnet 4.5, accessed through an OpenAI-compatible hosted API, generates a complete document-level EKG directly from the full FACTS section, without semantic chunking, intermediate mini-graphs, or staged merging. The exact prompt is provided in Appendix B.1. The baseline extracts no events for two of five validation documents and varies substantially in coverage on the others (Table 6), demonstrating the difficulty of performing event extraction and relational reasoning over the complete narrative in one generation.

Three LLM judges—DeepSeek-R1, Qwen3- Coder, and Mistral-Large—compare the one-shot and staged graphs on event grounding, evidence grounding, edge correctness, and overall quality. ARGUS is preferred on evidence grounding, edge correctness, and overall quality in all comparisons and on event grounding in 93.3% of comparisons (all $p = 0 . 0 3 1$ ; Table 7). Intra-model stability under temperature variation is 100% for Qwen and Mistral and 80% for DeepSeek-R1 (Appendix B.2, Table 11).

## 5 Experiments on Downstream Tasks

## 5.1 Classification

Setup. We formulate claim-type prediction as multi-label classification over 23 concepts from the SALI Alliance’s Legal Matter Specification Standard $\mathrm { ( L M S S ) ^ { 4 } }$ across 130 complaints (176 label assignments), using LLM-generated silver labels reviewed by humans. We compare three representations over the same label space: the preprocessed complaint text encoded from a 512-token input, a linearized entity/event EKG, and a graphstructured EKG using participation, temporal, and causal edges. Raw text provides a text baseline; the linearized EKG tests whether graph topology adds value beyond serialization of the extracted graph.

<table><tr><td>Dimension</td><td>Win</td><td>Agree.</td><td>AC1</td><td>p</td></tr><tr><td>Event Grounding</td><td>93.3%</td><td>86.7%</td><td>0.848</td><td>0.031</td></tr><tr><td>Evidence Grounding</td><td>100%</td><td>100%</td><td>1.0</td><td>0.031</td></tr><tr><td>Edge Correctness</td><td>100%</td><td>100%</td><td>1.0</td><td>0.031</td></tr><tr><td>Overall</td><td>100%</td><td>100%</td><td>1.0</td><td>0.031</td></tr></table>

Table 7: Multi-model evaluation of ARGUS relative to the one-shot baseline.
<table><tr><td>Representation</td><td>Micro-F1@0.5</td></tr><tr><td>Raw-text LEGAL-BERT</td><td>0.5897</td></tr><tr><td>Linearized-EKG LEGAL-BERT</td><td>0.5833</td></tr><tr><td>EKG-GNN</td><td>0.6415</td></tr></table>

Table 8: SALI claim-type classification on the 22- document held-out test set.

Results. On the 22-document held-out test set (Micro-F1@0.5; Table 8), the raw-text and linearized-EKG LEGAL-BERT models obtain 0.590 and 0.583, respectively, while the graphstructured EKG-GNN reaches 0.642. This is an absolute gain of 0.052 Micro-F1, or 8.8% relative to the raw-text baseline. The graph-structured representation therefore performs best on this held-out set.

GNN architecture. The EKG-GNN represents participant entities and events as nodes, with event– entity participation edges and event–event temporal and causal edges. Each node’s naturallanguage description is encoded with LEGAL-BERT (Chalkidis et al., 2020) and projected to 256 dimensions. Two mean-neighbor graphconvolution layers propagate information over the document graph:

$$
h _ { i } ^ { ( l + 1 ) } = \mathrm { R e L U } \left( W _ { s } h _ { i } ^ { ( l ) } + W _ { n } \frac { 1 } { | \mathcal { N } ( i ) | } \sum _ { j \in \mathcal { N } ( i ) } h _ { j } ^ { ( l ) } \right)
$$

Each layer is followed by dropout $( p = 0 . 1 )$ . Selfloops are excluded from the neighbor average because the node’s own representation is modeled separately through $W _ { s }$ . After two layers, node representations are mean-pooled and passed to a 23-label classification head. The reported $\tt f t { - } B E R T$ configuration jointly fine-tunes LEGAL-BERT and the graph layers using weighted binary cross-entropy and AdamW. Full architecture and training settings are provided in Appendix B.3.

## 5.2 Clustering vs. Distance

Setup. Each merged EKG is reduced to Core6D: log-counts of entities, events, temporal edges, and causal edges, plus graph density and temporal-edge ratio. The vectors are L2-normalized and clustered with K-means. We use the silhouette score to compare graph-derived representations with a TF-IDF bag-of-words baseline and to evaluate alternative values of K.

Core6D reaches a silhouette score of 0.649 at $K = 4 ,$ compared with 0.091 for TF-IDF, 0.363 for the full high-dimensional EKG representation, and 0.465 for Legacy7D. Although the $K = 2 – 4 0$ sweep favors very small K, we use $K = 4$ to retain stronger granularity with high cohesion. Compact structural summaries therefore produce substantially cleaner cluster geometry than TF-IDF or the unreduced EKG representation at this corpus size. Full results are reported in Appendix C, Table 13.

Feature ablations. We compare three Core6Dbased configurations using distance to the assigned centroid and an independent LLM-judged clustermembership score. Run A combines SALI claim tags with Core6D and produces one dominant cluster (111/14/6/2), with Pearson $r \approx - 0 . 0 6$ between centroid distance and judged membership. Run B augments Core6D with a 16-dimensional PCAcompressed EKG-context embedding, producing more balanced clusters, with Pearson r ≈ −0.28 and Spearman $\rho \approx - 0 . 3 1$ . Run C adds SALI tags and a kNN graph-neighbor block to Run B, reaching Pearson $r \approx - 0 . 2 9$ and Spearman $\rho \approx - 0 . 3 7 $ without clearly outperforming Run B.

The pattern indicates that continuous EKGderived context contributes more clustering signal than SALI claim tags alone, while the additional Run C components provide limited further benefit. Validation uses geometric and LLM-judged membership diagnostics rather than attorney relevance judgments. Detailed feature definitions, ablations, and plots are provided in Appendix C (Tables 14– 15 and Figures 5–7).

## 5.3 Document-Scoped Question Answering

Setup. We build a 300-question FAQ benchmark over 10 CourtListener employment-discrimination complaints (30 questions per document), spanning four graph-relevant question types: temporal order (what occurred before or after an event), causal/enablement (what allegedly caused or enabled an event), temporal overlap (whether two events were concurrent), and party resolution (who did what to whom), plus a small residual temporal-relation category. Each question has a gold answer grounded in its source complaint.

Holding GPT-4o and the answering prompt fixed, we compare three retrieval conditions:

1. Raw document — the FACTS section is chunked and retrieved directly, without graph structure.

2. Hybrid text + EKG — linearized documentlevel EKG context is combined with retrieved raw-text context.

3. EKG only — the linearized EKG is used without raw-text context.

Answer quality is measured with Token-F1 against the gold answer. Because each question is paired with its source complaint, this is a withindocument retrieval experiment: it evaluates representation and reasoning after the relevant document is known, not first-stage retrieval across CourtListener.

Results. EKG-only retrieval achieves 0.446 mean Token-F1, compared with 0.323 for raw text and 0.363 for hybrid text + EKG (Table 9), absolute gains of 0.123 and 0.083, respectively. Hybrid improves over raw text by 0.040, but adding raw context reduces performance relative to EKG alone. This pattern is consistent with additional unstructured context introducing noise, although the experiment does not isolate that mechanism.

Results by question type. The EKG gain over raw text is largest for temporal order $( 0 . 3 8 1 ~  ~ 0 . 5 3 8 , ~ + 0 . 1 5 7 )$ and causal/enablement $( 0 . 3 8 7  0 . 5 2 2 , \ + 0 . 1 3 4 )$ , categories directly aligned with graph relations. Temporaloverlap questions also improve substantially $( 0 . 2 4 1 ~  ~ 0 . 3 8 4 , ~ + 0 . 1 4 3 )$ The smallest improvement is for party resolution $( 0 . 1 5 2  0 . 1 9 7$ +0.045), where participant names and mentions are already accessible in source text. Full questiontype results are reported in Appendix D.1, Table 16.

<table><tr><td>Context condition</td><td>Mean Token-F1</td></tr><tr><td>Raw document</td><td>0.323</td></tr><tr><td>Hybrid text + EKG</td><td>0.363</td></tr><tr><td>EKG only</td><td>0.446</td></tr></table>

Table 9: Document-scoped FAQ answer quality over 300 questions from 10 CourtListener complaints (GPT-4o).

Significance of RAG gains. We test the $n = 2 9 9$ questions scored under all three conditions; one failed hybrid LLM call is excluded from all three arms to preserve pairing. We use a paired bootstrap 95% confidence interval over the mean F1 difference (10,000 resamples), Wilcoxon signed-rank test, and sign-flip paired permutation test (10,000 permutations).

All three tests agree in direction and significance. EKG-only exceeds raw text by +0.123 F1 $( p <$ $1 0 ^ { - 3 0 } )$ and hybrid by $+ 0 . 0 8 3 ( p < 1 0 ^ { - 2 0 } )$ , while hybrid exceeds raw text by $+ 0 . 0 4 0 \ ( p < 1 0 ^ { - 1 0 } )$ Bootstrap confidence intervals and exact Wilcoxon results are reported in Appendix D.1, Table 17.

Per-type tests show significant EKG-versus-raw improvements for every category with $n \ \geq \ 1 2 \colon$ temporal order $( p < \bar { 1 0 } ^ { - 1 7 } )$ , causal/enablement $( p < 1 0 ^ { - 1 5 } )$ , temporal overlap $( p = 2 . 4 \times 1 0 ^ { - 3 } )$ and party resolution $( p = 8 . 0 \times 1 0 ^ { - 5 } )$ . The residual temporal-relation category contains three questions and is not significant $( p = 0 . 2 5 )$ ; full results are reported in Appendix D.1, Table 18.

## 6 Generalization to External Datasets

Multiple-Choice Legal QA (Bar Exam). To test generalization beyond the employment-complaint FAQ benchmark, we evaluate on Multistate Bar Examination (MBE) multiple-choice questions (RegLab, Stanford University, 2024). We report two experiments with different retrieval setups and corpus scales; they are evaluated separately.

## 6.1 Experiment A: Retrieval-Condition Comparison

Setup. We evaluate 117 MBE questions, each containing a fact pattern, four answer choices, and a gold supporting passage drawn from 100 short appellate opinion excerpts. Unlike the FAQ benchmark, the task selects an answer choice (A–D) rather than generating free text, and its passages contain legal holdings and reasoning rather than narrative complaints. We compare raw document, chunked text, and EKG graph retrieval (top 3 passages each), holding Claude Sonnet 5 fixed.

Results. Raw document retrieval reaches 85.47% accuracy, while chunked text and EKG retrieval both reach 86.32%, a 0.85 percentage-point gain (Table 19). Among the 100 questions with an available gold-source EKG, EKG accuracy is 86.00%.

Significance. We use exact McNemar tests, paired bootstrap 95% confidence intervals (10,000 resamples), and Wilcoxon signed-rank tests. No pairwise comparison is significant: all McNemar p-values exceed 0.68 and all bootstrap intervals include zero. Only 6–9 outcomes differ between any pair of conditions. Full statistics are reported in Appendix D.2, Table 20.

Interpretation. The lack of a significant retrieval-condition difference contrasts with the larger gains on document-scoped FAQ (Section 5.3), where improvements are strongest for temporal order, causality, and enablement. Experiment A instead yields nearly identical chunked-text and EKG performance. EKG benefits therefore depend on the information required by the task rather than holding uniformly across legal QA.

## 6.2 Experiment B: Open-Retrieval and Oracle-Document Analysis

Setup. We evaluate 100 MBE-style multiplechoice questions from the legal\_bench subset, retrieving from the reglab/barexam\_qa corpus of 856,835 legal passages. Claude Sonnet 4.5 selects the final answer (A–D). We first measure open-retrieval Recall@10 and then downstream multiple-choice accuracy. BM25 and E5-large-v2 search the full corpus directly; hybrid Graph-RAG reranks the BM25 top-1,000 candidate set using graph-relevance scores where available.

Retrieval results. BM25 and E5 retrieve the gold passage in the top ten for 1% of questions, while hybrid Graph-RAG reaches 8% Recall@10 (Table 10). The gold passage appears in the BM25 top-1,000 pool for eight questions, and graph reranking moves all eight into the final top ten. The Recall@10 gain therefore occurs when first-stage retrieval has already placed the relevant passage in the candidate pool; graph reranking cannot recover passages absent from that pool.

<table><tr><td>Retrieval method</td><td>Recall@10</td></tr><tr><td>BM25</td><td>0.0100</td></tr><tr><td>E5-large-v2</td><td>0.0100</td></tr><tr><td>Hybrid Graph-RAG</td><td>0.0800</td></tr></table>

Table 10: Open-retrieval performance on the 100- question Bar Exam QA subset.

Downstream accuracy. BM25-RAG achieves 73% accuracy and open Graph-RAG 72%, both below the 76% no-passage baseline (Table 21). The gold-passage condition reaches 79%, showing that relevant evidence improves answer selection. In the open setting, however, the gold passage is absent from most retrieved contexts, so first-stage retrieval errors limit the downstream value of graph reranking.

Oracle-document setting. We additionally evaluate Graph-RAG when the relevant source document is known. The original configuration reaches 75% accuracy but produces empty graph retrievals for 18 questions. A revised configuration adds document-scoped retrieval, an MCQ-specific prompt, and hybrid context combining the gold passage with graph evidence, reaching 82%. Because these changes are introduced together, the 82% result is a configuration-level comparison rather than a graph-only ablation.

Interpretation. These experiments separate firststage retrieval from downstream use of structured evidence. Graph reranking raises open Recall@10 from 1% to 8%, but relevant passages enter the BM25 top-1,000 candidate pool for only eight of 100 questions, limiting downstream gains. In the oracle-document setting, the revised configuration reaches 82%, but it changes retrieval scope, prompting, and context together. The results support EKG structure for organizing and reasoning over alreadyrelevant evidence rather than replacing first-stage full-corpus retrieval.

## 7 Conclusion

We presented ARGUS, a staged pipeline for converting U.S. employment-discrimination complaints into source-grounded Event Knowledge

Graphs. The strongest gains occur when tasks exploit participant, temporal, or causal structure: the graph-structured classifier achieves higher heldout Micro-F1 than raw and linearized baselines, and EKG-only document-scoped retrieval improves FAQ answer quality. The bar-exam experiments provide a boundary condition: Experiment A shows no significant retrieval-condition difference, while open-retrieval gains occur only when relevant passages enter the first-stage candidate pool. AR-GUS therefore supports organizing and reasoning over relevant legal evidence rather than replacing first-stage retrieval.

## Limitations

The corpus contains 153 U.S. federal employmentdiscrimination complaints from five districts and does not establish generalization to other claims, jurisdictions, or legal genres. Several evaluations use small samples: 24 chunks for mini-graph quality, five documents for the merger comparison, and 100–117 questions for the bar-exam studies. The FAQ benchmark is document-scoped, and the clustering analysis has no attorney relevance judgments, so neither establishes end-to-end similar-case retrieval. Human graph evaluation was performed by four project members rather than legal practitioners, and LLM judges may share biases with the systems they evaluate. OCR, extraction, coreference, and inferred temporal or causal errors can propagate. Finally, the configuration yielding the 82% oracledocument result changes retrieval scope, prompting, and context together and should not be interpreted as a graph-only ablation.

## Ethical Considerations

Court complaints are public records but contain names and sensitive allegations. Automatically extracted roles, events, and causal relations can be incomplete or wrong and should not be treated as legal findings or used without attorney review. Any artifact release should follow the source repository’s terms, minimize unnecessary personal information, and state clearly that ARGUS is a research prototype for retrieval and organization rather than legal advice.

## Acknowledgments

This work grew out of a master’s capstone project at the University of Massachusetts Amherst conducted in collaboration with Bloomberg mentors. The authors gratefully acknowledge the longstanding research and educational collaboration between Bloomberg and UMass Amherst, including student mentorship, joint research, and scholarly exchange in natural language processing, data science, and related areas.

## References

Lakshya A. Agrawal, Shangyin Tan, Dilara Soylu, Noah Ziems, Rishi Khare, Krista Opsahl-Ong, Arnav Singhvi, Herumb Shandilya, Michael J. Ryan, Meng Jiang, Christopher Potts, Koushik Sen, Alexandros G. Dimakis, Ion Stoica, Dan Klein, Matei Zaharia, and Omar Khattab. 2026. GEPA: Reflective prompt evolution can outperform reinforcement learning. In The Fourteenth International Conference on Learning Representations. Oral presentation.

Claire Barale, Leslie Barrett, Vikram Sunil Bajaj, and Michael Rovatsos. 2025. LexTime: A benchmark for temporal ordering of legal events. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2025, pages 5220–5236, Suzhou, China. Association for Computational Linguistics.

Ilias Chalkidis, Manos Fergadiotis, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos. 2020. LEGAL-BERT: The muppets straight out of law school. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, pages 2898– 2904. Association for Computational Linguistics.

Ilias Chalkidis, Nicolas Garneau, Catalina Goanta, Daniel Katz, and Anders Søgaard. 2023. LeXFiles and LegalLAMA: Facilitating english multinational legal language model development. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15513–15535. Association for Computational Linguistics.

Zerui Chen, Qinggang Zhang, Zhishang Xiang, Zhimin Wei, Linfeng Gao, Xiao Huang, Zhihong Zhang, and Jinsong Su. 2026. LegalGraphRAG: Multi-agent graph retrieval-augmented generation for reliable legal reasoning. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 37455– 37484. Association for Computational Linguistics.

Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. 2024. From local to global: A graph RAG approach to query-focused summarization. arXiv preprint arXiv:2404.16130.

Yi Feng, Chuanyi Li, and Vincent Ng. 2024. Legal case retrieval: A survey of the state of the art. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 6472–6485. Association for Computational Linguistics.

Erwin Filtz, María Navas-Loro, Cristiana Santos, Axel Polleres, and Sabrina Kirrane. 2020. Events matter: Extraction of events from court decisions. In Legal Knowledge and Information Systems: JURIX 2020 – The Thirty-Third Annual Conference, volume 334 of Frontiers in Artificial Intelligence and Applications, pages 33–42. IOS Press.

Kilem Li Gwet. 2008. Computing inter-rater reliability and its variance in the presence of high agreement. British Journal ofMathematical and Statistical Psychology, 61(1):29–48.

Felix Hamborg, Corinna Breitinger, and Bela Gipp. 2019. Giveme5W1H: A universal system for extracting main events from news articles. In Proceedings of the 7th International Workshop on News Recommendation and Analytics, volume 2554 of CEUR Workshop Proceedings, pages 35–43.

Parisa Kordjamshidi, Samer Aslan, Madhavan Seshadri, Leslie Barrett, and Enrico Santus. 2026. Reasoners or translators? contamination-aware evaluation and neuro-symbolic robustness on tax law. In Proceedings ofthe First Workshop on Structured Understanding, Retrieval, and Generation in the LLM Era (SURGeLLM 2026), pages 344–360, San Diego, California, United States. Association for Computational Linguistics.

Jun Li, Lu Qian, Peifeng Liu, and Taoxiong Liu. 2024. Construction of legal knowledge graph based on knowledge-enhanced large language models. Information, 15(11):666.

Ying Lin, Heng Ji, Fei Huang, and Lingfei Wu. 2020. A joint neural model for information extraction with global features. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7999–8009. Association for Computational Linguistics.

Qiang Ning, Hao Wu, and Dan Roth. 2018. A multiaxis annotation scheme for event temporal relations. In Proceedings of the 56th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1318–1328. Association for Computational Linguistics.

Paolo Pedinotti, Peter Baumann, Nathan Jessurun, Leslie Barrett, and Enrico Santus. 2026. MetaGraph: A large-scale meta-analysis of GenAI in financial NLP (2022–2025). In Proceedings ofthe Fifth Workshop on Generation, Evaluation and Metrics (GEM), pages 848–861, San Diego, California, USA. Association for Computational Linguistics.

RegLab, Stanford University. 2024. Bar exam qa. https://huggingface.co/datasets/ reglab/barexam\_qa.

Yunqiu Shao, Jiaxin Mao, Yiqun Liu, Weizhi Ma, Ken Satoh, Min Zhang, and Shaoping Ma. 2020. BERT-PLI: Modeling paragraph-level interactions for legal case retrieval. In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, pages 3501–3507.

Omar Sharif, Joseph Gatto, Madhusudan Basak, and Sarah Masud Preum. 2025. REGen: A reliable evaluation framework for generative event argument extraction. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 12146– 12168, Suzhou, China. Association for Computational Linguistics.

Saurabh Srivastava, Sweta Pati, and Ziyu Yao. 2025. Instruction-tuning LLMs for event extraction with annotation guidelines. In Findings of the Association for Computational Linguistics: ACL 2025, pages 13055–13071, Vienna, Austria. Association for Computational Linguistics.

Harish Tayyar Madabushi, Taylor Pellegrin, and Claire Bonial. 2025. Generative FrameNet: Scalable and adaptive frames for interpretable knowledge storage and retrieval for LLMs powered by LLMs. In Proceedings of Bridging Neurons and Symbols for Natural Language Processing and Knowledge Graphs Reasoning @ COLING 2025, pages 107–119. ELRA and ICCL.

David Wadden, Ulme Wennberg, Yi Luan, and Hannaneh Hajishirzi. 2019. Entity, relation, and event extraction with contextualized span representations. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pages 5784–5789. Association for Computational Linguistics.

Zairun Yang, Yilin Wang, Zhengyan Shi, Yuan Yao, Lei Liang, Keyan Ding, Emine Yilmaz, Huajun Chen, and Qiang Zhang. 2025. EventRAG: Enhancing LLM generation with event knowledge graphs. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 16967–16979. Association for Computational Linguistics.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-Bench and chatbot arena. In Advances in Neural Information Processing Systems, volume 36, pages 46595–46623.

Lucia Zheng, Neel Guha, Javokhir Arifov, Sarah Zhang, Michal Skreta, Christopher D. Manning, Peter Henderson, and Daniel E. Ho. 2025. A reasoning-focused legal retrieval benchmark. In Proceedings ofthe 2025 Symposium on Computer Science and Law, pages 169–193. Association for Computing Machinery.

Xiangrong Zhu, Yuexiang Xie, Yi Liu, Yaliang Li, and Wei Hu. 2025. Knowledge graph-guided retrieval augmented generation. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8912–8924. Association for Computational Linguistics.

![](images/5a69ae9cbfeb902cbf69d789a6b6b58d8af8ed22657c9b72811a550a19bae09c.jpg)  
Figure 2: Two-stage mini-graph extractor with GEPAdriven optimization. Stage 1 extracts grounded entities and events; Stage 2 adds temporal and causal relations.

![](images/48c9cefd1e1a95fa496a72b2ace099414226e77beff12dc7415769e7dd2c7c86.jpg)  
Figure 3: GEPA candidate optimization loop: seed, evaluate, reflect, mutate, and retain Pareto-optimal configurations.

## A Additional Implementation Figures

Figure 2 illustrates the two-stage mini-graph extraction procedure, Figure 3 summarizes the GEPA optimization loop, and Figure 4 shows an example extracted mini-graph.

## B Additional Experimental Details

## B.1 One-Shot Event-Graph Prompt

The document-level baseline receives the complete sentence-labeled FACTS narrative and generates one event graph directly, without semantic chunking, intermediate mini-graphs, or staged documentlevel merging. The exact prompt is:

“You are a legal event extraction analyst. You are given one full legal document with sentence-labeled facts. Extract ONE overall document-level event graph, just extract what you think are events. Return strict JSON only. Schema requires: entities, events, temporal\_edges, causal\_edges with consistent IDs (E1..En, EV1..EVn). Events are the only nodes; entities attach to events as participants. Temporal relations: BE-FORE, SAME\_TIME. Causal relation:

![](images/235ada398f1e93c42010ccec0712b580dc7b7bfe18d080a27aad05dcf38d8974.jpg)  
Figure 4: Example of an event mini-graph extracted from a single chunk.

<table><tr><td>Model</td><td>Stability</td><td>Flip</td><td>AC1</td></tr><tr><td>DeepSeek</td><td>80%</td><td>20%</td><td>1.0</td></tr><tr><td>Qwen</td><td>100%</td><td>0%</td><td>1.0</td></tr><tr><td>Mistral</td><td>100%</td><td>0%</td><td>1.0</td></tr></table>

Table 11: Intra-model stability under temperature variation.

CAUSES, ENABLES.”

## B.2 Intra-Model Judge Stability

We additionally evaluate the stability of documentlevel graph judgments under temperature variation. Table 11 reports the fraction of judgments that remain unchanged and the corresponding flip rate.

## B.3 Full GNN Architecture

The EKG-GNN represents each merged documentlevel EKG as a homogeneous, undirected graph that contains both entity and event nodes and includes self-loops. Event–entity edges connect events to participant entities, while event–event edges represent temporal and causal relations. LEGAL-BERT encodes each node’s short natural-language description. The resulting [CLS] embedding is projected to 256 dimensions and passed through two meanneighbor graph-convolution layers. Each layer computes

$$
h _ { i } ^ { ( l + 1 ) } = \mathrm { R e L U } \Biggl ( W _ { s } h _ { i } ^ { ( l ) } + W _ { n } \frac { 1 } { | \mathscr { N } ( i ) | } \sum _ { j \in \mathscr { N } ( i ) } h _ { j } ^ { ( l ) } \Biggr ) .
$$

Each layer is followed by dropout $( p = 0 . 1 )$ Self-loops are stored in the graph but excluded from the neighbor average because the self contribution is handled separately by $W _ { s } .$ After two layers, node representations are mean-pooled and passed to a 23-label linear classifier.

The reported ft-BERT configuration jointly fine-tunes LEGAL-BERT and the graph layers using weighted binary cross-entropy and AdamW with a learning rate of $3 \times 1 0 ^ { - 4 }$ and a weight decay of 0.01. Training uses eight epochs, a batch size of 2, and gradient clipping at 1.0.

<table><tr><td>Mode</td><td>LEGAL-BERT</td><td>LexLM</td></tr><tr><td>Classifier</td><td>0.8718</td><td>0.8616</td></tr><tr><td>Hybrid</td><td>0.8711</td><td>0.8627</td></tr><tr><td>LLM</td><td>0.7115</td><td>0.7186</td></tr></table>

Table 12: U.S. holdout Fact-class F1 by fact-extraction mode.

<table><tr><td>Feature set</td><td>Silhouette</td></tr><tr><td>TF-IDF baseline</td><td>0.091</td></tr><tr><td>EKG full (2k dims)</td><td>0.363</td></tr><tr><td>EKG Legacy7D</td><td>0.465</td></tr><tr><td>EKG Core6D  $( K = 4 )$ </td><td>0.649</td></tr></table>

Table 13: Clustering silhouette by feature configuration.

## B.4 Fact Extraction Details

Table 12 reports the U.S. holdout F1 scores for classifier, hybrid, and LLM extraction modes.

## C Detailed Clustering Analysis

The clustering experiments are exploratory and examine whether graph-derived document representations yield coherent groupings of complaints. Table 13 compares the principal feature configurations using silhouette score.

We further evaluate three Core6D-based feature configurations (Table 14), whose components are summarized in Table 15. Validation compares distance to the assigned centroid with an LLM-judged cluster-membership score at K = 4.

Run A combines SALI claim tags with Core6D and produces one dominant cluster (111/14/6/2), with a weak distance–membership association (Pearson $\approx - 0 . 0 6 , n \approx 1 7 )$ . Run B combines Core6D with a 16-dimensional PCA-compressed EKG-context embedding and produces more balanced clusters, with Pearson $\approx ~ - 0 . 2 8$ and Spearman $\approx - 0 . 3 1 \ : ( n = 4 0 )$ . Run C adds SALI tags and a kNN cross-document graph-neighbor component to Run B; it produces cluster sizes of 31/61/19/22 and correlations of Pearson ≈ −0.29 and Spearman $\approx - 0 . 3 7 \ : ( n = 4 0 )$ . The difference between Runs B and C is small, so this exploratory analysis does not establish a clear advantage for the additional Run C features.

Separately, the distance–membership correlation for Run B remains similar when the validation sample is increased from five to ten documents per cluster (n = 20 versus $n = 4 0 )$ , providing a check on sensitivity to validation-sample size.

<table><tr><td></td><td>Run Representation</td><td>Sizes  $( K = 4 )$ </td><td></td><td>Pearson Spearman</td></tr><tr><td>A</td><td> $\mathbf { S A L I } ( 2 3 ) + \mathbf { C o r e } 6 \mathbf { D }$ </td><td>111/14/6/2</td><td>≈-0.06</td><td>一</td></tr><tr><td>B</td><td> $\mathrm { C o r e 6 D + H y b r i d P C A } ( 1 6 )$ </td><td>balanced, 30s–50s ≈ −0.28</td><td></td><td> $\approx - 0 . 3 1$ </td></tr><tr><td>C</td><td> $\mathrm { A } + \mathrm { H y b r i d ~ P C A } + k \mathrm { N N } ( 1 2 )$ </td><td>31/61/19/22</td><td>≈-0.29</td><td> $\approx - 0 . 3 7$ </td></tr></table>

Table 14: Cluster validation across three feature configurations. Correlations compare centroid distance with LLM-judged cluster membership; more negative values indicate stronger association between proximity to the centroid and judged membership. Validation n = 40 for Runs B and C and n ≈ 17 for Run A.

<table><tr><td>Feature</td><td>A</td><td>B</td><td>C</td></tr><tr><td>SALI claim tags</td><td>yes</td><td>no</td><td>yes</td></tr><tr><td>Core6D structure</td><td>yes</td><td>yes</td><td>yes</td></tr><tr><td>Hybrid PCA context</td><td>no</td><td>yes</td><td>yes</td></tr><tr><td>Graph neighbors (kNN)</td><td>no</td><td>no</td><td>yes</td></tr></table>

Table 15: Features included in each clustering configuration.

## D Additional Evaluation Tables

## D.1 FAQ Evaluation

Table 16 reports document-scoped FAQ performance by question type.

For the n = 299 questions successfully scored under all three retrieval conditions, we compare methods using paired bootstrap confidence intervals, Wilcoxon signed-rank tests, and sign-flip paired permutation tests. Table 17 reports the paired differences.

Table 18 reports the EKG-versus-raw comparison separately by question type. All categories with at least 12 examples show a significant difference; the three-example temporal-relation category is too small to support a conclusion.

## D.2 Bar-Exam Experiment A

Table 19 reports multiple-choice accuracy under the retrieval conditions used in Experiment A.

Pairwise significance is evaluated using exact McNemar tests, paired bootstrap confidence intervals, and Wilcoxon signed-rank tests. No pairwise comparison reaches statistical significance (Table 20).

![](images/35eb916008bfb9302ceb24f655e23860e9325808c05aa1670d3e75ba829951ac.jpg)  
Figure 5: Run C fused representation: Core6D, hybrid PCA context, a type-Jaccard graph-neighbor component, and LLM-assigned SALI tags are combined into a single vector Z before K-means clustering (K = 4).

![](images/c3899074afbe45ceb360b88238f47a3f3823b4e12e0388fb2fbd5ccb68694747.jpg)  
Figure 6: Run C validation: centroid distance versus LLM-judged cluster membership $( n ~ = ~ 4 0 ~ $ , Pearson $r \approx - 0 . 2 9$ , Spearman $\rho \approx - 0 . 3 7 )$

## D.3 Bar-Exam Experiment B

Table 10 reports Recall@10 for the full-corpus retrieval conditions in Experiment B.

Table 21 reports downstream multiple-choice accuracy under open-retrieval, gold-passage, and oracle-document conditions.

![](images/4a992997c72e328bae9cbfdf995dabcf8b41365ce0e1b72f14ad4a04569fbaed.jpg)

![](images/4ab96780bdbd75702e4630045e0f93f61612c65d903b1d96a238f47aa995e015.jpg)  
Core6D only (baseline)  
Run A: SALI + Core6D

Figure 7: Cluster validation by centroid distance and LLM-judged cluster membership for Core6D and Run A. This pairwise analysis preceded the threeconfiguration comparison in Table 14.
<table><tr><td>Question type</td><td>Raw F1</td><td>EKG F1</td></tr><tr><td>Temporal order</td><td>0.381</td><td>0.538</td></tr><tr><td>Causal/enablement</td><td>0.387</td><td>0.522</td></tr><tr><td>Temporal overlap</td><td>0.241</td><td>0.384</td></tr><tr><td>Party resolution</td><td>0.152</td><td>0.197</td></tr></table>

Table 16: FAQ performance by question type.

<table><tr><td>Comparison</td><td>∆F1</td><td>Bootstrap 95%CI</td><td>Wilcoxon p</td></tr><tr><td>EKG - Raw</td><td>+0.123</td><td> $[ + 0 . 1 0 9 , + 0 . 1 3 7 ]$ </td><td> $3 . 4 \times 1 0 ^ { - 3 9 }$ </td></tr><tr><td>EKG – Hybrid</td><td>+0.083</td><td> $[ + 0 . 0 6 9 , + 0 . 0 9 6 ]$ </td><td> $2 . 3 \times 1 0 ^ { - 2 7 }$ </td></tr><tr><td>Hybrid – Raw</td><td>+0.040</td><td> $[ + 0 . 0 3 0 , + 0 . 0 5 2 ]$ </td><td> $8 . 5 \times 1 0 ^ { - 1 1 }$ </td></tr></table>

Table 17: Paired significance tests for the FAQ benchmark (n = 299). All comparisons are significant at $\alpha = 0 . 0 5 ;$ sign-flip permutation tests agree in every comparison $( p < 1 0 ^ { - 3 }$ , not shown).

<table><tr><td>Question type</td><td>n</td><td>∆F1 (EKG-Raw)</td><td>Wilcoxon p</td></tr><tr><td>Temporal order</td><td>119</td><td>+0.157</td><td> $1 . 0 \times 1 0 ^ { - 1 8 }$ </td></tr><tr><td>Causal/enablement</td><td>99</td><td>+0.134</td><td> $2 . 7 \times 1 0 ^ { - 1 6 }$ </td></tr><tr><td>Temporal overlap</td><td>12</td><td>+0.143</td><td> $2 . 4 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Party resolution</td><td>66</td><td>+0.045</td><td> $8 . 0 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Temporal relation</td><td>3</td><td>+0.019</td><td>0.25</td></tr></table>

Table 18: EKG-versus-raw FAQ results by question type. Categories with $n \geq 1 2$ are significant at $\alpha = 0 . 0 5 ;$ no conclusion is drawn for the temporal-relation category $( n = 3 )$

<table><tr><td>Retrieval condition</td><td>MC accuracy</td></tr><tr><td>Raw document (top 3)</td><td>0.8547</td></tr><tr><td>Chunked text (top 3)</td><td>0.8632</td></tr><tr><td>EKG graph (top 3)</td><td>0.8632</td></tr><tr><td>Gold EKG subset (n = 100)</td><td>0.8600</td></tr></table>

Table 19: MBE multiple-choice accuracy by retrieval condition in Experiment A $( n = 1 1 7$ , Claude Sonnet 5).

<table><tr><td>Comparison</td><td>McNemar p</td><td>Bootstrap 95% CI</td><td>Wilcoxon p</td></tr><tr><td>Document vs. chunk</td><td>1.000</td><td> $[ - 5 . 1 3 , + 3 . 4 2 ]$  pp</td><td>0.813</td></tr><tr><td>Document vs. EKG</td><td>1.000</td><td> $[ - 5 . 9 8 , + 4 . 2 7 ] \ \mathrm { p p }$ </td><td>0.820</td></tr><tr><td>Chunk vs. EKG</td><td>0.688</td><td> $[ - 4 . 2 7 , + 4 . 2 7 ] \ \mathrm { p p }$ </td><td>1.000</td></tr></table>

Table 20: Pairwise significance tests for MBE accuracy in Experiment A. No comparison is significant at α = 0.05.

<table><tr><td>Context condition</td><td>MCQ accuracy</td></tr><tr><td>No passage</td><td>0.7600</td></tr><tr><td>BM25-RAG (open retrieval, top 10)</td><td>0.7300</td></tr><tr><td>Hybrid Graph-RAG (open retrieval, top 10)</td><td>0.7200</td></tr><tr><td>Gold passage</td><td>0.7900</td></tr><tr><td>Original oracle-document Graph-RAG</td><td>0.7500</td></tr><tr><td>Improved oracle-document Graph-RAG</td><td>0.8200</td></tr></table>

Table 21: Downstream Bar Exam multiple-choice accuracy in Experiment B (Claude Sonnet 4.5).