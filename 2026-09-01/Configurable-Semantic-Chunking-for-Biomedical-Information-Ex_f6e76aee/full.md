# Configurable Semantic Chunking for Biomedical Information Extraction in Retrieval-Augmented Generation

Riya Ahuja<sup>1,2,∗</sup> , Tim Kacprowski<sup>1,2</sup> , Roya Shiasi Sardoabi<sup>1,2,∗</sup>

<sup>1</sup>Institute of Data Science in Biomedicine, Technische Universitat Braunschweig, Germany¨ <sup>2</sup>Braunschweig Integrated Centre of Systems Biology (BRICS), Technische Universitat Braunschweig,¨

Germany

<sup>∗</sup>Corresponding authors: Riya Ahuja and Roya Shiasi Sardoabi r.ahuja@tu-braunschweig.de, t.kacprowski@tu-braunschweig.de, roya.shiasi-sardoabi@tu-braunschweig.de

## Abstract

BioMedRAG introduced retrieval-augmented generation with a learned chunk scorer for biomedical information extraction. However, it relies on fixed-size chunking which can fragment semantic evidence. We propose a configurable semantic chunking framework that addresses this limitation by combining entity-preserving windows, trigger-centered chunking, proposition-first extraction, tiered trigger prioritization, and hierarchical relation resolution. The framework integrates with BioMedRAG by replacing only the chunk construction stage while preserving the embedding model, learned chunk scorer, generator, and evaluation protocol. We evaluate the framework on biomedical relation extraction benchmarks (GM-CIHT, DDI, ChemProt) and adverse event classification (ADE). On GM-CIHT, the full hybrid configuration achieves 82.6% F1, improving over the fixed-size baseline (74.2% F1) by 8.4 points under our experimental setup. Cross-dataset analysis shows that semantic chunking improves extraction datasets with explicit relation cues, such as GM-CIHT and DDI, while fixed chunking remains competitive or stronger for dense biochemical extraction and binary classification settings such as ChemProt and ADE. By externalizing chunking logic into configuration files, the framework provides an interpretable and adaptable alternative to rigid fixed-size chunking for biomedical RAG pipelines.

## 1 Introduction

Biomedical literature is expanding quickly, with PubMed now indexing more than 39 million articles [National Library of Medicine, 2026]. Extracting structured knowledge, such as drug-disease interactions, gene-protein relationships, and adverse drug events, from this vast corpus is critical for clinical decision support, drug discovery, and precision medicine [Grouin and Grabar, 2024]. Retrieval-augmented generation (RAG) has emerged as a powerful paradigm for this task [Lewis et al., 2020], combining large language models (LLMs) with external evidence retrieval to improve factual grounding and interpretability.

BioMedRAG [Li et al., 2025] introduced a specialized RAG framework for biomedical information extraction and reports strong results on relation extraction benchmarks. However, it relies on fixed-size chunking, uniformly splitting sentences into 5-word windows regardless of semantic structure. This strategy frequently fragments relation-bearing expressions, resulting in incomplete or misaligned evidence. For example, the sentence “Aspirin treats headache by inhibiting prostaglandin synthesis” may be split into chunks such as “Aspirin treats headache by inhibiting” and “prostaglandin synthesis”, separating the drug mechanism from its target. Such fragmentation reduces retrieval precision and forces the LLM to infer missing relational context.

To address this limitation, we propose a configurable semantic chunking framework that preserves meaningful semantic boundaries through an incremental design. Our framework builds incrementally on entity-aware segmentation that respects named entity boundaries. We progressively incorporate: (1) relation-trigger-centered extraction with tiered prioritization to distinguish explicit triggers (treats, inhibits) from generic terms (medication, drug); (2) hierarchical relation resolution to handle competing relation types; and (3) proposition-first extraction that isolates minimal subjecttrigger-object spans. These components integrate through a unified scoring mechanism that balances trigger confidence, relation specificity, and contextual alignment, enabling seamless integration with BioMedRAG’s trained chunk scorer.

Accurate extraction of biomedical relations from literature has direct implications for clinical practice and translational medicine. Drug-drug interaction detection (DDI) informs prescription safety and adverse event prevention, while chemical-protein binding relationships (ChemProt) accelerate drug target discovery and repurposing. Gene-disease associations extracted from biomedical texts support precision medicine and clinical genomics. By improving retrieval quality in RAG systems, our semantic chunking framework enhances the reliability of automated biomedical knowledge ex traction, reducing manual curation burden and enabling realtime literature-based clinical decision support.

We evaluate our framework on four biomedical benchmarks from the BioMedRAG repository [Li et al., 2025]: GM-CIHT, DDI, and ChemProt for triple extraction, and ADE for adverse event classification. Experimental results reveal substantial improvements on complex relational tasks requiring precise evidence localization, while detailed component-wise analysis characterizes the contribution of each framework element. Cross-dataset comparison provides practical insights into when semantic chunking benefits retrieval-augmented generation systems.

Our key contributions are summarized as follows:

1. We propose a configuration-driven semantic chunking framework for biomedical retrieval-augmented information extraction. The framework replaces fixedsize windows with entity-preserving, trigger-aware, and proposition-oriented evidence candidates.

2. We integrate the proposed chunking framework into BioMedRAG while keeping the embedding model, learned chunk scorer, generator, preprocessing pipeline, and evaluation protocol unchanged. This controlled design isolates the effect of chunk construction from other system components.

3. We conduct an empirical evaluation on four biomedical benchmarks, including GM-CIHT, DDI, ChemProt, and ADE, and show that semantic chunking is most effective for extraction tasks with explicit relation cues and moderate entity density.

4. We provide component-wise ablation studies, including the effects of entity-aware segmentation, proposition extraction, trigger prioritization, relation hierarchy resolution, ranking strategy, and NER-based entity detection.

## 2 Related Work

## 2.1 Retrieval-Augmented Generation

Large language models encode knowledge parametrically during training, but this knowledge is inherently limited: it becomes outdated, lacks domain-specific coverage, and can lead to hallucinated outputs [Lewis et al., 2020]. Retrievalaugmented generation (RAG) mitigates these limitations by conditioning generation on externally retrieved evidence. Typical RAG pipelines operate in two stages: a retriever identifies relevant passages from a knowledge corpus, and a generator produces predictions conditioned on the retrieved evidence, improving factual accuracy and interpretability [Lewis et al., 2020].

Originally proposed for open-domain question answering [Lewis et al., 2020], RAG has since been applied to structured tasks such as knowledge base completion, dialogue systems, and code generation [Shuster et al., 2021; Zhou et al., 2023]. In biomedical NLP, accessing current literature is critical for accurate information extraction. BioMedRAG [Li et al., 2025] applies this paradigm to biomedical relation extraction and is the system we extend; we describe its pipeline in Section 3.1.

## 2.2 Text Chunking for Retrieval

Text chunking strategies strongly influence retrieval quality in RAG systems [Gao et al., 2023]. Fixed-size chunking splits text into uniform windows, while sliding windows add overlap to preserve context. Recent work has investigated optimal retrieval granularity, comparing document, passage, sentence, and proposition-level units [Chen et al., 2024]. However, these methods are designed for multi-sentence documents, making them ill-suited to biomedical relation extraction tasks, where relations are typically expressed within single sentences of 15-30 words.

Sub-sentence chunking methods aim to capture finergrained semantics than sentence or paragraph-level segmentation. Proposition-based approaches decompose sentences into atomic, self-contained units for retrieval [Chen et al., 2024; Hosseini et al., 2024]. Other work explores contextual chunk embeddings using long-context models [Gunther¨ et al., 2024]. These methods, however, are largely domainagnostic and do not enforce biomedical entity preservation or explicit modeling of relation triggers.

## 2.3 Biomedical Information Extraction

Biomedical relation extraction aims to identify structured relationships between entities in scientific literature [Zhou et al., 2014]. Traditional approaches employ supervised learning with manually designed linguistic and semantic features [Fundel et al., 2007] or neural architectures such as CNNs [Zeng et al., 2014], LSTMs [Zhou et al., 2016], and Graph Neural Networks [Zhu and others, 2019]. BioMedRAG [Li et al., 2025] demonstrates that evidence quality critically influences extraction performance in retrieval-augmented biomedical systems.

## 3 Method

## 3.1 Preliminaries: The BioMedRAG Pipeline

We briefly review BioMedRAG [Li et al., 2025], the retrievalaugmented extraction framework our approach extends.

BioMedRAG performs biomedical information extraction through retrieval-augmented generation. For each input sentence x, it retrieves the top-k relevant evidence chunks from a corpus D and generates triples conditioned on both x and the retrieved evidence [Li et al., 2025]. Candidate chunks are stored in a relational key–value memory, retrieved by embedding similarity, and re-ranked by a chunk scorer trained to prefer evidence that improves downstream prediction. Retrieval quality therefore depends fundamentally on chunk quality.

BioMedRAG constructs its chunk database by splitting each sentence into fixed 5-word windows with no overlap:

$$
\mathcal { C } _ { \mathrm { f i x e d } } ( x ) = \{ ( w _ { i } , \ldots , w _ { i + 4 } ) \ | \ i = 1 , 6 , 1 1 , \ldots \} ,\tag{1}
$$

which ignores semantic structure. This strategy frequently fragments entities, for instance, the hyphenated compound alpha-ketoglutarate may be split into alpha- and ketoglutarate across separate 5-word windows, and breaks relationbearing propositions across chunk boundaries, resulting in incomplete or misaligned evidence for the language model. Our framework replaces this chunk-construction stage; the embedding model, learned chunk scorer, generator, and evaluation protocol remain unchanged.

## 3.2 Problem Formulation

We address two biomedical information extraction tasks. The primary task is triple extraction: given a sentence $x =$ $( w _ { 1 } , \ldots , w _ { n } )$ , extract structured relations $( h , r , t )$ where h is the subject entity (head), $r \in \mathcal { R }$ is a relation type, and t is the object entity (tail). For example, from “aspirin inhibits prostaglandin synthesis”, we extract ⟨ASPIRIN, INHIBITS, PROSTAGLANDIN⟩. The secondary task is relation classification: given a sentence with marked entity spans, predict the relation type between them. Our chunking framework addresses both tasks, as both require identifying relationbearing evidence within sentences.

## 3.3 Configurable Semantic Chunking Framework

Our framework mitigates the fragmentation introduced by fixed-width chunking by constructing a hybrid candidate pool and selecting evidence with an explicit bias toward structurally complete relation spans.

Stage 1: Multi-Source Candidate Generation. For each sentence, we generate candidate chunks from three sources. (i) Entity-aware sliding windows preserve biomedical entity boundaries and capture relation-trigger context (Sections 3.4 and 3.5). (ii) Proposition-first extraction isolates minimal subject–trigger–object spans (Section 3.8); overlapping propositions are internally prioritized using tiered trigger specificity and relation hierarchy (Sections 3.6 and 3.7). (iii) Fixed-widthfallback windows (the original 5-word splits) are added to ensure non-empty coverage when entity/trigger signals are weak.

Stage 2: Similarity-Based Selection with Proposition Bias. Candidates are ranked by cosine similarity between chunk embeddings and the target relation definition, consistent with the embedding-based retrieval step in BioMedRAG [Li et al., 2025]. We then apply a proposition bias: if at least one proposition-based candidate appears among the top-N ranked candidates, we promote the best such proposition into the final top-k set. This guarantees that the downstream generator receives at least one chunk with verified subject–trigger– object structure, while retaining high-similarity contextual chunks for coverage. The selected chunks are finally passed to BioMedRAG’s trained chunk scorer for re-ranking.

Configurability. All dataset-specific knowledge, including trigger vocabularies, tier definitions, relation hierarchies, negation rules, and context markers, is externalized in JSON configuration files. This design makes the chunking decisions explicit and inspectable, enabling adaptation to new biomedical relation sets without code changes.

Shared Trigger Lexicon. Both entity-aware windows and proposition extraction use a common tiered trigger vocabulary for relation signal detection, ensuring consistent identification of relation-bearing spans across candidate types.

## 3.4 Entity-Aware Segmentation

This subsection describes the first candidate source in Stage 1: entity-aware sliding windows.

Biomedical named entities (drugs, proteins, diseases) frequently span multiple tokens and often include hyphenation (e.g., alpha-ketoglutarate, acetyl-CoA carboxylase). Fixedwidth chunking can cut through such entities, producing partial strings that degrade retrieval similarity and downstream relation evidence.

Entity Detection. We detect entity spans using lightweight pattern rules: bracketed markup (e.g., [aspirin]), capitalized multi-token terms (e.g., Tumor Necrosis Factor), and hyphenated compounds (e.g., alpha-ketoglutarate). Each entity e is represented as a character span $( e _ { \mathrm { s t a r t } } , e _ { \mathrm { e n d } } )$

Scope of Entity Detection. This step is intended as a boundary-preservation mechanism rather than as a complete biomedical named entity recognition module. The detected spans are used only to adjust chunk boundaries and construct proposition candidates; final relation prediction is still performed by the BioMedRAG scorer and generator. This design favors low overhead and direct integration with the existing pipeline, but it may miss lowercase or irregular biomedical mentions. To examine this limitation, we compare patternbased entity spans with NER-based spans in the ablation study.

Sliding Window Adjustment. Given a window size w and stride s, we adjust window boundaries whenever a boundary falls inside an entity span: (i) if the window end intersects an entity, we extend it to include the full entity; (ii) if the window start intersects an entity, we shift the start past the entity. This guarantees that no entity is split across chunks.

Example. Consider “Tumor necrosis factor alpha receptorcomplex regulates immune responses.” With fixed 5-token windows, one chunk may end at “... alpha receptor-” while the next begins with “complex ...”, splitting the hyphenated entity receptor-complex. Our entity-aware approach moves the boundary so that tumor necrosis factor alpha receptorcomplex remains intact within a single chunk.

## 3.5 Relation Trigger Detection

Entity-aware segmentation preserves entity boundaries but remains agnostic to whether a chunk actually expresses a semantic relation. We therefore augment the sliding window strategy with trigger-based detection to emphasize relationbearing content.

Trigger Lexicon Construction. For each dataset, candidate triggers are collected from the training split by identifying words and short phrases that occur between or near annotated entity pairs. These candidates are grouped by their associated gold relation labels and inspected for relation specificity. Terms that frequently co-occur with a single relation are retained as stronger triggers, whereas terms appearing across several relation types are assigned to lower tiers or removed when they provide limited discriminative value. Morphological variants expressing the same cue are merged, for example, “inhibit”, “inhibits”, “inhibited”, and “inhibition”. For instance, the INHIBITS relation may include triggers such as “inhibits”, “blocks”, “suppresses”, “antagonizes”, “downregulates”, and “attenuates”, depending on the dataset-specific relation definition. If no trigger is detected, the framework falls back to entity-aware sliding windows and fixed-width chunks to preserve coverage.

Trigger-Centered Chunking. When a trigger is detected, we generate a local context window of ±4 words around it to capture nearby entities and modifiers. For example, in “Aspirin effectively inhibits prostaglandin synthesis in tissues,” the trigger inhibits produces a chunk that captures both the agent (Aspirin) and the target (prostaglandin synthesis). The window size is intentionally small to emphasize local relational evidence.

Limitation. This approach treats all triggers equally, regardless of semantic specificity. Consequently, a chunk containing a generic term such as affects receives the same priority as one containing a precise term such as inhibits. This motivates tiered trigger prioritization, introduced next (Section 3.6).

## 3.6 Tiered Trigger Prioritization

Not all relation triggers carry equal semantic weight. Generic terms such as affects or modulates are ambiguous, while specific terms like inhibits or therapy convey precise mechanistic or clinical meaning. We therefore organize triggers into three tiers based on semantic specificity and directional clarity, following common distinctions in biomedical relation extraction between explicit actions, contextualized actions, and generic associations.

Trigger Tier Definitions. For each relation type, we define:

• Primary triggers (tier bonus 3): Highly specific terms that unambiguously indicate the relation (e.g., inhibits, therapy, activates).

• Secondary triggers (tier bonus 2): Moderately specific synonyms or related terms that express the relation with reduced precision (e.g., blocks, prescribed, enhances).

• Tertiary triggers (tier bonus 1): Generic or weaker signals that may indicate a relation but lack sufficient specificity when used in isolation (e.g., reduces, medication, associated).

Example. For TREATS, primary triggers include treats, treatment, and therapy; secondary triggers include prescribed, administered, and therapeutic; and tertiary triggers include medication and drug. The sentence “Aspirin therapy alleviates headache” contains a primary trigger (therapy), while “Aspirin medication alleviates headache” contains only a tertiary trigger (medication).

Usage. Tier membership is used during proposition-first extraction (Section 3.8) to rank overlapping candidate propositions. When multiple propositions cover the same text span, the proposition containing a higher-tier trigger is retained, ensuring that the most semantically precise relational evidence is passed to the selection stage.

<table><tr><td>Relation</td><td>Priority weight</td></tr><tr><td>TREATS</td><td>4</td></tr><tr><td>INHIBITS, STIMULATES, CAUSES, PREVENTS</td><td>3</td></tr><tr><td>AFFECTS, INTERACTS_WITH, REDUCES</td><td>2</td></tr><tr><td>COEXISTS_WITH</td><td>1</td></tr></table>

Table 1: Relation hierarchy for GM-CIHT. Higher weights indicate higher precedence during proposition ranking. Weights were tuned on the development set; alternative orderings yielded lower F1.

Tier Assignment. Triggers were assigned to tiers based on two criteria: (i) corpus frequency analysis of trigger–relation co-occurrence in training data, where high co-occurrence with a single relation indicates specificity, and (ii) linguistic directness, where verb forms (e.g., inhibits) are preferred over nominalizations (e.g., inhibition) and generic associations (e.g., affects). Tier assignments were validated on the development set (Table 2) by comparing proposition extraction precision across alternative classifications.

## 3.7 Relation Hierarchy Resolution

Building on tiered trigger prioritization, we address a second source of ambiguity: sentences that express multiple biomedical relations simultaneously. For example, “aspirin treats inflammation by inhibiting COX-2” contains both a TREATS relation (aspirin–inflammation) and an INHIBITS relation (aspirin–COX-2). Without explicit disambiguation, extracted propositions may emphasize relations that are less informative for the target extraction objective.

We resolve this ambiguity through configurable relation hierarchies that encode dataset- and task-specific priorities. For GM-CIHT, we define the hierarchy shown in Table 1.

Priority Assignment. Relation weights reflect two explicit criteria: (i) semantic specificity, indicating how directly a relation encodes an actionable interaction between entities, and (ii) task relevance, reflecting the importance of the relation for the target extraction objective. In GM-CIHT, TREATS receives the highest weight as the dataset emphasizes therapeutic relationships. Directional mechanistic relations such as INHIBITS and STIMULATES are informative but secondary, while COEXISTS WITH denotes weak associative evidence.

Hierarchy Tuning. Relation weights were determined through grid search on the development set (Table 2): we evaluated all permutations of relation orderings and selected the configuration maximizing extraction F1. The reported hierarchy in Table 1 outperformed flat (equal-weight) and inverted orderings by 1.2–2.1% F1. For other datasets, relation weights are specified through dataset-specific configuration files using the same principle, while the core chunking algo rithm and selection strategy remain unchanged.

Example. Consider “Metformin treatment reduces glucose levels by activating the AMPK pathway in diabetic patients.” This sentence supports multiple relations:

• TREATS: treatment (primary trigger), clinical context (diabetic patients)

• STIMULATES: activating (primary trigger)

• AFFECTS: reduces (secondary trigger)

Although all relations are supported by valid triggers, the hierarchy ensures that the proposition emphasizing the therapeutic relation (metformin–diabetes) is retained over mechanistic or associative alternatives.

Combined Disambiguation. Relation hierarchy resolution operates jointly with tiered trigger prioritization during proposition-first extraction (Section 3.8). When multiple propositions are extracted from the same sentence, they are ranked lexicographically: propositions associated with higher-priority relations are preferred, and ties are resolved by trigger tier strength. Only non-overlapping propositions with the highest precedence are retained.

## 3.8 Proposition-First Extraction

Sliding windows may capture entity context or relation context separately, but fail to preserve complete relational evidence in a single chunk. For example, “Metformin, a widely used antidiabetic, improves insulin sensitivity in diabetic patients” may yield windows containing the entity (“Metformin...antidiabetic”) or the relation (“improves insulin sensitivity”), but neither forms a complete triple.

We address this through proposition-first extraction, inspired by [Hosseini et al., 2024]. We extract candidate chunks corresponding to atomic propositions: self-contained units containing a subject, a relation trigger, and an object.

Proposition Patterns. We detect propositions using three syntactic patterns:

• Infix: Entity<sub>1</sub> – TRIGGER – Entity<sub>2</sub> (e.g., “Aspirin inhibits prostaglandins”);

• Prefix: TRIGGER – Entity<sub>1</sub> . . . Entity<sub>2</sub> (e.g., “Treatment ofdiabetes with metformin”);

• Postfix: Entity<sub>1</sub> . . . Entity<sub>2</sub> – TRIGGER (e.g., “Aspirin– COX-2 interaction”).

For each trigger, we identify nearest entities within a bounded context (±25 tokens) and extract the minimal span covering subject, trigger, and object. We expand each span by ±3 tokens to capture adjacent negation markers (e.g., does not,fails to) and uncertainty hedges (e.g., may, possibly) that can invert or weaken the expressed relation, ensuring the downstream model receives complete polarity information.

Pattern Coverage. These patterns cover the majority of explicit relation expressions in biomedical corpora. Complex constructions such as nested relations or coordinated arguments are handled implicitly: sliding window candidates provide fallback coverage when proposition patterns do not match.

Example. Consider the sentence: “Aspirin therapy reduces inflammation by inhibiting COX-2 expression.” Two propositions are extracted:

1. “Aspirin therapy reduces inflammation” - Trigger: reduces (tertiary for AFFECTS), Priority: $2 + 1 = 3$

2. “Aspirin inhibiting COX-2” - Trigger: inhibiting (primary for INHIBITS), Priority: $3 + 3 = 6$

The second proposition is ranked higher due to its stronger trigger and higher-priority relation, ensuring precise mechanistic evidence is retained.

Passive Handling. Passive constructions (e.g., “treated $b y ' )$ reverse surface entity order. We detect passive triggers followed by markers $( b y$ , with) and invert semantic roles accordingly.

Ranking and Integration. Overlapping propositions are deduplicated by retaining those with the highest combined priority:

$$
P ( p ) = \mathrm { W e i g h t } _ { \mathrm { h i e r a r c h y } } ( r ) + \mathrm { B o n u s } _ { \mathrm { t i e r } } ( t )\tag{2}
$$

where r is the detected relation type, t is the trigger word, $\mathsf { W e i g h t } _ { \mathsf { h i e r a r c h y } } ( r ) \ \in \ \{ 1 , 2 , 3 , 4 \}$ follows Table 1, and $\mathsf { B o n u s } _ { \mathrm { t i e r } } ( t ) \in \{ 3 , \dot { 2 } , 1 \}$ for primary, secondary, and tertiary triggers respectively. This formula integrates linguistic confidence (trigger specificity) with clinical importance (relation hierarchy) into a unified ranking score.

Proposition chunks complement sliding windows: propositions ensure structural completeness while windows preserve broader context. Both candidate types are passed to the hybrid selection stage (Section 3.9).

## 3.9 Hybrid Selection Strategy

The previous stages produce a diverse pool of candidates: entity-aware sliding windows, proposition spans (ranked using tiered triggers and relation hierarchy), and fixed-width fallback windows. We employ a hybrid strategy to select the final top-k chunks, integrating semantic scoring with structural prioritization.

Scoring. Following BioMedRAG [Li et al., 2025], all candidates are scored using cosine similarity between their averaged token embeddings and the target relation definition embedding.

Structural Prioritization via Proposition Bias. Embedding similarity alone may favor verbose chunks over precise propositions. To ensure that the contributions of tiered triggers and relation hierarchy (Sections 3.6–3.7) propagate to the final selection, we apply a proposition bias:

1. Slot 1 (Semantic Best): The highest-similarity candidate is selected for broad context.

2. Slot 2 (Structural Best): If a proposition candidate, already ranked by combined priority $P ( p )$ (Equation 2), exists in the top-N (N=5), it is promoted to this slot. This ensures that chunks containing high-tier triggers and high-priority relations are retained even if their embedding score is not the highest.

3. Fallback: Otherwise, the second-highest similarity chunk is selected.

Design Rationale. Tiered triggers and relation hierarchy are applied during proposition extraction rather than entityaware window generation. This reflects their distinct roles: entity-aware windows provide broad contextual coverage, capturing surrounding entities and modifiers that inform the language model even without explicit relation signals. Propositions, in contrast, target precise relational evidence where multiple competing triggers frequently co-occur within minimal spans. Tiered prioritization and hierarchy resolution are most effective when disambiguating such conflicts, which are common in propositions but rare in broader windows. Furthermore, applying internal scoring to entity-aware windows risks over-filtering useful contextual chunks that lack explicit triggers. The proposition bias ensures structurally complete evidence (ranked by tiers and hierarchy) reaches final selection, while entity-aware windows contribute complementary context ranked by semantic similarity alone.

<table><tr><td>Dataset</td><td>Task</td><td>Rels</td><td>Train</td><td>Dev</td><td>Test</td></tr><tr><td>GM-CIHT</td><td>Extraction</td><td>22</td><td>3,734</td><td>492</td><td>465</td></tr><tr><td>DDI</td><td>Extraction</td><td>4</td><td>1,027</td><td>258</td><td>1,094</td></tr><tr><td>ChemProt</td><td>Extraction</td><td>5</td><td>4,111</td><td>2,411</td><td>3,438</td></tr><tr><td>ADE</td><td>Classification</td><td>2</td><td>4,000</td><td>975</td><td>497</td></tr></table>

Table 2: Dataset statistics. Rels denotes the number of relation types; Train, Dev, and Test denote the number of instances (sentence-label pairs) in each split. GM-CIHT covers 22 general biomedical relations (therapeutic, mechanistic, associative). DDI focuses on drug-drug interactions. ChemProt targets chemical-protein bindings with subtle semantic distinctions. ADE is binary adverse event detection.

Integration. Selected chunks are passed to BioMedRAG’s trained chunk scorer for final re-ranking [Li et al., 2025], maintaining full pipeline compatibility.

## 4 Experiments

## 4.1 Datasets

We evaluate on four biomedical datasets (Table 2): GM-CIHT, DDI, ChemProt (triple extraction), and ADE (binary relation classification). As in the BioMedRAG evaluation setting, we do not consider link prediction, where inputs are too short to admit meaningful sub-sentence chunking.

GM-CIHT serves as the primary development dataset for designing the general semantic chunking strategy, including entity-aware windows, trigger-tier usage, relation hierarchy resolution, and proposition bias. For each benchmark, dataset-specific symbolic resources such as relation labels, trigger vocabularies, tier assignments, and hierarchy definitions are specified through external configuration files. We do not otherwise tune the chunk scorer or generator on DDI, ChemProt, or ADE. All corpora are converted to a unified JSONL schema (SENTENCE, SUBJECT TEXT, OBJECT TEXT, PREDICATE; ADE is mapped from its native format) so chunking, embedding, and generation share a consistent interface.

## 4.2 Baselines and Experimental Setup

Baseline. We compare against BioMedRAG (Fixed) [Li et al., 2025], which uses fixed 5-word chunking. This baseline is selected to isolate the effect of chunk construction while keeping the remaining pipeline unchanged, including the embedding model, learned chunk scorer, generator, preprocessing, and evaluation protocol. Thus, the comparison evaluates whether semantically structured evidence units improve

<table><tr><td>Dataset</td><td>Method</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>GM-CIHT</td><td>Fixed Ours</td><td>74.6 82.6</td><td>73.8 82.6</td><td>74.2 82.6</td></tr><tr><td>DDI</td><td>Fixed Ours</td><td>78.2 79.2</td><td>78.2 79.2</td><td>78.2 79.2</td></tr><tr><td>ChemProt</td><td>Fixed Ours</td><td>87.7 87.0</td><td>87.0 86.3</td><td>87.4 86.6</td></tr><tr><td>ADE</td><td>Fixed Ours</td><td>87.0 86.1</td><td>90.7 88.6</td><td>88.8 87.3</td></tr></table>

Table 3: Main results (%). P, R, and F1 denote micro-averaged Precision, Recall, and F1-score. Fixed is the BioMedRAG fixed 5-word chunking baseline; Ours is the proposed semantic chunking framework. Our semantic chunking achieves gains on GM-CIHT (+8.4 F1) and DDI (+1.0 F1), while fixed chunking remains competitive on ChemProt and ADE.

BioMedRAG-style retrieval over fixed-size windows. Comparisons with sentence-level chunking, dependency-based span extraction, and external proposition segmentation methods are left for future work.

Implementation. We use MedLLaMA-13B for computing chunk and relation embeddings with 512-token maximum length and mean-pooling. Following BioMedRAG [Li et al., 2025], we train chunk scorers initialized from Llama-2-13B with LoRA [Hu et al., 2022] (r=8, α=8, dropout 0.1) for 1,000 steps using AdamW with bfloat16 precision. Separate scorers are trained per dataset.

Chunking Parameters. Entity-aware windows use size w=8, stride s=3, bounds [5, 12] tokens. Proposition extraction searches ±25 tokens around triggers with ±3 token context margins. Proposition bias threshold N=5. These parameters are kept fixed across datasets to support controlled comparison; systematic sensitivity analysis of window size, context range, and proposition bias threshold is left for future work.

Configuration. The core chunking algorithm, window parameters, proposition bias threshold, embedding model, learned scorer, generator, preprocessing pipeline, and evaluation protocol are kept fixed across datasets. Dataset-specific symbolic resources are provided through external JSON configuration files.

Evaluation. Following BioMedRAG [Li et al., 2025], we report micro-averaged Precision, Recall, and F1 for the extraction tasks GM-CIHT, DDI, and ChemProt. For these tasks, a prediction is considered correct only if the head span, relation type, and tail span exactly match the ground truth. For ADE, we report binary classification performance using the provided entity spans. All experiments use NVIDIA A40 GPUs with 4-bit NF4 quantization and fixed random seeds.

## 4.3 Main Results

Table 3 presents results across all four datasets.

Our semantic chunking substantially improves extraction performance on GM-CIHT (+8.4 F1 points: 74.2 → 82.6) and DDI (+1.0 F1: 78.2 → 79.2). These datasets contain diverse relation types where tiered trigger prioritization and proposition extraction effectively disambiguate competing interpretations. On GM-CIHT, gains in both precision and recall indicate improved evidence quality without sacrificing coverage.

<table><tr><td>Configuration</td><td>F1</td><td>Δ</td></tr><tr><td>Fixed (Baseline)</td><td>74.2</td><td></td></tr><tr><td>+ Entity-Aware</td><td>76.1</td><td>+1.9</td></tr><tr><td>+ Proposition</td><td>76.3</td><td>+0.2</td></tr><tr><td>+ Tiered Triggers</td><td>78.7</td><td>+2.4</td></tr><tr><td>+ Hierarchy (Full)</td><td>82.6</td><td>+3.9</td></tr></table>

Table 4: Incremental component ablation on GM-CIHT (%). Each row adds one component to the configuration above it. F1 is the micro-averaged F1-score; ∆ is the change relative to the preceding row.

<table><tr><td>Method</td><td>1 Ex.</td><td>3 Ex.</td><td>Δ</td></tr><tr><td>Fixed</td><td>74.2</td><td>75.1</td><td>+0.9</td></tr><tr><td>Ours</td><td>82.6</td><td>77.2</td><td>-5.4</td></tr></table>

Table 5: Effect of the number of in-context examples on GM-CIHT (%). 1 Ex. and 3 Ex. denote micro-averaged F1 with one and three in-context demonstrations, respectively; ∆ is the change from one to three.

On ChemProt and ADE, fixed chunking remains competitive, outperforming our method by 0.8% and 1.5% F1 respectively. ChemProt involves fine-grained mechanistic distinctions (e.g., AGONIST vs. ACTIVATOR) that benefit from broader contextual cues preserved by fixed windows, while ADE is a binary classification task where comprehensive sentence-level context is more informative than targeted propositions. These results highlight that semantic chunking is most beneficial in settings with relational ambiguity rather than uniformly dense supervision, a pattern we analyze further in Section 4.5.

## 4.4 Ablation Studies

We analyze component contributions and in-context learning sensitivity on GM-CIHT.

Component Ablation. Table 4 shows incremental gains from each component.

Entity-aware chunking provides a strong foundation (+1.9 F1) by preventing fragmentation. Proposition extraction adds a small gain (+0.2 F1), and tiered triggers yield a larger step (+2.4 F1). The relation hierarchy yields the largest incremental gain (+3.9 F1) by aligning evidence selection with GM-CIHT’s therapeutic focus, consistently prioritizing TREATS relations over weaker associations such as COEXISTS WITH when multiple interpretations are present.

In-Context Learning Sensitivity. Table 5 examines prompting requirements.

Fixed chunking improves with additional examples (+0.9 F1), while our method performs best with one example, degrading with three (−5.4 F1). This suggests semantically coherent chunks saturate the model’s contextual needs with fewer demonstrations, potentially reducing inference costs compared to lower-quality evidence requiring more examples; the drop with three examples may also reflect context limits or sensitivity to demonstration choice.

<table><tr><td>Ranking</td><td>F1</td><td>∆</td></tr><tr><td>Cosine + proposition bias</td><td>82.6</td><td></td></tr><tr><td>70% cosine + 30% tiered/hierarchy</td><td>79.4</td><td>-3.2</td></tr></table>

Table 6: Ranking variant ablation on GM-CIHT (%). ∆ is the change in micro-averaged F1 relative to the default ranking (cosine similarity with proposition bias).
<table><tr><td>Entity detection</td><td>F1</td><td>∆</td></tr><tr><td>Regex (default)</td><td>82.6</td><td></td></tr><tr><td>NER</td><td>81.5</td><td>-1.1</td></tr></table>

Table 7: Entity detection variant on GM-CIHT (%). ∆ is the change in micro-averaged F1 relative to the default regex-based entity detection.

Ranking variant. Besides cosine similarity with proposition bias (Section 3.9), we evaluate a ranking that mixes 70% cosine similarity with 30% of a structural score derived from tiered triggers and relation hierarchy. Table 6 shows that this mixed ranking reduces GM-CIHT F1 to 79.4% (−3.2 vs. cosine with proposition bias at 82.6%). Proposition bias therefore appears sufficient to leverage structural signals: promoting a proposition into the final top-k when it appears among the top candidates by cosine avoids diluting or mis-scaling the semantic similarity signal; the 70/30 weighting may also be suboptimal or misaligned between score scales.

Entity detection: regex vs. NER. Entity spans in proposition extraction use the same lightweight pattern rules as Section 3.4. We replace them with a pre-trained NER system for boundaries, keeping the same ranking (cosine similarity with proposition bias). Table 7 reports GM-CIHT F1: NER reaches 81.5%, −1.1 below regex (82.6%). GM-CIHT often contains explicit markers, capitalization, and hyphenation that regex handles reliably; NER can introduce false positives or boundary errors that hurt minimal-span proposition extraction. NER may still help on corpora with less regular surface forms; the accuracy-latency trade-off also matters for deployment.

## 4.5 Performance of Semantic Chunking in Different Settings

We analyze dataset-dependent performance patterns to understand when semantic chunking is most effective.

Success on GM-CIHT and DDI. Both datasets contain diverse relation types with relatively clear semantic boundaries (e.g., TREATS vs. INHIBITS vs. COEXISTS WITH). Tiered trigger prioritization effectively disambiguates these categories, while entity-aware segmentation prevents fragmentation of complex biomedical terms. The observed gains on GM-CIHT (+8.4 F1) and DDI (+1.0 F1) indicate that semantic chunking is particularly beneficial for extraction tasks with explicit relation signals and moderate entity spacing.

ChemProt: Entity Density Challenges. ChemProt exhibits high entity density, with multiple chemicals and proteins often appearing within short spans (5–10 tokens). Entity-aware chunking can group multiple entity pairs into a single chunk (e.g., “compound X inhibits protein A and protein B”), increasing ambiguity during relation assignment. Moreover, ChemProt requires fine-grained biochemical distinctions (e.g., ANTAGONIST vs. DOWNREGULATOR) that rely on broader contextual cues beyond trigger words. Incremental ablations on ChemProt show an inverse pattern to GM-CIHT: adding tiered triggers and the tuned hierarchy can hurt F1, consistent with priorities designed for therapeutic relations misaligning with biochemical relation types. In this setting, fixed chunking may incidentally isolate simpler entity pairs, yielding slightly stronger performance.

ADE: Classification vs. Extraction. ADE is formulated as a binary classification task rather than structured relation extraction. Performance therefore depends on broad contextual signals such as symptom descriptions and temporal cues, rather than precise subject–trigger–object spans. Propositionfocused chunking may omit such supporting context, whereas fixed chunking preserves heterogeneous evidence useful for holistic classification.

Reproducibility. Our BioMedRAG Fixed GM-CIHT F1 (74.2%) is below the F1 reported in the original BioMedRAG publication (81.42%), likely due to differences in hardware, hyperparameters (e.g., shorter maximum sequence length when training the chunk scorer under GPU memory limits), and our own pipeline integration and dataset format unification. The comparison between fixed and semantic chunking remains meaningful because both conditions use the same scorer, generator, and preprocessing.

Generalization. Overall, semantic chunking is most effective when (i) relation types are coarse-grained and triggerexplicit, (ii) entity density is moderate, and (iii) the task emphasizes structured extraction over classification. Because our evaluation intentionally follows the four BioMedRAG benchmarks, these conclusions are best interpreted for sentence-level biomedical relation extraction over published literature rather than for clinical notes or document-level retrieval. The configuration-driven design improves transparency and reuse, but it still requires task-specific symbolic resources, such as trigger lexicons and relation hierarchies, which may limit scalability when transferring to many unseen biomedical domains. High-density corpora and longercontext settings may benefit from future extensions incorporating adaptive entity-pair isolation, pair-specific proposition selection, automatic trigger induction, or variable chunk granularity.

## 4.6 Error Analysis

We categorize the main failure modes of the proposed semantic chunking framework according to their likely root causes. First, errors occur when a relation is expressed without an explicit lexical trigger. In such cases, proposition-first extraction may not construct a complete subject–trigger–object span, and the framework relies on entity-aware or fixed fallback chunks. Second, high entity density can increase ambiguity, particularly in ChemProt, where multiple chemicals and proteins may appear within a short sentence. This makes it difficult to assign the correct relation to the correct entity pair. Third, fine-grained biochemical relations may require broader mechanistic context than a minimal proposition span provides. Relations such as AGONIST, ACTIVATOR, and DOWNREGULATOR can share similar surface cues while differing in biological meaning. Finally, ADE differs from the extraction datasets because binary adverse-event classification often depends on sentence-level context, including symptoms, temporal expressions, speculation, and negation. These failure modes explain why semantic chunking improves datasets with explicit relation cues and moderate entity density, while fixed-size chunking remains competitive for dense extraction and classification settings.

## 5 Conclusion

Fixed-size chunking in retrieval-augmented biomedical relation extraction frequently fragments entities and splits relation-bearing propositions, degrading the quality of retrieved evidence for downstream generation.

We introduced a configurable semantic chunking framework that combines entity-aware sliding windows with proposition-first extraction. Tiered trigger prioritization organizes relation signals by semantic specificity, while relation hierarchy resolution disambiguates competing interpretations by encoding task-specific priorities. All components are externalized in configuration files, enabling adaptation to new biomedical domains without code modification.

Our approach yields substantial gains on extractionfocused benchmarks, achieving +8.4 F1 points on GM-CIHT (82.6%) and +1.0 F1 on DDI (79.2%). Ablation studies show that relation hierarchy resolution contributes the largest in cremental gain in the GM-CIHT stack (+3.9 F1), and that semantically coherent chunks peak with a single in-context example (Table 5). The ranking ablation (Table 6) shows that proposition bias outperforms mixing structural scores into cosine ranking; the NER ablation (Table 7) shows that patternbased entity spans match or beat NER on GM-CIHT under our setup.

Absolute F1 for the fixed baseline is below published BioMedRAG figures, but the fixed-versus-semantic comparison is conducted under identical training and preprocessing and remains the primary evidence for our claims.

Analysis reveals that semantic chunking is most effective for datasets with coarse-grained relations and moderate entity density. In contrast, for high-density corpora (ChemProt) or classification-oriented tasks (ADE), fixed-size chunking remains competitive, as entity preservation may group multiple targets and proposition extraction can over-constrain contextual evidence.

Future work includes adaptive chunking strategies that respond to entity density, dataset-specific hierarchy design (or disabling hierarchy) for fine-grained biochemistry tasks, extension to clinical notes and drug labels, and entity-pairspecific isolation mechanisms to better handle multi-target scenarios. Future releases of the integration code, configuration files, and unified data conversion scripts will further support reproducibility and facilitate comparison with future BioMedRAG-based systems.

## Code and Data Availability

All datasets used in this study are publicly available through the BioMedRAG benchmark setting and the original dataset sources. To support reproducibility, we will release the semantic chunking implementation, dataset configuration files, preprocessing scripts, unified JSONL conversion format, and evaluation instructions upon publication. The configuration files include trigger vocabularies, tier assignments, relation hierarchies, negation rules, context markers, and chunking parameters. These resources are intended to reproduce the fixed-size and semantic chunking comparisons reported in this work under the same BioMedRAG scorer and generator pipeline.

## Acknowledgements

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – 527049502.

## References

[Chen et al., 2024] Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao, Hongming Zhang, and Dong Yu. Dense X retrieval: What retrieval granularity should we use? In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 15159–15177. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.emnlp-main.845. Available at https:// aclanthology.org/2024.emnlp-main.845/.

[Fundel et al., 2007] Katrin Fundel, Robert Kuffner, and¨ Ralf Zimmer. RelEx—relation extraction using dependency parse trees. Bioinformatics, 23(3):365–371, 2007.

[Gao et al., 2023] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2023.

[Grouin and Grabar, 2024] Cyril Grouin and Natalia Grabar. Year 2023 in biomedical natural language processing: A tribute to large language models and generative AI. Yearbook ofMedical Informatics, 33(1):241–248, 2024.

[Gunther¨ et al., 2024] Michael Gunther, Isabelle Mohr,¨ Daniel James Williams, Bo Wang, and Han Xiao. Late chunking: Contextual chunk embeddings using long-context embedding models. arXiv preprint arXiv:2409.04701, 2024. Available at https://arxiv.org/abs/2409.04701.

[Hosseini et al., 2024] Mohammad Javad Hosseini, Yang Gao, Tim Baumgartner, Alex Fabrikant, and Reinald Kim¨ Amplayo. Scalable and domain-general abstractive proposition segmentation. arXiv preprint arXiv:2406.19803, 2024.

[Hu et al., 2022] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022.

[Lewis et al., 2020] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen tau Yih,¨ Tim Rocktaschel, Sebastian Riedel, and Douwe Kiela.¨ Retrieval-augmented generation for knowledge-intensive NLP tasks. arXiv preprint arXiv:2005.11401, 2020. Also published in NeurIPS 2020.

[Li et al., 2025] Mingchen Li, Halil Kilicoglu, Hua Xu, and Rui Zhang. BiomedRAG: A retrieval augmented large language model for biomedicine. Journal of Biomedical Informatics, 162:104769, 2025. doi: 10.1016/j.jbi.2024.104769. Available at https://pubmed. ncbi.nlm.nih.gov/39814274/.

[National Library of Medicine, 2026] National Library of Medicine. PubMed. https://pubmed.ncbi.nlm.nih.gov, 2026. PubMed comprises more than 39 million citations for biomedical literature.

[Shuster et al., 2021] Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. Retrieval augmentation reduces hallucination in conversation. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 3784–3803, 2021. Available at https: //aclanthology.org/2021.findings-emnlp.320/.

[Zeng et al., 2014] Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou, and Jun Zhao. Relation classification via convolutional deep neural network. In Proceedings of COLING 2014, pages 2335–2344, 2014.

[Zhou et al., 2014] Deyu Zhou, Dingcheng Zhong, and Yulan He. Biomedical relation extraction: From binary to complex. Computational and Mathematical Methods in Medicine, 2014:298473, 2014.

[Zhou et al., 2016] Peng Zhou, Wei Shi, Jun Tian, Zhenyu Qi, Bingchen Li, Hongwei Hao, and Bo Xu. Attentionbased bidirectional long short-term memory networks for relation classification. In Proceedings of ACL 2016, pages 207–212, 2016.

[Zhou et al., 2023] Shuyan Zhou, Uri Alon, Frank F. Xu, Zhiruo Wang, Zhengbao Jiang, and Graham Neubig. DocPrompting: Generating code by retrieving the docs. In International Conference on Learning Representations (ICLR), 2023. Available at https://arxiv.org/abs/2207. 05987.

[Zhu and others, 2019] Yuhao Zhu et al. Graph neural networks with generated parameters for relation extraction. In Proceedings ofACL 2019, pages 1331–1339, 2019.