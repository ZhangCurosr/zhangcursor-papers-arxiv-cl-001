# TRACE: Target-Aware Retrieval, Attributed Evidence, and Contract-Constrained Extraction for LitTraceQA

Sachin Gupta Independent Researcher San Jose, USA sachinkg12@gmail.com

Divya Godara Independent Researcher San Jose, USA godaradivya@gmail.com

Accepted at the 1st Workshop on Grounding Language Models: Learning Faithfully and Efficiently (GroundLM 2026), co-located with EMNLP 2026.

## Abstract

Finding a relevant paper is not the same as producing a verifiable answer from it. LitTraceQA requires canonical paper identifiers, exact evidence at the page or object level, and typed answers that match the evaluator. We call the separation between source access and scorer-visible correctness the grounding contract gap. TRACE—Target-Aware Retrieval, Attributed Evidence, and Contract-Constrained Extraction—addresses this gap with target-grouped retrieval, independent typed evidence localization, multimodal table extraction, schema-driven table construction, and fail-closed validation. It indexes 27,487 papers through passage, object, alias, citation, and dense representations while retaining the question target behind each signal. For tables, TRACE predicts the observation unit before extracting values and assembles rows with evaluator-compatible key normalization. Our audited selected clean-track artifact scores 0.760613 on the official 71-question test set, including 0.9728 paper F1, 0.6847 evidence F1, 0.9800 multiple-choice accuracy, 0.5423 tablerow F1, and 0.3508 macro cell accuracy. On 11 public-development table records, a clean baseline and coordinate-aware visual fill obtain row F1 of 0.291 and 0.411, respectively; this diagnostic comparison includes fallback outputs and is not an official-test claim. Remaining errors chiefly concern locator, observation-unit, row-key, and source-value identity.

## 1 Introduction

Scientific question answering is often described as retrieval followed by generation. That abstraction is incomplete for LITTRACEQA (Liu et al., 2026). GroundLM 2026 pairs scientific-evidence grounding in LITTRACEQA with view-level visualevidence identification in GoldenViewVQA (Wang et al., 2026a); the shared-task findings paper describes the joint evaluation setting (Wang et al.,

2026b). We participate only in LITTRACEQA, under the official evaluator team name gabby. Given a research question and a fixed metadata pool, the system must produce a connected trace

$$
q \longrightarrow \widehat { P } \longrightarrow \widehat { E } _ { \widehat { P } } \longrightarrow \widehat { A } ,\tag{1}
$$

where $\widehat { P }$ is a set of canonical paper IDs, $\widehat { E }$ is a set of typed source locators inside those papers, and $\widehat { A }$ is one or more schema-constrained answers. The components are scored separately, and evidence must close over the selected paper set. A semantically correct answer can therefore score poorly when a page is adjacent to the annotated page, a printed object ID is normalized incorrectly, or a source value is placed under a paraphrased row key.

Our first pipeline had all three pathologies. It collapsed retrieval results for several named methods into one global rank, emitted only evidence quoted by the chosen answer path, and treated every selected paper as exactly one table row. These were not model-capacity failures. They were representation failures at the interfaces between retrieval, grounding, and structured generation. We call the resulting separation between information available in the corpus and information emitted in evaluatorvisible form the grounding contract gap.

We organize the system study around three questions:

RQ1: Which retrieval representation preserves the identity and cardinality of the requested paper targets?

RQ2: Why can source-correct evidence and values still fail structured evaluation, and which decomposition reduces that loss?

RQ3: Which execution contracts prevent operational errors from silently becoming valid but scoredead submissions?

Our contributions are:

• a target-aware paper selector that retains route, target group, group-local rank, route score, and semantic role instead of flattening heterogeneous evidence;

• an independent typed localizer over PDF text and detected tables, figures, equations, and citation contexts, with exact scorer-key normalization and paper–evidence closure;

• an observation-unit-first table pipeline that plans the row axis, extracts visual and textual values, and assembles typed rows deterministically; and

• an artifact-level audit protocol covering corpus/index alignment, semantic fallbacks, official validation, hashes, source revision, and recorded selected artifacts.

Code, configurations, and release-audit materials are available at https://github.com/ sachinkg12/trace.

## 2 Task, Data, and Related Work

## 2.1 The three-part output contract

LitTraceQA evaluates the complete path in Equation 1, rather than treating citations as optional explanations after answering (Liu et al., 2026). Its evidence types include text spans, tables, figures, equations or algorithms, and citation contexts. Paper and evidence sets receive macro precision, recall, and F1; multiple-choice answers receive accuracy; and structured tables receive row F1 plus macro and micro cell accuracy.

Our test input has 71 records: 50 request a multiple-choice component and 21 request a table component; no test record requests both. The candidate pool contains 27,487 papers. This scale is modest for first-stage retrieval but difficult for exact trace construction: the same question can name answer targets, evidence anchors, datasets, venues, and comparison criteria, only some of which denote papers that should be returned.

## 2.2 Corpus and indexes

We retain the organizer paper identifier as the primary key and store released PDFs in Google Cloud Storage (GCS). PyMuPDF (Artifex Software, Inc., 2026) extracts page text and structured blocks and renders pages for visual analysis. The frozen corpus build contains 1,409,382 passages, 500,928 detected objects, 3,711,590 aliases, and 492,892 citation/relation records. A

384-dimensional BAAI/bge-small-en-v1.5 (Xiao et al., 2023) cache embeds every title– abstract pair. Pool order, record counts, model identity, and file hashes are checked as invariants before inference.

## 2.3 Relation to prior systems

Our retrieval stage combines sparse and dense signals in the spirit of retrieval-augmented generation (Lewis et al., 2020), BM25 (Robertson and Zaragoza, 2009), and rank fusion (Cormack et al., 2009); our main finding is that fusion must follow, not erase, target grouping. For table construction, prior work shows that scientific cell identity depends on surrounding table and paper context (Lou et al., 2023), and that schema-constrained extraction can outperform unconstrained generation across heterogeneous tables (Bai et al., 2024). ScheMatiQ makes the observation unit explicit before discovering and populating a question-specific schema (Levy et al., 2026). We adapt this principle to a fixed evaluator schema: the row entity is inferred from the question and row-key columns, while all accepted rows and locators must remain compatible with the shared-task contract.

## 3 System Architecture

Figure 1 separates development and checkpoint selection from the recorded inference path. The system is not a monolithic agent: generative components propose typed intermediate objects, and deterministic components enforce identity, cardinality, and serialization contracts.

## 3.1 Question plan and retrieval signals

Stage 2 uses a temperature-zero planner to extract venue/year constraints, required modalities, multiplicity, desired paper count, named anchors, and target groups. Each target receives a semantic role: answer target, evidence anchor, or constraint. This prevents, for example, a dataset mentioned in an experimental criterion from being returned as though it were a requested paper. Invalid planner output produces a visible conservative fallback rather than aborting or silently dropping the record.

Stage 3 queries five complementary route families: (i) name, alias, and title resolution; (ii) BM25 property and target-property search over passages; (iii) citation-context search; (iv) role-aware ownership signals; and (v) dense title–abstract retrieval. Each candidate carries supporting snippets and route signals $( r , g , k , s , \rho )$ : route r, target group $^ { g , }$ group-local rank k, optional route score s, and role $\rho .$ Detected-object records are available to later localization and diagnostics, but the selected run did not activate the object index as a paper-retrieval route.

![](images/19d8309e4713ff362b2d3f624faa8514db437cd47a6cfccadc52a88a35180446.jpg)  
Figure 1: TRACE carries a typed request through an internal target-group contract to scorer-visible canonical papers P, evidence locators $E ( P )$ , and answer A. Input badges denote optional multiple-choice choices, an optional table schema, and the required answer type. A detached development-and-selection rail records public-development gates, aggregate-score checkpoint selection, source audits, and hashes. Inference uses released inputs and the corpus; hidden labels and per-question evaluator feedback are unavailable.

The group-local rank is essential. When results for four named methods are concatenated, global rank 35 may be rank 1 for the fourth method. Flat reciprocal-rank fusion sees weak evidence; the grouped representation sees strong coverage of a distinct requested target. Stage 4 first applies answer-scoped venue/year constraints and then covers target groups before spending remaining slots on evidence-bearing relevance. Explicit multiplicity determines the desired count, while a hard cap of five bounds PDF and vision work.

## 3.2 Independent evidence localization

Stage 5 fetches each selected PDF and creates a common source representation. Text spans keep page numbers; visual objects keep printed identifiers; references keep citation IDs and contexts. We normalize forms such as “Eq. (6)” and “Reference [24]” while rejecting arbitrary prose numbers as object IDs.

At stage 6, evidence is retrieved independently of the answerer. Gemini 2.5 Flash examines bounded PDF text to propose typed locators. Gemini 2.5 Pro vision (Gemini Team, Google, 2025) is used downstream for visual table and figure answer extraction, not for evidence localization. Candidates are ranked by structural validity, localizer confidence, BM25 relevance, and stable source order. They are deduplicated under the evaluator’s coarse key— paper, source type, page or section, and normalized object ID—before a cap of five is applied. This replaces answer-conditioned evidence, which had high apparent precision but omitted valid support not quoted by the final answer path.

## 3.3 Observation-unit-first tables

The table schema is not merely an output template; it defines what one row means. Stage 7 classifies the observation unit from every column marked as a row key. Paper Title indicates a paper axis; Method, Benchmark, Author, Reference, or a multi-key schema indicates an entity axis. When a question lists the desired entities, the planner constructs a closed row ledger from their shortest literal names.

Extraction then runs per source, not per output row. Visual pages and non-visual text, equation, and citation paths propose compatible rows and typed value cells. The assembler groups by the same normalized row-key tuple as the scorer, merges complementary cells across papers, removes all-null records, and deduplicates. For the selected clean track, deterministic row-key matching preserves non-null baseline cells and permits only source-scoped, schema-compatible fills. Repeated visual proposals are accepted only when they agree; otherwise the baseline table is retained.

![](images/4f8d7be8b47a559dd40ef2b052cfe74bb8bee5227515975e17df2a62d5b98e0e.jpg)  
Figure 2: Observation-unit-first table construction. Schema keys define the row contract; visual and native sources propose source-scoped rows and cells; deterministic assembly normalizes keys, merges compatible cells, type-checks, and deduplicates. Exact PDF attestation is additionally required for any post-generation correction. The paper is a source container, not necessarily the row entity.

Public-development example. One multi-paper question requests four named methods (TCM, sCT, ECM-XL, and IMM) under an experimental criterion. The legacy generator created one row per paper, confusing the paper containing a comparison with the methods being compared. The planned system instead fixes the four methods as candidate row identities, searches each method–criterion tuple, and accepts only rows whose keys normalize to the requested set. The example illustrates the broader distinction between source container and observation unit.

## 3.4 Fail-closed execution

Stages 1 and 8 make operational validity part of the architecture. Input normalization accepts object- and list-shaped multiple-choice options while preserving organizer labels and exact table schemas. Strict preflight checks 71 unique query IDs, 27,487 pool IDs, answer-type counts, all four multiple-choice labels, index record and byte counts, dense-cache shape/model, and official file hashes. The runner records timeouts, semantic fallbacks, repaired components, active configuration, and raw traces. It writes atomically only after paper–evidence closure and the pinned organizer validator pass.

Table 1 summarizes the design as recovered structure rather than a list of prompts.

## 4 Experimental Setup

Models and infrastructure. The hybrid pipeline uses Python 3.11, PyMuPDF, local BM25 indexes, and sentence-transformer embeddings. Index construction, retrieval scoring for a fixed plan, normalization, composition, and validation are deterministic; hosted generative proposals are not. Gemini 2.5 Flash performs question planning, text localization, and answer generation; Gemini 2.5 Pro is reserved for visual table and figure extraction. The corpus/index store is on GCS, and bounded workers with per-record deadlines run on a Google Cloud VM. Temperature is zero where supported, and raw replies are retained because hosted services can still vary between calls. No task-specific model weights are trained or fine-tuned.

Development and freezing. Architecture choices and local gates use the 55-example public development split. The table analysis below evaluates the same 11 public-development records in both conditions with organizer-compatible normalization. Aggregate official evaluator scores informed checkpoint selection, but hidden labels and per-question evaluator feedback were unavailable and do not enter the correction policies. The selected official artifact has 71 unique records, passes the pinned validator, and has SHA-256 prefix 2dd56b6009. It extends the clean 0.757968 checkpoint with three disjoint, source-attested record repairs: two table-object relocations from mention pages to their unique caption pages and one canonical method-key repair. The terminal policy revision a4962b9 preserves the other 68 serialized records byte-for-byte. Each accepted delta is derived from released questions, clean full-run traces, and URL/SHA-bound PDFs; the reports bind predecessor artifacts, source revisions, runtime archives, PDF manifests, the pinned validator, and the final digest. Projected and unscored variants are excluded from reported results.

## 5 Results

## 5.1 Official result and score anatomy

Table 2 reports every metric returned for the selected clean-track official artifact. Paper retrieval and multiple-choice answering are the strongest components, but neither is saturated. Figure 3 shows the largest scorer-visible headroom in evidence and table metrics; the analysis below discusses likely identity-related causes.

<table><tr><td>Observed failure</td><td>Information lost</td><td>Architectural response</td><td>Enforced invariant</td></tr><tr><td>Flat fusion misses later named methods</td><td>Target identity and group-local rank</td><td>Route signals preserve  $( r , g , k , s , \rho ) ;$  selection covers groups</td><td>Selected papers retain contributing route signals and target-group</td></tr><tr><td>Answer-coupled evidence under-recalls</td><td>Support not quoted by the answer path</td><td>Independent typed PDF-text localization; visual extraction remains separate</td><td>metadata Every locator closes over an emitted paper and a valid source type</td></tr><tr><td>One row per paper collapses methods/components</td><td>Observation unit and row cardinality</td><td>Row-axis classification, row ledger, multi-row extraction</td><td>Row keys and cell types conform to the supplied schema</td></tr><tr><td>Plausible values miss exact scoring</td><td>Printed form, page, or object identity</td><td>Source-conditioned candidates; PDF-exact attestation for accepted corrections</td><td>Normalization and assembly use scorer-parity keys; corrections are</td></tr><tr><td>Valid JSON hides empty behavior</td><td>Semantic success and corpus readiness</td><td>Strict preflight, fallback telemetry, atomic output</td><td>source-bound Counts/hashes pass; repairs and placeholders remain observable</td></tr></table>

Table 1: The main failure modes and the structure restored by TRACE. The common pattern is that a downstream model cannot reconstruct information already discarded at an upstream interface.

<table><tr><td>Component</td><td>Precision</td><td>Recall</td><td>F1 / Acc.</td></tr><tr><td>Papers</td><td>0.9859</td><td>0.9683</td><td>0.9728</td></tr><tr><td>Evidence</td><td>0.6594</td><td>0.7582</td><td>0.6847</td></tr><tr><td>Multiple choice</td><td></td><td></td><td>0.9800</td></tr><tr><td>Table rows</td><td></td><td></td><td>0.5423</td></tr><tr><td>Table cells (macro)</td><td></td><td></td><td>0.3508</td></tr><tr><td>Table cells (micro)</td><td>一</td><td></td><td>0.4023</td></tr><tr><td>Official composite</td><td></td><td></td><td>0.760613</td></tr></table>

Table 2: Official test metrics for the selected clean-track TRACE submission. Free-form exact match was not applicable.

The selected result supersedes, but does not erase, the immediately preceding 0.757968 checkpoint or the earlier clean results at 0.739561, 0.735774, and 0.710662. All remain in the verified clean-result history.

## 5.2 Public-development table analysis

Table 3 compares a clean general table baseline with the clean framework plus coordinate-aware visual fill on the same 11 public-development table records. The framework preserves target structure through source-container selection, then fills only schema-compatible missing cells from repeated visual agreement. Row F1 rises by 0.120, macro cell accuracy by 0.125, and micro cell accuracy by 0.259. An execution failure denotes a schemavalid placeholder rather than a usable table; failed records were not removed. The baseline has three such failures and the coordinate-fill candidate has two, and the metrics score all 11 records. Candidate evidence F1 falls from 0.460 to 0.397. We therefore treat this as public-development diagnosis, not a promotion gate or evidence about official-test improvement.

![](images/c0e976c4f83cfe2bf1004fa9677fb61ace0c92eb243193c7a6f5d6cad778c484.jpg)  
Figure 3: Official score profile, including both table-cell accuracies. Metrics have different definitions and are shown together only to locate residual headroom, not as an additive decomposition of the composite.

## 6 Analysis: Where Correctness Is Lost

RQ1: Preserve target identity before ranking. Target grouping, scoped constraints, and desired cardinality solve different parts of paper selection. The selected system’s paper F1 of 0.9728 makes retrieval its strongest trace component while leaving measurable headroom. The important negative result is flat fusion: no route weighting can recover which method produced a candidate after that group identity has been discarded. Representation precedes ranking.

<table><tr><td>System</td><td>Paper F1</td><td>Evidence F1</td><td>Row F1</td><td>Cell macro</td><td>Cell micro</td></tr><tr><td>General clean baseline</td><td>0.556</td><td>0.460</td><td>0.291</td><td>0.262</td><td>0.148</td></tr><tr><td>Framework + coordinate fill</td><td>0.594</td><td>0.397</td><td>0.411</td><td>0.387</td><td>0.407</td></tr></table>

Table 3: Public-development diagnostics on all 11 table-answer records. Generation failures emitted schema-valid placeholder rows and remained in scoring; the baseline failed on 3/11 records and the coordinate-fill candidate on 2/11. These are not official-test metrics.

RQ2: Source access does not guarantee exact identity. Paper F1 exceeds evidence F1 (0.9728 versus 0.6847), while macro cell accuracy is 0.3508. For evidence, an adjacent page or semantically equivalent source type is still the wrong locator tuple. For tables, a correct number under a descriptive rather than canonical row key is still the wrong cell. The observation-unit decomposition improves the latter, but aliases, multi-component rows, symbols, delimiters, and typed nulls remain difficult. Our recall exceeds precision for evidence, so adding more locator candidates without stronger exact corroboration is unlikely to be the best next step.

RQ3: Contracts turn silent failures into actionable ones. Several severe bugs produced schema-valid output: empty indexes, constant-label multiple-choice fallbacks, all-null rows, and repaired answer objects. Record counts, content hashes, closure checks, and semantic fallback telemetry distinguish these cases from genuine model errors. The lesson is broader than this task: when a pipeline combines retrieval, hosted models, PDF rendering, and structured output, the artifact boundary must be tested as rigorously as the model.

What did not work. Unbounded multi-paper plans propagated dozens of PDFs into per-paper vision and exhausted deadlines. Answer-conditioned evidence improved apparent precision but cut recall. Flat reciprocal-rank fusion promoted generic high-BM25 papers over later named targets. Unconstrained table generation paraphrased row keys and attached baseline values to the container paper. More model calls did not repair these losses reliably; explicit intermediate identities did. The four-record extension from 0.739561 to 0.757968 provides a sharper positive result: paper F1 rises from 0.9587 to 0.9728, evidence F1 from 0.6753 to 0.6847, row F1 from 0.4709 to 0.5185, and macro cell accuracy from 0.3032 to 0.3508, while multiple-choice accuracy remains 0.9800. The macro table increase of 0.0476 is consistent with one 1/21 record-level increment and comes from binding requested metric rows and columns to physical PDF coordinates before canonicalizing decorative direction glyphs. The evidence gain comes from source-local paper correction and one uniquely re-localized answer-bearing page, rather than increasing candidates globally. Source attestation is necessary, but table and evidence changes additionally require scorer-contract identity and a fail-closed mutation boundary. The final threerecord extension raises only row F1, from 0.5185 to 0.5423; paper, evidence, multiple-choice, and cell metrics are unchanged. This isolates the observed gain to canonical method identity, while the two caption-page relocations are source-correct but evaluator-neutral.

## 6.1 Lessons from adaptive diagnosis

A historical adaptive artifact reached 0.792252 only after repeated official-score diagnosis and manual row/cell adjudication. It is not a comparable system result and is excluded from our selected clean-track claim. Its value is diagnostic: the gap showed that source access and model capacity were not the principal bottlenecks; exact paper ownership, locator identity, observation units, row keys, and cell coordinates were. Those findings motivated the identity ledger, source-object contracts, and fail-closed mutation boundaries described here. We therefore report the number once as a lesson about system interfaces, not as the headline performance of TRACE.

## 7 Discussion and Reproducibility

The prior 0.757968, 0.739561, 0.735774, and 0.710662 clean checkpoints remain in the result ledger rather than being relabeled as adaptive or discarded. The selected 0.760613 artifact adds only generic, fail-closed corrections: exact table-caption localization and canonical method identity derived from immutable PDFs and the released schema. Its composition applies complete records by query ID, rejects overlapping chains, and preserves 68 of 71 records from the preceding clean checkpoint bytefor-byte; no hidden labels or per-question evaluator feedback enter these policies. Aggregate official scores did inform checkpoint selection, which we distinguish from the source-only inputs to each correction policy.

The public full-generation code implements and can execute the base architecture from released inputs and source assets, but hosted proposals may vary and the released configuration does not regenerate the exact selected artifact. In particular, the selected base used planned\_visual\_fill table extraction, whereas the public general configuration uses planned; the selected correction chain is retained in the audit package rather than exposed as the public full-generation command. An independent run of that public configuration matched 37 of 71 selected records as complete serialized objects, with most variation arising in table construction; we therefore do not claim exact output or score reproducibility. The exact 0.760613 JSONL is uploaded separately with the paper and is reported as an audited selected artifact; our retained audit package binds predecessor reports, source revisions, PDF manifests, output digests, and validator status.

## 8 Conclusion

The audited selected clean-track TRACE artifact reaches 0.760613 by preserving scorer-visible paper, evidence, and answer identities through target-aware retrieval, independent localization, observation-unit-first rows, and fail-closed validation.

## 9 Limitations

The public-development table analysis covers only 11 records and cannot establish official-test improvement. The fixed five-paper cap can underrecall unusually large answer sets. PDF parsing remains brittle for scans, complex multi-column layouts, and objects whose printed identifiers are absent. Exact evidence evaluation can penalize semantically valid adjacent support; the LitTraceQA task paper itself notes that locator normalization remains a benchmark-design challenge (Liu et al., 2026). Hosted model APIs introduce cost, version drift, and limited repeatability despite constrained prompts and retained traces. We perform no weight updates, task-specific fine-tuning, synthetic training, or external factual augmentation. The system operates only on the released scientific-paper pool, but its predictions can still misattribute claims or overstate support and should not replace inspection of the cited source.

## Acknowledgments

We thank the GroundLM 2026 organizers for organizing the shared task and maintaining its evaluation infrastructure.

## References

Artifex Software, Inc. 2026. PyMuPDF: A highperformance Python library for data extraction, analysis, conversion and manipulation of PDF documents. https://pymupdf.readthedocs.io/. Version 1.28.0.

Fan Bai, Junmo Kang, Gabriel Stanovsky, Dayne Freitag, Mark Dredze, and Alan Ritter. 2024. Schemadriven information extraction from heterogeneous tables. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 10252– 10273. Association for Computational Linguistics.

Gordon V. Cormack, Charles L. A. Clarke, and Stefan Buettcher. 2009. Reciprocal rank fusion outperforms condorcet and individual rank learning methods. In Proceedings of the 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 758–759.

Gemini Team, Google. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. Technical report.

Shahar Levy, Eliya Habba, Reshef Mintz, Barak Raveh, Renana Keydar, and Gabriel Stanovsky. 2026. ScheMatiQ: From research question to structured data through interactive schema discovery. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 3: System Demonstrations), pages 220–230. Association for Computational Linguistics.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive NLP tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459– 9474.

Xuye Liu, Yimu Wang, Peng Shi, Bo Xue, Xiangrui Ke, Songcheng Cai, Kath Choi, Di Wu, Freda Shi, and Krzysztof Czarnecki. 2026. LitTraceQA: A benchmark for multi-stage grounding and verification in scientific question answering. Preprint, arXiv:2608.07370.

Yuze Lou, Bailey Kuehl, Erin Bransom, Sergey Feldman, Aakanksha Naik, and Doug Downey. 2023. S2abEL: A dataset for entity linking from scientific tables. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3089–3101. Association for Computational Linguistics.

Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4):333–389.

Yimu Wang, Yee Man Choi, Barry Zhang, Mozhgan Nasr Azadani, Sean Sedwards, and Krzysztof Czarnecki. 2026a. Where does the answer come from? benchmarking view-level visual evidence identification in multi-view MLLMs for autonomous driving. Preprint, arXiv:2606.09644.

Yimu Wang, Xuye Liu, Yee Man Choi, Bo Xue, et al. 2026b. Findings of the first GroundLM shared tasks: Evaluating grounded language models across visual and scientific evidence. In Proceedings of the 1st Workshop on Grounding Language Models: Learning Faithfully and Efficiently (GroundLM 2026).

Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. 2023. C-Pack: Packed resources for general Chinese embeddings. arXiv preprint arXiv:2309.07597.

## A Reproducibility Checklist

The public release includes implementation code, a general configuration, result documentation, and validation instructions; the OpenReview submission includes the ACL-formatted PDF and exact 71-line selected prediction file. Our retained audit package additionally includes SHA-256 manifests, source revisions, resolved selected-run configuration, preprocessing/index statistics, predecessor reports, and validator records. The historical adaptive artifact and public-development table analysis are separately labeled and excluded from the selectedsystem claim. The official ACL style files are used without modifications to margins, spacing, fonts, or page dimensions.