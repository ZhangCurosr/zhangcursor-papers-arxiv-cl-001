# Do Agents Need a New Kind of Paper? SPRA: Structured Paper Representation for Agents

Yutong Leng<sup>1,2</sup> Haojiong Chen<sup>1</sup> Fukai Shang<sup>1</sup>

Xiaoyang Wang<sup>1</sup> Mindong Liu<sup>1</sup> Runyuan Ma<sup>1</sup> Jiantao Qiu<sup>1,∗</sup>

<sup>1</sup>Shanghai Artificial Intelligence Laboratory

<sup>2</sup>Shanghai Jiao Tong University

冰 Corresponding author.

Website · Paper Schema API · Agent Skill · Extraction Code

## Abstract

Scientific agents must connect research problems, methods, and findings dispersed through narrative articles. We introduce SPRA (Structured Paper Representation for Agents), a paper-level representation of persistent research objects and typed relations. Its empirical-AI profile organizes Problems, Contributions, Components, ExperimentSetups, Measures, and Findings for querying, traversal, and reuse. Selected paper-level citation relations record dependencies, component use, and comparisons with prior work; source locations support verification. We instantiate SPRA across more than one million AI papers, with 19.95 million research units and 24.07 million internal relations. Evaluations access SPRA through Schema-RAG with task evidence fixed before inference.

Across seven subfields, SPRA improves sourceconditioned scientific QA by 0.381/5 over passage retrieval while remaining close to Full Compact (0.078/5 lower on average). In directed cross-paper reasoning, qualification improves by 0.184/5 over passage retrieval; Top-5 synthesis shows mixed quality effects. Across all paired RQ2 tasks, including Top-5, input+output tokens decrease by 28.5% relative to passage retrieval and 95.3% relative to Full Compact. In idea generation, a 0.067- point lower Overall score than Full Compact accompanies 93.7% fewer input+output tokens. Together, these results support SPRA as a usable representation of scientific content and a foundation for research workflows that connect, inspect, and reuse paper-level research objects. The SPRA resource is accessible through the SciVerse scholarly-data platform.

![](images/d15f5bacc31f45dbacf75f9bc6d1d44699f1e733eb3402c44982ff986c350462.jpg)  
Figure 1: Research objects and typed relations in the empirical-AI profile. Problem, Method, and Evidence sections encode the paper’s research argument; arrows show relation directions. Units and relations carry source bindings for verification.

## 1 Introduction

AI paper submissions are growing rapidly (ICLR, 2026), and scientific agents increasingly participate in literature discovery, comparison, and synthesis (He et al., 2025; Asai et al., 2026). The same paper can inform method comparisons, evaluation queries, and research proposals. These operations require its problems, contributions, evaluations, and findings, together with their relationships. Conventional articles express this structure through narrative text, tables, and figures; agents must connect findings to methods, conditions, and comparisons. How should a paper expose these scientific relationships for agents to query, compare, and reuse?

Passage-based retrieval supplies relevant source text (Lewis et al., 2020), while task-specific synthesis assembles evidence for a question (Dasigi et al., 2021; Lála et al., 2023; Skarlinski et al., 2024). Research roles and dependencies may still need to be reconstructed from these passages. Repeated full-paper inputs incur token-based API charges (Chen et al., 2023) and occupy context needed for additional papers and task history. Larger context windows also need not ensure reliable evidence use: middle-position information can be underused, with positional attention bias implicated in this behavior (Liu et al., 2024; Hsieh et al., 2024). These costs motivate evaluating retained scientific utility jointly with token use. Compact, explicit research structure makes scientific connections directly available while preserving capacity for linked evidence.

We introduce SPRA, the Structured Paper Representationfor Agents. SPRA derives its research roles and predicates from paper understanding, method comparison, and evidence integration, representing the connections among a contribution, its evaluation, and its reported findings. Paperscoped identifiers let successive operations refer to the same stored objects; internal relations connect contributions to their evaluation and reported outcomes, and a separate citation layer records selected dependencies on prior work. Figure 1 illustrates SPRA; Schema-RAG is one access mechanism used to evaluate this representation and resource.

Consider the speech-emotion machineunlearning paper of Ren et al. (2025). Its privacy-cost Problem, forget-set-only Contribution, unlearning-accuracy Measure, and baselinecomparison Finding form a research argument. An evaluation question selects the linked Measure and Finding; a method comparison selects corresponding records from another paper. To check the reported advantage, an agent can follow these connections to the evaluation conditions and reopen the associated source evidence.

The contributions are:

1. a paper-level representation that combines persistent research objects, typed internal relations, citation dependencies, and source bindings, with an extraction and export workflow (Section 3);

2. a million-paper resource, programmatic access, and aligned representation inputs and frozen evaluation outputs (Table 2); and

3. paired evaluations of scientific QA, crosspaper reasoning, and idea generation that quantify answer quality and task-level token use (Sections 4–5).

## 2 Related Work

Scholarly documents and provenance. GRO-BID, S2ORC, and MinerU extract metadata, text, and layout (Lopez, 2009; Lo et al., 2020; Wang et al., 2024). JATS provides markup; PROV-O models provenance; PROV-CONSTRAINTS defines validity; RO-Crate packages research objects (National Information Standards Organization, 2024; Lebo et al., 2013; Cheney et al., 2013; Soiland-Reyes et al., 2022; Research Object Community, 2026).

Research roles and scientific extraction. Argumentative Zoning and CoreSC label scientific text by research function (Teufel et al., 2009; Liakata et al., 2012). SciERC and DyGIE++ model entities, relations, and events (Luan et al., 2018; Wadden et al., 2019); SciREX, NLPContributionGraph, SciER, and SciEvent represent document-level relations, contribution triples, or scientific events (Jain et al., 2020; D’Souza et al., 2021; Zhang et al., 2024; Dong et al., 2025). SPRA assembles research roles and extracted content into persistent paper-owned objects with typed connections and executable operations.

<table><tr><td>Relation</td><td>Scientific role</td><td>Source → Target</td></tr><tr><td> $\mathsf { p a r t \_ o f }$ </td><td>Component or setup membership</td><td> $\{ \mathrm { C o } , \mathrm { S } \}  \{ \mathrm { C } , \mathrm { C o } , \mathrm { S } \}$ </td></tr><tr><td>co_contribution</td><td>Joint contributions</td><td> $\mathrm { C }  \mathrm { C }$ </td></tr><tr><td>compares_to</td><td>Comparison of methods or measures</td><td> $\{ \mathrm { C } , \mathrm { C o } , \mathrm { S } , \mathrm { M } \}  \{ \mathrm { C } , \mathrm { C o } , \mathrm { S } , \mathrm { M } \}$ </td></tr><tr><td>evaluates</td><td>Measure applied to a method</td><td> $\mathrm { M }  \{ \mathrm { C } , \mathrm { C o } \}$ </td></tr><tr><td>about</td><td>Subject of a finding</td><td> $\mathrm { F }  \{ \mathrm { \bar { C } , C o , \bar { S } , M } \}$ </td></tr><tr><td>supports</td><td>Provides evidence links from research objects to Findings</td><td> $\{ \mathrm { M } , \mathrm { F } , \mathrm { C } , \mathrm { C o } \}  \mathrm { F }$ </td></tr><tr><td>motivates</td><td>Problem motivating a method or setup</td><td> $\mathrm { P }  \{ \mathrm { C } , \mathrm { C o } , \mathrm { S } \}$ </td></tr><tr><td>resolves</td><td>Finding addressing a problem</td><td> $\mathrm { F }  \mathrm { P }$ </td></tr></table>

Table 1: Paper-internal relations and admissible endpoint signatures. P = Problem; C = Contribution; Co = Component; S = ExperimentSetup; M = Measure; F = Finding. Braces denote allowed types. Construction stages are given in Appendix A.3.

Structured scholarly knowledge. ORKG supports open scholarly aggregation through paperlinked contributions, templates, comparisons, APIs, and ASK retrieval (Jaradeh et al., 2019; Open Research Knowledge Graph, n.d.; Oelen et al., 2024). SPRA derives roles and predicates from agent tasks and provides an extraction, modeling, and export workflow for paper-owned research objects. These goals are complementary: property alignment connects task-oriented objects to ORKG aggregation. Appendix A.7 describes this engineering interface.

Structured content for scientific assessment. SciCore combines full-manuscript review with a judgment of an extracted science core, evaluating stability under content-preserving rewrites alongside discrimination across papers (Wang et al., 2026). SciCore motivates an explicit view of scientific content for automated assessment. SPRA develops a persistent paper-level representation for broader research operations: paper-owned units, typed relations, and source bindings support querying and comparison across tasks.

Agent access to scientific literature. PaperQA/- PaperQA2, OpenScholar, and PaSa support literature retrieval and cited synthesis (Lála et al., 2023; Skarlinski et al., 2024; Asai et al., 2026; He et al., 2025); ResearchAgent iteratively develops research ideas (Baek et al., 2025). GraphRAG builds entity communities for query-focused summarization, and $\mathrm { { K G ^ { 2 } R A G } }$ expands and organizes retrieved chunks through graph connections (Edge et al., 2024; Zhu et al., 2025). Intern-Atlas exposes method evolution for graph-guided ideation (Wu et al., 2026). SPRA supplies a persistent paperlevel research representation that these workflows can query and compose.

## 3 SPRA: Paper-Level Research Representation

## 3.1 Representation and Profiles

The construction input is a source article parsed into section-aware Markdown with source markers. The output is a paper-scoped record of research units, typed relations, citation dependencies, and source bindings. Given a research question, an agent selects and connects these stored objects to assemble its evidence input.

SPRA stores research objects independently of a particular query. Structured JSON encodes each instance; a versioned profile specifies scientific roles, relation signatures, cardinalities, and sourcelinking rules.

The representation follows the paper’s scientific argument: a Problem motivates a Contribution; Components and ExperimentSetups describe what is built and tested; Measures define evaluation targets; and Findings record reported observations. These six roles organize the empirical-AI profile. Normally one focused Problem expresses the central research question, with a second permitted for an independent question. Finding attributes record applicable conditions and qualifications. Table 1 specifies the relations connecting these objects and their admissible endpoints; citation roles express external dependencies in a separate layer.

<table><tr><td>Statistic</td><td>Scope</td><td>Value</td></tr><tr><td>AI papers</td><td>AI literature; 1965–2026</td><td>1,059,112</td></tr><tr><td>Core units</td><td>Corpus; 18.84/paper</td><td>19,949,289</td></tr><tr><td>Typed relations</td><td>Corpus; 22.73/paper</td><td>24,073,381</td></tr><tr><td>Evaluation sample</td><td>7 subfields; 2025; 1,407 each</td><td>9,849</td></tr><tr><td>Locator references</td><td>Evaluation sample</td><td>409,082</td></tr></table>

Table 2: Corpus scale and evaluation scope.

The empirical-AI profile instantiates this contract for method and evidence comparison. Profiles make the representation extensible while retaining explicit role and relation constraints.

For paper $p ,$ write $S _ { p } = ( U _ { p } , R _ { p } , C _ { p } , L _ { p } , V _ { p } ) \colon$ typed units, internal relations, citation dependencies, source bindings, and source/profile revisions. $C _ { p }$ joins selected citation roles and verbatim signals to bibliographic metadata by citation key. Under profile $\pi ,$ an internal edge $( u , r , v ) \in R _ { p }$ is valid only if $u , v \in U _ { p }$ and their types satisfy r’s permitted signature. For example, motivates maps a Problem to a Contribution, Component, or ExperimentSetup. The supports predicate provides evidence links from research objects to a Finding, enabling agents to retrieve associated content and inspect available source evidence. Schema validity checks the relation contract, and source resolution checks whether a recorded address reaches a source item.

Citation dependencies express how the focal paper uses prior research. builds\_on identifies methodological dependencies, uses\_component records adopted components, and compares\_with records comparisons with cited work. Citation keys join these roles and their verbatim signals to bibliographic records, enabling dependency queries and attribution checks. These paper-level dependencies connect the focal research argument to the prior work it uses.

Running example. In the unlearning case, the Measure evaluates the forget-set-only Contribution and the Finding records its comparison with a baseline. Paper ownership assigns the proposed method to the focal paper; citation roles identify external work. An agent can retain these objects for comparison and reopen their source locators to check the evaluation conditions. Appendix C gives the records and source bindings.

## 3.2 Corpus and Community Access

SciVerse is the scholarly-data platform that hosts the SPRA collection and provides web and programmatic access to paper records. SPRA denotes the research representation defined in this paper; SciVerse provides its deployed resource and access services.

The September 2026 SPRA snapshot contains 1,059,112 structured paper records spanning 1965– 2026, drawn from AI conference literature and preprints, including arXiv. Each paper is processed through the construction workflow below into a stored SPRA instance, accessible through the structured-paper services described in Section 3.3.

The evaluation sample comprises 9,996 candidate papers, of which 9,849 are source-aligned and eligible. This domain-stratified sample covers 2025 publications, with 1,407 eligible papers in each of seven AI subfields (Table 2). Structural checks and task evaluations therefore assess deployed representations from the same construction workflow, with balanced coverage across the selected fields.

The accompanying data archive includes the complete 9,849-paper aligned RQ1 cohort with all three input representations and frozen study outputs; the software archive provides offline reanalysis scripts. Project software is Apache-2.0; benchmark data and documentation are CC BY 4.0, with source-derived content retaining its own terms. Appendix A.2 details access and licensing; Section 3.3 describes how SciVerse exposes SPRA to researchers and agents.

## 3.3 Public Interfaces and Data Access

SciVerse exposes structured paper records under the service name Paper Schema. The /paper-schema prefix identifies this service in API requests. SPRA specifies the representation studied here, while Paper Schema names the deployed access interface.

Researchers can explore SPRA through Sci-Verse and access the same paper-level research objects programmatically through the Paper Schema API. For example, a JSON POST to /paper-schema/search returns candidate paper identifiers; /paper-schema/materials then retrieves research objects and relations for selected papers. Appendix C.2 gives the HTTPS address, token authentication, and cURL and Python calls.

The Paper Schema Agent Skill makes this access sequence available to agents through task-oriented tools for paper discovery, object retrieval, and evidence assembly. SciVerse Agent Tools provides the integration resources for using these tools in agent workflows. The SPRA construction procedure and its extraction implementation are described in Section 3.4.

The online corpus and the frozen evaluation materials serve complementary uses. Tokenauthenticated API access supports discovery and retrieval of the evolving paper collection. The accompanying data.zip and software.zip preserve the study records and offline reanalysis workflow described in Appendix D. Corpus statistics refer to the September 2026 snapshot; later service responses may reflect updated records. API access follows account permissions and quotas, while redistribution of source-derived content follows its source terms.

## 3.4 Construction Implementation

The SPRA extraction implementation starts from section-aware Markdown with paragraph markers produced by MinerU (Figure 2). Five core LLM calls establish a shared unit census, predict relations, and fill Problem, Method, and Evidence content. Two citation passes join roles and verbatim signals to bibliographic metadata by citation key; implementation and decoding settings appear in Appendix A.3.

The assembly code resolves source markers, captures tables verbatim, reconciles identifiers, and checks types, endpoints, provenance, and census coverage. Optional numerical checks flag score values absent from captured table cells and explicitly record unchecked cases. These construction controls preserve dense evidence and make numerical transcription inspectable (Appendix A.3).

## 3.5 Operations on Research Objects

Typed lookup selects methods or findings; relation traversal connects them to measures and settings; faceted filtering selects research roles or attributes. The SciVerse Paper Schema API assembles goal-specific objects, relations, and evidence for paper overviews, literature surveys, benchmark analysis, method comparison, and reproduction. Related-paper discovery extends paper or object queries into literature sets; source opening resolves recorded locators for verification.

SciVerse presents research objects through reading paths and topic graphs, with source addresses for verification; the supplement includes an offline demonstration. The shared registry (Figure 2) preserves object identity across extraction passes and later tasks.

## 4 Evaluating Use of the Representation

RQ1–RQ3 assess scientific task quality under fixed research content and a common primary Reader. RQ2 and RQ3 additionally measure input and output token use for the evaluated workflows. Sci-Verse supplies storage, retrieval, and maintenance; Appendix D.5 defines the token-accounting scope.

## 4.1 Representation Quality

Structural validity. Structural checks assess recorded source pointers and relation endpoints in the sampled object registries. They measure addressability and registry connectivity.

## 4.2 Functional Adequacy of Extracted Content

RQ1 evaluates source-conditioned scientific QA across problems and contributions, methods and components, experiments and results, and limitations and assumptions. G-U1 measures correctness, coverage, and qualification against shared Full Compact-generated references; G-U3 focuses on Problem + Contribution. Structured questions test research-unit lookup, internal-relation recovery, and qualification filtering. Full Compact supplies complete parsed article text (Appendix B.2); model roles appear below.

![](images/fe0944777ba9176e617901df56be67f4a238857fad1eae9b4cccebf677d5f8bf.jpg)  
Figure 2: Five conceptual stages of SPRA construction. A shared unit census precedes relation planning and role-specific filling; assembly joins source-linked records and stores external citations separately. Validation checks profile conformance, relation endpoints, and source-link integrity.

## 4.3 Tasks and Research Questions

RQ1: source-conditioned scientific QA. The evaluation frame contains 9,849 eligible papers across seven AI subfields and seven question families for semantic QA and structured operations.

RQ2: cross-paper reasoning. We evaluate 3,070 complete paired tasks. We score correctness, grounding, and qualification on directed-relation and Top-5 synthesis questions.

RQ3: downstream idea generation. We evaluate 1,001 task variants across seven AI domains, using shared paper selections to generate and assess evidence-grounded proposals.

## 4.4 Controlled Representation Comparison

Conditions share questions, paper identities and order, answer format, rubric, and Reader. Full Compact supplies complete parsed article text; Text-RAG, verbatim task-relevant passages; and Schema-RAG, SPRA objects and internal relations. Complete multi-paper inputs can exceed context capacity; retrieval keeps evidence compact. Task evidence is fixed before inference. Both RQ2 retrieval conditions select one primary item per paper for each of four shared facets. Text-RAG ranks passages with question-derived BM25F-like queries; Schema-RAG ranks deterministically assembled SPRA objects and typed relations.

## 4.5 Reader and Evaluation Protocol

Figure 3 summarizes model roles and criterionbased evaluation (Liu et al., 2023; Zheng et al., 2023; Kim et al., 2025). Llama 4 Scout is the primary Reader; two long-context RQ2 Full Compact tasks use a GPT fallback (Appendix B). For RQ1/RQ2, DeepSeek-v4-pro generates shared reference answers from Full Compact sources. MiniMax-M3 scores Reader outputs against these references without condition labels. For RQ3, DeepSeek-v4-pro assesses implementation, and MiniMax-M2.5, Qwen3.5-27B, and GLM-5.1 independently review proposals.

Frozen outputs and scripts support offline replay of RQ1–RQ3 statistics; Appendices A.4 and D.5 document structural checks and token-accounting scope.

<table><tr><td>Study</td><td>Condensed task</td><td>Required evidence</td></tr><tr><td>RQ1 G-U3</td><td>Identify the problem and central contribution. Problem + Contribution</td><td></td></tr><tr><td>RQ1-qual.</td><td>Bind qualifications to claims or methods.</td><td>Finding + qualification</td></tr><tr><td>RQ2</td><td>Assess directed relations; synthesize papers. Typed relations + multi-paper context</td><td></td></tr><tr><td>RQ3</td><td>Generate a grounded benchmark-transfer idea.</td><td>Selected papers + qualifications</td></tr></table>

Table 3: Representative benchmark operations. Task formulations are condensed; the reproducibility package contains the complete prompts.

![](images/e061b2ebff50553aafda4b69a007e8b312dc418ffde05989a843e2281bd043a6.jpg)  
Figure 3: Reader and evaluation protocol for the three representation conditions. Section 4.5 specifies model roles and scoring.

## 4.6 Metrics and Statistical Analysis

The primary reported outcomes compare Schema-RAG access with Text-RAG for RQ1 G-U1 and RQ2 directed correctness, and with Full Compact for RQ2 Top-5 correctness. G-U2, G-U3, grounding, qualification, and token use provide supporting analyses. RQ1 and RQ2 quality scores use a 0–5 scale. RQ2 qualification measures the conditions attached to a specified answer. RQ3 reports Overall quality and input+output tokens.

Paired bootstrap resampling preserves condition pairing and study sampling structure to estimate 95% confidence intervals (Koehn, 2004); the three RQ3 Overall contrasts receive Holm correction (Holm, 1979). Appendix B specifies resampling units, replicate counts, and tests.

<table><tr><td>Measure</td><td>Full Compact</td><td>Text- RAG</td><td>Schema- RAG</td></tr><tr><td>G-U1</td><td>2.420</td><td>1.961</td><td>2.342</td></tr><tr><td>G-U2</td><td>1.327</td><td>1.204</td><td>1.317</td></tr><tr><td>G-U3</td><td>3.201</td><td>2.167</td><td>3.216</td></tr></table>

Table 4: RQ1 mean scores on the 0–5 scale. G-U1: semantic answer quality judged against references generated from Full Compact; G-U2: structured operations; G-U3: Problem + Contribution. Columns show Full Compact, passage, and SPRA access; bold marks maxima.

## 5 Results

## 5.1 Structural and Semantic Assessment

In the sampled records, 203,628 of 214,239 core units (95.0%) in the 9,849-paper frame have a source pointer, and 106,439 of 109,591 relations (97.1%) in the complete relation records for 4,350 papers join to both endpoints. These rates measure addressability and connectivity; RQ1 evaluates content through scientific QA and structured operations. Appendix A.4 gives the structural checks.

In a model-based assessment against public NCG annotations, SPRA receives a mean contribution-level semantic-alignment score of 84.49/100 from GPT-4.1. Information-unit alignment reaches 4.436/5, supporting correspondence between the extracted research roles and annotated contributions (Appendix A.5).

## 5.2 Extracted Structure for Source-Conditioned QA

SPRA retains much of Full Compact’s sourceconditioned QA score while outperforming passage retrieval (Table 4; Figure 4a). G-U1 reaches 2.342/5 against Full Compact’s 2.420/5: the paired difference is −0.078/5 (95% CI [−0.110, −0.046]). Relative to passage retrieval, the gain is 0.381/5 (95% CI [+0.351,+0.411]). G-U1 combines correctness, coverage, and qualification against complete-article references.

<table><tr><td>Task</td><td>Metric</td><td>Full Compact</td><td>Text-RAG</td><td>Schema-RAG</td><td>Difference / reduction</td></tr><tr><td colspan="6">Answer quality (0–5)</td></tr><tr><td>Directed</td><td>Correctness</td><td>2.583</td><td>4.543</td><td>4.569</td><td>+0.026 [+0.004, +0.049]</td></tr><tr><td></td><td>Grounding</td><td>3.098</td><td>4.323</td><td>4.368</td><td>+0.045 [+0.016, +0.073]</td></tr><tr><td></td><td>Qualification</td><td>2.756</td><td>4.104</td><td>4.288</td><td>+0.184 [+0.154, +0.212]</td></tr><tr><td></td><td>Correctness</td><td>3.699</td><td>2.958</td><td>3.114</td><td>+0.156 [+0.114, +0.200]</td></tr><tr><td>Top-5</td><td>Grounding</td><td>3.439</td><td>3.110</td><td>2.966</td><td>-0.145 [−0.183, -0.105]</td></tr><tr><td></td><td>Qualification</td><td>3.725</td><td>2.858</td><td>2.533</td><td>-0.325 [−0.366, -0.284]</td></tr><tr><td colspan="6">Input+output (millions of tokens)</td></tr><tr><td>All tasks</td><td>Total tokens</td><td>203.26</td><td>13.30</td><td>9.51</td><td>-28.5% / -95.3%</td></tr></table>

Table 5: RQ2 scores (0–5) and input+output tokens across 3,070 paired tasks. Quality differences are Schema-RAG minus Text-RAG [paired 95% CI]; token reductions are relative to Text-RAG / Full Compact, respectively. Bold marks the highest quality score in each row and the lowest token total.

For structured operations, SPRA exceeds Text-RAG by 0.113/5 on G-U2. It also has the highest observed G-U3 mean for identifying the problem and contribution, 3.216/5 versus 3.201/5 for Full Compact. These results support the use of SPRA for answering source-conditioned questions and identifying a paper’s problem and contribution.

(a) RQ1: single-paper quality  
![](images/606bc7eefe0ce93dbad3bb85d7a8e1e69aaedbc29a75ed8ded02058d6317cfc0.jpg)

## 5.3 Cross-Paper Reasoning

Across 3,070 paired tasks (Appendix B.3), SPRA uses 9.51M input+output tokens: 28.5% fewer than Text-RAG (13.30M) and 95.3% fewer than Full Compact (203.26M; Table 5).

Schema-RAG improves directed judgments across all three measured criteria (Table 5). Relative to Text-RAG, qualification improves by 0.184/5 (95% CI [+0.154,+0.212]), correctness by 0.026/5, and grounding by 0.045/5. The largest gain concerns qualification: retaining support, uncertainty, and applicable conditions.

Top-5 synthesis shows a different result pattern. SPRA improves correctness over Text-RAG by 0.156/5, while grounding and qualification decrease by 0.145/5 and 0.325/5; Full Compact has the highest means on all three criteria. These synthesis results motivate further evaluation of input selection that retains claim-specific qualifications alongside core findings.

(b) RQ2: cross-paper token use  
![](images/a3bbaa50998708d238ce980301158712733bc80cbf4302851ba4220dc4ed9c58.jpg)  
Figure 4: Quality and token use: (a) RQ1 G-U1 semantic QA against Full Compact-generated references and G-U2 structured operations (0–5); (b) input+output tokens across all paired RQ2 tasks, including Top-5. Gray and blue annotations show SPRA’s reductions relative to Full Compact and Text-RAG.

<table><tr><td>Metric</td><td>SPRA seeds</td><td>SPRA + citations</td></tr><tr><td>Cross-paper</td><td>1.779</td><td>2.032</td></tr><tr><td>integration Novelty</td><td>2.100</td><td>2.172</td></tr><tr><td>Evidence grounding</td><td>1.848</td><td>1.931</td></tr><tr><td>Qualification</td><td>2.750</td><td>2.744</td></tr><tr><td>handling Independent Overall</td><td>2.302</td><td>2.379</td></tr></table>

Table 6: Scientific quality with SPRA seeds and citation expansion: 1,001 paired questions; quality scores use 0–5. Means average two reviewers; Overall is assessed independently.

## 5.4 Idea Generation and Citation Expansion

The main RQ3 comparison evaluates evidencegrounded proposals across 1,001 variants with shared paper selections. The family-balanced SPRA–Full Compact Overall contrast is −0.067 points (95% CI [−0.102, −0.034]); the SPRA– Text-RAG contrast is +0.011 points (95% CI [−0.029, +0.058]). SPRA uses 93.7% fewer input+output tokens than Full Compact and 17.7% fewer than Text-RAG. Appendix B reports all three paired contrasts and token totals.

The separate citation-expansion study tests access to additional research through represented dependencies. Using Qwen3.5-397B, both conditions start from the same seed objects under a shared 10,000-token ceiling. Adding up to two resolved citation neighbors improves cross-paper integration by +0.253/5 (95% CI [+0.195, +0.308]). Table 6 reports dimension-specific results. Appendix B.6 specifies neighbor selection, added evidence, and paired analysis.

## 6 Discussion

SPRA makes the research argument an explicit, reusable object of scientific work. Findings remain connected to contributions, evaluations, and conditions, while selected citation dependencies connect the focal paper to prior work. These connections support queries and relation traversal. Citation expansion demonstrates their use in assembling additional evidence under a shared context ceiling.

The evaluations connect representation quality to scientific use across QA, cross-paper reasoning, and idea generation. Token savings quantify workflow efficiency alongside the scientific utility of explicit research structure.

## 7 Conclusion

SPRA addresses the title’s question by exposing the research structure already present in scientific papers as persistent, queryable objects and relations. The empirical-AI profile connects problems, contributions, methods, evaluations, and findings, with source bindings for inspection and selected citation dependencies for cross-paper access. SciVerse makes this representation accessible at million-paper scale through a website, the Paper Schema API, and an Agent Skill; the extraction implementation supports construction from parsed articles.

The evaluations show that SPRA supports source-conditioned QA and directed cross-paper reasoning with substantially lower token use in the evaluated retrieval workflows. RQ1 gains 0.381/5 over passage retrieval while scoring 0.078/5 below Full Compact; RQ2 uses 28.5% fewer input+output tokens than passage retrieval across all paired tasks. Mixed Top-5 synthesis results and the lower RQ3 Overall score relative to Full Compact show that compact structure must be assessed against each workflow’s information needs. SPRA provides a concrete foundation for research systems that query, compare, and reuse paper-level scientific content through a shared representation.

## Limitations

Generalization and successive workflows. The evaluations cover the empirical-AI profile and the specified task workflows. Transfer to other domains, successive-task reuse, and stability under meaning-preserving rewrites remain future evaluation directions. The comparisons assess the instantiated representation and access workflows; they do not isolate the contribution of individual role or predicate labels.

Services and cost coverage. Updates to hosted services can change behavior in new runs; the results characterize the recorded service invocations. Token savings cover model inputs and outputs for downstream tasks; storage, retrieval, and maintenance are provided by SciVerse.

## Ethical Considerations

SPRA preserves attribution and inspectable source links. Automated extraction may omit or misstate qualifications, which can change how a research claim is understood. Corpus coverage is shaped by the included venues, years, and source availability, so missing or underrepresented work should not be taken as evidence that a topic or finding is absent from the literature. Users may also overrely on concise structured records and lose context available in the full article. For important research or other consequential uses, users should check the relevant claims and qualifications against the source article. Access and redistribution follow source licenses; researchers retain responsibility for interpretation and downstream validation.

## Acknowledgements

OpenAI Codex assisted with editorial revision and routine monitoring of experimental progress. The authors reviewed the assisted work and take responsibility for the research decisions, analyses, citations, interpretation, and presentation.

## References

Akari Asai, Jacqueline He, Rulin Shao, Weijia Shi, Amanpreet Singh, Joseph Chee Chang, Kyle Lo, Luca Soldaini, Sergey Feldman, Mike D’Arcy, David Wadden, Matt Latzke, Jenna Sparks, Jena D. Hwang, Varsha Kishore, Minyang Tian, Pan Ji, Shengyan Liu, Hao Tong, and 9 others. 2026. Synthesizing scientific literature with retrieval-augmented language models. Nature, 650(8103):857–863.

Jinheon Baek, Sujay Kumar Jauhar, Silviu Cucerzan, and Sung Ju Hwang. 2025. ResearchAgent: Iterative research idea generation over scientific literature with large language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6709–6738, Albuquerque, New Mexico. Association for Computational Linguistics.

Sebastián Brzovic, Cristóbal Rojas, and Andrés Abeliuk. 2025. Networks multiscale entropy analysis. Preprint, arXiv:2510.11524v1.

Lingjiao Chen, Matei Zaharia, and James Zou. 2023. FrugalGPT: How to use large language models while reducing cost and improving performance. Preprint, arXiv:2305.05176.

James Cheney, Paolo Missier, and Luc Moreau. 2013. Constraints of the PROV data model. W3C recommendation, World Wide Web Consortium.

Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, and Matt Gardner. 2021. A dataset of information-seeking questions and answers anchored in research papers. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4599–4610, Online. Association for Computational Linguistics.

Bofu Dong, Pritesh Shah, Sumedh Sonawane, Tiyasha Banerjee, Erin Brady, Xinya Du, and Ming Jiang. 2025. SciEvent: Benchmarking multi-domain scientific event extraction. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 17225–17255, Suzhou, China. Association for Computational Linguistics.

Jennifer D’Souza, Sören Auer, and Ted Pedersen. 2021. SemEval-2021 task 11: NLPContributionGraph - structuring scholarly NLP contributions for a research knowledge graph. In Proceedings of the 15th International Workshop on Semantic Evaluation (SemEval-2021), pages 364–376, Online. Association for Computational Linguistics.

Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. 2024. From local to global: A graph rag approach to query-focused summarization. Preprint, arXiv:2404.16130. Version 2, revised February 2025.

Yichen He, Guanhua Huang, Peiyuan Feng, Yuan Lin, Yuchen Zhang, Hang Li, and Weinan E. 2025. PaSa: An LLM agent for comprehensive academic paper search. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 11663–11679, Vienna, Austria. Association for Computational Linguistics.

Sture Holm. 1979. A simple sequentially rejective multiple test procedure. Scandinavian Journal ofStatistics, 6(2):65–70.

Cheng-Yu Hsieh, Yung-Sung Chuang, Chun-Liang Li, Zifeng Wang, Long Le, Abhishek Kumar, James Glass, Alexander Ratner, Chen-Yu Lee, Ranjay Krishna, and Tomas Pfister. 2024. Found in the middle: Calibrating positional attention bias improves long context utilization. In Findings of the Association for Computational Linguistics: ACL 2024, pages 14982– 14995. Association for Computational Linguistics.

ICLR. 2026. ICLR 2026 fact sheet. International Conference on Learning Representations. Accessed 2026-10-13.

Sarthak Jain, Madeleine van Zuylen, Hannaneh Hajishirzi, and Iz Beltagy. 2020. SciREX: A challenge dataset for document-level information extraction. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 7506–7516, Online. Association for Computational Linguistics.

Mohamad Yaser Jaradeh, Allard Oelen, Kheir Eddine Farfar, Manuel Prinz, Jennifer D’Souza, Gábor Kismihók, Markus Stocker, and Sören Auer. 2019. Open Research Knowledge Graph: Next generation infrastructure for semantic scholarly knowledge. In Proceedings ofthe 10th International Conference on Knowledge Capture, pages 243–246. ACM. Openaccess manuscript.

Seungone Kim, Juyoung Suk, Ji Yong Cho, Shayne Longpre, Chaeeun Kim, Dongkeun Yoon, Guijin Son, Yejin Cho, Sheikh Shafayat, Jinheon Baek, Sue Hyun Park, Hyeonbin Hwang, Jinkyung Jo, Hyowon Cho, Haebin Shin, Seongyun Lee, Hanseok Oh, Noah Lee, Namgyu Ho, and 13 others. 2025. The BiGGen bench: A principled benchmark for fine-grained evaluation of language models with language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5877– 5919, Albuquerque, New Mexico. Association for Computational Linguistics.

Philipp Koehn. 2004. Statistical significance tests for machine translation evaluation. In Proceedings ofthe 2004 Conference on Empirical Methods in Natural Language Processing, pages 388–395, Barcelona, Spain. Association for Computational Linguistics.

Timothy Lebo, Satya Sahoo, and Deborah McGuinness. 2013. PROV-O: The PROV ontology. W3C recommendation, World Wide Web Consortium.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive NLP tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459– 9474. Curran Associates, Inc.

Maria Liakata, Shyamasree Saha, Simon Dobnik, Colin Batchelor, and Dietrich Rebholz-Schuhmann. 2012. Automatic recognition of conceptualization zones in scientific articles and two life science applications. Bioinformatics, 28(7):991–1000.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language

models use long contexts. Transactions ofthe Associationfor Computational Linguistics, 12:157–173.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-eval: NLG evaluation using gpt-4 with better human alignment. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2511–2522, Singapore. Association for Computational Linguistics.

Kyle Lo, Lucy Lu Wang, Mark Neumann, Rodney Kinney, and Daniel Weld. 2020. S2ORC: The semantic scholar open research corpus. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 4969–4983, Online. Association for Computational Linguistics.

Patrice Lopez. 2009. GROBID: Combining automatic bibliographic data recognition and term extraction for scholarship publications. In Research and Advanced Technologyfor Digital Libraries, pages 473– 474. Springer.

Yi Luan, Luheng He, Mari Ostendorf, and Hannaneh Hajishirzi. 2018. Multi-task identification of entities, relations, and coreference for scientific knowledge graph construction. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, pages 3219–3232, Brussels, Belgium. Association for Computational Linguistics.

Jakub Lála, Odhran O’Donoghue, Aleksandar Shtedritski, Sam Cox, Samuel G. Rodriques, and Andrew D. White. 2023. PaperQA: Retrieval-augmented generative agent for scientific research. Preprint, arXiv:2312.07559.

Meta. 2025. Llama 4 Scout model card.

National Information Standards Organization. 2024. ANSI/NISO Z39.96-2024: Journal Article Tag Suite (JATS), Version 1.4.

Allard Oelen, Mohamad Yaser Jaradeh, and Sören Auer. 2024. ORKG ASK: a neuro-symbolic scholarly search and exploration system. Preprint, arXiv:2412.04977.

Open Research Knowledge Graph. n.d. Creating templates in the ORKG. Accessed 2026-10-11.

Zhao Ren, Rathi Adarshi Rammohan, Kevin Scheck, Thanh Tam Nguyen, Sheng Li, and Tanja Schultz. 2025. Retain-free machine unlearning for speech emotion recognition. Preprint, arXiv:2510.04251.

Research Object Community. 2026. RO-Crate metadata specification 1.3. Recommendation, 22 June 2026.

Michael D. Skarlinski, Sam Cox, Jon M. Laurent, James D. Braza, Michaela Hinks, Michael J. Hammerling, Manvitha Ponnapati, Samuel G. Rodriques, and Andrew D. White. 2024. Language agents achieve superhuman synthesis of scientific knowledge. Preprint, arXiv:2409.13740.

Stian Soiland-Reyes, Peter Sefton, Mercè Crosas, Leyla Jael Castro, Frederik Coppens, José M. Fernández, Daniel Garijo, Björn Grüning, Marco La Rosa, Simone Leo, Eoghan Ó Carragáin, Marc Portier, Ana Trisovic, RO-Crate Community, Paul Groth, and Carole Goble. 2022. Packaging research artefacts with RO-Crate. Data Science, 5(2):97–138.

Jiarui Song, Yunheng Shen, Chengbin Hou, Pengyu Wang, Jinbao Wang, Ke Tang, and Hairong Lv. 2025. Fedaghn: Personalized federated learning with attentive graph hypernetworks. Preprint, arXiv:2501.16379v2.

Simone Teufel, Advaith Siddharthan, and Colin Batchelor. 2009. Towards discipline-independent argumentative zoning: Evidence from chemistry and computational linguistics. In Proceedings ofthe 2009 Conference on Empirical Methods in Natural Language Processing, pages 1493–1502, Singapore. Association for Computational Linguistics.

David Wadden, Ulme Wennberg, Yi Luan, and Hannaneh Hajishirzi. 2019. Entity, relation, and event extraction with contextualized span representations. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5784– 5789, Hong Kong, China. Association for Computational Linguistics.

Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, Bo Zhang, Liqun Wei, Zhihao Sui, Wei Li, Botian Shi, Yu Qiao, Dahua Lin, and Conghui He. 2024. MinerU: An open-source solution for precise document content extraction. Preprint, arXiv:2409.18839.

Chenguang Wang, Ming Li, Chengrui Fan, Jianpeng Chen, Han Chen, Tianyi Zhou, and Dawei Zhou. 2026. A missing piece for trustworthy AI reviewers: From benchmarking rhetorical robustness to SciCore review. Preprint, arXiv:2609.39027.

Yujun Wu, Dongxu Zhang, Xinchen Li, Jinhang Xu, Yiling Duan, Yumou Liu, Jiabao Pan, Qiyuan Zhu, Xuanhe Zhou, Jingxuan Wei, Siyuan Li, Jintao Chen, Conghui He, and Cheng Tan. 2026. Intern-Atlas: A methodological evolution graph as research infrastructure for ai scientists. Preprint, arXiv:2604.28158v2.

Qi Zhang, Zhijia Chen, Huitong Pan, Cornelia Caragea, Longin Jan Latecki, and Eduard Dragut. 2024. SciER: An entity and relation extraction dataset for datasets, methods, and tasks in scientific documents. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 13083–13100, Miami, Florida, USA. Association for Computational Linguistics.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin,

Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with mt-bench and chatbot arena. In Advances in Neural Information Processing Systems, volume 36, pages 46595–46623. Curran Associates, Inc.

Xiangrong Zhu, Yuexiang Xie, Yi Liu, Yaliang Li, and Wei Hu. 2025. Knowledge graph-guided retrieval augmented generation. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8912–8924, Albuquerque, New Mexico. Association for Computational Linguistics.

## A Resource and Construction

## A.1 Resource Scope and Sampling

The corpus contains 1,059,112 SPRA paper representations spanning 1965–2026, produced through the construction procedure in Appendix A.3 and exposed through SciVerse. The source-aligned evaluation sample contains 9,849 eligible papers published in 2025, with 1,407 per field across seven AI subfields (Table 7). The cohort follows the primary Reader’s documented August 2024 knowledge cutoff (Meta, 2025). The sampled SPRA instances retain their corpus research objects, relations, and source bindings, linking evaluation to the resource’s production workflow. The resulting assessments characterize the eligible 2025 cohort across these seven fields. RQ3 evaluates 1,001 task variants across seven AI domains, as detailed in Appendix B.

Corpus statistics describe the collection used in this study, measured in September 2026. The supplementary records specify the counting populations and preserve the aggregate results and evaluation data. The SciVerse Paper Schema API supports continued paper- and object-level access; Appendix D describes the accompanying evaluation materials for offline inspection and numerical reanalysis.

The corpus contains 19,949,289 core units (18.84/paper) and 24,073,381 internal relations (22.73/paper). Core units comprise Problems, Contributions, Components, ExperimentSetups, Measures, and Findings; relation counts cover the eight profile predicates and exclude citations. Machinereadable statistics report per-type counts and the year distribution.

The sample contains 214,239 core units: 9,853 Problems, 11,395 Contributions, 36,522 Components, 37,981 ExperimentSetups, 44,355 Measures, and 74,133 Findings (mean 21.75/paper; median 21; range 4–213). Separate counts are 9,849 Documents, 177,003 References, and 8,646 Resources. Its 409,082 locator references count repeated addresses separately.

Complete relation retrieval covers 4,350 papers and 109,591 relations (mean 25.19/paper; median 24; range 4–295). These complete groups define the structural-check population, rather than a second balanced sample.

## A.2 Sources, Acquisition, and Resource Access

Sources and acquisition. The evaluation roster includes AAAI, ACL, EMNLP, ICLR, ICML, NeurIPS, CVPR, ICCV, robotics proceedings, and arXiv. Venue labels identify publication outlets; recorded arXiv, ACL Anthology, AAAI, and DOI-linked URLs identify accessed versions. SciVerse supplies metadata, parsed sources, and SPRA instances; MinerU produces section-aware Markdown. The packaged statistics and source-linked examples are grouped in data/data/supplementary/paper\_results. json and data/data/supplementary/frozen\_ subset.json; the SciVerse Paper Schema API provides paper- and object-level corpus access.

Service and data access. The accompanying data archive contains the complete 9,849-paper aligned RQ1 cohort with all three input representations: complete text, retrieved passages, and SPRA. Together with the profile, attributed examples, corpus statistics, frozen study outputs, and analysis scripts, this supports offline inspection and numerical replay without service credentials (Appendix D). SciVerse exposes the million-record corpus through the token-authenticated SciVerse Paper Schema API for paper discovery and access to typed units, relations, citations, and source context. The API supports discovery by paper metadata or research objects, followed by goal-specific material selection for overview, survey, benchmark, method, and reproduction tasks. Agents can retrieve typed records, follow relations and resolved citation links, and reopen source context. Material responses report available and returned counts, including truncation, so callers can request additional records when needed. The Paper Schema API documentation specifies request and response formats; API Tokens are created through the Sci-Verse account console. The Paper Schema Agent Skill documentation describes the corresponding tools and installation. Access is subject to account permissions and quotas.

<table><tr><td>AI subfield</td><td>Eligible</td></tr><tr><td>Language, speech, and information access</td><td>1,407</td></tr><tr><td>Vision and multimodal perception</td><td>1,407</td></tr><tr><td>Learning, optimization, and general ML</td><td>1,407</td></tr><tr><td>Data, knowledge, and computational systems</td><td>1,407</td></tr><tr><td>Reasoning, agents, RL, and embodied AI</td><td>1,407</td></tr><tr><td>Trustworthy and human-centered AI</td><td>1,407</td></tr><tr><td>AI for science and domain applications</td><td>1,407</td></tr><tr><td>Total</td><td>9,849</td></tr></table>

Table 7: Eligible evaluation papers across seven AI sampling strata.

Licenses. Project software uses Apache-2.0; project-authored benchmark records, labels, prompts, and documentation use CC BY 4.0. Third-party articles and service payloads retain their source terms; redistribution follows the terms of each source. The seven inspection instances retain CC-BY attribution. Evaluation metadata records 8,340 positive, 569 negative, and 940 unspecified open-access flags, separately from redistribution permissions in the source-version access records.

## A.3 Construction Procedure

MinerU supplies Markdown with paragraph markers and verbatim tables. Five core calls share a paper-owned registry: an initial extraction stage assigns unit identifiers and source markers; a structural pass predicts part\_of, co\_contribution, compares\_to, and evaluates; three role-specific passes enrich Problem, Method, and Evidence units and predict relations such as motivates, about, and supports. The Evidence pass creates Findings and binds their measures and conditions. A supports edge connects a Finding to associated research objects, giving agents direct access to evidence-bearing content and, where recorded, its source locations.

Separate metadata, citation-role, and referencemetadata calls form the bibliographic and externaldependency layers. Deterministic assembly reconciles identifiers, merges records, derives resolves, and validates fields, endpoints, ownership, and unit coverage. Prompts constrain admissible types, identifier reuse, and evidence; citation keys join external records without merging them into internal claims.

The SPRA extraction implementation uses a shared paper-level registry and role-specific extraction stages. Its model configuration and request settings are summarized in Appendix D.3 and the supplementary registry.

Validation. Required extraction stages must produce schema-valid records; unresolved validation issues exclude a paper from task use. Assembly normalizes identifiers and source markers and removes duplicate, dangling, self-loop, unknownpredicate, and invalid-signature edges. Metadata or citation-extraction errors are retained with warnings when core extraction remains usable.

Endpoint contract. For compares\_to, both source and target types belong to {Contribution, Component, ExperimentSetup, Measure}. For motivates, the source is a Problem and the target is a Contribution, Component, or ExperimentSetup. Every internal endpoint must belong to the same paper’s unit registry. Table 1 lists the eight internal relation types and their admissible endpoint signatures.

Source markers are normalized against parsed blocks and headings. Tables are sliced verbatim. An optional numeric-presence checker flags Measure values absent from captured table cells, records unchecked cases, and leaves units and scores unchanged. The construction rules require preservation of values, units, comparison bases, and conditions; numeric presence alone does not verify these field associations.

## A.4 Structural Validation

These checks quantify two prerequisites for inspecting represented claims: source traceability and relation endpoint connectivity. Among 214,239 sampled research units, 203,628 (95.0%) have a nonempty provenance field. In the complete relation records for 4,350 papers, 106,439 of

109,591 relations (97.1%) join to both endpoints; 102,024 of these joined relations have provenance. These measures characterize the availability of source pointers and connected relation endpoints for agent access. The supports relation provides evidence links between research objects and Findings; agents can follow these links to retrieve associated content and inspect the source evidence when available. RQ1 complements these checks by evaluating the scientific use of extracted content in source-conditioned answers.

A sampled object assessment provides complementary record-level evidence; the supplementary materials document its protocol and records.

## A.5 Contribution-Level Semantic Alignment

A paper representation should preserve the scientific contributions expressed in the source article. We assess this property against the public annotations of NCG/SemEval 2021 Task 11 (D’Souza et al., 2021), comparing SPRA objects with annotated contribution sentences, concepts, information-unit families, and contribution facts.

The projection implementation reads precomputed extraction records and maps their types and content to NCG information-unit families: Problem objects supply research problems, ExperimentSetup objects supply experimental settings, and Finding and Measure objects supply results. For each gold contribution sentence, it selects the extracted string with the highest token F1 as a candidate alignment. GPT-4.1 receives the gold annotations and projected content, then judges semantic correspondence rather than lexical overlap. Thus, candidate matching organizes the evidence for a gold-conditioned coverage assessment. The rubric accepts paraphrase and canonicalization, and penalizes unsupported additions, incorrect attribution, and omitted contribution facts. Five dimensions receive scores from 0 to 5, alongside a separately judged Overall score from 0 to 100. These paperlevel judgments measure the preservation of annotated contribution structure.

SPRA achieves a mean Overall score of 84.49 and a median of 85. Information-unit alignment scores higher (4.436/5) than contribution-fact coverage (3.890/5; Table 8). This pattern suggests that SPRA captures the organization of scientific contributions more consistently than their fine-grained factual detail. The assessment complements RQ1 by examining correspondence with annotated scientific content before its use in downstream answers.

<table><tr><td>Dimension</td><td>Mean</td></tr><tr><td>Contribution-sentence coverage Key-concept coverage</td><td>3.967</td></tr><tr><td>Information-unit alignment</td><td>4.031</td></tr><tr><td>Contribution-fact coverage</td><td>4.436 3.890</td></tr><tr><td>Evidence traceability</td><td>4.998</td></tr><tr><td>Overall (0–100)</td><td>84.49</td></tr></table>

Table 8: NCG contribution-level semantic assessment. Dimension scores use 0–5; Overall is a separately returned model score on 0–100.

The supplement provides the projection and judge implementations, rubric, aggregate scores, domain summaries, and paper-level call metadata. Complete run-specific judge inputs and individual judgments are unavailable; the retained code documents the procedure, while the accompanying NCG documentation identifies the additional records needed to reproduce the reported aggregate scores.

## A.6 Structural Comparison of Scholarly Representations

Table 9 compares documented capabilities of scholarly representations and infrastructure. JATS addresses article markup (National Information Standards Organization, 2024); PROV-O defines the provenance ontology (Lebo et al., 2013), while PROV-CONSTRAINTS specifies validity constraints (Cheney et al., 2013); RO-Crate addresses research-object packaging (Research Object Community, 2026). SciREX and NLPContribution-Graph (NCG) address scientific extraction, while ORKG provides a maintained scholarly graph (Jain et al., 2020; D’Souza et al., 2021; Jaradeh et al., 2019; Open Research Knowledge Graph, n.d.; Oelen et al., 2024). The comparison concerns their published scope, rather than measured extraction accuracy.

Schema coverage and granularity. SciREX represents four entity types—Method, Task, Dataset, and Metric—with mentions, coreference, salience, and document-level n-ary result relations. NCG organizes contribution sentences, phrases, and triples under 12 information units and records a from sentence association. Both therefore already associate extracted content with a paper and its text. SPRA’s six-role empirical-AI profile instead makes the research argument explicit: Problems motivate Contributions, Components and ExperimentSetups describe implementation and evaluation context, and Measures connect to Findings with conditions. Its eight internal predicates have admissible endpoint signatures (Table 1). Type counts describe granularity, not a coverage or quality ranking.

<table><tr><td></td><td>JATS</td><td colspan="5"></td><td></td></tr><tr><td>Capability</td><td>1.4</td><td>ORKG</td><td>PROV-O</td><td>v1.3</td><td>SciREX</td><td>NCG</td><td>SPRA</td></tr><tr><td>Paper-associated research units</td><td>△</td><td>√</td><td>△</td><td>△</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Typed scientific relations</td><td> $\triangle$ </td><td>√</td><td> $\bigtriangleup$ </td><td> $\bigtriangleup$ </td><td>√</td><td>√</td><td>√</td></tr><tr><td>Units linked to source content</td><td> $\triangle$ </td><td>△</td><td> $\bigtriangleup$ </td><td> $\bigtriangleup$ </td><td>√</td><td>√</td><td>√</td></tr><tr><td>Persistent resource identifiers</td><td>△</td><td>√</td><td>√</td><td>√</td><td>△</td><td>△</td><td>√</td></tr><tr><td>Explicit schema/profile validation</td><td>√</td><td>√</td><td>△</td><td>√</td><td> $\triangle$ </td><td>△</td><td>√</td></tr><tr><td>Scientific lookup/traversal API</td><td>×</td><td>√</td><td>X</td><td>X</td><td>X</td><td>△</td><td>√</td></tr><tr><td>Matched downstream task evaluation</td><td>×</td><td>△</td><td>×</td><td>X</td><td>×</td><td>×</td><td>√</td></tr><tr><td>Condition/qualification access</td><td>△</td><td>△</td><td>△</td><td>△</td><td>△</td><td>△</td><td>√</td></tr><tr><td>Version/revision metadata</td><td> $\checkmark$ </td><td> $\checkmark$ </td><td> $\checkmark$ </td><td> $\checkmark$ </td><td> $\bigtriangleup$ </td><td>△</td><td>√</td></tr></table>

Table 9: Feature coverage of adjacent scholarly representations. $\checkmark$ denotes documented core support, $\triangle$ partial support or support requiring a profile, integration, or extension, and × no documented core support for the stated capability. NCG = NLPContributionGraph.

From scientific tasks to the profile. Taskderived roles express a design mapping from scientific operations to representation requirements. Paper-understanding questions call for Problem and Contribution objects and their motivates/resolves links. Method comparison calls for Components and ExperimentSetups, membership, and comparison edges. Interpreting reported evidence calls for Measures and Findings, evaluates/about/supports links, and qualification attributes. Here, supports is an extracted candidate evidence association for navigation; semantic support is assessed by comparing a recorded source locator, when present, with the Finding. Verification requires source locators; reuse requires stable paper-scoped identifiers. This is an explicit task-to-profile design rationale, rather than an automated schema-induction procedure.

Persistence, source binding, and operations. ORKG already provides persistent, paper-linked contributions, templates with ranges and cardinalities, comparisons, and APIs. Paper ownership is a shared design property of ORKG and SPRA. Source binding also differs in granularity: SciREX retains mention spans, NCG links triples to contribution sentences, and SPRA records addresses for passages, tables, and figures. ORKG can represent additional evidence and qualifiers through properties and templates. SPRA’s contribution is the integration of a paper-level research-argument profile, constrained internal relations, source-opening operations, and a common object registry into a corpusscale extraction and access workflow. Typed lookup, traversal, and filtering operate on these same objects; RQ1–RQ3 assess their scientific use under matched tasks. Matched downstream evaluation in the table refers to representation-access studies, distinct from extraction benchmarks. Property alignment connects this task-oriented representation to ORKG’s broader aggregation infrastructure.

## A.7 ORKG Graph Sample and Interoperability Scope

We examine interoperability using a sample of 200 papers from ORKG’s Artificial Intelligence field and subfields. We select 200 papers from a 2,918- paper frame by deterministic ordering before inspecting graph properties. The sample includes 313 contribution roots and 6,578 statement occurrences, with contribution subgraphs traversed to three hops.

The mapping analysis examines how ORKG research-role classes and predicate labels correspond to the SPRA profile while preserving scientific identities, reference-paper links, and relation directions. The graph sample, mapping rules, and protocol are grouped in the ORKG section of data/data/supplementary/paper\_results. json. These materials characterize the interoperability interface between the two representations.

<table><tr><td>Component</td><td>Description</td></tr><tr><td>Sample frame</td><td>2,918 listed papers; 200 selected deterministically</td></tr><tr><td>Graph contexts</td><td>313 contribution roots; three-hop traversal</td></tr><tr><td>Scientific</td><td>6,578 statement occurrences</td></tr><tr><td>statements</td><td></td></tr><tr><td>Mapping rules</td><td>Explicit research-role classes and predicate labels</td></tr><tr><td>Source identity</td><td>Scientific resource identifiers and paper-reference links</td></tr></table>

Table 10: ORKG graph sample and mapping specification used to examine interoperability.

## B Evaluation Details

## B.1 Protocol and Analysis Population

Complete paired analysis sets. Table 11 gives the denominators behind the reported comparisons. RQ1 excludes 411 items without complete threecondition results: 383 stop at Reader generation, 21 at reference generation, and seven at primary judging. G-U3 is the Problem + Contribution subset of G-U1; it is not an additional task population.

We compare representations using paired tasks under a common primary Reader and evaluation protocol. Evidence inputs are fixed before inference, and the model configurations and study records are provided in Appendix D.3.

All RQ1–RQ3 and citation-expansion quality scores use the 0–5 scale. Judge dimensions recorded on the 0–4 rubric are reported as s = 5r/4. Paired differences and their intervals are expressed in score points. Rates and input+output tokens use their stated units.

Llama 4 Scout is the primary Reader for RQ1– RQ3; two long-context RQ2 Full Compact tasks use the documented GPT fallback. DeepSeek-v4- pro generates shared RQ1/RQ2 answer references from Full Compact; MiniMax-M3, blind to condition, scores Reader responses against them. The

RQ1 evaluation frame contains 9,849 items across seven AI subfields. RQ2 evaluates 3,070 complete paired tasks; RQ3 evaluates 1,001 variants. Table 12 lists the primary comparisons.

SPRA names the representation; Schema-RAG names its access condition in this comparison. Task evidence is fixed before Reader inference. For RQ2, paper groups, source revisions, questions, and the per-paper, per-facet selection slots are shared across conditions; representation-specific input packs are frozen before inference, and retrieval is independent of generated answers. Both rankers use question-derived queries and BM25Flike field weighting to select one primary evidence item per paper for each of four scientific facets. Text-RAG ranks source passages; Schema-RAG ranks SPRA objects produced through schemaconstrained extraction and deterministic assembly, together with typed internal relations. The shared slots match selection opportunities while preserving each representation’s distinct evidence form and resulting input length.

## B.2 Controlled Representation Comparison

Full Compact input boundary. Full Compact exposes the complete available parsed article text in source order. This includes table contents, formulas, and figure captions where retained in the textual extraction. Figure pixels and page images are not supplied as visual inputs; image links in the text are not dereferenced. Thus, completeness refers to the parsed textual representation, not to every visual element of the original PDF. For RQ1, this text supplies the source for shared reference answers. Text-RAG exposes selected passages; Schema-RAG exposes SPRA objects and typed internal relations. Llama 4 Scout answers from each assigned condition, and MiniMax-M3 scores those responses against the Full Compact-generated references. Questions, paper identities, response contracts, and scoring are fixed.

Full-text context handling. RQ2 Full Compact concatenates complete section-aware article views in shared paper order. Groups that fail the input gate are excluded jointly rather than truncating the full-text condition into eligibility. The 32 input-gated groups are outside the 3,422-task inputeligible population. RQ3 evidence digestion processes at most one Full Compact paper per call and at most two papers per RAG call, with a 65,000- token batch gate. These limits govern admissible inputs; token accounting includes the complete requests.

<table><tr><td>Study or outcome</td><td>Analysis frame</td><td>Complete pairs Incomplete</td><td></td></tr><tr><td>RQ1, all question families</td><td>9,849</td><td>9,438</td><td>411</td></tr><tr><td>RQ1 G-U1, semantic QA</td><td>5,628</td><td>5,510</td><td>118</td></tr><tr><td>RQ1 G-U2, structured operations</td><td>4,221</td><td>3,928</td><td>293</td></tr><tr><td>RQ1 G-U3, Problem + Contribution</td><td>1,407</td><td>1,378</td><td>29</td></tr><tr><td>RQ2, directed</td><td>2,461</td><td>2,431</td><td>30</td></tr><tr><td>RQ2, Top-5</td><td>961</td><td>639</td><td>322</td></tr><tr><td>RQ3, proposal generation</td><td>1,001</td><td>1,001</td><td>0</td></tr></table>

Table 11: Analysis populations and complete paired observations. The frame is source-aligned for RQ1 and input-eligible for RQ2; RQ3 includes all planned variants. G-U3 is included within G-U1.

<table><tr><td>Study</td><td>Evaluation scope</td><td>Primary comparison</td></tr><tr><td>RQ1</td><td>9,849</td><td>Schema-RAG vs. Text-RAG (answer quality)</td></tr><tr><td rowspan="2">RQ2</td><td>3,070</td><td>Correctness: Schema-RAG</td></tr><tr><td>complete pairs</td><td>vs. Text-RAG (directed); Schema-RAG vs. Full</td></tr><tr><td>RQ3</td><td>1,001</td><td>Compact (Top-5) Overall idea quality, three conditions</td></tr></table>

Table 12: Evaluation scope and primary comparisons.

RQ1 metric construction. Native judge dimensions are scored from 0 to 4 and displayed as $s = 5 r / 4 .$ . G-U1 averages correctness, answer coverage, and qualification over semantic questions; G-U2 uses structured correctness on the three operation families; G-U3 applies the G-U1 scoring rule to Problem + Contribution questions. These are rubric-based model judgments against shared references. They measure task performance, rather than probabilities that individual extracted records are correct. On the 0–5 reporting scale, the retained G-U1 SPRA–Full Compact contrast is −0.078 (95% CI [−0.110, −0.046]). The references can contain source details absent from either selected representation, so the comparison assesses retained information together with Reader performance. Token totals are reported within their respective studies.

## B.3 RQ2 Token Accounting

Token accounting covers the same 3,070 scored tasks in all three conditions: 2,431 directed twopaper tasks and 639 Top-5 tasks. For each task and condition, we sum one retained Reader input and its corresponding output. Inputs include evidence, instructions, task metadata, output-format instructions, and chat formatting. Reference generation, judging, failed calls, and retries are outside this retained-call measure.

The shared offline BERT WordPiece tokenizer (bert-base-uncased; frozen accounting identifier offline\_bert\_wordpiece\_ conservative\_gate\_v1) measures both input and output length. These standardized counts describe model-visible text volume; provider billing uses its own tokenization and input/output rates. In Full Compact / Text-RAG / Schema-RAG order, input totals are 199,460,508 / 10,678,861 / 6,626,310 and output totals are 3,801,534 / 2,617,712 / 2,885,511. Their sums are 203,262,042 / 13,296,573 / 9,511,821 tokens, reported as 203.26 / 13.30 / 9.51 million.

Reductions use the unrounded aggregate totals, $1 0 0 ( 1 - T _ { \mathrm { S P R A } } / T _ { \mathrm { b a s e l i n e } } )$ : 28.5% relative to Text-RAG and 95.3% relative to Full Compact. Table 13 separates the two task types and reports per-task input+output distributions. Means divide each group’s total by its task count; pooled means weight tasks equally. The supplement provides the individual counts and unrounded summaries.

## B.4 RQ3 Idea-Generation Quality and Token Use

RQ3 evaluates 1,001 idea-generation variants across seven AI domains.

<table><tr><td>Tasks (n)</td><td>Condition</td><td>Mean</td><td>Median</td><td>Q1-Q3</td><td>P90</td></tr><tr><td>All (3,070)</td><td>Full Compact</td><td>66,209</td><td>56,093</td><td>39,863–88,487</td><td>120,199</td></tr><tr><td>All (3,070)</td><td>Text-RAG</td><td>4,331</td><td>3,550</td><td>3,204–4,157</td><td>7,591</td></tr><tr><td>All (3,070)</td><td>Schema-RAG</td><td>3,098</td><td>2,557</td><td>2,438–2,773</td><td>5,335</td></tr><tr><td>Directed (2,431)</td><td>Full Compact</td><td>54,032</td><td>48,610</td><td>36,786–64,701</td><td>85,691</td></tr><tr><td>Directed (2,431)</td><td>Text-RAG</td><td>3,472</td><td>3,413</td><td>3,125–3,704</td><td>3,973</td></tr><tr><td>Directed (2,431)</td><td>Schema-RAG</td><td>2,511</td><td>2,506</td><td>2,412–2,604</td><td>2,704</td></tr><tr><td>Top-5 (639)</td><td>Full Compact</td><td>112,534</td><td>114,423</td><td>100,636–127,501</td><td>136,319</td></tr><tr><td>Top-5 (639)</td><td>Text-RAG</td><td>7,600</td><td>7,535</td><td>7,155–7,968</td><td>8,463</td></tr><tr><td>Top-5 (639)</td><td>Schema-RAG</td><td>5,333</td><td>5,323</td><td>5,156–5,508</td><td>5,687</td></tr></table>

Table 13: RQ2 input+output tokens per retained Reader call. Each task contributes one call per condition. Q1–Q3 denotes the interquartile interval; P90 is the 90th percentile. Quantiles use linear interpolation; displayed values are rounded to the nearest token. Schema-RAG is SPRA-based access.

<table><tr><td>Access condition</td><td>Input representation</td><td>Overall (0–5)</td></tr><tr><td>Full Compact Complete text</td><td></td><td>2.965</td></tr><tr><td>Text-RAG</td><td>Retrieved passages</td><td>2.886</td></tr><tr><td></td><td>Schema-RAG SPRA research objects</td><td>2.898</td></tr><tr><td></td><td>and relations</td><td></td></tr></table>

Table 14: RQ3 conditions and independently judged Overall means across 1,001 variants (0–5). Means weight variants equally; the paired contrasts in Table 15 weight benchmark families equally.

Full Compact supplies complete text, Text-RAG supplies retrieved passages, and Schema-RAG supplies SPRA research objects and internal relations. These conditions use the same paper selections and order.

Tables 14 and 15 report the question-weighted means and family-balanced paired tests. Overall is independently judged rather than averaged from dimensions. Intervals use a 50,000-draw benchmarkfamily cluster bootstrap; paired tests enumerate all 2<sup>14</sup> family sign flips.

Input+output accounting. Table 16 sums input and output tokens for digestion, generation, and binding. Shared selection, reference generation, auditing, and review are outside this task-generation measure.

Storage, retrieval, and maintenance are supplied by SciVerse.

Evaluation scope. The studies use shared benchmark data and evaluators to define inference clusters. Some tasks transfer benchmark task forms to public datasets rather than reproduce complete source experiments.

## B.5 Statistical Inference

RQ1 uses 10,000 domain–family stratified pairedbootstrap replicates, preserving the pairedcomparison principle (Koehn, 2004). RQ2 uses 10,000 multi-membership paper-cluster percentile replicates: unique papers are sampled multinomially, and each task receives the mean multiplicity of its member papers. Both preserve within-task condition pairing. RQ3 averages the three Overall reviews within each task–condition and equally weights cluster means, with each cluster defined by shared benchmark data and an evaluator. The statistical analysis uses 50,000 benchmark-cluster percentile-bootstrap draws, enumerates all paired cluster sign flips for the exact test, and applies Holm correction across the three Overall contrasts (Holm, 1979).

The paired analyses use tasks with results under all three conditions and the specified evaluators. Appendix D describes numerical replay and supporting materials.

## B.6 Citation-Guided Research Integration

Across 1,001 variants from 14 benchmark families and seven domains, both conditions read SPRA objects from the first two condition-neutral selected papers. The expanded condition additionally opens up to two resolved outgoing one-hop citation targets with their recorded roles. Seeds, prompt, and 10,000-token ceiling are shared; the contrast tests relation-guided integration within this budget.

<table><tr><td>Contrast</td><td>Difference</td><td>95% cluster-bootstrap interval</td><td>Exact p</td><td>Holm p</td></tr><tr><td>Schema-RAG-Full Compact</td><td>-0.067</td><td> $[ - 0 . 1 0 2 , - 0 . 0 3 4 ]$ </td><td>0.0020</td><td>0.0059</td></tr><tr><td>Schema-RAG-Text-RAG</td><td>+0.011</td><td> $\left[ - 0 . 0 2 9 , + 0 . 0 5 8 \right]$ </td><td>0.6792</td><td>0.6792</td></tr><tr><td>Text-RAG-Full Compact</td><td>-0.078</td><td> $\left[ - 0 . 1 2 9 , - 0 . 0 3 1 \right]$ </td><td>0.0096</td><td>0.0193</td></tr></table>

Table 15: RQ3 paired Overall contrasts across 1,001 variants, expressed in score points. Intervals use the 50,000- draw benchmark-family cluster bootstrap; exact p values enumerate all 14-family sign flips. Each of the 14 benchmark families has equal weight in the paired estimate; Holm correction covers the three contrasts.
<table><tr><td>Workflow</td><td>Input+output (millions)</td><td>Reduction</td></tr><tr><td>Full Compact</td><td>155.92</td><td></td></tr><tr><td>Text-RAG</td><td>11.87</td><td>92.4%</td></tr><tr><td>Schema-RAG</td><td>9.77</td><td>93.7%</td></tr></table>

Table 16: RQ3 input+output token accounting across the generation workflow. Reductions are relative to Full Compact.

Selection and generation. Task-keyword overlap selects seed objects with research-role coverage and a shared per-paper cap; internal edges join visible endpoints. Citation neighbors rank by keyword overlap in target metadata and role evidence, with paper-ID tie-breaking. The first two available records among the top ten are opened; absent neighbors leave seeds unchanged. Generation uses qwen3.5-397b-a17b, temperature 0, disabled thinking, and 2,800 output tokens. Proposals bind hypotheses and plans to visible object and dependency evidence.

The contrast evaluates the citation-guided expansion workflow as a whole: it changes access to neighboring evidence. The study does not isolate the contribution of relation labels or compare alternative neighbor-selection strategies.

Scientific quality. MiniMax-M2.5 and glm-5.1 independently score proposals against the same union of source excerpts, with condition labels and execution outcomes hidden. Reported means average both reviewers after the individual-rating transformation specified in Appendix B.

We report cross-paper integration directly and summarize the four scientific dimensions with an equal-weight descriptive composite:

$$
S _ { \mathrm { s c i } } = ( N + G + I + Q ) / 4 ,\tag{1}
$$

where N is novelty, G evidence grounding, I crosspaper integration, and Q qualification handling. This descriptive composite and the independently judged Overall score both use 0–5. Table 17 gives the component scores and family-balanced contrasts.

Paired inference and evidence use. We average paired differences within each of 14 benchmark families, weight families equally, and bootstrap families 10,000 times. The full cohort includes the 28-question pilot and 973 remaining questions. The latter’s integration contrast is +0.254/5 $( [ + 0 . 1 9 8 , + 0 . 3 1 0 ] )$ , positive in all families; the composite $\mathrm { i s + 0 . 1 0 0 ( [ + 0 . 0 6 1 , + 0 . 1 4 1 ] ) }$ , positive in 13. Opened neighbors appear in 932/1,001 expanded proposals.

Reproducibility materials. The accompanying evaluation package contains 4,004 numeric reviewer records, frozen citation edges, study protocol, and full/remaining-cohort estimates in data/data/supplementary/paper\_results. json. The reported composites and intervals can be inspected alongside the underlying modelreviewer scores. The analysis includes all complete proposals and records reference conformance separately.

<table><tr><td>Metric</td><td>Seed</td><td>Expanded</td><td>Δ</td><td>95% CI</td></tr><tr><td>Cross-paper integration</td><td>1.779</td><td>2.032</td><td>+0.253</td><td>[+0.195, +0.308]</td></tr><tr><td>Novelty</td><td>2.100</td><td>2.172</td><td>+0.072</td><td>[+0.037, +0.105]</td></tr><tr><td>Evidence grounding</td><td>1.848</td><td>1.931</td><td>+0.083</td><td>[+0.029, +0.139]</td></tr><tr><td>Qualification</td><td>2.750</td><td>2.744</td><td>-0.006</td><td>[−0.076, +0.055]</td></tr></table>

Table 17: Scientific synthesis with SPRA relation expansion across all 1,001 paired questions; absolute scores use 0–5 and paired differences use score points. ∆ is the equal-weight mean over 14 benchmark families; brackets give 95% family-clustered bootstrap intervals. Seed and Expanded both use SPRA objects.

## C Research Objects in Use

The three cases are drawn from the publicly available papers on speech-emotion unlearning (Ren et al., 2025), personalized federated learning (Song et al., 2025), and multiscale network entropy (Brzovic et al., 2025). They illustrate how SPRA connects each paper’s problem and contribution to its evaluation and findings, making the research argument easier to inspect and reuse.

Machine unlearning. The speech-emotion unlearning study of Ren et al. (2025) asks how to remove sensitive information using only the forget set, without access to retained training examples. SPRA links this Problem to the proposed forgetset-only method, its unlearning-accuracy Measure, and the reported comparison with a baseline. An agent can retrieve the contribution and inspect the conditions attached to its result.

Personalized federated learning. In personalized federated learning, the FedAGHN case links static collaboration to the proposed method and reported overhead, scalability, and robustness findings (Song et al., 2025). These connected records let an agent inspect a method alongside its evaluation.

Multiscale network entropy. The multiscaleentropy case connects the limits of a single-scale description to a proposed framework and its evaluation using $R ^ { 2 }$ , including a reported comparison of 0.92 and 0.68 (Brzovic et al., 2025).

## C.1 Application Interface

SciVerse demonstrates the SPRA interface through paper discovery, typed lookup, filtering, relation traversal, and source inspection. It presents research units and their relationships around a question, with links to source passages and tables. An offline walkthrough is included in the accompanying materials; RQ1–RQ3 evaluate scientific task performance.

## C.2 From a Research Question to Reusable Objects

Consider the speech-emotion unlearning example in Appendix C. A researcher asks which proposed method removes information using only the forget set and under what evaluation conditions it succeeds. The access sequence is to discover the paper, select its Problem and Contribution, follow relations to the relevant Measure and Finding, and inspect the recorded source context. Keeping the paper and object identifiers with the selected evidence allows subsequent comparisons to refer to the same research objects.

REST access. Create a SciVerse API Token through the SciVerse account console and provide it as SCIVERSE\_API\_TOKEN in the local environment. Requests use a Bearer authorization header. The Paper Schema API accepts JSON request bodies for its POST operations. The examples use https://api.sciverse.space as the API base URL and /paper-schema as the service prefix. With the token configured, a paper search can be issued directly from a shell:

```shell
BASE=https://api.sciverse.space
TOKEN="${SCIVERSE_API_TOKEN:?Set token}"
AUTH="Authorization: Bearer $TOKEN"
curl --fail-with-body \
-X POST "$BASE/paper-schema/search" \
-H "$AUTH" \
-H "Content-Type: application/json" \
-d '{"query":"speech emotion unlearning",
"size":5}'
```

The response contains items, with schema\_id and paper metadata for each candidate, and total, which reports the match count and whether it is exact or a lower bound. Select the intended paper by its title and metadata. The following Python helper sends authenticated JSON requests:

```python
base = "https://api.sciverse.space"
token = os.environ["SCIVERSE_API_TOKEN"]
headers = {
"Authorization": "Bearer " + token,
"Content-Type": "application/json"
}
def post(route, body):
request = urllib.request.Request(
base + "/paper-schema/" + route,
data=json.dumps(body).encode(),
headers=headers)
with urllib.request.urlopen(
request, timeout=60) as response:
return json.load(response)
```

Enter the selected schema\_id from the search response to retrieve method-oriented materials:

```python
selected_id = input("schema_id: ").strip()
materials = post("materials", {
"schema_ids": [selected_id],
"goal": "method",
"include_relation_context": True,
"per_schema_entity_limit": 40,
"per_schema_relation_limit": 40,
"per_schema_attribute_limit": 40
})
```

Each returned item contains schema\_paper, entities, relations, evidence, and terms. Per-resource counts expose total, returned, and truncated. A material package selects content for the requested goal; a complete object listing requires pagination through the corresponding entity or relation interfaces. The capability endpoint lists the current service types; callers should use these names when selecting resources.

For an object or relation with source markers, /paper-schema/resolve-provenance accepts its schema\_id and returned marker\_nums, together with a bounded context window. Preserve these returned locators; do not derive them from section numbers. The response provides source segments and unresolved locators. Inspect warnings when partial is true and retrieve additional pages or narrower selections when truncated is true. These examples specify the access procedure; the paper’s quantitative findings come from the frozen evaluations.

Capabilities and errors. An authenticated GET to /paper-schema returns the current contract version, supported resource types, and limits. HTTP 401 indicates an authentication problem; HTTP 403 indicates insufficient account permissions. For HTTP 429, clients should wait according to Retry-After or the response’s retry\_after value. The returned X-Request-ID supports request tracing.

Agent access. The Paper Schema Agent Skill documentation provides installation and configuration instructions. An agent can use search\_paper\_schemas, query\_paper\_ entities, query\_paper\_relations, and resolve\_paper\_context for the same sequence, or build\_paper\_materials for task-specific selection. Both interfaces retain paper-scoped identities. Related-paper discovery and resolved citation traversal support expansion to other papers; names shared across papers do not by themselves establish object identity.

## D Reproducibility and Resource Use

## D.1 Evaluation Materials and Reanalysis

The supplements provide code, tests, model settings, source locators, corpus statistics, task-level results, SPRA examples, and frozen inputs for offline inspection and reanalysis under the terms in Appendix A.2.

software.zip contains extraction code, prompts, schemas, analysis and replay scripts, and tests. data.zip contains study records and fixtures plus the complete 9,849-paper aligned RQ1 cohort, with complete-text, retrieved-passage, and SPRA inputs for each item. The archive also records the 9,996-paper candidate catalog and 147 exclusion records: 12 candidates lack a positive Text-RAG candidate, and 135 are removed to balance domain–task cells. Each eligible paper has one saved three-representation input set; studyspecific complete-case counts remain as reported in Appendix B. Matched inputs support offline inspection; saved study outputs support numerical reanalysis without API access. The archives create sibling software/ and data/ directories and include file checksums. The aligned inputs are in data/data/main/; supporting records are in data/data/supplementary/.

Offline reproduction. Extract both archives into the same parent directory. From their shared parent directory, run the packaged consistency check and regression tests:

python3 software/software/verify\_release.py python3 -m unittest discover \

-s software/code

The complete analysis commands are documented in software/REPRODUCE.md. Numerical reanalysis uses saved records without hosted models; fresh extraction or answer generation requires source articles and hosted services.

## D.2 Executable Implementation and Tests

The SPRA extraction implementation is included with the evaluation code. Its prompts and JSON schemas define the extraction stages; assembly code validates identifiers, relation endpoints, and source bindings. Appendix A.3 describes the construction procedure.

The regression tests check selected reported values and their arithmetic against fixtures in the data archive. software/REPRODUCE.md documents the offline commands and the source articles and hosted services required for fresh model runs.

## D.3 Model Configurations and Request Settings

Llama 4 Scout is the primary Reader for RQ1–RQ3; DeepSeek-v4-pro generates sourcegrounded RQ1/RQ2 references and audits the RQ3 implementation; MiniMax-M3 scores RQ1/RQ2 answers against shared references; and MiniMax-M2.5, Qwen3.5-27B, and GLM-5.1 independently review RQ3 proposals. The supplementary data/data/supplementary/model\_ settings.json records model aliases, role-level output ceilings, and backend-specific request options. Primary evaluation calls use temperature 0; these per-call settings are distinct from Reader input-context budgets.

Construction configuration. The reference extractor uses qwen3.5-35b-a3b, temperature 0, disabled thinking, and output ceilings of 32,768 tokens for content and 24,576 for planning auxiliaries. These settings describe the reference extractor; the corpus statistics summarize the deployed collection rather than a single-model construction run.

Evaluation request limits. The frozen registry records temperature 0 and top-p 1 for RQ1–RQ3. RQ1 output ceilings are 24,000 tokens for the Reader, 8,000 for references, and 2,400 for judging. RQ2 uses 3,500/6,000 for directed/Top-5 answers, 12,000 for references, and 2,400 for judging. RQ3 uses 7,000 for digestion, 6,500 for candidate generation, 1,800 for binding, 2,200 for implementation assessment, and 2,400 for review. These are configured output limits; actual consumption is reported separately. Hosted APIs manage model hardware, so the recorded aliases and request settings identify the evaluated services.

## D.4 Task-Based Assessment of Extracted Content

RQ1 assesses the functional adequacy of the extracted representation through source-conditioned scientific QA and structured operations. DeepSeekv4-pro generates shared answer references from Full Compact, the complete parsed-text input; MiniMax-M3, blind to condition, scores Reader responses against them. G-U1 measures correctness, coverage, and qualification against these references; G-U3 focuses on Problem + Contribution. Source traceability and complete relation endpoints support inspection and research-object operations. The supplementary materials also document the sampled object assessment.

## D.5 Token-Accounting Scope

Storage, retrieval, and maintenance are supplied by SciVerse. Token totals measure complete inputs and outputs for the retained task calls. They quantify the evaluated workflows’ text volume; they do not represent the service bill or include storage and retrieval costs. Appendix B specifies the included stages and excluded calls for each study.