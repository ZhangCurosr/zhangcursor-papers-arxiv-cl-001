# CiteGuard-RAG: A Validation-Centered AI System for Evidence-Grounded Question Answering

Sumit Barua, Guan Hong, Halil Dursunoglu, Charles Rodgers, Alvis Fong

Department of Computer Science, Western Michigan University, Kalamazoo, MI, 49008, USA.

\*Corresponding author(s). E-mail(s): sumit.barua@wmich.edu;

## Abstract

Retrieval-augmented generation (RAG) can improve access to complex information; however, retrieving evidence alone does not ensure that answers are grounded, citation-valid, or appropriately refused. This paper introduces CiteGuard-RAG, a validation-centered AI system for evidence-grounded question answering. The system integrates hybrid semantic–lexical retrieval, citationconstrained generation, sentence-level grounding validation, and single-pass regeneration. Validation is used at runtime to determine whether a candidate answer should be accepted, refused, or regenerated before final delivery. CiteGuard-RAG is evaluated on 400 questions across a controlled housing-law dataset, PrivacyQA, and CUAD. In the controlled evaluation, it achieves 99.1% retrieval accuracy, 98.3% grounded-answer accuracy, and 98.3% citation validity, with no validation-detected hallucinations. Ablation results show that groundedanswer accuracy drops sharply when validation is removed, even when retrieval accuracy remains unchanged. External evaluation shows that while citation validity remains strong, evidence utilization, span alignment, and refusal calibration become harder under domain shift. These findings indicate that trustworthy RAG systems require explicit validation between retrieval and final answer delivery. CiteGuard-RAG provides a practical architecture for linking retrieval, generation, citation checking, abstention, and regeneration in high-stakes information access.

Keywords: Retrieval-Augmented Generation, Evidence-Grounded Question Answering, Citation Validation, Hallucination Mitigation

## 1 Introduction

## 1.1 Background and Motivation

Large language models (LLMs) have improved natural language understanding and generation, but their use in high-stakes information access remains limited by hallucination, weak source traceability, and unreliable refusal behavior [1, 2]. These limitations are especially important in legal question answering, where answers must remain faithful to source documents and unsupported claims can mislead users.

RAG addresses part of this problem by conditioning LLM outputs on retrieved evidence rather than relying only on parametric memory [3, 4]. However, retrieval success does not guarantee grounded generation. Even when relevant passages are retrieved, a model may omit key evidence, overgeneralize from partial context, merge retrieved text with parametric knowledge, or produce unsupported elaborations [5].

Legal information access further amplifies these challenges because legal corpora are large, heterogeneous, and structurally complex [6]. Legal texts often contain hierarchical organization, specialized language, exceptions, and argumentation patterns that complicate retrieval, chunking, and evidence grounding [7]. As a result, trustworthy legal RAG systems require explicit mechanisms for verifying whether generated claims are supported by cited evidence, rather than assuming that retrieval alone ensures reliability [8].

## 1.2 Literature Review and Research Gaps

## 1.2.1 Retrieval Strategies

Hybrid retrieval methods that combine lexical methods such as BM25 with dense semantic retrieval have been widely studied in open–domain question answering, and the findings show that lexical methods are better at exact term matching, while dense methods are better at capturing semantic similarity but can retrieve topically related but non-specific evidence [9, 10]. In the legal domain, transformer–based rerankers designed for very long queries and documents showed considerable performance gains while keeping computational eficiency [11]. Retrieval partitioning strategies have also proven that the organization and incorporation of recovered content greatly improve the quality of downstream generations [12]. Despite these developments, the interaction of retrieval strategies and grounding quality remains insuficiently understood in legal RAG systems.

## 1.2.2 Grounding and Hallucination

A recurrent dificulty for RAG systems is that retrieval quality alone does not guarantee grounded generation; models can hallucinate when retrieved evidence is marginally relevant or contradicts parametric knowledge [5]. This dificulty is exacerbated by the “lost-in-the-middle” efect, where LLMs sufer substantial degradation when crucial information is located in the middle of long input contexts. This phenomenon suggests that models do not robustly employ all returned evidence [13]. To mitigate hallucinations in legal QA, constrained RAG techniques have been developed that leverage aspect-based constraints throughout the retrieval and generation stages, demonstrating improvements in terms of accuracy and relevance of retrieved documents [14]. Likewise, systems based on natural language inference (NLI) and reinforcement learning have demonstrated the capacity of entailment-based validation to enhance the factual consistency of generated text [15]. However, prior work does not have a unified framework to impose grounding during generation and verify it at fine–grained, sentence–level resolution simultaneously.

## 1.2.3 Evaluation

The evaluation of RAG systems is dificult due to the connection between retrieval and generation. Standard end-to-end metrics often conflate retrieval mistakes with generation failures, such that system weaknesses are obscured [16, 17]. Benchmarking studies have also demonstrated that RAG systems still sufer from robustness, information integration, and negative rejection behavior [18]. Recent surveys in the legal domain indicate the importance of specialized evaluation metrics that measure faithfulness, citation validity, and hallucination at fine granularity [8]. In-context retrieval-augmented models show enhanced grounding and source attribution [4], but there is no systematic evaluation of grounding reliability across system components.

## 1.2.4 Research Gaps

The existing literature leaves four main gaps. First, it remains unclear how efectively RAG systems can enforce evidence-grounded and citation-valid answers in high-stakes information access tasks, especially under domain shift. Second, the efect of lexical, semantic, and hybrid retrieval strategies on downstream grounding and hallucination behavior is not yet well characterized. Third, existing validation approaches often remain post-hoc, answer-level, or evaluation-oriented, leaving limited understanding of how sentence-level citation and evidence checks can control whether a candidate answer is accepted, regenerated, or refused before final delivery. Fourth, retrieval, generation, validation, and regeneration are often evaluated as separate components, even though their interaction determines final answer reliability.

## 1.3 Proposed Approach

In this paper, we propose CiteGuard-RAG, a validation-centered intelligent information system for evidence-grounded question answering. The framework combines page-aware document parsing, paragraph-aware chunking, hybrid semantic–lexical retrieval, citation-constrained generation, sentence-level grounding validation, and one-pass validation-guided regeneration. The overall system design is shown in Figure 1.

The central idea is to treat grounding as a runtime control problem rather than only as a post-hoc evaluation step. The pipeline first generates an internal candidate answer and then checks whether its claims are supported by cited retrieved evidence before any response is delivered. The validation result determines whether the candidate is accepted as the final answer, revised through one-pass regeneration, or replaced with a standardized refusal.

![](images/9c240910907e602e4e1b679b3807d4d9c0ca2f472c8ca7f8c6344e358663aed8.jpg)  
Fig. 1 CiteGuard-RAG pipeline for evidence-grounded question answering. The framework integrates paragraph-aware chunking, hybrid semantic–lexical retrieval with score fusion, citationconstrained generation, post-generation grounding validation, and one-pass validation-guided regeneration.

## 1.4 Contributions

Motivated by the research gaps above, this study examines how retrieval, generation, validation, refusal, and regeneration interact in an evidence-grounded RAG system. The evaluation focuses on whether citation-grounding validation improves answer reliability, how retrieval and design choices afect grounding behavior, and what failure modes emerge under controlled and external domain-shift settings. The main contributions of this work are:

1. CiteGuard-RAG, a validation-centered intelligent information system that connects retrieval, citation-constrained generation, grounding validation, abstention, and one-pass regeneration for evidence-grounded question answering.

2. A sentence-level citation-grounding validation mechanism that jointly checks citation validity, lexical evidence overlap, and semantic support before final answer delivery.

3. Empirical analysis of retrieval–generation–validation interaction, showing that high retrieval accuracy alone does not guarantee grounded or citation-valid answers.

4. Multi-setting evaluation over 400 questions across a controlled housing-law dataset, PrivacyQA, and CUAD, allowing in-domain performance to be compared with external domain and task-shift behavior.

5. Failure case analysis identifying key reliability bottlenecks, including false refusal, citation/evidence-compliance failures, answer-span mismatch, and evidenceutilization errors.

## 2 Methodology

## 2.1 System Overview

Given a document corpus D and a user query q, the system retrieves a ranked evidence set $E \subset D$ and generates an internal candidate answer $ { \boldsymbol { a } } _ { c }$ . The candidate answer is

not delivered directly. Instead, each substantive claim is checked against the retrieved evidence and its associated citation. The final output is either a validated answer a or a standardized refusal. The pipeline is defined as:

D → Parsing → Chunking → Indexing

→ Retrieval → Candidate Generation → Validation

→ Accept/Regenerate/Refuse.

Algorithm 1 summarizes the proposed citation-constrained RAG workflow, including retrieval, evidence filtering, validation, and one-pass regeneration.

Algorithm 1 CiteGuard-RAG With Runtime Validation and One-Pass Regeneration   
Require: Query q, document corpus D, retriever R, generator G, validator V   
Ensure: Validated final answer a or standardized refusal   
1: Parse D into page-aware text units   
2: Construct paragraph-aware evidence chunks {C<sub>i</sub>}   
3: Retrieve hybrid-ranked evidence candidates $\dot { E ^ { \prime } } = R _ { \mathrm { h y b r i d } } ( q , \{ C _ { i } \} )$   
4: Deduplicate candidates and select top-k evidence $E \subseteq E ^ { \prime }$   
5: if E is insuficient then   
6: return standardized refusal   
end if   
8: Generate internal candidate answer $a _ { 0 } ~ = ~ G ( q , E )$ using citation-constrained   
prompting   
9: Compute validation result $v _ { 0 } = V ( a _ { 0 } , E )$   
10: if v passes then   
11: return validated final answer $a _ { 0 }$   
12: else   
13: Generate regenerated candidate $a _ { 1 } = G ( q , E , v _ { 0 } )$ using validation feedback   
14: Compute validation result $v _ { 1 } = V ( a _ { 1 } , E )$   
15: if v<sub>1</sub> passes then   
16: return validated final answer a   
17: else   
18: return standardized refusal   
19: end if   
20: end if

## 2.2 Document Parsing and Chunk Construction

Input legal documents are processed using page-aware text extraction to preserve evidence provenance. Each extracted page is represented as:

$$
P _ { i } = \{ { \mathrm { t e x t } } _ { i } , { \mathrm { p d f } } _ { \mathrm { - p a g e } _ { i } } , { \mathrm { p r i n t e d } } \mathrm { - p a g e } _ { i } \} ,
$$

where $\mathrm { t e x t } _ { i }$ is normalized page text, pdf page is the internal PDF page index, and printed page is an optional human-visible page number recovered from the document when available. When printed page numbers cannot be reliably detected, the system falls back to the canonical PDF page index. This preserves traceability across retrieval, citation display, and evaluation.

Documents are segmented using a paragraph-aware chunking strategy. Paragraph boundaries are preferred when preserved by PDF extraction; otherwise, a sentencelevel fallback is applied using punctuation-based boundary detection. Adjacent text units are aggregated into chunks under a maximum token budget with overlap to preserve local context across chunk boundaries. To avoid semantically difuse evidence, chunk expansion is constrained by page span, and low-information fragments such as navigation text, tables of contents, and very short segments are removed. Each final chunk is stored as:

$$
C _ { j } = \{ { \mathrm { t e x t } } _ { j } , { \mathrm { c h u n k . i d } } _ { j } , { \mathrm { p d f . r a n g e } } _ { j } , { \mathrm { p r i n t e d . r a n g e } } _ { j } \} .
$$

## 2.3 Hybrid Retrieval

The retrieval layer combines dense semantic retrieval and sparse lexical retrieval. Dense retrieval encodes each chunk $C _ { i }$ and query q using a sentence-transformer embedding model $f ( \cdot )$

$$
\mathbf { c } _ { i } = f ( C _ { i } ) , \qquad \mathbf { q } = f ( q ) .
$$

Semantic relevance is computed using cosine similarity:

$$
\mathrm { S i m } _ { \mathrm { s e m a n t i c } } ( q , C _ { i } ) = \frac { \mathbf { q } \cdot \mathbf { c } _ { i } } { \| \mathbf { q } \| \| \mathbf { c } _ { i } \| } .
$$

In parallel, lexical relevance is estimated using BM25:

$$
{ \mathrm { S c o r e } } _ { { \mathrm { l e x i c a l } } } ( q , C _ { i } ) = \sum _ { t \in q } { \mathrm { I D F } } ( t ) \cdot { \frac { t f _ { t , i } ( k _ { 1 } + 1 ) } { t f _ { t , i } + k _ { 1 } \left( 1 - b + b \cdot { \frac { | C _ { i } | } { { \mathrm { a v g d } } } } \right) } } .
$$

Semantic and lexical scores are min–max normalized and fused as:

$$
\mathrm { S c o r e } _ { \mathrm { h y b r i d } } = \alpha \hat { s } _ { \mathrm { s e m a n t i c } } + ( 1 - \alpha ) \hat { s } _ { \mathrm { l e x i c a l } } ,
$$

where α controls the semantic–lexical trade-of. This hybrid formulation supports exact legal phrase matching while retaining semantic flexibility for paraphrased queries.

## 2.4 Candidate Filtering and Optional Reranking

Retrieved candidates are deduplicated using normalized text similarity to reduce repeated evidence caused by overlapping chunks and improve context diversity before generation.

A lightweight heuristic reranker based on query overlap and legal cue words was evaluated as an optional module. However, because reranking reduced performance

in the controlled evaluation, the final system uses deduplicated hybrid retrieval without reranking. Neural cross-encoder rerankers are not used, as this study focuses on lightweight, locally deployable RAG components.

## 2.5 Controlled Evidence-Grounded Generation

Retrieved chunks are formatted as numbered source blocks and passed to the language model using a constrained prompt. The prompt requires the model to answer only from retrieved sources, cite supporting evidence using inline source markers, avoid external legal knowledge, and refuse when suficient evidence is unavailable. The standardized refusal response is:

The document does not contain suficient information to answer this question.

This prompting strategy constrains the language model to function as a sourcegrounded answer generator rather than a free-form legal advisor. For external datasets with diferent document types, the same principle is preserved: answers must be grounded in retrieved evidence, citations must point to valid retrieved sources, and unsupported claims trigger refusal or regeneration.

## 2.6 Post-Generation Validation and Regeneration

The validation layer checks whether each internally generated candidate answer is traceable to retrieved evidence before any answer is delivered as the final response. Candidate answers, including regenerated candidates, are decomposed into sentences, and each sentence is evaluated for citation validity and grounding support. Figure 2 illustrates the sentence-level validation workflow used to route candidate outputs toward final acceptance, one-pass regeneration, or refusal.

A citation is considered valid only if it refers to one of the retrieved source blocks. Lexical grounding is computed using token overlap between a candidate sentence and its cited evidence:

$$
\mathrm { O v e r l a p } ( s , e ) = \frac { | T _ { s } \cap T _ { e } | } { | T _ { s } | } ,
$$

where $T _ { s }$ and $T _ { e }$ denote normalized token sets for the candidate sentence and cited evidence, respectively.

Semantic grounding is computed using sentence embeddings:

$$
\mathrm { S i m } _ { \mathrm { s e m a n t i c } } ( s , e ) = { \frac { \mathbf { s } \cdot \mathbf { e } } { \| \mathbf { s } \| \| \mathbf { e } \| } } .
$$

A sentence is considered supported if it satisfies citation validity and passes either the lexical or semantic grounding threshold. The lexical-overlap and semantic-similarity thresholds were set to 0.10 and 0.40, respectively, based on pilot validation runs to balance strict hallucination control against valid paraphrased answers. These values are tunable implementation parameters rather than fixed theoretical constants.

![](images/d47c2eebeead57807507fced3b395a07b7e7c52dc0aa06bf69a26e5640214489.jpg)  
Fig. 2 Sentence-level grounding validation pipeline. The workflow verifies citations, lexical overlap, and semantic similarity against retrieved evidence, then routes the candidate output toward acceptance, one-pass regeneration, or refusal.

A candidate response is flagged as hallucinated if it contains one or more unsupported claims. The supported sentence ratio is computed as:

$$
{ \mathrm { S u p p o r t e d R a t i o } } = { \frac { n - u } { n } } ,
$$

where n is the total number of candidate sentences and u is the number of unsupported sentences.

When validation detects a grounding-risk signal, the system performs a single regeneration attempt using a stricter evidence-grounded prompt. Regeneration is triggered by invalid citation behavior, weak evidence support, unsupported content, malformed citation behavior, or refusal despite strong retrieved context. The regenerated candidate is accepted only if it improves validation compliance or, when evidence is insuficient, produces a clean refusal. Regeneration is limited to one pass to control inference cost and avoid unbounded correction loops. Thus, validation afects both safety and coverage: unsupported candidate answers are blocked before delivery, but valid answers may also be rejected when the system behaves conservatively.

## 2.7 Controlled Evaluation Set

The controlled evaluation set consists of 150 question–answer pairs derived from 18 housing-law documents across multiple U.S. states and cities. Each question is associated with gold evidence pages and a refusal label indicating whether the document contains suficient information to answer the query. The dataset includes 123 answerable questions and 27 unanswerable questions. This design supports evaluation of both answer generation and abstention behavior.

To reduce formulation bias, the evaluation set combines 100 human-authored questions with 50 LLM-assisted questions that were manually reviewed, edited, and verified against the source documents. Questions are stratified across factual, procedural, reasoning, and edge-case categories. Table 1 reports the question-type distribution.

Table 1 Distribution of controlled evaluation question types
<table><tr><td>Category</td><td>Count</td><td>Description</td></tr><tr><td>Factual</td><td>51</td><td>Direct questions asking for explicit information stated in the source document.</td></tr><tr><td>Procedural</td><td>40</td><td>Questions asking about steps, rights, obligations, dead- lines, or required actions.</td></tr><tr><td>Reasoning</td><td>32</td><td>Questions requiring interpretation or combination of related evidence from the retrieved text.</td></tr><tr><td>Edge Cases</td><td>27</td><td>Questions involving exceptions, unavailable evidence, ambiguous conditions, or refusal behavior.</td></tr></table>

## 2.8 External Validation Design

External validation is conducted using two complementary datasets. PrivacyQA [19] is used as the primary external QA-style validation dataset because it preserves the natural document-grounded question-answering structure of the controlled housinglaw evaluation while shifting to privacy-policy documents. PrivacyQA consists of user questions about privacy policies with expert-labeled relevant policy segments. We evaluate a fixed 200-question subset and treat expert-labeled relevant segments as gold evidence, allowing assessment of retrieval coverage, citation validity, evidence support, and regeneration beyond the housing-law corpus.

CUAD [20] is used as a supplementary contract-domain stress test rather than as a direct natural-QA benchmark. CUAD difers from the controlled evaluation because it asks for contract clause identification and span-like evidence, introducing both legal domain shift and task-formulation shift. We evaluate a fixed 50-question stratified subset containing answerable and unanswerable clause-extraction examples to assess retrieval coverage, citation validity, abstention behavior, hallucination control, and regeneration under contract-domain shift.

## 2.9 Implementation Details

The system was implemented as a modular Python pipeline. Dense embeddings were generated using BAAI/bge-base-en-v1.5 [21], while lexical retrieval used BM25 [22]. Hybrid retrieval combined min–max normalized semantic and lexical scores using a 0.65/0.35 weighting scheme. The final controlled configuration used 600-token chunks with 120-token overlap, top-k = 8, and up to four retrieved chunks passed to the generator. The heuristic reranker was disabled in the final system because it reduced controlled-setting performance.

Generation was performed using locally deployable instruction–tuned models under deterministic decoding with temperature set to 0. When validation failed, the system allowed at most one regeneration attempt using validation feedback. For external validation, the same RAG architecture, citation requirements, validation framework, and one-pass regeneration mechanism were retained, with dataset-specific preprocessing applied only to align PrivacyQA and CUAD with their native evidence structures.

## 2.10 Evaluation Protocol

Each controlled housing-law test instance is represented as:

$$
( q , y _ { \mathrm { r e f u s e } } , G ) ,
$$

where $q$ is the query, $y _ { \mathrm { r e f u s e } } \in \{ 0 , 1 \}$ indicates whether the system should abstain, and G denotes the gold evidence pages. For PrivacyQA, G corresponds to expert-labeled relevant policy segments. For CUAD, G corresponds to annotated contract evidence spans or relevant clause text.

Evaluation separates retrieval coverage, answer correctness, citation grounding, hallucination control, abstention behavior, and regeneration behavior. This separation allows retrieval failures to be distinguished from generation, validation, and refusalcalibration failures. The main metric groups are summarized in Table 2.

Two hallucination-related metrics are reported. Hallucination Rate measures broader answer-level failures, including incorrect, overextended, benchmarkinconsistent, or evidence-inconsistent final responses under the dataset-specific evaluation criteria. Unsupported Hallucination Rate is narrower and measures final responses that contain claims not supported by the retrieved and cited evidence. Thus, a response may count toward the broader hallucination rate when it fails a benchmark answer criterion, while not counting as an unsupported hallucination if its claims remain grounded in cited evidence.

Table 2 Evaluation metric groups
<table><tr><td>Metric group</td><td>Metrics and purpose</td></tr><tr><td>Retrieval coverage</td><td>Page Hit@k, Span/Segment Hit@k, and Context Recall@k measure whether gold evidence is retrieved and included in the generation context.</td></tr><tr><td>Answer correctness</td><td>Grounded Answer Accuracy and Token F1 measure whether the final answer is correct, evidence-aligned, and sufficiently overlaps with gold evidence or expert-labeled segments.</td></tr><tr><td>Citation and evidence support</td><td>Citation Validity and Evidence Support Rate measure whether final answers cite valid retrieved sources and satisfy lexical or semantic grounding checks.</td></tr><tr><td>Hallucination control</td><td>Hallucination Rate measures broader incorrect, overex- tended, benchmark-inconsistent, or evidence-inconsistent final responses. Unsupported Hallucination Rate measures claims not supported by retrieved and cited evidence.</td></tr><tr><td>Abstention behavior</td><td>True Refusal Rate, False Refusal Rate, and False Answer Rate measure whether the system appropriately refuses when evidence is insufficient and avoids answering unanswerable</td></tr><tr><td>Regeneration behavior</td><td>questions. Regeneration Attempt Rate and Regeneration Accepted Rate measure how often validation triggers regeneration and how often the regenerated candidate is accepted as the final output.</td></tr></table>

## 3 Results

## 3.1 Overall System Performance

Table 3 reports the performance of the final selected system configuration on the controlled housing-law evaluation set.

Table 3 Overall system performance under the final selected configuration
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Retrieval Accuracy</td><td>99.1%</td></tr><tr><td>Grounded Answer Accuracy</td><td>98.3%</td></tr><tr><td>Citation Validity</td><td>98.3%</td></tr><tr><td>Evidence Support Rate</td><td>98.3%</td></tr><tr><td>Hallucination Rate</td><td>0.0%</td></tr><tr><td>False Refusal Rate</td><td>1.8%</td></tr><tr><td>True Refusal Rate</td><td>96.4%</td></tr><tr><td>Regeneration Rate</td><td>0.9%</td></tr></table>

The system achieves high retrieval accuracy, grounded answer accuracy, citation validity, and evidence support. No hallucination is detected under the implemented validation criteria, while the false-refusal rate remains low. The low regeneration rate indicates that most candidate answers satisfy the grounding constraints on the first generation pass.

To assess whether this behavior depends on a specific generator, we also evaluated Mistral [23] and Qwen [24] under the same retrieval, prompting, and validation configuration. As shown in Table 4, both models achieve identical performance across the primary reliability metrics.

Table 4 Controlled-setting comparison of generation models
<table><tr><td>Model</td><td>Grounded Acc.</td><td>Citation Validity</td><td>Hallucination</td><td>True Refusal</td></tr><tr><td>Mistral</td><td>98.3%</td><td>98.3%</td><td>0.0%</td><td>96.4%</td></tr><tr><td>Qwen</td><td>98.3%</td><td>98.3%</td><td>0.0%</td><td>96.4%</td></tr></table>

The identical model-level results suggest that, in the controlled setting, system behavior is primarily determined by retrieval, prompting, and validation constraints rather than by generator-specific variability.

## 3.2 Component and Configuration Analysis

Table 5 compares three embedding models under the same hybrid retrieval configuration. BAAI/bge-base-en-v1.5 achieves the strongest overall performance, with the highest retrieval accuracy and grounded answer accuracy and the lowest hallucination rate.

Table 5 Embedding model comparison
<table><tr><td>Embedding Model</td><td>Retrieval</td><td>Grounded</td><td>Hallucination</td><td>False Refusal</td></tr><tr><td>all-MiniLM-L6-v2 [25]</td><td>90.8%</td><td>89.2%</td><td>3.1%</td><td>6.2%</td></tr><tr><td>intfloat/e5-base-v2 [26]</td><td>90.8%</td><td>90.8%</td><td>3.1%</td><td>0.0%</td></tr><tr><td>BAAI/bge-base-en-v1.5 [21]</td><td>99.1%</td><td>98.3%</td><td>0.0%</td><td>1.8%</td></tr></table>

Table 6 compares lexical-only, semantic-only, and hybrid retrieval using the selected BGE embedding model. Hybrid retrieval outperforms both single-mode baselines, suggesting that legal QA benefits from combining exact phrase matching with semantic similarity.

The validation layer has the largest efect on generation reliability. As shown in Table 7, disabling validation leaves retrieval accuracy unchanged but substantially reduces grounded answer accuracy and citation validity while increasing hallucination.

This result is important for legal QA because citation validity and evidence support cannot be inferred from retrieval success alone; they require explicit post-generation verification.

Table 6 Comparison of retrieval strategies
<table><tr><td>Mode</td><td>Retrieval Acc.</td><td>Grounded Acc.</td><td>Hallucination</td><td>Exact Hit</td></tr><tr><td>Lexical Only</td><td>84.6%</td><td>83.1%</td><td>4.6%</td><td>81.5%</td></tr><tr><td>Semantic Only</td><td>92.3%</td><td>90.8%</td><td>3.1%</td><td>84.6%</td></tr><tr><td>Hybrid (0.65/0.35)</td><td>99.1%</td><td>98.3%</td><td>0.0%</td><td>99.1%</td></tr></table>

Table 7 Efect of validation layer
<table><tr><td>Setting</td><td>Retrieval Acc.</td><td>Grounded Acc.</td><td>Citation Validity</td><td>Hallucination</td></tr><tr><td>Validation ON</td><td>99.1%</td><td>98.3%</td><td>98.3%</td><td>0.0%</td></tr><tr><td>Validation OFF</td><td>99.1%</td><td>8.7%</td><td>12.3%</td><td>18.6%</td></tr></table>

Reranking was evaluated as an optional retrieval-stage refinement. However, as shown in Table 8, the heuristic reranker reduced both retrieval accuracy and grounded answer accuracy while increasing false refusals. The final configuration therefore disables reranking.

Table 8 Efect of reranker
<table><tr><td>Setting</td><td>Retrieval Acc.</td><td>Grounded Acc.</td><td>False Refusal</td><td>Exact Hit</td></tr><tr><td>Reranker OFF</td><td>99.1%</td><td>98.3%</td><td>1.8%</td><td>99.1%</td></tr><tr><td>Reranker ON</td><td>87.7%</td><td>86.2%</td><td>3.1%</td><td>83.1%</td></tr></table>

Chunk size and hybrid weighting also afect system behavior. Table 9 shows that a 600-token chunk size provides the best trade-of between context completeness and retrieval specificity.

Table 9 Efect of chunk size
<table><tr><td>Chunk Size</td><td>Retrieval Acc.</td><td>Grounded Acc.</td><td>False Refusal</td><td>Hallucination</td></tr><tr><td>400</td><td>93.8%</td><td>92.3%</td><td>1.5%</td><td>1.5%</td></tr><tr><td>600</td><td>99.1%</td><td>98.3%</td><td>1.8%</td><td>0.0%</td></tr><tr><td>800</td><td>95.4%</td><td>93.8%</td><td>3.1%</td><td>1.5%</td></tr></table>

Table 10 evaluates diferent semantic–lexical score combinations within hybrid retrieval. The 0.65/0.35 configuration achieves the best overall balance, producing the highest retrieval accuracy and grounded answer accuracy while eliminating detected hallucination under the validation criteria. More semantic–heavy retrieval at 0.80/0.20 performs strongly but increases hallucination risk, indicating that maximizing semantic dominance does not necessarily yield the safest legal-QA configuration.

Table 10 Efect of hybrid semantic/lexical weighting
<table><tr><td>Semantic</td><td>Lexical</td><td>Retrieval Acc.</td><td>Grounded Acc.</td><td>Hallucination</td><td>False Refusal</td></tr><tr><td>0.80</td><td>0.20</td><td>96.9%</td><td>96.9%</td><td>4.6%</td><td>0.0%</td></tr><tr><td>0.65</td><td>0.35</td><td>99.1%</td><td>98.3%</td><td>0.0%</td><td>1.8%</td></tr><tr><td>0.50</td><td>0.50</td><td>90.8%</td><td>90.8%</td><td>1.5%</td><td>3.1%</td></tr><tr><td>0.35</td><td>0.65</td><td>89.2%</td><td>89.2%</td><td>1.5%</td><td>1.5%</td></tr></table>

## 3.3 External Validation and Stress-Test

Table 11 reports external validation results on PrivacyQA and CUAD. These datasets evaluate whether the framework maintains retrieval, citation, grounding, refusal, and regeneration behavior outside the controlled housing-law setting.

Table 11 External validation and stress-test results
<table><tr><td>Metric</td><td>PrivacyQA</td><td>CUAD Stress Test</td></tr><tr><td>Evaluation Size</td><td>200 questions</td><td>50 questions</td></tr><tr><td>Retrieval Recall@k</td><td>93.37%</td><td>100.00%</td></tr><tr><td>Citation Validity</td><td>100.00%</td><td>100.00%</td></tr><tr><td>Evidence Support Rate</td><td>81.33%</td><td>36.00%</td></tr><tr><td>Answer Accuracy</td><td>68.50%</td><td>54.00%</td></tr><tr><td>True Refusal Rate</td><td>76.47%</td><td>92.00%</td></tr><tr><td>False Refusal Rate</td><td>18.67%</td><td>56.00%</td></tr><tr><td>False Answer Rate</td><td>23.53%</td><td>8.00%</td></tr><tr><td>Hallucination Rate</td><td>22.00%</td><td>26.00%</td></tr><tr><td>Unsupported Hallucination Rate</td><td>0.00%</td><td>6.00%</td></tr><tr><td>Regeneration Attempt Rate</td><td>30.00%</td><td>28.00%</td></tr><tr><td>Regeneration Accepted Rate</td><td>29.50%</td><td>16.00%</td></tr></table>

On PrivacyQA, the system achieves high retrieval recall, citation validity, and evidence support, with no unsupported hallucination detected under the validation criteria. Regeneration is triggered for nearly one-third of the questions, indicating that runtime validation remains active under external evaluation.

The CUAD stress test shows a larger shift in evidence-support and refusal behavior. Retrieval recall and citation validity remain high, but evidence support decreases and false refusal increases. Regeneration is also less often accepted as the final output.

## 3.4 Failure Case Analysis

To examine residual errors, we analyze representative failures across the controlled evaluation and external validation. In the controlled setting, residual errors are rare and primarily involve false refusal or citation/evidence-compliance failures rather than unsupported generation. Under external validation, errors become more frequent and dataset-specific: PrivacyQA errors include gold-segment mismatch and false-answer behavior, while CUAD errors are dominated by over-refusal and clause-span extraction dificulty.

Observed residual errors are grouped into five categories: (i) false refusal, where the system abstains despite available evidence; (ii) citation or evidence-compliance failure, where an otherwise supported answer fails strict citation or validation requirements; (iii) segment mismatch, where a concise supported answer does not fully match the benchmark’s longer gold evidence segment; (iv) false answer, where the system answers despite insuficient evidence; and (v) extraction or span-selection failure, where relevant evidence is retrieved but not used completely or precisely.

Table 12 provides representative examples of these residual error types.

Table 12 Representative residual error cases across controlled and external settings
<table><tr><td>Setting</td><td>Type</td><td>Observed Behavior</td><td>Cause</td></tr><tr><td>Controlled</td><td>False refusal</td><td>Evidence is available but the system abstains</td><td>Conservative refusal behavior under validation</td></tr><tr><td>Controlled</td><td>Citation compliance</td><td>Supported answer fails strict citation or validation formatting</td><td>Strict citation-control requirements</td></tr><tr><td>PrivacyQA</td><td>Segment mismatch</td><td>Supported concise answer does not fully match gold segment</td><td>Gold labels are long pol- icy segments rather than canonical answers</td></tr><tr><td>PrivacyQA</td><td>False answer</td><td>System answers when evi- dence is insufficient for the query</td><td>Difficulty distinguishing partial evidence from suf- ficient evidence</td></tr><tr><td>CUAD</td><td>Over-refusal</td><td>Relevant clause is retrieved but system abstains</td><td>Clause-extraction task dif- fers from natural QA</td></tr><tr><td>CUAD</td><td>Extraction failure</td><td>Relevant clause is retrieved but answer is incomplete</td><td>Clause spans are dispersed across context</td></tr></table>

## 4 Discussion

## 4.1 Findings and Interpretation

The results show that trustworthy evidence-grounded RAG requires more than strong retrieval. Reliable answer generation depends on the interaction among retrieval, citation-constrained generation, post-generation validation, and refusal behavior. In the controlled housing-law evaluation, CiteGuard-RAG retrieves relevant evidence and constrains generation to cited, source-supported content under the implemented validation criteria.

The generator comparison further suggests that controlled-setting performance is not driven mainly by a specific language model. Mistral and Qwen produce efectively identical results under the same retrieval, prompting, and validation configuration, indicating that the retrieval and validation layers strongly constrain model behavior. This is important for intelligent information systems where reproducibility, deployment control, and privacy-sensitive local execution may be required.

External validation provides a more realistic view of system behavior under task and domain shift. PrivacyQA shows that citation validity and evidence support can remain strong when the framework is transferred to another document-grounded QA setting, although strict segment-overlap scoring reduces measured answer accuracy. CUAD presents a harder contract-domain stress test, where retrieval and citation behavior remain strong but false refusal and evidence-support errors increase. Overall, the framework generalizes most strongly in retrieval coverage, citation control, and unsupported-hallucination prevention, while answer-span selection and abstention calibration remain more sensitive to domain and task formulation.

## 4.2 Retrieval and Representation

The retrieval results show that hybrid semantic–lexical retrieval provides the most reliable evidence access. Lexical retrieval helps preserve exact legal and policy wording, while semantic retrieval improves matching when questions paraphrase the source text. The best performance is achieved when these signals are balanced rather than used independently.

The weighting and embedding ablations further show that retrieval design afects downstream grounding, not only evidence recall. Excessive semantic weighting increases hallucination risk, while stronger representations improve both retrieval and answer reliability. This indicates that evidence representation is a core reliability factor in RAG-based intelligent information systems.

## 4.3 Validation, Grounding, and Regeneration

The validation layer is the main mechanism for controlling unsupported generation. When validation is removed, retrieval accuracy remains unchanged, but grounded answer accuracy and citation validity decrease sharply. This confirms that retrieval alone is not suficient for trustworthy evidence-grounded QA.

The controlled 0.0% hallucination rate should be interpreted under the study’s operational definition: no unsupported claims were detected relative to retrieved and cited evidence. This does not mean that all possible legal or factual errors were eliminated. The remaining controlled-setting errors were mainly false refusals and citation or evidence-compliance failures rather than unsupported final answers.

The identical Mistral and Qwen results suggest that, in the controlled setting, system behavior is primarily shaped by retrieval, prompting, and validation constraints rather than by generator-specific variability. Regeneration further supports reliability by giving the system one opportunity to revise candidate answers that fail validation, although external results show that it cannot fully solve over-refusal or span-selection errors.

## 4.4 System Design Trade-ofs

The ablation results show that reliability depends on practical design choices. Moderate chunking provides the best balance between preserving enough context and avoiding irrelevant evidence. Smaller chunks can fragment legal rules, while larger chunks can add noise.

The reranking experiment also shows that added complexity does not always improve reliability. The heuristic reranker suppressed some relevant answer-bearing chunks, so the final system uses deduplicated hybrid retrieval without reranking. Overall, the results support optimizing the full retrieval–generation–validation pipeline rather than isolated components.

## 4.5 External Validation Insights

The external evaluations clarify the type of generalization achieved by CiteGuard-RAG. PrivacyQA suggests that the pipeline transfers reasonably well to another document-grounded legal/policy QA setting, particularly in citation control and unsupported-hallucination prevention. This supports the role of runtime validation beyond the controlled housing-law corpus.

CUAD reveals a diferent limitation. Unlike the controlled housing-law and PrivacyQA settings, CUAD is closer to contract clause extraction than natural question answering. It therefore introduces both domain shift and task-formulation shift. Although the system retrieves relevant evidence and maintains valid citations, it becomes more conservative and produces more false refusals. This suggests that clauselevel contract extraction requires more precise span-selection mechanisms than the current answer-generation pipeline provides. CUAD should therefore be interpreted as a stress test of robustness under task shift rather than as a direct comparison to the controlled QA setting.

Together, the two external evaluations show that retrieval coverage and citation validity generalize more robustly than answer extraction and refusal calibration. This distinction is important for evidence-grounded RAG systems: a system may retrieve the right evidence and cite valid sources, yet still fail to produce the exact answer expected by a benchmark. Reliable high-stakes question answering therefore requires both evidence access and evidence utilization.

## 4.6 Limitations and Future Work

Several limitations should be acknowledged. First, the controlled evaluation uses a curated housing-law corpus, and the external evaluations are limited to PrivacyQA and CUAD. Although these datasets provide meaningful evidence under domain and task shift, broader testing across statutes, regulations, case law, administrative guidance, and legal-aid documents would strengthen generalizability claims.

Second, the external benchmarks difer in answer format. PrivacyQA provides expert-labeled policy segments rather than canonical short-form answers, while CUAD focuses on clause-level contract evidence. As a result, answer accuracy and tokenoverlap scores should be interpreted alongside citation validity, evidence support, false-answer behavior, and unsupported hallucination.

Third, the validation framework relies on threshold-based lexical and semantic similarity checks. These checks help detect unsupported outputs, but they may miss implicit entailment, complex legal reasoning, or nuanced paraphrases. Future work should explore stronger validation methods based on natural language inference, learned evidence verification, and answer-span selection before generation.

Fourth, regeneration is limited to a single attempt for eficiency and reproducibility. Bounded multi-pass regeneration may improve recovery from dificult generation failures, but it would require careful stopping criteria to avoid excessive inference cost or unstable outputs.

Finally, the evaluated generators are general-purpose instruction-tuned models without legal-domain adaptation. Future work should compare domain-specialized and stronger general-purpose models under the same citation, grounding, refusal, and regeneration constraints.

## 5 Conclusion

This work presented CiteGuard-RAG, a validation-centered intelligent information system for evidence-grounded question answering combining hybrid semantic–lexical retrieval, citation-constrained generation, sentence-level grounding validation, abstention, and one-pass validation-guided regeneration. Results from the controlled housinglaw evaluation show that strong retrieval alone is not suficient for reliable answer generation; explicit post-generation validation is needed to enforce citation validity, evidence support, and hallucination control. External evaluation on PrivacyQA and CUAD further shows that retrieval coverage and citation validity transfer more robustly than answer extraction and refusal calibration under domain and task shifts. Overall, the findings suggest that trustworthy RAG systems should be designed as integrated retrieval–generation–validation pipelines, where candidate answers are checked against cited evidence before final delivery. Future studies should strengthen evidence utilization through improved span selection, entailment-based validation, and bounded multi-pass regeneration.

## References

[1] Ji Z, Lee N, Frieske R, Yu T, Su D, Xu Y, et al. Survey of Hallucination in Natural Language Generation. ACM Comput Surv. 2023 Mar;55(12). https: //doi.org/10.1145/3571730.

[2] Zhao Z, Fan W, Li J, Liu Y, Mei X, Wang Y, et al. Recommender Systems in the Era of Large Language Models (LLMs). IEEE Trans on Knowl and Data Eng. 2024 Nov;36(11):6889–6907. https://doi.org/10.1109/TKDE.2024.3392335.

[3] Fan W, Ding Y, Ning L, Wang S, Li H, Yin D, et al. A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models. In: Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. New York, NY, USA: Association for Computing Machinery; 2024. p. 6491–6501.

[4] Ram O, Levine Y, Dalmedigos I, Muhlgay D, Shashua A, Leyton-Brown K, et al. In-Context Retrieval-Augmented Language Models. Transactions of the Association for Computational Linguistics. 2023;11:1316–1331. https://doi.org/10.1162/ tacl a 00605.

[5] Mallen A, Asai A, Zhong V, Das R, Khashabi D, Hajishirzi H. When Not to Trust Language Models: Investigating Efectiveness of Parametric and Non-Parametric

Memories. In: Rogers A, Boyd-Graber J, Okazaki N, editors. Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Toronto, Canada: Association for Computational Linguistics; 2023. p. 9802–9822. Available from: https://aclanthology.org/2023.acl-long.546/.

[6] Niklaus J, Matoshi V, St¨urmer M, Chalkidis I, Ho D. MultiLegalPile: A 689GB Multilingual Legal Corpus. In: Ku LW, Martins A, Srikumar V, editors. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Bangkok, Thailand: Association for Computational Linguistics; 2024. p. 15077–15094. Available from: https://aclanthology.org/2024. acl-long.805/.

[7] Habernal I, Faber D, Recchia N, et al. Mining Legal Arguments in Court Decisions. Artificial Intelligence and Law. 2024;32:1–38. https://doi.org/10.1007/ s10506-023-09361-y.

[8] Hindi M, Mohammed L, Maaz O, Alwarafy A. Enhancing the Precision and Interpretability of Retrieval-Augmented Generation (RAG) in Legal Technology: A Survey. IEEE Access. 2025;13:46171–46189. https://doi.org/10.1109/ACCESS. 2025.3550145.

[9] Arivazhagan MG, Liu L, Qi P, Chen X, Wang WY, Huang Z. Hybrid Hierarchical Retrieval for Open-Domain Question Answering. In: Rogers A, Boyd-Graber J, Okazaki N, editors. Findings of the Association for Computational Linguistics: ACL 2023. Toronto, Canada: Association for Computational Linguistics; 2023. p. 10680–10689. Available from: https://aclanthology.org/2023.findings-acl.679/.

[10] Chirkova N, Rau D, D´ejean H, Formal T, Clinchant S, Nikoulina V. Retrievalaugmented generation in multilingual settings. In: Li S, Li M, Zhang MJ, Choi E, Geva M, Hase P, et al., editors. Proceedings of the 1st Workshop on Towards Knowledgeable Language Models (KnowLLM 2024). Bangkok, Thailand: Association for Computational Linguistics; 2024. p. 177–188. Available from: https://aclanthology.org/2024.knowllm-1.15/.

[11] Askari A, Verberne S, Abolghasemi A, Kraaij W, Pasi G. Retrieval for Extremely Long Queries and Documents with RPRS: A Highly Eficient and Efective Transformer-based Re-Ranker. ACM Trans Inf Syst. 2024 Apr;42(5). https: //doi.org/10.1145/3631938.

[12] Wang Z, Teo S, Ouyang J, Xu Y, Shi W. M-RAG: Reinforcing Large Language Model Performance through Retrieval-Augmented Generation with Multiple Partitions. In: Ku LW, Martins A, Srikumar V, editors. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Bangkok, Thailand: Association for Computational Linguistics; 2024. p. 1966–1978. Available from: https://aclanthology.org/2024.acl-long.108/.

[13] Liu NF, Lin K, Hewitt J, Paranjape A, Bevilacqua M, Petroni F, et al. Lost in the Middle: How Language Models Use Long Contexts. Transactions of the Association for Computational Linguistics. 2024;12:157–173. https://doi.org/10. 1162/tacl a 00638.

[14] Nguyen HT, Satoh K. ConsRAG: Minimize LLM Hallucinations in the Legal Domain. In: Legal Knowledge and Information Systems (JURIX 2024). vol. 385 of Frontiers in Artificial Intelligence and Applications. Amsterdam: IOS Press; 2024. .

[15] Roit P, Ferret J, Shani L, Aharoni R, et al. Factually Consistent Summarization via Reinforcement Learning with Textual Entailment Feedback. In: Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023). Toronto, Canada; 2023. p. 6252–6272.

[16] Salemi A, Zamani H. Evaluating Retrieval Quality in Retrieval-Augmented Generation. SIGIR ’24. New York, NY, USA: Association for Computing Machinery; 2024. p. 2395–2400. Available from: https://doi.org/10.1145/3626772.3657957.

[17] Zhang Y, Zhang M, Yuan H, Liu S, Shi Y, Gui T, et al. LLMEval: a preliminary study on how to evaluate large language models. In: Proceedings of the Thirty-Eighth AAAI Conference on Artificial Intelligence and Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence and Fourteenth Symposium on Educational Advances in Artificial Intelligence. AAAI’24/IAAI’24/EAAI’24. AAAI Press; 2024. Available from: https://doi.org/10.1609/aaai.v38i17.29934.

[18] Chen J, Lin H, Han X, Sun L. Benchmarking large language models in retrievalaugmented generation. In: Proceedings of the Thirty-Eighth AAAI Conference on Artificial Intelligence and Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence and Fourteenth Symposium on Educational Advances in Artificial Intelligence. AAAI’24/IAAI’24/EAAI’24. AAAI Press; 2024. Available from: https://doi.org/10.1609/aaai.v38i16.29728.

[19] Ravichander A, Black AW, Wilson S, Norton T, Sadeh N. Question Answering for Privacy Policies: Combining Computational and Legal Perspectives. In: Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). Hong Kong, China: Association for Computational Linguistics; 2019. p. 4949–4959. Available from: https://www.aclweb.org/ anthology/D19-1500.

[20] Hendrycks D, Burns C, Chen A, Ball S. CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review. NeurIPS. 2021;.

[21] Xiao S, Liu Z, Zhang P, Muennighof N, Lian D, Nie JY. C-Pack: Packed Resources For General Chinese Embeddings. arXiv. 2023;https://doi.org/10. 48550/arxiv.2309.07597.

[22] Robertson S, Zaragoza H. The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval. 2009;3(4):333–389. https://doi.org/10.1561/1500000019.

[23] Jiang AQ, Sablayrolles A, Mensch A, Bamford C, Chaplot DS, de Las Casas D, et al.: Mistral 7B. arXiv. Available from: https://arxiv.org/abs/2310.06825.

[24] Bai J, Bai S, Chu Y, Cui Z, Dang K, Deng X, et al. Qwen Technical Report. arXiv. 2023;https://doi.org/10.48550/arxiv.2309.16609.

[25] Wang W, Wei F, Dong L, Bao H, Yang N, Zhou M. MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. arXiv. 2020;https://doi.org/10.48550/arxiv.2002.10957.

[26] Wang L, Yang N, Huang X, Jiao B, Yang L, Jiang D, et al. Text Embeddings by Weakly-Supervised Contrastive Pre-training. arXiv. 2022;https://doi.org/10. 48550/arxiv.2212.03533.