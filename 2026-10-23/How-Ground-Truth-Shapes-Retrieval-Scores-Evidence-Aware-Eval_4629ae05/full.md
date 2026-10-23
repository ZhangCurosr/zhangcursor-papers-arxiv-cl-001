# How Ground Truth Shapes Retrieval Scores: Evidence-Aware Evaluation of a Retrieval-Augmented Generation System for Breast Lesions

Miguel Mochizuki Silva<sup>1</sup> Samuel Ismael<sup>1</sup> Mateus C. Dias<sup>1</sup>

Thais Gaudêncio<sup>1</sup> Yuri Malheiros<sup>1</sup> Bruno Barufaldi<sup>2,∗</sup>

<sup>1</sup>Universidade Federal da Paraíba (UFPB), João Pessoa, PB, Brazil

<sup>2</sup>Department of Radiology, Perelman School of Medicine, University of Pennsylvania, Philadelphia, PA, USA

<sup>∗</sup>Corresponding author: bruno.barufaldi@pennmedicine.upenn.edu

October 23, 2026

## Abstract

Purpose: Retrieval scores for clinical retrieval-augmented generation (RAG) are usually reported against annotated chunk per queries, which mixes retriever quality with annotation coverage. To describe the corpus, RAG (RAG) pipeline, and evidence-aware retrieval evaluation protocol developed for a question-answering system over the breast lesion management literature and to report retrieval performance under the selected configuration.

Methods: An 18-document corpus focused on breast lesions was used for retrieval benchmarking, ground-truth construction, and generation evaluation. One simple and direct evaluation query was created for each document, resulting in an 18-query development benchmark composed of Yes/No and WH-questions. Retrieval combines dense embeddings, BM25, and a cross-encoder reranker. Chunk size (128–1024 tokens), chunk overlap (0–20%) varied one factor at time, and four retriever configurations were compared in an ablation. Ground-truth relevance is defined at the level of evidence spans, which are mapped to every chunk containing them, so that chunks expressing the same underlying fact under overlapping chunking are scored as equivalent rather than as competing candidates. Retrieval quality is reported with R-Precision, Recall@K, MRR, and nDCG@K on an 18-query ground-truth development set. Generation quality was assessed in a separate experiment on an 512-token index, using DeepEval claimlevel metrics with deepseek-v4-pro as the LLM judge, together with classical BLEU/ROUGE metrics across context depths K = 1–5.

Results: The selected hybrid retriever (384-token chunks, 20% overlap) retrieved accepted evidence in the top three results for 17 of 18 queries. Recall@3, R-Precision, MRR@3, and nDCG@3 each reached 0.94. Ground-truth curation increased the number of evidence spans from 25 to 37 and accepted chunk targets from 36 to 51 without changing retrieval rankings.

Conclusion: Cross-encoder reranking produced the highest retrieval scores. Ground-truth curation expanded the set of recognized evidence without changing retrieval rankings, underscoring the dependence of evaluation scores on relevance annotations. The small development set and retrieval-informed curation warrant independent validation, including radiologist assessment of evidence suficiency.

## 1 Introduction

Management of benign and high-risk breast lesions requires integrating imaging findings, biopsy pathology, radiologic–pathologic concordance, and patient risk. Supporting evidence is distributed across society guidelines, appropriateness criteria, and lesion-specific studies, which may difer in their recommendations. For lesions of uncertain malignant potential (B3 lesions), international consensus supports vacuum-assisted excision as an alternative to open surgery for selected categories [20], whereas U.S. guidance permits imaging surveillance for selected radiologic–pathologic concordant lesions [38]. Retrieving the correct recommendation therefore requires preserving the clinical conditions under which it applies.

Large language models (LLMs) encode substantial clinical knowledge [23] but may generate unsupported statements [10]. Retrieval-augmented generation (RAG) conditions answers on retrieved passages and makes supporting sources available for inspection [14]. In medical question answering, retrieval has improved performance in some settings, although outcomes depend on the corpus and retrieval strategy [29]. Missing evidence can limit answer correctness, while supplying additional passages does not necessarily resolve the problem because language models may fail to use relevant information in long contexts [16].

Medical RAG systems have been evaluated using guidelines and clinical resources, including clinical question answering and simulated oncology cases [6, 32]. Radiology-focused studies have also reported improvements that vary across models and sources [25, 26]. However, evaluation protocols remain heterogeneous [1], and expert assessment has identified retrieval and evidence selection as important failure points [12]. These findings motivate evaluating retrieved evidence separately from generated answers.

Retrieval scores depend on which passages are annotated as relevant. When a benchmark recognizes only one chunk per question, another chunk containing the same supporting evidence may be scored as nonrelevant. Overlapping segmentation makes such alternatives more likely, while additional relevant passages may remain unannotated. In information retrieval, incomplete relevance judgments have long been recognized as a source of evaluation bias [2, 3, 33]. The efects on individual metrics depend on their definitions and the accepted evidence targets; overlap alone does not establish a fixed penalty.

Question-answering benchmarks have represented supporting evidence through answer spans, sentence-level facts, and source provenance [13, 19, 21, 30]. In RAG evaluation, conventional ranking metrics may not fully capture whether retrieved passages supply the information needed to answer a question [31]. Chunk-based evaluation is also sensitive to how evidence is partitioned and repeated across overlapping segments [24]. Evidence-level relevance annotations can therefore reduce dependence on a single chunk identifier, provided that accepted alternatives are explicitly defined and audited.

We present RAG-MAMA, a retrieval-augmented generation system evaluated on a curated 18- document breast-lesion corpus and an 18-query development benchmark. We investigate whether incomplete ground-truth annotations can underestimate retrieval performance. Canonical evidence spans are defined independently of chunk identifiers and mapped to each segmentation configuration. We distinguish alternative supporting chunks from complementary evidence requirements, audit previously unrecognized evidence while keeping retrieval rankings fixed, and separately evaluate retrieval configurations and generation quality. The system focuses on excision-versus-surveillance decisions, with fixed deferral for excluded categories. This study establishes a small-scale method ological baseline rather than demonstrating clinical efectiveness or validity on more complex, multi evidence questions.

## 2 Methods

Figure 1 shows the RAG-MAMA pipeline. The corpus comprised 18 papers, which were preprocessed and split into 757 chunks (size 384, overlap 77). Each query was retrieved in parallel by a dense retriever (BGE-Large) and a lexical retriever (BM25). The two ranked lists were combined with Reciprocal Rank Fusion and reranked with a BGE cross-encoder. The top-K chunks were passed to Qwen 3.6 (27B) to generate a grounded answer. Retrieval was evaluated with ranking metrics and generation with DeepEval.

![](images/6bbf6d6bac4daa4761ecc3dc48cd1345549aea8c85c7daaab16d8618c3ca1444.jpg)  
Figure 1: Overview of the RAG-MAMA architecture, illustrating corpus preparation, hybrid retrieval, response generation, and evaluation.

## 2.1 Corpus Compilation, Pre-processing, and Quality Assurance

The corpus was scoped a priori to a fixed reference table of 15 breast pathology categories defined by the American Society of Breast Surgeons (ASBrS) high-risk lesion taxonomy [38].

Thirty candidate documents were assembled by purposive expert selection, comprising society guidelines, American College of Radiology (ACR) appropriateness criteria, World Health Organization (WHO) classification, and lesion-specific primary literature. Documents were eligible if they provided management recommendations, upgrade-risk estimates, radiologic–pathologic concordance criteria, or nomenclature definitions for at least one retrieval-eligible category. Each was evaluated by category-specific term frequency and adjudicated by full-text review. Malignant categories and fibromatosis/desmoid tumor are not answered by retrieval: a scope gate applied to the normalized pathology label returns a fixed deferral before retrieval runs. Fibromatosis is excluded because current sarcoma guidance favors active surveillance over the excision and desmoid text remains in the indexed corpus.

Table 1: Classification and typical management of benign, high-risk, and malignant breast pathology entities. Malignant lesions are routed to a fixed response by the scope gate.
<table><tr><td>Pathology entities</td><td>Classification ASBrS</td><td>category</td><td>Typical management</td></tr><tr><td>Atypical ductal hyperplasia (ADH)</td><td>High-risk lesion ADH</td><td></td><td>Surgical excision</td></tr><tr><td>Flat epithelial atypia (FEA)</td><td>High-risk lesion Flat epithelial</td><td>atypia</td><td>Selective excision or observation</td></tr><tr><td>Lobular carcinoma in situ (LCIS); atypical lobular hyperplasia (ALH)</td><td>High-risk lesion Lobular</td><td>neoplasia</td><td>Observation or selective excision</td></tr><tr><td>Pleomorphic LCIS; florid LCIS</td><td>High-risk lesion Non-classic</td><td>LCIS</td><td>Surgical excision</td></tr><tr><td>Papilloma without atypia; intraductal papilloma; peripheral duct papilloma</td><td>Benign lesion</td><td>Papillary lesions</td><td>Observation or selective excision</td></tr><tr><td>Papilloma with atypia</td><td>High-risk lesion Papillary</td><td>lesions</td><td>Surgical excision</td></tr><tr><td>Radial scar; complex sclerosing lesion</td><td>High-risk lesion Radial scar /</td><td>complex sclerosing lesion</td><td>Observation or excision</td></tr><tr><td>Mucocele-like lesion</td><td>High-risk lesion Mucocele-like</td><td>lesion</td><td>Observation or excision</td></tr><tr><td>Pseudoangiomatous stromal hyperplasia (PASH)</td><td>Benign lesion</td><td>PASH</td><td>Observation in most patients</td></tr><tr><td>Fibromatosis; desmoid tumor</td><td>Benign lesion</td><td>Fibromatosis</td><td>Deferred: sarcoma multidisciplinary team</td></tr><tr><td>Sclerosing adenosis; adenosis; apocrine metaplasia; usual ductal hyperplasia; fibroadenomatoid hyperplasia</td><td>Benign proliferative lesion</td><td></td><td>Routine clinical management</td></tr><tr><td>Fibrocystic change; fibroadenoma; benign cyst; fat necrosis; duct ectasia; hamartoma; and related benign diagnoses</td><td>Benign lesion</td><td></td><td>Routine clinical management</td></tr><tr><td>Ductal carcinoma in situ (DCIS), all grades and histologic subtypes; intraductal (adeno)carcinoma, non-infiltrating</td><td>Malignant (Stage 0)</td><td></td><td>Breast cancer treatment guidelines</td></tr><tr><td>Invasive (infiltrating) ductal carcinoma; invasive (infiltrating) lobular carcinoma; mixed ductal and lobular carcinoma; invasive mammary carcinoma; tubular carcinoma; metaplastic carcinoma</td><td>Malignant</td><td></td><td>Breast cancer treatment guidelines</td></tr><tr><td>Metastatic carcinoma to axillary or intramammary lymph node; lymphoma involving the breast; metastatic melanoma to the breast</td><td>Malignant</td><td></td><td>Disease-specific oncologic management</td></tr></table>

Eighteen documents were retained and indexed, stratified into four tiers used as retrieval priors (Table 2): core management guidelines (n = 4), lesion-specific and concordance sources (n =

9), imaging appropriateness criteria (n = 3), and benign breast disease risk literature (n = 2). Ten documents were excluded: treatment guidelines for established malignancy (n = 3), since the reference table defers these categories without requiring their text and their volume (NCCN with over 820k chars) would dominate retrieval; disease entities outside the table $( n = 3 )$ ; documents without lesion-specific management content $( n = 2 )$ ; a superseded WHO edition $( n = 1 )$ ; and the EUSOMA B3 guideline $( n = 1 )$ , which specifies vacuum-assisted excision as the default therapeutic step whereas the retained guidelines frame management as excision versus surveillance. The retained corpus is therefore internally consistent on the excision-versus-surveillance axis and represents North American practice.

Table 2: Guidelines and scientific literature supporting the management of breast lesions.
<table><tr><td>Authors</td><td>Source</td><td></td><td>Year Document type</td></tr><tr><td>ASBrS</td><td>Resource Guide: Surgical Management of Benign or High-Risk 2024 Archived society Lesions</td><td></td><td>resource guide</td></tr><tr><td>Nakhlis et al.</td><td>American Society of Breast Surgeons, Society of Breast Imaging, and College of American Pathology 2026 Guidelines for the Management of Proliferative Lesions With Atypia and</td><td>2026</td><td>Society guideline / Special Communication</td></tr><tr><td>Zamora et</td><td>Lobular Carcinoma In Situ. JAMA Surg Management of High-Risk Breast Lesions: A Comprehensive Update. J Breast Imaging</td><td></td><td>2026Scientific review</td></tr><tr><td>al.</td><td>Eisen et al. High-Risk Benign Breast Lesions: An Ontario Health (Cancer 2026 Recommendations Care Ontario) Recommendations Report. Curr Oncol</td><td></td><td></td></tr><tr><td>et al.</td><td>Rosenberger American Society of Breast Surgeons and Society of Breast Imaging 2025 Guidelines for the Management of Benign</td><td>2025</td><td>report / review Society guideline / Special</td></tr><tr><td>al.</td><td>Breast Fibroepithelial Lesions. JAMA Surg Mitchell et American Society of Breast Surgeons, Society of Breast Imaging, and College of American Pathology 2025 Guidelines for the Management of Infectious and Inflammatory Lesions of</td><td></td><td>2026 Society guideline / Special</td></tr><tr><td>Guirguis et al.</td><td>Mimickers of Breast Malignancy: Imaging Findings, Pathologic Concordance and Clinical Management. Insights Imaging</td><td>2021</td><td>Educational review</td></tr><tr><td>Estes et al.</td><td>Pseudoangiomatous Stromal Hyperplasia: Overview and Clinical Management. Ann Breast Surg</td><td></td><td>2020 Review article</td></tr><tr><td>Yoon et al.</td><td>Optimal Treatment of Pseudoangiomatous Stromal Hyperplasia of the Breast. Asian J Surg</td><td></td><td>2020 Original cohort study</td></tr><tr><td></td><td>Quinn et al. World Health Organization Classification of Tumours of the Breast 6th Edition 2026. Histopathology</td><td>2026</td><td>WHO classification</td></tr><tr><td>ASBrS</td><td>Concordance Assessment Following Image-Guided Breast Biopsy</td><td></td><td>review 2010 Quality measure</td></tr><tr><td>ASBrS</td><td>Image-Guided Percutaneous Biopsy of Palpable and Nonpalpable Breast Lesions</td><td></td><td>2018 Society resource guide</td></tr><tr><td>Authors</td><td>Source</td><td>Year</td><td>Document type</td></tr><tr><td>ASBrS</td><td>Position Statement on Stereotactic Breast Procedures</td><td>2010 (rev.)</td><td>Position/ practice guideline</td></tr><tr><td>Sanford et al.</td><td>ACR Appropriateness Criteria: Evaluation of Nipple Discharge: 2022 Update. J Am Coll Radiol</td><td>2022</td><td>Appropriateness criteria</td></tr><tr><td>Klein et al.</td><td>ACR Appropriateness Criteria: Palpable Breast Masses: 2022 Update. J Am Coll Radiol</td><td>2023</td><td>Appropriateness criteria</td></tr><tr><td>Niell et al.</td><td>ACR Appropriateness Criteria: Evaluation of the Symptomatic Male Breast. J Am Coll Radiol</td><td>2018</td><td>Appropriateness criteria</td></tr><tr><td>al.</td><td>Hartmann et Benign Breast Disease and the Risk of Breast Cancer. N Engl JMed</td><td>2005</td><td>Cohort study</td></tr><tr><td>Dyrstad et al.</td><td>Breast Cancer Risk Associated With Benign Breast Disease: Systematic Review and Meta-analysis. Breast Cancer Res Treat</td><td>2015</td><td>Systematic review and meta-analysis</td></tr></table>

An 18-query retrieval benchmark was constructed, with one question per source document. Questions, reference answers, and supporting evidence were drafted with Codex (recorded session model: gpt-5.6-terra) and reviewed by the authors. Each Yes/No or WH query targeted a focused, source-supported answer.

Under the selected fixed-size chunking configuration of 384 tokens with a 77-token overlap, the 18-document corpus yielded a total of 757 chunks. Text is extracted with PyMuPDF (1.23.0+). Unicode control characters (U+0000–U+009F) are replaced with spaces, soft hyphens (U+00AD) and zero-width spaces (U+200B) are removed, and whitespace is collapsed. Repeated header/footer lines are removed when they recur three or more times. Reference sections are identified by heading pattern and removed when citation density exceeds 25% of lines.

A seven-feature noise classifier assigned chunks to five categories. All 757 chunks were stored in ChromaDB; 558 were eligible for retrieval after filtering.

## 2.2 Retrieval-Augmented Generation Pipeline

The system implements a modular RAG architecture. Dense retrieval uses BAAI/bge-large-en-v1.5 embeddings over a ChromaDB vector store, while lexical retrieval uses BM25 [22]. The semantic and lexical rankings are combined using reciprocal-rank fusion (RRF) [4], with an RRF parameter of 50. The selected environment configures a hybrid fetch multiplier of 2 and a semantic retrieval cap of 30 results. Up to 20 fused candidates are subsequently re-scored using the BAAI/bge-reranker-v2-m3 cross-encoder [17, 28]. The final retrieval depth used in the initial benchmark is K = 3. Table 3 shows the full configuration.

## 2.2.1 Chunking

Token counts used during chunk construction were computed using the tokenizer associated with the dense embedding model, BAAI/bge-large-en-v1.5, loaded through the Hugging Face AutoTokenizer interface.

Table 3: Selected retrieval and generation configuration.
<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Chunking</td><td>Fixed-size, 384 tokens; 77-token (20%) overlap; 757 chunks</td></tr><tr><td>Tokenizer for chunking</td><td>bge-large-en-v1.5 (Hugging Face AutoTokenizer)</td></tr><tr><td>Dense retrieval</td><td>BAAI/bge-large-en-v1.5; ChromaDB; semantic cap 30</td></tr><tr><td>Lexical retrieval</td><td>BM25 (rank-bm25); biomedical-aware tokenizer</td></tr><tr><td>Fusion</td><td>RRF, k = 50; hybrid fetch multiplier 2</td></tr><tr><td>Reranker</td><td>BAAI/bge-reranker-v2-m3; candidate pool 20</td></tr><tr><td>Final retrieval depth K = 3</td><td></td></tr><tr><td>Generator</td><td>qwen/qwen3.6-27b; temperature 0.3; max output 2048 tokens; prompt rag-evaluation-v1</td></tr><tr><td></td><td>Generation context Top-K chunks in rank order; each capped at 3,500 characters; metadata included</td></tr><tr><td>Judge</td><td>deepseek-v4-pro; DeepEval 4.0.7; threshold 0.5</td></tr></table>

The fixed-size chunking procedure operates on extracted text units rather than splitting the normalized document at arbitrary character positions. Text units are packed until adding another unit would exceed the configured token limit. Overlap between consecutive chunks is constructed from trailing text units whenever possible. Chunk-level provenance, including page ranges, contributing text blocks, and section metadata, is retained during this process.

Chunk size and overlap were varied one factor at a time. Chunk size was varied from 128 to 1024 tokens with approximately 20% overlap. After selecting a chunk size of 384 tokens, overlap was varied from 0% to 20%, corresponding to 0, 38, 58, and 77 tokens. The selected configuration uses 384-token chunks with a 77-token overlap.

## 2.2.2 Embedding, reranking, and lexical retrieval

Dense retrieval represents both documents and queries using BAAI/bge-large-en-v1.5. Lexical retrieval uses BM25 with a separate biomedical-aware tokenizer.

The BM25 tokenizer lowercases text, normalizes hyphen variants, removes common punctuation, and preserves clinically relevant lexical forms such as HER2+, ER-, BI-RADS, percentages, decima values, and comparison operators.

The semantic and lexical rankings are combined using RRF with parameter 50. The hybrid retrieval stage uses a fetch multiplier of 2, with the semantic retrieval stage configured to return up to 30 results. Up to 20 fused candidates are then re-scored using the BAAI/bge-reranker-v2-m3 cross-encoder, and the top three chunks are returned as the final retrieval output.

The retrieval ablation compares four configurations: BM25, dense semantic retrieval, hybrid RRF, and hybrid RRF followed by cross-encoder reranking. All variants use the same corpus, 18- query development set, selected 384-token/77-token segmentation, and relevance annotations for the ablation; only the retrieval component under evaluation is changed.

## 2.2.3 Answer generation

The answer-generation component was performed using qwen/qwen3.6-27b through an OpenAIcompatible chat-completions interface (Table 3). Generation context is constructed from the highest-ranked retrieved chunks in their final retrieval order. For each selected chunk, the implementation keeps at most the 3,500 characters of chunk text and then adds identifying metadata, including chunk ID, document ID, source file, DOI, and rank. Truncation is therefore characterbased rather than token-based and may occur inside a sentence, table, or section.

The implementation does not enforce a separate global token budget or a project-level model context-window check for answer generation. Chunks below the configured generation-context depth are omitted, while each selected chunk is independently capped at 3,500 characters. If the generation request fails, the run records a generation failure rather than rebuilding a smaller context. No generation failures occurred in the preserved K1–K5 full-RAG experiments.

## 2.3 Retrieval Evaluation

The development benchmark contains 18 queries, one per source document, with simple and direct Yes/No or WH formulations. Each query is associated with canonical evidence containing the information required to support its reference answer.

Retrieval was evaluated using Recall@3, R-Precision, MRR@3, and nDCG@3 [8]. Results were macro-averaged across 18 queries.

Canonical evidence is maintained independently from chunk identifiers. Because diferent chunksize and overlap configurations produce diferent chunk boundaries and identifiers, an evidence-tochunk binding is generated for each evaluated collection. Thus, the chunk-size and overlap sweeps use the same 18-query development ground truth while mapping its canonical evidence to the chunk representation produced by each segmentation configuration.

Automatic evidence-to-chunk binding matched the complete canonical evidence span to chunks from the same document after Unicode NFC normalization and whitespace collapsing, preserving case and punctuation. When no individual chunk contained the complete span, the mapper examined consecutive chunks, removed exact sufix–prefix token overlap, and identified the shortest sequence reconstructing the span. Partial containment alone did not qualify a chunk as a standalone evidence target. Additional accepted chunks could be included through manual curation; their clinical suficiency and preservation of required qualifiers were annotation judgments rather than guarantees of string matching.

Evidence routes distinguish alternative chunks (OR) from complementary chunks required jointly (AND). In the reported 18-query benchmark, however, each query had one required evidence group, and all accepted targets were individual chunks. No complementary multi-chunk routes were annotated. Each alternative was evaluated separately, retaining the maximum score per query and metric before macro-averaging. Consequently, Recall@3 indicates whether any accepted alternative appears within the first three ranks; R-Precision indicates an accepted rank-one result; MRR@3 uses the reciprocal of the best accepted rank up to three; and nDCG@3 uses discounted binary gain. Equivalent chunks were not counted as multiple relevant items within a single target, but the scores can still depend on the accepted alternatives and their positions. Invariance to overlapping chunks is therefore not established.

Following selection of 384-token chunks with 77-token overlap, high-ranking chunks absent from the initial ground truth were reviewed, including those ranked above the original gold evidence. Additional chunks judged to support the reference answers were added to the curated relevance set.

The final retrieval ablation used these annotations. Rankings remained fixed during ground-truth enrichment; score changes therefore reflect revised relevance labels, not retriever changes.

An exploratory analysis of query length, wording, question form, and perplexity is reported in Supplementary Section A.1.

## 2.4 Generation Evaluation

Generation evaluation separates retrieval from answer generation. The preserved full-RAG experiment uses the same 18 benchmark queries and keeps a fixed Top-5 retrieval result for each query. Generation-context depth is then varied from K = 1 to K = 5: the first one, two, three, four, or five retrieved chunks, respectively, are supplied to the generator. Thus, K in this experiment denotes generation-context depth rather than a change in the underlying retrieval depth.

These archived generation runs are associated with an earlier index of the same 18-document corpus, using fixed 512-token chunks with a 102-token overlap and yielding 560 chunks. This difers from the subsequently selected 384-token/77-token configuration used for the retrieval segmentation and ablation experiments. All five generation-context conditions preserve the same ordered Top-5 retrieval results and vary only the number of those chunks passed to answer generation.

For each selected chunk, at most 3500 characters of chunk text are supplied, so the maximum chunk-text contribution is 3500, 7000, 10500, 14000, and 17500 characters for K1 through K5, respectively, before metadata and prompt overhead. These values are derived per-condition bounds rather than enforced global context limits. No tokenizer-based generation-context budget is applied. All K1–K5 runs use the recorded model ID qwen/qwen3.6-27b, temperature 0.3, maximum output length of 2048 tokens, and prompt version rag-evaluation-v1.

Traditional answer evaluation metrics compare each generated answer with its reference\_answer. BLEU is computed with SacreBLEU 2.6.0 using the 13a tokenizer, lowercasing, Unicode NFC normalization, exponential smoothing, and efective order; reported BLEU-1 through BLEU-4 values are cumulative sentence BLEU-N scores averaged across queries. ROUGE is computed with rouge-score 0.1.2 using its standard tokenizer, stemming enabled, and F-measure for ROUGE-1, ROUGE-2, and ROUGE-L [15, 18].

Model-based evaluation uses DeepEval 4.0.7 with the recorded judge model deepseek-v4-pro through an OpenAI-compatible DeepSeek endpoint, synchronous execution, and a threshold of 0.5. The evaluated metrics are Contextual Precision, Contextual Recall, Contextual Relevancy, Faithfulness, and Answer Relevancy.

## 3 Results

With 384-token chunks and 20% overlap, hybrid retrieval followed by cross-encoder reranking retrieved accepted evidence among the top three results for 17 of the 18 development queries $( \mathrm { R e c a l l @ 3 } = 0 . 9 4 )$ . Ground-truth curation increased the number of accepted evidence spans from 25 to 37 without changing retrieval rankings. Generation performance varied across context depths.

## 3.1 Segmentation

Figure 2(A) compares chunk sizes from 128 to 1024 tokens at approximately 20% overlap. The 384-token configuration achieved the highest overall retrieval performance (Recall@3 = 0.94, R-Precision = 0.77, $\mathrm { M R R @ 3 } = 0 . 8 9$ , and $\mathrm { n D C G @ 3 } = 0 . 9 0 )$

At a fixed chunk size of 384 tokens, Recall@3 remained at 0.94 for 0%, 10%, and 20% overlap (Figure 2(B)). The 20% setting (77 tokens) achieved the highest R-Precision (0.77), MRR@3 (0.89), and nDCG@3 (0.90) and was selected for subsequent retrieval evaluations.

![](images/b71b6ce6f26aa33d44b1d3fa5e6c834504a101714cb346d87076a555f7a85cd3.jpg)  
(A) Chunk-size sweep

![](images/06ba318415b463f72e40ff27dcc5bb10f981b435129e90f1c8abe3c6ef4324e3.jpg)  
(B) Overlap sweep  
Figure 2: Retrieval performance under diferent segmentation configurations on the 18-query development benchmark. (A) Chunk-size sweep with approximately 20% overlap. (B) Overlap sweep at a fixed chunk size of 384 tokens.

## 3.2 Retrieval Ablation

Table 4 compares four retrieval configurations against the curated ground truth. BM25 and semantic retrieval achieved Recall@3 values of 0.89 and 0.83, respectively. Hybrid RRF maintained Recall@3 at 0.89 while increasing R-Precision to 0.78. Adding the cross-encoder reranker increased Recall@3, R-Precision, MRR@3, and nDCG@3 to 0.94.

Table 4: Retrieval ablation on the 18-query development set after ground-truth curation. → 1 indicates that values closer to 1 are better; bold values indicate the best observed scores.
<table><tr><td>Retriever</td><td>Recall@3 → 1</td><td>R-Precision → 1</td><td>MRR@3 → 1</td><td>nDCG@3 → 1</td></tr><tr><td>BM25</td><td>0.89</td><td>0.61</td><td>0.72</td><td>0.76</td></tr><tr><td>Semantic</td><td>0.83</td><td>0.56</td><td>0.69</td><td>0.73</td></tr><tr><td>Hybrid RRF</td><td>0.89</td><td>0.78</td><td>0.81</td><td>0.83</td></tr><tr><td>Hybrid RRF + Reranker</td><td>0.94</td><td>0.94</td><td>0.94</td><td>0.94</td></tr></table>

## 3.3 Efect of Ground-Truth Curation

Ground-truth curation increased the number of evidence spans from 25 to 37 and accepted chunk targets from 36 to 51 without changing retrieval rankings. Against the curated annotations, Hybrid RRF + Reranker retrieved an accepted evidence target within the top three for 17 of 18 queries (Recall@3 = 0.94). R-Precision, MRR@3, and nDCG@3 also reached 0.94.

## 3.4 Generation Quality by Context Depth

Table 5(A) presents the BLEU and ROUGE scores across generation-context depths from $K = 1$ to $K = 5$ . The highest BLEU-1 through BLEU-4, ROUGE-1, and ROUGE-2 scores occurred at $K = 5$ , including BLEU-4 = 0.22, ROUGE-1 = 0.56, and $\mathrm { R O U G E 2 } = 0 . 3 7$ . ROUGE-L was highest at K = 1 (0.51).

Table 5(B) presents the DeepEval metrics. Contextual Relevancy decreased from 0.72 at K = 1 to 0.47 at $K = 5$ , while Contextual Recall reached 0.94 at K = 2, K = 4, and $K = 5$ . Contextual Precision was highest at $K = 5 ~ ( 0 . 9 1 )$ . Faithfulness remained between 0.96 and 1.00, and Answer Relevancy reached 0.94 at K = 2, K = 4, and K = 5. No single context depth consistently maximized all metrics.

Table 5: Generation evaluation across context depths K1–K5. (A) Classical BLEU/ROUGE metrics. (B) DeepEval metrics. → 1 indicates that values closer to 1 are better; bold values indicate the best observed scores before rounding.  
(A) Classical generation metrics
<table><tr><td>Metric</td><td>K1</td><td>K2</td><td>K3</td><td>K4</td><td>K5</td></tr><tr><td>BLEU-1 → 1</td><td>0.44</td><td>0.41</td><td>0.43</td><td>0.38</td><td>0.44</td></tr><tr><td>BLEU-2 → 1</td><td>0.33</td><td>0.31</td><td>0.32</td><td>0.29</td><td>0.34</td></tr><tr><td>BLEU-3 → 1</td><td>0.26</td><td>0.24</td><td>0.25</td><td>0.23</td><td>0.27</td></tr><tr><td>BLEU-4 → 1</td><td>0.22</td><td>0.19</td><td>0.20</td><td>0.18</td><td>0.22</td></tr><tr><td>ROUGE-1 → 1</td><td>0.55</td><td>0.54</td><td>0.54</td><td>0.50</td><td>0.56</td></tr><tr><td>ROUGE-2 → 1</td><td>0.36</td><td>0.35</td><td>0.34</td><td>0.30</td><td>0.37</td></tr><tr><td>ROUGE-L → 1</td><td>0.51</td><td>0.47</td><td>0.47</td><td>0.42</td><td>0.50</td></tr></table>

(B) DeepEval metrics
<table><tr><td>Metric</td><td>K1</td><td>K2</td><td>K3</td><td>K4</td><td>K5</td></tr><tr><td>Contextual Precision → 1</td><td>0.89</td><td>0.89</td><td>0.88</td><td>0.87</td><td>0.91</td></tr><tr><td>Contextual Recall → 1</td><td>0.83</td><td>0.94</td><td>0.89</td><td>0.94</td><td>0.94</td></tr><tr><td>Contextual Relevancy → 1</td><td>0.72</td><td>0.58</td><td>0.55</td><td>0.51</td><td>0.47</td></tr><tr><td>Faithfulness → 1</td><td>0.96</td><td>1.00</td><td>0.97</td><td>1.00</td><td>0.97</td></tr><tr><td>Answer Relevancy → 1</td><td>0.87</td><td>0.94</td><td>0.93</td><td>0.94</td><td>0.94</td></tr></table>

## 3.5 Qualitative Retrieval Analysis

The following cases compare the clinical information need with the retrieved evidence. Excerpts are taken from the ranked passages, rather than from the reference answers.

## Q01 — Flat epithelial atypia (successful; rank 1).

Query:

“Is diagnostic excision generally indicated for flat epithelial atypia alone when radiologicpathologic concordance is confirmed?”

Retrieved:

“Provided radiologic-pathologic concordance is confirmed, diagnostic excision is generally not indicated after a diagnosis of FEA” (Nakhlis et al., 2026).

The passage directly states the management recommendation and preserves the concordance requirement.

## Q02 — Radial scar/complex sclerosing lesion (successful; rank 1).

Query:

“Can surgical excision generally be avoided for an adequately sampled, concordant radial scar or complex sclerosing lesion without atypia?”

## Retrieved:

“Routine excision is not required if pathology is benign concordant with radiology assessment” (Eisen et al., 2026; section on pure radial scars/complex sclerosing lesions without atypia).

The retrieved passage supports an afirmative answer to the Yes/No question, indicating that routine surgical excision is not required for radiologic–pathologic concordant lesions without atypia. However, sampling adequacy is not explicitly addressed in the quoted excerpt. Thus, although the passage was accepted as a successful retrieval in the benchmark, the excerpt alone does not establish explicit coverage of every clinical qualifier in the question.

## Q03 — Atypical lobular hyperplasia (successful; rank 1).

Query:

“Is excision generally required for atypical lobular hyperplasia when imaging is concordant and no other lesion is present?”

## Retrieved:

“Generally, ALH does not require excision if breast imaging has ruled out any other lesions and is reported as concordant” (Eisen et al., 2026).

The passage directly addresses the management decision and both clinical qualifiers.

## Q09 — Pseudoangiomatous stromal hyperplasia (Top-3 failure).

Query:

“Which biopsy technique is preferred for diagnosing pseudoangiomatous stromal hyperplasia?” Retrieved (rank 1):

“Core needle biopsy is indicated for diagnosis and must be concordant with imaging and clinical findings” (Estes et al., 2020).

## Accepted evidence (rank 7):

“Core needle biopsy is preferred over fine needle aspiration (FNA) or excisional biopsy.”

Although relevant to PASH diagnosis, the rank-1 passage does not compare biopsy techniques. The required comparative evidence was retrieved at rank 7, outside the Top-3 cutof.

## 4 Discussion

Retrieval performance depended on both the selected configuration and the coverage of the annotated ground truth. Increasing generation-context depth also produced diferent efects across evaluation metrics, without establishing a consistently superior setting. These findings characterize a small development benchmark and do not constitute independent clinical validation.

## 4.1 Technical Development

The selected configuration (384-token chunks, 20% overlap, Hybrid RRF with cross-encoder reranking) retrieved accepted evidence within the top three results for 17 of 18 queries (Recall@3 $= 0 . 9 4 )$ . In this benchmark, accepted evidence chunks were evaluated as alternative individual targets, retaining the maximum score per query and metric. Recall@3 therefore indicates whether at least one accepted alternative appears in the top three, rather than the proportion of all relevant chunks retrieved. This approach reduces dependence on a single annotated chunk but does not establish invariance to overlapping evidence.

Ground-truth curation increased the number of evidence spans from 25 to 37 and accepted chunk targets from 36 to 51. Because retrieval rankings remained unchanged, changes in evaluation scores reflect expanded annotation coverage rather than improved retrieval. However, the audit examined retrievals from the selected configuration, potentially favoring that system in subsequent comparisons.

Hybrid RRF increased R-Precision to 0.78, compared with 0.61 for BM25 and 0.56 for semantic retrieval, while Recall@3 remained at 0.89. Adding the cross-encoder reranker increased Recall@3, R-Precision, MRR@3, and nDCG@3 to 0.94. Segmentation also influenced performance: 384-token chunks achieved the highest MRR@3 (0.89) and nDCG@3 (0.90) in the size sweep, while 20% overlap produced the highest R-Precision, MRR@3, and nDCG@3 among the evaluated overlap settings. The lower performance at 15% overlap indicates that these efects were not monotonic.

Generation performance varied with context depth. Contextual Relevancy decreased from 0.72 at K = 1 to 0.47 at K = 5, whereas Contextual Precision was highest at $K = 5 ~ ( 0 . 9 1 )$ . Contextual Recall reached 0.94 at K = 2, K = 4, and K = 5, while Faithfulness remained between 0.96 and 1.00. Although K = 5 achieved the highest BLEU-1 through BLEU-4 and ROUGE-1/ROUGE-2 scores, ROUGE-L was highest at K = 1. Increasing context depth therefore did not consistently improve all measures of answer quality.

The qualitative cases illustrate the distinction between benchmark success and evidence sufficiency. In Q02, the retrieved passage supported an afirmative answer but did not explicitly address sampling adequacy in the quoted excerpt. In Q09, the rank-one passage addressed the relevant biopsy technique without establishing the requested comparison; the accepted comparative evidence appeared only at rank seven. These cases demonstrate that topical relevance and agreement with annotated targets do not necessarily establish coverage of every clinical qualifier.

Several limitations constrain interpretation. The corpus contained 18 documents and 18 simple, direct queries, with the same benchmark used for configuration selection and evaluation. A single query changes Recall@3 by approximately 0.06, making diferences between configurations sensitive to individual cases. Generation quality was evaluated using an LLM judge and lexical metrics without human validation. Additionally, noisy chunks were excluded from ground-truth candidates, and the corpus did not include the discordant guidance recommending vacuum-assisted excision as the default for B3 lesions. Generalization to independent questions, more complex evidence requirements, and conflicting clinical sources remains untested.

## 4.2 Clinical Implications

Following image-guided biopsy, breast radiologists assess radiologic–pathologic concordance and determine whether surgical excision, imaging surveillance, or return to screening is appropriate. These decisions depend on lesion-specific recommendations and clinical qualifiers, including atypia and sampling adequacy. Retrieving supporting passages with identifiable sources could facilitate verification against the relevant literature, although this potential benefit was not evaluated clinically.

For clinical decision support, retrieving an accepted evidence target is insuficient if the passage omits conditions necessary for the management recommendation. The evidence-aware evaluation identifies recognized supporting passages but does not independently establish their clinical suficiency. Expert assessment is therefore required to determine whether retrieved evidence preserves the relevant conditions and supports the intended interpretation.

A planned reader study will compare a breast imaging radiologist’s assessment of retrieved passages with the automated ranking. Disagreements may inform future ground-truth revisions and evaluation of clinical evidence suficiency. The system is intended to support evidence retrieval rather than replace clinical judgment, and its efectiveness in clinical workflows remains to be established.

## 5 Conclusion and Future Work

The RAG-MAMA evaluation illustrates how retrieval performance depends on both system configuration and the coverage of relevance annotations. On the 18-query development benchmark, hybrid RRF with cross-encoder reranking retrieved at least one accepted evidence alternative within the top three results for 17 queries (Recall@3 = 0.94). Ground-truth curation expanded canonical evidence spans from 25 to 37 and accepted chunk targets from 36 to 51 without changing retrieval rankings. Generation performance also varied across context depths, with no single setting consistently maximizing all metrics. These findings are limited to a small development benchmark and do not establish clinical evidence suficiency or clinical efectiveness.

The immediate priorities are a blinded audit of candidate passages pooled across multiple retrievers and evaluation on independent questions without further configuration tuning. A planned reader study will ask a fellowship-trained breast imaging radiologist to assess and rank the top five retrieved chunks for up to 20 stratified queries, flag irrelevant passages, and evaluate whether the evidence preserves the clinical qualifiers needed for management decisions. Human assessment of generated-answer faithfulness and correctness is also needed. The benchmark should subsequently be extended to comparative questions and questions requiring complementary evidence from multiple passages.

Further technical work will compare fixed-size chunking with semantic and section-aware segmentation, particularly for evidence spanning paragraphs, tables, and quantitative results. We also plan to investigate down-weighting noisy chunks instead of excluding them, including fixed versus proportional penalties, document-position efects, and query-dependent rules. Graph-based retrieval will be explored for relational and multi-hop evidence needs.

## References

[1] L. M. Amugongo, P. Mascheroni, S. G. Brooks, S. Doering, and J. Seidel. Retrieval augmented generation for large language models in healthcare: A systematic review. PLOS Digital Health, 4(6):e0000877, 2025. https://doi.org/10.1371/journal.pdig.0000877

[2] C. Buckley and E. M. Voorhees. Retrieval evaluation with incomplete information. In Proc. 27th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 25–32, 2004. https://doi.org/10.1145/1008992.1009000

[3] S. Büttcher, C. L. A. Clarke, P. C. K. Yeung, and I. Soborof. Reliable information retrieval evaluation with incomplete and biased judgements. In Proc. 30th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 63–70, 2007. https://doi.org/10.1145/ 1277741.1277755

[4] G. V. Cormack, C. L. A. Clarke, and S. Büttcher. Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. In Proc. 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 758–759, 2009. https://doi.org/10.1145/1571941. 1572114

[5] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert. RAGAS: Automated evaluation of retrieval augmented generation. In Proc. 18th Conference of the European Chapter of the ACL: System Demonstrations, pp. 150–158, 2024. https://doi.org/10.18653/v1/2024.eacl-demo.16

[6] D. Ferber et al. Development and validation of an autonomous artificial intelligence agent for clinical decision-making in oncology. Nature Cancer, 6(8):1337–1349, 2025. https://doi.org/10.1038/ s43018-025-00991-6

[7] A. Grattafiori et al. The Llama 3 herd of models. arXiv:2407.21783, 2024. https://doi.org/10.48550/ arXiv.2407.21783

[8] K. Järvelin and J. Kekäläinen. Cumulated gain-based evaluation of IR techniques. ACM Transactions on Information Systems, 20(4):422–446, 2002. https://doi.org/10.1145/582415.582418

[9] F. Jelinek, R. L. Mercer, L. R. Bahl, and J. K. Baker. Perplexity—a measure of the dificulty of speech recognition tasks. Journal of the Acoustical Society of America, 62(S1):S63, 1977. https://doi.org/ 10.1121/1.2016299

[10] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii, Y. J. Bang, A. Madotto, and P. Fung. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38, 2023. https://doi.org/10.1145/3571730

[11] M. G. Kendall. A new measure of rank correlation. Biometrika, 30(1/2):81–93, 1938. https://doi. org/10.1093/biomet/30.1-2.81

[12] H. Kim et al. Rethinking retrieval-augmented generation for medicine: A large-scale, systematic expert evaluation and practical insights. arXiv:2511.06738, 2025. https://doi.org/10.48550/arXiv.2511. 06738

[13] T. Kwiatkowski et al. Natural Questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453–466, 2019. https://doi.org/10.1162/tacl\_ a\_00276

[14] P. Lewis et al. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems 33, pp. 9459–9474, 2020.

[15] C.-Y. Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out (ACL Workshop), pp. 74–81, 2004.

[16] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157–173, 2024. https://doi.org/10.1162/tacl\_a\_00638

[17] R. Nogueira and K. Cho. Passage re-ranking with BERT. arXiv:1901.04085, 2019. https://doi.org/ 10.48550/arXiv.1901.04085

[18] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu. BLEU: A method for automatic evaluation of machine translation. In Proc. 40th Annual Meeting of the ACL, pp. 311–318, 2002. https://doi.org/10.3115/ 1073083.1073135

[19] F. Petroni et al. KILT: A benchmark for knowledge intensive language tasks. In Proc. 2021 Conference of the North American Chapter of the ACL (NAACL), pp. 2523–2544, 2021. https://doi.org/10. 18653/v1/2021.naacl-main.200

[20] C. J. Rageth, E. A. M. O’Flynn, K. Pinker, R. A. Kubik-Huch, A. Mundinger, T. Decker, et al. Second International Consensus Conference on lesions of uncertain malignant potential in the breast (B3 lesions). Breast Cancer Research and Treatment, 174(2):279–296, 2019. https://doi.org/10. 1007/s10549-018-05071-1

[21] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang. SQuAD: 100,000+ questions for machine comprehension of text. In Proc. 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 2383–2392, 2016. https://doi.org/10.18653/v1/D16-1264

[22] S. Robertson and H. Zaragoza. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4):333–389, 2009. https://doi.org/10.1561/1500000019

[23] K. Singhal et al. Large language models encode clinical knowledge. Nature, 620:172–180, 2023. https: //doi.org/10.1038/s41586-023-06291-2

[24] B. Smith and A. Troynikov. Evaluating chunking strategies for retrieval. Technical report, Chroma, 2024. https://www.trychroma.com/research/evaluating-chunking

[25] S. Tayebi Arasteh et al. RadioRAG: Online retrieval-augmented generation for radiology question answering. Radiology: Artificial Intelligence, 7(4):e240476, 2025. https://doi.org/10.1148/ryai. 240476

[26] D. A. Weinert and A. M. Rauschecker. Enhancing large language models with retrieval-augmented generation: A radiology-specific approach. Radiology: Artificial Intelligence, 7(3):e240313, 2025. https: //doi.org/10.1148/ryai.240313

[27] E. B. Wilson. Probable inference, the law of succession, and statistical inference. Journal of the American Statistical Association, 22(158):209–212, 1927. https://doi.org/10.1080/01621459.1927. 10502953

[28] S. Xiao, Z. Liu, P. Zhang, N. Muennighof, D. Lian, and J.-Y. Nie. C-Pack: Packed resources for general Chinese embeddings. In Proc. 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 641–649, 2024. https://doi.org/10.1145/3626772.3657878

[29] G. Xiong, Q. Jin, Z. Lu, and A. Zhang. Benchmarking retrieval-augmented generation for medicine. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 6233–6251, 2024. https: //doi.org/10.18653/v1/2024.findings-acl.372

[30] Z. Yang et al. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proc. 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 2369–2380, 2018. https://doi.org/10.18653/v1/D18-1259

[31] H. Yu, A. Gan, K. Zhang, S. Tong, Q. Liu, and Z. Liu. Evaluation of retrieval-augmented generation: A survey. arXiv:2405.07437, 2024. https://doi.org/10.48550/arXiv.2405.07437

[32] C. Zakka et al. Almanac: Retrieval-augmented language models for clinical medicine. arXiv:2303.01229, 2023. https://doi.org/10.48550/arXiv.2303.01229

[33] J. Zobel. How reliable are the results of large-scale information retrieval experiments? In Proc. 21st International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 307– 314, 1998. https://doi.org/10.1145/290941.291014

[34] G. Trappolini et al., “Redefining Retrieval Evaluation in the Era of LLMs,” in Proceedings of the 2026 Conference of the European Chapter of the Association for Computational Linguistics (EACL), 2026. Available: https://arxiv.org/abs/2510.21440.

[35] Y. Cao et al., “HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conver sational Agents,” in Findings of the Association for Computational Linguistics: ACL 2026. Available: https://arxiv.org/abs/2604.18349.

[36] N. C. Ide, R. F. Loane, and D. Demner-Fushman, “Essie: A Concept-based Search Engine for Structured Biomedical Text,” Journal of the American Medical Informatics Association, vol. 14, no. 3, pp. 253–263, 2007.

[37] S. Han, “Finding the Best Chunking Strategy for Accurate AI Responses,” NVIDIA Technical Blog, 2025. Available: https://developer.nvidia.com/blog/ finding-the-best-chunking-strategy-for-accurate-ai-responses/.

[38] American Society of Breast Surgeons. Resource Guide: Surgical Management of Benign or High-Risk Lesions. American Society of Breast Surgeons, 2024. Available: https://www.breastsurgeons.org/ docs/statements/asbrs-high-risk-lesions.pdf.

## A Supplementary Material

This supplementary material reports the exploratory query-formulation analysis, including its experimental setup and descriptive retrieval results.

## A.1 Query Formulation Sensitivity Analysis

## A.1.1 Experimental Setup

Query-formulation sensitivity was evaluated across seven conditions (Short, Medium, Long, Canonical, Paraphrased, Yes/No, and WH), with 18 queries per condition. Fixed variants were authored through Codex (recorded session model: gpt-5.6-luna; gpt-6.1-sol for eight additional indirect forms). Medium and Canonical retained the original queries; mean Short, Medium, and Long lengths were 11, 13, and 18 words, respectively. Paraphrases varied wording or syntax while aiming to preserve the original answers and evidence. Eight Yes/No variants were indirect requests containing WH clauses, limiting a strict binary-versus-open question comparison.

All 126 variants were evaluated on the selected 384-token index (77-token overlap; 757 chunks) using the curated ground truth and the final ablation’s retrieval settings and scoring protocol. GPT-2 perplexity was reused from prior evaluations of identical query texts and averaged by condition, without new perplexity inference. Answer generation and LLM-based judging were disabled. Analyses were descriptive.

## A.1.2 Results

Table S1 summarizes retrieval performance across seven query-formulation conditions. Mediumlength queries achieved the highest R-Precision (0.89), MRR@3 (0.92), and nDCG@3 (0.91) among the length variants. Canonical and paraphrased queries performed similarly, with R-Precision values of 0.89 and 0.88, respectively. Yes/No questions outperformed WH questions in Recall@3 (0.89 versus 0.83), R-Precision (0.87 versus 0.84), and nDCG@3 (0.91 versus 0.85), while WH questions achieved slightly higher MRR@3 (0.92 versus 0.91). Neither query length nor perplexity showed a consistent monotonic association with retrieval performance. These diferences are descriptive and were not tested for statistical significance.

Table S1: Query formulation sensitivity analysis on the 18-query benchmark. → 1 indicates the ideal retrieval score; ↓ indicates lower perplexity. Bold values identify the best retrieval scores within each formulation category before rounding.
<table><tr><td>Condition</td><td>Mean PPL ↓</td><td>Recall@3 → 1</td><td>R-Precision → 1</td><td>MRR@3 → 1</td><td>nDCG@3 → 1</td></tr><tr><td>Length: Short</td><td>300.05</td><td>0.85</td><td>0.80</td><td>0.86</td><td>0.84</td></tr><tr><td>Length: Medium</td><td>231.77</td><td>0.89</td><td>0.89</td><td>0.92</td><td>0.91</td></tr><tr><td>Length: Long</td><td>134.84</td><td>0.86</td><td>0.75</td><td>0.88</td><td>0.85</td></tr><tr><td>Wording: Canonical</td><td>231.77</td><td>0.89</td><td>0.89</td><td>0.92</td><td>0.91</td></tr><tr><td>Wording: Paraphrased</td><td>302.03</td><td>0.88</td><td>0.88</td><td>0.92</td><td>0.90</td></tr><tr><td>Form: Yes/No</td><td>234.39</td><td>0.89</td><td>0.87</td><td>0.91</td><td>0.91</td></tr><tr><td>Form: WH</td><td>169.93</td><td>0.83</td><td>0.84</td><td>0.92</td><td>0.85</td></tr></table>