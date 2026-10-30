# IndicIPR-QA: A Benchmark for Extractive Question Answering on Indian IPR Laws

M. Mohaiminul Islam

Department of Computer Science and Engineering

South Asian University, New Delhi, India

niloy@students.sau.ac.in

Muhammad Abulaish<sup>\*</sup>, SMIEEE

Department of Computer Science and Engineering

South Asian University, New Delhi, India

abulaish@ieee.org

Abstract—Legal Question Answering (QA) over statutory texts is a challenging task in Natural Language Processing (NLP), especially in low-resource settings with complex legal systems. It requires answers that are precise and grounded in the law. However, to the best of current knowledge, no existing extractive QA benchmark provides exact answer spans from statutory texts across all seven Indian Intellectual Property Rights (IPR) Acts. This paper presents IndicIPR-QA<sup>1</sup>, an extractive QA benchmark for Indian IPR laws. The proposed dataset contains 4,462 QA pairs from 1,116 statutory contexts across seven Acts. Questions and answers are generated using GPT-4o and validated through nine automatic quality checks, single and multi-model LLM evaluations, and a 200-pair human audit. Section-level partitioning prevents source-context overlap across the training, validation, and test sets. Further analysis examines answer span length, lexical overlap, and multi-clause structure across the corpus and individual Acts. BM25 achieves a token-level F1 score of 20.21, while fine-tuned InLegalBERT reaches 85.02 under a unified sliding-window inference protocol. The large performance gap between BM25 (20.21 F1) and InLegalBERT (85.02 F1) shows that simple word matching is not enough to answer these legal questions. Performance also varies across the seven Acts, indicating substantial differences in answer-extraction performance.

Index Terms—IndicIPR-QA Dataset, Legal NLP, Extractive QA, Indian IPR Law, Quality Validation.

## I. INTRODUCTION

Intellectual Property Rights (IPR) law covers many economically and socially important areas including pharmaceutical innovation, creative works, agricultural plant varieties, and region specific products. In India, IPR law is covered by seven principal statutes: the Patents Act, 1970; the Trade Marks Act, 1999; the Copyright Act, 1957; the Designs Act, 2000; the Geographical Indications of Goods (Registration and Protection) Act, 1999; the Semiconductor Integrated Circuits Layout-Design Act, 2000; and the Protection of Plant Varieties and Farmers’ Rights Act, 2001. Although these Acts belong to the same legal domain, each covers different subject matter and uses its own terminology, procedures, exceptions, and cross references. As a result, finding the relevant legal provision can be difficult for legal professionals, researchers, and other stakeholders. A startup may need to determine trademark registration requirements; an inventor may need the legal

conditions for patentability; a farmer or breeder may need information on plant-variety rights; and a designer or copyright owner may need to identify available forms of protection or remedies. The required information is present in the legislation, but it is often buried in long multi-clause provisions, nested subsections, and specialised legal language. The challenge is therefore not simply accessing the law, but locating a precise and traceable answer within it.

Statutory Question Answering (QA) also creates a specific evaluation challenge. A plausible paraphrase may be unsuitable when the exact wording of the law is required, while returning an entire provision may be correct but unnecessarily long. A useful benchmark should therefore evaluate both whether the relevant provision is found and whether the answer span is selected correctly. These considerations motivate IndicIPR-QA’s focus on answer in context validity, span minimality, source-separated splits, and item-level error analysis.

Extractive QA provides a suitable setting for this problem because each predicted answer can be traced directly to its source provision. However, existing QA benchmarks are mainly general domain, contract based, non extractive, or focused on court judgments rather than Indian statutory IPR law. IndicIPR-QA addresses this gap by providing a benchmark for developing and evaluating statutory search and grounded legal QA systems. It is intended as a research benchmark, not as a deployed legal advice system. The major contributions of this paper can be outlined as follows:

1) IndicIPR-QA contains 4,462 extractive QA pairs from 1,116 statutory contexts covering seven major Indian IPR Acts. The quality of the dataset is validated through nine automatic checks, LLM based evaluation, and a 200 pair human audit by two law graduates.

2) The analysis studies answer length, word overlap between questions and legal texts, multi-clause answers, and differences in answer extraction performance across the seven Acts.

3) Four BERT-based encoders and three open LLMs are benchmarked. BM25 achieves a token level F1 of 20.21, while InLegalBERT reaches 85.02, giving a 64.81 point gap that shows that simple word matching is not enough for this task (Section V).

## II. RELATED WORK

Existing research has introduced several benchmark datasets for question answering and reasoning over legal texts, covering tasks such as extractive QA, statutory retrieval, legal reasoning, contract understanding, and judgment analysis. SQuAD established the standard extractive QA setting, where the answer is selected as a span from a given passage [1], and SQuAD 2.0 later added unanswerable questions [2]. Although these datasets have had a major impact, they use generaldomain text and do not capture the structure, terminology, and constraints of legal statutes. Legal QA datasets cover several other settings. CUAD provides expert annotated answer spans from commercial contracts [3], while JEC-QA contains multiple choice questions from the Chinese judicial examination [4]. PrivacyQA, LLeQA, and VLQA focus on privacy policies, legal consultation, and questions from Vietnamese citizens, respectively [5]–[7]. Statutory retrieval benchmarks such as COLIEE [8] and BSARD [9] focus on finding relevant legal provisions, while IPBench evaluates LLM knowledge across a broader bilingual set of intellectual property tasks [10].

These datasets also differ in how they evaluate legal QA. Retrieval benchmarks mainly measure whether the relevant legal provision is found, whereas extractive QA also requires the correct start and end positions of the answer within that provision. Multiple choice and generative tasks do not directly measure whether the predicted answer matches the source text. Extractive QA is therefore useful when answers need to remain directly grounded in the law. In this setting, answer quality depends not only on finding the correct provision but also on selecting an appropriate answer span.

Research on Indian legal NLP has explored domain-specific language modelling and tasks such as legal statute identification, judgment segmentation, and court appeal prediction [11]. Statutes and judgments have different structures and purposes: statutes define rules through sections, definitions, provisions, and cross-references, while judgments explain how those rules are applied to specific disputes. LLM assisted data construction provides a scalable alternative to fully manual annotation, but it can also introduce incorrect spans, repeated question patterns, and inconsistent answer boundaries. These issues make careful quality control important when building automatically generated QA datasets.

Overall, existing resources cover legal retrieval, reasoning, contracts, judgments, and broader intellectual property tasks. To our knowledge, existing resources do not provide an extractive QA benchmark covering all seven major Indian IPR Acts. IndicIPR-QA addresses this gap by providing extractive QA pairs with answer spans directly grounded in Indian statutory text, along with quality checks.

## III. DATASET CONSTRUCTION

The dataset construction proceeded in five phases: corpus creation, QA generation, quality control, section level splitting, and audit/validation. Fig. 1 and Algorithm 1 provide an overview of the full pipeline, from raw statute text to the curated benchmark.

Algorithm 1 IndicIPR-QA: From raw statutes to curated   
benchmark   
Input: Act PDF set $\overline { { \mathcal { P } = \{ P _ { 1 } , \ldots , P _ { 7 } \} } }$ ; chunk size $\delta = 1 4 0 0$   
Output: Curated dataset D, retained contexts ${ \mathcal { C } } ^ { * }$ , and splits $\left( \mathcal { D } _ { \mathrm { t r } } , \mathcal { D } _ { \mathrm { v a } } , \mathcal { D } _ { \mathrm { t e } } \right)$   
Step 1: Corpus extraction   
1: $c \gets \emptyset$   
2: for all $P _ { i } \in \mathcal { P }$ do   
3: $T _ { i } \gets \mathrm { E x T R A C T T E X T } ( P _ { i } )$ // TOC removal, body start detection   
4: $\bar { T _ { i } } \gets \mathrm { C L E A N T E X T } ( \bar { T _ { i } } ) \qquad / /$ Remove editorial lines, normalize whitespace   
5: $\mathcal { C }  \mathcal { C } \cup \mathbf { C } \mathrm { H U N K C O N T E X T S } ( T _ { i } , \delta )$ // Sentence-safe chunking   
6: end for   
7: C ← VALIDATECONTEXTS(C) // 1,173 valid contexts   
Step 2: QA pair generation   
8: ${ \mathcal { R } } \gets \emptyset$   
9: for all $c \in { \mathcal { C } } \ .$ do   
10: $\mathcal { R }  \mathcal { R } \cup$ ∪ GENERATEQA(c) // GPT-4o; up to 5 extractive QA pairs   
11: end for   
// 5,721 raw candidate pairs   
Step 3: Quality control and context-balanced selection   
12: $\mathcal { R ^ {  }  \mathrm { A P P L Y \bar { F } I L T E R S } ( \mathcal { R } , F _ { 1 } , . . . , F _ { 9 } ) } \quad / \cup .$ 4,689 pairs after automatic filtering   
13: D ← CONTEXTBALANCEDSELECT(R, 4462) // Score-based selection   
14: C<sup>∗</sup> ← RETAINCOVEREDCONTEXTS $( \mathcal { C } , \mathcal { D } ) \quad \mathrm { ~ / ~ } 1 , 1 1 6$ retained statutory contexts   
Step 4: Section-level dataset splitting   
15: G ← GROUPBYSECTION(D) // 681 unique statute sections   
16: $( \mathcal { D } _ { \mathrm { t r } } , \mathcal { D } _ { \mathrm { v a } } , \mathcal { D } _ { \mathrm { t e } } ) \gets \mathrm { S T R A }$ TIFIEDSPLIT(G) // Per-Act split, seed 42;   
3,710/408/344   
Step 5: Audit and validation   
17: LLMJUDGEAUDIT(D, GEMMA-3:12B) // Single-judge audit   
18: LLMJUDGEAUDIT(D, ENSEMBLE-3) // Three-model ensemble   
19: $\mathrm { H U M A N V A L I D A T I O N } ( \mathcal { D } _ { \mathrm { t e } } )$ // 200-pair sample, 2 annotators   
20: return $\mathcal { D } , \mathcal { C } ^ { * } , ( \mathcal { D } _ { \mathrm { t r } } , \mathcal { D } _ { \mathrm { v a } } , \mathcal { D } _ { \mathrm { t e } } )$

## A. Source Corpus and QA Generation

The full text of all seven Indian IPR Acts is collected from official Government of India PDFs available on the India Code portal (https://www.indiacode.nic.in/) using pdfplumber. During preprocessing, table of contents pages were removed, editorial amendment markers such as “Ins.,” “Subs.,” and “w.e.f.” were deleted, and whitespace was normalised. The cleaned text is then divided into 1,173 paragraph-level contexts, each about 1,400 characters long, while preserving sentence and section boundaries. These chunks are called statutory contexts throughout the paper.

For each context, GPT-4o [12] is prompted to generate up to five QA pairs. The prompt requires a structured JSON output, questions based on specific facts or legal provisions, and answers copied exactly from the context. It also prohibits paraphrased answers, yes/no or vague questions, answer leakage, and near-duplicate questions. The full GPT-4o generation prompt and generation script are provided in the public repository.<sup>2</sup> This process generated 5,721 raw QA candidates.

## B. Nine-Step Quality Control Pipeline

All 5,721 raw pairs pass through nine sequential filters. F1 checks for non empty fields; F2 verifies that the answer is an exact span from the context; F3 keeps answers from 5 to 350 characters; F4 keeps answers covering less than 85% of the context; F5 requires questions to contain at least five words, end with a question mark, and avoid vague section-summary phrasing; F6 removes questions that contain the answer; F7 removes incomplete contexts; F8 removes editorial amendment markers from answers; and F9 removes standalone 1 to 4 digit answers. Together, these filters remove schema errors, <sup>†</sup>F3 removes 595 pairs during sequential filtering. When F3 is omitted, only 368 pairs are recovered because the remaining 227 F3-rejected candidates are still removed by one or more of the other active filters. <sup>‡</sup>F4 removes 417 pairs during sequential filtering. When F4 is omitted, all 417 pairs are recovered, increasing the accepted set from 4,689 to 5,106.

![](images/0549f53decfc25e1da755107b6849e89add057f0b528760c5627033b12a60613.jpg)  
Fig. 1. End-to-end IndicIPR-QA construction. Human-audit percentages denote joint acceptability on the 200-pair sample; LLM scores are post-hoc diagnostics.

unsupported answers, overly short or overly long spans, weak questions, and extraction artefacts.

F3 and F4 remove 595 and 417 pairs, respectively, accounting for 1,012 of the 1,032 automatic exclusions. This shows that answer span length is the main source of automatic filtering. The remaining 4,689 pairs then undergo context balanced selection. Candidates are ranked based on the length of questions and answers while maintaining coverage across contexts. This produces the final 4,462 QA pairs from 1,116 statutory contexts. Removing this selection step reduces test F1 by 1.34 points (Table II). Finally, section level partitioning creates training, validation, and test sets of 3,710/408/344 QA pairs, with no shared source context across the splits.

## C. Pipeline Ablation Study

To assess the contribution of each filter, a leave one filter out ablation is performed while keeping the remaining eight filters active. Table I reports the number of additional candidates admitted when each filter is disabled.

The results show that F3 and F4 have the largest effect on the accepted set. Disabling F3 admits 368 additional potentially noisy pairs (+7.8% relative to the 4,689 pair baseline), while disabling F4 admits 417 additional pairs (+8.9%). The main automatically detected issue is answer span length: generated spans are often either too short to be informative or too long, covering most of the source context.

## D. Model Level Filter Ablation

To test whether the quality control pipeline affects model performance, InLegalBERT is trained on six different trainingdata variants while keeping the validation and test sets fixed. All variants in Table II, including the clean-selected reference, are trained separately in this ablation study. The main baseline model in Table VII is trained separately.

Table II shows that the largest drop in F1 comes from the selection step rather than from any single filter. Removing

## TABLE I

QUALITY-FILTER ANALYSIS OF THE 5,721 RAW GPT-4O GENERATED CANDIDATE QA PAIRS. SEQUENTIAL FILTERING APPLIES FILTERS F1 TO F9 IN ORDER. PER-FILTER ABLATION OMITS EACH FILTER INDIVIDUALLY WHILE KEEPING THE REMAINING FILTERS ACTIVE.

<table><tr><td rowspan="2">ID Filter criterion</td><td rowspan="2"></td><td colspan="2">Sequential filtering</td><td colspan="2">Per-filter ablation</td></tr><tr><td>Removed Retained</td><td></td><td>Recovered pairs Accepted</td><td></td></tr><tr><td></td><td>F1 Required fields</td><td>0</td><td>5,721</td><td>0 (0.0%)</td><td>4,689</td></tr><tr><td>F2</td><td>Answer in context</td><td>0</td><td>5,721</td><td>0 (0.0%)</td><td>4,689</td></tr><tr><td>F3</td><td>Answer length 5–350†</td><td>595</td><td>5,126</td><td>368 (7.8%)</td><td>5,057</td></tr><tr><td>F4</td><td>Coverage &lt;85%‡</td><td>417</td><td>4,709</td><td>417 (8.9%)</td><td>5,106</td></tr><tr><td></td><td>F5 Question quality</td><td>13</td><td>4,696</td><td>13 (0.3%)</td><td>4,702</td></tr><tr><td></td><td>F6 No answer leakage</td><td>1</td><td>4,695</td><td>1 (0.0%)</td><td>4,690</td></tr><tr><td>F7</td><td>Context not truncated</td><td>3</td><td>4,692</td><td>3 (0.1%)</td><td>4,692</td></tr><tr><td></td><td>F8 No editorial artefacts</td><td>3</td><td>4,689</td><td>3 (0.1%)</td><td>4,692</td></tr><tr><td>F9</td><td>Answer not num-only</td><td>0</td><td>4,689</td><td>0 (0.0%)</td><td>4,689</td></tr><tr><td colspan="6">1,032 4,689</td></tr></table>

context balanced selection (heuristic all) lowers F1 by 1.34 points. Using all structurally valid raw candidates lowers F1 by 0.67 points, while removing F4 alone lowers it by 0.21 points.

The other results are less clear. Removing the answer length filter F3 alone increases test F1 slightly by 0.45 points, and removing F3 and F4 together increases it by 0.47 points. Since these results come from single seed runs and no statistical significance test was performed, the ablation results are treated as diagnostic rather than as proof that each filter improves model performance. The filters are retained because they enforce the answer validity and span quality constraints captured by the automatic checks in Table I. Model F1 alone is not enough to measure these properties.

TABLE II  
MODEL LEVEL FILTER ABLATION USING INLEGALBERT ON DIFFERENT TRAINING DATA VARIANTS. ALL ARCHIVED CHECKPOINTS ARE EVALUATED WITH THE UNIFIED DECODER; ∆F1 IS RELATIVE TO THE CLEAN SELECTED RUN.
<table><tr><td>Training variant</td><td>F1</td><td>Char-F1</td><td>∆F1</td></tr><tr><td>clean selected (ablation control)</td><td>84.11</td><td>86.30</td><td></td></tr><tr><td>without F3+F4 combined</td><td>84.58</td><td>87.25</td><td>+0.47</td></tr><tr><td>without F3 (answer length)</td><td>84.56</td><td>87.13</td><td>+0.45</td></tr><tr><td>without F4 (coverage)</td><td>83.90</td><td>86.66</td><td>-0.21</td></tr><tr><td>raw candidates (unfiltered)</td><td>83.44</td><td>85.64</td><td>-0.67</td></tr><tr><td>heuristic all (no selection)</td><td>82.77</td><td>86.59</td><td>-1.34</td></tr></table>

TABLE III

LLM AS JUDGE EVALUATION WITH A SINGLE JUDGE AND A THREE MODEL ENSEMBLE. SCORES RANGE FROM 1 (POOR/FLAGGED) TO 3 (GOOD).
<table><tr><td></td><td colspan="2">Single Judge</td><td colspan="2">Ensemble</td></tr><tr><td>Criterion</td><td>Avg. score</td><td>Flagged</td><td>Avg. score</td><td>Flagged</td></tr><tr><td>Question specificity</td><td>2.92</td><td>1.1%</td><td>2.93</td><td>0.0%</td></tr><tr><td>Answer sufficiency</td><td>2.92</td><td>2.8%</td><td>2.99</td><td>0.4%</td></tr><tr><td>Answer minimality</td><td>2.52</td><td>4.2%</td><td>2.85</td><td>0.4%</td></tr><tr><td>Legal relevance</td><td>2.97</td><td>0.7%</td><td>3.00</td><td>0.0%</td></tr><tr><td>Flagged QA pairs</td><td colspan="2">222 pairs (5.0%)</td><td colspan="2">26 pairs (0.6%)</td></tr></table>

## E. LLM as Judge Quality Verification

To obtain an additional automated assessment of the 4,462 retained QA pairs beyond rule based filtering, an LLM as judge protocol is applied under two different configurations, and their outputs are compared. These scores are used only as auxiliary diagnostic indicators and not as a replacement for human evaluation.

Gemma 3:12B [13] evaluates four criteria: Q SPECIFIC (the question targets a specific legal element), A SUFFICIENT (the answer fully addresses the question), A MINIMAL (the answer is the shortest correct span), and LEGAL RELEVANT (the question concerns substantive legal content). Scores are 1 (poor), 2 (acceptable), or 3 (good), and a pair is flagged if any criterion receives a score of 1.

Gemma 3:12B, Llama 3.1:8B [14], and Mistral 7B [15] evaluate the same four criteria independently. When at least two models assign the same score, that score is used as the ensemble score. The released code contains the complete aggregation rule. Table III compares the evaluation results of the single judge and the three-model ensemble.

The largest difference is in answer minimality. The single judge flags more long answer spans, while the ensemble is more permissive. The 26 pairs flagged by the ensemble include incomplete spans and long section level answers. The three judges give the same score on all four criteria for only 8.0% of the pairs, showing that their judgments can differ. Therefore, the LLM as Judge results are treated as diagnostic evidence rather than as final measures of dataset quality.

## F. Human Validation Study

Two law graduates independently audited 200 test pairs across all seven Acts (seed 42: copyright 25, designs 24, GI 32, patents 29, plant varieties 28, semiconductor 30, and trade marks 32). A sample of 200 pairs is selected as a practical post hoc audit size while maintaining coverage across all seven <sup>†</sup> The near-zero κ for Q CLEAR reflects a ceiling effect: both annotators gave scores $\geq 2$ to 198 of 200 pairs (99.0%).

TABLE IV  
HUMAN INTER-ANNOTATOR AGREEMENT ON 200 QA PAIRS. $\kappa _ { \mathrm { U W } } =$ UNWEIGHTED COHEN’S κ; κ = QUADRATIC-WEIGHTED COHEN’S κ.
<table><tr><td>Criterion</td><td>KUW</td><td>κQW</td><td>Accept. %</td></tr><tr><td>Question Clarity</td><td>-0.008</td><td>0.051</td><td>99.0%†</td></tr><tr><td>Answer Correctness</td><td>0.117</td><td>0.328</td><td>95.0%</td></tr><tr><td>Answer Minimality</td><td>0.312</td><td>0.613</td><td>75.5%</td></tr></table>

<sup>‡</sup> Joint acceptability requires both annotators to give a score ≥ 2. For minimality, 75.5% of pairs meet this threshold, while 82.0% are acceptable to at least one annotator. Both annotators rated 36 pairs as non-minimal (score 1). These pairs remain in the released dataset because the audit is conducted after dataset construction and is not used for filtering.

![](images/bdc3fec48909167258b77185266c0de359b94903fcaf0963e4b1b7af028d3d90.jpg)  
Fig. 2. Criterion-wise inter-annotator agreement for the human validation study. Answer minimality has the strongest agreement, while question clarity shows a ceiling effect due to skewed marginal distributions.

Acts. The audit is diagnostic rather than a precise estimate of quality across all 4,462 pairs or of expert IPR annotation accuracy. Pairs were rated from 1 (poor) to 3 (good) for question clarity, answer correctness, and answer minimality; agreement is measured using unweighted and quadratic-weighted Cohen’s κ [16], [17].

Table IV reports human inter-annotator agreement and joint acceptability. Joint acceptability is 99.0% for clarity and 95.0% for correctness. Most correctness disagreements are between scores 3 and $2 \ ( \kappa _ { \mathrm { Q W } } \ = \ 0 . 3 2 8 )$ . Minimality shows higher agreement $( \kappa _ { \mathrm { Q W } } = 0 . 6 1 3 )$ , with 75.5% joint acceptability.

This highlights the difficulty of selecting the shortest correct span for long, multi-clause answers. Since the audit interface truncated long contexts, these results should be viewed as diagnostic rather than as a complete measure of dataset quality. Figure 2 summarises agreement across the three criteria.

## IV. DATASET ANALYSIS

This section examines answer length, lexical overlap, span structure, and model performance across the dataset and the seven Acts.

## A. Corpus Level Statistics

Table V compares IndicIPR-QA with established benchmarks. Its average answer length is 35.6 words, about $1 1 \times { \mathrm { ~ S Q u A D } } { \mathrm { { s } } }$

TABLE V  
COMPARISON OF INDICIPR-QA WITH RELATED QA BENCHMARKS (SQ1 = SQUAD 1.1 [1]; SQ2 = SQUAD 2.0 [2]; CUAD [3]; JEC = JEC-QA [4]; IPB = IPBENCH [10]). “—” INDICATES UNAVAILABLE OR NOT DIRECTLY COMPARABLE STATISTICS. VALUES FOLLOW THE REPORTING CONVENTIONS OF EACH BENCHMARK’S SOURCE AND MAY USE DIFFERENT DATASET SPLIT DEFINITIONS.
<table><tr><td>Property</td><td>SQ1</td><td>SQ2</td><td>CUAD</td><td>JEC</td><td>IPB</td><td>IndicIPR-QA (Ours)</td></tr><tr><td>QA pairs</td><td>87.6k</td><td>150k</td><td>13.1k</td><td>26.4k</td><td>10.4k</td><td>4.5k</td></tr><tr><td>Contexts</td><td>18.9k</td><td>35k</td><td>510</td><td></td><td></td><td>1,116</td></tr><tr><td>QA/context</td><td>4.6</td><td>4.3</td><td>25.7</td><td></td><td></td><td>4.0</td></tr><tr><td>Ans len (wds)</td><td>3.2</td><td>3.2</td><td>30.0</td><td>MCQ</td><td>Mixed</td><td>35.6</td></tr><tr><td>Ctx len (wds)</td><td>117</td><td>117</td><td>200</td><td></td><td>200</td><td>171.3</td></tr><tr><td>Q len (wds)</td><td>10.1</td><td>10.1</td><td>9.5</td><td>一</td><td></td><td>11.9</td></tr><tr><td>Human review</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes†</td></tr><tr><td>Extractive</td><td>Yes</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td><td>Yes</td></tr><tr><td>Unanswerable</td><td>No</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td><td>No</td></tr><tr><td>Domain</td><td>General General Contracts Chinese Bar Exam US/Chinese IP Law</td><td></td><td></td><td></td><td></td><td>Indian IPR</td></tr></table>

<sup>†</sup>Audited on 200 test pairs by two law graduates; see Section III-F.

TABLE VI  
PER-ACT INDICIPR-QA STATISTICS, SORTED BY TEST F1. $n _ { \mathrm { a l l } }$ AND n<sub>test</sub> DENOTE FULL-DATASET AND TEST-SET COUNTS; Q-CTX DENOTES STOPWORD REMOVED QUESTION-CONTEXT UNIGRAM OVERLAP.
<table><tr><td>Act</td><td> $\mathbf { n } _ { \mathbf { a l l } }$ </td><td> $\scriptstyle \mathbf { n } _ { \mathbf { t e s t } }$ </td><td>Avg. ans. (words)</td><td>Q-Ctx</td><td>F1</td></tr><tr><td>Designs</td><td>173</td><td>24</td><td>43.4</td><td>0.320</td><td>100.00</td></tr><tr><td>Patents</td><td>1,143</td><td>69</td><td>35.8</td><td>0.429</td><td>98.40</td></tr><tr><td>Semicond.</td><td>585</td><td>47</td><td>22.1</td><td>0.418</td><td>91.53</td></tr><tr><td>Trade Marks</td><td>1,106</td><td>88</td><td>31.9</td><td>0.416</td><td>83.83</td></tr><tr><td>GI</td><td>551</td><td>51</td><td>51.8</td><td>0.458</td><td>83.66</td></tr><tr><td>Copyright</td><td>349</td><td>25</td><td>62.3</td><td>0.447</td><td>79.41</td></tr><tr><td>Plant Var.</td><td>555</td><td>40</td><td>21.9</td><td>0.636</td><td>53.20</td></tr></table>

3.2 words and 19% longer than CUAD’s 30.0 words, which may increase span boundary complexity. The dataset is fully extractive, contains no unanswerable questions, and uses section level train test separation.

## B. Per-Act Distribution and Performance

Table VI reports per-Act statistics and InLegalBERT test performance. Performance varies substantially across the seven Acts.

Test F1 ranges from 100.00 for Designs to 53.20 for Plant Varieties. The perfect score for Designs should be viewed with caution because the test set contains only 24 pairs. Across the seven Acts $( \mathfrak { n } = 7 )$ , question-context unigram overlap shows a strong negative correlation with F1 $( r = - 0 . 9 2 8 )$ . Given the small number of Acts, this correlation should be interpreted as exploratory rather than as evidence of a general relationship. Plant Varieties have the highest overlap (0.636) but the lowest F1. One possible reason is the presence of specialised biological and agricultural terms, although this is not directly tested. Copyright has more multi-clause answers (53.9%) than Patents (13.7%) and also has lower F1 (79.41 versus 98.40).

The answers cover several legal topics, including authority (29.4%), procedure (12.0%), rights (9.5%), conditions (7.9%), definitions (5.8%), duration (2.3%) and penalties (1.7%), while 31.4% fall into other categories. The average question to context word overlap is 0.448 after removing common stopwords. Also, 19.9% of answers contain semicolons and 5.9% contain “subject $\mathrm { t o } , \vec { \mathbf { \Lambda } }$ showing that many answers contain multiple clauses. When the answers are divided into short and long groups using the median length of 33 words, InLegalBERT F1 decreases from 86.24 for short answers to 83.78 for long answers. Multi-clause answers also have lower boundary-sensitive EM than singleclause answers (56.10 versus 66.03). Answers that appear later in the context have slightly lower F1 than earlier answers (84.21 versus 85.83). Overall, answer length, the number of clauses, and the answer’s position in the context are associated with extraction performance.

## V. BASELINE EXPERIMENTS & RESULTS

Four BERT based encoder models are fine tuned on the IndicIPR-QA training split (3,710 QA pairs) using the BertForQuestionAnswering architecture [18]. This architecture predicts the start and end positions of the answer span in the input sequence: [CLS] question [SEP] context [SEP]. The models cover different levels of domain specialisation: BERT base uncased [18] (general English), LegalBERT-US [19] (US legal text), InCaseLawBERT [20] (Indian court judgments), and InLegalBERT [11] (Indian legal text).

All encoder models use the same settings: AdamW optimiser, learning rate $3 \times 1 0 ^ { - 5 }$ with 50 warm-up steps and linear decay, weight decay 0.01, effective batch size 32, maximum sequence length 384, document stride 128, 3 training epochs, and random seed 42. The same settings are used for all encoder models to ensure a controlled comparison. The best checkpoint is selected using validation F1. During inference, long contexts are divided into overlapping windows using a stride of 128. A constrained decoder then selects the highest-scoring valid answer span across all windows and returns the exact text from the context. All encoder results reported here are recomputed from the archived checkpoints using this same inference procedure; no model is retrained.

Three open instruction tuned LLMs are also evaluated in zero shot and 3 shot settings: Llama 3.1:8B [14], Mistral 7B [15], and Qwen2.5:7B [21]. These models come from different model families and have similar sizes of about 7 to 8B parameters. In the zero shot setting, each model receives the statute passage and question and is instructed to copy the answer exactly from the context. In the 3 shot setting, three example QA pairs from the training set are provided as demonstrations.

TABLE VII  
BASELINE PERFORMANCE ON THE INDICIPR-QA TEST SPLIT (344 QA PAIRS). ZS = ZERO-SHOT; FS = 3-SHOT FEW-SHOT.
<table><tr><td>System</td><td>EM</td><td>F1</td><td>Char-F1</td></tr><tr><td colspan="4">Lexical retrieval</td></tr><tr><td>BM25 retrieval</td><td>0.00</td><td>20.21</td><td>77.10</td></tr><tr><td>BM25 oracle window</td><td>14.24</td><td>51.30</td><td>85.23</td></tr><tr><td colspan="4">Fine-tuned encoders</td></tr><tr><td>BERT-base</td><td>63.66</td><td>83.33</td><td>86.38</td></tr><tr><td>LegalBERT-US</td><td>65.12</td><td>84.66</td><td>86.67</td></tr><tr><td>InČaseLawBERT</td><td>63.08</td><td>84.37</td><td>86.40</td></tr><tr><td>InLegalBERT</td><td>63.66</td><td>85.02</td><td>87.26</td></tr><tr><td colspan="4">Prompted Open-Source LLMs</td></tr><tr><td>Mistral 7B (ZS)</td><td>18.31</td><td>67.11</td><td>75.07</td></tr><tr><td>Mistral 7B (FS)</td><td>24.71</td><td>68.83</td><td>76.01</td></tr><tr><td>Llama 3.1:8B (ZS)</td><td>33.43</td><td>74.78</td><td>79.61</td></tr><tr><td>Llama 3.1:8B (FS)</td><td>33.43</td><td>77.02</td><td>79.78</td></tr><tr><td>Qwen2.5:7B (ŻS)</td><td>29.94</td><td>77.59</td><td>81.04</td></tr><tr><td>Qwen2.5:7B (FS)</td><td>34.30</td><td>76.39</td><td>78.60</td></tr></table>

As a simple lexical baseline, BM25 [22] is also evaluated. For a document D and query $Q , f ( q , D )$ denotes the frequency of token $q , \ | D |$ the document length, and |D| the average document length. The standard values $k _ { 1 } = 1 . 5$ and $b = 0 . 7 5$ are used. In the oracle window setting, BM25 is given the correct answer containing context, isolating within context retrieval. The relevance score is

$$
\mathrm { B M 2 5 } ( D , Q ) = \sum _ { q \in Q } \mathrm { I D F } ( q ) \frac { f ( q , D ) ( k _ { 1 } + 1 ) } { f ( q , D ) + k _ { 1 } \left( 1 - b + b \frac { | D | } { | D | } \right) }\tag{1}
$$

## A. Results

Table VII and Figure 3 summarise the results of all baseline systems using exact match (EM), token level F1, and character level F1 (Char-F1). Standard BM25 retrieval [22] achieves a token level F1 of 20.21, while the oracle window setting reaches 51.30. Both remain below the supervised encoder models. Among the encoders, InLegalBERT achieves the highest F1 (85.02), followed by LegalBERT-US (84.66) and InCaseLawBERT (84.37). BERT-base has the lowest encoder F1 (83.33). However, the differences between the encoder models are small, so the results do not provide strong evidence that domain specific pretraining alone leads to better performance.

A paired bootstrap significance test with 10,000 resamples and Holm correction finds no statistically significant F1 difference between InLegalBERT and BERT-base $( \Delta = + 1 . 7 0$ adjusted $p = 0 . 3 9 7 )$ , LegalBERT-US $( \Delta = + 0 . 3 6$ , adjusted $p ~ = ~ 1 . 0 0 0 )$ , or InCaseLawBERT $( \Delta ~ = ~ + 0 . 6 6$ , adjusted $p = 1 . 0 0 0 )$

Among InLegalBERT’s 125 non exact match predictions, 100 (80.0%) are span boundary errors, 20 (16.0%) have low overlap, two (1.6%) differ by minor words, two (1.6%) partially overlap the reference, and one (0.8%) is completely wrong. For example, one Copyright prediction starts with the correct clause from Section 14 but continues into the next enumerated clause. This shows how long statutory lists can make it difficult to choose the exact answer boundaries.

Among the prompted LLMs, Qwen2.5:7B performs best in the zero-shot setting $( \mathrm { F 1 } = 7 7 . 5 9 )$ , while Llama 3.1:8B performs best in the 3-shot setting $( \mathrm { F } 1 = 7 7 . 0 2 )$ . The best prompted result is still 7.43 F1 points lower than finetuned InLegalBERT, which achieves the highest F1 among the evaluated models. Section VI discusses the scope and limitations of the generalisation.

![](images/d5d41fc69a3e28e5ba371f98baad0d69c503b0fdc344dd50a95f2e7ba6d0ea77.jpg)  
Fig. 3. Token-level F1 across encoder and lexical baselines and the best prompted setting for each open LLM on the IndicIPR-QA test split. The large gap between BM25 and InLegalBERT shows that lexical matching alone is insufficient for the benchmark.

## VI. GENERALIZABILITY ANALYSIS

Section-level splitting prevents the same source context from appearing in both the training and test sets. The train-test context Jaccard similarity is 0.00 for all seven Acts. Figure 4 shows that 100% of test contexts are unseen in the training split for every Act. Among the 1,234 unique answer token types in the test set, 137 (11.1%) do not appear in the training answers. In addition, 15.3% of test question tokens and 13.7% of test context tokens are not found in the training split. This shows that the test set contains unseen legal provisions and words.

![](images/678514f348251e47949f39a5acb3f991901ef34e27a3a3d31452f647714eadf7.jpg)  
Fig. 4. Novel test-context rate by Act. The 100% rate confirms that no source context is shared between the training and test splits.

However, the questions are not completely independent across the two splits. Of the 4,462 questions, 4,118 are unique after converting them to lowercase, replacing non-word characters with spaces, and removing extra spaces. Common question openings include “what does” (1,157), “under section”

(960), and “what is” (875). These repeated GPT-4o question patterns can appear in both training and test sets. Therefore, the results show how well models answer questions generated by the same process over new legal contexts. They do not measure performance on independently written legal questions.

The 64.81-point gap between BM25 and InLegalBERT indicates that simple word matching alone is not sufficient for this task. However, this gap alone does not prove that the models will generalise to real-world legal questions. The BM25 oracle-window F1 score of 51.30 shows that simple lexical matching still performs poorly even when the correct context is provided.

## VII. CONCLUSION AND FUTURE WORK

IndicIPR-QA is an extractive QA benchmark containing 4,462 question-answer pairs from 1,116 statutory contexts covering the seven principal Indian IPR Acts. The dataset is created using GPT-4o with nine automatic quality filters, context-balanced selection, model-based evaluation, and a 200- pair human audit. Section-level splitting prevents the same source context from appearing in both the training and test sets. Using the unified sliding-window inference method, fine-tuned InLegalBERT achieves a token-level F1 of 85.02, while BM25 retrieval achieves 20.21. This large difference shows that simple word matching is not enough for this task. Performance also differs across the seven Acts and different types of answers, and most InLegalBERT errors are related to choosing the exact answer boundaries. These results highlight the difficulty of precise answer-span extraction, especially for long and multi-clause answers. IndicIPR-QA provides a benchmark for studying extractive QA over Indian IPR law. The benchmark is limited to English, answerable, single-context questions, and the questions are generated using GPT-4o. Future work can include independently written questions, more Indian languages, unanswerable questions, and scenario based or multi section questions.

## VIII. DATA AND CODE AVAILABILITY

The IndicIPR-QA dataset and code are publicly available:

Dataset: https://huggingface.co/datasets/niloycste68/ IndicIPRExtractiveQA

Code: https://github.com/niloycste/Indic-IPR-Dataset-

## REFERENCES

[1] P. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, “SQuAD: 100,000+ Questions for Machine Comprehension of Text,” in Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, 2016, pp. 2383–2392.

[2] P. Rajpurkar, R. Jia, and P. Liang, “Know what you don’t know: Unanswerable questions for squad,” in Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), 2018, pp. 784–789.

[3] D. Hendrycks, C. Burns, A. Chen, and S. Ball, “CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review,” in Proceedings of the NeurIPS 2021 Track on Datasets and Benchmarks, 2021.

[4] H. Zhong, C. Xiao, C. Tu, T. Zhang, Z. Liu, and M. Sun, “JEC-QA: A Legal-Domain Question Answering Dataset,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, pp. 9701–9708, 2020.

[5] A. Ravichander, A. W. Black, S. Wilson, T. Norton, and N. Sadeh, “Question answering for privacy policies: Combining computational and legal perspectives,” in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 2019, pp. 4947–4958.

[6] A. Louis, G. Van Dijck, and G. Spanakis, “Interpretable long-form legal question answering with retrieval-augmented large language models,” in Proceedings of the AAAI conference on artificial intelligence, vol. 38, no. 20, 2024, pp. 22 266–22 275.

[7] T.-M. Nguyen, H.-T. Nguyen, T.-K. Dao, X.-H. Phan, H.-T. Nguyen, and T.-H.-Y. Vuong, “VLQA: The first comprehensive, large, and highquality Vietnamese dataset for legal question answering,” arXiv preprint arXiv:2507.19995, 2025.

[8] J. Rabelo, R. Goebel, M.-Y. Kim, Y. Kano, M. Yoshioka, and K. Satoh, “Overview and discussion of the competition on legal information extraction/entailment (COLIEE) 2021,” The Review of Socionetwork Strategies, vol. 16, no. 1, pp. 111–133, 2022.

[9] A. Louis and G. Spanakis, “A statutory article retrieval dataset in French,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2022, pp. 6789– 6803.

[10] Q. Wang, G. Chen, H. Wang, H. Liu, M. Zhu, Z. Qin, L. Li, Y. Yue, S. Wang, J. Li et al., “IPBench: Benchmarking the knowledge of large language models in intellectual property,” arXiv preprint arXiv:2504.15524, 2025.

[11] S. Paul, A. Mandal, P. Goyal, and S. Ghosh, “Pre-trained Language Models for the Legal Domain: A Case Study on Indian Law,” in Proceedings of the Nineteenth International Conference on Artificial Intelligence and Law, ser. ICAIL ’23. Association for Computing Machinery, 2023, pp. 187–196.

[12] A. Hurst, A. Lerer, A. P. Goucher et al., “GPT-4o System Card,” arXiv preprint arXiv:2410.21276, 2024.

[13] Gemma Team, A. Kamath, J. Ferret et al., “Gemma 3 Technical Report,” arXiv preprint arXiv:2503.19786, 2025.

[14] Meta AI et al., “The Llama 3 Herd of Models,” CoRR, vol. abs/2407.21783, 2024.

[15] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. l. Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, L. R. Lavaud, M.-A. Lachaux, P. Stock, T. L. Scao, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed, “Mistral 7B,” arXiv preprint arXiv:2310.06825, 2023.

[16] J. Cohen, “A Coefficient of Agreement for Nominal Scales,” Educational and Psychological Measurement, vol. 20, no. 1, pp. 37–46, 1960.

[17] ——, “Weighted kappa: Nominal scale agreement provision for scaled disagreement or partial credit.” Psychological Bulletin, vol. 70, no. 4, pp. 213–220, 1968.

[18] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pretraining of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), 2019, pp. 4171–4186.

[19] I. Chalkidis, M. Fergadiotis, P. Malakasiotis, N. Aletras, and I. Androutsopoulos, “LEGAL-BERT: The Muppets straight out of Law School,” in Findings of the Association for Computational Linguistics: EMNLP 2020, T. Cohn, Y. He, and Y. Liu, Eds. Association for Computational Linguistics, 2020, pp. 2898–2904.

[20] S. Paul, A. Mandal, P. Goyal, and S. Ghosh, “Pre-training transformers on Indian legal text,” arXiv preprint arXiv:2209.06049, 2022.

[21] Qwen, A. Yang, B. Yang et al., “Qwen2.5 Technical Report,” arXiv preprint arXiv:2412.15115, 2024.

[22] S. Robertson and H. Zaragoza, “The Probabilistic Relevance Framework: BM25 and Beyond,” Foundations and Trends in Information Retrieval, vol. 3, no. 4, pp. 333–389, 2009.