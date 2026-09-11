# Cross-Lingual Clinical Annotation Projection as Constrained Text Generation: A Six-Language Study

Álvaro Rey-Blanes<sup>1,2,3,\*</sup> Francisco J. Moreno-Barea<sup>1,2,3</sup> Francisco J. Veredas<sup>1,2,3</sup>

<sup>1</sup>Department of Programming Languages and Computer Sciences, Universidad de Málaga, Málaga, Spain <sup>2</sup>Research Institute of Multilingual Language Technologies, Universidad de Málaga, Málaga, Spain <sup>3</sup>IBIMA Plataforma BIONAND, Instituto de Investigación Biomédica de Málaga, Málaga, Spain <sup>\*</sup>Corresponding author: alvaroreyb@uma.es

## Abstract

Background: To determine whether cross-lingual clinical annotation projection can be formulated as a text-preserving, document-level generative task that produces verifiable character-level annotations for multilingual clinical corpus construction, and to characterize its robustness and computational trade-ofs relative to candidate-based projection pipelines.

Methods: We developed a constrained LLM projection workflow that inserts entity tags directly into immutable target-language text, followed by deterministic validation and character-ofset reconstruction. We evaluated it alongside supervised candidate-span projection and hybrid ML–LLM refinement for transferring Spanish Disease, Symptom, and Procedure annotations into six languages. Evaluation used MultiClinAI gold standard with strict span matching and character-overlap F1

Results: Direct LLM projection achieved the strongest and most consistent performance. GLM 5.2 obtained a mean Strict F1 of 0.9201 across 18 language–entity combinations, while locally deployable Gemma4:31B achieved 0.9133. The best LLM configuration improved Strict F1 over the previous state of the art in all 18 settings, by 0.0564–0.1512, yielding 55,416 grounded mentions with reconstructed ofsets.

Conclusions: Direct LLM-based projection enables high-quality multilingual clinical annotation transfer and provides a practical approach for extending clinical NLP resources to languages with fewer annotated datasets and language-specific tools. Combined with local inference and deterministic validation, it can substantially reduce expert time and cost for multilingual clinical corpus construction.

Keywords: cross-lingual annotation projection, clinical named entity recognition, multilingual clinical NLP, large language models, clinical corpora

## 1 Background and Significance

Clinical narratives contain essential information on clinical conditions, signs and symptoms, therapeutic interventions, and diagnostic procedures. Automatically identifying these concepts is therefore a central requirement for the secondary use of clinical text in applications such as cohort identification, clinical decision support, epidemiological surveillance, and retrospective research. Named Entity Recognition (NER) provides the span-level representations required by many of these applications, but its development remains highly dependent on manually annotated corpora produced or validated by domain experts [18].

Such resources are unevenly distributed across languages. English benefits from numerous annotated datasets, pretrained models, and evaluation benchmarks, whereas many other languages have substantially fewer reusable clinical corpora. This imbalance is particularly relevant in the clinical domain, where expert annotation is costly and time-consuming.

Cross-lingual annotation projection ofers an alternative to independently annotating equivalent resources in every language. Given an annotated source document and its translation, projection methods identify the corresponding target-language span while preserving its semantic label and associated metadata [30, 7]. A single expert-annotated source corpus can therefore support multiple multilingual resources or provide supervision for targetlanguage extraction systems.

Annotation projection is not equivalent to copying character ofsets between texts. Translation may expand, contract, reorder, or reformulate entity mentions, and sentence restructuring can alter their positions. The task therefore requires not only identifying semantic equivalence but also recovering the exact target boundaries required for span-based evaluation and downstream processing.

Clinical language further complicates projection through specialised terminology, abbreviations, numerical expressions, and variable descriptive formulations. Diseases, symptoms, and procedures may also difer in lexical form and annotation granularity across translations. Projection must therefore preserve both conceptual equivalence and exact span boundaries.

Building on these eforts, MultiClinAI provides a common benchmark for multilingual clinical information extraction and cross-lingual corpus construction [5]. It comprises MultiClinNER, which evaluates clinical entity recognition across seven languages, and MultiClinCorpus, which projects Disease, Symptom, and Procedure annotations from Spanish into English, Dutch, Italian, Romanian, Swedish, and Czech.

## 1.1 Related Work

Cross-lingual annotation projection transfers annotations from a source document to a semantically equivalent target-language text and has been used to create resources for languages with limited manually labelled data [30]. Unlike NER, the source entity, boundaries, and semantic category are known; the task is to locate the equivalent target expression and recover its character ofsets.

Early approaches relied on statistical word alignment, bilingual dictionaries, machine translation, and string similarity [7]. but are less reliable under paraphrasing, reordering, expansion, or contraction. More recent methods use multilingual contextual representations to align tokens or rank candidate spans.

Projected annotations can serve directly as multilingual corpus labels or as weak supervision for target-language NER models [19]. Direct corpus construction requires each projection to match an exact target substring while preserving entity type and annotation scope.

## 1.1.1 Word and Span Alignment

Word alignment identifies translational correspondences between source and target tokens. Contextual multilingual encoders enable this without relying solely on lexical overlap: SimAlign extracts links without task-specific training [14]. whereas AWESoME Align fine-tunes multilingual encoders on parallel data [6].

Token-level correspondences must still be converted into valid entity spans because translations may change the number or continuity of aligned tokens. Proposed solutions include propagating source-span evidence through token alignments [24], ranking explicit target-span candidates [21]. and span-oriented alignment methods designed to preserve annotation boundaries [13].

Boundary sensitivity motivates both exact and relaxed biomedical NER evaluation [29]. Multi-ClinAI combines strict entity-type and characterofset matching with character-overlap metrics, distinguishing localisation errors from boundary disagreements.

## 1.1.2 Clinical Annotation Projection and MultiClinCorpus

Clinical annotation projection has been applied through several paradigms. FRASIMED generated French clinical annotations using BERT-based alignment [31]; Rodríguez-Miret et al. generated Catalan resources from translated Spanish corpora with expert validation [26]; and E3C-3.0 used an LLM-based semi-automatic procedure followed by human revision across several European languages [17, 11]. These studies demonstrate the feasibility of multilingual clinical corpus construction using encoder-based, translation-based, and generative projection methods.

Within MultiClinAI, the MultiClinCorpus shared task provides a common benchmark for cross-lingual clinical annotation projection [5]. Submitted systems explored markedly diferent projection paradigms.

The systems submitted to MultiClinCorpus explored three complementary projection paradigms. Team blue adopted a lightweight lexical approach based on cognate detection and fuzzy string matching [28]. ICB-UMA formulated projection as targetspan candidate ranking: candidates were scored using heuristic constraints and ranked by an XG-Boost model combining surface, positional, and semantic features, with uncertain predictions optionally revised through LLM-based correction [25] .ClinicalAligner26AM used a domain-adapted crosslingual token aligner to project source-span evidence and decode target spans, with variants incorporating MultiClinNER predictions and taskspecific supervision [23]. These approaches represent lexical matching, supervised span selection with generative refinement, and neural token alignment, and provide the closest methodological comparators for this work.

Despite these advances, conventional projection pipelines remain dependent on explicit alignment, candidate generation, or ranking mechanisms. LLMs ofer a document-level alternative that can exploit broader semantic and contextual information without explicit candidate enumeration. However, clinical corpus construction requires projected annotations to remain exactly grounded in the target text, without paraphrasing, normalization, or surface-form alteration, and to yield deterministic character ofsets. The key question is therefore whether generative semantic matching can be constrained to preserve textual integrity and produce verifiable, corpus-ready annotations.

This study makes three contributions. First, we formulate projection as constrained document-level text tagging, treating the target document as an immutable textual surface rather than relying on explicit alignment or pre-generated candidates. Second, we couple LLM inference with deterministic validation and character-ofset reconstruction to verify tag structure, entity counts, and text preservation and produce reproducible character-level annotations. Third, we evaluate this formulation across six target languages and three entity types against supervised candidate-based and hybrid ML– LLM comparators, characterising robustness, exactboundary accuracy, and computational trade-ofs.

## 2 Objective

The objective of this study was to determine whether cross-lingual clinical annotation projection can be formulated as a constrained, text-preserving, document-level generative task capable of producing accurate and verifiable character-level annotations for multilingual clinical corpus construction. We further assessed the robustness of this formulation across languages and entity types and characterised its accuracy and computational trade-ofs relative to candidate-based supervised and hybrid projection approaches.

## 3 Materials and Methods

We evaluated three projection strategies: supervised machine learning (ML), hybrid ML–LLM correction, and direct LLM-based projection. All were evaluated on the same oficial test set using Strict F1 and character-overlap F1. Strict evaluation required exact agreement in entity type, and start and end ofsets; unmatched predictions and gold entities were counted as false positives and false negatives, respectively. Precision, recall, and F1 were computed as $P = T P / ( T P + F P )$ , R = $T P / ( T P + F N )$ , and $F _ { 1 } = 2 \times P \times R / ( P + R )$ . To quantify partial boundary agreement, overlap similarity between a gold span g and prediction p was defined as $\phi ( g , p ) = 2 \times | g \cap p | / ( | g | + | p | )$ . Character precision (P<sub>c</sub>) and recall (R<sub>c</sub>) were obtained from the best overlaps, with $F 1 _ { c } = 2 \times P _ { c } \times R _ { c } / ( P _ { c } + R _ { c } )$ when $P _ { c } + R _ { c } > 0$ , and 0 otherwise.

The gold-standard test annotations were hidden from participants; predictions were submitted to the oficial MultiClinAI evaluation server<sup>1</sup>, which returned the corresponding evaluation scores.

## 3.1 Data

The dataset consists of parallel clinical documents released for the MultiClinCorpus shared task [16, 5]. Spanish source documents are paired with translations into six target languages: English, Czech, Italian, Dutch, Romanian, and Swedish. The annotations cover three entity types: Disease, Procedure, and Symptom. The corpus is divided into independent training and test partitions, summarised in Table 1.

The complete test partition contains 3,260 parallel documents, with between 59,552 and 81,510 annotations depending on the target language and entity type. Oficial evaluation was performed on the gold-standard subset reported in Table 1. All test results reported in this work refer exclusively to this evaluated subset.

To investigate complementary approaches to cross-lingual clinical entity projection, we evaluated three strategies that difer in the level at which the target span is identified and in the role assigned to the LLM. Figure 1 provides an overview of the task to be performed by the three pipelines, which are described in detail in the following subsections.

## 3.2 Method 1: Window-based Projection with Machine Learning

We formulated cross-lingual clinical entity projection as a binary classification problem over candidate spans in the target document. Given an annotated source entity and its parallel target document, candidate windows with lengths close to the source entity were generated. During training, the aligned target entity constituted the positive instance, while nearby and sampled windows constituted negative instances (Figure 2).

Candidate generation combined local hard negatives surrounding the aligned target entity with soft negatives sampled from the target document. Each source entity–candidate pair was represented by 20 features covering length, positional differences, token and character similarity, word and character n-gram overlap, term frequency– inverse document frequency (TF–IDF) cosine similarity, Levenshtein and semantic similarity, Jensen– Shannon and Hellinger distances, BLEU, and ME-TEOR [27, 20, 1]. Absolute character positions, document identifiers, and raw text were excluded from the feature matrix; class labels were used only as supervision.

For each source–target language pair and entity type, source mentions were partitioned into training (80%) and validation (20%) subsets. All candidate windows derived from the same mention were assigned to the same subset to prevent information leakage. Each classifier was trained on the training subset, with its decision threshold selected by maximising validation F1.

We compared seven classifiers, sharing training and evaluation protocol: Random Forest, Extra Trees, Histogram Gradient Boosting, classbalanced logistic regression, a support vector machine (SVM) with a radial basis function (RBF) kernel, a multilayer perceptron, and XGBoost [2, 10, 8, 3, 4, 15]. Random Forest and Extra Trees used 500 trees with a minimum leaf size of two; Histogram Gradient Boosting used depth 6, a 0.05 learning rate, and 300 iterations; and the RBF SVM used standardised features with $C = 1 . 0 .$ The multilayer perceptron used 128- and 64-unit hidden layers, ReLU activations, Adam optimisation, and early stopping. XGBoost used 500 trees, depth 4, a 0.05 learning rate, and row and column subsampling of 0.9.

## 3.3 Method 2: Hybrid ML–LLM Refinement

The hybrid strategy used the ML projection as input and applied an LLM-based refinement step only to potentially incorrect projections. Figure 3 summarises this selective correction pipeline, including LLM-based assessment, span correction, and grounding of the resulting expression in the target document. For each target language and entity type, the classifier achieving the highest Strict F1 returned by the oficial MultiClinAI evaluation 2 was selected as the input to the LLM correction stage. Gold-standard test annotations were not accessible to the participants.

Table 1: Training and gold-standard test data. Training values indicate the number of annotated entities; test values indicate the number of annotated entities used for gold-standard evaluation. Each target language contains 1,258 training documents.
<table><tr><td rowspan="2">Target language</td><td colspan="3">Training partition</td><td colspan="3">Gold-standard test subset</td></tr><tr><td>Disease</td><td>Procedure</td><td>Symptom</td><td>Disease</td><td>Procedure</td><td>Symptom</td></tr><tr><td>English</td><td>25,118</td><td>26,733</td><td>27,465</td><td>2,567</td><td>3,568</td><td>3,080</td></tr><tr><td>Czech</td><td>25,793</td><td>27,501</td><td>27,806</td><td>2,560</td><td>3,603</td><td>3,094</td></tr><tr><td>Italian</td><td>26,159</td><td>27,394</td><td>27,929</td><td>2,569</td><td>3,553</td><td>3,087</td></tr><tr><td>Dutch</td><td>25,733</td><td>27,445</td><td>27,675</td><td>2,584</td><td>3,654</td><td>3,096</td></tr><tr><td>Romanian</td><td>25,561</td><td>27,313</td><td>27,015</td><td>2,576</td><td>3,572</td><td>3,080</td></tr><tr><td>Swedish</td><td>25,580</td><td>27,079</td><td>27,531</td><td>2,555</td><td>3,587</td><td>3,095</td></tr></table>

![](images/5b622c3e4a7887892d098c438700db20c23cb2b66458ee9fb15ad16e6f05ac7a.jpg)  
Figure 1: Overview of the main clinical entity cross-lingual projection task.

![](images/522f9f17503ab3c5fa17f13198612ea8961cd53235f0c8c390969d711c4a38fa.jpg)  
Figure 2: Overview of the window-based machinelearning approach for cross-lingual clinical entity projection. Candidate spans are generated in the target document, represented through surface, positional, structural, and semantic features, and ranked by a supervised binary classifier to select the final projected span.

Following the prompt-defined decision criteria, the LLM classified each projection as an exact match, mid match, or no match. Exact matches were retained, whereas mid match and no match cases were refined using the complete target document and afected projections (see Supplementary Appendix A for prompts). The proposed expression was grounded to character ofsets by exact string matching. Unique occurrences were accepted directly; when multiple matches existed, the occurrence with the start ofset closest to the corresponding Spanish annotation was selected, exploiting the approximate preservation of document structure across translations. Unresolved suggestions were discarded.

![](images/247af7e0b828d410b0f21a1eb49a1daa28d7a51e94b02f7294cef0541195c75c.jpg)  
Figure 3: Overview of the hybrid ML–LLM projection approach. The best machine-learning projection is first assessed by an LLM; exact matches are retained, whereas uncertain or incorrect projections are corrected and subsequently grounded in the original target document to obtain valid character ofsets.

## 3.4 Method 3: LLM Entity Projection

The third approach formulated annotation transfer as a constrained LLM-based span-projection task. Figure 4 summarises the pipeline from tagged Spanish source and unannotated target documents to generation, validation, and character-ofset reconstruction. The model inserted DISEASE, SYMPTOM, and PROCEDURE tags into the target text while preserving every original character. Source tags were normalised to this label set, and the source annotations determined which entity types were projected for each document pair.

For each document pair, the model received the tagged Spanish source and untagged target document. The prompt (Supplementary Appendix A) specified the active labels and required the model to return only the tagged target text, prohibiting translation, paraphrasing, commentary, or any modification beyond XML tag insertion. Tag counts for each active label were required to match the source annotations.

Generation and validation were treated separately. Outputs were checked for balanced tags, matching source–target entity counts, and preservation of the target text after tag removal, using canonicalised line endings and HTML entities. Invalid outputs were retained as .invalid.txt sidecars for auditability, while any recoverable mentions resolvable to valid target spans remained included in the quantitative evaluation.

Predictions were converted from inline XML to character ofsets by sequentially parsing the output, removing clinical tags, and tracking cumulative character position. This directly yielded start and end ofsets, including for repeated expressions, without separate occurrence selection. Strict evaluation required exact agreement in document, entity type, and ofsets, while character-overlap F1 captured partial boundary agreement.

## 3.5 Models and Inference Settings

We evaluated three models: gemma4:31b, qwen3.6:35b, and glm-5.2 [9, 22, 12]. The Gemma and Qwen models were served locally through an internal server, whereas GLM-5.2 was accessed through the Ollama cloud service. All models received the same source–target document pairs, prompt template, entity-label constraints, and post-processing procedure.

For the local runs, maximum generation length was set to num\_predict=4096 tokens, and the context window was set to num\_ctx=8192. Streaming was enabled, think was set to false, and keep\_alive was set to 30m. The connection timeout was set to 30 s and entity\_count\_max\_attempts was set to 5. Temperature was set to zero.

For the GLM-5.2 cloud runs, streaming was disabled and temperature was set to zero. The request timeout was set to 600 s, up to three API-level attempts were permitted per document, and a 0.2 s delay was applied between requests. No explicit num\_ctx, num\_predict, keep\_alive, or think parameter was supplied to the cloud endpoint; these settings therefore remained under the provider defaults.

![](images/99d06ae4bddea4540f44399daa7badcbc6132f2ab8d6225236cadbd62f07a774.jpg)  
Figure 4: Overview of the direct LLM-based entity projection approach. The tagged Spanish source document and its unannotated parallel translation are provided to the LLM under constrained generation instructions. The resulting tagged target text is validated and converted into character-ofset entity spans for evaluation.

## 4 Results

Results compare candidate-based projection with generative refinement, direct document-level projection across languages and entity types, exactboundary performance against previous methods, and the associated computational trade-ofs. Previous state-of-the-art results are used as an external reference for assessing the magnitude and consistency of the observed improvements.

## 4.1 Candidate-based Projection and the Efect of Generative Refinement

Candidate-based projection showed substantial variation across languages and entity types (Table 2). Tree-based ensembles provided the strongest ML baselines, yielding the highest Strict F1 in 17 of the 18 language–entity settings: Extra Trees was selected in 15 cases, Random Forest in two, and logistic regression in one. Performance was highest for English and Italian and lowest for Czech, while Disease consistently achieved higher Strict F1 than Procedure and Symptom. Mean Strict F1 for Symptom was 0.5122, compared with 0.5706 for Procedure. These diferences indicate that candidate-based projection is sensitive to the linguistic and boundary characteristics of the target expression despite the inclusion of multilingual semantic features.

Table 2: Best ML model and LLM refinement by target language and entity type. The ML model is selected by the highest test Strict F1 among the evaluated classifiers. Bold indicates the larger Strict F1 within each ML–LLM pair. Dashes indicate that LLM refinement was not completed.
<table><tr><td rowspan="2">Lang</td><td rowspan="2">Entity</td><td rowspan="2">ML Model</td><td rowspan="2">System</td><td colspan="6">Strict</td><td rowspan="2">Char F1</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>TP</td><td>FP</td><td>FN</td></tr><tr><td rowspan="6">EN</td><td rowspan="3">Disease</td><td rowspan="3">Extra Trees</td><td>ML model</td><td>0.7222</td><td>0.7242</td><td>0.7232</td><td>1859</td><td>715</td><td>708</td><td>0.8529</td></tr><tr><td>LLM Ref.</td><td>0.8899</td><td>0.8878</td><td>0.8888</td><td>2279</td><td>282</td><td>288</td><td>0.9522</td></tr><tr><td>ML model</td><td>0.6561</td><td>0.6508</td><td>0.6534</td><td>2322</td><td>1217</td><td>1246</td><td>0.7966</td></tr><tr><td rowspan="2">Procedure</td><td rowspan="2">Random Forest Extra Trees</td><td>LLM Ref.</td><td>0.8343</td><td>0.8299</td><td>0.8321</td><td>2961</td><td>588</td><td>607</td><td>0.9259</td></tr><tr><td>ML model</td><td>0.6367</td><td>0.6321</td><td>0.6344</td><td>1947</td><td>1111</td><td>1133</td><td>0.8007</td></tr><tr><td rowspan="4">Disease</td><td rowspan="2">Extra Trees</td><td>LLM Ref.</td><td>0.8759</td><td>0.8734</td><td>0.8747</td><td>2690</td><td>381</td><td>390</td><td>0.9561</td></tr><tr><td>ML model</td><td>0.4480</td><td>0.4461</td><td>0.4471</td><td>1142</td><td>1407</td><td>1418</td><td>0.5984</td></tr><tr><td rowspan="3">Procedure</td><td rowspan="3">Extra Trees</td><td>LLM Ref.</td><td>0.7982</td><td>0.7836</td><td>0.7909</td><td>2006</td><td>507</td><td>554</td><td>0.8912</td></tr><tr><td>ML model</td><td>0.4539</td><td>0.4294</td><td>0.4413</td><td>1547</td><td>1861</td><td>2056</td><td>0.5749</td></tr><tr><td>LLM Ref.</td><td>0.7587</td><td>0.7286</td><td>0.7433</td><td>2625</td><td>835</td><td>978</td><td>0.8477</td></tr><tr><td rowspan="4"></td><td rowspan="2">Symptom</td><td rowspan="2">Extra Trees</td><td>ML model LLM Ref.</td><td>0.3736 0.7521</td><td>0.3681 0.7414</td><td>0.3708 0.7467</td><td>1139 2294</td><td>1910 756</td><td>1955 800</td><td>0.5605 0.8754</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Disease</td><td rowspan="2">Extra Trees</td><td>ML model</td><td>0.7086 0.8993</td><td>0.7088 0.8898</td><td>0.7087 0.8945</td><td>1821 2286</td><td>749 256</td><td>748</td><td>0.7571</td></tr><tr><td>LLM Ref. ML model</td><td>0.6630</td><td>0.6583</td><td>0.6606</td><td>2339</td><td>1189</td><td>283 1214</td><td>0.9503 0.7271</td></tr><tr><td rowspan="3"></td><td rowspan="2">Procedure</td><td rowspan="2">Extra Trees</td><td>LLM Ref.</td><td>0.8573</td><td>0.8486</td><td>0.8529</td><td>3015</td><td>502</td><td>538</td><td>0.9249</td></tr><tr><td>ML model</td><td>0.6310</td><td>0.6093</td><td>0.6200</td><td>1881</td><td>1100</td><td>1206</td><td>0.7980</td></tr><tr><td rowspan="2">Symptom</td><td rowspan="2">Logistic Regression</td><td>LLM Ref.</td><td>0.8259</td><td>0.8024</td><td>0.8140</td><td>2477</td><td>522</td><td>610</td><td>0.9319</td></tr><tr><td>ML model</td><td>0.5748</td><td>0.5712</td><td>0.5730</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="7">NL</td><td rowspan="2">Disease Procedure</td><td rowspan="2">Extra Trees Extra Trees</td><td>LLM Ref.</td><td>0.8008</td><td>0.7810</td><td>0.7908</td><td>1476 2018</td><td>1092 502</td><td>1108 566</td><td>0.6671 0.8853</td></tr><tr><td>ML model</td><td>0.5618</td><td>0.5290</td><td>0.5449</td><td>1933</td><td>1508</td><td>1721</td><td>0.6431</td></tr><tr><td rowspan="3">Symptom</td><td rowspan="3">Extra Trees</td><td>LLM Ref.</td><td>0.7681</td><td>0.7351</td><td>0.7512</td><td>2686</td><td>811</td><td>968</td><td>0.8592</td></tr><tr><td>ML model</td><td></td><td>0.4761</td><td>0.4800</td><td>1473</td><td>1571</td><td>1621</td><td>0.6194</td></tr><tr><td>LLM Ref.</td><td>0.4839 0.7567</td><td>0.7489</td><td></td><td>2317</td><td>745</td><td>777</td><td>0.8911</td></tr><tr><td rowspan="2">Disease</td><td rowspan="2">Extra Trees</td><td></td><td></td><td></td><td>0.7528</td><td></td><td></td><td></td><td></td></tr><tr><td>ML model</td><td>0.6527</td><td>0.6522</td><td>0.6524</td><td>1680</td><td>894</td><td>896</td><td>0.7313</td></tr><tr><td rowspan="6">RO</td><td rowspan="2">Procedure</td><td rowspan="2">Extra Trees</td><td>LLM Ref.</td><td>0.8659</td><td>0.8521</td><td>0.8589</td><td>2195</td><td>340</td><td>381</td><td>0.9375</td></tr><tr><td>ML model</td><td>0.6119</td><td>0.5971</td><td>0.6044</td><td>2133</td><td>1353</td><td>1439</td><td>0.7072</td></tr><tr><td rowspan="2">Symptom</td><td rowspan="2">Extra Trees</td><td>LLM Ref.</td><td>0.8472</td><td>0.8287</td><td>0.8378</td><td>2960</td><td>534</td><td>612</td><td>0.9274</td></tr><tr><td>ML model</td><td>0.5084</td><td>0.5026</td><td>0.5055</td><td>1548</td><td>1497</td><td>1532</td><td>0.6155</td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td>LLM Ref.</td><td>0.8021</td><td>0.7987</td><td>0.8004</td><td>2460</td><td>607</td><td>620</td><td>0.9150</td></tr><tr><td>ML model</td><td>0.5439</td><td>0.5436</td><td>0.5437</td><td>1389</td><td>1165</td><td>1166</td><td>0.6635</td></tr><tr><td rowspan="6">SV</td><td rowspan="2">Procedure</td><td rowspan="2">Random Forest Extra Trees</td><td>LLM Ref. ML model</td><td>0.8002 0.5343</td><td>0.7933 0.5040</td><td>0.7968 0.5187</td><td>2027 1808</td><td>506 1576</td><td>528 1779</td><td>0.8918 0.6362</td></tr></table>

Generative refinement substantially reduced this dependency on the initial ML decision. In every completed comparison, LLM refinement increased Strict F1, with a mean absolute gain of 0.2462, while reducing both false-positive and falsenegative projections. Character-overlap F1 increased from a mean of 0.6878 for the ML baselines to 0.9087 after refinement, corresponding to a mean absolute gain of 0.2209. The larger characteroverlap scores relative to Strict F1 indicate that a substantial proportion of the residual errors involved inaccurate span boundaries rather than complete failure to localise the target entity. Nevertheless, because refinement operates on an upstream candidate-based projection, its ability to recover an entity remains dependent on the information supplied by the preceding projection stage.

## 4.2 Robustness of Document-level Projection across Languages and Entity Types

Direct document-level projection achieved high Strict F1 across all six target languages and three entity types (Table 3). GLM 5.2 obtained the highest overall mean Strict F1 (0.9201), followed closely by Gemma4:31B (0.9133), while Qwen3.6:35B obtained 0.8300; GLM 5.2 led in 13 of

Table 3: Strict span-level results and character-overlap F1 by target language and entity type. Bold indicates the highest Strict F1 among the available models within each pair.
<table><tr><td rowspan=1 colspan=9>LangEntity    Model                        Strict                  CharP    R    F1  TPFPFN    F1</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=8>Gemma4:31B0.95900.9657 0.96232479106 880.9868</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=7>Disease   Qwen3.6:35B0.89150.8703 0.88072234272333</td><td rowspan=1 colspan=1>0.9493</td></tr><tr><td rowspan=1 colspan=7>GLM 5.2    0.96840.96770.96802484 81 83</td><td rowspan=1 colspan=1>0.9851</td></tr><tr><td rowspan=2 colspan=1>EN</td><td rowspan=1 colspan=7>Gemma4:31B0.92940.9302 0.92983319252249</td><td rowspan=1 colspan=1>0.9780</td></tr><tr><td rowspan=1 colspan=7>ProcedureQwen3.6:35B0.89470.8307 0.86152964349604</td><td rowspan=1 colspan=1>0.9303</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>GLM 5.2    0.94010.93720.9387</td><td rowspan=1 colspan=3>3344213224</td><td rowspan=1 colspan=1>0.9786</td></tr><tr><td rowspan=3 colspan=1></td><td rowspan=1 colspan=8>Gemma4:31B0.94450.9445 0.944529091711710.9826</td></tr><tr><td rowspan=1 colspan=8>Symptom Qwen3.6:35B0.86830.83900.853425843924960.9396</td></tr><tr><td rowspan=1 colspan=8>GLM 5.2    0.95990.94710.953429171221630.9795</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=4>Gemma4:31B0.91350.9199 0.9167</td><td rowspan=1 colspan=4>23552232050.9719</td></tr><tr><td rowspan=1 colspan=4>Disease   Qwen3.6:35B0.85710.82700.8417</td><td rowspan=1 colspan=3>2117353443</td><td rowspan=1 colspan=1>0.9275</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>GLM 5.2    0.92500.93050.9278</td><td rowspan=1 colspan=3>2382193178</td><td rowspan=1 colspan=1>0.9740</td></tr><tr><td rowspan=2 colspan=1>CS</td><td rowspan=1 colspan=4>Gemma4:31B0.88960.8837 0.8867</td><td rowspan=1 colspan=3>3184395419</td><td rowspan=1 colspan=1>0.9645</td></tr><tr><td rowspan=1 colspan=4>ProcedureQwen3.6:35B0.86270.7880 0.8236</td><td rowspan=1 colspan=1>2839</td><td rowspan=1 colspan=2>452764</td><td rowspan=1 colspan=1>0.9135</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=4>GLM 5.2    0.89790.89120.8946</td><td rowspan=1 colspan=1>3211</td><td rowspan=1 colspan=2>365392</td><td rowspan=1 colspan=1>0.9646</td></tr><tr><td rowspan=1 colspan=2>Gemma4:31B0.9038</td><td rowspan=1 colspan=2>0.8992 0.9015</td><td rowspan=1 colspan=1>2782</td><td rowspan=1 colspan=1>296</td><td rowspan=1 colspan=1>312</td><td rowspan=1 colspan=1>0.9710</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=2>Symptom Qwen3.6:35B0.8427</td><td rowspan=1 colspan=1>0.7999</td><td rowspan=1 colspan=1>0.8208</td><td rowspan=1 colspan=1>2475</td><td rowspan=1 colspan=1>462</td><td rowspan=1 colspan=1>619</td><td rowspan=1 colspan=1>0.9283</td></tr><tr><td rowspan=1 colspan=2>GLM 5.2    0.9196</td><td rowspan=1 colspan=1>0.9131</td><td rowspan=1 colspan=1>0.9163</td><td rowspan=1 colspan=1>2825</td><td rowspan=1 colspan=1>247</td><td rowspan=1 colspan=1>269</td><td rowspan=1 colspan=1>0.9752</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Gemma4:31B0.9656</td><td rowspan=1 colspan=1>0.9720</td><td rowspan=1 colspan=1>0.9688</td><td rowspan=1 colspan=1>2497</td><td rowspan=1 colspan=1>89</td><td rowspan=1 colspan=1>72</td><td rowspan=1 colspan=1>0.9874</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=2>Disease   Qwen3.6:35B0.9025</td><td rowspan=1 colspan=1>0.8505</td><td rowspan=1 colspan=1>0.8758</td><td rowspan=1 colspan=1>2185</td><td rowspan=1 colspan=1>236</td><td rowspan=1 colspan=1>384</td><td rowspan=1 colspan=1>0.9397</td></tr><tr><td rowspan=1 colspan=2>GLM 5.2    0.9670</td><td rowspan=1 colspan=2>0.9685 0.9677</td><td rowspan=1 colspan=2>2488 85</td><td rowspan=1 colspan=1>81</td><td rowspan=1 colspan=1>0.9851</td></tr><tr><td rowspan=2 colspan=1>IT</td><td rowspan=1 colspan=4>Gemma4:31B0.94510.95520.9502</td><td rowspan=1 colspan=2>3394197</td><td rowspan=1 colspan=1>159</td><td rowspan=1 colspan=1>0.9833</td></tr><tr><td rowspan=1 colspan=4>ProcedureQwen3.6:35B0.92080.8283 0.8721</td><td rowspan=1 colspan=2>2943253</td><td rowspan=1 colspan=1>610</td><td rowspan=1 colspan=1>0.9214</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>GLM 5.2    0.9430</td><td rowspan=1 colspan=2>0.9538 0.9484</td><td rowspan=1 colspan=3>3389205164</td><td rowspan=1 colspan=1>0.9851</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Gemma4:31B0.9203</td><td rowspan=1 colspan=2>0.9206 0.9205</td><td rowspan=1 colspan=3>2842246245</td><td rowspan=1 colspan=1>0.9860</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Symptom Qwen3.6:35B0.8044</td><td rowspan=1 colspan=2>0.77520.7895</td><td rowspan=1 colspan=3>2393582694</td><td rowspan=1 colspan=1>0.9354</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>GLM 5.2    0.9391</td><td rowspan=1 colspan=2>0.93970.9394</td><td rowspan=1 colspan=2>2901188</td><td rowspan=1 colspan=1>186</td><td rowspan=1 colspan=1>0.9900</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Gemma4:31B0.8787</td><td rowspan=1 colspan=2>0.8777 0.8782</td><td rowspan=1 colspan=1>2268</td><td rowspan=1 colspan=1>313</td><td rowspan=1 colspan=1>316</td><td rowspan=1 colspan=1>0.9417</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Disease   Qwen3.6:35B0.8282</td><td rowspan=1 colspan=1>0.7910</td><td rowspan=1 colspan=1>0.8092</td><td rowspan=1 colspan=1>2044</td><td rowspan=1 colspan=1>424</td><td rowspan=1 colspan=1>540</td><td rowspan=1 colspan=1>0.9018</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>GLM 5.2    0.8949</td><td rowspan=1 colspan=1>0.8932</td><td rowspan=1 colspan=1>0.8941</td><td rowspan=1 colspan=1>2308</td><td rowspan=1 colspan=1>271</td><td rowspan=1 colspan=1>276</td><td rowspan=1 colspan=1>0.9463</td></tr><tr><td rowspan=3 colspan=1>NL</td><td rowspan=1 colspan=1>Gemma4:31B</td><td rowspan=1 colspan=1>0.8579</td><td rowspan=1 colspan=1>0.8407</td><td rowspan=1 colspan=1>0.8492</td><td rowspan=1 colspan=1>3072</td><td rowspan=1 colspan=1>509</td><td rowspan=1 colspan=1>582</td><td rowspan=1 colspan=1>0.9340</td></tr><tr><td rowspan=1 colspan=1>ProcedureQwen3.6:35B</td><td rowspan=1 colspan=1>0.8216</td><td rowspan=1 colspan=1>0.7411</td><td rowspan=1 colspan=1>0.7793</td><td rowspan=1 colspan=1>2708</td><td rowspan=1 colspan=1>588</td><td rowspan=1 colspan=1>946</td><td rowspan=1 colspan=1>0.8805</td></tr><tr><td rowspan=1 colspan=2>GLM 5.2    0.8640</td><td rowspan=1 colspan=2>0.84870.8563</td><td rowspan=1 colspan=1>3101</td><td rowspan=1 colspan=1>488</td><td rowspan=1 colspan=1>553</td><td rowspan=1 colspan=1>0.9371</td></tr><tr><td rowspan=3 colspan=1></td><td rowspan=1 colspan=2>Gemma4:31B0.8343</td><td rowspan=1 colspan=2>0.83110.8327</td><td rowspan=1 colspan=3>2573511523</td><td rowspan=1 colspan=1>0.9335</td></tr><tr><td rowspan=1 colspan=2>Symptom Qwen3.6:35B0.7664</td><td rowspan=1 colspan=2>0.72580.7455</td><td rowspan=1 colspan=3>2247685849</td><td rowspan=1 colspan=1>0.8862</td></tr><tr><td rowspan=1 colspan=4>GLM 5.2    0.83150.8224 0.8269</td><td rowspan=1 colspan=3>2546516550</td><td rowspan=1 colspan=1>0.9318</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=8>Gemma4:31B0.95320.9557 0.954424621211140.9854</td></tr><tr><td rowspan=1 colspan=8>Disease   Qwen3.6:35B0.90190.84980.875121892383870.9406</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=7>GLM 5.2    0.95600.96040.95822474114102</td><td rowspan=1 colspan=1>0.9877</td></tr><tr><td rowspan=5 colspan=1>RO</td><td rowspan=2 colspan=6>Gemma4:31B0.94200.9418 0.94193364207ProcedureQwen3.6:35B0.91520.8099 0.85932893268</td><td rowspan=1 colspan=1>208</td><td rowspan=1 colspan=1>0.9827</td></tr><tr><td rowspan=1 colspan=2>2893 268</td><td rowspan=1 colspan=1>679</td><td rowspan=1 colspan=1>0.9120</td></tr><tr><td rowspan=1 colspan=4>GLM 5.2    0.94480.94900.9469</td><td rowspan=1 colspan=3>3390198182</td><td rowspan=1 colspan=1>0.9859</td></tr><tr><td rowspan=1 colspan=4>Gemma4:31B0.92400.9234 0.9237</td><td rowspan=1 colspan=3>2844234236</td><td rowspan=1 colspan=1>0.9831</td></tr><tr><td rowspan=1 colspan=8>Symptom Qwen3.6:35B0.83240.80160.8167GLM 5.2    0.94870.94840.948529211581590.9868</td><td rowspan=1 colspan=1>2469 497 611</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=8>Gemma4:31B0.89130.9014 0.896323032812520.9626</td></tr><tr><td rowspan=1 colspan=8>Disease   Qwen3.6:35B0.82830.80820.818120654284900.9228GLM 5.2    0.90310.91510.909023382512170.9676</td></tr><tr><td rowspan=3 colspan=1>SV</td><td rowspan=1 colspan=8>Gemma4:31B0.89350.89380.893732063823810.9697ProcedureQwen3.6:35B0.85390.7775 0.814027894777980.9124GLM 5.2    0.89480.8918 0.893331993763880.9694</td></tr><tr><td rowspan=1 colspan=8>Gemma4:31B0.89030.88660.888527443383510.9730Symptom Qwen3.6:35B0.82620.7819 0.803524205096750.9206</td></tr><tr><td rowspan=1 colspan=8>GLM 5.2    0.88010.8682 0.874126873664080.9650</td></tr></table>

18 language–entity combinations and Gemma4:31B in five. English, Italian, and Romanian produced the strongest overall results, whereas Dutch was the most challenging target language. Disease was the highest-performing entity type for all three models, followed by Procedure and Symptom.

Character-overlap F1 was consistently higher than Strict F1, with macro-averages of 0.9719 for GLM 5.2, 0.9710 for Gemma4:31B, and 0.9222 for Qwen3.6:35B. This indicates that many residual errors involved entity-boundary mismatches rather than localisation of an unrelated span.

## 4.3 Exact-boundary Recovery and Improvement over Previous Projection Methods

The gains observed with document-level projection were preserved under the strictest evaluation criterion, which requires exact agreement of entity type and character boundaries. As shown in Table 4, the best document-level configuration exceeded the strongest previously reported Multi-ClinCorpus method, CA26AM+MCAI [23], in all 18 language–entity combinations. Mean Strict F1 increased from 0.8348 to 0.9214, an absolute improvement of 0.0866. Gains were observed for every language and entity type and ranged from +0.0602 for Romanian Disease to +0.1512 for Italian Procedure. The consistency of these improvements under exact-span evaluation is particularly relevant for corpus construction, because successful projection requires not only localisation of the corresponding clinical concept but also recovery of the precise textual boundaries needed to generate reproducible character-level annotations.

Table 4: Exact-span projection performance relative to the strongest previously reported MultiClinCorpus method. Results are reported using Strict F1, requiring exact agreement of entity type and character ofsets. ∆F1 denotes the absolute improvement over the previous best method.
<table><tr><td>Lang</td><td>Entity</td><td>Previous method</td><td>Previous F1</td><td>Best model</td><td>Strict F1</td><td>∆F1</td></tr><tr><td rowspan="3">EN</td><td>Disease</td><td>CA26AM+MCAI</td><td>0.8960</td><td>GLM 5.2</td><td>0.9680</td><td>+0.0720</td></tr><tr><td>Procedure</td><td>CA26AM+MCAI</td><td>0.8410</td><td>GLM 5.2</td><td>0.9387</td><td>+0.0977</td></tr><tr><td>Symptom</td><td>CA26AM+MCAI</td><td>0.8790</td><td>GLM 5.2</td><td>0.9534</td><td>+0.0744</td></tr><tr><td rowspan="3">CS</td><td>Disease</td><td>CA26AM+MCAI</td><td>0.8510</td><td>GLM 5.2</td><td>0.9278</td><td>+0.0768</td></tr><tr><td>Procedure</td><td>CA26AM+MCAI</td><td>0.8190</td><td>GLM 5.2</td><td>0.8946</td><td>+0.0756</td></tr><tr><td>Symptom</td><td>CA26AM+MCAI</td><td>0.8110</td><td>GLM 5.2</td><td>0.9163</td><td>+0.1053</td></tr><tr><td rowspan="3">IT</td><td>Disease</td><td>CA26AM+MCAI</td><td>0.8820</td><td>Gemma4:31B</td><td>0.9688</td><td>+0.0868</td></tr><tr><td>Procedure</td><td>CA26AM+MCAI</td><td>0.7990</td><td>Gemma4:31B</td><td>0.9502</td><td>+0.1512</td></tr><tr><td>Symptom</td><td>CA26AM+MCAI</td><td>0.8310</td><td>GLM 5.2</td><td>0.9394</td><td>+0.1084</td></tr><tr><td rowspan="3">NL</td><td>Disease</td><td>CA26AM+MCAI</td><td>0.8200</td><td>GLM 5.2</td><td>0.8941</td><td>+0.0741</td></tr><tr><td>Procedure</td><td>CA26AM+MCAI</td><td>0.7700</td><td>GLM 5.2</td><td>0.8563</td><td>+0.0863</td></tr><tr><td>Symptom</td><td>CA26AM+MCAI</td><td>0.7680</td><td>Gemma4:31B</td><td>0.8327</td><td>+0.0647</td></tr><tr><td rowspan="3">RO</td><td>Disease</td><td>CA26AM+MCAI</td><td>0.8980</td><td>GLM 5.2</td><td>0.9582</td><td>+0.0602</td></tr><tr><td>Procedure</td><td>CA26AM+MCAI</td><td>0.8560</td><td>GLM 5.2</td><td>0.9469</td><td>+0.0909</td></tr><tr><td>Symptom</td><td>CA26AM+MCAI</td><td>0.8610</td><td>GLM 5.2</td><td>0.9485</td><td>+0.0875</td></tr><tr><td rowspan="3">SV</td><td>Disease</td><td>CA26AM+MCAI</td><td>0.8280</td><td>GLM 5.2</td><td>0.9090</td><td>+0.0810</td></tr><tr><td>Procedure</td><td>CA26AM+MCAI</td><td>0.8060</td><td>Gemma4:31B</td><td>0.8937</td><td>+0.0877</td></tr><tr><td>Symptom</td><td>CA26AM+MCAI</td><td>0.8110</td><td>Gemma4:31B</td><td>0.8885</td><td>+0.0775</td></tr><tr><td colspan="2">Average</td><td></td><td>0.8348</td><td></td><td>0.9214</td><td>+0.0866</td></tr></table>

## 4.4 Accuracy-Eficiency Trade-ofs

The projection paradigms exhibited substantially diferent computational profiles. GLM 5.2 required an average of 6.54 s per document through cloud inference, whereas locally deployed Gemma4:31B required 27.01 s per document, with a mean generation rate of 33.32 tokens/s. In comparison, our local open-source implementation of the ClinicalAligner<sup>3</sup> approach required approximately 0.1 s per document after training.

ClinicalAligner additionally required a dedicated training stage, estimated at approximately 2–4 h on an RTX 3090 in our implementation, whereas the LLM approaches can be applied directly without task-specific training. Under these measurements, the approximate point at which the initial alignment-model training cost is amortised is 1,118– 2,236 documents relative to GLM 5.2 and 268–535 documents relative to Gemma4:31B. These values should be interpreted as implementation- and hardware-dependent estimates rather than intrinsic properties of the models.

## 5 Discussion

Our findings indicate that cross-lingual clinical annotation projection can be efectively reformulated as a constrained document-level generation prob lem, provided that generative inference is coupled with deterministic text-preservation validation and character-level grounding. This formulation substantially outperformed the current state of the art for cross-lingual clinical annotation projection. Rather than relying on explicit alignment or candidate generation and ranking, the LLM operates on the complete parallel documents and directly reconstructs the annotated target text. This formulation appears particularly well suited to clinical projection, where translation often preserves the underlying clinical information while modifying lexical form, word order, or entity boundaries.

Gemma4:31B is particularly relevant because, like CA26AM+MCAI, it can be deployed entirely locally. Its consistent advantage over the previous state of the art despite requiring no specialised alignment architecture indicates that documentlevel generation can improve both semantic localisation and exact span reconstruction. The strict improvement ranged from +0.0564 to +0.1512 F1, with the largest gains observed for Procedure entities. These gains indicate not only successful semantic localisation but also more accurate reconstruction of the complete annotated span.

The results across languages also show that projection dificulty cannot be explained solely by the expected availability of language resources. English, Italian, and Romanian generally produce the strongest LLM results, whereas Dutch is the most challenging target language. Czech and Swedish, despite being comparatively less represented in many multilingual NLP resources, remain competitive under direct LLM projection. This suggests that document-level contextual reasoning can partially compensate for weaker lexical correspondence or lower representation during multilingual pretraining, while also indicating that linguistic resource availability alone does not determine projection performance.

A clearer linguistic pattern emerges for the ML approach. Among the non-English targets, Italian and Romanian benefit most from candidate-based projection, while performance decreases for more distant languages and is particularly limited for Czech. This behaviour is consistent with the feature representation used by the classifier, which combines lexical overlap, edit similarity, character and word n-grams, positional information, and multilingual semantic similarity. Romance languages provide stronger surface and structural correspondence with Spanish, making the candidate-ranking problem easier. In contrast, direct LLM projection exhibits substantially less dependence on these explicit form-level similarities.

The results obtained with the hybrid strategy further suggest that the main advantage of LLMbased projection arises when the model can reason over the complete projection problem rather than being restricted to revising candidates produced by an upstream system. Candidate-based methods remain attractive because they substantially constrain the search space and reduce inference requirements, but their attainable performance is limited by candidate generation: if the correct target span is absent or poorly represented among the candidates, subsequent classification or generative refinement cannot fully recover it. Direct projection removes this dependency by locating and annotating the corresponding entity directly within the complete target document.

These findings are particularly relevant for multilingual corpus construction. A single LLM can be applied across languages and entity types without training separate projection classifiers or maintaining a specialised alignment model for each setting. In particular, locally deployable models make it possible to retain the complete projection workflow within institutional infrastructure, an important consideration in clinical NLP settings in which governance, privacy, or data-sharing restrictions may preclude externally hosted inference services.

Nevertheless, direct LLM projection introduces failure modes that difer from those of conventional alignment systems. Generative models may alter the target text, omit annotations, duplicate entities, or produce malformed tags. Deterministic validation and ofset reconstruction are therefore not merely post-processing conveniences but integral components of the proposed formulation: they ensure that accepted projections preserve the original target text and can be converted into reproducible character-level annotations before incorporation into the resulting corpus.

The diferent projection paradigms also involve a clear accuracy–eficiency trade-of. Once training has been amortised, alignment-based projection remains considerably more computationally eficient for large or repeatedly processed collections. Direct LLM projection instead exchanges higher perdocument inference cost for substantially higher exact-span accuracy, the absence of task-specific training, and a simpler projection pipeline. This trade-of is particularly relevant for local deployment, where both LLM-based and alignment-based approaches can operate without external inference services and the preferred strategy can therefore be selected according to corpus size, computational resources, and required annotation quality.

## 5.1 Limitations

This study has several limitations. First, all experiments were conducted within the MultiClin-Corpus benchmark, using Spanish as the single source language and six European target languages. Although these languages include Romance, Germanic, and Slavic families, the extent to which the observed robustness generalises to other source languages, more typologically distant target languages, or independently authored multilingual clinical documents remains unknown. The evaluation was also restricted to three entity types—Disease, Symptom, and Procedure—and therefore does not establish performance for other clinical concepts or more complex annotation structures. External validation on additional clinical corpora and language pairs will be necessary to assess the generalisability of the proposed formulation.

Second, evaluation relied on the oficial goldstandard span annotations and quantified exact and partial boundary agreement, but we did not conduct an additional bilingual or clinical-expert error analysis of the residual projections. Deterministic validation verifies tag structure, entity counts, text preservation, and character-ofset reconstruction, but cannot by itself establish the semantic correctness of a structurally valid projected span. Similarly, the present study evaluates annotation projection directly rather than the downstream utility of the resulting projected corpora; future work should determine whether models trained on automatically projected annotations retain performance when applied to independently annotated, nativelanguage clinical text. Finally, only three LLMs and specific inference configurations were evaluated, and computational measurements depend on model implementation, serving infrastructure, and hardware. Accordingly, absolute eficiency estimates and the relative advantage of particular models should not be interpreted as invariant across deployment environments or future model versions.

## 6 Conclusion

This study demonstrates that cross-lingual clinical annotation projection can be formulated as a constrained, text-preserving document-level generative task capable of producing accurate and verifiable character-level annotations across multiple languages. Direct LLM projection consistently outperformed candidate-based supervised, hybrid, and previous state-of-the-art projection approaches across six target languages and three clinical entity types, while avoiding the need for explicit alignment, candidate generation, or task-specific training.

Across the evaluated corpus, the best LLM configuration for each language–entity combination produced 55,416 projected clinical mentions across six target languages and three entity types. By transferring expert annotations from a single source language while preserving exact textual grounding and reproducible character ofsets, the proposed formulation provides a practical mechanism for extending annotated clinical resources, particularly where manually annotated corpora and languagespecific NLP resources remain limited.

Although direct LLM projection entails higher inference costs than specialised alignment-based approaches, it combines strong exact-span accuracy with a simpler task-specific workflow and, when locally deployed, can operate entirely within institutional infrastructure. Coupled with deterministic validation and ofset reconstruction, this makes text-preserving document-level projection a practical and auditable strategy for multilingual clinical corpus construction, with the potential to reduce the amount of expert annotation required when extending existing resources across languages.

## Funding

This work was supported by the Ministerio de Ciencia e Innovación (MICINN) under project PID2024- 155334OB-I00. Funding for the open access charge was provided by Universidad de Málaga / CBUA.

## References

[1] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments.

In Proceedings of the ACL Workshop on Intrinsic and Extrinsic Evaluation Measures for MT and/or Summarization, pages 65–72, 2005.

[2] Leo Breiman. Random forests. Machine Learning, 45(1):5–32, 2001.

[3] Tianqi Chen and Carlos Guestrin. Xgboost: A scalable tree boosting system. In Proceedings of KDD, pages 785–794, 2016.

[4] Corinna Cortes and Vladimir Vapnik. Supportvector networks. Machine Learning, 20(3):273– 297, 1995.

[5] Fernando Gallego Donoso, Salvador Lima-Lopez, Judith Rosell, Eulàlia Farré-Maduel, and Martin Krallinger. The MultiClinAI shared task on multilingual clinical corpus construction and concept extraction: Systems, evaluation, and datasets. In Guillermo Lopez-Garcia and Graciela Gonzalez-Hernandez, editors, Proceedings of the 11th Social Media Mining for Health Research and Applications (SMM4H-HeaRD 2026) Workshop and Shared Tasks, pages 309–331, San Diego, United States, July 2026. Association for Computational Linguistics.

[6] Zi-Yi Dou and Graham Neubig. Word alignment by fine-tuning embeddings on parallel corpora. In Paola Merlo, Jorg Tiedemann, and Reut Tsarfaty, editors, Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 2112–2128, Online, April 2021. Association for Computational Linguistics.

[7] Maud Ehrmann, Marco Turchi, and Ralf Steinberger. Building a multilingual named entityannotated corpus using annotation projection. In Ruslan Mitkov and Galia Angelova, editors, Proceedings of the International Conference Recent Advances in Natural Language Processing 2011, pages 118–124, Hissar, Bulgaria, September 2011. Association for Computational Linguistics.

[8] Jerome H. Friedman. Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29(5):1189–1232, 2001.

[9] Gemma-Team. Gemma 4 technical report, 2026.

[10] Pierre Geurts, Damien Ernst, and Louis Wehenkel. Extremely randomized trees. Machine Learning, 63(1):3–42, 2006.

[11] Soumitra Ghosh, Begona Altuna, Saeed Farzi, Pietro Ferrazzi, Alberto Lavelli, Giulia Mezzanotte, Manuela Speranza, and Bernardo Magnini. Low-resource information extraction with the european clinical case corpus, 2025.

[12] GLM-5-Team. Glm-5: from vibe coding to agentic engineering, 2026.

[13] Léo Jacqmin, Gabriel Marzinotto, Justyna Gromada, Ewelina Szczekocka, Robert Kołodyński, and Géraldine Damnati. SpanAlign: Eficient sequence tagging annotation projection into translated data applied to cross-lingual opinion mining. In Wei Xu, Alan Ritter, Tim Baldwin, and Afshin Rahimi, editors, Proceedings of the Seventh Workshop on Noisy User-generated Text (W-NUT 2021), pages 238–248, Online, November 2021. Association for Computational Linguistics.

[14] Masoud Jalili Sabet, Philipp Dufter, François Yvon, and Hinrich Schütze. SimAlign: High quality word alignments without parallel training data using static and contextualized embeddings. In Trevor Cohn, Yulan He, and Yang Liu, editors, Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1627–1643, Online, November 2020. Association for Computational Linguistics.

[15] Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In Proceedings of ICLR, 2015.

[16] Salvador Lima López, Judith Rosell, Jan Rodríguez Miret, Fernando Gallego-Donoso, and Martin Krallinger. Multiclinai corpus: Multilingual clinical entity annotation projection and extraction, 2026.

[17] Bernardo Magnini, Begoña Altuna, Alberto Lavelli, Anne-Lyse Minard, Manuela Speranza, and Roberto Zanoli. European Clinical Case Corpus, pages 283–288. Springer International Publishing, Cham, 2023.

[18] Aurélie Névéol, Hercules Dalianis, Sumithra Velupillai, Guergana Savova, and Pierre Zweigenbaum. Clinical natural language processing in languages other than english: Opportunities and challenges. Journal of Biomedical Semantics, 9(1):12, March 2018.

[19] Jian Ni, Georgiana Dinu, and Radu Florian. Weakly supervised cross-lingual named entity recognition via efective annotation and representation projection. In Regina Barzilay and Min-Yen Kan, editors, Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1470–1480, Vancouver, Canada, July 2017. Association for Computational Linguistics.

[20] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of ACL, pages 311–318, 2002.

[21] Andrei Politov, Oleh Shkalikov, Rene Jäkel, and Michael Färber. Revisiting projectionbased data transfer for cross-lingual named entity recognition in low-resource languages. In Richard Johansson and Sara Stymne, editors, Proceedings of the Joint 25th Nordic Conference on Computational Linguistics and 11th Baltic Conference on Human Language Technologies (NoDaLiDa/Baltic-HLT 2025), pages 499–507, Tallinn, Estonia, March 2025. University of Tartu Library.

[22] Qwen Team, Alibaba Group. Qwen3.6- 35b. https://qwen.ai/blog?id=qwen3. 6-35b-a3b, 2026.

[23] François Remy. Parallia at #SMM4H-HeaRD 2026: ClinicalAligner26AM: A crosslingual aligner for dataset translation; evidences from the MultiClinCorpus shared task. In Guillermo Lopez-Garcia and Graciela Gonzalez-Hernandez, editors, Proceedings of the 11th Social Media Mining for Health Research and Applications (SMM4H-HeaRD 2026) Workshop and Shared Tasks, pages 165– 172, San Diego, United States, July 2026. Association for Computational Linguistics.

[24] François Remy. Clinicalaligner26am: A crosslingual aligner for dataset translation; evidences from the multiclincorpus shared task, 2026.

[25] Álvaro Rey-Blanes, Sara Giménez-Gómez, Francisco J. Veredas, and Francisco J. Moreno-Barea. ICB-UMA at #SMM4H–HeaRD 2026: Hybrid clinical entity projection for MultiClinAI: Adaptive candidate windows, XGBoost, and LLM refinement. In Guillermo Lopez-Garcia and Graciela Gonzalez-Hernandez, editors, Proceedings of the 11th Social Media Mining for Health Research and Applications (SMM4H-HeaRD 2026) Workshop and Shared Tasks, pages 127–132, San Diego, United States, July 2026. Association for Computational Linguistics.

[26] Jan Rodríguez-Miret, Eulàlia Farré-Maduell, Salvador Lima-López, Laura Vigil, Vicent Briva-Iglesias, and Martin Krallinger. Exploring the potential of neural machine translation for cross-language clinical natural language processing (nlp) resource generation through annotation projection. Information, 15(10), 2024.

[27] Gerard Salton and Christopher Buckley. Termweighting approaches in automatic text retrieval. Information Processing & Management, 24(5):513–523, 1988.

[28] Krish Sharma, Rhea Singhal, and Jatin Bedi. blue at SMM4H-HeaRD 2026: Class-weighted transformer ensembles with structured decoding and chain-of-thought blending across six

health NLP shared tasks. In Guillermo Lopez-Garcia and Graciela Gonzalez-Hernandez, editors, Proceedings of the 11th Social Media Mining for Health Research and Applications (SMM4H-HeaRD 2026) Workshop and Shared Tasks, pages 72–81, San Diego, United States, July 2026. Association for Computational Linguistics.

[29] Richard Tzong-Han Tsai, Shih-Hung Wu, Wen-Chi Chou, Yu-Chun Lin, Ding He, Jieh Hsiang, Ting-Yi Sung, and Wen-Lian Hsu. Various criteria in the evaluation of biomedical named entity recognition. BMC Bioinformatics, 7(1):92, 2006.

[30] David Yarowsky, Grace Ngai, and Richard Wicentowski. Inducing multilingual text analysis tools via robust projection across aligned corpora. In Proceedings of the First International Conference on Human Language Technology Research, 2001.

[31] Jamil Zaghir, Mina Bjelogrlic, Jean-Philippe Goldman, Soukaïna Aananou, Christophe Gaudet-Blavignac, and Christian Lovis. FRASIMED: A clinical French annotated resource produced through crosslingual BERTbased annotation projection. In Nicoletta Calzolari, Min-Yen Kan, Veronique Hoste, Alessandro Lenci, Sakriani Sakti, and Nianwen Xue, editors, Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 7450–7460, Torino, Italia, May 2024. ELRA and ICCL.

## A Prompts

Listing A1: Match-classification prompt in English   
You are a strict bilingual clinical text   
matcher.   
Your task is to determine whether a <   
SOURCE\_LANGUAGE> entity mention is   
represented in a <TARGET\_LANGUAGE> text   
window.   
If there is any noise, consider it a no   
match.   
Source entity:   
<SOURCE\_ENTITY>   
Target window:   
<TARGET\_WINDOW>   
Return a JSON object with this exact schema:   
{   
"match\_label": "exact match | mid match |   
no match",   
"justification": "string"   
}   
Label definitions:   
exact match: the target window clearly   
contains the same concept or a direct   
translation of the source entity   
- mid match: partial, approximate, broader,   
narrower, abbreviated, or contextually   
related match   
no match: the target window does not   
represent the same concept   
Rules:   
- Compare meaning, not literal wording   
Consider translations, abbreviations,   
paraphrases, and morphological variants   
- If uncertain between exact match and mid   
match, choose mid match   
- If match\_label is "exact match",   
justification must be an empty string   
- If match\_label is "mid match" or "no match   
", justification must contain a brief   
explanation   
Return JSON only

## Listing A2: Span correction prompt in English

You are a biomedical NER expert.   
Below is a clinical document in <   
TARGET\_LANGUAGE>, followed by entity   
spans that were not good matches.   
For each entity, use the document context   
and the <SOURCE\_LANGUAGE> reference to   
suggest the most accurate <   
TARGET\_LANGUAGE> entity string from the   
document.   
DOCUMENT:   
<TARGET\_DOCUMENT>

```python
PROBLEMATIC ENTITIES:
<PROBLEMATIC_ENTITIES>
Respond ONLY with a JSON array, no
explanation:
[{"id": 1, "suggestion": "..."}, ...]
```

Listing A3: Cross-lingual projection prompt in   
Spanish   
Eres especialista en anotación de corpus clí   
nicos multilingües. Proyecta únicamente   
las siguientes etiquetas XML desde el   
documento fuente en español al   
documento traducido. Conserva   
exactamente el texto traducido y   
devuelve únicamente el documento final   
en el idioma que corresponda con las   
etiquetas insertadas.   
Proyecta al documento traducido las   
etiquetas XML clínicas presentes en el   
documento español.   
Reglas obligatorias:   
Usa únicamente estas etiquetas: {   
ACTIVE\_LABELS}.   
Conserva exactamente el texto traducido;   
solo puedes insertar etiquetas XML.   
Mantén el mismo número de etiquetas de   
apertura y cierre para cada etiqueta   
que en el documento español.   
No añadas explicaciones, markdown, comillas   
ni ningún texto adicional.   
Devuelve únicamente el documento traducido   
etiquetado.   
Documento español etiquetado:   
<documento\_español\_etiquetado\_xml>   
{SPANISH\_XML\_DOCUMENT}   
</documento\_español\_etiquetado\_xml>   
Documento traducido sin etiquetas:   
<documento\_traducido>   
{TARGET\_DOCUMENT}   
</documento\_traducido>

## Listing A4: Cross-lingual projection prompt in English

You are a specialist in multilingual   
clinical corpus annotation. Project   
only the following XML tags from the   
Spanish source document into the   
translated document. Preserve the   
translated text exactly and return only   
the final document in the appropriate   
language with the tags inserted.   
Project the clinical XML tags present in the   
Spanish document onto the translated   
document.   
Mandatory rules:

Use only these tags: {ACTIVE\_LABELS}.   
Preserve the translated text exactly; you   
may only insert XML tags.   
Maintain the same number of opening and   
closing tags for each label as in the   
Spanish document.   
Do not add explanations, markdown, quotation   
marks, or any additional text.   
Return only the tagged translated document.   
Tagged Spanish document:   
<spanish\_doc\_tagged\_xml>   
{SPANISH\_XML\_DOCUMENT}   
</spanish\_doc\_tagged\_xml>   
Untagged translated document:   
<translated\_document>   
{TARGET\_DOCUMENT}   
</translated\_document>