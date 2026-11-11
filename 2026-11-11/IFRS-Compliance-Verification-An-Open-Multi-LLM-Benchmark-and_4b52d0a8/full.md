# IFRS Compliance Verification: An Open Multi-LLM Benchmark and Content-Conditioned Ensemble Classifier

Aghoghomena Akasukpe Li Yang Xinyao Zhou Chen Shen Miguel Vargas Martin

Faculty of Business and IT, Ontario Tech University

Oshawa, Ontario, Canada

{aghoghomena.akasukpe, li.yang, xinyao.zhou, chen.shen1, miguel.vargasmartin}@ontariotechu.ca

## Abstract

Verifying corporate financial disclosures against the International Financial Reporting Standards (IFRS) is a core auditing task in which experts cross-reference disclosures with hundreds of fine-grained regulatory paragraphs: a laborious, costly, and error-prone process that motivates growing interest in large language models (LLMs) as compliance assistants. Yet single-model accuracy remains insufficient for high-stakes use, and naive ensembling is vulnerable to systematic per-model biases that can drag aggregate accuracy below that of the strongest individual annotator. This gap is addressed with a multi-stage framework, supported by a new benchmark of 19,998 IFRS regulation-disclosure pairs comprising 11,161 unanimous multi-LLM labels and 8,837 human annotations on items where the LLMs disagreed. The framework couples a three-stage domain-adapted sentence embedding pipeline (BERT masked language modeling on SEC EDGAR filings, unsupervised TSDAE and SimCSE adaptation, and supervised contrastive alignment) with a learned ensemble classifier that fuses frozen embeddings and LLM probability features from GPT-4.1, Claude Sonnet 4, and Gemini 2.5 Pro via Tree-structured Parzen Estimator (TPE) optimized gradient-boosted trees. On the disagreement set, the ensemble achieves 81.7% weighted F1, exceeding the strongest single LLM by 10.4 points and majority voting by 14.6 points, and outperforms eleven unsupervised aggregation baselines; the retrieval pipeline attains Recall@15 of 45.3% on unseen regulations. The dataset, model checkpoints, annotation platform, and classifier implementation will be publicly released.

## 1 Introduction

Publicly traded companies are required to prepare financial statements that comply with regulatory standards such as the International Financial Reporting Standards (IFRS) (IFRS Foundation, 2025).

Accurate compliance verification has measurable economic consequences: richer firm-level disclo sure is associated with lower bid-ask spreads, lower cost of equity capital, and higher equity valuations (Botosan, 1997; Leuz and Verrecchia, 2000; Daske et al., 2008), and undetected non-compliance can trigger regulatory penalties, costly restatements, and investor losses. Verifying that individual disclosures satisfy specific regulatory paragraphs is a core task in external auditing, traditionally performed by domain experts who cross-reference disclosure text against hundreds of checklist items drawn from the applicable standards (Biesner et al., 2022). This process is laborious, expensive, and error-prone: a single annual filing may reference dozens of IFRS standards, each containing finegrained requirements that must be matched to scattered passages across the report (Hillebrand et al., 2023). The audit profession has invested accordingly: audit-firm investment in AI is associated with lower restatement rates and reduced audit fees (Fedyk et al., 2022); all Big-4 firms have publicly deployed AI tools spanning risk assessment, transaction testing, analytics, and work paper preparation (Kokina and Davenport, 2017; Munoko et al., 2020); and recent surveys document a corresponding shift toward AI-assisted compliance workflows (Dong et al., 2024; KPMG, 2025; Gu et al., 2024). Recent benchmarks show that frontier large language models (LLMs) can pass major accounting certification exams (Eulerich et al., 2024), yet the same benchmarks reveal that LLMs struggle with the deeper compliance task of citing relevant ac counting standards and reasoning over hierarchical multi-document structures, with accuracy dropping 60–90% on taxonomy-grounded financial reasoning (Wang et al., 2025).

A line of work originating from the Fraunhofer IAIS and PwC Germany collaboration has progressively advanced automated compliance verification: from supervised multi-label classification (Ramamurthy et al., 2021), through zero-shot sentencetransformer matching (Biesner et al., 2022), to a two-stage retrieve-then-verify pipeline combining domain-adapted BERT retrieval with LLM filtering (Hillebrand et al., 2023). Berger et al. (Berger et al., 2023) further evaluated six LLMs on IFRS and Handelsgesetzbuch (HGB) compliance datasets, finding that GPT-4 performed best overall but that individual models exhibit systematic biases; for example, Llama-2 70B excels at detecting noncompliance while under-performing on positive matches. These findings expose a fundamental limitation of single-model approaches: no individual LLM is uniformly reliable across compliance categories (Fotoh and Mugwira, 2025), and raw model confidence scores are known to be poorly calibrated, with overconfident predictions that do not reflect empirical accuracy (Xiong et al., 2024).

Combining predictions from multiple models reduces variance and corrects individual-model biases relative to any single predictor, making ensembles a long-standing tool for improving classifier reliability across domains (Dietterich, 2000). Multi-LLM ensembles extend this principle to languagebased annotation. Recent work on multi-agent debate (Du et al., 2024) and confidence-weighted consensus (Chen et al., 2024) further confirms that aggregating across heterogeneous models reduces bias and improves reliability. However, naively averaging or voting across miscalibrated annotators can systematically distort the resulting labels, as overconfident models dominate the aggregate (Liu et al., 2024).

In this paper, a multi-stage framework for automated IFRS compliance verification is presented that addresses a critical failure mode in multi-LLM annotation. Since no public benchmark for IFRS compliance verification exists, one is first constructed by pairing IFRS regulation clauses with financial statement disclosures via semantic retrieval, using a domain-adapted BERT encoder (bert-base-cased continued-pretrained via MLM on SEC EDGAR 10-K) to yield 20,000 regulation-disclosure pairs. This encoder is then progressively refined through unsupervised Transformer-based Sequential Denoising AutoEncoder (TSDAE) and Simple Contrastive Sentence Embeddings (SimCSE) adaptation, followed by supervised contrastive alignment on positive pairs. To label the dataset, three LLMs (GPT-4.1, Claude Sonnet 4, Gemini 2.5 Pro) are deployed as parallel annotators, yielding 11,161 items with unanimous multi-LLM labels; among the remaining 8,839 disagreement items, 8,837 receive resolved human labels that serve as ground truth. The central methodological novelty of the framework is a learned ensemble classifier that reframes label aggregation as content-conditioned classification: rather than combining LLM votes through fixed rules or unsupervised competence estimation, it fuses the frozen domain-adapted sentence embedding of each regulation-disclosure pair with the three LLMs’ probability features via gradient-boosted trees, allowing the aggregation decision to depend on the regulation-disclosure content rather than on confidence alone. This design remedies a key blind spot of standard aggregation: when one annotator is systematically biased on a class, confidence-only aggregators inherit that bias, whereas a contentconditioned classifier can learn when to override it.

The primary contributions of this work are:

1. A public IFRS compliance benchmark of 19,998 regulation-disclosure pairs, comprising 11,161 unanimous multi-LLM labels and 8,837 human-annotated cases of LLM disagreement: the first of its kind, as all prior work in this lineage relies on proprietary data.

2. A diagnostic of multi-LLM aggregation: that benchmarks 11 unsupervised aggregation strategies on the disagreement subset and isolates four structural failure modes (singlemodel class bias, Dawid–Skene parameter under-determination, CROWDLAB minorityclass collapse, and a temperature-bound sensitivity in EM joint calibration).

3. A learned ensemble classifier that reframes multi-LLM aggregation as contentconditioned classification, fusing each regulation-disclosure pair’s frozen domainadapted sentence embedding with LLM probability features via an Optuna-tuned XGBoost. Unlike confidence-only aggregators (voting, Dawid-Skene, MACE, CROWDLAB), it conditions on content and can learn when to override a biased annotator.

4. Open-source release of 39,404 disclosure paragraphs, 3,489 regulatory paragraphs (subject to IFRS Foundation licensing constraints), 20,000 LLM-annotated pairs, 8,837 human annotations, all model checkpoints, classifier code, and the annotation platform.<sup>1</sup>

The rest of this paper is organized as follows. Section 2 reviews related work on zero-shot text matching, domain-adapted embeddings, LLMdriven compliance reasoning, and weak supervision. Section 3 describes the methodology. Section 4 presents experimental results, and Section 5 concludes.

## 2 Related Work

## 2.1 Automated Compliance Verification in Financial Auditing

Early approaches to automated auditing used supervised multi-label classification to map financial report passages to regulatory checklist items (Ramamurthy et al., 2021), but these models cannot generalize to unseen requirements. Zero-shot sentencetransformer matching (Biesner et al., 2022) addressed this limitation by computing cosine similarity between Sentence-BERT embeddings of disclosures and regulations, enabling evaluation against novel requirements without retraining. Retrieval quality was further improved by combining BERT-based candidate selection with GPT-4 reranking (Hillebrand et al., 2023), yielding gains in sensitivity, Mean Average Precision (MAP), and F1. Benchmarking six LLMs on IFRS and HGB compliance tasks revealed that GPT-4 performs best overall while Llama-2 70B excels at noncompliance detection, and that prompt design significantly influences accuracy (Berger et al., 2023), exposing systematic per-model biases that singlemodel approaches cannot resolve.

## 2.2 Domain-Adapted Sentence Embeddings

Raw BERT embeddings suffer from anisotropy, clustering in a narrow cone that limits discriminative power for similarity tasks. Two unsupervised methods address this. TSDAE (Wang et al., 2021) reconstructs corrupted sentences through an encoder bottleneck, forcing the representation to capture holistic semantics. SimCSE (Gao et al., 2021) uses dropout-based augmentation to form positive pairs and trains with InfoNCE contrastive loss over in-batch negatives. Both operate without labeled data. Applied to financial auditing, these methods yielded Recall@5 below 20% on IFRS regulation retrieval under unsupervised adaptation alone (Biesner et al., 2022). Appending a supervised contrastive stage on annotated regulation-disclosure pairs raises seen-requirement Recall@5 above 90% while maintaining above 30% on unseen requirements. This progressive strategy of unsupervised adaptation followed by supervised alignment is the closest prior work to this setting.

## 2.3 LLM Ensembles for Annotation and Evaluation

LLMs have emerged as viable text annotators: zeroshot ChatGPT outperforms crowd workers by approximately 25 percentage points at 30× lower cost (Gilardi et al., 2023), and GPT-4 exceeds both expert and crowd-worker agreement on political text classification (Törnberg, 2023). LLMs are systematically overconfident when verbalizing uncertainty (Xiong et al., 2024), and naively aggregating miscalibrated outputs lets overconfident models dominate the ensemble. Independent per-model temperature scaling before aggregation reduces calibration error by 46% (Liu et al., 2024).

## 2.4 Label Aggregation and Calibration Without Ground Truth

The Dawid-Skene model (Dawid and Skene, 1979) estimates per-annotator confusion matrices via EM, treating true labels as latent variables. Extensions including MACE (Hovy et al., 2013) and GLAD (Whitehill et al., 2009) add annotator competence and item difficulty parameters. All assume conditional independence of annotators given the true label, an assumption violated by LLM annotators that share training data and architectural priors; Balasubramanian et al. (Balasubramanian et al., 2026) show this can flip the Bayes-optimal label when errors are correlated.

Post-hoc temperature scaling (Guo et al., 2017) corrects overconfidence but requires held-out ground truth. Incorporating even small amounts of ground truth data substantially improves quality estimation (Ipeirotis et al., 2010), and CROWD-LAB (Northcutt et al., 2021) produces consensus labels by combining annotator votes with classifier predictions without requiring ground truth.

## 3 Methodology

An end-to-end pipeline for IFRS compliance verification is presented, integrating domain-adapted retrieval, multi-LLM annotation, and a contentconditioned learned aggregator (Figure 1). Candidate regulation-disclosure pairs are constructed by BERT retrieval over an EDGAR-adapted encoder, labeled by a three-LLM ensemble, and reconciled against independent human annotations on the disagreement subset. A three-stage sentenceembedding pipeline yields domain-specific representations, and an Optuna-tuned XGBoost classifier maps the LLM probability panel together with a frozen SBERT embedding of the pair to a final label, benchmarked against a pool of unsupervised aggregation baselines. Each stage is detailed in the subsections that follow.

![](images/15e14e304f1a95281e6a800ab60458aaecc27be1e3feb7f7dd80d6ea6451f070.jpg)  
Figure 1: Proposed framework. Top-5 regulations per disclosure are retrieved by an EDGAR-adapted BERT encoder and labeled by a three-LLM panel; 8,837 disagreement items are reconciled with humans. An Optuna-tuned XGBoost meta-learner over LLM probabilities and a frozen SBERT pair embedding predicts the final label, benchmarked against baseline aggregators.

## 3.1 Task Definition and Dataset Construction

Compliance verification is formalized as a three-way classification problem. Given a pair (r, d) consisting of an IFRS regulation paragraph r and a financial statement disclosure paragraph d, the goal is to assign a label y ∈ {COMPLIED, NOT COMPLIED, NOT APPLICABLE} indicating whether d satisfies the requirements specified in r. Concretely, COMPLIED means the requirement applies to the disclosure’s topic and d satisfies it; NOT COMPLIED means the requirement applies to the disclosure’s topic but d alone does not satisfy it, although other passages elsewhere in the filing may satisfy it (the label characterizes the passage-level pairing (r, d), not company-level compliance); and NOT APPLICABLE means the requirement does not apply to the disclosure’s topic (e.g., a leases requirement evaluated against a revenue disclosure).

The regulation-side corpus consists of 3,489 paragraphs extracted via LlamaParse from the official IASB published PDFs of all 18 active IFRS standards (IFRS 1–3, 5–19). The disclosureside corpus consists of 39,404 paragraphs from 329 audited annual financial statements filed via SEDAR+<sup>2</sup> (fiscal years 2022–2025); jurisdiction, sector composition, issuer-size profile, and selection rationale are detailed in Appendix A, with the full parsing pipeline in Appendix D.

To construct candidate pairs, all paragraphs are embedded using the domain-adapted BERT encoder pre-trained via MLM on SEC EDGAR 10-K filings, and the top-15 regulations per disclosure are retrieved by cosine similarity. Retrieved scores range from 0.80 to 0.94, confirming high semantic overlap between the two corpora. From the 39,404 disclosures, 4,000 are randomly selected (fixed seed), and the top-5 regulation matches per disclosure are retained by similarity rank, yielding 20,000 pairs over 1,981 distinct regulations from all 18 standards and 4,000 distinct disclosures (statistics in Appendix B). Two of these pairs remained unresolved after human review (§3.3) and are excluded, leaving 19,998 labeled pairs. The top-5 cutoff balances coverage against annotation cost and makes the benchmark conditional on retrieval: it evaluates verification of retrieved candidates, not open retrieve-and-verify. Unretrieved pairs carry no labels. The resulting dataset, labeled by the multi-LLM ensemble described in

Section 3.2, is imbalanced: NOT APPLICABLE accounts for 11,759 pairs (58.8%), NOT COM-PLIED for 7,369 (36.8%), and COMPLIED for only 872 (4.4%). The imbalance is partly structural (retrieved regulations are topically adjacent but mostly do not govern the disclosure) and partly inherent (one paragraph rarely satisfies a requirement in full). The distribution motivated class-weighted training and stratified evaluation throughout.

## 3.2 Multi-LLM Ensemble Annotation

Three frontier LLMs from three independent model families are deployed as parallel annotators: GPT-4.1 (OpenAI, 2025) from OpenAI, Claude Sonnet 4 (Anthropic, 2025) from Anthropic, and Gemini 2.5 Pro (Google DeepMind, 2025) from Google Deep-Mind. Each is competitive on financial-reasoning and instruction-following benchmarks individually, but published audits show that no single frontier model is uniformly reliable across compliance categories (Berger et al., 2023; Fotoh and Mugwira, 2025), and the per-class analysis (Appendix N) confirms complementary failure modes: GPT-4.1 over-predicts NOT COMPLIED, Claude is balanced but conservative on the minority COMPLIED class, and Gemini is the strongest single model on COM-PLIED and NOT APPLICABLE yet degrades on NOT COMPLIED. All three are queried at temperature $T { = } 0$ for deterministic outputs. Each model receives a shared system prompt specifying an expert IFRS auditor role (Appendix E.1) and outputs a structured JSON containing a discrete label and a probability vector $\mathbf { p } = [ p _ { \mathrm { y e s } } , p _ { \mathrm { n o } } , p _ { \mathrm { n a } } ]$ over the three classes. The prompt queries each class probability independently in [0, 1] rather than enforcing a softmax over the three classes, so each returned p is L1-normalized to satisfy $\textstyle \sum _ { k } p _ { k } = 1$ before downstream aggregation, calibration, or feature extraction. Predictions are aggregated via hard vote (majority rule) and soft vote (argmax of averaged probability vectors). Of the 20,000 items, 55.8% receive unanimous agreement, 44.0% majority agreement, and only 0.2% full three-way disagreement.

## 3.3 Human Annotation for Ground Truth

Independent human annotations on the nonunanimous subset are collected through a purposebuilt platform (Appendix I). Three junior IFRS analysts label each pair blind to LLM predictions and to one another, supervised by two senior IFRS specialists who designed the protocol and adjudicated escalated cases. Each analyst handles a disjoint partition of roughly 3,000 partial-agreement items; a fixed 163-item overlap slice, which includes all 44 items on which the three LLMs disagree three ways, is multiply annotated (161 by all three analysts, 2 by two). On the 161, mean pairwise agreement is 65.8% (Fleiss’ $\kappa = 0 . 3 9 )$ , modest, but measured on items the three LLMs already disagree on. The analysts are unanimous on 81 of these items, split two to one on 75, and split three ways on 5, where no majority exists. Two-to-one splits take the majority label. All 80 split items and the 21 items flagged as difficult were referred to the two senior specialists for joint review, which agreed a final label for three of the five three-way items; the other two remained unresolved when labeling closed and are excluded from the released labels. In total, 8,837 items are annotated (44.2% of the dataset): 73.9% Not Applicable, 15.8% Not Complied, 10.3% Complied. The minority COMPLIED share is 2.3× higher in this disagreement subset than in the full LLM-labeled set (4.4%, §3.1), confirming that LLM disagreement concentrates around the under-represented positive class (partition details in Appendix J).

## 3.4 Domain-Adapted Sentence Embedding Pipeline

Stage 1: MLM Pre-training. The encoder is initialized from bert-base-cased and further pretrained with MLM on the Management Discussion and Analysis sections of SEC EDGAR 10-K filings (1993–2020), chunked into overlapping 512-token segments with 64-token stride and 15% masking. Training uses 3 epochs, batch size 16, learning rate $5 \times 1 0 ^ { - 5 }$ , weight decay 0.01, and fp16 mixed precision. The cased tokenizer is retained to preserve capitalization-sensitive acronyms (IFRS, IAS, GAAP).

Stage 2A: TSDAE. The encoder compresses a 60% token deleted input into a 768-dimensional vector via mean pooling; a decoder with tied weights reconstructs the original sentence, forcing holistic semantic encoding. Training uses 42,893 unlabeled sentences for 1 epoch with batch size 8 and learning rate $3 \times 1 0 ^ { - 5 }$

Stage 2B: SimCSE. Positive pairs are formed by passing each sentence through the encoder twice with independently sampled dropout masks; all other batch members serve as negatives under InfoNCE loss. Training uses the same corpus for 1 epoch with batch size 128 (127 in-batch negatives per anchor), learning rate $3 \times 1 0 ^ { - 5 }$ , and temperature $\tau { = } 0 . 0 5$

Stage 3: Supervised Contrastive Fine-Tuning. Both Stage 2 checkpoints are independently finetuned on 11,161 unanimous-agreement items. Pairs labeled COMPLIED (547 pairs) or NOT COMPLIED (4,383 pairs) form 4,930 positive training pairs. After a 90/10 split, training uses MultipleNegatives-RankingLoss, batch size 16, learning rate $2 \times 1 0 ^ { - 5 }$ 9 $\tau { = } 0 . 0 5$ , and 4 epochs.

## 3.5 Learned Ensemble Classifier

This work frames aggregation as stacked generalization (Wolpert, 1992; Sagi and Rokach, 2018): the three LLMs of §3.2 are level-0 base learners emitting per-class probabilities, and an XG-Boost (Chen and Guestrin, 2016) classifier is the level-1 meta-learner mapping those probabilities, together with a frozen content representation, to the final label; unlike hard/soft vote or unsupervised competence models (Dawid–Skene, MACE, CROWDLAB), it is trained on human ground truth and also reads the input’s domain adapted embedding, so the decision depends on what the pair says as well as on what the panel said. The 1,565-dim input concatenates a 29-dim panel summary (nine raw class probabilities, per model Shannon entropies, hard-vote count, soft vote confidence, max per-pair confidence, five bi nary agreement-pattern indicators, per-class max– min spread and standard deviation across mod els, three pairwise symmetric KL divergences) and the 1,536-dim frozen SBERT embedding (Reimers and Gurevych, 2019) of the regulation–disclosure pair from the TSDAE Stage-2A (unsupervised, domain-adapted) checkpoint (§3.4); only the metalearner is trained. Gradient-boosted trees fit this setting on four counts: bounded panel statistics and the dense embedding are handled in native scales without standardization; at 8,837 items under 5- fold CV, regularized boosting dominates end-toend deep alternatives (Appendix Q, vs. fine-tuned BERT and confidence-weighted neural classifiers); per-sample weights, exposed as three free class weights, push capacity onto the minority COM-PLIED/NOT COMPLIED classes rather than collapsing to NA; and L1/L2 leaf penalties with depth and shrinkage fit the 1,565-dim input on ∼7,000 items per fold without overfit. Optuna (Akiba et al., 2019) with the Tree-structured Parzen Esti mator (TPE) sampler then selects hyperparameters over 100 trials, each scored by 5-fold stratified CV weighted F1, jointly tuning tree-structure parameters (estimators, depth, learning rate, subsample, column subsample, min child weight, $\gamma , \alpha , \lambda )$ and three class weights; the chosen configuration is 777 trees of depth 7, learning rate 0.038, weights $( w _ { \mathrm { N A } } , w _ { \mathrm { N O } } , w _ { \mathrm { Y E S } } ) { = } ( 1 . 0 , 3 . 5 , 7 . 1 )$ , without which the meta-learner collapses to an NA-dominant policy with W-F1 below 0.75. Which LLM to trust is content-dependent, not confidence-dependent.

Splits and leakage. All classifier predictions are out-of-fold under stratified 5-fold cross-validation on the 8,837 human-labeled items, which share no pair with the 11,161 unanimous items used for contrastive training; the classifier’s Stage-2A encoder sees no labels. Folds are stratified by label, not grouped by disclosure, so the evaluation measures generalization to unseen pairs, not unseen filings.

## 3.6 Evaluation Protocol

Four aspects of the framework are evaluated: retrieval quality, calibration diagnostics, aggregation comparison, and IAA.

Retrieval Evaluation. Recall@K for $K \in$ {1, 5, 10, 15} is reported separately on seen regulations (present in Stage 3 training) and unseen regulations (held out entirely), isolating in-domain retrieval from zero-shot generalization.

Calibration Diagnostics. Per-model accuracy and Cohen’s κ are computed against the 8,837 human labels for each LLM, hard vote, soft vote, and EM-calibrated predictions (algorithm in $\mathsf { A p - }$ pendix F). Calibration uses ECE (15 bins).

Aggregation Benchmark. Eleven aggregation strategies are evaluated on the 8,837-item set and compared on accuracy, weighted F1, macro F1, and Cohen’s κ against the learned ensemble classifier described in Section 3.5.

## 4 Experiments

## 4.1 Retrieval Results

The five model checkpoints from the embedding pipeline (§3.4) are evaluated on the retrieval task: given a regulation, all 39,404 disclosure candidates are ranked by cosine similarity and human-labeled relevant disclosures are checked for appearance in the top K positions. All models are evaluated on independent human annotations (8,837 items) collected separately from the LLM labels used for supervised training. Recall@K and Mean Reciprocal Rank (MRR) are reported with 95% bootstrap confidence intervals from 1,000 resamples, separately for seen regulations (present in supervised training with different disclosure pairings) and unseen regulations (held out entirely for zero-shot evaluation). Table 1 reports the headline results; full results including MAP@15 and NDCG@15 for all five checkpoints are in Appendix L; per-stage Hyperparameters are in Appendix K.

All methods ranked on the shared evaluation set
<table><tr><td>Model</td><td>R@5</td><td>R@15</td><td>MRR</td></tr><tr><td>Seen Regulations (n=530)</td><td></td><td></td><td></td></tr><tr><td>MLM only</td><td>8.9</td><td>20.9</td><td>0.060</td></tr><tr><td> $+ \mathrm { T S D } \dot { \mathrm { A E } }$ </td><td>7.7</td><td>19.8</td><td>0.053</td></tr><tr><td> $+ \operatorname { S i m C S E }$ </td><td>7.4</td><td>15.8</td><td>0.046</td></tr><tr><td> $+ \mathrm { T S D A E } + \mathrm { S u p }$ </td><td>11.9</td><td>27.5</td><td>0.080</td></tr><tr><td> $+ \mathrm { S i m C S E } + \mathrm { S i p }$ </td><td>11.3</td><td>26.4</td><td>0.073</td></tr><tr><td colspan="4">Unseen Regulations (n=106)</td></tr><tr><td>MLM only</td><td>11.3</td><td>28.3</td><td>0.086</td></tr><tr><td> $+ \mathrm { T S D A E }$ </td><td>12.3</td><td>26.4</td><td>0.069</td></tr><tr><td> $+ \operatorname { S i m C S E }$ </td><td>12.3</td><td>27.4</td><td>0.078</td></tr><tr><td>+  $\mathrm { T S D A E } + \mathrm { S u p }$ </td><td>18.9</td><td>41.5</td><td>0.122</td></tr><tr><td> $+ \mathrm { S i m C S E } + \mathrm { S u p }$ </td><td>20.8</td><td>45.3</td><td>0.137</td></tr></table>

Table 1: Retrieval performance on (3,489 regulations, 39,404 disclosures).

Supervised training is the decisive step. Both unsupervised methods alone provide marginal improvement over the MLM baseline, with Recall@5 below 13% for both splits, consistent with Biesner et al. (Biesner et al., 2022). After Stage 3 fine-tuning, all metrics improve substantially. Sim-CSE + Supervised reaches 20.8% unseen R@5 and 45.3% unseen R@15; TSDAE + Supervised leads on seen regulations (R@5 = 11.9%, MRR = 0.080). The two models are complementary: TSDAE’s denoising objective produces representations better suited to familiar regulation patterns, while Sim-CSE’s contrastive objective generalizes more effectively to novel regulations.

## 4.2 Classification Benchmark and Learned Ensemble Classifier

Table 2 reports results across individual LLMs, representative aggregation methods, and the learned ensemble classifier; per-class metrics and the full 11-strategy comparison are in Appendix N and Appendix O respectively.

Aggregation partially recovers signal. Figure 2 ranks all methods by weighted F1. GPT-4.1 exhibits severe class-level bias on the humanannotated disagreement subset: 95.5% of its predictions there are NOT COMPLIED, yielding nearchance agreement with humans $( \kappa = 0 . 0 1 9$ , Table 2). This biased contribution pulls hard vote (0.671 W-F1) and soft vote (0.672 W-F1) below the best single model, Gemini, at 0.713 W-F1. MACE (Hovy et al., 2013) is the strongest aggregator at 0.722 W-F1. Its single scalar reliability parameter is more stable at this panel size than Dawid-Skene’s full confusion matrix: with only three annotators the 9-parameter per-model matrix is undetermined, and the M-step collapses to majority voting, explaining why Dawid-Skene ties hard vote exactly at 0.671 (Table 2). Several other aggregators (GLAD, Wawa, One-Coin DS, M-MSR) fall in the same 0.671–0.687 band: extra parameters do not help with three annotators.

![](images/b8fe708ee5b0372cc36aef4a0623f9626a9c9b3c3daa1621fb577b8c60167898.jpg)  
Figure 2: Method leaderboard: all 15 evaluated methods (three LLMs, eleven aggregators, the proposed classifier) ranked by weighted F1 against 8,837 independent human annotations. The learned ensemble classifier (starred) reaches 0.817, exceeding the best single LLM (Gemini, 0.713) and the best unsupervised aggregator (MACE, 0.722).

<table><tr><td>Method</td><td>Acc</td><td>W-F1</td><td>κ</td><td>ECE</td></tr><tr><td>Individual LLMs</td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-4.1 (OpenAI, 2025)</td><td>0.183</td><td>0.095</td><td>0.019</td><td>0.811</td></tr><tr><td>Claude Sonnet 4 (Anthropic, 2025)</td><td>0.672</td><td>0.687</td><td>0.256</td><td>0.211</td></tr><tr><td>Gemini 2.5 Pro (Google DeepMind, 2025)</td><td>0.737</td><td>0.713</td><td>0.278</td><td>0.215</td></tr><tr><td>Aggregation</td><td></td><td></td><td></td><td></td></tr><tr><td>Hard Vote</td><td>0.641</td><td>0.671</td><td>0.254</td><td></td></tr><tr><td>Soft Vote</td><td>0.642</td><td>0.672</td><td>0.255</td><td>0.290</td></tr><tr><td>Dawid-Skene (Dawid and Skene, 1979)</td><td>0.641</td><td>0.671</td><td>0.254</td><td></td></tr><tr><td>One-Coin DS (Zhang et al.,</td><td>0.668</td><td>0.679</td><td>0.238</td><td></td></tr><tr><td>2016) EM Calibrated (T=5,</td><td>0.456</td><td>0.503</td><td>0.149</td><td>0.461</td></tr><tr><td>default) EM Calibrated (T=20)</td><td>0.647</td><td>0.672</td><td>0.245</td><td>0.292</td></tr><tr><td>(Guo et al., 2017) Wawa (Ustalov et al., 2024)</td><td>0.642</td><td>0.671</td><td>0.254</td><td></td></tr><tr><td>M-MSR (Ma and</td><td>0.672</td><td>0.687</td><td>0.256</td><td></td></tr><tr><td>Olshevsky, 2020) GLAD (Whitehill et al.,</td><td>0.642</td><td>0.671</td><td>0.254</td><td></td></tr><tr><td>2009) CROWDLAB (Goh et al.,</td><td>0.791</td><td>0.720</td><td>0.340</td><td></td></tr><tr><td>2022) MACE (Hovy et al., 2013)</td><td>0.748</td><td></td><td></td><td></td></tr><tr><td>Proposed</td><td></td><td>0.722</td><td>0.312</td><td></td></tr><tr><td>Optuna XGBoost</td><td>0.824</td><td>0.817</td><td>0.556</td><td>0.068</td></tr><tr><td>(proposed)</td><td></td><td></td><td></td><td></td></tr></table>

Table 2: Classification benchmark on 8,837 disagreement items.

![](images/d66f99ee36907046a07119c75dbe6b2c4eb5637292c5100a984ccef100b51fce.jpg)  
Figure 3: Per-class F1 across all 15 evaluated methods on 8,837 human-annotated items.

CROWDLAB narrows but does not close the gap to the proposed method. CROWDLAB (Goh et al., 2022) achieves the best accuracy (0.791) and Cohen’s kappa (0.340) among unsupervised aggregators (Table 2), with W-F1 (0.720) close to MACE. The split between W-F1 and accuracy indicates a deeper failure mode: CROWD-LAB attains 0.88 F1 on NOT APPLICABLE and 0.69 F1 on COMPLIED, but 0.00 F1 on NOT COMPLIED (Figure 3). CROWDLAB’s classifier-augmented consensus drives the model to abandon the minority class for NOT APPLICA-BLE.

EM calibration’s value is diagnostic. Under the adopted bound $T _ { \mathrm { m a x } } { = } 2 0 . 0 $ , learned temperatures correctly rank annotator reliability (Anthropic T=0.55, Gemini T=5.73, OpenAI T=14.36), with no model pinned at the bound; the calibrated ensemble reaches 0.672 W-F1 and ECE 0.292, matching soft vote. Per-model ECE improves substantially for the two over-confident annotators (OpenAI 0.811→0.508; Gemini 0.215→0.123) but rises slightly for the already near-calibrated Anthropic (0.211→0.276), since EM over-sharpens its raw outputs in service of the consensus fit (Appendix H). Under the textbook bound $T _ { \mathrm { m a x } } { = } 5 . 0$

OpenAI and Gemini both pin at 5.0 and ensemble W-F1 collapses to 0.503 (Table 2; bound sensitivity in Appendix R). Either way, EM does not exceed unsupervised aggregation: its practical value is identifying which models to trust, not directly producing labels.

The learned ensemble classifier closes 41.9% of the oracle gap. A per-item oracle selecting the correct model per pair reaches 0.949 W-F1, a 22.7- point gap above MACE that motivates a contentconditioned approach. The Optuna-tuned XGBoost classifier (§3.5) reaches 0.817 W-F1, beating every aggregation baseline by at least 9.5 points, with gains concentrated on the minority NOT COM-PLIED and COMPLIED classes (Figure 3). Probability features alone saturate at 0.727 W-F1; adding the frozen 1,536-dimensional SBERT embedding (Reimers and Gurevych, 2019) of the regulation– disclosure pair contributes +8.1 points and hyperparameter search a further 0.9 (Appendix Q), showing that which model to trust is content-dependent.

## 5 Conclusion

This paper presents a multi-stage framework for IFRS compliance verification and reports three findings. First, the domain-adapted embedding pipeline (BERT→TSDAE/SimCSE→supervised contrastive) reaches 45.3% Recall@15 on unseen regulations against 39,404 candidates, confirming industrial-scale transfer. Second, training the contrastive stage on 4,930 LLM-unanimous pairs yields overlapping 95% bootstrap confidence intervals with training on 1,616 human-annotated pairs, supporting LLM-unanimous labels as a practical weak-supervision alternative in this setting. Third, naive aggregation is fragile under single-model bias: GPT-4.1’s 95.5% “Not Complied” rate on the disagreement subset pulls hard and soft voting (0.671/0.672 W-F1) below the best single model (Gemini, 0.713), whereas an Optuna-tuned XG-Boost over frozen embeddings and LLM probability features reaches 81.7% W-F1, closing 41.9% of the per-item oracle gap.

The dataset, checkpoints, classifier code, and annotation platform will be released to support further work on multi-LLM annotation, financial NLP, and reliability-aware ensembles.

## Limitations

No claim of universality is made. The framework is evaluated on a single domain (IFRS-aligned annual financial statements), a single language (English), a single class taxonomy (COMPLIED / NOT COMPLIED / NOT APPLICABLE), and a fixed three-LLM annotator panel. Within that scope, the following caveats qualify the reported findings.

Paragraph-level scope. Labels are assigned to paragraph-level pairings (r, d) rather than to whole filings. A pair marked NOT COMPLIED only indicates that disclosure paragraph d alone does not satisfy regulation r; another passage in the same filing may, and a single COMPLIED pair establishes only that the requirement is met somewhere, not that no contradiction exists elsewhere. Deriving a filing-level verdict therefore requires an additional aggregation step over disclosure paragraphs, which is outside the reported evaluation.

Retrieval candidate pre-selection. Candidate regulation–disclosure pairs are pre-selected by a domain-adapted BERT (MLM-fine-tuned on EDGAR) using top-5 cosine ranking; pairs scored below rank 5 by this model are absent from the evaluation set. The relevant pairings this excludes cannot be counted from the benchmark, since unretrieved pairs carry no labels; the pre-selection encoder’s own Recall@15 of 20.9% (seen) and 28.3% (unseen regulations; Table 1, MLM only, ranking disclosures for a regulation) suggests the exclusion is substantial. A stratified random sample from the full 3, 489 × 39, 404 pair space would yield an unbiased benchmark; this benchmark is conditional on that model’s relevance prior.

Label reliability. Human labels cover only the disagreement subset; the 11,161 unanimous items carry LLM labels that no human has validated, so any score on them measures agreement with the LLM label, not accuracy (Appendix Q), and the 0.817 on the disagreement subset should not be read as a bound on performance over the full benchmark. Inter-annotator agreement on the overlap slice is modest (κ = 0.39), and most human items are singly annotated, so individual labels on the hardest pairs carry noise that majority voting and senior review reduce but do not remove.

Commercial audit tools. No comparison with commercial audit products is reported: the AI tools of large audit firms are proprietary, expose no public interface, and report no paragraph-level IFRS metrics, so a reproducible comparison is not currently possible. Once released, the benchmark is intended to make such comparisons possible for their developers.

Small annotator panel. Three LLMs is a small ensemble. Extensions to 5–10 annotators would exercise Dawid–Skene and related matrix-estimation methods more fairly and may change the ranking of aggregation methods, and consequently the gain attributable to the learned ensemble classifier.

Class imbalance. The dataset is heavily skewed (4.4% COMPLIED / 36.8% NOT COMPLIED / 58.8% NOT APPLICABLE). The learned ensemble classifier was trained and evaluated on this distribution; performance on regulatory regimes with materially different applicability rates is not verified.

Corpus and transfer. The disclosures come from an unstratified convenience sample of 329 SEDAR+ filings, weighted toward small- and midcap Canadian issuers and the mining sector (Appendix A); results may not transfer to large-cap or other-sector reporting without re-evaluation. Only two components are IFRS-specific, the domainadapted encoder and the annotation prompt; the stacking architecture is not. Transfer to other regimes (US GAAP, Basel III), which use different document conventions and taxonomies, to non-English filings, and to finer label taxonomies is untested.

Commercial LLM dependency and reproducibility cost. The annotation pipeline relies on three commercial APIs (GPT-4.1, Claude Sonnet 4, Gemini 2.5 Pro). Specific model versions may be deprecated; classifier features derived from their probability outputs may not transfer to successor models without re-tuning. Open-weight annotators are not evaluated here. The classifier accepts any annotator from which per-class scores can be extracted but must be refit when the panel changes; the frozen content embedding, which carries most of the gain over probability features (Appendix Q), does not depend on the panel. Deployment cost is dominated by this panel: the classifier adds one boosted-tree pass per pair, while the three LLM calls, independent and parallelizable, cost at least USD 0.003 per pair at list prices, about USD 60 for the 20,000-pair corpus (Appendix C).

## References

Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. 2019. Optuna: A Nextgeneration Hyperparameter Optimization Framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD ’19, pages 2623–2631, New York, NY, USA. Association for Computing Machinery.

Anthropic. 2025. Introducing Claude 4. Claude Sonnet 4 announcement; used as one of three LLM annotators in this work.

Krishnakumar Balasubramanian, Aleksandr Podkopaev, and Shiva Prasad Kasiviswanathan. 2026. Dependence-Aware Label Aggregation for LLMas-a-Judge via Ising Models. arXiv preprint. ArXiv:2601.22336 [cs].

Armin Berger, Lars Hillebrand, David Leonhard, Tobias Deußer, Thiago Bell Felix De Oliveira, Tim Dilmaghani, Mohamed Khaled, Bernd Kliem, Rudiger Loitz, Christian Bauckhage, and Rafet Sifa. 2023. Towards Automated Regulatory Compliance Verification in Financial Auditing with Large Language Models. In 2023 IEEE International Conference on Big Data (BigData), pages 4626–4635.

David Biesner, Maren Pielka, Rajkumar Ramamurthy, Tim Dilmaghani, Bernd Kliem, Rüdiger Loitz, and Rafet Sifa. 2022. Zero-Shot Text Matching for Automated Auditing using Sentence Transformers. In 2022 21st IEEE International Conference on Machine Learning and Applications (ICMLA), pages 1637–1642.

Christine A. Botosan. 1997. Disclosure Level and the Cost of Equity Capital. The Accounting Review, 72(3):323–349.

Justin Chih-Yao Chen, Swarnadeep Saha, and Mohit Bansal. 2024. ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics. Association for Computational Linguistics.

Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’16, pages 785–794, New York, NY, USA. Association for Computing Machinery.

Holger Daske, Luzi Hail, Christian Leuz, and Rodrigo Verdi. 2008. Mandatory IFRS Reporting around the World: Early Evidence on the Economic Consequences. Journal of Accounting Research, 46(5):1085–1142.

A. P. Dawid and A. M. Skene. 1979. Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm. Applied Statistics, 28(1):20–28.

Thomas G. Dietterich. 2000. Ensemble Methods in Machine Learning. In Multiple Classifier Systems, volume 1857 of Lecture Notes in Computer Science, pages 1–15. Springer Berlin Heidelberg.

Mengming Michael Dong, Theophanis C. Stratopoulos, and Victor Xiaoqi Wang. 2024. A scoping review of ChatGPT research in accounting and finance. International Journal ofAccounting Information Systems, 55:100715.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. 2024. Improving Factuality and Reasoning in Language Models through Multiagent Debate. In Proceedings of the 41st International Conference on Machine Learning. PMLR.

Marc Eulerich, Aida Sanatizadeh, Hamid Vakilzadeh, and David A. Wood. 2024. Is it All Hype? Chat-GPT’s Performance and Disruptive Potential in the Accounting and Auditing Industries. Review ofAccounting Studies, 29:2318–2349.

Anastassia Fedyk, James Hodson, Natalya Khimich, and Tatiana Fedyk. 2022. Is artificial intelligence improving the audit process? Review ofAccounting Studies, 27(3):938–985.

Lilian Elmvall Fotoh and Tafadzwa Mugwira. 2025. Exploring Large Language Models in External Audits: Implications and Ethical Considerations. International Journal ofAccounting Information Systems, 56:100748.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple contrastive learning of sentence embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6894–6910, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli. 2023. ChatGPT Outperforms Crowd Workers for Text-Annotation Tasks. Proceedings ofthe National Academy ofSciences, 120(30):e2305016120.

Hui Wen Goh, Ulyana Tkachenko, and Jonas Mueller. 2022. CROWDLAB: Automated Machine-Learning-Based Consensus Labeling from Multiple Annotators. In NeurIPS 2022 Workshop on Human in the Loop Learning.

Google DeepMind. 2025. Gemini 2.5: Our newest Gemini model with thinking. Announcement of Gemini 2.5; Gemini 2.5 Pro used as one of three LLM annotators in this work.

Haocheng Gu, Marco Schreyer, Kevin Moffitt, and Miklos A. Vasarhelyi. 2024. Artificial Intelligence Co-Piloted Auditing. International Journal of Accounting Information Systems, 54:100698.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. 2017. On Calibration of Modern Neural Networks. In Proceedings ofthe 34th International Conference on Machine Learning, pages 1321–1330. PMLR.

Lars Hillebrand, Armin Berger, Tobias Deußer, Tim Dilmaghani, Mohamed Khaled, Bernd Kliem, Rüdiger Loitz, Maren Pielka, David Leonhard, Christian Bauckhage, and Rafet Sifa. 2023. Improving Zero-Shot Text Matching for Financial Auditing with Large Language Models. In Proceedings ofthe ACM Symposium on Document Engineering 2023, DocEng ’23, pages 1–4, New York, NY, USA. Association for Computing Machinery.

Dirk Hovy, Taylor Berg-Kirkpatrick, Ashish Vaswani, and Eduard Hovy. 2013. Learning Whom to Trust with MACE. In Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1120–1130. Association for Computational Linguistics.

IFRS Foundation. 2025. International financial reporting standards (ifrs). https://www.ifrs.org/. Accessed: 2025-09-30.

Panagiotis G. Ipeirotis, Foster Provost, and Jing Wang. 2010. Quality Management on Amazon Mechanical Turk. Proceedings of the ACM SIGKDD Workshop on Human Computation, pages 64–67.

Julia Kokina and Thomas H. Davenport. 2017. The Emergence of Artificial Intelligence: How Automation is Changing Auditing. Journal of Emerging Technologies in Accounting, 14(1):115–122.

KPMG. 2025. AI in Financial Reporting and Audit: Navigating the New Era. https://kpmg.com/ xx/en/our-insights/ai-and-technology/ ai-in-financial-reporting-and-audit.html. Accessed: 2025-09-30.

Christian Leuz and Robert E. Verrecchia. 2000. The Economic Consequences of Increased Disclosure. Journal of Accounting Research, 38:91–124. Supplement: Studies on Accounting Information and the Economics of the Firm.

Yuxuan Liu, Tianchi Yang, Shaohan Huang, Zihan Zhang, Haizhen Huang, Furu Wei, Weiwei Deng, Feng Sun, and Qi Zhang. 2024. Calibrating LLM-Based Evaluator. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 2638–2656. ELRA and ICCL.

LlamaIndex. 2024. LlamaParse: Cloud-Based document parsing for LLM applications. Accessed via llama-cloud-services Python SDK v0.6.83.

Qianqian Ma and Alex Olshevsky. 2020. Adversarial Crowdsourcing Through Robust Rank-One Matrix Completion. In Advances in Neural Information Processing Systems 33 (NeurIPS 2020).

Ivy Munoko, Helen L. Brown-Liburd, and Miklos Vasarhelyi. 2020. The Ethical Implications of Using Artificial Intelligence in Auditing. Journal of Business Ethics, 167(2):209–234.

Curtis G. Northcutt, Lu Jiang, and Isaac L. Chuang. 2021. Confident Learning: Estimating Uncertainty in Dataset Labels. Journal ofArtificial Intelligence Research, 70:1373–1411.

OpenAI. 2025. Introducing GPT-4.1 in the API. Announcement of the GPT-4.1 series; used as one of three LLM annotators in this work.

Rajkumar Ramamurthy, Maren Pielka, Robin Stenzel, Christian Bauckhage, Rafet Sifa, Tim Dilmaghani Khameneh, Ulrich Warning, Bernd Kliem, and Rüdiger Loitz. 2021. ALiBERT: Improved automated list inspection (ALI) with BERT. In Proceedings of the 21st ACM Symposium on Document Engineering, DocEng ’21, pages 1–4, New York, NY, USA. Association for Computing Machinery.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.

Omer Sagi and Lior Rokach. 2018. Ensemble learning: A survey. WIREs Data Mining and Knowledge Discovery, 8(4):e1249.

Petter Törnberg. 2023. ChatGPT-4 Outperforms Experts and Crowd Workers in Annotating Political Twitter Messages with Zero-Shot Learning. arXiv preprint. ArXiv:2304.06588 [cs].

Dmitry Ustalov, Nikita Pavlichenko, and Boris Tseitlin. 2024. Learning from Crowds with Crowd-Kit. Journal ofOpen Source Software, 9(96):6227.

Kexin Wang, Nils Reimers, and Iryna Gurevych. 2021. TSDAE: Using transformer-based sequential denoising auto-encoder for unsupervised sentence embedding learning. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 671–688, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Rushi Wang, Jiateng Liu, Weijie Zhao, Shenglan Li, and Denghui Zhang. 2025. AuditBench: A Benchmark for Large Language Models in Financial Statement Auditing. In Communications in Computer and Information Science, volume 2533, pages 59–81. Springer.

Jacob Whitehill, Ting-fan Wu, Jacob Bergsma, Javier Movellan, and Paul Ruvolo. 2009. Whose Vote Should Count More: Optimal Integration of Labels from Labelers of Unknown Expertise. In Advances in Neural Information Processing Systems, volume 22.

David H. Wolpert. 1992. Stacked generalization. Neural Networks, 5(2):241–259.

Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. 2024. Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. In Proceedings ofthe Twelfth International Conference on Learning Representations.

Yuchen Zhang, Xi Chen, Dengyong Zhou, and Michael I. Jordan. 2016. Spectral Methods Meet EM: A Provably Optimal Algorithm for Crowdsourcing. Journal ofMachine Learning Research, 17(102):1– 44.

## Appendix

## A Disclosure Corpus Provenance

The 329 audited annual financial statement (AFS) PDFs that form the disclosure side of the bench mark (Section 3.1) are predominantly Canadian, obtained from SEDAR+ (the public regulatory portal operated by the Canadian Securities Administrators, on which reporting issuers listed on Canadian exchanges including TSX, TSX-V, CSE, and NEO are required to file their continuous-disclosure documents), with a small number of cross-listed U.S. 10-K reports (12 filings) and Quebec-origin Frenchlanguage filings (3 filings); English-language Canadian filings constitute the bulk of the set. Fiscal year-ends span 2022 through 2025, with 93% of filings reporting on fiscal year 2025; the three most common quarter-end dates are 30 June (159 filings), 31 July (80 filings), and 31 August (28 filings), reflecting the typical Canadian junior-issuer fiscal calendar. The corpus is heavily weighted toward small- and mid-cap reporters, including a substantial population of capital pool companies and junior exploration issuers; large-cap issuers such as chartered banks and senior producers are present but proportionally rare. Mining and resource explo ration is the single largest sector (approximately 30% of the corpus), with secondary representation from capital pool companies, financial-holding entities, and biotechnology and life sciences, and a long tail of consumer, industrial, and real-estate issuers. The 329 filings were assembled as an unstratified convenience sample of publicly available SEDAR+ filings within the data collection window, retained without sector, size, or auditor-firm stratification so that the benchmark preserves the natural distribution of disclosure language a compliance verifier encounters in practice rather than enforcing artificial balance.

## B Dataset Statistics

The 19,998 labeled pairs combine 1,981 distinct IFRS regulations, drawn from all 18 standards, with 4,000 distinct disclosures; each disclosure contributes its top-5 retrieved regulations. Table 3 breaks the pairs down by standard and Table 4 by label source. The distribution in §3.1 is the three-LLM hard vote over all pairs; the tables here instead use the human label wherever one exists, which is why COMPLIED is larger here (1,461 against 872): LLM disagreement concentrates on the positive class. Coverage is uneven across standards, from 2,556 pairs for IFRS 2 to 334 for IFRS 19, and IFRS 14 (regulatory deferral accounts) has no COMPLIED pair at all.

<table><tr><td>Standard</td><td>Regs</td><td>Pairs</td><td>C</td><td>NC</td><td>NA</td><td>Human %</td></tr><tr><td>IFRS 1</td><td>97</td><td>893</td><td>32</td><td>185</td><td>676</td><td>61</td></tr><tr><td>IFRS 2</td><td>138</td><td>2,556</td><td>297</td><td>798</td><td>1,461</td><td>45</td></tr><tr><td>IFRS 3</td><td>140</td><td>1,192</td><td>67</td><td>207</td><td>918</td><td>46</td></tr><tr><td>IFRS 5</td><td>61</td><td>607</td><td>26</td><td>101</td><td>480</td><td>45</td></tr><tr><td>IFRS 6</td><td>25</td><td>672</td><td>113</td><td>281</td><td>278</td><td>35</td></tr><tr><td>IFRS 7</td><td>227</td><td>2,395</td><td>26</td><td>985</td><td>1,384</td><td>42</td></tr><tr><td>IFRS 8</td><td>43</td><td>335</td><td>13</td><td>51</td><td>271</td><td>51</td></tr><tr><td>IFRS 9</td><td>127</td><td>2,167</td><td>306</td><td>696</td><td>1,165</td><td>36</td></tr><tr><td>IFRS 10</td><td>111</td><td>933</td><td>71</td><td>185</td><td>677</td><td>38</td></tr><tr><td>IFRS 11</td><td>47</td><td>447</td><td>31</td><td>76</td><td>340</td><td>51</td></tr><tr><td>IFRS 12</td><td>84</td><td>489</td><td>22</td><td>94</td><td>373</td><td>51</td></tr><tr><td>IFRS 13</td><td>88</td><td>994</td><td>74</td><td>490</td><td>430</td><td>39</td></tr><tr><td>IFRS 14</td><td>56</td><td>848</td><td>0</td><td>143</td><td>705</td><td>68</td></tr><tr><td>IFRS 15</td><td>176</td><td>1,129</td><td>63</td><td>297</td><td>769</td><td>40</td></tr><tr><td>IFRS 16</td><td>158</td><td>1,391</td><td>113</td><td>368</td><td>910</td><td>42</td></tr><tr><td>IFRS 17</td><td>222</td><td>1,390</td><td>73</td><td>358</td><td>959</td><td>36</td></tr><tr><td>IFRS 18</td><td>140</td><td>1,226</td><td>94</td><td>373</td><td>759</td><td>48</td></tr><tr><td>IFRS 19</td><td>41</td><td>334</td><td>40</td><td>89</td><td>205</td><td>68</td></tr><tr><td>Total</td><td>1,981</td><td>19,998</td><td>1,461</td><td>5,777</td><td>12,760</td><td>44</td></tr></table>

Table 3: Pairs per IFRS standard. Labels are human where available, otherwise the unanimous LLM label (C = COMPLIED, NC = NOT COMPLIED, NA = NOT APPLICABLE). Human % is the share of the standard’s pairs in the human-annotated disagreement subset.

<table><tr><td></td><td>Unanimous</td><td>Human</td></tr><tr><td>Pairs</td><td>11,161</td><td>8,837</td></tr><tr><td>COMPLIED</td><td>547</td><td>914</td></tr><tr><td>NOT COMPLIED</td><td>4,383</td><td>1,394</td></tr><tr><td>NOT APPLICABLE</td><td>6,231</td><td>6,529</td></tr></table>

Table 4: Class distribution by label source. The two subsets share no pair; the 2 remaining pairs are threeway analyst splits left unresolved and excluded from the human labels. Regulation texts average 44.3 words (55.1 BERT wordpieces) and disclosures 27.6 words (34.0 wordpieces).

## C Annotation Cost

Token usage and per-call latency were not logged during annotation, so Table 5 estimates cost from the exact annotation prompt of Appendix E.1 at each provider’s list price (per million input/output tokens: GPT-4.1 USD 2/8, Claude Sonnet 4 USD 3/15, Gemini 2.5 Pro USD 1.25/10).

<table><tr><td>Model</td><td>USD per 1,000 pairs</td></tr><tr><td>GPT-4.1 Claude Sonnet 4</td><td>0.89</td></tr><tr><td>Gemini 2.5 Pro</td><td>1.41 0.68</td></tr><tr><td>Panel total</td><td>2.97</td></tr></table>

Table 5: Estimated annotation cost at list prices: 344 input tokens per call on average (254 of them the system prompt) and 25 output tokens, counted with the o200k tokenizer on the exact prompt over all 20,000 pairs. Excludes structured-output schema overhead and Gemini 2.5 Pro reasoning tokens, so it is a lower bound.

## D Document Parsing Pipeline

Raw PDF documents, specifically IFRS standards issued by the IASB and corporate annual financial statements filed on SEDAR+, are converted to structured data through a three-stage pipeline: cloud-based PDF parsing, JSON refinement, and LLM-driven checklist extraction. Algorithm 1 summarizes the full document-to-pair construction process. Initial pairing uses the sentence encoder ϕ of Section 3.4.

## D.1 PDF Ingestion via LlamaParse

Parsing uses LlamaParse (LlamaIndex, 2024), a cloud-based document parser, in its agent-assisted mode. An auxiliary LLM (GPT-4.1-mini) interprets complex layouts (multi-page tables, nested lists, footnote structures) that are pervasive in both IFRS standards and note disclosures. Table 6 lists the configuration.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>parse_mode</td><td>parse_page_with_agent</td></tr><tr><td>model</td><td>openai-gpt-4-1-mini</td></tr><tr><td>high_res_ocr</td><td>True</td></tr><tr><td>adaptive_long_table</td><td>True</td></tr><tr><td>outlined_table_extraction</td><td>True</td></tr><tr><td>output_tables_as_HTML</td><td>True</td></tr></table>

Table 6: LlamaParse configuration for PDF ingestion.

Algorithm 1 Document-to-Pair Construction   
Pipeline   
Input: IFRS PDFs $\mathcal { P } _ { r }$ , financial statement PDFs   
$\mathcal { P } _ { d }$   
Output: Candidate pairs $\mathcal { D } = \{ ( r _ { i } , d _ { j } ) \}$   
1: for each $p \in \mathcal { P } _ { r } \cup \mathcal { P } _ { d }$ do   
2: struct<sub>p</sub> ← REFINE(LLAMAPARSE(p))   
3: $\textstyle { \mathcal { R } } \gets \bigcup _ { p \in { \mathcal { P } } _ { r } }$ EXTRACT(struct<sub>p</sub>, π<sub>reg</sub>)   
4: $\mathcal { F } \gets \bigcup _ { p \in \mathcal { P } _ { d } } \mathrm { E X T R A C T } ( \mathrm { s t r u c t } _ { p } , \pi _ { \mathrm { f s } } )$   
5: $\mathbf { E } _ { \mathcal { R } }  \dot { \phi } ( \mathcal { R } )$   
6: for each $d _ { j } \in \mathcal { F }$ do   
7: cand $j \gets \mathrm { T o p K } \big ( \cos ( \phi ( d _ { j } ) , \mathbf { E } _ { \mathcal { R } } ) , 1 5 \big )$   
8: pairs<sub>j</sub> ← SAMPLE(cand<sub>j</sub>, 5)   
9: return $\mathcal { D } = \cup _ { j } \mathrm { p a i r s } _ { j }$

## D.2 JSON Refinement

LlamaParse returns a verbose JSON structure with rendering metadata. A deterministic refinement step normalises this to a simplified schema: each page becomes an ordered list of typed items (heading, paragraph, table), discarding layout coordinates while preserving document hierarchy and table cell structure.

## D.3 Checklist Extraction and Pair Construction

Two LLM-driven extraction prompts (to be released with the code) are applied to the refined JSON:

1. The regulatory checklist extractor processes each IFRS standard, yielding 3,489 regulation items with paragraph-level references and verbatim requirement text.

2. The financial statement extractor processes each financial statement section, yielding 39,404 disclosure items preserving original IFRS/note references.

Initial pairing uses the Stage-1 domain-adapted BERT encoder (bert-base-cased continuedpretrained via MLM on SEC EDGAR 10-K MD&A; §3.4, Stage 1) for dense retrieval: each disclosure is encoded into a normalized 768- dimensional vector, and disclosures are matched against regulations via cosine similarity. The top-15 matches per disclosure are retrieved and sampled to the top-5, producing the 20,000 candidate pairs that enter the annotation pipeline. The later TSDAE/SimCSE and supervised contrastive stages cannot be used here because they require labeled regulation–disclosure pairs that do not yet exist at this point in the pipeline.

## E Prompt Templates

The annotation prompt is reproduced verbatim; the two extraction prompts are released with the code. Placeholders are shown in {braces}.

## E.1 IFRS Compliance Annotation Prompt

Each regulation–disclosure pair is presented to all three annotators (GPT-4.1, Claude Sonnet 4, Gemini 2.5 Pro) using the prompt below. The prompt enforces a three-class label with per-class confidence scores returned as a JSON object, parsed via provider-native structured-output enforcement (OpenAI function calling, Anthropic tool use, Gemini controlled generation). Temperature is set to 0 for all providers to ensure deterministic outputs. The user message consists of two fields: requirement: {regulation\_text} and document: {disclosure\_text}.

## System Prompt: IFRS Compliance Annotation

You are an expert auditor with perfect knowledge of the IFRS accounting standards. You must assess whether the provided financial statement passage complies with the given regulatory requirement.

Assume that only the provided passage is available for evaluation. If any required disclosure is missing from the passage, it must be considered not complied with.

• Answer “yes” if the passage fully satisfies all aspects of the requirement.

• Answer “no” if any part of the requirement is missing, incomplete, or not satisfied in the passage.

• Answer “not applicable” if the requirement is not relevant to the passage.

In addition to the selected answer, output a confidence score for each possible answer option.

## Output Requirements

• Output must be valid JSON only.

• The confidence score must be a numeric value between 0 and 1 (inclusive).

• The “confidence” field must be a list of three numeric values between 0 and 1 (inclusive).

• The confidence scores must correspond exactly and in this order to: ["yes", "no", "not applicable"]

• Higher values indicate higher confidence for that option.

The output must strictly conform to the following JSON   
format:   
{"answer": <"yes"|"no"|"not applicable">,   
"confidence": [number, number, number]}

## F EM Joint Calibration: Algorithm Details

The EM joint calibration procedure alternates between two steps.

Calibration step. Each annotator model m has a temperature parameter $T _ { m }$ optimized by minimizing the cross-entropy between $\sigma ( \mathbf { p } _ { m , i } / T _ { m } )$ and the current soft label estimate $\hat { \mathbf { y } } _ { i }$ over all items i. Minimization uses SciPy minimize\_scalar with the bounded method and bounds $T _ { m } \in [ 0 . 1 , 2 0 . 0 ]$ (see Appendix R for sensitivity analysis).

Label refinement step. Soft label estimates are updated as the normalized element-wise product of calibrated model outputs:

$$
\hat { \mathbf { y } } _ { i } \propto \prod _ { m } \sigma ( \mathbf { p } _ { m , i } / T _ { m } ) .\tag{1}
$$

The class prior $\pi$ is initialized to $1 / K$ . On the full 20,000-pair dataset under the adopted bound $T _ { m } \in [ 0 . 1 , 2 0 . 0 ]$ , the procedure converges in 20 iterations (threshold: max $\Delta < 1 0 ^ { - 5 } ) $ : GPT-4.1 fits at T=14.36 and Gemini 2.5 Pro at T=5.73, both indicating severe overconfidence in their raw probability outputs; Claude Sonnet 4 converges to T=0.55, slightly sharpening its already nearcalibrated raw outputs.

## G Per-Model Confusion Matrices

Tables 7–9 present the learned confusion matrices for each LLM annotator, evaluated against the hard-vote consensus on the full 20,000-pair dataset. These matrices reveal systematic prediction biases that motivate the EM calibration procedure (Section 3).

<table><tr><td></td><td>Pred: No</td><td>Pred: N/A</td><td>Pred: Yes</td></tr><tr><td>True: No</td><td>0.990</td><td>0.002</td><td>0.008</td></tr><tr><td>True: N/A</td><td>0.472</td><td>0.527</td><td>0.001</td></tr><tr><td>True: Yes</td><td>0.147</td><td>0.016</td><td>0.837</td></tr><tr><td>True: No</td><td>0.897</td><td>0.092</td><td>0.011</td></tr><tr><td>True: N/A</td><td>0.005</td><td>0.994</td><td>0.001</td></tr><tr><td>True: Yes</td><td>0.114</td><td>0.002</td><td>0.884</td></tr></table>

Table 7: GPT-4.1 confusion matrix vs. hard-vote consensus.

Table 8: Claude Sonnet 4 confusion matrix vs. hard-vote consensus.

<table><tr><td></td><td>Pred: No</td><td>Pred: N/A</td><td>Pred: Yes</td></tr><tr><td>True: No</td><td>0.703</td><td>0.250</td><td>0.047</td></tr><tr><td>True: N/A</td><td>0.003</td><td>0.996</td><td>0.002</td></tr><tr><td>True: Yes</td><td>0.060</td><td>0.016</td><td>0.925</td></tr></table>

Table 9: Gemini 2.5 Pro confusion matrix vs. hard-vote consensus.

## H Calibration Summary

Predicted confidence is compared to observed accuracy for each LLM annotator before and after EM calibration under the adopted bound $T _ { \mathrm { m a x } } { = } 2 0 . 0$ on the 8,837 human-annotated items. Three observations capture the calibration picture.

• Gemini 2.5 Pro shows substantial calibration improvement: ECE drops from 0.215 to 0.123 at fitted temperature T=5.73, with the postcalibration confidence-accuracy curve aligning closely to the diagonal.

• Claude Sonnet 4 is the best-calibrated annotator out of the box (raw ECE 0.211), and EM fits a sharpening temperature T=0.55 that raises its ECE to 0.276. This is the per-model cost of joint optimization: the temperature that best aligns Anthropic with the consensus soft labels over-sharpens its already nearcalibrated raw outputs.

• GPT-4.1 improves materially after calibration (ECE 0.811 to 0.508 at T=14.36) but remains far from the diagonal. The residual error is structural: GPT-4.1’s miscalibration reflects a class-level prediction bias (95.5% NOT COM-PLIED on the human-annotated disagreement subset, see Appendix N) rather than stochastic over- or under-confidence. Temperature scaling cannot correct a label-distribution shift.

## I Human Annotation Platform

Architecture. A purpose-built web application collects independent human judgments. The backend is a FastAPI service with MongoDB for storage; the frontend is a Next.js 16 application. Annotators are assigned items through a priority queue that routes disagreement cases (partial and threeway LLM splits) before unanimous items. Two senior annotators with domain expertise in IFRS reporting oversaw the process, resolving edge cases and ensuring labeling consistency.

Annotation schema. Each annotation captures four fields (Table 10); annotators are blind to LLM predictions:

<table><tr><td>Field</td><td>Type</td><td>Description</td></tr><tr><td>label</td><td>enum</td><td>COMPLIED, NOT COM- PLIED, or NOT APPLICA-</td></tr><tr><td>rationale</td><td>text</td><td>BLE Free-text justification (1-3 sentences)</td></tr><tr><td>time_spent</td><td>float</td><td>Auto-tracked annotation du- ration (seconds)</td></tr><tr><td>flagged_difficult bool</td><td></td><td>Annotator-flagged ambigu- ous items</td></tr></table>

Table 10: Human annotation schema.

Item selection and routing. All 8,795 partialagreement items (2 of 3 LLMs agree) and all 44 three-way disagreement items are prioritized for human annotation. Unanimous items (11,161) are excluded from human review and reserved for supervised training, so the LLM-unanimous pairs used for embedding fine-tuning are disjoint from the human-labeled disagreement set used for aggregation evaluation.

## J Data Partition Details

Table 11 reports the natural partition of the 20,000 LLM-labeled pairs by inter-annotator agreement. The 11,161 unanimous-agreement items (3 of 3 LLMs concur) form the supervised training pool for the Stage 3 contrastive head and never enter the human annotation queue. The 8,795 partialagreement items (2 of 3 concur) and the 44 threeway disagreement items (all three LLMs disagree) are routed to human annotators in priority order, yielding a routed pool of 8,839 items. The released labels cover 8,837 items (8,794 partial + 43 threeway); the two-item shortfall against the routed pool is one partial-agreement pair and one three-way pair on which the three analysts split three ways and which remained unresolved when the labeling round closed, so both are excluded (§3.3). By construction, every item used to train the embedding encoder is excluded from this 8,837-item evaluation set, so the embedding training pool and the classifier/aggregation evaluation set do not overlap.

<table><tr><td>Partition</td><td>Count</td><td>Labels</td><td>Used For</td></tr><tr><td>Unanimous (3/3)</td><td>11,161</td><td>0</td><td>Supervised training</td></tr><tr><td>Partial (2/3)</td><td>8,795</td><td>8,794</td><td>Evaluation</td></tr><tr><td>Three-way (1/1/1)</td><td>44</td><td>43</td><td>Evaluation</td></tr></table>

Table 11: Data partitions. Items where LLMs agreed unanimously were never sent to humans (0 human labels), keeping the embedding training pool disjoint from the human-labeled evaluation set. The one-label shortfall in each disagreement row (8,795 vs. 8,794; 44 vs. 43) reflects two routed pairs on which the three analysts split three ways and which remained unresolved when the labeling round closed; they are excluded from the released labels (§3.3).

## K Embedding Pipeline Hyperparameters

Table 12 reports complete hyperparameters for each stage of the three-stage embedding pipeline (Section 3.4). All stages train on a single RTX 5070 GPU. The longest stage (MLM) completes in approximately 11 hours on the 1.4M-sentence EDGAR-MD&A corpus, while TSDAE, SimCSE, and supervised contrastive each complete in under 90 minutes on the 42,893-sentence adaptation set and 4,930-pair supervised set respectively. The cased tokenizer is retained to preserve capitalization-sensitive acronyms (IFRS, IAS, GAAP) throughout the pipeline.

## L Extended Retrieval Results

Tables 13 and 14 report retrieval performance across all five embedding checkpoints on seen and unseen regulations respectively. Three observations stand out. First, SimCSE + Supervised is the strongest model on unseen regulations across every metric (R@15 = 45.3, MAP@15 = 0.125, NDCG@15 = 0.188), making it the preferred encoder for zero-shot retrieval generalization. Second, plain SimCSE without supervised fine-tuning under-performs the MLM-only baseline on seen regulations (R@15 = 15.8 vs. 20.9), indicating that contrastive dropout-noise alone does not generalize the encoder beyond its MLM initialization when training on a small in-domain corpus. Third, TS-DAE + Supervised wins on R@5 (both seen and unseen), suggesting that denoising auto-encoding produces more discriminative top-ranked candidates while SimCSE produces a flatter but more recallfavorable ranking.

## M Label Source Ablation: Extended Results

Training stage 3 on 4,930 LLM-unanimous pairs yields overlapping 95% bootstrap confidence intervals with training on 1,616 human-annotated pairs (Table 15). Since the LLM pairs are a zero-cost byproduct of the annotation pipeline, this supports using them as a practical alternative to expert labeling in stage 3.

<table><tr><td>Model</td><td>Labels</td><td>Seen@5</td><td>Unseen@15</td></tr><tr><td>TSDAE+Sup</td><td>Human (1.6k)</td><td>6.2</td><td>36.0</td></tr><tr><td>TSDAE+Sup</td><td>LLM (4.9k)</td><td>4.7</td><td>40.0</td></tr><tr><td>SimCSE+Sup</td><td>Human (1.6k)</td><td>7.8</td><td>32.0</td></tr><tr><td>SimCSE+Sup</td><td>LLM (4.9k)</td><td>5.7</td><td>48.0</td></tr></table>

Table 15: Label source results on shared test set (1,326 items). All CIs overlap between LLM and Human rows.

Table 16 isolates the effect of supervised label source on the final retrieval encoder. The test set is held fixed (1,326 items, 192 seen regulations, 25 unseen regulations) while the Stage 3 supervised contrastive head is retrained on two label sources: (i) 1,616 human-validated positive pairs from the 8,837 human-annotated items, and (ii) 4,930 LLMunanimous positive pairs harvested from the 11,161 unanimous-agreement subset.

Two findings stand out. First, neither label source dominates: human labels score higher on seen regulations, while LLM-unanimous labels score higher on unseen R@15 for both backbones (48.0 vs. 32.0 for SimCSE + Supervised; 40.0 vs. 36.0 for TSDAE + Supervised). Second, the gap is largest precisely where data scarcity hurts most: on unseen regulations, where the model must generalize beyond its training distribution. The 3.05× larger positive-pair budget from LLM unanimity dominates whatever label-noise penalty unanimous LLM agreement carries. Because LLM unanimity is a zero-cost byproduct of the annotation pipeline while human annotation requires paid labor, this result supports weak supervision as a practical alternative to expert labeling in this domain when three diverse LLM judges agree.

## N Per-Class Detailed Metrics

Table 17 decomposes each hard predictor’s performance by the three target classes. Three failure patterns emerge. (1) OpenAI’s “yes” row is near empty: only 233 of 8,837 predictions are positive, yielding undefined precision and recall under the macro definition. Its “no” prediction is highrecall (0.98) but low-precision (0.16), confirming the 95.5% over-prediction of NOT COMPLIED reported in Section 4.2. (2) Anthropic is balanced across classes but uniformly low on the minority “yes” class (F1 = 0.20). (3) Gemini achieves the strongest per-class F1 on “yes” (0.59) and “not applicable” (0.85) but degrades on “no” (0.16), mirroring the inverse of OpenAI’s bias. Hard vote partially compensates by averaging the biases (yes F1 = 0.42, no F1 = 0.32, n/a F1 = 0.78), but no hard predictor exceeds the learned ensemble classifier (Section 3.5) on any class. These class-specific failure modes are invisible at the aggregate W-F1 level and motivate the per-class oracle ceiling reported in Table 18.

<table><tr><td>Parameter</td><td>Stage 0 (MLM)</td><td>Stage 1A (TSDAE)</td><td>Stage 1B (SimCSE)</td><td>Stage 2 (Supervised)</td></tr><tr><td>Base model</td><td>bert-base-cased</td><td>Stage 0 output</td><td>Stage 0 output</td><td>Stage 1 output</td></tr><tr><td>Epochs</td><td>3</td><td>1</td><td>1</td><td>4</td></tr><tr><td>Batch size</td><td>4 (accum. 4)</td><td>8</td><td>128</td><td>16</td></tr><tr><td>Learning rate</td><td>5e-5</td><td>3e-5</td><td>3e-5</td><td>2e-5</td></tr><tr><td>fp16</td><td>Yes</td><td>No</td><td>Yes</td><td>Yes</td></tr><tr><td>Warmup</td><td></td><td>10%</td><td>10%</td><td>10%</td></tr><tr><td>Temperature (τ)</td><td></td><td></td><td>0.05</td><td>0.05</td></tr><tr><td>Token deletion</td><td>15% (MLM)</td><td>60% (TSDAE)</td><td></td><td></td></tr><tr><td>Tied weights Grad. checkpoint</td><td>No</td><td>Yes No</td><td>Yes</td><td></td></tr><tr><td>Training data</td><td>EDGAR 10-K</td><td></td><td>42,893 sentences</td><td>No</td></tr><tr><td></td><td></td><td>42,893 sentences</td><td></td><td>4,930 pos. pairs</td></tr><tr><td>Loss</td><td>MLM</td><td>Denoising AE</td><td>MNRL</td><td>MNRL</td></tr></table>

Table 12: Hyperparameters for each stage of the embedding pipeline. MNRL = MultipleNegativesRankingLoss. EDGAR 10-K sentences are filtered to 150–400 characters from Section 7 (MD&A) of annual filings (1993–2020).
<table><tr><td>Model</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@15</td><td>MRR</td><td>MAP@15</td><td>NDCG@15</td></tr><tr><td>MLM only</td><td>2.5</td><td>8.9</td><td>17.0</td><td>20.9</td><td>0.060</td><td>0.034</td><td>0.059</td></tr><tr><td>+ TSDAE</td><td>2.1</td><td>7.7</td><td>14.5</td><td>19.8</td><td>0.053</td><td>0.034</td><td>0.058</td></tr><tr><td>+ SimCSE</td><td>1.5</td><td>7.4</td><td>11.3</td><td>15.8</td><td>0.046</td><td>0.026</td><td>0.046</td></tr><tr><td> $+ \mathrm { T S D A E } + \mathrm { S u p }$ </td><td>2.5</td><td>11.9</td><td>22.1</td><td>27.5</td><td>0.080</td><td>0.050</td><td>0.084</td></tr><tr><td> $+ \thinspace \mathrm { S i m C S E } + \mathrm { S u p }$ </td><td>2.5</td><td>11.3</td><td>23.2</td><td>26.4</td><td>0.073</td><td>0.042</td><td>0.077</td></tr></table>

Table 13: Seen-regulation retrieval metrics.

Hardest standards. Per-standard scores use peritem out-of-fold predictions of the reported classifier configuration, re-run to save them (W-F1 0.813; over five XGBoost seeds 0.814, s.d. 0.001). This rerun does not reproduce the reported 0.817 exactly. Weighted F1 is lowest on IFRS 6 (0.735), IFRS 3 (0.772), IFRS 13 (0.776), IFRS 5 (0.779) and IFRS 2 (0.780). NOT COMPLIED is the weakest class throughout; among standards with at least 50 NOT COMPLIED items its F1 is lowest on IFRS 3 (0.42), IFRS 2 (0.44), IFRS 16 (0.46) and IFRS 5 (0.46). The rarest class is not the hardest: COM-PLIED, 10.3% of the human labels, reaches F1 0.71 against 0.50 for NOT COMPLIED.

## O Aggregation Methods: Extended Benchmark

Table 18 reports the full classification benchmark on 8,837 human-annotated items, ranked by W-F1. The table extends Section 4.2 (Table 2) with macro-F1 and Brier-score columns, and adds the per-item oracle (theoretical ceiling) and the learned ensemble classifier (Section 3.5) for direct comparison. The 11 aggregation strategies span majority voting (Hard/Soft Vote), confusionmatrix EM (Dawid–Skene, One-Coin DS), competence estimation (MACE, M-MSR, GLAD, Wawa), classifier-augmented consensus (CROWDLAB), and temperature-scaled EM under both bound choices (Section 3.6). Dashes indicate metrics not produced by the method (e.g., MACE, CROWD-LAB, and One-Coin DS output discrete labels rather than calibrated class probabilities, so ECE and Brier are undefined).

## P Semi-Supervised EM

This appendix tests whether clamping human gold labels during the EM label-refinement step improves calibration. Gold items are held fixed during the E-step; the calibration step optimizes temperatures using a mixture of inferred and known targets. Across gold budgets of 100, 500, and 1,000 items, W-F1 stays at 0.672–0.673 (indistinguishable from unsupervised EM) and ECE remains at 0.290–0.293. At budgets of 2,000 and 5,000, performance collapses: W-F1 drops to 0.582 and 0.513 respectively, with ECE rising to 0.322 and 0.364. Anthropic’s fitted temperature drifts from a well-calibrated T=0.55 at zero budget upward to T=1.45 at budget 2,000 and T=4.69 at budget 5,000.

<table><tr><td>Model</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@15</td><td>MRR</td><td>MAP@15</td><td>NDCG@15</td></tr><tr><td>MLM only</td><td>3.8</td><td>11.3</td><td>21.7</td><td>28.3</td><td>0.086</td><td>0.076</td><td>0.118</td></tr><tr><td>+ TSDAE</td><td>0.9</td><td>12.3</td><td>22.6</td><td>26.4</td><td>0.069</td><td>0.064</td><td>0.106</td></tr><tr><td>+ SimCSE</td><td>0.9</td><td>12.3</td><td>15.1</td><td>27.4</td><td>0.078</td><td>0.070</td><td>0.110</td></tr><tr><td> $+ \mathrm { T S D A E } + \mathrm { S u p }$ </td><td>4.7</td><td>18.9</td><td>32.1</td><td>41.5</td><td>0.122</td><td>0.110</td><td>0.171</td></tr><tr><td> $+ \thinspace \mathrm { S i m C S E } + \mathrm { S u p }$ </td><td>3.8</td><td>20.8</td><td>33.0</td><td>45.3</td><td>0.137</td><td>0.125</td><td>0.188</td></tr></table>

Table 14: Unseen-regulation retrieval metrics. SimCSE + Supervised achieves the strongest zero-shot generalization.

<table><tr><td>Model</td><td>Labels (n)</td><td>Seen R@5/R@15</td><td>Unseen R@5/R@15</td></tr><tr><td>TSDAE+S</td><td>Human (1.6k)</td><td>6.2/14.6</td><td>8.0/36.0</td></tr><tr><td>TSDAE+S</td><td>LLM (4.9k)</td><td>4.7/12.0</td><td>16.0/40.0</td></tr><tr><td>SimCSE+S</td><td>Human (1.6k)</td><td>7.8/15.6</td><td>20.0/32.0</td></tr><tr><td>SimCSE+S</td><td>LLM (4.9k)</td><td>5.7/13.5</td><td>12.0/48.0</td></tr></table>

Table 16: Label-source ablation on a fixed test set (1,326 items, 192 seen regs, 25 unseen regs). Each cell reports R@5 / R@15 (%). 95% bootstrap CIs omitted for space.
<table><tr><td>Predictor</td><td>Class</td><td>Prec.</td><td>Rec.</td><td>F1</td></tr><tr><td>OpenAI</td><td>yes</td><td></td><td></td><td></td></tr><tr><td>OpenAI</td><td>no</td><td>0.16</td><td>0.98</td><td>0.28</td></tr><tr><td>OpenAI</td><td>not applicable</td><td>0.80</td><td>0.02</td><td>0.04</td></tr><tr><td>Anthropic</td><td>yes</td><td>0.14</td><td>0.35</td><td>0.20</td></tr><tr><td>Anthropic</td><td>no</td><td>0.29</td><td>0.27</td><td>0.28</td></tr><tr><td>Anthropic</td><td>not applicable</td><td>0.83</td><td>0.82</td><td>0.83</td></tr><tr><td>Gemini</td><td>yes</td><td>0.73</td><td>0.49</td><td>0.59</td></tr><tr><td>Gemini</td><td>no</td><td>0.21</td><td>0.13</td><td>0.16</td></tr><tr><td>Gemini</td><td>not applicable</td><td>0.80</td><td>0.90</td><td>0.85</td></tr><tr><td>Hard Vote</td><td></td><td>0.79</td><td>0.28</td><td>0.42</td></tr><tr><td>Hard Vote</td><td>yes no</td><td>0.23</td><td>0.50</td><td>0.32</td></tr><tr><td>Hard Vote</td><td>not applicable</td><td>0.85</td><td>0.72</td><td>0.78</td></tr></table>

Table 17: Per-class precision, recall, and F1 for all predictors on 8,837 human-annotated items. OpenAI “yes” metrics are near-zero (only 233 of 8,837 predictions are “yes”).

Root cause. All 8,837 gold labels are on LLMdisagreement items, the hardest cases. At small budgets, clamped items are a negligible fraction of the M-step loss. At budgets ≥ 2, 000, clamped items dominate and the M-step is forced to find temperatures consistent with hard-case ground truth, which pushes Anthropic’s well-calibrated temperature toward over-flattening. Representative random sampling across all agreement tiers would be required for gold anchoring to help. The main text therefore keeps the unsupervised-only EM.

## Q Classifier Ablations

Architecture comparison (probability-only features, 29-d). Four classifier architectures are compared over the 29-dimensional probability feature vector: confidence-weighted selection, stacked generalization, direct 3-class XGBoost, and 4-class model selection. All four saturated within a 0.003 W-F1 band of 0.727 under 5-fold cross-validation on the 8,837-item set, confirming that the probability features alone carry limited routing signal regardless of architecture choice.

Embedding encoder comparison (hybrid features). Adding 1,536-d frozen SBERT embeddings to the 29-d probability features lifts performance by 8 points (from 0.727 to 0.808 W-F1). Within the family of compatible encoders tested (TSDAE, BERT-SEC MLM, TSDAE + Supervised, and vanilla bert-base-cased), all hybrid variants fell within a 0.4-point W-F1 band, suggesting that at this scale the choice of base encoder is dominated by the contribution of the embedding modality itself rather than the specific adaptation strategy. Notably, TSDAE-unsupervised (0.808) edged out TSDAE+Supervised (0.804), suggesting that the supervised contrastive objective optimizes for retrieval similarity rather than for the discriminative features the classifier requires; the TSDAE Stage-2A (unsupervised) checkpoint is therefore the encoder used as the frozen feature extractor in the headline learned-ensemble result. The full 5- checkpoint encoder sweep including the SimCSE family is left to follow-up work.

Optimization. Optuna hyperparameter search (TPE, 100 trials) on the XGBoost head, targeting the NOT COMPLIED bottleneck identified in Appendix N, achieved 0.817 W-F1, the headline classification result. Class-weighting alone (without hyperparameter search) reaches 0.808. Augmenting training with the 11,161 LLM-unanimous items as silver labels does not help (0.810). Full search trajectories will be included in the code release.

<table><tr><td>Strategy</td><td>Acc</td><td>W-F1</td><td>M-F1</td><td>κ</td><td>ECE</td><td>Brier</td></tr><tr><td>Oracle (ceiling)</td><td>0.952</td><td>0.949</td><td>0.900</td><td>0.882</td><td>0.031</td><td>0.102</td></tr><tr><td>Learned Ensemble (proposed)</td><td>0.824</td><td>0.817</td><td></td><td>0.556</td><td>0.068</td><td>一</td></tr><tr><td>MACE (Hovy et al., 2013)</td><td>0.748</td><td>0.722</td><td>0.561</td><td>0.312</td><td></td><td></td></tr><tr><td>CROWDLAB (Goh et al., 2022)</td><td>0.791</td><td>0.720</td><td></td><td>0.340</td><td></td><td></td></tr><tr><td>Gemini single (Google DeepMind, 2025)</td><td>0.737</td><td>0.713</td><td>0.532</td><td>0.278</td><td>0.215</td><td>0.478</td></tr><tr><td>M-MSR (Ma and Olshevsky, 2020)</td><td>0.672</td><td>0.687</td><td></td><td>0.256</td><td></td><td></td></tr><tr><td>Anthropic single (Anthropic, 2025)</td><td>0.672</td><td>0.687</td><td>0.499</td><td>0.256</td><td>0.211</td><td>0.579</td></tr><tr><td>One-Coin DS (Zhang et al., 2016)</td><td>0.668</td><td>0.679</td><td></td><td>0.238</td><td></td><td></td></tr><tr><td>Soft Vote</td><td>0.642</td><td>0.672</td><td>0.507</td><td>0.255</td><td>0.290</td><td>0.538</td></tr><tr><td>EM Calibrated  $( T _ { \mathrm { m a x } } { = } 2 0 )$  (Guo et al., 2017)</td><td>0.647</td><td>0.672</td><td></td><td>0.245</td><td>0.292</td><td>一</td></tr><tr><td>Hard Vote</td><td>0.641</td><td>0.671</td><td>0.505</td><td>0.254</td><td></td><td></td></tr><tr><td>Dawid–Skene (Dawid and Skene, 1979)</td><td>0.641</td><td>0.671</td><td>0.504</td><td>0.254</td><td></td><td></td></tr><tr><td>Wawa (Ustalov et al., 2024)</td><td>0.642</td><td>0.671</td><td></td><td>0.254</td><td></td><td>一</td></tr><tr><td>GLAD (Whitehill et al., 2009)</td><td>0.642</td><td>0.671</td><td></td><td>0.254</td><td></td><td></td></tr><tr><td>EM Calibrated  $( T _ { \mathrm { m a x } } { = } 5 ,$  default)</td><td>0.456</td><td>0.503</td><td>0.381</td><td>0.149</td><td>0.461</td><td>0.980</td></tr><tr><td>GPT-4.1 single (OpenAI, 2025)</td><td>0.183</td><td>0.095</td><td>0.198</td><td>0.019</td><td>0.811</td><td>1.621</td></tr></table>

Table 18: Classification benchmark on 8,837 human-annotated items, ranked by W-F1. Bold row: the learned ensemble classifier (Section 3.5). EM Calibrated under the textbook $T _ { \mathrm { m a x } } { = } 5$ bound is included to make the 17-point bound penalty visible (Appendix R).

Agreement with unanimous LLM labels. Trained only on disagreement items, the classifier agrees with the LLM labels on the 11,161 unanimous items at 0.624 W-F1 (accuracy 0.663). On these items the agreement features take values absent from training, so the score reflects distribution shift as much as label quality; with no human labels on this subset, it measures agreement with the LLM label, not accuracy.

Fine-tuned BERT baseline (rejected). End-toend fine-tuning of the domain-adapted BERT (3 epochs, batch 16) yields 0.755 W-F1, below the frozen hybrid at 0.808; with ∼7K training items per fold, frozen embeddings plus gradient boosting is more data-efficient.

## R EM Temperature Bound Sensitivity

The choice of upper bound for the per-model temperature parameter is the single largest non-data factor in EM calibration accuracy. Table 19 reports W-F1 and ECE under three bounds. The conventional textbook bound of $T _ { \mathrm { m a x } } { = } 5 . 0$ , used in most temperature-scaling literature, is binding for both OpenAI (T=5.0 at bound) and Gemini (T=5.0 at bound). Forcing two of three annotators against the bound suppresses the differential reweighting that EM relies on, collapsing W-F1 to 0.503, well below soft vote (0.672) and below every individual aggregator in Table 18. Relaxing the bound to $T _ { \mathrm { m a x } } { = } 2 0 . 0$ lets OpenAI settle at T=14.36 and Gemini at T=5.73, recovering W-F1 to 0.672 (matching soft vote) and reducing ECE from 0.461 to 0.292. The bound [0.1, 20.0] is adopted throughout the main results (Section 3, Appendix F). The 17-point W-F1 gap between the two bound choices is the silent EM temperature-bound error documented as a diagnostic of multi-LLM aggregation in this work.

<table><tr><td> $T _ { \mathrm { m a x } }$ </td><td> $T _ { \mathbf { o a i } }$ </td><td> $T _ { \mathbf { a n t } }$ </td><td> $T _ { \mathbf { g e m } }$ </td><td>W-F1</td><td>ECE</td></tr><tr><td>5.0</td><td>5.00*</td><td>0.95</td><td>5.00*</td><td>0.503</td><td>0.461</td></tr><tr><td>10.0</td><td>10.00*</td><td>0.66</td><td>5.84</td><td>0.642</td><td>0.301</td></tr><tr><td>20.0</td><td>14.36</td><td>0.55</td><td>5.73</td><td>0.672</td><td>0.292</td></tr></table>

Table 19: Temperature bound sensitivity. <sup>⋆</sup> marks fitted temperatures pinned at the upper bound. The [0.1, 5.0] bound introduces a 17-point W-F1 penalty by constraining two of three models to the bound.