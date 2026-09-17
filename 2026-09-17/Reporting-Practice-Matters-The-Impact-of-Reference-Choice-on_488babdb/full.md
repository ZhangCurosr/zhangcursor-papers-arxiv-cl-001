# Reporting Practice Matters: The Impact of Reference Choice on Chest X-ray Report Evaluation

Daniel P. Jeong<sup>1</sup>, Charles Q. Li<sup>2</sup>, Hossein Hosseiny<sup>3</sup>, Nitya M. Bhalla<sup>2</sup>, Fatma Uyar Morency<sup>3</sup>, Pradeep Ravikumar<sup>1</sup>, Zachary C. Lipton<sup>1</sup>, Michael Oberst<sup>4</sup> <sup>1</sup> Machine Learning Department, Carnegie Mellon University

<sup>2</sup> Department of Radiology and Imaging Sciences, Allegheny Health Network 3 Data Science R&D, Highmark Health Enterprise Data & Analytics <sup>4</sup> Department of Computer Science, Johns Hopkins University

## Abstract

Radiologists follow heterogeneous reporting practices. Two radiologists examining the same image and identifying the same clinicalfindings might nevertheless compose superficially distinct reports, varying in terminology, shorthand, formatting, and level of detail. These variations in reporting norms represent an under-appreciated obstacle in efforts to evaluate AI-based radiology report generation (RRG) models, where machine-generated reports are typically assessed based on their concordance with human-generated references. In this paper, we quantify the sensitivity of established evaluation metrics to variations in reporting practices, revealing impacts large enough to alter the rankings of models. We introduce a radiologist-informed taxonomy of variations in radiology reporting practice and a method (REREF) that rewrites reference reports along the axes of our taxonomy while preserving clinical interpretation. For instance, when comparing the performance of nine RRG models on MIMIC-CXR using RadCliQ-v1, condensing the discussion of normal findings in the reference reports causes Libra to drop from first to second place while CheXOne rises from third to first. Our results suggest that many current metrics fail to decouple clinical interpretation from conformity to reporting practices and that choosing the “right” references that accurately reflect the desired reporting practices can be important in practice. To support future research, we release MIMIC-CXR-EXT-REREF, a radiologist-validated dataset of 120 (original, alternative) reference report pairs derived from MIMIC-CXR.

## 1 Introduction

Automated generation of radiology reports, the primary medium through which radiologists document their interpretation of medical images (e.g., chest X-ray (CXR), computed tomography (CT) scans), has the potential to improve the efficiency of radiologist workflows, helping reduce physician burnout and improve patient care. To realize this potential, several works propose to train vision-language models (VLMs) that are capable of generating a text report summarizing the key clinical findings (i.e., the Findings and/or Impression sections of a radiology report) when provided with the radiological images (Liu et al., 2026; Zhou et al., 2026; Bannur et al., 2024). While some studies suggest that they are starting to show promise in assisting radiologists under controlled setups (Tanno et al., 2024; Huang et al., 2025), recent benchmarks indicate that the current crop of models are not yet reliable or accurate enough for deployment in real-world settings (Zhang et al., 2026a, 2025c).

Given the free-form nature of radiology reports, a major challenge to model development lies in designing evaluation methods that reliably characterize the clinical accuracy and utility of modelgenerated reports. As manual review by a radiologist is not scalable, a common approach is to employ an automated evaluation metric (e.g., GREEN (Ostmeier et al., 2024), RadCliQ-v1 (Yu et al., 2023), RadGraph-F1 (Delbrouck et al., 2022)) that compares the model-generated report to a reference report, written independently by a radiologist, and outputs a numerical score (e.g., between 0 and 1). State-of-the-art metrics are claimed to correlate well with radiologist judgment of report quality, e.g., as measured against radiologist annotations in datasets like ReXVal (Yu et al., 2023).

![](images/ef084dc71143a14360a17c2ed9ed3cd2aaf113452986a1408bcf385e4f4f32b7.jpg)  
Figure 1: Motivation and overview. (Left) Automated evaluation metrics for radiology report generation (RRG) models can be sensitive to the reporting practices of the reference: when individually compared against two references that communicate identical clinical findings but follow distinct reporting practices, a model-generated report may score higher against one vs. the other by better conforming to the reporting practices of the former (e.g., enumerating all normal findings). (Right) To study how these variations impact the conclusions of RRG model evaluation, we propose a method (REREF; Section 2.2) that rewrites a reference report to follow alternative reporting practices while preserving clinical interpretation, and quantify the impact of each perturbation on the model rankings.

However, radiologists follow highly heterogeneous reporting practices, often communicating interpretations with similar clinical impact in varying language and levels of detail (Weiss and Langlotz, 2008; Bosmans et al., 2014; Delbrouck et al., 2025), which raises concerns about whether current evaluation approaches are sensitive to these differences. For instance, a model-generated report that enumerates normal findings (e.g., “The lungs are clear”, “There is no pleural effusion”) may receive a higher score when evaluated against a similarly enumerated reference (e.g., “Lungs are clear”, “No pleural effusion or pneumothorax evident”), than against a more abstract and lexically different reference report (“No acute cardiothoracic abnormality”) (Figure 1).

Motivated by such observations, we investigate how variations in reporting practice can impact the conclusions about radiology report generation (RRG) model performance, using nine open-source models and four datasets for CXRs. We construct a taxonomy that characterizes the salient ways in which radiology reporting practice differs (e.g., report structure, level of detail, terminology preferences) (Section 2.1). We then develop REREF, an expert-validated method that uses a large language model (LLM) to rewrite reference reports along the axes of our taxonomy while preserving the overall clinical interpretation (Sections 2.2–2.3, 4.1). We use REREF to perturb the reference reports of all datasets to study how reporting practice variations impact model performance (Section 3).

Overall, we find that many existing metrics are sensitive to such variations and fail to decouple clinical interpretation from conformity to reporting practices (Section 4.2), large enough to alter the rankings of models (Section 4.3). For example, when comparing all models on MIMIC-CXR (Johnson et al., 2019) with RadCliQ-v1, condensing the reference reports to focus on the key abnormal findings causes Libra (Zhang et al., 2025b) to drop from first to second place in model ranking, while CheXOne (Zhang et al., 2026b) rises from third to first place. Our findings suggest that, in practice, selecting references that best-reflect the desired reporting practices for the setting of interest can be an important design decision for model evaluation. We also release MIMIC-CXR-EXT-REREF, a radiologist-validated dataset of 120 (original, alternative) report pairs derived from MIMIC-CXR.

Our main contributions can be summarized as follows:

1. We present a taxonomy that systematically captures the variations in radiology reporting practice, designed under the guidance of a board-certified radiologist (Section 2.1).

2. We develop REREF, a method for rewriting a given reference report along each axis of our taxonomy (Section 2.2), and validate it via radiologist annotations (Sections 2.3, 4.1).

3. We propose an evaluation framework based on REREF to study how variations in radiology reporting practice impact the conclusions about RRG model performance (Section 3).

4. We find that many current metrics for RRG model evaluation fail to decouple clinical interpretation from conformity to reporting practices of the reference (Sections 4.2–4.3).

5. We release MIMIC-CXR-EXT-REREF, a radiologist-validated dataset of 120 (original, alternative) reference report pairs derived from MIMIC-CXR to support future research.

To ensure the reproducibility of our results, we open-source the source code used for REREF and for all of our evaluations described below via our GitHub repository<sup>1</sup>.

## 2 Categorizing and Simulating Reporting Practice Variation

To study how reporting practice variation impacts the evaluation of RRG models, we first construct a taxonomy of such variation under the guidance of a board-certified radiologist (Section 2.1). We then introduce REREF, an LLM-based method that rewrites a reference report along a chosen axis of our taxonomy (Section 2.2). Finally, we conduct an annotation study to validate that REREF produces alternatives that are realistic (i.e., plausibly written by a radiologist) and clinically equivalent to the original reference (i.e., overall diagnosis and implied management remain unchanged) (Section 2.3).

## 2.1 Taxonomy of Reporting Practice Variation

We categorize the variations in radiology reporting practice along four dimensions—Organization, Completeness, Granularity, and Language—where each comprises one or more specific axes along which radiology reports often differ without substantial change in clinical interpretation. For example, under Granularity, we define the Anatomical Granularity and Quantitative Granularity axes, which each capture a specific way in which radiologists can control the level of detail in their reports.

Organization. Organization captures how report content is arranged at the report level and comprises two axes: Structure and Section Assignment. Structure contrasts highly templated formats—e.g., use of explicit anatomical headings, bullet points, numbered lists, subheadings for groups of findings (Structure (Structured))—with continuous free-text prose (Structure (Free-Text)). Section Assignment captures the variations in how radiologists allocate content to different sections of a radiology report. For example, comparisons to prior studies and differential diagnoses may be discussed in the Findings, in the Impression, or in both, depending on the radiologist’s preference.

Completeness. Completeness captures how exhaustively a report describes the findings visible in a study. We focus on a specific instantiation of this dimension, Findings Scope, which contrasts an exhaustive style that enumerates all normal and abnormal findings (Findings Scope (Exhaustive)), against a minimal style that selectively omits normal findings and reports only a focused list of abnormalities most likely to be clinically relevant (Findings Scope (Minimal)).

Granularity. Granularity captures the level of detail at which findings are reported. Anatomical Granularity contrasts reports that enumerate findings for each anatomical substructure (e.g., “pulmonary nodule in the right upper lobe anterior segment”) (Anatomical Granularity (High)) with those that aggregate findings at the organ/system level (e.g., “right lung pulmonary nodule”) (Anatomical Granularity (Low)). Quantitative Granularity contrasts reports that include explicit numerical measurements (e.g., “cardiothoracic ratio of 0.55”) (Quantitative Granularity (High)), with those that provide qualitative descriptions (e.g., “mild cardiomegaly”) (Quantitative Granularity (Low)).

Language. Language captures the linguistic variation in how radiologists report their interpretations. Terminology captures substitution of near-synonymous phrasings (e.g., “cardiomegaly” vs. “enlarged cardiac silhouette”), reflecting differences in training background, subspecialty, or institutional convention. Hedging captures how diagnostic uncertainty is expressed at an approximately fixed underlying confidence level (e.g., “worrisome for” vs. “concerning for”; “consistent with” vs. “likely represents”). Negation contrasts explicit negation of specific findings (e.g., “no pneumothorax, no pleural effusion”) (Negation (Explicit)), with implicit phrasings that convey the same information through positive statements about normality (e.g., “pleura appear normal”) (Negation (Implicit)).

![](images/c6905847477953965c5c4a0b3a0cc2a48dda8f305f2d9f4daa567716d5997265.jpg)  
Figure 2: Overview of REREF (Section 2.2)—our method for rewriting a reference report to follow an alternative reporting practice aligned with an axis of our taxonomy (Section 2.1). (a) Given a reference report and the target axis-aligned variation (e.g., Findings Scope (Minimal)), the Generator produces an alternative reference. The Verifier checks whether (i) the generated alternative report is clinically equivalent to the original report (e.g., no hallucinated findings) and (ii) the intended variation is accurately simulated. The generation–verification loop repeats until either a valid alternative is generated or the max number of generation attempts is reached. (b) Example output for Findings Scope (Minimal), which condenses the normal findings and focuses discussion on the key abnormalities. The highlighted findings are the abnormalities that stay preserved.

## 2.2 REREF: Rewriting a Reference Report to Follow Different Reporting Practices

Based on the proposed taxonomy in Section 2.1, we develop our REREF method, which takes as input a reference report (all sections) and an axis-aligned target variation (e.g., Terminology, Structure) and outputs an alternative reference rewritten to follow the specified reporting practices (Figure 2). We implement REREF by instantiating a Generator and a Verifier, which are each in charge of drafting the axis-aligned alternative reference, and validating if the generated report (i) accurately simulates the target variation, and (ii) is clinically equivalent to the original reference—i.e., the overall diagnosis of the patient and implied management plans do not change.

Notably, the Verifier also checks whether the target variation isfeasible for the input report, as certain variations are impossible to simulate without hallucinating new clinical information (e.g., Quantitative Granularity (High) would require adding new numerical measurements which are not present in the original reference). If the Verifier determines that the report sampled from the Generator either does not simulate the target variation correctly or alters the original clinical interpretation, then it rejects the sample, and the Generator is prompted again to sample a different report. This process is repeated until either the Verifier accepts the generated report or a max number of iterations (default: 5) is reached. In the latter case, we determine that an alternative reference cannot be generated reliably for the given target variation. We instantiate the Generator and Verifier with o3 (OpenAI, 2025), under a setup compliant with the data use agreements for all of the datasets that we consider in our evaluations (Section 3). We provide the remaining details (e.g., prompts, inference setup) in Appendix A.

## 2.3 Radiologist Validation of REREF

To ensure that REREF produces alternative reports that are (i) realistic (i.e., plausibly written by a practicing radiologist) and (ii) clinically equivalent to the original report (i.e., no change in diagnosis and patient management), we conduct a human annotation study to assess each aspect. We obtain our annotations from 13 board-certified radiologists—with specialties in cardiothoracic imaging, general radiology, abdomen/pelvis imaging, and neuroradiology—who routinely interpret CXR images. For each example, we first obtain annotations from two radiologists. If the two annotations disagree, we query a third radiologist to provide their own independent annotation. If all three annotators disagree, we query a fourth (cardiothoracic imaging) radiologist to provide the final “consensus” annotation, after accounting for the rationales provided by the first three annotators. We briefly describe each task below and include the detailed annotation guide provided to the radiologists in Appendix B.

Task 1: Realism. We randomly provide either the original radiologist-written report or its alternative version (from REREF) and ask each annotator to provide one of three scores (1–3): Realistic (3)—Reads as a report written by a practicing radiologist in routine clinical care; Minor Issues (2)—Mostly realistic, but contains subtle phrasing or formatting choices that are slightly unusual (though clinically correct); and Major Issues (1)—Contains major errors, incoherent phrasing, or unnatural elements that a practicing radiologist would not write in a standard radiology report. We obtain realism ratings for both the original and alternative reports to test whether the realism ratings for the alternative reports exhibit significant degradation from those of the original reports.

Task 2: Clinical Equivalence. We provide a randomly sampled (original, alternative) report pair and ask each annotator to provide one of three possible ratings: Equivalent—A treating physician reading either report would identify the same findings, and reach the same clinical decisions; Minor Discrepancy—Differences between the reports are limited to style, structure, terminology, or level of descriptive detail, and do not affect what clinical actions wouldfollow; and Major Discrepancy—The reports differ in major ways that could lead a treating physician to different clinical decisions.

We conduct our study using full radiology reports from four open-access CXR datasets (Section 3) by subsampling 10 (original, alternative) report pairs per (dataset, axis-aligned variation) combination, resulting in $4 \times 1 2 \times 1 0 = 4 8 0$ total pairs being annotated. We release the subset of 120 annotated MIMIC-CXR report pairs as $\mathbf { M I M I C - C X R - E X T - R E R E F } ^ { 2 }$ , excluding others with license restrictions.

## 3 Assessing the Impact of Reporting Practice Variation on RRG Evaluation

To investigate how reporting practice variations impact the conclusions from standard single-reference evaluation of RRG models, we use our REREF method (Section 2.2) to perturb the reference report for each imaging study along each axis of our taxonomy (Section 2.1), and quantify the resulting change in model performance $\mathrm { ( s c o r e _ { a l t e r n a t i v e } - s c o r e _ { o r i g i n a l } ) }$ as measured by an RRG evaluation metric. For each imaging study, we generate at most 12 alternative references—each corresponding to an axis-aligned perturbation—while skipping perturbations that are infeasible for the original reference provided (e.g., without hallucinating).

We focus on the task of generating the Findings section of each radiology report, which is most common for RRG evaluation<sup>3</sup>. For a given imaging study, each RRG model generates the Findings section based on the images and context (Indication, Comparison, and Technique sections) from the current exam only. We leave longitudinal settings, in which images and/or reports from prior exams are also provided as input (Zhang et al., 2024; Bannur et al., 2024; Moon et al., 2025), to future work.

Notably, for perturbations along the Section Assignment axis, which reorganize clinical findings between the Findings and Impression sections, the Findings-only scoping can amplify the measured sensitivity of metrics. We nonetheless retain this setup to reveal the ways in which current evaluation practices are sensitive to reporting practice variations in the reference. Below, we provide additional details on the datasets, models, and evaluation metrics used.

Datasets. We consider four widely used open-access CXR report datasets for evaluation: MIMIC CXR (Johnson et al., 2019), CheXpert Plus (Chambon et al., 2024), IU X-ray (Demner-Fushman et al., 2016), and ReXGradient-160K (Zhang et al., 2025a). For MIMIC-CXR, we use the official test set on PhysioNet (Goldberger et al., 2000) (3269 studies). For CheXpert Plus, we use the official validation set (200 studies), as the test set is not publicly available. For IU X-ray, we use the 590 imaging studies from the test split provided by Chen et al. (2020), following the ReXrank leaderboard (Zhang et al., 2025c). For ReXGradient-160K, we use the official public test set (10000 studies). From these test sets, we exclude any studies without a Findings section to be used as the reference (Appendix C). Table C1 shows the final number of examples from each dataset used for analysis.

Models. We evaluate 9 open-source RRG models: MAIRA-2 (Bannur et al., 2024), MedGemma-4B (Sellergren et al., 2025), CheXOne (Zhang et al., 2026b), MedVersa (Zhou et al., 2026), CheXagent-2- 3B (Chen et al., 2024), Med-CXRGen (Zhang et al., 2024), Libra (Zhang et al., 2025b), LLaVA-Rad (Zambrano Chaves et al., 2025), and CheX-MIMIC (Chambon et al., 2024). By CheX-MIMIC, we refer to the baseline model from Chambon et al. (2024) which was trained on both CheXpert Plus and MIMIC-CXR. We select these models to cover a wide range of model architectures and sizes, focusing on those that perform strongly in prior works (Zhang et al., 2025c). For models trained to handle multi-view CXR image inputs (e.g., frontal and lateral), we provide all views available for each imaging study, as feasible within the model’s constraints. Otherwise, we only provide the frontal view as input. We provide the remaining details (e.g., prompts, inference setup) in Appendix D.

(a)  
![](images/65a2bf687055b9459876c7a9bb250c14f6dd3401054b6aed46f081da4b462e2a.jpg)

![](images/f4f4d8867f303057920f0ee6ebac7eaae542a78ceaa14414e5a6efc0ada7b610.jpg)

(b)  
![](images/86dd1b5b0df6e5de39bd3075b9d5063b50c0d581f5542f93220b1a9263e395bb.jpg)

![](images/5972e42c3809faac5a53a7a3631402820d716717b50e2c3f55d720ef0b480740.jpg)  
Figure 3: Radiologists find that REREF generates alternative references that are realistic and preserve the original clinical interpretation (Section 4.1). (a) Task 1 (Realism): In 92.7% of all report pairs, the generated alternative reference is deemed as realistic as the original reference (left). (b) Task 2 (Clinical Equivalence): In 98.8% of all report pairs, the generated alternative reference is labeled as preserving the clinical interpretation without change in clinical impact (Equivalent or Minor Discrepancy) (left). In both tasks, trends vary across different axis-aligned perturbations (right).

Evaluation Metrics. We consider the following RRG evaluation metrics: (i) LLM judge metrics: CRIMSON (Baharoon et al., 2026), GREEN (Ostmeier et al., 2024); (ii) clinical entity-based metrics: RadGraph-F1 (Jain et al., 2021; Delbrouck et al., 2022), RaTEScore (Zhao et al., 2024); (iii) composite metrics: RadCliQ-v1 (Yu et al., 2023); (iv) disease label-based metrics: CheXbert-F1 (Smit et al., 2020); and (v) NLG metrics: BLEU-2 (Papineni et al., 2002), ROUGE-L (Lin, 2004), BERTScore (Zhang et al., 2020). For CRIMSON and GREEN, we use the open-source implementations based on MedGemma-4B (Sellergren et al., 2025) and Llama-2-7B (Touvron et al., 2023), respectively, as provided by the authors. For CheXbert-F1, we compute the micro-average F1 scores based on 5 out of the 14 supported disease labels—atelectasis, cardiomegaly, consolidation, edema, and pleural effusion—following prior works (Delbrouck et al., 2022). All metrics range between 0 and 1, except CRIMSON (range: [-1,1]) and RadCliQ-v1 (range: $( 0 , \infty ) )$ . As RadCliQ-v1 is the only metric where lower is better, we flip its sign unless specified otherwise, to match the behavior of the other metrics. While taking the reciprocal (i.e., 1/RadCliQ-v1) is more standard in the literature (Zhang et al., 2025c; Liu et al., 2026), we avoid doing so, as it becomes unstable when the raw values approach 0.

## 4 Results

Here, we summarize the main findings from our experiments. Unless specified otherwise, we focus our discussion on the following metrics that prior works suggest to correlate well with radiologist judgment and are widely used: CRIMSON, GREEN, RadCliQ-v1, RaTEScore, and RadGraph-F1.

## 4.1 REREF Generates Realistic Alternative References That Preserve Clinical Interpretation

From the expert annotation tasks outlined in Section 2.3, we find that radiologists generally find REREF to be effective in generating realistic alternative references without modifying the original clinical interpretation (Figure 3). For Task 1 (Realism), the alternative report is labeled to be as realistic as the original (i.e., reali $\mathrm { s m } _ { \mathrm { a l t e r n a t i v e } } - \mathrm { r e a l i s m } _ { \mathrm { o r i g i n a l } } \geq 0 )$ in 92.7% (445 out of 480) of the annotated (original, alternative) report pairs (Figure 3(a, left)). Of the $4 8 0 \times 2 = 9 6 0$ individual reports reviewed, radiologists unanimously agree on the realism rating in 567 cases (59.1%), achieve majority agreement in 345 cases (35.9%), and reach a three-way disagreement (resolved by a fourth annotator;

![](images/25cd9d2669728e955b44181bb403da614ba36c9fc966685fc5690b91b0ec84d8.jpg)  
Figure 4: Most of the evaluation metrics we consider are sensitive to reporting practice variations in the reference (Section 4.2). Each box plot shows the distribution, across $9 \times 4 = 3 6$ (model, dataset) pairs, of changes in the average model scores induced by each axis-aligned perturbation, normalized by the metric’s standard deviation across (model, dataset) pairs under the original references. The lower and upper whiskers correspond to the 5th and 95th percentiles of the normalized differences, respectively. For each metric, the mean and max interquartile range (IQR) values, aggregated over the different axes, indicate the overall and worst-case sensitivity of the metric to reporting practice variations. Score changes are computed over the subset of studies for which each perturbation is feasible (Section 3), whose proportion we report in parentheses below each x-axis label.

Section 2.3) in 48 cases (5.0%). The realism trends vary across different perturbations—e.g., Section Assignment leads to a slight degradation, while Findings Scope (Minimal) and Anatomical Granularity (Low) do not—but 9 out of 12 achieve no degradation on average (Figure 3(a, right)).

For Task 2 (Clinical Equivalence), only 1.2% of all report pairs (6 out of 480) are given the Major Discrepancy rating (Equivalent: 89.4%; Minor Discrepancy: 9.4%), indicating that REREF generally preserves the original clinical content without altering the implied diagnosis and patient management (Figure 3(b, left)). Of the 480 report pairs reviewed, radiologists unanimously agree on the equivalence rating in 329 cases (68.5%), achieve majority agreement in 140 cases (29.2%), and reach a three-way disagreement (resolved by a fourth annotator) in 11 cases (2.3%). The specific trends vary across different perturbations, with some (notably, those along the Granularity dimension) exhibiting higher Minor Discrepancy rates (Figure 3(b, right)). Meanwhile, we find that radiologists often use Minor Discrepancy to mark changes in the level of detail that they deem clinically inconsequential (e.g., “Report A is slightly more specific in location... a treating physician will notice without changing clinical management.”)—exactly the kind of variation these perturbations are designed to introduce.

## 4.2 Measured Clinical Accuracy Can Depend on the Reporting Practices of the Reference

Under targeted perturbations to the reference report along each axis of our taxonomy (Section 3), we find that many metrics shift substantially in response to reporting practice variations (Figure 4)—shifts that, as we show below, cannot be attributed to clinical content alone (Figure 5). In Figure 4, each box plot shows the distribution, across $9 \times 4 = 3 6$ (model, dataset) pairs, of changes in the average model scores $\begin{array} { r } { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( \mathrm { s c o r e } _ { \mathrm { a l t e r n a t i v e } } ^ { ( i ) } - \mathrm { s c o r e } _ { \mathrm { o r i g i n a l } } ^ { ( i ) } ) } \end{array}$ induced by each axis-aligned perturbation, normalized by the metric’s standard deviation across (model, dataset) pairs under the original references so that the shifts are interpretable on the scale of the metric’s natural variation. The lower and upper whiskers correspond to the 5th and 95th percentiles, respectively. We also report the mean and max interquartile range (IQR) values (aggregated across all axis-aligned perturbations) as summary statistics of the overall and worst-case sensitivity of each metric to reporting practice variations.

![](images/ddef98cd6a3ae4ba1d10d132ad3e65aba4885fa195c2d22e509d1e7ff8d0aa82.jpg)  
Figure 5: Assessment of clinical accuracy can be impacted by reporting practices of the reference (Section 4.2). We show an example based on the IU X-ray dataset, where a report generated by CheXagent is individually compared using GREEN against a radiologist-validated (original, alternative) report pair generated by a Findings Scope (Minimal) perturbation using REREF (Section 2.2). Findings highlighted with the same color indicate the “matched” findings, as determined by GREEN. Under the alternative reference, a mismatch in the granularity of the normal pleural findings causes the CheXagent score to drop to 0, despite partial agreement in clinical interpretation.

CRIMSON emerges as the most stable metric overall (mean IQR: 0.083), but still exhibits noticeable variations along certain axes (max IQR: 0.345), such as Findings Scope (Minimal) and Section Assignment. Compared to CRIMSON, GREEN exhibits higher overall and worst-case sensitivity (mean IQR: 0.245, max IQR: 0.739), partially due to its additional sensitivity to any changes to the normal findings in the reports, which CRIMSON ignores by design (Baharoon et al., 2026). The remaining non-LLM-based metrics—RadCliQ-v1 (IQR: mean = 0.512, max = 1.101), RaTEScore (IQR: mean = 0.384, max = 1.443), and RadGraph-F1 (IQR: mean = 0.741, max = 1.542)—exhibit substantially higher sensitivity, with some perturbations leading to shifts in the average model scores that are 1.5–2 times as large as each metric’s natural variation (e.g., RadCliQ-v1 in response to Structure (Structured), RadGraph-F1 in response to Anatomical Granularity (High)).

While some sensitivity to the details of the reference is expected, we find that reporting practice, rather than clinical content, can drive such score changes. To illustrate, we show an example from IU X-ray, where a report generated by CheXagent is individually compared against a radiologist-validated (original, alternative) report pair using GREEN, under the Findings Scope (Minimal) perturbation for which R R tends to be more reliable (Figure 3, right). In both cases, CheXagent misses the reference finding of “Stable left mid lung granuloma” but correctly identifies the absence of other pleural abnormalities. Against the original reference, GREEN credits CheXagent for this normal interpretation, resulting in a score of 0.833. However, against the alternative reference, the same prediction receives no credit and leads to a score of 0, as the Findings Scope (Minimal) perturbation condenses the reference’s pleural normal findings into a higher-level summary that GREEN no longer treats as a match. The two references are clinically equivalent, yet the score collapses from 0.833 to 0, and the entire score difference is attributable to how the reference chooses to express the normal findings. Such behavior illustrates the sensitivity of metrics like GREEN to the reporting practices of the reference, beyond the true accuracy of the underlying clinical interpretations.

## 4.3 Reporting Practice Variations Can Alter Conclusions from Model Comparison

We also find that the sensitivity of metrics to reporting practice can unevenly impact the scores of different models (Figure 6). For example, under RadCliQ-v1, Anatomical Granularity (Low) systematically benefits certain models over others on ReXGradient-160K, resulting in positive score changes for MedVersa (+0.07) and Med-CXRGen (+0.03), and negative score changes for others like MedGemma-4B (-0.13) and CheX-MIMIC (-0.11). These trends emerge as Anatomical Granularity (Low) merges separate anatomical mentions into compound anatomical findings (e.g., “The heart size and mediastinal contours are within normal limits.” → “Normal cardiomediastinal silhouette.”). Such a change leads to a higher number of matches in clinical entities for models like MedVersa, which is relatively terse in its reports and exhibits a similar tendency (e.g., uses “cardiomediastinal” in 59.6% of all studies in ReXGradient-160K), in contrast to others like MedGemma-4B that tend to enumerate the lower-level anatomical structures (e.g., uses “cardiomediastinal” in only 0.7% of all studies in ReXGradient-160K). As RadCliQ-v1 is computed based on metrics like RadGraph-F1 that require exact clinical entity matches, these changes disparately impact different models.

![](images/40bed0e9875de0241ea888b6ebcb76f1acf904d02f67da75fd4be48186832deb.jpg)  
Figure 6: Axis-aligned perturbations to the reference reporting practice can impact model scores unevenly (Section 4.3). On ReXGradient-160K, the Anatomical Granularity (Low) perturbation, which merges separate anatomical mentions (e.g., “heart”, “mediastinal”) into compound anatomical findings (e.g., “cardiomediastinal”), tends to advantage models like MedVersa (+0.07), which exhibit a similar tendency, over others like MedGemma-4B (-0.13) that exhibit the opposite tendency. The average model scores are computed based on the subset of studies in ReXGradient-160K for which the Anatomical Granularity (Low) perturbation is feasible (∼29.3%) (Section 3).

Ultimately, we find that such changes can alter the broader conclusions of RRG model comparison (Figure 7). For example, in Figure 7(a), we show the gap in average RadCliQ-v1 between Libra and CheXOne on MIMIC-CXR, under the original (left) vs. alternative references (middle–right), where we consider a pairwise comparison to be significant if the 95% bootstrap confidence interval (CI) for this gap<sup>4</sup> excludes zero, and a statistical tie otherwise. When a perturbation is infeasible for a study, we retain its original-reference score, and always compute the average scores using all studies. Under the original references, Libra significantly outperforms CheXOne. However, after the perturbations, Libra significantly underperforms CheXOne in 6 settings (e.g., Findings Scope (Minimal)), reaches a tie in 4 settings (e.g., Anatomical Granularity (Low)), and significantly outperforms it in only 2 settings (e.g., Structure (Free-Text)).

Aggregating the results across all datasets and model pairs, we find that for all metrics, the model rankings can deviate from those obtained under the original references, in response to the perturbations (Figure 7(b)). For 7 out of 9 metrics, such changes remain noticeable even when we limit to pairwise ranking reversals whose 95% bootstrap CI-based comparisons change (Figure E1), with CRIMSON and CheXbert-F1 being the exceptions (Appendix E). Notably, rankings under RadCliQ-v1, the default metric for leaderboards like ReXrank (Zhang et al., 2025c), show the highest median sensitivity to changes in reporting practice among all clinical metrics (Figure 7(b); in cyan).

Meanwhile, while we focus on single-axis perturbations in our main experiments, real-world radiology reports likely reflect multiple reporting practices simultaneously and in a correlated manner (e.g., Findings Scope (Minimal) + Anatomical Granularity (Low)). We find that compounding multiple perturbations can lead to even more disruptions to the model rankings (Figures F1–F2; Appendix F).

## 5 Conclusion

In this work, we demonstrated that many automated metrics for RRG model evaluation—including those reported to correlate well with radiologist judgment—are sensitive to variations in the reporting practices of the reference, to the extent that substantive conclusions about the performance of RRG models can meaningfully change. In surfacing these findings, we (i) introduced a taxonomy that characterizes the various ways in which radiologists vary in how they communicate the same clinical interpretation; (ii) developed REREF, an expert-validated method for rewriting reference reports to align with specific reporting practices; and (iii) conducted extensive evaluations with nine RRG models and four CXR datasets to quantify the impact of each variation on model performance.

(a)  
![](images/a3154a6d25bb0c6c2427a1a130ed5b1f3192d9a7cff20cc28ce7d672dfcbbc04.jpg)

![](images/27432ff8b5dcda9e2df03596d6637f442b6a3187978a0f53fbc803ed9fab1ab6.jpg)  
Figure 7: Model rankings are sensitive to the reporting practices of the reference (Section 4.3). (a) On MIMIC-CXR, the relative ordering of Libra (Zhang et al., 2025b) vs. CheXOne (Zhang et al., 2026b) based on average score varies across different axis-aligned perturbations. Each bar displays the difference in average RadCliQ-v1 (Libra − CheXOne), where the sign of RadCliQ-v1 has been flipped so that higher is better (Section 3), and the error bars denote the 95% bootstrap confidence intervals (CIs) in this difference. When a perturbation is infeasible for an imaging study, we retain its original-reference score, so the average scores are always computed using all studies. (b) Aggregating across all datasets and model pairs, we find that for all metrics, the model rankings can change after perturbation, albeit to varying extents. Each box plot shows, across the 12 axis-aligned perturbations, the percentage of all $4 \times { \binom { 9 } { 2 } } = 1 4 4$ (dataset, model pair) comparisons whose ordering by mean score reverses relative to the original reference, with the diamond-shaped markers denoting the max and min values across all perturbations. The cyan-colored box plots indicate the clinical metrics.

Our findings suggest that many current metrics may fail to decouple clinical interpretation from conformity to reporting practices of the reference. Thus, choosing the “right” references that bestreflect the desired reporting practices for the setting of interest can be an important design decision for model evaluation. For instance, when assessing potential deployment at an institution, the reference reports should reflect local reporting conventions, since adherence to those conventions may itself be an integral part of report quality. When assessing a model’s ability to document fine-grained clinical findings, the references should exhibit sufficient Completeness and Granularity, as a sparse, abstract reference cannot be used to verify the accuracy of more detailed findings generated by a model.

For broader model comparisons, we recommend using our REREF method to simulate and evaluate against multiple clinically equivalent references spanning practically relevant reporting practices and reporting the resulting variability in model scores and rankings. The objective is not for every mode to receive the same score under every reference, but to ensure that the positive conclusions about a model’s effectiveness do not depend on an arbitrary choice of reporting practice in the reference.

Limitations. While we conduct extensive radiologist validation to verify REREF’s capability in generating realistic and clinically equivalent alternative references, it is possible that the generated outputs exhibit hallucinations or subtle deviations in clinical interpretation. As REREF is implemented based on a proprietary, closed-source LLM (o3), reproducing the alternative references used in our perturbation experiments may be difficult. Nonetheless, our shared repository contains the complete generation and verification pipeline used in our experiments, and all of the prompting details are included in Appendix A. As LLM judge metrics (CRIMSON, GREEN) can be implemented with any LLM of choice, the specific trends observed in our study may change when a different model is used.

Acknowledgements. We gratefully acknowledge DARPA (FA8750-23-2-1015), ONR (N00014- 23-1-2368), NSF (IIS2211955), UPMC, Highmark Health, Abridge, Ford Research, Mozilla, the

PwC Center, Amazon AI, JP Morgan Chase, the Block Center, the Center for Machine Learning and Health, and the CMU Software Engineering Institute (SEI) via Department of Defense contract FA8702-15-D-0002, for their generous support of our research.

## References

Anthropic. Introducing Claude Sonnet 4.5. https://www.anthropic.com/news/ claude-sonnet-4-5, Sept. 2025.

M. Baharoon, T. Heintz, S. Raissi, M. Alabbad, M. Alhammad, H. AlOmaish, S. E. Kim, O. Banerjee, and P. Rajpurkar. CRIMSON: A Clinically-Grounded LLM-Based Metric for Generative Radiology Report Evaluation, 2026.

S. Bannur, K. Bouzid, D. Coelho de Castro, A. Schwaighofer, S. Bond-Taylor, M. Ilse, F. Pérez-García, V. Salvatelli, H. Sharma, F. Meissen, M. Ranjit, S. Srivastav, J. Gong, F. Falck, O. Oktay, A. Thieme, M. P. Lungren, M. T. Wetscherek, J. Alvarez-Valle, and S. Hyland. MAIRA-2: Grounded Radiology Report Generation. Technical Report MSR-TR-2024-18, Microsoft, June 2024. URL https://www.microsoft.com/en-us/research/publication/ maira-2-grounded-radiology-report-generation/.

J. M. L. Bosmans, E. Neri, O. Ratib, and C. E. K. Jr. Structured Reporting: A Fusion Reactor Hungry for Fuel. Insights into Imaging, 6(1):129–132, December 2014.

P. Chambon, J.-B. Delbrouck, T. Sounack, S.-C. Huang, Z. Chen, M. Varma, S. Q. Truong, C. T. Chuong, and C. P. Langlotz. CheXpert Plus: Augmenting a Large Chest X-ray Dataset with Text Radiology Reports, Patient Demographics and Additional Image Formats. arXiv:2405.19538, 2024.

Z. Chen, Y. Song, T.-H. Chang, and X. Wan. Generating Radiology Reports via Memory-driven Transformer. In B. Webber, T. Cohn, Y. He, and Y. Liu, editors, Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1439–1449, Online, Nov. 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.112. URL https://aclanthology.org/2020.emnlp-main.112/.

Z. Chen, M. Varma, J. Xu, M. Paschali, D. V. Veen, A. Johnston, A. Youssef, L. Blankemeier, C. Bluethgen, S. Altmayer, J. M. J. Valanarasu, M. S. E. Muneer, E. P. Reis, J. P. Cohen, C. Olsen, T. M. Abraham, E. B. Tsai, C. F. Beaulieu, J. Jitsev, S. Gatidis, J.-B. Delbrouck, A. S. Chaudhari, and C. P. Langlotz. A Vision-Language Foundation Model to Enhance Efficiency of Chest X-ray Interpretation, 2024.

J.-B. Delbrouck, P. Chambon, C. Bluethgen, E. Tsai, O. Almusa, and C. Langlotz. Improving the Factual Correctness of Radiology Report Generation with Semantic Rewards. In Y. Goldberg, Z. Kozareva, and Y. Zhang, editors, Findings of the Association for Computational Linguistics: EMNLP 2022, pages 4348–4360, Abu Dhabi, United Arab Emirates, Dec. 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-emnlp.319. URL https:// aclanthology.org/2022.findings-emnlp.319/.

J.-B. Delbrouck, J. Xu, J. Moll, A. Thomas, Z. Chen, S. Ostmeier, A. Azhar, K. Z. Li, A. Johnston, C. Bluethgen, E. P. Reis, M. S. Muneer, M. Varma, and C. Langlotz. Automated Structured Radiology Report Generation. In W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 26813–26829, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.1301. URL https://aclanthology.org/2025.acl-long.1301/.

D. Demner-Fushman, M. D. Kohli, M. B. Rosenman, S. E. Shooshan, L. Rodriguez, S. Antani, G. R. Thoma, and C. J. McDonald. Preparing a Collection of Radiology Examinations for Distribution and Retrieval. Journal of the American Medical Informatics Association, 23(2):304–310, 2016.

A. L. Goldberger, L. A. N. Amaral, L. Glass, J. M. Hausdorff, P. C. Ivanov, R. G. Mark, J. E. Mietus, G. B. Moody, C.-K. Peng, and H. E. Stanley. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation, 101(23): e215–e220, June 2000.

J. Huang, M. T. Wittbrodt, C. N. Teague, E. Karl, G. Galal, M. Thompson, A. Chapa, M.-L. Chiu, B. Herynk, R. Linchangco, A. Serhal, J. A. Heller, S. F. Abboud, and M. Etemadi. Efficiency and Quality of Generative AI–Assisted Radiograph Reporting. JAMA Network Open, 8(6):

e2513921–e2513921, 06 2025. ISSN 2574-3805. doi: 10.1001/jamanetworkopen.2025.13921. URL https://doi.org/10.1001/jamanetworkopen.2025.13921.

S. Jain, A. Agrawal, A. Saporta, S. Q. Truong, D. N. Duong, T. Bui, P. Chambon, Y. Zhang, M. P. Lungren, A. Y. Ng, C. P. Langlotz, and P. Rajpurkar. RadGraph: Extracting Clinical Entities and Relations from Radiology Reports. Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, 2021.

A. E. W. Johnson, T. J. Pollard, S. J. Berkowitz, N. R. Greenbaum, M. P. Lungren, C.-y. Deng, R. G. Mark, and S. Horng. MIMIC-CXR, A De-Identified Publicly Available Database of Chest Radiographs with Free-Text Reports. Scientific Data, 6(317), 2019.

C.-Y. Lin. ROUGE: A Package for Automatic Evaluation of Summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics. URL https://aclanthology.org/W04-1013/.

Q. Liu, S. Zhang, G. Qin, Y. Gu, Y. Jin, S. Preston, Y. Xu, S. Kiblawi, W. wai Yim, T. Ossowski, T. Naumann, M. Wei, and H. Poon. Scaling Medical Imaging Report Generation with Multimodal Reinforcement Learning. arXiv:2601.17151, 2026.

J. H. Moon, G. Choi, P. Rabaey, M. G. Kim, H. G. Hong, J.-O. Lee, H. Yoon, E. W. Doe, J. Kim, H. Sharma, D. C. Castro, J. Alvarez-Valle, and E. Choi. Lunguage: A Benchmark for Structured and Sequential Chest X-ray Interpretation. arXiv:2505.21190, 2025.

OpenAI. OpenAI o3 and o4-mini System Card, 2025. URL https://openai.com/index/ o3-o4-mini-system-card/.

S. Ostmeier, J. Xu, Z. Chen, M. Varma, L. Blankemeier, C. Bluethgen, A. E. Michalson, M. Moseley, C. Langlotz, A. S. Chaudhari, and J.-B. Delbrouck. GREEN: Generative Radiology Report Evaluation and Error Notation. In Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, editors, Findings of the Association for Computational Linguistics: EMNLP 2024, pages 374–390, Miami, Florida, USA, Nov. 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.21. URL https://aclanthology.org/2024.findings-emnlp.21/.

K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu. BLEU: A Method for Automatic Evaluation of Machine Translation. In Proceedings of the 40th Annual Meeting on Association for Computational Linguistics, ACL ’02, page 311–318, USA, 2002. Association for Computational Linguistics. doi: 10.3115/1073083.1073135. URL https://doi.org/10.3115/1073083.1073135.

A. Sellergren, S. Kazemzadeh, T. Jaroensri, A. Kiraly, M. Traverse, T. Kohlberger, S. Xu, F. Jamil, C. Hughes, C. Lau, J. Chen, F. Mahvar, L. Yatziv, T. Chen, B. Sterling, S. A. Baby, S. M. Baby, J. Lai, S. Schmidgall, L. Yang, K. Chen, P. Bjornsson, S. Reddy, R. Brush, K. Philbrick, M. Asiedu, I. Mezerreg, H. Hu, H. Yang, R. Tiwari, S. Jansen, P. Singh, Y. Liu, S. Azizi, A. Kamath, J. Ferret, S. Pathak, N. Vieillard, R. Merhej, S. Perrin, T. Matejovicova, A. Ramé, M. Riviere, L. Rouillard, T. Mesnard, G. Cideron, J. bastien Grill, S. Ramos, E. Yvinec, M. Casbon, E. Buchatskaya, J.-B. Alayrac, D. Lepikhin, V. Feinberg, S. Borgeaud, A. Andreev, C. Hardin, R. Dadashi, L. Hussenot, A. Joulin, O. Bachem, Y. Matias, K. Chou, A. Hassidim, K. Goel, C. Farabet, J. Barral, T. Warkentin, J. Shlens, D. Fleet, V. Cotruta, O. Sanseviero, G. Martins, P. Kirk, A. Rao, S. Shetty, D. F. Steiner, C. Kirmizibayrak, R. Pilgrim, D. Golden, and L. Yang. MedGemma Technical Report, 2025.

A. Smit, S. Jain, P. Rajpurkar, A. Pareek, A. Ng, and M. Lungren. Combining Automatic Labelers and Expert Annotations for Accurate Radiology Report Labeling Using BERT. In B. Webber, T. Cohn, Y. He, and Y. Liu, editors, Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1500–1519, Online, Nov. 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.117. URL https://aclanthology.org/2020.emnlp-main.117/.

R. Tanno, D. Barrett, A. Sellergren, S. Ghaisas, S. Dathathri, A. See, J. Welbl, C. Lau, T. Tu, S. Azizi, K. Singhal, M. Schaekermann, R. May, R. Lee, S. Man, S. Mahdavi, Z. Ahmed, Y. Matias, J. Barral, and S. I. Ktena. Collaboration Between Clinicians and Vision–Language Models in Radiology Report Generation. Nature Medicine, 31:599–608, 11 2024. doi: 10.1038/s41591-024-03302-1.

H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. C. Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M.-A. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom. Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv preprint arXiv:2307.09288, 2023.

D. L. Weiss and C. P. Langlotz. Structured Reporting: Patient Care Enhancement or Productivity Nightmare? Radiology, 249(3):739–747, 2008.

F. Yu, M. Endo, R. Krishnan, I. Pan, A. Tsai, E. P. Reis, E. K. U. N. Fonseca, H. M. H. Lee, Z. S. H. Abad, A. Y. Ng, C. P. Langlotz, V. K. Venugopal, and P. Rajpurkar. Evaluating Progress in Automatic Chest X-ray Radiology Report Generation. Patterns, 4(9), 2023.

J. M. Zambrano Chaves, S.-C. Huang, Y. Xu, H. Xu, N. Usuyama, S. Zhang, F. Wang, Y. Xie, M. Khademi, Z. Yang, H. Awadalla, J. Gong, H. Hu, J. Yang, C. Li, J. Gao, Y. Gu, C. Wong, M. Wei, T. Naumann, M. Chen, M. P. Lungren, A. Chaudhari, S. Yeung-Levy, C. P. Langlotz, S. Wang, and H. Poon. A Clinically Accessible Small Multimodal Radiology Model and Evaluation Metric for Chest X-ray Findings. Nature Communications, 16(3108), 2025.

T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi. Bertscore: Evaluating text generation with bert, 2020. URL https://arxiv.org/abs/1904.09675.

X. Zhang, Z. Meng, J. Lever, and E. S. Ho. Gla-AI4BioMed at RRG24: Visual Instruction-tuned Adaptation for Radiology Report Generation. In Proceedings of the 23rd Workshop on Biomedical Natural Language Processing. Association for Computational Linguistics, 2024.

X. Zhang, J. N. Acosta, J. Miller, O. Huang, and P. Rajpurkar. ReXGradient-160K: A Large-Scale Publicly Available Dataset of Chest Radiographs with Free-text Reports. Machine Learning for Health (ML4H), 2025a.

X. Zhang, Z. Meng, J. Lever, and E. S. L. Ho. Libra: Leveraging Temporal Images for Biomedical Radiology Analysis. In W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, editors, Findings of the Association for Computational Linguistics: ACL 2025, pages 17275–17303, Vienna, Austria, July 2025b. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/ v1/2025.findings-acl.888. URL https://aclanthology.org/2025.findings-acl.888/.

X. Zhang, H.-Y. Zhou, X. Yang, O. Banerjee, J. N. Acosta, J. Miller, O. Huang, and P. Rajpurkar. ReXrank: A Public Leaderboard for AI-Powered Radiology Report Generation. In J. Wu, J. Zhu, M. Xu, and Y. Jin, editors, Proceedings of The First AAAI Bridge Program on AI for Medicine and Healthcare, volume 281 of Proceedings of Machine Learning Research, pages 90–99. PMLR, 25 Feb 2025c. URL https://proceedings.mlr.press/v281/zhang25b.html.

X. Zhang, J. N. Acosta, X. Yang, S. Adithan, L. Luo, H.-Y. Zhou, J. Miller, O. Huang, Z. Zhou, I. E. Hamamci, S. Bannur, K. Bouzid, X. Zhang, Z. Meng, A. Nicolson, B. Koopman, I. Baek, H. Ko, M. P. Ranjit, S. Srivastav, S. G. Sambanthan, and P. Rajpurkar. Automated Chest X-ray Report Generation Remains Unsolved. Pacific Symposium of Biocomputing, 2026a.

Y. Zhang, C. Wang, Y. Gao, J. Liu, M. Varma, J. Xu, S. Ostmeier, J. Long, S. Gatidis, S. Dehkharghani, A. Michalson, E. K. Hong, C. Bluethgen, H. H. Guo, A. V. Ortiz, S. Altmayer, S. Bodapati, J. D. Janizek, K. Chang, J.-B. Delbrouck, A. S. Chaudhari, and C. P. Langlotz. A Reasoning-Enabled Vision-Language Foundation Model for Chest X-ray Interpretation, 2026b.

W. Zhao, C. Wu, X. Zhang, Y. Zhang, Y. Wang, and W. Xie. RaTEScore: A Metric for Radiology Report Generation. In Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 15004–15019, Miami, Florida, USA, Nov. 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024. emnlp-main.836. URL https://aclanthology.org/2024.emnlp-main.836/.

H.-Y. Zhou, J. N. Acosta, S. Adithan, S. Datta, E. J. Topol, and P. Rajpurkar. MedVersa: A Generalist Foundation Model for Diverse Medical Imaging Tasks. NEJM AI, 3(4), 2026.

## A Additional Details on REREF

Here, we provide additional details for REREF (Section 2.2). We first describe the precise definitions, allowed variations, illustrative examples, and axis-specific constraints used to operationalize each axis of our taxonomy (Appendix A.1). We then describe the prompts used to instantiate the Generator and Verifier (Appendix A.2), as well as the inference setup used to deploy them (Appendix A.3).

## A.1 Axis-Aligned Perturbations: Definitions, Examples, and Constraints

For each axis of our taxonomy (Section 2.1), we specify its (i) description, (ii) a set of allowed variations (each with a short definition), (iii) illustrative examples contrasting the variations carefully designed with a board-certified radiologist, and (iv) zero or more axis-specific constraints that supplement the default constraints listed in Appendix A.2. To operationalize perturbations along each axis, we further distinguish between axes whose variations can be characterized in a standalone manner (e.g., for the Structure axis, a reference is either Structured or Free-Text)—which we refer to as absolute axes—and axes whose variations are best characterized relative to the original reference (e.g., Terminology is defined as a near-synonymous rewording of the original)—which we refer to as relative axes. For the absolute axes, the Generator produces alternative references for every variation other than the one that the original reference is already in (e.g., both Structure (Structured) and Structure (Free-Text) are sampled, with infeasibility handled at the Verifier stage). For the relative axes, we simply prompt the LLM to sample an alternative version that differs from the original reference along the given axis. The descriptions, allowed variations, examples, and constraints below are provided verbatim to the Generator and Verifier as part of the prompts shown in Appendix A.2.

## A.1.1 Organization: Structure (Absolute)

Description: Radiologists differ in their use of structured templates versus continuous free-text prose. Some prefer highly templated formats—explicit headings for each anatomical region, bullet points, numbered lists, subheadings for groups of findings—while others write in continuous sentences and paragraphs.

## Variations:

• Structured: Written in a highly templated format (e.g., explicit headings for each anatomical region, bullet points, numbered lists, subheadings for groups of findings).

• Free-Text: Written in continuous free-text prose.

## Example 1:

• Structured: “CARDIOMEDIASTINAL: Normal silhouette. LUNGS: Well expanded and clear. PLEURA: Sharp costophrenic sulci and diaphragmatic contour. BONES: No acute abnormalities.”

• Free-Text: “The cardiomediastinal silhouette is normal. The lungs are well expanded and clear. The costophrenic sulci and diaphragmatic contour are sharp. No acute osseous abnormalities.”

## Example 2:

• Structured: “1. Single view of chest demonstrates no significant interval change in position of bilateral chest tubes. Median sternotomy wires are unchanged. 2. Previously described left pneumothorax is not well seen on today’s exam. No large pneumothorax is seen. 3. Interval increase in left pleural effusion. Small right pleural effusion present. Summary: Possible significant abnormality and change. May need action.”

• Free-Text: “Single view of chest demonstrates no significant interval change in position of bilateral chest tubes. Median sternotomy wires are unchanged. Previously described left pneumothorax is not well seen on today’s exam. No large pneumothorax is seen. Interval increase in left pleural effusion. Small right pleural effusion present. Possible significant abnormality and change. May need action.”

## Example 3:

• Structured: “1. Cardiomegaly. 2. Resolution of right basal infiltrate.”

• Free-Text: “Cardiomegaly. Resolution of right basal infiltrate.”

Axis-Specific Constraints: The Findings and Impression sections need not be structured in the same way; e.g., the Findings section can be more structured while the Impression section is more free-text, and vice versa.

## A.1.2 Organization: Section Assignment (Relative)

Description: Radiologists differ in where they place specific types of content within the report (Findings vs. Impression). For example, some discuss prior comparisons only in the Impression to keep the Findings fully grounded in the current study, while others discuss them in the Findings to provide immediate clinical context, and others discuss them in both. As another example, some discuss differential diagnoses in both the Findings and the Impression, while others only discuss them in the Impression.

## Variations:

• Original: Retains the original assignment of content to the Findings and Impression sections.

• Alternative: Reassigns content to different sections relative to the original reference.

## Example 1:

• Original: “FINDINGS: Chronic left-sided rib fractures are again noted. The cardiomediastinal and hilar contours are unchanged from \_\_\_. Pleural thickening and blunting at the right costophrenic angle is again demonstrated, and is stable from the prior exam in \_\_\_ and likely represents pleural scarring and a small pleural effusion. No focal consolidation or pneumothorax is identified. IMPRESSION: Multiple chronic appearing left-sided rib fractures. No pneumothorax. Blunting of the costophrenic angle on the right likely represents pleural scarring and a small effusion, not significantly changed from \_\_

• Alternative: “FINDINGS: Left-sided rib fractures are present. The cardiomediastinal and hilar contours are normal. Pleural thickening and blunting at the right costophrenic angle, likely representing pleural scarring and a small pleural effusion. No focal consolidation or pneumothorax is identified. IMPRESSION: Multiple chronic left-sided rib fractures, again noted and unchanged from \_\_\_. Cardiomediastinal and hilar contours remain stable. No pneumothorax. Blunting of the costophrenic angle on the right likely represents pleural scarring and a small effusion, again demonstrated and stable from prior exam in

## Example 2:

• Original: “FINDINGS: A hazy opacity is present in the right lung, which may represent aspiration, pleural effusion or hemorrhage. Moderate cardiomegaly is stable. Slight prominence of the pulmonary vasculature with cephalization and enlarged pulmonary arteries are consistent with mild pulmonary edema. There are no displaced rib fractures. IMPRESSION: 1. Hazy opacity in the right lung, which may represent aspiration versus pleural effusion or hemorrhage. 2. Mild pulmonary edema. 3. No displaced rib fractures.”

• Alternative: “FINDINGS: A hazy opacity is present in the right lung. Moderate cardiomegaly is stable. Slight prominence of the pulmonary vasculature with cephalization and enlarged pulmonary arteries. There are no displaced rib fractures. IMPRESSION: 1. Hazy opacity in the right lung, which may represent aspiration versus pleural effusion or hemorrhage. 2. Mild pulmonary edema. 3. No displaced rib fractures.”

Axis-Specific Constraints: The union of clinical content across the Findings and Impression sections must match that of the original reference. The Generator must not (i) introduce details from any other sections (e.g., Technique, Indication, Comparison) or (ii) omit any clinical details from the original Findings and Impression sections.

## A.1.3 Completeness: Findings Scope (Absolute)

Description: Radiologists differ in how exhaustively they describe normal and abnormal findings. Some describe all normal and abnormal findings exhaustively, while others selectively omit normal

findings and focus on a minimal list of abnormal findings most likely to be clinically relevant to the patient’s condition and treatment.

## Variations:

• Exhaustive: Exhaustively describes all normal and abnormal findings.

• Minimal: Selectively omits normal findings and focuses on a minimal list of key abnormal findings.

## Example 1:

• Exhaustive: “No significant technical limitations. Lung volumes are normal. No consolidative airspace disease. No pleural effusion or pneumothorax. No pulmonary nodule or mass noted. Pulmonary vasculature and the cardiomediastinal silhouette are within normal limits. Atherosclerosis in the thoracic aorta is noted.”

• Minimal: “No evidence of pneumothorax, pleural effusion, or pulmonary opacity. Cardiomediastinal silhouette is normal. Atherosclerosis in the thoracic aorta.”

## Example 2:

• Exhaustive: “No exam quality limitations. Patient positioning is appropriate. Trachea is midline. Image of the abdomen is within normal limits. No radiopaque lines, tubes, or implants. Hila are normal. Lung volumes are normal. Heart size is normal. Mediastinal contours are unremarkable. Lungs are clear. No effusion or pneumothorax. Bones are intact.”

• Minimal: “Heart size is normal. No evidence of pneumothorax, pleural effusion, or pulmonary opacity. No acute abnormality.”

## Example 3:

• Exhaustive: “QUALITY: Portable AP radiograph demonstrates low lung volumes with mild rotation. No motion artifact. The lungs are clear bilaterally. No pleural effusion or pneumothorax. The cardiomediastinal silhouette is within normal limits for this projection. Visualized osseous structures are unremarkable.”

• Minimal: “The lungs are clear bilaterally. No pleural effusion or pneumothorax. The cardiomediastinal silhouette is within normal limits. Visualized osseous structures are unremarkable.”

Axis-Specific Constraints: None.

## A.1.4 Granularity: Anatomical Granularity (Absolute)

Description: Radiologists differ in the anatomical level at which they describe their findings. Some radiologists exhaustively enumerate findings for each anatomical substructure separately (e.g., “pulmonary nodule in the right upper lobe anterior segment”), while others only enumerate high-level findings at the level of the organ or system (e.g., “right lung pulmonary nodule”).

## Variations:

• High: Describes findings for each anatomical substructure separately.

• Low: Describes findings at the level of the organ or system.

## Example 1:

• High: “A pulmonary nodule is present in the right upper lobe anterior segment. A coronary artery stent is present in the left anterior descending artery. Small volume of fluid in the left posterior costophrenic sulcus. Right third through sixth healed rib fracture deformities. T9 anterior wedge compression fracture. Cholecystectomy clips are present.”

• Low: “Right lung pulmonary nodule. Prior coronary artery stenting. Small left pleural effusion. Healed rib fracture deformities. Thoracic compression fracture. Right upper quadrant surgical clips present.”

## Example 2:

• High: “The cardiac silhouette appears enlarged. Pulmonary vascular markings are prominent in the bilateral perihilar regions. There is mild interstitial prominence in the bilateral lower lungs. Trace fluid blunts the right posterior costophrenic sulcus. A left chest wall pacemaker is present with leads projecting over the cardiac silhouette. No focal airspace opacity. No pneumothorax.”

• Low: “Enlarged cardiac silhouette. Prominent pulmonary vasculature. Mild interstitial lung prominence. Trace right pleural effusion. Left-sided pacemaker present. No focal airspace opacity or pneumothorax.”

## Example 3:

• High: “The lungs are well expanded and clear without focal airspace opacity. The pulmonary vasculature is within normal limits. The costophrenic angles are sharp bilaterally. The diaphragms are smooth with normal contour. The cardiomediastinal silhouette is normal in size and shape. The trachea is midline. The hila are unremarkable. No pneumothorax is identified. The visualized osseous structures are intact.”

• Low: “Normal cardiomediastinal silhouette. Lungs are clear without consolidation, effusion, or pneumothorax. Pulmonary vasculature is within normal limits. No acute osseous abnormalities.”

Axis-Specific Constraints: None.

## A.1.5 Granularity: Quantitative Granularity (Absolute)

Description: Radiologists differ in whether they include explicit numerical measurements (e.g., “cardiothoracic ratio of 0.55”, “2.4 cm at the apex”) or rely on qualitative descriptors (e.g., “mild cardiomegaly”, “moderate”).

## Variations:

• High: Includes precise quantitative findings.

• Low: Includes qualitative descriptions only.

## Example 1:

• High: “Left pleural effusion measuring 4 cm in height. Cardiothoracic ratio of 0.55.”

• Low: “Moderate left pleural effusion. Mild cardiomegaly.”

## Example 2:

• High: “There is an endotracheal tube whose distal tip is seen 6.2 cm above the carina appropriately sited.”

• Low: “There is an endotracheal tube whose distal tip is appropriately sited above the carina.”

## Example 3:

• High: “Moderate right apical pneumothorax has minimally decreased since yesterday. The maximum width at the apex measures 2.4 cm as compared to yesterday measuring 2.9 cm.”

• Low: “Moderate right apical pneumothorax has minimally decreased since yesterday.”

Axis-Specific Constraints: The Generator should only modify the reference along this axis if the original reference contains precise quantitative findings, so as to avoid hallucinating new measurements when generating Quantitative Granularity (High). In practice, this constraint causes the Quantitative Granularity (High) alternatives to be flagged as infeasible by the Verifier for the vast majority of original references, and we report results only for Quantitative Granularity (Low) in our experiments (Section 3).

## A.1.6 Language: Terminology (Relative)

Description: Radiologists may use different terms to describe the same finding, depending on personal preference, training background, subspecialty, or institutional convention. These are nearsynonymous phrasings that convey essentially the same clinical meaning.

## Variations:

• Original: Retains the original terminology.

• Alternative: Substitutes alternative terminology that conveys essentially the same clinical meaning.

## Example 1:

• Original: “No gross consolidation, atelectasis or infiltrate. No pleural fluid collection or pneumothorax. Cardiomediastinal silhouette is within normal limits.”

• Alternative: “The lungs are clear without evidence of focal airspace disease or consolidation. There is no evidence of pneumothorax or pleural effusion. The cardiac and mediastinal contours are within normal limits.”

## Example 2:

• Original: “No pulmonary opacity.”

• Alternative: “Lungs are clear.”

## Example 3:

• Original: “Cardiomegaly.”

• Alternative: “Enlarged cardiac silhouette.”

## Example 4:

• Original: “Bones are intact.”

• Alternative: “Osseous structures are within normal limits.”

## Example 5:

• Original: “No evidence of intraabdominal free air.”

• Alternative: “Visualized upper abdomen is normal.”

Axis-Specific Constraints: None.

## A.1.7 Language: Hedging (Relative)

Description: Radiologists differ in how they express diagnostic uncertainty even when their underlying interpretations are similar. Different phrases can convey similar levels of uncertainty but use different linguistic constructions.

## Variations:

• Original: Retains the original hedging language.

• Alternative: Substitutes alternative hedging language that conveys essentially the same level of uncertainty.

## Example 1:

• Original: “Peripheral wedge-shaped opacity is worrisome for pulmonary embolism.”

• Alternative: “Peripheral wedge-shaped opacity is concerning for pulmonary embolism.”

## Example 2:

• Original: “Opacity consistent with atelectasis.”

• Alternative: “Opacity likely represents atelectasis.”

## Example 3:

• Original: “Right lung opacity, query pneumonia.”

• Alternative: “Right lung opacity may represent pneumonia.”

## Example 4:

• Original: “Left lung opacity, query pneumonia.”

• Alternative: “Left lung opacity suggestive of pneumonia.”

## Example 5:

• Original: “Bilateral perihilar opacities could represent pulmonary edema.”

• Alternative: “Bilateral perihilar opacities suggest possible pulmonary edema.”

Axis-Specific Constraints: The alternative reference must convey the same level of uncertainty as the original reference.

## A.1.8 Language: Negation (Absolute)

Description: Radiologists differ in how they express negation, using either explicit statements about the absence of specific findings or implicit phrasings that convey the same information through positive statements about normality.

## Variations:

• Explicit: Expresses negation explicitly (e.g., “no pneumothorax, no pleural effusion”).

• Implicit: Conveys negation implicitly through positive statements about normality (e.g., “pleural spaces are clear”).

## Example 1:

• Explicit: “No pneumothorax. No pleural effusion.”

• Implicit: “Pleural spaces are clear.”

## Example 2:

• Explicit: “No focal consolidation. No pneumothorax. No pleural effusion.”

• Implicit: “The lungs are clear. Pleural spaces are normal.”

## Example 3:

• Explicit: “No mediastinal widening. No hilar adenopathy.”

• Implicit: “Mediastinal contours are within normal limits.”

Axis-Specific Constraints: Implicit variations must clearly convey the same clinical information as the corresponding Explicit negations. Ambiguous implicit statements that could be misinterpreted should be avoided.

## A.2 Prompts for the Generator and the Verifier

We instantiate REREF with two prompts: a Generator prompt that conditions an LLM on the original reference and the target axis-aligned variation, and a Verifier prompt that conditions an LLM on the original reference, the target variation, and the candidate alternative reference produced by the Generator, and asks it to assess (i) whether the alternative reference preserves the clinical interpretation of the original (i.e., clinical equivalence) and (ii) whether the perturbation is correctly simulated (i.e., the alternative actually reflects the target axis-aligned variation). Both prompts are used with the system message “You are a medical expert in radiology.”. Below, fields enclosed in angle brackets (e.g., <orig\_report>, <axis\_name>) are placeholders that are populated at runtime with the original reference, the per-axis definitions, allowed variations, examples, and axis-specific constraints from Appendix A.1, and the target variation.

The Generator prompt is constructed by concatenating the original reference, the per-axis description, allowed variations, and examples (Appendix A.1), the union of the default constraints listed below and the axis-specific constraint, and the target variation to generate. The default constraints are:

• Assume that all sections of the alternative reference other than the Findings and Impression (e.g., Indication, Technique) are identical to those of the original reference.

• Do not add any new information that is not substantiated by the original reference. In particular, if the original reference does not include the details required for perturbation along the specified axis, output the original reference unchanged.

• The alternative reference should remain clinically equivalent to the original reference.

• Introduce minimal changes beyond the specified axis along which the reference is being perturbed.

• Output only the alternative Findings and Impression sections, with no explanations or commentary.

## Prompt for the Generator LLM

![](images/d5775023c1a0967166c005378875fe31e60f091818fe2e1383ed567dfb21bc5b.jpg)

The Verifier is provided with the original reference, the same axis-specific definitions, allowed variations, and examples that were shown to the Generator, the target variation that the Generator was asked to produce, and the alternative Findings and Impression sections that were sampled. It returns a structured response with four boolean fields plus a free-text reasoning field: is\_clinically\_consistent (whether the alternative reference is clinically equivalent to the original), is\_valid\_perturbation (whether the alternative reference correctly simulates the target axis-aligned variation), does\_not\_contain\_required\_info (whether the perturbation is infeasible because the original reference does not contain the information required to support it; e.g., the original lacks any quantitative measurements but the target variation is Quantitative Granularity (High)), and is\_already\_in\_expected\_form (whether the original reference is already in the form expected after perturbation; e.g., the original is already structured but the target variation is Structure (Structured)). When either of the latter two fields is True, the Verifier returns is\_valid\_perturbation = True, so that the REREF pipeline distinguishes infeasible cases from incorrectly perturbed ones; the Verifier’s reasoning field allows us to subsequently flag and exclude infeasible studies from the corresponding axis-aligned analysis (Section 3).

## Prompt for the Verifier LLM

You will be given a ground-truth reference report for an imaging study, written by a radiologist, and an alternative version of the report, generated by perturbing the ground-truth report along one or more axes of variation (e.g., report structure, terminology preferences, level of detail). Your task is to check if the alternative version is clinically consistent with the ground-truth report. Additionally, you will be given the axis along which the alternative version was meant to be perturbed, and you should check if the alternative version is a valid perturbation of the ground-truth report for the specified axis.

## IMPORTANT:

• The alternative version should not add any new information that is not substantiated by the original ground-truth reference report.

• Minor paraphrases and reordering (within and across sections) of clinical content are allowed, as long as they are clinically identical in meaning.

• There can be cases where the perturbation was not made, either because the groundtruth report (i) does not contain sufficient information to make the perturbation without adding new information, or (ii) contains sufficient information but is already in the form expected after the perturbation along the specified axis. In both cases, return True for the is\_valid\_perturbation field (otherwise, return False). For case (i), also return True for does\_not\_contain\_required\_info; for case (ii), also return True for is\_already\_in\_expected\_form. Provide a detailed explanation in the reasoning field.

• Examples of case (i): the original report does not discuss technical quality, but the perturbation requires technical-quality discussion; the original report does not contain any quantitative measurements, but the perturbation requires them.

• Examples of case (ii): the original report is already structured, but the perturbation is to make it structured; the original report already lists all normal and abnormal findings, but the perturbation is to make the list of findings exhaustive.

## ======== GROUND-TRUTH REFERENCE REPORT ========

<orig\_report>

======== AXIS CHOSEN TO VARY ========

• NAME: <axis\_name>

• DESCRIPTION: <axis\_description>

• VARIATIONS:

– <variation\_1>: <variation\_1\_description>

– <variation\_2>: <variation\_2\_description>

• EXAMPLES: <axis\_examples> (same JSON-formatted examples shown to the Generator).

======== VARIATION GENERATED ========

<variation\_generated>

======== ALTERNATIVE FINDINGS + IMPRESSION ========

FINDINGS: <alt\_findings>

IMPRESSION: <alt\_impression>

## A.3 Inference Setup

We instantiate both the Generator and the Verifier with o3 (OpenAI, 2025), accessed via Microsoft Azure OpenAI under a setup compliant with the data use agreements for all of the datasets considered in our evaluations (Section 3). The Generator is sampled with temperature 0.7 and a maximum output length of 16382 tokens; the Verifier is run at temperature 0 to make its accept/reject decisions deterministic. Outputs of both LLMs are parsed into Pydantic schemas via instructor, with the Generator producing a (findings, impression) pair and the Verifier producing the four boolean fields and reasoning field described in Appendix A.2. For each (study, axis-aligned variation) combination, the generation–verification loop runs for up to 5 iterations; the loop terminates as soon as the Verifier returns is\_clinically\_consistent = True and is\_valid\_perturbation = True, and otherwise the last sampled alternative reference is retained together with its Verifier output, which we use downstream to filter out studies for which the Verifier could not certify a clean axis-aligned alternative reference (Section 3).

## B Radiologist Annotation Guide for REREF

Here, we provide additional details on the radiologist annotation tasks outlined in Section 2.3. Below, we include the full annotation guide provided to each of the radiologist annotators, including the Overview, Task 1 (Realism) Annotation Guide, and Task 2 (Clinical Equivalence) Annotation Guide.

## Overview of the Annotation Tasks

The quality of AI-generated radiology reports (in particular, the Findings and Impression sections) is often assessed using an automated evaluation metric, due to the challenges of obtaining clinician annotations at large scale. These automated metrics typically (i) compare each AI-generated report to a human-written reference report for the same imaging study and (ii) output a numerical score (e.g., between 0 and 1). For instance, an evaluation metric called GREEN (Ostmeier et al., 2024) is calculated as the proportion of clinical findings in the AI-generated report that match those in the human-written reference report, where the accuracy of each clinical finding is determined by an LLM (e.g., ChatGPT, Claude).

We are studying how robust such automated metrics are to stylistic variations in how different radiologists write radiology reports (e.g., report structure, expression of uncertainty, terminology preferences, level of detail). To do so, we made systematic edits to a large collection of human-written radiology reports from several chest X-ray (CXR) datasets (e.g., MIMIC-CXR, CheXpert Plus), simulating the various ways in which a practicing radiologist could have written the report. Each edit results in one alternative version of the original human-written CXR report. All such alternative reports are intended to be “clinically equivalent” to the original report that they were derived from, in terms of the key diagnostic findings and the downstream clinical decisions that would follow.

Your task is to evaluate whether these alternative reports are (i) realistic as standalone radiology reports (i.e., they look like what a practicing radiologist could have written) and (ii) clinically equivalent to the original reports. Your annotations will help us validate that the stylistic edits we study are grounded in real-world radiology reporting practice.

The annotation is organized into two phases, as described below.

## Task 1 (Realism) Annotation Guide

## Goal

Determine whether a given radiology report (the Findings and Impression sections) reads as a realistic, professionally written radiology report. Importantly, you should allow for the fact that certain stylistic and structural features of radiology reporting systematically vary across different practice environments.

## What You Will See

Each annotation example will include a full radiology report, divided into two parts:

1. Context: Indication, Comparison, Technique sections

2. Report: Findings, Impression sections

The realism rating should be based only on the “Report” portion. The “Context” portion is included solely as background information about the imaging study.

## What To Do

For each report, provide the following information, allowing for the fact that certain stylistic and structural features of radiology reporting vary across different practice environments:

## 1. Realism Rating

3 — Realistic: Reads as a report written by a practicing radiologist in routine clinical care.

2 — Minor Issues: Mostly realistic, but contains subtle phrasing or formatting choices that are slightly unusual (though clinically correct).

1 — Major Issues: Contains major errors, incoherent phrasing, or unnatural elements that a practicing radiologist would not write in a standard radiology report.

2. Free-Text Comments (required for 1 and 2): Please briefly explain your reasoning.

## Important Notes (Please Read)

• “Realistic” does not mean “you would personally write the report this way“. Radiologists have diverse reporting styles across training backgrounds, practice environments, etc. The question is whether some practicing radiologist might plausibly have written such a report.

• As noted above, the realism rating should only be given based on the Findings and Impression sections of the radiology report, and not the other auxiliary sections (Comparison, Indication, Technique).

• Because all reports have undergone a de-identification process (often automated in an imprecise manner), the reports shown to you may contain redactions (e.g., “XXXX” or “\_\_\_\_”) that render the reports awkward or grammatically wrong. Please ignore such de-identification artifacts when determining the realism of each report.

## Task 2 (Clinical Equivalence) Annotation Guide

## Goal

Given a pair of radiology reports about the same imaging study, determine whether they present the same clinical assessment—i.e., whether a treating physician reading either report would reach the same diagnostic conclusions and clinical decisions.

## What You Will See

Each annotation example will include the following parts, all pertaining to the same CXR study:

1. Context: Indication, Comparison, Technique sections

2. Report A: Findings, Impression

3. Report B: Findings, Impression

The equivalence rating should be based only on the “Report A” and “Report B” portions. The two reports may differ in style, structure, level of detail, or terminology. The “Context” portion is included solely as background information about the imaging study and is assumed to stay the same between the two reports.

## What To Do

For each pair of reports, provide the following information:

## 1. Clinical Equivalence Rating

Equivalent: A treating physician reading either report would identify the same findings, reach the same diagnostic conclusions, and make the same clinical decisions. Differences between the reports are limited to style, structure, terminology, or level of descriptive detail, and do not affect what clinical actions would follow. Examples of differences that are still “equivalent”:

• One report lists both normal and abnormal findings exhaustively, while the other mentions only the key abnormalities.

• One report specifies “airspace opacity”, while the other says “consolidation”.

• One report includes precise measurements, while the other uses qualitative descriptions (e.g., “moderate”).

Minor Discrepancy: The reports agree on the main diagnoses and no critical findings are missing from either report, but there is a small difference that a treating physician might notice. The difference is unlikely to change clinical management. Examples:

• One report mentions a minor incidental finding (e.g., mild degenerative changes) that the other omits.

• One report characterizes a finding as “mild”, while the other says “mild-tomoderate”.

• One report includes a differential possibility that the other does not, but the primary assessment is the same.

Major Discrepancy: The reports differ in major ways that could lead a treating physician to different clinical decisions. Examples:

• One report identifies a finding that the other does not mention at all (e.g., a pulmonary nodule, pneumothorax, or consolidation).

• The reports differ in the characterization of a finding in a way that would change urgency or management (e.g., “small” vs. “large” effusion).

• One report suggests a diagnosis that the other contradicts or omits entirely.

2. Free-Text Comments (required for Minor and Major Discrepancy): Describe the specific differences that led you to your rating.

## Important Notes (Please Read)

• Two reports can differ substantially in wording, structure, and level of detail while still being clinically equivalent. The question is not whether the reports are identical, but whether they would lead to the same clinical actions.

• Please do not determine the equivalence rating based on the realism of each report. Determining the realism of each report is reserved as a separate annotation task. Example: Suppose that two reports contain exactly the same text for describing the findings, but one report writes all of them under the Findings section (i.e., no Impression section), while the other writes all of them under the Impression section (i.e., no Findings section). The former reads less realistic than the latter, as radiology reports generally always have an Impression section, but since they contain identical findings and would lead to the same actions, this pair should be labeled as “Equivalent”.

• A report that mentions fewer findings is not automatically discrepant. Some radiologists routinely omit normal findings or reduce descriptive detail without altering the clinical interpretation and conclusion. Only flag omissions that could cause a treating physician to miss something actionable.

• Differences in how uncertainty is expressed (e.g., “concerning for” vs. “suspicious for”) are equivalent unless they convey meaningfully different levels of confidence that would change clinical decision making.

• Because all reports have undergone a de-identification process (often automated in an imprecise manner), the reports shown to you may contain redactions (e.g., “XXXX” or “\_ ”) that render the reports awkward or grammatically wrong. Please ignore such de-identification artifacts when determining the equivalence between reports in each pair.

Table C1: Summary of the four CXR datasets used in our evaluation (Section 3). For each dataset, we report the final number of examples retained (i.e., those with a Findings section in the reference report) and summary statistics on the number of images per study.
<table><tr><td colspan="3"></td><td colspan="4"># Images in Study</td></tr><tr><td>Dataset</td><td># Studies</td><td>Total Images</td><td>Mean ± Std</td><td>Median</td><td>Min</td><td>Max</td></tr><tr><td>CheXpert Plus</td><td>61</td><td>72</td><td> $1 . 1 8 \pm 0 . 4 6$ </td><td>1</td><td>1</td><td>3</td></tr><tr><td>IU X-ray</td><td>590</td><td>1,223</td><td> $2 . 0 7 \pm 0 . 2 8$ </td><td>2</td><td>2</td><td>4</td></tr><tr><td>MIMIC-CXR</td><td>2,882</td><td>4,633</td><td> $1 . 6 1 \pm 0 . 7 0$ </td><td>1</td><td>1</td><td>5</td></tr><tr><td>ReXGradient-160K</td><td>10,000</td><td>17,029</td><td> $1 . 7 0 \pm 0 . 7 3$ </td><td>2</td><td>1</td><td>29</td></tr><tr><td>Total</td><td>13,533</td><td>22,957</td><td></td><td></td><td></td><td></td></tr></table>

## C Additional Details on Datasets

In Table C1, we provide a summary of the number of imaging studies included in each of the four datasets used in our evaluation (Section 3). For all datasets, we pre-extract the key sections from each reference report (i.e., the Indication, Comparison, Technique, Findings, and Impression sections) to identify the Findings section to use for RRG model evaluation. For all datasets except MIMIC-CXR, all of the sections are already available in parsed form, so we use them directly as available. For MIMIC-CXR, we extract all sections by prompting Claude Sonnet 4.5 (Anthropic, 2025) using the prompt below, which is adapted from the prompt used by Zambrano Chaves et al. (2025)<sup>5</sup>:

## Prompt for Section Extraction from MIMIC-CXR

## System Prompt:

You are an expert medical assistant AI capable of modifying clinical documents to user specifications. You make minimal changes to the original document to satisfy user requests. You never add information that is not already directly stated in the original document.

## Section Extraction Prompt:

Extract the following sections from the input radiology report: [section\_names]. Leave an extracted section as "N/A" if it does not exist in the original report. The output should be in JSON format. An Indication section can refer to the History, Indication or Reason for Study sections in the original report. An Impression section can include the Recommendations and Conclusion sections in the original report. The extracted sections should be verbatim from the original report and should not overlap with each other. Remove any linebreaks and whitespace that have been added only for better readability.

Examples of inputs and expected outputs (when e.g., extracting the [‘Examination’, ‘Indication’, ‘Findings’, ‘Impression’] sections):

## INPUT:

EXAMINATION: XR CHEST AP PORTABLE

INDICATION: Small right apical pneumothorax after lung biopsy.

FINDINGS: Single portable view of the chest was obtained. Copared with 10:42 AM. The small right apical pneumothorax has decreased slightly in size, the improvement best appreciated laterally where it now measures 10 mm compared to 14 mm before. At the lung apex it now measures 1.6 and compared to 2.1 cm previously. A subtle right apical pulmonary contusion is grossly stable. Minor chest wall emphysema along the right exilla has not changed significant delay. There is no metastatic shift. No pleural effusion is evident.

OUTPUT: {

"EXAMINATION": "XR CHEST AP PORTABLE.",

"INDICATION": "Small right apical pneumothorax after lung biopsy.", "FINDINGS": "Single portable view of the chest was obtained. The small right apical pneumothorax measures 10mm. At the lung apex it measures 1.6cm. A subtle right apical pulmonary contusion is grossly stable. Minor chest wall emphysema is noted along the right exilla. There is no metastatic shift. No pleural effusion is evident.",

"IMPRESSION": "N/A"   
}   
\*\*Here is the report that you should process\*\*:   
{Report}

## D Additional Details on Models

Here, we provide additional details on how we generate the reports from each model considered in our study. By default, we generate the reports from each model via greedy decoding. For the models that support multi-image inputs, we provide as many CXR views as feasible from each imaging study, up to 10 images (or whatever number of images each model can support). If more images are present, we prioritize the inclusion of frontal over lateral images. We note that there are no imaging studies that have more than 10 images in the test sets for MIMIC-CXR, CheXpert Plus, and IU X-ray. In ReXGradient-160K, there are only 2 out of the 10K imaging studies in the test set that contain more than 10 images for an imaging study (one with 21 images, the other with 29 images). For all inference runs, we use up to 8 NVIDIA A6000 GPUs, each with 48GB of memory.

## D.1 MAIRA-2 (Bannur et al., 2024)

For MAIRA-2 (Bannur et al., 2024), we use the official checkpoint on HuggingFace<sup>6</sup> and follow the instructions in the model card for generating the Findings section in the “ungrounded” setting (i.e., the model does not generate bounding boxes as evidence for visual findings localizable in the CXR images). For Findings generation, we use the prompts provided in Table B.1 of the paper, providing the Indication, Technique, and Comparison sections of the report as additional context when available for the given study. Any missing sections are marked as “N/A”. As MAIRA-2 supports inputs with up to one frontal image and one lateral image (i.e., maximum of two images corresponding to different views), we always include both views if available. If one of the two views is missing, we provide as input whichever view is available.

## D.2 MedGemma-4B (Sellergren et al., 2025)

For MedGemma-4B (Sellergren et al., 2025), we use the official checkpoint for the instruction-tuned model on HuggingFace<sup>7</sup>. For Findings generation, we use our own custom prompt, additionally providing the Indication, Technique, and Comparison sections if available for the given study: “Generate the FINDINGS section of the radiology report for this CXR image.\n\nBelow are the prefilled sections of the report that you can consider:\n\nINDICATION: ⟨indication⟩\n\nTECHNIQUE: ⟨technique⟩\n\nCOMPARISON: ⟨comparison⟩\n\nDO NOT generate anything other than the FIND-INGS section of the report, and DO NOT repeat the details of the prefilled sections in any way.” As MedGemma-4B is originally developed for single-image input settings, we only provide the frontal view of the CXR as input.

## D.3 CheXOne (Zhang et al., 2026b)

For CheXOne (Zhang et al., 2026b), we use the official checkpoint on HuggingFace<sup>8</sup>. For Findings generation, we use the prompts provided in Figure S12 of the paper: “Given the indication ⟨indication⟩, write an example findings section for the CXR.” We always sample generations from the model in its “reasoning mode” (see Section 2.7 of the paper), where we append the following at the end of the main prompt: “Please reason step by step and put your final answer within \boxed{}”. For imaging studies with multiple CXR views, we select up to 10 images to provide as input, prioritizing the frontal images over lateral images.

## D.4 MedVersa (Zhou et al., 2026)

For MedVersa (Zhou et al., 2026), we use the official checkpoint on HuggingFace<sup>9</sup>. We select one of the 10 prompts for Findings generation listed in Table 5 of their arXiv preprint version<sup>10</sup> of the paper: “How would you characterize the findings from ⟨<img0>...<imgN>⟩?” For imaging studies with multiple CXR views, we select up to 10 images to provide as input, prioritizing the frontal images over lateral images.

## D.5 CheXagent-2-3B (Chen et al., 2024)

For CheXagent-2-3B (Chen et al., 2024), we use the official checkpoint on HuggingFace<sup>11</sup>. For Findings generation, we use the prompt provided in the demo script in the official GitHub repository<sup>12</sup>: “Given the indication ⟨indication⟩, write a structured findings section for the CXR.” For multi-image inputs, we include only up to two images per study, following the authors’ approach on their CheXbench evaluation suite (Chen et al., 2024)<sup>13</sup>.

## D.6 Med-CXRGen (Zhang et al., 2024)

For Med-CXRGen (Zhang et al., 2024), we use the official checkpoints on HuggingFace, which are released as two separate models for the Findings (Med-CXRGen-F<sup>14</sup>) and Impression (Med-CXRGen-I<sup>15</sup>) sections. For Findings generation, we use the prompt provided in the official codebase: “Provide a detailed description of the findings in the radiology image.” For Impression generation, we use the analogous prompt with “findings” replaced by “impression”. Following the reference implementation, we additionally append the Indication, Technique, and Comparison sections of the report as clinical context to the prompt when available. For imaging studies with multiple CXR views, we follow the authors’ multi-image strategy and stitch up to 4 images horizontally with a 10-pixel black separator between adjacent images, prioritizing the key (frontal) view first followed by remaining frontal views over lateral views.

## D.7 Libra (Zhang et al., 2025b)

For Libra (Zhang et al., 2025b), we use the official checkpoint on HuggingFace<sup>16</sup>. For Findings generation, we use the prompt provided in the official codebase: “Provide a detailed description of the findings in the radiology image.” As with Med-CXRGen, we additionally append the Indication, Technique, and Comparison sections as clinical context to the prompt when available. Libra is designed to take as input a paired set of CXR images consisting of one image from the current study and one image from a prior study; as our test sets do not include prior studies, we follow the authors recommended fallback and duplicate the current image as a placeholder for the prior. Accordingly, we provide as input only the key (frontal) view of the current study, since Libra was not trained or evaluated on multi-view input settings.

## D.8 LLaVA-Rad (Zambrano Chaves et al., 2025)

For LLaVA-Rad (Zambrano Chaves et al., 2025), we use the official checkpoint on HuggingFace<sup>17</sup>, loaded on top of the lmsys/vicuna-7b-v1.5 base language model as specified by the authors. For Findings generation, we use the prompt provided in the official demo script: “Describe the findings of the chest x-ray.” As LLaVA-Rad was developed and evaluated only on single-image inputs, we provide as input only the key (frontal) view of the CXR.

![](images/649553e07ce9554ae62680f96f9c4884c897a385951164ad49882bd061bfa87e.jpg)  
Figure E1: Model rankings can be sensitive to the reporting practices of the reference (Section 4.3), even when we limit to reversals that also change the 95% bootstrap CI-based comparison for each model pair. Aggregating the results across all datasets and model pairs, we find that for 7 out of 9 metrics, pairwise ranking reversals are often accompanied by a change in the 95% bootstrap CI-based comparison for each pair. Each box plot shows, across the 12 axis-aligned perturbations, the percentage of all $4 \times { \binom { 9 } { 2 } } = 1$ 44 (dataset, model pair) comparisons whose pairwise ordering by mean score reverses relative to the original reference and whose 95% bootstrap CI-based comparison also changes (Section 4.3). The diamond-shaped markers indicate the max and min values across all perturbations. The cyan-colored box plots indicate the clinical metrics.

## D.9 CheX-MIMIC (Chambon et al., 2024)

For CheX-MIMIC (Chambon et al., 2024), we use the official Findings generation baseline checkpoint released alongside CheXpert Plus (Chambon et al., 2024) on HuggingFace<sup>18</sup>. As CheX-MIMIC is a purely image-conditioned encoder-decoder model that does not accept any text prompt, no instruction or clinical context is provided alongside the input image. As CheX-MIMIC was developed and evaluated only on single-image inputs, we provide as input only the key (frontal) view of the CXR.

## E Additional Results: Reporting Practice Variations Can Impact Conclusions from Model Comparison (Section 4.3)

Figure 7(b) includes a pairwise ranking reversal whenever the ordering of two models by average score reverses relative to the original reference, irrespective of the magnitude of the underlying score gap. As some of these gaps are small enough to fall within noise, we additionally report in Figure E1 the subset of reversals for which the 95% bootstrap CI-based comparison for each pair also changes, i.e., where the perturbation alters the conclusion one would draw about the pair rather than only its ordering. When pooled across all metrics, datasets, and perturbations, roughly 60% of all pairwise ranking reversals (in Figure 7(b)) are accompanied by a change in the CI-based comparison, and the metrics most affected remain broadly similar (median reversal in Figure E1: RadGraph-F1: 7.6%; RadCliQ-v1: 7.0%; RaTEScore: 4.7%). The two notable exceptions are CRIMSON and CheXbert-F1, for which the reversals are rarely accompanied by a significant change in pairwise relative mean scores. For CRIMSON, such a result is consistent with its comparatively low overall sensitivity to reporting practice variations, as observed in Figure 4.

## F Additional Results: Compound Reporting Practice Variations

While we primarily measure metric sensitivity to reporting practice variations along a single axis in our main experiments (Section 4), real-world radiology reports likely reflect multiple reporting practices simultaneously and in a correlated manner. For instance, a radiologist who prefers to write concise reports that only discuss the key abnormal findings (Findings Scope (Minimal)) may also be less detailed in describing their findings across anatomical substructures (Anatomical Granularity (Low)). In contrast, a radiologist who prefers to write exhaustive reports that thoroughly discuss all normal and abnormal findings (Findings Scope (Exhaustive)) may also be more thorough in describing their findings across all anatomical substructures (Anatomical Granularity (High)).

Based on this intuition, we repeat the experiments in Section 4 under the setup where references are subject to compound reporting practice variations (i.e., applying multiple axis-aligned perturbations).

![](images/07c06704759d03fd42e6aad1b1d2f0aaad39bd06e9cc00ad851ea4837555d9a5.jpg)  
Figure F1: Compound reporting practice perturbations along multiple axes (2-step, 3-step) generally result in larger deviations from the model rankings under the original reference, compared to the single-axis perturbation (1-step) setting. Here, we show the results under the minimalist persona, which composes Findings Scope (Minimal) with Anatomical Granularity (Low), followed in the 3-step setting by one additional axis (indicated on the x-axis; Appendix F). Each bar reports the percentage of all $4 \times { \binom { 9 } { 2 } } = 1 4 4$ (dataset, model pair) comparisons whose pairwise ordering reverses relative to the original reference. The darker lower segment is the subset of those reversals for which the gap between the two models’ mean scores also changes between the three cases distinguished by a 95% bootstrap confidence interval in relative score: significantly favoring one model, significantly favoring the other, or statistically indistinguishable from zero (tie).

To do so, we use the above example and consider two radiologist personas: (i) a minimalist who aligns with Findings Scope (Minimal) and Anatomical Granularity (Low), and (ii) a maximalist who aligns with Findings Scope (Exhaustive) and Anatomical Granularity (High). For each persona, we also allow for variations along the remaining axes (e.g., Structure, Terminology, Hedging). Thus, the alternative references we use for analysis are subject to up to three reporting practice perturbations.

![](images/9bf148655bd018db2404c4b79aa8217c738ff97879b9fe4e4e6ae171f57a0ab1.jpg)  
Figure F2: Compound reporting practice perturbations along multiple axes (2-step, 3-step) generally result in larger deviations from the model rankings under the original reference, compared to the single-axis perturbation (1-step) setting. Here, we show the results under the maximalist persona, which composes Findings Scope (Exhaustive) with Anatomical Granularity (High), followed in the 3-step setting by one additional axis (indicated on the x-axis; Appendix F). Each bar reports the percentage of all $4 \times { \binom { 9 } { 2 } } = 1 4 4$ (dataset, model pair) comparisons whose pairwise ordering reverses relative to the original reference. The darker lower segment is the subset of those reversals for which the gap between the two models’ mean scores also changes between the three cases distinguished by a 95% bootstrap confidence interval in relative score: significantly favoring one model, significantly favoring the other, or statistically indistinguishable from zero (tie).

Given that the specific order in which the Findings Scope and Anatomical Granularity perturbations are applied matters, we account for both directions when computing the scores under the compound alternatives. For instance, in the minimalist setting, we perturb each reference report once in the order of Findings Scope (Minimal) → Anatomical Granularity (Low), and once in the opposite direction, and then average the scores calculated against each resulting alternative reference. For the 3-step setting, we apply one further axis-aligned perturbation (e.g., Terminology) to each of these 2-step alternatives, again averaging over the two orderings of the first two axes.

Since not every perturbation is feasible for every imaging study (as determined by the Verifier in REREF), we score the model-generated reports on each study against the most heavily perturbed alternative reference available for it within a given setting. In the 3-step setting, we score against the 3-step alternative where feasible, followed by the 2-step alternatives, then the single-axis (Findings Scope and Anatomical Granularity) alternatives (averaged when both are feasible), and finally their original-reference scores, as considered in Section 4.3. In the 2-step setting, we apply an analogous procedure, starting from the 2-step alternatives.

Compared to the single-axis perturbation case, we observe that deviations in model ranking from the original-reference setting tend to be greater as we perturb the references further (1-step → 2-step → 3- step), while the exact trends vary across metrics (Figures F1–F2). Notably, some 3-step perturbations lower the reversal rate relative to the 2-step case, likely because the third perturbation partially counteracts the effect of the first two perturbations. For instance, Negation (Explicit) can expand the discussion of findings in the reference by adding explicit statements about absent abnormalities (e.g., “pleura appear normal” → “no pneumothorax, no pleural effusion”), which can counteract the first two perturbations under the minimalist persona and lowers the pairwise ranking reversal rate (e.g., CRIMSON: 6.2% (2-step) → 4.9% (3-step; Negation (Explicit)); Figure F1).