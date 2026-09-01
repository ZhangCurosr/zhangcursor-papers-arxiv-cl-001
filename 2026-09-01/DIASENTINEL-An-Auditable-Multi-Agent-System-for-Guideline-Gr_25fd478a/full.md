# DIASENTINEL: An Auditable Multi-Agent System for Guideline-Grounded Diabetes Risk Screening

Yung Wei Shueh<sup>1</sup>\*, Zhi-Jie Chen<sup>1</sup>\*, Chia-Hsuan Hsu<sup>1</sup>, Hsin-Ling Hsu<sup>1</sup>, Donghua Zhang<sup>4</sup>, Chenwei Wu<sup>4</sup>, Jun-En Ding<sup>2</sup>, Tongze Zhang<sup>3</sup>, Shihao Yang<sup>2</sup>, Pengfei Hu<sup>3</sup>, Fang-Ming Hung<sup>1</sup>, Feng Liu<sup>2†</sup>

<sup>1</sup>Far Eastern Memorial Hospital, Taiwan <sup>2</sup>Rutgers University, USA <sup>3</sup>Stevens Institute of Technology, USA <sup>4</sup>University of Michigan, USA

## Abstract

Large language models (LLMs) offer promising clinical decision support but remain vulnerable to hallucinated facts, unsupported recommendations, and citation errors. We present DI-ASENTINEL, a fully on-premise multi-agent system for one-year type 2 diabetes mellitus (T2DM) risk screening and guideline-grounded report generation from electronic health records (EHRs). The system integrates calibrated risk prediction, deterministic clinical signal extraction, Reciprocal Rank Fusion over American Diabetes Association (ADA) guidelines, and a hybrid verification layer combining rule-based checks with LLM entailment. The demonstration provides a real-time batch-screening dashboard and an interactive patient report interface with cited recommendations, verification results, and raw EHR comparison. DIASEN-TINEL demonstrates a practical framework for reliable, auditable, and privacy-preserving LLM-based clinical decision support. Our system demonstration can be found at https: //diasentinel-demo.onrender.com/

## 1 Introduction

Type 2 diabetes mellitus (T2DM) is among the most prevalent chronic diseases worldwide. According to the International Diabetes Federation, more than 500 million adults were living with diabetes in 2024, corresponding to approximately one in nine adults globally (International Diabetes Federation, 2025). T2DM is often preceded by an asymptomatic but modifiable high-risk period, during which early identification and intervention can delay or prevent disease onset and reduce subsequent complications (Diabetes Prevention Program Research Group, 2002; American Diabetes Association Professional Practice Committee for Diabetes, 2026).

In clinical practice, routine clinical documentation is required for patients with potential chronic diseases, including measurements of glycated hemoglobin (HbA1c), fasting plasma glucose, body mass index (BMI), blood pressure, and lipid profiles. Concurrently, an effective risk indicator is needed to assess the near-term risk of type 2 diabetes mellitus (T2DM). Despite the significant advances of LLMs in the medical domain and EHRbased diagnosis in recent years (Griot et al., 2025; Ding et al., 2024), most models have focused on algorithmic improvements without practical clinical deployment and application. In particular, LLMs introduce three major risks in medical applications. First, their predicted probabilities may be poorly calibrated and thus fail to reflect the actually observed disease incidence. Second, the model may hallucinate (Asgari et al., 2025; Kim et al., 2025) laboratory values, clinical findings, or recommendations that have no basis in the patient’s medical record or clinical guidelines. Third, retrievalaugmented generation (RAG) (Wu et al., 2025) may still associate otherwise valid recommendations with incorrect sources, thereby leading to citation drift. These limitations reduce the interpretability, verifiability, and clinical reliability of LLM-based decision support.

To better reflect real-world clinical practice, we design a deployable, end-to-end LLM-based clinical decision support system, DIASENTINEL, that decomposes the diagnostic workflow into a sequence of auditable, evidence-grounded subtasks. This modular design not only improves transparency and traceability but also enables more efficient retrieval and reasoning within each system component. More specifically, our contributions are threefold: (1) we develop a calibrated LLM-based risk predictor by fine-tuning Qwen2.5- 14B (Yang et al., 2024) with low-rank adaptation (LoRA) (Hu et al., 2022), aligning its estimated probabilities with the observed one-year T2DM incidence and supporting clinically meaningful risk stratification; (2) we introduce a two-stage retrieval pipeline over the American Diabetes Association (ADA) Standards of Care in Diabetes guideline corpus that combines dense retrieval and crossencoder reranking through Reciprocal Rank Fusion (RRF) (Cormack et al., 2009), mitigating the length bias observed when the reranker is used alone; and (3) we integrate an in-the-loop verification agent that combines four deterministic checks with an LLM-based entailment check to audit factual consistency, unsupported content, and citation drift before the report is presented to clinicians.

Following this design, we present DIASEN-TINEL<sup>12</sup>, a fully on-premise multi-agent clinical decision support system for batch screening of oneyear new-onset T2DM risk and on-demand generation of individualized, guideline-grounded reports. Implemented with LangGraph, DIASENTINEL applies LLMs to risk estimation, report generation, and entailment-based verification, while all other components use deterministic rules or retrievalonly methods.

## 2 DIASENTINEL System Architecture

As shown in Figure 1, DIASENTINEL is an onpremise clinical decision support pipeline orchestrated with LangGraph (Duan and Wang, 2024). It separates population-level screening from patientspecific report generation through two modes:

• Batch risk screening. The Risk Function assigns each eligible patient a calibrated oneyear T2DM risk probability and a high-, medium-, or low-risk tier.

• Detailed report generation. Given a clinician-selected patient, the system retrieves longitudinal EHR records, extracts clinical signals, retrieves guideline evidence, synthesizes a structured report, and verifies it before display.

In this system architecture, DIASENTINEL applies LLMs only to prediction, report synthesis, and semantic verification, while keeping EHR retrieval, factual extraction, numerical comparison, and evidence retrieval deterministic. Specifically, the LLM-based modules include the Risk Function, the Synthesizer, and the entailment check in the Verification layer. For screening, the Risk Function uses a LoRA fine-tuned Qwen2.5-14B model (Yang et al., 2024) to estimate one-year newonset T2DM risk from structured EHR-derived features. Token-level log probabilities are converted into a raw risk estimate via softmax and calibrated with Platt scaling (Platt, 1999). The calibrated score is cached and reused across the dashboard, report summary, and audit checks to ensure consistency.

## 2.1 Deterministic Clinical Tracking

Real-Time Retrieval System. To support realtime diagnosis, the physician first retrieves the patient’s historical medical records from the hospital database via SQLAlchemy<sup>3</sup>. The retrieved context comprises SOAP (Subjective, Objective, Assessment, Plan) records (Sorgente et al., 2005), lab test results, and nursing assessments collected within a one-year time window. Next, we design an Explanation Agent to extract clinically interpretable risk signals using six deterministic threshold rules. These rules cover HbA1c, fasting glucose, BMI, blood pressure, low-density lipoprotein (LDL) cholesterol, and a metabolic-syndromerelated lipid pattern defined by elevated triglycerides with low high-density lipoprotein (HDL). When a rule is triggered, the agent emits a structured signal containing the variable, observed value, severity level, and a concise rationale, such as “An HbA<sub>1c</sub> value of 6.1% falls within the prediabetes range”. The resulting signals are then ranked by clinical priority so that the Synthesizer can present the most relevant risk factors first.

Longitudinal Trend Tracking Annotation. In parallel, the Trend Agent summarizes longitudinal changes over the same 365-day window. For each monitored variable, it compares the earliest and most recent values using variablespecific noise-tolerance bands, assigns a trend label such as stable, rising, improving, or insufficient\_data, and records whether a clinically meaningful cut point has been crossed. The agent outputs both structured trend facts and a short natural-language summary, denoted as summary\_nl. The Synthesizer is instructed to reuse this summary verbatim rather than re-inferring the trend from raw values, which further limits opportunities for numerical hallucination.

![](images/82d227b07e7df3c0936973570e8d75aada45fe22e7f3fb9496883658c68bf617.jpg)  
Figure 1: The pipeline architecture. Blue: LLM-based; gray: deterministic; purple: hybrid.

## 2.2 Knowledge-based Guideline Retrieval

To ground report generation in clinical guidelines, the Evidence Agent builds a local retrieval corpus from the ADA Standards of Care in Diabetes (2026). The guideline is split into 382 section-aware chunks (512 tokens max) using Docling (Auer et al., 2024), embedded with bgem3 (Chen et al., 2024), and indexed in Chroma with source, section, and page metadata. At inference, the agent retrieves dense candidates, reranks them with bge-reranker-v2-m3 (Chen et al., 2024), and fuses both rankings via RRF to obtain the final evidence\_chunks; if reranking is unavailable, dense retrieval is used instead.

The Synthesizer (Qwen2.5-14B-Instruct) generates a five-part clinical report. Each recommendation is explicitly bound to retrieved metadata (source, section, and page), while unsupported statements are rendered as generic, non-cited guidance. This metadata-driven design prevents fabricated citations and grounds all referenced recommendations in the retrieved ADA evidence.

## 2.3 Clinically Reliable Verification

Before the report is presented to clinicians, the Verification Agent validates it against upstream structured outputs. Four deterministic checks ensure consistency between the report and (i) predicted risk score and tier, (ii) EHR-derived numerical values, (iii) longitudinal findings, and (iv) retrieved guideline citations, detecting common errors such as altered values, incorrect risk stratification, unsupported trends, and citation drift.

An additional Qwen2.5-14B-Instruct entailment check verifies whether recommendations are semantically supported by the retrieved guideline passages. Each check returns pass, flag, or skipped. The agent is strictly annotate-only: it records all results in an append-only JSON Lines (JSONL) audit log and displays them to clinicians without modifying the generated report, preserving transparency and clinician authority.

## 3 Demonstration

In this section, we demonstrate DIASENTINEL through two interfaces: (1) a daily batch-screening dashboard and (2) a per-patient detailed report page. Our goal is to assist metabolic-care clinicians in more rapidly assessing patients’ metabolic conditions and their risk of new-onset diabetes.

## 3.1 Daily batch-screening

As illustrated in Figure 2, we developed a clinicianfacing tracking dashboard for batch patient screening (Figure 2 (a)) and risk monitoring (Figure 2 (c)). Clinicians can initiate screening by clicking the “Run Screening” button, after which the system evaluates all eligible patients in batch. The frontend displays real-time progress via server-sent events (SSE), while the backend sequentially performs risk prediction and data-completeness assessment for each patient. After screening, the dashboard stratifies patients into high-, medium-, and low-risk groups.

More specifically, each patient receives a datacompleteness indicator reflecting input-data quality. This score is derived from four key clinical measurements, namely HbA1c, fasting glucose, BMI, and blood pressure, with each available measurement contributing one point, for a total of 0 to 4. Patients are labeled as high completeness (3–4 measurements), medium completeness (2), or in-

complete (0–1).

The dashboard also reports the execution status of each screening run. When new laboratory data are detected, it reports “N new patients were included in this screening run”; otherwise, it displays “No new laboratory data were detected; previous screening results were reused,” reflecting the high-watermark-based idempotent skipping mechanism. Furthermore, patients with confirmed diabetes are excluded from the screening cohort. Because ICD-10 codes alone (E08–E13) are insufficient for confirmation, the system additionally reviews diagnostic notes, medication history, and abnormal glucose-related laboratory results. Cases meeting these criteria are flagged as potential newly developed T2DM and displayed on the frontend (Figure 2(b)). This panel stratifies cases by flagging time into three views: “newly identified today,” “added within the past 30 days,” and “all cases.”

## 3.2 Interactive Patient Reports

When a clinician selects a patient from any risk tier, the system opens a detail page presenting a complete clinical report. The top of the page shows the five-part report generated by DIASEN-TINEL (Figure 3), followed by a panel reporting the verification-layer checks (Figure 4). Below it, the raw EHRs, including recent encounters, laboratory results, and nursing assessments, are shown in tabular form. Crucially, these records are fetched via direct SQL queries and never pass through an LLM, so the source data shown to clinicians is guaranteed to be faithful to the underlying database rather than paraphrased by the model.

As an example, Figure 3 demonstrates Key Risk Factors, which lists the signals extracted by the Explanation Agent, such as prediabetes, abnormal fasting glucose, and overweight, ordered by clinical urgency, while Longitudinal Changes surfaces the Trend Agent’s summary.

Sample case: the patient #12345 (male; BMI 28.3,   
HbA1c 6.1%, fasting glucose 112, blood pressure 140/85,   
LDL 162).

Furthermore, the remaining sections, such as Clinical Recommendations, are grounded in the retrieved ADA guidelines. Each statement is annotated with its corresponding source and page number, and the report ends with a disclaimer. To keep the interface responsive, the detail page reuses the risk probabilities cached during batch screening and reruns only the downstream agents when necessary, generating the full report on demand.

## 3.3 Verification Artifacts

On the detailed report page, the verification results appear in a collapsible panel between the generated report and the raw EHR tables; the panel is collapsed when all checks pass and expands automatically when any item is flagged. The panel header summarizes the overall status (e.g., Verified or N issues flagged); expanding it lists one row per check finding across the five check types, each with its status and a one-line rationale (the trend check reports one row per lab series). For a flagged item, the interface additionally juxtaposes what the report states with what the upstream facts support, letting the physician inspect the discrepancy at a glance, as summarized below.

## Human-in-the-Loop Verification

The system automatically flags potential errors, such as tampered values or incorrect citation sources, while the physician retains final authority over the diagnosis via the ACCEPT and OVERRIDE actions.

The panel in Figure 5 is itself produced by an injected error, following the injected-error protocol summarized in Appendix Table 8: we tamper with an example report (patient #70003) to claim that fasting blood glucose is rising, whereas the structured trend fact records it as stable.

## 4 System Evaluation

## 4.1 Evaluation Metrics

We evaluate DIASENTINEL along three dimensions. For risk prediction and stratification, we report the area under the receiver operating characteristic curve (AUROC) and the area under the precision–recall curve (AUPRC), with particular emphasis on AUPRC for the low-prevalence newonset T2DM outcome (≈ 6%), and assess calibration via the Brier score $\begin{array} { r } { \mathbf { B S } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( p _ { i } - o _ { i } ) ^ { 2 } } \end{array}$ where $p _ { i }$ is the calibrated predicted risk and $o _ { i }$ the observed outcome; at a fixed threshold we additionally report accuracy, precision/positive predictive value (PPV), recall/sensitivity (Se), specificity (Sp), negative predictive value (NPV), and F1-score, with the high-risk threshold selected via Youden’s $J = \mathrm { S e } + \mathrm { S p } - 1$ and the medium-risk threshold chosen under a minimum-sensitivity constraint to prioritize screening recall. For guideline retrieval, we report Recall@5 and mean reciprocal rank (MRR) at the chunk level, Chapter@5 at the chapter level, and Cohen’s $\begin{array} { r } { \kappa = \frac { p _ { o } - p _ { e } } { 1 - p _ { e } } } \end{array}$ between two independent annotators (where $p _ { o }$ is the observed and $p _ { e }$ the chance-expected agreement) to assess evaluation-set reliability. For verification reliability, we report the false-alarm rate on clean reports, the interception rate on manually injected errors, and the stability of the LLM-based entailment check under repeated fixed-input runs.

![](images/0a50fc66760abe5eaa36e57ddb161f8f74ad68107e8dd3e1ab3aee3069cf9416.jpg)

![](images/d4545e45cfcdbedb6fb976b24ac9c678469849d1aab026b449768544ae4a3e6b.jpg)

![](images/e34774726ba5f83f82899f27244d279a961b711051757e630f2957fbbb5c3251.jpg)

Figure 2: The batch-screening interface shows (a) one-click assessment and run statistics, (b) an audit panel for suspected new-onset T2DM cases, and (c) a calibrated risk dashboard with data-completeness and screening-status indicators.  
![](images/ece40f60c77fe5e9380835ab10dc7adafbd88cba3b7386b903d5a9272f9c5613.jpg)  
The above content is generated by an Al assistant based on medical records and is for clinica reference only.  
Figure 3: The five-part clinical report generated by DI-ASENTINEL, including a risk summary, key risk factors, longitudinal trends, ADA-grounded recommendations with citations, and a disclaimer (see §3.2).

![](images/d2cd70caa304c6fb744bf7e6bfbc849d11c57dcf4676af1d2d0df93a74fc3608.jpg)  
Figure 4: Lower half of the detail page: the verificationlayer check-results panel, followed by the raw EHR tables

![](images/f2dcf134b34318cbe3f4fb70a096342fa650e4697c4a173a91830a77bb6e19cd.jpg)  
Figure 5: The flagged verification panel shows 11 of 12 checks passing, with the Longitudinal Trend discrepancy displayed alongside the expected value and ACCEPT/OVERRIDE controls.

## 4.2 Performance of Risk Function

To prevent the model output from exhibiting overconfidence (Guo et al., 2017), we apply Platt scaling $( a = 1 . 0 0 4 9 , b = - 1 . 5 6 9 3 )$ to calibrate the predictions against the observed one-year disease incidence of approximately 6% (Figure 6). The resulting calibrated probability serves as a unified risk score used for database storage, dashboard visualization, and automated report generation. From the validation set, we stratify patients into risk tiers based on thresholds for potential highrisk cases: high risk $( p \ge 0 . 0 5 6 )$ , medium risk $( 0 . 0 3 5 \le p < 0 . 0 5 6 )$ , and low risk $( p < 0 . 0 3 5 )$ Our results show that the high-risk threshold maximizes Youden’s J, yielding 72% sensitivity and 62% specificity in the validation set, while the medium-risk threshold is selected as the value achieving at least 90% sensitivity. This sensitivityoriented design prioritizes the identification of potential high-risk cases while accepting additional false positives, consistent with the practical needs of clinical screening. Although these thresholds appear numerically low, in a low-prevalence setting they represent meaningful absolute risk, as the calibrated score aligns with the true incidence rate. Under a strict validation/test split, Risk Function achieves an AUROC of 0.737 (95% confidence interval: 0.694–0.773) and a Brier score of 0.054, indicating effective discrimination and calibration. Next, we evaluate the held-out data: Table 2 reports the full held-out set $( N = 4 { , } 9 8 2 )$ at the default threshold $p \ = \ 0 . 5 ,$ whereas Table 3 reports the test subset $( N = 2 , 4 9 1 )$ with matched calibration. Both yield an AUROC of 0.737, with AUPRCs of 0.158 and 0.146, respectively, reflecting differences in cohort composition.

Table 1: Comparison of Retrieval Strategies.
<table><tr><td>Retrieval Strategy</td><td>Recall@5</td><td>MRR</td><td>Chapter@5</td></tr><tr><td>Vector (BGE-M3)</td><td>0.704</td><td>0.659</td><td>0.918</td></tr><tr><td>Vector → Rerank</td><td>0.697</td><td>0.625</td><td>0.939</td></tr><tr><td>Vector + Rerank (RRF)</td><td>0.745</td><td>0.672</td><td>0.939</td></tr></table>

Table 2: Overall Performance of the Proposed Model at the Default Threshold $( p = 0 . 5 )$ 1
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Accuracy</td><td>0.8212</td></tr><tr><td>Precision / Positive Predictive Value</td><td>0.1623</td></tr><tr><td>Recall / Sensitivity</td><td>0.4733</td></tr><tr><td>F1-score</td><td>0.2417</td></tr><tr><td>Specificity</td><td>0.8434</td></tr><tr><td>Negative Predictive Value</td><td>0.9615</td></tr><tr><td>AUROC</td><td>0.7370</td></tr><tr><td>AUPRC</td><td>0.1582</td></tr></table>

![](images/762959c4a94934e105ded88f879f91f2f2d9d1b3f451eb4acbbd3b8857a0ace3.jpg)  
Figure 6: Reliability diagram of the risk predictor $( n = 4 , 9 8 2 )$ , showing that Platt scaling corrects overconfidence and reduces the mean predicted risk from 0.213 to 0.060, matching the observed one-year T2DM prevalence of 0.060.

Beyond this global calibration, we verify that each deployed triage tier is itself well-calibrated. Table 4, computed on the same held-out split as the reliability diagram in Figure 6, reports the mean predicted probability and the observed one-year T2DM incidence within each risk tier. In all three tiers the predicted probability closely tracks the observed rate (High 0.115 vs. 0.117; Medium 0.048 vs. 0.052; Low 0.024 vs. 0.022), with the highrisk tier concentrating roughly twice the population prevalence (≈ 0.060). This tier-level calibration (not merely global calibration) is what makes the stratification directly actionable for triage: a patient placed in the high-risk bucket truly does carry an elevated, correctly-quantified one-year risk.

Comparison with Tabular Baselines To contextualize the Risk Function against standard tabular predictors, we evaluate Logistic Regression and XGBoost under the same patient-level split, feature window, calibration protocol, and held-out test cohort. The proposed LLM achieves an AU-ROC of 0.737, comparable to XGBoost (0.731) and higher than Logistic Regression (0.697), with overlapping 95% confidence intervals. XGBoost achieves higher AUPRC (0.211 vs. 0.146), while the three models exhibit similar Brier scores after identical Platt calibration (Table 3).

Table 3: Comparison with standard tabular baselines on the strict test set (N = 2,491).
<table><tr><td>Model</td><td>AUROC</td><td>AUPRC</td><td>Brier</td><td>Sens. Spec.</td></tr><tr><td>Logistic Regression</td><td>0.697</td><td>0.130</td><td>0.057</td><td>0.673 0.648</td></tr><tr><td>XGBoost</td><td>0.731</td><td>0.211</td><td>0.053</td><td>0.673 0.706</td></tr><tr><td>★ DIASENTINEL</td><td>0.737</td><td>0.146</td><td></td><td>0.0540.6600.704</td></tr></table>

## 4.3 Guideline Retrieval Performance

To evaluate the Evidence Agent’s retrieval quality, we manually construct a 50-question evaluation set spanning five query types (high-level, boundary, abbreviation, negation, and trend), with 10 questions each. Of these, 49 are answerable from the corpus and 1 is a control question testing the system’s ability to abstain. For each question, we annotate the golden chunk(s) from the corresponding ADA passage and record its chapter and page number (golden\_citation), yielding Cohen’s κ = 0.76. RRF outperforms the dense-retrieval baseline on both Recall@5 and MRR, and achieves a chapterlevel hit rate (Chapter@5) of 0.939.

The annotation schema was first validated on a 15-question pilot before being scaled to the full dataset. During the pilot, severity was reduced from three levels to two because the three-level scheme proved too subjective. We evaluate retrieval performance at both the chunk level using Recall@5 and MRR and the chapter level using Chapter@5. These metrics are used to compare different retrieval strategies (Table 1). The main finding is described in Appendix A.3.

## 4.4 Verification Agent Evaluation

As a demonstration-scale evaluation, we assess the Verification Agent on three aspects: false alarms on clean reports, detection of injected errors, and the stability and discrimination of the LLM-based entailment check. For each of the four deterministic checks, we construct three injected-error cases and three clean control cases, covering distinct error subtypes, magnitudes, and boundary conditions. Across the 24 cases, all 12 injected errors are successfully detected, while none of the 12 clean controls is falsely flagged. This corresponds to a sensitivity of 100% and a specificity of 100%.

Table 4: Per-tier calibration on the held-out set (n = 4,982). In every triage bucket the mean predicted probability closely matches the observed one-year T2DM incidence, confirming that the deployed operating thresholds deliver their promised risk levels (overall prevalence ≈ 0.060).
<table><tr><td>Risk Tier</td><td>n</td><td>Mean Pred.</td><td>Observed</td></tr><tr><td>High (≥ 0.056)</td><td>1619</td><td>0.115</td><td>0.117</td></tr><tr><td>Medium (≥ 0.035)</td><td>1184</td><td>0.048</td><td>0.052</td></tr><tr><td>Low (&lt; 0.035)</td><td>2179</td><td>0.024</td><td>0.022</td></tr></table>

For the LLM-based guideline-entailment check, we evaluate 20 grounded and 20 perturbedunsupported claim–evidence pairs, each anchored to an ADA guideline chunk, achieving 80% sensitivity and 100% specificity (Appendix Table 9). Across three runs at temperature = 0, the judge yields identical verdicts for all 40 cases, indicating reproducible rather than stochastic behavior. Three of the four false negatives are conservative uncertain judgments that our logic maps to pass, while the fourth results from numerical overlap between an unsupported claim and the retrieved evidence, highlighting a limitation of the judge.

These results offer a controlled assessment of the agent’s ability to separate designed errors from clean cases and reject unsupported claims. However, the evaluation is limited to synthetic, manually constructed cases and does not reflect the error distribution or base rate of clinical deployment; it therefore does not establish real-world clinical reliability.

## 5 Conclusion

In this work, we demonstrated DIASENTINEL, an auditable on-premise LLM system for one-year T2DM risk screening and guideline-grounded clinical reporting. The system integrates calibrated risk prediction, evidence-grounded retrieval, and hybrid verification to improve the transparency and reliability of LLM-assisted clinical decision support while keeping all patient data within the hospital environment. Our system illustrates how these components work together through a real-time screening dashboard and interactive patient reports.

## References

American Diabetes Association Professional Practice Committee for Diabetes. 2026. 3. prevention or delay of diabetes and associated comorbidities: Standards of care in diabetes–2026. Diabetes Care, 49(Supplement\_1):S50–S60.

Elham Asgari, Nina Montaña-Brown, Magda Dubois, Saleh Khalil, Jasmine Balloch, Joshua Au Yeung, and Dominic Pimenta. 2025. A framework to assess clinical safety and hallucination rates of llms for medical text summarisation. NPJ digital medicine, 8(1):274.

Christoph Auer, Maksym Lysak, Ahmed Nassar, Michele Dolfi, Nikolaos Livathinos, Panos Vagenas, Cesar Berrospi Ramis, Matteo Omenetti, Fabian Lindlbauer, Kasper Dinkla, and 1 others. 2024. Docling technical report. arXiv preprint arXiv:2408.09869.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2024. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216.

Gordon V Cormack, Charles LA Clarke, and Stefan Buettcher. 2009. Reciprocal rank fusion outperforms condorcet and individual rank learning methods. In Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval, pages 758–759.

Diabetes Prevention Program Research Group. 2002. Reduction in the incidence of type 2 diabetes with lifestyle intervention or metformin. New England journal ofmedicine, 346(6):393–403.

Jun-En Ding, Phan Nguyen Minh Thao, Wen-Chih Peng, Jian-Zhe Wang, Chun-Cheng Chug, Min-Chen Hsieh, Yun-Chien Tseng, Ling Chen, Dongsheng Luo, Chenwei Wu, and 1 others. 2024. Large language multimodal models for new-onset type 2 diabetes prediction using five-year cohort electronic health records. Scientific reports, 14(1):20774.

Zhihua Duan and Jialin Wang. 2024. Exploration of llm multi-agent application implementation based on langgraph+ crewai. arXiv preprint arXiv:2411.18241.

Maxime Griot, Jean Vanderdonckt, and Demet Yuksel. 2025. Implementation of large language models in electronic health records. PLOS Digital Health, 4(12):e0001141.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. 2017. On calibration of modern neural networks. In International conference on machine learning, pages 1321–1330. PMLR.

Kai He, Qika Lin, Hao Fei, Eng Siong Chng, Dehan Hong, Marcus Eng Hock Ong, and Mengling Feng. 2025. Intriage: Intelligent telephone triage in prehospital emergency care. In Proceedings ofthe 2025

Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 873–885.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

International Diabetes Federation. 2025. IDF diabetes atlas. Technical report, International Diabetes Federation, Brussels, Belgium.

Yubin Kim, Hyewon Jeong, Shan Chen, Shuyue Stella Li, Chanwoo Park, Mingyu Lu, Kumail Alhamoud, Jimin Mun, Cristina Grau, Minseok Jung, and 1 others. 2025. Medical hallucinations in foundation models and their impact on healthcare. arXiv preprint arXiv:2503.05777.

John Platt. 1999. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In Advances in Large Margin Classifiers, volume 10, pages 61–74. MIT Press.

Tami Sorgente, Eduardo B Fernandez, and MM Larrondo Petrie. 2005. The soap pattern for medical charts. In Proceedings ofPLoP, volume 2005.

Kevin Wu, Eric Wu, Kevin Wei, Angela Zhang, Allison Casasola, Teresa Nguyen, Sith Riantawan, Patricia Shi, Daniel Ho, and James Zou. 2025. An automated framework for assessing how well llms cite relevant medical references. Nature Communications, 16(1):3615.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, and 1 others. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

## A Appendix

## A.1 Model Development and Calibration

Figure 7 illustrates the dataset construction pipeline and the train/test split used for training and evaluating the Risk Function. The Risk Function is finetuned and calibrated on de-identified real-world patient EHR (private data) from a single hospital, attaining an AUROC of 0.737 (95% CI [0.694, 0.773]) and a Brier score of 0.054 under a strict validation/test split, indicating good discrimination together with good probability calibration. Before calibration, the raw probabilities from the model’s 2-token softmax are markedly overconfident: the mean predicted probability on the test set is about 0.213, whereas the true one-year T2DM incidence is only about 0.060. After Platt scaling, the predicted-probability distribution aligns much more closely with the observed prevalence (Figure 6), making the high/medium/low stratifi cation better suited to clinical interpretation and use. This calibration step is especially important for DIASENTINEL, because its downstream stages (dashboard risk stratification and report generation) all depend directly on this risk probability. Without proper calibration, the risk values could lead to over-alerting or mis-triage, degrading the quality of clinical decisions. Separately from this probability calibration, Table 5 compares the classifier’s default and optimized decision thresholds on the raw (pre-calibration) score, with the corresponding confusion matrix in Table 6. These are raw-score thresholds, not the calibrated screening thresholds (0.035/0.056) in §4.2.

Table 5: Performance Comparison Between Default and Optimized Thresholds
<table><tr><td>Metric</td><td>Threshold = 0.5000</td><td>Threshold = 0.5622</td></tr><tr><td>Accuracy</td><td>0.8212</td><td>0.8418</td></tr><tr><td>Precision / PPV</td><td>0.1623</td><td>0.1747</td></tr><tr><td>Recall / Sensitivity</td><td>0.4733</td><td>0.4367</td></tr><tr><td>F1-score</td><td>0.2417</td><td>0.2495</td></tr><tr><td>Specificity</td><td>0.8434</td><td>0.8678</td></tr></table>

Table 6: Confusion Matrix After Threshold Optimization
<table><tr><td>Actual Class</td><td>Predicted 0</td><td>Predicted 1</td></tr><tr><td>Control (0)</td><td>4063</td><td>619</td></tr><tr><td>Case (1)</td><td>169</td><td>131</td></tr></table>

Table 7: Clinical Interpretation of the Proposed LLM-Based Prediction Model
<table><tr><td>Aspect</td><td>Interpretation</td></tr><tr><td>Discrimination ability</td><td>The model achieved an AUROC of 0.7370, indicating moderate discrimina- tive ability.</td></tr><tr><td>Risk enrichment</td><td>The AUPRC of 0.1582 was higher than the positive class prevalence of 0.0602, suggesting enrichment of high-</td></tr><tr><td>Sensitivity</td><td>risk cases. The model detected 47.33% of true inci- dent diabetes cases at the default thresh- old.</td></tr><tr><td>Positive predic- tion</td><td>The PPV was 0.1623, indicating that most positive predictions were false pos- itives.</td></tr><tr><td>Negative predic- tion</td><td>The NPV was 0.9615, suggesting rel- atively reliable negative predictions, partly due to class imbalance.</td></tr><tr><td>Clinical role</td><td>The model is more suitable for risk strat- ification and preliminary screening than for standalone diagnosis.</td></tr></table>

![](images/10440478010798abec62a7d1e54b003d0790500b6d9b129bb614c32a26706103.jpg)  
Figure 7: Dataset construction pipeline for model training and evaluation.

## A.2 Model Evaluation Results

The clinical interpretation of the model performance metrics is summarized in Table 7 (full heldout set, $N = 4 { , } 9 8 2$ , at $p = 0 . 5$ , matching Table 2).

Table 8: Performance of the deterministic verification checks on clean and error-injected synthetic cases
<table><tr><td>Check</td><td>Pos.</td><td>Neg.</td><td>Sens.</td><td>Spec.</td></tr><tr><td>Risk summary</td><td>3</td><td>3</td><td>100%</td><td>100%</td></tr><tr><td>Numerical agreement</td><td>3</td><td>3</td><td>100%</td><td>100%</td></tr><tr><td>Trend agreement</td><td>3</td><td>3</td><td>100%</td><td>100%</td></tr><tr><td>Citation source</td><td>3</td><td>3</td><td>100%</td><td>100%</td></tr><tr><td>Total</td><td>12</td><td>12</td><td>100%</td><td>100%</td></tr></table>

Table 9: Performance of the guideline-entailment judge on synthetic claim–evidence pairs
<table><tr><td>Evaluation</td><td>Pos. Neg. TP FP FN TN Sens. / Spec.</td></tr><tr><td>Guideline entailment 20</td><td>20 16 0 4 20 80% / 100%</td></tr></table>

## A.3 Key finding

On this clinical-guideline corpus, a cross-encoder reranker used alone degrades ranking quality: its MRR is only 0.625, a drop of 0.034 below the baseline. We observe that the reranker tends to favor longer, narrative passages, whereas the content that actually contains the answer is often a shorter recommendation sentence or a table-like chunk, producing a verbosity bias. Per-question analysis shows only 2 questions are missed, neither solvable by the retrieval algorithm itself: one from a semantic gap in the corpus linking obstructive sleep apnea (OSA) to diabetes, the other from postchunking/OCR information loss leaving only an abbreviated fragment. The main bottleneck thus lies in corpus coverage and chunk quality, not the retriever, making RRF without added LLM-based reranking a reasonable design choice.

## B Ethics and Limitations

Decision Support, Not Replacement DIASEN-TINEL is positioned to assist, not replace, clinical decision-making. Accordingly, every report is required to carry a “for clinical reference only” disclaimer that clearly delimits the system’s role. The verification layer is limited to flagging and recording potential issues; it neither regenerates content nor blocks report output, and the final clinical judgment and treatment decisions remain the responsibility of trained medical professionals. As observed in prior work on emergency-care decision support (He et al., 2025), automated tools can improve efficiency and convenience but may also lead users toward over-reliance, weakening their independent judgment. To mitigate this risk, we deliberately keep human decision-making within the system’s flow and avoid having the system directly provide drug prescriptions or specific dosage recommendations. Concretely, the interface lets physicians accept or override each verification result (§3.3) as a practical human-in-the-loop mechanism, ensuring that final decision authority always rests with human professional judgment.

Fairness and subgroup robustness are not yet fully evaluated We report tier-level calibration (Table 4), confirming that each triage bucket is wellcalibrated; however, demographic subgroup calibration—e.g., whether calibration and error rates are consistent across sexes—remains future work. This is limited by the demographic variables available in the data (sex is the only demographic feature the model uses; no age or ethnicity), and we have likewise not yet evaluated robustness across clinical departments.

## License

The DIASENTINEL production system and its source code are not released under a public software license. The publicly accessible demonstration uses only synthetic patient data and is provided solely for demonstrating the system’s functionality.

## Funding

This work was supported by the Far Eastern Memorial Hospital Innovation Research Project (Grant No. FEMH-2023-007).