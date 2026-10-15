# Evidence-Anchored Overdose Death Prediction from Death Investigation Reports

Raavi Gupta<sup>1</sup>, Ziqing Yang<sup>1</sup>, Karli R. Hochstatter<sup>2</sup>, Talia Nadel<sup>2</sup>,

Nicole D’Anna<sup>3</sup>, Jason K. Graham<sup>3</sup>, Hannah Johnson<sup>4</sup>, Kelsey Opozda<sup>3</sup>,

Nabila El-Bassel<sup>1</sup>, Anubhav Jangra<sup>1</sup>, Smaranda Muresan<sup>5</sup>

<sup>1</sup>Columbia University, USA, <sup>2</sup>Friends Research Institute, USA

<sup>3</sup>The New York City Office of Chief Medical Examiner, USA

<sup>4</sup>Independent Researcher, USA, <sup>5</sup>Barnard College, Columbia University, USA

## Abstract

Confirmation of overdose deaths in the United States is often delayed due to the complexities of certifying drug-involved deaths. Death investigation reports represent a rich but underutilized data source that could be used to predict whether a death is a suspected overdose (i.e., a death preliminarily identified as an overdose prior to the final cause and manner of death determination). We present an automated framework that classifies suspected overdoses from death investigation reports, while producing supporting explanations guided by expert knowledge. Through an expert-guided design process, we investigate 1) what type of explanations are best for model prediction performance, and 2) which architecture is better suited for this task: pipeline models that rationalize first and then predict, or self-rationalizing models that jointly predict and rationalize? Our experiments show that a pipeline model with explanations grounded in the input death investigation reports outperforms the other settings. Our best model also outperforms a rule-based system currently in deployment by 15%, while allowing expert interventions for final decisions.

## 1 Introduction

The age-adjusted rate of overdose death in the US jumped from 8.9 per 100,000 population in 2003 to 31.3 in 2023 (Garnett and Miniño, 2024). Lag time for drug overdose deaths is generally longer than that for all other causes of death by more than six months (Ahmad et al., 2026), and the resulting delays prevent rapid surveillance and obstruct timely public health interventions (Gelberg et al., 2024). An underutilized source of early information lies in the narrative reports produced by onscene death investigators. These reports are typically available shortly after a fatality and contain rich information about physical findings, substance involvement, and witness observations (Davis et al., 2020; O’Donnell et al., 2022).

Existing early-detection systems depend on manual investigator input, ranging from a single suspected-overdose flag (Hochstatter et al., 2024) to pre-defined features extracted from the reports (Hochstatter et al., 2022; Vannice et al., 2022), making these systems time-intensive and dependent on a small, highly trained workforce. Feature-based ML pipelines (Goodman-Meza et al., 2022) and traditional NLP (Ward et al., 2019) have been used to automate parts of this process, but these methods operate over death certificates, which may not be available until six months after the death (Spencer and Ahmad, 2017).

While Large language models (LLM) have revolutionized NLP applications in various domains, their risks of hallucinations (Turpin et al., 2023; Huang et al., 2025), non-deterministic output (He and Lab, 2025), and black box nature requires care in applications in high-stake domains such as overdose prediction. To overcome these limitations, we adopt an expert-guided design process (Cho et al., 2026), directly collaborating with domain experts to explore the use of open-source LLMs for predicting overdose death directly from death investigation reports, while producing supporting explanations guided by expert knowledge.

We investigate two questions: 1) what type of explanations are best for model prediction performance and 2) are pipeline models that rationalize first and then predict or self-rationalizing models that jointly predict and rationalize (Wiegreffe et al., 2021) better for this task. To answer the first question, we explore two types of explanations: 1) feature verbalization, which are natural language descriptions of the expert-provided features associated with a death investigation report, and 2) feature-guided rationales, which are explicit text spans extracted from the death investigation reports that correspond to an expert-provided feature. Figure 1 shows an expert-written mock death investigation report and the corresponding rationales obtained by the overdose prediction system: R represents the feature verbalization for the feature "Injection Equipment on Scene", while $R _ { G }$ shows the exact text span from the death investigation report corresponding to this expert-provided feature (i.e., feature-guided rationale).

![](images/f11d8d79d23394f8958a3b7be97e7298d7af9469ba7eb66fc78ef237befb2bfc.jpg)  
Figure 1: The two explanation types on an example death investigation report of mock data (completely deidentified). Feature-verbalizations R (top) are predefined statements per expert-identified feature; featureguided rationales $R _ { G }$ (highlighted) are verbatim sentences extracted from the report corresponding to the given expert-identified feature.

Our results show that a pipeline model with feature-guided rationales using open-source models<sup>1</sup> is the best setup for this tasks. Moreover, our best model achieves a 15% improvement in overdose prediction over an adopted expert rule-based system that use death investigation reports for overdose predictions (Hochstatter et al., 2022). The pipeline model also allows experts to refine explanations during inference rather than starting from scratch, which could lead to substantial time savings per report. We make three key contributions in this work:

• To the best of our knowledge, we propose the first end-to-end system for automated decision support in overdose prediction from death investigation reports, eliminating the need for manual data extraction and enabling the system to scale.

• We present an interpretable approach, supporting an expert-guided design in which feedback drives model refinement during both training and inference.

• We conduct a systematic comparison of explanation types (feature verbalization and featureguided rationales) and model types (pipeline and self-rationalizing models) for overdose prediction.

## 2 Related Work

## 2.1 Explanation generation

Early work on model interpretability focused on post-hoc explanation methods such as attention visualization (Bahdanau et al., 2016), saliency maps, and LIME-style perturbation approaches (Ribeiro et al., 2016). However, these methods do not provide human-readable rationales that domain experts can provide feedback upon. Lei et al. (2016) pioneered the extraction of human-readable rationales from the input that are sufficient to support the model’s predictions.

The advent of LLMs significantly improved rationale generation quality. Wei et al. (2022) demonstrated that chain of thought prompting that produces intermediate reasoning steps before answering improves performance on tasks. Sun et al. (2022) found that incorporating rationales to even a small amount of training data can considerably improve model performance. Subsequent work, however, examined whether such rationales reflect the model’s true reasoning or merely serve as post-hoc justifications (Turpin et al., 2023), motivating approaches that grounds rationales in input evidence rather than generating them freely. Wiegreffe et al. (2021) formalized this concern and proposed association tests to assess whether a model’s rationale and prediction are genuinely linked. They further unveiled that free-form rationales are not faithful by construction. Extractive rationale, on the other hand, constrains explanations to verbatim spans of the source document (Wei Jie et al., 2024) and makes the reasoning auditable, unlike free-form rationales.

![](images/08ed57e4ac44d5bd6b79cea76e22e2c4d79f9a70889c2dc50f085a31865ca584.jpg)  
Figure 2: Overview of the proposed method. We compare two types of explanations: feature-guided rationales $R _ { G }$ (sentences extracted from the report) and feature-verbalizations R (predefined statements for each present feature). A classifier model then predicts the overdose label O from the explanation, together with the report. We show here the decoupled regime, in which explanation generation and prediction are two separate steps rather than a single joint generation.

Expert feedback has emerged as a mechanism for improving model-generated rationales. Saakyan and Muresan (2024) introduced an in-context learning framework that integrates scarce human annotations into model training to generate high-quality explainable outputs, illustrating how expert feedback loops can enhance the accuracy of the model. Hase et al. (2024) demonstrated that LLMs can generalize from easy annotated data to harder instances via lightweight fine-tuning, suggesting that carefully curated supervision signals can substitute for large-scale human annotation.

## 2.2 Medical NLP and Explanation-Driven Decision Support

Medical NLP has a history of extracting structured signals from free text, spanning rule-based systems (Friedman et al., 2004) and, more recently, transformer-based models such as BioBERT (Lee et al., 2020) and ClinicalBERT (Huang et al., 2020), fine-tuned specifically on biomedical corpora. McInerney et al. (2024) proposed an interpretable risk prediction model that uses LLMs to surface evidence passages in electronic health records supporting diagnostic hypotheses, demonstrating that evidence-aligned predictions reduce diagnostic delays.

Early approaches to overdose classification utilize traditional ML (Goodman-Meza et al., 2022; Tang et al., 2023) techniques on death certificates.

More recent work has demonstrated the use of LLMs for structured information extraction from clinical free text with high precision (Wiest et al., 2024; Funnell et al., 2026).

Moreover, prior work has focused primarily on analysis from death certificates, which may not be available for more than six months. Our work operate on death investigation reports, which are available within days of death rather than months enabling rapid identification of overdose deaths. Hochstatter et al. (2022) and Harruff et al. (2023) also use death investigation reports to identify suspected overdose deaths. However, their rule-based systems require extensive expert annotation. Other systems such as Hochstatter et al. (2024) output only an expert provided global flag based on the totality of evidence, discarding the criterion-level evidence that our automated and intervenable framework preserves.

## 3 Methods and Experimental Setting

Let I denote the input death investigation report, $\mathbf { O } \in \{ 0 , 1 \}$ the suspected overdose prediction, and $V = \{ v _ { 1 } , \ldots , v _ { | V | } \}$ the set of expert-defined features. We also define the explanations R that justifies the prediction O. The explanations are humanreadable and each explanation map back to a feature $v _ { i }$

We define two forms of explanations - featureverbalizations (R): natural language template mapped to the expert-provided features $v _ { i }$ and feature-guided rationales $( R _ { G } ) _ { \mathrm { \Omega } }$ : extracted evidence spans corresponding to the expert-defined features $v _ { i } .$ . An example of the two types of explanation is provided in Figure 1. Table 7 in Appendix E shows all the expert-defined features predictive of overdose death that can be associated with a death investigation report (e.g., History of Prior Overdose(s), Benzodiazepine(s), Pills/Capsules on Scene). We use $\mathbf { R } \in \{ R , R _ { G } \}$ to refer to either form when a statement applies to both.

Table 1: Overdose label distribution across the Featureannotated and Weakly-annotated partitions. Yes and No denote whether the death was certified as an overdose.
<table><tr><td>Partition</td><td>Set</td><td>No</td><td>Yes</td><td>Total</td></tr><tr><td rowspan="2">Feature-annotated</td><td>Train</td><td>786</td><td>3,697</td><td>4,483</td></tr><tr><td>Val Test</td><td>95 96</td><td>457 449</td><td>552 545</td></tr><tr><td rowspan="4">Weakly-annotated</td><td>Train</td><td>1,828</td><td>1,099</td><td>2,927</td></tr><tr><td>Val</td><td>229</td><td>145</td><td>374</td></tr><tr><td>Test</td><td>233</td><td>149</td><td>382</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

## 3.1 Dataset

Our dataset’s structure informs the design of our proposed methodology, including the explanation generation and prediction models. The dataset comprises of 9,263 death investigation reports with two splits:

• Feature-annotated: Reports for which expert-labeled boolean features $\{ v _ { k } \}$ and the final overdose labels O are available.

• Weakly-annotated: Reports for which only the final overdose labels O are available, without feature annotations $\{ v _ { k } \}$

Table 1 presents the distribution of overdose labels across the train, validation, and test sets, created by random sampling from the full pool of reports. Overall, 64.7% of reports are labeled as overdoses; given this class imbalance, we report macro-F1 throughout. Moreover, 39.8% of reports lack expert feature annotations, motivating our use of models to infer them and further obtain downstream explanations R.

## 3.2 Explanation Generation and Prediction Models

Our dataset contains the binary values for each feature only for the feature-annotated partition. To augment the weakly-annotated partition with these missing annotations, we used five-shot question answering with Llama-3.1-8B (Grattafiori et al., 2024) to extract feature labels. The system prompt, per-feature questions, and configuration details are provided in Appendix A. Table 5 provides the per-feature extraction performance for the featureguided rationales. We used this fully augmented feature-labeled dataset to generate the two types of explanations. We denote the completed feature value by $\hat { v } _ { k } \in \{ 0 , 1 \}$ , which corresponds to the annotated value $v _ { k }$ when available and otherwise to the feature value predicted by Llama-3.1-8B.

## 3.2.1 Feature Verbalization

Each variable $v _ { k }$ (e.g., “Drugs/Paraphernalia: Injection Equipment on Scene”) is verbalized to a statement $s _ { k } \ ( \mathrm { e . g . }$ “There is injection equipment reported on-scene.”); the full set of mappings is provided in Table 7. The feature-verbalized explanation is then constructed by enumerating the statements for all variables indicated as present:

$$
R = \{ s _ { k } \ | \ \hat { v } _ { k } = 1 , k = 1 , \ldots , n \}
$$

We obtained the feature indicators from two sources: for the feature-annotated partition, we used the expert-labeled indicators directly, whereas for the weakly-annotated partition, we inferred them using Llama-3.1-8B, as described in Section 3.2. The indicators were then mapped to their corresponding statements $s _ { k }$ and concatenated to form the feature-verbalized explanations.

Prediction Models. We considered two paradigms for producing the final prediction from the constructed explanations. Pipeline $( I  R ;$ $I R  O )$ , where the explanation generation and prediction are separated into two stages. The first model generates the explanation from the input report, and the second model then takes the explanation, alongside the original report $( I )$ , as input to produce the final prediction. Self-rationalizing $( I \to R , O )$ : A single model receives the input report and produces both the explanation and the final prediction in a single pass. We trained two decoder-only LLMs for both pipeline and self-rationalize models $( \mathbf { I } \ \to \ \mathbf { O } , \ \mathbf { I } \ \to \ R , \mathbf { O }$ , and $R \ \to \ \mathbf { O } )$ , namely Llama-3.1-8B Instruct (Grattafiori et al., 2024) and Qwen3-8B (Yang et al., 2025). These models are of comparable size but differ in their pretraining emphasis: Llama-3.1 provides a strong general-purpose baseline, while Qwen3 offers strong performance on instruction-following tasks. We fine-tuned each model with LoRA (Hu et al., 2022) on the constructed explanations and gold output labels.

## 3.2.2 Feature-guided Rationales

Feature-guided rationales are extracted directly from the death investigation report where each rationale is a verbatim sentence-level span<sup>2</sup> that can be traced back to the specific part of the report. We cast this rationale extraction as a multi-label sentence classification task. Let $\mathbf { I } = \{ x _ { 1 } , \dots , x _ { m } \}$ denote the m sentences of a report. For each sentencevariable pair $( x _ { i } , v _ { k } )$ , we predict a binary label $\hat { r } _ { i , k } ^ { G } \in \{ 0 , 1 \}$ indicating whether $x _ { i }$ constitutes evidence for feature $v _ { k }$ . The feature-guided rationale is the set of sentences supporting at least one variable:

$$
R _ { G } = \{ x _ { i } \mid \exists v _ { k } \in V : \hat { r } _ { i , k } ^ { G } = 1 \}
$$

Our dataset contains no gold-labeled evidence spans for either partition. We therefore construct silver-standard labels for $\hat { r } _ { i , k } ^ { G }$ over the full dataset via a teacher–student pipeline. We use an ensemble of teacher LLMs’ labels for each sentence–feature pair independently, and aggregate their outputs by majority vote to obtain the silver labeled $R _ { G } ^ { s i l v e r }$ for each report. The teacher ensemble used to construct silver labels comprises MedGemma-27B (Sellergren et al., 2026), Llama-70B (Grattafiori et al., 2024), and Qwen3- 30B (Yang et al., 2025), inferred in full precision.

These silver labels are then used to fine-tune a student encoder-only model, for sentence-level multi-label classification predicting $\hat { r } _ { i , k } ^ { G }$ for all features $v _ { k } \in \ V$ simultaneously given I. At inference, the fine-tuned student extracts $R _ { G }$ directly from I, without the teacher ensemble. For the student model, we used Clinical-ModernBERT (Lee et al., 2025), a lightweight encoder-only model pretrained on medical corpora.

Prediction Models. Since these explanations are extracted by an encoder-only model, selfrationalization is not applicable for this explanation type, and we therefore consider only the pipeline models. Pipeline $( I \to R _ { G } ; \ [ I ] R _ { G } \to O ) ;$ : the paradigm in which a student model first extracts grounded rationales $R _ { G }$ from I, and a separate classifier then predicts O from $R _ { G }$ alone or from $( I , R _ { G } )$ jointly, ablating the contribution of the original report at the classification stage. We additionally investigate whether rationales alone are sufficient for classification, viz., whether $R _ { G }  O$ can match the performance of a model with full access to the original report. This motivates a direct comparison between $R _ { G } \ \to O$ and $I  O$ , isolating the information contributed by the extracted rationales versus the original narrative.

## 3.2.3 Implementation Details

All decoder-only models are fine-tuned using LoRA (Hu et al., 2022) with rank $ { r } \ = \ 3 2$ and scaling factor $\alpha = 3 2$ , applied to all attention and feed-forward projection layers on single NVIDIA A100-40GB GPU. Models are loaded in 4-bit quantization and fine-tuned with gradient checkpointing enabled. The maximum sequence length is set to 4,096 tokens to accommodate longer reasoning traces. Inference is performed using vLLM (Kwon et al., 2023) for efficient generation. Clinical-ModernBERT is fine-tuned in full precision for the sentence-level multi-label classification task, with a classification head.

## 4 Results

Table 2 reports accuracy and macro-F1 across all model variants and evaluation splits. The rulebased baseline with expert-human annotations, SPOT (Hochstatter et al., 2022), achieves 0.721 Macro-F1. However, in contrast, the best overall configuration of our proposed approach (Qwen3 with $I , R _ { G }  O )$ , achieves 0.831 Macro-F1 without any expert knowledge. This configuration illustrates that grounded rationales together with the original report result in the strongest prediction signal. We examine the results through three views: the explanation type, pipeline versus selfrationalizing models, and the data splits.

Analysis 1: Explanation types Feature-guided rationales outperform feature-verbalized explanations on every split for both models, with the best feature-guided model exceeding the best featureverbalized model by 2.3% macro-F1. Apart from final prediction accuracy, we also compare perfeature extraction quality on the feature-annotated split for the feature-guided rationale type and find that across all the features the model extracted features match the human extracted features around 75% of the cases and there is a discrepancy of at least one feature in the remaining 25%. (More details in Appendix $\mathbf { A } )$ . We also test whether just indicating the presence of feature alone carries the signal required to predict overdose and subsequently trained a logistic regression classifier on the binary variable indicators. The best logistic regression model trained using feature-guided rationales achieved 0.621 macro-F1 which is substantially lower than the corresponding best Qwen3 model by 33.8%. Therefore, knowing which features are present is not sufficient; the rationale text and report are needed to predict overdose outcomes, underscoring the value of feature-guided rationales over feature-verbalized explanations and bare feature presence.

Table 2: Comparison of all variants across test splits. $I  O \colon$ direct prediction from report. $R _ { G } \to O ;$ pipeline model with just feature-guided rationale as input for prediction. I, $R _ { G } \to O ;$ pipeline model with feature-guided rationales and report as input. $I  R , O ;$ : self-rationalized model with feature verbalization as explanations. $I , R \to O { \mathrm { : } }$ pipeline model with report and feature verbalization as explanations as input for prediction. Bold = best configuration per row.
<table><tr><td rowspan="2">Split</td><td rowspan="2">Model</td><td colspan="2"> $I  O$ </td><td colspan="2"> $R _ { G }  O$ </td><td colspan="2"> $I , R _ { G }  O$ </td><td colspan="2"> $I  R , O$ </td><td colspan="2"> $I , R  O$ </td></tr><tr><td> $\mathbf { A c c . }$ </td><td>Macro-F1</td><td> $\mathbf { A c c . }$ </td><td>Macro-F1</td><td> $\mathbf { A c c . }$ </td><td>Macro-F1</td><td> $\mathbf { A c c . }$ </td><td>Macro-F1</td><td> $\mathbf { A c c . }$ </td><td>Macro-F1</td></tr><tr><td rowspan="2">Overall</td><td>Llama-3.1</td><td>0.832</td><td>0.811</td><td>0.798</td><td>0.778</td><td>0.847</td><td>0.829</td><td>0.809</td><td>0.779</td><td>0.832</td><td>0.812</td></tr><tr><td>Qwen3</td><td>0.789</td><td>0.766</td><td>0.786</td><td>0.765</td><td>0.849</td><td>0.831</td><td>0.799</td><td>0.768</td><td>0.794</td><td>0.776</td></tr><tr><td rowspan="2">Feature-annotated</td><td>Llama-3.1</td><td>0.851</td><td>0.674</td><td>0.828</td><td>0.656</td><td>0.866</td><td>0.728</td><td>0.837</td><td>0.619</td><td>0.842</td><td>0.639</td></tr><tr><td>Qwen3</td><td>0.820</td><td>0.637</td><td>0.818</td><td>0.632</td><td>0.873</td><td>0.738</td><td>0.829</td><td>0.597</td><td>0.828</td><td>0.605</td></tr><tr><td rowspan="2">Weakly-annotated</td><td>Llama-3.1</td><td>0.804</td><td>0.791</td><td>0.757</td><td>0.736</td><td>0.819</td><td>0.809</td><td>0.770</td><td>0.762</td><td>0.817</td><td>0.801</td></tr><tr><td>Qwen3</td><td>0.743</td><td>0.725</td><td>0.741</td><td>0.717</td><td>0.814</td><td>0.803</td><td>0.757</td><td>0.747</td><td>0.746</td><td>0.705</td></tr></table>

We compute SPOT’s (Hochstatter et al., 2022) performance on our test split obtaining 0.736 accuracy and 0.721 macro-F1.

Analysis 2: Pipeline vs Self-rationalizing Models. For feature-verbalized explanations, the pipeline regime outperforms the self-rationalizing regime for both Llama-3.1 (0.812 vs. 0.779 macro-F1) and Qwen3 (0.776 vs. 0.768), i.e., conditioning the prediction on the report together with a pre-generated explanation is more effective than generating the explanation and prediction in a single pass. This comparison is only possible for feature-verbalized explanations, since feature-guided rationales use an encoder-only student that cannot generate explanation and prediction jointly. Wiegreffe et al. (2021) report the opposite trend, where self-rationalizing models outperform pipeline models. Their pipeline, however, predicts from the explanation alone, and they show that explanations are often not sufficient without the original input. Our pipeline gives the classifier both the death investigation report and the explanation, which avoids this limitation. Our results for feature-guided rationales also confirm their finding on sufficiency: the rationale alone is not sufficient, as $R _ { G } \to O$ performs significantly worse than both $I  O$ and $I , R _ { G }  O$

Analysis 3: Across different data splits. The macro-F1 for weakly-annotated split is higher than the corresponding feature-annotated split for the best model which is surprising given the training data in the feature-annotated split is human labeled. One of the possible explanations for this can be that almost all samples in the feature-annotated split contain features that are indicative of overdose and hence distinguishing between them is harder than weakly-annotated split despite having humanlabeled data. Another possible reason can be due to the class imbalance in the Feature-annotated dataset.

## 4.1 Teacher Agreement for Silver Data Construction

We analyzed the pairwise Cohen’s κ agreement between the three teacher models used to construct the silver-labeled dataset for feature-guided rationale extraction $R _ { G }$ (see Table 8). Aggregate three-way agreement between all the teacher models is moderate $( \kappa = 0 . 3 6 1 )$ , with MedGemma-LLaMa showing the strongest pairwise agreement $~ ( \kappa ~ = ~ 0 . 5 0 0 )$ and Qwen3-LLaMa the weakest $( \kappa = 0 . 2 8 4 )$

We found that the agreement varies considerably across features, which also correlates with annotation difficulty. Scene evidence variables (e.g., Powder/Residue on Scene $( \kappa = 0 . 4 9 7 )$ and Packaging on Scene) $( \kappa = 0 . 4 3 3 )$ that are easy to annotate tend to have higher agreement as compared to substance use history features, particularly Prescription Medication(s) $( \kappa = 0 . 1 7 1 )$ . This pattern is consistent with the F1 results in Table 3, where the same feature showed the largest gains from expert feedback (see Section 4.2), suggesting that low inter-teacher agreement is a reliable signal of features requiring targeted correction.

Table 3: F1 scores for sentence-level classification without and with expert feedback integration. % Improvement is computed as $\frac { \mathrm { w } / \mathrm { F e e d b a c k - w } / \mathrm { o } \mathrm { F e e d b a c k } } { \mathrm { w } / \mathrm { o } \mathrm { F e e d b a c k } } \times 1 0 0$ . "S.U.H." denotes "Substance Use History" and $" \mathrm { D } / \mathrm { P } "$ denotes "Drugs/Paraphernalia".
<table><tr><td>Variable</td><td>w/Feedback</td><td>w/o Feedback</td><td>% Improv.</td></tr><tr><td>History of Prior Overdose(s)</td><td>0.5882</td><td>0.5195</td><td>+13.2</td></tr><tr><td>S.U.H.: Benzodiazepine(s)</td><td>0.5185</td><td>0.2632</td><td>+97.0</td></tr><tr><td>S.U.H.: Cocaine (including Crack)</td><td>0.8673</td><td>0.8689</td><td>-0.2</td></tr><tr><td>S.U.H.: Injection Drug Use</td><td>0.5116</td><td>0.4835</td><td>+5.8</td></tr><tr><td>S.U.H.: Ketamine</td><td>1.0000</td><td>1.0000</td><td>0.0</td></tr><tr><td>S.U.H.: Opioid(s)</td><td>0.7820</td><td>0.7070</td><td>+10.6</td></tr><tr><td>S.U.H.: PCP</td><td>0.0000</td><td>0.0000</td><td></td></tr><tr><td>S.U.H.: Prescription Medication(s)</td><td>0.3243</td><td>0.0000</td><td></td></tr><tr><td>S.U.H.: Methamphetamine/Other Stimulant(s)</td><td>0.6222</td><td>0.3030</td><td>+105.3</td></tr><tr><td>S.U.H.: Unspecified/Other</td><td>0.7985</td><td>0.7845</td><td>+1.8</td></tr><tr><td>Substance Use Treatment History</td><td>0.7864</td><td>0.7311</td><td>+7.6</td></tr><tr><td>D/P: Injection Equipment on Scene</td><td>0.7869</td><td>0.8176</td><td>-3.8</td></tr><tr><td>D/P: Other Paraphernalia on Scene D/P: Packaging on Scene</td><td>0.6559</td><td>0.4797</td><td>+36.7</td></tr><tr><td></td><td>0.8145</td><td>0.7821</td><td>+4.1</td></tr><tr><td>D/P: Pills/Capsules on Scene</td><td>0.8247</td><td>0.6990</td><td>+18.0</td></tr><tr><td>D/P: Powder/Residue on Scene</td><td>0.8070</td><td>0.7539</td><td>+7.0</td></tr><tr><td>D/P: Smoking Equipment on Scene</td><td>0.7729</td><td>0.7914</td><td>-2.3</td></tr><tr><td>D/P: Straw or Similar on Scene Aggregate</td><td>0.8451 0.6836</td><td>0.8488</td><td>-0.4</td></tr></table>

## 4.2 Expert driven online testing

We fine-tuned ClinicalModernBERT on the silverlabeled dataset for multi-label sentence classification across the features in V as described in 3.2.2, producing the student model for feature-guided rationale extraction $R _ { G }$ . Table 3 reports per-feature F1 scores for extraction performance on the featureannotated split with and without expert feedback integration. The aggregate F1 improves from 0.602 without feedback to 0.684 with feedback, a 13.6% gain.

Expert feedback yields the most substantial improvements for rare or underspecified features (eg. Prescription Medication(s) and Benzodiazepines). These improvements highlight the value of domain expertise in disambiguating forensic terminology. Conversely, well-represented features such as Cocaine and Ketamine remain stable, indicating that the teacher ensemble already provides strong silver supervision for these categories. PCP achieves 0.000 F1 in both conditions, likely due to extreme rarity in the dataset. For physically observable scene evidence such as Injection Equipment and Straw, feedback marginally hurts performance which may be attributed to noise.

We provide the taxonomy of types of major er-

rors found from expert feedback in Appendix F.

## 5 Case Study

To characterize where the pipeline paradigm succeeds and where it does not, we examine three representative cases from the feature-annotated test set (Table 4). Together they separate two different failure modes: errors of aggregation, where evidence is correctly surfaced but incorrectly weighed, and errors of extraction, where the evidence never reaches the classifier at all and hence leads to error in the downstream prediction task as well.

## 6 Discussion

In our experiments, feature-guided rationales consistently outperform feature verbalizations across all data splits, which highlights the value of grounding explanations in exact spans of the input. Although feature-guided rationales alone $( R _ { G }  O )$ do not match the full report $( I \ \to O ;$ ; Table 2), they complement it: when we condition on both I and $R _ { G } ,$ , we obtain our best configuration (0.831 macro-F1). This signal lies in the evidence text rather than in the features it represents, as a logistic regression over the same features reaches only 0.621 macro-F1. In case (a) of Table 4, for instance, the rationale reveals that a flagged substance use history refers only to marijuana, which allows the model to correctly reject the case.

Table 4: Qualitative analysis of three representative cases under the decoupled paradigm. Evidence spans are extracted verbatim and abridged here. Cases (a) and (b) show correct extraction of evidence; case (c) shows failure propagating from the extraction stage.
<table><tr><td>(a) Correct rejection Gold: No</td><td>Pred: No √</td><td>(b) Missed suspected overdose Gold: Yes Pred: No ×</td><td>(c) Extraction error Gold: Yes Pred: No ×</td></tr><tr><td>Evidence</td><td>• person with PMHx of CAD, HTN, DM, COPD, remote h/o drug use, depression and schizophrenia; found down in a shelter. • The only drug use she is aware of is marijuana use in their teens-20s.</td><td>• Drank socially; family denied any known illicit drug use. • A cut red straw and a blue pill (imprint B706) found on a nearby table. • Powder, appearing to be crushed pills, in the clear plastic wrapper of a cigarette box.</td><td>• No illicit drugs were found. • Family stated the decedent was not known to use drugs or alcohol.</td></tr><tr><td> $\hat { v } _ { k }$ </td><td>History of substance use: 1 Drug paraphernalia: 0</td><td>History of substance use: 0 Paraphernalia at scene: 1 Loose/non-prescribed pills: 1</td><td>History of substance use: 0 Paraphernalia at scene: 0 Loose/non-prescribed pills: 0</td></tr><tr><td>Analysis</td><td>The model correctly classifies this case as a non-overdose. A history-of-substance-use flag is present, but the extracted spans reveal that the “remote h/o drug use” refers specifically to marijuana. Since marijuana is excluded from the substance use history in the overdose prediction problem due to low risk, this is a false annotation that is correctly rejected using the additional context from the extracted evidence.</td><td>The model fails to classify this case as a suspected overdose despite extracting substantial scene evidence: drug packaging (a clear plastic wrapper), a loose pill not prescribed to the decedent, a cut straw, and powder/residue. The rationale is accurate, but the classifier does not weigh the accumulated scene indicators strongly enough against the family&#x27;s denial of drug use.</td><td>Here the error originates in the extraction stage: the model extracts misleading spans, which propagate to an incorrect classification. This failure mode is specific to the decoupled paradigm and marks the point where expert review of the rationale prior to prediction would intervene and help in downstream classification.</td></tr></table>

Our pipeline decouples explanation generation from prediction, which allows experts to correct the explanation behind each feature before it reaches the classifier. When we incorporate expert feedback, rationale extraction F1 improves by 13.6% (Table 3). In our study, experts most often remove on-scene evidence that the model reads as a history of use and add synonyms that the model misses. Without these corrections, such errors propagate to the final prediction (Table 4, case c).

Prior work on overdose prediction either treats the model as a black-box labeler (Goodman-Meza et al., 2022; Funnell et al., 2026) or depends on manual annotation of death investigation reports (Hochstatter et al., 2022; Harruff et al., 2023). Our framework requires no manual annotation at inference, allows experts to refine the evidence behind each prediction, and uses only open-weight models that run locally, so sensitive data never leaves the practitioner’s environment.

## 7 Conclusion

The practical bottleneck our system targets is timeliness. Finalized overdose mortality data are delayed by months because confirmation depends on toxicology findings, autopsy reports, and sometimes other investigative information. Whereas, the deployed alternatives that work on death investigation reports for identifying suspected overdose deaths depend on meticulous manual annotation that scales poorly. Our paper presented a study that investigated what types of explanations and what type of explanation-guided models (pipeline or self-rationalizing) are best for this task, showing that a pipeline approach with feature-guided rationales grounded in the death investigation report is the best setup for this task. This setup enables an expert to refine machine-extracted rationales rather than reading each report from scratch, offering an earlier signal at lower human cost and without asking public-health practitioners to trust an opaque model. The correctable explanation can increase the trustworthiness of the system: it keeps a human accountable for each determination while removing the manual-extraction burden that currently limits surveillance throughput. We see this as the central contribution for the public-health community, not just a faster model but a model whose outputs domain experts can review and adjust as needed to ensure validity within their framework.

## Limitations

Our evaluation is conducted on a single jurisdiction, and it remains an open question how well the learned models transfer to coroner or medical examiner offices with different documentation practices. The expert-defined features, while operationally validated, may not capture the full range of overdose-relevant evidence across jurisdictions or evolving drug landscapes. Additionally, the silverlabeled training data for feature-guided rationale extraction is inherently noisy, moderate aggregate teacher agreement $( \kappa = 0 . 3 6 1 )$ means that a fraction of sentence-level labels are incorrect, potentially introducing systematic errors for the lowestagreement variables. Finally, the expert feedback loop we evaluate is retrospective; prospective deployment would require careful interface design to ensure that corrections are both efficient and consistent across reviewers.

## Ethical considerations

Robust protocols were put in place to prevent the release of confidential information. This study was approved by our institutions IRB. All data sharing occurred via a secure SharePoint environment that had been configured to be both Health Insurance Portability and Accountability Act (HIPAA) and Food and Drug Administration (FDA) compliant. Model training and development occurred within a certified cloud-based environment approved for use with Personally Identifiable Information (PII), Personal Health Information (PHI), and HIPAA data. Before the research team had access to the data, narrative death investigation reports were deidentified using the Presidio analyzer to prevent the unnecessary use and release of identifiable data during AI model training and development.

## AI use disclosure

We have used generative AI tools to aid in writing the code for metric generation and generating tables. We have not used AI tools in our core experiment implementation and have also refrained from using them for ideation, literature review, figure generation, or end-to-end paper or section writing. We have reviewed all AI-assisted work, validated the code used for metric generation, cross-checked the illustration artifacts, and ensured that all authors had full agency over the paper-writing process. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## References

F. B. Ahmad, J. A. Cisewski, L. M. Rossen, and P. Sutton. 2026. Provisional drug overdose death counts. Technical report, National Center for Health Statistics. Accessed July 18, 2026.

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2016. Neural machine translation by jointly learning to align and translate.

Dongkyu Cho, Miao Zhang, and Rumi Chunara. 2026. Expert-guided clinical text augmentation via querybased model collaboration.

Gregory G. Davis, Amy B. Cadwallader, Corinne L. Fligner, Thomas P. Gilson, E. Rainn Hall, Kent E. Harshbarger, Robert Kronstrand, Craig T. Mallak, Jerri L. McLemore, Robert A. Middleberg, Owen L. Middleton, Lewis S. Nelson, Agnieszka Rogalska, Emily Tonsfeldt, Jeffrey P. Walterscheid, and Ruth E. Winecker. 2020. Position paper: Recommendations for the investigation, diagnosis, and certification of deaths related to opioid and other drugs. The American Journal of Forensic Medicine and Pathology, 41(3):152–159.

Carol Friedman, Lyudmila Shagina, Yves Lussier, and George Hripcsak. 2004. Automated encoding of clinical documents based on natural language processing. Journal ofthe American Medical Informatics Association, 11(5):392–402. Epub 2004 Jun 7.

Arthur J. Funnell, Panayiotis Petousis, Fabrice Harel-Canada, Ruby Romero, Alex A. T. Bui, Adam Koncsol, Hritika Chaturvedi, Chelsea Shover, and David Goodman-Meza. 2026. Improving drug identification in overdose death surveillance by using clinical natural language processing models. Journal of Forensic Sciences, 71(3):1329–1337.

Matthew F. Garnett and Arialdi M. Miniño. 2024. Drug overdose deaths in the United States, 2003–2023. Technical Report 522, National Center for Health Statistics.

Kitty H. Gelberg, Rachel Chase, Megan Lindstrom, Karli R. Hochstatter, Lauren Larochelle, and Timothy Hunt. 2024. Need for improved timeliness of reporting on drug overdose fatalities: The HEALing communities study. Public Health Reports, 139(1):72– 78.

David Goodman-Meza, Chelsea L. Shover, Jesus A. Medina, Amber B. Tang, Steven Shoptaw, and Alex A. T. Bui. 2022. Development and validation of machine models using Natural Language Processing to classify substances involved in overdose deaths. JAMA Network Open, 5(8):e2225593.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The llama 3 herd of models.

Richard C. Harruff, Nicole A. Yarid, William L. Barbour, and Yang H. Martin. 2023. Medical examiner response to the drug overdose epidemic in king county washington: “real-time” surveillance, data science, and applied forensic epidemiology. Journal ofForensic Sciences, 68(5):1632–1642.

Peter Hase, Mohit Bansal, Peter Clark, and Sarah Wiegreffe. 2024. The unreasonable effectiveness of easy training data for hard tasks. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7002–7024, Bangkok, Thailand. Association for Computational Linguistics.

Horace He and Thinking Machines Lab. 2025. Defeating nondeterminism in llm inference. Thinking Machines Lab: Connectionism. Https://thinkingmachines.ai/blog/defeatingnondeterminism-in-llm-inference/.

K. R. Hochstatter, S. Rastogi, K. Klein, C. Tait-Ozer, N. El-Bassel, and J. Graham. 2022. Predicting accidental drug overdose as the cause of fatality in near real-time using the suspected potential overdose tracker (spot): public health implications. BMC Public Health, 22(1):1311.

Karli R. Hochstatter, Marlon Williams, Shanna Latham, David Fenton, and Andrew L. Falzon. 2024. Rapid identification of suspected drug overdose deaths by death investigators, new jersey, 2020. Public Health Reports, 139(5):549–556.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Kexin Huang, Jaan Altosaar, and Rajesh Ranganath. 2020. Clinicalbert: Modeling clinical notes and predicting hospital readmission.

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. 2025. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems, 43(2):1—-55.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention.

Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. 2020. BioBERT: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics, 36(4):1234–1240.

Simon A. Lee, Anthony Wu, and Jeffrey N. Chiang. 2025. Clinical modernbert: An efficient and long context encoder for biomedical text.

Tao Lei, Regina Barzilay, and Tommi Jaakkola. 2016. Rationalizing neural predictions.

Denis McInerney, William Dickinson, Lucy Flynn, Andrea Young, Geoffrey Young, Jan-Willem van de Meent, and Byron Wallace. 2024. Towards reducing diagnostic errors with interpretable risk prediction. In Proceedings ofthe 2024 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7193–7210, Mexico City, Mexico. Association for Computational Linguistics.

Julie O’Donnell, R. Matt Gladden, Mbabazi Kariisa, and Christine L. Mattson. 2022. Using death scene and toxicology evidence to define involvement of heroin, pharmaceutical morphine, illicitly manufactured fentanyl and pharmaceutical fentanyl in opioid overdose deaths, 38 states and the district of columbia, january 2018–december 2019. Addiction, 117(5):1483– 1490.

Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. 2016. “why should I trust you?”: Explaining the predictions of any classifier. In Proceedings ofthe 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’16, pages 1135–1144, New York, NY, USA. Association for Computing Machinery.

Arkadiy Saakyan and Smaranda Muresan. 2024. ICLEF: In-context learning with expert feedback for explainable style transfer. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 16141– 16163, Bangkok, Thailand. Association for Computational Linguistics.

Andrew Sellergren, Sahar Kazemzadeh, Tiam Jaroensri, Atilla Kiraly, Madeleine Traverse, Timo Kohlberger, Shawn Xu, Fayaz Jamil, Cían Hughes, Charles Lau, et al. 2026. Medgemma technical report.

Merianne R. Spencer and Farida Ahmad. 2017. Timeliness of death certificate data for mortality surveillance and provisional estimates. Technical Report 1, National Center for Health Statistics, Hyattsville, MD.

Jiao Sun, Swabha Swayamdipta, Jonathan May, and Xuezhe Ma. 2022. Investigating the benefits of freeform rationales.

Leigh Anne Tang, Jessica Korona-Bailey, Dimitrios Zaras, Allison Roberts, Sutapa Mukhopadhyay, Stephen Espy, and Colin G. Walsh. 2023. Using natural language processing to predict fatal drug overdose from autopsy narrative text: Algorithm development and validation study. JMIR Public Health and Surveillance, 9:e45246.

Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. 2023. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting.

Kirsten Vannice, Julia Hood, Nicole Yarid, Meagan Kay, Richard Harruff, and Jeff Duchin. 2022. Accuracy of medical examiner’s assessment for near-real-time surveillance of fatal drug overdoses, King County, Washington, March 2017–February 2018. Public Health Reports, 137(3):463–470.

Patrick J. Ward, Peter J. Rock, Svetla Slavova, April M. Young, Terry L. Bunn, and Ramakanth Kavuluru. 2019. Enhancing timeliness of drug overdose mortality surveillance: A machine learning approach. PLoS ONE, 14(10):e0223318.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems.

Yeo Wei Jie, Ranjan Satapathy, and Erik Cambria. 2024. Plausible extractive rationalization through semi-supervised entailment signal. In Findings of the Associationfor Computational Linguistics: ACL 2024, pages 5182–5192, Bangkok, Thailand. Association for Computational Linguistics.

Sarah Wiegreffe, Ana Marasovic, and Noah A. Smith.´ 2021. Measuring association between labels and free-text rationales. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 10266–10284, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Isabella Catharina Wiest, Dyke Ferber, Jiefu Zhu, Marko van Treeck, Sonja K. Meyer, Radhika Juglan, Zunamys I. Carrero, Daniel Paech, Jens Kleesiek, Matthias P. Ebert, Daniel Truhn, and Jakob Nikolas Kather. 2024. Privacy-preserving large language models for structured medical information retrieval. npj Digital Medicine, 7(1):257.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report.

Prompt: Five-shot feature extraction   
System Prompt.   
You are a Medical Investigator that answers   
questions based on provided context. Answer   
the question ‘Yes’ or ‘No’ with direct   
evidence from the context.   
Do not infer specific drugs from generic   
mentions of terms like ‘substance use   
disorder’, ‘illicit substance’ etc. For   
questions about history of substance use,   
answer ‘Yes’ only if the text explicitly   
mentions past use of that substance regardless   
of equipment found on scene. For questions   
about equipment found on scene, answer ‘Yes’   
only if the text explicitly mentions the   
presence of that equipment on the scene   
and not based on inferences. Do not infer   
information.   
User Input.   
Examples:   
{Feature specific examples}   
Context:   
{report}   
Question:   
{feature\_question}   
Output.   
Yes / No  
Figure 3: Prompt used for five-shot feature indicator extraction with Llama-3.1-8B-Instruct.

## A Feature Extraction for the Weakly-annotated Partition

For each feature, we paired the death investigation report with a question asking whether that variable was present, and the model answered Yes or No. The prompt template is shown in Figure 3. The average macro-F1 across the features for the featureannotated split is 0.967. More detailed results can be found in Table 5.

## A.1 Generation Setting

We extract features with Llama-3.1-8B-Instruct using greedy decoding with five in-context examples per feature. Each question is posed separately against each section of the report, and the feature indicator $\hat { v } _ { k }$ is set to present if any section yields an affirmative response.

## A.2 Question for Each Feature

Table 6 lists the question $q _ { k }$ paired with each feature variable $v _ { k }$ in the five-shot extraction prompt described above.

Table 5: Feature extraction accuracy for on Feature-annotated test split. H = human; AI = our system
<table><tr><td>Variable</td><td> $\mathbf { P o s } _ { \mathbf { H } }$ </td><td> $\bf { P o s } _ { \bf { A I } }$ </td><td>TP</td><td>FP</td><td>FN</td><td>TN</td><td>Recall</td><td>κ</td></tr><tr><td>Substance use history</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>History of prior overdose(s)</td><td>45</td><td>41</td><td>41</td><td>0</td><td>4</td><td>500</td><td>0.911</td><td>0.950</td></tr><tr><td>Benzodiazepine(s)</td><td>25</td><td>22</td><td>22</td><td>0</td><td>3</td><td>520</td><td>0.880</td><td>0.933</td></tr><tr><td>Cocaine (including Crack)</td><td>173</td><td>173</td><td>173</td><td>0</td><td>0</td><td>372</td><td>1.000</td><td>1.000</td></tr><tr><td>Injection drug use</td><td>48</td><td>47</td><td>47</td><td>0</td><td>1</td><td>497</td><td>0.979</td><td>0.988</td></tr><tr><td>Ketamine</td><td>2</td><td>2</td><td>2</td><td>0</td><td>0</td><td>543</td><td>1.000</td><td>1.000</td></tr><tr><td>Opioid(s)</td><td>164</td><td>160</td><td>160</td><td>0</td><td>4</td><td>381</td><td>0.976</td><td>0.982</td></tr><tr><td>PCP</td><td>3</td><td>2</td><td>2</td><td>0</td><td>1</td><td>542</td><td>0.667</td><td>0.799</td></tr><tr><td>Other Rx used not as prescribed</td><td>10</td><td>7</td><td>7</td><td>0</td><td>3</td><td>535</td><td>0.700</td><td>0.821</td></tr><tr><td>Methamphetamine / other stimulant</td><td>14</td><td>14</td><td>14</td><td>0</td><td>0</td><td>531</td><td>1.000</td><td>1.000</td></tr><tr><td>Unspecified / other</td><td>367</td><td>338</td><td>338</td><td>0</td><td>29</td><td>178</td><td>0.921</td><td>0.884</td></tr><tr><td>Treatment history</td><td>96</td><td>92</td><td>92</td><td>0</td><td>4</td><td>449</td><td>0.958</td><td>0.974</td></tr><tr><td>Drugs / paraphernalia on scene</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Injection equipment</td><td>53</td><td>52</td><td>52</td><td>0</td><td>1</td><td>492</td><td>0.981</td><td>0.989</td></tr><tr><td>Other paraphernalia</td><td>205</td><td>103</td><td>103</td><td>0</td><td>102</td><td>340</td><td>0.502</td><td>0.558</td></tr><tr><td>Packaging</td><td>120</td><td>120</td><td>120</td><td>0</td><td>0</td><td>425</td><td>1.000</td><td>1.000</td></tr><tr><td>Pills / capsules</td><td>44</td><td>40</td><td>40</td><td>0</td><td>4</td><td>501</td><td>0.909</td><td>0.948</td></tr><tr><td>Powder / residue</td><td>133</td><td>133</td><td>133</td><td>0</td><td>0</td><td>412</td><td>1.000</td><td>1.000</td></tr><tr><td>Smoking equipment</td><td>89</td><td>89</td><td>89</td><td>0</td><td>0</td><td>456</td><td>1.000</td><td>1.000</td></tr><tr><td>Straw or similar</td><td>71</td><td>70</td><td>70</td><td>0</td><td>1</td><td>474</td><td>0.986</td><td>0.992</td></tr></table>

## B Feature-verbalized Statements

Table 7 lists the statements used to construct feature-verbalized explanations R. For each feature $v _ { k }$ predicted as present $( \hat { v } _ { k } = 1 )$ , the corresponding statement $s _ { k }$ is included in the explanation.

## C Teacher Ensemble Prompt

Figure 4 shows the prompt used by each teacher model to extract feature-guided rationales via question answering. Prompts are reproduced verbatim from our experiments.

## C.1 Teacher Agreement

Pairwise agreement among the three teacher models is reported in Table 8.

## C.2 Teacher Accuracy

Teacher accuracy against expert feature labels is reported in Table 9.

## D Comparison with Logistic regression

We also see whether the use of language models is even necessary for the second stage $R _ { G }  O$ and whether simple models like logistic regression are sufficient to make the prediction. We report three logistic regression baselines across three experiments. Experiment 1 trains on human-annotated evidence variables from the feature-annotated split; because these annotations exist only for featureannotated cases, it can be evaluated on the featureannotated test subset alone. Experiment 2 trains on model-extracted evidence variables using the same train/test case IDs as Experiment 1 to directly compare with human annotations, but can be evaluated across all subsets. Experiment 3 trains on model-extracted variables pooled across all splits. All models use balanced class weights. Table 12 reports macro-F1. We find that the performance remains subpar as compared to using language models and hence the context in the evidence and the report is required to actually get the final prediction.

## E Analysis where the evidence extracted is correct but the final prediction is not

For Llama-3.1-8B, we also analyze the cases where the upstream evidence extraction is correct for all the features, but there is still a discrepancy in the prediction. We find that the majority of the cases are correctly labeled.

<table><tr><td rowspan="2"></td><td colspan="2">Predicted</td></tr><tr><td>Yes</td><td>No</td></tr><tr><td rowspan="2">Yes Actual No</td><td>317 (77.7%)</td><td>12 (2.9%)</td></tr><tr><td>39 (9.6%)</td><td>40 (9.8%)</td></tr></table>

## F Expert feedback types

An analysis of types of expert feedback is present in Table 10.

![](images/d3931e12e5fee564962ce7767440fd9ec855b623b63e0420cd23c73707f5a1a6.jpg)  
Figure 4: Prompt used for extracting feature-guided rationales via question answering in teacher models.

## G Additional Model Results

Table 11: Macro-F1 of additional prediction models.
<table><tr><td>Input</td><td>ClinicalModernBERT</td><td>DeepSeek</td></tr><tr><td> $I  O$ </td><td>0.78</td><td>0.82</td></tr><tr><td> $R _ { G }  O$ </td><td>0.71</td><td>0.77</td></tr><tr><td> $I , R _ { G }  O$ </td><td>0.81</td><td>0.83</td></tr></table>

Table 12: Macro-F1 for logistic regression baselines across evidence variable source and training distribution. Exp. 1: Feature-annotated train samples trained with human evidence variables. Exp. 2: Feature-annotated train samples trained with model-extracted variables. Exp. 3: Entire train split trained with model-extracted variables. Dashes indicate evaluations unavailable by construction.
<table><tr><td rowspan="2">Test subset</td><td colspan="3">Macro-F1</td></tr><tr><td>Exp. 1</td><td>Exp. 2</td><td>Exp. 3</td></tr><tr><td>Feature-annotated</td><td>0.552</td><td>0.561</td><td>0.523</td></tr><tr><td>Weakly-annotated</td><td></td><td>0.485</td><td>0.478</td></tr><tr><td>All</td><td>一</td><td>0.545</td><td>0.621</td></tr></table>

Table 6: Questions used to extract each feature variable from the MLI report. Each question is substituted into the {feature\_question} slot of the prompt template.
<table><tr><td>Variable</td><td>Question (qk)</td></tr><tr><td>History of Prior Overdose(s)</td><td>Is there any mention of a prior overdose?</td></tr><tr><td>Benzodiazepine(s)</td><td>Is there any reported substance use history of benzodiazepines?</td></tr><tr><td>Cocaine (including crack)</td><td>Is there any reported substance use history of cocaine (including crack)?</td></tr><tr><td>Injection Drug Use</td><td>Is there any reported substance use history of injection drug use (answer No if only the word paraphernalia is present without any further evidence)?</td></tr><tr><td>Ketamine</td><td>Is there any reported substance use history of ketamine?</td></tr><tr><td>Opioid(s)</td><td>Is there any reported substance use history of opioid(s)?</td></tr><tr><td>PCP</td><td>Is there any reported substance use history of phencyclidine (PCP)?</td></tr><tr><td>Prescription Medication(s)</td><td>Is there any reported substance use history of prescription medications, excluding</td></tr><tr><td>Methamphetamine/Other</td><td>specified opioids, benzodiazepines, or stimulants? Is there any reported substance use history of methamphetamine?</td></tr><tr><td>Stimulant(s) Unspecified/Other</td><td>Is there any reported substance use history of substance(s) other than benzodiazepines, cocaine, injections, ketamine, opioids, PCP, methamphetamine (answer No if only the</td></tr><tr><td>Substance Use Treatment</td><td>words illicit substance&#x27; etc. are present without naming the substance)? Is there any record of past substance use treatment by a professional?</td></tr><tr><td>History Injection Equipment on Scene</td><td>Is there any indicator of injection equipment reported on-scene, including syringes, needle caps, cookers, spoons, tourniquets (answer No if only the word paraphernalia is</td></tr><tr><td>Other Paraphernalia on Scene</td><td>present without any further evidence)? Is there any indicator of evidence of paraphernalia (excluding syringes, needle caps, cookers, spoons, tourniquets) reported on-scene, such as scales, fentanyl test strips</td></tr><tr><td>Packaging on Scene</td><td>(answer No if only the word paraphernalia is present without any further evidence)? Was any drug packaging, such as glassines, baggies, or flip-top vials, reported on the</td></tr><tr><td>Pills/Capsules on Scene</td><td>scene, regardless of whether drugs were present? Is there any indicator of loose pills/capsules reported on-scene, not prescribed to the</td></tr><tr><td>Powder/Residue on Scene</td><td>decedent? Is there any indicator of powder or residue reported on-scene (solid capsules/pills are not</td></tr><tr><td>Smoking Equipment on Scene</td><td>considered as powder/residue)? Is there any indicator of smoking equipment reported on-scene, including glass pipes,</td></tr><tr><td>Straw or Similar on Scene</td><td>filters, thin rods, lighters (not alone)? Is there any indicator of straw or similar equipment reported on-scene, including cut</td></tr></table>

Table 7: Predefined natural-language statements mapped to each of the feature variables, used to construct the feature-verbalized explanations R.
<table><tr><td>Feature</td><td>Statement (sk)</td></tr><tr><td>History of Prior Overdose(s)</td><td>There is mention of a prior overdose.</td></tr><tr><td>Benzodiazepine(s)</td><td>There is reported substance use history of benzodiazepines.</td></tr><tr><td>Cocaine (including Crack)</td><td>There is reported substance use history of cocaine.</td></tr><tr><td>Injection Drug Use</td><td>There is reported substance use history of IV injection drugs.</td></tr><tr><td>Ketamine</td><td>There is reported substance use history of ketamine.</td></tr><tr><td>Opioid(s)</td><td>There is reported substance use history of opioid(s).</td></tr><tr><td>PCP</td><td>There is reported substance use history of PCP.</td></tr><tr><td>Prescription Medication(s)</td><td>There is a reported history of prescription medicines, other than opioids, benzodiazepines, or stimulants, not being used as prescribed.</td></tr><tr><td>Methamphetamine/Other Stimulant(s)</td><td>There is reported substance use history of methamphetamine.</td></tr><tr><td>Unspecified/Other Substance Use Treatment</td><td>There is reported substance use history of unspecified substance(s).</td></tr><tr><td>History</td><td>There is substance use treatment history by a professional.</td></tr><tr><td>Injection Equipment on Scene</td><td>There is injection equipment reported on-scene.</td></tr><tr><td>Other Paraphernalia on Scene</td><td>There is paraphernalia (excluding injection equipment) reported on-scene.</td></tr><tr><td>Packaging on Scene</td><td>There is drug packaging such as glassines, baggies, or flip-top vials reported on-scene.</td></tr><tr><td>Pills/Capsules on Scene Powder/Residue on Scene</td><td>There are loose pills/capsules/tablets not prescribed to the decedent reported on-scene.</td></tr><tr><td>Smoking Equipment on Scene</td><td>There is powder/residue reported on-scene.</td></tr><tr><td>Straw or Similar on Scene</td><td>There is smoking equipment such as glass pipes or filters reported on-scene. There are straws or similar equipment like rolled paper or deconstructed pens reported</td></tr></table>

Table 8: Cohen’s κ agreement between teacher models for sentence-level evidence extraction across 18 forensic variables. Values represent pairwise and three-way agreement in the silver dataset construction.
<table><tr><td>Variable</td><td>MedGemma and Qwen3</td><td>MedGemma and Llama</td><td>Qwen3 and Llama</td><td>All Annotators</td></tr><tr><td>History of Prior Overdose(s)</td><td>0.406112</td><td>0.656521</td><td>0.391848</td><td>0.484827</td></tr><tr><td>Substance Use History: Benzodiazepine(s)</td><td>0.251875</td><td>0.423022</td><td>0.265501</td><td>0.313466</td></tr><tr><td>Substance Use History: Cocaine (including Crack)</td><td>0.326150</td><td>0.562726</td><td>0.347551</td><td>0.412142</td></tr><tr><td>Substance Use History: Injection Drug Use</td><td>0.131958</td><td>0.404878</td><td>0.104612</td><td>0.213816</td></tr><tr><td>Substance Use History: Ketamine</td><td>0.297646</td><td>0.620090</td><td>0.324882</td><td>0.414206</td></tr><tr><td>Substance Use History: Opioid(s)</td><td>0.285187</td><td>0.504151</td><td>0.319806</td><td>0.369715</td></tr><tr><td>Substance Use History: PCP</td><td>0.335513</td><td>0.508541</td><td>0.315829</td><td>0.386628</td></tr><tr><td>Substance Use History: Prescription Medication(s)</td><td>0.120104</td><td>0.286659</td><td>0.105637</td><td>0.170800</td></tr><tr><td>Substance Use History: Methamphetamine/Other Stimulant(s)</td><td>0.172003</td><td>0.397143</td><td>0.144348</td><td>0.237832</td></tr><tr><td>Substance Use History: Unspecified/Other</td><td>0.234560</td><td>0.505633</td><td>0.214964</td><td>0.318385</td></tr><tr><td>Substance Use Treatment History</td><td>0.347706</td><td>0.555369</td><td>0.338634</td><td>0.413903</td></tr><tr><td>Drugs/Paraphernalia: Injection Equipment on Scene</td><td>0.397759</td><td>0.513754</td><td>0.311943</td><td>0.407819</td></tr><tr><td>Drugs/Paraphernalia: Other Paraphernalia on Scene Drugs/Paraphernalia: Packaging on Scene</td><td>0.205337</td><td>0.327110</td><td>0.187710</td><td>0.240052</td></tr><tr><td>Drugs/Paraphernalia: Pills/Capsules on Scene</td><td>0.383756</td><td>0.569978</td><td>0.347442</td><td>0.433726</td></tr><tr><td>Drugs/Paraphernalia: Powder/Residue on Scene</td><td>0.230831</td><td>0.492686</td><td>0.225899</td><td>0.316472</td></tr><tr><td>Drugs/Paraphernalia: Smoking Equipment on Scene</td><td>0.424763 0.423799</td><td>0.607269 0.521117</td><td>0.460214</td><td>0.497416</td></tr><tr><td>Drugs/Paraphernalia: Straw or Similar on Scene</td><td>0.396304</td><td>0.543575</td><td>0.326593 0.378239</td><td>0.423836</td></tr><tr><td></td><td></td><td></td><td></td><td>0.439373</td></tr><tr><td>Aggregate</td><td>0.298409</td><td>0.500012</td><td>0.283981</td><td>0.360801</td></tr></table>

Table 9: Per-variable extraction accuracy of three teacher models against human-annotated labels across all Featureannotated cases $( n = 5 { , } 5 8 0 ;$ ; train / val / test = 4,483 / 552 / 545). A variable is predicted positive if the model’s extracted span list is non-empty. $\mathrm { P P V } = 1 . 0 0 0$ for all variables and all models (zero false positives) and is omitted for brevity. Cohen’s κ is computed against human labels as ground truth.
<table><tr><td></td><td></td><td colspan="3">Llama-70B</td><td colspan="3">MedGemma-27B</td><td colspan="3">Qwen3-30B</td></tr><tr><td>Variable</td><td>PoSH</td><td> $\bf { P o s } _ { \bf { A I } }$ </td><td>Recall</td><td>κ</td><td> $\bf { P o s } _ { \bf { A I } }$ </td><td>Recall</td><td>κ</td><td> $\bf { P o s } _ { \bf { A I } }$ </td><td>Recall</td><td>κ</td></tr><tr><td>Substance use history</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Benzodiazepine(s)</td><td>211</td><td>182</td><td>0.863</td><td>0.924</td><td>176</td><td>0.834</td><td>0.906</td><td>120</td><td>0.569</td><td>0.717</td></tr><tr><td>Cocaine (including Crack)</td><td>1609</td><td>1595</td><td>0.991</td><td>0.994</td><td>1564</td><td>0.972</td><td>0.980</td><td>1295</td><td>0.805</td><td>0.854</td></tr><tr><td>Injection drug use</td><td>523</td><td>496</td><td>0.948</td><td>0.971</td><td>396</td><td>0.757</td><td>0.850</td><td>140</td><td>0.268</td><td>0.399</td></tr><tr><td>Ketamine</td><td>27</td><td>27</td><td>1.000</td><td>1.000</td><td>26</td><td>0.963</td><td>0.981</td><td>20</td><td>0.741</td><td>0.850</td></tr><tr><td>Opioid(s)</td><td>1638</td><td>1594</td><td>0.973</td><td>0.981</td><td>1586</td><td>0.968</td><td>0.977</td><td>1252</td><td>0.764</td><td>0.821</td></tr><tr><td>PCP</td><td>53</td><td>50</td><td>0.943</td><td>0.971</td><td>47</td><td>0.887</td><td>0.939</td><td>36</td><td>0.679</td><td>0.808</td></tr><tr><td>Prescription medication(s)</td><td>217</td><td>145</td><td>0.668</td><td>0.795</td><td>94</td><td>0.433</td><td>0.595</td><td>25</td><td>0.115</td><td>0.200</td></tr><tr><td>Methamphetamine / other stimulant</td><td>141</td><td>135</td><td>0.957</td><td>0.978</td><td>130</td><td>0.922</td><td>0.958</td><td>100</td><td>0.709</td><td>0.826</td></tr><tr><td>Unspecified / other</td><td>3715</td><td>3107</td><td>0.836</td><td>0.774</td><td>3333</td><td>0.897</td><td>0.854</td><td>1664</td><td>0.448</td><td>0.352</td></tr><tr><td>Treatment history</td><td>834</td><td>783</td><td>0.939</td><td>0.963</td><td>792</td><td>0.950</td><td>0.970</td><td>544</td><td>0.652</td><td>0.761</td></tr><tr><td>History of prior overdose(s)</td><td>502</td><td>443</td><td>0.882</td><td>0.932</td><td>426</td><td>0.849</td><td>0.911</td><td>329</td><td>0.655</td><td>0.776</td></tr><tr><td>Drugs / paraphernalia on scene</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Injection equipment</td><td>559</td><td>556</td><td>0.995</td><td>0.997</td><td>551</td><td>0.986</td><td>0.992</td><td>491</td><td>0.878</td><td>0.929</td></tr><tr><td>Other paraphernalia*</td><td>2229</td><td>858</td><td>0.385</td><td>0.429</td><td>774</td><td>0.347</td><td>0.390</td><td>260</td><td>0.117</td><td>0.137</td></tr><tr><td>Packaging</td><td>1120</td><td>1113</td><td>0.994</td><td>0.996</td><td>1117</td><td>0.997</td><td>0.998</td><td>1017</td><td>0.908</td><td>0.940</td></tr><tr><td>Pills / capsules</td><td>422</td><td>363</td><td>0.860</td><td>0.919</td><td>395</td><td>0.936</td><td>0.964</td><td>137</td><td>0.325</td><td>0.471</td></tr><tr><td>Powder / residue</td><td>1239</td><td>1225</td><td>0.989</td><td>0.993</td><td>1227</td><td>0.990</td><td>0.994</td><td>1064</td><td>0.859</td><td>0.904</td></tr><tr><td>Smoking equipment</td><td>995</td><td>993</td><td>0.998</td><td>0.999</td><td>993</td><td>0.998</td><td>0.999</td><td>922</td><td>0.927</td><td>0.954</td></tr><tr><td>Straw or similar</td><td>822</td><td>815</td><td>0.991</td><td>0.995</td><td>817</td><td>0.994</td><td>0.996</td><td>774</td><td>0.942</td><td>0.965</td></tr><tr><td>Overall‡</td><td></td><td></td><td>0.976</td><td>0.910</td><td></td><td>0.976</td><td>0.909</td><td></td><td>0.934</td><td>0.718</td></tr></table>

Table 10: Types of expert feedback on feature-guided rationales $R _ { G }$ extracted by the student model. Numbers in parentheses give the count of edited samples per type.
<table><tr><td>Type</td><td>Error type</td><td>Description</td></tr><tr><td>Over-extraction (111)</td><td>On-scene evidence as history Prescription as misuse Generic mention as a specific drug Neighboring scene category Indirect clinical sign</td><td>Scene items cited as a history of use Prescribed medications or medication lists cited as misuse Vague mentions of drug use mapped to a specific substance Scene item assigned to an adjacent scene feature</td></tr><tr><td>Missed evidence (70)</td><td>stance Synonyms and street names Drug inside a listed history Signs of treatment</td><td>Substance named by a synonym or street name Substance named within a list in the medical history Treatment programs or support meetings</td></tr><tr><td>Boundary (23)</td><td>Fragment merging</td><td>Fragmented spans merged into full sentences</td></tr></table>