# A Source-Grounded Framework for Constructing and Evaluating Progressive Multimodal Diagnostic Dialogues from Clinical Case Reports

Yufan Wang<sup>1,†</sup>, Rui Yang<sup>1,†</sup>, Yi Liu<sup>2</sup>, Yi Lin<sup>1</sup>, Yifan Peng<sup>1,\*</sup>

<sup>1</sup>Population Health Sciences, Weill Cornell Medicine, New York City, USA

<sup>2</sup>Department of Medicine, Weill Cornell Medicine, New York City, USA

Corresponding author(s). Email(s): yip4002@med.cornell.edu

<sup>†</sup>These authors contributed equally to this work.

## Abstract

Clinical diagnosis requires progressive integration of patient history, physical examination, laboratory findings, medical images, and diagnostic-informative tests. However, most multimodal medical benchmarks evaluate fixed inputs or endpoint answers, while fully interactive diagnostic agents conflate evidence selection with evidence interpretation. We present a sourcegrounded framework to construct progressive multimodal diagnostic dialogues from case reports and an evaluation strategy for assessing MLLMs on final diagnosis, diagnostic reasoning, and image-finding interpretation. Evaluation on 24 internal medicine case reports showed that our framework can accurately convert case reports into reference dialogues, achieving a diagnosis F1 of 0.99 and a reasoning-quality score of 4.79 out of 5. Evaluation on two frontier MLLMs (o4-mini and Claude Haiku 4.5) achieved reasoning-quality scores of 2.75 and 2.50, respectively, with substantially lower diagnosis, reasoning, and image-finding F1 scores. The results demonstrate that fluent responses do not necessarily reflect evidence-grounded clinical reasoning and highlight the utility of the proposed framework for evaluating multimodal diagnostic reasoning.

Keywords: multimodal large language models; clinical reasoning; progressive diagnostic dialogue

## 1. Introduction

Multimodal large language models (MLLMs) have expanded medical artificial intelligence beyond image classification toward visual question answering, open-ended image interpretation, and diagnostic synthesis. In clinical diagnosis, however, information is rarely available as a single fixed input [1–4]. Clinicians progressively integrate patient history, physical examination, routine laboratory findings, medical images, and diagnosis-informative tests as a case unfolds. Evaluating whether MLLMs can perform this progressive integration is therefore important for assessing their potential role in diagnostic decision support.

Recent work has begun to evaluate medical models through diagnostic dialogue and interactive consul tation [5–8]. However, fully interactive diagnostic agents must simultaneously decide which question or test to request, interpret the evidence returned, and integrate it into an evolving diferential diagnosis. Joint evaluation of these capabilities makes it dificult to attribute diagnostic failures to inappropriate test selection, misinterpretation of evidence, or inadequate reasoning; a controlled setting is therefore needed to isolate evidence interpretation and integration before fully assessing interactive diagnostic agents.

Published case reports provide a useful substrate for such controlled evaluation because they connect patient history, physical examination, laboratory testing, medical figures, diagnostic reasoning, and final diagnosis at the patient-case level. PubMed Central (PMC) further provides machine-readable full-text articles with associated media files [9]. However, case reports also contain information that may directly reveal the answer. Direct conversion of raw articles risks information leakage from figure captions, later article sections, and confirmatory diagnostic results.

Additionally, evaluating MLLM performance in diagnostic dialogues requires more than final-diagnosis accuracy. A model may reach the correct diagnosis while omitting case-defining evidence, introducing unsupported image findings, contradicting the source case, or using clinically invalid intermediate inferences. Conversely, exact matching of complete responses is too brittle because medically equivalent statements may difer in wording, specificity, and granularity. These limitations highlight the need for an evaluation approach that assesses not only whether an MLLM reaches the correct diagnosis, but also whether its reasoning is complete, clinically valid, and grounded in the available multimodal evidence.

To bridge these gaps, we present a source-grounded framework for constructing progressive multimodal diagnostic dialogues from case reports, and an evaluation strategy for assessing MLLMs on final diagnosis and diagnostic reasoning. Article-derived references are extracted into computer-interpretable case representations, aligned with local medical images, and separated from the evidence visible during dialogue construction. Reference dialogues and test-model dialogues are generated under the same progressive evidence-disclosure protocol. During the evaluation of MLLM in these cases, final diagnoses, diagnostic reasoning, and image findings are assessed using atomic-item alignment. In contrast, overall reasoning quality is assessed using case-level factuality, validity, coherence, and utility (FVCU) scoring.

The main contributions of this work are threefold: (i) We introduce a source-grounded framework that converts PMC case reports into computer-interpretable case representations and progressive multimodal diagnostic dialogues. (ii) We define a controlled evidence-disclosure protocol that isolates multimodal evidence interpretation and integration from autonomous evidence-selection behavior. (iii) We provide an evaluation framework that combines atomic item alignment for final diagnosis, diagnostic reasoning, and image findings with case-level FVCU scoring of clinical reasoning quality.

## 2. Related Work

Early medical multimodal benchmarks, including VQA-RAD and SLAKE, evaluate responses to predefined questions about medical images. At the same time, LLaVA-Med and Med-Flamingo support more open-ended generative responses to biomedical visual inputs [1–4]. These resources provide important measurements of multimodal capability but primarily operate under fixed-input settings. More recent systems and benchmarks, including AMIE, multimodal AMIE, MMD-Eval, and 3MDBench, move toward interactive diagnostic consultation and proactive information exchange [5–8]. However, fully interactive settings jointly evaluate evidence selection, interpretation, and diagnostic integration, making it dificult to determine which capability caused a diagnostic failure. Our work instead uses a fixed staged evidencedisclosure protocol to isolate evidence interpretation and integration from autonomous evidence selection.

Open-ended diagnostic dialogue also requires evaluation beyond exact matching or final-answer accuracy. Prior clinical LLM evaluation has used structured judges and checklist-based comparisons between free-form outputs and reference information [10]. Such methods can assess whether important clinical content is retained, but component-level alignment alone does not determine whether the complete di agnostic synthesis is clinically sound. We therefore combine atomic-item alignment for final diagnosis, diagnostic reasoning, and image findings with case-level factuality, validity, coherence, and utility scoring adapted from reasoning-trace evaluation [11]. This combination localizes specific evidence errors while also assessing the overall quality of the diagnostic reasoning.

![](images/944b0c6e0a6437f9e9cfd99cbf1d7aacb635033fa6a1dffb13321c4b6afc9612.jpg)  
Figure 1: Overview of the source-grounded framework for constructing and evaluating progressive multimodal diagnostic dialogues from PMC case reports.

## 3. Methods

## 3.1. Study Design and System Overview

We developed and evaluated a source-grounded workflow for constructing progressive multimodal diagnostic dialogues from medical case reports. As summarized in Figure 1, the workflow comprised three stages: computer-interpretable representation extraction, progressive dialogue generation, and multifaceted evaluation.

Throughout this paper, article-derived references denote structured, source-supported information extracted from the original case report. These references were organized into a computer-interpretable case representation containing clinical case elements, figure-linked evidence, and diagnostic assessment references. Reference dialogues denote dialogues generated from the article-derived references. MLLM-generated dialogues denote dialogues generated by the evaluated MLLMs.

## 3.2. Input Requirements

This framework leverages the JATS-compliant full-text XML format used by PubMed Central (PMC) (https: //jats.nlm.nih.gov/nlm-dtd/publishing/), in which Open Access article packages provide full-text article metadata together with associated media files, including images [12].

A case was eligible if it satisfied all of the following input requirements: (i) A parseable XML article. (ii) At least one retrievable image. (iii) An extractable article-reported final diagnosis. (iv) Suficient article-reported content to support progressive dialogue construction and evaluation, including patient information, timeline, diagnostic tests (e.g., labs, imaging, pathology), clinical findings, and diagnostic reasoning.

For cases containing multiple figures, we preserved the image order defined in the original article. Figure references were first identified in the article source and then matched to the corresponding image files. This ordering preserved the source publication progression of visual evidence and ensured that multi image cases remained aligned with the accompanying textual narrative and diagnostic reasoning.

## 3.3. Computer-interpretable representation extraction

Each case report was processed using a source-grounded LLM extractor to construct a computer-interpretable representation. The extraction schema was informed by clinically relevant items of the CARE case-report checklist, including patient information, clinical findings, timeline, diagnostic assessment, interventions

or outcomes when reported, and diagnostic rationale [13].

The extractor was instructed to return source-supported text spans or minimally reorganized source content when available, and to leave fields empty when the corresponding information was not explicitly reported (Prompt 1). The structured output mapped case report information into three workflow-specific groups: clinical case elements, figure-linked evidence, and diagnostic assessment references.

Prompt 1: Source-grounded extraction   
Extract only information explicitly stated in the source case report. Every non-empty field must be di  
rectly supported by the source text; otherwise, leave the field empty. Do not infer, speculate, normalize,   
or generate new medical content.   
Organize the output into clinical case elements, figure-linked evidence, and diagnostic assessment refer  
ences. Clinical case elements must exclude the final diagnosis and image-derived diagnostic information.   
Keep routine laboratory findings separate from key diagnosis-informative findings. For figure-linked evi  
dence, extract only source-text descriptions ofthe figure identifier, image type or modality, image finding,   
and diagnostic significance; do not infer findings from image pixels. For diagnostic assessment references,   
extract the article-reported final diagnosis and source-supported diagnostic reasoning.

## 3.3.1. Clinical case elements

captured patient information, primary concern, medical, family, and psychosocial history, physical examination, and important clinical findings, routine laboratory testing, key diagnosis-informative findings, diagnostic methods, and timeline from the episode of care. Routine laboratory testing captured baseline clinical tests, whereas key diagnosis-informative findings captured special tests or results that directly supported the diagnosis, such as immunohistochemistry, molecular or genetic testing, microbiological identification, tumor markers, or flow cytometry. The diagnostic tests field contained only the names or types of procedures and examinations performed, with all associated diagnostic results excluded.

## 3.3.2. Figure-linked evidence

included the image type or modality, clinical findings from the image, and diagnostic significance when available. Diagnostic significance was defined as source-article text that explicitly stated how an image finding supported, confirmed, suggested, or ruled out a diagnosis or diferential diagnosis. These fields were extracted from article text, figure captions, or adjacent figure legends when explicitly available. The extraction process did not inspect or interpret image pixels.

## 3.3.3. Diagnostic assessment references

included the article-reported final diagnosis and source-supported sentences describing the case-specific evidence or diagnostic process leading to that diagnosis. The extractor was prohibited from rewriting the diagnosis or generating additional clinical reasoning.

## 3.4. Progressive Dialogue Construction

Dialogues were constructed as progressive multimodal diagnostic dialogues using the computer-interpretable case representations described above. Figure 2 summarizes the saved dialogue QA format.

The progressive evidence-disclosure protocol followed a clinical workflow structure inspired by prior work on autonomous medical AI agents, in which disease history, physical examination, laboratory testing, imaging, and downstream diagnostic actions are represented as sequential clinical information nodes [14]. In our implementation, the model first received the clinical history, including patient demographics, chief complaint or primary concern, and, when available, the history of present illness. It then received physical examination findings and routine laboratory results. Key diagnosis-informative findings were withheld from the initial clinical turn because highly diagnostic laboratory, molecular, microbiologic, genetic, pathology, or immunohistochemistry results could directly reveal the final diagnosis before image review.

![](images/ec5caa6ac0e65eff8a4a1797c148a9fc8231345f39213edacfe503313ff33f68.jpg)  
Figure 2: Example of a progressive multimodal diagnostic dialogue constructed from a PMC case report under the staged evidence-disclosure protocol.

Medical images were then presented sequentially in source-article order. Each image-review turn included the raw image and non-diagnostic metadata, such as examination type or image modality. The prompt instructed the evaluated LLM to describe visible clinical findings, distinguish direct observations from interpretation, and use cautious diagnostic language when uncertain. Reference image-review responses were constructed from the article-derived image finding and diagnostic significance for each figure; the generator was instructed to preserve the medical meaning of these references while rewriting them as concise physician-style responses without unsupported additional findings.

After all image-review turns were completed, key diagnosis-informative findings were disclosed, and the final synthesis turn was constructed. This turn required diagnostic reasoning and a final diagnosis based on all clinical information, image review responses, and key laboratory or diagnostic findings. For the reference dialogue, the final diagnostic reasoning was generated using the article-derived diagnostic rea

soning.

## 3.5. Atomic Item Evaluation

Generated dialogues were evaluated using an item-decomposed LLM-as-judge workflow for final diagnosis, diagnostic reasoning, and image findings. For each target, the article-derived reference and dialogue output were decomposed into target-specific atomic items, and a separate LLM matcher compared the resulting items under one-to-one matching constraints (Prompt 2).

From the resulting atomic item matches, true positives (TPs) were defined as generated items matched to reference items under the specified criterion. False negatives (FNs) were reference items without a corresponding generated item, representing missing evidence. False positives (FPs) were generated items without a corresponding reference item, representing extra generated content. For strict metrics, only strict matches were counted as TPs; for lenient metrics, both strict and lenient matches were counted as TPs.

## Prompt 2: Atomic item matching

Match article-derived reference items to dialogue-output items. Use extracted items only, not raw source paragraphs. Do not further decompose items, calculate metrics, or match by list order.

Apply one-to-one matching: each item may be used at most once. Return strict matches, lenient matches, and contradictions. Strict matches preserve the clinically essential meaning of the reference item. Lenient matches share the same diagnostic or visual evidence core but may be broader, narrower, incomplete, or less precise. Contradictions require direct clinical incompatibility. Vague overlap, same-organ overlap, broad disease-category overlap, or matches requiring external inference should be treated as no match. Unmatched dialogue-output items should not automatically be classified as contradictions.

## 3.5.1. Final-Diagnosis Evaluation

Atomic items included diseases, syndromes, etiologies, causal organisms, disease subtypes, named complications, and causal diagnoses asserted as final diagnoses. The item decomposer preserved clinically essential modifiers when they were part of the diagnosis, including etiology, causal organism, anatomical site, disease subtype, named complication, severity when clinically diagnostic, and causal relationships such as “caused by,” “secondary to,” or “associated with.” We report strict precision, recall, and F1 because omission of these modifiers can change the clinical meaning of the diagnosis.

## 3.5.2. Diagnostic-Reasoning Evaluation

Atomic items were clinically meaningful steps used to support, exclude, diferentiate, localize, characterize, or explain the diagnosis. The same decomposition scheme was applied to the article-derived diagnostic reasoning. Still, the LLM was instructed to keep diagnostically relevant claims made by the evaluated model, even when they were incorrect, unsupported, or inconsistent with the reference diagnosis. This design ensures that missing evidence, unsupported additions, and contradictory reasoning are identified during the matching stage rather than being filtered out during decomposition. We report lenient precision, recall, and F1 because clinically relevant reasoning can difer in wording or granularity while preserving the same diagnostic evidence.

## 3.5.3. Image-Finding Evaluation

An image-finding item was defined as an atomic, clinically meaningful visual observation extracted from either an article-derived image finding or an image-review response. Image-finding items could represent anatomy, lesion morphology, location, laterality, distribution, density, signal, enhancement, color, staining pattern, cellular morphology, nuclear features, or tissue architecture. For example, in one pathology figure, the visible-image units included “spindle-shaped cells,” “fasciculated and disarrayed architecture,” and “nuclei in a palisading pattern.” In contrast, diagnostic impressions such as schwannoma or nerve-sheath tumor were not treated as visible-image units. Image-finding alignment was computed only for figures with both an article-derived image finding and an available generated image-review response. We report lenient recall and F1 because visible findings may be described at diferent levels of specificity.

## 3.6. Case-Level FVCU Scoring

Diagnostic reasoning quality was further evaluated at the case level using a clinical adaptation of reasoningtrace evaluation criteria inspired by prior work [11] (Prompt 3).

For each case, the FVCU judge received a structured summary of the item-level evaluation, containing the reference diagnosis and diagnostic evidence, the model-generated diagnosis and reasoning, and the item-level alignment results, including matched, missing, extra, and contradictory evidence. Based on this information, the judge assigned a score from 1 to 5 for each FVCU dimension.

Factuality assessed whether case-specific clinical, imaging, laboratory, pathology, microbiology, molecular, or genetic claims were supported by the available evidence. Validity assessed whether the diagnostic inferences were clinically reasonable given the evidence presented in the model’s reasoning. Coherence assessed internal consistency and logical organization. Utility assessed whether the reasoning covered, prioritized, and synthesized the case-defining evidence needed to support the article-reported diagnostic pathway.

## Prompt 3: Case-level FVCU judging

Evaluate the model-generated diagnostic reasoning using the structured item-level evaluation summary, including the reference diagnosis and evidence, model-generated diagnosis and reasoning, and matched, missing, extra, or contradictory evidence. Assign a score from 1 to 5 for each predefined FVCU dimension and provide a brief evidence-grounded justification. Then assign a separate overall reasoning-quality score from 1 to 5 as a holistic judgment rather than the arithmetic mean of the four dimension scores. Major factuality or validity errors, or omission of case-defining evidence, should lower the overall score.

## 4. Experiments

## 4.1. Evaluation Dataset

The experimental corpus consisted of 24 publicly available internal medicine case reports from PMC. Across these cases, the image set covered diverse clinical modalities, including radiology and angiography, pathology and cytology, endoscopy, ophthalmic imaging, echocardiography, and electrocardiography. Each case contained a median of 3.5 images (range: 1–8).

## 4.2. Framework Implementation and Evaluation

## 4.2.1. Setup

Within the framework, GPT-4.1 was used for source-grounded information extraction, reference dialogue generation, and item decomposition for automated evaluation. GPT-5.1 was used for semantic item matching and case-level FVCU scoring.

## 4.3. Benchmarking Existing MLLMs

We benchmarked two MLLMs, o4-mini [15] and Claude Haiku 4.5 [16]. Both followed identical staged prompts, source-image order, and evidence-disclosure schedules. Initially, each received only clinical case elements and routine laboratory findings. At each image-review turn, the model received the raw image and non-diagnostic metadata, including the figure identifier and modality when available. Key diagnosisinformative findings were disclosed only during final synthesis. Neither model received figure captions, article-derived image findings, diagnostic reasoning references, or the final diagnosis.

Table 1: Evidence alignment and FVCU scores across dialogue settings. Alignment values are macroaveraged; FVCU values are mean 1–5 scores.
<table><tr><td>Metric</td><td>Reference Dialogue</td><td>04-mini</td><td>Claude</td></tr><tr><td colspan="4">Final Diagnosis</td></tr><tr><td>Strict Precision</td><td>0.99</td><td>0.32</td><td>0.20</td></tr><tr><td>Strict Recall</td><td>1.00</td><td>0.45</td><td>0.39</td></tr><tr><td>Strict F1</td><td>0.99</td><td>0.35</td><td>0.25</td></tr><tr><td colspan="4">Diagnostic Reasoning</td></tr><tr><td>Lenient Precision</td><td>0.67</td><td>0.25</td><td>0.19</td></tr><tr><td>Lenient Recall</td><td>0.77</td><td>0.34</td><td>0.38</td></tr><tr><td>Lenient F1</td><td>0.69</td><td>0.28</td><td>0.24</td></tr><tr><td colspan="4">Image Finding</td></tr><tr><td>Lenient Precision</td><td>0.62</td><td>0.12</td><td>0.06</td></tr><tr><td>Lenient Recall</td><td>0.81</td><td>0.41</td><td>0.39</td></tr><tr><td>Lenient F1</td><td>0.67</td><td>0.17</td><td>0.11</td></tr><tr><td colspan="4">FVCU</td></tr><tr><td>Factuality</td><td>4.62</td><td>3.00</td><td>2.88</td></tr><tr><td>Validity</td><td>4.92</td><td>3.12</td><td>2.62</td></tr><tr><td>Coherence</td><td>5.00</td><td>4.54</td><td>4.29</td></tr><tr><td>Utility</td><td>4.25</td><td>2.29</td><td>2.12</td></tr><tr><td>Overall</td><td>4.79</td><td>2.75</td><td>2.50</td></tr></table>

Final-diagnosis and reasoning metrics were macro-averaged over 24 cases. Image-finding metrics were macro-averaged over 86 evaluable image turns, defined as turns with both an article-derived finding and a generated response. FVCU scores were averaged over 24 cases.

## 5. Results and Discussion

## 5.1. Reliability of Reference Dialogue Construction

We first evaluated whether reference dialogues preserved article-derived diagnostic information after conversion into progressive dialogue format. For each target, reference-dialogue outputs were aligned with the corresponding article-derived reference items. Atomic item evaluation showed high preservation of the article-reported final diagnosis, with macro-averaged strict precision, recall, and F1 of 0.99, 1.00, and 0.99, respectively (Table 1).

Reference dialogues also preserved most article-derived reasoning and image-finding evidence, although precision was lower than recall for both components. Macro-averaged lenient F1 scores were 0.69 for reasoning and 0.67 for image finding. These results reflected core evidence preservation with additional or diferently segmented items introduced during concise dialogue reformulation. The high overall FVCU score of 4.79 further supported their case-level reliability.

## 5.2. Performance of Existing Multimodal LLMs

Both test-model conditions showed substantially lower alignment with the article-derived references (Table 1). Final-diagnosis strict F1 scores were 0.35 for o4-mini and 0.25 for Claude, with strict precision of

only 0.32 and 0.20, respectively.

The largest performance gaps were observed for diagnostic reasoning and image finding. Lenient reasoning F1 was 0.28 for o4-mini and 0.24 for Claude. Lenient image-finding F1 was 0.17 for o4-mini and 0.11 for Claude. The low image-finding precision of the test models, 0.12 for o4-mini and 0.06 for Claude, indicates that many generated visual claims were not supported by the article-derived figure evidence.

Both models also produced relatively coherent reasoning narratives, with coherence scores of 4.54 for o4- mini and 4.29 for Claude. However, their factuality, validity, utility, and overall reasoning-quality scores were lower than those of the reference dialogues. Utility showed the largest gap, decreasing to 2.29 for o4- mini and 2.12 for Claude. This pattern suggests that fluent organization alone was insuficient for clinically useful diagnostic reasoning; evidence coverage and factual grounding were the main limiting factors.

## 5.3. Error Analysis

For the reference dialogue, lower reasoning and image-finding F1 scores were mainly driven by unmatched generated items introduced during dialogue reformulation. The item-level extra generated-item rate, defined as $\frac { F P } { T P + F P }$ , was 0.03 for final diagnosis, 0.35 for diagnostic reasoning, and 0.38 for image findings. These unmatched items were not necessarily clinically implausible; rather, they often reflected additional explanatory wording or diferences in item granularity relative to the article-derived reference.

A separate source-side mismatch arose in one reference image turn. The article-derived ECG finding described “total atrioventricular block with a junctional escape rhythm of 45 bpm.” The generated dialogue preserved the core findings of atrioventricular block and junctional escape rhythm. However, the sourceside extracted image-finding items also included annotation details, such as color markers for P waves and QRS complexes, that were not repeated in the concise dialogue response and therefore contributed to unmatched reference items.

The evaluated MLLMs showed a diferent error pattern. Their image-finding extra-generated-item rates were much higher, 0.89 for o4-mini and 0.94 for Claude, indicating that their low image-finding F1 reflected not only missed article-derived findings but also many generated visual claims that were not supported by the article-derived figure evidence.

## 5.4. Limitations

Several limitations should be considered. First, the study used a fixed, progressive evidence-disclosure protocol with predefined prompts, so it evaluated the interpretation and integration of provided evidence rather than autonomous follow-up questioning, test selection, evidence prioritization, or stopping behavior. Second, the corpus was limited to 24 publicly available internal medicine case reports and two evaluated MLLMs; published case reports may overrepresent unusual or diagnostically challenging presentations and may not reflect routine clinical records, common disease distributions, or other specialties. Third, although the article-derived references were source-grounded, they were not independent or exhaustive expert annotations of image pixels or diagnostic reasoning. Unmatched generated items may therefore represent unsupported claims, wording, or diferences in granularity, or clinically plausible findings not reported in the source article. Fourth, reference dialogue generation, item decomposition, semantic matching, and FVCU scoring relied partly on LLM-based processing, introducing potential judge-model bias and sensitivity to prompt design. Because case-level FVCU scoring was conditioned on the preceding item-decomposition and semantic-matching outputs, it did not provide an independent validation of these evaluation stages. Moreover, the adapted FVCU rubric has not yet been validated as a clinical rating instrument, and formal clinician scoring was not included in the present study. Preliminary clinician feedback suggested that a single holistic score was dificult to apply consistently to reconstructed QA-format dialogues, which may not fully preserve the sequential diagnostic narrowing represented in the original case reports. Future work will therefore develop dimension-specific clinician rating criteria and directly compare clinician assessments with LLM-as-judge scores.

## 6. Conclusion

We introduced a source-grounded framework for evaluating evidence integration in progressive multimodal diagnostic dialogues constructed from PMC case reports. Under a shared staged evidence-disclosure protocol, reference dialogues largely preserved article-derived diagnoses and evidence, whereas two evaluated MLLMs showed substantially lower alignment, especially for image findings. Their high coherence but lower factuality and utility demonstrate that fluent clinical reasoning is not necessarily evidence-grounded. This framework provides a controlled basis for evaluating multimodal evidence interpretation and developing adaptive diagnostic agents and clinician-facing evaluation protocols.

## Acknowledgment

This work was supported by the National Library of Medicine grants R01LM014344 and R01LM014573.

## References

[1] Jason J. Lau, Soumya Gayen, Asma Ben Abacha, and Dina Demner-Fushman. A dataset of clinically generated visual questions and answers about radiology images. Scientific Data, 5(1):180251, 2018. doi: 10.1038/sdata.2018.251.

[2] Bo Liu, Li-Ming Zhan, Li Xu, Lin Ma, Yan Yang, and Xiao-Ming Wu. SLAKE: A semantically-labeled knowledge-enhanced dataset for medical visual question answering. In 2021 IEEE 18th International Symposium on Biomedical Imaging (ISBI), pages 1650–1654. IEEE, 2021. doi: 10.1109/ISBI48211.2021. 9434010.

[3] Chunyuan Li, Clif Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Nau mann, Hoifung Poon, and Jianfeng Gao. LLaVA-Med: Training a large language-and-vision assistant for biomedicine in one day. In Advances in Neural Information Processing Systems, volume 36, pages 28541–28564, 2023.

[4] Michael Moor, Qian Huang, Shirley Wu, Michihiro Yasunaga, Yash Dalmia, Jure Leskovec, Cyril Zakka, Eduardo Pontes Reis, and Pranav Rajpurkar. Med-Flamingo: A multimodal medical few-shot learner. In Proceedings of the 3rd Machine Learning for Health Symposium, volume 225 of Proceedings of Machine Learning Research, pages 353–367. PMLR, 2023.

[5] Tao Tu, Mike Schaekermann, Anil Palepu, et al. Towards conversational diagnostic artificial intelligence. Nature, 642:442–450, 2025. doi: 10.1038/s41586-025-08866-7.

[6] Khaled Saab, Chunjong Park, Tim Strother, et al. Advancing conversational diagnostic AI with multimodal reasoning. Nature Medicine, 32:1726–1736, 2026. doi: 10.1038/s41591-026-04371-0.

[7] Ruoyu Liu, Kui Xue, Xiaofan Zhang, and Shaoting Zhang. Interactive evaluation for medical LLMs via task-oriented dialogue system. In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 4871–4896, Abu Dhabi, UAE, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.coling-main.325.

[8] Ivan Sviridov, Amina Miftakhova, Artemiy Tereshchenko, Galina Zubkova, Pavel Blinov, and Andrey Savchenko. 3MDBench: Medical multimodal multi-agent dialogue benchmark. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 26614–26654, Suzhou, China, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.emnlp-main.1353.

[9] National Library of Medicine, National Center for Biotechnology Information. PMC Open Access Subset. [Online]. Available: https://pmc.ncbi.nlm.nih.gov/tools/openftlist/, 2026. Accessed: Jun. 21, 2026.

[10] Ming Zhang, Yujiong Shen, Zelin Li, Huayu Sha, Binze Hu, Yuhui Wang, Chenhao Huang, Shichun Liu, Jingqi Tong, Changhao Jiang, Mingxu Chai, Zhiheng Xi, Shihan Dou, Tao Gui, Qi Zhang, and Xuanjing Huang. LLMEval-Med: A real-world clinical benchmark for medical LLMs with physician validation. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 4888–4914, Suzhou, China, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025. findings-emnlp.263.

[11] Jinu Lee and Julia Hockenmaier. Evaluating step-by-step reasoning traces: A survey. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 1789–1814, Suzhou, China, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.findings-emnlp.94.

[12] Donald C. Comeau, Chih-Hsuan Wei, Rezarta Islamaj Doğan, and Zhiyong Lu. PMC text mining subset in BioC: About three million full-text articles and growing. Bioinformatics, 35(18):3533–3535, 2019.

[13] Joel J. Gagnier, Gunver Kienle, Douglas G. Altman, David Moher, Harold Sox, and David Riley. The care guidelines: Consensus-based clinical case report guideline development. Journal of Clinical Epidemiology, 67(1):46–51, 2014.

[14] Dyke Ferber, Lars Hilgers, Christiane Höper, et al. Towards autonomous medical artificial intelligence agents. Nature, 2026. doi: 10.1038/s41586-026-10675-5.

[15] OpenAI. Introducing OpenAI o3 and o4-mini. https://openai.com/index/ introducing-o3-and-o4-mini/, April 2025. Accessed: Jun. 24, 2026.

[16] Anthropic. Introducing Claude Haiku 4.5. https://www.anthropic.com/news/claude-haiku-4-5, October 2025. Accessed: Jun. 24, 2026.