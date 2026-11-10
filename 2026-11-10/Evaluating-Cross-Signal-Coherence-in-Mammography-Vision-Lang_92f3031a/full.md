# Evaluating Cross-Signal Coherence in Mammography Vision-Language Models

Rebeca Nam<sup>1,2</sup>, Chase M. Fensore<sup>2</sup>, Christian A. Fauria-Robinson<sup>3</sup>, Judy Gichoya<sup>3</sup>, Joyce C. Ho<sup>2</sup>

<sup>1</sup>Yonsei University, 50 Yonsei-ro, Seodaemun-gu, 03722, Seoul, Republic of Korea.

<sup>2</sup>Department of Computer Science, Emory University, 400 Dowman Drive, Atlanta, 30322, GA, USA.

<sup>3</sup>Department of Radiology and Imaging Sciences, Emory University, 100 Woodruf Circle, Atlanta, 30322, GA, USA.

Contributing authors: rebeca.nam@emory.edu; chase.fensore@emory.edu; christian.allison.fauria-robinson@emory.edu; judywawira@emory.edu; joyce.c.ho@emory.edu;

## Abstract

As vision-language models (VLMs) move toward interpreting medical images directly, their outputs increasingly include not just a label but a stated confidence and a written justification. Yet the field still predominantly evaluates VLMs using just accuracy. We evaluate the coherence between answer, confidence, and explanation across four VLMs on breast density and BI-RADS categorization, using two mammography datasets. We perform controlled perturbations of the question, multiple choice option orders, and the images and assess changes in the three outputs. Our results suggest confidence does not decrease even as the mammography image is changed. Shufling the options also changes most VLMs’ answers. Notably, there are cases where a model’s own stated reasoning contradicts its final answer, a distinction accuracy alone cannot capture but one that governs whether an AI-generated answer can be trusted.

Keywords: vision-language models, medical VQA, mammography

## 1 Introduction

An aging population and rising chronic disease burden have placed increasing imaging demand on the U.S. radiology workforce, lengthening patient wait times and contributing to radiologist burnout [1]. Breast cancer screening ofers a case study for when this workforce strain has clinical consequence. The 2024 U.S. Preventive Services Task Force recommendation statement finds biennial mammography screening from ages 40–74 carries a moderate but real mortality benefit, with modeling estimating a 30% relative reduction in breast-cancer mortality compared to no screening [2]. Thus any degradation in screening capacity or turnaround may negatively impact patient outcomes.

Most mammography AI today assists a human reader rather than replacing one, typically serving as a second reader in the standard double-reading workflow. However, replacing both readers is a fundamentally diferent demand than replacing one. A second reader can assume a first read already exists and need only add a risk score or detection, whereas a system replacing both must produce an assessment with reasoning a clinician can audit. This is a diferent capability than the AI models evaluated in the second-reader studies [3, 4], which were typically trained to output a risk score or bounding box rather than a categorical clinical judgment with attached reasoning. This shift to autonomous roles has started to happen with at least one AI system receiving certification for autonomous mammogram triage, bypassing radiologist review for lowrisk cases [5].

Vision-language models (VLMs), or models that take images and text as input to reason across both modalities and typically produce text as output, are a natural candidate for this automation task. When framed as a visual question answering (VQA) problem, a VLM can generate a category assessment (e.g., a density score), a confidence estimate, and a natural-language rationale over the screening mammography images, organically mirroring how a radiologist conducts a full read of a mammogram. VLM usage for chest x-rays has been studied heavily, with several VQA benchmarks and robustness analyses [6–9], but these works largely score only the generated answer, and high performance on such a benchmark is not necessarily suficient to establish VLM understanding or generalizability. Limited work assesses whether a chest X-ray VLM’s stated reasoning can be trusted rather than just its answer [10] or whether existing explanation methods can identify the image regions that actually drove the model’s prediction [11].

Each of those evaluations, however, still treats one signal at a time. Separate strands of general-domain VQA research show that each signal is individually unreliable: answers are sensitive to superficial reformatting [12, 13], models routinely hallucinate a choice rather than abstain [14, 15], and confidence and reasoning are each known to be poorly calibrated or unfaithful [16–20]. What remains unseen is the extent to which the output signals (i.e., confidence-derived answer, confidence, and reasoning) agree with each other, and whether all three are grounded in the visual evidence, when elicited jointly from the same response. This evidence is necessary for a radiologist to trust or audit an AI-generated read.

We study this question in mammography rather than chest radiography, since its interpretation tasks ofer several properties well suited to testing coherence. Its two standard interpretation tasks, (i) breast density with American College of Radiology (ACR) categories A–D, and (ii) BI-RADS assessment with categories 0–6, are ordinal and multi-category. Thus a model’s errors, confidence, and reasoning can be graded rather than simply right or wrong. Both label sets are standardized clinical lexicons defined by the ACR and follow a standard imaging protocol so behavior under perturbation can be cleanly attributed. Breast density is also subject to federal reporting and patient-notification requirements in the U.S., and BI-RADS categories directly inform clinical action, from routine recall to biopsy. Here, a model’s confidence and reasoning shape what a patient is told and what happens next.

We test this using two subsets, RSNA and EMBED, from MammoVQA [21], the largest VQA benchmark for mammography. Both provide breast density and BI-RADS labels over a large, real screening population. We evaluate four open VLMs. MedGemma-4B and Lingshu-7B are medically trained, while Qwen2-VL-7B and InternVL3-8B are general-purpose. Rather than scoring a single response, we jointly elicit the answer, a per-option confidence score, and a natural-language rationale from the same output, then perturb every question. We replace, blank, or remove the mammogram to probe visual grounding, and reword the question or reorder the options to probe elicitation robustness. A grounded model should grow less confident as the image degrades and reason consistently with the final assessment. Instead, across both datasets and all four models, we find the opposite. Reasoning is sometimes correct where the scored answer is wrong, confidence fails to track the evidence and at times moves the wrong way, models conflate appropriate abstention with overconfident guessing, and answers flip with the question’s format rather than its content. None of these are visible from accuracy alone.

## 2 Related Work

Evaluating VLMs beyond accuracy. A large body of work probes VLMs beyond a single accuracy score. Reordering multiple-choice options or rephrasing prompts can flip answers and shift reported confidence [12, 13]. Perturbation-based protocols test visual dependence by corrupting, masking, or substituting images [10, 22]. Similarly, blind baselines and negation probes expose failures in medical VQA [23]. Adversarial “None of the Above” (NOTA) protocols reveal that models often hallucinate answers when no valid option exists, in both general multimodal domains [14] and medical diagnostics [15]. These methods primarily assess answer correctness or visual dependence, whereas we examine how answers, confidence, and reasoning jointly respond to visual perturbations.

Output coherence and confidence calibration. Faithfulness methods assess whether explanations reflect the model’s internal decision process. Siegel et al. [24] use counterfactual edits to measure changes in predicted probabilities and whether explanations acknowledge those edits. Parcalabescu and Frank [25] examine explanation faithfulness and self-consistency, while interventional tests probe chain-of-thought faithfulness by corrupting reasoning traces [26]. These methods address internal faithfulness, whereas we assess whether a rationale agrees with its selected answer and the imaging evidence. Confidence calibration is likewise studied through aggregate calibration error [27] and methods that elicit fidelity across sampled responses [28]. In contrast, we examine whether confidence responds to visual evidence and agrees with the answer and rationale within a single response. Aggregate calibration and cross-sample consistency do not establish such within-response coherence.

![](images/41067e22d2949e6a7645348ff6ac18c6bb7b0a049933683605d0acd2669dcd87.jpg)  
Fig. 1 Our proposed evaluation methodology with six conditions: the original image and five perturbations.

Medical VLM evaluation. VLMs for chest radiography are evaluated predominantly using report-derived accuracy metrics, including exact- or close-match accuracy on VQA pairs [6, 7] and overlap between generated and reference reports [6]. Mammography-specific VLMs follow a similar pattern. MammoVLM [29] reports performance approaching junior-radiologist-level performance on a private dataset, based on radiologist ratings of correctness, rationality, helpfulness, professionalism, and completeness. A separate reinforcement learning-based VLM [30] is trained on EMBED data using a domain-specific reward but evaluates performance primarily through answer correctness. Neither model’s weights are publicly released.

## 3 Methodology

## 3.1 Overview

General VLM research probes accuracy, reasoning, and confidence individually with limited work on how these signals interrelate. There is limited work detecting when a model is confidently wrong, right in its explanation yet wrong in its answer, or unchanged by the removal of the image. We address this gap directly. For each visual question, a VLM produces three signals a clinician might rely on: the answer it selects, the confidence it reports, and the reasoning it gives. We jointly elicit all three from a single response and test whether they agree with each other and with the visual evidence, using mammography as a testbed, under five controlled perturbations, shown in Figure 1, known to afect model responses. We also provide the code to reproduce our results.

1) State your single best answer as the option letter.   
2) For EACH option, give a confidence score from 0 to 100 for how likely THAT option is the correct answer (score each option independently; they need not sum to 100). 3) Give a brief one-sentence explanation of your reasoning.

Fig. 2 Prompt used to jointly elicit an explicit answer, per-option confidence scores, and reasoning in a single response.

## 3.2 Dataset and Tasks

We use the MammoVQA benchmark [21], focusing on its two largest cohorts, the RSNA screening mammography dataset [31] and EMBED [32]. We study two interpretation tasks: breast density (ACR categories A–D) and BI-RADS assessment (categories 0–6), both of which in theory require reading the image rather than exploiting coarse spatial cues. The RSNA screening cohort contains N=5,106 BI-RADS assessment and $N { = } 5 , 8 2 1$ breast density / ACR items while EMBED contains N=14,303 BI-RADS assessment items. Each item is a single mammographic image (RSNA) or multiple exam images (EMBED) paired with a templated question, a fixed option set, and a benchmark label y (detailed statistics in Appendix B).

## 3.3 Signal elicitation

We elicit the answer, confidence, and reasoning jointly using a structured multiplechoice prompt with per-option confidence scores following the Matchmaker formulation [18]. [18]. The original MammoVQA benchmark poses each question as a closed multiple-choice question (MCQ) with no option for rejecting the ofered choices. Since we will perturb the original image, none of the original options are correct, so the existing options would force the model into an arbitrary wrong answer and provide no mechanism for distinguishing a grounded refusal from a guess. We therefore augment each question with an explicit NOTA (“None of the above”) option, representing the claim that no listed answer is correct.

Each question is thus presented as an MCQ over its K options plus NOTA (Figure 2). From the response we recover three signals: the answer ${ \hat { y } } ,$ the option the model explicitly states; the confidence $c = s _ { \hat { y } }$ , the score it assigns to that answer, from the per-option scores $\mathbf { s } \in [ 0 , 1 0 0 ] ^ { K } ;$ and the reasoning $\hat { y } _ { e } ,$ the answer named in the free-text explanation. The independent per-option scores do not force a normalized distribution, allowing the model to assign all options low scores when no option is well supported by the evidence.

## 3.4 Perturbation conditions

Each question is evaluated under 6 conditions: an original baseline using the unmodified question and mammogram, and 5 conditions that perturb the visual evidence or the elicitation while holding the question otherwise fixed. The K options plus NOTA are presented under every condition. The 3 visual perturbations probe grounding: a VLM that reads the image should change its behavior when the image is made uninformative or removed. Under the unrelated condition, the mammogram is replaced with an unrelated natural photograph. In the blank scenario, it is replaced with a blank, grayscale image. For the no-image perturbation, the image is omitted entirely and the model answers from the question text alone, probing the model’s language prior in the absence of any visual input. The two elicitation perturbations hold the mammogram fixed and instead alter how the question and answer are posed, isolating a format efect independent of the evidence. Under the perturbed-question condition, any domain-specific terminology is neutralized (e.g., “mammogram” is replaced with “image”) to probe sensitivity to wording. In perturbed-options scenario, the answer options are shufled and augmented with paraphrase filler distractors (e.g., “Not Assessable,” “Not Applicable”) alongside NOTA, probing sensitivity to option order and format.

NOTA correctness. In the three conditions with no valid mammogram (unrelated, blank, no-image), the appropriate answer is to select NOTA. We therefore define a second target in which the correct answer is NOTA, and report performance under both the original label (did the model recover the benchmark answer) and this NOTA-aware label (did the model correctly decline). The former measures classification while the latter appropriate refusal.

## 3.5 Coherence metrics

Let yˆ, c, and $\hat { y } _ { e }$ denote the answer, confidence, and reasoning-answer respectively for a given item and condition, and let y be the benchmark label. The reasoning-answer $\hat { y } _ { e }$ is extracted from the explanation with a rule-based matcher.

Reasoning Accuracy. We combine a label matcher with a definition-based semantic matcher. (i) The label matcher identifies explicit category mentions by locating task cues $( \mathrm { ^ { 6 4 } B I { - } R A D S ^ { 3 } }$ for assessment; “level”, “category”, “ACR”, or “density” for density) and extracting the nearby category token (digit 0–6 for BI-RADS; letter A–D for density), using the last match when several occur. (ii) The semantic matcher maps descriptions of findings to categories using oficial definitions. For example, “no abnormality / negative” → BI-RADS 1 and “almost entirely fatty / scattered fibroglandular / heterogeneously dense / extremely dense $\mathrm { " \to A / B / C / D }$ , while also identifying “none of the above” conclusions.

To validate the matcher we drew a random sample of 120 explanations (15 per model per task, stratified across the four models and both tasks) and manually annotated which category the explanation supports or $\mathrm { ^ { * } n o n e ^ { * } } /$ “unparsable”. The annotator was blind to the matcher output and the benchmark label. Agreement between the matcher and the human labels was substantial: Cohen’s $\kappa = 0 . 7 8$ (80.8% exact agreement), and comparable on both tasks (BI-RADS $\kappa = 0 . 7 9$ , 83.3%; density $\kappa = 0 . 7 3 , 7 8 . 3 \% )$ . Unparseable explanations are excluded from the reasoning-accuracy denominator (see Appendix C.1 for examples of unparsable instances).

Signal agreement. We report how often the answer disagrees with the reasoning $\left( \hat { y } \neq \hat { y } _ { e } \right)$ and with the highest-scored option $\left( \hat { y } \neq \arg \operatorname* { m a x } _ { k } s _ { k } \right)$ , and how often all three agree. A coherent model should state the option it scores highest and describe that same option in its explanation.

Confidence–evidence alignment. For a given item and condition, let $c _ { y } =$ $s _ { y }$ denote the score placed on the benchmark-correct option y, and $c _ { \hat { y } } ~ = ~ s _ { \hat { y } } ~ = ~ c$ the score placed on the model’s chosen option. We track both, averaged per model and condition, across the evidence and elicitation perturbations. A grounded, wellcalibrated model should have $c _ { y }$ increase as informative visual evidence is restored, and $c _ { \hat { y } }$ should track $c _ { y }$ closely. We report the gap $c _ { \hat { y } } - c _ { y } .$ a positive gap measures overconfidence in an incorrect selection, while a negative gap means the model scored the correct option higher than the answer it chose, which is itself an incoherence between answer and confidence.

NOTA sensitivity. We report two summary quantities per model and task. The real→visual diference is the NOTA-selection rate on the three visual perturbations (unrelated, blank, no-image), averaged, minus the rate on the real image. A positive value indicates the model abstains more often when evidence is genuinely absent. $\Delta _ { \mathrm { N O T A } } = \mathrm { A c c } _ { \mathrm { N O T A } } - \mathrm { A c c } _ { \mathrm { b a s e } }$ is the NOTA-aware accuracy minus the originallabel accuracy. A large ∆ paired with a small real→visual diference indicates blanket refusal rather than genuine evidence-sensitivity.

Reasoning–answer accuracy gap. On items with a parseable explanation, we compare reasoning accuracy $( \hat { y } _ { e } = y )$ to answer accuracy $( \hat { y } = y )$ on the same items. A positive gap means the explanation identifies the correct label more often than the stated answer does; a negative gap means the reverse.

Coherence taxonomy. Each item with a parseable explanation is classified by whether $\hat { y }$ and $\hat { y } _ { e }$ each equal $y ,$ giving four buckets: A $\scriptstyle ( { \hat { y } } = y , \ { \hat { y } } _ { e } = y )$ , coherent and correct; B $( \hat { y } { = } y , \hat { y } _ { e } { \neq } y )$ , a correct answer with unsupported reasoning; $\mathbf { C _ { \lambda } } ( \hat { y } \neq y , \hat { y } _ { e } = y )$ , a wrong answer with correct reasoning; and D $( \hat { y } \neq y , \ \hat { y } _ { e } \neq y )$ , both wrong. Restricting to confident errors $( c \geq 9 0 )$ , the recoverable rate is the confident subset of C over the confident subsets of C and D combined: the share of a model’s confident errors whose own reasoning already contains the right answer.

## 3.6 Models

We evaluate four open VLMs spanning medical and general-purpose training: MedGemma-4B and Lingshu-7B (medical), and Qwen2-VL-7B and InternVL3-8B (general). These models were selected for three reasons: (1) open weights enable reproducibility without costly API access, excluding closed-source models (e.g., GPT-4V and Gemini Ultra) and mammography-specific systems [29, 30] whose weights are unavailable; (2) the selection enables comparison of medical versus generalpurpose training; and (3) the 4–8B parameter range, using unquantized models, limits diferences attributable to model scale and quantization. All models receive an identical instruction; only the model-specific chat template and image-token convention difer (checkpoint and inference settings in Appendix A). Decoding is greedy for reproducibility.

## 4 Results

Table 1 provides the overall picture for each task. Under direct elicitation, no model exceeds the majority-class prior on BI-RADS, while on density several approach or exceed it (e.g. Lingshu at 49.3% vs. the 42.6% prior). Explicit abstention is rare for most models $( \approx 0 \% )$ , with Qwen the exception (62.8% / 33.2% on BI-RADS / density).

Table 1 Per-task output accuracy and “None of the above” selection rate, with the majority-class prior. The answer is the model’s explicitly stated single-best choice.
<table><tr><td>Model</td><td>Output acc. %</td><td>NOTA sel. %</td></tr><tr><td>RSNA – BI-RADS (6 true classes; majority 59.4%) MedGemma-4B Lingshu-7B</td><td>42.4 12.0</td><td>0.0 0.0</td></tr><tr><td>Qwen2-VL-7B InternVL3-8B</td><td>2.9 7.7</td><td>62.8 13.7</td></tr><tr><td>RSNA − Density (ACR) (4 true classes; majority 42.6%)</td><td></td><td></td></tr><tr><td>MedGemma-4B</td><td>33.9</td><td>0.0</td></tr><tr><td>Lingshu-7B</td><td>49.3</td><td>0.0</td></tr><tr><td> $\mathrm { Q w e n 2 - V L - 7 B }$ </td><td></td><td></td></tr><tr><td></td><td>24.8</td><td>33.2</td></tr><tr><td>InternVL3-8B</td><td>42.0</td><td>0.0</td></tr><tr><td></td><td></td><td></td></tr><tr><td>EMBED – BI-RADS (6 true classes; majority 62.1%)</td><td></td><td></td></tr><tr><td>MedGemma-4B</td><td>36.7</td><td>0.0</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Lingshu-7B</td><td>10.3</td><td>0.0</td></tr><tr><td> $\mathrm { Q w e n 2 - V L - 7 B }$ </td><td>15.2</td><td>0.7</td></tr><tr><td>InternVL3-8B</td><td>5.1</td><td>0.1</td></tr></table>

Output acc.: the explicit answer ˆy equals ground truth y. NOTA sel. %: rate at which the model explicitly selects “None of the above”.

Low accuracy does not necessarily reflect simple ignorance, however, the confidence and reasoning signals carry additional information about VLM behavior.

## 4.1 Answer–Reasoning inconsistency

Elicited directly, a model’s explicit answer and its written rationale are largely coherent: the taxonomy (Figure 3) is dominated by the two agreement buckets, A (both correct) and D (both wrong), while the disconnect bucket C (answer wrong but reasoning correct) is negligible (< 5% for every model/task). Coherence, however, is not competence. The largest bucket on BI-RADS is D—the model’s answer and its reasoning agree and are both wrong—at 56% (MedGemma), 83% (Lingshu), 89% (InternVL), and 92% (Qwen), against a coherent-correct bucket A of only 3–34%. Density is less extreme (A = 32–48%, D = 52–67%) but shows the same pattern.

Coherent does not mean valid. Within the agreement buckets, the findings a model describes often fail to warrant the category it assigns. Figure 4 shows this on a single screening exam (ground truth BI-RADS 0). InternVL assigns BI-RADS 0 and is therefore scored correct, yet its rationale describes the breast as normal with no abnormalities, a BI-RADS 1 reading, so the category it reports contradicts the finding it describes. MedGemma reads the exam the same way but, reasoning validly from “normal,” assigns BI-RADS 1 and is scored wrong. Accuracy thus credits the model whose reasoning is self-contradictory and penalises the one whose reasoning is sound: a correct label is no evidence that the model read the image correctly, and only inspecting the rationale exposes the disconnect. This is not an isolated case. Across the agreement buckets, the findings a model describes routinely fail to justify the category it assigns. Additional results are in Appendix C.1.

![](images/566ea45b5217182a716767c86bef1863cfc8b3fcbe30ac98fab380215f98d328.jpg)  
Fig. 3 Answer × reasoning coherence on RSNA. Each item is classified by whether the explicit answer ˆy = y and the reasoning answer $\hat { y } _ { e } = y \colon \mathrm { A }$ (both correct), B (answer correct, reasoning wrong), C (answer wrong, reasoning correct; the disconnect), D (both wrong).

![](images/2183ddc464ce4f42737fdb21486eea8bcfb74946c6ab62d179c00cae32ffac73.jpg)  
Fig. 4 Two models on the same RSNA mammogram (ground truth BI-RADS 0). Each model’s stated answer agrees with the category implied by its explanation, yet the correct answer (InternVL, BI-RADS 0) rests on a self-contradictory rationale—it calls the exam “normal” (a BI-RADS 1 finding) while assigning BI-RADS 0—whereas the incorrect answer (MedGemma, BI-RADS 1) follows validly from the same normal reading.

## 4.2 Confidence–evidence inconsistency

Figure 5 reports the mean confidence each model assigns to the condition-correct answer, $c _ { y }$ (solid), and to the option it explicitly chose, $c _ { \hat { y } }$ (dashed). The correct answer is the original category on the real-image conditions and NOTA on the three visual perturbations. A grounded, calibrated model should maintain a high $c _ { y }$ across all conditions, confident in the category when the mammogram is present, and in NOTA once it is removed.

![](images/c3c853b6e15951f421dd165db2f8fcdb029a9c22ba56184b57e1560833079397.jpg)  
Fig. 5 Mean confidence on the condition-correct answer, $c _ { y }$ (solid) the original category on realimage conditions, NOTA on the visual perturbations vs. on the explicitly chosen answer, $c _ { \hat { y } }$ (dashed), across the six RSNA conditions.

Table 2 RSNA “None of the above” selection rate on the real image vs. its average over the three visual perturbations (unrelated/blank/no image), where NOTA is the correct response, with ∆ = Visual − Real.
<table><tr><td>Model Real % Visual %</td></tr><tr><td>∆ (pp) BI-RADS</td></tr><tr><td>MedGemma-4B 0.0 0.0 +0.0</td></tr><tr><td>Lingshu-7B 0.0 20.6 +20.6 Qwen2-VL-7B 62.8 91.7 +28.9</td></tr><tr><td>InternVL3-8B 13.7 91.0 +77.3</td></tr><tr><td>Density (ACR)</td></tr><tr><td>MedGemma-4B 0.0 12.7 +12.7</td></tr><tr><td>Lingshu-7B 0.0 20.9 +20.9</td></tr><tr><td>Qwen2-VL-7B 33.2 81.5 +48.3</td></tr><tr><td>InternVL3-8B 0.0 75.3 +75.3</td></tr></table>

The models instead keep $f c _ { y }$ low while $c _ { \hat { y } }$ remains near 90–100, indicating confidence that is consistent but misdirected. Two opposite failure modes emerge. MedGemma is confident in the correct category on the real image (BI-RADS $c _ { y } =$ 60.3), yet this collapses to near zero under the visual perturbations, where it keeps backing a category $( c _ { \hat { y } } \approx 9 5 )$ instead of abstaining. Qwen does the reverse: it defaults to NOTA almost regardless of the image, so $c _ { y }$ is near zero on the real image (2.9) and high on the junk conditions (75–100), resulting correct only because it always abstains. Density follows the same split (MedGemma $4 3 . 0  1 . 0 $ ; Lingshu $4 6 . 8  0 . 0$ without an image). In every case the models are confident, however, in none does that confidence track the evidence.

![](images/973e5a6b60dcfa210ba0ff1b49b19c9a3817518a00eb2d50cf4077b14df1eb7e.jpg)  
Fig. 6 Perturbed-Options condition efect on RSNA. Bars = share of items whose stated answer changes when the options are reordered, split by efect on accuracy.

## 4.3 NOTA selection

Table 2 reports the explicit NOTA-selection rate on the real image against its average over the three visual perturbations (unrelated / blank / no image), where NOTA is the correct response, alongside $\Delta _ { \mathrm { N O T A } }$ . Two regimes emerge. Evidence-sensitive models raise their NOTA rate substantially under perturbation: InternVL from 14 to 91 (+77) on BI-RADS, and Qwen from 63 to 92 (+29). Evidence-insensitive models barely respond: MedGemma holds at 0 on BI-RADS (∆ = +0).

The shift alone, however, overstates how well these models abstain. Lingshu’s +21 raises its visual NOTA rate only to 21%: it still returns a categorical answer to roughly four of every five invalid images. MedGemma is worse, abstaining on 0% of junk BI-RADS images and 13% on density; it assigns a definite category to essentially every image that carries no valid evidence. Only InternVL and Qwen abstain on the majority of junk (75–92%), and even they commit to a category on up to a quarter of the density perturbations. Additional results are in Appendix C.2.

## 4.4 Perturbation sensitivity

Figure 6 reports what happens when the answer options are shufled, measured as the share of items whose stated answer changes, split by whether the change helps, hurts, or leaves accuracy unchanged. An answer read from the mammogram would be invariant to the order in which the choices are listed; instead we see flip rates of 40–99% (77% for MedGemma and 57% for InternVL on BI-RADS, 55–78% on density). That a single reordering overturns most answers indicates that the choice is anchored to the surface arrangement of the options rather than to evidence in the image. The direction of the flips makes the same point: they are overwhelmingly neutral (wrong→wrong), so reordering reshufles an already-ungrounded guess rather than correcting it, and where accuracy does move it predominantly hurts (net −24.8 for MedGemma to +13.4 for InternVL on BI-RADS). Qwen is the most volatile, flipping on 99% of BI-RADS and 75% of density items almost entirely by retreating to a newly-introduced NOTA/filler option (56% and 48% of all items): its answer tracks whichever safe option is present, not the image. The asymmetries are significant under McNemar’s test (Appendix C.4). Rewording the question, by contrast, leaves most answers intact (Appendix C.3), so the fragility is specific to option structure, not phrasing.

## 5 Discussion

Our results show that a model’s answer, confidence, and explanation are largely decoupled from the imaging evidence, and that coherence among these signals does not imply sound reasoning. This is a diferent problem from the one most medical VLM evaluation is built to catch. Accuracy-based benchmarks assume a wrong answer reflects a gap in knowledge or perception, but the gap here can instead sit within the reasoning itself, or between the stated answer and the explanation. We cannot distinguish whether the explanation reflects the model’s actual decision process or is a post-hoc rationalization generated somewhat independently of it. Prior work on chainof-thought faithfulness suggests the latter is common [19, 20], and recent work on visual grounding of VLM rationales finds explanations are often produced after only a superficial inspection of the image, with individual steps unsupported by the evidence they cite [33]. Either way, the explanation is a signal a clinician would plausibly act on, so its disagreement with the stated answer, or its invalidity, is a practical problem for trust and auditability regardless of which signal is closer to the model’s true internal state. This same ambiguity extends to abstention: a high refusal rate under evidence removal can reflect genuine grounding or a fixed refusal policy, and only comparing it against the real-image rate tells the two apart.

## 5.1 Limitations and Future Work.

Our evaluation relies solely on MammoVQA’s templated multiple-choice formulation, so some option-order and wording sensitivity may reflect the task framing. Extending the protocol to free-response or report-generation formats would test its generality. We also evaluate open, general-purpose and medically-adapted VLMs because mammography-specialized systems have not released their weights; whether domain-specific fine-tuning changes the coherence failures remains an open question.

A separate direction concerns the confidence-evidence inversion we observe. This may relate to mirage reasoning, where multimodal models produce confident clinical descriptions without image input [34]. Testing explicit versus implicit image-removal prompts could help distinguish these phenomena.

Our perturbations omit clinically relevant changes such as swapped laterality or view, and same-patient prior images. Post-hoc verification methods could test whether reasoning-correct, answer-wrong items reflect visual grounding [33]. Our reasoning matcher was validated on a modest sample, and parse rates vary across models. Future work could compare verbalized with logit-based confidence, since independent option scores do not necessarily form a distribution the way normalized probabilities do.

## References

[1] Jing, A.B., Garg, N., Zhang, J., Brown, J.J.: Ai solutions to the radiology

[2] Force, U.P.S.T., Nicholson, W.K., Silverstein, M., Wong, J.B., Barry, M.J., Chelmow, D., Coker, T.R., Davis, E.M., Ja´en, C.R., Krousel-Wood, M., et al.: Screening for breast cancer: Us preventive services task force recommendation statement. Jama 331(22), 1918–1930 (2024)

[3] Elhakim, M.T., Stougaard, S.W., Graumann, O., Nielsen, M., Gerke, O., Larsen, L.B., Rasmussen, B.S.: Ai-integrated screening to replace double reading of mammograms: a population-wide accuracy and feasibility study. Radiology: Artificial Intelligence 6(6), 230529 (2024)

[4] Warren, L.M., Venton, J., Young, K.C., Halling-Brown, M., Kelly, C.J., Wilson, M., Morigami, M., Khoo, L., Cunningham, D., Sidebottom, R., et al.: Impact of using artificial intelligence as a second reader in breast screening including arbitration. Nature Cancer 7(3), 507–521 (2026)

[5] Vara: Vara Receives World-First CE Certification for Autonomous AI in Breast Cancer Screening. https://www.businesswire.com/news/home/20260901360645/ en/Vara-Receives-World-First-CE-Certification-for-Autonomous-AI-in-Breas t-Cancer-Screening (2026)

[6] Liu, B., et al.: GEMeX: A large-scale, groundable, and explainable medical VQA benchmark for chest X-ray diagnosis. arXiv preprint arXiv:2411.16778 (2024)

[7] Lau, J., Gayen, S., Ben Abacha, A., Demner-Fushman, D.: A dataset of clinically generated visual questions and answers about radiology images. Scientific Data 5(1), 180251 (2018)

[8] Pal, A., Lee, J.-O., Zhang, X., Sankarasubbu, M., Roh, S., Kim, W.J., Lee, M., Rajpurkar, P.: Rexvqa: A large-scale visual question answering benchmark for generalist chest x-ray understanding. In: Biocomputing 2026: Proceedings of the Pacific Symposium, pp. 251–264 (2025). World Scientific

[9] Fallahpour, A., Ma, J., Munim, A., Lyu, H., Wang, B.: MedRAX: Medical reasoning agent for chest x-ray. In: Singh, A., Fazel, M., Hsu, D., Lacoste-Julien, S., Berkenkamp, F., Maharaj, T., Wagstaf, K., Zhu, J. (eds.) Proceedings of the 42nd International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 267, pp. 15661–15676. PMLR, Vancouver, Canada (2025). https://proceedings.mlr.press/v267/fallahpour25a.html

[10] Moll, J., Graf, M., Lemke, T., Lenhart, N., Truhn, D., Delbrouck, J.-B., Pan, J., Rueckert, D., Adams, L.C., Bressem, K.K.: Evaluating reasoning faithfulness in medical vision-language models using multimodal perturbations. In: Proceedings of Machine Learning for Health (ML4H) (2025)

[11] Xiong, G., Jin, Q., Sinha, S., Lu, Z., Zhang, A.: Rethinking visual attribution for chest x-ray reasoning in large vision language models. arXiv preprint arXiv:2605.20158 (2026)

[12] Pezeshkpour, P., Hruschka, E.: Large language models sensitivity to the order of options in multiple-choice questions. In: Findings of the Association for Computational Linguistics: NAACL 2024, pp. 2006–2017 (2024)

[13] Zheng, C., Zhou, H., Meng, F., Zhou, J., Huang, M.: Large language models are not robust multiple choice selectors. In: Proceedings of the International Conference on Learning Representations (ICLR) (2024)

[14] Tong, B., Xia, J., Shang, S., Zhou, K.: Measuring epistemic humility in multimodal large language models. arXiv preprint arXiv:2509.04664 (2025)

[15] Demirci, A., Bayraktar, E., Dere, B., Erdem, E., Erdem, A.: When the right answer is missing: Probing hallucination in medical reasoning models via nota evaluation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops (2026)

[16] Tian, K., Mitchell, E., Zhou, A., Sharma, A., Rafailov, R., Yao, H., Finn, C., Manning, C.D.: Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 5433–5442 (2023)

[17] Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., DasSarma, N., Tran-Johnson, E., et al.: Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221 (2022)

[18] Seedat, N., Van Der Schaar, M.: Bootstrapping self-improvement of language model programs for zero-shot schema matching. In: Singh, A., Fazel, M., Hsu, D., Lacoste-Julien, S., Berkenkamp, F., Maharaj, T., Wagstaf, K., Zhu, J. (eds.) Proceedings of the 42nd International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 267, pp. 53791–53826. PMLR, Vancouver, Canada (2025). https://proceedings.mlr.press/v267/seedat25a.html

[19] Turpin, M., Michael, J., Perez, E., Bowman, S.R.: Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting. In: Advances in Neural Information Processing Systems (NeurIPS), vol. 36, pp. 74952–74965 (2023)

[20] Bowman, S.R., Hyun, J., Perez, E., Chen, E., Pettit, C., Heiner, S., Lukosiute, K., Askell, A., Jones, A., Chen, A., et al.: Measuring progress on scalable oversight for large language models. arXiv preprint arXiv:2211.03540 (2022)

[21] Zhu, J., Huang, F., Luo, Q., Chen, H.: A benchmark for breast cancer screening

and diagnosis in mammogram visual question answering. Nature Communications 16, 66507 (2025)

[22] Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., Xie, S.: Eyes wide shut? exploring the visual shortcomings of multimodal llms. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 9568–9578 (2024)

[23] Yan, Q., He, X., Yue, X., Wang, X.E.: Worse than Random? An Embarrassingly Simple Probing Evaluation of Large Multimodal Models in Medical VQA. arXiv (2024). https://doi.org/10.48550/ARXIV.2405.20421

[24] Siegel, N., Camburu, O.-M., Heess, N., Perez-Ortiz, M.: The Probabilities Also Matter: A More Faithful Metric for Faithfulness of Free-Text Explanations in Large Language Models. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 530– 546. Association for Computational Linguistics, Bangkok, Thailand (2024). https: //doi.org/10.18653/v1/2024.acl-short.49

[25] Parcalabescu, L., Frank, A.: On Measuring Faithfulness or Self-consistency of Natural Language Explanations. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 6048–6089. Association for Computational Linguistics, Bangkok, Thailand (2024). https://doi.org/10.18653/v1/2024.acl-long.329

[26] Lanham, T., Chen, A., Radhakrishnan, A., Steiner, B., Denison, C., Hernandez, D., Li, D., Durmus, E., Hubinger, E., Kernion, J., Lukosiute, K., Nguyen, K., Cheng, N., Joseph, N., Schiefer, N., Rausch, O., Larson, R., McCandlish, S., Kundu, S., Kadavath, S., Yang, S., Henighan, T., Maxwell, T., Telleen-Lawton, T., Hume, T., Hatfield-Dodds, Z., Kaplan, J., Brauner, J., Bowman, S.R., Perez, E.: Measuring Faithfulness in Chain-of-Thought Reasoning

[27] Du, Y., Wang, Y., Kong, M., Liang, T., Long, Q., Chen, B., Zhu, Q.: Confidence Calibration for Multimodal LLMs: An Empirical Study through Medical VQA. arXiv (2026). https://doi.org/10.48550/ARXIV.2606.19950

[28] Zhang, M., Huang, M., Shi, R., Guo, L., Peng, C., Yan, P., Zhou, Y., Qiu, X.: Calibrating the Confidence of Large Language Models by Eliciting Fidelity. In: Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 2959–2979. Association for Computational Linguistics, Miami, Florida, USA (2024). https://doi.org/10.18653/v1/2024.emnlp-main.173

[29] Cao, Z., Deng, Z., Ma, J., Hu, J., Ma, L.: Mammovlm: A generative large vision– language model for mammography-related diagnostic assistance. Information Fusion 118, 102998 (2025)

[30] Li, Q., Chen, X., Chen, J., Lai, Y.-H., Wang, S., Safari, M., Hu, M., Gichoya,

Table A1 Model configurations. All models use the default (latest) Hugging Face checkpoint revision, bfloat16 precision (no quantization), greedy decoding (temperature 0), and a 384-token generation budget.
<table><tr><td>Model</td><td>Hugging Face checkpoint</td><td>Domain</td><td>Params</td></tr><tr><td>MedGemma-4B</td><td>google/medgemma-4b-it</td><td>Medical</td><td>4B</td></tr><tr><td>Lingshu-7B</td><td>lingshu-medical-mllm/Lingshu-7B</td><td>Medical</td><td>7B</td></tr><tr><td>Qwen2-VL-7B</td><td>Qwen/Qwen2-VL-7B-Instruct</td><td>General</td><td>7B</td></tr><tr><td>InternVL3-8B</td><td>OpenGVLab/InternVL3-8B</td><td>General</td><td>8B</td></tr></table>

J.W., Trivedi, H.M., Yang, X.: Mammovqa: a reinforcement learning framework for clinically aligned mammographic question answering. In: Medical Imaging 2026: Computer-Aided Diagnosis, vol. 13926, pp. 49–54 (2026). SPIE

[31] Radiological Society of North America: RSNA Screening Mammography Breast Cancer Detection AI Challenge. https://www.rsna.org/rsnai/ai-image-challen ge/screening-mammography-breast-cancer-detection-ai-challenge. Accessed: 2026-09-08 (2023)

[32] Jeong, J.J., Vey, B.L., Bhimireddy, A., Kim, T., Santos, T., Correa, R., Dutt, R., Mosunjac, M., Oprea-Ilies, G., Smith, G., Woo, M., McAdams, C.R., Newell, M.S., Banerjee, I., Gichoya, J., Trivedi, H.: The EMory BrEast imaging dataset (EMBED): A racially diverse, granular dataset of 3.4 million screening and diagnostic mammographic images. Radiology: Artificial Intelligence 5(1), 220047 (2023) https://doi.org/10.1148/ryai.220047

[33] Yi, S., Shang, L.: Corgi: Verified chain-of-thought reasoning with post-hoc visual grounding. arXiv preprint arXiv:2508.00378 (2025)

[34] Asadi, M., O’Sullivan, J.W., Cao, F., Nedaee, T., Rajabalifardi, K., Li, F.-F., Adeli, E., Ashley, E.: Mirage: The illusion of visual understanding. arXiv preprint arXiv:2603.21687 (2026)

## Appendix A Model Configurations

Table A1 lists the checkpoint and inference settings for each evaluated model.

## Appendix B Dataset Descriptions

Table B2 reports the item count and majority-class prior for each cohort-task pair evaluated in this work. The majority-class prior is computed as the accuracy of a trivial classifier that always predicts the single most frequent label in that cohorttask’s label distribution over the benchmark items, and serves as the reference point for VLM performance. EMBED is evaluated only on the BI-RADS assessment task.

Full per-category label distributions underlying each majority-class prior, rather than just the majority share, are available on request but omitted here for space.

Table B2 Cohort and task composition. Majority-class prior is the accuracy of always predicting the most frequent label.
<table><tr><td>Cohort</td><td>Task</td><td>N</td><td>Majority-class prior</td></tr><tr><td>RSNA</td><td>BI-RADS</td><td>5,106</td><td>59.4%</td></tr><tr><td>RSNA</td><td>Density</td><td>5,821</td><td>42.6%</td></tr><tr><td>EMBED</td><td>BI-RADS</td><td>14,303</td><td>62.1%</td></tr></table>

## Appendix C Additional RSNA Results

## C.1 Answer vs. Reasoning

Table C3 reports explicit-answer and reasoning accuracy per model and task. Because both accuracies here are computed only on the parseable subset, they are not directly comparable to the unconditioned accuracies reported in Table 1. Figure C1 gives representative explanations for the parser’s four outcomes. We use a deterministic, released rule-based matcher as the primary reasoning evaluator: it is reproducible, matches explanations against the standardized BI-RADS/ACR lexicon, and scales to ∼950k generations, whereas an LLM judge could itself exhibit the incoherence this paper studies. On 120 hand-labeled explanations the matcher matches a blinded annotator at κ = 0.78 (80.8%), comparably to an LLM judge (gpt-oss-20b) blinded to the ground truth and the matcher (Table C4). At scale, the judge matches the matcher on ≈70.6% of ∼8,000 explanations (58–94% per cell; Table C5). Disagreements fall in both directions, mostly on rationales that describe a finding without naming a category, so the matcher’s rates are a conservative lower bound.

## C.2 NOTA selection and confidence across all conditions

Figure C2 reports the NOTA-selection rate and the confidence placed on NOTA across all six conditions. Only InternVL and Qwen raise their NOTA rate substantially under the visual perturbations (75–100% on BI-RADS); Lingshu abstains only on the unrelated photo, not on blank or no-image; and MedGemma barely abstains at all (0% on BI-RADS junk, ≤ 25% on density). Most models thus do not treat the loss of a valid image as grounds to abstain. When NOTA is chosen its confidence is near 100 for Qwen in every condition, but InternVL and Lingshu sometimes abstain at very low confidence (InternVL ≈ 0 on no-image, Lingshu ≈ 0.5 on the unrelated photo), a fallback rather than a deliberate call.

Table C3 Explicit answer vs. reasoning accuracy on the parsable fraction of RSNA results. Both accuracies are computed only on items whose explanation yields a readable category, so the comparison within each row is paired. Parse rate varies across model×task cells, from 66% (Qwen2-VL-7B, BI-RADS) to ∼100% (MedGemma-4B, BI-RADS); low-parse cells rest on a correspondingly smaller subset. The gap ∆ = (reasoning − answer) is small and mixed in sign (−6.7 to +4.3): the free-text rationale is no more accurate than the explicitly stated answer.
<table><tr><td>Model</td><td>Task</td><td>Parsed %</td><td>Answer acc. %</td><td>Reasoning acc. %</td><td>∆(pp)</td></tr><tr><td rowspan="2">MedGemma-4B</td><td>BI-RADS</td><td>99.8</td><td>42.4</td><td>35.8</td><td>-6.7</td></tr><tr><td>Density</td><td>94.1</td><td>33.8</td><td>33.5</td><td>-0.3</td></tr><tr><td rowspan="2">Lingshu-7B</td><td>BI-RADS</td><td>97.6</td><td>12.3</td><td>16.6</td><td>+4.3</td></tr><tr><td>Density</td><td>92.2</td><td>48.4</td><td>48.3</td><td>+0.0</td></tr><tr><td rowspan="2">Qwen2-VL-7B</td><td>BI-RADS</td><td>66.3</td><td>3.7</td><td>7.1</td><td>+3.4</td></tr><tr><td>Density</td><td>75.9</td><td>32.5</td><td>32.9</td><td>+0.5</td></tr><tr><td rowspan="2">InternVL3-8B</td><td>BI-RADS</td><td>96.4</td><td>7.9</td><td>11.4</td><td>+3.5</td></tr><tr><td>Density</td><td>97.5</td><td>42.5</td><td>42.5</td><td>+0.0</td></tr></table>

![](images/d467c51e340c2233ce5c4f4de4a86fd5edd3f8948b65136fcc693c26d810ae87.jpg)  
Fig. C1 Example explanations for the four reasoning-parser outcomes: BI-RADS label, ACR-density label, “None of the above,” and unparsable. Highlighted text is the phrase the parser matches.

Table C4 Reasoning-extraction agreement on 120 hand-labeled RSNA explanations.
<table><tr><td>Comparison</td><td>Agreement %</td><td>κ</td></tr><tr><td>Matcher vs. human</td><td>80.8</td><td>0.777</td></tr><tr><td>Judge vs. human</td><td>82.5</td><td>0.798</td></tr><tr><td>Matcher vs. judge</td><td>78.3</td><td>0.753</td></tr></table>

Table C5 Matcher vs. LLM-judge agreement at scale (1,000 explanations per task, judge blinded). Overall ≈70.6%.
<table><tr><td>Model</td><td>Task</td><td>Agreement %</td></tr><tr><td rowspan="2">MedGemma-4B</td><td>BI-RADS</td><td>93.5</td></tr><tr><td>Density</td><td>74.5</td></tr><tr><td rowspan="2">Lingshu-7B</td><td>BI-RADS</td><td>58.0</td></tr><tr><td>Density</td><td>66.7</td></tr><tr><td rowspan="2">Qwen2-VL-7B</td><td>BI-RADS</td><td>84.9</td></tr><tr><td>Density</td><td>59.1</td></tr><tr><td rowspan="2">InternVL3-8B</td><td>BI-RADS</td><td>60.9</td></tr><tr><td>Density</td><td>67.0</td></tr></table>

![](images/cecb499946279d60ac3bbbe10e50234ba27656f6c6c83467bef09c4c36dd7f2d.jpg)  
Fig. C2 NOTA-selection rate (left) and mean confidence when NOTA is chosen (right), for BI-RADS (top) and Density/ACR (bottom), across all six conditions. Shaded band marks the three conditions where NOTA is the correct response.

## C.3 Question Rewording

Rewording the question (restricted to items whose wording actually changed) barely moves accuracy: the net change is small for every model, from −5.4 points (MedGemma, density) to +1.8 (Qwen, density). Answers still change more often than this net efect suggests, though far less than under option shufling: rewording changes

5–22% of BI-RADS answers and 3–38% of density answers (Figure C3). Most of these changes are from one wrong answer to another (for Qwen, much of the BI-RADS change (15.8 of 21.8%) is a switch to NOTA) so rewording alters answers without making them more correct. InternVL is the most sensitive to rewording (a 37.6% change rate on density), yet even there accuracy rises only +1.3 points. Because rewording preserves the meaning of the question, a model that read the image should give the same answer. Answers changing at all, shows the fragility lies in the surface form of the task.

![](images/2b8c7aad81d4a4a5a6ef7c84bd908ae238e974b1e1c7ffe54f536c9f6a4bdc07.jpg)  
Fig. C3 Efect of rewording the question, on items whose wording actually changed. Each bar is the share of items whose answer changed, split by efect on accuracy. The cross-hatched portions are the share that changed to a NOTA. Numbers above the bars give the net accuracy change (percentage points).

## C.4 Perturbation Sensitivity Details

For each model×task×perturbation cell, we test whether a perturbation induces an asymmetric change in correctness using McNemar’s exact test on the paired outcomes (correct/incorrect before vs. after perturbation). Let $n _ { C  W }$ and $n _ { W  C }$ denote the number of items that flip from correct to wrong and from wrong to correct, respectively, under the perturbation. Items whose correctness is unchanged (correct→correct, wrong→wrong) are uninformative for this test and excluded. Under the null hypothesis that a perturbation is equally likely to help or hurt an initially-discordant item, $n _ { C \to W } \sim \mathrm { { B i n o m i a l } } ( n _ { C \to W } + n _ { W \to C } , 0 . 5 )$ ; we report the exact two-sided binomial pvalue and flag $p < 0 . 0 5$ with an asterisk in Table C6. A significant result indicates the perturbation reliably pushes answers in one direction (typically correct→wrong), even when the net accuracy change is small; both flip directions can be individually large and nearly cancel (Qwen on density: net +1.8 yet significant). Option shufling produces a significant asymmetry in seven of the eight cells, often with a large net drop (down to −24.8 for MedGemma on BI-RADS).

Table C6 McNemar test results for the two elicitation perturbations (reworded question, shufled options), per model and task, on the explicitly stated answer. Density refers to ACR breast density. C→W and W→C are item counts; Net ∆ is the percentage-point change in accuracy. \*: p < 0.05.
<table><tr><td>Model</td><td>Task</td><td>Pert.</td><td>Orig.%</td><td>Pert.%</td><td>C→W</td><td>W→C</td><td>Net ∆</td></tr><tr><td>MedGemma-4B</td><td>BI-RADS</td><td>Question</td><td>46.0</td><td>45.6</td><td>88</td><td>75</td><td>-0.3</td></tr><tr><td></td><td>BI-RADS</td><td>Options</td><td>42.5</td><td>17.7</td><td>1685</td><td>431</td><td>-24.8*</td></tr><tr><td></td><td>Density</td><td>Question</td><td>31.5</td><td>26.0</td><td>544</td><td>303</td><td>-5.4*</td></tr><tr><td></td><td>Density</td><td>Options</td><td>33.9</td><td>21.7</td><td>1406</td><td>697</td><td>-12.2*</td></tr><tr><td>Lingshu-7B</td><td>BI-RADS</td><td>Question</td><td>10.5</td><td>10.0</td><td>52</td><td>36</td><td>-0.4</td></tr><tr><td></td><td>BI-RADS</td><td>Options</td><td>14.3</td><td>5.5</td><td>433</td><td>68</td><td>-8.9*</td></tr><tr><td></td><td>Density</td><td>Question</td><td>50.2</td><td>48.6</td><td>92</td><td>24</td><td>-1.6*</td></tr><tr><td></td><td>Density</td><td>Options</td><td>48.5</td><td>39.8</td><td>1169</td><td>777</td><td>-8.7*</td></tr><tr><td>Qwen2-VL-7B</td><td>BI-RADS</td><td>Question</td><td>3.0</td><td>1.4</td><td>60</td><td>1</td><td>-1.6*</td></tr><tr><td></td><td>BI-RADS</td><td>Options</td><td>2.4</td><td>1.6</td><td>103</td><td>68</td><td>-0.8*</td></tr><tr><td></td><td>Density</td><td>Question</td><td>32.4</td><td>34.2</td><td>156</td><td>235</td><td>+1.8*</td></tr><tr><td></td><td>Density</td><td>Options</td><td>25.4</td><td>15.9</td><td>760</td><td>263</td><td>-9.5*</td></tr><tr><td>InternVL3-8B</td><td>BI-RADS</td><td>Question</td><td>8.3</td><td>8.3</td><td>88</td><td>89</td><td>+0.0</td></tr><tr><td></td><td>BI-RADS</td><td>Options</td><td>7.7</td><td>21.0</td><td>81</td><td>701</td><td>+13.4*</td></tr><tr><td></td><td>Density</td><td>Question</td><td>44.1</td><td>45.4</td><td>675</td><td>731</td><td>+1.3</td></tr><tr><td></td><td>Density</td><td>Options</td><td>41.8</td><td>40.8</td><td>813</td><td>763</td><td>-1.0</td></tr></table>

## Appendix D Additional EMBED Results

We repeat the four main-text analyses on EMBED BI-RADS (up to N=14,303 items; majority-class prior 62.1%, “BI-RADS 1”). EMBED is exam-level: each instance is a full mammography exam of 1–15 views rather than a single image, and the model sees all views at once. The visual perturbations preserve this structure—each exam’s views are replaced by the same number of blank images, or by the same number of distinct unrelated photos—so an exam of n real views becomes n blank (or n unrelated) images. The trends broadly follow RSNA and are often more extreme, driven by EMBED’s stronger class imbalance and exam-level setting.

## D.1 Answer–reasoning inconsistency.

Answer accuracy stays far below the class prior (4.8–36.8% vs. 62.1%; Table D7). For MedGemma, Qwen and InternVL the answer and its reasoning are largely coherent and jointly wrong: bucket D dominates (58–92%) and the disconnect bucket C is small (≤ 5.4%; Fig. D4). Lingshu is the exception: its explanation names the correct category far more often than its stated answer (57.6% vs. 10.3%), a large disconnect (C=54.6%), resulting in an output collapse in which it commits to a fixed wrong category regardless of a correct rationale.

## D.2 Confidence–evidence inconsistency.

MedGemma’s chosen-answer confidence rises as the image is removed (57 on the real image → 95 on a blank or absent image) while its confidence on the correct label falls to $\leq 1 0 ;$ InternVL likewise grows more confident on junk (18.5 → 57.6). Lingshu and Qwen place near-zero confidence on both the chosen and the correct option at the real image. As in the main text, the declared confidence does not track whether an answerable image is present (Fig. D5).

Table D7 Explicit answer vs. reasoning accuracy on the parsable fraction of EMBED BI-RADS, both computed on items with parseable reasoning (paired). Parsed % is relative to the ∼14.3k item set; ∆ = reasoning − answer.
<table><tr><td>Model</td><td>Parsed %</td><td>Answer acc. %</td><td>Reasoning acc. %</td><td>∆ (pp)</td></tr><tr><td>MedGemma-4B</td><td>96.3</td><td>36.8</td><td>20.9</td><td>-15.9</td></tr><tr><td>Lingshu-7B</td><td>99.8</td><td>10.3</td><td>57.6</td><td>+47.3</td></tr><tr><td>Qwen2-VL-7B</td><td>91.4</td><td>15.4</td><td>2.3</td><td>-13.1</td></tr><tr><td>InternVL3-8B</td><td>82.4</td><td>4.8</td><td>4.7</td><td>-0.1</td></tr></table>

![](images/3114b0b912c73acffc0c34c6fba978ae14542ccbce115a6d9bb90c412a0ef874.jpg)  
Fig. D4 Answer × reasoning coherence (EMBED, BI-RADS). Orange = disconnect (answer wrong, reasoning correct).

## D.3 NOTA selection.

At the real image the models rarely abstain $( \leq 0 . 7 \% )$ . Under the visual perturbations only InternVL raises its NOTA rate substantially (0.1 → 75.8% on average, reaching 100% on no-image); Qwen abstains only on the unrelated photo (67.9%, versus 0% on blank and no-image), while MedGemma and Lingshu almost never abstain even on junk (≤ 3.7%). As on RSNA, abstention does not reliably track the loss of a valid image (Fig. D6).

![](images/1a32a004d55570401be730a177a3f84f9204aa09bf4ccbbaba274c9afd93dfa1.jpg)

Fig. D5 Mean confidence on the condition-correct answer $c _ { y }$ (solid; original category on real images, NOTA on the visual perturbations) vs. the explicitly chosen answer $c _ { \hat { y } }$ (dashed), across the six EMBED (BI-RADS) conditions.  
![](images/3e0865e28d08fc8af9b743decd1d7e6a80deb9edb77ecfdc4fbc32712d85ed86.jpg)  
Fig. D6 NOTA-selection rate on the real image vs. junk images (EMBED, BI-RADS). Evidence driven NOTA should rise under junk; most models do not.

## D.4 Perturbation sensitivity.

Rewording the question changes few answers (5–20%, highest for InternVL) with little net efect $( | \Delta | \leq 2 . 3 ~ \mathrm { p p } )$ . Shufling the options and adding fillers changes 48–92% of

![](images/803a60d384cd184fbc73fe3fd4bbc4be608bd296e43b0245ee4745ec335436d9.jpg)  
Fig. D7 Perturbation sensitivity (EMBED, BI-RADS): question wording (left) vs. option shufling + fillers (right). Option shufling flips most answers without raising accuracy.

answers, predominantly from one wrong answer to another, often with a large net accuracy drop (MedGemma −21.9, Qwen −11.6 pp; Fig. D7), the same option-order fragility seen on RSNA (Section 4.4).