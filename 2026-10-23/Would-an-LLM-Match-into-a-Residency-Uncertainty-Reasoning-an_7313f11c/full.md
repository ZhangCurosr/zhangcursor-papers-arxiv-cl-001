# Would an LLM Match into a Residency? Uncertainty, Reasoning and Image Use of Open Language Models on the Spanish MIR Examination

Miguel Fernandez-de-Retana <sup>a†</sup>, Unai Zulaika <sup>b</sup>, Ruben S´ anchez-Corcuera´ <sup>c</sup>,

Aritz Bilbao-Jayo <sup>d</sup> and Aitor Almeida <sup>e</sup>

Faculty of Engineering, University of Deusto (Bilbao, Spain)

<sup>†</sup>Corresponding author: m.fernandezderetana@deusto.es

Keywords: Clinical Reasoning, Large Language Models, Medical Question Answering, Open-Weight Models, Selective Prediction, Uncertainty Quantification, Vision-Language Models.

Abstract:

Accuracy on medical examinations is common evidence offered for the clinical use of language models, yet a health system that intends to run an open model on its own hardware needs to know considerably more: whether the model recognizes when it is likely to be wrong, whether medical fine-tuning improves this ability, whether its correct answers rest on sound clinical reasoning, or whether it actually uses the images it is given. We examine these questions for four open models of 8 to 27 billion parameters on the Spanish MIR residency examination, scoring 1,552 questions from eight editions under the exam’s own rules. Answering every question, the three larger models would rank among roughly the first 600 to 3,000 candidates, with 41 to 43 specialties still open to them. Abstention, however, never pays, because even their least confident answers are right more often than the 25% break-even. Medical fine-tuning reduced overconfidence at the cost of accuracy, without improving how well confidence separates right from wrong answers. Moreover, 13–38% of correct answers rejected a distractor for a clinically unsound reason that confidence did not reflect, and removing or swapping the provided contextual images left accuracy unchanged.

## 1 INTRODUCTION

Large language models (LLMs) now pass medical licensing examinations, often by a wide margin (Kung et al., 2023; Singhal et al., 2023; Nori et al., 2023; Kasai et al., 2023), and the Spanish residency examination is no exception (Guillen-Grima et al., 2023). Most of this evidence, however, comes from large proprietary models accessed through closed APIs, whose weights are not available and whose prompting, decoding or training data remain outside the user’s control. A hospital that cannot send patient data beyond its own infrastructure must thus instead turn to open models that can run on its own hardware, which also gives it the ability to inspect, control and audit them (Riedemann et al., 2024; Thirunavukarasu et al., 2023). At the scale that fits on a single workstation GPU of reasonable cost and memory, i.e., models of a few tens of billions of parameters, accuracy on medical examinations is far from a solved problem, and the question is whether the performance of these models is good enough to support clinical decision-making or education purposes.

Yet, a headline accuracy figure answers none of the questions that a health system would need to ask before deploying such a model for clinical decision support or medical education (Hager et al., 2024). It says nothing about whether the model knows when it is wrong, whether medical fine-tuning improves that self-knowledge or merely changes accuracy, whether a correct answer is reached through reasoning that a physician would consider clinically sound, or whether the model actually looks at the image when a case comes with one. A model can thus be confidently wrong, or right for the wrong reasons, without either showing in its score.

The Spanish Medico Interno Residente´ (MIR) examination makes it possible to address all four questions under realistic conditions. Because it penalizes wrong answers, a candidate’s score depends on when they choose to answer or abstain; because it allocates training locations and specialties by rank, a score translates directly into a specialty and hospital where different candidates would be placed under various degrees of competitiveness; and because physicians publish per-option commentaries on many of its questions, and about one question in eight includes a clinical image, it also lends itself to an analysis of reasoning and image use. We evaluate four open models, namely Qwen3-8B, Qwen3-14B, Gemma-3-27B and MedGemma-27B, the last two of which share a base model and differ only in medical tuning, and address the following research questions:

RQ1. Scored like a real candidate, where does each model land in the national ranking?

RQ2. Does medical fine-tuning improve a model’s judgement of its own uncertainty, or only its accuracy/practical performance?

RQ3. When a model gets the answer right, does it reject the wrong options for reasons that a physician would consider clinically sound?

RQ4. Do vision language models actually use the clinical images they are given, or do they answer instead from the text alone?

Our results show that a score good enough to win a competitive residency hides substantial weaknesses. The three larger models would rank among the first few thousand candidates and could choose almost any specialty, yet none of the four can profit from the exam’s penalty by abstaining, since even their least confident answers are right more often than the 25% break-even. Medical tuning makes a model less overconfident but also less accurate, and no better at telling its right answers from wrong ones. Between 13 and 38% of correct answers dismiss at least one wrong option for a clinically unsound reason, which the model’s confidence does not reveal, and withholding or swapping the clinical image leaves accuracy unchanged.

In summary, our contributions are: (i) a corpus of 2,595 questions extracted from the last twelve official MIR exams, from 2014 to 2025, with answer keys, physician commentary, clinical plates and the published score-to-rank data of the 2023 and 2024 calls; (ii) an evaluation under the exam’s own scoring and allocation rules of four open models of 8 to 27 billion parameters, with and without medical tuning; (iii) a controlled comparison of the effect of medical tuning on calibration, with an exam sat after the model training cutoff as a contamination control; and (iv) evidence on model reasoning and image use in questions with physician commentary and clinical plates.

## 2 RELATED WORK

LLMs on Medical Examinations. Since ChatGPT approached the pass mark of the USMLE (Kung et al., 2023), medical examinations have become the standard showcase for clinical LLMs. MedQA (Jin et al., 2021) and its successors track accuracy (Singhal et al., 2023; Singhal et al., 2025; Nori et al., 2023), and national examinations such as those of Japan (Kasai et al., 2023) and Spain (Guillen-Grima et al., 2023) have since been added. The MIR in particular is available as a benchmark through HEAD-QA (Vilares and Gomez-Rodr´ ´ıguez, 2019; Correa-Guillen et al.,´ 2026), and rheumatologists have rated the reasoning of LLMs on its rheumatology questions (Madrid-Garc´ıa et al., 2023). These studies, however, report accuracy or at most the raw mark, and we are not aware of any that scores a model under the MIR’s penalty and allocation rules, or that separates what a model gains by answering from what it could gain by abstaining.

Explanations and Reasoning. A correct answer does not guarantee correct reasoning. GPT-4V was found to give flawed rationales on over a third of the image challenges it answered correctly (Jin et al., 2024), and the explanations a model produces do not necessarily reflect the process that actually led to its answer (Turpin et al., 2023). To this end, CasiMedicos provides physician-written explanations of MIR questions that cover the wrong options as well as the right one (Alonso et al., 2024; Goenaga et al., 2024), which makes it possible to ask whether a model rejects each distractor for a clinically valid reason. We grade that reasoning against the physician’s text with an LLM acting as a judge (Zheng et al., 2023).

Calibration and Abstention. For a model’s confidence to support deferral, it must be both calibrated (Guo et al., 2017; Naeini et al., 2015) and able to separate right from wrong answers (Hendrycks and Gimpel, 2017), a trade-off between coverage and risk formalized by selective prediction (Geifman and El-Yaniv, 2017). LLMs do carry some knowledge of their own correctness (Kadavath et al., 2022), although the confidence they verbalize tends to be overinflated and sycophantic (Xiong et al., 2024). It has recently been argued that benchmarks which do not penalize wrong answers reward guessing, and that penalized scoring would instead reward honest abstention (Kalai et al., 2025). The MIR provides such a benchmark, with a penalty that makes it possible to ask whether a model can identify the questions it would answer worse than chance, and whether medical fine-tuning improves that ability.

Images and Contamination. Multimodal models have been shown to score well on medical visual questions while ignoring or misreading the image (Yan et al., 2025; Jin et al., 2024). Examination results are also exposed to training data contamination (Sainz et al., 2023), for which comparing performance before and after a model’s training cutoff is a standard control (Roberts et al., 2024).

## 3 THE MIR EXAMINATION

## 3.1 Format and Scoring

The MIR is a single, national, multiple-choice examination sat once a year by every graduating physician who seeks a specialty training post in Spain. Following the Ministry of Health, we name each exam by the year of its call, even though it is sat early in the following year, so the 2025 exam, for instance, took place on 24 January 2026. Papers contained 225 scored questions up to the 2018 exam, 175 in 2019 and 2020 and 200 from 2021 onwards, in each case with 10 reserve questions that replace any question annulled by the examining commission. Questions had five options up to the 2014 exam and four thereafter. According to the official scoring rules, a correct answer scores +3, a wrong answer −1 and a blank 0, so that the total mark S is

$$
S = 3 c - w ,\tag{1}
$$

where c and w are the numbers of correct and incorrect answers, respectively. Published results divide this mark by three, giving the net score $S / 3 = c - w / 3 ,$ i.e., the number of correct answers net of the penalty for wrong ones. We report net scores throughout.

Since answering a question that one gets right with probability p is worth $3 p - ( 1 - p ) = 4 p - 1$ points, against zero for a blank, it follows that

$$
\mathrm { A n s w e r } \iff p > { \frac { 1 } { 4 } } ,\tag{2}
$$

regardless of the number of options. With four options, a random guess therefore has an expected value of exactly zero, and a blank is better than a guess that is right less than 25% of the time. The relevant question for a model is thus not whether it abstains often enough, but whether it can identify those questions.

## 3.2 From Score to Training Post

Candidates are ranked by a total mark that weighs the examination at 90% and the academic record at 10%, with the resulting rank determining the order in which they choose a specialty and a hospital in the national territory, so that each specialty’s cut-off is the rank of the candidate who took its last place. Since the Ministry of Health stopped publishing per-candidate result lists after the 2021 exam, the only public information consists of the per-specialty summaries that the trade press compiles from the allocation (ConSalud, 2024; ConSalud, 2025). For each specialty, these report the mean net score and mean rank of its entrants and, for the 2023 exam, the net score and rank of its last entrant, together with the cut-off and the hospital where the last place went.

## 4 DATA

Questions. We parsed the official Version 0 booklets of the last twelve MIR examinations, from 2014 to 2025, containing 2,595 questions including 120 reserves (Table 1). Answer keys were taken from the answer tables published by CasiMedicos and from HEAD-QA v2 (Correa-Guillen et al., 2026´ ), matched on question text. Where both sources key the same question they agree on 730 of 773 cases, and 30 of the 43 disagreements fall in the 2017 exam: the four models’ unanimous answers agree with it on only 36% of questions, against 85–93% in every other year. We therefore start the analysis of RQ1, RQ2 and RQ4 at the 2018 exam, which leaves only four-option papers. Annulled questions are excluded throughout.

Physician Commentary. CasiMedicos also publishes clinician explanations of questions (Alonso et al., 2024; Goenaga et al., 2024), and for some of them, the commentary discusses the wrong options one by one. We use the 123 official questions from the 2015 and 2017–2021 exams whose commentary covers at least two options, since only these can reveal whether a distractor was rejected for the right reason.

Clinical Plates. We extracted the clinical plates from the image annexes of the 2019–2025 exams, the only ones available, obtaining 174 questions with a plate, of which 171 are scorable.

Specialty Labels. CasiMedicos further assigns each commented question to a specialty, with 619 of the 1,552 scored questions from 2018–2025 (40%) carrying such a label.

Rank Data. We built a score-to-rank curve for the 2023 exam (88 points, covering mean and last entrants) and for the 2024 exam (45 points, mean entrants only), together with the cut-offs of their 45 and 44 specialties. The 2025 exam has published cut-offs but no scores yet, and for the 2022 exam, we found cut-offs but no per-specialty scores.

Table 1: Question sets used by each research question.
<table><tr><td>Set</td><td>Questions</td><td>Use</td></tr><tr><td>Official papers, 2014–2025</td><td>2,595</td><td>Corpus</td></tr><tr><td>Scored, 2018–2025</td><td>1,552</td><td>RQi, RQ2</td></tr><tr><td>With commentary on</td><td>123</td><td>RQ3</td></tr><tr><td>two or more options With a scorable image plate</td><td>171</td><td>RQ4</td></tr><tr><td>Rank-curve points, 2023 / 2024</td><td>88 / 45</td><td>RQ1</td></tr><tr><td>Specialty cut-offs, 2023 / 2024</td><td>45 / 44</td><td>RQ1</td></tr></table>

Table 2: Evaluated models and the judge.
<table><tr><td>Model</td><td>Params</td><td>Precision</td><td>Images</td></tr><tr><td>Qwen3-8B</td><td>8.2B</td><td>16-bit</td><td>No</td></tr><tr><td>Qwen3-14B</td><td>14.8B</td><td>16-bit</td><td>No</td></tr><tr><td>Gemma-3-27B</td><td>27.4B</td><td>8-bit</td><td>Yes</td></tr><tr><td>MedGemma-27B</td><td>27.4B</td><td>8-bit</td><td>Yes</td></tr><tr><td>Llama-3.3-70B (Judge)</td><td>70.6B</td><td>4-bit</td><td>No</td></tr></table>

## 5 METHOD

Figure 1 summarizes the study design. Pass A answers every scored question of the 2018–2025 exams in label mode, and serves RQ1 and RQ2. Pass B asks for a reason per option on the questions with physician commentary (RQ3). Pass C repeats the image questions with the plate shown, withheld or swapped (RQ4). The four models are evaluated in parallel on each pass, and all runs are deterministic.

## 5.1 Models

We evaluate Qwen3-8B and Qwen3-14B (Qwen Team, 2025) together with Gemma-3-27B and MedGemma-27B (Gemma Team, 2025; Sellergren et al., 2025), whose sizes and precisions are listed in Table 2. All four were released in 2025, before the 2025 exam was sat. Since MedGemma-27B is built on Gemma-3-27B through medical tuning, the pair isolates the effect of that tuning at a fixed scale and architecture, and as both accept images, they are also the two models evaluated in Pass C. All experiments ran on a single 48GB NVIDIA Quadro RTX 8000, the kind of hardware a hospital can realistically own and operate. Qwen3’s thinking mode was disabled so that all four models answer directly.

## 5.2 Answering Protocol

All prompts are written in Spanish, the language of the examination. The system prompt introduces the model as a MIR candidate and asks it to answer in the language of the question, while the user turn contains the stem and the numbered options. The model is not told about the scoring rule, since abstention is applied afterwards on the basis of its confidence (Section 5.3).

Label Mode. In this mode, used for RQ1, RQ2 and RQ4, the model is asked only for the number of the correct option. In a single forward pass we read the probability $q _ { k }$ that the model assigns to each option number k as the first token of its reply and renormalize it over the options, $\begin{array} { r } { \hat { p } _ { k } = q _ { k } / \sum _ { j } q _ { j } . } \end{array}$ . The answer is then arg max<sub>k</sub> $\hat { p } _ { k }$ and the confidence max<sub>k</sub> $\hat { p } _ { k }$ , thus derived from the model’s distribution over the options.

Generate Mode. In this mode, used for RQ3, the model is asked for a reply stating, for every option, whether it is the one to mark together with a one sentence clinical reason, followed by the final answer and a stated confidence. Asking whether an option is “the one to mark” rather than whether it is “true” keeps the verdict unambiguous on negatively phrased stems (e.g., “which of the following is FALSE”), which make up about a fifth of the MIR.

## 5.3 Scoring and Placement (RQ1)

Each full-paper run is scored according to (1) under three answering policies: always hands in every answer, oracle hands in only the correct ones and thus sets the ceiling for a perfectly self-aware model, and calibrated applies the break-even rule of (2). The latter requires an estimate of $p$ that the confidence cannot provide, because the largest of four probabilities summing to one is at least 1/4, so the rule “answer if max<sub>k</sub> $\hat { p } _ { k } \ge 1 / 4 ^ { \cdots }$ would never leave a blank. We therefore estimate $p$ from the model’s record on its other seven examinations, sorting their answers by confidence into ten equal-sized groups and taking as P<sup>ˆ</sup>(correct) the accuracy of the group into which a new answer’s confidence falls, so that no examination’s policy is fitted on its own answers. From these policies we derive the abstention gain, calibrated minus always, and the headroom, oracle minus calibrated, that is, the part of the net score that only a more accurate model could recover.

To place a model in the national ranking, we fit a score-to-rank curve for each exam through the published points and interpolate linearly between the resulting knots. A specialty is considered open at a given rank if its last place went to that rank or a later one, so the most competitive open specialty is the best choice still available, and the hospital that took its last place is where it remained available at the end. Mapping the 95% interval of the net score through the curve then gives the corresponding range of ranks.

![](images/07a5934c307c1c7daa2fe2c9d464e3944983262c2237a5929b7aea51ce1dd988.jpg)  
Figure 1: Study Design. Pass A answers every scored question of the 2018–2025 exams and serves RQ1 and RQ2. Pass B asks for a reason per option on the questions with physician commentary (RQ3). Pass C repeats the image questions with the plate shown, withheld or swapped (RQ4).

## 5.4 Calibration (RQ2)

For each model, we report accuracy, mean confidence, expected calibration error (ECE, with ten equal bins) (Naeini et al., 2015), the multi-class Brier score over the option distribution (Brier, 1950), and the area under the ROC curve (AUROC) with which confidence separates right from wrong answers (Hendrycks and Gimpel, 2017). Whereas ECE measures whether confidence matches accuracy on average, AUROC measures whether it ranks answers correctly. The main contrast is between MedGemma-27B and Gemma-3- 27B on the same questions. As a contamination control, we also report every measure separately for the pre-release examinations, whose questions have circulated publicly for years, and for the 2025 exam, which was sat after all four models had been released.

## 5.5 Rationale Grading (RQ3)

Each model performs one generate mode run over the 123 questions with commentaries on two or more options. For every option that has both a physician span and a model reason, an external judge, i.e., a Llama-3.3-70B-Instruct (Grattafiori et al., 2024) in 4-bit precision, is shown the question, the options, the key, the model’s verdict on the option, the physician’s text for it and the model’s reason, and labels the reason as same if it gives the physician’s key clinical reason, valid if it gives a different but clinically correct and sufficient one, or wrong if it is clinically wrong, unsupported, irrelevant or too vague, or argues for the wrong verdict. Because the reasons are graded against the physician’s commentary, RQ3 is limited to the 108 questions on which the physician’s answer matches the exam key. Our main measure of interest, right answer, wrong reason, is the share of correctly answered questions on which at least one distractor was rejected for a reason labelled wrong, with which we compare the label mode confidence of these answers with that of correct answers whose distractor reasons were all sound.

## 5.6 Image Conditions (RQ4)

The two vision models answer the 171 image questions in label mode under three conditions, i.e., with the question’s own plate, with no plate, and with the plate of a different question. We report accuracy per condition, the paired difference between the real plate and each alternative, the share of answers that change, and the share of questions answered correctly in all three conditions, for which the text alone is evidently sufficient.

## 5.7 Statistical Analysis

All confidence intervals are 95% intervals computed over questions, so they express how much a measure would change had a different set of questions been asked. Since every answer is deterministic, this is the only source of variation. For proportions, namely the accuracies and the RQ3 rate, we use Wilson intervals (Wilson, 1927). For the measures that are not simple proportions (e.g., net score, ECE, Brier score and AUROC), and for the difference between two values of any measure, we use a percentile bootstrap with 2,000 resamples of the questions (Efron and Tibshirani, 1993). Two kinds of difference arise. When both values come from the same questions, as in MedGemma-27B minus Gemma-3-27B and in the real plate minus each alternative, the questions are resampled once and both values recomputed on them (a paired bootstrap). When they come from different questions, as in the 2025 exam minus the earlier ones and in the RQ3 confidence gap, the two sets are resampled independently.

As for significance tests, two models or two image conditions answering the same questions are compared with McNemar’s exact test (McNemar, 1947), which considers only the questions on which the two disagree. The RQ3 rates, in contrast, are computed over each model’s own correct answers and therefore over different questions, so we compare them with Fisher’s exact test. Because this comparison involves all six pairs of the four models, its six p-values are corrected with Holm’s adjustment (Holm, 1979). No other test in the paper is adjusted.

## 6 RESULTS

## 6.1 RQ1: Scored Like a Real Candidate

When answering every question, Gemma-3-27B and Qwen3-14B obtain an average net score per examination of 131.3 and 131.0, respectively, followed by MedGemma-27B with 125.5 and Qwen3-8B with 95.8, with a 95% interval of roughly ±16 on any single examination. Since the eight examinations differ in length, from 172 to 221 scored questions, Table 3 reports the net score per 100 questions over all 1,552 of them, which is 67.7, 67.5, 64.7 and 49.4 for the same four models. Furthermore, Figure 2 and Table 4 show how these net scores map to the 2023 and 2024 score-to-rank curves. In 2023, Gemma-3-27B’s net score of 142.7 maps to an estimated rank of about 1,628, at which 43 of the 45 specialties were still open, the most competitive being Oral and Maxillofacial Surgery. Qwen3-14B lands close behind with the same choice, whereas MedGemma-27B would reach Cardiology and Qwen3-8B, with 22 specialties left, Haematology. In 2024 the three larger models reach ranks between about 600 and 1,400, with Plastic Surgery or Ophthalmology still open. These ranges are nonetheless wide, since a single examination places a model in the 95% interval of roughly 2,000 ranks, and the 2024 curve, which is based on only 45 points, places any given score markedly higher than the 2023 curve (Section 8).

Abstention, on the other hand, never helps. The calibrated policy answers every question on every examination, so the abstention gain is zero for all 32 model-examination pairs. This is not because confidence is uninformative, since it separates right from wrong answers with an AUROC of 0.80 for the three larger models and 0.76 for Qwen3-8B (Table 5), but because even the least confident tenth of each model’s answers on its other examinations was right 29–47% of the time, above the 25% threshold of (2). Indeed, skipping the least confident tenth chosen in hindsight within each examination would have lowered the net score on 28 of the 32 pairs and gained at most 3.7 on the others. Figure 3 illustrates why: the net score rises almost monotonically as answers are added from most to least confident. The net score lost to wrong answers, 15.7–24.5 per examination (8.1–12.7 per 100 questions), is thus headroom that a more accurate model could recover.

Table 3: RQ1, pooled over the 2018–2025 exams (1,552 questions). Net score, gain and headroom per 100 questions. Gain is calibrated minus always, headroom oracle minus calibrated. Tenth: accuracy of the tenth of answers each model was least sure of.
<table><tr><td>Model</td><td>Acc.</td><td>Net</td><td>Gain</td><td>Headroom</td><td>Tenth</td></tr><tr><td>Gemma-3-27B</td><td>.758</td><td>67.7</td><td>0.0</td><td>8.1</td><td>.387</td></tr><tr><td>MedGemma-27B</td><td>.735</td><td>64.7</td><td>0.0</td><td>8.8</td><td>.368</td></tr><tr><td>Qwen3-14B</td><td>.756</td><td>67.5</td><td>0.0</td><td>8.1</td><td>.445</td></tr><tr><td>Qwen3-8B</td><td>.621</td><td>49.4</td><td>0.0</td><td>12.7</td><td>.303</td></tr></table>

## 6.2 RQ2: Medical Tuning and Uncertainty

All four models are overconfident (Figure 4), with a mean confidence of 0.91–0.99 against an accuracy of 0.62–0.76 and an ECE of 0.21–0.29 (Table 5). Below a confidence of 0.9, in fact, accuracy falls under 0.5 in all but one of the bins with at least ten answers. Medical tuning, however, changed the level of confidence. Over all eight examinations, MedGemma-27B is 2.3 points less accurate than Gemma-3-27B on the same questions (p = 0.027) and 0.047 less confident, which lowers its ECE by 0.026 [0.005, 0.045]. Its AUROC, however, is essentially unchanged, with a difference of +0.002 [−0.025, +0.028], so it is no better at telling its right answers from its wrong ones. Given that no confidence-based abstention pays on this examination (RQ1), the lower ECE brings no gain in net score, whereas the loss of accuracy costs about 3 points of net score per 100 questions.

Moreover, we find no sign of memorization (Figure 5). On the 2025 exam, which none of the models may have seen during training, all are at least as accurate as on the older exams, by 1.2 to 4.9 points (Table 5), and its mean confidence matches that on the older exams to within 0.003. If anything, the Qwen models rank their answers better on 2025, with AUROC differences of +0.075 [+0.007, +0.138] and +0.082 [+0.010, +0.143]. Likewise, the accuracy gap between Gemma and MedGemma is no smaller on 2024 (4.0 points) than on 2018–2023 (2.5 points). On the 2025 exam the two models are equally accurate while MedGemma’s ECE advantage grows to 0.059, although a single post-release exam cannot distinguish an easier one from a less memorized one.

![](images/a16a2b0a26e0f68ef1417b27304bbc62b6cb476c3f676914152cbf7b393de1b6.jpg)

![](images/5ea5a7e83ca4797510a283445641784db33aae220e3e2d020db07de9f64b6dcb.jpg)  
Figure 2: RQ1: Each Model on the Score-to-Rank Curves of the 2023 and 2024 Exams, Answering Every Question. Grey points are the published per-specialty entrants; dotted lines mark a selection of specialty cut-offs, named in grey. Bars show the 95% interval of the net score and the rank range it maps to. A solid line leads from each model to the right, where the most competitive specialty still open at its rank is named in the model’s colour.

![](images/e671b7fded9fd2a04afe7d7782af4cf16e3d1c5ecd92f522085844fdd0b2e2a3.jpg)  
Figure 3: RQ1: Net Score per 100 Questions Against the Share Answered, Most Confident First, Pooled Over 2018– 2025.

## 6.3 RQ3: Right Answers, Wrong Reasons

A correct answer frequently rests on a distractor rejected for a wrong reason (Figure 6, Table 6). This occurs on 38% [27, 51] of the correct answers of Qwen3-8B, compared with 19% [12, 29] for Qwen3- 14B, 15% [9, 25] for MedGemma-27B and 13% [7, 24] for Gemma-3-27B. After Holm’s adjustment, Qwen3-8B differs from Gemma-3-27B (p = 0.011) and from MedGemma-27B $( p = 0 . 0 2 4 )$ and falls just short of significance against Qwen3-14B (unadjusted $p = 0 . 0 1 3$ , adjusted 0.053), while no other pair differs. In particular, Gemma-3-27B and MedGemma-27B do not $( p = 0 . 8 1$ before adjustment), so medical tuning did not change how often distractors are dismissed for wrong reasons. Across all distractor reasons, the judge labels 64–74% as valid, 14–18% as the same as the physician’s and 11–21% as wrong.

![](images/44750dfd24bdc5e1b5f87d9d281d944e4d4d34baf8d18a2eaebde4c22b6277d0.jpg)  
Figure 4: RQ2: Reliability Diagram, Pooled Over the 2018–2025 Exams. Only bins with at least ten answers are drawn; most answers fall in the top bin.

Table 4: RQ1: Placement answering every question. Net score; estimated rank with the range that the 95% interval of the net score maps to (233 is the best rank the 2024 curve can give); open specialties; and the most competitive specialty still open at that rank, with the hospital that took its last place (CHU: Complejo Hospitalario Universitario; CH: Complejo Hospitalario; CAU: Complejo Asistencial Universitario; H. U.: Hospital Universitari).
<table><tr><td>Model</td><td>Score</td><td></td><td>Rank</td><td>Open Best Open Specialty</td><td>Its Last Place</td></tr><tr><td colspan="6">2023 Exam (196 questions)</td></tr><tr><td>Gemma-3-27B</td><td>142.7</td><td>1,628 (561–3,687)</td><td>43/45</td><td>Oral and Maxillofacial Surgery</td><td>CHU de A Coruña</td></tr><tr><td>Qwen3-14B</td><td>137.3</td><td>2,284 (643–4,189)</td><td>43/45</td><td>Oral and Maxillofacial Surgery</td><td>CHU de A Coruña</td></tr><tr><td>MedGemma-27B</td><td>130.7</td><td>3,029 (1,198–5,226)</td><td>41/45</td><td>Cardiology</td><td>Hospital de Mérida</td></tr><tr><td>Qwen3-8B</td><td>105.3</td><td>6,212 (4,058–7,962)</td><td>22/45</td><td>Haematology</td><td>H. Ú. Arnau de Vilanova, Lleida</td></tr><tr><td colspan="6">2024 Exam (200 questions)</td></tr><tr><td>Gemma-3-27B</td><td>130.7</td><td>626 (233–2,025)</td><td></td><td>43/44 Plastic Surgery</td><td>CHU de Cáceres</td></tr><tr><td>Qwen3-14B</td><td>129.3</td><td>719 (233–2,398)</td><td>43/44</td><td>Plastic Surgery</td><td>CHU de Cáceres</td></tr><tr><td>MedGemma-27B</td><td>120.0</td><td>1,387 (250–3,883)</td><td>42/44</td><td>Ophthalmology</td><td>CH de Mérida</td></tr><tr><td>Qwen3-8B</td><td>94.7</td><td>4,843 (2,201–7,219)</td><td>30/44</td><td>Gastroenterology</td><td>CAU de Palencia</td></tr></table>

Table 5: RQ2: Calibration before release (Pre: the 2018–2024 exams, 1,358 questions), after release (2025, 194 questions) and over both (All), with the paired difference MedGemma-27B minus Gemma-3-27B. p is McNemar’s exact test on accuracy.
<table><tr><td>Model</td><td>Exams</td><td></td><td>Accuracy Confidence</td><td></td><td>ECE</td><td>Brier</td><td>AUROC</td><td>p</td></tr><tr><td>Gemma-3-27B</td><td>Pre</td><td></td><td>.756</td><td>.988</td><td>.234 [.212, .257]</td><td>.472</td><td>.792 [.766, .818]</td><td></td></tr><tr><td></td><td>2025</td><td></td><td>.768</td><td>.990</td><td>.222[.166, .283]</td><td>.448</td><td>.854 [.790, .910]</td><td></td></tr><tr><td>MedGemma-27B</td><td>Pre</td><td></td><td>.729</td><td>.941</td><td>.214 [.194, .237]</td><td>.469</td><td>.799 [.776, .823]</td><td></td></tr><tr><td></td><td>2025</td><td></td><td>.778</td><td>.941</td><td>.163 [.117, .219]</td><td>.359</td><td>.812 [.731, .883]</td><td></td></tr><tr><td>Qwen3-14B</td><td>Pre</td><td></td><td>.752</td><td>.972</td><td>.221 [.200, .244]</td><td>.464</td><td>.790 [.765, .815]</td><td></td></tr><tr><td></td><td>2025</td><td></td><td>.789</td><td>.975</td><td>.186[.135, .240]</td><td>.369</td><td>.866 [.803, .921]</td><td></td></tr><tr><td>Qwen3-8B</td><td>Pre</td><td></td><td>.616</td><td>.913</td><td>.297 [.274, .322]</td><td>.650</td><td>.748 [.723, .775]</td><td></td></tr><tr><td></td><td>2025</td><td></td><td>.655</td><td>.915</td><td>.261 [.204, .322]</td><td>.571</td><td>.830 [.770, .884]</td><td></td></tr><tr><td>Difference</td><td>Pre</td><td>-.027 [−.049, −.006]</td><td></td><td>-.046</td><td>-.020 [−.041, +.002]</td><td>-.003</td><td>+.006 [−.024, +.036]</td><td>.013</td></tr><tr><td></td><td>2025</td><td>+.010 [−.041, +.057]</td><td></td><td>-.049</td><td>-.059 [-.105, −.011]</td><td>-.089</td><td>−.042 [−.128, +.038]</td><td>.84</td></tr><tr><td></td><td>All</td><td>-.023 [−.043, −.004]</td><td></td><td>-.047</td><td>−.026[-.045, -.005]</td><td>-.014</td><td>+.002 [−.025, +.028]</td><td>.027</td></tr></table>

Table 6: RQ3 on 108 questions. Acc.: generate mode accuracy, with label mode accuracy on the same questions in brackets. Right: correct answers with graded distractors. Gap: label mode confidence on wrong reason minus sound reason correct answers.
<table><tr><td>Model</td><td></td><td>Acc. Right Wrong Reason</td><td>Gap</td></tr><tr><td>Qwen3-8B</td><td>.56 (.60)</td><td>60</td><td> $. 3 8 \quad + . 0 2 0 \quad$ </td></tr><tr><td>Qwen3-14B</td><td>.73 (.77)</td><td>79</td><td> $. 1 9 + . 0 1 4$ </td></tr><tr><td>MedGemma-27B .67 (.71)</td><td></td><td>72</td><td> $. 1 5 \mathrm { ~ \ } + . 0 0 5$ </td></tr><tr><td>Gemma-3-27B</td><td>.62 (.77)</td><td>67</td><td> $. 1 3 \ - . 0 0 4$ </td></tr></table>

Moreover, the models’ confidence does not reflect these errors. The confidence of correct answers reached with a wrongly rejected distractor is no lower than that of correct answers with sound reasons, the difference ranging from −0.004 to +0.020 (Table 6) with only Qwen3-14B’s interval, [+0.002, +0.031], excluding zero, and in the wrong direction. The stated confidence in generate mode does not separate them either. Likewise, asking for a reason per option also changes the answers themselves. On the same 108 questions, generate mode is 4–5 points less accurate than label mode for three of the models and 15 points less accurate for Gemma-3-27B (from 0.77 to 0.62).

Table 7 illustrates the judge’s grading of distractor reasons. In the first two examples the model’s reason is clinically wrong, since pulmonary fibrosis is a classic cause of digital clubbing and diffuse idiopathic skeletal hyperostosis is diagnosed radiographically. The third example, in contrast, shows a judge error: on a “which statement is true” stem, the model rejects the statement that proteinuria above 500mg/24h is unlikely by arguing that proteinuria is in fact likely, which is also the physician’s reason, and yet the judge labels this wrong for three of the four models. The judge can thus misread a reason that negates a negatively worded option, which inflates the wrong-reason rate for every model.

## 6.4 RQ4: Use of the Clinical Image

The clinical images available along questions make no measurable difference in terms of model accuracy (Figure 7). Gemma-3-27B answers 67.8% of the 171 image questions correctly with the real plate, 69.6% with no plate and 67.3% with another question’s plate, and MedGemma-27B 62.6%, 64.9% and 63.2%, respectively. Relative to no plate, the real plate thus changes accuracy by −1.8 points [−7.0, +2.9] and −2.3 points [−7.6, +2.4] (p = 0.65 and 0.50), so it adds at most about 3 points, and it stays within 0.6 points of a mismatched plate (p = 1.0 for both). The plate is not ignored outright, since it changes 13–20% of the answers, but the gains and losses cancel each other out, and 58% (Gemma) and 56% (MedGemma) of the image questions are answered correctly in all three conditions, i.e., from the text alone.

![](images/a055f534f5b30b76ae47c1649d8c4737a1854e69e5f27b203956ec5302ceb148.jpg)

![](images/78dd54b7599b44b30e87952b4228424a3b1c4b7ad6caaaea75e0938129a5bd63.jpg)

Figure 5: RQ2: Accuracy and Expected Calibration Error by Exam Year. The 2025 exam was sat after all models released.  
![](images/6878c96ae4ab48fd08c6df76af02374d7e5778ff2120a720aba64e2998a6ccf3.jpg)

![](images/38b3c45ebed89129ab767e3b0b46e7e092e169bdf2f81770eac4ff9cdd620f5e.jpg)  
Figure 6: RQ3: Share of Correct Answers With at Least One Distractor Rejected for a Wrong Reason, With 95% Wilson Intervals (Left), and Mean Label Mode Confidence on Those Answers Against Correct Answers Whose Distractor Reasons Were All Sound (Right).

## 6.5 Accuracy by Specialty

Figure 8 breaks accuracy down for the 18 specialties with at least ten labelled questions, which together account for 589 questions. Radiology and Nuclear Medicine is the hardest specialty for every model except Qwen3-14B, with accuracies of 47– 68%, whereas the smallest model falls furthest behind in Orthopaedics and Traumatology (49% against 80–

![](images/9b02638503ad7f44c464ef21c2c72c6dbe6c14ef548a5a50104d242c41fb9a72.jpg)  
Figure 7: RQ4: Accuracy of the Vision Models on the 171 Image Questions With the Question’s Own Plate, No Plate and Another Question’s Plate, With 95% Wilson Intervals.

85%) and in Paediatrics (54% against 76–78%). Medical tuning does not produce a distinct specialty profile either, as MedGemma-27B is less accurate than Gemma-3-27B in 13 of the 18 specialties, ties in two, and is ahead only in Haematology, Neurology and Anaesthesiology, by at most 7 points. With 10 to 119 questions per specialty, however, single differences amount to a few questions only.

Table 7: Graded distractor reasons on questions the model answered correctly, translated from Spanish and abridged.
<table><tr><td>Question and Key</td><td>Option Rejected</td><td>Physician</td><td>Model Reason</td><td>Judge</td></tr><tr><td>Ex-smoker with COPD and new digital clubbing; most plausible cause? Key: Lung carcinoma (MIR 2015, No. 125)</td><td>Pulmonary fibrosis.</td><td>The other conditions can also cause it, but not so suddenly.</td><td>MedGemma-27B: &quot;Pulmonary fibrosis is not usually associated with digital clubbing.&quot;</td><td>Wrong, correctly.</td></tr><tr><td>70 y.o., a week of progressive low back pain and low-grade fever, normal radiograph; what to rule out first, and how? Key: Pyogenic spondylodiscitis, MRI (2019, No. 122).</td><td>Ankylosing vertebral hyperostosis, with a dorsolumbar radiograph.</td><td>Vertebral collapse, ankylosing hyperostosis and many metastases would be seen on the plain radiograph.</td><td>Qwen3-8B: &quot;Radiography is insufficient to diagnose ankylosing hyperostosis, which requires more specific studies.&quot;</td><td>Wrong, correctly.</td></tr><tr><td>Diabetic nephropathy, eGFR 14 ml/min; which statement is true? Key: Stage 5, consider renal replacement (2021, No. 149).</td><td>Proteinuria above 500 mg/24 h is unlikely.</td><td>These patients usually have A3 albuminuria, possibly nephrotic-range (option false).</td><td>Gemma-3-27B: &quot;Proteinuria is common in diabetic nephropathy and likely above 500 mg/24 h at such an advanced stage.&quot;</td><td>Wrong, judge</td></tr></table>

![](images/625c53641509326d19334730b837daf321bcb86e2a9045c79a0a8bfd01d56099.jpg)  
Figure 8: Accuracy by Specialty Over the 2018–2025 Exams, for the 18 Specialties With at Least Ten Labelled Questions (Question Count in Brackets). The grey line spans the four models and markers are offset so that equal accuracies stay visible.

## 7 DISCUSSION

Penalty Is Too Mild to Reward Self-Knowledge. Penalized scoring has been proposed as a remedy for benchmarks that reward guessing (Kalai et al., 2025), and the MIR illustrates the limits of this approach. With a reward a for a right answer and a penalty b for a wrong one, answering every question is optimal whenever $p > b / ( a + b )$ , i.e., 0.25 for the MIR. Models whose least confident answers are right a third of the time therefore never benefit from abstaining, however good their confidence may be, and the examination cannot distinguish a model that knows what it does not know from one that does not. A clinical deployment, in contrast, faces much steeper penalties, since a wrong recommendation that is acted upon can cost far more than a deferral. In that setting the breakeven rises towards one, and the ability of confidence to rank errors, here an AUROC of about 0.80, becomes decisive and should be reported alongside.

Medical Tuning Is Not a Safety Proxy. In the absence of a penalty that rewards self-knowledge, a hospital may be tempted to assume that a medical model is safer than its base model. In our experiments, MedGemma-27B was less overconfident than its base model, which may be taken as a sign of better selfknowledge. However, the improvement came from lowering confidence across the board rather than from a better knowledge of which answers were wrong, and it came at a cost in accuracy. A hospital choosing between a general model and its medical variant should thus test both on task and on language, rather than assume that the medical label brings better judgement.

Explanations Need to Be Checked. When a model’s explanation is shown to a clinician or a student, between 13 and 38% of its correct answers come with a distractor dismissed for a wrong reason. In medical education, where the explanation is the main object of interest, this is a serious flaw, and in clinical deployment, it is a safety concern. The model’s confidence does not reveal it, and the judge’s grading shows that the problem is not limited to a single model. Asking for explanations also changed the answers, by 15 points for one model, so an evaluation in which the model is asked to explain its answer may not reflect its performance when it is not.

Images Are Optional on This Examination. More than half of the image questions were answered correctly without any image, and neither model used the images to improve accuracy. For clinical use, this implies that the ability to accept an image is no evidence that the model ultimately looks at it, and that image dependent tasks require their own evaluation.

## 8 LIMITATIONS

Judge Validation. No physician has rated the judge’s grading, and Table 7 shows that the judge errs on reasons that negate negatively worded options. Physician ratings of a blind sample, as well as the expert ratings of LLM reasoning available for rheumatology questions (Madrid-Garc´ıa et al., 2023), would provide a more reliable estimate. In addition, RQ3 excludes the 15 questions whose physician answer differs from the exam key, and covers pre-release examinations only.

Ranks Are Estimates. Since the rank combines the examination with the academic record, a rank read off the net score assumes a typical record. Moreover, the 2024 curve rests on specialty means only and, at an equal score, yields a much better rank than the 2023 curve, so its placements should be read as optimistic.

Scope. Our study covers four open models of 8–27B parameters, a single prompt in Spanish and direct answering, so larger or reasoning models may behave differently. The Gemma pair ran in 8-bit precision and the Qwen models in 16-bit, although the Gemma models were tested in equal conditions. Confidence is measured as a probability over the options, and verbalized or sampled confidence may behave differently. Finally, images are available only from 2019 onwards, and the 2017 exam is excluded from RQ1, RQ2 and RQ4 because of parsing issues.

## 9 CONCLUSIONS

Scored as real candidates, open models that a hospital can run on its own hardware would pass the Spanish residency examination with room to spare and could choose almost any specialty. That score, however, reveals little of what a health system needs to know before relying on them. The exam’s penalty is too mild for any of these models to benefit from abstaining, so it cannot reward a model for knowing its limits, and medical tuning lowered confidence without improving the models’ self-knowledge, and it cost accuracy. Likewise, up to 38% of correct answers rested on distractors rejected for a wrong reason, which the models’ confidence did not reveal, and the provided clinical images made no difference to accuracy. Taken together, these show that a model can match into a residency while being overconfident, right for the wrong reasons, and indifferent to the images it is shown. Evaluations intended to support clinical deployment should therefore look beyond the score, reporting how well confidence ranks errors, verifying the reasoning behind correct answers, and testing the use of images directly. Validating the rationale judge against physician ratings and extending the analysis to larger and reasoning models are natural next steps.

## ACKNOWLEDGEMENTS

This work was supported by the Basque Government under grant DEUSTEK6 – Humanized Computing for Smart Sustainable and Healthier Communities and Environments (IT1901-26), and by the European Union’s Horizon Europe Research and Innovation Programme under the LATE-AYA project (Grant Agreement No. 101214326).

## REFERENCES

Alonso, I., Oronoz, M., and Agerri, R. (2024). MedExpQA: Multilingual benchmarking of large language models for medical question answering. Artificial Intelligence in Medicine, 155:102938.

Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. Monthly Weather Review, 78(1):1–3.

ConSalud (2024). MIR 2024: Con que nota media en el´ examen se escogio cada especialidad.´ https://www. consalud.es/formacion/mir/. Per-specialty mean and last-admitted netas and rank, convocatoria 2023.

ConSalud (2025). MIR 2025: Cuantas respuestas netas ´ y numero de orden medio hubo que tener para en-´ trar en cada especialidad. https://www.consalud.es/

formacion/mir/. Per-specialty mean netas and rank, convocatoria 2024.

Correa-Guillen, A., G´ omez-Rodr´ ´ıguez, C., and Vilares, D. (2026). HEAD-QA v2: Expanding a healthcare benchmark for reasoning. In Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026).

Efron, B. and Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall, New York.

Geifman, Y. and El-Yaniv, R. (2017). Selective classification for deep neural networks. In Advances in Neural Information Processing Systems, volume 30.

Gemma Team (2025). Gemma 3 technical report. arXiv preprint arXiv:2503.19786.

Goenaga, I., Atutxa, A., Gojenola, K., Oronoz, M., and Agerri, R. (2024). Explanatory argument extraction of correct answers in resident medical exams. Artificial Intelligence in Medicine, 157:102985.

Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al. (2024). The Llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Guillen-Grima, F., Guillen-Aguinaga, S., Guillen-Aguinaga, L., Alas-Brun, R., Onambele, L., Ortega, W., Montejo, R., Aguinaga-Ontoso, E., Barach, P., and Aguinaga-Ontoso, I. (2023). Evaluating the efficacy of ChatGPT in navigating the Spanish medical residency entrance examination (MIR): Promising horizons for AI in clinical medicine. Clinics and Practice, 13(6):1460–1487.

Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On calibration of modern neural networks. In Proceedings ofthe 34th International Conference on Machine Learning, pages 1321–1330.

Hager, P., Jungmann, F., Holland, R., Bhagat, K., Hubrecht, I., Knauer, M., Vielhauer, J., Makowski, M., Braren, R., Kaissis, G., and Rueckert, D. (2024). Evaluation and mitigation of the limitations of large language models in clinical decision-making. Nature Medicine, 30(9):2613–2622.

Hendrycks, D. and Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. In International Conference on Learning Representations.

Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal ofStatistics, 6(2):65–70.

Jin, D., Pan, E., Oufattole, N., Weng, W.-H., Fang, H., and Szolovits, P. (2021). What disease does this patient have? A large-scale open domain question answering dataset from medical exams. Applied Sciences, 11(14):6421.

Jin, Q., Chen, F., Zhou, Y., Xu, Z., Cheung, J. M., Chen, R., Summers, R. M., Rousseau, J. F., Ni, P., Landsman, M. J., et al. (2024). Hidden flaws behind expert-level accuracy of multimodal GPT-4 vision in medicine. npj Digital Medicine, 7:190.

Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., Das-Sarma, N., Tran-Johnson, E., et al. (2022). Language

models (mostly) know what they know. arXiv preprint arXiv:2207.05221.

Kalai, A. T., Nachum, O., Vempala, S. S., and Zhang, E. (2025). Why language models hallucinate. arXiv preprint arXiv:2509.04664.

Kasai, J., Kasai, Y., Sakaguchi, K., Yamada, Y., and Radev, D. (2023). Evaluating GPT-4 and ChatGPT on Japanese medical licensing examinations. arXiv preprint arXiv:2303.18027.

Kung, T. H., Cheatham, M., Medenilla, A., Sillos, C., De Leon, L., Elepano, C., Madriaga, M., Aggabao, R.,˜ Diaz-Candido, G., Maningo, J., and Tseng, V. (2023). Performance of ChatGPT on USMLE: Potential for AI-assisted medical education using large language models. PLOS Digital Health, 2(2):e0000198.

Madrid-Garc´ıa, A., Rosales-Rosado, Z., Freites-Nunez,˜ D., Perez-Sancristobal, I., Pato-Cour, E., Plasencia-´ Rodr´ıguez, C., Cabeza-Osorio, L., Abasolo-Alcazar,´ L., Leon-Mateos, L., Fern´ andez-Guti´ errez, B., and´ Rodr´ıguez-Rodr´ıguez, L. (2023). Harnessing Chat-GPT and GPT-4 for evaluating the rheumatology questions of the Spanish access exam to specialized medical training. Scientific Reports, 13:22129.

McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 12(2):153–157.

Naeini, M. P., Cooper, G. F., and Hauskrecht, M. (2015). Obtaining well calibrated probabilities using Bayesian binning. In Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence, pages 2901– 2907.

Nori, H., King, N., McKinney, S. M., Carignan, D., and Horvitz, E. (2023). Capabilities of GPT-4 on medical challenge problems. arXiv preprint arXiv:2303.13375.

Qwen Team (2025). Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Riedemann, L., Labonne, M., and Gilbert, S. (2024). The path forward for large language models in medicine is open. npj Digital Medicine, 7.

Roberts, M., Thakur, H., Herlihy, C., White, C., and Dooley, S. (2024). To the cutoff... and beyond? A longitudinal perspective on LLM data contamination. In International Conference on Learning Representations.

Sainz, O., Campos, J. A., Garc´ıa-Ferrero, I., Etxaniz, J., Lopez de Lacalle, O., and Agirre, E. (2023). NLP evaluation in trouble: On the need to measure LLM data contamination for each benchmark. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 10776–10787.

Sellergren, A., Kazemzadeh, S., Jaroensri, T., Kiraly, A., Traverse, M., Kohlberger, T., Xu, S., Jamil, F., Hughes, C., Lau, C., et al. (2025). MedGemma technical report. arXiv preprint arXiv:2507.05201.

Singhal, K., Azizi, S., Tu, T., Mahdavi, S. S., Wei, J., Chung, H. W., Scales, N., Tanwani, A., Cole-Lewis, H., Pfohl, S., et al. (2023). Large language models encode clinical knowledge. Nature, 620:172–180.

Singhal, K., Tu, T., Gottweis, J., Sayres, R., Wulczyn, E., Amin, M., Hou, L., Clark, K., Pfohl, S. R., Cole-

Lewis, H., et al. (2025). Toward expert-level medical question answering with large language models. Nature Medicine, 31:943–950.

Thirunavukarasu, A. J., Ting, D. S. J., Elangovan, K., Gutierrez, L., Tan, T. F., and Ting, D. S. W. (2023). Large language models in medicine. Nature Medicine, 29:1930–1940.

Turpin, M., Michael, J., Perez, E., and Bowman, S. R. (2023). Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting. In Advances in Neural Information Processing Systems, volume 36.

Vilares, D. and Gomez-Rodr´ ´ıguez, C. (2019). HEAD-QA: A healthcare dataset for complex reasoning. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 960–966.

Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. Journal ofthe American Statistical Association, 22(158):209–212.

Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., and Hooi, B. (2024). Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs. In International Conference on Learning Representations.

Yan, Q., He, X., Yue, X., and Wang, X. E. (2025). Worse than random? An embarrassingly simple probing evaluation of large multimodal models in medical VQA. In Findings of the Association for Computational Linguistics: ACL 2025.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., and Stoica, I. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems (Datasets and Benchmarks Track), volume 36.