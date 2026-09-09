# Performance of Clinical AI System and Physicians and Frontier Language Models in primary care diagnostics

Andy Nkansah<sup>1\*</sup>, Hanna Plotnitskaya<sup>1</sup>, Stanislau Salavei<sup>1</sup>, Anna Kozlova<sup>1</sup>, Piotr Gibas<sup>2</sup>, Julian Milek, Viktar Harbachou, Aleksey Ropan<sup>1</sup>, Pavel Satalkin<sup>1</sup>

<sup>1</sup>A.I. Doctor Medical Assist LTD, Limassol, Cyprus.

<sup>2</sup>Family Medicine Specialist, Kraków, Poland.

Corresponding author: Andy Nkansah (andynkansah@ik.me)

Author contacts: Hanna Plotnitskaya (doctora.plotnitskaya@gmail.com); Julian Milek (j.milek@protonmail.com);   
Viktar Harbachou (harbachou@outlook.com); Pavel Satalkin (satalkin.p@gmail.com).

## Abstract

Clinical AI evaluation should encompass diagnosis and management after adaptive information gathering. We compared Doctorina, eight physicians and four standalone frontier language models in 150 synthetic Polish-language primary-care consultations. Doctorina achieved 82.0% Top-1 concordance versus 57.0% for physicians (diference, 25.0 percentage points; 95% confidence interval, 17.7–32.7) and 97.3% versus 85.0% primary-or-reference-diferential concordance. Across 149 case pairs, normalized workup and treatment scores were 89.4 versus 66.9 and 83.7 versus 61.2. Doctorina had the highest diagnostic point estimates among all six groups; Kimi K3 ranked next, while Claude Opus 5 led the closely spaced management estimates of Opus, Doctorina and Kimi. A second Doctorina execution reproduced the advantages over physicians across all outcomes. Doctorina’s advantage over physicians therefore extended from primary-diagnosis selection to higher-rated diagnostic workup and initial treatment after adaptive consultation.

## Introduction

Large language models (LLMs) have achieved substantial gains on medical knowledge and question answering tasks.<sup>1</sup> Examination-style and otherwise closed-ended tasks measure medical knowledge under fixed information conditions, whereas adaptive clinical consultation additionally requires respondent-directed questioning, sequential reasoning and management planning. A systematic review and meta-analysis of 83 studies found no statistically detectable diference between generative artificial intelligence and physicians overall or non-expert physicians; generative artificial intelligence performed worse than expert physicians, and 76% of studies were judged at high risk of bias. <sup>2</sup> Comparative performance therefore depends on the clinical task, the information available to each respondent, comparator expertise, interaction design and outcome definition.

Clinical consultation is a sequential process in which the respondent identifies relevant questions, revises a diferential diagnosis, selects investigations and translates clinical judgement into an initial management plan. In an adaptive consultation, diagnostic and management performance is contingent on the respondent’s information-seeking decisions because the respondent determines what to ask and when to complete the encounter. Simulation-based evaluations have characterized variation in information gathering, adherence to diagnostic and treatment guidelines, and robustness to the order and quantity of information supplied, <sup>3</sup> while replacing case vignettes with simulated multi-turn consultations changed, and often reduced, diagnostic accuracy across the evaluated models and task formats.<sup>4</sup> Conversational and case-based studies have consequently evaluated history-taking, diagnosis, diferential diagnosis and management as related but distinct dimensions.<sup>5,6</sup> Controlled clinical vignettes enable standardized comparisons using common case presentations and constructed reference outcomes. <sup>7</sup>

Doctorina is a patient-facing clinical AI system built as a clinical scafold around underlying language models. This scafold combines task-specific instructions, agent coordination and consultation-state management to support adaptive, multi-turn consultations. Specialized components handle clinical reasoning, patient communication and final synthesis, supported by translation, attachment-processing, reference and follow-up functions. Across successive exchanges, the system maintains the evolving consultation state and produces a patient-visible recommendation comprising a most-likely diagnosis, diferential diagnoses, diagnostic workup and initial treatment. The integrated Doctorina product, rather than any component in isolation, was the unit of evaluation.

Application-level orchestration can shape the clinical behaviour of an AI product beyond the contribution of an individual language model. In one evaluation, tested clinical-agent systems produced modest accuracy gains over baseline LLMs while using more than ten times as many tokens and more than twice the latency in some evaluated settings; <sup>8</sup> in another, three generalpurpose frontier LLMs outperformed two specialized clinical AI tools across medical-knowledge, clinician-alignment and real-clinician-query evaluations. <sup>9</sup> A direct comparison of an integrated patient-facing clinical AI product with physicians and standalone frontier LLMs, each conducting an adaptive consultation and producing diagnostic and management outputs, therefore addresses an important evaluation question.

Language was also part of the evaluated condition. A Polish medical-examination benchmark containing 22,604 valid questions included paired Polish–English LEK and LDEK subsets and identified model- and examination-dependent cross-language diferences, <sup>10</sup> while a translationcontrolled clinical-vignette study identified a model-by-language interaction, with some models performing worse in Polish and others showing similar overall performance across languages. <sup>11</sup> These findings motivate direct evaluation in Polish-language consultations.

Here, we conducted a controlled reader-by-case study using 150 synthetic Polish-language primary-care cases administered as adaptive simulated consultations. Doctorina was compared with eight physicians practising or training in family or internal medicine in Poland and with Google’s Gemini 3.1 Pro Preview, Anthropic’s Claude Opus 5, OpenAI’s GPT-5.6-sol and Moonshot AI’s Kimi K3. The analytical hierarchy designated the case-standardized diference between Doctorina and the physician cohort in Top-1 concordance with the designated primary diagnosis as the primary comparison. Primary-or-reference-diferential concordance was the secondary diagnostic outcome; physician-reviewer ratings of diagnostic workup and initial treatment and comparisons with the standalone LLMs were exploratory. All groups were assessed on the same case corpus, clinical task and outcome definitions.

## Results

## Analytical dataset

All 150 synthetic Polish-language cases contributed to the diagnostic analyses. The diagnostic analysis included 240 assigned responses from eight physicians—five specialists and three residents: 60 cases had one physician response and 90 had two. Management outcomes were available for 239 responses, with at least one retained assessment for every case.

Both eligible Doctorina executions attempted all 150 cases. Run 481 yielded 148 technically valid diagnostic outputs and 147 technically valid management records from 149 recorded management attempts. Run 479 yielded 145 technically valid diagnostic outputs and 143 technically valid management outputs.

Gemini 3.1 Pro Preview, Claude Opus 5 and Kimi K3 yielded final outputs for all 150 cases, while GPT-5.6-sol yielded 148. The case-standardized six-group analyses comprised 150 equally weighted diagnostic cases and 149 complete management case pairs. Doctorina and physician outputs were assessed within one review block, while the four standalone-model outputs were assessed within a second aligned block using the same case references, endpoint definitions and rating direction. The six-group results combine a within-block Doctorina–physician comparison with aligned cross-block comparisons involving the standalone models.

## Diagnostic concordance

For the primary analytical comparison within the Doctorina–physician review block, concordance was 82.0% for Doctorina and 57.0% for physicians, a diference of 25.0 percentage points (95% paired-case bootstrap CI, 17.7 to 32.7; Fig. 1). In the aligned comparisons with the separately reviewed standalone-model outputs, Top-1 concordance was 78.7% for Kimi, 74.0% for Opus, 70.7% for GPT and 66.7% for Gemini. Doctorina-minus-model diferences were 3.3 percentage points for Kimi, 8.0 points for Opus, 11.3 points for GPT and 15.3 points for Gemini; the respective 95% CIs were -4.0 to 10.0, 0.7 to 15.3, 2.7 to 19.3 and 7.3 to 23.3.

The small aggregate separation between Doctorina and the two highest-scoring standalone models reflected predominantly concordant case-level classifications. Doctorina and Kimi had the same Top-1 status in 121 of 150 cases (80.7%); Doctorina alone was concordant in 17 cases and Kimi alone in 12. Doctorina and Opus had the same status in 116 cases (77.3%), with 23 Doctorina-only and 11 Opus-only concordant cases.

For the secondary primary-or-reference-diferential endpoint, concordance was 97.3% for Doctorina and 85.0% for physicians, a diference of 12.3 percentage points (95% CI, 7.0 to 18.0). Standalone-model concordance was 91.3% for Kimi, 90.0% for Opus, 82.0% for GPT and 78.7% for Gemini; the corresponding Doctorina-minus-model diferences were 6.0, 7.3, 15.3 and 18.7 percentage points, with respective 95% CIs of 1.3 to 10.7, 2.7 to 12.7, 8.7 to 22.0 and 12.0 to 25.3.

## Diagnostic-workup and initial-treatment outcomes

On the exploratory 0–100 normalized management scale, diagnostic-workup scores across 149 complete case pairs were 90.3 for Opus, 89.4 for Doctorina, 86.4 for Kimi, 80.5 for GPT, 72.1 for Gemini and 66.9 for physicians (Fig. 1). Doctorina-minus-comparator diferences were 22.6 normalized-score points for physicians (95% CI, 18.2 to 26.8), 17.3 points for Gemini (12.6 to 22.1), -0.8 points for Opus (-4.9 to 3.2), 8.9 points for GPT (4.5 to 13.4) and 3.0 points for Kimi (-0.8 to 6.9).

Initial-treatment scores were 86.6 for Opus, 83.7 for Doctorina, 82.4 for Kimi, 73.2 for GPT, 62.9 for Gemini and 61.2 for physicians. Doctorina-minus-comparator diferences were 22.6 normalizedscore points for physicians (95% CI, 16.9 to 28.2), 20.8 points for Gemini (15.6 to 26.0), -2.9 points for Opus (-7.0 to 1.2), 10.6 points for GPT (5.9 to 15.4) and 1.3 points for Kimi (-3.0 to 5.4). Opus, Doctorina and Kimi had closely spaced management point estimates; the Doctorina-minus-Opus and Doctorina-minus-Kimi intervals spanned modest diferences in either direction in both domains.

The case-weighted verdict distributions are shown in Table 1. Doctorina, Opus and Kimi ratings were concentrated in categories 1 and 2 in both management domains, whereas physician ratings were distributed more broadly across the five categories. On the 150-case planned denominator, the case-weighted proportions receiving a poor or very poor verdict (categories 4–5) were 5.3% for Doctorina, 20.7% for physicians, 14.0% for Gemini, 4.7% for Opus, 6.7% for GPT and 5.3% for

## a Six-group case-standardized diagnostic and management performance

150 diagnostic cases; 149 complete management pairs; common case and endpoint definitions.

![](images/0772d7fe920d520996edb1e408fd929f9d049b841afcde60f16d5e85df122f1b.jpg)

## b Doctorina-minus-comparator case-standardized differences

Mean differences with two-sided 95% paired whole-case bootstrap intervals (10,000 resamples).

![](images/78a703e3b4a8338352585d7252f682757868d737ef4994b6e1b8dc8b0fb00c88.jpg)  
Difference (percentage points or normalized-score points)

Fig. 1 Six-group case-standardized diagnostic and management performance. a, Top-1 concordance, primary-orreference-diferential concordance, diagnostic-workup score and initial-treatment score for Doctorina, physicians, Gemini 3.1 Pro Preview, Claude Opus 5, GPT-5.6-sol and Kimi K3. b, Paired Doctorina-minus-comparator diferences; positive values favour Doctorina. Error bars show pointwise, two-sided 95% percentile intervals from 10,000 paired whole-case bootstrap resamples. Doctorina was represented by run 481, physicians by within-case means and each standalone model by one output. Diagnostic analyses included 150 cases and management analyses 149 complete case pairs. Doctorina and physician outputs were reviewed together; standalone-model outputs were reviewed in a second aligned block. Only the Doctorina–physician Top-1 comparison was primary; the broader diagnostic endpoint was secondary, and management and standalone-model comparisons were exploratory. Diagnostic efects are reported in percentage points and management efects in normalized-score points.

Kimi in workup. The corresponding treatment proportions were 7.3%, 26.3%, 20.7%, 5.3%, 10.7% and 6.0%, respectively.  
Table 1 Physician-reviewer verdict distributions for diagnostic workup and initial treatment.
<table><tr><td>Domain</td><td>System</td><td>Numeric ratings</td><td>1 (%)</td><td>2 (%)</td><td>3 (%)</td><td>4 (%)</td><td>5 (%)</td><td>Invalid Unavailable Median (%)</td><td>(%)</td><td>(IQR)</td></tr><tr><td>Diagnostic workup</td><td>Doctorina</td><td>147</td><td>76.7</td><td>11.3</td><td>4.7</td><td>5.3</td><td>0.0</td><td>1.3</td><td>0.7</td><td>1 (1−1)</td></tr><tr><td></td><td>Physicians</td><td>150 32.3 26.7</td><td></td><td></td><td>20.3</td><td>18.0</td><td>2.7</td><td>0.0</td><td>0.0</td><td>2 (1–3)</td></tr><tr><td></td><td>Gemini 3.1 Pro Preview</td><td></td><td></td><td></td><td>15040.0 26.7 19.3</td><td>10.0</td><td>4.0</td><td>0.0</td><td>0.0</td><td>2 (1–3)</td></tr><tr><td></td><td>Claude Opus 5</td><td>15076.7 13.3</td><td></td><td></td><td>5.3</td><td>3.3</td><td>1.3</td><td>0.0</td><td>0.0</td><td>1 (1-1)</td></tr><tr><td></td><td>GPT-5.6-sol</td><td>148</td><td>54.7</td><td>22.7</td><td>14.7</td><td>6.0</td><td>0.7</td><td>1.3</td><td>0.0</td><td>1 (1–2)</td></tr><tr><td></td><td>Kimi K3</td><td>150</td><td>64.7</td><td>22.0</td><td>8.0</td><td>4.7</td><td>0.7</td><td>0.0</td><td>0.0</td><td>1 (1–2)</td></tr><tr><td>Initial treatment</td><td>Doctorina</td><td>147</td><td>64.7</td><td>16.0</td><td>10.0</td><td>6.0</td><td>1.3</td><td>1.3</td><td>0.7</td><td>1 (1-2)</td></tr><tr><td></td><td>Physicians</td><td>15031.0</td><td></td><td>16.0</td><td>26.7</td><td>20.3</td><td>6.0</td><td>0.0</td><td>0.0</td><td>3 (1–4)</td></tr><tr><td></td><td>Gemini 3.1 Pro Preview</td><td>150 31.3</td><td></td><td>21.3</td><td>26.7</td><td>9.3</td><td>11.3</td><td>0.0</td><td>0.0</td><td>2 (1–3)</td></tr><tr><td></td><td>Claude Opus 5</td><td>150</td><td>67.3</td><td>19.3</td><td>8.0</td><td>3.3</td><td>2.0</td><td>0.0</td><td>0.0</td><td>1 (1-2)</td></tr><tr><td></td><td>GPT-5.6-sol</td><td>148</td><td>44.0</td><td>22.7</td><td>21.3</td><td>6.0</td><td>4.7</td><td>1.3</td><td>0.0</td><td>2 (1–3)</td></tr><tr><td></td><td>Kimi K3</td><td>15058.7 20.015.3</td><td></td><td></td><td></td><td>4.7</td><td>1.3</td><td>0.0</td><td>0.0</td><td>1 (1-2)</td></tr></table>

Verdict columns are percentages of 150 planned cases. The physician category distribution therefore uses all 150 planned cases, whereas the physician management mean reported in Fig. 1 and the text is restricted to the 149 complete Doctorina–physician case pairs. Each case contributes equal weight; the physician rows summarize 239 response ratings across 150 cases, with each of two ratings contributing half of the case’s weight when both were present. Medians and interquartile ranges (IQRs) use the same case weights. The scale was ordered from 1 (Excellent) to 5 (Very Poor). All-attempt normalized scores used the planned denominator, with 0 assigned to technical-error and incomplete output states. The Invalid attempt column comprises Doctorina records labelled Technical error and GPT attempts without a final output; the Record unavailable column represents the single absent run-481 management record. Verdict-category summaries use observed numeric ratings. Doctorina and physician outputs were evaluated within one review block, and the four standalone-model outputs within a second aligned block.

## Consistency across observed physician assignments

Across the five disjoint physician packet-membership blocks, Doctorina-minus-physician diferences were positive in every block for Top-1 concordance, diagnostic workup and treatment. Top-1 diferences ranged from 10.0 to 46.7 percentage points; the corresponding ranges were 0.0 to 33.3 percentage points for primary-or-reference-diferential concordance, 13.3 to 30.8 normalized-score points for workup and 14.7 to 38.3 points for treatment. The lower bound of 0.0 for the broader diagnostic endpoint represented one block-level tie. In respondent-specific paired comparisons, Doctorina-minus-physician Top-1 diferences ranged from 6.7 to 46.7 percentage points across the eight physicians (Supplementary Fig. 1). Complete management-pair counts ranged from 28 to 30 per physician. No block- or respondent-specific point estimate favoured physicians, although ties occurred on the broader diagnostic endpoint and some intervals included zero in these 28–30-case comparisons.

## Exploratory case-level heterogeneity and diagnostic-error overlap

Complexity levels 1, 2 and 3 contained 97, 38 and 15 diagnostic cases, respectively; the corresponding complete management strata contained 96, 38 and 15 cases. Doctorina Top-1 concordance across these levels was 83.5%, 78.9% and 80.0%, compared with 63.9%, 47.4% and 36.7% for physicians.

Initial-treatment scores were 87.8, 80.9 and 65.0 for Doctorina and 66.8, 53.6 and 44.2 for physicians. Although the Doctorina–physician Top-1 point-estimate margin was larger at higher recorded complexity, all pairwise complexity-by-group interaction intervals for the physician comparison spanned zero across the four outcomes. The stratified estimates were therefore compatible with similar Doctorina–physician margins across the recorded complexity levels. Six-group estimates for every outcome and complexity stratum are reported in Supplementary Data 10.

Across 149 Doctorina cases with complete management data, Spearman correlations between Top-1 concordance and diagnostic-workup and treatment scores were 0.243 and 0.142, respectively. Across all six groups, the corresponding Top-1 correlations ranged from 0.073 to 0.267 for workup and from 0.093 to 0.289 for treatment. Twenty-three Doctorina cases combined Top-1 concordance with at least one management score below the Good threshold, whereas 14 combined Top-1 discordance with Good-or-Excellent scores in both management domains.

The five AI systems were all Top-1 concordant in 70 of 150 cases and all discordant in eight; exactly one AI system was concordant in nine cases. Doctorina was the sole concordant AI system in six cases. Of the 27 cases in which Doctorina’s Top-1 diagnosis was discordant, at least one standalone LLM was concordant in 19. Of the eight cases in which all five AI systems were Top-1 discordant, seven were also Top-1 discordant for every observed physician response. These case-level patterns quantified diagnostic-concordance overlap among the five observed AI-system outputs and the physician cohort.

## Execution-level sensitivity and observed run agreement

Run 479 reproduced positive Doctorina-minus-physician diferences across all four outcomes: 23.7 percentage points for Top-1 concordance, 10.3 points for primary-or-reference-diferential concordance, 18.3 normalized-score points for workup and 16.9 points for treatment; the respective 95% CIs were 16.0 to 31.3, 4.7 to 16.0, 13.1 to 23.4 and 10.7 to 23.0.

The comparisons with the two highest-scoring standalone models were more sensitive to the eligible Doctorina execution. With run 479, Doctorina-minus-Opus diferences were 6.7 percentage points for Top-1 concordance, 5.3 points for primary-or-reference-diferential concordance, -4.8 normalized-score points for workup and -8.3 points for treatment; the respective 95% CIs were -0.7 to 14.0, 0.0 to 10.7, -9.8 to 0.0 and -12.8 to -4.0. Doctorina-minus-Kimi diferences were 2.0 percentage points for Top-1 concordance, 4.0 points for primary-or-reference-diferential concordance, -1.0 normalized-score point for workup and -4.2 points for treatment; the respective 95% CIs were -4.7 to 8.7, -1.3 to 9.3, -6.0 to 3.8 and -9.5 to 0.8. Diagnostic point estimates relative to Opus and Kimi remained positive in both executions, while management comparisons varied by execution.

Across all attempted cases from runs 479 and 481, Top-1 status agreed in 128 of 150 cases (85.3%; Cohen’s $\kappa = 0 . 5 1 7 )$ , and primary-or-reference-diferential status agreed in 141 cases (94.0%; κ = 0.153). Among the 143 cases with technically valid diagnostic outputs from both executions, Top-1 status agreed in 126 cases (88.1%; κ = 0.567), and broader-endpoint status agreed in 141 (98.6%; $\kappa = 0 . 4 9 3 )$ . Across all 149 matched management attempts, exact all-attempt normalized-score agreement was observed in 121 cases (81.2%) for workup and 110 (73.8%) for treatment. Among the 142 technically valid management pairs, exact verdict-category agreement was observed in 119 cases (83.8%) for workup and 108 (76.1%) for treatment. Workup and treatment were each within one verdict category in 133 cases (93.7%); 124 cases (87.3%) were within one category in both domains. Run 481 exceeded run 479 by 4.2 workup points (95% CI, 0.5 to 8.4) and 5.5 treatment points (1.8 to 9.4) across 149 matched attempts; among 142 technically valid pairs, the diferences were 0.9 points (-1.9 to 3.7) and 2.8 points (-0.2 to 5.8), respectively. Complete case-level robustness, heterogeneity, overlap and observed run-agreement tables are provided in Supplementary Data 10.

## Discussion

Doctorina’s principal advantage relative to physicians was higher performance across both diagnosis and management after adaptive consultation. Within the shared Doctorina–physician review block, the 25.0-percentage-point diference in Top-1 concordance was accompanied by higher workup and treatment scores and smaller proportions of Poor or Very Poor management ratings. This combination is clinically relevant to the evaluated task: stronger primary-diagnosis selection was accompanied by higher-rated investigation and initial treatment plans. Positive physiciancomparison diferences were reproduced in the second eligible execution and across the assignment stratified point estimates, apart from ties on the broader diagnostic endpoint. In exploratory cross-block comparisons, Doctorina’s point estimates exceeded Gemini and GPT across all four outcomes and led all standalone models on both diagnostic endpoints. Management estimates placed Opus, Doctorina and Kimi in a closely spaced leading group, with Opus numerically highest. Doctorina’s comparative strength was therefore broad relative to the physician cohort and predominantly diagnostic relative to the strongest standalone models.

These comparisons are best interpreted in relation to the adaptive consultation task. Unlike complete-vignette evaluations, adaptive assessment makes performance contingent on respondent directed information gathering. AMIE (Articulate Medical Intelligence Explorer) showed higher diagnostic accuracy than primary-care physicians in randomized, double-blind crossover text consultations with validated patient actors, <sup>5</sup> whereas general-purpose LLMs evaluated on cases derived from real patient data performed below physicians and deteriorated further when required to gather information stepwise.<sup>3</sup> In a randomized diagnostic-reasoning trial, an LLM performed well alone, while physician access to the LLM did not significantly improve diagnostic-reasoning scores compared with conventional resources. <sup>12</sup> Together, these studies show that comparative performance depends on the information environment, interaction design and role assigned to the AI. The current findings extend this evidence to an integrated clinical AI product evaluated in Polish through adaptive multi-turn consultations in which information acquisition preceded the scored diagnostic and management outputs.

Doctorina’s product-level performance reflects the joint operation of its underlying language models and clinical scafold; the present comparison does not isolate their respective contributions. General-purpose frontier models have surpassed specialized clinical tools on some benchmarks, <sup>9</sup> while agent systems have achieved more modest gains at appreciably greater computational cost in others.<sup>8</sup> Replacing Doctorina’s underlying models with models more capable in their assigned clinical tasks could improve system-level performance, but this remains an untested hypothesis. Controlled studies could vary the underlying models while holding the scafold, case corpus and evaluation conditions constant, and complement these comparisons with component ablations under a fixed model configuration. Such studies should assess diagnostic and management quality alongside safety, technical completion and computational cost to determine whether model replacement improves the complete system.

The case-level findings further distinguished diagnostic prioritization from management quality. Most Doctorina Top-1 discordances still matched an accepted reference-diferential diagnosis, indicating prioritization within the accepted set rather than complete diagnostic-set exclusion. The weak association between diagnostic concordance and management scores across groups further showed that diagnosis, workup and treatment capture distinct dimensions. Correspondingly, some Doctorina cases paired a correct primary diagnosis with sub-Good management, whereas others paired a discordant primary diagnosis with Good-or-Excellent management. These results support separate assessment and optimization of primary-diagnosis ranking, diagnostic workup and treatment.

Agreement across the two eligible executions provided a separate product-performance dimension. Diagnostic endpoints showed high raw agreement, although chance-corrected estimates were lower, particularly for the near-ceiling broader endpoint, and increased after technical failures were excluded. Management agreement was less exact. All-attempt diferences favoured run 481, whereas valid-pair intervals included zero, indicating that technical completion contributed to the observed run-level diference; comparisons with Opus and Kimi likewise varied by execution. These findings support endpoint-specific agreement reporting and repeated-execution testing for closely ranked systems.

The complexity analysis refined the interpretation of the aggregate performance. Doctorina’s Top-1 point estimates remained broadly similar across the three recorded complexity levels, whereas physician point estimates declined descriptively; Doctorina–physician point-estimate diferences were positive in every stratum. Interaction intervals were compatible with no complexity-related change in the Doctorina–physician margin for each outcome, and the 15-case level-3 stratum produced wide intervals. Management point estimates were lower in the recorded higher-complexity strata for Doctorina and most comparator groups.

Diagnostic errors were partly shared and partly complementary across the five AI systems. Intersystem disagreement may therefore identify cases warranting additional review, whereas cases missed by every AI system and every observed physician response provide a focused set for examining reference ambiguity, incomplete elicitation, atypical presentation and common reasoning failure.

The Polish-language setting provides a locally relevant contribution. Prior Polish medical benchmarks have shown model- and task-dependent cross-language performance in examinations and translated vignettes. <sup>10,11</sup> This study adds adaptive history-taking and physician-rated management in the intended language and consultation format.

Several design features strengthen the evidence. All groups used one 150-case corpus, common constructed references, shared endpoint definitions and a common permitted-facts boundary. Case standardization gave each case equal weight regardless of whether one or two physician responses were available. The adaptive design embedded respondent-directed information seeking within the consultation that generated the diagnostic and management outputs. Whole-system analyses retained technical failures, while ordinal distributions separated rated content from failure burden. The central Doctorina–physician finding was examined in a second eligible execution and across physician, packet-membership and complexity analyses.

The controlled synthetic corpus enabled standardized comparison across a broad range of conditions, with the 150 cases distributed in proportion to the relative frequencies of selected eligible main ICD-10 diagnosis codes in 2024 National Health Fund service-reporting data. The designated diagnoses were established through the protocol’s three-clinician review procedure and served as the constructed study reference for concordance assessment. The stochastic simulator produced respondent-specific histories within a common permitted-facts boundary. Within this design, the scored construct was final-output performance after adaptive text consultation. Simulator fidelity, information elicitation and consultation eficiency were outside the scored endpoints, and the evaluation did not encompass routine clinical workflow or patient outcomes. Generalization to clinical practice therefore requires prospective validation.

The physician comparison represented eight purposively recruited clinicians who completed partially crossed 30-case assignments in a single, approximately six-hour, resource-restricted text session. The analytical records did not preserve the packet-allocation rule, the number of physicians approached or reasons for non-participation, limiting assessment of recruitment and assignment efects. The purposive, non-randomized physician sample and reader-by-case structure constrain generalization to broader clinician populations, and the comparison estimates the complete Doctorina product relative to unaided physicians under the recorded study conditions. Doctorina– physician and standalone-model outputs were evaluated in separate, source-visible review blocks. The first block retained one final rating per output but not its achieved reviewer-pool size, outputlevel evaluator assignments, duplicate ratings or escalation identifiers. Consequently, cross-block AI contrasts may incorporate reviewer-pool and presentation efects, execution and scoring variation cannot be separated fully, and inter-rater reliability could not be estimated. A common blinded review pool, repeated ratings and broader physician sampling would strengthen the next validation.

The four Doctorina run identifiers referred to separate executions of the same case corpus. Under the development lead’s branch-eligibility mapping, runs 478 and 480 were excluded, leaving runs 479 and 481 eligible for analysis. Run 481 was selected for the primary analysis because it contained more technically valid diagnostic and management records; run 479 was analysed separately as the execution-level sensitivity analysis. Because technical completeness contributed directly to all-attempt outcomes, this selection rule may favour the primary execution on those outcomes. Each standalone model contributed one corpus-wide execution, so cross-system estimates characterize the recorded execution samples. Routes, prompts, concurrency and completion states were available, whereas batch-wide generation settings and a run-linked manifest documenting the deployed Doctorina build and component configuration were unavailable, limiting exact technical replication. The standalone-model executions followed the physician–Doctorina comparison by more than two months; cross-system contrasts may therefore include temporal variation alongside system diferences.

The normalized management score assumes equal spacing between adjacent ordinal categories; category distributions and medians therefore provide complementary summaries. The retained dataset provided a source-assigned complexity level for each case but did not include the corresponding classification criteria or assessor attribution; the stratified analyses were interpreted descriptively.

The primary analytical hierarchy was post-specified because the retained review field operationalized reference-diferential concordance rather than ranked Top-3 accuracy. The retained protocol specified Top-1 and ranked Top-3 diagnostic accuracy as co-primary outcomes, each at a two-sided alpha of 0.025. This timing limits confirmatory interpretation. The Doctorina–physician Top-1 contrast was treated as primary; the remaining diagnostic, management, standalone-model, complexity and post hoc analyses were secondary or exploratory. The standalone-model analysis plan was finalized after output generation and before completion of the standalone-model physician-review block. Confidence intervals were pointwise without multiplicity adjustment. The analytical sample comprised the available 150-case corpus and eight completed physician assignments rather than a formal power target, and bootstrap intervals quantify case variation conditional on the observed physicians and executions. These conditions define the inferential scope of the reported estimates.

Prospective multicentre evaluation should assess the complete Doctorina system in routine clinical workflows, incorporating repeated executions, complexity-enriched sampling, disagreementtriggered review and patient-relevant outcomes while preserving distinct diagnostic and management endpoints.

## Methods

## Study design and analytical scope

This controlled reader-by-case study compared six respondent types in Polish-language simulated primary-care consultations: physicians, the Doctorina clinical AI system, Gemini 3.1 Pro Preview, Claude Opus 5, GPT-5.6-sol and Kimi K3. Eight physicians completed allocated subsets of the 150-case synthetic corpus. Two branch-correct Doctorina executions and one execution of each standalone large language model (LLM) attempted all 150 cases.

Physician consultations and the two eligible Doctorina executions were conducted on 17 May 2026; the standalone-model consultations followed on 3–4 August 2026.

All groups used the same case corpus, constructed references, clinical task and outcome definitions. Case-standardized efects quantified the performance of each complete AI implementation—including its model, instructions, orchestration, state management and patient-facing presentation—or the observed physician cohort under this shared task. In every adaptive consultation, the final diagnostic and management outputs were contingent on the questions selected by the respondent. The casebased comparative design was informed by published vignette- and simulation-based evaluations of clinical AI systems and physicians. <sup>3,5–7,12</sup>

Relevant items from STARD 2015, STARD-AI and TRIPOD-LLM informed reporting. <sup>13–15</sup> The study design and information boundaries are summarized in Fig. 2.

## Setting and physician participants

The physician comparison was conducted during one approximately six-hour in-person session in Warsaw, Poland. Recruitment was purposive and used direct professional outreach. Eligibility required a medical degree, a valid licence to practise in Poland, current practice or postgraduate training in family medicine, general practice or internal medicine and availability for the complete session. Eligibility also required professional independence from Doctorina development, prompting, evaluation design and case creation, with first exposure to the analytical case set occurring during the assessed procedure.

Eight physicians completed the study: five specialists and three second-year residents. The study procedure provided each participant with access to the study interface and one non-analytical test case on the day before the assessed session to practise the interface and response format. During the assessed session, all eight physicians worked contemporaneously in the same controlled space under common instructions; an independent observer unafiliated with Doctorina was present with responsibility for monitoring protocol adherence and recording any deviations or technical incidents. Physician participation and output review were compensated activities. Participants used numerical study identifiers and were distinct from the physicians who evaluated the outputs.

## Clinical-vignette corpus and physician allocation

Clinician authors deliberately constructed the 150 synthetic cases (E1025–E1174) using a common template. The case authors were external to Doctorina and separate from the physician comparator, prompt-tuning and model-behaviour-optimization teams. The diagnostic composition of the corpus was informed by 2024 National Health Fund (NFZ) patient counts for services reported under main ICD-10 diagnosis codes at the five-character level. Eligible codes represented clinically informative underlying diagnoses reported for at least five patients; administrative encounter and symptomonly codes were outside the eligible diagnosis set. Relative frequencies were calculated across the eligible diagnoses; the most common eligible conditions were then selected and scaled to define the 150-case diagnostic distribution.<sup>16</sup> Each vignette contained a presenting complaint of one or two short sentences and a separate permitted-facts bank covering demographics; symptom onset, course, character and severity; aggravating and relieving factors; relevant positive and negative findings; medicines, allergies and previous disease; family and social context; vital signs; and investigations available at presentation. Withheld reference fields specified the designated primary diagnosis and ICD-10 code, acceptable diferential diagnoses, diagnostic workup, initial treatment and case-specific critical conditions.

The final case-review protocol required field-by-field review by the author and two additional clinicians, a three-of-three acceptance threshold and renewed review after revision of any disputed field. The accepted final fields constituted the constructed case references. The corpus contained 127 distinct recorded disease labels and 128 distinct ICD-10 values; 82 cases described female patients and 68 described male patients. The age distribution is summarized in Table 2. The source case table recorded complexity on an ordinal scale from level 1 (lowest complexity) to level 3 (highest complexity): 97 cases were level 1, 38 were level 2 and 15 were level 3; these source classifications defined the exploratory strata. A named comorbidity was recorded in 119 cases.

Table 2 Age distribution of the synthetic primary-care case corpus.
<table><tr><td>Age group, years</td><td>Cases, n</td><td>%</td></tr><tr><td>0-1</td><td>2</td><td>1.3</td></tr><tr><td>2-5</td><td>3</td><td>2.0</td></tr><tr><td>6-12</td><td>3</td><td>2.0</td></tr><tr><td>13-17</td><td>8</td><td>5.3</td></tr><tr><td>18-25</td><td>14</td><td>9.3</td></tr><tr><td>26-39</td><td>39</td><td>26.0</td></tr><tr><td>40-59</td><td>40</td><td>26.7</td></tr><tr><td>60-74</td><td>32</td><td>21.3</td></tr><tr><td>75 or older</td><td>9</td><td>6.0</td></tr><tr><td>Total</td><td>150</td><td>100.0</td></tr></table>

Note: Counts and percentages are shown separately. Percentages use the full 150- case denominator and may not sum to 100% because of rounding.

Ten packets of 30 cases formed five paired case-membership sets. The two packets in each pair contained the same cases in diferent orders, placing every case in two packet numbers. One packet was assigned to each physician. Eight completed assignments yielded 240 retained physician diagnostic consultations: 90 cases had two physician responses and 60 had one.

## Consultation environment and physician procedure

Cases were administered as adaptive text dialogues. Each dialogue opened with the presenting complaint. Matched execution traces identified o3-2025-04-16 as the simulated-patient model; the simulator template specified a temperature of 1, and the model generated replies from the casespecific permitted-facts bank and preceding dialogue. The simulator was instructed to answer briefly and truthfully in the language of the case, remain within case-supported facts, preserve diagnostic neutrality and state when the case information did not resolve a request. Messages containing several unrelated questions elicited responses to the first one or two and could prompt brief clarification. Consultations used a fixed evidence environment comprising the investigations documented in each case. These rules standardized the factual boundary while allowing each respondent to determine the sequence and content of questioning.

Physicians completed each consultation independently through an external text-based conversational interface and determined when to end the exchange. Each started case was completed before submission. The assessment interface presented the case dialogue as the sole evidence source, and physicians worked independently under standardized closed-resource conditions. At completion, each physician submitted a most-likely diagnosis, diferential diagnoses, recommended diagnostic workup and initial treatment or medication plan. Diagnostic evaluation used semantic concordance of the submitted free-text diagnosis; ICD-10 coding was optional.

## AI systems and executions

The study protocol identified Doctorina as version 2.1.3 and described an integrated clinical AI product comprising doctor, assistant, final-recommendation, translation, attachment-processing, reference and follow-up functions coordinated by a routing agent. In the source code for the protocollinked v2.1.3 release (commit 5f134f4), the configured OpenAI model identifiers were o3 for the doctor agent; gpt-5-mini-2025-08-07 for the assistant, translation and follow-up agents; gpt-5-2025-08-07 for the final-recommendation and attachment-processing agents; and o3-mini-2025-01-31 for the references agent. The routing component dispatched between agents without a separately assigned language model.

Doctorina was evaluated as the complete product with the protocol-described native functions. The execution protocol used an isolated session for each case, presented the Polish opening complaint, routed questioning through the case-bounded simulator and required the same four clinical domains as the physician procedure. A fixed execution protocol was applied across cases, and the generated patient-visible outputs passed directly to evaluation.

Runs 478, 479, 480 and 481 were four separately recorded Doctorina executions, each attempting the same 150 cases; the identifiers denoted execution batches, not diferent case sets. Eligibility and analytical role were determined in two steps. First, the development lead’s run-to-branch mapping identified runs 479 and 481 as originating from the software branch designated for the study. The same mapping placed runs 478 and 480 on a diferent branch, and these runs were excluded from the reported comparisons. Second, technical completeness determined the primary and sensitivity roles within the eligible pair.

For runs 479 and 481, case-level quality control reconciled the retained records against the 150-case corpus and reviewed logged completion and technical-error states. Run 481 yielded 295 technically valid records across the two retained domains—148 diagnostic outputs and 147 management records from 149 recorded management attempts—and supplied the primary Doctorina estimates. Run 479 yielded 288 technically valid records—145 diagnostic outputs and 143 management outputs—and was analysed separately for execution-level sensitivity. The patient-visible final recommendation from each eligible execution constituted the evaluated output. Full execution-level completion states are provided in Supplementary Data 10.

The standalone comparators were Gemini 3.1 Pro Preview, Claude Opus 5, GPT-5.6-sol and Kimi K3, executed through the recorded routes vertex\_ai/gemini-3.1-pro-preview, bedrock/eu. anthropic.claude-opus-5, openai/gpt-5.6-sol and moonshot/kimi-k3, respectively. Gemini, Opus and GPT were executed on 3 August 2026, and Kimi on 4 August 2026. Following initial specification of Gemini, Opus and GPT, the final comparator set incorporated Kimi before completion of the standalone-model physician-review block. Each model underwent one corpus-wide execution comprising 150 attempted cases under the identical instruction, “Pretend you’re a doctor and consult a patient.” The execution protocol supplied the same case-specific Polish opening complaint, used the same case-bounded simulated-patient design and required the same four clinical domains, with constructed reference fields reserved for evaluation. The concurrency limit was 150 conversations for Gemini, Opus and GPT and 20 for Kimi.

![](images/46e5b87648ceadd058ffacfeb752d6fd17b38df2bb563f58753dcb3d9b4f4cdf.jpg)  
Fig. 2 Synthetic-case construction and physician–Doctorina consultation architecture. a, Each of the 150 synthetic Polish-language cases comprised three controlled information layers: an opening complaint shown at consultation start, a permitted-facts bank released through questioning and constructed diagnosis and management reference fields withheld for scoring; every field required approval by three clinicians. b, Physicians and Doctorina completed adaptive consultations through the same case-bounded simulated patient and returned a primary diagnosis, diferential diagnoses, diagnostic workup and initial treatment. Eight physicians received 30 cases each; provenance-eligible runs 479 and 481 attempted all 150 cases.

Each case began with its case-specific opening complaint and used a separate conversation history; the logs preserved the accumulated patient-visible dialogue within each case. The execution harness recorded conversation completion according to its chat-state classification. Gemini, Opus and Kimi yielded final outputs for all 150 cases. GPT yielded 148 final outputs; its two remaining attempts reached the 30-pair execution limit and were retained with outcome value 0 in all-attempt analyses. Kimi’s five first-attempt provider rate-limit events were successfully retried within the same corpus execution. The permitted-facts bank and constructed reference fields remained within the simulated-patient and evaluation layers.

## Data capture and quality control

Analyses integrated the case-distribution and packet workbooks, diagnostic and management-rating worksheets, model execution logs and the source case table. A case–respondent ledger linked source values and formulas to identifiers, provenance, completion states, inclusion flags and analytical outcomes. Analyses used Python 3.12.13, NumPy 2.3.5, pandas 2.2.3 and openpyxl 3.1.5.

Physician assignments were reconstructed by matching submitted cases to the five packet-pair sets. Matching produced a retained diagnostic cohort of 240 assigned observations; one additional E1164 response did not correspond to its respondent’s reconstructed assignment and therefore remained outside the analytical cohort. The management cohort comprised 239 observed records spanning all 150 cases, with E1129 contributing to the diagnostic analysis.

Standalone-model datasets were validated for unique case ordinals, complete start and finish markers, opening-complaint agreement and completion-status consistency.

## Physician review and outcomes

Physician and Doctorina outputs comprised one review block and underwent manual evaluation through a resident-physician review process distinct from the eight physician respondents. The review protocol specified one primary evaluator per output, with escalation of uncertain judgments to a senior physician.

Standalone-LLM outputs were evaluated subsequently in a second aligned review block by four physician reviewers using the same case references, endpoint definitions and rating direction. Cases were the allocation unit: the same reviewer evaluated all four standalone-model outputs for a given case. The realized allocation was 104, 26, 15 and 5 cases across the four reviewers. In both blocks, reviewers assessed the retained final outputs against the constructed case references. One final diagnostic judgment, workup verdict and treatment verdict was retained per evaluated output.

## Diagnostic concordance

Top-1 concordance indicated semantic agreement between the submitted most-likely diagnosis and the designated primary diagnosis. Standard abbreviations, clinically equivalent terminology, synonyms and Polish–English equivalents were accepted when they denoted the same condition. For Doctorina, the diagnosis explicitly identified as primary or most likely in the patient-visible terminal recommendation defined the submitted primary diagnosis; for each standalone LLM, the first diagnosis listed in the retained final model output did so. Diagnosis text, rather than a submitted ICD-10 code, governed concordance for all groups.

After a Top-1 mismatch, a secondary binary field recorded whether the submitted primary diagnosis matched a condition in the designated reference diferential. Primary-or-reference-diferential concordance equalled 1 when either field equalled 1. This endpoint measured whether the submitted primary diagnosis belonged to the constructed primary-and-diferential reference set.

## Diagnostic workup and treatment

Diagnostic workup and initial treatment were rated independently of diagnostic correctness on an ordered five-category scale: 1, Excellent; 2, Good; 3, Adequate; 4, Poor; and 5, Very Poor, with lower values representing better performance.

The review guideline anchored Adequate (3) to essential investigations or basic therapy addressing the patient’s main needs, with further refinement possible. Poor (4) denoted important omissions, inappropriate investigations or treatment of questionable appropriateness; Very Poor (5) denoted critical under-investigation or a treatment plan judged highly likely to worsen the patient’s condition. Categories 1–3 therefore represent Adequate-or-better management quality, and categories 4–5 performance below that level.

For n numeric verdicts, the arithmetic mean verdict was transformed to a normalized descriptive score:

$$
{ \mathrm { N o r m a l i z e d ~ s c o r e } } = 1 0 0 \times { \frac { 5 - { \mathrm { m e a n ~ v e r d i c t } } } { 4 } } .
$$

Mean verdicts 1–5 therefore mapped to 100, 75, 50, 25 and 0. Category distributions and medians with interquartile ranges were calculated among numeric physician-reviewer verdicts using the case weights defined below. The normalized score was used for case-standardized group comparisons.

## Technical completeness and analysis populations

A technical error denoted a source-recorded failure state, labelled Error, NONE (Error) or Technical error, for which the afected outcome was classified as technically invalid in the analysis. These states were distinguished from clinically discordant diagnoses, numeric management ratings, consultations that reached the execution limit without a final output and administratively absent records.

All-attempt analyses preserved the planned denominators by assigning outcome value 0 to technical-error and incomplete attempts: diagnostic outcomes were coded 0 for both binary endpoints, and management outcomes received a normalized score of 0. Numeric verdicts were used for category distributions and medians, observed analytical records for available-record means and planned denominators for separately tabulated administrative absences.

## Statistics and reproducibility

The final analytical hierarchy designated the Doctorina–physician Top-1 comparison as primary, primary-or-reference-diferential concordance as secondary, and management and standalone-model comparisons as exploratory. The standalone-model analysis plan was finalized before completion of the standalone-model physician-review block.

The primary analytical comparison was the paired, case-standardized diference between Doctorina run 481 and the physician within-case mean in Top-1 concordance. The analytical sample comprised the complete accepted corpus of 150 cases and all eight completed physician assignments. Each diagnostic comparison included all 150 cases, and management comparisons comprised 149 complete case pairs; E1154 contributed a diagnostic record but not a management record in run 481. Doctorina contributed the run-481 value for each available case, physicians contributed the mean of one or two retained responses and each standalone LLM contributed one output. The Doctorina minus-comparator efect was the mean paired within-case diference, reported in percentage points for diagnostic outcomes and normalized-score points for management outcomes.

Two-sided 95% percentile confidence intervals were estimated from 10,000 paired whole-case bootstrap resamples with seed 20260730. Each resampled case carried all retained physician and system records, preserving the observed reader-by-case and review-block structure. The intervals quantified case-level variation conditional on the observed physician sample, system executions and retained final ratings. Confidence intervals were pointwise, and comparator and exploratory analyses were interpreted descriptively.

Workup and treatment category distributions were case weighted so that every rated case contributed one unit of probability mass. For cases with two physician ratings, each rating contributed one-half of that case’s mass; single-rating cases contributed one complete unit. Doctorina run 481 and each standalone LLM contributed one case-level physician-reviewer verdict when available.

Execution-level sensitivity analysis replaced primary run 481 with the second eligible execution, run 479. All 150 cases contributed to each run-479 sensitivity outcome. Fig. 3 summarizes the observation structure, endpoints and case-standardized estimand.

Additional exploratory analyses characterized case-level robustness and heterogeneity. Poor or very poor management verdicts were defined as categories 4 or 5 and summarized on the 150-case denominator using the case weights and outcome categories defined above. Doctorinacentred win–tie–loss summaries classified each standardized case value as higher than, equal to or lower than the corresponding comparator value. Assignment-stratified consistency was examined by comparing run 481 separately with each physician across that physician’s assigned cases and within the five identifiable packet-pair membership blocks. Complexity-stratified estimates used the source-recorded levels 1, 2 and 3; bootstrap resampling was performed within each stratum, and pairwise diferences between stratum-specific Doctorina–comparator efects were summarized as exploratory interaction contrasts. Diagnostic-error overlap was summarized at case level across the binary outcomes of Doctorina, Gemini, Opus, GPT and Kimi and was crossreferenced with the observed physician responses on cases missed by every AI system. The relation between diagnostic correctness and management performance was summarized using Spearman rank correlations between binary diagnostic outcomes and normalized management scores. Case-level cross-classification used Good-or-Excellent management scores of at least 75 under the all-attempt coding defined above.

Observed agreement between the all-attempt records from runs 479 and 481 was summarized by raw agreement and Cohen’s κ for the binary diagnostic outcomes. Management agreement across all common attempts was summarized by exact agreement of the all-attempt normalized outcomes and paired mean diferences. Within the technically valid subset, agreement was summarized by exact and within-one-verdict-category agreement and paired mean diferences. These measures characterize agreement in the combined execution-and-evaluation record represented by the final retained ratings. Exploratory comparisons used the same bootstrap procedure. Reproducible case-level tables and supplementary analyses are provided in Supplementary Data 10.

Normalized score = 100 × (5 − mean verdict) / 4

## <sup>a</sup> Physician–Doctorina reader-by-case design

PHYSICIAN DIAGNOSTIC RESPONSE BLOCKS
<table><tr><td>Block</td><td>1/2</td><td>3/4</td><td>5/6</td><td>718</td><td>9/10</td></tr><tr><td>P01</td><td></td><td>30</td><td></td><td></td><td></td></tr><tr><td>P02</td><td></td><td></td><td>30</td><td></td><td></td></tr><tr><td>P03</td><td></td><td></td><td></td><td>30</td><td></td></tr><tr><td>P04</td><td>30</td><td></td><td></td><td></td><td></td></tr><tr><td>P05</td><td></td><td></td><td></td><td></td><td>30</td></tr><tr><td>P06</td><td></td><td></td><td></td><td></td><td>30</td></tr><tr><td>P07</td><td></td><td></td><td></td><td>30</td><td></td></tr><tr><td>P08</td><td></td><td>30</td><td></td><td></td><td></td></tr></table>

60 cases had 1 retained physician diagnostic response; 90 had 2  
DOCTORINA EXECUTIONS  
Runs 478 and 480: different software branch

150 cases — Sensitivity

## <sup>b</sup> Operational endpoints and record accounting

DIAGNOSIS ENDPOINTS

Retained binary label indicating primary-diagnosis semantic concordance

After Top-1 = 0, the submitted primary diagnosis matched a condition in the designated reference differential

## MANAGEMENT ENDPOINTS

## Diagnostic-workup domain

Initial-treatment domain

<table><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>Excellent</td><td>Good</td><td>Adequate</td><td>Poor</td><td>Very poor</td></tr></table>

Rated outputs form distributions/medians; failures remain separate

RECORD ACCOUNTING
<table><tr><td rowspan=1 colspan=1>Domain</td><td rowspan=1 colspan=1>Arm</td><td rowspan=1 colspan=1>Record disposition → analyzedcount</td></tr><tr><td rowspan=1 colspan=1>Diagnosis</td><td rowspan=1 colspan=1>Doctorina</td><td rowspan=1 colspan=1>300 eligible attempts; run 481 primary(2 failures), run 479 sensitivity (5)</td></tr><tr><td rowspan=1 colspan=1>Diagnosis</td><td rowspan=1 colspan=1>Physicians</td><td rowspan=1 colspan=1>241 – 1 outside set (E1164) → 240</td></tr><tr><td rowspan=1 colspan=1>Management</td><td rowspan=1 colspan=1>Doctorina</td><td rowspan=1 colspan=1>300 – 1 absent (E1154) → 299; run481: 2 failures, run 479: 7</td></tr><tr><td rowspan=1 colspan=1>Management</td><td rowspan=1 colspan=1>Physicians</td><td rowspan=1 colspan=1>240 present = 239 in-set + 1outside-set E1164; E1129 expectedin-set absent → 239 analyzed</td></tr></table>

## <sup>c</sup> Six-group case-standardized contrasts and uncertainty

Case-standardized all-attempt analysis: diagnostic failures coded 0; management failures assigned normalized score 0; administrative absences omitted

![](images/c3acbe688b8fbc7180bfca958e1659215daf15afa04616a9255794e2b457885a.jpg)  
Fig. 3 Reader-by-case observation structure and case-standardized estimand. a, Run 481 served as the primary Doctorina execution and run 479 as the execution-level sensitivity analysis. The physician cohort contributed 240 diagnostic observations; 90 cases had two responses and 60 had one. b, Diagnostic endpoints compared the submitted primary diagnosis with the constructed reference diagnosis set; workup and treatment were rated on the five-category scale and transformed to normalized scores. c, Each case contributed equally to Doctorina-minus-comparator contrasts, with physicians represented by their within-case mean. Uncertainty was estimated using 10,000 paired whole-case bootstrap resamples.

## Data availability

The deidentified analytical data accompanying this article are provided as Supplementary Data 1–10. Supplementary Data 1–4 contain case metadata, physician allocation and coded Doctorina–physician and standalone-model rating ledgers; Supplementary Data 5–9 contain the case-standardized analysis ledger, eligible-execution sensitivity results, arm estimates, contrasts and verdict distributions; and Supplementary Data 10 contains validated robustness, heterogeneity, overlap and observed run-agreement tables. Deidentified source data for Fig. 1 and Supplementary Fig. 1 are included with the figure package. Researchers seeking to reproduce or verify the analyses may request access to retained case narratives, dialogue outputs and execution records from the corresponding author. Access is limited to the records required for the stated verification purpose and is subject to a signed confidentiality and data-use agreement and applicable data restrictions.

## Code availability

Supplementary Code 1 reproduces the primary six-group arm estimates and Doctorina-centred paired-bootstrap contrasts. Supplementary Code 2 validates the public data files, workbook structure and file hashes. Reproducible generation code for Supplementary Fig. 1 and validation code for Fig. 1 and Table 1 are included with the manuscript source package.

## Acknowledgements

The authors thank the physicians who authored and reviewed the synthetic clinical cases, the eight participating physicians, the independent session observer and the physician reviewers who evaluated the study outputs. The authors also thank Shubhanan Upadhyay for independent scientific and methodological advice and critical review of the manuscript. This study received no external grant funding.

## Author contributions

H.P.: Conceptualization, methodology, resources, supervision and project administration. S.S.: Investigation, data curation, project administration, resources and formal analysis. A.K.: Software, data curation, formal analysis and methodology. A.N.: Formal analysis, visualization, writing— original draft, and writing—review and editing. P.G.: Methodology, validation, investigation, resources, project administration and supervision. J.M.: Methodology, validation, investigation, data curation and formal analysis. V.H.: Methodology, validation, investigation and data curation. A.R.: Software and resources. P.S.: Software and resources.

## Competing interests

H.P. is a co-founder and Chief Medical Oficer of Doctorina. S.S. is Clinical Lead at Doctorina. A.K. contributed to the technical development of Doctorina and the study infrastructure. A.N. is an afiliated medical expert at Doctorina. P.G., J.M. and V.H. declare no competing interests. A.R. and P.S. contributed to the technical development of Doctorina and the study infrastructure.

## References

[1] Singhal, K. et al. Large language models encode clinical knowledge. Nature 620, 172–180 (2023).

[2] Takita, H. et al. A systematic review and meta-analysis of diagnostic performance comparison between generative AI and physicians. npj Digit. Med. 8, 175 (2025).

[3] Hager, P. et al. Evaluation and mitigation of the limitations of large language models in clinical decision-making. Nat. Med. 30, 2613–2622 (2024).

[4] Johri, S. et al. An evaluation framework for clinical use of large language models in patient interaction tasks. Nat. Med. 31, 77–86 (2025).

[5] Tu, T. et al. Towards conversational diagnostic artificial intelligence. Nature 642, 442–450 (2025).

[6] McDuf, D. et al. Towards accurate diferential diagnosis with large language models. Nature 642, 451– 457 (2025).

[7] Gilbert, S. et al. How accurate are digital symptom assessment apps for suggesting conditions and urgency advice? A clinical vignettes comparison to GPs. BMJ Open 10, e040269 (2020).

[8] Liu, Y. et al. Benchmarking large language model-based agent systems for clinical decision tasks. npj Digit. Med. 9, 259 (2026).

[9] Vishwanath, K. et al. General-purpose large language models outperform specialized clinical AI tools on medical benchmarks. Nat. Med. 32, 2405–2409 (2026).

[10] Grzybowski, Ł., Pokrywka, J., Ciesiółka, M., Kaczmarek, J. I. & Kubis, M. Polish-English medical knowledge transfer: a new benchmark and results. In Findings of the Association for Computational Linguistics: EMNLP 2025, 9042–9063 (Association for Computational Linguistics, 2025).

[11] Ibrahim-El-Nur, J. M., Kaczmarek, W., Winiarska, W., Kuś, A. & Łoś, M. Language-specific diferences in large language model diagnostic reasoning: a translation-controlled clinical vignette study. J. Clin. Med. 15, 4082 (2026).

[12] Goh, E. et al. Large language model influence on diagnostic reasoning: a randomized clinical trial. JAMA Netw. Open 7, e2440969 (2024).

[13] Bossuyt, P. M. et al. STARD 2015: an updated list of essential items for reporting diagnostic accuracy studies. BMJ 351, h5527 (2015).

[14] Sounderajah, V. et al. The STARD-AI reporting guideline for diagnostic accuracy studies using artificial intelligence. Nat. Med. 31, 3283–3289 (2025).

[15] Gallifant, J. et al. The TRIPOD-LLM reporting guideline for studies using large language models. Nat. Med. 31, 60–69 (2025).

[16] National Health Fund (Narodowy Fundusz Zdrowia). Liczba pacjentów z udzielonym w danym roku świadczeniem, które zostało sprawozdane do Narodowego Funduszu Zdrowia z danym kodem rozpoznania głównego świadczenia wg klasyfikacji ICD-10 (do poziomu 5 znaków) dla 2024 roku. Polish Open Data Portal, resource 65397 (2025). https://dane.gov.pl/pl/dataset/2557/resource/65397 (accessed 1 September 2026).