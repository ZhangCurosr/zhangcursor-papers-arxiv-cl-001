# OCR-MetaReasoning Benchmark: Evaluating the Meta-Reasoning Ability of MLLMs in Text-Rich Image Understanding

Gengxu Li<sup>1</sup> Yuan Wu<sup>1</sup>\* Yi Chang<sup>1,2,3</sup>

<sup>1</sup>School of Artificial Intelligence, Jilin University

<sup>2</sup>Engineering Research Center of Knowledge-Driven Human-Machine Intelligence, MOE, China <sup>3</sup>International Center of Future Science, Jilin University gxli25@mails.jlu.edu.cn, yichang@jlu.edu.cn, yuanwu@jlu.edu.cn

## Abstract

Text-rich image understanding requires multimodal large language models (MLLMs) to organize OCR (Optical Character Recognition)- grounded evidence across words, layout, fields, charts, and visual correspondences. Existing evaluations often conflate extraction with reasoning and rarely test whether models follow the required reasoning direction: applying visible rules, abstracting hidden regularities, or recovering missing premises. We introduce OCR-MetaReasoning, a controlled single-image benchmark that treats deduction, induction, and abduction as distinct directions and separates final-answer correctness from reasoning-process compliance. The benchmark contains 1,500 verified samples in a balanced 3 × 5 taxonomy crossing three reasoning types with five OCR-object categories, along with reference reasoning steps, automatic answer scoring, the Meta-Reasoning Macro Score (MRMS), and the Reasoning Process Compli ance Score (RPCS). Experiments with representative closed-source and open-source MLLMs show that OCR-grounded meta-reasoning remains far from saturated: models struggle with visible-rule application and layout-sensitive inference, while process-compliant rationales can accompany incorrect final answers under exact-match evaluation. The code is available at https://github.com/gengxuli/ OCR-MetaReasoning.

## 1 Introduction

Recent advances in multimodal large language models (MLLMs) have substantially improved visual question answering, document understanding, chart interpretation, and instruction following for mixed visual-textual inputs (Team, 2025; Hu et al., 2025a). Among these settings, text-rich image understanding is particularly important because receipts, forms, tables, notices, posters, webpages, and infographics encode meaning through recognized words, layout, fields, constraints, visual groupings, and cross-region correspondences. A capable MLLM must do more than read OCR (Optical Character Recognition) tokens; it must organize visible evidence for the reasoning direction required by each question.

![](images/ad547edfcdaf3082523a43c290d1d68fb0d99493c946f178d459eac7baf64fc7.jpg)  
Figure 1: OCR-grounded abduction in OCR-MetaReasoning. The model infers the hidden intention behind “9\$7\$1” from the keyboard layout, contrasting local symbol matching with layout-aware reasoning.

Despite this progress, reasoning over OCRgrounded evidence remains difficult (Li et al., 2025a; Chen et al., 2024). Beyond visible-text extraction, models must determine how recognized words, layout, fields, and visual correspondences should be used to support an answer. Many questions therefore require meta-reasoning: a model may need to apply an explicit rule to a case, infer a latent rule from aligned observations, or recover a missing premise that explains an observed outcome. These directions correspond to deductive, inductive, and abductive reasoning, respectively, with distinct failure modes. Models may copy salient text, rely on shallow field matching, or produce plausible explanations weakly grounded in the image. Appendix A illustrates this distinction. Final-answer accuracy alone cannot diagnose whether a model follows the intended reasoning process (Chen et al., 2024; Chang et al., 2024).

Recent benchmarks address this challenge. OCRBench v2 evaluates visual text localization and reasoning across broad text-centric scenarios (Fu et al., 2025); LogicOCR, Reasoning-OCR, and OCR-Reasoning build challenging logical or complex reasoning tasks from OCR (Ye et al., 2025; He et al., 2025; Huang et al., 2025); and LogicVista, VisualPuzzles, and MME-Reasoning examine visual logical reasoning, knowledge-light puzzles, and explicit deductive, inductive, and abductive reasoning in MLLMs (Xiao et al., 2024; Song et al., 2025; Yuan et al., 2025). Other work studies difficulty grading, multi-image reasoning, visual chain-of-thought supervision, longchain visual reasoning, region-level prompting, and cross-modal reasoning-chain verification (Li et al., 2025b; Cheng et al., 2025; Shao et al., 2024; Zhou et al., 2025; Dong et al., 2025; Cai et al., 2024; Zhang et al., 2025). However, existing benchmarks are organized mainly by OCR task, document domain, visual puzzle type, or general logical ability. They do not make OCR-grounded meta-reasoning the central target while jointly controlling reasoning direction, OCR-object category, final-answer correctness, and process-level compliance.

We introduce OCR-MetaReasoning, a benchmark for evaluating MLLM meta-reasoning in textrich image understanding. Each sample makes the required reasoning direction explicit, so models are evaluated on reading relevant text and organizing OCR-grounded evidence for deduction, induction, or abduction. This setup separates OCR perception from reasoning and supports fine-grained diagnosis across heterogeneous text-rich visual scenarios.

OCR-MetaReasoning uses a balanced 3 × 5 taxonomy crossing three meta-reasoning types with five OCR-object categories: transaction analysis, data interpretation, field dependency, document logic, and layout semantics. This balance prevents strong performance on a familiar document type or reasoning pattern from masking localized weaknesses. It also provides reference reasoning steps and an evaluation protocol separating finalanswer correctness from reasoning-process compliance. The benchmark contains 1,500 single-image samples, an automatic answer scorer, and processlevel criteria for capability match, groundedness, step completeness, and non-hallucination.

Contributions. Our main contributions are summarized as follows:

• We introduce OCR-MetaReasoning, a benchmark centered on OCR-grounded metareasoning, with balanced coverage of deductive, inductive, and abductive reasoning across diverse text-rich image objects.

• We develop a unified construction and evaluation protocol that combines MLLM-assisted synthesis, human verification, reference reasoning steps, the Meta-Reasoning Macro Score (MRMS), and the Reasoning Process Compliance Score (RPCS).

• We evaluate representative closed-source and open-source MLLMs and identify persistent weaknesses in grounded rule verification and layout-sensitive inference, showing that current models remain far from saturated on OCRgrounded meta-reasoning.

## 2 Related Work

## 2.1 OCR Reasoning Benchmarks

Text-rich visual benchmarks have progressed from reading-oriented VQA tasks, including TextVQA, ST-VQA, and DocVQA (Singh et al., 2019; Biten et al., 2019; Mathew et al., 2021), to structured reasoning over charts, documents, and OCR-heavy images (Masry et al., 2022; Liu et al., 2024; Fu et al., 2025). Recent benchmarks examine logical reasoning from OCR cues: LogicOCR and Reasoning-OCR construct text-rich reasoning tasks (Ye et al., 2025; He et al., 2025), while OCR-Reasoning introduces challenging single-image samples with process annotations to reduce answer leakage from direct text extraction (Huang et al., 2025). Table 1 summarizes the coverage of OCR grounding, reasoning types, process evaluation, and object diversity. These resources improve OCR-centered evaluation, but they remain organized mainly by domain, skill, or scenario. By contrast, OCR-MetaReasoning uses the dominant reasoning direction as its primary axis and requires each textrich sample to instantiate deduction, induction, or abduction over OCR-grounded evidence.

<table><tr><td>Benchmark</td><td>Ability</td><td>Text-rich OCR OCR Meta-Reason. D/I/A Process Eval. Object Diversity</td><td></td><td></td><td></td><td></td></tr><tr><td>OCRBench v2 (Fu et al., 2025)</td><td>Visual text localization and reasoning</td><td>√</td><td>△</td><td>X</td><td>X</td><td>V</td></tr><tr><td>LogicVista (Xiao et al., 2024)</td><td>Visual logical reasoning</td><td>X</td><td>X</td><td>△</td><td>△</td><td>X</td></tr><tr><td>VisualPuzzles (Song et al., 2025)</td><td>Knowledge-light multimodal reasoning</td><td>X</td><td>X</td><td>△</td><td>X</td><td>X</td></tr><tr><td>LogicOCR (Ye et al., 2025)</td><td>Logical reasoning on text-rich images</td><td>√</td><td>△</td><td>△</td><td>X</td><td>△</td></tr><tr><td>MME-Reasoning (Yuan et al., 2025)</td><td>Multimodal logical reasoning</td><td>×</td><td>△</td><td>V</td><td>X</td><td>X</td></tr><tr><td>Reasoning-OCR (He et al., 2025)</td><td>Logical reasoning from OCR cues</td><td>V</td><td>△</td><td>X</td><td>X</td><td>△</td></tr><tr><td>OCR-Reasoning (Huang et al., 2025)</td><td>Complex text-rich image reasoning</td><td>√</td><td>△</td><td>X</td><td>V</td><td>△</td></tr><tr><td>OCR-MetaReasoning (Ours)</td><td>OCR-grounded meta-reasoning</td><td>√</td><td>V</td><td>V</td><td>V</td><td>V</td></tr></table>

Table 1: Comparison with related OCR, text-rich visual reasoning, and multimodal logical reasoning benchmarks. “OCR Meta-Reason.” marks OCR-grounded meta-reasoning as the central target; $\mathrm { ^ { 6 6 } D } / \mathrm { I } / \mathrm { A } ^ { \prime \prime }$ marks explicit deductive, inductive, and abductive coverage; “Process Eval.” marks process-level annotation or evaluation. $\checkmark , \triangle ,$ and ×indicate full, partial, and absent coverage.

Our quantitative construct audit covers six of the related benchmarks; VisualPuzzles is included in Table 1 for conceptual comparison only and is not included in the quantitative audit. LogicOCR-Real denotes the real-image subset of Logic-OCR. The projection maps 205/300 LogicOCR-Real items, 457/1,069 OCR-Reasoning items, and 69/150 Reasoning-OCR items cleanly to our metareasoning and OCR-object axes; corresponding coverage is 48/300 for OCRBench v2, 52/300 for LogicVista, and 0/300 for MME-Reasoning. Rerunning the same eight model versions on selected subsets yielded Spearman correlations of 0.905, 0.695, and 0.405 on OCR-Reasoning, LogicOCR-Real, and Reasoning-OCR, respectively. These results show that the benchmarks are related but do not have identical construct coverage: the external resources do not provide the complete controlled (3×5) grid together with our process-compliance evaluation. The full projection and diagnostic results are reported in Appendix J.5.

## 2.2 Multimodal Reasoning Evaluation

Other benchmarks evaluate multimodal reasoning beyond OCR-centric settings. LogicVista studies logical reasoning in visual contexts (Xiao et al., 2024), VisualPuzzles reduces reliance on domain knowledge through knowledge-light visual puzzles (Song et al., 2025), and MME-Reasoning explicitly covers deductive, inductive, and abductive reasoning in MLLMs (Yuan et al., 2025). The Beyond “Aha!” framework motivates our formulation by treating deduction, induction, and abduction as meta-abilities under a unified view of hypotheses, rules, and observations (Hu et al., 2025b). These benchmarks treat reasoning as an explicit evaluation target, but they do not focus on the evidence structure of OCR-rich images, including fields, tables, receipts, forms, footnotes, layout groupings, and cross-region textual dependencies. OCR-MetaReasoning combines OCR-grounded evidence with explicit coverage of deduction, induction, and abduction, process-level evaluation, and diverse OCR-object categories.

## 3 OCR-MetaReasoning

## 3.1 Task Definition

OCR-MetaReasoning tests whether an MLLM organizes evidence from a text-rich image according to the question’s required reasoning direction rather than merely extracting visible text. Given a single image I and a question q, the model receives $x = ( I , q )$ and produces a solution process π and a final answer y. Relevant evidence may include recognized text, tables, chart marks, fields, checkboxes, layout groups, footnotes, and crossregion correspondences. We frame meta-reasoning in terms of hypotheses, rules, and observations (Hu et al., 2025b): H denotes a hypothesis, premise, or candidate state; R denotes a rule, constraint, mapping, or regularity; and O denotes an observation, outcome, or consequence grounded in the image.

The benchmark assigns each sample a dominant reasoning type, determined by the critical bottleneck in its solution path. Meta-deductive reasoning follows $H + R  O$ The model must bind explicit rules, clauses, formulas, legends, thresholds, or field dependencies to image evidence, test a candidate case, and derive the conclusion supported by the rule. Examples include verifying invoice totals under a printed tax rule and deciding eligibility from visible conditions and exception clauses.

Meta-inductive reasoning follows $H + O  R .$ The model must align multiple visible instances, abstract an unstated rule, and apply it to a missing, unlabeled, future, or hypothetical target. Examples include inferring a table-filling pattern, a legend-tolayout mapping, or a recurring field schema.

![](images/bca55f92c4c5a14108265b1c430e2e0cd2658cddaa0dcabbc93fb5eb54aafc90.jpg)  
Figure 2: Overview of OCR-MetaReasoning construction. We collect text-rich seeds, synthesize OCR-grounded samples with MLLMs, assess candidates using taxonomy-aligned checklist criteria, and organize valid samples by meta-reasoning type and OCR-object category. We regenerate prompts or images when necessary.

Meta-abductive reasoning follows O + R → H. The model must start from an observed result, anomaly, goal, or downstream state and reason backward through visible constraints to recover the unique or minimal hidden premise. Examples include identifying a hidden discount from a final total and recovering the missing condition that explains an approval state.

When a problem combines several local operations, we assign its label according to the target variable requiring the stated direction. For example, a receipt item asking for a masked discount from a visible final total is abductive even if its solution contains arithmetic, because the answer is a hidden premise rather than a result computed in the forward direction.

## 3.2 Benchmark Construction

Two-axis taxonomy As shown in Figure 2, OCR-MetaReasoning uses a 3 × 5 taxonomy with two axes: meta-reasoning type (deductive, inductive, and abductive) and OCR-object category (transaction analysis, data interpretation, field dependency, document logic, and layout semantics). The categories cover receipts and invoices from CORD (Park et al., 2019) and WildReceipt (Sun et al., 2021); tables and charts from ChartQA (Masry et al., 2022) and ChartXiv (Wang et al., 2024); forms and certificates from FUNSD (Jaume et al., 2019) and DocVQA (Mathew et al., 2021); notices and policy-like documents from DocVQA (Mathew et al., 2021) and InfoVQA (Mathew et al., 2022); and posters, webpages, and infographics from InfoVQA (Mathew et al., 2022) and TextVQA (Singh et al., 2019). The final benchmark contains 1,500 samples, with 100 for each type-category combination. This balance supports comparison across reasoning directions while controlling for text-rich visual content. Appendix D reports benchmark statistics.

Seed collection We collect text-rich image seeds from public OCR-oriented resources and create controlled edited/reconstructed variants of those seeds. The seeds cover receipts, forms, charts, notices, posters, webpages, and infographics. We screen them for text density, legibility, structural richness, and cross-region or cross-field relations. We discard images solvable by copying one visible span or reading one field without understanding the question.

MLLM-assisted synthesis For each retained seed, an MLLM creates or modifies a sample consisting of a text-rich image, a question, a canonical answer, and reference reasoning steps. The prompt specifies the target meta-reasoning type and OCRobject category. It also makes the reasoning bottleneck explicit: deductive samples contain a visible rule or constraint and a candidate to verify; inductive samples contain multiple alignable support instances and a generalization target; and abductive samples contain an observed result or anomaly and enough constraints for backward recovery. When an initial sample can be solved by direct OCR extraction, prompt or image modification introduces distractors, masked fields, exception clauses, nonadjacent evidence, or layout correspondences.

Human verification Three PhD students with research experience in multimodal large language models performed construction verification. They saw the proposed meta-reasoning and OCR-object labels because this was taxonomy-aligned candidate assessment. A valid sample had to satisfy five checklist conditions: the primary evidence had to be grounded in the image; at least two evidence points had to be required; the answer had to be unique or uniquely minimal; the declared metareasoning type had to match the dominant bottleneck; and the final answer had to be automatically scorable after normalization. Candidates failing any condition were rejected or filtered. When a candidate resembled an existing task, we generated a new candidate through separate prompt or image regeneration and counted it separately.

We separately measured label reliability with a blinded re-annotation of 300 final samples, stratified to 20 samples per cell across 15 cells; the released labels were hidden during this study. Independent agreement was 96.0% raw (Fleiss $\kappa = 0 . 9 6 0$ , Krippendorff $\alpha = 0 . 9 6 0 )$ for metareasoning type and 92.7% raw $( \kappa = 0 . 9 3 9 , \alpha =$ 0.939) for OCR-object category. After adjudication, all 300 validated samples satisfied the validity checklist and matched the released labels on both axes. Construction yielded 7,652 generated candidates, $^ { 6 , 8 7 4 }$ successful MLLM syntheses, 2,326 verified pass candidates, and 1,500 selected samples. Appendix J reports the complete per-type summary and confirms 100 samples in every taxonomy cell.

We use edited/reconstructed for final images whose content does not exactly match the samenamed image in the selected source collection. These are controlled edits of public OCR seeds, such as masking a field, inserting a distractor, changing a local value, or rebuilding a document-like layout while preserving OCRgrounded evidence, not unconstrained images generated from scratch. The provenance analysis shows that 1,299/1,500 images are unmodified and 201/1,500 are edited/reconstructed; all 201 edited/reconstructed images are in the abductive split (Appendix J.7).

## 3.3 Evaluation Protocol

Model input and answer scoring During evaluation, each model receives the image and question and is instructed to produce numbered reasoning steps followed by a final-answer line. We measure final-answer correctness with a sample-specific scorer: normalized exact match for short strings, normalized numeric match for integer or floatingpoint answers, and micro-F1 for structured answers. We report scores for the three meta-reasoning types, five OCR-object categories, and the overall macro metric.

Meta-Reasoning Macro Score The primary correctness metric is the Meta-Reasoning Macro Score (MRMS), averaging performance across the three reasoning directions:

$$
M R M S = \frac { A c c _ { \mathrm { D e d } } + A c c _ { \mathrm { I n d } } + A c c _ { \mathrm { A b d } } } { 3 } .
$$

Here, $A c c _ { \mathrm { D e d } } , A c c _ { \mathrm { I n d } }$ , and $A c c _ { \mathrm { A b d } }$ are mean finalanswer scores on deductive, inductive, and abductive samples, respectively. The macro average prevents strong performance in one direction from masking weakness in another.

Reasoning Process Compliance Score Because final-answer accuracy alone cannot determine whether a model follows the intended direction, we report the Reasoning Process Compliance Score (RPCS) as a separate process-level metric. RPCS evaluates the visible solution process using four binary criteria: capability match $( c _ { \mathrm { m a t c h } } )$ , indicating whether the dominant process follows the target direction represented by $H , R , O ;$ groundedness $( c _ { \mathrm { g r o u n d } } )$ , indicating whether key claims are tied to image evidence or question constraints; step completeness $( c _ { \mathrm { s t e p } } )$ , indicating whether necessary intermediate nodes are covered; and nonhallucination $( c _ { \mathrm { n o n h a l l } } )$ , indicating whether the process avoids unsupported rules, entities, fields, or values. Raw RPCS is their sum, and normalized RPCS lies in [0, 1]:

$$
R P C S = \frac { c _ { \mathrm { m a t c h } } + c _ { \mathrm { g r o u n d } } + c _ { \mathrm { s t e p } } + c _ { \mathrm { n o n h a l l } } } { 4 } .
$$

Both MRMS and RPCS use the normalized [0, 1] scale. Unless otherwise noted, aggregate MRMS, RPCS, and criterion rates in the main text and result tables are reported as percentage points (100 times the normalized score), whereas sample-level scores and difficulty thresholds remain in [0, 1]. We score RPCS against each sample’s reference reasoning steps and image context without requiring exact wording or identical segmentation. Compressed or paraphrased reasoning receives credit if it preserves the required evidence bindings and direction. MRMS and RPCS are reported separately to distinguish answer-level success from process-level compliance.

The RPCS judge is GPT-5.4 at temperature 0.0. It receives the image, question, target labels, reference reasoning steps, and the model process with the standalone final-answer line removed; the gold answer, predicted answer, and MRMS score are withheld. On 300 balanced process pairs independently rated by the same three PhD annotators, human inter-rater agreement was 90.7% raw with macro Fleiss $\kappa = 0 . 8 6 0$ . The main judge reached Pearson r = 0.934, Spearman $\rho = 0 . 9 3 1$ , criterion macro-F1 = 0.936, κ = 0.873, and RPCS MAE = 0.050 against the human majority. A rerun had a 3.1% criterion flip rate and a 0.040 RPCS standard deviation. We therefore use RPCS as a validated diagnostic, while MRMS remains the judge-independent primary leaderboard metric; the exact prompt and complete calibration tables appear in Appendix J.2.

Appendix F describes the remaining evaluation details.

## 4 Experiment

## 4.1 Setup

We evaluate representative closed-source and opensource models. The closed-source set includes GPT-5.4-Mini and its medium variant (OpenAI, 2026b), GPT-5.4 and its medium variant (OpenAI, 2026a), Gemini-3.1-Flash-Lite (Google, 2026a), Gemini-3.1-Pro-Preview (Google, 2026b), Claude-Sonnet-4.6 (Anthropic, 2026), Grok-4.20 (nonreasoning) (xAI, 2026), and Doubao-Seed-2.0-Lite and Pro (ByteDance, 2026). The open-source set includes Kimi-K2.5 (Instant mode) (MoonshotAI, 2025), GLM-4.6V (ZaiOrg, 2025), and Qwen3-VL at 8B, 30B-A3B, and 235B-A22B, with both Instruct and Thinking variants (Qwen, 2025e,f,c,d,a,b). Appendix C gives model descriptions and decoding settings.

## 4.2 Main Results

Leaderboard and Saturation Table 2 and Figure 3 show a clear ranking and considerable headroom on OCR-MetaReasoning. Gemini-3.1-Pro-Preview achieves the highest MRMS, at 89.3, followed by GPT-5.4-Medium at 87.7 and Doubao-Seed-2.0-Pro at 86.4. The strongest open-source model, Qwen3-VL-235B-A22B-Thinking, reaches 81.5, 7.8 points below the leading model. Closedsource models average 82.9 MRMS versus 73.4 for open-source models, a 9.5-point gap. The heatmap shows similar ordering across most columns.

We computed 95% confidence intervals with 1,000 sample-level bootstrap replicates over all 1,500 samples (seed 42). Gemini-3.1-Pro-Preview, GPT-5.4-Medium, and Doubao-Seed-2.0-Pro have MRMS values of 89.3 [87.9, 90.8], 87.7 [86.0, 89.3], and 86.4 [84.7, 88.0], respectively. Paired bootstrap intervals exclude zero for the gap between the top two models (+1.7 [0.1, 3.1]), the gap between the top closed-source and best opensource models (+7.8 [6.1, 9.6]), and the gap between the closed-source and open-source averages (+9.5 [8.5, 10.5]). Layout semantics is 14.7 points below the non-layout average (95% CI [-18.8, - 10.8]). The benchmark is not saturated: 214 items have a mean score below 0.5, 118 below 0.25, and only 453 are solved with full score by all 18 models (Appendix J.3).

Performance Varies Across OCR-Object Categories Performance varies across the five OCRobject categories. Across all models, data interpretation has the highest average score, at 83.3, followed by field dependency at 82.0, document logic at 81.3, and transaction analysis at 80.0. Layout semantics averages only 66.9, which is 13.1 points below the next-lowest category. Among leading systems, GPT-5.4-Medium achieves the best layout semantics score, at 77.6, while Gemini-3.1-Pro-Preview and Doubao-Seed-2.0-Pro reach 76.2 and 75.1, respectively. The same models score above 87.5 on most non-layout categories, indicating that the benchmark distinguishes OCR text extraction and field-level reasoning from layout-sensitive inference over visual grouping and cross-region semantics.

Deduction Is the Main Reasoning Bottleneck Scores by reasoning type reveal a difficulty pattern distinct from object categories. Averaged across all models, inductive reasoning reaches 82.5 and abductive reasoning reaches 80.1, whereas deductive reasoning reaches only 73.6. Among the three highest-MRMS models, deductive scores range from 80.1 to 80.7, while inductive scores range from 90.2 to 93.1 and abductive scores range from 88.9 to 94.1. Because deductive samples require candidate verification against visible rules or constraints, current MLLMs appear better at aggregating repeated evidence or recovering plausible causes than at consistently applying OCRgrounded rules.

<table><tr><td rowspan="2">Model</td><td colspan="4">OCR Object</td><td rowspan="2"></td><td colspan="2">Reasoning Type</td><td rowspan="2">ABD.</td><td rowspan="2">MRMS</td></tr><tr><td colspan="6">D.I. D.L. F.D. L.S. T.A. DED. IND.</td></tr><tr><td></td><td>Closed-source Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>96.0</td><td>93.0</td><td>91.7</td><td>76.2</td><td>89.7</td><td>80.7</td><td>93.1</td><td>94.1</td><td>89.3</td></tr><tr><td>GPT-5.4-Medium</td><td>92.3</td><td>90.8</td><td>90.0</td><td>77.6</td><td>87.5</td><td>80.6</td><td>91.5</td><td>90.8</td><td>87.7</td></tr><tr><td>Doubao-Seed-2.0-Pro</td><td>91.7</td><td>87.8</td><td>87.9</td><td>75.1</td><td>89.4</td><td>80.1</td><td>90.2</td><td>88.9</td><td>86.4</td></tr><tr><td>Claude-Sonnet-4.6</td><td>88.5</td><td>88.1</td><td>89.1</td><td>69.9</td><td>86.0</td><td>77.5</td><td>87.6</td><td>87.9</td><td>84.3</td></tr><tr><td>Doubao-Seed-2.0-Lite</td><td>86.8</td><td>86.3</td><td>88.3</td><td>72.3</td><td>87.4</td><td>76.7</td><td>89.2</td><td>86.8</td><td>84.2</td></tr><tr><td>GPT-5.4-Mini-Medium</td><td>86.7</td><td>87.8</td><td>87.2</td><td>73.6</td><td>84.7</td><td>77.7</td><td>87.2</td><td>87.1</td><td>84.0</td></tr><tr><td>Gemini-3.1-Flash-Lite</td><td>87.7</td><td>84.9</td><td>87.8</td><td>72.2</td><td>83.8</td><td>78.4</td><td>85.4</td><td>86.2</td><td>83.3</td></tr><tr><td>GPT-5.4</td><td>87.4</td><td>85.0</td><td>88.0</td><td>69.1</td><td>85.7</td><td>77.9</td><td>85.7</td><td>85.5</td><td>83.0</td></tr><tr><td>Grok-4.20</td><td>79.3</td><td>76.6</td><td>79.8</td><td>61.9</td><td>78.3</td><td>70.3</td><td>77.4</td><td>78.0</td><td>75.2</td></tr><tr><td>GPT-5.4-Mini</td><td>72.7</td><td>75.1</td><td>76.9</td><td>61.0</td><td>74.4</td><td>70.7</td><td>75.3</td><td>70.1</td><td>72.0</td></tr><tr><td colspan="10">Open-source Models</td></tr><tr><td>Qwen3-VL-235B-A22B-Thinking</td><td>87.3</td><td>86.3</td><td>82.7</td><td>69.2</td><td>82.2</td><td>74.8</td><td>87.8</td><td>82.1</td><td>81.5</td></tr><tr><td>Kimi-K2.5</td><td>86.5</td><td>77.5</td><td>81.2</td><td>70.2</td><td>82.8</td><td>71.8</td><td>84.6</td><td>82.5</td><td>79.6</td></tr><tr><td>Qwen3-VL-235B-A22B-Instruct</td><td>80.7</td><td>78.8</td><td>80.7</td><td>69.5</td><td>75.5</td><td>72.5</td><td>80.9</td><td>77.7</td><td>77.0</td></tr><tr><td>GLM-4.6V</td><td>83.6</td><td>78.1</td><td>79.3</td><td>60.3</td><td>77.9</td><td>72.0</td><td>79.2</td><td>76.3</td><td>75.9</td></tr><tr><td>Qwen3-VL-30B-A3B-Thinking</td><td>81.5</td><td>78.6</td><td>77.8</td><td>60.5</td><td>75.1</td><td>69.1</td><td>81.8</td><td>73.3</td><td>74.7</td></tr><tr><td>Qwen3-VL-8B-Thinking</td><td>80.7</td><td>76.8</td><td>77.3</td><td>60.6</td><td>76.8</td><td>70.3</td><td>79.5</td><td>73.6</td><td>74.5</td></tr><tr><td>Qwen3-VL-30B-A3B-Instruct</td><td>72.1</td><td>66.6</td><td>67.0</td><td>55.1</td><td>64.4</td><td>64.0</td><td>67.9</td><td>63.2</td><td>65.0</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>58.3</td><td>65.5</td><td>62.4</td><td>49.9</td><td>59.1</td><td>59.1</td><td>60.5</td><td>57.6</td><td>59.0</td></tr></table>

Table 2: MLLM performance on OCR-MetaReasoning. The three highest results in each column are highlighted in blue. D.I., D.L., F.D., L.S., and T.A. denote data interpretation, document logic, field dependency, layout semantics, and transaction analysis; DED., IND., and ABD. denote deductive, inductive, and abductive meta-reasoning. MRMS is the Meta-Reasoning Macro Score. All scores are percentages (100 times the normalized score).

Reasoning Variants Help Unevenly Within Qwen3-VL, Thinking variants improve MRMS over Instruct variants, with larger gains at smaller scales: +15.5 at 8B, +9.7 at 30B-A3B, and +4.5 at 235B-A22B. The trend varies by category: Qwen3- VL-235B-A22B-Thinking improves on all three reasoning types and most OCR-object categories, but scores 0.3 points lower than its Instruct counterpart on layout semantics. Thus, gains are larger at smaller scales and do not eliminate the layoutsensitive bottleneck.

Closed-Source Models Have a Broad but Uneven Advantage The closed-source advantage appears across OCR-object categories and reasoning types, although the column leaders differ. Gemini-3.1- Pro-Preview ranks first in MRMS and leads every column except layout semantics, where GPT-5.4-Medium achieves the highest score, at 77.6. Claude-Sonnet-4.6 also ranks among the top three for document logic and field dependency despite ranking fourth in MRMS. On average, closedsource models outperform open-source models by 9.5 MRMS points, with the largest reasoning-type gap for abduction, at 12.3 points, and the smallest for deduction, at 7.9 points. Closed-source models perform better overall, but their strengths vary across OCR-object categories.

The Balanced Taxonomy Separates Model Profiles The balanced taxonomy prevents strong performance on one axis from masking localized weaknesses. Qwen3-VL-235B-A22B-Thinking, the strongest open-source model, reaches 81.5 MRMS and 87.8 on induction, but its scores of 74.8 on deduction and 69.2 on layout semantics keep it behind the leading closed-source models. Kimi-K2.5 exhibits a different profile: it leads the opensource models on layout semantics (70.2), transaction analysis (82.8), and abduction (82.5), but its lower scores on document logic (77.5) and deduction (71.8) limit its MRMS to 79.6. This comparison shows why both axes are needed: models must align visual text evidence with the required reasoning direction across heterogeneous OCR-object categories.

<table><tr><td>Gemini-3.1-Pro Preview</td><td>96.0</td><td>93.0</td><td>91.7</td><td>76.2</td><td>89.7</td><td>80.7</td><td>93.1</td><td>94.1</td><td>89.3</td><td rowspan="8">100 90 80 70</td></tr><tr><td>GPT-5.4 Medium</td><td>92.3</td><td>90.8</td><td>90.0</td><td>77.6</td><td>87.5</td><td>80.6</td><td>91.5</td><td>90.8</td><td>87.7</td></tr><tr><td>Doubao-Seed-2.0 Pro</td><td>91.7</td><td>87.8</td><td>87.9</td><td>75.1</td><td>89.4</td><td>80.1</td><td>90.2</td><td>88.9</td><td>86.4</td></tr><tr><td>Qwen3-VL-235B A22B-Thinking</td><td>87.3</td><td>86.3</td><td>82.7</td><td>69.2</td><td>82.2</td><td>74.8</td><td>87.8</td><td>82.1</td><td>81.5</td></tr><tr><td>Kimi-K2.5</td><td>86.5</td><td>77.5</td><td>81.2</td><td>70.2</td><td>82.8</td><td>71.8</td><td>84.6</td><td>82.5</td><td>79.6</td></tr><tr><td>Qwen3-VL-235B A22B-Instruct</td><td>80.7</td><td>78.8</td><td>80.7</td><td>69.5</td><td>75.5</td><td>72.5</td><td>80.9</td><td>77.7</td><td>77.0</td></tr><tr><td></td><td>D.I.</td><td>D.L.</td><td>F.D.</td><td>L.S.</td><td>T.A.</td><td>DED.</td><td>IND.</td><td>ABD.</td><td>60 MRMS.</td></tr></table>

Figure 3: MRMS heatmap for the three highest-ranked closed-source and open-source models from Table 2.

## 5 Further Discussion

Process Compliance Is High but Uneven Table 3 assesses adherence of visible solution processes to the intended OCR-grounded metareasoning direction. RPCS separates process quality from correctness and reveals uneven compliance: scores range from 72.5 to 96.2 (mean 87.5). Capability match and groundedness average 93.6 and 92.2, respectively, versus 81.6 and 82.5 for step completeness and non-hallucination. Models usually choose plausible directions and ground claims in the image; their main weaknesses are omitted intermediate evidence and unsupported rules, entities, fields, or values. Across reasoning types, RPCS is highest for deduction (89.9), followed by abduction (86.6) and induction (86.0). Thus, surface process form alone cannot explain the deductive MRMS bottleneck.

Strong Processes Do Not Guarantee Correct Answers RPCS exceeds MRMS for every model, showing that compliant reasoning traces are easier to obtain than correct answers: ∆ ranges from +4.9 for GPT-5.4-Mini-Medium to +13.5 for Qwen3- VL-8B-Instruct (mean +8.8). The strongest closedsource systems score well on both metrics: Gemini-3.1-Pro-Preview achieves RPCS 96.2 and MRMS 89.3, while GPT-5.4-Medium achieves RPCS 95.2 and MRMS 87.7. Rankings diverge: Doubao-Seed-2.0-Lite has higher average RPCS than Doubao-Seed-2.0-Pro (93.3 versus 93.0) but lower MRMS (84.2 versus 86.4). Similarly, Kimi-K2.5 has the highest open-source average RPCS (91.1), but its MRMS (79.6) remains below Qwen3-VL-235B-A22B-Thinking (81.5). Appendix H shows this pattern across deductive, inductive, and abductive samples. Accuracy still depends on precise OCR-grounded operations, not only on plausible, grounded explanations.

On the same stratified 300-sample subset, the PhD-student majority reaches 96.0 MRMS (95% CI [93.7, 98.0]), versus 90.5 [87.2, 93.6] for Gemini-3.1-Pro-Preview; the human–Gemini gap is +5.5 [1.6, 9.2]. This leaves measurable headroom above the strongest model while confirming practical solvability. The complete human baseline, ambiguity handling, and paired comparisons appear in Appendix J.4.

Process and Outcome Gaps Identify Training Targets Closed-source models exceed opensource models by 7.3 points in average RPCS, below the 9.5-point MRMS gap, and their average ∆ is also lower, at 7.8 versus 10.1. The largest criterion gap is non-hallucination (87.1 versus 76.8); the smallest is capability match (95.8 versus 90.9). RPCS–MRMS gaps are 16.4 points for deduction, 3.5 for induction, and 6.5 for abduction. The remaining challenge is precision: models must verify OCR-grounded details, cover necessary intermediate steps, and avoid unsupported decisions. These gaps justify reporting MRMS and RPCS separately.

Controls on the same 300 samples with five representative models separate perception from reasoning. Averaged over models, MRMS is 80.7 for the full image, 16.5 for question-only input, 71.5 for OCR transcript-only input, 74.4 for a layout-aware transcript, 81.3 for image plus third-party OCR, and 81.7 under an answer-format oracle. Thus question-only input is insufficient; layout-aware text recovers part of the transcript loss, and the small gains from third-party OCR or the format oracle do not remove the deduction and layout gaps. Full per-model controls and condition definitions are in Appendix J.6.

<table><tr><td rowspan="2">Model</td><td colspan="3">Process Criteria</td><td rowspan="2"></td><td colspan="3">RPCS by Reasoning Type</td><td colspan="3">Outcome Contrast</td></tr><tr><td>CM GRD</td><td></td><td>STEP NH</td><td>DED. IND. ABD. AVG.</td><td></td><td></td><td></td><td>MRMS</td><td>∆</td></tr><tr><td colspan="10">Closed-source</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>98.7</td><td>98.7</td><td>92.6</td><td>94.5</td><td>95.2</td><td>96.7</td><td>96.6</td><td>96.2</td><td>89.3</td><td>+6.9</td></tr><tr><td>GPT-5.4-Medium</td><td>98.5</td><td>97.7</td><td>91.4</td><td>93.1</td><td>95.0</td><td>95.3</td><td>95.2</td><td>95.2</td><td>87.7</td><td>+7.5</td></tr><tr><td>Doubao-Seed-2.0-Pro</td><td>96.7</td><td>96.6</td><td>87.5</td><td>91.3</td><td>93.5</td><td>92.0</td><td>93.5</td><td>93.0</td><td>86.4</td><td>+6.6</td></tr><tr><td>Claude-Sonnet-4.6</td><td>97.1</td><td>95.3</td><td>91.6</td><td>83.7</td><td>93.0</td><td>90.8</td><td>92.0</td><td>91.9</td><td>84.3</td><td>+7.6</td></tr><tr><td>Doubao-Seed-2.0-Lite</td><td>97.0</td><td>96.5</td><td>90.3</td><td>89.7</td><td>93.8</td><td>93.0</td><td>93.2</td><td>93.3</td><td>84.2</td><td>+9.1</td></tr><tr><td>GPT-5.4-Mini-Medium</td><td>94.9</td><td>94.1</td><td>75.1</td><td>91.7</td><td>90.5</td><td>86.1</td><td>90.2</td><td>88.9</td><td>84.0</td><td>+4.9</td></tr><tr><td>Gemini-3.1-Flash-Lite</td><td>93.3</td><td>95.7</td><td>82.8</td><td>85.3</td><td>92.7</td><td>86.1</td><td>89.1</td><td>89.3</td><td>83.3</td><td>+6.0</td></tr><tr><td>GPT-5.4</td><td>96.9</td><td>95.8</td><td>89.3</td><td>88.1</td><td>94.0</td><td>91.1</td><td>92.3</td><td>92.5</td><td>83.0</td><td>+9.5</td></tr><tr><td>Grok-4.20</td><td>93.7</td><td>91.7</td><td>81.5</td><td>76.0</td><td>88.3</td><td>83.5</td><td>85.4</td><td>85.7</td><td>75.2</td><td>+10.5</td></tr><tr><td>GPT-5.4-Mini</td><td>91.3</td><td>87.7</td><td>68.5</td><td>78.0</td><td>86.1</td><td>78.3</td><td>79.8</td><td>81.4</td><td>72.0</td><td>+9.4</td></tr><tr><td colspan="9">Open-source</td><td></td><td></td></tr><tr><td>Qwen3-VL-235B-A22B-Thinking</td><td>95.1</td><td>93.1</td><td>85.4</td><td>84.7</td><td>90.5</td><td>90.0</td><td>88.2</td><td>89.6</td><td>81.5</td><td>+8.1</td></tr><tr><td>Kimi-K2.5</td><td>96.7</td><td>96.0</td><td>89.7</td><td>82.0</td><td>93.0</td><td>90.5</td><td>89.8</td><td>91.1</td><td>79.6</td><td>+11.5</td></tr><tr><td>Qwen3-VL-235B-A22B-Instruct</td><td>93.5</td><td>92.4</td><td>84.2</td><td>77.9</td><td>90.5</td><td>84.6</td><td>85.9</td><td>87.0</td><td>77.0</td><td>+10.0</td></tr><tr><td>GLM-4.6V</td><td>90.0</td><td>88.1</td><td>75.4</td><td>82.9</td><td>87.1</td><td>83.0</td><td>82.3</td><td>84.1</td><td>75.9</td><td>+8.2</td></tr><tr><td>Qwen3-VL-30B-A3B-Thinking</td><td>91.5</td><td>89.8</td><td>76.7</td><td>79.5</td><td>87.2</td><td>83.9</td><td>82.2</td><td>84.4</td><td>74.7</td><td>+9.7</td></tr><tr><td>Qwen3-VL-8B-Thinking</td><td>91.1</td><td>86.9</td><td>75.1</td><td>77.5</td><td>85.4</td><td>81.7</td><td>81.0</td><td>82.7</td><td>74.5</td><td>+8.2</td></tr><tr><td>Qwen3-VL-30B-A3B-Instruct</td><td>86.1</td><td>83.6</td><td>69.3</td><td>66.0</td><td>83.1</td><td>73.0</td><td>72.7</td><td>76.3</td><td>65.0</td><td>+11.3</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>83.5</td><td>80.5</td><td>62.2</td><td>63.7</td><td>79.8</td><td>68.3</td><td>69.3</td><td>72.5</td><td>59.0</td><td>+13.5</td></tr></table>

Table 3: Comparison of RPCS and MRMS. CM, GRD, STEP, and NH denote capability match, groundedness, step completeness, and non-hallucination. DED., IND., and ABD. report RPCS in percentage points (100 times the normalized RPCS) for deductive, inductive, and abductive meta-reasoning. AVG. is mean RPCS in percentage points; MRMS is the answer-level score from Table 2; ∆ is the difference between AVG. RPCS and MRMS. All score columns are percentage points.

Difficulty stratification identifies this headroom’s sources. Among 34 very-hard items, the dominant mechanisms are cross-region layout binding (26.5%), exception/boundary/negation handling (23.5%), and implicit pattern induction (17.6%); among 44 hard-only qualitative-analysis items, exception/boundary handling (27.3%), implicit induction (22.7%), layout binding (20.5%), and hidden backward dependency (18.2%) dominate. In contrast, the 45 easy items mainly use common template calculations (77.8%) or bounded candidate sets (22.2%). Process scores match: step completeness/non-hallucination are 92.2/95.9 on easy items versus 57.2/53.6 on very-hard items. Appendix K reports automatic buckets, coding standard, and representative cases.

## 6 Conclusion

This work addresses meta-reasoning evaluation in text-rich image understanding, where models must go beyond OCR extraction to organize evidence by the required reasoning direction. We introduce OCR-MetaReasoning, adapting deduction, induction, and abduction to OCR-centered scenarios and evaluating models on representative document, chart, transaction, form, and layout reasoning tasks. The benchmark provides a unified protocol measuring final-answer correctness and reasoningprocess compliance. Experiments with representative MLLMs show differences across reasoning types and OCR-object categories, and current models still struggle with grounded rule verification and layout-sensitive inference. Although OCR-MetaReasoning focuses on single-image settings and uses judge-assisted process evaluation, future work could extend it to multi-page evidence chains and training methods for more faithful, directionaware OCR reasoning.

## Limitations

OCR-MetaReasoning is a controlled diagnostic benchmark for single-image OCR-grounded metareasoning. This scope isolates how models bind visible textual evidence to deductive, inductive, and abductive reasoning directions, but it excludes multi-page documents, cross-image evidence aggregation, retrieval-augmented document reasoning, and interactive clarification. The reported results should therefore be interpreted within singleimage, text-rich reasoning settings.

The dataset is balanced by design rather than sampled from a natural task distribution. Equal coverage across meta-reasoning types and OCR-object categories supports controlled comparison, but it does not reflect real-world category frequencies. Although the samples undergo human verification and satisfy the stated validity criteria, controlled edited/reconstructed images and MLLM-assisted synthesis may still introduce distributional differences relative to fully natural documents. The reasoning labels should be interpreted as dominantbottleneck annotations, because individual examples may also contain local arithmetic, lookup, comparison, or extraction steps.

RPCS is a process diagnostic, not direct evidence of internal model reasoning. Although the judge is outcome-isolated and structurally validated, it evaluates visible rationales and cannot fully rule out plausible post-hoc explanations. Future work could extend OCR-MetaReasoning to multi-page evidence chains, natural-distribution test sets, compositional reasoning annotations, and stronger verification methods that link intermediate steps to OCR evidence.

## Acknowledgments

This work is supported by the National Key Research and Development Program of China (No.2023YFF0905400), the National Natural Science Foundation of China (No.U2341229) and the Reform Commission Foundation of Jilin Province (No.2024C003).

## References

Anthropic. 2026. Introducing Claude Sonnet 4.6. https://www.anthropic.com/news/ claude-sonnet-4-6. Accessed: May 2026.

Ali Furkan Biten, Rubèn Tito, Andrés Mafla, Lluís Gómez i Bigorda, Marçal Rusiñol, C. V. Jawa-

har, Ernest Valveny, and Dimosthenis Karatzas. 2019. Scene text visual question answering. In 2019 IEEE/CVF International Conference on Computer Vision, ICCV 2019, Seoul, Korea (South), October 27 - November 2, 2019, pages 4290–4300. IEEE.

ByteDance. 2026. Seed-2. https://seed.bytedance. com/zh/seed2. Accessed: May 2026.

Mu Cai, Haotian Liu, Siva Karthik Mustikovela, Gregory P. Meyer, Yuning Chai, Dennis Park, and Yong Jae Lee. 2024. Vip-llava: Making large multimodal models understand arbitrary visual prompts. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 12914–12923. IEEE.

Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang, Yi Chang, Philip S. Yu, Qiang Yang, and Xing Xie. 2024. A survey on evaluation of large language models. ACM Trans. Intell. Syst. Technol., 15(3):39:1– 39:45.

Yangyi Chen, Karan Sikka, Michael Cogswell, Heng Ji, and Ajay Divakaran. 2024. Measuring and improving chain-of-thought reasoning in vision-language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), NAACL 2024, Mexico City, Mexico, June 16-21, 2024, pages 192– 210. Association for Computational Linguistics.

Ziming Cheng, Binrui Xu, Lisheng Gong, Zuhe Song, Tianshuo Zhou, Shiqi Zhong, Siyu Ren, Mingxiang Chen, Xiangchao Meng, Yuxin Zhang, Yanlin Li, Lei Ren, Wei Chen, Zhiyuan Huang, Mingjie Zhan, Xiaojie Wang, and Fangxiang Feng. 2025. Evaluating mllms with multimodal multi-image reasoning benchmark. CoRR, abs/2506.04280.

Yuhao Dong, Zuyan Liu, Hai-Long Sun, Jingkang Yang, Winston Hu, Yongming Rao, and Ziwei Liu. 2025. Insight-v: Exploring long-chain visual reasoning with multimodal large language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2025, Nashville, TN, USA, June 11-15, 2025, pages 9062–9072. Computer Vision Foundation / IEEE.

Ling Fu, Zhebin Kuang, Jiajun Song, Mingxin Huang, Biao Yang, Yuzhe Li, Linghao Zhu, Qidi Luo, Xinyu Wang, Hao Lu, Zhang Li, Guozhi Tang, Bin Shan, Chunhui Lin, Qi Liu, Binghong Wu, Hao Feng, Hao Liu, Can Huang, and 5 others. 2025. Ocrbench v2: An improved benchmark for evaluating large multimodal models on visual text localization and reasoning. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025.

Google. 2026a. Gemini 3.1 flash-lite preview. https://ai.google.dev/gemini-api/docs/ models/gemini-3.1-flash-lite-preview. Accessed: May 2026.

Google. 2026b. Gemini 3.1 pro preview. https://ai.google.dev/gemini-api/docs/ models/gemini-3.1-pro-preview. Accessed: May 2026.

Haibin He, Maoyuan Ye, Jing Zhang, Xiantao Cai, Juhua Liu, Bo Du, and Dacheng Tao. 2025. Reasoning-ocr: Can large multimodal models solve complex logical reasoning problems from OCR cues? CoRR, abs/2505.12766.

Anwen Hu, Haiyang Xu, Liang Zhang, Jiabo Ye, Ming Yan, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. 2025a. mplug-docowl2: High-resolution compressing for ocr-free multi-page document understanding. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 5817–5834. Association for Computational Linguistics.

Zhiyuan Hu, Yibo Wang, Hanze Dong, Yuhui Xu, Amrita Saha, Caiming Xiong, Bryan Hooi, and Junnan Li. 2025b. Beyond ’aha!’: Toward systematic metaabilities alignment in large reasoning models. CoRR, abs/2505.10554.

Mingxin Huang, Yongxin Shi, Dezhi Peng, Songxuan Lai, Zecheng Xie, and Lianwen Jin. 2025. Ocrreasoning benchmark: Unveiling the true capabilities of mllms in complex text-rich image reasoning. CoRR, abs/2505.17163.

Guillaume Jaume, Hazim Kemal Ekenel, and Jean-Philippe Thiran. 2019. FUNSD: A dataset for form understanding in noisy scanned documents. CoRR, abs/1905.13538.

Ming Li, Ruiyi Zhang, Jian Chen, Jiuxiang Gu, Yufan Zhou, Franck Dernoncourt, Wanrong Zhu, Tianyi Zhou, and Tong Sun. 2025a. Towards visual text grounding of multimodal large language model. CoRR, abs/2504.04974.

Siqi Li, Yufan Shen, Xiangnan Chen, Jiayi Chen, Hengwei Ju, Haodong Duan, Song Mao, Hongbin Zhou, Bo Zhang, Bin Fu, Pinlong Cai, Licheng Wen, Botian Shi, Yong Liu, Xinyu Cai, and Yu Qiao. 2025b. Gdi-bench: A benchmark for general document intelligence with vision and reasoning decoupling. CoRR, abs/2505.00063.

Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. 2024. Ocrbench: on the hidden mystery of OCR in large multimodal models. Sci. China Inf. Sci., 67(12).

Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq R. Joty, and Enamul Hoque. 2022. Chartqa: A benchmark for question answering about charts with visual

and logical reasoning. In Findings ofthe Association for Computational Linguistics: ACL 2022, Dublin, Ireland, May 22-27, 2022, volume ACL 2022 of Findings of ACL, pages 2263–2279. Association for Computational Linguistics.

Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and C. V. Jawahar. 2022. Infographicvqa. In IEEE/CVF Winter Conference on Applications of Computer Vision, WACV 2022, Waikoloa, HI, USA, January 3-8, 2022, pages 2582– 2591. IEEE.

Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. 2021. Docvqa: A dataset for VQA on document images. In IEEE Winter Conference on Applications ofComputer Vision, WACV 2021, Waikoloa, HI, USA, January 3-8, 2021, pages 2199–2208. IEEE.

MoonshotAI. 2025. Kimi-k2.5. https:// huggingface.co/moonshotai/Kimi-K2.5. Accessed: May 2026.

OpenAI. 2026a. Introducing GPT-5.4. https:// openai.com/index/introducing-gpt-5-4/. Accessed: May 2026.

OpenAI. 2026b. Introducing GPT-5.4 mini and nano. https://openai.com/index/ introducing-gpt-5-4-mini-and-nano/. Accessed: May 2026.

Seunghyun Park, Seung Shin, Bado Lee, Junyeop Lee, Jaeheung Surh, Minjoon Seo, and Hwalsuk Lee. 2019. {CORD}: A consolidated receipt dataset for post-{ocr} parsing. In Workshop on Document Intelligence at NeurIPS 2019.

Qwen. 2025a. Qwen3-vl-235b-a22binstruct. https://huggingface.co/Qwen/ Qwen3-VL-235B-A22B-Instruct. Accessed: January 2026.

Qwen. 2025b. Qwen3-vl-235b-a22bthinking. https://huggingface.co/Qwen/ Qwen3-VL-235B-A22B-Thinking. Accessed: May 2026.

Qwen. 2025c. Qwen3-vl-30b-a3binstruct. https://huggingface.co/Qwen/ Qwen3-VL-30B-A3B-Instruct. Accessed: January 2026.

Qwen. 2025d. Qwen3-vl-30b-a3bthinking. https://huggingface.co/Qwen/ Qwen3-VL-30B-A3B-Thinking. Accessed: May 2026.

Qwen. 2025e. Qwen3-vl-8b-instruct. https:// huggingface.co/Qwen/Qwen3-VL-8B-Instruct. Accessed: January 2026.

Qwen. 2025f. Qwen3-vl-8b-thinking. https:// huggingface.co/Qwen/Qwen3-VL-8B-Thinking. Accessed: May 2026.

Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. 2024. Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. In Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards VQA models that can read. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, pages 8317–8326. Computer Vision Foundation / IEEE.

Yueqi Song, Tianyue Ou, Yibo Kong, Zecheng Li, Graham Neubig, and Xiang Yue. 2025. Visualpuzzles: Decoupling multimodal reasoning evaluation from domain knowledge. CoRR, abs/2504.10342.

Hongbin Sun, Zhanghui Kuang, Xiaoyu Yue, Chenhao Lin, and Wayne Zhang. 2021. Spatial dualmodality graph reasoning for key information extraction. CoRR, abs/2103.14470.

Qwen Team. 2025. Qwen3-vl technical report. CoRR, abs/2511.21631.

Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, Alexis Chevalier, Sanjeev Arora, and Danqi Chen. 2024. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. In Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.

xAI. 2026. Grok 4.20 non-reasoning. https: //docs.x.ai/developers/models/grok-4. 20-non-reasoning. Accessed: May 2026.

Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. 2024. Logicvista: Multimodal LLM logical reasoning benchmark in visual contexts. CoRR, abs/2407.04973.

Maoyuan Ye, Jing Zhang, Juhua Liu, Bo Du, and Dacheng Tao. 2025. Logicocr: Do your large multimodal models excel at logical reasoning on text-rich images? CoRR, abs/2505.12307.

Jiakang Yuan, Tianshuo Peng, Yilei Jiang, Yiting Lu, Renrui Zhang, Kaituo Feng, Chaoyou Fu, Tao Chen, Lei Bai, Bo Zhang, and Xiangyu Yue. 2025. Mmereasoning: A comprehensive benchmark for logical reasoning in mllms. CoRR, abs/2505.21327.

ZaiOrg. 2025. Glm-4.6v. https://huggingface.co/ zai-org/GLM-4.6V. Accessed: May 2026.

Jusheng Zhang, Kaitong Cai, Xiaoyang Guo, Sidi Liu, Qinhan Lv, Ruiqi Chen, Jing Yang, Yijia Fan, Xiaofei Sun, Jian Wang, Ziliang Chen, Liang Lin, and Keze Wang. 2025. Mm-cot:a benchmark for probing visual chain-of-thought reasoning in multimodal models. CoRR, abs/2512.08228.

Yiyang Zhou, Haoqin Tu, Zijun Wang, Zeyu Wang, Niklas Muennighoff, Fan Nie, Yejin Choi, James Zou, Chaorui Deng, Shen Yan, Haoqi Fan, Cihang Xie, Huaxiu Yao, and Qinghao Ye. 2025. When visualizing is the first step to reasoning: Mira, a benchmark for visual chain-of-thought. CoRR, abs/2511.02779.

![](images/5840835d2fed190b554ec886b6d50aff4bd9853c3ff99189fcde49620755a935.jpg)  
Figure 4: Abstract illustration of OCR-grounded metareasoning.

## A Analysis of Meta-Reasoning Types

Figure 4 illustrates the motivation for OCRgrounded meta-reasoning. The upper panel presents OCR evidence as structured visual information rather than as a plain text transcript. Words, layout, fields, and charts each provide distinct evidence for reasoning. The three lower panels distinguish questions according to the required reasoning direction. Deduction applies visible rules to image-grounded cases, so errors often arise when a model copies a salient text span without verifying all relevant conditions. Induction infers hidden regularities from aligned observations, making shallow field matching a common source of mistakes. Abduction reasons backward from an observed result to recover a missing premise; the main risk is a plausible explanation that is insufficiently grounded in the image. This distinction motivates the evaluation design of OCR-MetaReasoning. The benchmark evaluates both final-answer correctness and reasoning-process compliance because the same answer can result from different ways of organizing visual evidence.

<table><tr><td>Setting / Split</td><td>No Hint</td><td>Task-specific Hint</td><td>∆</td></tr><tr><td>Overall answer score</td><td>70.08</td><td>71.09</td><td>+1.01</td></tr><tr><td>Full-score rate</td><td>68.18</td><td>69.10</td><td>+0.92</td></tr><tr><td>Meta-abductive</td><td>69.28</td><td>71.46</td><td>+2.18</td></tr><tr><td>Meta-deductive</td><td>67.25</td><td>68.16</td><td>+0.91</td></tr><tr><td>Meta-inductive</td><td>73.72</td><td>73.66</td><td>-0.06</td></tr></table>

Table 4: Paired comparison of the no-hint and taskspecific-hint settings across the six shared models. Scores are percentages.

## B Effect of Reasoning Hints

We conduct an ablation study to examine whether explicit reasoning hints improve model performance. The comparison uses two prompting strategies. The generic reasoning hint adds the same instruction to every question: “Before answering, think step by step and carefully verify the evidence in the image.” The task-specific hint instead provides an instruction specific to the target metareasoning type. For abductive samples, it asks the model to reason backward from the stated outcome, identify the hidden constraint, compare plausible candidates, and eliminate alternatives. For deductive samples, it asks the model to extract the rule or threshold, list candidate cases, and verify every condition, including boundaries, units, legends, and exceptions. For inductive samples, it asks the model to infer the underlying pattern from multiple examples, test a contrasting case, and apply the pattern to the target.

For a fair paired comparison, we restrict the analysis to the six models with results under both the no-hint and task-specific-hint settings: Qwen3-VL-8B-Instruct, Qwen3-VL-8B-Thinking, Qwen3-VL-30B-A3B-Instruct, Qwen3-VL-30B-A3B-Thinking, GPT-5.4-Mini, and Grok-4.20. These models yield the same 9,000 paired modelsample evaluations. Task-specific hints increase the average final-answer score from 70.08 to 71.09, corresponding to a gain of +1.01 percentage points with a paired bootstrap 95% confidence interval of [+0.22, +1.78]. The full-score rate also increases from 68.18 to 69.10 (+0.92 points). The gain is largest for abductive reasoning (+2.18 points), smaller for deductive reasoning (+0.91 points), and nearly absent for inductive reasoning (-0.06 points). Table 4 reports the paired comparison.

We also conduct a smaller generic-reasoning control on Qwen3-VL-8B-Instruct. The generic control raises the score from 59.03 to 62.95 (+3.91 points, 95% CI [+1.96, +5.91]), whereas the taskspecific hint reaches 63.35 (+4.31 points relative to the no-hint setting). The task-specific hint achieves the highest score, but its margin over the generic control is small (+0.40 points, 95% CI [-1.57, +2.38]). Table 5 reports this control comparison.

<table><tr><td>Setting</td><td>Overall</td><td>Abductive</td><td>Deductive</td><td>Inductive</td></tr><tr><td>No Hint</td><td>59.03</td><td>57.60</td><td>59.06</td><td>60.45</td></tr><tr><td>Generic Reasoning</td><td>62.95</td><td>62.46</td><td>61.43</td><td>64.95</td></tr><tr><td>Task-specific Hint</td><td>63.35</td><td>63.34</td><td>63.39</td><td>63.32</td></tr></table>

Table 5: Generic-reasoning control for Qwen3-VL-8B-Instruct. Scores are percentages.

The results suggest that explicit reasoning guidance improves final-answer performance, with the clearest gains for abductive and deductive reasoning. The control experiment suggests that general evidence-checking guidance accounts for much of the improvement, rather than task-specific wording alone. We interpret the hint effect as an answerlevel improvement, not as direct evidence of higher reasoning-process quality.

The effect is also model-dependent. The score for Grok-4.20 decreases by 2.19 points under taskspecific hints, mainly on data-interpretation and integer-answer cases. The score for Qwen3-VL-8B-Thinking decreases by 0.56 points, mainly on exact-match string answers. In the inspected error cases, hints sometimes lead to excessive deliberation, incorrect candidate filtering, misreading of OCR details, or drift in answer granularity. RPCS also does not improve under hints, indicating that the primary benefit is higher final-answer accuracy rather than consistently improved judged reasoning processes.

## C Experimental Setup

We evaluate both proprietary and open-source models using decoding settings based on their published configurations. The Qwen3-VL-Instruct series uses $T ~ = ~ 0 . 7$ $t o p _ { - } p = 0 . 8 .$ , top\_k = 20, a repetition penalty of 1.0, and a presence penalty of 1.5. The Qwen3-VL-Thinking series uses $T \ = \ 1 . 0 .$ top $p = 0 . 0 5$ , top\_ $k = 2 0$ , a repetition penalty of 1.0, and a presence penalty of 1.5. Kimi-K2.5-Instant uses $T ~ = ~ 1 . 0$ and top $ { p } = \ 0 . 9 5$ whereas GLM-4.6V uses $T = 0 . 8 ,$ top $\begin{array} { r } { p = 0 . 6 } \end{array}$ $t o p \_ k = 2$ , and a repetition penalty of 1.1. Table 24 summarizes the evaluated models and their reasoning modes.

<table><tr><td>Category</td><td># Instances</td></tr><tr><td>Meta-deductive Reasoning Meta-inductive Reasoning Meta-abductive Reasoning</td><td>500 500 500</td></tr><tr><td>Transaction Analysis Data Interpretation Field Dependency</td><td>300 300 300</td></tr><tr><td>Document Logic Layout Semantics</td><td>300 300</td></tr><tr><td>String Answers Integer Answers</td><td>513</td></tr><tr><td></td><td>561</td></tr><tr><td></td><td></td></tr><tr><td>Floating-point Answers</td><td>146</td></tr><tr><td>JSON Ánswers</td><td>280</td></tr><tr><td></td><td></td></tr><tr><td>Exact-Match Scoring</td><td>513</td></tr><tr><td>Numeric Scoring</td><td></td></tr><tr><td></td><td>707</td></tr><tr><td>JSON Micro-F1 Scoring</td><td>280</td></tr><tr><td></td><td></td></tr><tr><td>Total Instances</td><td>1,500</td></tr><tr><td>Average # Tokens per Question</td><td>75.77</td></tr><tr><td>Maximum # Tokens per Question</td><td>201</td></tr><tr><td></td><td></td></tr><tr><td>Minimum # Tokens per Question</td><td>21</td></tr><tr><td>Average # Reasoning Steps</td><td>4.22</td></tr><tr><td>Maximum # Reasoning Steps</td><td>8</td></tr><tr><td>Minimum # Reasoning Steps</td><td>2</td></tr><tr><td>Average # Images per Sample</td><td>1.00</td></tr></table>

Table 6: Statistics for the OCR-MetaReasoning benchmark.

## D Dataset Statistics

We summarize the composition of OCR-MetaReasoning by meta-reasoning type, OCRobject category, and sample complexity, following the two-axis design in Figure 2. Table 6 reports these statistics.

## D.1 Meta-Reasoning Taxonomy and OCR-Object Distribution

OCR-MetaReasoning contains 1,500 instances, evenly divided across the three meta-reasoning types. Meta-deductive, meta-inductive, and metaabductive reasoning each contain 500 samples. Equal sizes keep the overall metric from being dominated by any single reasoning direction and support direct comparison among forward rule application, pattern abstraction, and backward explanation.

The dataset is also balanced across five OCRobject categories: transaction analysis, data interpretation, field dependency, document logic, and layout semantics. Each category contains 300 instances, and every cell in the $3 \times 5$ taxonomy contains 100 samples. This balance reduces confounding between reasoning type and visual document genre. For example, deductive and abductive reasoning are both evaluated on receipts, tables, forms, long document snippets, and layout-centric images rather than being tied to a single source format.

## D.2 Answer Formats and Scoring Diversity

The benchmark uses multiple answer formats, so performance cannot be reduced to a single response style. Integer answers are the most common format, with 561 instances, followed by 513 string answers, 280 JSON answers, and 146 floating-point answers. Scoring functions are assigned at the sample level: 707 instances use numeric matching, 513 use normalized exact match, and 280 use JSON micro-F1. This mixture reflects the range of OCR-grounded reasoning tasks. Some samples require a single year, amount, count, or percentage, whereas others require an entity name, a formatted text span, or a structured set of field values.

## D.3 Sample Complexity

Each instance uses a single text-rich image, separating image-grounded reasoning from multi-image retrieval or document aggregation. The controlled setting still requires multi-step reasoning. The average question length is 75.77 tokens, with a range from 21 to 201 tokens. Reference solutions contain 4.22 reasoning steps on average, with a minimum of 2 and a maximum of 8. Most samples therefore require several intermediate operations, such as locating separated evidence, binding a rule to a visible field, abstracting a repeated pattern, or tracing a result back to a hidden premise. The benchmark focuses on structured reasoning rather than direct OCR span extraction.

## E Benchmark Construction

The construction process described in Section 3.2 and Figure 2 is assessed at two levels: structural validity and human verification. Each benchmark instance is verified as a complete annotated item containing a text-rich image, a question, a canonical answer, an answer format, a scoring rule, a metareasoning label, an OCR-object label, and a reference reasoning path. Structural checks cover field completeness and scoring compatibility, whereas human verification examines semantic validity, answer uniqueness, and taxonomy fit.

## E.1 Construction Pipeline

Seed selection We collect text-rich visual seeds from public OCR-oriented resources, together with controlled edited/reconstructed variants of those seeds, including receipts, charts, forms, policy-like documents, posters, webpages, and infographics. Candidate seeds are retained only when the visible content requires reasoning over OCR evidence rather than direct copying. We reject seeds when the answer can be copied from a single visible text span, when relevant text is illegible, or when the solution depends primarily on external knowledge rather than the image.

Taxonomy-controlled synthesis For each retained seed, the generator receives a target metareasoning type and OCR-object category. Deductive samples require a visible rule, threshold, clause, legend, formula, or field dependency to be applied to image evidence. Inductive samples require multiple visible support cases from which a pattern can be inferred and then applied to a target. Abductive samples require an observed result, anomaly, or downstream state from which a hidden premise must be recovered through backward reasoning. This procedure keeps the benchmark from collapsing into generic OCR question answering.

Human verification and candidate filtering During construction, three PhD students with research experience in multimodal large language models assessed each candidate using a checklist aligned with the benchmark definition. The annotators saw the proposed taxonomy labels because this stage involved taxonomy-aligned candidate assessment. A sample was retained only when its primary evidence was grounded in the image and at least two evidence points were required. It also had to have a unique or uniquely minimal answer, a meta-reasoning label matching the primary reasoning bottleneck, an OCR-object label matching the visual scenario, and an answer that could be scored automatically after normalization. Candidates that failed any requirement were rejected or filtered. When a candidate was identified as similar to an existing task, we generated a new candidate through separate prompt or image regeneration and counted the new candidate separately. Common rejection cases included answer leakage from a single OCR span, unsupported rules, ambiguous wording, non-unique abductive explanations, weak inductive evidence, and mismatched answer formats.

## E.2 Benchmark Validity and Scoring Consistency

After human verification, we evaluated the released benchmark for annotation completeness, image– question correspondence, uniqueness of questions and question–answer pairs, answer-format validity, scoring compatibility, and the presence of multistep reference reasoning. Table 7 summarizes these validity and scoring properties.

<table><tr><td>Benchmark property</td><td>Observed / Total</td><td>Rate</td></tr><tr><td>Complete annotations</td><td>1,500 / 1,500</td><td>100.0%</td></tr><tr><td>Image-question correspondence</td><td>1,500 / 1,500</td><td>100.0%</td></tr><tr><td>Unique normalized questions</td><td>1,500 / 1,500</td><td>100.0%</td></tr><tr><td>Unique normalized question-answer pairs</td><td>1,500 / 1,500</td><td>100.0%</td></tr><tr><td>Compatible answer format and scoring rule</td><td>1,500 /1,500</td><td>100.0%</td></tr><tr><td>Valid structured answers</td><td>280 / 280</td><td>100.0%</td></tr><tr><td>Valid numeric answers</td><td>707 / 707</td><td>100.0%</td></tr><tr><td>Multi-step reference reasoning</td><td>1,500 / 1,500</td><td>100.0%</td></tr></table>

Table 7: Validity and scoring properties for the 1,500 benchmark instances.

All 1,500 instances contain complete annotations and coherent image–question associations. All normalized questions and question–answer pairs are unique, reducing the risk that benchmark performance is inflated by repeated prompts. The scoring specifications are internally consistent: 561 integer and 146 floating-point answers use numeric scoring, 513 string answers use normalized exact match, and all 280 structured answers conform to the declared format and use JSON micro-F1. Every sample has at least two reference reasoning steps, with an average of 4.22 steps, consistent with the goal of evaluating reasoning paths rather than single-span extraction.

The split is exactly balanced along both benchmark axes: 500 samples for each meta-reasoning type, 300 samples for each OCR-object category, and 100 samples in every 3 × 5 taxonomy cell. This balance keeps aggregate scores from being dominated by one reasoning direction or one document genre. Repeated source images are treated as source reuse rather than duplicate examples when the corresponding instances have different questions or reasoning targets. No exact duplicate normalized questions or question–answer pairs are present.

## E.3 Why These Properties Support Benchmark Reliability

These validity criteria address three failure modes that are especially harmful for OCR-grounded meta-reasoning benchmarks. Human review detects semantic failures that structural criteria cannot capture, such as non-unique abductive explanations or inductive rules supported by too few examples. The complementary structural criteria detect incomplete items, invalid structured answers, incompatible scoring specifications, and repeated prompts. The balanced taxonomy helps ensure that the reported MRMS and category analyses reflect model behavior rather than uneven sampling. Together, human verification and these structural properties provide evidence that the benchmark is well-formed, deterministically scorable under the stated metrics, and aligned with the intended metareasoning taxonomy.

## F Evaluation Process

The evaluation protocol follows Section 3.3 and the three-stage structure used by OCR-Reasoning (Huang et al., 2025): response generation, answer extraction, and score computation. We use a separate MLLM-as-judge stage for reasoningprocess compliance. Answer scoring and process judging remain separate: deterministic answer scoring determines MRMS, whereas the MLLM judge evaluates only the visible reasoning process for RPCS.

## F.1 Answer-Level Score Computation

Response generation For each sample i, the evaluated model receives the image $I _ { i } ,$ the question $q _ { i } .$ and an instruction specifying the required answer format. The model returns numbered reasoning steps followed by a separate final-answer line. Numeric, string, and structured responses must conform to their declared answer formats.

Answer extraction The complete model response is retained for process evaluation, whereas answer-level scoring uses only the extracted final answer $\hat { y } _ { i }$ . We identify the final-answer segment and apply the appropriate format-specific normalization. The extraction procedure does not use the gold answer $y _ { i } ,$ , the reference reasoning steps, or the model score. If a valid final answer cannot be identified, the prediction is treated as empty for answer scoring.

Per-sample scoring Each sample specifies a scoring metric $m _ { i } ,$ and the per-sample score is

$$
s _ { i } = \mathrm { S c o r e } ( \hat { y } _ { i } , y _ { i } , m _ { i } ) , \quad s _ { i } \in [ 0 , 1 ] .
$$

For exact-match responses, both strings are normalized by lowercasing, removing redundant whitespace and lightweight formatting, and standardizing common date, currency, and percentage variants. The score is 1 only when the normalized strings match exactly. For numeric responses, formatting such as commas, currency symbols, and trailing percent markers is removed before exact equality is tested. We do not use a numeric tolerance because benchmark answers are constructed to be uniquely recoverable. For structured responses, both answers are represented as normalized key–value sets and scored by micro-F1:

$$
\begin{array} { l } { { P _ { i } = \displaystyle \frac { \lvert K _ { \hat { y } _ { i } } \cap K _ { y _ { i } } \rvert } { \lvert K _ { \hat { y } _ { i } } \rvert } , } } \\ { { R _ { i } = \displaystyle \frac { \lvert K _ { \hat { y } _ { i } } \cap K _ { y _ { i } } \rvert } { \lvert K _ { y _ { i } } \rvert } , } } \\ { { s _ { i } = \displaystyle \frac { 2 P _ { i } R _ { i } } { P _ { i } + R _ { i } } . } } \end{array}
$$

If either structured response is invalid, the score is zero.

Aggregate answer metrics The primary score is the Meta-Reasoning Macro Score:

$$
M R M S = \frac { A c c _ { \mathrm { D e d } } + A c c _ { \mathrm { I n d } } + A c c _ { \mathrm { A b d } } } { 3 } .
$$

$$
A c c _ { t } = \frac { 1 } { | D _ { t } | } \sum _ { i \in D _ { t } } s _ { i } .
$$

Here, $D _ { t }$ is the subset for reasoning type t. We also report the overall micro score,

$$
M i c r o = \frac { 1 } { | D | } \sum _ { i \in D } s _ { i } ,
$$

as well as answer-type and OCR-object scores computed by averaging $s _ { i }$ within each answer format or OCR-object category. All item-level answer scores and formula-defined aggregate metrics are on the normalized [0, 1] scale. Unless a table states otherwise, aggregate result tables display these values as percentage points; the difficulty table below retains the [0, 1] scale.

## F.2 Reasoning-Process Score Computation

Judge input RPCS evaluates the visible process $\pi _ { i } ,$ not the final answer. Before scoring, standalone final-answer lines are removed from the model output. The MLLM judge receives only the image, question, target reasoning type, OCR-object category, answer format, scoring rule, reference reasoning steps, and the remaining visible reasoning process. It does not receive the gold answer, predicted answer, or answer-level score.

Four binary criteria The judge assigns four binary scores:

$$
c _ { i } ^ { k } \in \{ 0 , 1 \} , \quad k \in \{ { \mathrm { m a t c h } } , { \mathrm { g r o u n d } } , { \mathrm { s t e p } } , { \mathrm { n o n h a l l } } \} .
$$

Capability match assesses whether the main process follows the target reasoning direction represented by $H , R , O ;$ rule-bound candidate verification for deduction, multi-instance pattern abstraction for induction, and backward recovery of a hidden premise for abduction. Groundedness assesses whether key claims are traceable to the image or to question constraints. Step completeness assesses whether the necessary intermediate nodes are present, while allowing the steps to be compressed or reordered. Non-hallucination assesses whether the process avoids invented rules, entities, fields, values, or explanations.

The raw and normalized process scores are:

$$
\begin{array} { c } { { R P C S _ { i } ^ { r a w } = c _ { i } ^ { m a t c h } + c _ { i } ^ { g r o u n d } } } \\ { { + c _ { i } ^ { s t e p } + c _ { i } ^ { n o n h a l l } , } } \\ { { R P C S _ { i } = \frac { R P C S _ { i } ^ { r a w } } { 4 } . } } \end{array}
$$

Model-level RPCS is the mean of $R P C S _ { i }$ over scorable samples. Inference failures, empty outputs, and outputs containing only a final answer receive zero process credit rather than being silently dropped.

## F.3 Reliability Controls for MLLM-as-Judge

We use an MLLM-as-judge because process evaluation requires comparing free-form reasoning traces with image-grounded reference paths, a task that is costly and difficult to standardize through purely manual scoring. The judge protocol incorporates four safeguards.

Outcome isolation The judge is explicitly instructed not to evaluate final-answer correctness. The final-answer line is removed before judgment, and answer fields are withheld. Removing these fields prevents the judge from assigning high process credit merely because the answer is correct or low process credit merely because the final answer is wrong.

Reference-guided but wording-invariant judging The judge receives reference reasoning steps as process anchors rather than as a required script. It is instructed to credit compressed, merged, reordered, or paraphrased reasoning when the required evidence bindings and reasoning direction are preserved. This protocol reduces sensitivity to surface style and focuses the score on OCRgrounded reasoning behavior.

Criterion consistency The judge reports four binary criterion decisions, an aggregate process score, and concise rationales. The aggregate process score is defined directly by the four criterion decisions.

Separate reporting MRMS and RPCS are reported separately. MRMS is determined only by automatic answer scoring, whereas RPCS is used for process diagnostics and mismatch analysis. This separation preserves the process-aware motivation of OCR-Reasoning while avoiding dependence on an MLLM judge for the primary answercorrectness leaderboard.

## G More Results

The main results summarize the model ranking and the primary differences across reasoning types and OCR-object categories. We report additional breakdowns by answer format, OCR-object category, process-compliance criteria, and the crossed 3 × 5 taxonomy.

Answer Format Remains a Structured-Output Bottleneck Table 19 complements the main leaderboard in Table 2. Answer format changes the difficulty profile even within the same benchmark. The leading model, Gemini-3.1-Pro-Preview, reaches 95.2% on float answers and 93.6% on string answers, but drops to 77.0% on JSON answers. GPT-5.4-Medium follows the same pattern, with 92.5% on float answers and 78.5% on JSON. This gap shows that the strongest models are limited not only by OCR-grounded reasoning but also by the need to produce structurally exact outputs under automatic scoring. Open-source models show a narrower performance range but a lower ceiling. Kimi-K2.5 obtains 78.9% on string, 81.3% on integer, 80.1% on float, and 77.3% on JSON answers, suggesting more uniform behavior than the leading closed-source systems.

Deduction Drives the Main Accuracy Gap Table 20 reports the full meta-reasoning breakdown underlying Table 2. Among the top closed-source models, deductive scores remain substantially lower than inductive and abductive scores. Gemini-3.1-Pro-Preview obtains 80.7% on deduction but 93.1% on induction and 94.1% on abduction. GPT-5.4-Medium shows a similar gap, with 80.6% on deduction compared with 91.5% on induction and 90.8% on abduction. The same bottleneck appears among open-source models. Qwen3-VL-235B-A22B-Thinking reaches 87.8% on induction and 82.1% on abduction, but only 74.8% on deduction. This pattern supports the conclusion in the main text that applying visible rules and constraints is harder than abstracting repeated patterns or recovering plausible hidden premises.

Layout Semantics Is Consistently Hardest Table 21 expands the OCR-object analysis in Table 2. The strongest models score above 87% on most non-layout categories, but their performance on layout semantics remains substantially lower. Gemini-3.1-Pro-Preview scores 76.2% on layout semantics, GPT-5.4-Medium scores 77.6%, and Doubao-Seed-2.0-Pro scores 75.1%. This weakness is more pronounced among open-source models. Qwen3- VL-235B-A22B-Thinking reaches 87.3% on data interpretation and 86.3% on document logic, but only 69.2% on layout semantics. This pattern distinguishes text recognition and field-level reasoning from visually grounded grouping, alignment, and cross-region layout inference.

Process Scores Are High but Incomplete Table 22 breaks down the RPCS comparison in Table 3 by criterion. Most models achieve high rates for capability match and groundedness, indicating that they often choose the intended reasoning direction and connect their claims to image-grounded evidence. Step completeness and non-hallucination are weaker. For example, GPT-5.4-Mini-Medium has 94.9% capability match and 94.1% groundedness, but only 75.1% step completeness. Qwen3- VL-8B-Instruct drops further to 62.2% step completeness and 63.7% non-hallucination. This difference helps explain why RPCS is generally higher than MRMS in Table 3: models can follow the broad process form while still omitting necessary intermediate evidence or introducing unsupported details.

The Crossed Taxonomy Localizes Failure Modes Table 23 uses the balanced 3 × 5 design described in Table 6. The all-model average is highest for inductive reasoning, at 82.5%, followed by abduction at 80.1% and deduction at 73.6%. The hardest cell is abductive layout semantics, at 63.2%, closely followed by deductive layout semantics at 65.4%. In contrast, inductive document logic and transaction analysis reach 87.3% and 86.8%, respectively. The closed-source and open-source splits preserve the same structure: closed-source models average 69.6% on abductive layout semantics, whereas open-source models average 55.1%. The interaction table shows that benchmark difficulty depends on the combination of reasoning type and OCR-object category, not on either axis alone.

## H Qualitative Analysis of RPCS and MRMS Mismatches

Figures 14, 15, and 16 show representative cases where the visible reasoning process receives full RPCS credit but the final answer receives zero MRMS credit. This contrast reflects the distinct roles of the two metrics. RPCS evaluates whether the model follows the intended meta-reasoning direction, grounds its claims in the image, includes the necessary intermediate steps, and avoids unsupported additions. MRMS instead compares the extracted final answer with the canonical answer. A model can satisfy the process rubric and still make an OCR-grounded error that changes the final value.

In the deductive example in Figure 14, the question asks for the earliest date on which all visible constraints are satisfied. The model identifies the required constraint checklist and takes the latest date among the relevant entries, which is consistent with deductive reasoning. The mistake occurs because it treats the March 16 entry as satisfying the capacity threshold, whereas the reference solution uses the March 22 entry. The error is not a failure of reasoning direction but an incorrect binding between the capacity condition and the OCR field within an otherwise valid deductive structure.

The inductive example in Figure 15 shows a different form of mismatch. The model correctly infers the latent mapping from sections to scores in the examples: high, above average, average, below average, and low correspond to descending scores. It then applies this rule to the target city. The final answer is incorrect because the target cell for Long Beach is assigned to the wrong section for one indicator. The induced pattern is process-compliant, but a local OCR-layout error changes the computed sum.

The abductive example in Figure 16 illustrates an error in backward recovery. The model starts from the observed total savings, sums the visible itemized components, and subtracts them to recover the hidden unlisted discount, matching the intended abductive schema. MRMS is zero because the model undercounts an explicit coupon subtotal before the final subtraction. This case shows that an abductive process can have the correct structure even when a small arithmetic or transcription error changes the recovered hidden premise.

The three cases explain why high RPCS does not necessarily imply high MRMS. The process judge can recognize a valid reasoning plan and a grounded intermediate structure, whereas answer scoring remains sensitive to exact OCR binding, table localization, and arithmetic precision. This separation helps diagnose errors: RPCS indicates whether the model selects the appropriate reasoning mode, whereas MRMS indicates whether the model executes the required OCR-grounded operations accurately enough to produce the correct answer.

## I OCR-MetaReasoning Examples

Figures 17, 18, and 19 show examples of the 15 meta-reasoning tasks in text-rich visual scenarios.

## J Benchmark Validation and Diagnostic Analyses

## J.1 Human Validation and Construction Evidence

We independently re-annotated 300 final benchmark samples (seed 42), with 20 samples from each of the 15 combinations of meta-reasoning type and OCR-object category. Three PhD students with research experience in multimodal large language models performed the validation. During construction, the annotators saw the proposed taxonomy labels because that stage involved taxonomyaligned candidate assessment. The separate reliability study was blind to the released labels and used independent re-annotation before adjudication. Table 8 combines the pre-adjudication agreement with the post-adjudication outcomes.

The main label disagreements involved boundary cases, such as abductive versus deductive bottlenecks or document logic versus field dependency. Representative disagreements concerned reasoning-direction boundaries, visually similar OCR-object categories, and image-grounding or validity judgments. After adjudication, all 300 validated samples matched the released meta-reasoning and OCR-object labels and satisfied the validity checklist. Table 9 summarizes the construction outcomes by meta-reasoning type; the final benchmark contains 100 samples in every taxonomy cell.

<table><tr><td colspan="5">(a) Pre-adjudication agreement</td></tr><tr><td>Field</td><td>Raw exact</td><td>Pairwise</td><td>Fleiss κ</td><td>Krippendorff α</td></tr><tr><td>Meta-reasoning type</td><td>96.0%</td><td>97.3%</td><td>0.960</td><td>0.960</td></tr><tr><td>OCR-object category</td><td>92.7%</td><td>95.1%</td><td>0.939</td><td>0.939</td></tr><tr><td>Answer uniqueness</td><td>100.0%</td><td>100.0%</td><td>1.000</td><td>1.000</td></tr><tr><td>Image groundedness</td><td>97.3%</td><td>98.2%</td><td>-0.009</td><td>-0.009</td></tr><tr><td>Multiple evidence required</td><td>97.0%</td><td>98.0%</td><td>0.490</td><td>0.490</td></tr><tr><td>Direct OCR solvability</td><td>99.3%</td><td>99.6%</td><td>-0.002</td><td>-0.002</td></tr><tr><td>Sample accepted</td><td>97.3%</td><td>98.2%</td><td>0.264</td><td>0.264</td></tr></table>

<table><tr><td colspan="3">(b) Post-adjudication outcome</td></tr><tr><td>Outcome</td><td>Count</td><td>Rate</td></tr><tr><td>Valid after adjudication</td><td>300/300</td><td>100.0%</td></tr><tr><td>No correction required</td><td>299/300</td><td>99.7%</td></tr><tr><td>Correction required</td><td>1/300</td><td>0.3%</td></tr><tr><td>Rejected</td><td>0/300</td><td>0.0%</td></tr><tr><td>Unique answer</td><td>300/300</td><td>100.0%</td></tr><tr><td>Image-grounded</td><td>300/300</td><td>100.0%</td></tr><tr><td>Multi-evidence required</td><td>300/300</td><td>100.0%</td></tr><tr><td>Not direct-OCR-solvable</td><td>300/300</td><td>100.0%</td></tr><tr><td>Released meta label match</td><td>300/300</td><td>100.0%</td></tr><tr><td>Released OCR-object label match</td><td>300/300</td><td>100.0%</td></tr></table>

Table 8: Human validation on 300 samples, with 20 samples from each of the 15 taxonomy cells. Pre-adjudication agreement is separated from the final adjudicated outcomes. Because the binary validity fields are highly skewed, chance-corrected coefficients should be interpreted together with raw and pairwise agreement.

Here, rejected/filtered candidates are successful syntheses that failed the human checklist, whereas pass-not-selected candidates passed verification but were omitted during balanced final sampling. Construction consists of synthesis, checklist-based filtering, and balanced selection; similar tasks were produced through separate prompt or image regeneration.

## J.2 RPCS Human Agreement and Judge Stability

We sampled 300 (sample, model output) pairs for RPCS validation. The validation set contains 100 pairs per meta-reasoning type, 60 pairs per OCRobject category, 20 pairs per taxonomy cell, 150 correct and 150 incorrect final-answer outcomes, 150 high- and 150 low-RPCS outcomes, and 75 pairs from each of the top-closed, mid-closed, bestopen, and small-open model groups. Three independent PhD annotators with research experience in multimodal large language models rated the four binary process criteria.

The annotators saw the image, question, target labels, reference reasoning steps, and the model process after the standalone final-answer lines were removed. They did not see the gold answer, predicted answer, MRMS, or judge score. Human raw agreement was 91.7%, 91.3%, 89.7%, and 90.0% for capability match, groundedness, step completeness, and non-hallucination, respectively; the corresponding Fleiss κ values were 0.856, 0.856, 0.862, and 0.866. Mean raw agreement was 90.7%, and the macro κ was 0.860. Table 10 combines the judge-to-human-majority results with setting-level stability.

The main judge is GPT-5.4 with temperature 0.0, top\_p=1.0, and max\_tokens=2048. Referencestep overlap has a Pearson correlation of 0.040 and a Spearman correlation of 0.036 with judge–human error. The no-reference variant is weaker, while the near-zero overlap diagnostic shows that RPCS is not a surface trace-matching score. MRMS remains the judge-independent primary leaderboard metric, and RPCS remains a visible-process diagnostic.

To isolate outcomes, the RPCS judge receives the image, question, target taxonomy labels, reference reasoning steps, and the visible reasoning process of the model after the standalone final-answer lines are removed. The gold answer, predicted answer, MRMS, and judge score are withheld. The judge assigns four binary process criteria: capability match, groundedness, step completeness, and non-hallucination. These criteria are aggregated into RPCS as defined above. Figure 5 gives the complete prompt template, including the criterion definitions, outcome-isolation instructions, structured response specification, and sample-level input fields. Sample-specific fields are instantiated for each evaluation pair.

<table><tr><td>Meta type</td><td>Generated</td><td>Synthesis success</td><td>Synthesis failed</td><td>Human pass</td><td>Human rejected/filtered</td><td>Final selected</td><td>Human pass not selected</td></tr><tr><td>Deductive</td><td>3,052</td><td>2,723</td><td>329</td><td>688</td><td>2,035</td><td>500</td><td>188</td></tr><tr><td>Inductive</td><td>1,650</td><td>1,281</td><td>369</td><td>824</td><td>457</td><td>500</td><td>324</td></tr><tr><td>Abductive</td><td>2,950</td><td>2,870</td><td>80</td><td>814</td><td>2,056</td><td>500</td><td>314</td></tr><tr><td>Overall</td><td>7,652</td><td>6,874</td><td>778</td><td>2,326</td><td>4,548</td><td>1,500</td><td>826</td></tr></table>

Table 9: Construction outcomes by meta-reasoning type. The human rejected/filtered column reports successful synthesis candidates that failed the human checklist; the human pass-not-selected column reports verified candidates omitted during balanced final sampling. The final dataset was formed through synthesis, checklist-based filtering, and balanced selection.
<table><tr><td colspan="6">(a) Main judge versus human majority</td></tr><tr><td>Criterion</td><td colspan="2">Accuracy</td><td colspan="2">Macro-F1</td><td>Kappa</td></tr><tr><td>Capability match</td><td colspan="2">94.7%</td><td colspan="2">0.923</td><td>0.847</td></tr><tr><td>Groundedness</td><td colspan="2">94.3%</td><td colspan="2">0.925</td><td>0.851</td></tr><tr><td>Step completeness</td><td colspan="2">95.0%</td><td colspan="2">0.950</td><td>0.900</td></tr><tr><td>Non-hallucination</td><td colspan="2">94.7%</td><td colspan="2">0.946</td><td>0.893</td></tr><tr><td colspan="6">(b) Setting-level calibration and stability</td></tr><tr><td>Setting</td><td>Pearson</td><td>Spearman</td><td>Macro-F1</td><td>Kappa</td><td>RPCS MAE Flip rate / std.</td></tr><tr><td>Main judge</td><td>0.934</td><td>0.931</td><td>0.936</td><td>0.873</td><td>0.050 reference setting</td></tr><tr><td>Judge rerun</td><td>0.896</td><td>0.895</td><td>0.905</td><td>0.811</td><td>0.073 3.1% / 0.040</td></tr><tr><td>No-reference judge</td><td>0.822</td><td>0.816</td><td>0.850</td><td>0.703</td><td>0.111 8.6% / n/a</td></tr></table>

Table 10: RPCS calibration and stability on 300 balanced model-output pairs. A representative disagreement concerned groundedness: the judge marked the process as grounded, whereas all three human annotators judged it ungrounded. This isolated disagreement does not affect the aggregate statistics.

## J.3 Uncertainty, Paired Tests, and Item Headroom

We computed confidence intervals with 1,000 sample-level bootstrap replicates over all 1,500 benchmark samples using seed 42. Table 11 gives the intervals for the three highest MRMS values and six paired tests.

Item-level scores show substantial headroom: 453/1,500 items receive full score from all 18 models, 214/1,500 have a mean model score below 0.5, and 118/1,500 have a mean model score below 0.25. The hardest cells are abductive–layout semantics at 63.2, deductive–layout semantics at 65.4, deductive–transaction analysis at 71.8, inductive– layout semantics at 72.2, and deductive–document logic at 73.2.

## J.4 Human Baseline

We evaluated a human baseline on the same 300- sample stratified subset (seed 42; 20 samples from each of the 15 taxonomy cells). Three PhD students with research experience in multimodal large language models received the image, question, and answer-format hint but no reference reasoning steps. Any item flagged as ambiguous by an annotator was conservatively counted as incorrect. Table 12 reports the model and human scores, ambiguity rates, confidence, and paired human–model gaps.

The human majority reaches an MRMS of 96.0, compared with 90.5 for Gemini-3.1-Pro-Preview. The majority ambiguity flag rate is 1.3%, the anyannotator rate is 4.0%, and mean confidence is 0.930. The subset is therefore practically solvable while still leaving measurable headroom above current models.

## J.5 External Benchmark Construct Comparison

The quantitative construct audit covers six related benchmarks; VisualPuzzles is included in the conceptual comparison table but is not included in this audit. LogicOCR-Real denotes the real-image subset of LogicOCR. Projectable and mixed/notapplicable counts partition each selected subset. Direct OCR, multi-evidence, and process annotation are non-mutually-exclusive diagnostic attributes; Direct OCR cases are included within the mixed/not-applicable counts and are not additional partition counts. The source collections contained 10,000 OCRBench v2 items, 1,069 OCR-Reasoning items, 1,680 LogicOCR-Real items, 150 Reasoning-OCR items, 1,188 MME-Reasoning items, and 448 LogicVista items. Table 13 reports the selected subsets and their projection statistics. None of the published evaluations for these benchmarks use exactly the same model versions as the 18-model evaluation reported here.

<table><tr><td colspan="3">(a) Top-three MRMS estimates</td></tr><tr><td>Model</td><td>MRMS</td><td>95% CI</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>89.3</td><td>[87.9,90.8]</td></tr><tr><td>GPT-5.4-Medium</td><td>87.7</td><td>[86.0, 89.3]</td></tr><tr><td>Doubao-Seed-2.0-Pro</td><td>86.4</td><td>[84.7, 88.0]</td></tr></table>

(b) Paired sample-level bootstrap tests
<table><tr><td>Comparison</td><td>Difference</td><td></td><td>95% CI Significant at 95%</td></tr><tr><td>Top model – second model</td><td>+1.7</td><td>[0.1, 3.1]</td><td>yes</td></tr><tr><td>Top closed-source – best open-source</td><td>+7.8</td><td>[6.1, 9.6] yes</td><td></td></tr><tr><td>Closed-source average – open-source average</td><td>+9.5</td><td>[8.5, 10.5] yes</td><td></td></tr><tr><td>Deduction – induction</td><td>-8.9</td><td>[-12.5, -5.2] yes</td><td></td></tr><tr><td>Deduction – abduction</td><td>-6.5</td><td>[-10.2, -3.0] yes</td><td></td></tr><tr><td>Layout semantics – non-layout average</td><td>-14.7</td><td>[-18.8, -10.8] yes</td><td></td></tr></table>

Table 11: Confidence intervals for the three highest MRMS values and paired tests from 1,000 sample-level bootstrap replicates over the 1,500 benchmark samples with seed 42. MRMS values and paired differences are reported in percentage points.

We evaluated the same eight model versions on selected subsets in a same-model rerun. Table 14 reports the resulting rank correlations. Because the model generations differ from those used in published evaluations, we did not map results across generations.

## J.6 OCR, Layout, and Answer-Format Controls

The control experiment uses the same 300-sample stratified subset and five representative models: Gemini-3.1-Pro-Preview, GPT-5.4-Medium, Qwen3-VL-235B-A22B-Thinking, Doubao-Seed-2.0-Lite, and Qwen3-VL-8B-Instruct. It compares six input conditions, including a third-party PP-OCRv6 transcript and an answer-format oracle. Because the task is not solvable without OCR text, we measure layout by comparing OCR transcript-only with layout-aware transcript rather than by introducing a layout-only condition. Table 15 reports the six conditions.

The values are five-model averages on the control subset, not 18-model results on the full benchmark. The delta column is relative to full-image input. Layout-aware transcript recovers part of the OCR-transcript loss, while third-party OCR and answer-format hints provide only small average gains.

## J.7 Artifact Provenance and Split Composition

We traced the image provenance of all 1,500 final samples by comparing each final image with the selected source collection. The final set contains 1,299 unmodified images and 201 edited/reconstructed images. All 201 edited/reconstructed samples are abductive, whereas the deductive and inductive splits contain 500 unmodified images each. Within the edited/reconstructed abductive split, the OCRobject counts are 27 data interpretation, 48 document logic, 45 field dependency, 55 layout semantics, and 26 transaction analysis samples.

We use edited/reconstructed to denote a final image whose content does not exactly match the image with the same name in the selected source collection. These are controlled edits of public OCR seeds, such as masking a field, inserting a distractor, changing a local value, or rebuilding a documentlike layout while preserving OCR-grounded evidence; they are not unconstrained images generated from scratch. Table 16 reports provenance, split performance, and image/text reuse statistics.

Gemini-3.1-Pro-Preview is the top model in both artifact splits. Because all edited/reconstructed samples are abductive, this split comparison is descriptive and does not isolate an editing effect. We report provenance and split composition here; the uniqueness of questions and question–answer pairs is reported in the benchmark validity and scoring analysis above.

(a) Human baseline and representative models
<table><tr><td>System</td><td>MRMS</td><td>95% CI</td><td>Ded.</td><td>Ind.</td><td>Abd.</td><td>Layout</td></tr><tr><td>Human PhD-student majority</td><td>96.0</td><td>[93.7, 98.0]</td><td>96.0</td><td>95.0</td><td>97.0</td><td>90.0</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>90.5</td><td>[87.2, 93.6]</td><td>83.5</td><td>93.5</td><td>94.5</td><td>75.8</td></tr><tr><td>GPT-5.4-Medium</td><td>89.3</td><td>[86.2, 92.3]</td><td>83.8</td><td>92.5</td><td>91.7</td><td>75.6</td></tr><tr><td>Doubao-Seed-2.0-Pro</td><td>86.6</td><td>[82.7, 90.2]</td><td>81.5</td><td>89.0</td><td>89.2</td><td>72.5</td></tr><tr><td>Qwen3-VL-235B-A22B-Thinking</td><td>83.0</td><td>[78.7, 87.2]</td><td>77.8</td><td>89.0</td><td>82.2</td><td>71.2</td></tr></table>

(b) Clarity and human–model paired gaps
<table><tr><td>Measure</td><td>Value</td></tr><tr><td>Majority ambiguity flag rate</td><td>1.3%</td></tr><tr><td>Any-annotator ambiguity flag rate</td><td>4.0%</td></tr><tr><td>Mean human confidence</td><td>0.930</td></tr><tr><td>Human minus Gemini-3.1-Pro-Preview</td><td>+5.5 [1.6, 9.2]</td></tr><tr><td>Human minus GPT-5.4-Medium</td><td>+6.7 [2.7, 10.5]</td></tr><tr><td>Human minus Doubao-Seed-2.0-Pro</td><td>+9.4 [5.6, 13.3]</td></tr><tr><td>Human minus Qwen3-VL-235B-A22B-Thinking</td><td>+13.0 [8.8, 17.5]</td></tr></table>

Table 12: Human baseline on the same 300-sample stratified subset used for validation. The annotators received the image, question, and answer-format hint but no reference steps; any annotator ambiguity flag was counted as incorrect. MRMS and reasoning-type scores are percentages, whereas mean confidence is reported on [0, 1]. Confidence intervals use 1,000 bootstrap replicates with seed 42.
<table><tr><td>Benchmark</td><td>Subset n</td><td>Projectable</td><td>Cells</td><td>Mixed/not-applicable</td><td>Direct OCR</td><td>Multi-evidence</td><td>Process annotation</td></tr><tr><td>LogicOCR-Real</td><td>300</td><td>205 (68.3%)</td><td>10/15</td><td>95 (31.7%)</td><td>1 (0.3%)</td><td>187 (62.3%)</td><td>0 (0.0%)</td></tr><tr><td>OCR-Reasoning</td><td>1,069</td><td>457 (42.8%)</td><td>8/15</td><td>612 (57.2%)</td><td>0 (0.0%)</td><td>317 (29.7%)</td><td>742 (69.4%)</td></tr><tr><td>Reasoning-OCR</td><td>150</td><td>69 (46.0%)</td><td>8/15</td><td>81 (54.0%)</td><td>1 (0.7%)</td><td>124 (82.7%)</td><td>0 (0.0%)</td></tr><tr><td>OCRBench v2</td><td>300</td><td>48 (16.0%)</td><td>5/15</td><td>252 (84.0%)</td><td>52 (17.3%)</td><td>291 (97.0%)</td><td>0 (0.0%)</td></tr><tr><td>LogicVista</td><td>300</td><td>52 (17.3%)</td><td>7/15</td><td>248 (82.7%)</td><td>0 (0.0%)</td><td>258 (86.0%)</td><td>100 (33.3%)</td></tr><tr><td>MME-Reasoning</td><td>300</td><td>0 (0.0%)</td><td>0/15</td><td>300 (100.0%)</td><td>18 (6.0%)</td><td>215 (71.7%)</td><td>0 (0.0%)</td></tr></table>

Table 13: Construct-level projection of six related benchmarks onto the OCR-MetaReasoning axes. Projectable and mixed/not-applicable counts partition each selected subset. Direct OCR, multi-evidence, and process annotation are non-mutually-exclusive diagnostic attributes; Direct OCR cases are included within the mixed/not-applicable counts and are not additional partition counts.

## K Difficulty-Stratified Qualitative Analysis

## K.1 Automatic Difficulty Buckets

We assign each of the 1,500 items to a bucket using its mean final-answer score across the 18 evaluated models: very-hard items have a mean score below 0.25, hard-only items have a mean score of at least 0.25 and below 0.5, middle items have a mean score in [0.5, 1.0), and easy items receive full score from all 18 models. The aggregate hard set contains 214 items, consisting of 118 very-hard items and 96 hard-only items. Table 17 combines the bucket composition with the RPCS summary. From veryhard to easy, the average number of reference steps is 4.25, 4.27, 4.26, and 4.12, and the corresponding average question lengths are 61.4, 59.0, 63.1, and 60.9 words, respectively.

## K.2 Qualitative Coding

The qualitative-analysis subset contains 123 items: 34 very-hard, 44 hard-only, and 45 easy. The 44 hard-only items form a qualitative-analysis subset distinct from the full automatic hard-only bucket of 96 items and the aggregate hard set of 214 items. Of the 123 items, 107 required image-dependent checks. Table 18 reports the primary-code distribution.

The hard/easy coding standard uses H1 for crossregion layout binding, H2 for fine visual/OCR cues, H3 for multi-candidate constraint search, H4 for exception/boundary/negation handling, H5 for hidden backward dependency, H6 for implicit pattern induction, H7 for compositional arithmetic, and H8 for structured exact output. Easy codes are E1 for salient local evidence, E2 for common template calculation, and E3 for bounded candidate space. H8 is commonly a secondary code; primary codes follow the dominant mechanism defined by the coding protocol.

<table><tr><td>External subset</td><td>n</td><td>3×5-projectable n</td><td>Spearman ρ</td><td>Kendall τ</td></tr><tr><td>OCR-Reasoning</td><td>300</td><td>191</td><td>0.905</td><td>0.786</td></tr><tr><td>LogicOCR-Real</td><td>300</td><td>140</td><td>0.695</td><td>0.630</td></tr><tr><td>Reasoning-OCR</td><td>150</td><td>69</td><td>0.405</td><td>0.357</td></tr></table>

Table 14: Same-model rerun of the same eight model versions on selected external subsets. Results were not mapped across model generations.
<table><tr><td>Condition</td><td>MRMS</td><td>∆ vs. full image</td><td>Ded. Ind.</td><td></td><td>Abd. Layout JSON</td><td></td></tr><tr><td>Full image</td><td>80.7</td><td>0.0</td><td>76.8 82.4</td><td>83.1</td><td>66.3</td><td>75.4</td></tr><tr><td>Question-only</td><td>16.5</td><td>-64.3</td><td>15.4 24.2</td><td>9.9</td><td>26.4</td><td>21.0</td></tr><tr><td>OCR transcript-only</td><td>71.5</td><td>-9.2</td><td>64.7 77.6</td><td>72.3</td><td>59.2</td><td>66.5</td></tr><tr><td>Layout-aware transcript</td><td>74.4</td><td>-6.4</td><td>70.3 78.4</td><td>74.4</td><td>63.5</td><td>69.1</td></tr><tr><td>Image + third-party OCR</td><td>81.3</td><td>+0.5</td><td>77.0 83.6</td><td>83.2</td><td>67.3</td><td>73.2</td></tr><tr><td>Answer-format oracle</td><td>81.7</td><td>+0.9</td><td>78.5 82.4</td><td>84.1</td><td>67.8</td><td>77.8</td></tr></table>

Table 15: Five-model averages on the 300-sample stratified control subset. All score entries are reported as percentages. The models are Gemini-3.1-Pro-Preview, GPT-5.4-Medium, Qwen3-VL-235B-A22B-Thinking, Doubao-Seed-2.0-Lite, and Qwen3-VL-8B-Instruct. Third-party OCR uses PP-OCRv6. Layout is controlled by comparing OCR transcript-only with layout-aware transcript because a solvable layout-only condition would contain no OCR text.

## K.3 Eight Representative Case Panels

Figures 6, 7, 8, 9, 10, 11, 12, and 13 show eight representative cases: three very-hard cases, three hard cases, and two easy cases. Each panel contains the original image, the question, the gold answer, representative model outputs, a human diagnosis, and the expected failure mechanism.

<table><tr><td colspan="5">(a) Artifact provenance and split performance</td></tr><tr><td>Artifact split / measure</td><td>Count</td><td>Percent</td><td></td><td>Top split model Top / mean-model MRMS</td></tr><tr><td>Final samples with verified provenance</td><td>1,500/1,500</td><td>100.0%</td><td></td><td></td></tr><tr><td>Unmodified</td><td>1,299</td><td>86.6%</td><td>Gemini-3.1-Pro-Preview</td><td>89.3 / 79.1</td></tr><tr><td>Edited/reconstructed</td><td>201</td><td>13.4%</td><td>Gemini-3.1-Pro-Preview</td><td>89.4 / 76.1</td></tr><tr><td colspan="5">(b) Image and text reuse analysis</td></tr><tr><td>Unit</td><td></td><td>Repeated groups Items in repeated groups</td><td>Additional items</td><td>Largest group</td></tr><tr><td>Final image content</td><td>76</td><td>156</td><td>80</td><td>3</td></tr><tr><td>Source image content</td><td>112</td><td>235</td><td>123</td><td>3</td></tr><tr><td>Normalized question strings</td><td>0</td><td>0</td><td>0</td><td>1</td></tr><tr><td>Normalized question-answer pairs</td><td>0</td><td>0</td><td>0</td><td>1</td></tr></table>

Table 16: Aggregate artifact provenance, split performance, and image and text reuse analysis. The provenance analysis distinguishes source reuse from repeated questions and question–answer pairs. Top / mean-model MRMS values are reported in percentage points.

<table><tr><td>Bucket</td><td></td><td>Items Model-output pairs Mean answer score ([0,1])</td><td></td><td>RPCS ([0,1])</td><td>Capability ([0,1])</td><td>Groundedness ([0,1])</td><td>Step completeness ([0,1]) Non-hallucination ([0,1])</td><td></td></tr><tr><td>Very-hard</td><td>118</td><td>2,124</td><td>0.088</td><td>0.679</td><td>0.835</td><td>0.772</td><td>0.572</td><td>0.536</td></tr><tr><td>Hard-only</td><td>96</td><td>1,728</td><td>0.347</td><td>0.676</td><td>0.815</td><td>0.766</td><td>0.583</td><td>0.541</td></tr><tr><td>Middle</td><td>833</td><td>14,994</td><td>0.821</td><td>0.877</td><td>0.939</td><td>0.924</td><td>0.820</td><td>0.826</td></tr><tr><td>Easy all-18-full</td><td>453</td><td>8,154</td><td>1.000</td><td>0.965</td><td>0.985</td><td>0.992</td><td>0.922</td><td>0.959</td></tr></table>

Table 17: Automatic difficulty buckets based on the 18-model item-level mean score, with RPCS aggregated over model-output pairs. All score columns in this table remain on the normalized [0, 1] scale; very-hard items have a mean score below 0.25, hard-only items have a mean score in [0.25, 0.5), and easy items receive full score from all 18 models. The aggregate hard set contains 214 items: 118 very-hard items and 96 hard-only items.

<table><tr><td>Bucket</td><td>n</td><td>Primary-code distribution</td></tr><tr><td>Very-hard</td><td>34</td><td>H1 9 (26.5%), H2 4 (11.8%), H3 3 (8.8%), H48 (23.5%), H5 4 (11.8%), H6 6 (17.6%)</td></tr><tr><td>Hard-only</td><td>44</td><td>H1 9 (20.5%), H2 5 (11.4%), H412 (27.3%), H5 8 (18.2%), H6 10 (22.7%)</td></tr><tr><td>Easy</td><td>45</td><td>E2 35 (77.8%), E3 10 (22.2%)</td></tr><tr><td>Total</td><td>123</td><td>Image-dependent checks: 107</td></tr></table>

Table 18: Primary-code distribution in the qualitative-analysis subset. The automatic hard-only bucket contains 96 items, whereas the qualitative-analysis hard-only subset contains 44 items; these are distinct quantities.

![](images/4d0ed2da857dda2cb522ccaf3fe30dcc35a407712abfef5829150e98be4c9900.jpg)  
Figure 5: RPCS judge prompt template used with GPT-5.4 at temperature 0.0, top\_p=1.0, and max\_tokens=2048. The template specifies outcome isolation, the four binary process criteria, the structured response format, and sample-level input fields; gold answer, predicted answer, MRMS, and standalone final-answer content are excluded from process scoring.

![](images/f8dff65de43890773408dadcde540839231a55545b89c51cb451c79e119486e9.jpg)  
Figure 6: A very-hard H1 case illustrating cross-region layout binding in meta-abductive reasoning.

![](images/a6441cf15522a4b9da8a82bc624a1efa4b968f1938e96af00ff23d23aa40721b.jpg)  
Figure 7: A very-hard H4 case illustrating exception and boundary rules in meta-deductive reasoning.

![](images/6beab2405d451b171072a7c52af250322a15edb8cd9abb9d3469790c37379298.jpg)  
Figure 8: A very-hard H6 case illustrating implicit discourse-pattern induction in meta-inductive reasoning.

![](images/ac050af441364f17eaebbde478619ae0d628b7169951b4faa33d2f0a4122cd00.jpg)  
Figure 9: A hard H5 case illustrating hidden backward dependency under tax constraints in meta-abductive reasoning.

![](images/b6b4d4039bc6e2c25f68c0271ee64d0e3a298a582be5b79435e4074c7c8c0f8d.jpg)

Figure 10: A hard H2 case illustrating fine visual/OCR cues in meta-abductive reasoning.  
![](images/a5afd523cd444d4a70af475d3f156d3b653bbba1cfb02725cb440b32b521d179.jpg)  
Figure 11: A hard H1 case illustrating cross-region layout binding in meta-abductive reasoning.

![](images/0d54873c2449495ceae1d9dd96ce7fae2f174c486467e4a84d772793f956c6dd.jpg)

Figure 12: An easy E2 case illustrating a common template calculation in meta-abductive reasoning.  
![](images/efd5d04ab69da547cf32a3affde067efc0a2764d175f78e3cc727d91732a7e75.jpg)  
Figure 13: An easy E3 case illustrating a bounded candidate space and a deterministic decision in meta-deductive reasoning.

![](images/053cc736e538693b572b696c9d983c9b886520dc47497d45b0326a18a7a72408.jpg)  
Figure 14: Deductive case in which RPCS passes while MRMS fails. The model follows the constraint-checking structure but binds the capacity condition to the wrong timeline entry.

![](images/c6d74284bf125c2c25a84212db3b97e2a3a19cd32066d4c8aef60f7ed56b21b2.jpg)  
Figure 15: Inductive case in which RPCS passes while MRMS fails. The model induces the correct section-to-score rule but reads the target cell from the wrong section.

Abductive Case: RPCS Passes While MRMS Fails  
![](images/2f02487732d2969303ad4d1cba99d09f522eeea46aab3cd8f7023f76410a8f6d.jpg)  
Figure 16: Abductive case in which RPCS passes while MRMS fails. The model uses the correct backward-recovery schema but undercounts an explicit coupon subtotal before subtraction.

<table><tr><td>Model</td><td>String</td><td>Integer</td><td>Float</td><td>JSON</td><td>MRMS</td></tr><tr><td colspan="6">Closed-source Models</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>93.6</td><td>90.0</td><td>95.2</td><td>77.0</td><td>89.3</td></tr><tr><td>GPT-5.4-Medium</td><td>88.3</td><td>90.4</td><td>92.5</td><td>78.5</td><td>87.7</td></tr><tr><td>Doubao-Seed-2.0-Pro</td><td>86.5</td><td>87.7</td><td>91.1</td><td>81.0</td><td>86.4</td></tr><tr><td>Claude-Sonnet-4.6</td><td>87.7</td><td>85.9</td><td>84.9</td><td>74.6</td><td>84.3</td></tr><tr><td>Doubao-Seed-2.0-Lite</td><td>84.2</td><td>85.7</td><td>89.0</td><td>78.7</td><td>84.2</td></tr><tr><td>GPT-5.4-Mini-Medium</td><td>84.4</td><td>85.7</td><td>90.4</td><td>76.4</td><td>84.0</td></tr><tr><td>Gemini-3.1-Flash-Lite</td><td>86.5</td><td>82.9</td><td>82.9</td><td>78.4</td><td>83.3</td></tr><tr><td>GPT-5.4</td><td>84.2</td><td>83.6</td><td>88.4</td><td>77.0</td><td>83.0</td></tr><tr><td>Grok-4.20</td><td>73.9</td><td>78.6</td><td>74.7</td><td>71.1</td><td>75.2</td></tr><tr><td>GPT-5.4-Mini</td><td>70.2</td><td>75.6</td><td>73.3</td><td>67.7</td><td>72.0</td></tr><tr><td colspan="6">Open-source Models</td></tr><tr><td>Qwen3-VL-235B-A22B-Thinking</td><td>83.0</td><td>82.6</td><td>84.1</td><td>76.7</td><td>81.5</td></tr><tr><td>Kimi-K2.5</td><td>78.9</td><td>81.3</td><td>80.1</td><td>77.3</td><td>79.6</td></tr><tr><td>Qwen3-VL-235B-A22B-Instruct</td><td>77.4</td><td>79.9</td><td>76.6</td><td>71.2</td><td>77.0</td></tr><tr><td>GLM-4.6V</td><td>77.2</td><td>76.8</td><td>74.7</td><td>72.5</td><td>75.9</td></tr><tr><td>Qwen3-VL-30B-A3B-Thinking</td><td>74.9</td><td>78.4</td><td>71.9</td><td>68.5</td><td>74.7</td></tr><tr><td>Qwen3-VL-8B-Thinking</td><td>75.9</td><td>76.8</td><td>71.9</td><td>69.5</td><td>74.5</td></tr><tr><td>Qwen3-VL-30B-A3B-Instruct</td><td>65.5</td><td>66.7</td><td>58.9</td><td>64.2</td><td>65.0</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>60.0</td><td>60.4</td><td>52.4</td><td>58.0</td><td>59.0</td></tr></table>

Table 19: Extended answer-level results by answer type. MRMS is the model-level score reported in Table 2. All scores are percentages.

<table><tr><td>Model</td><td>DED.</td><td>IND.</td><td>ABD.</td><td>MRMS</td></tr><tr><td colspan="5">Closed-source Models</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>80.7</td><td>93.1</td><td>94.1</td><td>89.3</td></tr><tr><td>GPT-5.4-Medium</td><td>80.6</td><td>91.5</td><td>90.8</td><td>87.7</td></tr><tr><td>Doubao-Seed-2.0-Pro</td><td>80.1</td><td>90.2</td><td>88.9</td><td>86.4</td></tr><tr><td>Claude-Sonnet-4.6</td><td>77.5</td><td>87.6</td><td>87.9</td><td>84.3</td></tr><tr><td>Doubao-Seed-2.0-Lite</td><td>76.7</td><td>89.2</td><td>86.8</td><td>84.2</td></tr><tr><td>GPT-5.4-Mini-Medium</td><td>77.7</td><td>87.2</td><td>87.1</td><td>84.0</td></tr><tr><td>Gemini-3.1-Flash-Lite</td><td>78.4</td><td>85.4</td><td>86.2</td><td>83.3</td></tr><tr><td>GPT-5.4</td><td>77.9</td><td>85.7</td><td>85.5</td><td>83.0</td></tr><tr><td>Grok-4.20 GPT-5.4-Mini</td><td>70.3 70.7</td><td>77.4 75.3</td><td>78.0 70.1</td><td>75.2 72.0</td></tr><tr><td colspan="5"></td></tr><tr><td>Qwen3-VL-235B-A22B-Thinking</td><td>Open-source Models 74.8</td><td>87.8</td><td>82.1</td><td>81.5</td></tr><tr><td>Kimi-K2.5</td><td>71.8</td><td>84.6</td><td>82.5</td><td>79.6</td></tr><tr><td>Qwen3-VL-235B-A22B-Instruct</td><td>72.5</td><td>80.9</td><td>77.7</td><td>77.0</td></tr><tr><td>GLM-4.6V</td><td>72.0</td><td>79.2</td><td>76.3</td><td>75.9</td></tr><tr><td>Qwen3-VL-30B-A3B-Thinking</td><td>69.1</td><td>81.8</td><td>73.3</td><td>74.7</td></tr><tr><td>Qwen3-VL-8B-Thinking</td><td>70.3</td><td>79.5</td><td>73.6</td><td>74.5</td></tr><tr><td>Qwen3-VL-30B-A3B-Instruct</td><td>64.0</td><td>67.9</td><td>63.2</td><td>65.0</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>59.1</td><td>60.5</td><td>57.6</td><td>59.0</td></tr></table>

Table 20: Extended answer-level results by meta-reasoning type. DED., IND., and ABD. denote deductive, inductive, and abductive meta-reasoning. MRMS is the macro average over the three types, consistent with Table 2. All scores are percentages.

<table><tr><td>Model</td><td>D.I.</td><td>D.L.</td><td>F.D.</td><td>L.S.</td><td>T.A.</td><td>Avg.</td></tr><tr><td colspan="7">Closed-source Models</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>96.0</td><td>93.0</td><td>91.7</td><td>76.2</td><td>89.7</td><td>89.3</td></tr><tr><td>GPT-5.4-Medium</td><td>92.3</td><td>90.8</td><td>90.0</td><td>77.6</td><td>87.5</td><td>87.7</td></tr><tr><td>Doubao-Seed-2.0-Pro</td><td>91.7</td><td>87.8</td><td>87.9</td><td>75.1</td><td>89.4</td><td>86.4</td></tr><tr><td>Claude-Sonnet-4.6</td><td>88.5</td><td>88.1</td><td>89.1</td><td>69.9</td><td>86.0</td><td>84.3</td></tr><tr><td>Doubao-Seed-2.0-Lite</td><td>86.8</td><td>86.3</td><td>88.3</td><td>72.3</td><td>87.4</td><td>84.2</td></tr><tr><td>GPT-5.4-Mini-Medium</td><td>86.7</td><td>87.8</td><td>87.2</td><td>73.6</td><td>84.7</td><td>84.0</td></tr><tr><td>Gemini-3.1-Flash-Lite</td><td>87.7</td><td>84.9</td><td>87.8</td><td>72.2</td><td>83.8</td><td>83.3</td></tr><tr><td>GPT-5.4</td><td>87.4</td><td>85.0</td><td>88.0</td><td>69.1</td><td>85.7</td><td>83.0</td></tr><tr><td>Grok-4.20</td><td>79.3</td><td>76.6</td><td>79.8</td><td>61.9</td><td>78.3</td><td>75.2</td></tr><tr><td>GPT-5.4-Mini</td><td>72.7</td><td>75.1</td><td>76.9</td><td>61.0</td><td>74.4</td><td>72.0</td></tr><tr><td colspan="7">Open-source Models</td></tr><tr><td>Qwen3-VL-235B-A22B-Thinking</td><td>87.3</td><td>86.3</td><td>82.7</td><td>69.2</td><td>82.2</td><td>81.5</td></tr><tr><td>Kimi-K2.5</td><td>86.5</td><td>77.5</td><td>81.2</td><td>70.2</td><td>82.8</td><td>79.6</td></tr><tr><td>Qwen3-VL-235B-A22B-Instruct</td><td>80.7</td><td>78.8</td><td>80.7</td><td>69.5</td><td>75.5</td><td>77.0</td></tr><tr><td>GLM-4.6V</td><td>83.6</td><td>78.1</td><td>79.3</td><td>60.3</td><td>77.9</td><td>75.9</td></tr><tr><td>Qwen3-VL-30B-A3B-Thinking</td><td>81.5</td><td>78.6</td><td>77.8</td><td>60.5</td><td>75.1</td><td>74.7</td></tr><tr><td>Qwen3-VL-8B-Thinking</td><td>80.7</td><td>76.8</td><td>77.3</td><td>60.6</td><td>76.8</td><td>74.5</td></tr><tr><td>Qwen3-VL-30B-A3B-Instruct</td><td>72.1</td><td>66.6</td><td>67.0</td><td>55.1</td><td>64.4</td><td>65.0</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>58.3</td><td>65.5</td><td>62.4</td><td>49.9</td><td>59.1</td><td>59.0</td></tr></table>

Table 21: Extended answer-level results by OCR-object category. D.I., D.L., F.D., L.S., and T.A. denote data interpretation, document logic, field dependency, layout semantics, and transaction analysis. Avg. is the mean over the five object categories, consistent with Table 2. All scores are percentages.

<table><tr><td>Model</td><td>CM</td><td>GRD</td><td>STEP</td><td>NH</td><td>Avg.</td></tr><tr><td colspan="6">Closed-source Models</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>98.7</td><td>98.7</td><td>92.6</td><td>94.5</td><td>96.2</td></tr><tr><td>GPT-5.4-Medium</td><td>98.5</td><td>97.7</td><td>91.4</td><td>93.1</td><td>95.2</td></tr><tr><td>Doubao-Seed-2.0-Pro</td><td>96.7</td><td>96.6</td><td>87.5</td><td>91.3</td><td>93.0</td></tr><tr><td>Claude-Sonnet-4.6</td><td>97.1</td><td>95.3</td><td>91.6</td><td>83.7</td><td>91.9</td></tr><tr><td>Doubao-Seed-2.0-Lite</td><td>97.0</td><td>96.5</td><td>90.3</td><td>89.7</td><td>93.3</td></tr><tr><td>GPT-5.4-Mini-Medium</td><td>94.9</td><td>94.1</td><td>75.1</td><td>91.7</td><td>88.9</td></tr><tr><td>Gemini-3.1-Flash-Lite</td><td>93.3</td><td>95.7</td><td>82.8</td><td>85.3</td><td>89.3</td></tr><tr><td>GPT-5.4</td><td>96.9</td><td>95.8</td><td>89.3</td><td>88.1</td><td>92.5</td></tr><tr><td>Grok-4.20</td><td>93.7</td><td>91.7</td><td>81.5</td><td>76.0</td><td>85.7</td></tr><tr><td>GPT-5.4-Mini</td><td>91.3</td><td>87.7</td><td>68.5</td><td>78.0</td><td>81.4</td></tr><tr><td colspan="6">Open-source Models</td></tr><tr><td>Qwen3-VL-235B-A22B-Thinking</td><td>95.1</td><td>93.1</td><td>85.4</td><td>84.7</td><td>89.6</td></tr><tr><td>Kimi-K2.5 Qwen3-VL-235B-A22B-Instruct</td><td>96.7</td><td>96.0 92.4</td><td>89.7</td><td>82.0</td><td>91.1</td></tr><tr><td>GLM-4.6V</td><td>93.5 90.0</td><td>88.1</td><td>84.2 75.4</td><td>77.9 82.9</td><td>87.0</td></tr><tr><td>Qwen3-VL-30B-A3B-Thinking</td><td>91.5</td><td>89.8</td><td>76.7</td><td>79.5</td><td>84.1</td></tr><tr><td>Qwen3-VL-8B-Thinking</td><td>91.1</td><td>86.9</td><td>75.1</td><td>77.5</td><td>84.4 82.7</td></tr><tr><td>Qwen3-VL-30B-A3B-Instruct</td><td></td><td>83.6</td><td>69.3</td><td>66.0</td><td></td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>86.1 83.5</td><td>80.5</td><td>62.2</td><td>63.7</td><td>76.3 72.5</td></tr></table>

Table 22: Extended RPCS criterion pass rates. CM, GRD, STEP, and NH denote capability match, groundedness, step completeness, and non-hallucination. Avg. is the normalized RPCS reported in percentage points, consistent with Table 3. All scores are percentages.

<table><tr><td>Type</td><td>D.I.</td><td>D.L.</td><td>F.D.</td><td>L.S.</td><td>T.A.</td><td>Avg.</td></tr><tr><td colspan="7">All Models</td></tr><tr><td>DED.</td><td>83.1</td><td>73.2</td><td>74.4</td><td>65.4</td><td>71.8</td><td>73.6</td></tr><tr><td>IND.</td><td>80.5</td><td>87.3</td><td>85.9</td><td>72.2</td><td>86.8</td><td>82.5</td></tr><tr><td>ABD.</td><td>86.5</td><td>83.7</td><td>85.7</td><td>63.2</td><td>81.7</td><td>80.1</td></tr><tr><td colspan="7">Closed-source Models</td></tr><tr><td>DED.</td><td>88.1</td><td>76.3</td><td>78.3</td><td>67.7</td><td>74.9</td><td>77.0</td></tr><tr><td>IND.</td><td>84.1</td><td>91.6</td><td>89.5</td><td>75.5</td><td>90.5</td><td>86.2</td></tr><tr><td>ABD.</td><td>88.5</td><td>88.7</td><td>92.2</td><td>69.6</td><td>88.7</td><td>85.5</td></tr><tr><td colspan="7">Open-source Models</td></tr><tr><td>DED.</td><td>76.8</td><td>69.2</td><td>69.5</td><td>62.6</td><td>67.9</td><td>69.2</td></tr><tr><td>IND.</td><td>76.0</td><td>81.8</td><td>81.2</td><td>68.2</td><td>82.3</td><td>77.9</td></tr><tr><td>ABD.</td><td>84.0</td><td>77.3</td><td>77.7</td><td>55.1</td><td>72.9</td><td>73.4</td></tr></table>

Table 23: Mean answer-level scores for the 3 × 5 OCR-MetaReasoning taxonomy. Each cell averages model scores for the corresponding meta-reasoning type and OCR-object category. All scores are percentages.

![](images/14462141ea5a1eccc42aa02b1931226e728ca45a29b8d4879ae569d3aab35ee5.jpg)  
Figure 17: Examples of meta-deductive reasoning tasks in OCR-MetaReasoning.

![](images/bf31888c03e08f284cb8ee7f664228c190286bd860eb039fc9d40e73add56ed2.jpg)  
Figure 18: Examples of meta-inductive reasoning tasks in OCR-MetaReasoning.

![](images/96210530ce17ebd7c776527b4d1cb38beecf08a941382c32d86d7c125d14455d.jpg)  
Figure 19: Examples of meta-abductive reasoning tasks in OCR-MetaReasoning.

<table><tr><td>Model</td><td>Reasoning Mode</td><td>Model Link</td></tr><tr><td colspan="3">Closed-source Models</td></tr><tr><td>Doubao-Seed-2.0-Lite</td><td>√</td><td>https://seed.bytedance.com/en/seed2</td></tr><tr><td>Doubao-Seed-2.0-Pro</td><td>√</td><td>https://seed.bytedance.com/en/seed2</td></tr><tr><td>GPT-5.4-Mini</td><td>X</td><td>https://developers.openai.com/api/docs/ models/gpt-5.4-mini</td></tr><tr><td>GPT-5.4-Mini-Medium</td><td>√</td><td>https://developers.openai.com/api/docs/ models/gpt-5.4-mini</td></tr><tr><td>Gemini-3.1-Flash-Lite</td><td>V</td><td>https://ai.google.dev/gemini-api/docs/ models/gemini-3.1-flash-lite</td></tr><tr><td>Gemini-3.1-Pro-Preview</td><td>√</td><td>https://ai.google.dev/gemini-api/docs/ models/gemini-3.1-pro-preview</td></tr><tr><td>GPT-5.4</td><td>X</td><td>https://developers.openai.com/api/docs/ models/gpt-5.4</td></tr><tr><td>GPT-5.4-Medium</td><td>√</td><td>https://developers.openai.com/api/docs/ models/gpt-5.4</td></tr><tr><td>Claude-Sonnet-4.6</td><td>√</td><td>https://www.anthropic.com/claude/sonnet</td></tr><tr><td>Grok-4.20</td><td>X</td><td>https://ai.azure.com/catalog/models/ grok-4-20-non-reasoning</td></tr><tr><td colspan="3">Open-source Models</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>X</td><td>https: //huggingface.co/Qwen/Qwen3-VL-8B-Instruct</td></tr><tr><td>Qwen3-VL-8B-Thinking</td><td>√</td><td>https: //huggingface.co/Qwen/Qwen3-VL-8B-Thinking</td></tr><tr><td>Qwen3-VL-30B-A3B-Instruct</td><td>X</td><td>https://huggingface.co/Qwen/ Qwen3-VL-30B-A3B-Instruct</td></tr><tr><td>Qwen3-VL-30B-A3B-Thinking</td><td>√</td><td>https://huggingface.co/Qwen/ Qwen3-VL-30B-A3B-Thinking</td></tr><tr><td>Qwen3-VL-235B-A22B-Instruct</td><td>X</td><td>https://huggingface.co/Qwen/ Qwen3-VL-235B-A22B-Instruct</td></tr><tr><td>Qwen3-VL-235B-A22B-Thinking</td><td>√</td><td>https://huggingface.co/Qwen/ Qwen3-VL-235B-A22B-Thinking</td></tr><tr><td>Kimi-K2.5</td><td>X</td><td>https: //huggingface.co/moonshotai/Kimi-K2.5</td></tr><tr><td>GLM-4.6V</td><td>√</td><td>https://huggingface.co/zai-org/GLM-4.6V</td></tr></table>

Table 24: Model sources used in the evaluation. The “Medium” suffix denotes a reasoning-effort setting rather than a separate model release.