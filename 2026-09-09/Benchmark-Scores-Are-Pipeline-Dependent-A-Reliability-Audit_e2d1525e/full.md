# Benchmark Scores Are Pipeline-Dependent: A Reliability Audit of Cybersecurity LLM Benchmarks

Aymene Berriche, Cathrine Shalby, Mohannad Alhanahnah, Yazan Boshmaf Qatar Computing Research Institute, HBKU

## Abstract

Large language model (LLM) benchmarks are often treated as fixed datasets with stable scores, yet their outcomes depend on configurable evaluation pipelines. We audit eight cybersecurity benchmarks across 10 proprietary, open-weight, and cybersecurity-specialized LLMs. By modeling benchmarks as measurement pipelines, we identify 15 systematic failure modes and show that a single pipeline choice can change a model’s score by more than 80 percentage points and substantially alter model rankings. At the cross-benchmark level, two semantically similar task pairs rank the same models differently because of incompatible evaluation conventions. Under an evaluation harness that standardizes pipeline choices while preserving task semantics, nine of 10 models shift by at least three ranks on at least one benchmark. These results show that cybersecurity LLM benchmark scores are pipeline-dependent and motivate pipeline-aware auditing as a core requirement for reliable model evaluation.

## 1 Introduction

Large language model (LLM) evaluation is increasingly benchmark-driven. Benchmark scores guide model selection, support claims of state-of-the-art performance, influence deployment decisions, and shape leaderboards. Yet, these scores are often interpreted as stable measurements of model capability, even though they are produced by configurable evaluation scripts involving prompts, inference settings, output extraction, scoring, and aggregation.

Prior work shows that LLM evaluation is sensitive to prompt wording, decoding, evaluator design, answer extraction, and scoring (Shi et al., 2024; Sun et al., 2024; Wang et al., 2023). These effects are amplified in generative settings, where outputs are open-ended, multiple answers may be valid, and evaluation often relies on heuristic extraction or approximate matching. Consequently, benchmark outcomes may reflect the measurement process as much as underlying model capability.

We study this problem in cybersecurity, a highstakes domain whose benchmarks span factual recall, vulnerability analysis, threat-intelligence extraction, attacker attribution, attack-technique mapping, mitigation selection, and general security reasoning. These tasks depend on evolving, structured authoritative sources such as CVE, CWE, CVSS, and MITRE ATT&CK (Sikos, 2023), making evaluation especially sensitive to pipeline errors and inconsistent labels. Such failures can distort claims of model specialization, obscure genuine capability differences, and mislead deployment decisions. Cybersecurity also provides a mature and heterogeneous benchmark ecosystem that has not been systematically audited, including in recent largescale meta-evaluations (Bean et al., 2025).

In this paper, we ask: to what extent do design choices ofevaluation pipelines affect the reliability of cybersecurity LLM benchmark outcomes? We argue that benchmarks should be viewed not as static datasets paired with fixed metrics, but as measurement pipelines that transform tasks and model outputs into numerical performance estimates (§2). Under this formulation, a benchmark score is conditional on its evaluation pipeline’s stages, consisting of dataset construction, prompt specification, inference, extraction and scoring, and aggregation.

We audit eight cybersecurity benchmarks comprising 48,662 questions across 23 tasks against 10 proprietary, open-weight, and cybersecurityspecialized LLMs (§3) and identify 15 recurring failure modes across the evaluation pipeline (§4). Fixing these failures can shift a model’s benchmark score by over 80 percentage points. For example, RedSage-Bench (Suryanto et al., 2026) uses “\n” as a stop sequence, causing generation to halt at the first newline. For Qwen3.6, this sequence fires inside the reasoning preamble before any answer token is produced, yielding empty outputs. Generating until the end-of-sequence token, using a token budget large enough for the reasoning span to close, and stripping the reasoning span before extraction, recovers 85.9 percentage points.

Across benchmarks, broader task coverage can still yield redundant measurements and unstable model rankings (§5). Using Principal Component Analysis (PCA), we find that the first component explains 95.25% of the variance across the 23 task × 10 model score matrix, showing that most tasks largely capture the same broad performance dimension. However, task-induced rankings can still disagree. Two semantically similar task pairs in CTI-Bench (Alam et al., 2024) and AthenaBench (Alam et al., 2025) show weak rank agreement under Kendall’s τ -b (0.29 and 0.24), with pairwise inversions revealing changes in specific model orderings. These discrepancies stem largely from incompatible scoring conventions, specifically binary versus partial-credit scoring and alias handling.

To isolate pipeline effects, we implement a harness that standardizes nine pipeline configuration fields, including prompt formatting, decoding, and extraction rules, wherever benchmark semantics permit. Under this harness, nine of 10 models shift by at least three ranks on at least one benchmark (§5). This has an important practical implication: cybersecurity-specialized models are often selected based on leaderboard rankings, yet those rankings can reflect evaluation-pipeline choices as much as underlying domain capability. We use these findings to formulate a set of recommendations for more reliable benchmarking (§6).

Our contributions are threefold. First, we formalize LLM benchmarks as measurement pipelines and audit every pipeline stage of eight cybersecurity benchmarks across 10 LLMs under a common framework. Second, we quantify stage-level effects through controlled perturbations of pipeline configuration fields, showing that pipeline choices affect both model scores and rankings. Third, we release an evaluation harness that makes heterogeneous benchmark assumptions explicit and reproducible. Although some individual failure modes identified in our audit have been documented before, our contribution is a systematic end-to-end audit of cybersecurity benchmarks under a single formalization. Our findings motivate treating benchmark scores as pipeline-dependent measurements and adopting benchmark audits, executable reference evaluators, invalid-response reporting, and reliability statistics as standard evaluation practice.

We note that our harness is intended as a diagnostic tool for exposing and correcting evaluationpipeline choices, rather than as a universally correct evaluator. While our audit methodology is domaingeneral, the observed failure rates are specific to the cybersecurity benchmarks studied. Our goal is to identify these failures and quantify their impact, rather than to propose failure-free benchmarks or a universally applicable evaluation harness. We publicly release our audit code<sup>1</sup> and evaluation harness,<sup>2</sup> and publish our meta-evaluation results on EveryEvalEver (Batzner et al., 2026).

## 2 Benchmarks as Measurement Pipelines

LLM benchmarks are often described as datasets paired with metrics, where a metric specifies how model performance is quantified and a score is the resulting numerical measurement. In practice, that score is produced by a multi-stage evaluation procedure in which each stage transforms the output of the preceding stage and introduces choices that can affect the final measurement. We therefore model a benchmark as a measurement pipeline, represented as a nested composition of stage-specific functions. For benchmark b and model $m ,$ , the reported score $\mathcal { S } _ { b } ( m )$ can be written as:

$$
\begin{array} { r } { S _ { b } ( m ) = \mathcal { A } _ { b } ( \mathcal { E } _ { b } ( \mathcal { T } _ { b } ( m , \mathcal { P } _ { b } ( \mathcal { D } _ { b } ) ) ) ) , } \end{array}\tag{1}
$$

where $\mathcal { D } _ { b }$ is the benchmark dataset, $\mathcal { P } _ { b }$ the prompt specification, $\mathcal { T } _ { b }$ the inference procedure, $\mathcal { E } _ { b }$ the extraction and scoring procedure, and $\mathcal { A } _ { b }$ the aggregation rule. Under this formulation, the benchmark score is the output of the composed evaluation pipeline rather than an intrinsic property of the model alone. As a result, $\mathcal { S } _ { b } ( m )$ is conditional on the full pipeline used to produce it.

Pipeline stages. Dataset construction $( \mathcal { D } _ { b } )$ defines the evaluation distribution, including task coverage, label quality, and answer representation. Prompt specification $( \mathcal { P } _ { b } )$ determines how tasks are presented to the model, including instructions, demonstrations, chat formatting, and output-format constraints. Inference configuration $( \mathcal { T } _ { b } )$ controls how responses are generated, including decoding parameters, token budgets, stop sequences, serving backends, and backend-specific constraints. Extraction and scoring $( \mathcal { E } _ { b } )$ converts model outputs into predictions or graded judgments. Aggregation $( \mathcal { A } _ { b } )$ combines question- and task-level measurements into final benchmark scores. Variation at any stage can therefore change reported performance without reflecting a change in model capability.

The five stages in Eq. 1 are instantiated through nine pipeline configuration fields, which we specify, perturb, and standardize throughout the paper: prompt template and chat formatting (P); decoding, maximum new tokens, and stop sequences (I); extraction rule, scoring rule, and denominator policy (E); and aggregation rule (A). App. D.4 records these fields for every benchmark.

## 2.1 Meta-Evaluation Methodology

The pipeline view separates benchmark reliability into two levels. At the benchmark level, we inspect each pipeline for stages where a design or implementation choice changes the reported score. We call such a pattern afailure mode: a recurring feature of the evaluation procedure that can change reported scores without any corresponding change in model capability. Examples include prompt templates that induce unparseable outputs and aggregation rules that combine non-equivalent metrics. At the cross-benchmark level, we ask whether the audited benchmarks support stable conclusions about which model performs better.

We denote the i-th failure mode at pipeline stage $\mathcal { X } _ { b } ~ \in ~ \{ \mathcal { D } _ { b } , \mathcal { P } _ { b } , \mathcal { T } _ { b } , \mathcal { E } _ { b } , \mathcal { A } _ { b } \}$ for benchmark b and model m as $\mathcal { F } _ { i } ( m , \mathcal { X } _ { b } )$ . When discussing failure modes independent of a particular benchmark or model, we use the shorthand $\mathcal { F } _ { i } ( \mathcal { X } )$ to denote the i-th failure mode at stage X.

We audit every stage of each benchmark, so no stage–benchmark pair is left unexamined. For each benchmark, we separately reconstruct the documented and released pipelines, record where they disagree, and identify any choices we must supply because neither source specifies them. We log raw model outputs together with the exact prompt, decoding parameters, and extraction rules used to produce them. Where a single pipeline configuration field can be varied while holding the others fixed, we perturb that field and measure the resulting change. This perturbation step is necessarily opportunistic, and App. F.1 records where such isolation is not possible. Because not all failure modes admit a meaningful per-model score shift, we quantify each using a measure appropriate to its mechanism, including score deltas, invalidresponse rates, extractor disagreement, denominator ratios, affected-question or benchmark shares, and rank changes under alternative but semantically equivalent implementations (App. F.6).

<table><tr><td>Benchmark</td><td>Tasks</td><td>Questions</td><td>Reference</td></tr><tr><td>MMLU-CS</td><td>1</td><td>100</td><td>(Hendrycks et al., 2020)</td></tr><tr><td>CyberMetric</td><td>1</td><td>500</td><td>(Tihanyi et al., 2024)</td></tr><tr><td>SecBench</td><td>1</td><td>661</td><td>(Jing et al., 2024)</td></tr><tr><td>SecEval</td><td>1</td><td>2,189</td><td>(Li et al., 2023)</td></tr><tr><td>SECURE</td><td>3</td><td>2,502</td><td>(Bhusal et al., 2024)</td></tr><tr><td>CTI-Bench</td><td>5</td><td>4,610</td><td>(Alam et al., 2024)</td></tr><tr><td>AthenaBench</td><td>6</td><td>8,100</td><td>(Alam et al., 2025)</td></tr><tr><td>RedSage-Bench</td><td>5</td><td>30,000</td><td>(Suryanto et al., 2026)</td></tr><tr><td>Total</td><td>23</td><td>48,662</td><td></td></tr></table>

Table 1: Cybersecurity benchmarks included in our audit. Tasks reports the number of scored tasks, while Questions reports the total number of scored questions across those tasks. The counts may differ from the reported or released dataset sizes; see App. B.1 for details.

Each failure-mode analysis is either a re-scoring analysis, in which we reuse stored model outputs while changing a downstream pipeline configuration field (e.g., the extraction rule), or a regeneration analysis, in which we produce new model outputs after changing an upstream pipeline configuration field (e.g., the prompt template). Rescoring holds model outputs fixed and therefore isolates the effect of the modified downstream field exactly. Re-generation, by contrast, captures the effect of changing an upstream field through the new outputs it induces. We classify each failure mode as re-scoring or re-generation in App. F.2.

At the cross-benchmark level, we use PCA to characterize redundancy in task-level scores and Kendall’s τ-b, together with pairwise rank inversions, to measure agreement among task-induced model rankings. Here, a pairwise inversion occurs when two models are ordered one way by one task and in the opposite order by another (App. H.3). While benchmark-level analysis identifies instability within individual benchmarks, cross-benchmark analysis examines whether different tasks provide distinct evidence and support consistent comparative conclusions.

## 3 Experimental Setup

We evaluate benchmark reliability across eight cybersecurity benchmarks and 10 LLMs, measuring how pipeline choices affect reported scores and comparative conclusions.

## 3.1 Benchmarks

Table 1 lists the benchmarks included in our audit. Together, they cover a broad range of cybersecurity evaluation tasks, including multiple-choice knowledge questions, vulnerability scoring, root-cause mapping, threat-actor attribution, attack-technique extraction, mitigation selection, and general cybersecurity reasoning. These benchmarks draw on structured and semi-structured authoritative cybersecurity sources, including CVE, CWE, CVSS, MITRE ATT&CK, vulnerability advisories, and cyber threat intelligence reports (Sikos, 2023).

We score each benchmark using its full released dataset except where the release structure or validation protocol requires a subset. For example, SecBench (Jing et al., 2024) reports 47,910 questions, but only 3,000 are publicly released, comprising 2,730 Multiple-Choice Questions (MCQs) and 270 Short-Answer Questions (SAQs). Of these, only 661 MCQs are in English, and none of the SAQs are. We therefore score a subset of 661 questions out of the reported 47,910. App. B.1 reports the reported, released, and scored sizes for every benchmark, together with the rationale for each subset, where applicable.

## 3.2 Models

We evaluate 10 LLMs spanning proprietary, openweight, and cybersecurity-specialized models, as summarized in Table 2. This mix allows us to test whether reliability failures are model-class specific or broader properties of the benchmark ecosystem. When a benchmark specifies an inference configuration, we use it as part of the original pipeline. Otherwise, we use a fixed configuration implemented in our evaluation harness (§3.3). Open-weight models are served locally, while proprietary models are evaluated through their respective APIs.

## 3.3 Evaluation Harness

We implement a common harness to isolate evaluation artifacts across cybersecurity benchmarks.<sup>3</sup> The harness standardizes the nine pipeline configuration fields defined in §2 wherever benchmark semantics permit. Importantly, these standardizations modify the evaluation procedure, not the task itself: benchmark questions, gold answers, and the intended capability being measured remain unchanged (App. D). The harness records the full evaluation trace, including raw model outputs, extracted predictions, question-level and aggregate scores, invalid-response rates, and pipeline configuration, making differences between original and standardized evaluations explicit and reproducible.

<table><tr><td>Model</td><td>Category</td><td>Reference</td></tr><tr><td>GPT-5.4 [Claude] Sonnet 4.6</td><td>Proprietary Proprietary</td><td>(Singh et al., 2025) (Anthropic, 2025)</td></tr><tr><td>Gemma-4[-31B]</td><td>Open-weight</td><td>(Google, 2026)</td></tr><tr><td>Qwen3.6[-35B]</td><td>Open-weight</td><td>(Yang et al., 2025)</td></tr><tr><td>Llama-3.3[-70B] GPT-OSS[-20B]</td><td>Open-weight</td><td>(Grattafiori et al., 2024)</td></tr><tr><td></td><td>Open-weight</td><td>(Agarwal et al., 2025)</td></tr><tr><td>Primus-Nemotron[-70B]</td><td>Cybersecurity</td><td>(Yu et al., 2025)</td></tr><tr><td>Primus-Merged[-8B]</td><td>Cybersecurity</td><td>(Yu et al., 2025)</td></tr><tr><td>Foundation-Sec[-8B]</td><td>Cybersecurity</td><td>(Yang et al., 2026)</td></tr><tr><td>RedSage-Qwen3[-8B-DPO]</td><td></td><td></td></tr><tr><td></td><td>Cybersecurity</td><td>(Suryanto et al., 2026)</td></tr></table>

Table 2: Evaluated LLMs grouped by model category. Bracketed segments are dropped when we refer to models elsewhere in the paper.

App. D documents each standardization field by field, distinguishing corrections to released implementations from choices we supply when a benchmark leaves a field unspecified. Some choices are not uniquely determined by the benchmark specification. In particular, answer extraction, invalidresponse handling, partial-credit scoring, and logprobability versus generative scoring admit defensible alternatives; Table 12 reports these alternatives and quantifies their effects. We also reviewed all eight benchmarks to determine whether outputformat compliance is part of the capability being evaluated. None treats format compliance as an evaluation objective. So, the standardized pipeline uses a consistent semantic extraction rule rather than marking superficial formatting differences as incorrect. Specifically, it applies one pinned LLMbased extraction and judging policy with a fixed judge model, prompt, and temperature (App. E.1). For example, the harness uses binary scoring while allowing the judge to recognize equivalent aliases between the gold answer and the model response.

## 4 Benchmark-Level Reliability Failures

We first audit reliability within individual benchmark pipelines. Across eight cybersecurity benchmarks, we identify 15 recurring failure modes spanning all five pipeline stages. Table 3 summarizes their observed effects. The main text highlights representative cases with the largest empirical impact, while Apps. G.1–G.4 provide additional evidence and per-model breakdowns.

<table><tr><td>ID</td><td>Failure mode</td><td>Impact measure</td><td>Affected units</td><td>Maximum effect Median effect</td><td></td></tr><tr><td> $\mathcal { F } _ { 1 } ( \mathcal { D } )$ </td><td>Limited capability coverage</td><td>Dominant question type</td><td>5 of 8 benchmarks 7 of 8 benchmarks</td><td>&gt;95% of questions are of one type 23.8% of 998 checked flags</td><td></td></tr><tr><td> $\mathcal { F } _ { 2 } ( \mathcal { D } )$ </td><td>Gold-label correctness</td><td>Flag precision</td><td></td><td></td><td></td></tr><tr><td> $\mathcal { F } _ { 1 } ( \mathcal { P } )$ </td><td>Format-token leakage</td><td>Peer score gap</td><td>4 of 10 models</td><td>90.9pp</td><td>81.3 pp</td></tr><tr><td> $\mathcal { F } _ { 2 } ( \mathcal { P } )$ </td><td>Prompt-question conflict</td><td>Single-letter response rate</td><td>8 of 10 models</td><td>33.0%</td><td>26.2%</td></tr><tr><td> $\mathcal { F } _ { 3 } ( \mathcal { P } )$ </td><td>Template incompatibility</td><td>Peer score gap</td><td>1 of 10 models</td><td>48pp</td><td></td></tr><tr><td> $\mathcal { F } _ { 1 } ( \mathbb { Z } )$ </td><td>Stop-sequence mismatch</td><td>Score gap</td><td>1 of 10 models</td><td> $8 5 . 9 \mathrm { p p }$ </td><td></td></tr><tr><td> $\mathcal { F } _ { 2 } ( \mathcal { T } )$ </td><td>Token-budget filter</td><td>Score gap</td><td>1 of 10 models</td><td> $8 1 \mathrm { p p }$ </td><td></td></tr><tr><td> $\mathcal { F } _ { 3 } ( \mathbb { Z } )$ </td><td>Decoding drift</td><td>Score gap</td><td>1 of 10 models</td><td> $4 0 \mathrm { p p }$ </td><td></td></tr><tr><td> ${ \mathcal { F } } _ { 1 } ( { \mathcal { E } } )$ </td><td>Extractor divergence</td><td>Extractor score gap</td><td>8 of 10 models</td><td> $7 9 . 7 \mathrm { p p }$ </td><td>1.3 pp</td></tr><tr><td> $\mathcal { F } _ { 2 } ( \mathcal { E } )$ </td><td>Denominator inflation</td><td>Score gap</td><td>3 benchmarks</td><td>99.8pp</td><td></td></tr><tr><td> $\mathcal { F } _ { 3 } ( \mathcal { E } )$ </td><td>Metric-direction mismatch</td><td>Rank shift</td><td>10 of 10 models</td><td>5 ranks</td><td>1 rank</td></tr><tr><td> $\mathcal { F } _ { 4 } ( \mathcal { E } )$ </td><td>Prompt-mode sensitivity</td><td>Score spread</td><td>10 of 10 models</td><td> $4 0 \mathrm { p p }$ </td><td>7pp</td></tr><tr><td> $\mathcal { F } _ { 1 } ( \mathcal { A } )$ </td><td>Logprob vs. generative scoring</td><td>Score gap</td><td>8 of 10 models</td><td> $4 0 . 9 \mathrm { p p }$ </td><td>2.8 pp</td></tr><tr><td> $\mathcal { F } _ { 2 } ( \mathcal { A } )$ </td><td>Task-level metric drift</td><td>Aggregation score gap</td><td>10 of 10 models</td><td> $7 0 \mathrm { p p }$ </td><td> $3 3 . 5 \mathrm { p p }$ </td></tr><tr><td> $\mathcal { F } _ { 3 } \left( \mathcal { A } \right)$ </td><td>Aggregation inconsistency</td><td>Denominator gap</td><td>2 of 8 benchmarks</td><td> $9 0 . 5 \mathsf { p p }$ </td><td></td></tr></table>

Table 3: Summary of the 15 recurring pipeline failure modes and their observed effects. Affected units reports where each failure was observed; maximum and median effects summarize the affected units when a per-unit distribution is available, and $^ { 6 6 } - ^ { 5 5 }$ indicates that a median is not meaningful. More information in App. F.6, App. G, and Table 16.

## 4.1 Dataset Failures

Dataset construction determines what a benchmark can measure before any model is queried.

$\mathcal { F } _ { 1 } ( \mathcal { D } )$ : Limited capability coverage. A domain benchmark should capture a meaningful range of the capabilities it is intended to assess. We therefore examine whether each benchmark covers both knowledge recall and analytical reasoning. Using a majority vote of four LLM classifiers, we classify each question type as either knowledge-oriented or analytical (App. G.1.1). Figure 1 shows that four benchmarks are highly skewed toward knowledgeoriented questions: RedSage-Bench, CyberMetric, MMLU-CS, and SecBench each contain over 95% knowledge-oriented questions. AthenaBench shows the opposite pattern, with analytical questions comprising over 95% of its questions. Hence, their aggregate scores capture only a narrow slice of domain capability, emphasizing either knowledge recall or analytical reasoning rather than both.

$\mathcal { F } _ { 2 } ( \mathcal { D } )$ : Gold-label correctness. Noise in gold labels is non-negligible. Model-majority disagreement flags 1,140 questions, of which an automated search-grounded verifier checks 998 (App. G.1.2). It confirms 238 (23.8%) as label errors and 653 (65.4%) as false positives. Importantly, the 23.8% figure is the precision of the flagging procedure on this disagreement-selected set, not a benchmarkwide label-error rate. Model disagreement is therefore useful for triage, but cannot by itself establish label correctness. In evolving domains such as cybersecurity, gold labels require explicit verification and uncertainty handling.

![](images/6b02ea6b9a688e2eb1638d0367e16a19d9b1cd211c654d55db4ab5a9fedc95b1.jpg)  
Figure 1: Knowledge-oriented and analytical question composition per benchmark using 4-classifier majority vote over 2,155 stratified questions (Fleiss’s κ=0.753).

## 4.2 Prompt Failures

Prompt choices can distort evaluation by changing task interpretation, expected response format, or compatibility with extraction rules, producing score differences unrelated to model capability.

$\mathcal { F } _ { 1 } ( \mathcal { P } )$ : Format-token leakage. In CyberMetric, the answer-format template contains a literal placeholder that some models interpret incorrectly. Four of the 10 models are affected: Foundation-Sec and Primus-Nemotron produce mostly empty outputs, Gemma-4 reproduces the placeholder, and Primus-Merged continues the instruction text instead of answering. Removing the placeholder restores valid responses for all four models (App. G.1.4).

$\mathcal { F } _ { 2 } ( \mathcal { P } )$ : Prompt–question conflict. In SecEval, benchmark-level instructions to “select the correct answers” sometimes conflict with question wording that suggests a single-select response, leading models to return one choice for questions that require multiple selections. Under set-exact-match scoring, these responses receive zero credit even when the selected choice is partially correct. Across the eight models that produce parseable output, singleselect responses occur on 10.1%–33.0% of the 927 multi-select questions (App. G.1.5).

$\mathcal { F } _ { 3 } ( \mathcal { P } )$ : Template incompatibility. In SECURE, Primus-Merged responds to MCQ questions with free-form explanations rather than the expected single- or multi-select answers. The benchmark extractor then treats the first answer-choice letter appearing in the explanation as the prediction, yielding a 48 percentage-point (pp) gap from peer models. This peer score gap reflects template incompatibility rather than task difficulty (App. G.1.6).

## 4.3 Inference Failures

Inference choices determine whether a model’s answer is generated and observable to the evaluator. Unlike prompt failures, which arise from task presentation, inference failures arise from generationtime pipeline configuration fields such as stop sequences, token budgets, and decoding parameters.

$\mathcal { F } _ { 1 } ( \mathcal { T } )$ : Stop-sequence mismatch. The largest inference-stage effect happens in RedSage-Bench, where the official stop sequence, the newline character $^ { 6 6 } \backslash \boldsymbol { \mathsf { n } } ^ { \prime 9 }$ , is triggered within Qwen3.6’s reasoning preamble before any answer token is produced. Generating until the end-of-sequence token, allowing sufficient output tokens for the reasoning span to close, and stripping that span before extraction increases the score by 85.9 pp (App. G.2.1).

$\mathcal { F } _ { 2 } ( \mathcal { T } )$ : Token-budget filter. In SecEval, the five-token output budget falls below Azure OpenAI’s 16-token minimum, causing every GPT-5.4 request to fail and return error payloads rather than model answers. Therefore, the resulting 0.3% score comes from accidental letter matches within these payloads. Raising the budget to 16 tokens restores valid generation and yields 81.4% accuracy (App. G.2.2), showing how an operational constraint can act as a hidden capability filter.

$\mathcal { F } _ { 3 } ( \mathcal { T } )$ : Decoding drift. In CyberMetric, the paper specifies sampling with temperature 1.0, top-p

0.9, and top-k 50, whereas the released script effectively defaults to greedy decoding. For Primus-Merged, this discrepancy changes output-format compliance and results in 40 pp score gap, where 274 of the 500 questions change correctness between the two configurations (App. G.2.3). The effect is therefore driven largely by generation behavior and format compliance rather than knowledge. So, inference configurations should be pinned in executable code, not only described in text.

## 4.4 Extraction and Scoring Failures

Most benchmark pipelines convert free-form generations into scored predictions, making extraction a major source of measurement error.

$\mathcal { F } _ { 1 } ( \mathcal { E } )$ : Extractor divergence. Strict extractors can fail when model outputs do not match their expected format. For example, regex extractors may miss correct answers when models use free-form responses, place answers in unexpected locations, or deviate from the prescribed answer template. In CTI-Bench vulnerability scoring, the prompt asks for the CVSS vector on the final line, while the released extractor takes the last vector found anywhere in the response. These rules disagree on 30.4% of model-item pairs where at least one extracts a vector, and on 96.5% of Primus-Merged’s pairs (App. G.3.1). RedSage-Bench introduces a related ambiguity by reporting three metrics on the same generations without designating one as canonical. Primus-Merged scores 0.3% under exact match but 80.0% under prefix match, resulting in 79.7 pp score gap (App. G.4.1). These findings show that extraction and scoring rules should be explicit and consistent.

$\mathcal { F } _ { 2 } ( \mathcal { E } )$ : Denominator inflation. In CTI-Bench’s Root-Cause Mapping (CTI-RCM) task, invalid predictions, such as unparseable responses, are excluded from the denominator. Gemma-4 produces only two parseable outputs out of 1,000, which are both correct. Correct-over-valid scoring therefore reports 100.0%, whereas correct-over-total scoring reports 0.2%, a 99.8 pp difference and 500× inflation (Example 10; App. G.3.2). AthenaBench’s vulnerability-scoring task exhibits the same mechanism: unparseable CVSS vectors are excluded from the released denominator, affecting nine of the 10 models. Re-scoring the same generations while retaining these predictions and assigning the maximum deviation lowers scores by up to 85.4 pp, with a median decrease of 55.9 pp across the affected models (App. G.3.2). Thus, invalid-response handling should be defined and reported explicitly.

$\mathcal { F } _ { 3 } ( \mathcal { E } )$ : Metric-direction mismatch. CTI-Bench and AthenaBench both evaluate CVSS-vector prediction but use different scoring conventions. CTI-Bench reports mean absolute deviation (MAD), where lower is better, while AthenaBench transforms MAD into a normalized percentage, where higher is better (App. G.3.3). Across the 10 models, this mismatch shifts rankings by up to five positions, so similar tasks are not directly comparable unless their metric direction and scale are aligned.

$\mathcal { F } _ { 4 } ( \mathcal { E } )$ : Prompt-mode sensitivity. In AthenaBench’s Attack-Technique Extraction (Athena-ATE) task, the extractor reads only the final line of the response. A Chain-of-Thought (CoT) model may identify the correct techniques in its reasoning but omit them from the final line, causing correct evidence to be ignored. Holding the task and extractor fixed, we compare zero-shot, few-shot, and CoT prompting on the same samples. The resulting Athena-ATE scores differ by up to 40 pp across the 10 models (App. G.3.4). As each prompt mode generates new responses, we treat this as prompt-mode sensitivity rather than an extraction failure.

## 4.5 Aggregation Failures

Aggregation determines how question- and tasklevel scores become benchmark-level conclusions.

F<sub>1</sub>(A): Logprob vs. generative scoring. MCQ scores can change substantially depending on how answers are scored. Log-probability scoring ranks the answer choices directly without generating a response, while generative scoring extracts an answer from generated text. On RedSage-Bench, Gemma-4 scores 45.7% under generative regex extraction but 86.6% under log-probability scoring. Qwen3.6 shows the opposite pattern, scoring 85.9% and 59.2%, respectively (App. G.4.1). These differences are not caused by the RedSage-Bench stop sequence, since the generative scores use stop-free outputs. Neither scoring method is uniformly preferable because they measure different interactions between the model and evaluator.

F<sub>2</sub>(A): Task-level metric drift. In attackerattribution, scores vary substantially with the scoring rules. CTI-Bench provides strict exact-match scoring and a lenient variant that additionally credits alias-connected or related threat actors (e.g.,

APT28 and FancyBear). AthenaBench instead uses a strict binary verdict. Across the 10 models, differences among these task-level scores reach 70 pp (App. G.4.2). Aggregating task scores defined under different scoring rules can make benchmarklevel scores and rankings depend on those rules rather than on a consistent measure of capability.

$\mathcal { F } _ { 3 } ( \mathcal { A } )$ : Aggregation inconsistency. Aggregation becomes inconsistent when scores computed over different populations are combined or compared as if they measured the same quantity. For example, accuracy may use all questions as the denominator in one task but only valid predictions in another. Both are percentages, so this difference can disappear in benchmark-level comparisons.

Gemma-4 illustrates the effect. CTI-RCM reports 100.0%, but only two of the model’s 1,000 predictions are considered valid and both are correct (App. G.3.2). SecEval reports 9.5%, producing an apparent 90.5 pp gap. Its score is also affected by prompt wording that leads Gemma-4 to return single-select answers on multi-select questions, with a 33.0% single-select rate on this subset (App. G.1.5). Using a correct-over-total denominator reduces CTI-RCM to 0.2%. Under the standardized pipeline, the two scores become 70.9% and 78.3%, respectively (App. G.4.3). Aggregation can therefore propagate upstream pipeline inconsistencies into benchmark-level comparisons.

Implications for Model Comparison. Pipeline failures are widespread: every audited benchmark has at least two documented failure-mode interactions (Table 4). Large score changes can occur without changing the dataset or model, driven instead by configuration choices spanning all pipeline stages. But does this pipeline sensitivity also change comparative conclusions about models?

## 5 Cross-Benchmark Instability

The benchmark-level analysis shows that scores can change under different pipeline choices. We next investigate how this instability affects model comparison. We analyze the 23×10 task-by-model score matrix from three perspectives: score-level redundancy, rank-level agreement, and ranking shifts under pipeline standardization.

## 5.1 Score-Level Redundancy

We first ask whether the 23 tasks provide independent evidence about model capability. As shown in Figure 2, PCA of the column-standardized 23×10 accuracy matrix shows that the first component explains 95.25% of the variance (App. H.1). Thus, most tasks separate stronger from weaker models along one dominant performance axis rather than capturing distinct cybersecurity capabilities. This redundancy limits what broader task coverage can establish. Adding tasks may increase benchmark scale without adding much measurement diversity. Aggregate scores may therefore provide limited evidence for claims about fine-grained cybersecurity specialization. A Mantel test finds a moderate association between task-content similarity and taskinduced rank agreement (r = 0.516; App. H.4), suggesting that semantic overlap explains only part of the observed ranking similarity. This is consistent with the dominant performance axis reflecting more than repeated or similar question content.

<table><tr><td></td><td colspan="2">Dataset</td><td colspan="2">Prompt</td><td colspan="2"></td><td colspan="2">Inference</td><td colspan="4">Extraction</td><td colspan="4">Aggregation</td><td></td></tr><tr><td>Benchmark</td><td>F1(D) F2(D)</td><td></td><td></td><td></td><td>F1(P) F2(P) F3(P)</td><td>F1(T)</td><td>F2(T)</td><td>F3(T)</td><td>F1(ε)</td><td></td><td>F2(ε)</td><td>F3(ε)</td><td>F4(E) F1(A) F2(A) F3(A)</td><td></td><td></td><td></td><td>Total</td></tr><tr><td>MMLU-CS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>3</td></tr><tr><td>SecEval</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>4</td></tr><tr><td>SECURE</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>3</td></tr><tr><td>CTI-Bench</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>8</td></tr><tr><td>AthenaBench</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>6</td></tr><tr><td>CyberMetric</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>3</td></tr><tr><td>RedSage-Bench</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>5</td></tr><tr><td>SecBench</td><td></td><td>✓</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>2</td></tr><tr><td>Total</td><td>5</td><td>7</td><td>1</td><td>1</td><td>2</td><td></td><td></td><td></td><td></td><td></td><td></td><td>2</td><td>1</td><td>2</td><td>2</td><td>2</td><td>34</td></tr></table>

Table 4: Failure-mode incidence across the eight audited benchmarks. A red cell with a check indicates an observed incident; a green cell with a dot indicates no incident was observed. Totals report observed incidents by benchmark and failure mode. All 120 benchmark–failure-mode pairs were examined, resulting in a total of 34 incidents.

## 5.2 Rank-Level Disagreement

Score-level redundancy does not imply stable model rankings. We measure rank agreement using Kendall’s τ-b, which accounts for ties, together with pairwise rank inversions (Apps. H.2–H.3). Tasks may broadly agree on stronger and weaker models while disagreeing on specific model orderings. This pattern is clearest for semantically similar tasks. CTI-Bench and AthenaBench both evaluate vulnerability scoring and attacker attribution, yet they rank the same 10 models differently. The rank agreement is weak for vulnerability scoring $\left( \tau - \mathbf { b } { = } 0 . 2 9 \right)$ and attacker attribution (τ-b=0.24). The vulnerability-scoring pair also reorders models by up to five positions (App. H.3). These differences coincide with incompatible extraction rules, alias handling, and scoring directions/rules. So, comparative claims, such as $^ { \ast } m _ { i }$ outperforms $m _ { j }$ at vulnerability scoring,” can depend on the pipeline.

![](images/1960806b33ddd4579a5b9493fb16bc8eb0943c72c0f335c47efa4bf5b1f0ac28.jpg)  
Figure 2: PCA scree plot of the task-by-model accuracy matrix. Only PC1 exceeds Horn’s parallel-analysis nul and explains 95.25% of the variance.

## 5.3 Ranking Shifts Under Standardization

Using the evaluation harness (§3.3), we standardize the nine pipeline configuration fields wherever benchmark semantics allow. App. I.1 specifies which fields are standardized for each benchmark, and App. I.2 reports the underlying scores.

For benchmark b and model m, let $r _ { b } ^ { \mathrm { o r i g } } ( m )$ and $r _ { b } ^ { \mathrm { s t d } } ( m )$ denote the model’s rank under the original and standardized pipelines, respectively, with rank 1 indicating the highest-scoring model. Ties are broken deterministically using a stable sort (App. I.3). We define the rank shift as

$$
\Delta r _ { b } ( m ) = r _ { b } ^ { \mathrm { o r i g } } ( m ) - r _ { b } ^ { \mathrm { s t d } } ( m ) ,\tag{2}
$$

where $\Delta r _ { b } ( m ) > 0$ indicates upward shift after standardization and $\Delta r _ { b } ( m ) < 0$ indicates downward shift. Table 5 shows that nine of the 10 models shift by at least three ranks on at least one benchmark; only GPT-5.4 does not. Gemma-4 rises on every benchmark where its rank changes, consistent with the extraction and denominator effects in §4. Because standardization changes upstream fields like the prompt formatting, decoding, and token budget, these scores require re-generation rather than only re-scoring stored outputs.

![](images/6869f929ff4eac431b3f3b5178b365e97e5ccc1e15d5f1b845e4f653bbb46cf9.jpg)  
Figure 3: Spearman $\rho$ between original and standardized model rankings for each benchmark. Whiskers show question-level bootstrap 95% confidence intervals (App. I.4). Lower $\rho$ indicates greater reordering.

As shown in Figure 3, the original and standardized rankings also show substantial disagreement. Spearman’s $\rho$ is -0.47 for MMLU-CS and 0.03 for SecEval, indicating strong reordering, and ranges from 0.42 to 0.73 on the other benchmarks. On MMLU-CS, all four cybersecurity-specialized models shift downward by three to seven positions, while Gemma-4 shifts upward by six.

These shifts are robust to question-sampling variability. Using 5,000 paired bootstrap samples, every rank shift of at least three positions keeps the same direction in at least 97.7% of samples. The 95% confidence intervals for Spearman’s $\rho$ all exclude 1, indicating that the original and standardized rankings are not statistically consistent with being identical. Thus, the ranking changes cannot be explained by question-sampling noise alone.

Implications for Benchmark Design. Broader task coverage does not guarantee more informative or stable model comparisons. Many tasks provide redundant score-level evidence, while model rankings remain sensitive to pipeline choices. Reliable benchmarking therefore requires both diverse capability coverage and explicit, consistent evaluation pipelines.

## 6 Toward Reliable Benchmarking

Benchmark reliability deserves the same care as task design. A benchmark release should specify the full measurement pipeline used to produce its scores, not only the dataset and metric. This includes dataset and label provenance, the exact prompt and chat template, inference settings, the executable extractor or judge, denominator policy, scoring and aggregation rules, and reliability statistics such as invalid-response rates and rank stability. App. J.2 provides a complete field list.

Evaluation harnesses should also minimize avoidable nondeterminism. Open-weight models should use pinned model and serving configurations. Closed-source evaluations should report the exact model ID and run timestamp. Judge-based evaluation should fix the judge model, prompt, and decoding parameters. Storing raw outputs further improves reproducibility: changes to extraction, scoring, denominators, or aggregation can then be re-scored without querying the evaluated model again. An analysis that requires new generations should report the model and serving configuration used for those runs.

Benchmarks that evaluate similar tasks should align metric direction, scoring rules, alias handling, and aggregation conventions. If not, their scores should be reported as distinct measurements rather than treated as directly comparable. In evolving domains such as cybersecurity, gold labels should also be periodically audited against authoritative sources, with uncertainty reported explicitly.

Not every reliability failure can be fixed automatically. Some arise from underspecified benchmark intent, such as whether partial credit should be awarded or which semantic equivalences (e.g., aliases) should count as correct. These choices cannot be recovered reliably from the released artifact alone. Evaluation harnesses should therefore make such decisions explicit and encode them in executable form. App. J.1 classifies the identified failure modes as automatically detectable, automatically fixable, or requiring manual judgment.

## 7 Related Work

Large-scale frameworks such as BIG-bench (Srivastava et al., 2022), HELM (Liang et al., 2023), and DynaBench (Kiela et al., 2021) have made benchmark-based comparison central to LLM evaluation. Recent work shows that reported performance can be sensitive to prompt wording, decoding parameters, evaluator design, and extraction rules (Shi et al., 2024; Sun et al., 2024; Wang et al., 2023). We build on this work by modeling benchmarks as end-to-end measurement pipelines and auditing failures across all five stages. Prior work also shows that aggregate score correlations can obscure disagreement in model rankings (Perlitz et al., 2024). We extend this perspective to cybersecurity, where even semantically similar tasks can rank the same models differently when their evaluation conventions differ.

<table><tr><td>Model</td><td>MMLU-CS</td><td>SecEval</td><td>SECURE</td><td>CTI-Bench</td><td>AthenaBench</td><td>CyberMetric</td><td>RedSage-Bench</td><td>SecBench</td></tr><tr><td>GPT-5.4</td><td>+1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>-1</td></tr><tr><td>Sonnet 4.6</td><td>+5</td><td>-5</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Gemma-4</td><td>+6</td><td>+7</td><td>+4</td><td>+3</td><td>+7</td><td>+5</td><td>+5</td><td>+6</td></tr><tr><td>Qwen3.6</td><td>+6</td><td>+3</td><td>-1</td><td>-2</td><td>-3</td><td>0</td><td>+1</td><td>+4</td></tr><tr><td>Llama-3.3</td><td>-3</td><td>-5</td><td>-3</td><td>-2</td><td>-2</td><td>-2</td><td>0</td><td>-3</td></tr><tr><td>GPT-OSS</td><td>+5</td><td>+4</td><td>0</td><td>-5</td><td>-5</td><td>-1</td><td>+2</td><td>+1</td></tr><tr><td>Primus-Nemotron</td><td>-7</td><td>-1</td><td>+4</td><td>+3</td><td>+2</td><td>+2</td><td>-2</td><td>-1</td></tr><tr><td>Primus-Merged</td><td>-3</td><td>-1</td><td>0</td><td>0</td><td>0</td><td>-1</td><td>-3</td><td>-1</td></tr><tr><td>Foundation-Sec</td><td>-6</td><td>-5</td><td>-4</td><td>0</td><td>+2</td><td>0</td><td>-1</td><td>-4</td></tr><tr><td>RedSage-Qwen3</td><td>-4</td><td>+3</td><td>0</td><td>+3</td><td>-1</td><td>-3</td><td>-2</td><td>-1</td></tr><tr><td>Spearman ρ</td><td>-0.47</td><td>0.03</td><td>0.65</td><td>0.64</td><td>0.42</td><td>0.73</td><td>0.71</td><td>0.50</td></tr></table>

Table 5: Rank shifts under pipeline standardization. Each cell reports $\Delta r _ { b } ( m )$ , where positive values indicate upward movement and negative values indicate downward movement. The bottom row reports Spearman’s ρ between the original and standardized rankings; 95% bootstrap confidence intervals are shown in Figure 3.

Bean et al. (Bean et al., 2025) study the complementary problem of construct validity: whether benchmarks measure the phenomena they claim to measure and support the resulting claims. Their review spans multiple benchmark domains but does not include cybersecurity benchmarks. We instead focus on measurement reliability of executable evaluation pipelines and provide a systematic audit of this problem in cybersecurity.

## 8 Conclusion

We audited cybersecurity LLM benchmarks as endto-end measurement pipelines. Across eight benchmarks, 23 tasks, and 10 LLMs, we identified 15 recurring failure modes spanning all five pipeline stages. Individual pipeline choices can shift scores by more than 80 pp, and nine of the 10 models shift by at least three ranks under standardization. Also, broader task coverage does not guarantee more informative or stable comparisons: many tasks provide redundant evidence, while similar tasks can rank models differently. Thus, benchmark scores should be treated as pipeline-dependent measurements, supported by meaningful capability coverage and explicit, consistent evaluation pipelines.

## 9 Software

The harness is released on PyPI as an open-source Python package for model-agnostic evaluation of cybersecurity $\mathrm { L L M s . ^ { 4 } }$ It provides a common interface for hosted APIs and locally served models. Open-ended responses can be scored using a configurable LLM judge, allowing the evaluated model and judge model to be selected independently. For transparency, an adapter exports aggregate scores and pipeline metadata to EveryEvalEver (Batzner et al., 2026), while withholding question-level content in accordance with our Ethical Considerations.

## Limitations

Our empirical findings are specific to cybersecurity. We chose this domain because it makes pipeline failures unusually observable. Much of its ground truth can be checked against authoritative sources such as CVE, CWE, CVSS, and MITRE ATT&CK (App. G.1.2). Many answers are also structured, so certain failures such as an invalid CVSS vector can be detected mechanically. Finally, the benchmark ecosystem is heterogeneous: 44 of 72 pipeline configuration fields cannot be reproduced from documentation alone (App. C). Mechanical failures involving token budgets, stop sequences, extraction, denominators, and aggregation may transfer to other domains, but we do not test this directly.

Our audit covers eight benchmarks, 23 tasks, and 10 LLMs. Other benchmarks, models, serving backends, and evaluators may exhibit failures outside our taxonomy. Some benchmark artifacts are also underspecified, requiring us to make implementation choices. We document these choices in the paper and appendix, but alternative interpretations may be defensible.

We study measurement reliability, not construct validity (Bean et al., 2025). A benchmark can be reproducible and internally consistent while still failing to measure the real-world capability it claims to represent. Reliability is therefore necessary for valid evaluation, but it is not sufficient.

Finally, some conclusions depend on contestable standardization choices. Table 12 identifies four such choices: answer extraction, invalid-response denominators, partial credit, and log-probability versus generative scoring. These choices are defensible but not unique, so comparative conclusions remain conditional on them. Our standardized extraction also uses a pinned LLM judge. An independent judge agrees on 99.6% of verdicts and produces nearly identical rankings (App. E.3), reducing but not eliminating concerns about judge dependence. We quantify question-sampling variability with a question-level bootstrap (App. I.4), but do not estimate variability across repeated model generations. Gold-label and judge decision verification are also LLM-assisted, with manual validation on random stratified samples of questions drawn across each benchmark’s tasks (Apps. E.2 and G.1.2).

## Ethical Considerations

This work studies the reliability of cybersecurity benchmarks rather than the development of new offensive capabilities. The audited tasks cover vulnerabilities, attack techniques, malware behavior, and threat intelligence, but are drawn from existing public benchmarks and authoritative cybersecurity sources. We do not introduce new exploit procedures, malware implementations, offensive datasets, or attack automation.

The main ethical concern is the harm caused by unreliable evaluation. In a high-stakes domain such as cybersecurity, unstable benchmark pipelines can support misleading claims about model capability, safety, or specialization. Such claims may influence model selection, deployment decisions, and trust in automated security systems. By exposing pipeline failures and making evaluation choices explicit, our goal is to improve transparency, reproducibility, and scientific reliability.

A secondary concern is dual use. Detailed descriptions of benchmark failure modes could facilitate benchmark-specific optimization or gaming. We therefore focus on evaluation mechanisms rather than methods for exploiting security systems, and our public reporting emphasizes aggregate results and pipeline metadata rather than unnecessary question-level content.

## References

Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K

Arora, Yu Bai, Bowen Baker, Haiming Bao, and 1 others. 2025. GPT-OSS-120B & GPT-OSS-20B Model Card. arXiv preprint arXiv:2508.10925.

Md Tanvirul Alam, Dipkamal Bhusal, Salman Ahmad, Nidhi Rastogi, and Peter Worth. 2025. AthenaBench: A Dynamic Benchmark for Evaluating LLMs in Cyber Threat Intelligence. arXiv preprint arXiv:2511.01144.

Md Tanvirul Alam, Dipkamal Bhusal, Le Nguyen, and Nidhi Rastogi. 2024. CTIBench: A Benchmark for Evaluating LLMs in Cyber Threat Intelligence. Advances in Neural Information Processing Systems, 37:50805–50825.

Anthropic. 2025. System Card: Claude Sonnet 4.6.

Jan Batzner, Sree Harsha Nelaturu, Damian Stachura, Anastassia Kornilova, Jon Crall, Tommaso Cerruti, Yanan Long, Yifan Mai, Sanchit Ahuja, Asaf Yehudai, and 1 others. 2026. Every eval ever: A unifying schema and community repository for ai evaluation results. arXiv preprint arXiv:2606.14516.

Andrew M Bean, Ryan Othniel Kearns, Angelika Romanou, Franziska Sofia Hafner, Harry Mayne, Jan Batzner, Negar Foroutan Eghlidi, Chris Schmitz, Karolina Korgul, Hunar Batra, and 1 others. 2025. Measuring what matters: Construct validity in large language model benchmarks. Advances in Neural Information Processing Systems 38, pages 19868– 19949.

Dipkamal Bhusal, Md Tanvirul Alam, Le Nguyen, Ashim Mahara, Zachary Lightcap, Rodney Frazier, Romy Fieblinger, Grace Long Torales, Benjamin A Blakely, and Nidhi Rastogi. 2024. Secure: Benchmarking large language models for cybersecurity. In 2024 Annual Computer Security Applications Conference (ACSAC), pages 15–30. IEEE.

Avijit Ghosh, Anka Reuel, Jenny Chim, Wm Matthew Kennedy, Srishti Yadav, Jennifer Mickel, Yanan Long, Andrew Tran, Anastassia Kornilova, Damian Stachura, and 1 others. 2026. Evaluation cards: An interpretive layer for ai evaluation reporting. arXiv preprint arXiv:2606.09809.

Google. 2026. Gemma 4 Model Card.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring Massive Multitask Language Understanding. arXiv preprint arXiv:2009.03300.

Pengfei Jing, Mengyun Tang, Xiaorong Shi, Xing Zheng, Sen Nie, Shi Wu, Yong Yang, and Xiapu Luo. 2024. SecBench: A comprehensive multidimensional benchmarking dataset for LLMs in cybersecurity. arXiv preprint arXiv:2412.20787.

Douwe Kiela, Max Bartolo, Yixin Nie, and 1 others. 2021. Dynabench: Rethinking Benchmarking in NLP. In Proceedings ofNAACL.

Guancheng Li, Yifeng Li, Wang Guannan, Haoyu Yang, and Yang Yu. 2023. SecEval: A Comprehensive Benchmark for Evaluating Cybersecurity Knowledge of Foundation Models. GitHub.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D Manning, Christopher Re, Diana Acosta-Navas, Drew A. Hudson, and 31 others. 2023. Holistic Evaluation of Language Models. Transactions on Machine Learning Research.

Yotam Perlitz, Ariel Gera, Ofir Arviv, Asaf Yehudai, Elron Bandel, Eyal Shnarch, Michal Shmueli-Scheuer, and Leshem Choshen. 2024. Benchmark Agreement Testing Done Right: A Guide for LLM Benchmark Evaluation. In NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling.

Cheng Shi, Zhisheng Zhang, Yujiu Yang, and 1 others. 2024. A Thorough Examination of Decoding Methods in the Era of Large Language Models. In Proceedings of EMNLP.

Leslie F. Sikos. 2023. Cybersecurity Knowledge Graphs. Knowledge and Information Systems, 65(9):3511–3531.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, and 1 others. 2025. OpenAI GPT-5 System Card. arXiv preprint arXiv:2601.03267.

Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, and 1 others. 2022. Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models. Transactions on Machine Learning Research.

Jiuding Sun, Chantal Shaib, and Byron Wallace. 2024. Evaluating the Zero-Shot Robustness of Instruction-Tuned Language Models. In International Conference on Learning Representations, volume 2024, pages 48103–48141.

Naufal Suryanto, Muzammal Naseer, Pengfei Li, Syed Talal Wasim, Jinhui Yi, Juergen Gall, Paolo Ceravolo, and Ernesto Damiani. 2026. RedSage: A Cybersecurity Generalist LLM. In The Fourteenth International Conference on Learning Representations.

Norbert Tihanyi, Mohamed Amine Ferrag, Ridhi Jain, Tamas Bisztray, and Merouane Debbah. 2024. CyberMetric: A Benchmark Dataset Based on Retrieval-Augmented Generation for Evaluating LLMs in Cybersecurity Knowledge. In 2024 IEEE International

Conference on Cyber Security and Resilience (CSR), pages 296–302. IEEE.

Peiyi Wang, Lei Li, Liang Chen, and 1 others. 2023. Large Language Models are not Fair Evaluators. arXiv preprint arXiv:2305.17926.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Zhuoran Yang, Ed Li, Jianliang He, Aman Priyanshu, Baturay Saglam, Paul Kassianik, Sajana Weerawardhena, Anu Vellore, Blaine Nelson, Neusha Javidnia, and 1 others. 2026. Llama-3.1- FoundationAI-SecurityLLM-Reasoning-8B Technical Report. arXiv preprint arXiv:2601.21051.

Yao-Ching Yu, Tsun-Han Chiang, Cheng-Wei Tsai, Chien-Ming Huang, and Wen-Kwang Tsao. 2025. Primus: A Pioneering Collection of Open-Source Datasets for Cybersecurity LLM Training. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 10402– 10424.

## A Where to Find Each Artifact

Table 6 provides a roadmap to the artifacts supporting our pipeline audit and standardization decisions. Materials that are impractical to include in the paper, such as raw generations, run logs, plotting code, and interactive notebooks, are released in the audit repository, which is also linked in the table.

Four box types recur throughout the appendix. A prompt box reproduces model or judge instructions. An example box shows a concrete failure-mode instance, including the prompt, raw output, evaluator behavior, and consequence. A case box presents a verified gold-label case grounded in authoritative sources. A code box shows an executable harness artifact. Prompt, example, case, and code boxes are numbered by type.

## B Evaluation Setup

This section provides the implementation details needed to reproduce our evaluation. We first define the scored benchmark subsets and task inventory. We then describe the serving environment, prompt construction, and the configuration records produced by the evaluation harness. Benchmarkspecific pipeline choices and their standardization are documented separately in App. D.

<table><tr><td>Artifact</td><td>Location</td></tr><tr><td>Pipeline specification</td><td></td></tr><tr><td>Original and standardized pipelines, by benchmark</td><td>App. D.4; Table 11</td></tr><tr><td>Specification status of all 72 pipeline fields Standardizations with defensible alternatives</td><td>App. C; Table 10 App. D.2, Table 12</td></tr><tr><td>Field-level pipeline ledger and measured effects</td><td>App. D.4, Table 13</td></tr><tr><td>Harness configuration</td><td>App. D</td></tr><tr><td>Audit protocol</td><td></td></tr><tr><td>Audit procedure</td><td>§2.1; App. F.1</td></tr><tr><td>Stage-by-benchmark audit coverage</td><td>App. F.5; Table 18</td></tr><tr><td>Re-scoring, re-generation, and automation limits</td><td>App. F.2; Table 16</td></tr><tr><td>Evidence and results</td><td></td></tr><tr><td>Evidence for each failure mode</td><td>App. G</td></tr><tr><td>Original and standardized scores</td><td>App. I.2; Table 28</td></tr><tr><td>Gold-label audit and verified cases</td><td></td></tr><tr><td>Benchmark sizes and task inventory</td><td>Apps. G.1.2–G.1.2 Tables 7, 8</td></tr><tr><td></td><td></td></tr><tr><td>Repository-only artifacts</td><td></td></tr><tr><td>Raw generations, run logs, plots code, and notebooks</td><td>Audit repository</td></tr></table>

Table 6: Roadmap to the artifacts supporting the benchmark audit. The appendix contains the information needed to assess the reported pipeline decisions; larger execution artifacts are provided in the audit repository.

## B.1 Evaluation Scale and Sampling

Table 7 distinguishes three dataset sizes for each benchmark. Reported is the number of questions stated by the benchmark paper or release documentation for the corresponding evaluation scope. Released is the number of questions available in the public artifact. Scored is the subset evaluated in our audit. The scored subsets sum to 48,662 questions across 23 tasks, which is the evaluation scope reported in the main presentation. Per-task counts are given in Table 8.

Only three benchmarks have matching reported and released sizes. Several of the remaining differences arise from counting or release conventions rather than from our sampling. SecEval reports an overall total of 2,126 in its README, although its per-topic counts match the 2,189 distinct questions in the released file. CyberMetric reports 10,000 questions, while its release contains 10,180. CTI-Bench does not report a benchmarkwide total. Summing the paper’s per-task figures gives 4,947, partly because CTI-ATE is described using 397 unique ATT&CK techniques rather than its 60 evaluation questions. For RedSage-Bench, the paper reports 30,240 questions, of which 240 are open ended Q&A. The released artifact also contains a separate 50-question validation split.

SecBench has the largest substantive difference between the reported and released data. The paper reports 47,910 questions, but only 3,000 are publicly released. These comprise 2,730 MCQs and 270 SAQs. Only 661 of the released MCQs are in English, and none of the SAQs are. We therefore evaluate the 661 English MCQs.

<table><tr><td>Benchmark</td><td>Reported</td><td>Released</td><td>Scored</td></tr><tr><td>MMLU-CS</td><td>100</td><td>100</td><td>100</td></tr><tr><td>SecEval</td><td>2,126</td><td>2,189</td><td>2,189</td></tr><tr><td>SECURE</td><td>4,068</td><td>4,068</td><td>2,502</td></tr><tr><td>CTI-Bench</td><td>4,947</td><td>5,610</td><td>4,610</td></tr><tr><td>AthenaBench</td><td>8,100</td><td>8,100</td><td>8,100</td></tr><tr><td>CyberMetric</td><td>10,000</td><td>10,180</td><td>500</td></tr><tr><td>RedSage-Bench</td><td>30,240</td><td>30,290</td><td>30,000</td></tr><tr><td>SecBench</td><td>47,910</td><td>3,000</td><td>661</td></tr><tr><td>Total scored</td><td></td><td></td><td>48,662</td></tr></table>

Table 7: Benchmark sizes used to define the audit scope. Per-task scored sizes are given in Table 8.

The scored set is also smaller than the release when a benchmark contains tasks or splits outside our evaluation scope. For SECURE, we evaluate three of its six tasks: MAET, CWET, and KCV. For CTI-Bench, we evaluate five tasks and exclude the separate 1,000 question CTI-RCM-2021 split. For RedSage-Bench, we use the 30,000 closed-form question test split and exclude the 240 open-ended questions and the 50-question validation split. For CyberMetric, we use the separately released 500- question set that its authors describe as humanvalidated. Finally, SECURE-CWET contains one all-empty row with no prompt; the harness drops this row before evaluation.

## B.2 Task Inventory

The audit covers 23 tasks across the eight benchmarks. Table 8 gives the task identifier (ID) used in the harness, task-level metric, denominator policy, and number of scored questions for each task. These counts sum to the 48,662 questions, as reported in Table 7.

Several task abbreviations are used throughout the appendix. Root-Cause Mapping (RCM) maps a CVE description to a CWE ID. Vulnerability Severity Prediction (VSP) predicts a CVSS vector. Attack-Technique Extraction (ATE) extracts MITRE ATT&CK technique IDs, while Threat-Actor Attribution (TAA) identifies the actor associated with a threat-intelligence description. Response and Mitigation Selection (RMS) maps scenarios to ATT&CK mitigation IDs. SECURE’s MAET and CWET tasks are multiple-choice tasks based on MITRE ATT&CK and CWE, respectively, while KCV is a true-or-false task over CVE records.

<table><tr><td>Benchmark</td><td>Task ID</td><td>Metric</td><td>Denominator policy</td><td>Questions</td></tr><tr><td>MMLU-CS</td><td>mmlu_cs</td><td>Accuracy</td><td>Correct / total</td><td>100</td></tr><tr><td>SecEval</td><td>seceval</td><td>Set-exact-match accuracy</td><td>Correct / total</td><td>2,189</td></tr><tr><td>SECURE</td><td>secure_maet</td><td>Accuracy</td><td>Correct / total</td><td>1,072</td></tr><tr><td></td><td>secure_cwet</td><td>Accuracy</td><td>Correct / total</td><td>964</td></tr><tr><td></td><td>secure_kcv</td><td>Accuracy</td><td>Correct / total</td><td>466</td></tr><tr><td>CTI-Bench</td><td>cti_mcq</td><td>Accuracy</td><td>Correct / total</td><td>2,500</td></tr><tr><td></td><td>cti_rcm</td><td>CWE accuracy</td><td>Correct / total</td><td>1,000</td></tr><tr><td></td><td>cti_vsp</td><td>CVSS MAD</td><td>All questions</td><td>1,000</td></tr><tr><td></td><td>cti_ate</td><td>Accuracy</td><td>Correct / total</td><td>60</td></tr><tr><td></td><td>cti_taa</td><td>Binary / partial</td><td>All questions</td><td>50</td></tr><tr><td>AthenaBench</td><td>ckt</td><td>Accuracy</td><td>Correct / total</td><td>3,000</td></tr><tr><td></td><td>rms</td><td>Question-level set-F1</td><td>All questions</td><td>500</td></tr><tr><td></td><td>athena_taa</td><td>Binary / partial</td><td>All questions</td><td>100</td></tr><tr><td></td><td>athena_ate</td><td>Accuracy</td><td>Correct / total</td><td>500</td></tr><tr><td></td><td>athena_rcm</td><td>CWE accuracy</td><td>Correct / total</td><td>2,000</td></tr><tr><td></td><td>athena_vsp</td><td>Normalized CVSS score</td><td>All questions</td><td>2,000</td></tr><tr><td>CyberMetric</td><td>cybermetric</td><td>Accuracy</td><td>Correct / total</td><td>500</td></tr><tr><td>RedSage-Bench</td><td>frameworks</td><td>Accuracy</td><td>Correct / total</td><td>5,000</td></tr><tr><td></td><td>generals</td><td>Accuracy</td><td>Correct / total</td><td>5,000</td></tr><tr><td></td><td>skills</td><td>Accuracy</td><td>Correct / total</td><td>10,000</td></tr><tr><td></td><td>cli</td><td>Accuracy</td><td>Correct / total</td><td>5,000</td></tr><tr><td></td><td>kali</td><td>Accuracy</td><td>Correct / total</td><td>5,000</td></tr><tr><td>SecBench</td><td>secbench</td><td>Set-exact-match accuracy</td><td>Correct / total</td><td>661</td></tr><tr><td>Total</td><td></td><td></td><td></td><td>48,662</td></tr></table>

Table 8: Inventory of the 23 scored tasks. Metric names describe the task-level quantities recorded by the harness. The standardized denominator retains attempted questions rather than dropping unparseable model outputs. For TAA, both binary and partial-credit variants are retained for the audit, while standardized model comparisons use strict binary scoring (App. G.4.2). Benchmark-level summaries include only comparable, bounded, higher-is-better metrics; raw MAD, set-F1, and partial-credit attribution are excluded where appropriate (App. D.4).

AthenaBench’s CKT is a five-option cybersecurity knowledge test.

The denominator column in Table 8 describes the standardized scoring rule. Empty or unparseable model answers remain in the evaluation population rather than being discarded. For accuracy tasks, these answers therefore count as incorrect. Taskspecific handling for structured metrics such as VSP is described in App. G.3.2. For attacker attribution, the harness retains the alternative scoring variants needed for the audit, while standardized model comparisons use the strict binary convention described in App. G.4.2.

## B.3 LLM Serving Stack

We serve all eight open-weight models locally with vLLM in bfloat16 and without quantization. Each evaluation job uses four NVIDIA H200 141 GB GPUs, with tensor parallelism set to the number of visible devices. We set gpu\_memory\_utilization to 0.90 by default and to 0.85 for Qwen3.6 and RedSage-Qwen3, which require additional memory headroom. Table 9 reports the maximum context length used for each checkpoint.

The standardized inference configuration uses greedy decoding with temperature 0 and top-p 1.0. Local vLLM runs additionally use min\_tokens=50 to avoid empty end-of-sequence completions. No backend stop sequence is used in the standardized configuration. Maximum output length is calibrated by task, with 1,024 tokens as the default. These settings describe the standardized configuration. When reproducing an original benchmark pipeline or conducting a controlled failure-mode analysis, we instead use the benchmark-specific setting being studied, as documented in App. D.4.

Two analyses use sampled decoding. For SE-CURE, we additionally evaluate the documented temperature of 0.7 and pin seed 42. For the Cyber-Metric decoding-drift analysis, F<sub>3</sub>(I), we compare the greedy control with the documented temperature of 1.0 and top-p of 0.9 (App. G.2.3). These sampled runs pin a seed; the main greedy runs do not depend on one.

GPT-5.4 and Sonnet 4.6 are evaluated through Azure-hosted endpoints (i.e., REST APIs). GPT-5.4 uses Azure OpenAI chat completion API with api-version=2024-12-01-preview. Sonnet 4.6 uses an Azure Anthropic-messages passthrough with anthropic-version=2023-06-01. Both are evaluated at temperature 0. Provider-side content filtering, such as guardrails, is disabled through the deployment configuration for both models. This avoids failed requests caused solely by the cybersecurity content of benchmark questions and makes pipeline behavior more reproducible.

<table><tr><td>Checkpoint</td><td>Context (token)</td><td>Memory (%)</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>32,768</td><td>85</td></tr><tr><td>RedSage-Qwen3-8B-DPO</td><td>16,384</td><td>85</td></tr><tr><td>Llama-3.3-70B-Instruct</td><td>8,192</td><td>90</td></tr><tr><td>Llama-Primus-Nemotron-70B</td><td>8,192</td><td>90</td></tr><tr><td>Foundation-Sec-8B-Instruct</td><td>8,192</td><td>90</td></tr><tr><td>gpt-oss-20b</td><td>8,192</td><td>90</td></tr><tr><td>gemma-4-31B-it</td><td>4,096</td><td>90</td></tr><tr><td>Llama-Primus-Merged</td><td>Default</td><td>90</td></tr></table>

Table 9: vLLM serving configuration for open-weight models. Context is the configured max\_model\_len. Memory is gpu\_memory\_utilization.

The local runtime uses CUDA 12.1, Python 3.10, PyTorch, transformers, and vLLM. Open-weight checkpoints were retrieved at their then-current Hugging Face revisions, or from a fixed local checkpoint where applicable. We record the resolved checkpoint commit hashes with the released artifacts so that the evaluated weights can be recovered. Local inference was run from April 19–20, 2026, with two additional task runs from May 4–5, 2026. Hosted inference on Azure was run on May 5, and judge calls were performed between April 21 and May 5, 2026.

## B.4 Prompt Templates

Prompt construction depends on both the benchmark and the task. The standardized harness preserves the benchmark question and intended answer semantics while making prompt and chat formatting explicit. Single-select multiple-choice tasks use the template shown in Prompt 1. Multi-select and structured-output tasks use task-specific output instructions instead. Benchmark-specific system prompts are retained where they define the task presentation. Representative system prompts for CTI-Bench and SecEval are shown in Prompt 2 and Prompt 3, respectively. MMLU-CS and SecEval also prepend fixed few-shot exemplar blocks when required by the corresponding pipeline. The complete benchmark-by-benchmark prompt provenance is given in App. D.4.

PROMPT 1 : Shared single-select MCQ user template   
You are given multiple choice questions. Answer with the   
option letter (A, B, C, D) from the given choices   
directly.   
Question: {question}   
{choices}   
Answer:   
PROMPT 2 : CTI-Bench system prompt   
You are a cybersecurity expert specializing in   
cyberthreat intelligence.   
PROMPT 3 : SecEval system prompt   
Below are multiple-choice questions concerning   
cybersecurity. Please select the correct answers   
and respond with the letters ABCD only.

## B.5 Reproducible Configuration

Each run stores the configuration needed to interpret its reported score. This includes the relevant pipeline configuration fields defined in Apps. D, together with model, judge, backend, task, and schema provenance. In particular, the record preserves the decoding configuration, token budget, stop-sequence handling, extraction and scoring rules, denominator policy, and prompt configuration associated with the run. This allows each reported score to be traced back to the pipeline that produced it. A representative configuration and policy record is shown in Code 1. The complete benchmark-specific values and their provenance are given in App. D.4.

```ini
CODE 1 : Harness config and per-run policy record
# decoding / token budget
temperature = 0.0
top_p = 1.0
max_tokens = per-task-calibrated
min_tokens = 50 # local vLLM only
answer_stop = None
# scoring policies
denominator_policy =
"accuracy = correct / total over all attempted
questions;
unparseable/empty model answers count as incorrect;
judge-API failures are excluded from both numerator
and denominator."
think_handling =
"strip <think>...</think> before judging;
if an answer-level stop is configured, apply it
after removing reasoning."
scoring =
"llm-as-judge: one call performs answer extraction
and returns a CORRECT/INCORRECT verdict."
# provenance stored with each result:
# model {name, provider, endpoint}
# judge {name, provider}
# tasks []
# backend
# run_timestamp
# schema_version
```

CODE 2 : Installing and running the harness   
pip install sayf-eval   
# inference + judge   
sayf-eval run \   
--tasks mcq seceval vsp taa \   
--model openai/gpt-4o \   
--judge anthropic/claude-sonnet-4-6 \   
--output-dir outputs/run1   
# local vLLM endpoint as the model under test   
vllm serve Qwen/Qwen3-8B --port 8000 --enforce-eager   
sayf-eval run --tasks mcq \   
--model hosted\_vllm/Qwen/Qwen3-8B \   
--base-url http://localhost:8000/v1 \   
--api-key EMPTY \   
--output-dir outputs/qwen

## B.6 Software

The released harness, Sayf-Eval,<sup>5</sup> is distributed via PyPI and can run inference and judging through a common command-line interface. Hosted models and locally served OpenAI-compatible endpoints use the same evaluation workflow. Code 2 shows the basic workflow. These commands illustrate package usage and are not intended to reproduce the exact experimental configuration used in this paper. Exact run configurations and provenance are provided with the released audit repository.<sup>6</sup>

## C Pipeline Specification Gaps

Across eight benchmarks and nine pipeline configuration fields, we inspect 72 field–benchmark pairs. We classify each pair by comparing the benchmark documentation with its released implementation. A field is undefined when the documentation does not specify it, contradicted when the documented and released layers do not define the same executable behavior, and matched when they agree. Of the 72 pairs, 36 are undefined, 8 are contradicted, and only 28 are specified and matched. Thus, 44 of 72 pipeline configuration fields (61%) cannot be reproduced from benchmark documentation alone. Table 10 gives the complete field-level breakdown.

This underspecification matters because an unspecified field must be supplied by the evaluator. Different choices can produce different prompts, generations, extracted answers, scores, or aggregations even when the benchmark questions and evaluated model are unchanged. The same ambiguity also affects our reconstruction: when neither the documentation nor released implementation uniquely determines a field, our selected value is an explicit evaluation choice rather than a uniquely correct setting. We record these choices in the pipeline ledger, defined in Table 13, instead of treating them as part of benchmark specification.

The eight contradicted fields illustrate the different ways in which the documented and released layers can diverge:

• CTI-Bench scoring rule: the documentation describes exact-match scoring, while the released TAA scorer also credits alias-connected and related threat actors.

• AthenaBench scoring rule: the VSP normalization constant R=7.7 appears in config.yaml but is absent in the paper, so the reported metric cannot be reconstructed from the paper alone.

• SECURE decoding: the paper specifies temperature T=0.7, but no inference implementation is released to enact that setting.

• CyberMetric prompt template: the prompt in the README differs from the prompt hardcoded in the released evaluator.

• CyberMetric decoding: the paper specifies T=1.0, top-p 0.9, and top-k 50, while the released evaluator sets none of these parameters and therefore inherits backend defaults.

• RedSage-Bench chat formatting: the two released inference examples use incompatible chat-formatting configurations.

• RedSage-Bench decoding: the two released inference examples specify different temperatures.

• RedSage-Bench maximum output tokens: the two released inference examples specify different token budgets.

The pipeline ledger also reports measured effects for fields that we perturb. These are isolated single-field counterfactuals: we vary one pipeline configuration field on the same questions while holding the remaining fields fixed. Downstream changes, such as extraction or denominator policy, can be evaluated by re-scoring stored outputs. Upstream changes, such as prompts or inference settings, require re-generation (App. F.2). These effects are therefore distinct from the original-tostandardized differences in Table 28, which reflect the combined change from the original pipeline to the standardized pipeline.

For each original pipeline, the ledger records the extractor used to reproduce its released behavior. When no extractor is released, we explicitly document the one we supply. The standardized pipeline instead uses the pinned LLM-based extraction and scoring rules described in App. E.1. Some failures cannot be isolated with a controlled counterfactual because the original pipeline cannot be executed without the failure or because the effect depends on an interaction between model output style and the evaluator. In these cases, we report a clearly labeled peer-gap estimate, defined as the median score of unaffected models minus the affected model’s score. A ledger entry marked not isolated indicates that the field was supplied or changed but was not independently ablated.

<table><tr><td></td><td colspan="2">Prompt</td><td colspan="3">Inference</td><td colspan="3">Extraction</td><td>Aggregation</td><td></td></tr><tr><td>Benchmark</td><td>Template</td><td>Chat</td><td>Decoding</td><td>Max tokens</td><td>Stop</td><td>Extractor</td><td>Scoring</td><td>Denominator</td><td>Rule</td><td>Gaps</td></tr><tr><td>MMLU-CS</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1</td></tr><tr><td>SecEval</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td><td>6</td></tr><tr><td>SECURE</td><td>1</td><td></td><td>X</td><td></td><td></td><td></td><td></td><td></td><td></td><td>6</td></tr><tr><td>CTI-Bench</td><td></td><td></td><td></td><td></td><td></td><td></td><td>X</td><td></td><td></td><td>5</td></tr><tr><td>AthenaBench</td><td>V</td><td></td><td>√</td><td></td><td></td><td></td><td>×</td><td></td><td></td><td>5</td></tr><tr><td>CyberMetric</td><td>×</td><td></td><td>X</td><td></td><td></td><td></td><td>√</td><td></td><td></td><td>5</td></tr><tr><td>RedSage-Bench</td><td></td><td>×</td><td>X</td><td>×</td><td></td><td></td><td></td><td></td><td></td><td>8</td></tr><tr><td>SecBench</td><td></td><td></td><td></td><td></td><td></td><td></td><td>J</td><td></td><td></td><td>8</td></tr></table>

Table 10: Specification status of the nine pipeline configuration fields across the eight benchmarks. A green check indicates that a field is specified in the documentation and matched by the released implementation. A gray dot indicates an undefined field, and a red cross indicates that the documented and released layers do not define the same executable behavior, so one has contradicted the other. Gaps counts undefined and contradicted fields. Overall, 44 of 72 fields (61%) are gaps; stop sequences are undefined for all eight benchmarks.

## D Benchmark Pipelines

This section documents how each benchmark pipeline is reconstructed and standardized. We distinguish three sources of information. The documented layer is the behavior stated in the benchmark paper or documentation. The released layer is the behavior implemented by the public artifact. When these layers are incomplete or inconsistent, we explicitly record the resolution used in our audit. We then identify the settings used by the standardized harness. This separation is important because a supplied or normalized setting is an evaluation choice, not necessarily a uniquely correct interpretation of the benchmark.

## D.1 Pipeline Summary

Table 11 summarizes the released artifacts and the resulting benchmark-level scores. The artifact columns indicate whether the corresponding component is released and usable as specified. These indicators describe artifact availability and consistency; they are not failure-mode observations. Binary failure-mode incidence is reported in Table 4.

The original and standardized columns report the mean and range across the 10 evaluated models. Benchmark-level means include only comparable, bounded, higher-is-better task scores. We exclude CTI-Bench VSP because it reports raw MAD, CTI-Bench TAA because its “correct+plausible” score includes partial credit, and AthenaBench RMS because it reports set-F1. AthenaBench VSP is retained because its normalization, max(0, 1 − MAD/7.7) × 100, produces a bounded higher-isbetter score.

## D.2 Standardization Choices

The standardized harness uses the common interface and configuration described in App. B. It records the prompt and inference configuration, raw model response, extracted prediction, questionlevel score, invalid-response status, and aggregate score for every run. Standardized extraction uses the pinned LLM-based policy described in App. E.1. We reviewed all eight benchmarks and found that none defines output-format compliance as the capability being evaluated. The judge can therefore recognize semantically equivalent answers without changing the intended task. A benchmark that explicitly evaluated output-format compliance would require a different policy.

Most pipeline resolutions follow documented behavior or supply a missing mechanical setting. Four choices admit meaningful alternatives and are therefore judgment-dependent. Table 12 makes these choices explicit. They should not be interpreted as uniquely correct fixes. Instead, the measured differences show that comparative conclusions can depend on defensible evaluation conventions.

<table><tr><td></td><td colspan="4">Released artifacts</td><td></td><td colspan="3">Original score (%)</td><td colspan="3">Standardized score (%)</td><td></td></tr><tr><td>Benchmark</td><td>Prompt</td><td>Inference</td><td>Evaluator</td><td>Params</td><td>Questions</td><td>Min</td><td>Mean</td><td>Max</td><td>Min</td><td>Mean</td><td>Max</td><td>∆(mean)</td></tr><tr><td>MMLU-CS</td><td>✓</td><td>√</td><td>√</td><td></td><td>100</td><td>36.0</td><td>68.8</td><td>87.0</td><td>74.0</td><td>83.0</td><td>90.0</td><td>+14.2</td></tr><tr><td>SecEval</td><td>√</td><td>√</td><td>√</td><td>x</td><td>2,189</td><td>0.3</td><td>39.3</td><td>78.0</td><td>57.0</td><td>71.4</td><td>82.0</td><td>+32.1</td></tr><tr><td>SECURE</td><td>✓</td><td>x</td><td>x</td><td>x</td><td>2,502</td><td>25.8</td><td>72.7</td><td>92.4</td><td>79.7</td><td>88.2</td><td>92.5</td><td>+15.5</td></tr><tr><td>CTI-Bench</td><td>√</td><td>x</td><td>✓</td><td>x</td><td>4,610</td><td>27.3</td><td>45.7</td><td>55.8</td><td>37.5</td><td>47.0</td><td>55.4</td><td>+1.3</td></tr><tr><td>AthenaBench</td><td>√</td><td>√</td><td>✓</td><td>x</td><td>8,100</td><td>21.0</td><td>48.0</td><td>76.8</td><td>50.6</td><td>58.3</td><td>75.3</td><td>+10.3</td></tr><tr><td>CyberMetric</td><td>√</td><td>√</td><td>√</td><td>x</td><td>500</td><td>1.0</td><td>59.4</td><td>96.8</td><td>85.2</td><td>92.3</td><td>96.2</td><td>+32.9</td></tr><tr><td>RedSage-Bench</td><td>x</td><td>√</td><td>√</td><td>√</td><td>30,000</td><td>25.5</td><td>75.2</td><td>90.9</td><td>75.1</td><td>84.6</td><td>91.1</td><td>+9.4</td></tr><tr><td>SecBench</td><td>x</td><td>x</td><td>x</td><td>x</td><td>661</td><td>33.9</td><td>59.8</td><td>85.2</td><td>66.3</td><td>81.9</td><td>89.7</td><td>+22.1</td></tr></table>

Table 11: Original and standardized pipeline summary. A green check indicates that the artifact is released and usable as specified; a red cross indicates that it is absent, contradictory, or not executable as written. For each pipeline, min, mean, and max summarize benchmark scores across the 10 evaluated models. ∆(mean) is the standardized mean minus the original mean, in percentage points.
<table><tr><td>Standardized choice</td><td>Scope</td><td>Alternative</td><td>Measured effect</td></tr><tr><td>Semantic LLM extraction</td><td>All benchmarks</td><td>Benchmark-specific released or reconstructed extractors</td><td>Released extraction rules differ by up to 79.7 pp on identical RedSage-Bench generations. CTI-Bench VSP extractors disagree on 30.4% of contested model-question pairs (F1 (E)).</td></tr><tr><td>Correct-over-total denominator</td><td>CTI-Bench, SECURE, AthenaBench</td><td>Exclude unparseable predictions and score correct over valid</td><td>CTI-RCM changes from 100.0% to 0.2% for Gemma-4, corresponding to 500 × inflation under correct-over-valid. AthenaBench VSP shifts by up to 85.4pp (F2(ε)).</td></tr><tr><td>Strict binary attribution</td><td>CTI-Bench, AthenaBench</td><td>Award partial credit to related threat actors</td><td>Scoring rules differ by up to 70 pp across the attribution tasks. In CTI-Bench, &quot;Correct+Plausible&quot; increases scores by up to 30 pp (F2 (A)).</td></tr><tr><td>Generative response scoring</td><td>MMLU-CS, RedSage-Bench</td><td>Rank answer choices using log probabilities</td><td>Differences reach 40.9 pp and can favor either scoring method depending on the model (F1(A)).</td></tr></table>

Table 12: Standardization choices for which a defensible alternative materially changes the measurement. The measured effects quantify sensitivity to these choices. They do not imply that one convention is uniquely correct.

## D.3 Prompt Sources

Prompt provenance differs substantially across benchmarks. CTI-Bench, AthenaBench, and SE-CURE provide task prompts with the released data. SecEval and CyberMetric encode their prompts in evaluation code. MMLU-CS relies on its established evaluation convention, while RedSage-Bench constructs its prompt in benchmark harness code rather than storing it with each question. SecBench releases the question data but no evaluation prompt, so we reconstruct one from the released fields.

Having a prompt somewhere in the release does not by itself pin the experiment. CyberMetric’s README prompt differs from the evaluator prompt. RedSage-Bench also ships inference examples with conflicting chat-formatting and generation settings. As a result, following the documentation and executing the released artifact can produce different pipelines under the same benchmark name. Table 10 summarizes this distinction across all 72 pipeline configuration fields.

## D.4 Pipeline Ledger

Table 13 records the benchmark-specific pipeline resolutions. The Released column describes executable behavior where an implementation exists. Resolution records the setting used to reconstruct or standardize the pipeline. The Action column uses four labels: retain preserves released behavior, fix changes behavior that prevents valid execution, normalize applies a consistent evaluation convention, and supply fills an unspecified field. Common settings already described in App. B are not repeated unless they resolve a benchmark-specific ambiguity or contribute to a measured effect.

When possible, the Measured effect column reports an isolated single-field counterfactual on the same questions. Downstream changes can be evaluated by re-scoring stored responses, while upstream changes require re-generation (App. F.2). Not isolated means that the field was changed or supplied but was not independently ablated. A peer-gap estimate, marked with <sup>†</sup>, is used only when a controlled counterfactual is unavailable; it is not treated as a controlled ablation.

Table 13: Benchmark pipeline ledger. Released describes the behavior implemented by the public artifact. Resolution records the choice used to reconstruct or standardize the pipeline. Action indicates whether we retain released behavior, fix a broken setting, normalize a working but inconsistent convention, or supply an unspecified field. Measured effect reports an isolated counterfactual where available; “Not isolated” indicates that the field was not independently ablated. The <sup>†</sup> symbol denotes a peer-gap estimate rather than a controlled counterfactual.
<table><tr><td>Field</td><td>Released</td><td>Resolution</td><td>Action</td><td>Measured effect</td></tr><tr><td colspan="5">MMLU-CS Chat formatting</td></tr><tr><td></td><td>Official evaluation uses raw completion without a chat template.</td><td>Use each model&#x27;s native chat template for the generative path; preserve raw completion for the logprob reproduction.</td><td>Normalize</td><td>Not isolated.</td></tr><tr><td>Max output tokens</td><td>Official logprob evaluation uses max_tokens=1.</td><td>Retain one token for logprob scoring and Supply use a sufficient budget for generative scoring; reasoning prompt-mode runs use 4,096 tokens.</td><td></td><td>Not isolated.</td></tr><tr><td>Extraction rule</td><td>The official logprob path generates no free-form answer and therefore has no response extractor.</td><td>Use the pinned LLM judge for the generative path.</td><td>Supply</td><td>Not isolated.</td></tr><tr><td>Scoring rule</td><td>Rank A-D using next-token log probability.</td><td>Use generative response scoring for standardized comparisons; retain logprob scoring as an audit alternative.</td><td>Normalize</td><td>Qwen3.6 differs by 23.0 pp: 57.0% generative versus 80.0% logprob.</td></tr><tr><td colspan="5">SecEval</td></tr><tr><td>Decoding</td><td>Not specified; the evaluator inherits Use T=0 and top-p=1.0. backend defaults.</td><td></td><td>Supply</td><td>Not isolated.</td></tr><tr><td>Max output tokens</td><td>The evaluator sets max_new_tokens=5.</td><td>Retain five tokens where supported; use Fix the 16-token minimum on ÁPI backends that reject smaller values.</td><td></td><td>GPT-5.4 changes from 0.3% to 81.4%, a +81.1 pp shift. At five tokens, all 2,189 requests fail before producing valid model output.</td></tr><tr><td colspan="5">SECURE</td></tr><tr><td>Chat formatting</td><td>No inference implementation is released.</td><td>Use each model&#x27;s native chat template with no additional system prompt.</td><td>Supply</td><td>Not isolated.</td></tr><tr><td>Decoding</td><td>The paper specifies T=0.7, but no released implementation enacts it.</td><td>Enact T=0.7 with top-p=1.0; sampled Supply runs pin seed 42.</td><td></td><td>Not isolated.</td></tr><tr><td>Max output tokens</td><td>Not specified in a released implementation.</td><td>Use 1,024 tokens.</td><td>Supply</td><td>Not isolated.</td></tr><tr><td>Stop sequences</td><td>Not specified in a released implementation.</td><td>Use no stop sequence.</td><td>Supply</td><td>Not isolated.</td></tr><tr><td>Extraction rule</td><td>No reference extractor is released.</td><td>Reconstruct the original path using a final-answer letter extractor, with True/False forms mapped to T/F; standardized scoring uses the pinned judge.</td><td>Supply</td><td>Not isolated.</td></tr><tr><td>Scoring rule</td><td>No executable scorer is released.</td><td>Use exact match for the reconstructed original path and the standardized judge verdict for standardized scoring.</td><td>Supply</td><td>Not isolated.</td></tr><tr><td>Denominator policy</td><td>No implementation is released; reported scores are consistent with</td><td>Retain every attempted question and count unparseable model answers as</td><td>Normalize</td><td>Primus-Nemotron on CWET changes from 100.0% correct-over-valid to 9.4%</td></tr><tr><td>Aggregation</td><td>excluding invalid predictions. No benchmark-level aggregation implementation is released.</td><td>incorrect. Report task scores separately; do not introduce an additional cross-task mean.</td><td>Supply</td><td>correct-over-total, a 90.6 pp difference. Not isolated.</td></tr><tr><td colspan="5">CTI-Bench</td></tr><tr><td>Chat formatting</td><td>Released notebooks target hosted chat APIs and do not define a</td><td>Use each model&#x27;s native chat template together with the CTI system prompt.</td><td>Supply</td><td>Not isolated.</td></tr><tr><td>Max output tokens</td><td>local-model chat template. The released notebook uses max_tokens=2048.</td><td>Retain 2,048 for MCQ, RCM, and VSP; Normalize use calibrated budgets of 8,192 for ATE</td><td></td><td>Not isolated.</td></tr><tr><td>Extraction rule</td><td>MCQ takes the final line; RCM and Reproduce the released rules for the VSP take the last matching expression anywhere in the</td><td>and 4,096 for TAA. original pipeline; use the pinned judge for standardized scoring.</td><td>Normalize</td><td>On VSP, the final-line and anywhere extractors disagree on 30.4% of model-question pairs for which at least one rule extracts a vector; disagreement</td></tr><tr><td>Scoring rule</td><td>request a final-line answer. TAA additionally awards Correct+Plausible credit to alias-connected or related actors.</td><td>Use strict binary scoring for standardized Normalize comparisons while recognizing true semantic aliases. Retain</td><td></td><td>reaches 96.5% for Primus-Merged. Correct+Plausible raises scores by up to 30.0 pp relative to strict scoring (Qwen3.6: 32.0% to 62.0%).</td></tr><tr><td>Denominator policy</td><td>Unparseable predictions are excluded from the denominator on</td><td>Correct+Plausible as an audit alternative. Use correct-over-total and report invalid-response rates separately.</td><td>Normalize</td><td>Gemma-4 on RCM changes from 100.0% to 0.2%, a 99.8 pp difference and 500× inflation under</td></tr><tr><td colspan="5">AthenaBench</td></tr><tr><td>Decoding</td><td>Uses T=0; top-p and no seed.</td><td>Retain T=0 and set top-p=1.0.</td><td>Supply</td><td>Not isolated. Continued on next page</td></tr></table>

Pipeline ledger, continued
<table><tr><td rowspan=1 colspan=7>Field           Released                    Resolution                      Action      Measured effect</td></tr><tr><td rowspan=1 colspan=7>Scoring rule       VSP uses                    Retain the released metric and make   Retain      Relative to CTI-Bench&#x27;s raw-MADmax(0, 1 − MAD/R) × 100,  R=7.7 explicit whenever VSP is                  convention, the metric direction andwith R read from config. yaml.   reported.                                      scale shift model ranks by up to fivepositions.</td></tr><tr><td rowspan=1 colspan=7>Five tasks retain all questions; VSPFor VSP, retain failed extractions and   Normalize   Gemma-4 changes from 85.4% toexcludes predictions whose CVSS assign the maximum deviation of 10;                0.0%; the median decrease across thevectors cannot be parsed.        other tasks are unchanged.                        10 models is 55.9 pp.</td></tr><tr><td rowspan=4 colspan=7>CyberMetricPrompt template    The released evaluator contains theUse the evaluator prompt as the original Fix         90.9 pp peer-gap estimate† forliteral answer placeholder ANSWER: reference, but replace the literal                    Foundation-Sec: 1.0% versus a 91.9%X; its prompt also differs from the  placeholder with an unambiguous                  peer median.README.                   answer-format instruction.</td></tr><tr><td rowspan=1 colspan=1>Prompt template</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Chat formatting</td><td rowspan=9 colspan=6>The evaluator targets a hosted chat Use each model&#x27;s native chat template Supply     Not isolated.API and defines no local-model   while preserving the evaluator&#x27;s systemath.                        instruction.Enact the documented sampling      Fix         Primus-Merged changes from 17.2% toparameters and inherits backend   configuration for the corresponding                 57.2%, a 40.0 pp shift; correctnessdefaults, although the paper      analysis; use greedy decoding as the                changes on 274 of 500 questions.specifies T=1.0, top-p=0.9, anddeterministic control.ot specified.                 Use 1,024 tokens.                 Supply     Not isolated.</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>p</td></tr><tr><td rowspan=1 colspan=1>Decoding</td><td rowspan=1 colspan=3>The evaluator sets no decoding</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>defaults, although the paper</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>specifies T=1.0, top-p=0.9, and</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=1>Max output tokens  N</td></tr><tr><td rowspan=1 colspan=1>RedSage-Bench</td><td rowspan=5 colspan=6>The prompt is constructed by     Use the released prompt layout with   Supply     Not isolated.cybersec_prompt_fn; its        include_context=False.include_context sefting differs</td></tr><tr><td rowspan=1 colspan=1>Prompt template</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>cybersec_prompt_fn; its</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>include_context setting differs</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>across call sites.</td></tr><tr><td rowspan=1 colspan=1>Chat formatting</td><td rowspan=1 colspan=3>Released run configurations</td><td rowspan=2 colspan=3>Use each model&#x27;s native chat template. Normalize   Not isolated.</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>override chat-template handling.</td></tr><tr><td rowspan=1 colspan=1>Decoding</td><td rowspan=1 colspan=3>The generative configuration uses</td><td rowspan=2 colspan=3>Retain greedy decoding and set       Normalize  Not isolated.top-p=1.0; no sampling seed is requiredfor the deterministic run.</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>T=0, with top-p and seed unset.</td></tr><tr><td rowspan=1 colspan=1>Max output tokens</td><td rowspan=1 colspan=3>The generative MCQ task uses</td><td rowspan=4 colspan=3>Use 2,048 tokens for the generative pathNormalize   Evaluated jointly with stop-sequenceso reasoning spans can complete before              handling below.answer extraction.</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>generation_size=100; the</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>logprob path does not generate a</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>response.</td></tr><tr><td rowspan=1 colspan=1>Stop sequences</td><td rowspan=1 colspan=3>The generative MCQ task uses</td><td rowspan=1 colspan=2>Generate without a backend newline stop</td><td rowspan=1 colspan=1>, Fix         Qwen3.6&#x27;s benchmark mean changes</td></tr><tr><td rowspan=3 colspan=1></td><td rowspan=1 colspan=3>stop_sequence=[&quot;\n&quot;].</td><td rowspan=1 colspan=2>remove the completed reasoning span,</td><td rowspan=3 colspan=1>from 0.0% to 85.9%; per-task recoveryranges from 81.4 to 90.1 pp.</td></tr><tr><td rowspan=2 colspan=3></td><td rowspan=2 colspan=2>and apply answer handling only after</td></tr><tr><td rowspan=1 colspan=1>reasoning is stripped.</td></tr><tr><td rowspan=1 colspan=1>Extraction rule</td><td rowspan=1 colspan=3>Three metrics are registered on the</td><td rowspan=1 colspan=2>Use the pinned judge for standardized</td><td rowspan=4 colspan=1>Normalize   Exact versus prefix match differs by upto 79.7 pp on identical generations(Primus-Merged: 0.3% versus 80.0%).</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>same generations: exact match,</td><td rowspan=1 colspan=2>scoring and retain the released metrics</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>prefix exact match, and regex MCQ for</td><td rowspan=1 colspan=2>sensitivity analysis.</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>accuracy; none is designatedcanonical.</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>Scoring rule</td><td rowspan=1 colspan=3>Both logprob and generative scoring U</td><td rowspan=1 colspan=2>se generative response scoring for</td><td rowspan=3 colspan=1>Normalize   Differences reach 40.9 pp forGemma-4; Qwen3.6 moves in theopposite direction by 26.7 pp.</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>are available.</td><td rowspan=1 colspan=2>standardized comparisons; retain logprob</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2>scoring as an audit alternative.</td></tr><tr><td rowspan=1 colspan=1>SecBench</td><td rowspan=2 colspan=3>No evaluation prompt is released.</td><td rowspan=2 colspan=2>Reconstruct the prompt from the released</td><td rowspan=2 colspan=1>Supply      Not isolated; no released prompt</td></tr><tr><td rowspan=1 colspan=1>Prompt template</td></tr><tr><td rowspan=3 colspan=1></td><td rowspan=3 colspan=3></td><td rowspan=1 colspan=2>question and answer-choice fields,</td><td rowspan=3 colspan=1>provides a controlled baseline.</td></tr><tr><td rowspan=1 colspan=2>requiring only the selected option letter</td></tr><tr><td rowspan=1 colspan=2>or letters.</td></tr><tr><td rowspan=2 colspan=1>Chat formatting</td><td rowspan=2 colspan=3>Not specified.</td><td rowspan=2 colspan=2>Use each model&#x27;s native chat templatewith no additional system prompt.</td><td rowspan=2 colspan=1>Supply     Not isolated.</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Decoding</td><td rowspan=1 colspan=3>Not specified.</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Use greedy decoding with T=0 andtop-p=1.0.</td></tr><tr><td rowspan=1 colspan=1>Max output tokens</td><td rowspan=1 colspan=3>Not specified.</td><td rowspan=1 colspan=2>Use 16 tokens for the short answerformat.</td><td rowspan=1 colspan=1>Supply     Not isolated.</td></tr><tr><td rowspan=1 colspan=1>Stop sequences</td><td rowspan=1 colspan=3>Not specified.</td><td rowspan=1 colspan=2>Use no stop sequence.</td><td rowspan=1 colspan=1>Supply     Not isolated.</td></tr><tr><td rowspan=1 colspan=4>Extraction rule</td><td rowspan=1 colspan=1>No extractor is released.</td><td rowspan=1 colspan=2>Reconstruct the original path by</td></tr><tr><td rowspan=1 colspan=1>Scoring rule</td><td rowspan=1 colspan=3>No executable scorer is released.</td><td rowspan=1 colspan=3>Use exact match on the normalized    Supply     Not isolated.answer set.</td></tr><tr><td rowspan=1 colspan=1>Denominator policyNo</td><td rowspan=1 colspan=6>t specified.                 Use correct-over-total; unparseable    Supply     Not isolated.answers count as incorrect.</td></tr><tr><td rowspan=1 colspan=7>Aggregation      Not specified.                 Report one accuracy over the 661     Supply     Not isolated.English multiple-choice questions.</td></tr></table>

## E Extraction and Judging

This section describes the standardized extraction and judging procedure used by the harness and evaluates its reliability. We first specify the pinned judge policy, then validate its output behavior and decisions through human and targeted stress-test analyses, and finally assess sensitivity to the choice of judge using an independent model.

## E.1 Judge Policy

The standardized scoring path uses a single LLM judge call to extract the model’s stated answer and grade it. This reduces dependence on benchmarkspecific surface-form extractors while preserving the task’s intended answer semantics. For reference, Code 3 shows the deterministic multiplechoice extractor used in our surface-form comparisons. It prioritizes an explicit final answer and otherwise falls back through progressively weaker surface-form cues. Unlike the standardized judge, it cannot distinguish a committed answer from a letter or identifier mentioned only during reasoning. This distinction contributes to the extractor divergence measured under F<sub>1</sub>(E).

```python
CODE 3 : Deterministic MCQ extractor
# for task_type=="mcq"; else return last line
extract_mcq(response):
lines = nonempty_lines(response)
# explicit line, "**Answer:** X" then "Answer: X"
# case-insensitive, scan last->first
for pat in [r"\*\*(?:Final )?Answer:\*\*\s*([A-D])",
r"(?:Final )?Answer:\s*([A-D])"]:
for line in reversed(lines):
if m = search(pat, line, IGNORECASE):
return upper(m)
# a line that is exactly one letter A-D
for line in reversed(lines):
if upper(line) in {A,B,C,D}:
return upper(line)
# a line starting with a letter not followed by ")"
for line in reversed(lines):
if upper(line[0]) in {A..D} and line[1] != ")":
return upper(line[0])
# any A-D token, case-insensitive, scan last->first
for line in reversed(lines):
if m = search(r"\b([A-D])\b", upper(line)):
return m
return last_line
```

The pinned judge is GPT-5.4 served through Azure OpenAI at temperature 0. Before judging, reasoning enclosed in <think> tags is removed. A judge-side failure, such as an API error or empty judge response, excludes the corresponding question from both numerator and denominator. By contrast, an empty or unparseable model answer

remains in the denominator and is scored incorrect.   
No judge-side failures occurred in the evaluation.

The judge prompt is shown in Prompt 4. Each task supplies a format\_hint and compare\_rule that define the expected answer representation and grading rule. Examples include a single multiplechoice letter, a normalized CVSS:3.1/ vector for VSP, and a sorted set of parent ATT&CK technique identifiers for ATE. The judge must first extract what the model actually states and then issue a binary verdict. It is explicitly instructed not to infer or repair an answer.

PROMPT 4 : Pinned judge prompt   
You are a strict evaluator for cybersecurity benchmarks.   
Your role has TWO steps.   
{question\_block}   
Correct Answer: {ground\_truth}   
Model's Response:   
{model\_answer}   
STEP 1 - EXTRACT the model's answer.   
- Look for the model's explicit final answer (e.g.,   
"Answer:", concluding line, single bolded line).   
- IGNORE letters/IDs/text that appear only inside   
explanations of other options or thinking-aloud prose.   
- If the model gives multiple inconsistent answers, use   
its most prominent/final selection.   
- DO NOT correct, infer, or improve the answer - extract   
verbatim what it actually said.   
- Format the extracted answer as: {format\_hint}   
STEP 2 - VERDICT.   
{compare\_rule}   
Output ONLY this JSON, nothing else:   
{   
"extracted\_answer": "<your formatted extraction>",   
"verdict": "CORRECT" or "INCORRECT",   
"justification": "<one short sentence>"   
}

## E.2 Judge Validation

We audit the pinned judge at three levels: outputformat compliance, human validation, and targeted stress testing. Across all 48,662 questions and 10 evaluated models, the judge makes 486,620 grading calls. Re-parsing every raw judge response shows near-perfect adherence to the requested schema. JSON conformance is 100.0% for multiple-choice calls and 99.99% for open-ended calls; fewer than 0.01% require the tolerant parsing fallback, and every successfully parsed verdict is either CORRECT or INCORRECT. Table 14 summarizes extraction behavior by answer type.

Multiple-choice outputs are almost always directly extractable. Open-ended tasks are more difficult because models can mention several candidate identifiers or entities without committing to one. This occurs particularly in free-form ID-set and attribution tasks. The judge is instructed not to synthesize an answer from such transient mentions.

<table><tr><td>Answer type</td><td>Calls</td><td>JSON (%)</td><td>Extract (%)</td><td>Format (%)</td></tr><tr><td>Multiple-choice</td><td>414,520</td><td>100.00</td><td>99.98</td><td>99.92</td></tr><tr><td>Open-ended</td><td>72,100</td><td>99.99</td><td>97.87</td><td>97.87</td></tr></table>

Table 14: Pinned-judge behavior by answer type over 486,620 grading calls. Calls is the number of grading calls made by the judge. JSON is the rate of valid three field JSON responses, Extract is the rate of non-empty extracted answers, and Format is the rate at which the extraction satisfies the task’s expected answer format.

Example 1 illustrates this behavior: the model mentions the correct CWE during its reasoning but later contradicts itself and never commits to a final CWE, so the judge correctly records NONE rather than crediting the earlier mention.

EXAMPLE 1 : No committed final answer   
(CTI-Bench RCM, Llama-3.3)   
Prompt: Map the CVE to a CWE; The last line of your   
response must contain only the CWE ID.   
Correct answer: CWE-79.   
Raw output: Opens correctly, quoting “. . . a classic ex  
ample of a CWE-79 . . . Cross-site Scripting . . . ”, then   
degenerates into a contradictory loop (“. . . the closest   
match is actually CWE-184, no, . . . CWE-707, no . . . ”)   
and never emits a final CWE line.   
Extractor: Judge JSON: extracted\_answer=“NONE”,   
verdict=INCORRECT, justification=“never pro  
vides a clear final CWE selection . . . unresolved con  
tradictory reasoning.”   
Consequence: The correct answer appears during rea  
soning but is never committed as the final answer. The   
judge therefore records NONE, and the question is scored   
incorrect.

We next manually validate a random stratified sample of 80 judge decisions, 10 from each benchmark and distributed across its tasks so that structured and open-ended outputs are represented. Human verification agrees with the judge’s extraction and verdict on all 80 questions, including 64 multiple-choice and 16 open-ended cases.

Because a random sample contains relatively few difficult extractions, we additionally target two hard strata: the 1,733 calls for which the pinned judge returns NONE, and the 50,456 multiple-choice-family calls for which its verdict differs from a deterministic final-answer regex on the same response. On the NONE stratum, the independent Sonnet 4.6 judge corroborates the absence of a committed answer on 94.6% of calls. On the regex-disagreement stratum, the two judges agree on 98.1% of verdicts, with Cohen’s κ=0.83. Agreement across the two hard strata is 98.0%, compared with 99.6% over the full independent-judge evaluation given in App. E.3.

<table><tr><td>Evaluated model</td><td>Calls</td><td>Agree (%)</td><td>κ</td><td>∆(pp)</td></tr><tr><td>GPT-5.4</td><td>48,612</td><td>99.68</td><td>0.988</td><td>+0.22</td></tr><tr><td>Gemma-4</td><td>48,612</td><td>99.44</td><td>0.981</td><td>-0.18</td></tr><tr><td>Qwen3.6</td><td>48,612</td><td>99.60</td><td>0.988</td><td>-0.19</td></tr><tr><td>Llama-3.3</td><td>48,612</td><td>99.68</td><td>0.991</td><td>-0.21</td></tr><tr><td>GPT-OSS</td><td>48,612</td><td>98.43</td><td>0.957</td><td>+0.20</td></tr><tr><td>Primus-Nemotron</td><td>48,612</td><td>99.81</td><td>0.995</td><td>-0.10</td></tr><tr><td>Primus-Merged</td><td>48,612</td><td>99.93</td><td>0.998</td><td>-0.00</td></tr><tr><td>Foundation-Sec</td><td>48,612</td><td>99.84</td><td>0.996</td><td>+0.08</td></tr><tr><td>RedSage-Qwen3</td><td>48,612</td><td>99.91</td><td>0.997</td><td>-0.01</td></tr><tr><td>All</td><td>437,508</td><td>99.59</td><td>0.989</td><td>-0.02</td></tr></table>

Table 15: Agreement between the pinned judge (GPT-5.4) and the independent judge (Sonnet 4.6) on identical stored model outputs. Agree is verdict agreement. κ is Cohen’s kappa. ∆ is the evaluated model’s accuracy under the pinned judge minus its accuracy under the independent judge. The All row pools all 437,508 verdict pairs when computing agreement and Cohen’s κ; ∆ is the population-weighted mean score difference across models, which equals the unweighted mean here because each model contributes 48,612 calls.

We also manually inspect an adversarial sample of 120 questions, split evenly between these two hard strata. Two annotators independently label every question using a written guideline and adjudication procedure released with the artifacts. Inter-annotator agreement is Cohen’s κ=0.93, with 99.2% raw agreement. After adjudication, the pinned judge is correct on 113 of 120 questions (94.2%; Wilson 95% CI [88.4, 97.1]). The seven raw judge errors include both over-crediting an uncommitted answer and missing a committed answer obscured by model-output artifacts. Two correspond to the known SECURE true-or-false formathint mismatch and are corrected before benchmark scores are computed.

## E.3 Independent Judge

Because GPT-5.4 is both the pinned judge and one of the evaluated models, we test whether the choice of judge materially favors it or changes model comparisons. Sonnet 4.6 independently re-grades the stored outputs of the other nine evaluated models on 22 of the 23 tasks, excluding CTI-Bench attacker attribution. Model outputs are held fixed; only the judge changes. This produces 48,612 regrading calls per model and 437,508 calls in total.

As listed in Table 15, the judges agree on 99.6% of verdicts, with Cohen’s κ=0.99. Per-model score differences are also small. GPT-5.4 scores 0.22 pp higher under the pinned judge than under Sonnet 4.6, the largest absolute difference among the nine re-graded models. The population-weighted mean difference is -0.02 pp. Recomputing the ninemodel ranking under the independent judge yields Spearman $\rho { = } 0 . 9 8$ , with GPT-5.4 ranked first under both judges. Agreement falls below $\kappa { = } 0 . 9$ on only two tasks: CyberMetric $\scriptstyle ( \kappa = 0 . 8 3 )$ and SECURE-KCV $\scriptstyle ( \kappa = 0 . 7 3 )$ . The SECURE-KCV disagreement is driven by a true-or-false vs. letter format-hint mismatch in the pinned judge’s raw outputs. This mismatch is detected and corrected before score computation, so it does not affect the reported SECURE-KCV scores.

## F Audit Protocol

This section details the audit procedure, evidence types, the affected models, benchmarks, or questions, and stage coverage for the 15 failure modes.

## F.1 Procedure

The meta-evaluation methodology defined in §2.1 describes the audit protocol and treats controlled perturbation as opportunistic. Here we clarify how the audit was executed and how we handle cases where an effect cannot be isolated. The authors performed the audit. The harness automates configuration logging, output collection, parsing diagnostics, re-scoring, and aggregate comparisons. Inspection of benchmark documentation and released implementations, anomaly tracing, and interpretation of observed discrepancies combine automated diagnostics with manual review. Gold-label auditing additionally uses the search-grounded verification procedure in App. G.1.2.

Failure incidence and effect estimation are separate. We inspect every benchmark for every failure mode, yielding 120 benchmark–failure interactions. Each interaction is recorded as observed or not observed, independent of whether a controlled perturbation is possible. The resulting binary incidence matrix is reported in Table 4. After a failure is identified, we estimate its effect when the pipeline admits a meaningful comparison. A controlled perturbation may be unavailable because the original configuration cannot be reconstructed, the relevant setting is not exposed, or isolating the field would require generations that were not run. In such cases, we report the available inspection evidence or a clearly identified peer-gap estimate rather than treating it as a controlled ablation.

## F.2 Perturbations

The audit uses four forms of evidence. Dataset audits inspect benchmark questions or labels without perturbing model outputs. Re-scoring applies an alternative extraction, scoring, denominator, metric, or aggregation rule to stored responses, holding the model generations fixed. Re-generation changes a prompt or inference configuration and generates new responses for the same questions. Finally, paired evaluation compares evaluation interfaces that cannot be reduced to re-scoring identical generated text, such as logprob and generative multiple-choice scoring.

The distinction matters for reproducibility. Rescoring is deterministic given the stored responses and evaluation configuration. Re-generation reproduces the pinned experimental configuration rather than guaranteeing byte-identical outputs. For openweight models, we pin checkpoint revisions and the serving stack. For GPT-5.4 and Sonnet 4.6, we additionally record the hosted model identifier and run timestamp because a hosted endpoint may change while retaining the same API-facing name. Table 16 records the primary evidence used for each failure mode and whether the corresponding comparison holds model outputs fixed.

## F.3 Affected Units

Failure incidence is defined at the benchmark– failure level, whereas the effect in Table 3 can have different affected units. Dataset-stage failures are summarized across benchmarks. Most prompt, inference, extraction, and scoring failures are summarized across affected models. Aggregation failures can instead be benchmark-level. So, affected units should be read as the population over which the reported effect is observed and summarized, not as a second failure-incidence matrix. Per-model outputs and logs are retained in the audit repository.

## F.4 Evidence Index

Table 17 indexes the evidence supporting each failure mode. Nine modes admit question-level examples showing the prompt, raw model output, extraction behavior, and scoring consequence; $\mathcal { F } _ { 1 } ( \mathcal { P } )$ has two such examples. These boxes are illustrative witnesses rather than the basis for the aggregate effect estimates. Other modes are inherently distributional or comparative: capability coverage and label correctness require dataset-level evidence, metric direction and aggregation operate over collections of predictions, and prompt-mode and logprob sensitivity require paired evaluations. We therefore report each failure at the level at which its mechanism can be demonstrated.

<table><tr><td>ID</td><td>Failure mode</td><td>Analysis</td><td>Held</td><td>Effect evidence</td></tr><tr><td>F1(D)</td><td>Limited capability coverage</td><td>Dataset audit</td><td></td><td>Distribution of question types within each benchmark.</td></tr><tr><td>F2(D)</td><td>Gold-label correctness</td><td>Label audit</td><td></td><td>Search-grounded verification of disagreement flags.</td></tr><tr><td>F1(P)</td><td>Format-token leakage</td><td>Output audit / peer gap</td><td>Yes</td><td>Invalid-response behavior and peer gap; standardized scoring requires re-generation with the corrected prompt.</td></tr><tr><td>F2(P)</td><td>Prompt-question conflict</td><td>Output audit</td><td>Yes</td><td>Single-letter response rate on questions whose gold answer requires multiple selections.</td></tr><tr><td>F3(P)</td><td>Template incompatibility</td><td>Output audit / peer gap</td><td>Yes</td><td>Output-format behavior and peer-gap estimate</td></tr><tr><td>F1(I)</td><td>Stop-sequence mismatch</td><td>Re-generation</td><td>No</td><td>Paired runs with and without the conflicting stop sequence.</td></tr><tr><td>F2(T)</td><td>Token-budget filter</td><td>Re-generation</td><td>No</td><td>Runs under the rejected and valid token budgets.</td></tr><tr><td>F3(I)</td><td>Decoding drift</td><td>Re-generation</td><td>No</td><td>Paired runs under released/default and documented decoding settings.</td></tr><tr><td>F1(ε)</td><td>Extractor divergence</td><td>Re-scoring</td><td>Yes</td><td>Alternative extractors applied to identical responses.</td></tr><tr><td>F2(ε)</td><td>Denominator inflation</td><td>Re-scoring</td><td>Yes</td><td>Correct-over-valid and correct-over-total applied to identical predictions.</td></tr><tr><td>F3(ε)</td><td>Metric-direction mismatch</td><td>Re-scoring</td><td>Yes</td><td>Identical task results ranked under opposite metric directions.</td></tr><tr><td>F4(E)</td><td>Prompt-mode sensitivity</td><td>Re-generation</td><td>No</td><td>Same questions evaluated under zero-shot, few-shot, and CoT prompts.</td></tr><tr><td>F1(A)</td><td>Logprob vs. generative scoring</td><td>Paired evaluation</td><td>No</td><td>Same questions evaluated through logprob and generative scoring paths.</td></tr><tr><td>F2(A)</td><td>Task-level metric drift</td><td>Re-scoring</td><td>Yes</td><td>Alternative credit rules applied to identical extracted predictions.</td></tr><tr><td>F3(A)</td><td>Aggregation inconsistency</td><td>Re-scoring</td><td>Yes</td><td>Alternative denominator and aggregation conventions applied to the corresponding question-level results.</td></tr></table>

Table 16: Primary analysis used to characterize each failure mode. Held indicates whether model outputs are held fixed, so the comparison operates on identical stored model responses. “—” indicates the analysis does not involve model outputs. Failure incidence is determined independently by the binary inspection reported in Table 4.

<table><tr><td>Mode</td><td>Witness</td><td>Reference</td></tr><tr><td>F1(D)</td><td>Question-type distribution</td><td>Fig. 1; App. G.1.1</td></tr><tr><td></td><td>F2(D) Search-grounded label verification and verified cases</td><td>App. G.1.2; Cases 1–5</td></tr><tr><td></td><td>F1(P) Two CyberMetric responses</td><td>App. G.1.4; Examples 2–3</td></tr><tr><td>F2(P)</td><td>SecEval multi-select responses</td><td>Table 20; Example 4</td></tr><tr><td></td><td>F3 (P) SECURE template response</td><td>App. G.1.6; Example 5</td></tr><tr><td>F1(I)</td><td>RedSage stop-sequence ablation</td><td>Table 21; Example 6</td></tr><tr><td>F2(I)</td><td>SecEval API rejection</td><td>App. G.2.2; Example 7</td></tr><tr><td>F3(I)</td><td>CyberMetric decoding ablation</td><td>Table 22; Example 8</td></tr><tr><td>F1(ε)</td><td>Same-generation extractor comparisons</td><td>Tables 23, 24; Example 9</td></tr><tr><td>F2(E)</td><td>Alternative denominator policies</td><td>Table 25; Example 10</td></tr><tr><td>F3(E)</td><td>Same predictions under opposite metric directions</td><td>App. G.3.3</td></tr><tr><td>F4(E)</td><td>Three prompt modes on the same questions</td><td>App. G.3.4</td></tr><tr><td>F1(A)</td><td>Logprob and generative scoring</td><td>Table 26; App. G.4.1</td></tr><tr><td>F2(A)</td><td>Alternative attacker-attribution credit rules</td><td>Table 27; Example 11</td></tr><tr><td></td><td>F3(A) Cross-benchmark aggregation comparison</td><td>App. G.4.3; Table 28</td></tr></table>

Table 17: Evidence index for the 15 failure modes. Question-level examples provide illustrative witnesses, while tables and appendix sections report the corresponding aggregate or comparative evidence.

<table><tr><td>Benchmark</td><td>D</td><td>P</td><td>I</td><td>ε</td><td>A</td></tr><tr><td>MMLU-CS</td><td>2/2</td><td>0/3</td><td>0/3</td><td>0/4</td><td>1/3</td></tr><tr><td>SecEval</td><td>1/2</td><td>1/3</td><td>1/3</td><td>0/4</td><td>1/3</td></tr><tr><td>SECURE</td><td>1/2</td><td>1/3</td><td>0/3</td><td>1/4</td><td>0/3</td></tr><tr><td>CTI-Bench</td><td>1/2</td><td>1/3</td><td>1/3</td><td>3/4</td><td>2/3</td></tr><tr><td>AthenaBench</td><td>2/2</td><td>0/3</td><td>0/3</td><td>3/4</td><td>1/3</td></tr><tr><td>CyberMetric</td><td>1/2</td><td>1/3</td><td>1/3</td><td>0/4</td><td>0/3</td></tr><tr><td>RedSage-Bench</td><td>2/2</td><td>0/3</td><td>1/3</td><td>1/4</td><td>1/3</td></tr><tr><td>SecBench</td><td>2/2</td><td>0/3</td><td>0/3</td><td>0/4</td><td>0/3</td></tr><tr><td>Total</td><td>12/16</td><td>4/24</td><td>4/24</td><td>8/32</td><td>6/24</td></tr></table>

Table 18: Stage-level audit coverage. Each cell reports observed failure interactions over failure modes examined at that stage. All 120 benchmark–failure interactions were inspected; 34 were observed.

## F.5 Stage Coverage

Table 18 aggregates the binary failure-incidence matrix by pipeline stage. Every benchmark is inspected for every failure mode associated with each stage. Consequently, all 40 benchmark–stage combinations are covered, and the denominators sum to the 120 benchmark–failure interactions in Table 4. The numerators sum to the 34 observed interactions. An unobserved interaction therefore means that the failure was inspected but not found; it does not mean that the corresponding stage was omitted from the audit.

PROMPT 5 : Knowledge-vs.-analytical classifier   
SYSTEM   
You are a careful annotator classifying cybersecurity benchmark questions into one of two classes:   
K = Knowledge-oriented. Answered by retrieving a fact, definition, standard parameter, or procedural step from   
cybersecurity domain knowledge. Single-step factual recall. Surface form is usually a short MCQ on definitions,   
properties, configuration values, standards, or named entities.   
A = Analytical. Requires reasoning over a scenario, structured extraction from text, multi-step inference, scoring, or   
attribution. Surface form usually involves a free-form description that the answerer must analyze (CVSS vector   
composition, MITRE ATT&CK technique extraction, CWE root-cause classification, threat-actor attribution).   
Decision rules:   
- If a single fact or definition produces the answer, choose K.   
- If the answer requires reading a scenario / report / description and inferring or extracting structured information from   
it, choose A.   
- Multi-choice format alone does not make a question K - what matters is whether the \*answer\* requires recall or reasoning.   
- Choose A if the question presents a non-trivial input (code, log, CVE description, attack narrative) that must be   
analyzed before answering.   
Always reply with strict JSON in a fenced \`\`\`json\`\`\` block, with exactly these keys: "class" (K or A), "confidence" (low/   
medium/high), "rationale" (one short sentence).   
USER   
Here are six worked examples; classify the final question in the same format.   
Question: Which of the following is a desirable property of a biometric system?   
Options: A) Permanent, B) Transferability, C) Uniformity, D) Forgiveness   
\`\`\`json {"class": "K", "confidence": "high", "rationale": "Single-step factual recall about biometric-system design   
properties."} \`\`   
[Five additional worked examples omitted here; included in the released artifact.]   
Question: {question}

## F.6 Reading the Failure Table

Table 3 reports four quantities for each failure. Impact measure defines the quantity being summarized. Affected units identifies the population over which the effect is observed, such as models or benchmarks. Maximum effect is the largest observed value over that population, and Median effect is its median when enough affected units exist to summarize a distribution. A dash indicates that a meaningful median is unavailable. The two dataset-stage rows instead report a single aggregate quantity spanning the final two columns.

The impact measures are interpreted as follows. Dominant question-type share is the fraction of sampled questions assigned to the most common capability type. Flag precision is the fraction of checked disagreement flags confirmed as gold-label errors. Peer score gap is the score difference between an affected model and the median of unaffected peers under the same benchmark condition. Single-letter response rate is the fraction of multianswer questions for which a model returns only one answer letter. Score gap is the score difference induced by the compared pipeline settings. Extractor score gap applies alternative extraction rules to the same generations. Rank shift is the displacement in model rank under the compared metric conventions. Score spread is the range across the evaluated prompt modes. Convention gap measures the difference between alternative task-level credit rules. Denominator gap measures the score difference associated with inconsistent denominator or aggregation conventions.

## G Failure Evidence

This section provides evidence for the 15 failure modes in Table 3, organized by pipeline stage with aggregate results and question-level examples.

## G.1 Dataset

## G.1.1 Capability classification (F<sub>1</sub>(D))

As presented in §4, we classify each question in each benchmark as knowledge-oriented or analytical by majority vote of four LLM classifiers: GPT-5.4, Qwen3.6, Llama-3.3, and RedSage-Qwen3. Each classifier uses the same six-shot prompt at temperature 0 and returns a strict JSON label. Across a stratified sample of 2,155 questions from all tasks, agreement is substantial, with Fleiss’s κ=0.753 and pairwise Cohen’s κ ranging from 0.66 to 0.87. Because the classifiers overlap with the evaluated models, we treat these labels as a coverage estimate rather than gold annotations. Prompt 5 shows the classification instruction and output schema; the complete six-shot prompt is released with the audit artifacts.

“ADP: CISA-ADP, Base Score: 7.5 HIGH Vector:   
CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H.”

<table><tr><td>Benchmark</td><td>Flagged questions</td></tr><tr><td>MMLU-CS</td><td>2</td></tr><tr><td>SecEval</td><td>216</td></tr><tr><td>SECURE</td><td>33</td></tr><tr><td>CTI-Bench</td><td>37</td></tr><tr><td>AthenaBench</td><td>795</td></tr><tr><td>CyberMetric</td><td>0</td></tr><tr><td>RedSage-Bench</td><td>53</td></tr><tr><td>SecBench</td><td>4</td></tr><tr><td>Total</td><td>1,140</td></tr></table>

Table 19: Distribution of the 1,140 disagreement flags across benchmarks. Of these, 998 are checked by the search-grounded verifier

## G.1.2 Gold-label correctness $( \mathcal { F } _ { 2 } ( \mathcal { D } ) )$

The gold-label audit has two automated stages followed by human validation. First, for each question we consider models with parseable predictions and flag the question when at least 50% select the same non-gold answer, effectively acting as a disagreement filter. The threshold applies to the most common non-gold answer, that is, questions with a split non-gold vote below 50% or a majority matching the gold label are not flagged. This stage is a triage mechanism, not a label-quality judgment.

Flagged questions are then checked by a GPT-5.4 search-grounded verifier restricted to multi-tiered, authoritative cybersecurity sources. The verifier returns one of four outcomes: gold correct, gold mislabel, both wrong, or uncertain. Tier 1 sources include CVE, CWE, CVSS, MITRE ATT&CK, NVD, CISA, and relevant RFC and NIST documents (Sikos, 2023). Tier 2 includes coordinateddisclosure and vendor advisories. Tier 1 evidence takes precedence when sources conflict; unresolved or insufficient evidence yields uncertain rather than a forced binary decision.

The disagreement filter flags 1,140 questions, of which 998 are checked by the grounded verifier. Among these, 238 are confirmed gold mislabels (23.8%), 653 are false-positive flags for which the gold label is correct (65.4%), 16 are both wrong (1.6%), and 91 are uncertain (9.1%). Thus, 23.8% is the flag precision on the checked set, not a benchmark-wide label-error rate. Its Wilson 95% CI is [21.3%, 26.6%]. As the flagged set is selected through model disagreement rather than random sampling, we do not extrapolate this rate to the full benchmarks. Of the 1,140 flags, 405 reach at least 75% model agreement and 84 are unanimous. Table 19 shows that the flags concentrate heavily in AthenaBench and SecEval.

Two human annotators independently validate a stratified 50-question sample spanning all four verifier outcomes. Inter-annotator agreement is Cohen’s $\kappa = 0 . 6 8$ with 94% raw agreement. After adjudication, the human check confirms 46 of 50 verifier decisions (92%; Wilson 95% CI [81.2, 96.8]). Of the remaining four cases, two are verifier errors and two cannot be resolved from the cited evidence. We therefore describe the verifier results as automatically verified, so manual validation refers only to judgments made by the human annotators.

Gold-label cases. For each benchmark with a verifier-confirmed mislabel, we give one representative case showing the benchmark question, published gold answer, model-majority prediction, verifier reasoning, and supporting source (Cases 1–5).

CASE 1 : RedSage-Bench / skills (idx 2149)   
Prompt (truncated): “What is the total length of an   
IP packet, including the header and payload, when the   
payload size is 32 bytes and the IP header size is 20   
bytes? (A) 84 (B) 52 (C) 60 (D) 72”   
Gold: C (60 bytes)   
Model majority: B (52 bytes)   
Verifier: The IPv4 total-length field counts the whole   
datagram, including header and data, so $2 0 + 3 2 = 5 2 $   
bytes. Option B is correct and the published gold label C   
is wrong.   
Sources:   
• https://www.rfc-editor.org/rfc/rfc791   
“Total Length . . . is the length of the datagram,   
measured in octets, including internet header and   
data.”

## CASE 2 : CTI-Bench / vsp (idx 125)

Prompt (truncated): “Analyze the following CVE description and calculate the CVSS v3.1 Base Score . . . Summarize each metric’s value and provide the final CVSS v3.1 vector string.”

Gold: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/   
I:H/A:H

Model majority: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/   
S:U/C:N/I:N/A:H

Verifier: The NVD record for CVE-2021-42141 contains two assessments. NVD’s enrichment gives C:H/I:H/A:H, while CISA-ADP gives C:N/I:N/A:H. The description states only that the flaw may cause denial of service, supporting availability impact alone. This means the published gold overstates confidentiality and integrity impact.

Sources:

```csv
• https://nvd.nist.gov/vuln/detail/
CVE-2021-42141
```

## CASE 3 : AthenaBench / athena\_vsp (idx 1642)

Gold: CVSS:3.1/AV:L/AC:L/PR:N/UI:R/S:U/C:H/   
I:H/A:H

Model majority: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/   
S:U/C:H/I:H/A:H

Verifier: The CVE description states that remote attackers can execute arbitrary code without authentication, supporting AV:N and PR:N. NVD and the Zero Day Initiative both assign the network-accessible vector; the published AV:L/UI:R values conflict with those sources.

• https://nvd.nist.gov/vuln/detail/   
CVE-2025-3484   
“CNA: ZDI, Base Score: 9.8 CRITICAL Vector:   
CVSS:3.0/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H.”

## CASE 4 : SECURE / secure\_kcv (idx 84)

Prompt (truncated): “Given the CVE-2024-5048   
record, a SourceCodester Student Management System   
SQL-injection issue . . . True or False: remote code exe  
cution is a potential impact.”   
Gold: T   
Model majority: F   
Verifier: The CVE/NVD record classifies the issue as   
CWE-89 with C:L/I:L/A:L impacts and provides no ev  
idence of code execution. The published gold appears   
incorrect and the model majority is supported by the au  
thoritative record.   
Sources:   
• https://nvd.nist.gov/   
vuln/detail/CVE-2024-5048   
“CVSS:3.1/AV:N/AC:L/PR:L/UI:N/S:U/C:L/I:L/A:L.”

## CASE 5 : SecEval / seceval (idx 98)

Prompt (truncated): “A web application uses GET re  
quests with query parameters to perform state-changing   
actions . . . What are the security implications?   
A. Sensitive data may be exposed in server logs   
B. GET requests can’t have a body   
C. . . . D. . . . ”   
Gold: AD   
Model majority: ABCD

Verifier: OWASP warns that query-string data can appear in logs, history, caches, and referrers, supporting A; state-changing GET requests also create additional security concerns. The published gold omits at least one supported option on this multiple-answer question.

## Sources:

https://owasp.org/www-community/   
vulnerabilities/Information\_exposure\_   
through\_query\_strings\_in\_url “The param  
eter values . . . will be exposed in: Referer Header,   
Web Logs, Shared Systems, Browser History, Browser   
Cache . . .

## G.1.3 Prompt

## G.1.4 Format-token leakage $( \mathcal { F } _ { 1 } ( \mathcal { P } ) )$

CyberMetric’s released prompt ends with the literal template Always return in this format: ‘ANSWER: X’, where X is intended as a placeholder. This token triggers systematic response failures in four models. Foundation-Sec produces an empty response on 490 of 500 questions, Primus-Nemotron on 401, Gemma-4 reproduces the placeholder on 404, and Primus-Merged continues the instruction rather than answering on 343. The released extractor silently maps these outputs to None, which the scorer counts as an incorrect answer. No rejected response contains a recoverable answer, so the failure originates in the prompt rather than extraction. Across the four affected models, the peer score gap reaches 90.9 pp, with a median of 81.3 pp. Examples 2–3 show the two dominant behaviors.

## EXAMPLE 2 : CyberMetric / Foundation-Sec (F<sub>1</sub>(P))

Prompt: “. . .Always return in thisformat: ‘ANSWER:   
X’ ” (prompt tail); gold is A   
Raw output: "" (empty; the model emits nothing at all   
on 490 of 500 questions)   
Extractor: released regex finds no ANSWER: [A-D]   
match in an empty string → None, which the scorer   
compares directly against the gold letter   
Consequence: Foundation-Sec scores 1.0% on these   
questions against a peer median of 91.9%, a 90.9 pp peer   
gap. The empty responses come from the template, not   
from a failed generation run.

## EXAMPLE 3 : CyberMetric / Gemma-4 (F<sub>1</sub>(P))

Prompt: “. . . Choose the correct answer (A, B, C, or D) only. Always return in thisformat: ‘ANSWER: X’ ” Raw output: ’ANSWER: X’ ’ANSWER: X’ ’ANSWER: X’ ’ANSWER: X’. . . (the placeholder reproduced verbatim and repeated to the token budget) Extractor: released regex re.search(r"ANSWER: ?\s\*([A-D])", resp, re.I) requires a letter in [A-D]; the literal X is not, so the match fails → None Consequence: 473 of the 500 outputs are non-empty yet return no prediction, giving 4.0% against a peer median of 91.9%. No re-extraction over these stored generations recovers an answer because the model never emits one, so the defect is in the prompt rather than the extractor. The standardized score comes from re-generating with the placeholder removed, so we report a peer-gap estimate rather than a controlled ablation.

## G.1.5 Prompt–question conflict $( \mathcal { F } _ { 2 } ( \mathcal { P } ) )$

SecEval’s benchmark-level instruction says to “select the correct answers,” but individual questions can encourage a singular response. On the 927 questions with multiple correct answers, the eight models with parseable responses produce a singleletter answer on 10.1%–33.0% of questions, with a median of 26.2% (Table 20). Set-exact-match scores these responses as incorrect even when the selected letter is part of the gold set. Example 4 illustrates the failure.

EXAMPLE 4 : SecEval / Gemma-4 (F<sub>2</sub>(P))   
Prompt: “You are evaluating the network security ofa   
mobile application. Select the controls that should be   
implemented to ensure secure communication between   
the mobile application and its backend servers.”   
Raw output: D: Ensuring backend services   
Extractor: single standalone letter → D; gold is AD   
Consequence: Set-exact-match scores 0 despite D being   
one of the two correct options. 306 of the 2,189 ques  
tions (14.0%) pair a multi-letter gold with a single-letter   
prediction; over the 927 multi-answer questions alone   
this is 33.0%.

## G.1.6 Template incompatibility $( \mathcal { F } _ { 3 } ( \mathcal { P } ) )$

On SECURE, the failure arises from a mismatch between the task template, Primus-Merged’s response style, and the reconstructed extractor. Rather than returning only the requested option, Primus-Merged frequently produces explanations or echoes the answer choices; on MAET, this occurs on 950 of 1,072 questions. The extractor then reads the first answer-choice letter it encounters, which can come from the echoed list rather than the model’s intended selection. Consequently, many responses are scored according to incidental output structure rather than task correctness, producing a 48 pp peer score gap. Example 5 shows a representative case.

![](images/2f84b55a17a8b0a57145adda28f1fc8d98bd8d820ccca378beed496e1ef2fd0b.jpg)

<table><tr><td>Model Single-letter (%)</td></tr><tr><td>Sonnet 4.6 10.8</td></tr><tr><td>Gemma-4 33.0</td></tr><tr><td>Qwen3.6 10.1</td></tr><tr><td>Llama-3.3 23.7</td></tr><tr><td>Primus-Nemotron 26.1</td></tr><tr><td>Primus-Merged 31.6</td></tr><tr><td>Foundation-Sec 26.4</td></tr><tr><td>RedSage-Qwen3 28.7</td></tr><tr><td>Median 26.2</td></tr></table>

Table 20: Single-letter response rate on SecEval’s 927 multi-answer questions for the eight models with parseable outputs on this subset.
<table><tr><td></td><td colspan="2">Accuracy (%)</td><td></td></tr><tr><td>Task</td><td>Released</td><td>Stop-free</td><td>∆ (pp)</td></tr><tr><td>cli</td><td>0.0</td><td>88.8</td><td>+88.8</td></tr><tr><td>frameworks</td><td>0.0</td><td>83.7</td><td>+83.7</td></tr><tr><td>generals</td><td>0.0</td><td>85.4</td><td>+85.4</td></tr><tr><td>kali</td><td>0.0</td><td>81.4</td><td>+81.4</td></tr><tr><td>skills</td><td>0.0</td><td>90.1</td><td>+90.1</td></tr><tr><td>Mean</td><td>0.0</td><td>85.9</td><td>+85.9</td></tr></table>

Table 21: RedSage-Bench stop-sequence ablation for Qwen3.6. The benchmark-mean recovery is 85.9 pp.

## G.2 Inference

## G.2.1 Stop-sequence mismatch $( \mathcal { F } _ { 1 } ( \mathcal { T } ) )$

RedSage-Bench’s released generative path stops on a newline. For Qwen3.6, the newline appears inside the reasoning preamble before any answer token, producing empty responses. Removing the backend stop, allowing reasoning to complete, and applying answer handling afterward raises the benchmark mean from 0.0% to 85.9%. Per-task recovery ranges from 81.4 to 90.1 pp. Example 6 shows the mechanism.

![](images/400d92c5de5b326ff4d20a6c52a479b2ff583f4c7b29f92495d77aa711fdb164.jpg)

## G.2.2 Token-budget filter $( \mathcal { F } _ { 2 } ( \mathcal { T } ) )$

SecEval specifies a five-token output budget, below the 16-token minimum accepted by the GPT-5.4 Azure OpenAI endpoint used in our evaluation. All 2,189 requests therefore return HTTP 400 rather than a model generation. The evaluator stores these error payloads as responses, allowing the answer extractor to match occasional A–D characters inside the error text and report 0.3% accuracy. Raising the budget to 16 tokens produces valid generations and raises accuracy to 81.4%, a difference of 81.1 pp. Example 7 shows the failure.

EXAMPLE 7 : SecEval / GPT-5.4 (F<sub>2</sub>(I))   
Prompt: system “Below are multiple-choice questions   
concerning cybersecurity. Please select the correct an  
swers and respond with the letters ABCD only.”; user   
“Question: Which mitigation prevent stack overflow bug?   
A: Stack Canary. B: ALSR. C: CFI. D: Code Sig. . . ”   
Raw output: ERROR: HTTP 400: {"error":   
{"message": "Invalid ‘max\_output\_tokens’:   
integer below minimum value. Expected   
a value >= 16, but got 5 instead.",   
"type": "invalid\_request\_error", "param":   
"max\_output\_tokens". . . }}   
Extractor: the evaluator stores the error string as if it   
were a response, and the A–D letter scan runs over the   
JSON payload   
Consequence: No generation ever occurred. All 2,189   
questions carry a payload of this form, so the reported   
0.3% is entirely due to accidental letter matches inside   
error text. A backend constraint is being reported as a   
capability score.

## G.2.3 Decoding drift (F<sub>3</sub>(I))

CyberMetric documents temperature 1.0, top-p 0.9, and top-k 50, while its released evaluator sets none of them and therefore inherits backend defaults. Comparing the documented configuration with a greedy control on the same 500 questions changes Primus-Merged from 17.2% to 57.2%, a 40.0 pp difference; correctness changes on 274 questions (Table 22). Example 8 illustrates the format-compliance mechanism.

EXAMPLE 8 : I3 — CyberMetric / Primus-Merged   
(F<sub>3</sub>(I))   
Prompt: identical question and prompt under both con  
figurations; only decoding differs   
Raw output: greedy (T=0, released default): (where   
X is the letter of the correct answer). Do   
not leave a space between the colon and the   
letter. Do not use quotes. . . documented   
(T=1.0, top-p=0.9): ANSWER: A You must know: To   
provide a security solution. . .   
Extractor: greedy output continues the instruction text   
and yields no answer letter (→ None); the sampled out  
put emits the required ANSWER: A form (→ A)   
Consequence: Gold is A. Correctness changes on 274 of   
the 500 questions between the two configurations, mov  
ing Primus-Merged from 17.2% to 57.2%. The differ  
ence is driven by decoding-dependent format compliance   
rather than a change in the underlying question.

<table><tr><td rowspan="2">Model</td><td colspan="2">Accuracy (%)</td><td rowspan="2">∆ (pp)</td></tr><tr><td>Greedy</td><td>Documented</td></tr><tr><td>GPT-5.4</td><td>96.0</td><td>96.2</td><td>+0.2</td></tr><tr><td>Sonnet 4.6</td><td>96.8</td><td>96.6</td><td>-0.2</td></tr><tr><td>Gemma-4</td><td>4.0</td><td>4.8</td><td>+0.8</td></tr><tr><td>Qwen3.6</td><td>92.2</td><td>91.0</td><td>-1.2</td></tr><tr><td>Llama-3.3</td><td>91.6</td><td>89.6</td><td>-2.0</td></tr><tr><td>GPT-OSS</td><td>86.4</td><td>83.4</td><td>-3.0</td></tr><tr><td>Primus-Nemotron</td><td>19.2</td><td>30.6</td><td>+11.4</td></tr><tr><td>Primus-Merged</td><td>17.2</td><td>57.2</td><td>+40.0</td></tr><tr><td>Foundation-Sec</td><td>1.0</td><td>2.8</td><td>+1.8</td></tr><tr><td>RedSage-Qwen3</td><td>89.8</td><td>86.8</td><td>-3.0</td></tr></table>

Table 22: CyberMetric decoding sensitivity on the same 500 questions. Greedy uses $T = 0 ;$ Documented uses the benchmark’s stated sampling configuration.

## G.3 Extraction

## G.3.1 Extractor divergence $( \mathcal { F } _ { 1 } ( \mathcal { E } ) )$

Semantically equivalent output can receive different scores under different extraction rules. On CTI-Bench VSP, we apply two released conventions to identical stored generations. AthenaBench’s final-line rule searches the final answer line for a CVSS:3.1/ vector, whereas CTI-Bench’s anywhere rule takes the last vector appearing anywhere in the response. Among model–question pairs for which at least one rule extracts a vector, the two disagree on 30.4%; disagreement reaches 96.5% for Primus-Merged (Table 23). Example 9 shows a representative response.

EXAMPLE 9 : CTI-Bench VSP / Primus-Merged   
(F<sub>1</sub>(E))   
Prompt: “. . .provide the final CVSS v3.1 vector string.”   
Raw output: . . . The final answer is:   
CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H   
This course unit focuses on the   
CVE-2017-1000480 vulnerability. . . (the   
correct vector, followed by unrelated text for the rest of   
the 32-line response)   
Extractor: final-line reads the last non-empty line and   
returns None; anywhere returns the last CVSS vector   
found in the response, which matches the gold vector   
Consequence: The same generation is correct under   
one extraction rule and unparseable under the other. The   
two rules disagree on 96.5% of Primus-Merged’s 513   
contested questions, even though both are plausible   
interpretations of the released pipeline.

RedSage-Bench provides another comparison within the same generation. Its released evaluator computes Exact Match (EM), Prefix Exact Match (PEM), and Regex MCQ eXtraction (RX) on the same generated responses without designating one canonical. Primus-Merged changes from 0.3% under EM to 80.0% under PEM, a 79.7 pp difference, because the model often gives the correct letter first and then continues in prose. The full per-model comparison is reported in Table 24; the 79.7 pp difference is the maximum extractor-divergence effect summarized in Table 3.

<table><tr><td></td><td colspan="2">Disagreement (%)</td></tr><tr><td>Model</td><td>All Contested only</td><td>Contested (n)</td></tr><tr><td>GPT-5.4</td><td>0.0</td><td>0.0</td></tr><tr><td>Sonnet 4.6</td><td>0.1</td><td>1,000 0.1</td></tr><tr><td>Gemma-4</td><td>0.2</td><td>999 28.6</td></tr><tr><td>Qwen3.6</td><td>42.8</td><td>52.8 810</td></tr><tr><td>Llama-3.3</td><td>42.1</td><td>44.4 948</td></tr><tr><td>GPT-OSS</td><td>46.0</td><td>60.6 759</td></tr><tr><td>Primus-Nemotron</td><td>0.3</td><td>1.3 234</td></tr><tr><td>Primus-Merged</td><td>49.5</td><td>96.5 513 964</td></tr><tr><td>Foundation-Sec</td><td>19.2</td><td>19.9 18.6</td></tr><tr><td>RedSage-Qwen3</td><td>16.9</td><td>908</td></tr><tr><td>Pooled</td><td>21.7</td><td>30.4</td></tr></table>

Table 23: Disagreement between final-line and anywhere CVSS extraction on CTI-Bench VSP. All is the percentage of all 1,000 questions per model for which the two extractors return different results. Contested only is the disagreement percentage restricted to model– question pairs for which at least one extractor returns a CVSS vector. Contested (n) is the number of such pairs. The pooled row aggregates all 10 models.
<table><tr><td rowspan="2">Model</td><td colspan="3">Accuracy (%)</td></tr><tr><td>EM</td><td>PEM</td><td>RX</td></tr><tr><td>GPT-5.4</td><td>90.7</td><td>90.7</td><td>90.2</td></tr><tr><td>Sonnet 4.6</td><td>90.5</td><td>91.3</td><td>91.0</td></tr><tr><td>Gemma-4</td><td>2.1</td><td>45.6</td><td>45.7</td></tr><tr><td>Qwen3.6</td><td>86.6</td><td>86.6</td><td>85.9</td></tr><tr><td>Llama-3.3</td><td>61.3</td><td>85.8</td><td>85.1</td></tr><tr><td>GPT-OSS</td><td>0.1</td><td>0.1</td><td>25.5</td></tr><tr><td>Primus-Nemotron</td><td>84.2</td><td>86.0</td><td>85.9</td></tr><tr><td>Primus-Merged</td><td>0.3</td><td>80.0</td><td>79.0</td></tr><tr><td>Foundation-Sec</td><td>79.1</td><td>79.1</td><td>78.2</td></tr><tr><td>RedSage-Qwen3</td><td>10.8</td><td>86.0</td><td>85.3</td></tr></table>

Table 24: Three released RedSage-Bench extraction rules applied to identical stored generations. EM is exact match, PEM is prefix exact match, and RX is regex MCQ extraction.

## G.3.2 Denominator inflation (F<sub>2</sub>(E))

Let c denote correct predictions, v parseable predictions, and n all attempted questions. Correctover-valid reports $c / v ,$ , whereas correct-over-total reports $c / n$ . On CTI-RCM, Gemma-4 has only two parseable outputs among 1,000 questions, both correct. The released convention therefore reports 100.0%, whereas correct-over-total reports 0.2%, a 99.8 pp difference and 500× inflation. Example 10 shows representative outputs. Table 25 reports all accuracy-type model–task pairs for which the two denominator policies differ by more than

<table><tr><td></td><td></td><td colspan="2">Accuracy (%)</td><td></td></tr><tr><td>Model</td><td>Task</td><td>c/v</td><td>c/n</td><td>∆ (pp)</td></tr><tr><td>Gemma-4</td><td>secure-cwet</td><td>42.9</td><td>0.6</td><td>-42.2</td></tr><tr><td>Primus-Nemotron</td><td>secure-cwet</td><td>100.0</td><td>9.4</td><td>-90.6</td></tr><tr><td>Foundation-Sec</td><td>secure-cwet</td><td>82.8</td><td>43.4</td><td>-39.4</td></tr><tr><td>Gemma-4</td><td>secure-maet</td><td>58.3</td><td>0.7</td><td>-57.7</td></tr><tr><td>Primus-Nemotron</td><td>secure-maet</td><td>91.1</td><td>8.6</td><td>-82.5</td></tr><tr><td>Foundation-Sec</td><td>secure-maet</td><td>81.5</td><td>35.0</td><td>-46.5</td></tr><tr><td>Gemma-4</td><td>secure-kcv</td><td>84.8</td><td>12.0</td><td>-72.8</td></tr><tr><td>Foundation-Sec</td><td>secure-kcv</td><td>77.8</td><td>1.5</td><td>-76.3</td></tr><tr><td>Primus-Merged</td><td>secure-kcv</td><td>58.0</td><td>43.8</td><td>-14.2</td></tr><tr><td>Gemma-4</td><td>cti-rcm</td><td>100.0</td><td>0.2</td><td>-99.8</td></tr><tr><td>Primus-Nemotron</td><td>cti-rcm</td><td>71.7</td><td>9.1</td><td>-62.6</td></tr><tr><td>Foundation-Sec</td><td>cti-rcm</td><td>69.1</td><td>63.6</td><td>-5.5</td></tr><tr><td>Gemma-4</td><td>cti-mcq</td><td>52.9</td><td>12.9</td><td>-39.9</td></tr><tr><td>Qwen3.6</td><td>cti-mcq</td><td>74.9</td><td>67.2</td><td>-7.7</td></tr><tr><td>Llama-3.3</td><td>cti-mcq</td><td>66.1</td><td>24.7</td><td>-41.4</td></tr><tr><td>GPT-OSS</td><td>cti-mcq</td><td>70.2</td><td>60.4</td><td>-9.9</td></tr><tr><td>Primus-Merged</td><td>cti-mcq</td><td>17.6</td><td>0.2</td><td>-17.4</td></tr><tr><td>Foundation-Sec</td><td>cti-mcq</td><td>32.9</td><td>27.8</td><td>-5.1</td></tr><tr><td>RedSage-Qwen3</td><td>cti-mcq</td><td>67.6</td><td>49.8</td><td>-17.8</td></tr></table>

Table 25: Accuracy for different denominator policies. Correct-over-valid $( c / v )$ and correct-over-total $( c / n )$ differ by more than 5 pp on identical stored outputs. AthenaBench VSP is excluded because its score is a normalized deviation rather than a correct-count accuracy.

5 pp. AthenaBench VSP exhibits the same mechanism: excluding unparseable CVSS vectors affects nine of 10 models, and retaining them with maximum deviation lowers scores by up to 85.4 pp, with a median decrease of 55.9 pp.

## EXAMPLE 10 : CTI-RCM / Gemma-4 (F<sub>2</sub>(E))

Prompt: “Analyze thefollowing CVE description and map it to the appropriate CWE. . . Ensure the last line of your response contains only the CWE ID. CVE Description: In the Linux kernel through 6.7.1, there is a use-after-free in cec\_queue\_msg\_fh. . . Raw output: three representative outputs: L 262594. 262594. 262594. 262594.. . . 1”1 Extractor: last-line CWE-\d+ matching returns None on all three. Across the task, the extractor matches only 2 of 1,000 outputs (CWE-79 and CWE-287), both correct; 945 outputs are non-empty but contain no CWE ID Consequence: Here c=2, v=2, and n=1000. Correctover-valid scoring reports c/v = 100.0%, while correctover-total reports $c / \bar { n } = 0 . \dot { 2 } \%$ on the same generations. The resulting 500× inflation is entirely due to the denominator convention.

## G.3.3 Metric-direction mismatch $( \mathcal { F } _ { 3 } ( \mathcal { E } ) )$

CTI-Bench and AthenaBench score the same CVSS prediction problem in opposite directions. CTI-Bench reports mean absolute deviation (MAD), for which lower is better, whereas AthenaBench reports $1 0 0 \times \mathrm { m a x } ( 0 , 1 - \mathrm { M A D } / 7 . 7 )$ , for which higher is better. Without explicitly normalizing direction, the same model performances can therefore induce different orderings. Across the 10 models, the corresponding rank displacement reaches five positions.

<table><tr><td rowspan="4">Model</td><td colspan="4">Accuracy (%)</td></tr><tr><td colspan="2">MMLU-CS</td><td colspan="2">RedSage-Bench</td></tr><tr><td>Generative</td><td>Logprob</td><td>Generative</td><td>Logprob</td></tr><tr><td>Gemma-4</td><td>59.0</td><td>66.0</td><td>45.7 86.6</td></tr><tr><td>Qwen3.6</td><td>57.0</td><td>80.0</td><td>85.9</td><td>59.2</td></tr><tr><td>Llama-3.3</td><td>78.0</td><td>82.0</td><td>85.1</td><td>84.8</td></tr><tr><td>GPT-OSS</td><td>36.0</td><td>49.0</td><td>25.5</td><td>26.7</td></tr><tr><td>Primus-Nemotron</td><td>87.0</td><td>87.0</td><td>85.9</td><td>84.1</td></tr><tr><td>Primus-Merged</td><td>66.0</td><td>83.0</td><td>79.0</td><td>73.8</td></tr><tr><td>Foundation-Sec</td><td>80.0</td><td>80.0</td><td>78.2</td><td>74.4</td></tr><tr><td>RedSage-Qwen3</td><td>81.0</td><td>85.0</td><td>85.3</td><td>84.2</td></tr></table>

Table 26: Generative vs logprob multiple-choice scoring for the eight open-weight models on two benchmarks.

## G.3.4 Prompt-mode sensitivity $( \mathcal { F } _ { 4 } ( \mathcal { E } ) )$

AthenaBench’s ATE evaluator extracts only the final answer line. Changing the prompt mode can therefore change whether a correctly identified technique appears in the extractable position. We evaluate zero-shot, four-shot, and chain-of-thought prompts on the same questions while holding the task and extractor fixed. Across the 10 models, the score spread reaches 40 pp, with a median of 7 pp. Because the prompt modes produce different generations, this is a prompt–extractor sensitivity rather than a same-generation extraction ablation.

## G.4 Aggregation

G.4.1 Logprob vs. generative scoring $( \mathcal { F } _ { 1 } ( \mathcal { A } ) )$ Multiple-choice performance can also depend on whether the evaluator ranks answer choices by token log probability or generates a textual answer and scores the resulting response (Table 26). On RedSage-Bench, Gemma-4 scores 86.6% under logprob scoring but 45.7% under generative scoring, a 40.9 pp difference. Qwen3.6 moves in the opposite direction, from 59.2% logprob to 85.9% generative. The generative results here use the stopfree configuration, so this comparison is separate from $\mathcal { F } _ { 1 } ( \mathcal { T } )$ . Hosted GPT-5.4 and Sonnet 4.6 are omitted from the logprob comparison because their evaluated APIs do not expose the required token log probabilities.

## G.4.2 Task-level metric drift $( \mathcal { F } _ { 2 } ( \mathcal { A } ) )$

Attacker-attribution tasks use materially different credit rules. The standardized harness uses strict binary grading while recognizing true semantic aliases, such as APT28 and Fancy Bear, as equivalent. CTI-Bench additionally awards plausible credit to related actors, while AthenaBench uses a strict binary verdict. Within CTI-Bench, moving from strict to correct+plausible scoring rule raises a model’s score by up to 30 pp. Across CTI-Bench and AthenaBench, these scoring conventions produce cross-task score gaps of up to 70 pp (Table 27). Example 11 illustrates how the same CTI prediction receives different credit under the two released CTI conventions.

<table><tr><td></td><td colspan="3">Accuracy (%)</td><td></td></tr><tr><td></td><td colspan="3">CTI-Bench</td><td>AthenaBench</td></tr><tr><td>Model</td><td>Strict</td><td>C+P</td><td>Binary</td><td>∆ (pp)</td></tr><tr><td>GPT-5.4</td><td>74.0</td><td>86.0</td><td>33.0</td><td>+53.0</td></tr><tr><td>Sonnet 4.6</td><td>86.0</td><td>94.0</td><td>47.0</td><td>+47.0</td></tr><tr><td>Gemma-4</td><td>20.0</td><td>26.0</td><td>7.0</td><td>+19.0</td></tr><tr><td>Qwen3.6</td><td>32.0</td><td>62.0</td><td>14.0</td><td>+48.0</td></tr><tr><td>Llama-3.3</td><td>44.0</td><td>70.0</td><td>0.0</td><td>+70.0</td></tr><tr><td>GPT-OSS</td><td>42.0</td><td>64.0</td><td>5.0</td><td>+59.0</td></tr><tr><td>Primus-Nemotron</td><td>16.0</td><td>32.0</td><td>12.0</td><td>+20.0</td></tr><tr><td>Primus-Merged</td><td>6.0</td><td>14.0</td><td>8.0</td><td>+6.0</td></tr><tr><td>Foundation-Sec</td><td>20.0</td><td>42.0</td><td>25.0</td><td>+17.0</td></tr><tr><td>RedSage-Qwen3</td><td>12.0</td><td>20.0</td><td>21.0</td><td>-1.0</td></tr></table>

Table 27: Attacker-attribution accuracy under different aggregation conventions. CTI-Bench reports Strict and Correct+Plausible (C+P) scoring, while AthenaBench uses Binary scoring. The column ∆ is the crossbenchmark score gap in pp between CTI-Bench C+P and AthenaBench binary accuracy.

## EXAMPLE 11 : CTI-TAA / Llama-3.3 (F<sub>2</sub>(A))

Prompt: “Analysis of Campaign-2. The second cam  
paign of [PLACEHOLDER] observed is spread via a   
phishing link that downloads an archivefile named ‘In  
dian Army Recruitment. . .   
Raw output: prediction extracted as transparent   
tribe; gold is sidecopy   
Extractor: identical prediction, two verdicts from   
the same released evaluator: strict exact match → in  
correct; the shipped alias/related-actor graph records   
connection = P (related actor) → credited under Cor  
rect+Plausible   
Consequence: Across the 50 questions, 22 are strictly   
correct (44.0%) and 35 are Correct+Plausible (70.0%);   
13 questions receive different credit under the two con  
ventions. Neither convention is designated canonical   
in the release, producing a 26.0 pp spread on identical   
outputs.

## G.4.3 Aggregation inconsistency $( \mathcal { F } _ { 3 } ( \mathcal { A } ) )$

Gemma-4 illustrates how nominally comparable accuracy percentages can diverge under incompatible aggregation rules. CTI-RCM reports 100.0% because only its two parseable predictions enter the denominator, whereas SecEval reports 9.5% under correct-over-total with set-exact-match scoring, an apparent 90.5 pp gap. Re-scoring CTI-RCM as correct-over-total reduces it to 0.2%, showing that denominator alignment alone can reverse the apparent comparison. As reported in Table 28, under the full standardized pipeline, which additionally aligns prompting, extraction, and scoring where semantics permit, Gemma-4 scores 70.9% on CTI-RCM and 78.3% on SecEval. The example illustrates why benchmark-level comparisons require compatible aggregation rules rather than percentages alone.

## H Cross-Benchmark Analysis

This section provides additional analyses of how the audited tasks relate to one another and whether they support consistent model comparisons. We examine effective dimensionality, task-level rank agreement and pairwise order reversals, and the relationship between semantic overlap and ranking similarity.

## H.1 Effective Dimensionality

We column-standardize the 23 × 10 strict-verdict task-score matrix and apply PCA. The first principal component explains 95.25% of the variance. Horn’s parallel analysis with 500 random matrices at the 95th percentile finds that only this component exceeds the corresponding null eigenvalue, indicating that most score variation lies along a single dominant axis.

## H.2 Rank Agreement

For task-to-task comparisons, we measure rank agreement with Kendall’s τ-b, which accounts for ties. With only 10 models, small values may not be distinguishable from zero, so we interpret the coefficients as measures of correspondence rather than precise population estimates. For original-versusstandardized benchmark rankings in App. I.4, we use Spearman’s $\rho .$

## H.3 Pairwise Rank Changes

A pairwise order reversal occurs when two models are ordered one way by one task and the opposite way by another. We enumerate these reversals over all ${ \binom { 1 0 } { 2 } } = 4 5$ model pairs for each task pair and release the full catalog with the evaluation artifacts. The clearest examples arise between semantically similar tasks with different evaluation conventions. CTI-Bench and AthenaBench vulnerability scoring evaluate the same CVSS quantity but use opposite metric directions (App. G.3.3); their model rankings agree only weakly under Kendall’s τ-b (0.29), with rank displacement reaching five positions. Their attacker-attribution tasks likewise show weak agreement $( \tau { - } \mathbf { b } = 0 . 2 4 )$

## H.4 Semantic Overlap

We embed benchmark questions after removing boilerplate, represent each task by the centroid of its question embeddings, and compute pairwise task-content similarity using cosine similarity. We compare this matrix with a task-ranking agreement matrix based on pairwise Kendall’s τ using a Mantel permutation test. Across the ${ \binom { 2 3 } { 2 } } = 2 5 3$ task pairs, content similarity and ranking agreement are moderately associated $( r = 0 . 5 1 6 , p < 0 . 0 0 2 )$ with a squared correlation of $r ^ { 2 } \approx 0 . 2 7$ . Thus, semantic similarity explains only part of the variation in task-induced rankings. Because each $\tau$ is estimated from rankings over only 10 models, sampling variability may attenuate the observed association, so we avoid interpreting $r ^ { 2 }$ as a precise fraction of variance explained. Independently, PCA of the task-score matrix recovers a single dominant component (App. H.1), consistent with a shared model-strength axis producing similar score patterns across tasks with substantially different content. Together, these results suggest that the strong score-level redundancy across tasks cannot be attributed primarily to duplicated or semantically overlapping questions.

## I Standardization and Rank Stability

This section records the benchmark-specific standardization choices, the resulting score changes, and the stability of the observed rank shifts. It also separates changes attributable to generation from those attributable to extraction.

## I.1 Standardized Components

The pipeline ledger (Table 13) records the action taken for every benchmark-field pair. Retain preserves released behavior, whereas Fix, Normalize, and Supply identify fields changed or supplied by the standardized pipeline. Read together with Table 10, the two tables distinguish what the benchmark specifies from what the standardized evaluation executes.

## I.2 Original vs. Standardized Scores

Table 28 reports the per-model task scores before and after pipeline standardization. CyberMetric det/samp and MMLU-CS gen/logp are alternative configurations of one task each, not additional tasks, so the inventory in Table 8 still contains 23 tasks. MMLU-CS logprob scoring is undefined for GPT-5.4 and Sonnet 4.6 because the evaluated hosted APIs do not expose the required token log probabilities.

The symbol <sup>†</sup> marks an original score dominated by a denominator artifact: the released evaluator scores only parseable predictions, and the valid set is a small fraction of the attempted questions. GPT-5.4’s original SecEval score requires a different caution. The released five-token budget is rejected by the backend, so the resulting 0.3% reflects accidental extraction from API error payloads rather than model capability (App. G.2.2). Finally, standardized scores use the pinned extraction and grading policy of App. E.1, so an original-to-standardized difference can reflect changes in prompting, inference, extraction, scoring, or aggregation as specified in the pipeline ledger.

## I.3 Tie Breaking

When two models have identical standardized scores on a benchmark, we break the tie by stable sorting in descending score, preserving a fixed model order. The same rule is applied to original and standardized rankings, making every rank difference in Table 5 deterministic. Exact ties are rare at the reported precision, so this rule affects only tied cells and does not change the reported count of large rank shifts.

## I.4 Bootstrap Stability

The reported standardized scores are point estimates from one stored evaluation run. We quantify sampling variability with a paired question-level bootstrap that requires no new generation. Within each task, we resample stored question outcomes with replacement for 5,000 replicates, applying the same resample to every model so that comparisons retain a shared question basis. For each replicate, we recompute the benchmark score, re-rank the models, and calculate Spearman’s $\rho$ against the fixed original ranking.

The observed samples reproduce the $\rho$ values in Table 5 to two decimal places. Every 95% bootstrap interval in Table 29 excludes 1, indicating that the observed reorderings are not explained by question sampling alone. MMLU-CS remains negatively correlated with its original ranking, while SecEval’s interval includes zero, indicating an almost complete reshuffling in both cases.

The individual large shifts are also stable. Each of the 35 benchmark-level movements of at least three positions retains its direction in at least 97.7% of bootstrap replicates; 33 do so in at least 98%, and 24 retain their direction in all 5,000 replicates. The two least stable cases are three-position drops on MMLU-CS. Its larger shifts are substantially more stable: Gemma-4 moves +6 positions with a bootstrap interval of $[ + 3 , + 7 ]$ , Primus-Nemotron moves −7 with [−9, −5], and Foundation-Sec moves −6 with [−7, −3].

Score uncertainty is smaller than the resulting ranking changes but still limits fine-grained comparisons. Bootstrap half-widths range from approximately ±0.4 pp on RedSage-Bench to $\pm 7 . 5$ pp on MMLU-CS, and adjacent models often have overlapping intervals. On SecBench, for example, Sonnet 4.6 scores 89.7 [87.4, 92.0], Qwen3.6 scores 89.1 [86.7, 91.4], and GPT-5.4 scores 87.7 [85.3, 90.2]. Thus, the effect of pipeline standardization on the rankings is robust to resampling even when the precise ordering of nearby models is not statistically resolvable.

## I.5 Generation vs. Extraction

The original-to-standardized comparison changes both how responses are generated and how they are interpreted. We extend the pipeline notation of §2 to separate these effects. Let

$$
\mathcal { G } _ { b } ^ { x } ( m ) = \mathcal { I } _ { b } ^ { x } ( m , \mathcal { P } _ { b } ^ { x } ( \mathcal { D } _ { b } ) ) , x \in \{ o , s \} ,\tag{3}
$$

denote the generations produced under the original (o) or standardized (s) prompt and inference configuration. We further distinguish three extractionand-scoring rules: the original benchmark rule $\mathcal { E } _ { b } ^ { o }$ , a deterministic final-answer regex $\mathcal { E } _ { b } ^ { r }$ , and the pinned judge $\mathcal { E } _ { b } ^ { j }$ . The resulting pipeline score is

$$
\begin{array} { r } { S _ { b } ^ { x , y } ( m ) = A _ { b } ^ { y } \big ( \mathcal { E } _ { b } ^ { y } ( \mathcal { G } _ { b } ^ { x } ( m ) ) \big ) , y \in \{ o , r , j \} , } \end{array}\tag{4}
$$

where $\mathcal { A } _ { b } ^ { y }$ denotes the aggregation associated with that evaluation rule. The four comparison cases are: $\mathcal { S } _ { b } ^ { o , o }$ , the original benchmark score; $S _ { b } ^ { o , r }$ , regex evaluation of the original generations; $S _ { b } ^ { s , r }$ , the same regex evaluation of the standardized generations; and $ { \mathcal { S } } _ { b } ^ { s , j }$ , judge evaluation of the standardized generations.

<table><tr><td rowspan=1 colspan=20>GPT-5.4*     Sonnet 4.6*     Gemma-4      Qwen3.6      Llama-3.3      GPT-OSS  Primus-NemotronPrimus-Merged Foundation-SecRedSage-Qwen3Benchmark     Task     Metric      Before After Before After Before After Before After Before After Before After Before After Before After Before After Before After</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>MMLU-CS     gen      Accuracy     74.0</td><td rowspan=1 colspan=1>87.0</td><td rowspan=1 colspan=1>70.0</td><td rowspan=1 colspan=1>90.0</td><td rowspan=1 colspan=1>59.0</td><td rowspan=1 colspan=1>90.0</td><td rowspan=1 colspan=1>57.0</td><td rowspan=1 colspan=1>88.0</td><td rowspan=1 colspan=1>78.0</td><td rowspan=1 colspan=1>81.0</td><td rowspan=1 colspan=1>36.0</td><td rowspan=1 colspan=1>86.0</td><td rowspan=1 colspan=1>87.0</td><td rowspan=1 colspan=1>76.0</td><td rowspan=1 colspan=1>66.0</td><td rowspan=1 colspan=1>74.0</td><td rowspan=1 colspan=1>80.0</td><td rowspan=1 colspan=1>76.0</td><td rowspan=1 colspan=1>81.0</td><td rowspan=1 colspan=1>82.0</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1>logp     Accuracy      1</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>=</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>66.0</td><td rowspan=1 colspan=1>90.0</td><td rowspan=1 colspan=1>80.0</td><td rowspan=1 colspan=1>88.0</td><td rowspan=1 colspan=1>82.0</td><td rowspan=1 colspan=1>81.0</td><td rowspan=1 colspan=1>49.0</td><td rowspan=1 colspan=1>86.0</td><td rowspan=1 colspan=1>87.0</td><td rowspan=1 colspan=1>76.0</td><td rowspan=1 colspan=1>83.0</td><td rowspan=1 colspan=1>74.0</td><td rowspan=1 colspan=1>80.0</td><td rowspan=1 colspan=1>76.0</td><td rowspan=1 colspan=1>85.0</td><td rowspan=1 colspan=1>82.0</td></tr><tr><td rowspan=1 colspan=1>SecEval        seceval  Accuracy     0.3</td><td rowspan=1 colspan=2>82.0  78.0</td><td rowspan=1 colspan=1>71.8</td><td rowspan=1 colspan=1>9.5</td><td rowspan=1 colspan=1>78.3</td><td rowspan=1 colspan=1>54.3</td><td rowspan=1 colspan=1>74.2</td><td rowspan=1 colspan=1>71.0</td><td rowspan=1 colspan=4>69.7  0.4  72.6  61.0</td><td rowspan=1 colspan=1>72.8</td><td rowspan=1 colspan=1>12.6</td><td rowspan=1 colspan=1>61.7</td><td rowspan=1 colspan=1>58.5</td><td rowspan=1 colspan=1>57.0</td><td rowspan=1 colspan=1>47.2</td><td rowspan=1 colspan=1>74.2</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>SECURE      maet     Accuracy     93.3</td><td rowspan=1 colspan=1>93.1</td><td rowspan=1 colspan=1>94.2</td><td rowspan=1 colspan=1>94.1</td><td rowspan=1 colspan=1>58.3</td><td rowspan=1 colspan=1>92.3</td><td rowspan=1 colspan=1>88.8</td><td rowspan=1 colspan=1>90.2</td><td rowspan=1 colspan=1>85.2</td><td rowspan=1 colspan=1>86.7</td><td rowspan=1 colspan=1>71.4</td><td rowspan=1 colspan=1>87.3</td><td rowspan=1 colspan=1>91.1†</td><td rowspan=1 colspan=1>91.0</td><td rowspan=1 colspan=1>11.4</td><td rowspan=1 colspan=1>78.5</td><td rowspan=1 colspan=1>81.5</td><td rowspan=1 colspan=1>84.6</td><td rowspan=1 colspan=1>59.0</td><td rowspan=1 colspan=1>89.6</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1>cwet     Accuracy     94.3</td><td rowspan=1 colspan=1>95.0</td><td rowspan=1 colspan=1>94.6</td><td rowspan=1 colspan=1>95.0</td><td rowspan=1 colspan=1>42.9</td><td rowspan=1 colspan=1>92.3</td><td rowspan=1 colspan=1>89.3</td><td rowspan=1 colspan=1>92.0</td><td rowspan=1 colspan=1>87.7</td><td rowspan=1 colspan=1>90.0</td><td rowspan=1 colspan=1>71.6</td><td rowspan=1 colspan=1>88.9</td><td rowspan=1 colspan=1>100.0†</td><td rowspan=1 colspan=1>94.1</td><td rowspan=1 colspan=1>8.0</td><td rowspan=1 colspan=1>78.4</td><td rowspan=1 colspan=1>82.8</td><td rowspan=1 colspan=1>83.4</td><td rowspan=1 colspan=1>60.1</td><td rowspan=1 colspan=1>91.5</td></tr><tr><td rowspan=1 colspan=1>kcv      Accuracy     87.8</td><td rowspan=1 colspan=1>88.4</td><td rowspan=1 colspan=1>88.4</td><td rowspan=1 colspan=1>88.4</td><td rowspan=1 colspan=1>84.8</td><td rowspan=1 colspan=1>84.8</td><td rowspan=1 colspan=1>88.4</td><td rowspan=1 colspan=1>89.5</td><td rowspan=1 colspan=1>81.5</td><td rowspan=1 colspan=1>87.5</td><td rowspan=1 colspan=1>81.9</td><td rowspan=1 colspan=1>88.2</td><td rowspan=1 colspan=1>0.0†</td><td rowspan=1 colspan=1>86.9</td><td rowspan=1 colspan=1>58.0</td><td rowspan=1 colspan=1>82.2</td><td rowspan=1 colspan=1>77.8</td><td rowspan=1 colspan=1>81.1</td><td rowspan=1 colspan=1>67.2</td><td rowspan=1 colspan=1>81.5</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>CTI-Bench     mcq      Accuracy     78.3</td><td rowspan=1 colspan=1>78.5</td><td rowspan=1 colspan=1>83.6</td><td rowspan=1 colspan=1>84.1</td><td rowspan=1 colspan=1>52.9</td><td rowspan=1 colspan=1>74.4</td><td rowspan=1 colspan=1>74.9</td><td rowspan=1 colspan=1>73.0</td><td rowspan=1 colspan=1>66.1</td><td rowspan=1 colspan=1>65.6</td><td rowspan=1 colspan=1>70.2</td><td rowspan=1 colspan=1>69.3</td><td rowspan=1 colspan=1>71.6</td><td rowspan=1 colspan=1>69.2</td><td rowspan=1 colspan=1>17.6</td><td rowspan=1 colspan=1>45.0</td><td rowspan=1 colspan=1>32.9</td><td rowspan=1 colspan=1>56.4</td><td rowspan=1 colspan=1>67.6</td><td rowspan=1 colspan=1>65.2</td><td rowspan=5 colspan=1></td></tr><tr><td rowspan=1 colspan=1>rcm      Accuracy     74.0</td><td rowspan=1 colspan=1>74.2</td><td rowspan=1 colspan=1>75.5</td><td rowspan=1 colspan=1>75.4</td><td rowspan=1 colspan=1>100.0†</td><td rowspan=1 colspan=1>70.9</td><td rowspan=1 colspan=1>68.0</td><td rowspan=1 colspan=1>71.8</td><td rowspan=1 colspan=1>66.1</td><td rowspan=1 colspan=1>63.1</td><td rowspan=1 colspan=1>66.2</td><td rowspan=1 colspan=1>65.9</td><td rowspan=1 colspan=1>71.7</td><td rowspan=1 colspan=1>65.3</td><td rowspan=1 colspan=1>64.2</td><td rowspan=1 colspan=1>67.5</td><td rowspan=1 colspan=1>69.1</td><td rowspan=1 colspan=1>69.4</td><td rowspan=1 colspan=1>78.0</td><td rowspan=1 colspan=1>75.7</td></tr><tr><td rowspan=1 colspan=1>vsp      MAD↓       0.98</td><td rowspan=1 colspan=1>1.02</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>1.38</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>1.38</td><td rowspan=1 colspan=1>1.34</td><td rowspan=1 colspan=1>1.59</td><td rowspan=1 colspan=1>1.49</td><td rowspan=1 colspan=1>1.27</td><td rowspan=1 colspan=1>1.95</td><td rowspan=1 colspan=1>1.43</td><td rowspan=1 colspan=1>1.34</td><td rowspan=1 colspan=1>1.67</td><td rowspan=1 colspan=1>1.91</td><td rowspan=1 colspan=1>1.56</td><td rowspan=1 colspan=1>1.39</td><td rowspan=1 colspan=1>1.15</td><td rowspan=1 colspan=1>1.42</td></tr><tr><td rowspan=1 colspan=1>ate      Accuracy     5.0</td><td rowspan=1 colspan=1>5.0</td><td rowspan=1 colspan=1>8.3</td><td rowspan=1 colspan=1>6.7</td><td rowspan=1 colspan=1>5.0</td><td rowspan=1 colspan=1>8.3</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>3.3</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>1.7</td><td rowspan=1 colspan=1>3.3</td><td rowspan=1 colspan=1>3.3</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>1.7</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>1.7</td><td rowspan=1 colspan=1>1.7</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td></tr><tr><td rowspan=1 colspan=1>taa      Accuracy     86.0</td><td rowspan=1 colspan=1>70.0</td><td rowspan=1 colspan=1>94.0</td><td rowspan=1 colspan=1>82.0</td><td rowspan=1 colspan=1>26.0</td><td rowspan=1 colspan=1>50.0</td><td rowspan=1 colspan=1>62.0</td><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1>70.0</td><td rowspan=1 colspan=1>38.0</td><td rowspan=1 colspan=1>64.0</td><td rowspan=1 colspan=1>26.0</td><td rowspan=1 colspan=1>32.0</td><td rowspan=1 colspan=1>40.0</td><td rowspan=1 colspan=1>14.0</td><td rowspan=1 colspan=1>14.0</td><td rowspan=1 colspan=1>42.0</td><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1>30.0</td></tr><tr><td rowspan=1 colspan=1>AthenaBench   ckt      Accuracy     91.3</td><td rowspan=1 colspan=1>91.0</td><td rowspan=1 colspan=1>92.5</td><td rowspan=1 colspan=1>92.8</td><td rowspan=1 colspan=1>8.4</td><td rowspan=1 colspan=1>86.1</td><td rowspan=1 colspan=1>79.4</td><td rowspan=1 colspan=1>84.9</td><td rowspan=1 colspan=1>70.5</td><td rowspan=1 colspan=1>82.2</td><td rowspan=1 colspan=1>69.7</td><td rowspan=1 colspan=1>81.1</td><td rowspan=1 colspan=1>66.6</td><td rowspan=1 colspan=1>81.3</td><td rowspan=1 colspan=1>19.5</td><td rowspan=1 colspan=1>76.6</td><td rowspan=1 colspan=1>68.9</td><td rowspan=1 colspan=1>78.5</td><td rowspan=1 colspan=1>80.9</td><td rowspan=2 colspan=1>79.223.7</td><td rowspan=3 colspan=1></td></tr><tr><td rowspan=1 colspan=1>rms      Set-F1       40.7</td><td rowspan=1 colspan=1>41.4</td><td rowspan=1 colspan=1>60.1</td><td rowspan=1 colspan=1>59.6</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>21.5</td><td rowspan=1 colspan=1>5.0</td><td rowspan=1 colspan=1>3.4</td><td rowspan=1 colspan=1>3.5</td><td rowspan=1 colspan=1>10.9</td><td rowspan=1 colspan=1>2.5</td><td rowspan=1 colspan=1>2.3</td><td rowspan=1 colspan=1>4.1</td><td rowspan=1 colspan=1>14.9</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>8.4</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>24.8</td><td rowspan=1 colspan=1>15.6</td></tr><tr><td rowspan=1 colspan=1>taa      Accuracy     33.0</td><td rowspan=1 colspan=1>30.0</td><td rowspan=1 colspan=1>47.0</td><td rowspan=1 colspan=1>42.0</td><td rowspan=1 colspan=1>7.0</td><td rowspan=1 colspan=1>22.0</td><td rowspan=1 colspan=1>14.0</td><td rowspan=1 colspan=1>16.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>18.0</td><td rowspan=1 colspan=1>5.0</td><td rowspan=1 colspan=1>11.0</td><td rowspan=1 colspan=1>12.0</td><td rowspan=1 colspan=1>19.0</td><td rowspan=1 colspan=1>8.0</td><td rowspan=1 colspan=1>19.0</td><td rowspan=1 colspan=1>25.0</td><td rowspan=1 colspan=1>21.0</td><td rowspan=1 colspan=1>21.0</td><td rowspan=1 colspan=1>19.0</td></tr><tr><td rowspan=1 colspan=1>ate      Accuracy     68.6</td><td rowspan=1 colspan=1>66.4</td><td rowspan=1 colspan=1>82.2</td><td rowspan=1 colspan=1>79.2</td><td rowspan=1 colspan=1>2.4</td><td rowspan=1 colspan=1>49.4</td><td rowspan=1 colspan=1>40.6</td><td rowspan=1 colspan=1>49.6</td><td rowspan=1 colspan=1>27.0</td><td rowspan=1 colspan=1>29.4</td><td rowspan=1 colspan=1>24.2</td><td rowspan=1 colspan=1>26.8</td><td rowspan=1 colspan=1>39.0</td><td rowspan=1 colspan=1>53.2</td><td rowspan=1 colspan=1>27.2</td><td rowspan=1 colspan=1>33.6</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>38.0</td><td rowspan=1 colspan=1>51.8</td><td rowspan=1 colspan=1>51.0</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>rcm      Accuracy     71.0</td><td rowspan=1 colspan=1>71.5</td><td rowspan=1 colspan=1>73.7</td><td rowspan=1 colspan=1>73.6</td><td rowspan=1 colspan=1>1.9</td><td rowspan=1 colspan=1>64.8</td><td rowspan=1 colspan=1>59.9</td><td rowspan=1 colspan=1>65.7</td><td rowspan=1 colspan=1>51.0</td><td rowspan=1 colspan=1>61.0</td><td rowspan=1 colspan=1>57.3</td><td rowspan=1 colspan=1>58.2</td><td rowspan=1 colspan=1>21.2</td><td rowspan=1 colspan=1>57.8</td><td rowspan=1 colspan=1>38.8</td><td rowspan=1 colspan=1>55.7</td><td rowspan=1 colspan=1>9.2</td><td rowspan=1 colspan=1>60.5</td><td rowspan=1 colspan=1>68.4</td><td rowspan=1 colspan=1>68.3</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1>vsp      MAD-norm   85.8</td><td rowspan=1 colspan=1>85.7</td><td rowspan=1 colspan=1>88.6</td><td rowspan=1 colspan=1>88.7</td><td rowspan=1 colspan=1>85.4</td><td rowspan=1 colspan=1>87.8</td><td rowspan=1 colspan=1>83.1</td><td rowspan=1 colspan=1>58.9</td><td rowspan=1 colspan=1>74.4</td><td rowspan=1 colspan=1>71.6</td><td rowspan=1 colspan=1>78.8</td><td rowspan=1 colspan=1>75.7</td><td rowspan=1 colspan=1>73.6</td><td rowspan=1 colspan=1>74.6</td><td rowspan=1 colspan=1>53.3</td><td rowspan=1 colspan=1>72.3</td><td rowspan=1 colspan=1>70.2</td><td rowspan=1 colspan=1>65.7</td><td rowspan=1 colspan=1>73.5</td><td rowspan=1 colspan=1>72.0</td></tr><tr><td rowspan=1 colspan=1>CyberMetric    det      Accuracy     96.0</td><td rowspan=1 colspan=1>96.0</td><td rowspan=1 colspan=1>96.8</td><td rowspan=1 colspan=1>96.2</td><td rowspan=1 colspan=1>4.0</td><td rowspan=1 colspan=1>95.2</td><td rowspan=1 colspan=1>92.2</td><td rowspan=1 colspan=1>95.6</td><td rowspan=1 colspan=1>91.6</td><td rowspan=1 colspan=1>93.0</td><td rowspan=1 colspan=1>86.4</td><td rowspan=1 colspan=1>92.0</td><td rowspan=1 colspan=1>19.2</td><td rowspan=1 colspan=1>93.4</td><td rowspan=1 colspan=1>17.2</td><td rowspan=1 colspan=1>86.0</td><td rowspan=2 colspan=1>1.02.8</td><td rowspan=2 colspan=1>85.285.2</td><td rowspan=2 colspan=1>89.886.8</td><td rowspan=2 colspan=1>90.290.2</td><td></td></tr><tr><td rowspan=1 colspan=1>samp     Accuracy     96.2</td><td rowspan=1 colspan=1>96.0</td><td rowspan=1 colspan=1>96.6</td><td rowspan=1 colspan=1>96.2</td><td rowspan=1 colspan=1>4.8</td><td rowspan=1 colspan=1>95.2</td><td rowspan=1 colspan=1>91.0</td><td rowspan=1 colspan=1>95.6</td><td rowspan=1 colspan=1>89.6</td><td rowspan=1 colspan=1>93.0</td><td rowspan=1 colspan=1>83.4</td><td rowspan=1 colspan=1>92.0</td><td rowspan=1 colspan=1>30.6</td><td rowspan=1 colspan=1>93.4</td><td rowspan=1 colspan=1>57.2</td><td rowspan=1 colspan=1>86.0</td><td></td></tr><tr><td rowspan=1 colspan=1>RedSage-Bench  cli      Accuracy     93.0</td><td rowspan=1 colspan=1>92.7</td><td rowspan=1 colspan=1>93.7</td><td rowspan=1 colspan=1>93.6</td><td rowspan=1 colspan=1>43.5</td><td rowspan=1 colspan=1>88.8</td><td rowspan=1 colspan=1>88.8</td><td rowspan=1 colspan=1>90.1</td><td rowspan=1 colspan=1>87.3</td><td rowspan=1 colspan=1>86.9</td><td rowspan=1 colspan=1>24.5</td><td rowspan=1 colspan=1>89.7</td><td rowspan=1 colspan=1>87.9</td><td rowspan=1 colspan=1>87.1</td><td rowspan=1 colspan=1>78.5</td><td rowspan=1 colspan=1>75.9</td><td rowspan=1 colspan=1>78.3</td><td rowspan=1 colspan=1>75.2</td><td rowspan=1 colspan=1>86.7</td><td rowspan=1 colspan=1>86.6</td><td></td></tr><tr><td rowspan=1 colspan=1>frameworksAccuracy     87.9</td><td rowspan=1 colspan=1>88.1</td><td rowspan=1 colspan=1>90.0</td><td rowspan=1 colspan=1>90.2</td><td rowspan=1 colspan=1>46.4</td><td rowspan=1 colspan=1>86.2</td><td rowspan=1 colspan=1>83.7</td><td rowspan=1 colspan=1>84.6</td><td rowspan=1 colspan=1>83.2</td><td rowspan=1 colspan=1>83.0</td><td rowspan=1 colspan=1>24.6</td><td rowspan=1 colspan=1>81.1</td><td rowspan=1 colspan=1>84.8</td><td rowspan=1 colspan=1>84.6</td><td rowspan=1 colspan=1>78.9</td><td rowspan=1 colspan=1>73.4</td><td rowspan=1 colspan=1>79.6</td><td rowspan=1 colspan=1>77.4</td><td rowspan=1 colspan=1>85.2</td><td rowspan=1 colspan=1>84.4</td><td></td></tr><tr><td rowspan=1 colspan=1>generals Accuracy     90.6</td><td rowspan=1 colspan=1>90.2</td><td rowspan=1 colspan=1>90.4</td><td rowspan=1 colspan=1>90.7</td><td rowspan=1 colspan=1>47.1</td><td rowspan=1 colspan=1>87.1</td><td rowspan=1 colspan=1>85.4</td><td rowspan=1 colspan=1>86.7</td><td rowspan=1 colspan=1>85.8</td><td rowspan=1 colspan=1>85.8</td><td rowspan=1 colspan=1>26.5</td><td rowspan=1 colspan=1>82.1</td><td rowspan=1 colspan=1>86.6</td><td rowspan=1 colspan=1>86.8</td><td rowspan=1 colspan=1>78.6</td><td rowspan=1 colspan=1>74.9</td><td rowspan=1 colspan=1>77.7</td><td rowspan=1 colspan=1>74.5</td><td rowspan=1 colspan=1>84.1</td><td rowspan=1 colspan=1>83.7</td><td></td></tr><tr><td rowspan=1 colspan=1>kali     Accuracy     86.3</td><td rowspan=1 colspan=1>86.4</td><td rowspan=1 colspan=1>87.3</td><td rowspan=1 colspan=1>87.2</td><td rowspan=1 colspan=1>39.6</td><td rowspan=1 colspan=1>81.2</td><td rowspan=1 colspan=1>81.4</td><td rowspan=1 colspan=1>83.3</td><td rowspan=1 colspan=1>79.5</td><td rowspan=1 colspan=1>80.3</td><td rowspan=1 colspan=1>25.3</td><td rowspan=1 colspan=1>80.5</td><td rowspan=1 colspan=1>80.2</td><td rowspan=1 colspan=1>80.3</td><td rowspan=2 colspan=1>74.084.9</td><td rowspan=2 colspan=1>70.480.7</td><td rowspan=2 colspan=1>71.983.5</td><td rowspan=2 colspan=1>68.781.1</td><td rowspan=2 colspan=1>80.889.5</td><td rowspan=2 colspan=1>80.588.8</td><td></td></tr><tr><td rowspan=1 colspan=1>skills   Accuracy     93.2</td><td rowspan=1 colspan=1>93.2</td><td rowspan=1 colspan=1>93.3</td><td rowspan=1 colspan=1>93.6</td><td rowspan=1 colspan=1>51.8</td><td rowspan=1 colspan=1>91.4</td><td rowspan=1 colspan=1>90.1</td><td rowspan=1 colspan=1>91.6</td><td rowspan=1 colspan=1>89.7</td><td rowspan=1 colspan=1>89.4</td><td rowspan=1 colspan=1>26.4</td><td rowspan=1 colspan=1>90.2</td><td rowspan=1 colspan=1>90.1</td><td rowspan=1 colspan=1>89.5</td><td></td></tr><tr><td rowspan=1 colspan=1>SecBench       secbench Accuracy     84.6</td><td rowspan=1 colspan=1>87.7</td><td rowspan=1 colspan=1>85.2</td><td rowspan=1 colspan=1>89.7</td><td rowspan=1 colspan=1>33.9</td><td rowspan=1 colspan=1>86.7</td><td rowspan=1 colspan=1>62.3</td><td rowspan=1 colspan=1>89.1</td><td rowspan=1 colspan=1>71.4</td><td rowspan=1 colspan=1>82.1</td><td rowspan=1 colspan=2>41.1  81.6</td><td rowspan=1 colspan=1>69.6</td><td rowspan=1 colspan=1>84.8</td><td rowspan=1 colspan=1>38.1</td><td rowspan=1 colspan=1>66.3</td><td rowspan=1 colspan=1>67.0</td><td rowspan=1 colspan=1>70.9</td><td rowspan=1 colspan=1>44.3</td><td rowspan=1 colspan=1>80.2</td><td></td></tr><tr><td rowspan=2 colspan=21>Table 28: Per-task scores before and after pipeline standardization for all 10 models. Gray columns show standardized scores. The Metric column identifies the reportedquantity: accuracy (%), question-level Set-F1 (%), mean absolute CVSS-score deviation (MAD, lower is better), or normalized MAD (%). CTI-Bench taa uses Correct+Plausibleaccuracy before standardization and strict binary accuracy after standardization. CyberMetric det/samp and MMLU-CS gen/logp are alternative configurations of one task each.hosted/API model; † denominator artifact.</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=12></td></tr></table>

<table><tr><td rowspan="2">Benchmark</td><td colspan="3">Spearman ρ</td></tr><tr><td>Lower</td><td>Estimate</td><td>Upper</td></tr><tr><td>MMLU-CS</td><td>-0.71</td><td>-0.47</td><td>-0.25</td></tr><tr><td>SecEval</td><td>-0.08</td><td>+0.03</td><td>+0.15</td></tr><tr><td>SECURE</td><td>+0.56</td><td>+0.65</td><td>+0.72</td></tr><tr><td>CTI-Bench</td><td>+0.56</td><td>+0.64</td><td>+0.76</td></tr><tr><td>AthenaBench</td><td>+0.39</td><td>+0.42</td><td>+0.47</td></tr><tr><td>CyberMetric</td><td>+0.47</td><td>+0.73</td><td>+0.81</td></tr><tr><td>RedSage-Bench</td><td>+0.60</td><td>+0.71</td><td>+0.72</td></tr><tr><td>SecBench</td><td>+0.32</td><td>+0.50</td><td>+0.58</td></tr></table>

Table 29: Spearman correlation between original and standardized model rankings with question-level bootstrap uncertainty. Estimate is the observed correlation; Lower and Upper are the endpoints of the 95% confidence interval from 5,000 paired bootstrap resamples. Every interval excludes 1; only SecEval includes 0.

For benchmark $b ,$ let $\mathbf { S } _ { b } ^ { x , y }$ denote the vector of $S _ { b } ^ { x , y } ( m )$ over the 10 models. We measure the generation contribution by

$$
\rho _ { \mathrm { g e n } } = \rho \big ( \mathbf { S } _ { b } ^ { o , r } , \mathbf { S } _ { b } ^ { s , r } \big ) ,\tag{5}
$$

which changes $\mathcal { P } _ { b }$ and $\mathcal { T } _ { b }$ while holding the regex evaluation fixed. We measure the extraction contribution by

$$
\rho _ { \mathrm { e x t } } = \rho \Big ( \mathbf { S } _ { b } ^ { s , r } , \mathbf { S } _ { b } ^ { s , j } \Big ) ,\tag{6}
$$

which holds standardized generations fixed while changing their interpretation. Finally,

$$
\rho _ { \mathrm { o r i g } } = \rho ( \mathbf { S } _ { b } ^ { o , r } , \mathbf { S } _ { b } ^ { o , o } )\tag{7}
$$

checks how closely the deterministic regex reconstruction preserves the original benchmark ranking. Lower $\rho$ indicates greater ranking change.

The comparison includes tasks that admit deterministic extraction of single- or multi-letter answers, true-or-false responses, CWE and ATT&CK identifiers, or CVSS vectors. The two attackerattribution tasks are excluded because semantic alias handling cannot be represented reliably by a generic regex. The reconstruction check is strong on seven benchmarks, where $\rho _ { \mathrm { o r i g } }$ ranges from 0.78 to 0.95. CTI-Bench is lower at 0.53 because its CWE and CVSS extraction rules diverge more substantially from the generic regex.

Table 30 shows that generation changes produce more ranking reordering than extraction on five of the eight benchmarks: SecEval, CTI-Bench, AthenaBench, RedSage-Bench, and SecBench. The contrast is strongest on AthenaBench, where $\rho _ { \mathrm { g e n } }$ 0.34 and $\rho _ { \mathrm { e x t } } = 0 . 9 9$ , and CTI-Bench, where they are 0.59 and 0.89. Extraction contributes more on

<table><tr><td>Benchmark</td><td> $\pmb { \rho } _ { \mathbf { g e n } }$ </td><td>ρext</td><td>ρorig</td></tr><tr><td>MMLU-CS</td><td>+0.25</td><td>+0.20</td><td>+0.91</td></tr><tr><td>SecEval</td><td>-0.19</td><td>+0.27</td><td>+0.82</td></tr><tr><td>SECURE</td><td>+0.80</td><td>+0.58</td><td>+0.78</td></tr><tr><td>CTI-Bench</td><td>+0.59</td><td>+0.89</td><td>+0.53</td></tr><tr><td>AthenaBench</td><td>+0.34</td><td>+0.99</td><td>+0.91</td></tr><tr><td>CyberMetric</td><td>+0.93</td><td>+0.82</td><td>+0.94</td></tr><tr><td>RedSage-Bench</td><td>+0.55</td><td>+0.72</td><td>+0.84</td></tr><tr><td>SecBench</td><td>+0.38</td><td>+0.61</td><td>+0.95</td></tr></table>

Table 30: Decomposition of the original-to-standardized rank shift. $\rho _ { \mathrm { g e n } } = \rho ( \mathbf { S } _ { b } ^ { o , r } , \mathbf { S } _ { b } ^ { s , r } )$ holds extraction fixed while changing generation; $\rho _ { \mathrm { e x t } } = \rho ( \mathbf { S } _ { b } ^ { s , r } , \mathbf { S } _ { b } ^ { s , j } )$ holds standardized generations fixed while changing extraction; and $\rho _ { \mathrm { o r i g } } = \rho ( \mathbf { S } _ { b } ^ { o , r } , \mathbf { S } _ { b } ^ { o , o } )$ checks the regex reconstruction against the original ranking. Lower $\rho$ indicates greater ranking change. All correlations use stored original and standardized generations; this decomposition requires no additional inference.

SECURE and CyberMetric, although both Cyber-Metric correlations remain high. MMLU-CS is sensitive to both, with $\rho _ { \mathrm { g e n } } = 0 . 2 5$ and $\rho _ { \mathrm { e x t } } = 0 . 2 0$ Thus, the rank shifts cannot be attributed simply to replacing benchmark extractors with an LLM judge: substantial reordering is already present when extraction is held fixed. The extraction effect is also reproduced by Sonnet 4.6 (App. E.3). CTI-Bench differs from the full comparison in Table 5 as its attacker-attribution task is excluded here.

## J Recommendation Details

This section expands on §6 by outlining automation limits and the information benchmark releases should report for reliable, reproducible evaluation.

## J.1 Automation Scope

The harness automates configuration logging, model-output collection, response validation, parsing diagnostics, deterministic re-scoring, and aggregate comparisons. It also makes controlled pipeline changes reproducible when the relevant configuration is exposed. Automation does not, however, determine benchmark intent or establish semantic ground truth. Decisions such as whether related threat actors deserve partial credit, which aliases should be equivalent, which capabilities a benchmark should cover, or whether a published label is correct require externally justified policies or evidence. The gold-label audit therefore uses an automated search-grounded verifier followed by human validation (App. G.1.2) rather than treating automated agreement as authoritative ground truth.

## J.2 Evaluation Pipeline Card

We recommend that benchmark releases provide an evaluation pipeline card for each scored task rather than only one specification for the benchmark as a whole. As shown in this paper, different tasks within the same benchmark can use different prompts, inference settings, extraction procedures, metrics, denominator policies, and aggregation rules, so each reported task score should resolve to the exact pipeline that produced it.

For practical reuse, the card should be provided both as human-readable documentation and as a versioned machine-readable record, such as JSON conforming to a public schema. This would allow evaluation repositories and reporting systems to compare results only when their task and pipeline specifications are compatible. This recommendation complements Evaluation Cards (Ghosh et al., 2026), which replaces flat model–benchmark– score reporting with structured evaluation records that resolve results to their underlying benchmark, split, and metric configuration.

For each task, the pipeline card should report:

• Identity: benchmark, task, split, metric, pipeline version, and stable identifiers needed to distinguish the evaluation from related variants.

• Dataset: scored question population, question and label sources, label provenance, inclusion or exclusion rules, and the date or snapshot of evolving references such as CVE and ATT&CK.

• Prompt: exact task prompt, system prompt, demonstrations, chat-formatting policy, and answer-format instructions.

• Inference: decoding parameters, output-token budget, stop sequences, seed policy, serving behavior, and relevant backend constraints.

• Extraction: the executable extraction procedure or judge, including its prompt and decoding configuration when LLM-based.

• Scoring: metrics, credit rule, alias and invalidresponse handling, and denominator policy.

• Aggregation: how question-level measurements are combined into the reported task score.

• Reliability: known invalid-response, extraction, scoring, or configuration sensitivities and the diagnostics used to detect them.

• Reproducibility: what to record in each evaluation run and which analyses require re-scoring or re-generation.

Model identity, model-specific serving details, timestamps, and numerical results belong to the corresponding evaluation-run record rather than the task-level pipeline card. A result can be interpreted as a run-specific score linked to a versioned tasklevel pipeline specification. Eval Card 1 shows this distinction using only the CTI-Bench VSP task.

## EVAL CARD 1 : CTI-Bench / VSP / CVSS MAD

Identity. Benchmark: CTI-Bench. Task: Vulnerability Severity Prediction (VSP). Scored split: 1,000 released VSP questions. Reported metric: CVSS mean absolute deviation (MAD), lower is better.

Dataset. Each question provides a CVE description and asks for a CVSS v3.1 vector. All 1,000 released VSP questions are scored. The benchmark does not specify a snapshot date for the underlying CVE/CVSS references.

Prompt. The released CTI-Bench task prompt is preserved. The system instruction is “You are a cybersecurity expert specializing in cyberthreat intelligence.” The task prompt asks for analysis of the supplied CVE description and requires the final CVSS v3.1 vector string. The standardized pipeline uses the task system prompt and the evaluated system’s native chat formatting.

Inference. The task uses a maximum output budget of 2,048 tokens. The standardized pipeline uses greedy decoding with temperature 0, top-p = 1.0, and no backend stop sequence. No task-specific sampling seed is required under deterministic decoding.

Extraction. The released pipeline returns the last matching CVSS:3.1/ vector found anywhere in the response, although the prompt requests the final vector in the answer. The standardized pipeline uses the pinned semantic extraction policy and expects a normalized CVSS:3.1/ vector. Reasoning enclosed in <think>...</think> is removed before extraction, and the extractor must not repair or infer an unstated answer.

Scoring. The extracted vector is scored using mean absolute deviation between the predicted and gold CVSS values. Lower MAD indicates better performance. An empty or unparseable response remains part of the attempted-question population and is not discarded.

Aggregation. Question-level CVSS deviations are aggregated over the 1,000 scored questions into one task-level MAD value. No cross-task aggregation is part of this task specification.

Reliability. The extraction rule is a known sensitivity point. Applying final-line and anywhere CVSS extraction to identical stored responses yields 30.4% disagreement among cases for which at least one rule extracts a vector. Invalid and unparseable responses should therefore be reported explicitly alongside the task score.

Reproducibility. A run using this task should record the pipeline version, exact prompt configuration, decoding parameters, token budget, stop-sequence configuration, extraction rule, scoring rule, denominator policy, and raw response. Changes to extraction, scoring, or denominator policy can then be evaluated by re-scoring stored responses; changes to prompting or inference require new generations.

In practice, each task card should have an equivalent machine-readable representation with stable field names and versioned identifiers, allowing a reported result to link unambiguously to the task, metric, and pipeline configuration that produced it.

The same task-level specification of Eval Card 1 can be represented in a machine-readable form, as illustrated in Code 4.

## CODE 4 : Machine-readable version of Eval Card 1

```json
{
"identity": {
"benchmark": "CTI-Bench",
"task": "vsp",
"pipeline_version": "standardized"
},
"dataset": {
"questions": 1000,
"input": "CVE description",
"target": "CVSS v3.1 vector",
"reference_snapshot": null
},
"prompt": {
"system": "cyberthreat-intelligence expert",
"chat_format": "native",
"answer format": "CySS:3.1 yector'
},
"inference": {
"temperature": 0,
"top_p": 1.0,
"max_tokens": 2048,
"stop": null
},
"extraction": {
"method": "pinned semantic extractor",
"output": "normalized CVSS:3.1 vector"
},
"scoring": {
"metric": "CVSS MAD",
"direction": "lower",
"i lid " " i d"
},
"aggregation": {
"level": "task",
"questions": 1000
},
"reliability": {
"extractor_disagreement_contested_pct": 30.4
},
"reproducibility": {
"store_raw_outputs": true,
"rescoring_supported": true,
"prompt_or_inference_change_requires_regeneration":
true
}
}
```