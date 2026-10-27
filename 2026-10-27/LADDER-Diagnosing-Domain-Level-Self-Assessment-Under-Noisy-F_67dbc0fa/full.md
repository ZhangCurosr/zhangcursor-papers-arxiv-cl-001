# LADDER: Diagnosing Domain-Level Self-Assessment Under Noisy Feedback

Zhenshuai Yin<sup>1,\*</sup> Jianshu Liang<sup>1,\*</sup> Xuyue Xiong<sup>1</sup> Yongxin Guo<sup>3</sup> Mingda Zhang<sup>1</sup> Zhongze Luo<sup>1</sup> Haoyue Liu<sup>1</sup> Lianmin Zhang<sup>2</sup>

Xiaoying Tang<sup>1,†</sup>

<sup>1</sup>School of Science and Engineering, The Chinese University of Hong Kong (Shenzhen), Longgang, Shenzhen, Guangdong, 518172, P.R. China <sup>2</sup>Shenzhen Research Institute of Big Data, Shenzhen, China <sup>3</sup>ByteDance <sup>\*</sup>Equal contribution. <sup>†</sup>Corresponding author: Xiaoying Tang.

## Abstract

Knowing when they are reliable is essential for language models to support human–AI collaboration. Yet interaction feedback is not always complete or correct, and may mislead models understanding of their own capabilities. We introduce LADDER to study how language models assess themselves under unreliable feedback. Our findings show that models can accurately report supplied performance information while still struggling to form trustworthy self-assessments from interaction experience. Apparently accurate assessments can obscure capability differences and do not necessarily predict future task performance. These findings suggest that trustworthy self-assessment should be evaluated against observed task performance and its predictive value. Our study provides empirical evidence of the limits of model self-knowledge and a basis for evaluating capability awareness in reliable human–AI collaboration.

## 1 Introduction

Repeated use of language models exposes differences in their reliability across domains. A useful self-model would summarize these differences and provide evidence for deciding when additional verification is needed. Yet an interaction history is an imperfect source of such evidence: users may accept an incorrect answer, question a correct one, or provide no usable judgment. Before studying how a capability profile changes an agent’s decisions, we need to establish whether that profile accurately summarizes its observed performance. We investigate this prerequisite through domain-level reconstruction from question-answering histories.

![](images/64969b2371192d957ae73785e4fa1d80bb7f4c606f2978d0f371ac1416e6640f.jpg)  
Figure 1: The LADDER research question. The upper panel illustrates an application in which user feedback can be incorrect. The evaluated setting in the lower panel reconstructs a domain profile from QA records with simulated feedback; profile estimates are scored against recorded answer outcomes.

We organize the question into three diagnostic contrasts. H1 asks whether a model can report supplied numerical statistics. H2 measures the accuracy of its prior domain estimates. H3 asks how well observed feedback supports reconstruction of performance. These contrasts motivate controls with different information access, from no history to verified outcomes. Work on selfknowledge (Barkan et al., 2026), calibration (Kadavath et al., 2022; Guo et al., 2017; Xiong et al., 2024), and memory (Packer et al., 2023; Chhikara et al., 2025; Tan et al., 2025; Xu et al., 2025; Li et al., 2025c; Hu et al., 2026) motivates these questions. Our goal is to distinguish errors associated with the available information and its aggregation, rather than infer a unique internal cause from a benchmark score.

![](images/14802a2b869c05677efe57085d985a94aa56c82667956b98c1c08cee6c7ecd5e.jpg)  
Evidence across interaction settingsFigure 2: LADDER’s diagnostic architecture. Interaction records connect tasks, agent responses, and user feedback to a domain-level self-model. The L0–L4 ladder varies access to prior knowledge, noisy feedback, verified <sup>MMLU</sup> <sup>reference Human</sup> <sup>feedback Telecom</sup> <sup>pil</sup>outcomes, and supplied statistics. Estimated profiles are evaluated against empirical domain performance.

We study this question with LADDER. The benchmark is organized around an oracle-bracketed five-level baseline ladder: L0 a constant nomemory profile, L1 a zero-shot per-domain prior, L2 noisy-feedback learners (LLM self-assessment, classical recalibration, EM-Joint, and Dawid– Skene/GLAD-style estimators), L3 exact counting of verified outcomes, and L4 model reporting of supplied statistics. L4–L3 examines numerical reporting (H1), L1–L3 measures prior error (H2), and L2–L3 measures reconstruction from feedback (H3). Each agent is paired with a domainheterogeneous simulated user. Sessions last 50 turns, and Capability Boundary Fidelity (CBF) evaluates the resulting domain profile alongside per-query calibration measures such as ECE.

Across 18 models from seven families, the ladder reveals a distinction between recovering overall accuracy and learning differences across domains. A global feedback mean outperforms domain-specific Count, Beta, DS, and GLAD-style estimators on 14 of 18 models at the session endpoint. Prior self-assessments and LSA can improve on these references, but their advantage varies by model. A matched-input control on GPT-5.5, DeepSeek-V4-Pro, and Qwen2.5-7B shows that clean-label aggregation and reporting supplied statistics are also different tasks: LSA reaches CBF .989, .995, and .835, respectively, while all three reproduce the supplied statistics exactly. Figure 2 links these contrasts to the diagnostic ladder. Centered domainshape error e<sub>shape</sub> complements CBF by measuring domain-profile error after removing overall bias.

Contributions. (i) A five-level diagnostic protocol for comparing prior knowledge, feedbackbased estimation, and numerical reporting; (ii) domain-level CBF with baseline adjustment and complementary signed and centered errors; and (iii) an 18-model empirical study, with matched-input, noise, domain-partition, and cross-task controls that identify when feedback improves the estimated capability profile.

## 2 Related Work

LADDER connects model self-knowledge with learning from interaction feedback. We organize related work around capability assessment, interactive agents, memory and feedback learning, and statistical estimation. Table 1 compares representative benchmarks by feedback channel, interaction horizon, accumulated self-knowledge, and userreliability inference.

Model self-knowledge and calibration. Selfknowledge research examines whether models can recognize and report the correctness of their own answers. Prior work examines answer confidence (Kadavath et al., 2022), activation monitoring (Li et al., 2025a), and behavioral selfprediction (Binder et al., 2025). Behavioral evaluations find context-dependent metacognition (Ackerman, 2026), while position work argues for its role in self-improvement (Liu and van der Schaar, 2025). SAD (Laine et al., 2024) and Self-Aware (Yin et al., 2023) assess self-knowledge without an accumulated feedback history, providing a reference for prior self-assessment.

Knowledge-boundary studies and taskperformance benchmarks (Li et al., 2025b; Yin et al., 2024; Wei et al., 2024; Hendrycks et al., 2021; Lin et al., 2022; Liang et al., 2022; Geng et al., 2024) characterize models’ capabilities on scored questions. Selective prediction (Kapoor et al., 2024; Zhang et al., 2024; Traub et al., 2024; Wen et al., 2025; Kirichenko et al., 2025; Ahdritz et al., 2024) and confidence calibration (Xu et al., 2024; Shen et al., 2024; Chen et al., 2024; Manggala et al., 2025) connect confidence to query-level judgments. LADDER extends these questions to capability profiles estimated from repeated interactions with imperfect feedback, evaluating whether reported domain performance agrees with recorded answer outcomes. Appendix G.2 provides further comparisons.

Interactive agent evaluation. Multi-step and multi-turn benchmarks relate model reliability to behavior during task execution. Barkan et al. (2026) study models’ predictions of their own success on BigCodeBench and SWE-Bench. Agent-Board (Ma et al., 2024) evaluates intermediate progress and task completion, while τ -bench (Yao et al., 2024) measures performance in interactive tasks. These evaluations motivate a complementary target: whether interaction experience yields an accurate account of a model’s capabilities. LADDER studies this target by comparing estimated domain profiles with observed performance under imperfect feedback.

Agent memory and feedback learning. Memory systems study how information from previous interactions can be retained and used in subsequent reasoning. MemGPT (Packer et al., 2023), Mem0 (Chhikara et al., 2025), RMM (Tan et al., 2025), A-MEM (Xu et al., 2025), MemOS (Li et al., 2025c), and MEM1 (Zhou et al., 2026b) explore this capability, while MemoryAgentBench (Hu et al., 2026) evaluates memory use over extended interactions. RLHF (Ouyang et al., 2022) and DPO (Rafailov et al., 2023) use feedback to update model parameters. More broadly, Wu et al. (2025) review how knowledge sharing and complementary model strengths support collaboration and coevolution. LADDER examines the accuracy of capability estimates formed from interaction records with fixed model parameters, connecting accumulated experience to an explicit self-assessment target.

Imperfect feedback and statistical estimation. Research on implicit feedback (Liu et al., 2025) and confidence after feedback training (Leng et al., 2025) motivates examining how feedback informs capability judgments. Statistical methods for noisy labels provide complementary references: Dawid– Skene (Dawid and Skene, 1979) models annotator confusion, while GLAD (Whitehill et al., 2009) relates annotator ability to item difficulty. LAD-DER adapts these estimation ideas to domain performance and compares them with direct counting and a global feedback mean, examining how feedback quality and aggregation affect the resulting capability profile.

## 3 LADDER

Evaluation objective. Reliable self-assessment should agree with observed task performance. Yet interaction feedback may be missing or incorrect, so user approval or criticism alone does not establish whether a model accurately understands its capabilities. LADDER therefore asks whether models can use imperfect feedback to accurately summarize their performance across domains. By relating reported capabilities to recorded answer outcomes, the benchmark examines whether interaction experience supports trustworthy selfknowledge.

Design rationale. We record model answers and natural-language feedback from simulated users, and use scored answer outcomes to establish a domain-performance reference. Against this common target, LADDER compares capability estimates under different information conditions. Prior self-assessments and feedback histories examine existing capability knowledge and reconstruction from interaction experience, while verified outcomes and supplied statistics provide references with more complete information. This ladder allows us to examine whether prior assessments are accurate, whether feedback improves them, and whether models can report known performance (Figure 2).

Task definition. A session contains T interactions. Turn t records a question $q _ { t }$ , an answer $a _ { t } .$ reported confidence $c _ { t }$ , and natural-language feedback $f _ { t } .$ . Let $\mathcal { D }$ denote the domain set. A profile estimator uses its available information to produce domain-accuracy estimates ${ \hat { p } } : { \mathcal { D } }  [ 0 , 1 ]$ , evaluated against empirical domain accuracy $p ^ { * } ( d )$ from scored answer outcomes. The answering model sees the current question; profile estimators receive the records permitted by their information condition after the session. Domain identifiers are supplied to the estimators.

<table><tr><td>Benchmark</td><td>Feedback</td><td>Horizon</td><td>Self-knowledge</td><td>2nd- order</td></tr><tr><td>SAD (Laine et al., 2024), SelfAware (Yin et al., 2023)</td><td>none</td><td>single-shot</td><td>static probe</td><td></td></tr><tr><td>Barkan (Barkan et al., 2026) MemoryAgentBench (Hu et al., 2026) AgentBoard (Ma et al., 2024),</td><td>execution traces implicit (mem ops) progress / task reward</td><td>multi-step exec long-horizon multi-turn</td><td>per-task exogenous mem only none</td><td></td></tr><tr><td>τ-bench (Yao et al., 2024) SimpleQA (Wei et al., 2024)</td><td>none</td><td>single-shot</td><td>static</td><td></td></tr><tr><td>LADDER (ours)</td><td>noisy domain-dependent NL</td><td>50-turn session</td><td>per-domain</td><td>√</td></tr></table>

Table 1: Related benchmarks along four design axes. LADDER combines domain-dependent feedback, repeated observations, and explicit reliability-estimation controls.

The benchmark uses 50 recorded sessions of 50 attempted turns per model over 20 MMLU subjects (Hendrycks et al., 2021). Strong/Medium/Weak tiers summarize each model’s subject accuracy. Every simulated persona has three assigned expertise subjects. This assignment controls the requested feedback behavior; realized label agreement, coverage, confirmation, and correction are measured separately (Appendix B.3).

We additionally construct three extension tasks— GSM8K (Cobbe et al., 2021), HumanEval (Chen et al., 2021), and BFCL tool-calling (Yan et al., 2024), collectively called the extension suite—to test cross-task generalisation (Appendix F.1). Full task formalisation, session protocol, question-pool partitioning, and persona parameterisation are deferred to Appendix B.1.

Capability Boundary Fidelity (CBF). Our central metric scores how closely the agent’s selfassessed per-domain accuracy tracks the truth:

$$
\begin{array} { c } { \displaystyle \mathrm { C B F } = 1 - \frac { 1 } { | { \mathcal D } | } \sum _ { d \in { \mathcal D } } | \hat { \boldsymbol p } ( d ) - \boldsymbol p ^ { * } ( d ) | , } \\ { \displaystyle \mathrm { C B F } \in [ 0 , 1 ] . } \end{array}\tag{1}
$$

CBF is computed at the session endpoint over domains containing at least two valid answers. All estimators share this target and support. ECE uses individual confidence–outcome pairs, whereas CBF uses domain-profile pairs (Figure 3). A constant profile can achieve high CBF when domain accuracies are close to its value; CBF therefore measures profile error, not task competence.

![](images/eabc1c763d4e7be3b25f49bff51e9309f653619b166640128276706360f5dba9.jpg)  
Figure 3: CBF and ECE describe different targets. Constant-profile CBF and recorded-confidence ECE for all 18 models.

Baseline-adjusted CBF: relative to a fixed reference. To express improvement relative to the no-memory reference, we define a normalized variant that measures the fraction of the oracle gap a method closes relative to the constant baseline:

$$
\begin{array} { r l } & { \widetilde { \mathrm { C B F } } ( m , \mathrm { m e t h o d } ) = } \\ & { \frac { \mathrm { C B F } _ { \mathrm { m e t h o d } } ( m ) - \mathrm { C B F } _ { \mathrm { N o M e m } } ( m ) } { \mathrm { C B F } _ { \mathrm { O r a c l e } } ( m ) - \mathrm { C B F } _ { \mathrm { N o M e m } } ( m ) } , } \end{array}\tag{2}
$$

Here NoMem is the constant .5 profile and Oracle is exact counting with CBF 1. A value of 1 closes the reference error, 0 ties NoMem, and a negative value performs worse. Raw CBF supports withinmodel comparisons; adjusted CBF reports relative error reduction on the same target.

Direction and domain shape. To distinguish overall miscalibration from domain discrimination,

we also report

$$
b = | \mathcal { D } | ^ { - 1 } \sum _ { d } ( \hat { p } _ { d } - p _ { d } ^ { * } ) ,\tag{3}
$$

$$
e _ { \mathrm { s h a p e } } = | \mathcal { D } | ^ { - 1 } \sum _ { d } | ( \hat { p } _ { d } - \bar { \hat { p } } ) - ( p _ { d } ^ { * } - \bar { p } ^ { * } ) | .\tag{4}
$$

Positive b denotes overestimation; lower $e _ { \mathrm { s h a p e } }$ denotes better recovery of domain contrasts. These complement absolute error rather than form an additive decomposition.

Auxiliary metrics and domain structure. We report error rate, ECE, and Brier score as auxiliary measures; their definitions appear in Appendix B.1. ECE follows Guo et al. (2017); Brier measures squared confidence–outcome error. CBF requires multiple domains. For the single-domain extension tasks (GSM8K, HumanEval, and the BFCL name/key control), we report task scores and confidence–score gaps (Appendix F.1).

Cross-model comparability. The 20 subjects are grouped into Strong/Medium/Weak tiers relative to each model’s measured subject accuracy. CBF compares methods against the same model-specific target $p ^ { * } ( d )$ . Means across models summarize profile-estimation error; task accuracy is reported separately. Appendix B.1 gives the subject inventory and scoring units.

Baselines: a five-level diagnostic ladder. The baselines and reliability controls share five information levels.

• L0: NoMemory assigns $\hat { p } _ { d } = . 5 ;$ per-answer confidence is evaluated separately.

• L1: ZS-SK asks the model for domain accuracies without a session history.

• L2: LSA reads question and feedback history. UF-PDC, Platt, HistBin, Bayes, EM-Joint, Global, DS, and GLAD-style aggregate the same strictly decoded positive/negative label stream; ambiguous labels are missing. Platt and Hist-Bin recalibrate per-answer confidence but use domain counts for their CBF profiles.

• L3: OraclePDC counts verified outcomes on the same scored turns. Its CBF of 1 is an exact reconstruction reference.

• L4: GT-SelfStats asks the model to report supplied empirical domain accuracies.

LSA’s natural-language history and the algorithmic label stream are distinct input representations. The matched-input control fixes the supplied label history to examine aggregation directly. Baseline implementations appear in Appendix C.1.

## 4 Diagnostic Results: Where Does Agent Self-Knowledge Fail?

We evaluate the five information levels across 18 models, then examine matched controls, cross-task variation, and held-out prediction.

## 4.1 Setup

The evaluation covers 18 models from seven families (Appendix G.3), with 50 recorded sessions of 50 turns per model. Accuracy is the fraction of correct answers among parsable A–D responses with A–D references. All estimators use the same scored answers and corresponding feedback; missing feedback supplies no label but does not remove an answer from the target. Each session profile covers domains with at least two scored answers. Session means compare estimators on the same interaction histories, while pooled scores assess estimation after combining a model's histories. The five levels compare different information conditions against these shared targets.

## 4.2 Main Results

Table 2 retains the complete model panel and the original five-level comparison. Three results organize the diagnosis: prior self-assessment differs from history-based estimation; pooled aggregation can change the ordering of noisy-feedback estimators; and numerical reporting can succeed even when aggregation is inaccurate.

Overall accuracy and domain structure. Global leads the five matched algorithmic controls on 14/18 session-level rows (Table 3). On GPT-5.5, its CBF is 0.800, compared with 0.603 for Beta and 0.609 for GLAD-style. Global assigns every domain one value: its score measures recovery of the overall level. A domain profile must additionally reduce centered shape error $e _ { \mathrm { s h a p e } }$ to demonstrate useful differentiation. This contrast is a benchmark finding: high CBF alone can reward overall calibration without learning domain differences. The paired level and shape measures identify which improvement an estimator provides.

H1: reporting supplied statistics. All 18 complete main-panel L4 evaluations reproduce the supplied domain statistics, giving CBF 1.000. The three-model matched-input experiment gives the same result. In contrast, clean-label LSA on Qwen2.5-7B reaches .835. This contrast distinguishes successful numerical reporting from accurate aggregation of a history.

<table><tr><td rowspan="2">Model (Acc)</td><td rowspan="2">N</td><td rowspan="2">L0 NoMem</td><td rowspan="2">L1 ZS-SK</td><td colspan="4">L2</td><td rowspan="2">L3</td><td rowspan="2">L4</td><td colspan="2">L1 errors</td></tr><tr><td>LSA</td><td>Count*</td><td>Bayes</td><td>EM OraPDC</td><td>GT-Stats</td><td>b eshape</td></tr><tr><td>Mistral-7B (41.6%)</td><td>2263</td><td>0.745</td><td>0.677</td><td>0.638</td><td>0.734</td><td>0.748</td><td>0.748</td><td>1.000</td><td>1.000</td><td>+0.202</td><td>0.269</td></tr><tr><td>Llama-3-8B (57.7%)</td><td>2268</td><td>0.736</td><td>0.734</td><td>0.549</td><td>0.739</td><td>0.747</td><td>0.748</td><td>1.000</td><td>1.000</td><td>-0.028</td><td>0.259</td></tr><tr><td>GLM-5 (95.8%)</td><td>1605</td><td>0.542</td><td>0.904</td><td>0.568</td><td>0.595</td><td>0.575</td><td>0.572</td><td>1.000</td><td>1.000</td><td>-0.034</td><td>0.071</td></tr><tr><td>DeepSeek-V3.2 (88.6%)</td><td>1831</td><td>0.596</td><td>0.834</td><td>0.761</td><td>0.638</td><td>0.628</td><td>0.626</td><td>1.000</td><td>1.000</td><td>-0.016</td><td>0.149</td></tr><tr><td>Qwen-Turbo (75.5%)</td><td>2358</td><td>0.688</td><td>0.772</td><td>0.745</td><td>0.698</td><td>0.707</td><td>0.711</td><td>1.000</td><td>1.000</td><td>+0.159</td><td>0.212</td></tr><tr><td>Qwen2.5-7B (60.6%)</td><td>2462</td><td>0.740</td><td>0.670</td><td>0.732</td><td>0.741</td><td>0.751</td><td>0.753</td><td>1.000</td><td>1.000</td><td>+0.251</td><td>0.254</td></tr><tr><td>Qwen-Max (78.8%)</td><td>2368</td><td>0.680</td><td>0.789</td><td>0.757</td><td>0.666</td><td>0.679</td><td>0.680</td><td>1.000</td><td>1.000</td><td>+0.063</td><td>0.204</td></tr><tr><td>Qwen-Plus (82.2%)</td><td>2238</td><td>0.646</td><td>0.806</td><td>0.558</td><td>0.672</td><td>0.667</td><td>0.667</td><td>1.000</td><td>1.000</td><td>+0.108</td><td>0.200</td></tr><tr><td>Qwen3.5-27B (95.9%)</td><td>1671</td><td>0.543</td><td>0.886</td><td>0.571</td><td>0.595</td><td>0.576</td><td>0.573</td><td>1.000</td><td>1.000</td><td>-0.046</td><td>0.083</td></tr><tr><td>GLM-5.1 (94.2%)</td><td>1556</td><td>0.557</td><td>0.608</td><td>0.473</td><td>0.579</td><td>0.577</td><td>0.578</td><td>1.000</td><td>1.000</td><td>-0.374</td><td>0.148</td></tr><tr><td>GPT-5.4-mini (75.0%)</td><td>2467</td><td>0.680</td><td>0.764</td><td>0.747</td><td>0.686</td><td>0.688</td><td>0.686</td><td>1.000</td><td>1.000</td><td>+0.053</td><td>0.229</td></tr><tr><td>GPT-5.4 (82.8%)</td><td>2472</td><td>0.642</td><td>0.797</td><td>0.762</td><td>0.647</td><td>0.650</td><td>0.651</td><td>1.000</td><td>1.000</td><td>+0.008</td><td>0.195</td></tr><tr><td>GPT-5.5 (92.8%)</td><td>2500</td><td>0.570</td><td>0.858</td><td>0.844</td><td>0.612</td><td>0.603</td><td>0.605</td><td>1.000</td><td>1.000</td><td>-0.059</td><td>0.106</td></tr><tr><td>Claude-Opus-4.7 (90.4%)</td><td>1167</td><td>0.582</td><td>0.742</td><td>0.706</td><td>0.624</td><td>0.613</td><td>0.609</td><td>1.000</td><td>1.000</td><td>-0.225</td><td>0.129</td></tr><tr><td>Claude-Sonnet-4.6 (93.0%)</td><td>914</td><td>0.560</td><td>0.801</td><td>0.739</td><td>0.592</td><td>0.583</td><td>0.579</td><td>1.000</td><td>1.000</td><td>-0.154</td><td>0.099</td></tr><tr><td>Claude-Opus-4.6 (93.0%)</td><td>949</td><td>0.563</td><td>0.771</td><td>0.753</td><td>0.620</td><td>0.601</td><td>0.592</td><td>1.000</td><td>1.000</td><td>-0.187</td><td>0.110</td></tr><tr><td>DeepSeek-V4-Pro (92.8%)</td><td>2452</td><td>0.562</td><td>0.316</td><td>0.414</td><td>0.598</td><td>0.590</td><td>0.592</td><td>1.000</td><td>1.000</td><td>-0.682</td><td>0.103</td></tr><tr><td>Qwen3.6-Plus (90.7%)</td><td>2357</td><td>0.589</td><td>0.719</td><td>0.821</td><td>0.616</td><td>0.614</td><td>0.616</td><td>1.000</td><td>1.000</td><td>-0.233</td><td>0.146</td></tr></table>

Table 2: Main results across 18 models and all five levels. Mean session CBF on shared domain targets. N is the number of scored answers; parentheses give accuracy. Count\* is the shared domain output of UF-PDC, Platt, and HistBin. Bold marks the best displayed L0–L2 score. The last columns report L1 signed bias and centered domain-shape error on the same session targets. Additional L2 controls appear in Table 3.

H2: prior domain assessment. L1 improves on the constant L0 reference for 14 of the 18 models, but the size and direction of the prior error vary. For example, Qwen2.5-7B scores 0.670 at L1, below its L0 score of 0.740. The signed and centered errors in Table 2 distinguish a misplaced overall confidence level from incorrect differences among domains. A useful prior must capture both: a high CBF alone is insufficient evidence of accurate domain discrimination.

H3: estimating performance from feedback. Reliability-aware estimation does not uniformly improve on simple aggregation. Session and pooled fits use different evidence budgets: for Qwen2.5- 7B, pooled Beta reaches 0.874 and GLAD-style 0.724. All main and sensitivity comparisons use the same positive-ability initialization for GLAD-style. The L2–L3 gap combines label errors, missing feedback, and estimator behavior; Appendix E.1 separates label correction from coverage. The comparisons describe the evaluated adaptations under frozen feedback rules; Tables 18 and 21 examine optimization and decoding sensitivity.

Input-matched aggregation. The common-pool experiment holds the 398 scored questions and feedback labels fixed for GPT-5.5, DeepSeek-V4- Pro, and Qwen2.5-7B. At full coverage, changing from clean labels to 30% symmetric label noise changes LSA CBF from .989 to .715, .995 to .760, and .835 to .886. The Qwen result shows that aggregation error and label noise need not affect CBF monotonically. Full results for all label channels and estimators appear in Appendix E.5.

Label quality and coverage. Across the 18 models, mean pooled Count CBF rises from 0.852 with decoded feedback to 0.886 after correcting labels at the same observed positions, and to 1.000 with complete correct labels. The larger gain from complete coverage shows that missing feedback, in addition to incorrect feedback, limits profile reconstruction (Appendix E.1).

## 4.3 Diagnosis via the Ladder Visualization

Figure 4 displays the five levels for all 18 models. Each row follows the same target across all levels, with LSA representing L2. The matrix exposes differences among L0–L2 while showing the L3 and L4 references explicitly. Algorithmic alternatives appear in Tables 2 and 3.

Task competence and profile accuracy. Figure 5 places answer accuracy beside the highest observed non-oracle CBF for each model. These quantities describe different capabilities: solving the questions and estimating the resulting performance profile. Their different model orderings show why task accuracy cannot substitute for selfassessment evaluation. The spectrum also places the five-level comparison in context: the value of a profile estimate depends on how closely it captures the model’s own outcomes, rather than on how often that model answers correctly.

<table><tr><td>Model</td><td>Global</td><td>Count</td><td>Beta</td><td>DS</td><td>GLAD-style</td></tr><tr><td>Mistral-7B</td><td>0.691/0.812</td><td>0.734/0.809</td><td>0.748/0.822</td><td>0.742/0.799</td><td>0.731/0.712</td></tr><tr><td>Llama-3-8B</td><td>0.717/0.835</td><td>0.739/0.852</td><td>0.747/0.876</td><td>0.748/0.868</td><td>0.734/0.730</td></tr><tr><td>GLM-5</td><td>0.774/0.934</td><td>0.595/0.862</td><td>0.575/0.842</td><td>0.575/0.883</td><td>0.594/0.908</td></tr><tr><td>DeepSeek-V3.2</td><td>0.769/0.925</td><td>0.638/0.849</td><td>0.628/0.852</td><td>0.627/0.845</td><td>0.636/0.885</td></tr><tr><td>Qwen-Turbo</td><td>0.751/0.897</td><td>0.698/0.892</td><td>0.707/0.911</td><td>0.703/0.813</td><td>0.694/0.793</td></tr><tr><td>Qwen2.5-7B</td><td>0.715/0.857</td><td>0.741/0.846</td><td>0.751/0.874</td><td>0.748/0.791</td><td>0.738/0.724</td></tr><tr><td>Qwen-Max</td><td>0.660/0.732</td><td>0.666/0.710</td><td>0.679/0.731</td><td>0.672/0.676</td><td>0.667/0.643</td></tr><tr><td>Qwen-Plus</td><td>0.750/0.887</td><td>0.672/0.878</td><td>0.667/0.883</td><td>0.670/0.871</td><td>0.670/0.849</td></tr><tr><td>Qwen3.5-27B</td><td>0.775/0.922</td><td>0.595/0.888</td><td>0.576/0.847</td><td>0.577/0.820</td><td>0.594/0.940</td></tr><tr><td>GLM-5.1</td><td>0.711/0.878</td><td>0.579/0.864</td><td>0.577/0.794</td><td>0.580/0.799</td><td>0.578/0.919</td></tr><tr><td>GPT-5.4-mini</td><td>0.700/0.861</td><td>0.686/0.820</td><td>0.688/0.859</td><td>0.686/0.849</td><td>0.685/0.811</td></tr><tr><td>GPT-5.4</td><td>0.689/0.908</td><td>0.647/0.839</td><td>0.650/0.851</td><td>0.648/0.781</td><td>0.647/0.857</td></tr><tr><td>GPT-5.5</td><td>0.800/0.931</td><td>0.612/0.888</td><td>0.603/0.873</td><td>0.601/0.818</td><td>0.609/0.909</td></tr><tr><td>Claude-Opus-4.7</td><td>0.776/0.908</td><td>0.624/0.847</td><td>0.613/0.837</td><td>0.613/0.830</td><td>0.621/0.890</td></tr><tr><td>Claude-Sonnet-4.6</td><td>0.772/0.902</td><td>0.592/0.872</td><td>0.583/0.783</td><td>0.589/0.838</td><td>0.590/0.915</td></tr><tr><td>Claude-Opus-4.6</td><td>0.779/0.907</td><td>0.620/0.877</td><td>0.601/0.830</td><td>0.599/0.833</td><td>0.616/0.917</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.780/0.904</td><td>0.598/0.870</td><td>0.590/0.860</td><td>0.592/0.819</td><td>0.596/0.913</td></tr><tr><td></td><td>0.777/0.893</td><td></td><td>0.614/0.859</td><td>0.613/0.847</td><td>0.612/0.903</td></tr><tr><td>Qwen3.6-Plus</td><td></td><td>0.616/0.868</td><td></td><td></td><td></td></tr></table>

Table 3: Additional matched L2 comparisons for all 18 models. Each cell reports session/pooled CBF. All five methods receive identical labels and use identical scoring domains.

<table><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>L0</td><td rowspan=1 colspan=1>L1</td><td rowspan=1 colspan=1>L2</td><td rowspan=1 colspan=1>L3  L4</td></tr><tr><td rowspan=1 colspan=1>NoMem</td><td rowspan=1 colspan=1>ZS</td><td rowspan=1 colspan=1>LSA</td><td rowspan=1 colspan=1>CountStats</td></tr><tr><td rowspan=2 colspan=1>Mistral-7BLlama-3-8B</td><td rowspan=1 colspan=1>0.745</td><td rowspan=1 colspan=1>0.677</td><td rowspan=1 colspan=1>0.638</td><td rowspan=1 colspan=1>1.0001.000</td></tr><tr><td rowspan=1 colspan=1>0.736</td><td rowspan=1 colspan=1>0.734</td><td rowspan=1 colspan=1>0.549</td><td rowspan=1 colspan=1>1.0001.000</td></tr><tr><td rowspan=2 colspan=1>GLM-5DeepSeek-V3.2</td><td rowspan=1 colspan=1>0.542</td><td rowspan=1 colspan=1>0.904</td><td rowspan=1 colspan=1>0.568</td><td rowspan=1 colspan=1>1.0001.000</td></tr><tr><td rowspan=1 colspan=1>0.596</td><td rowspan=1 colspan=1>0.834</td><td rowspan=1 colspan=1>0.761</td><td rowspan=1 colspan=1>1.0001.000</td></tr><tr><td rowspan=1 colspan=1>Qwen-Turbo</td><td rowspan=1 colspan=1>0.688</td><td rowspan=1 colspan=1>0.772</td><td rowspan=1 colspan=1>0.745</td><td rowspan=1 colspan=1>1.0001.000</td></tr><tr><td rowspan=1 colspan=1>Qwen2.5-7B</td><td rowspan=1 colspan=1>0.740</td><td rowspan=1 colspan=1>0.670</td><td rowspan=1 colspan=1>0.732</td><td rowspan=1 colspan=1>1.0001.000</td></tr><tr><td rowspan=1 colspan=1>Qwen-Max</td><td rowspan=1 colspan=1>0.680</td><td rowspan=1 colspan=1>0.789</td><td rowspan=1 colspan=1>0.757</td><td rowspan=1 colspan=1>1.0001.000</td></tr><tr><td rowspan=1 colspan=1>Qwen-Plus</td><td rowspan=1 colspan=1>0.646</td><td rowspan=1 colspan=1>0.806</td><td rowspan=1 colspan=1>0.558</td><td rowspan=1 colspan=1>1.0001.000</td></tr><tr><td rowspan=1 colspan=1>Qwen3.5-27B</td><td rowspan=1 colspan=1>0.543</td><td rowspan=1 colspan=1>0.886</td><td rowspan=1 colspan=1>0.571</td><td rowspan=1 colspan=1>1.0001.000</td></tr><tr><td rowspan=1 colspan=1>GLM-5.1</td><td rowspan=1 colspan=1>0.557</td><td rowspan=1 colspan=1>0.608</td><td rowspan=1 colspan=1>0.473</td><td rowspan=1 colspan=1>1.0001.000</td></tr><tr><td rowspan=1 colspan=1>GPT-5.4-mini</td><td rowspan=1 colspan=1>0.680</td><td rowspan=1 colspan=1>0.764</td><td rowspan=1 colspan=1>0.747</td><td rowspan=1 colspan=1>1.0001.000</td></tr><tr><td rowspan=1 colspan=1>GPT-5.4</td><td rowspan=1 colspan=1>0.642</td><td rowspan=1 colspan=1>0.797</td><td rowspan=1 colspan=1>0.762</td><td rowspan=1 colspan=1>1.0001.000</td></tr><tr><td rowspan=1 colspan=1>GPT-5.5</td><td rowspan=1 colspan=1>0.570</td><td rowspan=1 colspan=1>0.858</td><td rowspan=1 colspan=1>0.844</td><td rowspan=1 colspan=1>1.0001.000</td></tr><tr><td rowspan=1 colspan=1>Claude-Opus-4.7</td><td rowspan=1 colspan=1>0.582</td><td rowspan=1 colspan=1>0.742</td><td rowspan=1 colspan=1>0.706</td><td rowspan=1 colspan=1>1.0001.000</td></tr><tr><td rowspan=1 colspan=1>Claude-Sonnet-4.6</td><td rowspan=1 colspan=1>0.560</td><td rowspan=1 colspan=1>0.801</td><td rowspan=1 colspan=1>0.739</td><td rowspan=1 colspan=1>1.0001.000</td></tr><tr><td rowspan=1 colspan=1>Claude-Opus-4.6</td><td rowspan=1 colspan=1>0.563</td><td rowspan=1 colspan=1>0.771</td><td rowspan=1 colspan=1>0.753</td><td rowspan=1 colspan=1>1.0001.000</td></tr><tr><td rowspan=1 colspan=1>DeepSeek-V4-Pro</td><td rowspan=1 colspan=1>0.562</td><td rowspan=1 colspan=1>0.316</td><td rowspan=1 colspan=1>0.414</td><td rowspan=1 colspan=1>1.0001.000</td></tr><tr><td rowspan=1 colspan=1>Qwen3.6-Plus</td><td rowspan=1 colspan=1>0.589</td><td rowspan=1 colspan=1>0.719</td><td rowspan=1 colspan=1>0.821</td><td rowspan=1 colspan=1>1.0001.000</td></tr></table>

Figure 4: Five-level CBF for all 18 models. Each cell gives mean session CBF. L2 uses LSA; L3 counts verified outcomes and L4 reports supplied statistics.

The adjacent contrasts answer different questions. L4–L3 asks whether supplied statistics can be reported; L1–L3 quantifies prior error; L2–L1 measures the change from using a history; and L2– L3 measures the reconstruction gap. A large reconstruction gap motivates interventions on decoding, coverage, and aggregation. The matched-input and noise experiments distinguish these interventions without equating the performance of an external estimator with the answering model’s online learning ability.

A diagnostic per-model selection rule (R4). R4 is a descriptive comparison: select ZS if its CBF exceeds NoMem, and LSA otherwise. This rule uses a scored target to choose an estimator. It measures the potential value of choosing between prior and history on observed tasks; prediction on new tasks requires a separate held-out test (Appendix E.10).

## 4.4 Cross-Task Heterogeneity and Profile Transfer

Across MMLU, GSM8K, HumanEval, and BFCL, task accuracy and reported confidence can differ substantially for the same model. The 18-model extension tables compare confidence minus accuracy where model-produced confidence is available (Appendix F.1). MMLU-Pro and BBH add subject-level prior comparisons for all eight frontier models. To examine prediction directly, a separate ReAct analysis uses the first four questions per subject to predict performance on the next four. Across six configurations and 960 records, empirical ReAct Count profiles improve on a global training-accuracy predictor in two configurations, while zero-shot profiles improve in one. Domain structure is therefore a candidate prediction signal whose benefit needs to be established on heldout tasks. Estimator rankings also change across threshold, random, and semantic domain partitions; Global leads four of five models after merging subjects into four MMLU categories (Appendix E.6). Recorded tool-use rates across the six configurations range from 37.5% to 100%; this is toolassisted QA.

![](images/93b15c087c0f21189225303a1d3e05b79017e6852b8fd645a41a72d5c5e92853.jpg)  
Figure 5: Capability and profile-estimation spectrum. Answer accuracy and the highest observed non-oracle CBF across the 18-model panel.

## 5 Conclusion

LADDER evaluates domain-level self-assessment under noisy feedback through five information conditions. Across 18 models, the results separate overall calibration from domain discrimination and numerical reporting from history aggregation. Global feedback averages provide a strong reference, while the paired level and shape metrics reveal whether domain-specific estimates add useful information. Matched-input controls identify feedback quality, coverage, and aggregation as distinct sources of profile error.

The held-out ReAct analysis shows that accurate reconstruction need not imply predictive benefit. LADDER connects self-knowledge to recorded performance and tests its value on new tasks. Further evaluation should compare richer memory representations under matched targets and feedback budgets, using larger agentic task sets and documented human-feedback studies.

## Limitations

LADDER studies self-assessment primarily through structured question answering. Extending its conclusions to multilingual, open-ended, and long-horizon interactive tasks requires broader evaluation, where task success and useful feedback may be harder to define.

The feedback in this study comes from simulated users. Real human collaboration involves changing goals, implicit judgments, and more diverse response patterns. A central direction for future work is to test whether the resulting self-assessments can improve decisions in sustained human–agent interaction.

## Ethical considerations

The study uses public benchmark questions and LLM-simulated feedback. No human participants are involved in the reported experiments. Benchmark materials remain subject to their respective licenses. Domain-profile estimates can help identify mismatches between reported and observed performance, but their use in consequential decisions requires validation on the intended task and user population.

## References

Christopher Ackerman. 2026. Evidence for limited metacognition in llms. Proceedings ofICLR.

Gustaf Ahdritz, Tian Qin, Nikhil Vyas, Boaz Barak, and Benjamin L. Edelman. 2024. Distinguishing the knowable from the unknowable with language models. Proceedings ofICML.

Casey O. Barkan, Sid Black, and Oliver Sourbut. 2026. Do large language models know what they are capable of? Proceedings ofICLR.

Felix J Binder, James Chua, Tomek Korbak, Henry Sleight, John Hughes, Robert Long, Ethan Perez, Miles Turpin, and Owain Evans. 2025. Looking inward: Language models can learn about themselves by introspection. Proceedings ofICLR.

Lihu Chen, Alexandre Perez-Lebel, Fabian M Suchanek, and Gaël Varoquaux. 2024. Reconfidencing llms from the grouping loss perspective. Findings of EMNLP.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, et al. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building

production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

A Philip Dawid and Allan M Skene. 1979. Maximum likelihood estimation of observer error-rates using the em algorithm. Journal of the Royal Statistical Society Series C: Applied Statistics, 28(1):20–28.

Y. Dong, X. Jiang, et al. 2025. Rl-plus: Countering capability boundary collapse of llms in reinforcement learning with hybrid-policy optimization. arXiv preprint arXiv:2508.00222.

Ke Fang, Tianyi Zhao, and Lu Cheng. 2025. Credence calibration game? calibrating large language models through structured play. arXiv preprint arXiv:2508.14390v1.

Jiahui Geng, Fengyu Cai, Yuxia Wang, Heinz Koeppl, Preslav Nakov, and Iryna Gurevych. 2024. A survey of confidence estimation and calibration in large language models. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6577–6595.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. 2017. On calibration of modern neural networks. In ICML.

Haixia Han, Tingyun Li, Shisong Chen, Jie Shi, Chengyu Du, Yanghua Xiao, Jiaqing Liang, and Xin Lin. 2024. Enhancing confidence expression in large language models through learning from past experience. arXiv preprint arXiv:2404.10315.

Dan Hendrycks et al. 2021. Measuring massive multitask language understanding. In ICLR.

Yuanzhe Hu, Yu Wang, and Julian McAuley. 2026. Evaluating memory in llm agents via incremental multiturn interactions. Proceedings ofICLR.

Saurav Kadavath et al. 2022. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221.

Sanyam Kapoor, Nate Gruver, Manley Roberts, Katherine Collins, Arka Pal, Umang Bhatt, Adrian Weller, Samuel Dooley, Micah Goldblum, and Andrew Gordon Wilson. 2024. Large language models must be taught to know what they don’t know. Proceedings ofNeurIPS.

Polina Kirichenko, Mark Ibrahim, Kamalika Chaudhuri, and Samuel J. Bell. 2025. Abstentionbench: Reasoning llms fail on unanswerable questions. Proceedings ofNeurIPS, 38.

Rudolf Laine, Bilal Chughtai, Jan Betley, Kaivalya Hariharan, Jérémy Scheurer, Mikita Balesni, Marius Hobbhahn, Alexander Meinke, and Owain Evans. 2024. Me, myself, and ai: The situational awareness dataset (sad) for llms. arXiv preprint arXiv:2407.04694.

Jixuan Leng, Chengsong Huang, Banghua Zhu, and Jiaxin Huang. 2025. Taming overconfidence in llms: Reward calibration in rlhf. Proceedings ofICLR.

Ji-An Li, Huadong Xiong, Robert Wilson, Marcelo G. Mattar, and Marcus K. Benna. 2025a. Language models are capable of metacognitive monitoring and control of their internal activations. Proceedings of NeurIPS, 38.

Moxin Li, Yong Zhao, Wenxuan Zhang, Shuaiyi Li, Wenya Xie, See-Kiong Ng, Tat-Seng Chua, and Yang Deng. 2025b. Knowledge boundary of large language models: A survey. Proceedings of ACL, pages 5131–5157.

Zhiyu Li et al. 2025c. Memos: A memory os for ai system. arXiv preprint arXiv:2507.03724.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. 2022. Holistic evaluation of language models. arXiv preprint arXiv:2211.09110.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. Truthfulqa: Measuring how models mimic human falsehoods. Proceedings ofACL.

Tennison Liu and Mihaela van der Schaar. 2025. Position: Truly self-improving agents require intrinsic metacognitive learning. Proceedings ofICML.

Yuhan Liu, Michael JQ Zhang, and Eunsol Choi. 2025. User feedback in human-llm dialogues: A lens to understand users but noisy as a learning signal. Proceedings ofEMNLP, pages 2666–2681.

Chang Ma, Junlei Zhang, Zhihao Zhu, Cheng Yang, Yujiu Yang, Yaohui Jin, Zhenzhong Lan, Lingpeng Kong, and Junxian He. 2024. Agentboard: An analytical evaluation board of multi-turn llm agents. In Advances in Neural Information Processing Systems (NeurIPS).

Putra Manggala, Atalanti Mastakouri, Elke Kirschbaum, Shiva Prasad Kasiviswanathan, and Aaditya Ramdas. 2025. Qa-calibration of language model confidence scores. Proceedings ofICLR.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744.

Charles Packer, Sarah Wooders, Kevin Lin, et al. 2023. Memgpt: Towards llms as operating systems. arXiv preprint arXiv:2310.08560.

John C. Platt. 2000. Probabilities for sv machines. In Advances in Large-Margin Classifiers, pages 61–74. MIT Press.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728– 53741.

Maohao Shen, Subhro Das, Kristjan Greenewald, Prasanna Sattigeri, Gregory Wornell, and Soumya Ghosh. 2024. Thermometer: Towards universal calibration for large language models. Proceedings of ICML.

Mirac Suzgun, Nathan Scales, Nathanael Sch"arli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, and Jason Wei. 2022. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261.

Zhen Tan et al. 2025. In prospect and retrospect: Reflective memory management for long-term personalized dialogue agents. Proceedings ofACL, pages 8416– 8439.

Jeremias Traub, Till J. Bungert, Carsten T. Lüth, Michael Baumgartner, Klaus H. Maier-Hein, Lena Maier-Hein, and Paul F. Jäger. 2024. Overcoming common flaws in the evaluation of selective classification systems. Proceedings ofNeurIPS, 37.

Jason Wei et al. 2024. Measuring short-form factuality in large language models. OpenAI Technical Report.

Bingbing Wen, Jihan Yao, Shangbin Feng, Chenjun Xu, Yulia Tsvetkov, Bill Howe, and Lucy Lu Wang. 2025. Know your limits: A survey of abstention in large language models. Transactions ofthe Associationfor Computational Linguistics, 13:529–556.

Jacob Whitehill, Paul Ruvolo, Tingfan Wu, Jacob Bergsma, and Javier Movellan. 2009. Whose vote should count more: Optimal integration of labels from labelers of unknown expertise. In Advances in Neural Information Processing Systems, volume 22, pages 2035–2043.

Fei Wu, Tao Shen, Thomas Bäck, Jingyuan Chen, Gang Huang, Yaochu Jin, Kun Kuang, Mengze Li, Cewu Lu, Jiaxu Miao, Yongwei Wang, Ying Wei, Fan Wu, Junchi Yan, Hongxia Yang, Yi Yang, Shengyu Zhang, Zhou Zhao, Yueting Zhuang, and Yunhe Pan. 2025. Knowledge-empowered, collaborative, and co-evolving AI models: The post-LLM roadmap. Engineering, 44:87–100.

Miao Xiong, Zhiyuan Hu, Xinyang Lu, Yifei Li, Jie Fu, Junxian He, and Bryan Hooi. 2024. Can llms express their uncertainty? an empirical evaluation of confidence elicitation in llms. In ICLR.

Tianyang Xu, Shujin Wu, Shizhe Diao, Xiaoze Liu, Xingyao Wang, Yangyi Chen, and Jing Gao. 2024. Sayself: Teaching llms to express confidence with self-reflective rationales. Proceedings of EMNLP.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. 2025. A-mem: Agentic memory for llm agents. Proceedings ofNeurIPS.

Fanjia Yan, Huanzhi Mao, Charlie Cheng-Jie Ji, Ion Stoica, Joseph E. Gonzalez, Tianjun Zhang, and Shishir G. Patil. 2024. Berkeley functioncalling leaderboard. https://gorilla. cs.berkeley.edu/blogs/8\_berkeley\_ function\_calling\_leaderboard.html.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. 2024. tau-bench: A benchmark for tool-agent-user interaction in real-world domains. arXiv preprint arXiv:2406.12045.

Xunjian Yin, Xu Zhang, Jie Ruan, and Xiaojun Wan. 2024. Benchmarking knowledge boundary for large language models: A different perspective on model evaluation. Proceedings ofACL.

Zhangyue Yin, Qiushi Sun, Qipeng Guo, Jiawen Wu, Xipeng Qiu, and Xuanjing Huang. 2023. Do large language models know what they don’t know? In Findings of ACL.

Bianca Zadrozny and Charles Elkan. 2001. Obtaining calibrated probability estimates from decision trees and naive bayesian classifiers. In Proceedings of ICML.

Hanning Zhang, Shizhe Diao, Yong Lin, Yi Fung, Qing Lian, Xingyao Wang, Yangyi Chen, Heng Ji, and Tong Zhang. 2024. R-tuning: Instructing large language models to say i don’t know. Proceedings of NAACL, pages 7113–7139.

Xuhui Zhou, Weiwei Sun, Qianou Ma, et al. 2026a. Mind the sim2real gap in user simulation for agentic tasks. arXiv preprint.

Zijian Zhou, Ao Qu, Zhaoxuan Wu, et al. 2026b. Mem1: Learning to synergize memory and reasoning for efficient long-horizon agents. Proceedings ofICLR.

## A Datasets

The primary pool contains 500 questions from 20 MMLU subjects, with 25 questions per subject. Each model is evaluated in 50 sessions of 50 attempted turns. The interaction records contain the question, answer, confidence, simulated user feedback, and reference-based correctness label.

MMLU-Pro and BBH provide held-out subject and task comparisons. GSM8K, HumanEval, and BFCL supply additional task-level confidence comparisons, while ReAct records support prediction on held-out questions. Dataset-specific scoring and sample sizes accompany the corresponding results.

## B Benchmark Details

## B.1 Benchmark Details: Task, Protocol, Question Pool, Personas, Metrics

Figure 5 in the main text summarizes task accuracy and profile-estimation performance. The supplementary Figure 6 displays all five levels and their baselines; Figure 7 summarizes per-answer confidence calibration.

Task formalisation. A session contains $\left( q _ { t } , a _ { t } , c _ { t } , f _ { t } , d _ { t } , y _ { t } \right)$ records. Outcomes $y _ { t }$ are used to score $p _ { d } ^ { * }$ and are hidden from noisy-feedback estimators. The domain identifier $d _ { t }$ groups observations for profile estimation; the answering prompt contains the question. Feedback is an utterance that may confirm, correct, or leave the answer’s validity unclear.

Session protocol. At each turn, the answering model receives a question and reports an answer and confidence. The simulated user then reacts according to its assigned persona and the current turn. Profile estimators consume the saved records; the next answer is not conditioned on their estimates. Each model contributes 50 sessions of 50 attempted turns.

Question pool and tiers. The scoring pool contains 20 MMLU subjects. For descriptive tier summaries, subjects are ordered by each model’s empirical accuracy and split into 6 strong, 7 medium, and 7 weak subjects. Threshold and randompartition analyses examine sensitivity to this grouping. The 232 numerical-answer turns in the additional math-word-problems domain of each Mistral/Llama archive are outside the 20-subject MCQ target and are excluded.

<table><tr><td>Subject</td><td>Llama 3-8B</td><td>Qwen Turbo</td><td>Qwen Max</td></tr><tr><td>Abstract Algebra</td><td>W</td><td>W</td><td>M</td></tr><tr><td>Anatomy</td><td>S</td><td>S</td><td>M</td></tr><tr><td>Astronomy</td><td>M</td><td>S</td><td>S</td></tr><tr><td>Business Ethics</td><td>S</td><td>M</td><td>S</td></tr><tr><td>Clinical Knowledge</td><td>S</td><td>S</td><td>S</td></tr><tr><td>College Biology</td><td>S</td><td>S</td><td>S</td></tr><tr><td>College Chemistry</td><td>M</td><td>M</td><td>W</td></tr><tr><td>College Physics</td><td>M</td><td>W</td><td>W</td></tr><tr><td>Computer Security</td><td>S</td><td>M</td><td>W</td></tr><tr><td>Conceptual Physics</td><td>W</td><td>S</td><td>S</td></tr><tr><td>Econometrics</td><td>M</td><td>W</td><td>M</td></tr><tr><td>Electrical Engineering</td><td>M</td><td>M</td><td>S</td></tr><tr><td>Formal Logic</td><td>W</td><td>W</td><td>W</td></tr><tr><td>Global Facts</td><td>W</td><td>W</td><td>W</td></tr><tr><td>High School Biology</td><td>S</td><td>M</td><td>M</td></tr><tr><td>High School Chemistry</td><td>M</td><td>W</td><td>W</td></tr><tr><td>High School Mathematics</td><td>W</td><td>S</td><td>M</td></tr><tr><td>High School Physics</td><td>M</td><td>M</td><td>W</td></tr><tr><td>Machine Learning</td><td>W</td><td>M</td><td>M</td></tr><tr><td>Moral Scenarios</td><td>W</td><td>W</td><td>M</td></tr></table>

Table 4: MMLU subjects in the scored records and empirical per-model tiers (S/M/W).

User personas. Each persona has three expertise subjects, a response style, and a tolerance parameter between .20 and .40. Within expertise, the prompt requests confirmation of correct answers and correction of incorrect answers; the simulator receives the answer key. Outside expertise, the prompt requests acceptance of correct answers and either uncertainty or acceptance of incorrect answers according to the tolerance probability. Acceptance need not contain an explicit evaluative phrase. The measured channel is the output of the fixed rule decoder on these utterances.

Auxiliary metrics. Beyond CBF (Eq. 1, computed once at the end of the 50-turn session), we report: (i) Hallucination Rate $\mathrm { H } \mathbf { R } = | \{ t : c _ { t } > 0 . 7 \land$ $a _ { t } \neq a _ { t } ^ { * } \} | / T ;$ (ii) Expected Calibration Error following Guo et al. (2017) with M = 10 equal-width bins, $\begin{array} { r } { \mathrm { E C E } = \sum _ { m = 1 } ^ { M } \frac { | B _ { m } | } { T } | \mathrm { a c c } ( B _ { m } ) - \mathrm { c o n f } ( B _ { m } ) | } \end{array}$ (iii) Brier $\begin{array} { r } { \mathrm { S c o r e } = { \frac { 1 } { T } } \sum _ { t } ( c _ { t } - \mathcal { Y } [ a _ { t } = a _ { t } ^ { * } ] ) ^ { 2 } } \end{array}$

## B.2 MMLU Subject Selection

The 20 subjects comprise 13 STEM subjects, four in MMLU’s “Other” category, two in humanities, and one in social sciences. The following table illustrates the model-relative tiers.

## B.3 User Persona Details

Each simulated user persona is defined by the following parameters:

<table><tr><td></td><td>L0</td><td>L1 NoMem ZS-SK</td><td>L2 LSA</td><td>L2 UF-PDC</td><td>L2 Platt</td><td>L2</td><td>L2 HistBin Bayes</td><td>L2 EM</td><td>L2 Global</td><td>L2 DS</td><td>L2 GLAD</td><td>L3 Oracle</td><td></td><td>L4 Stats</td></tr><tr><td>Mistral-7B</td><td>0.745</td><td>0.677</td><td>0.638</td><td>0.734</td><td>0.734</td><td>0.734</td><td>0.748</td><td>0.748</td><td>0.691</td><td></td><td>0.742</td><td>0.731</td><td>1.000</td><td>1.000</td></tr><tr><td>Llama-3-8B</td><td>0.736</td><td>0.734</td><td>0.549</td><td>0.739</td><td>0.739</td><td>0.739</td><td>0.747</td><td>0.748</td><td></td><td>0.717</td><td>0.748</td><td>0.734</td><td>1.000</td><td>1.000</td></tr><tr><td>GLM-5</td><td>0.542</td><td>0.904</td><td>0.568</td><td>0.595</td><td>0.595</td><td>0.595</td><td>0.575</td><td>0.572</td><td>0.774</td><td></td><td>0.575</td><td>0.594</td><td>1.000</td><td>1.000</td></tr><tr><td>DeepSeek-V3.2</td><td>0.596</td><td>0.834</td><td>0.761</td><td>0.638</td><td>0.638</td><td>0.638</td><td>0.628</td><td>0.626</td><td>0.769</td><td></td><td>0.627</td><td>0.636</td><td>1.000</td><td>1.000</td></tr><tr><td>Qwen-Turbo</td><td>0.688</td><td>0.772</td><td>0.745</td><td>0.698</td><td>0.698</td><td>0.698</td><td>0.707</td><td>0.711</td><td>0.751</td><td></td><td>0.703</td><td>0.694</td><td>1.000</td><td>1.000</td></tr><tr><td>Qwen2.5-7B</td><td>0.740</td><td>0.670</td><td>0.732</td><td>0.741</td><td>0.741</td><td>0.741</td><td>0.751</td><td>0.753</td><td>0.715</td><td></td><td>0.748</td><td>0.738</td><td>1.000</td><td>1.000</td></tr><tr><td>Qwen-Max</td><td>0.680</td><td>0.789</td><td>0.757</td><td>0.666</td><td>0.666</td><td>0.666</td><td>0.679</td><td>0.680</td><td>0.660</td><td></td><td>0.672</td><td>0.667</td><td>1.000</td><td>1.000</td></tr><tr><td>Qwen-Plus</td><td>0.646</td><td>0.806</td><td>0.558</td><td>0.672</td><td>0.672</td><td>0.672</td><td>0.667</td><td>0.667</td><td>0.750</td><td></td><td>0.670</td><td>0.670</td><td>1.000</td><td>1.000</td></tr><tr><td>Qwen3.5-27B</td><td>0.543</td><td>0.886</td><td>0.571</td><td>0.595</td><td>0.595</td><td>0.595</td><td>0.576</td><td>0.573</td><td>0.775</td><td></td><td>0.577</td><td>0.594</td><td>1.000</td><td>1.000</td></tr><tr><td>GLM-5.1</td><td>0.557</td><td>0.608</td><td>0.473</td><td>0.579</td><td>0.579</td><td>0.579</td><td>0.577</td><td>0.578</td><td>0.711</td><td></td><td>0.580</td><td>0.578</td><td>1.000</td><td>1.000</td></tr><tr><td>GPT-5.4-mini</td><td>0.680</td><td>0.764</td><td>0.747</td><td>0.686</td><td>0.686</td><td>0.686</td><td>0.688</td><td>0.686</td><td></td><td>0.700</td><td>0.686</td><td>0.685</td><td>1.000</td><td>1.000</td></tr><tr><td>GPT-5.4</td><td>0.642</td><td>0.797</td><td>0.762</td><td>0.647</td><td>0.647</td><td>0.647</td><td>0.650</td><td>0.651</td><td></td><td>0.689</td><td>0.648</td><td>0.647</td><td>1.000</td><td>1.000</td></tr><tr><td>GPT-5.5</td><td>0.570</td><td>0.858</td><td>0.844</td><td>0.612</td><td>0.612</td><td>0.612</td><td>0.603</td><td>0.605</td><td></td><td>0.800</td><td>0.601</td><td>0.609</td><td>1.000</td><td>1.000</td></tr><tr><td>Claude-Opus-4.7</td><td>0.582</td><td>0.742</td><td>0.706</td><td>0.624</td><td>0.624</td><td>0.624</td><td>0.613</td><td>0.609</td><td></td><td>0.776</td><td>0.613</td><td>0.621</td><td>1.000</td><td>1.000</td></tr><tr><td>Claude-Sonnet-4.6</td><td>0.560</td><td>0.801</td><td>0.739</td><td>0.592</td><td>0.592</td><td>0.592</td><td>0.583</td><td>0.579</td><td></td><td>0.772</td><td>0.589</td><td>0.590</td><td>1.000</td><td>1.000</td></tr><tr><td>Claude-Opus-4.6</td><td>0.563</td><td>0.771</td><td>0.753</td><td>0.620</td><td>0.620</td><td>0.620</td><td>0.601</td><td>0.592</td><td></td><td>0.779</td><td>0.599</td><td>0.616</td><td>1.000</td><td>1.000</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.562</td><td>0.316</td><td>0.414</td><td>0.598</td><td>0.598</td><td>0.598</td><td>0.590</td><td>0.592</td><td></td><td>0.780</td><td>0.592</td><td>0.596</td><td>1.000</td><td>1.000</td></tr><tr><td>Qwen3.6-Plus</td><td>0.589</td><td>0.719</td><td>0.821</td><td>0.616</td><td>0.616</td><td>0.616</td><td>0.614</td><td>0.616</td><td></td><td>0.777</td><td>0.613</td><td>0.612</td><td>1.000</td><td>1.000</td></tr><tr><td colspan="14">0.25 0.50 0.75 Session-mean CBF</td></tr></table>

Figure 6: Full model-by-baseline CBF matrix. All 18 models and the complete five-level design.  
![](images/4c54c5fa2a5f574f89a28fabb89d1492109741ef724751b41cb570ba22de88cb.jpg)  
Figure 7: Recorded-confidence reliability curves. Bin-level mean confidence and answer accuracy; the last bin includes confidence 1.

• Expertise: three subjects stored with each persona.

• Feedback behavior: confirmation, correction, or uncertainty, conditioned on expertise and answer correctness.

• Style: natural-language presentation of the requested behavior.

The resulting feedback is decoded using the same rule classifier as the profile estimators. Appendix B.3 reports the observed channel, including missing labels.

Measured feedback channel. Coverage is decoded labels / valid answers. Agreement is labels matching the answer key / decoded labels. Confirmation is positive labels on correct answers / all correct answers, and correction is negative labels on incorrect answers / all incorrect answers. Rates are percentages and measure rule-decoded stance. The 353 feedback API-error records remain in the valid-answer target with missing labels.

## C Baseline Implementations

## C.1 Baseline Implementations

The original ten baselines retain their five-level organization. Global, DS, and GLAD-style add comparisons of overall-level estimation and explicit feedback-reliability modeling.

L0 — NoMemory (B1). Constant $\hat { p } _ { d } = . 5$ for domain CBF; recorded confidence for per-query metrics.

L1 — ZS-SK (B11). The model estimates domain accuracies without interaction history.

L2 — LSA (B6). The model estimates domain accuracies from its question and natural-language feedback history.

L2 — UF-PDC (B3). Domain proportions of strictly decoded positive feedback.

L2 — Platt/HistBin (B7/B8). Online Platt scaling (Platt, 2000) and histogram binning (Zadrozny and Elkan, 2001) recalibrate query confidence; their domain self-assessment interface returns the same decoded-label counts.

L2 — Bayes-Domain (B9). A Beta(1,1) prior gives (s<sub>d</sub> + 1)/(n<sub>d</sub> + 2).

L2 — EM-Joint (B10). A symmetric reliability model with domain correctness prevalence and in/out expertise reliability.

L3 — OraclePDC (B2). Exact counts of verified outcomes on the scored domain support.

L4 — GT-SelfStats (B12). The model reports supplied empirical statistics.

Additional L2 controls. Global assigns the pooled feedback Beta mean to every domain. DS uses state-specific confusion probabilities; GLADstyle uses additive ability and difficulty.

Information conditions. The ladder compares information available for a domain-profile estimate. LSA performs in-context aggregation; Count, Beta, EM-Joint, DS, and GLAD-style instantiate alternative estimators. L1 and L4 provide prior and supplied-statistic references. This grouping does not constitute a direct evaluation of the memory architectures cited in Related Work.

## C.2 Baseline Implementation Details

B1: NoMemory. Its domain self-assessment function returns .5. Answer confidence is a separate output and does not determine this profile.

B2: OraclePDC. The evaluator maintains verified correct/total counts per recorded domain. Using the same outcomes and support as the target yields CBF 1 exactly.

B3: User-Feedback PDC. The classifier maps explicit positive/negative feedback to 1/0. Uncertain and neutral responses are unobserved. Platt, HistBin, Bayes, EM-Joint, DS, and GLAD-style use this same stream.

B6: LSA. At the session endpoint, the model reads domain-indexed question/feedback summaries and returns one probability per domain. The saved representation retains up to 80 question characters and 120 feedback characters per turn. Its complete response is parsed once; a valid profile is not selected by its CBF.

B10: EM-Joint. The latent outcome $y _ { t }$ has prior $p _ { d _ { t } }$ and symmetric agreement probability $r _ { h _ { t } }$ for expertise state $h _ { t }$ . With a decoded label $z _ { t } ,$ the E-step computes its posterior outcome; smoothed M-steps update domain prevalence and reliability. Initialization uses $p _ { d } = . 5$ and $r _ { \mathrm { i n } } = . 7 , r _ { \mathrm { o u t } } = . 5 .$ followed by at most 30 iterations with tolerance 10<sup>−4</sup>. Missing-feedback domains retain .5.

B11: ZS-SK. L1 estimates domain accuracy without session history. Each 20-domain prior is evaluated on the same eligible session targets as the other levels. The JSON response format appears in Appendix G.

<table><tr><td>Model</td><td>State</td><td>Valid</td><td>Decoded</td><td>Coverage</td><td>Agreement</td><td>Confirm</td><td>Correct</td></tr><tr><td>Mistral-7B</td><td>in</td><td>355</td><td>221</td><td>62.3</td><td>91.9</td><td>86.7</td><td>35.6</td></tr><tr><td>Mistral-7B</td><td>out</td><td>1908</td><td>30</td><td>1.6</td><td>56.7</td><td>0.0</td><td>1.5</td></tr><tr><td>Llama-3-8B</td><td>in</td><td>336</td><td>260</td><td>77.4</td><td>95.8</td><td>90.9</td><td>46.5</td></tr><tr><td>Llama-3-8B</td><td>out</td><td>1932</td><td>45</td><td>2.3</td><td>62.2</td><td>0.0</td><td>3.4</td></tr><tr><td>GLM-5</td><td>in</td><td>246</td><td>218</td><td>88.6</td><td>99.5</td><td>89.5</td><td>50.0</td></tr><tr><td>GLM-5</td><td>out</td><td>1359</td><td>20</td><td>1.5</td><td>5.0</td><td>0.1</td><td>0.0</td></tr><tr><td>DeepSeek-V3.2</td><td>in</td><td>315</td><td>260</td><td>82.5</td><td>99.6</td><td>88.5</td><td>35.1</td></tr><tr><td>DeepSeek-V3.2</td><td>out</td><td>1516</td><td>19</td><td>1.3</td><td>15.8</td><td>0.1</td><td>1.2</td></tr><tr><td>Qwen-Turbo</td><td>in</td><td>392</td><td>305</td><td>77.8</td><td>100.0</td><td>87.1</td><td>49.5</td></tr><tr><td>Qwen-Turbo</td><td>out</td><td>1966</td><td>41</td><td>2.1</td><td>17.1</td><td>0.0</td><td>1.5</td></tr><tr><td>Qwen2.5-7B</td><td>in</td><td>376</td><td>268</td><td>71.3</td><td>100.0</td><td>89.7</td><td>40.8</td></tr><tr><td>Qwen2.5-7B</td><td>out</td><td>2086</td><td>30</td><td>1.4</td><td>23.3</td><td>0.0</td><td>0.8</td></tr><tr><td>Qwen-Max</td><td>in</td><td>371</td><td>121</td><td>32.6</td><td>97.5</td><td>19.8</td><td>76.9</td></tr><tr><td>Qwen-Max</td><td>out</td><td>1997</td><td>37</td><td>1.9</td><td>62.2</td><td>0.8</td><td>2.4</td></tr><tr><td>Qwen-Plus</td><td>in</td><td>321</td><td>277</td><td>86.3</td><td>99.3</td><td>88.9</td><td>68.0</td></tr><tr><td>Qwen-Plus</td><td>out</td><td>1917</td><td>32</td><td>1.7</td><td>15.6</td><td>0.1</td><td>0.9</td></tr><tr><td>Qwen3.5-27B</td><td>in</td><td>269</td><td>235</td><td>87.4</td><td>100.0</td><td>88.8</td><td>54.5</td></tr><tr><td>Qwen3.5-27B</td><td>out</td><td>1402</td><td>24</td><td>1.7</td><td>4.2</td><td>0.1</td><td>0.0</td></tr><tr><td>GLM-5.1</td><td>in</td><td>253</td><td>166</td><td>65.6</td><td>99.4</td><td>67.5</td><td>36.8</td></tr><tr><td>GLM-5.1</td><td>out</td><td>1303</td><td>24</td><td>1.8</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>GPT-5.4-mini</td><td>in</td><td>389</td><td>109</td><td>28.0</td><td>100.0</td><td>29.0</td><td>25.2</td></tr><tr><td>GPT-5.4-mini</td><td>out</td><td>2078</td><td>13</td><td>0.6</td><td>15.4</td><td>0.0</td><td>0.4</td></tr><tr><td>GPT-5.4</td><td>in</td><td>410</td><td>132</td><td>32.2</td><td>99.2</td><td>32.9</td><td>27.0</td></tr><tr><td>GPT-5.4</td><td>out</td><td>2062</td><td>15</td><td>0.7</td><td>6.7</td><td>0.0</td><td>0.3</td></tr><tr><td>GPT-5.5</td><td>in</td><td>417</td><td>357</td><td>85.6</td><td>100.0</td><td>88.1</td><td>54.8</td></tr><tr><td>GPT-5.5</td><td>out</td><td>2083</td><td>34</td><td>1.6</td><td>20.6</td><td>0.1</td><td>4.0</td></tr><tr><td>Claude-Opus-4.7</td><td>in</td><td>173</td><td>147</td><td>85.0</td><td>99.3</td><td>87.9</td><td>50.0</td></tr><tr><td>Claude-Opus-4.7</td><td>out</td><td>994</td><td>22</td><td>2.2</td><td>31.8</td><td>0.3</td><td>4.2</td></tr><tr><td>Claude-Sonnet-4.6</td><td>in</td><td>142</td><td>109</td><td>76.8</td><td>100.0</td><td>80.3</td><td>46.7</td></tr><tr><td>Claude-Sonnet-4.6</td><td>out</td><td>772</td><td>10</td><td>1.3</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Claude-Opus-4.6</td><td>in</td><td>144</td><td>128</td><td>88.9</td><td>100.0</td><td>90.5</td><td>57.1</td></tr><tr><td>Claude-Opus-4.6</td><td>out</td><td>805</td><td>14</td><td>1.7</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>DeepSeek-V4-Pro</td><td>in</td><td>352</td><td>296</td><td>84.1</td><td>100.0</td><td>87.1</td><td>46.2</td></tr><tr><td>DeepSeek-V4-Pro</td><td>out</td><td>2100</td><td>40</td><td>1.9</td><td>2.5</td><td>0.1</td><td>0.0</td></tr><tr><td>Qwen3.6-Plus</td><td>in</td><td>367</td><td>308</td><td>83.9</td><td>100.0</td><td>87.3</td><td>54.1</td></tr><tr><td>Qwen3.6-Plus</td><td>out</td><td>1990</td><td>45</td><td>2.3</td><td>15.6</td><td>0.2</td><td>1.6</td></tr></table>

Table 5: Observed feedback channel inside/outside persona expertise. Confirm = positive labels on correct answers / all correct answers; Correct = negative labels on incorrect answers / all incorrect answers. These are decoded stance rates. Outside expertise, neutral follow-up questions can express acceptance without a decoded positive label. Qwen-Max has 58/293 in-expertise correct answers with positive labels and 235/293 with neutral labels; 59 neutral replies contain that was accurate, which the fixed rules miss. Its correction count is 60/78. The low confirmation rate therefore includes decoder misses; these records do not isolate a simulator-family effect.

<table><tr><td>Model</td><td>DS-Beta</td><td>GLAD-style-Beta</td></tr><tr><td>Qwen2.5-7B</td><td>-0.083</td><td>-0.151</td></tr><tr><td>GPT-5.4-mini</td><td>-0.010</td><td>-0.048</td></tr><tr><td>GPT-5.4</td><td>-0.071</td><td>0.006</td></tr><tr><td>GPT-5.5</td><td>-0.056</td><td>0.036</td></tr><tr><td>DeepSeek-V4-Pro</td><td>-0.041</td><td>0.053</td></tr></table>

Table 6: Paired pooled CBF differences on shared labels and domains. Question-clustered intervals appear in Table 13.

B12: GT-SelfStats. The empirical accuracy for each eligible domain of the current session is supplied directly, and the model reports the same values in a structured profile.

## D Statistical Analysis

## D.1 Sample Size and Paired Comparisons

Statistical unit. A model’s session-level score averages its eligible sessions. Model-level comparisons pair methods on the same records; pooled comparisons refit each method on all valid records of that model.

Uncertainty. Session resampling quantifies variation among the observed sessions. Questioncluster resampling groups repeated questions within domain and refits the estimators, preserving the dependence induced by reused questions. The model inventory is a fixed selection rather than a random sample from a model population.

Sample selection. The 18-model panel includes an eight-model frontier subset. Method comparisons are descriptive across this fixed inventory; changing the subset changes the target population.

Detection limits. Sparse per-domain outcomes limit the precision of session profiles. Larger heldout task sets, rather than post-hoc power estimates from observed effects, are needed to resolve small predictive differences.

## D.2 Statistical Robustness

We report NoMemory CBF with session-bootstrap standard errors (2,000 draws, seed 20261011) for all 18 models. Confidence-based metrics are point estimates over valid answer records and include both endpoints of [0, 1].

<table><tr><td>Model</td><td>NoMem CBF</td><td>ECE</td><td>HR</td><td>Brier</td></tr><tr><td>Mistral-7B</td><td>0.745±0.008</td><td>0.582</td><td>0.583</td><td>0.581</td></tr><tr><td>Llama-3-8B</td><td>0.736±0.007</td><td>0.357</td><td>0.423</td><td>0.368</td></tr><tr><td>GLM-5</td><td>0.542±0.005</td><td>0.039</td><td>0.042</td><td>0.042</td></tr><tr><td>DeepSeek-V3.2</td><td>0.596±0.006</td><td>0.023</td><td>0.107</td><td>0.095</td></tr><tr><td>Qwen-Turbo</td><td>0.688±0.006</td><td>0.188</td><td>0.230</td><td>0.214</td></tr><tr><td>Qwen2.5-7B</td><td>0.740±0.007</td><td>0.259</td><td>0.374</td><td>0.297</td></tr><tr><td>Qwen-Max</td><td>0.680±0.007</td><td>0.167</td><td>0.209</td><td>0.190</td></tr><tr><td>Qwen-Plus</td><td>0.646±0.008</td><td>0.144</td><td>0.177</td><td>0.161</td></tr><tr><td>Qwen3.5-27B</td><td>0.543±0.005</td><td>0.033</td><td>0.041</td><td>0.040</td></tr><tr><td>GLM-5.1</td><td>0.557±0.011</td><td>0.040</td><td>0.056</td><td>0.053</td></tr><tr><td>GPT-5.4-mini</td><td>0.680±0.006</td><td>0.199</td><td>0.246</td><td>0.215</td></tr><tr><td>GPT-5.4</td><td>0.642±0.006</td><td>0.132</td><td>0.158</td><td>0.148</td></tr><tr><td>GPT-5.5</td><td>0.570±0.005</td><td>0.032</td><td>0.056</td><td>0.056</td></tr><tr><td>Claude-Opus-4.7</td><td>0.582±0.008</td><td>0.022</td><td>0.060</td><td>0.078</td></tr><tr><td>Claude-Sonnet-4.6</td><td>0.560±0.010</td><td>0.052</td><td>0.048</td><td>0.064</td></tr><tr><td>Claude-Opus-4.6</td><td>0.563±0.007</td><td>0.020</td><td>0.051</td><td>0.054</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.562±0.004</td><td>0.047</td><td>0.069</td><td>0.064</td></tr><tr><td>Qwen3.6-Plus</td><td>0.589±0.005</td><td>0.068</td><td>0.089</td><td>0.080</td></tr></table>

Table 7: NoMemory CBF and its session-bootstrap standard error; valid-answer confidence metrics.
<table><tr><td>Comparison</td><td>n</td><td>Mean difference</td></tr><tr><td>ZS - NoMem</td><td>18</td><td>0.124</td></tr><tr><td>ZS – LSA</td><td>18</td><td>0.073</td></tr><tr><td>LSA - NoMem</td><td>18</td><td>0.051</td></tr><tr><td>EM – NoMem</td><td>18</td><td>0.020</td></tr><tr><td>Platt — NoMem</td><td>18</td><td>0.024</td></tr></table>

Table 8: Paired mean differences over models with both estimates.

The errors reflect uncertainty conditional on the retained session pool. Paired differences should use their own resampled differences, rather than a threshold based on one method’s standard error.

Cross-model summaries. Table 8 averages paired CBF differences only where both methods are available. Table 9 reports the fixed frontier/nonfrontier descriptive contrast with its sample counts.

Cross-model averages summarize different empirical targets. A larger raw CBF can result from a target profile closer to the constant reference, so these contrasts are not an ordering of model competence.

<table><tr><td>Method</td><td>nF/no</td><td>Frontier – Other</td></tr><tr><td>NoMem</td><td>8/10</td><td>-0.054</td></tr><tr><td>ZS</td><td>8/10</td><td>-0.047</td></tr><tr><td>LSA</td><td>8/10</td><td>0.088</td></tr><tr><td>UF</td><td>8/10</td><td>-0.041</td></tr><tr><td>Platt</td><td>8/10</td><td>-0.041</td></tr><tr><td>HistBin</td><td>8/10</td><td>-0.041</td></tr><tr><td>Bayes</td><td>8/10</td><td>-0.048</td></tr><tr><td>EM</td><td>8/10</td><td>-0.049</td></tr></table>

Table 9: Descriptive raw-CBF contrasts for the fixed model subsets.

Normalized CBF is computed from the main table's session means using Eq. 2. NoMem is 0 and exact L3 counting is 1 for every model. It reports error reduction relative to the .5 profile.

## D.3 Normalized Hallucination Rate

We define normalized HR as $\operatorname* { P r } ( c > . 7 \ | \ y =$ $0 ) = \mathrm { H R } / \operatorname* { P r } ( y = 0 )$ . It is the fraction of errors assigned high confidence; a value of 1 means every error is high-confidence.

This conditional error fraction does not alone determine confidence discrimination: comparing confidence on correct answers is also necessary. Reliability curves in Figure 7 provide the complementary view.

## D.4 CBF vs. Rank Correlation

A natural alternative to CBF is Kendall’s τ over domain accuracy rankings. However, τ is undefined when the baseline produces constant predictions; e.g., NoMemory estimates 0.5 for all domains. CBF avoids this by measuring absolute deviation rather than rank correlation, making it applicable to all baselines including the constantprediction case. We recommend reporting both CBF and τ when applicable. Figure 9 visualizes the three-way relationship between MMLU accuracy, recorded-confidence ECE, and NoMemory CBF across the full n=18 cross-baseline panel, illustrating why CBF and ECE are non-redundant.

## D.5 Cross-Baseline Bootstrap Significance

Paired question-cluster intervals are used for the matched pooled comparisons in Table 13. All estimators are refitted within each of 400 resamples. The intervals describe uncertainty conditional on the recorded questions; no blanket significance threshold is assigned across models, metrics, and baselines.

## E Robustness Ablations

## E.1 Additional Robustness Ablations

This appendix contains additional ablation experiments orthogonal to the main diagnostic ladder.

A1: Information and estimator controls. Tables 2 and 3 compare the common endpoint target. UF-PDC, Platt, and HistBin have identical domain outputs, while their query-confidence interfaces differ.

A2: Feedback noise and coverage. Appendix E.7 reports five noise channels at 12.5%, 50%, and 100% coverage for five models, with 100 retained repetitions per condition.

A3: Session length. Prefix estimates are scored against the fixed end-of-session target, allowing the effect of additional feedback to be separated from changes in the target itself. Figure 10 follows this comparison over the 50-turn sessions.

A4: Prior assessment on MMLU-Pro. For each of eight frontier models, we sampled 15 questions in each of ten MMLU-Pro categories (150 questions/model) and queried domain priors on the same category names. Table 14 reports actual and estimated accuracy. These categories broaden the evaluation beyond the primary question pool.

Mean MMLU-Pro CBF is 0.870 across the eight models. GPT-5.5 and Claude-Opus-4.7 report conservative estimates on several categories, whereas individual categories exhibit larger errors. The table therefore supports a subject-level description beyond the primary MMLU pool.

A4b: Prior assessment on BBH. For the same eight models, we sampled 15 questions from each of ten BIG-Bench Hard tasks (Suzgun et al., 2022) and queried priors on those task names. Table 15 reports the complete task-level comparison. The stored scorer accepts a target string or its cleaned form when it occurs in the generated text.

Mean BBH CBF is 0.825 across the same eight models, compared with 0.870 on MMLU-Pro. The full task table reveals errors obscured by these averages: for example, Claude-Sonnet-4.6 estimates .80 on word sorting but obtains .00. The following task comparisons describe this fixed evaluation sample.

Together, MMLU-Pro and BBH show that priorprofile errors vary with task content. Their distinct task labels broaden the evaluation, but public benchmark differences do not identify whether the underlying tasks were seen during training.

A5: Reliability estimators. We compare adaptations of two established crowdsourcing models on the same decoded-label records.

Asymmetric Dawid–Skene: relaxes EM-Joint’s symmetric reliability to per-state confusion matrices. For each rater state $s \ \in \ \{ \mathrm { i n } , \mathrm { o u t } \}$ , we estimate $\mathrm { T P R } _ { s } ~ = ~ P ( \mathrm { d e c o d e d } ~ = ~ 1 ~ |$ correct = 1, s) and FPR<sub>s</sub> = P(decoded = 1 | correct = 0, s). EM with E-step posterior P(correct =

<table><tr><td>Model</td><td>L1</td><td>LSA</td><td>Count</td><td>Beta</td><td>EM</td><td>Global</td><td>DS</td><td>GLAD-style</td><td>L4</td></tr><tr><td>Mistral-7B</td><td>-0.267</td><td>-0.421</td><td>-0.042</td><td>0.011</td><td>0.012</td><td>-0.213</td><td>-0.011</td><td>-0.055</td><td>1.000</td></tr><tr><td>Llama-3-8B</td><td>-0.008</td><td>-0.712</td><td>0.009</td><td>0.039</td><td>0.043</td><td>-0.072</td><td>0.043</td><td>-0.009</td><td>1.000</td></tr><tr><td>GLM-5</td><td>0.791</td><td>0.057</td><td>0.117</td><td>0.073</td><td>0.066</td><td>0.508</td><td>0.072</td><td>0.113</td><td>1.000</td></tr><tr><td>DeepSeek-V3.2</td><td>0.590</td><td>0.407</td><td>0.103</td><td>0.079</td><td>0.073</td><td>0.427</td><td>0.077</td><td>0.099</td><td>1.000</td></tr><tr><td>Qwen-Turbo</td><td>0.269</td><td>0.180</td><td>0.029</td><td>0.058</td><td>0.072</td><td>0.202</td><td>0.047</td><td>0.017</td><td>1.000</td></tr><tr><td>Qwen2.5-7B</td><td>-0.266</td><td>-0.031</td><td>0.006</td><td>0.042</td><td>0.051</td><td>-0.094</td><td>0.031</td><td>-0.006</td><td>1.000</td></tr><tr><td>Qwen-Max</td><td>0.342</td><td>0.240</td><td>-0.041</td><td>-0.002</td><td>0.003</td><td>-0.062</td><td>-0.025</td><td>-0.040</td><td>1.000</td></tr><tr><td>Qwen-Plus</td><td>0.453</td><td>-0.247</td><td>0.076</td><td>0.061</td><td>0.062</td><td>0.295</td><td>0.070</td><td>0.069</td><td>1.000</td></tr><tr><td>Qwen3.5-27B</td><td>0.751</td><td>0.061</td><td>0.115</td><td>0.073</td><td>0.067</td><td>0.507</td><td>0.075</td><td>0.112</td><td>1.000</td></tr><tr><td>GLM-5.1</td><td>0.115</td><td>-0.190</td><td>0.050</td><td>0.044</td><td>0.047</td><td>0.347</td><td>0.052</td><td>0.048</td><td>1.000</td></tr><tr><td>GPT-5.4-mini</td><td>0.263</td><td>0.208</td><td>0.018</td><td>0.023</td><td>0.017</td><td>0.062</td><td>0.019</td><td>0.016</td><td>1.000</td></tr><tr><td>GPT-5.4</td><td>0.434</td><td>0.335</td><td>0.016</td><td>0.022</td><td>0.025</td><td>0.131</td><td>0.017</td><td>0.014</td><td>1.000</td></tr><tr><td>GPT-5.5</td><td>0.670</td><td>0.637</td><td>0.099</td><td>0.076</td><td>0.081</td><td>0.535</td><td>0.073</td><td>0.091</td><td>1.000</td></tr><tr><td>Claude-Opus-4.7</td><td>0.381</td><td>0.296</td><td>0.100</td><td>0.073</td><td>0.064</td><td>0.464</td><td>0.074</td><td>0.091</td><td>1.000</td></tr><tr><td>Claude-Sonnet-4.6</td><td>0.549</td><td>0.407</td><td>0.073</td><td>0.051</td><td>0.044</td><td>0.482</td><td>0.066</td><td>0.067</td><td>1.000</td></tr><tr><td>Claude-Opus-4.6</td><td>0.477</td><td>0.435</td><td>0.130</td><td>0.087</td><td>0.068</td><td>0.495</td><td>0.082</td><td>0.123</td><td>1.000</td></tr><tr><td>DeepSeek-V4-Pro</td><td>-0.561</td><td>-0.338</td><td>0.081</td><td>0.063</td><td>0.068</td><td>0.498</td><td>0.068</td><td>0.077</td><td>1.000</td></tr><tr><td>Qwen3.6-Plus</td><td>0.316</td><td>0.565</td><td>0.066</td><td>0.061</td><td>0.068</td><td>0.459</td><td>0.059</td><td>0.058</td><td>1.000</td></tr></table>

Table 10: Baseline-adjusted CBF on the main session target.  
![](images/3b23866412da43a5b3f55a9888c0d7c3e881568dc3179fa3a0e10b98d20761ce.jpg)  
Figure 8: Signed domain errors for L1, Global, and Beta across all 18 models on pooled targets. Each model contributes six strong, seven medium, and seven weak subjects ranked by observed accuracy. Boxes show quartiles and medians; whiskers extend to 1.5 IQR. Positive errors denote overestimation.

<table><tr><td>Model</td><td>Error rate</td><td>HR</td><td>Normalized HR</td></tr><tr><td>Mistral-7B</td><td>0.584</td><td>0.583</td><td>0.999</td></tr><tr><td>Llama-3-8B</td><td>0.423</td><td>0.423</td><td>1.000</td></tr><tr><td>Qwen-Turbo</td><td>0.245</td><td>0.230</td><td>0.939</td></tr><tr><td>Qwen-Max</td><td>0.212</td><td>0.209</td><td>0.990</td></tr></table>

Table 11: High-confidence error prevalence on valid answers.

![](images/938d47d48b2d6ac76f01b0667986642ba6e29f6c426256bbe8dd036fb7b8369c.jpg)  
Figure 9: Accuracy, confidence calibration, and domain-profile error. Answer accuracy versus ECE; bubble area represents NoMemory CBF.

1 | decoded, $s , { \hat { p } } .$ TPR, FPR) and M-step weighted-MLE for both $\hat { p } ( d )$ and the four reliability parameters.

GLAD-style: per-domain difficulty $\delta _ { d }$ and perstate ability $\alpha _ { s } .$ , with reliability $\sigma ( \alpha _ { s } - \delta _ { d } )$ (logistic). EM updates over $\hat { p } ( d ) , \delta _ { d } , \alpha _ { \mathrm { i n } } , \alpha _ { \mathrm { o u t } } .$

DS and GLAD-style are adaptations of established crowdsourcing models (Dawid and Skene, 1979; Whitehill et al., 2009). DS fits state-specific true/false positive rates with 50 EM iterations, initialized at $\left( . 7 5 , . 2 5 \right)$ in-expertise and $\left( . 5 5 , . 4 5 \right)$ outside expertise. GLAD-style uses $r _ { h d } = \sigma ( \alpha _ { h } - \delta _ { d } )$ initialized at $\alpha = ( 1 , 1 ) , \delta = 0 , p _ { d } = . 5$ , with 80 generalized-EM iterations and step size .05. Ability/difficulty are clipped to [−3, 3]. This additive parameterization differs from original GLAD. The same observed expertise state is supplied to both methods. Their gains depend on initialization and the information available in a single-label setting.

Likelihoods and updates. Dawid–Skene (DS) (Dawid and Skene, 1979) models asymmetric feedback errors. Let $h _ { t } \in \{ \mathrm { i n , o u t } \}$ be the assigned expertise state, available to DS and GLAD-style, and let $\theta _ { h 1 }$ and $\theta _ { h 0 }$ be the true- and false-positive rates. Define $L _ { t k } \ = \ \theta _ { h _ { t } k } ^ { z _ { t } } ( 1 \ : - \ : \theta _ { h _ { t } k } ) ^ { 1 - z _ { t } }$ On observed feedback, the likelihood is

$$
\ell ( p , \theta ) = \sum _ { t : z _ { t } \neq \bot } \log \left[ p _ { d _ { t } } L _ { t 1 } + ( 1 - p _ { d _ { t } } ) L _ { t 0 } \right] .\tag{5}
$$

EM alternates posterior correctness and domainprevalence updates:

$$
\rho _ { t } = \frac { p _ { d _ { t } } L _ { t 1 } } { p _ { d _ { t } } L _ { t 1 } + ( 1 - p _ { d _ { t } } ) L _ { t 0 } } ,\tag{6}
$$

$$
p _ { d } \gets \frac { 1 } { M _ { d } } \sum _ { t \in \mathcal { O } _ { d } } \rho _ { t } .\tag{7}
$$

The confusion rates are updated from posteriorweighted label counts. Verified $y _ { t }$ is excluded from these fits.

The evaluated GLAD-style variant uses ability $\alpha _ { h }$ and domain difficulty $\delta _ { d }$ to impose a symmetric channel:

$$
\begin{array} { r l } & { r _ { h d } = \sigma ( \alpha _ { h } - \delta _ { d } ) , } \\ & { \theta _ { h d , 1 } = r _ { h d } , \qquad \theta _ { h d , 0 } = 1 - r _ { h d } . } \end{array}\tag{8}
$$

It shares the posterior/profile updates, with gradient updates for α and δ. This additive parameterization is a variant of GLAD (Whitehill et al., 2009). DS uses 50 EM iterations; GLAD-style uses 80 generalized-EM iterations. Domain methods retain .5 when $M _ { d } = 0 ;$ initialization and update parameters are given in Appendix C.1.

Ability initialization. Both expertise states start at $\alpha _ { h } ~ = ~ 1$ , with $\delta _ { d } ~ = ~ 0$ and $p _ { d } ~ = ~ . 5 ,$ assuming positively oriented feedback in each state. This fixed initialization is used for every model, coverage, noise channel, and partition. Original GLAD (Whitehill et al., 2009) uses σ $( \alpha _ { i } \beta _ { j } )$ and a positive ability prior. Our pair (1, 1) instead denotes two abilities in an additive adaptation. Table 17 compares this default with a mixed-sign sensitivity condition. The latter can orient a domain toward the complementary label, so its low cleanlabel CBF does not establish that reliable feedback is unhelpful.

Optimization sensitivity. The fixed 50-step DS and 80-step GLAD-style fits are finite-budget references. Table 18 compares them with likelihood optimization on the same pooled labels and parameter bounds. Clean labels are deterministic; each 15% and 30% noise condition uses the same 100 saved draws. L-BFGS-B, followed by SLSQP when needed, stops at projected-gradient norm $\leq 1 0 ^ { - 6 }$ , with a 20,000-iteration cap per optimizer.

<table><tr><td>Model</td><td>L1 bias</td><td>L1 shape</td><td>Global bias</td><td>Global shape</td><td>GLAD-style bias</td><td>GLAD-style shape</td></tr><tr><td>Mistral-7B</td><td>0.204</td><td>0.138</td><td>0.184</td><td>0.118</td><td>0.217</td><td>0.226</td></tr><tr><td>Llama-3-8B</td><td>-0.011</td><td>0.129</td><td>0.107</td><td>0.130</td><td>0.164</td><td>0.197</td></tr><tr><td>GLM-5</td><td>-0.032</td><td>0.034</td><td>-0.060</td><td>0.041</td><td>-0.014</td><td>0.098</td></tr><tr><td>DeepSeek-V3.2</td><td>-0.026</td><td>0.073</td><td>-0.006</td><td>0.074</td><td>0.063</td><td>0.093</td></tr><tr><td>Qwen-Turbo</td><td>0.171</td><td>0.097</td><td>0.004</td><td>0.104</td><td>0.178</td><td>0.112</td></tr><tr><td>Qwen2.5-7B</td><td>0.245</td><td>0.134</td><td>0.100</td><td>0.129</td><td>0.229</td><td>0.167</td></tr><tr><td>Qwen-Max</td><td>0.080</td><td>0.096</td><td>-0.268</td><td>0.071</td><td>-0.231</td><td>0.340</td></tr><tr><td>Qwen-Plus</td><td>0.094</td><td>0.120</td><td>-0.041</td><td>0.101</td><td>0.106</td><td>0.107</td></tr><tr><td>Qwen3.5-27B</td><td>-0.054</td><td>0.047</td><td>-0.077</td><td>0.033</td><td>0.003</td><td>0.060</td></tr><tr><td>GLM-5.1</td><td>-0.381</td><td>0.133</td><td>-0.118</td><td>0.050</td><td>0.014</td><td>0.077</td></tr><tr><td>GPT-5.4-mini</td><td>0.070</td><td>0.148</td><td>-0.031</td><td>0.132</td><td>0.044</td><td>0.167</td></tr><tr><td>GPT-5.4</td><td>0.010</td><td>0.104</td><td>-0.023</td><td>0.087</td><td>0.030</td><td>0.136</td></tr><tr><td>GPT-5.5</td><td>-0.056</td><td>0.042</td><td>-0.056</td><td>0.051</td><td>0.039</td><td>0.076</td></tr><tr><td>Claude-Opus-4.7</td><td>-0.220</td><td>0.075</td><td>-0.077</td><td>0.059</td><td>0.012</td><td>0.104</td></tr><tr><td>Claude-Sonnet-4.6</td><td>-0.156</td><td>0.050</td><td>-0.087</td><td>0.049</td><td>-0.013</td><td>0.088</td></tr><tr><td>Claude-Opus-4.6</td><td>-0.200</td><td>0.089</td><td>-0.071</td><td>0.061</td><td>0.026</td><td>0.080</td></tr><tr><td>DeepSeek-V4-Pro</td><td>-0.680</td><td>0.056</td><td>-0.084</td><td>0.056</td><td>0.035</td><td>0.076</td></tr><tr><td>Qwen3.6-Plus</td><td>-0.241</td><td>0.091</td><td>-0.087</td><td>0.070</td><td>0.043</td><td>0.085</td></tr></table>

Table 12: Signed bias and centered domain-shape MAE on the pooled target; lower shape error is better.
<table><tr><td>Model</td><td>DS-Beta</td><td>GLAD-style-Beta</td><td>Global / Beta / GLAD-style shape MAE</td></tr><tr><td>Qwen2.5-7B</td><td>-0.083 [-0.116, -0.038]</td><td>-0.151 [-0.170, -0.084]</td><td>0.129 / 0.120 / 0.167</td></tr><tr><td>GPT-5.4-mini</td><td>-0.010 [-0.125, +0.008]</td><td>-0.048 [-0.108, -0.006]</td><td>0.132 / 0.105 / 0.167</td></tr><tr><td>GPT-5.4</td><td>-0.071 [-0.157, +0.018]</td><td>+0.006 [-0.063, +0.035]</td><td>0.087 / 0.107 / 0.136</td></tr><tr><td>GPT-5.5</td><td>-0.056 [-0.075, +0.034]</td><td>+0.036 [+0.011, +0.092]</td><td>0.051 / 0.089 / 0.076</td></tr><tr><td>DeepSeek-V4-Pro</td><td>-0.041 [-0.086, +0.051]</td><td>+0.053 [+0.014, +0.098]</td><td>0.056 / 0.121 / 0.076</td></tr></table>

Table 13: Pooled paired CBF differences and domain-shape error under strict decoding. Brackets give questionclustered intervals from 400 refitted resamples; lower shape error is better. Global is a constant profile.
<table><tr><td></td><td colspan="4">GPT-5.4 GPT-5.4-mini</td><td colspan="2">GPT-5.5</td><td colspan="2">Cl-Op-4-6 Cl-So-4-6 Cl-Op-4-7 DSv4-Pro Q3.6-Plus</td><td colspan="2"></td><td colspan="4"></td></tr><tr><td>Category</td><td>acc</td><td>ZS acc</td><td></td><td>ZS</td><td>acc</td><td>ZS acc</td><td>ZS</td><td>acc</td><td>ZS</td><td>acc</td><td>ZS</td><td>acc</td><td>ZS</td><td>acc</td><td>ZS</td></tr><tr><td>math</td><td></td><td>0.400.850.33</td><td></td><td>0.78 0.93</td><td></td><td>0.72 0.80</td><td>0.78</td><td>0.93</td><td>0.72</td><td>0.87</td><td>0.78</td><td>0.93</td><td>0.90</td><td>0.93</td><td>0.85</td></tr><tr><td>physics</td><td></td><td>0.400.840.47</td><td></td><td>0.72 1.00 0.70  1.00</td><td></td><td></td><td></td><td>0.801.00</td><td>0.74 0.73</td><td></td><td>0.801.00</td><td></td><td>0.95 1.00</td><td></td><td>0.90</td></tr><tr><td>chemistry</td><td>0.60 0.82 0.20</td><td></td><td></td><td>0.70 0.87 0.72 0.93</td><td></td><td></td><td>0.75</td><td>0.92</td><td>0.73</td><td>0.73</td><td></td><td>0.72 0.87</td><td>0.900.87</td><td></td><td>0.82</td></tr><tr><td>law</td><td></td><td>0.60 0.810.53</td><td></td><td>0.74 0.80 0.76 0.73</td><td></td><td></td><td></td><td>0.70 0.73</td><td>0.67 0.80</td><td></td><td></td><td>0.60 0.79</td><td>0.75 0.73</td><td></td><td>0.80</td></tr><tr><td>engineering</td><td>0.530.800.47</td><td></td><td></td><td>0.73 0.80 0.68 0.93</td><td></td><td></td><td></td><td>0.72 1.00</td><td>0.70 0.67</td><td></td><td></td><td>0.65 0.85</td><td>0.85 0.93</td><td></td><td>0.85</td></tr><tr><td>economics</td><td></td><td>0.80 0.83 0.73</td><td></td><td>0.79 0.93 0.78 1.00</td><td></td><td></td><td></td><td>0.84 0.87</td><td></td><td>0.76 0.87</td><td></td><td>0.82 0.93</td><td>0.80 0.93</td><td></td><td>0.90</td></tr><tr><td>health</td><td></td><td>0.67 0.86 0.73</td><td></td><td>0.77 0.87 0.76 0.79</td><td></td><td></td><td></td><td>0.79 0.80</td><td>0.75</td><td>0.71</td><td></td><td>0.72 0.87</td><td></td><td>0.900.87</td><td>0.90</td></tr><tr><td>psychology</td><td>0.80 0.88 0.73</td><td></td><td></td><td>0.82 0.80 0.82 0.87</td><td></td><td></td><td></td><td>0.87 0.87</td><td>0.78</td><td>0.73</td><td></td><td>0.85 0.93</td><td></td><td>0.85 0.80</td><td>0.85</td></tr><tr><td>business</td><td>0.470.840.20</td><td></td><td></td><td>0.80 0.80</td><td></td><td>0.82 0.93</td><td></td><td>0.81 0.80</td><td>0.75</td><td>0.80</td><td>0.78</td><td>0.73</td><td></td><td>0.900.87</td><td>0.90</td></tr><tr><td>history</td><td>0.73</td><td>0.87 0.67</td><td></td><td>0.84</td><td>0.80</td><td>0.80</td><td>0.80</td><td>0.76 0.73</td><td>0.79</td><td>0.73</td><td>0.75</td><td>0.73</td><td>0.85</td><td>0.80</td><td>0.90</td></tr><tr><td>mean</td><td></td><td>0.600.84 0.51</td><td></td><td>0.77</td><td></td><td>0.86 0.76</td><td>0.88</td><td>0.78 0.86</td><td>0.74</td><td>0.76</td><td>0.75</td><td>0.86</td><td>0.86 0.87</td><td></td><td>0.87</td></tr><tr><td>unparsed/150</td><td>0</td><td></td><td>0</td><td></td><td>0</td><td></td><td>1</td><td></td><td>8</td><td></td><td>1</td><td></td><td>3</td><td></td><td>0</td></tr><tr><td>CBF</td><td>0.760</td><td></td><td>0.738</td><td></td><td>0.888</td><td></td><td>0.902</td><td></td><td>0.863</td><td></td><td>0.941</td><td></td><td>0.931</td><td></td><td>0.937</td></tr></table>

Table 14: MMLU-Pro per-category accuracy and ZS-SK estimates across 8 frontier models (15 questions per category × 10 categories); bottom row = per-model CBF on this held-out set. Cl-Op = Claude-Opus, Cl-So = Claude-Sonnet, DSv4-Pro = DeepSeek-V4-Pro, Q3.6 = Qwen3.6.

<table><tr><td rowspan="2">Task</td><td colspan="2">GPT-5.4</td><td colspan="2">GPT-5.4-mini</td><td colspan="2">GPT-5.5</td><td colspan="2">Cl-Op-4-6 Cl-So-4-6 Cl-Op-4-7 DSv4-Pro Q3.6-Plus</td><td colspan="2"></td><td colspan="2"></td><td colspan="2"></td></tr><tr><td> $p ^ { * }$ </td><td>ZS</td><td> $p ^ { * }$ </td><td>ZS</td><td> $p ^ { * }$  ZS</td><td> $p ^ { * }$ </td><td>ZS</td><td> $p ^ { * }$ </td><td> ${ \cal Z } \mathrm { S }$   $p ^ { * }$ </td><td>ZS</td><td> $p ^ { * }$ </td><td>ZS</td><td> $p ^ { * }$ </td><td>ZS</td></tr><tr><td>boolean_expressions</td><td>0.87 0.94 0.87</td><td></td><td></td><td>0.82</td><td>1.000.95</td><td>0.93</td><td>0.96</td><td>1.00</td><td>0.88 0.87</td><td></td><td>0.95 1.00</td><td>0.85</td><td>1.00</td><td>0.95</td></tr><tr><td>causal_judgement</td><td>0.67 0.83 0.67</td><td></td><td></td><td>0.78</td><td>0.670.85</td><td>0.53</td><td>0.72</td><td>0.73 0.60</td><td>0.80</td><td>0.65</td><td>0.73</td><td>0.80</td><td>0.73</td><td>0.70</td></tr><tr><td>date_understanding</td><td>0.27 0.90 0.20</td><td></td><td></td><td>0.75</td><td>0.470.85</td><td>0.93</td><td>0.92</td><td>1.000.82</td><td>0.87</td><td>0.85</td><td>0.80</td><td>0.93</td><td>0.87 0.85</td><td></td></tr><tr><td>logical_deduction_5</td><td>1.000.860.67</td><td></td><td></td><td>0.70</td><td>1.00 0.901.00</td><td></td><td>0.88</td><td>1.000.65</td><td>1.00</td><td>0.90</td><td>0.53</td><td>0.85</td><td>1.000.75</td><td></td></tr><tr><td>navigate</td><td>0.80 0.88 0.53</td><td></td><td></td><td>0.68</td><td>1.000.850.93</td><td></td><td>0.95</td><td>1.000.76 0.93</td><td></td><td>0.90</td><td>1.00</td><td></td><td>0.901.000.70</td><td></td></tr><tr><td>object_counting</td><td>0.73 0.960.53</td><td></td><td></td><td>0.90</td><td>1.00 0.90 0.80</td><td></td><td>0.97</td><td></td><td>1.000.840.60</td><td>0.95</td><td>1.00</td><td></td><td>0.801.000.95</td><td></td></tr><tr><td>ruin_names</td><td>0.67 0.79 0.67</td><td></td><td></td><td>0.72</td><td>0.93 0.75</td><td>0.87</td><td>0.86</td><td>0.87</td><td>0.56 0.93</td><td>0.75</td><td>0.73</td><td>0.70</td><td>0.93</td><td>0.80</td></tr><tr><td>sports_understanding</td><td>1.000.84 0.93</td><td></td><td></td><td>0.74</td><td>0.80 0.90 0.93</td><td></td><td>0.96</td><td>1.000.900.87</td><td></td><td>0.90</td><td>0.60</td><td>0.65</td><td>1.000.95</td><td></td></tr><tr><td>tracking_shuffled_3</td><td>0.67 0.95 0.67</td><td></td><td></td><td>0.88</td><td>1.000.85</td><td>1.00</td><td>0.98</td><td>1.000.73</td><td></td><td>0.53 0.95</td><td>0.87</td><td>0.90</td><td>1.00</td><td>0.95</td></tr><tr><td>word_sorting</td><td>0.67 0.97 0.47</td><td></td><td></td><td>0.95</td><td>1.000.90</td><td>0.13</td><td>0.94</td><td>0.00</td><td>0.80</td><td>0.93 0.90</td><td>0.33</td><td>0.90</td><td>0.87</td><td>0.92</td></tr><tr><td></td><td></td><td></td><td></td><td>0.780</td><td>0.850</td><td></td><td>0.861</td><td>0.734</td><td></td><td>0.860</td><td></td><td></td><td></td><td></td></tr><tr><td>CBF</td><td>0.781</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.835</td><td>0.901</td><td></td></tr></table>

Table 15: BBH per-task accuracy $( p ^ { * } )$ and ZS-SK estimates across 8 frontier models (15 questions per task × 10 BBH tasks); BBH has no subject-name overlap with MMLU. Bottom row = per-model CBF on this held-out set. Cl-Op = Claude-Opus, Cl-So = Claude-Sonnet, DSv4-Pro = DeepSeek-V4-Pro, $\mathrm { Q 3 . 6 = Q w e n } 3 . 6$
<table><tr><td>Model</td><td>EM-Joint</td><td>DS</td><td>GLAD-style</td><td>L1</td></tr><tr><td>Mistral-7B</td><td>0.748/0.755</td><td>0.742/0.799</td><td>0.731/0.712</td><td>0.677</td></tr><tr><td>Llama-3-8B</td><td>0.748/0.796</td><td>0.748/0.868</td><td>0.734/0.730</td><td>0.734</td></tr><tr><td>GLM-5</td><td>0.572/0.909</td><td>0.575/0.883</td><td>0.594/0.908</td><td>0.904</td></tr><tr><td>DeepSeek-V3.2</td><td>0.626/0.900</td><td>0.627/0.845</td><td>0.636/0.885</td><td>0.834</td></tr><tr><td>Qwen-Turbo</td><td>0.711/0.896</td><td>0.703/0.813</td><td>0.694/0.793</td><td>0.772</td></tr><tr><td>Qwen2.5-7B</td><td>0.753/0.835</td><td>0.748/0.791</td><td>0.738/0.724</td><td>0.670</td></tr><tr><td>Qwen-Max</td><td>0.680/0.711</td><td>0.672/0.676</td><td>0.667/0.643</td><td>0.789</td></tr><tr><td>Qwen-Plus</td><td>0.667/0.902</td><td>0.670/0.871</td><td>0.670/0.849</td><td>0.806</td></tr><tr><td>Qwen3.5-27B</td><td>0.573/0.909</td><td>0.577/0.820</td><td>0.594/0.940</td><td>0.886</td></tr><tr><td>GLM-5.1</td><td>0.578/0.873</td><td>0.580/0.799</td><td>0.578/0.919</td><td>0.608</td></tr><tr><td>GPT-5.4-mini</td><td>0.686/0.895</td><td>0.686/0.849</td><td>0.685/0.811</td><td>0.764</td></tr><tr><td>GPT-5.4</td><td>0.651/0.879</td><td>0.648/0.781</td><td>0.647/0.857</td><td>0.797</td></tr><tr><td>GPT-5.5</td><td>0.605/0.938</td><td>0.601/0.818</td><td>0.609/0.909</td><td>0.858</td></tr><tr><td>Claude-Opus-4.7</td><td>0.609/0.880</td><td>0.613/0.830</td><td>0.621/0.890</td><td>0.742</td></tr><tr><td>Claude-Sonnet-4.6</td><td>0.579/0.844</td><td>0.589/0.838</td><td>0.590/0.915</td><td>0.801</td></tr><tr><td>Claude-Opus-4.6</td><td>0.592/0.876</td><td>0.599/0.833</td><td>0.616/0.917</td><td>0.771</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.592/0.920</td><td>0.592/0.819</td><td>0.596/0.913</td><td>0.316</td></tr><tr><td>Qwen3.6-Plus</td><td>0.616/0.928</td><td>0.613/0.847</td><td>0.612/0.903</td><td>0.719</td></tr></table>

Table 16: Session/pooled CBF of reliability-aware methods; L1 uses the session target.

<table><tr><td>Model</td><td>Clean Clean (1,1) (1, −.3)</td><td>30% (1,1)</td><td>30% (1, −.3)</td></tr><tr><td>Qwen2.5-7B</td><td>0.854 0.563</td><td>0.924</td><td>0.706</td></tr><tr><td>GPT-5.4-mini</td><td>0.879</td><td>0.411 0.922</td><td>0.563 0.482</td></tr><tr><td>GPT-5.4</td><td>0.880</td><td>0.248 0.920</td><td></td></tr><tr><td>GPT-5.5</td><td>0.934</td><td>0.127 0.912</td><td>0.296</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.938</td><td>0.080 0.911</td><td>0.283</td></tr></table>

Table 17: GLAD-style pooled CBF under two fixed ability initializations; 100 repetitions. The positive pair is the reported default.

2010/2010 fits meet this criterion; the largest norm is $1 . 0 1 \times 1 0 ^ { - 7 }$ . The per-label log-likelihood gains on clean labels are 0.000000–0.000061 for DS and 0.003–0.049 for GLAD-style. Final state reliabilities, domain parameters, and likelihood trajectories accompany the analysis materials.

Optimization raises clean-label GLAD-style CBF from .854–.938 to 0.920–0.986. Thus the fixed-budget score increase after adding noise does not establish a beneficial noise effect. DS improves its likelihood while retaining substantial calibration error. Its marginal observation model is $q _ { h d } = f _ { h } + p _ { d } ( t _ { h } - f _ { h } )$ . An affine transformation of p<sub>d</sub> can be offset by a transformation of $t _ { h } , f _ { h }$ while preserving $q _ { h d }$ whenever the parameters remain within their bounds. More single labels therefore need not uniquely determine prevalence and reliability. The pooled result cannot be explained solely by short sessions: both optimization budget and identifiability matter.

A6: Feedback decoding and coverage. We compare domain counts under three channels: decoded observed labels, verified labels at those same observed positions, and all verified labels. Holding the positions fixed isolates label correction; adding the remaining positions isolates coverage.

Decoded feedback selects which answers supply labels. For Qwen-Max, correct answers are less likely to yield a decoded label than incorrect answers, consistent with the confirmation misses in Table 5. This produces a low-accuracy observed subset. Correcting its labels alone cannot remove selection; all valid-answer domains remain in the scoring target.

Semantic-reference comparison. We sample 358 nonempty feedback utterances on valid answers, stratified by model, expertise state, and frozen decoder category, with at most three randomly selected utterances per nonempty stratum. One LLM annotator assigns positive, negative, uncertain, or neutral stance from the utterance and submitted answer. Estimates weight each sample by its stratum population divided by sample size. Precision is correctly decoded stance divided by all positive/negative decoder outputs; recall is correctly decoded stance divided by all positive/negative reference labels. Table 20 reports all 18 models. Per-expertise estimates and annotated utterances accompany the analysis materials.

The expanded rules recognize accurate, well done, and spot on, and distinguish answer rejection from questions about incorrect options. A statement of “the correct answer is A is compared with the submitted option: agreement is confirmation and a different option is correction. Outcome keys are not used by the decoder. The frozen rules remain the primary protocol; the expanded rules define a separate sensitivity condition.

Across the weighted sample, precision/recall are 0.890/0.224 for frozen rules and 0.992/0.728 for expanded rules. Applying expanded rules to the complete valid-answer stream changes Global's sessionlevel lead from 14/18 to 11/18 (Table 21). Count becomes the best matched control for Qwen-Turbo, Qwen-Plus, GPT-5.4-mini, and GPT-5.4. The sensitivity confirms that the observed channel depends on language interpretation as well as the simulator; a single decoder cannot establish a decoderindependent estimator ranking.

Exact counting reaches CBF 1 with complete verified labels by construction. That result does not imply that EM or a model aggregating the same labels will recover the target exactly. The matchedinput study and synthetic-noise grid evaluate those estimators separately.

## E.2 Learning Dynamics

![](images/8bcbfbc148eeb9fba0eded864ef82c4a4185b494321e5af3e812927f26514330.jpg)  
Figure 10: EM-Joint learning curves. Prefixes of valid interactions are scored against the fixed final-session domain target.

<table><tr><td>Model</td><td>Noise</td><td>Beta</td><td>DS fixed</td><td>DS opt.</td><td>GLAD fixed</td><td>GLAD opt.</td></tr><tr><td>Qwen2.5-7B</td><td>0</td><td>0.997</td><td>0.919</td><td>0.920</td><td>0.854</td><td>0.920</td></tr><tr><td>Qwen2.5-7B</td><td>15</td><td>0.943</td><td>0.906</td><td>0.906</td><td>0.909</td><td>0.909</td></tr><tr><td>Qwen2.5-7B</td><td>30</td><td>0.895</td><td>0.886</td><td>0.885</td><td>0.924</td><td>0.874</td></tr><tr><td>GPT-5.4-mini</td><td>0</td><td>0.996</td><td>0.838</td><td>0.838</td><td>0.879</td><td>0.961</td></tr><tr><td>GPT-5.4-mini</td><td>15</td><td>0.917</td><td>0.797</td><td>0.797</td><td>0.908</td><td>0.910</td></tr><tr><td>GPT-5.4-mini</td><td>30</td><td>0.843</td><td>0.772</td><td>0.771</td><td>0.922</td><td>0.849</td></tr><tr><td>GPT-5.4</td><td>0</td><td>0.994</td><td>0.765</td><td>0.716</td><td>0.880</td><td>0.942</td></tr><tr><td>GPT-5.4</td><td>15</td><td>0.897</td><td>0.723</td><td>0.719</td><td>0.917</td><td>0.915</td></tr><tr><td>GPT-5.4</td><td>30</td><td>0.800</td><td>0.697</td><td>0.696</td><td>0.920</td><td>0.843</td></tr><tr><td>GPT-5.5</td><td>0</td><td>0.992</td><td>0.713</td><td>0.702</td><td>0.934</td><td>0.982</td></tr><tr><td>GPT-5.5</td><td>15</td><td>0.867</td><td>0.633</td><td>0.631</td><td>0.947</td><td>0.904</td></tr><tr><td>GPT-5.5</td><td>30</td><td>0.741</td><td>0.606</td><td>0.597</td><td>0.912</td><td>0.800</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0</td><td>0.991</td><td>0.765</td><td>0.765</td><td>0.938</td><td>0.986</td></tr><tr><td>DeepSeek-V4-Pro</td><td>15</td><td>0.865</td><td>0.639</td><td>0.632</td><td>0.952</td><td>0.907</td></tr><tr><td>DeepSeek-V4-Pro</td><td>30</td><td>0.739</td><td>0.607</td><td>0.592</td><td>0.911</td><td>0.805</td></tr></table>

Table 18: Pooled CBF with complete coverage. Fixed denotes the reported iteration budget; Opt. denotes likelihood optimization to the stated stationary-point criterion. No outcome labels enter either fit. Noise columns are percentages; noisy scores average 100 draws.
<table><tr><td>Model</td><td>Coverage</td><td>Decoded</td><td>Corrected positions</td><td>Complete</td><td>∆Acc (pp)</td></tr><tr><td>Mistral-7B</td><td>11.1%</td><td>0.809</td><td>0.834</td><td>1.000</td><td>+14.9</td></tr><tr><td>Llama-3-8B</td><td>13.4%</td><td>0.852</td><td>0.848</td><td>1.000</td><td>+10.2</td></tr><tr><td>GLM-5</td><td>14.8%</td><td>0.862</td><td>0.963</td><td>1.000</td><td>+2.6</td></tr><tr><td>DeepSeek-V3.2</td><td>15.2%</td><td>0.849</td><td>0.898</td><td>1.000</td><td>+5.3</td></tr><tr><td>Qwen-Turbo</td><td>14.7%</td><td>0.892</td><td>0.870</td><td>1.000</td><td>+7.4</td></tr><tr><td>Qwen2.5-7B</td><td>12.1%</td><td>0.846</td><td>0.822</td><td>1.000</td><td>+17.0</td></tr><tr><td>Qwen-Max</td><td>6.7%</td><td>0.710</td><td>0.688</td><td>1.000</td><td>-28.8</td></tr><tr><td>Qwen-Plus</td><td>13.8%</td><td>0.878</td><td>0.886</td><td>1.000</td><td>+5.5</td></tr><tr><td>Qwen3.5-27B</td><td>15.5%</td><td>0.888</td><td>0.950</td><td>1.000</td><td>+1.8</td></tr><tr><td>GLM-5.1</td><td>12.2%</td><td>0.864</td><td>0.941</td><td>1.000</td><td>+2.1</td></tr><tr><td>GPT-5.4-mini</td><td>4.9%</td><td>0.820</td><td>0.848</td><td>1.000</td><td>-1.3</td></tr><tr><td>GPT-5.4</td><td>5.9%</td><td>0.839</td><td>0.860</td><td>1.000</td><td>+1.5</td></tr><tr><td>GPT-5.5</td><td>15.6%</td><td>0.888</td><td>0.932</td><td>1.000</td><td>+1.3</td></tr><tr><td>Claude-Opus-4.7</td><td>14.5%</td><td>0.847</td><td>0.918</td><td>1.000</td><td>+2.5</td></tr><tr><td>Claude-Sonnet-4.6</td><td>13.0%</td><td>0.872</td><td>0.915</td><td>1.000</td><td>+1.1</td></tr><tr><td>Claude-Opus-4.6</td><td>15.0%</td><td>0.877</td><td>0.925</td><td>1.000</td><td>+4.1</td></tr><tr><td>DeepSeek-V4-Pro</td><td>13.7%</td><td>0.870</td><td>0.924</td><td>1.000</td><td>+3.6</td></tr><tr><td>Qwen3.6-Plus</td><td>15.0%</td><td>0.868</td><td>0.926</td><td>1.000</td><td>+2.5</td></tr></table>

Table 19: Pooled CBF of Count under matched feedback-channel interventions. ∆Acc (pp) is 100 times accuracy on decoded positions minus accuracy on all valid answers. For Qwen-Max, 79/158 decoded positions are correct (50.0%), versus 1867/2368 valid answers (78.8%), yielding −28.8 pp.

<table><tr><td>Model</td><td>n</td><td>P(F)</td><td>R(F)</td><td>P(E)</td><td>R(E)</td><td>Cov.(F)</td><td>Cov.(E)</td></tr><tr><td>Mistral-7B</td><td>19</td><td>0.884</td><td>0.230</td><td>1.000</td><td>0.950</td><td>0.111</td><td>0.191</td></tr><tr><td>Llama-3-8B</td><td>18</td><td>0.852</td><td>0.741</td><td>1.000</td><td>0.800</td><td>0.134</td><td>0.175</td></tr><tr><td>GLM-5</td><td>19</td><td>0.916</td><td>0.190</td><td>0.999</td><td>0.588</td><td>0.148</td><td>0.360</td></tr><tr><td>DeepSeek-V3.2</td><td>21</td><td>0.939</td><td>0.813</td><td>0.997</td><td>0.943</td><td>0.152</td><td>0.360</td></tr><tr><td>Qwen-Turbo</td><td>21</td><td>0.893</td><td>0.297</td><td>0.969</td><td>0.365</td><td>0.147</td><td>0.354</td></tr><tr><td>Qwen2.5-7B</td><td>20</td><td>0.899</td><td>0.253</td><td>0.989</td><td>0.966</td><td>0.121</td><td>0.314</td></tr><tr><td>Qwen-Max</td><td>21</td><td>0.766</td><td>0.120</td><td>0.964</td><td>0.834</td><td>0.067</td><td>0.318</td></tr><tr><td>Qwen-Plus</td><td>21</td><td>0.900</td><td>0.293</td><td>0.998</td><td>0.959</td><td>0.138</td><td>0.346</td></tr><tr><td>Qwen3.5-27B</td><td>19</td><td>0.907</td><td>0.143</td><td>0.999</td><td>0.709</td><td>0.155</td><td>0.364</td></tr><tr><td>GLM-5.1</td><td>18</td><td>0.874</td><td>0.189</td><td>1.000</td><td>0.991</td><td>0.122</td><td>0.281</td></tr><tr><td>GPT-5.4-mini</td><td>22</td><td>0.904</td><td>0.137</td><td>0.997</td><td>0.974</td><td>0.049</td><td>0.337</td></tr><tr><td>GPT-5.4</td><td>23</td><td>0.909</td><td>0.167</td><td>0.992</td><td>0.998</td><td>0.059</td><td>0.347</td></tr><tr><td>GPT-5.5</td><td>19</td><td>0.913</td><td>0.145</td><td>0.989</td><td>0.444</td><td>0.156</td><td>0.356</td></tr><tr><td>Claude-Opus-4.7</td><td>21</td><td>0.858</td><td>0.287</td><td>0.989</td><td>0.352</td><td>0.145</td><td>0.304</td></tr><tr><td>Claude-Sonnet-4.6</td><td>18</td><td>0.916</td><td>0.273</td><td>0.974</td><td>0.317</td><td>0.130</td><td>0.323</td></tr><tr><td>Claude-Opus-4.6</td><td>18</td><td>0.901</td><td>0.312</td><td>1.000</td><td>0.974</td><td>0.150</td><td>0.315</td></tr><tr><td>DeepSeek-V4-Pro</td><td>19</td><td>0.881</td><td>0.171</td><td>0.999</td><td>0.583</td><td>0.137</td><td>0.335</td></tr><tr><td>Qwen3.6-Plus</td><td>21</td><td>0.877</td><td>0.301</td><td>0.984</td><td>0.964</td><td>0.150</td><td>0.345</td></tr></table>

Table 20: Stratified semantic-reference comparison. P/R are weighted stance precision/recall; Cov. is coverage of all valid answers. F = frozen, E = expanded. The same sampled utterances support rule inspection and reference comparison.
<table><tr><td>Model</td><td>Global</td><td>Count</td><td>Beta</td><td>DS</td><td>GLAD</td></tr><tr><td>Mistral-7B</td><td>0.590</td><td>0.634</td><td>0.711</td><td>0.727</td><td>0.633</td></tr><tr><td>Llama-3-8B</td><td>0.679</td><td>0.693</td><td>0.735</td><td>0.738</td><td>0.690</td></tr><tr><td>GLM-5</td><td>0.888</td><td>0.803</td><td>0.667</td><td>0.601</td><td>0.796</td></tr><tr><td>DeepSeek-V3.2</td><td>0.836</td><td>0.788</td><td>0.700</td><td>0.657</td><td>0.783</td></tr><tr><td>Qwen-Turbo</td><td>0.774</td><td>0.797</td><td>0.771</td><td>0.729</td><td>0.789</td></tr><tr><td>Qwen2.5-7B</td><td>0.697</td><td>0.754</td><td>0.776</td><td>0.763</td><td>0.748</td></tr><tr><td>Qwen-Max</td><td>0.798</td><td>0.785</td><td>0.755</td><td>0.718</td><td>0.782</td></tr><tr><td>Qwen-Plus</td><td>0.804</td><td>0.813</td><td>0.742</td><td>0.692</td><td>0.806</td></tr><tr><td>Qwen3.5-27B</td><td>0.891</td><td>0.807</td><td>0.669</td><td>0.601</td><td>0.801</td></tr><tr><td>GLM-5.1</td><td>0.801</td><td>0.745</td><td>0.647</td><td>0.607</td><td>0.740</td></tr><tr><td>GPT-5.4-mini</td><td>0.721</td><td>0.727</td><td>0.726</td><td>0.698</td><td>0.725</td></tr><tr><td>GPT-5.4</td><td>0.725</td><td>0.739</td><td>0.713</td><td>0.672</td><td>0.735</td></tr><tr><td>GPT-5.5</td><td>0.881</td><td>0.804</td><td>0.690</td><td>0.633</td><td>0.796</td></tr><tr><td>Claude-Opus-4.7</td><td>0.859</td><td>0.784</td><td>0.689</td><td>0.648</td><td>0.777</td></tr><tr><td>Claude-Sonnet-4.6</td><td>0.871</td><td>0.780</td><td>0.666</td><td>0.617</td><td>0.773</td></tr><tr><td>Claude-Opus-4.6</td><td>0.872</td><td>0.781</td><td>0.670</td><td>0.629</td><td>0.773</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.880</td><td>0.791</td><td></td><td>0.626</td><td></td></tr><tr><td></td><td></td><td>0.800</td><td>0.680</td><td></td><td>0.785</td></tr><tr><td>Qwen3.6-Plus</td><td>0.860</td><td></td><td>0.704</td><td>0.645</td><td>0.792</td></tr><tr><td>Model</td><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td></tr><tr><td>Llama-3-8B</td><td>0.736</td><td>0.739</td><td>0.743</td><td>0.745</td><td>0.748</td></tr><tr><td>Qwen-Max</td><td>0.680</td><td>0.679</td><td>0.679</td><td>0.680</td><td>0.680</td></tr><tr><td>Qwen-Plus</td><td>0.647</td><td>0.652</td><td>0.659</td><td>0.665</td><td>0.667</td></tr><tr><td>GPT-5.4</td><td>0.642</td><td>0.644</td><td>0.647</td><td>0.649</td><td>0.651</td></tr><tr><td>Claude-Opus-4.6</td><td>0.568</td><td>0.574</td><td>0.583</td><td>0.592</td><td>0.592</td></tr></table>

Table 21: Session-mean CBF under expanded feedback rules, on the same valid-answer targets and domain support as the primary comparison.

Table 22: EM-Joint CBF against a fixed final-session target at valid-turn prefixes.

Figure 10 follows EM-Joint estimates as more valid interactions become available. Each prefix is compared with the same final-session target. This distinguishes a changing estimate from a changing empirical reference.

## E.3 Prefix Estimation Curves

Table 22 reports endpoint-target CBF at successive valid-turn prefixes. A session with fewer valid turns uses all of its available turns at later checkpoints. The final column matches the EM-Joint main-table score.

These curves describe a 50-turn observation budget. They do not establish convergence with unlimited data; sparse informative feedback can leave domain estimates close to their initialization.

## E.4 Noise Sensitivity

We fix valid answers and apply five label-noise channels at 12.5%, 50%, and 100% coverage on five models. The channels have false-positive/falsenegative rates 0/0, .15/.15, .30/.30, .30/.10, and .10/.30. Each condition retains 100 repetitions and is scored per session and after pooling. The 12.5% condition approximates the order of observed feedback coverage while using random rather than naturally selected missingness. Table 23 gives fullcoverage symmetric-noise results; Appendix E.7 contains the complete grid.

Lower label noise often improves reconstruction, but the ordering of estimators changes with coverage and scoring unit. With complete clean labels, exact Count equals 1. Beta approaches this ceiling as observations accumulate; latent reliability fits additionally depend on optimization and identifiability (Table 18).

## E.5 Feedback Controls on the Common Pool

Each model receives the same 400 questions, 20 per subject. The answer budget is six attempts: the first correct answer is retained, otherwise the last valid answer. The target is domain performance on this fixed answer pool. All models are scored on the

398 questions with valid answers from each, giving 19–20 questions per subject. Symmetric noise flips either class; FA and FR flip labels only for incorrect and correct answers, respectively. Coverage is 50% or 100%. Profiles share the same domain targets and uniform domain weights. The table reports CBF for the fixed label draws.

Table 24 separates history aggregation from direct counting. With complete clean labels, LSA reaches .989 on GPT-5.5 and .995 on DeepSeek-V4-Pro, but .835 on Qwen2.5-7B, while Count is 1.000 for all three. Thus, even when the feedback contains the exact outcomes, reconstruction accuracy depends on how the history is aggregated.

## E.6 Domain-Partition Sensitivity

We use the same answers and strict feedback labels for five models. Tiers are formed at ±10/15/20 percentage points around the model mean. We also retain 100 random four-group partitions and 100 random five-subject subsets. Each draw preserves a common target and domain support across methods. Grouping affects both the target dispersion and CBF, so absolute values across different partitions have different difficulty.

Semantic and random partitions both change method rankings. In the four-category partition, Global leads four of five models; Qwen2.5-7B favors DS. Grouping changes the prediction target and its dispersion, so a higher CBF on coarse groups is not evidence of better fine-domain discrimination.

## E.7 Noise and Coverage: Complete Grid

Tables 27, 28, and 29 compare 100%, 50%, and 12.5% label coverage on the same five-model pool. Within each table, rows vary label error rates; across tables, corresponding rows isolate the amount of feedback. Each cell lists the session mean first and the pooled score second.

Tables 27–29 show that coverage can change the relative advantage of reliability estimation. On GPT-5.5 with clean labels, GLAD-style exceeds Global at full coverage (.971 vs. .885 session CBF), but falls below it at 12.5% coverage (.686 vs. .821). Sparse feedback therefore changes the ranking even without label errors. Session and pooled scores report this effect at different aggregation scales.

## E.8 Dual-Role Bias

The Qwen-Max agent–simulator pairing produces few explicit corrections of incorrect answers. Utterances such as “I believe there might be a small misunderstanding” illustrate why hedged language can remain undecoded by a strict binary classifier. The comparison varies both the generated feedback and its linguistic form; the measured effect is feedback coverage.

<table><tr><td>Model</td><td>Noise</td><td>Global</td><td>Beta</td><td>DS</td><td>GLAD-style</td></tr><tr><td>Qwen2.5-7B</td><td>0.00</td><td>0.749/0.871</td><td>0.895/0.997</td><td>0.977/0.919</td><td>0.934/0.854</td></tr><tr><td>Qwen2.5-7B</td><td>0.15</td><td>0.746/0.865</td><td>0.823/0.943</td><td>0.853/0.906</td><td>0.825/0.909</td></tr><tr><td>Qwen2.5-7B</td><td>0.30</td><td>0.739/0.854</td><td>0.764/0.895</td><td>0.757/0.886</td><td>0.727/0.924</td></tr><tr><td>GPT-5.4-mini</td><td>0.00</td><td>0.774/0.869</td><td>0.870/0.996</td><td>0.949/0.838</td><td>0.943/0.879</td></tr><tr><td>GPT-5.4-mini</td><td>0.15</td><td>0.756/0.854</td><td>0.795/0.917</td><td>0.834/0.797</td><td>0.835/0.908</td></tr><tr><td>GPT-5.4-mini</td><td>0.30</td><td>0.727/0.815</td><td>0.730/0.843</td><td>0.737/0.772</td><td>0.726/0.922</td></tr><tr><td>GPT-5.4</td><td>0.00</td><td>0.803/0.913</td><td>0.856/0.994</td><td>0.914/0.765</td><td>0.953/0.880</td></tr><tr><td>GPT-5.4</td><td>0.15</td><td>0.770/0.876</td><td>0.778/0.897</td><td>0.810/0.723</td><td>0.847/0.917</td></tr><tr><td>GPT-5.4</td><td>0.30</td><td>0.721/0.803</td><td>0.707/0.800</td><td>0.717/0.697</td><td>0.731/0.920</td></tr><tr><td>GPT-5.5</td><td>0.00</td><td>0.885/0.949</td><td>0.828/0.992</td><td>0.871/0.713</td><td>0.971/0.934</td></tr><tr><td>GPT-5.5</td><td>0.15</td><td>0.807/0.873</td><td>0.744/0.867</td><td>0.770/0.633</td><td>0.863/0.947</td></tr><tr><td>GPT-5.5</td><td>0.30</td><td>0.713/0.745</td><td>0.663/0.741</td><td>0.681/0.606</td><td>0.732/0.912</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.00</td><td>0.884/0.944</td><td>0.826/0.991</td><td>0.885/0.765</td><td>0.972/0.938</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.15</td><td>0.803/0.867</td><td>0.741/0.865</td><td>0.778/0.639</td><td>0.865/0.952</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.30</td><td>0.706/0.741</td><td>0.659/0.739</td><td>0.682/0.607</td><td>0.733/0.911</td></tr></table>

Table 23: Session/pooled CBF under complete-coverage symmetric label noise; means over 100 repetitions. This slice isolates label noise at fixed coverage; the full coverage comparison appears in Tables 27–29.
<table><tr><td>Model</td><td>Channel</td><td>Coverage</td><td>LSA</td><td>Global</td><td>Count</td><td>Beta</td><td>DS</td><td>GLAD-style</td></tr><tr><td>GPT-5.5</td><td>Clean</td><td>100%</td><td>0.989</td><td>0.938</td><td>1.000</td><td>0.961</td><td>0.820</td><td>0.939</td></tr><tr><td>GPT-5.5</td><td>Clean</td><td>50%</td><td>0.940</td><td>0.939</td><td>0.962</td><td>0.929</td><td>0.822</td><td>0.945</td></tr><tr><td>GPT-5.5</td><td>Sym. 15%</td><td>100%</td><td>0.844</td><td>0.852</td><td>0.846</td><td>0.826</td><td>0.723</td><td>0.914</td></tr><tr><td>GPT-5.5</td><td>Sym. 15%</td><td>50%</td><td>0.857</td><td>0.858</td><td>0.859</td><td>0.815</td><td>0.757</td><td>0.900</td></tr><tr><td>GPT-5.5</td><td>Sym. 30%</td><td>100%</td><td>0.715</td><td>0.723</td><td>0.718</td><td>0.707</td><td>0.639</td><td>0.796</td></tr><tr><td>GPT-5.5</td><td>Sym. 30%</td><td>50%</td><td>0.705</td><td>0.718</td><td>0.707</td><td>0.689</td><td>0.634</td><td>0.756</td></tr><tr><td>GPT-5.5</td><td>FA 30%</td><td>100%</td><td>0.884</td><td>0.891</td><td>0.884</td><td>0.863</td><td>0.744</td><td>0.932</td></tr><tr><td>GPT-5.5</td><td>FA 30%</td><td>50%</td><td>0.880</td><td>0.891</td><td>0.885</td><td>0.845</td><td>0.757</td><td>0.918</td></tr><tr><td>GPT-5.5</td><td>FR 30%</td><td>100%</td><td>0.710</td><td>0.713</td><td>0.713</td><td>0.700</td><td>0.644</td><td>0.797</td></tr><tr><td>GPT-5.5</td><td>FR 30%</td><td>50%</td><td>0.690</td><td>0.708</td><td>0.707</td><td>0.684</td><td>0.642</td><td>0.762</td></tr><tr><td>DeepSeek-V4-Pro</td><td>Clean</td><td>100%</td><td>0.995</td><td>0.898</td><td>1.000</td><td>0.968</td><td>0.845</td><td>0.905</td></tr><tr><td>DeepSeek-V4-Pro</td><td>Clean</td><td>50%</td><td>0.927</td><td>0.900</td><td>0.945</td><td>0.940</td><td>0.827</td><td>0.875</td></tr><tr><td>DeepSeek-V4-Pro</td><td>Sym. 15%</td><td>100%</td><td>0.858</td><td>0.850</td><td>0.876</td><td>0.861</td><td>0.768</td><td>0.901</td></tr><tr><td>DeepSeek-V4-Pro</td><td>Sym. 15%</td><td>50%</td><td>0.858</td><td>0.863</td><td>0.872</td><td>0.854</td><td>0.772</td><td>0.874</td></tr><tr><td>DeepSeek-V4-Pro</td><td>Sym. 30%</td><td>100%</td><td>0.760</td><td>0.772</td><td>0.783</td><td>0.771</td><td>0.723</td><td>0.838</td></tr><tr><td>DeepSeek-V4-Pro</td><td>Sym. 30%</td><td>50%</td><td>0.658</td><td>0.758</td><td>0.760</td><td>0.747</td><td>0.703</td><td>0.755</td></tr><tr><td>DeepSeek-V4-Pro</td><td>FA 30%</td><td>100%</td><td>0.915</td><td>0.879</td><td>0.919</td><td>0.904</td><td>0.809</td><td>0.899</td></tr><tr><td>DeepSeek-V4-Pro</td><td>FA 30%</td><td>50%</td><td>0.859</td><td>0.880</td><td>0.892</td><td>0.877</td><td>0.794</td><td>0.892</td></tr><tr><td>DeepSeek-V4-Pro</td><td>FR 30%</td><td>100%</td><td>0.735</td><td>0.748</td><td>0.753</td><td>0.744</td><td>0.713</td><td>0.817</td></tr><tr><td>DeepSeek-V4-Pro</td><td>FR 30%</td><td>50%</td><td>0.685</td><td>0.746</td><td>0.750</td><td>0.734</td><td>0.702</td><td>0.765</td></tr><tr><td>Qwen2.5-7B</td><td>Clean</td><td>100%</td><td>0.835</td><td>0.867</td><td>1.000</td><td>0.985</td><td>0.938</td><td>0.855</td></tr><tr><td>Qwen2.5-7B</td><td>Clean</td><td>50%</td><td>0.885</td><td>0.871</td><td>0.913</td><td>0.946</td><td>0.887</td><td>0.784</td></tr><tr><td>Qwen2.5-7B</td><td>Sym. 15%</td><td>100%</td><td>0.884</td><td>0.860</td><td>0.902</td><td>0.900</td><td>0.876</td><td>0.864</td></tr><tr><td>Qwen2.5-7B</td><td>Sym. 15%</td><td>50%</td><td>0.874</td><td>0.869</td><td>0.917</td><td>0.914</td><td>0.879</td><td>0.828</td></tr><tr><td>Qwen2.5-7B</td><td>Sym. 30%</td><td>100%</td><td>0.886</td><td>0.852</td><td>0.874</td><td>0.874</td><td>0.852</td><td>0.845</td></tr><tr><td>Qwen2.5-7B</td><td>Sym. 30%</td><td>50%</td><td>0.860</td><td>0.858</td><td>0.867</td><td>0.869</td><td>0.830</td><td>0.760</td></tr><tr><td>Qwen2.5-7B</td><td>FA 30%</td><td>100%</td><td>0.849</td><td>0.872</td><td>0.907</td><td>0.908</td><td>0.895</td><td>0.813</td></tr><tr><td>Qwen2.5-7B</td><td>FA 30%</td><td>50%</td><td>0.888</td><td>0.866</td><td>0.904</td><td>0.922</td><td>0.910</td><td>0.787</td></tr><tr><td>Qwen2.5-7B</td><td>FR 30%</td><td>100%</td><td>0.889</td><td>0.809</td><td>0.846</td><td>0.846</td><td>0.862</td><td>0.803</td></tr><tr><td>Qwen2.5-7B</td><td>FR 30%</td><td>50%</td><td>0.855</td><td>0.833</td><td>0.851</td><td>0.854</td><td>0.818</td><td>0.752</td></tr></table>

Table 24: Complete L2 comparisons on the common question pool (Appendix E.5). CBF is higher-is-better. Clean denotes correct labels; Sym. flips both classes, FA only incorrect-answer labels, and FR only correct-answer labels. Coverage is the fraction of available labels. All methods use the same answer targets and observed labels within each model.

<table><tr><td>Model</td><td>Partition</td><td>Global</td><td>Beta</td><td>DS</td><td>GLAD-style</td></tr><tr><td>Qwen2.5-7B</td><td>random_five_subjects</td><td>0.857</td><td>0.881</td><td>0.781</td><td>0.724</td></tr><tr><td>Qwen2.5-7B</td><td>random_four_groups</td><td>0.892</td><td>0.897</td><td>0.865</td><td>0.653</td></tr><tr><td>Qwen2.5-7B</td><td>threshold_0.10</td><td>0.841</td><td>0.915</td><td>0.949</td><td>0.683</td></tr><tr><td>Qwen2.5-7B</td><td>threshold_0.15</td><td>0.815</td><td>0.911</td><td>0.898</td><td>0.753</td></tr><tr><td>Qwen2.5-7B</td><td>threshold_0.20</td><td>0.779</td><td>0.894</td><td>0.889</td><td>0.757</td></tr><tr><td>GPT-5.4-mini</td><td>random_five_subjects</td><td>0.864</td><td>0.865</td><td>0.763</td><td>0.817</td></tr><tr><td>GPT-5.4-mini</td><td>random_four_groups</td><td>0.940</td><td>0.933</td><td>0.791</td><td>0.824</td></tr><tr><td>GPT-5.4-mini</td><td>threshold_0.10</td><td>0.849</td><td>0.946</td><td>0.860</td><td>0.804</td></tr><tr><td>GPT-5.4-mini</td><td>threshold_0.15</td><td>0.818</td><td>0.933</td><td>0.836</td><td>0.835</td></tr><tr><td>GPT-5.4-mini</td><td>threshold_0.20</td><td>0.810</td><td>0.947</td><td>0.841</td><td>0.879</td></tr><tr><td>GPT-5.4</td><td>random_five_subjects</td><td>0.903</td><td>0.854</td><td>0.741</td><td>0.860</td></tr><tr><td>GPT-5.4</td><td>random_four_groups</td><td>0.959</td><td>0.946</td><td>0.782</td><td>0.851</td></tr><tr><td>GPT-5.4</td><td>threshold_0.10</td><td>0.880</td><td>0.953</td><td>0.839</td><td>0.885</td></tr><tr><td>GPT-5.4</td><td>threshold_0.15</td><td>0.869</td><td>0.896</td><td>0.836</td><td>0.882</td></tr><tr><td>GPT-5.4</td><td>threshold_0.20</td><td>0.977</td><td>0.977</td><td>0.835</td><td>0.838</td></tr><tr><td>GPT-5.5</td><td>random_five_subjects</td><td>0.922</td><td>0.873</td><td>0.792</td><td>0.910</td></tr><tr><td>GPT-5.5</td><td>random_four_groups</td><td>0.942</td><td>0.933</td><td>0.790</td><td>0.938</td></tr><tr><td>GPT-5.5</td><td>threshold_0.10</td><td>0.939</td><td>0.919</td><td>0.769</td><td>0.885</td></tr><tr><td>GPT-5.5</td><td>threshold_0.15</td><td>0.942</td><td>0.942</td><td>0.776</td><td>0.938</td></tr><tr><td>GPT-5.5</td><td>threshold_0.20</td><td>0.942</td><td>0.942</td><td>0.776</td><td>0.938</td></tr><tr><td>DeepSeek-V4-Pro</td><td>random_five_subjects</td><td>0.896</td><td>0.858</td><td>0.778</td><td>0.911</td></tr><tr><td>DeepSeek-V4-Pro</td><td>random_four_groups</td><td>0.918</td><td>0.911</td><td>0.796</td><td>0.938</td></tr><tr><td>DeepSeek-V4-Pro</td><td>threshold_0.10</td><td>0.919</td><td>0.900</td><td>0.632</td><td>0.878</td></tr><tr><td>DeepSeek-V4-Pro</td><td>threshold_0.15</td><td>0.918</td><td>0.918</td><td>0.770</td><td>0.938</td></tr><tr><td>DeepSeek-V4-Pro</td><td>threshold_0.20</td><td>0.918</td><td>0.918</td><td>0.770</td><td>0.938</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 25: Fixed-answer domain-partition sensitivity: pooled CBF, higher-is-better. Random conditions average 100 draws; threshold conditions use deviations of 10, 15, or 20 percentage points from model mean accuracy. Methods are compared within each partition; see Appendix E.6.

<table><tr><td>Model</td><td>Groups</td><td>Global</td><td>Beta</td><td>DS GLAD-style</td></tr><tr><td>Qwen2.5-7B</td><td>20</td><td>0.857</td><td>0.874 0.791</td><td>0.724</td></tr><tr><td>Qwen2.5-7B</td><td>11</td><td>0.864</td><td>0.877 0.831</td><td>0.700</td></tr><tr><td>Qwen2.5-7B</td><td>4</td><td>0.878</td><td>0.849 0.900</td><td>0.636</td></tr><tr><td>GPT-5.4-mini</td><td>20</td><td>0.861</td><td>0.859 0.849</td><td>0.811</td></tr><tr><td>GPT-5.4-mini</td><td>11</td><td>0.873</td><td>0.868 0.747</td><td>0.827</td></tr><tr><td>GPT-5.4-mini</td><td>4</td><td>0.940</td><td>0.920 0.836</td><td>0.781</td></tr><tr><td>GPT-5.4</td><td>20</td><td>0.908</td><td>0.851 0.781</td><td>0.857</td></tr><tr><td>GPT-5.4</td><td>11</td><td>0.908</td><td>0.879 0.829</td><td>0.853</td></tr><tr><td>GPT-5.4</td><td>4</td><td>0.965</td><td>0.856 0.831</td><td>0.850</td></tr><tr><td>GPT-5.5</td><td>20</td><td>0.931</td><td>0.873 0.818</td><td>0.909</td></tr><tr><td>GPT-5.5</td><td>11</td><td>0.951</td><td>0.911 0.795</td><td>0.920</td></tr><tr><td>GPT-5.5</td><td>4</td><td>0.956</td><td>0.895 0.828</td><td>0.924</td></tr><tr><td>DeepSeek-V4-Pro</td><td>20</td><td>0.904</td><td>0.860 0.819</td><td>0.913</td></tr><tr><td>DeepSeek-V4-Pro</td><td>11</td><td>0.923</td><td>0.884 0.797</td><td>0.922</td></tr><tr><td>DeepSeek-V4-Pro</td><td>4</td><td>0.945</td><td>0.915 0.799</td><td>0.911</td></tr></table>

Table 26: Pooled CBF after refitting on 20 subjects, 11 MMLU subcategories, and 4 MMLU categories. This semantic grouping complements the random and threshold partitions in Table 25; joint interpretation in Appendix E.6.

The Qwen-Max/Qwen-Turbo simulator comparison varies both generated utterances and their decodability. It is a within-family intervention, so it cannot separate shared-family effects from modelspecific response style.

Simulator identity and classifier coverage are therefore reported as distinct properties of each trace.

## E.9 Within-Family Simulator Comparison

We compare a Qwen-Max answering model under Qwen-Max and Qwen-Turbo feedback simulators. Both conditions contain ten sessions of 50 attempted turns. Table 30 is a descriptive comparison within the Qwen family; the 0.6 percentage-point answer-accuracy difference also reflects different recorded answer samples.

The displayed runs differ by .6 percentage points in task accuracy and .005 in CBF. They provide a descriptive within-family comparison. Because answers precede user feedback, similar raw answer confidence does not establish that the feedback is equally informative.

<table><tr><td>Model</td><td>FPR/FNR</td><td>Global</td><td>Beta</td><td>DS</td><td>GLAD-style</td></tr><tr><td>Qwen2.5-7B</td><td>0.00/0.00</td><td>0.749/0.871</td><td>0.895/0.997</td><td>0.977/0.919</td><td>0.934/0.854</td></tr><tr><td>Qwen2.5-7B</td><td>0.15/0.15</td><td>0.746/0.865</td><td>0.823/0.943</td><td>0.853/0.906</td><td>0.825/0.909</td></tr><tr><td>Qwen2.5-7B</td><td>0.30/0.30</td><td>0.739/0.854</td><td>0.764/0.895</td><td>0.757/0.886</td><td>0.727/0.924</td></tr><tr><td>Qwen2.5-7B</td><td>0.30/0.10</td><td>0.741/0.868</td><td>0.812/0.929</td><td>0.827/0.915</td><td>0.803/0.794</td></tr><tr><td>Qwen2.5-7B</td><td>0.10/0.30</td><td>0.723/0.815</td><td>0.784/0.856</td><td>0.795/0.886</td><td>0.762/0.819</td></tr><tr><td>GPT-5.4-mini</td><td>0.00/0.00</td><td>0.774/0.869</td><td>0.870/0.996</td><td>0.949/0.838</td><td>0.943/0.879</td></tr><tr><td>GPT-5.4-mini</td><td>0.15/0.15</td><td>0.756/0.854</td><td>0.795/0.917</td><td>0.834/0.797</td><td>0.835/0.908</td></tr><tr><td>GPT-5.4-mini</td><td>0.30/0.30</td><td>0.727/0.815</td><td>0.730/0.843</td><td>0.737/0.772</td><td>0.726/0.922</td></tr><tr><td>GPT-5.4-mini</td><td>0.30/0.10</td><td>0.770/0.869</td><td>0.803/0.942</td><td>0.829/0.813</td><td>0.840/0.835</td></tr><tr><td>GPT-5.4-mini</td><td>0.10/0.30</td><td>0.705/0.782</td><td>0.736/0.804</td><td>0.756/0.766</td><td>0.739/0.852</td></tr><tr><td>GPT-5.4</td><td>0.00/0.00</td><td>0.803/0.913</td><td>0.856/0.994</td><td>0.914/0.765</td><td>0.953/0.880</td></tr><tr><td>GPT-5.4</td><td>0.15/0.15</td><td>0.770/0.876</td><td>0.778/0.897</td><td>0.810/0.723</td><td>0.847/0.917</td></tr><tr><td>GPT-5.4</td><td>0.30/0.30</td><td>0.721/0.803</td><td>0.707/0.800</td><td>0.717/0.697</td><td>0.731/0.920</td></tr><tr><td>GPT-5.4</td><td>0.30/0.10</td><td>0.792/0.907</td><td>0.793/0.946</td><td>0.815/0.738</td><td>0.863/0.867</td></tr><tr><td>GPT-5.4</td><td>0.10/0.30</td><td>0.702/0.768</td><td>0.709/0.767</td><td>0.730/0.692</td><td>0.738/0.872</td></tr><tr><td>GPT-5.5</td><td>0.00/0.00</td><td>0.885/0.949</td><td>0.828/0.992</td><td>0.871/0.713</td><td>0.971/0.934</td></tr><tr><td>GPT-5.5</td><td>0.15/0.15</td><td>0.807/0.873</td><td>0.744/0.867</td><td>0.770/0.633</td><td>0.863/0.947</td></tr><tr><td>GPT-5.5</td><td>0.30/0.30</td><td>0.713/0.745</td><td>0.663/0.741</td><td>0.681/0.606</td><td>0.732/0.912</td></tr><tr><td>GPT-5.5</td><td>0.30/0.10</td><td>0.842/0.923</td><td>0.770/0.922</td><td>0.797/0.645</td><td>0.896/0.935</td></tr><tr><td>GPT-5.5</td><td>0.10/0.30</td><td>0.702/0.730</td><td>0.661/0.727</td><td>0.684/0.603</td><td>0.732/0.893</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.00/0.00</td><td>0.884/0.944</td><td>0.826/0.991</td><td>0.885/0.765</td><td>0.972/0.938</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.15/0.15</td><td>0.803/0.867</td><td>0.741/0.865</td><td>0.778/0.639</td><td>0.865/0.952</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.30/0.30</td><td>0.706/0.741</td><td>0.659/0.739</td><td>0.682/0.607</td><td>0.733/0.911</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.30/0.10</td><td>0.839/0.913</td><td>0.767/0.920</td><td>0.800/0.655</td><td>0.896/0.939</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.10/0.30</td><td>0.696/0.727</td><td>0.658/0.725</td><td>0.686/0.605</td><td>0.733/0.892</td></tr></table>

Table 27: CBF at 100% label coverage. Each cell gives session/pooled scores averaged over 100 repetitions; higher is better. FPR/FNR are false-positive/false-negative label rates. Compare Tables 28 and 29 for reduced coverage; analysis in Appendix E.7.
<table><tr><td>Model</td><td>FPR/FNR</td><td>Global</td><td>Beta</td><td>DS</td><td>GLAD-style</td></tr><tr><td>Qwen2.5-7B</td><td>0.00/0.00</td><td>0.744/0.870</td><td>0.813/0.965</td><td>0.808/0.903</td><td>0.804/0.848</td></tr><tr><td>Qwen2.5-7B</td><td>0.15/0.15</td><td>0.740/0.865</td><td>0.777/0.931</td><td>0.743/0.889</td><td>0.738/0.888</td></tr><tr><td>Qwen2.5-7B</td><td>0.30/0.30</td><td>0.734/0.854</td><td>0.745/0.888</td><td>0.681/0.859</td><td>0.674/0.889</td></tr><tr><td>Qwen2.5-7B</td><td>0.30/0.10</td><td>0.738/0.868</td><td>0.775/0.921</td><td>0.734/0.891</td><td>0.730/0.797</td></tr><tr><td>Qwen2.5-7B</td><td>0.10/0.30</td><td>0.718/0.814</td><td>0.756/0.854</td><td>0.703/0.868</td><td>0.696/0.809</td></tr><tr><td>GPT-5.4-mini</td><td>0.00/0.00</td><td>0.769/0.869</td><td>0.785/0.969</td><td>0.819/0.843</td><td>0.829/0.873</td></tr><tr><td>GPT-5.4-mini</td><td>0.15/0.15</td><td>0.749/0.854</td><td>0.744/0.908</td><td>0.746/0.801</td><td>0.750/0.896</td></tr><tr><td>GPT-5.4-mini</td><td>0.30/0.30</td><td>0.721/0.815</td><td>0.705/0.838</td><td>0.674/0.769</td><td>0.673/0.893</td></tr><tr><td>GPT-5.4-mini</td><td>0.30/0.10</td><td>0.765/0.869</td><td>0.754/0.933</td><td>0.755/0.815</td><td>0.765/0.839</td></tr><tr><td>GPT-5.4-mini</td><td>0.10/0.30</td><td>0.700/0.782</td><td>0.707/0.802</td><td>0.682/0.766</td><td>0.679/0.838</td></tr><tr><td>GPT-5.4</td><td>0.00/0.00</td><td>0.797/0.913</td><td>0.767/0.972</td><td>0.822/0.772</td><td>0.850/0.884</td></tr><tr><td>GPT-5.4</td><td>0.15/0.15</td><td>0.763/0.876</td><td>0.722/0.890</td><td>0.747/0.726</td><td>0.764/0.911</td></tr><tr><td>GPT-5.4</td><td>0.30/0.30</td><td>0.715/0.803</td><td>0.679/0.798</td><td>0.671/0.699</td><td>0.676/0.892</td></tr><tr><td>GPT-5.4</td><td>0.30/0.10</td><td>0.786/0.907</td><td>0.736/0.935</td><td>0.761/0.744</td><td>0.786/0.871</td></tr><tr><td>GPT-5.4</td><td>0.10/0.30</td><td>0.697/0.768</td><td>0.678/0.765</td><td>0.674/0.693</td><td>0.678/0.856</td></tr><tr><td>GPT-5.5</td><td>0.00/0.00</td><td>0.875/0.949</td><td>0.733/0.977</td><td>0.781/0.725</td><td>0.887/0.935</td></tr><tr><td>GPT-5.5</td><td>0.15/0.15</td><td>0.797/0.872</td><td>0.680/0.861</td><td>0.744/0.645</td><td>0.783/0.947</td></tr><tr><td>GPT-5.5</td><td>0.30/0.30</td><td>0.706/0.744</td><td>0.629/0.738</td><td>0.658/0.612</td><td>0.675/0.895</td></tr><tr><td>GPT-5.5</td><td>0.30/0.10</td><td>0.832/0.922</td><td>0.700/0.914</td><td>0.764/0.658</td><td>0.818/0.937</td></tr><tr><td>GPT-5.5</td><td>0.10/0.30</td><td>0.695/0.730</td><td>0.626/0.724</td><td>0.657/0.610</td><td>0.673/0.879</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.00/0.00</td><td>0.873/0.943</td><td>0.730/0.976</td><td>0.788/0.770</td><td>0.889/0.941</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.15/0.15</td><td>0.793/0.866</td><td>0.677/0.858</td><td>0.744/0.647</td><td>0.784/0.951</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.30/0.30</td><td>0.699/0.741</td><td>0.624/0.735</td><td>0.657/0.615</td><td>0.676/0.891</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.30/0.10</td><td>0.829/0.912</td><td>0.696/0.911</td><td>0.765/0.661</td><td>0.818/0.942</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.10/0.30</td><td>0.688/0.726</td><td>0.622/0.721</td><td>0.657/0.612</td><td>0.674/0.874</td></tr></table>

Table 28: CBF at 50% random label coverage, using the same models and noise channels as Table 27. Each cell gives session/pooled scores averaged over 100 repetitions. Joint coverage analysis: Appendix E.7.

<table><tr><td>Model</td><td>FPR/FNR</td><td>Global</td><td>Count</td><td>Beta</td><td>DS</td><td>GLAD-style</td></tr><tr><td>Qwen2.5-7B</td><td>0.00/0.00</td><td>0.725/0.869</td><td>0.735/0.905</td><td>0.759/0.917</td><td>0.736/0.851</td><td>0.736/0.808</td></tr><tr><td>Qwen2.5-7B</td><td>0.15/0.15</td><td>0.720/0.864</td><td>0.713/0.889</td><td>0.749/0.895</td><td>0.718/0.817</td><td>0.715/0.816</td></tr><tr><td>Qwen2.5-7B</td><td>0.30/0.30</td><td>0.714/0.853</td><td>0.692/0.853</td><td>0.740/0.860</td><td>0.700/0.759</td><td>0.693/0.791</td></tr><tr><td>Qwen2.5-7B</td><td>0.30/0.10</td><td>0.724/0.868</td><td>0.713/0.880</td><td>0.749/0.894</td><td>0.718/0.815</td><td>0.714/0.782</td></tr><tr><td>Qwen2.5-7B</td><td>0.10/0.30</td><td>0.701/0.814</td><td>0.699/0.827</td><td>0.743/0.837</td><td>0.707/0.780</td><td>0.700/0.750</td></tr><tr><td>GPT-5.4-mini</td><td>0.00/0.00</td><td>0.745/0.869</td><td>0.714/0.915</td><td>0.711/0.923</td><td>0.702/0.827</td><td>0.714/0.846</td></tr><tr><td>GPT-5.4-mini</td><td>0.15/0.15</td><td>0.724/0.853</td><td>0.687/0.880</td><td>0.699/0.876</td><td>0.682/0.778</td><td>0.687/0.844</td></tr><tr><td>GPT-5.4-mini</td><td>0.30/0.30</td><td>0.699/0.815</td><td>0.659/0.821</td><td>0.687/0.818</td><td>0.661/0.728</td><td>0.660/0.810</td></tr><tr><td>GPT-5.4-mini</td><td>0.30/0.10</td><td>0.743/0.869</td><td>0.695/0.900</td><td>0.703/0.902</td><td>0.687/0.790</td><td>0.695/0.836</td></tr><tr><td>GPT-5.4-mini</td><td>0.10/0.30</td><td>0.681/0.782</td><td>0.660/0.788</td><td>0.687/0.789</td><td>0.663/0.725</td><td>0.661/0.771</td></tr><tr><td>GPT-5.4</td><td>0.00/0.00</td><td>0.768/0.912</td><td>0.706/0.929</td><td>0.679/0.928</td><td>0.677/0.779</td><td>0.706/0.880</td></tr><tr><td>GPT-5.4</td><td>0.15/0.15</td><td>0.732/0.875</td><td>0.673/0.871</td><td>0.666/0.859</td><td>0.659/0.725</td><td>0.673/0.872</td></tr><tr><td>GPT-5.4</td><td>0.30/0.30</td><td>0.690/0.802</td><td>0.640/0.788</td><td>0.653/0.779</td><td>0.637/0.685</td><td>0.641/0.814</td></tr><tr><td>GPT-5.4</td><td>0.30/0.10</td><td>0.756/0.906</td><td>0.684/0.904</td><td>0.671/0.898</td><td>0.663/0.736</td><td>0.684/0.871</td></tr><tr><td>GPT-5.4</td><td>0.10/0.30</td><td>0.675/0.768</td><td>0.640/0.761</td><td>0.652/0.754</td><td>0.637/0.685</td><td>0.640/0.786</td></tr><tr><td>GPT-5.5</td><td>0.00/0.00</td><td>0.821/0.948</td><td>0.688/0.954</td><td>0.621/0.935</td><td>0.610/0.761</td><td>0.686/0.937</td></tr><tr><td>GPT-5.5</td><td>0.15/0.15</td><td>0.750/0.870</td><td>0.645/0.860</td><td>0.605/0.831</td><td>0.608/0.672</td><td>0.644/0.922</td></tr><tr><td>GPT-5.5</td><td>0.30/0.30</td><td>0.673/0.743</td><td>0.603/0.741</td><td>0.588/0.722</td><td>0.589/0.627</td><td>0.603/0.833</td></tr><tr><td>GPT-5.5</td><td>0.30/0.10</td><td>0.782/0.921</td><td>0.661/0.905</td><td>0.611/0.878</td><td>0.611/0.689</td><td>0.660/0.933</td></tr><tr><td>GPT-5.5</td><td>0.10/0.30</td><td>0.664/0.729</td><td>0.601/0.728</td><td>0.587/0.709</td><td>0.588/0.624</td><td>0.601/0.817</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.00/0.00</td><td>0.817/0.942</td><td>0.687/0.952</td><td>0.615/0.927</td><td>0.607/0.776</td><td>0.685/0.942</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.15/0.15</td><td>0.746/0.865</td><td>0.643/0.857</td><td>0.599/0.824</td><td>0.605/0.663</td><td>0.642/0.920</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.30/0.30</td><td>0.667/0.740</td><td>0.600/0.738</td><td>0.582/0.717</td><td>0.584/0.639</td><td>0.600/0.829</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.30/0.10</td><td>0.777/0.911</td><td>0.658/0.901</td><td>0.605/0.869</td><td>0.607/0.683</td><td>0.657/0.933</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.10/0.30</td><td>0.658/0.725</td><td>0.599/0.724</td><td>0.581/0.704</td><td>0.584/0.634</td><td>0.599/0.815</td></tr></table>

Table 29: CBF at 12.5% random label coverage, including direct Count. Each cell gives session/pooled scores averaged over 100 repetitions. Together with Tables 27 and 28, this tests the effect of sparse feedback (Appendix E.7).

<table><tr><td>Simulator</td><td>Acc</td><td>ECE</td><td>HR</td><td>CBF</td></tr><tr><td>user = Qwen-Max</td><td>78.2%</td><td>0.165</td><td>20.2%</td><td>0.717</td></tr><tr><td>user = Qwen-Turbo</td><td>77.6%</td><td>0.165</td><td>20.0%</td><td>0.722</td></tr><tr><td>|∆|</td><td>0.6pp</td><td>0.001</td><td>0.2pp</td><td>0.005</td></tr></table>

Table 30: Within-family simulator comparison for Qwen-Max. Recorded 500-turn runs with Qwen-Max and Qwen-Turbo simulators.

## E.10 R4 Adaptive Selection and Second-Order Estimators: Full Details

R4 chooses ZS when its measured CBF is higher than NoMem and LSA otherwise. We evaluate this descriptive selection on the 18 models for which both self-assessment panels are complete. Figure 11 separately compares the reliability-aware estimators under the matched session and pooled units.

Observed selections. R4 selects ZS on 14 of 18 complete models. Its mean CBF is 0.744, compared with 0.747 for always-ZS and 0.674 for

always-LSA.

Selection target. The score used to select a branch depends on the empirical domain outcomes. This makes R4 an outcome-informed diagnostic, even though its threshold is fixed.

Prediction. A deployment rule must choose without the test outcomes. The held-out ReAct comparison below evaluates profiles formed only from the training half of each subject.

Best-of-two reference. Choosing the higher observed score of ZS and LSA gives mean CBF 0.762 on these 18 models. This is a selection reference, not an implementable predictor without outcome information.

## F Cross-Task Heterogeneity

## F.1 Cross-Task Heterogeneity: Full Per-Task Comparison

Table 31 includes all 18 models and the three extension tasks. MMLU uses valid-answer accuracy and ECE. GSM8K reports the archived numericanswer score; HumanEval reports the fraction passing the custom test harness, and BFCL reports function-name/required-key matches. Gaps compare recorded confidence with each task’s score. For Qwen3.6-Plus, the archived per-item flags reproduce 182/300 on GSM8K and 101/164 on HumanEval. These lower scores alone do not identify an API or parser failure. The extension archives use fixed task denominators; failed or unparsed cases cannot be consistently separated without the missing full responses. Their scores are therefore reported under their own archived scoring protocol, separately from valid-answer primary QA accuracy.

![](images/1a3e2d44a25ee0dd2d8c85dbb050167ac67ee2cd0cda046ad94400e9214a1ac9.jpg)

![](images/f3e6c0a208a4ada2d67bed727a736617c50b85a73a1b6b2b330990d74d50681c.jpg)  
Figure 11: Reliability-aware estimation at two scoring units. Session and pooled CBF for all 18 models under a shared label stream.

## F.2 Recorded Cross-Task Scores

A sign change indicates opposite directions of the recorded confidence–score gap across tasks. We display all 18 rows, with unavailable confidence gaps left blank. The comparison describes task dependence under the extension scorers.

Twelve of the 16 rows with available confidence gaps have opposite recorded signs across the three extension tasks. These signs describe the stored summaries; the task scores and confidence extraction remain part of what is being measured.

The full task matrix supports comparisons within each recorded task protocol. Further task-specific validation is needed before using these profiles to

![](images/4074801996b0fcf23204948570da61c9a3aee9689cfc1d5d76dcb6133994f187.jpg)  
Figure 12: Task-dependent recorded confidence gaps. Confidence minus task score; gray cells indicate unavailable confidence. BFCL is the name/key control.

<table><tr><td rowspan="2">Model</td><td colspan="2">MMLU</td><td colspan="2">GSM8K</td><td colspan="2">HumanEval</td><td colspan="2">BFCL name/key</td></tr><tr><td>Acc</td><td>ECE</td><td>Acc</td><td>Gap</td><td>Custom tests</td><td>Gap</td><td>Acc</td><td>Gap</td></tr><tr><td>Mistral-7B</td><td>41.6</td><td>0.582</td><td>55</td><td>+31</td><td>31</td><td>+52</td><td>71</td><td>+15</td></tr><tr><td>Llama-3-8B</td><td>57.7</td><td>0.357</td><td>84</td><td>-3</td><td>33</td><td>+44</td><td>46</td><td>+36</td></tr><tr><td>GLM-5</td><td>95.8</td><td>0.039</td><td>96</td><td>+1</td><td>91</td><td>-49</td><td>64</td><td>+32</td></tr><tr><td>DeepSeek-V3.2</td><td>88.6</td><td>0.023</td><td>94</td><td>-21</td><td>87</td><td>-30</td><td>75</td><td>+5</td></tr><tr><td>Qwen-Turbo</td><td>75.5</td><td>0.188</td><td>95</td><td>-33</td><td>74</td><td>-1</td><td>72</td><td>+22</td></tr><tr><td>Qwen2.5-7B</td><td>60.6</td><td>0.259</td><td>90</td><td>-18</td><td>64</td><td>-4</td><td>73</td><td>+18</td></tr><tr><td>Qwen-Max</td><td>78.8</td><td>0.167</td><td>97</td><td>-18</td><td>82</td><td>-31</td><td>74</td><td>+17</td></tr><tr><td>Qwen-Plus</td><td>82.2</td><td>0.144</td><td>90</td><td>-11</td><td>92</td><td>-48</td><td>80</td><td>+10</td></tr><tr><td>Qwen3.5-27B</td><td>95.9</td><td>0.033</td><td>98</td><td>-1</td><td>77</td><td>-35</td><td>72</td><td>+24</td></tr><tr><td>GPT-5.4-mini</td><td>75.0</td><td>0.199</td><td>95</td><td>-2</td><td>74</td><td>-25</td><td>76</td><td>+15</td></tr><tr><td>GPT-5.4</td><td>82.8</td><td>0.132</td><td>97</td><td>+0.3</td><td>81</td><td>-36</td><td>70</td><td>+25</td></tr><tr><td>Claude-Sonnet-4.6</td><td>93.0</td><td>0.052</td><td>88</td><td>-4</td><td>79</td><td>-40</td><td>72</td><td>+17</td></tr><tr><td>Claude-Opus-4.6</td><td>93.0</td><td>0.020</td><td>99</td><td>-5</td><td>74</td><td>-17</td><td>78</td><td>+9</td></tr><tr><td>GPT-5.5</td><td>92.8</td><td>0.032</td><td>97</td><td>-1</td><td>85</td><td>-31</td><td>97</td><td>-4</td></tr><tr><td>Claude-Opus-4.7</td><td>90.4</td><td>0.022</td><td>99</td><td>-10</td><td>95</td><td>-34</td><td>99</td><td>-9</td></tr><tr><td>GLM-5.1</td><td>94.2</td><td>0.040</td><td>96</td><td>一</td><td>79</td><td></td><td>67</td><td></td></tr><tr><td>DeepSeek-V4-Pro</td><td>92.8</td><td>0.047</td><td>97</td><td></td><td>94</td><td></td><td>70</td><td></td></tr><tr><td>Qwen3.6-Plus</td><td>90.7</td><td>0.068</td><td>61</td><td>+18</td><td>62</td><td>+15</td><td>70</td><td>+27</td></tr></table>

Table 31: Four-task results for all 18 models. MMLU: valid-answer accuracy (%) and ECE; extension tasks: recorded task scores (%) and confidence minus score (percentage points), with BFCL using function-name/argument-key checks. Dashes denote unavailable confidence. Scoring and cross-task interpretation: Appendix F.1 and F.2.

<table><tr><td>Model</td><td>Gap</td><td>GSM8K HumanEval BFCL Gap</td><td>Gap</td><td>Sign-flip in subset?</td></tr><tr><td>Mistral-7B</td><td>+31</td><td>+52</td><td>+15</td><td>no (all +)</td></tr><tr><td>Llama-3-8B</td><td>-3</td><td>+44</td><td>+36</td><td>yes</td></tr><tr><td>GLM-5</td><td>+1</td><td>-49</td><td>+32</td><td>yes</td></tr><tr><td>DeepSeek-V3.2</td><td>-21</td><td>-30</td><td>+5</td><td>yes</td></tr><tr><td>Qwen-Turbo</td><td>-33</td><td>-1</td><td>+22</td><td>yes</td></tr><tr><td>Qwen2.5-7B</td><td>-18</td><td>-4</td><td>+18</td><td>yes</td></tr><tr><td>Qwen-Max</td><td>-18</td><td>-31</td><td>+17</td><td>yes</td></tr><tr><td>Qwen-Plus</td><td>-11</td><td>-48</td><td>+10</td><td>yes</td></tr><tr><td>Qwen3.5-27B</td><td>-1</td><td>-35</td><td>+24</td><td>yes</td></tr><tr><td>GPT-5.4-mini</td><td>-2</td><td>-25</td><td>+15</td><td>yes</td></tr><tr><td>GPT-5.4</td><td>+0.3</td><td>-36</td><td>+25</td><td>yes</td></tr><tr><td>Claude-Sonnet-4.6</td><td>-4</td><td>-40</td><td>+17</td><td>yes</td></tr><tr><td>Claude-Opus-4.6</td><td>-5</td><td>-17</td><td>+9</td><td>yes</td></tr><tr><td>GPT-5.5</td><td>-1</td><td>-31</td><td>-4</td><td>no (all −)</td></tr><tr><td>Claude-Opus-4.7 GLM-5.1</td><td>-10</td><td>-34</td><td>-9</td><td>no (all −)</td></tr><tr><td>DeepSeek-V4-Pro</td><td></td><td></td><td>一</td><td></td></tr><tr><td>Qwen3.6-Plus</td><td>+18</td><td>+15</td><td>+27</td><td>no (all +)</td></tr><tr><td>Opposite recorded signs (available rows)</td><td></td><td></td><td></td><td>12/16</td></tr></table>

Table 32: Recorded confidence–score gaps across the three extension tasks (pp). Results for 18 models; unavailable confidence gaps are unscored.

<table><tr><td>Metric</td><td>Pearson r (100 vs 500)</td><td>Spearman ρ</td></tr><tr><td>ECE</td><td>0.966</td><td>0.911</td></tr><tr><td>CBF</td><td>0.967</td><td>0.933</td></tr></table>

Table 33: Lite/full budget correlations on 23 recorded run configurations. Pearson and Spearman statistics compare 100 versus 500 records.

allocate tool-use tasks.

## F.3 Lite protocol: T=100 vs T=500 rankings

The stored lite-protocol analysis compares the first 100 and 500 records across 23 run configurations, including repeated model variants. Table 33 reports its rank correlations as a descriptive budgetsensitivity comparison.

The correlations show agreement between these finite-budget summaries. Correlation is not a percentage of information retained, and this runconfiguration analysis does not establish convergence of the 18-model session protocol.

## F.4 Held-Out ReAct Prediction

The six configurations contribute 960 stored records. For each subject, the first four questions form the profile and the last four define the prediction target. Both direct and ReAct answers pass the same A–D parsing gate; a subject requires at least two valid answers in both halves and modes. Methods share test support within configuration. Count and Beta use training outcomes, while Global uses the pooled training accuracy. Lower MAE is better.

<table><tr><td>Config.</td><td>Global</td><td>ZS</td><td>Direct Count</td><td>ReAct Count</td><td>ReAct Beta</td></tr><tr><td>qwen25_1_5b</td><td>0.190</td><td>0.446</td><td>0.258</td><td>0.308</td><td>0.229</td></tr><tr><td>qwen25_7b</td><td>0.237</td><td>0.252</td><td>0.346</td><td>0.285</td><td>0.268</td></tr><tr><td>qwen25_14b</td><td>0.233</td><td>0.248</td><td>0.267</td><td>0.254</td><td>0.231</td></tr><tr><td>qwen3_4b_2507</td><td>0.242</td><td>0.216</td><td>0.384</td><td>0.190</td><td>0.247</td></tr><tr><td>qwen3_8b</td><td>0.166</td><td>0.211</td><td>0.197</td><td>0.167</td><td>0.176</td></tr><tr><td>qwen3_14b</td><td>0.206</td><td>0.217</td><td>0.267</td><td>0.171</td><td>0.191</td></tr></table>

Table 34: Predicting held-out ReAct domain accuracy: MAE, lower-is-better. Columns compare a global training mean, the zero-shot prior, and domain profiles from direct-answer or ReAct training records; see Appendix F.4.

Table 34 gives Global the lowest MAE in three of six configurations, ReAct Count in two, and Re-Act Beta in one. For Qwen3-14B, ReAct Count improves on Global (.171 vs. .206), whereas Qwen2.5- 7B favors Global (.237 vs. .285). Domain profiles can therefore aid held-out prediction, but their advantage over a global estimate varies by configuration.

## G Implementation Notes and Extended Material

## G.1 Implementation Note: Algorithmic-Baseline self\_assess Specification

Interface specification. The algorithmic baselines return a domain probability through self\_assess(domain). UF-PDC returns the positive-label fraction, Bayes-Domain a Beta posterior mean, and EM-Joint the fitted latentcorrectness rate. Platt-Online and HistBin-Online retain per-answer recalibration but use the labelcount profile for domain CBF, explaining their identical CBF columns. DS and GLAD-style use the likelihoods in Appendix C.1.

## Per-baseline definition.

• Platt-Online / HistBin-Online: $\begin{array} { r l } { \hat { p } ( d ) } & { { } = } \end{array}$ dom\_correct[d]/dom\_total[d] (or 0.5 if no observations).

• Bayes-Domain: $\hat { p } ( d ) = \alpha [ d ] / ( \alpha [ d ] + \beta [ d ] )$

• EM-Joint: $\hat { p } ( d ) = \mathrm { c l i p } ( p ^ { * } [ d ] , 1 0 ^ { - 3 } , 1 - 1 0 ^ { - 3 } )$ from EM iterate.

• UF-PDC: pˆ(d) = correct[d]/total[d] from decoded feedback.

Per-baseline summary. UF-PDC, Platt, and HistBin return identical domain counts on matched labels, so the main table shares their endpoint column as Count\*. Their query-confidence mechanisms remain distinct. Bayes adds smoothing and EM-Joint fits latent reliability.

## G.2 Extended Related Work Discussion

This section provides the detailed discussion that was compressed in the main text’s Related Work section. We organize the literature into thematic clusters and position LADDER against each cluster’s design space.

LLM Self-Knowledge and Metacognition. Prior work studies several forms of self-knowledge. Kadavath et al. (2022) examine answer confidence through P(True); Li et al. (2025a) study monitoring and control of internal activations; and Binder et al. (2025) use fine-tuning to test models’ predictions of their own behavior. Behavioral evaluation without self-reports (Ackerman, 2026) finds that frontier models exhibit limited, context-dependent metacognitive signals when not explicitly prompted. Self-improvement framings (Liu and van der Schaar, 2025) argue that intrinsic metacognitive learning is a precondition for autonomous improvement. Most directly relevant, Barkan et al. (2026) test LLMs on BigCodeBench and SWE-Bench under multi-step tool execution and find systematic overconfidence, which worsens during execution for several frontier models. LADDER differs by measuring whether the agent can convert noisy interactive feedback into a stable per-domain self-model—an ability orthogonal to one-shot probing or clean execution traces.

Knowledge Boundaries and Calibration Benchmarks. Existing benchmarks evaluate whether LLMs know what they know on a static, per-query basis. Boundary taxonomies (Li et al., 2025b; Yin et al., 2024) categorize how prompts and model knowledge affect answerability. Staticfactuality benchmarks (Wei et al., 2024; Hendrycks et al., 2021; Lin et al., 2022) evaluate short-answer factuality, multi-task understanding, and adversarial truthfulness. Multi-dimensional aggregations (Liang et al., 2022) include ECE alongside other metrics, and surveys (Geng et al., 2024) compile method comparisons. None evaluates whether the agent’s self-model can be accumulated over noisy interaction; LADDER targets this gap by making per-domain accuracy estimation the explicit measurement axis.

LLM agent memory systems. Memory architectures and benchmarks (Packer et al., 2023; Chhikara et al., 2025; Li et al., 2025c; Tan et al., 2025; Xu et al., 2025; Zhou et al., 2026b; Hu et al., 2026) motivate persistent summaries of interaction. Domain reliability adds an evaluable summary whose target is the model’s observed answer accuracy.

Learning from Implicit Feedback. Three regimes characterize this line. Coactive learning formalizes the user-edit setting but assumes structured, reliable signals. Retrospective dialogue distillation (Liu et al., 2025) mines real interaction logs (WildChat, LMSYS), finding follow-up utterances often contain feedback but distillation outcomes vary with dialogue length—helpful on short MTBench questions, less so on longer Wild-Bench ones. RLHF-induced overconfidence (Leng et al., 2025) shows that the dominant feedbacklearning paradigm itself produces miscalibrated models. LADDER targets the orthogonal regime of inference-time feedback accumulation under noisy, domain-dependent reliability—a setting where neither training-time RLHF nor offline distillation applies.

Selective Prediction and Abstention. These studies relate uncertainty estimates to answering or abstaining. Fine-tuningfor uncertainty (Kapoor et al., 2024; Zhang et al., 2024) produces generalizable abstention behavior with as few as 1k LoRA samples or via R-Tuning’s meta-skill formulation. Self-reflective rationale generation (Xu et al., 2024) optimizes per-query confidence without persistent memory. Selective classification (Traub et al., 2024) introduces AUGRC; surveys (Wen et al., 2025) map the methodological landscape. Reasoning-induced regression (Kirichenko et al., 2025) shows that reasoning fine-tuning degrades abstention by an average of 24% on Abstention-Bench. Epistemic vs. aleatoric separation (Ahdritz et al., 2024) uses linear probes on frozen embeddings. They focus on query-level uncertainty; LADDER examines domain profiles formed from accumulated feedback.

Simulated Users and Closest Competitors. The closest competitors share LADDER’s goal of evaluating self-knowledge under realistic interaction but differ on simulator design or feedback channel. Simulator validity (Zhou et al., 2026a) quantifies the Sim2Real gap and shows LLM simulators bias toward an “easy mode” with positivity bias. Calibration through interaction (Fang et al., 2025) (CCG) studies game-theoretic confidence adjustment using feedback computed from verified correctness and reported confidence. Learning confidence from sampled answers (Han et al., 2024) (LePe) uses correctness frequencies against ground truth as confidence targets and requires offline weight updates. Per-query confidence elicitation (Xu et al., 2024) (SaySelf) lacks persistent memory. Capability boundary collapse in RL (Dong et al., 2025) formalizes a related failure mode under RL training. Cross-task calibration (Shen et al., 2024) seeks transferable confidence estimates, while group-dependent calibration (Chen et al., 2024; Manggala et al., 2025) addresses variation across question groups. LADDER occupies a position none covers: noisy interactive feedback, accumulation horizon, and second-order user-reliability inference, scored against an oraclebracketed ladder rather than a clean ground-truth signal.

## G.3 Model Configurations

The panel contains 18 models across seven families (Table 2). Local inference uses Mistral-7B-Instructv0.3, Meta-Llama-3-8B-Instruct, and Qwen2.5-7B-Instruct. Other models are accessed through their named API configurations. LSA and L4 return one domain-indexed JSON profile with a 1,100-token output budget; local decoding is greedy and API requests use temperature zero.

## G.4 Extended Results

Table 35 reports pooled CBF within each empirical subject tier for Llama-3-8B, Qwen-Turbo, and Qwen-Max. It compares the constant profile, prior, and Beta estimate on the same tier targets.

## G.5 Reproducibility Index

The reproducibility index links the principal outputs to their analysis inputs. The saved records and machine-readable results accompany the manuscript sources.

Supplementary materials. The supplementary materials provide scoring code, analysis inputs, and repeated draws for reproducing the principal results.

<table><tr><td>Model</td><td>Method</td><td>Strong</td><td>Medium</td><td>Weak</td></tr><tr><td>Llama-3-8B</td><td>NoMemory</td><td>0.746</td><td>0.946</td><td>0.870</td></tr><tr><td>Llama-3-8B</td><td>ZS-SK</td><td>0.846</td><td>0.901</td><td>0.865</td></tr><tr><td>Llama-3-8B</td><td>Beta</td><td>0.926</td><td>0.831</td><td>0.879</td></tr><tr><td>Qwen-Turbo</td><td>NoMemory</td><td>0.615</td><td>0.726</td><td>0.890</td></tr><tr><td>Qwen-Turbo</td><td>ZS-SK</td><td>0.949</td><td>0.839</td><td>0.714</td></tr><tr><td>Qwen-Turbo</td><td>Beta</td><td>0.916</td><td>0.912</td><td>0.905</td></tr><tr><td>Qwen-Max</td><td>NoMemory</td><td>0.627</td><td>0.709</td><td>0.826</td></tr><tr><td>Qwen-Max</td><td>ZS-SK</td><td>0.948</td><td>0.927</td><td>0.803</td></tr><tr><td>Qwen-Max</td><td>Beta</td><td>0.542</td><td>0.739</td><td>0.887</td></tr></table>

Table 35: Per-tier pooled CBF on the observed subject partition.
<table><tr><td>Output</td><td>Input</td><td>Operation</td></tr><tr><td>Tables 2, 3</td><td>QA records; self-assessment</td><td>Matched domain scoring</td></tr><tr><td>Table 16</td><td>profiles Same decoded-label</td><td>Session/pooled</td></tr><tr><td>Table 23</td><td>records Fixed answers; 100</td><td>refitting Session/pooled</td></tr><tr><td>Table 4</td><td>noise repetitions Empirical subject</td><td>CBF Per-model ranks</td></tr><tr><td></td><td>outcomes</td><td></td></tr><tr><td>Table 34 Figures 5, 6, Main-table profiles</td><td>960 ReAct records</td><td>Within-subject held-out scoring</td></tr></table>

Table 36: Reproducibility index for the primary analyses.

## H Reproducibility Checklist

## H.1 Reproducibility Checklist

The following index summarizes the empirical reporting components of the study.

Claims and contributions. Sections 3–4.3 connect the five information levels to measured domain-profile error. Limitations discusses the task scope and the use of simulated feedback.

Data and code. The empirical material comprises 45,000 primary interaction records, permodel session summaries, baseline implementations, self-assessment replies, and the extension and sensitivity analyses. The scoring scripts reuse the stored answer and feedback streams. Model request configurations appear in Appendix G.3.

Experimental protocol. Main-study CBF reconstructs observed session accuracy. Held-out prediction uses a separate train/test split in the ReAct analysis. Appendix C.1 specifies estimator updates; Appendix D.2 reports resampling and paired contrasts. Noise and partition experiments retain the results of each repetition.

Repeatability. Stored traces permit deterministic comparisons of estimators on the same observations. New LLM calls can vary with sampling and service versions. Simulator comparisons and held-out predictions are reported as separate experiments in Appendices E.8 and F.4.

Ethics and broader impact. The primary interaction study uses simulated personas. Humanfeedback evaluation is a separate study requiring its own documented procedures. The benchmark measures domain-estimation performance and does not certify safe deployment.

## I Worked Examples and Prompt Templates

This appendix gives a worked tour of the protocol so that readers can verify the data shape and reproduce the LLM calls. §I.1 shows two real session excerpts (one mid-tier model, one frontier model). §I.2 catalogs the most common decoder and selfknowledge failure modes encountered in the released traces. §I.3 provides the answer/simulator templates, the feedback-rule specification, and current self-assessment prompts.

## I.1 Sample session trajectories

We show five turns from each of two recorded sessions. Each excerpt contains the agent’s answer, reported confidence, verified outcome, and simulateduser feedback. In-expertise and out-of-expertise denote the assigned user persona’s relationship to the subject.

Example 1: Qwen-Plus, biology persona. Persona expertise: college biology, high school biology, and anatomy; style academic, tolerance 0.2, explicitness 0.9.

• T1 high school biology (in-expertise). Q: “During which phase of the cell cycle does the quantity of DNA in a eukaryotic cell typically double?” Agent: D, conf 1.00 (✓). User: “That’s correct, the quantity of DNA in a eukaryotic cell doubles during the S phase of the cell cycle.”

• T2 high school chemistry (out). Q: organicacid functional group. Agent: C, conf 1.00 (✓). User: “Could you elaborate on that answer? I’m familiar with the concept but I’d like to be sure I understand which specific functional group you’re referring to.”

• T3 abstract algebra (out). Q: integral domain R implies R[x] integral domain. Agent: C, conf

1.00 (✓). User: “The answer C seems reasonable, but could you elaborate on how both statements are related and why that leads to the conclusion?”

• T4 global facts (out). Q: % of Americans valuing free media. Agent: C, conf 0.85 (wrong). User: “I see, thank you for the information. It’s reassuring to know that a significant portion of Americans value the importance of a free press.”

• T5 high school chemistry (out). Q: pH of a weak monoprotic acid. Agent: None, conf 0.50 (excluded: no parsed answer). User: “Thank you for the information. I’ll proceed with the assumption that the pKa value provided is correct. . . ”

T4 contains a wrong answer followed by a polite acknowledgement, illustrating why conversational agreement need not be a correctness label. T5 has an unparsed answer and is excluded from the validanswer accuracy target.

Example 2: GPT-5.4, biology persona. Same persona as Example 1; agent is a frontier model.

• T1 clinical knowledge (out). Q: ulnar nerve palsy. Agent: C, conf 0.96 (✓). User: “That makes sense—could you elaborate on what specifically happens in ulnar nerve palsy that makes C the correct answer?”

• T2 clinical knowledge (out). Q: chain-ofsurvival ordering. Agent: B, conf 0.99 (✓). User: “That makes sense—could you walk me through why that particular ordering is considered optimal. . . ?”

• T3 conceptual physics (out). Q: Doppler effect. Agent: C, conf 0.99 (✓). User: “That makes sense, though I’m curious what the full answer choice C said. . . ”

• T4 high school chemistry (out). Q: paradichlorobenzene IUPAC name. Agent: D, conf 0.99 (✓). User: “That makes sense—could you walk me through the IUPAC naming logic. . . ?”

• T5 astronomy (out). Q: when the next major impact will occur. Agent: D, conf 0.94 (✓). User: “That’s an interesting answer, though I’d love to hear more about the reasoning behind it. . . ”

GPT-5.4 reports confidence of at least .94 on all five displayed turns. Several user replies begin with “That makes sense”, but explicit agreement, follow-up questions, and correctness judgments play different roles in the feedback stream. Only utterances satisfying the decoder’s positive/negative rules supply algorithmic labels.

## I.2 Failure mode catalog

The examples illustrate several ways in which confidence, feedback, and domain estimates can diverge.

1. Acknowledgment without verification. Polite agreement need not certify answer correctness.

2. Saturated confidence. Little variation in reported confidence limits its usefulness for distinguishing outcomes.

3. Unparsed answers. Missing answer options are excluded from accuracy and retained in the attempt count.

4. Ambiguous feedback. Follow-up questions can leave a domain with few decoded labels.

5. Prior–history disagreement. A history-based estimate can be worse than a zero-shot profile, as on Qwen-Plus in Table 2.

## I.3 Prompt templates

The answer and simulator templates illustrate the interaction format. The self-assessment prompts below specify the inputs used in the completeprofile collection.

Agent answer prompt. Used at every turn to elicit an answer and a confidence value.

Answer the question and give your   
,→ confidence 0-100%.   
Question: {question}   
Choices:   
{choices\_one\_per\_line}   
Format your response exactly as:   
Answer: [A/B/C/D]   
Confidence: [0-100]

Simulated user prompt. The persona receives the current question, the agent’s answer, and a sampled stage direction.

Your previous question: {question}   
The assistant's answer: {agent\_answer}   
[Stage direction - do NOT quote]   
{scenario}   
Now write your natural next response   
,→ (1-2 sentences):

Feedback decoding. The primary rule classifier checks negative patterns before uncertainty and positive patterns; unmatched utterances are neutral. API error strings are missing feedback. Only positive and negative labels enter the estimators.

The complete implementation is distributed with the reproducibility scripts. These labels measure rule-decoded stance, not human-annotated semantic correctness.

## L1 JSON response format.

Without any interaction history or task   
outcomes, estimate your expected,→   
,→ multiple-choice accuracy in each of   
,→ these MMLU subjects. Return only a   
JSON object mapping each listed,→   
domain to a probability in [0,1].,→   
,→ Domains: {domains}

## L2 LSA and L4 supplied-statistic prompts. Both return one domain profile at the session endpoint.

```csv
Estimate what fraction of your answers
was correct in each domain of this,→
,→ session, using the following
question and user-feedback history.,→
Return only one JSON object mapping,→
each listed domain to a probability,→
in [0,1].,→
{history}
```

Report the following supplied per-domain   
accuracies exactly. Return only one,→   
JSON object mapping the same domain,→   
names to probabilities in [0,1].,→   
{statistics}