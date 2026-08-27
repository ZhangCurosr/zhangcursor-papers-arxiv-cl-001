# FinRiskAtlas: Decision-Aligned Evaluation of Large Language Models for Financial Risk Review

Suyang Zhong<sup>1,2,†</sup>, Jingzhe Zhu<sup>1,†</sup>, Qi Xu<sup>1,3,†</sup>, Liyao Sun<sup>1</sup>, Yin Wang<sup>4</sup>, Qingqing Sun<sup>1,\*</sup>, Shuai Chen<sup>1,\*</sup>, and Tianyi Zhang<sup>1,\*</sup>

<sup>1</sup>Ant International, <sup>2</sup>Xiamen University, <sup>3</sup>Shanghai University, <sup>4</sup>Tongji University

<sup>†</sup> These authors contributed equally to this work.

<sup>\*</sup> Corresponding authors to whom correspondence should be addressed.

Deploying large language models for professional financial review requires more than measuring general financial competence: models must perform the specific review operation required by a workflow and determine whether the available evidence is suficient for a defensible decision. Existing financial benchmarks provide broad coverage of knowledge, reasoning, compliance, and professional tasks, but their evaluation units are often organized around datasets or task formulations rather than the professional decisions that deployed systems support. We introduce FinRiskAtlas, a Chinese-language benchmark that evaluates financial LLMs through two complementary dimensions: operation execution under fixed evidence states and evidence-state control under evolving review conditions. The static benchmark contains 9,742 instances across 53 task families, including 42 Domain Knowledge families and eleven downstream review operations defined by explicit evaluation contracts. FinRisk-Ask extends this framework through ofline replay of 680 pre-action states from 104 de-identified professional trajectories, where future evidence is withheld during inference and used only to construct expert-verified evidence targets. Across 33 model configurations, operation-level evaluation produces non-redundant model rankings, with a mean pairwise Spearman correlation of 0.42 across downstream operations, and knowledge-based shortlisting can incur up to 18.01 points of regret on individual operations. FinRisk-Ask further shows that entering the Ask branch more frequently does not necessarily yield better request targeting or stronger end-to-end evidence acquisition. These results demonstrate that broad financial capability scores do not fully capture where models are reliable in professional workflows, motivating evaluation units aligned with the decisions and evidence states that deployed systems must support.

## 1. Introduction

Financial risk and compliance review is a sequence of evidence-dependent professional decisions rather than a single prediction task. A reviewer may need to identify transaction parties, extract decision-relevant evidence, verify financial quantities, determine applicable provisions, assign risk categories, or produce a supported professional recommendation. These operations may rely on overlapping records, but they resolve diferent professional questions and produce diferent artifacts for downstream decisions. Their reliability also depends on the evidence available at the time of review. For example, when documentation establishing a transaction party’s beneficial ownership is missing, a reliable system should recognize that the current evidence state is insuficient and request the necessary information before producing a risk judgment, rather than forcing a conclusion from an incomplete record.

This setting creates two fundamental questions for deploying large language models (LLMs) in professional financial workflows. The first is operation execution: given the currently visible record, which model configuration can reliably perform the required review operation? The second is evidence-state control: given an evolving review state, should the workflow proceed with the available information or acquire additional evidence before making a decision? These capabilities are related but fundamentally diferent. A model may retrieve the correct regulation yet apply it to the wrong entity, accurately extract requested fields yet fail to integrate them into a case-level assessment, or correctly recognize that information is missing while requesting evidence unrelated to the actual unresolved issue. Therefore, broad financial capability alone does not determine where a model configuration is reliable within a professional review workflow.

![](images/18ffb4c6582e4bb0b05a639cb472aece4929854f8669d893b749e6ff71b70483.jpg)  
Figure 1: Composition of FinRiskAtlas. The static benchmark evaluates whether a model can execute a specified financial review operation under a fixed evidence state. FinRisk-Ask evaluates whether a model should proceed or acquire additional evidence from an intermediate review state and, after choosing to ask, whether the request targets a trajectory-supported unresolved need. Static instances and trajectory states represent diferent evaluation units and are evaluated separately.

Existing financial LLM benchmarks have substantially expanded coverage of financial language understanding and numerical reasoning Shah et al. (2022), Chen et al. (2021, 2022), Zhu et al. (2021), broad financial knowledge, prediction, and professional applications Xie et al. (2023, 2024), Matlin et al. (2025), Nie et al. (2025), Guo et al. (2025), and compliance, safety, and tool-supported research Ding et al. (2026), Dou et al. (2026), Kim et al. (2026), Bigeard et al. (2025). These benchmarks provide valuable measurements of general financial competence. However, their primary evaluation units are commonly organized around knowledge domains, source datasets, or task formulations rather than the professional decisions that models are expected to support in deployment Xie et al. (2023, 2024), Nie et al. (2025), Guo et al. (2025), Ding et al. (2026). Such units often do not explicitly represent the evidence boundary available to the model, the decision object being resolved, the reviewer artifact being produced, or the criterion by which the output is evaluated. As a result, two tasks derived from the same case record may correspond to diferent professional operations, while tasks with diferent surface formats may represent the same operational capability. This creates an evaluation-unit mismatch: a benchmark score may indicate whether a model performs well on a task, but not whether that model is appropriate for a particular decision point in a financial review workflow.

A second limitation concerns the evolution of evidence during professional review. Many fixed-input financial evaluations implicitly assume that the provided record is suficient for solving the evaluated task Xie et al. (2023, 2024), Nie et al. (2025), Guo et al. (2025). Real-world financial risk-control processes, however, are iterative: evidence is collected, verified, reconciled, and expanded before decisions are finalized. Prior work on clarification and selective answering Rao and Daumé III (2018), Cole et al. (2023), abstention, expert deferral, and help seeking Feng et al. (2024), Mozannar and Sontag (2020), Ren et al. (2023), and active information acquisition Li et al. (2024, 2025), Zhou et al. (2025), Zhao et al. (2026) demonstrates the importance of recognizing underspecification. However, recognizing that information is missing is only the first step of evidence acquisition. In professional review, a useful request must identify the specific evidence gap that blocks the current decision, respect the temporal boundary of the review state, and request information that can advance the workflow. Thus, deciding whether to acquire evidence and determining what evidence to acquire represent distinct capabilities.

To address the first limitation, we introduce FinRiskAtlas, a benchmark that organizes financial LLM evaluation around fine-grained professional operations in financial risk-control and compliance review. FinRiskAtlas therefore takes a professional review operation as the downstream evaluation unit, rather than a source dataset or task format. Each downstream family is defined before model evaluation through an explicit contract specifying the visible evidence regime, professional decision object, required reviewer artifact, and scoring protocol. This design allows downstream scores to be interpreted as operation-specific performance signals rather than generic capability measurements. Unlike task-oriented evaluation, operation-oriented evaluation asks whether a model is suitable for the specific decision it is deployed to support. The static FinRiskAtlas benchmark contains 9,742 instances across 53 task families. Forty-two Domain Knowledge families, comprising 4,679 instances, evaluate the concepts, regulations, obligations, and risk mechanisms required for financial risk-control review. Eleven downstream operation families contain 5,063 instances, covering Evidence-Grounded Processing and Applied Review operations, including information extraction, entity matching, quantitative reasoning, risk classification, rule grounding, legal-outcome prediction, and professional review generation. FinRiskAtlas combines broad summary signals with operation-specific evaluation: the Domain Knowledge macro supports coarse screening, while the eleven downstream operations retain their task-native scores for fine-grained configuration selection.

To address the second limitation, we introduce FinRisk-Ask, a complementary benchmark for evidence state control through ofline replay of professional review trajectories. FinRisk-Ask contains 680 pre-action states reconstructed from 104 completed, de-identified trajectories, including 583 recorded Ask states and 97 recorded Proceed states. Candidate models observe only the case record and review history available before the recorded action; later evidence, subsequent reviewer actions, and downstream outcomes are withheld during inference. For each retained Ask state, later-observed evidence is used only to construct expert-verified unresolved evidence targets. Unlike incomplete-information benchmarks constructed from clinical dialogues or deliberately underspecified reasoning problems Li et al. (2024, 2025), Zhou et al. (2025), Zhao et al. (2026), FinRisk-Ask grounds request targets in later-observed evidence from completed professional trajectories and evaluates whether a model can acquire the evidence required to advance a professional decision under a realistic information boundary. The evaluation target is not the optimal question in an abstract dialogue, but the evidence request required to advance a concrete professional review decision.

Experiments across 33 model configurations demonstrate why both evaluation dimensions are necessary. The eleven fixed-evidence operations produce non-redundant configuration rankings, with a mean pairwise Spearman correlation of 0.42. Broad knowledge performance does not uniformly preserve operation-specific selection: restricting selection to the Domain Knowledge leader can forgo up to 18.01 points on an individual downstream operation. Evidence-state evaluation reveals another source of divergence. FinRisk-Ask uses ERA as the primary end-to-end measure of the evidence-acquisition pathway: a model receives credit only when it enters the recorded Ask branch and generates a request aligned with an expert-verified unresolved need. ERA is further decomposed into recorded-Ask recall and Conditional Request Alignment, allowing branch-entry failures to be distinguished from request-targeting failures. The results show that similar action-selection behavior can still yield substantially diferent end-to-end evidence-acquisition performance. Our contributions are as follows:

• An operation-centric evaluation paradigm for financial review. We introduce FinRiskAtlas, which defines professional review operations as the fundamental evaluation unit for financial LLM assessment. By explicitly modeling the evidence boundary, decision object, reviewer artifact, and scoring protocol of each operation, FinRiskAtlas provides fine-grained evaluation signals aligned with real financial risk-control decisions.

• A trajectory-grounded benchmark for evidence-state control. We introduce FinRisk-Ask, an ofline replay framework that evaluates whether models should proceed or acquire additional evidence and whether their requests target expert-verified unresolved needs, while preventing access to future trajectory information. ERA provides the primary end-to-end measure of the evidence-acquisition pathway, while action- and request-level metrics diagnose where end-to-end performance is gained or lost.

• Evidence that benchmark granularity afects model selection. Across 33 configurations, we demonstrate that operation-level evaluation and evidence-state evaluation reveal diferences hidden by broad capability scores, exposing distinct strengths and failure modes of LLMs in professional financial workflows.

## 2. Related Work

## 2.1. Financial LLM Evaluation: From Capability Coverage to Workflow Operations

Financial LLM evaluation has progressed from focused capability assessment to broader measurements of financial knowledge, reasoning, and professional applications. Early benchmarks isolate specific abilities. FLUE evaluates financial language understanding; FinQA and ConvFinQA study numerical reasoning over financial reports and conversational contexts; and TAT-QA combines textual and tabular evidence for financial reasoning Shah et al. (2022), Chen et al. (2021, 2022), Zhu et al. (2021). FinanceReasoning and BizBench further extend evaluation toward complex financial and business reasoning Tang et al. (2025), Krumdick et al. (2024). These benchmarks establish the importance of measuring financial capabilities beyond general language evaluation.

Subsequent benchmark suites broaden coverage across heterogeneous financial tasks. PIXIU and FinBen evaluate multiple financial capabilities including understanding, reasoning, prediction, and application, while FLaME, CFinBench, and FinEval further expand evaluation of financial knowledge and reasoning, including Chinese financial scenarios Xie et al. (2023, 2024), Matlin et al. (2025), Nie et al. (2025), Guo et al. (2025). Such benchmarks provide valuable capability profiles, but their evaluation units are typically inherited from knowledge categories, source datasets, or task formulations. Consequently, they may not directly correspond to the professional decisions supported by a deployed system. A set of tasks derived from the same case record may represent diferent review operations, while tasks with diferent surface formats may serve the same workflow purpose.

Recent work moves closer to realistic financial applications. FinED-Bench evaluates financial document error detection, while Finance Agent Benchmark studies multi-step financial research supported by external tools He et al. (2026), Bigeard et al. (2025). CNFinBench jointly considers financial capability, compliance, and safety; FinGuard derives compliance evaluation from financial regulations; and FinRED studies financespecific red-team behavior Ding et al. (2026), Dou et al. (2026), Kim et al. (2026). These eforts introduce realistic documents, safety considerations, and specialized professional scenarios. However, they generally retain capability- or task-oriented evaluation units, leaving open how benchmark organization should reflect the operational decisions supported by deployed systems.

## 2.2. Diagnostic Evaluation and Decision-Aligned Measurement

Beyond benchmark coverage, prior work has emphasized the importance of structured diagnostic evaluation. CheckList organizes behavioral tests around specific linguistic capabilities and expected model behaviors, while HELM evaluates models across diverse scenarios and multiple metrics under transparent protocols Ribeiro et al. (2020), Liang et al. (2023). These studies show that aggregate scores alone can hide meaningful diferences across capabilities.

Professional-domain benchmarks similarly adopt structured taxonomies to expose specialized abilities. C-Eval organizes evaluation by knowledge subjects, while LawBench, LegalBench, and DiagnosisArena construct domain-specific evaluations for legal and clinical reasoning Huang et al. (2023), Fei et al. (2024), Guha et al. (2023), Zhu et al. (2026). These benchmarks demonstrate the value of expert-defined capability structures.

However, existing diagnostic frameworks primarily organize evaluation around capabilities or scenarios rather than the professional operations and evidence boundaries involved in deployment. Extending diagnostic evaluation toward workflow-aligned measurement remains an open direction for professional applications Ribeiro et al. (2020), Liang et al. (2023), Fei et al. (2024), Guha et al. (2023).

## 2.3. From Abstention to Active Evidence Acquisition

A separate research line studies model behavior when available information is insuficient for reliable answering. Clarification-question formulation and selective answering methods investigate how models handle ambiguity or unanswerable inputs Rao and Daumé III (2018), Zhang et al. (2024), Zhao et al. (2024), Cole et al. (2023). Abstention, expert deferral, and help-seeking approaches similarly study whether models can recognize uncertainty and appropriately transfer or delay decisions Feng et al. (2024), Mozannar and Sontag (2020), Ren et al. (2023).

Recent benchmarks evaluate information acquisition more directly. MediQ studies follow-up question generation in clinical diagnosis, QuestBench evaluates whether models identify missing variables required for reasoning, and active-reasoning benchmarks examine when and what information should be requested under incomplete conditions Li et al. (2024, 2025), Zhou et al. (2025), Zhao et al. (2026). RealFin further studies whether financial questions contain unavailable necessary premises Dai et al. (2026). These works demonstrate that answering a fully specified problem and identifying missing information are distinct abilities.

Trajectory-based approaches provide another perspective by using expert interaction histories. Learn-to-Ask uses sequential expert demonstrations and later observations to learn proactive information-seeking policies Wei et al. (2026). FinRisk-Ask shares the use of completed trajectories and later evidence observations, but difers in objective and evaluation setting. It does not learn a policy or optimize future interaction. Instead, it performs ofline evaluation of a recorded professional boundary, measuring whether a model should proceed or acquire evidence and whether the requested evidence addresses an unresolved review need. These directions leave open how evidence acquisition behavior should be evaluated within a professional workflow, where the objective is not only to recognize missing information but also to acquire evidence required for a specific decision. This highlights the need for evaluation frameworks that jointly consider the operation being performed and the evidence state from which that operation is reached.

## 3. The FinRiskAtlas Benchmark

## 3.1. Benchmark Scope and Evaluation Units

FinRiskAtlas is built around the observation that model selection in professional workflows occurs at the level of decisions rather than isolated task formats. A useful benchmark unit should therefore correspond to a stable decision context: what information is available to the model, what decision must be resolved, what artifact is expected from the reviewer, and how the result should be evaluated. This design principle motivates operation-centric evaluation, where the benchmark unit reflects the professional action that a model is expected to support. The distinction is important because a deployment decision is not determined by whether a model performs well on an isolated task, but by whether it can reliably produce the artifact required at a specific workflow stage. Changing the evaluation unit changes the selection signal available to practitioners.

![](images/41ce1e02706b98fafa09f079ae5211b6087e7a58adec004b0fa3ce3155a96a16.jpg)  
Figure 2: Benchmark design around operation and evidence state. Professional source materials are organized through an operation-aligned taxonomy and fixed evaluation contracts to construct the static benchmark. FinRisk-Ask complements fixed-evidence operation evaluation through ofline replay of modelvisible pre-action states, measuring both the Ask-or-Proceed transition and the alignment of generated requests with later-observed, expert-verified evidence targets. Later evidence is withheld during inference and used only for target construction.

Figure 2 illustrates the two evaluation settings in FinRiskAtlas. The static benchmark evaluates fixedevidence review operations, while FinRisk-Ask evaluates control of evolving evidence states. These two settings share the same principle: evaluation should be aligned with the decision context rather than only the surface form of the task.

The static benchmark contains three nested units. A capability layer groups families according to their role in financial review. A task family is the finest stable analysis unit, representing either a review-relevant knowledge area or a concrete fixed-evidence review operation. An instance is one evaluation item governed by the contract of its family. The static benchmark contains 53 families: 42 Domain Knowledge families and eleven downstream operation families. FinRisk-Ask is a separate state-level evaluation setting and is not counted as an additional static family.

To operationalize this principle, we define each family through an explicit evaluation contract:

$$
\Gamma _ { f } = ( c _ { f } , \mathscr { T } _ { f } , d _ { f } , \mathscr { y } _ { f } , s _ { f } ) ,\tag{1}
$$

where $c _ { f }$ specifies the capability being measured, $\mathcal { T } _ { f }$ defines the model-visible information regime, $d _ { f }$ specifies the professional decision object, $\mathcal { V } _ { f }$ defines the required reviewer-artifact schema, and $s _ { f }$ specifies the fixed parsing and scoring protocol. The information regime includes the evidence sources, fields, and temporal boundary available to the model rather than merely the surface format of the input.

Table 1: Operation coverage in FinRiskAtlas. The static benchmark contains eleven fixed-evidence review operations. FinRisk-Ask evaluates the complementary state-level decision of whether the current evidence state should be advanced or expanded.
<table><tr><td>Evaluation component</td><td>Operations</td><td>Workflow outputs</td></tr><tr><td>Evidence-Grounded cessing</td><td>Pro- Case classification; information extraction; institution matching; person matching; quantitative reasoning</td><td>Procedure routing, structured evidence fields, entity-identity decisions, and normalized quantities</td></tr><tr><td>Applied Review</td><td>Risk classification; applicable-provision selection; legal-outcome prediction; legal-judgment generation; disputed-issue generation; decision-view generation</td><td>Risk grades, governing provisions, dispositions, reasoned analyses, issue lists, and review opinions</td></tr><tr><td>State-level extension</td><td>Ask-or-Proceed decision, with one focused</td><td>Advancement of the current state or a targeted</td></tr><tr><td>FinRisk-Ask</td><td>evidence request after selecting Ask</td><td>evidence request</td></tr></table>

The contract defines the measurement boundary rather than a fixed execution order for every review case. Two tasks may originate from the same source case but belong to diferent families when they resolve diferent decisions or produce diferent artifacts. Conversely, records with diferent surface formats may belong to the same family when they instantiate the same decision context. Therefore, family boundaries are determined by professional function rather than incidental properties of source data.

## 3.2. Operation-Aligned Capability Taxonomy

The static benchmark organizes financial risk-review capability into three layers:

Domain Knowledge → Evidence-Grounded Processing → Applied Review.

The arrows indicate information dependency rather than a mandatory linear workflow. Individual cases may skip, repeat, or enter conditional branches, and the taxonomy organizes evaluation coverage rather than prescribing the execution order of every professional review process.

Families are grouped according to their role in review. Table 1 provides a compact overview of the eleven fixed-evidence downstream operations and the complementary state-level decision evaluated by FinRisk-Ask. Detailed operation contracts, including the model-visible information regime, professional decision object, reviewer artifact, and scoring protocol, are reported in Appendix B, Table 7. Domain Knowledge is treated as a supporting layer because its families evaluate concepts, rules, obligations, and risk mechanisms rather than workflow actions. The taxonomy and operation definitions were fixed by domain experts before candidate-model evaluation.

Domain Knowledge. The Domain Knowledge layer evaluates conceptual, regulatory, and risk-related knowledge required for financial risk and compliance review. It contains 42 task families with 4,679 instances distributed across sixteen taxonomy groups, including financial risk, fraud, money laundering, compliance, international trade and finance, financial technology and security, commerce and payments, and financerelated law. These families provide the knowledge foundation required for interpreting evidence and making professional decisions.

Evidence-Grounded Processing. This layer evaluates whether models can transform heterogeneous financial and legal records into structured, decision-relevant evidence. It contains five task families with 1,502 instances: case classification, information extraction, institution matching, person matching, and quantitative reasoning.

Applied Review. This layer evaluates whether models can integrate available evidence, apply relevant rules, and produce professional review outputs. It contains six task families with 3,561 instances: risk classification, legal-outcome prediction, legal-judgment generation, applicable-provision selection, decision-view generation, and disputed-issue generation.

The static benchmark therefore contains 53 families and 9,742 instances. Domain Knowledge is summarized by an unweighted macro-average over its 42 families, while downstream operations retain their task-native metrics. Because downstream operations correspond to diferent reviewer artifacts and decision objectives, FinRiskAtlas does not collapse them into a single aggregate score.

## 3.3. Capability-First Construction and Quality Control

FinRiskAtlas follows a capability-first construction process. We first define the review-relevant capability or professional operation, establish its evaluation contract, identify suitable source materials, construct candidate instances, and perform expert review. Therefore, family boundaries are determined by intended professional function rather than by available data sources or output formats.

The benchmark draws from four source categories: professional examinations and textbooks; regulatory documents and industry guidance; business, transaction, entity, and legal records; and completed financialrisk review trajectories. Static instances are constructed through three routes: normalization of expertauthored questions, structured transformation of financial and legal records into evidence-processing tasks, and source-conditioned construction of case-level review tasks when the expected output is supported by visible evidence or documented professional decisions.

Importantly, family definitions and evaluation contracts are fixed before candidate-model evaluation. Construction decisions are therefore independent of model predictions and cannot be adapted according to observed model behavior. Six domain experts, including four experts in financial risk and compliance review and two experts in legal review of financial disputes, defined the taxonomy, reviewed family boundaries, audited candidate items, and resolved disagreements through consensus.

Quality control is conducted at both family and instance levels. Family-level review verifies that items within a family share the same capability target, information regime, decision object, reviewer artifact, and scoring protocol. Instance-level review checks evidence support, ambiguity, duplication, information leakage, parser validity, and provenance. Items without a stable reference or reproducible scoring procedure are removed.

## 3.4. FinRisk-Ask: Controlling an Evolving Evidence State

FinRisk-Ask extends the operation-centric evaluation principle to settings where the evidence state itself changes during review. It is not an interactive dialogue benchmark or a policy-learning framework of the kinds studied in prior information-acquisition work Li et al. (2024), Zhou et al. (2025), Wei et al. (2026); instead, it performs ofline evaluation of whether a model can make the appropriate evidence-state transition under a fixed historical decision boundary. The goal is not to optimize future interaction, but to measure whether a model can recognize when additional evidence is required and request evidence that supports a concrete professional decision.

The static benchmark assumes that the visible evidence state is fixed. FinRisk-Ask evaluates whether that state should be advanced or expanded through ofline replay of completed, de-identified financial-risk review trajectories collected from an enterprise risk-control workflow.

For a decision point t, let $C _ { t }$ denote the case record and review history visible immediately before the recorded action. Candidate models observe only $C _ { t } ;$ later evidence, subsequent actions, and downstream outcomes are withheld during inference. The model selects either to proceed with the available record or to request additional evidence.

A retained state must satisfy three conditions: the pre-action context can be reconstructed without later information, the immediately following reviewer action maps unambiguously to Ask or Proceed, and no post-action information is included in the model-visible context. Ask states require an additional condition:

later evidence must be verified by experts as unavailable at the decision point and relevant to an unresolved review need. Proceed states do not require request targets because their evaluated action is the decision to advance without initiating evidence acquisition.

The released FinRisk-Ask benchmark contains 680 states from 104 trajectories: 583 recorded Ask states and 97 recorded Proceed states. Every released Ask state supports both action evaluation and request evaluation. The reference action represents the recorded professional transition in one workflow; it does not claim to be the unique optimal review policy.

For each retained Ask state, the benchmark constructs an expert-verified set of later-observed evidence needs that were unavailable and unresolved at the replayed decision point. Candidate models never observe these targets during inference. Requests are evaluated against the underlying evidence need rather than the historical wording of the reviewer request, allowing semantically equivalent requests to receive credit.

FinRisk-Ask evaluates end-to-end evidence acquisition on recorded Ask states with ERA, which assigns credit only when a model both enters the Ask branch and requests evidence aligned with a verified unresolved need. CRA complements ERA by isolating request-targeting capability conditional on entering the Ask branch, while BAcc, AskR, and ProceedR characterize agreement with the recorded action boundary and its two branches. Together, these metrics complement static operation-level assessment with a structured evaluation of evidence-state control.

## 4. Experiments

## 4.1. Experimental Setup

Evaluation protocol. We evaluate the static FinRiskAtlas benchmark and FinRisk-Ask over 33 model config urations. All candidate evaluations use zero-shot direct-answer inference without in-context demonstrations or requested chain-of-thought. Each model–instance pair is evaluated once under the archived inference configuration, and malformed or unparseable outputs remain in the evaluation denominator. Open-ended Applied Review tasks and FinRisk-Ask request alignment are evaluated with a fixed semantic evaluator config uration. Reported results should therefore be interpreted as point estimates for the evaluated configurations rather than repeated-sampling estimates.

Models and comparison sets. The evaluation suite contains 33 configurations spanning open-weight checkpoints and API systems across multiple scales and model families, including Qwen, DeepSeek, Kimi, GLM, GPT, Gemini, Claude, Nemotron, Ling, Seed, Dianjin, and FinR1. The evaluation unit is the archived configuration rather than an abstract model family because serving configurations and inference settings can afect observed behavior. For analyses involving semantic evaluation, DeepSeek-V4-Flash is excluded from comparative rankings because it also serves as the fixed evaluator. All reported comparative analyses involving judge-scored operations therefore use the remaining 32 configurations. Let $\mathcal { M } _ { \mathrm { c m p } }$ denote the 32 non-evaluator configurations used in analyses involving judge-scored tasks. For each downstream operation $^ { O , }$ let $\mathcal { M } _ { o }$ denote its eligible configuration set: all 33 configurations for the eight structured operations and $\mathcal { M } _ { \mathrm { c m p } }$ for the three judge-scored operations. Configurations are ranked within $\mathcal { M } _ { o }$ in descending order of their Domain Knowledge macro score $K _ { m }$ . We let $\breve { r } _ { m , o } ^ { K }$ denote the resulting competition rank, with rank 1 assigned to the highest score and tied configurations receiving the same rank.

Metrics. For static operations, we retain each operation’s native metric because downstream families correspond to diferent reviewer artifacts and decision objectives. For FinRisk-Ask, Evidence-Request Alignment (ERA) measures end-to-end acquisition on recorded Ask states by jointly accounting for Ask-branch entry and alignment of the resulting request with an expert-verified evidence need. Conditional Request Alignment (CRA) complements ERA by measuring request-targeting capability conditional on entering the Ask branch, making it useful for comparing how efectively diferent configurations formulate evidence requests. Balanced Recorded-Action Agreement (BAcc), Ask recall (AskR), and Proceed recall (ProceedR) characterize agreement with the recorded action boundary and its two branches. Formal definitions are provided in Appendix C.

![](images/3c8b93eaf62d51dc5af8388c4ea96cb35912a995e15d2ffcbe29b2248aab91cd.jpg)

![](images/31f74c919e7d1b99abef2732a3b86bb7827930c8916da2de5c0edf905a16a6c1.jpg)

![](images/9509205faecd3c2c1d9201693ae76c52b85e575c5953f9dfe8b9ef7509f60ec7.jpg)  
Figure 3: Operation-level profiles across evaluated configurations. (a) Within-column percentile heatmap, with configurations ordered by the Domain Knowledge macro; hatched cells denote evaluator self-scores excluded from evaluator-dependent comparisons. (b) Raw downstream-operation scores for the knowledgenear pair Gemini-3-Flash-preview and Kimi-K3. (c) Spearman correlations between the Domain Knowledge macro and downstream operations, using 33 configurations for structured operations and 32 for judge-scored operations.

Research questions. The experiments investigate three consequences of decision-aligned evaluation. RQ1: How much operation-specific variation exists across downstream review operations? RQ2: How well does broad financial knowledge preserve downstream configuration selection? RQ3: How do action selection and request targeting jointly determine evidence-state performance?

## 4.2. RQ1: Operation-Level Evaluation Reveals Heterogeneous Model Strengths

The downstream review operations induce substantially diferent configuration rankings despite being evaluated on the same configuration pool. Figure 3(a) visualizes this behavior by ordering configurations according to their Domain Knowledge macro scores and displaying their within-operation percentile ranks. Rather than preserving a common ordering across downstream operations, many configurations change relative positions between evidence-processing and applied-review tasks, suggesting that diferent operations emphasize diferent capabilities.

We quantify this variation by computing the $1 1 \times 1 1$ Spearman rank-correlation matrix over $\mathcal { M } _ { \mathrm { c m p } }$ using the complete downstream operation scores (Tables 9 and 10). The mean pairwise correlation is $\rho = 0 . 4 2$ , with 37 of the 55 operation pairs exhibiting correlations below 0.5. Some operations exhibit almost independent rankings. For example, institution matching and risk classification have a Spearman correlation of only $\rho = - 0 . 0 3$ . The eigenspectrum of the same correlation matrix, shown in Figure 5(b), provides the same conclusion from a complementary perspective. The leading component explains 49.9% of the total eigenvalue mass, whereas the first five components explain 85.0%, indicating that downstream operations share a common capability component but cannot be reduced to a single latent ordering.

a  
![](images/f96eb47e21834cda3004969168ab8a76d48dbdfb4472bb2dd1152b8f45397677.jpg)

b  
![](images/e2fa5f50dfb9d46e5d67b912b294f2abecd7686ac9e9255c5bea48ee645267da.jpg)

c  
![](images/fa36a276f90ddf8528f2998ac03ceab430c35c860dad8ab8e17d7ff70558e52e.jpg)

d  
![](images/35c5b7b1b6378896ab1fefa529142ef226a855b065b8bc441db0a2d39a020faf.jpg)  
Figure 4: Evidence-state behavior and operation-specific screening regret. (a) Relationship between Balanced Recorded-Action Agreement and Evidence-Request Alignment across configurations. (b) Decomposition of request alignment into recorded-Ask recall and Conditional Request Alignment. (c–d) Score forgone under Domain-Knowledge-based shortlisting across diferent shortlist sizes.

Figure 3(c) shows that the correlation between the Domain Knowledge macro and downstream operations varies substantially, ranging from 0.33 for institution matching to 0.89 for case classification. This efect is also visible among configurations with nearly identical knowledge performance. Gemini-3-Flash-preview and Kimi K3 difer by only 0.01 points on the Domain Knowledge macro, yet Gemini-3-Flash-preview leads by 14.10 points on applicable-provision selection, whereas Kimi-K3 leads by 8.47 points on legal-outcome prediction. Consistent with this observation, knowledge-near configuration pairs (within 0.5 Domain Knowledge points) still exhibit a median downstream profile diference of 25.5 percentile points (Figure 5(a)). Moreover, the eleven downstream operation leaders are distributed across nine diferent configurations (Figure 5(d)), indicating that no single configuration consistently dominates all review operations. These observations show that operation-level evaluation provides configuration-selection signals that are not preserved uniformly by broad financial capability scores.

## 4.3. RQ2: Knowledge-Based Screening Incurs Operation-Dependent Regret

The heterogeneous rankings observed in RQ1 suggest that knowledge-based screening may not preserve the configurations preferred by individual downstream operations. We therefore quantify the resulting deployment cost by measuring the regret incurred when candidate configurations are shortlisted solely according to their Domain Knowledge ranking.

For operation $^ { O , }$ let $S _ { m , o }$ denote the score of configuration m. Restricting selection to the top-k configurations under the Domain Knowledge ranking, the observed shortlist regret is defined as

$$
\mathrm { R e g } _ { o } ( k ) = \operatorname* { m a x } _ { m \in \mathcal { M } _ { o } } S _ { m , o } - \operatorname* { m a x } _ { m \in \mathcal { M } _ { o } } S _ { m , o } .\tag{2}
$$

Ties at the rank threshold are retained, so the set $\{ m \ \in \ \mathcal { M } _ { o } \ : \ r _ { m , o } ^ { K } \ \leq \ k \}$ may contain more than k configurations. This convention avoids arbitrary tie-breaking. Because downstream operations follow heterogeneous evaluation contracts, regret is reported in each operation’s native score units. Figure $4 ( \mathrm { c } , \mathrm { d } )$ summarizes the resulting operation-specific regret curves. The cost of knowledge-based screening varies substantially across downstream operations. $\operatorname { A t } k = 1$ , selecting only the Domain Knowledge leader incurs no regret for institution matching and risk classification, but forfeits 18.01 points on information extraction, 11.21 points on quantitative reasoning, and 8.47 points on legal-outcome prediction. Increasing the shortlist reduces, but does not eliminate, these gaps. For example, decision-view generation still retains 6.70 points of regret even at $k = 1 5$ . These results suggest that the Domain Knowledge macro is well suited for coarse screening, whereas selecting the final deployment configuration benefits from operation-level evaluation.

## 4.4. RQ3: Evidence-State Control Separates Action Selection from Request Targeting

Evidence-state control requires both selecting the appropriate evidence-state transition and generating a request that targets the unresolved evidence need. FinRisk-Ask evaluates these behaviors through comple mentary metrics. ERA measures the end-to-end acquisition outcome on recorded Ask states, BAcc measures agreement with the full recorded Ask-or-Proceed boundary, and AskR and CRA expose branch entry and conditional request alignment.

Recorded-action agreement does not determine end-to-end evidence acquisition. Figure 4(a) shows that, among configuration pairs whose BAcc values difer by no more than 0.1 points, ERA difers by as much as 28.31 points. Configurations can therefore reproduce the recorded action boundary at similar rates while difering substantially in whether their behavior culminates in a request aligned with the unresolved evidence need. Figure 4(b) makes the decomposition of ERA explicit by plotting recorded-Ask recall against Conditional Request Alignment and overlaying iso-ERA contours. Configurations in the upper-right region combine broad coverage of recorded Ask states with strong request targeting and therefore achieve high end-to-end alignment. Ling-2.6-1T attains the highest ERA by combining an AskR of 96.57% with a CRA of 82.58%. Kimi-K3 and Claude-Opus-4.7 occupy a similar high-coverage and high-alignment region, yielding ERA scores of 77.10% and 75.98%, respectively. By contrast, Qwen3.7-Max achieves the highest CRA at 87.67% but enters the recorded Ask branch on only 57.80% of recorded Ask states, limiting its ERA to 50.68%. The panel therefore shows that strong conditional request alignment alone is insuficient for end-to-end evidence acquisition: high ERA requires both reliable entry into the Ask branch and a request that targets the unresolved evidence need.

The same separation is visible in the aggregate action behavior. Across all 32 non-evaluator configurations, AskR exceeds ProceedR. The median Ask–Proceed recall gap is 46.26 points, and 24 configurations exhibit gaps larger than 20 points. This systematic asymmetry indicates that the evaluated configurations reproduce recorded requests for additional evidence more readily than recorded decisions to proceed with the current record. It also motivates reporting BAcc together with the two branch-specific recalls rather than relying on overall action agreement alone. Taken together, ERA provides the primary end-to-end measure of evidence acquisition on recorded Ask states, while BAcc characterizes agreement with the full recorded action boundary and AskR, ProceedR, and CRA diagnose where performance is gained or lost. These results demonstrate that evidence-state control is not a single capability, but the combination of selecting an appropriate evidence-state transition and acquiring evidence that resolves the underlying professional decision.

## 5. Conclusion

FinRiskAtlas studies financial LLM evaluation through the lens of professional decision making. It evaluates whether a model configuration can perform the required review operation and control the evidence state from which that decision is made. The static benchmark provides operation-aligned evaluation under fixed evidence conditions, while FinRisk-Ask extends evaluation to evolving states where models must determine whether additional evidence is needed and whether the requested evidence addresses the underlying decision need. Our experiments show that these two dimensions reveal capability diferences hidden by broad benchmark scores. Diferent review operations induce distinct model profiles, and evidence-state evaluation separates recognizing the need for information from acquiring the right information. These findings suggest that selecting LLMs for professional financial workflows requires evaluation units aligned with deployment decisions rather than relying only on general financial competence. FinRiskAtlas provides the contracts, review operations, reconstructed evidence states, and reproducible evaluation protocols needed to study this perspective. We hope this perspective encourages future benchmarks to align evaluation units more closely with the professional decisions that deployed language models are expected to support.

## References

Antoine Bigeard, Langston Nashold, Rayan Krishnan, and Shirley Wu. Finance agent benchmark: Benchmarking llms on real-world financial research tasks. arXiv preprint arXiv:2508.00828, 2025. URL https://arxiv.org/abs/2508.00828.

Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, and William Yang Wang. FinQA: A dataset of numerical reasoning over financial data. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 3697–3711. Association for Computational Linguistics, 2021. doi: 10.18653/ v1/2021.emnlp-main.300. URL https://aclanthology.org/2021.emnlp-main.300/.

Zhiyu Chen, Shiyang Li, Charese Smiley, Zhiqiang Ma, Sameena Shah, and William Yang Wang. ConvFinQA: Exploring the chain of numerical reasoning in conversational finance question answering. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 6279– 6292. Association for Computational Linguistics, 2022. doi: 10.18653/v1/2022.emnlp-main.421. URL https://aclanthology.org/2022.emnlp-main.421/.

Jeremy Cole, Michael Zhang, Daniel Gillick, Julian Eisenschlos, Bhuwan Dhingra, and Jacob Eisenstein. Selectively answering ambiguous questions. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 530–543. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.emnlp-main.35. URL https://aclanthology.org/2023.emnlp-main.35/.

Yuyang Dai, Yan Lin, Zhuohan Xie, and Yuxia Wang. RealFin: How well do LLMs reason about finance when users leave things unsaid? In Findings of the Associationfor Computational Linguistics: ACL 2026, pages 25050–25080, San Diego, California, United States, July 2026. Association for Computational Linguistics. doi: 10.18653/v1/2026.findings-acl.1255. URL https://aclanthology.org/2026.findings-acl. 1255/.

Jinru Ding, Chao Ding, Yidong Jiang, Wenrao Pang, Boyi Xiao, Zhiqiang Liu, Jiayuan Chen, Yun Zhong, Tiantian Yuan, Junming Guan, Dawei Cheng, and Jie Xu. Beyond knowledge to agency: Evaluating expertise, autonomy, and integrity in finance with CNFinBench. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, pages 8765–8776. Association for Computing Machinery, 2026. doi: 10.1145/3770855.3817482. URL https://doi.org/10.1145/3770855.3817482.

Huaixia Dou, Jie Zhu, Minghao Wu, Shuo Jiang, Junhui Li, Lifan Guo, Feng Chen, and Chi Zhang. FinGuard: Detecting financial regulatory non-compliance in LLM interactions. arXiv preprint arXiv:2605.29427, 2026. URL https://arxiv.org/abs/2605.29427.

Zhiwei Fei, Xiaoyu Shen, Dawei Zhu, Fengzhe Zhou, Zhuo Han, Alan Huang, Songyang Zhang, Kai Chen, Zhixin Yin, Zongwen Shen, Jidong Ge, and Vincent Ng. LawBench: Benchmarking legal knowledge of large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 7933–7962. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024. emnlp-main.452. URL https://aclanthology.org/2024.emnlp-main.452/.

Shangbin Feng, Weijia Shi, Yike Wang, Wenxuan Ding, Vidhisha Balachandran, and Yulia Tsvetkov. Don’t hallucinate, abstain: Identifying LLM knowledge gaps via multi-LLM collaboration. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14664–14690. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.acl-long.786. URL https://aclanthology.org/2024.acl-long.786/.

Neel Guha, Julian Nyarko, Daniel E. Ho, Christopher Ré, Adam Chilton, Aditya Narayana, Alex Chohlas-Wood, Austin Peters, Brandon Waldon, Daniel N. Rockmore, Diego Zambrano, Dmitry Talisman, Enam Hoque, Faiz Surani, Frank Fagan, Galit Sarfaty, Gregory M. Dickinson, Haggai Porat, Jason Hegland, Jessica Wu, Joe Nudell, Joel Niklaus, John Nay, Jonathan H. Choi, Kevin Tobia, Margaret Hagan, Megan Ma, Michael Livermore, Nikon Rasumov-Rahe, Nils Holzenberger, Noam Kolt, Peter Henderson, Sean Rehaag, Sharad Goel, Shang Gao, Spencer Williams, Sunny Gandhi, Tom Zur, Varun Iyer, and Zehua Li. LegalBench: A collaboratively built benchmark for measuring legal reasoning in large language models. In Advances in Neural Information Processing Systems, volume 36, pages 44123–44279. Curran Associates, Inc., 2023. doi: 10.52202/075280-1915. URL https://proceedings.neurips.cc/paper\_files/paper/2023/ hash/89e44582fd28ddfea1ea4dcb0ebbf4b0-Abstract-Datasets\_and\_Benchmarks.html.

Xin Guo, Haotian Xia, Zhaowei Liu, Hanyang Cao, Zhi Yang, Zhiqiang Liu, Sizhe Wang, Jinyi Niu, Chuqi Wang, Yanhui Wang, Xiaolong Liang, Xiaoming Huang, Bing Zhu, Zhongyu Wei, Yun Chen, Weining Shen, and Liwen Zhang. FinEval: A Chinese financial domain knowledge evaluation benchmark for large language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6258–6292. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025.naacl-long.318. URL https://aclanthology.org/2025.naacl-long.318/.

Ying He, Zhouhong Gu, Zhecheng Hu, Yubo Zhou, Hao Shen, Jiaqing Liang, Zhaoqian Dai, Shuguang Ma, Fei Yu, Yanghua Xiao, and Zhixu Li. Are large language models reliable reviewers? a benchmark for error detection in financial documents. In Findings of the Associationfor Computational Linguistics: ACL 2026, pages 29625–29643. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026. findings-acl.1481. URL https://aclanthology.org/2026.findings-acl.1481/.

Yuzhen Huang, Yuzhuo Bai, Zhihao Zhu, Junlei Zhang, Jinghan Zhang, Tangjun Su, Junteng Liu, Chuancheng Lv, Yikai Zhang, Jiayi Lei, Yao Fu, Maosong Sun, and Junxian He. C-Eval: A multi level multi-discipline chinese evaluation suite for foundation models. In Advances in Neural Information Processing Systems, volume 36, pages 62991–63010. Curran Associates, Inc., 2023. doi: 10.52202/075280-2749. URL https://proceedings.neurips.cc/paper\_files/paper/2023/ hash/c6ec1844bec96d6d32ae95ae694e23d8-Abstract-Datasets\_and\_Benchmarks.html.

Chaeyun Kim, Dae-Young Park, Junghwan Kim, Jinyoung Jeong, Eunji Song, YongTaek Lim, and Minwoo Kim. FinRED: An expert-guided benchmark generation and evaluation framework for financial LLM red-teaming. arXiv preprint arXiv:2606.19887, 2026. URL https://arxiv.org/abs/2606.19887.

Michael Krumdick, Rik Koncel-Kedziorski, Viet Dac Lai, Varshini Reddy, Charles Lovering, and Chris Tanner. BizBench: A quantitative reasoning benchmark for business and finance. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8309– 8332. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.acl-long.452. URL https://aclanthology.org/2024.acl-long.452/.

Belinda Z. Li, Been Kim, and Zi Wang. QuestBench: Can LLMs ask the right question to acquire information in reasoning tasks? In Advances in Neural Information Processing Systems, volume 38. Curran Associates, Inc., 2025. doi: 10.52202/085713-4503. URL https://proceedings.neurips.cc/paper\_ files/paper/2025/hash/c42c8d51556fabb4b57fc86d3d3d0d09-Abstract-Datasets\_and\_ Benchmarks\_Track.html.

Shuyue Stella Li, Vidhisha Balachandran, Shangbin Feng, Jonathan S. Ilgen, Emma Pierson, Pang Wei Koh, and Yulia Tsvetkov. MediQ: Question-asking LLMs and a benchmark for reliable interactive clinical reasoning.

In Advances in Neural Information Processing Systems, volume 37, pages 28858–28888. Curran Associates, Inc., 2024. doi: 10.52202/079017-0908. URL https://proceedings.neurips.cc/paper\_files/ paper/2024/hash/32b80425554e081204e5988ab1c97e9a-Abstract-Conference.html.

Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Alexander Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew Arad Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri S. Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Andrew Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, and Yuta Koreeda. Holistic evaluation of language models. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview. net/forum?id=iO4LZibEqW.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-eval: NLG evaluation using GPT-4 with better human alignment. In Proceedings of the 2023 Conference on Empirical Method in Natural Language Processing, pages 2511–2522. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.emnlp-main.153. URL https://aclanthology.org/2023.emnlp-main.153/.

Glenn Matlin, Mika Okamoto, Huzaifa Pardawala, Yang Yang, and Sudheer Chava. Financial language model evaluation (FLaME). In Findings of the Association for Computational Linguistics: ACL 2025, pages 22633–22679. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025.findings-acl.1164. URL https://aclanthology.org/2025.findings-acl.1164/.

Hussein Mozannar and David Sontag. Consistent estimators for learning to defer to an expert. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 7076–7087. PMLR, 2020. URL https://proceedings.mlr.press/v119/ mozannar20b.html.

Ying Nie, Binwei Yan, Tianyu Guo, Hao Liu, Haoyu Wang, Wei He, Binfan Zheng, Weihao Wang, Qiang Li, Weijian Sun, Yunhe Wang, and Dacheng Tao. CFinBench: A comprehensive Chinese financial benchmark for large language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 876–891. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025.naacl-long.40. URL https://aclanthology.org/2025.naacl-long.40/.

Arjun Panickssery, Samuel R. Bowman, and Shi Feng. LLM evaluators recognize and favor their own generations. In Advances in Neural Information Processing Systems, volume 37, pages 68772–68802. Curran Asso ciates, Inc., 2024. doi: 10.52202/079017-2197. URL https://proceedings.neurips.cc/paper\_ files/paper/2024/hash/7f1f0218e45f5414c79c0679633e47bc-Abstract-Conference. html.

Sudha Rao and Hal Daumé III. Learning to ask good questions: Ranking clarification questions using neural expected value of perfect information. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2737–2746. Association for Computational Linguistics, 2018. doi: 10.18653/v1/P18-1255. URL https://aclanthology.org/P18-1255/.

Allen Z. Ren, Anushri Dixit, Alexandra Bodrova, Sumeet Singh, Stephen Tu, Noah Brown, Peng Xu, Leila Takayama, Fei Xia, Jake Varley, Zhenjia Xu, Dorsa Sadigh, Andy Zeng, and Anirudha Majumdar. Robots

that ask for help: Uncertainty alignment for large language model planners. In Proceedings of the 7th Conference on Robot Learning, volume 229 of Proceedings of Machine Learning Research, pages 661–682. PMLR, 2023. URL https://proceedings.mlr.press/v229/ren23a.html.

Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. Beyond accuracy: Behavioral testing of NLP models with CheckList. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4902–4912. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020.acl-main.442. URL https://aclanthology.org/2020.acl-main.442/.

Raj Shah, Kunal Chawla, Dheeraj Eidnani, Agam Shah, Wendi Du, Sudheer Chava, Natraj Raman, Charese Smiley, Jiaao Chen, and Diyi Yang. When FLUE meets FLANG: Benchmarks and large pretrained language model for financial domain. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 2322–2335. Association for Computational Linguistics, 2022. doi: 10.18653/v1/2022. emnlp-main.148. URL https://aclanthology.org/2022.emnlp-main.148/.

Zichen Tang, Haihong E, Ziyan Ma, Haoyang He, Jiacheng Liu, Zhongjun Yang, Zihua Rong, Rongjin Li, Kun Ji, Qing Huang, Xinyang Hu, Yang Liu, and Qianhe Zheng. FinanceReasoning: Benchmarking financial numerical reasoning more credible, comprehensive and challenging. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15721– 15749. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025.acl-long.766. URL https://aclanthology.org/2025.acl-long.766/.

Fei Wei, Daoyuan Chen, Ce Wang, Yilun Huang, Yushuo Chen, Xuchen Pan, Yaliang Li, and Bolin Ding. Grounded in reality: Learning and deploying proactive LLM from ofline logs. In Proceedings of the 43rd International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id= J4k8q8nku1. ICML 2026 Poster; arXiv:2510.25441.

Qianqian Xie, Weiguang Han, Xiao Zhang, Yanzhao Lai, Min Peng, Alejandro Lopez-Lira, and Jimin Huang. PIXIU: A comprehensive benchmark, instruction dataset and large language model for finance. In Advances in Neural Information Processing Systems, volume 36, pages 33469–33484. Curran Associates, Inc., 2023. doi: 10.52202/075280-1454. URL https://proceedings.neurips.cc/paper\_files/paper/ 2023/hash/6a386d703b50f1cf1f61ab02a15967bb-Abstract-Datasets\_and\_Benchmarks. html.

Qianqian Xie, Weiguang Han, Zhengyu Chen, Ruoyu Xiang, Xiao Zhang, Yueru He, Mengxi Xiao, Dong Li, Yongfu Dai, Duanyu Feng, Yijing Xu, Haoqiang Kang, Ziyan Kuang, Chenhan Yuan, Kailai Yang, Zheheng Luo, Tianlin Zhang, Zhiwei Liu, Guojun Xiong, Zhiyang Deng, Yuechen Jiang, Zhiyuan Yao, Haohang Li, Yangyang Yu, Gang Hu, Jiajia Huang, Xiao-Yang Liu, Alejandro Lopez-Lira, Benyou Wang, Yanzhao Lai, Hao Wang, Min Peng, Sophia Ananiadou, and Jimin Huang. FinBen: A holistic financial benchmark for large language models. In Advances in Neural Information Processing Systems, volume 37, pages 95716–95743. Curran Associates, Inc., 2024. doi: 10.52202/ 079017-3033. URL https://proceedings.neurips.cc/paper\_files/paper/2024/hash/ adb1d9fa8be4576d28703b396b82ba1b-Abstract-Datasets\_and\_Benchmarks\_Track.html.

Tong Zhang, Peixin Qin, Yang Deng, Chen Huang, Wenqiang Lei, Junhong Liu, Dingnan Jin, Hongru Liang, and Tat-Seng Chua. CLAMBER: A benchmark of identifying and clarifying ambiguous information needs in large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10746–10766. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.acl-long.578. URL https://aclanthology.org/2024.acl-long.578/.

Jiale Zhao, Ke Fang, and Lu Cheng. When and what to ask: AskBench and rubric-guided RLVR for LLM clarification. In Findings of the Association for Computational Linguistics: ACL 2026, pages 17120–17140. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026.findings-acl.845. URL https: //aclanthology.org/2026.findings-acl.845/.

Wenting Zhao, Ge Gao, Claire Cardie, and Alexander M. Rush. I could’ve asked that: Reformulating unanswerable questions. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 4207–4220. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024. emnlp-main.242. URL https://aclanthology.org/2024.emnlp-main.242/.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Sto ica. Judging LLM-as-a-judge with MT-bench and chatbot arena. In Advances in Neural Information Processing Systems, volume 36, pages 46595–46623. Curran Associates, Inc., 2023. doi: 10.52202/075280-2020. URL https://proceedings.neurips.cc/paper\_files/paper/2023/ hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets\_and\_Benchmarks.html.

Zhanke Zhou, Xiao Feng, Zhaocheng Zhu, Jiangchao Yao, Sanmi Koyejo, and Bo Han. From passive to active reasoning: Can large language models ask the right questions under incomplete information? In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pages 78714–78758. PMLR, 2025. URL https://proceedings.mlr.press/v267/ zhou25e.html.

Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng, and Tat-Seng Chua. TAT-QA: A question answering benchmark on a hybrid of tabular and textual content in finance. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3277–3287. Association for Computational Linguistics, 2021. doi: 10.18653/v1/2021.acl-long.254. URL https://aclanthology.org/2021.acl-long.254/.

Yakun Zhu, Zhongzhen Huang, Linjie Mu, Yutong Huang, Wei Nie, Jiaji Liu, Shaoting Zhang, Pengfei Liu, and Xiaofan Zhang. DiagnosisArena: Benchmarking diagnostic reasoning for large language models. In Findings of the Associationfor Computational Linguistics: ACL 2026, pages 3074–3098, San Diego, California, United States, July 2026. Association for Computational Linguistics. doi: 10.18653/v1/2026.findings-acl.151. URL https://aclanthology.org/2026.findings-acl.151/.

Table 2: Positioning of FinRiskAtlas relative to representative financial and information-acquisition benchmarks. Operation-aligned indicates that downstream units are defined by their role in a professional workflow rather than source format or task type. Decision contract indicates that the visible evidence regime, decision object, reviewer artifact, and scoring protocol are fixed at the family level. Evidence acquisition indicates evaluation of whether and what additional information should be requested. ✓ denotes present as a benchmark-level design unit, ✗ not documented as such, and ◦ partial.
<table><tr><td>Benchmark</td><td>Primary focus</td><td>Operation aligned</td><td>Decision contract</td><td>Evidence</td><td>Expert acquisition trajectories</td></tr><tr><td>FLUE Shah et al. (2022)</td><td>Financial language understanding</td><td>x</td><td>x</td><td>X</td><td>x</td></tr><tr><td>FinQA / ConvFinQA Chen et al. (2021, 2022)</td><td>Numerical reasoning over financial reports</td><td>x</td><td>x</td><td>X</td><td>x</td></tr><tr><td>BizBench Krumdick et al. (2024)</td><td>Business quantitative reasoning</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>PIXIU / FinBen Xie et al. (2023, 2024)</td><td>Broad financial capability coverage</td><td>x</td><td>0</td><td>X</td><td>x</td></tr><tr><td>CFinBench / FinEval Nie et al. (2025), Guo et al. (2025)</td><td>Chinese financial knowledge evaluation</td><td>x</td><td>o</td><td>x</td><td>x</td></tr><tr><td>CNFinBench Ding et al. (2026)</td><td>Financial capability, compliance, and</td><td>O</td><td>o</td><td>X</td><td>x</td></tr><tr><td>FinGuard Dou et al. (2026)</td><td>safety Regulation-derived compliance evaluation</td><td>O</td><td>o</td><td>x</td><td>x</td></tr><tr><td>FinED-Bench He et al. (2026)</td><td>Financial document error detection</td><td>o</td><td>x</td><td>X</td><td>x</td></tr><tr><td>Finance Agent Bigeard et al. (2025)</td><td>Tool-supported financial research</td><td>O</td><td>x</td><td>O</td><td>x</td></tr><tr><td>MediQ / QuestBench Li et al. (2024, 2025)</td><td>agents Clarification under incomplete</td><td>x</td><td>x</td><td>√</td><td>X</td></tr><tr><td>RealFin Dai et al. (2026)</td><td>information Unanswerable financial premises</td><td>x</td><td>x</td><td>O</td><td>x</td></tr><tr><td>Learn-to-Ask Wei et al. (2026)</td><td>Policy learning from expert dialogue</td><td>x</td><td>x</td><td>√</td><td>√</td></tr><tr><td>FinRiskAtlas</td><td>Operation-level financial risk and compliance review</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

## A. Benchmark Construction and Inventory

## A.1. Design Comparison with Existing Benchmarks

Table 2 compares FinRiskAtlas with representative financial and information-acquisition benchmarks along the design dimensions relevant to our evaluation framework. A property is marked as present only when it serves as a benchmark-level organizing principle rather than an isolated task capability. The comparison is intended to clarify diferences in evaluation design, rather than to rank existing benchmarks by overall quality.

This appendix describes the benchmark inventory, construction process, data sources, and quality-control procedures. The evaluation units and family contracts are defined in Section 3, while the FinRisk-Ask reconstruction protocol is described in Appendix C. Static instances and trajectory states represent diferent evaluation units and are therefore reported separately.

Table 3: Benchmark composition of FinRiskAtlas. Static instances and active-review states represent diferent evaluation units and are not combined.
<table><tr><td>Layer</td><td>Evaluation focus</td><td>Families / settings</td><td>Instances</td></tr><tr><td>Domain Knowledge</td><td>Financial concepts, regulations, obliga- tions, and risk mechanisms supporting professional review</td><td>42</td><td>4,679</td></tr><tr><td>Evidence-Grounded Processing</td><td>Transforming heterogeneous records into structured and decision-relevant ev- idence</td><td>5</td><td>1,502</td></tr><tr><td>Applied Review</td><td>Evidence-based case analysis, rule grounding, and professional review generation</td><td>6</td><td>3,561</td></tr><tr><td>Static benchmark FinRisk-Ask</td><td>Three-layer financial review evaluation Evidence-state control over recon- structed professional review states</td><td>53 1 setting</td><td>9,742 680 states (583 Ask, 97 Proceed)</td></tr></table>

## A.2. Capability-First Construction Pipeline

FinRiskAtlas follows a capability-first construction process. The central principle is that data collection procedures should instantiate predefined evaluation contracts rather than determine the benchmark organization. We first define the professional capability or review operation, specify its evaluation contract, identify suitable source materials, construct candidate instances, and then perform expert validation.

Three construction routes are used.

Direct normalization converts expert-authored questions and verified answers into standardized evaluation items while preserving their original capability targets and reference decisions.

Structured transformation converts financial and legal records into evidence-processing tasks such as extraction, matching, classification, and quantitative reasoning while preserving the relationship between source evidence and evaluation targets.

Source-conditioned construction derives case-level review tasks from complex business, transaction, or legal materials when the expected output is supported by model-visible evidence, documented professional decisions, or deterministic calculations.

The construction route determines how an item is obtained, but not how it is evaluated. After construction, every candidate item is assigned to a family only when it instantiates the same evaluation contract as other items in that family. This separation prevents source format, collection procedure, or annotation availability from becoming the implicit definition of a benchmark unit.

Table 3 summarizes the benchmark composition. The detailed family inventory is provided in Table 4. Family sizes are intentionally unequal because construction follows capability coverage rather than balanced sampling. Aggregation is therefore performed only over explicitly compatible family sets.

## A.3. Data Sources and Expert Involvement

FinRiskAtlas integrates multiple source categories covering diferent stages of professional financial review. Static benchmark construction involved six domain experts: four with experience in financial risk control and compliance review and two with experience in legal review of financial and commercial disputes. Experts participated in capability definition, family-boundary validation, candidate-item review, ambiguity analysis, and quality control.

Every retained instance underwent independent review by at least two experts with relevant financial or legal backgrounds before inclusion. Disagreements were resolved through adjudication, and items without stable expert consensus were removed rather than assigned majority labels. The released benchmark therefore

Table 4: Group-level composition of FinRiskAtlas. Each row reports the taxonomy group, coverage, number of families, and total number of instances. The complete per-family inventory, including sources and contracts, is released with the benchmark package.
<table><tr><td>Layer</td><td>Group</td><td>Coverage</td><td>Families</td><td>Instances</td></tr><tr><td>Domdn wddge</td><td>Financial risk</td><td>Risk management, markets, products, valuation, credit, operational, liq-</td><td>8</td><td>892</td></tr><tr><td rowspan="14"></td><td>Fraud risk</td><td>uidity, and fixed-income risk Card, insurance, telecom, contract, invoice, and AI-enabled fraud</td><td>6</td><td>624</td></tr><tr><td>Money-laundering risk</td><td>Anti-money-laundering obligations and typologies</td><td>2</td><td>244</td></tr><tr><td>Compliance risk</td><td>Regulatory obligations and compliance controls</td><td>4</td><td>442</td></tr><tr><td>Trade theory and policy</td><td>International trade theory and policy</td><td>4</td><td>457</td></tr><tr><td>FDI and multinationals</td><td>Foreign direct investment and multinational operations</td><td>2</td><td>208</td></tr><tr><td>International finance</td><td>Cross-border finance and settlement</td><td>4</td><td>499</td></tr><tr><td>International business</td><td>Business environment and operations</td><td>4</td><td>438</td></tr><tr><td>FinTech</td><td>Financial technology</td><td>1</td><td>103</td></tr><tr><td>AI security</td><td>AI-related security risk</td><td>1</td><td>106</td></tr><tr><td>Technical security</td><td>Security controls</td><td>1</td><td>103</td></tr><tr><td>Security engineering</td><td>Security engineering practice</td><td>1</td><td>104</td></tr><tr><td>E-commerce</td><td>E-commerce risk and operations</td><td>1</td><td>139</td></tr><tr><td>Payments</td><td>Payment systems and controls</td><td>1</td><td>109</td></tr><tr><td>Finance law</td><td>Finance-related legal knowledge</td><td>1</td><td>110</td></tr><tr><td>Law</td><td>General legal knowledge for review</td><td>1</td><td>101</td></tr><tr><td>Subtotal</td><td></td><td>42</td><td>4,679</td></tr><tr><td>Groded Case classification Eviddnnce-</td><td>Identifying case structure and category</td><td></td><td>1</td><td>445</td></tr><tr><td>Information extraction</td><td>Extracting requested fields from source records</td><td>1</td><td>249</td><td></td></tr><tr><td>Entity matching</td><td>Institution and person matching</td><td></td><td>2</td><td>750</td></tr><tr><td>Quantitative reasoning</td><td>Decision-relevant financial computation</td><td></td><td>1</td><td>58</td></tr><tr><td>Subtotal</td><td></td><td></td><td>5</td><td>1,502</td></tr><tr><td></td><td>Risk classification</td><td>Case-level risk categorization</td><td></td><td></td></tr><tr><td></td><td>Legal decision</td><td>Legal-outcome prediction and legal-judgment generation</td><td>1 2</td><td>366 1,445</td></tr><tr><td></td><td>Provision grounding</td><td>Applicable-provision selection</td><td>1</td><td>1,000</td></tr><tr><td></td><td>Review generation</td><td>Decision-view and disputed-issue generation</td><td>2</td><td>750</td></tr><tr><td>Apd ew Subtotal</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>6</td><td>3,561</td></tr><tr><td>Static benchmark total</td><td></td><td></td><td>53</td><td>9,742</td></tr><tr><td>FinRisk-Ask active review</td><td></td><td></td><td>1 setting</td><td>680 states</td></tr></table>

contains consensus references rather than unresolved annotations.

## A.4. Quality Control and Leakage Prevention

Quality control is conducted at both the family and instance levels.

At the family level, experts verify that all items within a family share the same target capability, model visible information regime, professional decision object, reviewer artifact, and scoring protocol. Candidate families are split when they contain distinct decision objects and merged when their contracts are equivalent.

At the instance level, we apply the following checks:

• Evidence support: reference outputs must be supported by supplied evidence, authoritative documents, deterministic transformations, or adjudicated expert decisions.

• Ambiguity control: items with unclear decision targets or unstable scoring criteria are revised or removed.

• Duplicate control: normalized inputs, source identifiers, and templates are examined to reduce accidental duplication.

Table 5: Data sources and expert roles in benchmark construction.
<table><tr><td>Source category</td><td>Capability coverage</td><td>Expert involvement</td></tr><tr><td>Professional examinations and textbooks</td><td>mechanisms</td><td>Financial concepts, regulations, and risk Reviewed capability alignment and ref- erence correctness</td></tr><tr><td>Regulatory documents and industry guidance</td><td>grounded reasoning</td><td>Compliance obligations and rule- Reviewed regulatory interpretation and applicability</td></tr><tr><td>Business, transaction, entity, and legal records</td><td>tion, quantitative reasoning, and applied support review</td><td>Evidence extraction, entity reconcilia- Reviewed decision targets and evidence</td></tr><tr><td>Completed risk-review trajectories</td><td>request evaluation</td><td>Ask-or-Proceed transitions and evidence- Verified reconstructed states, action map- pings, and evidence targets</td></tr></table>

• Leakage control: answer-bearing metadata, hidden targets, and unavailable future information are removed from model-visible contexts.

• Parser validation: released examples and adversarial formatting variants are used to validate deterministic output extraction.

• Provenance tracking: every released item is associated with source information and construction records. For publicly available materials, provenance information is retained and knowledge-oriented evaluation is distinguished from record-grounded review operations, where correctness depends on evidence available within the evaluation context rather than memorized facts. Family-level and instance-level review jointly enforce the five-part evaluation contract: the former defines the measurement unit, while the latter verifies that each retained item is a valid instantiation of that unit.

## B. Evaluation Protocol and Reproducibility

This appendix specifies the execution protocol, response processing, scoring procedures, semantic evaluation, and reproducibility practices of FinRiskAtlas. The benchmark follows a unified evaluation pipeline in which raw model responses, parsed predictions, and final evaluation outputs are retained for verification. FinRiskAtlas is evaluation-only: candidate models are not trained, adapted, or optimized on any benchmark component.

## B.1. Inference Configuration

All experiments use zero-shot direct-answer inference without in-context demonstrations or requested chain-of-thought. Models receive the task instruction and the model-visible information defined by the corresponding family contract.

All reported results correspond to archived evaluation runs. The run manifest records the exact model identifier, provider or checkpoint revision, decoding configuration, output length limit, evaluation timestamp, retry status, and prediction-file hash for every evaluated configuration. These records define the execution environment required to reproduce the reported results.

Each model–instance pair is evaluated once under the archived configuration. Empty, failed, or unparseable responses remain in the evaluation denominator. Therefore, reported values should be interpreted as point estimates of the evaluated configurations rather than estimates over repeated stochastic generations.

## B.2. Prompt and Output Processing

During evaluation, each family contract is instantiated through a fixed prompt template, output schema, parser, and scoring implementation. The output contract determines the expected response format and the corresponding parsing procedure before candidate evaluation.

Knowledge and classification tasks require labels, options, or option sets. Information-extraction tasks require structured fields. Entity-reconciliation tasks require identity decisions. Quantitative reasoning tasks

Table 6: Task-native evaluation contracts in FinRiskAtlas.
<table><tr><td>Task type</td><td>Output</td><td>Scoring rule</td><td>Metric</td></tr><tr><td>Knowledge and classification</td><td>Label, option, or option set</td><td>Correctness after family-specific extraction, set han- Accuracy-style score dling, and normalization</td><td></td></tr><tr><td>Information extraction</td><td>Requested fields</td><td>Normalized field-level comparison with reference fields, Field-level score preserving required field identity and order</td><td></td></tr><tr><td>Entity reconciliation</td><td>Entity identity</td><td>Exact correspondence with the reference identity deci- Accuracy sion</td><td></td></tr><tr><td>Quantitative reasoning</td><td>Numerical answer</td><td>Family-specific numerical normalization and correct- Accuracy-style score ness checking</td><td></td></tr><tr><td>Risk and legal decisions</td><td>Categorical decision</td><td>Exact decision matching after label normalization</td><td>Accuracy</td></tr><tr><td></td><td>Open-ended review generation Professional analysis</td><td>Fixed semantic rubric evaluating correctness, coverage, Rubric score and evidence use</td><td></td></tr><tr><td>FinRisk-Ask action</td><td>Ask or Proceed</td><td>Agreement with the recorded reviewer transition Three-level semantic alignment with later-observed, ERA, CRA, DEH, REH</td><td>RAA, AskR, ProceedR, BAcc</td></tr><tr><td>FinRisk-Ask evidence request</td><td>Focused evidence request</td><td>expert-verified evidence targets</td><td></td></tr></table>

require normalized numerical answers. Open-ended Applied Review tasks require professional analyses evaluated by semantic rubrics. FinRisk-Ask requires either an Ask decision with one evidence request or a Proceed decision.

The evaluation pipeline preserves:

1. raw model responses before parsing;

2. parsed outputs produced by deterministic parsers;

3. final metric values produced by family-specific scoring functions.

Answer normalization is restricted to formatting variation and does not modify the underlying prediction content. Parsed outputs are retained so that parsing success and valid-only analyses can be recovered without changing the primary evaluation protocol.

## B.3. Task-Native Scoring Protocol

For a model configuration m and a task family f containing $N _ { f }$ instances, let $\hat { y } _ { m , i }$ denote the raw response to instance i, let $p _ { f }$ denote the fixed family parser, and let $g _ { f } \in [ \dot { 0 } , 1 ]$ denote the family-specific scoring function. The configuration–family score is

$$
S _ { m , f } = { \frac { 1 0 0 } { N _ { f } } } \sum _ { i = 1 } ^ { N _ { f } } g _ { f } \big ( p _ { f } ( \hat { y } _ { m , i } ) , y _ { i } \big ) .\tag{3}
$$

Malformed or unparseable outputs are mapped by $p _ { f }$ to a designated invalid value and receive zero credit under $g _ { f }$ . For a downstream operation o, we write $S _ { m , o }$ for the corresponding operation-family score.

Table 7 instantiates the evaluation contract $\Gamma _ { f } = ( c _ { f } , \mathscr { T } _ { f } , d _ { f } , \mathscr { y } _ { f } , s _ { f } )$ for each of the eleven fixed-evidence downstream operations. The operation name specifies $c _ { f } ,$ while the remaining columns summarize $\textstyle { \mathcal { T } } _ { f } , d _ { f } .$ $\mathcal { V } _ { f }$ , and $s _ { f }$

Let $\dot { \mathcal { F } _ { K } }$ denote the set of 42 Domain Knowledge families. The Domain Knowledge macro score of configuration m is

$$
K _ { m } = \frac { 1 } { | \mathcal { F } _ { K } | } \sum _ { f \in \mathcal { F } _ { K } } S _ { m , f } , \qquad | \mathcal { F } _ { K } | = 4 2 .\tag{4}
$$

Each knowledge family therefore contributes equally regardless of its number of instances. No corresponding cross-operation macro is used as a primary result because downstream operations produce diferent reviewer artifacts and use diferent native scoring protocols.

Table 7: Evaluation contracts for the eleven fixed-evidence downstream operations. The operation name specifies the target capability $c _ { f } ;$ the remaining columns summarize the model-visible information regime $\mathcal { T } _ { f } ,$ decision object $d _ { f } ,$ reviewer-artifact schema $\mathcal { V } _ { f } .$ , and scoring protocol $s _ { f }$
<table><tr><td>Operation</td><td>Model-visible information</td><td>Decision object</td><td>Reviewer artifact</td><td>Scoring</td></tr><tr><td colspan="5">Evidence-Grounded Processing</td></tr><tr><td>Case classification</td><td>Full case record and a fixed candidate category set</td><td>Case category</td><td>Procedure-routing label</td><td>Accuracy</td></tr><tr><td>Information tion</td><td>extrac- Source record and an ordered list of requested fields</td><td>Values of the requested fields Structured evidence-field</td><td>sequence</td><td>Normalized field-level score</td></tr><tr><td>Institution matching</td><td>Two institution descriptions, possibly from different records or languages</td><td>Whether the descriptions denote the same institution</td><td>Institution-identity decision</td><td>Accuracy</td></tr><tr><td>Person matching</td><td>Two person records with partially overlapping</td><td>Whether the records denote the same natural person</td><td>Person-identity decision</td><td>Accuracy</td></tr><tr><td>ing</td><td>attributes Quantitative reason- Case figures together with the applicable definition or formula</td><td>Requested numerical quantity</td><td>One normalized quantity</td><td>Numerical correctness</td></tr><tr><td colspan="5">Applied Review Risk classification</td></tr><tr><td>Applicable-provision</td><td>Case-level evidence and the applicable risk-category set</td><td>Case-level risk category</td><td>Risk grade</td><td>Accuracy</td></tr><tr><td>selection Legal-outcome predic- Case facts and procedural</td><td>Case facts and a candidate provision set</td><td>Governing provisions</td><td>Provision set cited as the decision basis Disposition label</td><td>Set-based accuracy-style score Accuracy</td></tr><tr><td>tion</td><td>record, with the later outcome withheld</td><td>Case disposition</td><td></td><td></td></tr><tr><td>ation</td><td>Legal-judgment gener- Case facts, evidence, and procedural record Disputed-issue genera- Case record containing</td><td>a disposition</td><td>Reasoned analysis supporting Professional legal analysis</td><td>Fixed semantic rubric</td></tr><tr><td>tion</td><td>overlapping legal and contractual relationships</td><td>Set of contested issues</td><td>Structured issue list</td><td>Fixed semantic rubric</td></tr><tr><td>tion</td><td>Decision-view genera- Case record and the review question presented to the reviewer</td><td>Recommended review position</td><td></td><td>Professional review opinionFixed semantic rubric</td></tr></table>

## B.4. Semantic Evaluation Protocol

Three open-ended Applied Review operations use semantic evaluation: legal-judgment generation, decisionview generation, and disputed-issue generation. Candidate responses are evaluated by a fixed semantic evaluator, DeepSeek-V4-Flash, following established LLM-based, rubric-guided evaluation practice for open ended generation Liu et al. (2023), Zheng et al. (2023).

The evaluator receives:

• the original task input;

• reference information;

• the candidate response;

• the task-specific evaluation rubric.

Candidate-model identity is not provided as an evaluator input. Evaluator configuration, prompts, and parsing procedures are fixed independently of candidate generation and applied uniformly to all responses.

The evaluator configuration’s own candidate outputs are retained for transparency but excluded from best-value annotations, operation-level comparisons, and analyses involving judge-scored tasks.

FinRisk-Ask request alignment uses the same fixed evaluator with a three-level rubric. For each retained Ask state, the evaluator receives the generated request and the expert-verified future evidence target set. A score of 1 indicates direct alignment, 0.5 indicates a relevant but incomplete request, and 0 indicates an unrelated request. The evaluator does not determine the Ask-or-Proceed reference action, which is derived

from the recorded workflow transition.

The semantic evaluator is treated as a fixed measurement instrument rather than an oracle of professional correctness. Excluding evaluator self-comparison and releasing evaluator artifacts improve reproducibility but do not eliminate evaluator-specific efects Zheng et al. (2023), Panickssery et al. (2024). Semantic scores should therefore be interpreted as rubric measurements under the released evaluator configuration.

## B.5. Aggregation and Statistical Interpretation

Reported results are descriptive comparisons among archived configurations. Several configurations belong to related model families and should not be interpreted as independent samples from a model population. FinRisk-Ask states originate from 104 trajectories, and multiple states from the same trajectory may share case-level dependencies.

Interpretation of reported comparisons follows several principles. First, analyses involving operation maxima refer to observed point-estimate maxima rather than universal model rankings. Second, regret analyses consider near-tied Domain Knowledge configurations to reduce sensitivity to arbitrary ordering. Third, Ask-or-Proceed comparisons are evaluated against one shared recorded workflow boundary rather than treated as independent hypothesis tests. Fourth, FinRisk-Ask metrics are state-weighted, meaning trajectories contributing more retained states receive greater influence.

Percentile profiles are used only for comparing relative rankings across heterogeneous operations. They do not imply that extraction, classification, numerical reasoning, and rubric-based generation share a common performance scale.

## B.6. Reproducibility Artifacts

The released evaluation package includes the benchmark manifest, prompts, inference parameters, raw and parsed predictions, evaluator configurations, scoring implementations, and analysis scripts. These artifacts allow new model configurations to be evaluated under the same contracts and compared on the same operation and evidence-state dimensions.

## C. FinRisk-Ask Protocol

FinRisk-Ask is an ofline trajectory-replay evaluation of evidence-state control in financial review. Unlike interactive information-seeking benchmarks and policy-learning approaches Li et al. (2024, 2025), Zhou et al. (2025), Zhao et al. (2026), Wei et al. (2026), it does not optimize a dialogue policy or evaluate long-horizon interaction. Instead, it measures whether a model can make an appropriate evidence-state transition under a fixed historical decision boundary and whether a generated request addresses a trajectory-supported evidence need.

## C.1. Trajectory Reconstruction and Release Filtering

FinRisk-Ask is constructed from completed financial-risk review trajectories collected from an internal enterprise risk-control workflow. The trajectories originate from real review processes and are released only through de-identified reconstructed states.

For each trajectory, reconstruction proceeds chronologically to identify decision points where the recorded workflow either initiates additional evidence acquisition or advances using the currently available record. A candidate state is retained when three conditions are satisfied: (i) the pre-action context can be reconstructed without later observations, (ii) the immediately following reviewer action maps unambiguously to Ask or Proceed, and (iii) no post-action information is included in the model-visible context.

Ask-labeled states require one additional condition. At least one later-observed evidence item must be available from the completed trajectory and verified by experts as unavailable at the decision point, relevant to an unresolved review need, and appropriate as an evidence-acquisition target. During reconstruction,

10 Ask-labeled states were removed because no admissible future evidence target could be identified and verified. The released benchmark therefore contains 680 states from 104 trajectories, including 583 recorded Ask states and 97 recorded Proceed states.

The filtering rule is defined independently of candidate-model outputs. Every released state supports action evaluation, and every released Ask state additionally supports request evaluation, providing a shared denominator for AskR, ERA, and CRA. Multiple states may originate from the same trajectory, but each state preserves the evidence boundary that existed at its own historical decision point.

For each released decision point $t , C _ { t }$ denotes the complete case evidence and review history available immediately before the recorded action $a _ { t } ^ { * }$ . Subsequent evidence, later reviewer actions, and downstream outcomes are excluded from $C _ { t }$ and are accessible only to the evaluation pipeline.

## C.2. Reference Action and Evidence Target Construction

The Ask-or-Proceed reference is derived from the recorded reviewer action immediately following each reconstructed state. It is not inferred from whether additional evidence appears later in the trajectory. A state is labeled Ask when the recorded action initiates a new evidence-acquisition request, and Proceed when the workflow advances using the currently available record without initiating a request at that point. Evidence acquisition that occurs at later decision points does not change the current label.

The reference action is defined as:

$$
a _ { t } ^ { * } = \left\{ \begin{array} { l l } { \mathrm { A s k , } } & { \mathrm { i f ~ t h e ~ r e c o r d e d ~ a c t i o n ~ a t ~ } t \mathrm { ~ i n i t i a t e s ~ a d d i t i o n a l ~ e v i d e n c e ~ a c q u i s i t i o n , } } \\ { \mathrm { P r o c e e d , } } & { \mathrm { i f ~ t h e ~ w o r k f l o w ~ a d v a n c e s ~ a t ~ } t \mathrm { ~ w i t h o u t ~ i n i t i a t i n g ~ a ~ n e w ~ r e q u e s t . } } \end{array} \right.\tag{5}
$$

The released labels contain 583 Ask states and 97 Proceed states. Action mappings were reviewed by domain experts, and disagreements were resolved through adjudication. The reference represents an observed professional transition under one workflow rather than a claim that it is the only defensible review policy.

For retained Ask states, FinRisk-Ask constructs an expert-verified set of later-observed evidence needs:

$$
\mathcal { E } _ { t } ^ { + } = \{ e _ { t , 1 } , e _ { t , 2 } , \ldots , e _ { t , n _ { t } } \} , \qquad n _ { t } \geq 1 .\tag{6}
$$

Historical reviewer requests are not used as exact textual references. Instead, generated requests are evaluated against the underlying evidence need, allowing semantically equivalent requests to receive credit.

Future evidence targets are constructed by collecting later-observed materials after the decision point, normalizing evidence descriptions, and verifying the resulting targets with domain experts. A target is retained only if experts confirm that the evidence is temporally unavailable at state t, addresses an unresolved review need, and corresponds to a meaningful evidence-acquisition objective.

Candidate models never observe $\mathcal { E } _ { t } ^ { + }$ or any other future trajectory information during inference.

## C.3. Expert Validation and Quality Control

FinRisk-Ask applies expert validation to three aspects of trajectory reconstruction: state validity, action mapping, and evidence-target admissibility. Experts verify that reconstructed states preserve the original information boundary, that Ask-or-Proceed labels correspond to the recorded workflow transition, and that retained evidence targets represent unresolved decision-relevant needs at the replayed state.

The quality-control process separates three requirements:

• Temporal validity: target evidence must appear only after the evaluated decision point and must not already exist in the visible context.

• Decision relevance: the evidence must address a need that afects the professional decision at that state.

• Semantic validity: diferent descriptions referring to the same evidence need should be recognized as equivalent.

Targets failing any of these requirements are removed or revised before release.

## C.4. Evaluation Protocol and Parsing

Given the pre-action context $C _ { t : }$ , the required structured model output is

$$
\hat { y } _ { t } = \left\{ \begin{array} { l l } { \mathrm { A s k } ( \hat { q } _ { t } ) , } & { \mathrm { r e q u e s t ~ o n e ~ f o c u s e d ~ i t e m ~ o f ~ a d d i t i o n a l ~ e v i d e n c e } , } \\ { \mathrm { P r o c e e d } , } & { \mathrm { c o n t i n u e ~ w i t h ~ t h e ~ c u r r e n t l y ~ a v a i l a b l e ~ e v i d e n c e } . } \end{array} \right.\tag{7}
$$

The action parser maps $\hat { y } _ { t }$ to $\hat { a } _ { t } \in \{ \mathrm { A s k } _ { * }$ , Proceed, Invalid}. When $\hat { a } _ { t } = \operatorname { A s k } .$ , the request parser additionally extracts $\hat { q } _ { t }$ and checks whether it satisfies the one-request output contract. A parsed Ask action with an empty, malformed, rejected, or multi-item request remains an Ask prediction for AskR but receives zero request-alignment credit. A structurally valid but semantically unrelated request likewise receives zero credit under the alignment rubric. An Invalid action receives zero action-agreement credit, while a Proceed prediction on a recorded Ask state receives zero ERA credit because no request is produced.

The parsing rules and normalization procedures are fixed before model evaluation and released with the benchmark artifacts.

## C.5. Metrics

For a fixed model configuration, we suppress the configuration index. Let $\mathcal { T } = \{ 1 , \ldots , N \}$ denote the released state set, where $N = 6 8 0$ , and let $\hat { a } _ { t }$ and $a _ { t } ^ { * }$ denote the parsed model action and recorded reviewer action at state t. Recorded-Action Agreement is

$$
\mathrm { R A A } = \frac { 1 0 0 } { N } \sum _ { t \in \mathcal { T } } \mathbb { I } [ \hat { a } _ { t } = a _ { t } ^ { * } ] .\tag{8}
$$

Define the two reference-state subsets

$$
\mathcal { T } _ { \mathrm { A s k } } = \{ t \in \mathcal { T } : a _ { t } ^ { * } = \mathrm { A s k } \} , \qquad \mathcal { T } _ { \mathrm { P r o c e e d } } = \{ t \in \mathcal { T } : a _ { t } ^ { * } = \mathrm { P r o c e e d } \} ,\tag{9}
$$

where $| \mathcal { T } _ { \mathrm { A s k } } | = 5 8 3$ and $| \mathcal { T } _ { \mathrm { P r o c e e d } } | = 9 7$ . The branch-specific recalls are

$$
\mathrm { A s k R } = \frac { 1 0 0 } { \left| \mathcal { T } _ { \mathrm { A s k } } \right| } \sum _ { t \in \mathcal { T } _ { \mathrm { A s k } } } \mathbb { I } [ \hat { a } _ { t } = \mathrm { A s k } ] ,\tag{10}
$$

$$
\mathrm { P r o c e e d R } = \frac { 1 0 0 } { \left| \mathcal { T } _ { \mathrm { P r o c e e d } } \right| } \sum _ { t \in \mathcal { T } _ { \mathrm { P r o c e e d } } } \mathbb { I } [ \hat { a } _ { t } = \mathrm { P r o c e e d } ] .\tag{11}
$$

Balanced Recorded-Action Agreement gives equal weight to the two reference branches:

$$
\mathrm { B A c c } = \frac { \mathrm { A s k R } + \mathrm { P r o c e e d R } } { 2 } .\tag{12}
$$

For each $t \in { \mathcal { T } } _ { \mathrm { A s k } }$ define the request-alignment credit

$$
\begin{array} { r } { z _ { t } = \left\{ \begin{array} { l l } { \underset { e \in \mathcal { E } _ { t } ^ { + } } { \operatorname* { m a x } } \ell ( \hat { q } _ { t } , e ) , } & { \mathrm { i f } \ \hat { a } _ { t } = \mathrm { A s k } \ \mathrm { a n d } \ \hat { q } _ { t } \ \mathrm { i s } \ \mathrm { v a l i d } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. } \end{array}\tag{13}
$$

where

$$
\ell ( \hat { q } _ { t } , e ) = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { d i r e c t ~ a l i g n m e n t ~ w i t h ~ t h e ~ e v i d e n c e ~ n e e d , } } \\ { 0 . 5 , } & { \mathrm { r e l e v a n t ~ b u t ~ i n c o m p l e t e ~ a l i g n m e n t , } } \\ { 0 , } & { \mathrm { u n r e l a t e d ~ r e q u e s t . } } \end{array} \right.\tag{14}
$$

Evidence-Request Alignment measures end-to-end request alignment over all recorded Ask states:

$$
\mathrm { E R A } = \frac { 1 0 0 } { \left| \mathcal { T } _ { \mathrm { A s k } } \right| } \sum _ { t \in \mathcal { T } _ { \mathrm { A s k } } } z _ { t } .\tag{15}
$$

Let

$$
\mathcal { A } = \left\{ t \in \mathcal { T } _ { \mathrm { A s k } } : \hat { a } _ { t } = \mathrm { A s k } \right\}\tag{16}
$$

denote the recorded Ask states on which the model enters the Ask branch. For $| { \mathcal { A } } | > 0 .$ , Conditional Request Alignment is

$$
\mathrm { C R A } = \frac { 1 0 0 } { \left| \boldsymbol { A } \right| } \sum _ { t \in \mathcal { A } } z _ { t } .\tag{17}
$$

When $| { \mathcal { A } } | = 0 $ , CRA is undefined and is reported as $^ { 6 6 } - \stackrel { \triangledown } { \boldsymbol { \cdot } } \boldsymbol { \cdot }$ . For $| { \mathcal { A } } | > 0$ , the shared recorded-Ask denominator yields the exact identity

$$
\mathrm { E R A } = \frac { \mathrm { A s k R } \times \mathrm { C R A } } { 1 0 0 } .\tag{18}
$$

Direct Evidence Hit and Relevant Evidence Hit are

$$
\mathrm { D E H } = \frac { 1 0 0 } { | \mathcal { T } _ { \mathrm { A s k } } | } \sum _ { t \in \mathcal { T } _ { \mathrm { A s k } } } \mathbb { I } [ z _ { t } = 1 ] ,\tag{19}
$$

$$
\mathrm { R E H } = \frac { 1 0 0 } { \left| \mathcal { T } _ { \mathrm { A s k } } \right| } \sum _ { t \in \mathcal { T } _ { \mathrm { A s k } } } \mathbb { I } [ z _ { t } \geq 0 . 5 ] .\tag{20}
$$

Because $z _ { t } \in \{ 0 , 0 . 5 , 1 \}$ under the three-level rubric,

$$
\mathrm { E R A } = { \frac { \mathrm { D E H } + \mathrm { R E H } } { 2 } } .\tag{21}
$$

## C.6. Interpretation Boundary

FinRisk-Ask evaluates recorded professional transitions and trajectory-supported evidence needs under an ofline replay setting. Later observations establish target provenance, but the evaluation does not execute generated requests or estimate their causal value after acquisition.

The target set represents evidence needs recoverable from completed trajectories and does not enumerate every professionally reasonable acquisition strategy. Similarly, the recorded Ask-or-Proceed action reflects one operational workflow and should not be interpreted as the unique optimal review policy.

Therefore, BAcc measures agreement with recorded workflow behavior, while ERA and CRA measure alignment with the released evidence-target set. These metrics characterize evidence-state control under the benchmark protocol rather than provide a complete measure of professional review quality.

## D. Supplementary Capability Analyses

This appendix provides additional analyses supporting the main empirical claims in Section 4: operationspecific model profiles, the limitations of knowledge-based routing, and the separation between recorded action agreement and evidence-request targeting.

## D.1. Robustness of Operation-Specific Capability Profiles

The main experiments show that downstream review operations induce diferent model rankings. We further examine whether this observation is driven only by large diferences in broad Domain Knowledge performance or whether models with similar knowledge scores can still exhibit diferent operational profiles.

For each configuration pair $( m , n )$ , we define the absolute diference in Domain Knowledge macro performance as

$$
D _ { m n } ^ { K } = | K _ { m } - K _ { n } | .\tag{22}
$$

Let $M = | { \mathcal { M } } _ { \mathrm { c m p } } | = 3 2$ . For each downstream operation $^ { o , }$ let $R _ { m , o }$ denote the descending average rank of configuration m within $M _ { \mathrm { c m p } } ,$ with rank 1 assigned to the highest score and tied configurations receiving average ranks. We map this rank to a within-operation percentile:

$$
P _ { m , o } = 1 0 0 \frac { M - R _ { m , o } } { M - 1 } .\tag{23}
$$

The downstream profile gap between configurations m and n is

$$
G _ { m n } = \frac { 1 } { | \mathcal { O } | } \sum _ { o \in \mathcal { O } } | P _ { m , o } - P _ { n , o } | ,\tag{24}
$$

where O denotes the eleven downstream operations. Percentile normalization is used only to compare relative profiles across operations with heterogeneous native scoring scales.

For the spectral analysis, let $\mathbf { C } \in \mathbb { R } ^ { 1 1 \times 1 1 }$ denote the Spearman rank-correlation matrix of the downstream operation scores, with eigenvalues $\lambda _ { 1 } \geq \lambda _ { 2 } \geq \cdot \cdot \cdot \geq \lambda _ { 1 1 } \geq 0$ . The contribution of component j is measured by $\lambda _ { j } / \sum _ { \ell = 1 } ^ { 1 1 } \lambda _ { \ell }$

Among the 30 configuration pairs with $D _ { m n } ^ { K } \leq 0 . 5$ , the median downstream profile gap is 25.5 percentile points, demonstrating that similar broad knowledge performance does not imply similar operational behavior. Across all 496 pairs, the median profile gap is 34.2 percentile points. The eigenspectrum further shows that operation rankings share common structure but are not reducible to a single latent ordering: the leading eigenvalue accounts for 49.9% of the total eigenvalue mass, while the first five eigenvalues account for 85.0% cumulatively.

We additionally examine whether Domain Knowledge ranking can recover operation-specific leaders. Within their respective operation-eligible configuration sets, the observed leaders occupy Domain Knowledge ranks from 1 to 23. A shortlist of size one recovers 2 of the 11 operation maxima, while a shortlist of size ten recovers 8 of 11. Full recovery requires a shortlist of size 23. These results show that broad knowledge ranking provides useful but incomplete information for operation-specific model selection.

## D.2. Relationship Between Static Capability and Evidence-State Control

FinRisk-Ask evaluates evidence-state control separately from fixed-evidence operation execution. We further analyze whether static downstream capabilities are strongly associated with active-review behavior.

Figure 6(a) reports Spearman associations between each downstream static operation and three FinRisk-Ask metrics: Balanced Recorded-Action Agreement (BAcc), Evidence-Request Alignment (ERA), and Con ditional Request Alignment (CRA). These correlations are descriptive associations within the evaluated configuration pool and do not imply causal relationships.

Several applied-review operations exhibit stronger association with request targeting than with recordedaction agreement. For example, legal-judgment generation shows stronger correlation with CRA $( \rho = 0 . 7 8 )$ than with BAcc $( \rho = 0 . 1 4 )$ , while decision-view generation shows stronger correlation with CRA $( \rho = 0 . 7 9 )$ than with BAcc $( \rho = 0 . 0 7 )$ . These observations suggest that generating useful evidence requests relies on capabilities that are not fully captured by matching the recorded Ask-or-Proceed boundary.

a  
![](images/dd5938e60475c3984a1fe6d833a0d963f4e4bdd1e2d32f0ef2029d94808e6997.jpg)

b  
![](images/18e7cc7ba1abb68ed7441a89b8e7dbb2ac2693138adcad93aaa45a3a946c6685.jpg)

c  
![](images/a118abe67f682a35e5d901abf48c6215d169d78b4768fcb912e6361a114d81a0.jpg)  
d

![](images/160abd6aee23dcd309f614df9bda0734307763ef9395b12f4b83d1fafc0e19cd.jpg)  
Figure 5: Additional analyses of operation-specific capability profiles. (a) Relationship between Domain Knowledge diference and downstream profile gap over 496 configuration pairs. Knowledge-near pairs $( | \Delta K | \le 0 . 5 )$ still exhibit substantial downstream divergence. (b) Eigenspectrum of the eleven-operation Spearman correlation matrix, showing shared but non-identical ranking structure across operations. (c) Number of observed operation leaders recovered by Domain Knowledge-based shortlists of diferent sizes. (d) Domain Knowledge rank of each configuration attaining an observed operation maximum. All maxima correspond to point estimates from the evaluated configuration pool.

Figure 6(b–d) further examines action-level behavior. Across the 32 non-evaluator configurations, Ask recall is generally higher than Proceed recall, indicating an asymmetric tendency toward reproducing evidence requests rather than recorded advancement decisions. Figure 6(c) shows that overall RAA can difer from balanced action agreement because the two reference classes are imbalanced. Figure 6(d) illustrates that configurations can shift substantially in ranking depending on whether they are evaluated by action agreement or end-to-end request alignment.

Together, these analyses reinforce that FinRisk-Ask measures multiple aspects of evidence-state control: reproducing a recorded transition, recognizing when additional evidence is required, and requesting evidence aligned with the unresolved decision need.

a  
![](images/174799185e78dc10d8e0030bfe36be4f83925e185be9d882c6ad9ae88a07f166.jpg)  
b

c  
![](images/b197bb29c14fe0a431c643fb1ee507e7cbd91b0938a25039b7aa57228937d996.jpg)  
d

![](images/d626ec08134e86a0ed9b8c84bdf7b611dc41f79175a6fe68be226d5aace23029.jpg)

![](images/d0bd6ce1a7d8746bf092d9c24beb6a1b72c1f84d36e9744b8ac654cd041c4651.jpg)  
Figure 6: Additional analyses of static capability and evidence-state behavior. (a) Spearman associations between downstream operation scores and FinRisk-Ask metrics. (b) Ask recall and Proceed recall across non-evaluator configurations, showing asymmetric reproduction of recorded transitions. (c) Relationship between Recorded-Action Agreement (RAA) and Balanced Recorded-Action Agreement (BAcc). (d) Rank displacement between BAcc and Evidence-Request Alignment (ERA) for representative configurations.

## E. Complete Model Results

This appendix reports the complete configuration-level results underlying the analyses in Section 4. The tables provide the full score matrices for all benchmark components under the same evaluation protocols used in the main experiments. For semantically evaluated tasks, the evaluator configuration’s own score is retained for transparency but excluded from best-value annotations and comparative analyses restricted to $\mathcal { M } _ { \mathrm { c m p } }$

## E.1. Complete Domain Knowledge Results

The Domain Knowledge score is the unweighted macro-average over 42 knowledge task families. Each family contributes equally, preventing taxonomy groups with larger numbers of instances from dominating the overall knowledge score.

Table 8: Complete Domain Knowledge macro results over all 33 archived configurations (%).
<table><tr><td>Model</td><td></td><td></td><td>Risk &amp; Compliance Trade FinTech &amp; Security Commerce &amp; Payment Finance &amp; Law</td><td></td><td></td><td>Overall</td></tr><tr><td>Dianjin-R1-7B</td><td>64.00</td><td>63.43</td><td>70.25</td><td>72.50</td><td>68.00</td><td>65.00</td></tr><tr><td>FinR1-7B</td><td>66.35</td><td>68.00</td><td>78.25</td><td>75.50</td><td>76.00</td><td>68.93</td></tr><tr><td>Ling-2.6-Flash</td><td>71.80</td><td>74.71</td><td>80.00</td><td>79.00</td><td>77.50</td><td>74.17</td></tr><tr><td>Ling-2.6-1T</td><td>75.22</td><td>78.96</td><td>83.22</td><td>78.82</td><td>80.49</td><td>77.65</td></tr><tr><td>Seed-2.0-Mini</td><td>73.22</td><td>76.64</td><td>80.43</td><td>78.32</td><td>80.04</td><td>75.62</td></tr><tr><td>Seed-2.0-Lite</td><td>73.70</td><td>80.36</td><td>78.78</td><td>81.51</td><td>78.55</td><td>77.01</td></tr><tr><td>Nemotron-3-Nano</td><td>49.06</td><td>49.30</td><td>57.49</td><td>57.46</td><td>49.92</td><td>50.38</td></tr><tr><td>Nemotron-3-Super</td><td>64.74</td><td>68.49</td><td>73.37</td><td>70.36</td><td>72.16</td><td>67.43</td></tr><tr><td>Nemotron-3-Ultra</td><td>71.81</td><td>76.89</td><td>78.06</td><td>79.73</td><td>81.85</td><td>74.96</td></tr><tr><td>Qwen2.5-7B</td><td>62.23</td><td>61.79</td><td>67.11</td><td>71.85</td><td>70.94</td><td>63.42</td></tr><tr><td>Qwen3-1.7B</td><td>40.13</td><td>37.87</td><td>43.78</td><td>44.49</td><td>46.05</td><td>40.21</td></tr><tr><td>Qwen3-4B</td><td>63.17</td><td>60.38</td><td>68.45</td><td>69.46</td><td>69.10</td><td>63.32</td></tr><tr><td>Qwen3-8B</td><td>67.12</td><td>65.68</td><td>71.24</td><td>74.42</td><td>73.06</td><td>67.66</td></tr><tr><td>Qwen3-14B</td><td>67.92</td><td>70.93</td><td>76.49</td><td>73.66</td><td>77.65</td><td>70.48</td></tr><tr><td>Qwen3-32B Qwen3-Next-80B</td><td>70.90</td><td>74.46</td><td>79.73</td><td>79.20</td><td>80.66</td><td>73.79</td></tr><tr><td>Qwen3-235B-A22B</td><td>74.20</td><td>78.97 79.02</td><td>82.52</td><td>79.03</td><td>82.08</td><td>77.19</td></tr><tr><td>Qwen3.7-Max</td><td>73.25 77.15</td><td>82.49</td><td>83.02</td><td>82.08</td><td>81.26 83.65</td><td>76.90 80.41</td></tr><tr><td>DeepSeek-V3-0324</td><td></td><td>70.44 75.65</td><td>85.04 78.33</td><td>86.01</td><td></td><td>73.57</td></tr><tr><td>DeepSeek-V4-Flash</td><td>71.33</td><td>75.49</td><td>79.98</td><td>74.99 76.30</td><td>79.48 77.65</td><td>74.08</td></tr><tr><td>DeepSeek-V4-Pro</td><td>72.11</td><td>76.87</td><td>82.73</td><td>81.55</td><td>79.88</td><td>75.53</td></tr><tr><td>Kimi-K2.5</td><td>75.23</td><td>81.95</td><td>82.15</td><td>80.25</td><td>79.21</td><td>78.56</td></tr><tr><td>Kimi-K3</td><td>78.11</td><td>83.05</td><td>84.07</td><td>82.80</td><td>80.95</td><td>80.68</td></tr><tr><td>GLM-5</td><td>73.53</td><td>78.14</td><td></td><td></td><td></td><td></td></tr><tr><td>GLM-5.2</td><td>74.65</td><td>79.18</td><td>80.27</td><td>79.93</td><td>78.81</td><td>76.26</td></tr><tr><td>GPT-5.4-Mini</td><td>74.38</td><td>79.48</td><td>80.33</td><td>79.43</td><td>79.02</td><td>77.13</td></tr><tr><td>GPT-5.4</td><td>74.46</td><td>81.17</td><td>80.59 84.69</td><td>84.34 84.34</td><td>80.16 85.34</td><td>77.42 78.66</td></tr><tr><td>GPT-5.6-Luna</td><td>72.49</td><td>77.61</td><td>77.59</td><td>78.81</td><td>78.53</td><td>75.27</td></tr><tr><td>GPT-5.6-Terra</td><td>74.87</td><td>81.58</td><td>81.19</td><td>82.73</td><td>80.24</td><td>78.34</td></tr><tr><td>GPT-5.6-Sol</td><td>76.30</td><td>83.00</td><td>80.23</td><td>82.42</td><td>76.71</td><td>79.21</td></tr><tr><td>Gemini-2.5-Flash</td><td>70.27</td><td>77.68</td><td>80.56</td><td>80.51</td><td>77.98</td><td>74.57</td></tr><tr><td>Gemini-3-Flash-preview</td><td>77.24</td><td>83.80</td><td>84.33</td><td>86.41</td><td>80.53</td><td>80.69</td></tr><tr><td>Claude-Opus-4.7</td><td>76.55</td><td>82.15</td><td>86.92</td><td>86.81</td><td>82.75</td><td>80.19</td></tr></table>

## E.2. Complete Fixed-Evidence Operation Results

Table 9 reports the complete results for Evidence-Grounded Processing and structured-output Applied Review operations. Columns correspond to the downstream operations defined in Table 1. Each operation retains its native scoring protocol and is not combined into a single downstream aggregate score.

## E.3. Complete Open-Ended Applied Review Results

Table 10 reports the complete results for open-ended Applied Review operations evaluated with the fixed semantic evaluator described in Appendix B. The evaluator configuration’s self-score is retained for transparency and marked with †, but excluded from comparative analyses involving judge-scored operations.

## E.4. Complete FinRisk-Ask Results

Table 11 reports complete FinRisk-Ask results over all evaluated configurations. The metrics follow the defini tions in Appendix C: RAA measures agreement with recorded actions, BAcc balances Ask and Proceed recall, ERA measures end-to-end evidence-request alignment, and CRA measures request alignment conditional on entering the Ask branch. The request-alignment evaluator configuration is marked with † and excluded from evaluator-dependent comparative analyses.

Table 9: Complete Evidence-Grounded Processing and structured Applied Review results over all 33 archived configurations (%). Column abbreviations correspond to the operations summarized in Table 1; their full evaluation contracts are reported in Table 7.
<table><tr><td>Model</td><td>Case</td><td>Extract.</td><td>Inst.</td><td>Person</td><td>Quant.</td><td>Risk</td><td>Legal</td><td>Provision</td></tr><tr><td>Dianjin-R1-7B</td><td>77.00</td><td>59.00</td><td>90.00</td><td>70.00</td><td>63.00</td><td>31.00</td><td>38.00</td><td>33.00</td></tr><tr><td>FinR1-7B</td><td>76.00</td><td>10.00</td><td>82.00</td><td>82.00</td><td>73.00</td><td>29.00</td><td>64.00</td><td>35.00</td></tr><tr><td>Ling-2.6-Flash</td><td>80.00</td><td>72.00</td><td>78.00</td><td>80.00</td><td>62.00</td><td>75.00</td><td>60.00</td><td>40.00</td></tr><tr><td>Ling-2.6-1T</td><td>82.92</td><td>74.53</td><td>78.80</td><td>80.40</td><td>86.21</td><td>75.68</td><td>64.76</td><td>52.10</td></tr><tr><td>Seed-2.0-Mini</td><td>83.15</td><td>75.42</td><td>86.80</td><td>73.80</td><td>79.31</td><td>66.61</td><td>62.22</td><td>36.95</td></tr><tr><td>Seed-2.0-Lite</td><td>85.62</td><td>84.48</td><td>84.00</td><td>81.80</td><td>79.31</td><td>73.22</td><td>63.81</td><td>56.40</td></tr><tr><td>Nemotron-3-Nano</td><td>66.29</td><td>70.43</td><td>75.60</td><td>73.40</td><td>65.52</td><td>54.92</td><td>53.76</td><td>23.50</td></tr><tr><td>Nemotron-3-Super</td><td>79.78</td><td>72.44</td><td>82.40</td><td>79.20</td><td>75.00</td><td>71.86</td><td>56.72</td><td>30.15</td></tr><tr><td>Nemotron-3-Ultra</td><td>82.02</td><td>81.49</td><td>74.80</td><td>85.60</td><td>81.90</td><td>75.41</td><td>61.69</td><td>29.45</td></tr><tr><td>Qwen2.5-7B</td><td>76.63</td><td>59.14</td><td>78.00</td><td>82.00</td><td>75.00</td><td>53.28</td><td>62.96</td><td>40.55</td></tr><tr><td>Qwen3-1.7B</td><td>54.61</td><td>61.79</td><td>79.60</td><td>76.40</td><td>56.03</td><td>71.86</td><td>58.94</td><td>16.20</td></tr><tr><td>Qwen3-4B</td><td>76.40</td><td>75.72</td><td>80.00</td><td>84.00</td><td>75.00</td><td>71.31</td><td>62.33</td><td>12.80</td></tr><tr><td>Qwen3-8B</td><td>81.75</td><td>61.69</td><td>80.32</td><td>83.00</td><td>70.86</td><td>74.59</td><td>64.49</td><td>17.44</td></tr><tr><td>Qwen3-14B</td><td>81.80</td><td>76.82</td><td>78.80</td><td>85.40</td><td>74.14</td><td>68.85</td><td>64.97</td><td>33.40</td></tr><tr><td>Qwen3-32B</td><td>82.11</td><td>74.63</td><td>90.00</td><td>84.36</td><td>79.66</td><td>73.77</td><td>62.96</td><td>27.44</td></tr><tr><td>Qwen3-Next-80B</td><td>84.94</td><td>76.02</td><td>90.80</td><td>81.60</td><td>80.17</td><td>73.77</td><td>65.71</td><td>41.30</td></tr><tr><td>Qwen3-235B-A22B</td><td>83.37</td><td>81.25</td><td>88.00</td><td>83.40</td><td>81.03</td><td>62.84</td><td>65.08</td><td>40.50</td></tr><tr><td>Qwen3.7-Max</td><td>85.39</td><td>82.79</td><td>84.00</td><td>84.40</td><td>78.45</td><td>75.68</td><td>66.03</td><td>46.85</td></tr><tr><td>DeepSeek-V3-0324</td><td>83.15</td><td>79.12</td><td>84.40</td><td>80.40</td><td>76.72</td><td>74.86</td><td>63.70</td><td>34.15</td></tr><tr><td>DeepSeek-V4-Flash</td><td>82.92</td><td>51.84</td><td>88.00</td><td>83.40</td><td>72.41</td><td>75.41</td><td>63.70</td><td>53.00</td></tr><tr><td>DeepSeek-V4-Pro</td><td>85.62</td><td>76.02</td><td>86.80</td><td>85.20</td><td>68.97</td><td>70.49</td><td>64.97</td><td>41.45</td></tr><tr><td>Kimi-K2.5</td><td>88.09</td><td>84.28</td><td>87.20</td><td>83.60</td><td>75.86</td><td>56.56</td><td>65.93</td><td>46.40</td></tr><tr><td>Kimi-K3</td><td>87.19</td><td>74.43</td><td>84.80</td><td>87.00</td><td>77.59</td><td>74.86</td><td>68.15</td><td>51.65</td></tr><tr><td>GLM-5</td><td>84.72</td><td>81.39</td><td>87.60</td><td>84.20</td><td>84.48</td><td>73.22</td><td>64.76</td><td>51.50</td></tr><tr><td>GLM-5.2</td><td>84.94</td><td>81.69</td><td>85.60</td><td>83.60</td><td>78.45</td><td>71.86</td><td>65.71</td><td>51.15</td></tr><tr><td>GPT-5.4-Mini</td><td>85.62</td><td>83.78</td><td>81.60</td><td>85.80</td><td>76.72</td><td>71.86</td><td>64.97</td><td>55.35</td></tr><tr><td>GPT-5.4</td><td>86.29</td><td>83.98</td><td>82.00</td><td>85.60</td><td>75.86</td><td>74.04</td><td>67.51</td><td>60.45</td></tr><tr><td>GPT-5.6-Luna</td><td>84.04</td><td>85.97</td><td>85.60</td><td>85.80</td><td>75.00</td><td>71.86</td><td>65.93</td><td>50.80</td></tr><tr><td>GPT-5.6-Terra</td><td>85.62</td><td>83.48</td><td>90.40</td><td>85.80</td><td>76.72</td><td>73.77</td><td>67.83</td><td>56.95</td></tr><tr><td>GPT-5.6-Sol</td><td>87.42</td><td>85.07</td><td>90.00</td><td>86.00</td><td>77.59</td><td>74.59</td><td>67.09</td><td>67.55</td></tr><tr><td>Gemini-2.5-Flash</td><td>82.25</td><td>75.72</td><td>89.20</td><td>85.60</td><td>64.66</td><td>72.95</td><td>60.11</td><td>66.25</td></tr><tr><td>Gemini-3-Flash-preview</td><td>84.94</td><td>67.96</td><td>93.60</td><td>85.80</td><td>75.00</td><td>76.78</td><td>59.68</td><td>65.75</td></tr><tr><td>Claude-Opus-4.7</td><td>85.17</td><td>76.62</td><td>63.20</td><td>50.80</td><td>73.28</td><td>75.14</td><td>65.19</td><td>54.80</td></tr></table>

## F. Qualitative Case Studies

This section provides qualitative examples illustrating how FinRiskAtlas instantiates operation-level evaluation contracts. The examples are not intended to compare model performance or serve as additional benchmark statistics. Instead, they demonstrate how diferent professional decisions require diferent visible evidence boundaries, reviewer artifacts, and scoring criteria. The static examples are organized according to the three benchmark layers introduced in Section 3: Domain Knowledge, Evidence-Grounded Processing, and Applied Review.

Each static example contains the evaluation input, the reference decision or artifact, and an example model output. The displayed outputs are observable generations returned by the model and do not represent reconstructed hidden reasoning. For open-ended Applied Review tasks, evaluation scores are produced by the fixed semantic evaluator under the corresponding rubric; the examples illustrate the evaluation contract rather than standalone evidence of general model capability.

## F.1. Domain Knowledge

The Domain Knowledge layer measures whether models possess conceptual and regulatory knowledge required to interpret financial-review evidence. Figure 7 presents an AML compliance example in which the model must identify the applicable customer-identification penalty range. The case illustrates that even

Table 10: Complete open-ended Applied Review results over all 33 archived configurations (%). DeepSeek-V4-Flash is the fixed semantic evaluator. Its self-scored row is marked with † and excluded from best-value annotation and evaluator-dependent comparisons.
<table><tr><td>Model</td><td>Legal Judgment</td><td>Decision View</td><td>Disputed Issue</td></tr><tr><td>Dianjin-R1-7B</td><td>29.00</td><td>20.00</td><td>36.00</td></tr><tr><td>FinR1-7B</td><td>18.00</td><td>15.00</td><td>52.00</td></tr><tr><td>Ling-2.6-Flash</td><td>30.00</td><td>37.00</td><td>63.00</td></tr><tr><td>Ling-2.6-1T</td><td>40.27</td><td>55.69</td><td>63.30</td></tr><tr><td>Seed-2.0-Mini</td><td>28.95</td><td>47.11</td><td>72.73</td></tr><tr><td>Seed-2.0-Lite</td><td>29.66</td><td>58.73</td><td>73.35</td></tr><tr><td>Nemotron-3-Nano</td><td>21.48</td><td>17.09</td><td>39.57</td></tr><tr><td>Nemotron-3-Super</td><td>28.62</td><td>37.17</td><td>76.67</td></tr><tr><td>Nemotron-3-Ultra</td><td>30.68</td><td>52.00</td><td>71.58</td></tr><tr><td>Qwen2.5-7B</td><td>28.10</td><td>36.24</td><td>70.40</td></tr><tr><td>Qwen3-1.7B</td><td>16.18</td><td>7.76</td><td>41.13</td></tr><tr><td>Qwen3-4B</td><td>19.38</td><td>21.99</td><td>68.83</td></tr><tr><td>Qwen3-8B</td><td>17.37</td><td>30.05</td><td>37.30</td></tr><tr><td>Qwen3-14B</td><td>35.02</td><td>39.03</td><td>53.81</td></tr><tr><td>Qwen3-32B</td><td>34.54</td><td>35.60</td><td>53.30</td></tr><tr><td>Qwen3-Next-80B</td><td>32.43</td><td>53.10</td><td>67.51</td></tr><tr><td>Qwen3-235B-A22B</td><td>39.26</td><td>49.00</td><td>57.35</td></tr><tr><td>Qwen3.7-Max</td><td>36.22</td><td>57.84</td><td>70.58</td></tr><tr><td>DeepSeek-V3-0324</td><td>36.26</td><td>71.31</td><td>51.68</td></tr><tr><td>DeepSeek-V4-Flash†</td><td>25.16</td><td>54.86</td><td>81.23</td></tr><tr><td>DeepSeek-V4-Pro</td><td>30.67</td><td>54.48</td><td>77.12</td></tr><tr><td>Kimi-K2.5</td><td>34.49</td><td>62.74</td><td>61.26</td></tr><tr><td>Kimi-K3</td><td>35.05</td><td>64.13</td><td>74.84</td></tr><tr><td>GLM-5</td><td>31.60</td><td>57.16</td><td>72.49</td></tr><tr><td>GLM-5.2</td><td>31.59</td><td>58.59</td><td>68.37</td></tr><tr><td>GPT-5.4-Mini</td><td>39.86</td><td>48.77</td><td>62.13</td></tr><tr><td>GPT-5.4</td><td>46.36</td><td>62.91</td><td>54.04</td></tr><tr><td>GPT-5.6-Luna</td><td>41.70</td><td>52.77</td><td>61.33</td></tr><tr><td>GPT-5.6-Terra</td><td>40.98</td><td>54.72</td><td>58.29</td></tr><tr><td>GPT-5.6-Sol</td><td>42.09</td><td>62.25</td><td>58.11</td></tr><tr><td>Gemini-2.5-Flash</td><td>45.29</td><td>65.27</td><td>74.05</td></tr><tr><td>Gemini-3-Flash-preview</td><td>40.48</td><td>64.61</td><td>74.69</td></tr><tr><td>Claude-Opus-4.7</td><td>35.53</td><td>51.97</td><td>74.06</td></tr></table>

knowledge-oriented evaluation requires a precise decision target: the model must distinguish the requested administrative fine from other possible sanctions associated with diferent legal consequences.

## F.2. Evidence-Grounded Processing

Evidence-Grounded Processing evaluates whether models can transform heterogeneous records into structured, decision-relevant evidence. The following examples cover four representative operations: long-context case classification, quantitative verification, structured information extraction, and entity reconciliation. Together, they illustrate that evidence processing is not a single retrieval capability but a collection of operations with diferent decision objects and output contracts.

## F.3. Applied Review

The Applied Review layer evaluates whether models can integrate evidence, apply professional decision frameworks, and generate case-level artifacts. These examples highlight a central motivation of FinRiskAtlas: the same underlying case material can support multiple professional operations, while each operation requires a diferent output artifact and evaluation criterion.

Figure 12 illustrates disputed-issue generation, where the required artifact is a structured issue list rather than a final judgment. Figure 13 illustrates legal-judgment generation, which requires synthesizing evidence,

Table 11: Complete FinRisk-Ask results for two reference policies and 33 archived model configurations (%). ERA is the end-to-end evidence-acquisition metric. DeepSeek-V4-Flash is the fixed request-alignment evaluator; its self-scored row is marked with † and excluded from evaluator-dependent comparisons.
<table><tr><td rowspan="2">Model</td><td>End-to-end</td><td colspan="3">Action behavior</td><td rowspan="2"></td><td colspan="3">Request alignment</td></tr><tr><td>ERA</td><td>BAcc</td><td>AskR ProceedR</td><td></td><td>RAA</td><td>CRA DEH</td><td>REH</td></tr><tr><td>Reference policies</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Always-Ask</td><td></td><td>50.00</td><td>100.00</td><td>0.00</td><td>85.74</td><td></td><td></td><td></td></tr><tr><td>Always-Proceed</td><td></td><td>50.00</td><td>0.00</td><td>100.00</td><td>14.26</td><td></td><td></td><td></td></tr><tr><td>Model configurations</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Dianjin-R1-7B</td><td>57.72</td><td>45.97</td><td>91.94</td><td>0.00</td><td>78.82 62.78</td><td></td><td>51.97</td><td>63.46</td></tr><tr><td>FinR1-7B</td><td>48.46</td><td>50.00</td><td>100.00</td><td>0.00</td><td>85.74 48.46</td><td></td><td>45.45</td><td>51.46</td></tr><tr><td>Ling-2.6-Flash</td><td>51.80</td><td>56.03</td><td>75.99</td><td>36.08</td><td>70.29 68.17</td><td></td><td>49.39</td><td>54.20</td></tr><tr><td>Ling-2.6-1T</td><td>79.75</td><td>51.89</td><td>96.57</td><td>7.22</td><td>83.82 82.58</td><td></td><td>76.50</td><td>83.01</td></tr><tr><td>Seed-2.0-Mini</td><td>54.54</td><td>48.65</td><td>75.64</td><td>21.65</td><td>67.94 72.10</td><td></td><td>52.31</td><td>56.77</td></tr><tr><td>Seed-2.0-Lite</td><td>42.96</td><td>44.36</td><td>60.89</td><td>27.84</td><td>56.18 70.55</td><td></td><td>40.30</td><td>45.62</td></tr><tr><td>Nemotron-3-Nano</td><td>24.78</td><td>39.48</td><td>40.82</td><td>38.14</td><td>40.4460.70</td><td></td><td>22.98</td><td>26.58</td></tr><tr><td>Nemotron-3-Super</td><td>36.27</td><td>46.77</td><td>56.43</td><td>37.11</td><td>53.6864.27</td><td></td><td>34.99</td><td>37.56</td></tr><tr><td>Nemotron-3-Ultra</td><td>67.75</td><td>55.69</td><td>82.50</td><td>28.87</td><td>74.85 82.12</td><td></td><td>66.20</td><td>69.29</td></tr><tr><td>Qwen2.5-7B</td><td>26.84</td><td>43.52</td><td>45.80</td><td>41.24</td><td>45.15 58.61</td><td></td><td>25.04</td><td>28.64</td></tr><tr><td>Qwen3-1.7B</td><td>31.38</td><td>51.38</td><td>94.51</td><td>8.25</td><td>82.21 33.20</td><td></td><td>29.33</td><td>33.44</td></tr><tr><td>Qwen3-4B</td><td>34.30</td><td>54.68</td><td>55.75</td><td>53.61</td><td>55.44 61.53</td><td></td><td>32.93</td><td>35.67</td></tr><tr><td>Qwen3-8B</td><td>48.79</td><td>59.39</td><td>69.30</td><td>49.48</td><td>66.47 70.41</td><td></td><td>46.14</td><td>51.45</td></tr><tr><td>Qwen3-14B</td><td>50.77</td><td>54.83</td><td>75.64</td><td>34.02</td><td>69.71</td><td>67.12</td><td>48.88</td><td>52.65</td></tr><tr><td>Qwen3-32B</td><td>48.54</td><td>58.19</td><td>65.87</td><td>50.52</td><td>63.68 73.69</td><td></td><td>45.96</td><td>51.11</td></tr><tr><td>Qwen3-Next-80B</td><td>57.71</td><td>56.21</td><td>73.24</td><td>39.18</td><td>68.38 78.79</td><td></td><td>56.08</td><td>59.34</td></tr><tr><td>Qwen3-235B-A22B</td><td>62.86</td><td>53.63</td><td>79.42</td><td>27.84</td><td>72.06 79.15</td><td></td><td>61.57</td><td>64.15</td></tr><tr><td>Qwen3.7-Max</td><td>50.68</td><td>43.85</td><td>57.80</td><td>29.90</td><td>53.82 87.67</td><td></td><td>49.91</td><td>51.45</td></tr><tr><td>DeepSeek-V3-0324</td><td>45.79</td><td>49.52</td><td>57.80</td><td>41.24</td><td>55.44 79.22</td><td></td><td>44.42</td><td>47.16</td></tr><tr><td>DeepSeek-V4-Flash†</td><td>59.51</td><td>50.19</td><td>81.82</td><td>18.56</td><td>72.79 72.73</td><td></td><td>57.63</td><td>61.40</td></tr><tr><td>DeepSeek-V4-Pro</td><td>60.29</td><td>50.71</td><td>75.64</td><td>25.77</td><td>68.53 79.70</td><td></td><td>58.49</td><td>62.09</td></tr><tr><td>Kimi-K2.5</td><td>66.80</td><td>55.68</td><td>83.53</td><td>27.84</td><td>75.59 79.97</td><td></td><td>65.35</td><td>68.26</td></tr><tr><td>Kimi-K3</td><td>77.1059.46</td><td></td><td>89.02</td><td>29.90</td><td>80.5986.61</td><td></td><td>76.50</td><td>77.70</td></tr><tr><td>GLM-5</td><td>55.66</td><td>55.26</td><td>73.41</td><td>37.11</td><td>68.24 75.82</td><td></td><td>54.20</td><td>57.11</td></tr><tr><td>GLM-5.2</td><td>58.49</td><td>53.37</td><td>71.70</td><td>35.05</td><td>66.47 81.58</td><td></td><td>57.28</td><td>59.69</td></tr><tr><td>GPT-5.4-Mini</td><td>39.27</td><td>48.15</td><td>51.97</td><td>44.33</td><td>50.88 75.56</td><td></td><td>38.25</td><td>40.30</td></tr><tr><td>GPT-5.4</td><td>59.17</td><td>50.45</td><td>72.04</td><td>28.87</td><td>65.88 82.13</td><td></td><td>57.97</td><td>60.37</td></tr><tr><td>GPT-5.6-Luna</td><td>77.87</td><td>54.21</td><td>95.03</td><td>13.40</td><td>83.38 81.95</td><td></td><td>75.81</td><td>79.93</td></tr><tr><td>GPT-5.6-Terra</td><td>75.64</td><td>56.11</td><td>91.60</td><td>20.62</td><td>81.47 82.58</td><td></td><td>74.09</td><td>77.18</td></tr><tr><td>GPT-5.6-Sol</td><td>67.75</td><td>54.05</td><td>83.36</td><td>24.74</td><td>75.0081.27</td><td></td><td>66.55</td><td>68.95</td></tr><tr><td>Gemini-2.5-Flash</td><td>62.60</td><td>49.33</td><td>74.96</td><td>23.71</td><td></td><td>67.65 83.51</td><td>61.23</td><td>63.97</td></tr><tr><td>Gemini-3-Flash-preview</td><td>64.49</td><td>52.51</td><td>77.19</td><td>27.84</td><td></td><td>70.15 83.55</td><td>62.95</td><td>66.03</td></tr><tr><td>Claude-Opus-4.7</td><td>75.98</td><td>56.11</td><td>88.51</td><td>23.71</td><td></td><td>79.26 85.85</td><td>74.09</td><td>77.87</td></tr></table>

legal elements, and remedies into a supported analysis. Figure 14 illustrates a structured instance of decisionview generation, where the required review position is expressed through a constrained dispositive schema rather than unrestricted prose.

## F.4. FinRisk-Ask: Action Agreement versus Request Targeting

FinRisk-Ask evaluates evidence-state control through ofline replay rather than interactive dialogue. The model receives only the information available before the recorded reviewer transition. Later trajectory information is withheld during inference and is used only to construct expert-verified evaluation targets.

The three examples below illustrate three request-alignment outcomes. They correspond to direct alignment, partial alignment, and missed evidence acquisition. Each example separates the visible pre-action context from the observed future evidence used only for evaluation. The examples demonstrate why entering the Ask branch and producing a useful evidence request are distinct capabilities.

![](images/e96d42c7dcfe27bb7ff50e058034803903ba0cd6b1b8462c380577329ec8c367.jpg)  
Figure 7: Domain Knowledge example: AML customer-identification compliance. The task evaluates whether the model can retrieve the applicable customer-identification requirement and determine the stated upper bound of the corresponding administrative fine. Solving the item requires distinguishing the directly applicable penalty range from more severe sanctions associated with diferent consequences or aggravating conditions.

![](images/d8d9f6d2b78a3c33f18865e914b6a2e360fe4945c522e7ccde04b48f1d9765fb.jpg)  
Figure 8: Evidence-Grounded Processing example: long-context maritime contract classification. The input contains contracts, waybills, delivery records, freight and insurance amounts, performance evidence, and procedural events. The model must distinguish the operative legal relationship from contextual and procedural distractors and select the most specific case category.

Where case cards display CONTINUE and STOP, CONTINUE denotes Ask and STOP denotes Proceed in the archived output protocol. These tokens represent the replayed action transition and should not be interpreted as an interactive dialogue policy or as a claim that the recorded workflow is the only valid professional strategy.

An Ask prediction with an invalid or unrelated request receives zero request-alignment credit despite entering the correct branch. Conversely, a Proceed prediction on a recorded Ask state fails to enter request evaluation because no evidence request is produced. These cases illustrate why action agreement, Ask-state coverage, and request targeting are reported separately in FinRisk-Ask.

## G. Limitations and Responsible Use

Scope and interpretation. FinRiskAtlas focuses on Chinese-language financial risk-control and compliance review and evaluates model configurations under explicitly defined operation contracts. The reported scores should therefore be interpreted as measurements of performance on the released review operations and evidence states, rather than as a general certification of financial-review capability across all institutions, jurisdictions, or workflows. The benchmark is designed to study how evaluation units influence model selection in professional review settings.

![](images/89405db6ee099ebc48f00171659696a467ce688592ee9fb90c5ff12d06d96144.jpg)  
Figure 9: Evidence-Grounded Processing example: quantitative financial reasoning. The efectiveprotection-rate task requires the model to identify the relevant variables, apply the correct formula, compute the tarif burden on imported inputs, and normalize by domestic value added.

![](images/c8728f46d9e7573b291553d53fef796d17d83d22bedf48d91cbcfe740d63e2a7.jpg)  
Figure 10: Evidence-Grounded Processing example: multi-field extraction from a long legal record. The model must extract five evidence spans in a predefined order while ignoring unrelated monetary values, dates, and procedural events. The task contract evaluates field correctness, field ordering, delimiter validity, and agreement with the reference output.

![](images/57ea49a2d49d1bafa0792757c639c681593313028f1c53500c3c3e21b7b86936.jpg)  
Figure 11: Evidence-Grounded Processing example: cross-lingual company entity resolution. The task requires the model to determine whether Chinese and English company names refer to the same entity. The decision depends on the joint alignment of geographic information, transliterated name components, translated industry terms, and legal-entity sufixes.

![](images/d8a218a00b19cfb53ff256cd6709097784ed890adbb95c0d97ef99c30093b25c.jpg)  
Figure 12: Applied Review example: open-ended disputed-issue generation. The source case contains overlapping agency, letter-of-credit, guarantee, insurance, and payment relationships. The model must transform these facts into a structured set of issues covering the responsible parties, contractual instruments, disputed conduct, claimed amounts, interest and fee calculations, and alternative liability relationships.

![](images/0f3483ee4e81d5ce93f550110c5ce70003ea5e806680bc7e944f27c86bb1de09.jpg)  
Figure 13: Applied Review example: legal-judgment generation. The model must identify the appropriate legal route, establish the claimant’s standing, synthesize transaction and valuation evidence, determine whether the challenged asset transfer impaired creditors, address the disputed execution settlement, and produce an appropriate dispositive analysis.

Evidence-state evaluation boundary. FinRisk-Ask evaluates evidence-state control through ofline replay of completed professional trajectories. The benchmark measures whether a model follows the recorded transition and whether its request aligns with trajectory-supported evidence needs under the released evaluation protocol. Because targets are derived from evidence realized in completed trajectories, FinRisk-Ask evaluates alignment with verified evidence needs rather than executing requests in a live review environment or estimating their downstream operational impact.

Professional reference behavior. The Ask-or-Proceed labels represent observed reviewer transitions from one enterprise risk-control workflow. They provide a consistent reference boundary for evaluation while preserving the fact that professional workflows may difer across organizations, policies, and operating conditions. Accordingly, FinRisk-Ask measures agreement with the released workflow behavior and requesttarget alignment, rather than defining a universal review policy.

![](images/c87c3d1f4d48dcb13557a6ff6c6411487ef85d76b2c6fed8a81acea39da41d91.jpg)  
Figure 14: Applied Review example: structured decision-view generation. The model must express the supported review position in a constrained dispositive schema that preserves the afected parties, monetary obligations, and procedural treatment. This example illustrates a structured output form within the decisionview generation contract rather than an additional downstream operation.

![](images/81fdc64a99adb56553674dd00687b1b3bfeb52bb43af1a69a357faff335a9f1f.jpg)  
Figure 15: Direct request alignment (z<sub>t</sub> = 1). The visible record leaves a proxy-registration or batchonboarding concern unresolved. The model follows the recorded Ask branch and requests verification of that concern. The request matches a retained need supported by the later trajectory, so the state is both an Ask-action hit and a direct alignment hit. The observed-future panel was unavailable at inference.

Data provenance and annotation. FinRiskAtlas combines benchmark instances constructed from profes sional materials, regulatory resources, financial and legal records, and de-identified review trajectories. Static benchmark families and FinRisk-Ask states were defined and reviewed by domain experts with financial risk and legal backgrounds. Expert review was applied to taxonomy design, family contracts, instance quality control, action mapping, and evidence-target verification. Raw enterprise trajectories are not released; only de-identified reconstructed states and evaluation artifacts are provided.

Responsible use. FinRiskAtlas is intended as an evaluation resource for studying and comparing model configurations in professional financial-review scenarios. It is not designed to replace qualified reviewers or support autonomous financial or legal decisions. Deployment in real workflows should additionally consider institution-specific policies, regulatory requirements, human oversight, and operational constraints beyond the benchmark contracts.

![](images/56afe311585b68e98db6ecc2c0a6357d5f16c08ef43ae12ceae9aaea58e63e4b.jpg)

Figure 16: Partial request alignment $( z _ { t } = 0 . 5 )$ . The model follows the recorded Ask branch, but the question captures only part of the retained storefront-authenticity need and remains underspecified relative to the closest verified target. Selecting Ask is an action hit; full request-alignment credit is not automatic. Credit is assigned by the best-matching retained target, so the model need not enumerate every item that later appeared.  
![](images/2aa79470e7e0d7a8949030ed9b18dc07c2d7fcafd3c7cde5496997364faf2344.jpg)  
Figure 17: Missed information acquisition $( z _ { t } = 0 )$ . The recorded state is Ask, but the model selects Proceed and issues no request. Because the Ask branch is not entered, no candidate request is compared with the retained target set and the state contributes zero to ERA. The state is an Ask-recall error even if proceeding from the visible record appears locally plausible.