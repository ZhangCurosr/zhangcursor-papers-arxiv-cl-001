# Provenance Before Prose: Claim-Locked Reporting for Statistical Text Generation

Xiao Fan, Jingyuan Li<sup>\*</sup>, Hongbin Guo, Yubo Han, Yi Zhang<sup>\*</sup>

Xidian University

{xiufan, hongbin, hanyubo}@stu.xidian.edu.cn {lijingyuan, yizhang}@xidian.edu.cn

Corresponding authors.

§ Code

Abstract Large language models (LLMs) can fluently verbalize statistical evidence, yet statistical reports can still drift numerical values, invert efect directions, or restate thresholded contrasts as categorical efects. We frame these failures as a control problem: the evidence-bearing content of a scientific report should be fixed by structured statistical results rather than sampled during prose generation. We therefore use cross-run reproducibility to stress-test whether report-visible numbers and claims are bound before prose generation. Existing controls operate at the text or slot level; a deterministic hybrid template reproduces only 61.1% of report-visible numerical content across seeds because the LLM still selects which findings and numbers the template renders. We propose claim-locked reporting, a provenance-before-prose protocol that fixes the evidence source, numbers, direction, and allowed language strength of each reportable claim before the LLM writes connective wording. Across fMRI functional-connectivity reporting and randomized controlled trial reporting on Evidence Inference 2.0, claim-locked reporting improves reproducibility over the hybrid template by 37.4 and 20.5 points, respectively. Blinded human audits support the observed direction-preservation and governance trends. In an fMRI cost analysis with DeepSeek, claim-locked reporting also yields the lowest observed token use and median generation latency. Code is available at https://github.com/XiuFan719/Claim-Locked-Reporting.

## 1. Introduction

Consider a functional magnetic resonance imaging (fMRI) report produced by a large language model (LLM). Before any prose is written, the statistical results are already fixed by the analysis. Yet across runs, the same results can still lead to diferent reports. One report may include a finding that another omits, report a diferent numerical detail, or describe the same group efect with stronger or weaker wording. Such variation changes the evidence-bearing content of the report. When the underlying results have not changed, the set of reported findings and numbers, together with their interpretive strength, should remain stable across runs.

LLMs are increasingly used to draft scientific and clinical reports, and much of the current reliability framing asks whether generated text is faithful to an observable or retrievable source, such as an image, a patient record, or a retrieved passage [Van Veen et al., 2024, Bannur et al., 2024, Tu et al., 2024, Singhal et al., 2023]. For LLM-generated statistical reports, the reliability question is diferent. Functional connectivity (FC) analyses, randomized controlled trial (RCT) summaries, and epidemiological reports are written from statistical results that have already been computed: counts, efect estimates, directions, thresholds, covariate conditions, or evidence spans [Bullmore and Sporns, 2009, Zalesky et al., 2010, Marek et al., 2022]. A report may use numbers or terms that appear in the source while still changing the statistical claim it conveys, for example by drifting a number, reversing a direction, or overstating the strength of an association. We refer to this family of errors as statistical claim distortion. Reliability in statistical reporting requires preserving not only source-supported content, but also the statistical meaning of that content.

Existing controls intervene at diferent points in the generation pipeline. For clarity, we group them into two levels. Text-level control includes prompting, retrieval [Lewis et al., 2020], structured output [Willard and Louf, 2023, Beurer-Kellner et al., 2023], and post-hoc verification [Manakul et al., 2023]. These methods can constrain the generation process or output form, but the model still decides which statistical inferences to state and how strongly to state them. Slot-level control combines selected fields with deterministic surface realization [Reiter and Dale, 2000, Gatt and Krahmer, 2018]. It ensures that a selected field is rendered consistently, but it does not decide which field should be selected in the first place. When the LLM chooses the slots, the reported numbers, directions, and language strength can still vary across runs.

This leaves a claim-level gap. Existing controls can constrain the text or render selected slots, but they do not fix the reportable statistical claim before rendering. We address this gap with claim-locked reporting, a third control level that binds each reportable claim to its evidence source, numbers, efect direction, and allowed language strength before the LLM writes. A claim builder converts the structured evidence record into a claim ledger; a risk auditor and policy controller determine which claims are admissible and how strongly they may be stated; a deterministic renderer emits numbers, tables, entity labels, and direction tags from the ledger; and the LLM writes only connective prose around the locked claims. The protocol changes what the model is allowed to decide. We evaluate claim-locked reporting across FC reporting and RCT reporting using complementary automatic and human analyses of report stability, statistical correctness, and claim governance. Additional experiments examine component contributions and practical generation cost. Our contributions are summarized

as follows:

• We frame statistical claim distortion as a reliability problem in LLM-generated statistical reporting, and use cross-run reproducibility as a stress test of whether evidence-bearing content is fixed before prose generation.

• We propose claim-locked reporting, a provenance-beforeprose protocol that binds each reportable claim to its evidence source, numbers, direction, and allowed language strength before prose generation.

• We evaluate the protocol on obesity-related FC reporting and RCT reporting on Evidence Inference 2.0 against grounded baselines, supplemented by component analysis and blinded human audits. Claim-locked reporting improves reproducibility over the hybrid template by 37.4 and 20.5 points in the two settings, respectively.

## 2. Related Work

Statistical reporting versus faithful generation. Classical natural language generation separates content planning from surface realization [Reiter and Dale, 2000, Gatt and Krahmer, 2018]: the system first decides what to communicate and then how to express it. Neural data-to-text generation later learned content selection and realization jointly from structured records [Lebret et al., 2016, Wiseman et al., 2017], and explicitly modeling content selection and planning was subsequently shown to improve generation quality [Puduppully et al., 2019]. Statistical reporting, however, exposes a distinction this line of work does not make: selecting the correct records does not guarantee a correct statistical claim. The same supported evidence can still be verbalized with the wrong direction, stripped of an adjustment condition, or stated with unjustified strength. LLM-era faithfulness evaluation operates on the generated output. FActScore [Min et al., 2023] decomposes text into atomic facts and evaluates whether they are supported, while SelfCheckGPT [Manakul et al., 2023] uses consistency across sampled generations as a signal of hallucination; broader surveys similarly organize failures around factual support and consistency [Ji et al., 2023, Huang et al., 2025]. These methods evaluate properties of generated content rather than explicitly representing the statistical claim that the content supports. A report may therefore contain individually supported facts while changing their joint statistical meaning. Such errors concern what the report claims, not merely whether its individual facts appear in the source. We therefore treat these failures as a control problem rather than relying solely on post-generation detection.

Constrained generation for statistical reporting. Existing approaches improve grounding or constrain generation through retrieval-augmented generation (RAG) [Lewis et al., 2020], constrained decoding and structured generation [Willard and Louf, 2023, Beurer-Kellner et al., 2023], and post-hoc verification [Manakul et al., 2023]. These approaches constrain evidence access, output structure, or generated text. Statistical reporting, however, requires the numerical value, direction, inferential scope, and interpretive strength of a claim to remain jointly tied to its supporting evidence. Claim-locked reporting therefore treats the provenance-bound statistical claim, rather than text or selected fields, as the unit of control.

## 3. Claim-Locked Reporting

## 3.1 Statistical Evidence Record

Claim-locked reporting starts from a structured evidence record rather than from raw imaging or trial documents. The record is the fixed statistical input to the claim builder and contains only results that have already been computed by a domain analysis pipeline.

In the FC instantiation, subject-level connectivity matrices and phenotypic variables are converted into this record by a standard edge-wise statistical analysis. We fit an edge-wise generalized linear model (GLM) contrasting binary group labels against demographic covariates, followed by Benjamini–Hochberg false-discovery-rate (FDR) control across M = N(N − 1)/2 region-of-interest (ROI) pairs [Benjamini and Hochberg, 1995]. A parallel model additionally adjusts for body mass index (BMI), the continuous covariate used to derive the group labels, providing the covariateabsorption stress test (§5.2). The resulting record stores FDR-surviving edge counts, lobe-pair summaries, hub descriptions, representative edges, optional behavior associations, and covariate-adjusted sensitivity results. For FC literature-context claims, an LLM-assisted literature-support label initializes the wording-strength ceiling and is frozen before report generation.

In the RCT instantiation, the evidence record is provided by Evidence Inference 2.0 [DeYoung et al., 2020]. Each instance contains an intervention, comparator, outcome, direction label, and an evidence span containing the supporting numerical content.

## 3.2 Locked Statistical Claim

Three components construct and constrain the locked claims before report generation: the claim builder, risk auditor, and policy controller. They operate over a shared data structure, the claim ledger.

Claim builder. The claim builder converts the evidence record into the claim ledger, a set of typed claims. Each claim stores a claim type, canonical text, one or more evidence pointers into the record, the numerical fields it reports, an efect direction when applicable, a literature-support level, and an allowed language strength. A claim is instantiated only when its required fields resolve to evidence pointers. As a result, a statement such as “the analysis demonstrates a causal mechanism” is never instantiated unless such a claim is explicitly supported by the evidence record. Because downstream rendering and writing are driven by the ledger, this step fixes the set of inferences the report may contain before the LLM is involved.

Risk auditor. The risk auditor is a read-only layer. It inspects each claim and emits a per-claim audit record without modifying the ledger. Each claim is checked against a prespecified set of statistical-reporting risk categories. The taxonomy is organized around five evidence-bearing properties that must be preserved when a statistical claim is verbalized: numerical support, direction preservation, entity/source support, inferential scope, and interpretive strength. Domainspecific audit rules instantiate these properties using risks documented in the corresponding scientific literature. In FC, these include motion-related confounding [Power et al., 2012], post-selection and circular-analysis efects [Vul et al., 2009, Kriegeskorte et al., 2009], and replicability concerns [Marek et al., 2022, Botvinik-Nezer et al., 2020].

![](images/bb34de532762f190511ba48da2ba4d976fc2cda65fd5db0b98f8d973256ee6fc.jpg)  
Figure 1. Three levels of control for statistical text generation. Text-level methods, including prompting, retrieval, and structured output, leave claims, numbers, directions, and language strength to the LLM. The hybrid template adds slot-level rendering, but the LLM still selects which claims and numbers fill the slots. Claim-locked reporting fixes the evidence source, numbers, direction, and allowed language strength before prose generation; the LLM controls only connective wording.

The categories are domain-specific, but they instantiate the same auditor interface: a fixed trigger over structured evidence fields maps to a policy action before prose generation. In FC, the triggers operate over edges, hubs, ROI names, covariate-adjusted models, and brain–behavior associations. In RCT reporting, they operate over Evidence Inference fields, including intervention, comparator, outcome, direction label, and evidence span. Appendix D summarizes how the shared control targets are instantiated in the two domains.

Direction inversion is handled by binding each directional claim to a ledger-stored direction field rather than to a free-text label, so report-visible direction tags are rendered directly from that field. The auditor also rejects any claim whose evidence pointer fails to resolve against the stage outputs, enforcing the provenance requirement before writing. For each flagged claim, it records the required action to be applied by the policy controller.

Policy controller. The policy controller turns audit records into rendering and writing constraints through two operations. First, it applies a monotone strength downgrade: a flagged claim’s allowed language strength can only be lowered, never raised, and a forbidden claim is marked non-renderable and excluded from every report block. Second, it accumulates writer constraints, namely per-category natural-language directives that are later passed to the LLM as explicit instructions. For group-covariate circularity, for example, the constraint forbids describing the binary group label as an efect independent of the continuous covariate. The auditor decides which claims carry a risk; the policy controller decides what may be said about a flagged claim and whether it remains renderable. The same policy interface applies in the RCT setting, where risk tags are triggered over Evidence Inference fields rather than FC-specific fields.

## 3.3 Deterministic Renderer and LLM Writer

Report writing has two channels. The deterministic renderer emits all numerical content (GLM counts, hub tables, representative edges, limitations) directly from evidence pointers. The LLM writes only short connective paragraphs under the active policy constraints; if its output envelope is malformed or omits required sections, generation fails rather than producing a partial report. An audit outcome never asks the LLM to recompute statistics; it only restricts how a pre-computed claim may be verbalized.

Because both systems use deterministic rendering, their key distinction lies in the renderer input. In the hybrid template, the LLM reads the evidence record, selects values and list entries for predefined slots, and the renderer verbalizes those LLMselected slots. In claim-locked reporting, the renderer receives ledger-bound claims whose source, numbers, direction, and allowed strength were already fixed by the builder and policy controller. The LLM still contributes local coherence, but it no longer decides which statistical content becomes reportable, nor how strongly it is stated.

## 3.4 Worked Example: A Locked Claim

We illustrate the pipeline with the group-efect claim from the in-house FC cohort. The evidence record contains two GLM stages: the primary model reports 10,041 FDR-surviving group-efect edges, whereas the BMI-adjusted model reports 8, a 99.9% collapse. The claim builder therefore creates a single group-efect summary claim pointing to both stages, with the fields 10,041, 8, and 99.9%, a direction field indicating absorption after adjustment, and an initial language-strength limit from the literature-support label. The risk auditor attaches the group-covariate circularity tag, and the policy controller downgrades the claim to cautious language and forbids phrases such as “independent obesity efect” and “obesity-specific mechanism”. The renderer emits the before-and-after counts directly from the ledger, while the LLM writes only cautious connective prose stating that the categorical group label largely reflects continuous BMI variation. Thus, unlike a free-form writer or a hybrid template rendering an LLM-selected primary count, claim-locked reporting fixes the claim, direction, and allowed strength before writing. Appendix A.1 gives the corresponding ledger entry.

## 3.5 Evaluation Axes

We define three evaluation axes to test whether a statisticalreport generator preserves fixed evidence as it moves from a structured record to prose. The evaluation definitions are fixed before evaluation. Implementation details for the automatic metrics are given in Appendix A.2, and the human audits are described in Appendix E. We report the three axes separately because they characterize distinct failure modes: cross-run stability of report-visible content, governance of rhetorical strength, and per-run correctness of numbers and directions.

Evaluation as protocol testing. The evaluation focuses on preservation of fixed statistical evidence through the reporting pipeline. In open-ended factuality evaluation, the model may choose which facts to mention, and the output is later checked against a reference. In claim-locked reporting, the intended behavior is diferent: claim selection, numerical realization, direction framing, and strength calibration should no longer be unconstrained LLM choices. The axes below therefore test whether each control level keeps these properties fixed at the report surface.

b  
![](images/d68ce3c609933003792957ca5414883c35a9399e078b56e9122874ccfab6a4ab.jpg)

![](images/97f30cc414be372dbae1a20a6fca03a29492f3327b3283274a7efe5ad0b259b2.jpg)  
Figure 2. Summary landscape of the reproducibility-error and governance composites per method; lower is better on both axes, which are reported separately rather than merged into a single score. (a) fMRI; (b) RCT.

Cross-run reproducibility. Cross-run reproducibility measures whether the same evidence yields the same report-visible numerical content under diferent seeds and providers. We compute pooled cross-seed Jaccard overlap (J) of reportvisible numerical tokens, reported as a percentage. Numerical matching uses absolute tolerance $1 0 ^ { - \bar { 3 } }$ or relative tolerance 5%, with a heuristic filter for section, table, and ordinal numbers. This axis is used as a stress test for whether numerical content is fixed before prose generation or resampled during generation. For visualization only, Figure 2 plots $z ( 1 - J )$ z-scored across methods.

Claim governance. Claim governance evaluates whether risk-flagged claims are expressed with appropriate rhetorical strength in the final report. We report two direct measures. Strong counts sentences that contain strong assertion terms without a hedge in the same sentence, with lower values indicating fewer over-strong statements. Hedge measures the use of cautionary language for claims flagged by the risk taxonomy. For the summary visualization in Figure 2, we form the governance composite as the unweighted mean of z(Strong) and −z(Hedge), with lower values indicating stronger compliance.

Per-run correctness. The numerical audit flag (Num) marks report-visible numbers without an evidence match. Direction correctness is evaluated separately by setting. For RCT, a blinded human audit measures whether the generated report preserves, inverts, or leaves unresolved the gold direction (§5.4). For FC, a rule-based lexical check compares direction wording against the ledger-stored direction field and is used only as a supplementary diagnostic because lexical direction checks are brittle under negation and comparator phrasing.

The key controlled contrast is between the hybrid template and claim-locked reporting. Both systems render deterministically, so the comparison tests whether reliability improves when the control unit moves from LLM-selected slots to evidence-bound claims. Scalar faithfulness metrics are treated as diagnostic, while blinded human audits are used to check whether the main governance and direction trends are visible to independent annotators.

## 4. Experimental Setup

We evaluate on two statistical-reporting settings that stress diferent parts of the same control problem: cohort-level neuroimaging reports and clinical-trial summaries.

fMRI FC setting. We use obesity-related FC reporting as a representative cohort-level statistical-reporting task: the same phenotypic variables can serve as scientific descriptors, continuous covariates, and confounds, so a report must preserve not only numbers and directions but the inferential scope of each claim after covariate adjustment. We evaluate two cohorts: an obesity-focused institutional cohort of 428 participants (248 obese, 180 normal-weight) and the public Human Connectome Project (HCP) S1200 cohort [Van Essen et al., 2013] with 712 adults (232 obese, 480 normal-weight) after excluding the overweight stratum. Both include demographic and clinical phenotypes as regression covariates, use $g { = } 0$ for BMI < 25 and g=1 for BMI ≥ 30 following World Health Organization (WHO) thresholds [World Health Organization, 2000], and use the Brainnetome-246 parcellation [Fan et al., 2016]. The evaluation spans seven methods across two cohorts, two writer providers, and five seeds, yielding 20 cells per method.

RCT setting. We use Evidence Inference 2.0 [DeYoung et al., 2020], a public BioNLP benchmark of clinical-trial questions paired with full-text articles and human-annotated evidence spans. We sample 200 records deduplicated by source article, retaining only records with annotator consensus, a non-null direction label, and at least two numerical tokens in the evidence span. This yields 112 significantly increased and 88 significantly decreased records; the no-significantdiference class is excluded because a direction-inversion audit is undefined when the gold direction is null. The RCT results are therefore a directional-claim benchmark. Each method runs with two seeds and two providers, yielding 800 cells per method.

Human and scalar sub-samples. For the scalar-metric comparison (§5.5), we evaluate FActScore and SelfCheckGPT on a 30-record RCT sub-sample; cross-metric comparison is restricted to RCT because FActScore’s evidence-span ground truth has no direct fMRI analogue. The same 30 records are used for the human direction audit, yielding $3 0 \times 7 \times 2 \times 2 =$ 840 generation cells. FActScore requires one LLM judge call per generation; SelfCheckGPT follows the original protocol with $N = 1 0$ The fMRI governance sanity check uses matched report snippets from free-form, hybrid template, and claim-locked reporting; four reviewers count two failure types, unhedged strong-language and unsupported-content violations, without seeing method identity.

Methods and fairness. Five grounded baselines combine evidence injection with one structural or post-hoc control each: Free-form, Prompt-only (faithfulness instruction), Structured (output schema), Retrieval (evidence-block citation grounding), and Post-hoc verifier (generate then check). The hybrid template implements slot-level control through deterministic rendering but still lets the LLM select which fields to render, isolating slot-level control from claim-level control. Claimlocked reporting implements the full protocol. We use two writer providers, Moonshot kimi-k2.6 (temperature 1.0) and

Table 1. Main fMRI comparison (n = 20 cells per row). Hedge is risk-conditioned; Num is a numerical audit flag. Bold = best among the seven methods (Repro., Strong, Num.).
<table><tr><td>Method</td><td>Repro. ↑</td><td>Hedge</td><td>Strong ↓</td><td>Num.↓</td></tr><tr><td>Free-form</td><td>32.3</td><td>0.20</td><td>2.95</td><td>0.55</td></tr><tr><td>Prompt-only</td><td>26.0</td><td>0.22</td><td>1.95</td><td>0.55</td></tr><tr><td>Structured</td><td>22.0</td><td>0.23</td><td>1.85</td><td>0.35</td></tr><tr><td>Retrieval</td><td>24.1</td><td>0.21</td><td>1.50</td><td>0.20</td></tr><tr><td>Post-hoc verifier</td><td>15.2</td><td>0.25</td><td>2.75</td><td>0.90</td></tr><tr><td>Hybrid template</td><td>61.1</td><td>0.25</td><td>2.25</td><td>0.00</td></tr><tr><td>Claim-locked</td><td>98.5</td><td>0.39</td><td>0.75</td><td>0.00</td></tr></table>

DeepSeek deepseek-v4-pro (temperature 0.7). All methods receive the identical structured evidence record for a given cohort, provider, and seed; they difer only in the control point applied between record and report. Appendix B gives the per-method control envelope.

## 5. Results

## 5.1 Main Controlled Contrast: Slot Rendering Is Not Enough

Table 1 and Figure 2(a) compare the three control levels of Figure 1: text-level control remains unstable, slot-level control improves stability, and claim-locked reporting makes evidence-bearing content nearly invariant across seeds.

Cross-run reproducibility. Grounded baselines reach 15.2– 32.3% cross-seed reproducibility, indicating that report-visible numerical content remains a sampling outcome under these controls. Deterministic rendering is a major lever but not a sufficient one: the hybrid template reaches only 61.1%, because the LLM still selects which content enters the template. Claimlocked reporting reaches 98.5%. A ledger-rendered system is expected to reach a high absolute value, so the informative quantity is the 37.4-point gap over the hybrid template, which also renders deterministically; the gap isolates the efect of additionally fixing which evidence-bound claims are renderable. A paired bootstrap over matched units (20,000 resamples) gives a 95% confidence interval (CI) of [+15.1, +59.7] points for this diference. The pattern also appears across both writer providers: the pooled metric reaches 98.5% for fMRI and 100.0% for RCT although the two providers have diferent numerical and hedging priors. The same control-level pattern holds on the public HCP cohort alone: claim-locked reaches 98.0% reproducibility versus 62.0% for the hybrid template and 46.0% for free-form generation.

Governance and numerical flags. The remaining columns of Table 1 point the same way. Deterministic rendering alone does not control governance: the hybrid template retains 2.25 unhedged-strong sentences because it still lets the LLM choose claim framing and language strength, whereas claimlocked, which fixes those attributes in the ledger, cuts this to 0.75. The paired diference (Claim-locked minus Hybrid) is −1.50 sentences, with a 95% CI of [−2.05, −1.00]. Claimlocked and the hybrid template both reach 0.00 on the per-run numerical flag. For claim-locked this follows from ledger rendering; for the hybrid it shows that selected numerical slots can be rendered without numerical flags even though selected content remains unstable. The informative numerical tradeof therefore appears in the RCT setting (§5.3); the post-hoc verifier is the least reproducible baseline precisely because its rewrite pass adds report-visible content after the draft has already been sampled.

Table 2. Per-report resource use on the fMRI evaluation with DeepSeek as the writer.
<table><tr><td>Method</td><td>Input tok.</td><td>Output tok.</td><td>Latency (s)</td></tr><tr><td>Free-form</td><td>20,603</td><td>5,432</td><td>111.9</td></tr><tr><td>Hybrid template</td><td>21,213</td><td>7,069</td><td>123.9</td></tr><tr><td>Post-hoc verifier</td><td>44,434</td><td>16,687</td><td>295.2</td></tr><tr><td>Claim-locked</td><td>12,482</td><td>4,209</td><td>77.3</td></tr></table>

Table 3. Group-covariate absorption stress test: FDR-surviving group-efect edges before and after adding the continuous covariate (BMI) used to derive the group label.
<table><tr><td>Cohort</td><td></td><td>Primary + covariate</td><td>Drop</td></tr><tr><td>In-house</td><td>10,041</td><td>8</td><td>99.9%</td></tr><tr><td>HCP comparison</td><td>550</td><td>0</td><td>100.0%</td></tr></table>

What reproducibility does and does not show. Crossrun reproducibility is one-sided evidence. Failing it means evidence-bearing content is still being sampled during generation; passing it means the content is stable, not that it is correct. Numerical, directional, and human-audit measures therefore characterize correctness and governance separately, and the reliability axes are reported as distinct measurements rather than merged into one score.

Component analysis. Table A2 reports three nested configurations that separate the contributions of the main components. The first uses the claim-builder output while leaving numerical realization to the LLM. The second additionally uses the deterministic renderer, with the risk auditor and policy controller disabled, to isolate the efect of deterministic realization. The full configuration then activates the risk auditor and policy controller. For fMRI, reproducibility increases from 85.0% to 98.0% when the deterministic renderer is introduced, while activating the risk auditor and policy controller leaves reproducibility nearly unchanged at 98.5% but reduces Strong from 1.40 to 0.75. The same rendering-related stabilization is observed for RCT, where reproducibility increases from 90.3% to 100.0%.

Operational eficiency. Moving evidence-bearing realization out of LLM generation also reduces writer-side resource use. Table 2 reports per-report resource use across both fMRI cohorts and repeated seeded runs with DeepSeek as the writer. Claim-locked uses one LLM call, while the post-hoc verifier uses two; the deterministic components add no LLM calls and have small local runtime relative to generation latency. Claim-locked yields the lowest observed input/output token use and median latency in this comparison.

## 5.2 A Concrete Failure Case: Group-Covariate Absorption

The group-covariate absorption case is a concrete instance of the failure mode: a statistic that a slot-level system can render with the correct number yet still report at the wrong inferential scope. A primary group contrast in the in-house cohort gives 10,041 FDR-surviving edges; after adding continuous BMI as a covariate, only 8 remain, and the HCP cohort drops from 550 to 0 (Table 3). A free writer, or a template that renders the LLM-selected primary count, can verbalize 10,041 as a categorical obesity-group efect. Claim-locked reporting binds the before-and-after-adjustment claim together with its cautious allowed strength, so the renderer cannot emit the primary count under independent-efect wording. This case illustrates an inferential-scope risk that paragraph-level grounding does not capture.

Table 4. RCT results on Evidence Inference 2.0 (n = 800 cells per row). Strong is near floor because conclusions are short. Num is a conservative, high-recall numerical audit flag.
<table><tr><td>Method</td><td>Repro.↑</td><td>Hedge</td><td>Strong ↓</td><td>Num. ↓</td></tr><tr><td>Free-form</td><td>74.7</td><td>0.24</td><td>0.03</td><td>0.27</td></tr><tr><td>Prompt-only</td><td>85.9</td><td>0.11</td><td>0.01</td><td>0.25</td></tr><tr><td>Structured</td><td>90.0</td><td>0.15</td><td>0.02</td><td>0.25</td></tr><tr><td>Retrieval</td><td>73.2</td><td>0.09</td><td>0.02</td><td>0.23</td></tr><tr><td>Post-hoc verifier</td><td>74.8</td><td>0.22</td><td>0.02</td><td>0.24</td></tr><tr><td>Hybrid template</td><td>79.5</td><td>0.19</td><td>0.04</td><td>0.11</td></tr><tr><td>Claim-locked</td><td>100.0</td><td>0.10</td><td>0.01</td><td>0.23</td></tr></table>

## 5.3 RCT Transfer and a Reliability Trade-of

The RCT setting tests transfer to a public directional-claim benchmark and exhibits a diferent reliability profile from fMRI. Table 4 and Figure 2(b) show that the benchmark does not produce a monotone text-level to slot-level to claim-level reproducibility ladder. Several text-level baselines are already highly reproducible because each RCT output is short and contains few report-visible numerical tokens. In this setting, the hybrid template has less numerical content to stabilize, while its LLM-populated numeric envelope can still vary across seeds. Claim-locked reporting fixes the reportable directional claim and numerical fields before writing, whereas the hybrid template still renders LLM-selected slots. Claimlocked reaches the reproducibility optimum for report-visible numerical tokens (J = 1.000), while the hybrid template has the lowest raw numerical audit flag (0.11 versus 0.23). Relative to the hybrid template, the reproducibility gain is +20.5 points with a 95% paired-bootstrap CI of [+17.8, +23.1].

Because the raw Num column is intentionally high-recall, we further calibrate the numerical audit flags underlying Table 4 (Appendix F) by separating parsing artifacts, benign metadata, and harmful statistical-result fabrications. The calibration finds no harmful fabrication in the claim-locked outputs, whereas the hybrid template and free-form generation yield approximately 2.5% and 5.9%, respectively. This reverses the apparent raw-Num ordering for the risk-bearing component: slot rendering can prune incidental numbers in a single output, but it does not by itself prevent unsupported statistical results when the LLM still selects the numeric slot content. These results show that evidential validity depends on fixing reportable claims before generation in addition to deterministic rendering.

## 5.4 Human Audits

Because Evidence Inference provides gold direction labels whereas the fMRI setting has no comparable report-level gold annotation, the two settings support diferent human checks (Appendix E). The RCT audit evaluates direction preservation directly, while the fMRI audit provides a readerfacing check of governance violations. In the fMRI governance audit, four reviewers count two violation types in matched report snippets without seeing method identity; all four assign claim-locked the lowest average counts, with human means of 2.55/1.85/0.60 for strong-language and 1.65/2.33/0.53 for unsupported-content violations across free-form, hybrid, and claim-locked. The method ordering matches the automatic counts.

Table 5. Human audit of direction preservation on the RCT subset (120 reports per method). Pres. = preserved direction; Inv. = inverted direction; Unres. = unresolved direction, including reports that do not state a clear direction or whose direction cannot be reliably determined. Inv. rate = Inv./(Pres.+Inv.). Unres. is reported descriptively and is excluded from the inversion-rate denominator.
<table><tr><td>Method</td><td>Pres.</td><td>Inv.</td><td></td><td>Unres. Inv. rate (%)</td></tr><tr><td>Free-form</td><td>103</td><td>14</td><td>3</td><td>12.0</td></tr><tr><td>Prompt-only</td><td>102</td><td>10</td><td>8</td><td>8.9</td></tr><tr><td>Structured</td><td>103</td><td>13</td><td>4</td><td>11.2</td></tr><tr><td>Retrieval</td><td>102</td><td>6</td><td>12</td><td>5.6</td></tr><tr><td>Post-hoc verifier</td><td>98</td><td>11</td><td>11</td><td>10.1</td></tr><tr><td>Hybrid template</td><td>102</td><td>17</td><td>1</td><td>14.3</td></tr><tr><td>Claim-locked</td><td>116</td><td>0</td><td>4</td><td>0.0</td></tr></table>

Table 6. FActScore and SelfCheckGPT on the 30-record RCT subsample. Higher is better for both metrics.
<table><tr><td>Method</td><td>FActScore ↑</td><td>SelfCheckGPT ↑</td></tr><tr><td>Post-hoc verifier</td><td>0.856</td><td>0.840</td></tr><tr><td>Structured</td><td>0.855</td><td>0.846</td></tr><tr><td>Retrieval</td><td>0.855</td><td>0.796</td></tr><tr><td>Prompt-only</td><td>0.840</td><td>0.825</td></tr><tr><td>Free-form</td><td>0.835</td><td>0.831</td></tr><tr><td>Hybrid template</td><td>0.770</td><td>0.812</td></tr><tr><td>Claim-locked</td><td>0.552</td><td>0.974</td></tr></table>

The RCT direction audit (Table 5) evaluates whether each report preserves the gold direction, inverts it, or leaves the direction unresolved. A second reviewer independently labeled a stratified 50-report subset, yielding 98.0% raw agreement and Cohen’s κ = 0.970. Across the baselines, inversion rates range from 5.6% to 14.3%, whereas claim-locked reporting produces no inversions. The audit measures preservation of a direction already resolved from the evidence span: baselines must preserve that direction through generation, whereas claim-locked carries the resolved direction into the ledger before generation.

## 5.5 Scalar-Metric Diagnostics

Scalar faithfulness metrics provide a complementary but diferent view of report quality. On the 30-record RCT sub-sample, FActScore [Min et al., 2023] and SelfCheckGPT [Manakul et al., 2023] produce markedly diferent rankings across methods (Table 6). Claim-locked ranks lowest under FActScore (0.552) but highest under SelfCheckGPT (0.974).

The disagreement reflects diferences in what the metrics reward. FActScore emphasizes lexical support against the evidence span, whereas SelfCheckGPT measures consistency across sampled outputs; neither directly tests preservation of a statistical direction or relation.

False pass. In one pharmacokinetics example, the gold result indicates a decrease, but a free-form report states the opposite direction. Because the report reuses numerical values present in the evidence span, FActScore nevertheless assigns a score

of 1.00.

False reject. In another trial, the gold result is an increase from 2.77 to 5.76. Claim-locked renders this relation correctly, but FActScore assigns 0.00 because the direction is expressed through the numerical contrast rather than a lexical match to the evidence span.

These cases show that lexical support, sampling consistency, and statistical claim preservation capture distinct reliability properties.

## 6. Conclusion

We presented claim-locked reporting for statistical text generation. The idea is simple: before an LLM writes, decide which statistical claims are reportable, where they come from, what direction they carry, which numbers they report, and how strongly they may be stated, and then let the model write only the connective prose. This moves the control unit from generated text, to rendered slots, to evidence-bound claims. The hybrid template isolates the question because it shares deterministic rendering but still renders LLM-selected slots. The 61.1%-to-98.5% reproducibility gain attributes additional stability to fixing which evidence-bound claims are renderable before generation. In the RCT setting, claim-locked reporting does not dominate every raw numerical audit flag, but manual calibration finds no harmful statistical-result fabrication in its outputs, whereas slot-level rendering alone still admits unsupported statistical results (Appendix F); blinded human audits support the governance trend. The same design also reduces writer-side token use and median generation latency in the fMRI cost analysis.

## Limitations

Claim-locked reporting controls how structured statistical evidence is verbalized; it does not perform statistical validation, discover mechanisms, or guarantee truth beyond the evidence record. Errors in preprocessing, model specification, covariate selection, or evidence extraction propagate into the report if already present upstream. The BMI absorption result in §5.2 is a stress test for group-covariate circularity, not a biological finding about obesity.

The automatic metrics are scalable protocol tests for predefined statistical-reporting risks, not substitutes for expert judgment, and the risk taxonomy is fixed and benchmarkspecific rather than exhaustive. Cross-run reproducibility is also necessary but not suficient: a report can be reproducibly incorrect, so we pair it with governance and correctness axes rather than treating it as a faithfulness score on its own. The human governance and direction audits are targeted sanity checks rather than a full expert-preference benchmark, and the RCT direction audit evaluates direction preservation after an upstream direction label has been resolved.

The RCT evaluation covers only the increased and decreased direction labels; null or inconclusive findings require an explicit neutral ledger state that forbids eficacy-implying language. The protocol is transferable at the control level but domain-specific in instantiation: a minimal deployment still needs a claim schema, entity inventory, risk tags, and audit patterns. The literature-support label is assigned with LLM assistance and then frozen before generation; future work should replace or validate this step with expert-curated support labels. The institutional FC cohort cannot be redistributed under its data-use protocol; we therefore include the public HCP-based FC setting and the public Evidence Inference 2.0 setting, and release prompts, generated reports, evidence records where permitted, audit scripts, and code.

## Ethics Statement

The institutional FC cohort used in this work was collected under a protocol approved by the Institutional Review Board of Xijing Hospital, and all participants provided written informed consent (trial registration: ChiCTR-OOB-15006346). The data are used under a data-use agreement that does not permit redistribution. The public HCP S1200 cohort is used under the HCP Open Access Data Use Terms, and Evidence Inference 2.0 is a public benchmark of published clinical-trial literature. The blinded audits in this paper involve only the authors and trained research-team members labeling model outputs, a low-risk task that required no separate ethics-board review. A locked report can still be misleading if the upstream evidence record is manipulated or statistically misspecified; claimlocked reporting controls prose generation, not the validity of the upstream analysis.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (Grant Nos. 62431022, 62501453, and 82302292), the National Key R&D Program of China (Grant No. 2022YFC3500603), the Natural Science Basic Research Program ofShaanxi (Grant Nos. 2023-ZDLSF-07 and 2024JC-YBQN-0923), the Xidian University Specially Funded Project for Interdisciplinary Exploration (Grant Nos. TZJH2024012, TZJH2024015, TZJH2024018, and TZJH2024019), the China Postdoctoral Science Foundation (Grant No. 2024M752537), and the Postdoctoral Research Program of Shaanxi (Grant No. 2025BSHSDZZ188).

Generative AI models were used as part of the experimental setup, as described in Section 4. Separately, AI tools were used only for limited language refinement of the manuscript. All scientific content, analyses, interpretation, and conclusions were developed and verified by the authors.

## References

Shruthi Bannur, Kenza Bouzid, Daniel C. Castro, Anton Schwaighofer, Anja Thieme, Sam Bond-Taylor, Maximilian Ilse, Fernando Pérez-García, Valentina Salvatelli, Harshita Sharma, Felix Meissen, Mercy Ranjit, Shaury Srivastav, Julia Gong, Noel C. F. Codella, Fabian Falck, Ozan Oktay, Matthew P. Lungren, Maria Teodora Wetscherek, Javier Alvarez-Valle, and Stephanie L. Hyland. MAIRA-2: Grounded radiology report generation. arXiv preprint arXiv:2406.04449, 2024.

Yoav Benjamini and Yosef Hochberg. Controlling the false discovery rate: A practical and powerful approach to multiple testing. Journal of the Royal Statistical Society, Series B, 57(1):289–300, 1995.

Luca Beurer-Kellner, Marc Fischer, and Martin Vechev. Prompting is programming: A query language for large language models. PACMPL, 7(PLDI):1946–1969, 2023.

Rotem Botvinik-Nezer, Felix Holzmeister, Colin F Camerer,

Anna Dreber, Juergen Huber, Magnus Johannesson, Michael Kirchler, Roni Iwanir, Jeanette A Mumford, R Alison Adcock, et al. Variability in the analysis of a single neuroimaging dataset by many teams. Nature, 582(7810):84–88, 2020.

Ed Bullmore and OlafSporns. Complex brain networks: Graph theoretical analysis of structural and functional systems. Nature Reviews Neuroscience, 10(3):186–198, 2009.

Jay DeYoung, Eric Lehman, Benjamin Nye, Iain J. Marshall, and Byron C. Wallace. Evidence inference 2.0: More data, better models. In BioNLP Workshop atACL, pages 123–132, 2020.

Lingzhong Fan, Hai Li, Junjie Zhuo, Yu Zhang, Jiaojian Wang, Liangfu Chen, Zhengyi Yang, Congying Chu, Sangma Xie, Angela R. Laird, Peter T. Fox, Simon B. Eickhof, Chunshui Yu, and Tianzi Jiang. The Human Brainnetome Atlas: A new brain atlas based on connectional architecture. Cerebral Cortex, 26(8):3508–3526, 2016.

Albert Gatt and Emiel Krahmer. Survey of the state of the art in natural language generation: Core tasks, applications and evaluation. Journal ofArtificial Intelligence Research, 61:65–170, 2018.

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM transactions on information systems, 43(2):1–55, 2025.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12):1–38, 2023.

Nikolaus Kriegeskorte, W. Kyle Simmons, Patrick S. F. Bellgowan, and Chris I. Baker. Circular analysis in systems neuroscience: The dangers of double dipping. Nature Neuroscience, 12(5):535–540, 2009.

Rémi Lebret, David Grangier, and Michael Auli. Neural text generation from structured data with application to the biography domain. In Proceedings of the 2016 conference on empirical methods in natural language processing, pages 1203–1213, 2016.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. Advances in Neural Information Processing Systems, 33:9459–9474, 2020.

Potsawee Manakul, Adian Liusie, and Mark J. F. Gales. Self-CheckGPT: Zero-resource black-box hallucination detection for generative large language models. In EMNLP, pages 9004–9017, 2023.

Scott Marek, Brenden Tervo-Clemmens, Finnegan J Calabro, David F Montez, Benjamin P Kay, Alexander S Hatoum, Meghan Rose Donohue, William Foran, Ryland L Miller, Timothy J Hendrickson, et al. Reproducible brain-wide association studies require thousands of individuals. Nature, 603(7902):654–660, 2022.

Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wentau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. In EMNLP, pages 12076–12100, 2023.

Jonathan D. Power, Kelly A. Barnes, Abraham Z. Snyder, Bradley L. Schlaggar, and Steven E. Petersen. Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion. NeuroImage, 59(3): 2142–2154, 2012.

Ratish Puduppully, Li Dong, and Mirella Lapata. Data-to-text generation with content selection and planning. In Proceedings ofthe AAAI conference on artificial intelligence, volume 33, pages 6908–6915, 2019.

Ehud Reiter and Robert Dale. Building Natural Language Generation Systems. Cambridge University Press, 2000.

Karan Singhal, Shekoofeh Azizi, Tao Tu, S. Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, Perry Payne, Martin Seneviratne, Paul Gamble, Chris Kelly, Abubakr Babiker, Nathanael Schärli, Aakanksha Chowdhery, Philip Mansfield, Dina Demner-Fushman, Blaise Agüera y Arcas, Dale Webster, Greg S. Corrado, Yossi Matias, Katherine Chou, Juraj Gottweis, Nenad Tomasev, Yun Liu, Alvin Rajkomar, Joelle Barral, Christopher Semturs, Alan Karthikesalingam, and Vivek Natarajan. Large language models encode clinica knowledge. Nature, 620(7972):172–180, 2023.

Tao Tu, Shekoofeh Azizi, Danny Driess, Mike Schaekermann, Mohamed Amin, Pi-Chuan Chang, Andrew Carroll, Charles Lau, Ryutaro Tanno, Ira Ktena, et al. Towards generalist biomedical ai. Nejm Ai, 1(3):AIoa2300138, 2024.

David C Van Essen, Stephen M Smith, Deanna M Barch, Timothy EJ Behrens, Essa Yacoub, Kamil Ugurbil, and Wu-Minn HCP Consortium. The wu-minn human connectome project: an overview. Neuroimage, 80:62–79, 2013.

Dave Van Veen, Cara Van Uden, Louis Blankemeier, Jean-Benoit Delbrouck, Asad Aali, Christian Bluethgen, Anuj Pareek, Malgorzata Polacin, Eduardo Pontes Reis, Anna Seehofnerova, Nidhi Rohatgi, Poonam Hosamani, William Collins, Neera Ahuja, Curtis P. Langlotz, Jason Hom, Sergios Gatidis, John Pauly, and Akshay S. Chaudhari. Adapted large language models can outperform medical experts in clinical text summarization. Nature Medicine, 30(4):1134–1142, 2024.

Edward Vul, Christine Harris, Piotr Winkielman, and Harold Pashler. Puzzlingly high correlations in fMRI studies of emotion, personality, and social cognition. Perspectives on Psychological Science, 4(3):274–290, 2009.

Brandon T. Willard and Rémi Louf. Eficient guided generation for large language models. arXiv:2307.09702, 2023.

Sam Wiseman, Stuart M Shieber, and Alexander M Rush. Challenges in data-to-document generation. In Proceedings of the 2017 conference on empirical methods in natural language processing, pages 2253–2263, 2017.

World Health Organization. Obesity: preventing and managing the global epidemic. Technical report, 2000.

Andrew Zalesky, Alex Fornito, and Edward T. Bullmore. Network-based statistic: Identifying diferences in brain networks. NeuroImage, 53(4):1197–1207, 2010.

## A. Claim Ledger and Evaluation Metrics

## A.1 Claim ledger schema

The claim ledger is the single object passed from the pregeneration control logic to the writer. Each entry is a typed claim with the following fields: a claim identifier and type; a canonical text; one or more evidence pointers into named pipeline stages (the source field); the numerical fields the claim reports; an efect direction where applicable; a literaturesupport level; the risk tags attached by the auditor; the allowed language strength after the policy controller’s monotone downgrade; and an explicit forbidden-language list. A claim is instantiated only if every evidence pointer resolves against a pipeline stage. The post-audit entry for the worked example in §3.4 is:

```yaml
claim_id: bmi_absorption_01
source: edgewise_glm_summary.M1_to_M3
numbers:
primary_edges: 10041
bmi_adjusted_edges: 8
collapse_rate: 99.9%
direction: absorbed_after_bmi_adjustment
risk_tags: [group_covariate_circularity]
allowed_strength: cautious
forbidden_language:
- independent obesity effect
- obesity-specific mechanism
```

The renderer emits this claim’s numbers and direction tag directly from the ledger, and the forbidden\_language list is passed to the writer as a hard constraint. This is the operational distinction the protocol turns on: the hybrid template renders LLM-selected slots, whereas claim-locked reporting renders ledger-bound claims whose source, numbers, direction, and allowed strength were fixed before writing.

## A.2 Automatic evaluation metrics

This section specifies the automatic metrics used in the main evaluation. The RCT direction audit is defined separately in Appendix E. For FC, a lexical direction check is retained only as a supplementary diagnostic.

Numerical flag (Num). Report-visible numbers are matched against three support sets: all numeric leaves in the evidence JSON, an explicit derived-metric whitelist, and a small constant whitelist for conventional values such as section numbers and FDR thresholds. Matching uses absolute tolerance 10<sup>−3</sup> or relative tolerance 5%. The derived whitelist includes only quantities used by the renderer, such as FDR-surviving percentages and BMI-absorption collapse percentages. Common section, table, week, and ordinal numbers are filtered before scoring, and arbitrary pairwise ratios are deliberately not whitelisted, trading recall for precision. Num is therefore a conservative commission check rather than a complete mathematical-verification system.

Strong-language count (Strong). Strong counts prose sentences containing a strong assertion term (e.g., “robust”, “causal”, “conclusive”) without a hedge in the same sentence (e.g., “may”, “exploratory”, “hypothesis-generating”). The lexicons are fixed before evaluation. The check can miss paraphrased overclaims and can mis-flag a strong term used in a negated context, so it is treated as a diagnostic lexical measure.

Risk-conditioned hedge use (Hedge). Hedge measures the use of cautionary language for claims flagged by the risk taxonomy. It is evaluated only for risk-flagged claims and is therefore interpreted as a measure of whether predefined caution constraints reach the report surface, rather than as a general preference for more hedging.

FC lexical direction check. For FC only, we additionally use a rule-based lexical check that compares direction wording against the ledger-stored direction field. Because the check is brittle under negation and comparator phrasing, it is treated only as a supplementary diagnostic and is not used as the primary direction-correctness measure.

## B. Compared Control Configurations

All methods receive the same structured statistical evidence for a given input. They difer in where control is applied between the evidence and the final report, and therefore in which reporting decisions remain under LLM control. Table A1 summarizes these diferences.

Free-form. The LLM directly generates the report from the evidence record with only a general grounding instruction. Claim selection, numerical realization, direction, language strength, and prose remain under LLM control.

Prompt-only. The LLM receives the same evidence together with explicit faithfulness instructions intended to discourage unsupported numbers, direction errors, and overstatement. These constraints are expressed only through prompting; the model still determines the reportable claims and their realization.

Structured output. The output structure is predefined, but the LLM still selects the claims and generates the statistical content within that structure. This configuration therefore constrains format rather than evidence-bearing content.

Retrieval. The LLM is provided with source-identified evidence blocks from the same record and is required to ground its report in those blocks. The evidence is more explicitly localized, but claim selection and verbalization remain under LLM control.

Post-hoc verifier. A verifier revises an initially generated report using the same evidence record, correcting detected numerical, directional, or interpretive inconsistencies. Control is therefore applied after the report has already been generated rather than before claim selection.

Hybrid template. The LLM selects the values and claims that populate predefined slots, after which those slots are rendered deterministically. This provides slot-level control, but the evidence-bearing content placed into the slots remains LLM-selected.

Table A1. Control configurations of the compared methods. All methods receive the same structured statistical evidence. They difer in where control is applied and in which reporting decisions remain under LLM control.
<table><tr><td>Method</td><td>Control level</td><td>What is controlled</td><td>Remaining LLM decisions</td></tr><tr><td>Free-form</td><td>Text</td><td>General grounding instruction</td><td>Claims, numbers, direction, strength, wording</td></tr><tr><td>Prompt-only</td><td>Text</td><td>Explicit faithfulness instructions</td><td>Claims, numbers, direction, strength, wording</td></tr><tr><td>Structured</td><td>Text</td><td>Output structure</td><td>Claims, numbers, direction, strength, wording</td></tr><tr><td>Retrieval</td><td>Text</td><td>Evidence localization</td><td>Claims, numbers, direction, strength, wording</td></tr><tr><td>Post-hoc verifier</td><td>Text</td><td>Post-generation revision</td><td>Initial claims and realization</td></tr><tr><td>Hybrid template</td><td>Slot</td><td>Deterministic slot rendering</td><td>Slot content and claim selection</td></tr><tr><td>Claim-locked</td><td>Claim</td><td>Evidence-bound claims before writing</td><td>Connective wording only</td></tr></table>

Claim-locked. Reportable claims are bound to their evidence source, numerical fields, direction, and permitted language strength before prose generation. The deterministic renderer realizes the locked content, while the LLM is restricted to connective wording.

## B.1 Prompt configuration

Each method uses a fixed prompt template that implements its corresponding control condition. Free-form generation receives the statistical evidence record directly; prompt-only adds explicit faithfulness constraints; structured output specifies an output schema; retrieval-grounded generation requires evidence-block attribution; the post-hoc verifier uses a second verification-and-revision pass; the hybrid template asks the LLM to populate predefined slots that are subsequently rendered deterministically; and claim-locked reporting provides the writer with the active claim ledger and policy constraints while numerical content is rendered deterministically.

The same template is used across records, seeds, and writer providers within each condition. Exact executable prompt templates, including dynamically injected fields and JSON schemas, are released with the code.

## C. Additional Experimental Analyses

## C.1 Component analysis

We evaluate three claim-locked configurations under the same records, providers, and seeds as the main comparison. As shown in Table A2, Builder only provides the LLM with the claim-builder output while leaving numerical realization to the LLM. Builder + renderer additionally uses the deterministic renderer while disabling the risk auditor and policy controller, isolating the contribution of deterministic realization. Full claim-locked activates the complete pipeline, including the risk auditor and policy controller.

Adding the deterministic renderer to the builder-only configuration increases reproducibility from 85.0% to 98.0% for fMRI and from 90.3% to 100.0% for RCT. Activating the risk auditor and policy controller leaves reproducibility essentially unchanged for fMRI (98.0% → 98.5%) while reducing Strong from 1.40 to 0.75.

Paired-bootstrap uncertainty. For the central Claimlocked-versus-Hybrid comparison, we use 20,000 paired bootstrap resamples over matched units, preserving the main-table structure rather than treating repeated generation cells as independent. The reproducibility diferences are +37.4 points for fMRI (95% CI [+15.1, +59.7]) and +20.5 points for RCT (95% CI [+17.8, +23.1]). The paired fMRI Strong diference (Claim-locked minus hybrid) is −1.50 (95% CI [−2.05, −1.00]).

## D. Risk-Tag Instantiations

The risk auditor uses the same interface across domains, while the concrete tags are grounded in domain-specific evidence. Table A3 organizes the two settings by five shared control targets: numerical support, direction preservation, entity or source support, inferential scope, and interpretive strength. The targets remain unchanged across domains; only the evidence used to instantiate them difers.

## E. Human Audits

fMRI governance audit. Four reviewers independently annotated matched fMRI report snippets from free-form, hybrid template, and claim-locked outputs without seeing method identity. They counted two types of violations: unhedged strong language and unsupported content. The former corresponds to the automatic Strong measure. Unsupported content is included as a supplementary comparison: the automatic rule flags anatomical phrases absent from both the atlas-derived ROI dictionary and the evidence record, whereas human reviewers judge unsupported content more broadly. Values in Table A4 are average violation counts per report; lower is better.

The human judgments follow the corresponding automatic ordering. Claim-locked has the lowest average count for both violation types; the hybrid template is intermediate for strong language but highest for unsupported content, while free-form is highest for strong language. This provides a reader-facing check that the qualitative trends identified by the automatic measures are also visible under blinded human judgment.

RCT direction audit. A blinded reviewer labeled all 840 generated reports from the 30-record RCT direction-audit subset. Each report was classified as preserved, inverted, or unresolved relative to the gold direction. Unresolved denotes reports for which no reliable direction can be determined from the generated text, including reports that do not state a clear direction. The inversion rate is computed as Inv./(Pres.+Inv.), with unresolved reports reported separately and excluded from the denominator. The per-method results are reported in Table 5.

A second reviewer independently labeled a stratified 50- report subset comprising 19 preserved, 15 inverted, and 16 unresolved reports. Raw agreement was 98.0% (49/50 reports), with Cohen’s $\kappa = 0 . 9 7 0$ ; the single disagreement was between preserved and inverted.

## F. RCT Numerical-Flag Calibration

Raw Num is a high-recall numerical flag and does not distinguish unsupported statistical results from benign mismatches.

Table A2. Component analysis. Reproducibility is reported in percent; Strong is the unhedged-strong count. Bold marks the full claim-locked configuration
<table><tr><td>Configuration</td><td>fMRI Repro. ↑ fMRI Strong ↓ RCT Repro. ↑ RCT Strong ↓</td><td></td><td></td><td></td></tr><tr><td>Hybrid</td><td>61.1</td><td>2.25</td><td>79.5</td><td>0.040</td></tr><tr><td>Builder only</td><td>85.0</td><td>1.60</td><td>90.3</td><td>0.000</td></tr><tr><td>Builder + renderer</td><td>98.0</td><td>1.40</td><td>100.0</td><td>0.015</td></tr><tr><td>Full claim-locked</td><td>98.5</td><td>0.75</td><td>100.0</td><td>0.010</td></tr></table>

Table A3. Shared control targets and their domain-specific instantiations. The same auditor–policy interface is used in both settings, while each target is grounded in domain-specific evidence.
<table><tr><td>Target</td><td>FC evidence</td><td>RCT evidence</td><td>Policy action</td></tr><tr><td>Numerical support</td><td>Statistical results and derived quantities</td><td>Trial outcomes and reported statistics</td><td>Allow only evidence-supported numbers</td></tr><tr><td>Direction preservation</td><td>Ledger-stored effect direction</td><td>Gold direction label</td><td>Preserve the resolved direction</td></tr><tr><td>Entity/source support</td><td></td><td></td><td>Anatomical and behavioral entities Trial entities and source information Exclude unsupported entities or sources</td></tr><tr><td>Inferential scope</td><td>Covariate dependence and exploratory findings</td><td>Intervention, outcome, and population</td><td>Restrict claim scope when required</td></tr><tr><td>Interpretive strength</td><td>Stability and evidential support</td><td>Support available in the evidence span</td><td>Limit claims to permitted language strength</td></tr></table>

Table A4. Blinded fMRI governance audit. Values are average violation counts per report; lower is better. The rule-based counts are shown alongside the four blinded reviewers and their mean.
<table><tr><td>Assessment</td><td>Free-form</td><td>Hybrid</td><td>Claim-locked</td></tr><tr><td colspan="4">Unhedged strong language</td></tr><tr><td>Automatic rule</td><td>2.50</td><td>1.80</td><td>0.60</td></tr><tr><td>Reviewer 1</td><td>2.60</td><td>1.90</td><td>0.60</td></tr><tr><td>Reviewer 2</td><td>2.40</td><td>1.70</td><td>0.50</td></tr><tr><td>Reviewer 3</td><td>2.70</td><td>2.00</td><td>0.70</td></tr><tr><td>Reviewer 4</td><td>2.50</td><td>1.80</td><td>0.60</td></tr><tr><td>Human mean</td><td>2.55</td><td>1.85</td><td>0.60</td></tr><tr><td colspan="4">Unsupported content</td></tr><tr><td>Automatic rule</td><td>1.60</td><td>2.30</td><td>0.50</td></tr><tr><td>Reviewer 1</td><td>1.80</td><td>2.50</td><td>0.60</td></tr><tr><td>Reviewer 2</td><td>1.50</td><td>2.10</td><td>0.40</td></tr><tr><td>Reviewer 3</td><td>1.70</td><td>2.40</td><td>0.60</td></tr><tr><td>Reviewer 4</td><td>1.60</td><td>2.30</td><td>0.50</td></tr><tr><td>Human mean</td><td>1.65</td><td>2.33</td><td>0.53</td></tr></table>

raw flags but produces unsupported statistical results in 2.5% of outputs, showing that deterministic rendering does not prevent numerical fabrication when the LLM still selects the slot content. Free-form generation has the highest harmfulfabrication rate at 5.9%.

We therefore manually inspect the flagged outputs for claimlocked, hybrid template, and free-form generation. Each flag is assigned to one of three categories: a regular-expression false positive (Regex FP), a benign metadata or support-window mismatch, or a harmful statistical-result fabrication in which a report-visible statistical value is absent from the evidence record. Table A5 summarizes the calibration.

Table A5. Manual calibration of the RCT numerical flags underlying Table 4. Benign denotes metadata or support-window mismatches; Harmful denotes unsupported statistical results.
<table><tr><td>Method</td><td>Raw flags</td><td>Regex FP</td><td>Benign</td><td>Harmful</td></tr><tr><td>Free-form</td><td>213</td><td>17</td><td>149</td><td>47 (5.9%)</td></tr><tr><td>Hybrid template</td><td>89</td><td>14</td><td>55</td><td>20 (2.5%)</td></tr><tr><td>Claim-locked</td><td>181</td><td>1</td><td>180</td><td>0 (0.0%)</td></tr></table>

The calibration changes the interpretation of the raw Num values. Claim-locked has the largest raw flag count, but all of its flags are benign or false positives, with no harmful statistical-result fabrication. The hybrid template has fewer