# RubricAtlas: Mining Type-Level Rubrics from Expert Reports to Evaluate Deep Research Agents

Delin Wang<sup>1,3</sup>, Sinuo Wang<sup>1,4</sup>, Rui Sun<sup>1</sup>, Beidi Luan<sup>1</sup>, Qiming Shi<sup>1,5</sup>, Jing Li<sup>1</sup>, Daxin Jiang<sup>1</sup>, Zuo Bai<sup>1,2,†</sup>

<sup>1</sup>StepFun <sup>2</sup>FinStep <sup>3</sup>Renmin University of China <sup>4</sup>University of Adelaide <sup>5</sup>Zhejiang University

<sup>†</sup>Corresponding author: baizuo@stepfun.com

## Abstract

Deep research agents conduct multi-step Web searches and synthesise retrieved information into analytical reports. Evaluating these outputs requires checking which analyses they deliver, beyond the presence of relevant facts. Query-specific binary rubrics make individual requirements explicit, but task-level checklists may leave the overall analytical structure underexamined. We introduce RubricAtlas, which derives type-level analytical requirements from expert reports on diferent objects within the same research type and organises detailed checks around recurring analytical questions. The requirements give related tasks a shared standard for what to analyse, without prescribing their answers. Our framework links binary checkpoints to report passages and evaluates completeness separately from argument validity. Built from 3,079 Chinese financial research reports, RubricAtlas covers 99 research types. Referencereport selection experiments show that individual reports cover only about half of their type’s analytical questions, and that selecting requirements from diferent reports changes evaluation scores and agent rankings. Across ten agents, two judge pipelines achieve checkpoint agreement of � = 0.71 and agent-ranking agreement of Kendall � = 0.87. Controlled deletions reduce scores on the affected analyses, while padding produces no detectable increase. These findings support type-level rubrics for comparing the analytical completeness of Web deep research outputs, with individual decisions traceable to report evidence.

## CCS Concepts

• Information systems → Evaluation of retrieval results; Test collections; Web searching and information discovery.

## Keywords

deep research agents, agentic search, rubric-based evaluation, LLMas-a-judge, Web agents, expert reports

## 1 Introduction

Deep research agents carry out multi-step Web searches, read retrieved pages and synthesise findings into analytical reports [4, 7, 23]. Unlike short-answer Web tasks [21, 35], these reports often require more than factual retrieval, including comparisons, explanations and examination of the conditions behind their conclusions. A report on an oil and gas producer, for example, may list production, reserves and costs without comparing the firm with peers or explaining whether its advantage can persist. Including relevant facts is not the same as completing the relevant analysis.

A common approach to evaluating deep research outputs is to use query-specific binary rubrics: requirements written for an individual task and judged one by one as met or unmet [1, 17, 29]. These rubrics make individual requirements explicit. However, an evaluation centred on checklist items can foreground task-specific details while leaving the report’s overall analytical structure underexamined. The issue is not whether decisions are binary, but whether they reveal which analyses the details collectively support and which remain incomplete. We therefore organise evaluation around analytical questions, with detailed checks specifying what each analysis should deliver. To ground this organisation beyond the choices made for one query or reference report, we derive recurring requirements from expert reports on diferent research objects.

Expert reports provide examples of how professionals analyse diferent objects. We define a research type as a research-question family within an industry, such as studies of oil and gas producers’ operations and value realisation. Within a type, reports may attribute advantages to diferent resources or distribution networks, but repeatedly ask: which verifiable capabilities distinguish the firm from its peers, and under what conditions can that advantage persist? By combining these reports, we identify recurring requirements: name the capability, compare it with named peers using comparable metrics, and examine what could sustain or erode the advantage. These requirements describe what to analyse, not what to conclude. They reflect the source reports’ professional practice, rather than a universal checklist for every request.

Our framework combines two designs (Figure 1). Type-level induction derives reusable analytical requirements across research objects. A question-centred hierarchy organises their evaluation into questions, required results (Targets) and binary checkpoints. For example, a question asks which capabilities give the firm an advantage; one required result is a peer comparison; its checkpoints ask whether the report names peers and provides comparable metrics. The details are thus evaluated as parts of an analysis, not as isolated facts. Passages are assigned to questions, checkpoint decisions cite the corresponding report blocks, and scores are aggregated within the hierarchy. This produces both a report score and a profile of completed and missing analyses. Completeness measures whether the required analytical content is present, not whether its facts or reasoning are correct. Argument validity is assessed separately.

We construct RubricAtlas from 3,079 Chinese financial research reports issued by 64 institutions, yielding 99 research types, 921 questions and 8,160 checkpoints. Our reference-report selection experiments examine how the requirements used to evaluate these outputs depend on the expert reports behind them. We measure the fraction of a type’s questions covered by each report and average within types; the median across types is 54%. Evaluating only questions supported by each task’s source report raises completeness scores by 10.5 points and gives Kendall � = 0.60 with the full-atlas agent ranking. Thus, selecting one report’s analyses changes both the coverage of the evaluation and its outcomes. These experiments restrict the frozen atlas; they do not reconstruct query-specific rubrics.

![](images/0f8c51f32929e6b58c9b29541b8030957150639c2a1c179fbae26cfb0405f4d6.jpg)  
Figure 1: Construction and evaluation. (a) Questions from expert reports on diferent firms are aligned and abstracted into type-level requirements. (b) Aligned report passages feed separate completeness and validity evaluations, yielding scores and analytical profiles. (c) An oil-and-gas question has three Targets; the peer-comparison Target is expanded into checkpoints. A named-peer comparison passes; an industry average triggers the negative criterion. Source questions are paraphrased; outcomes are schematic.

Table 1: Source aggregation in related evaluations. All RubricAtlas counts are medians.
<table><tr><td>Benchmark</td><td>Construction / reference basis</td><td>Scope</td></tr><tr><td>DeepResearch Bench [4]</td><td>LLM-generated, with one reference report</td><td>task</td></tr><tr><td>DRB ⅡI [17]</td><td>extracted from one expert report</td><td>task</td></tr><tr><td>ResearchRubrics [29]</td><td>expert-written for each prompt</td><td>prompt</td></tr><tr><td>FinResearchBench [30]</td><td>report-level logic trees; shared structural metrics and LLM judgments</td><td>shared criteria</td></tr><tr><td>FinResearchBench II [19]</td><td>multi-system reports for one query; consistency and distinguishability filters</td><td>query</td></tr><tr><td>RubricAtlas (ours)</td><td>30 expert reports per type; 14 supporting reports and 9 institutions per question</td><td>research type</td></tr></table>

Our contributions are threefold:

• Type-level induction of analytical requirements. We derive recurring analytical requirements across research objects and construct RubricAtlas for financial research, with source provenance, tasks and evaluation artifacts.

• Question-centred, evidence-linked report evaluation. We organise detailed checks around analytical questions and retain question-level profiles alongside report scores. Two end-to-end judge pipelines achieve checkpoint agree ment of � = 0.71 and agent-ranking agreement of Kendall � = 0.87. Controlled deletions lower scores on targeted questions; padding produces no detectable increase. A firstauthor audit examines sampled source mappings and checkpoint decisions.

• Reference sensitivity and Web-agent profiles. We measure reference-selection efects on coverage, scores and rankings, and compare ten Web research agents on consensus and diferentiating analyses. We examine associations with browsing, length and source-report exposure. Generic validity rules remain separate diagnostics because crossjudge agreement is limited.

## 2 Related Work and Design Requirements

Four requirements guide the framework: professional grounding, question-centred organisation and type-level reuse, attributable decisions, and reproducible comparisons. Table 1 compares the source aggregation used by closely related evaluations.

D1 Grounding. A standard should represent recurring expert analyses, not one document’s choices. Existing rubrics are written per instance [1, 29], extracted from one reference [17], or mined from one survey per query [40].

Multi-reference methods aggregate content for one input: Pyramid weights units by reference frequency [24], and TREC evaluates vital and okay nuggets [32]. Nuggets can be weighted by assessor agreement [18] or extracted and evaluated using LLMs [20, 26]. We abstract analyses across diferent objects within a type.

D2 Organised and reusable. Requirements should be organised around analytical questions, so that individual decisions reveal which analyses are complete or missing. This organisation should be reusable across tasks within a research type. Checklists turn holistic judgments into binary decisions [3, 16, 36]; instance-specific evaluators use rubrics or skill sets [15, 39], and rubrics also serve as rewards [8, 28]. DEER and FinDeepResearch provide top-down task-level templates [9, 43]. We retain binary checks but organise them under questions derived bottom-up from expert reports across research objects.

D3 Attributable. Judgments should identify the unmet requirement or supporting evidence. Factuality evaluation decomposes text into atomic claims [22]; Mind2Web 2 verifies task-specific rubric trees against cited pages [7], and ReportLogic evaluates reasoning [41]. Our checkpoints cite report blocks retrieved verbatim by the program, not reconstructed by the judge.

D4 Reproducible. Comparisons should remain stable across judges. LLM judges exhibit position, verbosity and self-preference efects [5, 25, 34, 42] and criterion confusion [11]. IR studies examine assessor disagreement, incomplete judgments, topic sampling and metric discrimination [27, 31, 33, 44]. Reproducibility difers from measurement validity [12].

Evaluation of Web research. Web benchmarks assess short answers [21, 35] and deep research reports through rubrics or references [4, 7, 17]. Financial benchmarks cover numerical question answering [2], research reports [13, 14, 43] and search [10]. We focus on the analytical completeness of reports produced through Web search and synthesis.

FinResearchBench [30] combines report-level logic-tree metrics with LLM judgments. FinResearchBench II [19] generates rubrics from multiple systems’ reports for one query, then filters for judge consistency and distinguishability. Structured evaluation is therefore not unique to our approach: existing methods also use rubric trees, logic trees and templates. Our distinction is how the analytical structure is grounded and reused. Expert reports on diferent objects supply type-level questions and required results, under which evidence-linked checks are organised. The consensus tier denotes analyses in a majority of source reports, not agreement among judges.

## 3 Evaluation Framework

The framework separates deriving an analytical standard from applying it. Type-level construction identifies recurring analyses across research objects; question-centred evaluation determines what each report delivers. Here, analytical structure means the organisation of questions and required results, not the full logical coherence of a report’s reasoning. Evaluation aligns report passages to this structure and judges completeness and argument validity independently.

Research types. A research-question family is a recurring kind of analysis. Combining it with an industry defines a research type, such as operations and value realisation of oil and gas producers. A task specifies a firm, industry or policy within that type.

Analytical requirements. The rubric is a tree: dimension → question → required result → checkpoint. Questions identify analyses, required results specify their expected outputs, and checkpoints assess the content needed to deliver those outputs. Each checkpoint decision therefore remains attached to an analytical requirement. A canonical question represents an analysis found in the type’s reports. Its support records the reports and institutions addressing it; its tier is consensus if a majority of reports address it and diferentiating otherwise. A required result (Target) specifies what an answer must deliver without naming a company or date. Binary checkpoints state acceptance and non-acceptance conditions. A Target may have a negative criterion identifying a substitute that misses the intended analysis, such as industry averages instead of named-peer comparisons. Requirements are constructed once per type, not per task.

Evidence alignment and completeness. The judge selects passage IDs; the program retrieves original text, preventing insertion of new material. Target evaluation checks content in these passages, citing blocks for yes and explaining no. Missed evidence during alignment can lower scores. Factual and inferential correctness are not assessed; Target requirements provide context, not extra scoring criteria.

In the current implementation, a Target’s completeness is its checkpoint pass rate; a triggered negative criterion sets it to 0. Question completeness T (0–1) is the mean of its Targets. Report completeness T (0–100) is the weighted mean of question completeness (diferentiating 20, consensus 10) ×100. Averaging within Targets and questions prevents extra checkpoints from increasing an analysis’s report-level weight. Diferentiating questions were model-screened and author-verified for analytical depth. The higher weight emphasises this depth and was fixed before examining agent evaluation results; alternatives appear in §7. Question, Target and checkpoint profiles locate incomplete analyses alongside report scores.

Argument validity. This layer concerns evidence, inference and quantitative reasoning. Flaw definitions can be shared, with question-specific conditions. General evaluation tests generic rules on actual claims independently of Targets and scores (§7.4).

A worked example. Figure 1 illustrates a consensus question for oil and gas producers: which verifiable capabilities give the firm an advantage over comparable firms, and can that advantage persist? It is supported by 14 of 21 reports from 13 institutions, each on a diferent firm. Source questions concern a gas block’s resource advantage or a sales network’s contribution to stability. Its 3 Targets and 9 checkpoints require capabilities with checkable metrics, named-peer comparisons on like-for-like metrics, and conditions sustaining or eroding the advantage. The second Target’s negative criterion excludes industry averages in place of named peers. A report can thus receive credit for naming a capability but not for a peer comparison it never makes. Providing capability metrics does not substitute for comparing peers or examining persistence conditions: the separate Targets preserve these omissions within the same analytical question.

Output. We evaluate completeness end to end (§6–7); the separate validity profiles diagnose one generic-rule implementation (§7.4).

## 4 Constructing RubricAtlas for Financial Research

RubricAtlas derives a shared analytical structure through report classification, question alignment and Target abstraction. Fixed AI workflows and prompts govern aggregation and applicability screening. Source questions determine the analyses to cover; answering passages supply each analysis’s content requirements.

Corpus and research types. The corpus comprises 3,079 Chi nese financial research reports on the Chinese market, issued by 64 institutions from January to September 2026. It covers 31 industries and eleven genres, including company coverage, earnings reviews, industry deep dives and strategy reports.

An LLM decomposes reports into dimension → question → conclusion trees. Reports are grouped first by genre × industry (331 cells), then by research-question family × industry. The 13 fami lies include firm operations and value realisation, industry cycle tracking and policy transmission. Of 106 candidates, 3 with heterogeneous branches were deferred and 4 excluded by a predefined discrimination criterion. The 99 retained pools contain a median of 30 reports (10–60; Appendix E).

Question alignment. Within each type, an LLM maps source questions to canonical questions, background or out-of-scope content; a second LLM checks the mappings. Records preserve supporting reports and source questions. Questions below 3 supporting reports are removed unless judged to have distinct analytic value.

Targets and checkpoints. For each canonical question, an swering passages are decomposed into data used → reasoning → conclusion, abstracted into company- and date-independent Targets, and divided into checkpoints. We add negative criteria for plausible substitutes. Before scoring, an AI reviewer checked each criterion for one verifiable requirement, clear wording and no adjacent redundancy [1]; authors reviewed all flagged items. Scores use this frozen atlas, unchanged by the later author audit.

Applicability screening. The workflow excludes requirements outside the industry and research-question family, and those specific to an individual object. It also checks consistency between retained requirements and the requests’ analytical scope. These filters produce type-level standards rather than report-content unions. Requirements are frozen before scoring; request checks match tasks to these standards without regenerating rubrics or adapting them to agent outputs.

Scale and support. RubricAtlas contains 99 trees, 921 questions (503 consensus, 418 diferentiating), 2,784 Targets, 8,160 checkpoints and 2,520 negative criteria. Median question support is 14 reports and 9 institutions; 94% of questions have at least three institutions and 79% at least five. The largest institution’s median support share is 25%.

Median pool coverage is 68% for consensus and 28% for diferentiating questions. The latter still have median support from 8 reports and 6 institutions (Table 2), representing recurring analyses rather than isolated contributions.

Table 2: Multi-source question support. Report and institution counts are medians per question; percentages concern diferentiating questions.
<table><tr><td>Question tier</td><td>Questions</td><td>Reports</td><td>Institutions</td></tr><tr><td>Consensus</td><td>503</td><td>21</td><td>11</td></tr><tr><td>Differentiating</td><td>418</td><td>8</td><td>6</td></tr><tr><td colspan="4">Multi-source support among differentiating questions</td></tr><tr><td>At least three supporting reports</td><td></td><td></td><td>96%</td></tr><tr><td>At least three supporting institutions</td><td></td><td></td><td>90%</td></tr></table>

Tasks. A seeded hash selects one report per type and converts its topic into a conclusion-free request, yielding 99 tasks. Checks cover research type, object scope, analytical scope and readability, with recorded adjustments (Appendix E). Requests are matched to the type-level requirements, not used to regenerate them.

Resource contents and access. The resource includes trees with provenance, rules, tasks, prompts, checkpoint judgments, scoring and analysis code, and a datasheet [6]. The materials will be made publicly available after the review process. Re-scoring uses judgments and code; independent re-judging and source verification also need report text and source materials. A verbatim-overlap check sampled 800 criteria, extracted 3,881 spans of 20 normalised characters and searched 3,228 source texts: no matches, with a positive control detected.

## 5 Experimental Setup

Agents and harness. The ten LLMs in Table 4 run as deep research agents in a fixed ReAct harness [38]. WebSearch returns Google results via Serper; WebFetch reads pages with Jina and extracts goalrelevant content with gpt-5. The prompt sets information cutof 2026-10-09, requires citable sources and prohibits other institutions’ research conclusions as evidence. Each agent runs once per task with provider-default decoding, up to 1,000 turns and 128k output tokens per turn. Tasks and reports are Chinese.

Evaluation sets. One run failed, leaving 989 reports. Agent and browsing comparisons use 98 common tasks (980 reports), covering 912 of 921 atlas questions. Checkpoint agreement uses available paired judgments. Agent scores equally average common-task report scores; tier scores are averaged after within-report computation, so global tier counts cannot reconstruct totals.

Judges and protocol. Judge A is DeepSeek-V4.1-Flash; B is GPT-6-luna. Calls use temperature 0 and are grouped by dimension, with decisions assigned to questions. Evidence alignment selects block IDs for programmatic text retrieval. Target evaluation returns checkpoint and negative-criterion decisions, citations for each yes and reasons for each no. General evaluation runs separately. Neither judge accesses the Web.

The API parameter reasoning\_effort is none for alignment, medium/low for A’s Target/General calls and high/high for B’s. Model-specific labels are not a shared budget scale: we compare configured pipelines, not matched-budget model substitution. Settings were fixed in pilots before finalisation; token use and alternatives appear in Appendix D. Failed jobs are listed separately, not scored zero.

A is also an evaluated backbone; B shares the tree builders’ GPT family. Both judges expose sensitivity to these overlaps but cannot rule out family bias. All agents share the gpt-5 page extractor, whose efect is not isolated. Task resampling quantifies sampling uncertainty; score intervals add a shared repeat-run variance proxy (§6.3), not independently estimated B variance.

## 6 Assessing the Resource and Evaluation Protocol

We assess two aspects of the approach: which analytical requirements the resource represents, and how consistently evaluation applies them. Reference-report selection analyses examine requirement coverage and dependence on particular expert reports; the author audit examines sampled mappings and decisions; cross-judge comparisons and controlled content changes assess the evaluation protocol.

## 6.1 Author audit

The first author, a finance graduate student, annotated blank forms without AI assistance or access to model judgments: 300 checkpoint decisions and 297 source mappings for 50 canonical questions. Agreement with A/B is 74.3%/71.7% $\left( \kappa = 0 . 4 9 / 0 . 4 3 \right)$ . Mappings are 77.8% faithful, 15.2% partial and 7.1% not faithful. These checks concern checkpoint application and question grouping, not every Target abstraction or task applicability.

The sample overrepresents judge disagreements; evidence is the union of both judges’ aligned passages. Agreement cannot be directly compared with the full-corpus judge–judge rate. All 300 labels are retained as final judgments. This domain-informed author audit is not independent expert validation; sampling, confusion matrices and exploratory preferences appear in Appendix I.

## 6.2 Reference-report selection and analytical coverage

How does reference selection change requirement coverage and agent comparisons? Using the full atlas as an internal reference, we retain questions supported by selected reports and re-score frozen main-run judgments (§5). Reports and checkpoint decisions stay fixed; rubrics are not re-extracted (Appendix G). These comparisons measure reference sensitivity, not which requirements best fit each task.

Coverage. Averaging reports within types, then taking the median across types gives 54% coverage (type means 14–83%); tasksource reports cover 57%. The gap is not solely expert disagreement: genre medians are 73% for company coverage and 33% for retrospective reviews; the shortest and longest thirds cover 50% and 63%.

Removing the measured report from support and dropping questions below 3 reports leaves median coverage at 53%. The other half covers 56% of questions selected from one random half of a pool; the within-half threshold retains more common questions. A single report thus represents only part of a type’s recurring analyses, even excluding its contribution to question selection.

Score shifts. Source-supported subsets raise mean completeness by 10.5 points (task-level 95% confidence interval (CI) [7.1, 12.1]), with increases on 77% of tasks. Size-matched random subsets give

Table 3: �-report question subsets per type (200 draws; smaller pools use all reports). Cov.: mean union coverage over types/draws, not median per-report coverage. �: mean correlation with the full ranking; 5th: fifth percentile. Rand.: count-matched random questions, without tier matching. Gain: mean score increase (points). Top-1: fraction changing the overall leader.
<table><tr><td>k</td><td> $\operatorname { C o v . }$ </td><td>τ</td><td>τ (5th)</td><td>τ rand.</td><td>Gain</td><td>Gain rand.</td><td>Top-1 (rand.)</td></tr><tr><td>1</td><td>51%</td><td>0.69</td><td>0.51</td><td>0.78</td><td>4.6</td><td>0.7</td><td>24% (7%)</td></tr><tr><td>3</td><td>79%</td><td>0.79</td><td>0.64</td><td>0.89</td><td>2.0</td><td>0.3</td><td>0% (0%)</td></tr><tr><td>10</td><td>96%</td><td>0.95</td><td>0.87</td><td>0.97</td><td>0.3</td><td>0.0</td><td>0% (0%)</td></tr></table>

(a) Coverage  
(b) Ranking agreement  
![](images/8bc40eb39e6a33a6f7549eefff1d93323a86ceaec951c6eef5a84b93ea0e3248.jpg)

![](images/30890eb4bdaa417251640ff4cf7a49a220cd9d2d75492870dc7ac52a42d3810e.jpg)  
Reports per type (k, discrete conditions)  
Figure 2: Full-atlas coverage by � reports per type (a) and ranking Kendall � (b), with size-matched random questions. Shading spans the report draws’ 5th percentile to mean. Equally spaced ticks denote discrete � conditions; smaller pools use all reports.

1.2 (200 draws; maximum 4.5). Source subsets contain 71% consensus questions, versus 55% overall; consensus scores are higher. Composition matching gives 5.6 (maximum 8.6 across 200 draws), still 4.9 below source subsets, which exceed every matched draw. The remainder is consistent with tasks eliciting source-emphasised analyses, but object-specific applicability is not isolated.

Ranking sensitivity. Single-report subsets have lower ranking agreement and more leader changes than size-matched random questions (Table 3). Task-source subsets give 0.60, versus 0.81 for size-matched subsets and mean 0.74 (5th percentile 0.56) for composition-matched subsets. Size matching alone does not isolate source dependence: composition explains part of the ranking change. Source subsets change the task-level leader on 50% of tasks, versus 45% for size-matched random subsets (Appendix H). Both reductions afect task-level recommendations, motivating aggregate comparisons.

Aggregation across reports. Three and ten reports cover 79% and 96% of the atlas (Figure 2), with ranking correlations 0.79 and 0.95, near size-matched controls (0.89, 0.97). Single-report small pools (10–20) give � = 0.59 and 45% coverage, but sensitivity is nonmonotonic (21–39: 0.79; 40–60: 0.65). About ten reports recover much of the full pool’s coverage and ranking: an internal planning reference, not an external suficiency threshold.

Sensitivity to source reports. Disjoint random half-pools give � = 0.84 across 200 splits (0.76 with three supporting reports within each half), always with the same leader. Task resampling gives 0.79 against the full ranking, a similar scale of variation.

Removing task-source support drops 14 of 912 questions across 98 common tasks below threshold; re-scoring gives � = 0.96, with model score changes at most 0.27 points. The aggregate evaluation is therefore not driven by the single report used to construct each task. This supports type-level standards; new-task applicability is discussed in §8.

## 6.3 Cross-judge consistency and repeat runs

B independently re-evaluates all 989 reports, including alignment. The 81,469 paired checkpoints agree at 85.4%, with Cohen’s � = 0.71 (95% CI [0.70, 0.72], report-clustered) and Jaccard overlap 0.75 for yes decisions. Question-level T correlation is 0.83 over 9,196 questions.

B gives fewer yes decisions (48% versus 56%) and scores reports 8.0 points lower on average, but rankings agree at � = 0.87 (95% CI [0.56, 0.96], task resamples). Both judges have the same first two agents, top-four set and bottom-three order (Table 4, Figure 3a). Of A’s 36 pairs separated by over one point, 1 reverses under B; none of 28 separated by over two points reverses. None of B’s 34 pairs separated by over one point reverses under A. Judge choice shifts absolute levels more than clearly separated agents’ ordering. Negative criteria rarely trigger (A 262, B 151 of 25,164); agreement is 99.0% (� = 0.39).

A DeepSeek repeat run on 100 stratified reports (10 tasks × ten agents) uses the same model and efort but condensed negativecriterion wording, correlating at 0.96 with the main run. Its standard deviation (SD) proxy is 4.1 points per report and 0.42 for a 98- task mean. Both judges use this estimate, mixing run and wording variation. Alternative settings correlate highly, but nominal high efort need not increase tokens (Appendix D).

## 6.4 Response to deletion and padding

We perturb 30 reports (ten tasks × three agents) and re-evaluate each whole report. Deleting all blocks answering one question lowers its T from 0.64 to 0.21 under A and from 0.57 to 0.19 under B. Median relative reductions are 71% and 83%, with decreases on 20/27 and 22/27 target questions<sup>1</sup>. Questions sharing no deleted block change by 0.033 and 0.028 on average, close to unchanged re-evaluation (0.025 and 0.029). The response is concentrated on questions whose content was removed, linking the score change to an identifiable missing analysis rather than only a reduction in report length; Appendix D also reports changes on questions sharing deleted blocks.

Adding 40% irrelevant text from another research family or dupli cating paragraphs gives no detectable net increase. A’s unchanged re-evaluation itself shifts scores by −2.5 points. Relative to each judge’s unchanged re-evaluation, irrelevant/duplicated efects are +1.1 [−0.9, 3.2] and +0.5 [−1.3, 2.5] under A, +0.8 [−1.4, 3.0] and 0.0 [−2.1, 2.1] under B (Figure 3c). With 30 reports, only efects above about 3 points are excluded. A’s observed length slope implies 3.1 points, so modest rewards remain compatible with the intervals.

## 7 Evaluating Deep Research Agents

We compare the final reports of Web research agents through completeness totals and analytical tier profiles. We then examine associations with browsing, report length and exposure to source reports. Generic-rule profiles remain separate from completeness.

## 7.1 Main results

Leading agents and unresolved margins. Claude leads under both judges (92%, 84% first-place resamples), followed by GLM. Their gaps of 2.0 and 1.0 points remain unresolved: 95% intervals [−0.7, 4.7], [−1.4, 3.4] include shared repeat-run variance. Ranks 3–7 span 1.5 and 0.8 points, separated from Kimi, Gemini and MiniMax at ranks 8–10 by 5.0 and 3.6 points under A/B.

Claude remains first under consensus/diferentiating weights 1:1, 1:3 and 2:1, inverse-frequency and frequency weighting (� 0.73– 0.96). Frequency weighting emphasises recurring source analyses (0.78); consensus-only scoring puts GLM first (0.60; Appendix B). The evaluated analyses afect rankings even with a stable leader.

Distinct analytical profiles. Similar consensus scores can conceal diferences on diferentiating analyses. Under A, GLM, Doubao and Claude are within half a point on consensus questions, leading in 39%, 34% and 21% of resamples. Claude leads diferentiating questions by 2.4 points under A (86% first-place resamples) and 0.8 under B (55%).

Diferentiating scores have greater dispersion across agents: 4.1 versus 2.8 points under A (diference 1.3, 95% CI [0.33, 2.3]) and 3.1 versus 2.1 under B (0.9, [0.00, 1.9]). Dispersion is the population SD across evaluated agents. B’s interval reaches zero, so the stronger statistical evidence comes from A. Reporting both tiers shows which groups of analyses separate agents with similar totals. These profiles summarise coverage of recurring requirements, not the coherence or validity of the agents’ reasoning.

## 7.2 Browsing, length and completeness

Claude makes 53 searches and 50 page reads per report; Gemini makes 13 and 5. Portals and financial media supply 29% of reads, oficial sources 20%. Within tasks, log searches and reads correlate with T at 0.26 and 0.24 under A, 0.18 and 0.16 under B. Oficialsource share is weakly associated (� = 0.07).

With agent/task fixed efects and task-clustered standard errors (SEs), A’s page-read coeficient is 1.2 points per log unit (SE 0.55), versus 1.1 for searches (SE 0.9, not significant). Adding log length gives page coeficient 1.0 (SE 0.54) and length coeficient 8.6 (SE 1.8). B’s page coeficient is 0.3 (SE 0.57, not significant). Browsing evidence is stronger under A; page reads and length correlate at � = 0.44 within tasks; these observational data do not isolate their contributions.

Longer reports score higher: 9.2 points per log length unit under A (SE 1.7), 5.7 under B (SE 1.3). This association does not identify a causal length efect; padding is tested in §6.4. Length adjustment leaves Claude first under both judges, with Mimo-V2.6-Pro second under A and GLM under B $( \tau = 0 . 7 3 , 0 . 8 7$ with main rankings).

(c) Padding (+40%)

![](images/575103f83842baf2d6886c75a656f26e1ffaf7b7770b9ad7b6eb0690b8bb56a3.jpg)

![](images/1729fc0aebe8b3ab9824fec13f48feb9d7c5c27e2a0f8cc991ad19142ec0eec5.jpg)

![](images/861180737a4317a499f4b84d7dae94221c5d6e9e70d9e8aff52cb226ec964243.jpg)  
Judge A: DeepSeek-V4.1-Flash Judge B: GPT-6-luna  
Figure 3: Completeness diagnostics (levels on 0–100). (a) Ten agents’ report T, ordered by A; Kendall � summarises ranking agreement. (b) Mean completeness on 27 target questions before (solid) and after (pale) deletion. (c) Net report T changes after 40% irrelevant or duplicated text, with paired 95% intervals against unchanged re-evaluation. The dashed A length-slope reference is not an equivalence margin.

Table 4: Completeness T and tier scores on 98 common tasks (0–100). ±: 95% half-width from task bootstrap (2,000 resamples) and shared repeat-run variance. Rank: task-resampling 95% range; P(1st): first-place fraction. Cons./Dif.: consensus/diferentiating scores. Rows follow unrounded judge-A means. Absolute levels depend on the judge.
<table><tr><td rowspan="2">Agent</td><td colspan="6">Judge A (DeepSeek-V4.1-Flash)</td><td colspan="4">Judge B (GPT-6-luna)</td></tr><tr><td>T</td><td>±</td><td>Rank</td><td>P(1st)</td><td>Cons.</td><td>Diff.</td><td>T</td><td>±</td><td>Rank</td><td>P(1st)</td></tr><tr><td>Claude-Opus-5.5</td><td>59.5</td><td>4.3</td><td>1-2</td><td>92%</td><td>67.1</td><td>54.2</td><td>50.6</td><td>4.1</td><td>1-3</td><td>84%</td></tr><tr><td>GLM-5.3</td><td>57.5</td><td>4.1</td><td>1-6</td><td>5%</td><td>67.4</td><td>51.1</td><td>49.6</td><td>3.9</td><td>1-4</td><td>15%</td></tr><tr><td>Mimo-V2.6-Pro</td><td>57.0</td><td>4.3</td><td>2-7</td><td>1%</td><td>65.8</td><td>51.1</td><td>48.1</td><td>4.0</td><td>2-7</td><td>1%</td></tr><tr><td>Mimo-V2.6-Flash</td><td>57.0</td><td>4.6</td><td>2-7</td><td>2%</td><td>65.5</td><td>51.8</td><td>48.3</td><td>4.2</td><td>2-7</td><td>1%</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>56.5</td><td>4.3</td><td>2-7</td><td>1%</td><td>66.1</td><td>50.4</td><td>47.5</td><td>4.2</td><td>3-7</td><td>0%</td></tr><tr><td>Hunyuan-4-Preview</td><td>56.0</td><td>4.5</td><td>2-7</td><td>0%</td><td>64.6</td><td>49.9</td><td>47.8</td><td>4.1</td><td>3-7</td><td>0%</td></tr><tr><td>Doubao-Seed-2.1-Pro</td><td>55.5</td><td>4.3</td><td>3-7</td><td>0%</td><td>67.3</td><td>47.8</td><td>47.6</td><td>4.0</td><td>2-7</td><td>0%</td></tr><tr><td>Kimi-K3</td><td>50.5</td><td>4.3</td><td>8-10</td><td>0%</td><td>61.0</td><td>43.0</td><td>43.9</td><td>4.0</td><td>8-10</td><td>0%</td></tr><tr><td>Gemini-3.8-Flash</td><td>49.9</td><td>3.8</td><td>8-10</td><td>0%</td><td>59.4</td><td>43.1</td><td>43.6</td><td>3.7</td><td>8-10</td><td>0%</td></tr><tr><td>MiniMax-M3</td><td>49.4</td><td>4.1</td><td>8-10</td><td>0%</td><td>60.9</td><td>41.7</td><td>42.2</td><td>3.7</td><td>8-10</td><td>0%</td></tr></table>

Length-matched win rates are 52% for Claude and 60% for GLM (Appendix B); Claude’s lead does not carry over to matching, although it remains first after regression adjustment.

Judge diferences and self-preference. Between-judge score shifts are associated with length. DeepSeek’s length-adjusted shift does not establish self-preference; Appendix B reports estimates and intervals.

## 7.3 Do agents retrieve the source reports?

Research-report sites account for 1.7% of 15,798 successful reads and occur in 17% of generated reports’ trajectories. Sell-side content appears in 31% of reports, with scores +0.5 points above task means, consistent with browsing associations. Task-source titles appear in search results or pages in 53 of 980 trajectories (5.4%, 95% CI [4.2, 7.0]); exposed reports show no detectable gain (−0.7 points, [−3.5, 2.1]). Source-title exposure is uncommon with no detected benefit. The prompt prohibits other institutions’ conclusions, but its efect is not isolated.

## 7.4 Generic validity-rule diagnostics

Completeness records delivered analyses, not their validity. General evaluation applies ten argument-flaw rules covering facts, evidence and reasoning (G01–G10; Appendix A) to aligned material in separate calls (A low, B high efort). It reports claim blocks, basis blocks and reasons per question, without Target requirements, scores or treating missing content as a violation.

Limited agreement on individual rules. Cross-judge � has median 0.16 and maximum 0.29 (G04, calculations); flagged-question Jaccard has median 0.10 (Table 5). A and B flag 11,956 and 6,719 question–rule violations. For G09 (source qualifications), they flag 341 and 25 questions, with 8 shared. Despite shared definitions, rule decisions are less consistent than completeness checkpoints (� = 0.71) on the same reports.

Agent-level violation profiles. To assess whether aggregation reduces disagreement, we compare each agent’s fraction of questions with any violation, separately from completeness. A flags 62–75%, versus B’s 39–63%; rates depend strongly on the judge. Flag-rate rankings give Kendall � = 0.53 (95% CI [0.33, 0.78]; � =

Table 5: Generic rules on 9,196 shared questions: A/B flags, overlap and Cohen’s �. Agent level: flag-rate Kendall � and Pearson � (ten agents).
<table><tr><td rowspan="2">Rule</td><td colspan="4">Question level</td><td colspan="2">Agent level</td></tr><tr><td>A</td><td>B</td><td>Both</td><td>K</td><td>τ</td><td>r</td></tr><tr><td>G04</td><td>3,030</td><td>1,194</td><td>898</td><td>0.29</td><td>0.60</td><td>0.75</td></tr><tr><td>G06</td><td>1,516</td><td>1,362</td><td>567</td><td>0.28</td><td>0.22</td><td>0.49</td></tr><tr><td>G02</td><td>2,481</td><td>1,896</td><td>875</td><td>0.22</td><td>0.47</td><td>0.64</td></tr><tr><td>G07</td><td>1,138</td><td>711</td><td>271</td><td>0.22</td><td>0.64</td><td>0.91</td></tr><tr><td>G03</td><td>640</td><td>339</td><td>101</td><td>0.17</td><td>0.69</td><td>0.73</td></tr><tr><td>G10</td><td>180</td><td>44</td><td>18</td><td>0.15</td><td>0.09</td><td>0.46</td></tr><tr><td>G01</td><td>1,073</td><td>317</td><td>114</td><td>0.12</td><td>0.18</td><td>0.57</td></tr><tr><td>G05</td><td>851</td><td>638</td><td>125</td><td>0.10</td><td>0.49</td><td>0.66</td></tr><tr><td>G08</td><td>706</td><td>193</td><td>41</td><td>0.06</td><td>0.69</td><td>0.76</td></tr><tr><td>G09</td><td>341</td><td>25</td><td>8</td><td>0.04</td><td>-0.04</td><td>-0.12</td></tr></table>

0.76), versus completeness 0.87 [0.56, 0.96]. The intervals overlap, so the point-estimate diference does not establish a ranking-level contrast; the clearer diference is in individual checkpoint and rule agreement (� 0.71 versus median 0.16). Across rules, agent-level � has median 0.48 and range −0.04 (G09) to 0.69 (G03; Table 5). These are diagnostics, not validity scores: combining them with completeness would introduce judge dependence without validated weights.

Descriptive diagnostics. G07, G03, G08 and G04 give relatively consistent agent profiles, unlike G09 and G01. Both judges flag Doubao least often. These describe detected flaws, not validated question judgments (Appendix A).

Interpreting the gap. Checkpoints specify conditions; generic rules leave thresholds such as “overstated” to judges. Binary presence checks difer from multi-label defect detection. Checkpoints pass about half the time; most rules flag fewer than one question in six. A uses medium Target and low General efort, B high for both. These diferences are plausible explanations, not separately tested causes.

## 8 Discussion and Limitations

Implications for evaluation. Web research reports need evaluation beyond retrieved facts. Type-level requirements and questioncentred checks locate incomplete analyses, preserving profiles behind report scores. These diagnose output content, not whether omissions originate in retrieval or synthesis. Tier profiles locate agent diferences on sampled requests. Keeping validity separate prevents conflating missing content with defective reasoning. Absolute scores should identify the judge; relative comparisons are most credible where judges agree.

Scope and transfer. The resource covers Chinese-language financial research (31 industries, 13 question families). Applicability screening and request-scope checks are part of construction, but their accuracy has not been independently quantified. Sampled requests do not establish applicability to every new object. New types need report pools and construction. About ten reports is an internal planning reference, not a transfer guarantee (Table 8).

Beyond financial research. A broader direction is type-level rubric induction as a reusable paradigm for document evaluation.

Academic papers, literature surveys [40], legal documents and other expert-written texts could supply genre-specific requirements, with completeness traced to evidence and argument validity assessed separately. Adapting this paradigm beyond finance will require domain-specific corpora, applicability checks and validation of both layers.

Mining versus task-generated rubrics. An LLM could instead generate a rubric for each task, including a structured rubric. Our distinction is not the presence of a hierarchy alone, but the basis for its analytical questions: model-generated expectations for one request versus analyses documented across expert reports on diferent objects. Mining preserves source-question records for auditing each abstraction, while the question-centred hierarchy connects detailed checks to those analyses. This is a design argument, not a head-to-head comparison.

Developing validity evaluation. Question-specific checks, such as whether growth follows from stated inputs, require separate validation. Once both layers are validated, Pareto dominance compares agents no worse on either dimension and strictly better on at least one, otherwise retaining the trade-of. Future directions include crediting checkpoints only when cited passages have no established flaw, and propagating unsupported premises through question dependencies.

Limitations. The first-author audit is not independent expert validation and does not establish task applicability or the fidelity of every Target. Source support measures recurrence, not importance; tier weights are a design choice. Source experiments restrict a frozen atlas rather than re-extract single-report or task-generated rubrics. We do not isolate the contribution of hierarchical organisation against a flat-checklist baseline. Perturbations test unrelated and duplicated text, not on-topic empty expansion. Rule efort and output format are not isolated by ablation. Shared repeat-run variance is a proxy that includes wording changes. Agents run once per task, so intervals exclude generation randomness; search has no date filter.

Ethics and access. Materials will be made publicly available after the review process. Trees omit analyst names and use opaque institution IDs. Completeness does not assess investment suitability. The audit involved only the first author, no external annotators.

## 9 Conclusion

Expert reports on diferent objects supply recurring analytical requirements, not merely reference answers. RubricAtlas organises these requirements by type and links checks to analytical questions, yielding profiles of completed and missing analyses in Web deep research reports. Reference-report selection experiments motivate cross-report aggregation; cross-judge evaluation and controlled deletions support evidence-linked completeness comparisons. A first-author audit adds a human check on sampled mappings and decisions. Argument validity remains separate: the tested generic rules require further validation before entering a combined score.

## References

[1] Afra Feyza Akyürek, Advait Gosai, Chen Bo Calvin Zhang, Vipul Gupta, Jaehwan Jeong, Anisha Gunjal, Tahseen Rabbani, Maria Mazzone, David Randolph, Mohammad Mahmoudi Meymand, Gurshaan Chattha, Paula Rodriguez, Diego Mares, Pavit Singh, Michael Liu, Subodh Chawla, Pete Cline, Lucy Ogaz, Ernesto Hernandez, Zihao Wang, Pavi Bhatter, Marcos Ayestaran, Bing Liu, and Yunzhong He. 2025. PRBench: Large-Scale Expert Rubrics for Evaluating High-Stakes Professional Reasoning. arXiv:2511.11562 doi:10.48550/arXiv.2511.11562

[2] Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, and William Yang Wang. 2021. FinQA: A Dataset of Numerical Reasoning over Financial Data. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Online and Punta Cana, Dominican Republic, 3697–3711. doi:10.18653/v1/2021.emnlp main.300

[3] Jonathan Cook, Tim Rocktäschel, Jakob Foerster, Dennis Aumiller, and Alex Wang. 2024. TICKing All the Boxes: Generated Checklists Improve LLM Evalua tion and Generation. arXiv:2410.03608 doi:10.48550/arXiv.2410.03608

[4] Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, and Zhendong Mao. 2025. DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents. arXiv:2506.11763 doi:10.48550/arXiv.2506.11763

[5] Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B. Hashimoto. 2024. Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators. arXiv:2404.04475 doi:10.48550/arXiv.2404.04475

[6] Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé Iii, and Kate Crawford. 2021. Datasheets for datasets. Commun. ACM 64, 12 (2021), 86–92. doi:10.1145/3458723

[7] Boyu Gou, Zanming Huang, Yuting Ning, Yu Gu, Michael Lin, Weijian Qi, Andrei Kopanev, Botao Yu, Bernal Jimenez Gutierrez, Yiheng Shu, Chan Hee (Luke) Song, Jiaman Wu, Shijie Chen, Hanane Moussa, Tianshu Zhang, Jian Xie, Yifei Li, Tianci Xue, Zeyi Liao, Kai Zhang, Boyuan Zheng, Zhaowei Cai, Viktor Rozgic, Morteza Ziyadi, Huan Sun, and Yu Su. 2025. Mind2Web 2: Evaluating Agentic Search with Agent-as-a-Judge. In Advances in Neural Information Processing Systems 38. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 192393–192451. doi:10.52202/085713-5778

[8] Anisha Gunjal, Anthony Wang, Elaine Lau, Vaskar Nath, Yunzhong He, Bing Liu, and Sean Hendryx. 2025. Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains. arXiv:2507.17746 doi:10.48550/arXiv.2507.17746

[9] Janghoon Han, Heegyu Kim, Changho Lee, Dahm Lee, Min Hyung Park, Hosung Song, Stanley Jungkyu Choi, Moontae Lee, and Honglak Lee. 2025. DEER: A Benchmark for Evaluating Deep Research Agents on Expert Report Generation. arXiv:2512.17776 doi:10.48550/arXiv.2512.17776

[10] Liang Hu, Jianpeng Jiao, Jiashuo Liu, Yanle Ren, Zhoufutu Wen, Kaiyuan Zhang, Xuanliang Zhang, Xiang Gao, Tianci He, Fei Hu, Yali Liao, Zaiyuan Wang, Cheng hao Yang, Qianyu Yang, Mingren Yin, Zhiyuan Zeng, Ge Zhang, Xinyi Zhang, Xiying Zhao, Zhenwei Zhu, Hongseok Namkoong, Wenhao Huang, and Yuwen Tang. 2025. FinSearchComp: Towards a Realistic, Expert-Level Evaluation of Financial Search and Reasoning. arXiv:2509.13160 doi:10.48550/arXiv.2509.13160

[11] Xinyu Hu, Mingqi Gao, Sen Hu, Yang Zhang, Yicheng Chen, Teng Xu, and Xiaojun Wan. 2024. Are LLM-based Evaluators Confusing NLG Quality Criteria?. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Bangkok, Thailand, 9530–9570. doi:10.18653/v1/2024.acl-long.516

[12] Abigail Z. Jacobs and Hanna Wallach. 2021. Measurement and Fairness. In Proceedings ofthe 2021 ACM Conference on Fairness, Accountability, and Transparency. ACM, 375–385. doi:10.1145/3442188.3445901

[13] Jiajie Jin, Yuyao Zhang, Yimeng Xu, Hongjin Qian, Yutao Zhu, and Zhicheng Dou. 2026. FinSight: Towards Real-World Financial Deep Research. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 5868–5894. doi:10. 18653/v1/2026.acl-long.265

[14] Song Jin, Shuqi Li, Shukun Zhang, and Rui Yan. 2026. FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation. In Proceedings of the AAAI Conference on Artificial Intelligence. Association for the Advancement of Artificial Intelligence (AAAI), 507–515. doi:10.1609/aaai.v40i1.37014

[15] Seungone Kim, Jamin Shin, Yejin Cho, Joel Jang, Shayne Longpre, Hwaran Lee, Sangdoo Yun, Seongjin Shin, Sungdong Kim, James Thorne, and Minjoon Seo. 2024. Prometheus: Inducing Fine-grained Evaluation Capability in Language Models. In International Conference on Learning Representations (ICLR). https: //arxiv.org/abs/2310.08491

[16] Yukyung Lee, JoongHoon Kim, Jaehee Kim, Hyowon Cho, Jaewook Kang, Pilsung Kang, and Najoung Kim. 2025. CheckEval: A reliable LLM-as-a-Judge framework for evaluating text generation using checklists. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing. Association for Com putational Linguistics, Suzhou, China, 15771–15798. doi:10.18653/v1/2025.emnlpmain.796

[17] Ruizhe Li, Mingxuan Du, Benfeng Xu, Chiwei Zhu, Xiaorui Wang, and Zhendong Mao. 2026. DeepResearch Bench II: Diagnosing Deep Research Agents via Rubrics from Expert Reports. arXiv:2601.08536 doi:10.48550/arXiv.2601.08536

[18] Jimmy Lin and Dina Demner-Fushman. 2006. Will Pyramids Built of Nuggets Topple Over?. In Proceedings of the Human Language Technology Conference of the NAACL, Main Conference. Association for Computational Linguistics, New York City, USA, 383–390. https://aclanthology.org/N06-1049/

[19] Beidi Luan, Rui Sun, Sinuo Wang, Yan Gu, Chao Li, Zhenliang Xiong, Jing Li, and Zuo Bai. 2026. FinResearchBench II: A Deep Research Benchmark with Consensus-Derived Gold Rubrics for Distinguishing Financial Report Quality. arXiv:2607.12252 doi:10.48550/arXiv.2607.12252

[20] James Mayfield, Eugene Yang, Dawn Lawrie, Sean MacAvaney, Paul McNamee, Douglas W. Oard, Luca Soldaini, Ian Soborof, Orion Weller, Efsun Kayi, Kate Sanders, Marc Mason, and Noah Hibbler. 2024. On the Evaluation of Machine-Generated Reports. In Proceedings ofthe 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 1904–1915. doi:10. 1145/3626772.3657846

[21] Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. 2023. GAIA: a benchmark for General AI Assistants. arXiv:2311.12983 doi:10.48550/arXiv.2311.12983

[22] Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2023. FActScore: Finegrained Atomic Evaluation of Factual Precision in Long Form Text Generation. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Singapore, 12076–12100. doi:10.18653/v1/2023.emnlp-main.741

[23] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jef Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew Knight, Benjamin Chess, and John Schulman. 2021. WebGPT: Browser-assisted question-answering with human feedback. arXiv:2112.09332 doi:10.48550/arXiv. 2112.09332

[24] Ani Nenkova and Rebecca Passonneau. 2004. Evaluating Content Selection in Summarization: The Pyramid Method. In Proceedings of the Human Language Technology Conference of the North American Chapter of the Association for Computational Linguistics: HLT-NAACL 2004. Association for Computational Linguistics, Boston, Massachusetts, USA, 145–152. https://aclanthology.org/N04-1019/

[25] Arjun Panickssery, Samuel Bowman, and Shi Feng. 2024. LLM Evaluators Recognize and Favor Their Own Generations. In Advances in Neural Information Processing Systems 37. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 68772–68802. doi:10.52202/079017-2197

[26] Ronak Pradeep, Nandan Thakur, Shivani Upadhyay, Daniel Campos, Nick Craswell, Ian Soborof, Hoa Trang Dang, and Jimmy Lin. 2025. The Great Nugget Recall: Automating Fact Extraction and RAG Evaluation with Large Language Models. In Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval. ACM, 180–190. doi:10.1145/3726302.3730090

[27] Tetsuya Sakai. 2006. Evaluating evaluation metrics based on the bootstrap. In Proceedings ofthe 29th annual international ACM SIGIR conference on Research and development in information retrieval. ACM, 525–532. doi:10.1145/1148170.1148261

[28] Rulin Shao, Akari Asai, Shannon Zejiang Shen, Hamish Ivison, Varsha Kishore, Jingming Zhuo, Xinran Zhao, Molly Park, Samuel G. Finlayson, David Sontag, Tyler Murray, Sewon Min, Pradeep Dasigi, Luca Soldaini, Faeze Brahman, Wen tau Yih, Tongshuang Wu, Luke Zettlemoyer, Yoon Kim, Hannaneh Hajishirzi, and Pang Wei Koh. 2025. DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research. arXiv:2511.19399 doi:10.48550/arXiv.2511.19399

[29] Manasi Sharma, Chen Bo Calvin Zhang, Chaithanya Bandi, Clinton Wang, Ankit Aich, Huy Nghiem, Tahseen Rabbani, Ye Htet, Brian Jang, Sumana Basu, Aish warya Balwani, Denis Peskof, Marcos Ayestaran, Sean M. Hendryx, Brad Kenstler, and Bing Liu. 2025. ResearchRubrics: A Benchmark of Prompts and Rubrics For Evaluating Deep Research Agents. arXiv:2511.07685 doi:10.48550/arXiv.2511. 07685

[30] Rui Sun, Zuo Bai, Wentao Zhang, Yuxiang Zhang, Li Zhao, Shan Sun, and Zhengwen Qiu. 2025. FinResearchBench: A Logic Tree based Agent-as-a-Judge Evaluation Framework for Financial Research Agents. arXiv:2507.16248 doi:10.48550/arXiv.2507.16248

[31] Ellen M. Voorhees. 2000. Variations in relevance judgments and the measurement of retrieval efectiveness. Information Processing & Management 36, 5 (2000), 697–716. doi:10.1016/s0306-4573(00)00010-8

[32] Ellen M. Voorhees. 2003. Overview of the TREC 2003 Question Answering Track. In Proceedings of the Twelfth Text REtrieval Conference (TREC 2003). National Institute of Standards and Technology. https://trec.nist.gov/pubs/trec12/papers/ QA.OVERVIEW.pdf

[33] Ellen M. Voorhees and Chris Buckley. 2002. The efect of topic set size on retrieval experiment error. In Proceedings ofthe 25th annual international ACM SIGIR conference on Research and development in information retrieval. ACM, 316–323. doi:10.1145/564376.564432

[34] Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Lingpeng Kong, Qi Liu, Tianyu Liu, and Zhifang Sui. 2024. Large Language Models are not Fair Evaluators. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Bangkok, Thailand, 9440–9450. doi:10.18653/v1/ 2024.acl-long.511

[35] Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jefrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. 2025. BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents. arXiv:2504.12516 doi:10.48550/arXiv.2504.12516

[36] Tianjun Wei, Wei Wen, Ruizhi Qiao, Xing Sun, and Jianghong Ma. 2025. RocketE val: Eficient Automated LLM Evaluation via Grading Checklist. In International Conference on Learning Representations (ICLR). https://openreview.net/forum? id=zJjzNj6QUe

[37] Edwin B. Wilson. 1927. Probable Inference, the Law of Succession, and Statistical Inference. J. Amer. Statist. Assoc. 22, 158 (1927), 209–212. doi:10.1080/01621459. 1927.10502953

[38] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In International Conference on Learning Representations (ICLR). https: //arxiv.org/abs/2210.03629

[39] Seonghyeon Ye, Doyoung Kim, Sungdong Kim, Hyeonbin Hwang, Seungone Kim, YongraeJo,James Thorne,Juho Kim, and Minjoon Seo. 2024. FLASK: Fine-grained Language Model Evaluation based on Alignment Skill Sets. In International Conference on Learning Representations (ICLR). https://arxiv.org/abs/2307.10928

[40] Li S. Yifei, Allen Chang, Chaitanya Malaviya, and Mark Yatskar. 2025. ResearchQA: Evaluating Scholarly Question Answering at Scale Across 75 Fields with Survey-Mined Questions and Rubrics. arXiv:2509.00496 doi:10.48550/arXiv. 2509.00496

[41] Jujia Zhao, Zhaoxin Huan, Zihan Wang, Xiaolu Zhang, Jun Zhou, Suzan Verberne, and Zhaochun Ren. 2026. ReportLogic: Evaluating Logical Quality in Deep Research Reports. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computationa Linguistics, 8470–8502. doi:10.18653/v1/2026.acl-long.384

[42] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems 36. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 46595–46623. doi:10. 52202/075280-2020

[43] Fengbin Zhu, Xiang Yao Ng, Ziyang Liu, Chang Liu, Xianwei Zeng, Chao Wang, Tianhui Tan, Xuan Yao, Pengyang Shao, Min Xu, Zixuan Wang, Jing Wang, Xin Lin, Junfeng Li, Jingxian Zhu, Yang Zhang, Wenjie Wang, Fuli Feng, Richang Hong, Huanbo Luan, Ke-Wei Huang, and Tat-Seng Chua. 2025. FinDeepResearch: Evaluating Deep Research Agents in Rigorous Financial Analysis. arXiv:2510.13936 doi:10.48550/arXiv.2510.13936

[44] Justin Zobel. 1998. How reliable are the results oflarge-scale information retrieval experiments?. In Proceedings of the 21st annual international ACM SIGIR conference on Research and development in information retrieval. ACM, 307–314. doi:10.1145 290941.291014

## A Generic Argument-Flaw Rules

<table><tr><td>Rule</td><td>What is checked</td></tr><tr><td>G01</td><td>Factual and citation accuracy</td></tr><tr><td>G02</td><td>Consistency of objects, metrics and baselines</td></tr><tr><td>G03</td><td>Realisation status, timing and conditions</td></tr><tr><td>G04</td><td>Checkability of key calculations</td></tr><tr><td>G05</td><td>Necessary inferential steps</td></tr><tr><td>G06</td><td>Overstated causal and contribution claims</td></tr><tr><td>G07</td><td>Support for conclusion scope and certainty</td></tr><tr><td>G08</td><td>Treatment of material conflicts and uncertainties</td></tr><tr><td>G09</td><td>Source qualifications for the claims made</td></tr><tr><td>G10</td><td>Information validity at the research cutoff</td></tr></table>

The judge sees report material without Web access. G01 covers errors identifiable from the text or common knowledge; G10 checks time references against the cutof.

Descriptive diagnostics. Agent-level (�, �) is (0.64, 0.91) for G07, (0.69, 0.73) for G03, (0.69, 0.76) for G08 and (0.60, 0.75) for G04. B flags G08 on 2.1% of questions; G04 rates are 32.9%/13.0%. Both judges flag Doubao least (61.5%/38.6%). Question-level � ≤0.29.

These descriptive profiles are more consistent for report-internal features; G09/G01 can require external knowledge. They do not independently validate the rules.

## B Aggregation Robustness

Table 6 contrasts reweighting both tiers with retaining one tier, using A’s fixed judgments.

Table 6: Aggregation sensitivity under A. Ratios are consensus:diferentiating; � compares with the main agent ranking.
<table><tr><td>Scheme</td><td>Cons.:Diff.</td><td>τ</td><td>First</td></tr><tr><td>Both tiers retained</td><td></td><td></td><td></td></tr><tr><td>Equal</td><td>1:1</td><td>0.96</td><td>Claude</td></tr><tr><td>More differentiating</td><td>1:3</td><td>0.96</td><td>Claude</td></tr><tr><td>More consensus</td><td>2:1</td><td>0.73</td><td>Claude</td></tr><tr><td>Inverse frequency</td><td>varies</td><td>0.91</td><td>Claude</td></tr><tr><td>Frequency</td><td>varies</td><td>0.78</td><td>Claude</td></tr><tr><td>One tier retained</td><td></td><td></td><td></td></tr><tr><td>Consensus only</td><td>1:0</td><td>0.60</td><td>GLM</td></tr><tr><td>Differentiating only</td><td>0:1</td><td>0.87</td><td>Claude</td></tr></table>

For 1,607 same-task pairs within 20% in length, win rates are 60% for GLM, 54% for Doubao and 52% for Claude; their order remains unresolved. Log-length regression with agent/task fixed efects retains Claude first, followed by Mimo-V2.6-Pro under A (� = 0.73 with main ranking) and GLM under B (0.87). Matching and regression adjustment are distinct, not a single corrected ranking.

Agents’ A-minus-B scores correlate with mean log report length (� = 0.82). Relative shifts are +0.9 [−0.4, 2.2] for Claude, the longest writer, −1.4 [−2.8, 0.0] for Kimi, the shortest, and +1.0 [−0.5, 2.5] for DeepSeek. DeepSeek’s length-adjusted shift is +0.4 [−1.4, 2.1]: no detected self-preference beyond length, but up to two points remains compatible.

Group averaging. The current scoring manifest places each checkpoint in its own group. Averaging within groups and then across groups therefore equals the checkpoint pass rate for each Target before the negative-criterion gate.

## C Judge Prompt Structure

Evidence alignment. The archive prompt assigns report blocks to questions within a rubric dimension, returning IDs without evaluation.

Checkpoint evaluation. Decisions use full acceptance conditions, alternatives and exclusions. The judge cannot supply missing content from domain knowledge. It cites block IDs for yes, gives reasons for no and checks the negative criterion separately.

General evaluation. The prompt checks errors in actual claims, without Targets or scores. Findings identify question, rule, claim blocks, basis blocks and reason. Resource prompts produce JSON outputs subject to programmatic validation.

## D Re-runs, Perturbations and Execution Failures

Repeat-run protocol. With a fixed seed, we sample ten researchquestion families without replacement, one task per family rotating across pool-size tertiles, and all ten agents (100 reports). Mean mainrun T is 39.6, below the benchmark average; re-evaluation lowers scores by 1.8 points, with a shared agent shift. Perturbations use three agents on these ten tasks.

Repeat-run and perturbation experiments retain the main-run checkpoints but use condensed negative-criterion wording.

Variance proxy. Dividing the SD of paired repeat-minus-main report scores by the square root of two gives a report-level proxy of 4.1 points. Dividing this by the square root of 98 gives 0.42 points for the mean over these tasks. These steps assume independent, equal-variance run errors and independent errors across tasks. Score-interval half-widths combine the task-bootstrap halfwidth and 1.96 times the mean proxy by summing their squares and taking the square root. Agent-gap intervals additionally scale the mean proxy by the square root of two, assuming independent agent errors. Ranks and first-place fractions use task resampling alone. Both judges share A’s proxy, which includes wording variation; B-specific, correlated and generation errors are not separately estimated.

Perturbation results. Deletion lowers target-question T on 20/27 questions under A (95% Wilson interval [37] 55–87%) and 22/27 under B (63–92%). Questions sharing deleted blocks also lose credit: 0.19 and 0.19 on average, versus 0.11 and 0.08 under unchanged re-evaluation.

Padding changes from the original run are −1.4 [−3.4, 0.5] (irrelevant) and −2.1 [−4.2, 0.1] (duplicated) under A, and +0.8 [−1.2, 2.9] and 0.0 [−1.7, 1.7] under B. A’s unchanged re-evaluation shifts scores by −2.5. Figure 3c reports paired net efects against each judge’s unchanged re-evaluation.

Reasoning settings and failures. High efort uses 8,127 tokens under A versus 8,137 at medium (Table 7). B’s high setting uses fewer tokens than A’s medium setting; labels do not imply matched budgets.

Table 7: Settings on 100 reports. Tokens: mean reasoning tokens per Target call; �: report-T correlation with A’s main run on the paired sample.
<table><tr><td>Judge</td><td>Target/General effort</td><td>Tokens</td><td>r</td></tr><tr><td>A</td><td>low/none</td><td>5.8k</td><td>0.94</td></tr><tr><td>A</td><td>medium/low</td><td>8.1k</td><td>0.96</td></tr><tr><td>A</td><td>high/high</td><td>8.1k</td><td>0.96</td></tr><tr><td>B</td><td>medium/low</td><td>1.6k</td><td>0.93</td></tr><tr><td>B</td><td>high/high</td><td>3.5k</td><td>0.95</td></tr></table>

Output-validation failures (5 in the judge-B run, 20 in the repeat run and 15 in perturbations) are listed separately, not assigned zero.

## E Research Types and Task Sampling

The 99 types combine 13 research-question families with 31 industries. Pools contain a median of 30 reports (range 10–60). The type list records pool size, question count, tier composition, median support and institutions per question.

Standardised screening. Fixed AI workflows and prompts check industry/question-family relevance, cross-object applicability and request scope (§4), without regenerating task rubrics or adapting standards to agent outputs.

Table 8: Main threats to validity, the evidence we report and the residual risk.
<table><tr><td>Threat</td><td>Evidence</td><td>Residual risk</td></tr><tr><td>Circular</td><td>Held-out 53%; out-of-half 56%; composition-matched control</td><td>Split halves reuse the atlas questions</td></tr><tr><td>coverage Judge</td><td>Two families: κ = 0.71, τ = 0.87;</td><td>Author audit is not</td></tr><tr><td>dependence</td><td>re-run SD 4.1 points</td><td>independent validation</td></tr><tr><td>Length</td><td>Padding estimate +1.1 points; length-adjusted leader unchanged</td><td>Modest reward not excluded; matched and adjusted rankings</td></tr><tr><td>Self- preference</td><td>DeepSeek shift net of length</td><td>differ One agent is also a</td></tr><tr><td>Leakage</td><td>+0.4 [−1.4, 2.1] 1.7% of reads; no detected</td><td>judge Search API has no</td></tr><tr><td>Frequency ≠</td><td>source-title exposure gain Model-screened,</td><td>date filter Weights remain a</td></tr><tr><td>importance Scope</td><td>author-verified depth; weights fixed before agent results; same leader under five weightings</td><td>design choice</td></tr></table>

Reports are ordered within each type by the SHA-256 hash of the seed, category ID and report ID, concatenated with vertical-bar separators. The seed is dr-bench-99-tasks-20261009-v1. The sampling record documents research-type, object-scope and readability checks, including three object adjustments and six wording changes on 2026-10-10. The incomplete task belongs to a type with nine questions, leaving 912 questions across the 98 common types.

## F Further Discussion

Interpreting profiles. Figure 4 shows both tiers. Under A, GLM and Doubao are within half a point of Claude on consensus questions, but trail by about three and six on diferentiating questions. B’s overall scores are 8.0 points lower. Requirement reduction afects task-level recommendations (Appendix H); comparisons aggregate 98 tasks.

Nugget evaluation. TREC nuggets define content for one input [20, 26, 32]; our questions abstract analyses across objects. Assessor agreement [18] and report support measure recurrence, but scoring uses tier weights (§3), not institution counts. A comparison can satisfy required metrics yet use inappropriate peers: its completeness and validity are distinct.

## G Split-Half, Leave-One-Out, Source-Removal and �-Report Procedures

Table 9 summarises reference selection with frozen candidate questions and judgments. No rubrics are re-mined or reports re-judged.

Leave-one-out measures held-out report coverage. Split-half recomputes tiers using majority support, falling back to median support when tier counts difer by more than half the retained questions and the fallback gives a nonempty consensus tier.

Sampling � reports is without replacement (all reports for smaller pools). Controls draw equally many random questions per task. Task-source subsets use the report supplying the task (Appendix H).

![](images/c0363a41457951b4a42da20471a9e809d8706189bf1e92ff16f021a460c1f192.jpg)  
Figure 4: Tier completeness under bothjudges on 98 common tasks. Paired means are computed within reports, then averaged over tasks. Panels share a score axis and follow Table 4’s judge-A ordering.

Table 9: Reference-report procedures. Fixed: original tiers retained. Split-half recomputes tiers from within-half support.
<table><tr><td>Procedure</td><td>Questions retained</td><td>Tiers</td></tr><tr><td>Leave-one-out</td><td>Remove the measured report&#x27;s support; drop questions below 3 reports</td><td>Not used</td></tr><tr><td>Split-half</td><td>Support from at least two reports in the half (three in the variant)</td><td>Within-half</td></tr><tr><td>Source removal</td><td>Remove task-source support; drop questions below 3 reports</td><td>Fixed</td></tr><tr><td>k reports</td><td>Supported by at least one of k randomly Fixed sampled reports</td><td></td></tr><tr><td>Task source</td><td>Supported by the report used to construct the task</td><td>Fixed</td></tr></table>

## H Single-Reference Rubrics and Per-Type Recommendations

Restricting evaluation to questions supported by the task’s source report changes the task-level leader on 50% of tasks and the topthree set on 76%. Size-matched random subsets change the leader on 45% of tasks (5th–95th percentiles 38–52%). Task-level rankings are therefore sensitive to both tested reductions ofthe requirements, motivating our focus on aggregate agent comparisons.

## I First-Author Audit: Protocol and Results

The first author is a graduate student in finance and also participated in designing the framework. Annotation began from blank forms, without AI assistance or access to the model judgments. This is an author audit blind to model labels, not independent expert validation. It was conducted after the evaluated resource was fixed and does not alter the atlas, model judgments or agent scores.

Checkpoint sampling and materials. The 300 items were sampled with seed 20261023 from atlas checkpoints judged by both models. Quotas were 100 both-yes, 100 both-no, 50 A-yes/B-no and 50 A-no/B-yes. Each report–question contributed at most one item and each task at most six. Questions with no aligned passage under either judge were excluded. The author received the union of the passages aligned by both judges, not necessarily the same material either judge had evaluated. Disagreement can therefore reflect evidence selection as well as interpretation of a checkpoint.

Final labels. The author returned 177 yes and 123 no labels, with no abstentions or blanks. All 300 labels are confirmed as final judgments. Agreement is 223/300 with A and 215/300 with B, with � = 0.487 and 0.433. Table 10 gives the confusion counts and stratified agreement. The model judges agree on 200/300 items (66.7%) in this deliberately disagreement-enriched sample, rather than the full-corpus 85.4%.

Table 10: First-author checkpoint audit. Confusion counts use the author’s label as the row and model label as the column, not as an adjudicated ground truth. The lower panel reports agreement within sampling strata.
<table><tr><td>Author / model labels</td><td>Judge A</td><td>Judge B</td></tr><tr><td>Yes / yes</td><td>125</td><td>121</td></tr><tr><td>Yes / no</td><td>52</td><td>56</td></tr><tr><td>No / yes</td><td>25</td><td>29</td></tr><tr><td>No / no</td><td>98</td><td>94</td></tr><tr><td>Both judges yes (n = 100)</td><td>91%</td><td>91%</td></tr><tr><td>Both judges no (n = 100)</td><td>78%</td><td>78%</td></tr><tr><td>A yes, B no (n = 50)</td><td>68%</td><td>32%</td></tr><tr><td>A no, B yes (n = 50)</td><td>40%</td><td>60%</td></tr></table>

Of the items, 166 concern consensus questions and 134 diferentiating questions. Author–A agreement is 74.1% and 74.6%, respectively (� = 0.468/0.500); author–B agreement is 68.7% and 75.4% (� = 0.359/0.514). All items are included in the primary result.

Reweighting. The stored checkpoint population contains 36,457 both-yes, 33,149 both-no, 8,814 A-yes/B-no and 3,049 A-no/B-yes items. Weighting the four sample agreement rates by these proportions gives 81.3% for A and 78.2% for B. These auxiliary estimates do not correct the report–question and task caps or the exclusion of empty evidence, and are not unbiased estimates of population accuracy.

Source-question mappings. The audit sampled 25 consensus and 25 diferentiating questions, at most one per research type and six source mappings per question. Among 297 mappings, 231 were marked yes (77.8%), 45 partial (15.2%) and 21 no (7.1%). Partial mappings are reported separately: a source question may address only part of a broader canonical question. This audit examines question grouping, not every Target’s fidelity to source arguments or applicability to a task object.

Report preferences. The optional audit included 20 same-task report pairs stratified by judge-A score gaps below 3, from 3 to 8, and at least 8 points. The author’s final judgments preferred the first report in seven pairs, the second in six, and judged seven comparable. Among 13 directional preferences, eight (61.5%) matched A’s completeness ordering; the seven ties are not counted as errors.

This small author-only sample does not validate a general expertpreference ranking.