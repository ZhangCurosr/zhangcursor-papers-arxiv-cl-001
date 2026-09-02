# Skill Following: Evaluating Actual Skill Use in Retrieval-Enabled LLM Agents

Seonghyeon Cho, Chanjun Park<sup>†</sup>

Soongsil University chosh040@soongsil.ac.kr, chanjun.park@ssu.ac.kr

## Abstract

Large Language Model (LLM) agents increasingly rely on external skills, yet standard evaluations obscure whether retrieving these skills actually helps. Aggregate metrics often compare retrieved versus non-retrieved tasks, introducing severe selection bias and failing to isolate the true effect of skill use. To measure this actual-use capability—which we formalize as Skill Following (SF)—we introduce the Retrieval-Invoked Actual-Use Effect (RAE). RAE computes the same-task outcome difference between matched skill-enabled and skilldisabled executions, conditioned exclusively on tasks where the agent actively retrieved a skill. Evaluating 17 LLMs across coding and mathematical domains, we uncover a stark evaluation paradox: models frequently show positive aggregate retrieval lift but negative RAE. On MBPP+, multiple models that appear to benefit system-wide actually harm their own performance on the exact tasks where retrieval occurred. These findings demonstrate that aggregate averages can create a misleading illusion of tool-use proficiency, whereas RAE directly measures whether the retrieval-to-answer pipeline genuinely rescues more outcomes than it harms.

## 1 Introduction

Large language model (LLM) agents are increasingly deployed within external procedural environments, utilizing tools, memory, and skill libraries to navigate and solve complex tasks (Schick et al., 2023; Wu et al., 2024; Shen et al., 2024; Shi et al., 2024). As these models transition from standalone text predictors to retrieval-enabled, interactive agents, the evaluation paradigms used to assess their efficacy must evolve correspondingly. However, measuring the true utility of external skills remains fundamentally challenging. Existing evaluation protocols typically answer one of two broad, system-level questions: whether providing an agent with access to a skill environment improves average performance across an entire benchmark, or whether tasks where skills were retrieved yield higher accuracy than tasks where they were skipped. While these aggregate metrics provide a high-level summary of system behavior, they leave a critical, narrower question systematically unanswered: when an agent actually chooses to retrieve and inject a skill into its context, does that specific external knowledge genuinely improve the outcome of that precise task?

![](images/f7f4009c79a1065f7ac8891cb4e9d35f835073e3061005ff502754f96c19546b.jpg)  
Figure 1: Aggregate retrieval lift and RAE on MBPP+. The x-axis compares skill-enabled accuracy between retrieval-returned and retrieval-skipped task sets, whereas the y-axis compares paired skill-enabled and skill-disabled outcomes on the same retrieval-returned tasks. Shaded off-diagonal quadrants indicate sign disagreement; labels (a)–(c) mark the cases discussed in the text.

We argue that aggregate benchmark averages inherently obscure the true causal utility of skill retrieval, often creating a misleading illusion of capability. Whole-benchmark metrics dilute the impact of external tools by heavily weighting tasks where no skills were ever fetched. More problematically, aggregate retrieved-vs-skipped comparisons suffer from severe task-selection bias; they compare two self-selected subsets of tasks that may fundamentally differ in inherent difficulty, prompt complexity, or skill-library relevance. Because autonomous retrieval is merely the preamble of tool use—requiring the LLM to subsequently receive, interpret, logically ground, and synthesize the returned content into a valid final answer—evaluating this complete pipeline demands a more rigorous methodology. We formalize this multi-step, retrieval-to-answer cognitive process as Skill Following (SF). To accurately evaluate SF without the confounding variables of task selfselection, the measurement must be explicitly conditioned on the actual occurrence of retrieval, utilizing a strictly paired experimental design.

To clearly illustrate the severity of this evaluation gap, Figure 1 contrasts the conventional aggregate retrieval lift against our proposed same-task metric on the MBPP+ benchmark. The x-axis represents the aggregate lift—the standard approach that suggests many modern LLMs, such as Gemini-2.5- Flash-lite and DeepSeek-V3.2, achieve substantial performance gains when skills are retrieved.

However, when we isolate the exact subset of tasks where these models successfully retrieved a skill and compare their outcomes strictly against their own skill-disabled baselines on those identical tasks (the y-axis), a starkly different reality emerges. The shaded quadrants denote regions of critical sign disagreement: models positioned here exhibit a positive aggregate lift that statistically masks a negative actual-use effect. This paradox reveals that what appears to be a systemic improvement is often just a byproduct of the model selectively retrieving skills on inherently easier tasks, while the actual integration of those skills frequently disrupts the model’s reasoning, ultimately rescuing fewer outcomes than it harms.

To correct this methodological blind spot and isolate the actual-use capability of LLMs, we introduce the Retrieval-Invoked Actual-Use Effect (RAE). RAE is a protocol-conditional, same-task outcome metric computed exclusively on the tasks where retrieval was actively invoked and a skill was successfully returned in the skill-enabled execution. By mandating a paired skill-enabled versus skilldisabled comparison, RAE strips away populationlevel confounding factors and directly measures whether the retrieval-invoked SF chain operates as a genuine cognitive aid or a distracting bottleneck. In summary, this paper makes the following primary contributions:

• Identifying the Actual-Use Gap in Agent Evaluation: We conceptualize skill integration as the full Skill Following (SF) chain and mathematically demonstrate that standard aggregate metrics fail to isolate the true tasklevel impact of retrieved skills due to selection bias.

• Introducing the RAE Metric: We propose the Retrieval-Invoked Actual-Use Effect (RAE), establishing a rigorous, pairedevaluation protocol that conditions outcome measurement specifically on active retrieval events.

• Exposing Pervasive Metric Reversals across LLMs: Across an extensive evaluation of 17 diverse LLMs spanning coding and mathematical reasoning domains, we reveal that aggregate conclusions frequently reverse under RAE. Crucially, our comprehensive diagnostic analyses prove that mere context exposure to retrieved skills is vastly insufficient for successful Skill Following, challenging current assumptions about LLM tool-use capabilities.

## 2 Related Work

LLM agents are increasingly evaluated not merely as standalone text predictors, but as complex systems integrating external tools, memory, and retrieval mechanisms. Benchmarks such as Agent-Bench (Liu et al., 2024), AgentBoard (Ma et al., 2024), WebArena (Zhou et al., 2024), OS-World (Xie et al., 2024), TheAgentCompany (Xu et al., 2026), and τ-bench (Yao et al., 2024) measure end-to-end agent behavior in interactive or environment-grounded settings. While these benchmarks provide crucial system-level evaluations, their primary metrics focus on overall task success or step-wise process diagnostics. Consequently, they do not isolate whether the injection of retrieved external content directly changed the final outcome of that exact task via paired comparisons.

Benchmarks specifically targeting external skill access provide a closer point of comparison. Datasets such as SkillsBench (Li et al., 2026),

SWE-Skills-Bench (Han et al., 2026), and Skill-LearnBench (Zhong et al., 2026) employ paired designs to contrast skill-enabled conditions against no-skill or human-skill baselines. Although this paired approach effectively controls for task identity, their reported effects are still predominantly aggregated over the entire evaluation set. In contrast, RAE precisely isolates the actual-use component of the Skill Following chain by conditioning the paired comparison strictly on the subset of tasks where retrieval was actively invoked and content was returned.

A complementary line of research investigates autonomous retrieval, memory, and skill reuse within agents, including systems like Voyager (Wang et al., 2023), Agent KB (Tang et al., 2025), ExpeL (Zhao et al., 2024), Reflexion (Shinn et al., 2023), and CRITIC (Gou et al., 2024). While these frameworks preserve and demonstrate agentinitiated use of external artifacts, they typically do not systematically pair each retrieval-invoked execution with a corresponding skill-disabled baseline to measure the net performance shift. Furthermore, process-level benchmarks such as BFCL (Patil et al., 2025), RAGAS (Es et al., 2024), and IFEval (Zhou et al., 2023) diagnose specific sub-stages like tool calling, retrieval quality, or grounding. RAE is fully complementary to these diagnostics: rather than replacing stage-level metrics, it evaluates the end-to-end impact of the retrieval-invoked chain, strictly asking whether the complete process ultimately rescued more same-task outcomes than it harmed.

## 3 Skill Following

In this section, we establish the formal framework of paired outcome metrics required to rigorously evaluate the task-level efficacy of Skill Following (SF). Conceptualizing SF as the complete retrievalto-answer cognitive chain introduced in Section 1, we define these metrics through strictly paired comparisons between skill-enabled and skill-disabled executions on identical tasks.

## 3.1 Paired Execution Design

To construct a controlled comparison, each task t within the evaluation corpus undergoes two distinct execution paths. The skill-enabled (SE) execution grants the agent full access to the skill library and retrieval interface, whereas the skill-disabled (SD) baseline restricts this access. Let $y _ { t } ^ { + } , y _ { t } ^ { - } \in \{ 0 , 1 \}$ denote the binary correctness outcomes of the SE and SD executions, respectively. The paired outcomes $( y _ { t } ^ { + } , y _ { t } ^ { - } ) \ \in \ \{ 0 , 1 \} ^ { 2 }$ map each task into one of four distinct transition categories (Table 2). This strictly paired design anchors the evaluation to fixed task identities, inherently neutralizing the confounding effects of cross-population task difficulty.

<table><tr><td></td><td>SE correct</td><td>SE wrong</td></tr><tr><td>SD correct</td><td>concordant pass</td><td>harmful flip (c)</td></tr><tr><td>SD wrong</td><td>helpful flip (b)</td><td>concordant fail</td></tr></table>

Table 2: Outcome matrix for paired skill-enabled (SE) and skill-disabled (SD) executions. b denotes the number of tasks that failed in SD but succeeded in SE, whereas c denotes the converse.

## 3.2 Overall and Retrieval-Invoked Effects

Because autonomous retrieval is a selective process, the SE execution will successfully fetch skills for certain tasks but not others. Consequently, we delineate two distinct paired metrics: the Overall Skill-Access Effect (OAE), computed over the entire evaluation corpus, and the Retrieval-Invoked Actual-Use Effect (RAE), strictly conditioned on tasks where retrieval was actively invoked and a skill was returned.

Overall skill-access effect. Let $S _ { \mathrm { a l l } }$ denote the full evaluated task set. Equation 1 defines OAE as the macroscopic, paired outcome difference across the entire benchmark:

$$
\begin{array} { r l r } { \mathrm { O A E } = \Delta _ { \mathrm { a l l } } = \displaystyle \frac { 1 } { | S _ { \mathrm { a l l } } | } \sum _ { t \in S _ { \mathrm { a l l } } } ( y _ { t } ^ { + } - y _ { t } ^ { - } ) } & \\ { \quad \quad \quad \quad \quad = \displaystyle \frac { \mathrm { a l l } _ { b } - \mathrm { a l l } _ { c } } { n _ { \mathrm { p a i r s } } } , } \end{array}\tag{1}
$$

where $n _ { \mathrm { p a i r s } } = | S _ { \mathrm { a l l } } |$ , and $\mathrm { a l l } _ { b }$ and all<sub>c</sub> represent the total helpful and harmful flips over the full dataset.

OAE quantifies the systemic impact of exposing the agent to a skill environment. It computes the average outcome shift across all paired executions, deliberately encompassing tasks where the SE execution bypassed retrieval entirely. While OAE accurately captures the global utility of skill access, it fundamentally dilutes the signal of actual skill usage.

Retrieval-invoked actual-use effect To distill the genuine impact of active skill utilization, we strictly condition our measurement on the subset of tasks where the SE execution invoked retrieval and successfully acquired at least one skill. Let $S _ { \mathrm { c a l l } }$ be this retrieval-invoked subset, with size $n _ { \mathrm { c o n d } } = | S _ { \mathrm { c a l l } } |$ . Equation 2 defines RAE as the paired outcome difference isolated to this critical subset:

<table><tr><td>Metric</td><td>Conditioning set</td><td></td><td>Paired?</td><td>Comparison</td></tr><tr><td>Aggregate retrieval lift</td><td>SE retrieved vs. SE non- retrieved</td><td></td><td>No</td><td>Mean accuracy difference between two task subsets within the skill-enabled execution</td></tr><tr><td>OAE</td><td> $S _ { \mathrm { a l l } }$ </td><td></td><td>Yes</td><td>SE-SD outcome difference over all evalu- ated tasks</td></tr><tr><td>RAE (ours)</td><td> $S _ { \mathrm { c a l l } }$ </td><td></td><td>Yes</td><td>SE-SD outcome difference over retrieval- invoked tasks</td></tr></table>

Table 1: Structural taxonomy of aggregate retrieval lift, OAE, and RAE. OAE and RAE are paired, same-task quantities, whereas aggregate retrieval lift compares self-selected task subsets within the skill-enabled execution.

$$
\begin{array} { l } { \displaystyle \mathrm { R A E } = \Delta _ { \mathrm { c o n d } } = \frac { 1 } { \left| S _ { \mathrm { c a l l } } \right| } \sum _ { t \in S _ { \mathrm { c a l l } } } \left( y _ { t } ^ { + } - y _ { t } ^ { - } \right) } \\ { \displaystyle = \frac { \mathrm { c o n d } _ { b } - \mathrm { c o n d } _ { c } } { n _ { \mathrm { c o n d } } } . } \end{array}\tag{2}
$$

Crucially, unlike aggregate retrieval lift—which merely contrasts self-selected retrieved and nonretrieved populations within the SE condition— RAE is an unconfounded, same-task metric. It directly addresses whether the retrieval-invoked segment of the SF chain successfully improved outcomes relative to the mathematically matched SD baselines. Table 1 delineates the structural distinctions between aggregate retrieval lift, OAE, and RAE.

Relationship between OAE and RAE RAE operates as a conditional sub-component of the broader skill-access effect. Let $S _ { \mathrm { s k i p } } = S _ { \mathrm { a l l } } \setminus S _ { \mathrm { c a l l } }$ denote the subset of tasks where retrieval was circumvented, with size $n _ { \mathrm { s k i p } } = | S _ { \mathrm { s k i p } } |$ . We define the paired effect over this uninvoked complement as:

$$
\Delta _ { \mathrm { s k i p } } = \frac { \mathrm { s k i p } _ { b } - \mathrm { s k i p } _ { c } } { n _ { \mathrm { s k i p } } } ,\tag{3}
$$

where ski $\begin{array} { l l l } { { \rho _ { b } } } & { { = } } & { { { \mathrm { a l l } } _ { b } \mathrm { ~ - ~ } { \mathrm { c o n d } } _ { b } } } \end{array}$ and $\begin{array} { r l } { \operatorname { s k i p } _ { c } } & { { } = } \end{array}$ $\mathrm { a l l } _ { c } - \mathrm { c o n d } _ { c }$ . Combining the retrieval-invoked and skipped components yields a full decomposition of the benchmark effect:

$$
\mathrm { O A E } = { \frac { n _ { \mathrm { c o n d } } } { n _ { \mathrm { p a i r s } } } } \mathrm { R A E } + { \frac { n _ { \mathrm { s k i p } } } { n _ { \mathrm { p a i r s } } } } \Delta _ { \mathrm { s k i p } } .\tag{4}
$$

Equation 4 demonstrates that RAE does not supersede OAE; rather, it isolates the critical mechanism of actual usage. OAE answers the overarching system-design question, whereas RAE targets the precise actual-use inquiry. RAE is reported for each evaluated model–skill-pool–retrieval-policy configuration, whose retrieval-returned subset is system-specific. We therefore report RAE alongside its granular helpful and harmful transition counts; comprehensive details regarding statistical tests, confidence intervals, and edge-case handling are provided in Appendix A.

## 4 Experiment Setup

Our experimental design is structured to address four primary objectives. First, we investigate whether aggregate retrieval lift and RAE yield conflicting conclusions under an identical skill-enabled protocol. Second, we analyze whether aggregate retrieval lift is confounded by retrieval-induced task selection. Third, we validate the consistency of our findings across multiple task partitions, a secondary coding benchmark, and an entirely different domain (mathematical reasoning). Finally, we conduct transition-level diagnostics and skill-content controls to rigorously bound the interpretation of negative RAE outcomes.

## 4.1 Models and Datasets

We evaluate our proposed measurement framework across a comprehensive panel of 17 diverse LLMs. This panel spans various parameter scales and includes both closed-source APIs and open-weight models to ensure broad generalizability (see Appendix K for complete model details).

For our primary analysis, we sample 80 tasks from the MBPP+ benchmark (Liu et al., 2023) using seed 42. To verify cross-benchmark consistency, we evaluate the models on HumanEval+ (Chen et al., 2021). Furthermore, to confirm that the observed phenomena are not strictly domaindependent, we perform a cross-domain replication using 80-task partitions of Math500 (Hendrycks et al., 2021; Lightman et al., 2024).

## 4.2 Skill Environment and Retrieval Mechanism

All analyses operate on fixed, procedural skill libraries tailored to the respective domains. The Coding skill pool comprises 9 procedural coding skills used for both MBPP+ and HumanEval+. The Math skill pool comprises 8 procedural skills used for the Math500 evaluation. Appendix J summarizes both skill pools, and Appendix L reports the prompts and tool schema.

Each skill is formatted as a structured Markdown file (SKILL.md), containing lightweight frontmatter (name, description, signature) and a detailed body (application conditions, code snippets, invariants, and self-tests). The retrieval system indexes the skill descriptions and Markdown bodies using BM25 (Robertson and Zaragoza, 2009). When an agent autonomously invokes the search\_skills(query) function, the full texts of the top-3 retrieved skills are systematically appended to its working context. Agents are permitted up to three search invocations per task.

## 4.3 Paired Execution Protocol

To compute the same-task metrics detailed in §3, each task t undergoes two strictly controlled, parallel executions. In the SE condition, the LLM receives a system prompt that explicitly defines and grants access to the search\_skills(query) tool. In the SD baseline condition, the model receives an identical prompt, but with the tool definition entirely removed. Both execution paths share the exact same task prompt, generation seed, and decoding configuration, ensuring that any outcome variance is strictly attributable to the skill access and subsequent retrieval chain.

## 4.4 Diagnostic Controls and Annotation

To interpret the mechanistic drivers of negative RAE, we design two supplementary diagnostic setups for the strongest coding reversal models (DeepSeek-V3.2 and Gemini-2.5-Flash-lite).

Skill-Content Controls. We run identical evaluations on MBPP+ under five perturbed environments: Normal (the standard skill library), Schema-Empty (tool is available but returns no usable skills), Filler-Dummy (matched non-informative text), Random-skills (retrieval returns structurally valid but completely irrelevant skills), and Corrupted (retrieved content is intentionally misleading or mathematically inconsistent).

Adherence Annotation For transition-level diagnosis, a GPT-5.5 annotator annotates the retrievalinvoked executions. The annotator evaluates the task prompt, retrieval query, retrieved skill excerpts, and the final SE answer to assign a primary behavioral label: appropriate adherence, ignored/independent, misapplied/overapplied, format/interface failure, or unclear. The annotator is blinded to both the SD answers and the underlying correctness labels; the resulting post-hoc labels provide descriptive diagnostics of the observed SE traces, remain separate from the RAE computation, and are evaluated through a stratified human audit reported in Appendix G.1.

## 5 Results

In this section, we investigate whether aggregate retrieval lift and our proposed RAE yield consistent evaluations of LLM tool-use capabilities. Through a comprehensive analysis across coding and mathematical benchmarks, we demonstrate that standard aggregate metrics frequently present a misleading illusion of performance. We structurally unpack this discrepancy by diagnosing task-selection bias, cross-domain consistency, and the precise mechanistic failures that occur once external content enters the model’s context.

## 5.1 Do Aggregate Metrics and RAE Agree?

We first establish the prevalence of metric disagreement. Table 3 reports the coding-domain results, contrasting aggregate retrieval lift with RAE.

Sign Disagreements Across Model and Benchmark Sign reversals—where models exhibit a positive aggregate lift but a negative actual-use effect (RAE)—are alarmingly common. Across MBPP+, Table 3 shows repeated sign disagreements among reportable model configurations.

On HumanEval+, 3 out of 13 models exhibit the same paradox. For instance, Gemini-2.5-Flash-lite consistently shows positive aggregate lift across all three MBPP+ partitions, yet its RAE remains strictly negative. These findings highlight a critical vulnerability in current evaluation standards: seemingly beneficial skill access at the benchmark level frequently masks actual performance degradation on the specific tasks where skills are deployed.

<table><tr><td>Model</td><td>MBPP+ s42</td><td>MBPP+ s43</td><td>MBPP+ s44</td><td>HumanEval+ s42</td></tr><tr><td>Qwen3-8B</td><td> $- 2 6 . 4 / - 1 . 5$ </td><td> $- 0 . 5 / - 9 . 2$ </td><td> $- 2 6 . 6 / - 1 . 6$ </td><td> $- 1 0 . 5 I + 5 . 7$ </td></tr><tr><td>Qwen3.5-9B</td><td> $- 1 1 . 8 / + 0 . 0$ </td><td> $+ 2 6 . 1 / - 3 . 2$ </td><td> $- 3 6 . 4 / - 4 . 1$ </td><td> $- 5 . 4 I + 7 . 2$ </td></tr><tr><td>Gemini-2.5-Flash-lite</td><td> $+ 6 . 2 / - 1 5 . 6$ </td><td> $+ 1 6 . 2 / - 1 5 . 9$ </td><td> $+ 4 . 4 I - 2 2 . 2$ </td><td> $- 0 . 1 / - 2 1 . 1$ </td></tr><tr><td>Mistral-S-24B</td><td> $- 2 7 . 2 / - 2 2 . 5$ </td><td> $+ 1 0 . 8 / - 9 . 0$ </td><td> $- 3 7 . 9 / - 4 5 . 5$ </td><td> $- 1 3 . 9 / - 3 8 . 1$ </td></tr><tr><td>GPT-4o-mini</td><td> $- 2 4 . 0 / - 5 . 5$ </td><td> $+ 1 1 . 4 I - 3 . 4$ </td><td> $- 2 3 . 2 / - 8 . 2$ </td><td> $- 2 . 5 / - 1 . 4$ </td></tr><tr><td>Claude-3.5-H</td><td> $- 5 . 0 I + 2 . 5$ </td><td> $- 1 . 0 / - 2 . 1$ </td><td> $- 6 . 6 / - 3 . 8$ </td><td> $- 2 . 7 / - 7 . 7$ </td></tr><tr><td>Claude-4.5-H</td><td> $+ 1 . 4 / + 1 . 4$ </td><td> $+ 2 1 . 5 / + 0 . 0$ </td><td> $+ 4 8 . 0 / + 2 . 7$ </td><td> $+ 4 . 5 / + 0 . 0$ </td></tr><tr><td>Qwen3-32B</td><td> $- 1 6 . 9 / + 1 . 8$ </td><td> $- 2 7 . 9 / - 6 . 0$ </td><td> $- 1 2 . 4 / - 8 . 5$ </td><td> $- 1 1 . 2 / - 3 0 . 1$ </td></tr><tr><td>GLM4-32B</td><td> $- 2 5 . 1 / + 0 . 0$ </td><td> $+ 1 . 8 / - 1 . 4$ </td><td> $- 1 3 . 9 / - 6 . 8$ </td><td> $- 2 . 8 / - 5 . 6$ </td></tr><tr><td>Command-R</td><td> $- 6 . 2 / - 1 4 . 5$ </td><td> $- 3 6 . 1 / - 2 4 . 1$ </td><td> $- 3 3 . 3 / - 2 6 . 7 $ </td><td> $- 5 . 8 / + 0 . 0$ </td></tr><tr><td>Llama-3.3-70B</td><td> $+ 6 9 . 8 / + 4 . 2$ </td><td> $+ 4 4 . 2 / - 2 . 0$ </td><td> $+ 6 0 . 0 / + 7 . 5$ </td><td> $+ 6 1 . 5 / + 7 . 1$ </td></tr><tr><td>Qwen3-235B</td><td> $- 0 . 1 / + 7 . 0$ </td><td> $- 2 1 . 1 / + 0 . 0$ </td><td> $- 2 7 . 1 \ : / + 3 . 7$ </td><td> $- 7 . 0 / + 1 5 . 5$ </td></tr><tr><td>DeepSeek-V3.2</td><td> $+ 3 1 . 6 / - 9 . 1$ </td><td> $- 1 1 . 5 / - 1 5 . 4$ </td><td> $+ 4 6 . 2 / - 2 3 . 1$ </td><td> $- 1 7 . 1 / - 2 3 . 0$ </td></tr></table>

Table 3: Main coding-domain aggregate-vs-RAE comparison. Each cell reports aggregate retrieval lift / RAE in percentage points. Bold entries indicate opposite signs between the unpaired retrieved-vs-skipped aggregate and the retrieval-invoked same-task metric. The table includes cells satisfying the reporting filter; the full 17-model panel, including non-reportable cells, is provided in Appendix B.

![](images/455eae803c15e7c0c0fccb6e3d2742afad745ae8e2eefaf72af206cf80c0853c.jpg)  
Figure 2: Autonomous retrieval induces severe taskselection bias. The x-axis reports the skill-disabled selection gap $( \Delta _ { \mathrm { s e l e c t } } )$ between tasks that did and did not return a skill. The y-axis reports aggregate retrieval lift within the skill-enabled execution. Each point represents a reportable model cell, colored by the sign of its RAE. The prevalent nonzero selection gaps mathematically demonstrate that aggregate retrieval lift compares inherently mismatched task populations, failing to isolate the true causal effect of skill usage.

## 5.2 What Drives the Divergence Between Aggregate Lift and Actual Use?

To understand why aggregate retrieval lift diverges so sharply from RAE, we must analyze the task populations being compared. Aggregate lift evaluates retrieved versus non-retrieved tasks within the skill-enabled execution. However, these subsets are not randomized; they are self-selected by the

LLM’s own retrieval policy.

To quantify this underlying discrepancy, we introduce the skill-disabled selection gap $( \Delta _ { \mathrm { s e l e c t } } )$ Let $S _ { \mathrm { r e t } }$ and $S _ { \mathrm { n o r e t } }$ denote the subsets of tasks that did and did not return a skill, respectively. We measure their inherent difficulty using the paired SD baseline:

$$
\Delta _ { \mathrm { s e l e c t } } = \mathrm { A c c } ^ { - } ( S _ { \mathrm { r e t } } ) - \mathrm { A c c } ^ { - } ( S _ { \mathrm { n o r e t } } )\tag{5}
$$

Selection Bias Confounds Aggregate Lift As illustrated in Figure 2, autonomous retrieval induces substantial task-selection differences $( \Delta _ { \mathrm { s e l e c t } } \neq 0 )$ This provides the mechanistic explanation for the metric disagreement: aggregate retrieval lift effectively compares two distinct, self-selected task populations that already differ in baseline difficulty or prompt complexity. RAE, by contrast, neutralizes this bias by strictly comparing the same retrievalinvoked tasks against their paired SD executions.

## 5.3 Does the Evaluation Discrepancy Persist Beyond Benchmarks?

To confirm that this paradox is not an artifact of a single benchmark, we extend our analysis to alternative datasets and domains.

Cross-Benchmark Consistency The aggregatevs-RAE disagreement is not confined to MBPP+. On HumanEval+, 3 out of 13 reportable models show sign disagreement between aggregate retrieval lift and RAE. Unlike the dominant MBPP+ reversal emphasized in the introduction, these HumanEval+ disagreements occur in the opposite direction: aggregate retrieval lift is negative while RAE is positive for Qwen3-8B, Qwen3.5-9B, and Qwen3-235B. This indicates that aggregate retrieved-vs-skipped comparisons can misestimate retrieval-invoked actual use in either direction, either overstating or understating the effect of SF. Appendix D reports descriptive MBPP+–HumanEval+ correlation analyses, which we use only as diagnostic context rather than as benchmark-invariant model rankings.

![](images/8d051e87e9ea86240c89285eb9b82dd41f15fe0c2054562f44385bbf1db0a5fc.jpg)  
Figure 3: Math500 cross-domain results. Comparing aggregate retrieval lift, OAE, and RAE reveals that the metric disagreement generalizes beyond coding. Models with highly positive aggregate lift collapse into severely negative actual-use effects (RAE).

Cross-Domain Results on Math500 Figure 3 shows that the disagreement between aggregate retrieval lift and RAE also appears in the mathematical reasoning domain. On Math500, Llama-3.3- 70B obtains a +14.2 pp aggregate lift but a −39.4 pp RAE under the paired retrieval-invoked comparison. Gemini-2.5-Flash-lite shows a similar pattern (+13.2 pp aggregate lift vs. −11.0 pp RAE). These results indicate that the aggregate-vs-RAE disagreement is not limited to coding benchmarks. Full Math500 results are reported in Appendix C.

## 5.4 Does Higher Retrieval Coverage Translate to Better Downstream Integration?

A prevailing assumption in tool-use evaluation is that higher retrieval coverage naturally translates to improved downstream performance. We examine this assumption by mapping retrieval coverage against RAE (Figure 4). The results indicate that several models with high retrieval rates nevertheless yield negative actual-use effects. To unpack this discrepancy, the corresponding stage-level diagnostic definitions and values are detailed in Appendix E.

![](images/c4d0355aca1e4abc94b9d4e1e30b3b83fbce2ffbd09057c99576d2a2ef88b7c3.jpg)  
Figure 4: Retrieval coverage versus RAE on MBPP+ seed 42. The horizontal line marks zero RAE. Multiple high-retrieval models exhibit negative RAE, proving that successful retrieval invocation (exposure) does not reliably translate to successful SF (utilization).

Retrieval is not Following If mere exposure to external skills were sufficient, models with higher retrieval coverage would consistently show positive RAE. Instead, Figure 4 indicates that several high-coverage models still exhibit negative RAE. As detailed in the stage-diagnostic analysis (Appendix E), there is a notable gap between fetching a skill and successfully applying it: even when models retrieve skills at a high rate, their structural adherence to the provided content and effective success rates remain substantially lower. This disconnect highlights the distinction between exposure and actual use. Invoking a search function and receiving returned text does not guarantee that the LLM will correctly interpret, align, and incorporate that content into a valid final answer.

## 5.5 Mechanistic Failures: How and Where Does the SF Chain Break Down?

To diagnose how SF breaks down, we analyze harmful transitions—cases where the SD execution succeeds, but the SE execution retrieves a skill and subsequently fails. Using a GPT-5.5 annotator, we annotated 45 harmful transitions for DeepSeek-V3.2 and 40 for Gemini-2.5-Flash-lite across three

<table><tr><td>Control condition</td><td>DeepSeek-V3.2</td><td>Gemini-2.5-Flash-lite</td></tr><tr><td>Normal</td><td>-15.9</td><td>-17.9</td></tr><tr><td>Schema-Empty</td><td></td><td></td></tr><tr><td>Filler-Dummy</td><td>-10.2</td><td>-5.0</td></tr><tr><td>Random-skills</td><td>-15.1</td><td>-15.4</td></tr><tr><td>Corrupted</td><td>-9.1</td><td>-17.2</td></tr></table>

Table 4: Skill-content controls reporting pooled RAE in percentage points. Schema-Empty returns no usable skill, rendering RAE mathematically undefined $( n _ { \mathrm { c o n d } } = 0 )$ . Crucially, RAE remains strictly negative across all conditions where any content is returned.

MBPP+ seeds.

Ignored or Independent Use of Retrieved Skills The most frequent failure mode is ignored/independent. In these cases, the SE execution retrieves a skill, but the final answer shows no clear evidence that the returned content was incorporated into the solution. DeepSeek-V3.2 also showsformat/interfacefailures, where tool traces or interface artifacts appear in the final output. These diagnostics indicate that retrieval invocation alone is not a sufficient measure of SF: after retrieval, the model must still determine whether the returned content is relevant, integrate it into the task solution, and preserve the required output format. Appendix F provides the full harmful-transition taxonomy, and Appendix G reports the adherence-label distributions and labelconditioned RAE analysis.

## 5.6 Is Negative RAE Solely an Artifact of Defective Retrieved Content?

Finally, we test whether negative RAE can be attributed solely to the quality of the retrieved skill text. We run skill-content controls for the two strongest coding reversal models, DeepSeek-V3.2 and Gemini-2.5-Flash-lite, replacing the normal skill returns with empty, filler, random, or corrupted returned-content conditions. Table 4 summarizes the pooled RAE values, while Appendix H reports the corresponding raw transition counts, retrieval coverage, OAE, and aggregate retrieval lift for each control condition.

Returned Content Alone Does Not Determine RAE The content-control results indicate that negative RAE is not solely an artifact of poorly written skill content. For both DeepSeek-V3.2 and Gemini-2.5-Flash-lite, RAE remains negative across all non-empty returned-content conditions, including Normal skills, Random-skills returns, Corrupted content, and Filler-Dummy text. Thus, a structured skill artifact does not by itself guarantee a positive actual-use effect. RAE instead reflects the full retrieval-conditioned execution chain, including retrieval invocation, content injection, relevance assessment, skill integration, and finalanswer formatting. These controls therefore bound, rather than fully identify, the source of negative RAE.

## 6 Conclusion

As LLMs increasingly operate within external skill environments, standard evaluation metrics—which rely on whole-benchmark averages or self-selected task subsets—create a misleading illusion of tooluse proficiency. To isolate the true causal impact of retrieved knowledge, we introduced the RAE. By strictly pairing skill-enabled and skilldisabled executions on the exact tasks where retrieval occurs, RAE neutralizes task-selection bias and directly measures the efficacy of the entire SF chain. Our evaluations across coding and mathematical domains reveal a pervasive paradox: models frequently exhibit positive aggregate retrieval lift while suffering a negative RAE. Apparent benchmark-level gains often mask actual performance degradation on the specific tasks where skills were actively deployed. By decoupling mere context exposure from actual cognitive integration, RAE provides a necessary methodological lens. It ensures that as agents become more autonomous, their tool-use mechanisms are rigorously evaluated as genuine cognitive aids rather than deceptive liabilities.

## Limitations

Our experiments cover a focused evaluation regime: single-pass coding and mathematical tasks under a fixed retrieval interface and fixed procedural skill libraries. The results should therefore not be read as establishing the same prevalence or magnitude for long-horizon agents, executable code libraries, language-centric skill settings, natural-language annotator-based tasks, or systems with iterative memory and planning. RAE is also bounded in its statistical and causal interpretation. Because $S _ { \mathrm { c a l l } }$ is selected by the skill-enabled execution itself, RAE is not an unbiased causal effect over a pretreatment task population, nor a model-level skilluse ability score. It is a protocol-conditional paired outcome signal: under a fixed deployment protocol, when the agent entered the retrieval-returned portion of the SF chain, did the chain help or harm the same tasks? Finer causal decomposition would require additional ablations such as oracle retrieval, metadata-only retrieval, explicit full-skill fetch, multiple skill-pool constructions, and manually adjudicated causal labels.

## Ethical Considerations

This work does not involve human subjects, private user data, or demographic profiling; all experiments are conducted on coding and mathematical benchmarks. The main ethical relevance of the study is evaluation reliability. Skill- or toolaugmented agents may be presented as improved systems when aggregate metrics hide cases where retrieval-invoked skill use harms the same tasks it was meant to help. Such overstatement can lead to misleading deployment claims and weaker auditability of agent systems. RAE is intended to make this failure mode more visible, but it should not be interpreted as a safety guarantee or as a complete causal explanation of failures. Positive or negative RAE values remain specific to the evaluated protocol, model, retrieval policy, skill library, and benchmark, and should be reported alongside complementary diagnostics and reproducibility artifacts. AI assistant tools were used to assist with coding, script debugging, result organization, and language polishing; all experimental design decisions, analyses, and final manuscript content were reviewed and controlled by the authors.

## Acknowledgments

This work was supported by the Korea Internet & Security Agency (KISA) grant funded by the Korea government (PIPC) (No. RS-2026-25526342, Development of Technologies for Preventing Sensitive Information Inference and Risk Assessment in Foundation Model Operations). This work was also supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2026-25483747). This research was further supported by the Culture, Sports and

Tourism R&D Program through the Korea Creative Content Agency, funded by the Ministry of Culture, Sports and Tourism in 2026 (Project Name: Develop AI Agent Technology to Connect Knowledge through Public Cultural Facility-Based Discussion and Communication, Project Number: RS-2026- 25520645).

## References

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, and 1 others. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, and 1 others. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2024. Ragas: Automated evaluation of retrieval augmented generation. In Proceedings of the 18th conference of the european chapter of the associationfor computational linguistics: system demonstrations, pages 150–158.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yujiu Yang, Nan Duan, Weizhu Chen, and 1 others. 2024. Critic: Large language models can self-correct with toolinteractive critiquing. In International Conference on Learning Representations, volume 2024, pages 57734–57811.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Tingxu Han, Yi Zhang, Wei Song, Chunrong Fang, Zhenyu Chen, Youcheng Sun, and Lijie Hu. 2026. Swe-skills-bench: Do agent skills actually help in real-world software engineering? arXiv preprint arXiv:2603.15401.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. NeurIPS.

Xiangyi Li, Wenbo Chen, Yimin Liu, Shenghan Zheng, Xiaokun Chen, Yifeng He, Yubo Li, Bingran You, Haotian Shen, Jiankai Sun, and 1 others. 2026. Skillsbench: Benchmarking how well agent skills work across diverse tasks. arXiv preprint arXiv:2602.12670.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In International Conference on Learning Representations, volume 2024, pages 39578–39601.

Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, and 1 others. 2025. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. 2023. Is your code generated by chat-GPT really correct? rigorous evaluation of large language models for code generation. In Thirty-seventh Conference on Neural Information Processing Systems.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, and 1 others. 2024. Agentbench: Evaluating llms as agents. In International Conference on Learning Representations, volume 2024, pages 52989–53046.

Chang Ma, Junlei Zhang, Zhihao Zhu, Cheng Yang, Yujiu Yang, Yaohui Jin, Zhenzhong Lan, Lingpeng Kong, and Junxian He. 2024. Agentboard: An analytical evaluation board of multi-turn llm agents. Advances in neural information processing systems, 37:74325–74362.

Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E Gonzalez. 2025. The berkeley function calling leaderboard (bfcl): From tool use to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: BM25 and beyond, volume 4. Now Publishers Inc.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. Preprint, arXiv:2302.04761.

Weizhou Shen, Chenliang Li, Hongzhan Chen, Ming Yan, Xiaojun Quan, Hehong Chen, Ji Zhang, and Fei Huang. 2024. Small llms are weak tool learners: A multi-llm agent. Preprint, arXiv:2401.07324.

Zhengliang Shi, Shen Gao, Xiuyi Chen, Yue Feng, Lingyong Yan, Haibo Shi, Dawei Yin, Pengjie Ren, Suzan Verberne, and Zhaochun Ren. 2024. Learning to use tools via cooperative and interactive agents. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 10642–10657.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652.

Xiangru Tang, Tianrui Qin, Tianhao Peng, Ziyang Zhou, Daniel Shao, Tingting Du, Xinming Wei, Peng Xia, Fang Wu, He Zhu, and 1 others. 2025. Agent kb: Leveraging cross-domain experience for agentic problem solving. arXiv preprint arXiv:2507.06229.

Qwen Team. 2024. Qwen2.5: A party of foundation models.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291.

Shirley Wu, Shiyu Zhao, Qian Huang, Kexin Huang, Michihiro Yasunaga, Kaidi Cao, Vassilis N. Ioannidis, Karthik Subbian, Jure Leskovec, and James Zou. 2024. Avatar: Optimizing llm agents for tool usage via contrastive reasoning. Preprint, arXiv:2406.11200.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh J Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, and 1 others. 2024. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094.

Frank Fangzheng Xu, Yufan Song, Boxuan Li, Yuxuan Tang, Kritanjali Jain, Mengxue Bao, Zora Wang, Xuhui Zhou, Zhitong Guo, Murong Cao, and 1 others. 2026. Theagentcompany: benchmarking llm agents on consequential real world tasks. Advances in Neural Information Processing Systems, 38.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. 2024. τ-bench: A benchmark for toolagent-user interaction in real-world domains. arXiv preprint arXiv:2406.12045.

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. 2024. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19632–19642.

Shanshan Zhong, Yi Lu, Jingjie Ning, Yibing Wan, Lihan Feng, Yuyi Ao, Leonardo FR Ribeiro, Markus Dreyer, Sean Ammirati, and Chenyan Xiong. 2026. Skilllearnbench: Benchmarking continual learning methods for agent skill generation on real-world tasks. arXiv preprint arXiv:2604.20087.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, and 1 others. 2024. Webarena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, volume 2024, pages 15585–15606.

## A Metric Computation and Reporting Filters

All cell-level metrics are computed from pooled raw counts rather than from means of per-run rates. A cell is defined by a fixed model, benchmark, partition seed, skill pool or control condition, and evaluation protocol. When config-identical reruns exist for the same cell, we first sum the underlying counts and then compute the reported rate.

For RAE, we aggregate raw counts across configidentical runs:

$$
\begin{array} { l l } { { \displaystyle B _ { \mathrm { c o n d } } = \sum _ { i } \mathrm { c o n d } _ { b , i } , } } & { { \displaystyle C _ { \mathrm { c o n d } } = \sum _ { i } \mathrm { c o n d } _ { c , i } , } } \\ { { \displaystyle N _ { \mathrm { c o n d } } = \sum _ { i } n _ { \mathrm { c o n d } , i } . } } \end{array}\tag{6}
$$

We report

$$
\mathrm { R A E } = \Delta _ { \mathrm { c o n d } } = 1 0 0 \times { \frac { B _ { \mathrm { c o n d } } - C _ { \mathrm { c o n d } } } { N _ { \mathrm { c o n d } } } } .\tag{7}
$$

Here $B _ { \mathrm { c o n d } }$ counts helpful transitions, where the skill-disabled execution fails and the skill-enabled execution succeeds, and $C _ { \mathrm { c o n d } }$ counts harmful transitions, where the skill-disabled execution succeeds and the skill-enabled execution fails, restricted to tasks where retrieval was invoked and at least one skill was returned in the skill-enabled execution.

For any task subset S, the paired binary outcomes are $( y _ { t } ^ { + } , y _ { t } ^ { - } ) \in \{ 0 , 1 \} ^ { 2 }$ , where 1 denotes a correct final answer and 0 denotes an incorrect final answer. The four paired-outcome cells can be written as

$$
\displaystyle \frac { \partial _ { t } ^ { + } = 1 \quad y _ { t } ^ { + } = 0 } { y _ { t } ^ { - } = 1 \quad } \frac { a } { b } \qquad \stackrel { c } { d }
$$

where $a , b , c ,$ d are task counts. The discordant cells are b and c: b counts helpful transitions, $( y _ { t } ^ { + } , y _ { t } ^ { - } )$ (1, 0), and c counts harmful transitions, $( y _ { t } ^ { + } , y _ { t } ^ { - } ) =$ (0, 1).

The paired outcome difference over $S$ is

$$
\frac { 1 } { | S | } \sum _ { t \in S } ( y _ { t } ^ { + } - y _ { t } ^ { - } ) .\tag{8}
$$

Concordant pairs contribute zero, while helpful and harmful transitions contribute +1 and −1, respectively. Therefore,

$$
\sum _ { t \in S } ( y _ { t } ^ { + } - y _ { t } ^ { - } ) = b - c .\tag{9}
$$

Thus, the paired same-task difference over any subset S is $( b - c ) / | S |$

For OAE, we aggregate

$$
\begin{array} { l l } { { \displaystyle B _ { \mathrm { a l l } } = \sum _ { i } \mathrm { a l l } _ { b , i } , } } & { { ~ C _ { \mathrm { a l l } } = \sum _ { i } \mathrm { a l l } _ { c , i } , } } \\ { { \displaystyle N _ { \mathrm { a l l } } = \sum _ { i } n _ { \mathrm { p a i r s } , i } . } } \end{array}\tag{10}
$$

We report

$$
\mathrm { O A E } = \Delta _ { \mathrm { a l l } } = 1 0 0 \times \frac { B _ { \mathrm { a l l } } - C _ { \mathrm { a l l } } } { N _ { \mathrm { a l l } } } .\tag{11}
$$

The decomposition into retrieval-invoked and skipped subsets follows from $S _ { \mathrm { a l l } } = S _ { \mathrm { c a l l } } \cup S _ { \mathrm { s k i p } } \mathrm { : }$

$$
\mathrm { O A E } = { \frac { N _ { \mathrm { c o n d } } } { N _ { \mathrm { a l l } } } } \mathrm { R A E } + { \frac { N _ { \mathrm { s k i p } } } { N _ { \mathrm { a l l } } } } \Delta _ { \mathrm { s k i p } } ,\tag{12}
$$

where

$$
\begin{array} { r l } & { \Delta _ { \mathrm { s k i p } } = 1 0 0 \times \frac { B _ { \mathrm { s k i p } } - C _ { \mathrm { s k i p } } } { N _ { \mathrm { s k i p } } } , } \\ & { B _ { \mathrm { s k i p } } = B _ { \mathrm { a l l } } - B _ { \mathrm { c o n d } } , } \\ & { C _ { \mathrm { s k i p } } = C _ { \mathrm { a l l } } - C _ { \mathrm { c o n d } } . } \end{array}\tag{13}
$$

When $N _ { \mathrm { s k i p } } ~ = ~ 0$ , the skipped-subset term is omitted.

Unlike RAE, this quantity averages the paired skill-enabled/skill-disabled difference over the entire evaluated task set, including tasks where the agent never retrieved a skill.

For aggregate retrieval lift, let $R _ { i }$ and $\bar { R } _ { i }$ denote the retrieved and non-retrieved task subsets in run i. We aggregate

$$
\begin{array} { l l } { { \displaystyle R _ { \mathrm { c o r r } } = \sum _ { i } \mathrm { c o r r e c t } ^ { + } ( R _ { i } ) , } } & { { \displaystyle R _ { N } = \sum _ { i } | R _ { i } | , } } \\ { { \displaystyle S _ { \mathrm { c o r r } } = \sum _ { i } \mathrm { c o r r e c t } ^ { + } ( \bar { R } _ { i } ) , } } & { { \displaystyle S _ { N } = \sum _ { i } | \bar { R } _ { i } | . } } \end{array}\tag{14}
$$

When both denominators are nonzero, we report

$$
\Delta _ { \mathrm { a g g } } = 1 0 0 \times \left( \frac { R _ { \mathrm { c o r r } } } { R _ { N } } - \frac { S _ { \mathrm { c o r r } } } { S _ { N } } \right) .\tag{15}
$$

If either $R _ { N } ~ = ~ 0$ or $S _ { N } ~ = ~ 0 .$ , aggregate retrieval lift is undefined because one of the two skillenabled task populations is empty. This occurs, for example, when the agent invokes retrieval on every task.

For main sign-disagreement counts, a cell is included only when $N _ { \mathrm { c o n d } } \geq 1 0$ and both RAE and aggregate retrieval lift are defined. Cells below this retrieval-invocation threshold are omitted from prevalence counts because the conditional sametask estimate is too sparse. Cells with $N _ { \mathrm { c o n d } } \geq 1 0$ but undefined aggregate retrieval lift retain their RAE value, but they are not counted in aggregatevs-RAE sign-disagreement denominators.

<table><tr><td>Threshold</td><td>MBPP+ s42</td><td>MBPP+ s43</td><td>MBPP+ s44</td><td>HumanEval+ s42</td></tr><tr><td> $n _ { \mathrm { c o n d } } \geq 5$ </td><td>5/13</td><td>6/13</td><td>3/13</td><td>3/13</td></tr><tr><td> $n _ { \mathrm { c o n d } } \geq 1 0$ </td><td>5/13</td><td>6/13</td><td>3/13</td><td>3/13</td></tr><tr><td> $n _ { \mathrm { c o n d } } \geq 1 5$ </td><td>5/13</td><td>6/13</td><td>3/13</td><td>3/12</td></tr><tr><td> $n _ { \mathrm { c o n d } } \geq 2 0$ </td><td>5/13</td><td>6/13</td><td>3/13</td><td>3/11</td></tr></table>

Table 5: Sensitivity of aggregate-vs-RAE sign-disagreement counts to the retrieval-invoked reporting threshold. Each cell reports sign-disagreement cells divided by reportable cells. The main text uses $n _ { \mathrm { c o n d } } \geq 1 0 .$
<table><tr><td>Model</td><td>MBPP+ s42</td><td>MBPP+ s43</td><td>MBPP+ s44</td><td>HumanEval+ s42</td><td>Sign disagree / cells</td></tr><tr><td>Qwen2.5-7B</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-8B</td><td>-26.4/-1.5</td><td>-0.5/-9.2</td><td>-26.6 / -1.6</td><td> $- 1 0 . 5 I + 5 . 7$ </td><td>1/4</td></tr><tr><td>Qwen3.5-9B</td><td>-11.8/+0.0</td><td>+26.1/-3.2</td><td>-36.4 / -4.1</td><td> $- 5 . 4 I + 7 . 2$ </td><td>2/4</td></tr><tr><td>Llama-3.1-8B</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Mistral-Nemo</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini-2.5-Flash-lite</td><td>+6.2/-15.6</td><td>+16.2/-15.9</td><td>+4.4/-22.2</td><td>-0.1/-21.1</td><td>3/4</td></tr><tr><td>Gemini-2.0-Flash Mistral-S-24B</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>-27.2/-22.5</td><td>+10.8/-9.0</td><td>-37.9 / -45.5</td><td>-13.9 /-38.1</td><td>1/4</td></tr><tr><td>GPT-4o-mini</td><td>-24.0 /-5.5</td><td>+11.4/-3.4</td><td>-23.2 /-8.2</td><td>-2.5 /-1.4</td><td>1/4</td></tr><tr><td>Claude-3.5-H</td><td>-5.0/+2.5</td><td>-1.0/-2.1</td><td>-6.6 /-3.8</td><td>-2.7/-7.7</td><td>1/4</td></tr><tr><td>Claude-4.5-H Qwen3-32B</td><td>+1.4/+1.4</td><td>+21.5/+0.0</td><td>+48.0/+2.7</td><td>+4.5/+0.0</td><td>0/4</td></tr><tr><td></td><td>-16.9/+1.8</td><td>-27.9 /-6.0</td><td>-12.4/-8.5</td><td>-11.2/-30.1</td><td>1/4</td></tr><tr><td>GLM4-32B</td><td>-25.1/+0.0</td><td>+1.8/-1.4</td><td>-13.9 / -6.8</td><td>-2.8 / -5.6</td><td>1/4</td></tr><tr><td>Command-R</td><td>-6.2 / -14.5</td><td>-36.1/-24.1</td><td>-33.3 /-26.7</td><td>-5.8/+0.0</td><td>0/4</td></tr><tr><td>Llama-3.3-70B</td><td>+69.8/+4.2</td><td>+44.2/-2.0</td><td>+60.0/+7.5</td><td>+61.5/+7.1</td><td>1/4</td></tr><tr><td>Qwen3-235B</td><td>-0.1/+7.0</td><td>-21.1/+0.0</td><td>-27.1/+3.7</td><td>-7.0/+15.5</td><td>3/4</td></tr><tr><td>DeepSeek-V3.2</td><td>+31.6/-9.1</td><td>-11.5/-15.4</td><td>+46.2/-23.1</td><td>-17.1/-23.0</td><td>2/4</td></tr><tr><td>Sign disagree / both defined</td><td>5/13</td><td>6/13</td><td>3/13</td><td>3/13</td><td></td></tr><tr><td>Reporting cells</td><td>13</td><td>13</td><td>13</td><td>13</td><td></td></tr></table>

Table 6: Full 17-model coding-domain sign-disagreement panel. Each cell reports aggregate retrieval lift / RAE in percentage points. Bold entries indicate opposite signs between the two metrics. Dashes indicate non-reportable cells, typically because the retrieval-invoked subset is empty, below the reporting threshold, or aggregate retrieval lift is undefined. Non-reportable cells are retained for transparency but excluded from sign-disagreement denominators.

A sign disagreement is counted when the aggregate retrieval lift and RAE have opposite signs:

$$
\Delta _ { \mathrm { a g g } } \cdot \Delta _ { \mathrm { c o n d } } < 0\tag{16}
$$

This criterion is metric-level. It should not be confused with task-level helpful and harmful transitions, which are the paired skill-enabled/skilldisabled outcome events aggregated by RAE.

We use $n _ { \mathrm { c o n d } } \geq$ 10 as the main reporting cutoff. Table 5 shows that the aggregate-vs-RAE signdisagreement counts remain stable under alternative cutoffs of 5, 15, and 20.

## A.1 Representative Reversal Statistics

Table 8 reports paired transition counts and uncertainty for representative coding-domain reversal cells. Confidence intervals use 20,000 task-level paired multinomial bootstrap resamples, and pvalues are from exact McNemar tests over the discordant counts.

## B Full Coding-Domain Results

Table 6 reports the full 17-model coding-domain panel underlying Table 3. The main table includes reportable cells only, whereas this appendix table retains all evaluated models for transparency. Dashes indicate non-reportable cells, typically because the retrieval-invoked subset is empty, below the reporting threshold, or one of the aggregate retrieval lift denominators is undefined. These cells are retained in the full panel but excluded from sign-disagreement denominators.

<table><tr><td>Model Agg. lift OAE RAE b C Ncond</td></tr><tr><td>Qwen2.5-7B -65.8 0 0 0</td></tr><tr><td>Qwen3-8B +0.0-10.0 -20.0 1 4 15</td></tr><tr><td>Qwen3.5-9B +0.7 -4.2 -7.1 0 14</td></tr><tr><td>Llama-3.1-8B +5.8-10.4 -9.3 16 31 162</td></tr><tr><td>Mistral-Nemo -25.6 -5.0 +0.0 0 0 2</td></tr><tr><td>Gemini-2.5-Flash-lite +13.2-18.3 -11.0 416 109</td></tr><tr><td>Gemini-2.0-Flash -7.3 -1.7 +0.0 1 1 8</td></tr><tr><td>Mistral-S-24B -64.7 -1.7 -20.0 0 1 5</td></tr><tr><td>GPT-4o-mini +31.4 +1.7 +0.0 0 0 1</td></tr><tr><td>Claude-3.5-H -24.7 -6.2 -6.5 0 2 31</td></tr><tr><td>Claude-4.5-H -4.8 -1.7 -1.1 4 5 88</td></tr><tr><td>Qwen3-32B -7.5 -9.6 -9.7 719 124</td></tr><tr><td>GLM4-32B +16.6-18.3 -14.4 936 187</td></tr><tr><td>Command-R +0.8-10.4 -5.511 19 145</td></tr><tr><td>Llama-3.3-70B +14.2 -42.1 -39.4 342 99</td></tr><tr><td>Qwen3-235B -3.2 +2.1 +0.0 0 0 24</td></tr><tr><td>DeepSeek-V3.2 -8.7 -8.3-11.0 5 22 154</td></tr></table>

Table 7: Full Math500 cross-domain results. Values are in percentage points except for $b , c ,$ and $n _ { \mathrm { c o n d } } .$ . Aggregate retrieval lift is undefined when there are no retrieved tasks or no non-retrieved tasks for comparison. Rows with very small $n _ { \mathrm { c o n d } }$ are reported for completeness but are interpreted as sparse conditional estimates.

## C Math500 Cross-Domain Results

Table 7 reports the full Math500 cross-domain replication. The table includes aggregate retrieval lift, OAE, and RAE, together with the retrieval-invoked transition counts used to compute RAE.

## D Cross-Benchmark Correlation Details

Table 9 reports a descriptive cross-benchmark correlation analysis comparing MBPP+ seed 42 against HumanEval+ seed 42 under the same coding skill pool.

We include models with sufficient retrievalinvocation coverage on both benchmarks, requiring $n _ { \mathrm { c o n d } } ~ \geq ~ 1 0$ for both MBPP+ and HumanEval+. This yields 13 models. Gemini-2.0-Flash, Llama-3.1-8B, Mistral-Nemo, and Qwen2.5-7B are excluded from this filtered correlation because at least one benchmark has fewer than 10 retrieval-invoked tasks. Table 10 lists these exclusions and the corresponding retrieval-invoked subset sizes.

Skill-disabled accuracy is used as the baseline ability control. For each benchmark cell, skilldisabled accuracy is reconstructed from the paired identity

$$
\mathrm { A c c } ^ { - } = \mathrm { A c c } ^ { + } - \mathrm { O A E } / 1 0 0 .\tag{17}
$$

Partial correlations are computed by residualizing the MBPP+ and HumanEval+ metric values against the specified skill-disabled accuracy controls and then correlating the residuals.

In this rerun, aggregate retrieval lift has the highest raw cross-benchmark correlation, and OAE has the highest skill-disabled-accuracy-controlled partial correlation. We therefore use these correlations only to bound interpretation: RAE is a protocol-conditional actual-use signal over the agent’s retrieval-invoked subset, not a benchmarkinvariant ranking of model skill-use ability.

## E Stage-Diagnostic Definitions and Values

The stage diagnostics in this appendix are descriptive proxies computed from the skill-enabled execution logs and paired outcome counts. They are not used to define RAE. Their purpose is to support the main-text observation that high invocation or retrieval coverage does not imply a positive sametask actual-use outcome.

We use the following stage proxies. Invocation is the percentage of tasks where the agent called the skill-search interface at least once. Retrieval is the percentage of tasks where at least one skill was returned. Adherence is a lexical/structural proxy indicating whether at least one retrieved-skill identifier or code-relevant pattern appears in the submitted answer. Effective is the percentage of retrievalinvoked tasks where this adherence proxy is present and the final answer is correct. Table 11 reports the resulting stage-diagnostic values for the two anchor reversal models across MBPP+ seeds. The final column reports RAE, the paired same-task outcome metric over the retrieval-invoked subset.

## F Transition-Level Annotation Details

Table 12 reports the full harmful-transition taxonomy computed over skill-disabled-correct/skillenabled-wrong retrieval-invoked executions from the DeepSeek-V3.2 and Gemini-2.5-Flash-lite coding-skill runs. The unit is a log-level paired execution, not a unique task. The anchor set includes MBPP+ partition seeds 42, 43, and 44 for each model, so the counts are used as qualitative diagnostic evidence rather than as an independent estimate of task-frequency prevalence.

Each harmful transition receives one primary failure-mode label. When multiple symptoms are present, the primary label is assigned to the most proximate visible cause of the submitted answer failure. Table 13 defines the labels. Table 12 reports the full label distribution before the category compression used in the main-text taxonomy.

<table><tr><td>Model</td><td>Benchmark</td><td>b</td><td>C</td><td>Ncond</td><td>RAE</td><td>95% CI</td><td>Exact McNemar p</td></tr><tr><td>Mistral-S-24B</td><td>MBPP+ (partition 3)</td><td></td><td>31</td><td>66</td><td>-45.5</td><td> $- 5 7 . 6 , - 3 3 . 3 ]$ </td><td> $1 . 5 4 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>DeepSeek-V3.2</td><td>MBPP+ (partition 3)</td><td>1</td><td>19</td><td>78</td><td>-23.1</td><td> $[ - 3 3 . 3 , - 1 2 . 8 ]$ </td><td> $4 . 0 1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Gemini-2.5-Flash-lite</td><td>HumanEval+</td><td>1</td><td>9</td><td>38</td><td>-21.1</td><td> $[ - 3 6 . 8 , - 7 . 9 ]$ </td><td>0.0215</td></tr><tr><td>Qwen3-235B</td><td>HumanEval+</td><td>13</td><td>2</td><td>71</td><td>+15.5</td><td> $[ + 5 . 6 , + 2 5 . 4 ]$ </td><td>0.00739</td></tr></table>

Table 8: Representative coding-domain reversal cells. RAE and confidence intervals are in percentage points. b counts SD-wrong→SE-correct transitions, c counts SD-correct→SE-wrong transitions, and $n _ { \mathrm { c o n d } }$ is the retrievalreturned paired subset size.
<table><tr><td>Metric</td><td>n</td><td>Raw Pearson r</td><td>Partial r</td><td>Single-control range</td></tr><tr><td>Aggregate retrieval lift</td><td>13</td><td>+0.746</td><td>+0.763</td><td> $+ 0 . 7 4 5  – + 0 . 7 5 5$ </td></tr><tr><td>OAE</td><td>13</td><td>+0.700</td><td>+0.830</td><td> $+ 0 . 7 3 2 - + 0 . 8 2 5$ </td></tr><tr><td>RAE</td><td>13</td><td>+0.634</td><td>+0.776</td><td> $+ 0 . 6 3 3 - + 0 . 7 4 7$ </td></tr></table>

Table 9: Descriptive cross-benchmark correlations between MBPP+ and HumanEval+. The partial r column controls for both MBPP+ and HumanEval+ skill-disabled accuracies. The single-control range reports the minimum and maximum partial correlations obtained when controlling for only one benchmark’s skill-disabled accuracy. These values are reported as diagnostic context, not as evidence that RAE is the most cross-benchmark-stable metric.

## G Adherence Annotation Details

We annotate the pooled DeepSeek-V3.2 and Gemini-2.5-Flash-lite retrieval-invoked representative results with a GPT-5.5 annotator. The annotator receives the task prompt, retrieval query, retrieved skill names and excerpts, and the skill-enabled answer. It does not receive the skill-disabled answer or the paired correctness labels. The annotation is therefore used as diagnostic evidence about skill uptake behavior, not as part of the definition of RAE.

Each skill-enabled answer receives one primary behavioral label: appropriate adherence, ignored/independent, misapplied/overapplied, format/interface failure, or unclear. For Figure 5, we also collapse these into three groups: appropriate adherence, no clear adherence, and problematic adherence. The collapsed labels are used only for visualization. Table 14 reconciles the annotated logs with the transition counts used in RAE. Table 15 reports the GPT-5.5 label distribution over discordant transitions, and Table 16 reports RAE after grouping retrieval-invoked executions by collapsed adherence labels.

## G.1 Human Audit of Annotation Reliability

We evaluate annotation reliability on a stratified sample of 50 of the 85 harmful transitions. Nine non-author annotators provide 150 labels while blinded to the GPT-5.5 label, the SD answer, the paired correctness labels, and one another’s annotations. Table 18 summarizes the main agreement results.

The following is an English translation of the full Korean instructions provided to the human annotators.

Human Annotator Instructions   
Human audit labeling guide   
(16--17 cases per person; 20--35 minutes)   
Purpose. This audit tests whether the behavioral   
labels assigned by an LLM (GPT-5.5) are reliable.   
Relabel your cases independently; we will later   
compute agreement with the LLM labels. The LLM   
labels are hidden. Do not consult other annotators.   
Each row contains one coding-task record:   
- task\_prompt: the problem given to the model   
- retrieval\_queries: the model's skill-search queries   
skills\_retrieved / skill\_text\_excerpt: the skill   
documents actually returned to the model   
on\_answer: the model's final answer   
Read the problem, retrieved skills, and final answer,   
in that order. Ask: "How did this answer handle the   
retrieved skills?"   
Enter exactly one label in human\_primary\_label:   
- adhered\_appropriate: visibly and appropriately   
uses retrieved-skill content for the task   
misapplied\_or\_overapplied: uses a retrieved skill   
incorrectly or unnecessarily   
- ignored\_or\_independent: shows no trace of using   
the retrieved skill and solves independently   
format\_or\_interface\_failure: fails in output or   
interface format, such as omitting the required   
function or returning prose instead of code   
unclear: cannot be assigned to the four labels   
above from the available evidence   
Rules.   
1. Do not judge task correctness. All cases are   
skill-enabled executions that failed; label only   
how the answer handled the retrieved skill.   
2. If uncertain, choose unclear rather than forcing   
a narrower label.   
3. Work alone and only on your assigned sheet.   
Notes in human\_notes are optional.   
Enter confidence in human\_confidence\_1\_to\_5, from   
1 (guess) to 5 (certain). Low confidence is natural

<table><tr><td>Excluded model</td><td>MBPP+  $n _ { \mathrm { c o n d } }$ </td><td> $\mathbf { H } \mathbf { E } \mathbf { + } n _ { \mathrm { c o n d } }$ </td><td>Reason</td></tr><tr><td>Gemini-2.0-Flash</td><td>0</td><td>0</td><td>no retrieval-invoked conditional subset</td></tr><tr><td>Llama-3.1-8B</td><td>0</td><td>0</td><td>no retrieval-invoked conditional subset</td></tr><tr><td>Mistral-Nemo</td><td>0</td><td>0</td><td>no retrieval-invoked conditional subset</td></tr><tr><td>Qwen2.5-7B</td><td>0</td><td>0</td><td>no retrieval-invoked conditional subset</td></tr></table>

Table 10: Models excluded from the filtered cross-benchmark correlation analysis. The reporting filter requires n<sub>cond</sub> ≥ 10 on both MBPP+ and HumanEval+.
<table><tr><td>Model</td><td>Seed</td><td>Invocation</td><td>Retrieval</td><td>Adherence</td><td>Effective</td><td>RAE</td></tr><tr><td>DeepSeek-V3.2</td><td>42</td><td>100.0</td><td>100.0</td><td>18.8</td><td>11.2</td><td>-9.1</td></tr><tr><td>Gemini-2.5-Flash-lite</td><td>42</td><td>82.5</td><td>82.5</td><td>13.6</td><td>10.6</td><td>-15.6</td></tr><tr><td>DeepSeek-V3.2</td><td>43</td><td>100.0</td><td>100.0</td><td>15.0</td><td>11.2</td><td>-15.4</td></tr><tr><td>Gemini-2.5-Flash-lite</td><td>43</td><td>82.5</td><td>82.5</td><td>16.7</td><td>10.6</td><td>-15.9</td></tr><tr><td>DeepSeek-V3.2</td><td>44</td><td>97.5</td><td>97.5</td><td>19.2</td><td>12.8</td><td>-23.1</td></tr><tr><td>Gemini-2.5-Flash-lite</td><td>44</td><td>78.8</td><td>78.8</td><td>17.5</td><td>12.7</td><td>-22.2</td></tr></table>

Table 11: Stage-diagnostic values for the two anchor reversal models across MBPP+ seeds. Stage values are percentages. RAE is reported in percentage points. The adherence and effective columns are diagnostic proxies from the skill-enabled logs, whereas RAE is computed from paired skill-enabled/skill-disabled outcomes.

<table><tr><td>Model</td><td>Primary label</td><td>n</td><td>Share</td></tr><tr><td>DeepSeek-V3.2 DeepSeek-V3.2</td><td>appropriate adherence</td><td>6</td><td>13.3%</td></tr><tr><td rowspan="3">DeepSeek-V3.2 DeepSeek-V3.2</td><td rowspan="3">format/interface failure ignored/independent misapplied/overapplied</td><td>7</td><td>15.6%</td></tr><tr><td>29</td><td>64.4%</td></tr><tr><td>3</td><td>6.7%</td></tr><tr><td rowspan="3">Gemini-2.5-Flash-lite Gemini-2.5-Flash-lite Gemini-2.5-Flash-lite</td><td>appropriate adherence</td><td>4</td><td>10.0%</td></tr><tr><td>format/interface failure</td><td>1</td><td>2.5%</td></tr><tr><td>ignored/independent</td><td>29</td><td>72.5%</td></tr><tr><td>Gemini-2.5-Flash-lite</td><td>misapplied/overapplied</td><td>5</td><td>12.5%</td></tr><tr><td>Gemini-2.5-Flash-lite</td><td>unclear</td><td>1</td><td>2.5%</td></tr></table>

Table 12: Full harmful-transition primary-label summary before the category compression used in the main text. The unit is a skill-disabled-correct/ skill-enabledwrong retrieval-invoked execution.

for unclear cases.   
Submission. Fill human\_primary\_label and   
human\_confidence\_1\_to\_5 in your assigned   
audit\_annotator\_A#.csv file, save it, and return it.

Agreement is modest under the original fiveway taxonomy, indicating that the fine-grained category boundaries are interpretive. The binary engaged/not engaged distinction is more stable, with 84% human–human and human-majority– GPT agreement. We therefore use the labels as descriptive post-hoc diagnostics and limit our conclusion to the low prevalence of appropriate adherence among the audited harmful transitions.

## H Skill-Content Control Details

Table 17 reports the pooled raw counts underlying Table 4. The three Normal cells use distinct 80-task partitions, whereas the control replicates reuse the same 80-task partition; pooled arm values are therefore descriptive. On shared tasks, the Gemini Normal–Filler RAE differences are −7.7, −14.0, and −3.8 pp across the three control replicates, with only the second interval excluding zero. DeepSeek Normal–Filler differences are +4.1, +4.1, and 0.0 pp. These heterogeneous comparisons do not identify a single content mechanism. Schema-Empty has $n _ { \mathrm { c o n d } } = 0$ because the interface returns no usable skill, so RAE is undefined while OAE remains defined over the whole benchmark.

<table><tr><td>Failure mode</td><td>Definition</td></tr><tr><td>Appropriate adherence</td><td>The skill-enabled answer visibly uses relevant retrieved-skill con-</td></tr><tr><td>Ignored/independent</td><td>tent or patterns in a plausible way. The skill-enabled answer re- trieves a skill but shows no clear uptake of the returned content.</td></tr><tr><td>Misapplied/overapplied</td><td>The skill-enabled answer uses retrieved-skill content but applies it in a way that changes or over-</td></tr><tr><td>Format/interface failure</td><td>constrains the task solution. The skill-enabled answer fails at the output or benchmark- interface level, such as leaking tool traces, omitting the required function, or returning malformed</td></tr><tr><td>Unclear</td><td>code. The annotator cannot assign a stable narrower behavioral label from the available evidence.</td></tr></table>

Table 13: Primary failure-mode definitions used for harmful-transition taxonomy.

<table><tr><td>Model</td><td>b c 1</td><td>Ncond</td></tr><tr><td>DeepSeek-V3.2</td><td>8 45</td><td>233</td></tr><tr><td>Gemini-2.5-Flash-lite 5 40</td><td></td><td>196</td></tr></table>

Table 14: Annotation-count reconciliation. b and c are helpful and harmful retrieval-invoked transitions; counts match the RAE computation.

## I Retrieval-Configuration Sensitivity

We rerun three representative models on the same 80 MBPP+ tasks with retrieval depths $k \in \{ 1 , 3 , 5 \}$ and with a mechanically merged single-skill library. All runs use one generic agent role, temperature 0.7, maximum output length 8,192, reasoning disabled, and at most three agent turns. Confidence intervals use 20,000 paired multinomial bootstrap resamples; p-values are within-configuration exact McNemar tests.

Across k = 1, 3, 5, Gemini remains negative with significant within-configuration tests, DeepSeek remains negative with significance only at k = 1, and Claude remains near zero. The repeated direction for Gemini and DeepSeek rules out the default k = 3 choice as the sole explanation, while the merged-1 results remain jointly affected by lower coverage.

## J Skill Pool Details

Tables 20 and 21 summarize the fixed procedural skill pools used in the coding and mathematical evaluations. Skills are stored as Markdown files following the AgentSkill-style structure described in §4.2. The retrieval index uses the skill descriptions and Markdown bodies, and the top retrieved skill texts are appended to the agent context when retrieval is invoked.

The Math skill pool is summarized in Table 21.

## K Model Panel

Table 22 lists the model panel used in our experiments. The coding panel consists of the 17 models (Claude 3.5 Haiku<sup>1</sup>, Claude Haiku 4.5<sup>2</sup>, Command $\mathbb { R } ^ { 3 }$ , DeepSeek V3.2, Gemini 2.0 Flash<sup>4</sup>, Gemini 2.5 Flash-Lite, Llama 3.1 8B Instruct, Llama

![](images/226800db17a11d4cd8456e8b5df0ad7695418c2de29ce8fd9ac79e3d0135fee3.jpg)  
Appropriate adherence Format / interface Misapplied / overapplied

(a) GPT-5.5 adherence labels over helpful and harmful transitions.  
![](images/ea03488679179fd8d4309ddccbbb6bf5efce255cf216dfae13064c0113cb1103.jpg)  
(b) Label-conditioned RAE over retrieval-invoked executions.  
Figure 5: Adherence-conditioned diagnosis of retrievalinvoked outcome changes in the pooled DeepSeek-V3.2 and Gemini-2.5-Flash-lite runs. Panel (a) shows how helpful and harmful transitions are distributed across GPT-5.5 behavioral labels. Panel (b) reports RAE after grouping executions by collapsed adherence labels.

3.3 70B Instruct, Mistral NeMo<sup>5</sup>, Mistral Small 3.2 24B Instruct<sup>6</sup>, GPT-4o mini<sup>7</sup>, Qwen2.5 7B Instruct, Qwen3 235B A22B 2507, Qwen3 32B, Qwen3 8B, Qwen3.5 9B, GLM4-32B<sup>8</sup>) (Liu et al., 2025; Comanici et al., 2025; Grattafiori et al., 2024; Yang et al., 2025; Qwen Team, 2026; Team, 2024) evaluated on MBPP+ and HumanEval+. The Math500 cross-domain replication uses the same 17-model panel, with reportability varying according to retrieval-invoked coverage. Model identifiers correspond to the API-facing model names used in the experiment logs.

<table><tr><td>Model</td><td>Transition</td><td>Primary label</td><td>Share</td></tr><tr><td>DeepSeek-V3.2</td><td>Helpful</td><td>appropriate adherence</td><td>50.0%</td></tr><tr><td>DeepSeek-V3.2</td><td>Helpful</td><td>ignored/independent</td><td>50.0%</td></tr><tr><td>DeepSeek-V3.2</td><td>Harmful</td><td>appropriate adherence</td><td>13.3%</td></tr><tr><td>DeepSeek-V3.2</td><td>Harmful</td><td>format/interface failure</td><td>15.6%</td></tr><tr><td>DeepSeek-V3.2</td><td>Harmful</td><td>ignored/independent</td><td>64.4%</td></tr><tr><td>DeepSeek-V3.2</td><td>Harmful</td><td>misapplied/overapplied</td><td>6.7%</td></tr><tr><td>Gemini-2.5-Flash-lite</td><td>Helpful</td><td>ignored/independent</td><td>100.0%</td></tr><tr><td>Gemini-2.5-Flash-lite</td><td>Harmful</td><td>appropriate adherence</td><td>10.0%</td></tr><tr><td>Gemini-2.5-Flash-lite</td><td>Harmful</td><td>format/interface failure</td><td>2.5%</td></tr><tr><td>Gemini-2.5-Flash-lite</td><td>Harmful</td><td>ignored/independent</td><td>72.5%</td></tr><tr><td>Gemini-2.5-Flash-lite</td><td>Harmful</td><td>misapplied/overapplied</td><td>12.5%</td></tr><tr><td>Gemini-2.5-Flash-lite</td><td>Harmful</td><td>unclear</td><td>2.5%</td></tr></table>

Table 15: GPT-5.5 adherence annotation distribution over discordant retrieval-invoked transitions. The annotator is blinded to the skill-disabled answer and correctness labels.
<table><tr><td>Model</td><td>Collapsed label</td><td>b</td><td>c</td><td>n</td><td>RAE</td></tr><tr><td>DeepSeek-V3.2</td><td>appropriate</td><td>4</td><td>6</td><td>89</td><td>-2.2</td></tr><tr><td>DeepSeek-V3.2</td><td>no clear</td><td>4</td><td>31</td><td>129</td><td>-20.9</td></tr><tr><td>DeepSeek-V3.2</td><td>problematic</td><td>0</td><td>8</td><td>15</td><td>-53.3</td></tr><tr><td>Gemini-2.5-Flash-lite</td><td>appropriate</td><td>0</td><td>4</td><td>26</td><td>-15.4</td></tr><tr><td>Gemini-2.5-Flash-lite</td><td>no clear</td><td>5</td><td>31</td><td>152</td><td>-17.1</td></tr><tr><td>Gemini-2.5-Flash-lite</td><td>problematic</td><td>0</td><td>5</td><td>18</td><td>-27.8</td></tr></table>

Table 16: Label-conditioned RAE values using collapsed GPT-5.5 adherence labels. Values are in percentage points. These diagnostic subsets do not redefine RAE; they describe which behavioral labels account for helpful and harmful transitions in the retrieval- invoked representative results.

## L Agent Prompts and Tool Schema

This appendix reports the prompts and tool schema used in the paired SE and SD executions and in the adherence annotation. The SE and SD prompts are kept identical except for the availability of the search\_skills interface. The annotator prompt is used only for post hoc diagnostic labeling and is not used in the computation of RAE.

Execution settings. Generation uses temperature 0.7, a maximum output length of 8,192 tokens, provider-side reasoning disabled, and at most three sequential search\_skills calls. Each valid call returns the top-3 BM25 matches with their full skill text. Invalid arguments return no skill and generation continues; when several calls are emitted together, only the first is executed. No manual context truncation is applied. Transient requests are retried up to three times, and unrecoverable provider or context-length failures produce an empty prediction scored as incorrect.

Answer extraction and correctness. For coding tasks, the evaluator extracts the first Python-fenced block, falls back to the first generic fenced block, and otherwise evaluates the raw response. The extracted program is executed against the EvalPlus extended test suite and is correct only when every test passes. Math answers are evaluated with math-verify, followed by boxed-answer parsing and normalized exact matching.

<table><tr><td>Model</td><td>Condition</td><td>b/c</td><td> $\scriptstyle n _ { \mathbf { c o n d } }$ </td><td>RAE</td><td>OAE</td><td>Coverage</td><td>Agg. lift</td></tr><tr><td>DeepSeek-V3.2</td><td>Normal</td><td>8/45</td><td>233</td><td>-15.9</td><td>-15.8</td><td>97.1</td><td>+21.2</td></tr><tr><td>DeepSeek-V3.2</td><td>Schema-Empty</td><td>0/0</td><td>0</td><td></td><td>-10.0</td><td>0.0</td><td></td></tr><tr><td>DeepSeek-V3.2</td><td>Filler-Dummy</td><td>7/30</td><td>226</td><td>-10.2</td><td>-10.8</td><td>94.2</td><td>+28.0</td></tr><tr><td>DeepSeek-V3.2</td><td>Random-skilis</td><td>5/41</td><td>238</td><td>-15.1</td><td>-15.4</td><td>99.2</td><td>+7.1</td></tr><tr><td>DeepSeek-V3.2</td><td>Corrupted</td><td>8/29</td><td>230</td><td>-9.1</td><td>-9.6</td><td>95.8</td><td>+43.5</td></tr><tr><td>Gemini-2.5-Flash-lite</td><td>Normal</td><td>5/40</td><td>196</td><td>-17.9</td><td>-17.9</td><td>81.7</td><td>+7.8</td></tr><tr><td>Gemini-2.5-Flash-lite</td><td>Schema-Empty</td><td>0/0</td><td>0</td><td></td><td>-15.4</td><td>0.0</td><td></td></tr><tr><td>Gemini-2.5-Flash-lite</td><td>Filler-Dummy</td><td>5/14</td><td>181</td><td>-5.0</td><td>-7.1</td><td>75.4</td><td>+8.1</td></tr><tr><td>Gemini-2.5-Flash-lite</td><td>Random-skills</td><td>2/32</td><td>195</td><td>-15.4</td><td>-17.9</td><td>81.2</td><td>+14.0</td></tr><tr><td>Gemini-2.5-Flash-lite</td><td>Corrupted</td><td>5/36</td><td>180</td><td>-17.2</td><td>-17.5</td><td>75.0</td><td>+1.1</td></tr></table>

Table 17: Detailed skill-content control results pooled over MBPP+ seeds 42–44. Values for RAE, OAE, coverage, and aggregate retrieval lift are in percentage points. Coverage is the percentage of tasks where retrieval returned at least one skill. Aggregate retrieval lift is undefined when retrieved or non-retrieved task populations are empty.

<table><tr><td>Audit statistic</td><td>Result</td></tr><tr><td>Five-way Krippendorff&#x27;s α</td><td>0.377</td></tr><tr><td>Five-way human-majority-GPT agreement</td><td>52.4% (22/42)</td></tr><tr><td>Cases without a strict five-way majority</td><td>8/50</td></tr><tr><td>Binary engaged/not engaged human-human agreement</td><td>84%</td></tr><tr><td>Binary engaged/not engaged human-majority-GPT agreement</td><td>84% (41/49)</td></tr><tr><td>Binary Krippendorff&#x27;s α</td><td>0.401</td></tr><tr><td>Human / GPT appropriate adherence rate</td><td>9.3% / 10.0%</td></tr></table>

Table 18: Human-audit agreement for the GPT-5.5 annotations. The five labels are appropriate adherence, ignored/independent, misapplied/overapplied, format/interface failure, and unclear. For the binary analysis, appropriate adherence and misapplied/overapplied are grouped as engaged, while ignored/independent, format/interface failure, and unclear are grouped as not engaged.

<table><tr><td>Model</td><td>Configuration</td><td>b</td><td>C</td><td> $\scriptstyle n _ { \mathbf { c o n d } }$ </td><td>Coverage</td><td>RAE</td><td>95% CI</td><td>Exact p</td></tr><tr><td>Gemini-2.5-FL</td><td>k = 1</td><td>0</td><td>9</td><td>65</td><td>81.2</td><td>-13.8</td><td>[−23.1, −6.2]</td><td>0.00391</td></tr><tr><td></td><td>k = 3 (paper)</td><td></td><td>2 12</td><td>64</td><td>80.0</td><td>-15.6</td><td>-26.6, −4.7]</td><td>0.0129</td></tr><tr><td></td><td>k = 5</td><td>1</td><td>9</td><td>70</td><td>87.5</td><td>-11.4</td><td>-20.0, −2.9]</td><td>0.0215</td></tr><tr><td></td><td>merged-1</td><td>0</td><td>2</td><td>23</td><td>28.7</td><td>-8.7</td><td>[−21.7, +0.0]</td><td>0.500</td></tr><tr><td>DeepSeek-V3.2</td><td>k = 1</td><td></td><td>0 19</td><td>77</td><td>96.2</td><td>-24.7</td><td>[−35.1, −15.6]</td><td>3.81×10−⁶</td></tr><tr><td></td><td>k = 3 (paper)</td><td></td><td>3 10</td><td>77</td><td>96.2</td><td>-9.1</td><td>[−18.2, +0.0]</td><td>0.0923</td></tr><tr><td></td><td>k = 5</td><td></td><td>4 11</td><td>77</td><td>96.2</td><td>-9.1</td><td>[−19.5, +0.0]</td><td>0.118</td></tr><tr><td></td><td>merged-1</td><td>4</td><td>6</td><td>71</td><td>88.8</td><td>-2.8</td><td>[−11.3, +5.6]</td><td>0.754</td></tr><tr><td>Claude-4.5-H</td><td>k = 1</td><td>1</td><td>2</td><td>72</td><td>90.0</td><td>-1.4</td><td>[−6.9, +2.8]</td><td>1.000</td></tr><tr><td></td><td>k = 3 (paper)</td><td>3</td><td>2</td><td>70</td><td>87.5</td><td>+1.4</td><td>-4.3, +7.1]</td><td>1.000</td></tr><tr><td></td><td>k = 5</td><td>4</td><td>4</td><td>70</td><td>87.5</td><td>+0.0</td><td>[−8.6, +7.1]</td><td>1.000</td></tr><tr><td></td><td>merged-1</td><td>0</td><td>3</td><td>19</td><td>23.8</td><td>-15.8</td><td> $[ - 3 1 . 6 , + 0 . { \dot { 0 } } ]$ </td><td>0.250</td></tr></table>

Table 19: Retrieval-depth and merged-library sensitivity. Coverage is the percentage of tasks for which at least one skill was returned; RAE and confidence intervals are in percentage points. The merged-1 intervention changes both skill granularity and retrieval coverage, so it is not interpreted as a pure granularity effect.

<table><tr><td>Coding skill</td><td>Summary</td></tr><tr><td>binary-search-boundary</td><td>Find a threshold in a sorted or monotonic search space.</td></tr><tr><td>counting-hash</td><td>Count occurrences with a hash map for frequency, duplicate, or</td></tr><tr><td>dfs-iterative</td><td>anagram-style tasks. Traverse graphs iteratively for reachability, connected compo-</td></tr><tr><td>edge-case-guard</td><td>nents, or cycle-safe search. Add defensive checks for empty, None, singleton, and boundary in-</td></tr><tr><td>knapsack-01-dp</td><td>puts. Use one-dimensional bottom-up dynamic programming for 0/1</td></tr><tr><td>prefix-sum</td><td>knapsack-style constraints. Build prefix sums for constant-</td></tr><tr><td>rotate-left</td><td>time contiguous range queries. Rotate a sequence left by a speci- fied offset with wrap-around be-</td></tr><tr><td></td><td>havior. sliding-window-longest Find the longest contiguous win- dow satisfying a monotonic pred-</td></tr><tr><td>two-pointer-pair-sum</td><td>icate. Find pair or closest-sum patterns in sorted or sortable sequences.</td></tr></table>

Table 20: Coding skill pool used for MBPP+ and HumanEval+. The pool contains 9 procedural skills.

<table><tr><td>Math skill</td><td>Summary</td></tr><tr><td>algebraic-substitution</td><td>Substitute known values into an expression and simplify.</td></tr><tr><td>case-analysis</td><td>Split a problem into mutually exclusive cases such as sign, parity, range, or absolute-value</td></tr><tr><td>check-by-substitution</td><td>branches. Verify a derived answer by sub- stituting it back into the original</td></tr><tr><td>coordinate-distance</td><td>equations or constraints. Compute distance, midpoint, slope, or angle from coordinate</td></tr><tr><td>factor-then-solve</td><td>representations. Factor polynomial equations and solve by setting factors to zero.</td></tr><tr><td>modular-arithmetic</td><td>Work with remainders, divisibil- ity, cyclic patterns, and large-</td></tr><tr><td>sequence-formulae</td><td>power computations. Apply standard closed forms for arithmetic, geometric, and tele-</td></tr><tr><td>systematic-enumeration</td><td>scoping sequences. Count or list candidates by traversing a bounded finite search space.</td></tr></table>

Table 21: Math skill pool used for the Math500 cross-domain replication. The pool contains 8 procedural skills.

<table><tr><td>Paper alias</td><td>Model identifier</td><td>Provider / family</td><td>Access type</td></tr><tr><td>Claude-3.5-H</td><td>anthropic/claude-3.5-haiku</td><td>Anthropic Claude</td><td>Closed/API</td></tr><tr><td>Claude-4.5-H</td><td>anthropic/claude-haiku-4.5</td><td>Anthropic Claude</td><td>Closed/API</td></tr><tr><td>Command-R</td><td>cohere/command-r-08-2024</td><td>Cohere Command-R</td><td>Closed/API</td></tr><tr><td>DeepSeek-V3.2</td><td>deepseek/deepseek-v3.2</td><td>DeepSeek</td><td>Open-weight family</td></tr><tr><td>Gemini-2.0-Flash</td><td>google/gemini-2.0-flash-001</td><td>Google Gemini</td><td>Closed/API</td></tr><tr><td>Gemini-2.5-Flash-lite</td><td>google/Gemini-2.5-Flash-lite</td><td>Google Gemini</td><td>Closed/API</td></tr><tr><td>Llama-3.1-8B</td><td>meta-1lama/1lama-3.1-8b-instruct</td><td>Meta Llama</td><td>Open-weight family</td></tr><tr><td>Llama-3.3-70B</td><td>meta-llama/1llama-3.3-70b-instruct</td><td>Meta Llama</td><td>Open-weight family</td></tr><tr><td>Mistral-Nemo</td><td>mistralai/mistral-nemo</td><td>Mistral AI</td><td>Open-weight family</td></tr><tr><td>Mistral-S-24B</td><td>mistralai/mistral-small-3.2-24b-instruct</td><td>Mistral AI</td><td>Open-weight family</td></tr><tr><td>GPT-4o-mini</td><td>openai/gpt-4o-mini</td><td>OpenAI GPT</td><td>Closed/API</td></tr><tr><td>Qwen2.5-7B</td><td>qwen/qwen-2.5-7b-instruct</td><td>Qwen</td><td>Open-weight family</td></tr><tr><td>Qwen3-235B</td><td>qwen/qwen3-235b-a22b-2507</td><td>Qwen</td><td>Open-weight family</td></tr><tr><td>Qwen3-32B</td><td>qwen/qwen3-32b</td><td>Qwen</td><td>Open-weight family</td></tr><tr><td>Qwen3-8B</td><td>qwen/qwen3-8b</td><td>Qwen</td><td>Open-weight family</td></tr><tr><td>Qwen3.5-9B</td><td>qwen/qwen3.5-9b</td><td>Qwen</td><td>Open-weight family</td></tr><tr><td>GLM4-32B</td><td> $z \mathrm { - } \mathsf { a i } / \mathsf { g l m } \mathrm { - } 4 \mathrm { - } 3 2 \mathsf { b }$ </td><td>Z-AI GLM</td><td>Open-weight family</td></tr></table>

Table 22: Model panel used in the experiments. Access type is used only to describe the diversity of the evaluated panel; all models were accessed through the same OpenAI-compatible experiment interface.

```jsonl
SE Agent Prompt SEARCH_SKILLS_TOOL = {
You are a helpful assistant. 2 " type ": " function ",
3 " function ": {
" name ": " search_skills ",
You have a `search_skills` tool that searches a
5 " description ": (
skill library.
6 " Search the skill library
for reusable problem - solving
Use it if you think it could help solve the task.
artifacts . "
" Returns utility functions
Implement the requested function
callable helpers you can use
with the exact signature. directly ),
Reply inside a
```python block — only the function, 8 " procedure templates (
algorithm skeletons you can
no test code.
specialize ), and "
9 " guard checklists ( defensive
SD Agent Prompt checks to add at function entry )."
You are a helpful assistant. 11 10 ) , " parameters ": {
12 " type ": " object ",
Implement the requested function with the 13 " properties ": {
exact signature. 14 " query ": {
15 " type ": " string ",
Reply inside a ```python block 16 " description ": 11
— only the function, no test code.
Natural - language query describing
the subproblem or pattern you need ."
GPT-5.5 Annotator Prompt
17 } ,
You annotate whether an LLM answer " types ": {
18
substantively follows retrieved skill content. 19 " type ": " array ",
20 " items ": {" type ": "
Do not infer from task correctness; string ", " enum ": [" utility ", 11
no correctness labels are provided. procedure ", " guard "]} ,
21 " description ": (
Return one compact JSON object only. 22 " Optional filter
. Omit for all types . Use
23 "[’ utility ’, ’
procedure ’] to skip defensive - only
skills ."
24 ) ,
25 } ,
26 } ,
27 " required ": [" query "],
28 } ,
29 } ,
30 }
```  
Listing 1: Search skill tool schema