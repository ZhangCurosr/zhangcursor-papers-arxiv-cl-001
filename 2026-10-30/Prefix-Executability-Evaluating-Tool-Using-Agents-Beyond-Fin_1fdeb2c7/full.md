# Prefix Executability: Evaluating Tool-Using Agents Beyond Final Success

Amir H. Rezaeian<sup>∗</sup>, Weiyi Sun, Yassine Benajiba, Dan Roth Oracle AI

## Abstract

Tool-use benchmarks often score agents by final task success, but terminal outcomes can obscure whether the intermediate tool-call trajectory was executable. We propose prefix executability as a trajectory-level evaluation protocol for toolusing agents. The protocol measures whether every prefix of a predicted tool-call sequence remains executable under tool schemas, state preconditions, and data dependencies, and summarizes reliability using Prefix Executability Curves (PEC), First-Error Depth (FED), Trajectory Executability Rate (TER), horizon-normalized AUC-PEC, and Recovery Rate (RR). We instantiate the protocol on stratified samples from BFCL-v3 multi-turn and ToolSandbox, and on the complete Airline/Retail base splits of τ<sup>3</sup>-bench. We observe that outcome-only and prefix-level rankings diverge: agents whose execution is conditioned on an unverified plan often achieve higher point-estimate final success than non-plan-conditioned baselines while substantially lowering TER and AUC-PEC. These divergences are precisely the failure modes that prefix executability is designed to expose: they reveal whether apparent success rests on an executable trajectory, where invalidity first enters, and whether success follows recovery from an earlier failure. As a validation intervention, explicit pre-execution plan checking can improve executable-prefix survival while maintaining competitive terminal success. These results show that final success alone is insufficient for evaluating tool-use agents and that prefixlevel executability provides an actionable reliability signal for comparing agents, diagnosing failures, and assessing validation mechanisms.

## 1 Introduction

Tool-using language agents operate over APIs, user-defined tools, and interactive software environments where correctness depends on more than a final answer. A valid run must maintain an executable sequence of intermediate actions while respecting tool schemas, cross-step dependencies, and evolving environment state [1–4]. This requirement is especially acute in multi-turn settings, where an early invalid call can corrupt later state or force the agent into recovery behavior that is invisible to outcome-only scoring [3–6]. Consider a service workflow in which an agent first retrieves a record, then passes the returned identifier to update, payment, or notification tools. A later call may be syntactically well formed but non-executable if it references a stale identifier, skips a required precondition, or applies an update before the corresponding state exists; a subsequent repair can still produce the desired final state while hiding where the trajectory first became invalid.

Recent evaluation work has moved beyond simple terminal accuracy. T-Eval decomposes tool use into step-wise subskills [7]; ToolSandbox evaluates task-specified intermediate and final conditions and checks for prohibited states or events [4]; TRAJECT-Bench scores tool selection, argument correctness, and dependency ordering over full trajectories [8]; ToolPRMBench and AgentProcess-Bench assess local process quality [9, 10]; and procedure-aware evaluation shows that nominally successful runs can still violate task procedures [11]. TRACE adds reference-free assessment of efficiency, hallucination, and adaptivity over tool-use trajectories [12]. Robustness studies further probe query perturbations, tool-set changes, and injected execution faults [13, 14]. These efforts make a remaining gap clear: standard metrics still rarely quantify how long a naturally generated rollout remains executable before its first invalid prefix, or how often final success is achieved only after entering a failure regime.

Prefix executability complements these evaluations by checking the execution feasibility of the actual emitted rollout under explicit tool and state semantics. It credits alternative executable action sequences without requiring one gold path and records first failure, survival, and recovery. Appendix A compares what prior evaluation protocols measure.

We study prefix-level executability failures: trajectory prefixes that may look plausible as text but are non-executable under the available tool schema, explicit preconditions, cross-step references, or current environment state. We argue that prefix executability is a missing reliability axis because final success and reference-trajectory match conflate three questions: how long the rollout remains valid, where validity first breaks, and whether later actions recover from the break. We therefore formalize executable prefixes and report a metric family centered on Prefix Executability Curves (PEC), First-Error Depth (FED), horizon-normalized AUC-PEC, Trajectory Executability Rate (TER), and Recovery Rate (RR). These metrics primarily diagnose reliability; when reported alongside trajectory length, latency, or monetary cost, they also support efficiency-sensitive analysis by distinguishing short, valid rollouts from long or failure-prone ones.

These distinctions are operationally important in cost-sensitive tool use. In workflows that trigger external transactions, consume scarce resources, modify persistent state, or require human escalation, the depth of the first invalid action can determine both the cost of failure and the feasibility of recovery. Prefix executability therefore provides a practical evaluation signal for deciding not only whether an agent eventually succeeds, but also whether execution remains uninterrupted or requires recovery along the way.

The paper is evaluation-first. We deliberately use a minimal Dual-Loop planner-verifier intervention only as a controlled probe: we test whether pre-execution validation improves those metrics even when final success rankings tell a different story. This framing separates the evaluation contribution from the particular intervention and lets us ask when plan conditioning helps tool use, and when it amplifies invalid intermediate structure.

## Our contributions are:

1. We formalize executable prefixes for tool-using agents and define PEC, FED, TER, AUC-PEC, and RR as a prefix-level evaluation protocol for variable-horizon trajectories. Appendix B establishes exact properties and consistency under explicit sampling assumptions.

2. We instantiate the protocol on BFCL v3 multi-turn and ToolSandbox with stratified 200-example samples, and on all 164 Airline/Retail base-split tasks in $\tau ^ { 3 }$ -bench, using shared task IDs across methods and models, and task-level confidence intervals.

3. We show that outcome-only and prefix-level rankings can diverge: unverified plan conditioning can improve final success while sharply reducing executability, whereas pre-execution validation consistently improves TER, AUC-PEC, and early PEC across three LLM conditions on BFCL v3 and ToolSandbox. A representative BFCL setting illustrates the pattern: an unverified planconditioned executor can reach the correct terminal outcome while emitting invalid intermediate tool-call prefixes, whereas Dual-Loop preserves terminal performance while restoring prefix executability. The τ<sup>3</sup>-bench audit further demonstrates that prefix reliability and terminal success can rank methods differently.

We evaluate BFCL v3 and ToolSandbox because together they cover multi-step function composition and stateful, on-policy tool interaction [3, 4]. The Airline/Retail audit evaluates the protocol in policy-constrained user–agent–environment interaction in $\tau ^ { 3 }$ -bench. Together, the three benchmarks cover 564 task instances per model–method condition and 6,768 rollouts.

## 2 Prefix-Level Executability Evaluation

## 2.1 Executable Prefixes

We model an agent rollout as a predicted trajectory $\hat { \tau } = ( a _ { 1 } , \dots , a _ { T } )$ , where each ordinary action $a _ { t }$ is a tool invocation with arguments. Let $s _ { t }$ denote the environment state immediately before executing $a _ { t } .$ . An action is executable if the selected tool is available, the arguments conform to its schema, all explicit preconditions hold in $s _ { t } .$ , and any referenced intermediate values are present and type-consistent.

A prefix $\hat { \tau } _ { \le k } = ( a _ { 1 } , \dots , a _ { k } )$ is executable if every action in the prefix executes successfully under the states induced by prior actions. This definition evaluates feasibility rather than agreement with a single reference trajectory, which matters because many tool tasks admit multiple valid execution orders or alternative executable call sequences [3, 4, 6].

The benchmark-side scorer is deterministic and uses no LLM judge. It checks exposed tools, argument binding and validators, current and type-compatible references, information availability, explicit preconditions, and benchmark acceptance of the transition. Accepted actions advance state; the first rejection fixes FED. Prefix replay then stops, while the full rollout is retained for final success and recovery. This scorer is distinct from the LLM-based pre-execution verifier. Appendix C.1 specifies the procedure and benchmark-specific semantics.

## 2.2 Prefix-Level Reliability Metrics

For rollout i, let $T _ { i }$ denote the number of evaluated tool-action slots and let $Y _ { i } \in \{ 0 , 1 \}$ indicate benchmark-defined final task success. For ordinary trajectories, $T _ { i }$ is the number of predicted tool invocations. If an agent terminates before emitting any tool call in a selected tool-use task, the instrumentation represents this as a length-one terminal no-tool action; that action is executable only if the benchmark permits task completion without tool use. Thus $T _ { i } \geq 1$ for all evaluated rollouts. We use $T$ and $Y$ without subscripts for the corresponding random variables, and hats denote finite-sample estimates over the evaluated rollouts.

The first primitive is First-Error Depth (FED):

$$
\mathrm { F E D } _ { i } = \operatorname* { m i n } \{ k \in \{ 1 , \ldots , T _ { i } \} : \hat { \tau } _ { i , \le k } \mathrm { { i s } n o t \ e x e c u t a b l e } \} ,\tag{1}
$$

with $\mathrm { F E D } _ { i } = T _ { i } + 1$ if every prefix through the full trajectory is executable. Hence, for any $k \leq T _ { i }$

$$
\hat { \tau } _ { i , \leq k } \mathrm { { i s \ e x e c u t a b l e } } \quad \Longleftrightarrow \quad \mathrm { F E D } _ { i } > k .\tag{2}
$$

We define the main Prefix Executability Curve (PEC) as a fixed-cohort survival curve over first-error depth,

$$
\operatorname { P E C } ( k ) = \operatorname* { P r } [ \operatorname { F E D } > k ] = \operatorname* { P r } [ T \geq k \land \hat { \tau } _ { \leq k } { \mathrm { ~ i s ~ e x e c u t a b l e } } ] .\tag{3}
$$

This is a discrete survival function in the usual time-to-event sense [15, 16]. We do not apply a censoring correction: a rollout that terminates before depth k is not unobserved at depth $k ,$ but instead lies outside the event $T \geq k$ in Eq. (3). The empirical estimate over $N$ evaluated rollouts is

$$
\widehat { \mathrm { P E C } } ( k ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } \{ \mathrm { F E D } _ { i } > k \} = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } \{ T _ { i } \geq k \land \hat { \tau } _ { i , \leq k } \mathrm { i s ~ e x e c u t a b l e } \} .\tag{4}
$$

The denominator is fixed at $N$ , the number of rollouts for a given dataset, model, and agent variant. Therefore $\widehat { \mathrm { P E C } } ( k )$ is monotone non-increasing in $k ;$ the rollout horizon $T _ { i }$ should not be confused with the sample size $N$

For diagnostics, we also track the eligible-set size

$$
n _ { k } = \sum _ { i = 1 } ^ { N } \mathbf { 1 } \{ T _ { i } \geq k \} ,\tag{5}
$$

and, when $n _ { k } > 0$ , the conditional eligible-depth quantity

$$
\widehat { \mathrm { P E C } } _ { \mathrm { e l i g } } ( k ) = \frac { 1 } { n _ { k } } \sum _ { i = 1 } ^ { N } \mathbf { 1 } \{ T _ { i } \geq k \land \mathrm { F E D } _ { i } > k \} .\tag{6}
$$

This conditional curve answers a different question: among rollouts with at least k tool-action slots, what fraction have an executable prefix of length k? Because the eligible set changes with k, $\widehat { \mathrm { P E C } } _ { \mathrm { e l i g } } ( k )$ need not be monotone and is not the survival curve in Eq. (3). We use Eq. (4) for the main PEC plots and tables.

The two curves satisfy the exact support decomposition

$$
{ \widehat { \operatorname { P E C } } } ( k ) = { \frac { n _ { k } } { N } } { \widehat { \operatorname { P E C } } } _ { \mathrm { e l i g } } ( k ) , \qquad n _ { k } > 0 .\tag{7}
$$

Thus fixed-cohort PEC measures the fraction of all started rollouts that reach depth k with an executable prefix; its decline can reflect either a prefix failure or termination before that depth. Eligible-depth PEC isolates validity conditional on reaching k.

The Trajectory Executability Rate (TER) is the probability that the full predicted trajectory is executable:

$$
\mathrm { T E R } = \mathrm { P r } [ \mathrm { F E D } = T + 1 ] , \qquad { \widehat { \mathrm { T E R } } } = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } \mathbf { 1 } \{ \mathrm { F E D } _ { i } = T _ { i } + 1 \} .\tag{8}
$$

When all episodes share a common fixed horizon $T _ { 0 }$ , TER equals $\mathrm { P E C } ( T _ { 0 } )$ . With variable-horizon rollouts, Eq. (8) is the unambiguous endpoint definition.

The Recovery Rate (RR) is final success conditional on having experienced at least one invalid prefix,

$$
\mathrm { R R } = \mathrm { P r } [ Y = 1 \mid \mathrm { F E D } \leq T ] , \qquad \widehat { \mathrm { R R } } = \frac { \sum _ { i = 1 } ^ { N } \mathbf { 1 } \{ Y _ { i } = 1 \} \mathbf { 1 } \{ \mathrm { F E D } _ { i } \leq T _ { i } \} } { \sum _ { i = 1 } ^ { N } \mathbf { 1 } \{ \mathrm { F E D } _ { i } \leq T _ { i } \} } ,\tag{9}
$$

when the denominator is nonzero; otherwise RR is undefined for that condition. We report this denominator as RR-Eligible. RR is closest in spirit to the Recovery Rate reported by Beyond Success Rate, but conditions on first-prefix failures observed organically in the rollout rather than externally injected API faults [14].

Finally, to compare variable-length episodes with a scalar, we report a horizon-normalized area under the executable-prefix profile:

$$
\operatorname { A U C - P E C } = \mathbb { E } \left[ { \frac { 1 } { T } } \sum _ { k = 1 } ^ { T } \mathbf { 1 } \{ \mathrm { F E D } > k \} \right] = \mathbb { E } \left[ { \frac { \mathrm { F E D } - 1 } { T } } \right] .\tag{10}
$$

The empirical estimate is

$$
{ \widehat { \mathrm { A U C - P E C } } } = { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } { \frac { { \mathrm { F E D } } _ { i } - 1 } { T _ { i } } } .\tag{11}
$$

The second equality in Eq. (10) follows because $\begin{array} { r } { \sum _ { k = 1 } ^ { T } \mathbf { 1 } \{ \mathrm { F E D } > k \} = \mathrm { F E D } - 1 } \end{array}$ under the convention $\mathrm { F E D } = \mathrm { \bar { } } T + 1$ for fully executable trajectories. AUC-PEC lies in [0, 1], equals 1 for fully executable trajectories, and rewards later first failures more than earlier ones. Table headers later omit hats for readability; captions state that reported entries are empirical estimates.

The metric family satisfies $0 \ \leq \ \mathrm { T E R } \ \leq \ \mathrm { A U C } \mathrm { { \mathrm { - } P E C } } \ \leq \ 1$ in population and sample, and $n _ { \mathrm { R R - E l i g i b l e } } = N ( 1 - \widehat { \mathrm { T E R } } )$ exactly. A fully executable trajectory contributes one to both TER and normalized area; a failing trajectory contributes zero to TER and less than one to area. Let $q = \mathrm { T E R }$ $s = \mathrm { P r } ( Y = 1 \mid \mathrm { F E D } \stackrel { \smile } { = } T + 1 )$ , and $r = { \mathrm { R R } }$ . For $0 < q < 1$

$$
\operatorname* { P r } ( Y = 1 ) = q s + ( 1 - q ) r .\tag{12}
$$

This partitions terminal success into uninterrupted execution and recovery, without requiring a monotone relationship with TER. Appendix B proves these properties and gives consistency and concentration results under i.i.d. sampling, explicitly distinguished from the finite stratified cohorts studied here.

## 2.3 What Prefix-Level Metrics Add

Table 1 summarizes the evaluation signals used in our study and how the proposed prefix-level metrics complement standard outcome and reference-match scores. Exact-call and trajectory-match metrics

measure agreement with a reference solution, but they do not localize the first invalid prefix and may penalize alternative executable trajectories. Final success captures task utility, but does not indicate whether the rollout remained executable throughout or succeeded only after an earlier invalid action. PEC, FED, TER, AUC-PEC, and RR fill this gap by measuring prefix survival, first-error depth, full-trajectory executability, normalized executable-prefix area, and recovery after prefix failure.
<table><tr><td>Metric / signal</td><td>Ref.-free</td><td>Exec.-state</td><td>Prefix signal</td><td>Recovery</td></tr><tr><td>Exact Call Match</td><td>No</td><td>No</td><td>No</td><td>No</td></tr><tr><td>Trajectory Match</td><td>No</td><td>No</td><td>No</td><td>No</td></tr><tr><td>Final Success (Y)</td><td>Yes</td><td>Yes</td><td>No</td><td>Indirect</td></tr><tr><td> $\mathrm { P E C } ( k )$ </td><td>Yes</td><td>Yes</td><td>Survival at k</td><td>No</td></tr><tr><td> $\widehat \mathrm { P E C } _ { \mathrm { e l i g } } ( k )$ </td><td>Yes</td><td>Yes</td><td>Conditional at k</td><td>No</td></tr><tr><td>TER</td><td>Yes</td><td>Yes</td><td>Full horizon</td><td>No</td></tr><tr><td>Mean FEDi</td><td>Yes</td><td>Yes</td><td>First invalid</td><td>No</td></tr><tr><td>AUC-PEC</td><td>Yes</td><td>Yes</td><td>Normalized profile</td><td>No</td></tr><tr><td>RR</td><td>Yes</td><td>Yes</td><td>Conditional</td><td>Yes</td></tr></table>

Table 1: Coverage of standard and prefix-level evaluation metrics. Ref.-free means that a metric can credit alternative executable trajectories rather than requiring agreement with a single reference path. Exec.-state means that the metric depends on environment execution or explicit validation semantics. PEC(k) is the fixed-cohort survival curve in Eq. (3); $\widehat { \mathrm { P E C } } _ { \mathrm { e l i g } } ( k )$ is the empirical eligible-depth diagnostic in Eq. (6). TER is the endpoint event $\bar { \mathrm { F E D } } = T + 1 \checkmark$ , mean $\mathrm { F E D } _ { i }$ summarizes first-error depth, AUC-PEC summarizes the horizon-normalized prefix-survival profile, and RR conditions on $\mathrm { F E D } \le T$

## 2.4 First-Failure Annotation Protocol

When $\mathrm { F E D } _ { i } \le T _ { i }$ , we annotate the action at the first invalid prefix with one primary cause: schema failure, dependency failure, precondition failure, state-consistency failure, or information failure. The labels are mutually exclusive by deterministic tie-breaking in that order, so category counts sum to the RR-Eligible set. These labels are diagnostic only; they are not used to compute PEC, TER, AUC-PEC, or RR. The taxonomy is benchmark-agnostic but reflects stressors emphasized by BFCL, ToolSandbox, NESTFUL, FAIL-TaLMs, and function-calling robustness studies [3–5, 13, 17].

## 2.5 Validation Intervention

Dual-Loop is included as a controlled validation intervention, not as the main architectural claim. Given the user goal, tool schema, and a compact state summary, a planner proposes a structured plan. A verifier checks tool existence, schema conformity, explicit preconditions, and variable bindings. If verification fails, the planner revises the plan using verifier feedback; once verification succeeds, the approved plan conditions a ReAct executor, which alone issues environment tool calls. Planner and verifier messages are therefore method-side computation, not benchmark tool actions, and all prefix metrics are computed on the executed tool trajectory. The intervention follows planner-executor and verifier-guided generation work, but here serves as a stress test for the evaluation protocol [18–22]. Appendix D gives the schematic.

## 2.6 Benchmarks and Experimental Setup

We use BFCL v3 and ToolSandbox to test multi-step function-calling correctness and stateful onpolicy interaction. BFCL v3 provides multi-turn function-calling scenarios under a standardized interface, making it a test bed for schema, ordering, and cross-step dependency errors [3]. ToolSandbox adds stateful tool execution, implicit state dependencies, an on-policy user simulator, and dynamic intermediate/final milestone evaluation, making recovery after intermediate failure observable during the rollout [4].

For each benchmark, we evaluate a stratified sample of $N = 2 0 0$ instances. The BFCL v3 subset is stratified uniformly over the four multi-turn categories used in this study: multi\_turn\_base, multi\_turn\_miss\_func, multi\_turn\_miss\_param, and multi\_turn\_long\_context, with 50 tasks from each. The ToolSandbox subset contains 200 supported open-source scenario templates selected by a deterministic greedy multi-label stratifier over scenario-category labels. Sampling is performed once per dataset, and the same task IDs are reused for every model and method, so all method comparisons are paired at the task level. The BFCL-v3/ToolSandbox grid comprises $2 0 0 \times 2 \times 3 \times 4 = 4 , 8 0 0$ evaluated rollouts across two datasets, three models, and four agent variants. We report 95% task-level confidence intervals. Marginal-interval overlap does not establish paired significance, and we do not infer paired effects from it. Compute resources for the hosted-inference and local-orchestration components of the BFCL-v3/ToolSandbox grid are reported in Appendix E.

Both benchmarks are instrumented at the action level. For each rollout, we log whether each predicted call is executable, the first failing step, its primary error label, and whether the episode later reaches benchmark-defined task success. On BFCL v3, this requires prefix-by-prefix replay of the predicted sequence rather than scoring only the terminal trace. On ToolSandbox, we derive the same quantities from executed trajectories and benchmark-provided traces, so state transitions and downstream recovery are observed directly rather than inferred post hoc.

We evaluate all 50 Airline and 114 Retail base-split tasks in $\tau ^ { 3 } .$ -bench under the text half-duplex protocol (recorded version v1.0.1). This is complete coverage of those two base splits, not all benchmark domains. One rollout per task, model, and method yields 1,968 Airline/Retail episodes. Across the three benchmarks, the main grid contains $6 { , } 7 6 8$ rollouts over 564 tasks per model–method condition. Appendix F reports the full scalar metrics, uncertainty, costs, and a separate paired verifier-sensitivity study. Task-level intervals describe variation across tasks under the fixed protocol, not repeated-run variability within a task.

## 2.7 Agent Variants and Experimental Controls

We compare a nested intervention ladder with four variants:

1. ReAct: interleaved reasoning and acting without explicit plan validation [2].

2. Prompt Planning: a planner-first variant in which a plan is generated but not enforced during execution.

3. Plan-Conditioned ReAct: the generated plan is provided as an explicit control signal during execution.

4. Dual-Loop: the same plan-conditioned executor augmented with pre-execution verification.

Within each LLM condition, executor calls share the same model setting, tool specification format, decoding strategy, and benchmark horizon cap. Planner and verifier calls occur before execution and are counted as method cost rather than tool actions in T<sub>i</sub>. Additional provider-specific decoding and rollout settings are reported in Appendix E. We report three LLM conditions: GPT-5.4 (reasoning.effort=high), GPT-4.1, and Llama-4-Maverick-Instruct.

## 3 Benchmark Results

## 3.1 Main Benchmark Results

Table 2 reports the main comparison across three LLM conditions using the stratified samples described in Section 2.6. Bracketed intervals are 95% confidence intervals. For TER, mean FED, AUC-PEC, final success, and PEC checkpoints we use 10,000 task-level percentile bootstrap resamples; each resample preserves the shared task IDs across all model-method rollouts within a dataset. RR is a conditional proportion, so its interval is computed with a Wilson score interval using the observed RR-Eligible denominator.

On BFCL-v3 and ToolSandbox, the point estimates show a consistent separation between terminal success and prefix reliability. Under GPT-5.4, Plan-Cond. ReAct reaches the highest or tied-highest final success on BFCL and ToolSandbox (0.94 and 0.69), but its prefix reliability is much weaker than Dual-Loop’s: TER is 0.47 and 0.48, and AUC-PEC is 0.54 and 0.52. Dual-Loop instead attains TER of 0.94 on both datasets and AUC-PEC of 0.96 and 0.95. The same gap remains under weaker LLMs: Dual-Loop exceeds Plan-Cond. ReAct in TER by 0.61 under GPT-4.1 and by 0.83 under Llama-4 Maverick on BFCL, and by 0.61 and 0.71 on ToolSandbox.

AUC-PEC and mean FED clarify how outcome-only and prefix-level rankings separate. On Tool-Sandbox with GPT-5.4, Plan-Cond. ReAct has the highest final success (0.69), but Dual-Loop has much higher TER, mean FED, and AUC-PEC (0.94, 4.32, and 0.95 versus 0.48, 2.88, and 0.52).

<table><tr><td></td><td></td><td></td><td></td><td>Mean</td><td></td><td>RR-</td><td></td><td>Final</td></tr><tr><td>Dataset</td><td>LLM</td><td>Method</td><td>TER</td><td>FED</td><td>AUC-PEC</td><td>Eligible</td><td>RR</td><td>Success</td></tr><tr><td>BFCL</td><td>GPT-5.4 (high)</td><td>ReAct</td><td>0.69 [0.63,0.76] 0.78</td><td>5.61 [5.28,5.92]</td><td>0.83 [0.79,0.87]</td><td>61</td><td>0.00 [0.00,0.06]</td><td>0.61 [0.55,0.68]</td></tr><tr><td></td><td></td><td>Prompt Planning</td><td>[0.72,0.83] 0.47</td><td>5.72 [5.39,6.05] 4.11</td><td>0.85 [0.81,0.89]</td><td>44</td><td>0.00 [0.00,0.08]</td><td>0.78 [0.72,0.83]</td></tr><tr><td></td><td></td><td>Plan-Cond. ReAct</td><td>[0.40,0.54] 0.94</td><td>[3.73,4.50]</td><td>0.54 [0.48,0.60]</td><td>106</td><td>0.88 [0.80,0.93]</td><td>0.94 [0.90,0.96]</td></tr><tr><td></td><td></td><td>Dual-Loop</td><td>[0.91,0.97]</td><td>6.41 [6.12,6.70]</td><td>0.96 [0.93,0.98]</td><td>12</td><td>0.00 [0.00,0.24]</td><td>0.94 [0.90,0.97]</td></tr><tr><td></td><td>GPT-4.1</td><td>ReAct</td><td>0.62 [0.55,0.69]</td><td>5.18 [4.84,5.51]</td><td>0.76 [0.72,0.81]</td><td>76</td><td>0.00 [0.00,0.05]</td><td>0.52 [0.45,0.58]</td></tr><tr><td></td><td></td><td>Prompt Planning</td><td>0.79 [0.72,0.84]</td><td>5.84 [5.54,6.16]</td><td>0.87 [0.83,0.91]</td><td>43</td><td>0.02 [0.00,0.12]</td><td>0.77 [0.71,0.82]</td></tr><tr><td></td><td></td><td>Plan-Cond. ReAct</td><td>0.35 [0.28,0.41]</td><td>3.55 [3.16,3.96]</td><td>0.42 [0.35,0.48]</td><td>130</td><td>0.90 [0.84,0.94]</td><td>0.93 [0.90,0.96]</td></tr><tr><td></td><td></td><td>Dual-Loop</td><td>0.96 [0.94,0.99]</td><td>6.46 [6.18,6.74]</td><td>0.98 [0.95,0.99]</td><td>7</td><td>0.00 [0.00,0.35]</td><td>0.96 [0.93,0.98]</td></tr><tr><td></td><td>Llama-4 Maverick ReAct</td><td></td><td>0.37 [0.30,0.44]</td><td>4.39 [4.07,4.72]</td><td>0.62 [0.57,0.67]</td><td>126</td><td>0.01 [0.00,0.04]</td><td>0.27 [0.20,0.33]</td></tr><tr><td></td><td></td><td>Prompt Planning</td><td>0.60 [0.54,0.67]</td><td>5.51 [5.17,5.84]</td><td>0.79 [0.75,0.83]</td><td>79</td><td>0.00 [0.00,0.05]</td><td>0.56 [0.48,0.62]</td></tr><tr><td></td><td></td><td>Plan-Cond. ReAct</td><td>0.05 [0.02,0.08]</td><td>1.96 [1.78,2.13]</td><td>0.15 [0.12,0.18]</td><td>190</td><td>0.73 [0.66,0.79]</td><td>0.74 [0.68,0.80]</td></tr><tr><td></td><td></td><td>Dual-Loop</td><td>0.88 [0.83,0.92]</td><td>6.22 [5.92,6.53]</td><td>0.93 [0.89,0.95]</td><td>25</td><td>0.00 [0.00,0.13]</td><td>0.79 [0.72,0.84]</td></tr><tr><td>ToolSandbox</td><td>GPT-5.4 (high)</td><td>ReAct</td><td>0.74 [0.68,0.80]</td><td>3.84 [3.63,4.04]</td><td>0.83 [0.78,0.87]</td><td>52</td><td>0.27 [0.17,0.40]</td><td>0.59 [0.52,0.66]</td></tr><tr><td></td><td></td><td>Prompt Planning</td><td>0.82 [0.77,0.88]</td><td>4.08 [3.89,4.26]</td><td>0.88 [0.84,0.92]</td><td>35</td><td>0.17 [0.08,0.33]</td><td>0.61 [0.54,0.68]</td></tr><tr><td></td><td></td><td>Plan-Cond. ReAct</td><td>0.48 [0.41,0.56]</td><td>2.88 [2.65,3.13]</td><td>0.52 [0.46,0.59]</td><td>103</td><td>0.73 [0.64,0.80]</td><td>0.69 [0.62,0.76]</td></tr><tr><td></td><td></td><td>Dual-Loop</td><td>0.94 [0.90,0.96]</td><td>4.32 [4.15,4.47]</td><td>0.95 [0.93,0.98]</td><td>13</td><td>0.15 [0.04,0.42]</td><td>0.63 [0.56,0.69]</td></tr><tr><td></td><td>GPT-4.1</td><td>ReAct</td><td>0.72 [0.66,0.78]</td><td>3.84 [3.64,4.03]</td><td>0.84 [0.80,0.88]</td><td>56</td><td>0.20 [0.11,0.32]</td><td>0.49 [0.42,0.56]</td></tr><tr><td></td><td></td><td>Prompt Planning</td><td>0.81 [0.76,0.86]</td><td>4.11 [3.93,4.29]</td><td>0.89 [0.85,0.93]</td><td>37</td><td>0.22 [0.11,0.37]</td><td>0.56 [0.49,0.63]</td></tr><tr><td></td><td></td><td>Plan-Cond. ReAct</td><td>0.35 [0.29,0.42]</td><td>2.45 [2.22,2.69]</td><td>0.40 [0.34,0.46]</td><td>129</td><td>0.59 [0.50,0.67]</td><td>0.62 [0.56,0.69]</td></tr><tr><td></td><td></td><td>Dual-Loop</td><td>0.96 [0.93,0.98]</td><td>4.38 [4.22,4.53]</td><td>0.97 [0.95,0.99]</td><td>8</td><td>0.38 [0.14,0.69]</td><td>0.62 [0.56,0.69]</td></tr><tr><td></td><td>Llama-4 Maverick ReAct</td><td></td><td>0.49 [0.42,0.56]</td><td>3.33 [3.12,3.54]</td><td>0.68 [0.63,0.74]</td><td>101</td><td>0.23 [0.16,0.32]</td><td>0.38 [0.31,0.44] 0.49</td></tr><tr><td></td><td></td><td>Prompt Planning</td><td>0.64 [0.57,0.70]</td><td>3.69 [3.50,3.89]</td><td>0.78 [0.73,0.82]</td><td>73</td><td>0.18 [0.11,0.28]</td><td>[0.42,0.56]</td></tr><tr><td></td><td></td><td>Plan-Cond. ReAct</td><td>0.09 [0.06,0.13]</td><td>1.63 [1.49,1.79]</td><td>0.15 [0.11,0.19]</td><td>182</td><td>0.57 [0.50,0.64]</td><td>0.56 [0.48,0.62]</td></tr><tr><td></td><td></td><td></td><td>0.80 [0.74,0.85]</td><td>4.13 [3.96,4.30]</td><td>0.90 [0.87,0.93]</td><td></td><td>0.35</td><td>0.61 [0.54,0.68]</td></tr></table>

Table 2: Benchmark results with 200 stratified sampled instances for BFCL multi-turn and 200 multi-label-stratified sampled instances for ToolSandbox under three LLM conditions. All entries are empirical estimates; table headers omit hats for readability. Bracketed values are 95% confidence intervals. Intervals for TER, mean FED, AUC-PEC, and final success use 10,000 task-level percentile bootstrap resamples; RR intervals use Wilson score intervals with the RR-Eligible denominator. RR-Eligible counts episodes with FED ≤ T and is reported without an interval. Bold point estimates mark column maxima within each dataset–LLM block, including ties at the displayed precision; boldface does not indicate statistical significance. Higher TER, AUC-PEC, and final success indicate stronger respective scores; mean FED must also be interpreted with rollout horizon; RR is better only conditional on entering the failure regime and should be interpreted with RR-Eligible.

Success after a prefix failure. The BFCL-v3/Llama-4 Maverick Plan-Conditioned ReAct condition makes Eq. (12) concrete. Of 200 episodes, 10 were fully executable and all 10 succeeded; 138 of the 190 episodes with a prefix failure also succeeded. Thus TER is $1 0 / 2 0 0 = 0 . 0 5$ , RR is 138/190 ≈ 0.7263, and final success is $( 1 0 + 1 3 8 ) / 2 0 0 = 0 . 7 4$ . Fully $1 \dot { 3 } 8 / 1 4 8 = 9 3 . 2 \%$ of successful episodes reached the objective after an earlier invalid prefix. This is high terminal utility with low uninterrupted-execution reliability, rather than a contradiction between metrics. Appendix C.2 illustrates how a rejected contact-removal attempt can be repaired by first retrieving the required identifier.

RR should be read together with RR-Eligible. Across the three LLMs, Dual-Loop enters the RReligible regime far less often than Plan-Cond. ReAct: on BFCL the counts are 12, 7, and 25 for

Dual-Loop versus 106, 130, and 190 for Plan-Cond. ReAct, and on ToolSandbox they are 13, 8, and 40 versus 103, 129, and 182. Dual-Loop has zero observed RR on BFCL; the small eligible sets make these conditional estimates imprecise. Plan-Conditioned ReAct combines many more prefix failures with substantial success after those failures, so its RR measures recovery rather than uninterrupted executability. The size of the eligible set does not itself determine the recovery rate.

## 3.2 Prefix Survival by Depth

Figure 1 shows the fixed-cohort PEC curves for GPT-5.4; Appendix Table G.1 provides the corresponding checkpoints for all three LLM conditions. At early depths, the BFCL-v3/ToolSandbox audit shows large fixed-cohort differences. For example, BFCL PEC@4 for Plan-Conditioned ReAct is 0.42, 0.33, and 0.05 across the three model conditions, versus 0.82, 0.82, and 0.78 for Dual-Loop. Under Eq. (7), these are joint reach-and-validity statistics. Consequently, a fixed-cohort decline need not represent a newly invalid call; depth support and conditional validity answer complementary questions.

![](images/9714763f243ddc728686596489bb60dcc25857bc51656619701e016dcf4be368.jpg)  
Figure 1: Prefix-executability curves on the 200-instance BFCL multi-turn sample and the 200- instance ToolSandbox sample using GPT-5.4 (high). Each point plots $\widehat { \mathrm { P E C } } ( k )$ with the fixed-cohort estimator in Eq. (4). The BFCL panel displays depths $k \leq 1 0 ;$ under Eq. (3), late-depth declines can reflect both prefix failures and termination. Appendix Table G.1 reports early-depth checkpoints for all LLM conditions.

## 3.3 Transfer, Cost, and the Boundary of Validation

The $\tau ^ { 3 }$ -bench audit again separates terminal outcome from prefix reliability (Table F.1). With GPT-5.4, Plan-Conditioned ReAct has the highest final-success point estimate (0.51), while ReAct has the highest TER/AUC-PEC estimates (0.95/0.95). With GPT-4.1, Prompt Planning leads final success (0.51), while ReAct leads TER/AUC-PEC (0.84/0.91). With Llama-4, ReAct, Prompt Planning, and Plan-Conditioned ReAct tie at the displayed final-success precision (0.18), despite TER spanning 0.52–0.60 and AUC-PEC 0.56–0.65. These are model-dependent point-estimate orderings, not claims of paired significance.

The Airline/Retail audit does not establish the consistent validation advantage observed on BFCL-v3 and ToolSandbox. A possible explanation is that BFCL emphasizes pre-specified composition, whereas interactive user feedback and state changes in $\tau ^ { 3 } .$ -bench can make advance plans stale. This is a testable interpretation, not an identified causal mechanism.

Measured agent-side costs make the trade-off explicit. In the four-method $\tau ^ { 3 }$ audit, Dual-Loop changes mean tokens/cumulative agent-call latency relative to Plan-Conditioned ReAct by approximately +3%/+22%, +12%/+7%, and $- 3 \% / - 1 \dot { \% }$ for GPT-5.4, GPT-4.1, and Llama-4, respectively. These totals include verification and revision as well as any downstream execution savings; they exclude simulator and evaluator calls. Table F.2 gives all four methods and accounting details.

A separate sensitivity study fixes the initial plan and GPT-5.4-high planner/executor across all 164 tasks, varying verifier model and validation budget. All paired 95% intervals for TER and AUC-PEC changes include zero. The GPT-5.4 verifier at $K _ { \mathrm { v a l } } = 5 ,$ , for example, changes TER by +0.006 [−0.018, 0.030] while adding 40.7k tokens and 176.9s relative to its paired Plan-Conditioned rerun. The deterministic-acceptance wrapper also changes executor guidance and is not an equivalent-policy no-verifier control. Table F.3 reports the full comparisons and limitations. These findings delimit the benefit of validation in this environment; they do not prove equivalence or isolate verification logic from extra computation.

## 4 Conclusion

Reliable tool use is not only a question of whether an agent eventually succeeds, but whether its trajectory remains executable as it unfolds. This paper introduced prefix executability as an evaluation protocol for tool-using agents, formalizing PEC, FED, TER, AUC-PEC, and RR for variable-horizon rollouts. Across the three-benchmark audit, prefix reliability and terminal outcome provide complementary information. Validation improves prefix reliability on BFCL-v3 and ToolSandbox, while the Airline/Retail transfer audit and paired sensitivity study show that this benefit is environmentdependent. Exact identities explain how high final success can coexist with frequent prefix failure, and cost measurements expose the operational price of validation. Evaluations should therefore report uninterrupted executability, recovery, terminal success, and agent-side cost together.

Although we report PEC, FED, TER, AUC-PEC, and RR as retrospective evaluation metrics over completed rollouts, the underlying executable-prefix predicate is prefix-local. When available during execution, it can be checked after each candidate tool action to support runtime monitoring, repair, replanning, or human escalation before invalid state is propagated. Aggregated over rollouts, the same signals can guide offline choices such as prompt design, validation budgets, and agent selection. These uses remain complementary to the paper’s evaluation focus: prefix executability exposes whether and where a trajectory becomes invalid, but it does not by itself certify semantic correctness, safety, privacy, or fairness. Developing closed-loop agents and deployment policies that optimize against these signals is an important direction for future work.

## Limitations

The study is an audit rather than a complete leaderboard evaluation: BFCL-v3 and ToolSandbox each contribute 200 sampled tasks, ToolSandbox is restricted to supported local-tool scenarios, and $\tau ^ { 3 }$ coverage is complete only for the Airline/Retail base splits. Task-level intervals do not measure within-task repeated-seed variation. The main-grid tables provide marginal intervals; paired intervals are reported for the separate verifier-sensitivity study. Fixed-cohort PEC reflects both reaching a depth and retaining validity, so its decline is not an error-only statistic.

Executability assumes explicit schemas, preconditions, and benchmark-accessible state. Partially observed web or GUI environments may require state estimation or learned preconditions, and executability does not certify semantic correctness. We did not evaluate a matched schema-constrained decoding baseline, so the reported rankings are not established under such constraints. These constraints can prevent malformed arguments, while prefix checking additionally covers object existence, provenance, state preconditions, and transition acceptance.

The intervention combines validation/revision with extra inference. The sensitivity study varies verifier capacity and budget, but its acceptance wrapper changes executor guidance; it does not fully separate validation logic from compute or context effects. Per-method cost telemetry is available for the $\tau ^ { 3 }$ audit, whereas the BFCL-v3/ToolSandbox grid has aggregate runtime only. Broader claims require additional domains, repeated runs, and matched constrained-decoding and repair controls.

Broader impacts. Prefix-level executability can improve oversight of tool-using agents by identifying invalid API calls, unresolved dependencies, and state-precondition violations before they are obscured by final-success scores. This is most useful in workflows where tool calls modify persistent state, consume resources, trigger external actions, or require human escalation. The same diagnostics may also make autonomous tool use easier to deploy in settings with privacy, security, fairness, or service-access risks. Moreover, an executable trajectory can still be unsafe, biased, privacy-violating, or misaligned with user intent. Prefix executability should therefore be treated as one reliability signal and combined with domain-specific safety policies, privacy controls, security review, and human oversight in consequential applications.

## Acknowledgments

This work was funded by Oracle.

## References

[1] Timo Schick, Jane Dwivedi-Yu, Roberto Dessı, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In Advances in Neural Information Processing Systems, 2023. URL https://openreview.net/f orum?id=Yacmpz84TH.

[2] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. Re-Act: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=WE\_vluYUL-X.

[3] Shishir G. Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The Berkeley Function Calling Leaderboard (BFCL): From tool use to agentic evaluation of large language models. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267, pages 48371–48392. PMLR, 2025. URL https://proceedings.mlr.press/v267/patil25a.html.

[4] Jiarui Lu, Thomas Holleis, Yizhe Zhang, Bernhard Aumayer, Feng Nan, Haoping Bai, Shuang Ma, Shen Ma, Mengyu Li, Guoli Yin, Zirui Wang, and Ruoming Pang. ToolSandbox: A stateful, conversational, interactive evaluation benchmark for LLM tool use capabilities. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 1160–1183, Albuquerque, New Mexico, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.findings-naacl.65. URL https://aclanthology.org/2 025.findings-naacl.65/.

[5] Kinjal Basu, Ibrahim Abdelaziz, Kiran Kate, Mayank Agarwal, Maxwell Crouse, Yara Rizk, Kelsey Bradford, Asim Munawar, Sadhana Kumaravel, Saurabh Goyal, Xin Wang, Luis A. Lastras, and Pavan Kapanipathi. NESTFUL: A benchmark for evaluating LLMs on nested sequences of api calls. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 33538– 33547, Suzhou, China, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.emnlp-m ain.1702. URL https://aclanthology.org/2025.emnlp-main.1702/.

[6] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ-bench: A benchmark for tool-agentuser interaction in real-world domains. In International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=roNSXZpUDN.

[7] Zehui Chen, Weihua Du, Wenwei Zhang, Kuikun Liu, Jiangning Liu, Miao Zheng, Jingming Zhuo, Songyang Zhang, Dahua Lin, Kai Chen, and Feng Zhao. T-Eval: Evaluating the tool utilization capability of large language models step by step. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9510–9529, Bangkok, Thailand, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.515. URL https://aclantholo gy.org/2024.acl-long.515/.

[8] Pengfei He, Zhenwei Dai, Bing He, Hui Liu, Xianfeng Tang, Hanqing Lu, Juanhui Li, Jiayuan Ding, Subhabrata Mukherjee, Suhang Wang, Yue Xing, Jiliang Tang, and Benoit Dumoulin. TRAJECT-Bench: A trajectory-aware benchmark for evaluating agentic tool use. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=TZWnWvsQ0X.

[9] Dawei Li, Yuguang Yao, Zhen Tan, Huan Liu, and Ruocheng Guo. ToolPRMBench: Evaluating and advancing process reward models for tool-using agents. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 12378–12391, San Diego, California, United States, 2026. Association for Computational Linguistics. doi: 10.18653/v1/2026.findings-acl.602. URL https://aclanthology.org/2026. findings-acl.602/.

[10] Shengda Fan, Xuyan Ye, Yupeng Huo, Zhi-Yuan Chen, Yiju Guo, Shenzhi Yang, Wenkai Yang, Shuqi Ye, Jingwen Chen, Haotian Chen, Xin Cong, and Yankai Lin. AgentProcessBench: Diagnosing step-level process quality in tool-using agents. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2, pages 8823–8834. Association for Computing Machinery, 2026. doi: 10.1145/3770855.3817494. URL https://doi.org/10.1145/3770855.3817494.

[11] Hongliu Cao, Ilias Driouich, and Eoin Thomas. Beyond task completion: Revealing corrupt success in LLM agents through procedure-aware evaluation. arXiv preprint arXiv:2603.03116, 2026. URL https://arxiv.org/abs/2603.03116.

[12] Wonjoong Kim, Sangwu Park, Yeonjun In, Sein Kim, Dongha Lee, and Chanyoung Park. Beyond the final answer: Evaluating the reasoning trajectories of tool-augmented agents. In Proceedings ofthe 43rd International Conference on Machine Learning, volume 306 of Proceedings ofMachine Learning Research, pages 58757–58786. PMLR, 2026. URL https://proceedings.mlr.press/v306/kim26bg.html.

[13] Ella Rabinovich and Ateret Anaby Tavor. On the robustness of agentic function calling. In Proceedings of the 5th Workshop on Trustworthy Natural Language Processing, pages 298–304, Albuquerque, New Mexico, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.trustnlp-main.20. URL https://aclanthology.org/2025.trustnlp-main.20/.

[14] Kristina Lewandowska. Beyond success rate: Benchmarking robustness in tool-using language agents. In AAAI 2026 Workshop on Trust and Control in Agentic AI, 2026. URL https://trustagenticai.github.io/AAA I2026/AAAI-Workshop/17.pdf.

[15] David G. Kleinbaum and Mitchel Klein. Survival Analysis: A Self-Learning Text. Springer, 3 edition, 2012. doi: 10.1007/978-1-4419-6646-9.

[16] E. L. Kaplan and Paul Meier. Nonparametric estimation from incomplete observations. Journal of the American Statistical Association, 53(282):457–481, 1958. doi: 10.1080/01621459.1958.10501452.

[17] Eduardo Treviño, Hugo Contant, James Ngai, Graham Neubig, and Zora Zhiruo Wang. Benchmarking failures in tool-augmented language models. In Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Vol ume 1: Long Papers), pages 2916–2934, Albuquerque, New Mexico, 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.naacl-long.149. URL https://aclanthology.org/2025.naacl-long.149/.

[18] Archiki Prasad, Alexander Koller, Mareike Hartmann, Peter Clark, Ashish Sabharwal, Mohit Bansal, and Tushar Khot. ADaPT: As-needed decomposition and planning with language models. In Findings of the Associationfor Computational Linguistics: NAACL 2024, 2024. URL https://aclanthology.org/2024.findi ngs-naacl.264/.

[19] Binfeng Xu, Zhiyuan Peng, Bowen Lei, Subhabrata Mukherjee, Yuchen Liu, and Dongkuan Xu. Re-WOO: Decoupling reasoning from observations for efficient augmented language models. arXiv preprint arXiv:2305.18323, 2023. URL https://arxiv.org/abs/2305.18323.

[20] Lutfi Eren Erdogan, Nicholas Lee, Sehoon Kim, Suhong Moon, Hiroki Furuta, Gopala Anumanchipalli, Kurt Keutzer, and Amir Gholami. Plan-and-Act: Improving planning of agents for long-horizon tasks. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267, pages 15419–15462. PMLR, 2025. URL https://proceedings.mlr.press/v267/erdogan25a.html.

[21] Jishnu Ray Chowdhury and Cornelia Caragea. Zero-shot verification-guided chain of thoughts. arXiv preprint arXiv:2501.13122, 2025. URL https://arxiv.org/abs/2501.13122.

[22] Danil S. Grigorev, Alexey K. Kovalev, and Aleksandr I. Panov. VerifyLLM: LLM-based pre-execution task plan verification for robots. arXiv preprint arXiv:2507.05118, 2025. URL https://arxiv.org/abs/2507.05118.

[23] Victor Barres, Ben Shi, Ola Zytek, Soham Ray, Keshav Dhandhania, and Pedram Razavi. τ<sup>3</sup>-bench: Advancing agent benchmarking to knowledge and voice. Sierra research release, March 2026. URL https://sierra.ai/blog/bench-advancing-agent-benchmarking-to-knowledge-and-voice.

## A Relationship to Prior Evaluation Protocols

Table A.1 compares the evaluation units and signals of related protocols. Prefix executability measures whether the observed action sequence satisfies the benchmark’s schemas, state preconditions, and data dependencies. Alternative action sequences receive full executability credit whenever these conditions hold, regardless of agreement with a reference trajectory. Prefix metrics therefore complement evaluations of task outcomes, intermediate milestones, reference-trajectory agreement, and semantic process quality.

<table><tr><td>Evaluation</td><td>Primary signal</td><td>Relation to prefix executability</td></tr><tr><td>T-Eval [7]</td><td>Component capabilities: planning, reasoning, retrieval, understanding, instruction following, and review.</td><td>Diagnoses component abilities; PEC tracks executable prefixes across depths, and FED identifies the first invalid action.</td></tr><tr><td>ToolSandbox [4]</td><td>Task-specified required and prohibited conditions over states and traces.</td><td>Checks required or prohibited conditions; PEC/FED localize first non-executability and survival over depth.</td></tr><tr><td>τ-bench [6] and τ3-bench [23]</td><td>Task outcomes in policy-constrained user-agent-environment interaction, including terminal goal-state checks.</td><td>Provide stateful environments where prefix metrics complement terminal reward.</td></tr><tr><td>TRAJECT-Bench [8]</td><td>Reference-based tool/argument/order matching and reference-free, LLM-judged trajectory satisfaction.</td><td>Combines reference-based and judged trajectory quality; prefix metrics measure benchmark-grounded executability, first failure, and recovery.</td></tr><tr><td>ToolPRMBench [9]</td><td>Correct/incorrect step cases for process-reward-model evaluation.</td><td>Assesses process judges; prefix metrics use benchmark execution semantics without a learned reward model.</td></tr><tr><td>AgentProcessBench [10]</td><td>Human-labeled step effectiveness; step and first-error prediction accuracy.</td><td>Evaluates agreement with human process labels; prefix metrics directly test benchmark execution feasibility.</td></tr><tr><td>TRACE [12]</td><td>Reference-free LLM assessment of efficiency, hallucination, and adaptivity using an evidence bank.</td><td>Assesses broader trajectory quality; prefix metrics add deterministic feasibility, first failure, and recovery.</td></tr><tr><td>Procedure-Aware Evaluation [11]</td><td>Utility, efficiency, interaction quality, and procedural integrity.</td><td>Audits broader procedural quality; prefix metrics isolate execution feasibility.</td></tr><tr><td>Beyond Success Rate [14]</td><td>Recovery under deliberately injected tool/environment faults.</td><td>Measures recovery per injected fault; RR here measures terminal success conditional on prefix failure observed</td></tr></table>

Table A.1: Evaluation units and signals across trajectory and process evaluation protocols.

## B Exact Properties, Consistency, and Relationship to Final Success

The properties below distinguish identities that hold for every evaluated cohort from population guarantees that require a sampling assumption. Throughout, rollout i has an evaluated horizon $T _ { i } \geq 1$ , first-error depth $F _ { i } : = \mathrm { F E D } _ { i } \in \{ 1 , . . . , T _ { i } + 1 \}$ }, and final-success indicator $Y _ { i } \in \{ 0 , 1 \}$ The value $F _ { i } = T _ { i } + \bar { 1 }$ denotes a fully executable trajectory. Let $( T , F , Y )$ denote the corresponding population variables. By definition, an executable prefix contains only executable shorter prefixes. No independence assumption is needed for Propositions 1–3.

Proposition 1 (fixed-cohort monotonicity). For an integer depth $k \geq 1$ , define

$$
E _ { i , k } : = \mathbf { 1 } \{ T _ { i } \geq k \land \operatorname { E x e c } ( \hat { \tau } _ { i , \leq k } ) \} = \mathbf { 1 } \{ F _ { i } > k \} , \qquad \widehat { \operatorname { P E C } } ( k ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } E _ { i , k } .\tag{13}
$$

Then both the empirical curve and its population counterpart $\operatorname { P E C } ( k ) : = \operatorname* { P r } ( F > k )$ are nonincreasing:

$$
1 \geq { \widehat { \mathrm { P E C } } } ( k ) \geq { \widehat { \mathrm { P E C } } } ( k + 1 ) \geq 0 , \qquad 1 \geq \mathrm { P E C } ( k ) \geq \mathrm { P E C } ( k + 1 ) \geq 0 .\tag{14}
$$

Proof. If $E _ { i , k + 1 } = 1$ , rollout i reaches depth $k + 1$ with an executable prefix. It therefore reaches depth k with an executable prefix, so $E _ { i , k } = 1$ . Thus $E _ { i , k + 1 } \leq E _ { i , k }$ pointwise; averaging or taking expectations proves the result. The denominator remains N at every depth. A rollout ending before depth k contributes zero to the fixed-cohort event in Eq. (13); the statement does not require a censoring model.

Proposition 2 (bounds and finite-cohort identities). Define $G _ { i } : = \mathbf { 1 } \{ F _ { i } = T _ { i } + 1 \}$ and $A _ { i } : = $ $( F _ { i } - 1 ) / T _ { i }$ . Then

$$
0 \leq \widehat { \mathrm { T E R } } \leq \mathrm { A U C } \widehat { \mathrm { P E C } } \leq 1 , \qquad 0 \leq \mathrm { T E R } \leq \mathrm { A U C } \mathrm { \mathrm { P E C } } \leq 1 ,\tag{15}
$$

$$
n _ { \mathrm { R R - E l i g i b l e } } = \sum _ { i = 1 } ^ { N } \mathbf { 1 } \{ F _ { i } \leq T _ { i } \} = N \big ( 1 - \widehat { \mathrm { T E R } } \big ) .\tag{16}
$$

The empirical mean ${ \overline { { F } } } : = N ^ { - 1 } \textstyle \sum _ { i } F _ { i }$ satisfies

$$
1 \leq \overline { { F } } \leq \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( T _ { i } + 1 ) .\tag{17}
$$

If $\begin{array} { r } { \mathbf { \mathbb { E } } [ T ] < \infty , } \end{array}$ , its population analogue is $1 \leq \mathbb { E } [ F ] \leq \mathbb { E } [ T ] + 1$ . When every rollout has the same horizon $T _ { 0 }$

$$
{ \widehat { \mathrm { T E R } } } = { \widehat { \mathrm { P E C } } } ( T _ { 0 } ) , \qquad { \mathrm { A U C - P E C } } = { \frac { { \overline { { F } } } - 1 } { T _ { 0 } } } .\tag{18}
$$

Proof. A fully executable trajectory has $G _ { i } = A _ { i } = 1$ . Otherwise, $G _ { i } = 0$ and $0 \leq A _ { i } < 1$ . Hence $0 \leq \mathsf { \bar { G } } _ { i } \leq \dot { A _ { i } } \leq 1$ pointwise, proving Eq. (15) by averaging and taking expectations. The events $F _ { i } = T _ { i } + 1$ and $\bar { F _ { i } ^ { \mathrm { ~ ~ } } } \leq T _ { i }$ partition the cohort, giving Eq. (16). The bound ${ \mathrm { i } } \leq F _ { i } \leq T _ { i } + 1$ gives Eq. (17) and its expectation counterpart. Finally, when $T _ { i } = T _ { 0 }$ , the events $F _ { i } > T _ { 0 }$ and $F _ { i } = T _ { 0 } + 1$ coincide, and the normalization $T _ { 0 }$ can be taken outside the empirical average.

For variable horizons, TER remains the complete-trajectory event $F = T + 1$ , whereas AUC-PEC is the mean per-trajectory normalized area $\dot { \mathbb { E } } [ ( F - 1 ) / T ]$ . Thus the common-horizon identities in Eq. (18) must not be applied to variable-length rollouts. In particular, mean FED is generally not determined by TER or AUC-PEC alone.

Proposition 3 (final-success decomposition). Let $G : = \{ F = T + 1 \}$ and $B : = \{ F \leq T \}$ denote the fully executable and prefix-failure regimes. For every rollout distribution,

$$
\operatorname* { P r } ( Y = 1 ) = \operatorname* { P r } ( \{ Y = 1 \} \cap G ) + \operatorname* { P r } ( \{ Y = 1 \} \cap B ) .\tag{19}
$$

${ \mathrm { I f ~ } } 0 < q : = { \mathrm { P r } } ( G ) < 1$ , define $s : = \operatorname* { P r } ( Y = 1 \mid G )$ and $r : = \mathrm { P r } ( Y = 1 \mid B ) = \mathrm { R R }$ . Since $q = \mathrm { T E R } .$

$$
\operatorname* { P r } ( Y = 1 ) = q s + ( 1 - q ) r , \qquad ( 1 - q ) r \leq \operatorname* { P r } ( Y = 1 ) \leq q + ( 1 - q ) r .\tag{20}
$$

Proof. The events G and B are disjoint and exhaustive, proving Eq. (19). When both have positive probability, the definition of conditional probability gives the decomposition in Eq. (20); its bounds follow from $0 \leq s \leq 1 . { \mathrm { ~ A t ~ } } q = 0 { \mathrm { ~ o r ~ } } { \bar { q _ { \mathrm { ~ } } } } = 1$ , the conditional probability on the absent regime is undefined, and Eq. (19) supplies the valid statement without assigning a value to that conditional. The same partition argument applies to empirical counts, with conditional empirical rates defined only for nonempty regimes.

Terminal success therefore combines successful fully executable trajectories and successful trajectories with an earlier prefix break. Holding s and r fixed, its slope with respect to TER is $s - r { \mathrm { ; } }$ across methods, both conditional success rates can also change. Consequently, the decomposition does not imply a universal monotone relationship between TER and final success. Section 3.1 gives a concrete empirical instance of this distinction.

Proposition 4 (consistency and concentration under i.i.d. sampling). Assume that task–rollout pairs are independent and identically distributed draws from a fixed evaluation distribution. For a fixed depth k, let $Z _ { i }$ be any of

$$
E _ { i , k } , \qquad { \bf 1 } \{ F _ { i } = T _ { i } + 1 \} , \qquad { \frac { F _ { i } - 1 } { T _ { i } } } , \qquad Y _ { i } ,\tag{21}
$$

corresponding to PEC at k, TER, AUC-PEC, and final success, respectively. Then ${ \widehat { \theta } } : = N ^ { - 1 } \sum _ { i } Z _ { i }$ is strongly consistent for $\theta : = \mathbb { E } [ Z _ { i } ]$ , and, for every $\epsilon > 0$

$$
\begin{array} { r } { \operatorname* { P r } \bigl ( | \widehat { \theta } - \theta | \geq \epsilon \bigr ) \leq 2 \exp ( - 2 N \epsilon ^ { 2 } ) . } \end{array}\tag{22}
$$

Mean FED is strongly consistent for $\mathbb { E } [ F ] \mathrm { i f } \mathbb { E } [ T ] < \infty$ . If, additionally, $T \leq K$ with probability one for a finite constant K, then

$$
\mathrm { P r } \big ( | \overline { { F } } - \mathbb { E } [ F ] | \geq \epsilon \big ) \leq 2 \exp \biggl ( - \frac { 2 N \epsilon ^ { 2 } } { K ^ { 2 } } \biggr ) .\tag{23}
$$

Finally, if $\operatorname* { P r } ( B ) > 0$ , the recovery-rate estimator

$$
\widehat { \mathrm { R R } } = \frac { N ^ { - 1 } \sum _ { i = 1 } ^ { N } Y _ { i } \mathbf { 1 } \{ F _ { i } \leq T _ { i } \} } { N ^ { - 1 } \sum _ { i = 1 } ^ { N } \mathbf { 1 } \{ F _ { i } \leq T _ { i } \} }\tag{24}
$$

has a nonzero denominator once at least one eligible episode has been sampled. With probability one, such an episode occurs as the sample grows, and RRd converges to the population recovery rate $\operatorname* { P r } ( Y = 1 \mid B )$ as $N \to \infty$

Proof. Every variable in Eq. (21) lies in [0, 1]. The strong law of large numbers establishes strong consistency, and Hoeffding’s inequality gives Eq. (22). Since $1 \leq F \leq T + 1$ , the condition $\mathbb { E } [ T ] < \infty$ ensures that $F$ is integrable, so the strong law also applies to mean FED. Under the bounded-horizon assumption, $F \in [ 1 , K + 1 ]$ has range of width $K ;$ Hoeffding’s inequality yields Eq. (23). For RR, the numerator and denominator of Eq. (24) converge with probability one to $\bar { \operatorname { P r } ( \{ Y = 1 \} \cap B ) }$ and $\mathrm { P r } ( B )$ , respectively. When $\mathrm { P r } ( B ) { \bf \bar { \theta } } > 0 .$ , continuity of division at a positive denominator proves the result. For a finite cohort with no eligible episodes, RR remains undefined.

Interpretation for the reported audits. The finite-cohort identities and monotonicity above hold exactly for the reported evaluations, including the deterministic stratified cohorts. Population consistency and concentration hold under the stated i.i.d. sampling assumption. The success decomposition separates success on fully executable trajectories from success after a prefix failure, making recovery explicit alongside prefix reliability.

## C Deterministic Executability Checking and Recovery

## C.1 Executability Checks and First-Failure Classification

The executability scorer is deterministic and separate from the LLM-based pre-execution verifier used by Dual-Loop. The scorer evaluates the actual emitted actions against benchmark tool and state semantics; no learned judge, human step label, or single gold action sequence is required. The Dual-Loop verifier instead examines a proposed plan before execution and may change the executor’s subsequent behavior. Its acceptance decision does not establish that the emitted trajectory is executable.

For action $~ a _ { j } ,$ , let $V _ { j } = 1$ exactly when the tool is exposed, its arguments satisfy the declared interface and validators, referenced values are available and type-compatible, required information has been obtained, applicable state preconditions hold, and the benchmark accepts the transition. Otherwise $V _ { j } = 0$ . The checker determines first failure as follows.

1. Initialize replay from the benchmark’s initial state. Available values comprise the prompt and benchmark-visible initial values. Initialize $\mathrm { F E D } = T + 1$

2. Process emitted actions in order. Resolve the selected tool; an unavailable tool is a schema/toolavailability failure. For an exposed tool, check argument binding, declared types, and registered validators.

3. Check referenced values for absence, stale bindings, or incompatible types; these are dependency failures. Check whether required task information has been obtained, and whether applicable state preconditions hold.

4. If these checks pass, execute the action against the replay or sandbox state. A rejected transition is a state-consistency failure. For an accepted transition, record the output, update available values, and advance the state.

5. At the first failed predicate or rejected transition, set $V _ { j } = 0$ and $\mathrm { F E D } = j .$ . Choose the first applicable diagnostic category in the precedence order below, then stop prefix replay. Retain the full observed rollout for benchmark-defined final success Y and RR.

6. If every action satisfies the predicate, retain FED = T + 1. A terminal no-tool action is executable only when benchmark semantics permit completion without a tool call.

The diagnostic precedence is

$$
\mathrm { \ s c h e m a ~ \longrightarrow ~ d e p e n d e n c y ~ \longrightarrow ~ p r e c o n d i t i o n ~ \longrightarrow ~ s t a t e - c o n s i s t e n c y ~ \longrightarrow ~ i n f o r m a t i o n . }
$$

The tie-break assigns one primary cause at the first failed action; it changes neither action validity nor FED, PEC, TER, AUC-PEC, RR, or final success. In particular, the diagnostic ordering is distinct from the order in which applicable conditions are checked.

BFCL-v3 evaluates prefixes by replay in a copied state under the case-specific function namespace and benchmark evaluator semantics [3]. ToolSandbox uses tools visible at the relevant turn, their registered validators, and the recorded execution outcome and state transition [4]. These benchmarkspecific semantics ground the same executability predicate. The metrics then aggregate the resulting $( { \dot { T } } _ { i } , \mathrm { F E D } _ { i } , Y _ { i } )$ records deterministically.

Stopping the first-failure replay does not truncate the observed episode or replace its final outcome. Here $V _ { j }$ describes the validity of action $a _ { j }$ in its local benchmark state. Determining FED requires checking actions only through the first failure. Later actions in the observed episode can be executable, and the task can still succeed. For $k \leq T$ , however, prefix validity is

$$
E _ { k } = \prod _ { j = 1 } ^ { k } V _ { j } = \mathbf { 1 } \{ \mathrm { F E D } > k \} .
$$

Once an action fails, every subsequent prefix still contains that failure. Consequently, later successful execution can contribute to recovery while never restoring executable-prefix survival for that rollout.

## C.2 Illustrative Contact-Repair Comparison

Table C.1 illustrates a contact-removal task under ToolSandbox semantics. In the first action, the agent supplies a phone number in place of the stored contact identifier. Although both values have string type, the requested identifier is absent from the current state, so the action is rejected without changing the state. A subsequent lookup obtains the correct identifier and enables removal.

<table><tr><td>Step</td><td>Emitted action</td><td>Benchmark outcome</td></tr><tr><td>1</td><td>Attempt to remove a contact using its phone num- ber as its stored identifier.</td><td>Rejected because no matching identifier exists; state unchanged.</td></tr><tr><td>2</td><td>Search for the contact by phone number.</td><td>Returns the contact&#x27;s stored identifier.</td></tr><tr><td>3</td><td>Remove the contact using the returned identifier.</td><td>Accepted; the requested contact is removed.</td></tr></table>

Table C.1: Illustrative repair after an invalid first action. The complete observed trajectory is retained even though prefix survival ends at the first step.

Here $V = ( 0 , 1 , 1 )$ but $E = ( 0 , 0 , 0 )$ . For $T = 3 , \mathrm { F E D } = 1$ , the trajectory contributes 0 to TER and AUC-PEC, and it is RR-eligible; its terminal success can nevertheless be $Y = 1$ . An error-free alternative first searches for the contact and then removes it using the returned identifier. Both actions in this two-step sequence are executable, so the full trajectory receives executability credit.

A terminal ToolSandbox milestone could credit both the repaired and error-free trajectories. A task-specific check for prohibited actions could distinguish them, while prefix executability directly records the rejected transition. TRAJECT-Bench could penalize the extra action or recognize the later dependency and ordering; TRACE could assess whether the repair is adaptive or inefficient. Prefix metrics summarize the same example through failure at depth one, zero complete-trajectory executability, and terminal success despite that failure. Thus prefix metrics distinguish successful recovery after an invalid action from error-free execution, even when both trajectories receive the same terminal-success label.

## D Dual-Loop Validation Details

This appendix specifies the validation intervention in Section 2.5. Only the evaluated executor trajectory, including the terminal no-tool action defined in Section 2.2, enters the prefix metrics. Planner and verifier messages influence subsequent behavior and cost but do not contribute to $T _ { i }$ Figure D.1 summarizes the information flow.

## D.1 Objects and Information Boundary

Let x denote the user goal, U the available tool specification, and $s _ { 0 }$ a compact summary of benchmarkvisible state before execution. Dual-Loop uses a planner $P ,$ verifier $V ,$ , and executor ${ \dot { R } } .$ At validation iteration $j ,$ , the planner proposes a finite candidate plan

$$
z ^ { ( j ) } = ( z _ { 1 } ^ { ( j ) } , \dots , z _ { M _ { j } } ^ { ( j ) } ) ,
$$

where each step describes an intended tool-level operation, argument values or constraints, and dependencies on prior planned steps, observations to be produced by those steps, or user-provided information. The plan length $M _ { j }$ need not equal the eventual executed horizon $T _ { i }$ : the plan is conditioning context, not the object being scored.

The verifier maps $( x , \mathcal { U } , s _ { 0 } , z ^ { ( j ) } )$ to either acceptance or a structured rejection record. The planner and verifier do not receive the benchmark reference trajectory, hidden final-success label, or future observations. After validation terminates, the executor interacts with the benchmark and emits

$$
{ \hat { \tau } } _ { i } = ( a _ { i , 1 } , \ldots , a _ { i , T _ { i } } ) .
$$

The evaluator scores this emitted trajectory directly. Thus, if the executor deviates from a verified plan, terminates early, or emits an invalid call, the metrics reflect the emitted actions exactly as they occurred.

![](images/42d76eadb32da09e4e6803175290fa64a58604441fcf873f3cac1346c16a27ec.jpg)  
Figure D.1: Dual-Loop validation intervention. The planner proposes a structured candidate plan, the verifier either accepts it or returns targeted feedback, and the executor emits the only environmentfacing actions scored by the prefix-level metrics.

## D.2 Verifier Checklist

The verifier is designed to reject candidate plans that contain predictable sources of non-executable prefixes before any environment action is taken. It applies the following local checklist, aligned with the first-failure taxonomy in Section 2.4:

1. Tool and schema validity. Each planned operation must refer to an available tool in $u ,$ include required arguments, avoid unsupported arguments, and satisfy declared type or formatting constraints.

2. Dependency resolution. Values referenced by a planned step must be available from the prompt, the initial state summary, or an earlier planned step. The verifier rejects forward references, stale bindings, and type-incompatible reuse.

3. Information availability. Required user information or lookup results must either be present, requested before use, or represented as a dependency on an earlier planned step. This check catches plans that use information before collecting it.

4. Explicit preconditions. If a tool requires prior authorization, a preceding state update, or a specific object to exist before use, the plan must satisfy that prerequisite before the tool is called.

5. State consistency. Planned state mutations must be compatible with benchmark-visible state and with state changes induced by earlier planned steps. Examples include rejecting updates to absent objects or operations that consume unavailable resources.

The checklist is intentionally local. It is not a proof of global optimality, and it does not replace benchmark replay after execution. The benchmark-side evaluator remains the authority for FED, TER, final success, and RR.

## D.3 Repair Loop and Iteration Budgets

If the verifier rejects a candidate plan, its feedback identifies the earliest detected failing step, the failed check, and the information needed for repair when that information is available from $x , u ,$ , or $s _ { 0 }$ . The planner then proposes a revised candidate before any environment-facing action is emitted. In the BFCL-v3/ToolSandbox experiments, the validation loop is capped at $K _ { \mathrm { v a l } } = 5$ candidate plans. If no candidate is accepted within this cap, the validation stage returns a failure flag and the executor receives no accepted plan as a strong constraint. The rollout is not discarded; any environment actions subsequently emitted by the executor are scored in the same way as all other rollouts. Cap exhaustion is therefore method-side behavior, not a prefix failure.

The validation cap is separate from the ReAct execution cap. For BFCL-v3 and ToolSandbox, we use $K _ { \mathrm { e x e c } } = 1 0$ ReAct action iterations for the environment-facing executor. This separation is important because the two loops bound different resources: $K _ { \mathrm { v a l } }$ limits unscored pre-execution search over candidate plans, whereas $K _ { \mathrm { e x e c } }$ limits the scored action horizon available for multi-step tool use. A validation iteration does not increment $T _ { i } ;$ only executor-emitted environment actions, or the terminal no-tool action described in Section 2.2, contribute to $T _ { i }$ and to PEC, FED, TER, AUC-PEC, and RR.

## D.4 Metric Accounting and Controls

The following conventions keep the intervention aligned with the prefix-level definitions:

1. Candidate plans, verifier decisions, and verifier feedback are not benchmark actions. They affect metrics only through the executor’s subsequent environment-facing behavior.

2. FED is computed after execution from $\hat { \tau } _ { i } . \mathrm { A }$ verified plan that later leads the executor to an invalid call is counted as an ordinary prefix failure.

3. RR is computed only for rollouts with $\mathrm { F E D } _ { i } \le T _ { i }$ . Rejected candidate plans before execution are not RR-eligible because no benchmark prefix has failed.

4. Validation does not modify benchmark state before execution. It can improve PEC, TER, or AUC-PEC only by changing the environment-facing actions emitted by the executor.

Within each benchmark–model condition, all variants use the same task identifiers, tool specifications, executor horizon cap, and decoding rule. The Dual-Loop–Plan-Conditioned ReAct comparison measures the joint effect of validation, revision, and their additional inference cost under matched executor conditions. The scored trajectory semantics are the same across variants. Appendix F gives the verifier-sensitivity results and experimental controls. Planner and verifier calls are outside the scored trajectory, but should be included when comparing token use, latency, or deployment cost.

## D.5 Limitations of the Intervention

The verifier can miss hidden-state constraints, underspecified intent, or semantic requirements outside the tool interface, and can reject executable alternatives. Pre-execution validation also adds inference cost. The prefix metrics characterize how Dual-Loop’s effects vary across environments and configurations (Appendix F).

## E Additional Experimental Settings and Compute Resources

Decoding and rollout settings. The following decoding settings and aggregate runtime apply to the 4,800-rollout BFCL-v3/ToolSandbox grid. All LLM conditions used the least-stochastic provider supported configuration. For GPT-4.1 and Llama-4-Maverick-Instruct, we set temperature to 0, used top- $\mathbf { \nabla } \cdot p = 1$ when exposed, and disabled presence and frequency penalties. For the GPT-5.4 reasoning condition, unsupported sampling controls were not set; instead we fixed reasoning.effort=high and kept the remaining provider-exposed generation controls at their defaults. To equalize generation budget, every planner, verifier, and executor call used a 4,000-token response cap; for APIs that distinguish hidden reasoning tokens from visible text, this cap is applied to the provider’s total output-token budget. The environment-facing ReAct execution loop was capped at $\dot { K } _ { \mathrm { e x e c } } = 1 0$ toolaction iterations for every ReAct-style agent variant, while Dual-Loop additionally used the separate pre-execution planner–verifier budget $\bar { K } _ { \mathrm { v a l } } = 5$ described in Appendix D; validation attempts are method-side computation and are not counted as tool actions in $\bar { T _ { i } }$

Compute resources. Our experiments evaluate agent rollouts rather than train or fine-tune models. All LLM inference was performed through hosted model APIs; the local machine only orchestrated benchmark execution, tool-state updates, trajectory logging, and metric computation. Model inference was served by provider-managed accelerator infrastructure behind API endpoints, so per-request GPU type, GPU count, and accelerator occupancy were not exposed to us. We therefore report local orchestration compute separately from hosted inference compute. The BFCL-v3/ToolSandbox grid contains $2 0 0 \times 2 \times 3 \times 4 = 4 , 8 0 0$ rollouts across two datasets, three models, and four agent variants. A CPU-only local worker is sufficient for the evaluation harness; the BFCL-v3/ToolSandbox runs used a Linux worker with 16 vCPUs and 64 GB RAM, with no local GPU allocated. Wall-clock time was dominated by remote model latency and API rate limits rather than local computation; using 16 asynchronous API-request workers, the 4,800-rollout BFCL-v3/ToolSandbox grid completed in approximately 72 hours. Offline computation of PEC, FED, TER, AUC-PEC, RR, and bootstrap confidence intervals for this grid completed in under 1 hour on the same machine.

The four-method, three-benchmark audit comprises 4,800 BFCL-v3/ToolSandbox rollouts and 1,968 Airline/Retail rollouts, totaling 6,768. The verifier-sensitivity study uses separate paired runs outside this total. The approximately 72-hour runtime applies only to BFCL-v3/ToolSandbox. Appendix F reports agent-side token and latency measurements for Airline/Retail and cost differences for the sensitivity study, excluding simulator and evaluator calls.

## F Transfer Audit, Cost, and Verifier Sensitivity

## F.1 Airline and Retail Audit Protocol

We evaluate all 50 Airline and 114 Retail tasks in the base splits of $\tau ^ { 3 }$ -bench v1.0.1 in its text-based, turn-taking (half-duplex) environment, using the benchmark’s simulator protocol, task specifications, and reward schema. This is complete coverage of these two base splits, not of every $\tau ^ { 3 }$ -bench domain. Each task is evaluated once under each of the four methods and three model conditions. All methods are evaluated on the same tasks. Table F.1 pools the 164 tasks, assigning equal weight to each task rather than to each domain. RR is computed over the eligible episodes in this pooled cohort.

For TER, mean FED, AUC-PEC, and final success, the intervals are 95% task-level percentile bootstrap intervals from 10,000 resamples. RR uses a two-sided 95% Wilson interval and reports the exact number of recovery successes and eligible episodes. These intervals describe task-level variation under the fixed rollout protocol; one rollout per task does not estimate within-task variation across repeated runs. Intervals for individual methods do not quantify uncertainty in paired differences between methods. The following subsections report agent costs and a separate sensitivity study with paired method differences.

<table><tr><td>Method</td><td>TER</td><td>Mean FED</td><td>AUC-PEC</td><td>RR (s/n)</td><td>Final success</td></tr><tr><td>GPT-5.4 (high)</td><td>.95</td><td>5.92</td><td>.95</td><td>.22 [.06,.55]</td><td>.43</td></tr><tr><td>ReAct</td><td>[.91,.98] .94</td><td>[5.53,6.31] 5.73</td><td>[.91,.98] .94</td><td>(2/9) .10 [.02,.40]</td><td>[.35,.51] .46</td></tr><tr><td>Prompt</td><td>[.90,.98] .93</td><td>[5.34,6.13] 5.84</td><td>[.91,.98] .93</td><td>(1/10) .25 [.09,.53]</td><td>[.38,.54] .51</td></tr><tr><td>Plan-Cond.</td><td>[.88,.96] .93</td><td>[5.45,6.24] 5.74</td><td>[.89,.96] .93</td><td>(3/12) .27 [.10,.57]</td><td>[.43,.59] .46</td></tr><tr><td>Dual-Loop</td><td>[.89,.97]</td><td>[5.36,6.14]</td><td>[.89,.97]</td><td>(3/11)</td><td>[.38,.53]</td></tr><tr><td>GPT-4.1</td><td>.84</td><td>6.47</td><td>.91</td><td>.54 [.35,.71]</td><td>.46</td></tr><tr><td>ReAct</td><td>[.79,.90] .82</td><td>[5.98,6.99] 6.23</td><td>[.87,.94] .89</td><td>(14/26) .45 [.28,.62]</td><td>[.39,.54] .51</td></tr><tr><td>Prompt Plan-Cond.</td><td>[.76,.88] .79</td><td>[5.79,6.68] 6.20</td><td>[.84,.93] .87</td><td>(13/29) .41 [.26,.58]</td><td>[.43,.59] .49</td></tr><tr><td>Dual-Loop</td><td>[.73,.85] .81</td><td>[5.72,6.70] 6.29 [5.54,7.33]</td><td>[.82,.91] .86</td><td>(14/34) .48 [.32,.65]</td><td>[.41,.57] .45</td></tr><tr><td>Llama-4 Maverick</td><td>[.75,.87]</td><td></td><td>[.81,.91]</td><td>(15/31)</td><td>[.37,.52]</td></tr><tr><td>ReAct</td><td>.52 [.44,.59]</td><td>2.66 [2.37,2.98]</td><td>.56 [.49,.63]</td><td>.11 [.06,.20] (9/79)</td><td>.18</td></tr><tr><td>Prompt</td><td>.55 [.48,.63]</td><td>2.95 [2.63,3.29]</td><td>.60 [.52,.67]</td><td>.12 [.07,.22]</td><td>[.12,.24] .18</td></tr><tr><td>Plan-Cond.</td><td>.60</td><td>2.71</td><td>.65</td><td>(9/73) .06 [.02,.15]</td><td>[.12,.24] .18</td></tr><tr><td>Dual-Loop</td><td>[.53,.68] .51 [.43,.59]</td><td>[2.46,2.98] 2.56 [2.24,2.93]</td><td>[.58,.72] .54 [.47,.61]</td><td>(4/65) .06 [.03,.14] (5/80)</td><td>[.12,.24] .14 [.09,.20]</td></tr></table>

Table F.1: Complete Airline/Retail base-split audit. Brackets are 95% intervals. For RR, s/n is the exact recovery-success/eligible count. Prompt denotes Prompt Planning; Plan-Cond. denotes Plan-Conditioned ReAct. All values are empirical estimates.

## F.2 Agent-Side Token Use and Latency

Table F.2 reports mean agent token use and cumulative request latency per episode for the Airline/Retail audit. Agent tokens sum provider-reported input and output tokens across planner, verifier, planner-revision, and executor requests. Recorded reasoning tokens are included in output tokens and counted once. Latency is the client-observed cumulative latency of those same requests, averaged per episode. Simulator and evaluator requests are excluded from both quantities. These latency values measure cumulative agent-call time rather than complete benchmark wall-clock duration.
<table><tr><td>Model</td><td>ReAct</td><td>Prompt</td><td>Plan-Cond.</td><td>Dual-Loop</td></tr><tr><td>GPT-5.4 (high)</td><td>86k/57s</td><td>94k / 68s</td><td>95k / 68s</td><td>98k / 83s</td></tr><tr><td>GPT-4.1</td><td>142k / 65s</td><td>156k / 70s</td><td>161k/73s</td><td>180k / 78s</td></tr><tr><td>Llama-4 Maverick</td><td>374k/ 61s</td><td>409k / 66s</td><td>412k/67s</td><td>399k / 66s</td></tr></table>

Table F.2: Mean agent tokens / mean cumulative agent-call latency on the Airline/Retail audit; k denotes thousands of tokens and s denotes seconds. Prompt and Plan-Cond. abbreviate Prompt Planning and Plan-Conditioned ReAct.

Relative to ReAct, Dual-Loop changes mean tokens/latency by approximately +14%/ + 46% for GPT-5.4, +27%/ + 20% for GPT-4.1, and +7%/ + 8% for Llama-4 Maverick. Relative to Plan-Conditioned ReAct, the corresponding approximate differences are +3%/ + 22%, +12%/ + 7%, and −3%/ − 1%.

The Dual-Loop–Plan-Conditioned contrast holds the model condition, executor, task identifiers, tool specification, decoding rule, and execution horizon fixed while adding validation and revision. These are net episode-level cost differences: verifier and revision costs are combined with changes in downstream execution cost. The comparison varies validation behavior and inference budget together.

Token totals should be compared within a model condition because tokenization and reasoning-token accounting differ across providers and models.

Per-method token-use and latency measurements are available for the Airline/Retail audit. For BFCL-v3 and ToolSandbox, compute reporting is limited to aggregate runtime; the Airline/Retail measurements do not establish their per-method costs.

## F.3 Matched Verifier Sensitivity

A separate sensitivity study evaluates the same 164 Airline/Retail tasks with a GPT-5.4-high planner and executor. For each task, all experimental conditions use the same initial plan, tool specifications, and execution horizon. Each condition’s cost includes the generation of the shared initial plan once, together with its verification, plan-revision, and execution requests; simulator and evaluator requests are excluded. Differences in Table F.3 are measured against a separately evaluated, task-paired Plan-Conditioned ReAct baseline. This baseline belongs to the sensitivity experiment; it is distinct from the Plan-Conditioned ReAct estimates in Tables F.1 and F.2.

The 95% intervals for metric differences use 100,000 paired task-bootstrap resamples, preserving the task pairing between each condition and its baseline. These characterize variation across tasks under paired resampling and do not measure within-task variability over repeated runs. Cost differences are reported as point estimates without intervals.

<table><tr><td>Configuration</td><td>∆TER</td><td>ΔAUC-PEC</td><td>∆Final</td><td>∆Tokens / ∆Latency</td></tr><tr><td>Deterministic acceptance</td><td>+.000 [−.018, +.018]</td><td>-.003 [-.018, +.008]</td><td>-.073 [−.140, -.012]</td><td>-8.4k /-3.2s</td></tr><tr><td>GPT-5.4  $K _ { \mathrm { v a l } } = 1$ </td><td>-.006 [−.030, +.012]</td><td>-.010 [-.030, +.005]</td><td>-.012 [−.079, +.055]</td><td>+0.7k/+43.1s</td></tr><tr><td>GPT-4.1</td><td>+.000</td><td>+.000</td><td>-.085</td><td></td></tr><tr><td> $K _ { \mathrm { v a l } } = 5$ </td><td>[−.018, +.018]</td><td>[-.007, +.008]</td><td>[-.165, -.006]</td><td>+7.1k/+30.9s</td></tr><tr><td>GPT-5.4  $K _ { \mathrm { v a l } } = 5$ </td><td>+.006 [−.018,+.030]</td><td>+.006 [−.015,+.030]</td><td>+.000 [-.073, +.073]</td><td>+40.7k/+176.9s</td></tr></table>

Table F.3: Matched verifier-sensitivity differences against the paired Plan-Conditioned ReAct rerun, with 95% paired task-bootstrap intervals. Model names identify the verifier; the planner and executor remain GPT-5.4-high. The deterministic-acceptance condition makes no verifier call and changes the executor’s instructions.

All TER and AUC-PEC intervals include zero, with small point-estimate changes in this environment and configuration. Equivalence was not tested. The final-success intervals for deterministic acceptance and the GPT-4.1 verifier are negative relative to the paired baseline; the other two final-success intervals include zero. These paired measurements quantify prefix reliability, terminal outcomes, and inference cost separately for each verifier configuration.

Interpretation of verifier contrasts. Both deterministic acceptance and the Plan-Conditioned ReAct baseline omit verifier calls, but deterministic acceptance tells the executor that the plan was accepted without evaluation. Its contrast with the baseline therefore changes executor instructions and does not measure the effect of removing verification. Comparisons involving LLM verifiers also change inference cost. The reported differences are specific to the evaluated environment and configurations and do not isolate verification logic from executor instructions and computational budget.

## G Depth-Wise PEC Checkpoints

Table G.1 reports checkpoint values for the fixed-cohort PEC curves used in the main analysis. These checkpoints are included for auditability and to show where along the tool-call trajectory the methods begin to separate. Because $\widehat { \mathrm { P E C } } ( k )$ is a survival estimate over the same rollout cohort, all $N = 2 0 0$ rollouts in a dataset–model–method block remain in the denominator; the values should be compared within blocks and interpreted as prefix-reliability diagnostics, not as eligible-depth conditional accuracy or path-efficiency scores. Early depths are most informative for method comparison, while later depths document tail behavior where decreases can also reflect trajectories that terminate before depth k.

Equation (7) expresses fixed-cohort PEC as the fraction of rollouts reaching depth k, multiplied by the fraction of those rollouts with executable prefixes. Table G.1 reports their product; it does not separately report depth support and eligible-depth PEC.
<table><tr><td>Dataset</td><td>LLM</td><td>Method</td><td>PEC@1</td><td>PEC@2</td><td>PEC@3</td><td>PEC@4</td></tr><tr><td>BFCL</td><td>GPT-5.4 (high)</td><td></td><td>0.95 [0.93,0.98]</td><td>0.90 [0.85,0.94]</td><td>0.81 [0.75,0.86]</td><td>0.68 [0.61,0.74]</td></tr><tr><td rowspan="8"></td><td></td><td>ReAct</td><td>0.96</td><td>0.88</td><td>0.82</td><td>0.70</td></tr><tr><td></td><td>Prompt Planning</td><td>[0.93,0.98] 0.77</td><td>[0.83,0.93] 0.61</td><td>[0.77,0.88] 0.49</td><td>[0.64,0.77] 0.42</td></tr><tr><td></td><td>Plan-Cond. ReAct</td><td>[0.70,0.82] 0.98</td><td>[0.55,0.69] 0.97</td><td>[0.42,0.56] 0.94</td><td>[0.36,0.49] 0.82</td></tr><tr><td rowspan="4">GPT-4.1</td><td>Dual-Loop</td><td>[0.96,0.99] 0.93</td><td>[0.94,0.99] 0.84</td><td>[0.91,0.97] 0.74</td><td>[0.77,0.88] 0.61</td></tr><tr><td>ReAct</td><td>[0.89,0.96] 0.97</td><td>[0.79,0.89] 0.93</td><td>[0.68,0.80] 0.86</td><td>[0.55,0.68] 0.71</td></tr><tr><td>Prompt Planning</td><td>[0.94,0.99] 0.65</td><td>[0.89,0.96] 0.48</td><td>[0.81,0.91] 0.39</td><td>[0.65,0.77] 0.33</td></tr><tr><td>Plan-Cond. ReAct</td><td>[0.57,0.71] 0.99 [0.98,1.00]</td><td>[0.41,0.55] 0.97</td><td>[0.32,0.46] 0.95</td><td>[0.26,0.39] 0.82</td></tr><tr><td rowspan="5"></td><td>Dual-Loop</td><td>0.89</td><td>[0.95,0.99] 0.71</td><td>[0.93,0.98] 0.59</td><td>[0.77,0.88] 0.46</td></tr><tr><td>Llama-4 Maverick ReAct</td><td>[0.84,0.93] 0.94</td><td>[0.66,0.78] 0.88</td><td>[0.53,0.67] 0.80</td><td>[0.39,0.53] 0.65</td></tr><tr><td>Prompt Planning</td><td>[0.91,0.97] 0.53</td><td>[0.82,0.92] 0.23</td><td>[0.74,0.85] 0.10</td><td>[0.58,0.71] 0.05</td></tr><tr><td>Plan-Cond. ReAct</td><td>[0.46,0.59] 0.98</td><td>[0.18,0.29] 0.94</td><td>[0.06,0.14] 0.90</td><td>[0.02,0.08] 0.78</td></tr><tr><td>Dual-Loop</td><td>[0.96,1.00]</td><td>[0.91,0.97]</td><td>[0.85,0.94]</td><td>[0.72,0.83]</td></tr><tr><td rowspan="9">GPT-4.1</td><td>ToolSandbox GPT-5.4 (high)</td><td>ReAct</td><td>0.92 [0.88,0.95]</td><td>0.81 [0.75,0.86]</td><td>0.59 [0.52,0.66]</td><td>0.47 [0.40,0.54]</td></tr><tr><td rowspan="7"></td><td>Prompt Planning</td><td>0.94 [0.91,0.97]</td><td>0.88 [0.83,0.93]</td><td>0.63 [0.56,0.69]</td><td>0.56 [0.48,0.62]</td></tr><tr><td>Plan-Cond. ReAct</td><td>0.67 [0.60,0.73]</td><td>0.49 [0.42,0.56]</td><td>0.35</td><td>0.32 [0.26,0.39]</td></tr><tr><td>Dual-Loop</td><td>0.97 [0.95,0.99]</td><td>0.95 [0.92,0.98]</td><td>[0.29,0.42] 0.69 [0.62,0.75]</td><td>0.62 [0.56,0.69]</td></tr><tr><td>ReAct</td><td>0.94 [0.91,0.97]</td><td>0.81 [0.75,0.86]</td><td>0.58 [0.52,0.65]</td><td>0.45 [0.38,0.52]</td></tr><tr><td>Prompt Planning</td><td>0.96 [0.93,0.98]</td><td>0.90 [0.85,0.94]</td><td>0.64 [0.56,0.70]</td><td>0.56 [0.48,0.62]</td></tr><tr><td>Plan-Cond. ReAct</td><td>0.55 [0.47,0.61]</td><td>0.37 [0.30,0.44]</td><td>0.27 [0.21,0.34]</td><td>0.24 [0.18,0.30]</td></tr><tr><td>Dual-Loop</td><td>0.99 [0.97,1.00]</td><td>0.96 [0.94,0.99]</td><td>0.69 [0.63,0.76]</td><td>0.64 [0.57,0.70]</td></tr><tr><td rowspan="5">Llama-4 Maverick ReAct</td><td></td><td>0.87 [0.82,0.92]</td><td>0.68 [0.60,0.74]</td><td>0.45 [0.38,0.52]</td><td>0.29 [0.23,0.36]</td></tr><tr><td></td><td>0.92</td><td>0.80 [0.74,0.85]</td><td>0.53 [0.46,0.60]</td><td>0.41 [0.34,0.47]</td></tr><tr><td>Prompt Planning</td><td>[0.88,0.95] 0.36</td><td>0.10</td><td>0.09</td><td>0.07</td></tr><tr><td>Plan-Cond. ReAct</td><td>[0.30,0.43] 0.99</td><td>[0.06,0.14] 0.88</td><td>[0.06,0.13] 0.65</td><td>[0.04,0.12] 0.55</td></tr><tr><td>Dual-Loop</td><td>[0.97,1.00]</td><td>[0.83,0.93]</td><td>[0.58,0.71]</td><td>[0.48,0.62]</td></tr></table>

Table G.1: Depth-wise prefix executability for the 200-instance BFCL multi-turn sample and the 200-instance ToolSandbox sample under three LLM conditions. All entries are empirical estimates. Bracketed values are 95% percentile confidence intervals from 10,000 task-level bootstrap resamples. PEC@k uses the fixed-cohort survival estimate in Eq. (4): the numerator counts rollouts with FED > k, and the denominator is all evaluated rollouts in the dataset-LLM-method block.