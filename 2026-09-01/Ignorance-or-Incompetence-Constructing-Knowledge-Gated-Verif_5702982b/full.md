# Ignorance or Incompetence? Constructing Knowledge-Gated, Verifiable Tasks for LLM Agents

Hanlin Tian<sup>1⋆</sup>, Minhao Li<sup>1,2</sup>, Yu Mi<sup>1,3</sup>, Sihan Zhu<sup>1,4</sup>, Zhao Yang<sup>1,5</sup>, Yuxiang Wang<sup>1,6</sup>, Hongquan Zhu<sup>1</sup>, and Qiufei Hu<sup>1⋆⋆</sup>

<sup>1</sup>DataGrids <sup>2</sup>Shanghai University <sup>3</sup>Peking University <sup>4</sup>Northwestern Polytechnical University <sup>5</sup>Nanyang Technological University <sup>6</sup>East China Normal University huqf@datagrids.ai

Abstract. Professional agent tasks often depend on conventions that are absent from public corpora, yet benchmarks rarely control whether an agent has access to those conventions. We introduce a knowledge-gated task-construction protocol that separates a task instruction from a compact artefact containing private conventions, reference tables, and utility operators. Construction-time provenance, byte-identical task instructions across the provided- and withheld-artefact conditions, leak audits, and executable witnesses make dependence on the artefact explicit and testable. Across fifteen calibration tasks, one frontier agent configuration achieves a 68.0% pass rate with the artefact and 0% without it; on one task, a plausible but incorrect artefact also yields 0% across five trials. Deterministic solvers and rule corpora provide exact ground truth for structured tasks, while named criterion-level rubrics support outputs that cannot be checked by a single executable oracle. A configuration-relative calibration screen retains seven tasks satisfying our five-trial empirical knowledge-gating screen. These experiments validate the behavior of the construction protocol; they do not establish that the retained tasks improve post-training. We publicly release part of the task suite and supporting tooling at https://github.com/DatagridsAI/Knowledge-Gated-Task-Construction.

Keywords: Synthetic task construction · Knowledge gating · Task validation · Verifiable rewards · Agent evaluation

## 1 Introduction

Constructing agent tasks in professional domains presents a practical tension. If a generator already knows every rule needed to solve a task, the resulting task may add little beyond its own capability. If the relevant rules are genuinely unavailable, however, the task can become under-specified and difficult to verify. How can a curator make access to specialised knowledge necessary while still retaining an exact account of what a correct solution requires?

We treat this as a construction problem rather than a claim about recursive selfimprovement. The curator need not solve the task in the same way as the evaluated agent; the curator instead occupies an asymmetric position created at construction time. Three asymmetries suffice: (i) construction privilege: the answer is planted at construction time (an arbitrary internal threshold, a private normalisation table) and is therefore not derivable from the instruction or environment alone, regardless of the solver’s capability; (ii) cheap verification: checking a structured output against a planted ground truth is easy even when producing that output is not; (iii) compositional verification: an all-pass verifier over k named criteria allows task complexity to grow while each failure remains attributable.

Existing approaches to synthetic data struggle to overcome this paradox. Modelgenerated instruction tuning typically plateaus at the generator’s inherent capability [31], while traditional dataset distillation compresses massive corpora into synthetic tensors for weight-space updates [30]—a medium entirely inaccessible to an agent reasoning at runtime. Inspired by SkillsBench’s paired evaluation of agent performance with and without portable skill documents across diverse tasks [17], together with a controlled study of skill availability and presentation [35], we ask whether this asymmetry can be made a deliberate property of task construction rather than merely an evaluation condition. We turn these asymmetries into a controlled protocol for constructing and validating agent tasks.

Each task is paired with a small curated knowledge artefact—domain rules, private conventions, hard-to-hand-write operators—authored or adapted by a human curator from a larger domain corpus into a few kilobytes of text supplied in the agent’s context. The artefact is not produced by an optimisation procedure; what we contribute is the protocol that makes its contribution measurable, not an automatic compression algorithm. The task instruction never references the artefact, and a leak audit checks the instruction and environment for explicit gated content. Executable per-task audit scripts are present for nine of the fifteen calibration tasks; we do not claim automated audit coverage for the other six. Static audits provide evidence that the observed performance difference depends on access to the artefact, without claiming to exclude every form of semantic or indirect leakage.

The protocol has three operational components: The artefact creates a controlled dependency. Across fifteen tasks whose instructions were held byte-identical between the provided- and withheld-artefact conditions (uniformly five trials per cell), the tested Opus configuration moves from a 0% pass rate to 68.0% once the kilobyte artefact is provided. This establishes strong dependence under that configuration without claiming to isolate knowledge from tool use, file discovery, or other capabilities. Verification is automatable for structured tasks. Deterministic solvers and rule corpora compute exact references, while free-form deliverables are assessed with criterion-level rubrics. Calibration exposes unsuitable candidates. A configuration-relative screen rejects tasks that are not solved reliably with the artefact, are solved without it, or are already easy for the second tested configuration. This is a quality-control rule, not evidence that the retained tasks are better training data.

This paper makes three contributions:

1. A knowledge-gated task-construction contract. Tasks are paired with a kilobytescale artefact carrying the domain’s private conventions, and a static leak-audit protocol checks instructions and environments for exposed gated content. Executable per-task audit scripts cover nine of the fifteen calibration tasks. In this batch, the artefact accounts for the difference between a 0% and a 68.0% pass rate for Opus on byte-identical task instructions across the provided- and withheld-artefact conditions, which we take as evidence—not proof—that access to the artefact is the operative variable (§4.2).

2. A validation and attribution protocol. Deterministic witnesses support exact verification for structured outputs, named rubric criteria localise failures for openform outputs, and provenance plus leak audits delimit what each task can claim. The distinction between non-derivable convention gates and re-derivable operator gates prevents both from being presented as the same construct (§3).

3. A calibration study of the construction protocol. Paired artefact ablations, one perturbed-artefact control, and a two-task recitation probe show both where the intended dependency holds and where a full agent harness introduces additional failure modes. The seven-task retained set is therefore an operationally calibrated release candidate, not a demonstrated source of superior training signal (§4).

We do not train on this data. The experiments characterise whether the constructed tasks behave according to the protocol. Establishing downstream training value requires a held-out, end-to-end post-training comparison and remains outside the evidence in this paper.

## 2 Related Work

The methodology presented in this work intersects three dominant streams of research in data-centric agent training: the synthesis and compression of training environments, the grounding of models in external procedural knowledge, and the curation of datasets for verifiable-reward optimisation.

## 2.1 Synthetic Task Generation and Procedural Compression

Instruction synthesis initially relied on model-generated prompt–response pairs [31] and iterative, difficulty-increasing prompt rewrites [34]. More recently, agentic variants have shifted toward grounding tasks in real-world software artefacts. This includes mapping GitHub issue–patch pairs to test-driven environments [13], synthetically injecting bugs into functioning code to establish the original version as a definitive oracle [36], and scaling complex, executable environments for extended agent training [19]. Other approaches explore an environment first to reverse-engineer natural language instructions directly from observed trajectories [22, 23]. Domain-specific closed-loop frameworks likewise generate critical scenarios around observed agent performance gaps, as demonstrated in autonomous driving [25]. At an industrial scale, simulated tool ecosystems and multi-stage verification pipelines now supply massive agentic post-training corpora [14, 18], while fully synthetic environment generators scale multi-turn tool-use RL to thousands of database-backed worlds [9, 32]. Self-play proposer–solver loops [39] attempt to dynamically attack the capability ceiling of these generators.

Concurrently, traditional dataset distillation seeks to construct a minimal synthetic dataset for the training process that matches a larger corpus, whether by direct gradient optimisation [30], matching training trajectories [4], or aligning difficulty to the learner [10, 21, 41]. We do not claim membership in this line of work: distillation in that sense involves an explicit objective and an optimisation procedure, whereas our artefacts are authored by hand and delivered in context at inference time. The nearer relatives are prompt compression, context distillation, and the curation of portable skill documents [12, 28]; relative to those, our addition is the enforcement protocol (instruction/artefact separation plus leak audit) that lets the artefact’s contribution be isolated rather than merely observed.

## 2.2 Knowledge-Grounded Agents and Skill-Conditioned Benchmarks

Evaluating agents on their ability to ingest and apply external information has led to benchmarks explicitly pairing tasks with procedural or domain knowledge, including policy manuals in conversational tool use [2, 37], scientist-annotated background context in research coding [26], expert rubrics over occupational deliverables [20], and software development scenarios reliant on framework knowledge [11]. Similarly, agent skill libraries have evolved from self-generated, self-verified code repositories [28] into systematised, portable procedural documents that extend beyond simple tool wrappers [12].

Construction from private conventions does amount to planting hidden task specifications, and it is worth being direct about the consequence: success measures whether an agent holds and applies a supplied key, which overlaps with but is not identical to professional competence. Retrieval-augmented generation typically supplies helpful, derivable context; portable skill documents supply general, potentially derivable procedures. Our difference is that the gated content is fixed at construction time and not derivable from the instruction, and that a leak audit checks the instruction and environment for it. That audit is a static scan, so it provides evidence for—not a guarantee of—an enforced knowledge gap. What it buys is that the observed effect is a toggle between total failure and majority success rather than a marginal, context-dependent gain.

Two recent works sit close to this construction. Agentic context learning studies how an agent recovers specifications that are unspecified but discoverable from information already present in a visible context [40]; we instead plant a specification that is not derivable from the instruction or environment at all, and make its presence a controlled, audited experimental variable via the +A/-A conditions. Automated benchmark auditing searches existing benchmarks for hidden dependencies and specification gaps that should not be there [29]; we take the complementary position of deliberately constructing a private specification and exposing it as an explicit, audited variable rather than treating it as a defect to be found and removed.

## 2.3 Verifiable Rewards and Task Calibration

Reinforcement learning with verifiable rewards (RLVR) heavily underpins current reasoning and agentic post-training methodologies [8,16]. This paradigm is actively applied to software-engineering agents trained with execution-based environment rewards [7] and reasoning agents supervised by intermediate, process-level verifiers [38]. Because RLVR requires meaningful gradients, practitioners routinely discard tasks where empirical pass rates are either saturated or absolute zero [3,14]. Concurrent literature formalises this phenomenon online, demonstrating that policy-gradient signal peaks at intermediate pass rates, which motivates the use of pass-rate filtering, learning-zone scoring, and adaptive sampling during the training loop [1,6,33]. This mirrors classical psychometrics and item response theory (IRT), where test items that all examinees either pass or fail carry zero discriminative power [27].

We use these ideas only to motivate an offline calibration screen. The screen checks whether candidate tasks exhibit an intended operational profile under two fixed agent configurations; it neither estimates their value for policy optimisation nor proves downstream training benefit.

## 3 Methodology

This section specifies a construction contract designed to make dependence on supplied knowledge explicit and verifiable. The protocol separates task instructions from domain artefacts, audits the resulting boundary, derives checkable references, and uses empirical calibration to reject candidates that do not exhibit the intended behavior under the tested agent configurations.

## 3.1 Task/knowledge separation

Each unit of curation is a pair: a task (inputs, output schema, formatting and ordering requirements, structured metadata curation, multi-modal data validation, success criteria) and a knowledge artefact (domain terminology, conventions, formulas, reference tables, and low-level utility operators—never an end-to-end solution script). The instruction specifies what is required and never mentions the artefact; the artefact specifies how the domain computes, and is designed to be reusable across task instances. This separation is checked using the audit protocol in §3.4; executable per-task implementations are available for nine calibration tasks.

## 3.2 Data, ethics, and task lineage

The task records used in this study are synthetic: they do not contain real customer, borrower, passenger, employee, or confidential operational data. The calibration pool is not claimed as fifteen wholly new tasks. It combines study-specific synthetic tasks with tasks adapted from SkillsBench development material or identified in their metadata as SkillsBench Vendor Samples. In particular, the paper-index and trial-cohort tasks are vendor samples, while the software-dependency and SEC 13F tasks are adapted from SkillsBench task variants. The public five-task subset contains four study-specific synthetic tasks and paper-index, whose vendor-sample provenance is retained. Our reported rollouts are new calibration runs under the configurations described here; they are not official SkillsBench benchmark results. Dataset and component-specific notices accompany the public repository.

![](images/aa6794494a7de00f891a67c9bbca3bef6f175ce0585c6e4b0c2a99557a05d492.jpg)  
Fig. 1: Overview of the knowledge-gated curation protocol. A human curator condenses a large domain corpus into a kilobyte-scale artefact without including an end-to-end solution. After a leak audit, the same instruction is evaluated under three agent–knowledge configurations. “Pass”, “Fail”, and “Low pass” in the schematic denote empirical pass-rate bands over repeated trials, not single-run outcomes. Structured tasks use deterministic solvers and rule corpora for exact ground truth; free-form deliverables use criterion-level rubric judgement. Knowledge ablation and calibration diagnose the constructed tasks; no model training is performed in this work.

## 3.3 Two families of knowledge gates

Convention gates plant private, non-derivable choices: internal thresholds, fixed aliases and normalisation tables, or one variant selected among several defensible definitions (e.g., whether a turnover metric excludes the opening minutes; a baseline denominator of 235 rather than 240). A capable model can produce a reasonable answer; only the artefact determines the correct one, ensuring difficulty is strictly knowledge-gated. Operator gates place hard-to-hand-write algorithms in the artefact’s utility library (e.g., Brandes betweenness under a fixed normalisation convention; Garman–Klass volatility). Unlike convention gates, operators are not strictly non-derivable; rather, they establish a steep capability barrier where the agent must re-derive brittle numerics under time pressure without the artefact. Both gates leave the orchestration—parsing, grouping, deduplication, graph construction, serialisation—to the agent, so capability is still exercised and measured.

## 3.4 Leak audit and fairness protocol

The -A condition must represent a genuine knowledge ablation, not a broken environment. Instructions are byte-identical across conditions, refer to required conventions only as “the established in-house specification,” and include a fallback clause directing the agent to apply common conventions when no specification is present—so -A agents submit confident, verifiable answers rather than abstaining. A static audit scans instructions and environment data for explicit gated constants and any mention of the artefact. Executable per-task audit scripts are publicly available for part of the released subset; we do not claim automated audit coverage for every task. While static scanning does not definitively rule out semantic paraphrasing or adversarial leakage, it provides a useful baseline where implemented.

## 3.5 Configuration-relative calibration screen

Each candidate task is evaluated for five trials under three calibration conditions to estimate the empirical pass rate $\hat { p } ( A , c ) = k _ { t , A , c } / 5$ for task t, agent configuration A, and artefact condition $c \in \{ + \mathbf { A } , - \mathbf { A } \}$ . An agent configuration includes the model, harness, tools, budgets, and prompting policy. The screen acts as a heuristic indicator for task retention:

$$
\mathbb { I } _ { \mathrm { r e t a i n } } = \mathbf { 1 } \Big [ \hat { p } ( A _ { F } , + \mathbf { A } ) \ge 0 . 6 ~ \land ~ \hat { p } ( A _ { F } , - \mathbf { A } ) = 0 ~ \land ~ \hat { p } ( A _ { \mathrm { m i d } } , + \mathbf { A } ) \le 0 . 4 \Big ]\tag{1}
$$

where $A _ { F }$ is the frontier calibration configuration and $A _ { \mathrm { m i d } }$ is the second calibration configuration. Tasks where $\mathbb { I } _ { \mathrm { r e t a i n } } = 0$ are revised or discarded. Because the two configurations use different models and agent harnesses, retention is explicitly configurationrelative and must not be interpreted as an intrinsic ranking of the underlying models.

Three properties of this rule should be stated plainly. First, it does not target variance directly: it admits ${ \hat { p } } ( A _ { F } , + \mathbf { A } ) = 1$ , requires ${ \hat { p } } ( A _ { F } , - \mathbf { A } ) = 0$ , and permits $\hat { p } ( A _ { \mathrm { m i d } } , + \mathbf { A } ) = 0 .$ so all three calibration cells can individually have zero empirical variance while the task is retained. What the rule encodes is a separation between conditions. Second, with five trials per cell, one flipped outcome changes $\hat { p }$ by 20 percentage points; the nominal thresholds therefore reduce to at least 60% for $A _ { F } , + \mathsf { A }$ , exactly 0% for $A _ { F } , - \mathsf { A }$ , and at most 40% for $A _ { \mathrm { m i d } } , + \mathrm { A }$ . This is a coarse quality-control screen, not a statistical test. Third, any same-pool statistic computed after selection is descriptive: tasks are retained using these outcomes, so the survivors cannot provide independent evidence that the screen improves training data.

## 3.6 Verification and attribution

Verifiers parse structured outputs, recompute references via independent Python-based implementations, and check planted values using strict numerical tolerances (typically $1 0 ^ { - 4 }$ for floating-point comparisons). For free-form deliverables, we employ per-criterion rubric judging with strict all-pass semantics. Specifically, the final task reward $R \in \{ 0 , 1 \}$ over K named criteria is defined as:

$$
R = \prod _ { k = 1 } ^ { K } \nu _ { k }\tag{2}
$$

where $\nu _ { k } \in \{ 0 , 1 \}$ is the binary evaluation of the k-th criterion. While this product creates an exponentially sparse reward that poses known challenges for RL exploration, every failure $( \nu _ { k } = 0 )$ is attributed to its named criterion. This granular attribution supports automated data cleaning and generates labelled failure modes for future task construction, mitigating the learning difficulties of a strict all-pass reward. Every task in the fifteen-task calibration batch reported in §4 uses the deterministic verification path: the rubric-judging path described above is part of the protocol’s capability but is not exercised by any of the reported pass rates, so the headline numbers reflect exact executable verification only, with no LLM-judge involvement.

## 4 Experiments

Our evaluation asks whether the constructed tasks satisfy the protocol; it does not measure training outcomes. We report a fifteen-task calibration batch, a perturbed-artefact control, and a recitation probe that separates artefact use from full-harness execution (§4.2–4.3). Public paired-condition benchmarks provide context rather than a replication (§4.4). We then describe the retained set produced by the calibration screen (§4.5); a single-task artefact-shortening case study is deferred to the appendix (§7) to keep the main text focused on verification and audit evidence.

## 4.1 Experimental setup

The frontier configuration $A _ { F }$ uses Claude Opus 4.8 (claude-opus-4-8) through the Claude Code agent (BenchFlow’s claude-agent-acp integration), both with and without the knowledge artefact (+A/-A). The second configuration $A _ { \mathrm { m i d } }$ uses Qwen3.6- Plus (qwen3.6-plus) through the OpenCode agent, always +A. Every rollout executes in an isolated Docker sandbox with file-based I/O via BenchFlow (bench eval create -sandbox docker, commit 2a97db5), using OpenCode 1.17.7 and the claude-agent-acp integration v0.40.0; we do not override BenchFlow’s default per-run turn or token budget, and report this as an inherited harness default rather than a pinned value. Calibration rollouts were collected between 2026-06-25 and 2026-08-20. The calibration batch and all statistics derived from it use five independent rollouts per (task, configuration, condition) cell. Because both model and harness differ, comparisons between $A _ { F }$ and $A _ { \mathrm { m i d } }$ are configuration comparisons, not controlled model comparisons.

## 4.2 The artefact induces the intended task dependency

Across the fifteen candidate tasks (Fig. 2), the pooled pass rates are 68.0% for $A _ { F } , + \mathsf { A }$ 0% for $A _ { F } , - \mathsf { A }$ , and 22.7% for $A _ { \mathrm { m i d } } , + \mathrm { A }$ . Every cell contains exactly five trials, with short cells topped up by real new runs and long cells down-sampled to five by drawing uniformly at random from their valid runs – outcome-blind, fixed seed, never conditioned on which runs passed (see Table 1). Under the tested frontier configuration, the artefact is therefore operationally necessary for success in this batch, but it does not make every candidate reliably solvable.

![](images/664325e8c0fcf136f4f840d26d3f5713f725d6b53a5e13c6a40d5711dc404a42.jpg)  
Fig. 2: Fifteen-task calibration batch. Each cell reports pass rate; every cell contains exactly five trials – short cells were topped up with new runs, long cells down-sampled by drawing 5 valid runs uniformly at random (outcome-blind, fixed seed; see Table 1). Opus is evaluated with and without the artefact, while Qwen is evaluated with the artefact. Applying the configurationrelative screen retains seven of the fifteen candidates (legal-clause, paper-index, bom-costing, airline-booking, clinical-variant, mortgage-escrow, and dependency-audit); the other eight are marked for revision or removal. These constructionstage outcomes drive selection and are not an independent validation set.

We resist the stronger reading that this isolates “knowledge” from “capability.” The two configurations differ in model, harness, instruction following, tool use, long-context handling, and tolerance of the artefact’s format. The low aggregate score of $A _ { \mathrm { m i d } }$ therefore does not localise failures to the underlying model. §4.3 probes this ambiguity directly.

As a further control for whether this effect tracks the artefact’s correctness or merely its presence, we ran Opus +A on bom-costing against a perturbed artefact: the same two skill documents, near-identical in length and structure, with three gated numeric conventions changed to plausible-but-wrong values (the compounding overhead rate raised from 12% to 15%, the missing-leaf-price rule changed from “0.00” to “average of known sibling prices,” and the circular-reference value changed from $ { { } ^ { * * } } - 1 . 0 0 ^ { 9 ^ { 3 } }  { \mathrm { t o } } ^ { * * } 0 . 0 0 ^ { 3 9 } )$ . Opus achieves a 0% pass rate under the perturbed artefact, matching the no-artefact condition and falling below the 100% achieved with the correct artefact. Each condition contains five trials. A document that looks like the real artefact but states the wrong conventions is no better than no artefact at all on this task; we report this as a single-task, single-model control, not a general result.

Table 1 reports the complete task-level breakdown for this calibration batch. Under the stated screen, seven tasks are retained: legal-clause, paper-index, bomcosting, airline- booking, clinical- variant, mortgage- escrow, and dependency-audit. All seven clear the 60% frontier-configuration threshold and register a 0% pass rate without the artefact; three (airline-booking, mortgage- escrow, dependency- audit) sit exactly at the 40% secondconfiguration ceiling, and dependency-audit also sits exactly at the 60% frontier threshold. These are boundary cases under a coarse screen, not statistical classifications. Eight are rejected, in two distinct ways. Four are borderline: trial-cohort misses the $A _ { F } , + \mathsf { A }$ threshold; msa- redline, fuzz- secure, and esg- synthesis exceed the $A _ { \mathrm { m i d } } , + \mathrm { A }$ ceiling. Four fail more severely: cpi-tracker reaches only 40% under $A _ { F } , + \mathsf { A } ;$ enterprise-search, inventory-rollforward, and sec-13f remain at 0% under $A _ { F } , + \mathsf { A }$ , indistinguishable from their 0% -A baselines. These outcomes identify candidates for revision, but do not by themselves distinguish artefact defects, task difficulty, and harness failures.

Table 1: Per-task passes/runs with Wilson 95% intervals on the fifteen-task calibration batch. Withholding the artifact (−art.) yields no passes on any task; a mid-tier model with the artifact recovers few, except where the convention is not strictly gated (see §4.2). Every task is checked by a deterministic executable verifier; no task in this batch relies on LLM-judged rubric scoring.
<table><tr><td colspan="4">Task frontier —art. frontier +art. mid-tier +art.</td></tr><tr><td>Contract Clause Risk</td><td>0/5 [0.00,0.43]</td><td>4/5 [0.38,0.96]</td><td>0/5 [0.00,0.43]</td></tr><tr><td>MSA Redline Review</td><td>0/5 [0.00,0.43]</td><td>5/5 [0.57,1.00]</td><td>3/5 [0.23,0.88]</td></tr><tr><td>Publication Index</td><td>0/5 [0.00,0.43]</td><td>5/5 [0.57,1.00]</td><td>0/5 [0.00,0.43]</td></tr><tr><td>BOM Cost Rollup</td><td>0/5 [0.00,0.43]</td><td>5/5 [0.57,1.00]</td><td>0/5 [0.00,0.43]</td></tr><tr><td>Clinical Trial Cohort</td><td>0/5 [0.00,0.43]</td><td>2/5 [0.12,0.77]</td><td>0/5 [0.00,0.43]</td></tr><tr><td>Airline Booking Audit</td><td>0/5 [0.00,0.43]</td><td>5/5 [0.57,1.00]</td><td>2/5 [0.12,0.77]</td></tr><tr><td>Clinical Variant Mapper</td><td>0/5 [0.00,0.43]</td><td>5/5 [0.57,1.00]</td><td>1/5 [0.04,0.62]</td></tr><tr><td>Fuzz Secure Validator</td><td>0/5 [0.00,0.43]</td><td>5/5 [0.57,1.00]</td><td>3/5 [0.23,0.88]</td></tr><tr><td>Mortgage Escrow Analysis</td><td>0/5 [0.00,0.43]</td><td>5/5 [0.57,1.00]</td><td>2/5 [0.12,0.77]</td></tr><tr><td>ESG Synthesis</td><td>0/5 [0.00,0.43]</td><td>5/5 [0.57,1.00]</td><td>3/5 [0.23,0.88]</td></tr><tr><td>Dependency Audit</td><td>0/5 [0.00,0.43]</td><td>3/5 [0.23,0.88]</td><td>2/5 [0.12,0.77]</td></tr><tr><td>CPI Inflation Tracker</td><td>0/5 [0.00,0.43]</td><td>2/5 [0.12,0.77]</td><td>1/5 [0.04,0.62]</td></tr><tr><td>Enterprise Wide Deep Search</td><td>1 0/5 [0.00,0.43]</td><td>0/5 [0.00,0.43]</td><td>0/5 [0.00,0.43]</td></tr><tr><td>Inventory Rollforward</td><td>0/5 [0.00,0.43]</td><td>0/5 [0.00,0.43]</td><td>0/5 [0.00,0.43]</td></tr><tr><td>SEC 13F Report</td><td>0/5 [0.00,0.43]</td><td>0/5 [0.00,0.43]</td><td>0/5 [0.00,0.43]</td></tr><tr><td>All</td><td>0/75</td><td>51/75</td><td>17/75</td></tr></table>

The pooled $A _ { F }$ provided-versus-withheld difference is 68 percentage points. On tasks not constructed specifically for strict knowledge gating, concurrent work reports skill-availability deltas of 18–36 points [35]; this comparison is contextual rather than controlled because the tasks, models, and harnesses differ. The Wilson intervals in Table 1 make clear that task-level estimates remain wide at n = 5.

## 4.3 A recitation probe: locating a configuration failure

The aggregate result above cannot say whether Qwen’s low pass rate with the artefact reflects a knowledge gap, an execution gap, or both. We probe this on two of the three retained tasks whose artefacts and task data are small enough to place in a single conversation with no agent loop or tool use: bom-costing (< 2.5 KB combined) and legal-clause (< 3.5 KB combined, using a five-contract subset spanning five of its six jurisdiction multipliers). The third retained task, paper-index, requires crawling a multi-page HTML corpus and building a co-authorship graph to compute betweenness centrality; that shape does not reduce to a single-turn, no-tool probe the way the other two convention gates do, so we do not include it here rather than force an awkward synthetic stand-in.

For each task, turn one asks the model to restate the artefact’s private, non-derivable conventions alone as structured JSON (four conventions for bom-costing: the 12% overhead rate, that it compounds per level rather than applying once, that a missing leaf price is 0.00, and that a circular reference is −1.00; six for legal-clause: two keyword point values, the per-clause cap, the complexity bonus, and two jurisdiction multipliers). Turn two, continuing the same conversation, gives the task instruction and the real task data and asks for the final output, checked against each task’s own reference solution.

Across 10 independent conversations per task (20 total), Qwen recites every gated convention correctly and achieves a 100% pass rate—never once producing a correctrecitation-but-wrong-answer trial, which was the failure mode we designed this probe to detect. Taken alone, this would suggest the artefact’s content is neither a knowledge gap nor an execution gap for this model on either task. But Table 1 records Qwen at a 0% pass rate across five trials on both tasks under the full agentic harness (with-skill, sandboxed, file-based I/O). The results together localise the failure more precisely than either could alone: since the model holds and can apply the gated conventions once the problem is reduced to a single turn with the data inlined, the harness-level failure on both tasks is not explained by missing knowledge or an inability to perform the computation, and more likely lies in some combination of tool use, file discovery, and output-format compliance under the sandboxed, multi-turn agentic setting that this probe deliberately strips away.

We report this as a two-task, single-model case study, not a general result: it covers two convention gates, one model, and one reduced experimental condition that is not directly comparable to the full harness (no sandbox, no tool calls, data supplied inline rather than discovered). That the pattern held identically across both tasks is modest evidence it is not a one-off, but it also demonstrates why the calibration screen is configuration-relative: a low full-harness pass rate need not be a model knowledge deficit. Extending the probe to paper-index, other models, and a controlled common harness remains future work.

## 4.4 Contextual evidence from public paired-condition benchmarks

The following are not replications of our pipeline. These benchmarks supply knowledge but implement none of the mechanisms we propose—no enforced instruction/artefact separation, no leak audit, and no construction-stage calibration screen—so they support only the general claim that useful context raises pass rates. On $\tau ^ { 2 } .$ -bench telecom [2] (114 paired tasks, official four-trial results), adding the procedural workflow manual lifts GPT-4.1 from 34.2% to 51.8% and o4-mini from 42.1% to 59.4%. Our calculation from the official task-level results finds that, among tasks with zero baseline successes, the manual rescues 53% for GPT-4.1 and 57% for o4-mini.

SciCode reports a comparable trend for its scientist-annotated background toggle [26]: adding the background lifts Claude-3.5-Sonnet from 26.0% to 35.4% at the subproblem level, and GPT-4o from 25.0% to 35.4%. Across these external sources, the directional effect is consistent, reducing concerns that knowledge-grounding gains are solely an artefact of the calibration pool assembled for this study. What differs is the magnitude: external knowledge that is merely helpful moves pass rates by tens of points, whereas prompt-space knowledge that is necessary by construction shifts performance from absolute zero.

## 4.5 Descriptive behavior of the calibration screen

For a (task, configuration) cell with n trials and k passes, the empirical Bernoulli variance $\hat { p } ( 1 - \hat { p } )$ describes whether observed outcomes are mixed or degenerate. We report it as a compact audit of the screen’s survivors, not as a measurement of policy-gradient quality: actual training signal also depends on the learner, trajectory diversity, advantage estimation, and optimisation.

Pooling both +A configurations across all fifteen candidates gives 30 cells; “mixed” means $0 < k < n .$ . 40.0% of these cells are mixed, with mean variance 0.088. Restricting to the seven retained tasks gives 14 cells, 42.9% mixed, with mean variance 0.091. The difference is small and computed on the same outcomes used for selection.

The circularity noted in §3.5 applies directly: the retained set is selected using these configurations’ outcomes, so this comparison is not independent confirmation of improved data quality. It shows only the empirical profile produced by the stated rule in this calibration batch. We therefore do not use the 42.9% versus 40.0% difference as a headline result or evidence of training value.

Outcome variance is inexpensive to compute at construction time but remains configuration-conditioned. A held-out learner or a post-training experiment would be required to connect it to learning utility; process-level rewards [38] address a different, task-specific question.

## 5 Discussion and Future Work

## 5.1 Intended use and evidential scope

The contribution of this work is a construction and validation protocol. The retained tasks demonstrate controlled dependence on supplied artefacts under the tested configurations; they have not been shown to produce superior post-training. The same task also cannot serve simultaneously as released training data and an uncontaminated evaluation item. Any future training corpus should therefore use task families physically disjoint from evaluation, canary strings (e.g., random GUIDs) to detect contamination, and parameterised regeneration to reduce exact memorisation.

## 5.2 Durability under test-time self-improvement

As trained self-correction spreads [5,15], purely capability-gated difficulty is expected to erode: a model that re-explores under its own critique can often repair execution errors. We conjecture that tasks gated by private conventions will remain highly resistant to this loop. Because nothing in the task’s instruction or environment reveals what the planted thresholds or internal normalisation tables are, self-critique alone is unlikely to recover them. However, this durability strictly applies to non-derivable construction-time knowledge; operator gates may remain susceptible to advanced tool-use, search, or scaled test-time compute.

## 5.3 Limitations

Our calibration pool contains only fifteen tasks, with five runs per cell, so individual retention decisions are sensitive to one outcome. Automated leak audits cover nine tasks rather than the full pool, and static scanning cannot exclude semantic leakage. The two calibration configurations change both model and harness; the recitation probe shows that a low full-harness score can reflect file discovery, tool use, or output compliance rather than absent knowledge. Selection is therefore configuration-relative and intentionally biased toward a particular empirical profile. Finally, our strongest verification guarantee applies only to deterministic, checkable deliverables; every task in the present calibration batch uses this path. The protocol also supports free-form, rubric-judged deliverables for future tasks, which would inherit judge bias—named criteria improve attribution but do not turn such outputs into exact-ground-truth tasks—but this path is not exercised by any result reported here.

## 5.4 Public release scope

The experiments report a fifteen-task calibration pool, of which seven tasks satisfy the retention rule. The accompanying public repository distributes five representative retained tasks: contract clause risk scoring, publication indexing, BOM cost rollup, airline booking audit, and mortgage escrow analysis. The release is available at https://github. com/DatagridsAI/Knowledge-Gated-Task-Construction. Publication indexing is retained with its SkillsBench Vendor Sample provenance; the other four released tasks use study-specific synthetic data. The other calibration tasks are reported for transparency but are not distributed in the initial release. Consequently, the aggregate counts, table, and pass matrix in this paper describe the full calibration pool and cannot be reconstructed from the five-task public subset alone. The repository’s data-licence and third-party-notice files define the precise release scope; paper sources, evaluation trajectories, and reference solutions are not included.

## 5.5 Future work

The first next step is validation rather than training: complete executable leak audits for every task, increase repeated trials around decision boundaries, and rerun the screen with a common harness plus held-out agent configurations. An end-to-end post-training comparison can then test the separate hypothesis that calibrated tasks improve learning relative to matched ungated or unscreened tasks. Although SkillsBench [17] provides the immediate empirical inspiration, the construction contract should generalise to settings in which heterogeneous knowledge sources, tools, and environments jointly determine success. Extending beyond deterministic verifiers remains a separate frontier; evidence-lower-bound objectives over unverifiable data [24] may support more ambiguous domains, but would necessarily weaken the exact-verification guarantee central to the present protocol.

## 6 Conclusion

Treating task construction as a controlled procedure makes knowledge access an explicit, auditable variable rather than an incidental property of a prompt. Across fifteen calibration tasks, a compact artefact changes the tested frontier configuration from total failure to majority success under byte-identical instructions, while deterministic witnesses, leak audits, and named rubric criteria make the intended dependency inspectable. The configuration-relative screen identifies tasks that do and do not satisfy this contract; it does not establish that the retained tasks are better training data. The resulting contribution is therefore a method for constructing and validating knowledge-gated agent tasks, and a foundation for—not a substitute for—future training studies.

## 7 Appendix: Signal Retention under Procedural Compression

To explore how much an artefact can be compressed before the signal degrades, we conduct a single-task case study on the paper-index task. We author the knowledge artefact at three discrete sizes: the full SKILL.md (4379 B), a minimal rewrite keeping only the gated conventions while stripping explanations and worked examples (1078 B, a 4× compression), and an empty artefact (0 B). The task instruction, verifier, and harness remain byte-identical.

On this task, the minimal artefact is indistinguishable from the full document: both achieve a 100% pass rate at 1078 B and 4379 B, while the no-artefact condition achieves 0% (five trials per condition; Fig. 3). For this specific task, the operative content is a small set of conventions—an inclusion rule, a precedence order, two alias tables, and one operator convention—rather than the surrounding prose.

We emphasise what this does not establish. Two non-empty sizes both scoring 100% locate no degradation point: this shows only that the removed prose was not load-bearing on this one task, and the true breaking size lies somewhere below 1078 B, untested. The minimal artefact was written by hand rather than produced by any algorithm, the comparison covers one task in one domain at five trials per condition, and the 0 B condition is a knowledge ablation rather than a compression level. Establishing a compression–performance curve would require intermediate sizes, several tasks and domains, and per-rule ablations; we report this as a single existence proof that a kilobyte of conventions can suffice, not as a general compression result.

![](images/6ca6508096253818a119d5879427e2b4d36bd259c7027c53576862bc6789245c.jpg)  
Fig. 3: Compression–performance case study. Recovered pass rate against artefact size (linear near zero, log above – the 0 B point cannot sit on a pure log axis) for the paper-index task. The task-solving signal survives a 4× compression to a minimal 1078 B statement of conventions.

## References

1. Bae, S., Hong, J., Lee, M.Y., Kim, H., Nam, J., Kwak, D.: Online difficulty filtering for reasoning oriented reinforcement learning. arXiv preprint arXiv:2504.03380 (2025)

2. Barres, V., Dong, H., Ray, S., Si, X., Narasimhan, K.: τ<sup>2</sup>-bench: Evaluating conversational agents in a dual-control environment. arXiv preprint arXiv:2506.07982 (2025)

3. Bengio, Y., Louradour, J., Collobert, R., Weston, J.: Curriculum learning. In: Int. Conf. Mach. Learn. (2009)

4. Cazenavette, G., Wang, T., Torralba, A., Efros, A.A., Zhu, J.Y.: Dataset distillation by matching training trajectories. In: IEEE Conf. Comput. Vis. Pattern Recog. (2022)

5. Chen, G., Ye, S., Yang, W., Shen, S., Shen, G., Lin, Y.: CURE: Critique-driven unified reinforcement learning for test-time self-improvement. In: ACL (2026), outstanding Paper Award

6. Cui, P., Yang, B., Zhu, J.: Learning-zone energy: Online data selection for efficient RL post-training. arXiv preprint arXiv:2605.17003 (2026)

7. Da, J., Wang, C., Deng, X., Ma, Y., Barhate, N., Hendryx, S.: Agent-RLVR: Training software engineering agents via guidance and environment rewards. arXiv preprint arXiv:2506.11425 (2025)

8. DeepSeek-AI: DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948 (2025)

9. Gao, J., Chen, J., He, C., Xu, S., Jin, D., Wu, Y.: From self-evolving synthetic data to verifiable-reward RL: Post-training multi-turn interactive tool-using agents. arXiv preprint arXiv:2601.22607 (2026)

10. Guo, Z., Wang, K., Cazenavette, G., Li, H., Zhang, K., You, Y.: Towards lossless dataset distillation via difficulty-aligned trajectory matching. In: Int. Conf. Learn. Represent. (2024)

11. Jiang, X., et al.: KOCO-bench: Can large language models leverage domain knowledge in software development? arXiv preprint arXiv:2601.13240 (2026)

12. Jiang, Y., Li, D., Deng, H., Ma, B., Wang, X., Wang, Q., Yu, G.: SoK: Agentic skills — beyond tool use in LLM agents. arXiv preprint arXiv:2602.20867 (2026)

13. Jimenez, C.E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., Narasimhan, K.: SWE-bench: Can language models resolve real-world GitHub issues? In: Int. Conf. Learn. Represent. (2024)

14. Kimi Team: Kimi K2: Open agentic intelligence. arXiv preprint arXiv:2507.20534 (2025)

15. Kumar, A., Zhuang, V., Agarwal, R., Su, Y., Co-Reyes, J.D., Singh, A., Baumli, K., Iqbal, S., Bishop, C., Roelofs, R., Zhang, L.M., McKinney, K., Shrivastava, D., Paduraru, C., Tucker, G., Precup, D., Behbahani, F., Faust, A.: Training language models to self-correct via reinforcement learning. arXiv preprint arXiv:2409.12917 (2024)

16. Lambert, N., Morrison, J., Pyatkin, V., Huang, S., Ivison, H., Brahman, F., Miranda, L.J.V., Liu, A., Dziri, N., Lyu, S., Gu, Y., Malik, S., Graf, V., Hwang, J.D., Yang, J., Bras, R.L., Tafjord, O., Wilhelm, C., Soldaini, L., Smith, N.A., Wang, Y., Dasigi, P., Hajishirzi, H.: Tülu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124 (2024)

17. Li, X., Liu, Y., Chen, W., You, B., Di, Z., He, Y., Zheng, S., Choe, K.W., Sun, J., Wang, S., Tao, C., Li, B., Zhao, X., Geng, H., Wu, X., Zhou, J., Chen, X., Xing, H., Li, Y., Zeng, Q., Wang, D., Wang, Y., Ben Chaim, R., Jiang, P., Shen, H., Kong, L., Liu, X., Wang, R., Liu, X., Li, J., Lan, X., Lin, Y., Ye, W., He, J., Li, S., Zhang, Y., Gao, Y., Li, Y., Ma, Z., Jing, L., Wang, T., Li, K., Xue, Y., Lyu, H., He, Y., Tian, Y., Wu, S., Wang, B., Gao, Y., Chen, B., Liu, L., Cheng, S., Bao, J., Tong, S., Xu, S., Zhuo, T.Y., Ye, T., Qi, Q., Li, M., Liao, L., Tan, Z., Shi, C., Tang, X., Tankasala, S., Yuan, B., Qian, Y., Tu, J., Wang, C., Sun, Y., Wang, W., Taylor, A., Yang, Z., Guan, C., Dong, Z., Zhang, X., Dillmann, S., Lee, H.c., Song, D.: SkillsBench: Benchmarking how well agent skills work across diverse tasks. arXiv preprint arXiv:2602.12670 (2026)

18. Liu, Z., Hoang, T., Zhang, J., Zhu, M., Lan, T., Kokane, S., Tan, J., Yao, W., Liu, Z., Feng, Y., Murthy, R., Yang, L., Savarese, S., Niebles, J.C., Wang, H., Heinecke, S., Xiong, C.: APIGen: Automated pipeline for generating verifiable and diverse function-calling datasets. In: Adv. Neural Inform. Process. Syst. (2024)

19. Pan, J., Wang, X., Neubig, G., Jaitly, N., Ji, H., Suhr, A., Zhang, Y.: Training software engineering agents and verifiers with SWE-Gym. arXiv preprint arXiv:2412.21139 (2024)

20. Patwardhan, T., et al.: GDPval: Evaluating AI model performance on real-world economically valuable tasks. arXiv preprint arXiv:2510.04374 (2025)

21. Sorscher, B., Geirhos, R., Shekhar, S., Ganguli, S., Morcos, A.S.: Beyond neural scaling laws: Beating power law scaling via data pruning. In: Adv. Neural Inform. Process. Syst. (2022)

22. Su, H., Sun, R., Yoon, J., Yin, P., Yu, T., Arık, S.Ö.: Learn-by-interact: A data-centric framework for self-adaptive agents in realistic environments. arXiv preprint arXiv:2501.10893 (2025)

23. Sun, Q., Cheng, K., Ding, Z., Jin, C., Wang, Y., Xu, F., Wu, Z., Jia, C., Chen, L., Liu, Z., Kao, B., Li, G., He, J., Qiao, Y., Wu, Z.: OS-Genesis: Automating GUI agent trajectory construction via reverse task synthesis. arXiv preprint arXiv:2412.19723 (2024)

24. Tang, Y., Wang, S., Madaan, L., Munos, R.: Beyond verifiable rewards: Scaling reinforcement learning for language models to unverifiable data. In: Adv. Neural Inform. Process. Syst. (2025)

25. Tian, H., Reddy, K., Feng, Y., Quddus, M., Demiris, Y., Angeloudis, P.: Enhancing autonomous vehicle training with language model integration and critical scenario generation. arXiv preprint arXiv:2404.08570 (2024)

26. Tian, M., Gao, L., Zhang, S.D., Chen, X., Fan, C., Guo, X., Haas, R., Ji, P., Krongchon, K., Li, Y., Liu, S., Luo, D., Ma, Y., Tong, H., Trinh, K., Tian, C., Wang, Z., Wu, B., Xiong, Y., Yin, S., Zhu, M., Lieret, K., Lu, Y., Liu, G., Du, Y., Tao, T., Press, O., Callan, J., Huerta, E., Peng, H.: SciCode: A research coding benchmark curated by scientists. In: Adv. Neural Inform. Process. Syst. (2024)

27. Vania, C., Htut, P.M., Huang, W., Mungra, D., Pang, R.Y., Phang, J., Liu, H., Cho, K., Bowman, S.R.: Comparing test sets with item response theory. In: ACL (2021)

28. Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., Anandkumar, A.: Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291 (2023)

29. Wang, J., Bianchi, F., Zhu, S., Nie, F., Kwon, Y., Dhingra, B., Zou, J.: Automated benchmark auditing for AI agents and large language models. arXiv preprint arXiv:2605.26079 (2026)

30. Wang, T., Zhu, J.Y., Torralba, A., Efros, A.A.: Dataset distillation. arXiv preprint arXiv:1811.10959 (2018)

31. Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N.A., Khashabi, D., Hajishirzi, H.: Selfinstruct: Aligning language models with self-generated instructions. In: ACL (2023)

32. Wang, Z., Xu, C., Liu, B., Wang, Y., Han, S., Yao, Z., Yao, H., He, Y.: Agent world model: Infinity synthetic environments for agentic reinforcement learning. In: Int. Conf. Mach. Learn. (2026)

33. Xiong, W., Ye, C., Liao, B., Dong, H., Xu, X., Monz, C., Bian, J., Jiang, N., Zhang, T.: Reinforce-ada: An adaptive sampling framework under non-linear RL objectives. arXiv preprint arXiv:2510.04996 (2025)

34. Xu, C., Sun, Q., Zheng, K., Geng, X., Zhao, P., Feng, J., Tao, C., Jiang, D.: WizardLM: Empowering large language models to follow complex instructions. In: Int. Conf. Learn. Represent. (2024)

35. Xu, X., Wu, W.: Skill availability and presentation granularity in large-language-model agents: A controlled SkillsBench study. arXiv preprint arXiv:2605.31408 (2026)

36. Yang, J., Leret, K., Jimenez, C.E., Wettig, A., Khandpur, K., Zhang, Y., Hui, B., Press, O., Schmidt, L., Yang, D.: SWE-smith: Scaling data for software engineering agents. arXiv preprint arXiv:2504.21798 (2025)

37. Yao, S., Shinn, N., Razavi, P., Narasimhan, K.: τ-bench: A benchmark for tool-agent-user interaction in real-world domains. arXiv preprint arXiv:2406.12045 (2024)

38. Yuan, H., Xu, Z., Wang, H., Yi, X., Gao, J., Zhang, X.P., Wang, Y., Yu, C., Wu, Y.: Verifiable process rewards for agentic reasoning. arXiv preprint arXiv:2605.10325 (2026)

39. Zhao, A., Wu, Y., Yue, Y., Wu, T., Xu, Q., Yue, Y., Lin, M., Wang, S., Wu, Q., Zheng, Z., Huang, G.: Absolute zero: Reinforced self-play reasoning with zero data. arXiv preprint arXiv:2505.03335 (2025)

40. Zhong, J., Li, M., Lai, Y., Yang, Z., Xie, J., Kil, J., Mai, Z., Lo, S.Y., Xiang, R., Psounis, K., Lei, Y.: Agentic context learning with self-discovered specification. arXiv preprint arXiv:2607.09794 (2026)

41. Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X., Efrat, A., Yu, P., Yu, L., Zhang, S., Ghosh, G., Lewis, M., Zettlemoyer, L., Levy, O.: LIMA: Less is more for alignment. In: Adv. Neural Inform. Process. Syst. (2023)