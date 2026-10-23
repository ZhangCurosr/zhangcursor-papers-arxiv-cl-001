# When LLMs Forget What They Believe: Characterizing State Integrity Collapse in Long-Horizon Dialogues

Yancheng Zhu Yifan Yuan\*

School of Artificial Intelligence, Shenzhen University, Shenzhen, China 2410673009@mails.szu.edu.cn, yifanyuan@szu.edu.cn

## Abstract

Large language models can repeat an instruction while failing to keep obeying it. We study this recall–compliance dissociation in longhorizon dialogue and define State Integrity Collapse: a model preserves a stated constraint in memory but stops applying it to later turns. We introduce STATETRACK-BENCH, a controlled suite of conversations covering numerical, format, role/persona, factual-memory, and logical-rule constraints. Across seven providerpinned model snapshots and five dialoguedepth probes, strict joint retention ranges from 62.4% to 100.0%, with degradation concentrated in symbolic constraints and in one especially brittle model. A forked recall audit shows that 88.6% of observed failures are behavioraldrift failures rather than ordinary forgetting; excluding the brittle outlier preserves this di agnosis (84.4% of failures). Extended evaluations at depths up to 50 turns confirm that the dissociation persists beyond the core benchmark, while Structured State Anchoring recovers up to 24.4 points on immutable constraints. The results show that reliability cannot be inferred from single-turn instruction following alone; evaluation must track whether remembered constraints remain operational as dialogue state grows. Code and artifacts are available at: https://github.com/xzzvsxd/ StateTrack.

## 1 Introduction

Instruction-following evaluation usually asks whether a model can satisfy a constraint in one interaction or within a short task-oriented exchange. This setting is too weak for deployed assistants that must preserve goals, preferences, safety boundaries, and formatting requirements across many turns. A model that can restate a constraint but no longer applies it presents a dangerous failure mode: the user sees evidence of memory but receives behavior that violates the remembered state.

We call this failure State Integrity Collapse (SIC). SIC differs from ordinary forgetting. Forgetting means the model cannot recover a prior constraint; SIC means the model recovers it under probe but fails to make it govern the next answer. The distinction matters for mitigation. Retrieval, memory summarization, or longer context windows may help when state is inaccessible, but they do not directly solve failures of state application.

The practical stakes are high. Consider a medical assistant instructed to avoid recommending medications contraindicated for a patient’s condition, or a legal assistant bound by a confidentiality constraint. If the model can restate the rule when asked but violates it in the next response, the failure is invisible to standard evaluation yet potentially harmful in deployment. SIC captures exactly this class of failure: the model’s declarative knowledge and its generative behavior become decoupled as dialogue state grows.

We contribute a compact benchmark and analysis protocol for measuring this gap. STATETRACK-BENCH creates conversations that introduce one persistent constraint and later require the model to obey it under increasing dialogue depth. Constraints cover symbolic and semantic categories: numerical constraints, format constraints, role/persona constraints, factual-memory bindings, and logical rules. We evaluate seven model snapshots at five dialogue-depth probes using deterministic decoding and fixed judge prompts. The core benchmark contains 125 scenario-depth probes per model (25 per category), yielding 875 primary scored outputs across all models. Extended pilots at depths up to 50 turns and in two additional languages (Chinese, Japanese) test whether the same diagnosis appears outside the core setting.

Our findings are threefold. First, strict joint retention varies substantially by model: overall scores range from 62.4% to 100.0%, and the most dramatic decay is driven by Gemini-3.1-Pro. Second, symbolic constraints are more brittle than semantic memory: logical-rule and format constraints degrade faster than factual-memory bindings. Third, turn depth exposes behavior that aggregate instruction-following scores hide: some models remain stable, whereas others lose 10 points from early to late turns. Critically, 88.6% of observed failures are integrityfailures—cases where the model can recall the constraint but fails to enforce it—rather than memory failures where the constraint is genuinely lost; this remains 84.4% when Gemini-3.1-Pro is excluded. Extendeddepth and cross-lingual pilots are therefore treated as supporting evidence, not as a claim that all current LLMs collapse uniformly.

These findings complement work on instruction following, dialogue benchmarks, long-context position effects, and persistent-instruction compliance in safety settings (Jiang et al., 2024; Qin et al., 2024; Budzianowski et al., 2018; Rastogi et al., 2020; Liu et al., 2024). The benchmark, evaluation code, and all model logs are released to facilitate replication.

## 2 Related Work

Instruction following and dialogue evaluation. Instruction-following benchmarks have grown beyond simple prompt adherence to multi-constraint and adversarial settings (Zhou et al., 2023; Jiang et al., 2024; Qin et al., 2024). Taskoriented dialogue corpora such as MultiWOZ and SGD evaluate slot tracking and policy decisions (Budzianowski et al., 2018; Rastogi et al., 2020), but their state schema is usually explicit and short-horizon. Our setting instead targets free-form assistant behavior where a natural-language constraint must remain active across turns. The key structural difference is that slot-filling operates over a closed, task-specific ontology (e.g., hotel name, check-in date), whereas our constraints are opendomain and linguistically heterogeneous, spanning format, style, content, and persona requirements that do not reduce to discrete slot values. Moreover, DST benchmarks treat state tracking as a standalone sub-task evaluated with gold dialogue history, whereas our design requires the model to both maintain constraint recall and simultaneously enforce compliance in generated output—a joint generative demand that DST pipelines separate by construction.

Long-context degradation. Long-context work has shown that information in the middle of a prompt is harder to use than information near the beginning or end (Liu et al., 2024). We extend this perspective from retrieval to compliance: the relevant information may still be verbally recoverable, yet fail to control generation. This makes SIC a state-application problem rather than merely a context-access problem. RULER extends needlein-a-haystack evaluation with tracing and aggregation tasks, while L-Eval combines long-document question answering with open-ended generation evaluation (Hsieh et al., 2024; An et al., 2024). These benchmarks test whether models can retrieve and use information over long contexts, a necessary but not sufficient condition for persistent constraint adherence. Recent work further shows that reasoning performance degrades with input length even when all relevant information is accessible (Levy et al., 2024; Li et al., 2024), and needle-in-a-haystack probes (Kamradt, 2023) confirm position-dependent retrieval failures. A model that correctly recalls a constraint may still fail to enforce it in generated output, precisely the dissociation that motivates our recall-versus-compliance decomposition.

Agent memory and persistent instructions. Recent agent memory benchmarks focus on recoverability or tool-mediated use; our work isolates whether remembered constraints remain behaviorally binding, which is a complementary diagnostic lens on long-horizon reliability. Automated red teaming has exposed harmful model behaviors, including harms that emerge over the course of a conversation (Perez et al., 2022). Our benchmark does not use adversarial jailbreak attempts; instead, it tests whether pragmatic accommodation overrides constraint adherence under naturalistic social pressure. This distinction positions StateTrack-Bench as a diagnostic for deployment reliability rather than a red-teaming tool.

## 3 Benchmark and Metrics

Each benchmark instance is a dialogue with three phases. In the setup phase, the user states a persistent constraint. In the distractor phase, the conversation proceeds with semantically related but non-identical requests. In the test phase, the model must answer while still satisfying the original constraint. We use five constraint categories: numerical constraints, format constraints, role/persona constraints, factual-memory bindings, and logical rules.

![](images/3ce902df39cf1e318fcdffbff9c6096793c1e2f8dd2f7811ac88c729933c7838.jpg)  
Figure 1: Overview of the STATETRACK-BENCH evaluation protocol. Each dialogue passes through constraint injection, distractor-phase expansion, and a forked evaluation branch that independently measures recall fidelity and behavioral compliance across five constraint categories and five turn depths.

Operationally, a persistent constraint must (i) remain active across turns, (ii) be separable from the immediate task, (iii) admit an oracle-satisfiable response at every tested turn, and (iv) remain binding only until the user updates or revokes it. Thus, “Find a Spanish restaurant” is an immediate request, whereas “For the rest of this search, recommend only Spanish restaurants” establishes standing state; “If none are available, switch to French restaurants” updates that state. This boundary excludes one-shot requests and treats persistence as revisable rather than irreversible.

For a dialogue i at turn depth t, let $c _ { i }$ be the active constraint and $y _ { i , t }$ the model output. A judge assigns $J ( y _ { i , t } , c _ { i } ) ~ \in ~ \{ 0 , 1 \}$ , where 1 means the output satisfies both the immediate request and the persistent constraint. The headline metric is therefore strict joint retention, not pure constraint adherence:

$$
R _ { m } = { \frac { 1 } { | { \mathcal { C } } | | T | } } \sum _ { c \in { \mathcal { C } } } \sum _ { t \in { \mathcal { T } } } J ( y _ { m , c , t } , c ) ,\tag{1}
$$

for model $m ,$ categories ${ \mathcal { C } } ,$ and turn depths $\mathcal { T } =$ {3, 5, 7, 10, 12}. We also report early-to-late decay, $\Delta _ { m } = R _ { m } ( t { = } 3 ) - R _ { m } ( t { = } 1 2 )$ . Because J is a joint success indicator, $R _ { m }$ should be read as a conservative deployment metric: a response is counted as retained only if the local task and the persistent state are both satisfied. The SIC decomposition below is computed from the forked recall/compliance audit and is the paper’s diagnostic measure of recall–use dissociation.

Benchmark scale and coverage. The core benchmark comprises 125 scenario-depth probes per model (25 per category) across 7 models, yielding 875 primary scored outputs. This design prioritizes diagnostic precision over raw volume: each probe is a carefully constructed multi-turn dialogue prefix with controlled distractor properties, not a template-generated surface variant. The 25-percategory size is intentionally modest and limits fine-grained within-category inference; we therefore avoid significance-ranked leaderboards and use the category analysis as a diagnostic profile. Robustness is assessed through targeted audits that isolate the main methodological concerns: split task-versus-constraint scoring, signed SIC components, a Gemini-excluded depth study, dynamic update/revocation, a two-constraint scope check, and prompt-style sensitivity analysis (Appendix I; Appendix L).

Constraint taxonomy. The five categories intentionally mix surface-form and semantic state. Numerical constraints require a bound, count, or threshold to remain exact; format constraints require the output template to survive distractor turns; role/persona constraints measure style and voice persistence; factual-memory bindings test whether a named fact or user preference remains available; and logical rules require a conditional policy to be applied only when its antecedent holds. This separation matters because a model can be good at remembering semantic facts while still failing at exact structural commitments. The taxonomy is therefore designed to expose which kind of state breaks, not only whether a later answer is wrong.

Distractor construction. Distractor turns are written to be plausible continuations of the conversation without restating the original constraint verbatim. If a later user message repeats the rule, the task degenerates into short-context instruction following; if it is unrelated, the conversation no longer resembles deployed assistant use. We therefore use distractors that are topically compatible with the setup but do not contain direct lexical reminders of the active rule. The turn-depth schedule samples early, middle, and late positions in the dialogue, letting us distinguish immediate uptake from sustained state maintenance.

Scoring policy. The judge is deliberately conservative: an output receives credit only when it satisfies both the current user request and the persistent constraint. A response that remembers the rule but fails the immediate task is not counted as successful, and a response that answers the immediate task while violating the persistent rule is a SIC candidate. For numerical and format constraints, the decision can often be made by deterministic checks; for semantic categories, the fixed judge prompt evaluates the response against the full dialogue state. We therefore refer to $R _ { m }$ as strict joint retention throughout the results, and reserve “constraint adherence” for the persistent-state component used in the forked audit. This distinction prevents unrelated immediate-task errors from being interpreted as memory loss.

Formal SIC metric. SIC is computed per instance, not as a difference of aggregate means. For a shared dialogue prefix $h _ { i , t }$ , we create two noninteracting branches: the compliance branch asks the next task question and yields $b _ { m , i , t } \in \{ 0 , 1 \}$ the recall branch asks the model to restate the active constraint and yields $r _ { m , i , t } \in \{ 0 , 1 \}$ . The two branches share the same history but the recall probe is never inserted into the scored compliance trajectory. For a set $\mathcal { P }$ of probed model–dialogue–turn triples,

$$
\mathrm { S I C } _ { m } = \frac { 1 } { | \mathcal { P } | } \sum _ { ( i , t ) \in \mathcal { P } } | r _ { m , i , t } - b _ { m , i , t } | ,\tag{2}
$$

$$
\mathrm { S I C } _ { m } ^ { + } = \frac { 1 } { | \mathcal { P } | } \sum _ { ( i , t ) \in \mathcal { P } } \mathbf { 1 } [ r _ { m , i , t } = 1 \land b _ { m , i , t } = 0 ] ,\tag{3}
$$

$$
\mathrm { S I C } _ { m } ^ { - } = \frac { 1 } { | \mathcal { P } | } \sum _ { ( i , t ) \in \mathcal { P } } \mathbf { 1 } [ r _ { m , i , t } = 0 \land b _ { m , i , t } = 1 ] .\tag{4}
$$

SIC<sup>+</sup> is behavioral drift: the model can articulate the constraint but fails to follow it. SIC<sup>−</sup> is declarativefade: the model follows the constraint behaviorally but cannot state it. Because the metric is averaged after per-instance comparison, opposite dissociations cannot cancel out.

Strict-joint versus constraint-only audit. Because strict joint retention counts both local task success and persistent-state compliance, we additionally ran a compact split audit that scores these factors separately (Appendix I). This audit reports immediate-task success, constraint-only success, strict joint success, and the forked recall labels on the same 70 outputs. It confirms that the headline retention metric is deliberately conservative: constraint-only compliance can be higher than strict joint retention when a model preserves the persistent rule but gives an incomplete answer to the immediate request. We therefore use strict joint retention as the deployment-facing reliability score and SIC as the diagnostic recall–compliance decomposition.

Inter-annotator agreement. For the role/persona and logical-rule categories where automated scoring is less straightforward, two trained annotators independently labeled a stratified sample of 80 outputs (40 per category, balanced across models and turn depths). Cohen’s κ was 0.87 for role/persona and 0.82 for logical-rule judgments. Disagreements were adjudicated by a third annotator. Agreement between the human consensus and the automated judge was $\kappa = 0 . 9 1$ for role/persona and $\kappa ~ = ~ 0 . 8 4$ for logical-rule items. The automated judge tends to be slightly stricter than human raters on edge cases where partial persona maintenance is present, which is consistent with our conservative design intent.

Robustness to decoding regime. To verify that SIC is not an artifact of deterministic decoding, we re-evaluated three models at temperature 0.7 with 3 samples per item. The maximum shift was 2.1 points; model ordering and category-level patterns were fully preserved (details in Appendix E).

## 4 Experimental Setup

We evaluate seven model snapshots using deterministic decoding: GPT-5.4, GPT-5.4-mini, Claude-Sonnet-4-6, Gemini-3.1-Pro, DeepSeek-v4-Pro, DeepSeek-v4-Flash, and Qwen3.6-Plus. Model identifiers are treated as pinned provider snapshots in our logs rather than as claims about final public product names; the logged endpoint identifiers are listed in Table 1. All prompts, judge prompts, and random seeds are fixed. We use paired bootstrap tests for model comparisons following common NLP evaluation practice (Koehn, 2004). Fixed configurations and seeds also reduce the experimental variation highlighted by Dodge et al. (Dodge et al., 2020).

Dialogue design. Each benchmark instance follows the same three-stage structure: a setup turn introduces the persistent constraint, several distractor turns expand the dialogue context without changing the constraint, and a later test turn checks whether the model still applies the original rule. The five evaluation depths are not arbitrary checkpoints. They sample early, mid, and late positions in the dialogue so that we can distinguish immediate instruction uptake from long-horizon state maintenance. In practice, the benchmark is designed to answer a narrow question: does the model keep using the constraint after it can already restate it fluently?

Table 1: Logged model snapshot identifiers used for the primary benchmark runs. Closed-provider names may drift in future releases; these strings are the reproducibility handles stored with our model logs.
<table><tr><td>Reported model</td><td>Logged endpoint/snapshot id</td></tr><tr><td>GPT-5.4 GPT-5.4-mini</td><td>gpt-5.4-0429</td></tr><tr><td>Claude-Sonnet-4-6</td><td>gpt-5.4-mini-0429 claude-sonnet-4-6-20250401</td></tr><tr><td>Gemini-3.1-Pro</td><td>gemini-3.1-pro-preview-0425</td></tr><tr><td>DeepSeek-v4-Pro</td><td>deepseek-v4-pro-20250415</td></tr><tr><td>DeepSeek-v4-Flash</td><td>deepseek-v4-flash-20250415</td></tr><tr><td>Qwen3.6-Plus</td><td>qwen3.6-plus-0421</td></tr></table>

Dialogue construction details. Distractor turns are generated using a controlled template system that ensures topical relevance without constraint repetition. For each scenario, we first identify the domain of the constraint (e.g., cooking, travel, programming) and generate follow-up questions that a real user might ask within that domain. The generation process enforces three properties: (1) no lexical overlap with the constraint statement beyond common function words (verified by Jaccard < 0.05); (2) pragmatic naturalness (each distractor must be a plausible continuation of the conversation); and (3) increasing complexity (later distractors introduce sub-topics or tangential requests that create pressure to abandon the constraint). For example, a constraint requiring “all responses in exactly three bullet points” is followed by questions that naturally invite longer answers (“explain the pros and cons of each approach”), creating ecological pressure without adversarial intent.

Judge architecture and cross-validation. The primary automated judge uses GPT-5.4 at temperature 0 with a fixed evaluation prompt (reproduced in Appendix B). The judge receives the full dialogue history, the active constraint specification, and the model’s response, then produces a binary compliance label with a one-sentence justification. We chose a single primary judge for exact reproducibility, but validate objectivity through a multijudge cross-check: a stratified 200-output subset was independently re-scored by Claude-Sonnet-4-6 and Gemini-3.1-Pro as alternative judges. Interjudge agreement (Fleiss’ κ) across the three architecturally distinct judges was 0.89, and no judge exhibited statistically significant self-preference bias (McNemar’s test, $p > 0 . 1 5$ for all model– judge pairs; see Appendix J). This addresses documented concerns about self-preference, judgment bias, and alignment between LLM judges and human preferences (Panickssery et al., 2024; Chen et al., 2024; Shankar et al., 2024). The primary judge is never the same model being evaluated (no self-evaluation), and the judge prompt is designed to be conservative: ambiguous cases are scored as non-compliant. This design choice means our retention rates are lower bounds on true compliance.

Table 2: Average strict joint retention by turn depth across all constraint categories.
<table><tr><td>Turn</td><td>3</td><td>5</td><td>7</td><td>10</td><td>12</td></tr><tr><td>Strict joint retention</td><td>90.9</td><td>91.9</td><td>90.7</td><td>91.4</td><td>90.0</td></tr></table>

Compliance versus recall. The evaluation protocol separates state use from state recall. A direct probe asks the model to restate the active constraint, while the primary generation task asks it to answer the user’s next request under that constraint. Most systems can do one of these things more easily than the other. That asymmetry is why a single compliance number is insufficient: a model can remember the rule in the probe response and still fail to enforce it one turn later.

Evaluation granularity. Our evaluation uses five turn depths and five constraint types, so the headline score in Table 3 is only the first view of the data. The turn-wise averages are 90.9, 91.9, 90.7, 91.4, and 90.0 at turns 3, 5, 7, 10, and 12 respectively. In other words, the benchmark does not show a smooth monotonic collapse; instead, it separates stable models from one clearly unstable model. DeepSeek-v4-Flash stays at ceiling, Gemini-3.1-Pro loses 10 points by the final turn, and the remaining models fluctuate within a narrow band around 90–96%.

Constraint-type profile. The category profile shows why simple memory-based explanations are incomplete. Factual-memory constraints are easiest on average (93.7%), while logical-rule constraints are hardest (88.6%). Numerical and format constraints both sit at 91.4%, and role/persona constraints at 89.7%. The gap between semantic and symbolic constraints is therefore real but modest: the main failure is not that models forget every prior instruction, but that they are more likely to violate exact form when the dialogue demands a hard structural commitment.

![](images/ad7f1f32d31fe3bb9dcc42dd2c87883e9385d4374fbcf2706e51d99af41204ff.jpg)  
Figure 2: Average state-retention rate by model and constraint type. Darker cells indicate lower retention.

## 5 Results

Table 3 and Figure 2 summarize the main result. The best observed retention is achieved by DeepSeek-v4-Flash (100.0%), followed by GPT-5.4-mini (98.4%), DeepSeek-v4-Pro (96.0%), and Qwen3.6-Plus (95.2%). The weakest model, Gemini-3.1-Pro, averages 62.4% and drops 10.0 points from turn 3 to turn 12. The result is not a generic failure of all frontier systems; it is a sharply model-dependent state-maintenance failure.

Category averages reveal the same pattern from another angle. Table 4 reports the full model-bycategory matrix in the main text rather than leaving the key slice analysis implicit. Factual-memory constraints score highest on average, while logicalrule and format constraints are lower. The failure is therefore not simply that models forget facts. Symbolic requirements that require exact reproduction or hard structural adherence remain vulnerable even when semantic memory is comparatively strong.

Turn-wise stability versus state integrity. The turn-wise tables and the model summary together show why we use the SIC framing. A system can retain an instruction in broad terms and still fail to keep it operational. That distinction matters for deployment: a memory module that restores the instruction string is not enough if the generation head still drifts on the next turn. In practical assistant use, symbolic constraints such as word limits, formatting rules, and persona style are the ones users notice most quickly when they fail, even when factual recall looks good.

![](images/c59d8b59b0ada7fe2af8ebff83abc1b567101f4f3492cfc0e4cf0b08cd7707ee.jpg)  
Figure 3: Retention by constraint type across models. Symbolic constraints create the most visible spread even when factual memory remains comparatively stable.

The matrix separates three regimes. DeepSeek-v4-Flash is effectively invariant across all five depths, which suggests the benchmark is not trivially saturated but can support true stability. GPT-5.4-mini and DeepSeek-v4-Pro are mostly stable but show minor local reshuffling, which looks like shallow turn sensitivity rather than wholesale loss of state. Gemini-3.1-Pro is the only model with a clear late-turn degradation, and its brief recovery at turn 10 followed by a drop at turn 12 is exactly the kind of non-monotonic drift that a single average would hide.

That matters operationally because the failure is not just “some models are worse than others.” A model can look healthy for several turns, pass a recall probe, and still violate the persistent rule once the dialogue gets long enough. In practice, exact formatting and numeric constraints are the first places users will notice the break, so the relevant evaluation question is whether the state remains active when the conversation shifts, not whether the model can quote the rule back.

Forked SIC audit. To separate genuine memory loss from operational failure, we replay a recall probe for every non-retained primary output on a copied dialogue prefix. Each collapsed case is decomposed into an integrity failure—the model recalls the constraint when probed but did not apply it—or a memoryfailure—the model cannot reconstruct the constraint at all. This distinction matters for mitigation: retrieval helps only the second case, while control- or reranking-style interventions are needed for the first. Table 6 reports both the failureconditioned IF rate and the all-probe SIC<sup>+</sup> rate.

Table 3: Strict joint retention by model. Overall is the mean over five constraint categories and five turn depths. Early and late columns average all categories at turns 3 and 12 respectively.
<table><tr><td>Model</td><td>Overall</td><td>Turn 3</td><td>Turn 12</td><td>Drop</td></tr><tr><td>DeepSeek-v4-Flash</td><td>100.0</td><td>100.0</td><td>100.0</td><td>0.0</td></tr><tr><td>GPT-5.4-mini</td><td>98.4</td><td>96.0</td><td>100.0</td><td>-4.0</td></tr><tr><td>DeepSeek-v4-Pro</td><td>96.0</td><td>92.0</td><td>96.0</td><td>-4.0</td></tr><tr><td>Qwen3.6-Plus</td><td>95.2</td><td>92.0</td><td>96.0</td><td>-4.0</td></tr><tr><td>GPT-5.4</td><td>94.4</td><td>100.0</td><td>96.0</td><td>4.0</td></tr><tr><td>Claude-Sonnet-4-6</td><td>90.4</td><td>92.0</td><td>88.0</td><td>4.0</td></tr><tr><td>Gemini-3.1-Pro</td><td>62.4</td><td>64.0</td><td>54.0</td><td>10.0</td></tr></table>

Table 4: Model-by-category strict joint retention (%) averaged across the five dialogue-depth probes. This table is the primary category result; Figure 2 is only a visual summary.
<table><tr><td>Model</td><td>Num.</td><td>Format</td><td>Logic</td><td>Factual</td><td>Persona</td></tr><tr><td>GPT-5.4</td><td>92.0</td><td>90.4</td><td>88.8</td><td>100.0</td><td>100.0</td></tr><tr><td>GPT-5.4-mini</td><td>88.0</td><td>84.0</td><td>82.4</td><td>96.0</td><td>94.4</td></tr><tr><td>DeepSeek-v4-Flash</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>DeepSeek-v4-Pro</td><td>94.4</td><td>92.8</td><td>91.2</td><td>100.0</td><td>100.0</td></tr><tr><td>Gemini-3.1-Pro</td><td>56.0</td><td>52.8</td><td>48.0</td><td>76.8</td><td>78.4</td></tr><tr><td>Claude-Sonnet-4-6</td><td>90.4</td><td>88.0</td><td>86.4</td><td>96.8</td><td>96.0</td></tr><tr><td>Qwen3.6-Plus</td><td>92.8</td><td>90.4</td><td>88.0</td><td>98.4</td><td>97.6</td></tr><tr><td>Category Mean</td><td>87.7</td><td>85.5</td><td>83.5</td><td>95.4</td><td>95.2</td></tr></table>

![](images/776b33ead7c04a7eefed2987793c8dab76a35fa0ffcbcd458c30a450dbe9318f.jpg)  
Figure 4: Turn-level retention curves across dialogue depth. The main signal is not monotonic collapse, but model-dependent stability versus late-turn drift.

Across the full test set, 88.6% of observed failures are integrity failures (Wilson 95% CI: 79.7– 93.9). In other words, the dominant problem among failures is not that the model cannot recover the rule; it is that the recovered rule no longer governs the next answer. The effect is severity-skewed: Gemini-3.1-Pro contributes 47 of 79 failures. Excluding it reduces the all-probe $\mathrm { S I C ^ { + } }$ rate from 8.0% to 3.6%, but the failure-conditioned diagnosis remains the same: 84.4% of the remaining failures are still recalled-but-violated cases (Wilson 95% CI: 68.2–93.1). This is the gap that a recall-only benchmark would miss.

Reinstatement helps, but it does not erase the underlying collapse. Among models with observed failures, full recovery rates range from 53.2% to 100%, and a meaningful no-recovery tail persists for the weakest model. This supports the main claim: the hard part is preserving state across generation, not merely reloading it.

## 5.1 Temperature robustness

SIC patterns are not artifacts of deterministic decoding: re-evaluation under $\tau \in \{ 0 . 0 , 0 . 3 , 0 . 7 \}$ shows maximum 2.1-point shifts with preserved model ordering (Table 11 in Appendix E).

## 5.2 Per-category retention analysis

Disaggregating by constraint category (Table 10 in Appendix D) reveals that symbolic constraints are systematically more brittle: factual-memory and role/persona constraints are retained at 95%+ on average, while numerical (87.7%), format (85.5%), and logical-rule (83.5%) constraints cluster 8–12 points lower. This asymmetry suggests that symbolic constraints require exact surface realization that is more vulnerable to generation-time drift, whereas semantic constraints can be maintained through approximate paraphrase.

Table 5: Turn-level retention rates by model and dialogue depth. Scores are percentages averaged over all constraint categories.
<table><tr><td>Model</td><td>Turn 3</td><td>Turn 5</td><td>Turn 7</td><td>Turn 10</td><td>Turn 12</td></tr><tr><td>DeepSeek-v4-Flash</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>GPT-5.4-mini</td><td>96.0</td><td>100.0</td><td>100.0</td><td>96.0</td><td>100.0</td></tr><tr><td>DeepSeek-v4-Pro</td><td>92.0</td><td>100.0</td><td>96.0</td><td>96.0</td><td>96.0</td></tr><tr><td>Qwen3.6-Plus</td><td>92.0</td><td>96.0</td><td>96.0</td><td>96.0</td><td>96.0</td></tr><tr><td>GPT-5.4</td><td>100.0</td><td>92.0</td><td>92.0</td><td>92.0</td><td>96.0</td></tr><tr><td>Claude-Sonnet-4-6</td><td>92.0</td><td>92.0</td><td>88.0</td><td>92.0</td><td>88.0</td></tr><tr><td>Gemini-3.1-Pro</td><td>64.0</td><td>63.0</td><td>63.0</td><td>68.0</td><td>54.0</td></tr></table>

Table 6: Failure-side SIC decomposition across the primary test set. IF = integrity failure (recalled but violated), MF = memory failure (unable to recall). SIC<sup>+</sup> is normalized by all 125 probes per model; IF rate is conditioned on observed failures.
<table><tr><td>Model</td><td>Failures</td><td>IF</td><td>MF</td><td>IF Rate</td><td>SIC+</td></tr><tr><td>Gemini-3.1-Pro</td><td>47</td><td>43</td><td>4</td><td>91.5</td><td>34.4</td></tr><tr><td>Claude-Sonnet-4-6</td><td>12</td><td>10</td><td>2</td><td>83.3</td><td>8.0</td></tr><tr><td>GPT-5.4</td><td>7</td><td>6</td><td>1</td><td>85.7</td><td>4.8</td></tr><tr><td>GPT-5.4-mini</td><td>2</td><td>2</td><td>0</td><td>100.0</td><td>1.6</td></tr><tr><td>DeepSeek-v4-Flash</td><td>0</td><td>0</td><td>0</td><td></td><td>0.0</td></tr><tr><td>DeepSeek-v4-Pro</td><td>5</td><td>4</td><td>1</td><td>80.0</td><td>3.2</td></tr><tr><td>Qwen3.6-Plus</td><td>6</td><td>5</td><td>1</td><td>83.3</td><td>4.0</td></tr><tr><td>Aggregate</td><td>79</td><td>70</td><td>9</td><td>88.6</td><td>8.0</td></tr><tr><td>w/o Gemini</td><td>32</td><td>27</td><td>5</td><td>84.4</td><td>3.6</td></tr></table>

Table 7: Reinstatement intervention outcomes. Full ⊂ Partial (cumulative); No Recovery = 100% − Partial. Rates are computed over the set of scenarios in which collapse was detected.
<table><tr><td>Model</td><td>Collapsed Scen.</td><td>Full Recovery (%)</td><td>Partial Recovery (%) No Recovery (%)</td><td></td></tr><tr><td>Gemini-3.1-Pro</td><td>47</td><td>53.2</td><td>80.9</td><td>19.1</td></tr><tr><td>Claude-Sonnet-4-6</td><td>12</td><td>75.0</td><td>91.7</td><td>8.3</td></tr><tr><td>GPT-5.4</td><td>7</td><td>71.4</td><td>85.7</td><td>14.3</td></tr><tr><td>GPT-5.4-Mini</td><td>2</td><td>100.0</td><td>100.0</td><td>0.0</td></tr><tr><td>DeepSeek-v4-Pro</td><td>5</td><td>80.0</td><td>100.0</td><td>0.0</td></tr><tr><td>Qwen3.6-Plus</td><td>6</td><td>66.7</td><td>83.3</td><td>16.7</td></tr></table>

## 5.3 Rank-order stability

Model rankings are stable across evaluation conditions: Kendall’s $\tau = 0 . 8 1 ( p < 0 . 0 1 )$ between early-turn and late-turn orderings, and τ = 0.90 $( p < 0 . 0 0 5 )$ between symbolic and semantic category orderings (full analysis in Appendix D).

## 6 Analysis

The key qualitative distinction is between state recall and state use. In manual inspection, many failures preserve explicit references to the earlier instruction while violating its operational content. Models acknowledge a required output format and then add an unsolicited preface, or remember a word-count constraint but drift by a large margin. This is consistent with long-context position effects (Liu et al., 2024) but suggests a stronger diagnosis: accessibility is not sufficient for control.

Human validation of judge decisions. Three evaluators independently re-scored 120 randomly sampled outputs (stratified by model, category, and turn depth). Agreement between the automated judge and human majority vote was 93.3%. The 6.7% disagreement cases concentrated in role/persona items with minor stylistic drift (4.2%) and logical-rule items with genuinely ambiguous applicability (2.5%). In both cases, the automated judge was stricter than human raters, consistent with our conservative scoring design.

Distractor naturalness validation. Mean Jaccard similarity between distractor turns and the original constraint statement was 0.040 across all dialogues, confirming minimal lexical cuing. Five evaluators rated a sample of 40 dialogues for conversational naturalness on a 1–5 Likert scale; the mean rating was 4.2 (σ = 0.6), indicating that distractors read as plausible continuations rather than adversarial insertions.

Turn depth is also diagnostic. Gemini-3.1-Pro shows the clearest collapse profile, whereas DeepSeek-v4-Flash remains stable. This variance argues against treating model scale or benchmark-era recency as a proxy for multi-turn reliability. The recall probe often looks healthy exactly when the next response is about to drift—that is the failure mode a deployment team should care about.

Confirmatory symbolic-depth analysis. To test whether the diagnosis depends on the most brittle model, we ran a 216-probe analysis over the other six models, 12 numerical/format/logical scenarios, and depths 12, 30, and 50 (Appendix I). Compliance is 94.4%, 95.8%, and 90.3%, respectively;

depth-50 SIC<sup>+</sup> is 9.7%, and five of six models exhibit at least one recalled-but-violated case. Numerical constraints localize the mechanism most clearly: depth-50 recall remains 100.0% while compliance falls to 79.2%, yielding $\mathrm { S I C ^ { + } } = 2 0 . 8 \%$ All five such failures involve response-internal counting, which points to online execution or selfverification rather than loss of the rule representation.

Additional scope checks. Extended depth (turns 20–50), Chinese/Japanese, SSA, prompt-style sensitivity, dynamic update/revocation, and twoconstraint evaluations test distinct boundaries (Appendices F–I). The cross-lingual and two-constraint results remain compact scope checks; the sixmodel symbolic-depth analysis above is the confirmatory robustness result.

## 7 Discussion

Errors are not uniform across constraint types. Numerical and format constraints are the most brittle because they require exact surface realization surviving the full dialogue history. Role/persona constraints fail differently: the model slips into a generic assistant voice despite remembering the persona. Logical rules are hardest when follow-ups subtly change the original rule’s scope. Factualmemory bindings are the most stable.

Turn-level data shows that average performance hides meaningful variance. Our extended-depth pilot (Appendix F) shows that divergence amplifies for the three sampled models at turns 20–50, but we do not claim that every model has the same decay curve. The cross-lingual pilot (Appendix G) suggests that the recall–compliance dissociation is not English-specific, while remaining too small to support language-universal claims.

Benchmark design philosophy. StateTrack-Bench is designed as a diagnostic instrument rather than a leaderboard-scale corpus. Each scenario isolates a specific failure mode at a specific depth, enabling causal attribution. The targeted audits in Appendix I and the prompt sensitivity analysis in Appendix L test whether the main diagnosis survives alternative scoring decompositions, model subsets, state updates, interacting constraints, and prompt reformulation. Established diagnostic benchmarks routinely use comparable expert-crafted sets (Winograd Schema Challenge: 273 items; BIG-Bench Hard: 23 tasks of 50–250 examples) precisely because diagnostic precision requires careful construction over volume.

SSA as diagnostic decomposition. SSA is a diagnostic intervention rather than a novel algorithm. Prior work gives mixed evidence on free-form self-feedback: Self-Refine reports iterative gains (Madaan et al., 2023), whereas intrinsic self-correction can fail without external feedback (Huang et al., 2024). SSA instead uses a concrete verifiable predicate derived from the constraint specification. Its three-stage decomposition localizes where state integrity breaks: prepending alone recovers 62% of lost compliance (cost: +13% tokens), demonstrating that SIC is primarily a statebinding failure at generation time. The verification stage quantifies the residual enforcement gap. The result applies to constraints that remain active. In a 7-model, 5-scenario dynamic-state audit, stale SSA enforces a superseded rule in 74.3% of cases, while updating the ledger reduces stale enforcement to 0.0% and yields updated or clean behavior in 97.1% (Appendix I). A separate two-constraint audit obtains 76.5% joint success over 34 valid outputs, confirming that interacting constraints are harder than the isolated core setting.

Manual inspection reveals three recurring failure signatures: numerical predicate drift (63% of logical-rule failures), format overflow under fluency pressure, and affective override when emotionally charged user messages trigger persona abandonment (71% of role/persona failures).

Deployment implications. Among observed failures, the high integrity-failure share means RAG alone is insufficient: recovering the instruction text does not guarantee that it governs the next answer. Monitoring should prioritize symbolic constraints, which fail silently. The prepending-only SSA variant (13% token overhead) is a practical safeguard only when the active-state registry is correct; systems with user revocations or updates need an explicit state-update policy before re-anchoring.

## 8 Conclusion

STATETRACK-BENCH shows that persistent-state evaluation is necessary for multi-turn reliability. State Integrity Collapse (SIC), where a recalled constraint fails to govern generation, remains the dominant failure type even after excluding the most brittle model.

## Limitations

Our study has several limitations that should be considered when interpreting the results.

First, the core benchmark comprises 125 scenario-depth probes per model, and the percategory sample size (25 probes) limits statistical power for fine-grained within-category comparisons. We therefore use category results as diagnostic profiles rather than significance-ranked model comparisons. The targeted follow-up audits test specific methodological boundaries, but a community-scale expansion with hundreds of scenarios per category would still enable more precise effect estimation.

Second, while our multi-judge cross-validation (Appendix J) shows high inter-judge agreement (κ = 0.89) across three architecturally distinct models with no significant self-preference bias, the evaluation still relies on LLM-based judges. Future work should incorporate larger-scale human evaluation and investigate whether judge biases correlate with specific constraint types or dialogue structures.

Third, while our cross-lingual pilot (Appendix G) suggests that SIC-like patterns can appear in Chinese and Japanese with preserved model rankings, the 30-scenario pilot is insufficient for definitive cross-lingual claims. Languages with richer morphology (e.g., Finnish, Turkish) or fundamentally different discourse structures (e.g., Arabic) remain untested. A full multilingual benchmark covering diverse typological families is needed.

Fourth, our distractor turns are designed to be topically related but non-adversarial. In deployment, users may actively attempt to override constraints through social engineering or adversarial prompting. Our benchmark measures baseline SIC under cooperative conditions; adversarial robustness requires separate evaluation.

Fifth, while our extended depth pilot (Appendix F) shows that SIC patterns intensify at depths 20–50, the pilot uses a 40-scenario subset. Full-scale evaluation at depths beyond 50 turns— particularly in multi-session settings where context is summarized or truncated—remains future work.

Finally, the benchmark uses one persistent constraint per dialogue. This isolates recall– compliance dissociation cleanly but does not measure multi-constraint interference, conflicting priorities, or user-initiated constraint revocation.

Our proposed mitigation (SSA, Appendix H) is therefore an inference-time intervention for immutable active constraints; without an explicit update/revocation detector it may over-enforce a stale rule. The dynamic audit in Appendix I quantifies this risk but is still a stress test rather than a full dynamic-dialogue benchmark. Developing training-time solutions and dynamic statemanagement protocols that fundamentally improve state-binding in generation remains an open challenge.

## Ethical Considerations

All dialogue scenarios were authored by our research team and do not contain personally identifiable information, offensive content, or material that could cause harm if released. The benchmark does not involve human subjects in the traditional sense; crowdwork annotators were compensated at \$18 USD per hour, above the local minimum wage. We release the benchmark under CC BY 4.0 to facilitate replication while permitting commercial use for safety evaluation purposes. The main ethical risk is misuse of the results as a blanket ranking of model safety; our findings should be interpreted as a targeted audit of persistent constraint adherence rather than a comprehensive safety evaluation.

## Acknowledgments

This work is supported by the Intelligent Computing Center of Shenzhen University.

## References

Chenxin An, Shansan Gong, Ming Zhong, Xingjian Zhao, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. 2024. L-Eval: Instituting standardized evaluation for long context language models. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers).

Paweł Budzianowski, Tsung-Hsien Wen, Bo-Hsiang Tseng, Iñigo Casanueva, Stefan Ultes, Osman Ramadan, and Milica Gašic. 2018.´ MultiWOZ – a large-scale multi-domain wizard-of-oz dataset for task-oriented dialogue modelling. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing.

Guiming Hardy Chen, Shunian Chen, Ziche Liu, Feng Jiang, and Benyou Wang. 2024. Humans or LLMs as the judge? a study on judgement biases. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing.

Jesse Dodge, Gabriel Ilharco, Roy Schwartz, Ali Farhadi, Hannaneh Hajishirzi, and Noah Smith. 2020. Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping. In arXiv preprint arXiv:2002.06305.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. 2024. RULER: What’s the real context size of your long-context language models? In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing.

Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. 2024. Large language models cannot self-correct reasoning yet. In Proceedings of the Twelfth International Conference on Learning Representations.

Yuxin Jiang, Yufei Wang, Xingshan Zeng, Wanjun Zhong, Liangyou Li, Fei Mi, Lifeng Shang, Xin Jiang, Qun Liu, and Wei Wang. 2024. Follow-Bench: A multi-level fine-grained constraints following benchmark for large language models. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics.

Greg Kamradt. 2023. Needle in a haystack – pressure testing LLMs. GitHub repository.

Philipp Koehn. 2004. Statistical significance tests for machine translation evaluation. In Proceedings ofthe 2004 Conference on Empirical Methods in Natural Language Processing.

Mosh Levy, Alon Jacoby, and Yoav Goldberg. 2024. Same task, more tokens: the impact of input length on the reasoning performance of large language models. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics.

Tianle Li, Ge Zhang, Quy Duc Do, Xiang Yue, and Wenhu Chen. 2024. Long-context LLMs struggle with long in-context learning. In arXiv preprint arXiv:2404.02060.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions ofthe Association for Computational Linguistics, 12:157–173.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems.

Arjun Panickssery, Samuel R. Bowman, and Shi Feng. 2024. LLM evaluators recognize and favor their own generations. In arXiv preprint arXiv:2404.13076.

Ethan Perez, Saffron Huang, Francis Song, Trevor Cai, Roman Ring, John Aslanides, Amelia Glaese, Nat McAleese, and Geoffrey Irving. 2022. Red teaming language models with language models. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing.

Yiwei Qin, Kaiqiang Song, Yebowen Hu, Wenlin Yao, Sangwoo Cho, Xiaoyang Wang, Xuansheng Wu, Fei Liu, Pengfei Liu, and Dong Yu. 2024. InFoBench: Evaluating instruction following ability in large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics.

Abhinav Rastogi, Xiaoxue Zang, Srinivas Sunkara, Raghav Gupta, and Pranav Khaitan. 2020. Towards scalable multi-domain conversational agents: The schema-guided dialogue dataset. In Proceedings of the AAAI Conference on Artificial Intelligence.

Shreya Shankar, J.D. Zamfirescu-Pereira, Björn Hartmann, Aditya G. Parameswaran, and Ian Arawjo. 2024. Who validates the validators? aligning LLMassisted evaluation of LLM outputs with human preferences. In Proceedings of the 37th Annual ACM Symposium on User Interface Software and Technology.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-following evaluation for large language models. In arXiv preprint arXiv:2311.07911.

## A Dataset Details

STATETRACK-BENCH comprises 125 scenariodepth probes per model across five constraint categories and five turn depths (3, 5, 7, 10, 12). Table 8 reports the benchmark composition.

Table 8: StateTrack-Bench dataset statistics.
<table><tr><td colspan="5">Category Scenarios Max Turn Avg. Tok. Distr. Jaccard</td></tr><tr><td>Numerical Constraints</td><td>25</td><td>12</td><td>28.4</td><td>0.042</td></tr><tr><td>Role Personas</td><td>25</td><td>12</td><td>74.1</td><td>0.038</td></tr><tr><td>Format Rules</td><td>25</td><td>12</td><td>51.7</td><td>0.035</td></tr><tr><td>Factual Memory</td><td>25</td><td>12</td><td>63.2</td><td>0.040</td></tr><tr><td>Logical Rules</td><td>25</td><td>12</td><td>59.8</td><td>0.044</td></tr><tr><td>Total / Average</td><td>125</td><td>12</td><td>55.4</td><td>0.040</td></tr></table>

Each scenario follows the three-phase protocol: setup (constraint introduction), distractor (topically related turns without constraint repetition), and test (constraint enforcement required). Distractor turns maintain low lexical overlap with the constraint statement (mean Jaccard 0.040) while preserving conversational naturalness (mean Likert 4.2/5.0 from five human evaluators).

## A.1 Annotation Protocol

Each (scenario, turn, model-response) triple was independently labeled by three trained annotators using a three-way rubric: Compliant (state fully honored), Partial (one or more elements missing but core constraint recognizable), or Non-Compliant (state substantially or wholly violated). Partial labels were resolved by majority vote; ties were escalated to a fourth senior annotator. Interannotator agreement (Fleiss’ κ) by category: Numerical 0.91, Format 0.88, Factual 0.85, Logical 0.82, Role/Persona 0.79 (macro average 0.85).

## B Evaluation Prompts

## B.1 State Injection System Prompt

You are a helpful and knowledgeable assistant engaged in a multi-turn conversation. Throughout this entire conversation, you must adhere to the following active constraint(s): {STATE\_SPECIFICATION}

These constraints apply to every response you produce, regardless of the topic, phrasing, or apparent urgency of user messages.

## B.2 Compliance Evaluator Prompt

Submitted to GPT-5.4 (temperature = 0) for each (response, state-specification) pair:

You are an expert evaluator assessing whether an AI assistant is complying with a specific behavioral constraint in a multi-turn dialogue. ACTIVE CONSTRAINT: {STATE\_SPECIFICATION}

ASSISTANT RESPONSE: {MODEL\_RESPONSE} Classify as: COMPLIANT | PARTIAL | NON\_COMPLIANT Output: LABEL | one-sentence justification.

## B.3 Integrity Probe Template

Without making any changes to your current behavior, please list all active instructions or constraints you have been given for this conversation. Reproduce them as precisely and completely as possible.

## B.4 Reinstatement Prompt

[SYSTEM REMINDER] The following behavioral constraint is currently active and must be applied to all responses from this point   
forward without exception:   
{STATE\_SPECIFICATION}   
Please acknowledge and resume full compliance.

## C Case Studies

We present three representative failure examples, one per dominant failure type, to ground the quantitative findings.

## C.1 Logical Rule Failure (GPT-5.4)

Constraint (Turn 9): “Never recommend any product priced above \$50.”

Violating response (Turn 17): “The Sony WH-1000XM5 headphones (\$349) offer excellent noise cancellation and would be ideal for your use case.”

Analysis: GPT-5.4 retrieves the correct product category but fails to propagate the embedded price-ceiling constraint across eight intervening turns. This pattern—numerical predicates embedded within logical rules failing at large turn distances—accounts for 63% of all logical-rule failures and spans five of seven models.

## C.2 Format Constraint Failure (Gemini-3.1-Pro)

Constraint (Turn 8): “All responses must be structured as a numbered list of exactly three items.”

Violating response (Turn 14): “Of course! There are several considerations here: first, the timing matters; second, regional context introduces variability; and third, systemic factors are relevant. It is also worth noting that individual circumstances differ.”

Analysis: The model produces a structurally listlike response but appends a fourth, unnumbered continuation sentence. This exemplifies Gemini’s format-refusal failure mode: the constraint is partially respected but overridden by a fluency-driven generative tendency to append clarifying caveats.

## C.3 Role/Persona Failure (Claude-Sonnet-4-6)

Constraint (Turn 8): “You are a formal Victorianera correspondent; all responses must avoid contractions and contemporary slang.”

Violating response (Turn 16): “That’s a great point! I totally agree that the situation calls for a nuanced approach.”

Analysis: Despite correctly maintaining the Victorian register for turns 8–15, Claude abandons both contractions and formality constraints simultaneously at turn 16, in response to an emotionally charged user prompt. This pattern—affective override of stylistic persona constraints—appears in 71% of sampled role/persona failures across all seven models.

## D Extended Statistical Details

## D.1 Bootstrap Confidence Intervals

All reported confidence intervals use the percentile bootstrap method with 10,000 resamples. For model-level retention scores, we resample over scenarios (n=125) within each model; for categorylevel scores, we resample over scenarios within each category (n=25). The 95% CIs for the main retention scores are:

Table 9: 95% bootstrap confidence intervals for overall retention.
<table><tr><td>Model</td><td>Retention</td><td>95% CI</td></tr><tr><td>DeepSeek-v4-Flash</td><td>100.0</td><td>[100.0, 100.0]</td></tr><tr><td>GPT-5.4-mini</td><td>98.4</td><td>[96.0, 100.0]</td></tr><tr><td>DeepSeek-v4-Pro</td><td>96.0</td><td>[92.8, 98.4]</td></tr><tr><td>Qwen3.6-Plus</td><td>95.2</td><td>[92.0, 97.6]</td></tr><tr><td>GPT-5.4</td><td>94.4</td><td>[91.2, 96.8]</td></tr><tr><td>Claude-Sonnet-4-6</td><td>90.4</td><td>[86.4, 93.6]</td></tr><tr><td>Gemini-3.1-Pro</td><td>62.4</td><td>[56.8, 68.0]</td></tr></table>

## D.2 Paired Bootstrap Significance Tests

We use paired bootstrap tests (10,000 resamples) to assess whether differences between adjacent models in the ranking are statistically significant. The difference between Gemini-3.1-Pro and all other models is significant at $p < 0 . 0 0 1$ . Differences among the top-5 models (DeepSeek-v4-Flash through GPT-5.4) are not individually significant at $p < 0 . 0 5$ , consistent with these models occupying a shared high-retention tier.

Table 10: Per-category retention rates (%) averaged across all turn depths.
<table><tr><td>Model</td><td>Num.</td><td>Format Logic</td><td></td><td>Factual</td><td>Persona</td></tr><tr><td>GPT-5.4</td><td>92.0</td><td>90.4</td><td>88.8</td><td>100.0</td><td>100.0</td></tr><tr><td>GPT-5.4-mini</td><td>88.0</td><td>84.0</td><td>82.4</td><td>96.0</td><td>94.4</td></tr><tr><td>DeepSeek-v4-Flash</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>DeepSeek-v4-Pro</td><td>94.4</td><td>92.8</td><td>91.2</td><td>100.0</td><td>100.0</td></tr><tr><td>Gemini-3.1-Pro</td><td>56.0</td><td>52.8</td><td>48.0</td><td>76.8</td><td>78.4</td></tr><tr><td>Claude-Sonnet-4-6</td><td>90.4</td><td>88.0</td><td>86.4</td><td>96.8</td><td>96.0</td></tr><tr><td>Qwen3.6-Plus</td><td>92.8</td><td>90.4</td><td>88.0</td><td>98.4</td><td>97.6</td></tr><tr><td>Category Mean</td><td>87.7</td><td>85.5</td><td>83.5</td><td>95.4</td><td>95.2</td></tr></table>

## D.3 Per-Category Retention

## D.4 Effect Size Estimates

Cohen’s d for the difference between the highestretention model (DeepSeek-v4-Flash, 100.0%) and the lowest (Gemini-3.1-Pro, 62.4%) is $d = 3 . 2 .$ , indicating a very large effect. The category-level effect size between symbolic constraints (mean 85.6%) and semantic constraints (mean 95.3%) is d = 1.1, a large effect that confirms the practical significance of the symbolic-semantic distinction.

## E Temperature Robustness

Table 11: Retention rates (%) under varying decoding temperatures on a stratified 50-scenario subset.
<table><tr><td>Model</td><td> $\tau = 0 . 0$ </td><td> $\tau = 0 . 3$ </td><td> $\tau = 0 . 7$ </td></tr><tr><td>GPT-5.4</td><td>94.4</td><td>93.8</td><td>92.6</td></tr><tr><td>Gemini-3.1-Pro</td><td>62.4</td><td>61.2</td><td>60.8</td></tr><tr><td>DeepSeek-v4-Pro</td><td>96.0</td><td>95.4</td><td>94.2</td></tr></table>

The maximum shift across all temperature conditions is 2.1 percentage points. Model ordering is preserved under all conditions, confirming that SIC reflects a structural property of how models maintain state rather than a sensitivity to sampling noise.

## F Extended Depth Pilot

We extended evaluation to depths 20, 30, and 50 on a stratified 40-scenario subset (8 per category) for three models exhibiting the widest performance spread.

Table 12: Retention (%) at extended dialogue depths.
<table><tr><td>Model</td><td>Turn 12</td><td>Turn 20</td><td>Turn 30</td><td>Turn 50</td></tr><tr><td>DeepSeek-v4-Flash</td><td>100.0</td><td>100.0</td><td>97.5</td><td>95.0</td></tr><tr><td>GPT-5.4</td><td>96.0</td><td>92.5</td><td>87.5</td><td>82.5</td></tr><tr><td>Gemini-3.1-Pro</td><td>54.0</td><td>47.5</td><td>40.0</td><td>32.5</td></tr></table>

Within this 40-scenario pilot, even the most stable sampled model shows measurable degradation at depth 50 (−5.0 points). GPT-5.4 exhibits roughly linear decay of ∼0.35 points per turn beyond depth 12, while Gemini-3.1-Pro loses an additional 21.5 points from depth 12 to 50. Integrity failures constitute 91.3% of collapses at depth 30– 50 (vs. 88.6% at depth ≤12), suggesting that the recall–compliance dissociation can widen at longer horizons.

## G Cross-Lingual Pilot

We translated 30 representative scenarios (6 per category) into Mandarin Chinese and Japanese, with native-speaker verification of translation fidelity and naturalness.

Table 13: Cross-lingual retention (%) on a 30-scenario subset.
<table><tr><td>Model</td><td>English</td><td></td><td>Chinese Japanese</td></tr><tr><td>GPT-5.4</td><td>93.3</td><td>90.0</td><td>86.7</td></tr><tr><td>Claude-Sonnet-4-6</td><td>90.0</td><td>86.7</td><td>83.3</td></tr><tr><td>Gemini-3.1-Pro</td><td>63.3</td><td>56.7</td><td>53.3</td></tr></table>

The pilot observes SIC-like failures across all three languages with preserved model rankings $( \tau ~ = ~ 0 . 8 7 )$ Retention drops 3–7 points from English to Chinese and an additional 3–4 points to Japanese. Japanese showed distinctive persona degradation (−8.9 points vs. English) due to competing politeness norms. Chinese showed the largest numerical constraint weakness (−5.6 points), potentially from numeral-classifier interactions. Because the subset has only 30 scenarios and 3 models, these are diagnostic signals rather than definitive multilingual estimates.

## H Structured State Anchoring (SSA)

SSA operates in three stages: (1) structured [ACTIVE\_STATE] prepending at each turn; (2) constraint reformulation as verifiable predicates; (3) post-generation verification with up to 2 retries. This stress test assumes that the original constraint remains active throughout the dialogue. It therefore measures recovery under immutable constraints, not false-positive over-enforcement after user revocation or update.

SSA recovers substantial compliance without retraining when the active constraint is still valid. The largest gain is on Gemini-3.1-Pro (+24.4 points). SSA adds mean 1.3 retries per violation on Gemini and 0.2 on GPT-5.4. In dynamic dialogues, however, blind re-anchoring can over-enforce stale constraints; a deployment system must first update the active-state ledger when a user revokes or revises a rule.

Table 14: Retention (%) with SSA vs. baseline at depth 12.
<table><tr><td>Model</td><td></td><td>Baseline +Reinstate +SSA ∆ vs. Base</td><td></td><td></td></tr><tr><td>GPT-5.4</td><td>96.0</td><td>97.6</td><td>99.2</td><td>+3.2</td></tr><tr><td>Claude-Sonnet-4-6</td><td>88.0</td><td>92.0</td><td>96.0</td><td>+8.0</td></tr><tr><td>Gemini-3.1-Pro</td><td>54.0</td><td>65.6</td><td>78.4</td><td>+24.4</td></tr><tr><td>Qwen3.6-Plus</td><td>96.0</td><td>97.6</td><td>98.4</td><td>+2.4</td></tr></table>

## I Additional Metric and Dynamic-State Audits

To stress-test five methodological concerns, we ask whether strict joint retention conflates immediatetask failure with persistent-state failure, whether signed SIC components can be reported more directly by category, whether the symbolic-depth signal persists when the brittle Gemini outlier is excluded, whether SSA over-enforces stale constraints after revocation or update, and how the diagnosis changes when two persistent constraints are active simultaneously. We therefore ran compact follow-up audits using the same logged model snapshots, with one audit intentionally excluding Gemini, and saved all raw outputs, judge decisions, and summaries with the released experiment logs.

## I.1 Strict Joint versus Constraint-Only Scoring

Table 15 reports a 70-output audit (10 probes per model) that separately scores immediate-task success, constraint-only compliance, strict joint success, recall, SIC<sup>+</sup>, and SIC<sup>−</sup>. The audit verifies the interpretation used in the main text: strict joint retention is a conservative deployment metric, while constraint-only compliance isolates the persistentrule component. For example, several models preserve the constraint more often than they satisfy the full joint task, so treating strict joint retention as pure memory would be overly pessimistic.

## I.2 Signed SIC Components by Category

Table 16 reports the signed SIC components by category on the same compact forked audit. Because this audit contains only 14 probes per category, we use it as a diagnostic check rather than a replacement for the primary failure-side decomposition in

Table 15: Compact strict-joint versus constraint-only audit on controlled multi-turn histories (10 probes/model). Immediate-task success is judged while ignoring the persistent constraint; constraint-only uses the scenario checker; strict joint requires both.
<table><tr><td>Model</td><td>n</td><td>Task</td><td>Constraint</td><td>Strict joint</td><td>Recall</td><td> $\mathbf { S I C ^ { + } }$ </td><td> $\mathbf { S I C } ^ { - }$ </td></tr><tr><td>GPT-5.4</td><td>10</td><td>80.0</td><td>100.0</td><td>80.0</td><td>100.0</td><td>0.0</td><td>0.0</td></tr><tr><td> $\mathsf { G P T } \mathsf { - } 5 . 4 \mathsf { - m i n i }$ </td><td>10</td><td>80.0</td><td>90.0</td><td>70.0</td><td>100.0</td><td>10.0</td><td>0.0</td></tr><tr><td>Claude-Sonnet-4-6</td><td>10</td><td>80.0</td><td>60.0</td><td>40.0</td><td>10.0</td><td>10.0</td><td>60.0</td></tr><tr><td>Gemini-3.1-Pro</td><td>10</td><td>40.0</td><td>30.0</td><td>20.0</td><td>0.0</td><td>0.0</td><td>30.0</td></tr><tr><td>DeepSeek-v4-Pro</td><td>10</td><td>60.0</td><td>90.0</td><td>50.0</td><td>90.0</td><td>10.0</td><td>10.0</td></tr><tr><td>DeepSeek-v4-Flash</td><td>10</td><td>50.0</td><td>100.0</td><td>50.0</td><td>100.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Qwen3.6-Plus</td><td>10</td><td>90.0</td><td>100.0</td><td>90.0</td><td>100.0</td><td>0.0</td><td>0.0</td></tr></table>

Table 16: SIC components by constraint category in the compact forked audit. Values are percentages over 14 probes per category (2 scenarios $\times 7$ models).
<table><tr><td>Category</td><td>n</td><td>Constraint</td><td>Recall</td><td> $\mathbf { S I C ^ { + } }$ </td><td>SIC⁻</td></tr><tr><td>Numerical</td><td>14</td><td>71.4</td><td>71.4</td><td>7.1</td><td>7.1</td></tr><tr><td>Format</td><td>14</td><td>78.6</td><td>78.6</td><td>7.1</td><td>7.1</td></tr><tr><td>Persona</td><td>14</td><td>71.4</td><td>71.4</td><td>7.1</td><td>7.1</td></tr><tr><td>Factual</td><td>14</td><td>92.9</td><td>64.3</td><td>0.0</td><td>28.6</td></tr><tr><td>Logical</td><td>14</td><td>92.9</td><td>71.4</td><td>0.0</td><td>21.4</td></tr></table>

Table 6. It nevertheless makes the signed quantities explicit: $\mathrm { S I C ^ { + } }$ captures recalled-but-violated behavior, whereas SIC<sup>−</sup> captures behaviorally compliant outputs whose recall probe fails.

## I.3 Gemini-Excluded Symbolic-Depth Check

To further isolate the model-dependency concern, we ran a new diagnostic audit that excludes Gemini-3.1-Pro entirely and focuses on symbolic constraints at longer depths. The audit covers six non-Gemini models, twelve numerical/format/logical scenarios, three depths (12, 30, and 50), and forked behavior/recall branches, yielding 216 paired probes (432 real model calls). Table 17 shows that aggregate constraint compliance remains high but not perfect: 94.4% at depth 12, 95.8% at depth 30, and 90.3% at depth 50. The depth- ${ \cdot } 5 0 \mathrm { \ S I C ^ { + } }$ rate is 9.7%, and five of the six non-Gemini models exhibit at least one recalledbut-violated case. A descriptive mechanism audit further localizes the gap: numerical constraints retain 100.0% recall but only 79.2% compliance at depth 50, yielding $\mathrm { S I C ^ { + } } = 2 0 . 8 \% ;$ all five numerical $\mathrm { S I C ^ { + } }$ events occur in response-internal counting cases. This pattern is consistent with an online execution or self-verification bottleneck rather than simple loss of the rule representation. It supports a calibrated conclusion: the strongest failures are model-dependent, but the symbolic recall– compliance dissociation is not exclusively a Gemini artifact.

Table 17: Gemini-excluded symbolic-depth diagnostic audit. C@d is deterministic constraint compliance at depth d; ∆ is C@50–C@12.
<table><tr><td>Model</td><td>C@12</td><td>C@30</td><td>C@50</td><td>∆ SIC+@50</td><td></td></tr><tr><td>GPT-5.4</td><td>91.7</td><td>100.0</td><td>91.7</td><td>0.0</td><td>8.3</td></tr><tr><td>GPT-5.4-mini</td><td>91.7</td><td>83.3</td><td>91.7</td><td>0.0</td><td>8.3</td></tr><tr><td>Claude-Sonnet-4-6</td><td>91.7</td><td>100.0</td><td>91.7</td><td>0.0</td><td>8.3</td></tr><tr><td>DeepSeek-v4-Pro</td><td>100.0</td><td>91.7</td><td>83.3</td><td>-16.7</td><td>16.7</td></tr><tr><td>DeepSeek-v4-Flash</td><td>91.7</td><td>100.0</td><td>83.3</td><td>-8.4</td><td>16.7</td></tr><tr><td>Qwen3.6-Plus</td><td>100.0</td><td>100.0</td><td>100.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Aggregate</td><td>94.4</td><td>95.8</td><td>90.3</td><td>-4.1</td><td>9.7</td></tr></table>

## I.4 Dynamic Update and Revocation Stress Test

Table 18 reports a 7-model × 5-scenario × 3- condition stress test for dynamic-state handling. The base condition gives the model the dialogue history after the user explicitly revokes or updates a constraint. The stale-SSA condition intentionally injects the old active-state ledger, and the updated-ledger condition injects the revised ledger. The base dialogue rarely enforces the stale rule (2.9%), but stale SSA raises stale enforcement to 74.3%. Updating the active-state ledger before re-anchoring reduces stale enforcement to 0.0% and yields updated/clean behavior in 97.1% of cases. This is why our SSA claim is scoped to immutable active constraints and why deployment systems need explicit state-update tracking before re-anchoring.

Table 18: Dynamic revoke/update stress test for SSA. Stale enforced counts outputs that incorrectly follow the old constraint after it was revoked or updated. Base uses only the dialogue history; stale SSA intentionally injects the old active state; updated-ledger SSA injects the revised active state.
<table><tr><td>Condition</td><td>n</td><td>Stale enforced</td><td>Stale %</td><td>Updated/clean %</td></tr><tr><td>Base after change</td><td>35</td><td>1</td><td>2.9</td><td>94.3</td></tr><tr><td>Stale SSA ledger</td><td>35</td><td>26</td><td>74.3</td><td>20.0</td></tr><tr><td>Updated-ledger SSA</td><td>35</td><td>0</td><td>0.0</td><td>97.1</td></tr></table>

Table 19: Compact two-constraint diagnostic audit. C1/C2 are the two deterministic constraint checks; Joint requires both constraints; Recall is the forked recall probe.
<table><tr><td>Model</td><td>n</td><td>C1</td><td>C2</td><td></td><td>Joint Recall</td><td>SIC+</td></tr><tr><td>GPT-5.4</td><td></td><td>4 100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>0.0</td></tr><tr><td>GPT-5.4-mini</td><td>5</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>0.0</td></tr><tr><td>Claude-Sonnet-4-6</td><td>5</td><td>80.0</td><td>100.0</td><td>80.0</td><td>60.0</td><td>0.0</td></tr><tr><td>Gemini-3.1-Pro</td><td>5</td><td>20.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>DeepSeek-v4-Pro</td><td>5</td><td>100.0</td><td>80.0</td><td>80.0</td><td>60.0</td><td>20.0</td></tr><tr><td>DeepSeek-v4-Flash</td><td>5</td><td>80.0</td><td>100.0</td><td>80.0</td><td>100.0</td><td>20.0</td></tr><tr><td>Qwen3.6-Plus</td><td>5</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>0.0</td></tr><tr><td>Aggregate</td><td>34</td><td>82.4</td><td>82.4</td><td>76.5</td><td>73.5</td><td>5.9</td></tr></table>

## I.5 Two-Constraint Scope Check

Finally, Table 19 reports a low-cost diagnostic micro-audit with two simultaneous persistent constraints per dialogue. This audit directly probes the ecological-validity boundary of the singleconstraint benchmark design. Across 34 valid outputs (one GPT-5.4 probe encountered a repeated upstream 502 and is excluded), the joint twoconstraint success rate is 76.5%, lower than the main single-constraint benchmark, while forked recall is 73.5%. This confirms that multi-constraint interference is a harder setting and should be treated as future full-benchmark work, while preserving the paper’s main design rationale: the core benchmark intentionally isolates state integrity before adding interacting constraints.

## I.6 Illustrative Cases

The following compact examples show how the audit labels are assigned. In a $\mathrm { S I C ^ { + } }$ case, the setup required every response to end with an exact word count in brackets. GPT-5.4-mini answered the weather probe with a reasonable caveat but omitted the bracketed count, while its separate recall branch correctly restated the rule. In a $\mathrm { S I C ^ { - } }$ case under the same setup, Claude-Sonnet-4-6 produced a bracketed-count answer, but its recall branch refused to state the active constraint. For dynamic

SSA, after the user updated the suffix rule from END to DONE, stale SSA produced an answer ending in END, while updated-ledger SSA produced the same answer ending in DONE. These cases are illustrative; the quantitative labels come from the full compact audit above.

## J Multi-Judge Cross-Validation

To address concerns about single-judge bias, we re-scored a stratified 200-output subset using three architecturally distinct judges: GPT-5.4 (primary), Claude-Sonnet-4-6, and Gemini-3.1-Pro, each with the same fixed evaluation prompt at temperature 0.

Table 20: Multi-judge agreement and self-preference analysis on 200-output subset.
<table><tr><td>Judge</td><td>Agreement w/ Primary κ (pairwise)</td><td></td><td>Self-Pref. ∆ McNemar p Bias?</td><td></td><td></td></tr><tr><td>GPT-5.4 (primary)</td><td></td><td></td><td>+1.2%</td><td></td><td></td></tr><tr><td>Claude-Sonnet-4-6</td><td>94.0%</td><td>0.87</td><td>+0.8%</td><td>0.42</td><td>No</td></tr><tr><td>Gemini-3.1-Pro</td><td>91.5%</td><td>0.82</td><td>-0.4%</td><td>0.68</td><td>No</td></tr><tr><td>Fleiss&#x27; κ (3 judges)</td><td colspan="5">0.89</td></tr></table>

Self-preference $\Delta$ measures whether a judge scores outputs from its own model family more leniently than outputs from other models. The maximum observed ∆ is +1.2% (GPT-5.4 judging GPT-5.4 outputs), which is not statistically significant (McNemar’s $p = 0 . 3 1 \mathrm { ) }$ . Model rankings are identical across all three judges. The 6% disagreement cases concentrate in role/persona items with subtle stylistic drift, where Claude-Sonnet-4- 6 is marginally more lenient and Gemini-3.1-Pro marginally stricter—neither pattern is systematic enough to alter any model’s tier placement.

## K SSA Inference Cost Analysis

We quantify the latency and token overhead of Structured State Anchoring to enable informed deployment decisions.

Table 21: SSA inference overhead at depth 12. Latency is wall-clock time per response; token cost includes input+output tokens for all retries.
<table><tr><td>Model</td><td>Base Lat.</td><td>SSA Lat.</td><td>Lat. ×</td><td>Base Tok.</td><td>SSA Tok.</td><td>Tok. ×</td></tr><tr><td>GPT-5.4</td><td>2.1s</td><td>2.8s</td><td>1.34×</td><td>1,842</td><td>2,174</td><td>1.18×</td></tr><tr><td>Claude-Sonnet-4-6</td><td>2.4s</td><td>3.6s</td><td>1.50×</td><td>1,956</td><td>2,487</td><td>1.27×</td></tr><tr><td>Gemini-3.1-Pro</td><td>1.8s</td><td>3.8s</td><td>2.12×</td><td>1,724</td><td>2,534</td><td>1.47×</td></tr><tr><td>Qwen3.6-Plus</td><td>1.9s</td><td>2.7s</td><td>1.42×</td><td>1,801</td><td>2,198</td><td>1.22×</td></tr></table>

The overhead is model-dependent and correlates with baseline SIC severity: models that fail more often trigger more retries. GPT-5.4 averages 0.2 retries per response (mean 1 retry every 5 turns), while Gemini-3.1-Pro averages 1.3 retries. The structured prepending component alone (no verification/retry) adds only 12–15% token overhead and recovers 60–70% of lost compliance, making it suitable for latency-sensitive deployments. The full SSA pipeline (prepending + verification + retry) is recommended for high-stakes applications where correctness outweighs latency.

Component decomposition. SSA’s three stages contribute differently to recovery: (1) structured prepending alone recovers 62% of lost compliance on average (cost: +13% tokens, +8% latency); (2) adding constraint reformulation as verifiable predicates recovers an additional 14% (cost: +4% tokens, negligible latency); (3) post-generation verification with retry recovers the final 24% (cost: variable, 0–47% additional tokens depending on retry frequency). This decomposition demonstrates that SSA is not merely “retry until correct” but a principled diagnostic tool: the prepending stage isolates state-binding failures, while the verification stage isolates generation-time constraint enforcement failures.

## L Prompt Sensitivity Analysis

To verify that benchmark conclusions are robust to prompt formulation, we created three alternative phrasings for each of 50 randomly sampled constraints (maintaining semantic equivalence while varying surface form, specificity level, and imperative vs. declarative framing).

Table 22: Retention (%) under alternative prompt formulations (50-scenario subset).
<table><tr><td>Model</td><td></td><td></td><td>Original Variant A Variant B</td><td>Variant C</td></tr><tr><td>GPT-5.4</td><td>94.0</td><td>93.2</td><td>94.8</td><td>92.4</td></tr><tr><td>Claude-Sonnet-4-6</td><td>90.0</td><td>88.8</td><td>91.2</td><td>89.6</td></tr><tr><td>Gemini-3.1-Pro</td><td>62.0</td><td>60.4</td><td>63.6</td><td>59.2</td></tr></table>

Maximum deviation across prompt variants is 3.6 points (Gemini, Variant C), and model rankings are preserved across all formulations (τ = 1.0). This confirms that SIC findings reflect model behavior rather than prompt-specific artifacts.

StateTrack-Bench scenarios and derived benchmark metadata are released under CC BY 4.0, and the evaluation code is released under the MIT License. The complete release package is available at: https://github.com/xzzvsxd/StateTrack. It includes the scenario definitions, exact model prompts, judge prompts, evaluation scripts, raw model-output logs, and perexperiment summary files used to produce the tables. From the release root, python tools/reproduce\_reported\_metrics.py reconstructs the primary seven-model retention table and the dynamic-state, symbolic-depth, execution-bottleneck, and two-constraint results from cached JSON files; python -m unittest discover -s tests -v runs the credential-free evidence checks. For closed API systems, the logs store the endpoint/snapshot identifiers from Table 1 so that future re-executions can distinguish provider drift from benchmark variation. All user-turn content was authored originally by our team; no material was derived from proprietary or copyrighted external corpora. Model outputs included in case studies are used under fair-use provisions for non-commercial academic research. Crowdwork annotators were compensated at \$18 USD per hour. All scenario content was reviewed independently for demographic bias and cultural stereotyping before inclusion.