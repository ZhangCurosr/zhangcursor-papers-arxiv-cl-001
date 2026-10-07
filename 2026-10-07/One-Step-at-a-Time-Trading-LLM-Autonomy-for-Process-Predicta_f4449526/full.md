# One Step at a Time: Trading LLM Autonomy for Process Predictability

Hans Schabert Amazon Web Services schabert@amazon.com

Christoph Peters University of the Bundeswehr Munich christoph.peters@unibw.de

## Abstract

Organizations automating operational processes need more than a correct outcome: they need to predict how a process will run, know which one actually ran, and inspect it step by step. When an agent is the executor that predictability is normally lost—the prescribed procedure goes into the system prompt, and only a final answer comes back. We deliver the procedure step by step over the Model Context Protocol (MCP) instead: a server releases one step at a time, the agent executes it, and each step returns a structured step\_output. This trades autonomy for predictability—the agent no longer chooses its own path—and two properties then follow by construction, independent of which executor runs the process. The execution path is prescribed before the run, so the process is predictable in advance rather than reconstructed afterwards; and the completed step records form a machine-readable execution log that downstream tooling can audit and optimize step by step.

Evaluating 15,475 trials across 13 SOP-Bench domains and four open-weight executors spanning frontier (Kimi K2.5), large (DeepSeek V3.2, GLM 5), and lightweight (Ministral 3 8B) capability, we find steplevel delivery makes the executed process predictable and inspectable for every executor, and additionally raises accuracy when the executor is small. Across all four, process adherence rises significantly (76–95% to 95– 99%) and ungrounded answers—correct outputs produced without executing the SOP— near-vanish, falling from 2.1–4.5% to 0.2– 0.3% of trials (per-model reductions of −1.9 to −4.2pp; all 95% CIs exclude zero); under prompt-based delivery, 31–49% of correct answers on know\_your\_business bypass the SOP entirely, even for the frontier executor. Accuracy is where the executor’s capability enters: the lightweight executor gains +6.5pp grounded accuracy because supplying the process externally removes a reconstruction burden it cannot carry, while capable ones trade a small rawaccuracy decrement for a predictable, auditable process.

## 1 Introduction

On an industrial KYC-style benchmark domain, 31–49% of an LLM agent’s correct answers— including 48% of a frontier model’s (Table 8)— are produced without executing a single one of the prescribed verification tools. Standard accuracy metrics count every one of these as a success. For organizations automating the Standard Operating Procedures (SOPs) that govern operational work—classifying dangerous goods, triaging customer complaints, detecting referral fraud (de Treville et al., 2005; Dumas et al., 2018)—this is the deployment barrier in miniature: the answer can be right while the process that was supposed to produce it never ran, and nothing in the final answer reveals the difference. Deployments stall in regulated environments for exactly this reason: organizations cannot predict how an agent will behave, cannot see which steps it actually executed, and cannot attribute failures to actionable root causes (Kaur et al., 2023); delegation research identifies trust, control, and explainability among the key factors governing whether humans delegate to agentic systems (Strunk et al., 2024).

The prevailing approach embeds the full SOP in the system prompt and permits the model to execute autonomously (Yao et al., 2023; Chase, 2022). This introduces a fundamental problem: the model must reinvent the process from text at every invocation—parse the procedure, determine which step comes next, select the correct tool and parameters, interpret the result, and decide how to proceed.

On SOP-Bench (Nandi et al., 2025)—an industrial benchmark of 2,000+ tasks across 12 business domains—this reinvention produces two failure modes that undermine trust. First, behavior is unpredictable: a lightweight open-weight executor produces on average 20 distinct tool-call sequences per domain (up to 44 on a single procedure; Tables 6 and 7, which also report the sizerobust dominant-sequence share), so no supervisor can anticipate which path the next execution will follow; reasoning is opaque, and failures are undiagnosable short of forensic analysis of the full conversation. Second, and more insidiously, the model can hallucinate the process itself: it produces the right answer while behaving as if it executed a procedure it never ran—the know\_your\_business failure that opens this paper, where up to half of a frontier model’s correct answers bypass the prescribed verification tools entirely.

Approach. We propose externalizing the process structure. Rather than embedding the SOP in the system prompt, we deliver it step-by-step via the Model Context Protocol (MCP) (Anthropic, 2025). The server controls pacing: it releases one step instruction at a time, the model executes that step, and the server advances only after the model reports completion. The model’s autonomy is scoped to the current step.

This constitutes a deliberate trade of autonomy for predictability. The model forfeits the ability to cross-reference future steps or self-correct against the full procedure, and we measure the price of that forfeit: capable executors’ raw accuracy declines slightly, and branching domains that require lookahead lose grounded accuracy outright. In return the system gains three properties addressing the trust prerequisites: predictable behavior (the step decomposition fixes the execution path before the run, so what the agent will do is known in advance, and invented paths collapse toward one dominant sequence for capable executors), verifiable reasoning (each step declares its inputs and outputs as a structured record, a machine-readable audit trail no prompt-based approach provides), and diagnosable failures (that record localizes divergence to a single step, enabling step-level rather than wholesale SOP revision).

Contributions. We evaluate this approach on 15,475 trials across 13 SOP-Bench domains, four models spanning a capability range (a frontier openweight executor, Kimi K2.5; two large open-weight executors, DeepSeek V3.2 and GLM 5; and a lightweight open-weight executor, Ministral 3 8B), and two conditions (RFC 2119 SOP in system prompt vs. step-level MCP delivery); all figures below are detailed in §4. Our contributions are: (1) We introduce Grounded TSR—accuracy conditioned on process adherence—and show that 2.1– 4.5% of trials under prompt delivery (Table 7; up to 31–49% of correct answers on individual domains) are correct without executing the SOP— standard accuracy metrics systematically overstate process compliance; step-level delivery drives this ungrounded rate to 0.2–0.3% for every model (§4.2; all 95% CIs exclude zero; partly structural by design). (2) We decompose behavioral predictability into process adherence and path consistency and show step-level delivery improves adherence for every model and consolidates execution toward a dominant path for capable executors. (3) We find the accuracy effect—unlike the auditability gains in (1) and (2), which hold across the lineup—is capability-dependent: the lightweight open-weight executor gains +6.5pp grounded accuracy (Table 4; its adherence rising from 76% to 95%, Table 3) as externalized process control substitutes for capability, while frontier and large open-weight executors trade a small raw-accuracy decrement (−2.5 to −4.7pp) for full process visibility. (4) Because every step declares its inputs and outputs, the perstep record is an optimization substrate as well as an audit trail: it drives a targeted revision cycle that repaired an SOP defect in under a minute, and it surfaced two benchmark-environment defects invisible to prompt-based evaluation.

## 2 Background and Related Work

## 2.1 SOP-Bench

SOP-Bench (Nandi et al., 2025) provides 2,000+ tasks from human expert-authored SOPs across 12 business domains (healthcare, logistics, finance, content moderation, trust & safety, media), each with executable tool interfaces, precomputed mock API responses, and human-validated ground truth; the public release we evaluate contains 14 domains (2,163 tasks). The benchmark reports Execution Completion Rate (ECR), Task Success Rate (TSR), and Conditional TSR (C-TSR).

Their Function-Calling and ReAct baselines show that newer models do not guarantee better performance (Claude 4 Opus: 72.4% vs. Claude 4.5 Sonnet: 63.3% task success rate on ReAct), and that no model–agent combination dominates (best performances range 57–100% by domain).

These findings motivate our central hypothesis:

the bottleneck is not procedural complexity but instructional specificity—and the delivery mechanism determines how faithfully a model adheres to even well-specified instructions.

## 2.2 RFC 2119 Formalization

RFC 2119 (Bradner, 1997) defines keywords (MUST, MUST NOT, SHALL, SHOULD, MAY) for indicating requirement levels in technical specifications. Originally designed for internet protocol standards, these keywords establish a precise vocabulary for distinguishing absolute requirements from recommendations.

Prior work compiles ambiguous SOPs into plans and code templates (Giroh et al., 2025); we keep the SOP as prose and formalize only its normative force with RFC 2119 keywords. The hypothesis is that explicit normative keywords reduce ambiguity (“You MUST call calculate\_sds\_label\_score. . . ” vs. “calculate the safety data sheet score”). Our bridge study (Table 2) confirms this: RFC 2119 formalization lifts DeepSeek V3.2’s raw TSR from 61% to 83% (+21.8pp) on 12 domains, though not all of this lift is grounded in actual process execution (+17.8pp grounded; Table 2).

However, keyword insertion alone is insufficient: keyword-only conversion produced negligible improvement in early experiments. The effective conversion adds (1) exact tool-to-step mappings with parameter names, (2) decision thresholds, and (3) explicit edge-case handling—consistent with AgentIF (Qi et al., 2025), where even the best model follows fewer than 30% of agentic instructions perfectly, and a substantial share of conditional-constraint failures stem from incorrect condition checking rather than action execution.

## 2.3 SOP-MCP: Step-Level Delivery via MCP

MCP (Anthropic, 2025) provides a standardized interface for connecting LLMs to external tools and data sources. We use its tool-based delivery mechanism to implement step-by-step SOP execution: a run\_sop tool releases one instruction at a time, the model executes the step, and the server advances only after the model reports a step\_output. The server controls pacing—the model cannot see ahead.

With the full SOP in the system prompt, the model has SOP-level autonomy: it can crossreference steps, reorder operations, and self-correct. With step-level delivery, autonomy is scoped to the current step: the model executes each instruction but cannot compensate for errors in the SOP itself. This trade-off—less autonomy for more predictability—is the central design choice we evaluate.

## 2.4 Trust and Delegation

Delegation requires predictable behavior and verifiable compliance. Prompt-based SOP execution fails this test: with dozens of distinct execution paths for the same procedure (up to 44 in our data; Table 6), no supervisor can predict the next execution and no auditor can verify compliance. MCP delivery addresses this directly: the server enforces a defined path, and step-level documentation provides the evidence trail auditors require.

This connects to a broader literature on autonomy–control trade-offs in human organizations (de Treville et al., 2005): strict checklists improve consistency but reduce workers’ ability to catch errors in the checklist itself. The same tension applies to LLM agents; we quantify both sides in Section 4.

## 2.5 Decomposition: Advisory vs. Enforced

Our approach—externalizing process control so the model executes one step at a time—relates to a family of decomposition strategies in which planning and execution both remain inside the model’s own forward pass, whether by planning then solving, routing sub-problems to handlers, building up from simpler sub-questions, or using a controller LLM to sequence tools (Wang et al., 2023; Khot et al., 2023; Zhou et al., 2023; Shen et al., 2023). Closest to our setting, FlowBench (Xiao et al., 2024) benchmarks workflow-guided planning with workflow knowledge supplied in several formats, and ToolPlanner (Wu et al., 2024) trains for path planning with process feedback; step-level process signals have also been used to train agents (Xiong et al., 2024), whereas we use them to revise the procedure.

All of these share the intuition that decomposition improves reliability, and in all of them the model retains control over sequencing—the decomposition is advisory. In our MCP-based approach the server controls sequencing—the decomposition is enforced—and it is this enforced variant whose consistency and auditability we measure in Section 4.

## 3 Methodology

## 3.1 Experimental Design

We compare two SOP delivery conditions across four models and 13 industrial domains from SOP-Bench.

## Conditions.

1. rfc2119\_sop: The SOP converted to RFC 2119 normative format with explicit tool mappings, decision thresholds, and MUST/SHOULD/MAY keywords, delivered as a system prompt. The model has full SOPlevel autonomy.

2. sop\_mcp: The same RFC 2119 SOP decomposed into discrete steps and delivered one at a time via the run\_sop tool. The server controls pacing—the model executes each step, reports a step\_output, and receives the next instruction. The model’s autonomy is scoped to the current step.

Both conditions use identical SOP content. The only difference is the delivery mechanism: bulk system prompt vs. step-by-step MCP.

Models. Four open-weight models span a capability range to test capability dependence (Table 1), all accessed via Amazon Bedrock’s Converse API with a single code path (July–August 2026). List pricing and full per-model economics appear in Appendix I.

<table><tr><td>Executor</td><td>Version</td><td>Tier</td><td>max_tok</td></tr><tr><td>Ministral</td><td>3 8B</td><td>light</td><td>4,096</td></tr><tr><td>DeepSeek</td><td>V3.2</td><td>large</td><td>8,192</td></tr><tr><td>GLM</td><td>5</td><td>large</td><td>8,192</td></tr><tr><td>Kimi</td><td>K2.5</td><td>frontier</td><td>8,192</td></tr></table>

Table 1: Executors: family name (used throughout), version, capability tier, and output-token cap. Bedrock model IDs: mistral.ministral-3-8b-instruct, deepseek.v3.2, zai.glm-5,  
moonshotai.kimi-k2.5.

All models are invoked with provider-default sampling (we set neither temperature nor top\_p; notably not temperature=0) and the max\_tokens caps of Table 1; observed generation sits well below both caps (Table 12). Each task is run once per model×condition (one trial per task; the MCP condition uses a multi-turn step loop within one session), so bootstrap resamples draw over distinct tasks and involve no within-task clustering.

Domains. SOP-Bench provides 14 domains. We exclude one (warehouse\_package\_inspection): a pre-existing bug in SOP-Bench’s source code causes its validateBarcode tool to compute barcode match via PO-number parity rather than ground-truth values, making faithful SOP execution systematically incorrect in 41% of cases (Section 5.3). Of the remaining 13, all are run; video\_annotation is excluded from adherence metrics (wide branching defeats the fixed threshold; Limitations). The 13 evaluated domains span content moderation, healthcare, finance, retail, media, logistics, and trust & safety, totaling 15,475 trials; 12 form the primary adherence pool.

## 3.2 Bridge Study: Connecting SOP-Bench to Our Evaluation

Our study uses RFC 2119 SOPs and a different agent framework (Strands + Bedrock Converse) than SOP-Bench’s evaluation (LangChain + invoke\_model, Claude 3.5 v2). To validate comparability, we ran DeepSeek V3.2—an open-weight model whose DeepSeek-R1 sibling SOP-Bench reports—across all 14 domains and four conditions (plain SOP, plain + tool appendix, RFC 2119, SOP-MCP; 8,618 trials).

Table 2 shows both raw TSR and Grounded TSR—accuracy restricted to trials where the model actually followed the SOP (pa\_coverage ≥ 0.5, 12 domains excluding video\_annotation). The gap between raw and grounded reveals how much accuracy comes from process shortcuts.

<table><tr><td>Condition</td><td>TSR</td><td>Grounded</td><td>∆</td><td>Ungr%</td></tr><tr><td>plain_sop</td><td>60.8%</td><td>60.6%</td><td>+0.2pp</td><td>0.3%</td></tr><tr><td>plain_enriched</td><td>61.5%</td><td>61.2%</td><td>+0.2pp</td><td>0.4%</td></tr><tr><td>rfc2119_sop</td><td>82.6%</td><td>78.4%</td><td>+4.2pp</td><td>5.1%</td></tr><tr><td>sop_mcp</td><td>78.0%</td><td>77.8%</td><td>+0.1pp</td><td>0.2%</td></tr></table>

Table 2: Bridge study (DeepSeek V3.2, 12 domains, domain-mean). Grounded TSR counts only correct answers from adherent trials. ∆ = accuracy from shortcuts. Ungr% = fraction of correct answers that are ungrounded.

Three findings emerge: (1) our plain\_sop baseline (61%) falls inside SOP-Bench’s reported perdomain range for its own agent baselines (57– 100%), indicating our harness reproduces comparable behaviour rather than introducing an artifact; (2) RFC 2119 formalization lifts raw TSR +21.8pp but only +17.8pp grounded—the formalized SOP lets the model infer answers without calling tools:

ungrounded answers rise from 0.3% to 5.1% of correct answers; (3) appending tool information to informal SOPs gives negligible lift (+0.7pp raw). Formalization helps, but raw TSR overstates it.

## 3.3 SOP-MCP Protocol

The run\_sop tool delivers one step at a time: each call returns the step instruction, and the model must fully execute it before advancing. The system prompt names the SOP, prohibits improvisation, and requires calling run\_sop first; this suffices across all four models.

## 3.4 SOP Quality Assurance

Our RFC 2119 SOPs required iterative correction—an automated linter and steptrace inspection identified tool-name mismatches, schema/implementation conflicts, and incorrect branching logic across four domains during pilot runs. All corrections occurred before the primary evaluation, and every reported trial for a revised domain comes from a post-correction re-run of its full task set; pilot tasks were not a disjoint development split (Limitations). The debugging signal came exclusively from MCP step traces (no equivalent pass for the prompt condition); this asymmetry is acknowledged as a limitation but also demonstrates step-level traceability’s diagnostic value (Section 5.3).

## 3.5 Evaluation

We evaluate on three dimensions, all computed from OpenTelemetry tool-call traces.

Grounded Task Success Rate. For each task, the agent’s output is parsed from structured tags (<final\_decision>, JSON) and compared against ground truth using fuzzy matching (caseinsensitive, prefix-stripping, containment checks). We report Grounded TSR: correct answers restricted to trials where the model adhered to the SOP (pa\_coverage ≥ 0.5), divided by total trials; raw TSR follows by identity (raw = grounded + ungrounded). Effect signs are stable for thresholds 0.3–0.7 (Table 9). Conditioning success on evidence of actual execution is under active development in concurrent work (Flynt, 2026; Soni, 2026; Huang et al., 2026); related benchmarks instead verify final environment state (Trivedi et al., 2024) or reliability across repeated trials (Yao et al., 2025). We operationalize it for SOP adherence and quantify the raw-versus-grounded gap across a capability range. Appendix F gives the full pa\_coverage definition.

Process Adherence. The fraction of trials where the model called at least half the expected tools (pa\_coverage ≥ 0.5). This pragmatic threshold avoids marking correct runs on branching SOPs as non-adherent. The expected set is defined per task: for linear SOPs it is the SOP’s prescribed domain tools; for branching SOPs whose multi-field output identifies the branch taken (email\_intent) it is that branch’s necessary tools; for branching SOPs with single-field outputs (customer\_service, video\_classification) it is estimated empirically as the tools used by the majority of correct prompt-condition executions of that task. This per-case denominator avoids penalizing legitimate early exits and alternative valid paths. Coverage is name-based and orderinsensitive: it verifies which tools were called, not their order, parameters, or whether outputs were consumed (Limitations). Trials below the threshold indicate the model bypassed the SOP; we report the ungrounded rate—correct answers from nonadherent trials—alongside. (The bridge study, rerun with DeepSeek V3.2 through the same pipeline, shows the same pattern: ungrounded answers concentrate under rfc2119 delivery and vanish under MCP; Table 2.)

Path Consistency. We introduce dominant toolsequence percentage as a behavioral predictability metric, computed only among adherent trials. For each (domain, model, condition) group, we extract the ordered sequence of tool names, count the frequency of each unique sequence, and report the percentage following the most common (dominant) sequence. Higher values indicate more reproducible behavior.

## 4 Results

All accuracy results report Grounded TSR—correct answers from trials where the model adhered to the SOP (pa\_coverage ≥ 0.5), divided by total trials—measuring only accuracy backed by an auditable process trace. Results pool trials across 12 domains (video\_annotation excluded: branching makes the threshold unreliable; Limitations); deltas carry trial-level bootstrap CIs (10,000 resamples, 95%; n ≈ 1,880 per model×condition). GLM 5’s grounded delta is aggregation-sensitive (trial-pooled −2.9pp vs. domain-mean +0.8pp; domain sizes 30–274); all other effects are robust to the aggregation choice.

## 4.1 Failure Modes Under Prompt-Based Delivery

Table 7 (Appendix D) quantifies the two failure modes under prompt-based delivery that motivate this work. First, hallucinated process execution: every model produces correct answers without executing the prescribed tools—2.1–4.5% of all trials, and on know\_your\_business 31–49% ofeach model’s correct answers, including 48% of the frontier executor’s (Appendix E). Standard accuracy metrics count these as successes. Second, process reinvention: given the identical SOP, the lightweight executor averages 20 distinct tool-call sequences per domain, with barely half of adherent trials on its most common path. Under these conditions no auditor can certify compliance and no supervisor can predict the next execution.

## 4.2 Effect on Ungrounded Answers

Step-level delivery drives the ungrounded rates of Table 7 to ≈0.3% for every model: Ministral −4.2pp [−5.2, −3.2], DeepSeek −2.9pp $[ - 3 . 8 , - 2 . 1 ] , \mathrm { K i m i } - 2 . 1 \mathrm { p p } [ - 2 . 9 , - 1 . 4 ] , \mathrm { G L M } 5$ $- 1 . 9 \mathrm { p p } \ [ - 2 . 6 , - 1 . 2 ]$ (all CIs exclude zero); on know\_your\_business the ungrounded share of correct answers collapses from 31–49% to 0– 8% (Table 8). This near-elimination is partly structural—the server withholds the next step until the current one executes—and it is measured against coverage-based adherence (Appendix F); under a stricter completeness criterion it is robust for the lightweight executor but not identifiable for the capable ones (Limitations). Two findings survive regardless: the problem it removes is large under prompt-based delivery (up to 49% of correct answers on know\_your\_business), and all four models comply in practice rather than answering from priors.

## 4.3 Process Adherence and Path Consistency

Table 3 reports both predictability dimensions. Adherence (trial-pooled) rises significantly for every model, most for the lightweight one (+18.8pp). This holds for any threshold $t \in [ 0 . 3 , 0 . 7 ]$ , but requiring every expected tool (t = 1.0) reverses it for GLM and Kimi (87→84%, 89→83%). Among adherent trials, path consistency—the fraction following the single most common tool-call sequence, computed per domain and macro-averaged—rises for every model, but under a domain-level bootstrap the gain is significant only for DeepSeek V3.2 (+13.7pp) and GLM 5 (+7.0pp); Kimi and Ministral gain about +4.5pp with intervals spanning zero. Distinct-path counts shrink for all capable executors (path-count ratios MCP/RFC of 0.48–0.85×). For linear SOPs this approaches determinism: on dangerous\_goods, GLM’s paths collapse from 6 to 1 (100% dominant).

<table><tr><td rowspan="2">Executor</td><td colspan="2">Adherence</td><td colspan="2">Consistency</td></tr><tr><td>RFC</td><td>MCP ∆</td><td>RFC MCP</td><td>∆</td></tr><tr><td>Ministral</td><td>76%</td><td>95% +19*</td><td>53%</td><td>58% +5</td></tr><tr><td>DeepSeek</td><td>86%</td><td>97% +11*</td><td>65%</td><td>78%  $+ 1 4 ^ { * }$ </td></tr><tr><td>GLM</td><td>92%</td><td>99%  $+ 7 ^ { * }$ </td><td>70%</td><td>77%  $+ 7 ^ { * }$ </td></tr><tr><td>Kimi</td><td>95%</td><td>99%</td><td>+4* 74%</td><td>79% +4</td></tr></table>

Table 3: Adherence (trial-pooled; fraction of trials executing at least half the expected tools) and path consistency (dominant tool-sequence % among adherent trials; per-domain macro-average), 12 domains. All adherence deltas are significant. Consistency deltas carry two-stage cluster-bootstrap CIs (resampling domains, then trials within domain; 10,000 resamples): Ministral +4.6 [−1.8, +11.3], DeepSeek +13.7 [+5.8, +23.0], GLM 5 +7.0 [+0.1, +17.9], Kimi +4.5 [−0.3, +10.0]. \* = CI excludes zero.

The lightweight exception. One predictability dimension does not transfer to the lightweight executor: step-level delivery makes Ministralfollow the process nearly as often as the capable executors (95% vs. 97–99% adherence), but its path consistency barely moves (53%→58%) and its distinctpath count rises under MCP (path ratio 1.47×). Traces show genuine ordering variance concentrated on branching domains rather than execution noise. Executing the process the same way each run remains capability-bound, like the accuracy effect reported next.

## 4.4 Effect on Grounded Accuracy

Table 4 turns to accuracy, the one dimension where the effect depends on the executor.

The lightweight open-weight executor gains substantially in grounded accuracy (Ministral 3 8B, +6.5pp, significant), while the large open-weight and frontier executors show no grounded gain (DeepSeek and Kimi within noise; GLM 5 slightly negative). The raw column makes the composition explicit: Ministral’s raw TSR rises only +2.3pp (n.s.)—roughly two-thirds of its grounded gain is reclassification of shortcut answers into process-

<table><tr><td></td><td></td><td colspan="3">Grounded TSR</td><td>Raw TSR</td></tr><tr><td>Executor</td><td>Tier</td><td>RFC</td><td>MCP</td><td>∆ [95% CI]</td><td>∆ [95% CI]</td></tr><tr><td>Ministral</td><td>light</td><td>61.0%</td><td>67.5%</td><td> $+ 6 . 5 [ + 3 . 4 , + 9 . 6 ] ^ { \ast }$ </td><td> $+ 2 . 3 \ [ - 0 . 7 , + 5 . 4 ]$ </td></tr><tr><td>DeepSeek</td><td>large</td><td>78.6%</td><td>79.0%</td><td>+0.4 [−2.2, +3.0]</td><td> $- 2 . 5 [ - 5 . 1 , - 0 . 1 ] ^ { * }$ </td></tr><tr><td>GLM</td><td>large</td><td>84.8%</td><td>82.0%</td><td> $- 2 . 9 [ - 5 . 2 , - 0 . 5 ] ^ { * }$ </td><td> $- 4 . 7 \ [ - 6 . 9 , - 2 . 4 ] ^ { * }$ </td></tr><tr><td>Kimi</td><td>frontier</td><td>85.5%</td><td>83.8%</td><td> $- 1 . 7 \ [ - 4 . 0 , + 0 . 7 ]$ </td><td> $- 3 . 8 [ - 6 . 0 , - 1 . 5 ] ^ { * }$ </td></tr></table>

Table 4: Grounded and raw TSR deltas by executor (trial-pooled, 12 domains; raw = grounded + ungrounded). Bootstrap 95% CIs, 10,000 trial-level resamples; \* = CI excludes zero.

## 5 Discussion

backed ones—while capable executors’ raw TSR declines (GLM 5 −4.7pp, Kimi −3.8pp, DeepSeek −2.5pp, all significant) as shortcuts are removed without replacement. We consider this the point rather than a caveat: the intervention converts unaudited into audited accuracy. Ministral’s gain is adherence-driven (76%→95%, Table 3; accuracy among adherent trials roughly unchanged, Appendix H): the model could execute each step but failed to reconstruct the procedure from the prompt. Capable models (adherence 86–95%) preserve grounded accuracy and gain visibility. Excluding the two domains with empirically derived adherence denominators (see Evaluation) strengthens the pattern: Ministral +8.8pp [+5.5, +12.1], DeepSeek +5.0pp [+2.2, +7.9], GLM 5 +0.7pp and Kimi +0.9pp (both n.s.) on the remaining 10 domains.

Per-domain variation. Gains concentrate where prompt-based adherence was poor (traffic\_spoofing\_detection +39.8pp averaged over models, patient\_intake +13.3pp, know\_your\_business +7.6pp); losses concentrate in branching decision-tree domains where step-scoped autonomy prevents lookahead (video\_classification −21.6pp, referral\_abuse\_detection\_v2 −14.0pp). Appendices A and B give the full breakdown.

## 4.5 Cost, Latency, and Errors

The lightweight executor is cheapest and fastest (\$0.015 per grounded answer, 17s per trial under MCP) but does not match the frontier executor’s grounded accuracy; step-level delivery costs 2.1– 2.9× more, and tool errors stay at 1–3% of calls. Appendix I gives per-model tokens, cost, and latency.

## 5.1 The Autonomy–Predictability Trade-off

The central question is not whether step-level delivery uniformly improves accuracy—it does not: it repairs process errors, not judgement, and a stepscoped model cannot look ahead—but whether the trade of autonomy for predictability is worthwhile and for which models. The effect is capabilitydependent: all four models near-eliminate ungrounded answers; capable executors consolidate execution paths; the lightweight executor additionally gains grounded accuracy.

The interaction with model capability is instructive. The frontier and large open-weight executors reconstruct the process effectively from the prompt (Kimi K2.5: 85% Grounded TSR, 95% adherence, 74% path consistency; DeepSeek and GLM 5 similar). For them, step-level delivery does not raise grounded accuracy—but it lifts adherence toward 97–99%, raises path consistency (significantly for DeepSeek and GLM 5, +7 to +14pp), and produces a step-level audit trail. The frontier executor does not need externalized process control for accuracy; the organization deploying it needs the visibility.

The lightweight executor is different. Ministral 3 8B at 61% Grounded TSR under promptbased delivery does not fail for lack of domain knowledge—it has the same SOP, tools, and data. It fails to reconstruct a multi-step procedure from a long document: adherence is only 76%, and on some domains it abandons the SOP in most trials. Step-level delivery removes this reconstruction burden—each instruction is concise, specifies one action, names the exact tool—lifting adherence to 95% and grounded accuracy by +6.5pp. Here externalized process control partially substitutes for capability.

At ≈3.5× lower cost per grounded answer, this makes the lightweight executor a credible engine for process-grounded automation. What remains capability-bound is precisely scoped: at 67.5% grounded it does not match the larger executors (79.0–83.8%), most of its grounded gain is shortcut answers becoming process-backed (raw gain +2.3pp, n.s.), and it follows the process far more often without executing it reproducibly (Section 4).

## 5.2 SOP Structure and Delivery Fit

SOPs in practice range from sequential checklists, through hierarchical step/sub-step structures that still follow one path, toflowchart procedures whose next step depends on the current outcome (Dumas et al., 2018).

Our results reveal an interaction between SOP structure and delivery benefit along two axes: path consolidation and grounded accuracy. For sequential SOPs (dangerous\_goods, order\_fulfillment, content\_flagging), steplevel delivery achieves near-deterministic execution for capable executors: distinct paths collapse toward one and dominant-path consistency reaches 97–100%. For branching SOPs (customer\_service, email\_intent), path consistency stays low (21–37%) because the variation reflects legitimate branching—different inputs trigger different decision paths—which step-level delivery cannot (and should not) eliminate.

Grounded-accuracy gains, however, track adherence repair more than structure: the largest MCP lifts occur where prompt-based adherence was poor, while branching decision-tree domains that require look-ahead can lose grounded accuracy under stepscoped autonomy (per-domain figures in §4). The recent SOP-Maze benchmark (Wang et al., 2026) likewise finds that SOP structure shapes failure modes.

This suggests a natural scope: step-level delivery is most effective for sequential and hierarchical SOPs—the majority of operational procedures in regulated industries. Extending it to conditional branching, where the server selects the next step from tool outputs, is a clear direction for future work.

## 5.3 Traceability Enables Targeted Optimization

The step-level audit trail is not merely a compliance artifact—because each step declares its inputs and outputs in a fixed shape, it is an interface another program can consume, which is what makes an automated revision cycle possible at all.

Case study: customer\_service. In pilot runs, the customer\_service domain exhibited a −14pp MCP penalty. Step traces revealed the root cause: Steps 9 and 10 contained a decision tree for resolution vs. escalation, but the ordering caused the model to match ESCALATED before evaluating RESOLVED. An SOP optimizer agent, given the step-level failure traces, reordered the decision tree (RESOLVED first) within 60 seconds. Following the correction, the pilot MCP penalty was reduced to −4pp; the remaining gap traces to a separate conditional-execution issue in Step 9.

This optimization cycle—run trials, identify failing steps from traces, revise those steps, re-run—is feasible only with step-level granularity. Under prompt-based delivery, the same failure manifests as “wrong answer” at the trial level, requiring manual conversation analysis across hundreds of trials to locate the root cause.

The same traceability also exposed a benchmark defect in the excluded warehouse\_package\_inspection domain, where higher prompt-based accuracy came from circumventing the tools entirely (Appendix C).

## 5.4 Evidence a Domain Expert Can Check

Verifiability through step-level evidence. Every MCP step produces a step\_output documenting what the model did, called, and concluded. This is not obtainable from the full-SOP prompt baseline we evaluate, where the model reasons internally and emits only a final answer; a prompt-based agent could be wrapped to emit comparable checkpoints, but that externalizes the process structure by another route. This is extrinsic trust (Jacovi et al., 2021): observable evidence of correct behavior, versus impractical intrinsic trust in internal reasoning. MCP shifts the trust question from “Is this model capable?” to “Is this process correct?”— answerable by domain experts without ML expertise, and answerable before deployment, since the step decomposition fixes the path in advance.

## 6 Conclusion

The barrier to deploying LLM agents is not capability but trust (Strunk et al., 2024; Kaur et al., 2023). Step-level SOP delivery fixes the execution path in advance, raises adherence to 95–99%, documents every step, and cuts hallucinated process execution to ≈0.3%, at the price of a small raw-accuracy decrement for capable executors.

## Limitations

This study has several limitations.

1. Bundled mechanisms. Step-level delivery bundles three mechanisms that this study does not fully disentangle: information presentation (the model sees shorter, more specific instructions per turn), process enforcement (the model cannot see ahead), and per-step behavioral meta-instructions (e.g., “treat null tool returns as success”). An ablation delivering the SOP in sequential chunks without enforcement—where the model sees one step at a time but can request any step— would isolate these effects; the per-step metainstructions could similarly be added to the prompt condition to control for their contribution.

2. Mock tools. SOP-Bench uses mock tools with precomputed results—real-world APIs introduce additional failure modes (timeouts, rate limits, schema changes) not captured here.

3. Sampling temperature. We do not set temperature=0; while the 15,475-trial sample provides statistical power, individual domain×model cells (as few as 30 trials) have wider confidence intervals.

4. Formalization confound. The two-condition design (RFC 2119 prompt vs. MCP) does not isolate step-level delivery of informal SOPs, which would separate the delivery mechanism from formalization.

5. SOP quality amplification. MCP amplifies SOP quality in both directions—poorly specified SOPs produce consistent but wrong execution, raising the quality bar for SOP authoring; two benchmark defects we corrected (Section 3) illustrate both the risk and the diagnostic value of step traces.

6. Domain exclusion. One SOP-Bench domain was excluded due to a tool–ground-truth mismatch unrelated to our intervention.

7. Adherence threshold. Our process adherence metric (pa\_coverage ≥ 0.5) measures coverage against per-case necessary tools where the output identifies the branch, and against empirically derived tool sets for single-output branching domains; video\_annotation’s wide branching still defeats this threshold and is excluded from adherence analysis.

8. Trust is measured indirectly. We measure the technical infrastructure for trust— adherence, path consistency, grounded accuracy, and step-level evidence—not trust itself. Whether these predictability gains shift actual human delegation decisions remains an open empirical question requiring user studies.

9. Trace-guided SOP debugging. The SOP debugging cycle used MCP step-level traces, meaning corrections were optimized for defects visible under step-level delivery; an analogous cycle using prompt-based conversation logs might have produced differently structured SOPs. Two further caveats bear on interpretation. First, the pilot runs that surfaced these defects drew on the same SOP-Bench task set used for the reported evaluation, not a disjoint development split, so the corrections may be tuned to these tasks; a held-out split would be the cleaner design. Second, the corrections were frozen before the primary runs in the sense that every reported trial for a revised domain is a post-correction re-run over that domain’s full task set — no reported cell mixes pre- and post-correction trials. The revised SOPs are released with the artifacts so the exact evaluation environment can be reconstructed.

10. Aggregation sensitivity. Grounded-accuracy deltas can be aggregation-sensitive when domain sizes vary (30–274 tasks): GLM 5’s delta is −2.9pp trial-pooled but +0.8pp domain-mean; we report trial-pooled figures with matching trial-level CIs and flag this sensitivity explicitly.

11. Name-based coverage, and why a stricter criterion is not identifiable. Adherence coverage is name-based: it does not verify call order, parameters, or that tool outputs were consumed, so a trial can count as adherent while misusing results. We attempted a stricter completeness criterion—did the trial call every tool the task actually required—and found it is not identifiable on this benchmark, because SOP-Bench’s ground truth does not enumerate per-task required tools. Every available substitute reference is condition-dependent:

deriving the required set from correct promptcondition trials makes step-level delivery look worse, deriving it from correct step-level trials makes prompt delivery look worse by a similar margin, and the one reference independent of both—mapping ground-truth output fields to the tools that produce them—is available for only two of twelve domains, because the rest have scalar rather than structured ground truth. We therefore report the coverage-based metric, the only definition that generalises across all domains, and note that the ungrounded reduction is robust for the lightweight executor under every definition we tried but metric-dependent for the capable ones. Per-task required-tool annotation is the missing instrument and the clearest next step for this line of work. Separately, the fuzzy output matcher (containment-based) was spotchecked but not human-validated at scale, so verbose answers could in principle produce false-positive matches.

12. Per-domain uncertainty. Per-domain figures rest on cells of 30–274 tasks without perdomain CIs or multiple-comparison control and should be read as descriptive.

Artifacts. The RFC 2119 SOPs, step decompositions, system prompts, corrected tool specifications, per-trial OpenTelemetry traces, and analysis code will be released upon publication, so that the evaluation environment (including the two benchmark fixes) can be reproduced exactly. Model runs were launched from per-model configuration files that write into a shared results directory rather than from one combined config, so reproduction proceeds model by model.

## References

Anthropic. 2025. Model Context Protocol specification. Revision 2025-06-18. Accessed 2026-08-10.

Scott Bradner. 1997. Key words for use in RFCs to indicate requirement levels. RFC 2119, BCP 14. Updated by RFC 8174.

Harrison Chase. 2022. LangChain. https://github. com/langchain-ai/langchain. Released 2022- 10-17. Accessed 2026-08-10.

Suzanne de Treville, John Antonakis, and Norman M. Edelson. 2005. Can standard operating procedures be motivating? Reconciling process variability issues

and behavioural outcomes. Total Quality Management & Business Excellence, 16(2):231–241.

Marlon Dumas, Marcello La Rosa, Jan Mendling, and Hajo A. Reijers. 2018. Fundamentals of Business Process Management, 2 edition. Springer Berlin Heidelberg.

Jeffrey Flynt. 2026. GroundEval: A deterministic replacement for LLM-as-judge in stateful agent evaluation. arXiv:2606.22737.

Sachin Kumar Giroh, Pushpendu Ghosh, Aryan Jain, Harshal Giridhari Paunikar, Aditi Rastogi, Promod Yenigalla, and Anish Nediyanchath. 2025. <SYNTACT>: Structuring your natural language SOPs into tailored ambiguity-resolved code templates. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 2367–2376, Suzhou (China). Association for Computational Linguistics.

Donghao Huang, Joon Kiat Chua, and Zhaoxia Wang. 2026. Beyond task success: Measuring workflow fidelity in LLM-based agentic payment systems. In PAKDD 2026 Workshop on AI and Data Sciencefor Digital Finance (AI4DF).

Alon Jacovi, Ana Marasovic, Tim Miller, and Yoav´ Goldberg. 2021. Formalizing trust in artificial intelligence: Prerequisites, causes and goals of human trust in AI. In Proceedings ofthe 2021 ACM Conference on Fairness, Accountability, and Transparency, pages 624–635.

Davinder Kaur, Suleyman Uslu, Kaley J. Rittichier, and Arjan Durresi. 2023. Trustworthy artificial intelligence: A review. ACM Computing Surveys, 55(2):39:1–39:38.

Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal. 2023. Decomposed prompting: A modular approach for solving complex tasks. In Proceedings of the Eleventh International Conference on Learning Representations (ICLR).

Subhrangshu Nandi, Arghya Datta, Rohith Nama, Udita Patel, Nikhil Vichare, Indranil Bhattacharya, Prince Grover, Shivam Asija, Giuseppe Carenini, Wei Zhang, and 1 others. 2025. SOP-Bench: Complex industrial SOPs for evaluating LLM agents. arXiv preprint arXiv:2506.08119.

Yunjia Qi, Hao Peng, Xiaozhi Wang, Amy Xin, Youfeng Liu, Bin Xu, Lei Hou, and Juanzi Li. 2025. AgentIF: Benchmarking large language models instruction following ability in agentic scenarios. In Advances in Neural Information Processing Systems 38 (NeurIPS 2025) Datasets and Benchmarks Track.

Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. 2023. Hugging-GPT: Solving AI tasks with ChatGPT and its friends in Hugging Face. In Advances in Neural Information Processing Systems 36 (NeurIPS).

Harsh Soni. 2026. ToolFailBench: Diagnosing tool-use failures in LLM agents. In ICML 2026 Workshop on Agents in the Wild: Safety, Security, and Beyond (AIWILD).

Jobin Alexander Strunk, Leonardo Banh, Anika Nissen, Gero Strobel, and Stefan Smolnik. 2024. To delegate or not to delegate? factors influencing human-agentic IS interaction. In Proceedings of the International Conference on Information Systems (ICIS 2024). Paper 1403.

Harsh Trivedi, Tushar Khot, Mareike Hartmann, Ruskin Manku, Vinty Dong, Edward Li, Shashank Gupta, Ashish Sabharwal, and Niranjan Balasubramanian. 2024. AppWorld: A controllable world of apps and people for benchmarking interactive coding agents. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 16022–16076, Bangkok, Thailand. Association for Computational Linguistics.

Jiaming Wang, Zhe Tang, Zehao Jin, Hefei Chen, Yilin Jin, Peng Ding, Xiaoyu Li, and Xuezhi Cao. 2026. SOP-Maze: Evaluating large language models on complicated business standard operating procedures. In Findings of the Association for Computational Linguistics: ACL 2026, pages 14568–14588.

Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim. 2023. Planand-solve prompting: Improving zero-shot chain-ofthought reasoning by large language models. In Proceedings ofthe 61st Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 2609–2634.

Qinzhuo Wu, Wei Liu, Jian Luan, and Bin Wang. 2024. ToolPlanner: A tool augmented LLM for multi granularity instructions with path planning and feedback. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 18315–18339, Miami, Florida, USA. Association for Computational Linguistics.

Ruixuan Xiao, Wentao Ma, Ke Wang, Yuchuan Wu, Junbo Zhao, Haobo Wang, Fei Huang, and Yongbin Li. 2024. FlowBench: Revisiting and benchmarking workflow-guided planning for LLM-based agents. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, pages 10883–10900, Miami, Florida, USA. Association for Computational Linguistics.

Weimin Xiong, Yifan Song, Xiutian Zhao, Wenhao Wu, Xun Wang, Ke Wang, Cheng Li, Wei Peng, and Sujian Li. 2024. Watch every step! LLM agent learning via iterative step-level process refinement. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 1556–1572, Miami, Florida, USA. Association for Computational Linguistics.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. 2025. τ-bench: A benchmark for tool-

agent-user interaction in real-world domains. In Proceedings ofthe Thirteenth International Conference on Learning Representations (ICLR).

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing reasoning and acting in language models. In Proceedings ofthe Eleventh International Conference on Learning Representations (ICLR).

Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, and Ed Chi. 2023. Least-to-most prompting enables complex reasoning in large language models. In Proceedings of the Eleventh International Conference on Learning Representations (ICLR).

## Appendix

## A Per-Domain MCP Lift

Table 5 shows the grounded MCP lift by domain, averaged across the four models.

<table><tr><td>Domain</td><td>Avg MCP ∆ +39.8</td></tr><tr><td>traffic_spoofing_detection patient_intake know_your_business dangerous_goods content_flagging order_fulfillment</td><td>+13.3 +7.6 +3.1 +2.5 +2.5</td></tr><tr><td>aircraft_inspection customer_service email_intent</td><td>-2.2 -3.4 -3.5</td></tr><tr><td>referral_abuse_detection_v1 referral_abuse_detection_v2 video_classification</td><td>-6.4 -14.0 -21.6</td></tr></table>

Table 5: Average grounded MCP lift by domain in percentage points (simple mean of the four per-model Grounded TSR deltas, computed from unrounded permodel values). Gains concentrate where prompt-based adherence was poor (traffic\_spoofing\_detection, patient\_intake); losses concentrate in branching decision-tree domains where step-scoped autonomy prevents look-ahead.

## B Grounded TSR and Path Variance: Full Results

Table 6 reports Grounded TSR for every domain×model×condition cell, with unique execution-path counts in parentheses (computed among adherent trials, pa\_coverage ≥ 0.5). A cell shows what fraction of trials produced a correct answer while following the SOP, and how many distinct tool-call sequences the adherent trials took.

<table><tr><td rowspan="2">Domain</td><td colspan="4">rfc2119_sop</td><td colspan="4">sop_mcp</td></tr><tr><td>M8B</td><td>DS</td><td>GLM</td><td>Kimi</td><td>M8B</td><td>DS</td><td>GLM</td><td>Kimi</td></tr><tr><td>aircraft_inspection</td><td>79% (25)</td><td>79% (9)</td><td>88% (15)</td><td>86% (8)</td><td>77% (28)</td><td>79% (6)</td><td>80% (4)</td><td>87% (8)</td></tr><tr><td>content_flagging</td><td>88% (10)</td><td>89% (5)</td><td>97% (9)</td><td>99% (4)</td><td>89% (18)</td><td>96% (3)</td><td>99% (4)</td><td>99% (4)</td></tr><tr><td>customer_service</td><td>58% (28)</td><td>85% (22)</td><td>92% (38)</td><td>82% (21)</td><td>70% (60)</td><td>69% (13)</td><td>88% (19)</td><td>76% (13)</td></tr><tr><td>dangerous_goods</td><td>57% (18)</td><td>95% (12)</td><td>96% (6)</td><td>96% (1)</td><td>78% (15)</td><td>90% (4)</td><td>94% (1)</td><td>93% (4)</td></tr><tr><td>email_intent</td><td>70% (33)</td><td>86% (7)</td><td>92% (8)</td><td>91% (5)</td><td>73% (16)</td><td>85% (5)</td><td>77% (6)</td><td>91% (5)</td></tr><tr><td>know_your_business</td><td>31% (23)</td><td>41% (11)</td><td>50% (3)</td><td>41% (7)</td><td>40% (39)</td><td>39% (7)</td><td>56% (4)</td><td>59% (6)</td></tr><tr><td>order_fulfillment</td><td>80% (2)</td><td>83% (4)</td><td>83% (2)</td><td>87% (1)</td><td>87% (3)</td><td>83% (2)</td><td>87% (2)</td><td>87% (1)</td></tr><tr><td>patient_intake</td><td>59% (17)</td><td>88% (5)</td><td>48% (10)</td><td>98% (6)</td><td>70% (15)</td><td>86% (3)</td><td>100% (1)</td><td>91% (3)</td></tr><tr><td>referral_abuse_detection_v1</td><td>95% (7)</td><td>99% (4)</td><td>100% (2)</td><td>99% (1)</td><td>86% (20)</td><td>93% (2)</td><td>94% (1)</td><td>94% (4)</td></tr><tr><td>referral_abuse_detection_v2</td><td>60% (15)</td><td>95% (9)</td><td>94% (8)</td><td>96% (7)</td><td>38% (24)</td><td>83% (4)</td><td>84% (3)</td><td>85% (2)</td></tr><tr><td>traffic_spoofing_detection</td><td>6% (18)</td><td>9% (5)</td><td>50% (7)</td><td>52% (7)</td><td>64% (29)</td><td>68% (3)</td><td>72% (9)</td><td>72% (3)</td></tr><tr><td>video_classification</td><td>60% (44)</td><td>81% (14)</td><td>84% (20)</td><td>83% (11)</td><td>44% (85)</td><td>59% (5)</td><td>53% (7)</td><td>65% (14)</td></tr></table>

Table 6: Grounded TSR by domain, model, and condition; parenthesized numbers are unique execution paths among adherent trials. M8B = Ministral 3 8B, DS = DeepSeek V3.2, Kimi = Kimi K2.5.

## C Warehouse Package Inspection: When Higher Accuracy Is the Wrong Signal

The excluded warehouse\_package\_inspection domain illustrates why traceability matters beyond accuracy optimization. In our bridge study (DeepSeek V3.2, 600 trials), prompt-based delivery achieved 57% TSR while MCP achieved only 45%.

Under MCP, the model faithfully followed the SOP: call validateBarcode, branch on barcode\_match, execute downstream tools, and report the tool-determined resolution\_status. Step traces confirmed correct execution in every trial. Yet 55% were marked incorrect.

The root cause: the benchmark’s validateBarcode mock tool determines barcode\_match using PO number parity (po\_number % 2 == 0), while the ground truth CSV contains independently generated values. These disagree in 62 of 150 test cases (41%). A perfect tool-following agent would score at most (150 − 62)/150 = 58.7%; the observed MCP result of 45% falls below this ceiling due to additional failure sources in downstream tool logic.

The prompt-based model achieved 57% by circumventing the tools: it read the barcode\_match column directly from task input data and inferred the resolution status without relying on tool outputs. This is pattern matching on input features, not SOP execution. The higher accuracy masked a benchmark defect invisible to prompt-based evaluation.

This illustrates a general principle: accuracy without traceability can be actively misleading. In production, the equivalent scenario is an agent that produces plausible outputs by shortcutting the prescribed procedure—passing audits on accuracy while violating process compliance.

## D Failure Modes Under Prompt-Based Delivery

Table 7 quantifies the two failure modes that motivate this work (Section 4): hallucinated process execution and process reinvention under promptbased delivery.

<table><tr><td rowspan="2">Executor</td><td colspan="3">Ungrounded</td><td colspan="2">Reinvention</td></tr><tr><td>Ungr.</td><td>/Corr</td><td>KYB</td><td>#Paths</td><td>Cons.</td></tr><tr><td>Ministral</td><td>4.5%</td><td>6.8%</td><td>44%</td><td>20.0</td><td>53%</td></tr><tr><td>DeepSeek</td><td>3.1%</td><td>3.8%</td><td>49%</td><td>8.9</td><td>65%</td></tr><tr><td>GLM</td><td>2.1%</td><td>2.4%</td><td>31%</td><td>10.7</td><td>70%</td></tr><tr><td>Kimi</td><td>2.3%</td><td>2.6%</td><td>48%</td><td>6.6</td><td>74%</td></tr></table>

Table 7: Prompt-based delivery (RFC condition, 12 domains). Columns 2–4: correct answers produced without executing the prescribed tools (% of trials, share of correct, worst-domain share on know\_your\_business). Columns 5–6: mean distinct tool-call sequences per domain among adherent trials, and dominant-sequence consistency.

## E Ungrounded Answers on know\_your\_business

Table 8 backs the per-model figures quoted in the abstract, introduction, and Section 4: the fraction of correct answers on know\_your\_business that were produced without executing the prescribed verification tools.

<table><tr><td colspan="3">RFC</td><td colspan="2">MCP</td></tr><tr><td>Executor</td><td>Ungr/Corr</td><td>(n)</td><td>Ungr/Corr</td><td>(n)</td></tr><tr><td>Ministral</td><td>44%</td><td>22/50</td><td>8%</td><td>3/39</td></tr><tr><td>DeepSeek</td><td>49%</td><td>36/73</td><td>0%</td><td>0/35</td></tr><tr><td>GLM</td><td>31%</td><td>20/65</td><td>0%</td><td>0/50</td></tr><tr><td>Kimi</td><td>48%</td><td>34/71</td><td>0%</td><td>0/54</td></tr></table>

Table 8: know\_your\_business: ungrounded correct answers as a share of all correct answers, per model and condition (90 tasks per cell). Under prompt-based delivery, 31–49% of every model’s correct answers bypass the prescribed tools; under step-level delivery this falls to 0–8%.

## F Definition of pa\_coverage

For a trial, let C be the set of distinct tool names the model actually called and N the set of expected tool names for that task. Then

$$
{ \mathsf { p a _ { - } c o v e r a g e } } = { \frac { | C \cap N | } { | N | } } ,
$$

and the trial counts as adherent when pa\_coverage ≥ 0.5. Because both sides are sets of names, the metric has the following explicit behaviour, which we state so its limits are unambiguous:

• Duplicate calls and retries collapse to one element and neither help nor hurt.

• Call ordering is ignored; a trial calling the right tools in the wrong order scores identically to one in the prescribed order.

• Parameters and tool outputs are not inspected, so a trial can be adherent while passing wrong arguments or ignoring what a tool returned.

• Extra tools outside N are ignored; there is no penalty for calling tools the SOP did not prescribe.

• Conditional branches are handled through N, which is defined per task rather than per domain: for linear SOPs it is the SOP’s prescribed domain tools; where a multi-field output identifies the branch taken (email\_intent) it is that branch’s necessary tools; and for single-output branching domains (customer\_service, video\_classification) it is estimated empirically as the tools used by the majority of correct prompt-condition executions of the same task.

• Empty expected set. If N is empty the trial is assigned coverage 1.0 and is therefore vacuously adherent.

The metric is thus a coverage proxy for procedural compliance, not a proof of correct execution; Limitations records this, and the two domains whose N is prompt-derived are the ones we exclude in the robustness check reported in Section 4. This is a deliberately looser criterion than related benchmarks apply: FlowBench (Xiao et al., 2024) counts an invocation correct only when the tool name and all required parameters match, checked against the ground-truth sequence, and ToolPlanner (Wu et al., 2024) requires the solution to match the prescribed tool set at the instruction’s granularity level. We adopt coverage because SOP-Bench tasks admit multiple valid orderings and legitimate early exits, and because the stricter criterion is not identifiable here (Limitations).

## G Adherence Threshold Sensitivity

Table 9 shows Grounded TSR at varying adherence thresholds (pa\_coverage ≥ t). Effect directions are stable for Ministral (+6.3 to +8.2pp across $t \in \{ 0 . 3 , 0 . 5 , 0 . 7 \} )$ , Kimi $( - 1 . 7 \ \mathrm { t o } - 2 . 2 \mathrm { p p } , \mathrm { n . s . } )$ and GLM 5 (−2.9 to −3.0pp); DeepSeek hovers at zero (−0.3 to +0.9pp), crossing sign but never approaching significance.

<table><tr><td>t</td><td>Executor</td><td>rfc2119</td><td>sop_mcp</td><td>Δ</td></tr><tr><td>0.3</td><td>Ministral DeepSeek GLM Kimi</td><td>61.3% 79.2% 85.0% 86.0%</td><td>67.5% 79.0% 82.0% 83.8%</td><td>+6.3pp -0.3pp -3.0pp -2.2pp</td></tr><tr><td>0.5</td><td>Ministral DeepSeek GLM Kimi</td><td>61.0% 78.6% 84.8% 85.5%</td><td>67.5% 79.0% 82.0% 83.8%</td><td>+6.5pp +0.4pp -2.9pp -1.7pp</td></tr><tr><td>0.7</td><td>Ministral DeepSeek GLM Kimi</td><td>58.3% 78.1% 84.7% 85.4%</td><td>66.5% 79.0% 81.6% 83.6%</td><td>+8.2pp +0.9pp -3.0pp -1.8pp</td></tr></table>

Table 9: Grounded TSR sensitivity to adherence threshold t (trial-pooled, 12 domains). Directions are stable for Ministral, GLM 5, and Kimi; DeepSeek hovers at zero.

## H Grounded TSR Decomposition

Grounded TSR = adherence rate × accuracy among adherent trials. Table 10 decomposes the aggregate results. Acc|Adh declines under MCP for all models—a reclassification effect: step-level delivery pulls previously non-adherent trials (disproportionately hard cases the model used to abandon or shortcut) into the adherent population. Grounded TSR rises only where the adherence gain outweighs this dilution (Ministral +18.8pp adherence), stays flat where the two roughly cancel (DeepSeek), and dips slightly where adherence was already high (GLM 5, Kimi).

<table><tr><td>Executor</td><td>Cond.</td><td>Adh%</td><td>Acc|Adh</td><td>GTSR</td></tr><tr><td>Ministral Ministral</td><td>rfc2119 sop_mcp Δ</td><td>75.9% 94.7% +18.8pp</td><td>80.4% 71.2% -9.1pp</td><td>61.0% 67.5% +6.5pp</td></tr><tr><td>DeepSeek DeepSeek</td><td>rfc2119 sop_mcp ∆</td><td>86.5% 97.2% +10.7pp</td><td>90.8% 81.3% -9.6pp</td><td>78.6% 79.0% +0.4pp</td></tr><tr><td>GLM GLM</td><td>rfc2119 sop_mcp Δ</td><td>92.0% 98.9% +6.9pp</td><td>92.2% 82.9% -9.3pp</td><td>84.8% 82.0% -2.9pp</td></tr><tr><td>Kimi Kimi</td><td>rfc2119 sop_mcp ∆</td><td>94.5% 98.7% +4.2pp</td><td>90.4% 84.9% -5.6pp</td><td>85.5% 83.8% -1.7pp</td></tr></table>

Table 10: Grounded TSR decomposition (trial-pooled, 12 domains). Adh% = fraction of trials passing the adherence threshold; Acc|Adh = accuracy among adherent trials.

## I Token, Cost, and Latency Economics

Step-level delivery re-delivers the accumulated conversation at each step without prompt caching, multiplying input tokens by 2.1–2.9× and yielding a cost tax of 2.1–2.9× and latency tax of 2.2–2.8×. Despite this, the lightweight executor under MCP costs \$0.010/trial at 17s—4.3× cheaper per trial (equivalently ≈3.5× per grounded answer) and 4.8× faster than Kimi under MCP (\$0.044, 81s). Enabling prompt caching would substantially reduce the tax.

<table><tr><td>Executor</td><td>Cond.</td><td>Tok (in/out)</td><td>Dur (s)</td><td>$/trial</td></tr><tr><td>Ministral</td><td>rfc2119</td><td>22.9k / 0.8k</td><td>6.5</td><td>$0.004</td></tr><tr><td>Ministral</td><td>sop_mcp</td><td>66.0k / 1.8k</td><td>16.8</td><td>$0.010</td></tr><tr><td>DeepSeek</td><td>rfc2119</td><td>34.4k / 1.2k</td><td>33.4</td><td>$0.023</td></tr><tr><td>DeepSeek</td><td>sop_mcp</td><td>83.7k / 3.3k</td><td>93.9</td><td>$0.058</td></tr><tr><td>GLM</td><td>rfc2119</td><td>31.7k / 0.8k</td><td>32.3</td><td>$0.021</td></tr><tr><td>GLM</td><td>sop_mcp</td><td>68.9k / 1.7k</td><td>74.5</td><td>$0.045</td></tr><tr><td>Kimi</td><td>rfc2119</td><td>29.8k / 1.0k</td><td>36.6</td><td>$0.021</td></tr><tr><td>Kimi</td><td>sop_mcp</td><td>63.4k / 2.0k</td><td>80.8</td><td>$0.044</td></tr></table>

Table 11: Per-trial economics (trial-pooled, 12 domains). List prices (\$/M in/out): Ministral 3 8B \$0.15/\$0.15, DeepSeek V3.2 \$0.62/\$1.85, GLM 5 \$0.60/\$2.20, Kimi K2.5 \$0.60/\$3.00 (all verified against published list prices for the region used).

<table><tr><td>Executor</td><td>Cond.</td><td>Med.</td><td>p90</td><td>p99</td><td>Max</td></tr><tr><td>Ministral</td><td>RFC 2119</td><td>614</td><td>1,539</td><td>2,603</td><td>4,815</td></tr><tr><td rowspan="3">DeepSeek</td><td>SOP-MCP</td><td>1,413</td><td>3,357</td><td>5,790</td><td>9,814</td></tr><tr><td>RFC 2119</td><td>1,043</td><td>2,111</td><td>2,805</td><td>6,037</td></tr><tr><td>SOP-MCP</td><td>3,148</td><td>5,066</td><td>6,030</td><td>8,144</td></tr><tr><td>GLM</td><td>RFC 2119</td><td>756</td><td>1,462</td><td>1,828</td><td>2,137</td></tr><tr><td rowspan="2">Kimi</td><td>SOP-MCP</td><td>1,586</td><td>2,791</td><td>3,753</td><td>4,391</td></tr><tr><td>RFC 2119</td><td>972</td><td>1,814</td><td>2,756</td><td>6,824</td></tr><tr><td></td><td>SOP-MCP</td><td>1,809</td><td>3,495</td><td>4,386</td><td>5,064</td></tr></table>

Table 12: Output-token usage per trial (trial-pooled, 12 domains): total generated tokens summed across all responses of a trial. Because a trial spans multiple model responses, any single response – the unit the max\_tokens cap applies to – is strictly smaller than the trial total. Median usage sits at 15–77% of even the smaller 4,096 cap.