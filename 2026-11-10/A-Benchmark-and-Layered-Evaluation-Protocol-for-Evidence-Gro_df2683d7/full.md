# A Benchmark and Layered Evaluation Protocol for Evidence-Grounded Shared Belief Updating in Connected Robotics

Ranulfo Bezerra and Kimitoshi Yamazaki

Abstract—Connected physical systems fuse observations from robots, IoT sensors, and edge-AI modules into a shared world belief, yet most evaluations score only the final answer or action and cannot tell whether a method attributed the evidence, updated its belief, handled contradictions, and justified its decision. We argue that such shared reasoning should be evaluated as an evidence-to-belief trajectory and introduce the Connected Robotics Evidence Graph (CREG) Benchmark to measure it. CREG comprises 350 deterministically generated scenarios across seven regimes (justified revision, belief preservation, relation association, contradiction onset and resolution, stale evidence, and source reliability), each with a normative posterior reference at every evidence prefix, verified by an independent logodds checker and a blinded expert audit. Nine representation conditions separate relation content from serialization, ordering, and metadata, with residual confounds quantified. Evaluating four open-weight instruction models and six transparent fusion baselines yields three findings. First, entity grounding explains much of the weak-model deficit: weaker models lose trajectory score to outputs that never bind to the required entity, and in exploratory analysis their grounded updates approach those of stronger models. Second, deleting semantic relations costs 15.3 points of strict belief-state accuracy on relation-dependent scenarios (Holm-corrected p < 0.001), yet an informationmatched structured list outperforms the full graph and shuffling it does not hurt: relation content matters, graph serialization does not. Third, belief accuracy is strongly non-monotonic along the evidence sequence, with mid-trajectory failures that endpoint-only evaluation cannot detect. All scenarios, references, prompts, metrics, and code are available for offline, hardwarefree reproduction at https://creg-bench.github.io.

Index Terms—Belief revision, benchmark, connected robotics, evidence graphs, explainable AI, large language models, multiagent systems.

## I. INTRODUCTION

Connected robotic systems combine observations from mobile robots, Internet of Things (IoT) sensors, fixed cameras, and edge-AI modules to monitor environments that no single agent observes well enough to act alone [1]. The scientific problem they pose is not connectivity or sensing in isolation but the construction of a shared, defensible belief about the physical world from evidence that is distributed, uncertain, delayed, and sometimes contradictory. For inspection, monitoring, and response tasks, the quality of that belief, not merely the action it triggers, determines whether the system can be trusted.

Existing evaluations rarely test this construction directly: most score a final answer, a task-success flag, or a selected action. Such endpoint measures cannot separate a system that reasoned correctly from one that reached the expected output by accident, for example by binding its prediction to the wrong physical entity, citing the wrong evidence, missing a contradiction, or revising too early or too late; conversely, a correct belief may support an evidence-supported action that differs from the single reference action. An end-task score collapses all of these cases into one number.

Connected Robotics makes this gap consequential. Reports carry source identity and reliability varies by source, so two agreeing reports are not two independent confirmations; evidence may arrive late, describe a state that has since changed, or be superseded by a correction; and semantic relations, such as which sensor measures which asset, determine whether an observation applies to the target entity at all. Each mechanism can yield a correct final answer for the wrong reason, or a defensible intermediate step that an endpoint metric scores as failure.

We introduce the Connected Robotics Evidence Graph (CREG) Benchmark, a controlled benchmark and layered evaluation protocol for observing the full evidence → belief → contradiction → action → explanation chain (Fig. 1). Scenarios are generated from a normative specification that fixes source reliability, temporal and supersession semantics, and relation-transfer rules, so that every evidence prefix has a reference posterior and the belief trajectory can be scored step by step. Representation conditions present the same facts as flat records, typed lists, or graphs and separate relation content from serialization, ordering, and metadata where possible, with the remaining prompt-length and jointmanipulation confounds quantified. Grounding, belief revision, contradiction handling, action selection (reference-policy compliance, action-family agreement, and evidence support), and explanation grounding are scored separately.

Our central thesis is that shared reasoning in Connected Robotics should be evaluated as an evidence-to-belief trajectory, not as a final-answer problem. The contributions of this paper are:

1) We formulate shared belief updating in Connected Robotics as a trajectory-level evaluation problem that separately measures entity grounding, evidence attribution, belief revision, contradiction handling, action selection (policy compliance, family agreement, and evidence support), and explanation grounding.

2) We introduce CREG, a controlled benchmark of 350 distributed-evidence scenarios with per-prefix normative references, temporal and supersession semantics, source reliability, relation counterfactuals, and informationcontrolled representation conditions.

3) We evaluate CREG across four open-weight instruction models and six transparent fusion baselines, showing that (i) entity grounding accounts for a substantial part of the weak-model deficit, (ii) semantic relation content improves categorical belief accuracy where relational evidence is required while graph serialization itself shows no consistent advantage, and (iii) final-prefix evaluation misses large transient failures along the belief trajectory. The CREG benchmark, normative references, prompts, configurations, evaluation code, and analysis scripts are available from the project page at https://creg-bench.github.io.

## II. RELATED WORK

## A. Semantic World Models and Multi-Robot Scene Graphs

3D scene graphs give robots structured abstractions above metric geometry: 3D Dynamic Scene Graphs, Kimera, and Hydra construct layered place–object–agent representations online, Hydra-Multi merges them across robot teams, and ConceptGraphs and HOV-SG open the vocabulary for perception, planning, and language-grounded navigation [2], [3], [4], [5], [6], [7]. Closest to our setting, Semantic Belief Graphs and Semantic Belief Behavior Graphs couple uncertain semantic beliefs to navigation and inspection behaviors [8], [9], and heterogeneous inspection teams and the Connected Robotics formulation place such agents in a wider AI–IoT–robotics system [10], [1]. These works ask how to construct, merge, and act on world models. We ask a complementary question: given semantic observations with provenance, confidence, time, and possible conflict, does a reasoning method produce an appropriate shared belief, measured separately from whether its final action matches a policy? We propose no graph-construction pipeline and claim no intrinsic advantage for graph topology.

## B. Belief Fusion and Revision under Distributed Evidence

Evidence-theoretic combination [11], covariance intersection under unknown cross-correlation [12], and truth discovery under source dependence [13], [14] motivate benchmark fields for confidence, provenance, and source identity, and scenarios in which redundant reports must not count as independent confirmation. Belief revision formalizes how a knowledge state should change under conflicting information, through consistency, incorporation, and minimal change in the AGM account and its iterated extension [15], [16]. Recent language-model benchmarks operationalize parts of this problem: Belief-R tests retraction of defeasible conclusions and avoidance of unnecessary change [17], BeliefTrack scores update, preservation, and isolation over sequential evidence [18], and BayesBench compares turn-wise model beliefs with Bayesian references [19]. Their evidence is textual, logical, or simulated. CREG attaches reports to physical agents and entities, exposes time, confidence, provenance, and contradiction, and connects belief changes to operational actions; Table I summarizes the positioning against these and the embodied benchmarks below.

## C. Neural and Graph-Conditioned Robotic Reasoning

SayCan, PaLM-E, and RT-2 ground language-model decisions in learned affordances, embodied observations, and action tokens [24], [25], [26]; SayPlan plans over a hierarchical 3D scene graph with iterative correction [27]; and, outside robotics, Think-on-Graph and G-Retriever reason over or retrieve knowledge-graph structure [28], [29]. These systems motivate graph-conditioned evaluation, but they treat action success as the outcome and rarely control information content in graph-versus-text comparisons. A representation claim requires that graph and non-graph inputs preserve the same facts and differ only in controlled ways—serialization, ordering, identifiers, provenance, timestamps, and confidence—before an improvement can be assigned to structure rather than to prompt clarity or information loss. CREG approaches that requirement and quantifies where it falls short (Sec. III-C).

## D. Embodied and Explanation Benchmarks

ALFRED, TEACh, BEHAVIOR-1K, and OpenEQA define success at the instruction, dialogue, activity, or answer level [20], [21], [22], [23]; none of these outcome measures determines whether distributed evidence was attributed, fused, and revised correctly. Rationale evaluation supplies a second precedent: ERASER separates agreement with annotated evidence from whether a rationale influenced the prediction [30], and faithfulness work distinguishes plausible explanations from faithful accounts of model behavior [31]. We adopt the same decomposition: evidence-identifier overlap measures grounding coverage rather than faithfulness, detecting a contradiction is distinct from localizing it, and agreement with one deterministic policy measures compliance rather than decision quality. CREG brings these threads together: deterministic rules serve as a transparent reference policy, fusion strategies as inspectable comparison points, and language models are evaluated on the same frozen evidence under controlled representations.

## III. THE CREG BENCHMARK

## A. Scenario Suite and Normative References

The suite contains 350 scenarios generated deterministically from a normative specification. Each scenario concerns one target entity with a binary physical state (anomalous or normal) and a uniform prior. Evidence arrives as attributed reports, each carrying a source, a claimed value, a confidence, event and arrival times, and a state epoch. The specification fixes how these fields combine: a report’s effective accuracy is 0.5 + (confidence − 0.5) · reliability, so source reliability modulates stated confidence; state epochs and explicit supersession links determine which reports still apply to the current state; and relation-transfer rules determine when a sensor’s report reaches the target through a measures relation. $\mathsf { A p - }$ plying these rules to every cumulative evidence prefix yields a reference posterior, a categorical outcome (posteriors tied within $1 0 ^ { - 1 \bar { 2 } }$ are undetermined), a probability direction, supporting- and counterevidence sets, and a contradiction state. Reference action rules define the normative next action at the endpoint.

![](images/461a85dbca4061247dbf5a72ca289a6fe40be49e4da3782bb840c0f224d614ea.jpg)  
Fig. 1. Overview of CREG. Connected agents produce attributed semantic evidence organized in a shared graph; belief trajectory, contradiction records, next action, and explanation are exposed and scored separately, so an end-task success cannot conceal an earlier-layer error.

TABLE I  
CREG VERSUS THE CLOSEST PRIOR BENCHMARKS (∼: PARTIAL COVERAGE; †: CONCURRENT PREPRINT). THE DISTINGUISHING FEATURE IS THE JOINT OBSERVABILITY OF THE FULL CHAIN: ATTRIBUTED, TIME-STAMPED, POSSIBLY CONTRADICTORY PHYSICAL EVIDENCE IN; PER-PREFIX BELIEF REFERENCES AND SEPARATELY SCORED GROUNDING, CONTRADICTION, ACTION, AND EXPLANATION LAYERS OUT.
<table><tr><td></td><td>Belief-R BeliefTrack† [17]</td><td>[18]</td><td>[19]</td><td>BayesBench† ALFRED/TEACh/B-1K OpenEQA CREG [20], [21], [22]</td><td>[23]</td><td>(ours)</td></tr><tr><td>Evidence setting</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Sequential evidence with revision pressure</td><td></td><td></td><td></td><td></td><td></td><td>7</td></tr><tr><td>Reports attributed to distinct physical sources with reliability</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Per-report confidence, provenance, and timestamps</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Stale, delayed, and superseded evidence semantics</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Labeled contradictions (detection, localization, type)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Relation-dependent cases with counterfactual pairs</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Evaluation targets Probabilistic reference posterior at every evidence step</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Belief preservation / unnecessary updates penalized</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Entity grounding scored separately from belief quality</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Evidence support scored separately from exact policy compliance</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Evidence-grounded explanations scored</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Information-controlled representation ablations</td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Fifty scenarios are generated for each of seven regimes, each isolating one reasoning demand: justified revision (applicable evidence must reverse the top belief), belief preservation (weak counterevidence must not), relation association (a measures relation is required to connect sensor evidence to the target), contradiction onset (opposing same-epoch claims open an active contradiction), contradiction resolution (a correction supersedes a conflicting claim), stale/delayed evidence (older-epoch late arrivals are stale, whereas delayed current-epoch evidence applies), and source reliability (reliability changes the posterior relative to confidence-only fusion). Relation-association scenarios are built as counterfactual pairs that become observationally indistinguishable once relations are deleted, so any performance difference on them is attributable to relation content. Across all scenarios the suite exposes 1,150 scenario–prefix pairs.

Fig. 2 traces one scenario: four arrivals from three heterogeneous sources exercise preservation, distractor isolation, contradiction onset, and supersession-based resolution, and the reference posterior at each prefix is the scored target. An independent checker that reconstructs every prefix posterior in log-odds space, without importing the generator’s updater, agrees with the generator on all 1,150 prefixes.

![](images/8828288fe194c3efcd9db8facc66881bea8ef0a0d1c474f1ea23205a3a6e5cfd.jpg)  
Fig. 2. One contradiction-resolution scenario. Top: four attributed arrivals; the robot’s report contradicts the IoT sensor’s, and a delayed high-confidence correction supersedes it. Bottom: the normative reference posterior scored at every prefix, with the contradiction-active interval shaded.

## B. Expert Semantic-Plausibility Audit

Before any full model run, one independent expert reviewed a stratified sample of 35 scenario–prefix pairs (five per regime) under a blinded-then-reveal protocol: the reviewer recorded the expected top belief, probability direction, supporting and counterevidence, and contradiction status without access to the generated references, locked those answers, and only then compared them with the oracle. The first round accepted 29 of 35 cases; the six disagreements, concentrated in the relation-association, source-reliability, and stale/delayed regimes, traced to one specification issue (tied-posterior semantics), which was repaired at the source. The suite was regenerated and a second round accepted all 35 cases; no label was ever patched by hand. The audit is a semantic-plausibility check on the generator, not a measure of human consensus.

## C. Representation Conditions

The representation study separates serialization changes from targeted deletion of semantic or metadata content (Fig. 3). Canonical JSON (flat records with relation triples), structured list (typed records), and full graph (reported, observes, and semantic-relation edges) preserve the same scenario information, whereas graph without relations deliberately removes semantic-relation content; these four conditions are crossed with all models. Five factor ablations run on Qwen-7B only: flat edge-removed relations (relation content kept, graph serialization removed), no provenance, no timestamps, no confidence, and shuffled full graph (identical records in permuted order). The shuffled graph and the flat edge-removed condition are length-matched controls (length ratios 1.000 and 1.004 to the full graph). Two manipulations carry confounds that constrain interpretation and are stated here once: canonical JSON prompts average 0.843× the full-graph length, so serialization and prompt length covary; and the no-provenance condition also removes the source-reliability field that the normative likelihood consumes, so it is a joint manipulation. Remaining factor-level caveats are collected in Sec. VI-D and the appendix.

## IV. EVALUATION SETUP

## A. Model Panel and Runtime Protocol

The panel comprises Qwen2.5-1.5B-Instruct (Qwen-1.5B), Qwen2.5-7B-Instruct (Qwen-7B), Mistral-7B-Instruct-v0.3 (Mistral-7B), and Gemma-3-12B-IT (Gemma-12B), with fixed decoding (sampling, temperature 0.2, top-p 0.9), three generation seeds treated as repeated generations nested within the same 350 scenarios, and a 512-token output budget selected by a runtime pilot. Under the preregistered compute fallback, every model runs canonical JSON and Qwen-7B alone runs the complete representation and factor-ablation matrix. Qwen-1.5B and Mistral-7B, which cannot reliably ground their outputs in the required entity, are retained as a diagnostic tier so that their grounding failures are reported rather than filtered; Gemma runs on the three other principal representations are exploratory. Benchmark, references, prompts, metrics, and runtime configuration were frozen before analysis; selection details, budget sensitivity, and the freeze record are in the appendix.

## B. Evaluation Layers and Metrics

Each output layer is scored separately. Reliability: strict JSON validity and the rate of outputs recovered by offline repair; missing or malformed outputs score zero on categorical and F1 metrics and receive the worst-case trajectory distance. Grounding: whether the prediction binds to the required physical entity, with failures split into missing-entity and hallucinated-entity rates. Belief trajectory: the primary score is 1 − TV, one minus the mean total-variation distance between predicted and reference posteriors over all reference prefixes; strict prefix belief-state accuracy, belief-preservation accuracy, correct-revision rate, Brier distance to the normative posterior (agreement with the reference, not calibration), and update latency complete the family. A conditional variant that averages only over produced predictions is an exploratory, unpreregistered diagnostic; because it conditions on a nonrandom subset of prefixes it upper-bounds update quality. Evidence attribution: supporting- and counterevidence F1 against the reference sets. Contradiction handling: detection accuracy, localization F1 over the involved evidence identifiers (scored only when a contradiction is present), and type accuracy. Action selection (endpoint task, canonical-JSON systems): exact reference-policy compliance, action-family agreement, and an evidence-supported-action rate that asks whether the chosen action is justified by the graph evidence irrespective of the reference action; the support predicate is a scenariolevel function of the endpoint graph (active contradiction, maximum belief uncertainty, minimum report confidence) and on this suite accepts verify or inspect in 300 scenarios and only monitor in the 50 relation-association scenarios (Sec. V-D). No separate action-admissibility metric is reported: the references annotate a single reference action per scenario. Explanation grounding: coverage of the required evidence identifiers, i.e., citation coverage rather than faithfulness.

## C. Reference Systems and Transparent Baselines

Six transparent fusion baselines run on the identical suite: latest-observation, unweighted majority, confidence-weighted, recency-weighted, source-reliability-weighted, and a Dirichletstyle updater. These provide fully inspectable reference strategies against which neural belief trajectories can be compared. The deterministic rule pipeline (graph construction, contradiction detection, confidence-sum fusion, rule-based action selection) is reported as a transparent reference implementation: its action row measures agreement between this operational policy and the normative reference policy, which are different policies by design, and its belief row is the measured accuracy of its fusion rule against the normative labels.

## D. Statistical Analysis

The scenario is the independent unit; generation repeats are averaged within scenario before inference. Point estimates carry 95% scenario-clustered bootstrap confidence intervals (10,000 resamples). Paired continuous contrasts use two-sided sign-flip permutation tests (100,000 resamples), paired binary outcomes use exact McNemar tests on the scenario-level fullygrounded indicator, and Holm correction is applied within each preregistered family (eleven representation contrasts, including three regime-scoped versions, and two model contrasts, per metric). The exploratory Gemma comparison receives confidence intervals only. Regime-level analyses and evidenceprefix curves never treat prefixes as independent scenarios. All contrasts, metrics, exclusion rules, and seeds were preregistered before the full run; the conditional trajectory score and the Gemma non-canonical systems are the two labeled exceptions.

![](images/b939e9f13e308d94d7b87958913d1fac2a0d92e015ad34b73e4190fd4d502b83.jpg)  
Additional controlled factors (full graph, Qwen2.5-7B only): shuffled record order · flat edge-removed relations · no provenance · no timestamps · no confidence  
Fig. 3. Information control in the representation study, on one observation of a frozen relation-association scenario. Top: the source scenario’s attributed observation and measures relation. Conditions A–C preserve this content and differ only in serialization; D removes the measures relation, so relationdependent evidence cannot be associated with the scored target. The footer lists the factor ablations; confounds are stated in Sec. III-C.

## V. RESULTS

Table II reports the primary trajectory metrics for the panel and Table III the paired contrasts that support the three findings below, with scenario-clustered confidence intervals and Holmadjusted permutation p-values. Unless stated otherwise, differences quoted in this section are Holm-significant at p < 0.01; the complete contrast family is in the appendix.

## A. Entity Grounding Accounts for Much of the Weak-Model Deficit

Required-entity grounding under canonical JSON ranges from 0.694 (Mistral-7B) to 1.000 (Qwen-7B) across the four models (Fig. 4), and failures are almost entirely missingentity outputs: hallucinated-entity rates are at most 0.016 for any system. The two diagnostic-tier models emit mostly valid JSON (strict validity 0.807 for Qwen-1.5B and 0.897 for Mistral-7B) yet reach unconditional trajectory scores of only 0.397 and 0.239. Qwen-7B exceeds Qwen-1.5B by 0.401 trajectory-score points ([0.382, 0.419]) and Mistral-7B by 0.559 ([0.545, 0.572]), with strict-accuracy gaps of 0.499 and 0.609, and exact McNemar tests on the scenario-level fully-grounded indicator separate both pairs $( p < 1 0 ^ { - 9 8 } )$ .

The exploratory conditional decomposition locates much of this deficit before belief updating. Averaged only over prefixes with a grounded prediction, the diagnostic models score 0.747 (Qwen-1.5B) and 0.755 (Mistral-7B), within a few points of the core models, whose conditional and unconditional scores nearly coincide (Qwen-7B: 0.797 in both; Gemma-12B: 0.802 versus 0.801). Grounding also varies along the evidence sequence (Fig. 6b): Mistral’s grounded-prediction rate is 0.009 at the first prefix, 0.051 at the second, 0.860 at the third, and 0.470 at the fourth, whereas Qwen-7B and Gemma-12B remain at or near 1.0 throughout. Weak models therefore do not merely produce worse posterior updates; they frequently fail before belief updating can be assessed, because their outputs never bind to the required physical entity. Entity grounding thus accounts for a substantial fraction of the observed weak-model deficit. Because the conditional score is unpreregistered and conditions on a non-random, plausibly easier subset of prefixes, it upper-bounds update quality, and we do not conclude that grounding is the only source of the gap.

In the exploratory model comparison, Gemma-12B matches Qwen-7B on trajectory score (∆ = +0.004, [−0.005, +0.013]) but scores 0.101 lower on strict state accuracy ([−0.116, −0.087]); its predicted-versus-reference confidence-bin agreement is also lower (0.787 versus 0.905, Appendix).

## B. Semantic Relations Matter; Graph Serialization Does Not Consistently Help

Relation content is necessary for categorical belief accuracy in the relation-association regime, whose counterfactual pairs are indistinguishable without relations: deleting semantic relations reduces Qwen-7B’s strict belief-state accuracy there by 15.3 points ([0.098, 0.207], Holm-adjusted p < 0.001). The corresponding regime-scoped trajectory-score difference is smaller (+0.020, [0.002, 0.037]) and does not survive Holm correction (p = 0.058). Separately, deleting relations reduces the trajectory score by 0.088 over all regimes ([0.072, 0.103], Holm $p < 0 . 0 0 1 )$ . Relation content therefore matters where the task is constructed to require it, but not every relationsensitive metric registers the effect at the regime level.

TABLE II  
PRIMARY TRAJECTORY RESULTS (350 SCENARIOS; THREE SEEDS AVERAGED WITHIN SCENARIO; 95% SCENARIO-CLUSTERED BOOTSTRAP CIS) DIAGNOSTIC-TIER SYSTEMS RETAIN THEIR GROUNDING FAILURES. GEMMA NON-CANONICAL SYSTEMS ARE EXPLORATORY (APPENDIX).
<table><tr><td>System</td><td>Trajectory score [95% CI]</td><td>Strict state acc.</td><td>Grounding</td><td>Strict valid</td></tr><tr><td>Qwen-1.5B / JSON (diagnostic)</td><td>0.397 [0.380, 0.413]</td><td>0.331</td><td>0.786</td><td>0.807</td></tr><tr><td>Mistral-7B / JSON (diagnostic)</td><td>0.239 [0.229, 0.248]</td><td>0.221</td><td>0.694</td><td>0.897</td></tr><tr><td>Qwen-7B / JSON</td><td>0.797 [0.789, 0.806]</td><td>0.831</td><td>1.000</td><td>1.000</td></tr><tr><td>Gemma-12B / JSON</td><td>0.801 [0.794, 0.808]</td><td>0.729</td><td>1.000</td><td>0.999</td></tr><tr><td>Qwen-7B / structured list</td><td>0.842 [0.829, 0.854]</td><td>0.682</td><td>0.966</td><td>0.999</td></tr><tr><td>Qwen-7B / flat edge-removed</td><td>0.774 [0.760, 0.788]</td><td>0.605</td><td>0.917</td><td>0.999</td></tr><tr><td>Qwen-7B / graph w/o relations</td><td>0.727 [0.710, 0.743]</td><td>0.537</td><td>0.868</td><td>1.000</td></tr><tr><td>Qwen-7B / full graph</td><td>0.814 [0.808, 0.821]</td><td>0.695</td><td>1.000</td><td>1.000</td></tr><tr><td>Qwen-7B / graph, no provenance</td><td>0.831 [0.824, 0.838]</td><td>0.730</td><td>1.000</td><td>1.000</td></tr><tr><td>Qwen-7B / graph, no timestamps</td><td>0.825 [0.818, 0.831]</td><td>0.680</td><td>0.997</td><td>1.000</td></tr><tr><td>Qwen-7B / graph, no confidence</td><td>0.805 [0.798, 0.813]</td><td>0.716</td><td>1.000</td><td>1.000</td></tr><tr><td>Qwen-7B / shuffled graph</td><td>0.848 [0.840, 0.856]</td><td>0.741</td><td>0.999</td><td>1.000</td></tr></table>

TABLE III

PAIRED CONTRASTS SUPPORTING THE MAIN FINDINGS (QWEN-7B UNLESS A MODEL CONTRAST IS NAMED). ∆ IS THE MEAN PAIRED SCENARIO-LEVEL DIFFERENCE; REGIME-SCOPED ROWS RESTRICT TO THE NAMED REGIME; TWO-SIDED SIGN-FLIP PERMUTATION TESTS WITH HOLM CORRECTION WITHIN EACH PREREGISTERED FAMILY. THE GEMMA CONTRAST IS EXPLORATORY (CI ONLY).
<table><tr><td>Contrast (A — B)</td><td>Metric</td><td>∆</td><td></td><td>95% CI</td><td>Holm p</td></tr><tr><td>Relation content</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Full graph — graph w/o relations</td><td>traj. score</td><td>+0.088</td><td></td><td>[+0.072, +0.103]</td><td>0.0001</td></tr><tr><td>same, relation-association regime</td><td>traj. score</td><td>+0.020</td><td></td><td>[+0.002, +0.037]</td><td>0.0577</td></tr><tr><td>same, relation-association regime</td><td>strict acc.</td><td>+0.153</td><td></td><td>[+0.098, +0.207]</td><td>0.0001</td></tr><tr><td>Serialization and order</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Full graph — flat edge-removed</td><td>traj. score</td><td>+0.041</td><td></td><td>[+0.028, +0.053]</td><td>0.0001</td></tr><tr><td>Structured list — canonical JSON</td><td>traj. score</td><td>+0.045</td><td></td><td>[+0.033, +0.056]</td><td>0.0001</td></tr><tr><td>Shuffled graph — full graph</td><td>traj. score</td><td>+0.034</td><td></td><td>[+0.025, +0.042]</td><td>0.0001</td></tr><tr><td>Metadata fields (regime-scoped)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Full graph — no provenance, source-reliability regime</td><td>strict acc.</td><td>-0.200</td><td></td><td>[-0.233, -0.167]</td><td>0.0001</td></tr><tr><td>Full graph — no timestamps, stale/delayed regime</td><td>strict acc.</td><td>-0.003</td><td></td><td>[-0.033, +0.027]</td><td>0.9149</td></tr><tr><td>Models (canonical JSON)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen-7B — Qwen-1.5B</td><td>traj. score</td><td>+0.401</td><td></td><td>[+0.382, +0.419</td><td>&lt; 0.0001</td></tr><tr><td>Qwen-7B — Mistral-7B</td><td>traj. score</td><td>+0.559</td><td></td><td>[+0.545, +0.572]</td><td>&lt; 0.0001</td></tr><tr><td>Gemma-12B — Qwen-7B (exploratory)</td><td>traj. score</td><td>+0.004</td><td></td><td>[−0.005, +0.013]</td><td></td></tr></table>

![](images/834c66f6523e5ec738ab6e886d985530a16eda15c028720cf7eb8f81dbc34268.jpg)  
Fig. 4. Required-entity grounding rate, primary (unconditional) trajectory score, and exploratory conditional trajectory score (hatched; grounded predictions only, an upper bound) for the four models under canonical JSON.

Graph serialization itself shows no consistent advantage. The information-matched structured list scores 0.842 against 0.814 for the full graph (Table II), and the shuffled full graph— identical records in permuted order—scores 0.848, which is

0.034 above the ordered graph ([0.025, 0.042]). In the other direction, the flat edge-removed condition, which retains relation content without graph serialization, scores 0.041 below the full graph ([0.028, 0.053]), and canonical JSON trails the graph by 0.017. Representing the same facts as graph edges thus helps relative to one flat encoding, hurts relative to another, and record order carries no measurable value. We accordingly attribute no consistent advantage to graph serialization or to record order.

The two primary metrics rank serializations differently. Canonical JSON trails the graph on trajectory score (−0.017) but exceeds it on strict belief-state accuracy by 0.136 ([0.120, 0.151]), and exceeds the structured list on strict accuracy by 0.148 while trailing it on trajectory score by 0.045. Because canonical prompts are 0.843× the graph’s length for identical information, whereas the shuffled and edge-removed controls are length-matched, the canonical comparison confounds serialization with prompt length and is not assigned to format alone.

Among the field ablations (Fig. 5), removing provenance changes the overall trajectory score by +0.017 in favor of the ablated condition, the source-reliability-regime trajectory score by 0.019 in favor of the full graph, and strict accuracy on that regime by 0.200 in favor of the ablated condition. Removing timestamps yields a null result on the stale/delayed regime (strict accuracy $\Delta = - 0 . 0 0 3 , [ - 0 . 0 3 3 , 0 . 0 2 7 ] , p = 0 . 9 1 )$ and −0.010 overall in favor of the ablated condition; removing confidence costs 0.009. These small, direction-inconsistent effects do not support a clean per-field story (Sec. VI-B).

## C. Trajectory Evaluation Reveals Failures Hidden at the Endpoint

Belief-state accuracy is strongly non-monotonic along the evidence sequence (Fig. 6a). At the third evidence prefix— where most regimes introduce the report that tests the belief, such as the weak counterevidence that must not reverse it or the opposing claim that opens a contradiction (Fig. 2)— accuracy for the full-graph, relation-free-graph, and structuredlist conditions falls to between 0.20 and 0.25, then recovers to between 0.62 and 0.81 at the fourth prefix; canonical JSON reaches its minimum of 0.64 at the same prefix. Grounding shows the complementary pattern (Fig. 6b): Mistral’s grounded-prediction rate collapses on the first two prefixes and only partially recovers later.

A final-prefix metric would classify most of these trajectories as successful. It would credit the graph-family conditions with 0.62–0.81 accuracy while hiding a prefix at which three quarters of the beliefs were wrong, and it would credit Mistral with its best grounding rate while hiding that it almost never grounds on the initial-evidence situations where a monitoring system has least information. These failures concentrate at the intermediate prefix where the benchmark introduces the evidence that tests preservation, contradiction handling, or revision, rather than being distributed uniformly along the trajectory, and they are shared across serializations. Prefixlevel references are what make them measurable.

TABLE IV  
TRANSPARENT FUSION BASELINES (MAIN STRATEGIES; LATEST-OBSERVATION AND DIRICHLET IN THE APPENDIX).
<table><tr><td>Fusion baseline</td><td>Traj. score</td><td>Strict acc.</td><td>Preserv.</td><td>Revision</td></tr><tr><td>unweighted majority</td><td>0.680</td><td>0.631</td><td>0.333</td><td>0.655</td></tr><tr><td>confidence weighted</td><td>0.688</td><td>0.768</td><td>0.667</td><td>0.786</td></tr><tr><td>source reliability weighted</td><td>0.700</td><td>0.856</td><td>1.000</td><td>0.856</td></tr><tr><td>recency weighted</td><td>0.691</td><td>0.815</td><td>0.667</td><td>0.857</td></tr></table>

## D. Transparent Baselines and Secondary Diagnostics

Among the six fusion baselines (Table IV), source-reliability weighting scores highest at 0.700 and latest-observation lowest at 0.603; every baseline scores above both diagnostic-tier systems (0.397 and 0.239) and below every core neural system (0.727–0.848). The deterministic reference pipeline’s fusion rule attains 0.786 belief accuracy against the normative labels; its action policy selects the normative reference action on 125 of 350 scenarios (35.7%), matches the reference action family on 78.6%, and is evidence-supported on 100%.

On the endpoint action task, exact reference-policy compliance is 0.214, 0.213, 0.215, and 0.149 for Qwen-1.5B, Mistral-7B, Qwen-7B, and Gemma-12B, while the evidencesupported-action rate is 0.857 for all four. The identical rate is structural: the predicate accepts either verify or inspect in 300 of 350 scenarios and only monitor in the remaining 50, so it cannot distinguish among actions within the investigate family, which is why models with different exact reference-policy compliance (0.149–0.215) receive the same evidence-support score; the metric has limited discriminative value on this suite and is not evidence of equal action reasoning. Contradiction detection accuracy ranges from 0.416 (Qwen-1.5B) to 0.960 (Qwen-7B, no-confidence condition), and Qwen-7B’s full-graph localization F1 exceeds 0.95 on the contradiction-resolution and source-reliability regimes against 0.001 for Mistral-7B overall. Strict JSON validity is at or above 0.997 for every core system with repair rates below 0.3%. Contradiction-type accuracy, failure taxonomies, budget sensitivity, prompt lengths, confidence-bin agreement, the override diagnostic, and qualitative error examples are in the appendix.

## VI. DISCUSSION

## A. What CREG Reveals About Model Evaluation

The panel separates on entity grounding before it separates on belief updating: the diagnostic-tier models emit mostly valid JSON yet lose much of their trajectory score to predictions that never bind to the required entity, and in the exploratory conditional analysis their grounded updates land within a few points of the core models’. We therefore read the model contrasts of Sec. V-A as predominantly grounding gaps rather than pure reasoning gaps, subject to the upperbound caveat. Practically, output-side entity binding should be engineered and verified before poor performance is attributed to fusion or revision competence.

The prefix curves extend this message from grounding to belief. A mid-sequence collapse shared by every graph-family condition, an early-prefix grounding collapse in one model, and the difference between prompt and delayed revision are invisible to endpoint evaluation by construction, because a final answer aggregates over the dynamics one needs to inspect. Endpoint benchmarks remain useful for what they measure; they cannot show whether a system reached its answer through a defensible sequence of beliefs. We regard the prefix curves, rather than any scalar, as the primary scientific output of the trajectory track.

![](images/81fe72825028b601370e331ac4ffd32c927f47a803a84f00b8e50ee8b9297d63.jpg)  
Fig. 5. Trajectory score by representation condition and regime (Qwen-7B, nine conditions, common 0.5–1.0 scale). Effects vary by regime; the strong relation-specific categorical effect is reported in Table III, and the regime-scoped trajectory-score effect is small.

(a) accuracy by prefix — Qwen2.5-7B  
![](images/649ceb9316e49b826c17ef9df1881592430f32aa72f300ed10df75f4d6c4b076.jpg)

(b) grounding by prefix — canonical JSON  
![](images/8b37b486c4b7bb069a2f6a8f2b740d6065b3a56c282da15488ae25b26ea8da3b.jpg)  
Fig. 6. (a) Belief-state accuracy by evidence prefix for four Qwen-7B representation conditions. (b) Grounded-prediction rate by prefix for the four models under canonical JSON. The final prefix alone would hide the intermediate failures in both panels.

## B. What the Representation Study Supports

Relation semantics matter; graph form by itself does not. The relation effect is categorical: large and Holm-significant on strict belief-state accuracy in the regime built to require relations, small and not significant after correction on the regime-scoped trajectory score, so the two metrics register the manipulation differently. Part of the overall deficit persists when relation content is kept in flat form, so content and serialization both contribute; but the information-matched structured list outperforms the full graph and permuting record order slightly improves it, so neither graph serialization nor ordering carries measurable value here. The study manipulates serialization, relation content, record order, and selected metadata fields; it does not test arbitrary graph topologies.

Two features bound this conclusion. The two primary metrics dissociate—canonical JSON yields more exactly correct categorical states while richer serializations track the reference distribution more closely—which we attribute tentatively to shorter prompts favoring decisive commitments, without assigning the difference to format alone given the length confound. And the field ablations resist a clean story: the provenance manipulation is confounded by design and its regime-scoped effects point in opposite directions on the two metrics, the timestamp null is expected because state epochs remain visible, and no dedicated source-reliability ablation exists. These are boundaries on the evidence that the information controls make visible.

## C. Action Metrics and Transparent Baselines

All four models agree with the single reference action about a fifth of the time, and the transparent reference pipeline agrees with the normative policy on roughly a third of scenarios while remaining evidence-supported on all of them; exact compliance thus measures agreement with one normative reference policy, not decision quality, and the coarse evidence-support rate (Sec. V-D) cannot rank systems within the investigate family. The transparent fusion baselines are informative in both directions: the best inspectable strategy trails the best neural systems by roughly 0.15 trajectory-score points, so simple fusion does not solve the task, while every baseline outscores both diagnostic-tier models. Thus much of the weak-model deficit arises before distributional updating can be evaluated, through ungrounded or malformed outputs; this does not show that their updating is adequate, since the conditional analysis selects a non-random subset of prefixes. That reliability weighting is the best transparent strategy follows from the reference semantics and is not an independent discovery.

## D. Limitations and Outlook

Five limitations bound our claims: the suite is synthetic, with a binary state space and one scored target entity, and the physical-log replay in the appendix demonstrates the ingestion pathway rather than an empirical robot evaluation; the semantic-plausibility audit used one expert and does not establish human consensus; two representation ablations are confounded as disclosed in Sec. III-C, the timestamp manipulation is intentionally weak, and the source-reliability factor lacks a dedicated ablation; the conditional trajectory score and Gemma’s non-canonical runs are exploratory, and no conclusion rests on them; and the panel spans four openweight models between 1.5B and 12B parameters at a fixed 512-token budget, so larger models, tool-augmented systems, and longer budgets are untested.

The override diagnostic in the appendix shows that a verifier which always overrides the model discards many evidencesupported proposals (harmful-override rate 0.494), which motivates selective neurosymbolic verification that accepts, abstains, or overrides per decision with calibrated triggers. Richer multi-entity state spaces, longer recorded robotic sequences through the replay pathway, and a dedicated source-reliability manipulation would complete the factor set this benchmark was designed to expose.

## VII. CONCLUSION

We introduced CREG, a controlled benchmark and layered evaluation protocol for evidence-grounded shared belief updating in Connected Robotics. Across 350 scenarios, four open-weight instruction models, and six transparent fusion baselines, the results show that weak-model performance is strongly affected by entity-grounding failures that occur before belief updating can be judged; that semantic relation content improves categorical belief accuracy where relational evidence is required, while graph serialization alone offers no consistent advantage once information content is controlled; and that final-state evaluation hides substantial transient errors along the belief trajectory.

These findings support evaluating connected-robot reasoning as a layered evidence-to-belief process rather than as a single final decision. By separating grounding, revision, contradiction handling, and action selection, and by exposing a reference belief at every evidence prefix, CREG provides a reproducible basis for studying shared semantic world models, distributed physical reasoning, and the selective decision verification that these results motivate.

## ACKNOWLEDGMENT

This work was supported by JSPS KAKENHI Grant Number JP26K17338. We gratefully acknowledge the support of NVIDIA Corporation.

## REFERENCES

[1] R. Bezerra, S. Tadokoro, and K. Ohno, “AI–IoT–Robotics integration: Survey of frameworks, emerging trends, and the path toward connected robotics,” IEEE Internet of Things Journal, vol. 13, no. 10, pp. 20 398– 20 412, 2026.

[2] A. Rosinol, A. Gupta, M. Abate, J. Shi, and L. Carlone, “3D dynamic scene graphs: Actionable spatial perception with places, objects, and humans,” in Robotics: Science and Systems XVI. Robotics: Science and Systems Foundation, 2020. [Online]. Available: https://www.roboticsproceedings.org/rss16/p079.html

[3] A. Rosinol et al., “Kimera: From SLAM to spatial perception with 3D dynamic scene graphs,” The International Journal of Robotics Research, vol. 40, no. 12–14, pp. 1510–1546, 2021.

[4] N. Hughes, Y. Chang, and L. Carlone, “Hydra: A real-time spatial perception system for 3D scene graph construction and optimization,” in Robotics: Science and Systems XVIII. New York City, NY, USA: Robotics: Science and Systems Foundation, 2022. [Online]. Available: https://www.roboticsproceedings.org/rss18/p050.html

[5] Y. Chang, N. Hughes, A. Ray, and L. Carlone, “Hydra-multi: Collaborative online construction of 3D scene graphs with multi-robot teams,” in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023, pp. 10 995–11 002.

[6] Q. Gu et al., “Conceptgraphs: Open-vocabulary 3D scene graphs for perception and planning,” in 2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024, pp. 5021–5028.

[7] A. Werby, C. Huang, M. Büchner, A. Valada, and W. Burgard, “Hierarchical open-vocabulary 3D scene graphs for language-grounded robot navigation,” in Robotics: Science and Systems XX. Delft, Netherlands: Robotics: Science and Systems Foundation, 2024. [Online]. Available: https://www.roboticsproceedings.org/rss20/p077.html

[8] M. F. Ginting et al., “Safe and efficient navigation in extreme environments using semantic belief graphs,” in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 5653–5658.

[9] M. F. Ginting, D. D. Fan, S.-K. Kim, M. J. Kochenderfer, and A.-A. Agha-Mohammadi, “Semantic belief behavior graph: Enabling autonomous robot inspection in unknown environments,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 7604–7610.

[10] S. Kojima et al., “Heterogeneous robots coordination for industrial plant inspection and evaluation at world robot summit 2020,” Advanced Robotics, vol. 36, no. 21, pp. 1102–1119, 2022.

[11] A. P. Dempster, “Upper and lower probabilities induced by a multivalued mapping,” The Annals of Mathematical Statistics, vol. 38, no. 2, pp. 325–339, 1967.

[12] S. J. Julier and J. K. Uhlmann, “A non-divergent estimation algorithm in the presence of unknown correlations,” in Proceedings of the 1997 American Control Conference, vol. 4. IEEE, 1997, pp. 2369–2373.

[13] X. L. Dong, L. Berti-Équille, and D. Srivastava, “Integrating conflicting data: The role of source dependence,” Proceedings of the VLDB Endowment, vol. 2, no. 1, pp. 550–561, 2009.

[14] Y. Li et al., “A survey on truth discovery,” ACM SIGKDD Explorations Newsletter, vol. 17, no. 2, pp. 1–16, 2016.

[15] C. E. Alchourrón, P. Gärdenfors, and D. Makinson, “On the logic of theory change: Partial meet contraction and revision functions,” The Journal of Symbolic Logic, vol. 50, no. 2, pp. 510–530, 1985.

[16] A. Darwiche and J. Pearl, “On the logic of iterated belief revision,” Artificial Intelligence, vol. 89, no. 1–2, pp. 1–29, 1997.

[17] B. Wilie, S. Cahyawijaya, E. Ishii, J. He, and P. Fung, “Belief revision: The adaptability of large language models reasoning,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. Miami, Florida, USA: Association for Computational Linguistics, 2024, pp. 10 480–10 496. [Online]. Available: https://aclanthology.org/2024.emnlp-main.586/

[18] H. Xu et al., “When should models change their minds? contextual belief management in large language models,” 2026, arXiv preprint; accepted to EMNLP 2026 (main conference) per the arXiv listing. [Online]. Available: https://arxiv.org/abs/2605.30219

[19] A. Samanta et al., “BayesBench: Evaluating LLM belief trajectories under multi-turn evidence accumulation,” 2026, preprint. [Online]. Available: https://arxiv.org/abs/2606.30850

[20] M. Shridhar et al., “ALFRED: A benchmark for interpreting grounded instructions for everyday tasks,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020, pp. 10 737–10 746.

[21] A. Padmakumar et al., “TEACh: Task-driven embodied agents that chat,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 36, no. 2, pp. 2017–2025, 2022.

[22] C. Li et al., “BEHAVIOR-1K: A benchmark for embodied AI with 1,000 everyday activities and realistic simulation,” in Proceedings of the 6th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 205. PMLR, 2023, pp. 80–93. [Online]. Available: https://proceedings.mlr.press/v205/li23a.html

[23] A. Majumdar et al., “OpenEQA: Embodied question answering in the era of foundation models,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 16 488– 16 498.

[24] B. Ichter et al., “Do as i can, not as i say: Grounding language in robotic affordances,” in Proceedings of the 6th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 205. PMLR, 2023, pp. 287–318. [Online]. Available: https: //proceedings.mlr.press/v205/ichter23a.html

[25] D. Driess et al., “PaLM-E: An embodied multimodal language model,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 202. PMLR, 2023, pp. 8469–8488. [Online]. Available: https://proceedings.mlr.press/v202/driess23a.html

[26] B. Zitkovich et al., “RT-2: Vision-language-action models transfer web knowledge to robotic control,” in Proceedings of the 7th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 229. PMLR, 2023, pp. 2165–2183. [Online]. Available: https://proceedings.mlr.press/v229/zitkovich23a.html

[27] K. Rana, J. Haviland, S. Garg, J. Abou-Chakra, I. Reid, and N. Suenderhauf, “Sayplan: Grounding large language models using 3D scene graphs for scalable robot task planning,” in Proceedings of the 7th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, vol. 229. PMLR, 2023, pp. 23–72. [Online]. Available: https://proceedings.mlr.press/v229/rana23a.html

[28] J. Sun et al., “Think-on-graph: Deep and responsible reasoning of large language model on knowledge graph,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=nnVO1PvbTv

[29] X. He et al., “G-Retriever: Retrieval-augmented generation for textual graph understanding and question answering,” in Advances in Neural Information Processing Systems, vol. 37. Curran Associates, Inc., 2024, pp. 132 876–132 907. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2024/hash/ efaf1c9726648c8ba363a5c927440529-Abstract-Conference.html

[30] J. DeYoung et al., “ERASER: A benchmark to evaluate rationalized NLP models,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Online: Association for Computational Linguistics, 2020, pp. 4443–4458. [Online]. Available: https://aclanthology.org/2020.acl-main.408/

[31] A. Jacovi and Y. Goldberg, “Towards faithfully interpretable NLP systems: How should we define and evaluate faithfulness?” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Online: Association for Computational Linguistics, 2020, pp. 4198–4205. [Online]. Available: https://aclanthology.org/2020.acl-main.386/

This appendix collects the operational, reproducibility, and secondary-analysis material that supports the main text. No result reported here changes any conclusion of the main text.

## APPENDIX A RUNTIME PROTOCOL AND FREEZE RECORD

## A. Model Panel

The preregistered panel comprises Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-7B-Instruct,

mistralai/Mistral-7B-Instruct-v0.3, and google/gemma-3-12b-it, which the preregistration listed as optional pending access and which was run in full under canonical JSON. Decoding is fixed for all systems: sampling enabled, temperature 0.2, top-p 0.9. Three generation seeds (42, 43, 44) are treated as repeated generations nested within the same 350 scenarios and are averaged within scenario before any inference. The frozen benchmark manifest records the model identifiers, representation settings, prompt hashes, and per-scenario content hashes for every system; the model\_revision and tokenizer\_revision fields in the manifest are unset, so model snapshots are identified by name rather than by commit hash (see Sec. I).

## B. Token-Budget Selection

A seven-scenario runtime pilot (one scenario per regime, seed 42) evaluated the preregistered output-budget candidates {256, 384, 512, 768} on the twelve-system runtime-fallback panel. No budget produced context overflow or truncation. The 512-token budget gave the highest strict-validity and pooled entity-grounding rates with a projected full-run cost of 45.8 h, below the preregistered 48 h cap (Table A1). Because the full model×representation crossing exceeded the cap, the preregistered fallback applied: every model runs canonical JSON, and Qwen2.5-7B alone runs the complete representation and factor-ablation matrix (twelve systems in total). Mean trajectory score on the pilot varies by less than 0.02 across the tested budgets, so the budget choice does not drive the reported conclusions within the tested range; budgets above 512 tokens were not run at full scale.

## C. Diagnostic Tier

The preregistered runtime gate required a pooled requiredentity grounding rate at or above a fixed threshold. The pilot showed that two systems—Qwen2.5-1.5B and Mistral-7B under canonical JSON—would fail that gate for reasons that are a model-capability result rather than an infrastructure problem. A preregistration amendment (evaluation-v1.0-draft2, 2026-07-27, adopted before the freeze) therefore tiered the gate: it is evaluated over core systems only, and the two systems are retained in the matrix as a diagnostic tier whose grounding failures are reported in the main text rather than pooled into runtime-compatibility acceptance. No scenario, reference, prompt, or metric changed under the amendment.

TABLE A1  
TOKEN-BUDGET SENSITIVITY ON THE SEVEN-SCENARIO PILOT (SEED 42, TWELVE-SYSTEM PANEL). THE 512 CHOICE USED THE POOLED GROUNDING RATE; THE CORE-SCOPED RATE IS MARGINALLY HIGHEST AT 256.
<table><tr><td></td><td>Budget Strict valid</td><td>Repair</td><td>Grounding (pooled)</td><td>Grounding (core)</td><td></td><td>Truncated Mean traj. Proj. hours</td><td></td></tr><tr><td>256</td><td>0.928</td><td>0.021</td><td>0.928</td><td>0.997</td><td>0.000</td><td>0.572</td><td>39.4</td></tr><tr><td>384</td><td>0.954</td><td>0.000</td><td>0.912</td><td>0.990</td><td>0.000</td><td>0.573</td><td>43.6</td></tr><tr><td>512</td><td>0.961</td><td>0.003</td><td>0.930</td><td>0.986</td><td>0.000</td><td>0.589</td><td>45.8</td></tr></table>

TABLE A2

TRAJECTORY-TASK PROMPT LENGTHS BY CONDITION (SEED 42; PROMPTS ARE SEED-INVARIANT).
<table><tr><td>Condition</td><td>Mean chars</td><td>Max chars</td><td>Rel. to full graph</td></tr><tr><td>canonical_json</td><td>5284</td><td>6090</td><td>0.843</td></tr><tr><td>structured_list</td><td>5651</td><td>6513</td><td>0.901</td></tr><tr><td>flat-edge-removed-relations</td><td>6298</td><td>7657</td><td>1.004</td></tr><tr><td>graph_without_relations</td><td>6058</td><td>7430</td><td>0.966</td></tr><tr><td>graph</td><td>6271</td><td>7624</td><td>1.000</td></tr><tr><td>no-provenance</td><td>5834</td><td>6821</td><td>0.930</td></tr><tr><td>no-timestamps</td><td>6053</td><td>7224</td><td>0.965</td></tr><tr><td>no-confidence</td><td>6210</td><td>7511</td><td>0.990</td></tr><tr><td>shuffled-full-graph</td><td>6271</td><td>7624</td><td>1.000</td></tr></table>

## D. Freeze Gates

The benchmark was frozen under the annotated tag evaluation-v1.0 after seven gates passed: 350 unique scenario identifiers and semantic fingerprints; agreement of the independent log-odds reference checker on every prefix; verification that all relation-association counterfactual pairs are observationally indistinguishable without relations; completion of the 35-case expert audit without unresolved corrections; the runtime pilot meeting all thresholds; the full repository test suite; and the task output-contract tests. Post-freeze corrections are restricted to the report layer (metric aggregation and comparison code) and are logged in the repository changelog; no frozen artifact was modified. Gemma runs on the structuredlist, graph-without-relations, and full-graph representations were collected outside the preregistered fallback panel and are reported only in this document, as exploratory results.

## APPENDIX B

## REPRESENTATION CONDITIONS: ADDITIONAL DETAIL

Fig. A1 shows the same observation record exactly as the four principal conditions serialize it, excerpted verbatim from the frozen prompts. Table A2 quantifies prompt length for every condition.

Beyond the two confounds stated in the main text (canonical JSON is 0.843× the full-graph length; the no-provenance condition also removes the source-reliability field), the following caveats apply to the factor ablations. The no-timestamps condition removes event and arrival times but retains explicit state epochs, so temporal contradiction and supersession semantics remain recoverable; the manipulation is intentionally weak and a null result on the stale/delayed regime is the expected outcome. No no-source-reliability ablation exists; the sourcereliability regime, which is constructed so that reliability changes the posterior relative to confidence-only fusion, is the available probe for that factor, and the corresponding preregistered hypothesis is only partially testable. The shuffled full graph uses a fixed shuffle seed (1729) and is characteridentical to the full graph up to record order. The flat edgeremoved condition preserves relation content as flat records and removes only the graph edge serialization; its length ratio to the full graph is 1.004.

## APPENDIX C

## COMPLETE PREREGISTERED CONTRAST FAMILY

Table A3 lists every preregistered paired contrast, including the overall (non-regime-scoped) metadata-field contrasts omitted from the main text. Table A4 gives the regime composition of the suite.

## APPENDIX D

## TRANSPARENT BASELINES AND THE DETERMINISTICREFERENCE

Six transparent fusion baselines run on the identical suite. Table IV reports the four main strategies; the two remaining strategies score 0.603 (latest-observation) and 0.678 (Dirichletstyle updater) on trajectory score. Every fusion baseline scores above both diagnostic-tier neural systems and below every core neural system.

The deterministic rule pipeline (graph construction, contradiction detection, confidence-sum fusion, rule-based action selection) is reported as a transparent reference implementation. Its fusion rule attains 0.786 belief accuracy against the normative labels. Its action policy selects the normative reference action on 125 of 350 scenarios (35.7%), matches the reference action family on 78.6%, and selects an evidencesupported action on 100%. The 35.7% figure measures disagreement between two different policies by design and is not a competence score.

A post-hoc validation configuration in which the deterministic policy overrides model proposals (from Qwen2.5-7B under canonical JSON) is retained as a secondary diagnostic (Table A5). Its final action never depends on the proposal, so it is override accounting rather than selective proposethen-verify. The harmful-override rate of 0.494 indicates that

One observation (obs-1), as each information-controlled condition serializes it

canonical JSON (flat records, relation triples kept)   
{   
"id": "obs-1",   
"source": "iot-sensor-1",   
"entity\_id": "gauge-1",   
"attribute": "condition",   
"value": "normal",   
"confidence": 0.83,   
"event\_time": "2026-01-01T12:00:10+00:00",   
"arrival\_time": "2026-01-01T12:00:10+00:00",   
"state\_epoch": 1,   
"supersedes\_evidence\_ids": []   
}   
"relations": [

```jsonl
"edges": [
{"edge_type": "reported", "predicate": "reported",
"source_id": "iot-sensor-1", "target_id": "obs-1"}
{"edge_type": "observes", "predicate": "observes",
"source_id": "obs-1", "target_id": "gauge-1"}
{"confidence": 0.8, "edge_type": "semantic_relation",
"evidence_ids": [], "predicate": "adjacent_to",
"source_id": "gauge-1", "target_id": "gauge-2"}
...]
```

structured list (same facts, typed records)   
{   
"record\_type": "evidence",   
"id": "obs-1",   
"source": "iot-sensor-1",   
"entity\_id": "gauge-1",   
"attribute": "condition",   
"value": "normal",   
"confidence": 0.83,   
"state\_epoch": 1   
<sup>}</sup><sub>{</sub>   
"confidence": 0.8,   
"evidence\_ids": [],

graph without relations (relation content deleted)   
"edges": [   
{"edge\_type": "reported", "predicate": "reported",   
"source\_id": "iot-sensor-1", "target\_id": "obs-1"}   
{"edge\_type": "observes", "predicate": "observes",   
"source\_id": "obs-1", "target\_id": "gauge-1"}   
...]   
// no semantic\_relation edges: relation   
// content deleted, so relation-dependent   
// scenarios become observationally ambiguous  
Fig. A1. One observation from the worked-example scenario of the main text (Fig. 2), excerpted verbatim from the frozen prompts of the four principal representation conditions.

TABLE A3  
ALL PREREGISTERED PAIRED CONTRASTS (TRAJECTORY SCORE UNLESS MARKED; REGIME-SCOPED ROWS RESTRICT TO THE NAMED REGIME). ∆ IS THEMEAN PAIRED SCENARIO-LEVEL DIFFERENCE (SYSTEM A − SYSTEM B); p FROM TWO-SIDED SIGN-FLIP PERMUTATION TESTS; HOLM CORRECTIONWITHIN EACH FAMILY. THE GEMMA CONTRAST IS EXPLORATORY (CI ONLY).
<table><tr><td>Contrast</td><td>Metric</td><td>∆</td><td>95% CI</td><td></td><td>p Holm p</td></tr><tr><td>graph_vs_graph_without_relations</td><td>trajectory_score</td><td>+0.088</td><td>[+0.072, +0.103]</td><td>0.0000</td><td>0.0001</td></tr><tr><td>graph_vs_graph_without_relations @ relation_association</td><td>trajectory_score</td><td>+0.020</td><td>[+0.002, +0.037]</td><td>0.0288</td><td>0.0577</td></tr><tr><td>graph_vs_edge_removed</td><td>trajectory_score</td><td>+0.041</td><td>[+0.028, +0.053]</td><td>0.0000</td><td>0.0001</td></tr><tr><td>graph_vs_canonical_json</td><td>trajectory_score</td><td>+0.017</td><td>[+0.010, +0.024]</td><td>0.0000</td><td>0.0001</td></tr><tr><td>structured_list_vs_canonical_json</td><td>trajectory_score</td><td>+0.045</td><td>[+0.033, +0.056]</td><td>0.0000</td><td>0.0001</td></tr><tr><td>shuffled_vs_graph</td><td>trajectory_score</td><td>+0.034</td><td>[+0.025, +0.042]</td><td>0.0000</td><td>0.0001</td></tr><tr><td>graph_vs_no_provenance</td><td>trajectory_score</td><td>-0.017</td><td>[-0.023, -0.011]</td><td>0.0000</td><td>0.0001</td></tr><tr><td>graph_vs_no_provenance @ source_reliability</td><td>trajectory_score</td><td>+0.019</td><td>[+0.013, +0.025]</td><td>0.0000</td><td>0.0001</td></tr><tr><td>graph_vs_no_timestamps</td><td>trajectory_score</td><td>-0.010</td><td>[-0.015, -0.006]</td><td>0.0000</td><td>0.0001</td></tr><tr><td>graph_vs_no_timestamps @ stale_delayed_evidence</td><td>trajectory_score</td><td>+0.007</td><td>[+0.000, +0.014]</td><td>0.0494</td><td>0.0577</td></tr><tr><td>graph_vs_no_confidence</td><td>trajectory_score</td><td>+0.009</td><td>[+0.005, +0.013]</td><td>0.0000</td><td>0.0001</td></tr><tr><td>graph_vs_graph_without_relations @ relation_association</td><td>strict_prefix_belief_state_accuracy</td><td>+0.153</td><td>[+0.098, +0.207]</td><td>0.0000</td><td>0.0001</td></tr><tr><td>graph_vs_no_provenance @ source_reliability</td><td>strict_prefix_belief_state_accuracy</td><td>-0.200</td><td>[-0.233, -0.167]</td><td>0.0000</td><td>0.0001</td></tr><tr><td>graph_vs_no_timestamps @ stale_delayed_evidence</td><td>strict_prefix_belief_state_accuracy</td><td>-0.003</td><td>[-0.033, +0.027]</td><td>0.9149</td><td>0.9149</td></tr><tr><td>qwen7b_vs_qwen1.5b_canonical</td><td>trajectory_score</td><td>+0.401</td><td>[+0.382, +0.419]</td><td>0.0000</td><td>0.0000</td></tr><tr><td>qwen7b_vs_mistral7b_canonical</td><td>trajectory_score</td><td>+0.559</td><td>[+0.545, +0.572]</td><td>0.0000</td><td>0.0000</td></tr><tr><td>gemma12b_vs_qwen7b_canonical (expl.)</td><td>trajectory_score</td><td>+0.004</td><td>[-0.005, +0.013]</td><td></td><td></td></tr></table>

an always-override verifier discards many evidence-supported proposals; selective accept/abstain/override verification is left to subsequent work.

## APPENDIX E SECONDARY DIAGNOSTICS

## A. Contradiction Handling, Actions, and Reliability

On the trajectory track, contradiction detection accuracy ranges from 0.416 (Qwen2.5-1.5B) to 0.960 (Qwen2.5-7B, no-confidence condition). Localization F1 for the Qwen2.5- 7B full graph is 0.953 on contradiction-resolution scenarios and 0.993 on source-reliability scenarios, against 0.001 for Mistral-7B overall. Contradiction-type accuracy is sparse because explicit negatives receive no empty-set credit and few systems emit a type label consistently. On the endpoint action task (canonical-JSON systems), exact referencepolicy compliance is 0.214 (Qwen2.5-1.5B), 0.213 (Mistral-7B), 0.215 (Qwen2.5-7B), and 0.149 (Gemma-3-12B); the evidence-supported-action rate is 0.857 for all four. Strict JSON validity is at or above 0.997 for every core system with repair rates below 0.3%; the diagnostic-tier systems reach 0.807 and 0.897.

TABLE A4  
REGIME COMPOSITION OF THE FROZEN SUITE.
<table><tr><td>Regime</td><td>Constraint</td><td>Scenarios</td></tr><tr><td>justified revision</td><td>applicable evidence reverses the top belief</td><td>50</td></tr><tr><td>belief preservation</td><td>weak counterevidence does not reverse the top belief</td><td>50</td></tr><tr><td>relation association</td><td>measures relation is necessary to associate sensor evidence with the target</td><td>50</td></tr><tr><td>contradiction onset</td><td>opposing claims in one epoch create an active contra- diction</td><td>50</td></tr><tr><td>contradiction resolution</td><td>correction evidence supersedes a conflicting claim</td><td>50</td></tr><tr><td>stale delayed evidence</td><td>older-epoch late evidence is stale while delayed current-epoch evidence applies</td><td>50</td></tr><tr><td>source reliability</td><td>source reliability changes the posterior relative to confidence-only fusion</td><td>50</td></tr><tr><td colspan="3">Total (3 generation seeds, 512-token budget)</td></tr></table>

TABLE A5

DETERMINISTIC-POLICY OVERRIDE ACCOUNTING (SECONDARY DIAGNOSTIC; MEAN ± STD OVER THREE SEEDS). FINAL POLICY ACCURACY EQUALS THE DETERMINISTIC REFERENCE ROW BY CONSTRUCTION.
<table><tr><td>Diagnostic</td><td>Mean ± std (3 seeds)</td></tr><tr><td>Final policy accuracy (= deterministic row; validated action)</td><td> $0 . 3 5 7 \pm 0 . 0 0 0$ </td></tr><tr><td>Override rate</td><td> $0 . 6 3 7 \pm 0 . 0 0 3$ </td></tr><tr><td>Correct-override rate (unjustified proposal corrected)</td><td> $0 . 1 4 3 \pm 0 . 0 0 0$ </td></tr><tr><td>Harmful-override rate (defensible proposal overridden)</td><td> $0 . 4 9 4 \pm 0 . 0 0 3$ </td></tr><tr><td>Proposals accepted without correction</td><td> $0 . 3 6 3 \pm 0 . 0 0 3$ </td></tr><tr><td>Explanation coverage (model-side) Latency per decision (s)</td><td> $0 . 3 5 4 \pm 0 . 0 1 6$ </td></tr><tr><td></td><td> $1 . 7 8 0 \pm 0 . 0 0 1$ </td></tr></table>

## B. Confidence-Bin Agreement

Table A6 reports ordinal agreement between predicted and reference top-state probability bins over graded (non-tied), grounded prefixes. This is agreement with the normative reference, not empirical calibration. Under canonical JSON, Gemma-3-12B’s bin agreement is 0.787 versus 0.905 for Qwen2.5-7B.

## C. Failure Taxonomy

Table A7 counts scored failures by category for every system, pooled over regimes and generation seeds. Categories are premature revision, failure to revise, stale-evidence dominance, unreliable-source dominance, contradiction omission, incorrect attribution, relation failure, malformed output, and miscalibration (high Brier distance).

## APPENDIX F

## EXPERT AUDIT PROTOCOL

One independent expert reviewed a deterministic, stratified sample of 35 scenario–prefix pairs (five per regime). For each case the reviewer recorded, without access to the generated references, the expected top belief, probability direction, supporting- and counterevidence sets, and contradiction status; locked those answers; and only then compared them with the oracle. Round one accepted 29 of 35 cases. The six disagreements, in the relation-association, source-reliability, and stale/delayed regimes, traced to one specification-level issue (tied-posterior semantics), which was repaired in the normative specification; the suite was regenerated and round two accepted all 35 cases with no unresolved issues. No label was patched manually at any point. The audit is a semanticplausibility sanity check on the generator and specification;

it does not establish human consensus, human belief-revision behavior, or model–human agreement, and no inter-rater statistics are reported.

## APPENDIX G

## PHYSICAL-LOG REPLAY DEMONSTRATION

The toolkit includes an offline pathway from recorded observation logs to the same evidence-graph, contradiction, belief, decision, and explanation outputs used in the benchmark. The repository’s sample testbed (distributed\_gauge\_inspection) contains a twoobservation recorded log; the replay therefore demonstrates that the pathway works end to end and is not an empirical robot evaluation. In stepwise replay, the first observation (robot-1 reports gauge-1 normal, confidence 0.82) yields no contradiction and a monitor decision; the second (robot-1 reports abnormal, confidence 0.77) raises a contradiction between the two observations (severity 0.631) and escalates the decision to verify. The final fused belief is normal 0.516 / abnormal 0.484, a near-tie that reflects two conflicting, similarly confident observations. All metadata fields carried by the physical schema (pose, frame identifier, image path, modality, run and sequence identifiers) survive conversion and appear in the contradiction records, so replayed logs retain full provenance for audit.

## APPENDIX H

## QUALITATIVE ERROR EXAMPLES

The following excerpts (generation seed 42) illustrate the main failure categories. Outputs are truncated.

a) Ungrounded missing prediction (Mistral-7B, canonical JSON, justified-revision scenario, prefix 1): The model wraps its answer in prose (“Based on the provided structured semantic evidence, the shared belief after this arrival step can be represented as follows: . . . ”) and emits a distribution over a single state ("normal": 0.76) without the required entity binding; the output is parsed but scores as no prediction for the target entity.

b) Premature revision (Qwen2.5-7B, canonical JSON, belief-preservation scenario, prefix 3): On the arrival of weak counterevidence the model shifts the top state to anomalous with probability 0.53, citing only the new report as supporting evidence; the reference preserves the prior top state.

c) Stale-evidence dominance (Qwen2.5-7B, notimestamps condition, stale/delayed scenario, prefix 4): With event and arrival times removed, the model treats an older-epoch late arrival as current, returns a tied distribution (0.5/0.5, undetermined), and lists both the stale report and the current one as supporting evidence; the reference applies only the current-epoch report.

## APPENDIX I

## REMAINING REPRODUCIBILITY NOTES

Two items are recorded for completeness. First, the frozen manifest identifies models by repository name; the revisionhash fields are unset, so exact reproduction depends on the model snapshots available under those names at run time.

TABLE A6  
ORDINAL CONFIDENCE-BIN AGREEMENT BETWEEN PREDICTED AND REFERENCE TOP-STATE PROBABILITIES (GROUNDED, NON-TIED PREFIXES). GEMMA NON-CANONICAL ROWS ARE EXPLORATORY.
<table><tr><td>System</td><td>Prefixes</td><td>Bin agreement</td><td>Pred. high-bin</td><td>Ref. high-bin</td></tr><tr><td>gemma-3-12b-it-canonical_json</td><td>3297</td><td>0.787</td><td>0.730</td><td>0.711</td></tr><tr><td>gemma-3-12b-it-graph</td><td>3276</td><td>0.885</td><td>0.764</td><td>0.693</td></tr><tr><td>gemma-3-12b-it-graph_without_relations</td><td>3249</td><td>0.864</td><td>0.758</td><td>0.692</td></tr><tr><td>gemma-3-12b-it-structured_list</td><td>3271</td><td>0.874</td><td>0.688</td><td>0.654</td></tr><tr><td>mistral-7b-v0.3-canonical_json</td><td>1054</td><td>0.710</td><td>0.310</td><td>0.226</td></tr><tr><td>qwen2.5-1.5b-canonical_json</td><td>1727</td><td>0.768</td><td>0.660</td><td>0.540</td></tr><tr><td>qwen2.5-7b-canonical_json</td><td>3300</td><td>0.905</td><td>0.907</td><td>0.856</td></tr><tr><td>qwen2.5-7b-flat-edge-removed-relations</td><td>3012</td><td>0.995</td><td>0.600</td><td>0.600</td></tr><tr><td>qwen2.5-7b-graph</td><td>3300</td><td>0.960</td><td>0.706</td><td>0.700</td></tr><tr><td>qwen2.5-7b-graph_without_relations</td><td>2844</td><td>0.998</td><td>0.581</td><td>0.581</td></tr><tr><td>qwen2.5-7b-no-confidence</td><td>3300</td><td>0.999</td><td>0.680</td><td>0.680</td></tr><tr><td>qwen2.5-7b-no-provenance</td><td>3300</td><td>0.976</td><td>0.674</td><td>0.672</td></tr><tr><td>qwen2.5-7b-no-timestamps</td><td>3288</td><td>0.965</td><td>0.644</td><td>0.643</td></tr><tr><td>qwen2.5-7b-shuffled-full-graph</td><td>3297</td><td>0.919</td><td>0.712</td><td>0.676</td></tr><tr><td>qwen2.5-7b-structured_list</td><td>3182</td><td>0.990</td><td>0.634</td><td>0.634</td></tr></table>

TABLE A7

FAILURE COUNTS BY SYSTEM AND CATEGORY, POOLED OVER REGIMES AND THE THREE GENERATION SEEDS. GEMMA NON-CANONICAL ROWS ARE EXPLORATORY.
<table><tr><td>System</td><td>prem. rev.</td><td>fail. rev.</td><td>stale dom.</td><td>unrel. dom.</td><td>contr. om.</td><td>inc. attr.</td><td>rel. fail.</td><td>malformed</td><td>miscal.</td></tr><tr><td>gemma-3-12b-it-canonical_json</td><td>673</td><td>252</td><td>148</td><td>1</td><td>292</td><td>551</td><td>223</td><td>3</td><td>141</td></tr><tr><td>gemma-3-12b-it-graph</td><td>523</td><td>168</td><td>37</td><td>17</td><td>300</td><td>671</td><td>258</td><td>24</td><td>286</td></tr><tr><td>gemma-3-12b-it-graph_without_relations</td><td>551</td><td>157</td><td>59</td><td>25</td><td>302</td><td>684</td><td>229</td><td>51</td><td>295</td></tr><tr><td>gemma-3-12b-it-structured_list</td><td>530</td><td>186</td><td>58</td><td>7</td><td>391</td><td>687</td><td>280</td><td>29</td><td>210</td></tr><tr><td>mistral-7b-v0.3-canonical_json</td><td>1549</td><td>1153</td><td>150</td><td>317</td><td>764</td><td>640</td><td>394</td><td>2344</td><td>2536</td></tr><tr><td>qwen2.5-1.5b-canonical_json</td><td>1340</td><td>950</td><td>130</td><td>250</td><td>994</td><td>848</td><td>291</td><td>1635</td><td>1868</td></tr><tr><td>qwen2.5-7b-canonical_json</td><td>555</td><td>5</td><td>30</td><td>5</td><td>803</td><td>946</td><td>225</td><td>0</td><td>75</td></tr><tr><td>qwen2.5-7b-flat-edge-removed-relations</td><td>859</td><td>468</td><td>10</td><td>19</td><td>266</td><td>714</td><td>326</td><td>288</td><td>288</td></tr><tr><td>qwen2.5-7b-graph</td><td>557</td><td>465</td><td>100</td><td>147</td><td>328</td><td>738</td><td>232</td><td>0</td><td>0</td></tr><tr><td>qwen2.5-7b-graph_without_relations</td><td>914</td><td>644</td><td>113</td><td>103</td><td>233</td><td>866</td><td>301</td><td>456</td><td>456</td></tr><tr><td>qwen2.5-7b-no-confidence</td><td>485</td><td>440</td><td>0</td><td>140</td><td>163</td><td>537</td><td>252</td><td>0</td><td>1</td></tr><tr><td>qwen2.5-7b-no-provenance</td><td>420</td><td>469</td><td>5</td><td>57</td><td>321</td><td>774</td><td>228</td><td>0</td><td>0</td></tr><tr><td>qwen2.5-7b-no-timestamps</td><td>558</td><td>512</td><td>98</td><td>146</td><td>271</td><td>708</td><td>235</td><td>12</td><td>12</td></tr><tr><td>qwen2.5-7b-shuffled-full-graph</td><td>550</td><td>316</td><td>2</td><td>137</td><td>466</td><td>1074</td><td>265</td><td>3</td><td>29</td></tr><tr><td>qwen2.5-7b-structured_list</td><td>583</td><td>475</td><td>1</td><td>44</td><td>160</td><td>627</td><td>268</td><td>118</td><td>118</td></tr></table>

Second, per-scenario raw model outputs and the statistics workspace are distributed as release artifacts rather than in the source repository; all tables and figures in the main text are regenerated from those artifacts by the released analysis scripts without re-running any model.