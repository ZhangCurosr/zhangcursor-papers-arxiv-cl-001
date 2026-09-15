# Beyond Depth and Width: The Information–Slack Dilemma in Streaming Test-Time Compute

Xiaotian Zhang Trooly.AI

## Abstract

The same task and compute budget can require different reasoning policies when evidence arrives in a different order. Early computation has more time to finish but rests on incomplete or revisable evidence; waiting improves information while shrinking computational slack. We call this the information–slack dilemma.

We take the evidence-dependent computational job as the unit of analysis: when to start it, what supports its result, and when that result can be committed. Advance computation is valuable only insofar as its benefits survive the costs of verification, invalidation, and recovery. This applies to grounded incremental processing and reusable preparation as well as future-dependent speculation.

We propose a research agenda on computation under evolving evidence, prioritizing selective recovery under controlled evidence revisions. Evaluation should separate earlier-execution effects, deployment value against a full-input alternative, and the added value of predictive policies, while accounting for shared-resource costs. The objective is not maximal advance computation, but more trustworthy, on-time responses within a declared resource envelope.

## 1 Introduction

A reader counting Dallas’s first-quarter points can maintain a targeted tally if the question arrives before the sports passage. If it arrives last, the reader must instead prepare a general index, anticipate the question, or wait. The completed task is unchanged, but useful early work depends on its reveal schedule.

Test-time compute (TTC) commonly increases serial reasoning depth, parallel search width, or both after a complete prompt arrives (Muennighoff et al., 2025; Wang et al., 2023; Yao et al., 2023; Brown et al., 2024; Snell et al., 2025; Wu et al., 2025; Pan et al., 2025). Interactive input instead provides a stochastic compute window. Starting early buys slack while risking obsolete work; waiting reduces uncertainty but places more computation after input completion. We call this the information–slack dilemma.

## Position

Capability under streaming observation depends jointly on what an agent computes, which evidence the computation requires, and when its result is committed. Evaluation should make all three visible under a declared deadline and resource envelope.

A computational job need not predict a continuation: grounded processing and reusable preparation also exploit input time. We develop this distinction into questions about waiting, state recovery, verification, and shared serving.

## 1.1 An Illustrative Budgeted Example

Consider the following toy operator library. Four evidence blocks arrive at times 0, 1, 2, 3 seconds; both streams end at T = 4. The question arrives either at 0 or at 4, with identical completed input. One processor has a budget of 3.5 processorseconds, and the complete answer is due at 4.5 seconds. Jobs may start on input arrivals or prerequisite completion. An explicit end marker is released at T, with detection and transport delays ignored; policies cannot know its arrival time in advance. Answer rendering waits for this marker and the required computation.

A targeted tally costs 0.25 seconds per block and needs a specified question. A general index costs 0.75 seconds per block and supports every question in the toy family. Speculative tallying first predicts a question in 0.25 seconds, then runs targeted tallies; when the real question arrives, a

0.25-second exact-match gate accepts or rejects that guess. Every strategy spends another 0.25 seconds rendering its answer. Service is deterministic, tallies are not transferable across questions, and tallying, indexing, and the gate are exact. These are illustrative assumptions, not LLM measurements.

Table 1 summarizes the schedules. Earlyquestion tallying finishes its last block at 3.25; the general index finishes at 3.75. Both wait for the end marker before rendering. Late-question tallying instead starts at 4. Speculation with a late question must also pass its gate before rendering, or recompute all tallies on a miss.

Table 1: Illustrative schedules. Delivery is absolute time, not post-end latency. Work includes rendering, verification, and any recomputation; the deadline is 4.5 s.

<table><tr><td>Policy / question arrival</td><td>Delivery (s)</td><td>Work (s)</td></tr><tr><td>Targeted / early</td><td>4.25</td><td>1.25</td></tr><tr><td>Targeted / late</td><td>5.25</td><td>1.25</td></tr><tr><td>General index / either</td><td>4.25</td><td>3.25</td></tr><tr><td>Speculative / late, hit</td><td>4.50</td><td>1.75</td></tr><tr><td>Speculative / late, miss</td><td>5.50</td><td>2.75</td></tr></table>

At a 3.5 processor-second budget, the early question permits cheap targeted work; with a late question, the general index is reliably on time. If a speculative guess matches with probability 0.6, its expected on-time correctness is 0.6, versus 1 for the index. Reducing the budget to 2.75 excludes the complete general-index policy, while the listed speculative policy still fits on both hits and misses and delivers on time on hits.

Thus information arrival, the deadline, and budget jointly determine useful early work. This comparison concerns the declared policy menu, not global optimality: partial indexes, hybrid policies, and other operators could change the trade-off.

## 2 Intellectual Lineage

Allocating computation. Bounded optimality relates behavior to an agent’s architecture and environment (Russell and Subramanian, 1995); anytime algorithms expose quality as a function of deliberation (Zilberstein, 1996). Rational metareasoning values computation through subsequent decisions (Russell and Wefald, 1991), while deliberation scheduling allocates time among procedures (Boddy and Dean, 1994). Continual computation addresses current and possible future problems, shared subtasks, and stale results (Shahaf and Horvitz, 2009).

LLM metareasoning applies value-ofcomputation objectives to selective reasoning (De Sabbata et al., 2024); latency-aware TTC considers accuracy, tokens, and wall time (Huang et al., 2025). These complete-query allocation frameworks complement search and verification operators (Hao et al., 2023; Cobbe et al., 2021; Lightman et al., 2024; Setlur et al., 2025). The question here is allocation while the specification itself evolves.

Revising and committing. Truth maintenance records justifications for belief revision (Doyle, 1979). Incremental dialogue represents extendable and revocable information units (Schlangen and Skantze, 2011), with stability and timing measures (Baumann et al., 2011). Utterance interpretation and turn prediction anticipate future input (DeVault et al., 2011; Ekstedt and Skantze, 2021). Simultaneous translation studies READ/WRITE decisions and latency (Gu et al., 2017; Ma et al., 2019; Zheng et al., 2019a,b); SimulEval supplies causal replay (Ma et al., 2020). Online semantic parsing executes partial programs during utterances (Zhou et al., 2022).

Input-time inference. LiveMind accumulates intermediate inferences (Chen et al., 2024); Pred-Gen generates and verifies candidate responses during speech (Li and Grover, 2025); StreamingThinker combines causal reasoning units with concurrent execution (Tong et al., 2026a). The streaming-LLM survey covers interaction policies and dynamic budgeting as well as architectures (Tong et al., 2026b). Our narrower focus links evidence-dependent jobs to execution, reuse, recovery, and shared-resource costs. Appendix A compares representative mechanisms.

## 3 A Framework for Streaming TTC

## 3.1 Three Evidence-Dependency Regimes

We describe artifacts, rather than models, through three non-exclusive regimes. Grounding concerns evidential support; robustness concerns usefulness across futures. A grounded artifact may also be continuation-robust, as with an index of observed events. The crucial distinction is whether applicability requires an unconfirmed future hypothesis.

![](images/91a47c15fbad95e560a5908664d847d84d5efa8d107e18df9ce29415da51479a.jpg)  
Figure 1: Streaming TTC couples evidence sufficiency with a finite computation window. (a) The same prefix may support retrieval before it supports planning; curves are schematic, not fitted. (b) Overlapped work is not automatically useful: a premise can be revoked, a result can miss the candidate-selection cutoff $t _ { \mathrm { s e l } }$ , and verification occupies the response critical path. This example freezes the candidate set at the cutoff; other policies may admit later results. Positions are schematic.

## Evidence-grounded incremental computation.

The artifact is supported by observations already available: an entity index, a running tally, or retrieval for a stable named topic. New observations may extend it and revisions may invalidate it. Its value does not require predicting a hidden tail.

Continuation-robust preparation. The artifact is useful across a declared family of possible continuations: a reusable problem decomposition, a menu of neutral follow-up dimensions, or a general event index. It can be unused or insufficient without being factually wrong. Robustness must be tested over that family; a generic checklist is not useful merely because it is safe.

Future-dependent speculation. The artifact’s applicability depends on an unresolved hypothesis: a guessed intent, missing constraint, or likely answer. It must remain conditional until supported or rejected. Prediction coverage and false acceptance matter specifically here.

These regimes cut across operator types: retrieval or planning can belong to any of them. A note can contain fields from several regimes, which should be tracked separately. Predictable future content is one route to positive input-time value, not a necessary condition for the entire framework.

## 3.2 Episodes, Observations, and Clocks

Let T be an almost-surely finite source-end event: actual user stop-speaking time, or the last source release in a text replay. Let X be the completed, canonically serialized task input. The agent receives observations $O _ { \leq t }$ , including partial transcripts, revisions, and stability signals. Their history generates $\boldsymbol { \mathcal { T } } _ { t } ;$ the controller filtration $\mathcal { F } _ { t } \supseteq \mathcal { T } _ { t }$ also includes jobs, artifacts, queues, and resource events.

In speech, distinguish T from the detected endpoint $T _ { \mathrm { d e t } }$ and finalized-transcript availability $T _ { \mathrm { f i n } }$ The evaluator may know T retrospectively; it is not generally a stopping time of the agent’s observation filtration. The controller uses a causal posterior, not the annotated endpoint. The core protocol allows response commitment only after the episode ends, with premature commitment scored separately as a violation. Continual action before source completion requires a different deadline process.

Evidence sufficiency is operator-specific. A stable entity may suffice for retrieval before the user’s intent suffices for planning. Scheduling nevertheless requires three separate assessments: whether the current evidence supports a job, its completion-time distribution, and the downstream value of its result. No single maturity score is assumed sufficient. Appendix B gives population Bayes risk as one optional characterization, not a required controller statistic.

For delay allowance D, realized remaining deadline slack is $( T + D - t ) _ { + }$ . A controller estimates remaining time from causal observations; for example, $\widehat { S } _ { t } = \mathbb { E } [ ( T - t ) _ { + } \mid \mathcal { F } _ { t } ]$ estimates exbpected pre-end slack. This mean alone does not characterize deadline-miss risk, which depends on the joint distribution of remaining input time, job service, and queues. Endpoint detection, transcript finalization, and verification may consume part of D.

## 3.3 Causal Jobs and Resource Constraints

A state $h _ { t }$ summarizes observed evidence, operator-specific evidence estimates, the endpoint posterior, running jobs, queues, artifacts, and capacity. The action space is

$$
a _ { t } \in \left\{ \begin{array} { l } { \mathrm { w a i t , ~ s p a w n } ( o , c , P ) , } \\ { \mathrm { c o n t i n u e } ( j , c ^ { \prime } ) , \mathrm { ~ c a n c e l } ( j ) , } \\ { \mathrm { v e r i f y } ( j , c _ { v } ) , \mathrm { ~ c o m m i t } ( y ) } \end{array} \right\} ,\tag{1}
$$

where c is a budget and $P$ contains prerequisites. Operators include encoding/prefill, retrieval, extraction, planning, branching, verification, and response generation.

Each job records its operator, launch time, evidence version, dependency set, service demand, and output artifact. A frozen-snapshot job uses only its recorded launch prefix. An online-update job may consume later observations, but every update must be measurable with respect to $\mathcal { F } _ { t }$ at its execution time. Both are causal; a retrospective final transcript is not an admissible launch input.

If job $j$ receives service $\alpha _ { j } ( t )$ with demand vector $\mathbf { r } _ { j }$ , capacity requires

$$
\sum _ { j \in \mathcal { J } _ { t } } \alpha _ { j } ( t ) \mathbf { r } _ { j } \preceq \mathbf { K } ( t ) .\tag{2}
$$

Cancellation must include the time until a backend actually stops charging resources. Deleting a client-side handle does not imply reclaimed capacity.

## 3.4 Quality, Deadlines, and Critical Paths

Fix a task-specific response endpoint $A _ { \pi }$ and score the content $Y _ { \pi }$ available by that endpoint. It may be the complete answer-bearing response or a substantive interview follow-up. Set $L _ { \pi } = A _ { \pi } - T$ for a post-end response, and $L _ { \pi } = \infty$ if none is produced. Later content must not improve the score retroactively. For quality $q \in [ 0 , 1 ]$ , we take ontime quality as the primary objective:

$$
J _ { D } ( \pi ) = \mathbb { E } [ q ( Y _ { \pi } ; X , \Theta ) \mathbf { 1 } [ L _ { \pi } \le D ] ] .\tag{3}
$$

We maximize $J _ { D }$ over causal policies subject to resource and risk limits. Resource limits bound expected consumption $\mathbb { E } [ C _ { \pi , k } ] \le B _ { k }$ and instantaneous capacity in Equation (2); an application may additionally bound external harm. Gates, cancelled work, and state reconstruction count toward consumption. A launch is valuable when its expected gain in deadline utility exceeds that of waiting or another feasible use of the same resources, including effects on later decisions.

Maximizing eventual quality subject to a deadline-miss tolerance is a different objective. With nonzero allowed misses it can rank policies differently from $J _ { D } ,$ so the two must not be used interchangeably. Raw quality, latency, and cost remain necessary to interpret either.

Work overlapped with input is not latency saved. Response delay depends on the remaining dependency path, including detection, finalization, verification, and serving. Unused branches can still impose contention until stopped. Overlapped branch-seconds cannot be summed into response acceleration; Appendix E gives a residual-work lower bound.

## 4 Research Agenda: Computation under Revisable Evidence

The agenda is to predict when a job has positive value and to make its intermediate state safe to reuse. Time gained by starting early can be consumed by later validity checks and state reconstruction. Recovery cost therefore belongs in the launch decision itself: it is a consequence of the information–slack dilemma, not merely a maintenance concern. Each question below pairs a design problem with a discriminating experiment.

## 4.1 When Is Waiting Worth More than Computing?

Input progress conflates evidence sufficiency with remaining time. A late prefix may identify the task but leave too little time for a useful job. Inserting a pause into an otherwise identical stream creates additional computation time before source end; within a realized episode, slack still decreases as time passes. A pause can also change the endpoint posterior.

Vary informative-segment arrival independently of pauses, keeping completed input and resources fixed. Compare triggers based on online-estimated progress, evidence alone, slack alone, and their combination. True-final-length percentages are oracle diagnostics. A joint controller should improve deadline utility in mixed regimes; if evidence and time estimates add no predictive value beyond elapsed time, the richer state is unnecessary.

## 4.2 What Work Remains Useful across Futures?

A general event index can support several eventual questions without predicting any. A correctly guessed topic can nevertheless produce a useless plan. An artifact’s reuse region consists of terminal episodes in which retaining it improves the downstream decision under a fixed response budget.

Construct groups sharing early evidence but differing in withheld questions, constraints, or intentions. Compare the three regimes in Section 3.1 on utility and invalidation, not only textual match. Future-dependent preparation earns a distinct claim only by improving on strong nonpredictive processing. Unpredictable tails do not eliminate reusable shared subproblems.

## 4.3 When Is Local Retraction Enough?

Deleting an assumption from a note does not necessarily remove its influence. Two dependency regimes call for different recovery strategies.

Explicit dependence. A derived claim records the assumption or evidence version supporting it. Following truth maintenance and incrementalunit models (Doyle, 1979; Schlangen and Skantze, 2011), local invalidation can remove unsupported descendants, cancel dependent jobs, and retain claims with independent surviving justifications. This is sound only to the extent that the recorded support captures actual dependence.

Implicit influence. A summary, later generation, or model cache may have absorbed a premise without retaining its provenance. Removing its source may leave unsupported influence in downstream state. Recovery may require rebuilding from independently supported evidence or a checkpoint preceding exposure to that premise. Reconstruction rebuilds computational state from valid evidence; it does not guarantee a correct final answer. A text label does not certify that a cache is clean.

The research question is therefore when local invalidation is sufficient and when reconstruction is worth its cost. A controller might use artifact isolation, provenance coverage, state compression events, and contradiction severity as risk signals, rather than assume a complete dependency graph. Evaluate selective recovery against always-local and always-rebuild policies using controlled corrections. Measure stale-claim survival, valid-work retention, recovery latency, and total cost. The target is reliable selection of recovery scope, not merely a more elaborate note format.

## 4.4 When Does Verification Pay for Itself?

Compatibility, utility, and readiness differ. A supported branch may add nothing the final model needs; a useful branch may arrive too late. Gate runtime, added context, and regeneration can consume the gain.

Use paired accept/drop interventions on a candidate and final input. Compare similarity-only, support-aware, and utility-aware gates, alongside always-inject and never-inject. Measure correctto-wrong flips and critical-path change, including gate cost. Offline utility labels can train selection, but future evidence remains unavailable to the online gate.

## 4.5 Whose Latency Is Being Reduced?

Early jobs can improve one user’s response while delaying another. Compare admission, cancellation, and foreground-priority policies under fixed workload and hardware traces. Measure neighboring requests, throughput, tail latency, and resource use as load increases.

A deployment win should persist within a declared load envelope. The controller should throttle early work when opportunity cost exceeds benefit. Waiting is then a rational allocation decision, not failed prediction.

Where to start. We prioritize controlled replays with evidence revisions, comparing selective reconstruction against always-local invalidation and always-full reconstruction under matched resources. Retain a no-semantic-precomputation control that solves from the complete input, as in Section 5. Unlike always-full reconstruction, this control incurs neither advance semantic work nor its queueing effects. It separates better repair from the value of adopting advance computation and repair at all. Vary revision timing, dependency visibility, and the extent to which summaries or caches absorb revoked premises. Measure on-time quality, stale-claim survival, and recovery cost. These experiments can establish when retained state remains usable and what repair costs before a learned scheduler is asked to allocate work around it.

## 5 Evaluating Evidence-Dependent Computation

## 5.1 Match the Control to the Claim

Let π be any streaming policy under evaluation. Three contrasts answer different questions (Table 2); none requires that π predict future input or launch multiple waves.

A trace-conditioned timing control preserves the recorded job graph, evidence snapshots, and service demands in simulation, while delaying early releases. It tests whether overlap shortens the response path. Recomputing jobs from full evidence changes the information condition and answers a different question. Appendix D specifies one intervention and distinguishes frozentrace simulation from stochastic live re-execution.

Afull-input control waits for complete evidence and uses a declared allocation rule within the same budget and serving envelope. Restricting its operator portfolio aids controlled comparison but narrows the deployment claim. A practical study should also compare the strongest relevant deployed alternative; a win over one chosen portfolio does not establish global superiority.

A non-predictive control performs evidencegrounded processing and robust preparation. If its prompts, selection rules, or artifact types differ, the comparison estimates the benefit of the whole predictive policy, not isolated future assumptions. Prefill, native thinking, and single-wave controls are useful for specific additional claims and are listed in Appendix C.

## 5.2 Make Information and Time Observable

Pair policies on completed inputs, timestamped arrivals, model versions, output caps, and serving conditions. Include controlled reveal schedules that change question availability while preserving final serialization, as in Section 1.1, and pause interventions that preserve evidence order. Naturally occurring streams test whether the findings transfer. Future input length and annotated source end are evaluator-only information.

Measure response latency from actual source end T, while logging detection $T _ { \mathrm { d e t } }$ and transcript finalization $T _ { \mathrm { f i n } }$ separately. First token, first audio, first substantive audio, availability of the scored response, and task completion are distinct endpoints. Report their distributions as relevant, rather than treating them as interchangeable.

Estimate $J _ { D }$ by averaging quality multiplied by the indicator that the scored response was available by its deadline. Alternatively, evaluate the actual content delivered by $T + D$ under an incremental rubric. Neither convention credits Let me think at 100 ms with the quality of an answer delivered ten seconds later. Raw quality, timeout/error counts, and total compute and monetary cost should accompany deadline utility.

## 5.3 Interpret Gains at the Right Level

Preregister the main endpoint, deadline, resource envelope, and quality non-inferiority margin. Use paired comparisons with case-level uncertainty estimates; seeds and reveal schedules from one source case are not independent samples. Mechanism metrics should include supported-artifact reuse, stale-claim survival, correct-to-wrong flips, recovery cost, and the residual critical path. Interview harm rubrics should cover leading questions based on unsupported premises.

A streaming policy can succeed by preserving the quality of a stronger reasoning system while approaching the latency of a direct response. It need not exceed native-thinking accuracy. Conversely, an apparent speedup is insufficient if it trades away unreported quality, spends substantially more resources, or delays neighboring requests. Benefits should be attributed at the level identified by the relevant control: incremental processing, future-dependent preparation, or a particular scheduling mechanism.

## 6 Trooly as an Illustrative Design Pattern

Trooly’s interviewer motivates private preparation during speech, aiming for a useful follow-up within a short response window. After I mainly use Kimi for customer reports, a processor can record the use case, prepare neutral workflow questions, or speculate that raw-data organization is the bottleneck.

Confirmation may make a data-related plan reusable. But after The data is already organized; I only use it to adjust the tone, its premise should be revoked while the confirmed use case survives. Asking How much cleaning time did it save? then illustrates false acceptance: topical similarity conceals an unsupported assumption. If only an isolated branch depends on that premise, local invalidation can suffice. If a summary or cache absorbed it, the response context may need reconstruction.

Table 2: Three claims about an arbitrary streaming policy π. Timing, deployment, and prediction benefits require distinct comparisons.
<table><tr><td>Claim</td><td>Required comparison</td><td>Scope of the conclusion</td></tr><tr><td>Earlier execution helps</td><td>Freeze a computational trace and intervene on job release times</td><td>Scheduling value conditional on the recorded jobs and evidence</td></tr><tr><td>A streaming policy has deployment value</td><td>Compare with a specified full-input policy under the same resource and load envelope</td><td>Improvement over that alternative, not every possible post-input algorithm</td></tr><tr><td>Predicting future input adds value</td><td>Compare with strong incremental processing and continuation-robust preparation</td><td>Benefit of the predictive policy package; a pure assumption effect needs tighter ablation</td></tr></table>

The design pattern is to launch from causal evidence, track support and assumptions, invalidate obsolete work, and select useful artifacts before committing a response. Selection deadlines and protected response resources determine whether even a valid artifact can still help.

Evidence status. The scheduling example and interview alternatives are analytical illustrations. This paper proposes a problem and research agenda, not a measured performance improvement. Typed notes motivate the design; they do not establish complete dependency tracking or reliable recovery.

## 7 Objections and Scope

Will faster models make the problem disappear? They can eliminate its practical importance for some task–deadline pairs. If a full-input policy already meets the target quality, cost, and latency, anticipatory computation may add no value. The agenda concerns regimes where useful computation remains on a constrained response path; it predicts that some regimes will vanish as inference improves.

Why metareasoning if simple incremental processing works? A fixed incremental procedure may be the best policy when evidence is stable and its reuse is predictable. An adaptive controller earns its complexity only when selecting jobs, waiting, or changing recovery scope improves outcomes after controller overhead is counted. Metareasoning defines the allocation question; it does not require a learned scheduler in every deployment.

What is added beyond existing streaming research? Continual computation already studies advance work and shared subproblems; the streaming-LLM survey explicitly covers interaction policies and runtime budgeting (Shahaf and Horvitz, 2009; Tong et al., 2026b). Our proposed contribution is a narrower job-level research object linking evidence dependence to execution, reuse, and revocation, with controls separating timing from information and prediction effects. The test of this framing is whether it yields tractable questions such as selective state reconstruction, not whether it creates a new label for streaming architectures.

We restrict the model to episodic streams with an uncertain source end and post-end response deadline. Continuous action before source completion needs a sequence of decision opportunities. Inferring semantic dependence and calibrating recovery risk remain open, rather than assumed capabilities.

## 8 Conclusion

For streaming agents, evidence arrival changes which computations are useful before a deadline. The resulting choices extend beyond predicting future input: they include grounded incremental work, reusable preparation, and waiting. An evidence-dependent job can finish early yet become invalid, or remain valid yet arrive too late to help. A useful theory must account for both its contribution to the response path and the cost of trusting or rebuilding its result. The aim is not to maximize advance computation, but to convert it into trustworthy, on-time responses at acceptable verification and recovery cost. The question is which computations are worth doing now, and which results remain worth relying on as evidence changes.

Table 3: Representative contributions inherited by the agenda. Entries describe each mechanism’s focus; they do not assert that unlisted capabilities are absent.
<table><tr><td>Work</td><td>Computation object</td><td>Evidence and commitment</td><td>Control / objective</td></tr><tr><td>Russell-Wefald (1991)</td><td>Internal computations affecting an external decision</td><td>Evaluate a computational result before choosing action</td><td>Select computation by expected net decision value</td></tr><tr><td>Shahaf–Horvitz (2009)</td><td>Current/future problems and shared subtasks</td><td>Precompute reusable results; consider result</td><td>Allocate time under uncertain arrivals and utility</td></tr><tr><td>Doyle (1979)</td><td>Beliefs with recorded justifications</td><td>freshness Revise assumptions and their supported beliefs</td><td>Dependency-directed belief maintenance</td></tr><tr><td>STACL (2019)</td><td>Translation prefix</td><td>Read partial source, then emit target tokens</td><td>Control quality-latency trade-off through prefix lag</td></tr><tr><td>LiveMind (2024)</td><td>Intermediate inferences on incoming segments</td><td>Store and reuse inference memory as input grows</td><td>Decide whether to infer or await more input</td></tr><tr><td>PredGen (2025)</td><td>Candidate response and speech preparation</td><td>Verify/revise candidates before output</td><td>Hide response preparation within speech input</td></tr><tr><td>StreamingThinker (2026)</td><td>Streaming reasoning units and KV execution</td><td>Causal reading/reasoning; final answer follows</td><td>Concurrent execution and post-reading</td></tr><tr><td>Huang et al. (2025)</td><td>Reasoning method for a complete query</td><td>Allocate inference after query receipt</td><td>depth adjustment Select method using accuracy, tokens, and wall time</td></tr></table>

## A Representative Mechanisms

Table 3 compares specific mechanisms along evidence, commitment, and control dimensions. It is a selective intellectual map, not a systematic literature review or a completeness ranking.

## B One Candidate Measure of Evidence Sufficiency

For a fixed operator o, let $Z _ { o } ^ { \star } = \psi _ { o } ( X , \Theta )$ be a terminally relevant artifact and $\ell _ { o }$ its mismatch loss. One population measure of evidence sufficiency uses Bayes risk:

$$
\mathcal { R } _ { o } ( t ) = \mathbb { E } \left[ \operatorname* { i n f } _ { z \in \mathcal { Z } _ { o } } \mathbb { E } [ \ell _ { o } ( z , Z _ { o } ^ { \star } ) \mid \mathcal { T } _ { t } ] \right] .\tag{4}
$$

Let ${ \mathcal { R } } _ { o } ^ { \infty }$ be the risk after all episode observations and revisions arrive, excluding later episodes. If $\mathcal { R } _ { o } ( 0 ) > \mathcal { R } _ { o } ^ { \infty }$ , normalize:

$$
M _ { o } ( t ) = \frac { \mathcal { R } _ { o } ( 0 ) - \mathcal { R } _ { o } ( t ) } { \mathcal { R } _ { o } ( 0 ) - \mathcal { R } _ { o } ^ { \infty } } .\tag{5}
$$

Nested observation histories imply non-increasing population risk for a fixed loss, even when transcript text is corrected. Individual estimates can fluctuate. This information measure encodes neither runtime nor downstream utility.

Robust preparation may admit many equally useful representations. A loss allowing equivalent artifacts, or a downstream-regret measure, may be more appropriate. These equations are a candidate characterization, not the definition of the dilemma.

## C Additional Mechanism-Specific Controls

Choose additional controls according to the mechanism claimed:

• Direct response: an ordinary no-semanticprecomputation speed/quality anchor.

• Native reasoning: a serial-reasoning anchor, with measured cost and disclosed limits on hidden-budget control.

• Streaming prefill: causal encoding and genuine KV reuse, without semantic artifacts. Repeated full-prefix API calls are not equivalent.

• Single wave: for multi-wave claims, a causally selected single launch under the same budget envelope. True-final-length triggers are oracle diagnostics.

• Recovery and gates: always-local versus always-rebuild recovery; always-inject and never-inject versus selective reuse.

Match resource caps and final generation settings; report actual consumption, including discarded jobs. These are implementation-specific controls.

## D Replay Manifest and Trace Requirements

For frozen-snapshot jobs with release times $r _ { j }$ , a timing intervention can set

$$
r _ { j } ^ { \prime } = \operatorname* { m a x } ( T , r _ { j } ) .\tag{6}
$$

Prerequisites and queues still govern execution. This clamping intervention differs from a uniform time shift and must be preregistered.

Preserve job inputs, outputs, consumed service demands, priorities, and cancelled work. Keep exogenous neighboring arrivals fixed and recompute endogenous queues. Freezing outputs makes this a trace-conditioned scheduling diagnostic, not a new policy evaluation. Online-update jobs require recorded evidence-consumption steps. Live APIs that cannot reproduce service or cancellation behavior require replicated-run uncertainty rather than claims of exact isolation.

An auditable trace links timestamped evidence versions and revisions to job prompts, dependencies, budgets, and queue/start/end/cancellation events. It records artifact support, invalidation, selection, resource use, and all scored response events: $T , T _ { \mathrm { d e t } } , T _ { \mathrm { f i n } } .$ audio onset, substantive response, and completion. Future schedules and annotations remain evaluator-only information.

## E Residual Response-Time Accounting

Let $G _ { \pi }$ be the realized dependency DAG needed for the scored response, including detection, finalization, gates, and serving dependencies. With residual critical path $\mathrm { C P } _ { T } ( G _ { \pi } )$ and required residual work $W _ { k , T }$ on resource k, constant post-end capacity implies

$$
L _ { \pi } \geq \operatorname* { m a x } \left\{ \mathrm { C P } _ { T } ( G _ { \pi } ) , \operatorname* { m a x } _ { k } W _ { k , T } / K _ { k } \right\} .\tag{7}
$$

Unused branches are not response dependencies, but still impose contention until stopped. Queueing and scheduling can further increase latency. Overlapped branch-seconds are an accounting measure; they cannot be summed into response acceleration.

## References

Timo Baumann, Okko Buß, and David Schlangen. 2011. Evaluation and optimisation of incremental processors. Dialogue & Discourse, 2(1):113–141.

Mark S. Boddy and Thomas L. Dean. 1994. Deliberation scheduling for problem solving in time-constrained environments. Artificial Intelligence, 67(2):245–285.

Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, and Azalia Mirhoseini. 2024. Large language monkeys: Scaling inference com pute with repeated sampling. arXiv:2407.21787.

Chuangtao Chen, Grace Li Zhang, Xunzhao Yin, Cheng Zhuo, Ulf Schlichtmann, and Bing Li. 2024. LiveMind:

Low-latency large language models with simultaneous inference. arXiv:2406.14319.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv:2110.14168.

C. Nicolò De Sabbata, Theodore R. Sumers, Badr AlKhamissi, Antoine Bosselut, and Thomas L. Griffiths. 2024. Rational metareasoning for large language models. arXiv:2410.05563.

David DeVault, Kenji Sagae, and David Traum. 2011. Incremental interpretation and prediction of utterance meaning for interactive dialogue. Dialogue & Discourse, 2:143– 170.

Jon Doyle. 1979. A truth maintenance system. Artificial Intelligence, 12(3):231–272.

Erik Ekstedt and Gabriel Skantze. 2021. Projection of turn completion in incremental spoken dialogue systems. In SIGDIAL, pages 431–437.

Jiatao Gu, Graham Neubig, Kyunghyun Cho, and Victor O. K. Li. 2017. Learning to translate in real-time with neural machine translation. In EACL, pages 1053–1062.

Shibo Hao, Yi Gu, Haodi Ma, Joshua Hong, Zhen Wang, Daisy Wang, and Zhiting Hu. 2023. Reasoning with language model is planning with world model. In EMNLP, pages 8154–8173.

Jenny Y. Huang, Mehul Damani, Yousef El-Kurdi, Ramon Astudillo, and Wei Sun. 2025. Latency and token-aware test-time compute. arXiv:2509.09864.

Shufan Li and Aditya Grover. 2025. PredGen: Accelerated inference of large language models through input-time speculation for real-time speech interaction. In COLM.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In ICLR.

Mingbo Ma, Liang Huang, Hao Xiong, Renjie Zheng, Kaibo Liu, Baigong Zheng, Chuanqiang Zhang, Zhongjun He, Hairong Liu, Xing Li, Hua Wu, and Haifeng Wang. 2019. STACL: Simultaneous translation with implicit anticipation and controllable latency using prefix-to-prefix framework. In ACL, pages 3025–3036.

Xutai Ma, Mohammad Javad Dousti, Changhan Wang, Jiatao Gu, and Juan Pino. 2020. SIMULEVAL: An evaluation toolkit for simultaneous translation. In EMNLP System Demonstrations, pages 144–150.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. 2025. s1: Simple test-time scaling. In EMNLP, pages 20275–20321.

Jiayi Pan, Xiuyu Li, Long Lian, Charlie Snell, Yifei Zhou, Adam Yala, Trevor Darrell, Kurt Keutzer, and Alane Suhr. 2025. Learning adaptive parallel reasoning with language models. In COLM.

Stuart J. Russell and Devika Subramanian. 1995. Provably bounded-optimal agents. Journal ofArtificial Intelligence Research, 2:575–609.

Stuart Russell and Eric Wefald. 1991. Principles of metareasoning. Artificial Intelligence, 49(1–3):361–395.

David Schlangen and Gabriel Skantze. 2011. A general, abstract model of incremental dialogue processing. Dialogue & Discourse, 2:83–111.

Amrith Setlur, Nived Rajaraman, Sergey Levine, and Aviral Kumar. 2025. Scaling test-time compute without verification or RL is suboptimal. In ICML.

Dafna Shahaf and Eric Horvitz. 2009. Investigations of continual computation. In IJCAI, pages 285–291.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2025. Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning. In ICLR.

Junlong Tong, Yingqi Fan, Anhao Zhao, Yunpu Ma, and Xiaoyu Shen. 2026a. StreamingThinker: Large language models can think while reading. In ICLR.

Junlong Tong, Zilong Wang, Yujie Ren, Peiran Yin, Hao Wu, Wei Zhang, and Xiaoyu Shen. 2026b. From static inference to dynamic interaction: A survey of streaming large language models. In Findings ofACL, pages 10237– 10263.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models. In ICLR.

Yangzhen Wu, Zhiqing Sun, Shanda Li, Sean Welleck, and Yiming Yang. 2025. Inference scaling laws: An empirical analysis of compute-optimal inference for LLM problemsolving. In ICLR.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. In NeurIPS.

Baigong Zheng, Renjie Zheng, Mingbo Ma, and Liang Huang. 2019a. Simpler and faster learning of adaptive policies for simultaneous translation. In EMNLP-IJCNLP, pages 1349–1354.

Renjie Zheng, Mingbo Ma, Baigong Zheng, and Liang Huang. 2019b. Speculative beam search for simultaneous translation. In EMNLP-IJCNLP, pages 1395–1402.

Jiawei Zhou, Jason Eisner, Michael Newman, Emmanouil Antonios Platanios, and Sam Thomson. 2022. Online semantic parsing for latency reduction in task-oriented dialogue. In ACL, pages 1554–1576.

Shlomo Zilberstein. 1996. Using anytime algorithms in intelligent systems. AI Magazine, 17(3):73–83.