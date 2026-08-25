# LongWoF-Bench: Evaluating EvoMap Genes for Verifiable Long-Workflow Tasks

Xiao Zhang Qumeng Sun Jiahao Li Yiming Ren Xiang Liu Project Leader: Haoyang Zhang<sup>†</sup> Junjie Wang<sup>†</sup>

Infinite Evolution Lab, EvoMap Tsinghua University https://huggingface.co/datasets/EvoMapAI/LongWoF-Bench

## Abstract

Large language models are increasingly expected to execute complex workflows whose success depends on maintaining interdependent constraints and producing artifacts that satisfy strict end-to-end verification. Yet successful execution experience is typically lost after a single run, forcing subsequent models to rediscover strategies and failure modes from scratch. We study whether such experience can instead be externalized and reused through EvoMap, where verifier-confirmed execution trajectories are consolidated into structured Gene. To evaluate this setting, we introduce the Long-Workflow Benchmark (LongWoF-Bench), comprising 778 machine-verifiable tasks across code generation, agent-environment synthesis, mathematical reasoning, and rule following. On the 252 tasks with verifierconfirmed Opus trajectories, evolved EvoMap Gene outperform Skill across all seven evaluated models by 8.7–15.5 percentage points, with the gains extending to consumer models from different model families. In contrast, reference-distilled Gene do not exhibit the same advantage, indicating that compact representation alone is insufficient and that Gene utility is closely associated with verified experience provenance. For Claude Opus, Gene reuse also completes 39 more tasks than Skill while reducing solve-time token consumption by 9.9%. Together, these results show that verified execution experience can be retained and shared as a reusable external resource, enabling models to improve long-workflow completion without repeatedly paying the full cost of experience discovery.

## 1 Introduction

Large language models are increasingly expected to move beyond single-turn problem solving and execute complex end-to-end workflows [1, 2]. In software implementation, data processing, environment construction, rule-based decision making, and exact computation, success often depends not on whether each individual step appears reasonable, but on whether the entire execution jointly satisfies a set of interdependent constraints [3–7]. Later decisions may depend on states, artifacts, or requirements established earlier in the workflow, while a single interface violation, boundary error, or precedence mistake can invalidate an otherwise plausible solution. The difficulty of such tasks therefore lies not simply in longer contexts or more execution steps, but in maintaining globally consistent decisions under strict end-to-end requirements. We refer to this class of problems as verifiable long-workflow tasks, where success is determined by whether the final deliverable satisfies the complete task specification under objective verification.

This setting raises a broader question beyond solving any individual task [8]: what should happen to the execution experience once a model has successfully completed a difficult workflow? EvoMap treats such experience as a reusable external asset rather than a transient by-product of a single run [9]. Within this framework, Evolver organizes task execution and, when necessary, feedback-driven refinement until a verifier-confirmed trajectory is obtained [10]. The execution-critical knowledge distilled from this process is consolidated into a structured Gene and stored in EvoMap for subsequent reuse [11, 9]. Evolver turns verified execution into reusable experience, while EvoMap makes that experience available beyond the model and run that originally produced it.

A complementary approach to supporting long workflows is to package reusable procedural knowledge as a Skill, an increasingly common abstraction for equipping agents with task-specific instructions, workflows, and operational conventions [12]. Both Skill and Gene seek to reduce the burden of solving a complex workflow from scratch, but they preserve different forms of knowledge. A Skill primarily codifies how a task should be carried out, typically emphasizing reusable procedures, interfaces, requirements, and recommended practices. An evolved Gene instead originates from verifier-confirmed execution and preserves the strategies, corrections, boundary conditions, and failure guards that proved consequential in practice. The distinction is therefore not simply one of representation: Skill externalizes procedural knowledge, whereas evolved Gene externalizes verified execution experience.

For real-world workflows, this distinction raises a practical empirical question: whether reusable procedural knowledge and verifier-grounded execution experience provide different value when models must satisfy the same end-to-end requirements under strict verification. Existing benchmarks have substantially expanded agent evaluation toward end-to-end tasks in software, web, enterprise, operating-system, and tool-use environments [3, 13, 14, 5, 15], while recent studies have begun to explicitly evaluate reusable Skills and procedural memory [12, 16, 17]. However, the role of guidance provenance remains insufficiently understood, particularly whether experience grounded in verifierconfirmed execution offers advantages over static procedural guidance under otherwise controlled conditions. To study this question, we introduce the Long-Workflow Benchmark (LongWoF-Bench), comprising 778 machine-verifiable tasks across code generation, agent-environment synthesis, mathematical reasoning, and rule following. LongWoF-Bench compares No Context, Skill, and EvoMap Gene under shared task specifications, runtimes, and verifiers, enabling a controlled evaluation of how different sources of reusable guidance affect strict end-to-end workflow completion.

Our experiments reveal that the value of reusable guidance depends strongly on how it is acquired. On the 252 tasks with verifier-confirmed Opus trajectories, evolved EvoMap Gene consistently outperforms Skill across all seven evaluated models, with gains of 8.7–15.5 percentage points in strict pass rate. These improvements extend beyond the producer model to Sonnet, Gemini, MiniMax, and Qwen families, indicating that verified execution experience can be reused across model families. In contrast, reference-distilled Gene do not exhibit the same advantage over Skill, suggesting that compact representation alone is insufficient and that guidance provenance plays a critical role. For Claude Opus, Gene also solves 39 more tasks than Skill while reducing solve-time token consumption by 9.9%. Together, these results show that verifier-grounded execution experience can be externalized and reused to improve workflow reliability while reducing repeated exploration.

Our contributions are threefold.

• We introduce LongWoF-Bench, a benchmark of 778 machine-verifiable tasks spanning four complementary forms of long-workflow execution under strict end-to-end verification.

• We establish a controlled framework for comparing No Context, Skill, and EvoMap Gene, directly testing whether verifier-grounded execution experience provides additional value beyond reusable procedural knowledge.

• Across seven models, we show that evolved Gene consistently improve strict task completion over Skill, transfer across model families, and reduce solve-time cost, while further analyses reveal that their effectiveness is strongly associated with experience provenance.

## 2 LongWoF-Bench

## 2.1 Verifiable Long-Workflow Tasks

We formulate a verifiable long-workflow task as $\mathcal { T } = ( S , E , \mathcal { y } , V )$ , where S denotes the public task specification, E the model-accessible environment and resources, Y the space of admissible

deliverables, and V a task-specific machine verifier. Given S and E, a solver performs a sequence of dependent operations and produces a final deliverable $y \in \mathcal { V }$ , which is considered successful only if

$$
V ( S , E , y ) = 1 .\tag{1}
$$

The defining property of a long workflow is not its context length, number of actions, or execution time, but the dependency of later decisions on constraints, states, or intermediate artifacts established earlier in the process. Consequently, locally reasonable decisions can accumulate into an invalid final result when an interface contract, ordering constraint, boundary condition, or required artifact is violated.

Verification in LongWoF-Bench is designed to assess complete task fulfillment rather than intermediate plausibility. Depending on the task, V is instantiated through executable tests, hidden checks, or normalized exact matching, with success requiring all mandatory conditions to be satisfied. The information necessary to complete a task remains recoverable from the public specification, accessible environment, and output requirements, while evaluator-only artifacts are used solely to determine whether those requirements have been satisfied. A verifiable long-workflow task therefore measures whether a model can preserve interdependent requirements throughout execution and deliver an artifact that survives objective end-to-end verification.

## 2.2 Benchmark Construction and Coverage

LongWoF-Bench is constructed to place heterogeneous tasks under a common framework of verifiable end-to-end delivery. Candidate tasks are consolidated into a standardized package containing a modelvisible task specification, the required public assets, and a task-specific private evaluator. Tasks with ambiguous deliverables, insufficiently specified interfaces, unstable execution, or unreliable verification are excluded, such that every retained instance admits an objective and reproducible success criterion.

![](images/f2266e47ba93f0629797a114c6db30518e14e4642074ae23f1b39b19e331bc3b.jpg)  
Figure 1: Overview of LongWoF-Bench coverage. The benchmark contains 778 machine-verifiable tasks across four complementary forms of long-workflow execution, unified by a common task abstraction and strict end-to-end verification.

As shown in Figure 1, the current release contains 778 tasks spanning four complementary task families: 341 code-generation tasks, 127 agent-environment-synthesis tasks, 151 mathematical-reasoning tasks, and 159 rule-following tasks. Code generation emphasizes executable implementations, file contracts, interface compliance, and hidden functional tests. Agent-environment synthesis further stresses multi-artifact delivery, package interfaces, and environment-level behavior. Mathematical reasoning requires exact computation and careful treatment of formulas, boundaries, and answer normalization, whereas rule following focuses on applicability, precedence, overrides, and constrained output spaces. Although these task families differ substantially in their deliverables, each requires models to translate distributed requirements into a final artifact whose correctness can be determined through execution or exact verification.

To make this task structure concrete, Figure 2 presents a representative agent-environment instance involving the construction of an adaptive cruise-control system. The public specification provides vehicle parameters, sensor traces, and interface and performance requirements, while successful completion requires coordinated controller implementation, mode switching, parameter tuning, simulation, and multi-artifact delivery. The submitted system is then independently re-executed by a private evaluator that checks both interface compliance and behavioral constraints.

Representative Task: Adaptive Cruise-Control System   
Public Specification. Vehicle parameters, 150-second sensor traces, and interface and performance   
requirements for cruise, following, and emergency-braking modes.   
Required Workflow. Implement the PID controller and mode-switching logic, tune the controller,   
execute the simulation, and maintain consistency across generated artifacts.   
Deliverables. Controller and simulator code together with tuning results, simulation outputs, and an   
analysis report.   
Private Verification. The evaluator imports and executes the submitted implementation, reruns the   
simulation, checks interfaces and control modes, and validates performance constraints and sensitivity   
to changed PID parameters.  
Figure 2: A representative LongWoF-Bench task illustrating how a public specification, dependent execution steps, multiple deliverables, and a private verifier jointly define end-to-end task completion.

Although this example is drawn from agent-environment synthesis, the same benchmark abstraction applies across all four task families: completion requires satisfying publicly specified constraints while correctness is independently determined by a task-specific verifier. LongWoF-Bench is therefore unified not by a single application domain, but by a common requirement for constraint-sensitive execution under objective end-to-end verification.

## 2.3 Visibility Boundary and Verification Protocol

LongWoF-Bench enforces a strict separation between model-visible task information and evaluatoronly artifacts. At evaluation time, the solver receives the public specification, referenced assets, and condition-specific auxiliary context, while hidden tests, gold outputs, reference solutions, verifier logic, and evaluator-only constants remain inaccessible. Importantly, the private verifier introduces no hidden task requirements: all information necessary for successful completion is recoverable from the public specification, accessible environment, interfaces, and output contract. Model outputs are materialized as executable artifacts or structured answers and evaluated through program execution, hidden checks, or normalized exact matching. A task is counted as successful only when all mandatory verification criteria are satisfied, with the same visibility boundary and verifier held fixed across evaluation conditions.

## 2.4 Workflow Characteristics and Failure Sensitivity

Although the four task families differ in their deliverables and verification mechanisms, they share a common dependence on constraint-sensitive execution. Successful completion requires models to preserve interface contracts, ordering relations, boundary conditions, precedence rules, and crossartifact consistency throughout the workflow. Many failures therefore arise not from an absence of relevant knowledge, but from locally plausible choices that violate a task-specific requirement when propagated to the final deliverable. Such errors are particularly consequential under strict verification, where a single incorrect default, omitted artifact, or misapplied boundary can invalidate an otherwise reasonable solution. Across LongWoF-Bench, the central challenge is thus to maintain execution-critical constraints until the complete workflow survives objective end-to-end verification. This failure sensitivity motivates the study of whether reusable procedural knowledge and verifiergrounded execution experience provide different forms of support, which we examine next through Skill and Gene.

## 3 Reusable Experience with EvoMap Genes

Following the Evolver framework [10], we treat successful task execution as a source of reusable experience rather than an isolated solution. For each task, a producer model interacts with the verifier under a bounded evolution process, and the resulting verifier-confirmed trajectory is subsequently distilled into a structured Gene. The resulting asset can then be retained in EvoMap [9] and reused by either the original model or a different consumer model. This separates the cost of discovering a successful execution strategy from the cost of applying that experience in subsequent runs.

## 3.1 Constructing Genes with Evolver

We instantiate Gene construction following Evolver’s execution–verification–refinement paradigm with GDIv2. A producer model first attempts the task without access to private verifier information. If the attempt fails, the next rollout receives the previous solution together with sanitized verifier feedback and revises the solution accordingly. This process continues within a fixed rollout budget until a passing trajectory is obtained. Importantly, the object being refined during evolution is the task solution rather than the Gene itself. Once the verifier accepts a trajectory, we distill its executioncritical information into a structured Gene, preserving successful strategies, prerequisite checks, boundary conditions, and, when applicable, corrections to earlier failures. For our primary Gene set, Claude Opus serves as the producer model, yielding 252 Gene derived from verifier-confirmed execution trajectories.

## 3.2 Reusing and Sharing Genes through EvoMap

Once produced, a Gene can be reused without replaying the full trajectory or repeating the original verifier-guided refinement process. For subsequent execution, the consumer receives the public task information together with the corresponding Gene, while the producer trajectory and verifier feedback remain unavailable. This allows validated experience to support repeated execution by the same model, reducing the need to rediscover task-specific strategies and previously resolved failure modes.

EvoMap further decouples the producer of an experience from its consumer. A Gene produced from an Claude Opus [18] execution, for example, can be made available to Gemini 3.1 Pro Preview [19], MiniMax M3 [20], or other model families as external procedural experience. EvoMap therefore enables verified execution experience to persist beyond a single run and to be inherited across models, rather than remaining tied to the agent that originally discovered it. The effectiveness and efficiency of these two forms of reuse are evaluated in subsequent sections.

## 4 Experimental Setup

Evaluation protocol. We evaluate each task under three conditions: No Context, Skill, and EvoMap Gene. Across conditions, the public task specification, runtime, decoding configuration, and private verifier are held fixed, such that only the auxiliary guidance varies. Each task–condition pair is evaluated with one recorded trial, and models receive no additional verifier feedback during inference.

Models and evaluation sets. We evaluate seven models spanning four model families: Claude Opus 4.8 and Claude Sonnet 4.6 [18, 21], Gemini 3.1 Flash-Lite Preview and Gemini 3.1 Pro Preview [22, 19], MiniMax M3 [20], Qwen3-Coder-30B-A3B-Instruct [23], and Qwen3.5-397B-A17B [24]. Claude Opus 4.8 serves as the primary Gene producer, while all seven models are evaluated as consumers. The evaluation subsets and their roles are summarized in Table 1. Our primary comparison focuses on the 252 tasks for which Opus produced verifier-confirmed evolved Gene, while the remaining subsets support analyses of Gene provenance and producer effects.

Table 1: Evaluation sets and their roles. The 252 Opus-evolved and 526 reference-distilled tasks represent different Gene provenance within the full benchmark, while the 180-task common evolved set contains tasks for which both Opus and Gemini produced verifier-confirmed Genes.
<table><tr><td>Evaluation set</td><td>Tasks</td><td>Primary role</td></tr><tr><td>Full benchmark</td><td>778</td><td>Overall benchmark difficulty</td></tr><tr><td>Opus-evolved</td><td>252</td><td>Main Gene vs. Skill comparison</td></tr><tr><td>Reference-distilled</td><td>526</td><td>Gene provenance analysis</td></tr><tr><td>Opus-Gemini common evolved</td><td>180</td><td>Gene producer comparison</td></tr></table>

Note: One additional task uses a skill-distilled Gene and is included only in full-benchmark reporting.

Metrics and statistics. Our primary metric is strict pass rate, where a task is successful only if all mandatory verifier checks pass. For efficiency analysis, we additionally report solve-time token consumption, number of model calls, and tokens per passed task. Paired pass-rate differences are quantified using deterministic task-level bootstrap confidence intervals, with exact McNemar tests used to assess paired significance. Unless otherwise stated, reported uncertainty reflects variation over tasks rather than repeated hosted-model API runs.

## 5 Results

## 5.1 Verifier-Evolved Genes Improve Workflow Completion

We first evaluate the 252 tasks for which Claude Opus produced a verifier-confirmed trajectory and an evolved Gene. All three conditions are evaluated on the same tasks with the same private verifier, allowing a direct comparison of how different forms of auxiliary guidance affect strict workflow completion.

![](images/863357476b36bf7f0ce831047b3ace03c68ab39335238ccae6d83461b851991a.jpg)  
Figure 3: Strict task pass rate on the 252 tasks with verifier-evolved Opus Genes. Each line follows one consumer model across No Context, Skill, and EvoMap Gene, while the background bars show the unweighted mean across the seven models.

As shown in Figure 3, reusable guidance improves performance, but evolved EvoMap Gene provides the strongest and most consistent benefit. Averaged across the seven consumer models, strict pass rate increases from 41.0% with No Context to 51.2% with Skill and further to 62.9% with EvoMap Gene. More importantly, Gene outperforms Skill for every evaluated model, with gains ranging from 8.7 to 15.5 percentage points. For Claude Opus 4.8, for example, the pass rate increases from 63.9% with Skill to 79.4% with Gene.

The benefit is not restricted to the model that produced the experience. Although these Genes are derived from Opus trajectories, they also improve Sonnet, Gemini, MiniMax, and Qwen consumers over their corresponding Skill conditions. Verifier-confirmed execution experience can therefore remain useful beyond its producer model once it is consolidated into an EvoMap Gene.

## 5.2 Gene Utility Depends on Experience Provenance

The previous results establish the advantage of verifier-evolved Gene, but leave open whether this benefit follows from the Gene representation itself or from the experience used to construct it.

Table 2: Gene utility under verifier-evolved and reference-distilled provenance. ∆ denotes Gene minus Skill in percentage points.
<table><tr><td rowspan="2">Model</td><td colspan="3">Opus-evolved (n = 252)</td><td colspan="3">Reference-distilled (n = 526)</td></tr><tr><td>Skill</td><td>Gene</td><td>∆</td><td>Skill</td><td>Gene</td><td>∆</td></tr><tr><td>Anthropic Claude Opus 4.8</td><td>63.9%</td><td>79.4%</td><td>+15.5</td><td>28.2%</td><td>19.8%</td><td>-8.4</td></tr><tr><td>Anthropic Claude Sonnet 4.6</td><td>55.6%</td><td>66.3%</td><td>+10.7</td><td>22.1%</td><td>14.1%</td><td>-8.0</td></tr><tr><td>Google Gemini 3.1 Flash-Lite Preview</td><td>45.2%</td><td>54.0%</td><td>+8.7</td><td>18.1%</td><td>10.9%</td><td>-7.2</td></tr><tr><td>Google Gemini 3.1 Pro Preview</td><td>63.1%</td><td>72.2%</td><td>+9.1</td><td>26.9%</td><td>15.6%</td><td>-11.3</td></tr><tr><td>MiniMax M3</td><td>50.4%</td><td>63.5%</td><td>+13.1</td><td>24.0%</td><td>16.6%</td><td>-7.4</td></tr><tr><td>Qwen/Qwen3-Coder-30B-A3B-Instruct</td><td>31.8%</td><td>43.2%</td><td>+11.5</td><td>10.3%</td><td>7.0%</td><td>-3.3</td></tr><tr><td>Qwen/Qwen3.5-397B-A17B</td><td>48.4%</td><td>61.9%</td><td>+13.5</td><td>18.1%</td><td>13.5%</td><td>-4.6</td></tr></table>

Table 2 reveals a clear provenance-dependent reversal. On the 252 tasks where Opus obtained a verifier-confirmed trajectory within the evolution budget, Gene outperforms Skill for all seven

models by 8.7–15.5 percentage points. On the complementary 526 tasks where Opus failed to obtain such a trajectory and a fallback Gene was distilled from reference-side teacher signals, Gene instead trails Skill for every model by 3.3–11.3 points.

This contrast suggests that distillation alone cannot substitute for verified experience generation. Without a verifier-confirmed trajectory, the construction process lacks direct evidence of which plausible execution choices fail and which corrections actually change the verifier outcome. Effective Gene construction therefore depends not only on how experience is represented, but also on whether the underlying experience has survived end-to-end verification. Since the two provenance groups contain different tasks, this result should be interpreted as provenance-associated evidence rather than a same-task causal ablation, particularly because the reference-distilled subset consists of tasks that Opus could not solve within the evolution budget.

![](images/f4bcb6f5651e53e3ecf1757ffa880a0dcf0acac9ca957b1a940d968b66c5636a.jpg)  
Figure 4: Gene producer comparison on the common evolved set (n = 180). Each connector compares Gemini-authored and Opus-authored Genes for the same consumer model, with labels reporting the Opus-minus-Gemini difference in strict pass rate.

Gene provenance also varies with the model that produces the underlying experience. To control for task selection, we restrict this analysis to the 180 tasks for which both Opus and Gemini produced verifier-confirmed trajectories. As shown in Figure 4, Opus-authored Gene outperform Geminiauthored Gene for every consumer model, with gains ranging from 4.4 to 11.7 percentage points. Because the task set is fixed, this result provides stronger evidence that the quality of the producing process affects downstream Gene utility. It does not, however, isolate whether the difference arises from exploration quality, the experience selected during distillation, or how that experience is expressed in the final Gene. Verified provenance matters not only in whether a Gene is grounded in successful execution, but also in the quality ofthe execution experiencefrom which it is produced.

## 5.3 Gene Reuse Reduces Repeated Exploration

We next examine whether previously discovered execution experience can reduce the cost of solving the same workflows again. Under the same one-shot protocol on the 252 evolved tasks, Skill passes 161 tasks using 803,099 solve-time tokens, whereas EvoMap Gene passes 200 tasks using 723,480 tokens. Thus, Gene completes 39 additional tasks while consuming 79,619 fewer tokens, corresponding to a 9.9% reduction in total solve-time cost. Gene reuse therefore improves both task completion and inference efficiency relative to static procedural guidance.

Figure 5 further compares one-shot reuse with the original process used to discover the verified experience. Multi-round discovery requires 404 model calls and 1,333,968 solve-time tokens to obtain verifier-confirmed trajectories for all 252 tasks, whereas one-shot Gene reuse requires 252 calls and 723,480 tokens, reducing solve-time tokens by 45.8Since discovery can repeatedly consume verifier feedback and ultimately solves all 252 tasks, while one-shot reuse passes 200, this comparison measures the amortized cost of reusing previously discovered experience rather than an equal-success replacement. EvoMap does not eliminate the cost ofdiscovering successful experience; it allows that cost to be amortized across subsequent executions.

![](images/8cb7680d11642f4fe77cefaea47ecce418bdf20713952d83358f8564531198b7.jpg)  
Figure 5: Solve-time token cost of multi-round experience discovery and one-shot Gene reuse on the same 252 tasks. Discovery can consume verifier feedback across multiple calls, whereas reuse performs a single Gene-guided attempt per task.

Table 3: Discovery and one-shot Opus-Gene reuse cost by workflow family. Change denotes the reduction in average solve-time tokens per task.
<table><tr><td>Family</td><td>Tasks</td><td>Disc. calls</td><td>Disc. total</td><td>Disc. avg.</td><td>Gene passed</td><td>Gene total</td><td>Gene avg.</td><td>Change</td></tr><tr><td>Agent environment</td><td>64</td><td>90</td><td>310,624</td><td>4853.5</td><td>53</td><td>165,110</td><td>2579.8</td><td>-46.8%</td></tr><tr><td>Code generation</td><td>62</td><td>100</td><td>562,667</td><td>9075.3</td><td>43</td><td>248,955</td><td>4015.4</td><td>-55.8%</td></tr><tr><td>Math reasoning</td><td>30</td><td>42</td><td>264,794</td><td>8826.5</td><td>25</td><td>176,783</td><td>5892.8</td><td>-33.2%</td></tr><tr><td>Rule following</td><td>96</td><td>172</td><td>195,883</td><td>2040.4</td><td>79</td><td>132,632</td><td>1381.6</td><td>-32.3%</td></tr><tr><td>Total</td><td>252</td><td>404</td><td>1,333,968</td><td>5293.5</td><td>200</td><td>723,480</td><td>2871.0</td><td>-45.8%</td></tr></table>

As shown in Table 3, reuse reduces average solve-time tokens across all four workflow families, with the largest savings in code generation (55.8%) and agent-environment synthesis (46.8%). These tasks frequently require repeated construction or revision of executable artifacts, making the cost of rediscovering an execution strategy particularly high. The reported accounting includes all one-shot Gene attempts, including failures, but excludes the one-time cost of Gene distillation and auditing.

## 5.4 Gains Vary Across Workflow Types

![](images/d4d89a7178d3739aa50dceb9064a4ae817ad57b5adc7b322d50dcfa30fdb4e17.jpg)

![](images/acda6aadd59cff065ed51b8c766c4e691dd0223604d8c326a478ead385cf49dd.jpg)  
Figure 6: Skill and Gene pass rates by workflow family on the 252-task evolved subset. Signed label report Gene-minus-Skill differences in percentage points, with negative values and ties retained.

The benefit of evolved Gene varies across workflow types. As shown in Figure 6, the most consistent gains occur in agent-environment synthesis and rule following, where Gene improves all seven consumer models by 15.6–29.7 and 2.1–22.9 percentage points, respectively. These tasks frequently depend on fragile operational conventions, including interface and multi-artifact consistency, precedence, overrides, and boundary conditions. Verified execution experience appears particularly useful when success depends on preserving task-specific operational constraints throughout the workflow.

Code generation is more model-dependent, although the observed regressions are relatively small: the three negative differences range from only 1.7 to 4.9 points, while the remaining models show gains or a tie. This variation suggests that some coding tasks benefit from the broader interface and implementation coverage of Skill, whereas others depend on a small number of execution conventions that are more effectively preserved by Gene. Mathematical reasoning shows a different pattern, with a substantial gain for Opus but ties or regressions for several other models. Unlike interface, precedence, or workflow errors, many mathematical failures ultimately depend on whether the consumer model can carry out exact multi-step reasoning and computation after the relevant procedure has been identified. Gene can transfer formulas, boundary conventions, and solution strategies, but it cannot fully compensate when the dominant bottleneck lies in the model’s underlying reasoning capability. These results suggest that Gene utility may also be influenced by the interaction between the transferred experience, the target model, and the structure of the workflow.

## 5.5 Case Studies: Transferring Execution-Critical Experience

Aggregate results show when evolved Gene improve performance, while representative cases illustrate what execution-critical information is actually transferred. We focus on two tasks that expose different failure modes, with complete trajectories and additional cases provided in the Appendix.

T0161: Drill-Hole Assay Compositing. The task requires overlapping assay intervals to be resolved before compositing them into fixed-length bins. Opus initially adopts a plausible “later interval wins” convention, but this strategy repeatedly fails the verifier. The successful trajectory instead segments intervals at every breakpoint, averages values from all assays covering each segment, and then applies length-weighted aggregation. Without the Gene, Gemini Pro follows a similarly incorrect truncation strategy and produces an invalid mineralization score; with the evolved Gene, it applies the verified overlap semantics and passes in one attempt. The transferred experience is therefore not a generic instruction to handle overlaps, but the specific semantic convention that proved necessary under verification.

T0659: Emergency Dispatch. This task combines a strict distance threshold with precedence among injury, armed-without-injury, and distance-only rules. Both an intermediate Opus rollout and unguided Gemini Pro over-escalate the case by treating weapon presence as dominant and mishandling the equality boundary. The verifier-confirmed trajectory resolves both issues by treating equality as non-triggering and applying the correct precedence order. With this experience encoded in the Gene, Gemini Pro avoids the same failure and returns the verified dispatch decision. Together, these cases show how evolved Genes can transfer the precise corrections that distinguish a plausible execution pathfrom one that survives end-to-end verification.

## 6 Conclusion

We introduced LongWoF-Bench, a benchmark of 778 machine-verifiable tasks for evaluating strict end-to-end completion under interdependent workflow constraints. Using this benchmark, we studied whether execution experience consolidated as EvoMap Gene provides value beyond reusable procedural knowledge represented by Skill. On the tasks with verifier-confirmed Opus trajectories, evolved Gene consistently outperform Skill across all seven evaluated consumer models, demonstrating that verified execution experience can remain useful beyond the model that originally produced it. Our provenance analysis further shows that this advantage does not arise from the Gene representation alone, as reference-distilled Gene do not exhibit the same behavior. Gene reuse also reduces solve-time cost relative to both static Skill guidance and the original multi-round discovery process, indicating that successful exploration can be amortized across subsequent executions. Taken together, these results suggest that verified agent experience can be externalized, retained, and shared through EvoMap, turning successful task execution from a transient outcome into a reusable resource for future models and workflows.

## References

[1] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Jirong Wen. A survey on large language model based autonomous agents. Frontiers Comput. Sci., 18(6):186345, 2024.

[2] Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, Zhangyue Yin, Shihan Dou, Rongxiang Weng, Wenjuan Qin, Yongyan Zheng, Xipeng Qiu, Xuanjing Huang, Qi Zhang, and Tao Gui. The rise and potential of large language model based agents: A survey. Sci. China Inf. Sci., 68(2), 2025.

[3] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R. Narasimhan. Swe-bench: Can language models resolve real-world github issues? In ICLR. OpenReview.net, 2024.

[4] Léo Boisvert, Megh Thakkar, Maxime Gasse, Massimo Caccia, Thibault Le Sellier de Chezelles, Quentin Cappart, Nicolas Chapados, Alexandre Lacoste, and Alexandre Drouin. Workarena++: Towards compositional planning and reasoning-based common knowledge work tasks. In NeurIPS, 2024.

[5] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. In NeurIPS, 2024.

[6] Wangtao Sun, Chenxiang Zhang, Xueyou Zhang, Xuanqing Yu, Ziyang Huang, Haotian Xu, Shizhu He, Jun Zhao, and Kang Liu. Beyond instruction following: Evaluating inferential rule following of large language models. In CCL, volume 16052 of Lecture Notes in Computer Science, pages 408–434. Springer, 2025.

[7] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In NeurIPS Datasets and Benchmarks, 2021.

[8] Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: LLM agents are experiential learners. In AAAI, pages 19632–19642. AAAI Press, 2024.

[9] EvoMap. EvoMap: Ai self-evolution infrastructure. https://evomap.ai/, 2026.

[10] EvoMap. Evolver: Agent self-evolving engine. https://github.com/EvoMap/evolver, 2026. GitHub repository.

[11] Junjie Wang, Yiming Ren, and Haoyang Zhang. From procedural skills to strategy genes: Towards experience-driven test-time evolution. CoRR, abs/2604.15097, 2026.

[12] Xiangyi Li, Wenbo Chen, Yimin Liu, Shenghan Zheng, Xiaokun Chen, Yifeng He, Yubo Li, Bingran You, Haotian Shen, Jiankai Sun, Shuyi Wang, Qunhong Zeng, Di Wang, Xuandong Zhao, Yuanli Wang, Roey Ben Chaim, Zonglin Di, Yipeng Gao, Junwei He, Yizhuo He, Liqiang Jing, Luyang Kong, Xin Lan, Jiachen Li, Songlin Li, Yijiang Li, Yueqian Lin, Xinyi Liu, Xuanqing Liu, Haoran Lyu, Ze Ma, Bowei Wang, Runhui Wang, Tianfu Wang, Wengao Ye, Yue Zhang, Hanwen Xing, Yiqi Xue, Steven Dillmann, and Han-Chung Lee. SkillsBench: Benchmarking how well agent skills work across diverse tasks. CoRR, abs/2602.12670, 2026.

[13] Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. Webarena: A realistic web environment for building autonomous agents. In ICLR. OpenReview.net, 2024.

[14] Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H. Laradji, Manuel Del Verme, Tom Marty, David Vázquez, Nicolas Chapados, and Alexandre Lacoste. Workarena: How capable are web agents at solving common knowledge work tasks? In ICML, volume 235 of Proceedings of Machine Learning Research, pages 11642–11662. PMLR / OpenReview.net, 2024.

[15] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik R. Narasimhan. τ -bench: A benchmark for tool-agent-user interaction in real-world domains. In ICLR. OpenReview.net, 2025.

[16] Yujian Liu, Jiabao Ji, Li An, Tommi S. Jaakkola, Yang Zhang, and Shiyu Chang. How well do agentic skills work in the wild: Benchmarking LLM skill usage in realistic settings. CoRR, abs/2604.04323, 2026.

[17] Julia Belikova, Rauf Parchiev, Evgeny Egorov, Grigorii Davydenko, Gleb Gusev, Andrey V. Savchenko, and Maksim Makarenko. Managing procedural memory in LLM agents: Control, adaptation, and evaluation. CoRR, abs/2606.23127, 2026.

[18] Anthropic. Claude opus 4.8 system card, 2026.

[19] Google DeepMind. Gemini 3.1 pro model card, 2026.

[20] Xunhao Lai, Weiqi Xu, Yufeng Yang, Qiaorui Chen, Yang Xu, Lunbin Zeng, Xiaolong Li, Haohai Sun, Haichao Zhu, Vito Zhang, Jinkai Hu, Jiayao Li, Rui Gao, Zekun Li, Songquan Zhu, Jingkai Zhou, and Pengyu Zhao. Minimax sparse attention. CoRR, abs/2606.13392, 2026.

[21] Anthropic. Claude sonnet 4.6 system card, 2026.

[22] Google DeepMind. Gemini 3.1 flash-lite model card, 2026.

[23] Qwen Team. Qwen3 technical report. CoRR, abs/2505.09388, 2025.

[24] Qwen Team. Qwen3.5: Towards native multimodal agents, 2026.

## Appendix

## A Detailed Experimental Results

## A.1 Main Evolved-Gene Results

This section provides the complete numerical results underlying the main evolved-Gene comparison in Figure 3. As shown in Table 4, Gene achieves the highest strict pass rate for all seven consumer models on the same 252 Opus-evolved tasks. The corresponding paired effects are reported in Table 5, where all bootstrap confidence intervals exclude zero and all improvements are significant under exact McNemar tests.

Table 4: Strict pass rate on the 252 tasks with an Opus-evolved Gene.
<table><tr><td>Model</td><td>No Context</td><td>Skill</td><td>Opus Gene</td></tr><tr><td>Anthropic Claude Opus 4.8</td><td>54.8%</td><td>63.9%</td><td>79.4%</td></tr><tr><td>Anthropic Claude Sonnet 4.6</td><td>44.4%</td><td>55.6%</td><td>66.3%</td></tr><tr><td>Google Gemini 3.1 Flash-Lite Preview</td><td>35.3%</td><td>45.2%</td><td>54.0%</td></tr><tr><td>Google Gemini 3.1 Pro Preview</td><td>47.6%</td><td>63.1%</td><td>72.2%</td></tr><tr><td>MiniMax M3</td><td>38.5%</td><td>50.4%</td><td>63.5%</td></tr><tr><td>Qwen/Qwen3-Coder-30B-A3B-Instruct</td><td>31.8%</td><td>31.8%</td><td>43.2%</td></tr><tr><td>Qwen/Qwen3.5-397B-A17B</td><td>34.9%</td><td>48.4%</td><td>61.9%</td></tr></table>

Table 5: Paired Opus-Gene minus Skill effects on the 252-task evolved subset. Confidence intervals use a deterministic paired task bootstrap, and significance levels are based on exact McNemar tests.
<table><tr><td>Model</td><td>∆(pp)</td><td>Paired Bootstrap 95%CI</td><td>Significance</td></tr><tr><td>Anthropic Claude Opus 4.8</td><td>15.48</td><td>[8.33, 22.22]</td><td> $p < 0 . 0 0 1$ </td></tr><tr><td>Anthropic Claude Sonnet 4.6</td><td>10.71</td><td>[3.57, 17.46]</td><td> $p < 0 . 0 1$ </td></tr><tr><td>Google Gemini 3.1 Flash-Lite Preview</td><td>8.73</td><td>[1.98, 15.87]</td><td> $p < 0 . 0 5$ </td></tr><tr><td>Google Gemini 3.1 Pro Preview</td><td>9.13</td><td>[2.38, 15.49]</td><td> $p < 0 . 0 1$ </td></tr><tr><td>MiniMax M3</td><td>13.10</td><td>[5.94, 20.24]</td><td> $p < 0 . 0 0 1$ </td></tr><tr><td>Qwen/Qwen3-Coder-30B-A3B-Instruct</td><td>11.51</td><td>[4.37, 18.65]</td><td> $p < 0 . 0 1$ </td></tr><tr><td>Qwen/Qwen3.5-397B-A17B</td><td>13.49</td><td>[6.35, 20.24]</td><td> $p < 0 . 0 0 1$ </td></tr></table>

## A.2 Gene Provenance and Producer Effects

This section reports the complete results underlying the provenance and Gene-producer analyses in the main text. As shown in Table 6, reference-distilled Gene trail Skill across all seven consumer models on the complementary 526-task subset. Table 7 further reports results on the 180 tasks for which both Opus and Gemini produced verifier-confirmed trajectories, providing a task-matched comparison of Gene producers. Across this common set, Opus-authored Gene outperform Gemini-authored Gene for every consumer model.

Table 6: Strict pass rate on the 526 tasks with a reference-distilled Opus Gene.
<table><tr><td>Model</td><td>No Context</td><td>Skill</td><td>Opus Gene</td></tr><tr><td>Anthropic Claude Opus 4.8</td><td>3.6%</td><td>28.2%</td><td>19.8%</td></tr><tr><td>Anthropic Claude Sonnet 4.6</td><td>3.2%</td><td>22.1%</td><td>14.1%</td></tr><tr><td>Google Gemini 3.1 Flash-Lite Preview</td><td>2.1%</td><td>18.1%</td><td>10.9%</td></tr><tr><td>Google Gemini 3.1 Pro Preview</td><td>4.6%</td><td>26.9%</td><td>15.6%</td></tr><tr><td>MiniMax M3</td><td>4.2%</td><td>24.0%</td><td>16.6%</td></tr><tr><td>Qwen/Qwen3-Coder-30B-A3B-Instruct</td><td>2.5%</td><td>10.3%</td><td>7.0%</td></tr><tr><td>Qwen/Qwen3.5-397B-A17B</td><td>2.5%</td><td>18.1%</td><td>13.5%</td></tr></table>

## A.3 Experience Discovery and Reuse Cost

This section provides the detailed cost accounting underlying the experience-discovery and reuse analysis in Figure 5 and Table 3. As shown in Table 8, 140 of the 252 verifier-confirmed trajectories are discovered on the first attempt, while 112 tasks require at least one feedback-driven refinement. These

Table 7: Complete results on the 180 tasks for which both Opus and Gemini produced verifierconfirmed Gene. ∆ denotes Opus Gene minus Gemini Gene in percentage points.
<table><tr><td>Model</td><td>No Context</td><td>Skill</td><td>Gemini Gene</td><td>Opus Gene</td><td>∆</td></tr><tr><td>Anthropic Claude Opus 4.8</td><td>61.7%</td><td>68.9%</td><td>75.6%</td><td>86.1%</td><td>+10.6</td></tr><tr><td>Anthropic Claude Sonnet 4.6</td><td>53.3%</td><td>61.1%</td><td>62.8%</td><td>74.4%</td><td>+11.7</td></tr><tr><td>Google Gemini 3.1 Flash-Lite Preview</td><td>45.6%</td><td>50.6%</td><td>58.3%</td><td>62.8%</td><td>+4.4</td></tr><tr><td>Google Gemini 3.1 Pro Preview</td><td>63.3%</td><td>71.1%</td><td>72.2%</td><td>82.8%</td><td>+10.6</td></tr><tr><td>MiniMax M3</td><td>46.7%</td><td>57.8%</td><td>61.1%</td><td>70.0%</td><td>+8.9</td></tr><tr><td>Qwen/Qwen3-Coder-30B-A3B-Instruct</td><td>42.2%</td><td>37.2%</td><td>48.3%</td><td>53.9%</td><td>+5.6</td></tr><tr><td>Qwen/Qwen3.5-397B-A17B</td><td>41.7%</td><td>54.4%</td><td>59.4%</td><td>70.6%</td><td>+11.1</td></tr></table>

refinements introduce 152 additional model calls beyond a one-call-per-task baseline, illustrating the trial-and-error cost that a reusable Gene can subsequently avoid.

Table 8: Exploration rounds required to discover the 252 verifier-confirmed Opus trajectories. Additional calls are measured relative to one attempt per task.
<table><tr><td>Passing Round</td><td>Thinking</td><td>Tasks</td><td>Share</td><td>Additional Calls</td></tr><tr><td>1</td><td>off</td><td>140</td><td>55.6%</td><td>0</td></tr><tr><td>2</td><td>low</td><td>72</td><td>28.6%</td><td>72</td></tr><tr><td>3</td><td>high</td><td>40</td><td>15.9%</td><td>80</td></tr><tr><td>Total</td><td>off/low/high</td><td>252</td><td>100.0%</td><td>152</td></tr></table>

Table 9 reports the corresponding end-to-end solve-time cost on the same 252 tasks. Multi-round discovery reaches all 252 verified solutions using 404 calls and 1,333,968 tokens, whereas one-shot Opus-Gene reuse passes 200 tasks using 252 calls and 723,480 tokens. Under the same one-shot protocol, Gene also improves over Skill in both effectiveness and efficiency, passing 39 additional tasks while using 9.9% fewer tokens. The accounting includes all solve attempts, including failures, but excludes the one-time cost of Gene distillation and auditing.

Table 9: Solve-time cost on the same 252 evolved tasks. Multi-round discovery can use verifier feedback, whereas the remaining conditions make one call per task.
<table><tr><td>Mode</td><td>Calls</td><td>Passed</td><td>Pass Rate</td><td>Total Tokens</td><td>Avg./Task</td><td>Tokens/Pass</td></tr><tr><td>Multi-round discovery</td><td>404</td><td>252</td><td>100.0%</td><td>1,333,968</td><td>5293.5</td><td>5293.5</td></tr><tr><td>No Context, single call</td><td>252</td><td>138</td><td>54.8%</td><td>651,388</td><td>2584.9</td><td>4720.2</td></tr><tr><td>Skil1, single call</td><td>252</td><td>161</td><td>63.9%</td><td>803,099</td><td>3186.9</td><td>4988.2</td></tr><tr><td>Opus Gene, single call</td><td>252</td><td>200</td><td>79.4%</td><td>723,480</td><td>2871.0</td><td>3617.4</td></tr></table>

## A.4 Results by Workflow Type

This section provides the complete workflow-family results underlying Figure 6. Table 10 reports No Context, Skill, and Opus-Gene strict pass rates for all seven consumer models on the 252 evolved tasks. The full breakdown preserves the heterogeneous patterns discussed in the main text, including consistently positive Gene effects for agent-environment synthesis and rule following, as well as the more model-dependent results in code generation and mathematical reasoning.

Table 10: Complete workflow-family breakdown on the 252 tasks with an Opus-evolved Gene.
<table><tr><td>Family</td><td>Model</td><td>No Context</td><td>Skill</td><td>Opus Gene</td></tr><tr><td>Agent environment</td><td>Anthropic Claude Opus 4.8</td><td>75.0%</td><td>67.2%</td><td>82.8%</td></tr><tr><td></td><td>Anthropic Claude Sonnet 4.6</td><td>67.2%</td><td>54.7%</td><td>75.0%</td></tr><tr><td></td><td>Google Gemini 3.1 Flash-Lite Preview</td><td>56.2%</td><td>45.3%</td><td>71.9%</td></tr><tr><td></td><td>Google Gemini 3.1 Pro Preview</td><td>65.6%</td><td>59.4%</td><td>78.1%</td></tr><tr><td></td><td>MiniMax M3</td><td>51.6%</td><td>48.4%</td><td>73.4%</td></tr><tr><td></td><td>Qwen/Qwen3-Coder-30B-A3B-Instruct</td><td>54.7%</td><td>40.6%</td><td>65.6%</td></tr><tr><td></td><td>Qwen/Qwen3.5-397B-A17B</td><td>54.7%</td><td>46.9%</td><td>76.6%</td></tr><tr><td>Code generation</td><td>Anthropic Claude Opus 4.8</td><td>56.5%</td><td>71.0%</td><td>69.3%</td></tr><tr><td></td><td>Anthropic Claude Sonnet 4.6</td><td>38.7%</td><td>53.2%</td><td>62.9%</td></tr><tr><td></td><td>Google Gemini 3.1 Flash-Lite Preview</td><td>22.6%</td><td>37.1%</td><td>41.9%</td></tr><tr><td></td><td>Google Gemini 3.1 Pro Preview</td><td>30.6%</td><td>56.5%</td><td>51.6%</td></tr><tr><td></td><td>MiniMax M3</td><td>29.0%</td><td>53.2%</td><td>53.2%</td></tr><tr><td></td><td>Qwen/Qwen3-Coder-30B-A3B-Instruct</td><td>11.3%</td><td>16.1%</td><td>22.6%</td></tr><tr><td></td><td>Qwen/Qwen3.5-397B-A17B</td><td>27.4%</td><td>43.5%</td><td>40.3%</td></tr><tr><td>Mathematical reasoning</td><td>Anthropic Claude Opus 4.8</td><td>43.3%</td><td>43.3%</td><td>83.3%</td></tr><tr><td></td><td>Anthropic Claude Sonnet 4.6</td><td>16.7%</td><td>23.3%</td><td>20.0%</td></tr><tr><td></td><td>Google Gemini 3.1 Flash-Lite Preview</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr><tr><td></td><td>Google Gemini 3.1 Pro Preview</td><td>63.3%</td><td>60.0%</td><td>60.0%</td></tr><tr><td></td><td>MiniMax M3</td><td>33.3%</td><td>40.0%</td><td>23.3%</td></tr><tr><td></td><td>Qwen/Qwen3-Coder-30B-A3B-Instruct</td><td>0.0%</td><td>0.0%</td><td>0.0%</td></tr><tr><td></td><td>Qwen/Qwen3.5-397B-A17B</td><td>0.0%</td><td>20.0%</td><td>13.3%</td></tr><tr><td>Rule following</td><td>Anthropic Claude Opus 4.8</td><td>43.8%</td><td>63.5%</td><td>82.3%</td></tr><tr><td></td><td>Anthropic Claude Sonnet 4.6</td><td>41.7%</td><td>67.7%</td><td>77.1%</td></tr><tr><td></td><td>Google Gemini 3.1 Flash-Lite Preview</td><td>40.6%</td><td>64.6%</td><td>66.7%</td></tr><tr><td></td><td>Google Gemini 3.1 Pro Preview</td><td>41.7%</td><td>70.8%</td><td>85.4%</td></tr><tr><td></td><td>MiniMax M3</td><td>37.5%</td><td>53.1%</td><td>76.0%</td></tr><tr><td></td><td>Qwen/Qwen3-Coder-30B-A3B-Instruct</td><td>39.6%</td><td>45.8%</td><td>55.2%</td></tr><tr><td></td><td>Qwen/Qwen3.5-397B-A17B</td><td>37.5%</td><td>61.5%</td><td>81.2%</td></tr></table>