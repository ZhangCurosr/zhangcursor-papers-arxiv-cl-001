# From Belief Inference to Task Action: A Multi-Level Evaluation of Theory of Mind on Verifiable Tasks

Zhenshuai Yin<sup>1</sup> Yongxin Guo<sup>2</sup> Mingda Zhang<sup>1</sup> Zhongze Luo<sup>1</sup>

Haoyue Liu<sup>1</sup> Lianmin Zhang<sup>3</sup> Xiaoying Tang<sup>1∗</sup>

<sup>1</sup>School of Science and Engineering, The Chinese University of Hong Kong (Shenzhen), Longgang, Shenzhen, Guangdong 518172, P.R. China <sup>2</sup>ByteDance Inc. <sup>3</sup>Shenzhen Research Institute of Big Data, Shenzhen, China

## Abstract

Efective collaboration requires language models to understand what others know and believe, and to act accordingly. Models that assume others share their information may omit necessary communication and undermine coordination. Many theory-of-mind (ToM) benchmarks assess mental-state inference through story questions, but correct answers do not establish that models can use these judgments in subsequent tasks. We introduce a multi-level evaluation built on existing, verifiable tasks. We extend household collaboration and calendar scheduling with asymmetric information while retaining their original constraints and validators. Independent interfaces assess world understanding, information access, beliefs, decisions, and execution. Controls supply personal evidence or correct beliefs to distinguish state recovery from state use. Across six model configurations, these abilities diverge. On matched scheduling cases, Claude recovers all agent beliefs correctly but produces subjectively optimal plans in only 67.9% of cases. In household collaboration, supplying beliefs raises Qwen-27B success from 55.0% to 75.0%, yet Qwen-4B answers every fixed supplied-belief source question correctly while completing only 20.0% of the corresponding continuation condition. These results show why ToM evaluation must connect perspective inference to independently verified decisions and task consequences.

## 1 Introduction

Language-model theory of mind (ToM) is often tested through diferences in what characters know. One person leaves before another moves an object, and the model must infer where the absent person believes it is. A conversation participant misses a message, and the model must distinguish their knowledge from that of the reader (Le et al., 2019; Kim et al., 2023). These designs test whether models preserve personal perspectives rather than simply repeat the latest fact.

![](images/9c573f3be4ec90b13b0821f7762593a23716a4fdf0936febe49f0f95806b3fe5.jpg)  
Figure 1: An actual CalBench case connects private information to complete plans. The same five-meeting proposal is optimal for uninformed A but infeasible for B, who knows slot 8 is blocked. Both proposals fail in the actual world. Native replay and subjective solvers distinguish perspective-specific correctness from world success. Appendix M gives all five proposals.

Belief-based action prediction extends this question to what a character will do (Gandhi et al., 2023). Producing an entire constraint-satisfying plan, however, places a diferent demand on a model than predicting an action in a story. Functional ToM already studies adaptation to others and actual interaction (Riemer et al., 2025; Juneja et al., 2026). The remaining evaluation problem is to connect personal-state recovery, its use in decisions, and independently verifiable task outcomes without treating any one as a substitute for the others.

We ask whether models can recover what different agents know and use it within existing executable tasks. We extend C-WAH household collaboration (Zhang et al., 2024) and CalBench scheduling (Zou et al., 2026), preserving their goals, constraints, and validators while constructing personal observation and communication histories. Figure 1 illustrates the distinction: an uninformed person’s plan can be correct under their evidence yet fail in the world. Conversely, a successful plan can happen to satisfy several perspectives without recovering all of them.

![](images/a329cbede6a86073d2d9ee57feddbcf4f82536e331af6f2f9d45b8380656badc.jpg)  
Figure 2: Evaluation from personal evidence to task action. Independent interfaces probe world, access, beliefs, and decisions; native execution supplies action consequences and subsequent observations. Supplied-access and supplied-belief controls diagnose state recovery and use. The right-hand examples illustrate reference states, not measured model answers.

A single outcome cannot explain these cases. Failure may arise from misunderstanding events, misattributing evidence, losing an earlier belief, or planning incorrectly after the state is known. We connect five evaluation targets—world, access, belief, decision, and execution—and supply personal evidence or reference beliefs in diagnostic controls. Requests are independent, preventing errors in one answer from being mechanically propagated to the next.

Our contributions are a task-grounded ToM construction with native reference evidence and verifiable plans, a cross-domain protocol separating state recovery from state use, and a sixconfiguration analysis of their behavioral diferences. The results expose both strong belief inference without optimal planning and near-perfect fixed decisions without successful long-horizon coordination.

## 2 Related Work

Mental-state inference. Cognitive AI connects beliefs and intentions to action understanding (Zhu et al., 2020). ToMi controls story regularities in belief questions (Le et al., 2019). FANToM tests beliefs, information accessibility, and answerability in multi-party conversations (Kim et al., 2023). BigToM links perceptions, beliefs, desires, and actions through causal templates (Gandhi et al., 2023). Hi-ToM examines higher-order reasoning (He et al., 2023), while ToMBench and OpenToM broaden mental states and contextual diversity (Chen et al., 2024; Xu et al., 2024). ExploreToM searches programmatically tracked stories (Sclar et al., 2025), and ToMATO constructs questions from role-playing conversations (Shinoda et al., 2025). Percept-ToMi and Percept-FANToM isolate perception attribution (Jung et al., 2024). We additionally evaluate complete plans under personal constraints and their execution.

Functional and interactive ToM. Riemer et al. (2025) distinguish predicting opponents from using predictions in games. Collaborative bomb disposal combines beliefs with team outcomes (Li et al., 2023). LLM-Coordination diagnoses environment understanding, ToM, and planning (Agashe et al., 2025); instruction inference links intentions to plan feasibility and optimality (Saad et al., 2026). SOTOPIA evaluates interactions with private social goals (Zhou et al., 2024). MuMA-ToM studies observers’ social reasoning about embodied agents (Shi et al., 2025). Watch-And-Help and CoELA investigate household collaboration (Puig et al., 2021; Zhang et al., 2024), and EnactToM measures partner beliefs and task completion under controlled information (Juneja et al., 2026). We combine dynamic interaction with solver-verified scheduling, comparing independent interfaces and supplied-state conditions on frozen tasks. Subjective correctness remains separate from actual success.

Verifiable task evaluation. PlanBench checks action sequences and state changes (Valmeekam et al., 2023); ALFWorld connects textual decisions to embodied execution (Shridhar et al., 2021); AgentBench spans interactive tasks (Liu et al., 2024). AgentAD uses LLM agents to generate satellite-scheduling algorithms (Chen et al., 2026). CalBench adds distributed scheduling, private calendars, and displacement-cost optimization (Zou et al., 2026). We retain native constraints and validators to assess perspective-dependent plans, distinguishing personal evidence from world knowledge.

## 3 Problem Formulation

We focus on first-order belief attribution and use under asymmetric information. Two agents can hold diferent beliefs about the same world because they obtain diferent evidence. A model reading their complete story must distinguish what it knows from what each agent knows, retain old states without new evidence, and use the resulting perspectives for decisions.

C-WAH and CalBench provide complementary tests. Household actions change the world and generate observations, allowing evaluation of belief maintenance during sequential coordination. Scheduling has explicit constraints and optimal costs, allowing complete plans to be checked under each person’s information. Both provide taskbased standards independent of the model’s verbal explanation.

Our extensions retain the original environments, goals, and validators. C-WAH builds personal histories from actual routes, physical actions, and delivered messages. CalBench constructs initial calendars and subsequent public or private updates whose final calendars equal the frozen source task. Information conditions change which constraints each person knows, not the original task objective. The evaluation distinguishes recovering personal states, producing decisions correct under those states, and succeeding in the actual world.

This distinction separates predicting another person’s informed choice from coordinating a successful joint action. A plan consistent with an outdated belief can reveal accurate perspective taking even when it fails physically. A coordinator can instead communicate missing information before assigning work. Silently replacing the person’s belief with world truth tests neither operation. In our protocol, subjective planning checks whether the model respects the target perspective, while native collaboration tests whether it can use or change available information to advance a shared goal. Evaluating both prevents correct belief attribution from being confused with endorsement of an ineffective action.

## 4 Evaluation Framework

## 4.1 Why Five Targets?

The decomposition follows information dependencies. A model must understand what happened (world), determine who obtained which evidence (access), and integrate it over time (belief). Evidence and belief difer because an older message can be superseded by observation, while lack of evidence requires retention. It must then use the resulting state to select an action or solve constraints (decision). Actual execution can succeed, fail, or create new evidence (execution). These targets locate behavioral errors, not five internal modules or a dificulty ordering.

## 4.2 Evidence and Reference States

Let $w _ { t }$ be the world, $e _ { 1 : t }$ the recorded events, and R the observation and delivery rules. For agent a, references are

$$
H _ { a , t } = \operatorname { A c c e s s } ( a , e _ { 1 : t } , R ) ,
$$

$$
b _ { a , t } = \mathrm { U p d a t e } ( b _ { a , 0 } , H _ { a , t } , R ) .\tag{1}
$$

(2)

The model reads all events, whereas $H _ { a , t }$ contains only evidence obtained by a. Reference states follow task rules rather than model-generated answers.

C-WAH records native physical graphs, personal observations, action feedback, and message delivery. Its symbolic room-level observations respect container and possession relations. Direct observation can override a conflicting message; a completed unsuccessful search can yield UN-KNOWN. Without new evidence, a belief persists. Main histories contain physical events and public rules, not observer labels or reference beliefs. Rule replay verifies that the text supports the native references. Incidental encounters and observationdriven corrections remain valid events.

In CalBench, five agents broadcast their initial calendars. Two owners then learn that a movable commitment must remain in its slot. PUB-LIC informs the group; PRIVATE informs a shared intermediary. Owners and actual recipients update, while uninformed agents retain previous calendars. Construction is deterministic from source tasks (Appendix I).

Access and belief therefore have diferent reference semantics. In C-WAH, the access target retains positive location assertions a person has obtained, including assertions later superseded by observation. The belief target instead identifies the current state after those events. Remembering an old assertion is correct in the former and can be incorrect in the latter. Separate targets distinguish selecting evidence from resolving its temporal consequences.

## 4.3 Independent Interfaces and Controls

Seven shared interfaces request world/access, narrative beliefs, structured-history beliefs, direct decisions, decisions with textual beliefs, decisions with symbolic beliefs, and beliefs/decisions with supplied access. C-WAH adds spatial facts and public-map routes (Table 5). Each request has an independent context and reference target.

Supplied access explicitly identifies personal evidence while retaining temporal integration. In C-WAH it provides deduplicated positive location assertions, not a complete ordered observation history; unsuccessful searches must still be interpreted from the narrative. In CalBench it provides ordered event identifiers. Supplied beliefs remove state-recovery demands but retain perspectivespecific decisions. Narrative/structured histories and textual/symbolic beliefs test representation changes. Comparisons on the same stories distinguish attribution, maintenance, and constraint use without feeding predicted answers between inter-

faces.

These controls answer progressively narrower questions. If the relevant evidence is explicitly identified, can the model recover the target belief? If that belief is supplied correctly, can it choose an action appropriate to it? If physical movement succeeds, can it recover the perspectives produced by that movement? Independent requests make disagreements observable without an earlier incorrect answer determining every later input. Native continuation complements these controlled snapshots by allowing decisions to change the subsequent history. It measures whether the model can sustain coordination when information acquisition and action selection become its own responsibility.

## 4.4 Task Decisions and Execution

C-WAH fixed decisions specify an actor, object, and goal and ask for the source in that actor’s belief. Native continuation permits assignments, messages, and exploration within the original task budget. Route interventions compare model-selected and supplied public-graph paths from matched native checkpoints, followed by a belief query scored against each resulting trajectory.

CalBench requires each proposer to schedule five meetings among five agents over sixteen slots. Entries are free, movable at a cost, or immovable. Every pair of meetings shares a participant, requiring distinct slots. A subjective solver minimizes displacement cost under the proposer’s calendars. Native replay validates relocations and meetings. Independent optimization and replay must agree on reference costs; tied optima are accepted. The same proposal is checked in the actual world. A subjectively correct plan need not be world-feasible.

## 5 Experimental Setup

Configurations. We evaluate Qwen3.5-4B, 9B, and 27B with thinking disabled, plus Claude Opus 5.5, Gemini 3.1 Pro Preview, and GPT-6.1 Sol. Frozen inputs, references, and semantic scoring are shared. Qwen uses temperature 0, top-p 1, and limits of 4096 output tokens for C-WAH and 12288 for CalBench. Closed services request medium reasoning and 32768 tokens with unspecified sampling controls. Resources and parameters appear in Appendix C.

Tasks and common support. C-WAH has 43 configurations across seven houses, fixed seeds, 218 event-selected nodes, and nine interfaces: 1962 fixed requests per model. Direct and supplied-belief continuations each schedule 43 episodes; two route arms each schedule 36 configurations. Native technical coverage is 158/158 except 9B’s 154/158. Six-model paired comparisons use 40 common continuation configurations and 35 route configurations. Episodes have a 250- step budget including preparation and at most 20 continuation decisions. Three deterministic scripts provide 129 shared reference episodes.

CalBench has 45 uniform-cost and 45 tieredcost tasks, each with PUBLIC/PRIVATE stories and seven interfaces: 1260 requests per model. Complete coverage is 1136, 1015, and 1251 for Qwen-4B/9B/27B, and 1260 for each closed configuration. The main comparison uses 916 common complete request keys. Cross-interface analyses further match stories; public/private comparisons require both variants. All-valid results and per-interface coverage are separately reported.

Matching follows the question being compared. Within a calendar interface, every model answers the same complete request keys. A belief–planning comparison also requires both interfaces for the same story, and a PUBLIC/PRIVATE comparison requires both variants of the same source task. Consequently, these analyses have diferent denominators even though they begin with the same 90 tasks. Native comparisons similarly require both conditions to be technically valid in every configuration. This prevents diferences in observed task composition from being mistaken for diferences between models or input conditions.

Scoring. C-WAH uses English histories and scores person–object entries and full decisions. CalBench uses Chinese histories, requires all 80 entries for a correct agent belief, and all five meetings for a feasible, minimum-cost proposal. Timeouts/truncations receive bounded retries, adopting the first complete output without selecting by correctness. JSON extraction accepts fences and harmless surrounding text without changing content. Missing targets and semantic errors in complete answers remain incorrect. Incomplete outputs afect coverage, not capability scores. Nodes cluster within 43 configurations and calendar agents/stories within 90 source tasks. We report paired counts and configuration means, not significance from repeated entries.

## 6 Results and Analysis

## 6.1 Recovering Personal Evidence

World comprehension and evidence attribution differ (Table 1). Qwen-4B recovers 93.1% of household world entries but 33.8% of personal access entries. Qwen-9B is better at narrative beliefs yet worse at access. Joint evidence confirms that this is not merely a diference in scoring units: Qwen-27B has 192 nodes with fully correct spatial answers but incomplete access answers. Gemini has only one. Personal attribution is a distinct, modeldependent requirement beyond understanding the physical scene.

Supplied access raises household belief accuracy from 49.3% to 60.2% for 4B, 60.3% to 69.0% for 9B, and 62.8% to 91.4% for 27B. It removes only part of the smaller configurations’ dificulty: identifying relevant evidence is insuficient when integrating it into a complete state remains unreliable. Claude improves from 71.2% to 92.3%, whereas Gemini is near ceiling in both conditions.

![](images/ae8a97c4cb7dd96a863a3f1c1b39794af372fd310caeb49de6fa1998cc53f17e.jpg)  
Figure 3: Information boundaries on the same household references: 1446 current, 1159 unknown, and 247 stale entries. Narrative and supplied-access inputs test whether agents’ states remain distinct from world truth. Nodes repeat within 43 configurations.

Figure 3 separates current facts from unknown and stale states. The latter require resisting the reader’s knowledge. Explicit access thus concerns whose evidence controls a belief, not simply remembering more facts. Retention/update analyses further separate temporal demands (Appendix Figure 7): Claude errs on 35.7% of no-new-evidence entries but 7.6% of updates, while GPT’s rates are

C-WAH: fixed nodes
<table><tr><td></td><td>World</td><td>Access</td><td>Narrative belief</td><td>Struct. belief</td><td>Given access belief</td><td>Direct decision</td><td>Given belief decision</td></tr><tr><td>Qwen3.5-4B</td><td>93.1</td><td>33.8</td><td>49.3</td><td>38.6</td><td>60.2</td><td>66.5</td><td>100.0</td></tr><tr><td>Qwen3.5-9B</td><td>88.6</td><td>24.2</td><td>60.3</td><td>49.2</td><td>69.0</td><td>72.5</td><td>99.1</td></tr><tr><td>Qwen3.5-27B</td><td>96.3</td><td>42.3</td><td>62.8</td><td>63.6</td><td>91.4</td><td>82.1</td><td>100.0</td></tr><tr><td>Claude Opus 5.5</td><td>99.0</td><td>75.8</td><td>71.2</td><td>73.4</td><td>92.3</td><td>87.6</td><td>100.0</td></tr><tr><td>Gemini 3.1 Pro Preview</td><td>99.4</td><td>99.4</td><td>99.4</td><td>99.0</td><td>99.9</td><td>86.2</td><td>100.0</td></tr><tr><td>GPT-6.1 Sol</td><td>95.0</td><td>86.5</td><td>94.7</td><td>92.0</td><td>99.6</td><td>83.9</td><td>100.0</td></tr></table>

CalBench: common complete requests across models

<table><tr><td>Model</td><td>World</td><td>Access</td><td>Narrative belief</td><td>Struct. belief</td><td>Direct planning</td><td>Text belief planning</td><td>Symbolic planning</td></tr><tr><td>Qwen3.5-4B</td><td>23.8</td><td>6.3</td><td>3.2</td><td>3.8</td><td>13.2</td><td>30.6</td><td>35.2</td></tr><tr><td>Qwen3.5-9B</td><td>76.7</td><td>20.7</td><td>10.7</td><td>18.9</td><td>52.0</td><td>53.6</td><td>53.5</td></tr><tr><td>Qwen3.5-27B</td><td>99.9</td><td>81.5</td><td>96.2</td><td>96.2</td><td>88.7</td><td>85.9</td><td>91.2</td></tr><tr><td>Claude Opus 5.5</td><td>100.0</td><td>97.2</td><td>100.0</td><td>97.7</td><td>68.9</td><td>77.0</td><td>81.3</td></tr><tr><td>Gemini 3.1 Pro Preview</td><td>99.2</td><td>99.0</td><td>97.0</td><td>99.4</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>GPT-6.1 Sol</td><td>99.9</td><td>98.3</td><td>92.8</td><td>96.0</td><td>81.0</td><td>79.7</td><td>80.7</td></tr></table>

C-WAH: native tasks and post-action perspectives

<table><tr><td>Model</td><td>Direct task success</td><td>Given belief task success</td><td>Model route entries</td><td>Given route entries</td><td>Model route whole table</td><td>Given route whole table</td></tr><tr><td>Qwen3.5-4B</td><td>12.5</td><td>20.0</td><td>54.6</td><td>53.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Qwen3.5-9B</td><td>5.0</td><td>2.5</td><td>61.2</td><td>61.6</td><td>0.0</td><td>0.0</td></tr><tr><td>Qwen3.5-27B</td><td>55.0</td><td>75.0</td><td>62.0</td><td>61.0</td><td>2.9</td><td>0.0</td></tr><tr><td>Claude Opus 5.5</td><td>95.0</td><td>92.5</td><td>93.7</td><td>68.8</td><td>71.4</td><td>0.0</td></tr><tr><td>Gemini 3.1 Pro Preview</td><td>92.5</td><td>97.5</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>GPT-6.1 Sol</td><td>95.0</td><td>100.0</td><td>90.9</td><td>91.4</td><td>62.9</td><td>65.7</td></tr></table>

Table 1: Results on common support across six configurations (%); bold marks column maxima. C-WAH uses 218 fixed nodes (1,426 world entries; 2,852 access/belief entries). CalBench uses 916 common complete requests, with interface denominators in the appendix. Native comparisons pair 40 continuation and 35 route configurations. Belief entries are pooled; whole tables must be entirely correct. Coverage and all-valid results are separate. Columns measure diferent targets.

4.4% and 15.9%. Their residual weaknesses difer despite both benefiting from supplied evidence.

## 6.2 Using Beliefs in Decisions and Action

Fixed source selection is nearly solved with reference beliefs: five configurations achieve 218/218 and 9B achieves 216/218. Correct direct choices need not imply complete perspectives: all 145 correct 4B choices coexist with some incorrect belief entries, as do 157 of 158 for 9B (Figure 4). A decision may depend on only one correct entry. Conversely, Gemini has 26 nodes with a correct entire belief table but an incorrect independent source answer. State recovery and state use must both be tested.

The partition changes how success should be interpreted. For Gemini, 177 correct choices coincide with a fully correct belief table and only 11 with partial errors. For GPT, the corresponding counts are 113 and 70. The same successful decision can therefore accompany substantially diferent state recovery. These are paired observations at the same nodes, not an assumption that a correct choice requires every unrelated entry to be correct.

Native continuation makes the distinction consequential (Figure 5). Reference beliefs raise 27B success from 22/40 to 30/40 but 4B only from 5/40 to 8/40. For 9B, success remains 2/40 and 1/40 despite near-perfect fixed supplied-belief decisions. In each 9B continuation condition, 35/40 episodes issue no assignments. The model instead attempts extensive exploration (1658 and 1712 commands). Answering a specified source question does not establish the ability to select goals, turn information into assignments, or make progress within a decision budget. This is a policy-level observation, not evidence that larger models necessarily perform worse.

![](images/268de1546f998f70ce47a2eceacc49bd08e96158c5b9ef7cf8137be9127d6da9.jpg)  
Figure 4: Correct source choices partitioned by whether the independent narrative belief table is entirely correct, on the same 218 nodes. Correct choices coexist with partial state errors; a task-relevant answer does not certify the entire perspective.

![](images/3a36aa272668afb1c4b084d5a2176f994e636e8c08167a2ce0583e799c93ad6f.jpg)  
Figure 5: Native task success on 40 common configurations. Circles denote direct continuation and diamonds supplied beliefs. Labels show successful episodes. Lines connect input conditions, not time.

Closed configurations complete 37–40 paired episodes, but supplied beliefs do not uniformly help: Claude changes from 38 to 37. Shared scripts provide context on all 43 tasks: belief-first completes 42, world-first 13, and exploration-first 39. Both exploiting personal information and acquiring missing evidence can support collaboration; a true world location alone need not be a useful instruction for an uninformed executor.

## 6.3 Tracking Perspectives After Action

All supplied-route transports succeed on the 35 common configurations. Nevertheless, no 4B, 9B, or 27B post-action belief table is entirely correct in that condition. Their pooled-entry accuracies are 53.0%, 61.6%, and 61.0%, respectively. Successful movement therefore does not demonstrate tracking of what each person learned.

Claude’s configuration-mean belief accuracy is 93.9% after model-selected routes but 69.0% after supplied routes (Appendix Figure 10); Gemini is perfect in both. A route specifies movement, not automatically its informational consequences. Paths can induce diferent observations and histories. Native references expose this dependence while separating transport success from perspective recovery. The 4B model-selected arm also has three unsuccessful transports, which remain valid task failures.

## 6.4 Planning Under Subjective Constraints

On 91 stories complete for both narrative beliefs and direct planning across all models, Claude recovers 455/455 agent beliefs but produces only 309/455 optimal proposals (67.9%). Qwen-27B recovers 440 beliefs and produces 400 optimal proposals. Qwen-9B produces 227 optimal proposals but only 66 complete beliefs. These diferences do not yield one common ranking (Figure 6). A plan can require fewer distinctions than an entire calendar state, while recovering that state does not ensure solving its constraints.

![](images/eb50b19bafbd0bc7837feafe39f31158708b91a986f142ad5c31f00a695dc13a.jpg)  
Figure 6: Belief recovery versus planning on the same 91 stories (455 agent cases per axis). Beliefs require all 80 entries; plans require subjective feasibility and optimality. The dashed line marks equal percentages, not equivalent metrics.

Supplied-state controls use 67 stories complete in all three planning interfaces and all six configurations. For 4B, correct plans rise from 44/335 directly to 107/335 with textual beliefs and 141/335 with symbolic beliefs. State support helps, but downstream dificulty remains. Claude rises from 232 to 259 and 283, whereas 9B scores 182, 177, and 179. Correct input states do not universally remove the limiting demand (Appendix Figure 9). Gemini attains 335/335 in every condition, yet fails three direct household episodes: static optimization and dynamic coordination test diferent abilities.

Feasibility and optimality reveal a further difference hidden by this last comparison. On these same 335 proposals, 9B produces 192 feasible plans directly and 237 with textual beliefs. Yet its optimal count falls from 182 to 177: feasible but suboptimal plans increase from 10 to 60. Thus, a nearly unchanged optimality score conceals better constraint satisfaction alongside more failures to minimize cost. Claude’s feasible counts rise from 270 to 302 under textual beliefs, but 43 of the latter remain suboptimal (Appendix Figure 8). Supplied states can support compliance with known calendars without solving the remaining optimization problem. This distinction matters for interpreting ToM: violating a person’s known constraints, finding an unnecessarily costly feasible plan, and using information they never received are diferent behavioral failures. A task validator makes the first two measurable rather than reducing every unsuccessful proposal to a wrong belief answer.

Public/private comparisons use 29 complete source-task pairs. Among them, 9B and GPT lose all five optimal proposals on seven and five tasks, whereas Gemini remains perfect (Appendix Figure 11). The manipulation changes both who knows the constraints and how many subjective problems must be solved. Copying world calendars illustrates the need for targeted metrics: on common PRIVATE belief cases it scores 98.5% of entries but only 20.0% of complete agent states and zero stale entries. High ordinary entry accuracy can conceal failure to preserve another’s perspective.

Planning scores also require a perspectivesensitive interpretation. Reusing the solver’s world-optimal plan for every agent is subjectively optimal in 466/485 common direct-planning cases (96.1%). Only 19 cases from six source tasks have strictly lower subjective than world-optimal cost. On these cases, Qwen-4B/9B/27B, Claude, Gemini, and GPT score 0, 0, 13, 10, 19, and 10 out of 19, respectively (Appendix N.2). This privileged solver reference measures overlap between scoring targets, not model behavior or an equal-compute baseline. Aggregate planning success alone therefore does not establish use of a personal perspective.

## 7 Discussion

Task-grounded evaluation reveals two gaps. Understanding events does not ensure attributing evidence and maintaining personal beliefs. A correct state or fixed decision does not ensure efective planning and coordination. The 4B/9B results make the latter visible: fixed supplied-belief decisions are nearly perfect, but actual assignments and task completion remain weak. Closed configurations also separate calendar-state recovery from optimization and physical success from post-action beliefs.

These patterns imply diferent support needs. Explicit personal evidence can assist attribution but cannot replace temporal integration. Reference beliefs remove state uncertainty, but models must still preserve subjective constraints, choose goals, and execute productive commands. A single ToM or success score cannot identify which support is needed. Independent state questions, perspectivesensitive plans, and native consequences provide complementary evidence.

The two domains also clarify when information gathering belongs to the task. In scheduling, each proposer is evaluated under a fixed personal calendar: adding an unreceived update changes the problem being solved. In household continuation, communication and exploration are available actions that can legitimately change what an executor knows. The coordinator must decide when current information supports an assignment and when obtaining or transmitting evidence is useful. The common requirement is to respect the information available at the time of the decision, not to keep an agent uninformed indefinitely. This explains why passive belief questions and active coordination should be complementary tests rather than competing definitions of ToM.

For evaluation design, the unit of success should match the claim. Entry accuracy describes local state recovery, whole-agent accuracy tests a complete perspective, subjective feasibility checks known constraints, and optimality checks eficient use of those constraints. Native outcomes then measure whether the resulting policy reaches the shared goal. Reporting these targets together exposes both overinterpretations: a successful action does not establish complete mental-state understanding, and a correct mental-state answer does not establish competent collaboration.

## 8 Conclusion

We extend verifiable household and scheduling tasks with asymmetric information, retaining native constraints and independent references. Across six configurations, belief inference, use in decisions, and task success diverge in both directions. Evaluating collaborative ToM requires checking not only what a model says others believe, but also the plans it constructs from those beliefs and the consequences of executing them. Our controls make the distinction testable on the same source tasks: supplying evidence probes state recovery, supplying beliefs probes state use, and native execution checks whether either translates into task progress.

## Limitations

The evaluation covers six service configurations, 43 household configurations across seven houses, and 90 calendar tasks. Each uses one fixed seed without independent repeated runs. Reasoning budgets, sampling controls, and serving hardware difer. Results establish neither pure parameterscaling efects nor statistical significance across independent runs.

## Ethical Considerations

We use simulated households and controlled calendars, without new human-subject experiments or personal calendar collection. Synthetic messages do not confer real privacy authorization. Task scores cannot establish reliability in real collaboration or replace consent and verification. Resource provenance and licensing are described in Appendix H.

## References

Saaket Agashe, Yue Fan, Anthony Reyna, and Xin Eric Wang. 2025. LLM-Coordination: Evaluating and Analyzing Multi-agent Coordination Abilities in Large Language Models. In Findings of NAACL, pages 8053–8072.

Zhuang Chen, Jincenzi Wu, Jinfeng Zhou, Bosi Wen, Guanqun Bi, Gongyao Jiang, Yaru Cao, Mengting Hu, Yunghwei Lai, Zexuan Xiong, and Minlie Huang. 2024. ToMBench: Benchmarking Theory of Mind in Large Language Models. In Proceedings of ACL, pages 15959–15983.

Jiawei Chen, Yingguo Chen, Duc Truong Pham, Yanjie Song, Jian Wu, Lining Xing, and Yingwu Chen. 2026. A Large Language Model-Based Multi-Agent Framework to Autonomously Design Algorithms for Earth Observation Satellite Scheduling Problem. Engineering, 63:232–251.

Kanishk Gandhi, J.-Philipp Fränken, Tobias Gerstenberg, and Noah D. Goodman. 2023. Understanding Social Reasoning in Language Models with Language Models. In Advances in Neural Information Processing Systems, volume 36, pages 13518– 13529.

Yinghui He, Yufan Wu, Yilin Jia, Rada Mihalcea, Yulong Chen, and Naihao Deng. 2023. Hi-ToM: A Benchmark for Evaluating Higher-Order Theory of Mind Reasoning in Large Language Models. In Findings ofEMNLP, pages 10691–10706.

Gurusha Juneja, Dylan Lu, Saaket Agashe, Parth Diwane, Edward Gunn, Jayanth Srinivasa, Gaowen Liu, William Yang Wang, Yali Du, and Xin Eric Wang. 2026. EnactToM: An Evolving Benchmark for Functional Theory of Mind in Embodied Agents. arXiv preprint arXiv:2605.09826.

Chani Jung, Dongkwan Kim, Jiho Jin, Jiseon Kim, Yeon Seonwoo, Yejin Choi, Alice Oh, and Hyunwoo Kim. 2024. Perceptions to Beliefs: Exploring Precursory Inferences for Theory of Mind in Large Language Models. In Proceedings ofEMNLP, pages 19794–19809.

Hyunwoo Kim, Melanie Sclar, Xuhui Zhou, Ronan Le Bras, Gunhee Kim, Yejin Choi, and Maarten Sap. 2023. FANToM: A Benchmark for Stress-testing Machine Theory of Mind in Interactions. In Proceedings ofEMNLP, pages 14397–14413.

Matthew Le, Y-Lan Boureau, and Maximilian Nickel. 2019. Revisiting the Evaluation of Theory of Mind through Question Answering. In Proceedings of EMNLP-IJCNLP, pages 5872–5877.

Huao Li, Yu Quan Chong, Simon Stepputtis, Joseph Campbell, Dana Hughes, Michael Lewis, and Katia Sycara. 2023. Theory of Mind for Multi-Agent Collaboration via Large Language Models. In Proceedings ofEMNLP, pages 180–192.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. 2024. Agent-Bench: Evaluating LLMs as Agents. In International Conference on Learning Representations.

Xavier Puig, Tianmin Shu, Shuang Li, Zilin Wang, Yuan-Hong Liao, Joshua B. Tenenbaum, Sanja Fidler, and Antonio Torralba. 2021. Watch-And-Help: A Challenge for Social Perception and Human-AI Collaboration. In International Conference on Learning Representations.

Matthew Riemer, Zahra Ashktorab, Djallel Bounefouf, Payel Das, Miao Liu, Justin D. Weisz, and Murray Campbell. 2025. Position: Theory of Mind Benchmarks are Broken for Large Language Models. In Proceedings of ICML, volume 267, pages 82091– 82130.

Fardin Saad, Pradeep K. Murukannaiah, and Munindar P. Singh. 2026. Theory of Mind in Action: The Instruction Inference Task in Dynamic Human-Agent Collaboration. Artificial Intelligence, 361:104621.

Melanie Sclar, Jane Dwivedi-Yu, Maryam Fazel-Zarandi, Yulia Tsvetkov, Yonatan Bisk, Yejin Choi, and Asli Celikyilmaz. 2025. Explore Theory of Mind: program-guided adversarial data generation for theory of mind reasoning. In International Conference on Learning Representations.

Haojun Shi, Suyu Ye, Xinyu Fang, Chuanyang Jin, Leyla Isik, Yen-Ling Kuo, and Tianmin Shu. 2025. MuMA-ToM: Multi-modal Multi-Agent Theory of Mind. In Proceedings of AAAI, volume 39, pages 1510–1519.

Kazutoshi Shinoda, Nobukatsu Hojo, Kyosuke Nishida, Saki Mizuno, Keita Suzuki, Ryo Masumura, Hiroaki Sugiyama, and Kuniko Saito. 2025. ToMATO: Verbalizing the Mental States of Role-Playing LLMs for Benchmarking Theory of Mind. In Proceedings of AAAI, volume 39, pages 1520–1528.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. 2021. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. In International Conference on Learning Representations.

Karthik Valmeekam, Matthew Marquez, Alberto Olmo, Sarath Sreedharan, and Subbarao Kambhampati. 2023. PlanBench: An Extensible Benchmark for Evaluating Large Language Models on Planning and Reasoning about Change. In Advances in Neural Information Processing Systems, volume 36, pages 38975–38987.

Hainiu Xu, Runcong Zhao, Lixing Zhu, Jinhua Du, and Yulan He. 2024. OpenToM: A Comprehensive Benchmark for Evaluating Theory-of-Mind Reasoning Capabilities of Large Language Models. In Proceedings ofACL, pages 8593–8623.

Hongxin Zhang, Weihua Du, Jiaming Shan, Qinhong Zhou, Yilun Du, Joshua B. Tenenbaum, Tianmin Shu, and Chuang Gan. 2024. Building Cooperative Embodied Agents Modularly with Large Language Models. In International Conference on Learning Representations.

Xuhui Zhou, Hao Zhu, Leena Mathur, Ruohong Zhang, Haofei Yu, Zhengyang Qi, Louis-Philippe Morency, Yonatan Bisk, Daniel Fried, Graham Neubig, and Maarten Sap. 2024. SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents. In International Conference on Learning Representations.

Yixin Zhu, Tao Gao, Lifeng Fan, Siyuan Huang, Mark Edmonds, Hangxin Liu, Feng Gao, Chi Zhang, Siyuan Qi, Ying Nian Wu, Joshua B. Tenenbaum, and Song-Chun Zhu. 2020. Dark, Beyond Deep: A Paradigm Shift to Cognitive AI with Humanlike Common Sense. Engineering, 6(3):310–345.

Chelsea Zou, Yiheng Yao, Selena She, Noah Goodman, and Robert D. Hawkins. 2026. CalBench: Evaluating Coordination-Privacy Trade-ofs in Multi-Agent LLMs. arXiv preprint arXiv:2605.09823.

## A A Complete Perspective Example

This example illustrates perspective constraints. It is neither a sampled CalBench task nor a model response. Formal experiments use five agents and total displacement cost across five meetings.

Complete prompt. “A, B, and C need to hold a one-hour meeting attended by all three. Candidate start times are 14:00, 15:00, and 16:00. Each agent must choose the earliest feasible slot according to the information they have obtained. All agents remember their information accurately and retain earlier judgments unless they receive new evidence. Each immediately knows changes to their own calendar. Private messages are known only to their sender and specified recipient. Other people do not automatically learn their contents.

Initially, A, B, and C confirm in a shared group conversation that everyone is free at 14:00, 15:00, and 16:00.

Next, B learns that they are unavailable at 14:00. B privately messages only A: ‘My calendar has changed. I cannot attend at 14:00.’

Then C learns that they are unavailable at 15:00. C privately messages only A: ‘My calendar has changed. I cannot attend at 15:00.’

There are no other messages, relays, or observations. You can read the complete story, but must answer separately using only the information A, B, and C have obtained. Return JSON with a proposals object whose keys are A, B, and C and whose values are their proposed earliest feasible start times. Return only this JSON.”

Reference output.

```json
{"proposals":{
"A":"16:00",
"B":"15:00",
"C":"14:00"}}
```

A receives both updates, while B and C each know only their own change. Only 16:00 is feasible in the real world. Choosing 16:00 for everyone solves the global scheduling problem but does not select the subjectively earliest slot under B’s and C’s local knowledge.

## B Layer-wise Results and Belief Categories

Tables 2 and 3 give the numerators and denominators behind the main table. Table 4 further separates C-WAH current, unknown, and stale reference beliefs. Configurations use the same frozen references. An unknown state is a target to predict correctly, not a missing sample.

![](images/5ebf3e8d158ca923d51f68f8421a73c5bb784f9fbf95af6ae8f5dcb883834b27.jpg)  
Figure 7: Belief retention and updating across C-WAH models. Circles represent 2248 entries that should be retained without new evidence, and squares 170 actual updates. Blue denotes narrative input and green supplied access. The remaining 434 confirmation entries are excluded.

## C Model Settings and Output Targets

Table 6 records the actual request settings. Open and closed configurations share tasks and scoring but difer in reasoning controls. Unspecified sampling parameters are handled by each service and are not equivalent to explicit deterministic decoding.

C-WAH. The opening specifies house connectivity, relevant objects and containers, goal counts, initial state, observation rules, and an independent teaching example. Subsequent history contains actual actions, routes, and communication. Source questions retain the assigned executor, object identifier, and goal, asking for the believed source. Main histories contain neither observer labels nor reference belief tables. Structured conditions encode physical events only. Every executed action, feedback item, and message enters subsequent continuation contexts.

CalBench. The opening defines calendar semantics, displacement costs, meeting participants, message delivery, and retention of earlier beliefs without updates. Beliefs are organized by holder, calendar owner, and slot. Proposals are organized by proposer and meeting identifier. Suppliedbelief conditions show both world calendars and correct subjective calendars, explicitly stating that world truth does not automatically correct an agent’s beliefs. Identical subjective calendars may be grouped, but outputs still list a complete plan for each of the five agents.

Answer extraction and semantic scoring. The common extractor retrieves existing JSON targets from complete responses, accepting fences, surrounding explanations, and separate thinking sections. For CalBench, defects in the work field or outer packaging do not incur an error when the complete answer can be extracted. Missing calendars, meetings, or states are not supplied. Missing target structures and wrong content are recorded separately. Agent belief correctness requires all 80 entries. Scheduling requires five valid meeting slots, native feasibility, and minimum subjective cost.

## D Complete Coverage and Source-task Aggregation

All six configurations complete 1962 fixed C-WAH requests. Native technical coverage is 158/158 for five configurations and 154/158 for Qwen-9B. Accepted jobs have suficient inputs, and accepted route jobs match their handovers. Strict message/no-message pairs were not scheduled. Main native comparisons restrict to 40 configurations with both continuation conditions valid in all models and 35 with both route arms valid. Table 10 reports all valid jobs, including configurations outside this intersection.

<table><tr><td>Metric</td><td>Qwen-4B</td><td>Qwen-9B</td><td>Qwen-27B</td><td>Claude</td><td>Gemini</td><td>GPT</td></tr><tr><td>World</td><td>1327/1426</td><td>1263/1426</td><td>1373/1426</td><td>1412/1426</td><td>1418/1426</td><td>1355/1426</td></tr><tr><td>Access</td><td>964/2852</td><td>690/2852</td><td>1207/2852</td><td>2163/2852</td><td>2836/2852</td><td>2466/2852</td></tr><tr><td>Narrative beliefs</td><td>1407/2852</td><td>1719/2852</td><td>1790/2852</td><td>2031/2852</td><td>2836/2852</td><td>2701/2852</td></tr><tr><td>Structured beliefs</td><td>1102/2852</td><td>1404/2852</td><td>1813/2852</td><td>2094/2852</td><td>2823/2852</td><td>2624/2852</td></tr><tr><td>Given access: beliefs</td><td>1717/2852</td><td>1969/2852</td><td>2606/2852</td><td>2631/2852</td><td>2850/2852</td><td>2840/2852</td></tr><tr><td>Direct decisions</td><td>145/218</td><td>158/218</td><td>179/218</td><td>191/218</td><td>188/218</td><td>183/218</td></tr><tr><td>Given beliefs: decisions</td><td>218/218</td><td>216/218</td><td>218/218</td><td>218/218</td><td>218/218</td><td>218/218</td></tr><tr><td>Spatial facts</td><td>4159/4431</td><td>4146/4431</td><td>4406/4431</td><td>4359/4431</td><td>4431/4431</td><td>4408/4431</td></tr><tr><td>Public-map route</td><td>186/218</td><td>187/218</td><td>218/218</td><td>218/218</td><td>218/218</td><td>218/218</td></tr><tr><td>Given symbolic beliefs</td><td>216/218</td><td>217/218</td><td>218/218</td><td>218/218</td><td>218/218</td><td>218/218</td></tr></table>

Table 2: Exact C-WAH counts on the same 218 nodes. Textual and symbolic supplied beliefs are scored separately.

<table><tr><td>Metric</td><td>Qwen-4B</td><td>Qwen-9B</td><td>Qwen-27B</td><td>Claude</td><td>Gemini</td><td>GPT</td></tr><tr><td>World</td><td>2281/9600</td><td>7365/9600</td><td>9592/9600</td><td>9598/9600</td><td>9520/9600</td><td>9592/9600</td></tr><tr><td>Access</td><td>38/600</td><td>124/600</td><td>489/600</td><td>583/600</td><td>594/600</td><td>590/600</td></tr><tr><td>Narrative beliefs</td><td>26/825</td><td>88/825</td><td>794/825</td><td>825/825</td><td>800/825</td><td>766/825</td></tr><tr><td>Structured beliefs</td><td>33/875</td><td>165/875</td><td>842/875</td><td>855/875</td><td>870/875</td><td>840/875</td></tr><tr><td>Direct planning</td><td>64/485</td><td>252/485</td><td>430/485</td><td>334/485</td><td>485/485</td><td>393/485</td></tr><tr><td>Text beliefs: planning</td><td>225/735</td><td>394/735</td><td>631/735</td><td>566/735</td><td>735/735</td><td>586/735</td></tr><tr><td>Symbolic beliefs: planning</td><td>243/690</td><td>369/690</td><td>629/690</td><td>561/690</td><td>690/690</td><td>557/690</td></tr><tr><td>Given access: beliefs</td><td>14/370</td><td>22/370</td><td>365/370</td><td>359/370</td><td>365/370</td><td>332/370</td></tr><tr><td>Given access: planning</td><td>10/370</td><td>118/370</td><td>338/370</td><td>290/370</td><td>365/370</td><td>240/370</td></tr></table>

Table 3: Exact CalBench counts on common complete support, corresponding to the main table. Supplied access is listed separately because it jointly requests beliefs and proposals, unlike the separate belief and planning interfaces.

<table><tr><td>Metric</td><td>Qwen-4B</td><td>Qwen-9B</td><td>Qwen-27B</td><td>Claude</td><td>Gemini</td><td>GPT</td></tr><tr><td>Narrative: Current</td><td>785/1446</td><td>851/1446</td><td>1156/1446</td><td>1312/1446</td><td>1444/1446</td><td>1335/1446</td></tr><tr><td>Narrative: Unknown</td><td>600/1159</td><td>851/1159</td><td>534/1159</td><td>561/1159</td><td>1159/1159</td><td>1148/1159</td></tr><tr><td>Narrative: Stale</td><td>22/247</td><td>17/247</td><td>100/247</td><td>158/247</td><td>233/247</td><td>218/247</td></tr><tr><td>Structured: Current</td><td>885/1446</td><td>937/1446</td><td>1240/1446</td><td>1222/1446</td><td>1426/1446</td><td>1317/1446</td></tr><tr><td>Structured: Unknown</td><td>200/1159</td><td>450/1159</td><td>466/1159</td><td>738/1159</td><td>1159/1159</td><td>1109/1159</td></tr><tr><td>Structured: Stale</td><td>17/247</td><td>17/247</td><td>107/247</td><td>134/247</td><td>238/247</td><td>198/247</td></tr><tr><td>Given access: Current</td><td>1000/1446</td><td>1324/1446</td><td>1415/1446</td><td>1444/1446</td><td>1444/1446</td><td>1441/1446</td></tr><tr><td>Given access: Unknown</td><td>661/1159</td><td>585/1159</td><td>991/1159</td><td>983/1159</td><td>1159/1159</td><td>1159/1159</td></tr><tr><td>Given access: Stale</td><td>56/247</td><td>60/247</td><td>200/247</td><td>204/247</td><td>247/247</td><td>240/247</td></tr></table>

Table 4: Exact C-WAH counts by reference belief category, using the same frozen entries across configurations.

<table><tr><td>Interface</td><td>Input in addition to task rules</td><td>Output target</td></tr><tr><td>WORLD_ACCESS</td><td>Complete narrative</td><td>World state and personal evidence collections</td></tr><tr><td>BELIEF NL</td><td>Complete narrative</td><td>Each agent&#x27;s beliefs</td></tr><tr><td>BELIEF STRUCT</td><td>Structured event history</td><td>Each agent&#x27;s beliefs</td></tr><tr><td>GIVEN_NL</td><td>World state and textual reference beliefs</td><td>Decisions under personal information</td></tr><tr><td>GIVEN SYMBOL</td><td>World state and symbolic reference beliefs</td><td>Decisions under personal information</td></tr><tr><td>DIRECT</td><td>Complete narrative</td><td>Decisions under personal information</td></tr><tr><td>ACCESS_GIVEN</td><td>Complete narrative and reference access</td><td>Beliefs and decisions under personal information</td></tr></table>

Table 5: Shared interfaces. C-WAH decisions specify the believed source for a fixed assignment. CalBench decisions give a complete five-meeting proposal for each of five agents. Supplied states come from the evaluator, not model answers to other interfaces.

<table><tr><td>Configuration</td><td>Reasoning control</td><td>C-WAH limit</td><td>CalBench limit</td><td>Temperature / top-p</td></tr><tr><td>Qwen3.5-4B, 9B</td><td>Thinking off</td><td>4096</td><td>12288</td><td>0/1</td></tr><tr><td>Qwen3.5-27B</td><td>Thinking off</td><td>4096</td><td>12288</td><td>0/ 1</td></tr><tr><td>Claude Opus 5.5</td><td>Medium</td><td>32768</td><td>32768</td><td>Unspecified</td></tr><tr><td>Gemini 3.1 Pro Preview</td><td>Medium</td><td>32768</td><td>32768</td><td>Unspecified</td></tr><tr><td>GPT-6.1 Sol</td><td>Medium</td><td>32768</td><td>32768</td><td>Unspecified</td></tr></table>

Table 6: Evaluated configurations and output limits in tokens. Closed configurations request the recorded model names and medium reasoning through a compatible API. Equal reasoning computation across services is not assumed.

CalBench complete counts are 1136, 1015, and 1251 for Qwen-4B/9B/27B and 1260 for each closed configuration. The 916 common keys align models within each interface. Table 7 separately reports all complete outputs, and Table 9 gives interface coverage. Table 8 requires all five agents to be correct in both stories of a source task. Ten agent instances do not replace one source task.

## E Task Semantics of Planning Errors

Figure 8 decomposes outcomes for 67 stories complete across three planning interfaces and six configurations. Subjective optimality, feasible but suboptimal plans, subjective infeasibility, and incomplete structure are mutually exclusive. Each row contains 335 agent proposals. Recoverable fences and work-field packaging are excluded from the last category. Suboptimality means constraints are met but cost minimization fails; subjective infeasibility violates constraints known to the target agent.

## F Matched-story Supplied-belief Controls

Each main-table interface uses its common complete set across models. Table 11 and Figure 9 additionally hold 67 stories fixed across all three planning interfaces and six configurations. Textual and symbolic conditions both supply correct personal states but difer in representation and reading demands.

## G Joint Decisions, Routes, and Communication

Figure 4 in the main text connects correct source choices to independent belief judgments on 218 nodes. Figures 10 and 11 extend the analysis to post-action beliefs and communication conditions, using 35 matched route configurations and 29 complete source-task pairs. Each comparison retains its own outcome and statistical unit.

## H Resources and Computational Scope

The experiments use the C-WAH distribution associated with Zhang et al. (2024), VirtualHome’s Linux v2.3.0 executable, and the public uniform/varied CalBench fixtures of Zou et al. (2026). The frozen releases used here are the household protocol dated October 5, 2026 and calendar story package dated October 9, 2026. Calendar selection uses 45 tasks from each public fixture, not all 180 upstream tasks. Prompts, source identifiers, seeds, reference states, semantic scorers, and native receipts identify the evaluated material.

The Qwen checkpoints are the unquantized BF16 releases Qwen3.5-4B, Qwen3.5-9B, and Qwen3.5-27B; all three model cards specify Apache-2.0. The 4B service uses one RTX 3090 and vLLM 0.26.0. The 9B and reused 27B services run on Ascend infrastructure. Native simulation runs on a Linux host with RTX A6000 hardware, using Python 3.8; the collection environment uses Python 3.11. No training is performed.

The logical evaluation budget is 3222 fixed requests and 158 native jobs per configuration, with up to 20 decisions per continuation. Actual calls depend on trajectories and bounded retries.

The task implementations are available from the C-WAH source repository and the CalBench source repository.

## I Task Construction and Node Sampling

## I.1 C-WAH Native Prefixes and Checkpoints

The formal set retains 43 supported configurations across seven houses and their fixed seeds. Each configuration executes a native preparation trajectory of at most 120 steps. Failed-action feedback and actual states remain in the record. Nodes are selected at six event types: first possession, first placement, first incidental belief update, before a message, after a message, and a stable state. Only the first occurrence of each type is retained, with multiple types at one step merged. Sampling precedes model calls and yields 218 nodes, each with nine interfaces, totaling 1962 requests.

<table><tr><td>Metric</td><td>Qwen-4B</td><td>Qwen-9B</td><td>Qwen-27B</td><td>Claude</td><td>Gemini</td><td>GPT</td></tr><tr><td>Complete-output coverage</td><td>1136/1260</td><td>1015/1260</td><td>1251/1260</td><td>1260/1260</td><td>1260/1260</td><td>1260/1260</td></tr><tr><td>Narrative: complete beliefs</td><td>26/895</td><td>88/830</td><td>864/900</td><td>895/900</td><td>875/900</td><td>836/900</td></tr><tr><td>Structured: complete beliefs</td><td>33/900</td><td>165/875</td><td>867/900</td><td>880/900</td><td>895/900</td><td>865/900</td></tr><tr><td>Direct planning</td><td>107/815</td><td>281/545</td><td>768/895</td><td>582/900</td><td>900/900</td><td>669/900</td></tr><tr><td>Text beliefs: planning</td><td>255/855</td><td>424/785</td><td>759/885</td><td>688/900</td><td>900/900</td><td>712/900</td></tr><tr><td>Symbolic beliefs: planning</td><td>281/810</td><td>400/775</td><td>801/890</td><td>713/900</td><td>900/900</td><td>729/900</td></tr><tr><td>Given access: planning</td><td>18/765</td><td>133/415</td><td>733/885</td><td>654/900</td><td>890/900</td><td>564/900</td></tr></table>

Table 7: CalBench scores and coverage over all complete outputs. The main table compares the common support of all six configurations.
<table><tr><td>Metric</td><td>Qwen-4B</td><td>Qwen-9B</td><td>Qwen-27B</td><td>Claude</td><td>Gemini</td><td>GPT</td></tr><tr><td>Narrative beliefs</td><td>0/76</td><td>0/76</td><td>70/76</td><td>76/76</td><td>73/76</td><td>60/76</td></tr><tr><td>Structured beliefs</td><td>0/85</td><td>2/85</td><td>79/85</td><td>81/85</td><td>84/85</td><td>72/85</td></tr><tr><td>Direct planning</td><td>2/29</td><td>10/29</td><td>23/29</td><td>9/29</td><td>29/29</td><td>17/29</td></tr><tr><td>Text beliefs: planning</td><td>6/63</td><td>18/63</td><td>44/63</td><td>33/63</td><td>63/63</td><td>41/63</td></tr><tr><td>Symbolic beliefs: planning</td><td>5/53</td><td>15/53</td><td>40/53</td><td>35/53</td><td>53/53</td><td>34/53</td></tr></table>

Table 8: Strict CalBench source-task accuracy. All five agents must be correct in both PUBLIC and PRIVATE. Denominators count common source tasks with complete outputs in both conditions.

<table><tr><td>Metric</td><td>Qwen-4B</td><td>Qwen-9B</td><td>Qwen-27B</td><td>Claude</td><td>Gemini</td><td>GPT</td></tr><tr><td>WORLD_ACCESS</td><td>128/180</td><td>170/180</td><td>180/180</td><td>180/180</td><td>180/180</td><td>180/180</td></tr><tr><td>BELIEF_NL</td><td>179/180</td><td>166/180</td><td>180/180</td><td>180/180</td><td>180/180</td><td>180/180</td></tr><tr><td>BELIEF_STRUCT</td><td>180/180</td><td>175/180</td><td>180/180</td><td>180/180</td><td>180/180</td><td>180/180</td></tr><tr><td>GIVEN_NL</td><td>171/180</td><td>157/180</td><td>177/180</td><td>180/180</td><td>180/180</td><td>180/180</td></tr><tr><td>GIVEN_SYMBOL</td><td>162/180</td><td>155/180</td><td>178/180</td><td>180/180</td><td>180/180</td><td>180/180</td></tr><tr><td>DIRECT</td><td>163/180</td><td>109/180</td><td>179/180</td><td>180/180</td><td>180/180</td><td>180/180</td></tr><tr><td>ACCESS_GIVEN</td><td>153/180</td><td>83/180</td><td>177/180</td><td>180/180</td><td>180/180</td><td>180/180</td></tr></table>

Table 9: Complete-output coverage by CalBench interface (180 scheduled per interface). Incomplete outputs are excluded from capability scores.

<table><tr><td>Metric</td><td>Qwen-4B</td><td>Qwen-9B</td><td>Qwen-27B</td><td>Claude</td><td>Gemini</td><td>GPT</td></tr><tr><td>Native valid coverage Direct</td><td>158/158</td><td>154/158</td><td>158/158</td><td>158/158</td><td>158/158</td><td>158/158</td></tr><tr><td>task success Given belief</td><td>5/43</td><td>2/42</td><td>23/43</td><td>40/43</td><td>39/43</td><td>40/43</td></tr><tr><td>task success Model route</td><td>8/43</td><td>1/41</td><td>32/43</td><td>39/43</td><td>41/43</td><td>42/43</td></tr><tr><td>entries Given route</td><td>264/484</td><td>295/484</td><td>299/484</td><td>453/484</td><td>484/484</td><td>440/484</td></tr><tr><td>entries Model route</td><td>256/484</td><td>292/474</td><td>294/484</td><td>333/484</td><td>484/484</td><td>442/484</td></tr><tr><td>whole table Given route</td><td>0/36</td><td>0/36</td><td>1/36</td><td>25/36</td><td>36/36</td><td>22/36</td></tr><tr><td>whole table</td><td>0/36</td><td>0/35</td><td>0/36</td><td>0/36</td><td>36/36</td><td>23/36</td></tr><tr><td>Transport: model route</td><td>33/36</td><td>36/36</td><td>36/36</td><td>36/36</td><td>36/36</td><td>36/36</td></tr><tr><td>Transport: given route</td><td>36/36</td><td>35/35</td><td>36/36</td><td>36/36</td><td>36/36</td><td>36/36</td></tr></table>

Table 10: Exact counts over all technically valid native jobs. Coverage difers by configuration; the main table uses paired common configurations.
<table><tr><td>Metric</td><td>Qwen-4B</td><td>Qwen-9B</td><td>Qwen-27B</td><td>Claude</td><td>Gemini</td><td>GPT</td></tr><tr><td>Direct narrative</td><td>44/335</td><td>182/335</td><td>299/335</td><td>232/335</td><td>335/335</td><td>271/335</td></tr><tr><td>Given textual beliefs</td><td>107/335</td><td>177/335</td><td>280/335</td><td>259/335</td><td>335/335</td><td>261/335</td></tr><tr><td>Given symbolic beliefs</td><td>141/335</td><td>179/335</td><td>302/335</td><td>283/335</td><td>335/335</td><td>283/335</td></tr></table>

Table 11: Matched-story planning controls: 67 stories complete in all three interfaces and all six configurations, with five proposals per story.

![](images/2d47b1c209d201d04077a9b3563512d86f56485c3d3eff42151fccfd198aceb0.jpg)  
Figure 8: Planning outcomes on matched stories. Direct, textual-belief, and symbolic-belief rows each contain 335 proposals. Incomplete structure means the original answer lacks a scorable plan, excluding recoverable fences and surrounding prose.

![](images/9b8754c26e0fb44ddf9e7925422df45fbb675db9d9b3fae305081e002f478da8.jpg)  
Figure 9: Supplied-belief controls on the same 67 stories and 335 agent proposals. All conditions target complete subjectively optimal schedules.

![](images/1920a669f59a296c8c73688c739b38d0467a2ac1250f6981e38172b41b8f399d.jpg)  
Figure 10: Post-action belief accuracy, averaged equally over 35 common route configurations. Circles denote model-selected routes and diamonds supplied routes. Main-table entry accuracy instead pools all 474 entries.

![](images/3cd6c7083d92a22bc6a1130af240ed114839f5fec423cb804a1a8fa45b9049b2.jpg)  
Change in optimal agents (PRIVATE - PUBLIC)  
Figure 11: PRIVATE minus PUBLIC in the number of optimal agents on 29 common source-task pairs. Cells count tasks. Zero change need not mean correctness. Agents are repeated measurements, not independent tasks.

Continuation starts from the same preparation state within each configuration. Preparation and continuation share a 250-step budget, and the model coordinator makes at most 20 decisions. The two model conditions receive the full narrative with or without additional current beliefs. Belieffirst, world-first, and exploration-first scripts provide executable strategy references. Action feedback, message receipts, and personal observations enter later history. Information obtained through exploration can change the next decision. Merely proposing exploration or a message does not yet change anyone’s evidence.

<table><tr><td>Script</td><td>Success/43</td><td>Goals met</td><td>Steps</td></tr><tr><td>Belief-first</td><td>42</td><td>189/190</td><td>4710</td></tr><tr><td>World-first</td><td>13</td><td>122/190</td><td>4525</td></tr><tr><td>Exploration-first</td><td>39</td><td>186/190</td><td>5369</td></tr></table>

Table 12: Shared deterministic references. Each strategy uses the same tasks and native preparation states for 43 episodes. Step totals include preparation. Scripts are not counted again for each model configuration.

The route experiment selects a held object with an unfinished goal at a native checkpoint. It compares a route proposed on the public house graph with a supplied graph route. Of the 43 configurations, six lack a public path connecting the held object’s room to its goal room, and one lacks a valid held object with an unfinished goal. The 36-configuration route set is therefore fixed before model calls. Two conditions per configuration schedule 72 jobs and 108 logical model requests; accepted coverage is reported in Table 10.

Both conditions start from matched checkpoints: task, world relations, person rooms, relevant physical graph, belief values, and evidence timing must agree. Coordinate tolerance for people and the target object is 0.001. Native prefix replay must pass these and input-suficiency checks before a model request. Each logical request adopts one complete answer. Six-model paired analysis uses the 35 configurations valid in both arms for every model. Supplied routes are shortest paths on the public graph. Incidental encounters and observations along them update later beliefs.

## I.2 CalBench Story Construction

Frozen source files contain 45 uniform-cost and 45 tiered-cost tasks, with 30 each at easy, medium, and hard dificulty. Each retains five agents, sixteen slots, five meetings, native commitment costs, meeting participants, and its seed. Within each cost group, tasks are indexed by source-file order from i = 0. Change owners are i mod 5 and (i + 1) mod 5, and the intermediary is (i + 2) mod 5. Roles are fixed before model calls.

For each owner, immovable commitments in the final calendar are listed in slot order. We select index ⌊i/5⌋ mod k, where k is the owner’s number of immovable commitments. Initially, the story makes these two commitments movable, preserving their original costs and all other entries. Each owner later learns that the commitment must remain in its original slot and sends an update. The final world calendar equals the source task.

Events E01–E05 broadcast initial calendars. E06 and E08 are the two private changes, and E07 and E09 their messages. PUBLIC delivers both messages to all agents. PRIVATE delivers each only to its sender and the common intermediary. The variants share physical changes and difer in recipients. Each task has two stories, with independent requests for seven interfaces. In PRIVATE, the two owners and intermediary know diferent updates, while the other two agents retain their initial calendars.

## J Native Evidence and Reference Belief Rules

## J.1 C-WAH Observations, Messages, and Search

Personal observations are sampled only at recorded joint-step boundaries. A person occupies the room recorded at that boundary. Their native symbolic local view contains rooms and entities connected to it through INSIDE relations, except entities whose immediate INSIDE parent is not a room and lacks the OPEN state. Objects held by that person are included. The rule checks the immediate parent rather than recursively simulating visual occlusion.

A location relation supplies evidence only when both endpoints are visible. Possession takes precedence over storage, and storage uses the immediate parent rather than its room. Successful manipulation reveals the resulting object location to its actor. Co-presence in a room does not automatically reveal closed-container contents. ON and IN-SIDE are distinct. Physical feedback to the coordinator is not automatically delivered to executors, and imagined intermediate positions generate no observations.

Beliefs update through personal actions, actual observations, and delivered messages. A received assertion can induce a false belief. In the same exchange, direct observation overrides a conflicting message. A later assertion can update an older belief. A completed search that finds the object absent rules out the believed source. Without another learned source, belief becomes UNKNOWN. UN-KNOWN means the source is undetermined, not that the object does not exist.

Access scoring retains each person’s positive location assertions for each object, including superseded assertions, with duplicates removed. Negative search outcomes are excluded from this set. Belief is the current reference state after temporal updating. Access asks which location evidence the person has obtained, while belief asks where they now think the object is.

## J.2 CalBench Reference States and Subjective Planning

The initial five broadcasts give everyone the five initial calendars. A personal change updates only its owner. Group messages update the corresponding entries for all recipients, while private messages update only sender and recipient. Messages are true when sent. Agents remember accurately

and retain earlier values without new evidence.   
References follow chronological event replay.

CalBench access targets are ordered lists of original event identifiers. Sending, receiving, and personal changes count. A later relay does not additionally grant access to an original event the agent did not experience. These lists difer from C-WAH’s deduplicated sets of location assertions, so access scores are not pooled across domains.

Supplied-belief conditions show actual and correct subjective calendars with their roles distinguished. Identical subjective calendars are grouped, allowing one solution to be reused for proposers with the same perspective. PUBLIC has one shared perspective, whereas PRIVATE may require several. The communication comparison therefore also changes the number of planning problems and reading demands.

## K Formal Prompts and Output Protocol

## K.1 CalBench Private-communication Prompt

The following is an English translation of the complete SYSTEM and USER messages for DIRECT on tiered-cost task b032. The experiment used the Chinese original. Figure 1 shows the actual scheduling response and native validation. Numerical values in the output template illustrate structure rather than the answer. Messages are grouped by role and JSON is indented for readability.

## SYSTEM (English translation)

Solve the problem step by step, strictly respecting its information boundaries. Return one JSON object with exactly two outer fields: work is a string array recording the calculations, and answer is the final answer object. Write work before answer. Do not add Markdown fences or text outside JSON. The field names, nesting, and value types in answer must match the template. Write arrays compactly.

## USER (English translation)

There are five agents, A, B, C, D, and E. Each calendar has sixteen equal-length slots numbered 0–15. Each meeting occupies one slot.

A free slot costs 0. A movable commitment can be moved to another free slot in the same person’s calendar at its specified one-time cost. An immovable commitment cannot be moved.

All five meetings below must be scheduled. Meetings sharing a participant cannot overlap. Meetings with disjoint participants may occupy the same slot.

Every agent must propose a complete schedule for all five meetings, minimizing the total additional displacement cost across participants under that agent’s subjective calendars.

These are independent proposals, not an executed joint decision. Proposals may difer and do not update one another’s beliefs.

The executor moves movable commitments occupying meeting slots to free slots that the person does not use for any meeting. Each calendar has enough initial free slots for these moves. The model only needs to choose meeting times.

Everyone immediately knows changes to their own calendar. Group messages reach all five agents. Private messages are visible only to their sender and specified recipient and are not automatically forwarded.

Messages are true and accurate when sent. Each person remembers prior information accurately and updates the same fact using later information. Without new information, prior judgments persist.

Your access to the complete material does not make every fact available to every agent. There are no changes or communications other than those specified.

Calendar output uses sixteen-entry arrays indexed by slots 0–15. Zero means free, a positive integer is the displacement cost, and the string ”B” means immovable.

## All meetings

Meeting 1: participants B, D, E.   
Meeting 2: participants A, C, E.   
Meeting 3: participants A, C, D.   
Meeting 4: participants B, C, E.   
Meeting 5: participants A, B, D.

## Complete story in chronological order

E01. A says in the five-person group conversation: “My calendar: slot 0 has a movable commitment costing 1; slot 1 has a movable commitment costing 100; slot 2 has a movable commitment costing 10; slot 3 is free; slot 4 has an immovable commitment; slot 5 is free; slot 6 has a movable commitment costing 1; slot 7 has an immovable commitment; slot 8 has a movable commitment costing 100; slot 9 has a movable commitment costing 1; slots 10 and 11 are free; slot 12 has a movable commitment costing 1; slot 13 has a movable commitment costing 10; slot 14 has a movable commitment costing 100; slot 15 is free.”

E02. B says in the five-person group conversation: “My calendar: slot 0 has an immovable commitment; slot 1 has a movable commitment costing 10; slot 2 is free; slot 3 has an immovable commitment; slot 4 has a movable commitment costing 1; slot 5 is free; slot 6 has a movable commitment costing 10; slot 7 is free; slot 8 has a movable commitment costing 1; slot 9 is free; slot 10 has an immovable commitment; slot 11 has a movable commitment costing 10; slots 12, 13, and 14 are free; slot 15 has a movable commitment costing 1.”

E03. C says in the five-person group conversation: “My calendar: slot 0 has a movable commitment costing 1; slot 1 has an immovable commitment; slot 2 is free; slot 3 has a movable commitment costing 1; slot 4 has a movable commitment costing 100; slots 5 and 6 have immovable commitments; slot 7 is free; slot 8 has a movable commitment costing 10; slot 9 has a movable commitment costing 100; slot 10 has a movable commitment costing 10; slot 11 has a movable commitment costing 100; slot 12 has a movable commitment costing 1; slot 13 is free; slots 14 and 15 have immovable commitments.”

E04. D says in the five-person group conversation: “My calendar: slot 0 has an immovable commitment; slot 1 has a movable commitment costing 10; slot 2 is free; slot 3 has an immovable commitment; slots 4 and 5 are free; slot 6 has a movable commitment costing 100; slots 7 and 8 are free; slot 9 has a movable commitment costing 100; slot 10 has a movable commitment costing 1; slot 11 has an immovable commitment; slots 12 and 13 are free; slot 14 has an immovable commitment; slot 15 has a movable commitment costing 100.” E05. E says in the five-person group conversation: “My calendar: slot 0 has an immovable commitment; slots 1 and 2 have movable commitments costing 10; slot 3 is free;

slots 4 and 5 have movable commitments costing 1; slot 6 is free; slot 7 has a movable commitment costing 10; slots 8 and 9 are free; slot 10 has an immovable commitment; slots 11 and 12 have movable commitments costing 1; slot 13 has a movable commitment costing 10; slot 14 is free; slot 15 has a movable commitment costing 100.”

E06. B receives notice that the slot-8 commitment must occur at its original time and is now immovable. Other slots are unchanged.

E07. B privately tells D: “My slot-8 commitment is now immovable. Other slots are unchanged.”

E08. C receives notice that the slot-0 commitment must occur at its original time and is now immovable. Other slots are unchanged.

E09. C privately tells D: “My slot-0 commitment is now immovable. Other slots are unchanged.”

## Question

For each agent, solve for the minimum-total-cost schedule of all five meetings under that agent’s subjective calendars. Any tied optimum is acceptable. The final answer object maps A–E to objects mapping meeting identifiers ”1”–”5” to integer slots 0–15.

## Joint scheduling method

Every meeting has three participants, and any pair of meetings shares at least one. Thus, in any one agent’s proposal, the five slots for meetings 1–5 must be distinct. Diferent agents independent proposals can use the same slots.

Solve separately for each proposer using only the five calendars that proposer believes. Do not overwrite unknown changes with the true calendar.

1. For every meeting m and slot s, inspect each of the three participants. If any entry is ”B”, the candidate is infeasible. Otherwise, candidate cost c(m, s) is the sum of the three entries, with free slots costing 0.

2. Jointly select five distinct slots to minimize the sum of c(m, s) across meetings. Do not choose each meeting’s cheapest slot independently and simply combine them. For conflicts, compare the total costs of moving diferent meetings to alternatives, including changes to several meetings at once.

3. First obtain a feasible assignment with five distinct slots, then compare alternatives. The sum of independent minimum meeting costs is a lower bound. A conflict-free assignment attaining it is optimal. Otherwise compare the extra joint cost induced by conflicts. Do not treat the first feasible assignment as optimal.

4. Before output, verify that each proposal contains exactly meetings 1–5, that the five values are distinct, and that no participant has an immovable commitment at their meeting time. Check the total cost. Proposers with identical subjective calendars may reuse one complete optimal schedule.

## Independent teaching example, not this task’s calendars or answer

Three mutually conflicting meetings X, Y, Z have the following total displacement costs at candidate slots 0, 1, 2, 3:

X: [0, 1, 9, 9]; Y: [0, 5, 9, 9]; Z: [9, 9, 0, 2]. Putting both X and Y at 0 is invalid. X=0, Y=1, Z=2 is valid but costs 5.

X=1, Y=0, Z=2 costs 1 and is optimal. X and Y cannot both use 0, so at least one must pay at least 1, and this assignment attains that bound. Recompute using this task’s calendars. Do not copy the teaching example’s slots.

In work, write the actual calculations in this order: establish the diferences among subjective calendars; for each distinct subjective calendar, list feasible candidates for meetings 1–5 as (slot: sum of the three participant costs), excluding infeasible slots; compare joint assignments to five distinct times and give the chosen total cost. Do not merely claim that a plan was checked or is optimal. Compute each identical subjective calendar once and reuse it across proposers. Do not enumerate every permutation. Prioritize conflicts and reassignment among low-cost candidates.

## Complete output structure template

"work": [   
"Record derivations and   
calculations here"   
],   
"answer": {

```jsonl
"A": {"1": 0, "2": 1, "3": 2,
"4": 3, "5": 4},
"B": {"1": 0, "2": 1, "3": 2,
"4": 3, "5": 4},
"C": {"1": 0, "2": 1, "3": 2,
"4": 3, "5": 4},
"D": {"1": 0, "2": 1, "3": 2,
"4": 3, "5": 4},
"E": {"1": 0, "2": 1, "3": 2,
"4": 3, "5": 4}
}
```

The template illustrates structure and types only. Its numerical examples are not this task’s answer. Solve again and fill in the actual values. The only outer fields are work and answer, containing derivations and the answer, respectively. Calendar arrays must have sixteen entries. Meeting slots are integers, such as 3 rather than ”3”. Do not add costs, explanations, or other fields to answer.

## K.2 C-WAH Public Rules and Teaching Example

Houses, objects, and complete physical histories vary with the frozen node. The following reproduces the shared English SYSTEM rules and complete teaching dialogue of formal WORLD\_ACCESS requests. Node histories contain successive physical boundaries, supplied as described in the main text.

## SYSTEM

You read a household story about Alice and Bob. The narrator can know physical facts that neither person has learned. Each person uses only their own successful actions, actual observations, and messages they actually receive. Nobody has additional object-location memories before the opening story. Moving an object does not update an absent person’s belief. A received assertion can create a false belief. Direct observation takes precedence over a conflicting message during the same exchange; a later delivered assertion can update an older belief. A complete check that rules out the believed source leaves UNKNOWN unless another source has been learned. UNKNOWN is not absence of the object.

ASSIGN describes a plan using the assigned person’s existing belief, not a transmission of new location information. The source must equal that person’s current believed location. SEND communicates a location assertion supported by the sender’s belief. A proposed or future communication has not updated the recipient yet. Cognition-consistent plans are not necessarily eficient or physically successful. Return only the requested JSON, with the supplied semantic identifiers. Do not explain your answer.

The initial map and each physical report are coordinator information, not messages to the people. Personal evidence is sampled only at the recorded joint step boundaries, not at imagined intermediate positions. At a boundary each person occupies their recorded room. In the native symbolic observation rule, their local view includes all rooms and entities connected through INSIDE relations to their current room, except an entity whose immediate INSIDE parent is not a room and does not have state OPEN. Objects held by that person are also included. The rule checks the immediate parent, not photograph-level occlusion or recursive visual geometry. A location relation supplies evidence only when both its endpoints are included in that local view. Held-by relations take precedence over storage relations; for storage, the immediate parent takes precedence over its room. A successful manipulation supplies its actor with the resulting item location. Room co-presence alone does not reveal the contents of a closed container. An ON relation and an INSIDE relation are diferent. Only actual delivered message receipts update recipients. Unmentioned physical relations persist until removed or changed. A completed search rules out that searched source when its physical item is elsewhere. No one has extra pre-story memories.

## USER

The library and study are connected. Alice and Bob initially occupy the library. book\_1 is ON shelf and INSIDE library; shelf is INSIDE library; bench is INSIDE study. At boundary 1 Alice is in study and Bob remains in library. At boundary 2 Bob successfully grabs book\_1 in library. At boundary 3 Bob and book\_1 are in study with Alice; Bob holds book\_1. At boundary 4 Alice returns to library; Bob remains in study holding book\_1. At boundary 5 Bob successfully puts book\_1 ON bench and INSIDE study; Alice stays in library.

Return {”world”:{item:location},”access”: {”Alice”:{item:[locations]},”Bob”:{item: [locations]}}}. Include every listed item. Access lists all positive location assertions personally obtained by observation, own action, or received message, including superseded assertions. Do not include narrator-only facts or negative checks; deduplicate each list.

## ASSISTANT ASSISTANT

```jsonl
{
"world": {"book_1": "bench"},
"access": {
"Alice": {
"book_1": [
"shelf", "carried_by_Bob"
]
},
"Bob": {
"book_1": [
"shelf",
"carried_by_Bob", "bench"
]
}
}
}
```

## K.3 Interface Output Objects

All CalBench interfaces use an outer work string array and answer object. World/access answers contain world and access. The two belief interfaces contain only beliefs. Supplied access contains both beliefs and proposals. Other planning answers directly map each of the five agents to a proposal.

Calendars use 0, positive integer costs, or the string B, in arrays of exactly sixteen entries. Beliefs are nested by holder and calendar owner, with five arrays per holder. Proposals are nested by proposer and meeting identifiers “1”–“5”, with integer slots 0–15. The work field expresses derivations. Content scoring uses the final answer.

C-WAH beliefs specify Alice’s and Bob’s believed sources for all target objects. Fixed AS-SIGN answers preserve the specified actor, object, and goal, supplying only the believed source. Free continuation returns a commands array permitting ASSIGN, SEND, and EXPLORE. Commands for one person execute sequentially, and the two queues run concurrently. Cognitive checks use the state before submission of the entire command group. Undelivered messages cannot justify assignments in that group that depend on them.

## L Scoring and Aggregation

## L.1 Strict State and Plan Metrics

Let C denote scored factual entries for an interface. Entry accuracy is

$$
A _ { \mathrm { c e l l } } = { \frac { 1 } { | { \mathcal { C } } | } } \sum _ { c \in { \mathcal { C } } } \mathbf { 1 } [ { \hat { y } } _ { c } = y _ { c } ] .\tag{3}
$$

A CalBench agent belief is correct only if all 80 entries in its five calendars match. Complete-state accuracy is

$$
A _ { \mathrm { a g e n t } } = \frac { 1 } { | \mathcal { A } | } \sum _ { a \in \mathcal { A } } \prod _ { c \in \mathcal { C } _ { a } } \mathbf { 1 } [ \hat { y } _ { c } = y _ { c } ] ,\tag{4}
$$

where A is the set of scored agent instances in complete responses and $\scriptstyle { { \mathcal { C } } _ { a } }$ its 80 entries. C-WAH access uses exact set matching per person and object, while beliefs use entry matching. A missing extractable target receives zero for the relevant interface metric. Recoverable code fences and surrounding text do not afect scoring.

Each complete CalBench proposal is passed to the native executor. It checks meeting identifiers, integer slots, participant conflicts, and immovable commitments. Movable commitments occupying meeting slots are relocated to free positions outside all of that person’s reserved meeting slots, with every action validated. Feasible plans are optimal when their cost equals the frozen subjective minimum. All tied optima are accepted. The same proposal is separately validated in the actual calendar for world feasibility.

Because every pair of meetings shares a participant, optimization reduces to assigning diferent meetings to diferent slots. Let $P _ { m }$ be the participants in meeting m, and $c _ { a } ( m , s )$ the sum of their displacement costs at slot s as believed by proposer a. A candidate involving an immovable commitment is disabled. We solve

$$
\operatorname* { m i n } _ { z } \ \sum _ { m = 1 } ^ { 5 } \sum _ { s = 0 } ^ { 1 5 } c _ { a } ( m , s ) z _ { m s } ,\tag{5}
$$

$$
\mathrm { s . t . } \quad \sum _ { s } z _ { m s } = 1 , \quad \sum _ { m } z _ { m s } \leq 1 ,\tag{6}
$$

$$
z _ { m s } \in \{ 0 , 1 \} ,\tag{7}
$$

$$
z _ { m s } = 0 \quad \mathrm { f o r d i s a b l e d c a n d i d a t e s } .\tag{8}
$$

The reference optimum is established when CP-SAT, independent assignment optimization, and native replay agree. Task construction verifies sufficient free destination slots.

## L.2 Response Selection and Task-level Aggregation

A complete output requires finish\_reason=stop. An incomplete request permits at most three additional attempts, stopping at the first complete output. A prespecified transport-only supplement allows three further attempts for 61 afected 4B fixed keys, 110 9B fixed keys, and afected decisions in 20 9B native jobs, with absolute attempt counts preserved. A new truncation ends that supplement. Correctness and scorability never determine selection. The common extractor preserves genuinely missing fields, wrong states, infeasibility, and suboptimality. Six-model coverage is reported in Tables 7 and 9.

Scoring proceeds by (1) identifying the unique adopted response by story and interface, (2) extracting its existing target, (3) comparing world, access, and belief references, (4) replaying each proposal and comparing subjective cost and world feasibility, and (5) aggregating PUBLIC and PRI-VATE by source task. Both conditions must be complete for paired analysis. Complete wrong answers remain in denominators. Interface comparisons match stories without selecting on correctness.

C-WAH nodes are repeated observations within configurations. CalBench agents and communication stories repeat within source tasks. Configuration averages first compute accuracy within each configuration and then weight configurations equally. Pooled-entry main-table accuracy divides all correct entries by all scored entries. These weights are not interchanged. We report raw counts, paired diferences, and single-run configuration results, without treating repeated observations as independent trials for significance claims.

## M An Actual Case: Beliefs and Task Verification

This section presents actual Qwen3.5-27B outputs for the PRIVATE story of tiered-cost source task b032: a structured-belief response and the complete proposals in Figure 1. The initial broadcast says B’s slot-8 commitment is movable at cost 1. E06 states that B learns it must remain in its original slot. E07 is B’s private message only to D: “My slot-8 commitment is now immovable. Other slots are unchanged.” A, C, and E do not receive it. The other change makes C’s slot-0 commitment immovable, also reported only to D.

## M.1 Information Attribution in Belief Answers

<table><tr><td>Holder</td><td>Reference</td><td>Model</td><td>Update received</td></tr><tr><td>A</td><td>1</td><td>B</td><td>None</td></tr><tr><td>B</td><td>B</td><td>B</td><td>Own change</td></tr><tr><td>C</td><td>1</td><td>B</td><td>None</td></tr><tr><td>D</td><td>B</td><td>B</td><td>Private message</td></tr><tr><td>E</td><td>1</td><td>B</td><td>None</td></tr></table>

Table 13: Five agents’ beliefs about B’s slot 8 in the actual case. The number 1 is a displacement cost, and the string B means immovable. Model values come directly from the complete structured response.

Actual output excerpt. The following preserves the model’s complete sixteen-entry calendar for A’s belief about B. Positions are zero-indexed.

BELIEF\_STRUCT: output excerpt   
{"A": {"B": [   
"B", 10, 0, "B", 1, 0, 10, 0,   
"B", 0, "B", 10, 0, 0, 0, 1   
]}}

A obtains neither the E06 personal change nor the E07 private message, so slot 8 should remain 1 in A’s reference calendar. The output replaces this entry with world truth for all three uninformed agents. It gets 397 of 400 entries correct, but only B and D have fully correct eighty-entry states. Entry accuracy and complete-perspective accuracy thus provide diferent diagnoses.

## M.2 Complete Plans and Native Verification

DIRECT independently reads the same story and schedules all five meetings for each agent. Participants in meetings 1–5 are BDE, ACE, ACD,

BCE, and ABD. Table 14 gives the adopted complete output and validation from recorded native receipts.

<table><tr><td colspan="3">Agent Meeting slots Cost Opt.</td><td>Belief feasible</td><td>World feasible</td></tr><tr><td>A</td><td>8,3,12,2,5</td><td>14 14</td><td>Yes</td><td>No</td></tr><tr><td>B</td><td>8,3,12,2,5</td><td>15</td><td>No</td><td>No</td></tr><tr><td>C</td><td>8,3,12,2,5</td><td>14 14</td><td>Yes</td><td>No</td></tr><tr><td>D</td><td>12,3,2,7,5</td><td>22 15</td><td>Yes</td><td>Yes</td></tr><tr><td>E</td><td>8,3,12,2,5</td><td>14 14</td><td>Yes</td><td>No</td></tr></table>

Table 14: Complete proposals for b032 PRIVATE. Slots correspond to meetings 1–5. Cost is total displacement under the subjective calendar, and Opt. is the reference minimum.

A, C, and E propose cost-14 plans that attain their subjective optima. B’s identical proposal violates the known slot-8 constraint. D’s plan is feasible but costs 22 rather than the minimum 15. The case separates subjective infeasibility from feasible but suboptimal planning.

Figure 1 highlights A and B. Under A’s calendars, native replay first moves B’s commitment from slot 8 to 7, then schedules meeting 1 at 8. All five meetings require five commitment moves at total cost 14. Under B’s reference calendar, the slot-8 commitment is immovable and this action is rejected. Both proposals fail in the actual world, but A’s remains optimal under A’s information.

Structured-belief and direct-planning responses are independent requests at the same node. The former answer is not passed to the latter. This example distinguishes belief answers, perspectivespecific planning, and actual execution conditions.

## N Scoring Granularity and a Deterministic Reference

## N.1 Copying the World Does Not Establish Perspective Recovery

Assign every agent the actual final calendar as its predicted belief and score against the frozen references. The 180 stories have 72000 belief entries: 71460 agree with the world, while 540 should retain earlier values. Copying scores 99.25% by entry but only 540/900 complete agent states: 450/450 in PUBLIC and 90/450 in PRIVATE.

In PRIVATE, only the intermediary receives both updates. Each change owner lacks the other’s update, and the remaining two agents lack both. Copying the world therefore gets only the intermediary’s complete state correct in each task. Shared facts dominate entry accuracy, whereas complete personal states and stale entries directly test information boundaries. This deterministic reference follows reference states, not an additional model or model output.

![](images/42f2e0656f0b2d795d80aedd3b9f070f7ae3389c75aba30b65ee3ee16baf7a26.jpg)  
Figure 12: Belief granularity on 81 common PRIVATE stories: 32400 entries, 405 complete agent states, and 486 stale entries. Copying the world scores 98.5%, 20.0%, and zero, respectively. The construction-wide counts in the text use all 90 source tasks.

## N.2 World Plans and Perspective-sensitive Planning

For each frozen story, we take its solver-produced world-optimal plan and evaluate that same plan against every agent’s subjective calendars using the original feasibility and optimality checks. Across all 180 stories, it is subjectively feasible in 900/900 cases and optimal in 863/900 (95.9%). On PRIVATE stories, it is optimal in 413/450 (91.8%). These counts concern plan correctness, not recovery of the full personal state.

On the six-model common DIRECT support (97 stories), the world plan is optimal in 466/485 cases. In the remaining 19 cases from six source tasks, subjective optimal cost is strictly below worldoptimal cost, so the world plan cannot satisfy subjective optimality. Table 15 reports models on this strict-cost-separation subset. The subset diagnoses where this particular world-plan reference fails; it does not establish that every successful plan requires an explicit complete belief representation. Source tasks remain the clustering unit.

<table><tr><td>Configuration Optimal proposals</td></tr><tr><td>Qwen3.5-4B 0/19</td></tr><tr><td>Qwen3.5-9B 0/19</td></tr><tr><td>Qwen3.5-27B 13/19</td></tr><tr><td>Claude Opus 5.5 10/19</td></tr><tr><td>Gemini 3.1 Pro Preview 19/19</td></tr><tr><td>GPT-6.1 Sol 10/19</td></tr></table>

Table 15: Subjective optimality on 19 strict-costseparation agent cases from six source tasks within common DIRECT support. All cases retain their original task constraints and scoring.

## N.3 State Representation and Planning

The main table reports narrative and structured beliefs on each interface’s common complete requests. Their complete-agent metric requires every calendar entry, without excluding recoverable JSON packaging. Because calendar coverage difers by interface, these columns are descriptive, not matched estimates of a representation effect. Household narrative and structured conditions have identical full coverage and permit a same-target comparison.

Table 11 compares textual and symbolic supplied beliefs on matched stories. Both provide correct states, leaving the need to construct each agent’s subjective feasible set and minimize cost. Diferences concern representation and downstream use, not direct measurements of internal reasoning mechanisms.