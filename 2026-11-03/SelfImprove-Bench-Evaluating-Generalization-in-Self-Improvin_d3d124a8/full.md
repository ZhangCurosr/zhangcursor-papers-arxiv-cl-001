Generaliaztion Comparision

# SelfImprove-Bench: Evaluating Generalization in Self-Improving Agents

Bangrui Xu<sup>1,2,3∗</sup> Jie Yang<sup>1,2,4∗</sup> Sidi Chen<sup>1,2,3∗</sup> Xuanhe Zhou<sup>1,2†</sup> Jiantao Qiu<sup>4</sup>

Qianyu Yang<sup>3</sup> Zijie Wang<sup>3</sup> Fan Wu<sup>1,2</sup> Zhoufutu Wen<sup>3†</sup>

<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>Theseus Labs <sup>3</sup>Humanlaya <sup>4</sup>Shanghai AI Lab

§ github.com/theseus-labs-rsi/SelfImprove-Bench  theseus-labs-rsi.github.io/SelfImprove-Bench

## Abstract

Recursively self-improving (RSI) systems are increasingly viewed as a promising means of accelerating progress toward artificial general intelligence. They aim to autonomously improve their capabilities by refining their working pipelines and recursively improving the strategies that guide subsequent refinements, with gains generalizing to unseen scenarios. However, existing evaluations lack reliable metrics to distinguish capability gains from scenario fitting, sustainable and extensible protocols for unseen tasks and dificulties, and fine-grained attribution of improvement to evolution traces. To this end, we propose SelfImprove-Bench, an RSI evaluation framework that organizes tasks into three primary capabilities (i.e., Exploring, Planning, Reasoning) as scenarios for guiding model evolution. We instantiated the tasks on games for controllability and reproducibility, and pairs them with scalable generation and orchestration driven by task templates and dificulty coeficients. Beyond task-level scores, SelfImprove-Bench organizes evaluation into generalization ladder levels G0 to $G \textless$ that measure how far gains transfer from training conditions. Experiments reveal that the evolutionary gains of existing RSI methods are inconsistent across generalization levels.

![](images/352a9f066a34df7a659ede903b0cbd852b5443e0f7c3999801c791a279abdd36.jpg)

Model Comparision  
![](images/77a6702c95b12c1e37302423d0c828fd6a4053ceb7bbb5b631b40a990f234319.jpg)

![](images/decd7225dd7c6a666f0fe8f6cbec6f4d70576bef37b6adc500c3f49d3334c433.jpg)  
Figure 1 RSI at a glance. Left and center: representative RSI methods and editable artifacts. Top right: performance across diferent agent configurations. Bottom right: capability retention and generalization across G0–G4.

## 1 Introduction

Despite rapid advances in large language models (LLMs), conventional development still relies heavily on static datasets and ofline training, limiting continuous adaptation to changing environments. Recursive self improving (RSI) agents address this limitation by iteratively modifying their memories, skills, and harnesses using experience and feedback [1–3]. A central question is whether these improvements generalize beyond encountered instances to harder tasks, diferent task types, and ultimately distinct capabilities. Evaluating this progression is essential to understanding RSI as a path toward artificial general intelligence.

Existing benchmarks leave three challenges unresolved (Table 1). (i) Systematic generalization assessment. RSI benchmarks examine selected settings, such as cross-task or cross-dificulty transfer [4–6], but lack a unified taxonomy spanning instances, dificulty levels, tasks, and capabilities. (ii) Scalable RSI evaluation. Although some reinforcement learning benchmarks cover broader generalization settings [7], they lack a unified protocol for general-purpose RSI methods. Fixed task pools and limited dificulty ranges further constrain sustained evaluation as agents evolve and approach performance ceilings [8, 9]. (iii) Fine grained process diagnosis. Existing analyses rely primarily on aggregate metrics, selected case studies, or persistence ablations [10, 11], leaving the relationship between evolution trajectories and generalization outcomes insuficiently understood.

To address these challenges, we introduce SelfImprove-Bench, an RSI evaluation framework combining capability-oriented tasks with a generalization ladder. (i) The ladder assesses progressively broader generalization across tasks targeting planning, reasoning, and exploration. It distinguishes five levels: replay performance on the same instances (G0), unseen instances at the same dificulty (G1), new dificulty levels within the same task (G2), diferent tasks within the same capability category (G3), and tasks across capability categories (G4). (ii) A scalable task generation and orchestration module generates instances using task templates and dificulty coeficients, then assembles sequences with configurable task types, counts, dificulty levels, and replay frequencies. (iii) An agent-driven, fine-grained trace analysis module characterizes refinement semantics, modification scope, capability dimensions, and generalizability to reveal method-specific evolution patterns.

![](images/fd3cdf1b6f1e6103f318b2f1009609331c0a2b9f6432ca73380bcd5ecef9f894.jpg)  
Figure 2 Example RSI-driven refinements. The blue panel shows the performance of diferent agent systems. The red panel illustrates a common failure case. The yellow panels show refinement examples from diferent RSI Methods. The green panel shows RSI performance before and after refinement

Using SelfImprove-Bench, we evaluate three representative classes of RSI methods and observe substantial variation across the generalization ladder, including ranking reversals on harder instances and diferent task types. Note we focus on non-training RSI methods, whose explicit modifications to prompts, memories, skills, and harnesses support a unified evaluation protocol and interpretable evolution traces. Their lower computational requirements also make repeated evaluation more tractable. Experiments across three representative classes of RSI methods reveal substantial diferences along the generalization ladder, including ranking reversals on harder instances and diferent tasks. As illustrated in Figure 1, similar final scores can therefore conceal distinct evolution patterns and transfer ability.

Our main contributions are as follows.

(1) We introduce a capability-oriented generalization ladder for systematically evaluating RSI across instances, dificulty levels, tasks, and capabilities.

(2) We develop configurable task generation and orchestration to support scalable evaluation across the ladder.

(3) We evaluate representative RSI methods and analyze their evolution traces, revealing diferences in improvement efectiveness, generalization, and modification patterns.

## 2 Related Work

## 2.1 RSI Methods

We classify RSI methods into three non-exclusive strategies based on their modification targets:

• Experience-level RSI adapts prompts, memories, and playbooks, as in Reflexion, A-MEM, and ACE [2, 19, 20].

• Harness-level RSI modifies tools, workflows, and executable code, as in SICA, Self-Harness, and AHE [3, 21, 22].

• Meta-level RSI refines candidate generation or selection, as in DGM’s evolving self-modification agent and HyperAgents’ editable meta-agent [1, 23].

Other strategies include reinforcement learning (RL), which typically updates policies using reward feedback within a predefined optimization framework [24]. Data synthesis provides another strategy, seeking to improve model capabilities through generated training data [25].

RSI methods flexibly integrate and organize these strategies into a continual self-improvement process and has been explored as a potential route toward artificial general intelligence (AGI) [26]. RSI evaluation should therefore examine whether improvements achieved through refinement transfer to harder tasks and unseen task types within and across capability categories.

## 2.2 RSI Relevant Benchmarks

RSI-Specific Generalization Evaluation. Common practices include cross-task testing without explicit capability categories organizing refinement and transfer [4, 27], and refinement and evaluation over mixed task streams [5, 6, 11, 28]. Mixed-stream evaluation examines cumulative performance and experience reuse, but alone does not isolate source-to-target transfer. Capability-aware protocols often focus on related-task or within-domain generalization [9, 12, 16]. Capability breakdowns may also serve post-hoc scoring rather than organize refinement and transfer [6]. Method-specific evaluations additionally demonstrate cross-domain transfer, as in ADAS [15].

Dificulty-based task organization does not necessarily include testing whether gains acquired on easier tasks transfer to harder ones [13]. Outside agent RSI, Reasoning Gym provides model-level evaluations of within-domain and cross-domain transfer, alongside dificulty-based evaluations [7]. A unified evaluation of agent RSI that combines capability-based refinement, dificulty generalization, and transfer across task categories remains underexplored.

RSI-Specific Trace Level Analysis. Process analysis commonly illustrates refinement through selected modification cases [18, 29]. Quantitative approaches assess experience contributions through persistence ablations and memory-free comparisons [8, 27], or track capability changes across refinement checkpoints [12]. Finer analyses can depend on specific artifact formats: SkillLearnBench measures skill quality, usage, and alignment with execution trajectories, requiring explicit skill artifacts [10]. Related work also aggregates semantic categories of code modifications and relates them to performance, as in AI4AI-Bench [17]. However, statistical analysis jointly relating generalization performance, modification types, target capabilities, and the semantic content of individual changes remains underexplored.

<table><tr><td>Work</td><td>Quantitative Analysis of RSI Modifications</td><td>Easy-to-Hard Transfer</td><td colspan="2">Cross-Task Transfer Within-Capability</td></tr><tr><td>EvoPathBench [12]</td><td></td><td></td><td></td><td>Cross-Capability</td></tr><tr><td>EvoAgentBench [9]</td><td></td><td></td><td></td><td></td></tr><tr><td>SEAGym [4]</td><td></td><td></td><td></td><td></td></tr><tr><td>SkillLearnBench [10]</td><td></td><td></td><td></td><td></td></tr><tr><td>ServeLearnBench [13]</td><td></td><td></td><td></td><td></td></tr><tr><td>PAST-Bench [8]</td><td></td><td></td><td></td><td></td></tr><tr><td>STOP [14]</td><td></td><td></td><td></td><td></td></tr><tr><td>HyperAgents [1]</td><td></td><td></td><td></td><td></td></tr><tr><td>ADAS [15]</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RRSI [16]</td><td></td><td></td><td></td><td></td></tr><tr><td>AI4AI-Bench [17]</td><td></td><td></td><td></td><td></td></tr><tr><td>RSIBench-Data [18]</td><td></td><td></td><td></td><td></td></tr><tr><td>Reasoning Gym [7]</td><td></td><td></td><td></td><td></td></tr><tr><td>SelfImprove-Bench</td><td> $\checkmark$ </td><td> $\checkmark$ </td><td> $\checkmark$ </td><td> $\checkmark$ </td></tr></table>

Table 1 Comparison of evaluation coverage. Rows are grouped, from top to bottom, into RSI benchmarks, evaluations accompanying RSI methods, and related evaluations outside RSI. SelfImprove-Bench is shown separately. Task families are organized by agent ability. For task-type transfer, refinement uses type $A ,$ while type B is held out until evaluation; Same Ability and Cross Ability indicate whether the two types belong to the same ability-defined family. Easy-to-Hard requires evaluation on harder tasks than those used during refinement. Quantitative RSI Trace Analysis requires statistics over refinement trajectories rather than selected examples.

Tested Tasks. Existing RSI evaluations involve tasks like software engineering, terminal interaction, web automation, expert-level reasoning [4, 30–33], or construct domain-specific tasks for training-algorithm design and financial workflows [6, 17]. However, end-to-end task scores ofer limited capability-specific diagnosis, while fixed task pools may lose discriminative power as methods approach score ceilings.

## 3 SelfImprove-Bench

## 3.1 Evolution Protocol

Definition. Following the definition of RSI in [34], we call a refinement gain self-improvement to the extent that it persists on instances, dificulties, and tasks never exposed during refinement, and task fitting to the extent that it is confined to the training instances.

Task construction and orchestration. Each instance $\tau = G _ { g } ( d , \xi )$ is generated from game $^ { g , }$ dificulty $d$ , and seed ξ. The configuration θ first specify the games to include, then assigns dificulty levels and random seeds to generate their instances. It specifies how many instances enter the held-in and held-out sequences and in order, yielding $S _ { \mathrm { i n } } = ( \tau _ { 1 } , \dots , \tau _ { N } )$ for execution and refinement and $\boldsymbol { S _ { \mathrm { o u t } } } = ( \tilde { \tau } _ { 1 } , \dots , \tilde { \tau } _ { M } )$ for evaluation. Replay settings determine which held-in tasks can recur, how frequently, and how far back they are selected, counting only non-replay tasks. Replays do not trigger refinement. N includes replays. Detailed insertion rules are provided in Appendix A.2. The corresponding task collections $\mathcal { D } _ { \mathrm { i n } }$ and $\mathcal { D } _ { \mathrm { o u t } }$ omit order and replay multiplicity. The generator constructs the held-out pool first and rejects duplicate task-descriptor hashes, excluding task\_id, when generating non-replay held-in tasks (Appendix A.1). Game categorization is described in Section 3.2. Task configurations are provided in Appendix A.1.

We instantiate the framework on games, which satisfy the controllability and reproducibility requirements of RSI evaluation. Our game rules are explicit and state spaces are bounded, making it feasible to map planning, reasoning, and exploring onto diferent games and avoid the confounding of multiple abilities that complicates real-world tasks. Games inherently embody core characteristics of real-world problems such as partial observability, resource constraints, and multi-step decision making. Separating self-improvement from task fitting requires fresh instances and controlled dificulty, which games most readily provide. We use them as a controlled testbed that complements, rather than replaces, real-world RSI benchmarks. In addition, these games are agent tasks mainly based on text interaction and are related to the NLP community.

![](images/6c279f6a21391fcdb7574cad6cb15bd5c3befd4c11dc47441e18ae64e2f97c6b.jpg)  
Figure 3 SelfImprove-Bench Overview. It supports (i) task generation and orchestration targeting primary agentic capabilities to support RSI refinement and (ii) G0–G4 generalization ladder evaluation.

Agent initialization. For each RSI method, the initial task agent is $A _ { 0 } = \left( B , H _ { 0 } \right)$ , comprising a common backbone model B and a method-specific initial harness $H _ { 0 }$ . Backbone settings are reported in Section 4.1.

Refinement and selection. At step t, the current agent $A _ { t - 1 } = \left( B , H _ { t - 1 } \right)$ executes $\tau _ { t } \in S _ { \mathrm { i n } }$ through the game’s action interface, producing an interaction trace and task score. Each task execution, except replays, is followed by refinement using benchmark-provided feedback. During refinement, the method may use tools to inspect and modify its designated workspace, but may not access SelfImprove-Bench modules or directly access, execute, or evaluate task instances. Feedback is supplied only through the prescribed interface (Appendix A.3). When the method proposes candidate harnesses, SelfImprove-Bench evaluates their agents on the same task instance τ and supplies the results, together with the current agent’s result, to the method’s selection procedure. Selection follows the method-specific rule and may retain the current harness. The resulting agent $A _ { t } = \left( B , H _ { t } \right)$ proceeds to the next task. Candidate validation uses held-in tasks only, without accessing $S _ { \mathrm { o u t } }$ . The backbone B remains fixed, and refinement ends when $S _ { \mathrm { i n } }$ is completed.

Evaluation and analysis. Evaluation uses saved system states and task results according to the protocols in Section 3.3. Process analysis extracts individual modification proposals from saved refinement reports and edit records, then annotates and aggregates them using the criteria in Section 3.4. Annotation details are provided in Appendix A.4.

## 3.2 Task Categorization

For rule-constrained tasks with fixed action interfaces, we distinguish three interacting capabilities: Exploration, acquiring task-relevant information through observation and interaction. Reasoning, interpreting rules, deriving implications, and performing computations. Planning, organizing action sequences to achieve goals under constraints. Games are categorized by their primary capability demands, providing context for analyzing modification targets.

## 3.3 Generalization Ladder

The generalization ladder evaluates refinement gain, agent transfer, and refinement-mechanism efectiveness across G0–G4.

G0: Replay performance. For replay event $j ,$ let $A _ { \mathrm { i n i t } , j }$ denote the agent before its first execution of the replayed task, and $A _ { \mathrm { r e p l a y } , j }$ the agent executing the current replay. At G0, compare $A _ { \mathrm { r e p l a y } , j }$ with $A _ { \mathrm { i n i t } , j }$ on the replayed task $\tau _ { j }$

G1–G4: Agent generalization. Relative to a source game g at dificulty $d _ { 0 } , D _ { 1 }$ contains instances $G _ { g } ( d _ { 0 } , \xi ^ { \prime } )$ and $\mathcal { D } _ { 2 }$ contains $G _ { g } ( d _ { 1 } , \xi ^ { \prime } )$ with $d _ { 1 } > d _ { 0 }$ . Collections $\mathcal { D } _ { 1 }$ and $\mathcal { D } _ { 2 }$ are drawn from $\mathcal { D } _ { \mathrm { o u t } }$ , whose task descriptors are kept distinct from non-replay held-in descriptors by the construction protocol. Collections $\mathcal { D } _ { 3 }$ and $\mathcal { D } _ { 4 }$ contain other games $g ^ { \prime }$ not used for refinement, with $c ( g ^ { \prime } ) = c ( g )$ and $c ( g ^ { \prime } ) \neq c ( g )$ , respectively. $c ( g )$ denotes the primary capability category. For mixed-source sequences, source games and capability categories are specified for each transfer comparison. At G1–G4, compare $A _ { T }$ with $A _ { 0 }$ on the same target instances.

Metrics. Across G0–G4, we use the HCI (Headroom-Closed Index) [34] to normalize gains relative to the initial gap to a reference score. We first average the baseline and comparison scores over target instances, then compute HCI from the resulting means. Let $s _ { b }$ and $s _ { a }$ denote these mean baseline and comparison scores and $s _ { r }$ the reference score :

$$
\mathrm { H C I } = \frac { s _ { a } - s _ { b } } { s _ { r } - s _ { b } } .\tag{1}
$$

## 3.4 Semantic Process Analysis

Semantic relations. These indicate whether a proposal introduces new content, contradicts or revises existing proposals, or duplicates them, relative to the parent proposal snapshot and earlier proposals in the same stage.

Applicability scope. This distinguishes task-specific proposals, proposals applicable across tasks sharing a capability, and fully general proposals.

Functional targets. These identify which capability a proposal targets: exploration, reasoning, planning, or execution. Execution concerns operational mechanisms such as tool invocation, output formatting, retries, and error recovery.

Modification surfaces. These identify whether the change afects contextual artifacts, the execution harness, or the refinement mechanism itself.

<table><tr><td></td><td></td><td colspan="2">G0</td><td colspan="2"></td><td colspan="2"></td><td colspan="2">G2</td><td colspan="2">G3</td></tr><tr><td>Task</td><td>Method</td><td></td><td>Init. Final HCI (%)</td><td></td><td></td><td>Init. Final HCI (%)</td><td></td><td></td><td>Init. Final HCI (%)</td><td></td><td>Init. Final HCI (%)</td></tr><tr><td>Farm</td><td>ACE</td><td>1.078 1.067</td><td></td><td>-10.2</td><td>0.997 1.145</td><td>+81.3</td><td></td><td>0.784 0.859</td><td>+14.6</td><td>0.533 0.596</td><td>+7.0</td></tr><tr><td></td><td>AHE</td><td>0.7681.009</td><td></td><td>+58.7</td><td>1.105 1.080</td><td></td><td>-33.8</td><td>1.0731.069</td><td>-1.8</td><td>0.6740.792</td><td>+15.5</td></tr><tr><td></td><td>Hyper</td><td>1.0461.084</td><td></td><td>+28.0</td><td>1.0481.179</td><td>+100.0</td><td></td><td>0.8570.942</td><td>+19.3</td><td>0.597 0.596</td><td>-0.1</td></tr><tr><td>Combat</td><td>ACE</td><td>0.855 0.911</td><td></td><td>+9.7</td><td>0.5330.592</td><td></td><td>+6.6</td><td>0.227 0.385</td><td>+14.5</td><td>0.997 1.123</td><td>+69.2</td></tr><tr><td></td><td>AHE</td><td>1.065 1.410</td><td></td><td>+93.8</td><td>0.6741.427</td><td></td><td>+99.2</td><td>0.301 1.196</td><td>+88.4</td><td>1.105 1.087</td><td>-24.3</td></tr><tr><td></td><td>Hyper</td><td>0.8811.226</td><td></td><td>+62.5</td><td>0.5971.433</td><td>+100.0</td><td></td><td>0.2331.314</td><td>+100.0</td><td>1.0481.191</td><td>+109.2</td></tr><tr><td>Sweepmine ACE</td><td></td><td>0.239 0.193</td><td></td><td>-6.0</td><td>0.134 0.316</td><td></td><td>+21.0</td><td>0.043 0.234</td><td>+20.0</td><td>0.094 0.067</td><td>-3.0</td></tr><tr><td></td><td>AHE</td><td>0.4320.420</td><td></td><td>-2.0</td><td>0.1500.436</td><td>+33.7</td><td></td><td>0.1070.350</td><td>+27.2</td><td>0.0970.077</td><td>-2.2</td></tr><tr><td></td><td>Hyper</td><td>0.1930.216</td><td></td><td>+2.8</td><td>0.0860.205</td><td>+12.9</td><td></td><td>0.014 0.117</td><td>+10.5</td><td>0.1000.085</td><td>-1.7</td></tr><tr><td>Hero</td><td>ACE</td><td>0.169 0.124</td><td></td><td>-5.4</td><td>0.094 0.154</td><td></td><td>+6.7</td><td>0.017 0.124</td><td>+10.9</td><td>0.134 0.234</td><td>+11.5</td></tr><tr><td></td><td>AHE</td><td>0.1870.244</td><td></td><td>+7.0</td><td>0.097 0.526</td><td>+47.5</td><td></td><td>0.0540.257</td><td>+21.4</td><td>0.1500.318</td><td>+19.8</td></tr><tr><td></td><td>Hyper</td><td>0.087 0.070</td><td></td><td>-1.9</td><td>0.1000.085</td><td></td><td>-1.7</td><td>0.0260.082</td><td>+5.7</td><td>0.0860.264</td><td>+19.4</td></tr><tr><td>MC</td><td>ACE</td><td>0.162 0.077</td><td></td><td>-10.1</td><td>0.000 0.020</td><td></td><td>+2.0</td><td>0.000 0.350</td><td>+35.0</td><td>0.4000.400</td><td>+0.0</td></tr><tr><td></td><td>AHE</td><td>0.9030.728</td><td></td><td>-179.4</td><td>0.5050.720</td><td>+43.4</td><td></td><td>1.0000.600</td><td>N/A</td><td>0.4000.400</td><td>+0.0</td></tr><tr><td></td><td>Hyper</td><td>0.2780.483</td><td></td><td>+28.4</td><td>0.0500.220</td><td>+17.9</td><td></td><td>0.0000.400</td><td>+40.0</td><td>0.4000.200</td><td>-33.3</td></tr><tr><td>Astray</td><td>ACE</td><td>0.318 0.395</td><td></td><td>+11.4</td><td>0.4000.420</td><td></td><td>+3.3</td><td>0.222 0.211</td><td>-1.4</td><td>0.000 0.050</td><td>+5.0</td></tr><tr><td></td><td>AHE</td><td>0.617 0.769</td><td></td><td>+39.6</td><td>0.4000.440</td><td></td><td>+6.7</td><td>0.2220.378</td><td>+20.0</td><td>0.5050.550</td><td>+9.1</td></tr><tr><td></td><td>Hyper</td><td>0.274 0.265</td><td></td><td>-1.2</td><td>0.4000.520</td><td>+20.0</td><td></td><td>0.2440.222</td><td>-2.9</td><td>0.0500.000</td><td>-5.3</td></tr></table>

Table 2 Performance across G0–G3. Bold marks the highest available HCI for each game at each level, including ties. For G3, each row is the source game and Final is its agent’s score on the paired target (Farm–Combat, Sweepmine–Hero, MC–Astray).

## 4 Experiments

We examine the gains from iterative refinement in ACE, AHE, and HyperAgents and whether these gains extend to unseen instances, harder tasks, and other games. We further analyze their refinement trajectories, task scheduling, and execution-harness refinement.

## 4.1 Experimental Setup

Tested Baselines. We evaluate ACE, AHE, and HyperAgents as representative methods of experience-level, harness-level, and meta-level RSI, respectively (Section 2). ACE maintains an evolving structured playbook; AHE revises the prompt, tools, and execution harness based on task feedback; and HyperAgents uses an improver to modify the task agent’s code repository. We vary the backbone model, including GPT-5.6 Sol and Claude Opus 5.5, and the initial execution harness, including Codex and DeepSeek Harness, to assess how these choices afect RSI performance.

Configuration. Farm, Combat, Sweepmine, and Hero support dificulty levels 1–5. For MC and Astray, evaluation uses five fixed tasks per game from the oficial GameWorld [35] evaluation sets.

## 4.2 Main Results

We present the HCI scores of each RSI method at diferent levels (G0 to G4) in Tablee 2 and Figure 4.

(G0) Retention on Previously Encountered Instances. Refinement does not always preserve performance on previously encountered instances: replay scores fall below historical scores in 8 of the 18 method–task pairs. AHE on MC and ACE on Sweepmine improve on G1 but decline on G0.

(G1) Gains on Held-Out Instances. Most methods improve on unseen instances at the training dificulty: 16 of the 18 method–task pairs have positive HCI. HyperAgents has the highest HCI on Farm, Combat, and

Positive HCI Negative HCI Target-game evolved HCI

Astray, and AHE on Sweepmine, Hero, and MC. A stronger initial agent does not necessarily stay ahead. On Farm, AHE starts above HyperAgents but finishes below it, with HCI of −33.8% and 100.0%.

(G2) Generalization to Harder Instances. Gains at the training dificulty do not necessarily extend to harder instances. ACE on Astray improves on G1 but declines on G2, and rankings can reverse: on Farm, HyperAgents beats AHE at the training dificulty but trails it on the Hard held-out set. On MC’s Hard group, AHE falls from 1.000 to 0.600.

(G3) Within-Category Transfer. Within-category transfer can help or hurt, depending on direction. Among the Reasoning games, all three methods improve from Hero to Sweepmine but degrade in the reverse direction. An agent refined on another game of the same category can even outperform direct refinement on the target: HyperAgents reaches 109.2% HCI from Combat to Farm, above the 100.0% from refinement on Farm itself.

(G4) Cross-Category Transfer. Some HyperAgents improvements remain efective across capability categories, but not consistently: five of the eight directions in Figure 4 have positive HCI. The Farm-refined agent on Sweepmine and Hero, and the Combat-refined agent on Hero, exceed agents refined directly on those targets.

Overall, the evaluated methods show inconsistent refinement gains across generalization levels: improvements on unseen instances do not necessarily extend to higher dificulties or other tasks, and some settings exhibit performance degradation.

## 4.3 RSI Trajectory Analysis

The generalization ladder shows where gains persist; trajectory analysis examines what the methods change. We use GPT-5.6 Sol to split improvement artifacts into atomic opinions and annotate their semantic relations, applicability scope, functional targets, and modification surfaces. Table 3 covers the 915 opinions retained in the final snapshots of the six games.

Task-specific refinement can coexist with transferable gains. Task-specific opinions predominate in all three methods; category-general or general ones make up 14.2% of ACE’s retained opinions, 7.8% of AHE’s, and 36.5% of HyperAgents’. Yet G3 and G4 show positive transfer across games, sometimes exceeding direct refinement on the target.

![](images/6e015543ba0700e857b5fc9d13cdacda0a93ceae22916dd1b42253bbf2dc1d1e.jpg)  
Figure 4 HyperAgents cross-category transfer (G4). Each panel is a target game; lines give the HCI of the agent refined on each source game. Dashed lines mark the HCI of refinement directly on the target.

<table><tr><td></td><td></td><td colspan="3">Records</td><td colspan="4">Functional target (%)</td><td colspan="2">Applicability scope (%)</td><td colspan="2">General|</td><td colspan="2">Modification surface (%) Prompt/Mem. Harness Meta</td></tr><tr><td>Method Game ACE</td><td></td><td>28</td><td>All Rev. (%) Dup. (%) 0.00</td><td>10.71</td><td>25</td><td></td><td>NExplor. Reason. Plan. Exec. 84.0</td><td>48.0</td><td>4.0</td><td>Task Category 96.0</td><td>4.0</td><td></td><td></td><td></td></tr><tr><td rowspan="8"></td><td>Farm Combat</td><td></td><td></td><td>17.65</td><td>40</td><td>0.0 22.5</td><td>95.0</td><td>52.5</td><td>7.5</td><td></td><td></td><td>0.0 0.0</td><td>100.0 0.0 100.0 0.0</td><td>0.0 0.0</td></tr><tr><td>Hero</td><td>51</td><td>3.92</td><td></td><td></td><td></td><td>82.6</td><td></td><td></td><td>100.0 32.6</td><td>0.0 65.2</td><td>2.2</td><td>100.0</td><td></td></tr><tr><td></td><td>78</td><td>26.92</td><td>12.82</td><td>46</td><td>0.0</td><td></td><td>67.4</td><td>2.2</td><td></td><td></td><td></td><td>0.0</td><td>0.0</td></tr><tr><td>Sweepmine</td><td>44</td><td>11.36</td><td>6.82</td><td>36</td><td>8.3</td><td>72.2</td><td>41.7</td><td>8.3</td><td>100.0</td><td></td><td>0.0</td><td>100.0 0.0 100.0</td><td>0.0</td></tr><tr><td>MC</td><td>65</td><td>16.92</td><td>13.85</td><td>43</td><td>18.6</td><td>53.5</td><td>53.5</td><td>25.6</td><td>100.0</td><td></td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Astray</td><td>66</td><td>12.12</td><td>12.12</td><td>35</td><td>17.1</td><td>65.7</td><td>25.7</td><td>11.4</td><td>100.0</td><td>0.0</td><td>0.0</td><td>100.0 0.0</td><td>0.0</td></tr><tr><td>Farm</td><td>91</td><td>32.97</td><td>13.19</td><td>50</td><td>8.0</td><td>32.0</td><td>8.0</td><td>68.0</td><td>76.0</td><td>18.0</td><td>6.0</td><td>38.0 62.0</td><td>0.0</td></tr><tr><td>Combat Hero</td><td>56 397</td><td>26.79 21.16</td><td>14.29</td><td>23</td><td>0.0</td><td>43.5 52.2 82.0</td><td></td><td>30.4</td><td>91.3</td><td>0.0</td><td>8.7</td><td>39.1 60.9</td><td>0.0</td></tr><tr><td></td><td></td><td>27.27</td><td>46.10</td><td>89</td><td>16.9</td><td></td><td>27.0</td><td>25.8 26.4</td><td>91.0</td><td>4.5 0.0</td><td>4.5 0.0</td><td>98.9 71.7</td><td>1.1</td><td>0.0 0.0</td></tr><tr><td>MC</td><td>Sweepmine</td><td></td><td></td><td>18.18 53</td><td></td><td>5.7</td><td>50.9 41.5</td><td></td><td>100.0</td><td></td><td></td><td>98.8</td><td>28.3</td><td>0.0</td></tr><tr><td>Astray</td><td>210</td><td>40.48 40.00</td><td>15.71</td><td>82</td><td>17.1</td><td></td><td>24.4 82.9</td><td>4.9</td><td>95.1 97.9</td><td>2.4 2.1</td><td>2.4</td><td></td><td>1.2 0.0</td><td>0.0</td></tr><tr><td>Hyper</td><td></td><td>215</td><td></td><td>18.60</td><td>48</td><td>14.6</td><td>47.9</td><td>60.4</td><td>12.5</td><td></td><td></td><td>0.0</td><td>100.0 3.9</td><td></td></tr><tr><td rowspan="6"></td><td>Farm</td><td>434</td><td>55.53</td><td>17.05 102</td><td></td><td>2.9</td><td>77.5</td><td>56.9</td><td>12.7</td><td>42.2</td><td>46.1</td><td>11.8</td><td>96.1</td><td>0.0</td></tr><tr><td>Combat</td><td>265 383</td><td>36.60</td><td>30.94</td><td>80</td><td>10.0</td><td>62.5</td><td>43.8</td><td>41.2</td><td>66.2</td><td>31.2 14.9</td><td>2.5 4.3</td><td>46.2 53.8</td><td>0.0</td></tr><tr><td>Hero</td><td>386</td><td>18.80 38.08</td><td>67.62</td><td>47</td><td>0.0</td><td>57.4</td><td>42.6</td><td>6.4</td><td>80.9</td><td></td><td></td><td>100.0 0.0</td><td>0.0</td></tr><tr><td>Sweepmine</td><td>141</td><td>40.43</td><td>43.52 36.88</td><td>57</td><td>14.0</td><td>64.9 54.5</td><td>50.9</td><td>8.8</td><td>59.6</td><td>29.8</td><td>10.5</td><td>100.0 0.0</td><td>0.0</td></tr><tr><td>MC</td><td></td><td>51.81</td><td>18.48</td><td>22 37</td><td>22.7 21.6</td><td>45.9</td><td>81.8 64.9</td><td>45.5 2.7</td><td>81.8 89.2</td><td>9.1 10.8</td><td>9.1 0.0</td><td>95.5 4.5 97.3</td><td>0.0</td></tr><tr><td>Astray</td><td>276</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>2.7</td><td>0.0</td></tr></table>

Table 3 Improvement statistics across six games. Recorded and retained opinions are compared across ACE, AHE, and HyperAgents by functional target, applicability scope, and modification surface. All/N: recorded/retained counts; Rev./Dup.: revision/duplicate rates (%).

Refinement emphasizes solving observed problems over exploring alternatives. Reasoning and planning appear in 61.2% and 49.6% of retained opinions, versus 11.0% for exploration. Current RSI methods may rely on observed trajectories to find and fix problems rather than explore alternatives.

Main-experiment changes concentrate on the task agent. 87.9% of retained opinions target prompts and memory and 12.1% the execution harness; none modify the improver or the improvement process.

More active revision may contribute to better refinement outcomes. ACE, AHE, and HyperAgents revise 14.2%, 30.6%, and 40.2% of their recorded opinions, respectively. AHE and HyperAgents revise existing opinions more frequently and generally achieve better final performance than ACE, suggesting that efective refinement may depend not only on adding new opinions but also on continually revising existing strategies.

Frequent revision does not eliminate duplicate opinions. HyperAgents has the highest revision rate, yet on Hero its duplicate rate reaches 67.6%, against 46.1% for AHE and 12.8% for ACE.

## 4.4 Ablation and Extended Analysis

## 4.4.1 Comparison with General-Purpose Agents

Figure 5 compares the refined agents with Codex, Claude Code, and DeepSeek Harness on Medium held-out tasks. Refined agents score highest on five of the six games: HyperAgents on Farm, Combat, and Astray, and AHE on Sweepmine and Hero. On MC, Claude Code outperforms all refined agents.

## 4.4.2 Efect of Task Scheduling

We evaluate four training task schedules using HyperAgents.

Random medium-dificulty training outperforms fixed-instance training on both Combat held-out sets and on Hard Hero tasks (Table 4): fixed-instance training scores 0.772 and 0.507 on Combat and 0.100 and 0.017 on Hero, against 1.433 and 1.314, and 0.085 and 0.082, for random medium-dificulty training. On Medium Hero tasks, however, fixed-instance training scores higher (0.100 versus 0.085).

Harder training instances do not necessarily help. Mixed and increasing dificulty score 1.195 and 1.080 on Hard Combat tasks, below the 1.314 of random medium-dificulty training, and 0.037 and 0.024 on Hard Hero tasks, below 0.082. Both also score lower on Medium Combat tasks, but higher on Medium Hero tasks (0.145 and 0.152 versus 0.085), indicating that their benefits depend on the game and evaluation dificulty.

![](images/d7f4f974874d66de326d2a0249e778132f9834015dd9a04938d99ba833b63462.jpg)

Figure 5 Comparison with general-purpose agents. Medium held-out scores of general-purpose agents and refined agents, ranked within each game.
<table><tr><td rowspan="2">Schedule</td><td colspan="2">Combat</td><td colspan="2">Hero</td></tr><tr><td>Medium</td><td>Hard</td><td>Medium</td><td>Hard</td></tr><tr><td>Random medium</td><td>1.433</td><td>1.314</td><td>0.085</td><td>0.082</td></tr><tr><td>Fixed medium instance</td><td>0.772</td><td>0.507</td><td>0.100</td><td>0.017</td></tr><tr><td>Mixed difficulty</td><td>1.388</td><td>1.195</td><td>0.145</td><td>0.037</td></tr><tr><td>Increasing difficulty</td><td>1.332</td><td>1.080</td><td>0.152</td><td>0.024</td></tr></table>

Table 4 Efect of training task scheduling. Medium and Hard held-out scores on Combat and Hero under random medium-dificulty, fixed-instance, mixed-dificulty, and increasing-dificulty training. Bold marks the highest score in each column.

## 4.4.3 Exploratory Experiment: Directly Refining an Execution Harness

<table><tr><td>Task</td><td>Configuration</td><td>Score ↑</td></tr><tr><td rowspan="3">Combat</td><td>DeepSeek Harness HyperDSH (GPT-5.6 Sol)</td><td>0.751 1.274</td></tr><tr><td>HyperDSH (Claude Opus 5.5)</td><td>0.941</td></tr><tr><td>HyperDSH (DeepSeek) HyperDSH (DeepSeek, fixed improver)</td><td>1.326</td></tr></table>

Table 5 Direct execution-harness refinement. Medium held-out Combat scores of the original DeepSeek Harness, HyperDSH variants, and the main-experiment HyperAgents agent. HyperDSH is a variant that applies HyperAgent’s improvement mechanisms to Deepseek Harness.

We evaluate HyperDSH’s direct refinement of DeepSeek Harness on Combat, comparing it with the original DeepSeek Harness and the HyperAgents agent refined in the main experiment (Table 5). All three HyperDSH variants outperform DeepSeek Harness on Medium held-out tasks; the DeepSeek variant performs best among them at 1.326, below HyperAgents at 1.433. HyperDSH (DeepSeek, fixed improver), which keeps the improver unchanged while continuing to refine DeepSeek Harness, scores 1.207 on Medium, compared with 1.326 for HyperDSH (DeepSeek).

Despite permission to modify the entire repository, the main-experiment HyperAgents retains no Meta opinions, suggesting a limited tendency to modify its own improver spontaneously. In contrast, Table 6 shows that Meta accounts for 20.5–45.2% of retained opinions in HyperDSH variants that target DeepSeek Harness refinement and explicitly allow improver self-evolution, reflecting more modifications to the improver itself. Their retained modifications also shift toward the improver: none of the three variants retains Harness-level opinions, compared with 53.8% for HyperAgents. The three HyperDSH variants also produced fewer opinions: 57–127 recorded occurrences, compared with 265 for HyperAgents, indicating less active opinion generation when refining an existing execution harness.

<table><tr><td>Task</td><td>Configuration</td><td>All</td><td>N</td><td>Prompt/Mem. (%)</td><td>Harness (%)</td><td>Meta (%)</td></tr><tr><td rowspan="6">Combat</td><td>DeepSeek Harness</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HyperDSH (GPT-5.6 Sol)</td><td>107</td><td>44</td><td>79.5</td><td>0.0</td><td>20.5</td></tr><tr><td>HyperDSH (Claude Opus 5.5)</td><td>57</td><td>31</td><td>54.8</td><td>0.0</td><td>45.2</td></tr><tr><td>HyperDSH (DeepSeek)</td><td>127</td><td>37</td><td>59.5</td><td>0.0</td><td>40.5</td></tr><tr><td>HyperDSH (DeepSeek, fixed improver)</td><td>85</td><td>27</td><td>96.3</td><td>3.7</td><td>0.0</td></tr><tr><td>HyperAgents</td><td>265</td><td>80</td><td>46.2</td><td>53.8</td><td>0.0</td></tr></table>

Table 6 Trajectory comparison for execution-harness refinement. We compare HyperDSH variants with the main-experiment HyperAgents agent on Combat, reporting recorded and retained opinion counts and the modificationsurface distribution of retained opinions. Configurations correspond to Table 5.

## 5 Conclusion

We presented SelfImprove-Bench, a scalable benchmark for RSI evaluation. Diferent from conventional benchmarks, SelfImprove-Bench supports template- and dificulty-controlled task generation for configurable evaluation. We propose a generalization ladder to reflect the transfer to new instances, new dificulties, new tasks within a capability, and across capabilities. We evaluate three representative classes of RSI methods and find that their efectiveness varies across the generalization ladder, with their relative rankings reversing when transferring to tasks of higher dificulty or diferent types.

## References

[1] J. Zhang, B. Zhao, W. Yang, J. Foerster, J. Clune, M. Jiang, S. Devlin, and T. Shavrina, “Hyperagents,” arXiv preprint arXiv:2603.19461, 2026. [Online]. Available: https://arxiv.org/abs/2603.19461

[2] Q. Zhang, C. Hu, S. Upasani, B. Ma, F. Hong, V. Kamanuru, J. Rainton, C. Wu, M. Ji, H. Li, U. Thakker, J. Zou, and K. Olukotun, “Agentic context engineering: Evolving contexts for self-improving language models,” arXiv preprint arXiv:2510.04618, 2025. [Online]. Available: https://arxiv.org/abs/2510.04618

[3] J. Lin, S. Liu, C. Pan, L. Lin, S. Dou, Z. Xi, X. Huang, H. Yan, Z. Han, T. Gui, and Y.-G. Jiang, “Agentic harness engineering: Observability-driven automatic evolution of coding-agent harnesses,” arXiv preprint arXiv:2604.25850, 2026. [Online]. Available: https://arxiv.org/abs/2604.25850

[4] C. Zheng, C. Xue, B. Liang, J. Yang, and C. Zhang, “Seagym: An evaluation environment for self-evolving llm agents,” arXiv preprint arXiv:2606.17546, 2026.

[5] L. Huang, C. Yang, H. Zhou, H. Song, Z. Chen, R. Le, Y. Song, W. X. Zhao, and T. Zhang, “Evo-Bench: Can language models improve agent harness?” arXiv preprint arXiv:2608.09096, 2026. [Online]. Available: https://arxiv.org/abs/2608.09096v1

[6] B. Deng, K. Zhou, L. Guo, C. Tao, X. Chen, C. Xie, R. Liang, F. Chen, and C. Zhang, “FinEvo-Bench: A longitudinal benchmark for self-evolving agents in professional financial workflows,” arXiv preprint arXiv:2608.06144, 2026. [Online]. Available: https://arxiv.org/abs/2608.06144

[7] Z. Stojanovski, O. Stanley, J. Sharratt, R. Jones, A. Adefioye, J. Kaddour, and A. Köpf, “Reasoning Gym: Reasoning environments for reinforcement learning with verifiable rewards,” in Advances in Neural Information Processing Systems, vol. 38, 2025. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 528d56195a2c77c808494c86fa7c77ad-Abstract-Datasets\_and\_Benchmarks\_Track.html

[8] S. Xue, Z. Ding, Y. Shen, Y. Wang, Z. Yin, Y. Wu, Y. Chen, M. Wang, and L. Yang, “Pastbench: Benchmarking the foundations of recursive self-improvement in personal agents,” arXiv preprint arXiv:2608.04003, 2026.

[9] X. Gao, C. Hu, H. Chen, P. Yao, Z. Wang, Y. Bai, Z. Wu, Y. Han, X. Cong, J. Gui, Y. Deng, and T. Li, “EvoAgentBench: Benchmarking agent self-evolution via ability transfer,” arXiv preprint arXiv:2607.05202, 2026. [Online]. Available: https://arxiv.org/abs/2607.05202

[10] S. Zhong, Y. Lu, J. Ning, Y. Wan, L. Feng, Y. Ao, L. F. R. Ribeiro, M. Dreyer, S. Ammirati, and C. Xiong, “SkillLearnBench: Benchmarking continual learning methods for agent skill generation on real-world tasks,” arXiv preprint arXiv:2604.20087, 2026. [Online]. Available: https://arxiv.org/abs/2604.20087v1

[11] J. Zheng, X. Cai, Q. Li, D. Zhang, Z. Li, Y. Zhang, L. Song, and Q. Ma, “LifelongAgentBench: Evaluating LLM agents as lifelong learners,” arXiv preprint arXiv:2505.11942, 2025. [Online]. Available: https://arxiv.org/abs/2505.11942

[12] H. Lin, C. Liu, X. Bai, X. Jin, Y. Li, N. Zheng, and X. Cao, “Beyond endpoint performance: Process-level evaluation of self-evolving agents,” arXiv preprint arXiv:2609.24663, 2026. [Online]. Available: https://arxiv.org/abs/2609.24663

[13] H. Zheng, Y. Di, R. Sadhukhan, S. Jin, and B. Chen, “ServeLearnBench: How well can agents self-improve from serving experience?” arXiv preprint arXiv:2610.07792, 2026. [Online]. Available: https://arxiv.org/abs/2610.07792

[14] E. Zelikman, E. Lorch, L. Mackey, and A. T. Kalai, “Self-taught optimizer (STOP): Recursively self-improving code generation,” in First Conference on Language Modeling, 2024. [Online]. Available: https://arxiv.org/abs/2310.02304v3

[15] S. Hu, C. Lu, and J. Clune, “Automated design of agentic systems,” in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https://proceedings.iclr.cc/paper\_ files/paper/2025/hash/36b7acf6f6010652b3f2a433774a66fe-Abstract-Conference.html

[16] P. Xia, R. Han, Z. Wang, Y. Chen, Y. Zhang, Y. Lee, C. Huang, H. Yu, Z. CuiZhu, Y. Ming, H. Yao, B. Gokturk, T. Pfister, and C.-Y. Lee, “RRSI: Regularized recursive self-improvement of agent harnesses,” arXiv preprint arXiv:2609.24972, 2026. [Online]. Available: https://arxiv.org/abs/2609.24972v1

[17] Y. Chi, W. Li, D. Hong, X. Wang, M. Gao, K. Yang, B. He, Y. Zheng, C. Xiao, and Q. Na, “Ai4aibench: Benchmarking llm agents in algorithmic design for recursive self-improvement,” arXiv preprint arXiv:2608.20318, 2026.

[18] F. Meng, L. Du, Q. Chen, Z. Zhao, H. Lu, M. Hu, and M. Q. Shieh, “RSIBench-Data: Benchmarking data-centric research for recursive self-improvement,” arXiv preprint arXiv:2607.25886, 2026. [Online]. Available: https://arxiv.org/abs/2607.25886

[19] N. Shinn, F. Cassano, E. Berman, A. Gopinath, K. Narasimhan, and S. Yao, “Reflexion: Language agents with verbal reinforcement learning,” arXiv preprint arXiv:2303.11366, 2023. [Online]. Available: https://arxiv.org/abs/2303.11366

[20] W. Xu, Z. Liang, K. Mei, H. Gao, J. Tan, and Y. Zhang, “A-MEM: Agentic memory for LLM agents,” arXiv preprint arXiv:2502.12110, 2025. [Online]. Available: https://arxiv.org/abs/2502.12110

[21] M. Robeyns, M. Szummer, and L. Aitchison, “A self-improving coding agent,” arXiv preprint arXiv:2504.15228, 2025.

[22] H. Zhang, S. Zhang, K. Li, C. Zhang, Y. Chen, Y. Zhang, L. Bai, and S. Hu, “Self-harness: Harnesses that improve themselves,” arXiv preprint arXiv:2606.09498, 2026. [Online]. Available: https://arxiv.org/abs/2606.09498

[23] J. Zhang, S. Hu, C. Lu, R. Lange, and J. Clune, “Darwin gödel machine: Open-ended evolution of self-improving agents,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 104 223–104 294.

[24] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. L. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe, “Training language models to follow instructions with human feedback,” arXiv preprint arXiv:2203.02155, 2022. [Online]. Available: https://arxiv.org/abs/2203.02155

[25] E. Zelikman, Y. Wu, J. Mu, and N. D. Goodman, “STaR: Bootstrapping reasoning with reasoning,” arXiv preprint arXiv:2203.14465, 2022. [Online]. Available: https://arxiv.org/abs/2203.14465

[26] E. Nivel, K. R. Thórisson, B. R. Steunebrink, H. Dindo, G. Pezzulo, M. Rodriguez, C. Hernandez, D. Ognibene, J. Schmidhuber, R. Sanz, H. P. Helgason, A. Chella, and G. K. Jonsson, “Bounded recursive self-improvement,” arXiv preprint arXiv:1312.6764, 2013. [Online]. Available: https://arxiv.org/abs/1312.6764

[27] Y. Wang, Z. Zhang, M. Chi, K. Yu, Y. Li, M. Peng, B. Tong, C. Zhang, Y. Zhou, and J. Li, “EvoMemBench: Benchmarking agent memory from a self-evolving perspective,” arXiv preprint arXiv:2605.18421, 2026. [Online]. Available: https://arxiv.org/abs/2605.18421v2

[28] T. Guan, Y. Wang, H. Yang, S. Cao, S. Liu, Y. Hu, J. Li, and M. Zhang, “ContinualSkillBench: Can LLM agents truly evolve their capabilities?” arXiv preprint arXiv:2608.03874, 2026. [Online]. Available: https://arxiv.org/abs/2608.03874

[29] P. Asawa, C. M. Glaze, G. Orlanski, R. Ramakrishnan, B. Xu, A. Biswal, V. S. Chen, F. Sala, M. Zaharia, and J. E. Gonzalez, “Continual Learning Bench: Evaluating frontier AI systems in real-world stateful environments,” arXiv preprint arXiv:2606.05661, 2026. [Online]. Available: https://arxiv.org/abs/2606.05661

[30] C. E. Jimenez, J. Yang, A. Wettig, S. Yao, K. Pei, O. Press, and K. Narasimhan, “SWE-bench: Can language models resolve real-world GitHub issues?” arXiv preprint arXiv:2310.06770, 2023. [Online]. Available: https://arxiv.org/abs/2310.06770

[31] M. A. Merrill, A. G. Shaw, N. Carlini, B. Li, H. Raj, I. Bercovich, L. Shi, J. Y. Shin, T. Walshe, E. K. Buchanan et al., “Terminal-bench: Benchmarking agents on hard, realistic tasks in command line interfaces,” arXiv preprint arXiv:2601.11868, 2026. [Online]. Available: https://arxiv.org/abs/2601.11868

[32] S. Zhou, F. F. Xu, H. Zhu, X. Zhou, R. Lo, A. Sridhar, X. Cheng, T. Ou, Y. Bisk, D. Fried, U. Alon, and G. Neubig, “WebArena: A realistic web environment for building autonomous agents,” arXiv preprint arXiv:2307.13854, 2023. [Online]. Available: https://arxiv.org/abs/2307.13854

[33] L. Phan, A. Gatti, Z. Han, N. Li, J. Hu, H. Zhang, C. B. C. Zhang, M. Shaaban, J. Ling, S. Shi et al., “Humanity’s last exam,” arXiv preprint arXiv:2501.14249, 2025. [Online]. Available: https://arxiv.org/abs/2501.14249

[34] Y. Duan, Y. Liu, Z. Tang, H. Chen, J. Zhou, Y. Liu, B. Xu, Y. Wu, S. Chen, Y. Zhou, H. Wang, X. Yu, S. Han, X. Zhu, L. Zhou, B. Lu, W. Zhou, J. Liu, N. Fang, J. Tian, R. Chen, Y. Li, K. Zuo, K. Zhang, Q. Yang, Z. Wang, J. Qiu, C. He, G. Li, B. Zhou, Z. Liu, Z. Wen, J. Kang, X. Zhou, and F. Wu, “The last ai built by humans: Toward genuine recursive self-improvement,” 2026. [Online]. Available: https://arxiv.org/abs/2609.11873

[35] M. Ouyang, S. Hu, K. Q. Lin, H. T. Ng, and M. Z. Shou, “GameWorld: Towards standardized and verifiable evaluation of multimodal game agents,” arXiv preprint arXiv:2604.07429, 2026. [Online]. Available: https://arxiv.org/abs/2604.07429

[36] A. A. Zheng, K. Misra, D. I. Beaver, and J. J. Li, “Hero’s journey: Testing complex rule induction with text games,” arXiv preprint arXiv:2606.02556, 2026.

## A Evaluation Protocol Details

## A.1 Task Configurations

The task-related fields of θ determine which instances are generated and how they form $S _ { \mathrm { i n } }$ and $S _ { \mathrm { o u t } } \mathrm { i }$ ; model and runtime settings are configured separately. Game definitions, dificulty parameters, and scoring rules are provided in Appendix B.

Held-in tasks. The ordered schedule list contains blocks with game, difficulty, and count fields. Each block generates the requested number of non-replay instances of that game and dificulty. Blocks are concatenated in list order before replay insertion; count excludes inserted replays. The block-level replay flag controls eligibility for replay insertion, whose exact semantics are defined in Appendix A.2. The standard ordered setting uses random=false and repeat=false; these are also the defaults.

Held-out tasks. The ordered holdout list specifies evaluation blocks using the same game, difficulty, and count fields. For each requested game–dificulty pair, holdout\_tasks\_per\_difficulty sets the candidate pool size, while the block’s count selects that many instances from the beginning of its pool. Selected blocks form $S _ { \mathrm { o u t } }$ in list order, without replay insertion.

Seeds and reconstruction. The seed and holdout\_seed fields control held-in and held-out generation, respectively. For newly generated instances, task seeds are derived deterministically from the root seed, phase, game, dificulty, instance index, and generation attempt using SHA-256. The current generator rejects duplicate task-payload hashes, excluding task\_id, across the generated held-out pool and non-replay held-in instances; replay insertion deliberately reuses an existing task. This is descriptor-level deduplication: seed fields remain in the hashed payload, so diferent descriptors are not guaranteed to produce diferent underlying game states. Each run saves its resolved configuration and task manifest as config.resolved.json and task\_manifest.json, including task instances, seeds, sequence order, and replay links. These artifacts and the corresponding generator version specify the realized sequences.

## A.2 Replay Protocol

Execution. Each replay executes the original task instance using the current agent, and records the resulting score for G0 evaluation. Replay executions do not trigger refinement.

Insertion rule. Replays are inserted while constructing the held-in sequence, before execution. Let $u _ { 1 } , \ldots , u _ { n }$ be the non-replay tasks placed so far, with $u _ { n }$ belonging to the current schedule block. After appending $u _ { n } ,$ , insert a replay of $u _ { n - \ell }$ if the current block enables replay, the interval e and lag ℓ are positive, n mod $e = 0$ , and $n > \ell .$ . Both interval and lag count only non-replay tasks, globally across schedule blocks.

Boundary conditions. No replay is inserted when history is insuficient $( n \leq \ell )$ , when the current block disables replay, or when replay is globally disabled $( e = \ell = 0 )$ . No additional replay is inserted to complete a partial interval. The rule also applies after the final non-replay task: a replay is appended if the same conditions hold. The ordering above assumes random=false; enabling random shufles the assembled sequence, including its replay entries, after insertion.

## A.3 Feedback Specification

Abstract interface. The benchmark passes task execution results to refinement through a FeedbackResult object,

$$
\mathcal { F } = ( \mathrm { i d } , \mathrm { t a s k } , \mathrm { t r a j e c t o r y } , \mathrm { e v a l u a t i o n } ) .
$$

The identifier links feedback to its execution, and task contains the game-specific task descriptor. The ordered trajectory contains step records with step\_index, prompt, image\_records, action, environment\_- response, success, score, feedback, and an optional agent\_call. The latter retains the model response and, when available, reasoning, token usage, provider, model, and call metadata. Image records reference available images; text-only interactions use an empty image list. Step scores, success flags, and feedback may be absent or null when the game does not provide them.

The evaluation object contains execution status, task success, the task score, optional explanatory feedback, and game-specific metrics. Execution status distinguishes completion, error, timeout, and cancellation; a completed execution need not solve the task successfully. Metrics depend on the game and can include consumed actions or restart counts. Game-specific scoring is described in Appendix B.

## A.4 Annotation Setup and Validation

The analysis records use $\mathtt { g p t - 5 . 6 - s o l }$ for semantic annotation. We checked 100 pre-extracted proposals, stratified by game and method across six games and three methods, against their source modifications. Table 7 reports agreement with human labels after human-approved, assistant-assisted review with access to machine labels; these are post-review results rather than independent blind agreement estimates.

Scope and Surface each take one label. Function permits multiple labels because a single proposal can afect several functions, such as reasoning and planning. Complete-set agreement is therefore stricter than any-label overlap, which requires only $H _ { i } \cap M _ { i } \neq \emptyset$ for human and machine label sets $H _ { i }$ and $M _ { i }$ . Mean Jaccard similarity averages $| H _ { i } \cap M _ { i } | / | H _ { i } \cup M _ { i } |$ across proposals, accounting for both missing and additional labels. The results indicate strong correspondence after review, particularly for Surface, while diferences remain in Scope and complete Function combinations. The 96% Function overlap indicates a shared functional label on most proposals, not complete agreement or classification accuracy.

<table><tr><td rowspan=1 colspan=1>Measure</td><td rowspan=1 colspan=1>After review</td></tr><tr><td rowspan=1 colspan=1>Scope exact agreement</td><td rowspan=1 colspan=1>88%</td></tr><tr><td rowspan=1 colspan=1>Surface exact agreement</td><td rowspan=1 colspan=1>99%</td></tr><tr><td rowspan=1 colspan=1>Function any-label overlap</td><td rowspan=1 colspan=1>96%</td></tr><tr><td rowspan=1 colspan=1>Function mean Jaccard</td><td rowspan=1 colspan=1>0.828</td></tr></table>

Table 7 Agreement between fixed machine labels and reviewed human labels on 100 sampled proposals.

## B Game Design

In this appendix, we describe the game design in our dataset and detail how game instances are generated by combining task templates with dificulty coeficients.

## B.1 Planning Games

## B.1.1 Task Definition

Farm. In this game, an agent operates a farm under cash, land, and time constraints. Each task specifies the initial cash, number of plots, planning days horizon, and a task-specific crop catalog containing seed prices, maturation times, harvest yields, and selling prices. The agent receives the entire instance in one non-interactive call and returns a JSON plan indexed by day. Within each day, actions are executed in the fixed order buy, plant, harvest, sell. The objective is to maximize terminal cash profit, i.e., final cash minus initial cash.

Combat. Combat is a deterministic deck-building game comprising alternating combats and shops followed by a boss encounter. Player health, gold, and deck composition persist across steps. At a combat step, the agent observes its current state, owned cards, the enemy’s repeating public intent cycle, and the exact deterministic hands. It returns the cards to play on selected turns subject to hand multiplicity and energy constraints. At a shop step, the agent may buy each ofered card and purchase one heal, all within its current gold budget. The goal is to defeat the complete encounter chain, including the boss, while maximizing an objective that rewards defeated enemies, remaining health, and remaining gold.

## B.1.2 Task Generation

Farm. Let d denote dificulty. The generator fixes initial cash to 200, samples $2 d + 2$ crops uniformly without replacement from the global catalog, and sets both the number of plots and the days horizon to $4 d + 4$ Thus, increasing d simultaneously enlarges the action vocabulary, land capacity, and temporal planning depth. Crops are not tiered, and instance variation arises from the sampled combinations of crop prices, yields, and maturation times.

Combat. For dificulty $d \in \{ 1 , 2 , 3 , 4 , 5 \}$ , the available catalog contains every card of tier at most d. The initial deck contains a fixed card set, plus one uniformly sampled card from each tier $2 , \ldots , d .$ Player parameters remain fixed at 80 maximum and initial health, 120 initial gold, 3 energy, and 5 drawn cards per turn. Each combat permits at most 15 turns. For every tier $i \leq d ,$ the generator samples one non-boss enemy of tier i as the enemy, and then appends a shop. Each shop ofers three distinct cards. One ofer is forced to have tier min(d, i + 1), while the other two are sampled from cards at or below that tier. A tier-d boss is at last, producing d + 1 combats and $2 d + 1$ total steps. Enemies and cards become stronger with tier, while the longer step chain increases resource coupling across decisions.

## B.1.3 References Calculation

Planning a game involves complex permutations and combinations, and it is dificult to directly compute an optimal solution for it. Therefore, specific search algorithms are used to find a near-optimal solution within a

limited time as a pseudo ground-truth.

Farm. The pseudo ground-truth for each farm game is derived from a rolling search. On each day, matured batches are first harvested and sold. Then for each candidate crop, the heuristic tries to greedily plant it into as many free plots as cash permits, and rolls the resulting state to the days horizon using the same daily greedy rule. The candidate with the largest projected terminal cash is committed, and the pseudo ground-truth profit is the terminal cash minus the initial cash. In the above search branches, the daily decision is to plant as much of the same crop as possible, whereas in fact there may be better strategies such as planting a combination of multiple crops. Therefore, the ratio (score) of these strategies compared with the pseudo ground-truth can exceed 1.

Combat. Combat computes its reference in two search levels. For each combat stage, the solver tries every legal card combination by turns and simulates its outcome. Among winning plans, the solver prefers higher remaining health and then fewer turns. At each shop, it tries every afordable combination of the three ofered cards, both with and without healing. The solver then carries these alternatives through the following stages. To keep the search tractable, partial runs are ranked using remaining health, gold, and the damage and block provided by the deck, and only the top-k plans are continued. After the boss, the completed run with the highest oficial objective becomes the reference candidate.

## B.2 Reasoning Games

## B.2.1 Task Definition

Hero. We adapt Hero’s Journey [36], a text-based adventure benchmark for hidden-rule induction. Given demonstrations and a task description, the agent infers the missing rules and returns a complete action sequence in one response. The objective is to reach the task goal within the action budget.

Each benchmark task contains five separately initialized episodes with diferent seeds. A successful episode is scored by the ratio of reference solution length to executed action count; an unsuccessful episode receives zero. The task score is the mean episode score.

Sweepmine. The agent identifies mines on a partially observed board. Each revealed safe cell indicates the number of mines among its eight neighboring cells. At each step, the agent opens, marks, or unmarks a cell and receives the updated board. The objective is to mark every mine without marking any safe cell within the action budget. Opening a mine generates a new board without resetting the remaining budget.

An unsuccessful task receives zero. For a successful task, the score is $( B - a ) / ( B - m )$ , where B is the action budget, a the number of actions consumed, and m the number of mines.

<table><tr><td rowspan=1 colspan=1>Game</td><td rowspan=1 colspan=1>d1</td><td rowspan=1 colspan=1>d2</td><td rowspan=1 colspan=1>d3</td><td rowspan=1 colspan=1>d4</td><td rowspan=1 colspan=1>d5</td></tr><tr><td rowspan=1 colspan=1>Hero</td><td rowspan=1 colspan=1> $\mathrm { O v r / 1 / 3 }$ </td><td rowspan=1 colspan=1> $\mathrm { O r d } / 2 / 2$ </td><td rowspan=1 colspan=1> $\mathrm { O r d } / 3 / 2$ </td><td rowspan=1 colspan=1> $\mathrm { O r d } / 4 / 2$ </td><td rowspan=1 colspan=1> $\mathrm { O r d } / 5 / 2$ </td></tr><tr><td rowspan=2 colspan=1>Sweepmine</td><td rowspan=2 colspan=1> $\overline { { 5 \times 5 / } }$  $3 / 1 6$ </td><td rowspan=2 colspan=1> $6 \times 6 /$  $5 / 2 2$ </td><td rowspan=1 colspan=1> $7 \times 7 /$ </td><td rowspan=2 colspan=1> $\overline { { 8 \times 8 / } }$  $1 1 / 4 0$ </td><td rowspan=2 colspan=1> $\overline { { 9 \times 9 / } }$  $1 5 / 5 2$ </td></tr><tr><td rowspan=1 colspan=1>8/30</td></tr></table>

Table 8 Reasoning game dificulty configurations. Hero entries give rule template / hidden action blocks / budget multiplier; Sweepmine entries give board size / mines / action budget.

## B.2.2 Task Generation

Hero. We generate five episodes per task using diferent seeds, with templates and element pools adapted from the oficial implementation. Table 8 specifies the rule template, hidden block count, and action-budget multiplier. Override (Ovr) uses class-based defaults with a role-based exception; Ordered (Ord) requires inferring each block’s action type, repetition count, and relative order. The action budget is the reference solution length times the multiplier.

Sweepmine. We generate boards with seed-controlled mine placement. Dificulty-specific configurations specify the board size as rows × columns, the total number of mines, and the action budget as the maximum number of attempted actions across all board restarts (Table 8). The game rules and action interface remain fixed across levels.

## B.3 Exploring Games

## B.3.1 Task Definition

MC. Minecraft (MC) is a GameWorld browser-based first-person resource-collection and environmentexploration task. The agent starts with an empty inventory and uses visual observations to locate target resources, move, adjust its view, and interact with blocks to complete a collection goal within the step budget. Each step provides the task description, a short interaction history, and the current screenshot, and the agent returns one JSON action.

Astray. Astray is a GameWorld maze-navigation task in which the agent uses screenshots to guide a ball to the exit. We adapt the scoring to use the maximum relative reduction in breadth-first search (BFS) shortest-path distance to the exit over the run.

## B.3.2 Task Generation

MC. We define 20 training tasks with diferent collection goals and target quantities; evaluation uses the five fixed MC tasks from the oficial GameWorld evaluation set.

Astray. Training mazes have side length $L ( d ) = 2 d + 9$ for dificulty $d \in \{ 1 , \ldots , 5 \}$ , with layouts determined by random seeds. Evaluation uses the five fixed Astray tasks from the oficial GameWorld evaluation set.

## C Failure Patterns and RSI Refinement Examples

Table 9 presents failure patterns observed in both baseline systems and initial agent systems of evaluated RSI methods, along with representative refinements proposed by each evaluated RSI method to address these failures.

Table 9 Failure patterns and example refinements produced by ACE, AHE, HyperAgents, and HyperDSH. Each pattern was observed in at least one evaluated baseline’s traces. Red highlights flag flawed assumptions or rules, green highlights flag concrete strategies targeting the stated failure pattern and unhighlighted text presents general refinement guidance.

<table><tr><td colspan="1" rowspan="1">Game</td><td colspan="1" rowspan="1">Failure pattern</td><td colspan="1" rowspan="1">ACE refinement</td><td colspan="1" rowspan="1">AHE refinement</td><td colspan="5" rowspan="1">HyperAgentsrefinement</td><td colspan="1" rowspan="1">HyperDSHrefinement</td></tr><tr><td colspan="1" rowspan="4">Hero</td><td colspan="1" rowspan="4">Copyingdemonstrated plans,causing redundantactions.</td><td colspan="1" rowspan="4">[Experience]Decomposedemonstrations andcombine only actionblocks needed for thetarget attributes.</td><td colspan="1" rowspan="1">[Experience]</td><td colspan="5" rowspan="2">[Experience]Model each featureas an action block</td><td colspan="1" rowspan="3">[Experience]Select action blocksby target attributesto build the primaryplan and allow up totwo additionalrounds of blocksdrawn from otherdemonstrations.</td></tr><tr><td colspan="7" rowspan="1">Recombine blocks bytarget attribute,</td></tr><tr><td colspan="1" rowspan="2">retainingcontributions fromunchanged attributesand replacing othersonly with contrastiveevidence.</td><td colspan="7" rowspan="2">and combine theblocks inferred forthe target.[Error:   No guar-ànteed   one-to-oneattribute-block map-ping.]</td></tr><tr><td colspan="7" rowspan="1">[Correct:    Limitsretries while allowingalternative  actionblocks.]</td></tr><tr><td colspan="1" rowspan="1">Hero</td><td colspan="1" rowspan="1">Misinterpretingaction types, order,or counts indemonstrations.</td><td colspan="1" rowspan="1">[Experience]Infer action blocksthrough controlledcomparisons betweendemonstrations.</td><td colspan="1" rowspan="1">[Harness]Guide the agent toconstruct and runshell-based analysesthat extractdemonstrations andtrack evidence byattribute.</td><td colspan="5" rowspan="1">[Experience]Infer action typesand countsseparately.</td><td colspan="1" rowspan="1">[Experience]Select action blockswith their counts bytarget attribute, thencheck their order andplacement againstdemonstrations.</td></tr><tr><td colspan="1" rowspan="5">Sweepmine M</td><td colspan="1" rowspan="5">arking cellsincorrectly withoutchecking all currentboard constraints.</td><td colspan="1" rowspan="2">[Experience]Check all local</td><td colspan="1" rowspan="2">[Experience]Use stepwise</td><td colspan="5" rowspan="2">[Experience]Derive residual</td><td colspan="1" rowspan="2">[Experience]Use a checklist to</td></tr><tr><td colspan="2" rowspan="1">Derive res</td></tr><tr><td colspan="1" rowspan="3">constraints acrossthe board andrecompute themafter each action.</td><td colspan="1" rowspan="3">prompts to rebuildboard constraintsand incorporate thetotal mine-countconstraint.</td><td colspan="4" rowspan="2">constraints from thecurrent board and</td><td colspan="3" rowspan="1">from the</td><td colspan="1" rowspan="3">check localconstraints. A skillalso instructs theagent to trackfeasible mineplacements and usethe total mine countto guide inference.</td></tr><tr><td colspan="1" rowspan="1">currer</td><td colspan="2" rowspan="1">and</td></tr><tr><td colspan="5" rowspan="1">combine overlappingconstraints with thetotal mine count.</td></tr><tr><td colspan="1" rowspan="1">MC</td><td colspan="1" rowspan="1">Misidentifyingcollection targets orwhether they can becollected.</td><td colspan="1" rowspan="1">[Experience]Verify that the targetcan be mined.</td><td colspan="1" rowspan="1">[Harness]Verify the target'scurrent state. Aftera failed miningattempt, try adifferent target face.[Correct: A concretere-aiming policy aftermining failure.]</td><td colspan="5" rowspan="1">[Experience]Verify the target'spresence,reachability, andtype.</td><td colspan="1" rowspan="1">[Experience]Check target typeand state beforecollecting, andadditionally monitorcollection progress.</td></tr><tr><td colspan="1" rowspan="1">MC</td><td colspan="1" rowspan="1">Failing to adaptwhen navigationmakes no progress.</td><td colspan="1" rowspan="1">[Experience]Compare positionand distance to theresource before andafter movement toassess progress, thenreorient or excavateif movement isineffective.</td><td colspan="1" rowspan="1">[Harness]Šwitch to excavationtoward the resourcewhen explorationstalls.</td><td colspan="5" rowspan="1">[Harness]Prevent consecutiverepeated actions andturn or jump whenmovement stalls.[Correct: Enforces anàction-repeat limit incode.]</td><td colspan="1" rowspan="1">[Experience]Compare the currentposition with theresource's locationand stop movingfurther when theagent is far from it.</td></tr><tr><td colspan="1" rowspan="1">Astray</td><td colspan="1" rowspan="1">Misjudgingmovement directionor traversable paths</td><td colspan="1" rowspan="1">[Experience]Assess movementusing velocity anddisplacement alongthe commanded axis.</td><td colspan="1" rowspan="1">[Experience]Identify paths fromconnected floorregions and followconfirmed bendswithout repeatedlychanging direction.</td><td colspan="5" rowspan="1">[Experience]Assess movementusing nearby wallsand displacementalong thecommanded axis.</td><td colspan="1" rowspan="1">[Experience]Assess movementusing structuredcoordinate changes;use changes inposition relative tolandmarks insteadwhen coordinatedata are unavailable.</td></tr><tr><td colspan="1" rowspan="2">Astray</td><td colspan="1" rowspan="2">Confusing exploredroutes and knowndead ends withunexplored branches.</td><td colspan="1" rowspan="2">[Experience]Record explorationhistory andconnectivity to builda map.</td><td colspan="1" rowspan="2">[Experience]Carry forward acompact MEM ledgerof explored branchesand confirmed deadends in eachreasoning response.</td><td colspan="5" rowspan="1">[Experience]Add explicitwall-following andbacktracking rules.[Correct: A consis-tent strategy for sys-tematic maze explo-ration.]</td><td colspan="1" rowspan="1">[Experience]Follow a wall on afixed side, recordtried routes andswitch to an untriedroute afterconsecutive movesaway from the goal.</td></tr><tr><td colspan="5" rowspan="1"></td><td colspan="1" rowspan="1">[Correct: A consis-tent strategy for sys-tematic maze explo-ration.]</td></tr><tr><td colspan="1" rowspan="3">Combat</td><td colspan="1" rowspan="3">Prioritizing victoryor fast kills overremaining HP.</td><td colspan="1" rowspan="3">[Experience]Compare combatplans by remainingHP.</td><td colspan="1" rowspan="1">[Harness]Build a dynamicprogramming solverto maximizeremaining HP aftervictory.</td><td colspan="5" rowspan="1">[Harness]Build a solver tocompare completecombat plans andmaximize remainingHP after victory.</td><td colspan="1" rowspan="1">[Experience]Use text instructionsto compare combatplans throughdynamicprogramming and</td></tr><tr><td colspan="1" rowspan="1">[Correct: ImplementsHP-oriented search incode.]</td><td colspan="5" rowspan="1">[Correct: ImplementsHP-oriented search incode.]</td><td colspan="1" rowspan="2">maximize remainingHP.[Correct:  SpecifiesHP-oriented dynamicprogramming.]</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="5" rowspan="1"></td></tr></table>