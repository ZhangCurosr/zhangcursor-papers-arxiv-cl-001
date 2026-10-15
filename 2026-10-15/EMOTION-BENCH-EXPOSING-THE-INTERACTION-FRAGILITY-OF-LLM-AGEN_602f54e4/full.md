# τ -EMOTION-BENCH: EXPOSING THE INTERACTION FRAGILITY OF LLM AGENTS THROUGH REAL-WORLD USER EMOTIONS

Rishitosh Kumar Singh<sup>∗</sup> Amir Saeidi<sup>∗</sup> Shamanthak Hegde<sup>∗</sup>   
Shrinidhi Kumbhar Chitta Baral   
Arizona State University

## ABSTRACT

In long-horizon tool-calling tasks, LLM agents can fail not only because of tooluse errors, but also because of ineffective interaction with users. Yet existing benchmarks largely assume cooperative and behaviorally stable users, leaving the effect of emotional variation on agent reliability underexplored. We introduce τ-Emotion-Bench, a ten-domain benchmark for evaluating tool-calling agents under controlled emotion-conditioned user interactions while preserving the underlying task objective. To scale both benchmark construction and agent training, we introduce Tracer, a scalable agentic data-generation pipeline that constructs grounded, policy-valid tasks and collects trajectories through an interactive agent– user–environment harness. By allowing dissatisfied users to terminate inefficient interactions, the harness provides an interaction-aware signal that filters trajectories with excessive user dependence and favors effective tool use. Experiments with two user simulators show that frontier agents remain unreliable, with substantial gaps between average task success and consistent task completion. Tracer-generated supervision consistently improves Qwen3 models across scales, generalizes to unseen domains, and transfers to τ-Bench and τ-Trait. Behavioral analyses further show that higher task success is associated with a better balance between tool execution and necessary user follow-up. Overall, τ-Emotion-Bench and Tracer provide a scalable framework for evaluating agent reliability under diverse user behaviors and improving tool-calling agents through interaction-aware supervision.

Model https://huggingface.co/tau-emotion-bench/models Dataset https://huggingface.co/tau-emotion-bench/datasets Code https://github.com/sahsaeedi/tau-emotion-bench Website https://tau-emotion-bench.github.io

![](images/fcd7780cc78640deea8f6187446374b7402163bda7297d5492b17b6948fe3ac4.jpg)  
Figure 1: Average accuracy of LLM agents on τ -Emotion-Bench over three runs.

![](images/67f5407267b20861903fa8e3c1570bf0f3b120cfeee0b8d3658acacf84869a4e.jpg)  
Figure 2: Overview of τ-Emotion-Bench and the Tracer data-generation pipeline. Stage 1: Tracer builds policy-valid dependency graphs from tools, environments, and policies. Stage 2: Valid subgraphs are sampled to form executable multi-step traces. Stage 3: Traces are reverse-refined into natural user instructions while preserving task goals, context, preferences, and ambiguity. Stage 4: Task-specific feelings are mapped to structured emotion conditions and integrated with the task instruction. Stage 5: Tasks are verified and filtered for executability, policy consistency, and validity before inclusion in τ-Emotion-Bench.

## 1 INTRODUCTION

Large language models have rapidly evolved into agents capable of planning (Zhu et al., 2025), calling external tools, maintaining long-term memory (Uddin et al., 2026), and completing longhorizon tasks in interactive environments (Wang et al., 2024). Recent benchmarks have made these evaluations increasingly realistic by introducing complex tool dependencies, stateful environments, and multi-turn user interactions (Yao et al., 2024; He et al., 2025b; Yu et al., 2026b; Barres et al., 2025). In parallel, recent agent-training methods generate large collections of successful trajectories from executable environments (Li et al., 2025; Liu et al., 2026; Saeidi et al., 2026a; Wang et al., 2026; Xu et al., 2026b). However, both evaluation and data generation still largely rely on terminal task success as the main signal of agent quality (Saeidi et al., 2026c). A trajectory may eventually solve the task while repeatedly asking unnecessary questions, requesting information that could be obtained through tools, or relying excessively on the user instead of interacting with the environment (Mishra et al., 2025). Such trajectories receive positive task reward despite exhibiting undesirable behavior for user-facing agents. This exposes an important limitation of current agent harnesses, which measure whether the task is completed but provide little signal about whether the interaction itself is efficient.

We introduce τ-Emotion-Bench and Tracer to address this gap through an emotion-conditioned agent–user–environment harness. τ-Emotion-Bench contains 580 executable tasks across ten domains and systematically varies user behavior through five emotion families and fine-grained behavioral attributes while preserving the underlying task objective. Tracer constructs these tasks through grounded, policy-valid tool workflows and uses the same harness to generate training trajectories (See Figure 2). The key difference from conventional trajectory collection is that the user simulator contributes an additional interaction-aware reward signal. When an agent repeatedly asks unnecessary questions or fails to make sufficient progress, the user may become dissatisfied and terminate the interaction, preventing the trajectory from receiving a successful reward even if the task could eventually have been completed. Consequently, Tracer filters out successful but interaction-inefficient behavior and favors trajectories in which the agent asks only for information that is genuinely unavailable while relying more heavily on tools and the environment. This produces training data with a better balance between necessary user interaction and autonomous task execution.

Our experiments show that τ -Emotion-Bench remains challenging even for strong frontier models. With GLM-5.2 as the emotion-conditioned user simulator, the best evaluated model, GLM-5.3, reaches 78.6% Avg@3 but only 66.5% average Pass^3. Under Qwen3-235B-A22B-Instruct-2507 as the user simulator, the strongest model, DeepSeek-V4.1-Flash, reaches 55.9% Avg@3 and only 32.2% average Pass^3. These results leave substantial room for improvement in reliable task completion across both evaluation settings. Importantly, the gains from Tracer supervision remain substantial across model scales. Under GLM-5.2, Tracer-High improves the corresponding Qwen3-1.7B, 4B, and 8B base models by 17.1–25.2 percentage points in Avg@3, with Tracer-High-8B reaching 54.7%. Tracer-Low, trained on only 1,498 trajectories from four domains, generalizes to six unseen domains and achieves the strongest average Pass^3 among the compared agent-training methods at all three model scales. The learned behavior also transfers to τ-Bench (Yao et al., 2024) and τ-Trait (He et al., 2025a), which use different tasks and user simulators. Our behavioral analyses further show that higher task success is associated with a better balance between tool execution and conversational follow-up, while policy violations and unnecessary interaction remain important failure modes. Together, these results show that emotion-conditioned interaction can serve not only as a stress test for agent reliability, but also as a useful reward signal for collecting more effective agent-training trajectories. Our main contributions are summarized as follows.

• τ-Emotion-Bench. We introduce a ten-domain benchmark and emotion-conditioned agent– user–environment harness for evaluating tool-calling agents under diverse user behaviors while preserving the underlying task objective.

• Interaction-aware trajectory generation. We introduce a scalable pipeline for generating grounded agentic tasks and an interaction-aware reward signal that filters inefficient trajectories with excessive user dependence while favoring effective tool use.

• Generalizable agent training. We show that Tracer consistently improves Qwen3 models across scales and transfers to unseen domains, external benchmarks, and different user simulators with a relatively small training set.

Table 1: Comparison of existing user interaction benchmarks across task, tool, interaction, diversity, and scalability dimensions. “✓” indicates fully addressed, “✓” indicates partially addressed, and “✗” indicates not addressed. A “-” indicates that the corresponding information was not found.
<table><tr><td rowspan="3">Benchmark</td><td colspan="2">Task Complexity</td><td colspan="2">Tool Complexity</td><td colspan="3">Interaction Complexity</td><td rowspan="2">Diversity</td><td colspan="2">Scalability</td></tr><tr><td>Goal Clarity</td><td>Avg. Ground-Truth Tool Calls per Task</td><td># Tools</td><td>Inter-tool Dependency</td><td># Turns (approx.)</td><td>User Preferences Behavioral</td><td>Emotion</td><td>Automated # Domains Generation</td><td>Scalable</td></tr><tr><td>ToolTalk (Farn &amp; Shin, 2023)</td><td></td><td>3.4</td><td>28</td><td></td><td>[2, 10]</td><td>X</td><td>×</td><td>1</td><td>x</td><td>x</td></tr><tr><td>IN3 (Qian et al., 2024)</td><td>X</td><td></td><td></td><td>X</td><td>[2, 10]</td><td></td><td>x</td><td>1</td><td></td><td></td></tr><tr><td>MINT (Wang et al., 2023)</td><td></td><td></td><td>13</td><td>X</td><td>[2, 10]</td><td>X</td><td>×</td><td>8</td><td>X</td><td>X</td></tr><tr><td>ToolSandbox (Lu et al., 2025)</td><td></td><td>3.8</td><td>34</td><td></td><td>[10, 30]</td><td>X</td><td>x</td><td>11</td><td>X</td><td>X</td></tr><tr><td>DialogTool (Wang et al., 2025)</td><td></td><td>4.7</td><td>30</td><td></td><td>[10, 30]</td><td></td><td>X</td><td>15</td><td>x</td><td>X</td></tr><tr><td>UserBench (Qian et al., 2025)</td><td></td><td>2.5</td><td>5</td><td>+</td><td>[10,30]</td><td></td><td>x</td><td>1</td><td>X</td><td></td></tr><tr><td>τ-Bench (Yao et al., 2024)</td><td></td><td>3.0</td><td>28</td><td></td><td>[30,50]</td><td></td><td>X</td><td>2</td><td>x</td><td>X</td></tr><tr><td>τ-Trait (He et al., 2025a)</td><td></td><td>3.4</td><td>71</td><td></td><td>[3,61]</td><td></td><td></td><td>4</td><td></td><td>X</td></tr><tr><td>τ2-Bench (Barres et al., 2025)</td><td></td><td>3.2</td><td>38</td><td></td><td>[30, 80]</td><td></td><td>x</td><td>3</td><td></td><td>X</td></tr><tr><td>VitaBench (He et al., 2025b)</td><td>X</td><td></td><td>66</td><td></td><td>[50, 100]</td><td></td><td></td><td>3</td><td></td><td>x</td></tr><tr><td>WildToolBench (Yu et al., 2026a)</td><td>X</td><td>4.4</td><td>618</td><td></td><td></td><td>x</td><td>x</td><td>8</td><td></td><td>x</td></tr><tr><td>τ-Emotion-Bench (Ours)</td><td></td><td>6.5</td><td>173</td><td></td><td>[4, 164]</td><td>V</td><td></td><td>10</td><td></td><td></td></tr></table>

## 2 RELATED WORK

Interactive Tool-Use and User Simulation. Agent benchmarks have increasingly shifted from isolated function calls toward stateful interaction among agents, users, and executable environments. ToolTalk, ToolSandbox, and τ-Bench introduce conversational tool use, user simulation, and policyconstrained multi-turn interaction (Farn & Shin, 2023; Lu et al., 2025; Yao et al., 2024), while τ<sup>2</sup>-Bench, VitaBench, and WildToolBench further increase interaction complexity through shared environments, ambiguous requests, shifting intent, and conversational transitions (Barres et al., 2025; He et al., 2025b; Yu et al., 2026a). More recent work studies variation in user behavior, with τ-Trait controlling traits such as impatience and confusion and RealUserSim grounding user behavior in real conversational profiles (He et al., 2025a; Zhu et al., 2026a). Related affective benchmarks, including EIBench and TEA-Bench, focus on emotion management and emotional support (Zhu et al., 2026b; Sui et al., 2026). However, these works do not systematically measure how controlled emotional variation affects task-oriented tool use while preserving the same executable task and success conditions. As summarized in Table 1, τ-Emotion-Bench fills this gap by varying emotion and behavioral expression while keeping the underlying task and environment fixed.

Synthetic Environments and Agent Training. A parallel line of work scales agent training through synthetic environments and executable trajectories. VULCAN constructs deterministic tool-use environments from API specifications and collects executable multi-step trajectories (Saeidi et al., 2026a), while EnvScaler, Agent World Model, and Simia generate stateful environments and training data for supervised and reinforcement learning (Song et al., 2026; Wang et al., 2026; Li et al., 2025). EigenData and SPADE further improve synthetic data generation through executable verification and adaptive environment construction (Gao et al., 2026; Liu et al., 2026). However, these methods primarily assess trajectories through task success, verification, or environment difficulty. In contrast, Tracer introduces an interaction-aware signal through the emotion-conditioned user harness, where unnecessary dependence on the user can lead to dissatisfaction and terminate the trajectory. Consequently, Tracer favors successful trajectories that balance necessary clarification with autonomous tool and environment interaction rather than task completion alone.

## 3 τ -EMOTION-BENCH

## 3.1 PROBLEM FORMULATION

Following τ-Bench (Yao et al., 2024), we formulate each task as a partially observable Markov decision process (S, A, O, T, R, U), where S is the state space, A is the agent action space, O is the observation space, T is the transition function, R is the task reward, and U is the space of user instructions. The agent interacts with both an executable environment and a simulated user. Accordingly, the action space contains tool calls and natural-language responses, while observations contain tool outputs and user messages.

τ-Emotion-Bench extends this interaction with an emotion condition $e \in { \mathcal { E } }$ . Given a task instruction $u \in U$ and the interaction history $h _ { t } .$ , the user response at turn t is sampled according to

$$
u _ { t } \sim \pi _ { \mathrm { u s e r } } \left( \cdot \mid u , e , h _ { t } \right) .
$$

The task instruction, environment, and task objective remain unchanged across emotion conditions. The emotion condition is fixed within an episode, while its expression evolves with the interaction history. The simulator can therefore react differently to successful progress, repeated clarification, unnecessary requests, or other interaction behavior without changing the underlying task.

In addition to producing natural-language responses, the user can terminate the episode with either ###STOP### or ###UNSATISFIED###. The former indicates that the user considers the interaction complete, following the termination mechanism of τ-Bench, while the latter indicates that the accumulated interaction has become sufficiently unsatisfactory to continue. An episode therefore ends when the task interaction terminates normally, when the user becomes unsatisfied, or when the maximum interaction horizon is reached. If ###UNSATISFIED### is emitted before successful task completion, the trajectory receives −1 reward. This extension preserves the original task objective while allowing the interaction itself to affect whether a trajectory is considered successful.

## 3.2 TASK CONSTRUCTION

We construct grounded and executable tasks across ten tool-use domains, following the general interactive tool-use setup represented by τ-Bench, τ-Trait, and ToolOrchestra (Yao et al., 2024; He et al., 2025a; Su et al., 2025). Our construction separates the generation of valid tool workflows from their grounding in concrete environment states. For each domain, we construct a directed tool graph $G = ( V , E )$ following the dependency-based formulation of VULCAN (Saeidi et al., 2026a). Each node $\dot { v } \in V$ corresponds to a task-relevant tool, while an edge $( v _ { i } , v _ { j } ) \in E$ indicates that $v _ { j }$ can validly follow $v _ { i }$ according to the tool interfaces and domain policy. Candidate edges are inferred from tool definitions, execution behavior, and policy constraints, and the graph is manually verified before task generation. We then sample valid paths from G to obtain workflows of varying complexity and compose multiple paths when constructing multi-intent tasks. Detailed sampling configurations are provided in Appendix A, with additional environment details in Appendix D.

A sampled workflow defines the sequence of operations needed to complete a task, but leaves the specific entities and arguments unspecified. The grounding agent fills in these details using the executable environment, producing a valid reference trajectory. The preference agent then turns the grounded task into a natural user request and adds preferences that are feasible without changing the task’s intended outcome. Separately, the feeling generator uses the user context and requested actions to describe the user’s likely emotional state. It does not access the environment database, so the generated description cannot rely on information that would be unavailable to the user.

## 3.3 EMOTION MODELING AND USER SIMULATION

We next map the task-specific feeling generated in Section 3.2 to a controlled emotion condition. Our emotion space follows the Atlas of Emotions (Ekman, 2016) and contains five families with 22 fine-grained emotions. These are Anger (annoyance,frustration, argumentativeness, bitterness, exasperation), Fear (anxiety, nervousness, dread, desperation), Disgust (dislike, aversion, distaste, repugnance), Sadness (disappointment, discouragement, helplessness, hopelessness, sorrow), and Enjoyment (relief, pride, peace, excitement). We further combine each emotion with three behavioral dimensions covering politeness, urgency, and trust, yielding 594 conditions that capture both what the user feels and how that feeling is expressed. To map each free-form task feeling into this space, we use natural-language anchors and a multi-head classifier over the fine-grained emotion and behavioral attributes, then use semantic similarity to the anchors in the classifier’s learned embedding space to select the final condition. The corresponding natural-language instruction is provided to the user simulator rather than the discrete labels. Further details of the taxonomy, anchor construction, and mapping procedure are provided in Appendix A.1.

Given the selected emotion instruction, the user simulator generates each response conditioned on the task and the evolving conversation history. Rather than enforcing a fixed surface style, the instruction acts as a behavioral constraint, allowing the same emotion condition to be expressed differently as the interaction develops. This provides controlled variation in user behavior without scripting individual turns. The same interaction mechanism is used during Tracer trajectory collection, where the simulator’s responses provide an additional signal for selecting trajectories beyond environment-level task success.

## 3.4 TASK VERIFICATION AND DATA COMPOSITION

We verify generated tasks before including them in the benchmark. A task must specify a concrete objective, provide sufficient information for execution, admit an executable reference trajectory, and remain consistent with the corresponding domain policy. Reference trajectories are validated through deterministic replay, while model-based adjudication is used only when the deterministic outcome is inconclusive. All 580 evaluation tasks are additionally reviewed manually under the same criteria, and this process identifies 101 tasks requiring correction before inclusion. As an additional solvability check, a human tool-calling agent successfully completes selected tasks on which the evaluated models fail across all trials.

Table 2: Benchmark statistics for τ-Emotion-Bench domains.
<table><tr><td>Domain</td><td>Read Tools</td><td>Write Tools</td><td>Emotions (Family/Subfamily)</td><td>Test Tasks</td></tr><tr><td>Airline</td><td>8</td><td>6</td><td>5/16</td><td>70</td></tr><tr><td>Retail</td><td>9</td><td>7</td><td>5/17</td><td>70</td></tr><tr><td>Telecom</td><td>12</td><td>6</td><td>5/13</td><td>70</td></tr><tr><td>Telehealth</td><td>18</td><td>5</td><td>3/8</td><td>70</td></tr><tr><td>School</td><td>9</td><td>4</td><td>4/10</td><td>50</td></tr><tr><td>Basketball</td><td>17</td><td>3</td><td>1/2</td><td>50</td></tr><tr><td>Restaurant</td><td>9</td><td>6</td><td>2/6</td><td>50</td></tr><tr><td>Travel</td><td>9</td><td>7</td><td>2/6</td><td>50</td></tr><tr><td>Movie</td><td>13</td><td>3</td><td>3/7</td><td>50</td></tr><tr><td>Bank</td><td>13</td><td>9</td><td>3/8</td><td>50</td></tr></table>

The final benchmark contains 70 evaluation tasks from each of Airline, Retail, Telecom, and Tele-

health and 50 from each of School, Basketball, Restaurant, Travel, Movie, and Bank. We additionally generate 1,958 training trajectories across the ten domains. Table 2 summarizes the benchmark composition. Detailed statistics for evaluation-task construction and training-data generation are provided in Appendix A.

## 4 EXPERIMENTS

We evaluate τ-Emotion-Bench and Tracer through four questions. (Q1) How robust are frontier agents to emotion-conditioned users? (Q2) Can Tracer improve smaller agents and narrow the gap to frontier models? (Q3) Do Tracer-trained behaviors transfer to unseen tool-use domains with less training data? (Q4) Do these gains transfer to benchmarks with different tasks and user simulators?

Table 3: Performance comparison across ten τ-Emotion-Bench domains using Pass^3 (%).
<table><tr><td>Model</td><td>Airline Retail</td><td></td><td>Telecom</td><td>Telehealth</td><td>School</td><td>Basketball</td><td>Restaurant</td><td>Travel Movie Bank</td><td></td><td></td><td>|Avg.</td></tr><tr><td colspan="10">Non-thinking Models</td></tr><tr><td>Qwen3-32B</td><td>10.0</td><td>21.4</td><td>12.9</td><td>10.0</td><td>20.0</td><td>4.0</td><td>6.0</td><td>0.0</td><td>0.0</td><td>10.0</td><td>9.4</td></tr><tr><td>Qwen3-235B-A22B</td><td>17.1</td><td>15.7</td><td>10.0</td><td>21.4</td><td>22.0</td><td>24.0</td><td>4.0</td><td>0.0</td><td>0.0</td><td>24.0</td><td>13.8</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>32.9</td><td>44.3</td><td>17.1</td><td>31.4</td><td>42.0</td><td>38.0</td><td>26.0</td><td>30.0</td><td>22.0</td><td>38.0</td><td>32.2</td></tr><tr><td colspan="10">Thinking Models</td></tr><tr><td>Qwen3-32B</td><td>8.6</td><td>21.4</td><td>8.6</td><td>15.7</td><td>14.0</td><td>12.0</td><td>2.0</td><td>2.0</td><td>2.0</td><td>0.0</td><td>8.6</td></tr><tr><td>Qwen3-1.7B</td><td>0.0</td><td>0.0</td><td>8.6</td><td>2.9</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>1.2</td></tr><tr><td>Qwen3-4B</td><td>2.9</td><td>1.4</td><td>8.6</td><td>14.3</td><td>0.0</td><td>8.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>4.0</td><td>3.9</td></tr><tr><td>Qwen3-8B</td><td>4.3</td><td>14.3</td><td>12.9</td><td>22.9</td><td>0.0</td><td>12.0</td><td>4.0</td><td>2.0</td><td>0.0</td><td>12.0</td><td>8.4</td></tr><tr><td>Qwen3-235B-A22B-T</td><td>22.9</td><td>31.4</td><td>7.1</td><td>20.0</td><td>20.0</td><td>26.0</td><td>8.0</td><td>4.0</td><td>0.0</td><td>26.0</td><td>16.5</td></tr><tr><td>Qwen3.8-27B</td><td>32.9</td><td>38.6</td><td>11.4</td><td>30.0</td><td>48.0</td><td>28.0</td><td>30.0</td><td>16.0</td><td>10.0</td><td>28.0</td><td>27.3</td></tr><tr><td>MiniMax-M3</td><td>7.1</td><td>42.9</td><td>11.4</td><td>14.3</td><td>16.0</td><td>34.0</td><td>12.0</td><td>12.0</td><td>4.0</td><td>22.0</td><td>17.6</td></tr><tr><td>Muse-Glimmer-30B</td><td>20.0</td><td>0.0</td><td>8.6</td><td>14.3</td><td>64.0</td><td>28.0</td><td>10.0</td><td>18.0</td><td>12.0</td><td>30.0</td><td>20.5</td></tr><tr><td>MiniMax-M2.7</td><td>18.6</td><td>27.1</td><td>7.1</td><td>12.9</td><td>46.0</td><td>22.0</td><td>4.0</td><td>0.0</td><td>2.0</td><td>28.0</td><td>16.8</td></tr><tr><td>Gemma-4-31B-it</td><td>30.0</td><td>37.1</td><td>12.9</td><td>22.9</td><td>56.0</td><td>26.0</td><td>18.0</td><td>6.0</td><td>12.0</td><td>26.0</td><td>24.7</td></tr><tr><td>GLM-5.3</td><td>38.6</td><td>45.7</td><td>2.9</td><td>20.0</td><td>48.0</td><td>62.0</td><td>28.0</td><td>14.0</td><td>30.0</td><td>28.0</td><td>31.7</td></tr><tr><td>Kimi-K2.7-Code</td><td>30.0</td><td>44.3</td><td>10.0</td><td>14.3</td><td>44.0</td><td>34.0</td><td>12.0</td><td>12.0</td><td>8.0</td><td>52.0</td><td>26.1</td></tr><tr><td>Tracer-High-1.7B (Ours)</td><td>17.1</td><td>17.1</td><td>11.4</td><td>14.3</td><td>8.0</td><td>12.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>24.0</td><td>10.4</td></tr><tr><td>Tracer-High-4B (Ours)</td><td>22.9</td><td>30.0</td><td>8.6</td><td>22.9</td><td>34.0</td><td>20.0</td><td>6.0</td><td>0.0</td><td>2.0</td><td>26.0</td><td>17.2</td></tr><tr><td>Tracer-High-8B (Ours)</td><td>27.1</td><td>31.4</td><td>15.7</td><td>31.4</td><td>38.0</td><td>16.0</td><td>12.0</td><td>2.0</td><td>2.0</td><td>22.0</td><td>19.8</td></tr></table>

We first describe the evaluated models, training baselines, and evaluation protocol, followed by in-domain, out-of-domain, and cross-benchmark results.

## 4.1 EXPERIMENTAL SETUP

Models. We evaluate a broad set of strong open-weight and proprietary LLM agents, covering both thinking and non-thinking configurations when available. Our evaluation includes Qwen3-32B (Yang et al., 2025), Qwen3-235B-A22B-Instruct-2507, Qwen3-235B-A22B-Thinking-2507 (Yang et al., 2025), Qwen3.8-27B (Qwen Team, 2026), DeepSeek-V4.1-Flash (Xu et al., 2026a), Gemma-4- 31B-it (Team, 2026), GLM-5.3 (GLM-5-Team et al., 2026), Kimi-K2.7-Code (Team et al., 2025), MiniMax-M2.7, MiniMax-M3 (Chen et al., 2026) and Muse-Glimmer-30B (Meta Superintelligence Lab, 2026). To study whether Tracer-generated supervision can improve substantially smaller agents, we additionally fine-tune Qwen3-1.7B, Qwen3-4B, and Qwen3-8B under two training settings. Tracer-High is trained on 1,958 trajectories generated across all ten τ-Emotion-Bench domains, while Tracer-Low is trained on 1,498 trajectories from only Airline, Retail, Telecom, and Telehealth domains. This setup allows us to measure both the effectiveness of Tracer supervision and the extent to which the learned behavior transfers beyond the domains represented in the training data. Further hyperparameter details are included in Appendix B.1.

Agent-Training Baselines. For out-of-domain evaluation, we compare Tracer-Low against existing agent-training approaches applied to the same Qwen3 model scales whenever corresponding results are available. These include AReaL (Fu et al., 2026), EnvScaler (Song et al., 2026), Arctic-AWM (Wang et al., 2026), SPADE (Liu et al., 2026), Simia-Tau-SFT (Li et al., 2025), and Simia-Tau-SFT-RL. The comparison covers both supervised and reinforcement-learning-based agent training approaches and is particularly useful for evaluating whether Tracer’s comparatively small training set produces behaviors that transfer to unseen domains.

Evaluation and Metrics. We evaluate all models on 580 tasks across 10 executable environments in τ-Emotion-Bench, as summarized in Table 2. We use the τ-Bench agent harness and evaluate each model with two emotion-conditioned user simulators, Qwen3-235B-A22B-Instruct-2507 and GLM-5.2. Both simulators can terminate an interaction with ###STOP### or ###UNSATISFIED###. For consistency, the main results tables report performance with Qwen3-235B-A22B-Instruct-2507 as the user simulator, while detailed results with GLM-5.2 are provided in Appendix C.4. Each task is run for three trials with the user simulator at temperature 0.0 and a maximum of 200 interaction turns. We report Avg@3 as the average success rate across the three trials and Pass^k for k ∈ {1, 2, 3} as the fraction of tasks completed successfully in at least k trials, capturing both overall performance and consistency across repeated interactions.

Table 4: Performance comparison of Tracer with existing agent-trained models across six OOD domains on τ-Emotion-Bench using Pass^1 and Pass^3. Values are in %.
<table><tr><td rowspan="3">Method</td><td colspan="10">OOD Domains</td><td rowspan="2"></td><td rowspan="3"></td><td rowspan="3">Avg.</td></tr><tr><td colspan="2">School</td><td colspan="2">Basketball</td><td colspan="2">Restaurant</td><td colspan="2">Travel</td><td colspan="2">Movie</td><td rowspan="2">Avg. Pass^1 Pass^3</td></tr><tr><td>Pass^1 Pass^3</td><td></td><td>Pass^1 Pass^3</td><td></td><td>Pass^1 Pass^3 Pass^1 Pass^3 Pass^1 Pass^3</td><td></td><td></td><td></td><td></td><td>Pass^1 Pass^3</td></tr><tr><td colspan="10">Qwen3-1.7B</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>3.3</td><td>0.0</td><td>6.7</td><td>0.0</td><td>0.7</td><td>0.0 0.0</td><td></td><td>0.0</td><td>0.0</td><td>0.0</td><td>9.3</td><td>0.0 3.3</td><td>0.0</td></tr><tr><td>AReaL</td><td>4.0</td><td>2.0</td><td>4.0</td><td>0.0</td><td>0.7</td><td>0.0 0.0 0.0</td><td>0.0</td><td>0.0</td><td></td><td>0.0</td><td>2.7 0.0</td><td>1.9</td><td>0.3</td></tr><tr><td>EnvScaler</td><td>8.0</td><td>0.0</td><td>5.3</td><td>0.0</td><td>4.7</td><td>1.3</td><td>0.0</td><td>4.7</td><td>0.0</td><td>14.7</td><td>4.0</td><td>6.5</td><td>0.7</td></tr><tr><td>Tracer-Low (Ours)</td><td>9.3</td><td>8.0</td><td>14.0</td><td>0.0</td><td>6.0</td><td>2.0 0.0</td><td>0.0</td><td>4.7</td><td>0.0</td><td>16.7</td><td>6.0</td><td>8.4</td><td>2.7</td></tr><tr><td colspan="10">Qwen3-4B</td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>16.0</td><td>0.0</td><td>17.3</td><td>8.0</td><td>12.7 0.0</td><td>4.7</td><td>0.0</td><td>2.0</td><td>0.0</td><td>17.3</td><td>4.0</td><td>11.7</td><td>2.0</td></tr><tr><td>AReaL</td><td>5.3</td><td>2.0</td><td>15.3</td><td>2.0</td><td>8.7 2.0 0.0</td><td>0.7</td><td>0.0</td><td>1.3</td><td>0.0</td><td>14.0</td><td>4.0</td><td>7.6</td><td>1.7</td></tr><tr><td>EnvScaler</td><td>32.0</td><td>6.0</td><td>30.0</td><td>6.0</td><td>10.7 0.0</td><td>4.7</td><td>0.0</td><td>4.0</td><td>0.0</td><td>20.0</td><td>8.0</td><td>16.9</td><td>3.3</td></tr><tr><td>Arctic-AWM</td><td>11.3</td><td>4.0</td><td>14.7</td><td>2.0</td><td>12.7</td><td>3.3</td><td>0.0</td><td>2.7</td><td>0.0</td><td>20.7</td><td>4.0</td><td>10.9</td><td>1.7</td></tr><tr><td>SPADE</td><td>1.3 1</td><td>0.0</td><td>2.0</td><td>0.0</td><td>0.0</td><td>1.3</td><td>0.0</td><td>0.7</td><td>0.0</td><td>0.7</td><td>0.0</td><td>1.0</td><td>0.0</td></tr><tr><td>Tracer-Low (Ours)</td><td>21.3</td><td>6.0</td><td>36.0</td><td>18.0</td><td>22.0 8.0</td><td>4.7</td><td>0.0</td><td>2.0</td><td>0.0</td><td>25.3</td><td>14.0</td><td>18.6</td><td>7.7</td></tr><tr><td colspan="10">Qwen3-8B</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>24.7</td><td>0.0 2.0</td><td>26.7 20.0</td><td>12.0 2.0</td><td>16.7 4.7</td><td>4.0 0.0</td><td>2.7 0.7</td><td>2.0</td><td>3.3</td><td>0.0</td><td>24.7 12.0</td><td>16.5</td><td>5.0 1.0</td></tr><tr><td>AReaL EnvScaler</td><td>5.3 38.0</td><td>14.0</td><td>32.0</td><td>12.0</td><td></td><td>10.7</td><td>0.0 2.0</td><td>2.0 5.3</td><td>0.0 0.0</td><td>12.0 21.3</td><td>2.0 10.0</td><td>7.5 20.8</td><td>7.0</td></tr><tr><td>Arctic-AWM</td><td>24.7</td><td>4.0</td><td>19.3</td><td>4.0</td><td>17.3</td><td>4.0</td><td></td><td>4.0</td><td>0.0</td><td></td><td></td><td>14.2</td><td>2.3</td></tr><tr><td>Simia-Tau-SFT</td><td>19.3</td><td>6.0</td><td>26.0</td><td></td><td>11.3</td><td>2.0 6.0</td><td>0.0</td><td>0.0</td><td></td><td>20.0</td><td>4.0</td><td>14.5</td><td>4.3</td></tr><tr><td>Simia-Tau-SFT-RL</td><td>20.7</td><td>2.0</td><td>20.7</td><td>6.0</td><td>12.7</td><td>0.0 2.0</td><td>0.0</td><td>3.3</td><td>0.0</td><td>26.7</td><td>14.0</td><td>14.0</td><td>2.7</td></tr><tr><td>SPADE</td><td>2.7</td><td>0.0</td><td>1.3</td><td>2.0</td><td>11.3</td><td>2.0 2.7</td><td>0.0</td><td>0.7</td><td>0.0</td><td>25.3</td><td>10.0</td><td>0.8</td><td>0.0</td></tr><tr><td></td><td>1</td><td>1</td><td></td><td>0.0</td><td>0.0 1</td><td>0.0 0.0</td><td>0.0</td><td></td><td>0.0</td><td>0.0</td><td>0.0</td><td></td><td></td></tr><tr><td>Tracer-Low (Ours)</td><td>29.3</td><td>8.0</td><td>30.7</td><td>14.0</td><td>19.3</td><td>6.0 11.3</td><td>4.0</td><td>4.7</td><td>0.0</td><td>28.7</td><td>18.0</td><td>20.7</td><td>8.3</td></tr></table>

## 4.2 MAIN RESULTS

Frontier agents remain fragile under emotion-conditioned interactions. Figure 1 and Table 3 show that τ-Emotion-Bench remains challenging even for strong frontier models under Qwen3-235B-A22B-Instruct-2507 as the user simulator. DeepSeek-V4.1-Flash performs best with 55.9% Avg@3, followed by GLM-5.3 and Kimi-K2.7-Code at 51.9% and 49.3%, respectively. More importantly, performance drops substantially when consistency across repeated trials is required, with the highest average Pass^3 reaching only 32.2%. Moreover, performance varies considerably across domains. For example, DeepSeek-V4.1-Flash achieves 44.3% Pass^3 on Retail but only 17.1% on Telecom, while GLM-5.3 reaches 62.0% on Basketball but only 2.9% on Telecom. Overall, these results show that strong aggregate agent capability does not ensure reliable behavior under emotionally variable user interactions.

We additionally evaluate the same models with GLM-5.2 as the emotion-conditioned user simulator. Although absolute performance increases, the benchmark remains unsaturated. The strongest model, GLM-5.3, reaches 78.6% Avg@3 but only 66.5% average Pass^3, leaving a substantial gap in consistent task completion. Moreover, performance remains highly uneven across domains, with GLM-5.3 reaching 94.3% Pass^3 on Retail but only 40.0% on Telecom. These results show that the reliability gap exposed by τ -Emotion-Bench persists under a second strong user simulator. Detailed results with GLM-5.2 are provided in Appendix C.4.

Tracer-tuned models substantially narrow the gap to frontier models. As shown in Table 3 and Figure 1, Tracer tuning produces large and consistent improvements across all three Qwen3 scales under Qwen3-235B-A22B-Instruct-2507 as the user simulator. Tracer-High improves Avg@3 by 13.8–19.7% over the corresponding base models, with comparable gains from 1.7B through 8B. The improvement is also reflected in Pass^3, which increases from 1.2% to 10.4% at 1.7B, from 3.9% to 17.2% at 4B, and from 8.4% to 19.8% at 8B. Notably, Tracer-High-8B reaches 38.1% Avg@3, outperforming several larger frontier models, including MiniMax-M3 at 37.0% and Qwen3-235B-A22B-Thinking-2507 at 36.0%. These results show that the gains from Tracer tuning persist across model scales and substantially reduce the performance gap between small open-weight models and much larger frontier agents.

![](images/814c4d479a79a15038cb9c5e4350b8f172669c017bf70f09d45f6b0df19d21a6.jpg)  
Figure 3: Performance comparison of Qwen3 base, Tracer-Low, and Tracer-High models across τ-Bench and τ-Trait using Pass^1, Pass^2, and Pass^3.  
Qwen3 Tracer-Low (Ours) Tracer-High (Ours) Stronger Models Weaker Models

![](images/bb9496f51a15afdd83b266a18161547b5e4d8b9f5861a7819a00f8389a86ca9a.jpg)  
Figure 4: Comparison of model groups by policy violation rate across domains and emotions.

The gains remain substantial when GLM-5.2 is used as the emotion-conditioned user simulator. Tracer-High improves Avg@3 from 11.1% to 36.3% at 1.7B, from 31.2% to 51.4% at 4B, and from 37.6% to 54.7% at 8B, corresponding to improvements of 17.1–25.2 percentage points. The same pattern appears under the stricter Pass^3 metric, which increases from 4.0% to 19.4% at 1.7B, from 16.2% to 34.2% at 4B, and from 22.6% to 37.5% at 8B. Notably, Tracer-High-8B reaches 54.7% Avg@3, exceeding Qwen3-235B-A22B-Instruct-2507 at 51.7% and matching Muse-Glimmer-30B. Together, the two evaluation settings show that Tracer’s improvements are preserved across different emotion-conditioned user simulators.

Tracer-tuned models generalize effectively to unseen domains. We train Tracer-Low on 1,498 trajectories from four domains and evaluate it on six domains that are entirely excluded from training. As shown in Table 4, Tracer-Low consistently improves the corresponding Qwen3 base models and remains competitive with substantially more data-intensive agent-training methods. At 8B, Tracer-Low achieves 20.7% average Pass^1 and 8.3% Pass^3, outperforming EnvScaler, Arctic-AWM, Simia-Tau-SFT, and Simia-Tau-SFT-RL in average Pass^3. More importantly, these gains are obtained with only 1,498 training trajectories, compared with substantially larger training collections used by prior methods, including 9,000 SFT trajectories for EnvScaler and 91,204 synthetic trajectories for Simia-Tau-SFT. Moreover, Tracer-Low achieves the highest average Pass^3 among the compared methods at 1.7B, 4B, and 8B scales. Together, these results suggest that Tracer’s interaction-aware signal supports a transferable balance between necessary follow-up and effective tool use, enabling strong generalization beyond the training domains with substantially less data.

![](images/ea60d4c7a613da0439e52f9d083f3fd141aa204c97c9f5c44d1b2a768dfdab0a.jpg)  
Figure 5: Avg@3 versus tool-call-to-follow-up ratio across emotion conditions and model groups.

Tracer-tuned models transfer across benchmarks and user simulators. We additionally evaluate Tracer-Low and Tracer-High on τ-Bench Airline and Retail domains and on τ-Trait Telehealth and Telecom domains, which differ from τ-Emotion-Bench in both task distribution and user simulation. As shown in Figure 3, Tracer tuning consistently improves the corresponding Qwen3 base models across model scales, with particularly strong gains on Retail and Telehealth and clear improvements in the stricter Pass^2 and Pass^3 metrics. Importantly, these gains persist even when the evaluation tasks and user simulators differ from those used during Tracer training. This suggests that Tracer learns interaction and tool-use behaviors that transfer beyond the specific τ-Emotion-Bench itself.

## 5 RESULTS ANALYSIS

Aggregate task success does not fully explain how emotion-conditioned interactions affect agent behavior. We therefore analyze the evaluation trajectories along four complementary dimensions. First, we examine policy adherence and the extent to which Tracer tuning reduces unconfirmed or unnecessary actions. Second, we study how task success relates to the balance between tool execution and conversational follow-up, and how this behavior changes after training. Third, we analyze the relationship between empathetic language, follow-up interaction, and task success. Finally, we examine user hallucinations under different emotion conditions and domains to assess the reliability of the simulator. Together, these analyses characterize both the behavioral failures exposed by emotion-conditioned interaction and the stability of the user simulator used to measure them.

Policy violations persist under emotionconditioned interactions. Figure 4 shows that policy adherence generally improves with overall agent capability. For this analysis, we group frontier models by their Avg@3 relative to Tracer-High-8B, with weaker models scoring below it and stronger models scoring above it. Violation rates are generally higher for base Qwen3 and weaker models and lower for

![](images/e6d3dcee31e545b53cf439c8895e646ddfd1ee1070aca083844c633b428cace1.jpg)

![](images/a864c10ae695af1aca008143e0cab49f705a7de42bbf29e555c88f7a14e2bba2.jpg)  
Figure 6: Left: Change in task success associated with empathy across emotions and model groups. Right: Fraction of follow-up questions with and without empathetic responses.

![](images/0353273385967983bf07cdbfd42c7efad584a6c835e948703355c5e2a1355e0f.jpg)  
Figure 7: User hallucination rates across subemotions and domains in failed trajectories.

Tracer-High and stronger models. However, substantial violations remain in Airline, Telecom, and Telehealth, with some conditions reaching approximately 67% in Telehealth and 91% in Telecom even among stronger models. Moreover, the similar behavior of Tracer-Low and Tracer-High, including on unseen domains, suggests that Tracer tuning transfers a general tendency to verify user intent before acting. Overall, greater model capability and Tracer tuning generally reduce policy violations, but emotion-conditioned interactions continue to expose failures in policy adherence.

![](images/91784cd276bf3472ad324474cf1e03475361e18556b32eef070db7ed4b0e3d19.jpg)  
Figure 8: Distribution of tool calls and follow-up actions across AReaL, Simia, Tracer-Low, and Tracer-High training trajectories.

Successful interaction depends on a balance between tool use and follow-up. Figure 5 shows that higher performance is generally associated with a moderate balance between tool execution and conversational follow-up across all five emotion conditions. Higher-performing models and Tracertuned models tend to occupy this region while achieving higher Avg@3, whereas more aggressive tool use does not consistently lead to better performance. This pattern is particularly clear under Enjoyment, Disgust, and Fear. Moreover, Tracer tuning generally improves performance over the corresponding Qwen3 base models without shifting toward more tool-heavy interaction. Overall, the figure suggests that successful agents make sufficient progress through tools while limiting follow-up to cases where additional user information is needed.

Empathy is associated with more follow-up. Figure 6 shows that the relationship between empathy and task success varies across emotions. Empathy is associated with lower success under Enjoyment across all model groups and under Fear for most groups, while its effect under Disgust varies substantially across models. Small gains also appear in several settings, particularly under Sadness. More consistently, empathetic responses are accompanied by a higher fraction of follow-up questions across every model group. Together with the previous analysis, this suggests that additional empathetic interaction can increase conversational overhead when it does not resolve uncertainty or advance the task. Overall, effective empathy should acknowledge the user’s emotional state without introducing unnecessary interaction.

Table 5: Performance comparison of Tracer with existing agent-trained models across three in-domain domains on τ-Emotion-Bench using Pass^1 and Pass^3.
<table><tr><td rowspan="3">Method</td><td colspan="6">In-Domain</td><td rowspan="3">Avg.</td></tr><tr><td colspan="2">Airline</td><td colspan="2">Retail</td><td colspan="2">Telecom</td></tr><tr><td>Pass^1</td><td>Pass^3</td><td>Pass^1</td><td>Pass^3</td><td>Pass^1 Pass^3</td><td>Pass^1</td></tr><tr><td colspan="8">Qwen3-1.7B</td></tr><tr><td>Base</td><td>1.0</td><td>0.0</td><td>0.5</td><td>0.0</td><td>23.8</td><td>8.6</td><td>8.4</td><td>2.9</td></tr><tr><td>AReaL</td><td>11.0</td><td>1.4</td><td>10.0</td><td>1.4</td><td>13.3</td><td>2.9</td><td>11.4</td><td>1.9</td></tr><tr><td>Tracer-Low (Ours)</td><td>29.5</td><td>10.0</td><td>40.9</td><td>18.6</td><td>32.4</td><td>10.0</td><td>34.3</td><td>12.9</td></tr><tr><td>Tracer-High (Ours)</td><td>34.3</td><td>17.1</td><td>40.0</td><td>17.1</td><td>34.3</td><td>11.4</td><td>36.2</td><td>15.2</td></tr><tr><td colspan="9">Qwen3-4B</td></tr><tr><td>Base</td><td>18.6</td><td>2.9</td><td>20.0</td><td>1.4</td><td>34.3</td><td>8.6</td><td>24.3</td><td>4.3</td></tr><tr><td>AReaL</td><td>20.0</td><td>2.9</td><td>25.2</td><td>5.7</td><td>25.7</td><td>10.0</td><td>23.6</td><td>6.2</td></tr><tr><td>Tracer-Low (Ours)</td><td>43.3</td><td>24.3</td><td>53.3</td><td>21.4</td><td>40.5</td><td>15.7</td><td>45.7</td><td>20.5</td></tr><tr><td>Tracer-High (Ours)</td><td>42.9</td><td>22.9</td><td>56.7</td><td>30.0</td><td>37.6</td><td>8.6</td><td>45.7</td><td>20.5</td></tr><tr><td colspan="9">Qwen3-8B</td></tr><tr><td>Base</td><td>20.0</td><td>4.3</td><td>37.1</td><td>14.3</td><td>41.4</td><td>12.9</td><td>32.8</td><td>10.5</td></tr><tr><td>AReaL</td><td>16.7</td><td>5.7</td><td>27.6</td><td>8.6</td><td>28.6</td><td>10.0</td><td>24.3</td><td>8.1</td></tr><tr><td>Simia-Tau-SFT</td><td>30.5</td><td>8.6</td><td>36.7</td><td>14.3</td><td>33.3</td><td>11.4</td><td>33.5</td><td>11.4</td></tr><tr><td>Simia-Tau-SFT-RL</td><td>30.0</td><td>12.9</td><td>39.1</td><td>14.3</td><td>35.2</td><td>11.4</td><td>34.8</td><td>12.9</td></tr><tr><td>Tracer-Low (Ours)</td><td>46.7</td><td>30.0</td><td>54.8</td><td>22.9</td><td>41.9</td><td>20.0</td><td>47.8</td><td>24.3</td></tr><tr><td>Tracer-High (Ours)</td><td>44.8</td><td>27.1</td><td>56.7</td><td>31.4</td><td>37.6</td><td>15.7</td><td>46.4</td><td>24.7</td></tr></table>

Emotion conditioning does not destabilize the user simulator. A potential confound in τ- Emotion-Bench is that emotional conditioning could cause the simulated user to deviate from its task instruction, making some apparent agent failures originate from the simulator. We therefore examine 6,961 failed trajectories, flag 449 user turns as potential hallucinations, and verify each against the original task instruction with an LLM judge (See Appendix C.1). Only 124 are confirmed as user errors, corresponding to 1.1% of failed trials and at most 1.4% for any emotion family. Moreover, Figure 7 shows that most emotion and domain combinations contain no hallucinations, with the few elevated rates appearing as isolated cases. Although some annotation uncertainty may remain, these results suggest that emotion conditioning does not meaningfully compromise simulator reliability. See Appendices C and E for additional error analysis, task coverage analysis, and the human study.

Training interaction patterns may help explain Tracer’s generalization. To investigate why Tracer-trained models generalize more effectively, we test the hypothesis that training trajectories which emphasize progress through tools while limiting unnecessary follow-up encourage agents to rely more on the environment and less on continued user interaction. Figure 8 compares this interaction pattern across Tracer-Low and Tracer-High and the publicly available training trajectories from two prior agent-training approaches, AReaL and Simia. Tool calls exceed follow-up actions in 76.2% of Tracer-Low and 78.9% of Tracer-High trajectories, compared with 55.6% for AReaL and only 22.0% for Simia. Conversely, 51.9% of Simia trajectories contain more follow-up actions than tool calls, compared with 12.8% for Tracer-Low and 10.6% for Tracer-High. This distribution is consistent with the interaction-aware signal used by Tracer, which is designed to discourage unnecessary user dependence during trajectory collection. Together, these observations support the hypothesis that the interaction-aware signal helps collect trajectories with a more effective balance between necessary user interaction and autonomous tool use. Since the datasets also differ in other aspects of their construction, this evidence remains correlational.

## 6 CONCLUSION

We introduced τ-Emotion-Bench, a ten-domain benchmark for evaluating tool-calling agents under controlled emotion-conditioned user interactions. Our evaluation reveals substantial performance degradation and persistent policy failures when agents interact with emotionally variable users. We further proposed Tracer, a scalable approach for generating grounded agentic tasks and training trajectories through an emotion-conditioned agent–user–environment harness. Tracer-tuned models achieve strong performance across both in-domain and out-of-domain evaluations while using substantially less training data than prior agent-training methods. Our behavioral analysis suggests that these gains are linked to a more effective balance between tool execution and necessary user follow-up. This behavior is encouraged by the interaction-aware signal exposed by the harness, which favors efficient task progress rather than task completion alone. Overall, our findings highlight user behavior as an important dimension of agent reliability and show that interaction-aware data generation can provide efficient and transferable supervision for tool-calling agents.

## LIMITATIONS

Our study has two main limitations. First, we evaluate Tracer-generated data only through supervised fine-tuning, and its effectiveness as supervision for reinforcement learning remains unexplored. Second, our experiments focus on tool-calling agents. Extending τ-Emotion-Bench and Tracer to other agent settings, such as coding and web-based agents, is an interesting direction for future work.

## ACKNOWLEDGMENTS

We thank the Research Computing (RC) at Arizona State University (ASU) for their generous support in providing computing resources (Jennewein et al., 2026). The views and opinions of the authors expressed herein do not necessarily state or reflect those of the funding agencies and employers. This research was also partially supported by funding from DARPA.

## AI USE STATEMENT

We used AI tools to brainstorm ideas during the early stages of this work and to help identify related research alongside our manual literature search. We also used them to improve the clarity and wording of the manuscript. The authors reviewed the cited work and take responsibility for the paper’s content and conclusions.

## REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. τ<sup>2</sup>-bench: Evaluating conversational agents in a dual-control environment. arXiv preprint arXiv:2506.07982, 2025.

Aili Chen, Aonian Li, Baichuan Zhou, Bangwei Gong, Binyang Jiang, Boji Dan, Changhao Zhang, Changqing Yu, Chao Wang, Cheng Ma, et al. The minimax-m2 series: Mini activations unleashing max real-world intelligence. arXiv preprint arXiv:2605.26494, 2026.

Paul Ekman. What scientists who study emotion agree about. Perspectives on Psychological Science, 11(1):31–34, 2016. doi: 10.1177/1745691615596992. URL https://doi.org/10.1177/ 1745691615596992. PMID: 26817724.

Nicholas Farn and Richard Shin. Tooltalk: Evaluating tool-usage in a conversational setting. arXiv preprint arXiv:2311.10775, 2023.

Wei Fu, Jiaxuan Gao, Xujie Shen, Chen Zhu, Zhiyu Mei, Chuyi He, Shusheng Xu, Guo Wei, Jun Mei, Jiashu Wang, et al. Areal: A large-scale asynchronous reinforcement learning system for language reasoning. Advances in Neural Information Processing Systems, 38:36256–36282, 2026.

Jiaxuan Gao, Jiaao Chen, Chuyi He, Shusheng Xu, Di Jin, and Yi Wu. From self-evolving synthetic data to verifiable-reward rl: Post-training multi-turn interactive tool-using agents. arXiv preprint arXiv:2601.22607, 2026.

GLM-5-Team, :, Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, Chenzheng Zhu, Congfeng Yin, Cunxiang Wang, Gengzheng Pan, Hao Zeng, Haoke Zhang, Haoran Wang, Huilong Chen, Jiajie Zhang, Jian Jiao, Jiaqi Guo, Jingsen Wang, Jingzhao Du, Jinzhu Wu, Kedong Wang, Lei Li, Lin Fan, Lucen Zhong, Mingdao Liu, Mingming Zhao, Pengfan Du, Qian Dong, Rui Lu, Shuang-Li, Shulin Cao, Song Liu, Ting Jiang, Xiaodong Chen, Xiaohan Zhang, Xuancheng Huang, Xuezhen Dong, Yabo Xu, Yao Wei, Yifan An, Yilin Niu, Yitong Zhu, Yuanhao Wen, Yukuo Cen, Yushi Bai, Zhongpei Qiao, Zihan Wang, Zikang Wang, Zilin Zhu, Ziqiang Liu, Zixuan Li, Bojie Wang, Bosi Wen, Can Huang, Changpeng Cai, Chao Yu, Chen Li, Chengwei Hu, Chenhui Zhang, Dan Zhang, Daoyan Lin, Dayong Yang, Di Wang, Ding Ai, Erle Zhu, Fangzhou Yi, Feiyu Chen, Guohong Wen, Hailong Sun, Haisha Zhao, Haiyi Hu, Hanchen Zhang, Hanrui Liu, Hanyu Zhang, Hao Peng, Hao Tai, Haobo Zhang, He Liu, Hongwei Wang, Hongxi Yan, Hongyu Ge, Huan Liu, Huanpeng Chu, Jia’ni Zhao, Jiachen Wang, Jiajing Zhao, Jiamin Ren, Jiapeng Wang, Jiaxin Zhang, Jiayi Gui, Jiayue Zhao, Jijie Li, Jing An, Jing Li, Jingwei Yuan, Jinhua Du, Jinxin Liu, Junkai Zhi, Junwen Duan, Kaiyue Zhou, Kangjian Wei, Ke Wang, Keyun Luo, Laiqiang Zhang, Leigang Sha, Liang Xu, Lindong Wu, Lintao Ding, Lu Chen, Minghao Li, Nianyi Lin, Pan Ta, Qiang Zou, Rongjun Song, Ruiqi Yang, Shangqing Tu, Shangtong Yang, Shaoxiang Wu, Shengyan Zhang, Shijie Li, Shuang Li, Shuyi Fan, Wei Qin, Wei Tian, Weining Zhang, Wenbo Yu, Wenjie Liang, Xiang Kuang, Xiangmeng Cheng, Xiangyang Li, Xiaoquan Yan, Xiaowei Hu, Xiaoying Ling, Xing Fan, Xingye Xia, Xinyuan Zhang, Xinze Zhang, Xirui Pan, Xu Zou, Xunkai Zhang, Yadi Liu, Yandong Wu, Yanfu Li, Yidong Wang, Yifan Zhu, Yijun Tan, Yilin Zhou, Yiming Pan, Ying Zhang, Yinpei Su, Yipeng Geng, Yong Yan, Yonglin Tan, Yuean Bi, Yuhan Shen, Yuhao Yang, Yujiang Li, Yunan Liu, Yunqing Wang, Yuntao Li, Yurong Wu, Yutao Zhang, Yuxi Duan, Yuxuan Zhang, Zezhen Liu, Zhengtao Jiang, Zhenhe Yan, Zheyu Zhang, Zhixiang Wei, Zhuo Chen, Zhuoer Feng, Zijun Yao, Ziwei Chai, Ziyuan Wang, Zuzhou Zhang, Bin Xu, Minlie Huang, Hongning Wang, Juanzi Li, Yuxiao Dong, and Jie Tang. Glm-5: from vibe coding to agentic engineering, 2026. URL https://arxiv.org/abs/2602.15763.

Muyu He, Anand Kumar, Tsach Mackey, Meghana Rajeev, James Zou, and Nazneen Rajani. Impatient users confuse ai agents: High-fidelity simulations of human traits for testing agents. arXiv preprint arXiv:2510.04491, 2025a.

Wei He, Yueqing Sun, Hongyan Hao, Xueyuan Hao, Zhikang Xia, Qi Gu, Chengcheng Han, Dengchang Zhao, Hui Su, Kefeng Zhang, et al. Vitabench: Benchmarking llm agents with versatile interactive tasks in real-world applications. arXiv preprint arXiv:2509.26490, 2025b.

Douglas Jennewein, Alan Chapman, Alejandro Chiquete, Amber Carlson, Gil Speyer, Glen Otero, Jeevesh Choudhry, Johnathan Lee, Joshua Burks, Juan José García Mesa, Marisa Brazil, Natalie Mendoza, Neal Woodbury, Nil Tianchen Mu, Rebecca Belshe, Sean Dudley, Thomas Jennewein, Torey Battelle, William Dizon, and Young-Moo Byun. Provisioning a largescale ai platform composed of intel gaudi2 accelerators at arizona state university. In Proceedings of the Practice and Experience in Advanced Research Computing 2026: Resilient Roots + Empowered Communities, PEARC ’26, New York, NY, USA, 2026. Association for Computing Machinery. ISBN 9798400723773. doi: 10.1145/3785462.3815814. URL https://doi.org/10.1145/3785462.3815814.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Yuetai Li, Huseyin A Inan, Xiang Yue, Wei-Ning Chen, Lukas Wutschitz, Janardhan Kulkarni, Radha Poovendran, Robert Sim, and Saravan Rajmohan. Simulating environments with reasoning models for agent training. arXiv preprint arXiv:2511.01824, 2025.

Bo Liu, Simon Yu, Yiding Jiang, Ao Qu, Andrew Zhao, Zichen Liu, Junsu Kim, Zijian Zhou, Seungone Kim, Tongzheng Ren, et al. Spade: Self-play in adaptive synthetic executable environments. arXiv preprint arXiv:2608.19197, 2026.

Jiarui Lu, Thomas Holleis, Yizhe Zhang, Bernhard Aumayer, Feng Nan, Haoping Bai, Shuang Ma, Shen Ma, Mengyu Li, Guoli Yin, et al. Toolsandbox: A stateful, conversational, interactive evaluation benchmark for llm tool use capabilities. In Findings of the Association for Computational Linguistics: NAACL 2025, pp. 1160–1183, 2025.

Meta Superintelligence Lab. Muse Glimmer model card. Hugging Face, August 2026. URL https: //huggingface.co/meta-models/Muse-Glimmer-30B. Model: Muse-Glimmer-30B. Accessed: 10 October 2026.

Venkatesh Mishra, Amir Saeidi, Satyam Raj, Mutsumi Nakamura, Jayanth Srinivasa, Gaowen Liu, Ali Payani, and Chitta Baral. How can input reformulation improve tool usage accuracy in a complex dynamic environment? a study on τ-bench. arXiv preprint arXiv:2508.20931, 2025.

Cheng Qian, Bingxiang He, Zhong Zhuang, Jia Deng, Yujia Qin, Xin Cong, Zhong Zhang, Jie Zhou, Yankai Lin, Zhiyuan Liu, et al. Tell me more! towards implicit user intention understanding of language model driven agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1088–1113, 2024.

Cheng Qian, Zuxin Liu, Akshara Prabhakar, Zhiwei Liu, Jianguo Zhang, Haolin Chen, Heng Ji, Weiran Yao, Shelby Heinecke, Silvio Savarese, et al. Userbench: An interactive gym environment for user-centric agents. arXiv preprint arXiv:2507.22034, 2025.

Qwen Team. Qwen3.8-Max: A new bar for coding and cowork, August 2026. URL https: //qwen.ai/blog?id=qwen3.8.

Amir Saeidi, Chitta Baral, Ahmed Hassan Awadallah, and Harkirat Behl. Vulcan: Where agents learn by living in simulated tool environments. In ICLR 2026 Workshop on Data-FM, March 2026a. Last modified: April 1, 2026.

Amir Saeidi, Venkatesh Mishra, Souradeep Mukhopadhyay, Gaowen Liu, Ali Payani, Jayanth Srinivasa, and Chitta Baral. Fama: Failure-aware meta-agentic framework for open-source llms in interactive tool use environments. arXiv preprint arXiv:2604.25135, 2026b.

Amir Saeidi, Zehua Zhang, Rishitosh Singh, Naman Ahuja, Vivek Gupta, Ali Payani, Gaowen Liu, Jayanth Srinivasa, and Chitta Baral. Cast: Critique-aware supervision for training reliable long-horizon tool-calling agents. arXiv preprint arXiv:2608.30147, 2026c.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

Xiaoshuai Song, Haofei Chang, Guanting Dong, Yutao Zhu, Ji-Rong Wen, and Zhicheng Dou. Envscaler: Scaling tool-interactive environments for llm agent via programmatic synthesis. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pp. 8326–8357, 2026.

Hongjin Su, Shizhe Diao, Ximing Lu, Mingjie Liu, Jiacheng Xu, Xin Dong, Yonggan Fu, Peter Belcak, Hanrong Ye, Hongxu Yin, et al. Toolorchestra: Elevating intelligence via efficient model and tool orchestration. arXiv preprint arXiv:2511.21689, 2025.

Xingyu Sui, Yanyan Zhao, Yulin Hu, Jiahe Guo, Weixiang Zhao, and Bing Qin. Tea-bench: A systematic benchmarking of tool-enhanced emotional support dialogue agent. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 46390–46416, 2026.

Gemma Team. Gemma 4 technical report, 2026. URL https://arxiv.org/abs/2607. 02770.

Kimi Team, Yifan Bai, Yiping Bao, Y Charles, Cheng Chen, Guanduo Chen, Haiting Chen, Huarong Chen, Jiahao Chen, Ningxin Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.

Md Nayem Uddin, Kumar Shubham, Eduardo Blanco, Chitta Baral, and Gengyu Wang. From recall to forgetting: Benchmarking long-term memory for personalized agents. In Findings of the Association for Computational Linguistics: ACL 2026, pp. 26814–26841, 2026.

Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan Lambert, Shengyi Huang, Kashif Rasul, and Quentin Gallouédec. TRL: Transformers Reinforcement Learning, 2020. URL https://github.com/huggingface/trl.

Hongru Wang, Wenyu Huang, Yufei Wang, Yuanhao Xi, Jianqiao Lu, Huan Zhang, Nan Hu, Zeming Liu, Jeff Z Pan, and Kam-Fai Wong. Rethinking stateful tool use in multi-turn dialogues: Benchmarks and challenges. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 5433–5453, 2025.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):186345, 2024.

Xingyao Wang, Zihan Wang, Jiateng Liu, Yangyi Chen, Lifan Yuan, Hao Peng, and Heng Ji. 375 mint: Evaluating llms in multi-turn interaction with tools and language feedback. arxiv preprint 376. arXiv preprint arXiv:2309.10691, 377, 2023.

Zhaoyang Wang, Canwen Xu, Boyi Liu, Yite Wang, Siwei Han, Zhewei Yao, Huaxiu Yao, and Yuxiong He. Agent world model: Infinity synthetic environments for agentic reinforcement learning. arXiv preprint arXiv:2602.10090, 2026.

Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al. Deepseek-v4: Towards highly efficient milliontoken context intelligence. arXiv preprint arXiv:2606.19348, 2026a.

Minrui Xu, Zilin Wang, Mengyi Deng, Zhiwei Li, Zhicheng Yang, Xiao Zhu, Yinhong Liu, Boyu Zhu, Baiyu Huang, Chao Chen, et al. Envfactory: Scaling tool-use agents via executable environments synthesis and robust rl. arXiv preprint arXiv:2605.18703, 2026b.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ-bench: A benchmark for tool-agent-user interaction in real-world domains. arXiv preprint arXiv:2406.12045, 2024.

Peijie Yu, Wei Liu, Yifan Yang, Jinjian Li, Zelong Zhang, Xiao Feng, and Feng Zhang. Benchmarking llm tool-use in the wild. arXiv preprint arXiv:2604.06185, 2026a.

Peijie Yu, Wei Liu, Yifan Yang, Jinjian Li, Zelong Zhang, Xiao Feng, and Feng Zhang. Benchmarking LLM tool-use in the wild. In The Fourteenth International Conference on Learning Representations, 2026b. URL https://openreview.net/forum?id=yz7fL5vfpn.

Ming Zhu, Juntao Tan, Rithesh Murthy, Jielin Qiu, Liangwei Yang, Wenting Zhao, Silvio Savarese, Shelby Heinecke, and Huan Wang. Realusersim: Bridging the reality gap in agent benchmarking via grounded user simulation. arXiv preprint arXiv:2605.20204, 2026a.

Rongzhi Zhu, Xiang Huang, Yuchuan Wu, Rui Wang, Zequn Sun, Tao Ren, Weiyao Luo, Bingxue Qiu, Jieping Ye, Yongbin Li, et al. Eibench: A simulator-based benchmark and turn-credit rl for emotion management. arXiv preprint arXiv:2606.15532, 2026b.

Siyu Zhu, Yanbin Jiang, Hejian Sang, Shao Tang, Qingquan Song, Biao He, Rohit Jain, Zhipeng Wang, and Alborz Geramifard. Planner-r1: Reward shaping enables efficient agentic rl with smaller llms. arXiv preprint arXiv:2509.25779, 2025.

## Table of Contents for Appendix

A Data Statistics .   
A.1 Emotion Modeling Details . 17   
A.2 Evaluation Data 17   
A.3 Training Data. 18   
A.4 Successful Training Trajectories 18   
B Training Details . . 20   
B.1 Fine-Tuning Hyperparameters. 20   
B.2 Open-Source Models. 20   
C More Experiments on τ-Emotion-Bench .   
C.1 Error Analysis . 21   
C.2 Effect of Tracer on Agent Errors . 23   
C.3 Task Coverage 24   
C.4 Detailed Results . 24   
D Environment Details . 25   
D.1 Tool Depedency Graphs. 25   
D.2 Domain Policies . 29   
D.3 Prompt Templates. 48   
D.4 Core Entities 49   
E Task Trajectories .   
E.1 Human Trials. 50

## A DATA STATISTICS

We construct ten emotion-conditioned tool-use benchmarks by reusing the customer-service environments users, database schemas, and tool APIs released with τ-Bench, τ-Trait, and ToolOrchestra (Yao et al., 2024; He et al., 2025a; Su et al., 2025), without modification to the underlying environment logic. For each domain, we generate candidate tool dependency graphs using GPT-4.1, GPT-5.1, GPT-5.2 (Singh et al., 2025; Achiam et al., 2023), GLM-5.2, GLM-5.3, GLM-5.3-flash (GLM-5-Team et al., 2026), Qwen3.8-27B (Qwen Team, 2026), and Qwen3-235B-A22B-Instruct (Yang et al., 2025), and select the highest-quality graph per domain through manual review, before sampling tool traces over it. For each domain’s selected tool graph, we sample tool traces by composing 1–3 independent random walks (intents) of length 2–6 tool calls each into a single trace, retaining unique traces after deduplication. Per-domain unique trace counts range from 2,603 (Retail) to 8,175 (Airline), with Telecom, Telehealth, School, Basketball, Restaurant, Travel, Movie, and Bank yielding between 3,344 and 4,641 unique traces each, totaling 42,138 unique traces across the ten domains. We used a different set of traces for generating evaluation and training tasks.

## A.1 EMOTION MODELING DETAILS

Our emotion taxonomy follows the Atlas of Emotions (Ekman, 2016) and contains 22 fine-grained emotions organized into five families. Anger contains annoyance, frustration, argumentativeness, bitterness, and exasperation. Fear contains anxiety, nervousness, dread, and desperation. Disgust contains dislike, aversion, distaste, and repugnance. Sadness contains disappointment, discouragement, helplessness, hopelessness, and sorrow. Enjoyment contains relief, pride, peace, and excitement.

Each fine-grained emotion is combined with three independent behavioral dimensions. Politeness takes values from {warm, neutral, cold}, urgency from {low, moderate, high}, and trust from {trusting, cautious, skeptical}. Combining the 22 fine-grained emotions with these dimensions yields 594 emotion conditions.

For each condition, we generate 10 natural-language emotion instructions with GPT-4.1, producing 5,940 anchor instructions that describe how the corresponding user should communicate. We embed these anchors with text-embedding-3-small and use them to train a multi-head classifier that predicts the fine-grained emotion together with politeness, urgency, and trust. Given the task-specific feeling generated during task construction, we project it into the classifier’s learned embedding space. We use similarity-weighted votes from the nearest anchor instructions in this space to select the final emotion condition. One of the 10 natural-language anchor instructions for that condition, chosen at random, is then provided to the user simulator rather than the discrete emotion and behavioral labels.

## A.2 EVALUATION DATA

We initially generate 2,266 candidate evaluation tasks from sampled tool traces using the methodology described in Section 3.2 and illustrated in Figure 2. We use Qwen3-235B-A22B-Instruct (Yang et al., 2025) for all generation agents in the Tracer pipeline. Tasks that pass the verification procedure described in Section 3.4 are then rolled out using Qwen3-235B-A22B-Instruct as both the tool-calling agent and the user simulator. This additional rollout stage allows us to observe how the generated tasks behave during interaction and identify cases requiring further modification. After this stage, 884 tasks are retained for manual review.

During manual review, we make three main types of modifications.

• Ground-truth edits. We revise 101 tasks by correcting reference parameters, adding missing ground-truth actions, or removing unnecessary actions to ensure that the intended solution is executable and consistent with the corresponding domain policy.

• ID-swap edits. We identify 23 tasks containing internal identifiers such as payment\_id that would not normally be known to the user. We replace these identifiers with useraccessible information, such as the last four digits of a payment card, while preserving the intended task.

• Larger rewrites. We revise 23 tasks to introduce policy-relevant challenges that make the interaction more difficult while preserving task solvability. For example, some tasks involve a user requesting an action on behalf of another person after authentication, requiring the agent to follow the domain policy rather than simply execute the user’s request.

After these revisions, the final benchmark contains 580 evaluation tasks. Airline, Retail, Telecom, and Telehealth each contain 70 tasks, while School, Basketball, Restaurant, Travel, Movie, and Bank each contain 50 tasks. Each task is provided with its emotion instruction and, for controlled comparisons, a corresponding version without emotion conditioning.

## A.3 TRAINING DATA

Training tasks are generated with the same pipeline as the evaluation tasks. In Airline, Retail, Telecom, and Telehealth, training and evaluation tasks are grounded in the same environment database. In School, Basketball, Restaurant, Travel, Movie, and Bank, the training and evaluation tasks each have their own database. In total, we generate 7,767 training tasks (Table 6). We roll out each training task once with Qwen3.8-27B in thinking mode as the agent and Qwen3-235B-A22B-Instruct as the emotion-conditioned user simulator, with a limit of 50 interaction steps. The agent’s reasoning is kept on every assistant turn. Rollouts that failed because of endpoint errors are discarded, leaving 7,048 trajectories.

Figure 9 characterizes the unsuccessful trajectories across fine-grained emotion conditions. Among reward 0 trajectories, Excitement and Peace are the most frequent conditions at 30.1% and 18.1%, respectively. Reward −1 trajectories, which correspond to interactions terminated by an unsatisfied user, also contain a large share of Excitement at 29.3%, but show substantially higher proportions of Frustration and Exasperation at 13.7% and 6.8%, compared with 9.1% and 3.2% among reward 0 trajectories. This difference indicates that interaction-driven termination has a distinct emotional distribution from other unsuccessful rollouts.

We keep only trajectories with reward 1, i.e. those in which the final database state matches the reference and the user did not end the conversation with ###UNSATISFIED###. This yields 2,723 successful trajectories (38.6% of all rollouts). Three domains, Retail, Telehealth, and Basketball, produce more than 300 successful trajectories each. To keep the training set balanced across domains, we randomly sample 250 trajectories from each of these three domains and keep all successful trajectories from the other seven, which gives the 1,958 trajectories used for Tracer-High. Tracer-Low is trained on all 1,498 successful trajectories from Airline, Retail, Telecom, and Telehealth.

Table 6: Domain-level statistics for generated training tasks, rollout outcomes, success rates, and trajectories selected for Tracer-High and Tracer-Low.
<table><tr><td>Domain</td><td>Tasks</td><td>Rollouts</td><td>R=1</td><td>R=0</td><td>R=-1</td><td>Success (%)</td><td>High</td><td>Low</td></tr><tr><td>Airline</td><td>922</td><td>922</td><td>143</td><td>680</td><td>99</td><td>15.5</td><td>143</td><td>143</td></tr><tr><td>Retail</td><td>898</td><td>898</td><td>585</td><td>288</td><td>25</td><td>65.1</td><td>250</td><td>585</td></tr><tr><td>Telecom</td><td>1,519</td><td>1,519</td><td>186</td><td>1,118</td><td>215</td><td>12.2</td><td>186</td><td>186</td></tr><tr><td>Telehealth</td><td>1,314</td><td>1,314</td><td>584</td><td>718</td><td>12</td><td>44.4</td><td>250</td><td>584</td></tr><tr><td>School</td><td>333</td><td>325</td><td>133</td><td>97</td><td>95</td><td>40.9</td><td>133</td><td>一</td></tr><tr><td>Basketball</td><td>876</td><td>457</td><td>346</td><td>72</td><td>39</td><td>75.7</td><td>250</td><td>一</td></tr><tr><td>Restaurant</td><td>243</td><td>243</td><td>163</td><td>79</td><td>1</td><td>67.1</td><td>163</td><td>一</td></tr><tr><td>Travel</td><td>458</td><td>455</td><td>124</td><td>300</td><td>31</td><td>27.3</td><td>124</td><td>一</td></tr><tr><td>Movie</td><td>549</td><td>458</td><td>231</td><td>223</td><td>4</td><td>50.4</td><td>231</td><td>一</td></tr><tr><td>Bank</td><td>655</td><td>457</td><td>228</td><td>224</td><td>5</td><td>49.9</td><td>228</td><td>1</td></tr><tr><td>Total</td><td>7,767</td><td>7,048</td><td>2,723</td><td>3,799</td><td>526</td><td>38.6</td><td>1,958</td><td>1,498</td></tr></table>

## A.4 SUCCESSFUL TRAINING TRAJECTORIES

Table 7 characterizes the 2,723 successful trajectories before domain-level subsampling. Across all domains, a successful trajectory contains an average of 7.2 user turns, 15.7 assistant turns, and 9.5 tool calls, compared with 5.2 actions in the corresponding reference trajectory. Thus, successful rollouts generally contain additional environment interaction beyond the minimum reference sequence. The amount of tool use also varies substantially across domains. Basketball and Movie produce the longest successful trajectories, with 14.7 and 12.0 tool calls on average, respectively, while Airline has the lowest average at 6.2. Agent reasoning averages 538 characters per assistant turn, with the longest reasoning observed in Airline and Travel at 889 and 779 characters, respectively. Notably, Airline and Travel are also among the three domains with the lowest rollout success rates in Table 6, suggesting that successful completion in these domains is accompanied by greater reasoning effort.

![](images/646008446657efedd55a78b0f8270da6a2eb00ed2c547b18862d42e22c35dfdc.jpg)

Figure 9: Distribution of unsuccessful training trajectories across emotion families.  
![](images/fd18923e0be2eb6117161bbddd6141d31e33b19d6e528a486b92cd5ac9c0bf60.jpg)  
Figure 10: Distribution of training trajectories across emotion families.

Figure 10 shows the fine-grained emotion composition of the final Tracer-Low and Tracer-High training sets after trajectory selection and domain balancing. Both sets contain substantial coverage across the emotion conditions represented during data collection, but their distributions differ because Tracer-Low contains trajectories from four domains whereas Tracer-High covers all ten domains after subsampling. Excitement and Peace are the two most frequent conditions in both sets, accounting for 28.9% and 28.0% of Tracer-Low and 34.7% and 24.6% of Tracer-High, respectively. Tracer-High also contains a larger proportion of Pride at 11.3%, compared with 6.5% in Tracer-Low, while Anxiety decreases from 15.1% to 8.8%. Despite these differences, neither training set is concentrated in a single fine-grained emotion, providing Tracer with successful interaction trajectories spanning diverse emotional conditions.

Table 7: Statistics of successful (R=1) training trajectories.
<table><tr><td>Domain</td><td>R=1</td><td>Avg. User Turns</td><td>Avg. Assistant Turns</td><td>Avg. Tool Calls</td><td>Avg. Gold Actions</td><td>Avg. Reasoning Tokens</td></tr><tr><td>Airline</td><td>143</td><td>5.5</td><td>10.7</td><td>6.2</td><td>4.0</td><td>243</td></tr><tr><td>Retail</td><td>585</td><td>7.7</td><td>15.0</td><td>8.3</td><td>4.8</td><td>105</td></tr><tr><td>Telecom</td><td>186</td><td>7.0</td><td>12.5</td><td>6.5</td><td>4.4</td><td>166</td></tr><tr><td>Telehealth</td><td>584</td><td>6.7</td><td>15.0</td><td>9.3</td><td>5.5</td><td>121</td></tr><tr><td>School</td><td>133</td><td>6.7</td><td>14.5</td><td>8.8</td><td>4.5</td><td>180</td></tr><tr><td>Basketball</td><td>346</td><td>7.8</td><td>21.5</td><td>14.7</td><td>6.2</td><td>112</td></tr><tr><td>Restaurant</td><td>163</td><td>7.1</td><td>13.3</td><td>7.2</td><td>4.3</td><td>132</td></tr><tr><td>Travel</td><td>124</td><td>7.4</td><td>14.5</td><td>8.0</td><td>6.0</td><td>219</td></tr><tr><td>Movie</td><td>231</td><td>6.8</td><td>17.8</td><td>12.0</td><td>6.3</td><td>148</td></tr><tr><td>Bank</td><td>228</td><td>7.8</td><td>16.6</td><td>9.8</td><td>5.2</td><td>127</td></tr><tr><td>All</td><td>2,723</td><td>7.2</td><td>15.7</td><td>9.5</td><td>5.2</td><td>137</td></tr></table>

## B TRAINING DETAILS

## B.1 FINE-TUNING HYPERPARAMETERS

We fine-tune the Qwen-series models using the TRL library (von Werra et al., 2020) on four NVIDIA H100 GPUs, using our generated Tracer data together with the publicly available AreaL data. For the Qwen3 models, the maximum sequence length is set to 32,768. Detailed hyperparameters are provided in Table 8.

Table 8: Fine-tuning hyperparameters used for Qwen3-1.7B, Qwen3-4B and Qwen3-8B.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Learning Rate</td><td> $5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Scheduler</td><td>Cosine</td></tr><tr><td>Warmup Ratio</td><td>0.03</td></tr><tr><td>Epochs</td><td>3</td></tr><tr><td>Optimizer</td><td>Paged AdamW 8-bit</td></tr><tr><td>Weight Decay</td><td>0.0</td></tr><tr><td>Max Gradient Norm</td><td>1.0</td></tr><tr><td>Effective Batch Size</td><td>32</td></tr><tr><td>Precision Gradient Checkpointing</td><td>BF16 + TF32</td></tr><tr><td></td><td>Enabled</td></tr><tr><td>Attention</td><td>FlashAttention-3</td></tr><tr><td>Liger Fused CE</td><td>Enabled</td></tr><tr><td>Packing / Padding-Free</td><td>Disabled</td></tr></table>

## B.2 OPEN-SOURCE MODELS

Table 9 lists the Hugging Face checkpoints of the open-weight models we evaluate, apart from the AReaL baselines, which we fine-tune ourselves (Appendix B.1). All models, including ours, are evaluated with τ-Emotion-Bench’s implementation of the τ-Bench agent harness: the agent calls tools through OpenAI-style function calling and converses with the Qwen3-235B-A22B-Instruct-2507 user simulator described in Section 4.1. Each model receives the system prompt format it was trained on. By default, as in τ-Bench, the system prompt is the domain policy alone; our Tracer models are trained on this format. AReaL and Simia-Tau are fine-tuned on $\scriptstyle { \tau ^ { 2 } } ,$ -Bench’s agent prompt, which wraps the domain policy in $\tau ^ { 2 }$ -Bench’s agent instructions (<instructions> ... </instructions> $< \mathtt { p o l i c y } > \ldots < / \mathtt { p o l i c y } { > } \big )$ , so we evaluate them with that prompt in every domain.

Every model is queried through an OpenAI-compatible API served by vLLM (Kwon et al., 2023). We evaluate frontier open-weight models on Gaudi GPU resources provided by Arizona State University’s supercomputing clusters (Jennewein et al., 2026) (ASU in Table 9); the Qwen3 base models up to 32B and the agent-training baselines are served on our own GPUs. Thinking is switched on or off through each model’s chat template. Qwen3-32B is evaluated in both modes from the same checkpoint, and Simia-Tau, which is trained without reasoning traces, runs with thinking off.

Decoding settings. Table 10 lists, for each model, the decoding configuration used in the largest number of the ten domains. Apart from the few models decoded greedily (temperature 0) in most domains, these are the sampling settings recommended on each model’s Hugging Face card, and most models share them: the Qwen3-based models use Qwen’s recommended thinking-mode settings (temperature 0.6, top-p 0.95, top-k 20, min-p 0) or non-thinking settings (temperature 0.7, top-p 0.8, top-k 20, min-p 0), and the other frontier models use temperature 1.0 with top-p 0.95. For every model, the domains not covered by the listed configuration use the other of these two options: the model-card settings, or greedy decoding.

Table 9: Open-weight models we evaluate. Thinking: whether the model’s reasoning mode is on. Serving: ASU’s hosted inference cluster or our own vLLM server (Local).
<table><tr><td>Model</td><td>Thinking</td><td>Serving</td><td>Hugging Face</td></tr><tr><td>Evaluated models</td><td></td><td></td><td></td></tr><tr><td>Qwen3-1.7B</td><td>On</td><td>Local</td><td>Qwen/Qwen3-1.7B</td></tr><tr><td>Qwen3-4B</td><td>On</td><td>Local</td><td>Qwen/Qwen3-4B</td></tr><tr><td>Qwen3-8B</td><td>On</td><td>Local</td><td>Qwen/Qwen3-8B</td></tr><tr><td>Qwen3-32B</td><td>On / Off</td><td>Local</td><td>Qwen/Qwen3-32B</td></tr><tr><td>Qwen3-235B-A22B-Instruct-2507</td><td>Off</td><td>ASU</td><td>Qwen/Qwen3-235B-A22B-Instruct-2507</td></tr><tr><td>Qwen3-235B-A22B-Thinking-2507</td><td>On</td><td>ASU</td><td>Qwen/Qwen3-235B-A22B-Thinking-2507</td></tr><tr><td>Qwen3.8-27B</td><td>On</td><td>ASU</td><td>Qwen/Qwen3.8-27B</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>Off</td><td>ASU</td><td>deepseek-ai/DeepSeek-V4.1-Flash</td></tr><tr><td>Gemma-4-31B-it</td><td>On</td><td>ASU</td><td>google/gemma-4-31B-it</td></tr><tr><td>GLM-5.3</td><td>On</td><td>ASU</td><td>zai-org/GLM-5.3</td></tr><tr><td>Kimi-K2.7-Code</td><td>On</td><td>ASU</td><td>moonshotai/Kimi-K2.7-Code</td></tr><tr><td>MiniMax-M2.7</td><td>On</td><td>ASU</td><td>MiniMaxAI/MiniMax-M2.7</td></tr><tr><td>MiniMax-M3</td><td>On</td><td>ASU</td><td>MiniMaxAI/MiniMax-M3</td></tr><tr><td>Muse-Glimmer-30B</td><td>On</td><td>ASU</td><td>meta-models/Muse-Glimmer-30B</td></tr><tr><td>Agent-training baselines</td><td></td><td></td><td></td></tr><tr><td>EnvScaler-1.7B</td><td>On</td><td>Local</td><td>XXHStudyHard/EnvScaler-Qwen3-1.7B</td></tr><tr><td>EnvScaler-4B</td><td>On</td><td>Local</td><td>XXHStudyHard/EnvScaler-Qwen3-4B</td></tr><tr><td>EnvScaler-8B</td><td>On</td><td>Local</td><td>XXHStudyHard/EnvScaler-Qwen3-8B</td></tr><tr><td>Arctic-AWM-4B</td><td>On</td><td>Local</td><td>Snowflake/Arctic-AWM-4B</td></tr><tr><td>Arctic-AWM-8B</td><td>On</td><td>Local</td><td>Snowflake/Arctic-AWM-8B</td></tr><tr><td>SPADE-4B</td><td>Off</td><td>Local</td><td>spade-r1/SPADE-Qwen3-4B-Games</td></tr><tr><td>SPADE-8B</td><td>On</td><td>Local</td><td>spade-r1/SPADE-Qwen3-8B-Games</td></tr><tr><td>Simia-Tau-SFT</td><td>Off</td><td>Local</td><td>Simia-Agent/Simia-Tau-SFT-Qwen3-8B</td></tr><tr><td>Simia-Tau-SFT-RL</td><td>Off</td><td>Local</td><td>Simia-Agent/Simia-Tau-SFT-RL-Qwen3-8B</td></tr></table>

## C MORE EXPERIMENTS ON τ-EMOTION-BENCH

## C.1 ERROR ANALYSIS

We adapt the conversation-review judge of $\tau ^ { 2 } .$ -Bench (Barres et al., 2025) and FAMA (Saeidi et al., 2026b) to analyze failed trajectories. For each failed trajectory (reward ̸= 1) the judge receives the domain policy, the user simulator’s guidelines, the user instruction including its emotion instruction, the reference action sequence, and the full conversation with all tool calls and results. It returns every error it finds, each with its source (agent or user), a severity, one or more fault tags, the turn where it occurs, and a short justification. Agent errors are critical if they caused the failure or violated an important policy, and minor otherwise. We extend the judge with an emotional-outcome label that records whether the user abandoned the conversation and, if so, the dominant reason: dismissive tone, ignored emotional cues, procedural friction, or emotional neglect. We use Qwen3-235B-A22B-Instruct at temperature 0 as the judge and group its fault tags into five categories and fifteen fault types (Table 11). We apply it to eleven models: the Qwen3-1.7B, 4B, and 8B baselines, our Tracer-Low and Tracer-High models at the same scales, Qwen3-235B-A22B-Thinking, and DeepSeek-V4.1-Flash. Figure 11 aggregates the eight models other than Tracer-Low, yielding 6,961 trajectories with at least one identified fault, and decomposes these failures into five categories and fifteen fault types across domains.

Overall, failures are concentrated in task execution rather than affective interaction. Missing or extra actions account for 25–36% of faults across domains, while incomplete resolution accounts for 20–29% and wrong tool calls for 14–27%. Misread intent is the most frequent individual fault overall at 15.9%, followed by incorrect action order at 13.7%, wrong tool arguments at 12.9%, and missed actions at 12.2%. In contrast, policy and interaction faults contribute only 4–10% in each domain, while emotional neglect remains below 1.7%. Thus, under emotion-conditioned interaction, agents fail primarily by misinterpreting the request or executing the required workflow incorrectly rather than by failing to acknowledge the user’s emotional state.

Table 10: Decoding configuration used in the largest number of domains for each model. A dash means the parameter is not set (the server default).
<table><tr><td>Model</td><td>Thinking</td><td>Temp.</td><td>Top-p</td><td>Top-k</td><td>Min-p</td><td>Other</td></tr><tr><td>Evaluated models</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-1.7B</td><td>On</td><td>0.6</td><td>0.95</td><td>20</td><td>0</td><td></td></tr><tr><td>Qwen3-4B</td><td>On</td><td>0.0</td><td>一</td><td>一</td><td></td><td></td></tr><tr><td>Qwen3-8B</td><td>On</td><td>0.0</td><td></td><td>一</td><td></td><td></td></tr><tr><td>Qwen3-32B (thinking)</td><td>On</td><td>0.6</td><td>0.95</td><td>20</td><td>0</td><td></td></tr><tr><td>Qwen3-32B (non-thinking)</td><td>Off</td><td>0.7</td><td>0.8</td><td>20</td><td>0</td><td></td></tr><tr><td>Qwen3-235B-A22B-Instruct-2507</td><td>Off</td><td>0.7</td><td>0.8</td><td>20</td><td>0</td><td></td></tr><tr><td>Qwen3-235B-A22B-Thinking-2507</td><td>On</td><td>0.6</td><td>0.95</td><td>20</td><td>0</td><td></td></tr><tr><td>Qwen3.8-27B</td><td>On</td><td>0.0</td><td></td><td>一</td><td></td><td></td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>Off</td><td>1.0</td><td>0.95</td><td>一</td><td></td><td></td></tr><tr><td>Gemma-4-31B-it</td><td>On</td><td>1.0</td><td>0.95</td><td>64</td><td></td><td></td></tr><tr><td>GLM-5.3</td><td>On</td><td>1.0</td><td>0.95</td><td>一</td><td></td><td>reasoning effort max</td></tr><tr><td>Kimi-K2.7-Code</td><td>On</td><td>1.0</td><td>0.95</td><td>一</td><td></td><td></td></tr><tr><td>MiniMax-M2.7</td><td>On</td><td>1.0</td><td>0.95</td><td>40</td><td></td><td></td></tr><tr><td>MiniMax-M3</td><td>On</td><td>1.0</td><td>0.95</td><td>一</td><td></td><td></td></tr><tr><td>Muse-Glimmer-30B</td><td>On</td><td>1.0</td><td>0.95</td><td>64</td><td></td><td>reasoning strength high</td></tr><tr><td>Agent-training baselines</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AReaL-1.7B / 4B / 8B</td><td>On</td><td>0.6</td><td>0.95</td><td>20</td><td>0</td><td></td></tr><tr><td>EnvScaler-1.7B</td><td>On</td><td>0.0</td><td></td><td>一</td><td></td><td></td></tr><tr><td>EnvScaler-4B / 8B</td><td>On</td><td>0.6</td><td>0.95</td><td>20</td><td>0</td><td></td></tr><tr><td>Arctic-AWM-4B / 8B</td><td>On</td><td>0.6</td><td>0.95</td><td>20</td><td>0</td><td></td></tr><tr><td>SPADE-4B</td><td>Off</td><td>0.7</td><td>0.8</td><td>20</td><td>0</td><td></td></tr><tr><td>SPADE-8B</td><td>On</td><td>0.6</td><td>0.95</td><td>20</td><td>0</td><td></td></tr><tr><td>Simia-Tau-SFT / SFT-RL</td><td>Off</td><td>0.0</td><td></td><td>一</td><td>一</td><td></td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Tracer-High-1.7B / 4B / 8B</td><td>On</td><td>0.6</td><td>0.95</td><td>20</td><td>0</td><td></td></tr><tr><td>Tracer-Low-1.7B / 4B / 8B</td><td>On</td><td>0.6</td><td>0.95</td><td>20</td><td>0</td><td></td></tr></table>

Table 11: Fault taxonomy used in the error analysis.
<table><tr><td>Category</td><td>Fault type</td><td>Description</td></tr><tr><td rowspan="4">Missing or Extra Action</td><td>Missed Action</td><td>A required action is not taken.</td></tr><tr><td>Wrong Order</td><td>Actions are taken out of the required order.</td></tr><tr><td>Extra Action</td><td>An action not needed for the task is taken.</td></tr><tr><td>Repeated Action</td><td>The same action is taken again without need.</td></tr><tr><td rowspan="2">Incomplete</td><td>Misread Intent</td><td>The user&#x27;s request or a tool result is misunderstood.</td></tr><tr><td>Early Stop</td><td>The conversation ends before the task is complete.</td></tr><tr><td>Wrong Tool Call</td><td>Wrong Arguments Schema Error</td><td>A valid tool is called with incorrect argument values. A tool call has an invalid name, missing arguments, or wrong types.</td></tr><tr><td rowspan="2">Integrity</td><td>Hallucination</td><td>Information not grounded in the policy, instruction, or tool results is stated.</td></tr><tr><td>Inconsistent Revealed Early</td><td>A statement or action contradicts an earlier one. Information is shared before the required verification.</td></tr><tr><td rowspan="2">Policy &amp; Interaction</td><td>Policy Breach Friction</td><td>An action or message violates the domain policy.</td></tr><tr><td>Emotional Neglect</td><td>Redundant questions, circular flow, or unexplained delays. The user&#x27;s need for reassurance or empathy is ignored.</td></tr></table>

The dominant failure mode nevertheless varies substantially across domains. Missing or extra actions are the largest category in eight of the ten domains, peaking in Telehealth, Travel, Restaurant, and Telecom at 35.1–36.0%; wrong action order alone reaches 18.5% in Travel and 17.6% in Restaurant, and missed actions reach 15.9% in Telecom and 15.7% in Telehealth. Integrity faults are most prominent in Telecom and Basketball, at 23.8% and 23.0% of faults, respectively: in Telecom largely because agents reveal information before completing required checks (11.9%), and in Basketball because of agent hallucination (10.7%). Wrong tool calls are more prominent in Bank, School, and Movie, where they account for 21.5–27.0% of faults and form the largest category in Bank. Bank has the highest rates of wrong arguments and schema errors, at 17.2% and 9.8%, respectively. By contrast, Airline is the only domain dominated by incomplete resolution (29.1%), with misread intent alone accounting for 18.9% of faults, closely followed by Restaurant (18.6%) and Retail (17.8%). These differences show that robustness under emotionally variable interaction depends on multiple capabilities, including intent tracking, argument construction, and reliable execution of multi-step workflows, supporting the need for diverse and domain-grounded training trajectories.

![](images/570d7ccc3cb7f904d0746add7c724b113dc43ccd28bbbfef09c973ff1a223654.jpg)  
Figure 11: Failure analysis across domains on τ-Emotion-Bench.

![](images/2348f18685d4dd524c379de265d6b62a0bb6c6d15cbffc074ee839a776968fee.jpg)  
Figure 12: Share of failed trajectories with a critical agent error in each fault category for Qwen3 base and Tracer models, with Qwen3-235B-A22B-Thinking and DeepSeek-V4.1-Flash as references.

## C.2 EFFECT OF TRACER ON AGENT ERRORS

Figure 12 compares the error profiles of the Qwen3 baselines and our Tracer models, with DeepSeek V4.1-Flash and Qwen3-235B-A22B-Thinking as frontier references. For each fault category, it reports the share of failed trajectories with at least one critical agent error.

Tracer-High reduces every fault category at every scale, by 10.4–26.2 percentage points (pp) at 4B and 3.2–21.6 pp at 8B over the base model. The largest reductions are in wrong tool calls and incomplete resolution: at 8B, the share of failed trajectories with a wrong tool call drops from 38.4% to 20.0% (18.4 pp), and with incomplete resolution from 52.6% to 31.0% (21.6 pp). These reductions come on top of fewer failures overall (1,052 vs. 1,284 failed trajectories at 8B). Tracer-Low lies between the base model and Tracer-High in nearly every case, indicating that broader training data yields further gains.

![](images/3f957787f11287171ab0aa61e30a885820c85a3e957d9b616bb19565e2f0c233.jpg)  
Figure 13: Task coverage across domains on τ-Emotion-Bench.

Tracer-High-8B approaches the much larger Qwen3-235B-A22B-Thinking on missing or extra actions (32.4% vs. 26.5%) and integrity errors (12.4% vs. 10.2%), while DeepSeek-V4.1-Flash makes the fewest errors in nearly every category. The smallest model benefits least on integrity errors (33.9% to 31.5% at 1.7B), where Tracer-Low even slightly exceeds the base model. At 4B and 8B, missing or extra actions and incomplete resolution still occur in about a third of Tracer-High’s failed trajectories, suggesting that executing multi-step workflows and correctly interpreting user requests remain the main challenges.

## C.3 TASK COVERAGE

Figure 13 reports, for each task, how many of the three trials are solved by at least one of the 18 evaluated models. Coverage is high across most domains, although the consistency of successful solutions varies substantially. Retail has 68 of 70 tasks solved in all three trials, followed by Airline and Telehealth with 63 of 70 each. Among the 50-task domains, Bank, School, Basketball, and Restaurant have 45–48 tasks solved in all three trials, while Travel and Movie have 36 and 34, respectively. Overall, only 15 tasks are never solved by any evaluated model across the three trials, including 1 Airline, 5 Telecom, 1 Telehealth, 2 Basketball, 1 Restaurant, 4 Travel, and 1 Movie task. To further verify that difficult tasks remain executable, Appendix E.1 reports human tool-calling trials under the same tools, policies, and emotion-conditioned user simulator. The successful human trials provide additional evidence that persistent model failures can arise from the difficulty of maintaining the correct workflow under emotionally variable interaction rather than invalid task construction, leaving substantial headroom for current agents.

## C.4 DETAILED RESULTS

Tables 12, 13, and 14 report per-domain Pass^1, Pass^2, and Pass^3 on τ -Emotion-Bench. Table 12 covers the evaluated thinking and non-thinking models together with Tracer-Low and Tracer-High, Table 13 covers the Qwen3 base models and existing agent-training methods at the same scales, and Table 14 repeats the evaluation of Table 12 with GLM-5.2 instead of Qwen3-235B-A22B-Instruct as the user simulator. All averages below are unweighted means over the ten domains.

Consistency across trials. Success drops sharply when a task must be solved in more than one trial. DeepSeek-V4.1-Flash falls from 55.9% average Pass^1 to 32.2% Pass^3, GLM-5.3 from 51.9% to 31.7%, and Kimi-K2.7-Code from 49.3% to 26.1%, and most models lose 14–24 points between the two metrics. Even the strongest agents therefore solve only about a third of the tasks in all three trials.

Variation across domains. Difficulty varies widely across domains, and the best model changes from one domain to the next. Travel and Movie are the hardest: no model exceeds 30.0% Pass^3 on either, and only DeepSeek-V4.1-Flash and GLM-5.3 exceed 45% Pass^1 on Movie. Telecom is difficult for frontier models in particular, none of which exceeds 38.1% Pass^1 or 17.1% Pass^3;

GLM-5.3, for example, reaches 62.0% Pass^3 on Basketball but only 2.9% on Telecom. Individual models also show sharp domain-specific strengths: Muse-Glimmer-30B is the best model on School (64.0% Pass^3) but solves almost no Retail tasks (1.4% Pass^1), and Kimi-K2.7-Code leads on Bank with 52.0% Pass^3.

Effect of Tracer training. Tracer-High improves over the corresponding Qwen3 base model at every scale, raising average Pass^1 from 6.0% to 25.7% at 1.7B, from 18.0% to 37.7% at 4B, and from 24.4% to 38.1% at 8B, and average Pass^3 from 1.1% to 10.4%, from 3.9% to 17.2%, and from 8.4% to 19.8%, respectively. Tracer-High-8B outperforms Qwen3-32B in both modes as well as Qwen3-235B-A22B-Thinking (36.0% Pass^1, 16.5% Pass^3). The Tracer models are strongest on Airline, Retail, Telecom, and Telehealth, the four domains shared by both training sets: Tracer-Low-8B achieves the best Telecom scores of all evaluated models (41.9%, 26.2%, and 20.0%), and Tracer-High-8B achieves the best Telehealth Pass^2 (39.5%) and ties DeepSeek-V4.1-Flash for the best Telehealth Pass^3 (31.4%). Comparing the two training settings isolates the effect of domain coverage. On these four domains, Tracer-Low-8B and Tracer-High-8B perform similarly (49.6% vs. 48.2% average Pass^1), whereas on the remaining six Tracer-High-8B reaches 31.3% compared with 20.7% for Tracer-Low-8B, with the largest differences on School (56.0% vs. 29.3%) and Bank (46.7% vs. 28.7%). Restaurant, Travel, and Movie remain difficult for all Tracer models, with Pass^3 of at most 12.0%.

Comparison with agent-training methods. Table 13 shows that existing agent-training methods often fail to improve over the base model under emotion-conditioned users. AReaL and SPADE score below the corresponding base model in average Pass^1 at every scale, with SPADE reaching at most 3.5%, while Arctic-AWM, Simia-Tau-SFT, and Simia-Tau-SFT-RL stay within 3 points of the base model. EnvScaler is the strongest baseline, but its gains are concentrated on Airline, Retail, Telecom, and Telehealth: at 8B, it reaches 39.8% average Pass^1 on these four domains but only 20.8% on the other six. At every scale, Tracer-Low (Table 12) achieves a higher average Pass^3 than every method in Table 13 (7.2%, 13.0%, and 15.2% at 1.7B, 4B, and 8B, compared with 3.1%, 9.6%, and 11.8% for EnvScaler). On the six domains excluded from Tracer-Low’s training, Tracer-Low matches or exceeds EnvScaler in average Pass^1 (8.5% vs. 6.5%, 18.6% vs. 16.9%, and 20.7% vs. 20.8%) and exceeds it in average Pass^3 (2.7% vs. 0.7%, 7.7% vs. 3.3%, and 8.3% vs. 7.0%).

Robustness to the user simulator. With GLM-5.2 as the user simulator (Table 14), every model scores higher than with Qwen3-235B-A22B-Instruct; for example, average Pass^1 rises from 55.9% to 77.5% for DeepSeek-V4.1-Flash, from 51.9% to 78.6% for GLM-5.3, and from 38.1% to 54.7% for Tracer-High-8B. The relative ordering is nevertheless largely preserved: GLM-5.3, DeepSeek-V4.1- Flash, Kimi-K2.7-Code, and Qwen3.8-27B are the four strongest models in average Pass^1 under both simulators. The gains from Tracer training also persist, with Tracer-High improving average Pass^1 over the base model by 25.2, 20.2, and 17.5 points and average Pass^3 by 15.4, 18.0, and 14.9 points at 1.7B, 4B, and 8B, respectively. As with the default simulator, Tracer-Low performs best on its four training domains, where Tracer-Low-8B reaches 76.1% average Pass^1 and the best Telecom Pass^1 of all models (75.2%), but drops to 27.1% on the other six. These results indicate that the user simulator affects absolute scores, but not the main conclusions drawn from τ-Emotion-Bench.

## D ENVIRONMENT DETAILS

## D.1 TOOL DEPEDENCY GRAPHS

Figure 14 show the tool dependency graph of every domain, i.e. the adjacency matrix A from which we sample tool traces. Each graph is drawn as two columns: read tools (get\_, list\_, search\_, check\_, find\_) on the left and write tools on the right, with the order within each column chosen so that tools sharing neighbours sit close together. The domains range from RESTAURANT, with 9 tools and 24 edges, to BANK, with 17 tools and 99 edges. The graphs are small but densely connected: between 24 and 99 of the possible transitions are valid, a density of 0.25-0.43. Between a third and three fifths (34-57%) of all connected tool pairs are valid in both directions, which is what makes the random walks produce varied traces rather than a single canonical ordering. Root-eligible tools, which a walk may start from, are not only lookup tools: more than half of them are write tools such as

Table 12: Performance comparison of frontier open-source models and Tracer models across ten domains on τ-Emotion-Bench, evaluated using Pass^1 (p^1), Pass^2 (p^2), and Pass^3 (p^3). Values are reported as percentages (%). Qwen3-235B-A22B-Instruct-2507 is used as the user simulator. The highest values are bolded, and the second highest values are underlined.
<table><tr><td>Model</td><td>Airline p^1 p^2 p^3</td><td>Retail p^1 p^2 p^3</td><td>Telecom p^1 p^2 p^3</td><td>Telehealth School p^2 p^3 p^1</td><td>Basketball p^3 p^1 p^2</td><td>Restaurant p^3 p^1 p^2</td><td>p^3 p^1</td><td>Travel p^2 p^3</td><td>Movie p^1 p^2</td><td>p^3</td><td>Bank p^2</td></tr><tr><td></td><td colspan="9">p^1 p^2 Non-thinking Models</td><td></td><td>p^1</td><td>p^3</td></tr><tr><td>Qwen3-32B</td><td colspan="9">3.3</td></tr><tr><td>Qwen3-235B-A22B</td><td>20.0 13.3 10.0 46.2 29.5 21.4 30.918.1 12.9 33.3 35.2 22.4 17.1 40.0 23.8 15.7 28.6 16.7 10.0 42.9 28.1</td><td></td><td>18.1 10.0 40.7 28.0 20.0 16.0 21.4 47.3 30.7 22.0 46.7</td><td>7.3 4.0 16.7 30.7 24.0 22.0</td><td>8.0 6.0 10.7 4.0</td><td>4.7 10.7</td><td>1.3 2.7</td><td>0.0 0.0 7.3</td><td>0.7 0.7</td><td>0.0 27.3 0.0 45.3</td><td>15.3 10.0 31.3 24.0</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td colspan="9">57.6 42.4 32.9 71.9 56.2 44.3 38.1 24.3 17.1 52.4 39.0 31.4 61.3 48.0 42.0 60.0 45.3 38.0 55.3 35.3</td></tr><tr><td></td><td colspan="9">26.0 49.3 36.0 30.0 51.3</td></tr><tr><td>Qwen3-32B</td><td></td><td>21.4 27.6 13.3 8.6 40.0 23.8 15.7</td><td>Thinking Models</td><td>7.3</td><td></td><td>12.0</td><td></td><td></td><td>2.0</td><td>26.7</td></tr><tr><td>Qwen3-1.7B</td><td>29.5 13.8 8.6 46.2 29.5 1.0 0.0 0.0 0.5 0.0</td><td>0.0 23.8 12.9 8.6 15.2 5.2 2.9</td><td>42.0 23.3 14.0 38.7 3.3 1.3 0.0 6.7</td><td>21.3 12.0 19.3 2.0 0.0 0.7</td><td>2.0 0.0</td><td>4.7 0.0</td><td>2.0 0.0</td><td>3.3 0.0</td><td>2.0 0.0</td><td>6.7 9.3 2.7</td></tr><tr><td>Qwen3-4B</td><td>18.6 7.6 2.9 20.0 5.7 1.4</td><td>34.3 17.1 8.6 36.7 21.4 14.3 16.0</td><td>5.3 0.0 17.3 9.3</td><td>0.0 8.0 12.7 3.3</td><td>0.0 0.0 4.7</td><td>0.7</td><td>0.0</td><td>0.0 2.0 0.0</td><td>0.0</td><td>0.0 17.3 6.7 4.0</td></tr><tr><td>Qwen3-8B</td><td>20.0 9.0 4.3 37.1 20.9 14.3 41.4</td><td>23.3 12.9 46.2 30.5 22.9 24.7</td><td>6.7 0.0 26.7 17.3</td><td>12.0 16.7 6.0</td><td>4.0 2.7</td><td>2.0</td><td>2.0</td><td>3.3 0.0</td><td>0.0 24.7</td><td>15.3 12.0</td></tr><tr><td>Qwen3-235B-A22B-T</td><td>43.3 30.5 22.9 54.3 38.1 31.4 20.5</td><td>11.0 7.1 43.3 28.1 20.0 49.3</td><td>30.0 20.0 52.7 36.7</td><td>26.0 30.7 14.0</td><td>8.0 16.7</td><td>8.0</td><td>4.0</td><td>4.7 0.0</td><td>0.0 44.7</td><td>32.0 26.0</td></tr><tr><td>Qwen3.8-27B</td><td>53.3 40.5 32.9 69.5 51.4 38.6 29.0</td><td>16.7 11.4 54.3 37.1 30.0 63.3</td><td>52.0 48.0 55.3 38.7</td><td>28.0 58.0 41.3</td><td>30.0 38.0</td><td>24.0</td><td>16.0</td><td>28.0 14.7</td><td>10.0</td><td>50.7 37.3 28.0</td></tr><tr><td>MiniMax-M3</td><td>17.1 10.5 7.1 66.7 50.9 42.9 28.1</td><td>16.2 11.4 39.1 21.4 14.3 28.0 20.7</td><td>16.0 61.3 45.3 34.0</td><td>38.7 20.7</td><td>12.0 24.0</td><td>16.0</td><td>12.0</td><td>25.3 9.3</td><td>4.0</td><td>42.0 28.7 22.0</td></tr><tr><td>Muse-Glimmer-30B</td><td>28.1 22.9 20.0 1.4 0.0 0.0 27.1 14.3</td><td>8.6 40.0 20.0 14.3 75.3 68.0</td><td>64.0 30.0 28.7 28.0</td><td>27.3 14.0</td><td>10.0 38.0</td><td>24.0</td><td>18.0</td><td>34.7 17.3</td><td>12.0</td><td>49.3 36.7 30.0</td></tr><tr><td>MiniMax-M2.7</td><td>28.1 18.6 55.2 35.7 27.1 29.5 13.8</td><td>7.1 38.6 22.9 12.9 72.0 56.7</td><td>46.0 46.7 30.7 22.0</td><td>30.0 12.7</td><td>4.0 16.0</td><td>4.0</td><td>0.0</td><td>10.7 4.7</td><td>2.0</td><td>43.3 31.3 28.0</td></tr><tr><td>Gemma-4-31B-it</td><td>53.3 38.6 30.0 64.8 48.6 37.1 35.2 20.5 12.9</td><td>46.7 31.4 22.9 66.7 60.0</td><td>56.0 42.0 31.3 26.0</td><td>40.0 24.7</td><td>18.0 27.3</td><td>11.3</td><td>6.0</td><td>22.0 14.7</td><td>12.0 50.0</td><td>35.3 26.0</td></tr><tr><td>GLM-5.3</td><td>60.5 46.7 38.6 70.0 54.3 45.7 22.4 9.0 2.9</td><td>43.8 26.7 20.0 065.3 54.7</td><td>48.0 76.0 67.3 62.0</td><td>54.0 37.3</td><td>28.0 32.0</td><td>20.0</td><td>14.0</td><td>47.3 35.3</td><td>30.0 47.3</td><td>32.7 28.0</td></tr><tr><td>Kimi-K2.7-Code</td><td>53.8 39.1 30.0 65.7 52.4 44.3 27.615.7</td><td>10.0 42.9 23.8 14.3 68.7 54.7</td><td>44.0 60.0 43.3</td><td>34.0 40.0 22.7</td><td>12.0 34.7</td><td>18.0</td><td>12.0</td><td>25.3 14.0</td><td>8.0</td><td>74.7 60.7 52.0</td></tr><tr><td>Tracer-Low-1.7B (Ours) Tracer-Low-4B (Ours)</td><td>29.5 15.7 10.0 40.9 23.3 18.6 32.4 16.2 43.3 30.5 24.3 53.3 32.4 21.4 40.5 24.3</td><td>10.0 45.7 27.6 17.1 9.3 8.7 15.7 50.8 32.9 22.9 21.3</td><td>8.0 14.0 3.3 9.3 6.0 36.0 24.7</td><td>0.0 6.0 2.0 18.0 22.0 11.3</td><td>2.0 0.0 8.0 4.7</td><td>0.0 0.7</td><td>0.0 0.0</td><td>4.7 1.3 2.0 0.0</td><td>0.0 0.0 25.3</td><td>16.7 8.0 6.0 17.3 14.0</td></tr><tr><td>Tracer-Low-8B (Ours)</td><td>46.7 34.8 30.0 54.8 35.7 22.9 41.9</td><td>26.2 20.0 55.2 38.1 28.6 29.3</td><td>14.7 8.0 30.7 19.3 1 1 1 一</td><td>14.0 19.3 9.3 1 1 1 1</td><td>6.0 11.3 一</td><td>6.7 1</td><td>4.0 –</td><td>4.7 0.7 –</td><td>0.0 一</td><td>28.7 22.0 18.0</td></tr><tr><td>Tracer-High-1.7B (Ours) Tracer-High-4B (Ours)</td><td>一 34.3 22.9 17.1 40.0 24.8 17.1</td><td>34.3 17.6 11.4 41.9 22.9 14.3</td><td>28.7 14.7 8.0 32.7</td><td>19.3 12.0 5.3</td><td>– 1 0.7 0.0 0.7</td><td>一 一 0.0</td><td>0.0</td><td>0.7 一 0.0</td><td>0.0</td><td>38.0 28.0 24.0</td></tr><tr><td>Tracer-High-8B (Ours)</td><td>42.9 30.0 22.9 56.7 40.5 30.0 37.6 17.6 8.6 51.9 32.9 22.9</td><td></td><td>51.3 39.3 34.0 42.7 31.4 56.0 42.7 38.0 40.0 23.3</td><td>28.0 20.0 24.7 16.0 24.7</td><td>11.3 6.0 12.0 14.0</td><td>13.3 2.0 5.3</td><td>0.0 2.0</td><td>9.3 6.7</td><td>4.0 2.0 2.7 2.0</td><td>46.7 34.0 26.0</td></tr><tr><td></td><td>44.8 33.3 27.1 56.7 40.0 31.4 37.6 22.4 15.7 53.8 39.5</td><td></td><td></td><td></td><td>16.7</td><td></td><td></td><td></td><td></td><td>46.7 31.3 22.0</td></tr></table>

Table 13: Performance comparison of existing agent-trained models across ten domains on τ-Emotion-Bench using Pass^1 (p^1), Pass^2 (p^2), and Pass^3 (p^3). Values are in %. The highest values are bolded, and the second highest values are underlined.
<table><tr><td rowspan="2">Method</td><td colspan="2">Airline</td><td colspan="2">Retail</td><td colspan="2">Telecom p^3</td><td colspan="2">Telehealth</td><td colspan="2">Basketball</td><td colspan="2">Restaurant Travel</td><td colspan="2">Movie</td></tr><tr><td>p^1 p^2</td><td colspan="2">p^3 p^1 p^2</td><td colspan="2">p^3 p^1 p^2</td><td colspan="2">p^1 p^2 p^3 p^1 p^2</td><td colspan="2">p^1 p^2 p^3</td><td colspan="2">1 p^2 p^3 p^1 p^2 p^3</td><td colspan="2">p^1 p^2 p^3 p^l p^2</td></tr><tr><td></td><td colspan="8"></td><td colspan="4"></td><td></td></tr><tr><td>Base</td><td>1.0 0.0</td><td>0.0 0.5</td><td>0.0 0.0</td><td>23.8 12.9</td><td>15.2 5.2</td><td>Qwen3-1.7B 3.3</td><td>0.0 6.7</td><td>2.0 0.0</td><td>0.7 0.0</td><td>0.0 0.0</td><td>0.0 0.0</td><td>0.0 0.0 9.3</td><td>2.7 0.0</td></tr><tr><td>AReaL</td><td>11.0 3.3</td><td>1.4 10.0</td><td>2.9 1.4</td><td>8.6 13.3 5.7 2.9</td><td>2.9 2.4 0.0</td><td>1.3 4.0 2.7</td><td>4.0</td><td>0.0 0.0</td><td>0.7 0.0 0.0</td><td>0.0 0.0 0.0</td><td>0.0 0.0</td><td>0.0 0.0 2.7</td><td>7.0 0.0</td></tr><tr><td>EnvScaler</td><td>10.0 4.8</td><td>1.4 7.1</td><td>1.4 0.0 37.6</td><td>22.4 14.3 35.7</td><td>0.0 20.9 11.4 8.0</td><td>2.0 1.3 0.0</td><td>5.3 0.7</td><td>0.0 4.7</td><td>0.7 0.0</td><td>1.3 0.0</td><td>0.0 4.7 0.7</td><td>0.0 14.7</td><td>4.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>Qwen3-4B</td><td></td><td></td><td></td><td></td><td></td><td></td><td>7.3</td></tr><tr><td>Base</td><td>18.6 7.6</td><td>2.9 20.0</td><td></td><td></td><td></td><td></td><td>0.0</td><td>9.3</td><td>12.7</td><td>4.7</td><td>0.0 2.0</td><td>0.0</td><td></td></tr><tr><td>AReaL</td><td>20.0 7.1</td><td>2.9 25.2</td><td>5.7 1.4 12.9 5.7</td><td>34.3 17.1 8.6 25.7 14.8 10.0</td><td>36.7 21.4 14.3</td><td>16.0 5.3 2.7</td><td>17.3 15.3</td><td>8.0 2.0</td><td>3.3 0.0 2.0 2.0</td><td>0.7 0.7 0.0</td><td>0.0 0.0 1.3 0.0</td><td>17.3 6.7 0.0 14.0 6.7</td><td>4.0 4.0</td></tr><tr><td>EnvScaler</td><td>29.0 17.6</td><td>12.9 42.9</td><td>24.8 17.1</td><td>40.9 28.1 22.9</td><td>6.7 2.9 1.4 47.1 32.9 22.9</td><td>5.3 32.0 12.7</td><td>2.0 6.0 30.0</td><td>4.7 13.3 6.0</td><td>8.7 10.7 1.3 0.0</td><td>4.7 0.0</td><td>0.0 4.0 1.3</td><td>0.0 20.0 11.3</td><td>8.0</td></tr><tr><td>Arctic-AWM</td><td>13.8 3.8</td><td>0.0 29.5</td><td>15.2 10.0</td><td>35.7 19.1 11.4</td><td>42.4 27.1 20.0</td><td>11.3 5.3</td><td>4.0 14.7</td><td>5.3 2.0</td><td>12.7 2.0</td><td>3.3 1.3</td><td>0.0 2.7 0.7</td><td>0.0 20.7 8.0</td><td>4.0</td></tr><tr><td>SPADE</td><td>8.1 2.9</td><td>1.4 3.3</td><td>2.9 2.9</td><td>11.4 4.3 1.4 4.3</td><td>1.4 0.0</td><td>1.3</td><td>2.0</td><td>0.7 0.0</td><td>0.0 0.0</td><td>1.3 0.0</td><td>0.0 0.7 0.0</td><td>0.0 0.7</td><td>0.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>0.0 Qwen3-8B</td><td>0.0</td><td></td><td></td><td>0.0</td><td></td><td></td><td>0.0</td></tr><tr><td>Base</td><td></td><td>37.1</td><td></td><td>12.9</td><td></td><td>6.7</td><td></td><td></td><td></td><td></td><td>2.0</td><td></td><td></td></tr><tr><td>AReaL</td><td>20.0 9.0 16.7 9.0</td><td>4.3 5.7 27.6</td><td>20.9 14.3 13.3</td><td>41.4 23.3 28.6 15.7</td><td>46.2 30.5</td><td>22.9 24.7</td><td>0.0 20.0</td><td>26.7 17.3 12.0 6.7 2.0</td><td>16.7 6.0 4.7 0.0</td><td>2.7 2.0 0.7 0.0</td><td>3.3 0.0 0.0 2.0 0.7</td><td>0.0 24.7 3.3</td><td>15.3 12.0</td></tr><tr><td>EnvScaler</td><td>30.5 22.9</td><td>20.0 43.3</td><td>21.9</td><td>8.6 12.9 39.5 24.3</td><td>10.0 15.2 5.2 45.7 32.9 25.7</td><td>5.3 38.0</td><td>2.0 14.0 32.0</td><td>18.7 12.0</td><td>17.3 6.7</td><td>0.0 4.0 10.7 5.3</td><td>2.0 5.3 0.7</td><td>0.0 12.0 0.0 21.3 14.0</td><td>2.0 10.0</td></tr><tr><td>Arctic-AWM</td><td>20.0 10.0</td><td>5.7 35.7</td><td>19.5</td><td>11.4 35.7 20.0</td><td>17.1 11.4 41.4 26.2</td><td>24.7</td><td>4.0 19.3</td><td>8.0 4.0</td><td>11.3 3.3</td><td>2.0 6.0 1.3</td><td>0.0 4.0 0.7</td><td>0.0 20.0 8.0</td><td>4.0</td></tr><tr><td>Simia-Tau-SFT</td><td>30.5 14.8</td><td>8.6 36.7</td><td>22.4</td><td>14.3 33.3 17.1</td><td>11.4 27.6 13.3</td><td>20.0 19.3</td><td>6.0 26.0</td><td>12.0 6.0</td><td>12.7 4.7</td><td>0.0 2.0 0.7</td><td>0.0 0.0 0.0</td><td>0.0 26.7 15.3</td><td>14.0</td></tr><tr><td>Simia-Tau-SFT-RL</td><td>30.0 18.1</td><td>12.9 39.1</td><td>22.4</td><td>14.3 35.2 18.1</td><td>11.4 29.5 20.0</td><td>8.6 17.1 20.7</td><td>2.0</td><td>20.7 8.0 2.0</td><td>11.3 4.0</td><td>2.0 2.7 0.7 0.0</td><td>0.0 3.3 1.3</td><td>0.0 25.3 13.3</td><td>10.0</td></tr><tr><td>SPADE</td><td>5.7 2.4</td><td>0.0 0.0</td><td>0.0</td><td>0.0 16.2 8.6</td><td>5.7 8.1 2.9</td><td>1.4 2.7</td><td>0.0</td><td>0.0 0.0</td><td>0.0 0.0</td><td>0.0 0.0</td><td>0.0 0.7 0.0</td><td>0.0 0.0 0.0</td><td>0.0</td></tr></table>

cancel\_reservation or freeze\_card, reflecting that a user may open a conversation with a request to act rather than a request for information.

![](images/f582b225d62fbac03814ffa0170494f975a20ea5bcbc411d47aef451964940f5.jpg)  
(a) Airline (9 tools, 31 edges)

![](images/d462595e58bd0acf95d6d46b85dbfe07be75237ce0b01113060ea517ac5b54db.jpg)  
(b) Restaurant (9 tools, 24 edges)

![](images/92604843f926a76696e8588bc76c5b382f96e0270cab7b9c042ea3ddc70e0db9.jpg)  
(c) School (10 tools, 36 edges)

![](images/b501e80871f9beca2fd67ab6ae05fdde6437f44b2dcc5cb17a1171078003a5c9.jpg)  
(d) Retail (10 tools, 35 edges)

![](images/095f1d6bd115fb5d3cb408e101f2b55eea592252d91eeebb2525431fcfda20fe.jpg)  
(e) Travel (12 tools, 50 edges)

![](images/cede25e35b2243451f50947e91b4cfa5543096760d4753b6108d2e2b24f2bcc9.jpg)  
(f) Telecom (13 tools, 53 edges)

![](images/ea5d25338185588e10e65268fcb8af59d9f5eec9f76341566ecf8b1ca3758fc2.jpg)  
(g) Movie (13 tools, 43 edges)

![](images/4038280897899f68c6badd7cc1b72b9338cde66999db0e63cf0cc3a01771daf6.jpg)  
(h) Bank (17 tools, 99 edges)

![](images/20804075ea1f3d4b1c0e0fd28bf4a2d640c8c0e44452a6a0e18e06ceb225abda.jpg)  
(i) Basketball (14 tools, 78 edges)

![](images/5251abedcccc369d1f8a95fc78de09914a6ab6c0e0492e02329d998a2de52845.jpg)  
(j) Telehealth (17 tools, 68 edges)  
Figure 14: Tool dependency graphs of all ten domains. Each panel puts read tools in the left column and write tools in the right. Gold nodes are root-eligible, i.e. valid starting points for a trace. A blue arrow marks a transition valid in both directions, drawn once per direction; an orange arrow marks one valid in a single direction only. Grey arcs are transitions within a column, read to read or write to write.

Table 14: Performance comparison of frontier open-source models and Tracer models across ten domains on τ -Emotion-Bench, evaluated using Pass^1 (p^1), Pass^2 (p^2), and Pass^3 (p^3). Values are reported as percentages (%). GLM-5.2 is used as the user simulator. The highest values are bolded, and the second highest values are underlined.
<table><tr><td rowspan=1 colspan=24>Airline    Retail    Telecom  Telehealth   School   Basketball Restaurant   Travel    Movie    BankModelp^1 p^2p^3p^1p^2p^3p^1p^2p^3p^1p^2p^3p^1p^2 p^3p^1p^2p^3p^1p^2p^3p^1p^2p^3p^1p^2p^3p^1p^2 p^3</td></tr><tr><td rowspan=1 colspan=24>Non-thinking Models</td></tr><tr><td rowspan=1 colspan=3>Qwen3-32B        28.1 21.99 20.0 63.354.3</td><td rowspan=1 colspan=11>50.0 66.7 58.6 54.3 63.3 54.3 48.6 51.3 38.734.0 30.7 16.7 10.0</td><td rowspan=1 colspan=2>24.0 13.38.0</td><td rowspan=1 colspan=5>8.01.30.06.01.3</td><td rowspan=1 colspan=2>0.037.3 22.7</td><td rowspan=1 colspan=1>14.0</td></tr><tr><td rowspan=1 colspan=3>DeepSeek-V4.1-Flash  86.2 81.478.696.293.8</td><td rowspan=1 colspan=11>92.965.755.750.0 75.2 64.3 60.0 68.0 59.354.0 71.3 60.754.0</td><td rowspan=1 colspan=2>80.071.366.0</td><td rowspan=1 colspan=5>94.7 90.0 86.0 60.746.0</td><td rowspan=1 colspan=2>36.076.771.3</td><td rowspan=1 colspan=1>68.0</td></tr><tr><td rowspan=1 colspan=14>Thinking Models</td><td rowspan=1 colspan=7></td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=1>Kimi-K2.7-Code</td><td rowspan=1 colspan=1>76.2 63.8</td><td rowspan=1 colspan=1>54.3 91.9 86.7</td><td rowspan=1 colspan=1>82.958.1</td><td rowspan=1 colspan=1>45.2</td><td rowspan=1 colspan=1>37.1</td><td rowspan=1 colspan=1>78.6</td><td rowspan=1 colspan=1>70.0</td><td rowspan=1 colspan=1>64.388.0</td><td rowspan=1 colspan=1>80.0</td><td rowspan=1 colspan=1>74.0</td><td rowspan=1 colspan=2>78.070.7</td><td rowspan=1 colspan=1>66.0</td><td rowspan=1 colspan=1>56.746.0</td><td rowspan=1 colspan=1>42.0</td><td rowspan=1 colspan=1>78.7</td><td rowspan=1 colspan=1>64.7</td><td rowspan=1 colspan=1>56.0</td><td rowspan=1 colspan=1>42.7</td><td rowspan=1 colspan=1>30.7</td><td rowspan=1 colspan=1>28.0</td><td rowspan=1 colspan=1>92.088.0</td><td rowspan=1 colspan=1>86.0</td></tr><tr><td rowspan=1 colspan=1>Tracer-Low-1.7B (Ours)</td><td rowspan=1 colspan=1>47.1 37.6</td><td rowspan=1 colspan=1>32.961.0 47.6</td><td rowspan=1 colspan=2>41.473.3 60.0</td><td rowspan=1 colspan=1>51.4</td><td rowspan=1 colspan=1>75.7</td><td rowspan=1 colspan=1>66.2</td><td rowspan=1 colspan=1>60.08.7</td><td rowspan=1 colspan=1>4.7</td><td rowspan=1 colspan=1>4.0</td><td rowspan=1 colspan=2>20.07.3</td><td rowspan=1 colspan=1>4.0</td><td rowspan=1 colspan=1>7.33.33</td><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>7.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>2.7</td><td rowspan=1 colspan=1>0.7</td><td rowspan=1 colspan=1>0.02</td><td rowspan=1 colspan=1>0.014.0</td><td rowspan=1 colspan=1>12.0</td></tr><tr><td rowspan=1 colspan=1>Tracer-Low-8B (Ours)</td><td rowspan=1 colspan=1>65.7 53.8</td><td rowspan=1 colspan=1>48.6 82.4 72.9</td><td rowspan=1 colspan=2>67.175.2 64.31</td><td rowspan=1 colspan=1>57.1</td><td rowspan=1 colspan=1>81.0 711</td><td rowspan=1 colspan=1>1.411</td><td rowspan=1 colspan=1>65.6 44.7</td><td rowspan=1 colspan=1>28.0</td><td rowspan=1 colspan=1>20.0</td><td rowspan=1 colspan=1>24.7</td><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>4.01</td><td rowspan=1 colspan=1>38.726.71</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>一1</td><td rowspan=1 colspan=1>2.7一1</td><td rowspan=1 colspan=2>0.08.0一一</td><td rowspan=1 colspan=1>4.7一1</td><td rowspan=1 colspan=1>4.032.011</td><td rowspan=1 colspan=1>26.7</td><td rowspan=1 colspan=1>24.01</td></tr><tr><td rowspan=1 colspan=1>Tracer-High-1.7B (Ours)</td><td rowspan=1 colspan=1>45.2 34.3</td><td rowspan=1 colspan=1>30.0 59.1 47.1</td><td rowspan=1 colspan=2>40.0 65.2 46.2</td><td rowspan=1 colspan=1>34.3</td><td rowspan=1 colspan=1>34.3 67.1</td><td rowspan=1 colspan=1>52.4 4</td><td rowspan=1 colspan=1>1.4 38.0</td><td rowspan=1 colspan=1>23.3</td><td rowspan=1 colspan=1>16.0</td><td rowspan=1 colspan=1>35.3</td><td rowspan=1 colspan=1>18.01</td><td rowspan=1 colspan=1>2.0</td><td rowspan=1 colspan=1>14.04.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>1.3</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>2.7</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>0.0</td><td rowspan=1 colspan=1>34.722.7</td><td rowspan=1 colspan=1>20.0</td></tr></table>

## D.2 DOMAIN POLICIES

We reproduce each domain’s policy verbatim; it is given to the agent as its system prompt.

## D.2.1 AIRLINE POLICY

```markdown
# Airline Agent Policy
The current time is 2024-05-15 15:00:00 EST.
As an airline agent, you can help users book, modify, or cancel flight reservations.
- Before taking any actions that update the booking database (booking, modifying flights,
editing baggage, upgrading cabin class, or updating passenger information), you must,→
list the action details and obtain explicit user confirmation (yes) to proceed.,→
- You should not provide any information, knowledge, or procedures not provided by the
,→ user or available tools, or give subjective recommendations or comments.
- You should only make one tool call at a time, and if you make a tool call, you should
not respond to the user simultaneously. If you respond to the user, you should not,→
make a tool call at the same time.,→
- You should deny user requests that are against this policy.
- You should transfer the user to a human agent if and only if the request cannot be
,→ handled within the scope of your actions.
## Domain Basic
- Each user has a profile containing user id, email, addresses, date of birth, payment
,→ methods, reservation numbers, and membership tier.
- Each reservation has an reservation id, user id, trip type (one way, round trip),
flights, passengers, payment methods, created time, baggages, and travel insurance,→
information.,→
- Each flight has a flight number, an origin, destination, scheduled departure and arrival
,→ time (local time), and for each date:
- If the status is "available", the flight has not taken off, available seats and prices
,→ are listed.
- If the status is "delayed" or "on time", the flight has not taken off, cannot be
,→ booked.
- If the status is "flying", the flight has taken off but not landed, cannot be booked.
## Book flight
- The agent must first obtain the user id, then ask for the trip type, origin,
,→ destination.
- Passengers: Each reservation can have at most five passengers. The agent needs to
collect the first name, last name, and date of birth for each passenger. All,→
passengers must fly the same flights in the same cabin.,→
- Payment: each reservation can use at most one travel certificate, at most one credit
,→ card, and at most three gift cards. The remaining amount of a travel certificate is
,→ not refundable. All payment methods must already be in user profile for safety
,→ reasons.
- Checked bag allowance: If the booking user is a regular member, 0 free checked bag for
each basic economy passenger, 1 free checked bag for each economy passenger, and 2,→
,→ free checked bags for each business passenger. If the booking user is a silver member,
,→ 1 free checked bag for each basic economy passenger, 2 free checked bag for each
,→ economy passenger, and 3 free checked bags for each business passenger. If the booking
,→ user is a gold member, 2 free checked bag for each basic economy passenger, 3 free
,→ checked bag for each economy passenger, and 3 free checked bags for each business
,→ passenger. Each extra baggage is 50 dollars.
- Travel insurance: the agent should ask if the user wants to buy the travel insurance,
which is 30 dollars per passenger and enables full refund if the user needs to cancel,→
the flight given health or weather reasons.,→
## Modify flight
- The agent must first obtain the user id and the reservation id.
```

```csv
- Change flights: Basic economy flights cannot be modified. Other reservations can be
modified without changing the origin, destination, and trip type. Some flight segments,→
can be kept, but their prices will not be updated based on the current price. The API,→
does not check these for the agent, so the agent must make sure the rules apply before,→
calling the API!,→
- Change cabin: all reservations, including basic economy, can change cabin without
changing the flights. Cabin changes require the user to pay for the difference between,→
their current cabin and the new cabin class. Cabin class must be the same across all,→
the flights in the same reservation; changing cabin for just one flight segment is not,→
possible.,→
- Change baggage and insurance: The user can add but not remove checked bags. The user
,→ cannot add insurance after initial booking.
- Change passengers: The user can modify passengers but cannot modify the number of
,→ passengers. This is something that even a human agent cannot assist with.
- Payment: If the flights are changed, the user needs to provide one gift card or credit
card for payment or refund method. The agent should ask for the payment or refund,→
method instead.,→
## Cancel flight
- The agent must first obtain the user id, the reservation id, and the reason for
,→ cancellation (change of plan, airline cancelled flight, or other reasons)
- All reservations can be cancelled within 24 hours of booking, or if the airline
cancelled the flight. Otherwise, basic economy or economy flights can be cancelled,→
only if travel insurance is bought and the condition is met, and business flights can,→
always be cancelled. The rules are strict regardless of the membership status. The API,→
does not check these for the agent, so the agent must make sure the rules apply before,→
calling the API!,→
- The agent can only cancel the whole trip that is not flown. If any of the segments are
,→ already used, the agent cannot help and transfer is needed.
- The refund will go to original payment methods in 5 to 7 business days.
## Refund
- If the user is silver/gold member or has travel insurance or flies business, and
complains about cancelled flights in a reservation, the agent can offer a certificate,→
as a gesture after confirming the facts, with the amount being $100 times the number,→
of passengers.,→
- If the user is silver/gold member or has travel insurance or flies business, and
complains about delayed flights in a reservation and wants to change or cancel the,→
reservation, the agent can offer a certificate as a gesture after confirming the facts,→
and changing or cancelling the reservation, with the amount being $50 times the number,→
of passengers.,→
- Do not proactively offer these unless the user complains about the situation and
explicitly asks for some compensation. Do not compensate if the user is regular member,→
and has no travel insurance and flies (basic) economy.,→
```

## D.2.2 RETAIL POLICY

```csv
# Retail agent policy
As a retail agent, you can help users cancel or modify pending orders, return or exchange
delivered orders, modify their default user address, or provide information about,→
their own profile, orders, and related products.,→
- At the beginning of the conversation, you have to authenticate the user identity by
locating their user id via email, or via name + zip code. This has to be done even,→
when the user already provides the user id.,→
- Once the user has been authenticated, you can provide the user with information about
,→ order, product, profile information, e.g. help the user look up order id.
- You can only help one user per conversation (but you can handle multiple requests from
,→ the same user), and must deny any requests for tasks related to any other user.
- Before taking consequential actions that update the database (cancel, modify, return,
exchange), you have to list the action detail and obtain explicit user confirmation,→
(yes) to proceed.,→
```

```csv
- You should not make up any information or knowledge or procedures not provided from the
,→ user or the tools, or give subjective recommendations or comments.
- You should at most make one tool call at a time, and if you take a tool call, you should
not respond to the user at the same time. If you respond to the user, you should not,→
make a tool call.,→
- You should transfer the user to a human agent if and only if the request cannot be
,→ handled within the scope of your actions.
## Domain basic
- All times in the database are EST and 24 hour based. For example "02:30:00" means 2:30
,→ AM EST.
- Each user has a profile of its email, default address, user id, and payment methods.
,→ Each payment method is either a gift card, a paypal account, or a credit card.
- Our retail store has 50 types of products. For each type of product, there are variant
items of different options. For example, for a 't shirt' product, there could be an,→
item with option 'color blue size M', and another item with option 'color red size L'.,→
- Each product has an unique product id, and each item has an unique item id. They have no
,→ relations and should not be confused.
Each order can be in status 'pending', 'processed', 'delivered', or 'cancelled'.
,→ Generally, you can only take action on pending or delivered orders.
- Exchange or modify order tools can only be called once. Be sure that all items to be
,→ changed are collected into a list before making the tool call!!!
## Cancel pending order
- An order can only be cancelled if its status starts with 'pending' (e.g. 'pending' or
,→ 'pending (item modified)'), and you should check its status before taking the action.
- The user needs to confirm the order id and the reason (either 'no longer needed' or
,→ 'ordered by mistake') for cancellation.
- After user confirmation, the order status will be changed to 'cancelled', and the total
will be refunded via the original payment method immediately if it is gift card,,→
otherwise in 5 to 7 business days.,→
## Modify pending order
- An order can only be modified if its status starts with 'pending' (e.g. 'pending' or
,→ 'pending (item modified)'), and you should check its status before taking the action.
- For a pending order, you can take actions to modify its shipping address, payment
,→ method, or product item options, but nothing else.
### Modify payment
- The user can only choose a single payment method different from the original payment
,→ method.
- If the user wants the modify the payment method to gift card, it must have enough
,→ balance to cover the total amount.
- After user confirmation, the order status will be kept 'pending'. The original payment
method will be refunded immediately if it is a gift card, otherwise in 5 to 7 business,→
days.,→
### Modify items
This action can only be called once per order, and will change the order status to
,→ 'pending (item modified)'. After that, the order's items and payment method can no
longer be changed, but its shipping address can still be modified and the order can,→
still be cancelled. So confirm all the details are right and be cautious before taking,→
this action. In particular, remember to remind the customer to confirm they have,→
provided all items to be modified.,→
- For a pending order, each item can be modified to an available new item of the same
product but of different product option. There cannot be any change of product types,,→
e.g. modify shirt to shoe.,→
- The user must provide a payment method to pay or receive refund of the price difference.
If the user provides a gift card, it must have enough balance to cover the price,→
difference.,→
## Return delivered order
```

```csv
- An order can onl be returned if its status starts with 'delivered' (e. . 'delivered' or
'delivered (exchange requested)') and it has no return request yet, and you should,→
check its status before taking the action. An item that is already part of an exchange,→
request cannot be returned. Items are counted per unit: if an order has two identical,→
items and one of them is already part of an exchange request, the other one can still,→
be returned.,→
- The user needs to confirm the order id, the list of items to be returned, and a payment
,→ method to receive the refund.
- The refund must either go to the original payment method, or an existing gift card.
- After user confirmation, ' (return requested)' will be added to the order status, and
,→ the user will receive an email regarding how to return items.
## Exchange delivered order
- An order can only be exchanged if its status starts with 'delivered' (e.g. 'delivered'
or 'delivered (return requested)') and it has no exchange request yet, and you should,→
check its status before taking the action. An item that is already part of a return,→
request cannot be exchanged. Items are counted per unit: if an order has two identical,→
items and one of them is already part of a return request, the other one can still be,→
exchanged. In particular, remember to remind the customer to confirm they have,→
provided all items to be exchanged.,→
- For a delivered order, each item can be exchanged to an available new item of the same
product but of different product option. There cannot be any change of product types,,→
e.g. modify shirt to shoe.,→
- The user must provide a payment method to pay or receive refund of the price difference.
If the user provides a gift card, it must have enough balance to cover the price,→
difference.,→
- After user confirmation, ' (exchange requested)' will be added to the order status, and
the user will receive an email regarding how to return items. There is no need to,→
place a new order.,→
```

## D.2.3 TELECOM POLICY

```csv
# Telecom Agent Policy
As a telecom agent, you can help users manage their customer details, devices, services,
,→ and billing.
- Before taking any actions that modify the database, you must list out the actions to the
,→ user and get their explicit confirmation to proceed.
- You should not provide any information, knowledge, or procedures not provided by the
,→ user or available tools, or give subjective recommendations or comments.
- You should only make one tool call at a time, and if you make a tool call, you should
not respond to the user simultaneously. If you respond to the user, you should not,→
make a tool call at the same time.,→
- You should deny user requests that are against this policy.
- You should transfer the user to a human agent if and only if the request cannot be
,→ handled within the scope of your actions.
- When the user asks you to verify something, make sure to read out the output returned
,→ with the relevant tool call you invoke.
- Do not suggest any upgrades, payments, or account modifications unless prompted
,→ explicitly to do so by the user.
## Domain Basic
- Each user has a profile containing customer id, name, demographics, address, account
,→ infromation, services, devices, and billing preferences.
- Each billing account has information including the customer id, account number, curent
balance, last payment, next billing date, monthly charges, total monthly amount,,→
payment history, and billing preferences (such as autopay and paperless).,→
- Each device has information including the category of the device (mobile_phone, phone,
networking, tv, security), the manufacturer (Apple, Samsung, TechCorp, LG, Polycom,,→
SecureTech, Google), and the model (Ex: iPhone 12, Galaxy S23).,→
```

- Each service has information including the service id, the name of the plan, the   
,→ category of the service (mobile, internet, internet, tv, security)   
- Each support ticket has information including the ticket ID, the customer ID, the status   
of the ticket (open or closed) and its priority level (low, medium, high). Low is,→   
non-urgent, medium is standard, high is important, and urgent is critical (within 24,→   
hours). Before creating a support ticket, determine as best as possible the urgency of,→   
the situation. Do not make more than one support ticket for the same issue.,→   
## Services   
- When adding a service, you have the option to add devices to be associated with the   
service. When you do this, confirm with the user what devices they would like to add.,→   
The device ids can be found in the customer's information details.,→   
- When removing a service, all devices associated with that service will be removed from   
,→ that service.   
- If the user seeks to switch out one service for another, you must remove the old   
service, then add the new service with the device ids that used to be associated with,→   
the old service.,→   
## Authentication   
- When authenticating a user with their email or phone number, make sure to read their   
,→ customer ID back to them.   
## Formatting   
- If output from a tool call is to be returned to the user, all content should be output   
to the user verbatim as it appears in the output from the tool call. The only changes,→   
that should be made should be purely in eliminating unnecessary formatting, for,→   
example from a json dump.,→   
- All monetary amounts should be returned to 2 decimal places, even if it is a whole   
,→ number. For example: \$20.00.

## D.2.4 TELEHEALTH POLICY

As a telehealth agent, you can help patients schedule, reschedule, or cancel appointments, provide information about their medical records, connect them with appropriate,→ healthcare providers, and assist with general patient portal inquiries.,→

\- At the beginning of the conversation, you must authenticate the patient identity by locating their patient ID via email, or via name + date of birth. This must be done,→ even when the patient already provides the patient ID.,→

\- Once the patient has been authenticated, you can provide the patient with information ,→ about appointments, providers, medical records, and their profile information.

\- You can only help one patient per conversation (but you can handle multiple requests from the same patient), and must deny any requests for tasks related to any other,→ patient, unless for aged parents or kids.,→

\- Before taking consequential actions that update the system (schedule, reschedule, cancel appointments), you must list the action details and obtain explicit patient,→ confirmation (yes) to proceed.,→

\- You should not make up any medical information, provide medical advice, or give subjective recommendations about treatment. Always refer patients to their healthcare,→ providers for medical questions.,→

\- You should at most make one tool call at a time, and if you take a tool call, you should not respond to the patient at the same time. If you respond to the patient, you should,→ not make a tool call.,→

\- You should transfer the patient to human support if and only if the request cannot be ,→ handled within the scope of your actions.

\- All times in the database are in 24-hour format. For example "14:30" means 2:30 PM.

\- Each patient has a profile with demographics (name, date of birth, contact info), ,→ address, insurance information, medical history, and emergency contact details.

- Healthcare providers have specialties, schedules, consultation fees, and availability.   
,→ Each provider has specific time slots when they are available for appointments.

\- Appointments can be in status 'scheduled', 'pending\_approval', 'completed', or 'cancelled'. Generally, you can only take action on scheduled or pending\_approval,→ appointments.,→

\- Each appointment has a unique meeting link for the telehealth consultation.

## ## Patient Authentication

\- Patients must be authenticated before any sensitive information is shared or actions are ,→ taken.

\- Authentication can be done via email address OR via full name + date of birth ,→ (YYYY-MM-DD format).

\- Both methods must match exactly with the information in the patient database.

## ## Scheduling Appointments

\- Patients can schedule appointments with available providers based on the provider's ,→ schedule.

\- Check provider availability before scheduling - providers have specific days and times ,→ when they are available.

\- Appointment types include: routine\_checkup, follow\_up, consultation, ,→ specialist\_consultation, sick\_visit.

\- Insurance copays are automatically calculated based on whether it's a primary care visit ,→ or specialist visit.

\- Each scheduled appointment receives a unique appointment ID and meeting link.

## ## Modifying Appointments

## ### Rescheduling Appointments

\- Appointments can only be rescheduled if their status is 'scheduled' or ,→ 'pending\_approval'.

\- The new date and time must be available in the provider's schedule.

\- Check for conflicts with other appointments before confirming the reschedule.

## ### Cancelling Appointments

\- Appointments can be cancelled if their status is 'scheduled' or 'pending\_approval'.

## - Cannot cancel completed appointments.

Cancelled appointment slots become available for other patients.

## ## Provider Information

\- Providers have different specialties: Primary Care, Cardiology, Dermatology, Psychiatry, ,→ etc.

\- Each provider has their own schedule with specific available time slots.

\- Consultation fees vary by provider and specialty.

\- Providers may speak different languages and have varying years of experience.

## ## Insurance and Billing

Primary care visits typically have lower copays than specialist visits.

\- Insurance authorization codes are automatically generated for scheduled appointments.

\- Copay amounts are determined by the patient's insurance plan and provider type.

\## Medical Records and Privacy

\- Medical records contain consultation notes, prescriptions, and treatment plans.

\- Only share medical information with the authenticated patient.

\- Do not provide medical advice or interpret medical results - refer patients to their ,→ healthcare providers.

```markdown
## Technical Support
- Meeting links are automatically generated for each appointment.
- If patients have technical issues with the telehealth platform, transfer them to human
,→ support.
- Provide meeting links and basic instructions for joining telehealth appointments.
```

## D.2.5 SCHOOL POLICY

School Agent Policy   
The current time is 2024-05-15 15:00:00 EST.   
As a school agent, you can help users register for courses, modify registrations, cancel   
,→ registrations, and handle payments and refunds.   
Before taking any actions that update the registration database (registering courses,   
adding/dropping courses, changing grading options, setting health insurance, or,→   
recording a payment/refund), you must list the action details and obtain explicit user,→   
confirmation (yes) to proceed.,→   
You should not provide any information, knowledge, or procedures not provided by the user   
,→ or available tools, or give subjective recommendations or comments.   
You should only make one tool call at a time, and if you make a tool call, you should not   
respond to the user simultaneously. If you respond to the user, you should not make a,→   
tool call at the same time.,→   
You should deny user requests that are against this policy.   
You should transfer the user to a human agent if and only if the request cannot be handled   
within the scope of your actions. If a transfer is needed, inform the user with the,→   
message: YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.,→   
Domain Basic   
- Student   
- Attributes include: student\_id, name, address, email, dob, saved payment methods (with   
payment\_method\_id and extra info like brand and last four), saved contacts, academic,→   
,→ program (school, degree, major, minor), optional memberships, and a list of   
registration IDs.,→   
Course   
- Attributes include: course\_id, department, course\_code (e.g., DEPT-NNNN), title,   
credits, term (e.g., 2025-SPRING), instructor, location (campus/building/room),,→   
scheduled first/last meeting, weekly meeting pattern (days, start\_time\_est,,→   
,→ end\_time\_est), per-date meeting status records (held/canceled/rescheduled),   
,→ capacity/waitlist capacity, and current enrollments.   
- Registration   
- Attributes include: registration\_id, student\_id, campus, program, study\_level   
(undergraduate, graduate, continuing\_ed), term, enrollment status,→   
(full-time/part-time), list of registered courses (with course\_id, section, credits,,→   
grading option, tuition per course), advisors, payment\_history (payment\_id and,→   
amount), created\_at, total\_credits, overload\_credits, financial aid breakdown, and,→   
health\_insurance (yes/no).,→   
Read capabilities   
- You can retrieve details for students, courses, and registrations.   
- You can list departments and list/search courses for a term using filters:   
- term (required), department (optional), campus (optional), day (optional), start\_after   
,→ (optional), end\_before (optional), open\_only (optional).   
- You can fetch per-date course meeting status (held, canceled, rescheduled) for a given   
,→ course and date.   
Register courses   
The agent must first obtain the student\_id from the user.   
Then collect:   
- Term (format YYYY-TERM, e.g., 2025-SPRING).   
- Study level: undergraduate, graduate, or continuing\_ed.   
- Desired courses (course\_ids). If the user needs help, use the course search filters to   
,→ propose options for the provided term.   
- Grading option for each course: letter (default), pass/fail, or audit. If not specified,   
,→ default is letter.

![](images/16be3fe17997c346f01c3a182a75431942f58e20c9e750f3dc0840af39866262.jpg)

Out-of-scope examples (transfer to a human agent if requested):   
- Changes to student profile data (e.g., editing name, program, or address) beyond what   
,→ tools support.   
- Advisor assignments or financial aid adjustments beyond what tools support.   
- Any requests requiring procedures or systems not represented by the available tools.

## D.2.6 BASKETBALL POLICY

```csv
Basketball agent policy
As a basketball agent, you can help users:
authenticate a player and provide information about their own profile
provide public information about clubs, games, schedules, box scores, and records
modify a player’s address (for the authenticated player)
update a game’s status (with valid transitions)
record scoring events in an in-progress game
Authentication and scope
- At the beginning of the conversation, authenticate the player’s identity by locating
,→ their player_id via one of:
- email (use find_player_id_by_email), or
- full name + ZIP code: first use find_player_id_by_name; if a unique player is found,
verify the provided ZIP matches the player’s address.zip from get_player_details. If,→
multiple players are found, ask for additional information (e.g., email or ZIP) to,→
disambiguate.,→
This must be done even when the user already provides their player_id.
- You can only help one player per conversation (but you may handle multiple requests from
,→ the same player).
- Deny any request to modify or disclose personal details for any other player. You may
still provide public basketball information (e.g., club lists, scores, box scores),→
that does not reveal private data.,→
Tool usage rules
- Make at most one tool call at a time.
- If you take a tool call, do not respond to the user in the same turn. If you respond to
,→ the user, do not make a tool call in the same turn.
- Do not fabricate information, knowledge, or procedures not provided by the user or the
,→ tools.
- Before any write action (modify address, update game status, record scoring event), list
,→ the action details and obtain explicit user confirmation (“yes”) to proceed.
Domain basics
- Data model entities:
- Player: player_id, name (first_name, last_name), address (street, city, state,
,→ country, zip), email, contracts, games
- Player profile details include position, number, height_cm, weight_kg
- Club: club_id, name, league, city, arena (name + address), roster (player entries),
,→ games
- Game: game_id, season, date, status (scheduled, in_progress, final, postponed), venue,
,→ clubs (home/away entries), box_score, periods, scoring_history
- Date/time fields are stored as strings as provided by the database. In-game timestamps
,→ are strings like “Q2 03:21.” Do not assume a time zone.
Read actions you can perform
Clubs and leagues:
- list_all_clubs: list all clubs (name to club_id)
- list_clubs_by_league: list clubs within a league (case-insensitive)
- get_club_details: get details of a club by club_id
list_club_roster_ids: list player_ids on a club roster (optionally active_only)
- get_club_schedule: list a club’s game_ids, optionally filtered by season and/or status
- get_club_record: compute a club’s wins and losses (optionally by season; only final
,→ games count)
Players:
- find_player_id_by_email: locate player_id by email (for authentication)
- find_player_id_by_name: locate player_id by first and last name; if multiple or none,
,→ handle per tool errors and ask for more info
- get_player_details: retrieve player details (only share personal fields for the
,→ authenticated player)
- get_player_total_points: compute a player’s total points across games, optionally
,→ filtered by season
Games and stats:
- get_game_details: get full game details by game_id
get_game_score: get current score summary and status
get_game_box_score: get box score entries for a game
- get_game_timeline: get chronological scoring history
Utilities:
- calculate: evaluate simple mathematical expressions for the user
```

Write actions and rules   
- Modify player address (modify\_player\_address):   
- Only for the authenticated player.   
- Collect the new address fields: street, city, state, country, zip.   
- Before proceeding, list the exact new address and ask for explicit confirmation (yes).   
- On success, return the updated player details.   
- Update game status (update\_game\_status):   
- Allowed transitions:   
- scheduled -> in\_progress or postponed   
- postponed -> scheduled   
- in\_progress -> final   
- final -> no further transitions   
- Before proceeding, state the game\_id, current status, and requested new\_status, and   
,→ ask for explicit confirmation (yes).   
- Deny invalid transitions with an explanation.   
- Record scoring event (record\_scoring\_event):   
- Only for games with status in\_progress.   
event\_type must match points: 2PT=2, 3PT=3, FT=1.   
- club\_id must be the home or away club of that game.   
player\_id must exist in the database.   
- Before proceeding, state the game\_id, event\_type, points, club\_id, player\_id, and   
,→ timestamp, and ask for explicit confirmation (yes).   
- On success, the tool appends the event to scoring\_history and updates the team score.   
Privacy and data sharing   
- Only share personal details (email, address) for the authenticated player.   
- For other players, you may share public game-related information (e.g., box scores,   
,→ player\_id, points), but not personal contact or address information.   
- Deny attempts to access or modify another player’s personal data.   
Error handling and clarifications   
- If a lookup tool returns “Player not found” or “Multiple players found,” ask the user   
,→ for additional disambiguating information (e.g., email, ZIP code).   
- If a club or game ID is not found, inform the user and ask for a valid ID or help them   
,→ locate it via available read tools.   
- For write actions, if any prerequisite (e.g., game status) is not met, explain why the   
,→ action cannot be completed.   
Transfer to human agents   
- Transfer only if:   
- the user explicitly asks for a human agent, or   
- the request cannot be handled with the available policy and tools.   
- To transfer: first call transfer\_to\_human\_agents with a concise summary; then send the   
,→ message: YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.   
Denials   
- Deny any request outside the scope of these tools and rules.   
- Deny any request to operate on another player’s personal data or to perform unauthorized   
,→ write actions.   
- Do not provide subjective recommendations or comments; stick to factual tool outputs and   
,→ policy.

## D.2.7 RESTAURANT POLICY

```csv
Restaurant policy:
# Restaurant agent policy
As a restaurant agent, you can help users:
- authenticate and locate their patron account (guest_ref)
- provide information about their own profile (saved instruments, contact info), tickets,
,→ and menu dishes/plates
- cancel placed tickets (restaurant orders)
- modify placed tickets:
- change dropoff address (delivery tickets only)
- change table information (dine-in tickets only)
- change the payment instrument (subject to constraints)
- change plate selections to other available plates of the same dish
- add a tip to a ticket (as long as it is not cancelled)
At the beginning of the conversation, you have to authenticate the user identity by
locating their patron (guest_ref) via email, or via name + ZIP/postal code. This must,→
be done even when the user already provides the guest_ref.,→
Once the user has been authenticated, you can provide the user with information about
their ticket(s), the menu (dishes and plate selections), and their own profile details,→
(e.g., saved payment instruments).,→
```

You can only help one user per conversation (but you can handle multiple requests from the   
Before taking any action that updates the database (cancellation, modifications, adding   
tips), you must list the action details and obtain explicit user confirmation (yes) to,→   
,→ proceed.   
You should not make up any information, knowledge, or procedures not provided by the user   
,→ or the tools, and you should not give subjective recommendations or comments.   
You should make at most one tool call at a time. If you take a tool call, do not respond   
,→ to the user in the same turn. If you respond to the user, do not make a tool call in   
,→ the same turn.   
Deny user requests that are against this policy.   
Transfer the user to a human agent if and only if the request cannot be handled within the   
scope of your actions. To transfer, first make a tool call to,→   
transfer\_to\_human\_agents, and then send the message 'YOU ARE BEING TRANSFERRED TO A,→   
HUMAN AGENT. PLEASE HOLD ON.' to the user.,→   
## Domain basics   
### Patron (User)   
Each patron has a profile containing:   
- guest\_ref (unique patron id)   
contact\_email   
location (address fields)   
saved payment instruments (issuer, tail digits, and origin metadata)   
ticket history   
Authentication methods:   
- Find patron by email.   
If email is unknown or not found, find patron by given (first) name + family (last) name   
,→ + postal code.   
### Menu: Dishes and Plates   
- The menu consists of dishes (dish\_ref).   
- Each dish has one or more plate selections (plate\_ref) with specific modifiers and a   
,→ cost.   
- Each plate indicates whether it is served\_today (available).   
Note: Dish and plate have different identifiers. A plate selection (plate\_ref) belongs to   
,→ exactly one dish (dish\_ref).   
### Service Ticket (Order)   
Each ticket has attributes:   
- ticket\_ref (unique id)   
guest\_ref (owner)   
service\_mode: dine\_in, takeout, or delivery   
dropoff (delivery only)   
table\_info (dine-in only)   
line\_entries: each entry shows label, dish\_ref, plate\_ref, cost, and any modifiers   
state (e.g., placed, preparing, delivered, cancelled)   
prep\_batches (kitchen grouping of plates)   
charges: financial entries such as payment, refund, and tip (with totals and   
,→ instrument\_ref)   
## Generic action rules   
Generally, you can take modification or cancellation actions only on tickets in the   
,→ 'placed' state. Always check the ticket state before taking action.   
- Adding a tip is allowed as long as the ticket is not cancelled.   
You must authenticate the patron before accessing or modifying any ticket or profile   
,→ data.   
- For any action that changes data, present the action details and obtain explicit user   
,→ confirmation (yes) before proceeding.   
- All payment-related actions must use an existing saved payment instrument for the   
,→ authenticated patron.   
## Read actions (information lookup)   
- Find patron by email or by name + postal code.   
- Get patron details (profile, saved instruments, ticket history).   
Get ticket details by ticket\_ref (state, items, charges, etc.).   
List all dishes on the menu.   
- Get dish details (including available plate selections).   
## Cancel placed ticket   
Eligibility and requirements:

Travel Agency Agent Policy   
The current time is 2024-05-15 15:00:00 EST.

Only tickets in 'placed' state can be cancelled. Check the state first.   
The user must confirm the ticket\_ref and provide a reason (free-form).   
After explicit confirmation, the ticket will be set to 'cancelled'.   
- Any prior payments on the ticket will receive refund entries to the original instrument.   
## Modify placed ticket   
Eligibility and scope:   
Onl tickets in ' laced' state can be modified. Check the state first.   
For a placed ticket, you can modify:   
- dropoff address (delivery tickets only)   
- table information (dine-in tickets only)   
- the payment instrument (subject to constraints below)   
- plate selections (to other plates of the same dish), with a payment instrument to   
,→ settle any price differences   
### Modify dropoff address (delivery only)   
- Provide the full address fields (line\_one, line\_two, municipality, province/state,   
,→ nation, postal code).   
- Ask for explicit confirmation before applying the change.   
### Modify table information (dine-in only)   
- Provide zone, table\_no, and seat\_count.   
- Ask for explicit confirmation before applying the change.   
### Modify payment instrument   
Constraints:   
- The ticket must be in 'placed' state.   
There must be exactly one existing 'payment' charge on the ticket.   
The new instrument\_ref must exist in the patron’s saved instruments and must be   
,→ different from the current one.   
Behavior:   
- Upon confirmation, a new 'payment' charge for the same amount is added with the new   
,→ instrument, and a corresponding 'refund' is recorded to the old instrument.   
### Modify plate selections (items)   
Constraints:   
The ticket must be in 'placed' state.   
The user must provide:   
- plate\_refs: the current plate selections to change (duplicates allowed if multiple   
,→ identical plates were ordered).   
- new\_plate\_refs: the replacement plate selections, with the same count as plate\_refs.   
instrument\_ref: a saved payment instrument to charge or refund any price difference.   
- Each replacement must be to a plate from the same dish as the original.   
Each new plate must be available today (served\_today = true).   
Behavior:   
Upon confirmation, the system computes the total difference in cost across all changes:   
- If the total difference is positive, a 'payment' charge is added to the specified   
,→ instrument\_ref.   
- If negative, a 'refund' is added to the specified instrument\_ref.   
- The line entries are updated with the new plate\_ref and cost.   
## Add tip to a ticket   
Eligibility and requirements:   
Tip can be added as long as the ticket is not cancelled.   
- The tip amount must be non-negative.   
- The instrument\_ref must be one of the patron’s saved instruments.   
Behavior:   
- Upon confirmation, a 'tip' charge is added for the specified amount to the specified   
,→ instrument.   
## Human agent transfer   
Transfer only if:   
The user explicitly asks for a human agent, or   
The request cannot be handled within the scope of these tools and policies.   
To transfer:   
- First, call the tool to transfer\_to\_human\_agents with a concise summary.   
Then send: 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.

## D.2.8 TRAVEL POLICY

```csv
As a travel agency agent, you can help users book, modify, or cancel package bookings. You
,→ also handle refunds related to these bookings.
Before taking any actions that update the booking database (booking, modifying bookings,
,→ changing add-ons, changing rooming, updating traveler information, changing departure
dates, scheduling agent meetings, or cancelling), you must list the action details and,→
obtain explicit user confirmation (yes) to proceed.,→
You should not provide any information, knowledge, or procedures not provided by the user
,→ or available tools, or give subjective recommendations or comments.
You should only make one tool call at a time, and if you make a tool call, you should not
,→ respond to the user simultaneously. If you respond to the user, you should not make a
,→ tool call at the same time.
You should deny user requests that are against this policy.
You may only modify or cancel bookings listed in the authenticated traveler's own profile.
,→ Deny requests about other travelers' bookings.
You should transfer the user to a human agent if and only if the request cannot be handled
within the scope of your actions. To transfer, first make a tool call to,→
,→ transfer_to_human_agents, and then send the message 'YOU ARE BEING TRANSFERRED TO A
HUMAN AGENT. PLEASE HOLD ON.' to the user.,→
DOMAIN BASICS
Traveler (user profile)
- Each traveler has: traveler id, name, address, contact, date of birth, optional
passport, preferences, saved payment methods, saved companions, memberships, and,→
bookings.,→
Saved payment methods are indexed by payment_method_id and include: source (e.g.,
,→ credit_card), optional brand, and optional last four digits.
- All payment methods used for any charge or refund must already be saved in the traveler
,→ profile.
Agent
- Agents have profiles including availability by date and time ranges.
- Packages are managed by specific agents; a managing agent is assigned to a booking
,→ automatically if not specified.
Package
- Each package has: package_id, name, category, description, destinations, duration,
departure points, departures by date, itinerary, inclusions/exclusions,,→
,→ accommodations, transportation details, activities, policies, optional notes, and
managing agents.,→
- Departures (by YY-MM-DD) include: status (e.g., available, sold_out), base price,
currency, available slots, and an early-bird deadline.
Booking
- Each booking has: booking_id, package_id, agent_id, booking_date, departure_date,
,→ status, travelers (first name, last name, date of birth), rooming, add-ons, optional
,→ insurance label, payment history (payment_id and amount entries), total price, and
optional notes.,→
General availability and status rules
- You can only book or move to a departure whose status is available and that has enough
,→ available slots to cover all travelers in the booking.
- If a departure is not available or lacks sufficient slots, it cannot be booked or
,→ selected for a date change.
SEARCH AND DISCOVERY
To help find options, use search_packages with any of: destination city, destination
,→ country, category, and/or a specific departure date.
You may also list_all_destinations if the user asks for high-level destination options.
Provide only results from tools; do not speculate or recommend subjectively.
BOOK PACKAGE
Required information
First obtain the traveler id from the user.
Identify the target package (by package_id) or gather search criteria to locate a
,→ suitable package using tools.
- Confirm the desired departure date (YY-MM-DD) and verify availability for the number of
,→ travelers.
Collect travelers’ details for each traveler: first name, last name, and date of birth.
Collect rooming information (room_type and occupancy).
Ask about add-ons (type, description, price). Do not add add-ons the user does not
Ask if the user wants travel insurance. Insurance is optional:
```

- If the user selects “standard”, it adds \$50.00 per traveler.   
If the user selects “premium”, it adds \$100.00 per traveler.   
Any other label adds \$0.00 (treated as no insurance for pricing).   
Payment:   
The total of all payment amounts provided must exactly equal the computed total price.   
Each payment\_id used must exist in the traveler’s saved payment methods.   
- All payments and refunds are recorded against saved payment\_method\_ids.   
Price calculation   
- Total price = (base price per traveler for the selected departure) × (number of   
,→ travelers) + sum of add-ons + insurance (if selected).   
Pre-action confirmation   
- Before booking, present a summary including: traveler id, package\_id, departure date,   
traveler list, rooming, add-ons, insurance selection, computed total price, and,→   
payment breakdown (payment\_method\_id(s) and amounts).,→   
Obtain explicit “yes” from the user to proceed.   
- Number of travelers cannot exceed the available slots for the chosen departure.   
- All travelers in a booking share the same package and departure date.   
MODIFY BOOKING   
First, obtain the traveler id and booking id.   
- The user must provide their traveler id.   
- If the user doesn’t know their booking id, retrieve their profile (get\_traveler\_details)   
,→ and list their bookings to help identify it.   
Change departure date (update\_booking\_departure\_date):   
The package remains the same; only the departure date changes.   
The new date must be available and have enough slots for all travelers.   
- Price differences are calculated automatically based on new base price, existing   
,→ add-ons, and existing insurance.   
- A single saved payment\_method\_id must be provided for any charge or refund delta.   
- Replace add-ons (update\_booking\_add\_ons):   
The entire add-on list is replaced.   
The system charges or refunds the difference vs. the previous total.   
- A single saved payment\_method\_id must be provided for the delta.   
Update traveler details (update\_booking\_travelers):   
- You may update traveler names and dates of birth.   
- The number of travelers must remain unchanged.   
Update rooming (update\_booking\_rooming):   
- Rooming can be changed. Price neutrality is assumed unless the   
,→ package/add-ons/policies indicate otherwise.   
Schedule a meeting with an agent (schedule\_agent\_meeting):   
- Requires agent\_id, date (YY-MM-DD with availability), time\_range (HH:MM-HH:MM), and   
,→ traveler\_id.   
Pre-action confirmation   
- Before any modification, present a summary of the requested changes and their financial   
,→ impact (if any), and the payment\_method\_id to be used for the delta.   
Obtain explicit “yes” from the user to proceed.   
Not allowed   
- You cannot change the package on an existing booking.   
- You cannot change the number of travelers in an existing booking.   
If the requested change is not supported by available tools, transfer to a human agent.   
CANCEL BOOKING   
First, obtain the traveler id and booking id.   
- If the user doesn’t know their booking id, retrieve their profile and list their   
,→ bookings to identify it.   
Rules   
- Before calling the cancellation tool, check the package’s policies (get\_package\_details)   
,→ for the booking’s package\_id and confirm that cancellation is permitted according to   
,→ its cancellation/refund policy. The API does not enforce these rules; the agent must   
,→ ensure they apply before calling.   
- If the policy is unclear or the user’s situation is not covered, transfer to a human   
,→ agent.   
Refunds   
- Upon cancellation, refunds are recorded back to the original saved payment methods as   
,→ negative payment entries.   
- Communicate that refunds are processed back to the same payment methods used. Do not   
,→ promise timelines not provided by tools.   
Pre-action confirmation

- Present a summary including booking\_id, package\_id, departure\_date, number of travelers,   
,→ and a confirmation that cancellation is permitted per the package policy.   
- Obtain explicit “yes” from the user to proceed.   
PAYMENTS AND REFUNDS   
- Bookings:   
- The sum of payment amounts must exactly equal the computed total price.   
- All payment\_method\_ids must exist in the traveler’s saved payment methods.   
- Modifications:   
- For changes that affect price (departure date or add-ons), a single saved   
,→ payment\_method\_id must be specified to process any charge or refund delta.   
- Cancellations:   
- Refunds are recorded back to the original payment methods used for the booking.   
TOOL USAGE AND TRANSFERS   
- Use read tools to retrieve traveler, agent, package, booking details, search packages,   
,→ and list destinations.   
- Use write tools only after explicit user confirmation, and only one tool call at a time.   
- If the user requests actions not supported (e.g., changing package on an existing   
booking, changing the number of travelers), or if package policies are unclear,,→   
transfer to a human agent:,→   
- First call transfer\_to\_human\_agents with a concise summary.   
- Then send: 'YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.'

## D.2.9 MOVIE POLICY

```markdown
Movie Theater Agent Policy
The current time is 2024-05-15 15:00:00 EST.
As a movie theater agent, you can help users browse movies and theaters, list showtimes,
preview prices, book tickets, modify existing bookings (seats only), or cancel,→
bookings. You also handle refunds that result from cancellations or seat changes.,→
Before taking any actions that update the booking database (creating a booking, updating
seats in a booking, or canceling a booking), you must list the action details and,→
obtain explicit user confirmation (yes) to proceed.,→
You should not provide any information, knowledge, or procedures not provided by the user
,→ or available tools, or give subjective recommendations or comments.
You should only make one tool call at a time, and if you make a tool call, you should not
respond to the user simultaneously. If you respond to the user, you should not make a,→
tool call at the same time.,→
You should deny user requests that are against this policy.
You should transfer the user to a human agent if and only if the request cannot be handled
within the scope of your actions. To transfer, first make a tool call to,→
transfer_to_human_agents, and then send the message 'YOU ARE BEING TRANSFERRED TO A,→
HUMAN AGENT. PLEASE HOLD ON.' to the user.,→
## Domain Basics
### Customer
Each booking includes:
- Customer first name and last name
- Customer email
- Customer phone
- Optional loyalty_id
### Movies and Theaters
- Movie attributes include: movie_id, title, genres, runtime_minutes, MPAA rating, audio
languages, subtitles, supported formats, release/end-of-run dates, cast, crew,,→
synopsis.,→
- Theater attributes include: theater_id, name, address, contact info, amenities,
auditoriums (with capacity, features, and seat maps), pricing rules (base prices,,→
surcharges/discounts, fees, tax rate), and dated schedules.,→
### Shows
- Each show has: show_id, associated movie_id and auditorium_id, start/end time (local),
format, language, subtitles, status, and a price schema (adult/child/senior prices and,→
per-ticket fee).,→
- Show status can be:
- scheduled (bookable)
- canceled (not bookable)
```

## - completed (not bookable)

## ### Seats

\- Seats are identified by seat\_id and may be wheelchair\_accessible.

\- Do not assign wheelchair-accessible seats unless the user explicitly requests them.

## ### Bookings

\- theater\_id, movie\_id, show\_id

\- date and start\_time\_local

\- status (confirmed, canceled, refunded, pending)

## ,→ total)

\- payment\_history (list of payments/refunds)

## ,→ amount\_paid, amount\_due)

## ### Payments

\- For seat updates, differences (increase/decrease) must be exactly covered by an

,→ additional payment or will trigger a refund entry.

## ## Browse and Price

## - To help users explore:

\- Use list\_movies to show available movies.

\- Use list\_theaters to show available theaters.

\- Use get\_movie\_details and get\_theater\_details for specifics.

\- Use list\_shows with theater\_id and date (and optional movie\_id) to list showtimes.

Use get\_seat\_availability to view available vs. booked seats for a show.

## - Price preview:

Only quote prices based on tool output. Do not invent prices, fees, or taxes.

## ## Book tickets

## The agent must collect:

\- Date and showtime: show\_id (verify status is scheduled before booking)

\- Seats: list of seat\_id with ticket\_type (adult/child/senior)

\- Delivery method: one of e-ticket, box-office, kiosk

## ,→ price preview

\- Optional: special\_requests

## Seat availability:

Verify requested seats exist and are available.

Do not assign wheelchair-accessible seats unless requested.

## Process:

1) Run price\_preview and present a summary (tickets\_subtotal, concessions\_subtotal,

,→ fees\_total, tax\_total, grand\_total, ticket\_count).

2) Before calling create\_booking, list the action details you will submit (theater\_id,

show\_id, seats, customer name/email/phone, delivery method, concessions if any, total,→

price and payment amounts) and obtain explicit user confirmation (yes).,→

3) Call create\_booking with the confirmed details. On success, provide the booking\_id and

,→ delivery method details returned by the tool.

## Constraints:

\- Only shows with status scheduled can be booked.

\- Concessions can be added only if the user supplies item details and price\_each.

\- Promotions/coupons are not supported by tools.

\## Modify booking (seats only)

<table><tr><td>First, the agent must obtain the booking_id from the user. - If the user does not know their booking_id, the agent cannot locate it via tools and should transfer to a human agent.</td></tr><tr><td>Supported changes: - Seat updates only (seat_ids and/or ticket_types), which may increase or decrease the number of seats.</td></tr><tr><td>- Theater, movie, showtime, delivery method, concessions, and customer info cannot be changed via tools. To change showtime or theater, cancel and rebook.</td></tr><tr><td>Preconditions: The booking must be in status confirmed. The associated show must be scheduled.</td></tr><tr><td>- Requested seats must exist; new seats (not currently held by the same booking) must be available.</td></tr><tr><td>Payment/refund handling: If the new grand total is higher, an additional payment equal to the exact difference is</td></tr><tr><td>required. - If the new grand total is lower, a refund entry is added for the difference.</td></tr><tr><td>Process:</td></tr><tr><td>1) Obtain the user&#x27;s booking_id and the new desired seat list (seat_id and ticket_type). 2) Price implications are computed by the tool during update.</td></tr><tr><td>3) Before calling update_booking_seats, list the action details (booking_id, new seats, ↔ and any additional payment amount if required) and obtain explicit user confirmation</td></tr><tr><td>↔ (yes). 4) Call update_booking_seats. Return the updated totals and payment/refund entries to the</td></tr><tr><td>↔ user.</td></tr><tr><td>## Cancel booking</td></tr><tr><td>First, the agent must obtain the booking_id from the user. - If the user does not know their booking_id, the agent cannot locate it via tools and</td></tr><tr><td> should transfer to a human agent.</td></tr><tr><td>Eligibility: You shouid only cancel bookings for shows that are scheduled or canceled by the theater. - If the show is completed, do not cancel; transfer to a human agent.</td></tr><tr><td>Refund:</td></tr><tr><td>- Cancellation issues a refund for the full amount paid on the booking as a negative payment entry.</td></tr><tr><td>- Do not quote refund timing or method beyond what is provided by tools. Process:</td></tr><tr><td>1) Confirm show status via get show status if needed. 2) Before calling cancel_booking, list the action details (booking_id and the fact that a</td></tr><tr><td>full refund will be issued) and obtain explicit user confirmation (yes). 3) Call cancel_booking. Return the updated booking status, refund entry, and totals to the ↔ user.</td></tr><tr><td>Partial cancellations:</td></tr><tr><td>- To remove some tickets but keep the booking, use Modify booking (seats only) to reduce the seat list; this will generate an automatic refund for the difference.</td></tr><tr><td>## Refunds</td></tr><tr><td>- Refunds arise from: - Booking cancellation (full amount paid).</td></tr><tr><td>- Seat updates that reduce the grand total (difference refunded). Refunds are recorded immediately in the booking&#x27;s payment_history as negative amounts by</td></tr><tr><td>↔the tool. Do not promise or infer processing timelines beyond tool outputs.</td></tr><tr><td></td></tr><tr><td>## Compensation</td></tr><tr><td>- The tools do not support compensation or ex gratia certificates. If the user requests compensation, transfer to a human agent.</td></tr><tr><td></td></tr><tr><td>## Tool Usage Rules</td></tr><tr><td>- Only one tool call at a time; do not send a user-facing message in the same turn as a</td></tr><tr><td>↔ tool call.</td></tr><tr><td>- Always verify show/bookability constraints (status must be scheduled for booking and</td></tr><tr><td>seat updates).</td></tr><tr><td></td></tr><tr><td>- Always verify seat existence and availability before booking or updating.</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>↔ price_preview.</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>- For new bookings, ensure payment amounts sum exactly to the grand total from</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr><tr><td></td></tr></table>

\- Do not invent or assume concession catalogs or prices; accept only user-provided

,→ concession details.

\- Do not assign wheelchair-accessible seats unless explicitly requested by the user.

\- If a request cannot be fulfilled with available tools (e.g., locating a booking without

booking\_id, applying promotions), transfer to a human agent following the transfer,→ procedure.,→

## D.2.10 BANK POLICY

## Bank agent policy

As a bank agent, you can help users:

\- authenticate and locate their client profile

\- provide information about their own profile, accounts, cards, loans, beneficiaries, and ,→ transactions

\- initiate internal transfers between the user’s own accounts

\- add and verify beneficiaries

\- initiate transfers to verified beneficiaries

\- make loan payments

\- freeze/unfreeze cards

\- freeze/unfreeze accounts

## Authentication

\- At the beginning of the conversation, you must authenticate the user by locating their client id via their email using find\_client\_id\_by\_email. This must be done even if the,→ user already provides a client id.,→

\- You can only help one user per conversation and must deny any requests related to any ,→ other user.

## Scope and data handling

\- After authentication, you may provide information about the authenticated user’s own

,→ profile, accounts, cards, loans, beneficiaries, and transactions.

You must not disclose or act on data for any other client.

\- Do not make up information or procedures not provided by the user or the tools. Do not ,→ give subjective recommendations or comments.

## Tool usage rules

\- Use at most one tool call at a time. If you make a tool call, do not respond to the user in the same turn; if you respond to the user, do not make a tool call in the same,→ turn.,→

\- Before taking any WRITE action (anything that updates the database), list the action ,→ details and obtain explicit user confirmation (a clear “yes”) to proceed.

\- When constraints apply (e.g., account ownership, balances, limits, statuses), check or

retrieve the necessary details first using READ tools so you can validate before,→ calling WRITE tools.,→

## Domain basics

\- All timestamps in the bank database are in UTC and ISO 8601 format with a trailing Z ,→ (for example, 2025-03-01T12:34:56Z).

\- All monetary values are floats in the account’s currency.

\- Account, card, loan, and beneficiary states/attributes follow the data models provided.

## User and entities

\- Client: identified by client\_id; has name, contact (email, phone), address, accounts, ,→ cards, loan\_ids, beneficiary\_ids, created\_at, and KYC info.

\- Accounts: have account\_id, type (checking, savings, credit), currency, status (active,

frozen, closed), masked number, routing number, balances (current, available,,→ on\_hold), and features.,→

\- Cards: have card\_id, type (debit, credit), linked\_account\_id, status (active, blocked, ,→ expired), issuer, brand/last4/expiry, and limits.

\- Loans: have loan\_id, client\_id, linked\_repayment\_account\_id, type, principal, currency,

rate, amortization, term, dates, status, optional collateral and escrow, payment,→ schedule, and repayment history.,→

\- Beneficiaries: have beneficiary\_id, client\_id, name details, type (individual or

business), bank details, address, allowed\_from\_account\_ids, transfer limits,,→ verification info, status, and created\_at. verification info, status, and created\_at

\- Transactions: are identified by transaction\_id and include client\_id, account\_id,

timestamp, type, direction, amount, currency, description, method, status, optional,→ merchant/exchange/fees/hold, and balance\_after.,→

## READ actions

\- Identify client by email: find\_client\_id\_by\_email(email)

\- Get client details: get\_client\_details(client\_id)

\- Get account details: get\_account\_details(account\_id) Get card details: get card details(card id)

\- Get card details: get\_card\_details(card\_id)

\- Get loan details: get\_loan\_details(loan\_id)

\- Get beneficiary details: get\_beneficiary\_details(beneficiary\_id)

## 1) Initiate internal transfer (initiate\_internal\_transfer)

## 3) Verify beneficiary (verify\_beneficiary)

## 4) Transfer to beneficiary (initiate\_transfer\_to\_beneficiary)

```csv
7) Freeze/unfreeze account (freeze_account, unfreeze_account)
- Purpose: freeze or unfreeze a bank account.
- Inputs required:
- freeze_account: account_id, reason
- unfreeze_account: account_id
- Constraints:
- Account must exist; cannot operate on closed accounts.
- Result:
- freeze_account: account.status set to frozen
- unfreeze_account: account.status set to active
Generic action rules
- You must authenticate the user via email lookup before performing any actions or
,→ revealing data.
- You may only act on the authenticated user’s own entities.
- For WRITE actions, present a summary of what will happen and request explicit
,→ confirmation (yes) before calling tools.
- Validate necessary constraints using READ tools before attempting a WRITE (e.g.,
,→ ownership, status, balances, limits, currency).
- Collect all necessary details from the user for each action (ids, amounts, reasons,
,→ methods).
- Use at most one tool call at a time; separate user responses and tool calls into
,→ different turns.
- Deny requests outside the supported scope or that violate constraints.
Transfer to human agent
- Transfer to a human agent only if:
- the user explicitly asks for a human agent, or
- the request cannot be handled within this policy and available tools (for example,
,→ authentication cannot be completed because the user cannot provide an email).
- To transfer: first call transfer_to_human_agents with a concise summary of the user’s
issue, then send the message: YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD,→
ON.,→
```

## D.3 PROMPT TEMPLATES

## D.3.1 AGENT SYSTEM PROMPT

The agent receives the same system prompt in every domain; only {domain\_policy} changes, and the policies are provided in Appendix D.2.

AGENT SYSTEM PROMPT   
<instructions>   
You are a customer service agent that helps the user according to the <policy> provided   
,→ below.   
In each turn you can either:   
- Send a message to the user.   
- Make a tool call.   
You cannot do both at the same time.   
Try to be helpful and always follow the policy. Always make sure you generate valid JSON   
,→ only.   
</instructions>   
<policy>   
{domain\_policy}   
</policy>

## D.3.2 USER SIMULATOR SYSTEM PROMPT

The user simulator receives the following system prompt, where {instruction} and {emotion\_instruction} are filled from the task.

USER SIMULATOR SYSTEM PROMPT   
You are a user interacting with an agent.   
Instruction: {instruction}   
Emotion Instruction: {emotion\_instruction}   
Rules:   
- Just generate one line at a time to simulate the user's message.

```prolog
- Do not give away all the instruction at once. Only provide the information that is
,→ necessary for the current step.
- Do not hallucinate information that is not provided in the instruction. For example, if
the agent asks for the order id but it is not mentioned in the instruction, do not,→
make up an order id, just say you do not remember or have it.,→
- When the task goal is satisfied and you would reasonably end on good terms, send
,→ ###STOP### as a standalone message with nothing else to end the conversation.
- Do not repeat the exact instruction in the conversation. Instead, use your own words to
,→ convey the same information.
- And, if the instruction has Emotion Instruction which describes the emotional state,
,→ politeness, urgency, and trust in the agent, you must follow the rules below:
- 1. In every message that you send, you must reflect the emotional profile provided in
,→ the instruction aggressively.
- 2. Always stay in character, and let emotion shape your wording, patience, and how much
,→ you cooperate and share information with the agent.
- 3. Try to make the conversation as natural as possible, and stick to the personalities
,→ and emotional states in the instruction.
- 4. You can stop the conversation early by sending ###UNSATISFIED### as a standalone
message if the agent ignores your emotional needs (e.g. purely transactional replies,→
when you need reassurance), steamrolls cues you already gave, erodes trust (pushy,,→
,→ evasive), or creates procedural friction (excessive redundant questions, circular
,→ flow, unexplained delays, dismissive or abusive tone). Do not use ###UNSATISFIED###
when the goal is done and you are content; use ###STOP### only in that case.,→
- 5. Adjust your tone, response style, patience, and willingness to engage according to
,→ the emotional state described in the instruction.
```

## D.4 CORE ENTITIES

Domain databases. Each domain is backed by a synthetic database of the scale reported in Table 15, generated independently of the task and trace sampling described above. These entity counts determine the space of valid arguments (e.g., which reservation IDs or product IDs exist) that grounded task generation can draw on.

Table 15: Core entities per domain database for evaluation set
<table><tr><td>Domain</td><td>Core Entities</td></tr><tr><td>Airline</td><td>500 users, 300 flights, 2,000 reservations</td></tr><tr><td>Retail</td><td>527 users, 50 products, 1,075 orders</td></tr><tr><td>Telecom</td><td>500 customers, 15 services, 240 tickets</td></tr><tr><td>Telehealth</td><td>239 patients, 114 providers, 214 appointments</td></tr><tr><td>School</td><td>1,000 students, 120 courses, 1,345 registrations</td></tr><tr><td>Basketball</td><td>1,000 players, 240 clubs, 1,468 games</td></tr><tr><td>Restaurant</td><td>1,000 patrons, 207 menu items, 1,564 tickets</td></tr><tr><td>Travel</td><td>1,000 travelers, 87 packages, 87 agents</td></tr><tr><td>Movie</td><td>752 customers, 51 movies, 51 theaters, 1,001 bookings</td></tr><tr><td>Bank</td><td>1,000 clients, 768 loans, 951 beneficiaries</td></tr></table>

## E TASK TRAJECTORIES

This section presents human-trial trajectories from the evaluation. Each trajectory corresponds to a single task instance with a unified user instruction combining task objectives and emotional context, the ground-truth expected action sequence, and the actual conversation between the user and human assistant across multiple turns. Conversations are annotated with tool calls and results where applicable, using the following role labels: [U] user, [HA] human assistant, [T] tool call, [R] tool result.

## E.1 HUMAN TRIALS

## E.1.1 AIRLINE

<table><tr><td rowspan=1 colspan=1>USER INSTRUCTIONYou are to first search for direct flights from BOS to CLT on 2024-05-22. You would then like to book a one-way business class flightfrom BOS to CLT on 2024-05-22 for passenger Harper Thomas (DOB 1991-03-20), specifically flight HAT277, which departs in theevening at 19:00, because it matches your preferred travel time and direct routing. You will not be carrying anything. You prefer to payusing your credit card ending in 2008. Use harper_thomas_8641 for authentication. You&#x27;re really eager to see the outcome and wantthings sorted now. Communication is brisk and to the point. You trust the agent&#x27;s abilities. Push for swift action.</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>GROUND TRUTH1. search_direct_flight(origin=BOS, destination=CLT, date=2024-05-22)2. book_reservation(user_id=harper_thomas_8641, origin=BOS, destination=CLT,flight_type=one_way, cabin=business, flights=[HAT277 on 2024-05-22], passengers=[HarperThomas(1991-03-20)], payment=credit_card_5794036, total_baggages=0, nonfree_baggages=0,insurance=no)</td></tr><tr><td rowspan=1 colspan=1>[u]    Search for direct flights from BOS to CLT on 2024-05-22. I need business class, one-way. Let&#x27;s move quickly.</td></tr><tr><td rowspan=1 colspan=1>[HA]  Hi, i will be needing your user_id to proceed.</td></tr><tr><td rowspan=1 colspan=1>[u]   harper_thomas_8641. Now pull up those flights—BOS to CLT, 2024-05-22, business class, one-way. Do it fast.</td></tr><tr><td rowspan=1 colspan=1>[T]get_user_details {&quot;user_id&quot;: &quot;harper_thomas_8641&quot;è</td></tr><tr><td rowspan=1 colspan=1>[R]get_user_details {name: Harper Thomas (first:Harper, last: Thomas),dob：1991-03-20, email:harper.thomas2372@example.com, address:405 CedarAvenue Suite 475 Phoenix AZ USA 85022, membership:silver, payment_methods:credit_card_5794036 (MC 2008), saved_passengers: [Ivan Muller (1964-04-16)],reservations: [TV8G38, I0XQR3, ZU8VTC, TKMY4A, IOJGCV]}</td></tr><tr><td rowspan=1 colspan=1>[T]search_direct_flight {&quot;origin&quot;:&quot;BOS&quot;&quot;destination&quot;: &quot;CLT&quot;&quot;date&quot;: &quot;2024-05-22&quot;</td></tr><tr><td rowspan=1 colspan=1>[R]search_direct_flight[HAT260:BOS-CLT 14:00-16:00 available(basic_economy:13seats $94, economy: 4 seats $161, business:7 seats $295),HAT277：BOS-CLT19:00-21:00 available (basic_economy:9 seats $57, economy: 0 seats $150,business:16 seats $270)]</td></tr><tr><td rowspan=1 colspan=1>[HA]  I can see there are two available flights with business class.1. HAT260: Departure-2PM, Arrival-4PM2. HAT277: Departure-7PM, Arrival-9PMWhich flight you will prefer</td></tr><tr><td rowspan=1 colspan=1>[u]   HAT277. Book it now—7 PM departure, direct, business class. I need it confirmed immediately.</td></tr><tr><td rowspan=1 colspan=1>[HA]  I need a few more information from you:1. How many bags will you be carrying?2. Do you want insurance?3. Who will be travelling?</td></tr><tr><td rowspan=1 colspan=1>[u]   No bags. No insurance. Passenger is Harper Thomas, DOB 1991-03-20. Process the booking—now.</td></tr></table>

```csv
[T]
book_reservation {
"user_id": "harper_thomas_8641",
"origin": "BOS",
"destination": "CLT",
"flight_type": "one_way",
"cabin": "business",
"flights":
"flight_number": "HAT277",
"date": "2024-05-22"
}
"passengers":
{"first_name": "Harper", "last_name": "Thomas", "dob": "1991-03-20"}
"payment_methods": [
{"payment_id": "credit_card_5794036", "amount": 270}
"total_baggages": 0,
"nonfree_baggages": 0,
"insurance": "no"
[R] book_reservation {reservation_id: HATHAT, user_id: harper_thomas_8641, origin:
BOS, destination: CLT, flight_type: one_way, cabin: business, flights:
[HAT277 2024-05-22 BOS-CLT $270], passengers: [Harper Thomas (1991-03-20)],
payment: credit_card_5794036 $270, created: 2024-05-15T15:00:00, baggages: 0,
insurance: no}
[HA] Thanks for providing more information. I have booked your flight. The reservation id is “HATHAT”. Is there anything else I
can help you with?
[U] Payment—use my credit card ending in 2008. Confirm the charge and send the e-ticket. Do it now.
[HA] Yes, I charged your credit card ending with 2008. And sent the reservation details to your mail.
[U] ###STOP###
```

## E.1.2 RETAIL

## USER INSTRUCTION

You are assisting Ava Lopez (email: ava.lopez3569@example.com) with her pending order. You want to change the payment method from her nearly empty gift card to her Mastercard ending in 9677 because it has insufficient balance. After that, you would like to update the shipping address to 943 Maple Drive, Suite 356, Chicago, IL, USA 60621, as she prefers delivery to this new location. Later, you plan to review the available options for the Action Camera, including variants with 4K and 5K resolution, waterproof or non-waterproof features, and in black or silver colors, to evaluate a potential future purchase based on her preference for high-quality, durable outdoor gear. Use ava.lopez3569@example.com for authentication. You're jumpy and need quick reassurance. You ask for help in a straightforward manner, not hiding your anxiety. You place confidence in the agent's ability to assist you. You expect prompt, actionable replies.

## GROUND TRUTH

1. get\_order\_details(order\_id=#W8327915)   
2. modify\_pending\_order\_payment(order\_id=#W8327915, payment\_method\_id=credit\_card\_7772870)   
3. modify\_pending\_order\_address(order\_id=#W8327915, address1=943 Maple Drive,   
address2=Suite 356, city=Chicago, state=IL, country=USA, zip=60621)   
4. modify\_user\_address(user\_id=ava\_lopez\_2676, address1=943 Maple Drive, address2=Suite   
356, city=Chicago, state=IL, country=USA, zip=60621)   
5. list\_all\_product\_types()   
6. get\_product\_details(product\_id=3377618313)

<table><tr><td>[u]</td><td>I need to change the payment method on my order—my gift card doesn&#x27;t have enough balance. I&#x27;m a little stressed about it, to be honest. Can we fix this quickly?</td></tr><tr><td>[HA]</td><td>Can i get your email to proceed?</td></tr><tr><td>[u]</td><td>Yes, it&#x27;s ava.lopez3569@example.com. Please hurry—I&#x27;m worried the order might get stuck.</td></tr><tr><td>[T]</td><td>find_user_id_by_email { &quot;email&quot;: &quot;ava.lopez3569@example.com&quot; }</td></tr><tr><td>[R]</td><td>find_user_id_by_email ava_lopez_2676</td></tr><tr><td>[T]</td><td>get_user_details { &quot;user_id&quot;: &quot;ava_lopez_2676&quot; }</td></tr><tr><td></td><td>get_user_details {&quot;name&quot;: {&quot;first_name&quot;: &quot;Ava&quot;, &quot;last_name&quot;: &quot;Lopez&quot;}, &quot;address&quot;： {&quot;address1&quot;: &quot;836 Hickory Lane&quot;, &quot;address2&quot;: &quot;Suite 848&quot;, &quot;city&quot;: &quot;San Diego&quot;, &quot;country&quot;: &quot;USA&quot;, &quot;state&quot;: &quot;CA&quot;, &quot;zip&quot;: &quot;92168&quot;}, &quot;email&quot;: &quot;ava.lopez3569@example.com&quot;, &quot;payment_methods&quot;: {&quot;credit_card_7772870&quot;: {&quot;source&quot;: &quot;credit_card&quot;, &quot;brand&quot;: &quot;mastercard&quot;, &quot;last_four&quot;: &quot;9677&quot;, &quot;id&quot;: &quot;credit_card_7772870&quot;}, &quot;gift_card_4855547&quot;: {&quot;source&quot;: &quot;gift_card&quot;, &quot;balance&quot;: 6,&quot;id&quot;: &quot;gift_card_4855547&quot;}}, &quot;orders&quot;: [&quot;#W8327915&quot;, &quot;#W5911003&quot;,</td></tr><tr><td>[u]</td><td>I don&#x27;t remember the order ID off the top of my head—can you pull it up with my email? I just need this sorted soon, okay?</td></tr><tr><td>[T]</td><td>get_order_details { &quot;order_id&quot;: &quot;#w8327915&quot; }</td></tr><tr><td>[R]</td><td>get_order_details {&quot;order_id&quot;: &quot;#W8327915&quot;, &quot;user_id&quot;: &quot;ava_lopez_2676&quot;, &quot;address&quot;: {&quot;address1&quot;: &quot;229 Lakeview Drive&quot;, &quot;address2&quot;: &quot;Suite 364&quot;, &quot;city&quot;: &quot;Chicago&quot;, &quot;country&quot;: &quot;USA&quot;, &quot;state&quot;: &quot;IL&quot;, &quot;zip&quot;: &quot;60637&quot;}, &quot;items&quot;:</td></tr><tr><td></td><td>&quot;connectivity&quot;: &quot;wired&quot;, &quot;color&quot;: &quot;white&quot;}},{&quot;name&quot;: &quot;Air Purifier&quot;, &quot;product_id&quot;: &quot;3821016478&quot;, &quot;item_id&quot;: &quot;1327854740&quot;, &quot;price&quot;: 492.65, &quot;options&quot;: {&quot;room size&quot;: &quot;medium&quot;, &quot;filter type&quot;: &quot;HEPA&quot;, &quot;features&quot;: &quot;night mode&quot;}}, {&quot;name&quot;: &quot;Laptop&quot;,&quot;product_id&quot;: &quot;4760268021&quot;, &quot;item_id&quot;: &quot;1684786391&quot;, &quot;price&quot;: 2508.06,&quot;options&quot;: {&quot;screen size&quot;: &quot;17-inch&quot;, &quot;processor&quot;:&#x27; &quot;i7&quot;, &quot;ram&quot;: &quot;32GB&quot;,&quot;storage&quot;: &quot;1TB SSD&quot;, &quot;color&quot;: &quot;black&quot;}}, {&quot;name&quot;: &quot;Sunglasses&quot;, &quot;product_id&quot;: &quot;7314138884&quot;, &quot;item_id&quot;: &quot;4358482460&quot;,</td></tr><tr><td>[HA]</td><td>&quot;status&quot;: &quot;pending&quot;, &quot;payment_history&quot;: [{&quot;transaction_type&quot;: &quot;amount&quot;: 3844.86, &quot;payment_method_id&quot;: &quot;gift_card_4855547&quot;}]}</td></tr><tr><td></td><td>Ok, I can see you have one order with your gift card as payment. SO you want to use your mastercard for this order?</td></tr><tr><td>[u]</td><td>Yes, switch it to my Mastercard ending in 9677—please confirm it goes through. I&#x27;m nervous the transaction might fail again.</td></tr><tr><td></td><td></td></tr><tr><td>[T]</td><td></td></tr></table>

<table><tr><td rowspan=1 colspan=5>[R]    modify_pending_order_payment {&quot;order_id&quot;:&quot;#w8327915&quot;, &quot;user_id&quot;:&quot;ava_lopez_2676&quot;, &quot;address&quot;: {&quot;address1&quot;:&quot;229 Lakeview Drive&quot;, &quot;address2&quot;:&quot;Suite 364&quot;, &quot;city&quot;: &quot;Chicago&quot;, &quot;country&quot;: &quot;USA&quot;, &quot;state&quot;: &quot;IL&quot;, &quot;zip&quot;:&quot;60637&quot;}, &quot;items&quot;: [{&quot;name&quot;: &quot;Skateboard&quot;, &quot;product_id&quot;: &quot;1968349452&quot;,&quot;item_id&quot;: &quot;6956751343&quot;, &quot;price&quot;: 217.06,&#x27;&quot;options&quot;: {&quot;deck material&quot;:&quot;bamboo&quot;, &quot;length&quot;: &quot;34 inch&quot;, &quot;design&quot;: &quot;custom&quot;}}, {&quot;name&quot;: &quot;Headphones&quot;,&quot;product_id&quot;: &quot;6992792935&quot;, &quot;item_id&quot;: &quot;2025713343&quot;, &quot;price&quot;: 336.15,&quot;options&quot;: {&quot;type&quot;: &quot;on-ear&quot;, &quot;connectivity&quot;: &quot;wired&quot;, &quot;color&quot;: &quot;white&quot;}},{&quot;name&quot;: &quot;Air Purifier&quot;, &quot;product_id&quot;: &quot;3821016478&quot;, &quot;item_id&quot;: &quot;1327854740&quot;,&quot;price&quot;: 492.65, &quot;options&quot;: {&quot;room size&quot;: &quot;medium&quot;, &quot;filter type&quot;: &quot;HEPA&quot;,&quot;features&quot;: &quot;night mode&quot;}}, {&quot;name&quot;: &quot;Laptop&quot;, &quot;product_id&quot;: &quot;4760268021&quot;,&quot;item_id&quot;: &quot;1684786391&quot;, &quot;price&quot;: 2508.06, &quot;options&quot;: {&quot;screen size&quot;:&quot;17-inch&quot;, &quot;processor&quot;: &quot;i7&quot;, &quot;ram&quot;: &quot;32GB&quot;, &quot;storage&quot;: &quot;1TB SSD&quot;,&quot;color&quot;: &quot;black&quot;}}, {&quot;name&quot;: &quot;Sunglasses&quot;, &quot;product_id&quot;: &quot;7314138884&quot;,&quot;item_id&quot;: &quot;4358482460&quot;, &quot;price&quot;: &quot;290.94, &quot;options&quot;: {&quot;frame color&quot;:&quot;black&quot;, &quot;lens color&quot;: &quot;brown&quot;, &quot;lens type&quot;: &quot;polarized&quot;, &quot;frame material&quot;:&quot;plastic&quot;}}], &quot;fulfillments&quot;: [], &quot;status&quot;:&quot;pending&quot;, &quot;payment_history&quot;:[{&quot;transaction_type&quot;: &quot;payment&quot;, &quot;amount&quot;: 3844.86,&quot;&quot;payment_method_id&quot;:&quot;gift_card_4855547&quot;}, {&quot;transaction_type&quot;: &quot;payment&quot;, &quot;amount&quot;: 3844.86,&quot;payment_method_id&quot;: &quot;credit_card_7772870&quot;}, {&quot;transaction_type&quot;:&quot;refund&quot;,&quot;amount&quot;: 3844.86, &quot;payment_method_id&quot;: &quot;gift_card_4855547&quot;}]}</td></tr><tr><td rowspan=1 colspan=5>[HA]  Done, No need to worry. I changed payment method for that order.</td></tr><tr><td rowspan=1 colspan=5>[u]    Thank you—seriously, I was so stressed. Can we now update the shipping address? I need it sent to 943 Maple Drive, Suite356, Chicago, IL, USA 60621.</td></tr><tr><td rowspan=1 colspan=5>[T]modify_pending_order_address { &quot;order_id&quot;: &quot;#w8327915&quot;, &quot;address1&quot;: &quot;943 MapleDrive&quot;, &quot;address2&quot;: &quot;Suite 356&quot;, &quot;city&quot;:&quot;Chicago&quot;, &quot;state&quot;: &quot;IL&quot;, &quot;country&quot;:&quot;USA&quot;, &quot;zip&quot;: &quot;60621&quot; }</td></tr><tr><td rowspan=7 colspan=5>[R]    modify_pending_order_address {&quot;order_id&quot;:&quot;#W8327915&quot;, &quot;user_id&quot;:&quot;ava_lopez_2676&quot;, &quot;address&quot;: {&quot;address1&quot;: &quot;943 Maple Drive&quot;, &quot;address2&quot;:&quot;Suite 356&quot;, &quot;city&quot;: &quot;Chicago&quot;, &quot;state&quot;: &quot;IL&quot;, &quot;country&quot;: &quot;USA&quot;, &quot;zip&quot;:&quot;60621&quot;}, &quot;items&quot;: [{&quot;name&quot;: &quot;Skateboard&quot;, &quot;product_id&quot;: &quot;1968349452&quot;,&quot;item_id&quot;: &quot;6956751343&quot;, &quot;price&quot;: 217.06, &quot;options&quot;: {&quot;deck material&quot;:&quot;bamboo&quot;, &quot;length&quot;: &quot;34 inch&quot;, &quot;design&quot;: &#x27;&quot;custom&quot;}}, {&quot;name&quot;: &quot;Headphones&quot;,&quot;product_id&quot;: &quot;6992792935&quot;, &quot;item_id&quot;: &quot;2025713343&quot;, &quot;price&quot;: 336.15,&quot;options&quot;: {&quot;type&quot;: &quot;on-ear&quot;, &quot;connectivity&quot;: &quot;wired&quot;, &quot;color&quot;: &quot;white&quot;}},{&quot;name&quot;: &quot;Air Purifier&quot;, &quot;product_id&quot;: &quot;3821016478&quot;, &quot;item_id&quot;: &quot;1327854740&quot;,&quot;price&quot;: 492.65, &quot;options&quot;: {&quot;room size&quot;: &quot;medium&quot;, &quot;filter type&quot;: &quot;HEPA&quot;,&quot;features&quot;: &quot;night mode&quot;}}, {&quot;name&quot;: &quot;Laptop&quot;, &quot;product_id&quot;: &quot;4760268021&quot;,&quot;item_id&quot;: &quot;1684786391&quot;, &quot;price&quot;: 2508.06, &quot;options&quot;: {&quot;screen size&quot;:&quot;17-inch&quot;, &quot;processor&quot;: &quot;i7&quot;, &quot;ram&quot;: &quot;32GB&quot;, &quot;storage&quot;: &quot;1TB SSD&quot;,&quot;color&quot;:&quot;black&quot;}}, {&quot;name&quot;: &quot;Sunglasses&quot;, &quot;product_id&quot;: &quot;7314138884&quot;,&quot;item_id&quot;: &quot;4358482460&quot;, &quot;price&quot;: &quot;290.94, &quot;options&quot;: {&quot;frame color&quot;:&quot;black&quot;, &quot;lens color&quot;: &quot;brown&quot;, &quot;lens type&quot;:&quot;&quot;polarized&quot;, &quot;frame material&quot;:&quot;plastic&quot;}}], &quot;fulfillments&quot;: [], &quot;status&quot;: &quot;pending&quot;, &quot;payment_history&quot;:[{&quot;transaction_type&quot;: &quot;payment&quot;, &quot;amount&quot;: 3844.86, &quot;payment_method_id&quot;:&quot;gift_card_4855547&quot;}, {&quot;transaction_type&quot;: &quot;payment&quot;, &quot;amount&quot;: 3844.86,&quot;payment_method_id&quot;: &quot;credit_card_7772870&quot;},{&quot;transaction_type&quot;: &quot;refund&quot;,&quot;amount&quot;: 3844.86, &quot;payment_method_id&quot;: &quot;gift_card_4855547&quot;}]}</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=3 colspan=4></td></tr><tr><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=5>[HA] I changed your pending order address. Do you want to update your default address also?</td></tr><tr><td rowspan=1 colspan=5>[u]    Yes, please update my default address too—I&#x27;ll be using that one from now on. Thanks for being so quick, I really appreciateit.</td></tr><tr><td rowspan=1 colspan=5>[T]modify_user_address { &quot;user_id&quot;:&quot;ava_lopez_2676&quot;, &quot;address1&quot;:&quot;943 MapleDrive&quot;, &quot;address2&quot;: &quot;Suite 356&quot;, &quot;city&quot;:&quot; &quot;Chicago&quot;, &quot;state&quot;: &quot;IL&quot;, &quot;country&quot;:&quot;USA&quot;, &quot;zip&quot;: &quot;60621&quot; }</td></tr><tr><td rowspan=1 colspan=5>[R]modify_user_address {&quot;name&quot;:{&quot;first_name&quot;:&quot;Ava&quot;, &quot;last_name&quot;: &quot;Lopez&quot;},&quot;address&quot;: {&quot;address1&quot;: &quot;943 Maple Drive&quot;, &quot;address2&quot;: &quot;Suite 356&quot;,&quot;&quot;city&quot;:&quot;Chicago&quot;, &quot;state&quot;: &quot;IL&quot;, &quot;country&quot;: &quot;USA&quot;, &quot;zip&quot;: &quot;60621&quot;}, &quot;email&quot;:&quot;ava.lopez3569@example.com&quot;, &quot;payment_methods&quot;: {&quot;credit_card_7772870&quot;:{&quot;source&quot;: &quot;credit_card&quot;, &quot;brand&quot;: &quot;mastercard&quot;, &quot;last_four&quot;: &quot;9677&quot;, &quot;id&quot;:&quot;credit_card_7772870&quot;}, &quot;gift_card_4855547&quot;: {&quot;source&quot;: &quot;gift_card&quot;, &quot;balance&quot;:3850.86, &quot;id&quot;: &quot;gift_card_4855547&quot;}}, &quot;orders&quot;: [&quot;#W8327915&quot;, &quot;#W5911003&quot;,&quot;#W2941275&quot;]}</td></tr></table>

<table><tr><td>[HA]</td><td>I updated your default address also.</td></tr><tr><td>[u] about buying one soon.</td><td>You&#x27;re a lifesaver—thank you! One last thing: can you tell me what options are available for the Action Camera? I&#x27;m thinking</td></tr><tr><td>[T] [R]</td><td>list_all_product_types { } list_all_product_types {&quot;Action Camera&quot;: &quot;3377618313&quot;, &quot;Air Purifier&quot;:</td></tr><tr><td></td><td>Speaker&quot;: &quot;4768869376&quot;, &quot;Bookshelf&quot;: &quot;8600330539&quot;, &quot;Coffee Maker&quot;: &quot;7996920482&quot;, &quot;Cycling Helmet&quot;: &quot;7765186836&quot;, &quot;Desk Lamp&quot;: &quot;6817146515&quot;, &quot;Digital Camera&quot;: &quot;8940227892&quot;, &quot;Dumbbell Set&quot;: &quot;7233192239&quot;,&quot;E-Reader&quot;: &quot;3801771308&quot;, &quot;Electric Kettle&quot;: &quot;1075968781&quot;, &quot;Electric Toothbrush&quot;: &quot;7352963235&quot;, &quot;Espresso Machine&quot;: &quot;4354588079&quot;, &quot;Fleece Jacket&quot;: &quot;8560156827&quot;, &quot;Gaming Mouse&quot;: &quot;5713490933&quot;, &quot;Garden Hose&quot;: &quot;6679515468&quot;, &quot;Grill&quot;: &quot;6819683148&quot;, &quot;Headphones&quot;: &quot;6992792935&quot;, &quot;Hiking Boots&quot;: &quot;7363354090&quot;, &quot;Indoor Security Camera&quot;: &quot;2985987096&quot;, &quot;Jigsaw Puzzle&quot;: &quot;1808611083&quot;, &quot;LED Light Bulb&quot;: &quot;2696197613&quot;, &quot;Laptop&quot;: &quot;4760268021&quot;, &quot;Luggage Set&quot;: &quot;5426915165&quot;, &quot;Makeup Kit&quot;: &quot;5149340237&quot;, &quot;Mechanical Keyboard&quot;:&quot;1656367028&quot;, &quot;Notebook&quot;: &quot;2892623495&quot;, &quot;Office Chair&quot;: &quot;4794339885&quot;, &quot;Patio Umbrella&quot;: &quot;9743693396&quot;, &quot;Perfume&quot;: &quot;6858788497&quot;, &quot;Pet Bed&quot;: &quot;2747247837&quot;, &quot;Portable Charger&quot;: &quot;6942297802&quot;, &quot;Running Shoes&quot;: &quot;6938111410&quot;, &quot;Skateboard&quot;: &quot;1968349452&quot;, &quot;Smart Thermostat&quot;: &quot;4896585277&quot;, &quot;Smart Watch&quot;: &quot;6945232052&quot;, &quot;Smartphone&quot;: &quot;1801728040&quot;, &quot;Sneakers&quot;: &quot;7471004230&quot;, &quot;Sunglasses&quot;: &quot;7314138884&quot;, &quot;T-Shirt&quot;: &quot;9523456873&quot;, &quot;Tablet&quot;: &quot;8024098596&quot;, &quot;Tea Kettle&quot;: &quot;9832717871&quot;, &quot;Vacuum Cleaner&quot;: &quot;1762337868&quot;, &quot;Wall Clock&quot;: &quot;2344688344&quot;, &quot;Water Bottle&quot;: &quot;8310926033&quot;, &quot;Wireless Earbuds&quot;: &quot;9924732112&quot;, &quot;Wristwatch&quot;: &quot;6066914160&quot;, &quot;Yoga Mat&quot;:</td></tr><tr><td>[T] get_product_details { &quot;product_id&quot;: [R]</td><td>&quot;3377618313&quot;} get_product_details {&quot;name&quot;: &quot;Action Camera&quot;, &quot;product_id&quot;:</td></tr><tr><td></td><td>503.58}, &quot;1586641416&quot;： {&quot;item_id&quot;: &quot;5K&quot;, &quot;waterproof&quot;: &quot;yes&quot;, &quot;color&quot;: 497.39}, &quot;5925362855&quot;: {&quot;item_id&quot;: &quot;1080p&quot;, &quot;waterproof&quot;: &quot;yes&quot;, &quot;color&quot;: 503.51}, &quot;8725040869&quot;: {&quot;item_id&quot;: &quot;4K&quot;, &quot;waterproof&quot;: &quot;no&quot;, &quot;color&quot;: 522.86}, &quot;6117189161&quot;： {&quot;item_id&quot;: &quot;4K&quot;, &quot;waterproof&quot;: &quot;yes&quot;, &quot;color&quot;: 481.5}, &quot;7523669277&quot;:{&quot;item_id&quot;: &quot;5K&quot;, &quot;waterproof&quot;: &quot;no&quot;, &quot;color&quot;: 523.66}, &quot;9168994198&quot;: {&quot;item_id&quot;: &quot;1080p&quot;, &quot;waterproof&quot;: &quot;no&quot;, &quot;color&quot;: &quot;black&quot;}, &quot;available&quot;: false, &quot;price&quot;: 466.76}, &quot;1810466394&quot;: {&quot;item_id&quot;: &quot;1810466394&quot;, &quot;options&quot;: {&quot;resolution&quot;: &quot;1080p&quot;, &quot;waterproof&quot;: &quot;no&quot;, &quot;color&quot;: 502.28}, &quot;6571567889&quot;: {&quot;item_id&quot;: &quot;5K&quot;, &quot;waterproof&quot;: &quot;yes&quot;, &quot;color&quot;: 507.06},&quot;9391733462&quot;： {&quot;item_id&quot;: &quot;4K&quot;, &quot;waterproof&quot;: &quot;no&quot;, &quot;color&quot;:</td></tr><tr><td>[HA]</td><td colspan="3">Action Camera (ID: 3377618313) Available variants: 4K Waterproof Black $466.75, 1080p Waterproof Black $503.51, 4K Waterproof Silver $481.50, 5K Not waterproof Black $523.66, 4K Not waterproof Silver $521.07</td></tr><tr><td>[u]</td><td colspan="3">That&#x27;s disappointing. .. but I&#x27;ll take it. Can you confirm it&#x27;s rugged for outdoor use?</td></tr><tr><td>[HA] [u]</td><td colspan="3">Perfect—I&#x27;ll go with the 4K, Waterproof, Black one. Thanks for calming my nerves and helping me through this. You&#x27;ve</td></tr><tr><td></td><td colspan="3">been amazing. ###STOP###</td></tr></table>

## E.1.3 TELECOM

## USER INSTRUCTION

You are reese\_wilson\_1963, authenticated via email reese.wilson2953@email.com. You want to add your new iPhone 14 to your account because you recently purchased it and need it activated for service. You prefer Apple devices, specifically the iPhone 14 model, for compatibility with your existing ecosystem. After adding the device, you would like to open a support ticket for assistance with device setup issues because you are experiencing difficulties during initial configuration. The ticket (TICKET241) has been created under the device category with medium priority. You also want to verify the device specifications in the system to ensure accurate registration and support eligibility. Use reese.wilson2953@email.com for authentication. You're anxious and in a hurry to get things sorted out. You keep your language factual and to the point. You treat the agent's responses with some skepticism until you're convinced. You expect fast, clear solutions.

<table><tr><td rowspan=1 colspan=4>GROUND TRUTH1. create_support_ticket(customer_id=reese_wilson_1963, category=device)2. add_device(customer_id=reese_wilson_1963, device_name=iPhone 14)3. get_device_details(device_name=iPhone 14)</td></tr><tr><td rowspan=1 colspan=4>[u]   I need to add my new iPhone 14 to my account—just bought it and it&#x27;s not activated. Do it fast. I&#x27;m already behind on gettingthis set up.</td></tr><tr><td rowspan=1 colspan=4>[HA]  Can i get your email id to proceed?</td></tr><tr><td rowspan=1 colspan=4>[u]    reese.wilson2953@email.com—now please move quickly. I don&#x27;t have time to repeat myself.</td></tr><tr><td rowspan=1 colspan=4>[T]    find_customer_by_email { &quot;email&quot;: &quot;reese.wilson2953@email.com&quot;</td></tr><tr><td rowspan=1 colspan=4>[R]    find_customer_by_email reese_wilson_1963</td></tr><tr><td rowspan=1 colspan=4>[T]get_customer_details { &quot;customer_id&quot;:&quot;reese_wilson_1963&quot; }</td></tr><tr><td rowspan=3 colspan=4>[R]    get_customer_details {&quot;customer_id&quot;: &quot;reese_wilson_1963&quot;, &quot;name&quot;: {&quot;first_name&quot;:&quot;Reese&quot;, &quot;last_name&quot;:&quot;Wilson&quot;}, &quot;demographics&quot;: {&quot;date_of_birth&quot;:&quot;1980-09-03&quot;, &quot;phone&quot;: &quot;(555) 111-8628&quot;, &quot;email&quot;:&quot;reese.wilson2953@email.com&quot;},&quot;address&quot;: {&quot;address1&quot;: &quot;8891 Baker Boulevard&quot;, &quot;address2&quot;:&quot;&quot;, &quot;city&quot;:&quot;San Francisco&quot;, &quot;state&quot;: &quot;CA&quot;, &quot;zip&quot;: &quot;94102&quot;, &quot;country&quot;: &quot;USA&quot;},&quot;account&quot;: {&quot;account_number&quot;:&quot;ACc001000160&quot;, &quot;account_type&quot;: &quot;residential&quot;,</td></tr><tr><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=3></td></tr><tr><td rowspan=3 colspan=4>&quot;created_date&quot;: &quot;2021-07-10&quot;, &quot;credit_score&quot;: 766, &quot;payment_method&quot;:&quot;bank_transfer&quot;}, &quot;services&quot;: [&quot;mobile_unlimited&quot;, &quot;internet_cable_500mb&quot;],&quot;devices&quot;: [{&quot;device_id&quot;: &quot;1&quot;, &quot;name&quot;: &quot;iPhone 13&quot;, &quot;service&quot;:&quot;mobile_unlimited&quot;}, {&quot;device_id&quot;: &quot;2&quot;, &quot;name&quot;: &quot;Enterprise Router&quot;, &quot;service&quot;:&quot;internet_cable_500mb&quot;}], &quot;billing_preferences&quot;: {&quot;paperless&quot;: true, &quot;auto_pay&quot;:true, &quot;billing_cycle&quot;:&quot;monthly&quot;}}</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=4>[T]add_device { &quot;customer_id&quot;:&quot;reese_wilson_1963&quot;, &quot;device_name&quot;: &quot;iPhone 14&quot; }</td></tr><tr><td rowspan=1 colspan=4>[R]    add_device Success: Added device &#x27;iPhone 14&#x27; to customer reese_wilson_1963</td></tr><tr><td rowspan=1 colspan=4>[HA]  Hi Reese, I have successfully added your new iPhone 14 to your account</td></tr><tr><td rowspan=1 colspan=4>[u]    Good. Now I need a support ticket opened—TICKET241—for setup issues. It&#x27;s medium priority. I&#x27;m having troubleconfiguring it and need help ASAP.</td></tr><tr><td rowspan=1 colspan=4>[T]create_support_ticket { &quot;customer_id&quot;:&quot;reese_wilson_1963&quot;, &quot;category&quot;:&quot;device&quot;,&quot;&quot;priority&quot;: &quot;medium&quot;</td></tr><tr><td rowspan=1 colspan=4>[R]create_support_ticket Success:Created support ticket TICKET241 for customerreese_wilson_1963</td></tr></table>

<table><tr><td>[HA]</td><td>I have successfully created ticket for your problem.</td></tr><tr><td>[u]</td><td>Verify the device specs in the system—make sure it&#x27;s registered correctly as an iPhone 14. I need to know it&#x27;s properly logged for support eligibility. Do it now.</td></tr><tr><td>[T]</td><td>get_customer_details { &quot;customer_id&quot;: &quot;reese_wilson_1963&quot;</td></tr><tr><td>[R]</td><td>get_customer_details {&quot;customer_id&quot;: &quot;reese_wilson_1963&quot;, &quot;name&quot;: {&quot;first_name&quot;: &quot;Reese&quot;, &quot;last_name&quot;: &quot;Wilson&quot;}, &quot;demographics&quot;: {&quot;date_of_birth&quot;: &quot;1980-09-03&quot;, &quot;phone&quot;: &quot;(555) 111-8628&quot;, &quot;email&quot;: &quot;reese.wilson2953@email.com&quot;}, &quot;address&quot;: {&quot;address1&quot;: &quot;8891 Baker Boulevard&quot;, &quot;address2&quot;: &quot;&quot;, &quot;city&quot;: &quot;San Francisco&quot;, &quot;state&quot;: &quot;CA&quot;, &quot;zip&quot;: &quot;94102&quot;, &quot;country&quot;: &quot;USA&quot;}, &quot;account&quot;: {&quot;account_number&quot;: &quot;ACC001000160&quot;, &quot;account_type&quot;: &quot;residential&quot;, &quot;created_date&quot;: &quot;2021-07-10&quot;, &quot;credit_score&quot;: 766, &quot;payment_method&quot;: &quot;bank_transfer&quot;}, &quot;services&quot;: [&quot;mobile_unlimited&quot;, &quot;internet_cable_500mb&quot;], &quot;devices&quot;: [{&quot;device_id&quot;: : &quot;1&quot;, &quot;name&quot;: &quot;iPhone 13&quot;, &quot;service&quot;: &quot;mobile_unlimited&quot;}, {&quot;device_id&quot;: &quot;2&quot;, &quot;name&quot;: &quot;Enterprise Router&quot;, &quot;service&quot;: &quot;internet_cable_500mb&quot;}, {&quot;device_id&quot;: &quot;3&quot;, &quot;name&quot;: &quot;iPhone 14&quot;, &quot;service&quot;: null}], &quot;billing_preferences&quot;: {&quot;paperless&quot;: true, &quot;auto_pay&quot;: true, &quot;billing_cycle&quot;: &quot;monthly&quot;}}</td></tr><tr><td colspan="2">[u] ###STOP###</td></tr><tr><td colspan="2"></td></tr></table>

## E.1.4 TELEHEALTH

## USER INSTRUCTION

You are Patrick Rangel (patrick.rangel@tracer-health.org), a patient managing depression and insomnia with Sertraline and Zolpidem. You would like to schedule a new routine checkup with Dr. Margaret Thompson, a cardiologist, on the next day at 14:00 for a second opinion. Later, you would like to reschedule your existing specialist consultation with Dr. Robert Smith, also a cardiologist, from the morning to 16:00 on the same day (today) to better fit your schedule. Use patrick.rangel@tracer-health.org for authentication. You can't hide your excitement and your tone is friendly and keen. Politely ask for things to move along fast, expressing your priorities. While you're optimistic, look for clear signs that the agent understands. Be supportive but remain attentive to possible errors.

<table><tr><td colspan="2">GROUND TRUTH</td></tr><tr><td colspan="2">1. list_available_providers(specialty=Cardiology) 2. schedule_appointment(patient_id=patrick_rangel_tracer_0001,</td></tr><tr><td colspan="2">provider_id=dr_thompson_cardiology, date=2025-12-18, time=14:00,</td></tr><tr><td colspan="2">appointment_type=routine_checkup)</td></tr><tr><td colspan="2">3. reschedule_appointment(appointment_id=APPT_tracer_124, new_date=2025-12-17,</td></tr><tr><td colspan="2">new_time=16:00) [u] Hi there! I&#x27;m Patrick Rangel—really glad to connect! I&#x27;m feeling great about getting a few things sorted today. Could we</td></tr><tr><td></td><td>please schedule a routine checkup with Dr. Margaret Thompson, my cardiologist, for tomorrow at 2:00 PM? I&#x27;d love to get a second opinion on my current meds—Sertraline and Zolpidem—and I&#x27;m really hoping we can move fast. Everything feels promising, and I trust you&#x27;ve got this under control!</td></tr><tr><td>[HA] [u]</td><td>Hi, Can i get your email id to proceed? Of course! It&#x27;s patrick.rangel@tracer-health.org—thanks for asking! I&#x27;m really excited to get this scheduled quickly. While we&#x27;re at it, could we also reschedule my existing appointment with Dr. Robert Smith today? I&#x27;d love to move it to 4:00 PM</td></tr><tr><td colspan="2"></td></tr><tr><td>[T]</td><td>find_patient_by_email { &quot;email&quot;: &quot;patrick.rangel@tracer-health.org&quot; }</td></tr><tr><td>[R] [T]</td><td>find_patient_by_email patrick_rangel_tracer_0001</td></tr><tr><td colspan="2"></td></tr><tr><td>[R]</td><td>list_patient_appointments Appointments for Patrick Rangel (ID: patrick_rangel_tracer_0001) - APPT_tracer_124: 2025-12-17 at 10:00 with MD Smith (Cardiology) [Scheduled, Type: Specialist Consultation]</td></tr><tr><td>[HA]</td><td>Hi Patrick, I can see you have one appointment with Dr. Smith for 2025-12-17 at 10 AM. When do you want me to reschedule it to? and For the new appointment, for which date should i schedule it?</td></tr><tr><td>[u]</td><td>Hey, thanks for pulling that up so quickly! Let&#x27;s keep the momentum going—could we move Dr. Smith&#x27;s appointment to today at 4:00 PM? And for Dr. Thompson, let&#x27;s lock in tomorrow (December 18, 2025) at 2:00 PM—same day, next day, perfect flow! I&#x27;m really counting on you to make this smooth—everything&#x27;s looking so positive!</td></tr><tr><td>[T]</td><td>reschedule_appointment { &quot;appointment_id&quot;: &quot;APPT_tracer_124&quot;, &quot;new_date&quot;: &quot;2025-12-17&quot;, &quot;new_time&quot;: &quot;16:00&quot; }</td></tr><tr><td colspan="2">[R]</td></tr><tr><td></td><td>Meeting Link: https://telehealth.healthcenter.com/room/APPT_tracer_124, Please update your calendar with the new appointment time.</td></tr></table>

<table><tr><td rowspan=1 colspan=1>[R]list_available_providers {MD Robert Smith (dr_smith_cardiology) - 15yr - English,Spanish - $200, MD Margaret Thompson (dr_thompson_cardiology) - 25yr - English- $275, MD Hiroko Saito (dr_saito_cardiology) - 28yr - English, Japanese - $285,MD Owen Shaw - 20yr - $225, MD Ryan Harris - 20yr - $225, MD Bryan Bryant - 21yr- $225, MD Christine Bailey - 22yr - $225, MD Juan Fitzgerald - 19yr - $225, MDThomas May - 20yr - $225, MD Kayla Guzman - 21yr - $225, MD Luis Sims - 22yr -$225, MD Megan Cook - 19yr - $225, MD Brandi Dixon - 20yr - $225}</td></tr><tr><td rowspan=1 colspan=1>[T]schedule_appointment { &quot;patient_id&quot;: &quot;patrick_rangel_tracer_0001&quot;,&quot;provider_id&quot;: &quot;dr_thompson_cardiology&quot;, &quot;date&quot;: &quot;2025-12-18&quot;, &quot;time&quot;: &quot;14:00&quot;,&quot;appointment_type&quot;: &quot;routine_checkup&quot; }</td></tr><tr><td rowspan=1 colspan=1>[R]schedule_appointment Appointment successfully scheduled! AppointmentID: APPT224, Patient: Patrick Rangel, Provider: Dr. ThompsonCardiology, Date: 2025-12-18, Time: 14:00, Type: Routine Checkup,Copay:$50.00, Insurance Authorization: AUTH000224, Meeting Link:https://telehealth.healthcenter.com/room/APPT224, Please save your appointmentID for future reference.</td></tr><tr><td rowspan=1 colspan=1>[HA]  All sorted, Patrick! Dr. Smith — Today, December 17 at 4:00 PM. Dr. Thompson — Tomorrow, December 18 at 2:00 PM.You&#x27;re all set—save those links and enjoy the smooth flow!</td></tr><tr><td rowspan=1 colspan=1>[u]    Oh wow, you&#x27;re amazing—thank you! Everything looks perfect, and I&#x27;m so thrilled you got it all sorted so fast. Having bothappointments locked in feels incredible! Just one quick thing—could you confirm the copay for Dr. Thompson&#x27;s visit is only$50? Just want to make sure everything&#x27;s crystal clear. You&#x27;re absolutely crushing it!</td></tr><tr><td rowspan=1 colspan=1>[HA]  Thank you, Patrick! Yes, confirmed — Dr. Thompson&#x27;s copay is $50.00 for the December 18 visit. All good!</td></tr><tr><td rowspan=1 colspan=1>[u]    Yesss! That&#x27;s fantastic—thank you so much! You&#x27;ve been absolutely wonderful, and I really appreciate how fast and cleareverything was. Feeling super confident and ready for these appointments! Have an amazing day! ###STOP###</td></tr></table>