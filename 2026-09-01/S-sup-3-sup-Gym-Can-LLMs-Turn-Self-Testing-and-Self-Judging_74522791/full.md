# S<sup>3</sup>Gym: Can LLMs Turn Self-Testing and Self-Judging into Self-Improvement?

<sup>1</sup>ByteDance Seed, <sup>2</sup>M-A-P, <sup>3</sup>TokenWave.AI

Full author list in Contributions

## Abstract

Large language models (LLMs) increasingly interact with external environments and accumulate substantial behavioral experience, yet existing agent benchmarks largely evaluate them as fixed policies. It therefore remains unclear whether an agent can actively test its behavior, judge the resulting experience, and use that experience to improve future decisions. We introduce S<sup>3</sup>Gym, an interactive benchmark for evaluating LLM self-improvement through three coupled capabilities: Self-Testing, Self-Judging, and Self-Improvement. S<sup>3</sup>Gym separates permissive exploration from strict held-out evaluation and instantiates this protocol in seven text-based games with executable environment verifiers. We evaluate three pathways for incorporating interaction experience: direct History ICL, score-conditioned Summary Memory, and parameter Training.

Our experiments reveal that self-improvement is neither automatic nor uniform. Context-level experience improves performance for several model–game pairs, but the most efective pathway depends strongly on the task structure: summaries are beneficial when experience can be compressed into reusable strategic rules, yet often underperform raw history when success depends on precise, state-contingent information. Parameter training produces substantial gains on some tasks, but also exhibits unstable improvement and severe negative transfer on others. These findings show that recognizing successful actions is insuficient; agents must also transform feedback into executable and transferable policies. S<sup>3</sup>Gym provides a unified framework for diagnosing this process and identifying the bottlenecks that prevent agents from translating interaction experience into reliable self-improvement.

Date: September 1, 2026

Project Page: https://self-developing-agents.github.io/

## 1 Introduction

Large language models (LLMs) are increasingly deployed as autonomous agents that invoke tools, execute code, and interact with external environments through repeated observation–reasoning–action–feedback loops. Such interaction is central to applications including automated research, software engineering, and open-ended decision-making, where agents continually accumulate behavioral experience. Existing benchmarks evaluate multi-turn decision-making, environment exploration, and task completion [15, 16], but they typically treat the evaluated model as a fixed policy. As a result, they primarily answer a static question—how capable is the model at the time of evaluation?—while ofering limited insight into whether the model can use its own past interactions to improve future behavior. We refer to this experience-driven capability as Self-Improvement.

Theories of experiential learning and reflective inquiry suggest that experience becomes useful only when it is actively examined, abstracted, and tested again. Dewey and Kolb emphasize the transformation of experience into knowledge through reflection and reapplication [8, 14], while Popper characterizes progress as the iterative exposure of conjectures to tests that can reveal error [21]. The same principle applies to LLM agents: collecting trajectories alone does not guarantee learning. An agent must generate informative evidence, interpret the causes of success and failure, and convert those interpretations into improved behavior. We therefore decompose experience-driven learning into three interdependent capabilities: Self-Testing, in which the agent explores strategies and gathers diagnostic evidence; Self-Judging, in which it evaluates actions, outcomes, and their potential reusability; and Self-Improvement, in which the resulting experience alters future decisions [2, 23].

Experience that has been tested and judged can be incorporated at diferent levels of persistence. At the shortterm in-context learning level, an agent directly conditions future decisions on previous observations, actions, and scores. At the intermediate external-memory level, long trajectories are compressed into natural-language rules, failure patterns, or reusable skills, as in Reflexion and Voyager [26, 30]. At the long-term training level, selected or corrected trajectories are internalized through parameter updates, following the broader direction explored by STaR, ReST, and Re-ReST [9, 11, 37]. These mechanisms make diferent trade-ofs: raw histories preserve detailed evidence but consume context, summaries provide compact abstractions but depend on judgment quality, and training ofers persistent internalization but may also amplify incorrectly judged experience. A unified evaluation protocol is therefore needed to compare these pathways under the same interaction and testing conditions.

To address this need, we introduce $\mathsf { \pmb { S } } ^ { 3 } \mathsf { G y m }$ , short for Self-Testing, Self-Judging, and Self-Improvement Gym, an interactive benchmark for evaluating whether LLMs can become more capable through their own environmental experience. Each evaluation consists of an Exploration Phase and an Evaluation Phase, reflecting the distinction between searching for new strategies and exploiting acquired knowledge [17]. During exploration, the agent interacts with relatively permissive environment configurations, generating successful, unsuccessful, and partially successful trajectories. Importantly, $\mathrm { S ^ { 3 } G y m }$ explicitly separates model-generated self-judgments from environment-verifiable outcomes: self-judgments determine how interaction experience is organized and reused, whereas environment scores measure the resulting behavioral performance. The updated agent is then evaluated on stricter and held-out configurations, providing a direct test of whether experience acquired during exploration transfers to new conditions and whether errors in Self-Judging become a bottleneck to subsequent Self-Improvement.

We instantiate $\mathrm { S ^ { 3 } G y m }$ (Figure 1) with text-based games, following the game-based evaluation paradigm of $\operatorname { K O R G y m }$ [25]. Whereas KORGym evaluates LLM reasoning through diverse interactive games, $\mathrm { S ^ { 3 } G y m }$ uses such environments as controlled testbeds for experience-driven self-improvement. Their multi-turn interaction, partial observability, delayed consequences, and long-horizon decision-making provide rich behavioral trajectories, while programmatically verifiable rules enable reproducible evaluation with objective step-level or terminal scores. Related environments, including TextWorld, ALFWorld, and ScienceWorld, have similarly demonstrated the utility of textual interaction for evaluating planning, embodied reasoning, and scientific problem solving [5, 27, 31]. $\mathrm { S ^ { 3 } G y m }$ includes seven games—Chess, Minesweeper, Nullify, Tetris, Snake, $\mathrm { P V Z , }$ , and Trust Evolution—covering latent-rule induction, constraint satisfaction, numerical transforma tion, spatial planning, resource allocation, survival, and multi-agent strategy. Each game provides related but distinct exploration and evaluation configurations, requiring agents to extract transferable knowledge from experience rather than memorize isolated actions or seeds.

![](images/5c5624f21c1f622a7eb5d22cd3b073a36a43ab6afbe7215f0526bd0ab00fc6f7.jpg)  
Figure 1 Conceptual overview of experience-driven improvement in humans and LLM agents. Both processes involve generating experience through Self-Testing, interpreting it through Self-Judging, and incorporating it through reusable memory or learning mechanisms.

Our main contributions are summarized as follows:

• A unified formulation of experience-driven self-improvement. We formalize LLM self-improvement as an iterative process that connects Self-Testing, Self-Judging, and subsequent behavioral change, and introduce S<sup>3</sup>Gym with separate exploration and held-out evaluation phases.

• A systematic comparison of experience-integration pathways. We evaluate three mechanisms for incorporating interaction experience—History ICL, Summary Memory, and parameter Training—under a unified protocol.

• An explicit evaluation of Self-Judging. By comparing model-generated judgments with verifier-computed outcomes, we assess whether agents can identify useful and harmful experience and translate those judgments into actionable improvement directions.

• Comprehensive empirical and diagnostic analyses. We evaluate representative LLMs across seven interactive environments and analyze improvement dynamics, judgment reliability, summary efectiveness, cross-game consistency, and common failure modes.

## 2 Background

Most agent benchmarks begin after the problem has already been made executable: the task is specified, the reward or judge is defined, and the execution scafold is fixed. This setting is necessary for controlled comparison, but it hides the work that dominates real deployment. In industry, that work is distributed across several roles—including solutions architects, applied engineers, and platform engineers—but its most visible recent crystallization is the forward-deployed engineer (FDE), a title popularized by Palantir and since adopted by frontier-model companies [18, 20]. An FDE is embedded in the deployment environment and turns a general-purpose model into a system that works with a customer’s data formats, workflows, and operational constraints. The role’s success criterion is not a demonstration or a benchmark score, but whether the deployed system is genuinely used, continues to work, and improves in response to failures. The growth of

FDE roles across frontier-model and data-platform companies reflects a simple fact: a capable model is not yet a working system [3, 29], and today the gap is largely closed by human engineers.

Viewed from the model’s side, FDE work supplies three pieces of structure that benchmark designers normally presuppose. First, the target is vague: an informal deployment need must be translated into concrete objectives, constraints, and success criteria. Second, the feedback signal may be absent or unreliable: tests, judges, traces, or other validation mechanisms must be constructed before anyone can determine whether the system is improving. Third, the execution system may not exist in a usable form: the tools, context management, state, lifecycle logic, and verification interface through which future tasks will run must be built, adapted, and maintained as requirements change.

S<sup>3</sup>Gym focuses on the second layer: the agent-facing feedback signal and the experience-driven improvement loop that it enables. The benchmark retains an executable environment verifier as external ground truth, but withholds verifier-computed outcomes from the agent during exploration. The agent must instead generate informative evidence through Self-Testing, evaluate that evidence through Self-Judging, and convert it into better future behavior through Self-Improvement. This separation reveals both whether self-judgments agree with actual outcomes and whether judged experience produces transferable improvement. Whereas Aspire studies how broad deployment needs become capability growth and HarnessDev studies how models build and maintain the execution systems that carry them, S<sup>3</sup>Gym isolates the experience-to-improvement loop within an executable environment.

## 3 Related Work

## 3.1 Interactive Agent Benchmarks

Recent progress in large language model agents has motivated the development of interactive benchmarks that evaluate multi-turn decision-making, tool use, planning, and environment interaction. AgentBench [15] evaluates LLM agents across diverse domains, including operating systems, databases, and games, while AgentBoard [16] provides fine-grained analyses of agent progress and failure in multi-step tasks. Text-based environments such as TextWorld [5], ALFWorld [27], and ScienceWorld [31] further ofer controllable testbeds for language-grounded planning, embodied reasoning, and scientific problem solving. Recent game-oriented benchmarks also examine interactive reasoning and strategic behavior in textual or programmatically verifiable environments.

These benchmarks provide rich substrates for evaluating agent behavior, but they generally treat the evaluated model as a fixed policy. Interaction trajectories are primarily used to measure task completion or diagnose failures within a predefined evaluation setting, rather than to determine whether agents can transform their own experience into improved future behavior. S<sup>3</sup>Gym builds on the controllability and objective verification of interactive environments, while shifting the evaluation target from static task competence to experience-driven behavioral improvement.

## 3.2 Experience-Based Agent Improvement

A growing body of work studies how language agents can improve by reusing their own interaction experience. At the context level, Reflexion [26] introduces verbal reinforcement learning, where agents reflect on previous attempts and use textual feedback to guide future decisions. At the external-memory level, Voyager [30] accumulates reusable skills in an evolving library, enabling embodied agents to retain and reuse knowledge acquired during open-ended interaction. Recent memory-based approaches similarly compress trajectories into rules, failure patterns, or reusable strategies that can be retrieved in later episodes: ReasoningBank [19] distills generalizable reasoning strategies from an agent’s self-judged successful and failed experiences, Treeof-Experience [7] organizes accumulated experience into a hierarchy that supports feedback attribution and cross-task transfer, and MetaSkill-Evolve [33] extends skill-file rewriting to the improvement procedure itself through two-timescale meta-skill evolution.

Experience can also be incorporated through parameter updates. STaR [37], ReST [11], and Re-ReST [9] improve language models by generating, filtering, or refining self-produced reasoning trajectories. Self-Rewarding Language Models [36] further leverage model-generated preference signals for iterative optimization.

In interactive environments, RetroAgent [39] trains agents with retrospective dual intrinsic feedback so that experience acquired across episodes can be retrieved and reused rather than only implicitly encoded in parameters, and test-time self-improvement [1] constructs targeted training samples from the agent’s own interaction failures. These studies demonstrate that interaction histories and self-generated supervision can improve model performance. However, they typically investigate a specific improvement mechanism within a particular task or training setting, making it dificult to disentangle whether improvement or failure results from the quality of collected experience, the reliability of experience interpretation, or the mechanism used to incorporate that experience. S<sup>3</sup>Gym provides a unified evaluation protocol that compares diferent experience incorporation strategies, including History ICL, Summary Memory, and parameter training, under the same exploration and held-out evaluation setting.

## 3.3 Evaluating Adaptive and Self-Improving Agents

The reliability of model-generated supervision has been studied extensively in work on LLM-based evaluation. LLM-as-a-Judge [41] and JudgeBench [28] show that language models can provide useful evaluation signals, but their judgments remain vulnerable to biases, calibration errors, and limitations in reasoning. A recent audit of step-level credit assignment in ALFWorld further shows that LLM-judge scores, outcome-conditioned logprob ratios, and the policy’s own confidence all fail to identify causally important steps better than chance when measured against executed replay [38]. Such limitations are particularly important for self-improving agents, where inaccurate judgments may cause harmful experiences to be retained or valuable experiences to be discarded.

Recent benchmarks have started to move beyond static capability evaluation toward adaptive and selfimproving systems. PostTrainBench [22] evaluates automated post-training pipelines that improve model parameters on held-out tasks. SEA-Eval [13] studies long-term agent evolution through sequential task interactions, while SEAGym [40] investigates the evolution of persistent agent harnesses, including prompts, memories, tools, and workflows. PAST-Bench [34] tests whether retained experience actually improves personal agents by running matched task sequences with retained experience turned on and of. ContinualSkillBench [10] measures in-context continual skill learning and finds that gains vary substantially across models and domains, FinEvo-Bench [6] provides a longitudinal benchmark for self-evolving agents in professional financial workflows, Social Gym [12] benchmarks multi-agent social reasoning in games with rule-verifiable outcomes, and AI4AI Bench [4] isolates the design of training algorithms for recursive self-improvement. Beyond benchmarks, recent analyses reveal that self-improvement itself is often unreliable: memory-based methods exhibit high variance and strong sensitivity to task order [35], accumulated skills can contaminate later behavior once a defective skill enters the pool [24], and optimizer-driven agent gains need not compound across continual-learning rounds [32]. These benchmarks and analyses represent important steps toward evaluating adaptive systems, but they focus on diferent improvement artifacts or specific aspects of the adaptation process, and none couples self-testing with explicit self-judging under executable environment verification. Table 1 summarizes the key diferences between existing benchmarks and S<sup>3</sup>Gym.

Unlike existing benchmarks that primarily evaluate evolutionary outcomes or individual adaptation mechanisms, S<sup>3</sup>Gym explicitly studies the process through which agents transform interaction experience into future behavioral improvement. It evaluates multiple experience incorporation strategies within the same environments and interaction budgets, while testing updated agents on disjoint and generally stricter configurations. Moreover, S<sup>3</sup>Gym records both model-generated judgments and executable environment outcomes, enabling analysis of whether improvement failures originate from inaccurate experience evaluation or from an inability to transform correctly identified feedback into transferable behavior.

Table 1 Comparison of adaptive and self-improving language agent benchmarks.
<table><tr><td>Benchmark</td><td>Updated System</td><td>Self-Testing</td><td>Self-Judging</td><td>Experience Use</td><td>Evaluation</td></tr><tr><td>PostTrainBench</td><td>Model</td><td>X</td><td>X</td><td>Parameter training</td><td>Held-out tasks</td></tr><tr><td>SEA-Eval</td><td>Agent</td><td>Partial</td><td>X</td><td>Sequential adaptation</td><td>Long-term evolution</td></tr><tr><td>SEAGym</td><td>Agent harness</td><td>Partial</td><td>X</td><td>Prompt, memory, tool, workflow</td><td>ID/OOD transfer</td></tr><tr><td>PAST-Bench</td><td>Agent memory</td><td>Partial</td><td>X</td><td>updates Retained cross-session</td><td>Matched on/off conditions</td></tr><tr><td>ContinualSkillBench Skill library</td><td></td><td>Partial</td><td>X</td><td>experience In-context skill accumulation</td><td>Cross-task reuse</td></tr><tr><td>FinEvo-Bench</td><td>Agent</td><td>Partial</td><td>X</td><td>Longitudinal experience</td><td>Longitudinal task sequences</td></tr><tr><td>Social Gym</td><td>Agent</td><td>Partial</td><td>X</td><td>Tournament self-play</td><td>Rule-verifiable outcomes</td></tr><tr><td>S³Gym</td><td>Agent behavior</td><td>√</td><td>V</td><td>History ICL, Summary Memory, evaluation Training</td><td>Strict held-out</td></tr></table>

## 4 Method

## 4.1 Overview

S<sup>3</sup>Gym evaluates whether an LLM can improve its future behavior by testing strategies in interactive environments, judging the resulting experience, and reusing the acquired knowledge. One self-improvement cycle contains five stages:

$$
\mathcal { R } _ { t } ^ { ( p ) } = \mathrm { E x p l o r e }  \mathrm { J u d g e }  \mathrm { C o n s o l i d a t e }  \mathrm { U p d a t e } _ { p }  \mathrm { E v a l u a t e } ,\tag{1}
$$

where t is the cycle index and $p \in \{ { \mathrm { H i s t o r y } } $ , Memory, Training} denotes the improvement pathway. During exploration, the agent accumulates trajectories under relatively permissive game configurations. It then judges its own decisions and consolidates the resulting experience into raw history, summary memory, or training examples. Finally, the updated agent is evaluated on stricter and held-out configurations.

![](images/65e3dc501297055f4d5f40a2dfe5246f8c5a9fc8da7987a59b920a9f1cbe3f3b.jpg)  
Figure 2 Overview of $\mathrm { S ^ { 3 } G y m }$ . Exploration trajectories are judged and incorporated through History ICL, Summary Memory, or parameter Training. The updated agent is then evaluated on stricter game configurations.

## 4.2 Interaction Protocol and Formalization

Each game contains multiple independent episodes, and each episode consists of a sequence of interaction steps. For episode x at step i, we denote the current observation by $O _ { x , i } ,$ the within-episode interaction history by $H _ { x , i }$ , the agent action by $a _ { x , i } .$ , the agent’s self-judged immediate score by $s _ { x , i } .$ the observable environment feedback by $F _ { x , i }$ , the verifier-computed step reward by $r _ { x , i }$ , and the final environment score by $y _ { x }$

An episode is initialized as

$$
O _ { x , 0 } = \operatorname { R e s e t } ( { \sec _ { x } } , c _ { x } ) ,\tag{2}
$$

$$
H _ { x , 0 } = \mathcal { O } ,\tag{3}
$$

where $c _ { x }$ denotes the game configuration.

At each step, the agent receives the current observation, the within-episode history, and the pathway-specific experience state $Z _ { t } ^ { ( p ) }$ inherited from previous exploration episodes. Following the actual interaction interface, the model jointly produces an action and a self-judged immediate score:

$$
\begin{array} { r } { \left( a _ { x , i } , s _ { x , i } \right) = \pi _ { \theta _ { t } } \left( O _ { x , i } , H _ { x , i } ; Z _ { t } ^ { ( p ) } \right) , } \end{array}\tag{4}
$$

where $p \in$ {History, Memory, Training}. The score $s _ { x , i }$ represents the model’s estimate of the immediate reward associated with its proposed action under the game rules, rather than a free-form confidence score. In the context-level pathways, $Z _ { t } ^ { ( p ) }$ contains either retained raw history or summarized experience; for parameter training, acquired experience is instead reflected primarily through the updated parameters $\theta _ { t }$

The environment then executes the proposed action and returns the next observation, observable feedback, verifier-computed reward, and termination signal:

$$
\begin{array} { r } { ( O _ { x , i + 1 } , F _ { x , i } , r _ { x , i } , d _ { x , i } ) = \mathrm { E n v } ( O _ { x , i } , a _ { x , i } ) . } \end{array}\tag{5}
$$

Importantly, $F _ { x , i }$ denotes environment information available to the agent for subsequent interaction, whereas $r _ { x , i }$ is retained by the benchmark as an external verification signal. In the main setting, verifier-computed rewards are not exposed to the agent during exploration. This separation allows $\mathrm { S ^ { 3 } G y }$ m to compare the agent’s self-judgment $s _ { x , i }$ against the actual environment reward $r _ { x , i }$ without allowing ground-truth rewards to directly guide subsequent decisions.

The agent-visible transition is then appended to the within-episode history:

$$
H _ { x , i + 1 } = H _ { x , i } \oplus \mathrm { F o r m a t } \left( O _ { x , i } , a _ { x , i } , s _ { x , i } , F _ { x , i } , O _ { x , i + 1 } \right) ,\tag{6}
$$

where ⊕ denotes ordered concatenation. Thus, the interaction history records the agent’s own score together with the state transition, while the verifier reward $r _ { x , i }$ remains benchmark-side information. At the next step, the agent again applies Eq. (4) to the updated observation and history.

When the episode terminates or reaches the maximum number of steps, the environment returns the final verifier-computed score

$$
y _ { x } = \mathrm { S c o r e } ( \tau _ { x } ) ,\tag{7}
$$

where $\tau _ { x }$ denotes the complete executed trajectory of episode x. The step-level reward $r _ { x , i }$ is used to evaluate the reliability of Self-Judging, whereas $y _ { x }$ measures the episode-level behavioral performance used for benchmark evaluation. Neither signal is provided to the agent during exploration in the main setting.

## 4.3 Self-Judging and Experience Consolidation

Self-Judging provides an internal signal for identifying useful and harmful experience when precise environment rewards are unavailable to the agent. Low-scoring trajectory segments indicate possible mistakes or strategies to avoid, whereas high-scoring segments provide candidate strategies to retain.

S<sup>3</sup>Gym organizes the judged experience in two context-level forms.

History Context. Score-annotated trajectories are directly serialized and appended to future contexts:

$$
C _ { t + 1 } = \mathrm { S e r i a l i z e } ( C _ { t } , { \mathcal { T } } _ { t } ^ { \mathrm { e x p } } , S _ { t } ^ { \mathrm { e x p } } ) ,\tag{8}
$$

where $\mathcal { T } _ { t } ^ { \mathrm { e x p } }$ and $S _ { t } ^ { \mathrm { e x p } }$ are the exploration trajectories and their self-judgment scores in cycle t.

Summary Memory. The model compresses the judged trajectories into reusable experience:

$$
M _ { t + 1 } = \mathrm { S u m m a r i z e } ( M _ { t } , { \mathcal { T } } _ { t } ^ { \mathrm { e x p } } , S _ { t } ^ { \mathrm { e x p } } ) .\tag{9}
$$

The summary contains strategies to retain, mistakes to avoid, and concrete directions for the next interaction cycle:

$$
M _ { t + 1 } = \left( R _ { t } ^ { \mathrm { r e t a i n } } , R _ { t } ^ { \mathrm { a v o i d } } , D _ { t } ^ { \mathrm { n e x t } } \right) .\tag{10}
$$

## 4.4 Exploration and Evaluation Phases

Each game provides an exploration distribution $\mathcal { C } _ { g } ^ { \mathrm { e x p } }$ and an evaluation distribution $\mathcal { C } _ { g } ^ { \mathrm { e v a l } }$ . Exploration environments preserve the main task structure but use more permissive conditions, such as smaller state spaces, weaker failure penalties, or additional trial opportunities. Evaluation environments use stricter rules or more complex configurations.

A self-improvement cycle contains $N _ { \mathrm { e x p } }$ exploration episodes followed by $N _ { \mathrm { e v a l } }$ evaluation episodes:

$$
\{ \tau _ { t , x } ^ { \mathrm { e x p } } \} _ { x = 1 } ^ { N _ { \mathrm { e x p } } }  \mathrm { I m p r o v e }  \{ \tau _ { t , x } ^ { \mathrm { e v a l } } \} _ { x = 1 } ^ { N _ { \mathrm { e v a l } } } .\tag{11}
$$

Exploration provides diverse experience for judgment and consolidation, whereas evaluation verifies whether the resulting improvement transfers to stricter conditions. The two phases use disjoint random seeds, and evaluation trajectories are not added to history, memory, or training data.

For pathway p and game $^ { g , }$ evaluation performance after cycle t is

$$
\overline { { Y } } _ { t , g } ^ { ( p ) } = \frac { 1 } { N _ { \mathrm { e v a l } } } \sum _ { x = 1 } ^ { N _ { \mathrm { e v a l } } } y _ { t , g , x } ^ { ( p ) } ,\tag{12}
$$

and improvement over the original model is

$$
\begin{array} { r } { \Delta _ { t , g } ^ { ( p ) } = \overline { { Y } } _ { t , g } ^ { ( p ) } - \overline { { Y } } _ { 0 , g } . } \end{array}\tag{13}
$$

We report both the final evaluation score and $\Delta _ { t , g } ^ { ( p ) }$ . The former measures final capability, while the latter measures how efectively the model benefits from its own interaction history.

## 4.5 Self-Improving Pathways

$\mathrm { S ^ { 3 } G y m }$ evaluates three pathways for incorporating exploration experience.

History ICL. Score-annotated trajectories are directly inserted into the context of later episodes:

$$
Z _ { t + 1 } ^ { ( \mathrm { H i s t o r y } ) } = C _ { t + 1 } .\tag{14}
$$

This pathway preserves detailed interaction evidence but is constrained by context length.

Summary Memory. The agent receives the consolidated summary rather than the complete trajectory:

$$
Z _ { t + 1 } ^ { ( \mathrm { M e m o r y } ) } = M _ { t + 1 } .\tag{15}
$$

This pathway evaluates whether the model can abstract transferable strategies from lengthy interaction histories.

Parameter Training. Exploration trajectories and self-judgments are converted into supervised fine-tuning examples:

$$
\mathcal { D } _ { t } ^ { \mathrm { S F T } } = \mathrm { B u i l d S F T } ( \mathcal { T } _ { t } ^ { \mathrm { e x p } } , S _ { t } ^ { \mathrm { e x p } } ) ,\tag{16}
$$

followed by the parameter update

$$
\theta _ { t + 1 } = \mathrm { S F T } \big ( \theta _ { t } , { \mathcal { D } } _ { t } ^ { \mathrm { S F T } } \big ) .\tag{17}
$$

High-scoring actions are retained as positive examples, while low-scoring actions are filtered or replaced with corrected actions. Training is treated as an auxiliary pathway because it introduces optimization variables absent from context-level improvement.

All pathways start from the same base model and use the same exploration seeds, evaluation seeds, interaction budget, and decoding settings.

## 5 Experiments

## 5.1 Benchmark Games

$\mathrm { S ^ { 3 } G y m }$ contains seven text-based games that cover complementary forms of interactive reasoning, including latent-rule induction, constraint satisfaction, numerical transformation, spatial planning, long-horizon control, resource allocation, and multi-agent strategy. Each game provides an exploration configuration and an evaluation configuration. The exploration configuration is generally more permissive, while the evaluation configuration introduces stricter termination conditions, a larger search space, or denser dynamics. Table 2 summarizes the code-level configurations used in our experiments.

<table><tr><td>Game</td><td>Exploration</td><td>Evaluation</td><td>Max steps</td></tr><tr><td>Chess</td><td>history.</td><td>15 × 15 board, 12 pieces, and a 7-step Same rules as exploration but 22 pieces.</td><td>5</td></tr><tr><td>Minesweeper</td><td>extra lives.</td><td>9 × 9 board with 10–20 mines and two The first mine hit terminates the episode.</td><td>64</td></tr><tr><td>Nullify</td><td>Expressions generated with 3-7 con- struction steps.</td><td>Expressions generated with 5-10 con- struction steps.</td><td>50</td></tr><tr><td>Plants-vs-Zombies Snake</td><td>3 × 7 battlefield.</td><td>5 × 6 battlefield.</td><td>64</td></tr><tr><td></td><td>Collisions and invalid actions are treated as no-ops.</td><td>Collisions and invalid actions terminate the episode.</td><td>64</td></tr><tr><td>Tetris</td><td>10 × 10 board.</td><td>8 × 8 board.</td><td>64</td></tr><tr><td>Trust Evolution</td><td>Opponent sampling is biased toward eas- Opponent strategies are sampled uni- ier strategies.</td><td>formly.</td><td>10</td></tr></table>

Table 2 Exploration and evaluation configurations for the seven S<sup>3</sup>Gym games. Exploration generally uses more permissive rules, while evaluation uses stricter or more dificult configurations.

## 5.2 Experimental Setup

Models. We evaluate seven representative proprietary LLMs for which all seven game results are complete: GPT-4o, GPT-4.1, o3-mini, Gemini-2.5-Flash, Gemini-2.5-Pro, GPT-5.5, and Gemini-3.5-Flash. Models with unfinished checkpoints are excluded from the main table to avoid comparisons based on diferent game coverage.

Context-level pathways. We compare two non-parametric self-improvement pathways. History ICL directly appends the interaction history to the context, whereas Summary Memory compresses previous trajectories into model-generated rules, mistakes, and future directions. The former preserves detailed evidence but requires a larger context budget; the latter imposes an information bottleneck and therefore depends more strongly on the quality of Self-Judging and summarization.

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>History-ICL input limit</td><td>100,000 tokens</td></tr><tr><td>Output generation cap</td><td>14,000 new tokens per response</td></tr><tr><td>Controller interaction cap</td><td>64 steps per episode</td></tr><tr><td>Exploration budget</td><td>30 exploration episodes</td></tr><tr><td>Evaluation interval</td><td>Every 3 exploration episodes</td></tr><tr><td>Evaluation set size</td><td>3 strict-mode episodes per checkpoint</td></tr><tr><td>Evaluation checkpoints</td><td> $x \in \{ 0 , 3 , 6 , \ldots , 3 0 \}$ </td></tr></table>

Table 3 Runtime configuration for the main context-level experiments. Game-specific horizons smaller than 64 override the controller-level step cap.

Metrics. Let $y _ { m , p , g , k }$ denote the evaluation score of model m, pathway $p ,$ and game g at checkpoint $x _ { k }$ . We report three complementary metrics for every model–game pair:

$$
\mathrm { A v g } _ { m , p , g } = \frac { 1 } { K + 1 } \sum _ { k = 0 } ^ { K } y _ { m , p , g , k } ,\tag{18}
$$

$$
\mathrm { M a x } _ { m , p , g } = \operatorname* { m a x } _ { 0 \leq k \leq K } y _ { m , p , g , k } ,\tag{19}
$$

$$
\mathrm { A U C } _ { m , p , g } ^ { + } = \int _ { 0 } ^ { 3 0 } \operatorname* { m a x } ( 0 , y _ { m , p , g } ( x ) - y _ { m , p , g , 0 } ) \ d x .\tag{20}
$$

The continuous curve $y ( x )$ in $\mathrm { A U C ^ { + } }$ is obtained through piecewise-linear interpolation between adjacent checkpoints. Avg. measures overall performance throughout exploration; Max. captures the best observed capability, and $\bar { \mathrm { A U C ^ { + } } }$ measures sustained improvement above the initial evaluation score. Because the games use diferent score scales, raw $\mathrm { A U C ^ { + } }$ values should be compared within the same game, rather than directly across games.

## 5.3 Main Results

Table 4 reports Avg., Max., and $\mathrm { A U C ^ { + } }$ for all complete models under History ICL and Summary Memory.   
Bold values indicate the best result for each metric within the corresponding pathway and game.

Overall context-level performance. The results show that self-improvement potential is jointly determined by the base model, the experience pathway, and the environment. Under History ICL, Gemini-3.5-Flash obtains the strongest Chess results and the largest AUC on Snake and Trust Evolution, whereas GPT-5.5 leads the more brittle Minesweeper, Nullify, and Tetris tasks and achieves the largest PvZ AUC. Under Summary Memory, GPT-5.5 becomes strongest on Chess, Nullify, and Tetris and reaches the highest average and maximum scores on Snake. Gemini-3.5-Flash remains strongest in average performance on PvZ and Trust Evolution. These results indicate that high base capability alone does not determine self-improvement; models also difer in how efectively they retrieve, compress, and apply their own experience.

History ICL versus Summary Memory. Summary Memory is not uniformly superior to raw history. It can substantially improve models for which long trajectories are dificult to use directly. For example, Gemini-2.5-Flash increases its Minesweeper $\mathrm { A U C ^ { + } }$ from 0.000 under ICL to 7.794 with Summary Memory, and its PvZ $\mathrm { A U C ^ { + } }$ from 24.402 to 238.501. GPT-5.5 similarly improves its Chess $\mathrm { A U C ^ { + } }$ from 0.474 to 16.840. However, summarization may discard state-specific evidence or preserve incorrect judgments. GPT-5.5’s PvZ $\mathrm { A U C ^ { + } }$ decreases from 548.499 to 33.219, while Gemini-3.5-Flash decreases from 12.280 to 0.000 on Chess and from 200.000 to 55.095 on Trust Evolution. Summary Memory should therefore be viewed as a selective compression mechanism rather than an unconditional improvement over History ICL.

Table 4 Main results on seven $\mathrm { S ^ { 3 } G y m }$ games. For each model–game pair, Avg. is the mean score over all available evaluation checkpoints, Max. is the maximum score, and $\mathrm { A U C ^ { + } }$ is the positive area above the initial-score baseline. Higher is better for all metrics. Explicitly marked concurrency failures are excluded.
<table><tr><td>Model</td><td>Chess</td><td>Minesweeper</td><td>Nullify</td><td>Tetris</td><td>Snake</td><td>PvZ</td><td>TrustEvo</td></tr><tr><td colspan="8">ICL (History Context)</td></tr><tr><td>GPT-40</td><td>0.007</td><td>0.044</td><td>0.030</td><td>0.030</td><td>0.212</td><td>11.636</td><td>8.242</td></tr><tr><td>GPT-4.1</td><td>0.010</td><td>0.042</td><td>0.030</td><td>0.152</td><td>0.515</td><td>16.909</td><td>7.371</td></tr><tr><td>o3-mini</td><td>0.012</td><td>0.222</td><td>0.030</td><td>1.273</td><td>4.000</td><td>26.333</td><td>9.242</td></tr><tr><td>Gemini-2.5-Flash</td><td>0.009</td><td>0.063</td><td>0.030</td><td>0.121</td><td>1.424</td><td>20.697</td><td>9.008</td></tr><tr><td>Gemini-2.5-Pro</td><td>0.025</td><td>0.421</td><td>0.212</td><td>0.273</td><td>1.576</td><td>28.242</td><td>14.500</td></tr><tr><td>GPT-5.5</td><td>0.017</td><td>0.604</td><td>0.545</td><td>4.242</td><td>9.061</td><td>44.033</td><td>7.848</td></tr><tr><td>Gemini-3.5-Flash</td><td>0.547</td><td>0.482</td><td>0.394</td><td>1.970</td><td>8.455</td><td>44.697</td><td>17.242</td></tr></table>

<table><tr><td>GPT-4o</td><td>0.005</td><td>0.013</td><td>0.000</td><td>0.030</td><td>0.303</td><td>11.818</td><td>13.947</td></tr><tr><td>GPT-4.1</td><td>0.010</td><td>0.012</td><td>0.030</td><td>0.182</td><td>0.545</td><td>14.727</td><td>13.371</td></tr><tr><td>o3-mini</td><td>0.006</td><td>0.425</td><td>0.061</td><td>1.788</td><td>6.727</td><td>26.424</td><td>7.174</td></tr><tr><td>Gemini-2.5-Flash</td><td>0.074</td><td>0.438</td><td>0.091</td><td>0.152</td><td>1.364</td><td>21.939</td><td>8.720</td></tr><tr><td>Gemini-2.5-Pro</td><td>0.110</td><td>0.466</td><td>0.182</td><td>0.545</td><td>1.273</td><td>19.667</td><td>9.568</td></tr><tr><td>GPT-5.5</td><td>0.533</td><td>0.742</td><td>0.576</td><td>3.515</td><td>10.697</td><td>33.500</td><td>8.697</td></tr><tr><td>Gemini-3.5-Flash</td><td>0.208</td><td>0.603</td><td>0.455</td><td>1.091</td><td>10.242</td><td>43.879</td><td>14.394</td></tr></table>

(b) Maximum score (Max.)
<table><tr><td>Model</td><td>Chess</td><td>Minesweeper</td><td>Nullify</td><td>Tetris</td><td>Snake</td><td>PvZ</td><td>TrustEvo</td></tr><tr><td colspan="8">ICL (History Context)</td></tr><tr><td>GPT-40</td><td>0.018</td><td>0.131</td><td>0.333</td><td>0.333</td><td>0.667</td><td>17.667</td><td>11.500</td></tr><tr><td>GPT-4.1</td><td>0.018</td><td>0.189</td><td>0.333</td><td>0.667</td><td>1.667</td><td>21.667</td><td>10.250</td></tr><tr><td>o3-mini</td><td>0.022</td><td>0.500</td><td>0.333</td><td>2.667</td><td>9.667</td><td>29.333</td><td>13.250</td></tr><tr><td>Gemini-2.5-Flash</td><td>0.022</td><td>0.670</td><td>0.333</td><td>0.333</td><td>2.333</td><td>29.000</td><td>13.583</td></tr><tr><td>Gemini-2.5-Pro</td><td>0.089</td><td>0.659</td><td>0.667</td><td>1.000</td><td>4.333</td><td>33.000</td><td>19.000</td></tr><tr><td>GPT-5.5</td><td>0.067</td><td>0.974</td><td>1.000</td><td>11.333</td><td>11.333</td><td>48.000</td><td>23.000</td></tr><tr><td>Gemini-3.5-Flash</td><td>0.742</td><td>0.705</td><td>1.000</td><td>3.000</td><td>13.000</td><td>62.667</td><td>21.000</td></tr></table>

<table><tr><td colspan="8">Summary Memory</td></tr><tr><td>GPT-4o</td><td>0.018</td><td>0.033</td><td>0.000</td><td>0.333</td><td>1.000</td><td>20.333</td><td>19.333</td></tr><tr><td>GPT-4.1</td><td>0.031</td><td>0.083</td><td>0.333</td><td>0.667</td><td>1.000</td><td>19.333</td><td>15.917</td></tr><tr><td>o3-mini</td><td>0.018</td><td>0.608</td><td>0.333</td><td>4.000</td><td>10.333</td><td>29.000</td><td>10.083</td></tr><tr><td>Gemini-2.5-Flash</td><td>0.147</td><td>0.665</td><td>0.333</td><td>0.667</td><td>2.333</td><td>26.333</td><td>13.750</td></tr><tr><td>Gemini-2.5-Pro</td><td>0.258</td><td>0.974</td><td>0.333</td><td>2.000</td><td>3.000</td><td>26.667</td><td>14.333</td></tr><tr><td>GPT-5.5</td><td>0.773</td><td>1.000</td><td>1.000</td><td>6.667</td><td>14.000</td><td>41.333</td><td>17.000</td></tr><tr><td>Gemini-3.5-Flash</td><td>0.524</td><td>0.961</td><td>1.000</td><td>2.333</td><td>12.000</td><td>51.667</td><td>20.333</td></tr></table>

(c) Positive area above baseline (AUC<sup>+</sup>)
<table><tr><td>Model</td><td>Chess</td><td>Minesweeper</td><td>Nullify</td><td>Tetris</td><td>Snake</td><td>PvZ</td><td>TrustEvo</td></tr><tr><td colspan="8">ICL (History Context)</td></tr><tr><td>GPT-40</td><td>0.233</td><td>1.437</td><td>0.500</td><td>1.000</td><td>0.000</td><td>0.052</td><td>3.443</td></tr><tr><td>GPT-4.1</td><td>0.186</td><td>1.329</td><td>1.000</td><td>0.000</td><td>8.501</td><td>129.000</td><td>5.966</td></tr><tr><td>o3-mini</td><td>0.004</td><td>0.000</td><td>1.000</td><td>0.000</td><td>0.000</td><td>5.383</td><td>62.638</td></tr><tr><td>Gemini-2.5-Flash</td><td>0.017</td><td>0.000</td><td>1.000</td><td>0.000</td><td>7.317</td><td>24.402</td><td>6.251</td></tr><tr><td>Gemini-2.5-Pro</td><td>0.000</td><td>3.539</td><td>1.500</td><td>0.000</td><td>0.000</td><td>60.416</td><td>161.876</td></tr><tr><td>GPT-5.5</td><td>0.474</td><td>0.959</td><td>1.000</td><td>96.251</td><td>0.163</td><td>548.499</td><td>164.411</td></tr><tr><td>Gemini-3.5-Flash</td><td>12.280</td><td>1.217</td><td>0.417</td><td>34.000</td><td>37.387</td><td>161.945</td><td>200.000</td></tr></table>

<table><tr><td colspan="8">Summary Memory</td></tr><tr><td>GPT-4o</td><td>0.007</td><td>0.046</td><td>0.000</td><td>0.500</td><td>1.917</td><td>12.778</td><td>10.350</td></tr><tr><td>GPT-4.1</td><td>0.177</td><td>0.046</td><td>1.000</td><td>0.000</td><td>0.000</td><td>75.462</td><td>154.876</td></tr><tr><td>o3-mini</td><td>0.079</td><td>0.001</td><td>2.000</td><td>28.598</td><td>0.300</td><td>40.425</td><td>30.521</td></tr><tr><td>Gemini-2.5-Flash</td><td>2.147</td><td>7.794</td><td>3.000</td><td>0.500</td><td>0.000</td><td>238.501</td><td>18.443</td></tr><tr><td>Gemini-2.5-Pro</td><td>0.573</td><td>0.721</td><td>0.000</td><td>10.084</td><td>11.321</td><td>11.872</td><td>10.624</td></tr><tr><td>GPT-5.5</td><td>16.840</td><td>2.951</td><td>7.418</td><td>50.801</td><td>0.000</td><td>33.219</td><td>240.500</td></tr><tr><td>Gemini-3.5-Flash</td><td>0.000</td><td>0.793</td><td>5.584</td><td>15.808</td><td>0.750</td><td>123.535</td><td>55.095</td></tr></table>

Note. AUC<sup>+</sup> is computed over checkpoints $\mathbf { \sigma } : \in \{ 0 , 3 , \ldots , 3 0 \}$ using piecewise-linear trapezoidal integration of $\begin{array} { r } { \int _ { 0 } ^ { 3 0 } \operatorname* { m a x } ( 0 , y ( x ) - y _ { 0 } ) } \end{array}$ dx. Means and maxima exclude explicitly marked invalid concurrency points; for AUC<sup>+</sup>, adjacent valid checkpoints are connected. Models with unfinished results in any game are omitted.

![](images/03fb18c5cd5e88fa6c6239b06f127e5265152c5abd6f2964f2388f31fd03108b.jpg)

![](images/da2a72a014ab182603a3df40e89be18af542d734a425e1820c6040e1e1c4322b.jpg)

![](images/788f55b79883b45702e199038ffa28a48e00bd55bfe997c2c67d4578701a9235.jpg)

![](images/8140610c8f57142cf31b6054ced216ec2eb63ee342a7451001be36f7f39e983e.jpg)  
Figure 3 Paired History-ICL and Summary-Memory trajectories for Chess and Minesweeper. The horizontal axis denotes exploration episodes, and the vertical axis denotes strict-mode evaluation performance. Within each game, the two panels use the same vertical scale.

Cross-game consistency. The seven games reveal diferent self-improvement regimes. Chess, Minesweeper, and Nullify have sparse or highly discrete scores, so a single successful episode can strongly afect Max. while producing only a small Avg. or AUC. Tetris, Snake, $\mathrm { P v } { \cal Z } ,$ and Trust Evolution provide denser sequential feedback and expose clearer diferences in sustained adaptation. The three metrics are consequently complementary: Max. identifies occasional capability breakthroughs, Avg. reflects overall performance, and $\mathrm { A U C ^ { + } }$ distinguishes persistent improvement from isolated peaks. Since score scales difer substantially across games, cross-game consistency should be assessed through within-game ranks or the number of games improved, rather than by averaging raw AUC magnitudes.

## 6 Analysis

## 6.1 RQ1: Can Parameter Training Enable Self-Improvement?

RQ1 examines whether an agent can internalize its interaction experience through parameter updates and subsequently improve without access to historical trajectories or external memory at inference time. Compared with context-level adaptation, parameter training ofers a more persistent improvement pathway, but it also carries greater risk: noisy or incorrectly judged trajectories may be consolidated into the model and overwrite previously efective behavior.

We evaluate Qwen3-8B at 20 consecutive checkpoints, with epoch 0 representing the original model and epochs 1–19 representing models updated using exploration trajectories. At each checkpoint, the model is evaluated on the strict configurations of all seven $\mathrm { S ^ { 3 } G }$ ym games without History ICL or Summary Memory. We report the post-training average and maximum scores, the final change $\Delta _ { \mathrm { F i n a l } } = y _ { 1 9 } - y _ { 0 }$ , and the positive

![](images/3d220244fde87dee5e10b1a11bd252ec03a0eb47b3837b681d9bd0542b606852.jpg)

![](images/a3a02e799bda05f58fdcb27d61a58ca83553ef2ebe4c386db4e4fbeb8ee641c3.jpg)

![](images/bd2d0a07a2758f3c7b5cec4bc993051e72179c9c72a0bcd0f029bf876eddfa7a.jpg)

![](images/0d7b654dec45d3872943d9d06b491c73f0940b53927f8bde8116b8fabfe64b20.jpg)  
Figure 4 Paired History-ICL and Summary-Memory trajectories for Nullify and Tetris. Model colors and line styles are shared across all panels.

![](images/c6df7a431bfaf7bf8958a6ac9471170fa3292b20113fc4416e1f852b2872eeb6.jpg)

![](images/aa94f0b24aa49c8328a2b3fa4a0bfb605f6ecf75e859ff278bf9c8431ba4138c.jpg)

![](images/e5a96e97ad2d3967852ea17e1185f8c57e26d050a2d4cd8faeb9b09250b088a2.jpg)

![](images/e669d1f005808243845c5be11cd6ffda9fab4edddcdc440a200c81dd5524ae11.jpg)  
Figure 5 Paired History-ICL and Summary-Memory trajectories for Snake and Plants-vs-Zombies. Within each game, both pathways use the same vertical scale.

![](images/dc94c9b895499b2bc4b533c70a385e914cc94306a856d8c56b58c96646ec8b03.jpg)

![](images/90c87a1e3f8d5c895ed8ab8a5666bab60269600e5fbaeac26efd2190031d7273.jpg)  
Figure 6 Paired History-ICL and Summary-Memory trajectories for Trust Evolution.

area above the initial baseline,

$$
\mathrm { A U C ^ { + } } = \int _ { 0 } ^ { 1 9 } \mathrm { m a x } ( 0 , y ( e ) - y _ { 0 } ) ~ d e ,
$$

where y(e) is obtained by linearly interpolating adjacent checkpoints.

![](images/194ac8545cb5042e925d1da1c77b01acfdadaaba07d82993b2ce55d6b3bcdf32.jpg)  
Figure 7 Strict-mode evaluation performance of Qwen3-8B over 20 training checkpoints. Epoch 0 denotes the original model. Because the games use diferent score scales, each trajectory should be interpreted relative to its own initial performance.

• Training yields substantial self-improvement on Trust Evolution. The score rises from 0 to a maximum of 30 and remains above the initial baseline at 18 of the 19 updated checkpoints. The post-training average reaches 8.684 and AUC<sup>+</sup> reaches 163.5, providing the clearest evidence that interaction-derived strategies can be internalized into model parameters.

• The gains on Chess and Snake are limited or transient. Chess remains at zero for most checkpoints but reaches 0.0667 at the final checkpoint, indicating a sparse yet positive gain. Snake reaches a score of 1 at epochs 8, 12, and 15 but subsequently returns to zero. Training therefore discovers occasional successful behavior without retaining it consistently.

• Minesweeper, Nullify, and Tetris show no measurable improvement. Their evaluation scores remain zero throughout training. One possible explanation is that exploration produces too few successful trajectories to bootstrap an efective supervised signal, although verifying this hypothesis requires trajectory-level analysis.

• Training produces clear negative transfer on Plants-vs-Zombies. The initial score is 23, whereas every updated checkpoint obtains a score of 6. This persistent degradation indicates that parameter updates can overwrite previously efective behavior. Potential causes include overfitting to exploration trajectories, mismatch between relaxed exploration and strict evaluation, or unreliable Self-Judging signals.

• Overall, parameter training enables self-improvement, but not consistently across tasks. Qwen3-8B improves substantially on Trust Evolution, weakly on Chess, and intermittently on Snake, while showing no improvement on three games and severe degradation on Plants-vs-Zombies. Efective training-level self-improvement therefore depends not only on the ability to update parameters, but also on the quality, diversity, and correctness of the experience being internalized.

## 6.2 RQ2: Is Agent Self-Judging Reliable?

Self-Judging is a central component of $\mathrm { S ^ { 3 } G y m }$ because the model-generated score $s _ { i }$ is stored with each transition and subsequently influences trajectory selection, summary construction, and training-data generation. If $s _ { i }$ is misaligned with the environment signal, the agent may preserve harmful actions as useful experience or discard transitions that should have been retained. We therefore evaluate Self-Judging as an independent capability and examine whether judgment quality predicts subsequent improvement.

Step-level reliability. For each transition, we compare the model-generated score $s _ { i }$ with the environment reward $r _ { i }$ . We first measure event agreement, which indicates whether the model and environment agree on the presence of a positive outcome:

$$
\mathrm { A g r e e } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } \left[ \mathbf { 1 } ( s _ { i } > 0 ) = \mathbf { 1 } ( r _ { i } > 0 ) \right] .\tag{21}
$$

Because reward scales difer across games, we additionally report normalized mean absolute error:

$$
\mathrm { N M A E } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left. \frac { s _ { i } - s _ { \mathrm { m i n } } } { s _ { \mathrm { m a x } } - s _ { \mathrm { m i n } } } - \frac { r _ { i } - r _ { \mathrm { m i n } } } { r _ { \mathrm { m a x } } - r _ { \mathrm { m i n } } } \right. .\tag{22}
$$

We also measure over-confidence, $\operatorname* { P r } ( s _ { i } > 0 , r _ { i } \le 0 )$ , and under-confidence, $\operatorname* { P r } ( s _ { i } \leq 0 , r _ { i } > 0 )$ . The analysis covers the seven target models and games under both direct-history and summary-memory conditions, comprising 98 runs and 116,117 transitions.

<table><tr><td>Game</td><td>Agreement</td><td>NMAE</td><td>Over-conf.</td><td>Under-conf.</td></tr><tr><td>Chess</td><td>0.496</td><td>0.141</td><td>0.365</td><td>0.139</td></tr><tr><td>Minesweeper</td><td>0.820</td><td>0.524</td><td>0.062</td><td>0.118</td></tr><tr><td>Nullify</td><td>0.827</td><td>0.056</td><td>0.170</td><td>0.004</td></tr><tr><td>PvZ</td><td>0.881</td><td>0.882</td><td>0.032</td><td>0.087</td></tr><tr><td>Snake</td><td>0.879</td><td>0.121</td><td>0.110</td><td>0.011</td></tr><tr><td>Tetris</td><td>0.846</td><td>0.041</td><td>0.151</td><td>0.002</td></tr><tr><td>Trust</td><td>0.665</td><td>0.392</td><td>0.291</td><td>0.044</td></tr></table>

Table 5 Step-level reliability of model Self-Judging. Agreement measures binary agreement on whether a transition receives a positive score. NMAE measures score calibration after game-level min–max normalization.

Table 5 shows that Self-Judging is only partially reliable. PvZ, Snake, Tetris, Minesweeper, and Nullify exhibit relatively high event agreement, but this result is partly explained by the large proportion of zero-reward transitions. NMAE reveals whether the magnitude of the model’s judgment is also well calibrated. Tetris and Nullify achieve low NMAE, whereas PvZ exhibits the largest calibration error despite its high binary agreement. Models can therefore recognize whether an action appears locally useful while substantially misestimating its value.

Chess and Trust are the clearest failure cases. Chess has near-random event agreement and the highest over-confidence rate. A prediction may be syntactically complete while still misplacing several pieces, causing the model’s confidence in its response to diverge from the environment score. Trust similarly exhibits weak agreement, high NMAE, and substantial over-confidence. These results suggest that Self-Judging becomes less reliable when reward depends on hidden dynamics, opponent strategy, or exact multi-object state prediction rather than an immediately observable local consequence.

Judgment–improvement coupling. Reliable local judgments are useful only if the agent can convert them into better subsequent behavior. To test this relationship, we divide exploration trajectories into blocks $B _ { t }$ between adjacent evaluation checkpoints. For each block, we calculate event agreement $A _ { t } ,$ normalized calibration error $E _ { t } .$ , and the normalized score change at the following evaluation:

$$
g _ { t } = \frac { y _ { t } - y _ { t - 3 } } { \operatorname* { m a x } _ { g , \tau } y _ { g , \tau } - \operatorname* { m i n } _ { g , \tau } y _ { g , \tau } } .\tag{23}
$$

We then measure $\rho ( A _ { t } , g _ { t } )$ and $\rho ( - E _ { t } , g _ { t } )$ , together with the diference in average score gain between the highest- and lowest-quality judgment tertiles.

<table><tr><td>Game</td><td> $\rho ( A , g )$   $\rho ( - E , g )$ </td><td> $\mathbf { G a p } _ { A }$ </td><td> $\mathbf { G a p } _ { E }$ </td></tr><tr><td>All</td><td>-0.010</td><td>-0.018 -0.004</td><td>-0.016</td></tr><tr><td>Chess</td><td>-0.014</td><td>-0.058 0.028</td><td>-0.039</td></tr><tr><td>Minesweeper</td><td>0.071</td><td>0.006 0.069</td><td>0.005</td></tr><tr><td>Nullify</td><td>-0.032</td><td>-0.082 -0.065</td><td>-0.058</td></tr><tr><td>PvZ</td><td>-0.038</td><td>-0.023 0.014</td><td>0.010</td></tr><tr><td>Snake</td><td>0.022</td><td>0.022 0.009</td><td>0.009</td></tr><tr><td>Tetris</td><td>-0.048</td><td>-0.021 0.010</td><td>0.004</td></tr><tr><td>Trust</td><td>-0.014</td><td>-0.003 -0.001</td><td>0.006</td></tr></table>

Table 6 Blockwise judgment–improvement coupling. A is event agreement, E is normalized calibration error, and g is the normalized strict-mode score change at the subsequent evaluation. The gap metrics compare the highest and lowest judgment- quality tertiles.

The coupling results are close to zero across the full benchmark: $\rho ( A , g ) = - 0 . 0 1 0$ and $\rho ( - E , g ) = - 0 . 0 1 8$ The corresponding tertile gaps are also small and slightly negative. Minesweeper exhibits a weak positive relationship between event agreement and subsequent score gain, but its calibration coupling remains negligible. Overall, accurate local scoring does not reliably produce improvement at the next evaluation checkpoint.

The run-level analysis leads to the same conclusion. To quantify sustained improvement while accounting for diferences in initial performance, we use the normalized above-baseline area (NABA), defined as

$$
\mathrm { N A B A } = \frac { \mathrm { A U C ^ { + } } } { \operatorname* { m a x } ( y _ { 0 } , \epsilon ) } ,
$$

where $y _ { 0 }$ is the initial evaluation score and ϵ denotes the first non-zero score scale when $y _ { 0 } = 0$ . Event agreement exhibits only weak negative correlations with NABA under both Pearson and Spearman measures $( r = - 0 . 2 3$ and $\rho = - 0 . 1 1 )$ , while the positive-reward and positive-self-score rates show only weak positive correlations. These results indicate that accurate local Self-Judging alone is insuficient for Self-Improvement: agents must also transform feedback into reusable state abstractions, decision rules, and exploration strategies.

## 6.3 RQ3: Do Score-Conditioned Summaries Produce Effective Iteration Directions?

Summary Memory imposes a stronger requirement than direct history reuse. Rather than merely conditioning on previous trajectories, the model must identify which low-scoring behaviors should be suppressed, which high-scoring behaviors should be retained, and how these observations should be converted into an executable strategy for future episodes. We evaluate this capability by comparing score-conditioned summaries with direct-history ICL and by inspecting representative successful and unsuccessful summaries.

Summary Memory versus direct history. For every model–game pair, we calculate ∆NABA as Summary-Memory NABA minus Direct-History NABA. Positive values indicate that summarization produces a more efective improvement trajectory than retaining the raw interaction history.
<table><tr><td>Game</td><td>Pairs</td><td>Summary wins</td><td>History wins</td><td>Ties</td><td>Mean ∆NABA</td></tr><tr><td>Chess</td><td>7</td><td>3</td><td>3</td><td>1</td><td>3.37</td></tr><tr><td>Minesweeper</td><td>7</td><td>3</td><td>4</td><td>0</td><td>-1.12</td></tr><tr><td>Nullify</td><td>7</td><td>4</td><td>2</td><td>1</td><td>2.79</td></tr><tr><td>PvZ</td><td>7</td><td>3</td><td>4</td><td>0</td><td>-2.11</td></tr><tr><td>Snake</td><td>7</td><td>3</td><td>4</td><td>0</td><td>-0.81</td></tr><tr><td>Tetris</td><td>7</td><td>5</td><td>1</td><td>1</td><td>6.20</td></tr><tr><td>Trust</td><td>7</td><td>4</td><td>3</td><td>0</td><td>8.43</td></tr></table>

Table 7 Summary Memory compared with direct-history ICL. Each pair evaluates the same model and game under the two history-organization strategies.

Summary Memory improves average NABA on Nullify, Tetris, and Trust. These environments admit compact and reusable abstractions, such as planning backward toward exact cancellation, preserving a stable board surface, or adapting behavior to an inferred opponent strategy. In such cases, summaries remove redundant trajectory details while retaining the information most relevant to future decisions.

In contrast, direct history performs better on average in Minesweeper, PvZ, and Snake. These games depend heavily on local board configurations, lane timing, hazard positions, and partially observed risk. A summary may preserve a sensible high-level objective while discarding the state-contingent details required to execute it correctly. Summary Memory is therefore most efective when experience can be compressed into a stable policy rule, rather than when success depends on reconstructing the exact current state.

Trajectory-level evidence. The following cases illustrate when score-conditioned summaries become executable iteration directions and when they remain too generic to improve behavior.

## Successful summaries

## GPT-5.5 / Trust

Summary direction. Always defect when the observed payof structure consistently favors defection; avoid unnecessary reciprocal exploration.

Trajectory evidence. At the first decision, the agent selected cheat and received an environment reward of 3.0. Outcome. ∆NABA = +66.89.

Takeaway. The summary translates payof feedback into an explicit opponent-conditioned policy.

## o3-mini / Tetris

Summary direction. Jointly inspect the current and next blocks, test alternative rotations, prioritize line-clearing placements, and maintain a low surface without isolated holes.

Trajectory evidence. In episode 10, the agent cleared four lines. It selected 8 0 twice, with both its self-score and the environment reward equal to 1.0 on each move.

Outcome. ∆NABA = +14.42.

Takeaway. The summary converts successful line clears into an executable look-ahead rule.

## Gemini-2.5-Flash / Minesweeper

Summary direction. Apply local number constraints: expand zero regions, flag forced mines, and uncover cells that are provably safe.

Trajectory evidence. The agent selected flag (2,2) on a partially revealed board and received a reward of 0.0625. Outcome. ∆NABA = +25.49.

Takeaway. Summaries help when they express local reasoning as executable constraint rules.

## Failure summaries

## GPT-5.5 / PvZ

Summary direction. Prioritize sun production, early defense, Wall-nut stalling, and coverage of threatened lanes.

Trajectory evidence. The agent opened with X 1 0 and received a reward of 1.0, but its later strict score fell from 38.67 to 32.33.

Outcome. ∆NABA = −2.56.

Takeaway. The advice is reasonable but too coarse to capture lane timing, current zombie positions, and the available sun budget.

## GPT-4.1 / Minesweeper

Summary direction. Begin from corners, expand zero regions, flag only certain mines, and stop when no safe progress is available.

Trajectory evidence. The agent selected uncover (8,0) after exhausting its available lives, and the subsequent checkpoint score fell from 0.0606 to 0.

Outcome. ∆NABA = −22.09.

Takeaway. Generic heuristics do not substitute for exact constraint propagation.

## Gemini-3.5-Flash / Nullify

Summary direction. Plan backward to create exact opposites, use floor or ceiling operations to remove decimals, and track index changes after every merge.

Trajectory evidence. With terminal units -19 and 19, the agent selected 0 1 and received a reward of 1.0.

Outcome. ∆NABA = +8.50.

Takeaway. The summary succeeds because it specifies a concrete arithmetic planning procedure.

## Gemini-3.5-Flash / Chess

Summary direction. Track the movement rule of every piece and verify all coordinates before submitting the answer.

Trajectory evidence. In a low-scoring episode, the model produced END and received zero total reward despite having generated a detailed rule summary.

Outcome. ∆NABA = −33.27.

Takeaway. A high-level reminder does not resolve the exact multi-piece localization problem.

## GPT-4o / Snake

Summary direction. Move toward food while avoiding walls, obstacles, the snake body, and direct reversals.

Trajectory evidence. Given a board containing food, obstacles, and its own body, the agent selected DOWN; it received no reward, and the strict score fell from 1.0 to 0. Outcome. ∆NABA = −22.00.

Takeaway. The summary omits the precise geometry required for safe navigation.

Taken together, these cases reveal two requirements for efective summary-based improvement. First, the score must identify a meaningful and reusable failure mode rather than merely indicate that the episode ended poorly. Second, the summary must convert that signal into an action rule precise enough to execute in a new state. Summary Memory is therefore a selective improvement mechanism: it is efective when experience admits compact causal abstractions, but it cannot universally replace direct access to state-rich interaction histories.

## 7 Conclusion

S3GYM turns self-improvement from a broad claim into a measurable agent capability. By separating relaxed exploration from stricter evaluation, and by supporting both context-level and training-level reuse of trajectories, the benchmark tests whether LLMs can transform interaction history into future performance. The evidence so far is mixed in a useful way. Summary memory helps in games where compact rules transfer across episodes, while direct history is often better in reactive control games where details matter. Training on self-generated trajectories remains unstable in the current implementation. These findings suggest that future self-improving agents need better judgment calibration, memory selection, and trajectory filtering, not just more interaction data.

## 8 Contributions

## Core Contributors

Jiajun Shi, Siyuan Tao, Yuhao Wu, Zexuan Wang, Jingyuan Zhang

## Contributors

Jiaheng Liu, Xinping Lei, Xinrong Zhang, Siyuan Fang, Zhewen Tan, Tianle Cai, Junhao Fang, Jiameng Huang, Yueyang Wang, Jinkai Liu, Yuxuan Zhang

## Corresponding Authors

Jian Yang , Zhoujun Li, Shen Yan, Wenhao Huang , Ge Zhang

## References

[1] Emre Can Acikgoz, Cheng Qian, Heng Ji, Dilek Hakkani-Tür, and Gokhan Tur. Self-improving llm agents at test-time. arXiv preprint arXiv:2510.07841, 2025. URL https://arxiv.org/abs/2510.07841.

[2] Chris Argyris and Donald A. Sch"on. Organizational Learning: A Theory of Action Perspective. Addison-Wesley, Reading, MA, 1978. ISBN 9780201001747.

[3] Aditya Challapally, Chris Pease, Ramesh Raskar, and Pradyumna Chari. The GenAI divide: State of AI in business 2025. https://www.artificialintelligence-news.com/wp-content/uploads/2025/08/ai\_report\_2025.pdf, 2025. Industry report.

[4] Yizhe Chi, Wenyi Li, Deyao Hong, Xiaoqiu Wang, Mingju Gao, Kaisen Yang, Bingxiang He, Youjie Zheng, Calvin Xiao, and Qinhuai Na. Ai4ai-bench: Benchmarking llm agents in algorithmic design for recursive self-improvement. arXiv preprint arXiv:2608.20318, 2026. URL https://arxiv.org/abs/2608.20318.

[5] Marc-Alexandre Côté, Ákos Kádár, Xingdi Yuan, Ben Kybartas, Tavian Barnes, Emery Fine, James Moore, Ruo Yu Tao, Matthew Hausknecht, Layla El Asri, Mahmoud Adada, Wendy Tay, and Adam Trischler. TextWorld: A learning environment for text-based games. arXiv preprint arXiv:1806.11532, 2018. URL https://arxiv.org/ abs/1806.11532.

[6] Bo Deng, Kang Zhou, Lifan Guo, Chongyang Tao, Xuanren Chen, Chenggang Xie, Renzhao Liang, Feng Chen, and Chi Zhang. Finevo-bench: A longitudinal benchmark for self-evolving agents in professional financial workflows. arXiv preprint arXiv:2608.06144, 2026. URL https://arxiv.org/abs/2608.06144.

[7] Zihao Deng, Yining Zhu, Leiming Wang, Junbo Wang, and Jingfei Lu. Tree-of-experience: Hierarchical experience management for self-evolving agents. arXiv preprint arXiv:2608.09044, 2026. URL https://arxiv.org/abs/ 2608.09044.

[8] John Dewey. How We Think: A Restatement of the Relation of Reflective Thinking to the Educative Process. D. C. Heath and Company, Boston, MA, 2 edition, 1933.

[9] Zi-Yi Dou, Cheng-Fu Yang, Xueqing Wu, Kai-Wei Chang, and Nanyun Peng. Re-ReST: Reflection-reinforced selftraining for language agents. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 15394–15411, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.861. URL https://aclanthology.org/2024.emnlp-main.861/.

[10] Tianyi Guan, Yiding Wang, Haotong Yang, Siyuan Cao, Shirui Liu, Yi Hu, Jiaqi Li, and Muhan Zhang. Continualskillbench: Can llm agents truly evolve their capabilities? arXiv preprint arXiv:2608.03874, 2026. URL https://arxiv.org/abs/2608.03874.

[11] Caglar Gulcehre, Tom Le Paine, Srivatsan Srinivasan, Ksenia Konyushkova, Lotte Weerts, Abhishek Sharma, Aditya Siddhant, Alex Ahern, Miaosen Wang, Chenjie Gu, Wolfgang Macherey, Arnaud Doucet, Orhan Firat, and Nando de Freitas. Reinforced self-training (ReST) for language modeling. arXiv preprint arXiv:2308.08998, 2023. URL https://arxiv.org/abs/2308.08998.

[12] Keyu He, Xuhui Zhou, and Maarten Sap. Social gym and spartan: Benchmarking and improving llm social reasoning via multi-agent game tournaments. arXiv preprint arXiv:2608.09128, 2026. URL https://arxiv.org/ abs/2608.09128.

[13] Sihang Jiang, Lipeng Ma, Zhonghua Hong, Keyi Wang, Zhiyu Lu, Shisong Chen, Jinghao Zhang, Tianjun Pan, Weijia Zhou, Jiaqing Liang, and Yanghua Xiao. Sea-eval: A benchmark for evaluating self-evolving agents beyond episodic assessment. arXiv preprint arXiv:2604.08988, 2026. URL https://arxiv.org/abs/2604.08988.

[14] David A. Kolb. Experiential Learning: Experience as the Source of Learning and Development. Prentice-Hall, Englewood Clifs, NJ, 1984. ISBN 9780132952613.

[15] Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. AgentBench: Evaluating LLMs as agents. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=zAdUB0aCTQ.

[16] Chang Ma, Junlei Zhang, Zhihao Zhu, Cheng Yang, Yujiu Yang, Yaohui Jin, Zhenzhong Lan, Lingpeng Kong, and Junxian He. AgentBoard: An analytical evaluation board of multi-turn LLM agents. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://arxiv.org/abs/2401.13178.

[17] James G. March. Exploration and exploitation in organizational learning. Organization Science, 2(1):71–87, 1991. doi: 10.1287/orsc.2.1.71.

[18] Gergely Orosz. What are forward deployed engineers, and why are they so in demand? https://newsletter. pragmaticengineer.com/p/forward-deployed-engineers, 2025. The Pragmatic Engineer.

[19] Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T. Le, Samira Daruki, Xiangru Tang, Vishy Tirumalashetty, George Lee, Mahsan Rofouei, Hangfei Lin, Jiawei Han, Chen-Yu Lee, and Tomas Pfister. Reasoningbank: Scaling agent self-evolving with reasoning memory. arXiv preprint arXiv:2509.25140, 2025. URL https://arxiv.org/abs/2509.25140.

[20] Palantir Technologies. A day in the life of a Palantir forward deployed software engineer. https://blog.palantir. com/a-day-in-the-life-of-a-palantir-forward-deployed-software-engineer-45ef2de257b1, 2020.

[21] Karl R. Popper. The Logic of Scientific Discovery. Hutchinson, London, 1959.

[22] Ben Rank, Hardik Bhatnagar, Ameya Prabhu, Shira Eisenberg, Karina Nguyen, Matthias Bethge, and Maksym Andriushchenko. Posttrainbench: Can llm agents automate llm post-training? In International Conference on Machine Learning (ICML), 2026. URL https://arxiv.org/abs/2603.08640.

[23] Donald A. Sch"on. The Reflective Practitioner: How Professionals Think in Action. Basic Books, New York, NY, 1983. ISBN 9780465068746.

[24] Linfang Shang, Ming Xu, Yiding Sun, Tianle Xia, Lingxiang Hu, Lan Xu, and Ning Zheng. When self-evolution backfires: Pre-commit gating against skill contamination in llm agents. arXiv preprint arXiv:2608.05810, 2026. URL https://arxiv.org/abs/2608.05810.

[25] Jiajun Shi, Jian Yang, Jiaheng Liu, Xingyuan Bu, Jiangjie Chen, Junting Zhou, Kaijing Ma, Zhoufutu Wen, Bingli Wang, Yancheng He, Liang Song, Hualei Zhu, Shilong Li, Xingjian Wang, Wei Zhang, Ruibin Yuan, Yifan Yao, Wenjun Yang, Yunli Wang, Siyuan Fang, Siyu Yuan, Qianyu He, Xiangru Tang, Yingshui Tan, Wangchunshu Zhou, Zhaoxiang Zhang, Zhoujun Li, Wenhao Huang, and Ge Zhang. KORGym: A Dynamic Game Platform for LLM Reasoning Evaluation. arXiv e-prints, art. arXiv:2505.14552, May 2025. doi: 10.48550/arXiv.2505.14552.

[26] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, volume 36, pages 8634–8652, 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/ 1b44b878bb782e6954cd888628510e90-Abstract-Conference.html.

[27] Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. ALFWorld: Aligning text and embodied environments for interactive learning. In The Ninth International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=0IOX0YcCdTn

[28] Sijun Tan, Siyuan Zhuang, Kyle Montgomery, William Y. Tang, Alejandro Cuadron, Chenguang Wang, Raluca Ada Popa, and Ion Stoica. JudgeBench: A benchmark for evaluating LLM-based judges. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=G0dksFayVq.

[29] The New Stack. Why OpenAI and Anthropic are hiring forward deployed engineer teams. https://thenewstack. io/forward-deployed-engineers-ai/, 2026.

[30] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023. URL https://arxiv.org/abs/2305.16291.

[31] Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, and Prithviraj Ammanabrolu. ScienceWorld: Is your agent smarter than a 5th grader? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11279–11298, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.emnlp-main.775. URL https://aclanthology.org/2022.emnlp-main.775/.

[32] Wenxiao Wang, Priyatham Kattakinda, and Soheil Feizi. Do agent optimizers compound? a continual-learning evaluation on terminal-bench 2.0. arXiv preprint arXiv:2607.14004, 2026. URL https://arxiv.org/abs/2607. 14004.

[33] Zefeng Wang, Minxi Yan, Jinhe Bi, Sikuan Yan, Volker Tresp, and Yunpu Ma. Metaskill-evolve: Recursive self-improvement of llm agents via two-timescale meta-skill evolution. arXiv preprint arXiv:2607.05297, 2026. URL https://arxiv.org/abs/2607.05297.

[34] Shuhan Xue, Zixin Ding, Yichen Shen, Yinjie Wang, Zhenfei Yin, Yingcheng Wu, Yuxin Chen, Mengdi Wang, and Ling Yang. Past-bench: Benchmarking the foundations of recursive self-improvement in personal agents. arXiv preprint arXiv:2608.04003, 2026. URL https://arxiv.org/abs/2608.04003.

[35] Qinyuan Ye, Yu Li, Yada Pruksachatkun, Jiaxin Zhang, and Chien-Sheng Wu. On the fragility of self-improving agents: Variance, task order, and underspecification. arXiv preprint arXiv:2608.18066, 2026. URL https: //arxiv.org/abs/2608.18066.

[36] Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, and Jason E. Weston. Self-rewarding language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 57905–57923. PMLR, 2024. URL https://proceedings.mlr.press/v235/yuan24d.html.

[37] Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah D. Goodman. STaR: Bootstrapping reasoning with reasoning. In Advances in Neural Information Processing Systems, volume 35, pages 15476–15488, 2022. URL https: //arxiv.org/abs/2203.14465.

[38] Haiyue Zhang. Credit without ground truth: Auditing step-level credit assignment in llm agents against executed replay. arXiv preprint arXiv:2608.19760, 2026. URL https://arxiv.org/abs/2608.19760.

[39] Xiaoying Zhang, Zichen Liu, Yipeng Zhang, Xia Hu, and Wenqi Shao. Retroagent: From solving to evolving via retrospective dual intrinsic feedback. arXiv preprint arXiv:2603.08561, 2026. URL https://arxiv.org/abs/ 2603.08561.

[40] Congjie Zheng, Chuanyi Xue, Bin Liang, Jun Yang, and Changshui Zhang. Seagym: An evaluation environment for self-evolving llm agents. arXiv preprint arXiv:2606.17546, 2026. URL https://arxiv.org/abs/2606.17546.

[41] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, volume 36, pages 46595–46623, 2023. URL https://arxiv.org/abs/2306.05685.

At step 0 of each episode, Summary memory first makes a separate compression call over the game rule and   
truncated previous-episode history:   
Rule:   
<GAME SYSTEM PROMPT>   
History:   
<TRUNCATED RAW HISTORY FROM PREVIOUS EPISODES>   
Please summarize practical experience and recommendations ...   
Tips: <BULLET-POINTED STRATEGY SUMMARY>   
The extracted Tips are cached for the session and reused at later steps. The action-time input is then   
Hint Messages:   
<SUMMARIZED TIPS>   
<GAME SYSTEM PROMPT>   
Game history of current episode:   
<RAW HISTORY FROM EARLIER STEPS IN THIS EPISODE>   
<GAME SYSTEM PROMPT + CURRENT STEP PROMPT>   
Thus, only cross-episode experience is compressed; the current episode remains available as raw state–action–   
score history. In the released implementation, the game system prompt appears once with the cached summary   
prefix and again with the current step prompt.

## Appendix

## A Game Prompt Templates

This appendix documents the canonical prompts used for the seven games in the main experiments. The templates are transcribed from test/S3GYM/envs/\*/prompt.py. We normalize whitespace and replace episodespecific values with angle-bracket placeholders (e.g., <BOARD>); the game rules, action spaces, coordinate conventions, and required answer syntax follow the implementation. At inference time, the game-specific system prompt is concatenated with the current step prompt and, for memory-based agents, the available Hint Messages.

## A.1 Shared Interaction Protocol

Six games prepend the shared base instruction described below, whereas Trust Evolution uses its own gamespecific system instruction. Both memory conditions retain the same game rules, dynamic state, legal action space, and self-judging contract. They difer only in how experience from previous exploration episodes is represented before the current state.

Direct memory: raw-history in-context learning

Direct memory serializes every retained step from previous episodes as an environment–agent pair, Env: <STATE> followed by LLM: Answer: <ACTION>, Score: <SELF-SCORE>. At action time, the model receives the following composition:

Hint Messages:   
<RAW HISTORY FROM PREVIOUS EPISODES>   
Game history of current episode:   
<RAW HISTORY FROM EARLIER STEPS IN THIS EPISODE>   
<GAME SYSTEM PROMPT + CURRENT STEP PROMPT>

The implementation preserves the current-episode history and current game prompt in full, while truncating the tail of the previous-episode history when the context budget is exceeded.

## Summary memory: score-conditioned strategy compression

<table><tr><td>Chess-style state prediction</td></tr><tr><td>System instruction. You are playing a step-by-step piece-position prediction game on an N × N board. Empty cells are denoted by ., pieces are denoted by letters, coordinates are zero-based, and every piece follows its own hidden movement rule. Predict the next position of as many pieces as possible from the observed board-state sequence. Dynamic state. Current Game State: followed by &lt;B0ARD&gt;. Decision instruction. Predict the next-step location of each recoverable piece. Use the syntax A:(x,y), b:(x,y), c: (x,y), .. .. The agent may return END when it elects to stop.</td></tr><tr><td>Answer: A:(&lt;ROW&gt;,&lt;COL&gt;), b:(&lt;ROW&gt;,&lt;COL&gt;), ..., Score: &lt;SELF-JUDGED SCORE&gt;.</td></tr><tr><td>Minesweeper</td></tr><tr><td>System instruction. You are playing Minesweeper on a &lt;BOARD_ROWS&gt; × &lt;BOARD_COLS&gt; board with &lt;MINES&gt; hidden mines. Unknown cells are ?, flags are F, revealed counts are 0-8, and revealed mines are X. Coordinates are zero-based. Win by uncovering every safe cell or correctly flagging every mine; the episode ends at &lt;MAx_EPOCHS&gt; or when a mine is uncovered,</td></tr><tr><td>depending on the evaluation mode.</td></tr><tr><td>Dynamic state. Current Game State: followed by the partially observed &lt;B0ARD&gt;. Decision instruction. Choose exactly one of uncover (r,c), flag (r,c), or unflag (r,c). The score is the ratio of</td></tr><tr><td>correctly flagged mines to the total number of mines. Answer: uncover (&lt;ROW&gt;,&lt;COL&gt;), Score: &lt;SELF-JUDGED SCORE&gt;.</td></tr></table>

Regardless of memory condition, the model must report both an action and its self-judged immediate score. The shared final-line contract is

<table><tr><td>Shared response contract</td></tr><tr><td>Answer: &lt;ACTION&gt;, Score: &lt;SELF-JUDGED SCORE&gt;. If the agent determines that the episode has terminated, it returns Answer: END, Score: 0.</td></tr></table>

The self-judged score is intended to be the reward obtained by the proposed action under the stated game rules, rather than a free-form confidence value. Table 8 summarizes the observation and action interfaces before the full templates are presented.

<table><tr><td>Game</td><td>Main dynamic observation</td><td>Valid action form</td></tr><tr><td>Chess</td><td>N × N board with labeled pieces and empty cells Coordinates for one or more labeled</td><td>pieces</td></tr><tr><td>Minesweeper</td><td>Partially revealed grid with numbers, flags, and uncover, flag, or unflag unknown cells</td><td></td></tr><tr><td>Nullify</td><td>Indexed list of signed values and arithmetic oper- Two unit indices ators</td><td></td></tr><tr><td>Tetris Snake</td><td>Fixed-cell board and next-block rendering Grid containing walls, head, body, food, and ob- LEFT/RIGHT/UP/DOWN</td><td>Drop column and rotation angle</td></tr><tr><td>Plants-vs-Zombies</td><td>stacles Battlefield, sun budget, plants, zombies, and turn Zero or more plant placements</td><td></td></tr><tr><td></td><td>state</td><td></td></tr><tr><td>Trust Evolution</td><td>Interaction round and action/payoff history</td><td>collaborate or cheat</td></tr></table>

Table 8 Observation and action interfaces exposed by the canonical game prompts.

## A.2 State Inference and Deduction

## A.3 Symbolic and Spatial Planning

Nullify

System instruction. The board contains indexed independent units. Signed numbers (+n or -n) can be combined with another unit; available operators include multiplication, division, square root, square, reciprocal, floor, and ceiling. Combining equal-magnitude values with opposite signs produces zero and removes both units. The objective is to eliminate all units, yielding a final value of zero.

Dynamic state. Current game board: followed by one index,content pair per unit in <INDEXED UNITS>. Unit indices start at zero and are updated after each combination.

Decision instruction. Select the two indices to combine on the current turn. A successful complete cancellation receives score 1; otherwise the terminal score is 0.

Answer: <INDEX I> <INDEX J>, Score: <SELF-JUDGED SCORE>.

## Tetris

System instruction. You are playing simplified Tetris on a <BOARD\_WIDTH> × <BOARD\_HEIGHT> board. The board shows fixed blocks as \* and empty cells as .; the next-block panel uses #. A placed block falls until obstructed, each completed row is cleared for one point, and the episode terminates when a falling block extends above the board.

Dynamic state. Current Game State: <BOARD> and Next Block: <NEXT\_BLOCK>.

Decision instruction. Choose the one-based drop column of the block’s leftmost cell and a clockwise rotation in {0, 90, 180, 270}. The objective is to maximize cleared rows.

Answer: <DROP COLUMN> <ROTATION ANGLE>, Score: <SELF-JUDGED SCORE>.

## A.4 Reactive Control and Strategic Interaction

## Snake

System instruction. You control a snake on a <GRID\_WIDTH> × <GRID\_HEIGHT> grid. Walls are #, the head is H, body cells are S, apples are A, and obstacles are X. Eating an apple increases both length and score by one. A wall, self, obstacle, or invalid-action collision terminates the game under strict rules. The snake cannot reverse direction and the episode is capped at <MAX\_STEPS>.

Dynamic state. Current Game State: followed by <BOARD> and the current movement direction.

Decision instruction. Choose exactly one next movement from LEFT, RIGHT, UP, or DOWN.

Answer: <LEFT|RIGHT|UP|DOWN>, Score: <SELF-JUDGED SCORE>.

## Plants-vs-Zombies

System instruction. You are playing simplified Plants-vs-Zombies on a <ROWS> × <COLS> grid. Each turn gains at least 25 sun; zombies spawn every five turns and move left, becoming stronger over time. Available plants are Sunflower (X), Peashooter (W), Three-Line Shooter (S), Wall-nut (J), Torch Stump (H), and Fire Chili (F), each with the costs and efects specified in the system prompt. The score increases by one per survived turn; the game ends when a zombie reaches column −1 or after <MAX\_STEPS>.

Dynamic state. Current Game State: followed by <BATTLEFIELD>, including the turn, sun budget, plant layout, and zombie states.

Decision instruction. Place zero or more plants for the next turn. Each placement is PlantType Row Col; multiple placements are separated by semicolons. Rows and columns are zero-based, and an empty action is allowed.

Answer: X <ROW> <COL>; W <ROW> <COL>, Score: <SELF-JUDGED SCORE>.

## Trust Evolution

System instruction. You play a fixed-length repeated interaction against an opponent with a stable within-episode strategy. The two actions are collaborate and cheat. Payofs to the agent are +2 for mutual collaboration, −1 for collaborating against cheating, +3 for cheating against collaboration, and 0 for mutual cheating.

Dynamic state. Current Game State: followed by <ROUND AND INTERACTION HISTORY> for an episode of <MAX\_STEPS> rounds.

Decision instruction. Infer the opponent’s behavior from the observed interaction and select exactly one action for the next round.

Answer: <collaborate|cheat>, Score: <SELF-JUDGED SCORE>.