# ChunkAct: Action Chunking for Agentic Robotics

Bingxin Xu<sup>1</sup> Wenrui Bao<sup>2</sup> Xinxin Liu<sup>2</sup> Jiawei Zhou<sup>3</sup> Yuzhang Shang<sup>2</sup> Emilio Ferrara<sup>1</sup> <sup>1</sup>University of Southern California <sup>2</sup>University of Central Florida <sup>3</sup>Stony Brook University

## Abstract

Agentic robotics places a large language model (LLM) agent in the robot control loop, where it composes perception, motion, and visionlanguage-action (VLA) tools under the guidance of a memory and thereby generalizes to new tasks without retraining. This generality comes at a steep price in efficiency. Every primitive requires a round of LLM reasoning over the full interaction history, so the cost of an episode grows with the number of primitives. We find this reasoning largely redundant: once a sub-goal is chosen, its primitives follow a routine already prescribed by memory. We present ChunkAct, which introduces action chunking to agentic robotics. Online, a single call launches a chunk of chained primitives, which executes autonomously and returns control only upon failure. Offline, ChunkAct compiles the memory into these chunks, so execution is memoryfree at test time and the planner is reserved for decisions that require reasoning. LLM computation therefore scales with the number of decisions rather than the number of primitives. On LIBERO-Pro, ChunkAct improves task success by 5.0 points over the state of the art in agentic robotics while requiring 2.9× fewer LLM calls, 4.4× fewer input tokens, and 2.3× fewer output tokens.

## 1 Introduction

Agentic robotics has emerged as a new paradigm for robot manipulation (Liang et al., 2023; Kumar, 2026; Fu et al., 2026; Tsui et al., 2026; Zhang et al., 2026c; Jia et al., 2026; Xu et al., 2026b,a). Rather than training a policy that maps pixels and language directly to motor commands (Brohan et al., 2023a; Kim et al., 2024; Black et al., 2024), it places a large language model (LLM) agent in the control loop and equips it with perception, motion, and vision-language-action (VLA) tools, together with a memory of what has worked before. At each step, the agent observes the scene, consults the memory, reasons, and issues a single tool call (Figure 1, top). Because each task is composed by reasoning rather than learned from demonstrations, the agent handles new tasks without retraining and succeeds where end-to-end policies break down. For example, with no robot-specific training, GPT-6 Astra driven through a generic agent harness completes a real-arm pick-and-place task in 19 of 20 trials (OpenAI, 2026; Menon et al., 2026) and ranks above every public policy on the RoboDojo benchmark (Zhang et al., 2026b), which illustrates the generalization this paradigm affords.

However, this generalization comes at a steep price in efficiency. The planner acts through primitives, i.e., robot skills invoked as tool calls, each of which requires a dedicated round of LLM reasoning. Every such round reprocesses the full interaction history, and reasoning models further consume output tokens before responding (Sui et al., 2025). Consequently, the number of calls and output tokens scales linearly with the number of primitives in an episode, while input tokens scale quadratically (Kapoor et al., 2024; Kim et al., 2026a), as illustrated in Fig.1 Top. For instance, a single pickand-place task requires 24.2 LLM calls and 2.2M input tokens on average, costing about \$22 per episode with GPT-6 Astra and still \$11 even with the cheaper GPT-5.5.<sup>1</sup> As tasks grow longer, this overhead quickly dominates the cost of execution, hindering the scalability of LLM-based agents.

To address this inefficiency, we analyze the agent’s behavior and find that LLM reasoning is frequently redundant. Once a sub-goal is selected and its targets are localized, the subsequent primitives follow a fixed routine already encoded in memory, such that each additional call merely reproduces a decision that has already been made. Although a single read of memory suffices to capture this routine, the LLM tends to deliberate over each step and constantly returns to memory for guidance. Furthermore, the memory retrieval and LLM reasoning become less reliable as the context grows (Liu et al., 2024; Laban et al., 2026). This overhead could therefore be eliminated by compiling memory into generalizable action sequences that execute without the planner’s step-by-step involvement. This concept also echoes the action chunk (Zhao et al., 2023) in robotics.

![](images/f8eb24eb4e4bd52cc790458298e9622e16343b7181f782d72fbb6048aed13d19.jpg)  
Figure 1: ChunkAct introduces action chunking to agentic robotics, improving both task success and efficiency. Top: prior methods invoke the LLM planner once per primitive, so the number of LLM calls and the context length grow with every step. Bottom: ChunkAct invokes the planner once to launch an action chunk, which executes autonomously and returns a typed error only upon failure. Right: ChunkAct gains both task success and efficiency.

Building on this observation, we propose ChunkAct (Action Chunking for Agentic Robotics), which consists of two complementary components: chunked execution at test time and memory internalization offline. At test time, ChunkAct introduces action chunking to agentic robotics. It composes the primitives that realize a sub-goal into an executable chunk, which the planner invokes with a single call and which then runs to completion autonomously. This design confines the LLM to where reasoning is genuinely required: understanding the physical scene, diagnosing failures, and redirecting the plan. To support this, a failed chunk returns a typed exit that specifies the failed primitive along with an error log, providing the planner with structured feedback for recovery. In this way, LLM computation scales with the number of decisions rather than the number of primitives (See Fig.1 Bottom). Offline, ChunkAct builds these chunks as generalizable routines with memory compiled into them. The parameters and procedural knowledge that govern each chunk are compiled from memory before deployment, making its actions inherently memory-conditioned. As a result, chunks run memory-free at test time, eliminating frequent memory retrieval and further improving the efficiency of the LLM.

Together, these two components improve both efficiency and performance. By substantially reducing the number of LLM calls and the length of the context, ChunkAct lowers the overall cost of each episode. Meanwhile, because the planner reasons over a concise context and memory is internalized throughout execution, ChunkAct also achieves higher task success rates (See Fig.1 Right). To summarize, our contributions are threefold.

• To our knowledge, ChunkAct is the first LLM agent for robot manipulation that commits to a chunk of chained perception and control primitives per call rather than to a single primitive.

• ChunkAct compiles memory offline into generalizable action chunks, enabling memory-free execution at test time and reserving the LLM for decisions that genuinely require reasoning.

• Compared with the previous SOTA agentic robotics approach, ChunkAct improves success by 5.0 points on LIBERO-Pro, while requiring 2.9× fewer LLM calls, 4.4× fewer input tokens, and 2.3× fewer output tokens. It also exhibits strong zero-shot transfer capability.

## 2 Related Work

Agentic robotics. Language models entered manipulation as planners that sequence a fixed repertoire of skills, grounded in affordances and closed by textual feedback (Brohan et al., 2023b; Huang et al., 2022). Code as Policies turned the plan into a program over perception and control interfaces (Liang et al., 2023), and later work widened what the program could express with structured prompts, foundation perception, and value maps (Singh et al., 2022; Huang et al., 2023a,b; Mu et al., 2024). A parallel line places a reasoning model above a learned policy rather than around it (Belkhale et al., 2024; Shi et al., 2025; Physical Intelligence et al., 2025). The two lines have converged in agentic robotics, where an LLM agent sits in the control loop, calls perception, motion, and VLA tools, and keeps a memory of what has worked (Fu et al., 2026; Tsui et al., 2026; Kumar, 2026; Jia et al., 2026; Zhang et al., 2026c; Xu et al., 2026b,a). All of this work is measured by what the agent can accomplish, and none of it addresses the cost of consulting the planner before every primitive, which has been approached only by caching plans for tasks seen before (Zhang et al., 2026a; Kim et al., 2026a). To our knowledge, we are the first to improve efficiency in agentic robotics systems by consolidating recurring call sequences into reusable action routines.

Action chunking in robotics. How many actions a policy commits to per decision is a design choice that trades reactivity against consistency and cost. Policies that query the model once per control step (Brohan et al., 2022, 2023a; Kim et al., 2024) tie inference cost to control frequency and let errors compound between queries. Action chunking, introduced by ACT (Zhao et al., 2023) and adopted by Diffusion Policy (Chi et al., 2025), predicts a short sequence of actions per inference, and VLA models have since built it into their action heads and tokenizers (Black et al., 2024; Kim et al., 2025; Pertsch et al., 2025). Subsequent work tunes the trade-off by interleaving chunks for closed-loop reactivity (Liu et al., 2025; Black et al., 2026), and world-action models extend chunked prediction from actions to future observations (Du et al., 2023; Wu et al., 2024; Kim et al., 2026b). In all of these, a chunk is a short sequence of motor commands from a learned policy, and how much an LLM agent should commit to per decision has been asked only in text environments (Yang et al., 2026). To our knowledge, we are the first to explore action chunking in agentic robotics, where a chunk spans perception, control primitives, and internalized memory rather than motor commands.

Memory in agentic robotics. Memory lets an LLM agent improve without weight updates, and its forms differ in what is stored rather than in how it is used. Reflective memory stores trajectories or the lessons drawn from them and returns relevant entries to the prompt (Park et al., 2023; Shinn et al., 2023; Zhao et al., 2024; Packer et al., 2023). Procedural memory stores what worked as code, from skill libraries (Wang et al., 2023) to induced workflows (Wang et al., 2024) and robot programs adapted to new contexts (Kagaya et al., 2025; Woods and Seifi, 2026). Robot harnesses keep rule cards and task records written by the planner itself (Zhang et al., 2026c), transition conditions between subtasks (Xu et al., 2026b), or a demonstration video navigated at a chosen granularity (Anonymous, 2026). In every form, the planner decides what to retrieve and when, reads it into context, and interprets it before acting, so the memory is consumed at test time and its value is bounded by the planner’s ability to use it (Zhang et al., 2025). Li et al. (2025) comes closest to breaking this pattern by distilling recurring spans into a datastore that is emitted rather than read, yet its chunks remain text. We are the first work to explore compiling agentic memory offline into physical action, producing executable action chunks that require no memory access at deployment.

## 3 Action Chunking for Agentic Robotics

## 3.1 Preliminaries

We begin by formalizing agentic robotics, the setting in which ChunkAct operates.

The agentic loop. An agentic system solves a manipulation task with an LLM planner $\mathcal { G }$ that sequentially composes primitives from a library $\mathcal { P }$

$$
\begin{array} { r } { a _ { t } = ( p _ { t } , \theta _ { t } ) = \mathcal { G } ( c _ { t } ) , \qquad y _ { t } = p _ { t } ( \theta _ { t } ) , } \\ { c _ { t + 1 } = c _ { t } \oplus ( a _ { t } , y _ { t } ) . \qquad } \end{array}\tag{1}
$$

Here $c _ { t }$ denotes the context window of the planner at step $t ,$ initialized with the task instruction as $c _ { 0 } = \ell .$ . At each step, the planner reasons over $c _ { t }$ and outputs $a _ { t } .$ , which specifies the primitive to execute, $p _ { t } ~ \in ~ \mathcal { P }$ , and its arguments $\theta _ { t }$ . The primitive runs to completion and returns a result $y _ { t }$ , which is appended to the context with $a _ { t }$ . The process repeats until the task is completed at T.

Notably, not every planner output is a robot action. Action primitives alter the physical state of the robot, whereas informational primitives leave the scene intact and only return content, either from perception or from a memory M (Section B). Since every result is appended to $c _ { t }$ , an informational call is paid for not only once, but again as input at every subsequent step.

![](images/26d8c56dd3bf06c07cec3db9127a1342c0fb8d979759e61a62be52a9dcae7c8b.jpg)  
Figure 2: Distribution of LLM planner’s tool calls on LIBERO. Memory reads and perception dominate.

As the planner re-reads its entire context at every call, a T-step episode costs T planner calls and

$$
N _ { \mathrm { i n } } = \sum _ { t = 0 } ^ { T - 1 } \left| c _ { t } \right| = T \left| c _ { 0 } \right| + \sum _ { s = 0 } ^ { T - 2 } ( T - 1 - s ) \left| ( a _ { s } , y _ { s } ) \right|\tag{2}
$$

input tokens, where | · | counts tokens. Each call reasons before it answers, so calls and output tokens grow linearly in T, while input tokens grow quadratically. Equation (2) identifies two quantities that govern the cost: the number of steps T, and the length of each appended pair $( a _ { s } , y _ { s } )$ , which is paid again at every later step. For instance, Harness VLA takes on average 24.2 planner calls and 2.2M input tokens per episode (see Table 2).

We analyze the planner’s tool calls and reveal two sources of inefficiency. ❶ The planner conflates deciding with executing. Every primitive incurs a full round of reasoning, even though, once a sub-goal is fixed, the primitives that realize it are largely determined by the memory; the planner is thus invoked far more often than the task demands decisions. ❷ The planner spends most of its calls gathering information rather than acting. Only 20.5% of calls invoke an action primitive, while memory reads and perception account for 31.7% and 44.3%, respectively (See Fig.2). Since M is static across episodes, these reads retrieve the same knowledge again and again, inflating the context without informing new decisions. These two findings motivate ChunkAct, which brings action chunking to agentic robotics through offline action chunk building (Section 3.2) and online action chunk triggering (Section 3.3).

## 3.2 Offline action chunk building

Targeting the ❶ issue, ChunkAct outputs action chunks rather than single primitives. The planner is reengaged only when the chunk fails, reserving LLM reasoning for the decisions that require it.

The memory of an LLM agent already records the general, reusable success rules and and procedural context for tasks. ChunkAct compiles all memory once into executable chunk form rather than letting the planner re-derive it at every step (Figure 3(a)). Memory construction begins with a partition over task categories. A manipulation instruction decomposes into an ordered sequence of typed sub-goals, and the space of such types remains small even as the task set grows. For example, two types suffice to cover a broad range of tabletop manipulation: transport, which relocates an object onto or into a receptacle, and fixture, which actuates an articulated element of the scene. All memory entries pertaining to a given type, from general execution rules to task-specific records, are therefore consolidated into a single source and compiled into a unified chunk, parameterized by the object classes on which it operates.

Compilation retains only the procedural knowledge that defines how a task type is executed: the ordering of stages, the primitive assigned to each stage, the stage-level parameters conditioned on object class, the fallback strategies to attempt when a stage fails, and the criteria used to verify each stage. The primitive assignment, for instance, preserves the division of labor recorded in memory, delegating contact-rich stages such as grasping or fixture actuation to the VLA policy and the remaining stages to analytic primitives. Conversely, any quantity unobservable at deployment time is excluded. Recorded coordinates and privileged simulator state are discarded, and all geometric quantities are re-estimated from live perception of the current scene. This decoupling allows a chunk to transfer to scenes whose layouts differ from those encountered during memory construction.

The resulting chunk is an executable routine that captures the recurring procedure of its task type. A single chunk covers every task of its type, with class-dependent content resolved by lookup within the chunk. The planner is thus relieved of both memory retrieval and step-by-step primitive selection at run time. During execution, each stage is verified before the next begins, and the library is compiled once and kept frozen.

## 3.3 Online action chunk triggering

To sovle the ❷ problem, ChunkAct internalizes the memory M into executable chunks, allowing memory-free execution at run time.

At test time the planner invokes chunks in place of primitives, through a single tool call (Figure 3(b) labels ChunkAct). The instruction is parsed into its ordered sequence of typed sub-goals, each subgoal selects the chunk of its kind, and the planner’s interface to execution narrows to two operations, grounding a target in the current observation and launching the chunks. An episode therefore begins with the planner interpreting the task and the scene, grounding the first target, and issuing one call, after which the chunks run the sub-goals until the task is complete without any further request to the planner. No memory is read along the way, since everything the planner would have retrieved has been compiled into the chunks, and the planner’s context holds only the instruction, the observations it has requested, and the exits it has received.

![](images/06d39f66515963f3d47fb49f0acbb4e8e9cc3c1dd5ba2fe9fc021c095b346741.jpg)  
Figure 3: The two stages of ChunkAct. Left: offline, ChunkAct internalizes memory by compiling it into executable action chunks. Right: online, the LLM planner invokes ChunkAct via a single function call, which executes an entire action chunk autonomously and returns a typed error code to the planner only upon failure.

Chaining the stages under one call, rather than exposing each stage as a separate call, follows from where reasoning is actually needed. Compiling memory and perception into chunks removes the cost of retrieval and repeated observation, but a per-stage interface would still return control to the planner after every action, turning each step of execution into a round of deliberation. Most of these rounds involve no decision.

Once its target is grounded, the stages of a subgoal follow a fixed order, so a successful stage already determines the next one and requires no reasoning to select it. A chunk therefore chains the stages directly and returns control to the planner only when a stage fails.

Control returns to the planner only when a chunk fails, and it does so through a typed exit. Every stage of a chunk raises its own error code, so the exit names the stage that failed and the reason it failed, such as a perception call that cannot locate its target, a policy call that loses the object, or a verification that cannot confirm the sub-goal, and it carries the error log with the robot state and fresh observations (Section A). The planner then reperceives the scene from its true state, and chooses among three responses, namely grounding the target again and re-invoking the chunk, redoing a sub-goal whose verification failed, or falling back to the primitives for a situation the library does not cover. Each response is a decision that requires understanding the task and the scene, and none is a step of execution, so the planner is consulted at decision points and nowhere else.

## 3.4 Role of LLM planner in agentic robotics

We examine how this design changes the role of the planner, from stepwise executor to sub-goal selector and failure handler.

Taken together, the two stages redefine the role of the planner. In the per-primitive loop of Equation (1), the planner serves as both reasoner and executor, treating every step alike regardless of whether it requires judgment. ChunkAct separates the two: execution is delegated to the chunks, while the planner retains only what cannot be compiled, namely understanding the task and scene and responding to failure. This separation reshapes both sides of the loop.

On the input side, the dominant sources of context are removed. Memory is internalized in the chunks rather than retrieved online, and perception is performed within each chunk rather than requested before every primitive. The context thus stays compact across an episode, containing only the instruction, decision-time observations, and exits. On the output side, the unit of action shifts from a single primitive to a chained chunk. Each step $a _ { t }$ is either a chunk launch or a response to an exit, so the planner is invoked per decision rather than per action, and reasoning is spent only where it is required. Episode cost therefore scales with the number of decisions rather than procedure length.

Table 1: Task success on LIBERO-Pro. Task success rate (%) under instruction redirection (T) and position swap (S) for each suite. Harness VLA and ChunkAct both use GPT-5.6-sol with high thinking. Best entry is in bold.
<table><tr><td>Method</td><td></td><td>Spat-T</td><td>Spat-S</td><td>Obj-T</td><td> $\mathbf { O b j - S }$ </td><td>Goal-T</td><td>Goal-S</td><td>L10-T</td><td>L10-S</td><td>Overall</td></tr><tr><td rowspan="9">VTA</td><td>OpenVLA (Kim et al., 2024)</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>π0 (Black et al., 2024)</td><td>0.0</td><td>0.0</td><td>0.0</td><td>2.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.3</td></tr><tr><td>π0.5 (Physical Intelligence et al., 2025)</td><td>1.0</td><td>20.0</td><td>1.0</td><td>17.0</td><td>2.0</td><td>38.0</td><td>1.0</td><td>8.0</td><td>11.0</td></tr><tr><td>MolmoAct (Lee et al., 2026)</td><td>0.0</td><td>0.0</td><td>0.0</td><td>6.0</td><td>0.0</td><td>0.0</td><td>6.0</td><td>0.0</td><td>1.5</td></tr><tr><td>NORA (Hung et al., 2025)</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>X-VLA (Zheng et al., 2026)</td><td>0.0</td><td>0.0</td><td>8.0</td><td>2.0</td><td>9.0</td><td>1.0</td><td>10.0</td><td>0.0</td><td>3.8</td></tr><tr><td>AtomVLA (Sun et al., 2026)</td><td>1.0</td><td>16.0</td><td>0.0</td><td>10.0</td><td>11.0</td><td>2.0</td><td>9.0</td><td>1.0</td><td>6.3</td></tr><tr><td>πRLinf (Yu et al., 2026)</td><td>42.0</td><td>59.0</td><td>71.0</td><td>78.0</td><td>45.0</td><td>42.0</td><td>49.0</td><td>14.0</td><td>50.0</td></tr><tr><td>CaP-X (Fu et al., 2026)</td><td>14.0</td><td>12.0</td><td>18.0</td><td>22.0</td><td>17.0</td><td>26.0</td><td>1</td><td>1</td><td>18.2</td></tr><tr><td rowspan="2">gent A</td><td>Harness VLA (Zhang et al., 2026c)</td><td>100.0</td><td>80.0</td><td>80.0</td><td>100.0</td><td>70.0</td><td>80.0</td><td>60.0</td><td>60.0</td><td>78.8</td></tr><tr><td>+ ChunkAct (ours)</td><td>94.0</td><td>82.0</td><td>98.0</td><td>100.0</td><td>80.0</td><td>86.0</td><td>60.0</td><td>70.0</td><td>83.8</td></tr></table>

Table 2: Efficiency analysis and LLM role ablation on LIBERO-Pro. Average TSR (%), planner calls, input tokens (M), and output tokens (k) with GPT-5.6-sol in two reasoning settings. Green rows show the gain over Harness VLA. Last row runs chunks with no LLM call.
<table><tr><td>Method</td><td>Reasoning</td><td>TSR</td><td>Calls</td><td>In (M)</td><td>Out (k)</td></tr><tr><td>Harness VLA +ChunkAct</td><td>High High</td><td>78.8 83.8 +5.0</td><td>24.2 8.3 2.9×</td><td>2.2 0.5 4.4×</td><td>8.5 3.7 2.3×</td></tr><tr><td>Harness VLA None</td><td></td><td>56.0</td><td>24.9</td><td>2.4</td><td>6.4</td></tr><tr><td rowspan="3">+ChunkAct</td><td>None</td><td>80.0</td><td>6.6</td><td>0.4</td><td>1.3</td></tr><tr><td></td><td>+24.0</td><td>3.8×</td><td>6.0×</td><td>4.9×</td></tr><tr><td>No planner</td><td>57.5</td><td>1</td><td>/</td><td>1</td></tr></table>

## 4 Experiments

## 4.1 Setup

Benchmarks. We evaluate on two benchmarks from the LIBERO family. Our main experiments use LIBERO-Pro (Zhou et al., 2025), which perturbs the four LIBERO suites (Liu et al., 2023), Spatial, Object, Goal, and LIBERO-10, along objects, initial states, instructions, and environments. Following Zhang et al. (2026c), we evaluate the instruction and initial-state perturbations, denoted instruction redirection (T) and position swap (S). Each of the resulting eight suite and perturbation combinations contains 10 tasks. As in that protocol, we run 10 seeds per task, so that each combination aggregates 800 episodes. We further use LIBERO-Plus (Fei et al., 2025) benchmark, which varies the LIBERO tasks along seven axes: object layout, camera viewpoint, robot initial state, language, lighting, background, and sensor noise. As its 10,030 variants are too many to evaluate exhaustively, we run a single seed on every task under each of the seven perturbation axes, which covers the full range of perturbations at a tractable cost.

Metrics. We measure performance by the task success rate (TSR), as determined by the environment’s termination signal. We measure efficiency by the number of planner calls and the numbers of input and output tokens per episode.

Baselines. For the main experiment, we compare with eight end-to-end VLA models, Open-VLA (Kim et al., 2024), π<sub>0</sub> (Black et al., 2024), $\pi _ { 0 . 5 }$ (Physical Intelligence et al., 2025), MolmoAct (Lee et al., 2026), NORA (Hung et al., 2025), X-VLA (Zheng et al., 2026), AtomVLA (Sun et al., 2026), and the RLinf-fine-tuned $\pi _ { 0 . 5 }$ checkpoint that serves as our VLA primitive, which we denote π<sub>RLinf</sub> (Yu et al., 2026), and with CaP-X (Fu et al., 2026), an agent that writes code over perception and motion primitives without a VLA. Their results are those reported by Zhang et al. (2026c). Harness VLA and ChunkAct are run by us. For LIBERO-Plus experiment results, the VLA models performance are as reported by Fei et al. (2025).

Implementation. The planner is GPT-5.6-sol at high reasoning effort running in Codex (OpenAI, 2025), with a budget of 100 LLM calls per episode. The VLA primitive is $\pi _ { \mathrm { R L i n f } } ,$ and SAM 3 (Carion et al., 2026) provides segmentation. ChunkAct and Harness VLA (Zhang et al., 2026c) share the same set-up and the same task-specific and global memory. They differ only in how the memory is used: Harness VLA reads it at test time, whereas ChunkAct internalizes it offline into executable chunks and reads no explicit memory at test time.

Table 3: Task success on LIBERO-Plus. Task success rate (%) under all seven perturbation setting. ChunkAct runs GPT-5.6-sol at the high reasoning setting with seed 0. The best entry in each column is in bold.
<table><tr><td>Method</td><td>Camera</td><td>Robot</td><td>Language</td><td>Light</td><td>Background</td><td>Noise</td><td>Layout</td><td>Overall</td></tr><tr><td>OpenVLA (Kim et al., 2024)</td><td>1.1</td><td>4.1</td><td>26.8</td><td>4.4</td><td>25.3</td><td>19.3</td><td>31.6</td><td>16.1</td></tr><tr><td>OpenVLA-OFT (Kim et al., 2025)</td><td>59.7</td><td>37.2</td><td>81.5</td><td>85.8</td><td>92.4</td><td>76.7</td><td>77.1</td><td>72.9</td></tr><tr><td>OpenVLA-OFTw (Kim et al., 2025)</td><td>16.8</td><td>43.7</td><td>73.2</td><td>68.2</td><td>92.5</td><td>51.4</td><td>72.3</td><td>59.7</td></tr><tr><td>OpenVLA-OFTm (Kim et al., 2025)</td><td>57.9</td><td>30.6</td><td>83.6</td><td>91.6</td><td>83.6</td><td>76.3</td><td>73.2</td><td>71.0</td></tr><tr><td>π0 (Black et al., 2024)</td><td>15.8</td><td>6.6</td><td>61.0</td><td>79.6</td><td>78.5</td><td>79.4</td><td>70.4</td><td>55.9</td></tr><tr><td>π₀-FAST (Pertsch et al., 2025)</td><td>66.4</td><td>24.8</td><td>63.3</td><td>73.0</td><td>67.7</td><td>75.8</td><td>70.3</td><td>63.0</td></tr><tr><td>NORA (Hung et al., 2025)</td><td>4.0</td><td>41.1</td><td>67.0</td><td>31.0</td><td>50.5</td><td>17.6</td><td>63.9</td><td>39.3</td></tr><tr><td>WorldVLA (Cen et al., 2025)</td><td>0.3</td><td>30.2</td><td>44.2</td><td>29.4</td><td>14.5</td><td>12.2</td><td>39.4</td><td>24.3</td></tr><tr><td>UniVLA (Bu et al., 2025)</td><td>4.3</td><td>50.3</td><td>71.8</td><td>59.1</td><td>80.0</td><td>25.3</td><td>34.3</td><td>46.4</td></tr><tr><td>RIPT-VLÀ (Tan et al., 2025)</td><td>58.3</td><td>36.7</td><td>80.1</td><td>87.9</td><td>90.4</td><td>73.8</td><td>76.5</td><td>72.0</td></tr><tr><td>+ ChunkAct (ours)</td><td>77.5</td><td>75.0</td><td>87.5</td><td>95.0</td><td>82.5</td><td>87.5</td><td>82.5</td><td>83.9</td></tr></table>

## 4.2 Main results

Task success performance. As shown in Table 1, ChunkAct achieves the highest task success rate on LIBERO-Pro, outperforming both endto-end VLAs and agentic robotic systems. It attains an overall success rate of 83.8%, exceeding the strongest prior method, Harness VLA, by 5.0 points. The perturbations in LIBERO-Pro, instruction redirection and position swap, require a policy to act on instructions and object layouts that differ from those seen in training, and end-to-end VLAs largely fail under them. This failure extends to $\pi _ { \mathrm { R L i n f } }$ , the checkpoint that ChunkAct invokes as its contact primitive, which trails ChunkAct by 43.8 points. Since the primitive alone does not generalize to these perturbations, whatever success an agent recovers must come from how it grounds the task in the current scene before the primitive is called. Placing an LLM planner above the VLA recovers much loss, as the agentic baselines show, yet Harness VLA stops at 78.8% despite sharing the same planner, primitive, perception, and memory with ChunkAct. The two systems differ only in how memory is used. Harness VLA retrieves memory at test time and relies on the planner to reinterpret past experience at every call, whereas ChunkAct compiles it offline into executable chunks that encode how each task succeeds. The gap therefore isolates the benefit of memory internalization.

Efficiency analysis. As shown in Table 2, ChunkAct attains its higher success rate at a fraction of the LLM cost. The first two rows correspond to the Harness VLA and ChunkAct entries in Table 1. Relative to Harness VLA, ChunkAct issues 2.9× fewer LLM calls and consumes 4.4× fewer input tokens and 2.3× fewer output tokens, while improving success by 5.0 points. The third and fourth rows repeat this comparison with reasoning disabled, and the efficiency advantage of ChunkAct grows further. Both observations stem from memory internalization. Harness VLA spends most LLM calls on perception queries and memory reads, and each result is appended to the context, so the context keeps growing over the episode. ChunkAct folds these steps into the executable chunks, so they run at test time without invoking the LLM and their results never enter the context window. The planner is called only at the decision points that the chunks leave open. Fewer calls over a shorter context reduce the input tokens, and fewer planning decisions reduce the output tokens.

## 4.3 Analysis of LLM planner role in ChunkAct

Analyzing the LLM planner reasoning impact. As shown in Table 2, ChunkAct remains robust when the reasoning of LLM planner is disabled, whereas Harness VLA does not. The task success rate of ChunkAct stays stable across reasoning settings, while Harness VLA degrades sharply once reasoning is turned off. This contrast reflects how heavily each system relies on the planner. Harness VLA depends on the LLM at every step: each call must interpret the retrieved memory, decide what to perceive, and issue the correct next action, so the errors of a weaker planner accumulate over the episode. ChunkAct instead decouples the LLM from the execution it does not need to perform. With memory already internalized into the chunks, the planner is consulted only for the remaining decisions, which leaves a weaker planner far fewer opportunities to fail.

Quantifying the LLM agent’s contribution. To measure how much the planner contributes within ChunkAct, the last row of Table 2 executes the chunks under a scripted driver that makes no LLM call. The chunks alone reach 57.5%, which corresponds to episodes in which the compiled routines succeed without intervention. The LLM in ChunkAct without reasoning therefore accounts for the remaining 22.5 points. This contribution comes from recovery: when a chunk fails, the planner is invoked to interpret the current scene and direct the next step, which a scripted driver cannot. The LLM thus serves not as a step-by-step executor but as a recovery mechanism invoked where the compiled routine falls short, which is why it adds substantial success at a small cost in calls and tokens.

![](images/44c1c34477369ea5b16e495889bc5346dc94fcfb6e4dfc73350623b559db665c.jpg)  
Figure 4: Demo of a successful rollout on LIBERO-Pro with ChunkAct. Given the task “pick up the can and place it in the basket,” the LLM planner first perceives the scene and invokes ChunkAct to execute a chained action sequence. The can is grasped successfully but slips during transport, so ChunkAct returns the error code. The planner interprets this feedback, re-localizes the can, and invokes ChunkAct again, which completes the task.

## 4.4 Generalization to unseen perturbations

Out-of-distribution test. The memory of ChunkAct is compiled on LIBERO-Pro and includes task-specific memory, which prior work found necessary on that benchmark (Zhang et al., 2026c). To test whether ChunkAct overfits to these tasks, we apply it unchanged to LIBERO-Plus (Fei et al., 2025), which perturbs the LIBERO tasks along seven dimensions that the memory doesn’t cover. The chunk library and planner prompt are those of Section 4.2, so the evaluation is fully zeroshot, whereas the VLA baselines, whose numbers are taken from Fei et al. (2025), are fine-tuned on the benchmark data.

Zero-shot transfer. ChunkAct achieves the best overall success rate in Table 3, exceeding the strongest prior method by 11.0 points and leading in six of the seven dimensions. The margin is widest under camera and robot initial-state perturbations, where ChunkAct stays above three quarters while the best VLA falls to two thirds and one half, respectively. Because a chunk recomputes its targets from perception in every episode, such changes alter the measured geometry but not the procedure. These results show zero-shot transfer of internalized memory to unseen perturbations rather than overfitting to the tasks it was compiled from.

## 4.5 Qualitative analysis

Figure 4 follows a single LIBERO-Pro episode and shows how the planner’s eight calls are distributed over 22 seconds of execution. The planner is engaged at two points only. At the start it views the scene and launches a chunk, which grasps and lifts the can without further LLM request. During the move, however, the can slips out of the gripper, and the chunk terminates with a typed exit. It returns a hand-off block, a structured error report that adds context to the error code. The planner resumes from the true state of the scene rather than from its expectation of it, spends four calls re-pointing the can, and triggers ChunkAct again, which carries the can to the basket, places it, and finishes on its own once the environment reports success. The episode makes visible the division of labor that Table 2 quantifies. The planner is reserved for understanding, while the chunks handle execution. It plans the next action chunk from the scene, and re-plans when a chunk fails.

## 5 Conclusion

Agentic robotics generalizes to novel tasks without retraining, but at a steep cost in efficiency. We identify two bottlenecks: the LLM planner reasons before every primitive, although the primitives of a sub-goal follow a recurring pattern, and most of its calls are spent on perception and memory retrieval rather than action. ChunkAct is the first work to introduce action chunking to agentic robotics, and it internalizes memory into executable chunks offline. The LLM is thereby decoupled from routine execution and engaged only where reasoning is genuinely required, which improves LLM efficiency significantly. On LIBERO-Pro, ChunkAct improves task success by 5.0 points over the SOTA method with 2.9× fewer LLM calls, 4.4× fewer input tokens, and 2.3× fewer output tokens, and it transfers zeroshot to unseen tasks on LIBERO-Plus.

## References

Anonymous. 2026. Recursive video in-context learning for agentic robot. Under review.

Suneel Belkhale, Tianli Ding, Ted Xiao, Pierre Sermanet, Quon Vuong, Jonathan Tompson, Yevgen Chebotar, Debidatta Dwibedi, and Dorsa Sadigh. 2024. RT-H: Action hierarchies using language. arXiv preprint arXiv:2403.01823.

Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. 2024. π : A vision-language-action flow model for general robot control.

Kevin Black, Manuel Y. Galliker, and Sergey Levine. 2026. Real-time execution of action chunking flow policies. In Advances in Neural Information Processing Systems.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. 2023a. RT-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818.

Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. 2022. RT-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817.

Anthony Brohan, Yevgen Chebotar, Chelsea Finn, Karol Hausman, Alexander Herzog, Daniel Ho, Julian Ibarz, Alex Irpan, Eric Jang, Ryan Julian, et al. 2023b. Do as I can, not as I say: Grounding language in robotic affordances. In Conference on Robot Learning (CoRL).

Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. 2025. UniVLA: Learning to act anywhere with task-centric latent actions. arXiv preprint arXiv:2505.06111.

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris Coll-Vinent, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, et al. 2026. Sam 3: Segment anything with concepts. In International conference on learning representations.

Jun Cen, Chaohui Yu, Hangjie Yuan, Yuming Jiang, Siteng Huang, Jiayan Guo, Xin Li, Yibing Song, Hao Luo, Fan Wang, et al. 2025. WorldVLA: Towards autoregressive action world model. arXiv preprint arXiv:2506.21539.

Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. 2025. Diffusion policy: Visuomotor policy learning via action diffusion.

Yilun Du, Sherry Yang, Bo Dai, Hanjun Dai, Ofir Nachum, Josh Tenenbaum, Dale Schuurmans, and Pieter Abbeel. 2023. Learning universal policies via text-guided video generation. In Advances in Neural Information Processing Systems.

Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, et al. 2025. LIBERO-Plus: In-depth robustness analysis of vision-language-action models. arXiv preprint arXiv:2510.13626.

Letian Fu, Justin Yu, Karim El-Refai, Ethan Kou, Haoru Xue, Huang Huang, Wenli Xiao, Guanzhi Wang, Dantong Niu, Fei-Fei Li, et al. 2026. CaP-X: A framework for benchmarking and improving coding agents for robot manipulation. arXiv preprint arXiv:2603.22435.

Siyuan Huang, Zhengkai Jiang, Hao Dong, Yu Qiao, Peng Gao, and Hongsheng Li. 2023a. Instruct2Act: Mapping multi-modality instructions to robotic actions with large language model. arXiv preprint arXiv:2305.11176.

Wenlong Huang, Chen Wang, Ruohan Zhang, Yunzhu Li, Jiajun Wu, and Li Fei-Fei. 2023b. Voxposer: Composable 3d value maps for robotic manipulation with language models. arXiv preprint arXiv:2307.05973.

Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, et al. 2022. Inner monologue: Embodied reasoning through planning with language models. arXiv preprint arXiv:2207.05608.

Chia-Yu Hung, Qi Sun, Pengfei Hong, Amir Zadeh, Chuan Li, U Tan, Navonil Majumder, Soujanya Poria, et al. 2025. NORA: A small open-sourced generalist vision language action model for embodied tasks. arXiv preprint arXiv:2504.19854.

Mengzhao Jia, Yang Lin, Xixin Zhang, Zhihan Zhang, Xiaobai Liu, and Meng Jiang. 2026. Agent as policy for robotic manipulation. arXiv preprint arXiv:2609.12541.

Tomoyuki Kagaya, Subramanian Lakshmi, Yuxuan Lou, Thong Jing Yuan, Jayashree Karlekar, Sugiri Pranata, Natsuki Murakami, Akira Kinose, and Yang You. 2025. Memory transfer planning: LLM-driven context-aware code adaptation for robot manipulation. arXiv preprint arXiv:2509.24160.

Sayash Kapoor, Benedikt Stroebl, Zachary S Siegel, Nitya Nadgir, and Arvind Narayanan. 2024. AI agents that matter. arXiv preprint arXiv:2407.01502.

Hojoon Kim, Yuheng Wu, and Thierry Tambe. 2026a. AgenticCache: Cache-driven asynchronous planning for embodied AI agents. In Proceedings of Machine Learning and Systems.

Moo Jin Kim, Chelsea Finn, and Percy Liang. 2025. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645.

Moo Jin Kim, Yihuai Gao, Tsung-Yi Lin, Yen-Chen Lin, Yunhao Ge, Grace Lam, Percy Liang, Shuran Song, Ming-Yu Liu, Chelsea Finn, et al. 2026b. Cosmos policy: Fine-tuning video models for visuomotor control and planning. arXiv preprint arXiv:2601.16163.

Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. 2024. OpenVLA: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246.

Vaishak Kumar. 2026. Act-observe-rewrite: Multimodal coding agents as in-context policy learners for robot manipulation. arXiv preprint arXiv:2603.04466.

Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, and Jennifer Neville. 2026. LLMs get lost in multi-turn conversation. In International Conference on Learning Representations (ICLR).

Jason Lee, Jiafei Duan, Haoquan Fang, Yuquan Deng, Shuo Liu, Boyang Li, Bohan Fang, Jieyu Zhang, Yi Ru Wang, Sangho Lee, et al. 2026. MolmoAct: Action reasoning models that can reason in space. In IEEE International Conference on Robotics and Automation (ICRA).

Yanhong Li, Karen Livescu, and Jiawei Zhou. 2025. Chunk-distilled language modeling. In International Conference on Learning Representations (ICLR).

Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. 2023. Code as policies: Language model programs for embodied control. In 2023 IEEE International conference on robotics and automation (ICRA).

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. 2023. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. In Advances in Neural Information Processing Systems.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions ofthe Association for Computational Linguistics.

Yuejiang Liu, Jubayer Ibn Hamid, Annie Xie, Yoonho Lee, Max Du, and Chelsea Finn. 2025. Bidirectional decoding: Improving action chunking via guided test-time sampling. In International Conference on Learning Representations.

Achu Menon, Sravanthi Machcha, Sabrina Zou, Tzu Kit Chan, and Jay Chooi. 2026. GPT-6 Astra on robotic manipulation. Robocurve evaluation report, https: //openai.robocurve.org/gpt-6-astra/.

Yao Mu, Junting Chen, Qinglong Zhang, Shoufa Chen, Qiaojun Yu, Chongjian Ge, Runjian Chen, Zhixuan Liang, Mengkang Hu, Chaofan Tao, et al. 2024. RoboCodeX: Multimodal code generation for robotic behavior synthesis. arXiv preprint arXiv:2402.16117.

OpenAI. 2025. Codex CLI. https://github.com/ openai/codex.

OpenAI. 2026. Introducing GPT-6 Astra. https:// openai.com/index/gpt-6-astra/.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2023. MemGPT: Towards LLMs as operating systems. arXiv preprint arXiv:2310.08560.

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. In Proceedings ofthe 36th Annual ACM Symposium on User Interface Software and Technology.

Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. 2025. FAST: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747.

Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, et al. 2025. π : a vision-language-action model with open-world generalization. arXiv preprint arXiv:2504.16054.

Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, James Tanner, Anna Walling, Haohuan Wang, Niccolo Fusai, et al. 2025. Hi robot: Open-ended instruction following with hierarchical vision-language-action models. arXiv preprint arXiv:2502.19417.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems.

Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, and Animesh Garg. 2022. ProgPrompt: Generating situated robot task plans using large language models. arXiv preprint arXiv:2209.11302.

Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi Yuan, Hongyi Liu, Andrew Wen, Shaochen Zhong, Na Zou, et al. 2025.

Stop overthinking: A survey on efficient reasoning for large language models. arXiv preprint arXiv:2503.16419.

Xiaoquan Sun, Zetian Xu, Chen Cao, Zonghe Liu, Yihan Sun, Jingrui Pang, Ruijian Zhang, Zhen Yang, Kang Pang, Dingxin He, et al. 2026. AtomVLA: Scalable post-training for robotic manipulation via predictive latent world models. arXiv preprint arXiv:2603.08519.

Shuhan Tan, Kairan Dou, Yue Zhao, and Philipp Krähenbühl. 2025. Interactive post-training for vision-language-action models. arXiv preprint arXiv:2505.17016.

Brian Y Tsui, Alan Y Fang, and Tiffany J Hwu. 2026. Demonstration-free robotic control via llm agents. arXiv preprint arXiv:2601.20334.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291.

Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. 2024. Agent workflow memory. arXiv preprint arXiv:2409.07429.

John Woods and Hasti Seifi. 2026. SkillComposer: Learning reusable skills for natural-language robot programming. arXiv preprint arXiv:2608.14944.

Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. 2024. Unleashing large-scale video generative pre-training for visual robot manipulation. In International Conference on Learning Representations (ICLR).

Bingxin Xu, Yuzhang Shang, Zhen Dong, and Emilio Ferrara. 2026a. Coding agents with harness for safe robot control. arXiv preprint arXiv:2609.20822.

Bingxin Xu, Yuzhang Shang, and Emilio Ferrara. 2026b. Don’t drop the BATON: Long-horizon robot manipulation via agentic subtask exploration and transitionaware memory. arXiv preprint arXiv:2608.16889.

Yanting Yang, Can Jin, Jinman Zhao, Jiahao Wu, Yang Zhou, Zhepeng Wang, Zhendong Wang, Mu Zhou, and Dimitris N. Metaxas. 2026. Act more, decide less: Skill-guided adaptive action chunking for long-horizon LLM agents. arXiv preprint arXiv:2609.02042.

Chao Yu, Yuanqing Wang, Zhen Guo, Hao Lin, Si Xu, Hongzhi Zang, Quanlu Zhang, Yongji Wu, Chunyang Zhu, Junhao Hu, et al. 2026. RLinf: Flexible and efficient large-scale reinforcement learning via macroto-micro flow transformation. In 20th USENIX Symposium on Operating Systems Design and Implementation (OSDI 26).

Qizheng Zhang, Michael Wornow, and Kunle Olukotun. 2026a. Agentic plan caching: Test-time memory for fast and cost-efficient LLM agents. In Advances in Neural Information Processing Systems.

Wenbo Zhang, Kaixuan Wang, Yutao Ouyang, Xiaoyu Huang, Liyang Li, Kailun Su, Weiyang Jin, Wenhao Chai, Haotian Liang, et al. 2026b. An unexpected robot policy: Early evaluations of GPT-6 Astra on RoboDojo and beyond. arXiv preprint arXiv:2609.24170.

Yixian Zhang, Huanming Zhang, Feng Gao, Xiao Li, Zhihao Liu, Chunyang Zhu, Jiaxing Qiu, Yuchen Yan, Jiyuan Liu, Wenhao Tang, et al. 2026c. Harness VLA: Steering frozen VLAs into reliable manipulation primitives via memory-guided agents. arXiv preprint arXiv:2607.08448.

Zeyu Zhang, Quanyu Dai, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, , Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. 2025. A survey on the memory mechanism of large language model based agents. ACM Transactions on Information Systems.

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. 2024. ExpeL: LLM agents are experiential learners. In Proceedings ofthe AAAI Conference on Artificial Intelligence.

Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. 2023. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705.

Jinliang Zheng, Jianxiong Li, Zhihao Wang, Dongxiu Liu, Xirui Kang, Yuchun Feng, Yinan Zheng, Jiayin Zou, Yilun Chen, Jia Zeng, et al. 2026. X-VLA: Softprompted transformer as scalable cross-embodiment vision-language-action model. In International Conference on Learning Representations.

Xueyang Zhou, Yangming Xu, Guiyao Tie, Yongchao Chen, Guowen Zhang, Duanfeng Chu, Pan Zhou, and Lichao Sun. 2025. LIBERO-PRO: Towards robust and fair evaluation of vision-language-action models beyond memorization. arXiv preprint arXiv:2510.03827.

Table 4: The complete taxonomy of typed exits. Each exit names the reason a chunk stopped and the reaction expected from the planner.
<table><tr><td>Group</td><td>Exit</td><td>Meaning and expected planner reaction</td></tr><tr><td>Success</td><td>terminated</td><td>The environment reported success inside the chunk; the finish is recorded and the planner session interrupted.</td></tr><tr><td></td><td>placed_subgoal</td><td>A non-final sub-goal was placed and verified; the chunk continues with the next sub-goal (internal).</td></tr><tr><td></td><td>done_covered</td><td>Every covered sub-goal was executed but the task did not terminate; re-perceive and continue.</td></tr><tr><td>Hand-off</td><td>need_pixels</td><td>The next object was not found by name, or the pixels back-projected off the workspace; call approach with pixels on it.</td></tr><tr><td></td><td>need_receptacle_pixels</td><td>The target region cannot be derived (push or beside, or the receptacle point is implausibly high); pass receptacle_pixels.</td></tr><tr><td></td><td>receptacle_rejected</td><td>The receptacle was not found by name, lies at an implausible height, or the fixture geometry was not located; pass receptacle_pixels or use primitives.</td></tr><tr><td></td><td>wrong_object_under_gripper</td><td>The wrist camera sees the nearest object more than 5 cm from the pointed spot; confirm or re-point.</td></tr><tr><td></td><td>subgoal_unverified</td><td>The object was released but perception could not confirm the predicate; check the image, then redo or continue.</td></tr><tr><td>Manipulation</td><td>grasp_failed</td><td>Up to four grasp attempts, nothing held.</td></tr><tr><td></td><td>lost_at_lift</td><td>The object slipped during the lift, twice.</td></tr><tr><td></td><td>lost_in_carry</td><td>The object slipped during the carry, leveling, pitched reach, or hover</td></tr><tr><td></td><td>lost_in_descent</td><td>re-centering. The object left the gripper while lowering.</td></tr><tr><td></td><td>carry_stalled</td><td>The arm stopped short of the point above the receptacle (reach limit).</td></tr><tr><td></td><td>delivered_unverified</td><td>The VLA carried the object away and released it; the gripper is clear.</td></tr><tr><td></td><td>placed_not_terminated</td><td>The object was carried and released at the receptacle, but the task did not terminate.</td></tr><tr><td>Fixture</td><td>open_unverified</td><td>The drawer handle did not move after the contact skill.</td></tr><tr><td></td><td>not_terminated</td><td>The close, turn-on, or door-sweep skill ran but the task did not terminate.</td></tr><tr><td>Scope</td><td>not_covered</td><td>The sub-goal or placement is not covered by any chunk; continue with</td></tr><tr><td></td><td>error</td><td>primitives. Internal error, for instance no pixel can be back-projects; continue with</td></tr></table>

## A Typed Exit Taxonomy

Table 4 lists all exit codes that a chunk may return upon failure, with its meaning and expected planner reaction.

## B Primitive Library

Table 5 lists all the primitives that constitute the library P introduced in Section 3.1, which are employed in all experiments of Section 4. ChunkAct retains this library without modification and adds one call, shown in the last row.

Table 5: The primitive library P. Primitives of Harness VLA (Zhang et al., 2026c). The last row lists the single call that ChunkAct adds to this library.

<table><tr><td>Type</td><td>Primitives</td></tr><tr><td></td><td>Perception view_env_state, segment, back_project, view_camera_meta. Return the current images and robot state, segment by text or point prompt, and back-project pixels to 3D points.</td></tr><tr><td>Motion</td><td>move_to, move_pose, set_gripper, release, rotate_pitch, rotate_wrist. These primitives are executed in closed loop by an analytic controller until the target pose or gripper state is reached.</td></tr><tr><td>VLA</td><td>pi0_pick, pi0_doubled. Drive the frozen πo.5 policy under a language command until a lift or stop condition is met.</td></tr><tr><td>Memory</td><td>read_text_file,list_dir, write_text_file. Read memory cards and guides from M and write audit notes.</td></tr><tr><td>Control</td><td>finish. Declares the task complete.</td></tr><tr><td>ChunkAct</td><td>approach, which runs an action chunk.</td></tr></table>