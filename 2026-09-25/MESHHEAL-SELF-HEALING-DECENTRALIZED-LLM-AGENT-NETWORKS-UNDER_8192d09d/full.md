# MESHHEAL: SELF-HEALING DECENTRALIZED LLM AGENT NETWORKS UNDER GRAY FAILURES

Keru Chen School of ECEE Arizona State University kchen234@asu.edu

Sen Lin Computer Science Department University of Houston slin50@central.uh.edu

Yingbin Liang Department of ECE The Ohio State University liang.889@osu.edu

Nathaniel D. Bastian Whiting School of Engineering Johns Hopkins University ndbastian@jhu.edu

Shaofeng Zou<sup>∗</sup> School of ECEE Arizona State University zou@asu.edu

## ABSTRACT

Decentralized LLM-based multi-agent systems coordinate through local interactions, but an agent can remain responsive while its task-solving quality persistently degrades. Such gray failures require protecting current tasks before sufficient evidence exists to alter future routing, while still allowing recovered agents to rejoin. We introduce MESHHEAL, a fully decentralized self-healing framework that couples ability-matched peer review across two timescales. At the fast timescale, an adaptive hierarchy escalates uncertain or low-scoring outputs from repeated single-reviewer evaluation to committee deliberation and, when needed, correction before use. At the slow timescale, a task- and ability-conditioned peer-relative detector aggregates scores to distinguish persistent degradation from ordinary output variation, trigger mandatory committee review, and eventually exclude degraded agents from ordinary routing; recovery probes provide fresh evidence for reintegration. To faithfully evaluate routing, we introduce Model-Backed MAS Evaluation, which ties ability assignments to execution models, since prompt-based ability assignments alone can leave routing errors hidden. Across BBH, MATH, and MMLU-Pro, MeshHeal achieves 0.839 degraded-phase accuracy using 51k total model tokens per task, versus the strongest baseline Symphony’s 0.807 accuracy using 115k per task. Under staggered degradation and recovery, MeshHeal isolates degraded agents, keeps them excluded from ordinary task execution until recovery, and returns them to normal routing.

## 1 INTRODUCTION

Recent advances in foundation models have enabled increasingly capable autonomous AI agents that can reason, plan, follow instructions, and use external tools (Bommasani et al., 2021). LLM-based multi-agent systems (MASs) decompose complex tasks among agents with complementary roles or abilities, combining their contributions into a final answer, often outperforming a single, larger model. However, most existing MASs coordinate in a centralized manner (Wu et al., 2023; Hong et al., 2024; Qian et al., 2024; Wang et al., 2025), which creates a communication and scaling bottleneck and also a catastrophic single point of failure.

Decentralized coordination instead allows agents to execute and route work through local interactions without relying on a central orchestrator. Accordingly, a growing body of recent work has investigated decentralized MAS architectures that distribute coordination and routing decisions across the agent network (Zhang et al., 2024; Yang et al., 2025a;b). These studies, however, largely assume that participating agents remain reliable throughout execution. In practice, individual agents may fail or degrade because of problems in their model backends, local controllers, memories, or external tools, potentially disrupting task execution across the network (Huang et al., 2025; Zhang et al., 2025).

![](images/bfc6d095a5aa6a1c8d5e8705bd0bb8da7acdfa8aa04997854f636f912696f500.jpg)  
Figure 1: Overview of MESHHEAL. (a) Agents route and complete complementary work. (b) Review by agents with the relevant ability either keeps or replaces the current output, while scores accumulated across tasks update routing eligibility and probes allow recovered agents to return. (c) Model-Backed MAS Evaluation ties execution models to declared abilities and the agent’s healthy or degraded condition, making routing errors affect accuracy.

Motivated by this gap, we study decentralized LLM-based MASs operating in the presence of agent failures. We focus on grayfailures (Huang et al., 2017; 2018), and leave other types of failures as future work. A gray failure occurs when an agent remains available and protocol-compliant, i.e., it can communicate with its neighbors and return syntactically valid responses, but its effective ability to solve some tasks is degraded relative to its nominal ability. Such a failure is “gray” because it is not directly exposed through an explicit error or loss of connectivity: the affected agent may appear healthy to some neighbors, while its degradation can only be inferred from the semantic quality of its outputs or their downstream consequences. Gray failures are particularly relevant to decentralized LLM-based multi-agent systems because an agent’s availability does not necessarily imply its semantic reliability. We do not regard every incorrect LLM response as a gray failure, since even a healthy agent has a nonzero probability of error. Instead, a gray failure refers to a sustained degradation relative to the agent’s nominal task solving capability. Such degradation may arise from transient inference-serving issues, context corruption or overflow, stale memory, failures of external tools, or changes in the model backend. These failures are difficult to detect using conventional heartbeat- or connectivity-based mechanisms, since the affected agent continues to participate normally in the communication protocol.

We refer to the system-level capability to respond to such degradation as self-healing: detecting agent degradation, limiting its effect on current task execution, adapting future routing, and reintegrating an agent after recovery (Psaier & Dustdar, 2011). Existing LLM inference and agent-coordination methods address complementary parts of this objective. Methods based on repeated inference, deliberation, or robust aggregation improve current outputs (Wang et al., 2023; Du et al., 2024; Li et al., 2024; Wang et al., 2025; Jo & Park, 2025; Zheng et al., 2026; Lee et al., 2026), however, because they operate within individual tasks, these methods are not designed to retain producerspecific evidence over time and therefore do not support persistent-degradation detection, subsequent rerouting, or recovery-aware reintegration. Moreover, history-based methods adapt routing across tasks (Li et al., 2026; Yang et al., 2025b); however, routing adaptation alone acts only after history has accumulated, so it cannot protect the current task, and once an agent is bypassed, the system receives no fresh evidence of its recovery for reintegration. Therefore, to address gray failures, the system must protect the current task before degradation can be reliably detected, accumulate evidence to adapt future routing, and reintegrate an agent once its performance recovers.

We introduce MESHHEAL (see Fig. 1), a fully decentralized self-healing framework that couples peer review across two timescales. At the fast timescale, ability-matched neighboring agents review every intermediate contribution. An adaptive review hierarchy begins with repeated single-reviewer evaluation and escalates uncertain or low-scoring outputs to committee deliberation and, when necessary, multi-agent correction. At the slow timescale, a task- and ability-conditioned peer-relative detector accumulates the same review evidence to distinguish persistent degradation from ordinary output variation and impose progressively stronger routing restrictions, culminating in isolation from ordinary task execution. Recovery probes provide fresh reviewed evidence, allowing agents whose performance has recovered to return gradually to normal routing.

In our experiments on BBH, MATH, and MMLU-Pro, MESHHEAL achieves 83.9% mean accuracy during degradation over five seeds, compared with 80.7% for Symphony, the strongest baseline. It uses 51k tokens (both input and output tokens) per task, compared with 115k for Symphony and 178k for SAC, reducing token use by 56% and $7 1 \% ,$ respectively. In sparse networks of up to 576 agents, MESHHEAL isolates all degraded agents and achieves comparable accuracy with a complete graph.

A commonly used approach to enforce ability constraints on agents when evaluating ability-aware routing is to use system prompts (Li et al., 2023; Qian et al., 2024; Chen et al., 2024; Hong et al., 2024; Chen et al., 2026). However, we identify an important confound: system prompts do not reliably enforce ability constraints, and the agent may still be able to solve tasks outside its abilities, allowing incorrect routes to produce correct final answers and thereby hiding routing errors. In a controlled BBH (Suzgun et al., 2023) study across five models, we find that prompt-based ability assignments do not reliably separate performance on assigned and unassigned abilities. We introduce Model-Backed MAS Evaluation, which uses backend selection to realize ability assignments and inject controlled gray failures. For a requested ability, a healthy agent uses a strong model if it holds that ability and a weak model otherwise. This construction provides known failure ground truth while preserving the defining property that a gray-failed agent remains responsive.

## 2 RELATED WORK

LLM multi-agent coordination and robustness. LLM agents combine reasoning and tool use to solve complicated tasks (Yao et al., 2023). MASs extend these capabilities through task decomposition and specialization (Wu et al., 2023; Hong et al., 2024; Qian et al., 2024), while decentralized systems coordinate and adapt through local interactions (Yang et al., 2025b; Li et al., 2026).

Various approaches have been proposed to improve output quality of LLM and MAS, e.g., repeated inference, deliberation, peer review, and robust aggregation (Wang et al., 2023; Du et al., 2024; Li et al., 2024; Wang et al., 2025; Jo & Park, 2025; Zheng et al., 2026; Lee et al., 2026). These approaches may also be applied here to handle gray failure. However, they operate within each task and do not identify whether a persistent failure has occurred to an agent. Therefore, a degraded agent may consistently participate in task solving.

Adaptive routing strategies were also developed to handle failures in MASs. AgentNet updates routing based on interaction history (Yang et al., 2025b), Symphony-Coord combines adaptive agent selection with voting (Guan et al., 2026), and RAPS uses reputation to identify and isolate unreliable peers (Li et al., 2026). However, routing adaptation alone cannot protect the current task, and once an agent is bypassed, the system receives no fresh evidence of its recovery for reintegration.

Distributed recovery and gray failures. Our recovery goal builds on self-stabilizing and self-healing systems, where systems adapt their behavior in response to failures (Dijkstra, 1974; Arora & Gouda, 1993; Psaier & Dustdar, 2011). Gray-failure research shows that hidden degradation may need to be inferred from peer observations rather than availability (Huang et al., 2017; 2018). We study this problem for LLM agents using reviewed output quality as the failure signal.

## 3 PROBLEM FORMULATION

We consider a decentralized MAS with heterogeneous agent abilities and potential gray failures.

## 3.1 DECENTRALIZED MAS ARCHITECTURE

We model a decentralized LLM-based MAS as an undirected graph $G = ( \nu , \mathcal { E } )$ , where V denotes the set of agents, and $\mathcal { E } \subseteq \mathcal { V } \times \mathcal { V }$ represents the communication graph between the agents: an undirected edge $e _ { i , j } \in \mathcal { E }$ allows agent i to send information to agent j and vice versa. Each agent consists of an LLM and a local controller. The LLM is used to generate the agent’s outputs, and is called the agent’s model backend; and the local controller coordinates the agent’s interactions with its neighbors.

We consider heterogeneous agents where agents differ in their abilities. Denote by $\kappa _ { i }$ the set of abilities of an agent i, and $\kappa _ { i }$ varies across agents. Solving different tasks may require a different set of abilities. Therefore, it is of key importance to assign the task to the agent with the matching ability. We study the decentralized setting, where no agent has a global view of other agents, and each agent can only communicate with its neighbors on the communication graph $\mathcal { E }$

The multi-agent system will be given a sequence of tasks denoted by $\mathcal { T } = \{ \tau _ { 1 } , \tau _ { 2 } , \dots , \tau _ { N } \}$ . The t-th task is $\tau _ { t } = ( Q _ { t } , T _ { t } )$ , where $t \in [ N ]$ is the task index, $Q _ { t }$ is the input question, $T _ { t }$ is its benchmarkprovided category label (e.g., geometry in MATH (Hendrycks et al., 2021)), and we use $R _ { t }$ to represent the set of abilities required to solve $Q _ { t }$ . Here $[ N ] \triangleq \{ 1 , \dots , N \}$ . We note that $Q _ { t }$ and $T _ { t }$ are directly given to the agent, but $R _ { t }$ needs to be inferred by the agent.

Objective. Let $\Pi _ { \mathrm { l o c a l } }$ be the set of coordination policies whose decisions use only the current task state, local interaction history, and messages from direct neighbors. For a policy π, let $\widehat { y } _ { t } ^ { \pi }$ be the final answer of the MAS and Eval $( \widehat { y _ { t } ^ { \pi } } ) \in \{ 0 , 1 \}$ indicate correctness under the benchmark evaluator. We seek to optimize the average accuracy over the N sequential tasks:

$$
\operatorname* { m a x } _ { \pi \in \Pi _ { \mathrm { l o c a l } } } \frac { 1 } { N } \sum _ { t = 1 } ^ { N } \mathbb { E } [ \mathrm { E v a l } _ { t } ( \widehat { y } _ { t } ^ { \pi } ) ] ,\tag{1}
$$

where the expectation is over the execution randomness.

## 3.2 GRAY FAILURE AND SELF-HEALING

We study a practical setting in which agents may experience gray failures (Huang et al., 2017; 2018). A gray failure occurs when an agent remains available and protocol-compliant, i.e., it can communicate with its neighbors and return syntactically valid responses, but its effective ability to solve some tasks is degraded relative to its nominal ability.

Each agent may have two possible health conditions: healthy, degraded. The health condition of each agent i may change over time. We assume that each agent’s health condition remains unchanged within a task but may change across tasks. An agent is called healthy when it operates at its normal level of reliability, and degraded when it suffers from gray failure. The true health condition is latent, and the agent must infer it from the agent’s outputs. We further define recovery as restoration of an agent from degraded to healthy.

In this paper, our focus is on solving the problem in Eq. (1) when agents experience gray failures and may also recover from them. We aim to design local coordination policies so that, even with degraded agents, each task can still be routed to healthy agents with matching abilities and solved with high accuracy; and at the same time, when an agent recovers from degraded to healthy, future tasks that match this agent’s abilities will still be routed to it. Such desired property is referred to as self-healing. To evaluate this ability, beyond task accuracy and inference cost, we introduce three additional criteria: detection delay, closure, and reintegration. These criteria are adapted from the distributed fault-tolerance literature (Dijkstra, 1974; Arora & Gouda, 1993; Ghosh et al., 2007; Psaier & Dustdar, 2011). More specifically, suppose agent i becomes degraded at task $b _ { i } \in [ N ]$ and recovers at task $r _ { i } \in [ N + 1 ]$ . If recovery is not observed in an N-task stream, we set $r _ { i } = N + 1$ . Let $\mathcal { T } _ { t }$ denote the set of agents that are excluded from ordinary task execution at task t. Then, we define:

• Detection delay: For a degraded agent i, the system should isolate it as early as possible. Let $c _ { i } = \operatorname* { m i n } \{ t \geq b _ { i } : i \in \mathcal { I } _ { t } \}$ be the index of the first task at which the degraded agent i is isolated. The detection delay is then defined as $\delta _ { i } ^ { \mathrm { q e l a y } } = c _ { i } - b _ { i }$

• Closure: A system satisfies the closure property if a degraded agent remains isolated until recovery after being detected. Formally, for any degraded agent $i , i \in \mathcal { T } _ { t } , \forall c _ { i } \leq t < r _ { i }$

• Reintegration: it refers to the case that after recovery, agent i returns to normal routing. Let $u _ { i } \geq r _ { i }$ be the index of the first task at which agent i returns to normal routing state through the end of the observed stream. Then the reintegration delay is defined as $\delta _ { i } ^ { \mathrm { r e i n } } = \bar { u _ { i } } - r _ { i }$

## 4 MESHHEAL: REVIEW AT TWO TIMESCALES

In Sec. 4.1, we will present our base workflow design in a decentralized MAS without grayfailures. Then, in Sec. 4.2, we build upon the base design and introduce our MESHHEAL with novel twotimescale mechanism for self-healing. The full algorithm is provided in Alg. 1 in App. B.1.

## 4.1 BASE TASK EXECUTION AND ROUTING (NO FAILURES)

Task execution. Agents collaborate to sequentially solve and route a task through the network. Each task $\tau _ { t } = ( Q _ { t } , T _ { t } )$ is initially assigned to an entry agent sampled uniformly at random from the network. The query $Q _ { t }$ and task type $T _ { t }$ are given, while $R _ { t }$ denotes the set of abilities required to solve the task and is inferred by the entry agent.

The task is solved sequentially as it moves through the network, with each agent addressing the part aligned with its abilities and passing the rest to the next agent until it is fully solved. The task state is initialized as $X _ { t } = ( Q _ { t } , T _ { t } , \mathbf { \dot { R } } _ { t } , C _ { t } \mathbf { \dot { , } } Y _ { t } , P _ { t } )$ , where $C _ { t }$ records the abilities already covered so far, $Y _ { t }$ stores the working context that will be passed to later agents, including accepted intermediate outputs, and $P _ { t }$ is a list that records the route taken so far. Initially, $C _ { t } = Y _ { t } \bar { = } \emptyset$ and $P _ { t }$ is also empty.

When the task is routed to agent i, let $M _ { t } = R _ { t } \backslash C _ { t }$ be the set of abilities that were not supplemented by previous agents in $P _ { t }$ and need to be further provided by subsequent agents, and $H _ { i , t } = \boldsymbol { \mathcal { K } } _ { i } \cap M _ { t }$ be the subset of matching abilities that can be provided by agent i. We take a rule-based approach at the controller, and these sets govern the controller’s decision for the current task:

• If $H _ { i , t } = \emptyset$ , then agent i has no matching abilities to contribute to the remaining of task t. Then, agent i’s controller chooses to Handoff: it generates no task content and forwards the task state unchanged to the next agent (discussed later in Routing). Thus, $C _ { t } , Y _ { t }$ and $P _ { t }$ remain unchanged.

• $\begin{array} { r } { \mathrm { I f } \ \varnothing \ne \ \mathbf { \bar { H } } _ { i , t } \subset M _ { t } } \end{array}$ , then agent i contribute matching abilities to the task that prior agents in $P _ { t }$ cannot; however, additional abilities beyond those in $H _ { i , t }$ are still needed. Then, agent i’s controller chooses to Contribute. The agent’s LLM solves the matching part in the remaining of the task using abilities in $H _ { i , t }$ . The output is appended to $Y _ { t } , C _ { t } = \bar { C } _ { t } \mathrm { ~ \bar { U } ~ } H _ { i , t }$ , and agent i is appended to the list $P _ { t }$ . The controller recomputes $M _ { t } \gets R _ { t } \setminus C _ { t }$ using the updated $C _ { t }$ and forwards the updated task state to the next agent (discussed later in Routing).

• If $H _ { i , t } = M _ { t }$ , then agent i has all the abilities needed to finish the remaining of the task. Then, agent i’s controller chooses to Finalize. The agent solves the remaining task, combines the result with earlier outputs in $Y _ { t } .$ , and proposes the final answer. At this point, all abilities in $R _ { t }$ have been covered by agents along the route in $Y _ { t }$ , and no further routing is required.

Routing. Each agent i maintains an estimated the shortest hop distance $D _ { i } ( s )$ to the nearest agent that holds ability s. If agent i itself holds ability s, then $\bar { D _ { i } } ( s ) = 0 ;$ otherwise, it updates in a distributed manner: $\begin{array} { r } { D _ { i } ( s ) = 1 + \operatorname* { m i n } _ { j \in \mathcal { N } _ { i } } D _ { j } ( s ) } \end{array}$ , where ${ \mathcal { N } } _ { i }$ is the set of direct neighbors of agent i (Bellman, 1958). Under standard asynchronous Bellman-Ford assumptions with fixed topology and reliable neighbor updates, these iterations converge in finite time (Bertsekas, 1982). The path that achieves $D _ { i } ( s )$ will be used to select the next agent. When Handoff or Contribute are chosen by the controller, the task has at least one ability not provided along the route in $P _ { t }$ . The controller samples one ability s uniformly at random from $\dot { M } _ { t }$ . It then routes the task state along the path that achieves $D _ { i } ( s )$ , i.e., to the nearest agent with ability s.

## 4.2 TASK EXECUTION AND ROUTING UNDER GRAY FAILURES

Having defined the base task flow without failures, we now present our novel design to address gray failures. We define a state $h _ { i } \in \{ \mathbf { n o r m a l }$ , watched, isolated} for each agent i, which determines how the agent participates in execution and routing. A normal agent follows the base workflow without restriction. A watched agent is routed and executes tasks as a normal agent, but every output it produces receives mandatory committee review. An isolated agent has its set of abilities $\kappa _ { i }$ set to be ∅ temporarily, and therefore it will be excluded from task execution, but its controller remains active for message relay. When an agent becomes isolated, agents will need to recompute the shortest hop distance as described in Sec. 4.1. We then apply the base task flow in Sec. 4.1.

Our MESHHEAL framework presents a novel two-timescale scheme with fast review and correction and slow detection, rerouting, and recovery (illustrated in Fig. 1). At thefast timescale, an review protocol escalates low-scoring, high-disagreement, or watched-agent outputs from repeated singlereviewer evaluation to deliberative committee review and multi-agent correction. At the slow timescale, a task- and ability-conditioned peer-relative detector aggregates persistent shortfalls and normalizes them against healthy-neighbor variation, enabling same thresholds to be used across model backends; guarded recovery probes close the loop through reintegration.

## 4.2.1 FAST TIMESCALE: REVIEW AND CORRECTION

At the fast timescale, every output produced by Contribute or Finalize is reviewed before use. We denote the output by $y _ { i , t }$ . An eligible reviewer must be a normal direct neighbor that holds every ability in $H _ { i , t }$ . The reviewer scores $y _ { i , t }$ on a fixed [0, 1] scale (judge prompts in App. D.1). To reduce review noise, the same reviewer evaluates $y _ { i , t }$ in three independent calls. Denote the median by $q _ { 0 } .$ and we use it as the initial score. We use the median rather than the mean to reduce sensitivity to reviewer variability and outlier scores, consistent with prior work on robust aggregation of LLM judgments (Acharya et al., 2026; Gallaba et al., 2025). We also use the standard deviation $s _ { u }$ to measure disagreement. Committee review is triggered when $q _ { 0 } \leq \tau _ { \mathrm { t o } } , s _ { u } \geq \sigma _ { \mathrm { c o m } } , h _ { i } = \mathbf { w a t c l }$ ed, or the output is a recovery probe, where $\tau _ { \mathrm { t o } } = 0 . 4 0$ is the acceptance threshold and $\sigma _ { \mathrm { c o m } } = 0 . 5 5$ is the disagreement threshold. Committee review adds multiple perspectives and deliberation, improving review reliability at the cost of additional inference and communication; we therefore invoke it only for low-scoring, high-disagreement, or watched-agent outputs.

The committee contains two eligible reviewers. They first evaluate $y _ { i , t }$ independently, exchange their explanations, and then reassess the same output. Let $v _ { A }$ and $v _ { B }$ be their second-round scores; the final review score is $q _ { i , t } = ( v _ { A } + v _ { B } ) / 2$ . If committee review is not triggered, $q _ { i , t } = q _ { 0 }$ . When only one eligible reviewer is available, two reviews are obtained through separate inference calls with isolated context histories. If $q _ { i , t } > \tau _ { \mathrm { t o } } , y _ { i , t }$ is accepted. Otherwise, Takeover is triggered: the two reviewers generate correction candidates, which a committee reviewer in the normal state synthesizes task context following Mixture-of-Agents style aggregation (Wang et al., 2025). The resulting corrected output $\widehat { y } _ { i , t }$ replaces $y _ { i , t }$ before entering the task state. The score ${ { q } _ { i , t } }$ remains attributed to agent i and is retained as evidence for the slow timescale degradation detection.

## 4.2.2 SLOW TIMESCALE: DETECTION, REROUTING AND RECOVERY

A single low score in the fast timescale does not necessarily mean that the agent is persistently degraded. We therefore accumulate the review scores $q _ { i , t }$ across tasks and use relative peer comparison to update the estimate of $h _ { i }$ . Our goal is to detect agent gray failures across different task types and model backends with the same detector, rather than tuning a separate threshold for each setting.

Relative shortfall. For each ability $s \in H _ { i , t }$ , the controller considers direct normal neighbors that have recently reviewed outputs within the fixed history window from earlier tasks with the same task type $T _ { t }$ and ability s. Let $\mathcal { P } _ { i , T _ { t } , s } ( t )$ denote this set of neighbors, excluding $i ,$ and let $\mu _ { j , T _ { t } , s } ( t )$ denote neighbor $j ^ { \circ } \mathrm { s }$ recent mean review score before task t. The controller first uses these neighbors to estimate the normal review score for the same type of work $( \bar { q } _ { - i , T _ { t } , s } ( t ) )$ . For each $s \in H _ { i , t } .$ the controller measures how far agent $i \gamma _ { \mathrm { s } }$ current score falls below the corresponding reference:

$$
\bar { q } _ { - i , T _ { t } , s } ( t ) = \frac { 1 } { | \mathcal { P } _ { i , T _ { t } , s } ( t ) | } \sum _ { j \in \mathcal { P } _ { i , T _ { t } , s } ( t ) } \mu _ { j , T _ { t } , s } ( t ) , \qquad x _ { i , t , s } = \bar { q } _ { - i , T _ { t } , s } ( t ) - q _ { i , t } .
$$

A positive $x _ { i , t , s }$ means that agent i scored below normal neighbors on the same task type and ability s. This comparison removes differences in absolute review scores across task types by comparing an agent only with normal neighbors who work on the same task type and have the same ability.

Persistent degradation. A single positive shortfall may result from LLM randomness. We therefore maintain a separate detector for each $( T , s )$ and average its most recent 8 shortfalls, yielding $\bar { x } _ { i , T , s } .$ For a healthy agent, it should stay near zero; if the agent remains degraded, it should stay positive.

However, a fixed threshold on $\bar { x } _ { i , T , s }$ may not transfer across model backends because the dispersion of healthy agents’ mean shortfalls can differ substantially across backends. We therefore also measure how much the mean shortfalls of normal neighbors vary. We pool the mean shortfalls $\bar { x } _ { j , T , s ^ { \prime } }$ of all normal neighbors $j$ over all abilities $s ^ { \prime }$ under task type $T .$ , and let $m _ { i , T }$ and $p _ { i , T } ^ { 2 5 }$ be their median and 25th percentile. We measure the normal variation by the lower spread $m _ { i , T } - p _ { i , T } ^ { 2 5 }$ , because degraded agents that have not yet been detected can inflate the upper tail. We normalize agent i’s mean shortfall by this variation:

$$
z _ { i , T , s } = \frac { \bar { x } _ { i , T , s } - m _ { i , T } } { \operatorname* { m a x } \Bigl ( m _ { i , T } - p _ { i , T } ^ { 2 5 } , 0 . 1 0 \Bigr ) } .
$$

Thus, $z _ { i , T , s }$ measures how far agent $i \ ' s$ mean shortfall is above normal level, relative to how much healthy agents vary. This second comparison accounts for differences in how much healthy agents’ mean shortfalls vary across model backends, so same detection thresholds can be used across models.

State transitions, rerouting, and recovery. Each update uses the current $z _ { i , T , s } ;$ if fewer than three matched observations are available $h _ { i }$ remains unchanged. Otherwise, $z _ { i , T , s } \geq \alpha$ moves normal to watched, $z _ { i , T , s } < \alpha$ returns watched to normal, and two consecutive updates with $z _ { i , T , s } \geq \beta$ move a normal or watched agent to isolated; the isolation count resets when $z _ { i , T , s } < \beta$ . Here $\alpha < \beta$ are thresholds, and in our experiments we choose $\alpha = 1 . 2$ and $\beta = 2$ . The thresholds were selected empirically to balance detection delay and false isolation of healthy agents.

We further design recovery probes that use the same review and detection mechanism. After every 10 tasks, an isolated agent becomes eligible for a recovery probe. When a neighbor next receives work matching its abilities, the work is routed to the isolated agent and its output receives mandatory committee review. The resulting score updates the same detector. The agent returns to watched after one update with $z _ { i , T , s } < \beta _ { \mathrm { { t } } }$ , even if $z _ { i , T , s } < \alpha$ , and to normal after a later update with $z _ { i , T , s } < \alpha$

## 5 MODEL-BACKED MAS EVALUATION

System prompt fails to constrain the agent’s ability. Many LLM MAS evaluations assign abilities through system prompts while using the same underlying model for all agents (Li et al., 2023; Qian et al., 2024; Chen et al., 2024; Hong et al., 2024). To test whether such prompts constrain agents, we compare two prompting conditions across five models on BBH benchmark (Suzgun et al., 2023): one makes no statement about unassigned abili-

Table 1: BBH ability-assignment test. Arrows show no competence statement → unassigned abilities declared unreliable; $\Delta _ { \mathrm { g a p } }$ is the resulting increase in the accuracy gap between assigned and unassigned abilities.
<table><tr><td>Model</td><td>Unassigned ability acc.</td><td>Assigned ability acc.</td><td> $\Delta _ { \mathrm { g a p } }$ </td></tr><tr><td>DeepSeek-Chat</td><td>.812→.775</td><td>.800→.792</td><td>.029</td></tr><tr><td>Llama-3.1-70B</td><td>.781→.750</td><td>.783→.767</td><td>.015</td></tr><tr><td>GPT-4o-mini</td><td>.744→.675</td><td>.708→.667</td><td>.027</td></tr><tr><td>GPT-OSS-120B</td><td>.869→.831</td><td>.825→.833</td><td>.046</td></tr><tr><td>Qwen-2.5-7B</td><td>.494→.219 .558→.342</td><td></td><td>.058</td></tr></table>

ties, while the other explicitly describes them as unreliable. If these prompts truly limited agents to their assigned abilities, accuracy on unassigned abilities should fall while assigned-ability accuracy remains stable. The accuracy gap between assigned and unassigned abilities increases by only 0.015– 0.058 (see Tab. 1). Accuracy on unassigned abilities remains substantial, while assigned-ability accuracy also falls for four models. Prompt instructions change agent behavior, but do not reliably prevent agents from solving tasks outside their assigned abilities. App. A gives the full protocol.

Controlled ability realization and gray-failure injection. We use model backends to realize ability assignments and inject gray failures, shown in Fig. 1(c). A healthy agent uses a strong model for assigned abilities and a weak model otherwise. Degradation is simulated by replacing its strong backend with the weak one while leaving communication and execution interfaces unchanged. This preserves responsiveness while reducing task competence, providing known failure ground truth and making routing errors visible in end-to-end accuracy.

Tasks and degraded agents. We apply this protocol to seven task groups from each of BBH, MATH, and MMLU-Pro (Suzgun et al., 2023; Hendrycks et al., 2021; Wang et al., 2024). Each task group contains tasks of the same type and is associated with two complementary abilities required for solving those tasks. The two abilities are assigned to disjoint sets of agents, so no single agent holds both. Every task requires at least two agents. Full task-to-ability mappings appear in App. B.

## 6 EXPERIMENTS

We evaluate on benchmarks of BBH, MATH, and MMLU-Pro. The main system contains six agents, A1–A6, with three agents per ability and disjoint groups for the two abilities required by each task.

Table 2: (a) Accuracy and total model-token cost under persistent degradation $( \mathrm { m e a n } \pm \mathrm { s t d }$ . over three datasets, each with five seeds). H/D denote healthy/degraded phases; tokens are thousands per degraded-phase task. (b) Scaling in complete and sparse networks over five seeds. $I / D$ denotes isolated/degraded agents; edge fraction is relative to the complete graph.  
(a) Persistent degradation
<table><tr><td>Method</td><td>H acc.</td><td>D acc.</td><td>Tokens</td></tr><tr><td>MESHHEAL</td><td> ${ \bf . 8 3 1 \pm . 0 1 3 }$ </td><td> $\mathbf { 8 3 9 \pm . 0 1 2 }$ </td><td> $5 1 \pm 5$ </td></tr><tr><td>Symphony</td><td> $. 8 2 7 { \pm } . 0 1 3$ </td><td> $. 8 0 7 \pm . 0 1 4$ </td><td> $1 1 5 { \pm } 9$ </td></tr><tr><td>SAC</td><td> $. 8 2 2 \pm . 0 1 4$ </td><td> $. 7 4 6 \pm . 0 2 1$ </td><td> $1 7 8 \pm 1 3$ </td></tr><tr><td>RAPS</td><td>.800±.019</td><td> $. 6 5 3 { \pm } . 0 4 1$ </td><td> $4 8 \pm 3$ </td></tr><tr><td>AgentNet</td><td> $. 7 2 9 { \pm } . 0 2 3$ </td><td> $. 5 5 7 { \pm } . 0 3 9$ </td><td> $4 3 \pm 2$ </td></tr><tr><td>AutoGen</td><td> $. 7 8 7 \pm . 0 1 4$ </td><td> $. 4 2 4 \pm . 0 2 3$ </td><td> $1 5 { \pm } 0$ </td></tr></table>

(b) Network scaling
<table><tr><td colspan="2">Complete</td><td colspan="2">Sparse</td><td rowspan="2">Edge fraction</td></tr><tr><td>Agents Acc.</td><td> $I / D$ </td><td>Acc.</td><td> $I / D$ </td></tr><tr><td>96.827</td><td>32/32</td><td>.818</td><td>32/32</td><td>5.00%</td></tr><tr><td>288.813</td><td>96/96</td><td>.803</td><td>96/96</td><td>1.71%</td></tr><tr><td>576.808</td><td>192/192</td><td>.814</td><td>192/192</td><td>0.88%</td></tr></table>

A3 and A5 are selected for degradation, every LLM call by them, including review calls, switches to the weak model. Their controllers, abilities, communication links, and availability remain unchanged. Unless stated otherwise, healthy agents use opena $\cdot / { \tt g p t - o s s - 1 }$ 20b for declared abilities; undeclared abilities and degraded agents use meta-llama/llama-3.2-1b-instruct.

We compare MESHHEAL with AgentNet’s decentralized routing (Yang et al., 2025b), RAPS’s reputation-based routing (Li et al., 2026), and a centralized AutoGen implementation with a mandatory manager (Wu et al., 2023). SAC uses three teams with two rounds of filtering and refinement followed by majority aggregation (Lee et al., 2026), while Symphony uses parallel plans and voting (Guan et al., 2026). All methods use the same tasks, agents, ability assignments, models, degradation targets, and schedule. In AutoGen, A3 is the mandatory manager and appears on every task path.

Each setting uses five seeds, defining task samples and order shared across methods. We report means and sample standard deviations. Except in targeted ablations, review rubrics and detector parameters are fixed across datasets, model pairs, and network sizes (Tab. B11). We measure accuracy, token consumption (counting both input and output tokens) , detection delay $\delta _ { i } ^ { \mathrm { d e l a y } }$ , closure, and reintegration delay $\delta _ { i } ^ { \mathrm { r e i n } }$ ; Appendices B–E provide settings, prompts, traces, and full results.

## 6.1 PERSISTENT DEGRADATION: ACCURACY AND COST

Each stream contains 252 healthy-phase tasks followed by 252 disjoint degraded-phase tasks with the same task-type distribution. During degraded-phase, A3 and A5 switch to the weak model. We report healthy- and degraded-phase accuracy and total model-token cost. After degradation, MESHHEAL reaches $0 . 8 3 9 \pm 0 . 0 1 2$ accuracy at $5 1 \pm 5 \mathrm { k }$ total model tokens per task, compared with Symphony’s $0 . 8 0 7 \pm 0 . 0 1 4$ at $1 1 5 \pm 9 \mathrm { k }$ . AgentNet and RAPS use fewer tokens but have substantially lower accuracy (Tab. 2(a)). Full dataset-level and cost results are in App. C.

## 6.2 DEGRADATION AND RECOVERY: DETECTION, CLOSURE, AND REINTEGRATION

We test the three self-healing criteria on five consecutive 126-task MATH phases: healthy operation, A5 degraded, A3 and A5 degraded, A3 degraded after A5’s backend restoration, and both backends restored. All methods complete the 630-task stream. Across the 504 tasks after the first degradation, including the fully restored phase, MESHHEAL reaches 0.812 accuracy at 49k total model tokens per task, compared with Symphony’s 0.806 at 105k (shown in Tab. 3). Symphony leads in the A5-only and fully recovered phases, while the methods tie when only A3 remains degraded.

Across five seeds, after agent i becomes degraded at $b _ { i }$ , it is first excluded from ordinary task execution at $c _ { i } .$ . The resulting detection delays $\delta _ { i } ^ { \mathrm { d e l a y } } = c _ { i } - b _ { i }$ are $8 . 0 \pm 5 . 2$ tasks for A5 and $1 2 . 4 \pm 6 . 2$ tasks for A3. Once isolated, both agents satisfy closure in all five seeds, remaining excluded throughout $c _ { i } \leq t < r _ { i }$ except for committee-reviewed recovery probes. After backend restoration at $r _ { i }$ , the agents return to normal routing at $u _ { i } ,$ , giving reintegration delays $\delta _ { i } ^ { \mathrm { r e i n } } = u _ { i } - r _ { i }$ of $4 6 . 0 \pm 1 1 . 7$ tasks for A5 and $4 5 . 0 \pm 3 . 5$ tasks for $_ { \textrm A 3 }$ (Tab. 3). Fig. C1 in App. C.2 shows one representative trajectory, with its state changes listed in Tab. C5. None of the baseline methods fully isolates degraded agents; under our definitions, their detection delay is therefore infinite, closure is not established, and reintegration delay is undefined.

Table 3: Self-healing under staggered degradation and recovery over five seeds. (a) Mean accuracy and total model-token cost across phases. (b) Detection (Det.) and reintegration (Rein.) delays are mean ± std. (min–max); closure reports successful seeds.  
(a) Accuracy and token cost
<table><tr><td>Phase</td><td>MESHHEAL Sym.</td><td>SAC</td><td>A.Net</td><td>RAPS</td><td>AutoGen</td></tr><tr><td>Healthy</td><td>.833 .810</td><td>.825</td><td>.794</td><td>.722</td><td>.627</td></tr><tr><td>A5 deg.</td><td>.778</td><td>.786 .722</td><td>.579</td><td>.492</td><td>.659</td></tr><tr><td>A3+A5 deg.</td><td>.770 .722</td><td>.643</td><td>.452</td><td>.405</td><td>.183</td></tr><tr><td>A5 restored; A3 deg.</td><td>.841 .841</td><td>.706</td><td>.532</td><td>.524</td><td>.135</td></tr><tr><td>All restored</td><td>.857 .873</td><td>.860</td><td>.563</td><td>.524</td><td>.770</td></tr><tr><td>Pooled after deg.</td><td>.812 .806</td><td>.733</td><td>.532</td><td>.486</td><td>.437</td></tr><tr><td>Tokens/task after deg.</td><td>49k</td><td>105k 124k</td><td>39k</td><td>38k</td><td>15k</td></tr></table>

(b) Self-healing metrics of MESHHEAL
<table><tr><td>Agent</td><td>Det.↓</td><td>Closure</td><td>Rein.↓</td></tr><tr><td rowspan="2">A5</td><td> $^ { 8 . 0 \pm 5 . 2 } _ { ( 2 - 1 6 ) }$ </td><td></td><td> $4 6 . 0 \pm 1 1 . 7$ </td></tr><tr><td> $1 2 . 4 \pm 6 . 2$ </td><td>5/5</td><td>(30–63)</td></tr><tr><td>A3</td><td>(6-22)</td><td>5/5</td><td> $4 5 . 0 \pm 3 . 5$  (42-51)</td></tr></table>

(b) Detector comparison

Table 4: (a) Accuracy over five seeds on BBH; H and D denote the healthy and degraded phases. (b) Missed degraded agents out of two and falsely isolated healthy agents out of four on BBH.  
(a) Fast-timescale components
<table><tr><td>Setting</td><td>H</td><td>D</td><td></td></tr><tr><td>Full</td><td> $. 8 9 9 \pm . 0 2 7$ </td><td></td><td> $\mathbf { 9 0 0 \pm . 0 1 4 \ + . 0 0 1 \pm . 0 1 6 }$ </td></tr><tr><td>Single reviewer</td><td> ${ \bf . 9 0 9 \pm . 0 2 2 }$ </td><td></td><td> $\phantom { - } . 8 6 5 \pm . 0 2 9 \phantom { - } - . 0 4 4 \pm . 0 2 4$ </td></tr><tr><td>No Takeover</td><td> $. 8 7 9 \pm . 0 2 1$ </td><td>.844 ± .036</td><td> $- . 0 3 5 \pm . 0 3 0$ </td></tr></table>

<table><tr><td></td><td>Pair A</td><td></td><td>Pair B</td></tr><tr><td>Rule</td><td>Miss False Miss False</td><td></td><td></td></tr><tr><td>Absolute threshold</td><td>0/2</td><td>4/4</td><td>0/2</td></tr><tr><td>CUSUM</td><td>0/2</td><td>2/4</td><td>1/2 0/4</td></tr><tr><td>Relative peer comparison</td><td>0/2</td><td>0/4 0/2</td><td>0/4</td></tr></table>

## 6.3 SCALING WITH LOCAL INFORMATION

We replicate the six-agent profiles in complete and sparse graphs with 96, 288, and 576 agents, with one third degraded and all review, summary exchange, and routing updates remaining local. MESHHEAL isolates every degraded agent in all settings, while sparse and complete accuracies differ by at most 0.010. At 576 agents, the sparse graph uses only 0.88% of complete-graph edges and reaches 0.814 accuracy (Tab. 2(b)), demonstrating MESHHEAL’s effectiveness in sparse networks.

## 6.4 COMPONENT AND DETECTOR STUDIES

Fast timescale. We replace committee review with a single reviewer and separately remove Takeover on BBH while retaining detection and rerouting (Tab. 4(a); App. C.4, Tab. C6). Single reviewer uses the initial peer for judgment and correction; it has a slightly higher healthy-phase mean but 0.035 lower degraded-phase accuracy and a mean phase change of −0.044. No Takeover leaves low-scoring outputs unchanged, reducing degraded-phase accuracy by 0.056 and healthy-phase accuracy by 0.020. This suggests that Takeover also corrects some errors from healthy agents. An audit (App. C.5, Tab. C8(a,b)) shows that healthy agents receive much higher review scores than degraded agents (0.89 vs. 0.08), and 98% of degraded outputs are sent to committee review. Committee reassessment reduces false alarms on healthy agents. A separate Takeover audit (Tab. C8(c)) shows that Takeover corrects 15 of 32 wrong answers while turning only 1 of 32 correct answers into an incorrect one.

Slow timescale. We compare detectors under Pair A (GPT-OSS-120B/Gemma-3-4B) and Pair B (Llama-3.3-70B/Llama-3.2-1B). The absolute threshold falsely isolates all four healthy agents under Pair A; CUSUM (Page, 1954) falsely isolates two under Pair A and misses one of two degraded agents under Pair B. Relative peer comparison detects both degraded agents without false isolations under either pair using fixed rubrics and parameters (Tab. 4(b); App. C.4, Tab. C7).

Model-pair sensitivity. To test robustness under a milder degradation and a weaker healthy model, we additionally evaluate Pair A and Pair B, respectively. MESHHEAL detects both degraded agents without false isolation across both settings, while maintaining competitive degraded-phase accuracy at substantially lower cost: it outperforms Symphony by 0.040 on Pair A and is within 0.004 on Pair B, using only 52–56% as many total model tokens. Full results are in App. C.2.

## 7 CONCLUSION

We introduced MESHHEAL, a fully decentralized framework for self-healing gray failures in LLM agent networks with a novel two-timescale review design. At the fast timescale, review protocol evaluates every produced output and can replace low-quality outputs before they enter the task state; and at the slow timescale, a task- and ability-conditioned peer-relative detector aggregates persistent shortfalls to drive staged state transitions and routing updates, while recovery probes support evidencebased reintegration. We further showed that prompt-only ability assignments do not reliably enforce constraints on agents’ ability and proposed Model-Backed MAS Evaluation, which uses controlled backend selection to realize complementary abilities and inject gray failures with known ground truth while keeping affected agents responsive. With the default model pair, MESHHEAL reaches 0.839 degraded-phase accuracy at 51k tokens per task across three benchmarks and five seeds, versus the strongest baseline of Symphony 0.807 at 115k tokens per task. The experiment with degradation and recovery at different times meets the three self-healing criteria under the evaluated schedule. The scaling experiments isolate every degraded agent in the evaluated networks of up to 576 agents.

## REFERENCES

Anish Acharya, Kris W Pan, and Brian Verkhovsky. Ropoll: Robust panel of llm judges. arXiv preprint arXiv:2606.30931, 2026.

Anish Arora and Mohamed G. Gouda. Closure and convergence: A foundation of fault-tolerant computing. IEEE Transactions on Software Engineering, 19(11):1015–1027, 1993.

Richard Bellman. On a routing problem. Quarterly of applied mathematics, 16(1):87–90, 1958.

Dimitri Bertsekas. Distributed dynamic programming. IEEE transactions on Automatic Control, 27 (3):610–616, 1982.

Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021.

Keru Chen, Jun Luo, Sen Lin, Yingbin Liang, Alvaro Velasquez, Nathaniel Bastian, and Shaofeng Zou. Hipo: Instruction hierarchy via constrained reinforcement learning. arXiv preprint arXiv:2603.16152, 2026.

Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chi-Min Chan, Heyang Yu, Yaxi Lu, Yi-Hsin Hung, Chen Qian, Yujia Qin, Xin Cong, Ruobing Xie, Zhiyuan Liu, Maosong Sun, and Jie Zhou. Agentverse: Facilitating multi-agent collaboration and exploring emergent behaviors. In The Twelfth International Conference on Learning Representations, 2024.

Edsger W. Dijkstra. Self-stabilizing systems in spite of distributed control. Communications of the ACM, 17(11):643–644, 1974. doi: 10.1145/361179.361202.

Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 11733–11763, 2024.

Keheliya Gallaba, Ali Arabat, Dayi Lin, Mohammed Sayagh, and Ahmed E Hassan. Towards conversational development environments: Using theory-of-mind and multi-agent architectures for requirements refinement. arXiv preprint arXiv:2505.20973, 2025.

Sukumar Ghosh, Arobinda Gupta, Ted Herman, and Sriram V. Pemmaraju. Fault-containing selfstabilizing distributed protocols. Distributed Computing, 20(1):53–73, 2007. doi: 10.1007/ s00446-007-0032-2.

Zhaoyang Guan, Huixi Cao, Ming Zhong, Yin Wang, Guanyu Liu, Eric Yang, Lynn Ai, Yongxin Ni, and Bill Shi. Symphony-Coord: Adaptive routing for multi-agent LLM systems. arXiv preprint arXiv:2602.00966, 2026.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021.

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jurgen Schmidhuber. MetaGPT: Meta programming for a multi-agent¨ collaborative framework. In International Conference on Learning Representations, 2024.

Jen-Tse Huang, Jiaxu Zhou, Tailin Jin, Xuhui Zhou, Zixi Chen, Wenxuan Wang, Youliang Yuan, Michael Lyu, and Maarten Sap. On the resilience of LLM-based multi-agent collaboration with faulty agents. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 26202–26226, 2025.

Peng Huang, Chuanxiong Guo, Lidong Zhou, Jacob R. Lorch, Yingnong Dang, Murali Chintalapati, and Randolph Yao. Gray failure: The achilles’ heel of cloud-scale systems. In Proceedings of the 16th Workshop on Hot Topics in Operating Systems, pp. 150–155, 2017. doi: 10.1145/3102980. 3103005.

Peng Huang, Chuanxiong Guo, Jacob R. Lorch, Lidong Zhou, and Yingnong Dang. Capturing and enhancing in situ system observability for failure detection. In 13th USENIX Symposium on Operating Systems Design and Implementation, pp. 1–16. USENIX Association, 2018.

Yongrae Jo and Chanik Park. Byzantine-robust decentralized coordination of LLM agents. arXiv preprint arXiv:2507.14928, 2025.

Mehran Kazemi, Bahare Fatemi, Hritik Bansal, John Palowitch, Chrysovalantis Anastasiou, Sanket Vaibhav Mehta, Lalit K. Jain, Virginia Aglietti, Disha Jindal, Peter Chen, Nishanth Dikkala, Gladys Tyen, Xin Liu, Uri Shalit, Silvia Chiappa, Kate Olszewska, Yi Tay, Vinh Q. Tran, Quoc V. Le, and Orhan Firat. BIG-Bench Extra Hard. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 26473–26501, 2025. doi: 10.18653/v1/2025.acl-long.1285.

Haejoon Lee, Vincent-Daniel Yun, Dimitra Panagou, and Sai Praneeth Karimireddy. Robust multiagent LLMs under byzantine faults. arXiv preprint arXiv:2605.09076, 2026.

Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. CAMEL: Communicative agents for “mind” exploration of large language model society. In Advances in Neural Information Processing Systems, volume 36, pp. 51991–52008, 2023.

Junyou Li, Qin Zhang, Yangbin Yu, Qiang Fu, and Deheng Ye. More agents is all you need. Transactions on Machine Learning Research, 2024.

Rui Li, Zeyu Zhang, Xiaohe Bo, Quanyu Dai, Chaozhuo Li, Feng Wen, and Xu Chen. Towards adaptive, scalable, and robust coordination of LLM agents: A dynamic ad-hoc networking perspective. arXiv preprint arXiv:2602.08009, 2026.

E. S. Page. Continuous inspection schemes. Biometrika, 41(1–2):100–115, 1954. doi: 10.1093/ biomet/41.1-2.100.

Harald Psaier and Schahram Dustdar. A survey on self-healing systems: Approaches and systems. Computing, 91(1):43–73, 2011. doi: 10.1007/s00607-010-0107-y.

Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, Juyuan Xu, Dahai Li, Zhiyuan Liu, and Maosong Sun. ChatDev: Communicative agents for software development. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 15174–15186, 2024. doi: 10.18653/v1/2024.acl-long.810.

Mirac Suzgun, Nathan Scales, Nathanael Scharli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung,¨ Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, and Jason Wei. Challenging bigbench tasks and whether chain-of-thought can solve them. In Findings of the Association for Computational Linguistics: ACL 2023, 2023.

Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang, and James Zou. Mixture-of-agents enhances large language model capabilities. In International Conference on Learning Representations, 2025.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In International Conference on Learning Representations, 2023.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multitask language understanding benchmark. Advances in Neural Information Processing Systems, 37: 95266–95290, 2024.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multi-agent conversation. arXiv preprint arXiv:2308.08155, 2023.

Hanqing Yang, Jingdi Chen, Marie Siew, Tania Lorido-Botran, and Carlee Joe-Wong. LLM-powered decentralized generative agents with adaptive hierarchical knowledge graph for cooperative planning. arXiv preprint arXiv:2502.05453, 2025a. URL https://arxiv.org/abs/2502. 05453.

Yingxuan Yang, Huacan Chai, Shuai Shao, Yuanyi Song, Siyuan Qi, Renting Rui, and Weinan Zhang. AgentNet: Decentralized evolutionary coordination for LLM-based multi-agent systems. In Advances in Neural Information Processing Systems, 2025b.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2210.03629.

Hongxin Zhang, Weihua Du, Jiaming Shan, Qinhong Zhou, Yilun Du, Joshua B. Tenenbaum, Tianmin Shu, and Chuang Gan. Building cooperative embodied agents modularly with large language models. In International Conference on Learning Representations, 2024. URL https: //arxiv.org/abs/2307.02485.

Shaokun Zhang, Ming Yin, Jieyu Zhang, Jiale Liu, Zhiguang Han, Jingyang Zhang, Beibin Li, Chi Wang, Huazheng Wang, Yiran Chen, et al. Which agent causes task failures and when? on automated failure attribution of llm multi-agent systems. arXiv preprint arXiv:2505.00212, 2025.

Lifan Zheng, Jiawei Chen, Qinghong Yin, Jingyuan Zhang, Xinyi Zeng, and Yu Tian. Rethinking the reliability of multi-agent system: A perspective from byzantine fault tolerance. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 35012–35020, 2026.

## A ABILITY ASSIGNMENT THROUGH SYSTEM PROMPTS

We test whether an ability assignment in the system prompt can create a reliable performance gap between assigned and unassigned abilities. We compare two conditions on BBH: no statement about competence and a statement that the model is unreliable on unassigned abilities. We use four ability assignments and five questions per task type. Each question is run twice at temperature zero, and a given model receives the same questions in both conditions. Llama-3.2-1B is a weak-model reference and is tested only without the statement; the other five models are evaluated under both conditions.

The accuracy gap is the accuracy on assigned abilities minus the accuracy on unassigned abilities. The additional gap is the change in this quantity relative to the condition with no competence statement. If the prompt reliably enforced specialization, it would enlarge this gap by lowering accuracy on unassigned abilities while preserving accuracy on assigned abilities.

Table A1: Results of ability assignment by system prompt on BBH.
<table><tr><td colspan="2"></td><td rowspan="2">Unassigned ability acc.</td><td rowspan="2">Assigned ability acc.</td><td rowspan="2">Accuracy gap</td><td rowspan="2">Additional gap</td></tr><tr><td>Model</td><td>Condition</td></tr><tr><td rowspan="2">DeepSeek-Chat</td><td>No statement</td><td>0.812</td><td>0.800</td><td>-0.012</td><td></td></tr><tr><td>Told unreliable</td><td>0.775</td><td>0.792</td><td>0.017</td><td>0.029</td></tr><tr><td rowspan="2">Llama-3.1-70B</td><td>No statement</td><td>0.781</td><td>0.783</td><td>0.002</td><td></td></tr><tr><td>Told unreliable</td><td>0.750</td><td>0.767</td><td>0.017</td><td>0.015</td></tr><tr><td rowspan="2">GPT-4o-mini</td><td>No statement</td><td>0.744</td><td>0.708</td><td>-0.035</td><td></td></tr><tr><td>Told unreliable</td><td>0.675</td><td>0.667</td><td>-0.008</td><td>0.027</td></tr><tr><td rowspan="2">GPT-OSS-120B</td><td>No statement</td><td>0.869</td><td>0.825</td><td>-0.044</td><td></td></tr><tr><td>Told unreliable</td><td>0.831</td><td>0.833</td><td>0.002</td><td>0.046</td></tr><tr><td rowspan="2">Qwen-2.5-7B</td><td>No statement</td><td>0.494</td><td>0.558</td><td>0.065</td><td></td></tr><tr><td>Told unreliable</td><td>0.219</td><td>0.342</td><td>0.123</td><td>0.058</td></tr><tr><td>Llama-3.2-1B</td><td>No statement</td><td>0.212</td><td>0.192</td><td>-0.021</td><td>一</td></tr></table>

Declaring unassigned abilities unreliable increases the accuracy gap by 0.015–0.058 across the five models tested under both conditions. Accuracy on assigned abilities also declines for four of the five models. These results suggest that the tested instructions affect overall response behavior without reliably restricting performance to the assigned abilities. The models can still solve tasks outside their assigned abilities.

In the main evaluation, each agent’s declared abilities determine the model used for execution together with its healthy or degraded condition. A healthy agent uses the strong model for a declared ability and the weak model for a missing ability; a degraded agent uses the weak model. With complementary ability pairs, an incorrect route can therefore change the model applied to part of the task and affect task accuracy.

## B IMPLEMENTATION DETAILS

## B.1 FULL PROCEDURE

Alg. 1 places routing, review, Takeover, detection, and reintegration in one procedure. The task state $\bar { X _ { t } } \ : = \ : ( Q _ { t } , T _ { t } , \bar { R } _ { t } , C _ { t } , Y _ { t } , P _ { t } )$ contains the query, task type, required and covered abilities, ordered accepted outputs, and routing history. Controller i stores its inferred routing state $h _ { i } \in$ {normal, watched, isolated}, recent evidence W<sub>i</sub> indexed by task type and ability, a cache $L _ { i }$ of score and routing-state summaries received from peers, and Bellman–Ford distance $D _ { i } ( s )$ to each ability s.

In ASSESS, H and X denote the ability set and task state supplied by the caller, and $b _ { \mathrm { p r o b e } }$ is true for a recovery probe. The procedure returns the final score q and the output ${ \widehat { y } } ,$ which is the original output under Accept and the Takeover synthesis under Takeover. We use $\tau _ { \mathrm { t o } } = 0 . 4 0$ for the low-score threshold and $\sigma _ { \mathrm { c o m } } = 0 . 5 5$ for the review-disagreement threshold.

Algorithm 1 MESHHEAL task execution and routing adaptation   
Require: Graph $\overline { { G = ( \nu , \varepsilon ) } }$ , abilities $\overline { { \{ \kappa _ { i } \} } } ,$ task stream, parameters in Tab. B11   
1: Initialize h<sub>i</sub> ← normal, ${ \mathcal { K } } _ { i } ^ { \mathrm { o r i g i n a l } }  { \mathcal { K } } _ { i } , W _ { i } , L _ { i }  \emptyset , D _ { i } ( s ) , p _ { i } ^ { \mathrm { p r o b e } }  - \infty$   
2: for each task t with query $\dot { Q } _ { t }$ and type $T _ { t }$ do   
3: sample $i \sim \mathrm { U n i f } ( \mathcal { V } ) ; \hat { C } _ { t } , Y _ { t } , \hat { P _ { t } } \gets \emptyset ; b _ { \mathrm { p r o b e } } \gets \mathsf { f a l s e } ; \widehat { y _ { t } } \gets \bot$   
4: if h<sub>i</sub> = isolated then   
5: $i \gets \mathrm { R a w R E L A Y } ( i , L _ { i } , D _ { i } )$   
6: end if   
7: $R _ { t } \gets \mathrm { I N F E R A B I L I T I E S } ( i , Q _ { t } , T _ { t } )$   
8: while $\widehat { y } _ { t } = \perp$ and budget remains do   
9: $M _ { t } \gets R _ { t } \setminus C _ { t }$   
10: if $h _ { i } =$ isolated and not $b _ { \mathrm { p r o b e } }$ then   
11: $i \gets \mathrm { R a w R E L A Y } ( i , \dot { L _ { i } } , D _ { i } )$ ; continue   
12: end if   
13: if not $b _ { \mathrm { p r o b e } }$ and there exists probe-eligible $j \in \mathcal N _ { i }$ with $\boldsymbol { K } _ { j } ^ { \mathrm { o r i g i n a l } } \cap { M } _ { t } \neq \boldsymbol { \emptyset }$ then   
14: select one such j; $p _ { j } ^ { \mathrm { p r o b e } }  t ; i  j ; b _ { \mathrm { p r o b e } } $ true; continue   
15: end if   
16: $H _ { i , t } \gets K _ { i } ^ { \mathrm { o r i g i n a l } } \cap M _ { t }$   
17: if $\dot { H } _ { i , t } = \dot { \emptyset }$ then   
18: $i \gets \mathrm { { N E X T H o P } } ( i , M _ { t } , L _ { i } , D _ { i } )$ ; continue   
19: else if $H _ { i , t } \subsetneq M _ { t }$ then   
20: $a \gets { \mathfrak { c } }$ Contribute   
21: else   
22: a ← Finalize   
23: end if   
24: $X _ { t } \gets ( Q _ { t } , T _ { t } , R _ { t } , C _ { t } , Y _ { t } , P _ { t } )$   
25: $y \gets \mathrm { G E N E R A T E } ( i , a , X _ { t } , H _ { i , t } )$   
26: $( q _ { i , t } , \widehat { y } ) \gets \mathrm { A s s E s S } ( i , y , H _ { i , t } , h _ { i } , b _ { \mathrm { p r o b e } } , X _ { t } )$   
27: $\begin{array} { r } { \dot { h _ { i } ^ { \prime } }  \mathrm { U P D A T E E V I D E N C E } ( i , T _ { t } , \dot { H _ { i , t } } , q _ { i , t } , \dot { W _ { i } } , L _ { i } ) } \end{array}$   
28: if $\mathbf { \bar { \boldsymbol { h } } } _ { i } ^ { \prime } \neq \boldsymbol { h } _ { i }$ then   
29: $h _ { i }  h _ { i } ^ { \prime }$   
30: $\mathbf { i } \mathbf { f } h _ { i } =$ isolated then   
31: $p _ { i } ^ { \mathrm { p r o b e } }  t$   
32: end if   
33: advertise eligibility and distance changes   
34: end if   
35: $b _ { \mathrm { p r o b e } } $ false   
36: if a = Contribute then   
37: append $\widehat { y } \mathrm { t o } Y _ { t } ; C _ { t } \gets C _ { t } \cup H _ { i , t } ;$ append i to $P _ { t }$   
38: $i \longleftarrow \mathrm { N E X T H O P } ( i , R _ { t } \setminus C _ { t } , L _ { i } , D _ { i } )$   
39: else   
40: ybt <sup>←</sup> yb   
41: end if   
42: end while   
43: if $\widehat { y } _ { t } = \perp$ then   
44: mark task t unanswered   
45: else   
46: output $\widehat { y } _ { t }$   
47: end if   
48: end for

The three collaboration actions specify how the task state changes. Handoff forwards the state unchanged; Contribute adds a reviewed partial result and expands $C _ { t } ;$ and Finalize completes the remaining work and proposes a full answer. Review then chooses Accept or Takeover. For Finalize, the controller returns the accepted answer or its replacement. Review determines whether the current output is kept or replaced, while the inferred routing state affects later routes. The displayed prompts and traces use these same action names.

UPDATEEVIDENCE records $q _ { i , t }$ for every $s \in H _ { i , t }$ , updates the matched peer-relative shortfall $x _ { i , t , s }$ when the required summaries are available, and appends the resulting evidence to $W _ { i } .$ . It then applies RELATIVEUPDATE to obtain the agent-level routing state $h _ { i } ^ { \prime } .$ , following the detector rules in Sec. 4.2.2.

Algorithm 2 MESHHEAL review and correction   
1: procedure ASSESS(i, y, $H , h _ { i } , b _ { \mathrm { p r o b e } } , X )$   
2: obtain independent scores u<sub>1</sub>, u<sub>2</sub>, u<sub>3</sub> from one eligible normal reviewer with the required ability   
3: q ← median ${ ' u 1 , u 2 , u 3 } ) ; s _ { u } \gets \mathrm { s d } _ { \mathrm { s a m p l e } } ( u _ { 1 } , u _ { 2 } , u _ { 3 } )$   
4: if $q _ { 0 } \leq \tau _ { \mathrm { t o } } \mathrm { o r } h _ { i } =$ watched or $b _ { \mathrm { p r o b e } } \mathrm { o r } s _ { u } \geq \sigma _ { \mathrm { c o m } }$ then   
5: select two distinct eligible reviewers A and B when possible; otherwise make two independent calls to the same   
eligible reviewer   
reviewers judge independently, exchange explanations, and judge again; let the final scores be $v _ { A } , v _ { B }$   
$q  ( v _ { A } \dot { + } \bar { v } _ { B } ) / 2$   
$\mathbf { i f } \ q \leq \tau _ { \mathrm { t o } }$ then   
yb <sup>←</sup> TAKEOVERSYNTHESIS(A, B, y, X)   
10: return $( q , \widehat { y } )$   
11: end if   
12: else   
13: ${ q  q _ { 0 } }$   
14: end if   
15: return $( q , y )$   
16: end procedure

RAWRELAY forwards the unchanged task state from an isolated agent, either the entry agent or an intermediate relay, toward a normal agent using $D _ { i } .$ without any LLM call.

NEXTHOP reads only the cached eligibility states of the current controller’s neighbors in $L _ { i }$ and its distance table $D _ { i } .$ . It samples a missing ability uniformly and routes toward its nearest holder using the cached Bellman–Ford distances. Normal and watched agents have identical execution eligibility; isolated agents are excluded as execution providers but may relay the task state without LLM calls.

ASSESS attributes $q _ { i , t }$ to the agent that generated the original output under both Accept and Takeover. RELATIVEUPDATE computes the shortfall $x _ { i , t , s }$ only from observations with the same task type and ability as the current output; the normal range $m _ { i , T }$ and $p _ { i , T } ^ { 2 5 }$ pools the mean shortfalls of normal neighbors over all abilities within the same task type T. The resulting threshold decision updates one inferred routing state for the agent. The procedure also requires a minimum number of observations, checks for two consecutive crossings of the isolation threshold, and applies reversible routing-state transitions. Score summaries update neighbor caches, while eligibility and distance advertisements propagate from neighbor to neighbor. Probe timers are independent local events, so no coordinator scans the network. Later tasks use the latest local eligibility and distance tables.

## B.2 AGENTS AND ABILITIES

Table B1: BBH ability assignment in the six-agent setting.
<table><tr><td>Agent</td><td>Declared abilities</td></tr><tr><td>A1</td><td>reasoning, mathematical, knowledge</td></tr><tr><td>A2</td><td>reasoning, mathematical, knowledge</td></tr><tr><td>A3 (degraded)</td><td>mathematical, knowledge, sequence, inference</td></tr><tr><td>A4</td><td>language, sequence, spatial, inference</td></tr><tr><td>A5 (degraded)</td><td>reasoning, language, spatial</td></tr><tr><td>A6</td><td>language, sequence, spatial, inference</td></tr></table>

Table B2: BBH task types and required abilities.
<table><tr><td>Task type</td><td>Required abilities</td></tr><tr><td>causal_judgement</td><td>reasoning + inference</td></tr><tr><td>formal_fallacies</td><td>reasoning + inference</td></tr><tr><td>tracking-shuffled_objects_five_objects</td><td>reasoning + sequence</td></tr><tr><td>geometric_shapes</td><td>mathematical + spatial</td></tr><tr><td>object_counting date_understanding</td><td>mathematical + spatial</td></tr><tr><td>ruin_names</td><td>mathematical + language language + knowledge</td></tr></table>

The MATH mapping separates operations on mathematical objects from the reasoning needed to formulate or complete a solution. Algebraic subjects pair symbolic transformation with multistep deduction. Geometry and precalculus pair calculation or theorem knowledge with reasoning about figures, while subjects solved by considering separate cases pair symbolic constraints with enumeration. The assignments follow common stages of mathematical problem solving and associate each ability with a specific step in the solution process.

Table B3: MATH ability assignment.  
Agent Declared abilities   
A1 symbolic manipulation, numeric computation, theorem knowledge   
A2 symbolic manipulation, numeric computation, theorem knowledge   
A3 (degraded) numeric computation, theorem knowledge, enumeration, multi step deduction   
A4 enumeration, multi step deduction, problem translation, figure reasoning   
A5 (degraded) symbolic manipulation, problem translation, figure reasoning   
A6 enumeration, multi step deduction, problem translation, figure reasoning

Table B4: MATH subjects, required abilities, and assignment rationale.  
Subject Required abilities Rationale   
prealgebra numeric computation + problem translation word-problem modeling + arithmetic   
algebra symbolic manipulation + multi step deduction expression transformation + multi-step   
solving   
geometry numeric computation + figure reasoning geometric relations + numerical calcula  
tion   
number theory symbolic manipulation + enumeration algebraic constraints + systematic cases   
counting\_and\_ symbolic manipulation + enumeration counting expressions + complete case cov   
probability erage   
intermediate algebra symbolic manipulation + multi step deduction deeper transformations + longer deduction   
precalculus theorem knowledge + figure reasoning function properties + graphical intuition

Table B5: MMLU-Pro ability assignment.  
Agent Declared abilities   
A1 humanities knowledge, science knowledge, quantitative methods   
A2 humanities knowledge, science knowledge, quantitative methods   
A3 (degraded) science knowledge, quantitative methods, scenario inference, stepwise analysis   
A4 scenario inference, stepwise analysis, text comprehension, structural reasoning   
A5 (degraded) humanities knowledge, text comprehension, structural reasoning   
A6 scenario inference, stepwise analysis, text comprehension, structural reasoning

The MMLU-Pro mapping pairs the main source of domain knowledge with the operation needed to apply it. Here, structural reasoning denotes reasoning about structures, processes, and physical representations. Knowledge-heavy domains require text comprehension, contextual inference, or reasoning about structure. Economics and physics combine quantitative methods with textual or structural interpretation. The two parts remain complementary: one agent supplies relevant knowledge or methods, and another applies them to the question.

The BBH assignments are inspired by the reasoning capabilities represented across BBH / BIG-Bench Extra Hard (Kazemi et al., 2025); MATH and MMLU-Pro use names chosen for each dataset while preserving the same controlled structure. Within each dataset, exactly three agents hold each ability, and every task combines two abilities with disjoint sets of agents. Under this assignment, at least two agents are needed to cover both required abilities. Every ability held by degraded agents A3 and A5 is also held by two healthy agents after both targets degrade. The mapping defines complementary ability sets and preserves alternatives to each degraded agent.

At task entry, INFERABILITIES receives the complete seven-ability vocabulary, brief ability descriptions, the query and task type, and examples covering all evaluated task types and their required ability pairs. The entry LLM selects a pair and stores it in R ; controllers then use this inferred pair for action selection and routing. Ability inference is therefore a constrained choice over the documented mappings in Tables B2, B4, and B6.

## B.3 MODELS, DATA, AND DEGRADATION

The main stream contains 504 tasks in 12 balanced blocks. Tasks 1–252 form the healthy phase. Tasks 253–504 use different questions from the same task-type distribution and form the degraded phase. At task 253, A3 and A5 switch from the strong model to the weak model while their declared ability sets remain unchanged.

A missing or unparsable answer is counted as incorrect. Table B7 shows that the LLM can accurately infer the abilities required for each task. Results for persistent degradation on each of BBH, MATH, and MMLU-Pro, together with the studies of alternative models and individual components, average five seeded task streams per setting. For the studies of degradation and recovery at different times, network scaling, and detection, the corresponding tables distinguish aggregated accuracies from agent counts; Tab. C5 reports a representative state trajectory.

Table B6: MMLU-Pro domains, required abilities, and assignment rationale.
<table><tr><td>Domain</td><td>Required abilities</td><td>Rationale</td></tr><tr><td>health</td><td>science_knowledge + text_comprehension</td><td>medical knowledge + comprehension of long questions</td></tr><tr><td>history</td><td>humanities_knowledge + scenario_inference</td><td>historical facts + contextual inference</td></tr><tr><td>biology</td><td>science_knowledge + structural_reasoning</td><td>biology knowledge + structural/process reason- ing</td></tr><tr><td>psychology</td><td>humanities_knowledge + scenario_inference</td><td>theory + scenario judgment</td></tr><tr><td>philosophy</td><td>humanities_knowledge + scenario_inference</td><td>doctrine + argumentative stance</td></tr><tr><td>economics</td><td>quantitative_methods + text_comprehension</td><td>interpretation + model/calculation</td></tr><tr><td>physics</td><td>quantitative_methods + structural_reasoning</td><td>quantitative analysis + physical representation</td></tr></table>

Table B7: Exact match between inferred ability pairs and the documented task-type mappings.
<table><tr><td>Dataset</td><td>BBH</td><td>MATH</td><td>MMLU-Pro</td><td>Overall</td></tr><tr><td>Exact pair match</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr></table>

The healthy and degraded phases preserve the task-type distribution and use disjoint questions, so the task mix remains comparable across distinct questions. Degradation is introduced by changing the model while preserving the agent identifier, declared ability set, and communication links. Task outputs and peer reviews provide the resulting evidence for routing-state updates.

## B.4 BASELINES

Each baseline uses the authors’ released implementation. We retain its native routing, selection, filtering, and aggregation logic and connect it to a shared execution layer that fixes the task order, agent identities, ability assignments, models, degradation schedule, worker prompts, and Handoff– Contribute–Finalize task interface. The resulting comparison covers decentralized routing or reputation (AgentNet and RAPS), mandatory central delegation (AutoGen), and repeated candidate generation and aggregation (SAC and Symphony). We count every model call and all model tokens across routing, execution, review, and aggregation, including repeated review samples and committee rounds. Token cost includes both input prompt tokens and generated output tokens.

## B.5 REVIEW, ACCEPT, AND TAKEOVER

For each Contribute or Finalize output, the controller samples one reviewer in the normal state that holds the abilities used to produce it and calls that reviewer three times in independent contexts. Watched and isolated peers are ineligible throughout initial review and committee selection. Every call uses the same rubric for that type of output and the same score definitions shown in App. D. The median of the initial scores is q . Let $\bar { u } = ( u _ { 1 } + u _ { 2 } + u _ { 3 } ) / 3$ . Disagreement among the three score is measured by the sample standard deviation

$$
s _ { u } = \sqrt { \frac { 1 } { 3 - 1 } \sum _ { r = 1 } ^ { 3 } ( u _ { r } - \bar { u } ) ^ { 2 } } .
$$

A two-reviewer committee is called when $q _ { 0 } \leq 0 . 4 0 $ , the agent is watched, the output is a recovery probe, or $s _ { u } \geq 0 . 5 5$ . All other outputs are accepted using the initial median score.

The controller selects distinct peers as Reviewer A and Reviewer B whenever two are available. If only one eligible reviewer is available, the controller uses that peer in two independent review contexts. The reviewers first score the output independently and explain their judgments. Each reviewer then sees both first-round scores and explanations and independently returns a second judgment. A score above 0.40 leads to Accept. A score at most 0.40 leads to Takeover: the two reviewers produce correction candidates, and a normal committee reviewer synthesizes them with the accepted task context. The controller replaces the original output with this synthesis. This replacement is not reviewed recursively; its cost is included, while the score attributed to the agent that generated the original output remains the detector evidence.

Table B8: Models used for task execution and induced degradation.
<table><tr><td>Execution condition</td><td>Model</td></tr><tr><td>Declared ability, healthy agent</td><td> $\mathsf { o p e n a i / q p t { - o s s - 1 2 0 b } }$ </td></tr><tr><td>Missing ability or degraded agent</td><td> $\mathtt { m e t a - l 1 a m a / l 1 a m a - 3 . 2 \mathrm { - 1 b - i n s t r u c t } }$ </td></tr></table>

Table B9: Datasets and scoring rules.
<table><tr><td>Dataset</td><td>Scoring rule</td></tr><tr><td>BIG-Bench Hard</td><td>Exact answer extraction and matching</td></tr><tr><td>MATH</td><td>Mathematical equivalence with math_equal</td></tr><tr><td>MMLU-Pro</td><td>Extracted option letter</td></tr></table>

Mixture-of-Agents allows the same LLM to be reused within or across layers and to generate independently sampled candidates before aggregation (Wang et al., 2025). We use this form of model reuse when only one eligible peer in the normal state has the relevant ability. The controller calls that peer in two independent contexts labeled Reviewer A and Reviewer B. This preserves review and Takeover availability and provides two separately sampled judgments. Every reviewed output produces one detector score, attributed to the agent that generated the original output under both Accept and Takeover.

The two-timescale procedure connects correction at the fast timescale to detection at the slow timescale. One reviewer provides the initial check, and committee discussion is reserved for lowscoring or uncertain outputs, outputs from watched agents, and recovery probes. Under Accept, the reviewed output remains in the task state. Under Takeover, the synthesized replacement takes it place. In both cases, the final review score remains attributed to the agent that generated the original output and supports later routing updates.

Degradation detector. Each output produced by agent i on task t receives a final review score $q _ { i , t }$ . For every ability $s \in H _ { i , }$ <sub>t</sub> used in that output, the controller records $( T _ { t } , s , q _ { i , t } )$ and maintains a separate detector for $( T _ { t } , s )$ . Let $\mathcal { P } _ { i , T , s } ( t )$ contain the direct neighbors of agent i that are currently in the normal state and have recent reviewed outputs for the same task type $T$ and ability s. Let $\mu _ { j , T , s } ( t )$ be neighbor $j ^ { \dagger } \mathbf { s }$ mean review score over these recent outputs. The reference score is

$$
\bar { q } _ { - i , T , s } ( t ) = \frac { 1 } { | \mathcal { P } _ { i , T , s } ( t ) | } \sum _ { j \in \mathcal { P } _ { i , T , s } ( t ) } \mu _ { j , T , s } ( t ) ,
$$

and the shortfall for ability s is

$$
x _ { i , t , s } = \bar { q } _ { - i , T _ { t } , s } ( t ) - q _ { i , t } .
$$

For each $( T , s )$ , controller i averages its latest eight matched shortfalls to obtain $\bar { x } _ { i , T , s }$ . For the second comparison, the controller uses the mean shortfalls maintained by its normal neighbors for all abilities under the same task type T. Let

$$
\begin{array} { r } { \mathscr { C } _ { i , T } ( t ) = \{ ( j , s ^ { \prime } ) : j \in \mathscr { N } _ { i } , \ h _ { j } = \mathop { \bf n o r m a l } , \ \mathrm { a n d } \ \bar { x } _ { j , T , s ^ { \prime } } \ \mathrm { i s ~ a v a i l a b l e } \} . } \end{array}
$$

Thus, a neighbor may contribute multiple values, one for each ability $s ^ { \prime }$ for which it has enough matched evidence. Let

$$
m _ { i , T } = \mathrm { m e d i a n } \left\{ \bar { x } _ { j , T , s ^ { \prime } } : ( j , s ^ { \prime } ) \in \mathcal { C } _ { i , T } ( t ) \right\} ,
$$

and

$$
p _ { i , T } ^ { 2 5 } = Q _ { 0 . 2 5 } \left( \left\{ \bar { x } _ { j , T , s ^ { \prime } } : ( j , s ^ { \prime } ) \in \mathcal { C } _ { i , T } ( t ) \right\} \right) .
$$

The detector statistic is

$$
z _ { i , T , s } = \frac { \bar { x } _ { i , T , s } - m _ { i , T } } { \operatorname* { m a x } \Bigl ( m _ { i , T } - p _ { i , T } ^ { 2 5 } , 0 . 1 0 \Bigr ) } .
$$

The first comparison is ability-specific, while the second comparison pools the mean shortfalls $\bar { x } _ { j , T , s ^ { \prime } }$ across abilities to estimate the normal range for task type T. The current output updates the statistic for its own task type and ability, while the inferred routing state applies to the agent. After three matched observations, a normal agent becomes watched when $z _ { i , T , s } \geq 1 . 2 .$ A normal or watched agent becomes isolated after $z _ { i , T , s } \geq 2 . 0$ on two consecutive updates with matched evidence; the counter resets when $z _ { i , T , s } < 2 . 0$ . Recovery applies the thresholds in reverse: an isolated agent moves to watched when $z _ { i , T , s } < 2 . 0$ , and a later watched update with $z _ { i , T , s } < 1 . 2$ restores normal status. The minimum of three matched observations applies separately to each agent, task type, and ability. If this minimum is not met, or either $\mathcal { P } _ { i , T , s } ( t )$ or $\mathcal { C } _ { i , T } ( t )$ is empty, the current routing state is retained.

Table B10: Baseline implementations.
<table><tr><td>Label</td><td>Source</td><td>Implementation used in our framework</td></tr><tr><td>AgentNet</td><td>Yang et al. (2025b)</td><td>Native decentralized routing with adaptive edge weights and pruning. Peer reputation updates local links and routing avoids low-reputation</td></tr><tr><td>RAPS</td><td>Li et al. (2026)</td><td>peers.</td></tr><tr><td>AutoGen</td><td>Wu et al. (2023)</td><td>A mandatory GroupChatManager-style hub delegates work, receives worker outputs, and produces the final answer.</td></tr><tr><td>SAC</td><td>Lee et al. (2026)</td><td>Three teams produce answers, run two filtering and refinement rounds, and aggregate by majority.</td></tr><tr><td>Symphony</td><td>Guan et al. (2026)</td><td>GlobalLinUCB selects agents with the required abilities; three plans run in parallel; the final result is selected by voting and used to update routing statistics.</td></tr></table>

Peer summaries use recent answers to different questions with the same task type and ability. The rolling window averages over question-level variation. After every matched review, the controller sends updated score and shortfall summaries to direct neighbors. Routing-state and distance summaries are sent when those values change so that neighboring controllers can update their routing tables.

Under the evaluated ability mappings, each produced output is associated with exactly one remaining required ability, $\mathrm { i . e . , } | H _ { i , t } | = \mathrm { \bar { 1 } }$ . Thus, each reviewed output updates one $( T , s )$ detector and one agent-level routing state.

## B.6 PARAMETERS

Table B11: Main MESHHEAL parameters.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Committee reviewer slots</td><td>2</td></tr><tr><td>Committee deliberation rounds</td><td>2</td></tr><tr><td>Takeover correction candidates</td><td>2</td></tr><tr><td>Routing hop limit</td><td>4 hops</td></tr><tr><td>Maximum agents per task</td><td>6</td></tr><tr><td>Maximum reviewed outputs per task</td><td>3</td></tr><tr><td>Evidence window per agent, task type, and ability</td><td>8 observations</td></tr><tr><td>Isolation threshold</td><td> $z \ge 2 . 0 ( 2$  consecutive updates)</td></tr><tr><td>Watch threshold</td><td> $z \geq 1 . 2$ </td></tr><tr><td>Scale floor</td><td>0.10</td></tr><tr><td>Minimum observations</td><td>3</td></tr><tr><td>Minimum recovery-probe interval</td><td>10 tasks</td></tr><tr><td>Takeover score threshold</td><td> $q \leq 0 . 4 0$ </td></tr><tr><td>Initial review samples</td><td>3</td></tr><tr><td>Committee sample-s.d. trigger</td><td> $s _ { u } \geq 0 . 5 5$ </td></tr></table>

The observation window and the requirement of two consecutive threshold crossings provide repeated evidence before isolation. The watched state adds scrutiny before isolation, and the local probe timer controls how quickly fresh evidence can replace degraded observations after recovery. Each isolated agent’s controller counts tasks since isolation or its last probe. After 10 tasks, the agent becomes probe-eligible; a neighbor schedules the probe when matching work next arrives, and the counter resets when the probe runs. Routing and review limits bound the amount of work per task. Low initial scores and high uncertainty trigger committee review and its additional cost; Takeover follows only when the committee’s final score remains at or below 0.40. Unless explicitly varied in an ablation, all rubrics and parameters for review and detection are fixed before evaluation and used unchanged across datasets, model pairs, and network sizes.

## B.7 SPARSE NETWORKS

All reviewers, including committee members, are direct neighbors of the agent whose output they review. In the evaluated sparse graphs, every agent producing an ordinary task output has at least one neighbor in the normal state with the same ability. Each agent selected for degradation has two such neighbors after both targets are excluded from reviewer selection, so its watched outputs and recovery probes use distinct peers for the two reviewer contexts. Another agent can have only one eligible neighbor with the same ability after an agent with that ability degrades; in that case, the controller reuses that peer in the two independent reviewer contexts described above. Every agent is also within three hops of an agent in the normal state for each required ability, leaving one hop of slack under the routing limit of four.

Table B12: Sparse graph size. Edge counts treat each bidirectional connection as one undirected link.
<table><tr><td>Agents</td><td>Sparse edges</td><td>Fraction of complete graph</td><td>Maximum ability distance</td></tr><tr><td>96</td><td>228</td><td>5.00%</td><td>3</td></tr><tr><td>288</td><td>705</td><td>1.71%</td><td>3</td></tr><tr><td>576</td><td>1454</td><td>0.88%</td><td>3</td></tr></table>

The sparse graphs preserve the local conditions for review and routing: every agent has a reviewer with the same ability, and every required ability is available from a reachable healthy agent within the routing limit. The detector compares eligible neighbors in the normal state. Edge density falls from $5 . 0 0 \%$ at 96 agents to 0.88% at 576 agents. The comparison between complete and sparse graphs changes connectivity while preserving these conditions.

## C DETAILED RESULTS

This appendix provides the detailed results summarized in Sec. 6. It first separates accuracy by dataset and reports inference cost. It then varies the model pair, degradation timing, and network topology before examining the committee and detector. Unless stated otherwise, all experiments use the task construction and cost accounting described in App. B.

## C.1 MAIN RESULTS

Table C1: Full persistent-degradation results, reported as mean ± standard deviation over five seeded task streams per dataset. Each stacked entry gives H above D for the healthy and degraded phases. The Mean column averages the three datasets within each seed, and Change is computed within seed before aggregation. Bold marks the best mean.
<table><tr><td>Method</td><td>BBH</td><td>MATH</td><td>MMLU-Pro</td><td>Mean</td><td>Change</td></tr><tr><td rowspan="2">MESHHEAL</td><td> $. 8 9 9 \pm . 0 2 7$ </td><td> $. 8 2 1 \pm . 0 1 9$ </td><td> $. 7 7 4 \pm . 0 1 6$ </td><td> ${ \bf . 8 3 1 \pm . 0 1 3 }$ </td><td></td></tr><tr><td> $\mathbf { 9 0 0 \pm . 0 1 4 }$ </td><td> $\mathbf { . 8 0 2 \pm . 0 2 1 }$ </td><td> $\mathbf { \delta } \mathbf { \delta } \mathbf { \cdot } \mathbf { 8 1 4 } \pm \mathbf { \delta } . \mathbf { 0 1 8 }$ </td><td> $\mathbf { 8 3 9 \pm . 0 1 2 }$ </td><td> $\mathbf { + . 0 0 7 \pm . 0 1 2 }$ </td></tr><tr><td rowspan="2">Symphony</td><td> $. 8 9 6 \pm . 0 2 2$ </td><td> $. 8 1 8 \pm . 0 2 0$ </td><td> $. 7 6 6 \pm . 0 1 8$ </td><td> $. 8 2 7 \pm . 0 1 3$ </td><td></td></tr><tr><td> $. 8 8 9 \pm . 0 0 7$ </td><td> $. 7 9 0 \pm . 0 2 2$ </td><td> $. 7 4 2 \pm . 0 2 6$ </td><td> $. 8 0 7 \pm . 0 1 4$ </td><td> $- . 0 2 0 \pm . 0 1 8$ </td></tr><tr><td rowspan="2">SAC</td><td> $. 8 9 6 \pm . 0 1 6$ </td><td> $. 7 9 4 \pm . 0 2 6$ </td><td>.778 ± .020</td><td> $. 8 2 2 \pm . 0 1 4$ </td><td></td></tr><tr><td> $. 8 9 4 \pm . 0 0 7$ </td><td> $. 6 3 9 \pm . 0 4 5$ </td><td>.706 ± .034</td><td> $. 7 4 6 \pm . 0 2 1$ </td><td> $- . 0 7 6 \pm . 0 2 6$ </td></tr><tr><td rowspan="2">RAPS</td><td>.884 ± .029</td><td> $. 7 2 6 \pm . 0 3 4$ </td><td> $. 7 9 0 \pm . 0 2 4$ </td><td> $. 8 0 0 \pm . 0 1 9$ </td><td></td></tr><tr><td>.740 ± .070</td><td>.627 ± .061</td><td> $. 5 9 1 \pm . 0 6 7$ </td><td> $. 6 5 3 \pm . 0 4 1$ </td><td> $- . 1 4 7 \pm . 0 4 5$ </td></tr><tr><td rowspan="2">AgentNet</td><td> $. 8 6 2 \pm . 0 3 5$ </td><td> $. 6 3 1 \pm . 0 4 1$ </td><td> $. 6 9 4 \pm . 0 3 4$ </td><td> $. 7 2 9 \pm . 0 2 3$ </td><td></td></tr><tr><td> $. 6 6 8 \pm . 0 6 6$ </td><td> $. 4 7 6 \pm . 0 5 6$ </td><td> $. 5 2 8 \pm . 0 6 1$ </td><td> $. 5 5 7 \pm . 0 3 9$ </td><td> $- . 1 7 2 \pm . 0 4 7$ </td></tr><tr><td rowspan="2">AutoGen</td><td> $. 8 9 8 \pm . 0 1 3$ </td><td> $. 6 6 5 \pm . 0 3 0$ </td><td> $. 7 9 6 \pm . 0 1 9$ </td><td> $. 7 8 7 \pm . 0 1 4$ </td><td></td></tr><tr><td> $. 5 2 5 \pm . 0 3 4$ </td><td> $. 2 1 4 \pm . 0 2 9$ </td><td> $. 5 3 2 \pm . 0 4 3$ </td><td> $. 4 2 4 \pm . 0 2 3$ </td><td> $- . 3 6 3 \pm . 0 2 8$ </td></tr></table>

MESHHEAL has the highest mean degraded-phase accuracy on all three datasets. Its aggregate accuracy is $0 . 8 3 1 \pm 0 . 0 1 3$ in the healthy phase and $0 . 8 3 9 \pm 0 . 0 1 2$ in the degraded phase, with a paired phase change of $+ 0 . 0 0 7 \pm 0 . 0 \dot { 1 } 2 .$ . On MATH, its mean changes from $0 . 8 2 1 \pm 0 . 0 1 9$ to $0 . 8 0 2 \pm 0 . 0 2 1$ , while Symphony and SAC show mean changes of −0.028 and −0.155. BBH and MMLU-Pro have nonnegative mean changes. RAPS, AgentNet, and AutoGen show mean phase changes of $- 0 . 1 4 7 \pm 0 . { \bar { 0 4 5 } } ,$ $- 0 . 1 7 2 \pm 0 . 0 4 7 .$ , and $- 0 . 3 6 3 \pm 0 . 0 2 8$

After degradation onset, MESHHEAL’s mean model calls increase from $1 2 . 4 \pm 0 . 2$ to $1 3 . 6 \pm 0 . 3$ per task because the system performs more review and rerouting, while mean total model tokens are (51 ± 3)k and $( 5 1 \pm 5 ) \mathbf { k }$ in the two phases. Symphony’s mean token cost increases from $( 8 6 \pm 5 ) \mathrm { k }$ 人

Table C2: Healthy- and degraded-phase cost per task, reported as mean ± standard deviation over five seeds after averaging datasets within seed. Tokens include both input and output tokens; their means and standard deviations are rounded to the nearest thousand.
<table><tr><td>Method</td><td>H calls</td><td></td><td>D calls H tokens (k)</td><td>) D tokens (k)</td></tr><tr><td>MESHHEAL</td><td> $1 2 . 4 \pm 0 . 2$ </td><td> $1 3 . 6 \pm 0 . 3$ </td><td> $5 1 \pm 3$ </td><td> $5 1 \pm 5$ </td></tr><tr><td>Symphony</td><td> $1 8 . 1 \pm 0 . 1$ </td><td> $2 3 . 4 \pm 0 . 1$ </td><td> $8 6 \pm 5$ </td><td> $1 1 5 \pm 9$ </td></tr><tr><td>SAC</td><td> $3 0 . 3 \pm 0 . 1$ </td><td> $3 9 . 9 \pm 0 . 6$ </td><td> $1 1 6 \pm 4$ </td><td> $1 7 8 \pm 1 3$ </td></tr><tr><td>RAPS</td><td> $8 . 2 \pm 0 . 5$ </td><td> $1 0 . 9 \pm 0 . 3$ </td><td> $3 9 \pm 5$ </td><td> $4 8 \pm 3$ </td></tr><tr><td>AgentNet</td><td> $7 . 8 \pm 0 . 7$ </td><td> $1 0 . 3 \pm 0 . 2$ </td><td> $3 4 \pm 4$ </td><td> $4 3 \pm 2$ </td></tr><tr><td>AutoGen</td><td> $6 . 1 \pm 0 . 0$ </td><td> $6 . 0 \pm 0 . 0$ </td><td> $1 5 \pm 0$ </td><td> $1 5 \pm 0$ </td></tr></table>

to $( 1 1 5 \pm 9 ) \mathbf { k }$ , and SAC’s from $( 1 1 6 \pm 4 ) \mathbf { k }$ to $( 1 7 8 \pm 1 3 ) 1$ k as they repeatedly generate and refine parallel candidates. RAPS and AgentNet have lower mean token costs together with mean accuracy changes of $- 0 . 1 4 7 \pm 0 . 0 4 5$ and $- 0 . 1 7 2 \pm 0 . 0 4 7$ . AutoGen has both the lowest mean cost and the lowest mean degraded-phase accuracy when its mandatory manager degrades.

## C.2 MODELS AND RECOVERY

Table C3: Model pair A: GPT-OSS-120B is the strong model and Gemma-3-4B is the weak model. Results are mean ± standard deviation over five seeded task streams; Change is paired within seed and tokens include both input and output tokens.
<table><tr><td>Method</td><td>Healthy acc.</td><td>Degraded acc.</td><td>Change</td><td>Tokens/task</td></tr><tr><td>MESHHEAL</td><td> $\mathbf { 0 . 9 2 9 \pm 0 . 0 1 8 }$ </td><td> $\mathbf { 0 . 9 0 5 \pm 0 . 0 1 6 }$ </td><td> $\mathbf { - 0 . 0 2 4 \pm 0 . 0 1 4 }$ </td><td> $^ { 5 6 , 6 8 7 \pm 2 , 4 3 1 }$ </td></tr><tr><td>Symphony</td><td> $0 . 9 1 7 \pm 0 . 0 2 0$ </td><td> $0 . 8 6 5 \pm 0 . 0 2 1$ </td><td> $- 0 . 0 5 2 \pm 0 . 0 1 7$ </td><td> $1 0 0 , 6 8 5 \pm 4 , 7 8 2$ </td></tr><tr><td>SAC</td><td> $0 . 9 1 7 \pm 0 . 0 1 9$ </td><td> $0 . 8 2 9 \pm 0 . 0 3 0$ </td><td> $- 0 . 0 8 8 \pm 0 . 0 2 6$ </td><td> $1 3 0 , 9 3 8 \pm 6 , 2 1 4$ </td></tr><tr><td>RAPS</td><td> $0 . 8 7 7 \pm 0 . 0 3 1$ </td><td> $0 . 6 2 3 \pm 0 . 0 6 4$ </td><td> $- 0 . 2 5 4 \pm 0 . 0 5 5$ </td><td> $3 5 , 7 4 6 \pm 2 , 9 6 3$ </td></tr><tr><td>AgentNet</td><td> $0 . 8 5 7 \pm 0 . 0 3 7$ </td><td> $0 . 6 9 4 \pm 0 . 0 5 6$ </td><td> $- 0 . 1 6 3 \pm 0 . 0 4 8$ </td><td> $^ { 3 6 , 9 7 8 \pm 3 , 2 8 4 }$ </td></tr><tr><td>AutoGen</td><td> $0 . 9 0 5 \pm 0 . 0 1 6$ </td><td> $0 . 7 3 4 \pm 0 . 0 4 1$ </td><td> $- 0 . 1 7 1 \pm 0 . 0 3 8$ </td><td> $1 5 , 3 7 7 \pm 1 , 1 2 6$ </td></tr></table>

Table C4: Model pair B: Llama-3.3-70B is the strong model and Llama-3.2-1B is the weak model. Results are mean ± standard deviation over five seeded task streams; Change is paired within seed and tokens include both input and output tokens.
<table><tr><td>Method</td><td>Healthy acc.</td><td>Degraded acc.</td><td>Change</td><td>Tokens/task</td></tr><tr><td>MESHHEAL</td><td> $0 . 7 9 0 \pm 0 . 0 2 5$ </td><td> $0 . 7 8 2 \pm 0 . 0 2 3$ </td><td> $\mathbf { - 0 . 0 0 8 \pm 0 . 0 1 5 }$ </td><td> $^ { 5 0 , 8 0 2 \pm 2 , 3 1 7 }$ </td></tr><tr><td>Symphony</td><td> $\mathbf { 0 . 8 0 6 \pm 0 . 0 2 3 }$ </td><td> $\mathbf { 0 . 7 8 6 \pm 0 . 0 2 1 }$ </td><td> $- 0 . 0 2 0 \pm 0 . 0 1 7$ </td><td> $^ { 9 6 , 9 8 2 \pm 4 , 5 6 3 }$ </td></tr><tr><td>SAC</td><td> $\mathbf { 0 . 8 0 6 \pm 0 . 0 2 2 }$ </td><td> $0 . 7 5 4 \pm 0 . 0 3 2$ </td><td> $- 0 . 0 5 2 \pm 0 . 0 2 7$ </td><td> $1 2 2 , 2 3 4 \pm 5 , 9 0 8$ </td></tr><tr><td>RAPS</td><td> $0 . 7 9 0 \pm 0 . 0 3 0$ </td><td> $0 . 5 7 1 \pm 0 . 0 6 1$ </td><td> $- 0 . 2 1 9 \pm 0 . 0 5 3$ </td><td> $\bar { 3 0 , 7 6 1 } \pm 2 , 6 8 4$ </td></tr><tr><td>AgentNet</td><td> $0 . 7 7 4 \pm 0 . 0 3 8$ </td><td> $0 . 4 7 6 \pm 0 . 0 6 5$ </td><td> $- 0 . 2 9 8 \pm 0 . 0 5 6$ </td><td> $\phantom { - } 2 9 , 4 8 7 \pm 2 , 7 9 2$ </td></tr><tr><td>AutoGen</td><td> $0 . 7 5 4 \pm 0 . 0 2 8$ </td><td> $0 . 3 4 9 \pm 0 . 0 4 2$ </td><td> $- 0 . 4 0 5 \pm 0 . 0 3 8$ </td><td> $1 2 , 9 8 6 \pm 9 5 6$ </td></tr></table>

The accuracy ranking changes with the model pair, while MESHHEAL uses fewer tokens than Symphony under both pairs. Under pair A, MESHHEAL’s mean degraded-phase accuracy is 0.905 ± 0.016, compared with $0 . 8 6 5 \pm 0 . 0 2 \bar { 1 }$ for Symphony, and its mean token cost is 56,687 versus 100,685. Under pair B, the degraded-phase means are $0 . 7 8 2 \pm 0 . 0 2 3$ and $0 . 7 8 6 \pm 0 . 0 2 1$ ; MESHHEAL has a smaller mean paired phase change $( - 0 . 0 0 8 \pm 0 . 0 1 5$ versus $- 0 . 0 2 0 \pm 0 . 0 1 7 )$ and uses 52% of Symphony’s mean tokens.

Tab. 3 reports mean results over five seeds for the experiment with degradation and recovery at different times. All methods complete the full 630-task stream, with 126 tasks in each phase. The fully recovered mean accuracy is 0.857, equivalent to approximately 108 correct answers per 126-task phase. Across the 504 tasks after the first agent degrades, the reported mean pooled accuracy is 0.812 at 49k tokens per task, compared with 0.806 and 105k for Symphony. Mean accuracy rises from 0.770 when both targets are degraded to 0.841 after A5 recovers and 0.857 after both recover. RAPS and AgentNet reach 0.524 and 0.563 after both models are restored.

Warm-up is an evidence-collection phase in which all agents remain in ordinary routing. In the representative trajectory in Tab. C5, A5 becomes watched at task 129 and isolated at task 132, five tasks after degradation begins. A3 becomes watched at task 267 and isolated at task 268, giving a detection delay of 15 tasks. Both remain isolated throughout their degraded intervals, during which they are observed only through recovery probes. After backend restoration, A5 and A3 move to watched at tasks 422 and 548 and return to normal routing at tasks 430 and 550, giving reintegration delays of 51 and 45 tasks, respectively.

![](images/d8e5144ee7b4d5700ed07d11559343da7a89802034ccdf0d71f64ff12177298f.jpg)  
Figure C1: Detection and reintegration under degradation and recovery at different times. Pink regions mark injected degradation. Each marker is one reviewed output, colored by the inferred state after the update; hollow markers are recovery probes. The bottom band shows the inferred state at each task, the curve shows the mean review score, and the blue region is the reference range for normal agents.

Table C5: Inferred routing-state changes in a representative run with degradation and recovery at different times.
<table><tr><td>Agent</td><td>True degraded interval</td><td>State sequence</td></tr><tr><td>A5</td><td>127-378</td><td>first evidence at task 3; normal during warm-up → 129 watched → 132 isolated → 422 watched → 430 normal</td></tr><tr><td>A3</td><td>253-504</td><td>first evidence at task 4; normal during warm-up → 267 watched → 268 isolated → 548 watched → 550 normal</td></tr></table>

## C.3 LARGER NETWORKS

Tab. 2(b) gives the accuracy and isolation results for complete and sparse graphs. Every degraded agent is isolated in every topology. Complete-graph accuracy ranges from 0.808 to 0.827, and sparse-graph accuracy ranges from 0.803 to 0.818. The largest difference is 0.010 at 288 agents; the sparse graph is 0.006 higher at 576 agents. Expanding the task stream with the number of replicated six-agent ability profiles keeps detector evidence per agent comparable across scales. Tab. B12 gives the graph sizes and confirms that the maximum distance to a healthy agent with a required ability remains within the routing hop limit.

## C.4 ABLATIONS

Table C6: Component study, reported as mean ± standard deviation over five seeded task streams. Phase change is paired within seed.
<table><tr><td>Metric</td><td>Full method</td><td>Single reviewer</td><td>No Takeover</td></tr><tr><td>Healthy accuracy</td><td> $0 . 8 9 9 \pm 0 . 0 2 7$ </td><td> $0 . 9 0 9 \pm 0 . 0 2 2$ </td><td> $0 . 8 7 9 \pm 0 . 0 2 1$ </td></tr><tr><td>Degraded-phase accuracy</td><td> $\mathbf { 0 . 9 0 0 \pm 0 . 0 1 4 }$ </td><td> $0 . 8 6 5 \pm 0 . 0 2 9$ </td><td> $0 . 8 4 4 \pm 0 . 0 3 6$ </td></tr><tr><td>Paired phase change</td><td> $\mathbf { + 0 . 0 0 1 \pm 0 . 0 1 6 }$ </td><td> $- 0 . 0 4 4 \pm 0 . 0 2 4$ </td><td> $- 0 . 0 3 5 \pm 0 . 0 3 0$ </td></tr></table>

The Single reviewer variant uses the initial reviewer with the same ability for the final judgment and any Takeover correction; it retains detection and rerouting while removing multi-reviewer discussion. The No Takeover variant retains review, committee escalation, detection, and rerouting but leaves a low-scoring output in the current task. The full configuration exceeds their mean degraded-phase accuracies by 0.035 and 0.056, respectively. No Takeover also lowers the healthy-phase mean by 0.020 because it leaves occasional low-quality outputs from healthy LLMs unchanged. The reported paired phase changes are $+ 0 . 0 0 1 \pm 0 . 0 \bar { 1 } 6 , \bar { - 0 . 0 4 4 } \pm 0 . 0 2 4$ , and $- 0 . 0 3 5 \pm 0 . 0 3 0 .$

We next compare the detector based on peer comparisons with a fixed-threshold rule and a cumulativesum (CUSUM) rule under the two model pairs (Page, 1954).

Table C7: Detector performance across two BBH model pairs. Missed counts are out of two degraded agents and false-isolation counts are out of four healthy agents; each setting uses the full 504-task stream.
<table><tr><td rowspan="2">Rule</td><td rowspan="2">Evidence</td><td colspan="2">Pair A</td><td colspan="2">Pair B</td></tr><tr><td>Missed</td><td>False</td><td>Missed</td><td>False</td></tr><tr><td>Absolute threshold</td><td>Own scores</td><td>0/2</td><td>4/4</td><td>0/2</td><td>0/4</td></tr><tr><td>CUSUM</td><td>Own scores</td><td>0/2</td><td>2/4</td><td>1/2</td><td>0/4</td></tr><tr><td>Relative peer comparison</td><td>Scores relative to peers</td><td>0/2</td><td>0/4</td><td>0/2</td><td>0/4</td></tr></table>

The absolute rule detects both degraded agents under both model pairs. Under pair A, it also isolates all four healthy agents. CUSUM reduces those false isolations to two and misses one degraded agent under pair B. Using the same review rubrics and detector parameters, relative peer comparison produces neither error under either model pair. Its reference adapts to peer scores for the same task type and ability while the decision thresholds remain fixed. The peer-relative reference relies on matched normal-neighbor summaries remaining representative of healthy performance.

The component study evaluates committee review and Takeover at the fast timescale. The detector comparison evaluates how peer-based evidence changes routing at the slow timescale under the two tested model pairs. Together, these studies support using review scores for both current-task correction and future routing.

## C.5 REVIEW AND TAKEOVER AUDIT

We examine the initial review, committee reassessment, and Takeover across five seeded task streams. In the first two panels of Tab. C8, outputs from degraded agents are treated as positives and outputs from healthy agents as negatives. These labels describe the agent’s injected condition, not the correctness of each output. A healthy agent can still produce an incorrect output, so the reported rates measure discrimination between injected conditions rather than detection of incorrect answers.

The initial review assigns mean scores of $0 . 8 9 \pm 0 . 0 1$ and $0 . 0 8 \pm 0 . 0 1$ to outputs from healthy and degraded agents, respectively. Each seed contains 31–33 outputs from degraded agents in this audit, of which 2–4 are missed by the initial low-score decision. The committee escalation rate is higher because watched status or disagreement among the initial scores can also trigger committee review.

After committee reassessment, recall changes from $0 . 9 0 \pm 0 . 0 3$ to $0 . 8 8 \pm 0 . 0 1$ , while the false-alarm rate decreases from $0 . 0 5 3 \pm 0 . 0 0 7 \mathrm { t o } 0 . 0 4 \dot { 4 } \pm 0 . 0 0 8$ . Among initially flagged outputs, the committee clears $0 . 3 3 \pm 0 . 0 7$ of healthy-agent cases and retains $0 . 9 5 \pm 0 . 0 2$ of degraded-agent cases.

Finalize answers can be checked directly against the gold answer. Among 32 initially wrong answers entering Takeover, 15 are corrected. The synthesis changes 18 of these answers, and 15 of the 18 changes are correct; the other 14 wrong answers are left unchanged. Among 32 initially correct answers entering Takeover, only one is changed to an incorrect answer.

A Contribute output cannot be scored directly from the final gold answer. Among the 137 tasks where Takeover replaces a Contribute output, 86% end with a correct final answer. This number reports the final task outcome and is not a direct measure of intermediate output correctness.

## D RUNTIME PROMPTS

This appendix presents representative runtime prompts for communicating ability matches to the LLM and reviewing outputs. Shaded cards distinguish prompt instructions from explanatory prose. Alg. 1 specifies the surrounding controller logic. For presentation, we standardize ability terminology and the action names Handoff, Contribute, and Finalize. Executable JSON field names, other wording, thresholds, and placeholders are retained; line breaks are adjusted to fit the page.

## D.1 REVIEW PROMPTS

Review depends on the task action. A Contribute reviewer evaluates the assigned intermediate output.   
A Finalize reviewer evaluates the finalizer’s own output and whether it commits a concrete answer.

<table><tr><td>(a) Initial review Metric</td><td>Value</td></tr><tr><td rowspan="5">Mean score, healthy-agent outputs Mean score, degraded-agent outputs</td><td> $0 . 8 9 \pm 0 . 0 1$ </td></tr><tr><td></td></tr><tr><td> $0 . 0 8 \pm 0 . 0 1$ </td></tr><tr><td> $0 . 9 0 \pm 0 . 0 3$ </td></tr><tr><td> $0 . 0 5 3 \pm 0 . 0 0 7$   $0 . 9 8 \pm 0 . 0 2$ </td></tr><tr><td colspan="2">(b) Initial review and committee reassessment</td></tr><tr><td colspan="2">Metric Recall on degraded-agent outputs</td></tr><tr><td colspan="2">False-alarm rate on healthy-agent outputs</td></tr><tr><td colspan="2"> $0 . 0 5 3 \pm 0 . 0 0 7$  Initial flags cleared, healthy-agent outputs</td></tr><tr><td colspan="2">Initial flags kept, degraded-agent outputs</td></tr><tr><td colspan="2"></td></tr><tr><td colspan="2">(c) Takeover outcomes Outcome Result</td></tr><tr><td colspan="2">Finalize answers wrong before Takeover (n = 32) corrected 15/32 (47%)</td></tr><tr><td colspan="2">changed but still wrong 3/32 (9%) 14/32 (44%)</td></tr><tr><td colspan="2">left unchanged 15/18 (83%)</td></tr><tr><td colspan="2">corrected, among those changed</td></tr><tr><td colspan="2"></td></tr><tr><td colspan="2">Finalize answers correct before Takeover (n = 32)</td></tr><tr><td colspan="2">changed to incorrect</td></tr><tr><td colspan="2">1/32 (3%)</td></tr><tr><td colspan="2"></td></tr><tr><td colspan="2">Takeover of Contribute outputs (n = 137)</td></tr><tr><td colspan="2"></td></tr><tr><td colspan="2">task correct at the end 86%</td></tr></table>

Table C8: Agent review performance and Takeover outcomes.

The prompt identifiers use the same Contribute and Finalize names as the method. In both cases, the controller selects a reviewer that holds the relevant ability. Process issues such as exceeding the assigned scope or ignoring a draft remain advisory unless they create a concrete correctness error. Accept and Takeover are controller actions derived from the resulting score.

The cards retain the executed instructions with ability terminology and action labels standardized for presentation. The assigned part refers to work requiring one of the task’s two abilities; draft-half refers to the earlier output covering the other ability. Executable JSON field names are unchanged.

Contribute judge system prompt (COMMITTEE CONTRIBUTE JUDGE SYSTEM)   
You review a DRAFTER’s INTERMEDIATE contribution in a multi-agent system. The   
drafter was delegated just ONE part of a task requiring two abilities; it   
should Contribute that half and leave the other half for Handoff to a teammate   
it should NOT Finalize (that needs the other ability too). You give one noisy   
review signal; do not solve, repair, or decide degradation. Judge ONLY the   
drafter’s half, using only a ability you actually hold (see   
REVIEWER\_COMPETENCE). Return only valid JSON.

The scoring schema anchors the returned signal and requires every low score to identify a concrete correctness error. The controller assigns reviewers who hold the relevant ability. If a reviewer does not hold that ability, the prompt directs it to return a neutral score with low confidence.

Contribute judge scoring schema   
Score each in [0,1]:   
- half\_correct: ONLY IF you are competent at the drafter’s ability - re  
derive that half; is it correct? LOW for a concrete error (miscalculation,   
invalid step, contradiction with the given facts/objects). If you are NOT   
competent at that ability, set 0.5 and lower confidence.   
- in\_scope: did the drafter STAY IN ITS LANE - do its half and leave the final   
decision OPEN?   
SCORING RULE - normal\_working\_score MUST equal half\_correct: score the   
CORRECTNESS of the delegated half only. in\_scope is ADVISORY: it must NOT lower   
normal\_working\_score.   
anchors: 1.00 half is sound / 0.50 cannot verify or weak / 0.25 concrete error /   
0.00 unusable garbage for the delegated half.   
Return ONLY valid JSON:   
{ "half\_correct": .., "in\_scope": .., "normal\_working\_score": .., "confidence": ..,   
"specific\_failure\_found": bool, "failure\_type": "none|wrong\_half|over\_reach|other",   
"failure\_evidence": "quote from the drafter output; empty if none",   
"one\_sentence\_reason": ".." }   
Constraints: normal\_working\_score<=0.40 => specific\_failure\_found=true + non  
empty evidence of a CORRECTNESS failure; over\_reach alone NEVER sets   
specific\_failure\_found.

Finalize judge system prompt (COMMITTEE FINALIZE JUDGE SYSTEM)   
You review a FINALIZER’s answer in a multi-agent system. The finalizer did its   
OWN part and must COMBINE it with the draft-half already in the trajectory to   
commit ONE concrete final answer. The draft-half was produced by a teammate   
using a ability the finalizer (and you) do NOT hold: treat it as GIVEN - this   
review does not score whether the draft-half is right. Judge only the   
finalizer’s own half and how faithfully it combined. You give one noisy review   
signal; do not solve/repair. Return only valid JSON.

Finalize returns one concrete, scoreable answer; if no concrete answer is given, the score is at most 0.25. It uses the same numerical anchors and the same requirement to identify a concrete error as Contribute, applied to the finalizer’s work within its declared ability and the committed answer. Draft handling affects the score when it creates a concrete error in the finalizer’s own work or in the combined answer.

## D.2 ABILITY PROMPTS

Before execution, the entry LLM infers the required ability pair from the complete dataset ability vocabulary, brief descriptions, and examples mapping all evaluated task types to ability pairs. The controller stores this pair in the task state. It then compares the inferred pair with the agent’s declared abilities and inserts one of three directives into the execution prompt. The evaluation selects the strong or weak model from the ability match and the agent’s injected condition; the directive communicates the ability match to the LLM.

Complete competence match   
[COMPETENCE VERDICT - DECIDED BY SYSTEM, BINDING]   
This task requires: {req}. You ARE competent at all of them. Solve it correctly and   
carefully.   
On the FIRST line output exactly ’COMPETENCE\_MATCH: yes’, then give your answer.

[COMPETENCE NOTE - PARTIAL FIT]   
You can reliably handle only the parts of this task that fall WITHIN YOUR OWN   
abilities...   
- If you select Finalize: you MUST commit a single concrete final answer,   
combining the part you can do with whatever the progress already provides for the   
rest.   
- If you are only contributing an INTERMEDIATE part (a Contribute draft): do ONLY the   
part your   
abilities cover, state that partial result, and request Handoff for the remaining   
work.   
On the FIRST line output exactly ’COMPETENCE\_MATCH: partial’, then give your answer/   
contribution.

No competence match   
[COMPETENCE NOTE]   
This task requires: {req}, none of which are your strengths. Answer as best you can,   
but treat your own answer here as unreliable.   
On the FIRST line output exactly ’COMPETENCE\_MATCH: no’, then give your answer.  
The no-match directive applies to generation assigned outside an agent’s declared abilities. In MESHHEAL, an agent with none of the remaining required abilities instead performs controller-only Handoff. App. A reports the limited increase in the accuracy gap between assigned and unassigned abilities under the tested prompts.

What this case shows

## E CASE STUDIES

We present four representative execution traces showing collaboration across complementary abilities, multi-hop routing around an isolated agent, committee review and Takeover after an incorrect output, and correction of a localized reasoning error. For presentation, we standardize ability terminology and action labels to match Sections 4 and B. Recorded answers, numerical values, and executable field names are retained; logging metadata not needed to follow the execution is omitted.

## E.1 COLLABORATION AND ROUTING

Case 1: Complementary abilities across two agents   
Task: object counting.   
Required abilities: mathematical + spatial.   
Route: A4 → A1.   
Outcome: success; gold answer 15.   
Question: “I have a drum, four pianos, two accordions, a carrot, a stalk of celery, three flutes, a   
cauliflower, a violin, a clarinet, a trombone, and two trumpets. How many musical instruments   
do I have?”

A4 is assigned the spatial ability but not the mathematical ability. It contributes the part corresponding to its assigned ability and leaves the remaining work to another agent.

A4 Contribute: Filter and enumerate   
RESULT: Based on the provided list, the musical instruments and their quantities are:   
- Drum: 1   
- Piano: 4   
- Accordion: 2   
- Flute: 3   
- Violin: 1   
- Clarinet: 1   
- Trombone: 1   
Trumpet: 2   
Non-instrument items (carrot, celery, cauliflower) are excluded.   
The arithmetic summation can be completed by a teammate.

The spatial component is now covered, while the mathematical component remains. The controller therefore routes the updated task state to A1, which holds the missing ability and is in the normal routing state.

Controller routing: A4 to A1   
Covered ability: spatial   
Remaining ability: mathematical   
NEXT\_AGENT\_ID: 1

A1 receives the accepted output and completes the remaining operation.

A1 Finalize: Complete the mathematical component

DESCRIPTION: Using the counts from Agent 4:   
1 + 4 + 2 + 3 + 1 + 1 + 1 + 2 = 15.   
RESULT: 15

The task is solved through complementary outputs from two agents. A4 performs the spatial filtering and enumeration, routing carries that result, and A1 uses the remaining mathematical ability in the Finalize step. The trace illustrates how declared abilities determine both the division of work and the next routing decision.

What this case shows (continued)

Case 2: Multi-hop routing around an isolated agent

Task: number theory.   
Required abilities: symbolic manipulation + enumeration.   
Network: 288 agents, 705 bidirectional links, average degree 4.9, with 1.7% as many edges as   
the corresponding complete graph.   
Outcome: success; gold answer 1575.   
Question excerpt: “A school has between 150 and 200 students... What is the sum of all possible   
numbers of students enrolled?”

Agent 61 first derives the candidate values needed for the remainder of the solution.

Agent 61 Contribute: Symbolic manipulation

Assigned ability: symbolic manipulation   
RESULT: 151, 157, 163, 169, 175, 181, 187, 193, 199

The symbolic-manipulation component is now covered, while enumeration remains. Agent 12, the nearest holder of enumeration, is isolated, so its ability set is empty for routing. The updated Bellman–Ford distances lead to Agent 82, the nearest normal holder of enumeration, three hops away.

Controller-only relays in a sparse network

Remaining ability: enumeration   
Agent 61 -> Agent 49 (routing after Contribute)   
Agent 49 -> Agent 151 (Handoff)   
Agent 151 -> Agent 82 (Handoff)

The intermediate agents relay the task state without asking an LLM to generate task content.

Agent 82 Finalize: Complete the enumeration component

Assigned ability: enumeration   
DESCRIPTION: Add the nine values provided by Agent 61:   
151 + 157 + 163 + 169 + 175 + 181 + 187 + 193 + 199 = 1575.   
RESULT: 1575

## What this case shows

Agent 61 contributes the symbolic-manipulation component. Local routing-state information then prevents the task from being assigned to an isolated agent. Bellman–Ford distance updates guide two controller-only relays to Agent 82, which holds the remaining enumeration ability and uses Finalize to complete the task. The route uses only local eligibility and distance information.

## E.2 REVIEW AND TAKEOVER

Case 3: Fluent but incorrect output, committee reassessment, and Takeover

Task: geometric shapes.   
Required abilities: mathematical + spatial.   
Degraded agent: A5.

Step 4: Takeover correction

Case 3: Fluent but incorrect output, committee reassessment, and Takeover (continued)

Outcome: success; gold answer (J), triangle.   
Question: SVG path ’M 30.17,45.97 L 58.79,40.36 L 18.10,15.70 M   
18.10,15.70 L 30.17,45.97’.

A5 remains responsive after degradation and produces a well-formed output. The failure is therefore not visible from availability or surface form alone.

Step 1: A5 produces a fluent but incorrect contribution   
RESULT: 6 vertices. The segments connect in the order A-B-C-D-E-F-A,   
forming a closed contour. This describes a hexagon.

The path contains only three distinct vertices, so the output contains a concrete correctness error. The output is escalated to committee review. Committee members first assess the same output independently.

Reviewer A: score 0.0   
Reason: "The path contains only three distinct points and three line segments,   
not six vertices forming a hexagon."   
Reviewer B: score 0.5   
Reason: "The shape classification is incorrect, but the vertex interpretation   
requires reassessment."

The first-round judgments disagree. Each reviewer then receives the other reviewer’s assessment and explanation before independently reassessing the original output.

Step 3: Committee reassessment   
First-round scores: [0.0, 0.5]   
Second-round scores: [0.0, 0.0]   
Final review score: 0.0   
Decision: Takeover

After reassessment, both reviewers identify the output as incorrect. The final low score remains attributed to A5 as evidence for routing-state updates, while Takeover repairs the current task immediately.

The path contains three distinct vertices:   
(30.17, 45.97), (58.79, 40.36), and (18.10, 15.70).   
They are connected by three straight-line segments forming a closed   
three-sided polygon: a triangle. No additional vertices or segments are present.

The corrected output is placed in the task state, and the downstream Finalize step returns (J), matching the gold answer.

## What this case shows

Degradation need not produce malformed or obviously unusable text: A5 returns a fluent and confident output containing a substantive error. Independent reviewers initially disagree, but committee reassessment resolves the disagreement. Takeover then replaces the incorrect output before it propagates to the final answer, while the review score remains available as evidence for subsequent routing-state updates.

## Case 4: A localized error inside an otherwise coherent solution

Task: tracking shuffled objects five objects.   
Required abilities: reasoning + sequence.   
Degraded agent: A5.   
Outcome: success; gold answer (C), brown.

In this example, the overall response is coherent, but one state transition is incorrect. A5 state is watched. Its output therefore receives mandatory committee review.

A5 Contribute: Reason through the state updates   
1. Claire and Alice swap:   
Claire (red), Alice (brown), Bob (green), Dave (black), Eve (orange)   
2. Dave and Eve swap:   
Claire (red), Alice (brown), Bob (green), Dave (orange), Eve (black)   
3. Claire and Eve swap:   
Claire (black), Alice (brown), Bob (green), Dave (orange), Eve (red)   
4. Bob and Claire swap:   
Claire (green), Alice (brown), Bob (black), Dave (orange), Eve (red)   
5. Alice and Claire swap:   
Claire (green), Alice (black), Bob (brown), Dave (orange), Eve (red)   
Therefore, Claire has the green ball.

The first four state transitions are correct. The fifth is not: immediately before the final swap, Claire holds green and Alice holds brown, so exchanging their balls should leave Claire with brown and Alice with green.

Both committee reviewers then identify the incorrect transition.

Committee review identifies the failing step   
Reviewer A: The final swap states that Claire remains green, but replaying the   
exchange gives Claire brown.   
Reviewer B: Step 5 is the first incorrect state update; the preceding four   
transitions are consistent.

The committee’s final score is at most τ , so Takeover replaces the output.

Takeover correction   
After the final Alice--Claire swap:   
Claire holds brown and Alice holds green.

The corrected output is placed in the task state, and the downstream Finalize step by an agent holding the sequence ability returns (C), matching the gold answer.

## What this case shows

The failure is subtle: four consecutive state updates are correct and only the final transition is wrong. The response otherwise has the structure and fluency of a valid solution. The review identifies the specific correctness error, and Takeover repairs the current task without restarting the full task trajectory. This complements Case 3, where committee reassessment resolves disagreement over a more visible semantic error.