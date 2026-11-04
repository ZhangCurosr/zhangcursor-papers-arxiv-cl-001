# ToolSpace: Scaling Tool-Centric Environment Synthesis for Long-Horizon Agents

Guanting Dong<sup>1,∗</sup>, Zhichao Sun<sup>2,∗</sup>, Xinyu Yang<sup>1,∗</sup>, Rui Wang<sup>2</sup>, Xin Deng<sup>2</sup>, Jiajie Jin<sup>1</sup>, Zhixiong Yang<sup>2</sup>, Junhao Huang<sup>3</sup>, Han Han<sup>2</sup>, Mingyu Ma<sup>2</sup>, Xiaotao Wang<sup>2</sup>, Yuyang Hu<sup>1</sup>, Xiaoshuai Song<sup>1</sup>, Xiaoxi Li<sup>1</sup>, Jiejun Tan<sup>1</sup>, Zhecan Wang<sup>2</sup>, Shihan Dou<sup>2</sup>, Honglin Guo<sup>2</sup>, Zhiheng Xi<sup>2</sup>, Tao Gui<sup>2,†</sup>, Tao Ji<sup>2</sup>, Qi Zhang<sup>2</sup>, Yutao Zhu<sup>1</sup>, Ji-Rong Wen<sup>1</sup>, Zhicheng Dou<sup>1,†</sup>

<sup>1</sup>Renmin University of China <sup>2</sup>Fudan University <sup>3</sup>Shanghai Jiao Tong University <sup>∗</sup>Equal contribution. <sup>†</sup>Corresponding authors.

## Abstract

Large language models are deployed as long-horizon agents that compose diverse tools across MCP services into executable workflows. Existing MCP environment-scaling methods are data-centric: they first synthesize a database and scenario, then derive tools and tasks from them. As a result, tool interfaces may not match real ones, tool compositions are not grounded in execution, and tasks rarely span more than one environment. We propose ToolSpace, a tool-centric framework that reverses this order. ToolSpace mines data, state, and retry dependencies from adjacent calls in execution traces of real MCP services and open agent corpora, using deterministic rules rather than LLM judgment, and clusters the resulting weighted tool-composition graph into environments. Databases are synthesized only for state-mutating environments, while query and stateless tools replay recorded returns. ToolSpace then performs state-aware random walks with rollback and decompiles each chain into a natural-language task with assertion-based checks, keeping only samples that replay successfully from the initial state. Applied to AutomationBench and �<sup>3</sup>-Bench Banking, the pipeline produces harder Pro versions on which 7 frontier models drop by up to 38.2 points. Across seven long-horizon tool-use benchmarks, ToolSpace raises Qwen3.6-35B-A3B from 23.0 to 41.0 average score (43.2 with RL) and GLM-5.2 from 44.6 to 56.6, on par with Claude Opus 5. Ablations show that execution-witnessed edges clearly outperform LLM-judged ones, and performance keeps improving with data and environment scale.

\# Contact: dongguanting@ruc.edu.cn, dou@ruc.edu.cn

<sup></sup> Project Page: https://toolspace-agent.github.io/

<sup>§</sup> Github: https://github.com/dongguanting/ToolSpace

Models: https://huggingface.co/collections/dongguanting/toolspace

![](images/fa0c39ce6891f2a78edcc4d806ac96f0075b1d9affe2df2d53b85ca5173471ca.jpg)

![](images/431a52ba80f919e3298a8fe334963a18b92b19bb50c1d8d2f1eb966c3380cc0e.jpg)

(c) Generalization across 7 Benchmarks  
![](images/4023fea7f9ed6a11618652b2ed238e9afa04088cea0859d9bfca68fcbedca7b2.jpg)

![](images/bddd39cca89c552d9cade176e81686c7f5cfcb279d98d5a8389bf2f6a2635e35.jpg)

(d) Performance on Pro Benchmarks  
![](images/3e8f813bab74ff500e6f7bf0fc5e5801e1c3fe4554ac55a58210b01901ec5a33.jpg)

![](images/f6a2fb4909fe668fe764310828fed10f6ab0a3772f13cba16b9fa0ec92610bfa.jpg)  
Figure 1: Overview of ToolSpace. (a) Design Philosophy: a tool space layer between agents and MCP environments. (b) The tool-composition graph with a state-aware walk. (c) Generalization across 7 benchmarks. (d) Performance on two Pro benchmarks.

## 1 Introduction

Recent advances have extended large language models (LLMs) from single-turn chatbots to collaborative agents that complete long-horizon tasks in dynamic environments (Patil et al., 2023; Qin et al., 2023; Schick et al., 2023; Yao et al., 2023). Such agents must not only master a scalable tool set, but also orchestrate diverse tools into executable long-horizon chains. Because tools are the atomic units of agent execution, how tools are organized and composed at scale has become a foundational question for expanding agent capability (Anthropic, 2024; Li et al., 2023; Song et al., 2023; Xu et al., 2025).

To scale tool use, recent work follows two main routes. One family, exemplified by Simia (Li et al., 2025) and GenEnv (Guo et al., 2025), treats the LLM as an implicit textual world model and generates simulated feedback from tool specifications and interaction histories. These methods are inexpensive, but their feedback is prone to hallucination and state consistency is dificult to guarantee. The second family explicitly reproduces interaction logic by synthesizing executable tools together with environment databases: it extracts or constructs an underlying database from a real service theme, generates matching tools, and synthesizes tasks with verifiable outcomes (Dong et al., 2026a; Fang et al., 2025; Song et al., 2026; Wang et al., 2026). Later systems further evolve whole interaction stacks or repair environments from execution feedback (Dong et al., 2026b; He et al., 2026; Mao et al., 2026).

Despite these advances, existing paradigms still follow Data-Centric Environment Synthesis: synthesize the underlying database and thematic scenario first, and only then derive tools and tasks. This ordering has three inherent limitations:

• Tool–functionality mismatch. Executable tools are synthesized from the environment database and theme description, so their parameter logic easily diverges from real tool interface definitions. Agents consequently learn a synthetic tool protocol rather than operations transferable to a real tool ecosystem.

• Inauthentic tool orchestration. Existing work builds tool-link graphs and synthesizes tasks through random linking or shallow LLM judgments, so the resulting links lack authenticity. Real tool services are moreover unstable and noisy, and their logical relations are hard to establish subjectively.

• Lack of cross-environment task synthesis. Current task synthesis relies on the data and rules of a single environment, whereas real-world tasks routinely span several environments. Because tools are locked inside their own sandboxes, cross-environment composition has no explicit reachability relation, and long-horizon complexity cannot grow naturally.

Together, these limitations share a single root cause: Data-Centric Environment Synthesis presumes a realistically synthesized environment and derives tool composition from it, reversing the dependency that governs real agent execution. How to synthesize agent training data that preserves real tool orchestration across diverse environments therefore remains an open challenge.

Our starting point is that, within agent execution, tools are the atomic units of capability growth, not static environment data. As illustrated in Figure 1(a)–(b), a Tool Space that has not yet been explicitly modeled always lies between an agent and its underlying environment databases, and the relations among tools naturally connect an extensible environment ecosystem. A natural path is therefore to model the real connectivity and state transfer among tools, and to build a crossenvironment tool-graph layer on top of these relations. The complexity of synthesized tasks then arises from expanding the real tool-composition space, rather than from the data scale of any single environment.

Following this insight, we propose ToolSpace, a tool-centric framework for cross-environment agent data synthesis. Unlike Data-Centric Environment Synthesis, ToolSpace does not mine a new environment ecosystem from scratch; it follows a Trace-to-ENV principle (Long et al., 2026; Wu et al.,

2026): it first reconstructs a scalable, explorable tool space from execution traces, then synthesizes long-horizon agent tasks across the tool and environment clusters connected in this space. ToolSpace consists of two stages:

• Tool Space Exploration. We extract tool calls from real-world execution traces and mine data, state, and retry links with deterministic rules, building a scalable tool-composition graph. Tightly coupled tools are clustered into environment clusters; an underlying database is synthesized on demand only for operation environments, while original tool returns are reused directly otherwise. This preserves tool-composition relationships while substantially reducing environment synthesis cost.

• Cross-Environment Verifiable Task Synthesis. Over the ToolSpace graph, State-Aware Random Walks with Rollback Sampling sample state-compatible successors by edge weight and execute them step by step, retaining only chains that execute end to end under runtime-observed state transitions. Each retained walk is then decompiled into a natural-language task with assertion-based verification scripts, yielding training samples that can be replayed and audited across environments.

To validate ToolSpace’s ability to flexibly synthesize complex cross-environment tasks, we anchor the full pipeline to AutomationBench (Shepard and Salimans, 2026) and $\tau ^ { 3 }$ -Bench Banking (Shi et al., 2026), two long-horizon agent benchmarks and extend their dificulty, producing AutomationBench Pro and �<sup>3</sup>-Bench Banking Pro. The upgraded versions exceed the standard ones in task dificulty and number of environments crossed. On AutomationBench Pro, seven frontier models drop by up to 38.2 points relative to the standard version (Figure 1).

In summary, our main contributions are as follows:

• We identify limits of data-centric synthesis: reverse-synthesized tools mismatch real tool semantics, tool graphs lack execution evidence, and tasks are confined to single-environment complexity; and we then propose Tool-Centric Environment Synthesis.

• We propose ToolSpace, which reconstructs a scalable, explorable tool-composition space from real execution traces and synthesizes long-horizon, verifiable agent tasks across diverse tool and environment combinations within this space.

• We show that ToolSpace plugs into long-horizon tool-use benchmarks, producing AutomationBench Pro and �<sup>3</sup>-Bench Banking Pro, which exceed their standard versions in dificulty, tool-call turns, and cross-environment breadth.

• Across seven long-horizon tool-use benchmarks, experiments demonstrate clear advantages across model scales (8B–744B). ToolSpace-744B-A40B surpasses GPT 5.6 Sol in average score (56.6 vs. 48.9) and matches Claude Opus 5. Further analyses reveal clear scaling relationships of data scale, environment scale, and task diversity with agent performance.

## 2 Related Work

Agent Environment Scaling. Agent Environment Scaling has become an important foundation for the evolution of agent capabilities, by scaling the environments in which agents act. Simulator-style methods, exemplified by Simia (Li et al., 2025) and GenEnv (Guo et al., 2025), use an LLM as a textual world model to emulate tool feedback: cheap to scale, but prone to hallucination and state inconsistency. Executable-environment methods instead synthesize databases, tools, and verifiable tasks programmatically (Dong et al., 2026a; Fang et al., 2025; Song et al., 2026; Wang et al., 2026), with later variants that evolve environments or whole interaction systems (He et al., 2026; Mao et al., 2026) or repair them from execution feedback (Dong et al., 2026b). In all of these, the unit of scaling remains the environment (or the surrounding harness), so which tools may compose is fixed before any task is drawn. ToolSpace is complementary: it recovers tool-composition structure witnessed by executions and synthesizes databases only on demand, so the same machinery can upgrade benchmarks.

Graph-Based Agentic Data Synthesis. Building agent data with intrinsic dependencies on structured graphs is a promising direction. For search and knowledge agents, prior work builds knowledge graphs or formalizes information seeking tasks to generate hard questions and trajectories (Tao et al., 2025; Wu et al., 2025). For general tool use, function-signature or API-dependency graphs are walked to produce multi-turn calls and verified blueprints (Dong et al., 2026a; Prabhakar et al., 2025; Yin et al., 2025); similar recipes reverse-synthesize trajectories for multimodal GUI and web agents (Zhang et al., 2025). These chains then feed SFT, agentic ${ \mathrm { R L } } ,$ and reflection-style training (Dong et al., 2026b; Shao et al., 2024; Shinn et al., 2023; Yao et al., 2023). ToolSpace difers in where the graph comes from and what it guarantees: edges are mined by deterministic rules from adjacent real calls (Qin et al., 2023; Wang et al., 2025; Xu et al., 2025; Yang et al., 2024) across environments, typed as data, state, or retry and weighted by witnessed frequency; a walk enters training only after replay from $s _ { 0 }$ passes state, trace, and output checks. ToolSpace is the framework to make witnessed tool composition the single primitive behind cross-environment tasks and benchmarks.

## 3 Preliminaries and Problem Formulation

We formalize tool-use tasks, data-centric environment synthesis, and the tool-centric synthesis problem addressed by ToolSpace.

Tool-use tasks. Let $\mathcal { F }$ denote a tool space, where each tool $f \in { \mathcal { F } }$ has input and output schemas $\alpha _ { f }$ and $\beta _ { f }$ . An environment $e = \left( \mathcal { D } , \mathcal { F } _ { e } \right)$ pairs a persistent database D with a tool subset $\mathcal { F } _ { e } \subseteq \mathcal { F }$ and induces a state space $S _ { e }$ with snapshots $s _ { t }$ . Given a natural-language task $q ,$ an agent policy �<sub>�</sub> produces

$$
\tau = ( a _ { 1 } , o _ { 1 } , \dots , a _ { L } , o _ { L } ) , \quad \quad \quad a _ { t } = \pi _ { \theta } ( q , h _ { t } ) , \quad o _ { t } = f _ { t } ( x _ { t } ; s _ { t - 1 } ) ,\tag{1}
$$

where $h _ { t }$ is the interaction history, $f _ { t }$ is the tool invoked with arguments $x _ { t } , o _ { t }$ is its observation, � is the number of tool calls, and $s _ { t }$ is the environment state after step �. A verifier $V$ maps $( q , \tau , s _ { L } )$ to $\{ 0 , 1 \}$ , or to a graded assertion score.

Data-centric environment synthesis. Prior work samples a theme �, first synthesizes an underlying database $\mathscr { D } \sim p ( \mathscr { D } \mid m )$ , then synthesizes derived tools $\mathcal { F } _ { e } \sim p ( \mathcal { F } _ { e } \mid \mathcal { D } , m )$ conditioned jointly on the theme and the database, and further synthesizes tasks $q \sim p ( q \mid \mathcal { D } , \mathcal { F } _ { e } )$ . Because $\mathcal { F } _ { e }$ is a synthetic tool set conditioned on a fabricated D, its argument logic need not coincide with any real interface, and the resulting task synthesis is likewise detached from real tool-logic relations; cross-environment reachability is undefined, as tools from diferent (D, F<sub>�</sub>) pairs share no typed relation.

Tool-centric environment synthesis (our problem). Given real execution traces $\boldsymbol { \mathcal { M } } = \{ ( f _ { t } , x _ { t } , o _ { t } ) \}$ , we seek: (i) a typed tool graph $\mathcal { G } _ { \mathrm { T S } } = ( \mathcal { V } , \mathcal { E } , w )$ whose tool nodes and inter-tool connections are extracted from and filtered against M; (ii) environment clusters aggregated by tool-edge density together with real tool definitions; (iii) underlying data synthesized on demand for operation environments; and (iv) a dataset $X _ { \mathrm { T S } } = \{ ( q , V , \omega , s _ { 0 } ) \}$ . Each tool chain $\omega = ( \nu _ { 1 } , \ldots , \nu _ { L } )$ must be a walk on $\mathcal { G } _ { \mathrm { T S } }$ satisfying dependency reachability, state preconditions with respect to $s _ { t } ,$ and parameter provenance: every argument must be traceable to $q , s _ { 0 } .$ , or an earlier observation $O _ { < t }$ . It must also be replay-verified, such that re-executing � from $s _ { 0 }$ passes all assertions in �. Our objective is to maximize the longhorizon, cross-environment coverage of $\chi _ { \mathrm { T S } }$ subject to replay verification, rather than the fidelity of any single synthesized database. Under this tool-centric formulation, tools conform to real toolinterface definitions, the execution logic of synthesized tasks follows real tool-connection relations, and databases are synthesized only on demand, making synthesis eficient and cost-saving.

## 4 Method

Overview. We present ToolSpace, a tool-centric framework for cross-environment agent data synthesis (Figure 2). ToolSpace comprises two tightly coupled components, detailed below, and seamlessly raises the dificulty of existing environment benchmarks (§5):

• Tool Space Exploration (§4.1) extracts tool calls from real traces, mines data/state/retry dependencies by rules, builds a tool-composition graph, clusters tools into environments, and synthesizes a database only for operation-bearing clusters.

• Cross-Environment Task Synthesis (§4.2) explores executable long-horizon chains by State-Aware Random Walks and decompiles them into tasks with verification scripts, retaining only samples that replay from $s _ { 0 }$ and pass all checks.

## 4.1 Tool Space Exploration

## 4.1.1 Agent Trajectory Collection

To obtain diverse, large-scale real execution evidence, we build a unified agent-trajectory corpus M from two source families.

MCP-ecosystem trajectories. Existing MCP environment scaling work (Dong et al., 2026a; Xu et al., 2025) typically synthesizes task queries over real MCP servers from registries such as Smithery. Because their tool definitions come from real servers, such tasks can be seamlessly re-executed against the live services. We collect a real tool catalog from the registry, keep only tasks whose tool definitions exactly match the real servers, execute them with diverse agents on real MCP services, and record the complete call chains as M<sub>MCP</sub>.

Web-sourced agent trajectories. We additionally use a deep-search agent to collect general tool-use trajectories from the web. We group them by execution provenance, rank them from high to low by the degree of execution authenticity, and apply progressively stricter filtering to lower-authenticity data:

• Real-server interaction trajectories (High). We retain records executed on real services. Toucan-1.5M (Xu et al., 2025) covers 495 MCP servers, 2,000+ tools, and 1.5M trajectories, with synthetic instructions but real executions. ToolBench/ToolLLM (Qin et al., 2023) covers 16,464 RapidAPI interfaces and ∼469K API calls; we remove unavailable interfaces.

• Real-workspace interaction trajectories (Medium). We collect trajectories executed by SWEagent (Yang et al., 2024) and OpenHands (Wang et al., 2025) in real containers and repositories. We retain only explicit tool calls and exclude shell wrappers such as execute\_bash.

• Synthetic-tool trajectories (Low). We collect trajectories generated in synthetic environments (Dong et al., 2026a,b; Song et al., 2026). Because of potential tool inconsistency, we retain only calls whose tool names and schema formats match corresponding real tools.

The three groups merge into $M _ { \mathrm { o p e n } } ;$ all tool calls form

$$
M = M _ { \mathrm { M C P } } \cup M _ { \mathrm { o p e n } } = \{ ( f _ { t } , x _ { t } , o _ { t } ) \} ,\tag{2}
$$

where $\left( f _ { t } , x _ { t } , o _ { t } \right)$ is the �-th call of a trajectory: $f _ { t }$ the tool, $x _ { t }$ its inputs, and $o _ { t }$ its result. Calls within a trajectory keep their order.

## 4.1.2 Tool-Space Graph Modeling

To explicitly model these implicit tool relations, we organize M into a typed tool-composition graph $\mathcal { G } _ { \mathrm { T S } } = ( \mathcal { V } , \mathcal { E } , w )$ that can chain diferent services into one tool world (Figure 2).

![](images/6e1f4df44386fe826951d88a1c68b67ad34c8b6e2f0dda2763dc8ec3ef573a5d.jpg)  
Figure 2: Overview of the ToolSpace framework. Stage 1 builds the Tool Space from real traces; Stage 2 synthesizes crossenvironment tasks by State-Aware Random Walks, retaining only samples that pass replay verification.

Nodes. For every tool call � occurring in M, we create a node $\nu _ { f , \tilde { x } } = ( n _ { f } , \alpha _ { f } , \beta _ { f } , \tilde { x } )$ , where $n _ { f }$ is the tool name, $\alpha _ { f }$ and $\beta _ { f }$ are its input and output schemas, and $\tilde { x }$ is its normalized argument configuration. Thus, calls to the same canonical tool with diferent argument configurations form distinct nodes, as they may produce diferent outputs and downstream dependencies. We write $\upsilon _ { t }$ for the node $\upsilon _ { f _ { t } , \tilde { x } _ { t } }$ instantiated by the �-th call, and $\nu _ { i } / \nu _ { j }$ for generic nodes below.

Relations. For adjacent calls $\left( f _ { t } , x _ { t } , o _ { t } \right)$ and $( f _ { t + 1 } , x _ { t + 1 } , o _ { t + 1 } )$ in a trajectory, we extract three edge types by deterministic rules, no LLM judgment is used:

• Data dependency $( \nu _ { t } \xrightarrow { \mathrm { d a t a } } \nu _ { t + 1 } )$ : the input $x _ { t + 1 }$ uses a value from $o _ { t }$ and field types match. Concrete data is passed.

• State dependency $( \nu _ { t } \xrightarrow { \mathrm { s t a t e } } \nu _ { t + 1 } ) \mathrm { : }$ : the first call mutates an environment object and the second call reads or further mutates the same object, not merely a coincidentally equal value.

• Retry dependency $( \nu _ { t } \xrightarrow { \mathrm { r e t r y } } \nu _ { t + 1 } )$ : the first call returns an error, the trajectory switches tools, and the next call succeeds. Such errors are treated as recoverable observations and retained together with the successful retry in the trajectory.

Hence, the edge set is defined as $\mathcal { E } = \mathcal { E } _ { \mathrm { d a t a } } \cup \mathcal { E } _ { \mathrm { s t a t e } } \cup \mathcal { E } _ { \mathrm { r e t r y } } ,$

Weights. To retain recurring tool dependencies rather than tools that merely appear consecutively, we weight each candidate edge by its observed transition frequency. Let $N _ { i }$ denote the number of occurrences of node $\nu _ { i }$ in M and $N _ { i j }$ the number of times $\nu _ { j }$ immediately follows $\nu _ { i }$ . We define

$$
w ( \nu _ { i } , \nu _ { j } ) = \frac { N _ { i j } } { N _ { i } } .
$$

We remove low-frequency edges. For candidate data edges, we additionally require the downstream arguments to be traceable to the upstream output, which prevents false links caused by common constants such as limit=10. Consequently, $\mathit { G } _ { \mathrm { T S } }$ captures tool transitions during execution rather than merely listing available tools.

## 4.1.3 Agent Environment Data Synthesis

ToolSpace materializes underlying data only when an environment depends on persistent state, avoiding the need to synthesize a complete database in advance. We cluster the weighted undirected projection of $\mathcal { G } _ { \mathrm { T S } }$ by connection density and organize the clusters using the environment and tool configurations of real MCP services. This produces $C = \{ c _ { 1 } , \ldots , c _ { K } \}$ with $K = 8 0 4$ , where each cluster defines an environment $\boldsymbol { e } _ { k } = \left( \mathcal { D } _ { k } , \mathcal { F } _ { k } \right)$ with tool set $\mathcal { F } _ { k }$ and database $\mathcal { D } _ { k }$ . We consider three cases:

• Operation environments. These environments contain tools that create, modify, or delete objects and therefore require an underlying database. Following Agent-World (Dong et al., 2026a), we derive a theme $m _ { k }$ from each cluster and use a research agent $\mathcal { G } _ { \mathrm { r e s } } ,$ built on policy model $\pi _ { \theta }$ and tools $\mathcal { T } _ { \mathrm { r e s : } }$ , to collect and align web data: $\mathcal { D } _ { k } ^ { ( 0 ) } = \mathcal { G } _ { \mathrm { r e s } } ( m _ { k } ; \pi _ { \theta } , \mathcal { T } _ { \mathrm { r e s } } )$ . We then iteratively expand it as $\mathcal { D } _ { k } ^ { ( n + 1 ) } = \phi ( \mathcal { D } _ { k } ^ { ( n ) } , m _ { k } , \mathcal { T } _ { \mathrm { r e s } } )$ for $n = 0 , \ldots , N - 1$ . The environment is retained after unit tests and state-transition checks.

• State-query environments. Many state-query tools return partial snapshots of the environment, such as google\_drive.files.list and gmail.users.messages.get. We use these outputs as anchors to initialize the relevant environment state rather than synthesizing a database from scratch. The resulting state serves as the initial snapshot $s _ { 0 }$

• Stateless environments. These environments contain tools that require no environment-specific persistent state, such as web search and fixed-index lookup. For matching replay calls, we use cached tool results from M as execution outputs, without issuing live API calls or synthesizing a database.

The first two types enter the executable environment system $\mathcal { E } _ { \mathrm { T S } } = \{ e _ { k } \}$ , while stateless environments use cached tool results as execution outputs. This design reduces environment-synthesis and API costs and accelerates subsequent trajectory synthesis.

Algorithm 1: ToolSpace   
Input: trace corpus ${ \mathcal { M } } ,$ target length $L ,$ target environments $K ,$ walk budget   
Output: verified dataset $\chi _ { \mathrm { T S } }$   
1 Build $\mathcal { G } _ { \mathrm { T S } } = ( \mathcal { V } , \mathcal { E } , w )$ from adjacent calls in M (data/state/retry rules); prune untraceable or   
low-weight edges   
2 Cluster $\mathcal { G } _ { \mathrm { T S } }$ into environments $\{ e _ { k } = ( \mathcal { D } _ { k } , \mathcal { F } _ { k } ) \}$ ; synthesize $\mathcal { D } _ { k }$ on demand (operation), restore   
$s _ { 0 }$ (state-query), or cache returns (stateless)   
3 foreach walk attempt do   
4 Start at a node bound to $( \mathcal { D } _ { k } , s _ { 0 } ) ;$ ; repeatedly sample an admissible neighbor by Eq. (3),   
execute $\mathbf { i t } ,$ and update $s _ { t }$   
5 if execution fails or violates the traversed edge relation then restore $s _ { t - 1 }$ and resample   
(rollback)   
6 if length � and � environments are reached then record � with $s _ { 0 } , \ldots , s _ { L }$   
7 end   
8 foreach $\omega \in \Omega$ do   
9 Decompile � into task $q$ (no tool names/order) and assertions $V = V _ { \mathrm { s t a t e } } \cup V _ { \mathrm { t r a c e } } \cup V _ { \mathrm { o u t } }$   
10 Replay � from $s _ { 0 } ;$ if all assertions pass then add $( q , V , \omega , s _ { 0 } )$ to $\chi _ { \mathrm { T S } }$   
11 end   
12 Deduplicate $\chi _ { \mathrm { T S } } ;$ generate teacher trajectories for retained � and keep those passing �

## 4.2 Cross-Environment Task Synthesis

![](images/005fa5259b386add0d5b88f5ce0ef90dfe186948a27c9679ad3efab75e18e153.jpg)  
Figure 3: The Interactive demo of state-aware random walking. An example walk from the interactive demo: 28 tool calls across 6 environments, completed after 2 rollbacks and kept after replay verification. Candidate successors and their sampling probabilities $P ( \nu ^ { \prime } \mid \nu )$ are shown on the right. Demo: https://toolspace-agent.github.io/#demo

## 4.2.1 State-Aware Random Walks

To collect long-horizon chains that preserve real logical connections, we perform State-Aware Random Walks on $\mathit { G } _ { \mathrm { T S } }$ (Stage 2 of Figure 2). Given a number of crossed environments � and chain length $L ,$ a walk starts at a tool node bound to a database $\mathcal { D } _ { k }$ and a snapshot �<sub>0</sub>; $s _ { t }$ denotes the � step environment-state snapshot during execution.

Graph Walk. Let $\upsilon _ { t }$ be the current node and consider its neighbors in E. A neighbor is kept only if its next-step inputs can be filled from earlier returns or the current snapshot $s _ { t }$ (dependency reachability, state preconditions, parameter provenance). Among the kept neighbors we sample proportionally to edge weight,

$$
P ( \nu _ { t + 1 } = \nu \mid \nu _ { t } ) = \frac { w ( \nu _ { t } , \nu ) } { \sum _ { u } w ( \nu _ { t } , u ) } ,\tag{3}
$$

where the sum ranges only over admissible neighbors. Connections that occur more frequently in real traces are therefore more likely to be extended. The walk continues until its length reaches � and its tools cover � environments, yielding a chain $\omega = ( \nu _ { 1 } , \ldots , \nu _ { L } )$

Rollback Sampling. If a selected tool fails, or the state after execution is inconsistent with the data/state relation of the traversed edge, the step is discarded, the environment is restored to $s _ { t - 1 }$ , and another neighbor of $\nu _ { t - 1 }$ is resampled. Chains that still cannot proceed after repeated rollbacks are discarded. The retained walk set is $\Omega = \{ \omega \}$ , each walk carrying $s _ { 0 }$ and snapshots $s _ { 1 } , \ldots , s _ { L }$ . Finally, each retained chain is replayed from $s _ { 0 }$ (Replay & Acceptance in Figure 2): re-executing its tools against the recorded snapshots guarantees state continuity along the chain and that every tool remains executable. Figure 3 shows one completed walk produced by this procedure in the interactive demo on our project page (https://toolspace-agent.github.io/#demo).

## 4.2.2 Reverse Verifiable Task Synthesis

To turn executable chains into auditable tasks, we reverse-synthesize each $\omega \in \Omega$ . A language model writes a natural-language task $q$ from the chain and its state changes; � contains no tool names and no call order. Instead of relying on a reference answer, we generate a verification set for each chain,

$$
V = V _ { \mathrm { s t a t e } } \cup V _ { \mathrm { t r a c e } } \cup V _ { \mathrm { o u t } } ,\tag{4}
$$

![](images/febf3a53eaf28fa41f1104ae013d886cf78e0404224d0edae8f8b34600be12b4.jpg)  
(a) Environment domain diversity

![](images/777b43144fd9dbe7e8e5b5deffe2eedd07b16a43dc3fecea5bb0664c2dde4b6e.jpg)  
(b) MCP domain distribution

![](images/5d56a0df08678259a39764e0815dc83c705326916998083eefaf85b523c81bce.jpg)  
(c) Tools per environment

![](images/b1bfe0bec5a470106b91a7ee968022d2ed08134ca4c9278d14c91c3969c5ee55.jpg)  
(d) Distribution of tool dependency relations

![](images/05cfe7018dc76c346e8e184abd81f977d2d50c83ac700e691e82cc06de3d9514.jpg)  
(e) Tool state-interaction diversity

![](images/aee2fdee06d71cdd5518ce3dfeee1ccf9f206d2abf8f2d7ce339dcd42ea7d8bb.jpg)  
(f) Environment state-interaction diversity

Figure 4: ToolSpace graph statistics. (a) Environment-name word cloud. (b) Environment distribution across MCP domains. (c) Number of tools per environment. (d) Proportions of data, state, and retry dependencies. (e) Tool distribution by operational category. (f) Environment distribution by operational category.  
![](images/e567c004b76a5785298a193d8e2aace94b6ab1ad76fd22f13c121be287dfe105.jpg)  
(a) Trajectory token length

![](images/ab90a7adbc3021daa07a1e6055d188b5a6e16bc2ca90377d72776280d0f50813.jpg)  
(b) Tool calls per trajectory

![](images/9b5ef5ff36178d25851bda7cb3bbf586c82be9e5775193d2d85f6e0e1bc4b01e.jpg)  
(c) Observed service breadth  
Figure 5: Synthesized task and trajectory statistics. Distributions across 21,385 trajectories with at least five tool calls: (a) trajectory token length, (b) tool calls per trajectory, and (c) environments per trajectory.

where $V _ { \mathrm { s t a t e } }$ checks the final state $s _ { L } , V _ { \mathrm { t r a c e } }$ checks key intermediate snapshots, and $V _ { \mathrm { o u t } }$ checks fields required to appear in returned results (here � denotes verification, distinct from the node set $\mathcal { V } )$ A sample is retained only if re-executing � from �<sub>0</sub> passes all checks in �. Dificulty is raised via �, target �, and the share of cross-environment edges. The dataset is $X _ { \mathrm { T S } } = \{ ( q , V , \omega , s _ { 0 } ) \}$ : task, verification script, tool chain, and snapshot. We present the complete ToolSpace algorithm flowchart in Algorithm 1.

## 4.3 Detailed Statistics of ToolSpace

To more comprehensively showcase ToolSpace, Figures 4 and 5 characterize the static statistics of ToolSpace’s graph, environments, and synthesized tasks.

(A) Graph Scale and Dependency Structure. Reconstructed from collected tool-call trajectories, $\mathcal { G } _ { \mathrm { T S } }$ contains 77,760 tool-call-pattern nodes derived from 2,042 distinct canonical tools. Its mean and maximum outdegrees (outgoing dependency edges per node) are 2.3 and 56, respectively. Panel (a) presents a word cloud of the environment types covered by ToolSpace synthesis, including Gmail, Google Sheets, Slack, Google Drive, Salesforce, and Airtable. Panel (d) shows that data, state, and retry edges account for 69%, 25%, and 6%, respectively. The dominance of data edges indicates that tool composition is driven primarily by intermediate-result transfer, while state and retry edges capture state prerequisites and failure recovery. Every edge is witnessed by adjacent calls in real trajectories, providing execution-grounded tool reachability.

(B) Environment Diversity and Selective Grounding. The reference inventory covers 804 real MCP environments. Figure 4(b) identifies Communication & General Utilities, Finance & Markets, and Document & Design as the largest domains, accounting for 27.4%, 11.2%, and 10.1% of environments, respectively. Panel (c) reports an average of 10.86 matched tools per environment and a maximum of 436; tools such as web search, file reading, and email sending are available across multiple environments. Panel (e) distinguishes 453 operational tools (22.2%) from 1,589 non-operational tools (77.8%), with structured record mutation being the largest operational category at 95 tools. Among the 804 environments with assigned tools, Panel (f) identifies 181 operational (22.5%) and 623 non-operational environments (77.5%), with document persistence being the largest operational category at 42 environments. An environment is operational if it contains at least one tool capable of modifying its state. These statistics further show that stateless environments dominate the real world environments synthesized by ToolSpace. ToolSpace materializes databases on demand only for operational workspaces, restores snapshots for state-query tools, and reuses cached returns for stateless environments and tools, thereby reducing database-synthesis requirements and costs relative to data-centric methods.

(C) Task Scale and Complexity. In total, ToolSpace synthesizes 34,376 trajectories. Figure 5 summarizes the 21,385 trajectories with at least five tool calls. Panel (a) reports mean and maximum lengths of 30,466 and 855,386 tokens, while Panel (b) reports an average of 18.86 tool calls per trajectory and a maximum of 209, demonstrating that ToolSpace can synthesize long-horizon agent execution trajectories. Panel (c) shows an average of 2.29 environments per trajectory and a maximum of 22, with at least 66.2% of trajectories involving two or more environments. During synthesis, tool chains are sampled through state-aware random walks, executed step by step, and retained only after replay verification. The resulting tasks therefore require both long-horizon tool orchestration and consistent environment state.

## 5 AutomationBench Pro and $\tau ^ { 3 }$ -Bench Banking Pro

To test whether ToolSpace can raise the cross-environment and long-horizon dificulty of frontier benchmarks, we anchor the pipeline to AutomationBench (Shepard and Salimans, 2026) and $\tau ^ { 3 } .$ Bench Banking (Shi et al., 2026), showing the pipeline’s flexible dificulty expansion.

![](images/abe35cb771f2a9e623e4b479472492bf8a1c11e5685f260ecbe6285299f2a836.jpg)

Benchmark Construction. For a benchmark � with original environment $e _ { b } ^ { \mathrm { o r i g } } = ( \mathcal { D } _ { b } ^ { \mathrm { o r i g } } , \mathcal { F } _ { b } ^ { \mathrm { o r i g } } )$ , we collect its real tool calls $M _ { b }$ , build a tool graph $\mathcal { G } _ { b }$ by the rules of §4.1.2, and thereby link in tools outside the original benchmark that share data or state relations with benchmark tools, expanding a single environment into a cross-environment cluster. We then restore or synthesize the necessary databases and snapshots (§4.1), walk with larger � and � than the original benchmark, and reverse-generate tasks and verification scripts. After replay and deduplication we obtain AutomationBench Pro (300 tasks)

![](images/0db5513d05457c1874b29841d15804812caee937af12eaa3ea22b09fe277a0a9.jpg)  
Figure 6: Average tool calls of frontier models on (a) AutomationBench and (b) $\tau ^ { 3 } .$ -Bench Banking (Standard vs. Pro).

and $\tau ^ { 3 }$ -Bench Banking Pro (98 tasks). To ensure that Pro benchmarks reflect genuine dificulty rather than task quality, three doctoral-level experts independently reviewed all candidate tasks and their verification scripts over a two-day period. Only tasks approved by at least two experts were retained. Inter-rater agreement was strong, with Fleiss’ $\kappa = 0 . 8 2$ and an observed agreement of 93.1%.

Data Statistics. Figure 6 shows that Pro tasks require substantially more tool calls than the standard splits on both benchmarks. On AutomationBench, average tool calls rise sharply for every model, from 26.8–43.6 under Standard to 56.3–60.6 under Pro. On $\tau ^ { 3 }$ -Bench Banking, average calls likewise rise for every model, from 23.1–32.2 to 27.3–36.3, fully confirming that the added dificulty arises from longer and more frequent tool orchestration.

## 6 Experiments

## 6.1 Experimental Setup

Benchmarks. We evaluate on seven long-horizon tool-use benchmarks, referred to by their full names throughout: AutomationBench (Shepard and Salimans, 2026) (workflow automation), Skills-Bench (Li et al., 2026a) (skill composition), JobBench (Li et al., 2026b) (job-style professional tasks), $\tau ^ { 3 }$ -Bench Banking (Shi et al., 2026) (banking tool–dialogue interaction), the Berkeley Function Calling Leaderboard V4 (BFCL V4) (Patil et al., 2025) (function calling), Harvey LAB (Harvey AI, 2026) (legal agent workflows), and OficeQA Pro (Opsahl-Ong et al., 2026) (ofice document question answering).

Baselines. We evaluate four model groups: (i) proprietary frontier models: Claude Opus 5 (Anthropic, 2026b) and GPT 5.6 Sol (OpenAI, 2026a); (ii) open-weight models above 200B parameters: GLM 5.3 (Zhipu AI, 2026b), GLM 5.2 (Zhipu AI, 2026a), DeepSeek-V4-Pro-0813 (DeepSeek-AI et al., 2026), Kimi K3 (Moonshot AI, 2026), and Qwen 3.8 Max (Qwen Team, 2026c); (iii) 7–8B environment-scaling models: EnvScaler-8B (Song et al., 2026), AWM-8B (Wang et al., 2026), Simulator-8B (Li et al., 2025), Agent-World-8B (Dong et al., 2026a), Agent-Reflex-8B (Dong et al., 2026b), and their Qwen3-8B backbone (Yang et al., 2025); and (iv) 27–35B open-weight models: Qwen3.6- 27B (Qwen Team, 2026a), Qwen3.6-35B-A3B (Qwen Team, 2026b), Nex-N2 Mini (Nex-AGI, 2026), Muse-Glimmer-30B (Meta, 2026), and WorkRover-35B-A3B (He et al., 2026). For the Pro benchmark, we further evaluate Claude Opus 5.5 (Anthropic, 2026a), GPT 6.1 Sol (OpenAI, 2026b), and DeepSeek-V4.1-Flash (DeepSeek-AI, 2026).

Implementation Details. We train long-horizon agents on three backbones, Qwen3-8B, Qwen3.6- 35B-A3B, and GLM-5.2-744B-A40B (the resulting 744B model is denoted ToolSpace-744B-A40B). Detailed ToolSpace statistics are reported in §4.3.

Trajectory collection and environment synthesis. We collect over 200,000 agent–tool interaction trajectories as raw material for ToolSpace construction, with strict quality control during collection. We perform decontamination at both the task level and the trajectory level: any collected trajectory whose task, tool-call sequence, or initial state overlaps with an evaluation test set is excluded, ensuring that the synthesized data are free of test-set contamination. All data sources and MCP services used are covered by open-source licenses. We mine executable environments following the agentic exploration procedure and settings of Agent-World Dong et al. (2026a).

Task and trajectory synthesis. We use Kimi K3 (K3) for both teacher-trajectory generation and reverse verifiable task synthesis. All synthesized samples are validated against � using verification scripts, yielding 34,376 SFT samples and 653 RL samples from the same ToolSpace synthesis pipeline, except that RL candidates undergo additional Pass@16 screening with the 35B model, retaining only candidates with one or two correct rollouts to ensure informative RL training.

Training and evaluation. All data-synthesis baselines use the same SFT configuration: the 8B and 35B models are trained for one epoch on 16 NVIDIA GPUs, while the 744B model is trained for one epoch on 256 GPUs. For agent RL, we adopt the GRPO objective Shao et al. (2024),

$$
J _ { \mathrm { G R P O } } ( \theta ) = \mathbb { E } _ { x \sim \mathcal { D } , \{ y _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta \mathrm { d d } } ( \cdot | x ) } \left[ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { | y _ { i } | } \sum _ { t = 1 } ^ { | y _ { i } | } \operatorname* { m i n } \left( r _ { i , t } ( \theta ) \hat { A } _ { i } , \ \mathrm { c l i p } ( r _ { i , t } ( \theta ) , c _ { \mathrm { l o w } } , c _ { \mathrm { h i g h } } ) \hat { A } _ { i } \right) \right] ,\tag{5}
$$

Table 1: Performance on seven tool-use benchmarks. Bold/underline mark the best/second-best score per column within each same-scale pool (the 8B, 27–35B, and 744B pools, each including the ToolSpace model(s) at that scale). “–” indicates results not oficially reported.
<table><tr><td>Model</td><td>Size</td><td>Automation Bench</td><td>Skills Bench</td><td>Job Bench</td><td> $\tau ^ { 3 }$  -Bench Banking</td><td>BFCL V4</td><td>Harvey LAB</td><td>OfficeQA Pro</td><td>Avg.</td></tr><tr><td colspan="10">Proprietary Models</td></tr><tr><td>Claude Opus 5</td><td></td><td>45.6</td><td>63.7</td><td>68.0</td><td>48.7</td><td>80.6</td><td>22.7</td><td>67.2</td><td>56.6</td></tr><tr><td>GPT 5.6 Sol</td><td></td><td>47.1</td><td>62.5</td><td>45.4</td><td>46.9</td><td>68.8</td><td>5.0</td><td>66.4</td><td>48.9</td></tr><tr><td colspan="10">Open-Weight Models (&gt;200B total parameters)</td></tr><tr><td>Ż GLM 5.3</td><td>744B</td><td>52.5</td><td>63.3</td><td>58.2</td><td>40.2</td><td>74.1</td><td>22.5</td><td>65.8</td><td>53.8</td></tr><tr><td>Z GLM 5.2</td><td>744B</td><td>29.3</td><td>56.7</td><td>43.4</td><td>37.1</td><td>75.6</td><td>13.3</td><td>56.9</td><td>44.6</td></tr><tr><td>DeepSeek-V4-Pro-0813</td><td>1.7T</td><td>41.7</td><td>65.0</td><td>54.1</td><td>44.3</td><td>71.4</td><td>6.7</td><td>66.2</td><td>49.9</td></tr><tr><td>Kimi K3 K</td><td>2.8T</td><td>44.8</td><td>51.9</td><td>54.3</td><td>37.1</td><td>69.1</td><td>25.4</td><td>65.4</td><td>49.7</td></tr><tr><td>Qwen 3.8 Max</td><td>2.4T</td><td>49.7</td><td>66.7</td><td>52.7</td><td>55.2</td><td>66.8</td><td>24.2</td><td>66.2</td><td>54.5</td></tr><tr><td colspan="10">Environment Scaling Baselines (7–8B)</td></tr><tr><td>六 Qwen3-8B</td><td>8B</td><td>0.0</td><td>0.1</td><td>0.1</td><td>4.1</td><td>42.8</td><td>0.0</td><td>3.0</td><td>7.2</td></tr><tr><td>EnvScaler-8B</td><td>8B</td><td>0.0</td><td>1.6</td><td>1.1</td><td>7.2</td><td>47.5</td><td>0.0</td><td>1.5</td><td>8.4</td></tr><tr><td>{}AWM-8B</td><td>8B</td><td>0.0</td><td>1.4</td><td>0.5</td><td>5.2</td><td>42.0</td><td>0.0</td><td>5.3</td><td>7.8</td></tr><tr><td>Simulator-8B</td><td>8B</td><td>0.0</td><td>0.4</td><td>0.1</td><td>1.0</td><td>20.5</td><td>0.0</td><td>6.0</td><td>4.0</td></tr><tr><td>Agent-World-8B</td><td>8B</td><td>0.3</td><td>2.5</td><td>0.7</td><td>6.2</td><td>49.0</td><td>0.0</td><td>3.8</td><td>8.9</td></tr><tr><td>Agent-Reflex-8B</td><td>8B</td><td>0.5</td><td>3.3</td><td>0.8</td><td>3.1</td><td>51.3</td><td>0.0</td><td>5.3</td><td>9.2</td></tr><tr><td colspan="10">Open-Weight Foundation Models (27–35B)</td></tr><tr><td>Qwen3.6-27B 六</td><td>27B</td><td>10.7</td><td>22.1</td><td>5.6</td><td>15.8</td><td>58.5</td><td>0.8</td><td>38.4</td><td>21.7</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>35B</td><td>8.2</td><td>30.7</td><td>6.4</td><td>9.3</td><td>60.0</td><td>0.8</td><td>45.9</td><td>23.0</td></tr><tr><td>Nex-N2 Mini</td><td>35B</td><td>13.2</td><td>16.8</td><td>11.5</td><td>18.6</td><td>63.3</td><td>0.0</td><td>38.4</td><td>23.1</td></tr><tr><td>∞ Muse-Glimmer-30B</td><td>30B</td><td>14.2</td><td>32.9</td><td>3.5</td><td>11.3</td><td>55.5</td><td>0.0</td><td>13.5</td><td>18.7</td></tr><tr><td>WorkRover-35B-A3B</td><td>35B</td><td>28.7</td><td>一</td><td>36.9</td><td>一</td><td>一</td><td>一</td><td>48.9</td><td>一</td></tr><tr><td colspan="10">ToolSpace (Ours)</td></tr><tr><td>ToolSpace-8B</td><td>8B</td><td>32.3</td><td>3.9</td><td>13.1</td><td>10.3</td><td>65.3</td><td>0.0</td><td>12.0</td><td>19.6</td></tr><tr><td>+ Agent RL</td><td>8B</td><td>34.3</td><td>4.1</td><td>15.5</td><td>11.3</td><td>67.5</td><td>0.0</td><td>12.0</td><td>20.7</td></tr><tr><td>ToolSpace-35B-A3B</td><td>35B</td><td>45.8</td><td>42.4</td><td>40.0</td><td>34.0</td><td>74.3</td><td>1.7</td><td>48.9</td><td>41.0</td></tr><tr><td>+ Agent RL</td><td>35B</td><td>45.8</td><td>45.3</td><td>40.0</td><td>42.3</td><td>74.8</td><td>2.5</td><td>51.9</td><td>43.2</td></tr><tr><td>ToolSpace-744B-A40B</td><td>744B</td><td>54.7</td><td>67.8</td><td>57.5</td><td>43.3</td><td>78.4</td><td>27.5</td><td>67.1</td><td>56.6</td></tr></table>

where $r _ { i , t } ( \theta )$ is the token-level importance ratio between the current and rollout policies, and $c _ { \mathrm { l o w } }$ and $c _ { \mathrm { h i g h } }$ are the clipping bounds, set to 0.8 and 1.28 in our training. The advantage and the reward are defined as

$$
\hat { A } _ { i } = \frac { R _ { i } - \mu _ { g } } { s _ { g } + 1 0 ^ { - 6 } } , \qquad R _ { i } = \mathbb { I } \bigl [ \mathrm { E x e c u t e } ( V _ { \mathrm { c o d e } } ( y _ { i } , y _ { i } ^ { * } ) ) \bigr ] \in \{ 0 , 1 \} ,\tag{6}
$$

where $\mu _ { g }$ and $s _ { g }$ are the mean and standard deviation of $\{ R _ { i } \} _ { i = 1 } ^ { G }$ within rollout group �, i.e., the � rollouts sampled for the same prompt, and $R _ { i }$ is 1 only when execution-based verification of the completed trajectory passes, and 0 otherwise. Training runs for 48 steps with a learning rate of $2 \times 1 0 ^ { - 6 }$ , a total training batch size of 256, and a rollout group size of 16, in a fully asynchronous (full-async) manner with a maximum policy staleness of 4. Inference uses the oficial recommended settings, each benchmark follows its standard evaluation protocol, and all ToolSpace results are averaged over three independent runs.

Evaluation settings and result provenance. We prioritize oficial leaderboard results, followed by provider-reported scores under the oficial protocol and, when unavailable for open-weight models, local reproduction using the oficial harness, split, metric, tool-call protocol, and recommended inference settings. Local results are averaged over three independent runs; external scores are used as published.

Table 2: Results on AutomationBench Pro and $\tau ^ { 3 } .$ -Bench Banking Pro. Std. denotes the standard split; Pro denotes our upgraded split. Δ is Pro−Standard. Std. scores are taken from the Atria Dawn public leaderboard (https://atria-asi.ai/#benchmark-results). The scores of DeepSeek-V4.1-Flash follow their oficial release notes.
<table><tr><td>Frontier model</td><td colspan="3">AutomationBench</td><td colspan="3">τ3-Bench Banking</td></tr><tr><td></td><td>Std.</td><td>Pro</td><td>Δ↓</td><td>Std.</td><td>Pro</td><td>Δ↓</td></tr><tr><td>米Claude Opus 5.5</td><td>59.2</td><td>26.3</td><td>-32.9</td><td>51.5</td><td>35.7</td><td>-15.8</td></tr><tr><td>GPT 6.1 Sol</td><td>50.8</td><td>24.7</td><td>-26.1</td><td>46.4</td><td>29.6</td><td>-16.8</td></tr><tr><td>DeepSeek-V4.1-Flash-1017</td><td>60.5</td><td>22.3</td><td>-38.2</td><td>51.3</td><td>36.9</td><td>-14.4</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>54.8</td><td>20.0</td><td>-34.8</td><td>50.3</td><td>35.7</td><td>-14.6</td></tr><tr><td>Z GLM 5.3</td><td>52.5</td><td>22.7</td><td>-29.8</td><td>40.2</td><td>20.4</td><td>-19.8</td></tr><tr><td>Qwen 3.8 Max</td><td>49.7</td><td>24.0</td><td>-25.7</td><td>55.2</td><td>29.6</td><td>-25.6</td></tr><tr><td>Z GLM 5.2</td><td>29.3</td><td>18.0</td><td>-11.3</td><td>37.1</td><td>12.2</td><td>-24.9</td></tr></table>

## 6.2 Main Results

Long-horizon tool use and ofice productivity. Table 1 reports performance on seven benchmarks.   
ToolSpace is strong across model scales; three observations summarize this.

(1) Environment scaling learns the form of function calling, not tool orchestration. All 7–8B environment scaling baselines remain weak (<10) on the cross-environment, long-horizon suites AutomationBench, $\tau ^ { 3 } .$ -Bench Banking, JobBench, and Harvey LAB. Single-environment data therefore correct the form of calls without teaching cross-environment composition.

(2) ToolSpace exhibits cross-environment long-horizon execution across scales. Gains hold in each domain: ToolSpace-8B leads same-size environment scaling baselines on real productivity suites such as AutomationBench, JobBench, and OficeQA Pro; ToolSpace-35B-A3B exceeds its Qwen backbone on all seven benchmarks, with AutomationBench rising by more than 35 points (8.2→45.8). The pattern persists to 744B: ToolSpace-744B-A40B surpasses GLM 5.3, GPT 5.6 Sol, and its teacher policy K3 on average, and matches Claude Opus 5. These consistent gains across model scales, including performance beyond the teacher, argue against simple capability distillation and instead indicate that ToolSpace teaches transferable tool-composition structure for cross-environment execution.

(3) On-policy agent RL further unlocks capacity. SFT alone lifts ToolSpace by 18.0 points over the 35B backbone (23.0→41.0). On-policy RL on ToolSpace data adds a further 2.2 points at 35B (41.0→43.2), with the largest gain on $\tau ^ { 3 }$ -Bench Banking (34.0→42.3), showing that environment RL still leaves room for improvement. This indicates that on-policy environment RL can, through exploration, further unlock the multi-tool orchestration potential of SFT-trained models.

## 6.3 Results on AutomationBench Pro and $\tau ^ { 3 } .$ -Bench Banking Pro

To test whether ToolSpace raises frontier-benchmark dificulty, Table 2 compares seven frontier models on the two Pro splits with Standard. All seven frontier models drop substantially from Standard to Pro. Even Claude Opus 5.5 and GPT 6.1 Sol fall by 32.9 and 26.1 points on AutomationBench, and by 15.8 and 16.8 points on $\tau ^ { 3 } .$ -Bench Banking, respectively; the largest AutomationBench drop reaches 38.2 points. Together with Figure 6, where Pro tasks require more tool calls, this shows that, as crossenvironment and tool-execution dificulty rises sharply, even frontier models still struggle on these tasks, further underscoring ToolSpace’s flexible dificulty control.

## 6.4 Quantitative Analyses

Environment Scaling Analysis. To examine whether the breadth of the tool space matters independently of data volume, we fine-tune Qwen3.6-35B-A3B on data synthesized from increasing fractions of ToolSpace environments, from 1/16 to the full environment set. The backbone, training configuration, and evaluation protocol are held fixed, and zero denotes the untuned backbone. Figure 7(b)

![](images/c6ee3b42f1ca8d8239c65b916e967a9aa7c38c4129009701eaba89bc1dbf7105.jpg)

Figure 7: Data and environment scaling results. (a) Data scaling by training-data fraction. (b) Environment scaling by training-environment fraction. Zero denotes the untuned Qwen3.6-35B-A3B backbone.  
![](images/8d37cf6b7c04a5b68ec8c0e44710ef60ec66322968ec67283a35a66cb6955a31.jpg)  
Figure 8: Task-Synthesis Diversity Visualization. Synthesized task embeddings colored by DBSCAN cluster.

reports the results. Despite fluctuations at intermediate fractions, performance improves overall with environment scale and reaches the full-setting optimum, following a clear scaling trend. The increase on AutomationBench between $1 / 1 6$ and $1 / 8$ is particularly sharp, from 9.2 to 33.8. This suggests that new environment clusters contribute new cross-environment edges, and that space breadth is the limiting factor when coverage is low. In contrast, $\tau ^ { 3 } .$ -Bench Banking improves more slowly under environment scaling, which is consistent with its reliance on a narrower set of stateful banking tools. This supports the central motivation of ToolSpace: performance scales with the breadth of environments that witnessed chains connect. From the perspective of sustained scaling returns, AutomationBench gains about +4 points per doubling beyond $1 / 8$ on average, despite a plateau between 1/4 and $1 / 2$ , while �<sup>3</sup>-Bench Banking shows its largest jump only at the full environment set (19.6→34.0), indicating that highly connected environment clusters contribute late.

Data Scaling Analysis. To examine whether the benefit of ToolSpace grows with the amount of synthesized data, we fine-tune Qwen3.6-35B-A3B on increasing fractions of the ToolSpace SFT dataset, namely $1 / 1 6 , 1 / 8 , 1 / 4 , 1 / 2 .$ , and the full dataset. The backbone, training configuration, and evaluation protocol are held fixed, and zero denotes the untuned backbone. Figure $7 ( \mathrm { a } )$ reports the results. Performance improves overall as more data are used, and the full dataset performs best on all five benchmarks. SkillsBench shows a small intermediate decrease from 32.4 at $1 / 1 6$ to 32.0 at 1/8. AutomationBench rises from 25.2 at $1 / 1 6$ to 45.8 with the full dataset, and $\tau ^ { 3 } .$ -Bench Banking rises from 20.6 to 34.0. The gain is therefore not explained by a small set of high-quality examples. Each additional fraction contributes new witnessed tool chains and dependency structures, which is consistent with our motivation that capability grows from the composition space covered by the data. The small decreases below the backbone at the 1/16 fraction on BFCL V4 and OficeQA Pro indicate interference when the SFT mixture is too small, and the trend becomes stable once the data fraction increases. This demonstrates the scalability of ToolSpace’s synthesized data. From the perspective of sustained scaling returns, the step from 1/2 to the full dataset still improves all five benchmarks (AutomationBench +4.6), with no plateau, indicating that the current synthesis scale has not yet hit the returns ceiling of data volume.

![](images/d6588de74e81a5e386c621ecbac8c2cf7173be435ef654b9f09eeec186552428.jpg)

![](images/76fc1071dc72e4b4d6d0290af8125c5301fcda5015e835708e8325bcafc2dbd7.jpg)  
Figure 9: Environment relevance and synthesis cost. (a) Efect of environment relevance: Full, Top 50%, and Random 50% environments; Δ is the Top–Random gap. (b) Environment synthesis cost proxy: synthesized files by type over 500 sampled environments.

Environment Relevance Analysis. To verify that ToolSpace’s gains do not stem from overfitting to benchmark-like environments, we rank environment clusters by relevance to each benchmark using BGE (Chen et al., 2024) embeddings, and compare SFT on (i) the full data, (ii) only the top-50% most relevant environments (Top-50%), and (iii) a random 50% of environments (Random-50%) (Figure 9(a)). If the gains were driven by benchmark-specific environment fitting, Top-50% should substantially outperform Random-50%. Random-50% uses the same number of environments but excludes many of the most relevant ones. Instead, the two difer by at most 1.7 points on AutomationBench, SkillsBench, BFCL V4, and OficeQA Pro, and both approach full-data performance using only half the environments. ToolSpace’s gains therefore do not hinge on benchmark-aligned environments, but reflect transferable tool-composition structure.

Task-Synthesis Diversity Visualization. We examine whether ToolSpace spans a broader space of environments and tasks than prior environment-scaling methods by randomly sampling 2k questions from four SFT corpora (EnvScaler, Agent-World, Agent-Reflex, and ToolSpace). We employ BGE-M3 (Chen et al., 2024) as the embedding model, apply t-SNE for dimensionality reduction, and use DBSCAN (Ester et al., 1996) for clustering. As shown in Figure 8, ToolSpace forms the largest number of question clusters (42), compared with EnvScaler, Agent-World, and Agent-Reflex (25, 31, and 34, respectively), while retaining compact local groupings and broader coverage of the embedding space. This pattern suggests that ToolSpace expands task-space coverage beyond prior environment-scaling methods. We attribute the gain to edge-weighted sampling over observed tool compositions, which connects a wider range of environments.

Environment Synthesis Cost Analysis. We estimate ToolSpace’s environment synthesis cost savings using the number of synthesized files as a proxy. From 500 environments randomly sampled from Agent-World’s public library, we identify the files that ToolSpace would need to synthesize. Because file generation and validation complexity may vary, this proxy does not directly measure token, wall-clock, API, or compute cost.

As shown in Figure 9(b), Agent-World synthesizes all 2,724 files in the sample, whereas ToolSpace synthesizes only 1,218 files associated with operation-bearing environments—1,506 fewer files, or a 55.3% reduction. Specifically, JSON files decrease from 2,643 to 1,205 and CSV files from 42 to 7.

This reduction follows from ToolSpace’s selective-grounding strategy: persistent artifacts are materialized only for operation-bearing environments, while recorded returns are reused for stateless interactions. The lower artifact-generation workload indicates reduced environment synthesis cost, although exact token, time, and compute savings are not measured.

Table 3: Component ablation. Darker shading marks a larger within-column drop from Full.
<table><tr><td>Variant</td><td>Automation Bench</td><td>Skills Bench</td><td>BFCL V4</td><td> $\tau ^ { 3 } { \mathrm { - } } \mathbf { B e n c h }$  Banking</td><td>OfficeQA Pro</td></tr><tr><td>Full ToolSpace</td><td>45.8</td><td>42.4</td><td>74.3</td><td>34.0</td><td>48.9</td></tr><tr><td>w/o state edges</td><td>36.5</td><td>38.0</td><td>70.1</td><td>21.5</td><td>44.0</td></tr><tr><td>w/o data edges</td><td>38.2</td><td>37.5</td><td>62.4</td><td>29.8</td><td>40.6</td></tr><tr><td>w/o retry edges</td><td>43.9</td><td>41.2</td><td>72.8</td><td>32.6</td><td>47.9</td></tr><tr><td>w/o rollback</td><td>41.0</td><td>39.8</td><td>71.5</td><td>30.1</td><td>46.3</td></tr><tr><td>Uniform walk (no w)</td><td>39.6</td><td>38.9</td><td>69.8</td><td>28.4</td><td>45.1</td></tr><tr><td>LLM-judged edges</td><td>37.8</td><td>36.4</td><td>66.9</td><td>26.7</td><td>43.5</td></tr></table>

Component Ablation. Beyond which environments are selected and how much they cost to synthesize, we finally examine which ToolSpace components drive the gains. Table 3 reports a component ablation of ToolSpace at 35B, where w/o denotes removing or replacing one specific component. We make three observations. (1) Removing any component degrades all five benchmarks, showing that the components are complementary rather than redundant. (2) The two structural edge types divide the labor: state edges matter most for $\tau ^ { 3 }$ -Bench Banking and AutomationBench, where state preconditions decide whether a chain is executable at all, while data edges matter most for BFCL V4 and OficeQA Pro, where argument provenance carries the task. Their comparable average drops (7.1 vs. 7.4 points) fall on disjoint benchmarks, confirming that state dependencies, not data flow alone, enable crossenvironment execution. (3) How edges are obtained and traversed matters as much as which edges exist: LLM-judged edges degrade every benchmark almost as much as removing state edges, so edges must be witnessed by execution rather than merely plausible; rollback and edge-weighted sampling contribute in between, and retry edges have the smallest efect, consistent with their role in robustness to noisy tools.

## 7 Conclusion

In this paper, we present ToolSpace, a tool-centric framework for cross-environment agent data synthesis. ToolSpace mines data, state, and retry dependencies between adjacent calls in execution traces from real MCP services and open agent corpora, using deterministic rules rather than LLM judgment, and clusters the resulting weighted tool-composition graph into environments. Databases are synthesized only for state-mutating environments, while query and stateless tools replay recorded returns. ToolSpace then performs state-aware random walks with rollback and decompiles each chain into a natural-language task with assertion-based checks, keeping only samples that replay successfully from the initial state. Applied to AutomationBench and $\tau ^ { 3 } .$ -Bench Banking, the pipeline produces harder Pro versions on which seven frontier models drop by up to 38.2 points. Across seven long-horizon tool-use benchmarks, ToolSpace raises Qwen3.6-35B-A3B from 23.0 to 41.0 average score (43.2 with RL) and GLM-5.2 from 44.6 to 56.6, on par with Claude Opus 5. Ablations show that execution-witnessed edges clearly outperform LLM-judged ones, and performance keeps improving with data and environment scale.

## References

Anthropic. Model context protocol (MCP) specification. https://modelcontextprotocol.io, 2024. Accessed 26 Oct. 2026.

Anthropic. System card: Claude opus 5.5. https://www-cdn.anthropic.com/ fc1b44717c85dc068bc6ba5024219938094694bd/Claude%20Opus%205.5%20System%20Card. pdf, September 2026a. Released September 22, 2026.

Anthropic. Introducing Claude Opus 5. https://www.anthropic.com/news/claude-opus-5, 2026b. Oficial announcement, 24 Jul. 2026; accessed 26 Oct. 2026.

Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. M3-embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings of the Association for Computational Linguistics: ACL 2024, pages 2318–2335, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-acl.137. https:// aclanthology.org/2024.findings-acl.137/.

DeepSeek-AI. Deepseek-v4.1-flash: Pushing the limits of kv cache compression, 2026.

DeepSeek-AI, Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chengyu Hou, Chenhao Xu, Chenze Shao, Chong Ruan, Conner Sun, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Donghao Li, Dongjie Ji, Erhang Li, Fang Wei, Fangyun Lin, Fangzhou Yuan, Feiyu Xia, Fucong Dai, Guangbo Hao, Guanting Chen, Guoai Cao, Guolai Meng, Guowei Li, Han Yu, Han Zhang, Hanwei Xu, Hao Li, Haofen Liang, Haoling Zhang, Haoming Luo, Haoran Wei, Haotian Yuan, Haowei Zhang, Haowen Luo, Haoyu Chen, Haozhe Ji, Hengqing Zhang, Honghui Ding, Hongxuan Tang, Huanqi Cao, Huazuo Gao, Hui Qu, Hui Zeng, J Yang, JQ Zhu, Jia Luo, Jia Song, Jia Yu, Jialiang Huang, Jialu Cai, Jian Liang, Jiangting Zhou, Jiasheng Ye, Jiashi Li, Jiaxin Xu, Jiewen Hu, Jieyu Yang, Jin Chen, Jin Yan, Jingchang Chen, Jingli Zhou, Jingting Xiang, Jingyang Yuan, Jingyuan Cheng, Jingzi Zhou, Jinhua Zhu, Jiping Yu, Joseph Sun, Jun Ran, Junguang Jiang, Junjie Qiu, Junlong Li, Junmin Zheng, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Kexing Zhou, Kezhao Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Wang, Leyi Xia, Li Zhang, Liang Zhao, Lihua Guo, Lingxiao Luo, Linwang Ma, Linyan Zhu, Litong Wang, Liyu Cai, Liyue Zhang, Longhao Chen, MS Di, MY Xu, Max Mei, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Mingxu Zhou, Minmin Han, Ning Wang, Panpan Huang, Panpan Wang, Peixin Cong, Peiyi Wang, Peng Zhang, Qiancheng Wang, Qihao Zhu, Qingyang Li, Qinyu Chen, Qiushi Du, Qiwei Jiang, Rui Tian, Ruifan Xu, Ruijie Lu, Ruiling Xu, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runqian Chen, Runqiu Yin, Runxin Xu, Ruomeng Shen, Ruoyu Zhang, Ruyi Chen, SH Liu, Shanghao Lu, Shangmian Sun, Shangyan Zhou, Shanhuang Chen, Shaofei Cai, Shaoheng Nie, Shaoqing Wu, Shaoyuan Chen, Shengding Hu, Shengyu Liu, Shiqiang Hu, Shirong Ma, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, Shuying Yu, Songyang Zhou, Tao Ni, Tao Yun, Tian Jin, Tian Pei, Tian Ye, Tianle Lin, Tianran Ji, Tianyi Cui, Tianyuan Yue, Tingting Yu, Tun Wang, W Zhang, WL Xiao, Wangding Zeng, Wei An, Weilin Zhao, Wen Liu, Wenfeng Liang, Wenjie Pang, Wenjing Luo, Wenjing Yao, Wenjun Gao, Wenkai Yang, Wenlve Huang, Wenqing Hou, Wentao Zhang, Wenting Ma, Xi Gao, Xiang He, Xiangwen Wang, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaokang Zhang, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingchen Liu, Xingkai Yu, Xingyou Li, Xinyu Yang, Xinyu Zhang, Xu Chen, Xuanyu Wang, Xuecheng Su, Xueyin Chen, Xuheng Lin, Xuwei Fu, YC Yan, YQ Wang, YW Ma, Yanfeng Luo, Yang Zhang, Yanhong Xu, Yanru Ma, Yanwen Huang, Yao Li, Yao Li, Yao Xu, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Qian, Yi Shao, Yi Yu, Yichao Zhang, Yifan Ding, Yifan Shi, Yijia Wu, Yiliang Xiong, Yiling Ma, Ying He, Ying Tang, Ying Zhou, Yingjia Luo, Yinmin Zhong, Yishi Piao, Yisong

Wang, Yixiang Zhang, Yixiao Chen, Yixuan Tan, Yixuan Wei, Yiyang Ma, Yiyuan Liu, Yonglun Yang, Yongqiang Guo, Yongtong Wu, Yu Wu, YuKun Li, Yuan Cheng, Yuan Ou, Yuanfan Xu, Yuanhao Li, Yuduan Wang, Yuehan Yang, Yuer Xu, Yuhan Wu, Yuhao Meng, Yuheng Zou, Yukun Zha, Yunfan Xiong, Yupeng Chen, Yuping Lin, Yuqian Cao, Yuqian Wang, Yushun Zhang, Yuting Yan, Yutong Lin, Yuxian Gu, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuxuan Zhou, Yuyang Zhou, Yuzhen Huang, ZF Wu, Zehao Wang, Zehua Zhao, Zehui Ren, Zekai Zhang, Zhangli Sha, Zhe Fu, Zhe Ju, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zheren Gao, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhixian Huang, Zhixuan Chen, Zhiyu Wu, Zhizhou Ren, Zhongyu Wu, Zhuoshu Li, Zhuping Zhang, Zian Xu, Zihao Wang, Zihua Qu, Zihui Gu, Zijia Zhu, Zilin Li, Zipeng Zhang, Ziwei Xie, Ziyi Gao, Ziyi Wan, Zizheng Pan, and Zongqing Yao. Deepseek-v4: Towards highly eficient million-token context intelligence, 2026. https://arxiv.org/abs/2606.19348.

Guanting Dong, Junting Lu, Junjie Huang, Wanjun Zhong, Longxiang Liu, Shijue Huang, Zhenyu Li, Yang Zhao, Xiaoshuai Song, Xiaoxi Li, Jiajie Jin, Yutao Zhu, Hanbin Wang, Fangyu Lei, Qinyu Luo, Mingyang Chen, Zehui Chen, Jiazhan Feng, Ji-Rong Wen, and Zhicheng Dou. Agent-world: Scaling real-world environment synthesis for evolving general agent intelligence. CoRR, abs/2604.18292, 2026a. doi: 10.48550/ARXIV.2604.18292. https://doi.org/10.48550/arXiv.2604.18292.

Guanting Dong, Xinyu Yang, Jiajie Jin, Yuyang Hu, Xiaoshuai Song, Yifei Chen, Xiaoxi Li, Jiejun Tan, Yutao Zhu, Tat-Seng Chua, Ji-Rong Wen, and Zhicheng Dou. Agent-reflex: Teaching language agents to act reflectively, 2026b. https://arxiv.org/abs/2610.17953.

Martin Ester, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. A density-based algorithm for discovering clusters in large spatial databases with noise. In Proceedings of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 226–231, 1996.

Runnan Fang, Shihao Cai, Baixuan Li, Jialong Wu, Guangyu Li, Wenbiao Yin, Xinyu Wang, Xiaobin Wang, Liangcai Su, Zhen Zhang, Shibin Wu, Zhengwei Tao, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. Towards general agentic intelligence via environment scaling, 2025. https: //arxiv.org/abs/2509.13311.

Jiacheng Guo, Ling Yang, Peter Chen, Qixin Xiao, Yinjie Wang, Xinzhe Juan, Jiahao Qiu, Ke Shen, and Mengdi Wang. Genenv: Dificulty-aligned co-evolution between llm agents and environment simulators, 2025. https://arxiv.org/abs/2512.19682.

Harvey AI. Harvey lab: The legal agent benchmark, 2026. https://github.com/ harveyai/harvey-labs/tree/v1.0. Announcement: https://www.harvey.ai/blog/ introducing-harveys-legal-agent-benchmark.

Hang He, Linting Wang, Guanming Liu, Bo Mao, Jinxiu Liu, Yiran Suo, Maosen Zhou, Zhiheng Xi, Guanting Dong, Zhou Shao, Rui Zheng, Jie Zhou, Chengcheng Wan, Tao Gui, Qi Zhang, Xipeng Qiu, and Xuanjing Huang. Workrover: Scaling long-horizon cowork agent training via recursive environment evolution, 2026. https://arxiv.org/abs/2610.18067.

Minghao Li, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, Haiyang Yu, Zhoujun Li, Fei Huang, and Yongbin Li. API-bank: A comprehensive benchmark for tool-augmented LLMs. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3102–3116, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.187. https://aclanthology.org/ 2023.emnlp-main.187/.

Xiangyi Li, Wenbo Chen, Yimin Liu, Shenghan Zheng, Xiaokun Chen, Yifeng He, Yubo Li, Bingran You, Haotian Shen, Jiankai Sun, Shuyi Wang, Qunhong Zeng, Di Wang, Xuandong Zhao, Yuanli Wang, Roey Ben Chaim, Zonglin Di, Yipeng Gao, Junwei He, Yizhuo He, Liqiang Jing, Luyang

Kong, Xin Lan, Jiachen Li, Songlin Li, Yijiang Li, Yueqian Lin, Xinyi Liu, Xuanqing Liu, Haoran Lyu, Ze Ma, Bowei Wang, Runhui Wang, Tianfu Wang, Wengao Ye, Yue Zhang, Hanwen Xing, Yiqi Xue, Steven Dillmann, and Han-Chung Lee. Skillsbench: Benchmarking how well agent skills work across diverse tasks. CoRR, abs/2602.12670, 2026a. https://arxiv.org/abs/2602.12670.

Yuetai Li, Huseyin A Inan, Xiang Yue, Wei-Ning Chen, Lukas Wutschitz, Janardhan Kulkarni, Radha Poovendran, Robert Sim, and Saravan Rajmohan. Simulating environments with reasoning models for agent training, 2025. https://arxiv.org/abs/2511.01824.

Yuetai Li, Yichen Feng, Zhangchen Xu, Zixian Ma, Kaiyuan Zheng, Fengqing Jiang, Xinghua Sun, Rulin Shao, Zichen Chen, Yue Huang, Xinyang Han, Brian Lee, Kayla Xu, Shenglai Zeng, Hang Hua, Xiangliang Zhang, Basel Alomair, Ranjay Krishna, Luke Zettlemoyer, Pang Wei Koh, Bhaskar Ramasubramanian, Luyao Niu, Xiang Yue, and Radha Poovendran. Jobbench: Aligning agent work with human will, 2026b. https://arxiv.org/abs/2605.26329.

Quanyu Long, Xiao Chen, Jianda Chen, Haozhen Zhang, Qisheng Hu, Jianzhu Bao, and Wenya Wang. From traces to agentic worlds: Agentic language world models for interactive environment simulation. CoRR, abs/2610.06100, 2026. https://doi.org/10.48550/arXiv.2610.06100.

Bo Mao, Hang He, Linting Wang, Lizhi Lin, Maosen Zhou, Guanming Liu, Jinxiu Liu, Tianyu Huai, Chaoyun Zhang, Bingxuan Li, Kepeng Lei, Guanting Dong, Zhou Shao, Rui Zheng, Hang Yan, Jie Zhou, Chengcheng Wan, Tao Gui, Liang He, and Xipeng Qiu. Weft: Scaling tool-use post-training for general-purpose agents, 2026. https://arxiv.org/abs/2609.36887.

Meta. Muse Glimmer 30B: An on-device agentic model. https://huggingface.co/meta-models/ Muse-Glimmer-30B, 2026. Oficial model card; accessed 26 Oct. 2026.

Moonshot AI. Kimi K3. https://huggingface.co/moonshotai/Kimi-K3, 2026. Oficial model card and technical report; accessed 26 Oct. 2026.

Nex-AGI. Nex-N2-mini. https://huggingface.co/nex-agi/Nex-N2-mini, 2026. Oficial model card; accessed 26 Oct. 2026.

OpenAI. GPT-5.6: Frontier intelligence that scales with your ambition. https://openai.com/index/ gpt-5-6/, 2026a. Oficial announcement of the GPT-5.6 family incl. GPT-5.6 Sol, 9 Jul. 2026; accessed 26 Oct. 2026.

OpenAI. Addendum: Gpt-6.1 sol system card. https://cdn.openai.com/pdf/ 38e3efcf-545e-44cd-99ec-2b7eb395f4cc/oai\_GPT\_6\_1\_Sol.pdf, September 2026b. Released September 29, 2026.

Krista Opsahl-Ong, Arnav Singhvi, Jasmine Collins, Ivan Zhou, Cindy Wang, Ashutosh Baheti, Owen Oertell, Jacob Portes, Sam Havens, Erich Elsen, Michael Bendersky, Matei Zaharia, and Xing Chen. Oficeqa pro: An enterprise benchmark for end-to-end grounded reasoning. CoRR, abs/2603.08655, 2026. https://arxiv.org/abs/2603.08655.

Shishir G. Patil, Tianjun Zhang, Xin Wang, and Joseph E. Gonzalez. Gorilla: Large language model connected with massive apis, 2023. https://arxiv.org/abs/2305.15334.

Shishir G. Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The berkeley function calling leaderboard (bfcl): From tool use to agentic evaluation of large language models. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 48371–48392. PMLR, 2025. https://proceedings.mlr.press/v267/patil25a.html.

Akshara Prabhakar, Zuxin Liu, Ming Zhu, Jianguo Zhang, Tulika Awalgaonkar, Shiyu Wang, Zhiwei Liu, Haolin Chen, Thai Hoang, Juan Carlos Niebles, Shelby Heinecke, Weiran Yao, Huan Wang, Silvio Savarese, and Caiming Xiong. Apigen-mt: Agentic pipeline for multi-turn data generation via simulated agent-human interplay, 2025. https://arxiv.org/abs/2504.03601.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. Toolllm: Facilitating large language models to master 16000+ real-world apis, 2023. https://arxiv.org/abs/2307.16789.

Qwen Team. Qwen3.6-27B: Flagship-level coding in a 27B dense model, April 2026a. https://qwen. ai/blog?id=qwen3.6-27b.

Qwen Team. Qwen3.6-35B-A3B. https://huggingface.co/Qwen/Qwen3.6-35B-A3B, 2026b. Oficial model card; accessed 26 Oct. 2026.

Qwen Team. Qwen3.8-Max: A new bar for coding and cowork. https://qwen.ai/blog?id=qwen3.8, 2026c. Oficial release post; accessed 26 Oct. 2026.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 68539–68551. Curran Associates, Inc., 2023. doi: 10.52202/075280-2997. https://proceedings.neurips.cc/paper\_files/ paper/2023/file/d842425e4bf79ba039352da0f658a906-Paper-Conference.pdf.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. https://arxiv.org/abs/2402.03300.

Daniel Shepard and Robin Salimans. Automationbench, 2026. https://arxiv.org/abs/2604.18934.

Quan Shi, Alexandra Zytek, Pedram Razavi, Karthik R Narasimhan, and Victor Barres. �-knowledge: Evaluating conversational agents over unstructured knowledge. In Tong Zhang, Miroslav Dudik, Martin Jaggi, Alekh Agarwal, Sharon Li, Dale Schuurmans, Jerry Zhu, Felix Berkenkamp, Hanze Dong, and Alberto Bietti, editors, Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 112443–112471. PMLR, 06–11 Jul 2026. https://proceedings.mlr.press/v306/shi26ah.html.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: language agents with verbal reinforcement learning. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 8634–8652. Curran Associates, Inc., 2023. doi: 10.52202/075280-0377. https://proceedings.neurips.cc/paper\_files/paper/2023/file/ 1b44b878bb782e6954cd888628510e90-Paper-Conference.pdf.

Xiaoshuai Song, Haofei Chang, Guanting Dong, Yutao Zhu, Ji-Rong Wen, and Zhicheng Dou. Envscaler: Scaling tool-interactive environments for llm agent via programmatic synthesis, 2026. https: //arxiv.org/abs/2601.05808.

Yifan Song, Weimin Xiong, Dawei Zhu, Wenhao Wu, Han Qian, Mingbo Song, Hailiang Huang, Cheng Li, Ke Wang, Rong Yao, Ye Tian, and Sujian Li. Restgpt: Connecting large language models with real-world restful apis, 2023. https://arxiv.org/abs/2306.06624.

Zhengwei Tao, Jialong Wu, Wenbiao Yin, Junkai Zhang, Baixuan Li, Haiyang Shen, Kuan Li, Liwen Zhang, Xinyu Wang, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. Webshaper: Agentically data synthesizing via information-seeking formalization, 2025. https://arxiv.org/abs/2507.15061.

Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighof, Yizhe Zhang, Binyuan Hui, Junyang Lin, Robert Brennan, Hao Peng, Heng Ji, and Graham Neubig. Openhands: An open platform for ai software developers as generalist agents, 2025. https://arxiv.org/abs/2407.16741.

Zhaoyang Wang, Canwen Xu, Boyi Liu, Yite Wang, Siwei Han, Zhewei Yao, Huaxiu Yao, and Yuxiong He. Agent world model: Infinity synthetic environments for agentic reinforcement learning, 2026. https://arxiv.org/abs/2602.10090.

Jialong Wu, Baixuan Li, Runnan Fang, Wenbiao Yin, Liwen Zhang, Zhengwei Tao, Dingchu Zhang, Zekun Xi, Gang Fu, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. Webdancer: Towards autonomous information seeking agency, 2025. https://arxiv.org/abs/2505.22648.

Jie Wu, Zhenru Zhang, Beichen Zhang, Xuwu Wang, Yuhui Su, Mouxiang Chen, Peng Wang, Zhihai Wang, Que Shen, Hao Zhou, An Yang, Fei Huang, Yujiu Yang, and Dayiheng Liu. Terminal-universe: Turning agent trajectories into scalable terminal environments. CoRR, abs/2609.04148, 2026. https://doi.org/10.48550/arXiv.2609.04148.

Zhangchen Xu, Adriana Meza Soria, Shawn Tan, Anurag Roy, Ashish Sunil Agrawal, Radha Poovendran, and Rameswar Panda. Toucan: Synthesizing 1.5m tool-agentic data from real-world mcp environments, 2025. https://arxiv.org/abs/2510.01179.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. https://arxiv.org/abs/2505.09388.

John Yang, Carlos Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 50528–50652. Curran Associates, Inc., 2024. doi: 10.52202/079017-1601. https://proceedings.neurips.cc/paper\_files/paper/2024/file/ 5a7c947568c1b1328ccc5230172e1e7c-Paper-Conference.pdf.

Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models, 2023. https://arxiv.org/abs/2210.03629.

Fan Yin, Zifeng Wang, I-Hung Hsu, Jun Yan, Ke Jiang, Yanfei Chen, Jindong Gu, Long Le, Kai-Wei Chang, Chen-Yu Lee, Hamid Palangi, and Tomas Pfister. Magnet: Multi-turn tool-use data synthesis and distillation via graph translation. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 32600–32616, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.1566. https://aclanthology.org/2025.acl-long.1566/.

Bofei Zhang, Zirui Shang, Zhi Gao, Wang Zhang, Rui Xie, Xiaojian Ma, Tao Yuan, Xinxiao Wu, Song-Chun Zhu, and Qing Li. Tongui: Internet-scale trajectories from multimodal web tutorials for generalized gui agents, 2025. https://arxiv.org/abs/2504.12679.

Zhipu AI. GLM-5.2: Built for long-horizon tasks. https://huggingface.co/zai-org/GLM-5.2, 2026a. Oficial model card; accessed 26 Oct. 2026.

Zhipu AI. GLM-5.3: Frontier coding with emergent cyber capabilities. https://z.ai/blog/glm-5.3, 2026b. Oficial release post, 14 Aug. 2026; accessed 26 Oct. 2026.