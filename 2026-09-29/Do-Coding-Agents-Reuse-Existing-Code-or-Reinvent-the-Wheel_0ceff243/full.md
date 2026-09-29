# Do Coding Agents Reuse Existing Code or Reinvent the Wheel?

Dongsheng Ma<sup>†1</sup>, Sizhe Wang<sup>†2</sup>, Xinyi Huang<sup>1</sup>, Zhengren Wang<sup>3</sup>

Yuhan Wang<sup>1</sup>, Luyang Si<sup>4</sup>, Xincheng Wei<sup>5</sup>, Wentao Zhang<sup>\*1,6</sup>

<sup>1</sup>Peking University <sup>2</sup>Fudan University <sup>3</sup>Shanghai Jiao Tong University <sup>4</sup>Tsinghua University

<sup>5</sup>The Chinese University of Hong Kong, Shenzhen <sup>6</sup>Zhongguancun Academy madongsheng26@stu.pku.edu.cn, wangsz26@m.fudan.edu.cn, wentao.zhang@pku.edu.cn

## Abstract

Coding agents are increasingly deployed for iterative development on real repositories, yet existing evaluation barely answers a basic question: do coding agents reuse existing code or reinvent the wheel? The question matters: every duplicated implementation is a fix applied twice and agents produce code far faster than humans can audit, so redundancy accumulates unsupervised. Thus, we present RepoReuse, a multi-turn benchmark for auditing code reuse in real repositories, where requirements are revealed turn by turn and the workspace accumulates across turns. It is built by a fully automated pipeline combining AST-based dependency graphs, guided evidence collection, and execution-verified task synthesis, and scales readily to new repositories. Beyond pass rates, we measure the reuse rate together with recall and cross-turn structural redundancy. An audit over 3,000 turns shows that agents progressively stop exploring relevant repository code, reuse their own history less even when it is fully in the workspace, and leave duplicated logic in 50.8% of task chains by turn 5—all while pass rates barely move. Such deficiencies are invisible to pass rates, underscoring the need to evaluate code generation beyond functional correctness.

## 1 Introduction

Coding agents continue to break records on coding benchmarks (Team et al., 2026; DeepSeek-AI et al., 2026; Yang et al., 2024; Wang et al., 2025a; Jimenez et al., 2024), and are increasingly deployed in real repositories to assist humans with iterative development. Yet passing tests is only half of what makes code good. Code is read, extended, and maintained long after it is written; a solution that reinvents the wheel is not a good solution, even when every test is green.

Motivation Redundancy is not a cosmetic issue. Over time, every duplicated implementation means a bug fix or an interface change that must be applied twice, a piece of functionality that silently drifts between parallel copies, and more code that every later turn—human or agent—must wade through. Recent studies have already observed the signs: in long-horizon iterative tasks, the redundancy of agent-produced code keeps rising, and agent-generated patches are markedly longer than reference solutions that reuse existing modules (Orlanski et al., 2026; Li et al., 2026; Abbassi et al., 2025; Liu et al., 2025). What is truly alarming is the asymmetry of speed: agents write code far faster than human experts can audit it, so this failure mode accumulates at a staggering rate, quietly piling up into “spaghetti code.”

Yet our evaluation instruments are, in principle, blind to all of this. Existing benchmarks are built almost entirely around functional correctness: HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021), and SWE-bench (Jimenez et al., 2024) score only whether tests pass (i.e., the pass rate), paying little attention to another key dimension of how models write code—reusability. Prior work touches on this dimension but does not measure it in realistic multi-turn, repository-scale development: benchmarks on reuse and maintainability either iterate only at the function level (Wang et al., 2026) or are confined to a single turn (Wang et al., 2025b), while repository-level efforts stop at oneshot completion or well-documented entry points rather than the discovery and reuse of internal modules (Ding et al., 2023; Liu et al., 2023; Tang et al., 2024).

Real iterative development unfolds precisely at the repository scale, across turns: a model must keep developing on an existing multi-file repository, reusing existing code—not only the repository’s own modules, but also the code it wrote in previous turns, which has by then become part of the repository. To this end, we construct RepoReuse, a multiturn benchmark for auditing code reuse in real repositories (See Figure 1). Requirements are revealed turn by turn and the workspace accumulates across turns, so that an agent’s reuse of both preexisting repository modules and its own historical implementations can be measured turn by turn. RepoReuse is built by a fully automated pipeline that combines AST-based dependency graphs, guided evidence collection, and execution-verified task synthesis, and scales readily to new repositories (Le et al., 2026; Jain et al., 2024).

![](images/462f785da25a9644d9d3e27865a01e598e020347fa77c83f8810fbad401a9ed3.jpg)  
Figure 1: Overview of RepoReuse. Left: the benchmark protocol—turn-wise requirements over a workspace that accumulates the agent’s own code, scored per turn by functional tests and by recall, reuse, and redundancy. Right: an illustrative example—the agent reuses one repository symbol and one of its own earlier functions but rewrites the two remaining targets; tests pass while both reuse rates are 1/2.

Our metrics mirror the agent’s per-turn workflow: reading code, deciding what to reuse, and writing new code. The central metric, reuse rate, asks whether the submission calls each target through a real call edge, identified by AST analysis against the reference solution’s actual dependencies; repository and self-produced targets are measured separately $( \mathrm { r e u s e } _ { \mathrm { r e p o } } / \mathrm { r e u s e } _ { \mathrm { s e l f } } )$ . To explain why a target is missed, upstream recall records the fraction of target source code read in the current turn, separating exploration-side gaps (never read) from execution-side gaps (read but not used); downstream, $C _ { \mathrm { d u p } }$ counts targets that are not called yet have most of their logic rewritten, capturing the redundancy they leave behind (Jiang et al., 2007). Together, the three trace a reuse decision from exploration to consequence, forming a complete auditing instrument for code reuse.

With this auditing instrument, we uncover systematic reuse deficiencies of coding agents. On the exploration side, agents gradually stop retrieving as turns progress: they read 83.6% of relevant repository code at turn 1 on average, but only 35.4% at turn 5. Once the workspace is filled with their own code, reuse decisions are increasingly made from signatures and priors rather than grounded in current-turn observation. On the execution side, a more counterintuitive phenomenon is that seeing is not using: recall over their own historical code is nearly saturated in every turn, yet self-reuse still decays from 83.9% to 69.1%. Ablations further show that disclosing the complete source of historical implementations barely helps, performing on par with providing no memory at all—the bottleneck is not access to information, but the agent’s own disposition to build on existing code. Downstream, bypassed targets steadily accumulate as structural redundancy in the workspace: the fraction of task chains containing a cross-turn reimplementation climbs from 13.8% at turn 1 to

50.8% at turn 5, while pass rates barely move. Agents solve task after task correctly while the codebase grows messier turn after turn, pass-ratecentered evaluation is structurally blind to all of it.

Our contributions are as follows:

• An important question: we raise a question neglected by existing evaluation: do coding agents reuse existing code or reinvent the wheel? We give the first systematic, turn-byturn measurable answer: agents steadily accumulate redundant code that piles up into “spaghetti code.”

• The RepoReuse benchmark: a repositorylevel multi-turn iterative development benchmark with turn-wise requirement revelation, measuring an agent’s reuse of both preexisting repository code and its own historical code turn by turn. It is built by a fully automated pipeline that requires no human involvement and scales readily to new repositories.

• Auditing findings: a systematic audit over 3,000 turns of development shows that reuse deficiencies are pervasive and deepen over turns: agents see their own historical code yet fail to reuse it (self reuse decays from 83.9% to 69.1% even under saturated recall); by turn 5, half of the task chains have accumulated cross-turn re-implementations while pass rates barely move—deficiencies entirely invisible to pass-rate-centered evaluation.

## 2 Related Work

Coding Agents and Harnesses Coding agents are moving from research prototypes into real development workflows: new-generation models (e.g., Kimi K3 (Team et al., 2026), DeepSeek v4.1 (DeepSeek-AI et al., 2026)) treat agentic coding as a core capability, while harnesses range from research frameworks (SWE-agent (Yang et al., 2024), OpenHands (Wang et al., 2025a)) to industrial products (Claude Code (Anthropic, 2025), Codex (OpenAI, 2025), DeepSeek Harness (DeepSeek-AI, 2026)). It is increasingly clear that harness and interaction design have become performance variables on par with the backbone model (Vats and Golev, 2026): SWE-agent highlights the role of tool-interface design, and Agentless (Xia et al., 2024) shows that a simple localization–repair–validation pipeline is competitive even without an autonomous interaction loop. Yet evaluation along this line is defined almost entirely by task success rate—how an agent writes code (whether it reuses existing implementations or piles up redundancy) is not measured. RepoReuse instead audits the reuse behavior of existing agents and harnesses.

Code Generation Evaluation Code generation evaluation has long centered on functional correctness, from function-level tests (HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021)) to issue resolution in real repositories (SWE bench (Jimenez et al., 2024)) and repository-level completion (CrossCodeEval (Ding et al., 2023), RepoBench (Liu et al., 2023), ML-Bench (Tang et al., 2024)). Closest to us are multi-turn benchmarks: CodeFlowBench (Wang et al., 2026) iterates within a single file; MaintainCoder (Wang et al., 2025b) varies requirements but remains single-turn; SWE-Chain (Lam et al., 2026) asks agents to upgrade a package step by step along its version chain, each step building on the agent’s own previous codebase; EvoCode-Bench (Shen et al., 2026) and SWE-EVO (Le et al., 2026) evaluate multi-turn iterative interaction and long-horizon version evolution under per-turn functional tests; LoopsBench (Li et al., 2026) organizes each task as a dependency DAG with tests released along the ready frontier, observing bloated patches and prerequisite dependencies missed by agent plans; SlopCodeBench (Orlanski et al., 2026) lets agents repeatedly extend their own solutions under evolving specifications, observing structural erosion and rising redundancy; SWE-Explore (Zhang et al., 2026b) measures linelevel exploration recall, but over static snapshots with execution deliberately removed. These works measure outcomes and artifacts, and none audits models’ reuse behavior in realistic repository-level multi-turn development. RepoReuse fills this gap: under a repository-level multi-turn iterative setting, it measures, turn by turn, an agent’s reuse of both pre-existing repository code and its own historical code.

## 3 RepoReuse

## 3.1 Benchmark Construction

Constructing RepoReuse requires delivering two things at once: multi-turn task chains, and a groundtruth answer to “what should be reused at each turn.” To this end, we implement a fully automatic construction pipeline: starting from mature opensource repositories, it produces tasks through three stages—evidence collection, task synthesis, and execution verification—without any human involvement. The pipeline guarantees that the reference solution genuinely reuses the designated modules and that the test cases come from actual execution, so that reuse auditing rests on verifiable ground truth.

Repository Sources RepoReuse builds its tasks on mature, widely used open-source Python libraries spanning domains such as scientific computing and data visualization. Selected repositories must satisfy three conditions: (1) a multi-file pure-Python codebase with a rich internal implementation layer beneath the public API—the evidence for our tasks comes from these undocumented internal subpackages rather than well-documented public interfaces; (2) a reproducible environment: each repository is pinned to a specific commit, and its dependency installation recipe is directly inherited from SWE-bench’s (Jimenez et al., 2024) install table, ensuring consistency between the construction and evaluation environments.

Dependency Graph and Evidence Collection We first build a callable “map” of the repository: via AST-based static analysis, functions and methods are represented as nodes carrying precise line ranges, signatures, and docstrings, with call relations as edges, forming a dependency tree that covers the whole repository; internal symbols beneath the public API surface are marked as candidate starting points. A strong model (GPT-5.6 Sol (OpenAI, 2026)) then performs guided walks over the dependency tree: starting from an internal symbol, it selects the next symbol hop by hop with justifications, collecting a group of interdependent modules as an evidence pack (Li et al., 2025a; Wu et al., 2025; Ma et al., 2026; Zhang et al., 2026a). Walks from turn 2 onward depart from the previous turn’s artifact, making the reuse of earlier-turn functions a natural requirement of the task chain; to prevent evidence packs from degenerating into self-wrapping over turns, an additional nearby but previously unused repository symbol is injected during each walk.

Task Synthesis Based on the evidence pack, the authoring model (GPT-5.6 Sol) synthesizes the complete content of a turn: (1) a requirement that describes only the target behavior and never names any evidence-pack symbol—otherwise exploration degenerates into a string search; (2) a reference solution that must genuinely call the evidence-pack modules and the chain’s previous-turn artifacts; (3) test cases whose expected values are taken from actual runs of the reference solution rather than the authoring model’s claimed outputs. Modules that pass verification are written back into the repository package under semantic names, side by side with real code, for later turns to depend on.

Table 1: Statistics of RepoReuse.
<table><tr><td>Statistic Value</td></tr><tr><td>Tasks 75</td></tr><tr><td>Avg. turns per task 5.0</td></tr><tr><td>Avg. test cases per turn 10.8</td></tr><tr><td>Avg. requirement length (chars) ~2,570</td></tr><tr><td>New functions per turn 3</td></tr><tr><td>Reused symbols per turn (repo / self) 1.9 / 2.4</td></tr></table>

Verification The core of verification is the reliability of the test cases themselves: every expected value is obtained by actually executing the reference solution, and cases that raise errors or produce unstable outputs are discarded; if too many are discarded for the remaining cases to cover the declared functions, the whole turn is regenerated. On top of this, each turn must pass automatic checks—e.g., the reference solution passes all tests, an empty implementation must fail, and AST analysis confirms that the reference solution genuinely reuses the evidence-pack modules. Finally, an independent audit, trusting none of the generator’s outputs, reinstalls the entire task chain and re-runs all tests before the task is packaged for delivery.

## 3.2 Benchmark Overview

Table 1 summarizes the scale and composition of RepoReuse. Each turn ships with 10.8 executionverified test cases on average; from turn 2 onward, the reference solution reuses 1.9 pre-existing repository symbols and 2.4 functions produced in earlier turns per turn—both types of reuse targets genuinely exist in every turn, providing a stable measurement basis for reuse auditing.

## 3.3 Evaluation Framework

In each turn, a coding agent reads existing code, decides which modules to reuse, and generates new code. We first formalize this multi-turn setting and

the reuse targets it induces, and then define metrics organized around reuse along the same workflow.

## 3.3.1 Task Formulation

Let $W _ { 0 }$ denote the initial repository. At each turn $t ,$ the agent receives a requirement $q _ { t }$ and the current workspace $W _ { t - 1 }$ , and produces a submission $s _ { t } \colon$

$$
( q _ { t } , \ W _ { t - 1 } ) \ { \xrightarrow { \mathrm { \ a g e n t } } } \ s _ { t } , \qquad W _ { t } = W _ { t - 1 } \oplus s _ { t }
$$

where ⊕ denotes installing the submission into the workspace. The agent’s own submissions, not the reference solutions, are carried into later turns, so every turn builds on what the agent itself wrote. For each turn, we extract the reuse targets from the actual dependencies of the reference solution via AST analysis and split them into two disjoint sets: repo targets ${ \mathcal { T } } _ { t } ^ { \mathrm { r e p o } } \subseteq W _ { 0 }$ , functions present in the original repository, and $s e l f$ targets $\mathcal { T } _ { t } ^ { \mathrm { s e l f } } \subseteq$ $\{ s _ { 1 } , \ldots , s _ { t - 1 } \}$ , functions produced by the agent in earlier turns. Self targets exist only from turn 2 onward.

## 3.3.2 Metric Definition

Our metrics mirror the per-turn workflow. The central reuse metric asks whether the agent adopts each target. Upstream metrics describe what the agent explored before writing code, and help explain why a target was or was not reused. Downstream metrics describe the outcome, both whether the new code works and what bypassing a target leaves in the workspace.

Reuse Metric For $s \in \{ \mathrm { r e p o } , \mathrm { s e l f } \}$ , the reuse rate is the fraction of targets that the submission directly calls:

$$
\operatorname { r e u s e } _ { s } ( t ) = { \frac { | \{ f \in { \mathcal { T } } _ { t } ^ { s } : f { \mathrm { ~ i s ~ c a l l e d ~ i n ~ } } s _ { t } \} | } { | { \mathcal { T } } _ { t } ^ { s } | } }\tag{1}
$$

A call is identified by a real call edge to the target through AST-based value-reference name matching. Each target is scored 0/1 regardless of invocation count, and a function the agent redefines locally under the same name does not count as a call. For example, reus $\mathfrak { e } _ { \mathrm { s e l f } } ( t ) = 0 . 5$ means that the submission calls half of the earlier-turn functions the reference solution builds on and bypasses the other half.

Upstream Metrics Reuse rate tells us what happened but not why: a target that was not reused may never have been found, or may have been found but not adopted. To separate the two, we measure recall. A target is recalled if the agent’s file-read actions in the current turn overlap at least one line of its source body; directory listings do not count as reads:

$$
\operatorname { r e c a l l } _ { s } ( t ) = { \frac { | \{ f \in { \mathcal { T } } _ { t } ^ { s } : f { \mathrm { ~ i s ~ r e a d ~ i n ~ t u r n ~ } } t \} | } { | { \mathcal { T } } _ { t } ^ { s } | } }\tag{2}
$$

Combined with reuse, recall sorts every missed target into one of two cases. A target that was read but not reused points to an execution-side gap, where the agent saw the code but chose not to use it; a target neither read nor reused points to an explorationside gap, where the agent never found it. To characterize how much the agent explores, we also report two behavior statistics per turn: Files, the number of distinct files viewed, searched, or listed, and Lines, the number of distinct source lines viewed. These statistics are descriptive rather than evaluative and help interpret recall differences across models and harnesses.

Downstream Metrics We first retain standard functional correctness. Pass rate is the proportion of test cases passed in a turn, and resolve rate is the proportion of turns in which all tests pass. Correctness, however, captures only one consequence of reuse. As agents produce code faster than humans can audit it, an agent may solve each task correctly while repeatedly re-implementing functionality that already exists, quietly turning the codebase into redundant “spaghetti code.” We therefore also measure whether the agent re-implements existing code (Jiang et al., 2007). A target $f \in \mathcal { T } _ { t } ^ { \mathrm { r e p o } } \cup \mathcal { T } _ { t } ^ { \mathrm { s e l f } }$ is re-implemented at turn t if $s _ { t }$ does not call $f$ but calls at least 80% of the non-builtin functions that $f$ itself calls, considering only targets that call at least three such functions. Because every turn inherits the workspace of earlier turns, a single reimplementation leaves a redundant copy for the rest of the task chain; we call any chain containing one a duplicated chain, and measure the duplicatedchain rate, the fraction of task chains C that have become duplicated by turn t:

$$
C _ { \mathrm { d u p } } ( t ) = \frac { 1 } { | \mathcal { C } | } \sum _ { c \in \mathcal { C } } \mathbf { 1 } \big [ \exists i \leq t : \mathcal { R } _ { c , i } \neq \emptyset \big ]\tag{3}
$$

where $\mathcal { R } _ { c , i }$ is the set of re-implemented targets at turn i of chain c. $C _ { \mathrm { d u p } }$ is cumulative, never decreases along a chain, and is better when lower.

## 4 Experiments

## 4.1 Experiment Setup

Models and Harnesses Since both the backbone model and the harness shape how an agent explores and writes code (Vats and Golev, 2026), we evaluate a grid of model–harness combinations. We use two representative harnesses: mini-SWEagent (SWE-agent Team, 2024), a minimal harness whose only action is a bash command, and OpenCode (Anomaly (formerly SST), 2026), an open-source industrial harness with native read, grep, glob, edit, and write tools. We pair them with four backbone models: GPT-5.6 Terra (OpenAI, 2026), DeepSeek-v4.1-flash (DeepSeek-AI et al., 2026), Qwen3.7-plus (Qwen Team, 2026), and GLM-5.3 (Z.ai, 2026).

Evaluation Protocol All runs follow the selfinvoking protocol: each turn starts a fresh agent session, while the workspace persists across turns and keeps the agent’s own earlier implementations; a failed turn does not end the chain. From turn 2 onward, the requirement lists the public interfaces of the functions the agent implemented in earlier turns but not their source code, and repository reuse targets are never named. All configurations share the same per-turn budget. All metrics are macroaveraged over turns where they are defined.

## 4.2 Main Results

Table 2 reports overall results where we draw several core insight:

Reuse leaves substantial room for improvement No configuration reliably builds on existing code. Even the strongest misses 24.3% of repository targets and 13.6% of its own earlier functions, although the requirement explicitly lists these functions and encourages reusing them; Qwen3.7-plus misses about half of the repository targets and a third of the self targets under both harnesses, leave a substantial room for improvement.

Reuse depends on exploration yet reading is not enough On the repository side, reuse tracks upstream exploration. Configurations that view at least 900 source lines per turn reach 47–61% repository recall, while those viewing fewer than 700 lines reach only 35–42%, and targets that were read are consistently more likely to be reused than targets that were not. On the self side, reading is not the bottleneck: every configuration reads at least 98.9% of its own earlier targets, yet self reuse ranges only from 67.0% to 86.4%. The gap therefore lies in the decision to build on the code that was found, not in finding it.

Reuse aligns with both correctness and redundancy Downstream, reuse first aligns with correctness. Configurations that reuse more repository code generally also resolve more turns and pass more tests: DeepSeek-v4.1-flash leads on both reuse and correctness, while Qwen3.7-plus trails on both. Reuse also aligns with structure. Targets that are bypassed are far more likely than reused ones to have their logic re-implemented, so the two DeepSeek-v4.1-flash configurations, which reuse the most, have clearly the lowest $C _ { \mathrm { d u p } }$ at 17.1% and 21.1%, while all others lie between 33.6% and 46.7%. Across configurations, then, reuse, correctness, and redundancy move together; the following sections show that within a configuration, as turns progress or the memory changes, reuse and redundancy shift while pass rates barely move.

## 4.3 Reuse Across Turns

The aggregate results above average over all turns and hide how reuse evolves as a task chain grows. Since each turn adds the agent’s own code to the workspace, later turns offer more opportunities to reuse and, equally, to bypass existing implementations. Table 3 therefore compares the first and last turn of each task chain.

Reuse declines over turns, especially for self targets As task chains progress, agents build less on existing code. Self reuse drops by 8–24 points from turn 2 to the last turn in every configuration, even though the requirement keeps listing the agent’s earlier functions. The decline is steepest for Qwen3.7- plus, which loses 23–24 points, about twice the 10–13 points lost by DeepSeek-v4.1-flash.

Repository decline is driven by recall On the repository side, the decline in reuse follows the collapse of exploration. Agents read 76–91% of repository targets at turn 1, but repository recall drops sharply once their own earlier code is available, reaching only 16–51% by the last turn. Reuse falls far less than recall, so later-turn repository reuse is increasingly made without reading the target, plausibly from call sites seen in the agent’s own earlier code. Self targets show the opposite pattern: recall stays saturated in every turn, so the decline in self reuse cannot be attributed to exploration.

Table 2: Main results on RepoReuse. Columns are organized around reuse: upstream metrics describe how the agent explores, and downstream metrics describe the outcome. Reuse, recall, correctness, and $C _ { \mathrm { d u p } }$ are in $\% ; C _ { \mathrm { d u p } }$ is averaged over the five cumulative turn-wise rates. Best reuse, recall, and correctness per column in bold.
<table><tr><td></td><td></td><td colspan="2"></td><td colspan="4">Upstream</td><td colspan="3">Downstream</td></tr><tr><td></td><td></td><td colspan="2">Reuse</td><td colspan="2">Recall</td><td colspan="2">Agent Behavior</td><td colspan="2">Correctness</td><td>Redundancy</td></tr><tr><td>Harness</td><td>Model</td><td>repo</td><td>self</td><td>repo</td><td>self</td><td>Files</td><td>Lines</td><td>Resolve</td><td>Pass</td><td> $C _ { \mathrm { d u p } }$ </td></tr><tr><td rowspan="4">mini-SWE-agent</td><td>GPT-5.6 Terra</td><td>62.8</td><td>71.1</td><td>52.7</td><td>99.6</td><td>20.5</td><td>1150</td><td>48.8</td><td>81.2</td><td>42.7</td></tr><tr><td>DeepSeek-v4.1-flash</td><td>68.8</td><td>84.2</td><td>61.4</td><td>99.9</td><td>45.4</td><td>1267</td><td>66.4</td><td>90.8</td><td>21.1</td></tr><tr><td>Qwen3.7-plus</td><td>48.9</td><td>67.7</td><td>36.2</td><td>99.6</td><td>22.0</td><td>680</td><td>27.7</td><td>71.1</td><td>37.1</td></tr><tr><td>GLM-5.3</td><td>55.3</td><td>71.7</td><td>47.2</td><td>99.2</td><td>36.5</td><td>1104</td><td>44.3</td><td>81.2</td><td>39.7</td></tr><tr><td rowspan="4">OpenCode</td><td>GPT-5.6 Terra</td><td>61.6</td><td>67.0</td><td>42.3</td><td>100.0</td><td>18.5</td><td>479</td><td>44.5</td><td>80.1</td><td>36.8</td></tr><tr><td>DeepSeek-v4.1-flash</td><td>75.7</td><td>86.4</td><td>61.4</td><td>98.9</td><td>44.1</td><td>1098</td><td>72.3</td><td>91.6</td><td>17.1</td></tr><tr><td>Qwen3.7-plus</td><td>50.5</td><td>67.8</td><td>35.1</td><td>99.7</td><td>13.5</td><td>576</td><td>30.4</td><td>71.9</td><td>46.7</td></tr><tr><td>GLM-5.3</td><td>56.1</td><td>73.7</td><td>51.5</td><td>99.9</td><td>29.3</td><td>900</td><td>44.3</td><td>81.0</td><td>33.6</td></tr></table>

Table 3: Recall, reuse, and redundancy at the first and last turn (%). Self reuse is defined from turn $2 ; \Delta$ is the change from the first to the last turn. Self reuse declines while the share of task chains containing re-implemented targets grows over turns.
<table><tr><td></td><td></td><td colspan="3"> $\mathrm { r e c a l l _ { r e p o } }$ </td><td colspan="3"> $\mathrm { r e u s e } _ { \mathrm { r e p o } }$ </td><td colspan="3"> $\mathrm { r e u s e } _ { \mathrm { s e l f } }$ </td><td colspan="3"> $C _ { \mathrm { d u p } }$ </td></tr><tr><td>Harness</td><td>Model</td><td>T1</td><td>T5</td><td>∆</td><td>T1</td><td>T5</td><td>Δ</td><td>T2</td><td>T5</td><td>∆</td><td>T1</td><td>T5</td><td> $\Delta$ </td></tr><tr><td>mini-SWE-agent</td><td>GPT-5.6 Terra</td><td>86.3</td><td>38.8</td><td>-47.5</td><td>61.7</td><td>64.3</td><td>+2.6</td><td>77.1</td><td>69.0</td><td>-8.1</td><td>14.7</td><td>60.0</td><td>+45.3</td></tr><tr><td>mini-SWE-agent</td><td>DeepSeek-v4.1-flash</td><td>88.1</td><td>49.8</td><td>-38.3</td><td>74.4</td><td>72.2</td><td>-2.2</td><td>92.2</td><td>79.3</td><td>-12.9</td><td>10.7</td><td>33.3</td><td>+22.6</td></tr><tr><td>mini-SWE-agent</td><td>Qwen3.7-plus</td><td>80.1</td><td>21.7</td><td>-58.4</td><td>52.5</td><td>50.1</td><td>-2.4</td><td>84.2</td><td>61.3</td><td>-22.9</td><td>14.7</td><td>57.3</td><td>+42.6</td></tr><tr><td>mini-SWE-agent</td><td>GLM-5.3</td><td>82.4</td><td>33.6</td><td>-48.8</td><td>63.9</td><td>54.7</td><td>-9.2</td><td>80.9</td><td>66.2</td><td>-14.7</td><td>16.0</td><td>58.7</td><td>+42.7</td></tr><tr><td>OpenCode</td><td>GPT-5.6 Terra</td><td>76.3</td><td>30.7</td><td>-45.6</td><td>62.4</td><td>57.2</td><td>-5.2</td><td>77.1</td><td>65.3</td><td>-11.8</td><td>10.7</td><td>52.0</td><td>+41.3</td></tr><tr><td>OpenCode</td><td>DeepSeek-v4.1-flash</td><td>91.3</td><td>50.7</td><td>-40.6</td><td>81.3</td><td>76.1</td><td>-5.2</td><td>93.6</td><td>83.6</td><td>-10.0</td><td>9.3</td><td>25.3</td><td>+16.0</td></tr><tr><td>OpenCode</td><td>Qwen3.7-plus</td><td>77.4</td><td>15.8</td><td>-61.6</td><td>54.4</td><td>50.8</td><td>-3.6</td><td>82.9</td><td>58.9</td><td>-24.0</td><td>21.3</td><td>69.3</td><td>+48.0</td></tr><tr><td>OpenCode</td><td>GLM-5.3</td><td>86.7</td><td>41.8</td><td>-44.9</td><td>65.6</td><td>64.0</td><td>-1.6</td><td>83.1</td><td>69.5</td><td>-13.6</td><td>13.3</td><td>50.7</td><td>+37.4</td></tr></table>

Reuse decline leads to structural redundancy The decline in reuse has a structural consequence. When agents stop building on existing code, they reimplement it: targets that are bypassed are far more likely than reused ones to have their logic rewritten in the submission. Accordingly, $C _ { \mathrm { d u p } }$ grows steadily over turns in every configuration, from 9– 21% of task chains at turn 1 to 51–69% by the last turn; only DeepSeek-v4.1-flash, which retains the most self reuse, stays at 25–33%. Because each turn inherits the workspace left by earlier turns, these re-implementations persist, leaving parallel versions of the same logic scattered across modules. None of this is visible in pass rates.

## 4.4 Effect of Historical Memory

In multi-turn development, what an agent is told about its earlier work may shape whether it builds on that work (Packer et al., 2024; Li et al., 2025b; Zhang et al., 2024). We therefore vary the form of historical memory in the requirement while keeping everything else fixed, using OpenCode with Qwen3.7-plus. Under all three settings, the agent’s own earlier implementations remain in the workspace; only the requirement differs from turn 2 onward:

• No memory: the requirement contains no information about earlier turns.

• Interface memory (our default): the requirement lists the module, name, signature, return value, and behavior of each function implemented in earlier turns, and encourages reusing them.

• Source memory: the requirement includes the complete source code of every earlier submission, verbatim.

Table 4 reports overall results under each memory setting, and Table 5 compares their first and last turns.

A concise interface helps self reuse; full source code does not To get agents to build on their own earlier code, a short description of what already exists works, while handing them the complete source does not. Listing the interfaces of earlier functions raises self reuse from 30.0% with no memory to

Table 4: Effect of historical memory on OpenCode with Qwen3.7-plus (75 task chains, 375 turns per setting). Reuse, recall, correctness, and $C _ { \mathrm { d u p } }$ are in $\% ;$ agent behavior and $C _ { \mathrm { d u p } }$ are averaged over turns. Best reuse, recall, and correctness per column in bold.
<table><tr><td></td><td colspan="2"></td><td colspan="4">Upstream</td><td colspan="3">Downstream</td></tr><tr><td></td><td colspan="2">Reuse</td><td colspan="2">Recall</td><td colspan="2">Agent Behavior</td><td colspan="2">Correctness</td><td>Redundancy</td></tr><tr><td>Memory</td><td>repo</td><td>self</td><td>repo</td><td>self</td><td>Files</td><td>Lines</td><td>Resolve</td><td>Pass</td><td> $C _ { \mathrm { d u p } }$ </td></tr><tr><td>No memory</td><td>58.9</td><td>30.0</td><td>55.5</td><td>85.5</td><td>27.7</td><td>878</td><td>30.4</td><td>72.4</td><td>65.1</td></tr><tr><td>Interface memory</td><td>50.5</td><td>67.8</td><td>35.1</td><td>99.7</td><td>13.5</td><td>576</td><td>30.4</td><td>71.9</td><td>46.7</td></tr><tr><td>Source memory</td><td>51.3</td><td>29.2</td><td>38.5</td><td>84.4</td><td>16.8</td><td>567</td><td>27.7</td><td>70.2</td><td>70.7</td></tr></table>

Table 5: Effect of historical memory across turns on OpenCode with Qwen3.7-plus (%). Self reuse is defined from turn $2 ; \Delta$ is the change from the first to the last turn.
<table><tr><td></td><td colspan="3"> $\mathrm { r e c a l l _ { r e p o } }$ </td><td colspan="3"> $\mathrm { \ r e u s e _ { r e p o } }$ </td><td colspan="3"> $\mathrm { r e u s e } _ { \mathrm { s e l f } }$ </td><td colspan="3"> $C _ { \mathrm { d u p } }$ </td></tr><tr><td>Memory</td><td>T1</td><td>T5</td><td>∆</td><td>T1</td><td>T5</td><td>∆</td><td>T2</td><td>T5</td><td>∆</td><td>T1</td><td>T5</td><td>∆</td></tr><tr><td>No memory</td><td>77.0</td><td>46.5</td><td>-30.5</td><td>56.4</td><td>61.4</td><td>+5.0</td><td>34.2</td><td>32.1</td><td>-2.1</td><td>22.7</td><td>89.3</td><td>+66.6</td></tr><tr><td>Interface memory</td><td>77.4</td><td>15.8</td><td>-61.6</td><td>54.4</td><td>50.8</td><td>-3.6</td><td>82.9</td><td>58.9</td><td>-24.0</td><td>21.3</td><td>69.3</td><td>+48.0</td></tr><tr><td>Source memory</td><td>77.8</td><td>20.6</td><td>-57.2</td><td>53.4</td><td>50.1</td><td>-3.3</td><td>29.6</td><td>29.1</td><td>-0.5</td><td>17.3</td><td>94.7</td><td>+77.4</td></tr></table>

67.8%, more than doubling it. Supplying the full source of earlier submissions instead leaves self reuse at 29.2%, no better than giving no memory at all, and it stays flat at about 29% in every turn. More information about earlier work is thus not what agents need; what helps is a compact map of which functions exist and where to find them. Interface memory mitigates rather than removes the problem, however: self reuse still declines over turns, from 82.9% to 58.9%. Note also that interface memory explicitly encourages reuse while source memory does not, so the two factors cannot be fully separated.

Memory substitutes for exploration Without memory, agents explore the workspace more broadly, opening 27.7 files per turn, about twice the 13.5 opened under interface memory, which raises repository recall from 35.1% to 55.5% and repository reuse from 50.5% to 58.9%. With interface memory, they rely on the note rather than exploring, and repository recall collapses twice as fast over turns, losing 61.6 points against 30.5 without memory. Memory thus shifts reuse between sources: it strengthens reuse of the agent’s own code but partly at the expense of repository code that the note does not cover.

Memory changes redundancy but not correctness Correctness is nearly identical across the three settings, with resolved rates of 27.7–30.4% and passed rates of 70.2–72.4%, yet their reuse behavior differs sharply. Redundancy is lowest under interface memory, where $C _ { \mathrm { d u p } }$ is 46.7%, and highest under source memory, where it reaches 70.7%: with the full source of earlier turns in context, agents tend to rewrite its logic rather than import it, and by the last turn 94.7% of task chains contain a re-implemented target. As in the main results, a pass-rate-centered evaluation would treat these settings as equivalent, even though they leave very different workspaces behind.

## 5 Conclusion

We asked a question existing evaluation cannot answer—do coding agents reuse existing code or reinvent the wheel? We built RepoReuse to make it measurable: a multi-turn benchmark on real repositories in which requirements are revealed turn by turn, the workspace accumulates across turns, and reuse of both pre-existing repository modules and the agent’s own historical code is audited per turn. The benchmark is produced by a fully automated pipeline and scales readily to new repositories. The answer it gives is sobering: agents progressively stop exploring the repository they work in; they reuse their own history less even when it is fully in view, and handing them the complete source does not help; and the targets they bypass accumulate as cross-turn duplication that pass rates never register. Agents can solve every task and still degrade the codebase and pass rates cannot tell the difference. We hope RepoReuse provides a sustainable instrument for auditing, and ultimately improving, how agents build on existing code.

## Limitations

RepoReuse currently covers five mature, pure-Python libraries and 75 five-turn task chains, evaluated with two harnesses and four backbone models; whether the observed deficiencies generalize to other languages, domains, and longer development horizons remains open. Widening the repository pool and the model–harness grid is work in progress, and the fully automated construction pipeline makes both straightforward to extend.

## References

Altaf Allah Abbassi, Leuson Da Silva, Amin Nikanjam, and Foutse Khomh. 2025. A taxonomy of inefficiencies in llm-generated python code. Preprint, arXiv:2503.06327.

Anomaly (formerly SST). 2026. OpenCode: The open source AI coding agent. https://github.com/ sst/opencode. Accessed: 2026-09.

Anthropic. 2025. Claude code: Ai-assisted coding in real-world codebases. Accessed: 2026-05.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. 2021. Program synthesis with large language models. Preprint, arXiv:2108.07732.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, and 39 others. 2021. Evaluating large language models trained on code. Preprint, arXiv:2107.03374.

DeepSeek-AI, :, Anyi Xu, B. Li, Bangcai Lin, Bing Xue, BingCheng Xian, Bingzheng Xu, Bochao Wu, Bowei Zhang, Boyi Deng, C. C. Yu, Chao Jin, Chaofan Lin, Chen Dong, Chenbing Wang, Chenfan Feng, Chengda Lu, Chenggang Zhao, and 574 others. 2026. Deepseek-v4.1-flash: Pushing the limits of kv cache compression. Preprint, arXiv:2609.19969.

DeepSeek-AI. 2026. Deepseek harness: Everything is a plugin. https://github.com/deepseek-ai/ deepseek-harness.

Yangruibo Ding, Zijian Wang, Wasi Uddin Ahmad, Hantian Ding, Ming Tan, Nihal Jain, Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth, and Bing Xiang. 2023. Crosscodeeval: A diverse and multilingual benchmark for cross-file code completion. Preprint, arXiv:2310.11248.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. 2024. Livecodebench: Holistic and contamination free evaluation of large language models for code. Preprint, arXiv:2403.07974.

Lingxiao Jiang, Ghassan Misherghi, Zhendong Su, and Stephane Glondu. 2007. Deckard: Scalable and accurate tree-based detection of code clones. In 29th International Conference on Software Engineering (ICSE’07), pages 96–105.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. 2024. Swe-bench: Can language models resolve real-world github issues? Preprint, arXiv:2310.06770.

Man Ho Lam, Chaozheng Wang, Hange Liu, Jingyu Xiao, Haau sing Li, Jen tse Huang, Terry Yue Zhuo, and Michael R. Lyu. 2026. Swe-chain: Benchmarking coding agents on chained release-level package upgrades. Preprint, arXiv:2605.14415.

Tue Le, Minh V. T. Thai, Dung Nguyen Manh, Huy Phan Nhat, and Nghi D. Q. Bui. 2026. Swe-evo: Benchmarking coding agents in longhorizon software evolution scenarios. Preprint, arXiv:2512.18470.

Han Li, Zhemin Fang, Rili Feng, Yingqi Zhao, Jiaheng Liu, Pengfei Gao, He Ye, Dayi Lin, Qingwei Lin, Saravan Rajmohan, and Dongmei Zhang. 2026. Loopsbench: From harness engineering to loop engineering in coding agent evaluation. Preprint, arXiv:2608.00267.

Kuan Li, Zhongwang Zhang, Huifeng Yin, Liwen Zhang, Litu Ou, Jialong Wu, Wenbiao Yin, Baixuan Li, Zhengwei Tao, Xinyu Wang, Weizhou Shen, Junkai Zhang, Dingchu Zhang, Xixi Wu, Yong Jiang, Ming Yan, Pengjun Xie, Fei Huang, and Jingren Zhou. 2025a. Websailor: Navigating super-human reasoning for web agent. Preprint, arXiv:2507.02592.

Zhiyu Li, Chenyang Xi, Chunyu Li, Ding Chen, Boyu Chen, Shichao Song, Simin Niu, Hanyu Wang, Jiawei Yang, Chen Tang, Qingchen Yu, Jihao Zhao, Yezhaohui Wang, Peng Liu, Zehao Lin, Pengyuan Wang, Jiahao Huo, Tianyi Chen, Kai Chen, and 20 others. 2025b. Memos: A memory os for ai system. Preprint, arXiv:2507.03724.

Mingwei Liu, Juntao Li, Ying Wang, Xueying Du, Zuoyu Ou, Qiuyuan Chen, Bingxu An, Zhao Wei, Yong Xu, Fangming Zou, Xin Peng, and Yiling Lou. 2025. Code copycat conundrum: Demystifying repetition in llm-based code generation. Preprint, arXiv:2504.12608.

Tianyang Liu, Canwen Xu, and Julian McAuley. 2023. Repobench: Benchmarking repositorylevel code auto-completion systems. Preprint, arXiv:2306.03091.

Dongsheng Ma, Jiayu Li, Zhengren Wang, Yijie Wang, Jiahao Kong, Weijun Zeng, Jutao Xiao, Jie Yang, Wentao Zhang, Bin Wang, and Conghui He. 2026. Citevqa: Benchmarking evidence attribution for trustworthy document intelligence. Preprint, arXiv:2605.12882.

OpenAI. 2025. Introducing codex. OpenAI product announcement.

OpenAI. 2026. Introducing GPT-5.6. https:// openai.com/index/gpt-5-6/. Accessed 2026-09- 23.

Gabriel Orlanski, Devjeet Roy, Alexander Yun, Changho Shin, Alex Gu, Albert Ge, Dyah Adila, Nicholas Roberts, Frederic Sala, and Aws Albarghouthi. 2026. SlopCodeBench: Benchmarking How Coding Agents Degrade Over Long-Horizon Iterative Tasks. Preprint, arXiv:2603.24755.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2024. Memgpt: Towards llms as operating systems. Preprint, arXiv:2310.08560.

Qwen Team. 2026. Qwen3.7-Plus: Multimodal agent intelligence. https://qwen.ai/blog?id=qwen3. 7-plus. Accessed: 2026-09.

Haiyang Shen, Xuanzhong Chen, Wendong Xu, Yun Ma, Liang Chen, and Kuan Li. 2026. Evocode-bench: Evaluating coding agents in multi-turn iterative interactions. Preprint, arXiv:2605.24110.

SWE-agent Team. 2024. mini-SWE-agent: A minimalist agent for SWE-bench and beyond. https: //github.com/SWE-agent/mini-swe-agent. Accessed: 2026-09.

Xiangru Tang, Yuliang Liu, Zefan Cai, Yanjun Shao, Junjie Lu, Yichi Zhang, Zexuan Deng, Helan Hu, Kaikai An, Ruijun Huang, Shuzheng Si, Sheng Chen, Haozhe Zhao, Liang Chen, Yan Wang, Tianyu Liu, Zhiwei Jiang, Baobao Chang, Yin Fang, and 5 others. 2024. Ml-bench: Evaluating large language models and agents for machine learning tasks on repositorylevel code. Preprint, arXiv:2311.09835.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, M. C., Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y. Charles, H. S. Che, Guanduo Chen, Guangyu Chen, Guanzheng Chen, Huarong Chen, Jia Chen, Jianlong Chen, Jun Chen, and 383 others. 2026. Kimi k3: Open frontier intelligence. Preprint, arXiv:2607.24653.

Naman Vats and Oleg Golev. 2026. The scaffold effect in coding agents: Harness choice as a hidden variable in coding-agent evaluation. Preprint, arXiv:2607.22585.

Sizhe Wang, Zhengren Wang, Dongsheng Ma, Yongan Yu, Rui Ling, Zhiyu li, Feiyu Xiong, and Wentao Zhang. 2026. CodeFlowBench: A multi-turn, iterative benchmark for complex code generation. In

Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4369–4402, San Diego, California, United States. Association for Computational Linguistics.

Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighoff, Yizhe Zhang, Binyuan Hui, and 5 others. 2025a. Openhands: An open platform for ai software developers as generalist agents. Preprint, arXiv:2407.16741.

Zhengren Wang, Rui Ling, Chufan Wang, Yongan Yu, Sizhe Wang, Zhiyu Li, Feiyu Xiong, and Wentao Zhang. 2025b. Maintaincoder: Maintainable code generation under dynamic requirements. Preprint, arXiv:2503.24260.

Jialong Wu, Baixuan Li, Runnan Fang, Wenbiao Yin, Liwen Zhang, Zhengwei Tao, Dingchu Zhang, Zekun Xi, Gang Fu, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. 2025. Webdancer: Towards autonomous information seeking agency. Preprint, arXiv:2505.22648.

Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. 2024. Agentless: Demystifying llm-based software engineering agents. Preprint, arXiv:2407.01489.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. 2024. Swe-agent: Agent-computer interfaces enable automated software engineering. Preprint, arXiv:2405.15793.

Z.ai. 2026. GLM-5.3: Frontier coding with emergent cyber capabilities. https://z.ai/blog/glm-5.3. Accessed: 2026-09.

Qintong Zhang, Xinjie Lv, Jialong Wu, Baixuan Li, Zhengwei Tao, Guochen Yan, Huanyao Zhang, Bin Wang, Jiahao Xu, Haitao Mi, and Wentao Zhang. 2026a. Docdancer: Towards agentic document-grounded information seeking. Preprint, arXiv:2601.05163.

Shaoqiu Zhang, Yuhang Wang, Jialiang Liang, Yuling Shi, Wenhao Zeng, Maoquan Wang, Shilin He, Ningyuan Xu, Siyu Ye, Kai Cai, and Xiaodong Gu. 2026b. Swe-explore: Benchmarking how coding agents explore repositories. Preprint, arXiv:2606.07297.

Zeyu Zhang, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Quanyu Dai, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. 2024. A survey on the memory mechanism of large language model based agents. Preprint, arXiv:2404.13501.

## Appendix

## A Experimental Setup Details

## A.1 Evaluation Protocol

Turn Execution Each task chain runs in its own workspace, and its turns are executed in order. Before turn t, the workspace contains the original repository together with every file the agent submitted in turns $1 , \ldots , t - 1$ , each at its required module path; reference solutions never enter the workspace. At each turn, the requirement is rendered according to the memory setting (Appendix A.3) and handed to a fresh agent session with no dialogue history, trajectory, or test feedback from earlier turns. When the agent stops, whether by finishing, exhausting its budget, or failing, we read back whatever file exists at the required path and score it. A turn without a file at that path fails all of its tests. The hidden tests are copied into the workspace only after the agent stops, run from a temporary directory with a 600-second timeout, and removed afterwards. A failed turn does not end the chain: the next turn starts from whatever the agent actually left behind.

Workspace Preparation Before the first turn, we remove information that would let the agent bypass exploration. Git history is deleted, since git log would expose later changes. All test files are deleted file by file, including test\_\*.py, conftest.py, and cached test listings, because repository tests are where the construction pipeline found its evidence and often spell out the conventions a task relies on. Test runners and helpers that the package imports at runtime are kept, and we verify that the package still imports after this step.

Environment Each repository is pinned to its benchmark commit and installed in editable mode in a dedicated conda environment, following the installation recipe of SWE-bench (Jimenez et al., 2024). The same environment is used by all configurations, so that differences between runs come only from the agent.

## A.2 Harness and Model Configuration

mini-SWE-agent The agent interacts with the repository only through bash commands, each executed in a fresh subshell. Its prompt is adapted from the harness’s default template with our delivery instructions (Appendix A.3). We make one addition to the default observation template: each observation reports how many commands have been used, since without it agents tended to spend the whole budget exploring and submit nothing. The agent ends a turn with an explicit submit command. Table 6 lists its budgets; a turn is also stopped after three consecutive malformed responses.

Table 6: Per-turn budgets and settings of the two harnesses.
<table><tr><td colspan="2">mini-SWE-agent</td><td>OpenCode</td></tr><tr><td>Tools</td><td>bash only</td><td>native tools</td></tr><tr><td>Step limit</td><td>120 commands</td><td>none</td></tr><tr><td>Cost limit</td><td>$5</td><td>none</td></tr><tr><td>Per-command timeout</td><td>120s</td><td>harness default</td></tr><tr><td>Wall-time limit</td><td>2,400 s</td><td>2,400 s</td></tr><tr><td>Temperature</td><td>0</td><td>provider default</td></tr></table>

OpenCode We run OpenCode in its headless mode with external plugins disabled and every tool call auto-approved. Each run uses an isolated configuration with a single model provider, so that no user-level settings, plugins, or agents affect the run. OpenCode has no built-in step or cost limit, so each turn is bounded only by the 2,400-second wall time. All models are accessed through the same OpenAI-compatible gateway under both harnesses.

Failures Agent errors, timeouts, and exceeded budgets are not retried. As described above, the turn is scored on whatever file the agent left at the required path, so such failures count against correctness rather than being excluded.

## A.3 Requirement and Memory Formats

Requirement Structure Every requirement follows the same structure: a behavioral description of the task, the exact path and importable module name of the file to create, and for each required function its signature, a description of its behavior, and a precise contract for its return value. From turn 2 onward, a historical note follows, whose form depends on the memory setting. Repository reuse targets are never named in the requirement.

Delivery Instructions Both harnesses wrap the requirement in the same delivery instructions, shown in Appendix B.2 for OpenCode; mini-SWEagent uses the same wording within its own action protocol. These instructions are identical across all memory settings, so a generic encouragement to reuse existing code is present even when the historical note is removed or replaced.

Memory Settings The three memory settings of Section 4.4 differ only in the historical note. No memory removes the note entirely. Interface memory, our default, lists for every earlier turn its module and, for each function, its signature, return contract, and behavior, and encourages reusing them. Source memory removes the interface note and instead embeds the exact source file submitted in each earlier turn, delimited by markers and labeled with its path. The box in Appendix B.2 shows the interface-memory note for turn 2 of a seaborn task, whose first turn implemented three color-mapping functions.

## B Prompt Templates

## B.1 Prompts for RepoReuse Pipeline

## Prompt for Extracting Evidence Packages

You are assembling an \*evidence pack\*: a   
small set of repository symbols that   
a programming task will be built on   
top of. You are walking a call   
graph, one hop at a time, like a   
developer clicking through   
cross-references.   
Repository: {repo\_name}   
{round\_ctx}   
## Symbols collected so far   
{collected}   
## Current position   
{current\_block}   
## Menu -- you may expand ONE of these,   
or stop   
{menu}   
## Your goal   
Collect {lo}-{hi} symbols that could   
plausibly be combined into ONE   
coherent programming task (a small   
script with a few functions). Prefer   
symbols that genuinely belong   
together in a real workflow over   
symbols that are merely adjacent in   
the graph.   
Reply with STRICT JSON only:   
{{"action": "expand" | "stop",   
"choice": <menu number, required iff   
action=expand>,   
"reason": "<one sentence: why this   
symbol belongs with the ones   
collected>",   
"task\_sketch": "<if stop: one sentence   
describing the task these symbols   
support>"}}

## Prompt for Generating Tasks

You are authoring a programming task for a benchmark that evaluates whether coding agents explore and reuse repository internals. Inputs: an evidence pack (repository symbols to build on), a suggested direction, the package layout, and previous-round artifacts.

Write a task a real developer would plausibly need, whose correct implementation GENUINELY requires reading the evidence symbols' source -- i.e., it must exploit internal conventions invisible in signatures (zero representation, coefficient ordering, implicit domain transforms). Any task solvable from docstrings alone is rejected.

## Output six blocks:

1. module\_name: snake\_case; reads as if it always belonged to the package.

2. goal: 2-4 behavioural sentences, as a maintainer would request; never reference earlier rounds discovering what the repo already provides is what is being measured.

3. functions: {nfun} functions (name / signature / returns / behaviour); house-style names, no shared prefix; 5+ lines of real logic each; \`returns\` pins container, element type, and ordering; >=2 functions return plainly repr-comparable values.

4. solution: imports and calls the evidence symbols non-trivially; function names match block 3 verbatim; no module-level helpers.

5. calls: 8-14 single-line calls with predicted reprs, including edge cases; we execute each call and assert the real return value.

6. read\_targets locations a developer MUST read: the evidence symbols plus callees encoding conventions (even if never imported); each \`why\` names the convention, not the task.

## B.2 Prompts for RepoReuse Evaluation

## Delivery Instructions

You are working in the repository checkout at {workdir} (your working directory). The package is installed in editable mode, so \`python -c "import ..."\` picks up any file you write there. The repository's own test suite is not available; write and run your own checks if you need to.

\# Requirement

{requirement}   
# What you must deliver   
A new Python module at exactly   
\`{workdir}/{submit\_to}\` (importable   
as \`{module}\`), defining exactly the   
functions the requirement lists,   
with those names and signatures. The   
requirement may mention modules that   
already exist in the repository --   
find and reuse them where it makes   
sense; do not modify existing files.   
Write the file, check that it imports and   
behaves as required, and fix what   
fails. A round with no file at that   
path scores zero however much you   
learned.

## Interface Memory (turn 2 of a seaborn task)

```markdown
## Note
The repository already contains modules
you implemented in earlier rounds.
You are encouraged to reuse their
public interfaces:
### Round 1 --
`seaborn._core.color_mapping`
`color_lookup`
- Signature: `def color_lookup(data,
values=None, order=None)
Returns: list[tuple[Any, tuple[float,
...]]] -- one entry per resolved
category in seaborn order; NumPy
scalar category labels are converted
to equivalent Python scalars, and
each color tuple contains exactly
three RGB or four RGBA floats
Behaviour: Resolve the categorical
levels and return the standardized
color assigned to each level.
`map_color_values`
- Signature: `def
map_color_values(data, values=None,
order=None)
`describe_palette
- Signature: `def
describe_palette(colors)`
```

## ## Note

## Source Memory (turn 2 of the same task)

```markdown
### Round 1 --
`seaborn._core.color_mapping`
Submit path:
`seaborn/_core/color_mapping.py
- UTF-8 bytes: <length>
SHA-256: `<digest>
<<<CODEFLOW_L5_SELF_SOURCE
bytes=<length>>>>
[complete source of the round-1
submission]
<<<END_CODEFLOW_L5_SELF_SOURCE>>>
```

## C Case Study

Figures 2 and 3 show two consecutive turns of one task chain (mini-SWE-agent with DeepSeekv4.1-flash). At turn 4, the agent implements center\_within\_groups, which centers a numeric column within seaborn’s categorical groups, and passes all tests. At turn 5, the requirement asks for per-category summaries of centered values and explicitly lists the turn-4 interface. The agent reads the turn-4 source in full (recall = 1.0), yet never imports it. Instead, it rewrites the dtype coercion, group masking, categorical ordering, and meancentering logic from scratch in 262 lines, where the ground truth solution reaches the same behavior in 68 lines by calling the turn-4 function and aggregating its output. The submission passes 10/10 tests: nothing in the pass rate registers that the workspace now hosts two parallel implementations of the same centering semantics.

![](images/00bc874b85871817c152b402c26239f5313a814ba544fa9001aa47f8c71a9e41.jpg)  
Figure 2: Case study, turn 4. Left: the turn-4 requirement. Right: the agent’s center\_within\_groups, centering a numeric column within seaborn’s categorical groups.

![](images/63a0816b8307a25da0b86fec8093865cb452a427a01678cc495acf61822bc792.jpg)  
Figure 3: Case study, turn 5. Left: the turn-5 requirement, which explicitly lists the turn-4 interface. Right: the agent reads that code (recall = 1.0) yet rewrites the same coercion, grouping, and centering logic from scratch instead of importing it—passing 10/10 tests while duplicating its own earlier implementation.