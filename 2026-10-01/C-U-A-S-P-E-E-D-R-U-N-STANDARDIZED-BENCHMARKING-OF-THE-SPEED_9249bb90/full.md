# C U A-S P E E D R U N: STANDARDIZED BENCHMARKING OF THE SPEED OF COMPUTER-USE AGENTS

Pranjal Aggarwal<sup>∗</sup> Lawrence Keunho Jang<sup>∗</sup> Sean Welleck Daniel Fried Ruslan Salakhutdinov Jing Yu Koh

Carnegie Mellon University

{pranjala,ljang,jingyuk}@cs.cmu.edu

## ABSTRACT

Computer use agents (CUAs), which use graphical user interfaces (GUIs) to complete tasks on a computer, have recently surpassed human performance on many standard benchmarks, including difficult long-horizon tasks. Their capabilities are undoubtedly impressive, however, a key barrier to the widespread adoption and deployment of CUAs remains their speed and cost. Progress towards faster yet capable CUAs requires reliable evaluation of their speed, but many CUA benchmarks currently face a reproducibility crisis. Benchmarks are based on complex infrastructure with varying machine and container configurations that confound the evaluation of the execution speed of CUAs. Towards addressing this gap, we propose cua-speedrun, which introduces standardized infrastructure and task sets, with a focus on evaluating the speed and efficiency of CUAs. cua-speedrun uses a uniform virtual machine setup and execution pipeline, along with a common agent interface that enables single-agent implementations to operate seamlessly across different benchmarks. Across four different CUA benchmarks, we evaluate how reasoning effort, agent harnesses, and environment latency affect performance, speed, and cost. We find no single model family is optimal for all three; none of the open-weight models are on the frontier, and also, unintuitively, for some models increasing the reasoning effort can speed up task completion, while faster environment input-output can slow down overall task completion time. We also demonstrate that we can effectively reduce the evaluation task set of most CUA benchmarks without degrading overall statistical power, allowing for more efficient benchmarking and comparison. We believe cua-speedrun will enable structured progress towards fast, efficient CUAs, unlocking new real-world use cases and applications. All code, infrastructure, and analysis are available at cuaspeedrun.com.

## 1 INTRODUCTION

Computer use agents (CUAs) are language model agents that interact with graphical user interfaces (GUIs) to achieve a user-specified goal. CUAs operate over the same user interface as human users, enabling them to potentially automate a much wider variety of tasks without software-specific APIs.

When measured in terms of success alone, recent model releases have pushed the capabilities of CUAs past human baselines. The best frontier models today achieve success metrics above human performance on popular CUA benchmarks: Qwen3.8-Max, Claude Mythos 5, GPT-5.5, and Gemini 3.6 Flash report successes ranging from 78.7% to 86.1% on OSWorld-Verified, far above the 72.4% human reference score (Xie et al., 2024). Across long-horizon benchmarks (Aggarwal et al., 2026; Yuan et al., 2026; Jang et al., 2026b), frontier models have also demonstrated the ability to achieve strong performance on complex, long-horizon tasks.

Despite their strong performance, a major bottleneck for widespread adoption of current CUAs in practical settings is their speed and cost of deployment (Abhyankar et al., 2025). In particular, we believe that research on the speed and cost efficiency of CUAs has trailed capabilities research, in part due to the difficulty of fairly benchmarking speed and cost of different models in a consistent manner. Specifically, CUAs require complex infrastructure to run and evaluate. Different benchmarks use varying execution environments, ranging from local virtual machines to cloud-parallelized desktops (Xie et al., 2024; Bonatti et al., 2025). There are drastic differences in deployment setup, API interfaces, hyperparameter settings, agent harnesses, and even benchmark implementations across the community.

![](images/ec3189e0ed8f87f16bbe35ca5c0ba1cfdde8b9a7eec361728accb3b2738ef4b6.jpg)  
Figure 1: cua-speedrun overview. cua-speedrun is a standardized platform for benchmarking the performance, speed, and cost of computer-use agents. It combines representative task selection with standardized infrastructure on Modal, enabling consistent comparisons across models, agent harnesses, and reasoning settings. Our findings reveal that more reasoning can reduce both task time and model cost for CUAs, while faster environment I/O can make agents slower.

Towards addressing the problem of consistent benchmarking of CUA wall-time, and igniting research on improving the speed and deployment practicality of CUAs, we propose cua-speedrun: a standardized platform that can use existing benchmarks to measure the speed and cost efficiency of CUAs (Fig. 1). There are several challenges that cua-speedrun addresses, towards standardizing the evaluation of CUAs across different model families (in particular on wall-time metrics):

• Uniform CUA evaluation infrastructure. We standardize evaluations through a serverless cloud-based provider<sup>1</sup>, where all evaluations run on the exact same execution pipeline and the same virtual machine setup.

• Common agent interfaces. Existing CUA implementations are varied and do not always follow the optimal scaffolds. We implement a common agent interface for executing actions and interacting with the environment while allowing for portable agent harnesses and implementations. This allows a single agent implementation to work across different benchmarks. Our analysis also measures time differences due to factors such as the latency of closed-source models, which can vary by provider.

• A maximally representative subset for fast and comparable evaluations. CUA benchmarks take a long time to run. We propose an energy minimization framework that selects a minimal representative subset of tasks within existing CUA benchmarks. This allows us to minimize end-to-end evaluation time while preserving the signal of the results, and maintain the relative ordering of agent performance. Our approach reduces evaluation time on OSWorld by 84.9% while maintaining high correlation with evals run on the full set.

We implement these features on cua-speedrun, and benchmark current frontier CUA models on several widely adopted CUA benchmarks, such as OSWorld-Verified (Xie et al., 2024) and OSWorld 2.0 (Yuan et al., 2026). In addition, we also perform evaluations and analysis on CUA-World (Aggarwal et al., 2026) and MyPCBench (Jang et al., 2026a), which test for long-horizon and personalized computer use, respectively. cua-speedrun allows us to maintain a live Pareto ranking of the most cost-effective, fastest, and most capable models. Through an extensive analysis of multiple cua-speedrun runs, we identify several interesting and applicable insights. Success alone hides large differences in speed: on OSWorld, the open-weight Kimi K3 (max reasoning) matches GPT-6 Astra (high), but takes 4.4× longer per task, and none of the open-weight models we evaluate (Kimi K3, MiniMax M3, GLM-5V Turbo) lie on the performance–time or performance–cost Pareto frontier of any benchmarks. Speed also interacts with reasoning in unintuitive ways: raising reasoning effort can improve benchmark scores while reducing task completion time, since the added reasoning avoids repeated unsuccessful actions. Using a fast I/O system can also slow agents, as the models have not been trained to interact with desktops in different I/O speed configurations.

Our findings highlight the importance of turning our attention beyond performance and success rates. cua-speedrun maintains a standardized evaluation of the performance and efficiency of CUAs, two dimensions that we believe are essential future research directions. We hope that the public release of cua-speedrun encourages the research community to work on making CUAs faster, cheaper, and more efficient, paving the way towards more widespread real-world adoption.

## 2 METHOD

In this section, we describe our approach to address the aforementioned challenges with benchmarking CUAs. We base our techniques to improve CUA benchmarking on standardizing measurements and infrastructure (Sec. 2.1), selecting representative tasks instead of running entire benchmarks (Sec. 2.2), and building infrastructure that executes and times every agent consistently (Sec. 2.3).

## 2.1 STANDARDIZED COMPUTER-USE EVALUATION

Standardizing computer-use evaluations requires invariance across four different dimensions: (1) the agent, driven by a large-language-model; (2) the benchmark, which determines what an agent must accomplish; (3) the environment infrastructure, which is responsible for managing VMs, the environment lifecycle, and action/observation contracts; and (4) the agent loop, which determines how the model interacts with the infrastructure. Many computer-use evaluations often make different choices for these four components, making it difficult to study them independently. For example, adopting a new model may introduce a benchmark-specific action loop, or adopting a new benchmark requires changes to the agent or infrastructure. The goal of our setup is to ensure the decoupling of benchmarks from infrastructure from agent design, in order to ensure fair and independent evaluation across these four dimensions.

Problem setup. Following prior computer-use benchmarks, each task specifies an initial desktop state, a natural-language instruction, and a verifier, either a programmatic check or an LLM judge, that scores the agent’s trajectory and final state. Agents observe screenshots of the desktop and act through keyboard and mouse actions; Appendix J.1 gives the formal definition. Comparisons between agents hold the benchmark and infrastructure fixed, and similarly comparisons across benchmarks fix agent and infrastructure, so each change behaves as a clean ablation.

Measuring speed. For each task, our infrastructure starts measuring time when the instruction is given to the agent and stops when the agent terminates or reaches the task limit (steps taken, wall-time, or both). Environment provisioning, task setup, agent initialization, and verification occur outside of this interval. The infrastructure records the total task time and additionally separates it into the time spent executing environment operations and the time spent executing agent operations. Thus, the agent time can be fully attributed to the “speed” of the agent, regardless of any time needed to set up CUA evaluations. Additionally, we also log the total cost of model calls. In addition, infra failures are retried, while agent failures end the trajectory, which is then scored.

## 2.2 SELECTING A REPRESENTATIVE SET

Choosing a representative evaluation set. A full computer-use evaluation is often expensive because of long trajectories adding to API/GPU cost, and often requires overhead for managing virtual machines (VMs), adding substantial evaluation cost. For example, GPT-5.4 costs approximately \$4000 on the CUA-World-Long benchmark (Aggarwal et al., 2026). We therefore ask the question: can we reduce the size of the benchmark needed for eval, while still capturing the success rate of the agents we evaluate on the full set? We want to choose a subset that keeps the individual agents’ scores similar to their full benchmark scores and preserves the ordering of agents’ performance.

Selection criteria based on success. Our goal is to automatically select a subset K of the benchmark that is representative of the full benchmark. To ensure that the selected subset generalizes beyond the agents used to construct it, we evaluate the selection procedure using leave-one-agent-out validation. For each agent $m ,$ , we construct a subset using the results of only the other agents (M-1) and then compare the held-out agent’s score on the selected subset with its score on the full benchmark.

We found through multiple iterations of leave-one-model-out evaluation that the best method to select tasks was to minimize the energy distance between the distributions of agent scores on the full benchmark and subset (Szekely & Rizzo, 2013). Specifically, let ´ $C _ { m i } \in [ 0 , 1 ]$ be the partial score of agent m on task $i ,$ and let $B _ { m i } = \mathbf { 1 } [ C _ { m i } = 1 ]$ denote exact completion. We represent task i by

$$
z _ { i } = ( C _ { 1 i } , \hdots , C _ { M i } , B _ { 1 i } , \hdots , B _ { M i } ) \in \mathbb { R } ^ { 2 M } .\tag{1}
$$

Thus, two tasks are considered similar when agents exhibit similar patterns of partial and exact completion on them. Let $D _ { i j } = \| z _ { i } - z _ { j } \|$ and let D be the mean distance between distinct tasks. For a subset $S _ { K }$ of K tasks, we minimize

$$
\mathcal { E } ( S _ { K } ) = \frac { 1 } { \overline { { D } } } \Bigg ( \frac { 2 } { K N } \sum _ { i \in S _ { K } } \sum _ { j = 1 } ^ { N } D _ { i j } - \frac { 1 } { K ^ { 2 } } \sum _ { i , i ^ { \prime } \in S _ { K } } D _ { i i ^ { \prime } } - \frac { 1 } { N ^ { 2 } } \sum _ { j = 1 } ^ { N } \sum _ { j ^ { \prime } = 1 } ^ { N } D _ { j j ^ { \prime } } \Bigg ) .\tag{2}
$$

This objective favors subsets whose tasks represent the performance patterns present in the full benchmark while avoiding redundant tasks with nearly identical patterns. The final term depends only on the full benchmark and is therefore constant during subset selection. We approximately minimize this objective by starting with 100 task subsets and replacing one selected task with an unselected task whenever this reduces the objective. Thus, we retain the subset with the lowest objective value across the 100 runs. During leave-one-agent-out evaluation, this procedure is repeated after removing the held-out agent’s results. After determining the desired subset size, we construct the final deployed subset once using all available agents.

What subset is sufficient to preserve model rankings? The previous selection criteria can find the representative set for a given value of K. Next, we ask for the smallest subset of K tasks that consistently preserves the ranking of agents by performance. Let $\rho _ { q } ( K )$ be the Spearman correlation between leave-one-agent-out estimates and full benchmark results for evaluation quantity $q .$ We choose

$$
K ^ { * } = \operatorname* { m i n } \left\{ K : \operatorname* { m i n } _ { q } \operatorname* { m i n } _ { k \in \{ K - 1 , K , K + 1 \} } \rho _ { q } ( k ) \geq 0 . 9 5 \right\} .\tag{3}
$$

We require the correlation threshold to hold for (K-1), (K), and (K+1) to prevent us from selecting a value of (K) that performs well only by chance. This criterion selects 50 of 295 OSWorld tasks, 52 of 63 OSWorld2 tasks, 26 of 143 CUA-World tasks, and 38 of 184 MyPCBench tasks (Table 13), which form the cua-speedrun evaluation sets. We also evaluate how well the subsets preserve full-benchmark performance and model comparisons in our experiments (Appendix G).

## 2.3 INFRASTRUCTURE

How do we host our infrastructure? We adapt the virtual machine runtime from Gym-Anything (Aggarwal et al., 2026) to run task environments natively in Modal sandboxes. Our infrastructure manages separate sandboxes for agents and environments, ensuring isolation, while Modal sandboxes provide a standardized hosted runtime so that differences in users’ infrastructure do not affect evaluation results. For self-hosted open-weight models, we use vLLM (Kwon et al., 2023) inference servers on fixed L40S GPUs. Keeping the inference servers and GPUs fixed allows consistency in evaluation. We develop a Python library to handle GPU scheduling, environment allocation, and parallelization, with further details in Appendix B.

Action and observation modalities. Following standard practice in computer-use agents (Xie et al., 2024; Qin et al., 2025), the action space consists of keyboard actions (e.g., typing text or pressing Ctrl+C) and mouse actions (e.g., clicking at coordinates (x, y) or double-clicking). Agent observations are RGB screenshots of the desktop at a resolution of $\mathrm { 1 9 2 0 \times 1 0 8 0 }$

To ensure that these interfaces behave consistently across VM runtimes, we developed CUA-AutoDebug, an end-to-end test suite in which a controlled application records the keyboard and mouse inputs it receives, compares them with the intended inputs for a large catalog of actions, and checks that screenshots match the application state. These tests uncovered several input errors in widely used CUA tools, which we correct in our infrastructure (Appendix I).

FastCUA: Developing a new fast I/O system for CUAs. Standard computer-use runtimes typically take roughly 2–3 seconds from issuing an action to receiving an observation. This latency is dominated by a variety of elements, such as action execution, waiting for the application to respond, and networking. The delay incidentally gives the application time to process the action (e.g., open a menu), allowing the agent to observe its effect. We therefore follow this design for our main evaluations. However, we develop a fast I/O mode, FastCUA, by optimizing networking, action execution, and image processing latencies (see Appendix B). FastCUA reduces action-to-observation latency to 2–28 ms, an improvement of more than an order of magnitude over typical runtimes. At this speed, a screenshot can be captured before the application has updated its UI in response to an action. We evaluate this setting to test whether lower infrastructure latency reduces overall wall time.

Implementing agents on cua-speedrun. Since we have standardized infrastructure, new agents can be implemented with a single Python file that contains the agent loop. We also provide standard templates for various popular agents, which users can tweak directly. We provide options for both hosted evaluations and local evaluations through a single CLI call. See Appendix C for more details.

## 3 RESULTS

## 3.1 EXPERIMENTAL SETUP

We evaluate 56 agent configurations on OSWorld and 21 on OSWorld2 using the representative task sets from Sec. 2.2. These configurations cover frontier open-weight and proprietary models, each run through its public reference agent implementation or native computer-use API at selected reasoning-effort settings, with the tasks and infrastructure held fixed (Appendix J). We validate all agent designs for correctness through manual inspection and CUA-AutoDebug on representative tasks. For selected models, we also compare different agent harness designs, standard and fast I/O, and single versus batched tool calls per model response. Repeated evaluations are stable: across five seeds, GPT-6 Astra (low) scores 90.8 ± 1.8% on OSWorld with a mean task time of $8 9 . 5 \pm 2 . 2 \mathrm { s }$ (Appendix F).

Metrics. For each configuration, we report the mean verifier score, which retains partial credit, together with the mean task time (Sec. 2.1) and mean model cost per task. We report the performance– time and performance–cost Pareto frontiers, as well as the joint frontier over all three. Interaction turns, generated tokens, and exact completion are defined in Appendix D.

## 3.2 ANALYSIS

No single model family dominates the frontier. Fig. 2 shows the performance–time and performance–cost Pareto frontiers on both benchmarks. On OSWorld, the high-performance end of the time frontier extends from Claude Opus 5 (low), with a score of 87.6% in 86 s, through GPT-6 Astra (low), with 90.8% in 90 s, to Astra (xhigh), with 91.6% in 127 s. Gemini 3.8 Flash (low) matches the highest score and takes almost the same time as GPT-6 Astra, but costs \$0.11 rather than \$0.71 per task. At the lower-cost end, GPT-5.6 Luna (low, direct API) costs about \$0.01 per task but scores 61.8%. Thus, cua-speedrun enables comparison across various dimensions such as speed, cost and performance, rather than reducing the comparison to a single model ranking.

(a) OSWorld: time  
![](images/0c8a0770cf0505e504d08b0ed881634e52c0d0e4820e27f03c6b8cd0425b8b79.jpg)

(b) OSWorld: cost  
![](images/9397fbb00cf52ad0f381e7c884417f4972b2af533ef51b2e673c7081c2831c43.jpg)  
Model GPT-6 Astra Gemini 3.8 Flash Claude Opus 5 Claude Sonnet 5 GPT-5.6 Sol GPT-5.6 Luna Kimi K3 Muse Spark 1.3 GLM-5V Turbo Yutori n2

(c) OSWorld2: time  
![](images/9e909e1bdfa59d80aecfa38cc90512691f752607b159b9d2ec8d95031f3673be.jpg)  
Mean task time (s)

(d) OSWorld2: cost  
![](images/96996e375acb6f6d67c01441b8467c7a5311a89e0cf057ff461247dec932f238.jpg)  
Reasoning effort (darker within each model color) None High Minimal/off xhigh Low Max Medium On

Mean model cost (\$/task)  
Figure 2: Different agents define the speed and cost frontiers. Mean score vs. mean task time and model cost on OSWorld (top, 54 configs) and OSWorld2 (bottom, 19 configs), excluding MiniMax M3. Dashed lines connect the observed Pareto-frontier points, which are circled. Colors and marker shapes identify models, and darker shades indicate increased reasoning effort. The plots compare complete agent configs. Costs use recorded charges or token usage at the applicable API prices. Astra (low) uses five-seed means for each I/O setting. Full configs and plots in Appendix E.  
(a) Task time  
![](images/70ed8a0951f565e67fc656be9ea34ade9f67024cb0210dd70d78dc41bcbfe603.jpg)

(b) Generated tokens  
![](images/b62b985ba5e2c5da8def7b8c6c25a93bcf190dfc7974dc6bc8ba6cd19fe9f480.jpg)

(c) Model calls  
![](images/d4bae31111ae917b1cff2d62a8ba3c6143f9d4b7c26166b3fe8dfcb4d4c9b9e1.jpg)  
Figure 3: Less reasoning can make an agent slower. Gemini 3 Flash Preview at low, medium, and high effort on OSWorld. The panels show mean task time, generated tokens including reasoning, and model calls across all 50 tasks. Scores are 33.6%, 57.6%, and 59.6%, respectively. Medium and high effort generate more tokens than low effort but require fewer calls and complete tasks faster.

The Pareto frontier changes across benchmarks. Fig. 2 also shows that, unlike the OSWorld time frontier, the OSWorld2 time frontier is formed entirely by GPT-6 Astra configurations. Astra (high) achieves 75.0% in 914 s, while xhigh reaches 76.9% in 1,237 s, achieving an additional 1.9 percentage points with 35% more time. However, the cost frontier includes other models such as Gemini 3.8 Flash (medium), which achieves 59.9% at \$2.91 per task, compared to Astra (high) at \$7.39. Muse Spark 1.3 (minimal) extends the frontier to lower cost, with a score of 34.0% at an estimated \$0.39. Surprisingly, there is no open model on either the cost or the time frontier for either OSWorld or OSWorld2. Tracking these frontiers allows us to highlight the trade-off explicitly. The preferred agent depends on both the benchmark and the performance required within a time or cost budget, and there is no universal model family that dominates across different benchmarks.

Less reasoning can sometimes make agents slower. Fig. 3 compares Gemini 3 Flash Preview at different reasoning settings on OSWorld, keeping the model and harness fixed. Moving from low to medium effort improves the score from 33.6% to 57.6% while reducing mean task time from 492 s to 268 s. We diagnose this as follows: at lower reasoning effort, while the model generates fewer tokens (1,598 vs. 4,125 at medium effort), it takes substantially more steps (56.0 vs. 32.5), since most of its steps are suboptimal. For instance, the trajectories show repeated unsuccessful actions at low effort, such as repeatedly trying to enter spreadsheet text through key combinations instead of text entry. Counterintuitively, higher reasoning effort can therefore shorten the interaction enough to reduce total task time, even when it increases the number of generated tokens per step.

![](images/885d05919a7fe757f8e3350cb7f3ec5f2a1bc3bf79b5f584c86301d9901fdf86.jpg)  
Mean task time (s)

![](images/2f197dfcfc7359db052b477b0f192003c22ed276b16b4c1772a3aaf098a27cd3.jpg)  
Mean task time (s)

Figure 4: The useful reasoning setting changes across benchmarks. Gemini 3.8 Flash at low, medium, and high effort; labels also report mean cost per task. Each curve keeps the model, agent harness, and execution settings fixed. Additional reasoning brings no gain on OSWorld, whereas moving from low to medium substantially improves OSWorld2 performance.  
![](images/c81c1bf5040e970a1db377214b065ceac2c1ef8507a2dba9d3e956b9340f3f16.jpg)

![](images/3ea4b67166c2a3069c051f8291e04a6fddd074015e491c6f8ddd3b0977fc3450.jpg)

![](images/415445e486b6a267143b014cfeb60647d26326b548f558706697da75d9e9248a.jpg)  
Figure 5: Faster I/O can be offset by more waiting and agent interaction. GPT-6 Astra (low), with normal and fast I/O. Panel (a) separates agent time, agent-requested waits, and remaining environment processing, averaged over five seeds per setting; error bars show the standard deviation of mean task time across seeds. Panel (b) shows two consecutive screenshots after opening Save As: the dialog is absent in the first and visible in the second, requested 5.8 s later with no intervening action.

More reasoning can increase time and cost without improving performance. Fig. 4 compares reasoning settings within Gemini 3.8 Flash on each benchmark. On OSWorld, low and medium reasoning both score 91.6%, but medium increases mean task time from 127 s to 250 s and approximately doubles cost. However, the trend changes on OSWorld2: moving from low to medium improves performance from 35.3% to 59.9%. But increasing effort further to high adds only 0.2 percentage points while increasing time by 49% and cost by 72%. GPT-6 Astra shows a similar dependence on the benchmark: on OSWorld, medium effort is within 2 points of xhigh at 29% less time and 15% less cost, whereas on OSWorld2, high effort gains 7.8 points over medium for 10% more time. The useful reasoning setting therefore depends on the tasks as well as the model: additional reasoning can be valuable on one (harder) benchmark and unnecessary on another (simpler) benchmark.

Faster infra can make the agent slower. Fig. 5 compares GPT-6 Astra (low) with the normal input-output mode and our faster input-output mode, FastCUA (Sec. 2.3 and Appendix B.3) across five seeds per setting. Interestingly, while the mean environment processing time falls from 12.68 s to 0.34 s per task, total task time increases from 89.5 s to 99.0 s. We also find that the number of steps per trajectory increases in FastCUA. This is because fast I/O can return screenshots before an application has updated its interface. For example, after opening Save As, the agent receives a screenshot without the dialog and requests another screenshot 5.8 s later, which shows it open.

Although theoretically a model could account for such faster I/O by using appropriate wait times (e.g., 100 ms in this case) for the application to update, we find that current models are not able to optimally utilize this, resulting in much higher times.

The agent harness plays an important role in the speed–performance trade-off. Fig. 17 compares GPT-5.6 Luna through the direct-API agent and the Codex harness on OSWorld. At low effort, Codex improves the score from 61.8% to 75.6%, but increases mean task time from 74 s to 145 s. The same model and reasoning setting therefore produce different performance and efficiency when used through different harnesses. At medium and high effort, the direct-API agent is instead both more accurate and faster (75.6% in 119 s vs. 67.6% in 189 s at medium; 81.6% in 161 s vs. 79.6% in 325 s at high), so neither harness is uniformly better.

Batching actions reduces task time. Each step costs a screenshot, a model call, and environment latency, so agents that issue several actions per response (which we execute sequentially) finish sooner: GPT-6 Astra (low) completes OSWorld tasks in 5.9 action batches on average (Table 11). Restricting GPT-6 Astra (xhigh) to one action per response keeps its score at 91.6% but raises mean task time from 127 s to 183 s. The benefit depends on how well a model plans multi-action sequences, and batching alone does not make an agent fast (Appendix K.1).

Fast agents generate fewer tokens, not tokens faster. Claude Opus 5 (low) generates 1,795 tokens per task compared with 1,579 for GPT-6 Astra (low), while scoring 87.6% vs. 90.8%. At a common token rate (Fig. 10), it falls behind the Astra configurations, which remain the fastest high-scoring agents. Similarly, Claude Sonnet 5 (high) takes 141 s for the score that Astra (medium) reaches in 90 s, generating 3.5× more tokens, and Gemini 3.8 Flash (high) generates 5.6× more tokens than at low effort and finishes 2.7× later.

Extending to other benchmarks. We additionally evaluate GPT-6 Astra with Codex at four reasoning settings on MyPCBench (Jang et al., 2026a) and CUA-World (Aggarwal et al., 2026), which test personalized and long-horizon computer use and grade trajectories with vision-language models rather than programmatic checks. Astra exceeds 90% rubric scores on both (Appendix E.5), highlighting the need for more challenging task sets.

## 3.3 PRACTICAL SUGGESTIONS FOR BUILDING CUAS

Based on the analysis above, we make the following suggestions for building efficient CUAs:

• Choose a frontier configuration for the required performance and budget. No single configuration is best on every metric, and the frontier shifts across benchmarks (Figs. 2 and 6), so candidates should be measured on tasks that resemble the target workload.

• Tune reasoning effort rather than setting it to an extreme. Too little reasoning can lengthen trajectories, while too much can add time and cost without improving performance (Figs. 3 and 4). Start with low or medium effort and raise it only for a measured gain.

• Batch actions when intermediate screenshots are unnecessary. Fewer steps mean fewer screenshots, model calls, and environment round trips.

• Reduce generated tokens and steps rather than maximizing token rate or I/O speed. Faster generation or I/O helps only if the agent does not spend it on longer outputs or additional steps (Fig. 5).

• Optimize the harness alongside the model. The same model and reasoning setting can land at different points on the frontier depending on its harness (Fig. 17).

## 4 RELATED WORK

Computer-use agent benchmarks. The first computer-use agent benchmarks used synthetic interfaces (Liu et al., 2018; Yao et al., 2022). Follow-up evaluations moved to self-hosted realistic website analogs (Zhou et al., 2024; Koh et al., 2024; Drouin et al., 2024), static datasets of real live website tasks (Deng et al., 2023), and live-Internet evaluations (He et al., 2024; Xue et al.,

![](images/5ef32ef16b94829c16b8ecf70d997a063cad9887c30970b61d342a4da459d17d.jpg)  
Time / task (s)  
Figure 6: Choosing an agent by performance, time, and cost. Joint Pareto frontier on OSWorld; the OSWorld2 frontier is in Appendix E, Fig. 13. Circled points are the configurations on the Pareto frontier. Time and cost axes are logarithmic.

2025). Live, dynamic desktop benchmarks such as OSWorld (Xie et al., 2024) evaluate agents on real Linux desktops using programmatic verifiers, with WindowsAgentArena (Bonatti et al., 2025), macOSWorld (Yang et al., 2025), iOSWorld (Jang et al., 2026c), and AndroidWorld (Rawles et al., 2025) extending coverage to other platforms and professional workflows. Recent benchmarks stress long horizons and scale, such as OSWorld 2.0 (Yuan et al., 2026), CUA-World from Gym-Anything (Aggarwal et al., 2026), and Odysseys (Jang et al., 2026b). Another emphasis has become personal assistant evaluations, including OpenClaw-style works such as ClawBench (Zhang et al., 2026b), WildClawBench (Ding et al., 2026), and MyPCBench (Jang et al., 2026a), which evaluate personalized computer use and digital assistants. OSWorld-Human (Abhyankar et al., 2025) measures agent latency on OSWorld and compares step counts against human reference trajectories, finding that agents take longer than humans.

Computer-use agent models. Frontier model releases have increasingly prioritized computer use (OpenAI, 2026; Anthropic, 2026; Google, 2026; Qwen Team, 2026), with several reporting above human scores on OSWorld-Verified. Models such as Kimi K3 (Kimi Team, 2026), GLM-5 (GLM-5 Team, 2026), and Muse Spark (Menghini et al., 2026) have also included agentic workflows in their tech releases. Alongside these large general models, a line of work trains specialized computer-use models, including UI-TARS (Qin et al., 2025), MolmoWeb (Gupta et al., 2026), Fara-1.5 (Awadallah et al., 2026), ScaleCUA (Lv et al., 2026), and Qwen-CUA (Lu et al., 2026), often in verifiable environments generated at scale (Wang et al., 2026; Aggarwal et al., 2026).

Reproducible and efficient evaluation. Agent evaluations are noisy (Kapoor et al., 2024; 2026), as Xue et al. (2025) showed that reported web-agent success is inflated by evaluation issues, and Sahu & Pandey (2026) show that single-run claims are unreliable because outcomes vary substantially across data draws and run non-determinism. A separate line of work reduces the evaluation cost by selecting a small subset of a benchmark that predicts full-benchmark scores, via item response theory (Polo et al., 2024; Kipnis et al., 2025; Zhou et al., 2026), sparse optimization (Zhang et al., 2026a), or human validation (SWE-bench Verified; Chowdhury et al., 2024). PACE (Song et al., 2026) predicts agentic benchmark scores from cheap, non-agentic evaluations.

## 5 CONCLUSION

In this work, we introduced cua-speedrun, a framework for evaluating the performance, speed, and cost of computer-use agents under standardized infrastructure. Our evaluations on OSWorld and OSWorld2 identify the performance–time and performance–cost Pareto frontiers and show how these change across benchmarks. We further find that more reasoning can reduce task time by avoiding repeated unsuccessful actions, while faster I/O can increase it through additional interaction. These results highlight the need to study reasoning and interaction together when developing faster agents. We hope cua-speedrun enables the community to improve both capability and efficiency, making computer-use agents more practical to deploy.

## ACKNOWLEDGEMENTS

We thank Modal for their generous support in cloud compute credits to build cua-speedrun. We thank Eunsu Kim, Naveen Raman, Mareks Woodside, Seungone Kim, Zeyu Zheng, and others for feedback and helpful discussions. Jing Yu Koh is supported by a Jane Street Graduate Research Fellowship. Lawrence Jang is supported by a Susquehanna International Group PhD Fellowship. Pranjal Aggarwal is supported by a SoftBank Group-Arm Fellowship. This work is partially supported by the National Science Foundation under Grant No. DMS-2502281, and a grant from Amazon on Useful Chain of Thought Reasoning.

## AI USE STATEMENT

We used AI assistance for general software coding, making figures, polishing writing across the manuscript, and verifying related work.

## REFERENCES

Reyna Abhyankar, Qi Qi, and Yiying Zhang. OSWorld-Human: Benchmarking the efficiency of computer-use agents. arXiv preprint arXiv:2506.16042, 2025.

Pranjal Aggarwal, Graham Neubig, and Sean Welleck. Gym-Anything: Turn any software into an agent environment. arXiv preprint arXiv:2604.06126, 2026.

Anthropic. Claude Fable 5 and Claude Mythos 5. https://www.anthropic.com/news/ claude-fable-5-mythos-5, 2026.

Ahmed Awadallah, Sahil Gupta, Yash Lara, Yadong Lu, Hussein Mozannar, Akshay Nambi, Zach Nussbaum, Yash Pandya, Aravind Rajeswaran, Corby Rosset, Alexey Taymanov, Luiz do Valle, Vibhav Vineet, Spencer Whitehead, and Andrew Zhao. Fara-1.5: Scalable learning environments for computer use agents. arXiv preprint arXiv:2606.20785, 2026.

Rogerio Bonatti, Dan Zhao, Francesco Bonacci, Dillon Dupont, Sara Abdali, Yinheng Li, Yadong Lu, Justin Wagle, Kazuhito Koishida, Arthur Bucker, Lawrence Jang, and Zack Hui. Windows Agent Arena: Evaluating multi-modal OS agents at scale. In ICML, 2025.

Neil Chowdhury, James Aung, Chan Jun Shern, Oliver Jaffe, Dane Sherburn, Giulio Starace, Evan Mays, Rachel Dias, Marwan Aljubeh, Mia Glaese, Carlos E. Jimenez, John Yang, Kevin Liu, and Aleksander Madry. Introducing SWE-bench Verified. https://openai.com/index/ introducing-swe-bench-verified/, 2024.

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2Web: Towards a generalist agent for the web. In NeurIPS, 2023.

Shuangrui Ding, Xuanlang Dai, Long Xing, Shengyuan Ding, Ziyu Liu, Yang JingYi, Penghui Yang, Zhixiong Zhang, Xilin Wei, Xinyu Fang, Yubo Ma, Haodong Duan, Jing Shao, Jiaqi Wang, Dahua Lin, Kai Chen, and Yuhang Zang. WildClawBench: A benchmark for real-world, long-horizon agent evaluation. arXiv preprint arXiv:2605.10912, 2026.

Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H. Laradji, Manuel Del Verme, Tom Marty, David Vazquez, Nicolas Chapados, and Alexandre Lacoste. WorkArena: How capable are web agents at solving common knowledge work tasks? In ICML, 2024.

GLM-5 Team. GLM-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026.

Google. Introducing Gemini 3.6 Flash, 3.5 Flash-Lite, and 3.5 Flash Cyber. https:// blog.google/innovation-and-ai/models-and-research/gemini-models/ gemini-3-6-flash-3-5-flash-lite-3-5-flash-cyber/, 2026.

Tanmay Gupta, Piper Wolters, Zixian Ma, Peter Sushko, Rock Yuren Pang, Diego Llanes, Yue Yang, Taira Anderson, Boyuan Zheng, Zhongzheng Ren, Harsh Trivedi, Taylor Blanton, Caleb Ouellette, Winson Han, Ali Farhadi, and Ranjay Krishna. MolmoWeb: Open visual web agent and open data for the open web. arXiv preprint arXiv:2604.08516, 2026.

Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. WebVoyager: Building an end-to-end web agent with large multimodal models. In ACL, 2024.

Lawrence Keunho Jang, Andrew Keunwoo Jang, Jing Yu Koh, and Ruslan Salakhutdinov. MyPCBench: A benchmark for personally intelligent computer-use agents. arXiv preprint arXiv:2606.16748, 2026a.

Lawrence Keunho Jang, Jing Yu Koh, Daniel Fried, and Ruslan Salakhutdinov. Odysseys: Benchmarking web agents on realistic long horizon tasks. arXiv preprint arXiv:2604.24964, 2026b.

Lawrence Keunho Jang, Mareks Woodside, Geronimo Carom, Andrew Keunwoo Jang, Jing Yu Koh, and Ruslan Salakhutdinov. iosworld: A benchmark for personally intelligent phone agents, 2026c. URL https://arxiv.org/abs/2606.09764.

Sayash Kapoor, Benedikt Stroebl, Zachary S. Siegel, Nitya Nadgir, and Arvind Narayanan. AI agents that matter, 2024. URL https://arxiv.org/abs/2407.01502.

Sayash Kapoor, Benedikt Stroebl, Peter Kirgis, Nitya Nadgir, Zachary S Siegel, Boyi Wei, Tianci Xue, Ziru Chen, Felix Chen, Saiteja Utpala, Franck Ndzomga, Dheeraj Oruganty, Sophie Luskin, Kangheng Liu, Botao Yu, Amit Arora, Dongyoon Hahm, Harsh Trivedi, Huan Sun, Juyong Lee, Tengjun Jin, Yifan Mai, Yifei Zhou, Yuxuan Zhu, Rishi Bommasani, Daniel Kang, Dawn Song, Peter Henderson, Yu Su, Percy Liang, and Arvind Narayanan. Holistic agent leaderboard: The missing infrastructure for AI agent evaluation. In The Fourteenth International Conference on Learning Representations (ICLR), 2026. URL https://arxiv.org/pdf/2510.11977.

Kimi Team. Kimi K3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026.

Alex Kipnis, Konstantinos Voudouris, Luca M. Schulze Buschoff, and Eric Schulz. metabench – a sparse benchmark of reasoning and knowledge in large language models. In ICLR, 2025.

Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Chong Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Ruslan Salakhutdinov, and Daniel Fried. VisualWebArena: Evaluating multimodal agents on realistic visual web tasks. In ACL, 2024.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with PagedAttention. In SOSP, 2023.

Evan Zheran Liu, Kelvin Guu, Panupong Pasupat, Tianlin Shi, and Percy Liang. Reinforcement learning on web interfaces using workflow-guided exploration. In ICLR, 2018.

Dunjie Lu, Shuai Bai, Tianyi Bai, et al. Qwen-CUA: Native computer use for (almost) everything. arXiv preprint arXiv:2608.02352, 2026.

Bowen Lv, Xiao Liu, Yanyu Ren, Hanyu Lai, Bohao Jing, Hanchen Zhang, Yanxiao Zhao, Shuntian Yao, Jie Tang, and Yuxiao Dong. ScaleCUA: Scaling computer use agents with verifiable task synthesis and efficient online RL. arXiv preprint arXiv:2607.11185, 2026.

Cristina Menghini, Peter Ney, Hamza Kwisaba, Zifan Wang, Miles Turpin, et al. Muse Spark safety & preparedness report. arXiv preprint arXiv:2606.12429, 2026.

OpenAI. Introducing GPT-6. https://openai.com/index/gpt-6-astra//, 2026.

Felipe Maia Polo, Lucas Weber, Leshem Choshen, Yuekai Sun, Gongjun Xu, and Mikhail Yurochkin. tinyBenchmarks: evaluating LLMs with fewer examples. In ICML, 2024.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, Wanjun Zhong, Kuanye Li, Jiale Yang, Yu Miao, Woyu Lin, Longxiang Liu, Xu Jiang, Qianli Ma, Jingyu Li, Xiaojun Xiao, Kai Cai, Chuang Li, Yaowei Zheng, Chaolin Jin, Chen Li, Xiao Zhou, Minchao Wang, Haoli Chen, Zhaojian Li, Haihua Yang, Haifeng Liu, Feng Lin, Tao Peng, Xin Liu, and Guang Shi. UI-TARS: Pioneering automated GUI interaction with native agents. arXiv preprint arXiv:2501.12326, 2025.

Qwen Team. Qwen3.8-Max: A new bar for coding and cowork. https://qwen.ai/blog?id= qwen3.8, 2026.

Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, Daniel Toyama, Robert Berry, Divya Tyamagundlu, Timothy Lillicrap, and Oriana Riva. AndroidWorld: A dynamic benchmarking environment for autonomous agents. In ICLR, 2025.

Barada Sahu and Shivesh Pandey. Teach it to stop, not just to click. arXiv preprint arXiv:2607.17136, 2026.

Yueqi Song, Lintang Sutawika, Jiarui Liu, Lindia Tjuatja, Jiayi Geng, Yunze Xiao, Daniel Lee, Aditya Bharat Soni, Vincent Lo, Xiang Yue, and Graham Neubig. PACE: A proxy for agentic capability evaluation. arXiv preprint arXiv:2607.02032, 2026.

Gabor J. Sz´ ekely and Maria L. Rizzo. Energy statistics: A class of statistics based on distances.´ Journal ofStatistical Planning and Inference, 143(8):1249–1272, 2013.

Bowen Wang, Dunjie Lu, Junli Wang, Tianyi Bai, Shixuan Liu, Zhipeng Zhang, Haiquan Wang, Hao Hu, Tianbao Xie, Shuai Bai, Dayiheng Liu, Que Shen, Junyang Lin, and Tao Yu. CUA-Gym: Scaling verifiable training environments and tasks for computer-use agents. arXiv preprint arXiv:2605.25624, 2026.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. OSWorld: Benchmarking multimodal agents for open-ended tasks in real computer environments. In NeurIPS, 2024.

Tianci Xue, Weijian Qi, Tianneng Shi, Chan Hee Song, Boyu Gou, Dawn Song, Huan Sun, and Yu Su. An illusion of progress? Assessing the current state of web agents. In COLM, 2025.

Pei Yang, Hai Ci, and Mike Zheng Shou. macOSWorld: A multilingual interactive benchmark for GUI agents. In NeurIPS, 2025.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. WebShop: Towards scalable real-world web interaction with grounded language agents. In NeurIPS, 2022.

Mengqi Yuan, Zilong Zhou, Xinzhuang Xiong, Weiming Wu, Jiayang Sun, Jiamin Song, Kaiqian Cui, Bowen Wang, Haoyuan Wu, Yitong Li, Dunjie Lu, Haikong Lu, Qi Zhen, Xinyuan Wang, Jiaqi Deng, Yuhao Yang, Cheng Chen, Boyuan Zheng, Alex Su, Xiao Yu, Hao Zou, Saaket Agashe, Xing Han Lu, Manpreet Kaur, Zhengyang Qi, Vincent Sunn Chen, Frederic Sala, Dayiheng Liu, Junyang Lin, Zhou Yu, Yu Su, Siva Reddy, Xin Eric Wang, Peng Qi, Tianbao Xie, and Tao Yu. OSWorld 2.0: Benchmarking computer use agents on long-horizon real-world tasks. arXiv preprint arXiv:2606.29537, 2026.

Taolin Zhang, Hang Guo, Wang Lu, Tao Dai, Shu-Tao Xia, and Jindong Wang. SparseEval: Efficient evaluation of large language models by sparse optimization. In ICLR, 2026a.

Yuxuan Zhang, Yubo Wang, Yipeng Zhu, Penghui Du, Junwen Miao, Xuan Lu, Zhuofeng Li, Xingwei Qu, Zhengkang Guo, Yuanzhe Shen, Dingjie Song, Han Zhou, Tuney Zheng, Xian Wu, Hao Yu, Songcheng Cai, Yi Lu, Yunzhuo Hao, Minyi Lei, Liang Chen, Kai Zou, Huifeng Yin, Wendong Xu, Dongfu Jiang, Ping Nie, Jiaheng Liu, Wenhu Chen, and Kelsey R. Allen. ClawBench: Can AI agents complete everyday online tasks? arXiv preprint arXiv:2604.08523, 2026b.

Hongli Zhou, Hui Huang, Ziqing Zhao, Lvyuan Han, Huicheng Wang, Kehai Chen, Muyun Yang, Wei Bao, Jian Dong, Bing Xu, Conghui Zhu, Hailong Cao, and Tiejun Zhao. Lost in benchmarks? Rethinking large language model benchmarking with item response theory. In AAAI, 2026.

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. WebArena: A realistic web environment for building autonomous agents. In ICLR, 2024.

## APPENDIX TABLE OF CONTENTS

A Limitations .   
B Infrastructure: Technical Details   
B.1 Agent and Environment Sandboxes 16   
B.2 Model Serving, Resource Allocation, and Parallel Evaluation 16   
B.3 Fast I/O Implementation 16   
B.4 Action Latency: Fast I/O, Gym-Anything, and OSWorld 17   
B.5 Latency Breakdown. 17   
B.6 Fast I/O Consistency Across Five Seeds . 18   
C Implementing and Evaluating an Agent . 18   
C.1 Agent Templates 19   
C.2 Minimal Agent Example 19   
C.3 Hosted and Local Evaluation 20   
D Interaction and Token Measurements . 20   
D.1 Model Calls, Action Batches, and Individual Actions . 20   
D.2 Token Accounting 21   
D.3 Full Interaction and Token Comparisons 21   
D.4 Performance–Interaction and Performance–Token Frontiers . 23   
E Full Evaluation Results . 25   
E.1 Model, Harness, and Reasoning-Effort Configurations. 25   
E.2 OSWorld: Performance–Time and Performance–Cost Frontiers 27   
E.3 OSWorld2: Performance–Time and Performance–Cost Frontiers 28   
E.4 OSWorld2: Joint Performance–Time–Cost Frontier 29   
E.5 Reasoning Effort on MyPCBench . 30   
E.6 Reasoning Effort on CUA-World 30   
F Run-to-Run Variation . 31   
F.1 Five-Seed Evaluation Setup . 31   
F.2 Scores and Times Across Seeds . 31   
F.3 Task-Level Score Agreement 31   
F.4 Variation in Action Sequences 32   
G Representative Task Selection 33   
G.1 Models and Tasks Used for Selection 33   
G.2 Initialization, Optimization, and Task Count. 33   
G.3 Held-Out Score and Ranking Accuracy 35   
G.4 Preserving the Speed–Performance Frontier . 36   
G.5 Comparing Task-Selection Methods. 37   
G.6 Selected Task Identities . 37   
H Internet-Dependence Audit . 39   
H.1 Inclusion Rule 39   
H.2 Reviewer Agreement 39   
Validating Actions with CUA-AutoDebug. 40   
I.1 Separating Model, Harness, and Runtime Errors. 40   
I.2 Input Failures Detected by the Runtime Tests 40   
I.3 Failures in Agent Harnesses. 40   
Experimental Details 41   
J.1 Task Definition 41   
J.2 Observations, Actions, and Interaction History 41   
J.3 Task Limits and Execution 42   
J.4 Benchmark Verification . 42   
J.5 Model Cost . 42   
K Additional Agent Analyses 42   
K.1 Controlled Batching Comparisons. 43   
K.2 Harness Comparisons Across Reasoning Efforts. 44   
K.3 Comparing Reasoning Efforts on the Same Successful Tasks 45   
L Automated Search for Task-Selection Methods 46   
L.1 Search and Evaluation 46   
L.2 Validation Gains Do Not Generalize to the Held-Out Model . 46

## A LIMITATIONS

Reported times and costs reflect API inference conditions at the time of evaluation (Appendix J). Providers control subsequent changes to inference speed and pricing. We validate representative task subsets on held-out agents (Appendix G); subset scores approximate full-benchmark performance. Changes in agent capabilities may require revalidation of these subsets.

Our evaluations cover a broad range of agents across four benchmarks (Appendix E). Evaluation cost limits coverage of every combination of model, harness, reasoning effort, and benchmark.

## B INFRASTRUCTURE: TECHNICAL DETAILS

## B.1 AGENT AND ENVIRONMENT SANDBOXES

We run the agent and task environment in separate sandboxes. The environment contains the desktop, applications, and initial task state; the agent interacts with it through screenshot and action requests. Task initialization and verification remain independent of the agent, allowing the same agent implementation to run across benchmarks. When the agent terminates or reaches its limit, the benchmark’s verifier scores the trajectory or final environment state.

We use Modal to host the sandboxes and adapt the environment runtime from Gym-Anything (Aggarwal et al., 2026). The environment measures task time and the time spent executing actions and returning observations, independently of the agent implementation. As in Sec. 2.1, task preparation, agent initialization, and verification are excluded from task time.

## B.2 MODEL SERVING, RESOURCE ALLOCATION, AND PARALLEL EVALUATION

Agents access models through API endpoints or a self-hosted vLLM inference server (Kwon et al., 2023). We use L40S GPUs for self-hosted inference. The evaluation configuration specifies the compute resources, task limits, and number of concurrent agents. Model initialization and weight loading finish before task timing begins.

We assign tasks to available agents and environments. Each task starts from a fresh environment state. Its clock starts after the agent and environment are ready and stops when the agent finishes or reaches its limit, excluding time waiting for an execution slot. Model servers reuse loaded weights across tasks. We record a trajectory for each task.

## B.3 FAST I/O IMPLEMENTATION

FastCUA, our Fast I/O implementation, reduces the time between issuing an action and receiving a screenshot. We encode screenshots in the background and combine action execution and screenshot delivery in a single request.

Screenshot capture and encoding. We use QEMU’s D-Bus display interface to keep the current screen in memory and encode updated frames in the background. We use JPEG at quality 95 with no chroma subsampling and retain the full 1920 × 1080 resolution.

Action execution and communication. We execute the action and return its screenshot in one request, reusing the network connection across requests. A lightweight command client reduces per-action startup overhead. We also place agent and environment sandboxes in the same Modal region to reduce network latency. Fast I/O returns the screenshot as soon as action execution and image preparation finish. It can therefore return a screenshot before the application has updated its interface, as illustrated in Figure 5.

## B.4 ACTION LATENCY: FAST I/O, GYM-ANYTHING, AND OSWORLD

<table><tr><td>Infrastructure</td><td>Click</td><td>Escape</td><td>Ctrl+U</td><td>Type 100 characters</td></tr><tr><td>Gym-Anything</td><td>2758.70</td><td>2865.00</td><td>2858.29</td><td>3521.72</td></tr><tr><td>OSWorld</td><td>2515.41</td><td>2707.74</td><td>2708.41</td><td>2756.47</td></tr><tr><td>Fast I/O (ours)</td><td>27.58</td><td>2.43</td><td>2.91</td><td>14.98</td></tr></table>

Table 1: Fast I/O reduces action-to-observation latency to milliseconds. Median latency in milliseconds over 20 calls per action. For each system, the timer starts immediately before env.step(action) and stops when it returns the observation. All default action waits are included.

Fast I/O reduces latency by one to three orders of magnitude. Table 1 compares Fast I/O with Gym-Anything (Aggarwal et al., 2026) and OSWorld (Xie et al., 2024) for mouse clicks, key presses, and text entry. Fast I/O takes 2.4–27.6 ms per call, compared to 2.5–3.5 s for the existing implementations. For example, a click takes 27.6 ms compared with 2.8 s with Gym-Anything, while typing 100 characters takes 15.0 ms compared with 3.5 s.

Measurement setup. We time each runtime’s native env.step(action) call, from issuing the action until the observation is returned. We test four actions on a 1920 × 1080 Ubuntu desktop: clicking the Activities button, pressing Escape, pressing Ctrl+U, and typing 100 ASCII characters into a terminal. The caller and VM run in the same Modal sandbox. OSWorld uses its Docker provider with screenshot observations; both baselines use their default waits and action implementations. Table 1 reports the median of 20 calls per action, without profiling. Setup, pauses between calls, and saving returned screenshots occur outside the timed interval.

## B.5 LATENCY BREAKDOWN

<table><tr><td>Component (ms)</td><td>Click</td><td>Escape</td><td>Ctrl+U</td><td>Type 100 characters</td></tr><tr><td>Gym-Anything</td><td></td><td></td><td></td><td></td></tr><tr><td>Fixed waits</td><td>2100.2</td><td>2100.2</td><td>2100.2</td><td>2100.2</td></tr><tr><td>Action request</td><td>153.5</td><td>153.7</td><td>154.0</td><td>805.5</td></tr><tr><td>Screenshot capture</td><td>258.9</td><td>354.0</td><td>354.3</td><td>354.1</td></tr><tr><td>Screenshot transfer</td><td>238.1</td><td>212.4</td><td>220.6</td><td>211.3</td></tr><tr><td>Cleanup and other</td><td>53.1</td><td>53.1</td><td>53.1</td><td>53.1</td></tr><tr><td>Total</td><td>2803.9</td><td>2873.4</td><td>2882.2</td><td>3524.2</td></tr><tr><td>Total minus 2 s</td><td>803.9</td><td>873.4</td><td>882.2</td><td>1524.2</td></tr><tr><td>OSWorld</td><td></td><td></td><td></td><td></td></tr><tr><td>Fixed wait</td><td>2000.1</td><td>2000.1</td><td>2000.1</td><td>2000.1</td></tr><tr><td>Action request</td><td>198.6</td><td>191.7</td><td>192.8</td><td>245.1</td></tr><tr><td>Screenshot request</td><td>324.7</td><td>521.2</td><td>521.8</td><td>523.2</td></tr><tr><td>Other</td><td>0.2</td><td>0.2</td><td>0.2</td><td>0.2</td></tr><tr><td>Total</td><td>2523.6</td><td>2713.3</td><td>2714.9</td><td>2768.7</td></tr><tr><td>Total minus 2 s</td><td>523.6</td><td>713.3</td><td>714.9</td><td>768.7</td></tr></table>

Table 2: Fixed waits account for most baseline latency. Mean component times from 10 profiled calls per action for Gym-Anything and 20 for OSWorld. Components sum to the total before rounding. The final row for each system is the profiled total minus its two-second post-action wait.

Where do the baselines spend time? Table 2 shows that fixed sleeps account for most of the baseline latency. Both implementations sleep for two seconds after an action, and Gym-Anything adds another 100 ms per step. Subtracting the two-second sleep leaves 804–1524 ms for Gym-Anything and 524–769 ms for OSWorld. Gym-Anything captures a PNG with FFmpeg, transfers it, and deletes the temporary file; typing also adds a 6 ms interval between characters. In OSWorld, most of the remaining time is spent in the action and screenshot requests, including processing inside the VM and communication.

<table><tr><td>Component (ms)</td><td>Click</td><td>Escape</td><td>Ctrl+U</td><td>Type 100 characters</td></tr><tr><td>Input execution</td><td>27.30</td><td>1.85</td><td>2.55</td><td>18.39</td></tr><tr><td>Frame capture</td><td>1.03</td><td>0.77</td><td>0.77</td><td>0.80</td></tr><tr><td>Foreground image preparation</td><td>0.27</td><td>0.02</td><td>0.02</td><td>0.18</td></tr><tr><td>Other environment processing</td><td>0.93</td><td>0.94</td><td>1.00</td><td>0.90</td></tr><tr><td>Transport</td><td>4.67</td><td>4.99</td><td>4.76</td><td>4.62</td></tr><tr><td>Screenshot-file delivery</td><td>0.43</td><td>0.45</td><td>0.43</td><td>0.44</td></tr><tr><td>Command overhead</td><td>4.41</td><td>4.52</td><td>4.38</td><td>4.38</td></tr><tr><td>Total</td><td>39.06</td><td>13.55</td><td>13.90</td><td>29.71</td></tr></table>

Table 3: Fast I/O completes the four tested actions in under 50 ms. Mean times over 20 commands per action, from the agent issuing a command to receiving the screenshot file. This includes command overhead, communication between sandboxes, and screenshot-file delivery in addition to the env.step call. Components sum to the total before rounding.

Full command-to-screenshot latency. Table 3 measures latency from the agent issuing a command to receiving the screenshot file, including communication between sandboxes. We place both sandboxes in New York and test the four actions on two OSWorld desktops, with 0.5 s and 8 s gaps between requests. All 80 timed commands complete in under 50 ms. Median times are 38.5 ms for a click, 12.8 ms for Escape, 13.6 ms for Ctrl+U, and 29.5 ms for typing 100 characters. Communication contributes 4.6–5.0 ms on average. Image preparation after the request takes 0.02–0.27 ms; encoding runs in the background. Clicks take longer because input execution includes a 25 ms interval between pressing and releasing the mouse button.

## B.6 FAST I/O CONSISTENCY ACROSS FIVE SEEDS

<table><tr><td>Run</td><td>Score (%)</td><td>Time (s)</td><td>Agent (s)</td><td>Steps</td><td>Env. (s)</td><td>Waits (s)</td><td>Env.—waits (s)</td></tr><tr><td>1</td><td>91.62</td><td>103.96</td><td>98.07</td><td>8.68</td><td>5.891</td><td>5.553</td><td>0.338</td></tr><tr><td>2</td><td>91.62</td><td>93.98</td><td>89.02</td><td>8.04</td><td>4.955</td><td>4.628</td><td>0.327</td></tr><tr><td>3</td><td>91.62</td><td>96.48</td><td>91.67</td><td>7.96</td><td>4.812</td><td>4.488</td><td>0.324</td></tr><tr><td>4</td><td>87.62</td><td>100.22</td><td>95.28</td><td>8.20</td><td>4.934</td><td>4.604</td><td>0.330</td></tr><tr><td>5</td><td>89.62</td><td>100.46</td><td>95.68</td><td>8.12</td><td>4.785</td><td>4.424</td><td>0.361</td></tr><tr><td>Mean</td><td>90.42</td><td>99.02</td><td>93.94</td><td>8.20</td><td>5.075</td><td>4.739</td><td>0.336</td></tr><tr><td>SD</td><td>1.79</td><td>3.87</td><td>3.58</td><td>0.28</td><td>0.462</td><td>0.462</td><td>0.015</td></tr></table>

Table 4: Fast I/O remains consistent across five evaluations. Each row averages all 50 OSWorld tasks for GPT-6 Astra at low reasoning effort. SD is the sample standard deviation across run-level means; score SD is in percentage points. Steps count action batches, and waits are explicitly requested by the agent.

Environment processing time remains low across runs. Table 4 reports five evaluations of GPT-6 Astra at low effort on the same 50 OSWorld tasks with Fast I/O. We keep the agent prompt fixed and use fresh task seeds, a 500-step limit, and agent and environment sandboxes in New York. Environment time excluding agent-requested waits is $0 . 3 3 6 \pm 0 . 0 1 5 \mathrm { s }$ per task (mean ± standard deviation across runs). The mean task time is 99.0 ± 3.9 s, and the score is 90.4 ± 1.8%.

Most environment time is spent on agent-requested waits. Table 4 also separates the waits requested by the agent from other environment processing. These waits account for 4.739 s of the 5.075 s of mean environment time per task, leaving only 0.336 s for the remaining processing. Appendix D defines these timing components.

## C IMPLEMENTING AND EVALUATING AN AGENT

To evaluate a new agent with cua-speedrun, users implement its model calls, prompts, and interaction history. The infrastructure handles task setup, desktop interaction, and verification through the common interface in Sec. 2.3. The same agent implementation and environment interface can then be used across benchmarks.

## C.1 AGENT TEMPLATES

We provide templates for direct model-API agents, Codex, Claude Code, and open-weight agents served through vLLM. Each template contains two files: init.py prepares dependencies and model serving, and agent.py implements the agent loop. Initialization finishes before task timing begins. Users can retain init.py and modify only the agent loop to test a new model, prompt, or interaction strategy.

## C.2 MINIMAL AGENT EXAMPLE

Listing 1 shows a minimal agent. The script receives the environment URL and task instruction as command-line arguments and uses the Computer client to request screenshots, execute actions, and end the task. In each iteration, choose actions receives the instruction, screenshot, and interaction history. This user-defined function calls the model and returns an action list, or None when the agent considers the task complete. Users determine how to present this information to the model.

```python
Listing 1: Minimal agent loop.
import os
import sys
from cua_speedrun.client import Computer
def run(env_url, task):
computer = Computer(env_url)
history = []
limit = int(os.environ.get("CS_MAX_STEPS", "500"))
for _ in range(limit):
observation = computer.observe()
actions = choose_actions(task, observation["png"], history)
if actions is None:
break
computer.step(actions)
history.append((observation["png"], actions))
computer.done()
if __name__ == "__main__":
run(sys.argv[1], sys.argv[2])
```

For example, an action list can contain $\{ { \mathfrak { m o u s e } } ^ { \mathfrak { u } } : \quad \{ { \mathfrak { n } } \atop \mathbf { l e f t \_ c l i c k ^ { \mathfrak { u } } } : \quad [ { \mathfrak { \omega } } \in { \mathfrak { u } }  0 0 ] \} \}$ followed by $\{ " k \mathrm { e y b o a r d } " : \{ " \mathrm { e x t } " : \quad " \mathrm { n e l } 1 \mathrm { o } " \} \}$ . These two actions are executed in order within one step. Convenience methods such as click $( 6 4 0 , 4 0 0 )$ , type text $( " \mathrm { h e l 1 o " } )$ and $\mathtt { k e y s ( [ } ^ { \mathfrak { n } } \mathtt { C t r 1 " } , \mathrm { ~ \mathfrak { n } _ { C } " ] ) }$ ) each submit a single action. The agent can execute several actions before requesting another screenshot. Calling done() ends the interaction; the benchmark’s verifier determines the score.

## C.3 HOSTED AND LOCAL EVALUATION

Listing 2 runs the Codex template on the representative OSWorld set using Modal. The command specifies the submission directory, benchmark, and number of parallel evaluations. Omitting --remote runs both the agent and environment VMs on the evaluator host. For local desktop evaluation, this must be a Linux host with the VM runtime and hardware required by the benchmark, as well as model-serving resources when using open-weight models.

```shell
Listing 2: Evaluating a supplied agent template.
cua-speedrun run --remote \
--submission templates/codex_cli \
--benchmark benchmarks/osworld-energy50-representative \
--agents-per-evaluation 1 --parallel-evaluations 8
```

Users can also submit agents through the dashboard. For each evaluation, we record the benchmark, hardware, evaluation settings, and task seeds, along with task scores and timings. Action logs and screenshots allow users to inspect how the agent completed or failed each task.

## D INTERACTION AND TOKEN MEASUREMENTS

Alongside performance, time, and cost, we measure how often an agent calls its model, how many actions it executes, and how many tokens it generates. These measurements support the analysis of agent speed in Sec. 3. The full configuration tables average over all tasks, including unsuccessful attempts.

Aggregate metrics. For an agent π evaluated on $K$ tasks, the mean verifier score $P ,$ mean task time $T _ { \ast }$ , and mean model cost $\check { C }$ are

$$
P ( \pi ) = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } r _ { i } , \qquad T ( \pi ) = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } t _ { i } , \qquad C ( \pi ) = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } c _ { i } ,\tag{4}
$$

where $r _ { i }$ is the verifier score defined in Appendix J.1, and $t _ { i }$ and $c _ { i }$ are the execution time and model cost of task i. The mean score retains partial credit; we also report the fraction of exactly completed tasks and the median time on those tasks.

Agent time, environment time, and requested waits. We separate total task time into agent time and environment time. For task i, environment time $e _ { i }$ sums the intervals recorded by the environment server for executing actions and returning observations. This includes explicit waits requested by the agent; we report their requested durations separately and subtract them when reporting environment time excluding agent waits. Agent time is the remainder, $t _ { i } - e _ { i }$ , which includes model requests and agent-side processing.

## D.1 MODEL CALLS, ACTION BATCHES, AND INDIVIDUAL ACTIONS

Model responses. We count each completed model response, including responses without a computer action. A response can contain several tool calls, each of which may execute several actions. We therefore count model responses and desktop actions separately.

Action batches and individual actions. An agent can execute several actions in one request. For example, clicking a field, typing text, and pressing Enter in one request counts as one batch and three actions. Issuing these actions separately counts as three batches and three actions. A step denotes one request containing keyboard, mouse, or wait actions. For task i with $S _ { i }$ batches, let $\mathcal { A } _ { i j }$ be the action list in request $j .$ . The total number of individual actions is $\begin{array} { r } { A _ { i } = \sum _ { j = 1 } ^ { S _ { i } } | \mathcal { A } _ { i j } | } \end{array}$

We count actions as submitted by the agent: typing a string, pressing a key combination, or requesting a wait each counts as one action.

## D.2 TOKEN ACCOUNTING

We count every generated token once per response, including reasoning and other output, using the provider’s reported usage. We also report provider-supplied reasoning-token counts separately.

To compare token rates, we divide total generated tokens by total agent time: $\textstyle \sum _ { i } G _ { i } / \sum _ { i } ( t _ { i } - e _ { i } )$ where $\bar { G } _ { i }$ is the generated-token count for task i. This effective token rate includes the time spent on model requests and agent-side processing.

## D.3 FULL INTERACTION AND TOKEN COMPARISONS

Tables 5 and 6 report mean batches, individual actions, model responses, and generated tokens for the configurations in Appendix E. Configuration IDs match the frontier plots and result tables.

Table 5: Interaction and token counts on OSWorld. Means per task, including unsuccessful tasks. IDs match the OSWorld configuration tables in Appendix E. Tokens include reasoning and other output. Astra (low) uses five-seed means for each I/O setting.
<table><tr><td>ID</td><td>Configuration</td><td>Batches</td><td>Actions</td><td>Responses</td><td>Tokens</td></tr><tr><td>1</td><td>GPT-6 Astra / xhigh</td><td>7.14</td><td>19.90</td><td></td><td>2,060</td></tr><tr><td>2</td><td>Gemini 3.8 Flash / low</td><td>16.60</td><td>20.02</td><td>18.52</td><td>1,564</td></tr><tr><td>3</td><td>Claude Opus 5 / high</td><td>19.92</td><td>26.92</td><td>20.62</td><td>3,735</td></tr><tr><td>4</td><td>Gemini 3.8 Flash / medium</td><td>27.16</td><td>34.70</td><td>29.10</td><td>5,365</td></tr><tr><td>5</td><td>GPT-6 Astra / medium</td><td>6.30</td><td>19.26</td><td></td><td>1,366</td></tr><tr><td>6</td><td>GPT-6 Astra / high</td><td>6.52</td><td>19.76</td><td></td><td>1,593</td></tr><tr><td>7</td><td>GPT-6 Astra / low / fast</td><td>8.20</td><td>27.10</td><td></td><td>1,328</td></tr><tr><td>8</td><td>Claude Sonnet 5 / high</td><td>20.24</td><td>29.20</td><td>22.28</td><td>4,840</td></tr><tr><td>9</td><td>Gemini 3.7 Flash / medium</td><td>19.44</td><td>23.34</td><td>21.36</td><td>3,594</td></tr><tr><td>10</td><td>Gemini 3.7 Flash / high</td><td>24.26</td><td>28.30</td><td>26.20</td><td>5,731</td></tr><tr><td>11</td><td>Kimi K3 / max / batched</td><td>11.70</td><td>53.58</td><td>11.68</td><td>10,435</td></tr><tr><td>12</td><td>Claude Opus 5 / low</td><td>14.04</td><td>23.80</td><td>16.08</td><td>1,795</td></tr><tr><td>13</td><td>GPT-6 Astra / low</td><td>5.86</td><td>19.10</td><td></td><td>1,579</td></tr><tr><td>14</td><td>Gemini 3.7 Flash / low</td><td>18.24</td><td>20.88</td><td>20.14</td><td>2,944</td></tr><tr><td>15</td><td>Gemini 3.8 Flash / high</td><td>37.18</td><td>51.46</td><td>39.08</td><td>8,789</td></tr><tr><td>16</td><td>GPT-5.6 Luna / medium / API</td><td>14.42</td><td>48.88</td><td>15.34</td><td>1,751</td></tr><tr><td>17</td><td>GPT-5.6 Sol / xhigh</td><td>12.66</td><td>36.78</td><td>13.62</td><td>2,402</td></tr><tr><td>18</td><td>Kimi K3 / high / single</td><td>8.92</td><td>113.60</td><td>10.70</td><td>7,567</td></tr><tr><td>19</td><td>GPT-5.6 Sol / medium</td><td>12.72</td><td>42.64</td><td>13.68</td><td>2,717</td></tr><tr><td>20</td><td>GPT-5.6 Luna / high / Codex</td><td>15.58</td><td>39.80</td><td></td><td>8,637</td></tr><tr><td>21</td><td>Kimi K3 / high / batched</td><td>8.76</td><td>60.20</td><td>9.20</td><td>7,125</td></tr><tr><td>22</td><td>Muse Spark 1.1 / xhigh</td><td>69.82</td><td>145.56</td><td>26.02</td><td>10,701</td></tr><tr><td>23</td><td>GPT-5.6 Luna / xhigh / Codex</td><td>13.88</td><td>33.10</td><td></td><td>8,160</td></tr><tr><td>24</td><td>Muse Spark 1.3 / xhigh</td><td>56.58</td><td>86.72</td><td>30.62</td><td>11,505</td></tr><tr><td>25</td><td>Muse Spark 1.3 / medium</td><td>41.54</td><td>69.32</td><td>26.10</td><td>8,952</td></tr><tr><td>26</td><td>Claude Sonnet 5 / low</td><td>21.24</td><td>29.50</td><td>24.16</td><td>3,468</td></tr><tr><td>27</td><td>GPT-5.6 Luna / low / Codex</td><td>7.22</td><td>26.44</td><td></td><td>3,191</td></tr><tr><td>28</td><td>MiniMax M3 / thinking-off</td><td>34.32</td><td>51.10</td><td>35.14</td><td></td></tr><tr><td>29</td><td>GPT-5.6 Luna / high / API</td><td>18.28</td><td>53.88</td><td>19.24</td><td>3,358</td></tr><tr><td>30</td><td>Kimi K3 / low / batched</td><td>5.74</td><td>30.00</td><td>6.90</td><td>1,865</td></tr><tr><td>31</td><td>Muse Spark 1.3 / high</td><td>51.30</td><td>84.78</td><td>28.98</td><td>10,538</td></tr><tr><td>32</td><td>Kimi K3 / low / single</td><td>8.12</td><td>55.10</td><td>9.62</td><td>3,171</td></tr><tr><td>33</td><td>MiniMax M3 / thinking_on</td><td>35.00</td><td>55.30</td><td>35.94</td><td></td></tr><tr><td>34</td><td>Muse Spark 1.1 / medium</td><td>54.36</td><td>114.00</td><td>22.82</td><td>7,352</td></tr><tr><td>35</td><td>Muse Spark 1.1 / high</td><td>69.90</td><td>126.88</td><td>25.66</td><td>10,132</td></tr><tr><td>36</td><td>Muse Spark 1.1 / low</td><td>58.48</td><td>149.86</td><td>22.12</td><td>6,526</td></tr><tr><td>37</td><td>Muse Spark 1.2 / xhigh</td><td>66.54</td><td>200.52</td><td>43.08</td><td>17,198</td></tr><tr><td>38</td><td>GPT-5.6 Luna / medium / Codex</td><td>9.50</td><td>25.64</td><td></td><td>4,524</td></tr><tr><td>39</td><td>Muse Spark 1.1 / minimal</td><td>53.44</td><td>131.68</td><td>20.62</td><td>5,957</td></tr><tr><td>40</td><td>Muse Spark 1.3 / low</td><td>40.50</td><td>71.82</td><td>23.54</td><td>6,747</td></tr><tr><td>41</td><td>GPT-5.6 Sol / low</td><td>11.20</td><td>45.34</td><td>12.16</td><td>1,236</td></tr><tr><td>42</td><td>Muse Spark 1.2 / low</td><td>31.12</td><td>137.82</td><td>23.94</td><td>4,723</td></tr></table>

Continued on next page

Table 5: OSWorld interaction counts, continued.
<table><tr><td>ID</td><td>Configuration</td><td>Batches</td><td>Actions</td><td>Responses</td><td>Tokens</td></tr><tr><td>43</td><td>Muse Spark 1.2 / medium</td><td>44.20</td><td>152.84</td><td>28.34</td><td>7,789</td></tr><tr><td>44</td><td>GPT-5.6 Luna / low / API</td><td>10.12</td><td>37.66</td><td>11.06</td><td>1,063</td></tr><tr><td>45</td><td>Muse Spark 1.3 / minimal</td><td>32.04</td><td>59.52</td><td>19.76</td><td>3,792</td></tr><tr><td>46</td><td>Muse Spark 1.2 / high</td><td>45.72</td><td>145.58</td><td>33.74</td><td>9,551</td></tr><tr><td>47</td><td>Gemini 3 Flash Preview / high</td><td>33.28</td><td>34.56</td><td>37.86</td><td>5,167</td></tr><tr><td>48</td><td>GLM-5V Turbo / thinking_off</td><td>16.86</td><td>21.52</td><td>17.54</td><td>8,150</td></tr><tr><td>49</td><td>Gemini 3 Flash Preview / medium</td><td>28.26</td><td>30.68</td><td>32.52</td><td>4,125</td></tr><tr><td>50</td><td>GLM-5V Turbo / thinking-on</td><td>17.86</td><td>19.24</td><td>18.62</td><td>8,563</td></tr><tr><td>51</td><td>Muse Spark 1.2 / minimal</td><td>23.20</td><td>87.38</td><td>18.72</td><td>2,379</td></tr><tr><td>52</td><td>Gemini 3 Flash Preview / low</td><td>39.52</td><td>40.26</td><td>56.02</td><td>1,598</td></tr><tr><td>53</td><td>Yutori n2 / none</td><td>18.74</td><td>56.54</td><td>20.90</td><td>3,652</td></tr><tr><td>54</td><td>Yutori n2 / low</td><td>22.96</td><td>58.72</td><td>25.32</td><td>12,188</td></tr><tr><td>55</td><td>Yutori n2 / medium</td><td>19.06</td><td>55.14</td><td>21.12</td><td>7,562</td></tr><tr><td>56</td><td>Yutori n2 / xhigh</td><td>17.56</td><td>48.16</td><td>19.72</td><td>7,499</td></tr></table>

Table 6: Interaction and token counts on OSWorld2. Means per task, including unsuccessful tasks. IDs match the OSWorld2 configuration tables in Appendix E. Tokens include reasoning and other output.
<table><tr><td>ID</td><td>Configuration</td><td>Batches</td><td>Actions</td><td>Responses</td><td>Tokens</td></tr><tr><td>1</td><td>GPT-6 Astra / xhigh</td><td>52.81</td><td>194.10</td><td></td><td>18,332</td></tr><tr><td>2</td><td>GPT-6 Astra / high</td><td>51.29</td><td>149.02</td><td></td><td>14,146</td></tr><tr><td>3</td><td>GPT-6 Astra / low</td><td>42.58</td><td>151.40</td><td></td><td>10,190</td></tr><tr><td>4</td><td>GPT-6 Astra / medium</td><td>45.46</td><td>151.08</td><td></td><td>11,382</td></tr><tr><td>5</td><td>Gemini 3.8 Flash / high</td><td>216.54</td><td>331.88</td><td>218.48</td><td>72,149</td></tr><tr><td>6</td><td>Gemini 3.8 Flash / medium</td><td>167.79</td><td>252.65</td><td>169.65</td><td>49,532</td></tr><tr><td>7</td><td>Claude Opus 5 / low</td><td>146.52</td><td>316.17</td><td>158.29</td><td>30,820</td></tr><tr><td>8</td><td>Claude Sonnet 5 / low</td><td>237.25</td><td>378.54</td><td>253.50</td><td>57,018</td></tr><tr><td>9</td><td>Gemini 3.8 Flash / low</td><td>127.04</td><td>200.75</td><td>128.83</td><td>19,840</td></tr><tr><td>10</td><td>Muse Spark 1.3 / minimal</td><td>115.35</td><td>488.06</td><td>66.02</td><td>16,150</td></tr><tr><td>11</td><td>Muse Spark 1.3 / low</td><td>130.50</td><td>655.08</td><td>80.73</td><td>25,131</td></tr><tr><td>12</td><td>Muse Spark 1.3 / medium</td><td>149.81</td><td>832.44</td><td>89.75</td><td>41,149</td></tr><tr><td>13</td><td>Muse Spark 1.3 / xhigh</td><td>147.29</td><td>832.75</td><td>93.96</td><td>52,628</td></tr><tr><td>14</td><td>Muse Spark 1.3 / high</td><td>151.21</td><td>754.98</td><td>89.83</td><td>44,241</td></tr><tr><td>15</td><td>Kimi K3 / low</td><td>47.25</td><td>313.75</td><td>49.17</td><td>21,852</td></tr><tr><td>16</td><td>MiniMax M3 / thinking_on</td><td>97.56</td><td>407.33</td><td>98.62</td><td></td></tr><tr><td>17</td><td>MiniMax M3 / thinking-off</td><td>98.54</td><td>368.62</td><td>98.65</td><td></td></tr><tr><td>18</td><td>GLM-5V Turbo / thinking_off</td><td>49.15</td><td>79.35</td><td>48.38</td><td>28,558</td></tr><tr><td>19</td><td>GLM-5V Turbo / thinking_on</td><td>47.75</td><td>91.75</td><td>47.12</td><td>28,385</td></tr><tr><td>20</td><td>Kimi K3 / high</td><td>58.77</td><td>350.06</td><td>59.04</td><td>81,494</td></tr><tr><td>21</td><td>Kimi K3 / max</td><td>66.38</td><td>386.56</td><td>60.79</td><td>123,603</td></tr></table>

![](images/1c04f1db264eb55d943780a316dcfea099ddb24d7074c4d45243f0ceced4e3f2.jpg)

(b) Individual actions  
![](images/da2aa2a6eab8f6b70963d01d06c849d6925936d4da789a63e7674feddf5c0441.jpg)

(c) Generated tokens  
![](images/22e655601ac42d6482469c40a71544eb0ed2890893b80a4e468b390d06ac8320.jpg)  
Figure 7: Performance–interaction and performance–token frontiers on OSWorld. Mean score against mean action batches, individual actions, and generated tokens per task. Tokens include reasoning and other output. Panels (a)–(b) show all 56 configurations; panel (c) shows the 54 with token counts. Dashed lines connect the circled frontier points. IDs match the configuration tables in Appendix E; darker green indicates greater reasoning effort. Astra (low) uses five-seed means for each I/O setting.

## D.4 PERFORMANCE–INTERACTION AND PERFORMANCE–TOKEN FRONTIERS

Figures 7 and 8 compare performance with the number of action batches, individual actions, and generated tokens on OSWorld and OSWorld2. Circled points mark the Pareto frontier for each metric. We use the same trajectories as the time and cost comparisons and average over all tasks.

![](images/80a308c3ca1c9731cde73090d21bdd8df3cb5046665a00892b650045f8878e3c.jpg)

(b) Individual actions  
![](images/07281a930d2940d8b1c1b4e1231250b2bc38d4350e0f258a29fc265bb428165f.jpg)

![](images/fe7374dad2f288c0604e104108a7368540acba26238ecde5c7cefd900931d5a3.jpg)  
Figure 8: Performance–interaction and performance–token frontiers on OSWorld2. Mean score against mean action batches, individual actions, and generated tokens per task. Panels (a)–(b) show all 21 configurations; panel (c) shows the 19 with token counts. IDs match the OSWorld2 configuration table in Appendix E. Other plotting conventions follow Figure 7.

## E FULL EVALUATION RESULTS

## E.1 MODEL, HARNESS, AND REASONING-EFFORT CONFIGURATIONS

Tables 7–9 report mean score, time, cost, and token use for each model, harness, and reasoning setting. Astra (low) uses five-seed means for each I/O setting. The configuration IDs identify the corresponding points in the figures.

We evaluate Luna through both the direct API and Codex, and Kimi with single-call and batched-call harnesses. We use the Yutori cookbook.

<table><tr><td>ID</td><td>Configuration</td><td>Score (%)</td><td>Time (s)</td><td>Cost ($)</td><td>Tokens</td></tr><tr><td>1</td><td>GPT-6 Astra / xhigh</td><td>91.6</td><td>126.8</td><td>0.708</td><td>2,060</td></tr><tr><td>2</td><td>Gemini 3.8 Flash / low</td><td>91.6</td><td>127.1</td><td>0.108</td><td>1,564</td></tr><tr><td>3</td><td>Claude Opus 5 / high</td><td>91.6</td><td>136.0</td><td>0.661</td><td>3,735</td></tr><tr><td>4</td><td>Gemini 3.8 Flash / medium</td><td>91.6</td><td>250.1</td><td>0.221</td><td>5,365</td></tr><tr><td>5</td><td>GPT-6 Astra / medium</td><td>89.6</td><td>90.2</td><td>0.600</td><td>1,366</td></tr><tr><td>6</td><td>GPT-6 Astra / high</td><td>89.6</td><td>100.8</td><td>0.636</td><td>1,593</td></tr><tr><td>7</td><td>GPT-6 Astra / low / fast</td><td>90.4</td><td>99.0</td><td>0.908</td><td>1,328</td></tr><tr><td>8</td><td>Claude Sonnet 5 / high</td><td>89.6</td><td>140.8</td><td>0.309</td><td>4,840</td></tr><tr><td>9</td><td>Gemini 3.7 Flash / medium</td><td>89.6</td><td>198.9</td><td>0.944</td><td>3,594</td></tr><tr><td>10</td><td>Gemini 3.7 Flash / high</td><td>89.6</td><td>254.3</td><td>1.412</td><td>5,731</td></tr><tr><td>11</td><td>Kimi K3 / max / batched</td><td>89.6</td><td>443.7</td><td>0.503</td><td>10,435</td></tr><tr><td>12</td><td>Claude Opus 5 / low</td><td>87.6</td><td>85.9</td><td>0.390</td><td>1,795</td></tr><tr><td>13</td><td>GPT-6 Astra / low</td><td>90.8</td><td>89.5</td><td>0.531</td><td>1,579</td></tr><tr><td>14</td><td>Gemini 3.7 Flash / low</td><td>87.6</td><td>207.3</td><td>0.926</td><td>2,944</td></tr><tr><td>15</td><td>Gemini 3.8 Flash / high</td><td>87.6</td><td>347.7</td><td>0.347</td><td>8,789</td></tr><tr><td>16</td><td>GPT-5.6 Luna / medium / API</td><td>75.6</td><td>118.9</td><td>0.019</td><td>1,751</td></tr><tr><td>17</td><td>GPT-5.6 Sol / xhigh</td><td>81.6</td><td>108.3</td><td>0.527</td><td>2,402</td></tr><tr><td>18</td><td>Kimi K3 / high / single</td><td>81.6</td><td>305.1</td><td>0.414</td><td>7,567</td></tr><tr><td>19</td><td>GPT-5.6 Sol / medium</td><td>79.6</td><td>121.3</td><td>0.539</td><td>2,717</td></tr><tr><td>20</td><td>GPT-5.6 Luna / high / Codex</td><td>79.6</td><td>324.8</td><td>0.076</td><td>8,637</td></tr><tr><td>21</td><td>Kimi K3 / high / batched</td><td>79.6</td><td>358.1</td><td>0.382</td><td>7,125</td></tr><tr><td>22</td><td>Muse Spark 1.1 / xhigh</td><td>79.6</td><td>417.2</td><td>1.685</td><td>10,701</td></tr><tr><td>23</td><td>GPT-5.6 Luna / xhigh / Codex</td><td>77.6</td><td>297.2</td><td>0.063</td><td>8,160</td></tr><tr><td>24</td><td>Muse Spark 1.3 / xhigh</td><td>77.6</td><td>612.1</td><td>0.161</td><td>11,505</td></tr><tr><td>25</td><td>Muse Spark 1.3 / medium</td><td>77.5</td><td>397.6</td><td>0.126</td><td>8,952</td></tr><tr><td>26</td><td>Claude Sonnet 5 / low</td><td>75.8</td><td>129.4</td><td>0.294</td><td>3,468</td></tr><tr><td>27</td><td>GPT-5.6 Luna / low / Codex</td><td>75.6</td><td>144.8</td><td>0.025</td><td>3,191</td></tr><tr><td>28</td><td>MiniMax M3 / thinking-off</td><td>75.6</td><td>253.8</td><td></td><td></td></tr><tr><td>29</td><td>GPT-5.6 Luna / high / API</td><td>81.6</td><td>160.5</td><td>0.029</td><td>3,358</td></tr><tr><td>30</td><td>Kimi K3 / low / batched</td><td>73.6</td><td>149.5</td><td>0.174</td><td>1,865</td></tr><tr><td>31</td><td>Muse Spark 1.3 / high</td><td>73.6</td><td>506.9</td><td>0.148</td><td>10,538</td></tr><tr><td>32</td><td>Kimi K3 / low / single</td><td>71.6</td><td>176.6</td><td>0.280</td><td>3,171</td></tr><tr><td>33</td><td>MiniMax M3 / thinking_on</td><td>69.6</td><td>345.1</td><td></td><td></td></tr><tr><td>34</td><td>Muse Spark 1.1 / medium</td><td>69.6</td><td>359.8</td><td>1.379</td><td>7,352</td></tr><tr><td>35</td><td>Muse Spark 1.1 / high</td><td>69.6</td><td>368.5</td><td>1.706</td><td>10,132</td></tr><tr><td>36</td><td>Muse Spark 1.1 / low</td><td>69.6</td><td>370.4</td><td>1.312</td><td>6,526</td></tr><tr><td>37</td><td>Muse Spark 1.2 / xhigh</td><td>69.6</td><td>720.9</td><td>0.244</td><td>17,198</td></tr><tr><td>38</td><td>GPT-5.6 Luna / medium / Codex</td><td>67.6</td><td>188.8</td><td>0.035</td><td>4,524</td></tr><tr><td>39</td><td>Muse Spark 1.1 / minimal</td><td>67.6</td><td>318.5</td><td>1.184</td><td>5,957</td></tr><tr><td>40</td><td>Muse Spark 1.3 / low</td><td>67.6</td><td>361.0</td><td>0.107</td><td>6,747</td></tr><tr><td>41</td><td>GPT-5.6 Sol / low</td><td>65.8</td><td>100.1</td><td>0.319</td><td>1,236</td></tr><tr><td>42</td><td>Muse Spark 1.2 / low</td><td>63.6</td><td>363.6</td><td>0.105</td><td>4,723</td></tr><tr><td>43</td><td>Muse Spark 1.2 / medium</td><td>63.6</td><td>466.2</td><td>0.136</td><td>7,789</td></tr><tr><td>44</td><td>GPT-5.6 Luna / low / API</td><td>61.8</td><td>73.6</td><td>0.011</td><td>1,063</td></tr><tr><td>45</td><td>Muse Spark 1.3 / minimal</td><td>61.6</td><td>264.4</td><td>0.079</td><td>3,792</td></tr><tr><td>46</td><td>Muse Spark 1.2 / high</td><td>61.6</td><td>669.8</td><td>0.174</td><td>9,551</td></tr><tr><td>47</td><td>Gemini 3 Flash Preview / high</td><td>59.6</td><td>337.2</td><td>0.809</td><td>5,167</td></tr><tr><td>48</td><td>GLM-5V Turbo / thinking_off</td><td>57.8</td><td>752.4</td><td>0.220</td><td>8,150</td></tr><tr><td>49</td><td>Gemini 3 Flash Preview / medium</td><td>57.6</td><td>268.0</td><td>0.592</td><td>4,125</td></tr><tr><td>50</td><td>GLM-5V Turbo / thinking-on</td><td>53.8</td><td>359.2</td><td>0.235</td><td>8,563</td></tr><tr><td>51</td><td>Muse Spark 1.2 / minimal</td><td>53.6</td><td>248.8</td><td>0.070</td><td>2,379</td></tr><tr><td>52</td><td>Gemini 3 Flash Preview / low</td><td>33.6</td><td>492.2</td><td>1.401</td><td>1,598</td></tr><tr><td>53</td><td>Yutori n2 / none</td><td>75.6</td><td>298.5</td><td>0.081</td><td>3,652</td></tr><tr><td>54</td><td>Yutori n2 / low</td><td>77.6</td><td>331.0</td><td>0.157</td><td>12,188</td></tr><tr><td>55</td><td>Yutori n2 / medium</td><td>67.6</td><td>265.4</td><td>0.110</td><td>7,562</td></tr><tr><td>56</td><td>Yutori n2 / xhigh</td><td>79.6</td><td>225.0</td><td>0.104</td><td>7,499</td></tr></table>

Table 7: OSWorld configurations. Bold IDs mark the joint performance–time–cost frontier. Tokens include reasoning and other generated output.

Table 8: OSWorld configurations, continued. Bold IDs mark the joint performance–time–cost frontier. Tokens include reasoning and other generated output.

<table><tr><td>ID</td><td>Configuration</td><td>Score (%)</td><td>Time (s)</td><td>Cost ($)</td><td>Tokens</td></tr><tr><td>1</td><td>GPT-6 Astra / xhigh</td><td>76.9</td><td>1237.3</td><td>8.175</td><td>18,332</td></tr><tr><td>2</td><td>GPT-6 Astra / high</td><td>75.0</td><td>914.0</td><td>7.391</td><td>14,146</td></tr><tr><td>3</td><td>GPT-6 Astra / low</td><td>68.2</td><td>904.2</td><td>6.336</td><td>10,190</td></tr><tr><td>4</td><td>GPT-6 Astra / medium</td><td>67.2</td><td>828.1</td><td>6.519</td><td>11,382</td></tr><tr><td>5</td><td>Gemini 3.8 Flash / high</td><td>60.2</td><td>3058.8</td><td>5.019</td><td>72,149</td></tr><tr><td>6</td><td>Gemini 3.8 Flash / medium</td><td>59.9</td><td>2051.8</td><td>2.915</td><td>49,532</td></tr><tr><td>7</td><td>Claude Opus 5 / low</td><td>52.1</td><td>1385.7</td><td>10.006</td><td>30,820</td></tr><tr><td>8</td><td>Claude Sonnet 5 / low</td><td>39.1</td><td>1993.6</td><td>8.863</td><td>57,018</td></tr><tr><td>9</td><td>Gemini 3.8 Flash / low</td><td>35.3</td><td>1386.7</td><td>1.875</td><td>19,840</td></tr><tr><td>10</td><td>Muse Spark 1.3 / minimal</td><td>34.0</td><td>1516.0</td><td>0.390</td><td>16,150</td></tr><tr><td>11</td><td>Muse Spark 1.3 / low</td><td>33.7</td><td>2280.8</td><td>0.503</td><td>25,131</td></tr><tr><td>12</td><td>Muse Spark 1.3 / medium</td><td>32.3</td><td>2292.1</td><td>0.588</td><td>41,149</td></tr><tr><td>13</td><td>Muse Spark 1.3 / xhigh</td><td>32.2</td><td>2261.0</td><td>0.640</td><td>52,628</td></tr><tr><td>14</td><td>Muse Spark 1.3 / high</td><td>31.5</td><td>2179.3</td><td>0.599</td><td>44,241</td></tr><tr><td>15</td><td>Kimi K3 / low</td><td>19.2</td><td>1860.9</td><td>2.088</td><td>21,852</td></tr><tr><td>16</td><td>MiniMax M3 / thinking-on</td><td>4.1</td><td>1673.2</td><td></td><td></td></tr><tr><td>17</td><td>MiniMax M3 / thinking_off</td><td>3.2</td><td>1446.9</td><td></td><td></td></tr><tr><td>18</td><td>GLM-5V Turbo / thinking-off</td><td>1.5</td><td>1979.4</td><td>0.739</td><td>28,558</td></tr><tr><td>19</td><td>GLM-5V Turbo / thinking-on</td><td>1.5</td><td>2641.9</td><td>0.725</td><td>28,385</td></tr><tr><td>20</td><td>Kimi K3 / high</td><td>22.6</td><td>3733.9</td><td>4.517</td><td>81,494</td></tr><tr><td>21</td><td>Kimi K3 / max</td><td>28.5</td><td>4825.1</td><td>5.625</td><td>123,603</td></tr></table>

Table 9: OSWorld2 configurations. Bold IDs mark the joint performance–time–cost frontier. Tokens include reasoning and other generated output.

![](images/3d158178fb418e95827f01b69343a84e0b474f9174ce6fd45353422969b2bfbc.jpg)

![](images/ab7a63b2105a35f9ebf2dd2e69c52e994c3d9d9e872aee1cb5557df5e8380e59.jpg)

Figure 9: Full OSWorld performance–time and performance–cost comparisons. All 56 configurations appear in the time comparison; the cost comparison uses configurations with reported or estimated costs. IDs identify the model, reasoning effort, and harness in Tables 7–8. Dashed lines connect nondominated configurations for each pair of axes. Scores, times, and costs are means across the same 50 tasks, including unsuccessful tasks.  
![](images/70546b298669c4c61da73d5921f6402d0b6198bf62153151985b0dcb579fa9a3.jpg)

![](images/8f2cee16b44f0928a21aa47440c62f157a82973df5fca7cb95f8d91a32a578c8.jpg)  
Figure 10: Full OSWorld comparison at a common token rate. Both plots use the same 54 configurations with token counts. The right plot replaces measured mean task time with mean recorded environment time plus generated tokens divided by 100 tokens/s. Tokens include reasoning and other output; scores and trajectories are unchanged. Point IDs refer to the OSWorld configuration tables.

## E.2 OSWORLD: PERFORMANCE–TIME AND PERFORMANCE–COST FRONTIERS

Figure 9 shows the full OSWorld comparison from Figure 2, with configuration IDs for all 56 points.

A common token-generation rate. Figure 10 compares measured task time with an estimate that assigns every model the same generation rate. We compute this estimate as recorded environment time plus generated tokens divided by 100 tokens/s, using the same trajectories and scores. Opus 5 (low) scores 87.6% in 85.9 s, while Astra (low) scores 90.8% in 89.5 s. Opus generates 1,795 tokens per task compared with 1,579 for Astra. With generation time assigned at the common rate, Astra is faster. Environment time includes agent-requested waits (Appendix D).

![](images/2a9f1449edd4ac34c57db27aa844127603d205d99f4fa0def1c7033144fb675e.jpg)

![](images/65ecabb64d79cc58e96ac292cdefe851d724717292262e140c00800e4fe67b6f.jpg)

Figure 11: Full OSWorld median-time view and fast-region detail. Mean verifier score versus median task time for all 56 configurations. For Astra (low), the median task times are averaged across five seeds. The right plot enlarges the region containing the fastest configurations; point IDs match the full OSWorld tables. Green dotted lines show the published human references of 72.36% and 111.94 s from the original 369-task OSWorld benchmark (Xie et al., 2024).  
![](images/2eb9e1029f1dd9fabd8d8ff8f6c863e9b2f8a9f0267d377be80a6cb2d28179eb.jpg)

![](images/afcdfabf87655f0e921479c34b7f431cd87503566f1ae6cc52ed6c07bf9747a5.jpg)  
Figure 12: Full OSWorld2 performance–time and performance–cost comparisons. All 21 configurations are evaluated on the same 52 tasks. Point IDs refer to the OSWorld2 configuration table. Marker shapes follow Figure 2; darker green indicates greater reasoning effort.

Median task time. Figure 11 reports median task time, which summarizes the time for a typical task.

## E.3 OSWORLD2: PERFORMANCE–TIME AND PERFORMANCE–COST FRONTIERS

Figure 12 compares all 21 OSWorld2 configurations. Astra defines the entire time frontier, while Gemini 3.8 Flash and Muse Spark 1.3 offer lower-cost choices on the cost frontier.

![](images/b2d944635435ccd8c24b39fa4358dba553beea9b48c958e4978f575cd749b41c.jpg)  
Figure 13: Joint performance–time–cost frontier on OSWorld2. Circled points are the evaluated frontier configurations. Other configurations appear in gray. Time and cost axes are logarithmic. Marker shapes follow Figure 2; darker green indicates greater reasoning effort.

## E.4 OSWORLD2: JOINT PERFORMANCE–TIME–COST FRONTIER

Figure 13 extends the joint frontier in Figure 6 to OSWorld2. A configuration is on this frontier if no other evaluated configuration improves one of the three metrics without worsening another. It includes Astra at all four reasoning settings, Gemini 3.8 Flash at all three settings, and Muse Spark 1.3 at minimal effort, allowing a choice based on the desired score and the available time and cost budgets.

(a) MyPCBench: time  
![](images/67eab4f296042f12e09eb1958dc325ef53a2188ee5e1c55c23a293930a4ef463.jpg)

(b) MyPCBench: cost  
![](images/28bf5d083b64ba0d9ab7278b63636fde61f8984820a557b48994f2d99796bd8e.jpg)

(c) CUA-World: time  
![](images/abab907189518759945c97cadab62d71ed32a1a1b031ff4d9e6f1b0941ac95da.jpg)

(d) CUA-World: cost  
![](images/64856464408fff8cfb8175097d86abbf8c0d61ef43f1bf1d9517c224275335fb.jpg)  
Figure 14: Performance–time and performance–cost frontiers on MyPCBench and CUA-World. GPT-6 Astra through Codex at four reasoning settings. Each point averages the same 38 MyPCBench tasks (top) or 26 CUA-World tasks (bottom). Darker green indicates greater reasoning effort. Dashed lines connect the observed Pareto-frontier points, which are circled.

<table><tr><td rowspan="2">Effort</td><td colspan="3">MyPCBench (38 tasks)</td><td colspan="3">CUA-World (26 tasks)</td></tr><tr><td>Score (%)</td><td>Time (s)</td><td>Cost ($)</td><td>Score (%)</td><td>Time (s)</td><td>Cost ($)</td></tr><tr><td>Low</td><td>89.9</td><td>514</td><td>4.57</td><td>90.3</td><td>807</td><td>9.29</td></tr><tr><td>Medium</td><td>88.3</td><td>495</td><td>4.33</td><td>86.7</td><td>810</td><td>8.42</td></tr><tr><td>High</td><td>88.6</td><td>568</td><td>4.81</td><td>94.2</td><td>1,108</td><td>11.11</td></tr><tr><td>Xhigh</td><td>93.6</td><td>619</td><td>4.81</td><td>93.0</td><td>1,515</td><td>15.06</td></tr></table>

Table 10: Reasoning effort on MyPCBench and CUA-World. GPT-6 Astra through Codex, reporting mean partial-credit score, time, and model cost per task. Bold indicates the highest score, lowest time, or lowest cost within each benchmark.

## E.5 REASONING EFFORT ON MYPCBENCH

Figure 14 and Table 10 compare GPT-6 Astra through Codex at four reasoning settings on MyPCBench. Xhigh achieves the highest score, 93.6%, while medium has the lowest mean time and cost, 495 s and \$4.33 per task. We evaluate the 38-task representative set from Sec. 2.2, with a limit of 100 action batches. All reported means include unsuccessful tasks.

## E.6 REASONING EFFORT ON CUA-WORLD

Figure 14 and Table 10 show a different outcome on CUA-World. High effort scores 94.2%, compared with 93.0% at xhigh, while taking less time (1,108 vs. 1,515 s) and costing less (\$11.11 vs. \$15.06 per task). All four settings use Astra through Codex on the same 26 long-horizon tasks, with a limit of 500 action batches.

<table><tr><td>Run</td><td>Score (%)</td><td>Time (s)</td><td>Agent (s)</td><td>Steps</td><td>Env. (s)</td><td>Waits (s)</td><td>Env.—waits (s)</td></tr><tr><td>1</td><td>87.62</td><td>86.11</td><td>72.90</td><td>5.90</td><td>13.209</td><td>1.328</td><td>11.881</td></tr><tr><td>2</td><td>91.62</td><td>89.56</td><td>75.31</td><td>5.82</td><td>14.248</td><td>1.082</td><td>13.166</td></tr><tr><td>3</td><td>91.62</td><td>90.13</td><td>76.52</td><td>5.88</td><td>13.606</td><td>1.040</td><td>12.566</td></tr><tr><td>4</td><td>91.62</td><td>89.81</td><td>75.78</td><td>5.78</td><td>14.033</td><td>1.152</td><td>12.881</td></tr><tr><td>5</td><td>91.62</td><td>92.10</td><td>78.19</td><td>5.94</td><td>13.914</td><td>1.012</td><td>12.902</td></tr><tr><td>Mean</td><td>90.82</td><td>89.54</td><td>75.74</td><td>5.86</td><td>13.802</td><td>1.123</td><td>12.679</td></tr><tr><td>SD</td><td>1.79</td><td>2.16</td><td>1.93</td><td>0.06</td><td>0.405</td><td>0.126</td><td>0.494</td></tr></table>

Table 11: Variation across five seeds for GPT-6 Astra at low reasoning effort. Each row averages all 50 OSWorld tasks. SD is the sample standard deviation of these means across the five runs; score SD is in percentage points. Steps count action batches, and waits are explicitly requested by the agent.

## F RUN-TO-RUN VARIATION

We repeat the evaluation across five seeds to measure variation in scores, task times, and action sequences. We also compare scores on individual tasks to determine whether the same tasks succeed across runs. Appendix B.6 reports the corresponding timing results for Fast I/O.

## F.1 FIVE-SEED EVALUATION SETUP

We evaluate GPT-6 Astra at low reasoning effort using Codex on the same 50 OSWorld tasks. Each run uses the same agent prompt and a 500-step limit. Only the task seed changes.

## F.2 SCORES AND TIMES ACROSS SEEDS

Table 11 shows that both score and task time vary little across seeds. Mean task time is 89.54 ± 2.16 s, and the score is $9 0 . 8 2 \pm 1 . 7 9 \%$ (mean ± standard deviation). The mean number of action batches is also stable at $5 . 8 6 \pm 0 . 0 6$ per task. Timing components are defined in Appendix D.

## F.3 TASK-LEVEL SCORE AGREEMENT

<table><tr><td>I/O</td><td>k</td><td>Pass@k (%) Best-of-k partial score (%)</td></tr><tr><td rowspan="5">Standard I/O</td><td>1</td><td>87.20 90.82</td></tr><tr><td>2</td><td>88.00 91.62</td></tr><tr><td>3</td><td>88.00 91.62</td></tr><tr><td>4</td><td>88.00 91.62</td></tr><tr><td>5</td><td>88.00 91.62</td></tr><tr><td rowspan="5">Fast I/O</td><td>1</td><td>86.80</td><td>90.42</td></tr><tr><td>2</td><td>88.80</td><td>92.42</td></tr><tr><td>3</td><td>89.20</td><td>92.82</td></tr><tr><td>4</td><td>89.60</td><td>93.22</td></tr><tr><td>5</td><td>90.00</td><td>93.62</td></tr></table>

Table 12: Task-level repeatability across five seeds of Astra at low effort. Values average uniformly over all size-k subsets of the five runs, then over the 50 tasks. Pass@k measures whether any of the k attempts succeeds; best-of-k uses the highest partial score.

Table 12 shows that additional attempts improve performance only slightly. With standard I/O, 42 tasks receive full credit in all five runs, two in four runs, and six in none. Partial-credit scores are identical across all five runs on 48 of the 50 tasks. The low variation in mean score therefore reflects consistent performance on individual tasks: almost every task receives the same score on every attempt. With Fast I/O, 40 tasks receive full credit in all five runs, four in four runs, one in one run, and five in none. Partial-credit scores are identical on 45 tasks.

Pass@k and partial credit. For a task with c fully successful runs among the five, the probability that a uniformly selected set of k runs contains a success is $\displaystyle 1 - \left( { ^ 5 } _ { k } ^ { - c } \right) / \left( { ^ 5 } _ { k } ^ { - } \right)$ . Pass@k averages this quantity over tasks. For partial credit, the table averages the maximum task score within each size-k set. Standard-I/O pass@1 is 87.2% and pass@5 is 88.0%; its best-of-five partial score is 91.62%.

## F.4 VARIATION IN ACTION SEQUENCES

We next compare the number of action batches used for the same task across seeds. For each task, we compute the standard deviation and range across its five trajectories. With standard I/O, the median of these task-level standard deviations is 0.55 batches, and the median range is one batch. For Fast I/O, the corresponding values are 0.99 and two batches.

Two ways to complete the same spreadsheet task. For example, on the Calc task that cleans movie titles, all five standard-I/O runs receive full credit while taking 7, 6, 5, 6, and 2 batches. The seven-batch trajectory enters separate PROPER(TRIM(...)) formulas for successive rows and revisits the paste-special dialog. The two-batch trajectory selects C2:C29, enters one formula with Alt+Enter to fill the selection, then saves. The two trajectories take 97.4 and 40.3 s, respectively. Here, the agent reduces the number of batches by applying the formula to all rows at once.

## G REPRESENTATIVE TASK SELECTION

We select a small set of tasks from each benchmark to reduce evaluation time while preserving the scores and relative performance of agents. This section gives the selection procedure from Sec. 2.2 and tests its score estimates, rankings, and speed–performance frontier on held-out models. Table 13 lists the resulting task counts.

<table><tr><td>Benchmark</td><td>Full set</td><td>Selected subset</td></tr><tr><td>OSWorld (Xie et al., 2024)</td><td>295</td><td>50</td></tr><tr><td>OSWorld2 (Yuan et al., 2026)</td><td>63</td><td>52</td></tr><tr><td>CUA-World (Aggarwal et al., 2026)</td><td>143</td><td>26</td></tr><tr><td>MyPCBench (Jang et al., 2026a)</td><td>184</td><td>38</td></tr></table>

Table 13: Task counts for the representative subsets selected for cua-speedrun.

## G.1 MODELS AND TASKS USED FOR SELECTION

OSWorld. We use nine configurations evaluated on the same 295 tasks: Claude Sonnet 5 xhigh, Gemini 3.6 Flash, Gemini 3 Flash Preview, GLM-5V Turbo, GPT-5.6 Luna xhigh, Kimi K3, Meta Muse Spark, MiniMax M3, and Qwen3.5-9B-Thinking. Each task is represented by its nine partial scores and nine exact-completion indicators. The internet-dependence audit defining this task population is described in Appendix H.

OSWorld2. Selection uses seven configurations: MiniMax-M3, Claude Opus 4.7, Claude Sonnet 4.6 at medium and maximum effort, GPT-5.5, GPT-5.6, and Qwen3.7. The resulting evaluation set contains 52 tasks.

CUA-World. Selection uses seven configurations with scores on more than 30 tasks each: Qwen3- VL-2B-Thinking, GPT-5.4 through Azure, GPT-5.4, Claude Opus 4.7, Claude Sonnet 4.6, Gemini 3 Flash Preview, and Kimi K2.5. Selection uses each model’s distribution of observed partial scores. Repeated runs remain separate observations. The selected set contains 26 tasks.

MyPCBench. The evaluation uses 38 of the 184 canonical tasks, selected with the unweighted energy method and 100 initializations. The task identities for all four benchmarks are listed in Appendix G.6.

## G.2 INITIALIZATION, OPTIMIZATION, AND TASK COUNT

We seek a subset that represents the full benchmark’s distribution of score patterns. For OSWorld and OSWorld2, let $z _ { i }$ concatenate the partial scores and exact-completion indicators for task i. Distances are $d _ { i j } = \lVert z _ { i } - z _ { j } \rVert _ { 2 }$ . For a subset S of size K from N tasks, the normalized energy objective is

$$
E ( S ) = { \frac { 1 } { \bar { d } } } \left[ { \frac { 2 } { K N } } \sum _ { i \in S } \sum _ { j = 1 } ^ { N } d _ { i j } - { \frac { 1 } { K ^ { 2 } } } \sum _ { i , j \in S } d _ { i j } - { \frac { 1 } { N ^ { 2 } } } \sum _ { i , j = 1 } ^ { N } d _ { i j } \right] , \qquad { \bar { d } } = { \frac { \sum _ { i \not = j } d _ { i j } } { N ( N - 1 ) } } .\tag{5}
$$

We estimate a model’s benchmark score by its unweighted mean score on the selected tasks.

Balanced initializations. To cover a range of score profiles in each initial subset, we sort tasks by the leading principal coordinate of their centered score profiles and choose evenly spaced tasks from this ordering. Specifically, for a uniformly sampled phase $\phi \in [ 0 , 1 )$ , the initial positions are $\left\lfloor ( \phi + t ) N / \bar { K } \right\rfloor$ for $t = 0 , \ldots , K - 1$ . We use 100 initializations with seeds $7 0 0 0 , \ldots , 7 0 9 9$ optimizing each distinct initial subset once.

One-swap optimization. At each iteration, we evaluate every exchange of a selected task with an unselected task and apply the exchange that reduces energy the most. We stop when no exchange improves the objective beyond numerical tolerance and retain the lowest-energy subset across the initializations.

for each candidate task count K:   
for each model h:   
remove model h from the calibration models   
form task profiles from the remaining models   
optimize 100 balanced initial subsets by one-task swaps   
select the subset with the lowest calibration energy   
predict h’s partial and exact scores by subset means   
compare predictions with each model’s full-set means   
choose the smallest K for which both rank correlations   
are at least 0.95 at K-1, K, and K+1   
refit the K-task subset using all models

To choose the number of tasks, we repeat the full selection procedure with each model held out in turn (Listing 3). We construct task profiles and select tasks using the remaining models, then compare the held-out model’s subset score with its full-set score. We choose the smallest K for which both partial-score and exact-completion rank correlations are at least 0.95 at K − 1, K, and K + 1. Checking the neighboring task counts tests the stability of the ranking criterion. Each task count is optimized independently. After choosing K, we select the final evaluation set using all models.

Selection with missing scores. Some CUA-World models have scores on only part of the benchmark. We therefore compare each model’s distribution of observed partial scores on selected tasks with its distribution over all observed tasks, using one-dimensional energy distance. We first ensure that the subset contains an observation for every model used for selection, then minimize the mean of these distances. The task-count rule uses partial-score rank correlation at K − 1, K, K + 1. At K = 25, 26, 27, the seven-model held-out correlations are 0.964, 1.000, and 1.000, respectively; the corresponding score errors are 7.43, 5.12, and 2.10 percentage points. These errors compare unweighted means of observed run scores.

![](images/da7804eb678525f4811a5332e1f41fcd89eba20fb62526b1d7f7ab9b97131fdb.jpg)

![](images/9e90923a23787bc31ab99a98ce3f4e846ae5caea2e6b9a7fab5dc40f8a797b64.jpg)

![](images/c9aa4362323d57b7591b12b15abd735c96e5cba2e356dd91a43f1f5e7e7248ca.jpg)

![](images/910fe1f52b88e24822b136ec4c457384803619f210878a5d03d6a2febbb2d510.jpg)  
Figure 15: Score error and rank correlation on held-out models. OSWorld uses nine leave-oneagent-out folds; OSWorld2 uses seven. The randomized baseline averages 300 difficulty-stratified selections per task count. Dotted vertical lines mark 50 and 52 tasks; horizontal lines mark rank correlation 0.95. Each task count is optimized independently.

## G.3 HELD-OUT SCORE AND RANKING ACCURACY

Figure 15 compares energy-based selection with a randomized baseline that samples tasks across difficulty levels. We measure error as the mean absolute difference between each held-out model’s subset estimate and its full-set result. At 50 OSWorld tasks, energy-based selection reduces partialscore error from 4.02 to 2.16 percentage points. It also reduces mean-task-time error from 51.0 to 31.4 s, even though selection uses only scores. Thus, selecting tasks with representative score patterns also improves estimates of execution time.

<table><tr><td>Benchmark</td><td>K</td><td>Partial  $\rho$ </td><td>Exact  $\rho$ </td><td>Partial MAE</td><td>Exact MAE</td></tr><tr><td rowspan="3">OSWorld</td><td>49</td><td>0.967</td><td>0.967</td><td>4.36</td><td>4.57</td></tr><tr><td>50</td><td>0.983</td><td>0.975</td><td>2.16</td><td>2.17</td></tr><tr><td>51</td><td>1.000</td><td>0.992</td><td>4.21</td><td>3.95</td></tr><tr><td rowspan="3">OSWorld2</td><td>51</td><td>0.964</td><td>0.955</td><td>1.70</td><td>2.28</td></tr><tr><td>52</td><td>0.964</td><td>0.955</td><td>1.33</td><td>2.03</td></tr><tr><td>53</td><td>0.964</td><td>0.982</td><td>0.74</td><td>1.04</td></tr></table>

Table 14: Held-out-model validation around the selected task count. MAE is in percentage points. Each model is excluded from task selection in its fold. Both correlations exceed 0.95 at the selected task count and its two neighbors.

Table 14 shows that the selected task sets closely preserve model rankings on both benchmarks. At 50 tasks, OSWorld achieves partial-score and exact-completion rank correlations of 0.983 and 0.975; at 52 tasks, OSWorld2 achieves 0.964 and 0.955. Both correlations remain above 0.95 at the neighboring task counts.

![](images/14807af253cb1a3f7933dcfccc5d9aae9d78eed8b5367b480c07e90d87de4300.jpg)  
Figure 16: Speed–performance frontiers from full-set scores and held-out subset estimates. Dark points and connecting lines mark nondominated configurations. Model IDs are: 1, Sonnet 5; 2, Gemini 3.6 Flash; 3, Gemini 3 Flash Preview; 4, GLM-5V Turbo; 5, Luna; 6, Kimi K3; 7, Muse Spark; 8, MiniMax M3; 9, Qwen3.5-9B. Each point in (b) uses a 50-task subset selected with that model held out.

## G.4 PRESERVING THE SPEED–PERFORMANCE FRONTIER

Figure 16 compares the full-set frontier with the frontier estimated from 50-task subsets for held-out models. The estimated frontier recovers two of the three full-set frontier configurations and adds no others, giving 100% precision and 66.7% recall. Pairwise dominance agrees for 91.7% of model pairs. This analysis uses the models’ scores and times; selection itself uses only scores.

Evaluation savings. Selecting 50 of the 295 OSWorld tasks reduces the number of tasks by 5.9×. Across the nine models, these tasks account for 15.1% of the total recorded task time, a 6.61× reduction. We compute this saving by summing trajectory times, excluding setup and verification.

<table><tr><td>Method</td><td>Partial MAE</td><td>Exact MAE</td><td>Mean MAE</td></tr><tr><td>Uniform random</td><td>5.88</td><td>6.01</td><td>5.95</td></tr><tr><td>Difficulty-stratified random</td><td>5.05</td><td>5.11</td><td>5.08</td></tr><tr><td>IRT-inspired selection</td><td>7.94</td><td>8.21</td><td>8.08</td></tr><tr><td>Energy-based selection</td><td>4.50</td><td>4.34</td><td>4.42</td></tr></table>

Table 15: Task-selection methods on OSWorld with 32 tasks. Errors are averaged across nine leave-one-agent-out folds and reported in percentage points. Mean MAE averages partial-score and exact-completion MAE. Random baselines average 300 selections per fold.

## G.5 COMPARING TASK-SELECTION METHODS

Table 15 compares task-selection methods at a budget of 32 OSWorld tasks, using the same nine leave-one-agent-out folds. Energy-based selection achieves a mean error of 4.42 percentage points, compared to 5.08 for difficulty-stratified random selection and 8.08 for the IRT-inspired method.

We use energy-based selection for its simple scoring rule: each agent’s benchmark score is the average of its scores on the selected tasks. The IRT-inspired method fits a one-parameter ability model using the eight agents available in each fold and predicts full-benchmark performance from the selected-task scores.

## G.6 SELECTED TASK IDENTITIES

The following lists give the final task sets used in our evaluations. OSWorld IDs use the original task UUIDs. OSWorld2 uses its three-digit identifiers; MyPCBench IDs combine the task family and suffix; CUA-World IDs combine the environment and task name.

## G.6.1 OSWORLD

Table 16: Selected OSWorld task identities.
<table><tr><td colspan="4" rowspan="1">Application                     Task UUID</td></tr><tr><td colspan="4" rowspan="1">chrome                          06fe7178-4491-4589-810f-2e2bc9502122</td></tr><tr><td colspan="4" rowspan="1">chrome                          2ae9ba84-3a0d-4d4c-8338-3a1478dc5fe3</td></tr><tr><td colspan="4" rowspan="1">chrome                          3720f614-37fd-4d04-8a6b-76f54f8c222d</td></tr><tr><td colspan="4" rowspan="1">chrome                          44ee5668-ecd5-4366-a6ce-c1c9b8d4e938</td></tr><tr><td colspan="4" rowspan="1">chrome                          af630914-714e-4a24-a7bb-f9af687d3b91</td></tr><tr><td colspan="4" rowspan="1">gimp                            045bf3ff-9077-4b86-b483-a1040a949cff</td></tr><tr><td colspan="4" rowspan="1">gimp                            554785e9-4523-4e7a-b8e1-8016f565f56a</td></tr><tr><td colspan="4" rowspan="1">gimp                            734d6579-c07d-47a8-9ae2-13339795476b</td></tr><tr><td colspan="2" rowspan="1">gimp</td><td colspan="1" rowspan="1">a746add2-cab0-4740-ac36-c3769d9bfb46</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="2" rowspan="1">libreoffice_calc</td><td colspan="1" rowspan="1">0326d92d-d218-48a8-9ca1-981cd6d064c7</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="2" rowspan="1">libreoffice_calc</td><td colspan="1" rowspan="1">1de60575-bb6e-4c3d-9e6a-2fa699f9f197</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="2" rowspan="1">libreoffice_calc</td><td colspan="1" rowspan="1">1e8df695-bd1b-45b3-b557-e7d599cf7597</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="2" rowspan="1">libreoffice_calc</td><td colspan="1" rowspan="1">3a7c8185-25c1-4941-bd7b-96e823c9f21f</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="2" rowspan="1">libreoffice_calc</td><td colspan="1" rowspan="1">4de54231-e4b5-49e3-b2ba-61a0bec721c0</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="2" rowspan="1">libreoffice_calc</td><td colspan="1" rowspan="1">535364ea-05bd-46ea-9937-9f55c68507e8</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="2" rowspan="1">libreoffice_calc</td><td colspan="1" rowspan="1">7efeb4b1-3d19-4762-b163-63328d66303b</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">libreoffice_calc</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">8b1ce5f2-59d2-4dcc-b0b0-666a714b9a14</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">libreoffice_calc</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">a9f325aa-8c05-4e4f-8341-9e4358565f4f</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">libreoffice_calc</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">abed40dc-063f-4598-8ba5-9fe749c0615d</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">libreoffice_impress</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">39be0d19-634d-4475-8768-09c130f5425d</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">libreoffice_impress</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">455d3c66-7dc6-4537-a39a-36d3e9119df7</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">libreoffice_impress</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">986fc832-6af2-417c-8845-9272b3a1528b</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">libreoffice_impress</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">a097acff-6266-4291-9fbd-137af7ecd439</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">libreoffice_impress</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">a434992a-89df-4577-925c-0c58b747f0f4</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">libreoffice_impress</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">af2d657a-e6b3-4c6a-9f67-9e3ed015974c</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">libreoffice_impress</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">b8adbc24-cef2-4b15-99d5-ecbe7ff445eb</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">libreoffice_writer</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">0810415c-bde4-4443-9047-d5f70165a697</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">libreoffice_writer</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">0e47de2a-32e0-456c-a366-8c607ef7a9d2</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">libreoffice_writer</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">72b810ef-4156-4d09-8f08-a0cf57e7cefe</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">libreoffice_writer</td><td colspan="1" rowspan="1"></td><td colspan="2" rowspan="2">ecc2413d-8a48-416e-a3a2-d30106ca36cb</td></tr><tr><td colspan="1" rowspan="1">libreoffice_writer</td><td colspan="1" rowspan="1"></td></tr><tr><td>Application</td><td>Task UUID</td></tr><tr><td>multi_apps</td><td>42f4d1c7-4521-4161-b646-0a8934e36081</td></tr><tr><td>multi_apps</td><td>48d05431-6cd5-4e76-82eb-12b60d823f7d</td></tr><tr><td>multi_apps</td><td>6d72aad6-187a-4392-a4c4-ed87269c51cf</td></tr><tr><td>multi_apps</td><td>81c425f5-78f3-4771-afd6-3d2973825947</td></tr><tr><td>multi_apps</td><td>d9b7c649-c975-4f53-88f5-940b29c47247</td></tr><tr><td>multi_apps</td><td>e135df7c-7687-4ac0-a5f0-76b74438b53e</td></tr><tr><td>multi_apps</td><td>f5c13cdd-205c-4719-a562-348ae5cd1d91</td></tr><tr><td>oS</td><td>b6781586-6346-41cd-935a-a6b1487918fc</td></tr><tr><td>thunderbird</td><td>7b1e1ff9-bb85-49be-b01d-d6424be18cd0</td></tr><tr><td>thunderbird</td><td>9b7bc335-06b5-4cd3-9119-1a649c478509</td></tr><tr><td>thunderbird</td><td>a1af9f1c-50d5-4bc3-a51e-4d9b425ff638</td></tr><tr><td>thunderbird</td><td>dfac9ee8-9bc4-4cdc-b465-4a4bfcd2f397</td></tr><tr><td>thunderbird</td><td>f201fbc3-44e6-46fc-bcaa-432f9815454c</td></tr><tr><td>vlc</td><td>cb130f0d-d36f-4302-9838-b3baf46139b6</td></tr><tr><td>vlc</td><td>d06f0d4d-2cd5-4ede-8de9-598629438c6e</td></tr><tr><td>vlc</td><td>fba2c100-79e8-42df-ae74-b592418d54f4</td></tr><tr><td>vs_code</td><td>0ed39f63-6049-43d4-ba4d-5fa2fe04a951</td></tr><tr><td>vs_code</td><td>5e2d93d8-8ad0-4435-b150-1692aacaa994</td></tr><tr><td>vs_code</td><td>ec71221e-ac43-46f9-89b8-ee7d80f7e1c5</td></tr></table>

## G.6.2 OSWORLD2

The selected task IDs are 001, 002, 010, 012, 013, 015, 018, 020, 021, 022, 028, 029, 030, 033, 038, 040, 042, 043, 044, 046, 047, 049, 051, 053, 054, 057, 058, 059, 061, 063, 065, 066, 068, 070, 071, 072, 076, 080, 085, 086, 088, 091, 094, 096, 100, 101, 102, 103, 104, 106, 107, 108.

## G.6.3 MYPCBENCH

<table><tr><td>Task family</td><td>Selected IDs</td></tr><tr><td>aggregation</td><td>f002, f005, f029, f033</td></tr><tr><td>contradiction</td><td>f015, f023</td></tr><tr><td>counterfactual</td><td>f001, f004, f008</td></tr><tr><td>cua_only</td><td>f004, f006, f018, f021, f023</td></tr><tr><td>hard_app</td><td>f011, f019, f022</td></tr><tr><td>long_horizon</td><td>f028, f043, f045, f047, f050, f060, f074</td></tr><tr><td>preference_inference</td><td>f004, f014, f021, f023, f025</td></tr><tr><td>retrieval</td><td>f005, f029</td></tr><tr><td>situated_action</td><td>f001, f003, f006, f017, f026, f035, f039</td></tr></table>

## G.6.4 CUA-WORLD

Table 17: Selected CUA-World task identities.
<table><tr><td>Environment</td><td>Task name</td></tr><tr><td>ardour_env</td><td>broadcast-podcast_stem_delivery</td></tr><tr><td>dhis2_env</td><td>rmncah_scorecard_dashboard</td></tr><tr><td>docker_desktop_env</td><td>diagnose_broken_microservices_stack</td></tr><tr><td>gpredict_env</td><td>poes_downlink_schedule_setup</td></tr><tr><td>gvsig_desktop_env</td><td>vulnerability_map_remote_communities</td></tr><tr><td>jstock_env</td><td>quarterly-portfolio_rebalance</td></tr><tr><td>librehealth_ehr_env</td><td>implement_lab_workflow_and-process-patient</td></tr><tr><td>moodle_env</td><td>configure_tiered_assessment-pathway</td></tr><tr><td>nextgen_connect_integration_ engine_env</td><td>adt_census_lab_validation_pipeline</td></tr><tr><td>nosh_env</td><td>care_quality-remediation</td></tr><tr><td>odoo_inventory_env</td><td>pharma_lot_recall_quarantine</td></tr><tr><td>openclinic-ga_env</td><td>insured_consultation_billing</td></tr><tr><td>oracle_database_env</td><td>claims-pipeline_reconciliation</td></tr><tr><td>project_libre_env</td><td>schedule_recovery_rebaseline</td></tr><tr><td>pymol_env</td><td>kinase_selectivity_comparison</td></tr><tr><td>redmine_env</td><td>q1_milestone_reconciliation</td></tr><tr><td>rocket_chat_env</td><td>compliance_audit_remediation</td></tr><tr><td>snap_env</td><td>multicriteria_suitability-mapping</td></tr><tr><td>splunk_env</td><td>threat_intel_enrichment_pipeline</td></tr><tr><td>sumo_env</td><td>optimize_network_signal_timing</td></tr><tr><td>thunderbird_env</td><td>litigation_email_triage</td></tr><tr><td>wireshark_env</td><td>web_app_breach_investigation</td></tr><tr><td>wondershare_edrawmax_env</td><td>healthcare_it_architecture_review</td></tr><tr><td>woo_commerce_env</td><td>launch_coffee-product_line</td></tr><tr><td>wordpress_env</td><td>launch_woocommerce_coffee_roastery</td></tr><tr><td>wps_presentation_env</td><td>rebrand_restructure-pitch_deck</td></tr></table>

## H INTERNET-DEPENDENCE AUDIT

Live public services can change their content, become unavailable, or restrict access between evaluations. We therefore audit whether each task requires the agent to use such services. Tasks that use a local application or a benchmark-controlled website remain eligible, as do tasks that only require downloading fixed files during setup.

## H.1 INCLUSION RULE

We first use a coding agent to classify the 369 OSWorld tasks as local/offline, benign internet, or antibot/access-sensitive. Three human reviewers then independently assess whether each task requires interaction with a live public service. We include a task only when all three reviewers agree to include it. This gives 295 tasks.

<table><tr><td>Reviewer or outcome</td><td>Included</td><td>Excluded</td></tr><tr><td>Reviewer 1</td><td>297</td><td>72</td></tr><tr><td>Reviewer 2</td><td>302</td><td>67</td></tr><tr><td>Reviewer 3</td><td>295</td><td>74</td></tr><tr><td>Unanimous decision</td><td>295</td><td>67</td></tr><tr><td>Disagreement</td><td>7</td><td></td></tr></table>

Table 18: Independent inclusion decisions for 369 OSWorld tasks. The seven disagreements are excluded by the unanimous-inclusion rule.

## H.2 REVIEWER AGREEMENT

Table 18 shows that 362 of 369 tasks receive a unanimous decision. Pairwise agreement is 98.64%, 99.46%, and 98.10% for reviewer pairs 1–2, 1–3, and 2–3, respectively. Their Cohen’s κ values are 0.956, 0.983, and 0.939; Fleiss’ κ across all three reviewers is 0.959.

Applying the criterion across benchmarks. We apply the same criterion to the other benchmarks. OSWorld2 provides benchmark-controlled web applications, and MyPCBench provides local, preauthenticated services, so these tasks remain eligible. For CUA-World, we exclude tasks requiring internet access during agent interaction while retaining those that only use the network for setup.

## I VALIDATING ACTIONS WITH CUA-AUTODEBUG

A task can fail because the model chooses an incorrect action or because the infrastructure executes a correct action incorrectly. We developed CUA-AutoDebug to distinguish these errors. A controlled desktop application records the keyboard and mouse inputs it receives, including modifier keys and typed text. We also check the resulting application state: a click should reach the intended target, a drag should move the object, and typed text should match the requested string. For shortcuts handled by the operating system, we record input events before they reach the application.

## I.1 SEPARATING MODEL, HARNESS, AND RUNTIME ERRORS

We first submit predefined actions directly to the VM runtime. We then test agent harnesses by giving the model a short instruction, such as clicking a colored square, and comparing its response with the executed action and received input. An incorrect model action is a model error; incorrect translation of a correct action is a harness error.

## I.2 INPUT FAILURES DETECTED BY THE RUNTIME TESTS

<table><tr><td>Requested input</td><td>Received input</td><td>Cause</td></tr><tr><td>Type &lt;</td><td>&gt;</td><td>Incorrect X11 key/modifier map- ping</td></tr><tr><td>Type accented text</td><td>Accented characters omitted</td><td>Characters absent from the key map</td></tr><tr><td>able</td><td>Type a literal shell vari- Variable expanded into a path</td><td>Text interpreted by the command shell</td></tr><tr><td>Keypad Enter or Menu</td><td>No corresponding key event</td><td>Missing key-name translation</td></tr><tr><td>Toggle Caps Lock, then Lowercase text type</td><td></td><td>Caps Lock press silently omitted</td></tr></table>

Table 19: Input errors detected by CUA-AutoDebug when executing actions through SSH and PyAutoGUI. The tests compare requested input with the keyboard and mouse events received.

Table 19 shows examples of input errors detected by these tests, such as typing > when < was requested or silently dropping accented characters. We test a catalog of 100 cases covering clicks, drags, scrolls, key presses, key combinations, text entry, and sequences of these actions. Six cases are unsupported by the tested action interface. We repeat each of the remaining 94 cases five times; 83 pass and 11 fail when executed through SSH and PyAutoGUI. The failures include one shell-expansion case, one shifted-symbol case, five Unicode cases, and four named-key cases.

The corrected runtime passes all 470 tests. We correct these errors in CUA-Speedrun’s QEMU runtime by preserving literal text when sending commands, explicitly mapping key names and modifiers, and supporting characters absent from the default keyboard mapping. After these corrections, all 94 supported cases pass in all five repetitions.

## I.3 FAILURES IN AGENT HARNESSES

<table><tr><td>Harness</td><td>Model response</td><td>Executed behavior</td></tr><tr><td>Gemini</td><td>Scroll down by five wheel clicks</td><td>Scroll up by 600 ticks</td></tr><tr><td>Gemini</td><td>Press F5 or Page Down</td><td>Type the key name as text</td></tr><tr><td>Qwen3.5</td><td>Middle-click the target</td><td>No action</td></tr><tr><td>Qwen3.5</td><td>Ctrl-click the target</td><td>Release Ctrl before clicking</td></tr><tr><td>Qwen3.5</td><td>Move, then scroll or drag</td><td>Execute only the first tool call</td></tr></table>

Table 20: Harness errors can change a correct model action. Examples detected by comparing model responses with the actions executed and the input events received.

Table 20 shows that harness errors can prevent a correct model response from reaching the application. For example, the Gemini harness reversed scroll direction and interpreted a fallback scroll distance in pixels as a number of wheel ticks. All six scroll-direction test repetitions failed before correction and passed afterward. The Qwen3.5 harness released the modifier key before a click and silently discarded tool calls after the first. A model that described a triple-click but emitted a double-click produced a model error.

## J EXPERIMENTAL DETAILS

Appendix E.1 lists the model, harness, and reasoning setting for each configuration.

## J.1 TASK DEFINITION

We define a computer-use task as $x _ { i } = ( E _ { i } , s _ { i } ^ { 0 } , p _ { i } , V _ { i } )$ , where $E _ { i }$ is an interactive environment with initial state $\bar { s _ { i } ^ { 0 } } , p _ { i }$ is a natural-language instruction, and $V _ { i }$ is a verification function. An agent π receives $p _ { i }$ and a sequence of observations and produces computer actions until it terminates or reaches the task limit. This interaction produces a trajectory $\tau _ { i }$ and final state $s _ { i } ^ { T }$ , with score $r _ { i } = V _ { i } ( \tau _ { i } , s _ { i } ^ { T } )$ . A benchmark $B = \{ x _ { i } \} _ { i = 1 } ^ { N }$ is a collection of such tasks.

Agent implementations. We connect each model’s public reference agent or native computer-use API to the common desktop interface.

## J.2 OBSERVATIONS, ACTIONS, AND INTERACTION HISTORY

The environment provides $1 9 2 0 \times 1 0 8 0$ screenshots of the desktop. Mouse actions use pixel coordinates in these images. Harnesses that resize screenshots or use normalized coordinates convert the model’s coordinates to desktop pixels before executing an action.

<table><tr><td>Action family</td><td>Operations</td></tr><tr><td>Mouse clicks</td><td>Left, right, middle, double, and triple click at a coordinate</td></tr><tr><td>Pointer motion</td><td>Move, drag along coordinates, hold or release a mouse button</td></tr><tr><td>Scrolling</td><td>Signed vertical scroll at the current pointer position</td></tr><tr><td>Keyboard</td><td>Type text, press a key or chord, hold and release modifiers</td></tr><tr><td>Waiting</td><td>Wait for an agent-specified duration</td></tr><tr><td>Observation</td><td>Request a screenshot without changing the desktop</td></tr><tr><td>Completion</td><td>Stop interaction and invoke the benchmark verifier</td></tr></table>

Table 21: Desktop actions supported by the infrastructure. A batch contains one or more actions executed in order. The model-facing tool schema can differ between harnesses.

Table 21 lists the actions available to agents. Direct-API agents translate the model’s computer-use responses into these actions and return screenshots in the provider’s tool-result format. Codex uses the same actions from a separate agent sandbox. We instruct Codex to interact with the task desktop only through the supplied interface: its local shell and files belong to the agent sandbox (Listing 4).

Listing 4: Core desktop-isolation instruction in the Codex prompt.   
You are operating a remote computer in a separate VM.   
Your only interface to that computer is the HTTP proxy at   
\$CS\_COMPUTER\_URL.   
Authenticate every request with the header X-Gateway-Token:   
\$CS\_COMPUTER\_TOKEN.   
Your local shell and files are in the agent sandbox, not the task VM.   
Do not bypass the proxy to access the task computer.

Interaction history. Each harness determines which screenshots and previous responses remain in the model’s context. Kimi’s harness retains the current screenshot and the two most recent completedturn screenshots. Earlier turns are compacted into textual history; recent responses retain their reasoning fields and tool calls. Codex maintains its own interaction history and explicitly retrieves screenshots.

Reasoning settings. We vary reasoning effort within each model using the settings exposed by its provider, such as low and high.

## J.3 TASK LIMITS AND EXECUTION

Each evaluation fixes the task set, agent implementation, reasoning setting, environment resources, and number of parallel tasks. Every task starts from its benchmark-specified initial state. We measure task time from when the instruction is given to the agent until it terminates or reaches the task limit, excluding setup, initialization, and verification (Sec. 2.1).

Time and action limits. The OSWorld and OSWorld2 configurations in Appendix E.1 have an environment deadline of 39,600 s per task, while the MyPCBench Astra sweep uses 7,200 s. These deadlines apply separately from the harness’s action or model-turn limit and the timeout on an individual model request.

We allow 500 action batches per task in the Astra batching comparison, five-seed experiments, and CUA-World sweep, and 100 in the MyPCBench sweep. A batch can contain several actions before the next observation; the single-action Astra ablation requires a screenshot after every action (Appendix K.1). In the Fast I/O comparison, we keep the model and reasoning setting fixed and change the action and screenshot implementation (Appendix B.3).

## J.4 BENCHMARK VERIFICATION

We use each benchmark’s verification procedure. OSWorld and OSWorld2 score task completion with task-specific programmatic checks. CUA-World uses Gemini 3 Flash Preview to assess the recorded trajectory against its visual checklist. MyPCBench uses its visual trajectory judge and averages the rubric scores with equal weight; a task receives full credit only when every rubric passes. We also retain each benchmark’s task instructions and context, including MyPCBench’s persona and pre-authenticated local applications, alongside the agent’s system prompt.

## J.5 MODEL COST

We compute model cost by summing the cost of each task’s model requests. Each request’s cost comes from the provider’s recorded charge or its recorded token usage at the applicable API prices. Generated tokens follow the accounting in Appendix D.2.

Model cost excludes environment hosting, model-server rental, setup, and separate verification calls.

## K ADDITIONAL AGENT ANALYSES

We examine how action batching, agent harnesses, and reasoning effort affect task time through comparisons within the same model. Action batches, individual actions, and generated tokens are defined in Appendix D.

## K.1 CONTROLLED BATCHING COMPARISONS

<table><tr><td>Model</td><td>Effort</td><td>Interaction rule</td><td>Score</td><td>Time</td><td>Batches</td><td>Actions</td><td>Tokens</td></tr><tr><td>Astra</td><td>xhigh</td><td>Batched</td><td>91.62</td><td>126.82</td><td>7.14</td><td>19.90</td><td>2,060</td></tr><tr><td>Astra</td><td>xhigh</td><td>One action + image</td><td>91.62</td><td>182.73</td><td>16.50</td><td>16.50</td><td>2,559</td></tr><tr><td>Kimi K3</td><td>low</td><td>Single tool</td><td>71.62</td><td>176.61</td><td>8.12</td><td>55.10</td><td>3,171</td></tr><tr><td>Kimi K3</td><td>low</td><td>Batched tools</td><td>73.62</td><td>149.50</td><td>5.74</td><td>30.00</td><td>1,865</td></tr><tr><td>Kimi K3</td><td>high</td><td>Single tool</td><td>81.62</td><td>305.07</td><td>8.92</td><td>113.60</td><td>7,567</td></tr><tr><td>Kimi K3</td><td>high</td><td>Batched tools</td><td>79.62</td><td>358.13</td><td>8.76</td><td>60.20</td><td>7,125</td></tr><tr><td>Kimi K3</td><td>max</td><td>Single tool</td><td>85.62</td><td>606.27</td><td>12.18</td><td>61.24</td><td>14,359</td></tr><tr><td>Kimi K3</td><td>max</td><td>Batched tools</td><td>89.62</td><td>443.74</td><td>11.70</td><td>53.58</td><td>10,435</td></tr></table>

Table 22: Batching comparisons within the same model and reasoning effort on OSWorld. Score is a percentage; time is in seconds. All counts and times are means over the same 50 tasks. Tokens include reasoning and other output.

Batching makes Astra faster at the same score. Table 22 compares Astra xhigh with its standard batched actions and a restriction to one action between screenshots. Both settings score 91.62%, but the single-action setting takes 182.7 s per task compared with 126.8 s for batching. It executes fewer individual actions (16.50 versus 19.90), yet requires more action batches (16.50 versus 7.14) and generates more tokens (2,559 versus 2,060). We keep the model, reasoning effort, 50-task set, and configured limits fixed. To enforce the single-action restriction, we reject multi-action requests and prevent further actions until the agent retrieves a new screenshot, including actions issued in separate back-to-back requests.

Kimi benefits from batched tools at some reasoning settings. Table 22 also compares Kimi at matched low, high, and maximum effort. Kimi’s single-tool setting disables parallel tool calls and allows multiple GUI actions within one code block. At maximum effort, batched tools reduce mean task time from 606.3 to 443.7 s and improve the score from 85.62% to 89.62%. At high effort, batching increases time from 305.1 to 358.1 s while reducing the score from 81.62% to 79.62%.

![](images/109d5654ab572beb6b4de88f6a580257c947d3d54a21e03181118629cbabc983.jpg)  
Figure 17: Harness performance depends on reasoning effort. GPT-5.6 Luna at low, medium, and high reasoning effort, using the direct-API agent or Codex on OSWorld. Each curve connects reasoning settings within one harness. All six evaluations use the same 50-task set.

## K.2 HARNESS COMPARISONS ACROSS REASONING EFFORTS

Figure 17 compares GPT-5.6 Luna with Codex and the direct-API agent at low, medium, and high effort. At low effort, Codex improves the score from 61.8% to 75.6% but increases mean task time from 74 to 145 s. At medium and high effort, the direct-API agent is both more accurate and faster: it scores 75.6% in 119 s compared with 67.6% in 189 s at medium, and 81.6% in 161 s compared with 79.6% in 325 s at high.

K.3 COMPARING REASONING EFFORTS ON THE SAME SUCCESSFUL TASKS
<table><tr><td>Model</td><td>Tasks</td><td>Effort</td><td>Time</td><td>Batches</td><td>Actions</td><td>Tokens</td><td>Tok/s</td></tr><tr><td rowspan="3">Gemini 3.8 Flash</td><td rowspan="3">5</td><td>low</td><td>1291.0</td><td>116.40</td><td>171.60</td><td>29,596</td><td>47.57</td></tr><tr><td>medium</td><td>1363.9</td><td>112.60</td><td>158.40</td><td>34,446</td><td>43.10</td></tr><tr><td>high</td><td>3024.5</td><td>190.60</td><td>246.60</td><td>73,578</td><td>42.18</td></tr><tr><td rowspan="4">GPT-6 Astra</td><td rowspan="4">13</td><td>low</td><td>791.8</td><td>47.08</td><td>146.69</td><td>7,878</td><td>15.18</td></tr><tr><td>medium</td><td>752.6</td><td>47.15</td><td>138.85</td><td>8,376</td><td>17.07</td></tr><tr><td>high</td><td>817.3</td><td>53.15</td><td>171.23</td><td>11,001</td><td>21.31</td></tr><tr><td>xhigh</td><td>1131.6</td><td>55.38</td><td>223.46</td><td>14,359</td><td>18.04</td></tr><tr><td rowspan="4">Muse Spark 1.3</td><td rowspan="4">2</td><td>minimal</td><td>833.1</td><td>82.00</td><td>195.00</td><td>7,521</td><td>15.04</td></tr><tr><td>low</td><td>1183.3</td><td>99.50</td><td>291.00</td><td>12,997</td><td>16.79</td></tr><tr><td>medium</td><td>1008.8</td><td>78.50</td><td>235.00</td><td>14,621</td><td>21.55</td></tr><tr><td>high</td><td>907.3</td><td>68.50</td><td>246.00</td><td>14,095</td><td>22.87</td></tr><tr><td></td><td></td><td>xhigh</td><td>1546.2</td><td>109.50</td><td>400.00</td><td>23,362</td><td>21.77</td></tr></table>

Table 23: Reasoning effort on tasks completed fully at every compared effort on OSWorld2. Every reported score is 100%. Time is in seconds per task; token rate divides total generated tokens by total agent time. Each model uses its own common task set.

Higher effort can generate longer outputs and more actions on the same successful tasks. Table 23 restricts each OSWorld2 comparison to tasks that receive full credit at every evaluated effort for that model. On the five tasks that Gemini 3.8 Flash completes at all three efforts, high effort takes 3,024 s per task compared with 1,291 s at low effort. It generates more than twice as many tokens (73,578 versus 29,596) and executes more actions and batches. In comparison, its effective token rate falls from 47.57 to 42.18 tokens/s.

We observe a similar increase at Astra’s highest effort. On the 13 tasks it completes at all four settings, xhigh takes 1,132 s, compared with 792 s at low and 753 s at medium. It generates 14,359 tokens and executes 223 actions, compared with 7,878 tokens and 147 actions at low effort. Muse Spark 1.3 completes two tasks at all five efforts, while Kimi K3 has no task completed at all three efforts.

![](images/993f85648b618766f02a694f57fa1bd9bd920993657f9262eebdbac5d8b79b7a.jpg)  
Figure 18: Validation gains do not generalize to the held-out model. Across 34 candidate methods, the coding agent receives only validation errors. We measure test error separately on the held-out model. Each method selects 32 tasks. Error averages the mean absolute errors in partial score and exact completion, in percentage points.

## L AUTOMATED SEARCH FOR TASK-SELECTION METHODS

We investigate whether a coding agent can find task-selection methods that better estimate a new model’s score on the full task set. The agent develops these methods using results from six models; we evaluate them on a seventh model whose results are withheld throughout the search.

## L.1 SEARCH AND EVALUATION

We provide the coding agent with results from six models on 107 OSWorld2 tasks. It can modify both how the tasks are selected and how scores on the selected tasks are used to predict the full-set score. Each candidate method selects exactly 32 tasks. We evaluate it using six-fold leave-one-agent-out validation: five models are used to fit the method, and the sixth to measure prediction error. The agent receives only this validation error and can revise its method. We separately test each candidate on the seventh model after fitting to all six available models.

## L.2 VALIDATION GAINS DO NOT GENERALIZE TO THE HELD-OUT MODEL

Figure 18 compares validation and test error across 34 candidate methods. Validation error falls from 2.43 to 0.01 percentage points, while error on the held-out model changes from 3.90 to 4.00 percentage points. The search nearly eliminates validation error without improving the estimate for the new model. Validation errors guide repeated method selection, allowing the search to overfit these six models.